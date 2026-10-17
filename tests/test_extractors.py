import pytest

from tracelens.agent import FAULT_KINDS
from tracelens.eval.bench import eval_set
from tracelens.eval.metrics import evaluate_attribution, evaluate_summaries, wilson
from tracelens.extract.baselines import BASELINES
from tracelens.extract.evidence import EvidenceVerifier, RULES
from tracelens.extract.judge import all_at_once, binary_search, step_by_step
from tracelens.extract.summarize import (CriticalTraceSummarizer, Item, Summary, faithfulness, head_k, llm_summary,
                                         random_k, tail_k)
from tracelens.llm import ChatPolicy, FakeCompleter, make_completer
from tracelens.schema import VIEW_OUTPUT_ONLY, VIEW_NO_INPUTS


def test_every_baseline_returns_a_permutation(bench):
    t = eval_set(bench)[0]
    for name, fn in BASELINES.items():
        ids = [e for e, _, _ in fn(t.view()).ranking]
        assert sorted(ids) == list(range(len(t.events))), name


def test_verifier_finds_detectable_faults_exactly(bench, world):
    V = EvidenceVerifier(world)
    res = evaluate_attribution(eval_set(bench), V.attribute)
    for kind, v in res["by_kind"].items():
        if kind != "source_error":
            assert v["acc"] == 1.0, (kind, v)


def test_verifier_cannot_see_replay_consistent_source_errors(bench, world):
    res = evaluate_attribution(eval_set(bench), EvidenceVerifier(world).attribute)
    assert res["by_kind"]["source_error"]["acc"] == 0.0     # documented limitation, must stay visible


def test_verifier_has_no_false_positives_on_clean_runs(bench, world):
    V = EvidenceVerifier(world)
    for t in bench:
        if not t.labels["fault"]:
            assert V.check(t.view()) == [], t.trace_id


def test_verifier_beats_every_baseline(bench, world):
    ev = eval_set(bench)
    ours = evaluate_attribution(ev, EvidenceVerifier(world).attribute)["step_acc"]
    best = max(evaluate_attribution(ev, f)["step_acc"] for f in BASELINES.values())
    assert ours > best + 0.3


def test_rule_ablation_each_rule_matters(bench, world):
    ev = eval_set(bench)
    full = evaluate_attribution(ev, EvidenceVerifier(world).attribute)["step_acc"]
    for r in ("plan", "replay", "tool_error", "memory", "calc"):
        less = evaluate_attribution(ev, EvidenceVerifier(world, [x for x in RULES if x != r]).attribute)["step_acc"]
        assert less < full, r


def test_observability_degrades_attribution(bench, world):
    ev = eval_set(bench)
    V = EvidenceVerifier(world)
    full = evaluate_attribution(ev, V.attribute)["step_acc"]
    oo = evaluate_attribution(ev, V.attribute, VIEW_OUTPUT_ONLY)["step_acc"]
    ni = evaluate_attribution(ev, V.attribute, VIEW_NO_INPUTS)["step_acc"]
    assert oo < ni <= full


def test_bad_order_argument():
    with pytest.raises(ValueError):
        EvidenceVerifier(order="random")


def test_summary_respects_budget_and_cites_real_events(bench, world):
    S = CriticalTraceSummarizer(EvidenceVerifier(world))
    for t in eval_set(bench)[:40]:
        v = t.view()
        s = S.summarize(v, budget=5)
        assert len(s.items) <= 5
        assert all(0 <= i < len(v.events) for i in s.ids())
        assert s.ids()[0] == 0 and s.ids()[-1] == len(v.events) - 1
        assert faithfulness(s, v) == 1.0


def test_faithfulness_detects_fabricated_quotes(bench):
    t = eval_set(bench)[0].view()
    bad = Summary("x", [Item(2, "context", "tool search -> 999999999 totally invented"), Item(len(t.events) + 5, "context", "ghost")])
    assert faithfulness(bad, t) == 0.0
    assert faithfulness(Summary("x", []), t) == 0.0


def test_summary_beats_positional_baselines_on_fault_recall(bench, world):
    ev = eval_set(bench)
    S = CriticalTraceSummarizer(EvidenceVerifier(world))
    ours = evaluate_summaries(ev, lambda t: S.summarize(t, 6))["fault_recall"]
    for fn in (head_k, tail_k, random_k):
        assert ours > evaluate_summaries(ev, lambda t, f=fn: f(t, 6))["fault_recall"] + 0.2


def test_metrics_helpers():
    lo, hi = wilson(50, 100)
    assert 0.39 < lo < 0.41 and 0.59 < hi < 0.61
    assert wilson(0, 0) == [0.0, 0.0]


# ---- LLM-dependent code paths, exercised with a fake model --------------------------------
def test_judges_parse_answers_and_track_cost(bench):
    t = eval_set(bench)[0].view()
    n = len(t.events)
    aao = all_at_once(t, FakeCompleter(lambda p: '{"step": 3, "reason": "x"}'))
    assert aao.top == 3 and aao.cost["calls"] == 1
    assert all_at_once(t, FakeCompleter(lambda p: "no idea")).ranking[0][2] == "fallback order"
    sbs = step_by_step(t, FakeCompleter(lambda p: "YES, wrong" if "Step under review:\nStep 4 " in p else "NO"))
    assert sbs.top == 4
    bs = binary_search(t, FakeCompleter(lambda p: "FIRST"))
    assert bs.top == 1
    bs2 = binary_search(t, FakeCompleter(lambda p: "SECOND"))
    assert bs2.top == n - 1 and bs2.cost["calls"] >= 2


def test_llm_summary_filters_invalid_ids(bench):
    t = eval_set(bench)[0].view()
    s = llm_summary(t, 3, FakeCompleter(lambda p: "[0, 2, 9999, 5, 7]"))
    assert s.ids() == [0, 2, 5]
    assert llm_summary(t, 3, FakeCompleter(lambda p: "garbage")).items == []


def test_chat_policy_parses_and_falls_back(world, tasks):
    q = tasks[0]["question"]
    good = ChatPolicy(FakeCompleter(lambda p: '[["lookup", "Foo Labs", "revenue"]]'))
    assert good.plan(q) == [["lookup", "Foo Labs", "revenue"]] and good.fallbacks == 0
    bad = ChatPolicy(FakeCompleter(lambda p: "I cannot do that"))
    assert bad.plan(q)[0][0] == "lookup" and bad.fallbacks == 1
    rd = ChatPolicy(FakeCompleter(lambda p: "It is 1,234 million"))
    assert rd.read("x", "E", "revenue") == "1234"
    assert ChatPolicy(FakeCompleter(lambda p: "NONE")).read("x", "E", "revenue") is None


def test_agent_with_chat_policy_end_to_end(world, tasks):
    from tracelens.agent import Agent
    from tracelens.llm import ScriptedPolicy
    scripted = ScriptedPolicy()

    def fake(prompt):
        if "You are a planner" in prompt:
            import json
            q = prompt.split("Question:")[1].split("\nJSON:")[0].strip()
            return json.dumps(scripted.plan(q))
        import re
        m = re.search(r"revenue of (\d+) million|population of (\d+)|headquartered in ([A-Za-z]+)", prompt)
        return next(g for g in m.groups() if g)

    tr = Agent(world, ChatPolicy(FakeCompleter(fake))).run(tasks[0])
    assert tr.validate() == [] and tr.outcome["success"]


def test_make_completer_specs():
    assert make_completer("none") is None
    assert make_completer("ollama:llama3").name == "ollama:llama3"
    assert make_completer("openai:gpt-4o-mini").name == "openai:gpt-4o-mini"
    with pytest.raises(ValueError):
        make_completer("bogus:x")
