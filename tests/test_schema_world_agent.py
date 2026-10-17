import json

import pytest

from tracelens.agent import Agent, FAULT_KINDS, Fault, applicable
from tracelens.schema import (Event, Trace, VIEWS, VIEW_FULL, VIEW_NO_INPUTS, VIEW_NO_LINKS, VIEW_OUTPUT_ONLY,
                              read_jsonl, write_jsonl)
from tracelens.world import World


def test_world_is_deterministic():
    a, b = World(3), World(3)
    assert list(a.companies) == list(b.companies)
    assert a.search("revenue", 3) == b.search("revenue", 3)
    assert World(3).companies != World(4).companies


def test_search_finds_exact_entity(world):
    name = sorted(world.companies)[0]
    top = world.search(f"{name} annual revenue", 5)
    assert any(r["title"] == name for r in top)


def test_read_doc_unknown_raises(world):
    with pytest.raises(KeyError):
        world.read_doc("D9999")


def test_calc_ops(world):
    assert world.call("calc", {"op": "sum", "values": ["3", 4]}) == 7
    assert world.call("calc", {"op": "argmax", "values": [1, 9, 3]}) == 1
    with pytest.raises(ValueError):
        world.call("calc", {"op": "nope", "values": [1]})


def test_clean_runs_succeed_and_validate(agent, tasks):
    for t in tasks:
        tr = agent.run(t)
        assert tr.validate() == []
        assert tr.outcome["success"], (t["task_id"], tr.outcome)
        assert tr.labels["fault"] is None
        assert tr.labels["gold_evidence_events"], "gold evidence events must be labelled"


@pytest.mark.parametrize("kind", FAULT_KINDS)
def test_fault_has_location_and_expected_event_kind(agent, tasks, kind):
    expected = {"retrieval_miss": "tool_result", "retrieval_miss_transient": "tool_result", "tool_corruption": "tool_result",
                "source_error": "tool_result", "tool_error": "error", "memory_swap": "memory_read", "calc_error": "tool_result",
                "plan_omission": "plan", "unsupported_claim": "final_answer"}[kind]
    seen = 0
    for t in tasks:
        if not applicable(kind, t):
            continue
        tr = agent.run(t, Fault(kind, 0))
        assert tr.validate() == []
        crit = tr.labels["fault"]["critical_event"]
        assert crit is not None and tr.events[crit].kind == expected
        seen += 1
    assert seen


def test_transient_retrieval_fault_is_recovered(agent, tasks):
    tr = agent.run(tasks[0], Fault("retrieval_miss_transient", 0))
    assert tr.outcome["success"] and any(e.kind == "retry" for e in tr.events)


def test_persistent_faults_break_the_answer(agent, tasks):
    for kind in ("retrieval_miss", "tool_corruption", "tool_error", "unsupported_claim", "source_error"):
        t = tasks[0]
        assert not agent.run(t, Fault(kind, 0)).outcome["success"], kind


def test_labels_hidden_in_views(agent, tasks):
    tr = agent.run(tasks[0], Fault("tool_corruption", 0))
    for v in VIEWS:
        assert tr.view(v).labels == {}
    assert tr.labels["fault"]  # original untouched


def test_view_removes_exactly_the_requested_fields(agent, tasks):
    tr = agent.run(tasks[2])
    full, nl, ni, oo = (tr.view(v) for v in (VIEW_FULL, VIEW_NO_LINKS, VIEW_NO_INPUTS, VIEW_OUTPUT_ONLY))
    assert any(e.inputs for e in full.events) and any(e.parents for e in full.events)
    assert any(e.inputs for e in nl.events) and not any(e.parents for e in nl.events)
    assert not any(e.inputs for e in ni.events) and any(e.parents for e in ni.events)
    assert not any(e.inputs or e.parents for e in oo.events)
    assert [e.content for e in oo.events] == [e.content for e in tr.events]
    with pytest.raises(ValueError):
        tr.view("bogus")


def test_roundtrip_jsonl(tmp_path, agent, tasks):
    traces = [agent.run(tasks[0]), agent.run(tasks[1], Fault("calc_error", 0)) if applicable("calc_error", tasks[1]) else agent.run(tasks[1])]
    p = tmp_path / "t.jsonl"
    assert write_jsonl(traces, str(p)) == 2
    back = read_jsonl(str(p))
    assert [t.dumps() for t in back] == [t.dumps() for t in traces]


def test_validate_catches_problems():
    t = Trace("x", "q", [Event(0, "plan", "a"), Event(1, "bogus", "a", parents=[5], status="weird")])
    msgs = " ".join(t.validate())
    assert "unknown kind" in msgs and "not an earlier event" in msgs and "bad status" in msgs and "first event must be a goal" in msgs


def test_agent_is_reproducible(world, tasks):
    a = Agent(world).run(tasks[0], Fault("memory_swap", 0) if applicable("memory_swap", tasks[0]) else None, seed=2)
    b = Agent(world).run(tasks[0], Fault("memory_swap", 0) if applicable("memory_swap", tasks[0]) else None, seed=2)
    assert a.dumps() == b.dumps()
