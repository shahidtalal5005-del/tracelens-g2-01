"""Experiment runners for Phase 1 (baselines) and Phase 2 (gap-wise evaluation)."""
from __future__ import annotations

import json
import os
from collections import defaultdict
from functools import partial
from typing import Any, Dict, List, Optional

from ..agent import FAULT_KINDS
from ..extract.baselines import BASELINES
from ..extract.evidence import EvidenceVerifier, RULES
from ..extract.judge import JUDGES
from ..extract.summarize import (CriticalTraceSummarizer, SUMMARISERS, Summary, Item, llm_summary, _quote)
from ..llm import Completer
from ..schema import Trace, VIEWS, VIEW_FULL, trace_tokens
from ..world import World
from .bench import eval_set
from .metrics import evaluate_attribution, evaluate_summaries, md_table

SYNTH_NOTE = (
    "> **Read this first.** These numbers come from the *synthetic* benchmark (TraceBench-Synth): a rule-based stand-in "
    "agent, deterministic tools, and 9 injected fault types. Several verifier rules were written for exactly these fault "
    "types, so near-perfect scores on them test the machinery, not the research hypothesis. The honest signals are the "
    "`source_error` row (a fault the rules cannot see) and the real-data runs (`python -m tracelens whowhen ...`, and "
    "`run-agent` with a real LLM). Do not present synthetic numbers as evidence about real agents.\n")


def _write(outdir: str, name: str, obj: Any) -> None:
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, name), "w", encoding="utf-8") as fh:
        if name.endswith(".json"):
            json.dump(obj, fh, indent=2, ensure_ascii=False)
        else:
            fh.write(obj)


def bench_stats(traces: List[Trace]) -> Dict[str, Any]:
    clean = [t for t in traces if not t.labels.get("fault")]
    faulted = [t for t in traces if t.labels.get("fault")]
    per: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
    for t in faulted:
        k = t.labels["fault"]["kind"]
        per[k][1] += 1
        per[k][0] += t.labels["fault"]["effective"]
    ev = eval_set(traces)
    return {
        "traces": len(traces), "clean": len(clean), "faulted": len(faulted),
        "clean_success_rate": round(sum(t.outcome["success"] for t in clean) / max(1, len(clean)), 4),
        "faulted_success_rate": round(sum(t.outcome["success"] for t in faulted) / max(1, len(faulted)), 4),
        "attribution_eval_set": len(ev),
        "mean_events": round(sum(len(t.events) for t in traces) / len(traces), 2),
        "mean_raw_tokens": round(sum(trace_tokens(t) for t in traces) / len(traces), 1),
        "fault_effective": {k: {"effective": v[0], "injected": v[1]} for k, v in sorted(per.items())},
    }


def _judge_methods(judge: Optional[Completer]) -> Dict[str, Any]:
    return {f"llm_{n}": partial(f, llm=judge) for n, f in JUDGES.items()} if judge else {}


# ---------------------------------------------------------------------------------------
def run_phase1(traces: List[Trace], outdir: str, judge: Optional[Completer] = None, max_judge: int = 60) -> Dict[str, Any]:
    ev = eval_set(traces)
    rows = []
    for name, fn in BASELINES.items():
        rows.append(evaluate_attribution(ev, fn))
    for name, fn in _judge_methods(judge).items():
        rows.append(evaluate_attribution(ev[:max_judge], fn))
    res = {"benchmark": bench_stats(traces), "baselines_full_view": rows,
           "note": "LLM-judge rows (if present) use the first %d evaluation traces." % max_judge}
    _write(outdir, "phase1_results.json", res)
    md = ["# Phase 1 - baseline results\n", SYNTH_NOTE, "## Benchmark\n", "```json", json.dumps(res["benchmark"], indent=2), "```\n",
          "## Step-attribution baselines (full view)\n",
          md_table(rows, ["method", "n", "step_acc", "hit@3", "mrr", "actor_acc"],
                   ["method", "n", "step acc", "hit@3", "MRR", "actor acc"]),
          "\nstep acc = the top-ranked event is exactly the injected-fault event.\n"]
    if not judge:
        md.append("_LLM-judge baselines (all-at-once / step-by-step / binary search) were NOT run - no model configured. "
                  "Re-run with `--judge ollama:<model>`._\n")
    _write(outdir, "phase1_results.md", "\n".join(md))
    return res


# ---------------------------------------------------------------------------------------
def run_phase2(traces: List[Trace], outdir: str, world: Optional[World] = None, judge: Optional[Completer] = None,
               max_judge: int = 60, budgets=(4, 6, 8)) -> Dict[str, Any]:
    world = world or World(7)
    ev = eval_set(traces)
    V = EvidenceVerifier(world)
    out: Dict[str, Any] = {"benchmark": bench_stats(traces)}

    # ---- Gap 1: attribution vs baselines ----
    g1 = [evaluate_attribution(ev, fn) for fn in BASELINES.values()]
    g1.append(evaluate_attribution(ev, V.attribute))
    for name, fn in _judge_methods(judge).items():
        g1.append(evaluate_attribution(ev[:max_judge], fn))
    out["gap1_attribution"] = g1
    abl = [evaluate_attribution(ev, EvidenceVerifier(world, [r for r in RULES if r != drop]).attribute) for drop in RULES]
    for a, drop in zip(abl, RULES):
        a["ablation"] = f"without {drop}"
    solo = [evaluate_attribution(ev, EvidenceVerifier(world, [r]).attribute) for r in RULES]
    for a, r in zip(solo, RULES):
        a["ablation"] = f"only {r}"
    sev = evaluate_attribution(ev, EvidenceVerifier(world, order="severity").attribute)
    sev["ablation"] = "severity order (not earliest)"
    out["gap1_ablation"] = abl + solo + [sev]

    # ---- Gap 2: observability ----
    g2 = []
    for view in VIEWS:
        for fn in (V.attribute, BASELINES["first_error_signal"]):
            g2.append(evaluate_attribution(ev, fn, view))
    out["gap2_observability"] = g2

    # ---- Gap 3: summaries ----
    S = CriticalTraceSummarizer(V)
    S_nochain = CriticalTraceSummarizer(V, use_chain=False)
    g3 = []
    for b in budgets:
        for name, fn in SUMMARISERS.items():
            r = evaluate_summaries(ev, partial(fn, budget=b))
            r["budget"] = b
            g3.append(r)
        for s in (S, S_nochain):
            r = evaluate_summaries(ev, partial(s.summarize, budget=b))
            r["budget"] = b
            g3.append(r)
        if judge:
            r = evaluate_summaries(ev[:max_judge], partial(llm_summary, budget=b, llm=judge))
            r["budget"] = b
            g3.append(r)
    g3.append({"method": "raw_log", "budget": "all", "n": len(ev), "fault_recall": 1.0, "labelled_recall": "n/a",
               "evidence_recall": 1.0, "compression": 1.0,
               "mean_tokens": round(sum(trace_tokens(t) for t in ev) / max(1, len(ev)), 1), "faithfulness": 1.0})
    out["gap3_summaries"] = g3
    g3v = []
    for view in VIEWS:
        r = evaluate_summaries(ev, partial(S.summarize, budget=6), view)
        r["budget"] = 6
        g3v.append(r)
    out["gap3_views"] = g3v

    out["failure_analysis"] = {k: v for k, v in g1[len(BASELINES)]["by_kind"].items()}
    _write(outdir, "phase2_results.json", out)
    _write(outdir, "phase2_results.md", _phase2_md(out))
    return out


def _phase2_md(o: Dict[str, Any]) -> str:
    m = ["# Phase 2 - gap-wise results\n", SYNTH_NOTE]
    m += ["## Benchmark\n", "```json", json.dumps({k: v for k, v in o["benchmark"].items() if k != "fault_effective"}, indent=2), "```\n"]
    m += ["## Gap 1 - step attribution (full view)\n",
          md_table(o["gap1_attribution"], ["method", "n", "step_acc", "step_acc_ci95", "hit@3", "mrr", "actor_acc"],
                   ["method", "n", "step acc", "95% CI", "hit@3", "MRR", "actor acc"]), ""]
    m += ["### Verifier accuracy by injected fault type\n", "| fault | n | step acc |", "|---|---|---|"]
    for k, v in o["failure_analysis"].items():
        m.append(f"| {k} | {v['n']} | {v['acc']} |")
    m += ["", "### Rule ablation\n", md_table(o["gap1_ablation"], ["ablation", "n", "step_acc", "hit@3", "mrr"],
                                             ["variant", "n", "step acc", "hit@3", "MRR"]), ""]
    m += ["## Gap 2 - observability (which trace fields are logged)\n",
          md_table(o["gap2_observability"], ["method", "view", "n", "step_acc", "hit@3", "mrr"],
                   ["method", "view", "n", "step acc", "hit@3", "MRR"]), "",
          "`full` = inputs + dependency links; `no_links` = inputs only; `no_inputs` = links only; `output_only` = what Who&When-style logs contain.\n"]
    m += ["## Gap 3 - budgeted summaries\n",
          md_table(o["gap3_summaries"], ["method", "budget", "fault_recall", "labelled_recall", "evidence_recall", "compression", "mean_tokens", "faithfulness"],
                   ["method", "budget (events)", "fault recall", "labelled recall", "gold-evidence recall", "tokens vs raw", "mean tokens", "faithfulness"]), "",
          "`labelled recall` only applies to methods that attach a role label to each item (baselines have none, shown as 0). "
          "`faithfulness` is 1.0 for every method here because all quotes are copied from events; it only becomes informative for "
          "LLM-written summaries. `fault recall` = the injected-fault event is among the chosen events; `gold-evidence recall` = "
          "the tool results of the gold documents are chosen.\n",
          "### Critical-trace summary (budget 6) by observability view\n",
          md_table(o["gap3_views"], ["method", "view", "fault_recall", "labelled_recall", "evidence_recall", "compression", "faithfulness"]), ""]
    return "\n".join(m)
