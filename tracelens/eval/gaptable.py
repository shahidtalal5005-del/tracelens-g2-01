"""Assemble the Phase 2 gap-resolution table from result files (numbers only; the verdict column is yours)."""
from __future__ import annotations

import json
import os


def _load(path):
    return json.load(open(path, encoding="utf-8")) if os.path.exists(path) else None


def write_gap_table(outdir: str) -> str:
    p2 = _load(os.path.join(outdir, "phase2_results.json"))
    ww = _load(os.path.join(outdir, "whowhen_results.json"))
    if p2 is None:
        raise FileNotFoundError("run `phase2` first")
    g1 = {r["method"]: r for r in p2["gap1_attribution"]}
    ours = g1["evidence_verifier"]
    baseline_names = [m for m in g1 if m != "evidence_verifier"]
    best = max(baseline_names, key=lambda m: g1[m]["step_acc"])
    g2 = {(r["method"], r["view"]): r for r in p2["gap2_observability"]}
    g3 = [r for r in p2["gap3_summaries"] if r["budget"] == 6]
    ct = next(r for r in g3 if r["method"] == "critical_trace")
    bestb = max((r for r in g3 if r["method"] not in ("critical_trace", "critical_trace[no_chain]", "raw_log")), key=lambda r: r["fault_recall"])
    ww_line = "not run yet (`python -m tracelens whowhen --path <data>`)"
    if ww:
        rows = {r["method"]: r for r in ww["rows"]}
        ww_line = "; ".join(f"{m}: {r['step_acc']:.3f}" for m, r in rows.items())
    L = [
        "# Gap-resolution table (auto-generated numbers)\n",
        "Numbers below come from the **synthetic** benchmark unless stated. The `Addressed?` column is deliberately left for you: "
        "decide Fully / Partially / Not only after the real-data runs (Who&When, real-LLM traces) are in.\n",
        "| Gap | Method / component | Base-paper-style result | Our result | Addressed? (Fully / Partially / Not) |",
        "|---|---|---|---|---|",
        f"| 1 Step-level localisation | evidence-linked verifier (7 rules, earliest-violation rule) | paper reports 14.2% step-level (best of 3 LLM-judge methods, Who&When, different data). "
        f"Synthetic: best baseline `{best}` = {g1[best]['step_acc']:.3f}. Who&When baselines here: {ww_line} | "
        f"synthetic step acc {ours['step_acc']:.3f} (95% CI {ours['step_acc_ci95']}); `source_error` (undetectable by design) = {ours['by_kind']['source_error']['acc']:.2f} | TO DECIDE |",
        f"| 2 Observability | tool inputs + dependency links in the schema; field ablation | partial (output-only) logs | verifier step acc: full {g2[('evidence_verifier', 'full')]['step_acc']:.3f} → "
        f"no_inputs {g2[('evidence_verifier', 'no_inputs')]['step_acc']:.3f} → output_only {g2[('evidence_verifier', 'output_only')]['step_acc']:.3f} | TO DECIDE |",
        f"| 3 Summary faithfulness & utility | budgeted critical-trace summary with event-linked quotes (budget 6 events) | best positional baseline `{bestb['method']}`: fault recall {bestb['fault_recall']:.3f} | "
        f"fault recall {ct['fault_recall']:.3f}, tokens {ct['compression']:.2f}× raw, faithfulness {ct['faithfulness']:.2f} (by construction) | TO DECIDE |",
        "",
        "Caveat for every row: the synthetic benchmark was built together with the verifier rules, so it can only show that the machinery works.",
    ]
    path = os.path.join(outdir, "gap_resolution_table.md")
    open(path, "w", encoding="utf-8").write("\n".join(L) + "\n")
    return path
