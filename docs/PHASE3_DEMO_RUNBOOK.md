# Phase 3 — demo runbook (commands only; the slides and talk are yours to write)

## Live demo order (about 5 minutes)

1. `python -m tracelens all --out results` — show the four progress lines (benchmark, phase 1, phase 2, viewer).
2. Open `results/viewer.html`: tab **clean** (no fault), then **tool_corruption**, then **source_error** (verifier misses it — say so).
3. Show `results/phase2_results.md`: Gap 1 table, per-fault table, rule ablation, observability table, summary table.
4. Optional real-LLM moment: `python -m tracelens run-agent --policy ollama:qwen2.5:7b --n 4 --fault calc_error`, then open the produced JSONL and point at the faulted event.
5. Show `python -m pytest` → `45 passed`.

## Components every member must be able to explain (viva)

| Topic | File | Question to be ready for |
|---|---|---|
| Trace schema and why three field groups | `tracelens/schema.py` | why do `inputs` and `parents` exist separately? what does each view remove? |
| Fault injection and ground truth | `tracelens/agent.py` | how is the critical event id decided? why can `source_error` not be found by replay? |
| Evidence verifier rules | `tracelens/extract/evidence.py` | what does each rule check? why "earliest violation"? what are false-negative cases? |
| Summariser and faithfulness | `tracelens/extract/summarize.py` | how is faithfulness measured? why is it 1.0 for the templated summaries? |
| LLM-judge baselines | `tracelens/extract/judge.py` | how do all-at-once, step-by-step and binary search differ in cost? |
| Metrics | `tracelens/eval/metrics.py` | step accuracy vs hit@3 vs MRR; what is a Wilson interval? |
| Synthetic-benchmark circularity | `README.md` limits | why do the synthetic numbers not prove the research claims? |
| Who&When adapter | `tracelens/eval/whowhen.py` | how is the 0/1-based step index detected? |
