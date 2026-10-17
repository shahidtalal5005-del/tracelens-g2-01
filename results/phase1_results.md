# Phase 1 - baseline results

> **Read this first.** These numbers come from the *synthetic* benchmark (TraceBench-Synth): a rule-based stand-in agent, deterministic tools, and 9 injected fault types. Several verifier rules were written for exactly these fault types, so near-perfect scores on them test the machinery, not the research hypothesis. The honest signals are the `source_error` row (a fault the rules cannot see) and the real-data runs (`python -m tracelens whowhen ...`, and `run-agent` with a real LLM). Do not present synthetic numbers as evidence about real agents.

## Benchmark

```json
{
  "traces": 588,
  "clean": 48,
  "faulted": 540,
  "clean_success_rate": 1.0,
  "faulted_success_rate": 0.2981,
  "attribution_eval_set": 379,
  "mean_events": 16.8,
  "mean_raw_tokens": 709.8,
  "fault_effective": {
    "calc_error": {
      "effective": 24,
      "injected": 24
    },
    "memory_swap": {
      "effective": 18,
      "injected": 24
    },
    "plan_omission": {
      "effective": 18,
      "injected": 24
    },
    "retrieval_miss": {
      "effective": 65,
      "injected": 84
    },
    "retrieval_miss_transient": {
      "effective": 0,
      "injected": 84
    },
    "source_error": {
      "effective": 61,
      "injected": 84
    },
    "tool_corruption": {
      "effective": 61,
      "injected": 84
    },
    "tool_error": {
      "effective": 84,
      "injected": 84
    },
    "unsupported_claim": {
      "effective": 48,
      "injected": 48
    }
  }
}
```

## Step-attribution baselines (full view)

| method | n | step acc | hit@3 | MRR | actor acc |
|---|---|---|---|---|---|
| random | 379 | 0.0633 | 0.1873 | 0.2143 | 0.2507 |
| first_step | 379 | 0.0475 | 0.1003 | 0.1746 | 0.0475 |
| last_step | 379 | 0.1266 | 0.3166 | 0.2803 | 0.1266 |
| last_tool_output | 379 | 0.19 | 0.5805 | 0.4219 | 0.7784 |
| first_error_signal | 379 | 0.3483 | 0.5383 | 0.4746 | 0.3483 |

step acc = the top-ranked event is exactly the injected-fault event.

_LLM-judge baselines (all-at-once / step-by-step / binary search) were NOT run - no model configured. Re-run with `--judge ollama:<model>`._
