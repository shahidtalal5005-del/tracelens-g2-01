# Phase 2 - gap-wise results

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
  "mean_raw_tokens": 709.8
}
```

## Gap 1 - step attribution (full view)

| method | n | step acc | 95% CI | hit@3 | MRR | actor acc |
|---|---|---|---|---|---|---|
| random | 379 | 0.0633 | [0.0429, 0.0925] | 0.1873 | 0.2143 | 0.2507 |
| first_step | 379 | 0.0475 | [0.0302, 0.0738] | 0.1003 | 0.1746 | 0.0475 |
| last_step | 379 | 0.1266 | [0.0969, 0.1639] | 0.3166 | 0.2803 | 0.1266 |
| last_tool_output | 379 | 0.19 | [0.1537, 0.2325] | 0.5805 | 0.4219 | 0.7784 |
| first_error_signal | 379 | 0.3483 | [0.3021, 0.3976] | 0.5383 | 0.4746 | 0.3483 |
| evidence_verifier | 379 | 0.8391 | [0.7987, 0.8726] | 0.934 | 0.896 | 0.8391 |

### Verifier accuracy by injected fault type

| fault | n | step acc |
|---|---|---|
| calc_error | 24 | 1.0 |
| memory_swap | 18 | 1.0 |
| plan_omission | 18 | 1.0 |
| retrieval_miss | 65 | 1.0 |
| source_error | 61 | 0.0 |
| tool_corruption | 61 | 1.0 |
| tool_error | 84 | 1.0 |
| unsupported_claim | 48 | 1.0 |

### Rule ablation

| variant | n | step acc | hit@3 | MRR |
|---|---|---|---|---|
| without plan | 379 | 0.7916 | 0.8865 | 0.8558 |
| without retrieval | 379 | 0.8391 | 0.934 | 0.896 |
| without replay | 379 | 0.6781 | 0.8681 | 0.7921 |
| without tool_error | 379 | 0.6174 | 0.934 | 0.7483 |
| without memory | 379 | 0.7916 | 0.8865 | 0.8541 |
| without calc | 379 | 0.7757 | 0.934 | 0.8644 |
| without claim | 379 | 0.8391 | 0.934 | 0.896 |
| only plan | 379 | 0.1741 | 0.6491 | 0.4288 |
| only retrieval | 379 | 0.2982 | 0.7731 | 0.5305 |
| only replay | 379 | 0.4591 | 0.8391 | 0.6345 |
| only tool_error | 379 | 0.3483 | 0.6016 | 0.5364 |
| only memory | 379 | 0.1741 | 0.6491 | 0.4306 |
| only calc | 379 | 0.19 | 0.6016 | 0.4203 |
| only claim | 379 | 0.1266 | 0.6016 | 0.3886 |
| severity order (not earliest) | 379 | 0.8391 | 0.934 | 0.896 |

## Gap 2 - observability (which trace fields are logged)

| method | view | n | step acc | hit@3 | MRR |
|---|---|---|---|---|---|
| evidence_verifier | full | 379 | 0.8391 | 0.934 | 0.896 |
| first_error_signal | full | 379 | 0.3483 | 0.5383 | 0.4746 |
| evidence_verifier | no_links | 379 | 0.8391 | 0.934 | 0.896 |
| first_error_signal | no_links | 379 | 0.3483 | 0.5383 | 0.4746 |
| evidence_verifier | no_inputs | 379 | 0.6781 | 0.8681 | 0.7921 |
| first_error_signal | no_inputs | 379 | 0.3483 | 0.5383 | 0.4746 |
| evidence_verifier | output_only | 379 | 0.6148 | 0.8681 | 0.7604 |
| first_error_signal | output_only | 379 | 0.3483 | 0.5383 | 0.4746 |

`full` = inputs + dependency links; `no_links` = inputs only; `no_inputs` = links only; `output_only` = what Who&When-style logs contain.

## Gap 3 - budgeted summaries

| method | budget (events) | fault recall | labelled recall | gold-evidence recall | tokens vs raw | mean tokens | faithfulness |
|---|---|---|---|---|---|---|---|
| head_k | 4 | 0.1003 | 0.0 | 0.0 | 0.129 | 83.0 | 1.0 |
| tail_k | 4 | 0.3166 | 0.0 | 0.1488 | 0.0818 | 48.8 | 1.0 |
| random_k | 4 | 0.2902 | 0.0 | 0.2893 | 0.1123 | 71.1 | 1.0 |
| last_tool_results_k | 4 | 0.6675 | 0.0 | 0.9194 | 0.1417 | 92.4 | 1.0 |
| critical_trace | 4 | 0.9024 | 0.9024 | 0.2252 | 0.1566 | 98.2 | 1.0 |
| critical_trace[no_chain] | 4 | 0.8391 | 0.8391 | 0.126 | 0.1461 | 94.7 | 1.0 |
| head_k | 6 | 0.3113 | 0.0 | 0.2831 | 0.1914 | 122.2 | 1.0 |
| tail_k | 6 | 0.4591 | 0.0 | 0.2355 | 0.1408 | 85.8 | 1.0 |
| random_k | 6 | 0.4116 | 0.0 | 0.4174 | 0.1693 | 107.3 | 1.0 |
| last_tool_results_k | 6 | 0.7784 | 0.0 | 1.0 | 0.174 | 118.0 | 1.0 |
| critical_trace | 6 | 0.934 | 0.934 | 0.5269 | 0.2407 | 152.3 | 1.0 |
| critical_trace[no_chain] | 6 | 0.8391 | 0.8391 | 0.126 | 0.1679 | 109.2 | 1.0 |
| head_k | 8 | 0.4908 | 0.0 | 0.6054 | 0.2468 | 158.5 | 1.0 |
| tail_k | 8 | 0.6069 | 0.0 | 0.5062 | 0.1983 | 122.8 | 1.0 |
| random_k | 8 | 0.5303 | 0.0 | 0.5331 | 0.224 | 141.8 | 1.0 |
| last_tool_results_k | 8 | 0.7784 | 0.0 | 1.0 | 0.1773 | 120.9 | 1.0 |
| critical_trace | 8 | 1.0 | 1.0 | 0.8326 | 0.3118 | 201.6 | 1.0 |
| critical_trace[no_chain] | 8 | 0.8391 | 0.8391 | 0.126 | 0.1679 | 109.2 | 1.0 |
| raw_log | all | 1.0 | n/a | 1.0 | 1.0 | 677.8 | 1.0 |

`labelled recall` only applies to methods that attach a role label to each item (baselines have none, shown as 0). `faithfulness` is 1.0 for every method here because all quotes are copied from events; it only becomes informative for LLM-written summaries. `fault recall` = the injected-fault event is among the chosen events; `gold-evidence recall` = the tool results of the gold documents are chosen.

### Critical-trace summary (budget 6) by observability view

| method | view | fault_recall | labelled_recall | evidence_recall | compression | faithfulness |
|---|---|---|---|---|---|---|
| critical_trace | full | 0.934 | 0.934 | 0.5269 | 0.2407 | 1.0 |
| critical_trace | no_links | 0.8391 | 0.8391 | 0.126 | 0.1679 | 1.0 |
| critical_trace | no_inputs | 0.8681 | 0.8681 | 0.5062 | 0.2844 | 1.0 |
| critical_trace | output_only | 0.6148 | 0.6148 | 0.0 | 0.171 | 1.0 |
