# Gap-resolution table (auto-generated numbers)

Numbers below come from the **synthetic** benchmark unless stated. The `Addressed?` column is deliberately left for you: decide Fully / Partially / Not only after the real-data runs (Who&When, real-LLM traces) are in.

| Gap | Method / component | Base-paper-style result | Our result | Addressed? (Fully / Partially / Not) |
|---|---|---|---|---|
| 1 Step-level localisation | evidence-linked verifier (7 rules, earliest-violation rule) | paper reports 14.2% step-level (best of 3 LLM-judge methods, Who&When, different data). Synthetic: best baseline `first_error_signal` = 0.348. Who&When baselines here: not run yet (`python -m tracelens whowhen --path <data>`) | synthetic step acc 0.839 (95% CI [0.7987, 0.8726]); `source_error` (undetectable by design) = 0.00 | TO DECIDE |
| 2 Observability | tool inputs + dependency links in the schema; field ablation | partial (output-only) logs | verifier step acc: full 0.839 → no_inputs 0.678 → output_only 0.615 | TO DECIDE |
| 3 Summary faithfulness & utility | budgeted critical-trace summary with event-linked quotes (budget 6 events) | best positional baseline `last_tool_results_k`: fault recall 0.778 | fault recall 0.934, tokens 0.24× raw, faithfulness 1.00 (by construction) | TO DECIDE |

Caveat for every row: the synthetic benchmark was built together with the verifier rules, so it can only show that the machinery works.
