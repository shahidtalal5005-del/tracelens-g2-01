# Who&When results (output-only view)

traces: 365; step acc = exact match of the predicted step index with `mistake_step`.

| method | n | step acc | 95% CI | hit@3 | MRR | actor acc |
|---|---|---|---|---|---|---|
| random | 365 | 0.0959 | [0.0698, 0.1304] | 0.2904 | 0.2732 | 0.3288 |
| first_step | 365 | 0.1096 | [0.0815, 0.1458] | 0.3616 | 0.3143 | 0.3288 |
| last_step | 365 | 0.0411 | [0.0251, 0.0667] | 0.1781 | 0.214 | 0.3479 |
| last_tool_output | 365 | 0.0411 | [0.0251, 0.0667] | 0.1781 | 0.214 | 0.3479 |
| first_error_signal | 365 | 0.1123 | [0.0839, 0.1488] | 0.2493 | 0.2723 | 0.2986 |