# Phase 1 (Deliverable C) — where each required item lives

| Requirement (Phase 1 sheet, §6) | Where | How to show it |
|---|---|---|
| Finalised system architecture | `docs/architecture.png`, README | open the diagram |
| Working data / environment pipeline | `tracelens/world.py`, `eval/bench.py` | `python -m tracelens build-bench` |
| Implemented baseline with initial results | `extract/baselines.py`, `extract/judge.py`, `results/phase1_results.md` | `python -m tracelens phase1` |
| Initial model integration and provider setup | `tracelens/llm.py` (Ollama / OpenAI-compatible / scripted) | `python -m tracelens run-agent --policy ollama:<model>` |
| Initial tool interfaces | `World.call` (search, read_doc, calc) | `tests/test_schema_world_agent.py` |
| State / context structure and structured contracts | `schema.py`, README "Contracts" | `tests` (schema validation) |
| Initial evaluation dataset / test suite | `results/bench.jsonl` (588 traces), `tests/` | `python -m pytest` |
| End-to-end workflow | `python -m tracelens all` and `results/viewer.html` | live demo |
| README with run instructions, clean structure, commit history | `README.md`, `RUN_GUIDE.md` | — |
| Supervisor access | **you must do this on GitHub** (public repo or add Meemon Khan as collaborator) | — |

Honest status of the baseline: LLM-judge baselines need a model (`--judge ollama:...`); until you run them, Phase 1 results contain only the heuristic baselines. State that plainly.
