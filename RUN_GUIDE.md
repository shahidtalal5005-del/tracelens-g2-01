# RUN GUIDE — setting TraceLens up on another laptop

Everything below works **offline**. You need only Python 3.10 or newer (tested on 3.11, 3.12 and 3.13). No GPU, no API key, no pip packages for the core project.

## 1. Check Python

| System | Command | Expected |
|---|---|---|
| Windows (PowerShell or cmd) | `python --version` | `Python 3.10…3.13` |
| macOS / Linux | `python3 --version` | same |

If missing: install from https://www.python.org/downloads/ (Windows: tick **"Add python.exe to PATH"** in the installer). On Windows use `python`; on macOS/Linux use `python3` wherever this guide says `python`.

## 2. Unzip and open a terminal in the folder

Unzip `tracelens_v1.zip`; you get a folder `tracelens/`. Open a terminal **inside that folder** (the one that contains `README.md` and the `tracelens` sub-folder).
Windows: open the folder in Explorer, click the address bar, type `cmd`, press Enter.

## 3. Run everything with one command

* Windows: `run_all.bat`
* macOS / Linux: `bash run_all.sh`

or by hand (all systems):

```
python -m tracelens all --out results
```

It takes about 15 seconds and prints four progress lines. Then open **`results/viewer.html`** by double-clicking it.
Reports are in `results/`: `phase1_results.md`, `phase2_results.md`, `gap_resolution_table.md` (+ `.json` versions).

## 4. Run the tests (optional but recommended before the demo)

```
python -m pip install pytest
python -m pytest
```
Expected: `45 passed`.

If `pip` is blocked on that laptop, skip this step; the project itself does not need pytest.

## 5. Optional: use a real LLM (for honest, non-synthetic traces)

1. Install Ollama from https://ollama.com and run `ollama pull qwen2.5:7b` (about 4.7 GB; a 3B model also works: `qwen2.5:3b`).
2. Make sure Ollama is running (`ollama serve`, or the desktop app).
3. Generate traces with the real model acting as the agent, and use it as the LLM judge:

```
python -m tracelens run-agent --policy ollama:qwen2.5:7b --n 8 --fault calc_error
python -m tracelens all --out results_llm --policy ollama:qwen2.5:7b --judge ollama:qwen2.5:7b --tasks-per-type 3
```
The second command makes several hundred model calls; on a laptop CPU it can take a long time — lower `--tasks-per-type` or use the 3B model. If the model server is unreachable the program stops with a clear message.

For an OpenAI-compatible API instead: set `OPENAI_API_KEY` (and optionally `OPENAI_BASE_URL`) and use `openai:<model>`.

## 6. Optional: the base-paper dataset (Who&When)

This needs internet **once**, on any machine:

```
python -m pip install huggingface_hub pandas pyarrow
python -c "from huggingface_hub import snapshot_download as s; print(s('Kevin355/Who_and_When', repo_type='dataset', local_dir='data/who_and_when'))"
python -m tracelens whowhen --path data/who_and_when --inspect
python -m tracelens whowhen --path data/who_and_when
```
**Read the `--inspect` output** (rows, columns, detected 0/1-based index). The adapter was written from the dataset card and could not be tried on the real files while building this project, so if column names differ you may need a two-line change in `tracelens/eval/whowhen.py` (`_HIST`, `_AGENT`, `_STEP`, `_Q` tuples at the top).
Then re-run `python -m tracelens phase2 --out results` to refresh `results/gap_resolution_table.md`.

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| `python` not recognised (Windows) | reinstall Python with "Add to PATH", or use `py -3` instead of `python` |
| `No module named tracelens` | you are not in the folder that contains the `tracelens` directory — `cd` into it |
| `cannot reach model server ... Ollama` | start Ollama; check `http://localhost:11434` opens in a browser |
| `reading .parquet needs: pip install pandas pyarrow` | run the pip command from step 6 |
| Output looks different from the README | the benchmark is seeded; change only `--tasks-per-type` and numbers will move slightly |
| Antivirus blocks the `.bat` | run the `python -m tracelens all --out results` line yourself |

## 8. Put it on GitHub (you do this)

```
git init
git add .
git commit -m "Phase 1: schema, instrumented agent, baselines, evaluation"
git branch -M main
git remote add origin https://github.com/<you>/<repo>.git
git push -u origin main
```
Make the repository public or add the supervisor (Meemon Khan) as a collaborator, and paste the link into the Classroom submission.
