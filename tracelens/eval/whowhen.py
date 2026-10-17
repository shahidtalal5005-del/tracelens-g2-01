"""Adapter for the base-paper benchmark Who&When (Kevin355/Who_and_When on Hugging Face).

STATUS: written from the dataset card and the paper description; it could NOT be run against the
real files in the development sandbox (no access to huggingface.co).  The loader is therefore
defensive: it accepts several field spellings and auto-detects whether `mistake_step` is 0- or 1-based
by checking which base makes `history[step].name/role == mistake_agent` more often.  Run
`python -m tracelens whowhen --inspect <path>` first and read the printed summary.

Get the data on a machine with internet access:
    pip install huggingface_hub
    python -c "from huggingface_hub import snapshot_download as s; print(s('Kevin355/Who_and_When', repo_type='dataset'))"
Then pass the folder (or a single .parquet/.json/.jsonl file) as --path.
"""
from __future__ import annotations

import glob
import json
import os
from collections import Counter
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from ..extract.base import Attribution
from ..extract.baselines import BASELINES
from ..extract.judge import JUDGES
from ..llm import Completer
from ..schema import Event, Trace
from .metrics import evaluate_attribution, md_table

_HIST = ("history", "trajectory", "messages", "log")
_AGENT = ("mistake_agent", "failure_agent", "agent")
_STEP = ("mistake_step", "failure_step", "step")
_Q = ("question", "task", "query", "problem")


def _first(d: Dict[str, Any], keys: Tuple[str, ...], default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def _rows_from_file(path: str) -> List[Dict[str, Any]]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".parquet":
        try:
            import pandas as pd  # type: ignore
        except ImportError as e:
            raise RuntimeError("reading .parquet needs: pip install pandas pyarrow") from e
        return pd.read_parquet(path).to_dict("records")
    if ext == ".jsonl":
        with open(path, encoding="utf-8") as fh:
            return [json.loads(l) for l in fh if l.strip()]
    if ext == ".json":
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
        if isinstance(d, dict) and any(k in d for k in _HIST):
            return [d]
        return d if isinstance(d, list) else list(d.values())
    return []


def load_rows(path: str) -> List[Dict[str, Any]]:
    files = [path] if os.path.isfile(path) else sorted(
        f for ext in ("parquet", "jsonl", "json") for f in glob.glob(os.path.join(path, "**", f"*.{ext}"), recursive=True))
    rows: List[Dict[str, Any]] = []
    for f in files:
        sub = "algorithm" if "algorithm" in os.path.basename(f).lower() else ("hand" if "hand" in os.path.basename(f).lower() else "")
        for r in _rows_from_file(f):
            r = dict(r)
            r.setdefault("_subset", sub or os.path.basename(f))
            rows.append(r)
    if not rows:
        raise FileNotFoundError(f"no .parquet/.jsonl/.json rows found under {path}")
    return rows


def _history(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    h = _first(row, _HIST, [])
    if isinstance(h, str):
        h = json.loads(h)
    return [dict(m) for m in h]


def _speaker(m: Dict[str, Any]) -> str:
    return str(m.get("name") or m.get("role") or m.get("agent") or "unknown")


def detect_index_base(rows: Iterable[Dict[str, Any]]) -> int:
    score = {0: 0, 1: 0}
    for r in rows:
        h = _history(r)
        try:
            s = int(_first(r, _STEP))
        except (TypeError, ValueError):
            continue
        a = str(_first(r, _AGENT, ""))
        for base in (0, 1):
            i = s - base
            if 0 <= i < len(h) and _speaker(h[i]) == a:
                score[base] += 1
    return 0 if score[0] >= score[1] else 1


def to_traces(rows: List[Dict[str, Any]], index_base: Optional[int] = None) -> List[Trace]:
    base = detect_index_base(rows) if index_base is None else index_base
    out: List[Trace] = []
    for n, r in enumerate(rows):
        hist = _history(r)
        evs = [Event(id=i, kind="message", actor=_speaker(m), content={"text": str(m.get("content", ""))}, ts=float(i))
               for i, m in enumerate(hist)]
        try:
            step = int(_first(r, _STEP)) - base
        except (TypeError, ValueError):
            step = None
        if step is None or not (0 <= step < len(evs)):
            continue
        t = Trace(trace_id=f"ww-{r.get('_subset', '')}-{n}", question=str(_first(r, _Q, "")), events=evs,
                  outcome={"answer": None, "success": False},
                  labels={"fault": {"kind": str(r.get("_subset", "whowhen")), "critical_event": step,
                                    "agent": str(_first(r, _AGENT, "")), "effective": True}},
                  meta={"source": "external", "dataset": "Who&When", "index_base": base})
        out.append(t)
    return out


def inspect(path: str) -> str:
    rows = load_rows(path)
    base = detect_index_base(rows)
    traces = to_traces(rows, base)
    lens = [len(t.events) for t in traces]
    ag = Counter(t.labels["fault"]["agent"] for t in traces)
    return (f"rows={len(rows)} usable={len(traces)} detected index base={base}\n"
            f"columns={sorted(rows[0].keys())}\nevents/trace min={min(lens)} mean={sum(lens)/len(lens):.1f} max={max(lens)}\n"
            f"subsets={Counter(r.get('_subset') for r in rows)}\ntop mistake agents={ag.most_common(5)}\n")


def evaluate(path: str, outdir: str, judge: Optional[Completer] = None, limit: Optional[int] = None,
             index_base: Optional[int] = None) -> Dict[str, Any]:
    traces = to_traces(load_rows(path), index_base)
    if limit:
        traces = traces[:limit]
    rows = [evaluate_attribution(traces, fn, "output_only") for fn in BASELINES.values()]
    if judge:
        from functools import partial
        for n, fn in JUDGES.items():
            rows.append(evaluate_attribution(traces, partial(fn, llm=judge), "output_only"))
    os.makedirs(outdir, exist_ok=True)
    res = {"dataset": "Who&When", "n": len(traces), "rows": rows}
    with open(os.path.join(outdir, "whowhen_results.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)
    md = ["# Who&When results (output-only view)\n",
          f"traces: {len(traces)}; step acc = exact match of the predicted step index with `mistake_step`.\n",
          md_table(rows, ["method", "n", "step_acc", "step_acc_ci95", "hit@3", "mrr", "actor_acc"],
                   ["method", "n", "step acc", "95% CI", "hit@3", "MRR", "actor acc"])]
    with open(os.path.join(outdir, "whowhen_results.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(md))
    return res
