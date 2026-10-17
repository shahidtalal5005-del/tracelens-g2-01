"""LLM-judge attribution baselines in the style of the base paper (Who&When).

IMPORTANT: these are re-implementations written from the *descriptions* of the
three strategies (all-at-once, step-by-step, binary search); they are not the
authors' code or prompts.  Replace/compare with the official repository before
making claims about reproducing the paper's numbers.

All three need a ``Completer`` (Ollama / OpenAI-compatible).  Without one the
benchmark simply skips them.
"""
from __future__ import annotations

import json
import re
from typing import List, Optional, Tuple

from ..llm import Completer
from ..schema import Event, Trace
from .base import Attribution, complete_ranking

_HEAD = ("You are debugging a failed run of an AI agent system. The task was:\n{q}\n"
         "The system's final answer was: {a}\nThis answer is WRONG.\n")


def _render(events: List[Event], max_chars: int = 600) -> str:
    out = []
    for e in events:
        body = json.dumps(e.content, ensure_ascii=False)
        if len(body) > max_chars:
            body = body[:max_chars] + "...(truncated)"
        inp = f" inputs={json.dumps(e.inputs, ensure_ascii=False)}" if e.inputs else ""
        out.append(f"Step {e.id} | {e.actor} | {e.kind}{inp}: {body}")
    return "\n".join(out)


def _parse_int(text: str, valid: List[int]) -> Optional[int]:
    m = re.search(r'"?step"?\s*[:=]\s*(\d+)', text, re.I) or re.search(r"\b(\d+)\b", text)
    if m and int(m.group(1)) in valid:
        return int(m.group(1))
    return None


def _result(name: str, trace: Trace, pred: Optional[int], c: Completer, c0: dict) -> Attribution:
    u1 = c.usage()
    cost = {k: u1[k] - c0[k] for k in ("calls", "prompt_tokens", "completion_tokens", "seconds")}
    first = [(pred, 1.0, "LLM judge")] if pred is not None else []
    return Attribution(name, complete_ranking(first, trace), cost)


def all_at_once(trace: Trace, llm: Completer) -> Attribution:
    c0 = llm.usage()
    ids = [e.id for e in trace.events]
    prompt = (_HEAD.format(q=trace.question, a=trace.outcome.get("answer")) + "Full log:\n" + _render(trace.events) +
              '\n\nWhich single step is the decisive mistake that made the run fail? Reply as JSON {"step": <number>, "reason": "..."}.')
    return _result("llm_all_at_once", trace, _parse_int(llm(prompt), ids), llm, c0)


def step_by_step(trace: Trace, llm: Completer) -> Attribution:
    c0 = llm.usage()
    ids = [e.id for e in trace.events]
    pred = None
    for e in trace.events:
        if e.kind == "goal":
            continue
        prompt = (_HEAD.format(q=trace.question, a=trace.outcome.get("answer")) + "Log so far:\n" +
                  _render(trace.events[: e.id]) + "\n\nStep under review:\n" + _render([e]) +
                  "\n\nIs this step a mistake (wrong, unsupported or caused by earlier wrong data)? Answer YES or NO first, then a short reason.")
        if llm(prompt).strip().upper().startswith("YES"):
            pred = e.id
            break
    return _result("llm_step_by_step", trace, pred, llm, c0)


def binary_search(trace: Trace, llm: Completer) -> Attribution:
    c0 = llm.usage()
    lo, hi = 1, len(trace.events) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        prompt = (_HEAD.format(q=trace.question, a=trace.outcome.get("answer")) + "Log:\n" + _render(trace.events[lo:hi + 1]) +
                  f"\n\nThe decisive mistake lies somewhere in steps {lo}..{hi}. Is it in the FIRST half (steps {lo}..{mid}) "
                  f"or the SECOND half (steps {mid + 1}..{hi})? Answer FIRST or SECOND.")
        if llm(prompt).strip().upper().startswith("FIRST"):
            hi = mid
        else:
            lo = mid + 1
    return _result("llm_binary_search", trace, lo if len(trace.events) > 1 else None, llm, c0)


JUDGES = {"all_at_once": all_at_once, "step_by_step": step_by_step, "binary_search": binary_search}
