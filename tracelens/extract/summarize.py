"""Gap 3 component: budgeted critical-trace summaries with verifiable references.

A summary is a short list of *items*; each item points at one real event and
carries a verbatim ``quote`` taken from that event.  This lets us measure
faithfulness mechanically (does the quote occur in the cited event?) instead of
judging readability.  ``CriticalTraceSummarizer`` chooses items with the
evidence verifier (Gap 1) plus the dependency chain behind the final answer
(only possible when links are logged - Gap 2).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from ..llm import Completer
from ..schema import Event, Trace, approx_tokens, trace_tokens
from .base import rng_for
from .evidence import EvidenceVerifier, ROLE, Violation


@dataclass
class Item:
    event_id: int
    role: str            # goal | final_answer | <ROLE value> | retry | error | evidence_chain | context
    quote: str
    note: str = ""


@dataclass
class Summary:
    method: str
    items: List[Item] = field(default_factory=list)

    def ids(self) -> List[int]:
        return [i.event_id for i in self.items]

    def render(self) -> str:
        lines = []
        for it in self.items:
            tag = f"<{it.role}>" if it.role not in ("context",) else ""
            note = f"  -- {it.note}" if it.note else ""
            lines.append(f"[{it.event_id}] {tag} {it.quote}{note}".replace("  ", " ").strip())
        return "\n".join(lines)

    def tokens(self) -> int:
        return approx_tokens(self.render())


def _quote(e: Event, limit: int = 140) -> str:
    c = e.content
    if e.kind == "tool_result":
        out = c.get("output")
        if c.get("tool") == "search" and isinstance(out, list):
            out = ", ".join(f"{x.get('doc_id')}:{x.get('title')}" for x in out[:3])
        s = f"{c.get('tool')} -> {out}"
    elif e.kind == "tool_call":
        s = f"{c.get('tool')}({json.dumps(e.inputs.get('args', {}), ensure_ascii=False)})"
    elif e.kind == "final_answer":
        s = f"answer = {c.get('answer')}"
    elif e.kind == "plan":
        s = f"plan = {c.get('steps')}"
    elif e.kind == "goal":
        s = str(c.get("text", ""))
    else:
        s = json.dumps(c, ensure_ascii=False)
    s = re.sub(r"\s+", " ", s)
    return s if len(s) <= limit else s[: limit - 3] + "..."


def render_raw(trace: Trace) -> str:
    return "\n".join(e.text() for e in trace.events)


# ---------------------------------------------------------------------------------
class CriticalTraceSummarizer:
    name = "critical_trace"

    def __init__(self, verifier: Optional[EvidenceVerifier] = None, use_chain: bool = True):
        self.verifier = verifier or EvidenceVerifier()
        self.use_chain = use_chain

    def summarize(self, trace: Trace, budget: int = 6) -> Summary:
        ev = trace.events
        goal = next((e for e in ev if e.kind == "goal"), ev[0])
        fin = next((e for e in reversed(ev) if e.kind == "final_answer"), ev[-1])
        picked: Dict[int, Item] = {}

        def put(e: Event, role: str, note: str = "") -> None:
            picked.setdefault(e.id, Item(e.id, role, _quote(e), note))

        put(goal, "goal")
        put(fin, "final_answer")
        viol = sorted(self.verifier.check(trace), key=lambda v: v.event_id)
        for v in viol:                                          # 1) verified problems, earliest first
            if len(picked) >= budget:
                break
            put(ev[v.event_id], ROLE[v.rule], v.message)
        if self.use_chain and len(picked) < budget:             # 2) dependency chain behind the answer
            seen, stack = set(), list(fin.parents)
            order: List[int] = []
            while stack:
                i = stack.pop()
                if i in seen:
                    continue
                seen.add(i)
                order.append(i)
                stack.extend(trace.by_id(i).parents if trace.by_id(i) else [])
            for i in sorted(order, reverse=True):
                e = ev[i]
                if len(picked) >= budget:
                    break
                if e.kind in ("tool_result", "memory_write", "error", "retry"):
                    put(e, "evidence_chain")
        if len(picked) < budget:                                # 3) structural events: errors / retries / plan
            for e in ev:
                if len(picked) >= budget:
                    break
                if e.kind in ("error", "retry", "plan"):
                    put(e, "error" if e.kind == "error" else e.kind)
        items = sorted(picked.values(), key=lambda it: it.event_id)
        return Summary(self.name if self.use_chain else self.name + "[no_chain]", items)


# ---- baseline summarisers ---------------------------------------------------------------
def _mk(name: str, trace: Trace, ids: Sequence[int]) -> Summary:
    return Summary(name, [Item(i, "context", _quote(trace.events[i])) for i in sorted(set(ids))])


def head_k(trace: Trace, budget: int) -> Summary:
    return _mk("head_k", trace, [e.id for e in trace.events][:budget])


def tail_k(trace: Trace, budget: int) -> Summary:
    return _mk("tail_k", trace, [e.id for e in trace.events][-budget:])


def random_k(trace: Trace, budget: int) -> Summary:
    ids = [e.id for e in trace.events]
    rng_for(trace, "sum").shuffle(ids)
    return _mk("random_k", trace, ids[:budget])


def results_only_k(trace: Trace, budget: int) -> Summary:
    ids = [e.id for e in trace.events if e.kind in ("tool_result", "error")]
    return _mk("last_tool_results_k", trace, ids[-budget:])


def llm_summary(trace: Trace, budget: int, llm: Completer) -> Summary:
    """Prompted LLM baseline: pick `budget` events. Item ids are validated; quotes come from the real event."""
    prompt = ("Below is the log of an AI agent run. Pick the {k} steps most important for understanding why the run "
              "ended as it did. Reply with ONLY a JSON list of step numbers.\n\n{log}").format(k=budget, log=render_raw(trace))
    raw = llm(prompt)
    m = re.search(r"\[[^\]]*\]", raw)
    ids: List[int] = []
    try:
        ids = [int(x) for x in json.loads(m.group(0))] if m else []
    except (ValueError, json.JSONDecodeError):
        ids = []
    ids = [i for i in ids if 0 <= i < len(trace.events)][:budget]
    return _mk("llm_summary", trace, ids)


SUMMARISERS = {"head_k": head_k, "tail_k": tail_k, "random_k": random_k, "last_tool_results_k": results_only_k}


# ---- faithfulness ---------------------------------------------------------------------------
def faithfulness(summary: Summary, trace: Trace) -> float:
    """Fraction of items whose cited event exists and whose quote is grounded in that event.

    Grounded = every number and every capitalised/identifier token of the quote's payload occurs in the event.
    Exact-substring checking is too strict once quotes are truncated, so we compare tokens.
    """
    if not summary.items:
        return 0.0
    ok = 0
    for it in summary.items:
        e = trace.by_id(it.event_id)
        if e is None:
            continue
        hay = (json.dumps(e.content, ensure_ascii=False) + " " + json.dumps(e.inputs, ensure_ascii=False))
        hay_tokens = set(re.findall(r"[A-Za-z0-9_.]+", hay))
        q_tokens = [t for t in re.findall(r"[A-Za-z0-9_.]+", it.quote.rstrip(".")) if len(t) > 1 or t.isdigit()]
        generic = {"answer", "plan", "None", "tool", "args", "output", "question", "text", "steps", "lookup", "compute"}
        need = [t for t in q_tokens if t not in generic]
        if all(t in hay_tokens or any(t in h for h in hay_tokens) for t in need):
            ok += 1
    return ok / len(summary.items)
