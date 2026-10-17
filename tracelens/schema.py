"""Structured trace schema (v1).

A Trace is an ordered list of Events.  The schema separates three groups of
fields because Gap 2 of the project studies which of them an extractor needs:

* content   - what the actor produced / what the tool returned   ("output" group)
* inputs    - arguments and visible context given to the step     ("input" group)
* parents   - ids of earlier events this one depends on          ("link" group)

Ground-truth labels live in ``Trace.labels`` and are never part of a view that
is handed to an extractor (see ``Trace.view``).
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterable, List, Optional

SCHEMA_VERSION = "1.0"

KINDS = (
    "goal",          # the user task
    "plan",          # list of planned steps
    "tool_call",     # tool name + arguments (inputs)
    "tool_result",   # tool output
    "memory_write",  # fact stored in working memory
    "memory_read",   # fact read back from working memory
    "error",         # tool / runtime error
    "retry",         # agent decided to retry
    "final_answer",  # answer + claims with cited evidence event ids
    "message",       # generic agent message (used when importing external logs, e.g. Who&When)
)

# observability views used by the Gap 2 ablation
VIEW_FULL = "full"
VIEW_OUTPUT_ONLY = "output_only"      # like Who&When: only what actors emitted
VIEW_NO_LINKS = "no_links"            # inputs kept, dependency links removed
VIEW_NO_INPUTS = "no_inputs"          # links kept, tool arguments removed
VIEWS = (VIEW_FULL, VIEW_NO_LINKS, VIEW_NO_INPUTS, VIEW_OUTPUT_ONLY)


@dataclass
class Event:
    id: int
    kind: str
    actor: str
    content: Dict[str, Any] = field(default_factory=dict)
    inputs: Dict[str, Any] = field(default_factory=dict)
    parents: List[int] = field(default_factory=list)
    status: str = "ok"            # ok | error
    ts: float = 0.0               # logical clock (deterministic, not wall time)

    def text(self) -> str:
        """Compact one-line rendering used for token counting and prompts."""
        body = json.dumps(self.content, ensure_ascii=False, sort_keys=True)
        inp = json.dumps(self.inputs, ensure_ascii=False, sort_keys=True) if self.inputs else ""
        return f"[{self.id}] {self.kind}/{self.actor} {inp} -> {body}".strip()


@dataclass
class Trace:
    trace_id: str
    question: str
    events: List[Event] = field(default_factory=list)
    outcome: Dict[str, Any] = field(default_factory=dict)   # answer / success flag (observable after the run)
    labels: Dict[str, Any] = field(default_factory=dict)    # ground truth, hidden from extractors
    meta: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    # ---- access -------------------------------------------------------
    def by_id(self, eid: int) -> Optional[Event]:
        if 0 <= eid < len(self.events) and self.events[eid].id == eid:
            return self.events[eid]
        for e in self.events:
            if e.id == eid:
                return e
        return None

    def of_kind(self, *kinds: str) -> List[Event]:
        return [e for e in self.events if e.kind in kinds]

    # ---- views --------------------------------------------------------
    def view(self, mode: str = VIEW_FULL) -> "Trace":
        """Return a copy without labels and with fields removed per ``mode``."""
        if mode not in VIEWS:
            raise ValueError(f"unknown view {mode!r}; choose from {VIEWS}")
        t = copy.deepcopy(self)
        t.labels = {}
        for e in t.events:
            if mode in (VIEW_OUTPUT_ONLY, VIEW_NO_INPUTS):
                e.inputs = {}
            if mode in (VIEW_OUTPUT_ONLY, VIEW_NO_LINKS):
                e.parents = []
        t.meta = dict(t.meta, view=mode)
        return t

    # ---- (de)serialisation -------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Trace":
        evs = [Event(**e) for e in d.get("events", [])]
        return Trace(
            trace_id=d["trace_id"], question=d.get("question", ""), events=evs,
            outcome=d.get("outcome", {}), labels=d.get("labels", {}),
            meta=d.get("meta", {}), schema_version=d.get("schema_version", SCHEMA_VERSION),
        )

    def dumps(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    def validate(self) -> List[str]:
        """Return a list of schema problems (empty list = valid)."""
        problems: List[str] = []
        seen = set()
        for i, e in enumerate(self.events):
            if e.id != i:
                problems.append(f"event at position {i} has id {e.id}")
            if e.kind not in KINDS:
                problems.append(f"event {e.id}: unknown kind {e.kind!r}")
            if e.status not in ("ok", "error"):
                problems.append(f"event {e.id}: bad status {e.status!r}")
            for p in e.parents:
                if p >= e.id or p < 0:
                    problems.append(f"event {e.id}: parent {p} is not an earlier event")
            seen.add(e.id)
        if self.meta.get("source") != "external":
            if self.events and self.events[0].kind != "goal":
                problems.append("first event must be a goal")
            if self.events and self.events[-1].kind != "final_answer":
                problems.append("last event must be a final_answer")
        return problems


def write_jsonl(traces: Iterable[Trace], path: str) -> int:
    n = 0
    with open(path, "w", encoding="utf-8") as fh:
        for t in traces:
            fh.write(t.dumps() + "\n")
            n += 1
    return n


def read_jsonl(path: str) -> List[Trace]:
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(Trace.from_dict(json.loads(line)))
    return out


def approx_tokens(text: str) -> int:
    """Cheap, dependency-free token estimate (~4 characters per token)."""
    return max(1, (len(text) + 3) // 4)


def trace_tokens(trace: Trace) -> int:
    return sum(approx_tokens(e.text()) for e in trace.events)
