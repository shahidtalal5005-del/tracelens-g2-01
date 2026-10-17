"""Simple non-agentic, non-LLM baselines for step attribution.

These are the reference points the proposed extractors must beat; they also
work on any imported trace (Who&When) because they only use ordering and
status / keyword information.
"""
from __future__ import annotations

import re
from typing import List

from ..schema import Trace
from .base import Attribution, complete_ranking, rng_for

_ERR_WORDS = re.compile(r"\b(error|exception|traceback|failed|failure|cannot|unable|timeout|timed out|invalid|not found)\b", re.I)


def random_event(trace: Trace) -> Attribution:
    ids = [e.id for e in trace.events if e.kind != "goal"] or [e.id for e in trace.events]
    rng_for(trace, "random").shuffle(ids)
    return Attribution("random", complete_ranking([(i, 1.0 - k * 1e-3, "random") for k, i in enumerate(ids)], trace))


def last_event(trace: Trace) -> Attribution:
    ids = [e.id for e in reversed(trace.events)]
    return Attribution("last_step", [(i, 1.0 - k * 1e-3, "last-step") for k, i in enumerate(ids)])


def first_event(trace: Trace) -> Attribution:
    ids = [e.id for e in trace.events if e.kind != "goal"] or [e.id for e in trace.events]
    return Attribution("first_step", complete_ranking([(i, 1.0 - k * 1e-3, "first-step") for k, i in enumerate(ids)], trace))


def last_tool_output(trace: Trace) -> Attribution:
    """Blame the last tool/observation event (a common debugging habit)."""
    kinds = ("tool_result", "error", "message")
    ids = [e.id for e in reversed(trace.events) if e.kind in kinds]
    return Attribution("last_tool_output", complete_ranking([(i, 1.0 - k * 1e-3, "last tool output") for k, i in enumerate(ids)],
                                                            trace, [e.id for e in reversed(trace.events)]))


def first_error_signal(trace: Trace) -> Attribution:
    """First event with status=error / error keywords / a retry; otherwise last step."""
    flagged: List[int] = []
    for e in trace.events:
        if e.status == "error" or e.kind in ("error", "retry"):
            flagged.append(e.id)
        elif e.kind in ("tool_result", "message") and _ERR_WORDS.search(str(e.content)):
            flagged.append(e.id)
    first = [(i, 2.0 - k * 1e-3, "error signal") for k, i in enumerate(flagged)]
    return Attribution("first_error_signal", complete_ranking(first, trace, [e.id for e in reversed(trace.events)]))


BASELINES = {
    "random": random_event,
    "first_step": first_event,
    "last_step": last_event,
    "last_tool_output": last_tool_output,
    "first_error_signal": first_error_signal,
}
