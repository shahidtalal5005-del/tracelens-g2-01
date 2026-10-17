"""Common result type for step-attribution methods."""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..schema import Trace


@dataclass
class Attribution:
    method: str
    ranking: List[Tuple[int, float, str]]            # (event_id, score, reason), best first
    cost: Dict[str, Any] = field(default_factory=dict)

    @property
    def top(self) -> Optional[int]:
        return self.ranking[0][0] if self.ranking else None


def rng_for(trace: Trace, salt: str = "") -> random.Random:
    h = hashlib.sha256((trace.trace_id + salt).encode()).hexdigest()
    return random.Random(int(h[:12], 16))


def complete_ranking(first: List[Tuple[int, float, str]], trace: Trace,
                     tail_order: Optional[List[int]] = None) -> List[Tuple[int, float, str]]:
    """Append every event not already ranked (keeps ranking a permutation of the trace)."""
    seen = {e for e, _, _ in first}
    rest = tail_order if tail_order is not None else [e.id for e in trace.events]
    out = list(first)
    score = (min((s for _, s, _ in first), default=0.0)) - 1.0
    for eid in rest:
        if eid not in seen:
            out.append((eid, score, "fallback order"))
            seen.add(eid)
            score -= 1e-3
    return out
