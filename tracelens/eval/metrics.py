"""Metrics for step attribution and for trace summaries."""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, List, Optional

from ..extract.base import Attribution
from ..extract.summarize import Summary, faithfulness, render_raw
from ..schema import Trace, VIEW_FULL, approx_tokens, trace_tokens


def wilson(k: int, n: int, z: float = 1.96) -> List[float]:
    """95% Wilson score interval for a proportion."""
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def evaluate_attribution(traces: Iterable[Trace], method: Callable[[Trace], Attribution],
                         view: str = VIEW_FULL, label_key: str = "fault") -> Dict[str, Any]:
    n = top1 = hit3 = agent_ok = 0
    rr = 0.0
    cost = defaultdict(float)
    by_kind: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
    name = ""
    for t in traces:
        crit = t.labels[label_key]["critical_event"]
        att = method(t.view(view))
        name = att.method
        order = [e for e, _, _ in att.ranking]
        rank = order.index(crit) + 1 if crit in order else len(t.events) + 1
        n += 1
        top1 += rank == 1
        hit3 += rank <= 3
        rr += 1.0 / rank
        pred = att.top
        if pred is not None and t.events[pred].actor == t.events[crit].actor:
            agent_ok += 1
        for k, v in att.cost.items():
            cost[k] += v
        bk = by_kind[t.labels[label_key]["kind"]]
        bk[0] += rank == 1
        bk[1] += 1
    return {
        "method": name, "view": view, "n": n,
        "step_acc": round(top1 / n, 4) if n else 0.0, "step_acc_ci95": wilson(top1, n),
        "hit@3": round(hit3 / n, 4) if n else 0.0, "mrr": round(rr / n, 4) if n else 0.0,
        "actor_acc": round(agent_ok / n, 4) if n else 0.0,
        "by_kind": {k: {"acc": round(v[0] / v[1], 4), "n": v[1]} for k, v in sorted(by_kind.items())},
        "cost": {k: round(v, 3) for k, v in cost.items()},
    }


def evaluate_summaries(traces: Iterable[Trace], summarize: Callable[[Trace], Summary], view: str = VIEW_FULL) -> Dict[str, Any]:
    """Recall of the injected-fault event, labelled recall, evidence recall, compression, faithfulness."""
    n = rec = lab = 0
    comp = faith = 0.0
    toks = 0.0
    name = ""
    ev_hit = ev_tot = 0
    for t in traces:
        v = t.view(view)
        s = summarize(v)
        name = s.method
        crit = t.labels["fault"]["critical_event"]
        n += 1
        rec += crit in s.ids()
        lab += any(i.event_id == crit and i.role != "context" for i in s.items)
        raw = max(1, trace_tokens(v))
        comp += s.tokens() / raw
        toks += s.tokens()
        faith += faithfulness(s, v)
        gold = t.labels.get("gold_evidence_events", [])
        ev_tot += len(gold)
        ev_hit += sum(1 for g in gold if g in s.ids())
    return {"method": name, "view": view, "n": n,
            "fault_recall": round(rec / n, 4) if n else 0.0, "fault_recall_ci95": wilson(rec, n),
            "labelled_recall": round(lab / n, 4) if n else 0.0,
            "evidence_recall": round(ev_hit / ev_tot, 4) if ev_tot else 0.0,
            "compression": round(comp / n, 4) if n else 0.0, "mean_tokens": round(toks / n, 1) if n else 0.0,
            "faithfulness": round(faith / n, 4) if n else 0.0}


def md_table(rows: List[Dict[str, Any]], cols: List[str], headers: Optional[List[str]] = None) -> str:
    headers = headers or cols
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    return "\n".join(out)
