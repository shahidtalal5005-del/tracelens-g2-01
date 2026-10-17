"""Gap 1 component: evidence-linked verification for step attribution.

Instead of asking an LLM "where did it go wrong?", every step is checked against
cheap, deterministic invariants that tie claims to the evidence they cite:

  plan       every entity named in the goal is covered by the plan
  retrieval  a fact stored for entity E must come from a document *titled* E
  replay     re-executing a logged tool call must give the logged output
  tool_error an error that was never followed by a successful retry is a root cause
  memory     a value read from memory must equal the value that was written
  calc       a calculator result must equal its recomputation
  claim      the final answer must equal the value of the evidence it cites

Violations are ordered by position and the *earliest* one is returned as the
root cause (later symptoms of the same fault are ranked below it).

Which rules can fire depends on the observability view (Gap 2): replay and calc
need tool arguments (or dependency links); the other rules only need event
contents / order.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ..llm import _COMPANY_RE
from ..schema import Event, Trace
from ..world import World
from .base import Attribution, complete_ranking

RULES = ("plan", "retrieval", "replay", "tool_error", "memory", "calc", "claim")
SEVERITY = {"tool_error": 3.0, "replay": 3.0, "memory": 3.0, "calc": 3.0, "retrieval": 2.5, "plan": 2.5, "claim": 2.0}
ROLE = {"plan": "plan_gap", "retrieval": "wrong_evidence_retrieved", "replay": "tool_output_not_reproducible",
        "tool_error": "unrecovered_tool_error", "memory": "memory_inconsistency", "calc": "wrong_computation",
        "claim": "unsupported_claim"}
_PRIOR = {"final_answer": 0.5, "error": 0.45, "retry": 0.4, "tool_result": 0.3, "memory_read": 0.25, "plan": 0.2,
          "memory_write": 0.15, "tool_call": 0.1, "message": 0.3, "goal": 0.0}


@dataclass
class Violation:
    event_id: int
    rule: str
    message: str


def _calc(op: str, values: Sequence[Any]) -> Optional[int]:
    vals = [float(v) for v in values]
    if not vals:
        return None
    if op == "sum":
        return int(sum(vals))
    if op == "argmax":
        return int(max(range(len(vals)), key=lambda i: vals[i]))
    return None


class EvidenceVerifier:
    name = "evidence_verifier"

    def __init__(self, world: Optional[World] = None, rules: Iterable[str] = RULES, order: str = "earliest"):
        self.world = world
        self.rules = tuple(rules)
        if order not in ("earliest", "severity"):
            raise ValueError("order must be 'earliest' or 'severity'")
        self.order = order

    # ---- helpers ----------------------------------------------------------
    @staticmethod
    def _call_of(trace: Trace, res: Event) -> Optional[Event]:
        if res.parents:
            c = trace.by_id(res.parents[0])
            return c if c and c.kind == "tool_call" else None
        prev = trace.by_id(res.id - 1)
        return prev if prev is not None and prev.kind == "tool_call" else None

    def check(self, trace: Trace) -> List[Violation]:
        ev = trace.events
        out: List[Violation] = []
        on = set(self.rules)

        # plan coverage -------------------------------------------------------
        if "plan" in on:
            goal = next((e for e in ev if e.kind == "goal"), None)
            plans = [e for e in ev if e.kind == "plan"]
            if goal and plans:
                need = set(_COMPANY_RE.findall(str(goal.content.get("text", ""))))
                have = {s[1] for s in plans[-1].content.get("steps", []) if s and s[0] == "lookup"}
                if need - have:
                    out.append(Violation(plans[-1].id, "plan", f"goal mentions {sorted(need - have)} but the plan never looks them up"))

        # retrieval: stored fact must come from a document titled like its entity ----
        if "retrieval" in on:
            for w in (e for e in ev if e.kind == "memory_write"):
                doc = w.content.get("source_doc")
                if not doc:
                    continue
                entity = str(w.content.get("key", "")).rsplit(".", 1)[0]
                searches = [e for e in ev if e.kind == "tool_result" and e.content.get("tool") == "search" and e.id < w.id]
                hit = None
                for r in searches:                       # prefer the earliest search that ranked the used document first
                    outp = r.content.get("output", [])
                    if outp and outp[0].get("doc_id") == doc:
                        hit, src = outp[0], r
                        break
                if hit is None:
                    for r in searches:
                        hit = next((x for x in r.content.get("output", []) if x.get("doc_id") == doc), None)
                        if hit is not None:
                            src = r
                            break
                if hit is not None and hit.get("title") != entity:
                    out.append(Violation(src.id, "retrieval", f"document {doc} ('{hit.get('title')}') was used for '{entity}'"))

        # replay (needs tool arguments and a replayable environment) ---------------
        if "replay" in on and self.world is not None:
            env = self.world.with_overrides(trace.meta.get("replay_overrides"))
            for r in (e for e in ev if e.kind == "tool_result" and e.content.get("tool") in ("search", "read_doc")):
                call = self._call_of(trace, r)
                args = (call.inputs.get("args") if call else None)
                if not args:
                    continue
                try:
                    redo = env.call(r.content["tool"], args)
                except Exception as exc:                      # noqa: BLE001
                    out.append(Violation(r.id, "replay", f"replay raised {exc!r}"))
                    continue
                if redo != r.content.get("output"):
                    out.append(Violation(r.id, "replay", f"re-running {r.content['tool']}({args}) gives a different output"))

        # unrecovered tool errors (positional: needs only kinds and order) ---------
        if "tool_error" in on:
            for e in (x for x in ev if x.kind == "error"):
                if self._is_continuation(trace, e):
                    continue
                if not self._resolved(trace, e):
                    out.append(Violation(e.id, "tool_error", f"{e.content.get('message', 'error')} was never recovered"))

        # memory consistency ----------------------------------------------------------
        if "memory" in on:
            last: Dict[str, Any] = {}
            for e in ev:
                if e.kind == "memory_write":
                    last[e.content.get("key")] = e.content.get("value")
                elif e.kind == "memory_read" and e.content.get("key") in last:
                    if str(e.content.get("value")) != str(last[e.content["key"]]):
                        out.append(Violation(e.id, "memory", f"read {e.content['value']} but {last[e.content['key']]} was stored for {e.content['key']}"))

        # calculator recomputation ---------------------------------------------------------
        if "calc" in on:
            plan_ops = [s[1] for p in ev if p.kind == "plan" for s in p.content.get("steps", []) if s and s[0] == "compute"]
            for r in (e for e in ev if e.kind == "tool_result" and e.content.get("tool") == "calc"):
                call = self._call_of(trace, r)
                op = vals = None
                if call and call.inputs.get("args"):
                    op, vals = call.inputs["args"]["op"], call.inputs["args"]["values"]
                elif call and call.parents and plan_ops:       # rebuild inputs from dependency links
                    reads = [trace.by_id(p) for p in call.parents]
                    vals = [x.content.get("value") for x in reads if x is not None and x.kind == "memory_read"]
                    op = plan_ops[-1]
                if op is None or not vals:
                    continue
                want = _calc(op, vals)
                if want is not None and want != r.content.get("output"):
                    out.append(Violation(r.id, "calc", f"{op}({list(vals)}) = {want}, logged {r.content.get('output')}"))

        # final claims vs cited evidence ---------------------------------------------------
        if "claim" in on:
            fin = next((e for e in reversed(ev) if e.kind == "final_answer"), None)
            if fin:
                ans = fin.content.get("answer")
                for cl in fin.content.get("claims", []):
                    cited = [trace.by_id(i) for i in cl.get("evidence", [])]
                    cited = [c for c in cited if c is not None]
                    if not cited:
                        out.append(Violation(fin.id, "claim", "final answer cites no evidence"))
                        continue
                    if not any(self._supports(trace, c, ans) for c in cited):
                        out.append(Violation(fin.id, "claim", f"answer {ans!r} is not backed by cited events {[c.id for c in cited]}"))
        return out

    # ---- internals ---------------------------------------------------------------------
    @staticmethod
    def _supports(trace: Trace, c: Event, ans: Any) -> bool:
        if c.kind == "memory_write":
            return str(c.content.get("value")) == str(ans)
        if c.kind == "tool_result" and c.content.get("tool") == "calc":
            out = c.content.get("output")
            if isinstance(ans, int) or str(ans).isdigit():
                return str(out) == str(ans)
            names: List[str] = []
            for r in trace.events:
                if r.id < c.id and r.kind == "memory_read" and str(r.content.get("key", "")).endswith(".revenue"):
                    n = r.content["key"].rsplit(".", 1)[0]
                    if n in names:
                        names.remove(n)
                    names.append(n)
            return isinstance(out, int) and 0 <= out < len(names) and names[out] == ans
        return False

    @staticmethod
    def _is_continuation(trace: Trace, e: Event) -> bool:
        k = [trace.by_id(e.id - j) for j in (1, 2, 3)]
        return all(x is not None for x in k) and [x.kind for x in k] == ["tool_call", "retry", "error"]

    @staticmethod
    def _resolved(trace: Trace, e: Event) -> bool:
        i = e.id + 1
        while i < len(trace.events) and trace.events[i].kind == "retry":
            i += 1
        if i + 1 < len(trace.events) and trace.events[i].kind == "tool_call":
            return trace.events[i + 1].kind == "tool_result"
        return False

    # ---- public --------------------------------------------------------------------------
    def attribute(self, trace: Trace) -> Attribution:
        t0 = time.time()
        vs = self.check(trace)
        by_event: Dict[int, List[Violation]] = {}
        for v in vs:
            by_event.setdefault(v.event_id, []).append(v)
        if self.order == "earliest":
            flagged = sorted(by_event)
        else:
            flagged = sorted(by_event, key=lambda i: (-max(SEVERITY[v.rule] for v in by_event[i]), i))
        ranking = [(i, 10.0 - 0.01 * k, "; ".join(f"{v.rule}: {v.message}" for v in by_event[i])) for k, i in enumerate(flagged)]
        tail = sorted((e.id for e in trace.events), key=lambda i: (-_PRIOR.get(trace.events[i].kind, 0.1), -i))
        return Attribution(self.name if len(self.rules) == len(RULES) else f"{self.name}[{','.join(self.rules)}]",
                           complete_ranking(ranking, trace, tail),
                           {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "seconds": round(time.time() - t0, 4)})
