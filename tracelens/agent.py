"""Instrumented research agent (plan -> search -> read -> remember -> compute -> answer).

Every action is written to a ``Trace`` through ``Recorder``.  The agent also
supports *fault injection*: exactly one fault of a chosen kind is applied and
the id of the event where the fault becomes visible is stored in
``trace.labels['fault']['critical_event']``.  That label is the ground truth for
step-level attribution (it is never visible through ``Trace.view``).
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .llm import ScriptedPolicy
from .schema import Event, Trace
from .world import World

FAULT_KINDS = (
    "retrieval_miss",            # search never surfaces the right document (persistent)
    "retrieval_miss_transient",  # first search is wrong, the agent's retry recovers
    "tool_corruption",           # read_doc returns a document with an altered fact
    "source_error",              # the *source document itself* is wrong; replaying the tool reproduces it
    "tool_error",                # read_doc fails twice, agent proceeds without evidence
    "memory_swap",               # memory returns another key's value
    "calc_error",                # calculator returns a wrong result
    "plan_omission",             # planner drops a needed lookup
    "unsupported_claim",         # final answer is not supported by the cited evidence
)

APPLICABLE = {
    "retrieval_miss": {"lookup", "hop2", "sum", "compare"},
    "retrieval_miss_transient": {"lookup", "hop2", "sum", "compare"},
    "tool_corruption": {"lookup", "hop2", "sum", "compare"},
    "source_error": {"lookup", "hop2", "sum", "compare"},
    "tool_error": {"lookup", "hop2", "sum", "compare"},
    "memory_swap": {"sum", "compare"},
    "calc_error": {"sum", "compare"},
    "plan_omission": {"sum", "compare"},
    "unsupported_claim": {"lookup", "hop2", "sum", "compare"},
}

_ATTR_WORDS = {"revenue": "annual revenue", "hq_city": "headquartered", "population": "population"}


@dataclass
class Fault:
    kind: str
    step: int = 0          # which lookup (0-based) is affected


class Recorder:
    def __init__(self, trace_id: str, question: str):
        self.trace = Trace(trace_id=trace_id, question=question)

    def add(self, kind: str, actor: str, content: Dict[str, Any], inputs: Optional[Dict[str, Any]] = None,
            parents: Optional[List[int]] = None, status: str = "ok") -> int:
        eid = len(self.trace.events)
        self.trace.events.append(Event(id=eid, kind=kind, actor=actor, content=content, inputs=inputs or {},
                                       parents=sorted(set(parents or [])), status=status, ts=float(eid)))
        return eid


def applicable(fault_kind: str, task: Dict[str, Any]) -> bool:
    return task["type"] in APPLICABLE[fault_kind]


class Agent:
    def __init__(self, world: World, policy: Optional[ScriptedPolicy] = None, noise: bool = True):
        self.world = world
        self.policy = policy or ScriptedPolicy()
        self.noise = noise

    # ------------------------------------------------------------------
    def _corrupt(self, text: str, attr: str, entity: str) -> str:
        if attr == "hq_city":
            other = next(c for c in sorted(self.world.cities) if c not in text)
            return re.sub(r"headquartered in [A-Za-z]+", f"headquartered in {other}", text)
        pat = {"revenue": r"revenue of (\d+)", "population": r"population of (\d+)"}[attr]
        return re.sub(pat, lambda m: m.group(0).replace(m.group(1), str(int(m.group(1)) + 70 if attr == "revenue"
                                                                           else int(m.group(1)) + 14000)), text, count=1)

    def _wrong_doc(self, entity: str) -> str:
        if entity in self.world.distractor_of:
            return self.world.distractor_of[entity]
        pool = sorted(self.world.cities if entity in self.world.cities else self.world.companies)
        other = pool[(pool.index(entity) + 1) % len(pool)] if entity in pool else pool[0]
        return self.world.doc_of[other]

    # ------------------------------------------------------------------
    def run(self, task: Dict[str, Any], fault: Optional[Fault] = None, seed: int = 0) -> Trace:
        W, P = self.world, self.policy
        rng = random.Random(f"{task['task_id']}-{seed}")
        rec = Recorder(f"{task['task_id']}-{fault.kind if fault else 'clean'}-s{seed}", task["question"])
        crit: Optional[int] = None
        g = rec.add("goal", "user", {"text": task["question"]})

        plan = [list(s) for s in P.plan(task["question"])]
        lookups_all = [i for i, s in enumerate(plan) if s[0] == "lookup"]
        dropped = None
        if fault and fault.kind == "plan_omission" and len(lookups_all) >= 2:
            dropped = plan.pop(lookups_all[-1])
        plan_id = rec.add("plan", "planner", {"steps": plan}, inputs={"question": task["question"]}, parents=[g])
        if dropped is not None:
            crit = plan_id

        if self.noise and rng.random() < 0.4:   # harmless exploratory search (benign noise)
            mentioned = {x[1] for x in plan if x[0] == "lookup"}
            other = rng.choice([c for c in sorted(W.companies) if c not in mentioned and c not in task["question"]])
            c = rec.add("tool_call", "retriever", {"tool": "search"}, {"args": {"query": f"{other} overview", "k": 3}}, [plan_id])
            rec.add("tool_result", "tool", {"tool": "search", "output": W.call("search", {"query": f"{other} overview", "k": 3})}, parents=[c])

        facts: List[Dict[str, Any]] = []          # {key, value, write_id, entity, attr}
        calc_result_id: Optional[int] = None
        calc_out: Any = None
        lookup_idx = -1
        last_ctx = plan_id
        gave_up = False

        for step in plan:
            if step[0] == "lookup":
                lookup_idx += 1
                entity, attr = step[1], step[2]
                if gave_up:
                    break
                if entity == "<hq_city>":
                    prev = next((f for f in facts if f["attr"] == "hq_city"), None)
                    if prev is None or prev["value"] is None:
                        gave_up = True
                        break
                    mr = rec.add("memory_read", "memory", {"key": prev["key"], "value": prev["value"]},
                                 {"key": prev["key"]}, [prev["write_id"]])
                    entity = prev["value"]
                    last_ctx = mr
                faulted_here = bool(fault) and fault.step == lookup_idx
                value, doc_id, res_id = self._lookup(rec, rng, entity, attr, last_ctx, fault if faulted_here else None)
                if faulted_here and fault.kind in ("retrieval_miss", "retrieval_miss_transient", "tool_corruption", "source_error", "tool_error"):
                    crit = self._crit_from_marker(rec, fault.kind)
                w = rec.add("memory_write", "memory", {"key": f"{entity}.{attr}", "value": value, "source_doc": doc_id},
                            {"key": f"{entity}.{attr}"}, [res_id] if res_id is not None else [])
                facts.append({"key": f"{entity}.{attr}", "value": value, "write_id": w, "entity": entity, "attr": attr})
                last_ctx = w
            else:  # compute
                op = step[1]
                reads = []
                vals = []
                for k, f in enumerate(facts):
                    if f["attr"] != "revenue" or f["value"] is None:
                        continue
                    v = f["value"]
                    if fault and fault.kind == "memory_swap" and len(facts) >= 2 and not reads:
                        other = next(x for x in facts if x is not f and x["attr"] == "revenue")
                        v = other["value"]
                    mr = rec.add("memory_read", "memory", {"key": f["key"], "value": v}, {"key": f["key"]}, [f["write_id"]])
                    if fault and fault.kind == "memory_swap" and not reads:
                        crit = mr
                    reads.append(mr)
                    vals.append(v)
                if not vals:
                    gave_up = True
                    break
                call = rec.add("tool_call", "calculator", {"tool": "calc"}, {"args": {"op": op, "values": vals}}, reads)
                out = W.call("calc", {"op": op, "values": vals})
                if fault and fault.kind == "calc_error":
                    out = out + 10 if op == "sum" else (1 - out if len(vals) > 1 else out + 1)
                calc_result_id = rec.add("tool_result", "tool", {"tool": "calc", "output": out}, parents=[call])
                if fault and fault.kind == "calc_error":
                    crit = calc_result_id
                calc_out = out
                last_ctx = calc_result_id

        # ---------------- final answer --------------------------------
        answer: Any
        evidence: List[int]
        if gave_up or not facts or any(f["value"] is None for f in facts):
            answer, evidence = "unknown", [f["write_id"] for f in facts]
        elif calc_result_id is not None:
            if plan[-1][1] == "argmax":
                names = [f["entity"] for f in facts if f["attr"] == "revenue"]
                answer = names[int(calc_out)] if int(calc_out) < len(names) else "unknown"
            else:
                answer = int(calc_out)
            evidence = [calc_result_id]
        else:
            answer, evidence = int(facts[-1]["value"]) if facts[-1]["value"].isdigit() else facts[-1]["value"], [facts[-1]["write_id"]]
        if fault and fault.kind == "unsupported_claim":
            if isinstance(answer, int):
                answer = answer + 111
            else:
                names = [f["entity"] for f in facts if f["attr"] == "revenue"]
                answer = next((n for n in names if n != answer), "unknown")
        fid = rec.add("final_answer", "answerer", {"answer": answer, "claims": [
            {"text": "final answer", "value": answer, "evidence": evidence}]}, inputs={"question": task["question"]},
            parents=evidence or [last_ctx])
        if fault and fault.kind == "unsupported_claim":
            crit = fid

        success = str(answer) == str(task["gold"])
        t = rec.trace
        t.outcome = {"answer": answer, "success": success}
        t.meta.update({"task_id": task["task_id"], "task_type": task["type"], "policy": P.name, "seed": seed})
        gold_ev = [e.id for e in t.events if e.kind == "tool_result" and e.content.get("tool") == "read_doc"
                   and e.parents and t.events[e.parents[0]].inputs.get("args", {}).get("doc_id") in task["gold_docs"]]
        t.labels = {"gold": task["gold"], "gold_docs": task["gold_docs"], "gold_evidence_events": gold_ev,
                    "fault": None if not fault else {"kind": fault.kind, "step": fault.step, "critical_event": crit,
                                                      "effective": not success}}
        return t

    # ------------------------------------------------------------------
    def _crit_from_marker(self, rec: Recorder, kind: str) -> Optional[int]:
        return getattr(rec, "_marker", {}).get(kind)

    def _mark(self, rec: Recorder, kind: str, eid: int) -> None:
        if not hasattr(rec, "_marker"):
            rec._marker = {}                      # type: ignore[attr-defined]
        rec._marker.setdefault(kind, eid)         # type: ignore[attr-defined]

    def _search(self, rec: Recorder, query: str, parent: int, wrong_for: Optional[str], kind: Optional[str]):
        W = self.world
        call = rec.add("tool_call", "retriever", {"tool": "search"}, {"args": {"query": query, "k": 5}}, [parent])
        results = W.call("search", {"query": query, "k": 5})
        if wrong_for is not None:
            bad = self._wrong_doc(wrong_for)
            good = W.doc_of.get(wrong_for)
            results = [{"doc_id": bad, "title": W.docs[bad].title, "score": results[0]["score"] if results else 1.0}] + \
                      [r for r in results if r["doc_id"] not in (good, bad)]
        res = rec.add("tool_result", "tool", {"tool": "search", "output": results}, parents=[call])
        if wrong_for is not None and kind:
            self._mark(rec, kind, res)
        return results, res

    def _lookup(self, rec: Recorder, rng: random.Random, entity: str, attr: str, parent: int, fault: Optional[Fault]):
        W, P = self.world, self.policy
        kind = fault.kind if fault else None
        query = f"{entity} {_ATTR_WORDS[attr]}"
        persistent = kind == "retrieval_miss"
        results, res = self._search(rec, query, parent, entity if kind in ("retrieval_miss", "retrieval_miss_transient") else None, kind)
        pick = next((r for r in results if r["title"] == entity), None)
        if pick is None:                                # retry once with a plain entity query
            rt = rec.add("retry", "retriever", {"reason": "no result titled exactly like the entity"}, {"query": query}, [res])
            results, res = self._search(rec, entity, rt, entity if persistent else None, None)
            pick = next((r for r in results if r["title"] == entity), None) or (results[0] if results else None)
        if pick is None:
            return None, None, res
        doc_id = pick["doc_id"]
        for attempt in range(2):
            call = rec.add("tool_call", "reader", {"tool": "read_doc"}, {"args": {"doc_id": doc_id}}, [res])
            if kind == "tool_error":
                err = rec.add("error", "tool", {"tool": "read_doc", "message": "TimeoutError: document store did not respond"},
                              parents=[call], status="error")
                self._mark(rec, kind, err)
                if attempt == 0:
                    res = rec.add("retry", "reader", {"reason": "tool error"}, {"doc_id": doc_id}, [err])
                    continue
                return None, doc_id, err
            text = W.call("read_doc", {"doc_id": doc_id})
            if kind in ("tool_corruption", "source_error"):
                text = self._corrupt(text, attr, entity)
                if kind == "source_error":          # environment snapshot: replaying the tool reproduces the wrong page
                    rec.trace.meta.setdefault("replay_overrides", {})[doc_id] = text
            res = rec.add("tool_result", "tool", {"tool": "read_doc", "output": text}, parents=[call])
            if kind in ("tool_corruption", "source_error"):
                self._mark(rec, kind, res)
            break
        return P.read(text, entity, attr), doc_id, res
