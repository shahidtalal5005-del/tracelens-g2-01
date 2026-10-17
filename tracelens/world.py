"""Synthetic, fully deterministic document world + tools + task generator.

Why synthetic?  It gives (a) a known correct answer, (b) a known gold evidence
set and (c) tools that can be *replayed* exactly, so injected faults have a
ground-truth location.  This mirrors the programmatic fault-injection idea used
for building attribution data (see research/PAPER_NOTES.md) and keeps the
whole project runnable offline.  It is NOT a substitute for Who&When or real
agent logs - see eval/whowhen.py for that.
"""
from __future__ import annotations

import math
import random
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

_SYL_A = ["Ori", "Nex", "Vol", "Zen", "Qua", "Lum", "Tar", "Kai", "Mer", "Sol", "Pax", "Dyn", "Fer", "Hel", "Jun", "Cor"]
_SYL_B = ["on", "ra", "ex", "ith", "os", "ium", "ara", "eon", "ia", "ux", "yn", "al"]
_SUFFIX = ["Systems", "Labs", "Dynamics", "Works", "Networks", "Robotics", "Analytics", "Logistics"]
_FIRST = ["Aisha", "Bilal", "Chen", "Dana", "Elena", "Farid", "Gita", "Hamza", "Ines", "Jorge", "Kofi", "Lena", "Mira", "Noor", "Omar", "Priya"]
_LAST = ["Abbas", "Baig", "Cruz", "Das", "Evans", "Fox", "Gill", "Hart", "Iqbal", "Jones", "Khan", "Lopez", "Malik", "Nair", "Orr", "Park"]
_COUNTRIES = ["Norland", "Estria", "Varos", "Kelmar", "Tovia", "Ostrava"]


@dataclass
class Doc:
    doc_id: str
    title: str
    text: str
    kind: str            # company | city | distractor | news


def _tok(s: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", s.lower())


class World:
    """Companies, cities and distractor documents generated from a seed."""

    def __init__(self, seed: int = 7, n_companies: int = 40, n_cities: int = 16):
        rng = random.Random(seed)
        self.seed = seed
        self.cities: Dict[str, Dict[str, Any]] = {}
        while len(self.cities) < n_cities:
            name = rng.choice(_SYL_A) + rng.choice(_SYL_B) + "ville"
            self.cities[name] = {"population": rng.randrange(50_000, 9_000_000, 1000),
                                 "country": rng.choice(_COUNTRIES)}
        self.companies: Dict[str, Dict[str, Any]] = {}
        city_names = sorted(self.cities)
        while len(self.companies) < n_companies:
            name = f"{rng.choice(_SYL_A)}{rng.choice(_SYL_B)} {rng.choice(_SUFFIX)}"
            if name in self.companies:
                continue
            self.companies[name] = {
                "hq_city": rng.choice(city_names),
                "revenue": rng.randrange(120, 9000, 5),
                "founded": rng.randrange(1950, 2018),
                "ceo": f"{rng.choice(_FIRST)} {rng.choice(_LAST)}",
            }
        self.docs: Dict[str, Doc] = {}
        self.distractor_of: Dict[str, str] = {}      # company -> distractor doc id
        self.doc_of: Dict[str, str] = {}             # entity -> doc id
        n = 0

        def add(title: str, text: str, kind: str) -> str:
            nonlocal n
            n += 1
            did = f"D{n:04d}"
            self.docs[did] = Doc(did, title, text, kind)
            return did

        for cname in sorted(self.companies):
            c = self.companies[cname]
            self.doc_of[cname] = add(
                cname,
                f"{cname} is a technology company headquartered in {c['hq_city']}. Founded in {c['founded']}, "
                f"it reported annual revenue of {c['revenue']} million dollars last year. "
                f"Its chief executive is {c['ceo']}.", "company")
        for city in sorted(self.cities):
            c = self.cities[city]
            self.doc_of[city] = add(
                city, f"{city} is a city in {c['country']} with a population of {c['population']} people. "
                      f"It hosts several technology companies.", "city")
        # similar-name distractors with different numbers (retrieval faults pick these)
        for cname in sorted(self.companies)[: n_companies // 2 + 4]:
            rev2 = self.companies[cname]["revenue"] + rng.choice([-1, 1]) * rng.randrange(40, 400, 5)
            did = add(f"{cname} Holdings",
                      f"{cname} Holdings is an investment group based in {rng.choice(city_names)}. "
                      f"It reported annual revenue of {max(rev2, 10)} million dollars.", "distractor")
            self.distractor_of[cname] = did
        for i in range(10):
            a, b = rng.sample(sorted(self.companies), 2)
            add(f"Industry note {i+1}", f"Analysts compared {a} with {b} this quarter. "
                                         f"No financial figures were disclosed.", "news")
        self._build_index()

    # ---- BM25 -----------------------------------------------------------
    def _build_index(self) -> None:
        self._tf = {d: Counter(_tok(doc.title + " " + doc.text)) for d, doc in self.docs.items()}
        self._len = {d: sum(c.values()) for d, c in self._tf.items()}
        self._avg = sum(self._len.values()) / len(self._len)
        df = Counter()
        for c in self._tf.values():
            df.update(c.keys())
        n = len(self.docs)
        self._idf = {t: math.log(1 + (n - v + 0.5) / (v + 0.5)) for t, v in df.items()}

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        q = _tok(query)
        scores = []
        for d in self.docs:
            s = 0.0
            for t in q:
                f = self._tf[d].get(t, 0)
                if f:
                    s += self._idf.get(t, 0.0) * f * 2.2 / (f + 1.2 * (0.25 + 0.75 * self._len[d] / self._avg))
            if s > 0:
                scores.append((s, d))
        scores.sort(key=lambda x: (-x[0], x[1]))
        return [{"doc_id": d, "title": self.docs[d].title, "score": round(s, 3)} for s, d in scores[:k]]

    def read_doc(self, doc_id: str) -> str:
        if doc_id not in self.docs:
            raise KeyError(f"no such document: {doc_id}")
        return self.docs[doc_id].text

    def with_overrides(self, overrides: Optional[Dict[str, str]]) -> "World":
        """Environment snapshot in which some documents have different text (index is shared)."""
        if not overrides:
            return self
        import copy
        w = copy.copy(self)
        w.docs = dict(self.docs)
        for did, text in overrides.items():
            d = self.docs[did]
            w.docs[did] = Doc(d.doc_id, d.title, text, d.kind)
        return w

    # ---- deterministic tool dispatch (used by the agent AND by replay) --------
    def call(self, tool: str, args: Dict[str, Any]) -> Any:
        if tool == "search":
            return self.search(args["query"], int(args.get("k", 5)))
        if tool == "read_doc":
            return self.read_doc(args["doc_id"])
        if tool == "calc":
            vals = [float(v) for v in args["values"]]
            op = args["op"]
            if op == "sum":
                return int(sum(vals))
            if op == "argmax":
                return int(max(range(len(vals)), key=lambda i: vals[i]))
            raise ValueError(f"unknown calc op {op}")
        raise ValueError(f"unknown tool {tool}")

    # ---- tasks ------------------------------------------------------------
    def make_tasks(self, n_per_type: int, seed: int = 0) -> List[Dict[str, Any]]:
        rng = random.Random(f"tasks-{self.seed}-{seed}")
        comps = sorted(self.companies)
        tasks: List[Dict[str, Any]] = []
        for i in range(n_per_type):
            c = rng.choice(comps)
            tasks.append(self._task("lookup", i, f"What was the annual revenue, in millions of dollars, of {c}?",
                                    self.companies[c]["revenue"], [("lookup", c, "revenue")], [self.doc_of[c]]))
        for i in range(n_per_type):
            c = rng.choice(comps)
            city = self.companies[c]["hq_city"]
            tasks.append(self._task("hop2", i, f"How many people live in the city where {c} is headquartered?",
                                    self.cities[city]["population"],
                                    [("lookup", c, "hq_city"), ("lookup", city, "population")],
                                    [self.doc_of[c], self.doc_of[city]]))
        for i in range(n_per_type):
            a, b = rng.sample(comps, 2)
            tasks.append(self._task("sum", i, f"What is the combined annual revenue, in millions of dollars, of {a} and {b}?",
                                    self.companies[a]["revenue"] + self.companies[b]["revenue"],
                                    [("lookup", a, "revenue"), ("lookup", b, "revenue"), ("compute", "sum", "")],
                                    [self.doc_of[a], self.doc_of[b]]))
        for i in range(n_per_type):
            while True:
                a, b = rng.sample(comps, 2)
                if self.companies[a]["revenue"] != self.companies[b]["revenue"]:
                    break
            win = a if self.companies[a]["revenue"] > self.companies[b]["revenue"] else b
            tasks.append(self._task("compare", i, f"Which company had higher annual revenue, {a} or {b}?",
                                    win, [("lookup", a, "revenue"), ("lookup", b, "revenue"), ("compute", "argmax", "")],
                                    [self.doc_of[a], self.doc_of[b]], entities=[a, b]))
        return tasks

    @staticmethod
    def _task(kind: str, i: int, q: str, gold: Any, plan: List[Tuple[str, str, str]], gold_docs: List[str],
              entities: Optional[List[str]] = None) -> Dict[str, Any]:
        return {"task_id": f"{kind}-{i:03d}", "type": kind, "question": q, "gold": gold,
                "plan": [list(p) for p in plan], "gold_docs": gold_docs, "entities": entities}


# attribute -> regex used by the rule-based reader to pull a value out of a document
ATTR_PATTERNS = {
    "revenue": re.compile(r"revenue of (\d+) million"),
    "hq_city": re.compile(r"headquartered in ([A-Z][A-Za-z]+)"),
    "population": re.compile(r"population of (\d+)"),
}
