"""Build the TraceBench-Synth benchmark: clean runs + one injected fault per run."""
from __future__ import annotations

from typing import Callable, Iterable, List, Optional

from ..agent import Agent, FAULT_KINDS, Fault, applicable
from ..llm import ScriptedPolicy
from ..schema import Trace
from ..world import World

_TWO_STEP_KINDS = ("retrieval_miss", "retrieval_miss_transient", "tool_corruption", "source_error", "tool_error")


def build_bench(world_seed: int = 7, tasks_per_type: int = 12, task_seed: int = 0,
                seeds: Iterable[int] = (0,), policy: Optional[ScriptedPolicy] = None,
                progress: Optional[Callable[[int, int], None]] = None) -> List[Trace]:
    world = World(world_seed)
    agent = Agent(world, policy)
    tasks = world.make_tasks(tasks_per_type, task_seed)
    traces: List[Trace] = []
    total = len(tasks) * len(list(seeds))
    done = 0
    for seed in seeds:
        for task in tasks:
            traces.append(agent.run(task, None, seed))
            n_lookups = sum(1 for s in task["plan"] if s[0] == "lookup")
            for kind in FAULT_KINDS:
                if not applicable(kind, task):
                    continue
                steps = (0, 1) if (kind in _TWO_STEP_KINDS and n_lookups >= 2) else (0,)
                for st in steps:
                    t = agent.run(task, Fault(kind, st), seed)
                    traces.append(t)
            done += 1
            if progress:
                progress(done, total)
    for t in traces:
        problems = t.validate()
        if problems:
            raise AssertionError(f"{t.trace_id}: {problems}")
    return traces


def eval_set(traces: Iterable[Trace]) -> List[Trace]:
    """Traces usable for attribution: a fault was injected, it caused a wrong answer, and its location is known."""
    return [t for t in traces if t.labels.get("fault") and t.labels["fault"]["effective"]
            and t.labels["fault"]["critical_event"] is not None]
