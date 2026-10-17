"""Command line interface:  python -m tracelens <command> ..."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

from .agent import Agent, FAULT_KINDS, Fault, applicable
from .eval import whowhen
from .eval.bench import build_bench, eval_set
from .eval.report import render_viewer
from .eval.runs import run_phase1, run_phase2
from .llm import ChatPolicy, ScriptedPolicy, make_completer
from .schema import read_jsonl, write_jsonl
from .world import World

WORLD_SEED = 7


def _bench(args):
    spec = getattr(args, "policy", "none")
    name = "bench.jsonl" if spec in (None, "none") else "bench_" + "".join(c if c.isalnum() else "_" for c in spec) + ".jsonl"
    path = os.path.join(args.out, name)
    if os.path.exists(path) and not args.rebuild:
        return read_jsonl(path)
    os.makedirs(args.out, exist_ok=True)
    comp = make_completer(spec)
    policy = ChatPolicy(comp) if comp else None
    prog = (lambda d, n: print(f"  building benchmark: task {d}/{n}", end="\r", flush=True)) if comp else None
    traces = build_bench(WORLD_SEED, args.tasks_per_type, policy=policy, progress=prog)
    write_jsonl(traces, path)
    if comp:
        print(f"\nmodel usage: {comp.usage()}  parse fallbacks: {policy.fallbacks}")
    return traces


def cmd_build(args):
    args.rebuild = True
    t = _bench(args)
    print(f"wrote {len(t)} traces to {args.out} ({len(eval_set(t))} usable for attribution)")


def cmd_phase1(args):
    args.rebuild = False
    t = _bench(args)
    r = run_phase1(t, args.out, make_completer(args.judge))
    print(json.dumps(r["benchmark"], indent=1)[:600])
    print("baselines (step acc):", {x["method"]: x["step_acc"] for x in r["baselines_full_view"]})
    print("wrote", os.path.join(args.out, "phase1_results.md"))


def cmd_phase2(args):
    args.rebuild = False
    t = _bench(args)
    r = run_phase2(t, args.out, World(WORLD_SEED), make_completer(args.judge))
    from .eval.gaptable import write_gap_table
    print("wrote", write_gap_table(args.out))
    v = next(x for x in r["gap1_attribution"] if x["method"] == "evidence_verifier")
    print("evidence_verifier step acc:", v["step_acc"], "by kind:", {k: x["acc"] for k, x in v["by_kind"].items()})
    print("wrote", os.path.join(args.out, "phase2_results.md"))


def cmd_viewer(args):
    args.rebuild = False
    traces = _bench(args)
    pick, seen = [], set()
    for t in traces:
        k = (t.labels.get("fault") or {}).get("kind", "clean")
        if k not in seen and (k == "clean" or t.labels["fault"]["effective"]) and len(t.events) >= 8:
            seen.add(k)
            pick.append(t)
    path = render_viewer(pick, World(WORLD_SEED), os.path.join(args.out, "viewer.html"))
    print("open this file in a browser:", os.path.abspath(path))


def cmd_run_agent(args):
    comp = make_completer(args.policy)
    policy = ChatPolicy(comp) if comp else ScriptedPolicy()
    world = World(WORLD_SEED)
    agent = Agent(world, policy)
    tasks = world.make_tasks(max(1, args.n // 4 + 1))
    traces = []
    for t in tasks[: args.n]:
        fault = None
        if args.fault != "none":
            if not applicable(args.fault, t):
                continue
            fault = Fault(args.fault, 0)
        traces.append(agent.run(t, fault))
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"agent_runs_{policy.name.replace(':', '_').replace('[', '_').replace(']', '')}.jsonl")
    write_jsonl(traces, path)
    ok = sum(t.outcome["success"] for t in traces)
    print(f"{len(traces)} runs, {ok} correct; policy={policy.name}; fallbacks={getattr(policy, 'fallbacks', 0)}; wrote {path}")


def cmd_whowhen(args):
    if args.inspect:
        print(whowhen.inspect(args.path))
        return
    r = whowhen.evaluate(args.path, args.out, make_completer(args.judge), args.limit)
    for row in r["rows"]:
        print(f"{row['method']:22s} step_acc={row['step_acc']:.3f} hit@3={row['hit@3']:.3f} actor_acc={row['actor_acc']:.3f} (n={row['n']})")
    print("wrote", os.path.join(args.out, "whowhen_results.md"))


def cmd_all(args):
    t0 = time.time()
    args.rebuild = True
    traces = _bench(args)
    print(f"[1/4] benchmark: {len(traces)} traces")
    cmd_phase1(args)
    print("[2/4] phase 1 done")
    cmd_phase2(args)
    print("[3/4] phase 2 done")
    cmd_viewer(args)
    print(f"[4/4] viewer done in {time.time() - t0:.1f}s")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="tracelens", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, judge=True):
        p.add_argument("--out", default="results", help="output folder (default: results)")
        p.add_argument("--tasks-per-type", type=int, default=12)
        p.add_argument("--policy", default="none", help="agent model that generates the traces: none (scripted, offline) | ollama:<model> | openai:<model>")
        if judge:
            p.add_argument("--judge", default="none", help="none | ollama:<model> | openai:<model>  (LLM-judge baselines)")

    for name, fn, h in [("build-bench", cmd_build, "generate the synthetic benchmark"),
                        ("phase1", cmd_phase1, "baseline results"), ("phase2", cmd_phase2, "gap-wise experiments + ablations"),
                        ("viewer", cmd_viewer, "write results/viewer.html"), ("all", cmd_all, "everything above")]:
        p = sub.add_parser(name, help=h)
        common(p, judge=name in ("phase1", "phase2", "all"))
        p.set_defaults(fn=fn)
    p = sub.add_parser("run-agent", help="run the instrumented agent (optionally with a real LLM)")
    p.add_argument("--policy", default="none", help="none (scripted) | ollama:<model> | openai:<model>")
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--fault", default="none", choices=("none",) + FAULT_KINDS)
    p.add_argument("--out", default="results")
    p.set_defaults(fn=cmd_run_agent)
    p = sub.add_parser("whowhen", help="evaluate on the Who&When base-paper dataset")
    p.add_argument("--path", required=True, help="folder or file with the downloaded dataset")
    p.add_argument("--inspect", action="store_true", help="only print dataset structure")
    p.add_argument("--judge", default="none")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", default="results")
    p.set_defaults(fn=cmd_whowhen)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
