#!/usr/bin/env python3
"""Trace-conditioned replay + context instrumentation for the per-turn-cap paper.

Re-executes each episode's RECORDED CodeAct code blocks against the real Opaque
Knapsack environment (no LLM), faithfully reproducing persistent/stateless
interpreter semantics and the per-turn tool-call cap. Because the replay
reproduces the benchmark's own scores, it lets us decompose the quality outcome:

  #2 novel-acquisition throughput: per call, first-time inspect vs cached replay;
     cumulative unique items and replay calls by turn (the replay ceiling).
  #3 subset optimum: OPT over the item set the policy actually inspected (Q),
     separating coverage (OPT(Q)/OPT) from decision (achieved/OPT(Q)).
  #4 context behavior: per-turn input tokens, ceiling proximity, overflow errors
     (read directly from the trace; no replay needed).

Run: uv run python scripts/replay_analysis.py
"""
import asyncio
import glob
import json
import os
import re
import statistics

from codeact_runtime.codeact.agent import FinishTool
from codeact_runtime.codeact.interpreter import AsyncInterpreter
from codeact_runtime.families.knapsack import KnapsackEnv, _solve_01_knapsack

TASKS = "experiments/cap_sweep/knapsack/task_defs/tasks/knapsack"
CAP_RE = re.compile(r"total weight\):\s*(\d+)")
WINDOW, OUT_CAP = 40960, 8192
INPUT_CEIL = WINDOW - OUT_CAP  # effective per-turn input ceiling given output reservation


def _load(path):
    with open(path) as fh:
        return json.load(fh)


def _task_for(trace_path):
    """Map a trace file to its task JSON by index, asserting the goal capacity matches."""
    idx = int(os.path.basename(trace_path).split("-")[-1].split(".")[0])
    tj = f"{TASKS}/knapsack-{idx:010d}.json"
    if not os.path.exists(tj):
        return None
    goal = str((_load(trace_path).get("summary") or {}).get("task") or "")
    tk = _load(tj)
    m = CAP_RE.search(goal)
    if m and int(m.group(1)) != tk["public"]["capacity"]:
        return None
    return tk


def _blocks(trace_path):
    return [
        (e.get("data") or {}).get("code")
        for e in _load(trace_path).get("events", [])
        if e.get("type") == "StepEvent"
    ]


async def _replay_one(trace_path, persistent, cap, carryover=False):
    tk = _task_for(trace_path)
    if tk is None:
        return None
    env = KnapsackEnv.from_task(tk)
    turn = [0]
    calls = []  # (turn, item_id, was_cached)
    orig = env.inspect_item_json

    async def wrapped(item_id):
        calls.append((turn[0], item_id, item_id in env._inspect_cache))
        return await orig(item_id)

    setattr(env, "inspect_item_json", wrapped)
    # carryover replays the cap-boundary-carryover cell under its own semantics: the
    # workspace of a truncated turn is the one the next block actually ran against
    interp = AsyncInterpreter(
        persistent_state=persistent, max_tool_calls=cap, carryover_on_cap=carryover
    )
    for tool in env.get_tools() + [FinishTool()]:
        interp.register_tool(tool.name, tool)

    blocks = _blocks(trace_path)
    for i, code in enumerate(blocks):
        turn[0] = i
        if code:
            try:
                await interp.run_code(code)
            except Exception:
                pass

    q = list(env._inspect_cache.keys())
    allowed = [
        (iid, env.items[iid].weight, env.items[iid].value)
        for iid in q
        if env.items[iid].cls in env.allowed_classes
    ]
    opt_q = _solve_01_knapsack(allowed, env.capacity)[0] if allowed else 0
    opt_global = tk["reference"]["optimal_value"]
    seen, cum_u, cum_r, u, r = set(), {}, {}, 0, 0
    for ti, iid, cached in calls:
        if cached or iid in seen:
            r += 1
        else:
            u += 1
            seen.add(iid)
        cum_u[ti], cum_r[ti] = u, r
    return dict(
        nturns=len(blocks), ncalls=len(calls), unique=len(seen), replay=r,
        agent_val=env.total_value, opt_q=opt_q, opt_global=opt_global,
        cum_u=cum_u, cum_r=cum_r,
    )


async def _run_cell(pattern, persistent, cap, label):
    eps = [await _replay_one(t, persistent, cap) for t in sorted(glob.glob(pattern, recursive=True))]
    eps = [e for e in eps if e]
    if not eps:
        print(f"  {label}: NO DATA")
        return

    def mean(k):
        return statistics.mean(e[k] for e in eps)

    cov = statistics.mean(e["opt_q"] / e["opt_global"] for e in eps)
    ach = statistics.mean(e["agent_val"] / e["opt_global"] for e in eps)
    dec = statistics.mean(e["agent_val"] / e["opt_q"] for e in eps if e["opt_q"] > 0)
    print(f"\n  {label} (n={len(eps)}):")
    print(
        f"    calls/ep={mean('ncalls'):.1f} unique/ep={mean('unique'):.1f} "
        f"replay/ep={mean('replay'):.1f} replay-frac={mean('replay') / max(mean('ncalls'), 1e-9):.2f}"
    )
    print(f"    #3 achieved/OPT={ach:.3f}  coverage OPT(Q)/OPT={cov:.3f}  decision achieved/OPT(Q)={dec:.3f}")
    print("    turn:  unique(cum)  replay(cum)")
    for ti in (0, 2, 5, 10, 20, 30, 39):
        us = [e["cum_u"][ti] for e in eps if ti in e["cum_u"]]
        rs = [e["cum_r"][ti] for e in eps if ti in e["cum_r"]]
        if us:
            print(f"    t{ti:>2}: {statistics.mean(us):>7.1f}    {statistics.mean(rs):>7.1f}  (n={len(us)})")


def context_report(pattern, label):
    eps = []
    for t in glob.glob(pattern, recursive=True):
        evs = _load(t).get("events", [])
        pt = [
            e["data"]["prompt_tokens"]
            for e in evs
            if e.get("type") == "ModelCallEvent" and (e.get("data") or {}).get("prompt_tokens") is not None
        ]
        if not pt:
            continue
        ovf = sum(
            1 for e in evs
            if e.get("type") == "ErrorEvent" and "ContextWindow" in str((e.get("data") or {}).get("error") or "")
        )
        eps.append(dict(maxpt=max(pt), n_over_30k=sum(1 for x in pt if x > 30000), ovf=ovf))
    if not eps:
        print(f"  {label}: NO DATA")
        return
    maxpts = sorted(e["maxpt"] for e in eps)
    print(
        f"  {label:22} n={len(eps):>2} | max-input median={int(statistics.median(maxpts))} "
        f"max={maxpts[-1]} | eps w/ turn>30k={sum(1 for e in eps if e['n_over_30k'])} "
        f"| eps w/ CtxOverflow={sum(1 for e in eps if e['ovf'])}"
    )


async def _main():
    print("=== #2/#3 trace-conditioned replay (novel/replay throughput + subset optimum) ===")
    for pat, persist, cap, label in (
        ("experiments/cap_sweep/knapsack/qwen3_8b/main/PS_cap25/**/knapsack-knapsack-*.trace.json", False, 25, "PS@25 (mismatch)"),
        ("experiments/cap_sweep/knapsack/qwen3_8b/main/PP_cap25/**/knapsack-knapsack-*.trace.json", True, 25, "PP@25 (matched)"),
        ("experiments/cap_sweep/knapsack/qwen3_8b/dense/persistent_cap40/**/knapsack-knapsack-*.trace.json", False, 40, "PS@40 (near-threshold)"),
    ):
        await _run_cell(pat, persist, cap, label)
    print(f"\n=== #4 context behavior (window={WINDOW}, out_cap={OUT_CAP}, input_ceil={INPUT_CEIL}) ===")
    for cell in ("PP", "PS", "SP", "SS"):
        context_report(f"experiments/cap_sweep/knapsack/qwen3_8b/main/{cell}_cap25/**/knapsack-knapsack-*.trace.json", f"{cell}@25 T40")
    for name in ("PS_cap25_T80", "PS_cap25_T160", "PP_cap25_T160"):
        context_report(f"experiments/cap_sweep/knapsack/qwen3_8b/tmax/{name}/**/knapsack-knapsack-*.trace.json", name)


if __name__ == "__main__":
    asyncio.run(_main())
