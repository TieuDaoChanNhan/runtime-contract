#!/usr/bin/env python3
"""Cap-boundary carryover: the rescue fraction and its pre-specified diagnostic panel.

The design and the analysis plan were fixed BEFORE the cells were run. This script implements that plan and nothing else:

  primary    rho = (Q_ckpt - Q_PS) / (Q_PP - Q_PS), paired over the common task ids, with
             a paired bootstrap CI on the numerator (the quantity carrying the claim) and
             on rho itself (a ratio of two noisy differences -- read its interval loosely)
  secondary  replay fraction; unique items by turn 5; the post-cap prefix and the
             entity-unit re-fetched/held ratio of Table 3; decision realization and
             coverage; NameError rate overall and on turns that carried a checkpoint;
             share of episodes ending on the turn limit; the manipulation check

All behavioural quantities come from the same trace-conditioned replay the paper's other
mechanism numbers use (scripts/mechanism_traces.py), with the carryover cell replayed under
carryover semantics so its recorded errors reproduce.

Run: uv run python scripts/checkpoint_analysis.py [--json paper/checkpoint.json]
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import json
import os
import random
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scripts.analyze_paper as A  # noqa: E402
import scripts.mechanism_traces as M  # noqa: E402
from codeact_runtime.families.knapsack import _solve_01_knapsack  # noqa: E402

ARM = "knapsack_ckpt"
CAP = 25
FAM = "knapsack"
CELLS = ("PS", "PSckpt", "PP")
# The announced-carryover addendum: the announced cell plus a same-batch re-run of the
# silent one, which is what it must be compared against (the two batches ran on different
# serving instances). Absent until those cells exist.
ANN_CELLS = ("PSckpt_rerun", "PSckptann")
RUNS = "experiments/cap_sweep/knapsack/qwen3_8b/checkpoint/{cell}_cap{cap}"
ARCHIVED = "experiments/cap_sweep/knapsack/qwen3_8b/main/{cell}_cap25"
NB = 5000

# the carryover cells live outside the four-cell sweep layout, so the arm is registered
# here rather than in mechanism_traces' FAMILIES (whose members are replayed at both caps)
M.FAMILIES[ARM] = {
    "tasks": f"experiments/cap_sweep/knapsack/task_defs/tasks/{FAM}",
    "runs": RUNS,
    "env": FAM,
}
M.SHORT[ARM] = "ckpt"


# --------------------------------------------------------------------------- scores
def _scores(cells=CELLS):
    return {c: A._scores(RUNS.format(cell=c, cap=CAP), FAM) for c in cells}


def _task_ids():
    """Every knapsack instance the cohort is defined by, as zero-padded index strings."""
    return {
        os.path.basename(f).split("-")[-1].split(".")[0]
        for f in glob.glob(f"experiments/cap_sweep/knapsack/task_defs/tasks/{FAM}/*.json")
    }


def _cohort_problems(scores, expected):
    """Compare each cell's task IDs to the cohort, not just its file count.

    A cell that lost one task and gained a stale result elsewhere has the right count and
    the wrong cohort; `_paired` would then quietly intersect its way to a smaller or
    different set of tasks while this script claims to refuse exactly that."""
    problems = []
    for cell, got in sorted(scores.items()):
        ids = {k.split("-")[-1].split(".")[0] for k in got}
        missing, unexpected = sorted(expected - ids), sorted(ids - expected)
        if missing or unexpected:
            detail = f"{cell}: {len(ids)}/{len(expected)} tasks"
            if missing:
                detail += f", missing {missing[:5]}{' ...' if len(missing) > 5 else ''}"
            if unexpected:
                detail += (
                    f", unexpected {unexpected[:5]}"
                    f"{' ...' if len(unexpected) > 5 else ''}"
                )
            problems.append(detail)
    return problems


def _paired(scores):
    common = sorted(set.intersection(*(set(v) for v in scores.values())))
    return common, {c: [scores[c][k] for k in common] for c in scores}


def _primary(vals, n):
    """Paired bootstrap over task ids: resample tasks, not cells.

    Seeded locally rather than through the global `random` module: this runs inside
    build_paper_numbers.py, where re-seeding the shared stream would silently move every
    bootstrap CI computed after it."""
    rng = random.Random(0)
    d_obs = statistics.mean(vals["PSckpt"]) - statistics.mean(vals["PS"])
    g_obs = statistics.mean(vals["PP"]) - statistics.mean(vals["PS"])
    deltas, rhos = [], []
    for _ in range(NB):
        idx = [rng.randrange(n) for _ in range(n)]
        m = {c: statistics.mean(vals[c][i] for i in idx) for c in CELLS}
        d, g = m["PSckpt"] - m["PS"], m["PP"] - m["PS"]
        deltas.append(d)
        # a resample where the reference gap vanishes carries no information about the
        # SHARE recovered; it is dropped from rho's interval and counted instead
        if abs(g) > 1e-9:
            rhos.append(d / g)
    deltas.sort()
    rhos.sort()
    return {
        "n": n,
        "Q": {c: statistics.mean(vals[c]) for c in CELLS},
        "delta": d_obs,
        "delta_ci": [deltas[int(0.025 * NB)], deltas[int(0.975 * NB)]],
        "gap": g_obs,
        "rho": d_obs / g_obs if abs(g_obs) > 1e-9 else float("nan"),
        "rho_ci": [rhos[int(0.025 * len(rhos))], rhos[int(0.975 * len(rhos))]]
        if rhos
        else None,
        "rho_undefined_resamples": NB - len(rhos),
        "per_task": {
            "better": sum(
                1 for a, b in zip(vals["PSckpt"], vals["PS"]) if a > b + 1e-12
            ),
            "worse": sum(1 for a, b in zip(vals["PSckpt"], vals["PS"]) if a < b - 1e-12),
            "tied": sum(
                1 for a, b in zip(vals["PSckpt"], vals["PS"]) if abs(a - b) <= 1e-12
            ),
        },
    }


def _announced(vals, n, silent_first_batch):
    """Delta_ann = Q_ann - Q_ckpt(re-run), paired over tasks, same batch.

    Also reports how far the re-run of the silent cell sits from its first-batch twin: that
    distance is this run's own drift estimate, and it is the reason the comparison is made
    against the re-run rather than across batches."""
    rng = random.Random(0)
    d_obs = statistics.mean(vals["PSckptann"]) - statistics.mean(vals["PSckpt_rerun"])
    deltas = []
    for _ in range(NB):
        idx = [rng.randrange(n) for _ in range(n)]
        m = {c: statistics.mean(vals[c][i] for i in idx) for c in vals}
        deltas.append(m["PSckptann"] - m["PSckpt_rerun"])
    deltas.sort()
    return {
        "n": n,
        "Q": {c: statistics.mean(v) for c, v in vals.items()},
        "delta": d_obs,
        "delta_ci": [deltas[int(0.025 * NB)], deltas[int(0.975 * NB)]],
        "drift_of_silent_cell": statistics.mean(vals["PSckpt_rerun"]) - silent_first_batch,
        "per_task": {
            "better": sum(
                1
                for a, b in zip(vals["PSckptann"], vals["PSckpt_rerun"])
                if a > b + 1e-12
            ),
            "worse": sum(
                1
                for a, b in zip(vals["PSckptann"], vals["PSckpt_rerun"])
                if a < b - 1e-12
            ),
            "tied": sum(
                1
                for a, b in zip(vals["PSckptann"], vals["PSckpt_rerun"])
                if abs(a - b) <= 1e-12
            ),
        },
    }


# ---------------------------------------------------------------------- behaviour
def _finish_reason(trace_path):
    for ev in reversed(M._load(trace_path).get("events", [])):
        if ev.get("type") == "FinishEvent":
            return (ev.get("data") or {}).get("reason")
    return None


def _episode_panel(ep):
    """One episode's pre-specified secondary quantities."""
    M.classify(ARM, ep["calls"])
    # replay fraction, unique and calls are the paper's knapsack definitions (numbers.json
    # `replay`): the acquisition channel only, i.e. inspect(). Counting every classified
    # call instead would fold in list_items and take_item and stop being comparable.
    # (one difference from numbers.json's `replay`, worth 0.04 items per episode in the
    # archived P->P cell: a call the tool REJECTED acquires nothing, so classify() scores
    # it 'other' where the older knapsack-only replay counted it as an acquisition.)
    inspects = [r for r in ep["calls"] if r["tool"] == "inspect"]
    novel = [r for r in inspects if r["kind"] == "novel"]
    replayed = [r for r in inspects if r["kind"] == "replay"]
    by5 = {M._arg(r, "item_id") for r in inspects if r["turn"] <= 5}

    env, task = ep["env"], ep["task"]
    held = [
        (iid, env.items[iid].weight, env.items[iid].value)
        for iid in env._inspect_cache
        if env.items[iid].cls in env.allowed_classes
    ]
    opt_q = _solve_01_knapsack(held, env.capacity)[0] if held else 0
    opt_global = task["reference"]["optimal_value"]

    turns = ep["turns"]
    executed = [t for t in turns if t["executed"]]
    caphit = [i for i, t in enumerate(turns) if t["caphit"]]
    # a turn ran with a checkpoint iff the previous EXECUTED turn was truncated -- the same
    # rule the runtime applies, so it is reconstructible from the recording alone
    carried = []
    for i in caphit:
        nxt = next((j for j in range(i + 1, len(turns)) if turns[j]["executed"]), None)
        if nxt is not None:
            carried.append(nxt)

    def nameerrs(idxs):
        return sum(1 for i in idxs if "NameError" in (turns[i]["rec_err"] or ""))

    # manipulation check for the announced variant, read from the observation the model
    # actually saw: after a truncated turn, did the banner name the surviving bindings?
    announced = sum(1 for i in caphit if turns[i]["surviving"])

    return {
        "calls": len(inspects),
        "unique": len(novel),
        "replay": len(replayed),
        # the two calls that say what the policy is DOING with its turns: re-deriving the
        # catalogue, or committing to a selection
        "lists": sum(1 for r in ep["calls"] if r["tool"] == "list_items"),
        "takes": sum(1 for r in ep["calls"] if r["tool"] == "take_item"),
        "unique_by5": len(by5),
        "achieved": env.total_value / opt_global,
        "coverage": opt_q / opt_global,
        "decision": (env.total_value / opt_q) if opt_q > 0 else None,
        "nameerr_turns": nameerrs(range(len(turns))),
        "nameerr_carried": nameerrs(carried),
        "carried_turns": len(carried),
        "executed_turns": len(executed),
        "caphit_turns": len(caphit),
        "caphit_rate": len(caphit) / max(len(executed), 1),
        "announced_share": announced / len(caphit) if caphit else None,
        "max_turns_end": _finish_reason(ep["trace"]) == "max_turns",
    }


def _mean(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    if not vals:
        return None
    if isinstance(vals[0], bool):
        return sum(1 for v in vals if v) / len(vals)
    return statistics.mean(vals)


async def _behaviour(cells=CELLS):
    out = {}
    for cell in cells:
        eps = await M.replay_cell(ARM, cell, CAP)
        if not eps:
            continue
        rows = [_episode_panel(e) for e in eps]
        resume = M.resume_stats(ARM, eps)

        def rmean(key):
            vals = [r[key] for r in resume if r.get(key) is not None]
            vals = [float(v) for v in vals if v == v]  # drop NaN
            return statistics.mean(vals) if vals else None

        out[cell] = {
            "n": len(eps),
            "diverged_turns": sum(e["diverged"] for e in eps),
            **{
                k: _mean(rows, k)
                for k in (
                    "calls",
                    "unique",
                    "replay",
                    "lists",
                    "takes",
                    "unique_by5",
                    "achieved",
                    "coverage",
                    "decision",
                    "nameerr_turns",
                    "nameerr_carried",
                    "carried_turns",
                    "caphit_turns",
                    "caphit_rate",
                    "announced_share",
                    "max_turns_end",
                )
            },
            # the paper's cell-level ratio (mean replay over mean calls), not the mean of
            # per-episode ratios: a short episode must not weigh as much as a 700-call one
            "replay_frac": (_mean(rows, "replay") or 0) / max(_mean(rows, "calls") or 0, 1e-9),
            "resume": {
                "n_caphits": len(resume),
                "resume_rate": rmean("resume"),
                "re_enumerates": rmean("re_enumerates"),
                "prefix_calls": rmean("prefix_replay"),
                "refetched": rmean("refetched"),
                "held": rmean("held"),
                "refetch_frac": rmean("refetch_frac"),
            },
        }
    return out


# --------------------------------------------------------------------------- report
def _fmt(x, n=3):
    if x is None:
        return "--"
    if isinstance(x, float) and x != x:
        return "nan"
    return f"{x:.{n}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None, help="also write the tables to this path")
    args = ap.parse_args()

    scores = _scores()
    present = {c: len(v) for c, v in scores.items()}
    if not any(present.values()):
        print(
            "no checkpoint episodes yet: run, for i in 0..3,\n"
            "  SHARD_SPEC=i/4 nohup bash "
            "experiments/cap_sweep/runners/run_checkpoint_shard.sh CKPTi PS PSckpt PP &\n"
            "(and arm watchdog_ckpt.sh alongside it). Cells: "
            + ", ".join(RUNS.format(cell=c, cap=CAP) for c in CELLS)
        )
        return
    expected = _task_ids()
    problems = _cohort_problems(scores, expected)
    if problems:
        raise SystemExit(
            "cells do not cover the task cohort -- refusing to report means over a "
            "truncated or substituted one:\n  " + "\n  ".join(problems)
        )

    common, vals = _paired(scores)
    primary = _primary(vals, len(common))

    # the announced-carryover addendum, if its cells have been run
    ann_scores = _scores(ANN_CELLS)
    ann_ready = any(ann_scores.values()) and not _cohort_problems(ann_scores, expected)
    announced = None
    if ann_ready:
        common_ann, vals_ann = _paired(ann_scores)
        silent_first_batch = statistics.mean(
            scores["PSckpt"][k] for k in common_ann if k in scores["PSckpt"]
        )
        announced = _announced(vals_ann, len(common_ann), silent_first_batch)

    behaviour = asyncio.run(_behaviour(CELLS + (ANN_CELLS if ann_ready else ())))
    archived = {
        c: A._scores(ARCHIVED.format(cell=c), FAM) for c in ("PP", "PS")
    }

    print("=== cap-boundary carryover, knapsack @ c=25 (pre-specified plan) ===")
    print(f"  paired over {primary['n']} tasks\n")
    for c in CELLS:
        print(f"  Q[{c:7}] = {_fmt(primary['Q'][c])}")
    print(
        f"\n  rescue   delta = Q_ckpt - Q_PS = {_fmt(primary['delta'])} "
        f"[{_fmt(primary['delta_ci'][0])}, {_fmt(primary['delta_ci'][1])}]"
    )
    print(f"  reference gap = Q_PP  - Q_PS = {_fmt(primary['gap'])}")
    rho_ci = primary["rho_ci"]
    print(
        f"  rho = {_fmt(primary['rho'], 2)}"
        + (f"  [{_fmt(rho_ci[0], 2)}, {_fmt(rho_ci[1], 2)}]" if rho_ci else "")
    )
    pt = primary["per_task"]
    print(
        f"  per task: {pt['better']} better, {pt['worse']} worse, {pt['tied']} tied\n"
    )

    print("  drift check against the archived (unpaired, unseeded) cells:")
    for c in ("PP", "PS"):
        if archived[c]:
            print(
                f"    {c}: archived {_fmt(statistics.mean(archived[c].values()))} "
                f"vs re-run {_fmt(primary['Q'][c])}"
            )
    print()

    if announced:
        print("=== announced carryover: does TELLING the policy change what it does? ===")
        print(f"  paired over {announced['n']} tasks, one batch\n")
        for c in ANN_CELLS:
            print(f"  Q[{c:13}] = {_fmt(announced['Q'][c])}")
        print(
            f"\n  delta = Q_ann - Q_ckpt(re-run) = {_fmt(announced['delta'])} "
            f"[{_fmt(announced['delta_ci'][0])}, {_fmt(announced['delta_ci'][1])}]"
        )
        pa = announced["per_task"]
        print(
            f"  per task: {pa['better']} better, {pa['worse']} worse, {pa['tied']} tied"
        )
        print(
            f"  drift of the silent cell between batches: "
            f"{_fmt(announced['drift_of_silent_cell'])}\n"
        )

    hdr = (
        f"  {'cell':13} {'replay':>7} {'uniq':>6} {'u@5':>6} {'cover':>6} {'decis':>6} "
        f"{'NErr':>6} {'NErr/c':>7} {'carried':>8} {'cap/turn':>9} {'told':>5} {'maxT':>5}"
    )
    print("=== secondary panel (trace-conditioned replay) ===")
    print(hdr)
    for c in behaviour:
        b = behaviour.get(c)
        if not b:
            continue
        print(
            f"  {c:13} {_fmt(b['replay_frac'], 2):>7} {_fmt(b['unique'], 1):>6} "
            f"{_fmt(b['unique_by5'], 1):>6} {_fmt(b['coverage'], 2):>6} "
            f"{_fmt(b['decision'], 2):>6} {_fmt(b['nameerr_turns'], 1):>6} "
            f"{_fmt(b['nameerr_carried'], 1):>7} {_fmt(b['carried_turns'], 1):>8} "
            f"{_fmt(b['caphit_rate'], 2):>9} {_fmt(b['announced_share'], 2):>5} "
            f"{_fmt(b['max_turns_end'], 2):>5}"
        )
    print(
        "  replay = repeat share of executed calls; u@5 = distinct items inspected by turn 5;\n"
        "  cover = OPT(inspected)/OPT, decis = achieved/OPT(inspected); NErr = NameError turns\n"
        "  per episode, NErr/c the subset on turns that carried a checkpoint; carried = such\n"
        "  turns per episode; cap/turn = share of executed turns the cap truncated; told = share\n"
        "  of truncated turns whose banner named the surviving bindings (the announced\n"
        "  variant's manipulation check); maxT = share of episodes ending on the turn limit.\n"
    )

    print("=== resume vs restart after a cap hit (same units as Table 3) ===")
    print(
        f"  {'cell':13} {'caphits':>8} {'re-deriv':>9} {'prefix':>7} "
        f"{'refetch':>8} {'held':>6} {'frac':>6} {'resume':>7}"
    )
    for c in behaviour:
        r = (behaviour.get(c) or {}).get("resume")
        if not r:
            continue
        print(
            f"  {c:13} {r['n_caphits']:>8} {_fmt(r['re_enumerates'], 2):>9} "
            f"{_fmt(r['prefix_calls'], 1):>7} {_fmt(r['refetched'], 1):>8} "
            f"{_fmt(r['held'], 1):>6} {_fmt(r['refetch_frac'], 2):>6} "
            f"{_fmt(r['resume_rate'], 2):>7}"
        )

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(
                {
                    "_source": "scripts/checkpoint_analysis.py over "
                    "experiments/cap_sweep/knapsack/qwen3_8b/checkpoint (pre-specified analysis plan)",
                    "primary": primary,
                    "announced": announced,
                    "behaviour": behaviour,
                    "archived": {
                        c: statistics.mean(v.values()) for c, v in archived.items() if v
                    },
                    "tasks": common,
                },
                fh,
                indent=1,
                sort_keys=True,
                allow_nan=False,
            )
        print(f"\n  wrote {args.json}")


if __name__ == "__main__":
    main()
