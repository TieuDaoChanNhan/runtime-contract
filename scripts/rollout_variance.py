#!/usr/bin/env python3
"""Paired inference across knapsack 2x2 ARMS: rollouts of one model, or different bases.

The paper's caveat that "rollout variability is non-negligible" rests on one repeat of one
cell. A full second rollout of all eight cells (same adapters, same 25 instances, same
decoding configuration, no seed) turns that into an estimate, and lets every published
estimand be reported twice.

Two kinds of interval are produced, and they answer different questions:

  within-rollout   the estimand's own paired bootstrap over tasks, computed separately in
                   each rollout -- "what would this rollout have said?"
  test--retest     the paired-over-tasks difference between rollouts, bootstrapped -- "how
                   far does the same estimand move when nothing but decoding changes?"

Both resample TASKS, never cells: every cell covers the same instances, so a task is the
unit that varies. Because the cells share a cohort, the point estimate of any difference of
means equals the mean of paired differences -- the intervals are what need the per-task
files, which is why this script refuses to guess when they are missing.

An "arm" is one full 2x2 x {bind, slack} sweep over the SAME 25 instances: the paper's
Qwen3-8B run, its independent second rollout, or a base-model ablation. Because every arm
covers one cohort, any difference between arms is paired on the instance, and the same
bootstrap answers both questions the paper needs:

  rollout 1 vs rollout 2   how far does an estimand move when only the decode draw changes?
  Qwen vs Mistral/Llama    does the estimand survive a change of base model?

Run: uv run python scripts/rollout_variance.py                        # the two rollouts
     uv run python scripts/rollout_variance.py --arms qwen,mistral,llama31
     uv run python scripts/rollout_variance.py --arms ... --json paper/arms.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scripts.analyze_paper as A  # noqa: E402

FAM = "knapsack"
CAPS = (A.CAP_BIND, A.CAP_SLACK)
CELLS = ("PP", "PS", "SP", "SS")
ARMS = {
    # the paper's main run, and an independent second decode draw of the same adapters
    "r1": "experiments/cap_sweep/knapsack/qwen3_8b/main/{cell}_cap{cap}",
    "r2": "experiments/cap_sweep/knapsack/qwen3_8b/rollout2/{cell}_cap{cap}",
    # base-model ablations: same recipe and instances, different base checkpoint
    "mistral": "experiments/cap_sweep/knapsack/mistral_7b/main/{cell}_cap{cap}",
    "llama31": "experiments/cap_sweep/knapsack/llama31_8b/main/{cell}_cap{cap}",
}
ALIASES: dict[str, str] = {"qwen": "r1", "qwen_r2": "r2"}
DEFAULT_ARMS = ("r1", "r2")
NB = 5000


# ------------------------------------------------------------------ estimands
def _estimands(cells_bind, cells_slack, idx=None):
    """G_P, D, T, M and dM from per-task score lists, optionally on a resample.

    Definitions are analyze_paper's: G_P = PP-PS (the persistent model's runtime gap),
    D = (PP-PS)-(SP-SS) (train x runtime interaction), T = D(bind)-D(slack),
    M = PS-SS (which regime a reset deployment prefers), dM = M(slack)-M(bind).
    """

    def m(cells, cl):
        vals = cells[cl]
        return (
            statistics.mean(vals)
            if idx is None
            else sum(vals[i] for i in idx) / len(idx)
        )

    gp_b = m(cells_bind, "PP") - m(cells_bind, "PS")
    gp_s = m(cells_slack, "PP") - m(cells_slack, "PS")
    d_b = gp_b - (m(cells_bind, "SP") - m(cells_bind, "SS"))
    d_s = gp_s - (m(cells_slack, "SP") - m(cells_slack, "SS"))
    m_b = m(cells_bind, "PS") - m(cells_bind, "SS")
    m_s = m(cells_slack, "PS") - m(cells_slack, "SS")
    return {
        "GP_bind": gp_b,
        "GP_slack": gp_s,
        "A_P": gp_b - gp_s,
        "D_bind": d_b,
        "D_slack": d_s,
        "T": d_b - d_s,
        "M_bind": m_b,
        "M_slack": m_s,
        "dM": m_s - m_b,
    }


KEYS = tuple(_estimands({c: [0.0] for c in CELLS}, {c: [0.0] for c in CELLS}))


# ------------------------------------------------------------------ loading
def _load(root_tpl):
    """cap -> {cell: {task_id: score}}, or None for a rollout with no per-task files."""
    out = {}
    for cap in CAPS:
        cells = {
            cl: A._scores(root_tpl.format(cell=cl, cap=cap), FAM) for cl in CELLS
        }
        if not all(cells.values()):
            return None
        out[cap] = {
            cl: {k.split("-")[-1].split(".")[0]: v for k, v in cells[cl].items()}
            for cl in CELLS
        }
    return out


def _common_tasks(loaded):
    sets = [set(loaded[r][cap][cl]) for r in loaded for cap in CAPS for cl in CELLS]
    return sorted(set.intersection(*sets))


def _aligned(loaded, rollout, cap, tasks):
    return {cl: [loaded[rollout][cap][cl][t] for t in tasks] for cl in CELLS}


# ------------------------------------------------------------------ inference
def _ci(samples):
    samples = sorted(samples)
    return [samples[int(0.025 * len(samples))], samples[int(0.975 * len(samples))]]


def analyse(loaded, tasks, nb=NB):
    rng = random.Random(0)
    n = len(tasks)
    per_rollout = {
        r: _estimands(_aligned(loaded, r, A.CAP_BIND, tasks), _aligned(loaded, r, A.CAP_SLACK, tasks))
        for r in loaded
    }
    ref = next(iter(loaded))
    others = [r for r in loaded if r != ref]
    draws = {r: {k: [] for k in KEYS} for r in loaded}
    deltas = {r: {k: [] for k in KEYS} for r in others}
    for _ in range(nb):
        idx = [rng.randrange(n) for _ in range(n)]
        # ONE task resample, applied to both rollouts: the retest difference is paired on
        # the instance, so an instance that is hard in both must not be resampled twice
        boot = {
            r: _estimands(
                _aligned(loaded, r, A.CAP_BIND, tasks),
                _aligned(loaded, r, A.CAP_SLACK, tasks),
                idx,
            )
            for r in loaded
        }
        for r in loaded:
            for k in KEYS:
                draws[r][k].append(boot[r][k])
        for r in others:
            for k in KEYS:
                deltas[r][k].append(boot[r][k] - boot[ref][k])
    out = {
        "n_tasks": n,
        "rollouts": {
            r: {k: {"est": per_rollout[r][k], "ci": _ci(draws[r][k])} for k in KEYS}
            for r in loaded
        },
    }
    out["reference"] = ref
    out["deltas"] = {
        r: {
            k: {"delta": per_rollout[r][k] - per_rollout[ref][k], "ci": _ci(deltas[r][k])}
            for k in KEYS
        }
        for r in others
    }
    return out


def cell_retest(loaded, tasks, nb=NB, ref=None, other=None):
    """Per-cell paired difference between two arms, with its CI and the SD of the per-task
    differences -- the cell means alone hide whether a cell moved uniformly or on a few
    instances."""
    rng = random.Random(1)
    ref_arm: str = ref if ref is not None else next(iter(loaded))
    other_arm: str = (
        other if other is not None else [r for r in loaded if r != ref_arm][0]
    )
    rows = {}
    for cap in CAPS:
        for cl in CELLS:
            a = [loaded[ref_arm][cap][cl][t] for t in tasks]
            b = [loaded[other_arm][cap][cl][t] for t in tasks]
            diffs = [y - x for x, y in zip(a, b)]
            boots = []
            for _ in range(nb):
                idx = [rng.randrange(len(diffs)) for _ in range(len(diffs))]
                boots.append(sum(diffs[i] for i in idx) / len(idx))
            rows[f"{cl}_{cap}"] = {
                "r1": statistics.mean(a),
                "r2": statistics.mean(b),
                "delta": statistics.mean(diffs),
                "ci": _ci(boots),
                "sd_of_task_diffs": statistics.stdev(diffs) if len(diffs) > 1 else 0.0,
            }
    return rows


# ------------------------------------------------------------------ report
def _f(x, n=3):
    return f"{x:+.{n}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    ap.add_argument("--resamples", type=int, default=NB)
    ap.add_argument(
        "--arms",
        default=",".join(DEFAULT_ARMS),
        help="comma-separated arms; every estimand is differenced against the FIRST. "
        f"known: {', '.join(sorted(set(ARMS) | set(ALIASES)))}",
    )
    args = ap.parse_args()

    names = [a.strip() for a in args.arms.split(",") if a.strip()]
    wanted: list[str] = [ALIASES.get(a) or a for a in names]
    unknown = [a for a in wanted if a not in ARMS]
    if unknown:
        raise SystemExit(f"unknown arms: {unknown}; known: {sorted(ARMS)}")

    loaded = {}
    missing = []
    for name in wanted:
        got = _load(ARMS[name])
        if got is None:
            missing.append((name, ARMS[name]))
        else:
            loaded[name] = got
    if not loaded:
        raise SystemExit(
            "no arm has per-task results; unpack the eval-data mirror into "
            "experiments/cap_sweep/ first (see the Released artifacts section of "
            "the README)."
        )
    if missing:
        print(
            "NOTE: no per-task results for: "
            + ", ".join(f"{n} ({t.format(cell='PP', cap=25)}/...)" for n, t in missing)
        )
        print(
            "      Cell summaries alone fix every point estimate but no interval: a\n"
            "      bootstrap needs the per-task scores. Add the raw tree (results/*.json)\n"
            "      to the reproduction archive and re-run this script.\n"
        )

    tasks = _common_tasks(loaded)
    res = analyse(loaded, tasks, args.resamples)
    print(f"=== knapsack 2x2 estimands, paired over {res['n_tasks']} tasks ===")
    ref = res["reference"]
    head = f"  {'estimand':9}" + "".join(f"{r:>26}" for r in loaded)
    for r in res["deltas"]:
        head += f"{f'{r}-{ref}':>26}"
    print(head)
    for k in KEYS:
        line = f"  {k:9}"
        for r in loaded:
            e = res["rollouts"][r][k]
            line += f"{_f(e['est']):>10} [{_f(e['ci'][0])},{_f(e['ci'][1])}]"
        for r, d in res["deltas"].items():
            line += f"{_f(d[k]['delta']):>10} [{_f(d[k]['ci'][0])},{_f(d[k]['ci'][1])}]"
        print(line)

    payload = {"estimands": res}
    if len(loaded) == 2:
        cells = cell_retest(loaded, tasks, args.resamples)
        payload["cells"] = cells
        print(f"\n=== per-cell paired difference ({[r for r in loaded][1]} - {ref}) ===")
        print(f"  {'cell':9} {'r1':>7} {'r2':>7} {'delta':>8} {'95% CI':>20} {'SD(task diffs)':>15}")
        for name, row in cells.items():
            print(
                f"  {name:9} {row['r1']:>7.3f} {row['r2']:>7.3f} {_f(row['delta']):>8} "
                f"[{_f(row['ci'][0])},{_f(row['ci'][1])}]{'':>3} {row['sd_of_task_diffs']:>15.3f}"
            )
        mags = [abs(r["delta"]) for r in cells.values()]
        print(
            f"\n  mean |delta| over {len(mags)} cells = {statistics.mean(mags):.3f}, "
            f"max = {max(mags):.3f}"
        )

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(payload, fh, indent=1, sort_keys=True, allow_nan=False)
        print(f"\n  wrote {args.json}")


if __name__ == "__main__":
    main()
