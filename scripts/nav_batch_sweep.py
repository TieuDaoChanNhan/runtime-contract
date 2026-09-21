#!/usr/bin/env python3
"""Preselect the navigation intervention's batch size from trace geometry alone.

Design rule for the reconstruction-bandwidth intervention: the reduced
interface must put a substantial share of action blocks above the binding cap while
leaving essentially none above the slack cap, so the two anchors straddle the
manipulation instead of both binding. Picking b by intuition is how you end up with an
arm whose "slack" cell is also truncated.

So this re-costs the RECORDED slack-cap behaviour under each candidate batch size: an
action block that issued c neighbors() calls covering `nodes` nodes would, at batch b,
issue ceil(nodes / b) calls instead. It reports, per candidate,

    E_25(b) = Pr(K_t(b) > 25)      exposure at the binding cap
    E_80(b) = Pr(K_t(b) > 80)      leakage at the slack cap  (want ~0)
    O_25(b) = E[(K_t(b) - 25)+]    tail mass beyond the binding cap

This is a DESIGN calculation, not evidence: it holds the recorded code fixed, and an
agent facing the narrower interface would write different code. Its only job is to choose
b before any outcome data exists.

Run: uv run python scripts/nav_batch_sweep.py
"""

import asyncio
import math
import statistics
from collections import Counter

from scripts.mechanism_traces import (  # type: ignore[reportMissingImports]
    CAP_BIND,
    CAP_SLACK,
    _nav_nodes,
    _traces,
    replay,
)

CANDIDATES = (1, 2, 3, 4, 5, 10, 50)
# a candidate is usable only if it exposes enough blocks at the binding cap to move an
# n=16 estimate, while leaving the slack cap genuinely slack
MIN_EXPOSURE_25 = 0.30
MAX_LEAKAGE_80 = 0.0


async def _turn_shapes(cell, persistent):
    """(executed calls, neighbors calls, nodes queried) per uncensored slack-cap block."""
    shapes = []
    for trace in _traces("navigation", cell, CAP_SLACK):
        ep = await replay("navigation", trace, persistent, CAP_SLACK)
        if ep is None:
            # replay() returns None when a trace cannot be paired with its task JSON; the
            # rest of the sweep skips such episodes too, and subscripting here is what
            # pyright flags
            continue
        calls, nodes = Counter(), Counter()
        for rec in ep["calls"]:
            if rec["tool"] == "neighbors" and rec.get("ok") is not False:
                calls[rec["turn"]] += 1
                nodes[rec["turn"]] += len(_nav_nodes(rec))
        for i, turn in enumerate(ep["turns"]):
            # a truncated block only recorded its prefix, so re-costing it would bound an
            # already-bounded number
            if turn["executed"] and not turn["caphit"]:
                shapes.append((turn["k"], calls[i], nodes[i]))
    return shapes


def _recost(shapes, b):
    return [k - c + math.ceil(n / b) if c else k for (k, c, n) in shapes]


async def main():
    shapes = {cell: await _turn_shapes(cell, cell[1] == "P") for cell in ("PS", "PP")}
    print("=== navigation batch-size sweep (design calculation, no outcome data) ===")
    print(
        f"  {'cell':6} {'b':>3} {'blocks':>7} {'mean K':>7} {'max K':>6} "
        f"{'E_25':>6} {'E_80':>6} {'O_25':>6}"
    )
    table = {}
    for cell in ("PS", "PP"):
        for b in CANDIDATES:
            ks = _recost(shapes[cell], b)
            row = {
                "blocks": len(ks),
                "mean_K": statistics.mean(ks),
                "max_K": max(ks),
                "E25": sum(1 for k in ks if k > CAP_BIND) / len(ks),
                "E80": sum(1 for k in ks if k > CAP_SLACK) / len(ks),
                "O25": statistics.mean(max(0, k - CAP_BIND) for k in ks),
            }
            table[(cell, b)] = row
            print(
                f"  {cell:6} {b:>3} {row['blocks']:>7} {row['mean_K']:>7.1f} "
                f"{row['max_K']:>6} {row['E25']:>6.2f} {row['E80']:>6.2f} {row['O25']:>6.1f}"
            )
        print()
    print(
        f"  selection on the P->S arm (it carries the effect): need E_25 >= {MIN_EXPOSURE_25:.2f} "
        f"and E_80 <= {MAX_LEAKAGE_80:.2f}"
    )
    chosen = []
    for b in CANDIDATES:
        r = table[("PS", b)]
        ok = r["E25"] >= MIN_EXPOSURE_25 and r["E80"] <= MAX_LEAKAGE_80
        why = (
            "clean window"
            if ok
            else (
                "leaks past the slack cap"
                if r["E80"] > MAX_LEAKAGE_80
                else "too little exposure"
            )
        )
        print(
            f"    b={b:<3} E_25={r['E25']:.2f} E_80={r['E80']:.2f} maxK={r['max_K']:<4} -> {why}"
        )
        if ok:
            chosen.append(b)
    print(
        f"\n  usable: {chosen or 'NONE -- do not run the intervention'}"
        + (
            f"; prefer b={max(chosen)} (widest interface that still exposes)"
            if chosen
            else ""
        )
    )
    return table


if __name__ == "__main__":
    asyncio.run(main())
