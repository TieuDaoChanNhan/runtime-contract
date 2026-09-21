#!/usr/bin/env python3
"""Derive a reduced-bandwidth navigation task set for the Sec. 5 intervention.

The intervention must change ONE thing: how many nodes a single `neighbors()` call may
query. So the tasks are not regenerated -- they are copied from the evaluated batched set
with `public.neighbors_batch_max` set to the chosen batch size, keeping the latent graph,
the locks, keys and traps, the probe and move budgets, the scoring, and the task IDs
byte-identical. That is what licenses the paired difference-in-differences

    D_batch = [G_P^reduced(25) - G_P^reduced(80)] - [G_P^batched(25) - G_P^batched(80)]

against the already-archived batched cells.

The batch size is PRESELECTED from trace geometry alone (scripts/nav_batch_sweep.py),
before any outcome data: it must put a substantial share of slack-policy action blocks
above c=25 while leaving essentially none above c=80, so the two anchors straddle the
intervention rather than both binding.

Run: uv run python scripts/make_nav_unbatched_tasks.py --batch 2 [--check]
"""

import argparse
import json
import os
import shutil
import sys

SRC = "experiments/cap_sweep/navigation/task_defs"
FAMILY = "navigation"


def dst_for(batch):
    return f"experiments/cap_sweep/navigation/task_defs_batch{batch}"


# fields that must be identical between the two task sets: everything except the knob
IMMUTABLE = ("task_id", "family", "seed", "difficulty", "private", "reference")


def _tasks(root):
    d = f"{root}/tasks/{FAMILY}"
    return (
        sorted(f for f in os.listdir(d) if f.endswith(".json"))
        if os.path.isdir(d)
        else []
    )


def build(BATCH_MAX, DST):
    src_files = _tasks(SRC)
    if not src_files:
        raise SystemExit(
            f"no source tasks under {SRC}/tasks/{FAMILY} -- fetch the archive first"
        )
    os.makedirs(f"{DST}/tasks/{FAMILY}", exist_ok=True)
    for name in src_files:
        task = json.load(open(f"{SRC}/tasks/{FAMILY}/{name}"))
        task["public"]["neighbors_batch_max"] = BATCH_MAX
        task["public"]["notes"] = (
            "Graph topology is queryable via neighbors(nodes), "
            + (
                "ONE node per call. "
                if BATCH_MAX == 1
                else f"up to {BATCH_MAX} nodes per call. "
            )
            + "Hidden node properties (traps, keys, locks) require probe()."
        )
        with open(f"{DST}/tasks/{FAMILY}/{name}", "w") as fh:
            json.dump(task, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
    for meta in ("_cfg.json", "manifest.json"):
        if os.path.exists(f"{SRC}/{meta}"):
            shutil.copy(f"{SRC}/{meta}", f"{DST}/{meta}")
    # record the derivation in the copied config so the variant is self-describing
    cfg_path = f"{DST}/_cfg.json"
    if os.path.exists(cfg_path):
        cfg = json.load(open(cfg_path))
        cfg.setdefault(FAMILY, {})["neighbors_batch_max"] = BATCH_MAX
        cfg["derived_from"] = SRC
        cfg["derivation"] = (
            "copied verbatim; only public.neighbors_batch_max (and its notes line) changed"
        )
        with open(cfg_path, "w") as fh:
            json.dump(cfg, fh, indent=2)
            fh.write("\n")
    print(
        f"wrote {len(src_files)} tasks to {DST}/tasks/{FAMILY} (neighbors_batch_max={BATCH_MAX})"
    )
    return check(BATCH_MAX, DST)


def check(BATCH_MAX, DST):
    """Verify the variant differs from its source in the knob and nothing else."""
    src_files, dst_files = _tasks(SRC), _tasks(DST)
    if src_files != dst_files:
        print(f"MISMATCH: {len(src_files)} source tasks vs {len(dst_files)} derived")
        return 1
    problems = []
    for name in src_files:
        src = json.load(open(f"{SRC}/tasks/{FAMILY}/{name}"))
        dst = json.load(open(f"{DST}/tasks/{FAMILY}/{name}"))
        for field in IMMUTABLE:
            if src.get(field) != dst.get(field):
                problems.append(f"{name}: {field} differs")
        changed = {
            k
            for k in set(src["public"]) | set(dst["public"])
            if src["public"].get(k) != dst["public"].get(k)
        }
        if changed - {"neighbors_batch_max", "notes"}:
            problems.append(f"{name}: unexpected public changes {sorted(changed)}")
        if dst["public"].get("neighbors_batch_max") != BATCH_MAX:
            problems.append(f"{name}: neighbors_batch_max not set")
    if problems:
        print("\n".join(problems[:10]))
        return 1
    print(f"OK: {len(src_files)} tasks differ from {SRC} only in neighbors_batch_max")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--batch", type=int, required=True, help="nodes per neighbors() call"
    )
    ap.add_argument(
        "--check", action="store_true", help="verify an existing derivation"
    )
    args = ap.parse_args()
    dst = dst_for(args.batch)
    sys.exit(check(args.batch, dst) if args.check else build(args.batch, dst))
