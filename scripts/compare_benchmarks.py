#!/usr/bin/env python3
"""Compare two benchmark result tarballs with Wilcoxon signed-rank test.

Usage:
    uv run python scripts/compare_benchmarks.py <tarball_a> <tarball_b>
    uv run python scripts/compare_benchmarks.py results/easy_lora.tar.gz results/easy_base.tar.gz

Extracts per-task scores from each tarball, pairs by task ID, and runs
a two-sided Wilcoxon signed-rank test.
"""

import json
import sys
import tarfile
from pathlib import Path

import numpy as np
from scipy import stats


def extract_scores(tar_path: str) -> dict[str, float]:
    """Extract {task_id: score} from a benchmark result tarball."""
    scores = {}
    with tarfile.open(tar_path, "r:gz") as tf:
        for member in tf.getmembers():
            name = member.name
            # Match per-task result files, skip trace files
            if (
                "knapsack-knapsack-" in name
                and name.endswith(".json")
                and ".trace.json" not in name
            ):
                task_id = Path(name).stem.split("-")[-1]
                f = tf.extractfile(member)
                if f is None:
                    continue
                data = json.loads(f.read())
                result = data.get("result", {})
                scores[task_id] = result.get("score", 0.0) * 100
    return scores


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <tarball_a> <tarball_b>")
        sys.exit(1)

    path_a, path_b = sys.argv[1], sys.argv[2]
    name_a = Path(path_a).stem.replace("_results", "")
    name_b = Path(path_b).stem.replace("_results", "")

    scores_a = extract_scores(path_a)
    scores_b = extract_scores(path_b)

    common = sorted(set(scores_a) & set(scores_b))
    only_a = set(scores_a) - set(scores_b)
    only_b = set(scores_b) - set(scores_a)

    if not common:
        print("ERROR: No common task IDs between the two benchmarks.")
        sys.exit(1)

    a = np.array([scores_a[k] for k in common])
    b = np.array([scores_b[k] for k in common])
    diff = a - b

    print(f"A: {name_a} ({len(scores_a)} tasks)")
    print(f"B: {name_b} ({len(scores_b)} tasks)")
    print(f"Paired: {len(common)} tasks")
    if only_a:
        print(f"  Only in A: {len(only_a)}")
    if only_b:
        print(f"  Only in B: {len(only_b)}")
    print()

    # Descriptive stats
    print(f"{'':20s} {'Mean':>8s} {'Std':>8s} {'Median':>8s} {'Min':>8s} {'Max':>8s}")
    print(
        f"{'A':20s} {np.mean(a):8.2f} {np.std(a):8.2f} {np.median(a):8.2f} {np.min(a):8.2f} {np.max(a):8.2f}"
    )
    print(
        f"{'B':20s} {np.mean(b):8.2f} {np.std(b):8.2f} {np.median(b):8.2f} {np.min(b):8.2f} {np.max(b):8.2f}"
    )
    print(
        f"{'Diff (A-B)':20s} {np.mean(diff):8.2f} {np.std(diff):8.2f} {np.median(diff):8.2f} {np.min(diff):8.2f} {np.max(diff):8.2f}"
    )
    print()

    # Wilcoxon signed-rank test
    nonzero = diff[diff != 0]
    if len(nonzero) < 2:
        print("Cannot run Wilcoxon: fewer than 2 non-zero differences.")
        return

    W, p = stats.wilcoxon(a, b)

    if p < 0.001:  # type: ignore
        sig = "*** (p<0.001)"
    elif p < 0.01:  # type: ignore
        sig = "**  (p<0.01)"
    elif p < 0.05:  # type: ignore
        sig = "*   (p<0.05)"
    else:
        sig = "ns  (p>=0.05)"

    print("Wilcoxon signed-rank test (two-sided)")
    print(f"  W statistic: {W:.1f}")
    print(f"  p-value:     {p:.4e}")
    print(f"  Significance: {sig}")
    print(f"  Mean diff:   {np.mean(diff):+.2f} pp")
    print()

    # Effect size: matched-pairs rank-biserial correlation
    n = len(nonzero)
    r = 1 - (2 * W) / (n * (n + 1))  # type: ignore
    print(f"  Effect size (rank-biserial r): {r:.3f}")

    if np.mean(diff) > 0:
        print(f"\n  => A ({name_a}) scores higher than B ({name_b})")
    elif np.mean(diff) < 0:
        print(f"\n  => B ({name_b}) scores higher than A ({name_a})")
    else:
        print("\n  => No difference")


if __name__ == "__main__":
    main()
