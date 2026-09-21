"""The rescue-fraction estimator of the cap-boundary-carryover test.

The panel itself is validated against the archived cells (it reproduces numbers.json's
replay, progress and resume rows); what needs pinning here is the arithmetic the claim
rests on, which no archived cell exercises.
"""

import scripts.checkpoint_analysis as C


def _vals(ps, ckpt, pp):
    return {"PS": ps, "PSckpt": ckpt, "PP": pp}


def test_rho_is_the_share_of_the_reference_gap_recovered():
    n = 20
    out = C._primary(_vals([0.0] * n, [0.3] * n, [0.6] * n), n)
    assert out["delta"] == 0.3
    assert out["gap"] == 0.6
    assert out["rho"] == 0.5
    assert out["per_task"] == {"better": n, "worse": 0, "tied": 0}


def test_a_null_intervention_reports_zero_and_straddles_zero():
    n = 24
    ps = [i / n for i in range(n)]
    out = C._primary(_vals(ps, list(ps), [1.0] * n), n)
    assert out["delta"] == 0.0
    assert out["rho"] == 0.0
    lo, hi = out["delta_ci"]
    assert lo == hi == 0.0
    assert out["per_task"]["tied"] == n


def test_bootstrap_is_paired_over_tasks():
    """Resampling tasks, not cells: a per-task constant offset has a zero-width interval."""
    n = 16
    ps = [0.1 * i for i in range(n)]
    out = C._primary(_vals(ps, [v + 0.2 for v in ps], [v + 0.4 for v in ps]), n)
    assert abs(out["delta"] - 0.2) < 1e-12
    lo, hi = out["delta_ci"]
    assert abs(hi - lo) < 1e-12
    assert abs(out["rho"] - 0.5) < 1e-12


def test_resamples_with_no_reference_gap_are_reported_not_silently_dropped():
    n = 12
    out = C._primary(_vals([0.5] * n, [0.5] * n, [0.5] * n), n)
    assert out["rho"] != out["rho"]  # nan: no gap to recover a share of
    assert out["rho_undefined_resamples"] == C.NB
    assert out["rho_ci"] is None


def test_cohort_check_catches_a_substituted_task_not_just_a_missing_one():
    """The failure this guards: one task lost, one stale result gained, count unchanged."""
    expected = {"0000000000", "0000000001", "0000000002"}
    ok = {"knapsack-knapsack-0000000000.json": 0.0,
          "knapsack-knapsack-0000000001.json": 0.0,
          "knapsack-knapsack-0000000002.json": 0.0}
    assert C._cohort_problems({"PS": ok}, expected) == []

    swapped = dict(ok)
    del swapped["knapsack-knapsack-0000000002.json"]
    swapped["knapsack-knapsack-0000000099.json"] = 0.0  # same count, wrong cohort
    problems = C._cohort_problems({"PS": swapped}, expected)
    assert len(problems) == 1
    assert "missing ['0000000002']" in problems[0]
    assert "unexpected ['0000000099']" in problems[0]

    short = {k: v for k, v in list(ok.items())[:2]}
    assert "2/3 tasks" in C._cohort_problems({"PS": short}, expected)[0]
