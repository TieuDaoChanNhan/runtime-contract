"""The test--retest estimator: paired resampling, and the definitions it inherits."""

import scripts.rollout_variance as R

CELLS = ("PP", "PS", "SP", "SS")


def _cells(pp, ps, sp, ss, n=8):
    return {"PP": [pp] * n, "PS": [ps] * n, "SP": [sp] * n, "SS": [ss] * n}


def test_estimand_definitions():
    bind = _cells(0.8, 0.1, 0.4, 0.3)
    slack = _cells(0.8, 0.6, 0.5, 0.4)
    e = R._estimands(bind, slack)
    assert e["GP_bind"] == pytest_approx(0.7)
    assert e["GP_slack"] == pytest_approx(0.2)
    assert e["A_P"] == pytest_approx(0.5)
    assert e["D_bind"] == pytest_approx(0.7 - 0.1)
    assert e["T"] == pytest_approx(0.6 - 0.1)
    assert e["M_bind"] == pytest_approx(0.1 - 0.3)
    assert e["dM"] == pytest_approx(0.2 - (-0.2))


def pytest_approx(x):
    import pytest

    return pytest.approx(x, abs=1e-12)


def _loaded(shift):
    """Two rollouts over the same tasks; rollout 2 is rollout 1 plus a constant shift."""
    tasks = [f"{i:010d}" for i in range(12)]
    out = {}
    for name, off in (("r1", 0.0), ("r2", shift)):
        out[name] = {
            cap: {
                cl: {t: 0.1 * i + off for i, t in enumerate(tasks)} for cl in CELLS
            }
            for cap in R.CAPS
        }
    return out, tasks


def test_a_constant_shift_cancels_in_every_contrast():
    """Every estimand is a difference of cell means, so a shift applied to all cells moves
    none of them -- and the retest interval must be exactly zero, not merely small."""
    loaded, tasks = _loaded(0.25)
    res = R.analyse(loaded, tasks, nb=200)
    assert res["reference"] == "r1"
    for k in R.KEYS:
        assert res["deltas"]["r2"][k]["delta"] == pytest_approx(0.0)
        lo, hi = res["deltas"]["r2"][k]["ci"]
        assert lo == pytest_approx(0.0) and hi == pytest_approx(0.0)


def test_per_cell_retest_recovers_the_shift_it_was_given():
    loaded, tasks = _loaded(0.25)
    rows = R.cell_retest(loaded, tasks, nb=200)
    assert len(rows) == len(CELLS) * len(R.CAPS)
    for row in rows.values():
        assert row["delta"] == pytest_approx(0.25)
        assert row["sd_of_task_diffs"] == pytest_approx(0.0)
        lo, hi = row["ci"]
        assert lo == pytest_approx(0.25) and hi == pytest_approx(0.25)


def test_the_two_rollouts_are_resampled_on_the_same_tasks():
    """A retest difference is paired on the instance: if the two rollouts were resampled
    independently, a cell-specific per-task pattern would leak into the interval."""
    tasks = [f"{i:010d}" for i in range(10)]
    hard = {t: (0.9 if i % 2 else 0.1) for i, t in enumerate(tasks)}
    loaded = {
        r: {cap: {cl: dict(hard) for cl in CELLS} for cap in R.CAPS}
        for r in ("r1", "r2")
    }
    res = R.analyse(loaded, tasks, nb=300)
    for k in R.KEYS:
        lo, hi = res["deltas"]["r2"][k]["ci"]
        assert lo == pytest_approx(0.0) and hi == pytest_approx(0.0)


def test_more_than_two_arms_are_all_differenced_against_the_first():
    """Base-model arms: every arm is compared to the reference, not pairwise to each other."""
    tasks = [f"{i:010d}" for i in range(6)]
    loaded = {
        name: {
            cap: {cl: {t: 0.5 + off for t in tasks} for cl in CELLS} for cap in R.CAPS
        }
        for name, off in (("r1", 0.0), ("mistral", 0.1), ("llama31", 0.2))
    }
    res = R.analyse(loaded, tasks, nb=50)
    assert res["reference"] == "r1"
    assert set(res["deltas"]) == {"mistral", "llama31"}
    # uniform per-arm offsets cancel inside every contrast, so all deltas are exactly zero
    for arm in res["deltas"]:
        for k in R.KEYS:
            assert res["deltas"][arm][k]["delta"] == pytest_approx(0.0)

