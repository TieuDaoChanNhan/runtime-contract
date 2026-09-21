import unittest
from typing import Any

from codeact_runtime.config import RuleDiagnosisConfig
from codeact_runtime.families.rule_diagnosis import (
    RuleDiagnosisEnv,
    _count_recurrences,
    _eval_base,
    sample_rule_diagnosis_instance,
)


def _cfg(**overrides: Any) -> RuleDiagnosisConfig:
    """Minimal default config — easy tier, no micro intervals."""
    defaults: dict[str, Any] = dict(
        probe_budget_range=(18, 20),
        mod_m_choices=[7, 11, 13],
        domain_range=(0, 999),
    )
    defaults.update(overrides)
    return RuleDiagnosisConfig(**defaults)


def _hard_cfg(**overrides: Any) -> RuleDiagnosisConfig:
    """Hard tier with one micro interval and library_size=3."""
    defaults: dict[str, Any] = dict(
        probe_budget_range=(36, 50),
        mod_m_choices=[7, 11, 13],
        domain_range=(0, 999),
        micro_interval_count=1,
        micro_interval_len_range=(3, 4),
        rule_library_size=3,
    )
    defaults.update(overrides)
    return RuleDiagnosisConfig(**defaults)


def _make_env(seed: int = 42, cfg: RuleDiagnosisConfig | None = None) -> RuleDiagnosisEnv:
    cfg = cfg or _cfg()
    task = sample_rule_diagnosis_instance(seed, cfg)
    return RuleDiagnosisEnv.from_task(task.model_dump(mode="json"))


def _true_hyp(env: RuleDiagnosisEnv) -> dict:
    return {"m": env.m, "family": env.family, **env.params, "exceptions": {}}


class TestGeneratorStructure(unittest.TestCase):
    """Generator output satisfies structural invariants."""

    def _intervals(self, seed: int, cfg: RuleDiagnosisConfig) -> list[dict]:
        return sample_rule_diagnosis_instance(seed, cfg).private.params["intervals"]

    def test_domain_fully_covered_no_gaps(self):
        cfg = _hard_cfg()
        for seed in range(20):
            ivs = self._intervals(seed, cfg)
            x_min, x_max = cfg.domain_range
            self.assertEqual(ivs[0]["x_min"], x_min, f"seed={seed}")
            self.assertEqual(ivs[-1]["x_max"], x_max, f"seed={seed}")
            for i in range(len(ivs) - 1):
                self.assertEqual(ivs[i]["x_max"] + 1, ivs[i + 1]["x_min"], f"seed={seed} gap at i={i}")

    def test_normal_intervals_meet_min_interval_len(self):
        cfg = _hard_cfg()
        for seed in range(30):
            ivs = self._intervals(seed, cfg)
            for iv in ivs:
                length = iv["x_max"] - iv["x_min"] + 1
                if length >= cfg.min_interval_len:
                    continue  # normal
                # micro — just ensure it's within the allowed range
                self.assertGreaterEqual(length, cfg.micro_interval_len_range[0], f"seed={seed}")
                self.assertLessEqual(length, cfg.micro_interval_len_range[1], f"seed={seed}")

    def test_all_intervals_respect_min_len_when_no_micro(self):
        cfg = _cfg()
        for seed in range(50):
            ivs = self._intervals(seed, cfg)
            for iv in ivs:
                length = iv["x_max"] - iv["x_min"] + 1
                self.assertGreaterEqual(
                    length, cfg.min_interval_len,
                    f"seed={seed} interval [{iv['x_min']}, {iv['x_max']}] too short"
                )

    def test_adjacent_intervals_have_different_rules(self):
        # Independent path (library_size=0): adjacent sub_families must differ.
        # Library path (library_size>=2): adjacent (sub_family, params) pairs must differ.
        # In both cases the key invariant is: no adjacent interval uses the identical rule.
        for cfg in [_cfg(), _hard_cfg()]:
            for seed in range(20):
                ivs = self._intervals(seed, cfg)
                for i in range(len(ivs) - 1):
                    left = (ivs[i]["sub_family"], tuple(sorted(ivs[i]["params"].items())))
                    right = (ivs[i + 1]["sub_family"], tuple(sorted(ivs[i + 1]["params"].items())))
                    self.assertNotEqual(left, right, f"seed={seed} adjacent identical rule at {i}")

    def test_determinism_same_seed_same_output(self):
        cfg = _hard_cfg()
        t1 = sample_rule_diagnosis_instance(99, cfg)
        t2 = sample_rule_diagnosis_instance(99, cfg)
        self.assertEqual(t1.model_dump(mode="json"), t2.model_dump(mode="json"))

    def test_no_exceptions_in_generated_tasks(self):
        cfg = _hard_cfg()
        for seed in range(20):
            task = sample_rule_diagnosis_instance(seed, cfg)
            self.assertEqual({}, task.private.exceptions, f"seed={seed}")

    def test_breakpoint_count_in_correct_tier(self):
        # Easy: exactly 1 breakpoint (2 intervals)
        easy_cfg = _cfg(probe_budget_range=(18, 20))
        for seed in range(10):
            ivs = self._intervals(seed, easy_cfg)
            self.assertEqual(2, len(ivs), f"easy seed={seed}")

        # Medium: 2–3 breakpoints (3–4 intervals)
        med_cfg = _cfg(probe_budget_range=(24, 26))
        for seed in range(10):
            ivs = self._intervals(seed, med_cfg)
            self.assertIn(len(ivs), {3, 4}, f"medium seed={seed}")

        # Hard (no micro): 4–6 breakpoints (5–7 intervals) — budget [36,50] > medium_max=35
        hard_cfg = _cfg(probe_budget_range=(36, 50))
        for seed in range(10):
            ivs = self._intervals(seed, hard_cfg)
            self.assertIn(len(ivs), {5, 6, 7}, f"hard seed={seed}")


class TestMicroIntervals(unittest.TestCase):
    """Micro-interval structural and semantic invariants."""

    def _micro_intervals(self, seed: int, cfg: RuleDiagnosisConfig) -> list[tuple[int, dict]]:
        ivs = sample_rule_diagnosis_instance(seed, cfg).private.params["intervals"]
        return [(i, iv) for i, iv in enumerate(ivs) if iv["x_max"] - iv["x_min"] + 1 < cfg.min_interval_len]

    def test_micro_count_matches_config(self):
        cfg = _hard_cfg(micro_interval_count=1)
        for seed in range(30):
            micros = self._micro_intervals(seed, cfg)
            self.assertEqual(1, len(micros), f"seed={seed}")

    def test_micro_intervals_are_interior(self):
        """Micro intervals must not be the first or last segment."""
        cfg = _hard_cfg()
        for seed in range(30):
            task = sample_rule_diagnosis_instance(seed, cfg)
            ivs = task.private.params["intervals"]
            for i, iv in enumerate(ivs):
                length = iv["x_max"] - iv["x_min"] + 1
                if length < cfg.min_interval_len:
                    self.assertGreater(i, 0, f"seed={seed}: micro at first position")
                    self.assertLess(i, len(ivs) - 1, f"seed={seed}: micro at last position")

    def test_micro_intervals_length_in_range(self):
        cfg = _hard_cfg(micro_interval_len_range=(3, 4))
        for seed in range(30):
            for _, iv in self._micro_intervals(seed, cfg):
                length = iv["x_max"] - iv["x_min"] + 1
                self.assertGreaterEqual(length, 3, f"seed={seed}")
                self.assertLessEqual(length, 4, f"seed={seed}")

    def test_short_micro_intervals_exclude_quadratic(self):
        """Intervals with length < 4 must not use quadratic_mod."""
        cfg = _hard_cfg()
        violations = 0
        for seed in range(100):
            for _, iv in self._micro_intervals(seed, cfg):
                length = iv["x_max"] - iv["x_min"] + 1
                if length < 4:
                    self.assertNotEqual(
                        iv["sub_family"], "quadratic_mod",
                        f"seed={seed} micro length={length} got quadratic_mod"
                    )
                    violations += 1  # at least one checked
        self.assertGreater(violations, 0, "no micro intervals with length < 4 found; test is vacuous")

    def test_micro_intervals_distinguishable_from_neighbors(self):
        """Each micro interval must differ from both its neighbors on at least one x."""
        m = 7  # use known prime for determinism in this assertion
        cfg_fixed_m = _hard_cfg(mod_m_choices=[7])
        for seed in range(30):
            task = sample_rule_diagnosis_instance(seed, cfg_fixed_m)
            ivs = task.private.params["intervals"]
            m = task.public.m
            for i, iv in enumerate(ivs):
                if iv["x_max"] - iv["x_min"] + 1 >= cfg_fixed_m.min_interval_len:
                    continue
                xs = list(range(iv["x_min"], iv["x_max"] + 1))
                if i > 0:
                    left = ivs[i - 1]
                    differs_left = any(
                        _eval_base(iv["sub_family"], iv["params"], x, m)
                        != _eval_base(left["sub_family"], left["params"], x, m)
                        for x in xs
                    )
                    self.assertTrue(differs_left, f"seed={seed} micro at idx={i} indistinguishable from left")
                if i < len(ivs) - 1:
                    right = ivs[i + 1]
                    differs_right = any(
                        _eval_base(iv["sub_family"], iv["params"], x, m)
                        != _eval_base(right["sub_family"], right["params"], x, m)
                        for x in xs
                    )
                    self.assertTrue(differs_right, f"seed={seed} micro at idx={i} indistinguishable from right")

    def test_boundary_point_visibility_all_adjacent_pairs(self):
        """Adjacent interval formulas must differ at the boundary point itself.

        check() returns a witness inside the first wrong run; if the two formulas
        agree at the boundary, the wrong run starts one step later than expected.
        """
        for cfg in [_cfg(), _hard_cfg()]:
            for seed in range(30):
                task = sample_rule_diagnosis_instance(seed, cfg)
                ivs = task.private.params["intervals"]
                m = task.public.m
                for i in range(len(ivs) - 1):
                    left, right = ivs[i], ivs[i + 1]
                    bx = right["x_min"]
                    left_val = _eval_base(left["sub_family"], left["params"], bx, m)
                    right_val = _eval_base(right["sub_family"], right["params"], bx, m)
                    self.assertNotEqual(
                        left_val, right_val,
                        f"cfg={cfg.probe_budget_range} seed={seed} boundary i={i} x={bx}: "
                        f"{left['sub_family']}={left_val} == {right['sub_family']}={right_val}"
                    )

    def test_no_micro_by_default(self):
        cfg = _cfg()
        for seed in range(20):
            micros = self._micro_intervals(seed, cfg)
            self.assertEqual([], micros, f"seed={seed}")


class TestRuleLibrary(unittest.TestCase):
    """Generator with rule_library_size >= 2 produces valid recurrence patterns."""

    def _lib_cfg(self, library_size: int, **overrides: Any) -> RuleDiagnosisConfig:
        """Medium tier with specified library size."""
        defaults: dict[str, Any] = dict(
            probe_budget_range=(24, 26),
            mod_m_choices=[7, 11, 13],
            domain_range=(0, 999),
            rule_library_size=library_size,
        )
        defaults.update(overrides)
        return RuleDiagnosisConfig(**defaults)

    def test_rejects_library_size_one(self):
        with self.assertRaises(ValueError):
            _cfg(rule_library_size=1)

    def test_library_size_two_always_produces_recurrences(self):
        # With library_size=2 and >= 3 intervals, the A-B-A pattern is forced.
        cfg = self._lib_cfg(2)
        for seed in range(20):
            task = sample_rule_diagnosis_instance(seed, cfg)
            ivs = task.private.params["intervals"]
            if len(ivs) < 3:
                continue
            rec = _count_recurrences(ivs)
            self.assertGreater(rec, 0, f"seed={seed}: expected at least one recurrence")

    def test_library_size_three_medium_recurrences(self):
        # Library of 3 with 3-4 intervals: some recurrences expected over many seeds.
        cfg = self._lib_cfg(3)
        total_rec = sum(
            _count_recurrences(sample_rule_diagnosis_instance(seed, cfg).private.params["intervals"])
            for seed in range(30)
        )
        self.assertGreater(total_rec, 0, "expected at least one recurrence across 30 seeds")

    def test_shared_library_rules_have_identical_params(self):
        # When library is used, intervals sharing the same (sub_family, params)
        # must have EXACTLY identical params — not just identical sub_family.
        cfg = self._lib_cfg(2)
        for seed in range(20):
            ivs = sample_rule_diagnosis_instance(seed, cfg).private.params["intervals"]
            # Build a map: (sub_family, params_frozen) → list of interval indices
            from collections import defaultdict
            groups: dict = defaultdict(list)
            for i, iv in enumerate(ivs):
                key = (iv["sub_family"], tuple(sorted(iv["params"].items())))
                groups[key].append(i)
            # If any group has >= 2 members, check they are non-adjacent
            for key, indices in groups.items():
                if len(indices) >= 2:
                    # All params must be byte-for-byte identical (guaranteed by library)
                    params_list = [ivs[i]["params"] for i in indices]
                    for p in params_list[1:]:
                        self.assertEqual(params_list[0], p, f"seed={seed}: shared rule params differ")

    def test_boundary_visibility_holds_with_library(self):
        # Even with shared params, all adjacent boundaries must be visible.
        cfg = self._lib_cfg(2)
        for seed in range(20):
            task = sample_rule_diagnosis_instance(seed, cfg)
            ivs = task.private.params["intervals"]
            m = task.public.m
            for i in range(1, len(ivs)):
                bx = ivs[i]["x_min"]
                left_val = _eval_base(ivs[i - 1]["sub_family"], ivs[i - 1]["params"], bx, m)
                right_val = _eval_base(ivs[i]["sub_family"], ivs[i]["params"], bx, m)
                self.assertNotEqual(
                    left_val, right_val,
                    f"seed={seed}: boundary at x={bx} invisible (left={left_val}, right={right_val})"
                )

    def test_recurrence_count_metric_nonzero_for_library_tasks(self):
        cfg = _hard_cfg(rule_library_size=3)
        found_recurrence = False
        for seed in range(20):
            task = sample_rule_diagnosis_instance(seed, cfg)
            if task.difficulty["recurrence_count"] > 0:
                found_recurrence = True
                break
        self.assertTrue(found_recurrence, "expected at least one recurrence in 20 hard seeds")


class TestConfigValidation(unittest.TestCase):
    """Config validator rejects invalid combinations."""

    def test_rejects_micro_count_exceeds_interior_slots(self):
        # Easy tier: 1 breakpoint → 0 interior slots → any micro_interval_count > 0 fails
        with self.assertRaises(ValueError, msg="easy tier with micro should fail"):
            _cfg(probe_budget_range=(18, 20), micro_interval_count=1)

    def test_rejects_micro_count_exceeds_min_breakpoints_interior(self):
        # Medium tier: min 2 breakpoints → 1 interior slot → count=2 must fail
        with self.assertRaises(ValueError):
            _cfg(probe_budget_range=(24, 26), micro_interval_count=2)

    def test_rejects_micro_len_range_hi_ge_min_interval_len(self):
        with self.assertRaises(ValueError):
            _hard_cfg(micro_interval_len_range=(2, 6), min_interval_len=6)

    def test_rejects_micro_len_range_lo_below_affine_min(self):
        # lo=2 < RD_MIN_AFFINE_LEN=3 must be rejected
        with self.assertRaises(ValueError):
            _hard_cfg(micro_interval_len_range=(2, 4))

    def test_rejects_micro_len_range_lo_zero(self):
        with self.assertRaises(ValueError):
            _hard_cfg(micro_interval_len_range=(0, 4))

    def test_rejects_micro_len_range_lo_gt_hi(self):
        with self.assertRaises(ValueError):
            _hard_cfg(micro_interval_len_range=(4, 2))

    def test_rejects_domain_too_narrow_for_micro(self):
        # Hard tier: 7 segs max, 1 micro max_len=4, 6 normal each need min_len+1=7
        # min_domain = 1*4 + 6*7 = 46; domain (0,44) is width 45 < 46
        with self.assertRaises(ValueError):
            RuleDiagnosisConfig(  # pyright: ignore[reportCallIssue]
                probe_budget_range=(32, 40),
                mod_m_choices=[7, 11, 13],
                domain_range=(0, 44),
                micro_interval_count=1,
                micro_interval_len_range=(3, 4),
                min_interval_len=6,
            )

    def test_rejects_no_prime_in_mod_choices(self):
        with self.assertRaises(ValueError):
            _cfg(mod_m_choices=[4, 6, 8])

    def test_rejects_probe_budget_below_minimum(self):
        with self.assertRaises(ValueError):
            _cfg(probe_budget_range=(3, 5))

    def test_accepts_medium_with_one_micro(self):
        # Medium min 2 breakpoints → 1 interior slot → count=1 should pass
        cfg = _cfg(probe_budget_range=(24, 26), micro_interval_count=1)
        self.assertEqual(1, cfg.micro_interval_count)

    def test_accepts_hard_with_three_micro(self):
        # Hard min 4 breakpoints → 3 interior slots → count=3 should pass
        cfg = _hard_cfg(micro_interval_count=3)
        self.assertEqual(3, cfg.micro_interval_count)

    def test_rejects_online_with_probes_per_turn(self):
        # online drops pull-based test_input(), so the per-turn throttle is dead;
        # combining them is a configuration error.
        with self.assertRaisesRegex(ValueError, "online"):
            _cfg(online=True, probes_per_turn=4)

    def test_accepts_online_defaults(self):
        cfg = _cfg(online=True)
        self.assertTrue(cfg.online)
        self.assertIsNone(cfg.probes_per_turn)


class TestTaskDescription(unittest.TestCase):
    """Task description content."""

    def test_no_exception_mention_when_max_exceptions_zero(self):
        env = _make_env(cfg=_cfg(max_hypothesis_exceptions=0))
        desc = env.get_goal_prompt()
        self.assertNotIn("exception", desc.lower())
        self.assertNotIn("Exception", desc)

    def test_strategy_guide_mentions_binary_search_and_micro_hint(self):
        env = _make_env(cfg=_hard_cfg())
        desc = env.get_goal_prompt()
        self.assertIn("BINARY SEARCH", desc)
        self.assertIn("short", desc.lower())

    def test_modulus_appears_in_description(self):
        env = _make_env()
        desc = env.get_goal_prompt()
        self.assertIn(str(env.m), desc)

    def test_probe_budget_appears_in_description(self):
        env = _make_env()
        desc = env.get_goal_prompt()
        self.assertIn(str(env.probe_budget), desc)


class TestCheckAndSubmit(unittest.IsolatedAsyncioTestCase):
    """check() and submit() correctness."""

    async def test_true_hypothesis_passes_check(self):
        cfg = _hard_cfg()
        for seed in range(10):
            env = _make_env(seed=seed, cfg=cfg)
            result = await env.check(_true_hyp(env))
            self.assertEqual({"status": "pass"}, result, f"seed={seed}")

    async def test_true_hypothesis_passes_submit(self):
        cfg = _hard_cfg()
        for seed in range(5):
            env = _make_env(seed=seed, cfg=cfg)
            result = await env.submit(_true_hyp(env))
            self.assertEqual({"status": "pass"}, result, f"seed={seed}")
            self.assertTrue(env.solved, f"seed={seed}")
            ratio = env.last_submit_match_ratio
            self.assertIsNotNone(ratio, msg=f"seed={seed}")
            assert ratio is not None  # narrow type for pyright
            self.assertAlmostEqual(1.0, ratio, msg=f"seed={seed}")

    async def test_check_returns_witness_in_first_wrong_run(self):
        from codeact_runtime.families.rule_diagnosis import _WITNESS_WINDOW
        env = _make_env()
        bad_hyp = {
            "m": env.m,
            "family": "stepwise_composition",
            "intervals": [
                {"x_min": env.x_min, "x_max": env.x_max, "sub_family": "affine_mod",
                 "params": {"a": 1, "b": 0}},
            ],
            "exceptions": {},
        }
        # Run multiple times to verify window constraint holds across samples
        first_wrong = None
        for _ in range(20):
            result = await env.check(bad_hyp)
            if result["status"] == "fail":
                w = result["x"]
                self.assertGreaterEqual(w, env.x_min)
                self.assertLessEqual(w, env.x_max)
                # witness must be a genuinely wrong point
                self.assertNotEqual(env._f(w), result["y_pred"])
                # find first wrong point on first iteration
                if first_wrong is None:
                    for x in range(env.x_min, env.x_max + 1):
                        if env._f(x) != (1 * x + 0) % env.m:
                            first_wrong = x
                            break
                # witness must be within the window
                if first_wrong is not None:
                    self.assertLessEqual(w, first_wrong + _WITNESS_WINDOW - 1)

    async def test_check_does_not_end_task(self):
        env = _make_env()
        bad = {
            "m": env.m,
            "family": "stepwise_composition",
            "intervals": [
                {"x_min": env.x_min, "x_max": env.x_max, "sub_family": "affine_mod",
                 "params": {"a": 1, "b": 0}},
            ],
            "exceptions": {},
        }
        await env.check(bad)
        self.assertFalse(env.done)

    async def test_check_witness_is_always_wrong(self):
        # check() is deliberately stochastic — repeated calls may return different
        # witness points, but each must be a genuinely wrong point.
        env = _make_env()
        bad = {
            "m": env.m,
            "family": "stepwise_composition",
            "intervals": [
                {"x_min": env.x_min, "x_max": env.x_max, "sub_family": "affine_mod",
                 "params": {"a": 1, "b": 0}},
            ],
            "exceptions": {},
        }
        for _ in range(10):
            r = await env.check(bad)
            if r["status"] == "fail":
                self.assertNotEqual(env._f(r["x"]), r["y_pred"])

    async def test_submit_ends_task_and_rejects_further_calls(self):
        env = _make_env()
        await env.submit(_true_hyp(env))
        self.assertTrue(env.done)

        with self.assertRaisesRegex(ValueError, "task already submitted"):
            await env.submit(_true_hyp(env))
        with self.assertRaisesRegex(ValueError, "task already submitted"):
            await env.check(_true_hyp(env))
        with self.assertRaisesRegex(ValueError, "task already submitted"):
            await env.test_input(env.x_min)

    async def test_probe_budget_enforced(self):
        env = _make_env()
        for _ in range(env.probe_budget):
            await env.test_input(env.x_min)
        with self.assertRaisesRegex(ValueError, "budget"):
            await env.test_input(env.x_min)

    async def test_probes_per_turn_none_is_unlimited(self):
        # Legacy default: no per-turn cap, so the whole budget can be spent in one turn.
        env = _make_env()
        self.assertIsNone(env.probes_per_turn)
        for _ in range(env.probe_budget):
            await env.test_input(env.x_min)
        self.assertEqual(env.probes_used, env.probe_budget)

    async def test_probes_per_turn_cap_enforced_and_resets(self):
        # With a per-turn cap, only `cap` probes succeed per turn; reset_turn()
        # (called by the interpreter's on_turn_start hook each turn) re-arms it.
        env = _make_env(cfg=_cfg(probes_per_turn=3))
        self.assertEqual(env.probes_per_turn, 3)
        for _ in range(3):
            await env.test_input(env.x_min)
        with self.assertRaisesRegex(ValueError, "per-turn probe cap"):
            await env.test_input(env.x_min)
        # New turn re-arms the cap.
        env.reset_turn()
        await env.test_input(env.x_min)
        self.assertEqual(env.probes_used, 4)

    async def test_probes_per_turn_propagates_to_public_and_prompt(self):
        env = _make_env(cfg=_cfg(probes_per_turn=4))
        self.assertIn("Per-turn probe cap", env.get_goal_prompt())

    async def test_hypotheses_with_exceptions_rejected_when_max_zero(self):
        env = _make_env(cfg=_cfg(max_hypothesis_exceptions=0))
        bad = {**_true_hyp(env), "exceptions": {"0": 3}}
        with self.assertRaisesRegex(ValueError, "too many exceptions"):
            await env.check(bad)

    async def test_schema_error_does_not_end_task(self):
        env = _make_env()
        with self.assertRaises(ValueError):
            await env.check({"m": env.m, "family": "stepwise_composition", "intervals": []})
        self.assertFalse(env.done)

    async def test_wrong_modulus_rejected(self):
        env = _make_env()
        bad = {**_true_hyp(env), "m": env.m + 1}
        with self.assertRaisesRegex(ValueError, "wrong modulus"):
            await env.check(bad)

    async def test_rejects_affine_interval_below_min_length(self):
        env = _make_env()
        # affine_mod interval with only 2 points — must be rejected
        hyp = {
            "m": env.m,
            "family": "stepwise_composition",
            "intervals": [
                {"x_min": 0, "x_max": 1, "sub_family": "affine_mod", "params": {"a": 1, "b": 0}},
                {"x_min": 2, "x_max": env.x_max, "sub_family": "affine_mod", "params": {"a": 2, "b": 1}},
            ],
            "exceptions": {},
        }
        with self.assertRaisesRegex(ValueError, "requires at least"):
            await env.check(hyp)

    async def test_rejects_quadratic_interval_below_min_length(self):
        env = _make_env()
        # quadratic_mod interval with 3 points — must be rejected (needs >= 4)
        hyp = {
            "m": env.m,
            "family": "stepwise_composition",
            "intervals": [
                {"x_min": 0, "x_max": 2, "sub_family": "quadratic_mod",
                 "params": {"a": 1, "b": 0, "c": 0}},
                {"x_min": 3, "x_max": env.x_max, "sub_family": "affine_mod",
                 "params": {"a": 1, "b": 0}},
            ],
            "exceptions": {},
        }
        with self.assertRaisesRegex(ValueError, "requires at least"):
            await env.check(hyp)

    async def test_accepts_affine_interval_at_exact_min_length(self):
        env = _make_env()
        # 3-point affine interval must be accepted by the validator
        hyp = {
            "m": env.m,
            "family": "stepwise_composition",
            "intervals": [
                {"x_min": 0, "x_max": 2, "sub_family": "affine_mod", "params": {"a": 1, "b": 0}},
                {"x_min": 3, "x_max": env.x_max, "sub_family": "affine_mod",
                 "params": {"a": 2, "b": 1}},
            ],
            "exceptions": {},
        }
        result = await env.check(hyp)
        self.assertIn(result["status"], ("pass", "fail"))


class TestEvaluateMetrics(unittest.IsolatedAsyncioTestCase):
    """evaluate() reports correct metrics."""

    async def test_score_zero_before_submit(self):
        env = _make_env()
        self.assertEqual(0.0, env.evaluate().score)

    async def test_score_one_after_correct_submit(self):
        env = _make_env(cfg=_hard_cfg())
        await env.submit(_true_hyp(env))
        result = env.evaluate()
        self.assertAlmostEqual(1.0, result.score)
        self.assertTrue(result.is_solved)

    async def test_true_interval_count_matches_generator(self):
        cfg = _hard_cfg()
        for seed in range(5):
            env = _make_env(seed=seed, cfg=cfg)
            task = sample_rule_diagnosis_instance(seed, cfg)
            expected = len(task.private.params["intervals"])
            await env.submit(_true_hyp(env))
            metrics = env.evaluate().metrics
            self.assertEqual(expected, metrics["true_interval_count"], f"seed={seed}")

    async def test_num_micro_true_counts_short_intervals(self):
        cfg = _hard_cfg()
        for seed in range(10):
            env = _make_env(seed=seed, cfg=cfg)
            task = sample_rule_diagnosis_instance(seed, cfg)
            expected_micro = sum(
                1 for iv in task.private.params["intervals"]
                if iv["x_max"] - iv["x_min"] + 1 < 5
            )
            await env.submit(_true_hyp(env))
            metrics = env.evaluate().metrics
            self.assertEqual(expected_micro, metrics["num_micro_true"], f"seed={seed}")

    async def test_submitted_interval_count_matches_hypothesis(self):
        env = _make_env(cfg=_hard_cfg())
        hyp = _true_hyp(env)
        await env.submit(hyp)
        metrics = env.evaluate().metrics
        self.assertEqual(len(hyp["intervals"]), metrics["submitted_interval_count"])

    async def test_first_unsolved_micro_true_when_first_fail_in_micro(self):
        cfg = _hard_cfg()
        # Find a seed where the micro interval comes before all normal failures
        found = False
        for seed in range(50):
            task = sample_rule_diagnosis_instance(seed, cfg)
            ivs = task.private.params["intervals"]
            micro_ivs = [(i, iv) for i, iv in enumerate(ivs) if iv["x_max"] - iv["x_min"] + 1 < 5]
            if not micro_ivs:
                continue
            micro_idx, micro_iv = micro_ivs[0]

            # Build hypothesis that omits the micro interval (merges it with left neighbor)
            wrong_ivs = []
            for i, iv in enumerate(ivs):
                if i == micro_idx:
                    continue  # skip micro
                if i == micro_idx - 1:
                    # extend left neighbor to cover the micro range too
                    merged = dict(iv)
                    merged["x_max"] = micro_iv["x_max"]
                    wrong_ivs.append(merged)
                else:
                    wrong_ivs.append(iv)

            hyp = {"m": task.public.m, "family": "stepwise_composition",
                   "intervals": wrong_ivs, "exceptions": {}}
            env = RuleDiagnosisEnv.from_task(task.model_dump(mode="json"))
            result = await env.submit(hyp)
            if result["status"] == "fail" and result["x"] >= micro_iv["x_min"]:
                metrics = env.evaluate().metrics
                if metrics["first_unsolved_micro"] is True:
                    found = True
                    break
        self.assertTrue(found, "could not construct a first_unsolved_micro=True case")

    async def test_first_unsolved_micro_none_when_correct_submit(self):
        env = _make_env(cfg=_hard_cfg())
        await env.submit(_true_hyp(env))
        metrics = env.evaluate().metrics
        self.assertIsNone(metrics["first_unsolved_micro"])

    async def test_efficiency_decreases_with_probe_usage(self):
        env = _make_env()
        await env.test_input(env.x_min)
        await env.test_input(env.x_min + 1)
        await env.submit(_true_hyp(env))
        metrics = env.evaluate().metrics
        self.assertLess(metrics["efficiency"], 1.0)
        self.assertGreater(metrics["efficiency"], 0.0)

    async def test_boundary_f1_and_family_acc_perfect_on_true_hyp(self):
        cfg = _hard_cfg()
        for seed in range(5):
            env = _make_env(seed=seed, cfg=cfg)
            await env.submit(_true_hyp(env))
            metrics = env.evaluate().metrics
            self.assertAlmostEqual(1.0, metrics["boundary_f1"], msg=f"seed={seed}")
            self.assertAlmostEqual(1.0, metrics["family_acc"], msg=f"seed={seed}")

    async def test_is_solved_false_for_over_split_hypothesis(self):
        """Functionally correct but over-split submission must not be marked solved."""
        cfg = _hard_cfg()
        for seed in range(20):
            task = sample_rule_diagnosis_instance(seed, cfg)
            ivs = task.private.params["intervals"]
            if len(ivs) < 2:
                continue
            # Split the first interval into two equal halves
            first = ivs[0]
            mid = (first["x_min"] + first["x_max"]) // 2
            split1 = dict(first, x_max=mid)
            split2 = dict(first, x_min=mid + 1)
            split_ivs = [split1, split2] + list(ivs[1:])
            hyp = {"m": task.public.m, "family": "stepwise_composition",
                   "intervals": split_ivs, "exceptions": {}}
            env = RuleDiagnosisEnv.from_task(task.model_dump(mode="json"))
            await env.submit(hyp)
            result = env.evaluate()
            # Even if functional score is high, is_solved must be False (over-split)
            self.assertFalse(result.is_solved, f"seed={seed}: over-split should not be solved")
            # Boundary F1 < 1 because we added a spurious interior boundary
            metrics = result.metrics
            self.assertLess(metrics["boundary_f1"], 1.0, f"seed={seed}")
            break  # one seed is enough

    async def test_composite_score_penalizes_over_split(self):
        """Score with over-split intervals is lower than score with true hypothesis."""
        cfg = _hard_cfg()
        for seed in range(20):
            task = sample_rule_diagnosis_instance(seed, cfg)
            ivs = task.private.params["intervals"]
            if len(ivs) < 2:
                continue
            first = ivs[0]
            mid = (first["x_min"] + first["x_max"]) // 2
            split_ivs = [dict(first, x_max=mid), dict(first, x_min=mid + 1)] + list(ivs[1:])
            hyp_split = {"m": task.public.m, "family": "stepwise_composition",
                         "intervals": split_ivs, "exceptions": {}}
            env_split = RuleDiagnosisEnv.from_task(task.model_dump(mode="json"))
            await env_split.submit(hyp_split)
            score_split = env_split.evaluate().score

            env_true = RuleDiagnosisEnv.from_task(task.model_dump(mode="json"))
            await env_true.submit(_true_hyp(env_true))
            score_true = env_true.evaluate().score

            self.assertGreater(score_true, score_split, f"seed={seed}")
            break

    async def test_boundary_f1_zero_when_no_boundaries_submitted_for_multi_interval_truth(self):
        """Single-interval submission against multi-interval truth has boundary_f1=0."""
        cfg = _hard_cfg()
        for seed in range(20):
            task = sample_rule_diagnosis_instance(seed, cfg)
            ivs = task.private.params["intervals"]
            if len(ivs) < 2:
                continue
            # Submit single interval covering entire domain using first interval's formula
            first = ivs[0]
            flat_hyp = {
                "m": task.public.m,
                "family": "stepwise_composition",
                "intervals": [{"x_min": task.public.x_domain["min"],
                                "x_max": task.public.x_domain["max"],
                                "sub_family": first["sub_family"],
                                "params": first["params"]}],
                "exceptions": {},
            }
            env = RuleDiagnosisEnv.from_task(task.model_dump(mode="json"))
            await env.submit(flat_hyp)
            metrics = env.evaluate().metrics
            self.assertEqual(0.0, metrics["boundary_f1"], f"seed={seed}")
            break

    async def test_complexity_penalty_nonzero_for_extra_intervals(self):
        cfg = _hard_cfg()
        for seed in range(20):
            task = sample_rule_diagnosis_instance(seed, cfg)
            ivs = task.private.params["intervals"]
            if len(ivs) < 2:
                continue
            first = ivs[0]
            mid = (first["x_min"] + first["x_max"]) // 2
            split_ivs = [dict(first, x_max=mid), dict(first, x_min=mid + 1)] + list(ivs[1:])
            hyp = {"m": task.public.m, "family": "stepwise_composition",
                   "intervals": split_ivs, "exceptions": {}}
            env = RuleDiagnosisEnv.from_task(task.model_dump(mode="json"))
            await env.submit(hyp)
            metrics = env.evaluate().metrics
            self.assertGreater(metrics["complexity_penalty"], 0.0, f"seed={seed}")
            break


def _easy_rd_cfg() -> RuleDiagnosisConfig:
    """Mirrors the easy.json rule_diagnosis block after the alignment-pilot tuning."""
    return RuleDiagnosisConfig(  # pyright: ignore[reportCallIssue]
        num_tasks=1,
        mod_m_choices=[7, 8, 9, 11],
        domain_range=(0, 999),
        probe_budget_range=(28, 33),
        min_interval_len=6,
        reveal_family_in_public=False,
        max_hypothesis_exceptions=0,
        require_multi_breakpoint=True,
    )


class StatePressureAcceptanceTests(unittest.TestCase):
    """Structural acceptance gate for the easy rule_diagnosis tier."""

    def test_easy_has_multi_breakpoint(self) -> None:
        cfg = _easy_rd_cfg()
        for seed in range(200):
            task = sample_rule_diagnosis_instance(seed, cfg)
            ivs = task.private.params["intervals"]
            self.assertGreaterEqual(
                len(ivs), 3,
                msg=f"seed={seed}: only {len(ivs)} intervals, expected >=3 (>=2 breakpoints)",
            )
            self.assertNotIn(
                "state_pressure_warning", task.difficulty,
                msg=f"seed={seed}: warning={task.difficulty.get('state_pressure_warning')}",
            )

    def test_easy_hides_family_hint(self) -> None:
        cfg = _easy_rd_cfg()
        for seed in range(200):
            task = sample_rule_diagnosis_instance(seed, cfg)
            self.assertIsNone(
                task.public.family_hint,
                msg=f"seed={seed}: family_hint leaked = {task.public.family_hint!r}",
            )


def _medium_rd_cfg() -> RuleDiagnosisConfig:
    """Mirrors medium.json rule_diagnosis block after the exceptions-driven tuning."""
    return RuleDiagnosisConfig(  # pyright: ignore[reportCallIssue]
        num_tasks=1,
        mod_m_choices=[11, 13],
        domain_range=(0, 999),
        probe_budget_range=(27, 32),
        min_interval_len=6,
        rule_library_size=0,
        reveal_family_in_public=False,
        exceptions_count_range=(0, 0),
        max_hypothesis_exceptions=0,
        include_piecewise_affine_mod=True,
        require_multi_breakpoint=False,
    )


class MediumStatePressureAcceptanceTests(unittest.TestCase):
    """Acceptance for the medium rule_diagnosis tier — piecewise_affine_mod enabled."""

    def test_medium_intervals_in_range(self) -> None:
        """Medium-tier probe budget [27, 32] → 2-3 breakpoints → 3-4 intervals."""
        cfg = _medium_rd_cfg()
        for seed in range(50):
            task = sample_rule_diagnosis_instance(seed, cfg)
            ivs = task.private.params["intervals"]
            self.assertGreaterEqual(len(ivs), 3, msg=f"seed={seed}: {len(ivs)} intervals")
            self.assertLessEqual(len(ivs), 4, msg=f"seed={seed}: {len(ivs)} intervals")

    def test_medium_piecewise_sub_family_appears(self) -> None:
        """Over 100 seeds, piecewise_affine_mod is sampled in a non-trivial fraction."""
        cfg = _medium_rd_cfg()
        n_with_pw = 0
        for seed in range(100):
            task = sample_rule_diagnosis_instance(seed, cfg)
            ivs = task.private.params["intervals"]
            if any(iv["sub_family"] == "piecewise_affine_mod" for iv in ivs):
                n_with_pw += 1
        # With 4 families and 3-4 intervals per task, expect >= ~30 tasks to have piecewise.
        self.assertGreaterEqual(
            n_with_pw, 30,
            msg=f"only {n_with_pw}/100 tasks have a piecewise_affine_mod interval",
        )

    def test_medium_piecewise_interval_length_safe(self) -> None:
        """Every piecewise_affine_mod interval is ≥ 2*k_max+2 = 16 points long."""
        cfg = _medium_rd_cfg()
        for seed in range(100):
            task = sample_rule_diagnosis_instance(seed, cfg)
            for iv in task.private.params["intervals"]:
                if iv["sub_family"] == "piecewise_affine_mod":
                    length = iv["x_max"] - iv["x_min"] + 1
                    self.assertGreaterEqual(
                        length, 16,
                        msg=f"seed={seed}: piecewise interval length {length} < 16",
                    )
                    k = iv["params"]["k"]
                    self.assertGreaterEqual(k, 3)
                    self.assertLessEqual(k, 7)

    def test_medium_no_exceptions(self) -> None:
        cfg = _medium_rd_cfg()
        for seed in range(50):
            task = sample_rule_diagnosis_instance(seed, cfg)
            self.assertEqual(len(task.private.exceptions), 0)
            self.assertEqual(task.public.max_exceptions, 0)

    def test_medium_no_state_pressure_warning(self) -> None:
        cfg = _medium_rd_cfg()
        ok = 0
        for seed in range(200):
            task = sample_rule_diagnosis_instance(seed, cfg)
            if "state_pressure_warning" not in task.difficulty:
                ok += 1
        self.assertGreaterEqual(
            ok, 195,
            msg=f"only {ok}/200 seeds satisfy structural acceptance",
        )


class TestOnlineMode(unittest.IsolatedAsyncioTestCase):
    """Streaming (online) cross-turn-state mode."""

    def _online_env(self, seed: int = 7, **overrides: Any) -> RuleDiagnosisEnv:
        cfg = _cfg(online=True, **overrides)
        task = sample_rule_diagnosis_instance(seed, cfg)
        return RuleDiagnosisEnv.from_task(task.model_dump(mode="json"))

    def test_online_tools_drop_pull(self):
        # Online mode exposes only observe()+submit(); no test_input/check.
        env = self._online_env()
        names = {t.name for t in env.get_tools()}
        self.assertEqual(names, {"observe", "submit"})
        self.assertNotIn("test_input", names)
        self.assertNotIn("check", names)

    def test_legacy_tools_unchanged(self):
        env = _make_env()  # online=False
        names = {t.name for t in env.get_tools()}
        self.assertEqual(names, {"test_input", "check", "submit"})

    async def test_observe_only_in_online_mode(self):
        env = _make_env()  # online=False
        with self.assertRaisesRegex(ValueError, "only available in online mode"):
            await env.observe()

    async def test_observe_returns_batch_dict(self):
        env = self._online_env()
        out = await env.observe()
        self.assertIn("batch", out)
        self.assertIn("round", out)
        batch = out["batch"]
        self.assertEqual(len(batch), env.stream_batch_size)
        for pair in batch:
            self.assertEqual(len(pair), 2)
            x, y = pair
            self.assertTrue(env.x_min <= x <= env.x_max)
            self.assertTrue(0 <= y < env.m)

    async def test_observe_idempotent_within_turn(self):
        # Repeated observe() in the same turn returns the SAME batch — you cannot
        # fast-forward the stream by calling it in a loop.
        env = self._online_env()
        env.reset_turn()
        a = await env.observe()
        b = await env.observe()
        self.assertEqual(a, b)

    async def test_stream_advances_across_turns(self):
        # Each turn boundary (reset_turn, fired by the interpreter's on_turn_start)
        # advances the stream cursor to a fresh batch.
        env = self._online_env()
        env.reset_turn()
        first = await env.observe()
        env.reset_turn()
        second = await env.observe()
        self.assertEqual(first["round"] + 1, second["round"])
        self.assertNotEqual(first["batch"], second["batch"])

    async def test_observe_blocked_after_submit(self):
        env = self._online_env()
        await env.submit(_true_hyp(env))
        with self.assertRaisesRegex(ValueError, "already submitted"):
            await env.observe()

    def test_label_noise_present_but_denoisable(self):
        # Across many rounds: some labels are corrupted (noise > 0), yet per-grid
        # majority vote recovers the TRUE f(x) for every well-sampled point.
        env = self._online_env(label_noise=0.2)
        votes: dict[int, dict[int, int]] = {}
        corrupted = total = 0
        for r in range(40):
            for x, y in env._stream_batch(r):
                votes.setdefault(x, {})[y] = votes.setdefault(x, {}).get(y, 0) + 1
                total += 1
                if y != env._f(x):
                    corrupted += 1
        self.assertGreater(corrupted, 0, "no label noise observed")
        self.assertLess(corrupted, total / 2, "noise should be a minority")
        for x, d in votes.items():
            if sum(d.values()) >= 5:
                majority = max(d, key=lambda k: d[k])
                self.assertEqual(majority, env._f(x))

    def test_stream_is_deterministic_per_round(self):
        # Same (seed, round) -> identical batch, so both runtimes see one stream.
        e1 = self._online_env(seed=3)
        e2 = self._online_env(seed=3)
        self.assertEqual(e1._stream_batch(5), e2._stream_batch(5))

    def test_online_prompt_describes_stream_not_probe_budget(self):
        env = self._online_env()
        prompt = env.get_goal_prompt()
        self.assertIn("observe()", prompt)
        self.assertIn("ACCUMULATE", prompt)
        self.assertNotIn("Probe budget:", prompt)

    def test_online_restricts_to_linear_families(self):
        # Online instances only use the two linear families (clean fit under noise).
        for seed in range(20):
            cfg = _cfg(online=True)
            ivs = sample_rule_diagnosis_instance(seed, cfg).private.params["intervals"]
            for iv in ivs:
                self.assertIn(iv["sub_family"], {"affine_mod", "affine_mod_popcount"})


if __name__ == "__main__":
    unittest.main()
