from __future__ import annotations

import json
from pathlib import Path
from typing import List, Literal, Optional, Tuple

from pydantic import BaseModel, Field, model_validator

FamilyName = Literal["knapsack", "navigation", "rule_diagnosis"]

# Probe-budget tier thresholds for rule_diagnosis breakpoint generation.
# Both the config validator and the instance generator must use these constants
# so that a config change and a generator change can never silently diverge.
#
#   budget <= RD_TIER_EASY_MAX  →  1 breakpoint  (easy)   probe_budget ≈ 18–23
#   budget <= RD_TIER_MEDIUM_MAX → 2–3 breakpoints (medium) probe_budget ≈ 30–35
#   budget >  RD_TIER_MEDIUM_MAX → 4–6 breakpoints (hard)   probe_budget ≈ 40–65
RD_TIER_EASY_MAX: int = 23
RD_TIER_MEDIUM_MAX: int = 35

# Minimum interval lengths enforced in both the hypothesis validator and the
# config validator (micro_interval_len_range lower bound must be >= RD_MIN_AFFINE_LEN).
# Quadratic needs one extra point over affine for unique fitting mod m.
RD_MIN_AFFINE_LEN: int = 3   # affine_mod and affine_mod_popcount
RD_MIN_QUADRATIC_LEN: int = 4  # quadratic_mod


def _is_prime_int(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    d = 3
    while d * d <= n:
        if n % d == 0:
            return False
        d += 2
    return True


class LLMConfig(BaseModel):
    """Configuration for LiteLLM calls.

    LiteLLM typically reads provider credentials from environment variables
    (e.g., OPENAI_API_KEY). This project uses `python-dotenv` to load those
    variables from a `.env` file.
    """

    model: str = "gpt-4o-mini"
    temperature: float | None = None
    max_tokens: int | None = None
    timeout_s: float = Field(60.0, ge=1.0)

    # Concurrency + resilience
    max_concurrent_requests: int = Field(8, ge=1)
    max_retries: int = Field(6, ge=0)
    backoff_base_s: float = Field(0.5, ge=0.0)
    backoff_max_s: float = Field(30.0, ge=0.0)
    jitter_s: float = Field(0.2, ge=0.0)

    # Persistent cache path (defaults under out_dir/cache/llm_cache.sqlite)
    cache_enabled: bool = True

    # Base decoding seed. Left unset the server samples freely (all runs before the
    # cap-boundary-carryover experiment did). When set, the benchmark derives a per-task
    # seed from it and the task id, so two cells that differ only in runtime start each
    # task from the same sampler state.
    seed: int | None = None

    @model_validator(mode="after")
    def _check_backoff(self):
        if self.backoff_max_s < self.backoff_base_s:
            raise ValueError("backoff_max_s must be >= backoff_base_s")
        return self


class KnapsackConfig(BaseModel):
    num_tasks: int = Field(100, ge=0)

    n_items_range: Tuple[int, int] = (10, 30)
    weight_range: Tuple[int, int] = (1, 20)
    value_range: Tuple[int, int] = (1, 50)

    classes: List[str] = Field(default_factory=lambda: ["A", "B", "C"])
    num_allowed_classes: int = 10

    # Capacity is sampled as ratio * total_weight(allowed_class_items)
    capacity_ratio_range: Tuple[float, float] = (0.35, 0.6)

    # Re-sample instances until the optimal solution selects >= 1 item.
    force_nonempty_optimum: bool = True

    @model_validator(mode="after")
    def _validate_ranges(self):
        lo, hi = self.n_items_range
        if lo <= 0 or hi < lo:
            raise ValueError("n_items_range must be (lo>0, hi>=lo)")
        for name, rng in [
            ("weight_range", self.weight_range),
            ("value_range", self.value_range),
        ]:
            rlo, rhi = rng
            if rlo <= 0 or rhi < rlo:
                raise ValueError(f"{name} must be (lo>0, hi>=lo)")
        clo, chi = self.capacity_ratio_range
        if not (0.0 < clo <= chi <= 1.0):
            raise ValueError("capacity_ratio_range must be within (0,1]")
        if not self.classes:
            raise ValueError("classes must be non-empty")
        return self


class NavigationConfig(BaseModel):
    num_tasks: int = Field(100, ge=0)

    # K controls the shortest path length from start to goal.
    horizon_range: Tuple[int, int] = (8, 25)

    # Additional nodes to add beyond the base path nodes.
    extra_nodes_range: Tuple[int, int] = (0, 15)

    # Additional edges added as a fraction of the maximum possible extra edges.
    extra_edge_factor_range: Tuple[float, float] = (0.1, 0.35)

    # Suggested step budget exposed in `public`.
    max_steps_multiplier: float = 3.0

    # Probability that a trap node is placed on the path to goal.
    trap_prob: float = Field(0.3, ge=0.0, le=1.0)

    # Multiplier applied to the computed probe budget (after the 30%-of-nodes cap).
    # Values < 1.0 tighten the budget; use to make harder tiers genuinely harder
    # even when their larger graphs would otherwise yield a more generous cap.
    probe_budget_multiplier: float = Field(1.0, gt=0.0, le=2.0)

    # Inclusive range for the number of decoy keys per instance. Decoys carry IDs
    # `D0/D1/...` and pollute the `key_nearby` hint so it is a region signal
    # rather than a singleton L0 locator.
    decoy_count_range: Tuple[int, int] = (1, 3)

    # When True, instances must satisfy `_check_state_pressure` (decoy near L0
    # hint, no singleton-L0 hint set, probe budget tighter than full audit).
    # Opt-in per tier; default False preserves legacy behavior.
    reject_singleton_l0_hint: bool = False

    # Inclusive range for the number of traps per instance. When None, the legacy
    # `trap_prob` path applies (0 or 1 trap with probability `trap_prob`). When
    # set, exactly k = randint(lo, hi) traps are placed (with graceful fallback
    # if the candidate pool is exhausted; surfaces as state_pressure_warning).
    trap_count_range: Optional[Tuple[int, int]] = None

    # Hop radius for the `key_nearby` hint. The hint is True for any node whose
    # k-hop neighborhood contains a key (real or decoy). Larger radius makes the
    # hint a wider region signal, weaker as a singleton localizer.
    key_hint_radius: int = Field(1, ge=1, le=3)

    # Hard cap on probe budget as a fraction of total nodes. Used in
    # `_compute_probe_budget` to prevent the budget from scaling unboundedly
    # with horizon on dense graphs. Tighter values force strategic allocation.
    probe_cap_fraction: float = Field(0.3, gt=0.0, le=1.0)

    # Nodes one `neighbors()` call may query: the reconstruction bandwidth of the free
    # topology channel. 50 is the batched interface all existing tasks use; 1 makes
    # rebuilding an n-node map cost n calls, crossing a tight per-turn tool-call cap.
    neighbors_batch_max: int = Field(50, ge=1)

    @model_validator(mode="after")
    def _validate_ranges(self):
        hlo, hhi = self.horizon_range
        if hlo <= 0 or hhi < hlo:
            raise ValueError("horizon_range must be (lo>0, hi>=lo)")
        nlo, nhi = self.extra_nodes_range
        if nlo < 0 or nhi < nlo:
            raise ValueError("extra_nodes_range must be (lo>=0, hi>=lo)")
        elo, ehi = self.extra_edge_factor_range
        if not (0.0 <= elo <= ehi <= 1.0):
            raise ValueError("extra_edge_factor_range must be within [0,1]")
        dlo, dhi = self.decoy_count_range
        if dlo < 1 or dhi < dlo:
            raise ValueError("decoy_count_range must be (lo>=1, hi>=lo)")
        if self.trap_count_range is not None:
            tlo, thi = self.trap_count_range
            if tlo < 0 or thi < tlo:
                raise ValueError("trap_count_range must be (lo>=0, hi>=lo)")
        return self


class RuleDiagnosisConfig(BaseModel):
    num_tasks: int = Field(100, ge=0)

    # Base rule output is always modulo m.
    mod_m_choices: List[int] = Field(default_factory=lambda: [7, 8, 9, 11, 13])

    # Public x-domain.
    domain_range: Tuple[int, int] = (0, 999)

    # Allowed test_input calls.
    probe_budget_range: Tuple[int, int] = (18, 20)

    # Minimum number of x values in every generated interval.  Must be >= 4
    # so the model can always probe 4 consecutive points starting at a new
    # interval boundary without bleeding into the adjacent interval.
    min_interval_len: int = Field(6, ge=4)

    # Number of "micro" intervals to inject per task (hard-difficulty source).
    # Micro intervals have length in [micro_interval_len_range[0], micro_interval_len_range[1]],
    # which is shorter than min_interval_len, making them harder to detect and fit.
    micro_interval_count: int = Field(0, ge=0)
    micro_interval_len_range: Tuple[int, int] = (3, 4)

    # Rule library: pre-sample L distinct (sub_family, params) rules and assign
    # intervals from that library (no adjacent same-index).  With a small library
    # the model can exploit recurrences: if it identifies rule A in interval 1 it
    # can reuse those params in interval 3 without extra probes.
    # 0 = disabled (independent sampling per interval, original behaviour).
    # >= 2 required when non-zero.
    rule_library_size: int = Field(0, ge=0)

    # If true, the task prompt includes "family hint: stepwise_composition".
    reveal_family_in_public: bool = False

    # Public cap for exception overrides in submitted hypotheses.
    max_hypothesis_exceptions: int = Field(0, ge=0)

    # If True, `piecewise_affine_mod(x) = (a * (x mod k) + b) % m` is included
    # in the per-interval sub-family pool. Adds a 4th hypothesis family the
    # agent must discriminate. Only emitted for intervals long enough to
    # disambiguate the period k (≥ 2*k + 2 points). Default False preserves
    # legacy 3-family behavior on easy/hard tiers.
    include_piecewise_affine_mod: bool = False

    # Inclusive range for the number of exception points injected per task.
    # Each exception is an x whose f(x) overrides the underlying piecewise rule.
    # Drives CEGIS depth: the agent's first hypothesis (without exceptions) fails
    # `check()` on these points, forcing iterative refinement. (0, 0) — the
    # legacy default — disables exceptions entirely. The high bound must not
    # exceed `max_hypothesis_exceptions` (otherwise no valid hypothesis exists).
    exceptions_count_range: Tuple[int, int] = (0, 0)

    # When True, sampled instances must have >=2 breakpoints (i.e. >=3 intervals)
    # and a probe budget that does not cover the full domain. Defensive guard;
    # the breakpoint count is already determined by `probe_budget_range` via the
    # tier logic in `sample_rule_diagnosis_instance`. Easy tier opts in to catch
    # silent regressions if the tier thresholds are ever edited.
    require_multi_breakpoint: bool = False

    # Per-turn cap on test_input() calls. Legacy "throttle" lever, retained only
    # as a possible ablation: it bounds work per turn far below the generic
    # max_tool_calls budget, which makes CodeAct less expressive rather than the
    # task more stateful. The shipped mechanism is `online` (below). None (the
    # default) leaves the pull-based task one-turn-solvable.
    probes_per_turn: Optional[int] = Field(None, gt=0)

    # === Online (streaming system-identification) mode ===
    # When True the task is reframed as online estimation: instead of pull-based
    # test_input()/check(), the hidden rule is observed through a NOISY stream that
    # arrives one batch per turn (observe()). The later data does not exist yet at
    # turn 1, so no agent can one-turn it — multi-turn structure is a property of
    # the task, not a per-turn throttle. The cheap policy accumulates running
    # sufficient statistics (per-x vote tallies) in interpreter state, which a
    # stateless runtime wipes. This is the genuine cross-turn-state mechanism.
    online: bool = False
    # Size of the fixed probe grid the stream samples from in online mode. The
    # same x-values recur across turns (with fresh noise), so per-x majority vote
    # over accumulated samples is what denoises — the grid is what makes repeated
    # sampling, and therefore accumulation, meaningful. Boundaries are recoverable
    # only to grid resolution, so the online score is functional-match dominated.
    stream_grid_size: int = Field(48, ge=2)
    # Number of noisy (x, y) samples delivered per turn in online mode.
    stream_batch_size: int = Field(24, ge=1)
    # Per-label corruption probability in online mode: each streamed y is replaced
    # by a uniform-random value in [0, m) with this probability. Forces repeated
    # sampling per region (denoising) so accumulation is genuinely required.
    label_noise: float = Field(0.2, ge=0.0, lt=1.0)
    # Online table-submission mode. When True (online only), the agent submits its
    # DENOISED per-grid-point table {x: y} instead of a fitted piecewise rule, and
    # the score is grid-recovery accuracy. This removes the hand-written fitter from
    # the scored path, so the score measures pure cross-turn state aggregation (the
    # runtime-relevant ability) rather than fitting skill.
    online_submit_table: bool = False

    @model_validator(mode="after")
    def _validate_ranges(self):
        if not self.mod_m_choices:
            raise ValueError("mod_m_choices must be non-empty")
        for m in self.mod_m_choices:
            if not isinstance(m, int) or m < 2:
                raise ValueError("mod_m_choices entries must be integers >= 2")
        # quadratic sub-intervals are only generated when m is prime; warn if no
        # prime is available so all instances will silently omit quadratic.
        if not any(_is_prime_int(m) for m in self.mod_m_choices):
            raise ValueError(
                "mod_m_choices contains no prime — quadratic sub-intervals will "
                "never be generated; add at least one prime (e.g. 7, 11, or 13)"
            )

        dx0, dx1 = self.domain_range
        if dx1 < dx0:
            raise ValueError("domain_range must be (lo<=hi)")
        domain_size = dx1 - dx0 + 1
        if domain_size < 2:
            raise ValueError("domain_range must contain at least two points")

        pa, pb_max = self.probe_budget_range
        if pa <= 0 or pb_max < pa:
            raise ValueError("probe_budget_range must be (lo>0, hi>=lo)")
        if pa < 4:
            raise ValueError("probe_budget_range lower bound must be >= 4")

        mlo, mhi = self.micro_interval_len_range
        if mlo < RD_MIN_AFFINE_LEN or mhi < mlo or mhi >= self.min_interval_len:
            raise ValueError(
                f"micro_interval_len_range must satisfy {RD_MIN_AFFINE_LEN} <= lo <= hi < min_interval_len "
                f"(lo must be >= RD_MIN_AFFINE_LEN={RD_MIN_AFFINE_LEN} so micro intervals are representable)"
            )

        if self.rule_library_size == 1:
            raise ValueError("rule_library_size must be 0 (disabled) or >= 2 (a library of 1 is degenerate)")

        # Compute the breakpoint range that can occur at runtime.
        # bp_lo / bp_hi are the min/max breakpoint counts across all sampled budgets.
        pb_lo, pb_max = self.probe_budget_range[0], self.probe_budget_range[1]
        if pb_lo <= RD_TIER_EASY_MAX:
            bp_lo = 1
        elif pb_lo <= RD_TIER_MEDIUM_MAX:
            bp_lo = 2
        else:
            bp_lo = 4
        if pb_max <= RD_TIER_EASY_MAX:
            bp_hi = 1
        elif pb_max <= RD_TIER_MEDIUM_MAX:
            bp_hi = 3
        else:
            bp_hi = 6

        # Micro intervals must fit strictly in interior positions (never first/last).
        # Use bp_lo (smallest runtime breakpoint count) to get the tightest interior
        # slot count; that is the binding constraint.
        interior_slots_min = max(0, bp_lo - 1)  # bp_lo+1 segs → bp_lo-1 interior
        if self.micro_interval_count > interior_slots_min:
            raise ValueError(
                f"micro_interval_count={self.micro_interval_count} exceeds interior "
                f"slot count ({interior_slots_min}) for the smallest possible "
                f"breakpoint count at runtime ({bp_lo}); reduce micro_interval_count "
                f"or raise probe_budget_range lower bound"
            )

        # Domain-width check: use bp_hi (most segments) and micro_hi (longest micro)
        # so the guarantee holds for every (budget, micro-len) combination.
        num_segs_max = bp_hi + 1
        num_micro = self.micro_interval_count
        num_normal_max = num_segs_max - num_micro
        # stars-and-bars on normal segs: remaining_for_normal >= num_normal
        # → domain_size >= num_micro*mhi + num_normal*(min_len+1)
        min_domain_width = num_micro * mhi + num_normal_max * (self.min_interval_len + 1)
        if domain_size < min_domain_width:
            raise ValueError(
                f"domain_range too narrow: budget up to {pb_max} gives up to "
                f"{bp_hi} breakpoint(s) ({num_segs_max} segments), "
                f"{num_micro} micro (max len {mhi}) and {num_normal_max} normal "
                f"(min len {self.min_interval_len}); domain width must be >= "
                f"{min_domain_width}, got {domain_size}"
            )

        # Reject ranges that straddle a tier boundary.  A straddle means some
        # generated instances will have 1 breakpoint and others 2–3 (or 2–3 vs
        # 4–6), producing inconsistent difficulty within the same config.
        def _tier(b: int) -> int:
            if b <= RD_TIER_EASY_MAX:
                return 0
            if b <= RD_TIER_MEDIUM_MAX:
                return 1
            return 2

        if _tier(pa) != _tier(pb_max):
            raise ValueError(
                f"probe_budget_range [{pa}, {pb_max}] straddles a breakpoint-tier "
                f"boundary (≤{RD_TIER_EASY_MAX}→1 bp, "
                f"≤{RD_TIER_MEDIUM_MAX}→2–3 bp, "
                f">{RD_TIER_MEDIUM_MAX}→4–6 bp). "
                f"Keep the range within one tier for consistent difficulty."
            )

        elo, ehi = self.exceptions_count_range
        if elo < 0 or ehi < elo:
            raise ValueError("exceptions_count_range must be (lo>=0, hi>=lo)")
        if ehi > self.max_hypothesis_exceptions:
            raise ValueError(
                f"exceptions_count_range upper {ehi} exceeds "
                f"max_hypothesis_exceptions {self.max_hypothesis_exceptions} — "
                f"the generator would create exceptions the agent cannot submit"
            )

        # Online mode and the legacy per-turn throttle are mutually exclusive:
        # online drops test_input() entirely (the stream is push-based), so a
        # probes_per_turn cap would be dead and confusing if both were set.
        if self.online and self.probes_per_turn is not None:
            raise ValueError(
                "online=True drops pull-based test_input(); set probes_per_turn=None "
                "(the per-turn throttle does not apply to the streaming task)"
            )

        # Table-submission is a sub-mode of online (it changes what submit() takes
        # and how scoring works); it is meaningless without the stream.
        if self.online_submit_table and not self.online:
            raise ValueError(
                "online_submit_table=True requires online=True"
            )

        return self


class Settings(BaseModel):
    """Top-level generator settings.

    The generator is designed to be resumable: already generated tasks are skipped.
    """

    out_dir: Path = Path("out")
    run_name: str = "run"
    seed_start: int = 0

    llm_enabled: bool = False

    # Number of concurrent *task workers*.
    max_workers: int = Field(16, ge=1)

    llm: LLMConfig = Field(default_factory=LLMConfig)  # type: ignore
    knapsack: KnapsackConfig = Field(default_factory=KnapsackConfig)  # type: ignore
    navigation: NavigationConfig = Field(
        default_factory=NavigationConfig  # type: ignore
    )
    rule_diagnosis: RuleDiagnosisConfig = Field(
        default_factory=RuleDiagnosisConfig  # type: ignore
    )

    cache_db_path: Path | None = Path("cache/llm_cache.sqlite")

    @model_validator(mode="after")
    def _normalize_paths(self):
        self.out_dir = Path(self.out_dir)
        return self

    def effective_cache_path(self) -> Path:
        if self.cache_db_path is not None:
            return self.cache_db_path
        return self.out_dir / "cache" / "llm_cache.sqlite"

    @classmethod
    def load(cls, path: str | Path) -> "Settings":
        p = Path(path)
        data = json.loads(p.read_text(encoding="utf-8"))
        return cls.model_validate(data)

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(self.model_dump(mode="json"), indent=2, ensure_ascii=False)
            + "\n",
            encoding="utf-8",
        )
