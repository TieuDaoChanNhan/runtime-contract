import asyncio
import json
import random
import ast
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from pydantic import BaseModel

from codeact_runtime.codeact.tool import Tool
from codeact_runtime.families.base import TaskData, TaskResult

from ..config import (
    RuleDiagnosisConfig, RD_TIER_EASY_MAX, RD_TIER_MEDIUM_MAX,
    RD_MIN_AFFINE_LEN, RD_MIN_QUADRATIC_LEN, _is_prime_int,
)

# Witness window for check(): the witness is sampled from at most this many
# points past the start of the first wrong run.  Caps binary-search cost at
# log2(_WITNESS_WINDOW) probes per boundary regardless of interval length.
_WITNESS_WINDOW: int = 32

HypothesisDict = dict[str, Any]
CheckResult = dict[str, Any]


class RuleDiagnosisPublic(BaseModel):
    m: int
    probe_budget: int
    x_domain: Dict[str, int]
    max_exceptions: int
    family_hint: Optional[str] = None
    # Per-turn probe cap (see RuleDiagnosisEnv.probes_per_turn). None = unlimited.
    probes_per_turn: Optional[int] = None
    # Online (streaming) mode — see RuleDiagnosisEnv / RuleDiagnosisConfig.
    online: bool = False
    stream_grid_size: int = 48
    stream_batch_size: int = 24
    label_noise: float = 0.2
    # Online table-submission mode (see RuleDiagnosisConfig.online_submit_table):
    # agent submits its denoised {x: y} table; score = grid-recovery accuracy.
    online_submit_table: bool = False


class RuleDiagnosisPrivate(BaseModel):
    family: str
    params: Dict[str, Any]
    exceptions: Dict[int, int]


class RuleDiagnosisReference(BaseModel):
    true_hypothesis: Dict[str, Any]


class RuleDiagnosisTaskData(
    TaskData[RuleDiagnosisPublic, RuleDiagnosisPrivate, RuleDiagnosisReference]
):
    pass


@dataclass
class RuleDiagnosisEnv:
    seed: int
    m: int
    probe_budget: int
    x_min: int
    x_max: int
    family: str
    params: dict[str, Any]
    exceptions: dict[int, int]
    max_exceptions: int
    min_interval_len: int = 6
    public_family_hint: str | None = None
    # Per-turn probe cap (legacy throttle, retained for ablation only — see
    # RuleDiagnosisConfig.probes_per_turn). None = unlimited.
    probes_per_turn: int | None = None

    # === Online (streaming system-ID) mode ===
    # When True the task drops pull-based test_input()/check() in favor of a
    # push-based noisy observation stream: observe() returns this turn's batch of
    # samples drawn from a fixed probe grid, each label corrupted with probability
    # label_noise. The stream cursor (stream_round) advances once per turn via the
    # interpreter's on_turn_start hook, so the data for later turns does not exist
    # yet at turn 1 — the task is genuinely multi-turn (temporal gating), not
    # throttled. Denoising requires accumulating per-x votes across turns, which a
    # stateless runtime wipes.
    online: bool = False
    stream_grid_size: int = 48
    stream_batch_size: int = 24
    label_noise: float = 0.2
    # Table-submission mode (online only): score = denoised-grid accuracy, fitter
    # removed from the scored path. See RuleDiagnosisConfig.online_submit_table.
    online_submit_table: bool = False
    # Stream cursor; -1 so the first on_turn_start advances it to round 0.
    stream_round: int = -1

    probes_used: int = 0
    probes_used_this_turn: int = 0
    done: bool = False
    solved: bool = False
    last_submit_match_ratio: float | None = None
    last_submit_interval_count: int | None = None
    last_boundary_f1: float | None = None
    last_family_acc: float | None = None
    first_unsolved_micro: bool | None = None
    probed_xs: set[int] = field(default_factory=set)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    @classmethod
    def from_task(cls, task_dict: dict[str, Any]) -> "RuleDiagnosisEnv":
        task = RuleDiagnosisTaskData.model_validate(task_dict)
        return cls(
            seed=task.seed,
            m=task.public.m,
            probe_budget=task.public.probe_budget,
            x_min=task.public.x_domain["min"],
            x_max=task.public.x_domain["max"],
            family=task.private.family,
            params=task.private.params,
            exceptions=task.private.exceptions,
            max_exceptions=task.public.max_exceptions,
            min_interval_len=task.difficulty.get("min_interval_len", 6),
            public_family_hint=task.public.family_hint,
            probes_per_turn=task.public.probes_per_turn,
            online=task.public.online,
            stream_grid_size=task.public.stream_grid_size,
            stream_batch_size=task.public.stream_batch_size,
            label_noise=task.public.label_noise,
            online_submit_table=task.public.online_submit_table,
        )

    def get_goal_prompt(self) -> str:
        if self.online:
            if self.online_submit_table:
                return self._get_online_table_goal_prompt()
            return self._get_online_goal_prompt()

        maybe_family = ""
        if self.public_family_hint is not None:
            maybe_family = f"- Family hint: {self.public_family_hint}\n"

        maybe_per_turn = ""
        per_turn_rule = ""
        if self.probes_per_turn is not None:
            maybe_per_turn = (
                f"- Per-turn probe cap: at most {self.probes_per_turn} test_input(x) "
                "calls per turn\n"
            )
            per_turn_rule = (
                f"- PER-TURN CAP: only {self.probes_per_turn} test_input(x) calls succeed "
                "each turn; further calls raise an error. You CANNOT probe the whole "
                "domain in one turn — you must spread probing across turns and carry "
                "your accumulated probe results (a dict of x->f(x)) plus your confirmed "
                "intervals in interpreter state between turns.\n"
            )

        return (
            "Goal\n"
            "- Rule Diagnosis: infer a hidden integer function f(x) and submit a matching hypothesis.\n\n"
            "What you are given:\n"
            f"- Domain: integers x in [{self.x_min}, {self.x_max}]\n"
            f"- Modulus m: {self.m}\n"
            f"- Probe budget: {self.probe_budget} calls to test_input(x)\n"
            f"{maybe_per_turn}"
            f"{maybe_family}"
            "\nRules:\n"
            "- test_input(x): probe f(x) for one in-domain integer; costs 1 from your budget.\n"
            f"{per_turn_rule}"
            "- check(hypothesis): free oracle — returns {'status':'pass'} or {'status':'fail','x':W,'y_pred':Y} "
            "where W is a witness point near (within ~32 steps of) the start of the first wrong run. "
            "The true boundary B satisfies last_confirmed_good < B <= W. "
            "Does NOT end the task. Call it as many times as you need.\n"
            "- submit(hypothesis): scores and ENDS the task immediately — irreversible. Call exactly once, when finished or out of budget.\n"
            "- If check() or submit() returns {'status':'error',...} the hypothesis schema was invalid "
            "and the task is NOT over — fix the hypothesis and call again.\n"
            "=== HIDDEN RULE TYPES ===\n"
            "The function f(x) in each interval is one of:\n"
            "1. `affine_mod`: (a*x + b) % m  — 2 unknowns, need 2+ probes to fit\n"
            "2. `quadratic_mod`: (a*x^2 + b*x + c) % m  — 3 unknowns, need 3+ probes to fit\n"
            "3. `affine_mod_popcount`: (a*x + b + popcount(x)) % m  [popcount(x) == x.bit_count()]  — 2 unknowns, need 2+ probes to fit\n"
            "4. `piecewise_affine_mod`: (a*(x mod k) + b) % m  with k ∈ [3, 7]  — 3 unknowns (a, b, k), need ≥2*k probes to fit. Output is periodic with period k.\n"
            "CONSTRAINT: a ≠ 0 is required for every sub-family. "
            "The validator will reject any hypothesis where a param 'a' equals 0 "
            "with {'status':'error','message':'invalid hypothesis: param out of range'}.\n\n"
            "=== STRATEGY GUIDE (CRITICAL) ===\n"
            "1. STEP-BY-STEP EXECUTION: Do NOT write a single massive while-loop to solve the entire domain locally. Act interactively: probe a small batch of points, print results, read them, then decide the next step.\n"
            "2. FIT THE FIRST INTERVAL: Probe 4 consecutive points starting at x_min (e.g. x=0,1,2,3). Use the outputs to fit formula1 by solving the linear system mod m for each candidate family. Verify your fit on a 5th point before proceeding.\n"
            "3. ORACLE CHECK: Build a working hypothesis covering [x_min, x_max] using formula1 for the whole domain. Call check() — it is free. If it returns pass, call submit() and you are done.\n"
            "4. FINDING THE NEXT BOUNDARY — BINARY SEARCH + LOCAL FIT:\n"
            "   check() returns a witness W inside the first wrong run — NOT the boundary itself.\n"
            "   The true boundary B satisfies last_confirmed_good < B ≤ W.\n"
            "   Step A — LOCATE THE BOUNDARY by binary searching [last_confirmed_good+1, W]:\n"
            "      W is within ~32 steps of the true boundary, so this search costs at most 5 probes.\n"
            "      Let lo = last_confirmed_good + 1, hi = W.\n"
            "      While hi - lo > 1: probe mid = (lo+hi)//2.\n"
            "        If hypothesis(mid) == f(mid): lo = mid  (boundary is above mid)\n"
            "        Else:                          hi = mid  (boundary is at or below mid)\n"
            "      After the loop, B = hi (first wrong point). Cost: ≤5 probes.\n"
            "   Step B — FIT THE NEW INTERVAL starting at B:\n"
            "      Probe B and B+1 (2 probes). Try affine_mod and affine_mod_popcount on (B, B+1).\n"
            "      If exactly one 2-parameter family fits, probe B+2 to verify.\n"
            "      If ambiguous or verification fails, probe B+3 and fit all three families on B..B+3.\n"
            "   Step C — EXTEND AND REPEAT:\n"
            "      Build hypothesis with the new interval covering [B, x_max] and call check().\n"
            "      The next witness W' tells you where the next wrong run starts. Repeat from Step A\n"
            "      (now last_confirmed_good = B-1, search [B, W']).\n"
            "   Micro-interval note: if after fitting you call check() and get W' ≤ B+4,\n"
            "      the interval [B, B+(W'-B-1)] is short (2–4 pts). Refit those points, shrink the\n"
            "      interval, and continue from W'.\n"
            "5. SYSTEMATIC PROGRESSION: Repeat step 4 for each boundary until check() returns pass,\n"
            "   then call submit() immediately. Binary search costs ~log2(interval_len) probes per\n"
            "   boundary; fitting costs 3–4 probes. Manage budget carefully.\n"
            f"6. STRICT SCHEMA: `hyp = {{\"m\": {self.m}, \"family\": \"stepwise_composition\", \"intervals\": [...]}}`. Each interval: `{{\"x_min\": .., \"x_max\": .., \"sub_family\": .., \"params\": {{..}}}}`.\n"
            "7. EXACT STRINGS: For `sub_family`, use exactly `'affine_mod'`, `'quadratic_mod'`, `'affine_mod_popcount'`, or `'piecewise_affine_mod'`. No abbreviations. For `piecewise_affine_mod`, params must include `{a, b, k}` with k ∈ [3, 7].\n"
            "8. BUDGET AWARENESS: Each probe is precious. Use check() (free) for validation, not test_input(). When the budget runs low, build the best hypothesis you can and call submit() once.\n"
        )

    def _get_online_goal_prompt(self) -> str:
        return (
            "Goal\n"
            "- Online Rule Diagnosis: infer a hidden integer function f(x) from a "
            "NOISY observation stream, then submit a matching hypothesis.\n\n"
            "What you are given:\n"
            f"- Domain: integers x in [{self.x_min}, {self.x_max}]\n"
            f"- Modulus m: {self.m}\n"
            f"- Each turn, observe() returns a dict {{'round': n, 'batch': [[x, y], ...]}} "
            f"with ~{self.stream_batch_size} samples. Each y equals f(x) but is "
            f"CORRUPTED to a random value with probability ~{self.label_noise:.2f}.\n"
            "- The samples are drawn from a FIXED set of probe points, so the SAME "
            "x-values recur across turns with fresh noise each time.\n\n"
            "Rules:\n"
            "- observe(): returns ONLY this turn's batch. The next batch is not "
            "available until the next turn — you cannot get more data by calling "
            "observe() repeatedly in one turn (it returns the same batch). There is "
            "NO test_input() and NO check() oracle.\n"
            "- submit(hypothesis): scores against the TRUE (clean) f over the whole "
            "domain and ENDS the task immediately — irreversible. Call exactly once.\n\n"
            "=== HOW TO SOLVE (CRITICAL) ===\n"
            "1. ACCUMULATE ACROSS TURNS. One turn's batch is too noisy and too sparse "
            "to fit the rule. You MUST keep a running tally of every (x, y) you have "
            "seen — e.g. `votes[x][y] += 1` — in a variable that persists between "
            "turns, and fold each new observe() batch into it.\n"
            "2. DENOISE BY MAJORITY VOTE. For each probe point x, the true f(x) is the "
            "value you have seen most often; corrupted labels are spread randomly, so "
            "they lose to the truth once you have a few samples per x. Aggregate over "
            "MANY turns until each point has a clear majority.\n"
            "3. FIT THE PIECEWISE RULE. Once your denoised points are stable, sort them "
            "by x and fit consecutive runs. f is stepwise: each segment is one of:\n"
            "   - `affine_mod`: (a*x + b) % m   (a != 0)\n"
            "   - `affine_mod_popcount`: (a*x + b + popcount(x)) % m   [popcount(x) == x.bit_count()]   (a != 0)\n"
            "   A boundary is where the fitted (a, b) stops predicting the next point.\n"
            "4. SUBMIT ONCE. Build intervals covering the whole domain and submit. "
            f"Schema: `{{\"m\": {self.m}, \"family\": \"stepwise_composition\", "
            "\"intervals\": [{\"x_min\": .., \"x_max\": .., \"sub_family\": .., "
            "\"params\": {..}}, ...]}}` with no gaps or overlaps. Use the exact strings "
            "`'affine_mod'` and `'affine_mod_popcount'`.\n"
            "WARNING: if your interpreter state is empty at the start of a turn (your "
            "tally is gone), do not submit from a single noisy batch — re-accumulate.\n"
        )

    def _get_online_table_goal_prompt(self) -> str:
        return (
            "Goal\n"
            "- Online Signal Recovery: a hidden integer function f(x) is observed "
            "through a NOISY stream. Recover its value at each probe point and submit "
            "your denoised table.\n\n"
            "What you are given:\n"
            f"- Modulus m: {self.m} (every value is an integer in [0, {self.m - 1}])\n"
            f"- Each turn, observe() returns {{'round': n, 'batch': [[x, y], ...]}} "
            f"with ~{self.stream_batch_size} samples. Each y equals f(x) but is "
            f"CORRUPTED to a random value with probability ~{self.label_noise:.2f}.\n"
            "- Samples are drawn from a FIXED set of probe points, so the SAME "
            "x-values recur across turns with fresh noise each time.\n\n"
            "Rules:\n"
            "- observe(): returns ONLY this turn's batch. Calling it again in the "
            "same turn returns the SAME batch; the next batch arrives next turn. "
            "There is NO test_input() and NO check() oracle.\n"
            "- submit(table): submit a Python dict {x: y} mapping each probe point x "
            "to your best estimate of f(x). Scores the fraction of probe points you "
            "recover EXACTLY against the true (clean) f, then ENDS the task. Call "
            "exactly once.\n\n"
            "=== HOW TO SOLVE (CRITICAL) ===\n"
            "1. ACCUMULATE ACROSS TURNS. One turn's batch is too noisy. Keep a "
            "running tally of every (x, y) seen — e.g. `votes[x][y] += 1` — in a "
            "variable that persists between turns, and fold each new batch into it.\n"
            "2. DENOISE BY MAJORITY VOTE. For each probe point x, your estimate is "
            "the value seen most often; random corruptions lose to the truth once you "
            "have several samples per x. Keep observing until every point has a clear "
            "majority and you have covered all probe points.\n"
            "3. SUBMIT THE TABLE. When your per-x majorities are stable, build "
            "`table = {x: majority_value_for_x}` over ALL probe points you have seen "
            "and call submit(table). NO rule-fitting is required — just the denoised "
            "values.\n"
            "WARNING: if your interpreter state is empty at the start of a turn (your "
            "tally is gone), do NOT submit from a single noisy batch — re-accumulate "
            "from the observations in the conversation so far.\n"
        )

    def get_tools(self) -> list[Tool]:
        if self.online:
            submit_tool = (
                SubmitTableTool(self) if self.online_submit_table else SubmitRuleTool(self)
            )
            return [ObserveTool(self), submit_tool]
        return [TestInputTool(self), CheckTool(self), SubmitRuleTool(self)]

    def evaluate(self) -> TaskResult:
        efficiency = (
            1.0 - (self.probes_used / self.probe_budget)
            if self.probe_budget > 0
            else 0.0
        )
        efficiency = max(0.0, min(1.0, efficiency))

        functional = self.last_submit_match_ratio if self.last_submit_match_ratio is not None else 0.0
        boundary_f1 = self.last_boundary_f1 if self.last_boundary_f1 is not None else 0.0
        family_acc = self.last_family_acc if self.last_family_acc is not None else 0.0

        true_intervals = (
            self.params.get("intervals", []) if self.family == "stepwise_composition" else []
        )
        true_iv = len(true_intervals)
        sub_iv = self.last_submit_interval_count or 0
        complexity_penalty = 0.05 * max(0, sub_iv - true_iv)

        if self.done and self.online_submit_table:
            # Table mode: score IS the denoised-grid accuracy (pure aggregation),
            # with the fitter removed from the scored path.
            score = functional
        elif self.done:
            score = (
                0.60 * functional
                + 0.25 * boundary_f1
                + 0.15 * family_acc
                - complexity_penalty
            )
        else:
            score = 0.0

        return TaskResult(
            is_solved=self.solved,
            score=max(0.0, min(1.0, score)),
            metrics={
                "probes_used": self.probes_used,
                "probe_budget": self.probe_budget,
                "efficiency": efficiency,
                "done": self.done,
                "submit_match_ratio": functional,
                "boundary_f1": boundary_f1,
                "family_acc": family_acc,
                "complexity_penalty": complexity_penalty,
                "true_interval_count": true_iv,
                "num_micro_true": sum(
                    1 for iv in true_intervals if iv["x_max"] - iv["x_min"] + 1 < self.min_interval_len
                ),
                "recurrence_count_true": _count_recurrences(true_intervals),
                "submitted_interval_count": sub_iv,
                "first_unsolved_micro": self.first_unsolved_micro,
            },
        )

    def _base_output(self, x: int, family: str, params: dict[str, Any]) -> int:
        try:
            return _eval_base(family, params, x, self.m)
        except ValueError:
            raise ValueError("invalid hypothesis: unsupported family") from None

    def _f(self, x: int) -> int:
        if x in self.exceptions:
            return self.exceptions[x]
        return self._base_output(x, self.family, self.params)

    def _validate_param_value(self, value: Any) -> int:
        if not isinstance(value, int) or isinstance(value, bool) or not (0 <= value < self.m):
            raise ValueError("invalid hypothesis: param out of range")
        return value

    def _normalize_hypothesis(
        self, hypothesis: dict[str, Any]
    ) -> tuple[str, dict[str, Any], dict[int, int]]:
        if not isinstance(hypothesis, dict):
            raise ValueError("invalid hypothesis: bad schema")
        if "m" not in hypothesis or "family" not in hypothesis:
            raise ValueError("invalid hypothesis: bad schema")

        m = hypothesis["m"]
        if not isinstance(m, int) or m != self.m:
            raise ValueError("invalid hypothesis: wrong modulus")

        family = hypothesis["family"]
        if not isinstance(family, str):
            raise ValueError("invalid hypothesis: bad schema")

        params: dict[str, Any] = {}
        if family == "affine_mod":
            if "a" not in hypothesis or "b" not in hypothesis:
                raise ValueError("invalid hypothesis: bad schema")
            params = {
                "a": self._validate_param_value(hypothesis["a"]),
                "b": self._validate_param_value(hypothesis["b"]),
            }
            if params["a"] == 0:
                raise ValueError("invalid hypothesis: param out of range")
        elif family == "quadratic_mod":
            if any(k not in hypothesis for k in ("a", "b", "c")):
                raise ValueError("invalid hypothesis: bad schema")
            params = {
                "a": self._validate_param_value(hypothesis["a"]),
                "b": self._validate_param_value(hypothesis["b"]),
                "c": self._validate_param_value(hypothesis["c"]),
            }
            if params["a"] == 0:
                raise ValueError("invalid hypothesis: param out of range")
        elif family == "affine_mod_popcount":
            if "a" not in hypothesis or "b" not in hypothesis:
                raise ValueError("invalid hypothesis: bad schema")
            params = {
                "a": self._validate_param_value(hypothesis["a"]),
                "b": self._validate_param_value(hypothesis["b"]),
            }
            if params["a"] == 0:
                raise ValueError("invalid hypothesis: param out of range")
        elif family == "stepwise_composition":
            if "intervals" not in hypothesis or not isinstance(hypothesis["intervals"], list):
                raise ValueError("invalid hypothesis: stepwise_composition requires 'intervals' list")
            raw_ivs = hypothesis["intervals"]
            if len(raw_ivs) == 0:
                raise ValueError("invalid hypothesis: intervals list must be non-empty")

            _VALID_SUB = {
                "affine_mod",
                "quadratic_mod",
                "affine_mod_popcount",
                "piecewise_affine_mod",
            }
            norm_ivs: list[dict[str, Any]] = []
            for iv in raw_ivs:
                if not isinstance(iv, dict):
                    raise ValueError("invalid hypothesis: each interval must be a dict")
                for key in ("x_min", "x_max", "sub_family", "params"):
                    if key not in iv:
                        raise ValueError(f"invalid hypothesis: interval missing '{key}'")

                iv_xmin, iv_xmax = iv["x_min"], iv["x_max"]
                if (not isinstance(iv_xmin, int) or isinstance(iv_xmin, bool)
                        or not isinstance(iv_xmax, int) or isinstance(iv_xmax, bool)):
                    raise ValueError("invalid hypothesis: interval x_min/x_max must be int")
                if iv_xmin > iv_xmax:
                    raise ValueError("invalid hypothesis: interval x_min > x_max")

                sub_fam = iv["sub_family"]
                if sub_fam not in _VALID_SUB:
                    raise ValueError(f"invalid hypothesis: unsupported sub_family '{sub_fam}'")

                iv_len = iv_xmax - iv_xmin + 1
                min_len_for_fam = (
                    RD_MIN_QUADRATIC_LEN if sub_fam == "quadratic_mod" else RD_MIN_AFFINE_LEN
                )
                if iv_len < min_len_for_fam:
                    raise ValueError(
                        f"invalid hypothesis: interval [{iv_xmin},{iv_xmax}] has {iv_len} point(s) "
                        f"but {sub_fam} requires at least {min_len_for_fam}"
                    )

                sp = iv["params"]
                if not isinstance(sp, dict):
                    raise ValueError("invalid hypothesis: interval params must be a dict")

                if sub_fam == "affine_mod":
                    if "a" not in sp or "b" not in sp:
                        raise ValueError("invalid hypothesis: affine_mod params must have a, b")
                    norm_sp: dict[str, Any] = {
                        "a": self._validate_param_value(sp["a"]),
                        "b": self._validate_param_value(sp["b"]),
                    }
                    if norm_sp["a"] == 0:
                        raise ValueError("invalid hypothesis: param out of range")
                elif sub_fam == "quadratic_mod":
                    for k in ("a", "b", "c"):
                        if k not in sp:
                            raise ValueError(f"invalid hypothesis: quadratic_mod params must have {k}")
                    norm_sp = {
                        "a": self._validate_param_value(sp["a"]),
                        "b": self._validate_param_value(sp["b"]),
                        "c": self._validate_param_value(sp["c"]),
                    }
                    if norm_sp["a"] == 0:
                        raise ValueError("invalid hypothesis: param out of range")
                elif sub_fam == "affine_mod_popcount":
                    if "a" not in sp or "b" not in sp:
                        raise ValueError("invalid hypothesis: affine_mod_popcount params must have a, b")
                    norm_sp = {
                        "a": self._validate_param_value(sp["a"]),
                        "b": self._validate_param_value(sp["b"]),
                    }
                    if norm_sp["a"] == 0:
                        raise ValueError("invalid hypothesis: param out of range")
                else:  # piecewise_affine_mod
                    if "a" not in sp or "b" not in sp or "k" not in sp:
                        raise ValueError("invalid hypothesis: piecewise_affine_mod params must have a, b, k")
                    norm_sp = {
                        "a": self._validate_param_value(sp["a"]),
                        "b": self._validate_param_value(sp["b"]),
                        "k": sp["k"],
                    }
                    if norm_sp["a"] == 0:
                        raise ValueError("invalid hypothesis: param out of range")
                    if (not isinstance(norm_sp["k"], int) or isinstance(norm_sp["k"], bool)
                            or not (3 <= norm_sp["k"] <= 7)):
                        raise ValueError(
                            "invalid hypothesis: piecewise_affine_mod 'k' must be int in [3, 7]"
                        )

                norm_ivs.append({
                    "x_min": iv_xmin,
                    "x_max": iv_xmax,
                    "sub_family": sub_fam,
                    "params": norm_sp,
                })

            # Sort by x_min for stable fingerprinting
            norm_ivs.sort(key=lambda iv: iv["x_min"])

            # Validate full-domain coverage (no gaps, no overlaps)
            if norm_ivs[0]["x_min"] != self.x_min:
                raise ValueError(
                    f"invalid hypothesis: intervals must start at x={self.x_min}, "
                    f"got x_min={norm_ivs[0]['x_min']}"
                )
            if norm_ivs[-1]["x_max"] != self.x_max:
                raise ValueError(
                    f"invalid hypothesis: intervals must end at x={self.x_max}, "
                    f"got x_max={norm_ivs[-1]['x_max']}"
                )
            for i in range(len(norm_ivs) - 1):
                cur_max = norm_ivs[i]["x_max"]
                nxt_min = norm_ivs[i + 1]["x_min"]
                if cur_max + 1 > nxt_min:
                    raise ValueError(
                        f"invalid hypothesis: overlap between intervals ending at {cur_max} "
                        f"and starting at {nxt_min}"
                    )
                if cur_max + 1 < nxt_min:
                    raise ValueError(
                        f"invalid hypothesis: gap between intervals ending at {cur_max} "
                        f"and starting at {nxt_min}"
                    )

            params = {"intervals": norm_ivs}
        else:
            raise ValueError("invalid hypothesis: unsupported family")

        if self.max_exceptions == 0:
            # Exceptions are disabled; reject immediately if caller supplied any.
            if hypothesis.get("exceptions"):
                raise ValueError("invalid hypothesis: too many exceptions")
            return family, params, {}

        # Full exception validation (only reached when max_exceptions > 0).
        raw_exceptions = hypothesis.get("exceptions", {})
        if raw_exceptions is None:
            raw_exceptions = {}
        if not isinstance(raw_exceptions, dict):
            raise ValueError("invalid hypothesis: bad exception")
        if len(raw_exceptions) > self.max_exceptions:
            raise ValueError("invalid hypothesis: too many exceptions")

        normalized: dict[int, int] = {}
        for raw_x, raw_y in raw_exceptions.items():
            x = _parse_exception_x(raw_x)
            if x is None or x < self.x_min or x > self.x_max:
                raise ValueError("invalid hypothesis: bad exception")
            if not isinstance(raw_y, int) or not (0 <= raw_y < self.m):
                raise ValueError("invalid hypothesis: bad exception")
            normalized[x] = raw_y

        if not set(normalized.keys()).issubset(self.probed_xs):
            raise ValueError("invalid hypothesis: exception x not probed")

        return family, params, normalized

    def reset_turn(self) -> None:
        """Per-turn hook fired once at the start of each agent turn (via the
        interpreter's on_turn_start, through TestInputTool / ObserveTool). Resets
        the legacy per-turn probe counter AND advances the online stream cursor so
        each turn observes a fresh batch. Runtime-agnostic: fires identically under
        persistent and stateless, so both runtimes see the same stream."""
        self.probes_used_this_turn = 0
        self.stream_round += 1

    def _stream_grid(self) -> list[int]:
        """The fixed set of probe points the stream samples from: stream_grid_size
        x-values spread uniformly across the domain. Deterministic, so identical
        across runtimes and across turns."""
        n = max(2, self.stream_grid_size)
        span = self.x_max - self.x_min
        if span <= 0:
            return [self.x_min]
        grid = sorted({self.x_min + round(i * span / (n - 1)) for i in range(n)})
        return grid

    def _stream_batch(self, round_idx: int) -> list[list[int]]:
        """Deterministic noisy batch for one stream round. Pure function of
        (seed, round_idx), so repeated observe() calls within a turn return the
        same batch and both runtimes see identical data. Each sample is a grid
        point with its label corrupted to a different value with prob label_noise."""
        rng = random.Random((self.seed * 1_000_003) ^ (round_idx * 0x9E3779B1))
        grid = self._stream_grid()
        out: list[list[int]] = []
        for _ in range(self.stream_batch_size):
            x = rng.choice(grid)
            y_true = self._f(x)
            if self.m > 1 and rng.random() < self.label_noise:
                y = (y_true + rng.randint(1, self.m - 1)) % self.m
            else:
                y = y_true
            out.append([x, y])
        return out

    async def observe(self) -> dict[str, Any]:
        async with self._lock:
            if not self.online:
                raise ValueError("observe() is only available in online mode")
            if self.done:
                raise ValueError("task already submitted")
            r = max(0, self.stream_round)
            return {"round": r, "batch": self._stream_batch(r)}

    async def test_input(self, x: str | int) -> int:
        async with self._lock:
            if self.done:
                raise ValueError("task already submitted")
            if not isinstance(x, int) or isinstance(x, bool):
                raise ValueError("invalid input: x must be int")
            if x < self.x_min or x > self.x_max:
                raise ValueError("invalid input: x out of domain")
            if self.probes_used >= self.probe_budget:
                raise ValueError("probe budget exceeded")
            if (
                self.probes_per_turn is not None
                and self.probes_used_this_turn >= self.probes_per_turn
            ):
                raise ValueError(
                    f"per-turn probe cap reached ({self.probes_per_turn}); "
                    "end this turn and continue probing next turn"
                )
            self.probes_used += 1
            self.probes_used_this_turn += 1
            self.probed_xs.add(x)
            return self._f(x)

    async def check(self, hypothesis: HypothesisDict) -> CheckResult:
        async with self._lock:
            if self.done:
                raise ValueError("task already submitted")
            family, params, exceptions = self._normalize_hypothesis(hypothesis)

            mismatches: list[tuple[int, int]] = []
            for x in range(self.x_min, self.x_max + 1):
                y_true = self._f(x)
                y_pred = (
                    exceptions[x]
                    if x in exceptions
                    else self._base_output(x, family, params)
                )
                if y_true != y_pred:
                    mismatches.append((x, y_pred))

            if not mismatches:
                return {"status": "pass"}

            # Return a random witness inside the first contiguous wrong run,
            # capped to a window of _WITNESS_WINDOW points from the run start.
            # The cap bounds binary-search cost to log2(_WITNESS_WINDOW) probes
            # per boundary regardless of interval length, making difficulty
            # predictable and independent of where in the run the true boundary
            # happens to fall.
            mismatch_dict = {x: y_pred for x, y_pred in mismatches}
            x_run_start = mismatches[0][0]
            x_run_end = x_run_start
            while x_run_end + 1 in mismatch_dict:
                x_run_end += 1
            x_window_end = min(x_run_start + _WITNESS_WINDOW - 1, x_run_end)
            witness = random.randint(x_run_start, x_window_end)
            return {
                "status": "fail",
                "x": witness,
                "y_pred": mismatch_dict[witness],
            }

    async def submit_table(self, table: dict[int, int]) -> dict:
        """Online table-submission scoring path. The agent submits its DENOISED
        per-grid-point estimate {x: f_hat(x)}; the score is grid-recovery accuracy
        (fraction of the fixed probe grid recovered exactly). The hand-written
        fitter is removed from the scored path, so this measures pure cross-turn
        state aggregation."""
        async with self._lock:
            if self.done:
                raise ValueError("task already submitted")
            self.done = True
            grid = self._stream_grid()
            correct = 0
            for x in grid:
                if table.get(x) == self._f(x):
                    correct += 1
            total = len(grid)
            ratio = correct / total if total else 0.0
            # Reuse the functional slot so evaluate()/metrics report it uniformly.
            self.last_submit_match_ratio = ratio
            self.last_submit_interval_count = 0
            self.last_boundary_f1 = None
            self.last_family_acc = None
            self.solved = correct == total and total > 0
            return {
                "status": "scored",
                "recovered": correct,
                "grid_size": total,
                "accuracy": round(ratio, 4),
            }

    async def submit(self, hypothesis: dict) -> dict:
        async with self._lock:
            if self.done:
                raise ValueError("task already submitted")
            family, params, exceptions = self._normalize_hypothesis(hypothesis)
            self.done = True

            matched = 0
            total = self.x_max - self.x_min + 1
            mismatches: list[tuple[int, int]] = []

            for x in range(self.x_min, self.x_max + 1):
                y_true = self._f(x)
                y_pred = (
                    exceptions[x]
                    if x in exceptions
                    else self._base_output(x, family, params)
                )
                if y_true == y_pred:
                    matched += 1
                else:
                    mismatches.append((x, y_pred))

            self.last_submit_match_ratio = matched / total if total else 0.0

            true_intervals = (
                self.params.get("intervals", []) if self.family == "stepwise_composition" else []
            )
            sub_intervals = params.get("intervals", []) if family == "stepwise_composition" else []
            self.last_submit_interval_count = len(sub_intervals)

            self.last_boundary_f1, self.last_family_acc = _compute_structural_metrics(
                true_intervals, sub_intervals, self.m
            )

            # Check whether the first failing point falls inside a micro interval
            # (length < 5, which covers the default micro_interval_len_range of 2-4).
            if mismatches and self.family == "stepwise_composition":
                first_bad_x = mismatches[0][0]
                for iv in true_intervals:
                    if iv["x_min"] <= first_bad_x <= iv["x_max"]:
                        self.first_unsolved_micro = (iv["x_max"] - iv["x_min"] + 1) < self.min_interval_len
                        break

            # is_solved requires exact structural match: all boundaries and families correct.
            # submit() return reflects only functional correctness (consistent with check()).
            # Structural quality is tracked in is_solved / score metrics transparently.
            exact_structure = (self.last_boundary_f1 == 1.0 and self.last_family_acc == 1.0)
            self.solved = (not mismatches) and exact_structure

            if not mismatches:
                return {"status": "pass"}

            x_bad, y_pred_bad = mismatches[0]
            return {
                "status": "fail",
                "x": x_bad,
                "y_pred": y_pred_bad,
            }


class TestInputTool(Tool):
    name: str = "test_input"
    doc: str = (
        "Membership query: return f(x) for an in-domain integer x. "
        "Consumes 1 probe from a strict total budget."
    )
    arg_doc: dict[str, str] = {"x": "integer x in the public domain"}

    def __init__(self, env: RuleDiagnosisEnv):
        self._env = env
        super().__init__()

    def on_turn_start(self) -> None:
        # Interpreter calls this once per agent turn; reset the per-turn probe cap.
        self._env.reset_turn()

    async def run(self, x: int) -> int:
        return await self._env.test_input(x)


class ObserveTool(Tool):
    name: str = "observe"
    doc: str = (
        "Online mode only. Return THIS turn's noisy observations as a dict "
        "{'round': n, 'batch': [[x, y], ...]}: each y == f(x) but corrupted to a "
        "random value with the task's label-noise probability. The samples come "
        "from a fixed set of probe points that recur across turns, so accumulate "
        "them and majority-vote per x to denoise. Calling observe() again in the "
        "same turn returns the SAME batch; the next batch arrives only on the next "
        "turn. Takes no arguments."
    )
    arg_doc: dict[str, str] = {}

    def __init__(self, env: RuleDiagnosisEnv):
        self._env = env
        super().__init__()

    def on_turn_start(self) -> None:
        # Interpreter calls this once per agent turn; advance the stream cursor.
        self._env.reset_turn()

    async def run(self) -> dict[str, Any]:
        return await self._env.observe()


class CheckTool(Tool):
    name: str = "check"
    doc: str = (
        "Equivalence-style oracle check over the full domain. "
        "Pass a Python dict directly (not a JSON string). "
        "Required keys: 'm' (int modulus), 'family' (str). "
        "For a single-rule domain: family='affine_mod' with keys a,b; "
        "'quadratic_mod' with a,b,c; 'affine_mod_popcount' with a,b. "
        "For a piecewise domain: family='stepwise_composition', "
        "intervals=[{'x_min':..,'x_max':..,'sub_family':..,'params':{..}}, ...] "
        "covering [x_min, x_max] without gaps or overlaps. "
        "IMPORTANT: param 'a' must be non-zero for every sub-family. "
        "Returns {'status':'pass'} or {'status':'fail','x':C,'y_pred':Y} where C is the "
        "first (lowest-x) failing point. "
        "Returns {'status':'error','message':...} if the schema is invalid — "
        "the task is NOT over; fix the hypothesis and retry. "
        "Does NOT consume probe budget."
    )
    arg_doc: dict[str, str] = {
        "hypothesis": "Python dict with keys m, family, and family-specific params"
    }

    def __init__(self, env: RuleDiagnosisEnv):
        self._env = env
        super().__init__()

    async def run(self, hypothesis: dict) -> dict:
        try:
            parsed_hyp = _parse_hypothesis_input(hypothesis)
            return await self._env.check(parsed_hyp)
        except ValueError as e:
            return {"status": "error", "message": str(e)}


class SubmitRuleTool(Tool):
    name: str = "submit"
    doc: str = (
        "Finalize and score your hypothesis. Same schema as check(). "
        "Pass a Python dict directly (not a JSON string). "
        "IMPORTANT: submit() ends the task immediately and cannot be undone — "
        "call it only once, when finished or out of probe budget. "
        "IMPORTANT: param 'a' must be non-zero for every sub-family. "
        "If the schema is invalid, returns {'status':'error','message':...} and the task "
        "is NOT ended — fix the hypothesis and call submit() again. "
        "On success or scored failure: returns {'status':'pass'} or "
        "{'status':'fail','x':C,'y_pred':Y} where C is the first failing point, "
        "and the task ends."
    )
    arg_doc: dict[str, str] = {
        "hypothesis": "Python dict with keys m, family, and family-specific params"
    }

    def __init__(self, env: RuleDiagnosisEnv):
        self._env = env
        super().__init__()

    async def run(self, hypothesis: dict) -> dict:
        try:
            parsed_hyp = _parse_hypothesis_input(hypothesis)
            return await self._env.submit(parsed_hyp)
        except ValueError as e:
            return {"status": "error", "message": str(e)}


def _parse_table_input(table: Any) -> dict[int, int]:
    """Coerce a submitted table into {int: int}. Accepts a dict (JSON keys arrive
    as strings) or a JSON string; values and keys are cast to int."""
    if isinstance(table, str):
        try:
            table = json.loads(table)
        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(f"invalid table: not valid JSON ({e})") from None
    if not isinstance(table, dict):
        raise ValueError("invalid table: must be a dict {x: y}")
    out: dict[int, int] = {}
    for k, v in table.items():
        try:
            out[int(k)] = int(v)
        except (TypeError, ValueError):
            raise ValueError(f"invalid table entry {k!r}: {v!r} (need int->int)") from None
    return out


class SubmitTableTool(Tool):
    name: str = "submit"
    doc: str = (
        "Finalize and score your DENOISED table. Pass a Python dict {x: y} mapping "
        "each probe point x (int) to your best estimate of f(x) (int in [0, m)). "
        "submit() ends the task immediately and cannot be undone — call it once, "
        "when your per-x majority votes are stable and you have covered the probe "
        "points. Returns {'status':'scored','recovered':k,'grid_size':n,"
        "'accuracy':r}. No rule-fitting is required — submit the values themselves."
    )
    arg_doc: dict[str, str] = {"table": "Python dict {x: y} of denoised estimates"}

    def __init__(self, env: RuleDiagnosisEnv):
        self._env = env
        super().__init__()

    async def run(self, table: dict) -> dict:
        try:
            parsed = _parse_table_input(table)
            return await self._env.submit_table(parsed)
        except ValueError as e:
            return {"status": "error", "message": str(e)}


def _count_recurrences(intervals: list[dict]) -> int:
    """Count intervals that exactly reuse (sub_family, params) from a non-adjacent earlier position."""
    count = 0
    for i in range(len(intervals)):
        key_i = (intervals[i]["sub_family"], tuple(sorted(intervals[i]["params"].items())))
        for j in range(max(0, i - 1)):  # j <= i-2: non-adjacent predecessors
            key_j = (intervals[j]["sub_family"], tuple(sorted(intervals[j]["params"].items())))
            if key_i == key_j:
                count += 1
                break
    return count


def _compute_structural_metrics(
    true_intervals: list[dict], sub_intervals: list[dict], m: int
) -> tuple[float, float]:
    """Return (boundary_f1, family_acc) comparing submitted vs true structure.

    An interval counts as a family match if either (a) sub_family and params
    are identical, or (b) the two formulas are functionally equivalent over
    [x_min, x_max] — i.e. produce the same output for every x in the interval.
    The functional fallback catches cases where different algebraic forms are
    identical mod m (e.g. quadratic_mod(4,0,5) ≡ affine_mod(4,5) mod 8).
    """
    true_bounds = {iv["x_min"] for iv in true_intervals[1:]}
    sub_bounds = {iv["x_min"] for iv in sub_intervals[1:]}

    if not true_bounds and not sub_bounds:
        boundary_f1 = 1.0
    elif not true_bounds or not sub_bounds:
        boundary_f1 = 0.0
    else:
        tp = len(true_bounds & sub_bounds)
        precision = tp / len(sub_bounds)
        recall = tp / len(true_bounds)
        denom = precision + recall
        boundary_f1 = (2.0 * precision * recall / denom) if denom > 0 else 0.0

    if not true_intervals:
        return boundary_f1, 1.0

    true_by_range = {(iv["x_min"], iv["x_max"]): iv for iv in true_intervals}
    sub_by_range  = {(iv["x_min"], iv["x_max"]): iv for iv in sub_intervals}

    correct = 0
    for rng, t_iv in true_by_range.items():
        s_iv = sub_by_range.get(rng)
        if s_iv is None:
            continue
        if s_iv["sub_family"] == t_iv["sub_family"] and s_iv["params"] == t_iv["params"]:
            correct += 1
            continue
        # Functional equivalence fallback: same outputs for every x in interval.
        x_min, x_max = rng
        if all(
            _eval_base(s_iv["sub_family"], s_iv["params"], x, m)
            == _eval_base(t_iv["sub_family"], t_iv["params"], x, m)
            for x in range(x_min, x_max + 1)
        ):
            correct += 1

    family_acc = correct / len(true_by_range)
    return boundary_f1, family_acc


def _parse_hypothesis_input(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(raw)
            except Exception:
                raise ValueError("invalid hypothesis: bad schema")
        
    raise ValueError("invalid hypothesis: bad schema")


def _parse_exception_x(raw_x: Any) -> int | None:
    if isinstance(raw_x, int):
        return raw_x
    if isinstance(raw_x, str):
        text = raw_x.strip()
        if not text:
            return None
        if text[0] in "+-":
            if not text[1:].isdigit():
                return None
            return int(text)
        if text.isdigit():
            return int(text)
    return None


def _sample_base_params(
    rng: random.Random, family: str, m: int
) -> dict[str, int]:
    if family == "affine_mod":
        a = 0
        while a == 0:
            a = rng.randint(0, m - 1)
        return {"a": a, "b": rng.randint(0, m - 1)}

    if family == "quadratic_mod":
        a = 0
        while a == 0:
            a = rng.randint(0, m - 1)
        return {
            "a": a,
            "b": rng.randint(0, m - 1),
            "c": rng.randint(0, m - 1),
        }

    if family == "affine_mod_popcount":
        a = 0
        while a == 0:
            a = rng.randint(0, m - 1)
        return {"a": a, "b": rng.randint(0, m - 1)}

    if family == "piecewise_affine_mod":
        # f(x) = (a * (x mod k) + b) % m. Period k in [3, 7].
        a = 0
        while a == 0:
            a = rng.randint(0, m - 1)
        return {
            "a": a,
            "b": rng.randint(0, m - 1),
            "k": rng.randint(3, 7),
        }

    raise ValueError("unsupported family")


# Min interval length needed to disambiguate the period of piecewise_affine_mod
# at the maximum k. 2 full cycles → 2 * k_max + 2 = 16 (with k_max=7). Anything
# shorter and multiple (a, b, k) combinations would fit the same probe results.
_PIECEWISE_MIN_INTERVAL_LEN: int = 16


def _eval_base(family: str, params: dict[str, Any], x: int, m: int) -> int:
    if family == "affine_mod":
        return (params["a"] * x + params["b"]) % m
    if family == "quadratic_mod":
        return (params["a"] * x * x + params["b"] * x + params["c"]) % m
    if family == "affine_mod_popcount":
        return (params["a"] * x + params["b"] + x.bit_count()) % m
    if family == "piecewise_affine_mod":
        return (params["a"] * (x % params["k"]) + params["b"]) % m
    if family == "stepwise_composition":
        for interval in params["intervals"]:
            if interval["x_min"] <= x <= interval["x_max"]:
                sub_family = interval["sub_family"]
                sub_params = interval.get("params", {})
                return _eval_base(sub_family, sub_params, x, m)
        raise ValueError(f"x={x} not covered by any interval")
    raise ValueError("unsupported family")


def _check_state_pressure(
    intervals: list[dict[str, Any]],
    probe_budget: int,
    exceptions: dict[int, int],
    *,
    require_multi_breakpoint: bool,
    require_exception_count: Optional[tuple[int, int]] = None,
) -> Optional[str]:
    """Belt-and-braces post-gen audit that tier promotions actually fired.

    The config validator already gates `probe_budget_range` and
    `micro_interval_count`, but the breakpoint count is decided in the sampler
    via tier thresholds, and the exception count is sampled from a range. This
    guard surfaces unexpected drift (single-breakpoint instances when multi-bp
    is required, or exception placement falling short of the target range) so
    they show up as `state_pressure_warning` in the task's difficulty dict
    rather than slipping through silently.
    """
    if require_multi_breakpoint:
        num_breakpoints = max(0, len(intervals) - 1)
        if num_breakpoints < 2:
            return "single_breakpoint_easy"
        full_audit = sum(iv["x_max"] - iv["x_min"] + 1 for iv in intervals)
        if probe_budget >= full_audit:
            return "budget_covers_full_domain"
    if require_exception_count is not None:
        lo, hi = require_exception_count
        if not (lo <= len(exceptions) <= hi):
            return "exception_count_out_of_range"
    return None


def _sample_exceptions(
    rng: random.Random,
    intervals: list[dict[str, Any]],
    m: int,
    count: int,
    *,
    min_distance_to_boundary: int = 1,
) -> dict[int, int]:
    """Pick `count` interior x values whose y is overridden to differ from base f(x).

    Avoid boundary-adjacent points (within `min_distance_to_boundary`) so an
    exception miss-detection by `check()` doesn't get confused with a boundary
    miss-detection by the witness-window logic. Each exception's y is drawn
    uniformly from `[0, m) \\ {base_y}` so it is a genuine override.
    """
    candidates: list[int] = []
    for iv in intervals:
        lo = iv["x_min"] + min_distance_to_boundary
        hi = iv["x_max"] - min_distance_to_boundary
        if hi >= lo:
            candidates.extend(range(lo, hi + 1))
    rng.shuffle(candidates)
    out: dict[int, int] = {}
    for x in candidates:
        if len(out) >= count:
            break
        iv = next(i for i in intervals if i["x_min"] <= x <= i["x_max"])
        base_y = _eval_base(iv["sub_family"], iv["params"], x, m)
        choices = [y for y in range(m) if y != base_y]
        out[x] = rng.choice(choices)
    return out


def sample_rule_diagnosis_instance(
    seed: int, cfg: RuleDiagnosisConfig
) -> RuleDiagnosisTaskData:
    rng = random.Random(seed)

    m = rng.choice(cfg.mod_m_choices)
    x_min, x_max = cfg.domain_range
    probe_budget = rng.randint(cfg.probe_budget_range[0], cfg.probe_budget_range[1])
    
    # Force the types to be stepwise_composition
    family = "stepwise_composition"
    # Exceptions sampled after intervals are finalized — see end of function.

    # Tier thresholds are defined in config.py (RD_TIER_EASY_MAX / RD_TIER_MEDIUM_MAX)
    # and validated there; keep this logic in sync with those constants.
    if probe_budget <= RD_TIER_EASY_MAX:
        num_breakpoints = 1
    elif probe_budget <= RD_TIER_MEDIUM_MAX:
        num_breakpoints = rng.randint(2, 3)
    else:
        num_breakpoints = rng.randint(4, 6)

    # Partition [x_min, x_max] into (num_breakpoints+1) contiguous segments.
    # Normal segments are >= min_interval_len wide (stars-and-bars).
    # Micro segments are [micro_lo, micro_hi] wide and placed at random interior
    # positions, making them harder to detect and fit within the probe budget.
    num_segs = num_breakpoints + 1
    domain_size = x_max - x_min + 1
    min_len = cfg.min_interval_len
    num_micro = min(cfg.micro_interval_count, max(0, num_segs - 2))

    if num_micro > 0:
        micro_lo, micro_hi = cfg.micro_interval_len_range
        # Validator guarantees micro_interval_count <= interior slots at runtime.
        interior = list(range(1, num_segs - 1))
        micro_indices = set(rng.sample(interior, num_micro))

        micro_lens = {i: rng.randint(micro_lo, micro_hi) for i in micro_indices}
        num_normal = num_segs - len(micro_indices)
        normal_remaining = domain_size - sum(micro_lens.values()) - num_normal * min_len

        if num_normal == 1:
            normal_lengths = [min_len + normal_remaining]
        else:
            # rng.sample(range(1, normal_remaining), num_normal-1) requires
            # normal_remaining >= num_normal; guaranteed by config validator
            # via: domain_size >= num_micro*mhi + num_normal*(min_len+1).
            cuts = sorted(rng.sample(range(1, normal_remaining), num_normal - 1))
            extras = [cuts[0]] + [cuts[i] - cuts[i - 1] for i in range(1, len(cuts))] + [normal_remaining - cuts[-1]]
            normal_lengths = [min_len + e for e in extras]

        normal_iter = iter(normal_lengths)
        seg_lengths = [
            micro_lens[i] if i in micro_indices else next(normal_iter)
            for i in range(num_segs)
        ]
    else:
        # rng.sample(range(1, remaining), num_segs-1) requires remaining >= num_segs;
        # guaranteed by config validator via: domain_size >= num_segs*(min_len+1).
        remaining = domain_size - num_segs * min_len
        cuts = sorted(rng.sample(range(1, remaining), num_segs - 1))
        extras = [cuts[0]] + [cuts[i] - cuts[i - 1] for i in range(1, len(cuts))] + [remaining - cuts[-1]]
        seg_lengths = [min_len + e for e in extras]

    interval_ranges: list[tuple[int, int]] = []
    current_min = x_min
    for length in seg_lengths:
        interval_ranges.append((current_min, current_min + length - 1))
        current_min += length

    # Assign a sub-family and params to each interval.
    # Exclude quadratic_mod when m is composite: over Z/mZ with non-prime m, fitting
    # a*x^2 + b*x + c from 3+ points can yield multiple valid parameter sets, breaking
    # the "small probe cluster uniquely determines params" guarantee.
    # quadratic_mod only for prime m; also needs >= 4 points to be uniquely fit+verified.
    all_sub_families = (
        ["affine_mod", "quadratic_mod", "affine_mod_popcount"]
        if _is_prime_int(m)
        else ["affine_mod", "affine_mod_popcount"]
    )
    if cfg.include_piecewise_affine_mod:
        all_sub_families = all_sub_families + ["piecewise_affine_mod"]
    non_quadratic = ["affine_mod", "affine_mod_popcount"]
    if cfg.include_piecewise_affine_mod:
        non_quadratic = non_quadratic + ["piecewise_affine_mod"]

    if cfg.online:
        # Online mode fits each segment from noisy samples; restrict to the two
        # linear families so per-segment fitting stays well-posed under noise (no
        # quadratic / periodic ambiguity to disentangle from corrupted labels).
        all_sub_families = ["affine_mod", "affine_mod_popcount"]
        non_quadratic = ["affine_mod", "affine_mod_popcount"]

    intervals: list[dict[str, Any]] = []

    if cfg.rule_library_size >= 2:
        # Library path: pre-sample L distinct rules, assign intervals from the library
        # (no adjacent same index).  All intervals with the same library index have
        # IDENTICAL params, enabling the model to exploit recurrences.
        # Visibility is checked all-or-nothing; the whole library is resampled on any
        # violation so that shared params stay consistent across reuse positions.
        _LIBRARY_RETRIES = 300
        for _retry in range(_LIBRARY_RETRIES):
            # 1. Sample library of distinct (sub_family, params) rules.
            library: list[tuple[str, dict[str, Any]]] = []
            for _ in range(cfg.rule_library_size):
                for _attempt in range(60):
                    sf = rng.choice(all_sub_families)
                    sp = _sample_base_params(rng, sf, m)
                    key = (sf, tuple(sorted(sp.items())))
                    if all(key != (ls, tuple(sorted(lp.items()))) for ls, lp in library):
                        library.append((sf, sp))
                        break

            if len(library) < cfg.rule_library_size:
                continue  # failed to build distinct library; retry

            # 2. Assign library indices to intervals (no adjacent same index).
            assignments: list[int] = []
            assignment_ok = True
            for i_rng, (i_min, i_max) in enumerate(interval_ranges):
                iv_len = i_max - i_min + 1
                eligible = [
                    idx for idx, (sf, _) in enumerate(library)
                    if not (sf == "quadratic_mod" and iv_len < RD_MIN_QUADRATIC_LEN)
                    and not (sf == "piecewise_affine_mod" and iv_len < _PIECEWISE_MIN_INTERVAL_LEN)
                ]
                prev_idx = assignments[-1] if assignments else -1
                candidates = [idx for idx in eligible if idx != prev_idx]
                if not candidates:
                    assignment_ok = False
                    break
                assignments.append(rng.choice(candidates))

            if not assignment_ok:
                continue

            # 3. Build interval list from library.
            intervals = [
                {
                    "x_min": r[0], "x_max": r[1],
                    "sub_family": library[idx][0],
                    "params": dict(library[idx][1]),
                }
                for r, idx in zip(interval_ranges, assignments)
            ]

            # 4. Check visibility all-or-nothing (no per-interval resampling here).
            vis_ok = True
            for i_iv, iv in enumerate(intervals):
                if i_iv > 0:
                    left = intervals[i_iv - 1]
                    bx = iv["x_min"]
                    if (_eval_base(left["sub_family"], left["params"], bx, m)
                            == _eval_base(iv["sub_family"], iv["params"], bx, m)):
                        vis_ok = False
                        break
                if iv["x_max"] - iv["x_min"] + 1 < cfg.min_interval_len:
                    xs = list(range(iv["x_min"], iv["x_max"] + 1))
                    left_ok = i_iv == 0 or any(
                        _eval_base(iv["sub_family"], iv["params"], x, m)
                        != _eval_base(intervals[i_iv - 1]["sub_family"], intervals[i_iv - 1]["params"], x, m)
                        for x in xs
                    )
                    right_ok = i_iv == len(intervals) - 1 or any(
                        _eval_base(iv["sub_family"], iv["params"], x, m)
                        != _eval_base(intervals[i_iv + 1]["sub_family"], intervals[i_iv + 1]["params"], x, m)
                        for x in xs
                    )
                    if not (left_ok and right_ok):
                        vis_ok = False
                        break
            if vis_ok:
                break
    else:
        # Independent path: sample fresh (sub_family, params) for each interval,
        # forbidding adjacent repeats of the same sub_family.
        prev_sub_family: str | None = None
        for i_min, i_max in interval_ranges:
            interval_len = i_max - i_min + 1
            pool = all_sub_families if interval_len >= RD_MIN_QUADRATIC_LEN else non_quadratic
            # piecewise_affine_mod needs enough points to disambiguate the period.
            if interval_len < _PIECEWISE_MIN_INTERVAL_LEN:
                pool = [sf for sf in pool if sf != "piecewise_affine_mod"]
            if prev_sub_family is None:
                sub_family = rng.choice(pool)
            else:
                candidates = [sf for sf in pool if sf != prev_sub_family]
                sub_family = rng.choice(candidates)

            sub_params = _sample_base_params(rng, sub_family, m)
            intervals.append(
                {
                    "x_min": i_min,
                    "x_max": i_max,
                    "sub_family": sub_family,
                    "params": sub_params,
                }
            )
            prev_sub_family = sub_family

        # Enforce two visibility guarantees, iterating until both hold simultaneously:
        #
        # 1. Boundary visibility (all adjacent pairs): the two formulas must differ at
        #    the first x of the right interval.  check() returns that x as the first
        #    fail, so if formulas agree there the model's boundary detection is wrong.
        #    Fix: resample the RIGHT interval's params.
        #
        # 2. Micro interior distinguishability: for each micro interval, at least one
        #    point inside it must differ from both neighbors' formulas, so the model
        #    can fit a formula on the short span without ambiguity.
        #    Fix: resample the micro interval's params.
        #
        # Sub-family is never changed here to preserve the adjacency-uniqueness invariant.
        _VISIBILITY_RETRIES = 100
        for _ in range(_VISIBILITY_RETRIES):
            any_violation = False
            for idx, iv in enumerate(intervals):
                if idx > 0:
                    left = intervals[idx - 1]
                    bx = iv["x_min"]
                    if (_eval_base(left["sub_family"], left["params"], bx, m)
                            == _eval_base(iv["sub_family"], iv["params"], bx, m)):
                        iv["params"] = _sample_base_params(rng, iv["sub_family"], m)
                        any_violation = True
                if iv["x_max"] - iv["x_min"] + 1 < cfg.min_interval_len:
                    xs = list(range(iv["x_min"], iv["x_max"] + 1))
                    left_ok = idx == 0 or any(
                        _eval_base(iv["sub_family"], iv["params"], x, m)
                        != _eval_base(intervals[idx - 1]["sub_family"], intervals[idx - 1]["params"], x, m)
                        for x in xs
                    )
                    right_ok = idx == len(intervals) - 1 or any(
                        _eval_base(iv["sub_family"], iv["params"], x, m)
                        != _eval_base(intervals[idx + 1]["sub_family"], intervals[idx + 1]["params"], x, m)
                        for x in xs
                    )
                    if not (left_ok and right_ok):
                        iv["params"] = _sample_base_params(rng, iv["sub_family"], m)
                        any_violation = True
            if not any_violation:
                break

    params = {"intervals": intervals}

    # Sample exception points (interior x values whose y differs from base f(x)).
    # When `exceptions_count_range == (0, 0)` this is a no-op (legacy behavior).
    if cfg.exceptions_count_range != (0, 0):
        exceptions_count = rng.randint(*cfg.exceptions_count_range)
        exceptions: dict[int, int] = _sample_exceptions(
            rng, intervals, m, exceptions_count
        )
    else:
        exceptions = {}

    true_hypothesis = {
        "m": m,
        "family": family,
        **params,
        "exceptions": exceptions,
    }

    difficulty: dict[str, Any] = {
        "family": family,
        "m": m,
        "probe_budget": probe_budget,
        "exceptions": len(exceptions),
        "micro_intervals": num_micro,
        "min_interval_len": cfg.min_interval_len,
        "rule_library_size": cfg.rule_library_size,
        "recurrence_count": _count_recurrences(intervals),
    }
    require_exc_range = (
        cfg.exceptions_count_range
        if cfg.exceptions_count_range != (0, 0)
        else None
    )
    pressure_reason = _check_state_pressure(
        intervals,
        probe_budget,
        exceptions,
        require_multi_breakpoint=cfg.require_multi_breakpoint,
        require_exception_count=require_exc_range,
    )
    if pressure_reason is not None:
        difficulty["state_pressure_warning"] = pressure_reason

    return RuleDiagnosisTaskData(
        family="rule_diagnosis",
        seed=seed,
        difficulty=difficulty,
        public=RuleDiagnosisPublic(
            m=m,
            probe_budget=probe_budget,
            x_domain={"min": x_min, "max": x_max},
            max_exceptions=cfg.max_hypothesis_exceptions,
            family_hint="stepwise_composition" if cfg.reveal_family_in_public else None,
            probes_per_turn=cfg.probes_per_turn,
            online=cfg.online,
            online_submit_table=cfg.online_submit_table,
            stream_grid_size=cfg.stream_grid_size,
            stream_batch_size=cfg.stream_batch_size,
            label_noise=cfg.label_noise,
        ),
        private=RuleDiagnosisPrivate(
            family=family,
            params=params,
            exceptions=exceptions,
        ),
        reference=RuleDiagnosisReference(true_hypothesis=true_hypothesis),
    )