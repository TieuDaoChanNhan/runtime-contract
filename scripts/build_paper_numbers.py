#!/usr/bin/env python3
"""Single source of truth for every number cited in the paper.

Computes all cited quantities from experiments/cap_sweep/ (+ the trace-conditioned
replay) and writes them to paper/numbers.json. The manuscript's main.tex and
appendix.tex are rendered from Jinja templates using this JSON, so no number is
hand-typed into it; the templates and the renderer are not part of this release.

Config/design constants (caps, window, LoRA hyperparameters, task-generation
parameters) are recorded here with an explicit source note rather than computed.

Usage:  uv run python -m scripts.build_paper_numbers   # writes paper/numbers.json
"""

import asyncio
import glob
import json
import math
import os
import random
import statistics
from collections import Counter

import scripts.analyze_paper as A
from scripts.replay_analysis import _load, _replay_one  # type: ignore[reportMissingImports]

KFAM = "knapsack"
CB, CS = A.CAP_BIND, A.CAP_SLACK
SEEDS = A.SEEDS


# ------------------------------------------------------------------ helpers
def _cells(seed, cap):
    c = A.knap_cells(seed, cap)
    if not c:
        return None
    cells, common = c
    m = {cl: statistics.mean(cells[cl]) for cl in cells}
    m["D"] = (m["PP"] - m["PS"]) - (m["SP"] - m["SS"])
    m["GP"] = m["PP"] - m["PS"]
    m["n"] = len(common)
    m["_paired"] = cells
    return m


def _mean_scores(path):
    return list(A._scores(path, KFAM).values())


def _crossed_mean_ci(lists_by_seed, nb=5000):
    """Crossed bootstrap over seeds x tasks for one scalar quantity per (seed,task)."""
    seeds = list(range(len(lists_by_seed)))
    boots = []
    for _ in range(nb):
        bs = [seeds[random.randrange(len(seeds))] for _ in seeds]
        vals = []
        for s in bs:
            lst = lists_by_seed[s]
            vals.append(statistics.mean(lst[random.randrange(len(lst))] for _ in lst))
        boots.append(statistics.mean(vals))
    boots.sort()
    return statistics.mean(boots), boots[int(0.025 * nb)], boots[int(0.975 * nb)]


# ------------------------------------------------------------------ builders (constants)
def build_config():
    return {
        "_source": "eval/training configs (experiments/cap_sweep, train/configs), README",
        "cap_bind": CB,
        "cap_slack": CS,
        "cap_nearthresh": 40,
        "tmax": 40,
        "tmax_grid": [40, 80, 160],
        "window": 40960,
        "out_cap": 8192,
        "input_ceil": 40960 - 8192,
        "temp": 0.2,
        "timeout_s": 600,
        "bootstrap_resamples": 5000,
        "ci_level": 95,
        "n_seeds": 3,
        "n_anchor": 25,
        "n_extra_seed": 20,
        "n_seed_small": 8,
        "n_dense": 12,
        "n_mech": 16,
        "n_control": 20,
        "n_nearthresh80": 11,
        "n_conditions": 4,
        "task_calls_needed": 50,
        "gt30k_threshold_k": 30,
    }


def build_task():
    return {
        "_source": "families/knapsack.py + knapsack/task_defs/_cfg.json (medium tier)",
        "n_lo": 40,
        "n_hi": 65,
        "easy_lo": 25,
        "easy_hi": 40,
        "hard_lo": 80,
        "hard_hi": 120,
        "w_lo": 5,
        "w_hi": 35,
        "wbar": 20,
        "v_lo": 10,
        "v_hi": 300,
        "n_classes": 26,
        "n_allowed": 4,
        "cap_ratio_lo": 0.35,
        "cap_ratio_hi": 0.5,
        "budget_coef_fill": 1.5,
        "budget_coef_floor": 1.1,
        "budget_pvalid_floor": 0.01,
        "budget_clamp_lo": 5,
        "reject_min_solution": 3,
        "reject_max_dominance_pct": 40,
        "reject_min_allowed": 5,
        "p_valid_num": 4,
        "p_valid_den": 26,
        "p_valid": 4 / 26,
        "eg_n": 60,
        "eg_capacity": 143,
        "eg_optset": 7,
        "eg_budget": 60,
        "eg_fill": (1.5 * 143 / 20) / (4 / 26),
        "eg_floor": 1.1 * 7,
    }


def build_nav():
    # Cross-family navigation control, exactly as evaluated in the cap sweep.
    cfg = json.load(open("experiments/cap_sweep/navigation/task_defs/_cfg.json"))["navigation"]
    hlo, hhi = cfg["horizon_range"]
    elo, ehi = cfg["extra_nodes_range"]
    # node count per families/navigation.py:1199-1200
    n_lo = max(35, hlo + elo + 20)
    n_hi = max(n_lo, hhi + ehi + 30)
    return {
        "_source": "experiments/cap_sweep/navigation/task_defs/_cfg.json + families/navigation.py "
        "(node count derived at :1199-1200; neighbors batch cap :698)",
        "n_tasks": cfg["num_tasks"],
        "horizon_lo": hlo,
        "horizon_hi": hhi,
        "n_lo": n_lo,
        "n_hi": n_hi,
        "neighbors_batch": 50,
        "n_locks": 1,
        "probe_cap_fraction": cfg["probe_cap_fraction"],
        "trap_lo": cfg["trap_count_range"][0],
        "trap_hi": cfg["trap_count_range"][1],
        "decoy_lo": cfg["decoy_count_range"][0],
        "decoy_hi": cfg["decoy_count_range"][1],
        "key_hint_radius": cfg["key_hint_radius"],
        "max_steps_mult": cfg["max_steps_multiplier"],
    }


def build_rule():
    # Cross-family rule-diagnosis control, exactly as evaluated: the OFFLINE (pull-based)
    # design (test_input/check/probe_budget), which is what the cap sweep ran ("online": false).
    # The shipped medium.json later switched to an online streaming design for separate work.
    cfg = json.load(open("experiments/cap_sweep/rule_diagnosis/task_defs/_cfg.json"))[
        "rule_diagnosis"
    ]
    return {
        "_source": "experiments/cap_sweep/rule_diagnosis/task_defs/_cfg.json (online:false, the evaluated "
        "offline design) + families/rule_diagnosis.py (composite score :330-336)",
        "n_tasks": cfg["num_tasks"],
        "mod_m": cfg["mod_m_choices"],
        "domain_lo": cfg["domain_range"][0],
        "domain_hi": cfg["domain_range"][1],
        "probe_budget_lo": cfg["probe_budget_range"][0],
        "probe_budget_hi": cfg["probe_budget_range"][1],
        "min_interval_len": cfg["min_interval_len"],
        "multi_breakpoint": cfg["require_multi_breakpoint"],
        # composite score weights (rule_diagnosis.py:330-336)
        "w_functional": 0.60,
        "w_boundary": 0.25,
        "w_family": 0.15,
        "complexity_penalty": 0.05,
    }


def build_training():
    return {
        "_source": "train/configs/axolotl_*.yaml, README, docs; token/length stats MEASURED on "
        "the released trace dataset (anonymized mirror at submission; "
        "data/knapsack/{regime}/traces.jsonl, N=1000/regime at cap; Qwen3-8B tokenizer). "
        "paired-filter yield from the documented prepare_paired_data.py run "
        "(valid=1109, skipped=791). "
        "seq_len/hardware/gpu_hours/licenses per the actual GH200 training run. "
        "train_gpu_hours is the author's figure for the main arm; "
        "train_gpu_hours_mistral/train_gpu_hours_llama31 are training-only GPU-hours read "
        "from sacct for the successful jobs of the replication arms, which ran on separate "
        "hardware. train_gpu_hours_crossfam is training-only GPU-hours for the four "
        "cross-family adapters, read from each run's logged train_runtime. Serving/evaluation "
        "compute for the full cap-sweep grid is still not tallied on any arm, which is why the "
        "compute checklist item answers No. base_model licences per the HF model cards. "
        "cluster name intentionally anonymized for double-blind submission.",
        "base_model": "Qwen3-8B",
        "qlora_bits": 4,
        "lora_r": 64,
        "lora_alpha": 128,
        "lora_dropout": 0.05,
        "epochs": 3,
        "peak_lr": "1\\times10^{-4}",
        "warmup_ratio": 0.03,
        "weight_decay": 0.01,
        "seq_len": 16384,
        "seq_len_mistral": 16384,
        "seq_len_llama31": 16384,
        "microbatch": 1,
        "grad_accum": 16,
        "train_gpu": "GH200",
        "train_gpu_count": 4,
        "train_nodes": 1,
        "train_cluster": "anonymized",
        # 14 adapters exist: 3 Qwen knapsack persistent/stateless seed pairs (the main arm);
        # one persistent/stateless pair each for navigation and rule diagnosis, trained on
        # those families' own paired traces; and one pair each on Mistral-7B-v0.3 and
        # Llama-3.1-8B. The total is summed from the parts so it cannot drift again.
        "train_n_adapters_main": 6,
        "train_n_adapters_crossfam": 4,
        "train_n_adapters_ablation": 4,
        "train_n_adapters": 6 + 4 + 4,
        "train_gpu_hours": 20,
        "train_gpu_hours_mistral": 7.94,
        "train_gpu_hours_llama31": 8.57,
        # sum of the four cross-family adapters' own reported GPU-hours.
        "train_gpu_hours_crossfam": round(2.59 + 2.69 + 2.63 + 2.80, 2),
        # teacher trace-generation config (Gemini 3 Flash agent; generate_traces.*.yaml +
        # REPRODUCE_OLD_PAPER.md:19). knapsack teacher used the SAME per-turn cap (25) as the
        # binding eval cap -> no train->deploy cap shift. The generation YAML omits llm.temperature
        # and llm.timeout_s, so decoding uses the PROVIDER DEFAULT temperature (LLMConfig.temperature
        # =None is dropped in _build_payload) and the harness-default 60s per-LLM-call timeout; the
        # configured 300s is the per-EPISODE budget wrapping the whole agent.run (benchmark.py:111).
        # Determinism comes from response caching, not a decode seed (seed_start feeds instance
        # generation at generator.py:52, not the model).
        "teacher_model": "Gemini 3 Flash",
        "teacher_model_id": "gemini/gemini-3-flash-preview",
        "gen_cap": 25,
        "gen_max_turns": 40,
        "gen_timeout_s": 300,
        "gen_call_timeout_s": 60,
        "license_model": "Apache-2.0",
        # Llama-3.1-8B is NOT an OSI-approved open-source licence -- it is Meta's custom
        # community licence with acceptable-use and naming terms, stated separately rather
        # than lumped in with the other two.
        "license_model_mistral": "Apache-2.0",
        "license_model_llama31": "Llama 3.1 Community License",
        "license_task": "MIT",
        "seed0_cfg": 3407,
        "seed_777": 777,
        "seed_1337": 1337,
        "serve_window": 40960,
        "serve_gpu_util": 0.95,
        "serve_max_lora_rank": 64,
        "serve_temp": 0.2,
        "serve_out_tokens": 8192,
        "serve_timeout_s": 600,
        "filter_min_score": 0.5,
        "filter_finish_window": 3,
        "filter_dup_sim": 0.9,
        "filter_dup_lookback": 4,
        "filter_bad_err_density": 0.1,
        "prep_window": 16384,
        "prep_margin": 100,
        # actual paired-filter yield of the knapsack training set, from the documented
        # prepare_paired_data.py uncapped run (REPRODUCE_OLD_PAPER.md; valid=1109,
        # skipped=791 -> 1900 paired candidates, 58.37%); capped to 1000/regime for training.
        "paired_candidates": 1900,
        "paired_valid": 1109,
        "paired_yield_pct": round(1109 / 1900 * 100, 1),
        "data_cap_per_regime": 1000,
        "cap_reached_both": True,
        # prepare_paired_data.py: random.seed(42) -> sorted(common task IDs) -> shuffle ->
        # keep valid pairs in shuffled order until data_cap_per_regime reached.
        "prep_seed": 42,
        "tok_mean_stateless_k": 10.7,
        "tok_mean_persistent_k": 4.6,
        "tok_ratio": 2.3,
        "tok_max_stateless_k": 16.2,
        "tok_max_persistent_k": 12.9,
        "turns_mean_stateless": 6.1,
        "turns_mean_persistent": 4.3,
        "turns_max_stateless": 14,
        "turns_max_persistent": 20,
        "pilot_n": 20,
        "pilot_retained": 16,
        "pilot_pct": 80,
    }


def build_prior():
    return {
        "_source": "prior published baseline, hard-split normalized optimality metric",
        "persistent": 68.2,
        "stateless": 67.7,
    }


# ------------------------------------------------------------------ builders (computed)
def build_knap2x2():
    out: dict = {"perseed": {}, "agg": {}, "rep": {}}
    for seed in SEEDS:
        s = {}
        for cap in (CB, CS):
            m = _cells(seed, cap)
            if m:
                s[f"cap{cap}"] = {
                    k: m[k] for k in ("PP", "PS", "SP", "SS", "D", "GP", "n")
                }
        if s:
            out["perseed"][f"seed{seed}"] = s
    sd = _seed_data()
    # MAIN Table 1: seed-0 per-cell mean + paired-bootstrap CI (n=25 single grid)
    for cap in (CB, CS):
        random.seed(0)
        cells = sd["0"][cap]
        tasks = sorted(set.intersection(*[set(cells[cl]) for cl in cells]))
        out["agg"][f"cap{cap}"] = {}
        for cl in ("PP", "PS", "SP", "SS"):
            vals = [cells[cl][t] for t in tasks]
            lo, hi = A._bootci(vals)
            out["agg"][f"cap{cap}"][cl] = {
                "mean": statistics.mean(vals),
                "ci_lo": lo,
                "ci_hi": hi,
            }
        out["agg"][f"cap{cap}"]["n"] = len(tasks)
    # REPLICATION: 3-seed common-task crossed-bootstrap mean + CI per cell
    g8 = _grid(sd, SEEDS)
    for cap in (CB, CS):
        random.seed(0)
        out["rep"][f"cap{cap}"] = {}
        for cl in ("PP", "PS", "SP", "SS"):
            lists = [[sd[s][cap][cl][t] for t in g8] for s in SEEDS]
            mean, lo, hi = _crossed_mean_ci(lists)
            out["rep"][f"cap{cap}"][cl] = {"mean": mean, "ci_lo": lo, "ci_hi": hi}
        out["rep"][f"cap{cap}"]["n"] = len(g8)
    # PER-SEED cell means on the SAME common set (Table 7b: sign replication). The common set
    # is the first len(g8) task IDs in generation order (indices 0..len(g8)-1), completed by all
    # three seeds -> outcome-independent selection, not a completion-filtered subset.
    out["rep_perseed"] = {}
    out["rep_n"] = len(g8)
    out["rep_first_idx"] = max(
        int(t.split("-")[-1].split(".")[0]) for t in g8
    )  # = len-1 if prefix
    for seed in SEEDS:
        ps = {}
        for cap in (CB, CS):
            m = {
                cl: statistics.mean(sd[seed][cap][cl][t] for t in g8)
                for cl in ("PP", "PS", "SP", "SS")
            }
            m["D"] = (m["PP"] - m["PS"]) - (m["SP"] - m["SS"])
            ps[f"cap{cap}"] = m
        out["rep_perseed"][f"seed{seed}"] = ps
    return out


def _cells_by_task(seed, cap):
    """{cell: {task_id: score}} for a (seed, cap); task_id is the result-file basename."""
    base = (
        "experiments/cap_sweep/knapsack/qwen3_8b/main"
        if seed == "0"
        else "experiments/cap_sweep/knapsack/qwen3_8b/seeds"
    )
    pfx = "" if seed == "0" else f"s{seed}_"
    return {
        cl: A._scores(f"{base}/{pfx}{cl}_cap{cap}", KFAM)
        for cl in ("PP", "PS", "SP", "SS")
    }


def _D_at(cells, tasks):
    m = {cl: statistics.mean(cells[cl][t] for t in tasks) for cl in cells}
    return (m["PP"] - m["PS"]) - (m["SP"] - m["SS"])


def _seed_data():
    return {s: {CB: _cells_by_task(s, CB), CS: _cells_by_task(s, CS)} for s in SEEDS}


def _grid(sd, seeds):
    """common task IDs across the given seeds x 4 cells x both caps."""
    return sorted(
        set.intersection(
            *[
                set(sd[s][cap][cl])
                for s in seeds
                for cap in (CB, CS)
                for cl in ("PP", "PS", "SP", "SS")
            ]
        )
    )


def build_interaction():
    # NOTE (reviewer v2 #3): report seed-0 (n=25) as the MAIN high-powered single-seed
    # decomposition, and the 3-seed COMMON-task-set analysis strictly as replication; never
    # pool the n=25 and n=8 grids together. Both bootstraps are task-paired across caps.
    sd = _seed_data()
    # ---- MAIN: seed-0, paired over its cross-cap common tasks ----
    g0 = _grid(sd, ["0"])
    b0, s0 = sd["0"][CB], sd["0"][CS]
    Db0, Ds0 = _D_at(b0, g0), _D_at(s0, g0)
    GPb0 = statistics.mean(b0["PP"][t] - b0["PS"][t] for t in g0)
    GPs0 = statistics.mean(s0["PP"][t] - s0["PS"][t] for t in g0)
    random.seed(0)
    n0 = len(g0)
    Tb = sorted(
        (lambda idx: _D_at(b0, idx) - _D_at(s0, idx))(
            [g0[random.randrange(n0)] for _ in range(n0)]
        )
        for _ in range(5000)
    )
    # ---- REPLICATION: 3-seed, common task set, crossed (seed x task) bootstrap ----
    g8 = _grid(sd, SEEDS)
    Tps = {s: _D_at(sd[s][CB], g8) - _D_at(sd[s][CS], g8) for s in SEEDS}
    random.seed(0)
    n8 = len(g8)
    Trep = []
    for _ in range(5000):
        bs = [SEEDS[random.randrange(len(SEEDS))] for _ in SEEDS]
        idx = [g8[random.randrange(n8)] for _ in range(n8)]
        Trep.append(
            statistics.mean(_D_at(sd[s][CB], idx) - _D_at(sd[s][CS], idx) for s in bs)
        )
    Trep.sort()
    return {
        "main": {
            "D_bind": Db0,
            "D_slack": Ds0,
            "GP_bind": GPb0,
            "GP_slack": GPs0,
            "T": Db0 - Ds0,
            "T_ci_lo": Tb[125],
            "T_ci_hi": Tb[4874],
            "n": n0,
        },
        "rep": {
            "T_perseed": {f"seed{s}": Tps[s] for s in SEEDS},
            "T_mean": statistics.mean(Tps.values()),
            "T_pooled": statistics.mean(Trep),
            "T_ci_lo": Trep[125],
            "T_ci_hi": Trep[4874],
            "n": n8,
        },
    }


def _msel(cells, tasks):
    return [cells["PS"][t] - cells["SS"][t] for t in tasks]


def build_modelsel():
    # M(c) = Q_PS - Q_SS under stateless deploy. Main = seed-0 (n=25); replication = 3-seed common-8.
    sd = _seed_data()
    # ---- MAIN: seed-0 ----
    gb = sorted(set(sd["0"][CB]["PS"]) & set(sd["0"][CB]["SS"]))
    gs = sorted(set(sd["0"][CS]["PS"]) & set(sd["0"][CS]["SS"]))
    m25, m80 = _msel(sd["0"][CB], gb), _msel(sd["0"][CS], gs)
    random.seed(0)
    m25_ci, m80_ci = A._bootci(m25), A._bootci(m80)
    g0 = _grid(sd, ["0"])  # cross-cap grid for a paired dM
    p25, p80 = _msel(sd["0"][CB], g0), _msel(sd["0"][CS], g0)
    random.seed(0)
    n = len(g0)
    dboot = sorted(
        (
            lambda ix: statistics.mean(p80[i] for i in ix)
            - statistics.mean(p25[i] for i in ix)
        )([random.randrange(n) for _ in range(n)])
        for _ in range(5000)
    )
    main = {
        "M25": statistics.mean(m25),
        "M25_ci_lo": m25_ci[0],
        "M25_ci_hi": m25_ci[1],
        "M80": statistics.mean(m80),
        "M80_ci_lo": m80_ci[0],
        "M80_ci_hi": m80_ci[1],
        "dM": statistics.mean(p80) - statistics.mean(p25),
        "dM_ci_lo": dboot[125],
        "dM_ci_hi": dboot[4874],
        "n": len(gb),
    }
    # ---- REPLICATION: 3-seed common-8 ----
    g8 = _grid(sd, SEEDS)
    pb = {s: _msel(sd[s][CB], g8) for s in SEEDS}
    ps = {s: _msel(sd[s][CS], g8) for s in SEEDS}
    random.seed(0)
    Mb, _, _ = _crossed_mean_ci([pb[s] for s in SEEDS])
    random.seed(1)
    Ms, _, _ = _crossed_mean_ci([ps[s] for s in SEEDS])
    rep = {
        "M25": Mb,
        "M80": Ms,
        "n": len(g8),
        "M25_perseed": {f"seed{s}": statistics.mean(pb[s]) for s in SEEDS},
        "M80_perseed": {f"seed{s}": statistics.mean(ps[s]) for s in SEEDS},
        "dM_perseed": {
            f"seed{s}": statistics.mean(ps[s]) - statistics.mean(pb[s]) for s in SEEDS
        },
    }
    return {"main": main, "rep": rep}


def build_dense():
    caps = [10, 20, 25, 30, 40, 50, 80, 100]
    # task-keyed scores per cap; the dense sweep uses the SAME task IDs at every cap.
    # per_cap    = P->S (persistent adapter on the STATELESS runtime, mismatched);
    # per_cap_pp = P->P (same adapter on the PERSISTENT runtime, matched) on the SAME 12 task IDs.
    per_cap = {
        c: A._scores(f"experiments/cap_sweep/knapsack/qwen3_8b/dense/persistent_cap{c}", KFAM)
        for c in caps
    }
    per_cap_pp = {
        c: A._scores(f"experiments/cap_sweep/knapsack/qwen3_8b/dense/matched_cap{c}", KFAM) for c in caps
    }
    common = sorted(
        set.intersection(
            *[set(per_cap[c]) for c in caps], *[set(per_cap_pp[c]) for c in caps]
        )
    )
    if not common:
        missing = (
            "P->S (persistent_cap*)"
            if not any(per_cap.values())
            else "P->P (matched_cap*)"
        )
        raise SystemExit(
            f"build_dense: no common task IDs across the dense cells -- the {missing} per-task "
            "result JSONs are missing. Their results/ dirs are gitignored; unpack the eval-data "
            "mirror into experiments/cap_sweep/ (it includes the matched_cap* cells; see the "
            "Released artifacts section of the README)."
        )
    xs = [math.log(c) for c in caps]
    xbar = sum(xs) / len(xs)

    def fe_slope(task_ids, pc):
        # within (task-fixed-effects) estimator of beta in Q_ic = alpha_i + beta*log c
        num = den = 0.0
        for t in task_ids:
            yi = [pc[c][t] for c in caps]
            ybar = sum(yi) / len(yi)
            for x, y in zip(xs, yi):
                num += (x - xbar) * (y - ybar)
                den += (x - xbar) ** 2
        return num / den

    NB = 5000  # matches the advertised config.bootstrap_resamples (A._bootci also uses 5000)

    def block(pc):
        # task-fixed-effects slope + task-ID bootstrap (resample task IDs, retain each task's full
        # cap-vector -> preserves pairing) + per-cap mean/CI on the common task set.
        slope = fe_slope(common, pc)
        random.seed(0)
        boots = sorted(
            fe_slope([common[random.randrange(len(common))] for _ in common], pc)
            for _ in range(NB)
        )
        per = {}
        for c in caps:
            vals = [pc[c][t] for t in common]
            clo, chi = A._bootci(vals)
            per[f"c{c}"] = {"mean": statistics.mean(vals), "ci_lo": clo, "ci_hi": chi}
        return slope, boots[int(0.025 * NB)], boots[int(0.975 * NB)], per

    ps_slope, ps_lo, ps_hi, ps_per = block(
        per_cap
    )  # P->S (identical to prior single-curve output)
    pp_slope, pp_lo, pp_hi, pp_per = block(per_cap_pp)  # P->P matched dense curve

    # Interaction = the paired slope CONTRAST beta_PS - beta_PP (not either marginal slope).
    # Bootstrap resamples task IDs and, for each resampled task, uses BOTH its P->S and P->P
    # cap-vectors together -> preserves the within-task, between-cell covariance.
    def slope_diff(task_ids):
        return fe_slope(task_ids, per_cap) - fe_slope(task_ids, per_cap_pp)

    random.seed(0)
    dboots = sorted(
        slope_diff([common[random.randrange(len(common))] for _ in common])
        for _ in range(NB)
    )
    return {
        "caps": caps,
        "slope": ps_slope,
        "slope_ci_lo": ps_lo,
        "slope_ci_hi": ps_hi,
        "slope_pp": pp_slope,
        "slope_pp_ci_lo": pp_lo,
        "slope_pp_ci_hi": pp_hi,
        "slope_diff": slope_diff(common),
        "slope_diff_ci_lo": dboots[int(0.025 * NB)],
        "slope_diff_ci_hi": dboots[int(0.975 * NB)],
        "n": len(common),
        "PP": pp_per,
        **ps_per,
    }


def _status_map(path, fam=None):
    """{result-file basename: top-level status} matching A._scores' keying."""
    fam = fam or KFAM
    out = {}
    for f in glob.glob(f"{path}/**/{fam}-{fam}-*.json", recursive=True):
        if f.endswith(".trace.json"):
            continue
        out[os.path.basename(f)] = (json.load(open(f)) or {}).get("status")
    return out


def build_tmax():
    # reviewer v2 #6: recompute ALL horizons on the SAME task set (the common IDs across
    # T_max=40/80/160) and report paired-delta CIs.
    #
    # codex P1 (#119): the horizon sweep is NOT a pure "only max_turns varies" control --
    # the T40 configs cap episode wall-clock at 3000s while T80/T160 use 4000s, and that
    # ceiling BINDS for P->S (most long-horizon P->S episodes end in `timeout`). So we also
    # report the per-cell timeout incidence and frame the sweep as a JOINT turn-and-time
    # increase that the mismatched agent still cannot convert into quality.
    out: dict = {"wall_t40": 3000, "wall_long": 4000}
    for cell in ("PP", "PS"):
        t40 = A._scores(f"experiments/cap_sweep/knapsack/qwen3_8b/main/{cell}_cap25", KFAM)
        t80 = A._scores(f"experiments/cap_sweep/knapsack/qwen3_8b/tmax/{cell}_cap25_T80", KFAM)
        t160 = A._scores(f"experiments/cap_sweep/knapsack/qwen3_8b/tmax/{cell}_cap25_T160", KFAM)
        s80 = _status_map(f"experiments/cap_sweep/knapsack/qwen3_8b/tmax/{cell}_cap25_T80")
        s160 = _status_map(f"experiments/cap_sweep/knapsack/qwen3_8b/tmax/{cell}_cap25_T160")
        common = sorted(set(t40) & set(t80) & set(t160))
        n = len(common)
        random.seed(0)
        d80 = sorted(
            (lambda ix: statistics.mean(t80[common[i]] - t40[common[i]] for i in ix))(
                [random.randrange(n) for _ in range(n)]
            )
            for _ in range(5000)
        )
        random.seed(1)
        d160 = sorted(
            (lambda ix: statistics.mean(t160[common[i]] - t40[common[i]] for i in ix))(
                [random.randrange(n) for _ in range(n)]
            )
            for _ in range(5000)
        )
        out[cell] = {
            "T40": statistics.mean(t40[k] for k in common),
            "T80": statistics.mean(t80[k] for k in common),
            "T160": statistics.mean(t160[k] for k in common),
            "n": n,
            "d80": statistics.mean(t80[k] - t40[k] for k in common),
            "d80_ci_lo": d80[125],
            "d80_ci_hi": d80[4874],
            "d160": statistics.mean(t160[k] - t40[k] for k in common),
            "d160_ci_lo": d160[125],
            "d160_ci_hi": d160[4874],
            # timeout incidence on the common set (wall-clock ceiling binding)
            "to80": sum(s80.get(k) == "timeout" for k in common),
            "to160": sum(s160.get(k) == "timeout" for k in common),
        }
    return out


def build_nearthresh():
    def mn(d):
        v = _mean_scores(d)
        return (statistics.mean(v), len(v)) if v else (None, 0)

    t40, n40 = mn("experiments/cap_sweep/knapsack/qwen3_8b/tmax/PS_cap40_T40")
    t80, n80 = mn("experiments/cap_sweep/knapsack/qwen3_8b/tmax/PS_cap40_T80")
    return {"PS_T40": t40, "PS_T40_n": n40, "PS_T80": t80, "PS_T80_n": n80}


def build_bandwidth():
    """The navigation reconstruction-bandwidth intervention (Sec. 5, Prediction 1).

    Two arms over the same 16 graphs, adapter, budgets and scorer, differing only in how
    many nodes one neighbors() call may query (50 vs the preselected 2). Reported with the
    manipulation check that says whether the interface change actually moved realized
    action width, and with the drift re-run that bounds how much of the contrast is
    single-rollout noise."""
    arms = {
        "batched": "experiments/cap_sweep/navigation/qwen3_8b/main/nav_{cell}_cap{cap}",
        "reduced": "experiments/cap_sweep/navigation/qwen3_8b/batch2/navb2_{cell}_cap{cap}",
    }
    cells = {
        (arm, cl, cap): A._scores(tpl.format(cell=cl, cap=cap), "navigation")
        for arm, tpl in arms.items()
        for cl in ("PP", "PS")
        for cap in (CB, CS)
    }
    rerun = A._scores(
        "experiments/cap_sweep/navigation/qwen3_8b/main/nav_PS_cap25_rerun", "navigation"
    )
    if not all(cells.values()):
        return {
            "_source": "navigation bandwidth intervention -- cells absent",
            "ran": False,
        }
    common = sorted(set.intersection(*[set(v) for v in cells.values()]))

    def amp(arm, t, ps25=None):
        bind = cells[(arm, "PP", CB)][t] - (
            ps25[t] if ps25 else cells[(arm, "PS", CB)][t]
        )
        slack = cells[(arm, "PP", CS)][t] - cells[(arm, "PS", CS)][t]
        return bind - slack

    def dbatch(ps25=None):
        per = [amp("reduced", t) - amp("batched", t, ps25) for t in common]
        random.seed(0)
        lo, hi = A._bootci(per)
        return {"D": statistics.mean(per), "ci_lo": lo, "ci_hi": hi}

    out = {
        "_source": "experiments/cap_sweep/navigation/qwen3_8b/{main,batch2} + the drift re-run; "
        "batch size preselected by scripts/nav_batch_sweep.py",
        "ran": True,
        "n": len(common),
        "batch": 2,
        "cells": {
            f"{arm}_{cl}_{cap}": statistics.mean(
                cells[(arm, cl, cap)][t] for t in common
            )
            for arm, cl, cap in cells
        },
        "dbatch": dbatch(),
        "dbatch_rerun_control": dbatch(rerun),
    }
    for arm in arms:
        gb = out["cells"][f"{arm}_PP_{CB}"] - out["cells"][f"{arm}_PS_{CB}"]
        gs = out["cells"][f"{arm}_PP_{CS}"] - out["cells"][f"{arm}_PS_{CS}"]
        out[f"GP_bind_{arm}"], out[f"GP_slack_{arm}"], out[f"AP_{arm}"] = (
            gb,
            gs,
            gb - gs,
        )
    if rerun and common:
        d = [rerun[t] - cells[("batched", "PS", CB)][t] for t in common if t in rerun]
        random.seed(0)
        lo, hi = A._bootci(d)
        out["drift"] = {
            "n": len(d),
            "archived": statistics.mean(
                cells[("batched", "PS", CB)][t] for t in common if t in rerun
            ),
            "rerun": statistics.mean(rerun[t] for t in common if t in rerun),
            "diff": statistics.mean(d),
            "ci_lo": lo,
            "ci_hi": hi,
            "identical": sum(
                1
                for t in common
                if t in rerun and abs(rerun[t] - cells[("batched", "PS", CB)][t]) < 0.01
            ),
        }
    return out


def build_exposure():
    """Realized per-turn call geometry -- the Sec. 5 mechanism variable.

    The paper previously ordered the families by a NOMINAL replay demand R; the traces say
    that quantity does not predict whether the cap binds. What does is how wide the policy's
    action blocks actually are, so these are the numbers Sec. 5 now rests on:

      exposure  E_25 = Pr(K_t > 25) and the realized cap-hit rate, per family/cell
      resume    what the next turn does after a truncation, per cell, and the P->S-minus-
                P->P contrast R_c that gate 2 actually claims
      intent    the first block's width under each ANNOUNCED cap, with the cap unenforced
      sweep     the offline batch-size selection for the navigation intervention

    Everything comes from scripts/mechanism_traces.py, which replays the recorded blocks
    through the real environments; nothing here is a new run."""
    from scripts.mechanism_traces import run_all  # type: ignore[reportMissingImports]
    from scripts.nav_batch_sweep import main as sweep_main  # type: ignore[reportMissingImports]

    fams = ["knapsack", "navigation", "rule_diagnosis"]

    # include the intervention arm only once ALL of its cells are complete: a directory
    # exists as soon as a cell starts, and replaying a half-written trace from a live run
    # is how you get numbers that change under you
    def _complete(cell, cap, want=16):
        pat = (
            f"experiments/cap_sweep/navigation/qwen3_8b/batch2/navb2_{cell}_cap{cap}"
            "/**/navigation-navigation-*.json"
        )
        return (
            len(
                [
                    f
                    for f in glob.glob(pat, recursive=True)
                    if not f.endswith(".trace.json")
                ]
            )
            >= want
        )

    if all(_complete(c, p) for c in ("PP", "PS") for p in (25, 80)):
        fams.append("navigation_batch2")
    tables = asyncio.run(run_all(fams, verbose=False))
    sweep = asyncio.run(sweep_main())
    return {
        "_source": "scripts/mechanism_traces.py (trace-conditioned replay) + "
        "scripts/nav_batch_sweep.py (offline batch-size selection)",
        "cap_exposure": tables.get("cap_exposure", {}),
        "replay_novel": tables.get("replay_novel", {}),
        "first_block_intent": tables.get("first_block_intent", {}),
        "rule_repair": tables.get("rule_repair", {}),
        "resume": tables.get("resume", {}),
        # gate 2 as a contrast rather than a level: R_c = Pr(restart | P->S, cap hit) -
        # Pr(restart | P->P, cap hit), the statistic that separates mismatch-specific
        # failure from generic cap failure
        "resume_delta": tables.get("resume_delta", {}),
        "nav_stages": tables.get("nav_stages", {}),
        "batch_sweep": {
            f"{cell}_b{b}": row for (cell, b), row in sweep.items() if cell == "PS"
        },
        "batch_chosen": 2,
    }


def build_replay():
    cells = {
        "PS25": (
            "experiments/cap_sweep/knapsack/qwen3_8b/main/PS_cap25/**/knapsack-knapsack-*.trace.json",
            False,
            25,
        ),
        "PP25": (
            "experiments/cap_sweep/knapsack/qwen3_8b/main/PP_cap25/**/knapsack-knapsack-*.trace.json",
            True,
            25,
        ),
        "PS40": (
            "experiments/cap_sweep/knapsack/qwen3_8b/dense/persistent_cap40/**/knapsack-knapsack-*.trace.json",
            False,
            40,
        ),
    }

    async def one(pat, persist, cap):
        eps = [
            await _replay_one(t, persist, cap)
            for t in sorted(glob.glob(pat, recursive=True))
        ]
        eps = [e for e in eps if e]

        def m(k):
            return statistics.mean(e[k] for e in eps)

        return {
            "n": len(eps),
            "calls": m("ncalls"),
            "unique": m("unique"),
            "replay": m("replay"),
            "replay_frac": m("replay") / max(m("ncalls"), 1e-9),
            "achieved": statistics.mean(e["agent_val"] / e["opt_global"] for e in eps),
            "coverage": statistics.mean(e["opt_q"] / e["opt_global"] for e in eps),
            "decision": statistics.mean(
                e["agent_val"] / e["opt_q"] for e in eps if e["opt_q"] > 0
            ),
        }

    async def run():
        return {k: await one(*v) for k, v in cells.items()}

    return asyncio.run(run())


def build_checkpoint():
    """The cap-boundary-carryover intervention.

    Three cells at c=25 over the same knapsack instances, differing only in what the
    runtime keeps at a cap-truncated turn boundary. Reports the pre-specified rescue
    fraction with its paired bootstrap, and the behavioural panel that says whether a
    quality change (or its absence) came with the predicted change in what the policy does
    after the cap. `ran: False` until the cells exist, so the paper can be rendered from a
    partial archive."""
    import scripts.checkpoint_analysis as C

    scores = C._scores()
    expected = C._task_ids()
    problems = C._cohort_problems(scores, expected) if expected else ["no task cohort"]
    if problems:
        # same rule as the standalone script: matching COUNTS is not matching cohorts
        return {
            "_source": "cap-boundary carryover -- cells absent or partial",
            "ran": False,
            "problems": problems,
            "expected": len(expected),
        }
    common, vals = C._paired(scores)
    primary = C._primary(vals, len(common))

    # the announced-carryover addendum, once its two cells exist
    ann_scores = C._scores(C.ANN_CELLS)
    ann_ready = any(ann_scores.values()) and not C._cohort_problems(ann_scores, expected)
    announced = None
    if ann_ready:
        common_ann, vals_ann = C._paired(ann_scores)
        announced = C._announced(
            vals_ann,
            len(common_ann),
            statistics.mean(scores["PSckpt"][k] for k in common_ann),
        )
    behaviour = asyncio.run(
        C._behaviour(C.CELLS + (C.ANN_CELLS if ann_ready else ()))
    )
    # a NaN rho (no reference gap in the resample) must not reach the JSON as a bare NaN
    if primary["rho"] != primary["rho"]:
        primary["rho"] = None
    return {
        "_source": "experiments/cap_sweep/knapsack/qwen3_8b/checkpoint via scripts/checkpoint_analysis.py "
        "(design and analysis plan fixed before the cells ran)",
        "ran": True,
        **primary,
        "announced": announced,
        "behaviour": behaviour,
        "archived": {
            c: statistics.mean(
                A._scores(f"experiments/cap_sweep/knapsack/qwen3_8b/main/{c}_cap{CB}", KFAM).values()
            )
            for c in ("PP", "PS")
        },
    }


def build_arms():
    """Cross-arm inference over the shared 25-instance cohort (scripts/rollout_variance.py).

    Two questions, one bootstrap: how far every estimand moves when only the decode draw
    changes (the paper's main Qwen run vs its independent second rollout), and whether it
    survives a change of base model (Mistral-7B-v0.3, Llama-3.1-8B on the same recipe).
    Each block reports `ran: False` until that arm's per-task results are present, so the
    paper still renders from a partial archive."""
    import scripts.rollout_variance as V

    def block(arms):
        loaded = {}
        for name in arms:
            got = V._load(V.ARMS[name])
            if got is None:
                return {"ran": False, "missing": name}
            loaded[name] = got
        tasks = V._common_tasks(loaded)
        out = {"ran": True, "n": len(tasks), **V.analyse(loaded, tasks)}
        if len(loaded) == 2:
            out["cells"] = V.cell_retest(loaded, tasks)
            mags = [abs(r["delta"]) for r in out["cells"].values()]
            # how far a CELL moves, which is the quantity a reader compares an effect size
            # against; the estimand-level intervals above answer a different question
            out["cells_mean_abs"] = statistics.mean(mags)
            out["cells_max_abs"] = max(mags)
        return out

    return {
        "_source": "scripts/rollout_variance.py over experiments/cap_sweep/"
        "knapsack/{qwen3_8b/{main,rollout2},mistral_7b/main,llama31_8b/main}",
        "retest": block(["r1", "r2"]),
        "crossmodel": block(["r1", "mistral", "llama31"]),
    }


def _at(cum, t):
    # cumulative value at turn t, holding the episode's last value past its end (fixed cohort)
    ks = [k for k in cum if k <= t]
    return cum[max(ks)] if ks else 0


async def _progress_curve(cell, persistent, turns):
    """From the trace-conditioned replay (executed calls, not source expressions):
    per-turn cumulative distinct items and executed inspect attempts, averaged over a
    fixed episode cohort (ended episodes carry their final value forward -> nondecreasing)."""
    pat = f"experiments/cap_sweep/knapsack/qwen3_8b/main/{cell}_cap{CB}/**/{KFAM}-{KFAM}-*.trace.json"
    eps = [
        await _replay_one(t, persistent, CB)
        for t in sorted(glob.glob(pat, recursive=True))
    ]
    eps = [e for e in eps if e]
    distinct = {str(t): statistics.mean(_at(e["cum_u"], t) for e in eps) for t in turns}
    attempts = {
        str(t): statistics.mean(_at(e["cum_u"], t) + _at(e["cum_r"], t) for e in eps)
        for t in turns
    }
    return {
        "distinct": distinct,
        "attempts": attempts,
        "distinct_final": statistics.mean(e["unique"] for e in eps),
        "attempts_final": statistics.mean(e["ncalls"] for e in eps),
    }


def build_progress():
    turns = [0, 5, 10, 20, 30, 39]

    async def run():
        return {
            "turns": turns,
            "PS": await _progress_curve("PS", False, turns),
            "PP": await _progress_curve("PP", True, turns),
        }

    return asyncio.run(run())


def build_context():
    cells = {
        "PP25_T40": "experiments/cap_sweep/knapsack/qwen3_8b/main/PP_cap25/**/knapsack-knapsack-*.trace.json",
        "PS25_T40": "experiments/cap_sweep/knapsack/qwen3_8b/main/PS_cap25/**/knapsack-knapsack-*.trace.json",
        "SP25_T40": "experiments/cap_sweep/knapsack/qwen3_8b/main/SP_cap25/**/knapsack-knapsack-*.trace.json",
        "SS25_T40": "experiments/cap_sweep/knapsack/qwen3_8b/main/SS_cap25/**/knapsack-knapsack-*.trace.json",
        "PS25_T80": "experiments/cap_sweep/knapsack/qwen3_8b/tmax/PS_cap25_T80/**/knapsack-knapsack-*.trace.json",
        "PS25_T160": "experiments/cap_sweep/knapsack/qwen3_8b/tmax/PS_cap25_T160/**/knapsack-knapsack-*.trace.json",
    }
    out = {}
    for key, pat in cells.items():
        eps = []
        for t in glob.glob(pat, recursive=True):
            evs = _load(t).get("events", [])
            pt = [
                e["data"]["prompt_tokens"]
                for e in evs
                if e.get("type") == "ModelCallEvent"
                and (e.get("data") or {}).get("prompt_tokens") is not None
            ]
            if not pt:
                continue
            ovf = sum(
                1
                for e in evs
                if e.get("type") == "ErrorEvent"
                and "ContextWindow" in str((e.get("data") or {}).get("error") or "")
            )
            eps.append(
                (max(pt), 1 if any(x > 30000 for x in pt) else 0, 1 if ovf else 0)
            )
        if eps:
            out[key] = {
                "n": len(eps),
                "max_input": max(e[0] for e in eps),
                "over30k": sum(e[1] for e in eps),
                "overflow": sum(e[2] for e in eps),
            }
    return out


def build_operational():
    import re

    out = {}
    for cap in (CB, CS):
        for cell in ("PP", "PS", "SP", "SS"):
            base = f"experiments/cap_sweep/knapsack/qwen3_8b/main/{cell}_cap{cap}"
            status = {}
            for f in glob.glob(f"{base}/**/{KFAM}-{KFAM}-*.json", recursive=True):
                if f.endswith(".trace.json"):
                    continue
                d = _load(f)
                r = d.get("result", {}) or {}
                status[os.path.basename(f)] = {
                    "status": d.get("status"),
                    "insp": (r.get("metrics") or {}).get("inspected_count"),
                    "score": r.get("score"),
                }
            rows = []
            for t in glob.glob(f"{base}/**/{KFAM}-{KFAM}-*.trace.json", recursive=True):
                tr = _load(t)
                s = tr.get("summary", {}) or {}
                steps = [
                    e for e in tr.get("events", []) if e.get("type") == "StepEvent"
                ]
                n = s.get("num_steps") or len(steps)

                def err(e):
                    return str(
                        (e.get("data", {}).get("interpreter_result") or {}).get("error")
                        or ""
                    )

                code_all = "\n".join(
                    (e.get("data", {}) or {}).get("code") or "" for e in steps
                )
                rows.append(
                    {
                        "turns": n,
                        "caphit": sum(
                            1 for e in steps if "Tool call limit exceeded" in err(e)
                        )
                        / max(n, 1),
                        "nameerr": sum(1 for e in steps if "NameError" in err(e)),
                        "restarts": len(re.findall(r"\blist_items\s*\(", code_all)),
                        "tokens": (s.get("token_usage") or {}).get("total_tokens"),
                        "finish": s.get("finish_reason"),
                        "base": os.path.basename(t).replace(".trace.json", ".json"),
                    }
                )
            if not rows:
                continue

            def g(k):
                return statistics.mean(r[k] for r in rows if r[k] is not None)

            insp = statistics.mean(
                status[r["base"]]["insp"]
                for r in rows
                if status.get(r["base"], {}).get("insp") is not None
            )
            sc = statistics.mean(
                status[r["base"]]["score"] for r in rows if r["base"] in status
            )
            term = Counter(
                "timeout"
                if status.get(r["base"], {}).get("status") == "timeout"
                else (r["finish"] or "none")
                for r in rows
            )
            out[f"{cell}{cap}"] = {
                "turns": g("turns"),
                "restarts": g("restarts"),
                "caphit_pct": g("caphit") * 100,
                "nameerr": g("nameerr"),
                "tokens_k": g("tokens") / 1000,
                "insp": insp,
                "score": sc,
                "fin": term.get("finish_tool", 0),
                "turnlimit": term.get("max_turns", 0),
                "timeout": term.get("timeout", 0),
            }
    return out


def build_mechanism():
    def cells(base, prefix, fam, metric):
        return {
            (cl, cap): A._scores(f"{base}/{prefix}{cl}_cap{cap}", fam, metric)
            for cl in ("PP", "PS")
            for cap in (CB, CS)
        }

    def AP(c):
        common = sorted(
            set(c[("PP", CB)])
            & set(c[("PS", CB)])
            & set(c[("PP", CS)])
            & set(c[("PS", CS)])
        )
        d = [
            (c[("PP", CB)][k] - c[("PS", CB)][k])
            - (c[("PP", CS)][k] - c[("PS", CS)][k])
            for k in common
        ]
        gpb = statistics.mean(c[("PP", CB)][k] - c[("PS", CB)][k] for k in common)
        gps = statistics.mean(c[("PP", CS)][k] - c[("PS", CS)][k] for k in common)
        lo, hi = A._bootci(d)
        return statistics.mean(d), lo, hi, gpb, gps

    def Rmean(td, fam, field, tr):
        vals = [
            tr(_load(f).get("public", {})[field])
            for f in glob.glob(f"experiments/cap_sweep/{td}/tasks/{fam}/*.json")
            if field in (_load(f).get("public", {}) or {})
        ]
        return statistics.mean(vals) if vals else float("nan")

    specs = [
        (
            "knap",
            "knapsack/qwen3_8b/main",
            "",
            "knapsack",
            "score",
            "knapsack/task_defs",
            "inspect_budget",
            lambda x: x,
        ),
        (
            "rule_score",
            "rule_diagnosis/qwen3_8b/main",
            "rule_",
            "rule_diagnosis",
            "score",
            "rule_diagnosis/task_defs",
            "probe_budget",
            lambda x: x,
        ),
        (
            "rule_bndF1",
            "rule_diagnosis/qwen3_8b/main",
            "rule_",
            "rule_diagnosis",
            "boundary_f1",
            "rule_diagnosis/task_defs",
            "probe_budget",
            lambda x: x,
        ),
        (
            "nav",
            "navigation/qwen3_8b/main",
            "nav_",
            "navigation",
            "score",
            "navigation/task_defs",
            "n",
            lambda x: math.ceil(x / 50),
        ),
    ]
    random.seed(0)
    out = {}
    for name, base, pfx, fam, metric, td, field, tr in specs:
        c = cells(f"experiments/cap_sweep/{base}", pfx, fam, metric)
        ap, lo, hi, gpb, gps = AP(c)
        R = Rmean(td, fam, field, tr)
        # R lower/upper bounds (nav R and rule R are not directly comparable).
        # For navigation the scalar R=ceil(n/50) counts only the FREE adjacency rebuild and
        # OMITS the budgeted key/trap/lock probes; the upper bound adds probe_budget so nav's R
        # is put on a comparable footing with rule (whose R already = the full probe_budget).
        R_lo = R_hi = R
        if name == "nav":

            def _nav_hi(f):
                p = _load(f).get("public", {}) or {}
                return math.ceil(p["n"] / 50) + (p.get("probe_budget") or 0)

            vals = [
                _nav_hi(f)
                for f in glob.glob(f"experiments/cap_sweep/{td}/tasks/{fam}/*.json")
                if "n" in (_load(f).get("public", {}) or {})
            ]
            R_hi = statistics.mean(vals) if vals else float("nan")
        out[name] = {
            "GP_bind": gpb,
            "GP_slack": gps,
            "AP": ap,
            "AP_ci_lo": lo,
            "AP_ci_hi": hi,
            "R": R,
            "Rc": R / CB,
            "R_lo": R_lo,
            "R_hi": R_hi,
            "Rc_lo": R_lo / CB,
            "Rc_hi": R_hi / CB,
        }
    return out


def build_timeout_sens():
    # Sensitivity of the model-selection contrast M(c)=Q_PS-Q_SS (seed-0,
    # knapsack/qwen3_8b/main, normalized optimality) to timeout episodes. A timed-out episode RETAINS its (low)
    # partial score rather than being zeroed (benchmark.py evaluates unconditionally); we recompute
    # M with task-pairs where PS or SS timed out excluded, to check the slack-cap point estimate.
    def load(cell, cap):
        base = f"experiments/cap_sweep/knapsack/qwen3_8b/main/{cell}_cap{cap}"
        out = {}
        for f in glob.glob(f"{base}/**/{KFAM}-{KFAM}-*.json", recursive=True):
            if f.endswith(".trace.json"):
                continue
            d = _load(f)
            r = d.get("result", {}) or {}
            out[os.path.basename(f)] = {
                "status": d.get("status"),
                "score": r.get("score"),
            }
        return out

    data = {(cell, cap): load(cell, cap) for cell in ("PS", "SS") for cap in (CB, CS)}

    def scored(cell, cap, k):
        return data[(cell, cap)].get(k, {}).get("score") is not None

    def M(cap, ks):
        return statistics.mean(
            data[("PS", cap)][k]["score"] - data[("SS", cap)][k]["score"] for k in ks
        )

    res = {}
    # Per-cap point estimates (valid within a cap): M(c) over that cap's PS/SS common set, and
    # the same set with that cap's timeout episodes excluded. Used for the slack-cap point-estimate
    # statement (dropping the slack timeouts moves M(80) toward 0).
    for cap in (CB, CS):
        ps, ss = data[("PS", cap)], data[("SS", cap)]
        common = sorted(
            k
            for k in (set(ps) & set(ss))
            if scored("PS", cap, k) and scored("SS", cap, k)
        )
        keep = [
            k
            for k in common
            if ps[k]["status"] != "timeout" and ss[k]["status"] != "timeout"
        ]
        res[f"cap{cap}"] = {
            "M_all": M(cap, common),
            "M_excl": M(cap, keep) if keep else float("nan"),
            "n": len(common),
            "n_kept": len(keep),
            "n_excl": len(common) - len(keep),
            "to_PS": sum(1 for k in common if ps[k]["status"] == "timeout"),
            "to_SS": sum(1 for k in common if ss[k]["status"] == "timeout"),
        }
    # ONE paired cohort for dM (Codex v7): the cross-cap intersection of tasks scored in PS AND SS
    # at BOTH caps, then EXCLUDE every task that timed out (PS or SS) at EITHER cap. This keeps the
    # dM_full vs dM_kept comparison on a single fixed cohort, so it isolates the cap effect from
    # task-composition differences.
    cells_caps = [(cell, cap) for cell in ("PS", "SS") for cap in (CB, CS)]
    cohort = sorted(
        k
        for k in set.intersection(*[set(data[cc]) for cc in cells_caps])
        if all(scored(cell, cap, k) for cell, cap in cells_caps)
    )
    kept = [
        k
        for k in cohort
        if all(data[cc][k]["status"] != "timeout" for cc in cells_caps)
    ]

    # Per-task paired dM contribution = [PS(slack)-SS(slack)] - [PS(bind)-SS(bind)]; its mean is dM,
    # and a task-paired bootstrap of it gives a CI on the retained cohort (Codex v7: report
    # uncertainty before calling moderation "robust").
    def deltas(ks):
        return [
            (data[("PS", CS)][k]["score"] - data[("SS", CS)][k]["score"])
            - (data[("PS", CB)][k]["score"] - data[("SS", CB)][k]["score"])
            for k in ks
        ]

    random.seed(0)
    dk_lo, dk_hi = A._bootci(deltas(kept))
    random.seed(0)
    df_lo, df_hi = A._bootci(deltas(cohort))
    res["cohort"] = {
        "n_full": len(cohort),
        "n_kept": len(kept),
        "n_excl": len(cohort) - len(kept),
        "M25_full": M(CB, cohort),
        "M80_full": M(CS, cohort),
        "dM_full": M(CS, cohort) - M(CB, cohort),
        "dM_full_ci_lo": df_lo,
        "dM_full_ci_hi": df_hi,
        "M25_kept": M(CB, kept),
        "M80_kept": M(CS, kept),
        "dM_kept": M(CS, kept) - M(CB, kept),
        "dM_kept_ci_lo": dk_lo,
        "dM_kept_ci_hi": dk_hi,
    }
    return res


def build_figures(n):
    """Assemble figure data (single source for the matplotlib plots)."""
    dense, agg = n["dense"], n["knap2x2"]["agg"]
    doseresponse = {
        "ps": [[c, dense[f"c{c}"]["mean"]] for c in dense["caps"]],
        # matched P->P dense curve on the SAME 12 tasks across all caps (was: 2 anchor points)
        "pp": [[c, dense["PP"][f"c{c}"]["mean"]] for c in dense["caps"]],
    }

    # seed-0 n=25 with task-paired bootstrap CIs (matches the model-selection inference)
    def _bar(cap, cl):
        d = agg[f"cap{cap}"][cl]
        return {"mean": d["mean"], "ci_lo": d["ci_lo"], "ci_hi": d["ci_hi"]}

    select = {
        "n": agg[f"cap{CB}"]["n"],
        "PS": {"bind": _bar(CB, "PS"), "slack": _bar(CS, "PS")},
        "SS": {"bind": _bar(CB, "SS"), "slack": _bar(CS, "SS")},
    }
    turns = [0, 2, 4, 6, 9, 12, 16, 20, 25, 30, 35, 39]

    async def series():
        pat = "experiments/cap_sweep/knapsack/qwen3_8b/main/PS_cap25/**/knapsack-knapsack-*.trace.json"
        eps = [
            await _replay_one(t, False, 25)
            for t in sorted(glob.glob(pat, recursive=True))
        ]
        eps = [e for e in eps if e]
        distinct = [statistics.mean(_at(e["cum_u"], t) for e in eps) for t in turns]
        attempts = [
            statistics.mean(_at(e["cum_u"], t) + _at(e["cum_r"], t) for e in eps)
            for t in turns
        ]
        return {"turns": turns, "distinct": distinct, "attempts": attempts}

    return {
        "doseresponse": doseresponse,
        "select": select,
        "throughput_ps25": asyncio.run(series()),
    }


def main():
    n = {
        "_meta": {
            "generated_by": "scripts/build_paper_numbers.py",
            "note": "single source of truth for all numbers cited in the paper; "
            "rendered into main.tex/appendix.tex by scripts/render_paper.py",
        },
        "config": build_config(),
        "task": build_task(),
        "nav": build_nav(),
        "rule": build_rule(),
        "training": build_training(),
        "prior": build_prior(),
        "knap2x2": build_knap2x2(),
        "interaction": build_interaction(),
        "modelsel": build_modelsel(),
        "timeout_sens": build_timeout_sens(),
        "dense": build_dense(),
        "tmax": build_tmax(),
        "nearthresh": build_nearthresh(),
        "operational": build_operational(),
        "mechanism": build_mechanism(),
        "progress": build_progress(),
        "context": build_context(),
        "replay": build_replay(),
        "exposure": build_exposure(),
        "bandwidth": build_bandwidth(),
        "checkpoint": build_checkpoint(),
        "arms": build_arms(),
    }
    n["figures"] = build_figures(n)
    out = os.path.join("paper", "numbers.json")
    with open(out, "w") as fh:
        json.dump(n, fh, indent=2, sort_keys=False)
        fh.write("\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
