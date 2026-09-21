# Evaluation cells

Every cell the paper reports, grouped by task family. This directory holds the **benchmark
configuration** of each cell — the authoritative record of what was run — not its outputs.

## Layout convention

```
<family>/<base model>/<arm>/
    _<cell>.yaml      driver config, committed        <- what was run
    <cell>/           results, NOT committed          <- appears after a run or a mirror download
<family>/task_defs*/  the evaluated task instances
runners/              pod orchestration (only needed to re-run cells)
```

`_<cell>.yaml` always writes to `<cell>/` beside it, so a config and its results sit together.
This layout is **identical to the published evaluation-data mirror**, so unpacking the mirror into
this directory drops every cell's results next to the config that produced it — no path mapping
(see the repository README for the link).

## What each directory is for

| Directory | Cells | What it establishes | Read by |
| :--- | :--- | :--- | :--- |
| `knapsack/qwen3_8b/main` | `{PP,PS,SP,SS}_cap{25,80}` | the primary 2×2 at a binding and a slack cap, `n=25` — the headline table, `G_P`, `D`, `T`, `M` | `analyze_paper.py`, `build_paper_numbers.py`, `mechanism_traces.py`, `replay_analysis.py`, `rollout_variance.py` |
| `knapsack/qwen3_8b/seeds` | `s{777,1337}_{PP,PS,SP,SS}_cap{25,80}` | two further training seeds, `n=20` — per-seed `T` and the crossed seed×task bootstrap | `analyze_paper.py`, `build_paper_numbers.py` |
| `knapsack/qwen3_8b/dense` | `persistent_cap{10,20,25,30,40,50,80,100}` (P→S), `matched_cap{…}` (P→P) | the dense cap grid, `n=12` — the dose–response figure and the `log c` slope contrast | `analyze_paper.py`, `build_paper_numbers.py`, `replay_analysis.py` |
| `knapsack/qwen3_8b/tmax` | `{PP,PS}_cap25_T{80,160}`, `PS_cap40_T{40,80}` | the episode-horizon control: more turns do not rescue the mismatched cell | `build_paper_numbers.py`, `replay_analysis.py` |
| `knapsack/qwen3_8b/checkpoint` | `PS`, `PSckpt`, `PP` (silent pair); `PSckpt_rerun`, `PSckptann` (announced pair) | the cap-boundary-carryover intervention — gap closure `ρ`, and the announced variant | `checkpoint_analysis.py` |
| `knapsack/qwen3_8b/rollout2` | `{PP,PS,SP,SS}_cap{25,80}` | an independent second rollout of the same adapters — replication arm 2 | `rollout_variance.py` |
| `knapsack/mistral_7b/main` | same 8 cells | the same recipe on Mistral-7B-v0.3 — replication arm 3 | `rollout_variance.py` |
| `knapsack/llama31_8b/main` | same 8 cells | the same recipe on Llama-3.1-8B — replication arm 4 | `rollout_variance.py` |
| `navigation/qwen3_8b/main` | `nav_{PP,PS,SP,SS}_cap{25,80}`, `nav_PS_cap25_rerun` | the navigation 2×2 (a cross-family control: low cap exposure, no outcome response) plus a same-cell drift re-run | `analyze_paper.py`, `build_paper_numbers.py`, `mechanism_traces.py` |
| `navigation/qwen3_8b/batch2` | `navb2_{PP,PS}_cap{25,80}` | the bandwidth-reduced arm (`b=2` nodes per `neighbors` call): manufacturing exposure collapses **both** runtimes, so exposure alone is not the mechanism | `analyze_paper.py`, `build_paper_numbers.py`, `mechanism_traces.py` |
| `rule_diagnosis/qwen3_8b/main` | `rule_{PP,PS,SP,SS}_cap{25,80}` | the rule-diagnosis 2×2 (the second cross-family control: the cap is never reached, `E_25 = 0.00`) | `analyze_paper.py`, `build_paper_numbers.py`, `mechanism_traces.py` |

Cell codes: `XY` is an `X`-trained adapter on a `Y` runtime, `X, Y ∈ {P, S}`; `cap25` / `cap80` is
the per-turn tool-call cap; `T80` / `T160` is `T_max`. In `dense/`, `persistent` is P→S and
`matched` is P→P.

## Task instances

| Directory | Used by |
| :--- | :--- |
| `knapsack/task_defs` | every knapsack cell |
| `navigation/task_defs` | `navigation/qwen3_8b/main` |
| `navigation/task_defs_batch2` | `navigation/qwen3_8b/batch2` (rebuilt from the same 16 instances with the batch limit cut to 2) |
| `rule_diagnosis/task_defs` | `rule_diagnosis/qwen3_8b/main` |

Committed here are `_cfg.json` (the generator settings) and `manifest.json`. The instances
themselves (`tasks/<family>/*.json`) come with the evaluation-data mirror, or can be regenerated
from `_cfg.json`.

## Cells that no analysis reads

Present because they were run, but not behind any reported number:

* `knapsack/qwen3_8b/dense/_stateless_cap{10,25,50,100}.yaml` — the S→S dense arm.
* `knapsack/qwen3_8b/tmax/_PP_cap40_T{40,80,160}.yaml`, `_PS_cap40_T160.yaml` — the cap-40 horizon
  cells beyond the two the near-threshold check uses.

## Runners

`runners/` holds the per-shard drivers that rent a GPU pod, serve the adapters, run their assigned
cells and tear the pod down again, plus the watchdogs, the pod provisioning helpers
(`rent_gpu.sh`, `provision.sh`) and the two serving chat templates. **None of
it is needed to reproduce a reported number** — the analyses read recorded results only. Each runner
resolves a cell name to its config, so it is called with cell names, not paths:

```bash
LORA_MODULES="np=<your-persistent-nav-lora> ns=<your-stateless-nav-lora>" \
  bash experiments/cap_sweep/runners/run_shard.sh NAV0 nav_PS_cap25 nav_PP_cap25
```

Logs and shard-completion markers go to `runners/logs/` (gitignored).
