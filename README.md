# Evaluating Agents Across Runtime Contracts: When Mismatch Costs Efficiency or Quality

Code for this paper (IAEval 2026, the NeurIPS 2026 Workshop on Evaluation of Interactive Agents). This repository contains everything the
paper's results are produced from: the task generators for all three families, the CodeAct harness
and its two runtime contracts, the teacher-trace / fine-tuning / serving pipeline, the benchmark
configuration of every cell the paper reports, and the analysis scripts that recompute every
reported number from the released evaluation data.

Environment-specific values (`SLURM_PARTITION`, `SLURM_ACCOUNT`, `WANDB_PROJECT`, adapter paths, API keys)
are placeholders to be filled in locally. The released artifacts (evaluation data, training traces and
LoRA adapters) are on Hugging Face under [`runtime-contracts`](https://huggingface.co/runtime-contracts);
see [Released artifacts](#released-artifacts-hugging-face) below.

---

## What the paper measures

**Runtime transfer**: whether an agent fine-tuned under one execution runtime still solves the task
when deployed under the other. For each task family we generate a *persistent* and a *stateless*
teacher trace for the **same** instances, fine-tune one Qwen3-8B agent on each set, and run **both**
agents on **both** runtimes over the same task IDs. Cell `XY` denotes an `X`-trained adapter on a
`Y` runtime, `X, Y ∈ {P, S}`.

| Runtime contract | Meaning | In this repo |
| :--- | :--- | :--- |
| **Persistent** | interpreter variables carry across turns | `persistent_state: true` |
| **Stateless** | every turn gets a fresh, empty set of variables (tool/environment objects persist; the agent's Python variables do not) | `persistent_state: false` |
| **Cap-boundary carryover** (Sec. 5 intervention) | resets every turn *except* one the cap truncated | `state_carryover_on_cap`, `announce_carryover` |

The flag is validated in `codeact_runtime/benchmark/config.py` and enforced by
`codeact_runtime/codeact/interpreter.py`. On top of the 2×2 we vary one setting evaluations usually
fix — the **per-turn tool-call cap** `c` (`max_tool_calls`) — holding the adapters, the prompts and
the task instances fixed at `T_max = 40` turns.

Reported quantities, all paired over the common task instances of the four cells:

| Quantity | Definition | Reads as |
| :--- | :--- | :--- |
| `G_P(c)` | `Q_PP(c) − Q_PS(c)` | the persistent model's runtime gap |
| `D(c)` | `(Q_PP − Q_PS) − (Q_SP − Q_SS)` | train × runtime interaction |
| `T` | `D(25) − D(80)` | train × runtime × cap moderation |
| `A_P` | `G_P(25) − G_P(80)` | cap amplification (replication arms, cross-family) |
| `M(c)` | `Q_PS(c) − Q_SS(c)` | model selection under a fixed stateless deployment |

`Q` is normalized outcome quality — for Opaque Knapsack, the value of the agent's selection divided
by the optimum.

### The headline

Opaque Knapsack, `n = 25` paired instances, seed 0:

| Cap | `P→P` | `P→S` | Runtime gap `G_P` | `P→S` tokens/ep. | `P→S` cap-hit rate |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `c = 80` (slack) | 0.77 | 0.61 | +0.15 | 162k | 4% |
| `c = 25` (binding) | 0.66 | **0.07** | +0.59 | 498k | 71% |

A slack cap absorbs the runtime mismatch as redundant reconstruction — the agent rebuilds records it
already holds, and pays in tokens and instability while quality stays near matched. A binding cap
converts the same mismatch into lost quality. The interaction moves with it
(`D`: +0.43 → +0.73, so `T = +0.31`), and the collapse replicates across three training seeds, an
independent second rollout, Mistral-7B-v0.3 and Llama-3.1-8B.

Two conditions have to hold for a cap to produce mismatch-specific failure, and the three task
families separate them:

* **Exposure** — the cap must truncate the action the agent actually intended. Measured as
  `E_c = Pr(K_t > c)` by re-executing every recorded code block against the real environment
  (`scripts/mechanism_traces.py`). Knapsack: `E_25 = 0.53`. Navigation: 0.04. Rule diagnosis: 0.00 —
  a *nominally* high tool demand does not imply a binding cap.
* **Continuation advantage** — after a truncation, the matched agent must turn what survived into
  new progress where the mismatched one restarts. Diagnosed by `R_c`, the re-derivation gap.

Knapsack is the only setting that meets both. The bandwidth-reduced navigation arm (`b = 2` nodes
per `neighbors` call) manufactures exposure and collapses *both* runtimes, which is what makes it a
control rather than a replication.

---

## Map: paper → this repository

| Paper element | Where |
| :--- | :--- |
| Runtime contracts, mode banner, few-shot scaffolding | `codeact_runtime/codeact/`, `codeact_runtime/config.py` |
| Per-turn cap enforcement, carryover intervention | `codeact_runtime/benchmark/config.py`, `codeact_runtime/codeact/agent.py` |
| Task generators, constraints and verifiers (all three families) | `codeact_runtime/families/{knapsack,navigation,rule_diagnosis}.py`, `codeact_runtime/generator.py` |
| Teacher-trace generation and trace filtering | `generate_traces.*.yaml`, `train/trace_validator.py`, `train/prepare_paired_data.py` |
| Fine-tuning recipe (QLoRA, `r=64`, `α=128`, 3 epochs, 16k) | `train/configs/axolotl_*.yaml`, `train/qwen3_8b_axolotl.py` |
| Benchmark configuration of every reported cell | `experiments/cap_sweep/**/_*.yaml` |
| Sec. 4 quantities `G_P`, `D`, `T`, `M`, dense `log c` trend | `scripts/analyze_paper.py`, `scripts/build_paper_numbers.py` |
| Sec. 5 mechanism tables (`K_t`, `E_c`, replay vs. novel progress, stage decomposition) | `scripts/mechanism_traces.py`, `scripts/replay_analysis.py` |
| Bandwidth intervention (navigation `b = 2`) | `scripts/make_nav_batch_tasks.py`, `scripts/nav_batch_sweep.py` |
| Cap-boundary carryover intervention | `scripts/checkpoint_analysis.py` |
| Replication arms (2nd rollout, Mistral-7B-v0.3, Llama-3.1-8B) | `scripts/rollout_variance.py` |
| Figures | `scripts/plot_figures.py` |

Every reported statistic is computed from the experiment logs and substituted into the manuscript
rather than transcribed by hand.

---

## Reproducing the reported numbers (no GPU, no new rollouts)

The evaluation data — the full 2×2 across both caps, the three training seeds, the dense cap grid,
the navigation and rule-diagnosis families, the `T_max` horizon control, the cap-boundary-carryover
cells and the replication arms — is not committed here; only the driver configs and orchestration
scripts under `experiments/cap_sweep/` are. It is published on Hugging Face (see below).

### Released artifacts (Hugging Face)

Everything is released under the [`runtime-contracts`](https://huggingface.co/runtime-contracts) organization on
Hugging Face. Scripts under `experiments/cap_sweep/` need `PERSISTENT_LORA` / `STATELESS_LORA` /
`LORA_MODULES` pointed at your own copy of an adapter (for example a local download of the repos below).

**Evaluation data** — every cell the paper reports, grouped by task family, each with its per-task
results and traces, its harness summary and the benchmark config it ran under:

* <https://huggingface.co/datasets/runtime-contracts/evaluation-traces>, revision `378805a69a6a6c0f628bdf9626491a2e48a8b368`

**Training traces** — the teacher agent traces (three task families x persistent/stateless) the
adapters below are fine-tuned on:

* <https://huggingface.co/datasets/runtime-contracts/teacher-traces>, revision `597f91c680a2cb7dc7aa9a6a1380f1b1300683cc`

**Qwen3-8B knapsack LoRA adapters** (three independently trained seeds):

| Seed | Persistent | Stateless |
| :--- | :--- | :--- |
| 3407 | <https://huggingface.co/runtime-contracts/qwen3-8b-knapsack-lora-persistent-seed3407> | <https://huggingface.co/runtime-contracts/qwen3-8b-knapsack-lora-stateless-seed3407> |
| 777 | <https://huggingface.co/runtime-contracts/qwen3-8b-knapsack-lora-persistent-seed777> | <https://huggingface.co/runtime-contracts/qwen3-8b-knapsack-lora-stateless-seed777> |
| 1337 | <https://huggingface.co/runtime-contracts/qwen3-8b-knapsack-lora-persistent-seed1337> | <https://huggingface.co/runtime-contracts/qwen3-8b-knapsack-lora-stateless-seed1337> |

**Replication-arm adapters:**

| Base model | Persistent | Stateless |
| :--- | :--- | :--- |
| Llama-3.1-8B | <https://huggingface.co/runtime-contracts/llama31-8b-knapsack-lora-persistent> | <https://huggingface.co/runtime-contracts/llama31-8b-knapsack-lora-stateless> |
| Mistral-7B-v0.3 | <https://huggingface.co/runtime-contracts/mistral-7b-knapsack-lora-persistent> | <https://huggingface.co/runtime-contracts/mistral-7b-knapsack-lora-stateless> |

**Cross-family adapters** (same Qwen3-8B recipe, trained on each family's own traces):

| Family | Persistent | Stateless |
| :--- | :--- | :--- |
| Rule diagnosis | <https://huggingface.co/runtime-contracts/qwen3-8b-rule_diagnosis-lora-persistent> | <https://huggingface.co/runtime-contracts/qwen3-8b-rule_diagnosis-lora-stateless> |
| Navigation | <https://huggingface.co/runtime-contracts/qwen3-8b-navigation-lora-persistent> | <https://huggingface.co/runtime-contracts/qwen3-8b-navigation-lora-stateless> |

All fourteen adapters the paper trains are released above. None of them is needed to check the
paper's numbers — that runs off the released evaluation data's recorded results — only to re-serve a
family from scratch, which needs `LORA_MODULES` pointed at your own copy.

### Placing the data

`experiments/cap_sweep/` in this repository is laid out exactly like the mirror — the same
`<family>/<base model>/<arm>/` tree — so the download drops each cell's results next to the config
that produced it:

```bash
# Pull the evaluation data at the revision the paper pins, straight into the tree.
# Nothing else to map: the dataset's directory structure is this one.
hf download runtime-contracts/evaluation-traces --repo-type dataset \
  --revision 378805a69a6a6c0f628bdf9626491a2e48a8b368 \
  --local-dir experiments/cap_sweep
```

Pin the revision. The paper's appendix cites these exact SHAs, and an unpinned download will
track the dataset's `main` if it is ever re-uploaded.

`experiments/cap_sweep/README.md` lists every cell, what it establishes in the paper, and which
script reads it.

### Running the analyses

```bash
# The canonical entry point. Recomputes every cited number, including the
# published T interval, into paper/numbers.json, from which the text and tables
# are rendered. No GPU.
mkdir -p paper && uv run python -m scripts.build_paper_numbers

# Per-seed descriptive rows for the Sec. 4 estimands (D, model selection M, dense
# log-c trend) and the Sec. 5 mechanism tables, printed to stdout. This does not
# compute the pooled T interval: that comes from build_interaction() above, which
# runs the crossed, cap-paired bootstrap the paper describes.
uv run python scripts/analyze_paper.py

# Replication arms: 2nd rollout + Mistral-7B-v0.3 + Llama-3.1-8B.
uv run python scripts/rollout_variance.py --arms qwen,mistral,llama31

# Cap-boundary carryover (silent and announced): gap closure, paired bootstrap
# and the behavioural panel fixed before the cells were run (App. J).
uv run python scripts/checkpoint_analysis.py

# Measured cross-family mechanism tables, derived from the recorded traces rather
# than from the nominal replay-demand proxy R. Each episode's recorded code blocks
# are re-executed against the real environment (no LLM) under the same runtime
# semantics and per-turn cap, with every tool wrapped in a recorder; it self-checks
# against the published knapsack replay anchors before printing. Runs as section
# A11 of analyze_paper.py, or standalone (the whole sweep replays in under a minute):
uv run python scripts/mechanism_traces.py                      # all three families
uv run python scripts/mechanism_traces.py --families navigation --json nav.json
```

### Mirror coverage

Every published family and arm carries its full per-task results and traces, so their bootstrap
intervals reproduce, not only their point estimates. The one thing excluded from every cell is the
SQLite benchmark cache; nothing else is held back.

---

## Rebuilding the study from scratch

### Installation

```bash
uv sync
uv pip install -e .
uv sync --extra train      # only if reproducing the LoRA fine-tuning
```

Teacher-trace generation calls an external LLM; put its key in `.env`:

```bash
GEMINI_API_KEY=...
```

Copy `.env.example` to `.env` and fill in the cluster placeholders before using any `slurm-*`
target.

### 1. Generate the configs

Benchmark and teacher-trace configs are rendered from a shared matrix
(`train/assets/benchmark_configs/_matrix.yaml`) through Jinja2 templates, which keeps `max_turns`,
`timeout_s` and `max_tool_calls` in sync across every config. Run this **first** — the knapsack
teacher-trace configs (`generate_traces.{persistent,stateless}.yaml`) are generated, not committed:

```bash
make gen-benchmark-configs
```

### 2. Generate task instances

All three families are procedurally generated with instance constraints that keep tasks from being
trivially solvable, plus an exact verifier.

```bash
make tasks
make split
```

Each task is one JSON object with `task_id`, `family`, `seed`, `tools` (the declared tool contract
for the episode), `public` (visible instance data), `private` (hidden environment state, for replay
and verification), `reference` (ground truth plus metadata) and an optional `nl` natural-language
wrapper.

### 3. Generate paired teacher traces

Both members of a pair are generated on the same instances; an instance enters the training set only
if *both* traces pass identical validity, quality and context-length filters. The six driver configs,
one per family x runtime, live under `configs/generate_traces/`:

```bash
make traces-persistent          # knapsack, persistent runtime
make traces-stateless           # knapsack, stateless runtime
make traces-persistent-nav      # navigation, persistent runtime
make traces-stateless-nav       # navigation, stateless runtime
make traces-persistent-rule     # rule diagnosis, persistent runtime
make traces-stateless-rule      # rule diagnosis, stateless runtime
```

Pairing equalizes *which* instances are used, not trace length: stateless traces rebuild state and
run about 2.3× longer, which is why cross-adapter comparisons are read as bundled.

### 4. Trace statistics

`make stats` prints a side-by-side comparison of two trace output directories:

| Metric | Description |
| :--- | :--- |
| Teacher success rate (%) | share of episodes the agent solved |
| Avg teacher optimality (0–1) | mean normalized closeness to the optimum |
| Avg capacity utilization (%) | share of knapsack capacity filled |
| Avg items inspected | distinct items queried |
| Avg steps / tool calls / tokens per episode | execution cost |

### 5. Fine-tuning

All training uses 4-bit QLoRA at a 16k-token context window on 4× NVIDIA GH200 GPUs (1 node,
anonymized cluster). The training scripts auto-convert the `.trace.json` files into ShareGPT-format
JSONL.

```bash
make train-persistent
make train-stateless
```

Cross-model arms use the same recipe via `train/configs/axolotl_{mistral_7b,llama31_8b}_*.yaml` and
`train/slurm_{mistral,llama31}.sh`, run directly (no `make` target).

### 6. Serving

Evaluation needs a running vLLM server. Inference serves in native bf16 at a 40k context window for
Qwen3-8B and Llama-3.1-8B; Mistral-7B-v0.3 serves at 32k, its native `max_position_embeddings`
ceiling.

```bash
make serve                     # base model
make serve-persistent-lora     # a specific LoRA adapter
```

### 7. Running cells

```bash
make bench-persistent-base-easy            # base model, persistent runtime, easy tasks
make bench-persistent-persistent-medium    # persistent LoRA, persistent runtime, medium tasks
make bench-persistent-stateless-easy       # cross-eval: stateless LoRA, persistent runtime
```

The cap-sweep cells reported in the paper are driven by the configs and shard scripts under
`experiments/cap_sweep/` rather than by these targets. To compare two runs directly:

```bash
uv run python scripts/compare_benchmarks.py \
    results/med_persistent_lora.tar.gz results/med_reset_lora.tar.gz
```

---

## Repository layout

```
codeact_runtime/     agent harness: CodeAct loop, interpreter (both runtimes), task families,
                     generators, benchmark driver
train/               trace filtering, ShareGPT conversion, Axolotl configs, SLURM + serving scripts
experiments/         benchmark configuration of every reported cell, grouped by task family,
                     plus the pod runners; see experiments/cap_sweep/README.md
scripts/             analysis: paper numbers, mechanism replay, interventions, figures
tests/               unit tests for the task families, verifiers and runtimes
```

## License

Apache License 2.0 (see [`LICENSE`](LICENSE)). The two Llama-3.1-8B adapters are the exception: they are
derivatives of Llama 3.1 and follow the [Llama 3.1 Community License](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/LICENSE).
