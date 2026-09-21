# Qwen3-8B LoRA Training & Inference

Fine-tune **Qwen3-8B** with **Axolotl** using `.trace.json` files. Two training configurations: persistent CodeAct and stateless CodeAct. All inference uses `/nothink` (thinking disabled). Training is assistant-only (`roles_to_train: [assistant]`).

## Setup

| Component | Configuration |
|-----------|---------------|
| **Base Model** | `Qwen/Qwen3-8B` |
| **Quantization** | 4-bit QLoRA (bitsandbytes NF4) |
| **Adapter** | LoRA (r=64, alpha=128) |
| **Training Context** | 32768 tokens |
| **Inference Context** | 65536 tokens |
| **Batch Size** | 1 (micro) x 16 (grad accum) = 16 effective |
| **Epochs** | 3 |
| **Optimizer** | AdamW torch |
| **GPU** | NVIDIA GPU with >= 80GB VRAM |
| **Attention** | FLASH_ATTN |
| **Thinking** | Disabled (`/nothink` via `enable_thinking: false`) |

## Training Configs

| Config | Agent | Output |
|--------|-------|--------|
| `axolotl_qwen3_8b_persistent.yaml` | CodeAct (persistent state) | `out/qwen3-8b-persistent` |
| `axolotl_qwen3_8b_stateless.yaml` | CodeAct (stateless) | `out/qwen3-8b-stateless` |

All configs share identical hyperparameters for fair comparison. Only trace data and output paths differ.

## Training

```bash
make train-persistent   # CodeAct persistent state LoRA
make train-stateless    # CodeAct stateless LoRA
```

The training script (`qwen3_8b_axolotl.py`) auto-converts `.trace.json` files to ShareGPT JSONL format. Use `--dry-run` to verify data conversion without training.

## Inference

```bash
make serve              # base model (bf16, 40k ctx, /nothink)
make serve-lora LORA=out/qwen3-8b-persistent   # with LoRA
```

`/nothink` is always enabled via `NOTHINK=true`, which sets `--default-chat-template-kwargs {"enable_thinking": false}`.

## Benchmarks

All benchmarks require vLLM running. Configs in `train/assets/benchmark_configs/`.

Benchmark targets follow the pattern `bench-{runtime}-{model}-{difficulty}`:

```bash
make bench-persistent-base-easy        # persistent runtime, base model, easy tasks
make bench-persistent-persistent-hard  # persistent runtime, persistent LoRA, hard tasks
make bench-persistent-stateless-easy   # cross-eval: stateless LoRA in persistent runtime
```

12 total: `{persistent,stateless}` x `{base,persistent,stateless}` x `{easy,hard}`

## SLURM (anonymized cluster, GH200 nodes)

For SLURM-managed clusters, use the `slurm-*` targets. These install deps into `.venv`, then submit via `sbatch`.

```bash
# Copy and edit .env with your SLURM_PARTITION and SLURM_ACCOUNT
cp .env.example .env

# Training
make slurm-train-persistent
make slurm-train-stateless

# Inference (foreground mode, TP=4)
make slurm-serve
make slurm-serve-persistent-lora
make slurm-serve-stateless-lora

# Benchmarks (starts vLLM + runs benchmark in one job)
make slurm-bench-persistent-base-easy
# ... 12 total: slurm-bench-{runtime}-{model}-{difficulty}

# Check job status
make slurm-status
```

The `train/slurm.sh` wrapper sources `.env`, activates the venv, and runs the given command. Compute nodes have no internet access, so `slurm-train-setup` pre-installs everything.

## VRAM

Training: ~50-65 GiB (QLoRA 4-bit + 32k context). Inference: ~70 GiB (bf16, 65k context, 4x GH200).
