#!/usr/bin/env bash
set -euo pipefail

# Training script
# Uses QLoRA (4-bit base) with 16k token context (see MAX_SEQ_LENGTH default below)

# Use plain python under SLURM (venv activated by slurm.sh), uv run otherwise
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  RUN="python"
  RUN_TRAIN="python"
else
  RUN="uv run python"
  RUN_TRAIN="uv run python"
fi

MODEL="${MODEL:-Qwen/Qwen3-8B}"
TRACE_ROOT="${TRACE_ROOT:-traces}"
OUTPUT_DIR="${OUTPUT_DIR:-out/qwen3-8b}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-16384}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
EPOCHS="${EPOCHS:-3}"
TRACE_FORMAT="${TRACE_FORMAT:-codeact}"
SESSION_NAME="${SESSION_NAME:-train}"
BASE_CONFIG="${BASE_CONFIG:-train/configs/axolotl_qwen3_8b_persistent.yaml}"
WANDB_PROJECT="${WANDB_PROJECT:-}"
DATA_FILE="${DATA_FILE:-}"

# Parse flags
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case $1 in
    --trace-root)
      TRACE_ROOT="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --max-seq-length)
      MAX_SEQ_LENGTH="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --grad-accum)
      GRAD_ACCUM="$2"
      shift 2
      ;;
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --session)
      SESSION_NAME="$2"
      shift 2
      ;;
    --model)
      MODEL="$2"
      shift 2
      ;;
    --trace-format)
      TRACE_FORMAT="$2"
      shift 2
      ;;
    --base-config)
      BASE_CONFIG="$2"
      shift 2
      ;;
    --wandb-project)
      WANDB_PROJECT="$2"
      shift 2
      ;;
    --data-file)
      DATA_FILE="$2"
      shift 2
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

# Effective batch size = BATCH_SIZE * GRAD_ACCUM = 8 by default
EFFECTIVE_BATCH=$((BATCH_SIZE * GRAD_ACCUM))

echo "========================================"
echo "QLoRA Training"
echo "========================================"
echo
echo "Config:"
echo "  MODEL:          $MODEL"
echo "  TRACE_ROOT:     $TRACE_ROOT"
echo "  OUTPUT_DIR:     $OUTPUT_DIR"
echo "  MAX_SEQ_LENGTH: $MAX_SEQ_LENGTH"
echo "  BATCH_SIZE:     $BATCH_SIZE"
echo "  GRAD_ACCUM:     $GRAD_ACCUM"
echo "  EFFECTIVE_BS:   $EFFECTIVE_BATCH"
echo "  EPOCHS:         $EPOCHS"
echo "  TRACE_FORMAT:   $TRACE_FORMAT"
echo
echo "Estimated VRAM: ~50-65GB (QLoRA 4-bit + 16k context)"
echo

# Determine data file path
if [[ -n "$DATA_FILE" ]]; then
  RESOLVED_DATA_FILE="$DATA_FILE"
  echo "Using pre-prepared data: $RESOLVED_DATA_FILE"
  echo ""
else
  RESOLVED_DATA_FILE="$OUTPUT_DIR/data/traces.jsonl"

  # 1. PREPARE DATA
  echo "Step 1: Preparing Data (Truncation & Tokenization)"
  $RUN train/prepare_data.py \
      --trace-root "$TRACE_ROOT" \
      --output-dir "$OUTPUT_DIR/data" \
      --model "$MODEL" \
      --max-seq-length "$MAX_SEQ_LENGTH" \
      --trace-format "$TRACE_FORMAT"
fi

# TRAIN
echo "Step 2: Training"
CMD=(
  $RUN_TRAIN train/qwen3_8b_axolotl.py
  --model "$MODEL"
  --data-file "$RESOLVED_DATA_FILE"
  --output-dir "$OUTPUT_DIR"
  --max-seq-length "$MAX_SEQ_LENGTH"
  --batch-size "$BATCH_SIZE"
  --grad-accum "$GRAD_ACCUM"
  --epochs "$EPOCHS"
  --base-config "$BASE_CONFIG"
)

if [[ -n "$WANDB_PROJECT" ]]; then
  CMD+=(--wandb-project "$WANDB_PROJECT")
fi

# If running under SLURM, execute directly (no tmux on compute nodes)
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  echo "Running under SLURM job ${SLURM_JOB_ID}"
  printf '  %q' "${CMD[@]}"
  echo
  echo
  "${CMD[@]}"
  exit $?
fi


# Start tmux server if not running
tmux start-server 2>/dev/null || true

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "tmux session '${SESSION_NAME}' already exists. Attach with:"
  echo "  tmux attach -t ${SESSION_NAME}"
  exit 1
fi

echo "Launching training in tmux session: ${SESSION_NAME}"
printf '  %q' "${CMD[@]}"
echo
echo

# Write command to temp script to preserve quoting
TMPSCRIPT=$(mktemp /tmp/train-cmd.XXXXXX.sh)
echo '#!/bin/bash' > "$TMPSCRIPT"
printf '%q ' "${CMD[@]}" >> "$TMPSCRIPT"
chmod +x "$TMPSCRIPT"
tmux new-session -d -s "${SESSION_NAME}" -- bash -lc "$TMPSCRIPT; rm $TMPSCRIPT"
tmux set-option -t "${SESSION_NAME}" remain-on-exit on
echo "Training started. Attach with:"
echo "  tmux attach -t ${SESSION_NAME}"
