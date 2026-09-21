#!/usr/bin/env bash
set -euo pipefail

# vLLM inference server
# No quantization (bf16 native), 65k context

# Parse flags
LORA_PATH=""
FOREGROUND=false
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case $1 in
    --lora)
      LORA_PATH="$2"
      shift 2
      ;;
    --foreground)
      FOREGROUND=true
      shift
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

MODEL="${MODEL:-Qwen/Qwen3-8B}"

# Default settings: no quantization, bf16, 65k context
ATTENTION_BACKEND="${ATTENTION_BACKEND:-FLASH_ATTN}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
DTYPE="${DTYPE:-bfloat16}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
# SLURM/foreground: 4 GPUs, higher concurrency
if [[ "$FOREGROUND" == "true" ]]; then
  MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
  TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-4}"
else
  MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
  TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
fi
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-true}"
SESSION_NAME="${SESSION_NAME:-vllm}"
NOTHINK="${NOTHINK:-false}"

# Set served model name and lora name based on --lora flag
if [[ -n "$LORA_PATH" ]]; then
  LORA_NAME="${LORA_NAME:-qwen3-8b-lora}"
  SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$LORA_NAME}"
else
  SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-8b}"
fi

# Build command (foreground uses venv directly; local uses uv run)
if [[ "$FOREGROUND" == "true" ]]; then
  UV_CMD=(python -m vllm.entrypoints.openai.api_server)
else
  UV_CMD=(uv run --with vllm python -m vllm.entrypoints.openai.api_server)
fi

CMD=(
  "${UV_CMD[@]}"
  --model "$MODEL"
  --host "$HOST"
  --port "$PORT"
  --dtype "$DTYPE"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-seqs "$MAX_NUM_SEQS"
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
  --served-model-name "$SERVED_MODEL_NAME"
  --attention-backend "$ATTENTION_BACKEND"
)

if [[ "$TRUST_REMOTE_CODE" == "true" ]]; then
  CMD+=(--trust-remote-code)
fi

if [[ -n "$LORA_PATH" ]]; then
  CMD+=(--enable-lora --lora-modules "${LORA_NAME}=${LORA_PATH}" --max-lora-rank 64)
fi

if [[ "$NOTHINK" == "true" ]]; then
  CHAT_KWARGS='{"enable_thinking":false}'
  CMD+=(--default-chat-template-kwargs "$CHAT_KWARGS")
fi

# Disable torch.compile on platforms without triton (e.g. aarch64)
if [[ "$FOREGROUND" == "true" ]]; then
  CMD+=(--enforce-eager)
fi

CMD+=("${EXTRA_ARGS[@]}")

echo "========================================"
echo "vLLM Inference Server"
echo "========================================"
echo
echo "Config:"
echo "  MODEL:          $MODEL"
echo "  DTYPE:          $DTYPE (no quantization)"
echo "  MAX_MODEL_LEN:  $MAX_MODEL_LEN"
echo "  MAX_NUM_SEQS:   $MAX_NUM_SEQS"
echo "  GPU_MEM_UTIL:   $GPU_MEMORY_UTILIZATION"
echo "  SERVED_NAME:    $SERVED_MODEL_NAME"
if [[ -n "$LORA_PATH" ]]; then
  echo "  LORA_PATH:      $LORA_PATH"
fi
if [[ "$NOTHINK" == "true" ]]; then
  echo "  NOTHINK:        enabled (thinking disabled)"
fi
echo
if [[ "$FOREGROUND" == "true" ]]; then
  echo "Launching vLLM server in foreground"
  printf '  %q' "${CMD[@]}"
  echo
  echo
  exec "${CMD[@]}"
fi

echo "Launching vLLM server in tmux session: ${SESSION_NAME}"
printf '  %q' "${CMD[@]}"
echo
echo

# Start tmux server if not running
tmux start-server 2>/dev/null || true

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "tmux session '${SESSION_NAME}' already exists. Attach with:"
  echo "  tmux attach -t ${SESSION_NAME}"
  exit 1
fi

# Write command to temp script to preserve quoting
TMPSCRIPT=$(mktemp /tmp/vllm-cmd.XXXXXX.sh)
echo '#!/bin/bash' > "$TMPSCRIPT"
printf '%q ' "${CMD[@]}" >> "$TMPSCRIPT"
chmod +x "$TMPSCRIPT"
tmux new-session -d -s "${SESSION_NAME}" -- bash -lc "$TMPSCRIPT; rm $TMPSCRIPT"
tmux set-option -t "${SESSION_NAME}" remain-on-exit on
echo "Server started. Attach with:"
echo "  tmux attach -t ${SESSION_NAME}"
