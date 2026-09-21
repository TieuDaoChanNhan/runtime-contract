#!/usr/bin/env bash
set -euo pipefail

# Single-cell variant, mirrors serve_and_run_rollout2_cell.sh: starts vLLM
# (both Mistral-7B-v0.3 knapsack LoRAs loaded together) and runs exactly ONE
# knapsack/mistral_7b/main cell, then tears the server down. Runs via train/slurm.sh
# (repo .venv - vllm already supports MistralForCausalLM, no isolated serving
# venv needed).
#
# Needs an explicit --chat-template: the base checkpoint ships no
# chat_template, and axolotl's mistral_v2v3 template has no system-role
# support, which codeact_runtime's CodeAct agent needs - see
# mistral_serve_chat_template.jinja.
#
# --max-model-len 32768: Mistral-7B-v0.3's max_position_embeddings is 32768
# (shorter than Qwen3-8B/Gemma-4-E4B).

CELL="${1:?usage: serve_and_run_mistral_cell.sh CELL_NAME (e.g. PP_cap25)}"

# Released adapters:
# persistent: https://huggingface.co/runtime-contracts/mistral-7b-knapsack-lora-persistent
# stateless:  https://huggingface.co/runtime-contracts/mistral-7b-knapsack-lora-stateless
PERSISTENT_LORA="${PERSISTENT_LORA:?set PERSISTENT_LORA to the Mistral persistent adapter path}"
STATELESS_LORA="${STATELESS_LORA:?set STATELESS_LORA to the Mistral stateless adapter path}"

PORT="${PORT:-8000}"
HEALTH_URL="http://localhost:${PORT}/health"
MAX_WAIT=600  # 10 minutes

VLLM_PID=""
cleanup() {
  if [[ -n "$VLLM_PID" ]]; then
    echo "Stopping vLLM (pid $VLLM_PID)"
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "[$CELL] Starting vLLM server with both Mistral-7B-v0.3 knapsack LoRAs (persistent + stateless)..."
python -m vllm.entrypoints.openai.api_server \
  --model mistralai/Mistral-7B-v0.3 \
  --host 0.0.0.0 --port "$PORT" \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.95 \
  --max-model-len 32768 \
  --max-num-seqs 16 \
  --tensor-parallel-size 4 \
  --served-model-name mistral-7b \
  --trust-remote-code \
  --enable-lora \
  --max-lora-rank 64 \
  --max-loras 2 \
  --lora-modules "persistent=${PERSISTENT_LORA}" "stateless=${STATELESS_LORA}" \
  --chat-template experiments/cap_sweep/runners/mistral_serve_chat_template.jinja \
  --enforce-eager &
VLLM_PID=$!

echo "[$CELL] Waiting for vLLM at $HEALTH_URL (max ${MAX_WAIT}s)..."
elapsed=0
while ! curl -sf "$HEALTH_URL" > /dev/null 2>&1; do
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "[$CELL] vLLM process died"
    exit 1
  fi
  if (( elapsed >= MAX_WAIT )); then
    echo "[$CELL] vLLM failed to start within ${MAX_WAIT}s"
    exit 1
  fi
  sleep 5
  elapsed=$((elapsed + 5))
done
echo "[$CELL] vLLM ready after ${elapsed}s"

MODELS_JSON=$(curl -sf "http://localhost:${PORT}/v1/models")
echo "[$CELL] Models served: $MODELS_JSON"
if ! echo "$MODELS_JSON" | grep -q '"persistent"' || ! echo "$MODELS_JSON" | grep -q '"stateless"'; then
  echo "[$CELL] ERROR: expected both 'persistent' and 'stateless' in /v1/models, got: $MODELS_JSON"
  exit 1
fi

export OPENAI_API_BASE="http://localhost:${PORT}/v1"
export OPENAI_API_KEY="EMPTY"

echo "=== Running $CELL ==="
if python -m codeact_runtime.benchmark.benchmark \
    --config "experiments/cap_sweep/knapsack/mistral_7b/main/_${CELL}.yaml" \
    --tasks_root experiments/cap_sweep/knapsack/task_defs \
    --max-examples 25; then
  echo "=== DONE $CELL ==="
else
  rc=$?
  echo "=== FAILED $CELL (exit $rc) ==="
  exit 1
fi
