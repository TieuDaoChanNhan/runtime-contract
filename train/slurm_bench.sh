#!/usr/bin/env bash
set -euo pipefail

# Runs vLLM + benchmark in a single SLURM job.
# Starts vLLM in background, waits for health, runs benchmark, then cleans up.

LORA_PATH=""
LORA_NAME=""
CONFIG=""
TASKS_ROOT="train/assets/tasks/easy"
EXTRA_BENCH_ARGS=()

while [[ $# -gt 0 ]]; do
  case $1 in
    --lora)        LORA_PATH="$2";  shift 2 ;;
    --lora-name)   LORA_NAME="$2";  shift 2 ;;
    --config)      CONFIG="$2";     shift 2 ;;
    --tasks-root)  TASKS_ROOT="$2"; shift 2 ;;
    *)             EXTRA_BENCH_ARGS+=("$1"); shift ;;
  esac
done

if [[ -z "$CONFIG" ]]; then
  echo "Error: --config is required"
  exit 1
fi

PORT="${PORT:-8000}"
HEALTH_URL="http://localhost:${PORT}/health"
MAX_WAIT=600  # 10 minutes

# Build vLLM serve command
SERVE_ARGS=(./train/inference.sh --foreground)
if [[ -n "$LORA_PATH" ]]; then
  SERVE_ARGS+=(--lora "$LORA_PATH")
fi
if [[ -n "$LORA_NAME" ]]; then
  export LORA_NAME
fi

# Clean up vLLM on exit
VLLM_PID=""
cleanup() {
  if [[ -n "$VLLM_PID" ]]; then
    echo "Stopping vLLM (pid $VLLM_PID)"
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

# Start vLLM in background
echo "Starting vLLM server..."
NOTHINK=true "${SERVE_ARGS[@]}" &
VLLM_PID=$!

# Wait for health
echo "Waiting for vLLM at $HEALTH_URL (max ${MAX_WAIT}s)..."
elapsed=0
while ! curl -sf "$HEALTH_URL" > /dev/null 2>&1; do
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "vLLM process died"
    exit 1
  fi
  if (( elapsed >= MAX_WAIT )); then
    echo "vLLM failed to start within ${MAX_WAIT}s"
    exit 1
  fi
  sleep 5
  elapsed=$((elapsed + 5))
done
echo "vLLM ready after ${elapsed}s"

# Run benchmark
export OPENAI_API_BASE="http://localhost:${PORT}/v1"
export OPENAI_API_KEY="EMPTY"

BENCH_CMD=(
  python -m codeact_runtime.benchmark.benchmark
  --config "$CONFIG"
  --tasks_root "$TASKS_ROOT"
)
if [[ ${#EXTRA_BENCH_ARGS[@]} -gt 0 ]]; then
  BENCH_CMD+=("${EXTRA_BENCH_ARGS[@]}")
fi

echo "Running benchmark: $CONFIG"
"${BENCH_CMD[@]}"
echo "Benchmark complete"
