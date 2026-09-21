#!/usr/bin/env bash
set -euo pipefail

# Defaults
TASKS_ROOT="${TASKS_ROOT:-train/assets/tasks/easy}"
SESSION_NAME="${SESSION_NAME:-bench-qwen3}"
CONFIG=""

# Parse flags
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case $1 in
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --tasks-root)
      TASKS_ROOT="$2"
      shift 2
      ;;
    --session)
      SESSION_NAME="$2"
      shift 2
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -z "$CONFIG" ]]; then
  echo "Error: --config is required. Use a config from train/assets/benchmark_configs/"
  exit 1
fi
echo "Running benchmark with config: $CONFIG"

CMD=(
  uv run python -m codeact_runtime.benchmark.benchmark
  --config "$CONFIG"
  --tasks_root "$TASKS_ROOT"
)

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  CMD+=("${EXTRA_ARGS[@]}")
fi

# Set required environment variables for local vLLM
export OPENAI_API_BASE="${OPENAI_API_BASE:-http://localhost:8000/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"

echo "Launching benchmark in tmux session: ${SESSION_NAME}"
printf '  %q' "${CMD[@]}"
echo
echo
echo "Config:"
echo "  CONFIG:         $CONFIG"
echo "  TASKS_ROOT:     $TASKS_ROOT"
echo "  OPENAI_API_BASE: $OPENAI_API_BASE"
echo

# Start tmux server if not running
tmux start-server 2>/dev/null || true

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "tmux session '${SESSION_NAME}' already exists. Attach with:"
  echo "  tmux attach -t ${SESSION_NAME}"
  exit 1
fi

# Write command to temp script to preserve quoting
TMPSCRIPT=$(mktemp /tmp/bench-cmd.XXXXXX.sh)
echo '#!/bin/bash' > "$TMPSCRIPT"
printf '%q ' "${CMD[@]}" >> "$TMPSCRIPT"
chmod +x "$TMPSCRIPT"
tmux new-session -d -s "${SESSION_NAME}" -e "OPENAI_API_BASE=$OPENAI_API_BASE" -e "OPENAI_API_KEY=$OPENAI_API_KEY" -- bash -lc "$TMPSCRIPT; rm $TMPSCRIPT"
tmux set-option -t "${SESSION_NAME}" remain-on-exit on
echo "Benchmark started. Attach with:"
echo "  tmux attach -t ${SESSION_NAME}"
