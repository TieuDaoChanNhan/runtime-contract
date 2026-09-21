#!/usr/bin/env bash
# Run cells on an ALREADY-CREATED pod, then self-teardown that pod.
# Usage: run_on_pod.sh <POD_ID> <SHARD_NAME> <CELL1> [CELL2 ...]
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1
# Cell -> config path. Cells are grouped by task family under experiments/cap_sweep/.
cell_cfg() {
  case "$1" in
    navb[0-9]*_*) echo "experiments/cap_sweep/navigation/qwen3_8b/batch2/_$1.yaml" ;;
    nav_*)        echo "experiments/cap_sweep/navigation/qwen3_8b/main/_$1.yaml" ;;
    rule_*)       echo "experiments/cap_sweep/rule_diagnosis/qwen3_8b/main/_$1.yaml" ;;
    *) echo "unknown cell family: $1" >&2; return 1 ;;
  esac
}
LOGDIR=experiments/cap_sweep/runners/logs; mkdir -p "$LOGDIR"
POD_ID="$1"; SHARD="$2"; shift 2; CELLS=("$@")
LOG="$LOGDIR/${SHARD}.log"; DONE="$LOGDIR/${SHARD}.done"
log(){ echo "[$(date +%m-%d_%H:%M:%S)] $*" >> "$LOG"; }
log "run_on_pod $POD_ID shard=$SHARD cells=${CELLS[*]}"

URL="https://${POD_ID}-8000.proxy.runpod.net/v1/models"
READY=0
for i in $(seq 1 150); do
  curl -s -m 10 "$URL" 2>/dev/null | grep -q '"data"' && { READY=1; log "ready after ~$((i*12))s"; break; }
  sleep 12
done
if [ "$READY" != "1" ]; then log "pod never ready; deleting"; runpodctl pod delete "$POD_ID" >> "$LOG" 2>&1; echo FAILED > "$DONE"; exit 1; fi

export OPENAI_API_BASE="https://${POD_ID}-8000.proxy.runpod.net/v1"
export OPENAI_API_KEY="dummy"
export CODEACT_EXTRA_BODY_JSON='{"chat_template_kwargs":{"enable_thinking":false}}'
for cell in "${CELLS[@]}"; do
  case "$cell" in rule_*) TROOT="experiments/cap_sweep/rule_diagnosis/task_defs";; navb[0-9]*_*) B="${cell#navb}"; B="${B%%_*}"; TROOT="experiments/cap_sweep/navigation/task_defs_batch$B";; nav_*) TROOT="experiments/cap_sweep/navigation/task_defs";; *) continue;; esac
  log "RUN $cell"
  .venv/bin/python -m codeact_runtime.benchmark.benchmark --config "$(cell_cfg "$cell")" --tasks_root "$TROOT" --max-examples 16 >> "$LOG" 2>&1
  log "DONE $cell"
done
echo DONE > "$DONE"
log "tearing down $POD_ID"
runpodctl pod delete "$POD_ID" >> "$LOG" 2>&1
log "shard complete"
