#!/usr/bin/env bash
# Per-pod driver for the cap-boundary-carryover test.
#
# Creates its OWN pod serving the knapsack PERSISTENT adapter, runs the given cells in the
# order they are passed, and self-tears-down.
#
# The work is split by TASK, not by cell: with SHARD_SPEC=i/n this shard runs task indices
# i, i+n, ... of EVERY cell, so all three cells of a given task hit the same pod. The
# estimand is paired within task, so that is the split that keeps a pod effect out of the
# comparison while still letting n pods run at once. Cell order matters too -- pass the
# paired contrast (PS, PSckpt) before PP, so a pod lost late still leaves the primary one.
#
# Usage: SHARD_SPEC=0/3 nohup bash experiments/cap_sweep/runners/run_checkpoint_shard.sh CKPT0 PS PSckpt PP & disown
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1
CK=experiments/cap_sweep/knapsack/qwen3_8b/checkpoint
LOGDIR=experiments/cap_sweep/runners/logs; mkdir -p "$LOGDIR"
SHARD="$1"; shift; CELLS=("$@")
SHARD_SPEC="${SHARD_SPEC:-}"
LOG="$LOGDIR/${SHARD}.log"; DONE="$LOGDIR/${SHARD}.done"; : > "$LOG"
log(){ echo "[$(date +%m-%d_%H:%M:%S)] $*" >> "$LOG"; }
log "cells: ${CELLS[*]}  task shard: ${SHARD_SPEC:-all}"
# Released adapter: https://huggingface.co/runtime-contracts/qwen3-8b-knapsack-lora-persistent-seed3407
PERSISTENT_LORA="${PERSISTENT_LORA:?set to your own served knapsack-persistent LoRA repo}"
SHARD_ARGS=()
[ -n "$SHARD_SPEC" ] && SHARD_ARGS=(--shard "$SHARD_SPEC")

GPUS=("NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe" "NVIDIA H100 80GB HBM3" "NVIDIA H100 PCIe" "NVIDIA H100 NVL" "NVIDIA A40" "NVIDIA RTX A6000")
PID=""
for attempt in 1 2 3 4 5 6 7 8; do
  for GPU in "${GPUS[@]}"; do
    CREATE=$(runpodctl create pod --name "ckpt-$SHARD" --gpuType "$GPU" --gpuCount 1 \
      --imageName "vllm/vllm-openai:latest" --containerDiskSize 80 --mem 80 --vcpu 16 \
      --ports "8000/http" --cost 2.5 \
      --args "--model Qwen/Qwen3-8B --enable-lora --max-lora-rank 64 --max-loras 2 --lora-modules persistent=$PERSISTENT_LORA --max-model-len 40960 --gpu-memory-utilization 0.95 --enforce-eager --port 8000" 2>&1)
    PID=$(echo "$CREATE" | sed -n 's/.*pod "\([a-z0-9]*\)".*/\1/p' | head -1)
    [ -n "$PID" ] && { log "created on [$GPU] attempt $attempt: $PID"; break 2; }
  done
  log "all GPU types unavailable (attempt $attempt), retrying in $((attempt*30))s"; sleep $((attempt*30))
done
[ -z "$PID" ] && { log "FAILED to create pod"; echo FAILED > "$DONE"; exit 1; }

# Teardown is registered IMMEDIATELY after creation, and clears PID only after a CONFIRMED
# delete, so a signal during readiness polling or a transient API failure still stops billing.
cleanup(){
  [ -n "${PID:-}" ] || return 0
  for a in 1 2 3; do
    log "cleanup: deleting pod $PID (attempt $a)"
    if runpodctl pod delete "$PID" >> "$LOG" 2>&1; then log "pod $PID deleted"; PID=""; return 0; fi
    sleep 5
  done
  log "WARNING: could not delete pod $PID -- MANUAL CLEANUP REQUIRED"
  return 1
}
trap cleanup EXIT
trap 'cleanup; exit 130' INT
trap 'cleanup; exit 143' TERM

URL="https://${PID}-8000.proxy.runpod.net/v1/models"
READY=0
for i in $(seq 1 150); do
  curl -s -m 10 "$URL" 2>/dev/null | grep -q '"data"' && { READY=1; log "ready after ~$((i*12))s"; break; }
  sleep 12
done
[ "$READY" = "1" ] || { log "pod never became ready"; echo FAILED > "$DONE"; exit 1; }

export OPENAI_API_BASE="https://${PID}-8000.proxy.runpod.net/v1"
export OPENAI_API_KEY="dummy"
export CODEACT_EXTRA_BODY_JSON='{"chat_template_kwargs":{"enable_thinking":false}}'
FAIL=0
for cell in "${CELLS[@]}"; do
  log "RUN $cell"
  if .venv/bin/python -m codeact_runtime.benchmark.benchmark \
      --config "$CK/_${cell}_cap25.yaml" \
      --tasks_root experiments/cap_sweep/knapsack/task_defs --max-examples 25 \
      ${SHARD_ARGS[@]+"${SHARD_ARGS[@]}"} >> "$LOG" 2>&1; then
    log "DONE $cell"
  else
    rc=$?; log "FAILED $cell (exit $rc)"; FAIL=1
  fi
done
cleanup
if [ "$FAIL" = "0" ]; then echo DONE > "$DONE"; log "shard complete";
else echo FAILED > "$DONE"; log "shard FAILED (>=1 cell)"; exit 1; fi
