#!/usr/bin/env bash
# Per-pod driver for the MATCHED (P->P) dense cap sweep: self-creates a pod serving the knapsack
# PERSISTENT adapter, runs the matched_cap* cells (persistent runtime) at n=12, self-tears-down.
# Usage: run_matched_shard.sh [CELL1 CELL2 ...]   (default: all 8 matched_cap cells)
#   CELLs are dense config basenames, e.g. matched_cap25
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1
SF=experiments/cap_sweep/knapsack/qwen3_8b/dense
LOGDIR=experiments/cap_sweep/runners/logs; mkdir -p "$LOGDIR"
SHARD=matched
LOG="$LOGDIR/run_${SHARD}_shard.log"; DONE="$LOGDIR/run_${SHARD}_shard.done"; : > "$LOG"
log(){ echo "[$(date +%m-%d_%H:%M:%S)] $*" >> "$LOG"; }

# RunPod auth from .env (RUNPOD_API_KEY=...) -- .env is gitignored.
set -a; [ -f .env ] && . ./.env; set +a
if [ -n "${RUNPOD_API_KEY:-}" ]; then runpodctl config --apiKey "$RUNPOD_API_KEY" >/dev/null 2>&1; fi

CELLS=("$@")
if [ "${#CELLS[@]}" -eq 0 ]; then
  CELLS=(matched_cap10 matched_cap20 matched_cap25 matched_cap30 matched_cap40 matched_cap50 matched_cap80 matched_cap100)
fi
log "cells: ${CELLS[*]}"
# Released adapter: https://huggingface.co/runtime-contracts/qwen3-8b-knapsack-lora-persistent-seed3407
PERSISTENT_LORA="${PERSISTENT_LORA:?set to your own served knapsack-persistent LoRA repo}"

GPUS=("NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe" "NVIDIA H100 80GB HBM3" "NVIDIA H100 PCIe" "NVIDIA H100 NVL" "NVIDIA A40")
PID=""
for attempt in 1 2 3 4 5 6 7 8; do
  for GPU in "${GPUS[@]}"; do
    CREATE=$(runpodctl create pod --name "matched-dense" --gpuType "$GPU" --gpuCount 1 \
      --imageName "vllm/vllm-openai:latest" --containerDiskSize 80 --mem 80 --vcpu 16 \
      --ports "8000/http" --cost 2.5 \
      --args "--model Qwen/Qwen3-8B --enable-lora --max-lora-rank 64 --max-loras 2 --lora-modules persistent=$PERSISTENT_LORA --max-model-len 40960 --gpu-memory-utilization 0.95 --enforce-eager --port 8000" 2>&1)
    PID=$(echo "$CREATE" | sed -n 's/.*pod "\([a-z0-9]*\)".*/\1/p' | head -1)
    [ -n "$PID" ] && { log "created on [$GPU]: $PID"; break 2; }
  done
  log "all GPUs unavailable (attempt $attempt), retry ${attempt}0s"; sleep $((attempt*10))
done
[ -z "$PID" ] && { log "FAILED create"; echo FAILED > "$DONE"; exit 1; }

# Register teardown IMMEDIATELY after creation so a SIGINT/SIGTERM or unexpected exit during
# readiness polling or any experiment cell still deletes the paid pod. cleanup retries and clears
# PID ONLY after a confirmed delete, so a transient delete failure is retried on the next exit
# rather than silently leaking the pod. Signal handlers clean up and then EXIT (they must not let
# the script resume against a deleted endpoint); the EXIT trap covers ordinary/error exits.
cleanup(){
  [ -n "${PID:-}" ] || return 0
  for a in 1 2 3; do
    log "cleanup: deleting pod $PID (attempt $a)"
    if runpodctl pod delete "$PID" >> "$LOG" 2>&1; then log "pod $PID deleted"; PID=""; return 0; fi
    sleep 5
  done
  log "WARNING: could not delete pod $PID after 3 attempts -- MANUAL CLEANUP REQUIRED"
  return 1
}
trap cleanup EXIT
trap 'cleanup; exit 130' INT
trap 'cleanup; exit 143' TERM

URL="https://${PID}-8000.proxy.runpod.net/v1/models"
READY=0
for i in $(seq 1 150); do curl -s -m 10 "$URL" 2>/dev/null | grep -q '"data"' && { READY=1; log "ready ~$((i*12))s"; break; }; sleep 12; done
[ "$READY" = "1" ] || { log "pod never became ready"; echo FAILED > "$DONE"; exit 1; }  # trap deletes pod on exit

export OPENAI_API_BASE="https://${PID}-8000.proxy.runpod.net/v1"
export OPENAI_API_KEY="dummy"
export CODEACT_EXTRA_BODY_JSON='{"chat_template_kwargs":{"enable_thinking":false}}'
FAIL=0
for cell in "${CELLS[@]}"; do
  log "RUN $cell"
  if .venv/bin/python -m codeact_runtime.benchmark.benchmark --config "$SF/_${cell}.yaml" --tasks_root experiments/cap_sweep/knapsack/task_defs --max-examples 12 >> "$LOG" 2>&1; then
    log "DONE $cell"
  else
    rc=$?; log "FAILED $cell (exit $rc)"; FAIL=1
  fi
done
# the EXIT trap tears the pod down (stop billing) regardless of how we leave here
if [ "$FAIL" = "0" ]; then echo DONE > "$DONE"; log "shard complete (pod $PID torn down on exit)";
else echo FAILED > "$DONE"; log "shard FAILED (>=1 cell; pod $PID torn down on exit)"; exit 1; fi
