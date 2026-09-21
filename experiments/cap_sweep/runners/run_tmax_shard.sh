#!/usr/bin/env bash
# Per-pod driver for the T_max shard: self-creates a pod serving the knapsack
# PERSISTENT adapter, runs the given tmax cells, self-tears-down.
# Usage: run_tmax_shard.sh <SHARD_NAME> <CELL1> [CELL2 ...]
#   CELLs are tmax config basenames, e.g. PS_cap25_T160
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1
SF=experiments/cap_sweep/knapsack/qwen3_8b/tmax
LOGDIR=experiments/cap_sweep/runners/logs; mkdir -p "$LOGDIR"
SHARD="$1"; shift; CELLS=("$@")
LOG="$LOGDIR/${SHARD}.log"; DONE="$LOGDIR/${SHARD}.done"; : > "$LOG"
log(){ echo "[$(date +%m-%d_%H:%M:%S)] $*" >> "$LOG"; }
# Released adapter: https://huggingface.co/runtime-contracts/qwen3-8b-knapsack-lora-persistent-seed3407
PERSISTENT_LORA="${PERSISTENT_LORA:?set to your own served knapsack-persistent LoRA repo}"

GPUS=("NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe" "NVIDIA H100 80GB HBM3" "NVIDIA H100 PCIe" "NVIDIA H100 NVL" "NVIDIA A40")
PID=""
for attempt in 1 2 3 4 5 6 7 8; do
  for GPU in "${GPUS[@]}"; do
    CREATE=$(runpodctl create pod --name "tmax-$SHARD" --gpuType "$GPU" --gpuCount 1 \
      --imageName "vllm/vllm-openai:latest" --containerDiskSize 80 --mem 80 --vcpu 16 \
      --ports "8000/http" --cost 2.5 \
      --args "--model Qwen/Qwen3-8B --enable-lora --max-lora-rank 64 --max-loras 2 --lora-modules persistent=$PERSISTENT_LORA --max-model-len 40960 --gpu-memory-utilization 0.95 --enforce-eager --port 8000" 2>&1)
    PID=$(echo "$CREATE" | sed -n 's/.*pod "\([a-z0-9]*\)".*/\1/p' | head -1)
    [ -n "$PID" ] && { log "created on [$GPU]: $PID"; break 2; }
  done
  log "all GPUs unavailable (attempt $attempt), retry ${attempt}0s"; sleep $((attempt*10))
done
[ -z "$PID" ] && { log "FAILED create"; echo FAILED > "$DONE"; exit 1; }

URL="https://${PID}-8000.proxy.runpod.net/v1/models"
READY=0
for i in $(seq 1 150); do curl -s -m 10 "$URL" 2>/dev/null | grep -q '"data"' && { READY=1; log "ready ~$((i*12))s"; break; }; sleep 12; done
[ "$READY" = "1" ] || { log "pod never became ready; deleting"; runpodctl pod delete "$PID" >> "$LOG" 2>&1; echo FAILED > "$DONE"; exit 1; }

export OPENAI_API_BASE="https://${PID}-8000.proxy.runpod.net/v1"
export OPENAI_API_KEY="dummy"
export CODEACT_EXTRA_BODY_JSON='{"chat_template_kwargs":{"enable_thinking":false}}'
FAIL=0
for cell in "${CELLS[@]}"; do
  log "RUN $cell"
  if .venv/bin/python -m codeact_runtime.benchmark.benchmark --config "$SF/_${cell}.yaml" --tasks_root experiments/cap_sweep/knapsack/task_defs --max-examples 20 >> "$LOG" 2>&1; then
    log "DONE $cell"
  else
    rc=$?; log "FAILED $cell (exit $rc)"; FAIL=1
  fi
done
# always tear the pod down (stop billing) regardless of cell outcomes
log "tearing down pod $PID"
runpodctl pod delete "$PID" >> "$LOG" 2>&1
if [ "$FAIL" = "0" ]; then echo DONE > "$DONE"; log "shard complete, pod $PID deleted";
else echo FAILED > "$DONE"; log "shard FAILED (>=1 cell), pod $PID deleted"; exit 1; fi
