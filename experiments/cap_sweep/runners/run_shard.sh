#!/usr/bin/env bash
# Per-pod shard driver for the second-family cap sweep.
# Creates its OWN pod serving the LoRAs named in LORA_MODULES (default: the rule pair
# rp/rs; navigation shards pass the np/ns pair), runs its assigned cells, then
# self-tears-down that pod. Designed to be launched detached, once per shard, so shards
# run in parallel on separate pods.
#
# Usage: [LORA_MODULES="np=org/repo ..."] run_shard.sh <SHARD_NAME> <CELL1> [CELL2 ...]
#   CELLs are per-cell YAML basenames without the leading _ or .yaml,
#   e.g. rule_PS_cap25  (config at experiments/cap_sweep/rule_diagnosis/qwen3_8b/main/_rule_PS_cap25.yaml)
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
SHARD="$1"; shift
CELLS=("$@")
LOG="$LOGDIR/${SHARD}.log"; DONE="$LOGDIR/${SHARD}.done"
: > "$LOG"
log(){ echo "[$(date +%m-%d_%H:%M:%S)] $*" >> "$LOG"; }

# 1. create pod — retry across a GPU-type fallback list (capacity-robust).
# LORA_MODULES selects which adapters this shard serves; the served names must match the
# `model:` in the cell YAMLs (rule cells use rp/rs, navigation cells np/ns).
LORA_MODULES="${LORA_MODULES:?set LORA_MODULES to your own served rule_diagnosis/navigation LoRA repos, e.g. rp=your-org/qwen3-8b-persistent-rule_diagnosis-lora rs=your-org/qwen3-8b-stateless-rule_diagnosis-lora}"
log "serving LoRAs: $LORA_MODULES"
GPUS=("NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe" "NVIDIA H100 80GB HBM3" "NVIDIA H100 PCIe" "NVIDIA H100 NVL" "NVIDIA A40" "NVIDIA RTX A6000")
PID=""
for attempt in 1 2 3 4 5 6 7 8; do
  for GPU in "${GPUS[@]}"; do
    CREATE=$(runpodctl create pod --name "capsweep-sf-$SHARD" \
      --gpuType "$GPU" --gpuCount 1 \
      --imageName "vllm/vllm-openai:latest" \
      --containerDiskSize 80 --mem 80 --vcpu 16 --ports "8000/http" --cost 2.5 \
      --args "--model Qwen/Qwen3-8B --enable-lora --max-lora-rank 64 --max-loras 2 --lora-modules $LORA_MODULES --max-model-len 40960 --gpu-memory-utilization 0.95 --enforce-eager --port 8000" 2>&1)
    PID=$(echo "$CREATE" | sed -n 's/.*pod "\([a-z0-9]*\)".*/\1/p' | head -1)
    if [ -n "$PID" ]; then log "created on [$GPU] attempt $attempt: $PID"; break 2; fi
  done
  log "all GPU types unavailable (attempt $attempt), retrying in $((attempt*30))s"; sleep $((attempt*30))
done
if [ -z "$PID" ]; then log "FAILED to create pod after retries"; echo FAILED > "$DONE"; exit 1; fi

# Register teardown IMMEDIATELY after creation so a SIGINT/SIGTERM or unexpected exit during
# readiness polling or any cell still deletes the paid pod. cleanup retries and clears PID ONLY
# after a confirmed delete, so a transient failure is retried on the next exit rather than
# silently leaking the pod. (Ported from dense/run_matched_shard.sh, commit 95cf879: this script
# still had the weaker version that reported success even when the delete failed.)
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

# 2. wait ready
URL="https://${PID}-8000.proxy.runpod.net/v1/models"
READY=0
for i in $(seq 1 120); do
  if curl -s -m 10 "$URL" 2>/dev/null | grep -q '"data"'; then READY=1; log "ready after ~$((i*12))s"; break; fi
  sleep 12
done
if [ "$READY" != "1" ]; then log "pod never became ready"; echo FAILED > "$DONE"; exit 1; fi  # trap deletes the pod

# 2b. the served adapter names must match the `model:` each cell YAML asks for. A mismatch
# (e.g. serving the rule LoRAs for a navigation shard) otherwise burns the whole pod-hour
# on episodes that 404 one by one.
SERVED=$(curl -s -m 10 "$URL" 2>/dev/null)
for cell in "${CELLS[@]}"; do
  WANT=$(sed -n 's|.*model: openai/||p' "$(cell_cfg "$cell")" | head -1)
  if ! echo "$SERVED" | grep -q "\"id\":\"$WANT\""; then
    log "cell $cell wants model '$WANT' but the pod serves: $(echo "$SERVED" | grep -o '"id":"[^"]*"' | tr '\n' ' ')"
    log "aborting shard before spending pod time"
    echo FAILED > "$DONE"; exit 1  # trap deletes the pod
  fi
done
log "adapter check passed for: ${CELLS[*]}"

# 3. run cells
export OPENAI_API_BASE="https://${PID}-8000.proxy.runpod.net/v1"
export OPENAI_API_KEY="dummy"
export CODEACT_EXTRA_BODY_JSON='{"chat_template_kwargs":{"enable_thinking":false}}'
FAIL=0
for cell in "${CELLS[@]}"; do
  case "$cell" in
    rule_*)     TROOT="experiments/cap_sweep/rule_diagnosis/task_defs" ;;
    navb[0-9]*_*)
      # navb2_PS_cap25 -> the b=2 derived task set; the digits between "navb" and "_"
      B="${cell#navb}"; B="${B%%_*}"
      TROOT="experiments/cap_sweep/navigation/task_defs_batch$B" ;;
    nav_*)      TROOT="experiments/cap_sweep/navigation/task_defs" ;;
    *) log "unknown family for $cell"; FAIL=1; continue ;;
  esac
  log "RUN $cell (tasks_root=$TROOT)"
  if .venv/bin/python -m codeact_runtime.benchmark.benchmark \
      --config "$(cell_cfg "$cell")" --tasks_root "$TROOT" --max-examples 16 >> "$LOG" 2>&1; then
    log "DONE $cell"
  else
    rc=$?; log "FAILED $cell (exit $rc)"; FAIL=1
  fi
done

# 4. self-teardown (ALWAYS, to stop billing). DONE is written only after a CONFIRMED delete:
#    a shard that reported success while its pod kept billing is how a previous run leaked ~$40.
cleanup
if [ -n "${PID:-}" ]; then
  echo FAILED > "$DONE"
  log "shard: could NOT confirm pod deletion -- MANUAL CLEANUP REQUIRED (cells FAIL=$FAIL)"
  exit 1
fi
if [ "$FAIL" = "0" ]; then echo DONE > "$DONE"; log "shard complete, pod deleted";
else echo FAILED > "$DONE"; log "shard FAILED (>=1 cell), pod deleted"; exit 1; fi
