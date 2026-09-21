#!/usr/bin/env bash
# Independent safety net for the unbatched-navigation shards.
#
# Each shard already tears down its own pod, but that only fires if the shard process is
# alive to run it: a killed driver, a hung episode, or a wedged pod leaves paid GPUs
# running. This watchdog is disowned from the launching session and force-deletes every
# capsweep-sf-* pod when any of these is true:
#   - all shards have written their .done marker  (normal finish)
#   - the wall-clock deadline passes             (hung run)
#   - the RunPod balance falls below the floor   (about to strand mid-episode anyway)
#
# Usage: nohup bash experiments/cap_sweep/runners/watchdog_nav.sh & disown
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1
LOGDIR=experiments/cap_sweep/runners/logs; mkdir -p "$LOGDIR"
LOG="$LOGDIR/watchdog_b2.log"
SHARDS=(B21 B22 B23 B24 DRIFT)
DEADLINE_S=${DEADLINE_S:-14400}   # 4h: cells are expected to take ~1-1.5h plus warm-up
BALANCE_FLOOR=${BALANCE_FLOOR:-2.0}
START=$(date +%s)
log(){ echo "[$(date +%m-%d_%H:%M:%S)] $*" >> "$LOG"; }

KEY=$(grep RUNPOD_API_KEY .env | cut -d= -f2- | tr -d '"'"'"' ')
balance(){
  curl -s -m 20 "https://api.runpod.io/graphql?api_key=$KEY" -H 'content-type: application/json' \
    -d '{"query":"query { myself { clientBalance } }"}' 2>/dev/null \
    | sed -n 's/.*"clientBalance":\([0-9.]*\).*/\1/p'
}
kill_pods(){
  local ids
  ids=$(runpodctl pod list 2>/dev/null | sed -n 's/.*"id": "\([a-z0-9]*\)".*/\1/p')
  for id in $ids; do
    if runpodctl pod list 2>/dev/null | grep -B5 "\"id\": \"$id\"" | grep -q "capsweep-sf-"; then
      log "force-deleting pod $id"; runpodctl pod delete "$id" >> "$LOG" 2>&1
    fi
  done
  # name-based sweep as a backstop, in case the id/name pairing above misses one
  for id in $(runpodctl pod list 2>/dev/null | python3 -c "
import json,sys
try: pods = json.load(sys.stdin)
except Exception: sys.exit(0)
print(' '.join(p['id'] for p in pods if str(p.get('name','')).startswith('capsweep-sf-')))
" 2>/dev/null); do
    log "force-deleting pod $id (name sweep)"; runpodctl pod delete "$id" >> "$LOG" 2>&1
  done
}

log "watchdog armed: deadline ${DEADLINE_S}s, balance floor \$${BALANCE_FLOOR}, shards ${SHARDS[*]}"
while true; do
  done_n=0
  for s in "${SHARDS[@]}"; do [ -f "$LOGDIR/$s.done" ] && done_n=$((done_n+1)); done
  if [ "$done_n" -eq "${#SHARDS[@]}" ]; then
    log "all ${#SHARDS[@]} shards reported done; sweeping for stragglers"; kill_pods
    log "watchdog exiting (normal)"; exit 0
  fi
  elapsed=$(( $(date +%s) - START ))
  if [ "$elapsed" -gt "$DEADLINE_S" ]; then
    log "DEADLINE exceeded (${elapsed}s, $done_n/${#SHARDS[@]} shards done) -- killing pods"
    kill_pods; log "watchdog exiting (deadline)"; exit 1
  fi
  bal=$(balance)
  if [ -n "$bal" ] && awk "BEGIN{exit !($bal < $BALANCE_FLOOR)}"; then
    log "BALANCE \$$bal below floor \$$BALANCE_FLOOR ($done_n/${#SHARDS[@]} done) -- killing pods"
    kill_pods; log "watchdog exiting (balance)"; exit 1
  fi
  log "alive: $done_n/${#SHARDS[@]} done, ${elapsed}s elapsed, balance \$${bal:-?}"
  sleep 300
done
