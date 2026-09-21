#!/usr/bin/env bash
# Independent safety net for the cap-boundary-carryover shard.
#
# The shard tears down its own pod, but only if it is alive to do so: a killed driver, a
# hung episode or a wedged pod otherwise leaves a paid GPU running. This watchdog is
# disowned from the launching session and force-deletes the pods OF THE SHARDS IT WAS GIVEN
# when any of:
#   - every named shard has written its .done marker  (normal finish)
#   - the wall-clock deadline passes                  (hung run)
#   - the RunPod balance falls below the floor        (about to strand mid-episode anyway)
#
# Pass EVERY shard the run launched. It only ever deletes pods named ckpt-<one of those
# shards>: a name-prefix sweep would let one shard's normal finish delete the pods of
# sibling shards still mid-episode, which is how a four-shard run loses three cells.
#
# Usage: nohup bash experiments/cap_sweep/runners/watchdog_ckpt.sh CKPT0 CKPT1 CKPT2 CKPT3 & disown
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1
CK=experiments/cap_sweep/knapsack/qwen3_8b/checkpoint
LOGDIR=experiments/cap_sweep/runners/logs; mkdir -p "$LOGDIR"
SHARDS=("$@"); [ ${#SHARDS[@]} -gt 0 ] || SHARDS=(CKPT1)
LOG="$LOGDIR/watchdog.log"
# three knapsack cells at concurrency 4: ~1.7h (PS) + ~1.5h (PSckpt) + ~0.7h (PP) + warm-up
DEADLINE_S=${DEADLINE_S:-21600}
BALANCE_FLOOR=${BALANCE_FLOOR:-3.0}
START=$(date +%s)
log(){ echo "[$(date +%m-%d_%H:%M:%S)] $*" >> "$LOG"; }

KEY=$(grep RUNPOD_API_KEY .env | cut -d= -f2- | tr -d '"'"'"' ')
balance(){
  curl -s -m 20 "https://api.runpod.io/graphql?api_key=$KEY" -H 'content-type: application/json' \
    -d '{"query":"query { myself { clientBalance } }"}' 2>/dev/null \
    | sed -n 's/.*"clientBalance":\([0-9.]*\).*/\1/p'
}
WANT=$(printf 'ckpt-%s\n' "${SHARDS[@]}" | paste -sd, -)
kill_pods(){
  local listed found
  listed=$(runpodctl pod list 2>/dev/null)
  found=$(WANT="$WANT" python3 -c "
import json,os,sys
want = set(os.environ['WANT'].split(','))
try: pods = json.load(sys.stdin)
except Exception: sys.exit(9)   # tell the caller the listing was unreadable
print(' '.join(p['id'] for p in pods if str(p.get('name','')) in want))
" <<< "$listed" 2>/dev/null)
  if [ $? -eq 9 ]; then
    # A silent no-op here is the dangerous failure: pods keep billing and nobody is told.
    log "WARNING: could not parse pod list -- MANUAL CLEANUP MAY BE REQUIRED for: $WANT"
    return 1
  fi
  for id in $found; do
    log "force-deleting pod $id"; runpodctl pod delete "$id" >> "$LOG" 2>&1
  done
}

log "watchdog armed: deadline ${DEADLINE_S}s, balance floor \$${BALANCE_FLOOR}, pods $WANT"
while true; do
  done_n=0
  for s in "${SHARDS[@]}"; do [ -f "$LOGDIR/$s.done" ] && done_n=$((done_n+1)); done
  if [ "$done_n" -eq "${#SHARDS[@]}" ]; then
    log "all ${#SHARDS[@]} shards reported done; sweeping for stragglers"; kill_pods
    log "watchdog exiting (normal)"; exit 0
  fi
  elapsed=$(( $(date +%s) - START ))
  if [ "$elapsed" -gt "$DEADLINE_S" ]; then
    log "DEADLINE exceeded (${elapsed}s, $done_n/${#SHARDS[@]} done) -- killing pods"
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
