#!/bin/bash
# Wait on the MODAL APP(S), not on the local log.
# A container restart kills the `modal run --detach` client that tails
# results/launch_<tag>.log, so that file never gets its completion line even though the
# remote app is fine. `modal app list` is the authority. A transient network failure must
# NOT end the wait, so the loop only breaks on a call that actually returned something.
#
# Usage: wait_app.sh <app_id> [<app_id> ...] [--interval SECONDS]
# Exits when EVERY listed app has left the ephemeral/running/pending states (or has fallen
# off the listing entirely). One waiter per launch batch, per the round's discipline.
INTERVAL=300
APPS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --interval) INTERVAL=$2; shift 2;;
    *) APPS+=("$1"); shift;;
  esac
done
while true; do
  out=$(MODAL_PROFILE=chromatic timeout 120 /usr/local/bin/modal app list 2>/dev/null)
  if [ -n "$out" ]; then
    pending=0
    for app in "${APPS[@]}"; do
      line=$(echo "$out" | grep "$app")
      if [ -n "$line" ] && echo "$line" | grep -qE "ephemeral|running|pending"; then
        pending=$((pending + 1))
      fi
    done
    if [ "$pending" -eq 0 ]; then
      for app in "${APPS[@]}"; do
        echo "DONE: $(echo "$out" | grep "$app" | head -1 || echo "$app GONE_FROM_LIST")"
      done
      break
    fi
  fi
  sleep "$INTERVAL"
done
