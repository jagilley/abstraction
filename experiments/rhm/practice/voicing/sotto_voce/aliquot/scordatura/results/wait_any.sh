#!/bin/bash
# [scordatura] Wait on MODAL APPS until ANY of them has left the ephemeral/running/pending states
# (wait_app.sh waits for ALL). `modal app list` is the authority; a failed listing never ends it.
INTERVAL=180
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
    for app in "${APPS[@]}"; do
      line=$(echo "$out" | grep "$app")
      if [ -z "$line" ] || ! echo "$line" | grep -qE "ephemeral|running|pending"; then
        echo "DONE: $app ${line:-GONE_FROM_LIST}"
        exit 0
      fi
    done
  fi
  sleep "$INTERVAL"
done
