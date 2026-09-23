#!/bin/bash
# Waits on the REMOTE app until it has STOPPED.  Counting output files is not enough: Modal
# volumes commit in the background, so a cell's json and npz became visible while its
# `_twins.npz` was still being written (the first version of this waiter exited there).
APP="${1:?usage: wait_app.sh <app-id>}"
while true; do
  st=$(timeout 180 modal app list 2>/dev/null | grep "$APP")
  if [ -n "$st" ] && echo "$st" | grep -qiE "stopped|failed|terminated"; then
    echo "APP ENDED: $st"; exit 0
  fi
  sleep 60
done
