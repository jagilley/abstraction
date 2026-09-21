#!/bin/bash
# Waits on the REMOTE app, not on the local launch log: `modal run --detach`'s client
# returns as soon as it disconnects, which is not the end of the run.
APP="${1:-ap-vD3PdhLEWEqYxrLSvdFk4K}"
DIR=/v16_s2_L6_m4_distinct/logit_reading/traj_a1_s42
while true; do
  n=$(timeout 180 modal volume ls rhm-scaling-data "$DIR" 2>/dev/null | grep -c "_adapt_\(a1\|swap65k\)\.json")
  [ "${n:-0}" -ge 2 ] && { echo "DONE: $n adapt jsons in traj_a1_s42"; exit 0; }
  st=$(timeout 180 modal app list 2>/dev/null | grep "$APP")
  if [ -n "$st" ] && echo "$st" | grep -qiE "stopped|failed|terminated"; then
    echo "APP ENDED: $st"; exit 0
  fi
  sleep 60
done
