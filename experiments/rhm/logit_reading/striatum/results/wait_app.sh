#!/bin/bash
# Waits on the REMOTE app rather than the local launch log: `modal run --detach`'s client
# returns as soon as it disconnects, which is not the end of the run.
APP="${1:-ap-U37ZhaQVTXuiZlSRIcWTZ9}"
DIR=/v16_s2_L6_m4_distinct/logit_reading/traj_eps01_s42
while true; do
  n=$(timeout 180 modal volume ls rhm-scaling-data "$DIR" 2>/dev/null | grep -c "_striatum_.*\.json")
  [ "${n:-0}" -ge 2 ] && { echo "DONE: $n striatum jsons in traj_eps01_s42"; exit 0; }
  st=$(timeout 180 modal app list 2>/dev/null | grep "$APP")
  if [ -n "$st" ] && echo "$st" | grep -qiE "stopped|failed|terminated"; then
    echo "APP ENDED: $st"; exit 0
  fi
  sleep 60
done
