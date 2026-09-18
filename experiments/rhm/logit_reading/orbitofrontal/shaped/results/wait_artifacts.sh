#!/bin/bash
# Waits for every expected artefact of the shaped round to land on the volume.
# `modal run --detach` returns before the run does, so this waits on the ARTEFACTS
# (striatum's gotcha), not on a local launch log.
D=/v16_s2_L6_m4_distinct/logit_reading
NEED_PER_ARM="step003000_striatum_a1.json step003000_striatum_swap65k.json \
step003000_striatum_a1_hexcess.npz step003000_striatum_swap65k_hexcess.npz \
step003000_junction_a1.json step001000_calibration.npz step003000_calibration.npz \
step003000_probe_a1.json step001000_probe_a1.json"
for i in $(seq 1 90); do
  missing=0
  for a in task task_ntp ntp; do
    ls_out=$(modal volume ls rhm-scaling-data $D/shape_${a}_s42 2>/dev/null)
    for f in $NEED_PER_ARM; do
      echo "$ls_out" | grep -q "/$f\$" || missing=$((missing+1))
    done
  done
  if [ "$missing" -eq 0 ]; then echo "ALL ARTEFACTS PRESENT after $i polls"; exit 0; fi
  sleep 45
done
echo "TIMEOUT: $missing artefacts still missing"
