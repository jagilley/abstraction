#!/bin/bash
# Pull the artefacts this node's reduction reads off the `rhm-scaling-data` volume
# (chromatic).  `modal volume get` has silently truncated a large npz and once returned
# one with a bad CRC on a single member, so `analyze.py` is run only after `verify.py`
# has touched every member of every npz.
ROOT=${1:?usage: fetch.sh <local-root>}
D=/v16_s2_L6_m4_distinct/logit_reading
get () {  # get <volume-relative-path> <local-dir>
  mkdir -p "$2"
  modal volume get --force rhm-scaling-data "$1" "$2/" >/dev/null 2>&1 \
    || echo "MISS $1"
}
for a in task task_ntp ntp; do
  S=$D/shape_${a}_s42; L=$ROOT/shape_${a}_s42
  for f in shape_log.json altitude_identity.json calibration_sweep.json \
           step003000_striatum_a1.json step003000_striatum_a1.npz \
           step003000_striatum_a1_hexcess.npz \
           step003000_striatum_swap65k.json step003000_striatum_swap65k.npz \
           step003000_striatum_swap65k_hexcess.npz \
           step003000_junction_a1.json step003000_junction_a1.npz \
           step001000_probe_a1.json step003000_probe_a1.json; do
    get $S/$f $L
  done
done
S=$D/traj_a1_s42; L=$ROOT/traj_a1_s42
for f in altitude_identity.json step064000_calibration.json \
         step064000_striatum_a1.json step064000_striatum_a1.npz \
         step064000_striatum_a1_hexcess.npz \
         step064000_striatum_swap65k.json step064000_striatum_swap65k.npz \
         step064000_striatum_swap65k_hexcess.npz \
         step064000_junction_a1.json step064000_junction_a1.npz \
         step064000_probe_a1.json; do
  get $S/$f $L
done
echo "fetched -> $ROOT"
