#!/usr/bin/env bash
# Pull this node's artefacts off the volume into a local mirror, file by file.
# `modal volume get` on a DIRECTORY returns a zip that has come back truncated before;
# fetch each member on its own and then run verify.py.
set -u
M=${1:?usage: fetch.sh <local-mirror> [tag]}
TAG=${2:-}
SFX=${TAG:+_$TAG}
D=/v16_s2_L6_m4_distinct/logit_reading
mkdir -p "$M"
for w in burst iid; do
  mkdir -p "$M/traj_regime_${w}${SFX}"
  modal volume get --force rhm-scaling-data "$D/regime_world_${w}${SFX}.json" "$M/" || true
  modal volume get --force rhm-scaling-data "$D/traj_regime_${w}${SFX}/train_log.json" \
       "$M/traj_regime_${w}${SFX}/" || true
  for st in 000000 008000 024000 064000; do
    for ext in json npz; do
      modal volume get --force rhm-scaling-data \
        "$D/traj_regime_${w}${SFX}/step${st}_regime_${w}${SFX}.$ext" \
        "$M/traj_regime_${w}${SFX}/" || true
    done
  done
done
find "$M" -type f | sort
