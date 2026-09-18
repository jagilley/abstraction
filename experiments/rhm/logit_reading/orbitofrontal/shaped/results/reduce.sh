#!/bin/bash
# The whole local reduction: verify the fetched npz members, write the per-arm tables in
# the banked format with the banked analyzers, then the cross-arm tables and figures.
set -e
ROOT=${1:?usage: reduce.sh <local-root>}
cd /home/user/research/experiments
OUT=rhm/logit_reading/orbitofrontal/shaped/results
python $OUT/verify.py "$ROOT"
mkdir -p $OUT/per_arm
for a in shape_task_s42 shape_task_ntp_s42 shape_ntp_s42; do
  python -m rhm.logit_reading.striatum.analyze "$ROOT/$a" --tags a1,swap65k \
      --out $OUT/per_arm/striatum_$a --figs /tmp/shaped_figs_$a >/dev/null
  python -m rhm.logit_reading.striatum.junction.analyze "$ROOT/$a" --tags a1 \
      --out $OUT/per_arm/junction_$a --figs /tmp/shaped_figs_$a >/dev/null
done
python -m rhm.logit_reading.orbitofrontal.shaped.analyze --root "$ROOT" \
    --out $OUT --figs rhm/logit_reading/orbitofrontal/shaped/figs
