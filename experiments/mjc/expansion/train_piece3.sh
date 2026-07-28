#!/usr/bin/env bash
# Cut #5, Piece 3 (REDUCED) — DOF accretion as the dimensionality-axis calibration.
#
# Closes the gap in Piece 2's null: Piece 1 certified the instrument responds to a WRONG MODEL
# (error), never to ADDED DIRECTIONS (dimensionality), which is the axis Piece 2 read flat.
#
# THREE SEEDS, and they matter here: the endpoint check put locked-vs-full separation at ~+0.5
# directions against a Piece 2 seed spread of +-0.3 to +-1.0, i.e. marginal. Whether this
# calibration passes or fails is a question about error bars, not about a point estimate.
#
# Seeds run SEQUENTIALLY — detached Modal clients launched from one shell evict each other, and a
# client killed before its function's final `volume.commit()` loses results.json entirely
# (../ballistic/arm/train.sh, ../arm_substrate/README.md gotcha (ii)).
#
# Usage:  cd experiments/ && bash mjc/expansion/train_piece3.sh
set -u
export MODAL_PROFILE=chromatic

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGS="$HERE/logs"
mkdir -p "$LOGS"

SEEDS="${SEEDS:-0 1 2}"
for s in $SEEDS; do
  LOG="$LOGS/support_growing_s$s.log"
  for attempt in 1 2 3; do
    echo "=== seed $s attempt $attempt ($(date)) ==="
    modal run --detach mjc/expansion/support_growing.py::support_growing \
        --tag "sg_s$s" --seed "$s" 2>&1 | tee "$LOG"
    if grep -q "\[save\] wrote results" "$LOG"; then
      echo "=== seed $s OK ($(date)) ==="; break
    fi
    echo "=== seed $s attempt $attempt FAILED, retrying ==="
  done
done
echo "ALL SEEDS COMPLETE ($(date))"
