#!/usr/bin/env bash
# Cut #5, Piece 2b — the DE-CONFOUNDED support-fixed null.
#
# Three corrections to the first pass (tags sf_s0/1/2, launched by train_piece2.sh, which is kept
# as-is so that run stays re-runnable):
#
#   1. `regions_static` COMPLETES THE 2x2. The first pass compared `regions` (four gated curl
#      regions, drifting) against `static` (one global curl, not drifting) — which prices the
#      GEOMETRY as if it were the drift. `regions` is a harder operator outright: live relative
#      residual 0.38-0.51 there vs 0.27-0.34 elsewhere. Each drift arm now differences against
#      its own geometry's no-drift control.
#
#   2. k=8, NOT k=14. Piece 3's calibration found the dimensionality signal is weakest at k=14
#      (+0.27 +- 0.74 for an intervention as blunt as adding three degrees of freedom) and
#      resolves at k=8 (+0.72 +- 0.42). The first pass read its null at the horizon with the
#      least power. Re-aggregating the SAME stored rows at k=8 already moved global-static from
#      +0.68 +- 1.00 to +0.29 +- 0.55, so half this correction was free.
#
#   3. `--buffer-mode accumulate`. `fresh` fine-tunes 1000 Adam steps on 2000 samples every
#      round — 256 epochs — which churns the model. Piece 3, which accumulates, produced less
#      than half the seed spread on the same readout. The smoke also shows the `rail` positive
#      control strengthening from 1.74-2.08x to 2.11-2.84x against Piece 1's 2.7x reference.
#
# Seeds run SEQUENTIALLY — detached Modal clients from one shell evict each other, and a client
# killed before its function's final `volume.commit()` loses results.json entirely.
#
# Usage:  cd experiments/ && bash mjc/expansion/train_piece2b.sh
set -u
export MODAL_PROFILE=chromatic

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGS="$HERE/logs"
mkdir -p "$LOGS"

SEEDS="${SEEDS:-0 1 2}"
for s in $SEEDS; do
  LOG="$LOGS/support_fixed2_s$s.log"
  for attempt in 1 2 3; do
    echo "=== seed $s attempt $attempt ($(date)) ==="
    modal run --detach mjc/expansion/support_fixed.py::support_fixed \
        --tag "sf2_s$s" --seed "$s" --buffer-mode accumulate 2>&1 | tee "$LOG"
    if grep -q "\[save\] wrote results" "$LOG"; then
      echo "=== seed $s OK ($(date)) ==="; break
    fi
    echo "=== seed $s attempt $attempt FAILED, retrying ==="
  done
done
echo "ALL SEEDS COMPLETE ($(date))"
