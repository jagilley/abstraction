#!/usr/bin/env bash
# Cut #5, Piece 2 — the support-fixed null. Does a drift that only moves the target function
# leave the model's frontier flat? (Prediction: yes, and that is health.)
#
# THREE SEEDS. The headline is a NULL, and a null needs a spread to be worth anything — "flat"
# only means something against the round-to-round variation the same measurement shows when
# nothing is drifting, which is what `static` plus the seed spread supply together.
#
# Seeds run SEQUENTIALLY on purpose — several detached Modal clients launched from one shell get
# the older ones evicted, and a client killed before its function's final `volume.commit()` loses
# results.json entirely (../ballistic/arm/train.sh, ../arm_substrate/README.md gotcha (ii),
# ../on_policy/README.md §Gotchas). `modal run --detach` still blocks this shell until the
# function finishes, so exactly one client is live at a time while each run stays protected.
#
# Usage:  cd experiments/ && bash mjc/expansion/train_piece2.sh
set -u
export MODAL_PROFILE=chromatic

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGS="$HERE/logs"
mkdir -p "$LOGS"

# `modal run` intermittently dies client-side with "Could not connect to the Modal server" BEFORE
# the function is created; it exits 0 through the pipe, so grep for the completion marker rather
# than trusting the status (../ballistic/arm/ hit this on 2 of 3 seeds).
SEEDS="${SEEDS:-0 1 2}"
for s in $SEEDS; do
  LOG="$LOGS/support_fixed_s$s.log"
  for attempt in 1 2 3; do
    echo "=== seed $s attempt $attempt ($(date)) ==="
    modal run --detach mjc/expansion/support_fixed.py::support_fixed \
        --tag "sf_s$s" --seed "$s" 2>&1 | tee "$LOG"
    if grep -q "\[save\] wrote results" "$LOG"; then
      echo "=== seed $s OK ($(date)) ==="; break
    fi
    echo "=== seed $s attempt $attempt FAILED, retrying ==="
  done
done
echo "ALL SEEDS COMPLETE ($(date))"
