#!/usr/bin/env bash
# Cut 4c-arm: reward-free FM re-adaptation under a Shadmehr curl-field drift, on the arm.
#
# Seeds run SEQUENTIALLY on purpose. Both ../README.md and ../../arm_substrate/README.md record
# the same hazard: launching several detached Modal clients from one shell gets the older ones
# evicted, and a client killed before its function's final `volume.commit()` loses results.json
# entirely. `modal run --detach` still blocks this shell until the function finishes, so this
# keeps exactly one live client at a time while protecting each run if the client dies.
#
# Usage:  cd experiments/ && bash mjc/ballistic/arm/train.sh
set -u
export MODAL_PROFILE=chromatic

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGS="$HERE/logs"
mkdir -p "$LOGS"

# `modal run` intermittently dies client-side with "Could not connect to the Modal server"
# BEFORE the function is ever created — it happened on 2 of 3 seeds on the first pass and twice
# more interactively. It is a connection flake, not a job failure, and it exits 0 through the
# pipe, so the loop cannot detect it from the status: grep the log for the completion marker.
SEEDS="${SEEDS:-0 1 2}"
for s in $SEEDS; do
  LOG="$LOGS/arm_readapt_armre_s$s.log"
  for attempt in 1 2 3; do
    echo "=== seed $s attempt $attempt ($(date)) ==="
    modal run --detach mjc/ballistic/arm/arm_readapt.py::arm_readapt \
        --tag "armre_s$s" --seed "$s" 2>&1 | tee "$LOG"
    if grep -q "\[save\] wrote results" "$LOG"; then
      echo "=== seed $s OK ($(date)) ==="; break
    fi
    echo "=== seed $s attempt $attempt FAILED, retrying ==="
  done
done
echo "ALL SEEDS COMPLETE ($(date))"
