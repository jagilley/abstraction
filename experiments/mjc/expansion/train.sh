#!/usr/bin/env bash
# Cut #5, Piece 1 — the saturation gate. Does the frontier instrument (beta,
# R_res_participation, frontier mass) have any dynamic range on the arm's state-space
# prediction target, and where in the rollout-horizon sweep?
#
# THREE SEEDS, and here the usual "don't think about seed-dependence" default does NOT apply:
# the verdict is a RATIO between FM variants (stale vs matched frontier mass) evaluated on
# n_windows=384 samples, and the smoke put that ratio at 0.67-1.77 straddling the 1.5 decision
# threshold. A gate whose call could flip on one seed is not a gate. Aggregated by
# `saturation_gate_agg.py`.
#
# Seeds run SEQUENTIALLY on purpose — the hazard recorded in ../ballistic/arm/train.sh,
# ../arm_substrate/README.md gotcha (ii) and ../on_policy/README.md §Gotchas: several detached
# Modal clients launched from one shell get the older ones evicted, and a client killed before
# its function's final `volume.commit()` loses results.json entirely. `modal run --detach` still
# blocks this shell until the function finishes, so this keeps exactly one live client at a time
# while protecting each run if the client dies.
#
# Usage:  cd experiments/ && bash mjc/expansion/train.sh
set -u
export MODAL_PROFILE=chromatic

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGS="$HERE/logs"
mkdir -p "$LOGS"

# `modal run` intermittently dies client-side with "Could not connect to the Modal server"
# BEFORE the function is ever created — it hit 2 of 3 seeds on ballistic/arm's first pass. It is
# a connection flake, not a job failure, and it exits 0 through the pipe, so the loop cannot
# detect it from the status: grep the log for the completion marker instead.
SEEDS="${SEEDS:-0 1 2}"
for s in $SEEDS; do
  LOG="$LOGS/saturation_gate_s$s.log"
  for attempt in 1 2 3; do
    echo "=== seed $s attempt $attempt ($(date)) ==="
    modal run --detach mjc/expansion/saturation_gate.py::saturation_gate \
        --tag "gate_s$s" --seed "$s" 2>&1 | tee "$LOG"
    if grep -q "\[save\] wrote results" "$LOG"; then
      echo "=== seed $s OK ($(date)) ==="; break
    fi
    echo "=== seed $s attempt $attempt FAILED, retrying ==="
  done
done
echo "ALL SEEDS COMPLETE ($(date))"
