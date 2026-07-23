#!/usr/bin/env bash
# E0 -- pricing the collection flag (see coverage_probe.py). Run from experiments/.
#
#   bash mjc/on_policy/train.sh verify     # backwards-compat gate; run this FIRST and after any
#                                          # edit to arm_env.collect_pool / embodied.py
#   bash mjc/on_policy/train.sh smoke      # ~5 min
#   bash mjc/on_policy/train.sh full       # 3 seeds, detached
set -euo pipefail
cd "$(dirname "$0")/../.."          # -> experiments/
export MODAL_PROFILE=chromatic
LOGS="mjc/on_policy/logs"
mkdir -p "$LOGS"

case "${1:-full}" in
  verify)
    modal run mjc/on_policy/verify_backcompat.py::verify 2>&1 | tee "$LOGS/verify_backcompat.log"
    ;;
  smoke)
    modal run mjc/on_policy/coverage_probe.py::coverage_probe --quick \
      2>&1 | tee "$LOGS/coverage_probe_smoke.log"
    ;;
  full)
    # 3 seeds: E0's P3 is a CONTROL readout, and this node has already been burned once by an
    # aggregate that a single seed flipped (ballistic/directed §The S2 audit). Launch one client
    # per seed; note the directed/ gotcha that several detached clients from one shell are fragile
    # and a killed client loses the local mirror -- the per-run log below is the fallback.
    for s in 0 1 2; do
      modal run --detach mjc/on_policy/coverage_probe.py::coverage_probe \
        --tag "e0_s$s" --seed "$s" 2>&1 | tee "$LOGS/coverage_probe_e0_s$s.log" &
      sleep 20
    done
    wait
    ;;
  readapt)
    # E1 -- Cut 4c-arm three ways (teleport / teleport_matched / on_policy), refined milestone
    # grid. Heavy (10 milestones x 3 arms x k_shoot=1024); detached, one client per seed. The
    # volume commit is at the end and the per-milestone log lines are the fallback if a client is
    # evicted -- `modal volume get .../results.json` recovers a clean artifact (as for e0b_s1).
    for s in 0 1 2; do
      modal run --detach mjc/on_policy/readapt_both_ways.py::readapt_both_ways \
        --tag "rbw_s$s" --seed "$s" 2>&1 | tee "$LOGS/readapt_both_ways_rbw_s$s.log" &
      sleep 90
    done
    wait
    ;;
  local)
    # E2 -- the LOCAL-drift version E1 predicted should stretch the on-policy step. Same 3 arms,
    # spatially-gated curl, in-region/out-region FM-error split. Detached, one client per seed.
    for s in 0 1 2; do
      modal run --detach mjc/on_policy/readapt_local.py::readapt_local \
        --tag "loc_s$s" --seed "$s" 2>&1 | tee "$LOGS/readapt_local_loc_s$s.log" &
      sleep 90
    done
    wait
    ;;
  controls)
    # The oracle-teleport controls, added after the first full run. `B0` is re-run deliberately:
    # it is the check that these logs merge with the `e0_s*` ones (same seed => bit-identical
    # instruments), which `coverage_probe_agg.py` verifies explicitly.
    for s in 0 1 2; do
      modal run --detach mjc/on_policy/coverage_probe.py::coverage_probe \
        --tag "e0b_s$s" --seed "$s" --rungs "B0,B0m,B0r" \
        2>&1 | tee "$LOGS/coverage_probe_e0b_s$s.log" &
      sleep 60
    done
    wait
    ;;
  *)
    echo "usage: train.sh [verify|smoke|full|controls|readapt|local]" >&2; exit 1
    ;;
esac
