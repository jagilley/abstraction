#!/bin/sh
# [sc_p2] THE FLOOR, seed 2, segment 1 (the pre-check segment, as on seed 0) (DESIGN §2). The ungated junk floor `sc_gn_yk` (`st_gn_yk` +
# `mine_junk` 1.0: the cycle's unsolved chosen answers appended after the donor's solved draw at the
# outcome-blind proportion), clock-yoked to `vo_s3d2:voi3_dp` like every seed-2 arm it is compared
# with, from cycle 0, STOPPED after c40 (`rb_stop_cycle`), saving at every 5th cycle (so the graded
# arms can restore before their first possible grade, which the junk brings forward: the window is
# counted in the panel's observations, and the panel now observes ~4x the rows a cycle) and at every
# era boundary. The fire survey (`sc_fire_keys`) runs on the c40 state. `rb_auto_resume` on.
# The states stay at
#   rhm-scaling-data:/rhm_practice_scordatura/sc_p2__sc_gn_yk/sc_gn_yk/ck/c{NNN}.pt  (+ .json)
cd "$(dirname "$0")/../../../../../../.."        # -> experiments/
export MODAL_PROFILE=chromatic
python3 rhm/practice/voicing/sotto_voce/aliquot/scordatura/launch_detached.py --fn sweep \
    --out-tag sc_p2 --seed 2 --arms "sc_gn_yk" \
    --flags-json '{"rb_ck_every": 5, "rb_ck_era": true, "rb_stop_cycle": 40, "rb_auto_resume": true, "sc_fire_keys": true}'
