#!/bin/sh
# [rb_s1] THE REFERENCE RE-RUN, seed 0: the ungated admission arm `st_gn_yk` (`admit="none"`),
# exactly `sostenuto`'s `st_s1` configuration (the flags are `SB_FLAGS`, unchanged), clock-yoked
# to the banked `vo_s3:voi3_dp`, with SAVED STATES ON: every era boundary and every 10 cycles.
# Gated after the fact against its banked compact mirror `../sostenuto/figures/st_s1/st_gn_yk/`
# (`reduce_rubato.py --bank-check`): bit-identical outside the rb_* knobs, the rb record and
# wall-clock fields. The saved states stay at
#   rhm-scaling-data:/rhm_practice_rubato/rb_s1__st_gn_yk/st_gn_yk/ck/c{NNN}.pt  (+ .json digest)
# and are what the next round's admit-then-grade op starts from.
cd "$(dirname "$0")/../../../../../../.."        # -> experiments/
export MODAL_PROFILE=chromatic
python3 rhm/practice/voicing/sotto_voce/aliquot/rubato/launch_detached.py --fn sweep \
    --out-tag rb_s1 --seed 0 --arms "st_gn_yk" \
    --flags-json '{"rb_ck_every": 10, "rb_ck_era": true}'
