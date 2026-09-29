#!/bin/sh
# [rb_s2] THE REFERENCE RE-RUN, seed 2: `RUN_rb_s1.sh` at seed 2, yoked to `vo_s3d2:voi3_dp`,
# gated against `../sostenuto/figures/st_s2/st_gn_yk/`. Saved states at
#   rhm-scaling-data:/rhm_practice_rubato/rb_s2__st_gn_yk/st_gn_yk/ck/c{NNN}.pt
cd "$(dirname "$0")/../../../../../../.."        # -> experiments/
export MODAL_PROFILE=chromatic
python3 rhm/practice/voicing/sotto_voce/aliquot/rubato/launch_detached.py --fn sweep \
    --out-tag rb_s2 --seed 2 --arms "st_gn_yk" \
    --flags-json '{"rb_ck_every": 10, "rb_ck_era": true}'
