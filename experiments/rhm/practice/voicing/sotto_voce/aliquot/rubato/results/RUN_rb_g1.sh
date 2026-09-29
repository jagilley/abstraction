#!/bin/sh
# [rb_g1] ADMIT-THEN-GRADE, seed 0 (DESIGN §10). Two arms, one container each (`sweep`), both
# RESTORED from the ungated arm's saved state at c20 (`rb_s1`'s `st_gn_yk`: the latest save before
# c25, the first pass at which a key admitted at c5 has been served for the window), both clock-
# yoked to `vo_s3:voi3_dp` like every arm they are compared with:
#   rb_gr_yk   the grade's value is the shaped projection's level on the fired consumers (0 bill)
#   rb_gw_yk   the grade's value is the world's success on the same fires (billed)
# Window 160 of the level-above panel's observations (DESIGN §10.6), margin 0.0, four consumers
# fired per key. `rb_allow_cfg` names exactly the grade's knobs, which the saved run did not have;
# `rb_ref_ck` points gate G-I at the ungated arm's own saves. Saved states every 10 cycles and every
# era boundary, as the reference. `st_gn_yk` (the no-revocation floor) is NOT re-run.
cd "$(dirname "$0")/../../../../../../.."        # -> experiments/
export MODAL_PROFILE=chromatic
python3 rhm/practice/voicing/sotto_voce/aliquot/rubato/launch_detached.py --fn sweep \
    --out-tag rb_g1 --seed 0 --arms "rb_gr_yk,rb_gw_yk" \
    --flags-json '{"rb_resume": "rb_s1__st_gn_yk/st_gn_yk/ck/c020.pt", "rb_allow_cfg": "adm_grade,adm_grade_obs,adm_grade_margin,adm_grade_cons,adm_grade_falsify", "rb_ref_ck": "rb_s1__st_gn_yk/st_gn_yk/ck", "adm_grade_obs": 160, "adm_grade_margin": 0.0, "adm_grade_cons": 4, "rb_ck_every": 10, "rb_ck_era": true}'
