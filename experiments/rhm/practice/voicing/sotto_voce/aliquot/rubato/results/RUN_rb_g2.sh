#!/bin/sh
# [rb_g2] ADMIT-THEN-GRADE, seed 2 (DESIGN §10). The seed-0 pair (`RUN_rb_g1.sh`) on seed 2: both
# arms RESTORED from the ungated arm's saved state at c20 (`rb_s2`'s `st_gn_yk`: the latest save
# before c30, the first pass at which a key admitted at c5 has been served for the window on this
# seed; DESIGN §10.6), both clock-yoked to `vo_s3d2:voi3_dp` like every seed-2 arm they are
# compared with. Same window, margin and consumers as seed 0; G-I against `rb_s2`'s own saves.
# `rb_auto_resume`: a preempted arm resumes from its own latest save (DESIGN §5.5).
cd "$(dirname "$0")/../../../../../../.."        # -> experiments/
export MODAL_PROFILE=chromatic
python3 rhm/practice/voicing/sotto_voce/aliquot/rubato/launch_detached.py --fn sweep \
    --out-tag rb_g2 --seed 2 --arms "rb_gr_yk,rb_gw_yk" \
    --flags-json '{"rb_resume": "rb_s2__st_gn_yk/st_gn_yk/ck/c020.pt", "rb_auto_resume": true, "rb_allow_cfg": "adm_grade,adm_grade_obs,adm_grade_margin,adm_grade_cons,adm_grade_falsify", "rb_ref_ck": "rb_s2__st_gn_yk/st_gn_yk/ck", "adm_grade_obs": 160, "adm_grade_margin": 0.0, "adm_grade_cons": 4, "rb_ck_every": 10, "rb_ck_era": true}'
