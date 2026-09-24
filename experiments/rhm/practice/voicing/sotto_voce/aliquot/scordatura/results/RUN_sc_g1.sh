#!/bin/sh
# [sc_g1] ADMIT-THEN-GRADE ON THE JUNK FLOOR, seed 0 (DESIGN §4). Two arms, each RESTORED from the
# floor's c10 save (`sc_p1`): the latest save before a grade can fire — read off the floor's own
# record: keys admitted at the c5 pass are 116 panel observations old at the c10 pass and 294 at c15
# (window 160; the panel observes ~32 rows a cycle with the junk against ~8 without), and the L3
# panel holds keys at support from c1, so c15 is the first cycle a grade can fire.
#   `sc_gw_yk`   the world grade exactly as `rb_gw_yk` (margin 0, window 160, four consumers)
#   `sc_grd_yk`  the read grade as `rb_grd_yk` (diet 512) + `adm_grade_persist` 2
# G-I against the floor's saves until each arm's first revocation. `rb_auto_resume` on.
cd "$(dirname "$0")/../../../../../../.."        # -> experiments/
export MODAL_PROFILE=chromatic
python3 rhm/practice/voicing/sotto_voce/aliquot/scordatura/launch_detached.py --fn sweep \
    --out-tag sc_g1 --seed 0 --arms "sc_gw_yk,sc_grd_yk" \
    --flags-json '{"rb_resume": "sc_p1__sc_gn_yk/sc_gn_yk/ck/c010.pt", "rb_auto_resume": true, "rb_allow_cfg": "adm_grade,adm_grade_obs,adm_grade_margin,adm_grade_cons,adm_grade_falsify,adm_grade_diet,adm_grade_persist", "rb_ref_ck": "sc_p1__sc_gn_yk/sc_gn_yk/ck", "adm_grade_obs": 160, "adm_grade_margin": 0.0, "adm_grade_cons": 4, "rb_ck_every": 10, "rb_ck_era": true}'
