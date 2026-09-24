#!/bin/sh
# [sc_g2w] ADMIT-THEN-GRADE ON THE JUNK FLOOR, seed 2, one arm (DESIGN §4): `sc_gw_yk`, the world grade exactly as `rb_gw_yk` (margin 0, window 160, four consumers).
# RESTORED from the seed-2 floor's c5 save (`sc_p2`): the latest save before a grade can fire,
# read off that floor's own record — keys admitted at the c5 pass are 161 panel observations old
# at the c10 pass (window 160), and the L3 panel holds keys at support from c1, so c10 is the first
# cycle a grade can fire on seed 2 (c15 on seed 0). The two seed-2 graded arms are launched as
# separate tags only so that no more than four containers run at once. G-I against the floor's saves
# until the first revocation. `rb_auto_resume` on.
cd "$(dirname "$0")/../../../../../../.."        # -> experiments/
export MODAL_PROFILE=chromatic
python3 rhm/practice/voicing/sotto_voce/aliquot/scordatura/launch_detached.py --fn sweep \
    --out-tag sc_g2w --seed 2 --arms "sc_gw_yk" \
    --flags-json '{"rb_resume": "sc_p2__sc_gn_yk/sc_gn_yk/ck/c005.pt", "rb_auto_resume": true, "rb_allow_cfg": "adm_grade,adm_grade_obs,adm_grade_margin,adm_grade_cons,adm_grade_falsify,adm_grade_diet,adm_grade_persist", "rb_ref_ck": "sc_p2__sc_gn_yk/sc_gn_yk/ck", "adm_grade_obs": 160, "adm_grade_margin": 0.0, "adm_grade_cons": 4, "rb_ck_every": 10, "rb_ck_era": true}'
