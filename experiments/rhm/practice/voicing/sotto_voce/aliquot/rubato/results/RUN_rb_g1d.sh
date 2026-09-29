#!/bin/sh
# [rb_g1d] THE READ GRADE WITH A PER-SPAN DIET, seed 0 (DESIGN §11). One arm, `rb_grd_yk`:
# `rb_gr_yk` exactly (margin 0, window 160, four consumers, per-pass decisions, the same
# re-offer rule) except that a key's consumers are priced only once the readout's fitted buffer
# holds 512 rows at their span (`adm_grade_diet`, the arm's). RESTORED from `rb_s1`'s c50, the
# latest save before `rb_gr_yk`'s first revocation (c55); `rb_gr_yk` was `rb_s1` by hash at c30,
# c40 and c50, so the restore point is common to both. G-I against `rb_s1`'s saves until this
# arm's first revocation. `rb_auto_resume` on (DESIGN §5.5).
cd "$(dirname "$0")/../../../../../../.."        # -> experiments/
export MODAL_PROFILE=chromatic
python3 rhm/practice/voicing/sotto_voce/aliquot/rubato/launch_detached.py --fn sweep \
    --out-tag rb_g1d --seed 0 --arms "rb_grd_yk" \
    --flags-json '{"rb_resume": "rb_s1__st_gn_yk/st_gn_yk/ck/c050.pt", "rb_auto_resume": true, "rb_allow_cfg": "adm_grade,adm_grade_obs,adm_grade_margin,adm_grade_cons,adm_grade_falsify,adm_grade_diet", "rb_ref_ck": "rb_s1__st_gn_yk/st_gn_yk/ck", "adm_grade_obs": 160, "adm_grade_margin": 0.0, "adm_grade_cons": 4, "rb_ck_every": 10, "rb_ck_era": true}'
