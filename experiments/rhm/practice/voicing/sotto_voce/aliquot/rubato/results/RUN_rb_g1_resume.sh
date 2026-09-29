#!/bin/sh
# [rb_g1, segment 3] THE SEED-0 GRADED ARMS RESUMED FROM THEIR OWN c130 SAVES (DESIGN §5.5).
# `RUN_rb_g1.sh` restored both arms from `rb_s1`'s c20 and ran them to c137/c140, when a
# preemption of the `sweep` coordinator restarted it with the same input, which cancelled both arms
# and re-spawned them from c20 (the restarted pair reproduced the cancelled pair's c30 and c40
# digests exactly, and was stopped at ~c45). This resumes each arm from its OWN last complete
# save, c130, in its own directory (`{arm}` in `rb_resume`), under the same tag, knobs and
# grade, with `rb_auto_resume` so a further preemption resumes from the latest save instead of
# from c130. Same arm config as `RUN_rb_g1.sh` in everything but the checkpoint's own knobs.
cd "$(dirname "$0")/../../../../../../.."        # -> experiments/
export MODAL_PROFILE=chromatic
python3 rhm/practice/voicing/sotto_voce/aliquot/rubato/launch_detached.py --fn sweep \
    --out-tag rb_g1 --seed 0 --arms "rb_gr_yk,rb_gw_yk" \
    --flags-json '{"rb_resume": "rb_g1__{arm}/{arm}/ck/c130.pt", "rb_auto_resume": true, "rb_allow_cfg": "adm_grade,adm_grade_obs,adm_grade_margin,adm_grade_cons,adm_grade_falsify", "rb_ref_ck": "rb_s1__st_gn_yk/st_gn_yk/ck", "adm_grade_obs": 160, "adm_grade_margin": 0.0, "adm_grade_cons": 4, "rb_ck_every": 10, "rb_ck_era": true}'
