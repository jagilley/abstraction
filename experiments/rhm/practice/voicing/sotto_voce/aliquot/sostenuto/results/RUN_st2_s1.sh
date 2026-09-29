#!/bin/sh
# [st2_s1] ROUND 2 — THE GATE IN THE NEXT LEVEL'S CURRENCY, seed 0. TWO arms, one CONTAINER
# EACH (the CPU coordinator `sweep` `.starmap`s them), both `sb_sv_yk` with the one `admit` knob,
# both CLOCK-YOKED to the banked seed-0 anchor `vo_s3:voi3_dp` — round 1's yoke — so every
# comparison with round 1's three arms and with the banked no-walk reference is at a matched clock.
#
#     st_gy_yk   admit iff the pooled mean of (the NEXT level's at-support share on the fired
#                configuration, from the learner's own reader against the observation panel's
#                level-(l+1) miner) x (the shaped projection's solve level on it) does not fall
#     st_gd_yk   admit iff the panel's level-(l+1) keys AT SUPPORT having the candidate's class
#                as a half sum to >= 1; silent (admit) where the next level holds none
#
# Everything else is round 1's: cadence `recert_every` 5, cap `extend_cap` 8, gate pool `n_aud`
# 192, count order, decided once, empty base, `extend_tol` 0.0, margin `adm_delta` 0.0. Both
# arms log both new objects on every trial, and neither is billed a world read. DESIGN §9.
#
# NOT RE-RUN: st_s1:{st_gw_yk,st_gr_yk,st_gn_yk} (round 1), sb_s1:sb_sv_yk (no-walk reference),
# vo_s3:voi3_dp (the anchor and yoke source).
#
# Gates run before launch: sostenuto_gates (round 1's 8/8 + 11/11 new RED), fidelity_smoke
# (G-F, st2_gf2), preflight st2_pf2 (voi3b_pf_src, st_pf_gy, st_pf_gd; A-1c on every pass,
# the separating pass, E-Y/E-D off the exports).
cd "$(dirname "$0")/../../../../../../.."        # -> experiments/
export MODAL_PROFILE=chromatic
python3 rhm/practice/voicing/sotto_voce/aliquot/sostenuto/launch_detached.py --fn sweep \
    --out-tag st2_s1 --seed 0 --arms "st_gy_yk,st_gd_yk"
