#!/bin/sh
# [st_s1] THE ADMISSION SET ON THE MINER'S LIVE BUILD — seed 0. THREE arms, one CONTAINER EACH
# (a CPU coordinator `.starmap`s them; `../soundboard/DESIGN.md` §6), all `sb_sv_yk` with ONE
# knob added, all CLOCK-YOKED to the banked seed-0 anchor `vo_s3:voi3_dp` — the same yoke the
# banked no-walk reference rides, so every comparison in the reduction is at a matched clock.
#
#     st_gw_yk   admit iff the WORLD'S error on a fresh gate pool of the era's own cell does
#                not rise (pp4's rule, `extend_tol` 0.0). The only arm BILLED for its auditions
#     st_gr_yk   admit iff the PROJECTION'S MEAN LEVEL over the same fired configurations does
#                not fall — pp5's strict pooled read gate, in the loop, at zero world queries
#     st_gn_yk   admit everything offered — the walk's own floor
#
# The gate is the ONLY difference between the three: same cadence (`recert_every` 5), same cap
# (`extend_cap` 8), same gate pool (`n_aud` 192), same tolerance, same order (the miner's own
# `(-count, key)`), same plant, readout, critic, chooser, probe channel, miner, quotient, merge
# op, commit policy, recert and clock. All three record the world's error and the DP's winning
# entry on EVERY trial, so the per-candidate confusion exists on all three.
#
# NOTHING IS RE-RUN THAT IS ALREADY BANKED:
#   sb_s1:sb_sv_yk   the NO-WALK REFERENCE — the arm all three are one knob from
#   vo_s3:voi3_dp    the anchor and the yoke SOURCE
#
# THE FLAGS ARE `SB_FLAGS` in `sostenuto.py` — `soundboard`'s line verbatim plus this node's own
# knobs (`adm_reoffer=False`, `adm_seed_commit=False`, `adm_n_test=256`, `vo_wm_every=5`,
# `vo_dump_era=True`, `vo_dump_refit=True`) — and the coordinator writes the exact per-arm
# kwargs into `<tag>/sweep.json`.
#
# Gates run before launch:
#   sostenuto_gates      A-0, A-1a, A-1b, A-4, A-5, A-7, A-8, A-10, A-11 + 8/8 falsifications
#   sostenuto_gates_gpu  A-2: the in-loop read == `preplay.pj_predict` elementwise, 4/4 red
#   fidelity_smoke       G-F against `../soundboard/soundboard.py` (tag st_gf1)
#   preflight            st_pf1: the whole in-loop path, plus A-1c/A-3/A-6/A-9 and A-1r..A-4r
cd "$(dirname "$0")/../../../../../../.."        # -> experiments/
export MODAL_PROFILE=chromatic
python3 rhm/practice/voicing/sotto_voce/aliquot/sostenuto/launch_detached.py --fn sweep \
    --out-tag st_s1 --seed 0 --arms "st_gw_yk,st_gr_yk,st_gn_yk"
