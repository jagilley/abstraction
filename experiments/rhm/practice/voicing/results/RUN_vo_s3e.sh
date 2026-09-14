#!/bin/sh
# [vo_s3e] Q3e — THE FRONTIER PANEL ON A SECOND SEED THAT REACHED IT. Two arms, one tag,
# seed 2, both CLOCK-YOKED to the banked seed-2 anchor:
#
#   voi3b_comp_yk      composed / filed
#   voi3b_comp_pr_yk   composed / filed + probes
#
# THE SOURCE IS BANKED, so `--yoke-from-tag vo_s3d2` is passed and no anchor is re-run:
# `vo_s3d2:voi3_dp` is `voi3_dp` at `--seed 2`, 201 cycles, commits L2@c59(13) / L3@c77(46) /
# L4@c157(152), advances 60/110/180/192/201 — every one on CAP, `0 quiet / 5 cap`. Its L4 slots
# therefore live from c157 to c201, which is the window [Z] reads.
#
# WHY THIS SEED AND WHY NOW. Q3d (`DESIGN.md` §37): L5 replicates at 1 of 4 seeds, L4 at 2 of 4,
# L3 at 3 of 4. §35's frontier result was measured on seed 0 alone, and Q3c could not check it
# because seed 1 never left L2. Seed 2 is the other draw that reached L4, so it is the one draw
# on which the frontier comparison can be made a second time. The scope statement stands either
# way and is not weakened by this run: on the draws where the ladder reaches the frontier, the
# composed chooser is better there — this asks whether "the draws" is more than one.
#
# NOTE WHAT THIS RUN CANNOT DO: seed 2 committed L4 and never L5, so it can test §35's L4 cells
# and NOT its L5 cells (5n0 0.526 vs 0.356 and 5n1 0.830 vs 0.691 remain single-draw). Said here
# so the reduction is not read as a replication of more than it is.
#
# `--seed 2` so the treated arms run on the same stochastic stream as their source, exactly as
# vo_s3b's arms ran on seed 0's. `rule_seed` stays 0 throughout: the world never changed.
#
# NO CODE CHANGED and G-F is NOT run — `voicing.py` is untouched since the Q3b landing, verified
# clean in `git status` before launch. What the reduction checks is Y-1 on the NEW source (the
# arms must replay 59/77/157 and 60/110/180/192/201, with any CANCELLED commit reported as the
# book finding it is) and V-1 on every filed write.
#
# Banked beside at reduction time, so seeds 0 and 2 sit in one frontier table:
#   vo_s3d2:voi3_dp                    seed 2, the anchor and the yoke source
#   vo_s3:voi3_dp                      seed 0, the anchor
#   vo_s3b:voi3b_comp_yk / _comp_pr_yk seed 0, the same two arms on seed 0's clock
#   vo_s3b:voi3b_rep_yk / _rep_pr_yk   seed 0, the replace half of the 2x2
cd "$(dirname "$0")/../../../.."          # -> experiments/
export MODAL_PROFILE=chromatic
python3 rhm/practice/voicing/launch_detached.py --fn voicing_run --tag vo_s3e \
    --eras "1:25:60,2:12:50,3:6:70,4:3:12,5:1:9" --era-caps "60,50,70,12,9" \
    --arms "voi3b_comp_yk,voi3b_comp_pr_yk" \
    --yoke-from-tag vo_s3d2 \
    --tol-dsil 0.0046 --question-k 2048 \
    --max-macro-level 5 --gy-level 6 \
    --tol-yield-l5 0.2197 --tol-yield-l6 0.1836 \
    --tol-mass-l3 0.005367557424645102 --tol-mass-l4 0.004712453259301464 \
    --tol-mass-l5 0.0 --tol-mass-l6 0.0 \
    --merge-gauge expected \
    --quot-spell-cap 4 --slot-rec --entry-rec-cap 4096 \
    --merge-max-rows 256 --merge-use-frac 1.0 --merge-n-probe 128 \
    --endo-price 267 --gate-frac 0.5 \
    --span-tau-fire 0.50 --perf-alpha 0.05 --perf-tau-w 0.20 \
    --budget 8 --g-budget 482 --n-corrupt 1 --mine-cap 8 --entry-rec \
    --span-min-hold 128 --collect-task-matched --tm-episodes 8192 \
    --recert-every 5 --n-aud 192 --probe-every 8 --n-rt 384 --n-score 256 \
    --sil-cv 0.15 --lp-min-drop 0.10 --seed 2 \
    --vo-rec-cap 8192 --vo-rec-batch 64 --vo-chunk 32 \
    --vo-readback-cell 96 --vo-rep-n 64 --vo-verify-cycles 8 \
    --vo-critic-lr 0.001 --vo-critic-min 256 --vo-critic-hold 0.1 \
    --vo-w 1.0 --vo-probe-n 64
