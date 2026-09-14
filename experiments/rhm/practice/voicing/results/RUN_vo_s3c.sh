#!/bin/sh
# [vo_s3c] Q3c — THE SEED PAIR ON THE LOAD-BEARING RESULT. Three arms, ONE tag, seed 1:
#
#   voi3_dp            the anchor character, SELF-PACED at seed 1 — its own commits and advances
#   voi3b_comp_yk      composed / filed,          yoked to THAT arm's clock
#   voi3b_comp_pr_yk   composed / filed + probes, yoked to THAT arm's clock
#
# WHY. `DESIGN.md` §35: with the ladder yoked, `comp_yk` is above the anchor at 6 of 6 frontier
# cells by the repair measure and 0.223 below it in era-5 error at matched clocks, the gain
# growing by era (0.05/0.08/0.22 at eras 3/4/5). That is the first consumption-era value signal
# in this lineage with L5 adopted, it is load-bearing, and its MAGNITUDE rests on one trajectory
# while the structural readouts triangulate only its SIGN. `experiments/CLAUDE.md`'s conditions
# for a seed check therefore hold. Nothing about the world changes: `rule_seed` stays 0, so the
# DGP, the ladder, the caps and every floor are the ones vo_s3b ran on and only the run's own
# stochastic stream moves.
#
# ONE TAG, NOT TWO, AND WHY IT RESOLVES. `phase_a` fills `measured_plans[label]` from each arm's
# realised `loop_actions` as it finishes, and only falls back to reading a banked tag when the
# source is absent AND `--yoke-from-tag` is passed. So an IN-TAG source resolves by arm ORDER
# alone — which is the donor's own mechanic ("arm ORDER is load-bearing") and is the same path
# `voi3b_pf_*` took off `voi3b_pf_src` in the q3b preflight. `--yoke-from-tag` is therefore NOT
# passed here: `voi3b_comp_yk`'s `of` is already `voi3_dp`, and with `voi3_dp` leading the arm
# list the two yoked arms take the SEED-1 plan. No code changed for this round.
#
# THE ANCHOR ARM IS `voi3_dp`, NOT THE BARE `endo_ledger_open_ung5_ra`, for one reason: [Z]'s
# anchor row needs the `rep` / `contains` instruments, which only an arm carrying `vo_record`
# produces, and without them the seed-1 frontier panel would have nothing to compare against.
# `voi3_dp` IS `endo_ledger_open_ung5_ra` plus that record: certified at seed 0 at 0.000e+00 on
# all thirteen series with commits equal and `t_cum` identical to the digit, so the record is
# stream-neutral by measurement and not merely by intent. SAID OUT LOUD: that certification is a
# seed-0 fact and is NOT re-run here — this tag has no banked seed-1 donor to check against.
#
# GATES. G-F is not needed and is not run: no shared path changed since the Q3b landing
# (`voicing.py` is untouched since `vo_s3b` launched; only DESIGN.md and FILES.md moved). What
# the reduction checks is Y-1 on the NEW source — the two yoked arms must replay whatever the
# seed-1 anchor realises — and V-1 on every filed write. If Y-1 shows a CANCELLED commit, that
# is the book finding it is (the anchor at seed 1 may reach different levels than at seed 0) and
# it is reported, not re-planned around.
#
# Banked for comparison at reduction time, so both seeds sit in one table:
#   vo_s3b:voi3b_comp_yk / voi3b_comp_pr_yk / voi3b_rep_yk / voi3b_rep_pr_yk   seed 0, yoked
#   vo_s3:voi3_dp                                                             seed 0, the anchor
cd "$(dirname "$0")/../../../.."          # -> experiments/
export MODAL_PROFILE=chromatic
python3 rhm/practice/voicing/launch_detached.py --fn voicing_run --tag vo_s3c \
    --eras "1:25:60,2:12:50,3:6:70,4:3:12,5:1:9" --era-caps "60,50,70,12,9" \
    --arms "voi3_dp,voi3b_comp_yk,voi3b_comp_pr_yk" \
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
    --sil-cv 0.15 --lp-min-drop 0.10 --seed 1 \
    --vo-rec-cap 8192 --vo-rec-batch 64 --vo-chunk 32 \
    --vo-readback-cell 96 --vo-rep-n 64 --vo-verify-cycles 8 \
    --vo-critic-lr 0.001 --vo-critic-min 256 --vo-critic-hold 0.1 \
    --vo-w 1.0 --vo-probe-n 64
