#!/bin/sh
# [ov_s0b] THE ONE-ARM RE-RUN, after DEFECT #5.
#
# `ov_s0`'s S2 combination columns (`comb_auc`, `prior_oof_auc`, and the `incr` the reduction
# derives from them) were the AUC of the POOLED out-of-fold logistic scores. Each fold's score
# is that fold's own affine map of the features, and the folds' INTERCEPTS differ whenever
# their base rates differ -- which they do on a small held-out set of recurring contexts. The
# pooled ranking then ranks fold membership as much as it ranks the verdict. Measured on
# synthetic rows with a known single-feature AUC of 0.698: pooled reads 0.482, per-fold 0.720.
# `ov_oof_auc` now takes the AUC PER FOLD and averages by pair count, and gate R-4 carries the
# demonstration so it cannot come back.
#
# NOTHING ELSE ABOUT THE ARM MOVES. The single-feature columns (`dpz`, `dp`, `conf`, `marg`,
# `crit`), [J1], [J3], [Z], [I] and [O] were never affected -- they are raw scores with no fold
# structure -- so `ov_s0` stands as the tag of record for all of them and this re-run exists
# only to recover the increment. One arm, 1.9 GPU-h.
#
# TWO IDENTITIES RIDE ON IT, both free:
#   `ov_s0b:ovt_comp_pr_sh` must be bit-identical to `ov_s0:ovt_comp_pr_sh` -- the estimator
#   lives in an UNPRICED instrument that consumes no draw, so a changed series would mean the
#   audit is not an instrument;
#   and therefore also to banked `vo_s3b:voi3b_comp_pr_yk` (gate R-1 at full scale, again).
#
# The arm also banks `vo_heads.pt` beside `vo_rows.npz`: the critic, both shadow readouts and
# the trunk they read. This re-run is the last one an audit change should ever cost -- with the
# heads and the rows banked, redoing an audit is a CPU job over the dump.
cd "$(dirname "$0")/../../../../.."        # -> experiments/
export MODAL_PROFILE=chromatic
python3 rhm/practice/voicing/launch_detached.py --fn voicing_run --tag ov_s0b \
    --arms "ovt_comp_pr_sh" \
    --eras "1:25:60,2:12:50,3:6:70,4:3:12,5:1:9" --era-caps "60,50,70,12,9" \
    --yoke-from-tag vo_s3 --ref-tag vo_s3b \
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
    --sil-cv 0.15 --lp-min-drop 0.10 --seed 0 \
    --vo-rec-cap 8192 --vo-rec-batch 64 --vo-chunk 32 \
    --vo-readback-cell 96 --vo-rep-n 64 --vo-verify-cycles 8 \
    --vo-critic-lr 0.001 --vo-critic-min 256 --vo-critic-hold 0.1 \
    --vo-w 1.0 --vo-probe-n 64 \
    --ov-shadow "lin,dir" --ov-free --ov-comb-folds 2 \
    --ov-dump --ov-probe-unif-frac 0.25
