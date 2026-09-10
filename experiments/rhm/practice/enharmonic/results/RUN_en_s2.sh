#!/bin/sh
# [enharmonic Q2] the endogenous quotient. Every flag but --arms and the merge knobs is
# `en_s0`'s, verbatim — so `given_cat_tok` in this tag is `en_s0`'s arm re-run (same seed, same
# stream) and is both the ceiling and the in-tag carrier the yokes are read against.
#
# ARM ORDER IS LOAD-BEARING: each yoke replays the arm before it, and `enharmonic_run` asserts
# the source has already run.
#
#   floors: en_smoke, treated arms only (see floors_en_smoke*.json)
#   merge:  tol 0.20 (SIZING.md 5(e)'s window), probe 256 instances over one representative per
#           class capped at 64 rows, every 8 cycles, 1 pair per proposal, use-mass 0.90
cd "$(dirname "$0")/../../../.."          # -> experiments/
export MODAL_PROFILE=chromatic
python3 rhm/practice/enharmonic/launch_detached.py --fn enharmonic_run --tag en_s2 \
    --eras "1:25:60,2:12:50,3:6:70,4:3:12,5:1:9" --era-caps "60,50,70,12,9" \
    --arms "given_cat_tok,endo_yield,flat_yk_endo_yield,endo_ledger,flat_yk_endo_ledger,endo_yield_force,flat_yk_endo_force" \
    --tol-dsil 0.0046 --question-k 2048 \
    --max-macro-level 5 --gy-level 6 \
    --tol-yield-l5 0.2197 --tol-yield-l6 0.1836 \
    --quot-spell-cap 4 --slot-rec --entry-rec-cap 4096 \
    --merge-max-rows 64 \
    --endo-price 267 --gate-frac 0.5 \
    --span-tau-fire 0.50 --perf-alpha 0.05 --perf-tau-w 0.20 \
    --budget 8 --g-budget 482 --n-corrupt 1 --mine-cap 8 --entry-rec \
    --span-min-hold 128 --collect-task-matched --tm-episodes 8192 \
    --recert-every 5 --n-aud 192 --probe-every 8 --n-rt 384 --n-score 256 \
    --sil-cv 0.15 --lp-min-drop 0.10 --seed 0
