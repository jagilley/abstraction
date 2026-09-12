#!/bin/sh
# [en_s8] THE SWEEP. `_try_merge` probes EVERY level with an operative table of >= 2 rows at
# each proposal, bottom-up, instead of only `min(active - 1, maxl)` — the level the era happens
# to be earning. `en_s7` is why: at c188 the L4 book carried 154 classes and collapsed to 8 in
# one proposal, while L2 and L3 had stopped moving cycles earlier and a level the loop had
# SKIPPED was never probed at all. Bottom-up is load-bearing rather than cosmetic: a merge at
# level l re-keys `miners[l+1]`, so probing l before l+1 means l+1 is probed against the map l
# has just installed; top-down would act on evidence one cycle stale inside a single proposal.
# Each level keeps its own `merge_proposal` record and a `merge_sweep` record lists the levels
# with rows against the levels probed — gate E-6 asserts they are equal and ascending.
#
# `merge_n_probe` IS DROPPED TO 128, pre-emptively and on arithmetic rather than after the
# fact. `en_s7`'s single-level probe cost 0.267% of priced time at 256 instances; the sweep
# multiplies the ROW count by the number of live levels (late in the run L2 ~20 + L3 ~111 +
# L4 ~218 + L5 ~68 rows), which at 256 instances projects to ~3.4% — past the ~2% line this
# round was given. At 128 it projects to ~1.7%. `sizing/SIZING.md`'s fixed point for the
# probe's verdict was 32-64 instances, so 128 is still well above where it stabilises; the
# realised share is reported either way, and the change is a confound against `en_s7`'s probe
# precision that the reduction states rather than hides.
#
# SELF-PACED, NO YOKES, and this is the correction the round turns on: a yoked arm replays its
# source's commit cycles and can commit at NO OTHER, and the banked `en_s4:endo_ledger` never
# committed L5 — so `en_s6`'s and `en_s7`'s yoked composed arms could not have committed L5 by
# construction, and their "no L5 commit" was a property of the replay, not a finding. Both arms
# here pace themselves: `endo_ledger_open_ung5` carries the open bit, the ungated L5 miner and
# the class-coverage re-arm hook (which held the loop on the full 201-cycle ladder in `en_s6`),
# and `endo_ledger` is the closed control. Banked for comparison: `given_cat_tok_open_ung5_yk`
# (fb_s1), `endo_ledger` (en_s4 / en_s7), `endo_ledger_open_ung5` (en_s6).
cd "$(dirname "$0")/../../../.."          # -> experiments/
export MODAL_PROFILE=chromatic
python3 rhm/practice/enharmonic/launch_detached.py --fn enharmonic_run --tag en_s8 \
    --eras "1:25:60,2:12:50,3:6:70,4:3:12,5:1:9" --era-caps "60,50,70,12,9" \
    --arms "endo_ledger_open_ung5,endo_ledger" \
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
    --sil-cv 0.15 --lp-min-drop 0.10 --seed 0
