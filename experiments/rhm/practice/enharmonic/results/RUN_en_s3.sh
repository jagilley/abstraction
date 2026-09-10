#!/bin/sh
# [en_s3] THE SECOND PASS AT THE ENDOGENOUS QUOTIENT. `en_s2b`'s flags verbatim except the
# three changes this round is for, so the two tags are comparable arm for arm and only the op
# moved:
#
#   1. WHOLE-PARTITION MERGES. Every within-tol pair from the same probe, closed into alias
#      groups by M-2's partition op, licensed group by group best-loss-first, each on the state
#      the taken ones left. `en_s2b` took one pair per proposal and left five of its L2 book's
#      six alias pairs on the table.
#   2. THE LEDGER AUDITION ON THE LEVEL A MERGE CHANGES (l+1, not l). `en_s2b`'s read
#      `e_keep == e_merge` on all 12 auditions and all 15 forced bad merges because a level-l
#      merge cannot move the level-l table. Gate M-7 is the check that was missing.
#   3. A DERIVED MASS FLOOR. `floors_mass.py` runs `null_abba` on `en_s2b`'s own
#      `mass_at_support` series (span 1, W 4, era boundaries and commits dropped), pooled over
#      its three treated arms: L3 0.005368 (1.05x the stated support/total it replaces), L4
#      0.004712 (1.75x). L5's series never moved in `en_s2b` (no arm built L5) and L6 has none,
#      so both are passed as 0, which means "fall back to the stated threshold and say so in
#      the record" -- every merge event logs which floor it used.
#
# ARM ORDER IS LOAD-BEARING: each yoke replays the arm before it, and `enharmonic_run` asserts
# the source has already run. `given_cat_tok` is NOT re-run here: `en_s2`'s banked arm is the
# ceiling and the reduction reads it from there.
cd "$(dirname "$0")/../../../.."          # -> experiments/
export MODAL_PROFILE=chromatic
python3 rhm/practice/enharmonic/launch_detached.py --fn enharmonic_run --tag en_s3 \
    --eras "1:25:60,2:12:50,3:6:70,4:3:12,5:1:9" --era-caps "60,50,70,12,9" \
    --arms "endo_yield,flat_yk_endo_yield,endo_ledger,flat_yk_endo_ledger,endo_yield_force,flat_yk_endo_force" \
    --tol-dsil 0.0046 --question-k 2048 \
    --max-macro-level 5 --gy-level 6 \
    --tol-yield-l5 0.2197 --tol-yield-l6 0.1836 \
    --tol-mass-l3 0.005367557424645102 --tol-mass-l4 0.004712453259301464 \
    --tol-mass-l5 0.0 --tol-mass-l6 0.0 \
    --quot-spell-cap 4 --slot-rec --entry-rec-cap 4096 \
    --merge-max-rows 64 \
    --endo-price 267 --gate-frac 0.5 \
    --span-tau-fire 0.50 --perf-alpha 0.05 --perf-tau-w 0.20 \
    --budget 8 --g-budget 482 --n-corrupt 1 --mine-cap 8 --entry-rec \
    --span-min-hold 128 --collect-task-matched --tm-episodes 8192 \
    --recert-every 5 --n-aud 192 --probe-every 8 --n-rt 384 --n-score 256 \
    --sil-cv 0.15 --lp-min-drop 0.10 --seed 0
