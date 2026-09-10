#!/bin/sh
# [en_s5] THE FOURTH PASS. `en_s4`'s flags verbatim except the two changes this round is for.
#
#   1. THE YIELD LICENCE READS DEFERRED ARRIVAL. `--merge-gauge expected`: over the next
#      level's keys, the EXPECTED mass at support once the level's remaining mining budget has
#      been spent — each key's future count modelled as Poisson at its observed share of the
#      stream times `N_rem` (remaining MINING cycles on the gated schedule times `mine_cap`),
#      and `pge` from `tutti/sizing/phase0_l5.py` for the tail. What a merge buys is that the
#      next level's keys reach support SOONER; the instantaneous mass gauge is that quantity at
#      horizon zero, which is why it read exactly 0.00000 on all six of `en_s4` c188's groups.
#      Two gauges considered and rejected on the way, both now gated as fixtures and logged as
#      currencies: key-BUILDABILITY (M-4d) cannot move at all under the current proposal rule —
#      a merge only ever shrinks the book's class set and the classes a proposal contains come
#      from the book's own rows — and ENTRY COUNT (M-4e) reproduces c188's 1 -> 16 exactly but
#      is redundancy the ledger has already shown to be execution-invisible. All four
#      currencies and their floors are logged on every proposal whichever one licenses. The
#      floor stays STATED at one at-support key's worth of mass, source in every record;
#      `ClassMiner.build` now logs `mass_at_support` and `mass_buildable` per cycle so the next
#      round can derive one.
#   2. THE EXPANSION-CHOICE INSTRUMENT, on in every arm, oracle-contained and inert (gate E-7):
#      per macro call on the AUDITION DP path, whether the row the max-sum DP chose has a token
#      class CONTAINING the feature the instance actually demands at that (level, node), read
#      off the clean derivation the instances already carry. Logged per cycle per cell as
#      calls / contains / successes.
#
# FOUR ARMS, sources before yokes. `given_cat_tok` is RE-RUN here with the instrument on: it
# must reproduce the banked `en_s2` arm bit for bit on every logged series, which is a free
# cross-tag identity check for the instrument's inertness. `given_cat_tok_open_yk` is
# `figured_bass`'s open-inventory arm, yoked to the in-tag `given_cat_tok`. `endo_ledger`
# (en_s4) and `flat` (en_s0) are banked, not re-run.
cd "$(dirname "$0")/../../../.."          # -> experiments/
export MODAL_PROFILE=chromatic
python3 rhm/practice/enharmonic/launch_detached.py --fn enharmonic_run --tag en_s5 \
    --eras "1:25:60,2:12:50,3:6:70,4:3:12,5:1:9" --era-caps "60,50,70,12,9" \
    --arms "endo_yield,flat_yk_endo_yield,given_cat_tok,given_cat_tok_open_yk" \
    --tol-dsil 0.0046 --question-k 2048 \
    --max-macro-level 5 --gy-level 6 \
    --tol-yield-l5 0.2197 --tol-yield-l6 0.1836 \
    --tol-mass-l3 0.005367557424645102 --tol-mass-l4 0.004712453259301464 \
    --tol-mass-l5 0.0 --tol-mass-l6 0.0 \
    --merge-gauge expected \
    --quot-spell-cap 4 --slot-rec --entry-rec-cap 4096 \
    --merge-max-rows 64 --merge-use-frac 1.0 \
    --endo-price 267 --gate-frac 0.5 \
    --span-tau-fire 0.50 --perf-alpha 0.05 --perf-tau-w 0.20 \
    --budget 8 --g-budget 482 --n-corrupt 1 --mine-cap 8 --entry-rec \
    --span-min-hold 128 --collect-task-matched --tm-episodes 8192 \
    --recert-every 5 --n-aud 192 --probe-every 8 --n-rt 384 --n-score 256 \
    --sil-cv 0.15 --lp-min-drop 0.10 --seed 0
