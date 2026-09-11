#!/bin/sh
# [en_s6] THE COMPOSED ARM: `temperament`'s merge on the live book past the commit, with
# `figured_bass`'s `open_inventory` and `ungate_l5` on. No arm has carried `merge_mode` and
# `open_inventory` together before (`fb_open.patch`'s own flag-vs-table audit says so), and the
# composition is the point: with the inventory open, a merge at level l changes what the
# executor RUNS at l+1 from the next cycle on, not only what a future commit could freeze.
#
# ORDER WITHIN A CYCLE, and it is the question the composition raises. `_rebuild_ms(rearm=True)`
# runs FIRST (block g0, before `port_spec`); `_try_merge` runs in block (g3.5), after the recert
# and before the advance. So a merge taken on cycle c lands AFTER everything that consumes `ms`
# on cycle c — the beams, the auditions and the recert all run the pre-merge tables — and the
# first `operative()` call that sees the merged map is the rebuild at the top of cycle c+1. The
# merge's effect on the executor is deferred by exactly one cycle and never splits a cycle.
# `operative()` keys by the merge's own `LearnedQuotient` (`ClassMiner.build` maps every lower
# row through `self.quot.id_of`), so the rebuilt move set IS built under the merged map.
#
# THE LICENCE is the round's working one: `fourwall`'s audition at l+1 (`merge_mode ledger`),
# strict in the absent case. Every currency is still logged on every proposal — count, arrival
# mass, buildable mass, buildable entries, expected mass with its horizon.
#
# THE RE-ARM. `fb_s0`'s self-paced open arm died at 66 cycles: with the inventory open the book
# moves every cycle, so "moved then quiet" never re-arms on a commit that no longer freezes.
# The regime change under the open bit is a change in what the book can SPELL — a class entering
# an adopted level's operative book, or a merge taking one out — and that now calls `_acted_all`
# exactly as a commit does. A new SPELLING of a held class is not one. Every re-arm is logged
# with its cause in `open_rearms`.
#
# ARMS, sources before yokes.
#   1. `endo_ledger` — RE-RUN in-tag, both knobs off. It must reproduce `en_s4`'s arm bit for
#      bit on every logged series, which is the free offline identity check that the open bit
#      and the expansion instrument are inert in the direction they claim (`given_cat_tok` vs
#      `en_s2` was the same check last round: 30 of 36 series identical, the rest added fields).
#      Running it FIRST also means arm 2's yoke plan is read IN-TAG rather than from the bank;
#      `--yoke-from-tag en_s4` stays as the fallback and as the statement of where the plan
#      came from if the in-tag source were ever absent.
#   2. `endo_ledger_open_ung5_yk` — the same clock, plus open + ungated + merges. Primary read.
#   3. `endo_ledger_open_ung5` — self-paced, with the re-arm hook. Whether the hook holds the
#      loop on the caps or lets it advance is the readout.
#   4. `flat_yk_endo_ledger_open_ung5` — the flat key on arm 3's realised clock.
# `given_cat_tok_open_ung5_yk` (fb_s1) and `flat` (en_s0) are banked, not re-run.
cd "$(dirname "$0")/../../../.."          # -> experiments/
export MODAL_PROFILE=chromatic
python3 rhm/practice/enharmonic/launch_detached.py --fn enharmonic_run --tag en_s6 \
    --eras "1:25:60,2:12:50,3:6:70,4:3:12,5:1:9" --era-caps "60,50,70,12,9" \
    --arms "endo_ledger,endo_ledger_open_ung5_yk,endo_ledger_open_ung5,flat_yk_endo_ledger_open_ung5" \
    --yoke-from-tag en_s4 \
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
