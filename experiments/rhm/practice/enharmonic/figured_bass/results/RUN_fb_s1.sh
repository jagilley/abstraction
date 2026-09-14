#!/bin/sh
# [figured_bass] fb_s1 — THE OPEN BOOK ON THE ANCHOR'S OWN CLOCK.
#
# `fb_s0` found the open bit paying in PACING, not in content: both open arms advanced on
# delta-silence at the first legal read of eras 2 and 3 and lived 66 cycles against the anchor's
# 176, so what an open book would hold over a full lifetime was never observed. These two arms
# fix the clock and leave only the inventory free, by replaying the BANKED `given_cat_tok`'s
# realised actions (`en_s2`) with `--yoke-from-tag`:
#
#     L2 commit @c12 | advance @c35 | advance @c85 | L4 commit @c95
#                    | advance @c155 | advance @c167 | advance @c176      (7 actions, verified)
#
# `given_cat_tok` is NOT re-run; the plan is read off the volume at
# rhm_practice_enharmonic/en_s2/given_cat_tok/results.json (present, checked before launch).
#
# WHAT A REPLAYED COMMIT INSTALLS UNDER AN OPEN INVENTORY. The yoke idiom has only ever run
# with the FLAT key, so this is stated rather than assumed. A yoke replays the source's CYCLE,
# never its table: the commit body is untouched, `tbl = miners[active].build(operative(active
# - 1), mine_support)`, so the L4 commit replayed at c95 installs THIS arm's own live L4 book
# at c95 over its own (open, hence live) L3 book. It will not equal the source's 64-row /
# 3-class book and is not meant to. Because the open bit is on, `committed[4]` is thereafter
# only the ADOPTION RECORD and `operative(4)` keeps rebuilding from c96. Every commit event
# logs `open_inventory`, `installed_is_live_build` and `by_clock`, `[open] ... COMMIT installs
# the LIVE build` prints at the cycle, and `log["open"]` carries operative-vs-committed per
# cycle -- so the claim is checkable in the record. If the live build is empty at a replayed
# cycle the donor's own `empty_table` guard cancels the commit and the reduction reports it.
#
# Q0's prediction for this configuration (`../sizing/SIZING.md` [1]/[2], an all-else-equal
# replay on the banked observation stream, NOT a prediction of a run): 9 of 13 L4 token classes
# and 52 of 73 lookup-able L5 class pairs by c176, against the frozen book's 3 and 6.
#
# GATES.
#   E-7 knob-off identity            banked, fb_pf2 (series delta 0.0, identical oracle bill).
#   E-8 / E-8b flag-vs-table         banked, fb_pf2 / fb_pf3.
#   E-9 / E-9b ungated L5            banked, fb_pf2.
#   G-F vs tutti                     banked, gf_fb1 (0.000e+00).
#   yoke + open bit, end to end      ALREADY RUN: `fb_s0`'s `flat_yk_open` and
#                                    `flat_yk_open_ung5` are yoke arms with `open_inventory`
#                                    on; 66 cycles, 55 rebuilds, 53 cycles moved, 0 empty
#                                    fallbacks, planned == realised on all 5 advances.
#   --yoke-from-tag                  ALREADY RUN: `en_s1`'s flat yokes took `en_s0`'s plan.
#   THE ONE NEW COMBINATION is a CLASS-KEYED yoke (a `quot` miner under `loop.kind == "yoke"`).
#   No loop code reads the quotient, and both halves have run separately, so this is not
#   preflighted separately -- said out loud rather than implied.
#   POST-RUN, in the reduction: the yoke replay check, planned vs realised on every commit and
#   advance, with cancellations REPORTED and not asserted away.
cd "$(dirname "$0")/../../../../.."        # -> experiments/
export MODAL_PROFILE=chromatic
python3 rhm/practice/enharmonic/launch_detached.py --fn enharmonic_run --tag fb_s1 \
    --eras "1:25:60,2:12:50,3:6:70,4:3:12,5:1:9" --era-caps "60,50,70,12,9" \
    --arms "given_cat_tok_open_yk,given_cat_tok_open_ung5_yk" \
    --yoke-from-tag en_s2 \
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
