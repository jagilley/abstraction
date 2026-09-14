#!/bin/sh
# [figured_bass] fb_s0 — COMMIT THE KEY, NOT THE CONTENT. The 2x2 of the two bits, plus the
# same bits in flat coordinates.
#
# `RUN_en_s3.sh`'s flags VERBATIM except `--arms`: same seed, same ladder, same era caps, same
# dsil floor, same L5/L6 yield floors, same derived mass floors, same budget, same meter, same
# question port. The merge knobs are OFF in every arm here (no arm carries `merge_mode`), so
# `--merge-max-rows` and the four `--tol-mass-*` are inert and are passed only to keep the
# command line diffable against `en_s3` line for line.
#
# THE TWO BITS (both per-ARM `cfg`, neither settable at run level -- an arm IS what it freezes):
#   open_inventory  a commit mints the level's pi slots and starts the corridor on it, and
#                   stops there. `committed[ell]` stays the ADOPTION flag; `operative(ell)`
#                   becomes the frozen partition applied to the live at-support keys.
#   ungate_l5       the COMMITTABLE L5 miner observes every cycle, at the node the observation
#                   panel already reads, instead of only from era 4.
# Why both: `figured_bass/sizing/SIZING.md` [2] found the first non-empty L5 build sitting at
# c156 under EVERY freeze regime it replayed, because the era gate and not the L4 book set that
# cycle -- so the two bits move different things and only the 2x2 separates them.
#
# ARM ORDER IS LOAD-BEARING: each yoke replays the arm before it, and `enharmonic_run` asserts
# the source has already run in this tag.
#   given_cat_tok_open       (a) inventory live
#   given_cat_tok_ung5       (b) L5 miner ungated
#   given_cat_tok_open_ung5  (c) both
#   flat_open                (d) the open bit in flat coordinates, free-running
#   flat_yk_open             yoke of (a) -- same knob, same clock, flat key
#   flat_yk_open_ung5        yoke of (c)
#
# NOT RE-RUN, read from the bank by the reducer:
#   given_cat_tok  <- en_s2      the frozen-inventory anchor and the ceiling
#   flat_yk_tok    <- en_s1      its matched-clock flat comparator
#   flat           <- en_s0      the free-running flat arm
#
# GATES RUN BEFORE THIS, each preflight sweep in its OWN outdir:
#   1. preflight --outdir-tag fb_pf1 --arms "en_pf_tok,en_pf_open_off,en_pf_open,en_pf_ung5,
#      en_pf_open_ung5,en_pf_openflat,en_pf_off"
#        E-7  knob-off identity: `en_pf_open_off` (both bits named False) == `en_pf_tok`
#             (both absent) at 0.000e+00 on e/succ/t_cum/n_moves/n_solved/n_mined and on
#             commits, AND the same oracle bill, AND zero move-set rebuilds.
#        E-8  the flag-vs-table audit -- asserted IN RUN inside `_rebuild_ms` (every macro of
#             an adopted level carries `operative(ell)`), reported here as reached and
#             non-vacuous. The half a machine cannot check is enumerated in `../fb_open.patch`.
#        E-9  the ungated committable L5 miner and the observation panel's L5 miner agree
#             cycle for cycle -- "the same stream made committable", checked.
#        E-0/E-2/E-5 come along for free on `en_pf_off`/`en_pf_tok`.
#   2. fidelity_smoke (G-F) -- the fork still reproduces `tutti.py` at 0.000e+00 with every
#      knob off, on its own tag.
cd "$(dirname "$0")/../../../../.."        # -> experiments/
export MODAL_PROFILE=chromatic
python3 rhm/practice/enharmonic/launch_detached.py --fn enharmonic_run --tag fb_s0 \
    --eras "1:25:60,2:12:50,3:6:70,4:3:12,5:1:9" --era-caps "60,50,70,12,9" \
    --arms "given_cat_tok_open,given_cat_tok_ung5,given_cat_tok_open_ung5,flat_open,flat_yk_open,flat_yk_open_ung5" \
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

# ---------------------------------------------------------------------------------------------
# GATE RECORD (all clean before this launched, 2026-09-10):
#   G-F   gf_fb1    max|fork - tutti| = 0.000e+00, control 0.000e+00, commits equal.
#   E-7   fb_pf2    `en_pf_open_off` (both bits named False) == `en_pf_tok` (both absent):
#                   series delta 0.0, commits identical, oracle bill identical (3,159,900 both),
#                   0 move-set rebuilds with the bit off.
#   E-8   fb_pf2    the in-run flag-vs-table assertion reached on en_pf_open / _ung5 / _openflat;
#                   25 rebuilds, 0 empty fallbacks.  Operative == committed on every cycle there,
#                   because `preflight_seed_miner` fills the miners before anything commits --
#                   which is why E-8b exists.
#   E-8b  fb_pf3    `en_pf_open_late` (seed top-up moved to c8, AFTER the L2 commit at c6):
#                   committed 10 rows, operative 19, on 20 cycle-level pairs c8..c27.  The
#                   executor demonstrably ran on rows that arrived after the freeze.
#   E-9   fb_pf2    the ungated committable L5 miner accrues exactly `n_mined` per cycle and
#                   accrues identically to the observation panel (0 divergences).
#   E-9b  fb_pf2    with the bit OFF, 0 of 15 pre-era-4 cycles saw the L5 miner accrue.
#
# ARM CHECK (against the ARMS literal, before launch): all six are vocab=earned, none carries
# `extend`, `surgery` or `merge_mode`, and each yoke's source precedes it in `--arms`.
