#!/bin/sh
# [en_s9] WHO HEARS THE OPEN BIT'S RE-ARM. One knob, `rearm_advance_only`, default off: with it
# on the class-coverage hook in `_rebuild_ms` calls `loop.acted("open_regime", ...)` ONLY,
# leaving the commit owner's clock alone. Everything else — the arm, the ladder, the caps, the
# floors, the merge licence, the two `figured_bass` knobs, the budget — is `en_s8`'s composed
# arm character for character.
#
# WHY. `en_s8`'s composed arm is the first endogenous arm whose L5 yield gauge ever MOVED, and
# the record of what happened to it is exact:
#   c181-184  commit owner in BURN (c_V=None) after the c180 advance
#   c185      c_V=0.2500, c_v_mult=1.3617 — the gauge moved, 1.36x its own measured floor,
#             and `moved` latched
#   c186      the hook fires (L4 class coverage 12->16, `class_added` — MINING, not a merge)
#             and `_acted_all` resets BOTH policies: `moved` cleared, V cleared, BURN restarted
#   c186-189  c_V=None (four cycles of burn bought by the reset)
#   c190-192  c_V=0.0000, c_v_mult=0.0000 — three quiet-VALUED reads that could license
#             nothing, because `moved` was gone
#   c192      era 4 ends on its cap
#   era 5     the hook fires at c193 and again at c196-c201, every read None; never left burn
# The closed control's era-4 L5 reads were 0.0 at c185-c188 and 0.0 again at c197-c201 — five
# quiet-valued reads and no commit either, for the same reason from the other side: its `moved`
# was never set at all. So the round's question is not "is the gauge quiet" but "was the latch
# alive when it went quiet", and the hook is what was clearing it.
#
# THE KNOB ACTS AT EVERY RE-ARM, NOT ONLY THE LATE ONES. `en_s8`'s composed arm re-armed
# throughout, so this arm's trajectory may diverge from it long before c185; the first cycle of
# divergence is a reading of this arm and is reported as one, not treated as a defect.
#
# NOT CHANGED, deliberately: the merge take path still calls `loop.acted("merge")` AND
# `loop_c.acted("merge")`. A merge installs a different book under the executor and that is an
# action with its own argument for re-arming both clocks; this round moves one hook, not two.
#
# Banked for comparison at reduction time (`analyze_enharmonic.py --bank TAG:ARM`):
#   en_s8:endo_ledger_open_ung5   the same arm with the hook addressing both clocks
#   en_s8:endo_ledger             the closed control
#   fb_s1:given_cat_tok_open_ung5_yk
#
# Gates run before launch: M-0 (offline, PASS), and preflight sweep `s9a` carrying E-6, E-8,
# E-10 and the new E-11 (the knob's own twins `en_pf_ra_on`/`en_pf_ra_off`).
cd "$(dirname "$0")/../../../.."          # -> experiments/
export MODAL_PROFILE=chromatic
python3 rhm/practice/enharmonic/launch_detached.py --fn enharmonic_run --tag en_s9 \
    --eras "1:25:60,2:12:50,3:6:70,4:3:12,5:1:9" --era-caps "60,50,70,12,9" \
    --arms "endo_ledger_open_ung5_ra" \
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
