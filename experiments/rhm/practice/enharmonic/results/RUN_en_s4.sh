#!/bin/sh
# [en_s4] THE THIRD PASS. `en_s3`'s flags verbatim except the two changes this round is for,
# and two arms fewer.
#
#   1. PROBE EVERY CLASS. `--merge-use-frac 1.0`: one representative per class over the whole
#      operative book, `merge_max_rows` 64 the only cap. `en_s3`'s alias audit is the reason —
#      at 0.90 the probe was shown 6 of 10 classes at L2 and 4-6 of 26-31 at L3, and on 10 of
#      15 ledger proposals every alias pair that existed among the surviving classes sat
#      outside that window. Under an argmax-latching DP the filter is adverse by construction:
#      the used rows are the winners and the starved rows are their aliases. The use share is
#      still logged per probed row, and so is what the 0.90 filter would have kept.
#   2. THE LEDGER IS DEFINED WHEN THE l+1 TABLE IS EMPTY under the current key: `e_keep` is the
#      same instances graded with no l+1 move applied, `e_merge` the same after the macro the
#      merged key builds; undefined only when neither key yields a table. `en_s3` refused all
#      four groups at c188 — where the probe had found 71 alias pairs at loss 0 — for want of
#      this branch. NOTE ON THE RECORD (gate M-7 carries it too): instances are sampled broken,
#      so the no-move `e_keep` is 1.0 exactly and in that case the licence reduces to "the
#      merged key builds a non-empty table"; `succ_differ` beside it says whether it repaired
#      anything.
#
# FOUR ARMS, sources before yokes. The forced ablation and its yoke are dropped: its role is
# established over two rounds (precision 0.000 against the demand partition, 0.533 in feature
# space; the ledger refuses it, the yield licence does not). `given_cat_tok` (en_s2) and `flat`
# (en_s0) are banked, not re-run. Same seed, ladder, caps, floors.
cd "$(dirname "$0")/../../../.."          # -> experiments/
export MODAL_PROFILE=chromatic
python3 rhm/practice/enharmonic/launch_detached.py --fn enharmonic_run --tag en_s4 \
    --eras "1:25:60,2:12:50,3:6:70,4:3:12,5:1:9" --era-caps "60,50,70,12,9" \
    --arms "endo_yield,flat_yk_endo_yield,endo_ledger,flat_yk_endo_ledger" \
    --tol-dsil 0.0046 --question-k 2048 \
    --max-macro-level 5 --gy-level 6 \
    --tol-yield-l5 0.2197 --tol-yield-l6 0.1836 \
    --tol-mass-l3 0.005367557424645102 --tol-mass-l4 0.004712453259301464 \
    --tol-mass-l5 0.0 --tol-mass-l6 0.0 \
    --quot-spell-cap 4 --slot-rec --entry-rec-cap 4096 \
    --merge-max-rows 64 --merge-use-frac 1.0 \
    --endo-price 267 --gate-frac 0.5 \
    --span-tau-fire 0.50 --perf-alpha 0.05 --perf-tau-w 0.20 \
    --budget 8 --g-budget 482 --n-corrupt 1 --mine-cap 8 --entry-rec \
    --span-min-hold 128 --collect-task-matched --tm-episodes 8192 \
    --recert-every 5 --n-aud 192 --probe-every 8 --n-rt 384 --n-score 256 \
    --sil-cv 0.15 --lp-min-drop 0.10 --seed 0
