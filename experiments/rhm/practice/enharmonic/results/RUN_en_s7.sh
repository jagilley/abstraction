#!/bin/sh
# [en_s7] THE ROW CAP RAISED. `en_s6`'s flags verbatim except `--merge-max-rows 256`, which is
# the one thing that bound: the probe takes one representative per class and then caps, and
# `en_s6`'s L4 books ran 150-218 classes while the cap was 64. So the L4 partition could never
# actually be taken — the probe saw under half of it. 256 covers every book either arm reached.
# `merge_n_probe` is UNCHANGED at 256 instances; the priced share is the thing to watch (at 218
# rows x 256 instances a proposal grades ~56k, and the reduction reports the realised share).
#
# THE INSTRUMENT gains its second reference: for each audition instance, the set of features
# that REPAIR it at the node (`fourwall.consistent_features`), so `contains_rep` means "the
# chosen class holds a repairing feature" beside `contains`, which is against the clean latent.
# It is a PER-ERA PRECOMPUTE — the audition pool is fixed per era, so it costs `v * n_score`
# gradings once per (era, cell), not one grader call per macro call — and it is counted in the
# instrument's own oracle bill, never in `counts["ground"]`.
#
# ARMS, sources before yokes — and the order is load-bearing. `endo_ledger_open_ung5_yk` runs
# FIRST so its plan is read from the BANKED `en_s4:endo_ledger` (L2@c48, L3@c97, L4@c179), the
# same clock `en_s6` put it on; running the in-tag `endo_ledger` first would hand it a plan
# that the raised cap may itself have moved. `endo_ledger` then runs with the same cap — the
# closed arm with a full probe, whose cap bound once in `en_s4` at c188 (100 classes -> 64).
# Its banked flat yoke `en_s4:flat_yk_endo_ledger` serves as the comparator IF the reduction
# confirms its plan still matches; if it does not, the reduction says so rather than yoking.
cd "$(dirname "$0")/../../../.."          # -> experiments/
export MODAL_PROFILE=chromatic
python3 rhm/practice/enharmonic/launch_detached.py --fn enharmonic_run --tag en_s7 \
    --eras "1:25:60,2:12:50,3:6:70,4:3:12,5:1:9" --era-caps "60,50,70,12,9" \
    --arms "endo_ledger_open_ung5_yk,endo_ledger" \
    --yoke-from-tag en_s4 \
    --tol-dsil 0.0046 --question-k 2048 \
    --max-macro-level 5 --gy-level 6 \
    --tol-yield-l5 0.2197 --tol-yield-l6 0.1836 \
    --tol-mass-l3 0.005367557424645102 --tol-mass-l4 0.004712453259301464 \
    --tol-mass-l5 0.0 --tol-mass-l6 0.0 \
    --merge-gauge expected \
    --quot-spell-cap 4 --slot-rec --entry-rec-cap 4096 \
    --merge-max-rows 256 --merge-use-frac 1.0 \
    --endo-price 267 --gate-frac 0.5 \
    --span-tau-fire 0.50 --perf-alpha 0.05 --perf-tau-w 0.20 \
    --budget 8 --g-budget 482 --n-corrupt 1 --mine-cap 8 --entry-rec \
    --span-min-hold 128 --collect-task-matched --tm-episodes 8192 \
    --recert-every 5 --n-aud 192 --probe-every 8 --n-rt 384 --n-score 256 \
    --sil-cv 0.15 --lp-min-drop 0.10 --seed 0
