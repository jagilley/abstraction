#!/bin/sh
# [enharmonic] Q1's main run. Every flag but --arms, --max-macro-level, --gy-level and the two
# new --tol-yield-l5/l6 is `tu_s0`'s, verbatim — which is what makes `flat` the mirror arm
# (`tu_m_exo`) with the quotient absent and the rung opened, and nothing else.
#
# THE TWO NEW FLOORS are NOT defaults. `gy_level` is 6 here, so the mirror's commit owner reads
# L5 while earning L4 and L6 while earning L5; `MEASURED_FLOORS` carries only a FLAT-keyed
# derivation for those levels and no banked run can be re-keyed offline (the L5/L6 key streams
# are never logged). So they are measured on this node's own smoke tag by
# `floors_l5l6.py` — `tol_dsil`'s idiom — and pasted here.
#
#   floors from: en_smoke, TREATED ARMS ONLY (given_cat_tok + given_cat_min, 76 windows).
#   `flat`s L5/L6 series is identically 0, so it contributes only zeros to sd(N) and deflates
#   the pooled floor -- a floor is the noise scale of a LIVE series, which is the same reason
#   MEASURED_FLOORS records for the L3 floor moving 0.2368 -> 0.4613. Calibration on the same
#   tag and statistic: L3/L4 come out at 0.53x / 0.45x of the banked full-config floors on the
#   treated arms and 0.86x / 0.51x on `flat`. A thermostat replay on the smokes own series is
#   floor-INSENSITIVE over 3.5x (0.0639 -> 0.2197 at L5 gives the same firings), so the choice
#   moves timing and not the decision. See floors_en_smoke*.json.
TOL_L5=0.2197
TOL_L6=0.1836

cd "$(dirname "$0")/../../../.."          # -> experiments/
export MODAL_PROFILE=chromatic
python3 rhm/practice/enharmonic/launch_detached.py --fn enharmonic_run --tag en_s0 \
    --eras "1:25:60,2:12:50,3:6:70,4:3:12,5:1:9" --era-caps "60,50,70,12,9" \
    --arms "flat,given_cat_tok,given_cat_min" \
    --tol-dsil 0.0046 --question-k 2048 \
    --max-macro-level 5 --gy-level 6 \
    --tol-yield-l5 "$TOL_L5" --tol-yield-l6 "$TOL_L6" \
    --quot-spell-cap 4 --slot-rec \
    --endo-price 267 --gate-frac 0.5 \
    --span-tau-fire 0.50 --perf-alpha 0.05 --perf-tau-w 0.20 \
    --budget 8 --g-budget 482 --n-corrupt 1 --mine-cap 8 --entry-rec \
    --span-min-hold 128 --collect-task-matched --tm-episodes 8192 \
    --recert-every 5 --n-aud 192 --probe-every 8 --n-rt 384 --n-score 256 \
    --sil-cv 0.15 --lp-min-drop 0.10 --seed 0
