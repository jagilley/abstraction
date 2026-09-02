#!/bin/sh
# [tutti] the main run. Every flag but --arms, --tol-dsil and --question-k is `ca_s0`'s,
# verbatim — which is what makes `tu_y_exo` reproduce `ca_s0/dsil_yield` bit for bit.
cd "$(dirname "$0")/../../../.."          # -> experiments/
export MODAL_PROFILE=chromatic
python3 rhm/practice/tutti/launch_detached.py --fn tutti_run --tag tu_s0 \
    --eras "1:25:60,2:12:50,3:6:70,4:3:12,5:1:9" --era-caps "60,50,70,12,9" \
    --arms "tu_y_exo,tu_d_exo,tu_s_exo,tu_s_endo,tu_s_yk,tu_d_endo,tu_y_endo,tu_m_exo" \
    --tol-dsil 0.0046 --question-k 2048 \
    --max-macro-level 4 --endo-price 267 --gate-frac 0.5 \
    --span-tau-fire 0.50 --perf-alpha 0.05 --perf-tau-w 0.20 \
    --budget 8 --g-budget 482 --n-corrupt 1 --mine-cap 8 --gy-level 4 --entry-rec \
    --span-min-hold 128 --collect-task-matched --tm-episodes 8192 \
    --recert-every 5 --n-aud 192 --probe-every 8 --n-rt 384 --n-score 256 \
    --sil-cv 0.15 --lp-min-drop 0.10 --seed 0
