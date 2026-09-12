#!/bin/bash
# [embouchure/Q2.2] the pair's launcher. ONE line, two knob values, so the only difference
# between em_q2b_l and em_q2b on disk is `--reader-target` (and, with --smoke, the scale).
#   usage: launch_q22.sh <tag> <listener|both> [--smoke]
set -e
TAG=$1; RT=$2; SMOKE=$3
HERE=/home/user/research/experiments/rhm/practice/embouchure
cd /home/user/research/experiments
COMMON=(--arms "canon_s,given_rule_s,fit_rule_s,own_scalar_rec_s,own_readback_s"
        --rule E_R8 --n-ctx 8 --practiced "0,3,7"
        --rule-seed 6 --setup-render rule --reader-steps 6000
        --lexicon-merge "0:1=1:1,2:0=4:0,5:0=6:0" --own-intent record
        --reader-target "$RT"
        --max-macro-level 3 --budget 8 --g-budget 482
        --tol-dsil 0.0046 --endo-price 267 --gate-frac 0.5
        --span-tau-fire 0.50 --perf-alpha 0.05 --perf-tau-w 0.20
        --n-corrupt 1 --mine-cap 8 --gy-level 4 --entry-rec
        --span-min-hold 128 --collect-task-matched
        --recert-every 5 --probe-every 8
        --sil-cv 0.15 --lp-min-drop 0.10 --seed 0)
if [ "$SMOKE" = "--smoke" ]; then
  SCALE=(--quick --quick-cycles 20 --transfer-n 16
         --eras "1:25,2:12,3:6" --era-caps "20,20,20"
         --question-k 512 --tm-episodes 2048 --n-aud 64 --n-rt 128 --n-score 128)
else
  SCALE=(--transfer-n 64
         --eras "1:25:40,2:12:45,3:6:55" --era-caps "40,45,55"
         --question-k 2048 --tm-episodes 8192 --n-aud 192 --n-rt 384 --n-score 256)
fi
LOG="$HERE/results/launch_${TAG}.log"
rm -f "$LOG"
MODAL_PROFILE=chromatic nohup setsid /usr/local/bin/modal run --detach \
  rhm/practice/embouchure/embouchure.py::embouchure_run --tag "$TAG" \
  "${COMMON[@]}" "${SCALE[@]}" > "$LOG" 2>&1 < /dev/null &
disown
echo "[launch] $TAG reader_target=$RT ${SMOKE:-full} -> $LOG"
