#!/bin/sh
# [vo_s3d] Q3d — THE SEED CHECK ON `en_s9`'s OWN L5 COMMIT. The anchor character ALONE
# (`voi3_dp`, self-paced, otherwise en_s9's script) at seed 2 and seed 3. No yoked arms.
#
# WHY. Q3c (`DESIGN.md` §36): at seed 1 the anchor commits L2@c47 and nothing else, runs 108
# cycles, and never certifies L3 — era 2 was left at c77 by `quiet` at 17 of a 50 cap and era 3
# at c87 by `quiet` at 10 of 70, while the commit owner (reading `yield`) never licensed. So the
# seed check on THIS node's frontier result has become a seed check on `en_s9`'s own L5 commit,
# which both this node and the enharmonic addendum lean on. Two more draws of the same character
# say whether seed 0's climb or seed 1's stall is the typical one.
#
# TWO LAUNCHES, NOT ONE TAG WITH PER-ARM `seed=` OVERRIDES, and the reason is a confound.
# `parse_arms` does support `voi3_dp:seed=2`, and `run_arm` would apply it to every per-arm
# stream. But `shared` is built ONCE per job, before the arm loop, and inside `_spiral_shared`
# the run-level `cfg["seed"]` is what draws `probe_clean` (the fixed clean probe pool behind the
# endo read and the plant probe). A per-arm override would therefore hold that pool fixed while
# moving everything else, whereas seeds 0 and 1 moved BOTH. Two launches keep seeds 2 and 3
# exactly comparable to the two already banked; the duplicated setup is ~510 s of net training
# that is seed-independent anyway (`rule_seed` 0, `train_seed` 1, unchanged throughout).
#
# NO CODE CHANGED for this round and G-F is NOT run: `voicing.py` is untouched since the Q3b
# landing. What the reduction reports is [A] plus, per seed, the commit and advance record with
# the REASON for each advance (`loop_actions[].why`: cap vs quiet), the certificate cycles, the
# L3 yield read through era 2, and the ladder reached — seeds 0, 1, 2, 3 in one table.
#
# Banked beside at reduction time:
#   vo_s3:voi3_dp    seed 0 — L2@48, L3@100, L4@151, L5@186, 201 cycles
#   vo_s3c:voi3_dp   seed 1 — L2@47 only, 108 cycles, cert L2@c17 and L3 never
cd "$(dirname "$0")/../../../.."          # -> experiments/
export MODAL_PROFILE=chromatic
for SEED in 2 3; do
python3 rhm/practice/voicing/launch_detached.py --fn voicing_run --tag "vo_s3d${SEED}" \
    --eras "1:25:60,2:12:50,3:6:70,4:3:12,5:1:9" --era-caps "60,50,70,12,9" \
    --arms "voi3_dp" \
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
    --sil-cv 0.15 --lp-min-drop 0.10 --seed "${SEED}" \
    --vo-rec-cap 8192 --vo-rec-batch 64 --vo-chunk 32 \
    --vo-readback-cell 96 --vo-rep-n 64 --vo-verify-cycles 8 \
    --vo-critic-lr 0.001 --vo-critic-min 256 --vo-critic-hold 0.1
sleep 5
done
