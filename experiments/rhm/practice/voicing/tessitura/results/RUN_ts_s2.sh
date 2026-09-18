#!/bin/sh
# [ts_s2] THE SECOND SEED OF THE PAID ARM — authorised, and the reason is the SHAPE of §8's two
# headlines rather than their size.
#
# `ts_s0` produced two claims that are RATE-LIKE: the norm's convergence onto a rising world
# (a per-era signed gap) and the fixed panels' drift on identical rows (a per-era level). A
# rate-like claim is a statement about a trajectory, and a trajectory is exactly what a seed
# moves; `voicing` Q3c-e is the lineage's own demonstration that its frontier magnitudes move
# 3-4x between seeds with the arms swapping order. The STATE-like claims read on identical rows
# (the norm's per-slot calibration, the twin contrast, the cost/structure conditioning) already
# reproduce at two seeds in `tessitura/results/rows.txt` and are not what this arm is for.
#
# WHAT IS ASKED OF IT: each of §8's headlines restated as a SIGN that either agrees across the
# two seeds or does not. Not an average — this lineage never averages seeds.
#
# THE IDIOM IS `overtone/results/RUN_ov_s2.sh`'s, verbatim: `--seed 2` and
# `--yoke-from-tag vo_s3d2`, with the anchor NOT re-run — `vo_s3d2:voi3_dp` is banked and the
# yoke reads its plan off the volume. Seed 2's realised ladder is L2@59, L3@77, L4@157 with
# advances 60/110/180/192/201, and it does NOT reach L5, so the era-5 columns will be thinner
# than seed 0's and that is a property of the ladder, not of this arm.
# No `--ref-tag`: `vo_s3e` did not use one at this seed and the metering draws move with
# `--seed`, so an assertion on `refs` is not the free check it is within a seed.
#
# THE ARM IS `ovt_comp_pr_sh`, the SAME arm as `ts_s0` — the uniform-probe one — and not
# `ov_s2`'s `ovt_comp_pr_dis`. That matters twice. It keeps the seed the only thing that moves
# between `ts_s0` and this run; and it gives gate TS-1 a banked twin at this seed
# (`vo_s3e:voi3b_comp_pr_yk` is `ovt_comp_pr_sh` without the instruments), so the bit-identity
# is free here as it was at seed 0.
#
# MEMORY: `--memory` is not a flag; `voicing_run`'s decorator now reads `VO_MEMORY_MB = 16384`
# (`voicing.py` ~12771), cut from the donor's inherited 32768 on the strength of `ts_s0`'s
# measured peak of 7048 MB. Modal bills the greater of the request and the use.
cd "$(dirname "$0")/../../../../.."        # -> experiments/
export MODAL_PROFILE=chromatic
python3 rhm/practice/voicing/launch_detached.py --fn voicing_run --tag ts_s2 \
    --arms "ovt_comp_pr_sh" \
    --eras "1:25:60,2:12:50,3:6:70,4:3:12,5:1:9" --era-caps "60,50,70,12,9" \
    --yoke-from-tag vo_s3d2 \
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
    --sil-cv 0.15 --lp-min-drop 0.10 --seed 2 \
    --vo-rec-cap 8192 --vo-rec-batch 64 --vo-chunk 32 \
    --vo-readback-cell 96 --vo-rep-n 64 --vo-verify-cycles 8 \
    --vo-critic-lr 0.001 --vo-critic-min 256 --vo-critic-hold 0.1 \
    --vo-w 1.0 --vo-probe-n 64 \
    --ov-shadow "lin,dir" --ov-free --ov-comb-folds 2 \
    --ov-dump --ov-probe-unif-frac 0.25 \
    --ts-norm --ts-panel 128 --ts-struct 64

# Afterwards, from experiments/:
#   python3 rhm/practice/voicing/fetch_compact.py --tag ts_s2 --fetch --replace
#   python3 rhm/practice/voicing/analyze_voicing.py --tag ts_s2 --yoke-src vo_s3d2:voi3_dp \
#       --bank vo_s3e:voi3b_comp_pr_yk \
#       --out rhm/practice/voicing/tessitura/results/ts_s2_reduction.txt      # gate TS-1 is [R]
#   python3 rhm/practice/voicing/tessitura/reduce_rerun.py --tag ts_s2
#   python3 rhm/practice/voicing/tessitura/reduce_rerun.py --seeds ts_s0,ts_s2   # the sign table
