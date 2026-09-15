#!/bin/sh
# [ov_s2] THE S3 ARM AT SEED 2 — one arm, authorised.
#
# WHY THIS ONE AND ONLY THIS ONE. The S3 gain at seed 0 is largest in eras 4 and 5, and that is
# exactly where `voicing` Q3c-e found the measurement unstable: the era-5 magnitude moved 3-4x
# between seeds with the two arms swapping order, and 5n0 already went the other way inside
# seed 0 itself (`dis` 0.334 against the uniform twin's 0.505). The L2/L3 cells are the ones
# that replicated at three seeds. So seed 2 is read at L2/L3 pooled and the four L4 cells --
# seed 2's ladder reaches L4 and not L5 -- plus era-3 and era-5 error at the matched clock.
#
# THE IDIOM IS `results/RUN_vo_s3e.sh`'s, verbatim: `--seed 2` and `--yoke-from-tag vo_s3d2`,
# with the anchor NOT re-run -- `vo_s3d2:voi3_dp` is banked and the yoke reads its plan off the
# volume. Seed 2's realised ladder is L2@59, L3@77, L4@157 with advances 60/110/180/192/201.
# No `--ref-tag`: `vo_s3e` did not use one at this seed and the metering draws move with
# `--seed`, so an assertion on `refs` is not the free check it is within a seed.
#
# It carries the round's instruments (`ov_shadow`, `ov_free`, `ov_dump`) because gate R-1 has
# now shown at FULL SCALE that they move nothing -- `ov_s0:ovt_comp_pr_sh` is banked
# `vo_s3b:voi3b_comp_pr_yk` at 0.000e+00 on all eleven series -- so S1, S2, the row dump and
# the banked heads come free at the second seed.
#
# Reduced against:
#   vo_s3e:voi3b_comp_pr_yk   seed 2, THE UNIFORM TWIN -- the same arm with §26's draw
#   vo_s3d2:voi3_dp           seed 2, the anchor and the yoke source
#   vo_s3b:voi3b_comp_pr_yk   seed 0, the same twin, so the two seeds sit in one table
cd "$(dirname "$0")/../../../../.."        # -> experiments/
export MODAL_PROFILE=chromatic
python3 rhm/practice/voicing/launch_detached.py --fn voicing_run --tag ov_s2 \
    --arms "ovt_comp_pr_dis" \
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
    --ov-dump --ov-probe-unif-frac 0.25

# Afterwards, from experiments/:
#   python3 rhm/practice/voicing/fetch_compact.py --tag ov_s2 --fetch --replace
#   python3 rhm/practice/voicing/analyze_voicing.py --tag ov_s2 --yoke-src vo_s3d2:voi3_dp \
#       --bank vo_s3d2:voi3_dp,vo_s3e:voi3b_comp_pr_yk,vo_s3b:voi3b_comp_pr_yk \
#       --out rhm/practice/voicing/overtone/figures/ov_s2_reduction.txt
