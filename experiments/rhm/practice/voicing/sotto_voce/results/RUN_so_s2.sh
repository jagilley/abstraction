#!/bin/sh
# [so_s2] WHO GRADES THE BABBLE — seed 2, the second draw whose ladder reached the frontier. Three arms in one tag, all the composed chooser,
# all with the probe channel ON, all CLOCK-YOKED to the banked SEED-2 anchor `vo_s3d2:voi3_dp`.
# The one knob that moves is `vo_om_mode`, the grader of record for the probe:
#
#        so_mg_yk   "model"      ONE outcome model answers; nothing is billed
#        so_cg_yk   "committee"  K=5 models answer, and only where they AGREE is the row
#                                filed at all; nothing is billed
#        so_hy_yk   "hybrid"     as committee, and where they disagree the WORLD is paid for
#                                the verdict — which is also the only probe verdict the
#                                outcome models are ever allowed to train on
#
# THE FLOOR AND THE CEILING ARE BANKED AND NOT RE-RUN:
#   vo_s3e:voi3b_comp_yk      composed chooser, FILED diet     (the floor, seed 2)
#   vo_s3e:voi3b_comp_pr_yk   composed chooser, WORLD-graded probes (the ceiling, seed 2)
#   vo_s3d2:voi3_dp           the seed-2 anchor and the yoke SOURCE
# All three are read at reduction time with `--bank`; `--yoke-from-tag vo_s3d2` resolves the
# source out of the PARENT node's volume dir (`YOKE_REMOTE`), so the arm this tag replays is
# byte-identical to the one `vo_s3e` replayed.
#
# WHAT SEED 2 CAN AND CANNOT DO, said here so the reduction is not read as more than it is:
# the seed-2 anchor commits L2@c59 / L3@c77 / L4@c157 and NEVER L5, so this tag has L4 cells
# and no L5 cells. `voicing` Q3e found the L2/L3 advantage replicates across every seed that
# produced those rungs and the L4/L5 frontier advantage does NOT (+16.1% at seed 0, -1.7% at
# seed 2). The replicated cell here is therefore L2/L3; the frontier is a second draw, not a
# replication.
#
# WHY THIS QUESTION. `voicing` Q3/Q3b: a critic fed only what the learner chose to write ranks
# counterfactual rows at CHANCE (0.491), and paying the world for substituted verdicts lifts it
# to 0.798 for ~0.55% of priced time. There are exactly three ways to get a label for a road not
# taken: vary the behaviour (Q1: 9% write variation at L2 costs the L3 commit window), pay the
# world, or ask a MODEL of outcomes trained on experience. This tag asks whether the third
# works and WHERE IT FAILS — and RHM is where that is measurable, because the world's verdict is
# exact and cheap, so it can be computed for every model-graded probe as an instrument that no
# arm consumes.
#
# THE MIRROR. A small classifier from a rendered configuration and its root to P(solved) —
# what the world's grader reads — trained by BCE on the learner's own experienced sentences and
# their world verdicts (the filed rows' `fin` and `y`, deduplicated by (configuration, root)).
# A separate object throughout: its own parameters, its own optimizer (NOT the plant's), its own
# numpy streams, every forward and backward sandboxed against the shared torch stream. It is
# NOT the surface model's NLL and NOT the DP's score — the composed chooser already uses the
# surface model as its prior and a critic trained on the prior's opinion would learn the prior.
#
# SIZED OFFLINE FIRST (`q0_sotto.py`, CPU, `figures/so_q0_reduction.txt`). On a proxy corpus
# built from the same world with the same corruption operator and the same substitution:
# held-out EXPERIENCE AUC 0.935 (K=5), COUNTERFACTUAL AUC 0.747, and the agreement rule at the
# q-th quantile of the held-out spread files
#     q=0.25 -> 20% of probes at accuracy 0.998 | q=0.50 -> 47% at 0.924 | q=1.0 -> 100% at 0.853
# with the dropped rows at 0.75-0.82. **q = 0.50 is the setting of record**: q=0.75 drops only
# ~22% (barely distinguishable from the model-graded arm) and q=0.25 starves the critic. The
# proxy's support is WIDER than the learner's, so these are optimistic on the counterfactual
# half; the run's own numbers are in `log["vo_om"]` and `vo_mirror`.
#
# NO CODE CHANGED between `so_s1` and this tag and G-F is not re-run; the gate table below is
# `so_s1`'s, and what this tag checks for itself is Y-1 against the NEW source (the arms must
# replay 59/77/157 and 60/110/180/192/201, any CANCELLED commit reported as the book finding it
# is), V-1 on every filed write, and M-1r/M-2r/M-3r/M-4r on its own arm files.
#
# Gates run before `so_s1`, all PASS (`_preflight_s1` + `so_gf1`):
#   G-F     0.000e+00 both arms, control 0.000e+00, commits equal. MANDATORY: the probe path,
#           the recorder and the critic audit are all shared paths this node touched.
#   Y-1     the replay MATCHES on all three mirror twins and the five Q3b twins, 0 cancelled
#   V-1     0 bad filed writes, asserting every cycle
#   V-4A/B  governance off, yoked: 0.0 on every series; with probes on, t_cum larger by
#           EXACTLY the probe count
#   V-5     off-stream on the objects
#   V-6b    the diet CHANGES THE RUN with governance on — and its mirror form, that a
#           MODEL-graded diet changes the run against the world-graded twin
#   M-1     the world's verdict exists for every probe, enters no loss (measured: the critic is
#           the same to the bit whether the world buffer contradicts the filed verdicts or
#           agrees with them), no buffer the critic's loss reads, and no bill
#   M-2/M-2b  the outcome models train on experience only — plus WORLD-graded disagreeing
#           probes on the hybrid alone — and every training row carries the world's own verdict
#           on its own configuration (re-graded, 0 mismatches)
#   M-3     the bill is exactly the world-graded probe count: 0 / 0 / the disagreeing fraction
#   M-4     the agreement rule is what files
#   M-5     the mirror is LIVE (its step moves its members, the members differ, the threshold is
#           finite, the shared torch stream never moves)
#   gates_cpu 24/24 | falsification harness (gates/falsify.py) 56/56
#
# SAID OUT LOUD: the filed target on a model-graded row is the mirror's PROBABILITY, not a
# verdict thresholded at 0.5. The critic's loss takes a soft target, and at the probe base rate
# the world measures a 0.5 threshold throws away almost everything the mirror knows (a constant
# "not solved" scores higher ACCURACY than the mirror does while the mirror's ranking is far
# above chance). The thresholded verdict is kept as the READOUT. The hard-verdict variant is
# untested; `DESIGN.md` records it.
cd "$(dirname "$0")/../../../../.."        # -> experiments/
export MODAL_PROFILE=chromatic
python3 rhm/practice/voicing/sotto_voce/launch_detached.py --fn voicing_run --tag so_s2 \
    --eras "1:25:60,2:12:50,3:6:70,4:3:12,5:1:9" --era-caps "60,50,70,12,9" \
    --arms "so_mg_yk,so_cg_yk,so_hy_yk" \
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
    --vo-om-k 5 --vo-om-dim 64 --vo-om-lr 0.001 --vo-om-steps 16 --vo-om-batch 256 \
    --vo-om-cap 30000 --vo-om-min 512 --vo-om-boot 0.8 --vo-om-hold 0.1 \
    --vo-om-agree-q 0.5 --vo-om-freq-share 0.05 --vo-om-sample-cap 4000
