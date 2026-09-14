#!/bin/sh
# [vo_s3b] Q3b — THE DIET AXIS RE-RUN, WITH THE LADDER HELD FIXED. Four arms in one tag, all
# CLOCK-YOKED to the anchor, two knobs:
#
#        choice \ critic's diet     filed writes       filed + probes
#        replace                    voi3b_rep_yk       voi3b_rep_pr_yk
#        composed                   voi3b_comp_yk      voi3b_comp_pr_yk
#
#   The anchor is banked twice: `vo_s3:voi3_dp` and `en_s9:endo_ledger_open_ung5_ra`, which are
#   the same arm at 0.000e+00 on all thirteen series. `dp` is NOT re-run.
#
# WHY THE LADDER IS YOKED. Q3 §31: no treated arm except Q2's replace-critic has ever been READ
# at the levels this node is about. The anchor committed L3 at c100, Q2's replace-critic at c81,
# and the composed arm never -- its L3 yield read rose EARLIER than the anchor's (0.36 vs 0.32 at
# c70), then flattened near 0.43 and never satisfied the latch before era 2's cap closed at c110.
# Each is the commit latch responding to that arm's own L2 write distribution, which is Q1 §19's
# window seen through a third distribution. So every treated arm replays the anchor's realised
# actions -- commits 48/100/151/186, advances 60/110/180/192/201 -- and the chooser is read at L4
# and L5 with the same slots open at the same cycles. `DESIGN.md` §32.
#
# WHY THE DIET AXIS IS BEING RE-RUN AT ALL. Defect #8 (`DESIGN.md` §30): `vo_critic_terms`'
# `use_probe` keyword was never passed at the call site, so 198,382 priced probes in vo_s3 were
# drawn, rendered, graded, billed and filed -- and both probe arms came out BIT-IDENTICAL to
# their no-probe twins on every behaviour series, with only t_cum moved and a bill residual of
# exactly 0.0. Every inertness gate passed, because all of them assert with governance OFF where
# a diet cannot reach the run whether it is wired or not.
#
# Gates run before launch, all PASS (`q3b` + `vo_gf9`):
#   G-F     0.000e+00 both arms, control 0.000e+00, commits equal. MANDATORY this round: the
#           defect-#8 fix touched `finetune_generator_span`, a shared path.
#   Y-1     the replay MATCHES on all five yoked twins -- commits [6], advances [6,12,17,22,27],
#           0 cancelled, against a 6-action plan
#   V-1     0 bad of 3,267 filed writes over five twins, asserting every cycle, unnamed = 0
#   V-4A/B  governance off, yoked: max|delta| 0.0 with commits equal; with probes on, 0.0 on
#           every series and t_cum larger by EXACTLY 329 groundings -- bill residual 0.0
#   V-5     the probe-on twin's n_mined, miner state, solved pool and vocabulary are dp's bit
#           for bit; 329 rows into pbuf across 16 slots and nowhere else
#   V-6b    THE GATE vo_s3 DID NOT HAVE: with governance ON the diet CHANGES THE RUN --
#           max|delta| 1.0 on n_solved, first differing at cycle 9, both arms governed, 116
#           probes and 116 rows filed. Shown to fail on defect #8 itself.
#   V-6     the CPU half: d_critic 5.31e-01, 117 -> 234 training rows
#   gates_cpu 18/18 | falsification harness (gates/falsify.py) 36/36
#
# TWO PREFLIGHT-SCALE ARTIFACTS, said out loud and NOT reported as measurements: the critic's
# held-out AUC has an EMPTY denominator at q3b (reads = 0 -- the yoked twins commit once, so no
# slot reaches the audit's 32-row / 16-held-out minimum; the path's coverage is q3a's 9-14 reads
# and vo_s3's 813 held-out probe rows per slot, and `vo_critic_audit` did not change between
# them), and `moved`/`dp_vs_head` read 1.000/0 for the same reason they did at q3a -- a
# forty-step head against a near-degenerate DP ranking. At full scale vo_s3 read 0.330 and
# 0.084-0.160. `DESIGN.md` §34.
#
# Banked for comparison at reduction time:
#   vo_s3:voi3_dp        the yoke SOURCE and the anchor (== en_s9 at 0.000e+00)
#   vo_s2:voi2_critic    the self-paced {replace, filed} cell, for the clock contrast
#   vo_s3:voi3_comp      the self-paced {composed, filed} cell, which never left L2
cd "$(dirname "$0")/../../../.."          # -> experiments/
export MODAL_PROFILE=chromatic
python3 rhm/practice/voicing/launch_detached.py --fn voicing_run --tag vo_s3b \
    --eras "1:25:60,2:12:50,3:6:70,4:3:12,5:1:9" --era-caps "60,50,70,12,9" \
    --arms "voi3b_comp_yk,voi3b_comp_pr_yk,voi3b_rep_yk,voi3b_rep_pr_yk" \
    --yoke-from-tag vo_s3 \
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
    --sil-cv 0.15 --lp-min-drop 0.10 --seed 0 \
    --vo-rec-cap 8192 --vo-rec-batch 64 --vo-chunk 32 \
    --vo-readback-cell 96 --vo-rep-n 64 --vo-verify-cycles 8 \
    --vo-critic-lr 0.001 --vo-critic-min 256 --vo-critic-hold 0.1 \
    --vo-w 1.0 --vo-probe-n 64
