#!/bin/sh
# [vo_s3] Q3 — THE TWO ORGANS AT THE CHOICE, AND BABBLING OFF-STREAM. Four arms in one tag on
# `en_s9`'s exact ladder and character, two knobs:
#
#        choice \ critic's diet      filed writes            filed + probes
#        replace                     (banked vo_s2:voi2_critic)  voi3_rep_pr
#        composed                    voi3_comp                   voi3_comp_pr
#
#   plus voi3_dp, the anchor (no critic, no probe) — 0.000e+00 against en_s9 by construction.
#   No epsilon anywhere.
#
# WHY THE ORGANS ARE COMBINED AND NOT SWAPPED. Q2's critic REPLACING the DP reached L3 nineteen
# and L4 forty cycles earlier than the anchor while its class accuracy against the repair set at
# those cells FELL (`rep` 0.20-0.36 at 4n0-4n3 in era 4 against 0.26-0.60, level at 4n0). A
# chooser that discards the DP's ranking discards a prior that is right more often than the
# critic is. So on a governed slot the write is argmax_t z(dp_t/span) + w*z(critic_t) over the
# on-table candidates, w = 1, one knob, logged. The `/span` is the file's own convention (the
# same division that makes gate V-2b equal F.cross_entropy); the z-scores are what make w read a
# preference rather than a unit. `DESIGN.md` §24-§25.
#
# WHY EXPLORATION LEFT THE MINED BEAM. Q2's frontier-only epsilon still stalled the ladder,
# because the highest ADOPTED level is the level that feeds the level being EARNED: both eps
# arms stayed at L2 all run and the frontier never left it. Exploration is therefore a PRICED
# COUNTERFACTUAL PROBE whose rows reach the critic and nothing else — the trajectory's final
# configuration with a different on-table class substituted at the governed slot, rendered and
# graded once. `DESIGN.md` §26.
#
# ONE THING MEASURED AT THE SMOKE THAT THE DESIGN DID NOT EXPECT, stated here because it changes
# how [N] is read: `dp_vs_head = 0` on all 24,765 governed open-path calls — the DP's argmax and
# the head's own on-table argmax are the SAME ROW. Checked offline rather than assumed: on an
# untrained head they disagree 128/128, and after FIVE steps of the corridor's own objective
# (whose target IS dp_features) they agree 126/128 while the head's free emission is still
# off-table 128/128. So on an open slot there are not two priors to combine — one surface prior
# read two ways, plus the critic. The arm table and the knob are unchanged; what changes is that
# `moved` is the critic's WHOLE contribution. `DESIGN.md` §25.
#
# Gates run before launch, all PASS (`q3a` + `vo_gf8`):
#   G-F     `fidelity_smoke` retargeted at `enharmonic.py` -> 0.000e+00 both arms, control
#           0.000e+00, commits equal
#   V-1     0 bad of 7,863 filed writes over five twins, asserting every cycle, unnamed = 0
#   V-4A    the filed-write critic with governance off IS voi3_pf_dp: max|delta| 0.0 over 27
#           cycles, commits equal, 0 probes billed, bill residual 0.0
#   V-4B    the same with PROBES ON: max|delta| 0.0 on every series including g_per_solve, and
#           t_cum larger by EXACTLY 451 groundings x d_fb — bill residual 0.0
#   V-5     the probe-on twin's n_mined, miner state, solved pool and vocabulary are dp's bit
#           for bit; 451 rows filed into pbuf across 30 slots and nowhere else
#   V-4c    the composed scorer: w=0 IS the DP's argmax, affine-invariant, |C|=1 passes
#   V-4d    the probe: on-table, a DIFFERENT class, off-stream, billed row for row
#   VO-8    composed chooser ran on 24,765 calls; probes 451/342 graded = billed
#   gates_cpu 17/17
#   falsification 21/21 — every new gate was SHOWN TO FAIL on a deliberate perturbation of the
#           thing it protects, which is why V-4/V-5 were factored out of `preflight` into
#           `vo_gate_v4_v5` (a gate welded into a remote entrypoint can only be perturbed by
#           paying for the entrypoint). `DESIGN.md` §28.
#
# THE BILL, SIZED ON THE BANKED ANCHOR AND NOT ON THE PREFLIGHT. Projected onto
# vo_s2/voi2_critic's own per-cycle ledger at n_probe = 64 and its realised governed-slot count:
# 0.64% mean, 0.81% max, 0.46% of the whole run. The ~5% cap does NOT bind, by an order of
# magnitude. (The preflight's own 0.39%/1.14% at n_probe = 8 is dust and is not a measurement.)
#
# Banked for comparison at reduction time:
#   en_s9:endo_ledger_open_ung5_ra   the arm all four fork (the identity check for voi3_dp)
#   vo_s2:voi2_critic                the {replace, filed} cell, Q3's fourth 2x2 corner
#   vo_s1:voi_dp                     Q1's anchor, itself 0.000e+00 against en_s9
cd "$(dirname "$0")/../../../.."          # -> experiments/
export MODAL_PROFILE=chromatic
python3 rhm/practice/voicing/launch_detached.py --fn voicing_run --tag vo_s3 \
    --eras "1:25:60,2:12:50,3:6:70,4:3:12,5:1:9" --era-caps "60,50,70,12,9" \
    --arms "voi3_dp,voi3_comp,voi3_comp_pr,voi3_rep_pr" \
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
