#!/bin/sh
# [vo_s2] Q2 — THE CRITIC AT THE CLASS, AND EXPLORATION AT THE FRONTIER. A 2x2 in one tag on
# `en_s9`'s exact ladder and character, two knobs:
#
#        write \ chooser           the donor's DP/head argmax     the critic
#        argmax                    voi2_dp                       voi2_critic
#        frontier eps-greedy       voi2_xp_f                     voi2_critic_xp
#
# WHY THE CHOOSER IS A CRITIC AND NOT A LOSS ON THE HEAD. Q1's calibration arm put BCE(p_C, y)
# on the head's own NORMALISED emission distribution; the mass on the written class went to the
# base rate (0.12-0.26 against a filed solve rate of 0.175-0.225), the rest went off-table,
# parity fell to 0.20-0.33 against a 0.50 firing gate and the corridor never opened. A
# distribution over WHAT TO WRITE cannot also hold HOW LIKELY EACH CLASS IS TO SOLVE — the
# first must sum to one over the alphabet and the second must not — so a composed loss only
# weights the conflict. The verdict moves to a separate organ: a value head reading the pooled
# context and a candidate's level-1 FEATURES (content, never a label), emitting P(solve),
# trained by BCE against the verdict on filed writes keyed by the record. A class's value is the
# max over its held spellings. On a governed slot the head's imitation target becomes the RECORD,
# so parity is against the executor's own choice. `DESIGN.md` §19-§20.
#
# WHY EXPLORATION IS FRONTIER-ONLY. Q1's second stall: every era advance fired on the cap and the
# commit owner reads the era's own level, so L3's window was era 2 and closed at c110; the anchor
# committed at c100 and `voi_xp`, deviating on 9% of its L2 writes, missed it and never had an
# L4 or L5 slot at all. The ladder's commit windows are fragile to any early perturbation of the
# mining stream. So epsilon-greedy fires on the HIGHEST ADOPTED level only, in the priced
# practice beam only; every lower level, the metering beam and every audition stay at argmax, and
# a per-cycle assertion in preflight holds epsilon to the frontier. There is no temperature to
# size (Q1 §17's problem): the realised deviation IS epsilon, measured at 0.18-0.35 per cell
# against a nominal 0.30. `DESIGN.md` §21.
#
# Gates run before launch, all PASS:
#   G-F     `fidelity_smoke` retargeted at `enharmonic.py` -> 0.000e+00 both arms, commits equal
#   V-4     the critic BUILT AND TRAINED with governance off and the head's target back at
#           `dp_features` IS `voi2_pf_dp`, max|delta| 0.0 on every series, commits equal
#   V-4b    the same claim on CPU in seconds, at the parameter AND logged-value level, with its
#           own non-vacuity assertion and verified to FAIL on two deliberate perturbations
#   V-1     the record renders to what the beam wrote: 0 bad of 7,057 filed writes over five
#           preflight twins, asserting on every cycle
#   VO-8    rows filed, epsilon fired, epsilon never off the frontier (per cycle), the critic
#           governed, `unnamed` 0, and its held-out AUC computed
#   gates_cpu 15/15
#
# THE CRITIC'S PREFLIGHT AUC IS A PATH CHECK AND NOT A MEASUREMENT, said out loud: held-out sets
# of 16-26 rows at base rates 0.045-0.08 are one or two positives, so an AUC of 1.000 there is
# three correctly-ordered pairs. The first read worth interpreting is this tag's.
#
# Banked for comparison at reduction time:
#   en_s9:endo_ledger_open_ung5_ra   the arm all four fork (the identity check for voi2_dp)
#   vo_s1:voi_dp                     Q1's anchor, itself 0.000e+00 against en_s9
cd "$(dirname "$0")/../../../.."          # -> experiments/
export MODAL_PROFILE=chromatic
python3 rhm/practice/voicing/launch_detached.py --fn voicing_run --tag vo_s2 \
    --eras "1:25:60,2:12:50,3:6:70,4:3:12,5:1:9" --era-caps "60,50,70,12,9" \
    --arms "voi2_dp,voi2_critic,voi2_xp_f,voi2_critic_xp" \
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
    --vo-critic-lr 0.001 --vo-critic-min 256 --vo-critic-hold 0.1
