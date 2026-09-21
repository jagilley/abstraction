#!/bin/sh
# [ts_s0] THE ONE PAID ARM — `overtone/results/RUN_ov_s0b.sh`'s arm, clock-yoked, re-run with
# three knob-gated, default-off instruments added.
#
# WHY IT EXISTS. The zero-cost pass (`tessitura/results/logged.txt`) establishes that the
# judge's own LEVEL is nowhere in the banked record: `vo_critic_audit` logs a rank statistic and
# the base rate, and `vo_critic_terms`' per-slot `cstat` is discarded at the call site
# (`voicing.py` ~line 2248). The CPU pass (`tessitura/results/rows.txt`) recovers the level from
# the two arms that banked `vo_heads.pt`, but only with the FINAL critic on the ring-buffer TAIL
# — one reader, at one time, on ~60 filed cycles. The within-subject question the era ladder
# makes available, and which `logit_reading/striatum/norm/`'s frozen trunk could not ask at all,
# needs the level logged AS IT IS TRAINED. So does the structural label: gate T-4 in
# `tessitura/structure.py` shows the root is not recoverable from the dump and the
# reconstruction that tries is biased upward on 25 of 30 slots.
#
# WHAT IS ADDED, all three RUN-LEVEL and all three default OFF:
#   --ts-norm        the judge's mean predicted P(solve) per slot per cycle, beside the base
#                    rate already logged: at the candidate the row carries, and max / mean over
#                    the operative table at the same contexts (the three ways a state value can
#                    be derived from a Q, since the judge is Q-shaped and a norm is not).
#   --ts-panel 128   a FIXED panel per (slot, era), frozen once and scored by the LIVE critic
#                    every cycle after — norm §1's identical-rows protocol, which is the only
#                    form in which "the reader drifted" is separable from "the world moved".
#   --ts-struct 64   the STRUCTURAL label per row under the true root
#                    (`fourwall.consistent_features`, the `rep` instrument's own call), on 64
#                    filed and 64 probe rows per slot per cycle, scored against the judge and
#                    the prior with each label held fixed in turn — striatum §2's form.
#
# THE ARM ITSELF DOES NOT MOVE. All three are instruments: no gradient, no decision, the RNG
# sandboxed, the oracle reads counted in `_EXP_REC["reads"]` as `rep`'s are and never in
# `counts["ground"]`. So this arm must come out BIT-IDENTICAL to banked
# `ov_s0b:ovt_comp_pr_sh` (which is itself banked `vo_s3b:voi3b_comp_pr_yk`) on every behaviour
# series including `t_cum` — GATE TS-1, read for free by `analyze_voicing.py`'s section [R].
# The only counter that may move is `exp_reads`, the unpriced instrument tally, and it must.
#
# THE FLAG LINE BELOW IS `RUN_ov_s0b.sh`'s VERBATIM plus the three flags. Nothing else differs,
# including `--seed 0`, `--yoke-from-tag vo_s3` and `--ref-tag vo_s3b`.
#
# Offline first (three tables and three harnesses), then this:
#   PYTHONPATH=. python3 -c "from rhm.practice.voicing import voicing as V; \
#       V.vo_gates_cpu(); V.ov_gates_cpu(); V.ts_gates_cpu()"
#   PYTHONPATH=. python3 rhm/practice/voicing/gates/falsify.py                 # 36/36
#   PYTHONPATH=. python3 rhm/practice/voicing/overtone/gates/falsify_ov.py     # 26/26
#   PYTHONPATH=. python3 rhm/practice/voicing/tessitura/gates/falsify_ts.py    # 11/12 + 1 named
# and on Modal:
#   python3 rhm/practice/voicing/launch_detached.py --fn fidelity_smoke --tag ts_gf1   # G-F
#   python3 rhm/practice/voicing/launch_detached.py --fn preflight --outdir-tag ts_pf1 \
#       --arms "voi3b_pf_src,ovt_pf_comp_pr" --ov-shadow "lin,dir" --ov-free --ov-dump
cd "$(dirname "$0")/../../../../.."        # -> experiments/
export MODAL_PROFILE=chromatic
python3 rhm/practice/voicing/launch_detached.py --fn voicing_run --tag ts_s0 \
    --arms "ovt_comp_pr_sh" \
    --eras "1:25:60,2:12:50,3:6:70,4:3:12,5:1:9" --era-caps "60,50,70,12,9" \
    --yoke-from-tag vo_s3 --ref-tag vo_s3b \
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
    --vo-w 1.0 --vo-probe-n 64 \
    --ov-shadow "lin,dir" --ov-free --ov-comb-folds 2 \
    --ov-dump --ov-probe-unif-frac 0.25 \
    --ts-norm --ts-panel 128 --ts-struct 64

# Afterwards, from experiments/:
#   python3 rhm/practice/voicing/fetch_compact.py --tag ts_s0 --fetch --replace
#   python3 rhm/practice/voicing/analyze_voicing.py --tag ts_s0 --yoke-src vo_s3:voi3_dp \
#       --bank vo_s3b:voi3b_comp_pr_yk \
#       --out rhm/practice/voicing/tessitura/results/ts_s0_reduction.txt      # gate TS-1 is [R]
#   python3 rhm/practice/voicing/tessitura/reduce_rerun.py --tag ts_s0
