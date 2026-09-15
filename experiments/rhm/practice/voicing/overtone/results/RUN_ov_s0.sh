#!/bin/sh
# [ov_s0] THE TAG OF RECORD for the readout round — how much of the judge's job is already in
# the main model, and how much needs the verdict diet.
#
# Three arms, seed 0, every one of them CLOCK-YOKED to the banked anchor `vo_s3:voi3_dp` (which
# is `en_s9:endo_ledger_open_ung5_ra` at 0.000e+00 on all thirteen series), so every arm has L4
# and L5 slots at the anchor's cycles and the readouts are read where the node is about. Each is
# `voi3b_comp_pr_yk` -- the Q3b cell whose seed-0 numbers are BANKED -- with one thing changed:
#
#   ovt_comp_pr_sh    nothing changed. It carries the S1 shadow readouts, the S2 zero-verdict
#                     scores and the row dump, and it must come out BIT-IDENTICAL to banked
#                     `vo_s3b:voi3b_comp_pr_yk` on every behaviour series INCLUDING `t_cum`
#                     (gate R-1 at full scale, reduction [R]). That identity is what licenses
#                     reading its shadows' AUCs as the banked arm's own.
#   ovt_comp_pr_dis   + `ov_probe_dis`: the SAME probe budget spent on the on-table candidate of
#                     another class maximising |z(dp/span) - z(critic)| at that context, with
#                     `ov_probe_unif_frac` of it still uniform so the audit keeps a candidate set
#                     comparable to the twin's. Same bill, row for row.
#   ovt_comp_pr_lin   + `ov_critic_hidden=0`: a LINEAR readout in the composed chooser's seat.
#
# `dp` is NOT re-run (the anchor is banked twice over and the yoke reads its plan off the
# volume), and `voi3b_comp_pr_yk` is not re-run either -- it is `ovt_comp_pr_sh`'s own identity
# target, which is the whole point of building the instruments to be inert.
#
# THE FLAG LINE BELOW IS `results/RUN_vo_s3b.sh`'s, VERBATIM, plus this round's knobs. Anything
# else moving would make the identity target meaningless.
#
# THREE CONTAINERS, ONE TAG. The arms are stream-independent by construction (`run_arm` reseeds
# per arm from TWIN/STREAM) and the substrate is deterministic from `train_seed`, so running them
# concurrently costs ~0.35 GPU-h of extra setup and saves ~2.3 h of wall clock. They share
# `setup.json`, `summary.json` and `done.txt` in the tag directory, last-writer-wins; NOTHING the
# reduction reads is among them (`analyze_voicing.py` reads per-arm `results.json` only). Wait on
# the launch logs or `modal app list`, NOT on `done.txt`. `overtone/DESIGN.md` §6.
#
# `--ref-tag vo_s3b` asserts IN RUN that this round trains the same substrate as the banked tag
# (`refs`, the stale value buffer, `read_acc`). Free, fails fast, and it is the PRECONDITION for
# gate R-1: a cross-tag bit-identity cannot hold if the substrate moved.
#
# Gates before launch (`ov_gf1` + `ov_pf1`): G-F 0.000e+00 · Y-1 exact on every yoked twin,
# 0 cancelled · V-4A/V-4B (bill residual 0.0) · V-5 · V-6b · R-1 (in-substrate pair, governance
# ON, 0.000e+00) · R-6 (the re-aimed diet is live) · gates_cpu 18/18 · ov_gates_cpu 6/6 ·
# falsify 36/36 · falsify_ov 26/26.
cd "$(dirname "$0")/../../../../.."        # -> experiments/
export MODAL_PROFILE=chromatic

# The launches are STAGGERED by 90 s. The three jobs each write `setup.json` once, about 660 s
# in; staggering makes a simultaneous write to that one shared path effectively impossible, and
# it costs nothing. Nothing the reduction reads is in a shared file either way.
for ARM in ovt_comp_pr_sh ovt_comp_pr_dis ovt_comp_pr_lin; do
python3 rhm/practice/voicing/launch_detached.py --fn voicing_run --tag ov_s0 --log-suffix "_$ARM" \
    --arms "$ARM" \
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
    --ov-dump --ov-probe-unif-frac 0.25
sleep 90
done

# Afterwards, from experiments/:
#   python3 rhm/practice/voicing/fetch_compact.py --tag ov_s0 --fetch --replace
#   python3 rhm/practice/voicing/analyze_voicing.py --tag ov_s0 --yoke-src vo_s3:voi3_dp \
#       --bank vo_s3:voi3_dp,vo_s3b:voi3b_comp_pr_yk,vo_s3b:voi3b_comp_yk \
#       --out rhm/practice/voicing/overtone/figures/ov_s0_reduction.txt
