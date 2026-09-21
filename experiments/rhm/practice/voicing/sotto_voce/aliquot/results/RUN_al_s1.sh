#!/bin/sh
# [al_s1] THE VERDICT READ OFF THE WORLD MODEL'S OWN STATE — seed 0. Two arms in one tag, both
# the composed chooser, both with the probe channel ON, both CLOCK-YOKED to the banked seed-0
# anchor `vo_s3:voi3_dp` — the same yoke `sotto_voce`'s three mirror arms and `voicing` Q3b's
# floor and ceiling ride, so every comparison in the reduction is at a matched clock.
#
#        al_pj_yk   `vo_om_mode="proj"`, `vo_pj_trunk="live"`   a ridge-logistic readout of the
#                   LIVE trunk's pooled hiddens answers every probe; nothing is billed
#        al_rt_yk   the same readout over a NEVER-TRAINED trunk of the same architecture at
#                   `vo_pj_rand_seed=20260918` — `overtone`'s control two, in the loop
#
# NOTHING IS RE-RUN THAT IS ALREADY BANKED:
#   vo_s3b:voi3b_comp_yk      composed chooser, FILED diet            (the floor)
#   vo_s3b:voi3b_comp_pr_yk   composed chooser, WORLD-graded probes   (the ceiling)
#   so_s1:so_mg_yk            composed chooser, MIRROR-graded probes  (the mirror)
#   so_s1:so_cg_yk / so_hy_yk the committee and the hybrid
#   vo_s3:voi3_dp             the anchor and the yoke SOURCE
# All of them are read at reduction time with `--bank`; `--yoke-from-tag vo_s3` resolves the
# source out of `voicing`'s volume dir (`YOKE_REMOTE`), so the arm this tag replays is
# byte-identical to the one `vo_s3b` and `so_s1` replayed.
#
# WHY THIS QUESTION. `sotto_voce` put a MIRROR in the grader's seat — a separate net with its own
# representation, trained by BCE on the learner's own experienced configurations and the world's
# verdicts on them — and found it recovers about half the floor->ceiling gap at zero bill, is
# near-perfect on its own support (held-out AUC 0.996) and degrades monotonically with level off
# it. `ideas/calibration_and_violation_are_one_object.md` §12.4 names a DIFFERENT object for the
# same seat: the verdict read off the WORLD MODEL'S OWN STATE through a linear readout, whose
# features come from the corpus and from continued training rather than from what the learner
# chose to write. This tag asks whether that changes where the grader is blind. No outcome is
# pre-registered.
#
# THE READOUT. `SN.trunk`'s pooled per-block hiddens over the substituted final configuration —
# the same rendered object the mirror grades and the world grades — pooled `overtone`-style (the
# mean over all blocks concatenated with the mean over the written span), with the root one-hot
# AND its interaction with the pooled state in the design, which makes it A LINEAR READOUT PER
# GOAL (the value arc's own object: `logit_reading/orbitofrontal/`). Ridge logistic by IRLS,
# SOLVED rather than stepped, with the ridge chosen per refit on a train-internal validation
# split so the bank's held-out slice stays a clean instrument. It trains on the mirror's diet and
# nothing else (src 0: the filed rows' final configurations and the world's verdicts on them) on
# the mirror's schedule, and files its PROBABILITY as the verdict exactly as `so_mg_yk` does.
#
# THE TWO SUBSTRATE FACTS IT HANDLES, both gated (DESIGN §2, §3):
#   the plant is a masked-infill model that has never seen a fully unmasked input, so the read
#   masks ONE BLOCK OUTSIDE THE SPAN (`ratchet/macros.py::parse_features`'s idiom). The unmasked
#   read `overtone` used offline is logged beside it every tenth refit and never filed.
#   the trunk is the LIVE generator, so the features MOVE: the buffered rows are re-read through
#   the current trunk at every refit and never cached (gate P-3, shown to fail on a caching
#   perturbation), and the trunk's own fingerprint is logged every refit so "the representation
#   moved" is read off the arm file rather than assumed.
#
# Gates run before launch, all PASS:
#   G-F       0.000e+00 both arms against `enharmonic.py`, control 0.000e+00, commits equal
#             (MANDATORY: the probe path, the outcome bank and the recorder are shared paths)
#   Y-1       the replay MATCHES on both projection twins, 0 cancelled
#   V-1       0 bad filed writes, asserting every cycle
#   V-4A/B    governance off, yoked: 0.0 on every series
#   V-5       off-stream on the objects
#   V-6b      the projection-graded diet CHANGES THE RUN against the world-graded twin, and the
#             two projection arms differ from EACH OTHER (if they did not, the trunk's
#             representation would be doing nothing)
#   M-1..M-5  unchanged: the world's verdict exists for every probe, enters no loss, no buffer
#             the critic reads and no bill; the readout trains on experience alone
#   P-1       the readout is AFFINE in the pooled state, the plant's parameters are untouched to
#             the bit, no trunk parameter accumulates a gradient, no shared RNG stream moves
#   P-2       the random twin is a different object and is frozen (checked against its MINT, not
#             against its state after its first fit — DESIGN §3, a withdrawn diagnosis)
#   P-3       the features FOLLOW the trunk: perturbing the live trunk moves the live arm's
#             probability on the same row and leaves the twin's unchanged
#   P-4       the readout is live: it refits, the weights move, the verdict is not constant
#   P-2r/3r/4r the run-level forms, off the arm file
#   gates_cpu 26/26 | falsification harness (gates/falsify.py) 72/72
#
# SAID OUT LOUD: the filed target is the readout's PROBABILITY, not a verdict thresholded at 0.5,
# for `sotto_voce` DESIGN §3d's reason — at the probe base rate a 0.5 threshold discards almost
# everything the grader knows. The thresholded verdict is the READOUT and is what [P]/[R] report
# against the world. The hard-verdict variant is untested here too.
cd "$(dirname "$0")/../../../../../.."        # -> experiments/
export MODAL_PROFILE=chromatic
python3 rhm/practice/voicing/sotto_voce/aliquot/launch_detached.py --fn voicing_run --tag al_s1 \
    --eras "1:25:60,2:12:50,3:6:70,4:3:12,5:1:9" --era-caps "60,50,70,12,9" \
    --arms "al_pj_yk,al_rt_yk" \
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
    --vo-w 1.0 --vo-probe-n 64 \
    --vo-om-k 5 --vo-om-dim 64 --vo-om-lr 0.001 --vo-om-steps 16 --vo-om-batch 256 \
    --vo-om-cap 30000 --vo-om-min 512 --vo-om-boot 0.8 --vo-om-hold 0.1 \
    --vo-om-agree-q 0.5 --vo-om-freq-share 0.05 --vo-om-sample-cap 4000 \
    --vo-pj-mask --vo-pj-root inter --vo-pj-ridge "1,32,1024" --vo-pj-iters 25 \
    --vo-pj-fit-cap 8192 --vo-pj-every 1 --vo-pj-rand-seed 20260918
