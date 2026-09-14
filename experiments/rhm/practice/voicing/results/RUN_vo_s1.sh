#!/bin/sh
# [vo_s1] Q1 — THE CHOOSER AT THE CLASS, TRAINED ON ITS OWN GRADED ATTEMPTS. A 2x2 in one tag
# on `en_s9`'s exact ladder and character (`endo_ledger_open_ung5_ra`, self-paced), two knobs:
#
#        write \ objective        donor self-imitation     record-verdict calibration
#        free-run argmax          voi_dp                   voi_own_record
#        sampled, on-table        voi_xp                   voi_own_record_xp
#
# `voi_dp` names NEITHER consumer knob, so it is `en_s9`'s arm character for character and is
# the in-tag identity check and the full-scale half of G-F at once. All four carry `vo_record`:
# the record and its four instruments are unpriced readouts and the anchor has to carry them or
# the arms are not comparable on them.
#
# WHY THE SECOND KNOB EXISTS. Q0 section 5: at L4/L5 the executor writes one or two rows of a
# 128-192 row book on nearly every call (top-1 share 0.93-0.996; ONE token class of six at the
# L5 commit), so solved and unsolved calls carry the SAME record and an own-attempt objective
# has no contrast to learn from. Every inverse model this node is framed on is trained on
# babbling. `vo_explore` is that missing half: in the PRICED PRACTICE BEAM ONLY, the entry is
# sampled from the chooser's own scores at a temperature instead of argmaxed -- through the
# head on an open slot (`vo_head_scores`, on-table by construction, which is the question) and
# through the DP's own per-entry scores on a closed one (`vo_dp_scores`, gate VO-4). The
# metering beam, every audition and every probe stay at argmax.
#
# THE TEMPERATURE IS APPLIED TO THE PER-BLOCK SCORE (`scores / span`). Both score functions sum
# over the span, so one temperature could not serve a span-2 slot and a span-16 slot; the
# division is the file's own convention for this quantity (`vo_train_terms` divides by `span`
# for the same reason, and that is what makes gate V-2b equal `F.cross_entropy`). See
# `DESIGN.md` section 12(d)-(e), including what the smoke could NOT size and why.
#
# Gates run before launch, all PASS:
#   G-F   `fidelity_smoke` retargeted at `enharmonic.py` -> 0.000e+00 both arms, commits equal,
#         donor self-replay control 0.000e+00                                    (tag vo_gf1)
#   V-1   the record renders to what the beam wrote: 0 bad of 5,432 filed writes over four
#         preflight arms, asserting on every cycle                    (preflight _preflight_vo1)
#   VO-8  rows filed, sampled arms left the argmax, argmax arms sampled nothing, the
#         calibration term produced a per-slot readout                (preflight _preflight_vo1)
#   gates_cpu 10/10: V-1, V-2b, V-3a/b/c (E-0, E-1, E-7), VO-3, VO-4, VO-5, VO-6, VO-7
#
# Banked for comparison at reduction time:
#   en_s9:endo_ledger_open_ung5_ra   the arm all four fork (the identity check for voi_dp)
#   en_s8:endo_ledger_open_ung5      the same arm with the re-arm hook on both clocks
#   en_s8:endo_ledger                the closed control
cd "$(dirname "$0")/../../../.."          # -> experiments/
export MODAL_PROFILE=chromatic
python3 rhm/practice/voicing/launch_detached.py --fn voicing_run --tag vo_s1 \
    --eras "1:25:60,2:12:50,3:6:70,4:3:12,5:1:9" --era-caps "60,50,70,12,9" \
    --arms "voi_dp,voi_own_record,voi_xp,voi_own_record_xp" \
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
    --vo-explore-t 0.25 --vo-push 1.0 --vo-rec-cap 8192 --vo-rec-batch 64 \
    --vo-chunk 32 --vo-readback-n 512 --vo-rep-n 64 --vo-verify-cycles 8
