#!/bin/sh
# [sb_s2] THE OUTCOME ERROR IN THE PLANT'S OWN WEIGHTS — seed 2. THREE arms, one CONTAINER EACH
# (a CPU coordinator `.starmap`s them; see DESIGN §6), all the composed chooser, all with the probe
# channel ON, all CLOCK-YOKED to the banked seed-2 anchor `vo_s3d2:voi3_dp` — the same yoke
# `sotto_voce`'s three mirror arms, `voicing` Q3b's floor and ceiling, and `aliquot`'s two
# projection arms all ride, so every comparison in the reduction is at a matched clock.
#
#     sb_sv_yk   an outcome head over the plant's pooled hiddens, trained by BCE on the WORLD'S
#                VERDICT of the learner's own experienced configurations (the mirror's src-0 diet),
#                its gradient reaching the TRUNK in the same optimizer step as the plant's own
#                masked-infill training and never the emission head
#     sb_yd_yk   the same wiring with the head's target NEXT-LEVEL YIELD: the share of a solved
#                configuration's level-(era+1) spans whose key is at support in the learner's OWN
#                miner, zero on an unsolved piece, computed at PUSH time from the miner's counts as
#                they stood then (DESIGN §3.2)
#     sb_so_yk   the verdict again with the plant's own masked-infill term OFF — `duplex`'s `out@N`
#                in the loop, where the executor's DP scores through a plant that may have left the
#                masked-infill family
#
# The readout in the grader's seat is `aliquot`'s, UNCHANGED, so the only difference from the
# banked `al_pj_yk` is that the plant has HEARD OUTCOMES.
#
# NOTHING IS RE-RUN THAT IS ALREADY BANKED:
#   vo_s3e:voi3b_comp_yk      composed chooser, FILED diet            (the floor)
#   vo_s3e:voi3b_comp_pr_yk   composed chooser, WORLD-graded probes   (the ceiling)
#   so_s2:so_mg_yk            composed chooser, MIRROR-graded probes  (the mirror)
#   so_s2:so_cg_yk / so_hy_yk the committee and the hybrid
#   al_s2:al_pj_yk            the FROZEN-PLANT projection — the arm these three are one knob from
#   al_s2:al_rt_yk            the random-trunk twin
#   vo_s3d2:voi3_dp             the anchor and the yoke SOURCE
# All read at reduction time with `--bank`.
#
# THE FLAGS ARE `SB_FLAGS` in `soundboard.py` — `aliquot`'s `RUN_al_s1.sh` flag line verbatim plus
# this node's own knobs — and the coordinator writes the exact per-arm kwargs into
# `<tag>/sweep.json`, so what ran is on the record rather than on a command line a coordinator had
# to relay. `--flags-json` overrides any of them.
#
# Gates run before launch, all PASS:
#   G-F       0.000e+00 against `enharmonic.py`, donor self-replay control 0.000e+00 (tag sb_gf1)
#   gates_cpu 27/27 | falsification harness 100/100 | P-1..P-5 and S-1..S-6 on an L4
#   preflight 5 twins (sb_pf1): Y-1 exact 0 cancelled, V-1, V-4A/B, V-5, V-6b, M-1..M-5 and their
#             run-level forms, P-2r/P-3r/P-4r, and S-1r..S-4r — including S-4r, the gate this round
#             rests on: the SHAPED arm's plant differs from the UNSHAPED `al_pf_pj` twin's
#   TWO inherited-gate corrections the preflight forced, both in DESIGN §7.4/§7.5 with the
#   perturbations they now fail on
cd "$(dirname "$0")/../../../../../../.."        # -> experiments/
export MODAL_PROFILE=chromatic
python3 rhm/practice/voicing/sotto_voce/aliquot/soundboard/launch_detached.py --fn sweep \
    --out-tag sb_s2 --seed 2 --arms "sb_sv_yk,sb_yd_yk,sb_so_yk"
