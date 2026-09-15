#!/bin/sh
# [ov_pf1] THE PREFLIGHT OF RECORD for the readout round.
#
# Seven arms. Three are the node's own Q3b preflight twins, re-run because this round's edits
# touch shared paths (`finetune_generator_span`, `vo_run_probes`, `run_arm`) and a gate asserted
# before an edit is not a gate asserted after one:
#
#   voi3b_pf_src     the plan emitter (a yoke resolves its plan from an arm earlier in the
#                    sweep, not from the volume; a yoke with an empty plan is a vacuous twin)
#   voi3b_pf_dp      the V-4 control, yoked
#   voi3b_pf_v4pr    the critic + probe with governance OFF -> V-4B (every series at 0.0 and
#                    `t_cum` larger by EXACTLY probe count x d_fb, bill residual 0.0) and V-5
#
# Four are the round's own, all yoked to the same source so Y-1 runs on them too:
#
#   ovt_pf_comp_pr   `voi3b_pf_comp_pr` CARRYING the shadows, the free scores and the dump
#   ovt_pf_noshadow  the same arm with `ov_shadow`/`ov_free`/`ov_dump` overridden to off in its
#                    own cfg  ->  GATE R-1: one INSTRUMENT boolean apart, governance ON in both,
#                    and they must be IDENTICAL on every behaviour series. Every other inertness
#                    gate in this lineage asserts with governance OFF, which is exactly where
#                    defect #8 proved they are blind (`../DESIGN.md` §30).
#   ovt_pf_dis       + `ov_probe_dis`  ->  GATE R-6: one ALLOCATION boolean apart, and they must
#                    NOT be identical. The size is measured, never asserted.
#   ovt_pf_lin       + `ov_critic_hidden=0`: the linear readout in the chooser's seat, so that
#                    branch runs before a paid setup
#
# Run-level instruments default ON in `preflight` so every branch they add is executed here.
# SAID OUT LOUD, as every round in this lineage has: preflight checks the CODE PATH and never a
# number. No AUC, share, increment or count from this tag is a measurement.
#
# Offline first (both tables and both harnesses), then this:
#   PYTHONPATH=. python3 -c "from rhm.practice.voicing import voicing as V; V.vo_gates_cpu(); V.ov_gates_cpu()"
#   PYTHONPATH=. python3 rhm/practice/voicing/gates/falsify.py               # 36/36
#   PYTHONPATH=. python3 rhm/practice/voicing/overtone/gates/falsify_ov.py   # 26/26
cd "$(dirname "$0")/../../../../.."        # -> experiments/
export MODAL_PROFILE=chromatic
python3 rhm/practice/voicing/launch_detached.py --fn preflight --outdir-tag ov_pf1 \
    --arms "voi3b_pf_src,voi3b_pf_dp,voi3b_pf_v4pr,ovt_pf_comp_pr,ovt_pf_noshadow,ovt_pf_dis,ovt_pf_lin" \
    --ov-shadow "lin,dir" --ov-free --ov-dump --ov-probe-unif-frac 0.25
