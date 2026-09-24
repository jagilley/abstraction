#!/bin/sh
# [sc_pf2] G-J IN RUN, its red runs, on gate R-1's vehicle (which solves, where the G-F smoke's
# substrate solved nothing in its first cycles and left the `solved` and `prepend` knobs BLIND):
# `sc_pf_gn` with each of J1-J4's red knobs, eight cycles each, one container, no saves.
cd "$(dirname "$0")/../../../../../../.."        # -> experiments/
export MODAL_PROFILE=chromatic
python3 rhm/practice/voicing/sotto_voce/aliquot/scordatura/launch_detached.py --fn rb_r1_run \
    --tag sc_pf2 --role multi --plan-tag ../rhm_practice_rubato/rb_r1_P --stop-cycle 8 \
    --cfg-extra '[{"arm": "sc_pf_gn", "label": "Frng", "cfg": {"mj_falsify": "rng"}}, {"arm": "sc_pf_gn", "label": "Fsolved", "cfg": {"mj_falsify": "solved"}}, {"arm": "sc_pf_gn", "label": "Fprepend", "cfg": {"mj_falsify": "prepend"}}, {"arm": "sc_pf_gn", "label": "Fshadow", "cfg": {"mj_falsify": "shadow"}}]'
