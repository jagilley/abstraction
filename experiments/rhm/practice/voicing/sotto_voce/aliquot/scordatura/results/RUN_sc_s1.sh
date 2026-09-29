#!/bin/sh
# [sc_s1] THE FLOOR, seed 0, segment 2 (DESIGN §4). `sc_gn_yk` RESUMED from the pre-check's own c40
# save (`sc_p1`, RUN_sc_p1.sh) and continued to the end of the ladder (201 cycles), saving at every
# 10th cycle and every era boundary; the fire survey on the final state. The arm's whole record
# (c1-c201) is in this tag's arm file: the log and the admission record are state.
cd "$(dirname "$0")/../../../../../../.."        # -> experiments/
export MODAL_PROFILE=chromatic
python3 rhm/practice/voicing/sotto_voce/aliquot/scordatura/launch_detached.py --fn sweep \
    --out-tag sc_s1 --seed 0 --arms "sc_gn_yk" \
    --flags-json '{"rb_resume": "sc_p1__sc_gn_yk/sc_gn_yk/ck/c040.pt", "rb_auto_resume": true, "rb_ck_every": 10, "rb_ck_era": true, "sc_fire_keys": true}'
