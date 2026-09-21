#!/usr/bin/env bash
# [preplay] the commands of record, tag pp1 (2026-09-21). Run from experiments/.
set -e
export MODAL_PROFILE=chromatic
B=rhm/practice/voicing/sotto_voce/aliquot/preplay
modal run $B/preplay.py::gates                       # F-1, F-3, F-4, F-5, F-6
modal run $B/preplay.py::fidelity_gate --arms s0_sv,s2_sv,s0_so,s2_so,s0_yd,s2_yd
modal run $B/preplay.py::falsify                     # 7/7 gates shown to fail
modal run $B/preplay.py::sweep --out-tag pp_smoke --arms s0_sv --smoke 1
modal run --detach $B/preplay.py::sweep --out-tag pp1 \
    --arms s0_sv,s2_sv,s0_so,s2_so,s0_yd,s2_yd
python3 $B/reduce_preplay.py --tag pp1 --fetch

# --- pp2: the read in the selector's seat (2026-09-21) ---
modal run $B/selector.py::gates2                     # S-1..S-5
modal run $B/selector.py::falsify2                   # 6/6 shown to fail
modal run $B/selector.py::sweep2 --out-tag pp2_smoke --arms s0_sv --smoke 1
modal run --detach $B/selector.py::sweep2 --out-tag pp2 \
    --arms s0_sv,s2_sv,s0_so,s2_so,s0_yd,s2_yd
python3 $B/reduce_selector.py --tag pp2 --fetch
