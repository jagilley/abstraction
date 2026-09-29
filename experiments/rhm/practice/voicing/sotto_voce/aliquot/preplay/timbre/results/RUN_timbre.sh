#!/usr/bin/env bash
# [preplay/timbre] the commands of record (2026-09-23). Run from experiments/.
set -e
export MODAL_PROFILE=chromatic
T=rhm/practice/voicing/sotto_voce/aliquot/preplay/timbre
modal run $T/timbre.py::gates_t                  # T-1..T-4, T-7
modal run $T/timbre.py::falsify_t                # 7/7 shown to fail
modal run $T/timbre.py::mlp_recipe               # the MLP recipe on validation rows (NOTES section 2)
modal run $T/timbre.py::sweep --out-tag tb_smoke --arms1 s0_sv --arms5 s0_sv --smoke 1
modal run --detach $T/timbre.py::sweep --out-tag tb1 > $T/results/launch_tb1.log 2>&1
python3 $T/reduce_timbre.py --tag tb1 --fetch
