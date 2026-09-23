#!/usr/bin/env bash
# [grokking/basis] Endogenous basis mint: the commands of record. Run from experiments/.
# No retraining: reads m1's banked snapshots (/data/m1/{true,shuffled}/snapshots.npz on grokking-mint-data).
# MODAL_BUILD_VALIDATION=warn: importing grokking.mint makes Modal mount the whole grokking/ folder
# (logs included) as an entrypoint mount; a log another client is still writing then fails the build check
# (NOTES.md defect B-D1). Numbers are unaffected.
set -e
export MODAL_PROFILE=chromatic MODAL_BUILD_VALIDATION=warn
B=grokking/basis
modal run $B/basis.py::basis_gates > $B/results/gates_basis.log 2>&1              # designed + instrument checks
modal run $B/basis.py::basis_run --tag bsmoke --smoke 1 > $B/results/smoke_basis.log 2>&1   # 7 snapshots, attached
modal run --detach $B/basis.py::basis_run --tag b1 > $B/results/launch_b1.log 2>&1 # run of record, 1 CPU container
python3 $B/reduce_basis.py --tag b1 --fetch                                         # -> figures/basis_b1_*
