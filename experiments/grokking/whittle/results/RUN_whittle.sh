#!/usr/bin/env bash
# [grokking/whittle] Whittle: the commands of record. Run from experiments/.
# No fresh training: every arm starts from m1's banked final net (/data/m1/true/snapshots.npz, e39999) or r2's
# banked minted-13 bilinear (/data/r2/B3_minted13_bil_stop_final.npz). CPU containers only.
# MODAL_BUILD_VALIDATION=warn: importing grokking.mint / grokking.basis mounts the whole grokking/ folder, and a
# launch log still being written fails the build check (basis/NOTES.md defect B-D1). Numbers are unaffected.
set -e
export MODAL_PROFILE=chromatic MODAL_BUILD_VALIDATION=warn
W=grokking/whittle
# gates G1-G4 at a 2000-epoch cap (G3 fails on all48 and both_sides: NOTES decision 6), then at the cap of record
modal run $W/whittle.py::whittle_gates --retrain 2000 > $W/results/gates_whittle_R2000.log 2>&1
modal run $W/whittle.py::whittle_gates --retrain 500 > $W/results/gates_whittle_R500.log 2>&1
# smoke: all 7 arms, cap 200, settle 20, 9 rounds each, attached
modal run $W/whittle.py::whittle_run --tag wsmoke --smoke 1 > $W/results/smoke_whittle.log 2>&1
# run of record: CPU coordinator starmaps one container per arm (7 arms)
modal run --detach $W/whittle.py::whittle_run --tag w1 > $W/results/launch_w1.log 2>&1
python3 $W/reduce_whittle.py --tag w1 --fetch                      # -> figures/whittle_w1_*, results/whittle_w1_summary.json

# --- w2 (2026-09-23): the channel move + biases as prunable groups, to the floor in frequencies (whittle2.py) ---
modal run $W/whittle2.py::whittle2_gates > $W/results/gates_whittle2.log 2>&1              # G5-G9
modal run $W/whittle2.py::whittle2_run --tag w2smoke --smoke 1 > $W/results/smoke_whittle2.log 2>&1
modal run --detach $W/whittle2.py::whittle2_run --tag w2 > $W/results/launch_w2.log 2>&1   # 8 arms, 1 container each
python3 $W/reduce_whittle2.py --tag w2 --fetch                                              # -> figures/whittle_w2_*
