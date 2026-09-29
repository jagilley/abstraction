#!/usr/bin/env bash
# [grokking] Node 1 (mint): the commands of record. Run from experiments/.
set -e
export MODAL_PROFILE=chromatic
B=grokking
modal run $B/mint.py::gates                                   # G-1..G-8 (instrument checks)
modal run $B/mint.py::mint --tag smoke --smoke 1              # 3k-epoch smoke, attached
modal run --detach $B/mint.py::mint --tag m1 > $B/results/launch_m1.log 2>&1   # the run of record
python3 $B/reduce_mint.py --tag m1 --fetch                    # -> figures/mint_m1_*.{txt,png}

# --- Node 2 (rung), go given 2026-09-22: K = B/lt1 = C/lt1 of m1 (the producer top-13), lowE 13, + top1 ---
modal run $B/rung.py::rung --tag rsmoke --mint-tag m1 --smoke 1          # code check (pre-top1 build)
modal run $B/rung.py::rung --tag rsmoke2 --mint-tag m1 --smoke 1         # code check with top1 / m4 arms
modal run --detach $B/rung.py::rung --tag r1 --mint-tag m1 --k-form B --k-rule lt1 \
    --cap-main 40000 --cap-sweep 150000 > $B/results/launch_r1.log 2>&1  # 23 arms, <= 4 CPU containers
python3 $B/reduce_rung.py --tag r1 --fetch                              # -> figures/rung_r1_*
# control added after r1: all48 rescaled to minted-13's per-input norm (2 arms)
modal run --detach $B/rung.py::rung --tag r1ctrl --mint-tag m1 --k-form B --k-rule lt1 --groups ctrl \
    --cap-main 40000 > $B/results/launch_r1ctrl.log 2>&1
python3 $B/reduce_rung.py --tag r1 --extra-tags r1ctrl --fetch                # table + figures of record

# --- Rung controls r2 (2026-09-22/23): all-48 as a reparameterization of one-hot; top1 vs the recipe ---
modal run $B/rung_controls.py::controls --tag r2smoke --groups A1,B1,B2,B3,A2 --smoke 1     # code check
modal run --detach $B/rung_controls.py::controls --tag r2 --groups A1,B1,B2,B3 \
    > $B/results/launch_r2.log 2>&1                                          # 18 arms
# A1 failed -> A2: find SGD settings where raw groks (raw only), then the q-transported twins
modal run --detach $B/rung_controls.py::controls --tag r2 --groups A2 --a2-pair 0 --a2-cap 100000 \
    --a2-grid "0.1:0.001:0.9,0.3:0.0003333:0.9,1.0:0.0001:0.9,1.0:0.001:0.9" --out-name A2probe \
    > $B/results/launch_r2_a2probe.log 2>&1
modal run --detach $B/rung_controls.py::controls --tag r2 --groups A2 --a2-pair 2 --a2-cap 100000 \
    --a2-grid "0.1:0.001:0.9,0.3:0.0003333:0.9,1.0:0.0001:0.9" --out-name A2twin \
    > $B/results/launch_r2_a2twin.log 2>&1
python3 $B/reduce_rung_controls.py --tag r2 --fetch                       # -> figures/rung_controls_r2_table.txt
