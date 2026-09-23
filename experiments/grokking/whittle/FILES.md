# grokking/whittle — file index

Up: [`../FILES.md`](../FILES.md) · [`../README.md`](../README.md) (the parent; its §4 headlines this node). The writeup is
[`README.md`](README.md).
Decisions, gates, defects and the run record are in [`NOTES.md`](NOTES.md).

## Code files

| File | Purpose |
|---|---|
| `whittle.py` | The grokked net continues under "keep your modular addition, zero out as many weights as you can". Loads m1's banked final net and Basis's recovered subspaces (`BasisSnapshot`, K re-walked by `_walk_K`), rewrites layer 0 (and optionally the head) in the minted coordinates Pm = QR([u0 \| Q_j]), and runs `whittle_walk`: per-layer round-robin group-magnitude pruning, an exact dead-unit cascade, a settle-or-cap AdamW retrain with pruned entries held at zero, and admission iff train argmax agreement with the original net is 100%. Arms: committed, unrestricted, both_sides, all48, committed_wd1, bilinear_both (optional secondary), committed_R2000 (budget sensitivity). Gates G1–G4 (`gates_all`, `MockPruner`), a zero-prune drift twin per arm, the probe from `rung_controls.make_probe`, the end-of-arm structure read. Entrypoints `whittle_gates`, `whittle_run` (CPU coordinator), `whittle_arm` (one container per arm). |
| `reduce_whittle.py` | Fetches `/data/<tag>/whittle.json`; writes `figures/whittle_<tag>_table.txt`, `figures/whittle_<tag>_survivors.png`, `results/whittle_<tag>_summary.json`. |
| `whittle2.py` | w2: the channel move (remove one minted symbol's every read and write at once, long retrain) and biases as first-class prunable groups, over w1's banked finals; walk alternating channel and group phases; plants of one channel; SGD and lr variants; the explicit read of a one-channel bilinear (tensor fit to the complex product up to rotation, margins on all p² pairs); gates G5–G9. Entrypoints `whittle2_gates`, `whittle2_run`, `whittle2_arm`. Imports from `whittle.py`, which is unchanged. |
| `reduce_whittle2.py` | Fetches `/data/<tag>/whittle2.json` and the w2 gates; writes `figures/whittle_<tag>_table.txt`, `figures/whittle_<tag>_walk.png`, `results/whittle_<tag>_summary.json`. |

## Results and figures

| Path | What |
|---|---|
| `results/RUN_whittle.sh` | Commands of record. |
| `results/gates_whittle_R2000.log`, `results/gates_whittle_R500.log` | Gates at a 2,000-epoch cap (G3 fails on all48 and both_sides) and at the cap of record. |
| `results/smoke_whittle.log`, `results/launch_w1.log` | Smoke and launch logs. |
| `results/gates_whittle2.log`, `results/smoke_whittle2.log`, `results/launch_w2.log` | w2 gates, smoke and launch logs. |
| `results/whittle_<tag>_summary.json` | Compact per-arm series (start + admitted rounds, rejected rounds, drift twin, final counts, structure summary). |
| `results/<tag>/whittle.json` | Local copy of the volume artifact (git-ignored; `/data/<tag>/whittle.json` on `grokking-mint-data` is the record, with `/data/<tag>/<arm>.json` and `/data/<tag>/<arm>_final.npz` beside it). |
| `figures/whittle_w1_table.txt` | Table of record: setup, reference sizes, G1 and recovery, final state per arm, walk end state and compute profile, drift twin, logit-table energy, per-arm trajectories, surviving structure per unit. |
| `figures/whittle_w1_survivors.png` | Per arm: held-out vs surviving weights over admitted rounds with the drift twin (top); per-layer survivors vs round with rejected rounds marked (bottom). |
| `figures/whittle_w2_table.txt` | w2 table of record: gates G5–G9, final state per arm in both currencies, plants, every channel attempt (epochs, amplitudes and margins before/after), the one-channel bilinear written out, MLP structure, drift twins, admitted rounds. |
| `figures/whittle_w2_walk.png` | Per w2 arm: surviving groups and scalars per admitted round with channel removals marked (top); held-out with the drift twin (bottom). |
