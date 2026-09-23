# grokking/basis — file index

Up: [`README.md`](README.md) (pointer) · [`../README.md`](../README.md) §2 (the writeup) · [`../FILES.md`](../FILES.md).
Decisions, gates, falsifiers and defects are in [`NOTES.md`](NOTES.md).

## Code files

| File | Purpose |
|---|---|
| `basis.py` | Endogenous basis mint on m1's banked snapshots (no retraining). Reads the net's operator M_g[c,a] = softmax(net(a,g))[c] for 6 generators, diagonalizes M_g_ref into real invariant subspaces (the candidates), logs the cross-g agreement certificate (endogenous) and the DFT overlap / eigenvalue-phase check (oracle, never consumed), then walks `census_walk` and compiles forms A / ALS / B / C over the recovered basis (`BasisSnapshot` subclasses `mint.Snapshot`). Designed checks (exact shift, random permutation, random 97-cycle, relabelled action, soft shift) and instrument checks (DFT basis injected reproduces m1). Entrypoints `basis_gates`, `basis_run`. |
| `reduce_basis.py` | Fetches `basis.json`, reads m1's `mint.json`, writes `figures/basis_<tag>_table.txt`, `figures/basis_<tag>_read.png`, `results/basis_<tag>_summary.json`. |

## Results and figures

| Path | What |
|---|---|
| `results/RUN_basis.sh` | Commands of record. |
| `results/gates_basis.log`, `results/smoke_basis.log`, `results/launch_b1.log` | Gate, smoke and launch logs. |
| `results/basis_<tag>_summary.json` | Compact per-snapshot series behind the figure. |
| `results/<tag>/basis.json` | Local copy of the volume artifact (git-ignored; `/data/<tag>/basis.json` on `grokking-mint-data` is the record). |
| `figures/basis_b1_table.txt` | Table of record: designed / instrument checks, per-snapshot read table (operator, certificate, oracle), per-snapshot walk tables per rule with kept K matched to DFT k vs m1, random-order controls, final-net per-subspace table, phase check, per-frequency recovery, falsifiers, first-threshold epochs. |
| `figures/basis_b1_read.png` | DFT overlap and cross-g agreement over epochs with the net's test accuracy and the bijection fraction; kept k (B, C; lt1) vs m1; per-frequency recovery heatmap. |
