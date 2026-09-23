# grokking — file index

Up: [`README.md`](README.md) (the writeup) · [`experiments/`](../CLAUDE.md).
Decisions, gates, falsifiers and defects are in [`NOTES.md`](NOTES.md).

## Code files

| File | Purpose |
|---|---|
| `shared.py` | Modal app `grokking-mint` / image / volume `grokking-mint-data`; `GrokMLP` and `generate_data` forked verbatim from `inverse_dynamics/grokking_fwd_vs_inv.py`; `split_pairs`, `onehot`, `char_features`; `train_net` (the donor's full-batch AdamW loop with snapshots and a test-acc stop). |
| `mint.py` | Node 1. Grok once (seed 42) with snapshots; at each snapshot read layer-0 frequencies, walk `census_walk` (forked verbatim from preplay `incremental.py`) with one gate per compiled form (A closed form, ALS LS amplitudes, B unit-pruned net, C frequency-filtered net) under three tolerances; shuffled-label and random-init falsifiers; SVD rank baseline on the final net. Entrypoints `gates`, `mint`, `analyze`. |
| `reduce_mint.py` | Fetches `mint.json`, writes `figures/mint_<tag>_table.txt`, `_curve.png`, `_read.png`, `_svd.png`, `results/mint_<tag>_summary.json`. |
| `rung.py` | Node 2. Fresh nets over a frozen alphabet (raw / minted K / all 48 / k lowest-energy / top1) × learner (the MLP with input dim swapped / a bilinear product net, width 4k, plus top1 at its minimal width 4), and a train-fraction sweep raw vs minted vs top1; optional (a+b+c) mod p (not run). Entrypoint `rung`. |
| `reduce_rung.py` | Fetches `rung.json`, writes `figures/rung_<tag>_table.txt`, `_curves.png`, `_sweep.png`. |
| `rung_controls.py` | Controls for Rung's two failures (tag r2). A: all-48 with DC as an orthogonal reparameterization of one-hot, with raw's init transported (AdamW), then raw vs its twin under SGD + L2. B: top1 with the exact solution planted, a weight-decay sweep, and the minted-13 bilinear readout's amplitude and margin. Entrypoint `controls`. |
| `reduce_rung_controls.py` | Fetches every `controls_*.json` of a tag, writes `figures/rung_controls_<tag>_table.txt`. |

## Results and figures

| Path | What |
|---|---|
| `results/RUN_mint.sh` | Commands of record (Mint and Rung). |
| `results/launch_m1.log`, `results/launch_r1.log`, `results/smoke_*.log` | Launch and smoke logs. |
| `results/mint_m1_summary.json` | Compact per-snapshot series behind the figures. |
| `results/<tag>/mint.json`, `results/<tag>/rung.json` | Local copies of the volume artifacts (git-ignored; `/data/<tag>/` on `grokking-mint-data` is the record, with the snapshots banked beside it). |
| `figures/mint_m1_table.txt` | Table of record: per-snapshot tables per rule, random-order controls, final-net read / K / cross / SVD / size tables, instrument checks, falsifiers, first-threshold epochs. |
| `figures/mint_m1_curve.png` | Held-out accuracy of each compiled form vs the net's test accuracy across snapshots, with kept k below, one column per gate rule. |
| `figures/mint_m1_read.png` | Producer score per frequency over training (heatmap); final-net score and logit-table diagonal energy per frequency. |
| `figures/mint_m1_svd.png` | SVD rank sweep vs the compiled forms, held-out accuracy against parameters. |
| `figures/rung_r1_table.txt` | Rung table of record: per-arm (main, sweep, the r1ctrl norm-matched control), sweep summary, consistency line. |
| `figures/rung_r1_curves.png` | Test accuracy vs epoch per alphabet at train fraction 0.3, MLP and bilinear. |
| `figures/rung_controls_r2_table.txt` | Rung controls A (all-48 reparameterization: init, optimizer) and B (top1: planted solution, weight-decay sweep, minted-13 readout amplitude/margin). |
| `figures/rung_r1_sweep.png` | Epochs to 0.99 / 1.0 test and final test accuracy vs train fraction, raw vs minted vs top1. |

## Children

| Folder | What |
|---|---|
| [`basis/`](basis/FILES.md) | Endogenous basis mint: diagonalize the net's own operator a ↦ net(a, g) at every banked m1 snapshot, certify by cross-g agreement, gauge against the DFT (logged only), and walk / compile over the recovered basis. Decisions and gates in `basis/NOTES.md`. |
