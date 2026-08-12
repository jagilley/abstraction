# difficulty_sweep — file index

**Up**: [../README.md](../README.md) (bridge_assembly) · **Run record**: [../../practice/README.md](../../practice/README.md)

## Code files

| File | Purpose |
|---|---|
| `sweep.py` | The cell table and detached launcher. Every cell is the **parent** runner (`../bridge_assembly.py`) with defaults untouched except the swept knob — difficulty via `--regions` on the y=+0.30 drift corridor, b(s)'s clock via `--bench-lr`, the frontier's clock via `--adapt-lr`, the noise tax via the decoy's amplitude. `phi12` is *not* a cell to run: it is bridge_assembly's own `asm_s{0,1,2}` (identical config), reused. Launches with `start_new_session=True` per the parent's TaskStop gotcha; logs to `logs/<tag>.log`. `--list` prints the table. |
| `analyze_sweep.py` | Local aggregation, no compute. `--btrack` decomposes b(s)'s estimation error by region class and benchmark learning rate (the Q4 measurement, incl. the Huber-regime check). Preconditions (stale→ceiling separation per cell, so a nominally-harder cell that merely saturated the eval is visible as such), the outcome as `AUC_fixed − AUC_delta` at every milestone truncation *and* instantaneously, the mechanism statistics (Λ, w_A, w_R, w_B, f_boost, T_rec), the EWMA budget-match diagnostic, a seed-aggregated headline table, and the five figures. `--pull` fetches missing `results.json` from the Modal volume. |

## Data & outputs

| Path | Contents |
|---|---|
| `figures/fig1_mechanism_axis.png` | Λ, w_A, f_boost vs recovery difficulty; individual seeds shown. The tight readout. |
| `figures/fig2_outcome_axis.png` | Late-window δ-vs-uniform advantage (% of control range) vs difficulty, and vs budget per cell. |
| `figures/fig3_benchmark_clock.png` | The *r* = adapt_lr/bench_lr legs for both geometries, with the `alr9e4` collapse-test points starred. |
| `figures/fig4_noise_tax.png` | δ's allocation with the aleatoric decoy vs a clean region in the same place. |
| `figures/fig5_budget_match.png` | δ's realized mean plasticity weight vs budget — the EWMA convergence gotcha. |
| `data/summary.json` | Machine-readable per-run scalars and budget-resolved series (one entry per `cell|tag`). |
| `logs/<tag>.log` | Launcher/streaming logs, one per run. |

## Cells (28 runs in this node; 31 analysed with bridge_assembly's 3 reused)

| cell | what it varies | seeds |
|---|---|---|
| `phi06` / `phi12` / `phi20` / `phi28` | one drifted region, φ = 0.6 / 1.2 / 2.0 / 2.8 | 3 / 3 (= `asm_s*`) / 3 / 1 |
| `two12` / `two24` | two opposed rotations on the eval corridor (φ = ±1.2 / ±2.4) — the plasticity_gain geometry | 3 / 3 |
| `phi12_blr1e2` / `phi12_blr1e3` / `two12_blr1e2` / `two12_blr1e3` | b(s)'s clock (bench_lr 1e-2 / 1e-3) | 1 / 1 / 1 / 3 |
| `phi12_alr9e4` / `two12_alr9e4` | the frontier's clock (adapt_lr 9e-4) at matched *r* — the two-sided collapse test | 3 / 3 |
| `two12_nonoise` | the off-path decoy made clean (amp 30 → 0) — the noise-tax control | 3 |

## Modal volume layout (`mujoco-control-data`)

```
/data/bridge_assembly/dsw_<cell>_s<seed>/   results.json (full config + traces), fig1-5 png, done.txt
```

Mirrored locally by the parent runner's local entrypoint to `../figures/bridge_assembly_dsw_<cell>_s<seed>/`;
`analyze_sweep.py` reads that mirror, or `data/<tag>.json` when pulled with `--pull`.
