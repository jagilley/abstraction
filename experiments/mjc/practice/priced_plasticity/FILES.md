# FILES — priced_plasticity

**Up**: [../README.md](../README.md) (practice) · [../../README.md](../../README.md) (mjc)

## Code files

| File | Purpose |
|---|---|
| `priced_plasticity.py` | The whole experiment. Fork of `../../bridge_assembly/bridge_assembly.py` (same two-corridor geometry, same corrected δ, same gain law and calibration protocol) with three cost knobs flipped (`fm_hidden`/`fm_layers`, `n_replay`, `norm_mode`) and the plasticity weight field factorized into **relative allocation × total spend** so δ can choose its own spend under Adam (`spend_mode`). Two modes: `--mode calibrate` (FM probes only, a grid over capacity × replay × lr on the `fixed` arm + the Adam scale-invariance probe) and `--mode main` (the full comparison, with control grading). Modal entrypoint `::priced`; `--quick` smoke. |
| `launch_detached.py` | setsid-isolated detached launcher (the plasticity_gain gotcha fix — a harness TaskStop reaching the client's process group can swallow a post-cancellation `volume.commit()`). |
|  `analyze_priced.py` | Local aggregation: calibration tables, the **frontier statistics** (retention saved at matched adaptation, and its transpose, by interpolation along the uniform-lr curve), and figures. |

## Arm-token syntax (`--arms`)

`kind[@lr][:spend][:norm][:rN][:hH][:xM]`, comma-separated. `kind ∈ {fixed, raw_err, delta}`;
`spend ∈ {raw_adam, free, unit}`; `norm ∈ {ewma, static}`; `rN` = `n_replay`; `hH` =
`fm_hidden` for that arm (each distinct capacity is pretrained once and shared);
`xM` = constant multiplier on the raw weight (the Adam-invariance probe). Omitted
qualifiers fall back to the run-level defaults.

Reproducing the parent's exact plasticity pipeline inside this fork:
`--arms "fixed@3e-4:raw_adam:ewma:r4:h256,delta@3e-4:raw_adam:ewma:r4:h256"` with
`--fm-layers 3`.

## Modal volume layout (`mujoco-control-data`)

```
/data/priced_plasticity/<tag>/   results.json (full config + per-arm traces/blogs/budgets
                                 + per-stack calibration), done.txt
```

Mirrored locally to `data/<tag>.json` (by the local entrypoint, or by
`analyze_priced.py --pull`). Figures are generated locally into `figures/`.
