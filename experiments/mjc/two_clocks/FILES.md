# two_clocks — file index

**Up**: [README.md](README.md)

## Code files

| File | Purpose |
|---|---|
| `two_clocks.py` | The whole experiment: two anonymous teaching signals (fast per-transition δ delayed d_fm; slow trial-level reward advantage delayed d_r) consumed through unit-area eligibility kernels as per-sample plasticity gains at matched budget; arms differ only in kernel architecture (`two_channel` vs `shared:*` vs specialists vs `fixed`). Scripted pre-drift-FM trials make the reward stream arm-independent. Modal entrypoint `::two_clocks`; `--quick` smoke; `--arms "fast:tX:dY,..."` for the window×delay sweep. |
| `analyze_two_clocks.py` | Local aggregation: 3-seed tables (R1 AUC, over-weightings, credit fidelities), the τ×d diagonal figure, per-run figure regeneration. |

## Modal volume layout (`mujoco-control-data`)

```
/data/two_clocks/<tag>/   results.json (full config + tables), traces.npz, fig1–fig4 png, done.txt
```

Mirrored locally to `results/<tag>/`; analysis figures in `figures/analysis/`.
