# plasticity_gain — file index

**Up**: [README.md](README.md)

## Code files

| File | Purpose |
|---|---|
| `plasticity_gain.py` | The whole experiment: Type-2 drift (2 on-reach rotations + 1 off-reach noise region via existing off-by-default env knobs), one shared reward-free stream, four plasticity-weight arms (`fixed`/`raw_err`/`delta`/`delta_scalar`) at matched average weight, graded under reactive + ballistic CEM with per-region FM probes. Incremental per-arm volume commits (see README gotcha). Modal entrypoint `::plasticity_gain`; `--quick` smoke. |
| `analyze_gain.py` | Local post-processing: aggregates the per-run `results.json` mirrors into the 3-seed tables (recovery AUC, allocation, retention) reported in the README. |

## Modal volume layout (`mujoco-control-data`)

```
/data/plasticity_gain/<tag>/   results.json (full config + all traces), fig1–fig4 png
```

Mirrored locally to `figures/plasticity_gain_<tag>/`.
