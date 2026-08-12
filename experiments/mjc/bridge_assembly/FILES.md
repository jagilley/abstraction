# bridge_assembly — file index

**Up**: [README.md](README.md)

## Code files

| File | Purpose |
|---|---|
| `bridge_assembly.py` | The whole experiment: two-corridor geometry (practiced B-mastered on the eval path; A-drift + off-path noise at t=0), three plasticity arms (`fixed`/`raw_err`/`delta`) at matched average lr over an identical stream, with delta = the fully corrected δ live (context-conditional b(s), centered gate per agency_gate's calibration, no-gate counterfactual logged). Graded under reactive + ballistic CEM with per-region probes; incremental per-arm/milestone volume commits. Modal entrypoint `::bridge_assembly`; `--quick` smoke. |
| `launch_detached.py` | setsid-isolated detached launcher (immune to harness TaskStop reaching the client's process group — the plasticity_gain gotcha). |
| `analyze_assembly.py` | Local aggregation across tags: 3-seed tables (recovery AUCs per controller/family, allocation, retention), figure regeneration. |

## Modal volume layout (`mujoco-control-data`)

```
/data/bridge_assembly/<tag>/   results.json (full config + traces), fig1–fig5 png, done.txt
```

Mirrored locally to `figures/bridge_assembly_<tag>/`.
