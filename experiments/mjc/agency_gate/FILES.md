# agency_gate — file index

**Up**: [README.md](README.md)

## Code files

| File | Purpose |
|---|---|
| `agency_gate.py` | The whole experiment: trains FM₁(s)/FM₂(s,u) on shared i.i.d.-command free-flight transitions ([`../pusher_env.py`](../pusher_env.py) untouched), evaluates the bridge signal δ=(b−e)·σ((g−g₀)/θ) twice per held-out trajectory (ACT: true u; PLAYBACK: u=0), injects observation-side distortion events, writes `results.json` + 4 figures + `eval_traces.npz`. Modal entrypoint `::agency_gate`; `--quick` smoke. |

## Modal volume layout (`mujoco-control-data`)

```
/data/agency_gate/<tag>/   results.json, fig1–fig4 png, eval_traces.npz
```

Mirrored locally to `figures/<tag>/`.
