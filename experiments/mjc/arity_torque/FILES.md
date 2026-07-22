# Cut #2 — arity on torque — File & figure index

Full file-by-file reference for Cut #2. Summarized in [README.md](README.md); this is the lookup
material. Code lives **in this folder** (per [STRUCTURE.md](../../../STRUCTURE.md): code lives at the
lowest node that shares it); the only shared dependencies are [`../pusher_env.py`](../pusher_env.py)
and [`../shared.py`](../shared.py), which stay at the `mjc` node because every experiment imports them.

## Code files

| File | Purpose |
|---|---|
| `arity_torque.py` | **Cut #2**. Modal fn `run_arity(cfg)`: collect with **i.i.d. commands** (`u ⊥ s`) → capacity sweep of arity-1 `f(s)` vs arity-2 `f(s,u)` (identical data, only the input differs) → free-flight R² per capacity + per-dim + **interventional command-sensitivity** (vary `u` at a fixed state via `set_state`, compare true vs captured) → 3 figures. `PUSHER_VEL`/`PUCK_VEL`/`POS` dim groups. `@app.local_entrypoint arity_torque(...)`. |
| `__init__.py` | Package marker (empty). |

## Env support (in the parent, shared)

| Symbol | Purpose |
|---|---|
| `collect_transitions(..., theta=1, seek_gain=0)` ([`../pusher_env.py`](../pusher_env.py)) | `theta=1` makes the OU command process memoryless → **i.i.d. commands**, killing the received-wisdom confound (measured max \|corr(u,s)\| = 0.003). This is the one env knob the whole cut turns on. |
| `PusherEnv.set_state(qpos, qvel)` ([`../pusher_env.py`](../pusher_env.py)) | Perfect-simulator query used for the **interventional** readout (fig3): reset to a fixed state, vary `u`, measure the true Δs spread. |

## Figures (`figures/arity_<tag>/`, mirrored from the volume)

| Directory | Content |
|---|---|
| `arity_full_v1/` | The result. `fig1_capacity_sweep` (free-flight R² vs capacity, arity-1 vs arity-2, all dims + pusher velocity — arity beats resolution) · `fig2_perdim_arity` (per-dim R² — the gap is localized on the actuated pusher velocities) · `fig3_command_sensitivity` (interventional: true command-driven Δs spread from the perfect sim vs captured by arity-2 vs arity-1 = 0). Plus `results.json`. |
| `arity_smoke/` | `--quick` output kept for debugging, not a result. |

## Modal volume layout (`mujoco-control-data`)

```
/data/arity_torque/<tag>/   results.json, fig1..fig3 .png
```
