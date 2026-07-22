# Cut #1 — contact-residual structure — File & figure index

Full file-by-file reference for Cut #1. Summarized in [README.md](README.md); this is the lookup
material. Code lives **in this folder** (per [STRUCTURE.md](../../../STRUCTURE.md): code lives at the
lowest node that shares it); the only shared dependencies are [`../pusher_env.py`](../pusher_env.py)
and [`../shared.py`](../shared.py), which stay at the `mjc` node because every experiment imports them.

## Code files

| File | Purpose |
|---|---|
| `contact_residual.py` | **Cut #1**. Modal fn `run_contact_residual(cfg)`: collect → train arity-2 MLP `f(s,u)→Δs` (Huber loss) → residual analysis (contact/free ratio + Cohen's d + AUC, FM cosine control, force dose-response, per-dim, eff-rank, **onset-aligned event-triggered average**) → 5 figures + `results.json`. Helpers: `_cohen_d`, `_auc`, `_event_triggered`, `_effective_rank`, `_spearman`, `_r2`. `@app.local_entrypoint contact_residual(...)` builds cfg (contact-rich DGP), calls remote, mirrors outputs to `figures/<tag>/`. |
| `__init__.py` | Package marker (empty). |

## Env support (in the parent, shared)

| Symbol | Purpose |
|---|---|
| `PusherEnv`, `collect_transitions`, `STATE_LABELS` ([`../pusher_env.py`](../pusher_env.py)) | The planar-pusher DGP this cut is defined on. `collect_transitions` runs the scripted OU + seek-the-puck behavior policy and returns `(S,U,S2)` plus the ground-truth contact labels (`force` / `puck_force` / `ncon` / `any_contact` / `puck_contact`) read straight from MuJoCo — the physics analog of RHM's known latents. The contact-rich DGP knobs used here (`arena_half=0.8`, `pusher_r=puck_r=0.15`, `puck_mass=2.0`) are set by this cut's local entrypoint, not baked into the env. |

## Figures (`figures/<tag>/`, mirrored from the volume)

| Directory | Content |
|---|---|
| `full_v1/` | The result. `fig1_trajectory` (FM residual over one test episode, contact shaded — spikes at contact onset) · `fig2_distributions` (residual-norm violin contact-vs-free + FM-cosine violin, the scale-free control) · `fig3_perdim` (per-state-dim mean \|residual\| — velocity-dim localization) · `fig4_doseresponse` (residual vs peak contact force, contact-only, log-x, binned) · **`fig5_onset_eta`** (onset- and separation-aligned event-triggered averages + `P(in contact)` overlay — the residual is an *event* detector, not a state detector). Plus `results.json`. |
| `smoke/` | `--quick` output kept for debugging (no `fig5`), not a result. |

## Modal volume layout (`mujoco-control-data`)

```
/data/contact_residual/<tag>/   results.json, fig1..fig5 .png
```
