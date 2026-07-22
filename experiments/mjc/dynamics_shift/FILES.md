# Cut #3 — operator-intervention re-adaptation — File & figure index

Full file-by-file reference for Cut #3. Summarized in [README.md](README.md); this is the lookup
material. Code lives **in this folder** (per [STRUCTURE.md](../../../STRUCTURE.md): code lives at the
lowest node that shares it); the only shared dependencies are [`../pusher_env.py`](../pusher_env.py)
and [`../shared.py`](../shared.py), which stay at the `mjc` node because every experiment imports them.

Note that this cut's substrate — puck-free momentum-dominated goal-reaching + an arity-2 FM + CEM
MPC — is the base that [`../directed_readapt/`](../directed_readapt/README.md),
[`../meta_adapt/`](../meta_adapt/README.md), [`../curiosity_control/`](../curiosity_control/README.md),
[`../drift_value_loop/`](../drift_value_loop/README.md) and [`../ballistic/`](../ballistic/README.md)
build on. Those nodes each **duplicate** the helpers rather than importing them (deliberate: it keeps
every prior result reproducible against its own frozen copy).

## Code files

| File | Purpose |
|---|---|
| `dynamics_shift.py` | **Cut #3**. `run_dynamics_shift(cfg)` (gpu=L4): momentum-dominated puck-free reaching; train arity-2 FM + CEM MPC (MB), behavior-clone a **committing motor-program** policy `π(s,g)→(re-step sequence)` (MF, matched to MB's open-loop commit — no commit-vs-reactive confound); operator shift = drag→ice; measure **reward-free re-adaptation** (MB re-fits FM self-supervised → recovers; MF flat by construction) vs **reward-driven** REINFORCE MF; the planner (and MF) **commit `replan_every` steps open-loop** (makes the world-model load-bearing). `run_replan_ablation(cfg)`: sweep `replan_every` (FMs trained once, eval-only) → the reward-free benefit vs commitment horizon. Entrypoints `dynamics_shift(...)`, `replan_ablation(...)`. Uniform `rollout(..., replan_every)` sequence interface; value dynamics-independent + fixed across d0/d1. |
| `__init__.py` | Package marker (empty). |

## Env support (in the parent, shared)

| Symbol | Purpose |
|---|---|
| `build_xml(dgp, with_puck=False)`, `PusherEnv` ([`../pusher_env.py`](../pusher_env.py)) | The puck-free, momentum-dominated variant of the planar pusher. The operator shift is a native DGP knob (`joint_damping 2.0 → 0.05`) — no env change was needed to intervene on the transition operator, which is the point. |

## Figures (`figures/dynshift_<tag>/`, mirrored from the volume)

| Directory | Content |
|---|---|
| `dynshift_full_v3/` | **The fair run** (committing motor-program MF). `fig1_shift_bites` (bars: d0 competence, d1 zero-shot MB-stale/MF, FM-refit oracle ceiling) · `fig2_recovery_curves` (two panels split by signal type: reward-free — MB recovers, MF flat — vs reward-driven MF) · **`fig3_sample_efficiency`** (both agents on one interaction axis, the money figure). |
| `dynshift_full_v2/` | The **earlier reactive-MF run**, kept for the record — this is the version with the commit-vs-reactive confound, superseded by `full_v3`. |
| `dynshift_full_v1/` | `--replan-every 1`: the muted, re-groundable endpoint (feedback substitutes for the model). |
| `dynshift_replan_abl_v1/` | `fig_replan_ablation` — reward-free adaptation benefit (stale − refit) vs `replan_every`; grows with commitment horizon, turns over at the FM composition limit. |
| `dynshift_smoke/`, `dynshift_replan_abl_smoke/` | `--quick` outputs kept for debugging, not results. |

## Modal volume layout (`mujoco-control-data`)

```
/data/dynamics_shift/<tag>/   results.json, fig1..fig3 .png (or fig_replan_ablation.png)
```
