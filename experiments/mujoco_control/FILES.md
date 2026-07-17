# MuJoCo control substrate — File & figure index

Full file-by-file reference. Summarized in [README.md](README.md); this is the lookup material.

## Code files

| File | Purpose |
|---|---|
| `shared.py` | Modal infra: image (mujoco 3.2.3 + torch + matplotlib/scipy), `mujoco-control-data` volume, `mujoco-control` app, `NumpyEncoder`. Top-level imports kept to modal/json only; `mujoco`/`torch`/`matplotlib` imported inside function bodies so the app submits from a laptop without them. |
| `pusher_env.py` | Planar pusher DGP. `DEFAULT_DGP` knob dict; `build_xml(dgp)` (MJCF string builder — gravity-free 2D, pusher + puck + 4 walls); `PusherEnv` (reset, `set_state(qpos,qvel)` for perfect-simulator queries, `step(ctrl, n_sub)` with contact aggregation via `mj_contactForce`); `collect_transitions(...)` (scripted OU + seek-the-puck rollouts → `(S,U,S2)` + contact labels `force`/`puck_force`/`ncon`/`any_contact`/`puck_contact`; `theta=1` gives i.i.d. commands). `STATE_LABELS` = the 8 state-dim names. |
| `contact_residual.py` | **Cut #1**. Modal fn `run_contact_residual(cfg)`: collect → train arity-2 MLP `f(s,u)→Δs` (Huber loss) → residual analysis (contact/free ratio + Cohen's d + AUC, FM cosine control, force dose-response, per-dim, eff-rank, **onset-aligned event-triggered average**) → 5 figures + `results.json`. Helpers: `_cohen_d`, `_auc`, `_event_triggered`, `_effective_rank`, `_spearman`, `_r2`. `@app.local_entrypoint contact_residual(...)` builds cfg (contact-rich DGP), calls remote, mirrors outputs to `figures/<tag>/`. |
| `arity_torque.py` | **Cut #2**. Modal fn `run_arity(cfg)`: collect with **i.i.d. commands** (`u ⊥ s`) → capacity sweep of arity-1 `f(s)` vs arity-2 `f(s,u)` (identical data, only input differs) → free-flight R² per capacity + per-dim + **interventional command-sensitivity** (vary `u` at a fixed state via `set_state`, compare true vs captured) → 3 figures. `PUSHER_VEL`/`PUCK_VEL`/`POS` dim groups. `@app.local_entrypoint arity_torque(...)`. |
| `dynamics_shift.py` | **Cut #3**. `run_dynamics_shift(cfg)` (gpu=L4): momentum-dominated puck-free reaching; train arity-2 FM + CEM MPC (MB), behavior-clone a **committing motor-program** policy `π(s,g)→(re-step sequence)` (MF, matched to MB's open-loop commit — no commit-vs-reactive confound); operator shift = drag→ice; measure **reward-free re-adaptation** (MB re-fits FM self-supervised → recovers; MF flat by construction) vs **reward-driven** REINFORCE MF; the planner (and MF) **commit `replan_every` steps open-loop** (makes the world-model load-bearing). `run_replan_ablation(cfg)`: sweep `replan_every` (FMs trained once, eval-only) → the reward-free benefit vs commitment horizon. Entrypoints `dynamics_shift(...)`, `replan_ablation(...)`. Uniform `rollout(..., replan_every)` sequence interface; value dynamics-independent + fixed across d0/d1. |
| `render_video.py` | Rendered video: top-down MuJoCo view + live FM-residual side panel (headless OSMesa). See §Videos. |
| `__init__.py` | Package marker (empty). |

## Figures (`figures/<tag>/`, mirrored from the volume)

| File | Content |
|---|---|
| `fig1_trajectory.png` | FM residual over one test episode, contact shaded — spikes at contact onset |
| `fig2_distributions.png` | residual-norm violin (contact vs free) + FM-cosine violin (the scale-free control) |
| `fig3_perdim.png` | per-state-dim mean \|residual\|, contact vs free — velocity-dim localization |
| `fig4_doseresponse.png` | residual vs peak contact force (contact-only, log-x, binned means) |
| `fig5_onset_eta.png` | onset- and separation-aligned event-triggered averages + `P(in contact)` overlay — the residual is an event detector, not a state detector |

### Cut #2 figures (`figures/arity_<tag>/`)

| File | Content |
|---|---|
| `fig1_capacity_sweep.png` | free-flight R² vs capacity, arity-1 vs arity-2 (all dims + pusher velocity) — arity beats resolution |
| `fig2_perdim_arity.png` | per-dim R², arity-1 vs arity-2 — the gap is localized on the actuated pusher velocities |
| `fig3_command_sensitivity.png` | interventional: true command-driven Δs spread (perfect sim) vs captured by arity-2 vs arity-1 (=0) |

### Cut #3 figures (`figures/dynshift_<tag>/`)

| File | Content |
|---|---|
| `fig1_shift_bites.png` | bars: d0 competence, d1 zero-shot (MB-stale, MF), FM-refit oracle ceiling — the shift bites both |
| `fig2_recovery_curves.png` | two panels split by signal type: reward-free (MB recovers, MF flat) vs reward-driven MF |
| `fig3_sample_efficiency.png` | both agents on one interaction axis (MF episodes × H) — MB ~ceiling from ~50 reward-free vs MF ~2400× more, reward-labeled (money figure) |
| `fig_replan_ablation.png` | reward-free adaptation benefit (stale − refit) vs `replan_every` — grows with commitment horizon, turns over at the FM composition limit (`dynshift_replan_abl_v1/`) |
