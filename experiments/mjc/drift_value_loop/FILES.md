# The two-timescale value loop under perpetual drift — File & figure index

Full file-by-file reference for the perpetual-drift value-loop arc (Cuts 1/2/3). Summarized in
[README.md](README.md); this is the lookup material. Code lives **in this folder** (per
[STRUCTURE.md](../../../STRUCTURE.md)); the only shared dependencies are
[`../pusher_env.py`](../pusher_env.py) and [`../shared.py`](../shared.py), which stay at the `mjc`
node because every experiment there imports them.

## Code files

| File | Purpose |
|---|---|
| `compounding_drift.py` | **Cuts 1+2.** `run_compounding_drift(cfg)`: a `push_rot` (or damping-floor) drift SEQUENCE; measures transitions-to-recover per drift for context-latent `f(s,u,z)` (compounds) vs monolithic ±replay (pooled = liability under conflict) + offline-meta/scratch refs; controls `--family damping`, `--drift-mode revisit`, `--carve-sweep`. `run_value_carved_drift(cfg)`: #4d/#4e capacity-competition substrate with a DRIFTING value-irrelevant puck (`puck_phase`); veridical vs value-carved re-adaptation, logged at BOTH FM and CEM-MPC control level (capacity boundary `--fm-hidden`, `--static-puck`). |
| `compounding_drift_seeds_figure.py` | Local post-processing: aggregates the compounding-drift seeds into the threshold-free few-shot-R² compounding curve + transitions-to-recover + cumulative (`figures/compounding_drift_seeds/`). |
| `value_carved_drift_capacity_figure.py` | Local post-processing: the Cut-2 capacity boundary — value-carving's FM/control benefit vs FM width (`figures/value_carved_drift_capacity/fig_capacity_boundary.png`). |
| `online_value_loop.py` | **Cut 3 — the corrected full online loop.** `run_online_value_loop(cfg)`: `meta_curiosity_loop`'s two-timescale machinery, but the outer loop that self-tunes the explore/exploit balance `b=σ(θ)` is graded by a selectable TEACHER — `online_front` (value-relevant FM prediction-error over the goal corridor, the cerebellum→VTA messenger) or `online_ctrl` (downstream control = the parent's blind grader); `b<val>` fixed arms give the landscape. **Result**: the FM-error teacher has a clean INTERIOR optimum at b=0.5 where control is FLAT; `online_front` self-tunes toward it, `online_ctrl` wanders (self-tuning directional-but-noisy on the shallow bowl). `--task-geom corridor`, `--noise`, `--b-init`, `--drift-mode none` (stationary). **`--dense-teacher`** (added 2026-08-11, default OFF so every prior result reproduces): grade the outer loop on the epoch MEAN of the per-round corridor FM error the drive already computes, instead of the single epoch-end probe. `corridor_err_dense` + raw `dense_samples` are logged **unconditionally**, so any run measures its own within/between-epoch variance split. See [`teacher_snr/`](teacher_snr/README.md). |
| `online_value_loop_figure.py` | Local post-processing: the Cut-3 teacher contrast — the two b-landscapes (FM-error vs control) + the displaced-init self-tuning trajectories (`figures/online_value_loop_contrast/`). |

## Children

| Folder | Summary |
|---|---|
| [`teacher_snr/`](teacher_snr/README.md) | SNR analysis of the Cut-3 outer-loop teacher + the dense-teacher and Type-1 (stationary) runs. The b-landscape reproduces but is **shallower than the per-epoch nuisance** (SNR 0.80); a dense teacher does not help because **74% of the spread is between-epoch state variation** that within-epoch averaging cannot touch (ceiling 0.86); removing drift collapses absolute noise ~8× (SNR→1.40) without changing that structure, and the loop still does not climb (local SNR ≈0.62 at the operating point). Reads: the obstruction is the search procedure, not the teacher's form. |

## Figures (`figures/<tag>/`, mirrored from the volume)

| Directory | Content |
|---|---|
| `compounding_drift_seeds/` | Cut 1 aggregate (local): **`fig_fewshot`** (threshold-free compounding — factored climbs & locks, pooled collapses), `fig_ttr` (transitions-to-recover per drift), `fig_cumulative` (pooled memory is a liability). |
| `compounding_drift_{full_v1,full_s1,full_s2}/` | Cut 1 per-run, the 3 headline seeds: `fig1_ttr_per_drift`, `fig2_cumulative`, `fig3_recovery_curves`, `results.json`. |
| `compounding_drift_floor_v1/` | Cut 1 **damping-floor control** (no conflict) — pooling stays healthy, the factored advantage vanishes ⇒ conflict is the cause, not exposure. |
| `compounding_drift_revisit_v1/` | Cut 1 **revisit probe** (`--drift-mode revisit`, cycling a fixed φ bank) — factored remembers, both monolithic arms degrade. |
| `compounding_drift_carve_d{1,2,4,8,16}/` | Cut 1 **latent-width carve sweep** (`--carve-sweep`) — helps monotonically, saturating at d≈8. |
| `value_carved_drift_capacity/` | Cut 2 aggregate (local): **`fig_capacity_boundary`** — the carving benefit is capacity-gated and never reaches control. |
| `value_carved_drift_{comp_h24,comp_h32,comp_v1}/` | Cut 2 per-run across the capacity boundary (h=24/32/64): `fig1_ttr_pusher_vel`, `fig2_cumulative`, `fig3_puck_drop`, `fig4_control_ttr`. |
| `value_carved_drift_static_v1/` | Cut 2 **static-puck control** (`--static-puck`) — the value-irrelevant subspace held still. |
| `online_value_loop_contrast/` | Cut 3 aggregate (local): **`fig_landscape`** (interior optimum in FM-error vs flat control — the headline) and **`fig_selftune`** (front climbs, control random-walks). |
| `online_value_loop_{teacher_s0,teacher_s1,teacher_s2}/` | Cut 3 per-run b-landscapes, 3 seeds: `fig1_learned_b`, `fig2_control`, `fig3_occupancy`, `fig4_orderparams`, `results.json`. |
| `online_value_loop_{selftune15_s0,selftune15_s1,selftune15_s2}/` | Cut 3 displaced-init (`--b-init 0.15`) self-tuning trajectories, 3 seeds. |
| `online_value_loop_{firm2_s0,firm3_s0}/` | Cut 3 firm-up attempts (deepen the bowl / more epochs) — recorded negatives behind the "shallow-bowl REINFORCE is genuinely noisy" caveat. |
| `online_value_loop_dense_s{0,1,2}/` | Dense-teacher under drift, 3 seeds (see [`teacher_snr/`](teacher_snr/README.md)). |
| `online_value_loop_type1_s0/` | Dense-teacher, **stationary** DGP (`--drift-mode none`), 1 seed. |
| `*_smoke/` | Smoke-test outputs from the `--quick`/short configurations; kept for debugging, not results. |

## Modal volume layout (`mujoco-control-data`)

```
/data/compounding_drift/<tag>/    results.json, fig1_ttr_per_drift.png, fig2_cumulative.png, fig3_recovery_curves.png   (Cut 1)
/data/value_carved_drift/<tag>/   results.json, fig1_ttr_pusher_vel.png, fig2_cumulative.png, fig3_puck_drop.png, fig4_control_ttr.png   (Cut 2)
/data/online_value_loop/<tag>/    results.json, fig1_learned_b.png, fig2_control.png, fig3_occupancy.png, fig4_orderparams.png   (Cut 3)
```
