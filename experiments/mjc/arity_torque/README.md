# Cut #2 — Arity on torque: the command slot is not a capacity problem

**Up**: [../README.md](../README.md) (mjc) · **Idea doc**: [../../../ideas/physical_control_substrate.md](../../../ideas/physical_control_substrate.md)
**Cousins** (the arity thread): [ACTIVE_VISION](../../a2a_forward/reaching/ACTIVE_VISION_README.md), [REACHING_INTERNAL](../../a2a_forward/reaching/REACHING_INTERNAL_README.md), RHM length-gen
**Code**: `arity_torque.py` (env: [`../pusher_env.py`](../pusher_env.py)) · File index: [FILES.md](FILES.md)
**Status**: done, single seed. **Date**: 2026-07-16.
**Builds on this**: [`dynamics_shift/`](../dynamics_shift/README.md) (Cut #3's model-based agent is an arity-2 FM built on this cut's idiom)

---

**Claim** (the a2a/RHM arity thread — now on a **real actuator**): a forward model of a *controlled* system must take the command as a second input. An arity-2 `f(s,u)` predicts command-driven dynamics; an arity-1 command-blind `f(s)` can only predict the command-averaged next state `E_u[Δs|s]`, and **no capacity buys the missing slot**. Adding the command turns an observational map (Pearl rung 1) into an interventional one (rung 2).

**The confound to kill** (the RHM "received-wisdom"/generator confound): if the scripted command `u` is a deterministic function of `s`, arity-1 recovers `u` from `s` and there is no gap. We collect with **i.i.d. commands** (`theta=1, seek_gain=0`), so `u_t ⊥ s_t` exactly — measured **max |corr(u,s)| = 0.003**. Arity is evaluated on **free-flight** test transitions (contact is [Cut #1](../contact_residual/README.md)'s regime; wall-bounce velocity reversals are unpredictable from anything and would cap *both* arities' ceiling, hiding that arity-2 fully captures the smooth command-driven dynamics).

**Apparatus**: 100K transitions (i.i.d. policy, 4.8% contact → 19,101 free-flight test transitions); capacity sweep hidden ∈ {8,16,32,64,128,256} × {arity-1, arity-2}, 2-layer MLP, Huber loss, **identical data — only the input differs**.

## Result (single seed)

| readout | finding |
|---|---|
| **arity beats resolution** (fig1) | arity-2 pusher-velocity R² = **0.998–0.999 at every capacity**; arity-1 **flat at ~0.00** across 8→256 hidden. The smallest arity-2 (232 params) beats the largest arity-1 (70,152 params) by ~0.99 R². The command slot is not a capacity problem. |
| **localized gap** (fig2) | positions and puck dims: R²≈1 for **both** arities (no command dependence). The entire arity gap sits on **pusher_vx / pusher_vy** — the directly-actuated dims. Command influence is physically legible. |
| **interventional** (fig3) | reset the **perfect simulator** to a fixed state, vary `u`, measure the true Δs spread: arity-2 reproduces it (0.0293 vs true 0.0289 on pusher velocity); arity-1 captures **exactly 0** by construction. Observational→interventional (rung 1→2), made physical. |

The pusher velocity change here is almost *entirely* command-driven (the actuator dominates drag at `gear=5`), so command-blindness is near-total failure — a maximally clean floor, cleaner and more legible than the glimpse/region-edit "commands" of the earlier arity results.

## Caveat

Under i.i.d./no-seek collection the puck is an **inert distractor** (rarely touched → tiny near-noise Δv), so its per-dim R² is noisy (`puck_vy` goes slightly negative at high capacity) and `R²_all` droops as capacity grows (big models overfit the tiny puck signal). This does not touch the conclusion — the actuated pusher dims are flat and definitive — but it is why the "all dims" panel (fig1 left) is not monotone. Single seed.

## Figures (`figures/arity_full_v1/`)

`fig1_capacity_sweep` (arity beats resolution) · `fig2_perdim_arity` (gap localized on pusher velocities) · `fig3_command_sensitivity` (interventional true-vs-captured).

## Reproduce

```bash
cd experiments/
# capacity sweep; ~4-5 min, auto-backgrounds
modal run mjc/arity_torque/arity_torque.py::arity_torque --quick     # smoke
modal run mjc/arity_torque/arity_torque.py::arity_torque --tag full_v1
```

Knobs are auto-exposed as CLI flags. The Modal function commits `results.json` + figures to `/data/arity_torque/<tag>/` on the `mujoco-control-data` volume; the local entrypoint mirrors them to `figures/arity_<tag>/`.
