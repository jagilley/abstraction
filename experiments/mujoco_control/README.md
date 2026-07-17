# MuJoCo control substrate — a controllable physical-dynamics DGP between RHM and language

**Idea doc**: [ideas/physical_control_substrate.md](../../ideas/physical_control_substrate.md)
**Cousins**: a2a reaching arc ([REACHING_INTERNAL](../a2a_forward/reaching/REACHING_INTERNAL_README.md), [ACTIVE_VISION](../a2a_forward/reaching/ACTIVE_VISION_README.md)), [RHM sculpting](../rhm/RHM_SCULPTING_README.md)
**Status**: cut #1 (contact-residual structure) done, single seed. File index in [FILES.md](FILES.md).

## What this is

A third controllable-dynamics substrate one rung of realism above RHM/MNIST-reaching and below language. We use it the way we use RHM — as a **DGP whose knobs we set and sweep**, not as a robotics benchmark to RL a policy to SOTA on. The physics parameters (friction, mass, damping, gear, arena, contact) are the knobs; the payoff is testing the program's control-oriented claims — arity, composition, latent planning, and operators-not-footprints robustness — on genuinely continuous dynamics with real contact, keeping the variable-control discipline that makes the toy results trustworthy.

**Discipline (the one fork that matters)**: controllable-dynamics DGP, **not** RL-to-SOTA. Policies are obtained the cheap way (scripted, MPC, short imitation); every experiment is a **knob sweep with a vanilla baseline**, so the deliverable is a *slope/dissociation*, never a single success number. No PPO-maxxing, no external benchmarking.

**Substrate choice**: plain **MuJoCo (CPU physics) + PyTorch forward models**, *not* MJX/JAX — for the current cuts the point is contact *fidelity* (plain MuJoCo's contact solver is the trustworthy one; MJX caps contact counts) and the data volume is tiny, and PyTorch reuses the a2a metric stack. MJX earns its cost later, at the sim2sim robustness sweep (thousands of parallel envs). Everything runs on Modal (workspace `chromatic`), volume `mujoco-control-data`.

## The env — planar pusher (`pusher_env.py`)

A gravity-free 2D world: a force-actuated `pusher` cylinder, a free `puck` cylinder, four static walls. Written as a ~30-line MJCF from a **DGP knob dict** so later cuts can sweep the physics.

- **State** `s` (8-dim) = `[qpos(4), qvel(4)]` = `[pusher_x, pusher_y, puck_x, puck_y, pusher_vx, pusher_vy, puck_vx, puck_vy]`.
- **Command** `u` (2-dim) = motor force on the pusher's two slide joints, `u ∈ [-1,1]²`, scaled by `gear`.
- **Ground-truth contact** read straight from MuJoCo (`data.ncon` + `mj_contactForce`) — the physics analog of RHM's known latents; we never *infer* when a contact happened.
- Free-flight dynamics are near-linear (a damped particle under a known force); contact is a stiff, near-discontinuous map from the coarse state. That gap is the whole point.

Cut-#1 world: `arena_half=0.8, pusher_r=puck_r=0.15, puck_mass=2.0` (others default: `friction=0.6, joint_damping=1.0, gear=5.0, timestep=0.002`); control at 100 Hz (`frame_skip=5`).

---

## Cut #1 — Contact-residual structure (`contact_residual.py`)

**Claim** (ported from the a2a residual story to physics): an arity-2 forward model `f(s,u) → Δs` of a contact-rich system has a residual (actual − predicted) that **concentrates at contact events**, because free flight is easy/near-linear and contact is stiff/near-discontinuous. No policy, no RL: a scripted OU + seek-the-puck behavior policy generates `(s,u,s')`; MuJoCo hands us the contact labels.

**Apparatus**: 125K transitions (500 eps × 250 steps); train/test split by episode (no leakage). A small MLP FM `10→256×3→8` predicts normalized `Δs`, trained with **Huber loss** (δ=1.0). Regime label = **any contact** (`ncon>0`, incl. walls) vs genuine free flight.

### Result (single seed, 17.8% contact in test)

| axis | metric | finding |
|---|---|---|
| **concentration** | residual-norm ratio contact/free | **8.1×** (AUC **0.86** = P(contact resid > free resid); Cohen's d 0.72) |
| **scale-free control** | FM cosine(Δŝ, Δs) | **0.988 free vs 0.819 contact** — the FM genuinely models free-flight *direction*; contact is directionally worse, so the effect is **not** just "bigger Δs" |
| **dose-response** | Spearman(peak contact force, residual) | **ρ=0.75** within contact (over 6 orders of magnitude of force) |
| **structure** | per-dim residual | residual lives almost entirely in the **velocity dims** (positions barely move at contact) — the physics analog of MNIST's digit-discriminative residual |

### The sharpened claim: the residual is an *event* detector, not a *state* detector (`fig5`)

An onset-aligned event-triggered average (573 impacts) shows the residual **spikes at the free→contact impact** (peak 2.14, **5.5×** the pre-onset approach) and **decays back toward baseline within ~4 steps** (**6.8× peak/sustained**) — *even though* `P(in contact)` stays ~40–50% out to +20 steps. So being in contact does not keep the residual high; the residual marks the **regime transition**. Confirmed by the **separation-aligned** average: at contact *release* the residual just steps down with **no spike** — the asymmetry isolates the mechanism (a stiff, near-discontinuous velocity *impulse* at impact; release is gentle).

**Why this is expected** (three co-located mechanisms, all peaking at the boundary): (1) `(s,u)→Δs` is **near-discontinuous** at the contact boundary and a smooth MLP cannot represent a step; (2) **sub-timestep collision-timing aliasing** — whether the collision lands at substep 1 vs 5 within the 0.01 s interval is below the sampling resolution but changes the integrated Δv; (3) **stiff impact forces** (huge local `∂Δs/∂s`). Sustained pushing (bodies moving together, quasi-static) and free flight are both smooth regime *interiors* the FM fits well — the spike is the switching surface.

### Load-bearing methodological decisions

- **Huber loss, not MSE** (`huber_delta`, strict MSE generalization). Under plain MSE, raising the contact fraction *collapses* the contrast: contact's heavy-tailed Δs dominates the loss and starves free-flight learning (free cosine drops 0.99→0.93 at 22% contact). Huber caps the outlier gradient so the FM keeps modeling the **predictable** dynamics and contact becomes the residual — exactly the "FM predicts the expected trajectory; residual = surprise" framing. (At 10% contact, plain MSE is already clean — the tension only bites as contact fraction rises.)
- **Regime label = any contact, not puck-only.** Labeling only pusher↔puck contacts mislabels pusher/puck↔wall contacts as "free" and pollutes the baseline (free R² collapsed until this was fixed). `puck_force` is kept separately for the dose-response.
- **The scale-free control is cosine, not R² or ‖r‖/‖Δs‖.** Both magnitude-normalizers are ill-conditioned here: per-regime R² goes *negative* on free flight (free Δs sits below the FM's global error floor), and ‖r‖/‖Δs‖ blows up as ‖Δs‖→0. Cosine is scale-invariant by construction and cleanly shows contact predictions are directionally worse. (R² is still logged, with this caveat, in `results.json`.)

### Caveats

- Single seed (fine per our RL-free convention; the effect is large and was stable across 6 config iterations during setup).
- **eff-rank does NOT support "contact residual is lower-rank."** Contact eff-rank (3.60) is *higher* than free (2.52) — the free residual is near-degenerate (rank ~1, a small systematic error), while contact spreads across the 4 velocity dims. The honest structure story is the per-dim figure (velocity-localized), not a rank comparison. (We'd already retired "residual rank ∝ DGP complexity" as too many-variabled anyway.)

### Figures (`figures/full_v1/`)
`fig1_trajectory` (residual spikes at contact onset over one episode) · `fig2_distributions` (residual norm + cosine, contact vs free) · `fig3_perdim` (velocity-dim localization) · `fig4_doseresponse` (residual vs contact force, log-x binned) · **`fig5_onset_eta`** (the onset/separation event-triggered averages — the cleanest single figure).

---

## Cut #2 — Arity on torque (`arity_torque.py`)

**Claim** (the a2a/RHM arity thread — [ACTIVE_VISION](../a2a_forward/reaching/ACTIVE_VISION_README.md), [REACHING_INTERNAL](../a2a_forward/reaching/REACHING_INTERNAL_README.md), RHM length-gen — now on a **real actuator**): a forward model of a *controlled* system must take the command as a second input. An arity-2 `f(s,u)` predicts command-driven dynamics; an arity-1 command-blind `f(s)` can only predict the command-averaged next state `E_u[Δs|s]`, and **no capacity buys the missing slot**. Adding the command turns an observational map (Pearl rung 1) into an interventional one (rung 2).

**The confound to kill** (the RHM "received-wisdom"/generator confound): if the scripted command `u` is a deterministic function of `s`, arity-1 recovers `u` from `s` and there is no gap. We collect with **i.i.d. commands** (`theta=1, seek_gain=0`), so `u_t ⊥ s_t` exactly — measured **max |corr(u,s)| = 0.003**. Arity is evaluated on **free-flight** test transitions (contact is cut #1's regime; wall-bounce velocity reversals are unpredictable from anything and would cap *both* arities' ceiling, hiding that arity-2 fully captures the smooth command-driven dynamics).

**Apparatus**: 100K transitions (i.i.d. policy, 4.8% contact → 19,101 free-flight test transitions); capacity sweep hidden ∈ {8,16,32,64,128,256} × {arity-1, arity-2}, 2-layer MLP, Huber loss, **identical data — only the input differs**.

### Result (single seed)

| readout | finding |
|---|---|
| **arity beats resolution** (fig1) | arity-2 pusher-velocity R² = **0.998–0.999 at every capacity**; arity-1 **flat at ~0.00** across 8→256 hidden. The smallest arity-2 (232 params) beats the largest arity-1 (70,152 params) by ~0.99 R². The command slot is not a capacity problem. |
| **localized gap** (fig2) | positions and puck dims: R²≈1 for **both** arities (no command dependence). The entire arity gap sits on **pusher_vx / pusher_vy** — the directly-actuated dims. Command influence is physically legible. |
| **interventional** (fig3) | reset the **perfect simulator** to a fixed state, vary `u`, measure the true Δs spread: arity-2 reproduces it (0.0293 vs true 0.0289 on pusher velocity); arity-1 captures **exactly 0** by construction. Observational→interventional (rung 1→2), made physical. |

The pusher velocity change here is almost *entirely* command-driven (the actuator dominates drag at `gear=5`), so command-blindness is near-total failure — a maximally clean floor, cleaner and more legible than the glimpse/region-edit "commands" of the earlier arity results.

### Caveat
Under i.i.d./no-seek collection the puck is an **inert distractor** (rarely touched → tiny near-noise Δv), so its per-dim R² is noisy (`puck_vy` goes slightly negative at high capacity) and `R²_all` droops as capacity grows (big models overfit the tiny puck signal). This does not touch the conclusion — the actuated pusher dims are flat and definitive — but it is why the "all dims" panel (fig1 left) is not monotone. Single seed.

### Figures (`figures/arity_full_v1/`)
`fig1_capacity_sweep` (arity beats resolution) · `fig2_perdim_arity` (gap localized on pusher velocities) · `fig3_command_sensitivity` (interventional true-vs-captured).

---

## Cut #3 — Operator intervention: reward-free re-adaptation after a dynamics shift (`dynamics_shift.py`)

**The move MuJoCo uniquely enables** (vs language/RHM/MNIST): **intervene on the transition operator** (the dynamics) while holding the task fixed. We use it to test the

**Factorization claim**: a model-based agent factorizes control into (world-model: *what the world does*) × (planner/value: *what I want*). A dynamics shift corrupts **only the world-model factor**. So the MB agent re-adapts from **reward-free self-supervised** interaction — every `(s,u,s')` is a dense labeled example of the new dynamics, after which the fixed planner is immediately re-optimal — while a model-free amortized policy `π(s,g)→u` entangles both factors and gets **no signal** from reward-free transitions (it needs reward). This is NOT the static "degradation slope" (a shifted operator makes the stale FM itself wrong — theoretically shaky, and would just replicate a2a [OOD_ROBUSTNESS](../a2a_forward/OOD_ROBUSTNESS_README.md)); the new, mechanistic result is a **re-adaptation sample-efficiency dissociation**.

**Task**: goal-conditioned, **puck-free, momentum-dominated** pusher reaching (low `joint_damping` → the pusher glides → reaching needs anticipatory braking; state `[px,py,vx,vy]`, command = force, short horizon so overshoot is penalized; `build_xml(with_puck=False)`). **Agents** (both from the same d0 knowledge, fair): MB = arity-2 FM `f(s,u)→Δs` (cut #2 idiom, self-supervised on random d0 interaction) + a **CEM MPC** rolling the FM (value = running distance-to-goal + terminal-velocity penalty — dynamics-independent, **fixed across d0/d1**); MF = a **committing motor-program policy** `π(s,g)→(re-step action sequence)`, behavior-cloned from the MB planner's committed plans and executed with the **same `replan_every` open-loop commit as MB**. So both agents emit committed k-step motor programs from `(s,g)` under an *identical* feedback protocol — the *only* difference is MB generates the program by rolling an adaptable FM, MF by an amortized net (this removes the commit-vs-reactive confound; a motor-program net needs more capacity/data than a reactive policy to reach d0 competence). **Operator shift** d0→d1: a **friction/drag collapse** (`joint_damping 2.0 → 0.05`, "high-drag → ice") — at d0 the drag does the stopping so the cloned policy barely learns to brake; on ice both committed programs overshoot, but only the FM can be re-fit from reward-free sliding.

**Key design (load-bearing)**: the MB planner **commits to `replan_every`-step open-loop segments** (default 8) rather than re-planning every step. This is essential — with per-step re-grounding, feedback *substitutes* for the model (a stale FM is tolerated — the RHM "search-substitutes-for-the-model" regime), so the shift doesn't bite and there is nothing to recover (an early per-step run was a near-null). Committing makes the world-model **load-bearing**: a stale model's errors accumulate over the open-loop window and it overshoots, so the shift bites and reward-free refit genuinely recovers. Quantified by the ablation below.

### Result (single seed; metric = final ‖pusher_pos − goal‖, lower better)

| stage | finding |
|---|---|
| **d0 competence** | MB 0.036, MF 0.036 (both solve; random floor 0.65) |
| **d0 competence** | MB 0.036, MF 0.036 (**identical** — fair BC) |
| **shift bites both, equally** (fig1) | d1 zero-shot MB-stale 0.087, MF 0.087 (**identical** — both are committing motor-program generators, so the shift degrades them equally; the protocol is matched); FM-refit **oracle ceiling 0.021** |
| **reward-free re-adaptation** (fig3 — headline) | **MB recovers to the ceiling from ~50 reward-free transitions**; **MF is flat at 0.087** (reward-free `(s,u,s')` gives a program-generator no target — by construction) |
| **MF needs reward, ~240× more** | reward-driven REINFORCE reaches the ceiling only in the ~10⁴ reward-*labeled*-transition range (~500+ episodes), and noisily |

The factorization made physical: the operator shift corrupted only the world-model factor, and only the agent that *has* that factor can fix it from self-supervision.

### The `replan_every` ablation — the effect is gated by re-groundability (`run_replan_ablation`)

Sweeping the MB planner's open-loop commit length (FMs trained once — they are replan-independent — only the MPC eval varies):

| replan_every | 1 | 2 | 4 | 6 | **8** | 12 |
|---|---|---|---|---|---|---|
| stale FM | 0.022 | 0.029 | 0.048 | 0.068 | **0.093** | 0.084 |
| oracle FM | 0.016 | 0.015 | 0.014 | 0.017 | 0.021 | 0.033 |
| **reward-free benefit** (stale − refit) | **+0.005** | +0.012 | +0.030 | +0.049 | **+0.071** | +0.048 |

The reward-free-adaptation benefit **grows with commitment horizon** (feedback tolerates a stale model when you re-ground every step; committing does not), and the refit **tracks the oracle throughout** (~5000 reward-free transitions recover ~all of a fully-adapted model's value at every commitment length). **Bonus — the turnover at 12**: even the *oracle* degrades (0.033) because no one-step FM composes over a 12-step open-loop rollout — the **~6–8-step composition horizon** from [REACHING_LOOKAHEAD](../a2a_forward/reaching/REACHING_LOOKAHEAD_README.md) reappearing on physics. The value of an accurate model peaks where the model is maximally load-bearing *and still veridical enough to compose*.

### Caveats
- **Protocol-fair (confound fixed).** MF is a committing motor-program policy with the *same* open-loop commit as MB, so both degrade **identically** at zero-shot (0.087) — the earlier commit-vs-reactive confound (a reactive MF that re-grounded every step) is removed. The agents differ *only* in how they re-adapt. (`figures/dynshift_full_v2/` holds the earlier reactive-MF run for the record; `full_v3` is the fair one.)
- **Reward-driven MF is noisy.** Minimal, un-tuned REINFORCE on the committing (sequence-output) policy has high variance — the reward-driven curve reaches the ceiling in the ~10⁴-transition range but non-monotonically. Only the order-of-magnitude gap is claimed, not a precise multiplier. Per "no RL-maxxing", it wasn't tuned further. Single seed.
- **Adaptation, not zero-shot robustness.** The result is strictly about *adaptation efficiency* — both agents degrade equally zero-shot; we do not claim MB is more robust zero-shot.

### Biological reading
`replan_every` ≈ **how ballistic the movement is**: fast movements outrun sensory feedback (~50–150 ms) and must run open-loop off an internal forward model (the cerebellum); slow movements are feedback-guided and need only a weak model. Making the FM load-bearing = the ballistic regime, where a stale model overshoots (**cerebellar dysmetria**). Reward-free FM refit = **cerebellar recalibration from sensory prediction error** (available from any movement, no reward); reward-driven MF = policy relearning by reinforcement (basal ganglia). The dissociation mirrors the cerebellum (fast, self-supervised, prediction-error-driven) vs basal-ganglia (slow, reward-driven) division of labor.

### Figures (`figures/dynshift_full_v3/`, `figures/dynshift_replan_abl_v1/`)
`fig1_shift_bites` · `fig2_recovery_curves` (reward-free vs reward-driven, split by signal type) · **`fig3_sample_efficiency`** (both agents on one interaction axis — the money figure) · `fig_replan_ablation` (benefit vs commitment horizon).

---

## Reproduce

```bash
cd experiments/
# smoke test (~1 min on Modal; builds the mujoco+torch image on first run)
modal run mujoco_control/contact_residual.py::contact_residual --quick
# full run (~90 s): 125K transitions, 60 epochs
modal run mujoco_control/contact_residual.py::contact_residual --tag full_v1

# cut #2 — arity on torque (capacity sweep; ~4-5 min, auto-backgrounds)
modal run mujoco_control/arity_torque.py::arity_torque --quick     # smoke
modal run mujoco_control/arity_torque.py::arity_torque --tag full_v1

# cut #3 — dynamics-shift re-adaptation (~2 min on L4) + the replan_every ablation
modal run mujoco_control/dynamics_shift.py::dynamics_shift --quick        # smoke
modal run mujoco_control/dynamics_shift.py::dynamics_shift --tag full_v3  # replan_every=8, fair committing MF
modal run mujoco_control/dynamics_shift.py::dynamics_shift --tag full_v1 --replan-every 1  # muted (re-groundable) endpoint
modal run mujoco_control/dynamics_shift.py::replan_ablation               # commit-horizon sweep
```

The Modal function commits `results.json` + figures to the volume; the local entrypoint also mirrors them to `mujoco_control/figures/<tag>/`. Knobs (`--seek-gain`, `--frame-skip`, `--huber-delta`, `--hidden`, `--epochs`, …) are auto-exposed as CLI flags.

## Modal volume layout (`mujoco-control-data`)

```
/data/contact_residual/<tag>/   results.json, fig1..fig5 .png
/data/arity_torque/<tag>/        results.json, fig1..fig3 .png
/data/dynamics_shift/<tag>/      results.json, fig1..fig3 .png (or fig_replan_ablation.png)
```

## Next steps

1. ~~**Cut #2 — arity on torque**~~: **Done** — arity beats resolution on a real actuator (arity-2 pusher-vel R²≈0.999 at every capacity, arity-1 flat at ~0), gap localized on the actuated dims, interventional command-sensitivity airtight.
2. ~~**Cut #3 — operator-intervention re-adaptation** (the reframed flagship)~~: **Done (2026-07-16)** — protocol-fair reward-free re-adaptation dissociation (MB & MF are matched committing motor-program generators degrading identically to 0.087; MB recovers from ~50 reward-free transitions; MF flat; MF needs ~240× more, reward-labeled), gated by re-groundability (`replan_every` ablation), peaking at the FM composition horizon. Reframed away from the shaky static "degradation slope." See above.
3. **Cut #3 hardening**: multi-seed the (noisy) reward-driven REINFORCE slope; add a **mass-shift** operator (vs the damping collapse) as a second, qualitatively different shift. (The commit-vs-reactive confound is now fixed — MF is a committing motor-program policy matched to MB.)
4. **Cut #4/5** — partial-obs latent planning; drifting-dynamics → dimensionality expansion (the frontier open question). See the idea doc.
5. **Mechanism follow-up (optional)**: cut #1 frame-skip sweep — the onset residual spike should *grow* with coarser control intervals (aliasing) while free-flight residual stays flat, separating the aliasing cause (#2) from the stiffness cause (#3).
6. **Rendered videos** (`render_video.py`): top-down sim view + live FM-residual side panel, one per cut, for communication.
