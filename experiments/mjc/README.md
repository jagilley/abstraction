# MuJoCo control substrate — a controllable physical-dynamics DGP between RHM and language

**Idea docs**: [ideas/physical_control_substrate.md](../../ideas/physical_control_substrate.md) (the substrate program) · [ideas/two_timescale_value_loop.md](../../ideas/two_timescale_value_loop.md) (the value↔FM interface the later arcs test)
**Cousins**: a2a reaching arc ([REACHING_INTERNAL](../a2a_forward/reaching/REACHING_INTERNAL_README.md), [ACTIVE_VISION](../a2a_forward/reaching/ACTIVE_VISION_README.md)), [RHM sculpting](../rhm/RHM_SCULPTING_README.md)
**Status**: Cuts #1–#3 done; the value↔FM interface arc runs eight nodes deep and is live. File index: [FILES.md](FILES.md).

> **Naming**: this directory is `mjc/`, not `mujoco/` — a local package named `mujoco` would shadow the real `mujoco` pip package on Modal's `sys.path` and break every experiment. The Modal volume (`mujoco-control-data`) and app (`mujoco-control`) keep their original names, so all prior results stay addressable.

## What this is

A third controllable-dynamics substrate one rung of realism above RHM/MNIST-reaching and below language. We use it the way we use RHM — as a **DGP whose knobs we set and sweep**, not as a robotics benchmark to RL a policy to SOTA on. The physics parameters (friction, mass, damping, gear, arena, contact) are the knobs; the payoff is testing the program's control-oriented claims — arity, composition, latent planning, and operators-not-footprints robustness — on genuinely continuous dynamics with real contact, keeping the variable-control discipline that makes the toy results trustworthy.

**Discipline (the one fork that matters)**: controllable-dynamics DGP, **not** RL-to-SOTA. Policies are obtained the cheap way (scripted, MPC, short imitation); every experiment is a **knob sweep with a vanilla baseline**, so the deliverable is a *slope/dissociation*, never a single success number. No PPO-maxxing, no external benchmarking.

**Substrate choice**: plain **MuJoCo (CPU physics) + PyTorch forward models**, *not* MJX/JAX — for the current cuts the point is contact *fidelity* (plain MuJoCo's contact solver is the trustworthy one; MJX caps contact counts) and the data volume is tiny, and PyTorch reuses the a2a metric stack. MJX earns its cost later, at the sim2sim robustness sweep (thousands of parallel envs). Everything runs on Modal (workspace `chromatic`), volume `mujoco-control-data`.

## Shared machinery (lives at this node — everything below imports it)

**`pusher_env.py` — the planar pusher.** A gravity-free 2D world: a force-actuated `pusher` cylinder, a free `puck` cylinder, four static walls. Written as a ~30-line MJCF from a **DGP knob dict** so cuts can sweep the physics.

- **State** `s` (8-dim) = `[qpos(4), qvel(4)]` = `[pusher_x, pusher_y, puck_x, puck_y, pusher_vx, pusher_vy, puck_vx, puck_vy]`.
- **Command** `u` (2-dim) = motor force on the pusher's two slide joints, `u ∈ [-1,1]²`, scaled by `gear`.
- **Ground-truth contact** read straight from MuJoCo (`data.ncon` + `mj_contactForce`) — the physics analog of RHM's known latents; we never *infer* when a contact happened.
- Free-flight dynamics are near-linear (a damped particle under a known force); contact is a stiff, near-discontinuous map from the coarse state. That gap is the whole point.
- Cut-#1 world: `arena_half=0.8, pusher_r=puck_r=0.15, puck_mass=2.0` (others default: `friction=0.6, joint_damping=1.0, gear=5.0, timestep=0.002`); control at 100 Hz (`frame_skip=5`).
- Every later perturbation (`puck_field`, `field_patch`, `noise_patch`, `patch`, `rot_regions`) is **additive and off by default**, so the XML is byte-identical when unused and all prior cuts stay reproducible.

**`arm_env.py` — the planar N-link arm.** The second task family (see [`arm_substrate/`](arm_substrate/README.md)), matching `PusherEnv`'s API. Its perturbations (`payload_mass`, `curl_field`, `torque_rot`, `joint_noise`, `n_passive`/`tool_mass`/`goal_site`) are likewise additive and off by default.

**`shared.py`** — Modal image (mujoco 3.2.3 + torch), the `mujoco-control-data` volume, the `mujoco-control` app, `NumpyEncoder`. **`render_video.py`** — top-down sim view + live FM-residual side panel (headless OSMesa).

## The cuts (foundations)

- **Cut #1 — contact-residual structure** — [`contact_residual/`](contact_residual/README.md). An arity-2 FM's residual **concentrates 8.1× at contact** (AUC 0.86), is directionally worse there (cosine 0.99 free / 0.82 contact — a scale-free control, not a magnitude artifact), scales with contact force (ρ=0.75), and lives in the velocity dims. Sharpened by an onset-aligned event-triggered average: the residual **spikes at the impact transition and decays while contact persists** — an *event* detector (regime transition), not a *state* detector. Load-bearing gotcha: **Huber, not MSE**, or contact's heavy tail starves free-flight learning.
- **Cut #2 — arity on torque** — [`arity_torque/`](arity_torque/README.md). Under **i.i.d. commands** (killing the received-wisdom confound, max |corr(u,s)| = 0.003), **arity beats resolution**: arity-2 pusher-velocity R² ≈ 0.999 at *every* capacity while arity-1 is flat at ~0 across 8→256 hidden — the smallest command-aware model beats the largest command-blind one. The gap is localized on the **directly-actuated** dims, and the interventional diagnostic (vary `u` at a fixed state in the perfect sim) makes rung-1→rung-2 physical.
- **Cut #3 — operator intervention: reward-free re-adaptation** — [`dynamics_shift/`](dynamics_shift/README.md). The move this substrate uniquely enables: **intervene on the transition operator**, holding the task fixed. A dynamics shift corrupts only the world-model factor, so the MB agent recovers to the oracle ceiling **from ~50 reward-free transitions** while a protocol-matched committing motor-program MF is **flat** and needs ~240× more *reward-labeled* data. Gated by re-groundability (the `replan_every` ablation: the benefit grows with open-loop commitment and turns over at the FM's ~6–8-step composition horizon). **This cut's substrate is the base most later nodes build on**, and its biological reading (`replan_every` ≈ how ballistic a movement is) seeds the ballistic arc.

## The value↔FM interface arc

- **Value-shaping — the learning-layer control** — [`value_shaping/`](value_shaping/README.md). On the puck-present pusher, a value signal ("reach with the pusher") re-allocates a capacity-limited FM away from a reducible-but-value-irrelevant **nonlinear puck force field** and toward the pusher, with the contact impulse as a physically-present *irreducible* foil. The **stationary control** for everything meta that follows: it isolates "how value shapes the FM" at the learning layer. Robust core (the re-allocation); the "teeth" (capacity efficiency) are a fragile data-scarcity effect.
- **Directed reward-free re-adaptation — two instructive negatives** — [`directed_readapt/`](directed_readapt/README.md). Does a value-directed exploration drive re-adapt the FM in fewer transitions than Cut #3's undirected collection? A **global** shift has no scarcity (null); a **localized force-jet patch** has scarcity but the **disagreement drive is blind to a confident-prior shift** (under-visits the patch, 3 seeds). Lesson: the drive needs *scarcity* **and** *a signal that flags confident-wrong regions*. Motivates the pivot to phenomenon-first work.
- **Cut #4 — phenomenon-first meta-learning (#4 → #4e)** — [`meta_adapt/`](meta_adapt/README.md). The full FM↔value interchange arc in one node. **#4**: meta-train a fast adapter over a *distribution* of dynamics vs a pooled control — the 1-D damping **floor collapses** (gap ≈ 0, the RHM anchor), while **actuator-rotation conflict opens the gap monotonically with Φ** (meta ~8× more sample-efficient at Φ=π/2). **#4b**: an explicit context latent `z` **decodes the task parameter** (system-ID R² ≈ 1.0) — and system-ID ⊥ adaptation-benefit. **#4c**: value-directed identification is a **robust scarcity-gated negative** (a navigating info-MPC verifiably concentrates on the informative patch but doesn't beat a trivial max-‖u‖ heuristic). **#4d**: value-shaping × meta-conditioning are **two separable capacity levers** — a value-support-weighted objective produces a value-*caused* re-allocation that is veridicality⊥usefulness and buys control **only under capacity competition**. **#4e**: closes the reward loop (disc-4) — an outer loop whose only signal is control performance both **prefers** the puck-drop and **rediscovers V's support**, with a concrete wireheading gotcha (normalize the weight budget, or the loop games loss-scale rather than allocation).
- **Curiosity on control — the afferent drive + grounding** — [`curiosity_control/`](curiosity_control/README.md). The *afferent* complement to #4d/#4e's *efferent* value-shaping: an intrinsic reducible-surprise drive directs **where** to collect as a scarce reducible frontier **drifts**. The drive **tracks the moving frontier**, and the benefit is specifically **non-stationary** (it ties on a static frontier); value-relevance **gates usefulness** (it tracks off-path too, but control only benefits on-path). Pure curiosity has a **noisy-TV pathology on low-dim control** (disagreement *chases* aleatoric noise — a substrate inversion of the active-vision result), and **grounding fixes it**: explore + exploit as two additive drives, with an intermediate-balance optimum — the `e`-tap and `p`-tap wired as one system.
- **The fully-online two-timescale loop — obstructed on both levers** — [`online_value_loop/`](online_value_loop/README.md). Built the keystone the idea doc names (reward continuously shaping a *live* FM) both ways. **Afferent** (self-tune where to collect): the drive's tracking gradient is real, but CEM-MPC control is robust to it. **Efferent** (self-tune what the FM models): the value-shaping benefit is a slow, commit-dependent, pre-convergence *transient*. **Net: value is a slow/committed quantity — the two-timescale split is *forced*, not chosen — and its benefit is *adaptation speed*, not converged competence.** The obstruction is a positive confirmation, and it named the next experiment.
- **The two-timescale loop under perpetual drift** — [`drift_value_loop/`](drift_value_loop/README.md). That next experiment, three cuts deep. **(1) Compounding is real but comes from the *memory*, not the value-carving**: a task-conditioned context latent compounds across a drift sequence (few-shot R² 0.31→0.97) while pooled monolithic memory is a *liability* under conflict. **(2) Value-relevance carving is capacity-gated and control-robust.** **(3) The corrected teacher**: grade the meta-loop by the value-relevant **FM prediction-error** (the literal cerebellum→VTA messenger), not control, and the explore/exploit balance gets a clean interior optimum the loop can climb. **Net: the missing piece was the *teacher*, not the value *structure*.**
- **Ballistic control — the efferent bridge** — [`ballistic/`](ballistic/README.md). Why the FM→behavior bridge kept failing to transmit: **replanning is a blind grader** (it re-grounds past a stale model). This resolves it as a fact about the **control mode**, not the value. A **ballistic** (feedforward, open-loop, committed) controller — the biological regime, since you cannot replan faster than sensorimotor delay — makes the FM **~3× more behaviorally load-bearing** than reactive, and after a Type-2 drift **online reward-free FM re-adaptation restores ballistic competence ~4.3× more than reactive**. Lead reading: the cerebellar FM is the **reward-free-maintainable** asset *and* the **feedforward-critical** asset — the same object. Its child [`ballistic/directed/`](ballistic/directed/README.md) then lands the directed-collection ladder that 4a's confounded drive could not: a reward-free `learning-progress × visitation` signal recovers the oracle allocation.
- **The planar arm substrate — machinery, not a cut** — [`arm_substrate/`](arm_substrate/README.md). A **second task family**, built to retire the ballistic arc's "single-family" caveat and to replace the pusher's *manufactured* capacity competition with competition intrinsic to the plant. Verified against every precondition the arc requires: capacity binds intrinsically **only at n ≥ 5** ("nonlinear" and "capacity-hungry" are separate axes); ballistic transmits FM quality **6.0×** more than reactive; composition horizon **14–23 steps** (pusher ~6–8); and a **passive tool** puts #4d's capacity-competition boundary on one knob measured in **kg**, with `goal_site` flipping value-relevance while the transition operator stays byte-identical. Characterization only — no cut ported yet.

## Reproduce

Every node's own README carries its exact commands. The shared conventions:

```bash
cd experiments/
modal run mjc/<node>/<script>.py::<entrypoint> --quick        # smoke test
modal run mjc/<node>/<script>.py::<entrypoint> --tag <tag>    # full run
```

Use `modal run --detach` for anything over ~2 minutes, and invoke the function explicitly (not the module). Knobs are auto-exposed as CLI flags from each script's cfg dict. The Modal function commits `results.json` + figures to the volume; the local entrypoint mirrors them to that node's `figures/<tag>/` (those paths are `__file__`-relative, so they follow the script).

## Modal volume layout (`mujoco-control-data`)

Result paths are set **in-script** and are independent of source location — they did not change when the tree was reorganized, so every prior result stays addressable:

```
/data/contact_residual/<tag>/       /data/arity_torque/<tag>/         /data/dynamics_shift/<tag>/
/data/value_shaping/<tag>/          /data/directed_readapt/<tag>/
/data/meta_adapt/<tag>/             /data/meta_context/<tag>/         /data/meta_active/<tag>/
/data/meta_value_shaping/<tag>/     /data/meta_value_learn/<tag>/
/data/curiosity_control/<tag>/      /data/meta_curiosity_loop/<tag>/  /data/meta_value_online/<tag>/
/data/compounding_drift/<tag>/      /data/value_carved_drift/<tag>/   /data/online_value_loop/<tag>/
/data/ballistic_control/<tag>/      /data/ballistic_transmission/<tag>/ /data/ballistic_readapt/<tag>/
/data/directed_separability/<tag>/  /data/directed_ladder/<tag>/      /data/directed_loop/<tag>/
/data/arm_probe/<tag>/              /data/arm_capacity_sweep/<tag>/   /data/arm_tool_probe/<tag>/
```

## Next steps

0. **The live thread** is [`ballistic/directed/`](ballistic/directed/README.md) — directed reward-free collection under a genuinely ballistic controller, which is where the [`directed_readapt/`](directed_readapt/README.md) negative and the [`curiosity_control/`](curiosity_control/README.md) drive finally have a sighted grader. Its open piece: the conjunction never beats `lprog-only` in a loop.
1. **Port a cut onto the arm.** [`arm_substrate/`](arm_substrate/README.md) is characterized but carries no cut yet — porting the ballistic transmission/re-adaptation result there retires the "single-family" caveat every ballistic-arc claim currently carries.
2. **Cut #3 hardening** — multi-seed the noisy reward-driven REINFORCE slope; add a **mass-shift** operator (vs the damping collapse) as a second, qualitatively different shift.
3. **Cut #5 candidates** — partial-obs latent planning; dimensionality expansion; raising identification difficulty to the boundary where VoI should earn its keep (the [`meta_adapt/`](meta_adapt/README.md) #4c negative).
4. **Mechanism follow-up (optional)** — Cut #1's frame-skip sweep: the onset residual spike should *grow* with coarser control intervals (aliasing) while free-flight residual stays flat, separating the aliasing cause from the stiffness cause.
5. **Back-translate the arc into the belief tree** — the interface findings (value is slow/committed; the teacher, not the structure; the control mode gates transmission) are belief-shaped and not yet crystallized.
6. **Rendered videos** (`render_video.py`) — top-down sim view + live FM-residual side panel, one per cut, for communication.
