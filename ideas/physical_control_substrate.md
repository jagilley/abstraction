# A Physical-Control Substrate: MuJoCo as a controllable dynamics DGP between RHM and language

**Status**: Idea / research-direction memo (no experiment run yet)
**Date**: 2026-07-16
**Builds on**: [self_model_needs_a_loop.md](self_model_needs_a_loop.md), [efference_copy_cancellation.md](efference_copy_cancellation.md), [beliefs/trees/operators_not_footprints.md](../beliefs/trees/operators_not_footprints.md), [beliefs/dimensionality_expansion.md](../beliefs/dimensionality_expansion.md), [beliefs/trees/cerebellum_and_cognitive_architecture.md](../beliefs/trees/cerebellum_and_cognitive_architecture.md)
**Relates to (experiments)**: [a2a_forward/reaching/REACHING_INTERNAL_README.md](../experiments/a2a_forward/reaching/REACHING_INTERNAL_README.md), [a2a_forward/reaching/REACHING_LOOKAHEAD_README.md](../experiments/a2a_forward/reaching/REACHING_LOOKAHEAD_README.md), [a2a_forward/reaching/ACTIVE_VISION_README.md](../experiments/a2a_forward/reaching/ACTIVE_VISION_README.md), [a2a_forward/OOD_ROBUSTNESS_README.md](../experiments/a2a_forward/OOD_ROBUSTNESS_README.md), [rhm/RHM_SCULPTING_README.md](../experiments/rhm/RHM_SCULPTING_README.md)
**Prompted by**: reading Sunday Robotics' ACT-2 preview (2026-07-16) and asking where a "vanilla" scaling+imitation robot policy is *structurally* confounded — see conversation of the same date.

## One-liner

Add **MuJoCo (via MJX: JAX, GPU-parallel, Modal-native)** as a third substrate one rung of realism above RHM and below language — but use it the way we use RHM: as a **controllable *dynamics* DGP** whose shift we dial and sweep, **not** as a robotics benchmark to RL a policy to SOTA on. The payoff is to test the program's control-oriented claims — arity, composition, latent planning, and above all *operators-not-footprints robustness* — on genuinely continuous physical dynamics with real contact, while keeping the variable-control discipline that makes the toy results trustworthy.

## The one fork that decides whether this is worth it

There are two completely different things you can do with MuJoCo, and only one of them fits us:

- **Controllable dynamics DGP (ours).** Treat the physics parameters (friction, mass, actuator latency, contact stiffness, observation noise/occlusion) as *DGP knobs*, exactly as `(v,s,L,m)` are RHM's knobs. Get a policy the cheap way (scripted expert, MPC, short-horizon imitation, or a frozen behaviour policy), then **sweep the knob and measure how our forward-model/loop machinery degrades relative to a vanilla baseline.** This is the RHM move with continuous dynamics.
- **RL-to-SOTA benchmark (not ours).** PPO/SAC a policy to beat a leaderboard, then domain-randomize toward a real arm. This drags us into seed-variance hell (in deep RL seed *does* matter — our "don't multi-seed" rule inverts), sim2real engineering, and a competition with far better-resourced robotics labs. **No PPO-maxxing, no external benchmarking.** The moment we're tuning PPO we've lost the plot.

Everything below assumes the first.

## Why an intermediate substrate at all

The ladder we already have has a gap in the middle:

| Substrate | Dynamics | Do we know the DGP? | Ground-truth latents? | Realism |
|---|---|---|---|---|
| RHM / MNIST-reaching | discrete / constructed commands | **yes** (rules, DP optimum) | **yes** (ancestor tree, position) | low |
| **MuJoCo (MJX)** | **continuous, contact, real u** | partially (we set the physics) | **no** (no discrete latent tree) | medium |
| Language | continuous, opaque | no | no | high |

MuJoCo buys **genuinely continuous, controlled dynamics with a real command channel** without giving up the ability to *set and vary* the world — the property language denies us and the reason the whole RHM program exists (the [language-reduction dead-end](language_reduction.md): there was no clean way to change natural language's DGP). It is the first substrate where "the command genuinely controls the state" is *physically true* rather than constructed, which is precisely the condition our July pivot ("sensing is a trap; you need control; acting ≠ planning") says is required for a self-forecast to earn causal use.

## What it buys the specific claims

1. **Arity, made physical.** The `f(s,u)` vs `f(s)` result ([ACTIVE_VISION](../experiments/a2a_forward/reaching/ACTIVE_VISION_README.md), [REACHING_INTERNAL](../experiments/a2a_forward/reaching/REACHING_INTERNAL_README.md), and the RHM length-gen arity thread) currently rides on a glimpse location or a region edit — constructed "commands." A continuous **torque/velocity command → next state** is *the* canonical arity-2 problem, no construction. Predict: the arity-1 (command-blind) forward model floors on a reacher/pusher for the same structural reason it floored before, now with a real actuator. Cleaner, more legible, more convincing.
2. **The perfect-simulator control is free.** [REACHING_LOOKAHEAD](../experiments/a2a_forward/reaching/REACHING_LOOKAHEAD_README.md)'s central dissociation (best planner ≠ most veridical FM; the value-shaped co-trained FM beat even a perfect env-sim) needed a hand-built perfect simulator. **MuJoCo *is* the perfect simulator** — the veridicality-vs-control-usefulness axis comes for free, on real dynamics.
3. **Contact makes the residual physically meaningful.** "Residual = surprise; residual rank reflects DGP complexity" ([RESIDUAL_RANK](../experiments/rhm/RESIDUAL_RANK_README.md), MNIST) gets a clean physical test: the FM residual should **concentrate at contact events** (stiff, discontinuous, locally high-complexity), the way the MNIST residual was digit-discriminative and the language residual was diffuse. No policy, no RL — one afternoon.
4. **The lossy channel is available on demand.** Impose pixels-only + occlusion + sensor noise and MuJoCo reproduces the Stage-3c/3d regime (partial observability, stochastic dynamics) where **planning in latents beats re-observe-and-re-encode** ([RHM_SCULPTING](../experiments/rhm/RHM_SCULPTING_README.md)). Same construction cost as RHM, now on real physics.
5. **Credibility.** "We show this on MuJoCo continuous control" reads as more serious than "MNIST reaching" to reviewers and fellowships. Sociological, but the program treats communication as load-bearing.

## What it costs — eyes open (per the repo's temperament)

- **We lose RHM's ground-truth latent probes.** A physics rollout has no discrete ancestor tree to probe per-level recovery against, no DP optimum to grade planning by. This is the engine that made RHM productive. **Therefore: MuJoCo is a *second* substrate, not a migration. Keep RHM primary for anything that needs known latents; use MuJoCo for what needs continuous dynamics.**
- **The interesting regimes are still not the default.** MuJoCo's default is low-dim, fully-observed `qpos/qvel`. Our ideas bite under partial obs / irreversibility / lossy channel — which we *impose by hand*, exactly as in RHM. So MuJoCo's "realism" advantage over RHM is smaller than it looks *for our questions*; the value is the continuous **dynamics**, not free access to the hard regime.
- **MuJoCo is realistic for the forgiving tasks and bad at the confounding ones.** Rigid-body sim is trustworthy for articulated rigid control (reaching, locomotion) and poor at exactly the **deformable/contact-rich** tasks (cloth, cable, granular) that are the *maximally confounding* targets for a vanilla policy. Scope accordingly: use MuJoCo for arity/composition/robustness on rigid bodies; do **not** treat it as a proxy for the deformable confounders (folding, untangling), which need real hardware or specialized deformable sim.

## The intellectual core: sim2real is a distribution shift, so make it a *measurement instrument*

Two claims from the conversation, kept separate because they come apart on inspection:

**(a) Is the sim2real gap "fake"?** Partly. Decompose it:
- *Perceptual / rendering gap* — mostly fake and dissolving, and **operators-not-footprints already explains why**: pretrain the perceptual *operator* on the rich (real) side and rendering shift is a *footprint* you shed. Big visual pretraining + domain randomization have largely eaten this.
- *Dynamics / contact gap* — **not** fake, and here the human-plays-video-game analogy *inverts*. A human generalizing into a game goes from a **richer** training distribution (a lifetime of real physics) into a **poorer**, simplified, deliberately-legible one — interpolation/reduction, the *easy* direction — while carrying the real prior they pretrained on. A sim-only robot goes from the **poorer** side to the **richer** side (real contact/friction/latency tails it never saw) — *extrapolation to added complexity*, the *hard* direction. So "sufficient inspection" doesn't show the gap is fake; it shows the human case is the easy direction of the same gap, which is why it feels effortless. (Note: ACT-2's actual fix — pretrain on real human data — is operators-not-footprints in disguise: learn the operator on the rich side so the residual is small.)
- *The tails* — genuinely irreducible for contact/deformation sim can't model, and worst exactly where deformables live. Don't pretend otherwise.

**(b) Is the gap "exactly what we're learning to be robust to"?** Yes — and this is the gold. [OOD_ROBUSTNESS](../experiments/a2a_forward/OOD_ROBUSTNESS_README.md) already found the forward model's robustness is **distribution-invariant** while the autoencoder's collapses off-manifold. Dynamics shift is just distribution shift in the *transition operator*. So stop treating the gap as a nuisance to minimize (domain randomization) and treat it as an **instrument that measures our crown-jewel claim on physics**:

> **Sim2sim controllable dynamics-shift (the flagship).** Train on friction/mass/latency/contact ∈ [a,b]; test on [c,d]. Measure the **degradation slope** of a looped / cancellation / forward-model agent vs a vanilla reactive policy. Prediction (operators-not-footprints): the forward-model agent's slope is flatter, because it compressed the *transformation* (weight-defined, distribution-invariant), not the training *manifold* (data-defined, distribution-bound).

This is the only kind of sim2real that fits us: a **controllable** one, where we dial the shift and keep variable control, with no real robot and no RL-to-SOTA.

**The ambitious extension — non-stationarity for free.** Let the dynamics **drift during training** (friction slowly changing). Drifting dynamics is a **novelty generator**. Then ask the single biggest open question of the whole program ([dimensionality_expansion](../beliefs/dimensionality_expansion.md)): does the forward-model agent **expand its representable directions** (`R_act` / participation ratio) to track the drift, where a fixed-dynamics agent exhausts its frontier in one pass? Physics hands us non-stationarity that we otherwise have to bolt onto RHM by hand.

## Animals are biological robots — biomechanics as a source of hypotheses

The biology has *predicted the next experiment* at every turn of this program (the loop, efference copy, corollary discharge / "you can't tickle yourself," the dual forward/backward cerebellar role — [cerebellum_and_cognitive_architecture](../beliefs/trees/cerebellum_and_cognitive_architecture.md)). Every one of those hypotheses is, in origin, about **motor control of a body** — the cerebellum is a motor organ first, and efference copy is a motor-control mechanism before it is anything else. RHM and MNIST let us test the *computational* shape of those ideas; they cannot test them on a **body with real morphology** — muscles, tendons, passive dynamics, delays, redundant actuation.

MuJoCo is where the biological hypotheses meet biologically-shaped bodies:
- Musculoskeletal models exist (`dm_control` creatures, MyoSuite-style muscle actuators) — the substrate is *made* for testing forward-model motor control on tendon-driven, over-actuated, delayed systems, which is where the cerebellar story actually lives.
- **Morphology is an inductive-bias knob.** Passive dynamics and body structure offload computation from the controller (the "morphological computation" idea). That is a clean, controllable variable for asking how much of "robustness" is in the *controller's* self-model vs the *body's* mechanics — a distinction RHM literally cannot express.
- Efference copy / corollary discharge and sensorimotor delay are *native* to a physical body: the reason biology subtracts the forecast ([efference_copy_cancellation](efference_copy_cancellation.md)) is to cancel self-generated sensory consequences under delay. A delayed, embodied MuJoCo loop is the first place we can test cancellation where the delay is *real*, not a design choice.

This is not decoration. It is the same "biology as compass, not costume" method the program already runs — and it is a genuine motivation, because the hypotheses were born in motor control and have only ever been tested off-body.

## Concrete first cuts (cheapest first)

1. **Contact-residual structure** — scripted pusher, fit `f(s,u)`, show the residual concentrates at contact. No policy, no RL. Tests residual-reflects-DGP-complexity on physics. **✅ Done (2026-07-16)** — [experiments/mjc/contact_residual/](../experiments/mjc/contact_residual/README.md). Residual concentrates at contact (8.1× ratio, AUC 0.86), directionally worse (cosine 0.99 free / 0.82 contact — a scale-free control, not a magnitude artifact), scales with contact force (ρ=0.75), and lives in the velocity dims. Sharpened via an onset-aligned event-triggered average: the residual **spikes at the impact transition and decays while contact persists** — it is an *event* detector (regime transition), not a *state* detector. The residual-rank angle did **not** hold (contact residual is higher-rank than the near-degenerate free residual) — consistent with having retired "rank ∝ DGP complexity."
2. **Arity on torque** — reacher/pusher with a real command; reproduce the arity-1 floor. Clean, cheap, more legible than the MNIST/RHM versions. **✅ Done (2026-07-16)** — [experiments/mjc/arity_torque/](../experiments/mjc/arity_torque/README.md). With **i.i.d. commands** (killing the received-wisdom confound: max |corr(u,s)|=0.003), **arity beats resolution**: arity-2 pusher-velocity R²≈0.999 at *every* capacity while arity-1 is flat at ~0 across 8→256 hidden (70k params buys nothing) — the smallest command-aware model beats the largest command-blind one. The gap is **localized on the directly-actuated dims** (positions/puck predicted equally by both). And the **interventional** diagnostic (vary `u` at a fixed state via the perfect simulator): arity-2 reproduces the true command-driven Δs spread exactly, arity-1 captures zero by construction — observational→interventional made physical.
3. **Sim2sim robustness (the flagship)** — controllable dynamics-shift sweep; forward-model/cancellation vs vanilla; degradation slope. Operators-not-footprints on physics. **✅ Done (2026-07-16), but REFRAMED** — [experiments/mjc/dynamics_shift/](../experiments/mjc/dynamics_shift/README.md). The static *degradation-slope* framing turned out theoretically shaky (a shifted operator makes the stale FM itself wrong; per-step replanning makes it a near-null; it would mostly replicate a2a OOD_ROBUSTNESS). The sharp, new result the operator-intervention uniquely enables is the **factorization claim**: a dynamics shift corrupts *only* the world-model factor, so the model-based agent re-adapts from **reward-free** self-supervised interaction (re-fit the FM → ~50 transitions to the oracle ceiling) while a model-free policy is **flat** on reward-free data and needs ~240× more *reward-labeled* interaction. (Both agents are matched **committing motor-program generators** — MB rolls an adaptable FM, MF is an amortized net — so they degrade *identically* zero-shot; the confound where a reactive MF re-grounded every step is removed.) The effect is **gated by re-groundability** (`replan_every` ablation: the reward-free-adaptation benefit grows with open-loop commit horizon — feedback substitutes for a stale model only when you re-ground every step — and peaks at the FM's ~6–8-step composition horizon). Biologically: cerebellar feedforward control + prediction-error recalibration (self-supervised) vs basal-ganglia reward-driven policy relearning. **Open**: the commit-vs-reactive confound (fair MF open-loop variant), multi-seed, and a second (mass) shift.
4. *(then)* **Partial-obs latent planning** — pixels + occlusion; Stage-3c/3d analog on real dynamics.
5. *(ambitious)* **Drifting dynamics** — non-stationarity → dimensionality expansion; the program's biggest open question, for free.

## Practicalities

- **MJX (MuJoCo-XLA)** is JAX, vectorizes thousands of envs on one GPU, and drops straight into the `fer` JAX stack and Modal GPU workflow — Modal credits, fast iteration, off the MacBook Air.
- The sim is cheap; the only expensive thing is RL, which cuts 1–3 all avoid. Realistic cost ≈ 3–10× a reaching experiment, not a different category — *provided* we stay disciplined and never let it become a policy-optimization project.
- Keep every experiment framed as a **knob sweep with a vanilla baseline**, so the deliverable is always a *slope/dissociation*, never a single success number.

## What would kill it / open questions

- **If the arity/robustness results don't survive continuous dynamics** — e.g., the arity-1 floor washes out because MuJoCo's low-dim state makes command-averaging near-sufficient — that itself is informative (the RHM length-gen arc already found arity is load-bearing only when search can't substitute; a fully-observed re-groundable MuJoCo task may be the "search substitutes" regime). Design tasks that force imagined rollout (expensive/irreversible feedback), or the result will be a null for the same reason folding is easy.
- **Is a controllable sim2sim shift a fair model of real sim2real?** It tests robustness to *parametric* dynamics shift; it does not test the *structural/tail* gap (unmodeled contact modes). Claim only what the knob covers.
- **Does anything here need MuJoCo over a cheaper continuous toy** (a differentiable 2-link arm, a cartpole we write ourselves)? For cuts 1–2, maybe not — a hand-written continuous system may be cleaner. MuJoCo earns its cost at contact-richness, musculoskeletal bodies, and credibility. Use the cheapest substrate that exhibits the phenomenon; escalate to MuJoCo when contact or morphology is the point.
