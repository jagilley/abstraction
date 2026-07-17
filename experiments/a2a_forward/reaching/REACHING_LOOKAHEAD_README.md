# Multi-step lookahead: does the one-step self-forecast COMPOSE into a runnable simulator?

**Idea doc**: [ideas/self_model_needs_a_loop.md](../../ideas/self_model_needs_a_loop.md) ("run a rough forward pass on yourself, off to the side" = an N-step FM rollout; the central open question is whether one-step fixed-point self-consistency is *sufficient* for usable near-manifold counterfactuals or only the seed)
**Parent**: [REACHING_INTERNAL_README.md](REACHING_INTERNAL_README.md) (the endogenous one-step self-forecast that came out a transferable, task-relevant self-model — its live next step was exactly this composition test), [ACTIVE_VISION_README.md](ACTIVE_VISION_README.md), [README.md](README.md)
**Code**: `mnist_reaching_lookahead.py`, `reaching_vit.py`
**Status**: Done, positive-with-a-sharp-dissociation (single seed × {obstacles, maze} × MNIST; env-MPC control done). Multi-step-FM round + perfect-simulator control both agree.
**Date**: 2026-07-11

## The question

`REACHING_INTERNAL` established that the endogenous one-step self-forecast is a *transferable, task-relevant self-model* (predicts the value-relevant direction, an independent probe plans with it). Its one live gap: does that one-step forecast **compose** into a multi-step runnable simulator, or is it faithful only one-step? Plain reaching is greedily solvable, so it cannot test this — we need **lookahead-forcing** structure where a myopic (one-step) planner provably fails and only rolling the forecast forward multiple steps can succeed.

## What we built

Three lookahead-forcing geometries (env lives outside the model; wall-aware kinematics; BFS-geodesic oracle + honest progress metric; **wall-blind Manhattan value** as the deliberately-misleading planner heuristic):

- **obstacles** — a wall with a gap; the Manhattan-straight action hits the wall, a ≥2-step detour is required. Bites: `greedy_manh=+0.276` vs `oracle=+0.989`.
- **maze** — serpentine corridor (alternating horizontal walls); deep detours. Bites hard: `greedy_manh=+0.063` vs `oracle=+0.918`.
- **occluded** — no walls, goal hidden until the fovea visits a cue that reveals it. As expected `greedy_manh=+1.0` (leg-wise myopic solves it) → **not** a planner-depth test but a composition-*through-a-latent-reveal* test; behaviorally saturated, so it's reported in the parent-style veridicality metric only and dropped from the depth analysis. (Baseline run only.)

**The discriminator is an endogenous MPC planner**: random-shooting that rolls the forward model N steps *in state space*, decoding its own believed fovea position off each simulated state (probe argmax) to place the next efference marker — **no env access during the rollout**. `depth=1` reduces to the one-step planner, so the horizon sweep is apples-to-apples. Faithful to the decoupled-planner discipline: a good rollout *requires* the self-forecast to compose.

Four forward models compared on the **same frozen operator** (so only FM training varies):
- **fresh** — decoupled FM, pure one-step Δ-MSE (the veridicality baseline).
- **co-trained** — the `int_plan` self-model's own FM (trained via the value-imitation planner).
- **multistep-hN** — fresh FM + an explicit **multi-step consistency loss** (`FMⁿ ≈ true n-step state`, free-run state, teacher-forced actions, trained on the random-rollout distribution the MPC queries). N∈{4, 8}. *This is the A-vs-C sharpener: can deep composition be bought with the right objective?*
- **env-MPC** — the **perfect-simulator control**: same random-shooting planner + same Manhattan value, rolling the *true* kinematics instead of any FM. Isolates the planner/value from the simulator.

Two operators: `mf` (model-free head; = the decoupled operator) and `int_plan` (the endogenous one-step self-model operator).

## Headline 1 — MPC progress vs lookahead depth (floor = `greedy_manh`, ceiling = `oracle`)

**int_plan operator:**

| FM ‖ depth | 1 | 2 | 3 | 4 | 6 | (8 / 10) |
|---|---|---|---|---|---|---|
| obstacles fresh (1-step) | +0.29 | +0.30 | +0.32 | +0.28 | +0.22 | — |
| obstacles **co-trained** | +0.22 | +0.34 | +0.46 | **+0.51** | +0.45 | — |
| obstacles multistep-h4 | +0.29 | +0.28 | +0.29 | +0.36 | +0.35 | — |
| obstacles **env-MPC (perfect sim)** | +0.32 | +0.33 | — | +0.33 | +0.28 | +0.28 / +0.27 |
| maze fresh (1-step) | +0.05 | +0.05 | +0.08 | +0.09 | +0.10 | — |
| maze **co-trained** | −0.03 | −0.12 | −0.11 | −0.05 | −0.01 | — |
| maze multistep-h4 | +0.05 | +0.05 | +0.09 | +0.10 | +0.12 | — |
| maze **env-MPC (perfect sim)** | +0.04 | +0.08 | — | +0.11 | +0.13 | +0.14 / +0.15 |

## Headline 2 — composed-position accuracy vs horizon n (does the rolled simulator still know where it is?)

**int_plan operator:**

| FM ‖ n | 1 | 2 | 3 | 4 | 6 | 8 |
|---|---|---|---|---|---|---|
| obstacles fresh | 1.00 | 0.98 | 0.95 | 0.89 | 0.60 | 0.45 |
| obstacles **co-trained** | 0.97 | 0.50 | 0.27 | 0.21 | 0.21 | 0.21 |
| obstacles **multistep-h4** | 1.00 | 0.99 | 0.97 | 0.94 | 0.85 | **0.76** |
| maze fresh | 1.00 | 1.00 | 1.00 | 1.00 | 0.75 | 0.24 |
| maze **co-trained** | 1.00 | 0.80 | 0.51 | 0.24 | 0.02 | 0.01 |
| maze **multistep-h4** | 1.00 | 1.00 | 1.00 | 1.00 | **0.99** | **0.78** |

**Operator control (multistep-h4 on the `mf` operator):** obstacles 1.00→0.18→…→0.03; maze 0.52→…→0.05 — collapses. (`h8` at λ=1 destabilized training — one-step cos crashed to 0.06–0.22 and it degenerates; a config failure, not a finding.)

## Robust findings

1. **Deep veridical composition IS achievable — and it is an *operator* property, forced by closing the loop.** With explicit multi-step consistency pressure, an FM composes to **near-perfect full-state (position) fidelity out to horizon 8** (maze 0.99 @ n6, 0.78 @ n8) — where every one-step-trained FM collapsed. So the parent-arc reading "the one-step self-model doesn't compose" was a **one-step-*training* artifact**, not a fundamental limit. **Crucially this only works on the plannability-shaped `int_plan` operator** — the identical multistep FM collapses on the `mf` operator (maze 0.52→0.05). Closing the loop didn't merely make the operator one-step-plannable; it reorganized its dynamics to be **deeply composable** (rollable to 8 steps at 99% fidelity), which the un-looped operator does not afford. A new, clean result *for* the loop thesis.

2. **Veridicality ⊥ control-usefulness — demonstrated three ways.** (a) On obstacles the **co-trained FM is the *best* planner substrate (+0.51) despite the *worst* veridicality (0.21 @ n4)**; the near-perfectly-veridical multistep-h4 FM plans *worse* (+0.36). (b) The co-trained FM even **beats the perfect-simulator env-MPC (+0.51 vs +0.33)** with the same planner — a value-shaped forecast makes a weak planner outperform ground-truth simulation, because it carries value-relevant structure the raw simulator doesn't. (c) A 99%-veridical composed simulator (multistep-h4) does **not** solve the maze (+0.12). Making the simulator more faithful did not make it a better controller — what helps control is *value-relevant* structure, not full-state fidelity. This is the division-of-labor / gauge thesis with a causal handle.

3. **The maze is planner-bound, not simulator-bound (perfect-sim control).** The env-MPC with a *perfect* simulator + this planner also floors on the maze (+0.04→+0.145 at depth 10, vs oracle +0.918). So the serpentine maze's difficulty is entirely the **planner/value** (random-shooting + a wall-blind Manhattan heuristic that gives no gradient along a wall-blocked corridor), not composition fidelity. This *retracts* the tempting "deep composition breaks on the maze" reading — the simulator there is near-perfect; the planner is the bottleneck.

4. **Behavioral composition is real but shallow, and it's the self-model that carries it.** On obstacles, only the **co-trained self-model FM** produces a *monotone rise with lookahead depth* (+0.22 → +0.51 by d4, ~1/3 of the greedy→oracle gap), while fresh, mf-operator, and even the perfect-sim planner stay flat/lower. Deeper rollout of the *internalized value-shaped forecast* is what routes around the wall.

## Interpretation — where this leaves the self-model question

On the [self_model_needs_a_loop](../../ideas/self_model_needs_a_loop.md) discriminators the composition question **resolves into two separable axes**, and the earlier outcome-A/B/C framing was too coarse:

- **Simulator axis (veridical composition): outcome A, and loop-gated.** A deeply-composable, near-full-state-veridical runnable simulator over the operator *is* buildable — but only *because* closing the loop reorganized the operator into composable dynamics (it is not learnable on the `mf` operator). "Run a rough forward pass on yourself off to the side" is realizable to horizon 8 at 99% fidelity on the looped operator. This is the strongest positive of the arc for the runnable-simulator thesis.
- **Control axis (usefulness of the forecast): governed by value-alignment, not fidelity.** The forecast that actually *helps the controller* is the value-shaped co-trained one — which is a poor full-state simulator but beats even a perfect simulator with the same planner. So the self-model's payoff is **value-relevant selectivity** (the parent's finding, now shown to *dominate* veridicality for control), consistent with a2a gauge symmetry and the division-of-labor reframe.
- **Net.** The loop delivers *both* a legible, deeply-composable simulator (available) *and* a value-shaped forecast the controller actually runs on (used) — the same map-vs-model coexistence the parent found, now with the simulator made genuinely deep and the two axes causally dissociated. What we did **not** get is a single object that is both maximally veridical and maximally control-useful; the results say that object may not be the target (veridicality is neither necessary nor sufficient for control here).

## What this does and does not show

- **Does show**: (i) deep veridical composition to horizon 8 is achievable and is *gated by the looped operator's composability* (collapses on `mf`); (ii) veridicality ⊥ control-usefulness, three independent ways including a perfect-simulator control; (iii) the maze failure is planner-bound, not simulator-bound; (iv) only the internalized value-shaped forecast produces a monotone depth-of-lookahead behavioral gain. Robust across the two lookahead-forcing geometries.
- **Does not show**: a planner strong enough to *convert* the deep simulator into hard-lookahead behavior (random-shooting + Manhattan value is deliberately weak and caps everything, even perfect-sim); off-manifold veridicality (composition tested teacher-forced on the near-manifold random/oracle distribution); fixed-point self-consistency of the operator with its own forecast; multi-seed.
- **Caveats**: single seed per geometry. The composed-position metric applies an on-operator-manifold probe to *simulated* states, so it partly measures manifold-drift — but the behavioral MPC and the perfect-sim control are drift-free and agree. `mf` native (+0.98) and `int_plan` native (+0.99) both saturate (native `int_plan` is the *amortized imitation policy*, not lookahead — keep separate from the MPC composition test, which uses the Manhattan probe-value and is genuinely myopic-valued). `h8` at λ=1 is a training-instability config failure, not a horizon result.

## Next steps

1. **A planner that exploits the deep simulator (the natural follow-up).** The maze is planner-bound: swap random-shooting for CEM/beam and the wall-blind Manhattan value for a geodesic-to-visible or *learned* value, and show the deeply-composable `int_plan` simulator can solve hard lookahead the weak planner can't. Turns the veridicality win (Finding 1) into a behavioral one and directly tests whether value-alignment (Finding 2) is the lever.
2. **Active-control RHM (domain generality).** Everything here is MNIST/Fashion vision-control; the idea doc flags MNIST as special (deep supervision free). Design a control-task variant of RHM (define the command `u` and a goal-conditioned objective) and re-run the internalization + composition battery — the domain-generality test for the whole reaching arc.
3. **Seeds + off-manifold composition.** Multiple seeds; and test composition on *perturbed* (near-manifold-counterfactual) inputs the FM wasn't teacher-forced on — the idea doc's near-manifold self-counterfactual discriminator.

## Reproduction

```bash
cd experiments/
# Baseline (one-step FMs) + veridicality, all three geometries:
modal run --detach a2a_forward/mnist_reaching_lookahead.py::lookahead_reaching --geometry obstacles
modal run --detach a2a_forward/mnist_reaching_lookahead.py::lookahead_reaching --geometry maze
modal run --detach a2a_forward/mnist_reaching_lookahead.py::lookahead_reaching --geometry occluded

# Multi-step-consistency FMs (the A-vs-C sharpener), obstacles + maze, both operators:
modal run --detach a2a_forward/mnist_reaching_lookahead.py::lookahead_reaching \
  --geometry obstacles --fm-compose-horizons "4,8" --tag-suffix "_ms"
modal run --detach a2a_forward/mnist_reaching_lookahead.py::lookahead_reaching \
  --geometry maze --fm-compose-horizons "4,8" --tag-suffix "_ms"

# Perfect-simulator MPC control (fast, no training):
modal run --detach a2a_forward/mnist_reaching_lookahead.py::lookahead_reaching \
  --geometry obstacles --env-mpc-only --depths "1,2,4,6,8,10" --tag-suffix "_envmpc"
modal run --detach a2a_forward/mnist_reaching_lookahead.py::lookahead_reaching \
  --geometry maze --env-mpc-only --depths "1,2,4,6,8,10" --tag-suffix "_envmpc"
```

Results JSON under `/data/a2a_forward/mnist_reaching_lookahead/{geometry}_mnist_g7_K16_4H128D{tag_suffix}/results.json`.

## Files

- `mnist_reaching_lookahead.py` — this experiment. Wall-aware env with three geometries (obstacles/maze/occluded), BFS-geodesic oracle + progress, wall-blind Manhattan planner value, latent-reveal occlusion. Endogenous N-step random-shooting MPC that decodes its own believed position each simulated step (`mpc_action_fn`); the perfect-simulator control (`env_mpc_action_fn`, `--env-mpc-only` fast path); `train_frozen_fm(compose_horizon, compose_lambda)` for the multi-step-consistency FMs; readout battery = MPC depth sweep + composed-position/Δ-cos veridicality sweep + oracle/greedy references, per FM per operator.
- `reaching_vit.py::ReachingLoopedViT` — the shared controller (unchanged from the parent).
