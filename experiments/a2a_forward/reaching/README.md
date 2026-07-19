# Active-Control (Reaching) Arc — Efference-Copy Self-Models on a Control Task

**Idea doc**: [ideas/self_model_needs_a_loop.md](../../../ideas/self_model_needs_a_loop.md) (the "missing `u`" thread; internalization; the runnable-simulator question)
**Parent experiments**: [../README.md](../README.md) (the feedforward a2a arc), [../LOOPED_README.md](../LOOPED_README.md) (the weight-shared looped arc this ports to control)

> Per-experiment sections below give the goal + the headline finding only. **Full detail, tables, and the reproduction command for each experiment live in its linked `Full writeup`** — all three writeups are in this folder. File-by-file one-line purposes for the whole a2a experiment (including these) are in the parent **[../FILES.md](../FILES.md)**.

## Goal / the thesis

The feedforward a2a forward model predicts `s_{t+1} = f(s_t)` — it conditions only on the current state. A **cerebellar** forward model predicts `s_{t+1} = f(s_t, u_t)`: it also takes an **efference copy** of the command `u_t` the system is about to issue, *before* the consequence arrives. The a2a model "dropped the `u`." The load-bearing consequence, in one sentence:

> **Arity, not resolution.** A forward model of a *controlled* system must take the command as a second input (be **arity-2**, `f(s, u)`). A command-blind **arity-1** model `f(s)` can only predict the command-*averaged* next state — and no amount of extra capacity ("resolution") can fix a missing input slot.

This directory is the **active-control** turn of the program. It gives a weight-shared looped model a real command and makes the objective *endogenous* — a **foveal-reaching** task where the command is a displacement **action**, the fovea is a self-controlled state, and the goal is defined over that state. On such a task a self-forecast is *causally* load-bearing (you must predict your action's effect to plan), which the autonomous perception tasks (MNIST/Fashion classification) structurally could not force.

## Why reaching (the null that flipped)

The looped-ViT arc ([../LOOPED_README.md](../LOOPED_README.md)) found its runnable-simulator probe **negative on MNIST** — because MNIST-classification is an *autonomous* system (no command to simulate over). The active-vision experiment gave the loop a command (where to look) and confirmed arity-2 necessity *observationally*, but its causal-injection arm was **null**: the glimpse command is content-free (it says *where*, not *what*), so it steers the **trajectory** but not the **answer** (the digit is the same regardless of glimpse order). Injecting a content-free forecast into a content-answer task is answer-irrelevant. **Reaching removes that mismatch**: the objective is defined over the command-controlled state, so a content-free forecast becomes answer-relevant — and the null flips to a clean positive.

## Architecture

- **`ReachingLoopedViT`** (`reaching_vit.py`) — the weight-shared looped ViT as a **controller**. Each recurrent step integrates a proprioceptive fovea marker at `p_t`, a goal marker at the target patch `g`, and (in the primary `with_content` mode) a `G×G` glimpse of image content at `p_t` — an answer-irrelevant distractor that gives the command a large state footprint. `policy_head` = the model-free controller (the imitation ceiling). `ACTION_DELTAS` = {stay, up, down, left, right}. Shares `ViTBlock` with the parent `../vit.py`.
- **`GlimpseLoopedViT`** (in the parent **[../looped_vit.py](../looped_vit.py)**) — the active-vision variant used by the observational "missing `u`" experiments (`full_view=True` = the no-`u` control; `one_step()` for counterfactuals; `content_delay` for the delayed arm). It stays in the parent because `looped_vit.py` also houses the `LoopedViT` used by the LOOPED / CANCELLATION classification arc.
- **Forward self-models**: `FM_state` (arity-1, `f(s)`, command-blind — today's a2a FM) vs `FM_eff` (arity-2, `f(s, u)` via a spatial efference-copy marker placed at the commanded next fovea). Both are `TransformerForwardModel` from the parent `../forward_model.py`, swept across capacity (`d_head`) to test arity-vs-resolution directly.

## Experiments

### Active-vision "missing u" + control-regime port (2026-07-09 → 07-10)

**Full writeup**: [ACTIVE_VISION_README.md](ACTIVE_VISION_README.md)

- **Observational (positive)**: giving the loop a real command creates command-conditional dynamics an arity-1 FM cannot predict *at any capacity*; the no-`u` control collapses the effect (`cmd_rel_spread` 0.24 → 0.00), and the *smallest* arity-2 FM beats the *largest* arity-1 FM (MNIST 0.911 > 0.901 whole-update cos; 0.54/0.67 vs a structural 0 on the command-conditional slice). Scales ~2× on Fashion (loop more load-bearing).
- **Causal perception arm (null, with a diagnosis)**: injecting the content-free forecast under a *classification* objective doesn't help (gate never opens; dependency sub-noise), including under a feedback delay — the command steers the trajectory, not the answer. This diagnosis is what redirects the program to a control task.
- **Control-regime port (positive)**: on foveal reaching, a one-step model-based planner rolling `FM_eff` reaches the goal (+0.6–0.8 net progress at every capacity, 3 MNIST seeds + Fashion) while `planner_state` (arity-1) is stuck at the random floor (−0.06), and the shuffled-command control is *worse* than random (−0.40). "Arity beats resolution" becomes a **behavioral impossibility**.

### Internalized forecasting — endogenous self-forecast vs external planner (2026-07-10)

**Full writeup**: [REACHING_INTERNAL_README.md](REACHING_INTERNAL_README.md)

- Replaces the reaching positive's **external `argmax`** scaffold with an **endogenous** co-trained self-forecast (`int_plan` differentiable internal planner; `int_inject` fed-back injection). **Internalization makes the operator dramatically more *plannable*** — a fresh, decoupled external planner jumps +0.55 → +1.00 on the co-trained operator and CKA vs `mf` collapses to 0.06/0.25 — via *task-relevant* legibility (position decodability ↑, full-state forward-predictability ↓), not uniform. **Map and model coexist**: the running loop *depends* on the forecast (ablate → floor) while a linear readout on the frozen operator recovers a latent legible map (+0.43/+0.73). Arity impossibility survives internalization (`int_plan_state` at the floor); injection reproduces the a2a gate-opens-with-dependency signature.
- **Collusion-vs-selectivity + reading-ladder diagnostics (both datasets)**: the **planner** forecast is a *transferable, task-relevant self-model* (predicts the value-relevant direction 1.7–3.0× better than the junk; a fresh independent probe plans with it to +0.5–0.7) — its low full-vector Δ-cos is *correct selectivity*, not advisor drift; an arity-1 control shows *veridicality ≠ usefulness* (Δ-cos 0.99, plans at floor). Transferability is gated by *how much the forecast is read*: only a **raw-scalar** injection gate yields a private advisor; injection with the **thalamic-relay projection gate** (`CerebellarGate`, what the prior a2a arc actually used) or an **explicit readout** is transferable (+0.4 to +0.75) — so "injection privatizes the forecast" is *retracted* (the crude scalar gate was the confound).

### Multi-step lookahead — does the one-step forecast compose? (2026-07-11)

**Full writeup**: [REACHING_LOOKAHEAD_README.md](REACHING_LOOKAHEAD_README.md)

- The live next step from `REACHING_INTERNAL`: does the transferable one-step self-forecast **compose** into a runnable multi-step simulator? An **endogenous N-step MPC** rolls the FM in state space, decoding its own believed position each step (no env access), on three lookahead-forcing geometries (obstacles / serpentine maze / occluded-reveal). The composition question resolves into **two separable axes**:
  1. **Deep veridical composition IS achievable — and it is loop-gated.** With explicit multi-step-consistency pressure an FM composes to **near-full-state fidelity out to horizon 8** (maze pos-acc 0.99@n6), but *only on the plannability-shaped `int_plan` operator* — the identical FM collapses on the `mf` operator (0.52 → 0.05). The parent's "one-step self-model doesn't compose" was a one-step-*training* artifact; closing the loop reorganized the operator into **deeply composable dynamics**.
  2. **Veridicality ⊥ control-usefulness.** The value-shaped **co-trained FM is the *best* planner substrate (+0.51) despite the *worst* veridicality (0.21@n4)** — it even **beats a perfect-simulator env-MPC (+0.33)** with the same planner. A perfect-sim control confirms the maze is **planner-bound, not simulator-bound** (perfect sim also floors). What helps control is value-alignment, not full-state fidelity.

### Curiosity drive — the value-side atom (two-timescale value loop) (2026-07-17)

**Full writeup**: [CURIOSITY_DRIVE_README.md](CURIOSITY_DRIVE_README.md) · **Idea doc**: [ideas/two_timescale_value_loop.md](../../../ideas/two_timescale_value_loop.md)

A distinct thread that complements the arc above: where the reaching experiments build the *forward-model / inner loop*, this builds the **value / outer-loop** atom the a2a program never had — an **intrinsic learning-progress drive** (`r = −d‖e‖/dt`) that **drives where to attend** (active vision), rather than an extrinsic/stationary value. Chosen on active vision, not RHM, because the RHM specialization line shows RHM's depth frontier is not *allocation*-steerable (only a direct deep target moves it), so curiosity has no lever there.

- **Phase 1 (`curiosity_reaching.py`) — the drive atom, clean positive.** On a stationary struct/noise/blank arena, the LP drive is *distinguishable* from both controls: **surprise pins to irreducible noise** (noisy-TV), **min-surprise pins to blank** (dark room), while **LP alone rides the reducible frontier then releases it** once mastered (the LP signature). Noise error sits exactly at the 1/12 irreducible floor.
- **Phase 2 (`curiosity_drift.py`) — non-stationarity, instructive negative.** Under **abrupt** content drift naive LP is *worse than random* (blind to re-opened frontiers, distracted by noise fake-LP, forgetful); **continuous `morph` drift fixes the abrupt-jump pathology** but LP still can't beat uniform — because with reducible = ⅓ of the space there is **no scarcity** for concentration to exploit.
- **Phase 2b (`curiosity_scarcity.py`) — scarcity + fresh-FM ensemble, clean positive.** With a small reducible needle in a large noise field, the **ensemble (cross-model disagreement)** reliably finds it — over-sampling it **3.7×→8.5×** uniform (monotone with scarcity) and best needle-error at every level. Honest nuance: naive LP is **not qualitatively broken** (partially finds the needle; at 5% scarcity beats everything but the ensemble) — so the ensemble is a *consistent quantitative* win, not a strict qualitative necessity.
- **Follow-up → control substrate + MuJoCo redirection ([CURIOSITY_CONTROL_README.md](CURIOSITY_CONTROL_README.md)).** **Step 0 (clean positive)**: the outer-loop drive is the reducible-**disagreement magnitude**, not the LP derivative (drift-robust at every scarcity). **v1 (substrate-negative)**: porting the drive to the reaching controller (`curiosity_reaching_control.py`) exposed that the reaching-ViT FM *copies* its efference cue rather than *learning* where-you-land dynamics, so it can't learn+re-learn a control map online — an **incidental** (discrete fovea-on-token-grid) limit, not fundamental. **Redirection**: reward-free re-adaptation after a shift is already done on MuJoCo ([../../mujoco_control/README.md](../../mujoco_control/README.md) Cut #3); the un-done work is the **drive/compounding** and **disc-4 (value-shaping)**, with disc-4 → the MuJoCo pusher (puck = value-irrelevant; contact = capacity pressure).

## Where the arc stands

On the [self_model_needs_a_loop](../../../ideas/self_model_needs_a_loop.md) discriminators, **causal necessity now passes cleanly in the control regime**: an endogenous, causally-load-bearing self-forecast that measurably reorganizes the representation. The loop yields *both* a legible, deeply-composable simulator (available) *and* a value-shaped forecast the controller actually runs on (used) — the map-vs-model coexistence, now with the two axes causally dissociated. Consistent with the a2a gauge-symmetry / division-of-labor reframe: a perfect self-internalization would make the cerebellum redundant; biology keeps it, and so does this.

## Code & files

Everything lives in the `a2a_forward.reaching` subpackage. Imports of parent modules (`a2a_forward.shared`, `a2a_forward.vit`, `a2a_forward.forward_model`, `a2a_forward.looped_vit`) are unchanged by the move; the intra-arc import of `ReachingLoopedViT` is now `a2a_forward.reaching.reaching_vit`. One-line purposes for every file are in the parent [../FILES.md](../FILES.md).

| File | Role |
|---|---|
| `reaching_vit.py` | `ReachingLoopedViT` controller + `ACTION_DELTAS` — the shared operator for all three experiments |
| `mnist_active_vision.py` | Active-vision "missing `u`" observational arm (`FM_state` vs `FM_eff` capacity sweep; `--full-view` no-`u` control) |
| `mnist_active_vision_causal.py` | Same-step (k=1) causal-injection arm (**null**) |
| `mnist_active_vision_delay.py` | Delayed-feedback causal arm (**null**) |
| `mnist_reaching.py` | Control-regime causal test (**positive**) — imitation ceiling → FM capacity sweep → one-step MB planner |
| `mnist_reaching_internal.py` | Internalized forecasting — coupling conditions + collusion/selectivity + reading-ladder diagnostics |
| `mnist_reaching_lookahead.py` | Multi-step lookahead / composition — endogenous N-step MPC + multistep-consistency FMs + perfect-sim control |
| `curiosity_reaching.py` | **Curiosity Phase 1** — intrinsic learning-progress drive vs surprise/min-surprise/random on a stationary struct/noise/blank arena ([CURIOSITY_DRIVE_README](CURIOSITY_DRIVE_README.md)) |
| `curiosity_drift.py` | **Curiosity Phase 2** — content drift (swap/morph); naive LP under non-stationarity |
| `curiosity_scarcity.py` | **Curiosity Phase 2b** — scarce needle-in-noise arena + fresh-FM ensemble (disagreement) drive |

The shared `GlimpseLoopedViT` stays in the parent **[../looped_vit.py](../looped_vit.py)**.

## Reproduction

All commands run from `experiments/` (so `a2a_forward` is importable); Modal jobs use `--detach` for anything over ~2 min. Full per-experiment commands (sweeps, ablations, tag suffixes) are in each writeup.

```bash
cd experiments/

# Active-vision observational arm + no-u control
modal run --detach a2a_forward/reaching/mnist_active_vision.py::active_vision --dataset mnist
modal run --detach a2a_forward/reaching/mnist_active_vision.py::active_vision --dataset mnist --full-view

# Control-regime causal port (positive)
modal run --detach a2a_forward/reaching/mnist_reaching.py::reaching --with-content --dataset mnist --seed 42

# Internalized forecasting (round-1 conditions)
modal run --detach a2a_forward/reaching/mnist_reaching_internal.py::internal_reaching --dataset mnist

# Multi-step lookahead (baseline one-step FMs, all geometries)
modal run --detach a2a_forward/reaching/mnist_reaching_lookahead.py::lookahead_reaching --geometry obstacles
```

## Modal volume

Results save to the `language-reduction-data` volume under `/data/a2a_forward/{mnist_active_vision, mnist_active_vision_causal, mnist_active_vision_delay, mnist_reaching, mnist_reaching_internal, mnist_reaching_lookahead, curiosity_reaching, curiosity_drift, curiosity_scarcity}/…`. **These data-dir names are unchanged by the code move** — they are set inside each script and are independent of the source path, so prior results remain reproducible in place.

## Next steps

1. **A planner that exploits the deep simulator.** The maze is planner-bound: swap random-shooting for CEM/beam and the wall-blind Manhattan value for a geodesic-to-visible or *learned* value, and show the deeply-composable `int_plan` simulator solves hard lookahead the weak planner can't — turning the veridicality win (LOOKAHEAD Finding 1) into a behavioral one and testing whether value-alignment (Finding 2) is the lever.
2. **Active-control RHM / a gauge-free substrate (domain generality).** Everything here is MNIST/Fashion vision-control; the idea doc flags MNIST as special (deep supervision free). Define a command `u` + goal-conditioned objective on RHM (or language) and re-run the internalization + composition battery — the domain-generality test for the whole arc, and the gauge-free substrate the [../CANCELLATION_README.md](../CANCELLATION_README.md) Payoff-1 also needs.
3. **Seeds + off-manifold composition.** Multiple seeds; and test composition on *perturbed* (near-manifold-counterfactual) inputs the FM wasn't teacher-forced on — the idea doc's near-manifold self-counterfactual discriminator. A reaching variant hard enough that `mf` doesn't trivially solve it would also de-saturate the behavioral headline.
4. **Task-aligned selectivity follow-up.** In the reading ladder, the `proj` gate transfers for planning but is *not* subspace-selective, while `readout` is both — a small follow-up on whether task-aligned selectivity specifically requires reading *to an interpretable scalar*.
