# Active-Vision Looped ViT — the "missing u" (efference copy) test

**Idea doc**: [ideas/self_model_needs_a_loop.md](../../../ideas/self_model_needs_a_loop.md) (the "missing u" thread)
**Parent experiments**: [README.md](../README.md) (feedforward a2a arc), [LOOPED_README.md](../LOOPED_README.md) (the weight-shared looped arc, incl. the negative runnable-simulator probe this explains)
**Status**: Observational half **done, positive** (single seed): a genuine command `u` creates command-conditional dynamics that a command-blind forward model cannot predict *at any capacity*, collapsing when the command is removed. Causal-injection half **done, null with a diagnosis** (2026-07-10): the loop does not causally use the injected efference-copy forecast, and the reason (below) redirects the productive next step to a *control* task. **Control-regime port done, positive** (2026-07-10, 3 MNIST seeds + Fashion): on a foveal-reaching control task, an arity-2 forward self-model is *causally usable for planning* while an arity-1 one is stuck at the random floor at every capacity — the positive the perception-task injection could not produce. See ["Causal use in the control regime"](#causal-use-in-the-control-regime-the-null-flips-when-the-objective-is-endogenous-2026-07-10) below.
**Date**: 2026-07-09 (observational), 2026-07-10 (causal null + control-regime port)

## The claim being tested

From the discussion behind [self_model_needs_a_loop.md](../../../ideas/self_model_needs_a_loop.md): the a2a forward model predicts `s_{t+1} = f(s_t)` — it conditions only on the current state. A **cerebellar** forward model predicts `s_{t+1} = f(s_t, u_t)` — it also conditions on an **efference copy** of the command `u_t` the system is about to issue, available *before* the consequence. The a2a model "dropped the `u`."

The load-bearing consequence, in one sentence:

> **Arity, not resolution.** A forward model of a *controlled* system must take the command as a second input (be **arity-2**, `f(s, u)`). A command-blind **arity-1** model `f(s)` can only predict the command-*averaged* next state — and no amount of extra capacity ("resolution") can fix a missing input slot. Absorbing "what I tend to do" over many passes sharpens `f(s)`'s resolution; it never grows its arity.

("Arity" = the number of arguments a function takes.) This predicts *why* the looped runnable-simulator probe was negative on MNIST ([LOOPED_README](../LOOPED_README.md#near-manifold-self-map-extrapolation--is-the-self-map-a-runnable-simulator-2026-07-09)): plain MNIST classification is an **autonomous** system (`f(s)`, no command), so there was nothing to run a counterfactual over. Give the loop a real `u` and the arity-2 structure becomes both present and necessary.

## What we built

**`u` on MNIST = where to look (active vision).** `GlimpseLoopedViT` (`looped_vit.py`) is the confound-free looped ViT with one change: each step it reveals only a `G×G` window of the 7×7 patch grid at a sampled center `u_t`, and the shared operator integrates glimpses across `T=8` steps into the recurrent state (the *only* memory of past looks) before classifying:

```
s_0     = 0
s_{t+1} = G( s_t + glimpse(x, u_t) )      # glimpse reveals a 3x3 patch window at u_t
```

- **Fixed stochastic policy** — `u_t` uniform-random each step (no RL, no policy confound). A random glimpse is still a genuine sampled command (conditionally independent of the label given the state), which is all the arity argument needs.
- **`--full-view` = the no-`u` control** — reveal *every* patch each step (reveal mask all-ones), so the command is inert. Same architecture, same code path, command turned off. This isolates the single variable "is there a command."

**Two forward models predict the loop's own one-step update `Δ_t = s_{t+1} − s_t`** on the *frozen* main model (Run-5 scaling-sweep style, so every FM sees identical transitions):

- **`FM_state`** (arity-1) — input `s_t` only. Command-blind. This is today's a2a FM.
- **`FM_eff`** (arity-2) — input `s_t` + a **spatial efference-copy marker** for `u_t`. It knows *where* the model is about to look, but **not the pixel content there** — exactly a cerebellar forward model predicting a consequence it has not yet observed.

Both are swept across capacity (`d_head ∈ {4,8,16,32,64}`) to test arity-vs-resolution directly.

### Two design corrections the pilot forced (load-bearing gotchas)

1. **Predict the update, not the next state.** Predicting the full `s_{t+1}` let `FM_state` score ~0.94 just by echoing the slowly-varying carried state — the same degenerate self-decodability [LOOPED_README](../LOOPED_README.md) flagged. Predicting `Δ_t = s_{t+1} − s_t` (where the command's effect actually lives) exposed the real structure (`cmd_rel_spread` 0.04 → 0.24).
2. **The efference copy must be delivered *spatially*.** Encoding `u` as one global vector broadcast to all positions made `FM_eff` *worse* than `FM_state` — because the command acts by *localizing which positions update*, so a broadcast vector is just noise everywhere. Delivering it as a per-position "about-to-be-revealed here" marker (a function of `u` only, no content) flipped `FM_eff` above `FM_state`. This is itself a small confirmation of the framing: the command's information is *where*, so the self-model needs it *where*.

## Headline result (single seed; 4000 main / 3000 FM steps)

| | `cmd_rel_spread` (controlledness) | FM_state, d=4→64 | smallest-eff vs largest-state | cmd-conditional cos (eff / state) | on-traj cos (state/eff) | **arity beats capacity** |
|---|---|---|---|---|---|---|
| **MNIST-glimpse** (`u` present) | 0.237 | 0.886 → 0.901 (saturates) | 0.911 **>** 0.901 | **0.544** / 0 | 0.898 / 0.923 | **True** |
| **MNIST full-view** (no `u`) | **0.000** | 0.910 → 0.915 | 0.912 **<** 0.915 | −0.012 / 0 | 0.913 / 0.914 | **False** |
| **Fashion-glimpse** (loop harder) | 0.312 | 0.884 → 0.895 (saturates) | 0.936 **>** 0.895 | **0.665** / 0 | 0.891 / 0.943 | **True** |

Metric definitions: `cmd_rel_spread` = relative deviation of the update `Δ_t` across different commands at the same state (≈0 ⇒ autonomous). **cmd-conditional cos** = of the command-driven part of the update (`Δ(u) − mean_u Δ`), how much of its direction `FM_eff` recovers; `FM_state` is **structurally 0** because its best prediction is the command-mean. **arity beats capacity** = does the *smallest* `FM_eff` beat the *largest* `FM_state` on whole-update cos.

## Interpretation

**The no-`u` control is the whole test.** With full view the command is inert: `cmd_rel_spread` is *exactly* 0.000, command-conditional cos ≈ 0, `FM_eff` ≈ `FM_state` at every capacity, and arity does **not** beat resolution. The dynamics are autonomous, so the arity-1 self-map is sufficient — the exact regime the earlier looped runnable-simulator probe was stuck in. This is direct evidence that that probe was negative *because MNIST-classification is autonomous* (no command to simulate over), not because of scale.

**Add the `u` and all three predictions from the discussion fire:**

1. **A command creates command-conditional dynamics.** A quarter (MNIST, 0.24) to a third (Fashion, 0.31) of the update is command-driven. The loop is now a genuinely *controlled* system.
2. **Resolution cannot buy arity.** `FM_state` *saturates* with capacity (16× params buys +0.015 on MNIST) and the **smallest command-aware FM beats the largest command-blind one** (MNIST 0.911 > 0.901; Fashion 0.936 > 0.895). The `FM_eff` advantage is not parameters: its extra weights are ~128 (the reveal marker) against the 68K→137K sweep.
3. **The gap is cleanest where content can't confound it.** In the command-conditional subspace `FM_eff` captures **54% (MNIST) / 67% (Fashion)** of the true command-driven update direction, while `FM_state` is **0 at any capacity**.

**Two internal controls strengthen it:**

- **Same-architecture command-blindness cap.** `FM_state` predicts the *autonomous* full-view loop *better* (0.915) than the *controlled* glimpse loop (0.901). Identical architecture — so the 0.90 glimpse ceiling is genuine command-blindness, not undertraining or optimization failure.
- **The arity gap scales with loop-necessity.** Fashion's loop is more load-bearing than MNIST's, and Fashion shows a *larger* gap on every measure (spread 0.31 > 0.24; cmd-conditional 0.67 > 0.54; whole-update Δ 0.051 > 0.024; cmd-diff 0.50 > 0.36). The harder the task's sequential integration, the more the command matters — the same channel × loop-necessity interaction signature as the [LOOPED_README baseline battery](../LOOPED_README.md#baseline-battery--the-dependency-is-not-forecast-specific-but-its-scaling-is-2026-07-09).

**Why `FM_eff` is 0.54/0.67 and not ~1 is correct, not a shortfall.** `FM_eff` knows *where* it will look, not *what is there*, so it predicts the *expected* command-driven update before the reafference arrives. The residual (`1 − 0.54`) is the unobserved pixel content — exactly the cerebellar ceiling (a forward model predicts the expected consequence of a command, then the real sensation corrects it).

## What this does and does not show

- **Does show (observational premise):** giving a loop a real command turns it into a controlled system whose own dynamics require an arity-2, efference-copy-conditioned forward model to predict — and capacity provably cannot substitute for the command input. Removing the command collapses this entirely. This is the premise the whole "missing u" argument rests on, now demonstrated with a clean control.
- **Does not show:** that the model can run a **near-manifold counterfactual** ("what would I conclude if I looked *there* instead", off to the side) — untested. The causal-injection question *was* tested and came back null; see the next section.
- **Caveats:** single seed; the whole-update "arity beats capacity" margin is modest in absolute terms (content dominates the update ~65–75%), though the un-confounded command-conditional metric (0.54/0.67 vs 0) is large and Fashion's margins are ~2× MNIST's.

## Causal-injection arm: null, with a diagnosis that redirects the program (2026-07-10)

We closed the loop with the forecast injected each step (bounded scalar gate, co-trained), contrasting arity-2 vs arity-1: `CL_eff` injects `gate·FM_eff(s_t,u_t)`, `CL_state` injects `gate·FM_state(s_t)`, `OL` no injection. Two designs, both **null**:

1. **Same-step (k=1)** injection (`mnist_active_vision_causal.py`): the gate never opened (stayed at init ~0.016 in every condition/mode), dependency on the injection ≤0.003 everywhere. The `CL_eff`−`CL_state` dependency gap was +0.003 in glimpse mode vs exactly 0.000 in the full-view control (`FM_eff`≡`FM_state` there, bit-identical) — the interaction *direction* is right but sub-noise.
2. **Delayed feedback** (`mnist_active_vision_delay.py`), testing the hypothesis that the same-step null was "no feedback delay": command `u_t`'s glimpse *content* arrives `d` steps late while the forecast is available immediately. Delay sweep `d∈{0,1,2,3}` on Fashion showed **no rising dependency/gate curve** (eff−state dep {+0.000, +0.008, −0.003, +0.000}; gate never opens). `FM_eff`'s forecast is genuinely better and *improves* with delay (cos 0.94→0.955 vs `FM_state` 0.91→0.93) — **the loop simply refuses to use it.**

**Diagnosis (the useful part).** The only thing arity-2 knows over arity-1 is the command `u`, and **`u` is content-free** — it says *where* you look, never *what's there*. So arity-2's distinctive predictive power lives entirely in the content-free, command-driven slice of the dynamics. That slice steers the **trajectory** (hence the observational win — the update depends on where you look) but not the **answer** (the digit is the same regardless of glimpse order). Injecting a content-free forecast into a task whose answer is *content* supplies "the average effect of looking here," which is identical across images and so answer-irrelevant. The two arms reconcile with no contradiction: **the command matters for the trajectory, not the answer.** (Both FMs still condition on `s` and lean on it for world-content — the paper's dissociation is intact; the point is narrower, about arity-2's *marginal* input.)

**Where this points.** Causal use of an efference-copy forecast needs a task where the command's *consequences are themselves the objective* — a **control** task (you steer a state toward a goal, so predicting your action's effect is the point, as the cerebellum does for the body), not a **perception** task (the command merely gates access to exogenous content, as in glimpse-classification). Active-vision-MNIST gave us the `u` but not the control structure. A third injection tweak on the perception setup (e.g. early-commitment readout) is expected to null for the same content-blindness reason and is **not** recommended; the productive move is a looped control task.

## Causal use in the control regime: the null flips when the objective is endogenous (2026-07-10)

**Code**: `reaching_vit.py` (`ReachingLoopedViT`), `mnist_reaching.py`. **Status**: positive, 3 MNIST seeds + Fashion (single seed).

We ported the exact same content-glimpse machinery to a **foveal-reaching control task** and changed only two things: the **objective** (reach a cued goal patch, not classify the digit) and the **causal use** (a model-based *planner* queries the forward self-model, not a co-trained injection). This is the minimal diff that isolates the diagnosis above — *does the command matter for the answer, or only the trajectory?*

**The task.** The agent steers its fovea across the 7×7 patch grid to a goal patch `g`. The command `u_t` is now a **displacement action** (stay/up/down/left/right); the fovea position `p_t` is an **endogenous controlled state** (the action moves it); the objective (reach `g`) is defined over that self-controlled state. Each step the loop integrates a proprioceptive fovea marker at `p_t`, a goal marker at `g`, and — the load-bearing choice — a `G×G` **glimpse of image content** at `p_t`. The content is an *answer-irrelevant distractor* (the goal is a coordinate, not a digit), but it makes moving the fovea a large, training-independent change in the state (the same footprint the observational arm relied on). The efference copy stays **content-blind**: `FM_eff` gets only a marker at the *commanded next fovea* `p_{t+1}`, never the pixels there — exactly a cerebellar forward model predicting a consequence before the reafference arrives.

**Three phases**, all on one shared frozen controller (Run-5 discipline):
1. **Build the controller** — imitation-train `ReachingLoopedViT`'s policy head against a shortest-path oracle over random-action rollouts. Yields the **model-free ceiling** (oracle-action acc ≈ 0.99) and a state that encodes position + goal.
2. **Observational (freeze)** — train a linear fovea-position probe (the planner's value), then `FM_state` (arity-1, `f(s)`) vs `FM_eff` (arity-2, `f(s, u)`) across a capacity sweep, predicting the loop update `Δ_t`.
3. **Behavioral (freeze the FMs)** — a one-step model-based planner picks each action by rolling the forward self-model: `a_t = argmax_a value(s_t + FM(s_t, a))`, with `value(s) = −E_pos[dist(pos, g)]` read from the frozen probe. `FM_state`'s value is *constant in `a`* (no command slot) → a random walk at any capacity. That is the impossibility result made behavioral.

### Results (headline metric = `norm_progress`, the net fraction of start-distance to goal that is closed; 1.0 = reached, ≤0 = no progress)

**Observational** — the command is now a large, answer-relevant fraction of the dynamics, and only the arity-2 FM recovers it:

| run | `cmd_rel_spread` | `cmd-cond cos` (eff / state) | arity-beats-capacity |
|---|---|---|---|
| MNIST s42 | 0.61 | **0.85** / 0 | **True** |
| MNIST s43 | 1.05 | **0.79** / 0 | **True** |
| MNIST s44 | 0.63 | **0.77** / 0 | **True** |
| Fashion s42 | 0.55 | **0.80** / 0 | **True** |

**Behavioral** — planner `norm_progress` by forward-model capacity `d_head`:

| planner | d=4 | d=8 | d=16 | d=32 | d=64 |
|---|---|---|---|---|---|
| **`FM_eff`** (MNIST s42) | +0.74 | +0.73 | +0.73 | +0.79 | **+0.81** |
| **`FM_eff`** (MNIST s43) | +0.54 | +0.55 | +0.51 | +0.55 | +0.56 |
| **`FM_eff`** (MNIST s44) | +0.72 | +0.70 | +0.68 | +0.71 | +0.70 |
| **`FM_eff`** (Fashion s42) | +0.61 | +0.61 | +0.58 | +0.61 | +0.60 |
| **`FM_state`** (all runs) | −0.05 | −0.06 | −0.05 | −0.06 | −0.06 |

Reference controllers (all runs): **`model_free` ceiling ≈ +1.0**, **`random` ≈ −0.05**, **`planner_eff_shuffled` ≈ −0.40** (efference marker placed at the target of a *different* action — the inert/wrong-command control).

### Interpretation

1. **The dissociation is total and capacity-robust.** The *smallest* arity-2 planner (d=4, ≈ +0.6) beats the *largest* arity-1 planner (d=64, −0.06) in every run, on both datasets, all three seeds. "Arity beats resolution" is now a **behavioral impossibility**, not a fit margin: no amount of `FM_state` capacity buys a `u`-slot to plan over, so it never leaves the random floor.
2. **The null flips exactly where the theory says it should.** The *same content-glimpse machinery* that nulled under a classification objective is causally load-bearing under a reaching objective. The command is still content-free; what changed is that the objective is now defined over the endogenous state the command controls, so a content-free forecast is answer-relevant. This is direct confirmation of the causal-arm diagnosis ("the command matters for the trajectory, not the answer") — make the trajectory the answer and it matters.
3. **The shuffled control isolates the mechanism.** A present-but-wrong efference copy is *worse than random* (−0.40), so the planner is using the *correct* command→consequence correspondence, not merely the presence of an extra input.
4. **The forward model is genuinely content-blind, as intended.** `cmd-cond cos` ≈ 0.8, not 1.0: the residual is the unpredictable reafferent content at the target, the cerebellar ceiling again.

**Why `norm_progress`, not exact-hit success.** `FM_eff`'s exact-goal success rises with capacity on MNIST (0.16 → 0.44 as d: 16 → 64) but sits near the floor on Fashion (≈ 0.01) — *even though `FM_eff` makes +0.60 net progress there*. The harder dataset gives a noisier position probe (0.71 vs 0.77), so the planner homes toward the goal region but can't reliably nail the final patch. Net progress is the un-confounded control signal; exact-hit success additionally taxes probe precision.

**Caveats.** Greedy one-step planner (multi-step lookahead — the full runnable-simulator claim — untested). Fashion is single-seed. The **blank-nav** variant (a single 1-token fovea marker, no content distractor) is *fragile*: the command's share of the update swung 0.81 → 0.07 across configs, because a one-token marker is too small a footprint relative to the integrated state — which is itself a small confirmation that the command's effect must be a real fraction of the dynamics for arity to bite. The content glimpse supplies that footprint (and is the tightest minimal-diff from the null experiment), so it is the primary; blank-nav is not relied upon.

## Reproduction

```bash
cd experiments/
# The test (glimpse loop -> FM_state vs FM_eff capacity sweep -> counterfactual eval)
modal run --detach a2a_forward/reaching/mnist_active_vision.py::active_vision --dataset mnist
modal run --detach a2a_forward/reaching/mnist_active_vision.py::active_vision --dataset fashion_mnist
# The no-u control (command inert)
modal run --detach a2a_forward/reaching/mnist_active_vision.py::active_vision --dataset mnist --full-view

# Causal-injection arm (null): OL / CL_state / CL_eff, co-trained. --full-view = control.
for c in ol cl_state cl_eff; do
  modal run --detach a2a_forward/reaching/mnist_active_vision_causal.py::train_condition --condition $c --dataset fashion_mnist
done
# Delayed-feedback sweep (null): content arrives d steps late.
for d in 0 1 2 3; do for c in ol cl_state cl_eff; do
  modal run --detach a2a_forward/reaching/mnist_active_vision_delay.py::train_condition --condition $c --dataset fashion_mnist --content-delay $d
done; done

# Control-regime port (positive): foveal reaching -> FM sweep -> MB planner.
# --with-content = content-glimpse footprint (primary); omit it for the blank-nav ablation.
modal run --detach a2a_forward/reaching/mnist_reaching.py::reaching --with-content --dataset mnist --seed 42
modal run --detach a2a_forward/reaching/mnist_reaching.py::reaching --with-content --dataset fashion_mnist --seed 42
```

Results JSON under `/data/a2a_forward/{mnist_active_vision,mnist_active_vision_causal,mnist_active_vision_delay,mnist_reaching}/...`.

## Files

- `looped_vit.py::GlimpseLoopedViT` — active-vision looped ViT: `G×G` patch-window glimpse at command `u_t`, `full_view` no-`u` control, `one_step()` for counterfactual consequences, `content_delay` for the delayed-feedback arm.
- `mnist_active_vision.py` — observational arm: frozen glimpse loop + `FM_state` vs `FM_eff` capacity sweep + counterfactual eval (`cmd_rel_spread`, command-conditional cos, arity-beats-capacity).
- `mnist_active_vision_causal.py` — causal arm (same-step k=1 injection): OL / CL_state / CL_eff, co-trained, bounded gate, full-view interaction control. **Null.**
- `mnist_active_vision_delay.py` — delayed-feedback causal arm: content arrives `d` steps late, forecast injected anticipatorily; delay sweep. **Null.**
- `reaching_vit.py::ReachingLoopedViT` — control-regime port: looped ViT as a foveal-reaching controller (displacement actions, endogenous fovea state, goal marker, `with_content` glimpse footprint, `policy_head` ceiling).
- `mnist_reaching.py` — control-regime causal arm (**positive**): imitation ceiling → frozen fovea-position probe + `FM_state`/`FM_eff` capacity sweep → one-step model-based planner (`argmax_a value(s+FM(s,a))`), efference = marker at commanded next fovea. `FM_eff` planner makes strong net progress at every capacity; `FM_state` is stuck at the random floor.

## Next steps

1. ~~**Control-task port.**~~ **Done, positive** (2026-07-10) — see ["Causal use in the control regime"](#causal-use-in-the-control-regime-the-null-flips-when-the-objective-is-endogenous-2026-07-10). The efference-copy forward self-model is causally usable for planning; the command-blind one is not, at any capacity.
2. **Multi-step lookahead planner (the full runnable simulator).** The current planner is greedy one-step. Test whether `FM_eff` supports genuine N-step rollout (compose its own predictions), which would upgrade "usable for control" to "runnable simulator" — the claim the autonomous-MNIST probe (LOOPED_README) came back negative on, now expected positive because the system is controlled.
3. **Footprint sweep.** Glimpse-size sweep (`G ∈ {1,3,5}`) to confirm `cmd_rel_spread` and the arity gap track the degree of partial observation, and to map when the blank-nav (single-marker) footprint becomes sufficient.
4. **Behavioral counterfactual off-trajectory.** Can `FM_eff` predict the consequence of an action it does *not* take (plan a path without executing it)? The planner already queries counterfactual actions one step ahead; extending to imagined rollouts tests off-manifold simulation directly.
