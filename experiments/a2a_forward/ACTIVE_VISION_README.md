# Active-Vision Looped ViT — the "missing u" (efference copy) test

**Idea doc**: [ideas/self_model_needs_a_loop.md](../../ideas/self_model_needs_a_loop.md) (the "missing u" thread)
**Parent experiments**: [README.md](README.md) (feedforward a2a arc), [LOOPED_README.md](LOOPED_README.md) (the weight-shared looped arc, incl. the negative runnable-simulator probe this explains)
**Status**: In progress. The **observational** half is done (single seed): a genuine command `u` creates command-conditional dynamics that a command-blind forward model cannot predict *at any capacity*, and this collapses when the command is removed. The **causal/behavioral** half (does the loop *use* an efference-copy-conditioned forecast; can it run a counterfactual) is not yet run.
**Date**: 2026-07-09

## The claim being tested

From the discussion behind [self_model_needs_a_loop.md](../../ideas/self_model_needs_a_loop.md): the a2a forward model predicts `s_{t+1} = f(s_t)` — it conditions only on the current state. A **cerebellar** forward model predicts `s_{t+1} = f(s_t, u_t)` — it also conditions on an **efference copy** of the command `u_t` the system is about to issue, available *before* the consequence. The a2a model "dropped the `u`."

The load-bearing consequence, in one sentence:

> **Arity, not resolution.** A forward model of a *controlled* system must take the command as a second input (be **arity-2**, `f(s, u)`). A command-blind **arity-1** model `f(s)` can only predict the command-*averaged* next state — and no amount of extra capacity ("resolution") can fix a missing input slot. Absorbing "what I tend to do" over many passes sharpens `f(s)`'s resolution; it never grows its arity.

("Arity" = the number of arguments a function takes.) This predicts *why* the looped runnable-simulator probe was negative on MNIST ([LOOPED_README](LOOPED_README.md#near-manifold-self-map-extrapolation--is-the-self-map-a-runnable-simulator-2026-07-09)): plain MNIST classification is an **autonomous** system (`f(s)`, no command), so there was nothing to run a counterfactual over. Give the loop a real `u` and the arity-2 structure becomes both present and necessary.

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

1. **Predict the update, not the next state.** Predicting the full `s_{t+1}` let `FM_state` score ~0.94 just by echoing the slowly-varying carried state — the same degenerate self-decodability [LOOPED_README](LOOPED_README.md) flagged. Predicting `Δ_t = s_{t+1} − s_t` (where the command's effect actually lives) exposed the real structure (`cmd_rel_spread` 0.04 → 0.24).
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
- **The arity gap scales with loop-necessity.** Fashion's loop is more load-bearing than MNIST's, and Fashion shows a *larger* gap on every measure (spread 0.31 > 0.24; cmd-conditional 0.67 > 0.54; whole-update Δ 0.051 > 0.024; cmd-diff 0.50 > 0.36). The harder the task's sequential integration, the more the command matters — the same channel × loop-necessity interaction signature as the [LOOPED_README baseline battery](LOOPED_README.md#baseline-battery--the-dependency-is-not-forecast-specific-but-its-scaling-is-2026-07-09).

**Why `FM_eff` is 0.54/0.67 and not ~1 is correct, not a shortfall.** `FM_eff` knows *where* it will look, not *what is there*, so it predicts the *expected* command-driven update before the reafference arrives. The residual (`1 − 0.54`) is the unobserved pixel content — exactly the cerebellar ceiling (a forward model predicts the expected consequence of a command, then the real sensation corrects it).

## What this does and does not show

- **Does show (observational premise):** giving a loop a real command turns it into a controlled system whose own dynamics require an arity-2, efference-copy-conditioned forward model to predict — and capacity provably cannot substitute for the command input. Removing the command collapses this entirely. This is the premise the whole "missing u" argument rests on, now demonstrated with a clean control.
- **Does not yet show (the causal/behavioral half):**
  1. that the **loop itself causally uses** an efference-copy-conditioned forecast (inject `FM_eff` vs `FM_state` into the loop and test acceleration/halting or dependency that scales with loop-necessity);
  2. that the model can run a **near-manifold counterfactual** ("what would I conclude if I looked *there* instead", off to the side, without committing the glimpse) — the runnable-simulator capability the no-`u` loop structurally failed.
- **Caveats:** single seed; the whole-update "arity beats capacity" margin is modest in absolute terms (content dominates the update ~65–75%), though the un-confounded command-conditional metric (0.54/0.67 vs 0) is large and Fashion's margins are ~2× MNIST's.

## Reproduction

```bash
cd experiments/
# The test (glimpse loop -> FM_state vs FM_eff capacity sweep -> counterfactual eval)
modal run --detach a2a_forward/mnist_active_vision.py::active_vision --dataset mnist
modal run --detach a2a_forward/mnist_active_vision.py::active_vision --dataset fashion_mnist
# The no-u control (command inert)
modal run --detach a2a_forward/mnist_active_vision.py::active_vision --dataset mnist --full-view
```

Results JSON: `/data/a2a_forward/mnist_active_vision/{mnist,fashion_mnist}_{glimpse,full_view}_g3_T8_4H128D/results.json`.

## Files

- `looped_vit.py::GlimpseLoopedViT` — active-vision looped ViT: `G×G` patch-window glimpse at command `u_t`, `full_view` no-`u` control, `one_step()` for counterfactual consequences.
- `mnist_active_vision.py` — trains the frozen glimpse loop (fixed stochastic policy), the `FM_state` vs `FM_eff` capacity sweep (spatial efference-copy marker), and the counterfactual eval (`cmd_rel_spread`, command-conditional cos, command-diff, arity-beats-capacity).

## Next steps

1. **Causal arm — does the loop *use* the efference copy?** Close the loop with the forecast: `s_{t+1} = G(s_t + glimpse(x,u_t) + gate·FM_eff(s_t,u_t))` vs the same with `FM_state`. Test whether the command-conditioned forecast buys acceleration/halting (reach the answer in fewer glimpses) or dependency that scales with loop-necessity — the arity-2 analog of the LOOPED_README injection result.
2. **Behavioral counterfactual (runnable simulator).** Can the model use `FM_eff` to predict "what I would conclude if I looked at `u'`" without taking that glimpse — the capability the no-`u` loop failed?
3. **Harden the observational result.** Seed-replicate the three runs; glimpse-size sweep (`G ∈ {1,3,5}`; `G=1` single-patch = maximally controlled) to confirm `cmd_rel_spread` and the arity-beats-capacity gap track the degree of partial observation.
