# Looped Transformer with Forward-Model Injection

**Idea doc**: [ideas/self_model_needs_a_loop.md](../../ideas/self_model_needs_a_loop.md)
**Parent experiment**: [README.md](README.md) (the feedforward a2a arc)
**Status**: In progress. The headline result (below) is a single seed; the load-bearing control (baseline battery) and discriminators 2–3 are **not yet run**.
**Date**: 2026-07-09

## Goal

Test the central claim of [self_model_needs_a_loop.md](../../ideas/self_model_needs_a_loop.md): a *separable, causally-used* self-forecast (a self-**model**, not merely absorbed legibility) is ill-posed in a feedforward net and becomes **forced** in a **weight-shared looped** model with forward-model (FM) injection at each step — because weight-sharing makes the same operator `g` both the *producer* of the state the FM reads and the *consumer* of the FM's forecast, and convergence forces the resting state to agree with a forecast of itself.

Concretely: build a looped ViT (`h_{t+1} = g(h_t + p + gate·FM(h_t))`), where the FM predicts a future iterate, and test whether the injected self-forecast becomes **causally necessary** — in a way the feedforward closed-loop model's absorbed map is not.

## Architecture (`looped_vit.py`)

`LoopedViT` — a weight-shared, recurrent-depth ViT with the same I/O contract as `vit.py`:

- **Shared operator `G`**: `n_loop_layers` pre-norm ViT blocks applied `T` times with weight sharing (default 1 block × T=8). This is the theory's reusable `g`.
- **Input re-injection** (Universal-Transformer / DEQ style): `s_0 = 0`, `s_{t+1} = G(s_t + p + inject_t)`, `p = prelude(embed(x))`. Re-injecting the input each step makes the fixed point input-dependent, and no per-step timestep embedding is used (that would make `g_t ≠ g` and break the "one operator to model" premise).
- **Optional non-shared bookends** `prelude` / `coda` (the "looparound"): with them, the FM setup is a near-exact analog of the feedforward a2a — FM reads a mid-stack recurrent state (≈ `post_block0`), predicts a future state (≈ the `post_block` target), injects into the recurrent stream (≈ inject-after-block1), and coda+head separate readout (≈ block3→head). **These turned out to be a confound (see trajectory) and default to 0 in the confound-free runs.**
- **Knobs added over the trajectory**: `readout_all_steps` (per-step logits for deep supervision), `step_inject_fn(t, operand, state)` hook (carries the gated FM forecast; exposes the raw state so the caller can inject the *update* `FM−s`), `predict_k` (FM predicts `s_{t+k}`), `injection_form` (`next_state` vs `update`), `gate_type` (`proj` vs bounded `scalar`), `damping` (`s += α·(G−s)` for convergence without deep supervision).

## The trajectory — what we tried and what each step taught

This experiment moved through a sequence of failures that were each individually informative. The short version: **the loop's self-forecast is inert or destabilizing under the obvious/naive settings, and only becomes a clean, load-bearing self-referential channel once four separate confounds are removed.**

| # | Setup | Result | Lesson |
|---|---|---|---|
| 1 | **Plain looped ViT**, no injection (`mnist_looped.py`) | Matches 4-layer FF at ¼ params (0.21M); but the timed-loop **overshoots** — acc peaks at train-T (97.3%) and *degrades* past it (94.7% @T=20). | The plain loop learns a *timed trajectory*, **not a fixed point** (rel-Δ still 17% at T=8). No stable resting state → nothing for the fixed-point argument to bite on. |
| 2 | **+ Deep supervision** (readout loss at every step) | Overshoot **gone** — loss flat to 4 decimals T=15→20, a genuine attractor. | Deep supervision manufactures a fixed point. *But it forces every iterate toward the answer (high-abstraction supervision) — later identified as a confound that makes the task easier than it should be.* |
| 3 | **Naive injection** `gate·FM(a_t)`, FM predicts next state (2×2, w/ prelude/coda + deep-sup) | **Unstable.** Trained fine until the gate opened past ~2.5, then the loop diverged (cl_last acc 0.94→0.56, gate ran 0.08→3.7). FM cosine *fell* 0.999→0.95 as it destabilized. | Injecting a full next-state-magnitude forecast **compounds** over 8 steps. The gate feels the same greedy upward pressure as feedforward (0.08→3.0) but the loop can't absorb it. *(This also explains the "0.97 cosine" — it was instability, not a bottleneck.)* |
| 4 | **Update injection** `gate·(FM(a_t) − s_t)` | **Stable but inert.** cl_last 0.972; injection benefit ≈ 0 (−0.015); no-inj ≈ inj; convergence/self-consistency identical to OL. | The update form vanishes at the fixed point (`FM(s*)≈s*`), so it's stable — but with a 1-step-ahead target it carries nothing the loop doesn't already have. Self-consistency achieved *trivially*. |
| 5 | **Longer horizon** `k=3` (predict `s_{t+3}`) | **Still inert.** fm_cos still 0.999; benefit −0.007. | The one-block loop's k-step trajectory is *still* trivially predictable. Not a horizon problem. |
| 6 | **FM-capacity sweep** (`mnist_looped_fm_sweep.py`) | Even a **1%-capacity FM (5.4K) predicts the loop's *update* at cos 0.92**; a full-size FM at 0.996. No FM in the family gives cos 0.5–0.7. | The loop's per-step transformation is **intrinsically low-complexity** (residual eff-rank ~20/128). But 0.92 is the same operating point where feedforward's 1% FM *was* useful — so "FM too accurate" isn't the whole story. |
| 7 | **Confound diagnosis** | The pure phase-1 loop *did* need its depth (T=1 68% → T=8 97%). Adding prelude/coda (a 3-block FF path) + deep-sup made MNIST look feedforward-solvable. | **Two of our own confounds** — the prelude/coda feedforward shortcut and deep supervision — were masking the loop's role. And **MNIST classification barely needs iterative computation** (the answer, not just the supervision, must require the loop). |
| 8 | **Fashion-MNIST** (harder perception, same DGP type) | Loop far more load-bearing: T=1 **40%** → T=8 85% (vs MNIST 68%→97%). | A harder task makes the loop's iterations genuinely necessary — a better substrate for testing whether the self-forecast matters. |
| 9 | **Confound-free** (pure loop, no deep-sup, small FM, Fashion), **unbounded gate** | Injection **helped a lot early** (−0.69 nats, acc 0.53 @ gate 0.43) — *then* the unbounded gate ran away (0.43→1.9) and collapsed the model to chance. | The injection **is** useful when the loop is load-bearing — but the projection gate is an **over-relaxation coefficient** and, unbounded, extrapolates past the stable region. Signal present; training unstable. |
| 10 | **+ Bounded scalar gate** (mixing coeff ∈ [0,1]) | **Stable and strongly causally necessary** — see headline. | Bounding the over-relaxation coefficient fixes the instability while keeping the confound-free pure loop. |

## Headline result (confound-free, bounded gate — single seed)

Pure loop (prelude=coda=0), no deep supervision, small 5.4K FM (update-cos ≈ 0.92), `predict_k=3`, update-form injection, bounded scalar gate ∈ [0,1]:

| task | OL acc @T=8 | CL **with inj** | CL **inj ablated** | dependency (Δacc) | inj benefit (Δloss) | final gate |
|---|---|---|---|---|---|---|
| **MNIST** (loop ~idle) | 0.973 | 0.964 | 0.925 | 0.039 | −0.16 | 0.015 |
| **Fashion** (loop load-bearing) | 0.852 | 0.847 | **0.667** | **0.180** | **−0.86** | 0.016 |

**The injected self-forecast is a real, load-bearing channel, and its load scales with how load-bearing the loop is.** Ablating the injection from the Fashion CL model (same weights, gate zeroed) collapses accuracy 0.847 → 0.667 (+0.86 nats). On MNIST, where the loop is nearly idle, ablation costs only 0.04. The dependency is **~5× larger on the task that actually needs the loop** — the opposite of the inert behavior every confounded setup showed.

### The self-regulated gate (the striking part)

The gate tells a clean story across the trajectory:

- **Unbounded projection gate** → runs away (0.08→3.7) → collapse. No self-regulation.
- **Bounded scalar gate** → **self-regulates to a small, stable value (~0.016) and holds it** (it even drifts *down* 0.018→0.015 during training, actively staying out of the over-relaxation regime) — while the model becomes **strongly dependent** on that tiny signal. A ~2% anticipatory injection, present at every step, becomes load-bearing. The model settles on a small self-chosen dose of its own forecast and organizes its computation around it.

## Interpretation

**No net loss benefit is expected, not a disappointment.** CL-with-injection ≈ OL in accuracy (0.847 vs 0.852 on Fashion). We have no reason to expect the loop's self-forecast to *lower task loss* — the injection genuinely adds a new stream of information the model must parse *in addition to* the task, so equal or even slightly-worse loss is the natural outcome. This connects directly to the idea doc's ["injection as a new self-referential input modality" / efference-copy](../../ideas/self_model_needs_a_loop.md) section: the dependency is not a bug but a **slot the model allocates for an expected input modality** it has learned to interpret. The right question is not "did loss go down" but "is the self-forecast a separable, causally-used channel" — and the ablation says yes.

**Why the confounds mattered.** Each confound independently made the self-forecast look inert: prelude/coda gave a feedforward path around the loop; deep supervision forced intermediate states to already *be* the answer; an over-capacity FM made the forecast redundant; MNIST barely needs the loop at all. Only with all four removed — pure loop, last-step supervision, bottlenecked FM, load-bearing task — does the causal necessity appear. This is itself a finding: **the loop's self-forecast is causally used precisely when (and to the degree that) the loop's iterative computation is load-bearing.**

## Caveats

1. **Ablation alone is weak evidence.** *Any* signal a model trains with becomes load-bearing when removed. The non-trivial part is the **scaling with loop-necessity** (Fashion 5× MNIST) — but that could partly reflect task difficulty (Fashion's higher baseline loss leaves more headroom for any auxiliary signal). **The baseline battery is the decisive control.**

   **UPDATE (2026-07-09, baseline battery run — PENDING DISCUSSION, not final):** random_proj (a frozen random-init FM) creates *equal-or-greater* absolute dependency than the genuine forecast, and forward is *not* more robust in operating mode. So **absolute causal necessity is not forecast-specific** — a consistent, position-aligned self-referential channel drives the dependency, not forecast-correctness (supports the weaker "slot/modality" framing, not the stronger "self-model" claim).
2. **Discriminators 2 & 3 untested.** Near-manifold self-counterfactual (2) and fixed-point self-consistency *structure* (3) are what separate a causally-used self-**model** from a merely useful auxiliary signal. Only discriminator 1 (causal necessity) has been probed.

## Next steps

1. **Baseline battery on Fashion (highest priority — the load-bearing control).** Re-run the confound-free bounded-gate setup with `random_proj` (inject a frozen random projection of `s_t`) and `shifted` (inject the forecast for a different position/step) conditions. If forward-prediction is special, only it should show the dependency **and** the MNIST→Fashion loop-necessity scaling. This directly tests caveat 2.
2. **Discriminators 2 & 3.** Near-manifold self-counterfactual ("what would I output if this input were slightly perturbed") and fixed-point self-consistency structure.
3. **Seed replication** of the headline Fashion result.
4. **Damping vs bounded-gate convergence** — whether the FM injection alone (bounded) provides the endogenous convergence that deep supervision faked (the "scrap deep-sup once self-knowledge suffices to halt" thread).

## Reproduction

```bash
cd experiments/

# 1. Plain looped ViT (phase 1): overshoot; + deep supervision -> attractor
modal run --detach a2a_forward/mnist_looped.py::a2a_mnist_looped
modal run --detach a2a_forward/mnist_looped.py::a2a_mnist_looped \
  --prelude-layers 1 --coda-layers 1 --deep-sup --deep-sup-frac 0.5 --no-train-ff-baseline

# 2. Injection 2x2 (per-condition container; run all four in parallel).
#    injection-form: update (stable) | next_state (naive, unstable)
for c in ol_last ol_deep cl_last cl_deep; do
  modal run --detach a2a_forward/mnist_looped_injection.py::train_condition \
    --condition $c --injection-form update --predict-k 3 & done; wait
modal run a2a_forward/mnist_looped_injection.py::aggregate --injection-form update --predict-k 3

# 3. FM-capacity sweep (state-cos vs update-cos vs FM size), confound-free pure loop
modal run --detach a2a_forward/mnist_looped_fm_sweep.py::fm_capacity_sweep --dataset mnist
modal run --detach a2a_forward/mnist_looped_fm_sweep.py::fm_capacity_sweep --dataset fashion_mnist

# 4. HEADLINE: confound-free + bounded gate, MNIST vs Fashion
for d in mnist fashion_mnist; do for c in ol_last cl_last; do
  modal run --detach a2a_forward/mnist_looped_injection.py::train_condition \
    --condition $c --dataset $d --prelude-layers 0 --coda-layers 0 \
    --fwd-d-head 1 --fwd-mlp-mult 0.125 --predict-k 3 --injection-form update \
    --gate-type scalar & done; done; wait
```

## Files

- `looped_vit.py` — `LoopedViT` (weight-shared recurrent-depth ViT; injection hook, prelude/coda, deep-sup readout, damping-compatible).
- `mnist_looped.py` — phase 1: plain loop, convergence dynamics, acc-vs-T sweep, deep-supervision, FF baseline.
- `mnist_looped_injection.py` — phase 2: the injection 2×2 (`train_condition`) + `aggregate`. Supports injection form, `predict_k`, dataset (mnist/fashion), damping, gate type, FM capacity.
- `mnist_looped_fm_sweep.py` — FM-capacity sweep on a frozen confound-free loop (state-cos vs update-cos vs size; loop-necessity).
- `forward_model.py::BoundedScalarGate` — bounded over-relaxation gate ∈ [0, max_gate].

Modal volume: `/data/a2a_forward/mnist_looped*/`.
