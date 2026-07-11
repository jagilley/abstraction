# A2A Forward Model (Cerebellum-Style Activation Prediction)

**Idea doc**: [ideas/activation_to_activation_forward.md](../../ideas/activation_to_activation_forward.md)
**Validated component**: fer/experiments/zipfian_grokking/cnb_self_regulation/README.md[^private]

> Per-experiment sections below give the goal + the headline finding only. **Full detail, tables, and the reproduction command for each experiment live in its linked `Full writeup`.** The complete file-by-file index and all reproduction commands are collected in **[FILES.md](FILES.md)**.

## Goal

Co-train a small "cerebellum" forward model alongside a GPT language model. The forward model predicts the main model's activations at a later layer from an earlier layer. The prediction residual (actual minus predicted) is the novelty signal — what the main model computed that the forward model's capacity couldn't anticipate.

This is the first step toward the full architecture described in the idea doc. No feedback loop yet — the two models train independently.

## Architecture

**Main model**: 4-layer, 4-head, 256-dim GPT-2 (28.9M params). Trained on τ=0.0 FineWeb-Edu (unchanged corpus, 10M tokens).

**Forward model**: 1-layer transformer with compressed attention (330K params, ~1% of main model).
- 1 causal attention head, 64-dim key/query/value (compressed from 256-dim input)
- MLP: 256 → 512 → 256
- Pre-norm (LayerNorm before attention and MLP)
- Residual connections

The architecture mirrors the cerebellar circuit:
- Compressed attention projections = pontine relay (dimensionality reduction of cortical state)
- MLP expansion = granule cell combinatorial coding
- Linear readout = Purkinje cell output

**Prediction**: post_block0 → post_block1 (single transformer layer step prediction).

**Training**: Both models use AdamW. Main model lr=3e-4, forward model lr=1e-3. Forward model targets are detached — no gradient flows from forward model loss into the main model. The forward model loss is MSE between predicted and actual post-block1 activations.

## Code & files

`model.py` (minimal GPT-2 with `return_intermediates`), `forward_model.py` (MLP + transformer forward models + gates), and `stages.py` (`train`/`loop_train` co-training entrypoints) are the core loop. Everything else — every `.py` with a one-line purpose, every auxiliary README, the code-theme map, and the checkpoint-compatibility gotcha — lives in **[FILES.md](FILES.md)**. Grep it on demand rather than carrying it here.

## Modal volume

Results saved to the `language-reduction-data` volume under `/data/a2a_forward/` (e.g. `P_10000000/` for the MLP run, `transformer/P_10000000/` for the transformer run; each holds `model.pt`, `fwd_model.pt`, `results.json`). Later experiments add their own subdirectories — see each writeup.

## CLI

Each stage file has its own `@app.local_entrypoint()`, so you run each stage directly; all CLI args are exposed via Modal's auto-generated flags. The canonical patterns:

```bash
cd experiments/

# Default: transformer forward model, post_block0 → post_block1
modal run a2a_forward/stages.py::train --n-tokens 10000000 --n-steps 10000

# Closed-loop training
modal run --detach a2a_forward/stages.py::loop_train --n-tokens 10000000 --n-steps 10000

# Analysis
modal run a2a_forward/analyze.py::analyze --n-tokens 10000000
```

Per-experiment reproduction commands live in each experiment's `Full writeup`.

## What model.py does

`model.py` contains a minimal GPT-2 with `return_intermediates=True` support on `GPT.forward()`. When set, it returns a third value: a dict mapping `"post_embed"`, `"post_block0"`, ..., `"post_blockN"` to their activation tensors. This is backward-compatible — existing callers that don't pass the flag get the same `(logits, loss)` tuple. This file was originally part of the `language_reduction` experiment and is now a local copy.

---

## Open-loop results (2026-05-25 → 05-26)

**Full writeup**: [OPEN_LOOP_ANALYSIS_README.md](OPEN_LOOP_ANALYSIS_README.md)

- **Runs 1–3 — transformer beats MLP forward model.** Per-position MLP reaches cosine 0.788 (structurally blind to cross-position effects); a 1-layer transformer reaches 0.972 with 7× lower MSE, capturing 97% of the computation through entirely different weights. A 2-layer transformer over a 3-layer gap drops to 0.935 — the gap is genuinely harder. **Key lesson**: the bottleneck must be the forward model's *capacity*, not its *structural inability to see the input*.
- **Structure analysis**: the forward model matches block1's attention patterns (cosine 0.989–0.998) with *zero* weight cosine — a different parameterization of the same function (attention gauge symmetry), CKA 0.98. Residual is full-rank and diffuse (effective rank 199.8/256, top-1 PC 2.4%) — the opposite of grokking's low-rank Fourier solution.
- **Causal substitution**: swapping the forward model in for block1 recovers ~94% of block1's KL contribution, with uniform degradation across all behavioral categories (no behavior-specific catastrophic failure).
- **Behavior-conditioned residual**: the residual reflects *computational complexity*, not *task difficulty*. Hardest at delimiter tracking and distributed attention (attention entropy correlates r=+0.33); prediction difficulty has negligible effect.

## Why activation predictions help LM loss, and why co-training produces self-knowledge

### Why predictions help

The forward model predicts post_block3 from post_block0, and that prediction is injected after block1. Before blocks 2-3 have computed anything, the model receives an approximate preview of where its own computation is going to end up.

This is useful because it lets the model allocate its remaining capacity differently. Without the prediction, blocks 2-3 have to do everything — both the predictable, routine components of the computation and the hard, input-specific components. With the prediction, the predictable components are partially pre-computed (cheaply, by the ~1% forward model). Blocks 2-3 can specialize on whatever the forward model *couldn't* capture — the genuinely hard part.

This is a division of labor. The forward model handles the expected trajectory; the remaining layers handle the deviation from expectation. Since the forward model is cheap and the main model's layers are expensive, this is a good trade — you're getting the predictable part of the computation almost for free.

Note: the injection helps most at focused-attention positions (max attn > 0.5, mean help = +0.153, nearly 2× average), not at the hardest or most distant positions. The forward model provides the biggest shortcut for the kind of computation it's best at — sharp, single-source retrieval. See the causal probes enriched analysis (CAUSAL_PROBES_README.md, Test 3-enriched).

### Co-training produces self-knowledge via backprop

The prediction is imperfect (cosine ~0.90-0.93). The model receives this imperfect prediction and must decide what to do with it. The NTP gradient pushes toward representations that encode not just *that* the prediction is imperfect, but *what kind of computation it missed* — because this enables the model to precisely target its remaining capacity at the missing components.

The controlled retrain (Run 6) confirmed this empirically. The closed-loop model's representations at every layer — including early layers that never directly see the injection — encode the forward model's residual dramatically better than the open-loop model (R²=0.21 vs 0.02 at post_block0, R²=0.42 vs 0.26 at post_block3). The self-knowledge is distributed through the model via backpropagation: the injection changes the loss landscape, and gradients flowing backward cause all layers to reorganize to complement the forward model's prediction.

The self-knowledge is **directional**, not magnitude-based. The full 256-d residual vector probe gap (Δ R² = +0.18) is 6× the scalar residual-norm probe gap (Δ R² = +0.03). The model encodes *what kind of computation was missed* (the direction of the residual), not *how much was missed* (the norm). The causal steering test (CAUSAL_PROBES_README.md, Test 2) confirmed the model does not gate reliance on the injection by scalar novelty magnitude.

### The whole argument in a paragraph

A model that receives predictions about its own future computation, and is trained on NTP, is under direct gradient pressure to evaluate those predictions — trusting them where accurate and overriding where wrong. This pressure propagates through all layers via backprop, causing the entire model to reorganize its representations to complement the prediction. The result is a structured representation of *what the prediction captures and what it misses* — a directional self-map encoding what kind of computation was surprising, not just how surprising it was. This is self-knowledge: the model's representations encode information about where its own computation will surprise a compressed model of itself.

**Self-knowledge is the optimal solution to the credit assignment problem of "when to trust a cheap approximation of your own computation."** You get it for free from NTP the moment you close the loop. The controlled retrain confirmed: the self-knowledge is real (not an lr artifact), directional (not magnitude-based), and distributed across all layers (via backprop, not localized to post-injection layers).

---

## Closed-loop & self-knowledge

### Run 4: Closed-loop cerebellar training (2026-05-25)

**Full writeup**: [CLOSED_LOOP_README.md](CLOSED_LOOP_README.md)

Feeding the 2-layer forward model's post_block3 prediction back into the residual stream (learned zero-init gate, injected after block 1) **helps LM loss** (Δ = -0.05 to -0.08 nats, ~9% perplexity) at every eval step; the gate opens monotonically (0.08 → 3.03, never saturating). The model **develops self-knowledge** (residual-probe R²=0.44 CL vs 0.28 OL) but also **becomes dependent** on the injection (0.11 nats worse without it) — the wake-sleep consolidation problem. Forward model cosine degrades 0.935 → 0.897 (moving target).

### Run 5: Forward model capacity scaling sweep (2026-05-28)

**Full writeup**: [SCALING_SWEEP_README.md](SCALING_SWEEP_README.md)

Frozen main model, forward models at 5 capacity points (1%–22%). The forward model **saturates at ~10% capacity** (cosine 0.999); computational-novelty effects shrink as predicted (delimiter d_BC +0.78 → +0.30). r(res,LM) stays at zero — but that's the wrong metric: the signal lives in the residual's 256-d *direction*, not its scalar norm. Residual stays high-rank (>235/256) at every capacity — language computation is distributed, unlike grokking's low-rank solution.

### Run 6: Controlled retrain — eliminating the lr/seed confound (2026-05-29)

**Full writeup**: [CONTROLLED_RETRAIN_README.md](CONTROLLED_RETRAIN_README.md)

Identical lr/seed/init/data-order, only the loop differs. Replicates Run 4 (injection helps, gate opens) and confirms **the R² gap is real, not an lr artifact** (R²=0.42 vs 0.26). The self-knowledge is **distributed uniformly across all layers via backprop** (Δ R² +0.185 at post_block0, +0.161 at post_block3 — ruling out downstream construction), **directional not magnitude-based** (vector gap 6× scalar), and comes with injection **dependency** (0.11 nats).

### Causal probes of the self-map (2026-05-28)

**Full writeup**: [CAUSAL_PROBES_README.md](CAUSAL_PROBES_README.md)

Magnitude gating is *not* the mechanism (steering the residual-norm direction has zero novelty-specific effect); help is organized by residual *direction*, not magnitude (η² 0.0017 vs 0.0003), strongest at focused-attention positions (+0.153, ~2× average). The directional form of self-knowledge was still untested causally after this — motivating the directional steering experiment.

### Directional causal steering (2026-05-30)

**Full writeup**: [DIRECTIONAL_STEER_README.md](DIRECTIONAL_STEER_README.md)

Tests whether directional self-knowledge is causally *used*. **Outcome B with partial A**: steering along per-cluster directions produces real effects (~4× random controls) with partial diagonality (|diag|/|off| = 1.33, likely underestimated due to 0.41 direction overlap). The **focused-attention cluster is the standout** (selectivity 2.41 — same cluster with the highest baseline injection help). The 4-layer non-looped GPT learned the highest-value discrimination and left the rest coarse; a looped transformer would give more depth to act on the directional self-knowledge it already encodes.

### Representational divergence analysis (2026-06-02)

**Full writeup**: [REPRESENTATIONAL_DIVERGENCE_README.md](REPRESENTATIONAL_DIVERGENCE_README.md)

CKA diverges monotonically (0.972 → 0.727 by post_block3). **Two distinct reorganization phenomena**: early layers change extensively for general-purpose reasons (change is *orthogonal* to self-knowledge, 0.49× random overlap), while late layers reorganize primarily *along* self-knowledge dimensions (2.42× random). This resolves Run 6's uniform Δ R²: the *amount* of self-knowledge is uniform, but its relationship to the dominant representational change differs qualitatively by layer.

### Mirror test for neural self-knowledge (2026-06-02)

**Full writeup**: [MIRROR_TEST_README.md](MIRROR_TEST_README.md) (v1, v2, v3)

Three iterations of a perturbation-based mirror test. The robust finding across all versions is a **2–3× robustness gap** — CL models absorb perturbations with less loss degradation. No active self-opposition (structural regularization, not deliberate self-correction), but v3 finds CL+M produces proportionally more topic-specific responses. The self-knowledge is in the weights (CL-M ≥ CL+M).

### Baseline battery: is forward self-prediction uniquely useful? (2026-06-04)

**Full writeup**: [BASELINE_BATTERY_README.md](BASELINE_BATTERY_README.md)

5 conditions (open-loop, forward, shifted k=10, random projection, autoencoder). **Robustness is uniquely strong with forward prediction** (2.4× improvement vs 1.5–1.6× for baselines) and its self-knowledge uniquely persists through full depth (2.4–2.9× baselines at post_block3). **LM loss and robustness are dissociated** — the autoencoder gives the *largest* LM benefit but *worse* robustness. Robustness is specifically a consequence of the injection being a prediction of the model's own future computation. The shifted baseline validates position-specificity (destroys all effects).

### Jacobian analysis: spectral structure of robustness (2026-06-05)

**Full writeup**: [JACOBIAN_ANALYSIS_README.md](JACOBIAN_ANALYSIS_README.md)

**Hessian trace quantitatively predicts robustness** (tr(H) 0.38–0.45× OL; E[ΔL]=ε²/(2d)·tr(H) matches empirical to within 8%). Gradient norms are 30% smaller. Two negative results: the gradient spectrum is *less* concentrated for CL, and the self-knowledge subspace does *not* align with sensitivity (orthogonal). **The mechanism is a uniformly flatter loss landscape**, not sensitivity organized along self-knowledge directions — consistent with the division-of-labor interpretation.

## Run 7: Extended co-training — 50K steps (2026-06-01)

**Full writeup**: [EXTENDED_TRAINING_README.md](EXTENDED_TRAINING_README.md)

**The injection benefit grows monotonically — 6× over training** (-0.08 → -0.48 nats, no saturation) while forward cosine *drops* (0.93 → 0.82): the prediction is used as a structured reference frame, not a literal preview. No differential overfitting (both models overfit identically). Self-knowledge probes narrow but persist at early layers.

**Run 7b (10% forward model, 15K steps)**: same design, capacity-sufficient FM. **Zero net val benefit — pure computation relocation** (injection benefit -0.213 exactly offset by dependency +0.210); U-shaped benefit; self-knowledge probes 2× stronger (residual magnitude itself becomes informative). **Capacity sweet spot for net benefit lies between 1% and 10%**: 1% is imperfect enough that the model can't fully offload; 10% is so capable the model offloads aggressively and becomes fully dependent.

## Model scale experiment: residual structure vs main model size (2026-06-03)

**Full writeup**: [MODEL_SCALE_README.md](MODEL_SCALE_README.md)

29M vs 77M on identical 100M-token data. **The bigger model's computation is more predictable and more concentrated**: cosine 0.994 vs 0.980 (despite 2× tighter FM compression), and seven independent rank-normalized metrics all point to a more concentrated residual. Sentence-start effects collapse at 77M (d=-0.35 vs -1.05); delimiter tracking persists at both scales. A transient early-training rank dip at 77M weakly echoes grokking's phase-transition compression. The grokking analogy holds in direction, not magnitude — language computation concentrates very slowly.

## MNIST experiment: cross-domain validation (2026-06-07)

**Full writeup**: [MNIST_README.md](MNIST_README.md)

Full A2A experiment ported to MNIST via a 4L/4H/128D ViT. **The residual is low-rank and digit-discriminative — opposite of language** (effective rank 18.3/128 vs 199.8/256; all top-5 PCs discriminate digit identity; cross-digit residual cosine mirrors visual similarity). Self-knowledge probes are 3× stronger, the robustness gap is 4× (12× at ε=2.0), and causal substitution shows *non-uniform* degradation (13× across digits) — unlike language's uniformity. Baseline battery shows *less* differentiation than language: the low-rank residual is easily captured by any injection. **Residual rank reflects DGP complexity** (grokking ~15 Fourier, MNIST 18 digit-discriminative, language 200 diffuse).

## Calibration transfer & OOD robustness (2026-06-09)

**Full writeup**: [OOD_ROBUSTNESS_README.md](OOD_ROBUSTNESS_README.md)

Baseline-battery models taken OOD (Wikipedia, code, French, math, shuffled), no retraining. **Experiment A — calibration transfer (negative)**: closing the loop does *not* produce transferable epistemic self-knowledge; output entropy beats every activation probe ID and OOD. **Experiment B — OOD robustness (positive)**: the forward model's robustness is distribution-invariant (Δloss ratio 0.43–0.51 on every natural corpus) while the autoencoder's collapses off-manifold (on code, tr(H) 1.02× = same curvature as open-loop). **Joint interpretation**: the self-knowledge from closing the loop is *computational, not epistemic; distribution-invariant, not data-bound* — forward prediction encodes the model's computational function, the autoencoder encodes the ID activation manifold.

## Single-cycle wake-sleep distillation (2026-06-11)

**Full writeup**: [DISTILLATION_README.md](DISTILLATION_README.md)

Can distillation force the main model to internalize the FM's contribution? **Distillation fully closed the dependency gap** (+0.083 → -0.005 nats, 105.6% closed; cost vs OL only +0.030). Innovation directions **migrated** (fresh-FM top PCs near-orthogonal to original) but behavioral conditioning is stable (r=0.81) — same positions hard, in different directional ways. Residual became more diffuse (no discrete abstraction extracted on flat webtext — the ratchet grinds smoothly). Robustness partially retained (0.942 vs OL); late-layer self-knowledge survived distillation.

## MNIST local prediction-error learning (2026-06-15)

**Full writeup**: [MNIST_LOCAL_LOSS_README.md](MNIST_LOCAL_LOSS_README.md)

Uses `λ·MSE(sg(FM(post_block0)), post_block3)` as an auxiliary main-model loss ([idea](../../ideas/local_prediction_error_learning.md)). **LL learns faster** (+1.4pp final, 31% lower val_loss, FM cosine 0.997 — "be predictable" pressure was beneficial). CL_LL reduces dependency 63%. **Strong negative — regularity ≠ robustness**: LL is 13× *more* sensitive than OL; CL's robustness comes from injection experience, not computational regularity (revises the Jacobian interpretation). CL_LL has the highest self-knowledge (R²=0.76–0.79) and is the **first high-fwd-cos closed-loop condition** (0.985). The absorbed knowledge is predominantly **meta-knowledge** (3:1 over object-level at early layers) — the local-loss gradient literally carries the meta signal `∂post_block3/∂early · (post_block3 − FM_pred)`.

## Computational property geometry probes (2026-06-16)

**Full writeup**: [GEOMETRY_README.md](GEOMETRY_README.md)

The "king − man + woman = queen" analogy applied to computational properties, across OL/CL/Distilled on MNIST and language. **Distilled has the most orthogonal early-layer representations** and the best final-layer vector arithmetic (both domains); **CL has the least orthogonal late-layer representations** (meta-knowledge creates entanglement — FM reliability entangled with block contribution and token frequency). The dissociation rules out "more self-knowledge → more organized": CL has the strongest self-knowledge R² yet the worst orthogonality. Distillation converts entangled meta-knowledge into orthogonal object-level knowledge.

## MNIST multi-cycle gated ratchet (2026-06-17)

**Full writeup**: [GATED_RATCHET_README.md](GATED_RATCHET_README.md)

Multi-cycle CL_LG (injection + bilevel-gated local loss) with periodic distillation + FM reinit. **Compounding val loss improvement** — WS_LG's gap over OL widens 34% → 48% across 4 cycles; neither distillation alone nor gated local loss alone sustains it (both components required). **The gate opens rather than closes** (0.31 → 0.77): on a fixed dataset more compression always helps within-distribution; selective closing requires novel inputs. Interpreted as a new form of **implicit regularization from self-compression** — the model finds solutions maximally legible to a compressed version of its own computation.

**Extended ratchet (16 cycles)**: two-phase dynamics — genuine compression (cycles 1–5) then **activation norm inflation** after the FM hits its compression ceiling (post_block3 norms grow 8.5×); the raw robustness improvement is a measurement artifact (norm-scaled, WS_LG is 16× *more* sensitive). A smaller FM makes it worse (gate rationally closes on noisy FM error dimensions). **The FM capacity sweet spot is narrow on a fixed dataset** — a continual-learning setting would sidestep this.

## OOD gate experiments: meta-learning under distribution shift (2026-06-18)

**Full writeup**: [OOD_GATE_README.md](OOD_GATE_README.md)

**Digit shift**: the FOMAML gate is input-selective (novel digits get higher gate weights, 6–8× amplification) — but selective in the *opening* direction. **Fashion shift**: on a cross-manifold shift the gate closes globally and selectivity vanishes — the FM's error decomposition is the gate's vocabulary, and it doesn't align with domain boundaries. **Unified gate (NTP-only, no bilevel)**: produces the same qualitative meta-learning signal through injection utility (known digits close, novel reopen) — first-order meta-learning without bilevel optimization, mechanistically cleaner though smaller in magnitude.

## Looped transformer with FM injection (2026-07-09)

**Full writeup**: [LOOPED_README.md](LOOPED_README.md) | **Status**: in progress (single seed)

Tests the [self_model_needs_a_loop.md](../../ideas/self_model_needs_a_loop.md) hypothesis: a *causally-used* self-forecast is forced only in a **weight-shared looped** model with per-step FM injection (`s_{t+1} = g(s_t + p + gate·FM(s_t))`), not feedforward. **The trajectory is the finding** — the self-forecast is inert or destabilizing until **four confounds** are removed (prelude/coda shortcut, deep supervision, over-capacity FM, MNIST barely needing the loop). Confound-free + bounded gate → **causal necessity that scales with loop-necessity**: on Fashion-MNIST (loop load-bearing) ablating the injection collapses accuracy 0.847 → 0.667, ~5× the MNIST cost. The **bounded scalar gate self-regulates** to ~0.016 and holds it. No net loss benefit is expected (the injection is a new self-referential input modality) — the discriminator is causal use, not lower loss. Baseline battery + imprint probes (done): only forward's dependency *scales* with loop-necessity and only forecast-shaped channels leave a channel-specific imprint — a veridical self-map used additively (no efference-copy cancellation). The near-manifold runnable-simulator probe is **negative on MNIST** (self-consistency is a generic low-rank property).

## Active-vision looped ViT: the "missing u" / efference-copy test (2026-07-09)

**Full writeup**: [ACTIVE_VISION_README.md](ACTIVE_VISION_README.md) | **Status**: observational half done (single seed)

Tests the [self_model_needs_a_loop.md](../../ideas/self_model_needs_a_loop.md) "missing u" thread — **arity, not resolution**: a forward model of a *controlled* system must take the command as a second input (arity-2), and no capacity lets a command-blind arity-1 model recover command-driven dynamics. `GlimpseLoopedViT` gives the loop a real command `u` (a glimpse location each step), then trains `FM_state` (arity-1) vs `FM_eff` (arity-2, via a spatial efference-copy marker). **The no-`u` control collapses the effect** (full-view mode: `cmd_rel_spread` = 0.000, `FM_eff` ≈ `FM_state` at every capacity) — direct evidence the earlier looped simulator negative was because MNIST-classification is *autonomous*. **With the `u`, arity beats resolution**: `FM_state` saturates with capacity and the smallest command-aware FM beats the largest command-blind one (MNIST 0.911 > 0.901). Scales ~2× on Fashion (harder, more sequential). **Boundary**: observational premise only; causal/behavioral half pending.

The control-regime port (`reaching_vit.py` / `mnist_reaching.py`) makes this **causal**: a model-based planner rolling `FM_eff` (arity-2) reaches the goal while `planner_state` (arity-1) is stuck at random at every capacity — the control-task positive that the perception-task injection nulled on. See [FILES.md](FILES.md) for the file map.

## Internalized forecasting on the reaching task (2026-07-10)

**Full writeup**: [REACHING_INTERNAL_README.md](REACHING_INTERNAL_README.md) | **Status**: done, positive + sharpened diagnosis (single seed × MNIST + Fashion, both agree)

Replaces the reaching positive's **external `argmax`** with an **endogenous** self-forecast (differentiable internal planner `int_plan` + fed-back injection `int_inject`, co-trained; matched init/coverage), asking what reorganizes when the forecast is used from inside the loop. **Internalization makes the operator dramatically more *plannable*** — a fresh decoupled external planner jumps +0.55→+1.00 on the co-trained operator and CKA vs mf collapses to 0.06/0.25 — via *task-relevant* legibility (position decodability ↑, full-state forward-predictability ↓), not uniform. **Map and model coexist** — the running loop *depends* on the forecast (ablate → floor) while a linear readout on the frozen operator recovers a latent legible map (+0.43/+0.73). Arity impossibility survives internalization (`int_plan_state` at the floor); injection reproduces the a2a gate-opens-with-dependency signature. **Collusion-vs-selectivity diagnostics (both datasets)**: the **planner** forecast is a *transferable, task-relevant self-model* (predicts the value-relevant direction 1.7–3.0× better than the junk; a fresh independent probe plans with it to +0.5–0.7) — its low full-vector Δ-cos is *correct selectivity*, not advisor drift; an arity-1 control shows *veridicality ≠ usefulness* (Δ-cos 0.99, plans at floor). **Reading ladder (both datasets)**: transferability is gated by *how much the forecast is read* — only a **raw-scalar** injection gate yields an advisor (worse-than-random to an independent probe); injection with the **thalamic-relay projection gate** (`CerebellarGate`, what the prior arc used) or an **explicit readout** is transferable (+0.4 to +0.75), so "injection privatizes the forecast" is *retracted* — the crude scalar gate was the confound. Live next step: whether the transferable one-step forecast **composes** into a multi-step runnable simulator (on a lookahead-requiring task).

## Multi-step lookahead: does the one-step self-forecast compose? (2026-07-11)

**Full writeup**: [REACHING_LOOKAHEAD_README.md](REACHING_LOOKAHEAD_README.md) | **Status**: done, positive-with-a-dissociation (single seed × {obstacles, maze} × MNIST; perfect-sim control done)

The live next step from `REACHING_INTERNAL`: does the transferable one-step self-forecast **compose** into a runnable multi-step simulator? Three lookahead-forcing geometries (obstacles / serpentine maze / occluded-reveal) where a myopic planner provably fails; the discriminator is an **endogenous N-step MPC** that rolls the FM in state space decoding its own believed position each step (no env access). **The composition question resolves into two separable axes.** (1) **Deep veridical composition IS achievable — and is gated by the loop**: an FM trained with explicit multi-step-consistency pressure composes to **near-full-state fidelity out to horizon 8** (maze pos-acc 0.99@n6), but *only on the plannability-shaped `int_plan` operator* — the identical FM collapses on the `mf` operator (0.52→0.05). Closing the loop reorganized the operator into **deeply composable dynamics**; the earlier "one-step self-model doesn't compose" was a one-step-*training* artifact. (2) **Veridicality ⊥ control-usefulness**: the value-shaped **co-trained FM is the *best* planner substrate (+0.51) despite the *worst* veridicality (0.21@n4)** — it even **beats a perfect-simulator env-MPC (+0.33)** with the same planner. A **perfect-simulator control** confirms the maze is **planner-bound, not simulator-bound** (perfect sim also floors, +0.15 at depth 10). Net: the loop yields *both* a legible deeply-composable simulator (available) and a value-shaped forecast the controller actually runs on (used); what helps control is value-alignment, not full-state fidelity. **Live next step**: a stronger planner (CEM/beam + geodesic/learned value) to convert the deep simulator into hard-lookahead behavior.

## Next steps

1. **Language gated ratchet**: language's full-rank residual and rich behavioral decomposition would make the gate's selectivity far more interpretable than MNIST's 10 digits — the gate might develop per-behavioral-category selectivity within a single run (language is inherently multi-task), without needing an explicit distribution shift.
2. ~~**OOD adaptation post-ratchet**~~: *Done* — WS_LG ≈ WS_UG_uniform (+8pp zero-shot OOD, ~25% faster adaptation); the val loss improvement is genuine generalization. See [OOD_GATE_README](OOD_GATE_README.md#experiment-4-ood-adaptation-after-gated-ratchet-2026-06-20).
3. ~~**Looped transformer**~~: *In progress* — confound-free bounded-gate injection is causally necessary and scales with loop-necessity; baseline battery + imprint probes done; runnable-simulator probe negative on MNIST. **Pending**: domain port (RHM / looped language) for the simulator question, cross-decodability matrix, chase the cancellation signature. See [LOOPED_README.md](LOOPED_README.md).
4. **Cross-model self-knowledge control**: train separate fresh FMs on each model's own activations (OL, CL, distilled), then probe each for its own FM's residual — eliminates the confound in the distillation self-knowledge probes.
5. **Model scale 350M**: third data point for the model scale experiment (~24L/16H/1024D on 500M+ tokens) — does the residual-concentration trend continue, accelerate, or saturate? See [MODEL_SCALE_README.md](MODEL_SCALE_README.md).
6. **Harden the OOD robustness result**: (a) direct manifold-displacement check (autoencoder reconstruction MSE per corpus should peak on code); (b) bootstrap CIs from the saved per-direction Δloss arrays. See [OOD_ROBUSTNESS_README.md](OOD_ROBUSTNESS_README.md).

[^private]: Not mirrored: this link points to a document in the private lab repo (the roadmap, the queue, an unrun spec, reading notes, or a conversation). See the top-level README for what is held back and why.
