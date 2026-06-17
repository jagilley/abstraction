# A2A Forward Model (Cerebellum-Style Activation Prediction)

**Idea doc**: [ideas/activation_to_activation_forward.md](../../ideas/activation_to_activation_forward.md)
**Validated component**: fer/experiments/zipfian_grokking/cnb_self_regulation/README.md[^private]

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

## Results (2026-05-25)

### Runs 1-2: MLP vs transformer forward model

**Details**: [OPEN_LOOP_ANALYSIS_README.md](OPEN_LOOP_ANALYSIS_README.md)

Run 1 (per-position MLP, post_embed → post_block1): cosine 0.788, structurally blind to cross-position effects. Run 2 (1-layer transformer, post_block0 → post_block1): cosine 0.972, 7× lower MSE. The transformer forward model captures 97% of the computation in direction through entirely different weights.

### Key lesson: capacity bottleneck vs structural blindness

The per-position MLP's residual captures "attention exists" — a trivially predictable structural fact. The transformer forward model's residual captures "computation too complex for this capacity" — genuine novelty. The information bottleneck must be the forward model's *capacity*, not its *structural inability to see the input*.

## Files

| File | Purpose |
|---|---|
| `forward_model.py` | `ForwardModel` (per-position MLP) and `TransformerForwardModel` (1-layer transformer) |
| `stages.py` | Modal stage `a2a_train` — co-training loop with eval and analysis |
| `analyze.py` | Structure analysis: PCA, CKA, attention pattern comparison, weight-space comparison |
| `causal_substitution.py` | Causal substitution: replace block1 with forward model, measure per-behavior degradation |
| `behavioral_residual.py` | Behavior-conditioned residual analysis: attention, syntactic, difficulty, context integration categories |
| `controlled_retrain.py` | Controlled retrain: identical lr/seed open-loop vs closed-loop, with layerwise self-map probes |
| `novelty_probe.py` | Layerwise probes for named syntactic categories and residual norm, open vs closed |
| `novelty_steer.py` | Causal steering of the novelty direction; reliance measurement |
| `injection_help.py` | Per-token Δloss analysis: where does the injection help? |
| `injection_help_structural.py` | Structural analysis: help by residual direction / attention shape |
| `directional_steer.py` | Directional causal steering: multivariate probe, per-cluster-direction sweep |
| `extended_training.py` | Extended co-training (50K steps): injection benefit trajectory, overfitting comparison |
| `llama_cache_acts.py` | Cache Llama 3.2 1B activations for scale-up experiment |
| `llama_head_decomposition.py` | Llama per-head decomposition: what the forward model captures vs misses per attention head and MLP |
| `representational_divergence.py` | Representational divergence: CKA, diff PCA, self-knowledge alignment between open- and closed-loop models |
| `prediction_trust.py` | Prediction trust: does the model's per-direction usage gain (gate + downstream sensitivity) track the forward model's reliability spectrum? Wiener-gain analysis + counterfactual error injection |
| `mirror_test.py` | Mirror test: perturbation response channeling through self-knowledge subspace |
| `mirror_test_v2.py` | Mirror test v2: compensatory response and perturbation discrimination (no subspace cherry-picking) |
| `mirror_test_v3.py` | Mirror test v3: topic-level perturbation discrimination (Vogel-inspired, semantic directions) |
| `mirror_test_geometry_control.py` | Geometry control: SK fraction of general activation variance vs perturbation response |
| `model_scale_experiment.py` | Model scale experiment: residual structure vs main model size (29M vs 77M) |
| `baseline_battery.py` | Baseline battery: 5-condition controlled comparison (forward, shifted, random_proj, autoencoder, open_loop) |
| `jacobian_analysis.py` | Jacobian analysis: Hessian trace, gradient spectrum, SK subspace alignment at injection point |
| `vit.py` | Vision Transformer for MNIST (same intermediate/cerebellar interface as GPT) |
| `mnist_experiment.py` | MNIST controlled retrain: open-loop vs closed-loop ViT, self-knowledge probes, robustness |
| `mnist_analysis.py` | MNIST residual direction analysis (PCA, digit conditioning) + causal substitution |
| `mnist_baseline_battery.py` | MNIST baseline battery: 5-condition controlled comparison |
| `mnist_distillation.py` | MNIST wake-sleep distillation: single-cycle distillation on low-rank residual domain, digit-discriminative innovation analysis |
| `mnist_distillation_c2.py` | MNIST cycle-2 diagnostic: wake-sleep on cycle-1 outputs, cross-cycle trajectory analysis |
| `mnist_wake_sleep_comparison.py` | MNIST multi-cycle wake-sleep vs compute-matched baselines (CL continuous, OL continuous, periodic KD from external teacher) |
| `mnist_adaptation.py` | MNIST OOD adaptation speed: rotated MNIST fine-tuning learning curves, forgetting resistance, zero-shot OOD across 4 compute-matched conditions |
| `calibration_transfer.py` | Calibration transfer: competence probes (activations → own per-token loss) trained ID, evaluated frozen OOD; also caches OOD corpora |
| `ood_robustness.py` | OOD robustness: perturbation Δloss + Hessian trace across distribution-shifted corpora |
| `distillation.py` | Single-cycle wake-sleep distillation: absorb FM contribution into main model, retrain fresh FM, compare innovation structures |
| `distillation_probes.py` | Post-distillation internalization probes: inter-layer self-predictability, old FM prediction accessibility, cross-model control |
| `mnist_local_loss.py` | MNIST local prediction-error learning: 4-condition comparison (OL, CL, LL, CL+LL) with FM prediction error as auxiliary loss |
| `mnist_local_loss_probes.py` | Representation probes: object-level vs meta-knowledge absorption (prediction probe, orthogonalized residual probe) |
| `mnist_geometry.py` | MNIST computational property geometry: probe orthogonality, compositionality, vector arithmetic across OL/CL/Distilled |
| `language_geometry.py` | Language computational property geometry: same tests as MNIST on GPT models |
| `mnist_precision_weighted.py` | MNIST precision-weighted local loss: Mahalanobis distance replaces MSE, 4-condition comparison (OL, CL, PW, CL_PW) |
| `mnist_learning_gate.py` | MNIST learning gate: bilevel-optimized per-dimension local loss weights via MAML-style virtual update, 6-condition comparison (OL, CL, LL, CL_LL, LG, CL_LG) |
| `README.md` | This file |
| `LLAMA_SCALE_README.md` | [Llama-scale A2A experiment](LLAMA_SCALE_README.md) — activation caching, forward model training, and per-head decomposition on Llama 3.2 1B |
| `REPRESENTATIONAL_DIVERGENCE_README.md` | [Representational divergence analysis](REPRESENTATIONAL_DIVERGENCE_README.md) — CKA, diff PCA, self-knowledge alignment; + [Prediction trust](REPRESENTATIONAL_DIVERGENCE_README.md#prediction-trust-what-form-the-self-knowledge-takes-2026-06-10) appended section (innovation map / error-monitoring geometry) |
| `MIRROR_TEST_README.md` | [Mirror test for neural self-knowledge](MIRROR_TEST_README.md) — perturbation response channeling, robustness gap |
| `MODEL_SCALE_README.md` | [Model scale experiment](MODEL_SCALE_README.md) — residual structure vs main model size, connection to grokking |
| `BASELINE_BATTERY_README.md` | [Baseline battery](BASELINE_BATTERY_README.md) — is forward self-prediction uniquely useful? 5-condition controlled comparison |
| `JACOBIAN_ANALYSIS_README.md` | [Jacobian analysis](JACOBIAN_ANALYSIS_README.md) — Hessian trace predicts robustness; SK alignment null result |
| `MNIST_README.md` | [MNIST experiment](MNIST_README.md) — cross-domain validation: low-rank residual, digit-discriminative structure, 4x robustness gap |
| `OOD_ROBUSTNESS_README.md` | [OOD robustness & calibration transfer](OOD_ROBUSTNESS_README.md) — what kind of self-knowledge survives distribution shift |
| `DISTILLATION_README.md` | [Single-cycle distillation](DISTILLATION_README.md) — wake-sleep knowledge absorption, innovation migration test, ratchet assessment |
| `MNIST_DISTILLATION_README.md` | [MNIST distillation](MNIST_DISTILLATION_README.md) — single-cycle wake-sleep on low-rank residual: digit-discriminative structure collapse, structural internalization at 2-6× language magnitude; + [multi-cycle comparison](MNIST_DISTILLATION_README.md#multi-cycle-comparison-with-compute-matched-baselines-2026-06-13) with compute-matched baselines decomposing val loss (distillation) from robustness (CL co-training) |
| `MNIST_ADAPTATION_README.md` | [MNIST OOD adaptation](MNIST_ADAPTATION_README.md) — rotated MNIST fine-tuning: zero-shot OOD tracks distillation, adaptation speed is uninformative, forgetting resistance tracks CL co-training (three-way dissociation) |
| `MNIST_LOCAL_LOSS_README.md` | [MNIST local prediction-error learning](MNIST_LOCAL_LOSS_README.md) — local loss as auxiliary training signal: regularity ≠ robustness dissociation, 3:1 meta-knowledge dominance, first high-fwd-cos closed-loop condition; + [precision weighting](MNIST_LOCAL_LOSS_README.md#precision-weighted-local-loss-2026-06-16) (negative: FM error structure ≠ task structure); + [learning gate](MNIST_LOCAL_LOSS_README.md#learning-gate-bilevel-optimized-local-loss-2026-06-16) (bilevel-optimized: 35% brittleness reduction, record self-knowledge R²=0.83) |
| `GEOMETRY_README.md` | [Computational property geometry](GEOMETRY_README.md) — probe orthogonality, compositionality, vector arithmetic: distillation converts entangled meta-knowledge into orthogonal object-level knowledge (cross-domain, MNIST + language) |
| `OPEN_LOOP_ANALYSIS_README.md` | [Open-loop analysis details](OPEN_LOOP_ANALYSIS_README.md) — detailed tables/discussion for Runs 1-2, structure analysis, causal substitution, behavioral residual |

**Checkpoint compatibility note**: The `transformer/P_10000000` forward model checkpoint was saved with the original flat `TransformerForwardModel` API (top-level `ln1`, `q_proj`, etc.). The code was later refactored to use `ForwardBlock`/`blocks` for multi-layer support. The analysis scripts (`analyze.py`, `causal_substitution.py`, `behavioral_residual.py`) use a `_LegacyFwdModel` class to load this checkpoint correctly. New checkpoints saved with the current `TransformerForwardModel` will have `blocks.0.*` keys and won't be loadable with the legacy class.

## Modal volume

Results saved to `language-reduction-data` volume:

```
/data/a2a_forward/
├── P_10000000/              # MLP run (post_embed → post_block1)
│   ├── model.pt
│   ├── fwd_model.pt
│   └── results.json
└── transformer/P_10000000/  # Transformer run (post_block0 → post_block1)
    ├── model.pt
    ├── fwd_model.pt
    └── results.json
```

## CLI

Each stage file has its own `@app.local_entrypoint()`, so you run each stage directly. All CLI args are supported via Modal's auto-generated flags.

```bash
cd experiments/

# Default: transformer forward model, post_block0 → post_block1
modal run a2a_forward/stages.py::train --n-tokens 10000000 --n-steps 10000

# MLP forward model (for comparison)
modal run a2a_forward/stages.py::train --n-tokens 10000000 --n-steps 10000 \
  --fwd-type mlp --predict-from post_embed --predict-to post_block1

# Closed-loop training
modal run --detach a2a_forward/stages.py::loop_train --n-tokens 10000000 --n-steps 10000

# Analysis
modal run a2a_forward/analyze.py::analyze --n-tokens 10000000
modal run a2a_forward/analyze.py::loop_analyze --n-tokens 10000000

# Other stages (each file has a `main` entrypoint)
modal run --detach a2a_forward/controlled_retrain.py --n-tokens 10000000
modal run --detach a2a_forward/mirror_test.py --n-tokens 10000000
modal run --detach a2a_forward/model_scale_experiment.py --n-tokens 100000000 --n-layer 8 --n-head 8 --n-embd 512
```

## What model.py does

`model.py` contains a minimal GPT-2 with `return_intermediates=True` support on `GPT.forward()`. When set, it returns a third value: a dict mapping `"post_embed"`, `"post_block0"`, ..., `"post_blockN"` to their activation tensors. This is backward-compatible — existing callers that don't pass the flag get the same `(logits, loss)` tuple. This file was originally part of the `language_reduction` experiment and is now a local copy.

## Structure analysis (2026-05-25)

**Code**: `analyze.py` | **Details**: [OPEN_LOOP_ANALYSIS_README.md](OPEN_LOOP_ANALYSIS_README.md#structure-analysis-2026-05-25)

The forward model achieves near-perfect attention pattern cosine similarity with block1's heads (0.989–0.998 for heads 0–2) while having zero weight cosine similarity — it found a completely different parameterization that produces the same function, explained by attention's gauge symmetry ((QR)(KR)^T = QK^T). CKA between forward model and block1 outputs is 0.98. The residual is full-rank and diffuse (effective rank 199.8/256, top-1 PC explains only 2.4%) — the opposite of grokking's low-rank Fourier solution. Residual correlates with token frequency (training exposure) but is flat across LM loss quartiles (prediction difficulty).

**Reproduction**: `modal run a2a_forward/analyze.py::analyze --n-tokens 10000000`

### Run 3: 2-layer transformer forward model, 3-layer gap (2026-05-25)

A 2-layer transformer (1 head/layer, 64-dim, 660K params) predicting post_block0 → post_block3 (3 main model layers).

| Metric | 1L fwd, 1-layer gap | 2L fwd, 3-layer gap |
|---|---|---|
| Forward model params | 330K | 660K |
| Final cosine sim | 0.972 | 0.935 |
| Final MSE | 0.003 | 0.015 |
| Residual norm | 0.85 | 1.93 |
| Residual-LM corr | -0.05 | -0.03 |

Despite doubling the forward model's parameters, cosine dropped from 0.972 to 0.935 — the 3-layer gap is genuinely harder to approximate. MSE bottomed at 0.013 (step ~4400) then climbed back to 0.015 by step 10K as the main model continued learning computation the forward model couldn't track. The main model also overfit during training (train LM loss 4.5 vs val 5.2), and the forward model's rising MSE tracked this divergence.

Residual-LM correlation remains near zero (-0.03), consistent with the 1-layer result: forward model errors reflect capacity limits, not prediction difficulty.

**Reproduction**: `modal run a2a_forward/stages.py::train --n-tokens 10000000 --n-steps 10000 --fwd-n-layer 2 --predict-from post_block0 --predict-to post_block3`

### Causal substitution: replacing block1 with the forward model (2026-05-26)

**Code**: `causal_substitution.py` | **Details**: [OPEN_LOOP_ANALYSIS_README.md](OPEN_LOOP_ANALYSIS_README.md#causal-substitution-replacing-block1-with-the-forward-model-2026-05-26)

The forward model recovers ~94% of block1's KL contribution (KL_sub=0.062 vs KL_abl=1.094). Degradation is strikingly uniform across all behavioral categories (KL_sub ranges 0.04–0.07) — no behavior-specific catastrophic failure, consistent with the full-rank/diffuse residual.

**Reproduction**: `modal run a2a_forward/causal_substitution.py --n-tokens 10000000`

### Behavior-conditioned residual analysis (2026-05-26)

**Code**: `behavioral_residual.py` | **Details**: [OPEN_LOOP_ANALYSIS_README.md](OPEN_LOOP_ANALYSIS_README.md#behavior-conditioned-residual-analysis-2026-05-26)

The residual reflects *computational complexity*, not *task difficulty*. The forward model struggles most at delimiter tracking (before closers: d=+0.84) and distributed attention, and least at local/focused computations (sentence starts: d=-0.85). Even controlling for block1 contribution magnitude, attention entropy correlates with residual norm (r=+0.33) — the single compressed head specifically fails on multi-source attention integration. Prediction difficulty has negligible effect (d < 0.1).

**Reproduction**: `modal run a2a_forward/behavioral_residual.py --n-tokens 10000000`

## Run 4: Closed-loop cerebellar training (2026-05-25)

**Full writeup**: [CLOSED_LOOP_README.md](CLOSED_LOOP_README.md)

Closed the cerebellar loop: the 2-layer forward model's predictions (post_block0 → post_block3) are fed back into the main model's residual stream via a learned gated projection (`CerebellarGate`, 65.8K params, zero-initialized). Injected after block 1. Forward model trains on MSE only (no LM gradient); main model + gate train on LM loss.

**Key results**:

1. **The injection helps LM loss**: Δ = -0.05 to -0.08 nats consistently (1.6% relative, ~9% perplexity reduction). Negative at every eval step across 10K training steps.

2. **The gate opens wide**: projection weight norm grew linearly 0.08 → 3.03 throughout training, never saturating. The model never stopped finding useful structure in the prediction.

3. **Forward model quality degrades**: cosine 0.935 (open-loop) → 0.897 (closed-loop). The injection changes the model's computation, creating a moving target the forward model can't fully track.

4. **The model develops self-knowledge**: Linear probes predicting the forward model's residual (actual - predicted) from post_block3 show R²=0.44 for the closed-loop model vs R²=0.28 for the open-loop baseline — a 59% improvement. A controlled retrain (Run 6) confirmed this gap is not an lr artifact and showed it is distributed across all layers via backprop, with the early-layer result (R²=0.21 vs 0.02 at post_block0) being the cleanest signal. The self-knowledge is encoded in the residual's *direction* (what kind of computation was missed), not its *magnitude* (how much was missed).

5. **The model becomes dependent on the injection**: Without the injection, the closed-loop model's LM loss is 0.11 nats worse than the open-loop baseline (confirmed in Run 6 controlled comparison). The model learned to complement the prediction rather than internalize it — the "wake-sleep consolidation" problem.

**Reproduction**: `modal run a2a_forward/stages.py::loop_train --n-tokens 10000000 --n-steps 10000 --lr 3e-4 --predict-from post_block0 --predict-to post_block3 --fwd-n-layer 2 --inject-after-block 1`

## Run 5: Forward model capacity scaling sweep (2026-05-28)

**Full writeup**: [SCALING_SWEEP_README.md](SCALING_SWEEP_README.md)

Froze the main model and trained forward models at 5 capacity points (1% to 22% of main model) on the same frozen activations, all predicting post_block0 → post_block3. Tests the bias-to-variance transition: does the residual shift from capturing computational novelty to epistemic novelty as capacity increases?

**Key results**:

1. **The forward model saturates at ~10% capacity**: Both 10% (3.2M params) and 22% (6.3M params) reach cosine 0.999 with identical residual norms. The capacity-sufficient regime begins around 10%, consistent with the main model's transformer blocks being only ~2.4M params.

2. **Computational novelty effects shrink**: Delimiter tracking (d_BC) drops from +0.78 at 1% to +0.30 at 10%. Sentence-start effects (d_SS) drop from -0.58 to -0.37. The bias term is shrinking as predicted.

3. **r(res,LM) stays at zero — but this is the wrong metric**: The residual-LM loss correlation never becomes positive. However, the closed-loop experiment (Run 4) already showed the prediction signal improves LM loss through downstream processing. The residual's informativeness lives in its 256-dimensional direction, not its scalar norm. Collapsing to a norm discards the signal.

4. **The residual is inherently high-rank in language**: Effective rank stays above 235/256 at all capacity points. Unlike grokking (where the Fourier solution is low-rank), language computation is distributed across all dimensions. Biologically consistent: the cerebellum's output is high-dimensional; the thalamus filters it before cortex processes it.

**Reproduction**: `modal run a2a_forward/scaling_sweep.py --n-tokens 10000000 --n-steps 10000 --predict-from post_block0 --predict-to post_block3`

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

## Run 6: Controlled retrain — eliminating the lr/seed confound (2026-05-29)

**Full writeup**: [CONTROLLED_RETRAIN_README.md](CONTROLLED_RETRAIN_README.md)

Retrained both open-loop and closed-loop models with identical lr (3e-4), seed (42), initial weights, and training data order. The only difference is whether the cerebellar loop is closed.

**Key results**:

1. **The injection helps LM loss** (replicates Run 4): Δ = -0.05 to -0.08 nats consistently. Gate opened 0.08 → 3.01. Robust to lr/seed control.

2. **The R²=0.44 vs 0.28 gap is real, not an lr artifact**: Controlled retrain gives R²=0.42 vs 0.26, closely replicating the original.

3. **Self-knowledge is distributed across all layers via backprop**: The residual vector probe gap is uniform — Δ R² = +0.185 at post_block0 (pre-injection), +0.161 at post_block3 (post-injection). This rules out the hypothesis that self-knowledge is built by downstream layers processing the injection signal. Instead, backpropagation distributes the information through all layers. The early-layer result (R²=0.21 vs 0.02 at post_block0) is the cleanest evidence: the model's first layer reorganized to make the forward model's blind spots linearly accessible.

4. **Self-knowledge is directional, not magnitude-based**: The vector probe gap (+0.18) is 6× the scalar gap (+0.03). The model encodes *what kind* of computation was missed, not *how much*.

5. **The model becomes dependent on the injection**: Without injection, the closed-loop model is 0.11 nats worse than the open-loop baseline. The model offloads predictable computation to the forward model rather than internalizing it.

**Reproduction**: `modal run a2a_forward/controlled_retrain.py --n-tokens 10000000 --n-steps 10000 --predict-from post_block0 --predict-to post_block3 --fwd-n-layer 2`

## Causal probes of the self-map (2026-05-28)

**Full writeup**: [CAUSAL_PROBES_README.md](CAUSAL_PROBES_README.md)

Causal follow-ups testing how the self-map works and how the injection helps. Key findings:

- **Magnitude gating is not the mechanism**: Steering along the residual-norm direction has zero novelty-specific effect on injection reliance (Test 2).
- **Help is organized by residual direction, not magnitude**: Residual direction clusters organize injection help 5× more than norm octiles (η²=0.0017 vs 0.0003). The injection helps most at focused-attention positions (mean help = +0.153, nearly 2× average).
- **The directional form of self-knowledge is untested causally**: The scalar/1-D causal tests were blind to directional structure. A directional causal test (steering/patching along residual vector clusters) would test whether the model's directional self-knowledge is functionally used.

## Directional causal steering (2026-05-30)

**Full writeup**: [DIRECTIONAL_STEER_README.md](DIRECTIONAL_STEER_README.md)

Tests whether the directional self-knowledge is causally used, not just encoded. Derives per-cluster steering directions from a multivariate probe (post_block1 → residual vector, R²=0.23), then steers along each direction and measures whether the effect on injection help is cluster-specific.

**Result: Outcome B with partial A for specific clusters.** The model partially uses directional self-knowledge, with strongest selectivity for the focused-attention error type.

- **Steering produces real effects**: Cluster direction slopes are ~4× larger than random controls (mean ~0.004 vs std ~0.001). Not noise.
- **Partial diagonality**: |diag|/|off| ratio = 1.33, diagonal enrichment = 0.160 (uniform = 0.125). Four of eight clusters show selectivity >1.5.
- **Focused-attention cluster is the standout**: Cluster 4 (the focused-attention cluster, frac max_attn>0.5 = 0.21) has selectivity = 2.41 — the strongest direction-specific effect. This is the same cluster with the highest baseline injection help (+0.126, 2× average). The model has learned to discriminate the error type where the injection is most useful.
- **Caveat — direction overlap**: The W@c steering directions have mean off-diagonal cosine 0.41. Steering along one direction partially steers along others, attenuating the diagonal signal. The 1.33 ratio likely underestimates the true selectivity.
- **Interpretation**: The 4-layer non-looped GPT has only 2 layers downstream of the injection to act on directional self-knowledge. It learned the highest-value discrimination (focused attention) and left the rest coarse. The looped transformer would give the model more computational depth to act on the information it already encodes.

**Reproduction**: `modal run a2a_forward/directional_steer.py --n-tokens 10000000 --predict-from post_block0 --predict-to post_block3 --fwd-n-layer 2 --inject-after-block 1`

## Representational divergence analysis (2026-06-02)

**Full writeup**: [REPRESENTATIONAL_DIVERGENCE_README.md](REPRESENTATIONAL_DIVERGENCE_README.md)

Uses CKA, PCA on activation differences, and subspace alignment to characterize *how* the closed-loop model's representations differ from the open-loop model — not just whether self-knowledge is present (Run 6), but whether the dominant representational change *is* the self-knowledge.

**Key results**:

1. **CKA diverges monotonically**: 0.972 at post_embed → 0.727 at post_block3. Even post_block0 (pre-injection) drops to 0.882.

2. **Early-layer change is concentrated, late-layer change is diffuse**: Post_block0 has the lowest effective rank (164.6/256) and highest top-10 concentration (16.7%). Later layers approach full-rank (eff_rank 219-227).

3. **Divergence-to-self-knowledge alignment grows through the network**: At post_block3, the top-5 divergence PCs capture 2.42x more residual variance than random directions. At post_block0, the ratio is barely above 1 (1.10x). The extra-self-knowledge overlap at post_block0 is actually *below* random (0.49x) — the early-layer change is orthogonal to self-knowledge.

4. **Two distinct reorganization phenomena**: Early layers changed extensively via backprop for general-purpose reasons (self-knowledge is a side-effect in low-variance directions). Late layers reorganized primarily along self-knowledge dimensions (the dominant change *is* the self-knowledge). This resolves the apparent contradiction from Run 6, where Δ R² was uniform across layers — the *amount* of self-knowledge is uniform, but its *relationship to the dominant representational change* differs qualitatively by layer.

**Reproduction**: `modal run --detach a2a_forward/representational_divergence.py --n-tokens 10000000 --predict-from post_block0 --predict-to post_block3 --fwd-n-layer 2 --inject-after-block 1`

## Mirror test for neural self-knowledge (2026-06-02)

**Full writeup**: [MIRROR_TEST_README.md](MIRROR_TEST_README.md) (covers v1, v2, v3)

Three iterations of a perturbation-based mirror test (Gallup, 1970 analog). v1 measured SK-subspace channeling (partially explained by geometry); v2 dropped subspace dependence and measured compensatory response (8% dampening, 2× robustness gap, but no active opposition — structural regularization, not deliberate self-correction); v3 used topic-specific perturbation directions and found CL models produce proportionally more topic-specific responses (diag/off-diag ratio 5.2–5.7 vs 4.6 OL). The robust finding across all three versions is the 2-3× robustness gap — CL models absorb perturbations with less loss degradation. The self-knowledge is in the weights (CL-M ≥ CL+M), though CL+M shows perturbation-specific enrichment beyond its general geometry.

**Reproduction**: `modal run --detach a2a_forward/mirror_test.py --n-tokens 10000000 --predict-from post_block0 --predict-to post_block3 --fwd-n-layer 2 --inject-after-block 1`

## Model scale experiment: residual structure vs main model size (2026-06-03)

**Full writeup**: [MODEL_SCALE_README.md](MODEL_SCALE_README.md)

Tests whether the full-rank, diffuse residual observed at 29M params is a property of language or of small models on language. Trains 29M (4L/4H/256D) and 77M (8L/8H/512D) models on identical data (100M tokens), with capacity-matched forward models (~1–1.5%), predicting post_block0 → post_block1.

**Key results**:

1. **Every rank-normalized metric points the same direction**: The 77M residual is more concentrated — lower relative effective rank (91.8% vs 93.8%), lower proportional rank-for-50%/90% variance, and 64% higher top-1 PC variance (2.86% vs 1.74%). Seven independent measures, all consistent.

2. **The 77M model's computation is much more predictable**: Cosine 0.994 vs 0.980, despite 2× tighter attention compression in the forward model (8x vs 4x). A bigger model develops more regular layer computation.

3. **Sentence-start effects collapse at 77M**: d=-0.35 vs d=-1.05. The forward model no longer struggles with sentence starts — the main model's sentence-start processing is regular enough to compress. Delimiter tracking persists as the dominant residual signal at both scales.

4. **Early-training rank dip at 77M**: The 77M model shows a transient rank decrease (69.8% → 66.9%) at step 500 before climbing — a weak echo of grokking's rank compression during the phase transition. The 29M model does not show this.

5. **Consistent direction, small magnitude**: The rank reduction (93.8% → 91.8%) is modest, as expected for a 2.7× scale-up on a DGP as complex as language. The grokking analogy holds in direction if not magnitude — the Fourier solution concentrates in ~15/128 dimensions; language computation concentrates very slowly as the model improves.

**Reproduction**:
```bash
# 29M
modal run --detach a2a_forward/model_scale_experiment.py \
  --n-tokens 100000000 --n-steps 30000 --n-layer 4 --n-head 4 --n-embd 256
# 77M
modal run --detach a2a_forward/model_scale_experiment.py \
  --n-tokens 100000000 --n-steps 30000 --n-layer 8 --n-head 8 --n-embd 512
```

## Run 7: Extended co-training — 50K steps (2026-06-01)

**Full writeup**: [EXTENDED_TRAINING_README.md](EXTENDED_TRAINING_README.md)

Same controlled retrain design (identical lr/seed/init) but trained for 50K steps (~45 epochs) to observe long-run dynamics.

**Key results**:

1. **The injection benefit grows monotonically — 6× over training**: -0.08 nats at 10K → -0.48 nats at 50K, with no sign of saturation. The forward model's cosine drops (0.93 → 0.82) while the benefit grows — the model extracts increasing value from a less accurate prediction. The prediction is being used as a structured reference frame, not a literal preview.

2. **No differential overfitting**: Both models overfit at the same rate (train-val gap: -4.86 at 50K for both). Validation loss on a static dataset measures compression of a fixed distribution, not representational quality.

3. **Self-knowledge probes narrow but persist at early layers**: Vector Δ R² drops from +0.185 to +0.094 at post_block0, disappears at post_block3. Scalar probes flip sign (the overfit open-loop model's memorized token statistics correlate with residual norm).

4. **Co-specialization deepens without saturating**: Injection benefit, dependency, and forward model cosine all evolve monotonically over 50K steps with no convergence.

**Reproduction**: `modal run --detach a2a_forward/extended_training.py --n-tokens 10000000 --n-steps 50000 --predict-from post_block0 --predict-to post_block3 --fwd-n-layer 2`

## Run 7b: 10% forward model, 15K steps (2026-06-02)

**Full writeup**: [EXTENDED_TRAINING_README.md](EXTENDED_TRAINING_README.md) (Run 7b section)

Same controlled retrain design but with a 10% forward model (3L, 4H, 128D, 3.2M params) instead of the 2.7% model (660K params). Trained for 15K steps. Tests whether a capacity-sufficient forward model changes the dynamics.

**Key results**:

1. **Train-val gaps still identical**: Even with a forward model exceeding each main model block's 2.4M params, the gaps differ by only 0.02–0.05 nats. The injection does not interact with memorization dynamics at any forward model capacity.

2. **Zero net val loss benefit — pure computation relocation**: The closed-loop model with injection achieves the same val loss as the open-loop model (5.569 vs 5.573 at 15K). The injection benefit (-0.213) is exactly offset by dependency (+0.210). The 10% model enables full offloading rather than supplementation.

3. **U-shaped injection benefit**: -0.27 at step 500 (huge early benefit), contracts to -0.08 at step 5K, then recovers to -0.21 at 15K. Opposite of the 1% model's monotonic growth.

4. **Self-knowledge probes 2× stronger**: Vector Δ R² = +0.34 at post_block0, +0.47 at post_block3 (vs +0.19 and +0.16 with 1% model). Scalar probes also show large gaps (+0.33 to +0.45), unlike the 1% model where scalar gaps were negligible. When the forward model captures 99.7% of computation, the residual magnitude itself becomes informative.

5. **Capacity sweet spot for net benefit**: The 1% model is imperfect enough that the main model can't fully offload, yielding a small net benefit. The 10% model is too capable — the main model offloads aggressively and becomes fully dependent with no net gain. The optimal capacity for injection benefit lies between 1% and 10%.

**Reproduction**: `modal run --detach a2a_forward/extended_training.py --n-tokens 10000000 --n-steps 15000 --predict-from post_block0 --predict-to post_block3 --fwd-n-layer 3 --fwd-d-head 128 --fwd-n-head 4 --fwd-mlp-mult 4`

## Baseline battery: is forward self-prediction uniquely useful? (2026-06-04)

**Full writeup**: [BASELINE_BATTERY_README.md](BASELINE_BATTERY_README.md)

Tests whether the benefits of forward model injection are specific to injecting a prediction of the model's own future computation, or whether any structured injection produces the same effects. 5 conditions with identical lr/seed/init: open-loop, forward prediction, causally shifted prediction (k=10), frozen random projection of post_block0, and autoencoder (same architecture, reconstructing post_block0 instead of predicting post_block3).

**Key results**:

1. **Robustness is uniquely strong with forward prediction.** The forward model takes only 41% of open-loop's loss degradation from identical perturbations (2.4× improvement). The autoencoder and random projection produce 1.5–1.6× improvement — the forward model's robustness gain is nearly twice the baselines'.

2. **Self-knowledge depth profile separates forward prediction from baselines.** All injections cause some representational reorganization at early layers (Δ R² ≈ +0.17–0.19 at post_block0). But only forward prediction maintains self-knowledge through the full depth of the model (+0.161 at post_block3). Baselines decay steeply (+0.055–0.068 at post_block3). The forward model's self-knowledge at the deepest layer is 2.4–2.9× the baselines'.

3. **LM loss and robustness are dissociated.** The autoencoder produces the *largest* injection benefit (−0.092 nats, vs −0.084 for forward) but *worse* robustness (0.636 vs 0.413). Robustness is not a byproduct of receiving useful additional information — it's specifically a consequence of the injection being a prediction of the model's own future computation.

4. **The shifted baseline validates position-specificity.** Shifting predictions by 10 positions destroys nearly all effects: zero injection benefit, minimal self-knowledge, lowest gate norm. The model learns that non-position-specific predictions aren't useful and ignores them.

**Reproduction**: `modal run --detach a2a_forward/baseline_battery.py::a2a_baseline_battery --n-tokens 10000000 --n-steps 10000 --predict-from post_block0 --predict-to post_block3 --fwd-n-layer 2 --inject-after-block 1`

## Jacobian analysis: spectral structure of robustness (2026-06-05)

**Full writeup**: [JACOBIAN_ANALYSIS_README.md](JACOBIAN_ANALYSIS_README.md)

Formalizes the robustness result via the Jacobian of the map from activations at the injection point through to the loss. Computes loss gradient norms, Hessian trace (via Hutchinson's estimator), gradient covariance spectrum, and self-knowledge subspace alignment — all on the controlled retrain models.

**Key results**:

1. **Hessian trace quantitatively predicts robustness.** tr(H) at the injection point is 0.453x OL for CL+M and 0.383x OL for CL-M. The Hessian prediction E[ΔL] = ε²/(2d) · tr(H) matches empirical ΔL to within 8% at every perturbation magnitude. The 2-2.5x robustness gap is fully explained by the curvature of the loss landscape at the injection point.

2. **Loss gradient norms are 30% smaller.** CL+M = 0.694x OL, CL-M = 0.797x OL. First-order sensitivity is uniformly reduced.

3. **Gradient spectrum is LESS concentrated for CL (negative result).** Effective rank: 170 (OL) → 178 (CL+M) → 183 (CL-M). The CL model's sensitivity is more uniformly distributed, not concentrated into fewer directions. This is the opposite of the original hypothesis.

4. **Self-knowledge subspace does NOT align with sensitivity (negative result).** At every subspace dimension tested (10–128), the SK subspace captures ≤ random baseline fraction of gradient energy (enrichment 0.92–1.08x). The self-knowledge directions and loss-sensitivity directions are orthogonal.

5. **The correct mechanism is a uniformly flatter loss landscape.** The robustness comes from total curvature being reduced everywhere, not from sensitivity being organized along self-knowledge directions. Consistent with the "division of labor" interpretation: the forward model pre-supplies predictable computation, leaving blocks 2-3 with less functional load and a smoother input-output mapping.

**Reproduction**: `modal run --detach a2a_forward/jacobian_analysis.py::a2a_jacobian_analysis --n-tokens 10000000 --predict-from post_block0 --predict-to post_block3 --fwd-n-layer 2 --inject-after-block 1`

## MNIST experiment: cross-domain validation (2026-06-07)

**Full writeup**: [MNIST_README.md](MNIST_README.md)

Adapts the full A2A forward model experiment to MNIST classification via Vision Transformer (4L/4H/128D ViT, 4x4 patches, 0.80M params). Tests whether the cerebellar phenomena generalize from autoregressive language modeling to image classification. Includes controlled retrain, residual analysis, causal substitution, and baseline battery.

**Key results**:

1. **The residual is low-rank and digit-discriminative -- opposite of language.** Effective rank = 18.3/128 (vs 199.8/256 in language). Top-1 PC explains 15.7% (vs 2.4%). All top-5 PCs discriminate digit identity (eta^2 = 0.14--0.23). The forward model misses specific class-conditional computation, not a uniform capacity shortfall. Cross-digit residual cosine similarity mirrors visual similarity (3<->8 = 0.84, 1<->3 = -0.45).

2. **Self-knowledge probes are 3x stronger.** Δ R^2 = +0.52 at post_block0 (vs +0.19 in language). The low-rank residual provides specific directions for the model to encode, making self-knowledge more precise. CL R^2 reaches 0.63 at the first layer (vs 0.21 in language).

3. **Robustness gap is 4x (vs 2-2.5x in language).** CL/OL loss degradation ratio = 0.244 at eps=1.0. At eps=2.0, the gap widens to 12x. The MNIST model's loss landscape at the injection point is dramatically flatter.

4. **Causal substitution shows non-uniform degradation.** The forward model recovers 85.6% of blocks 1-3 KL contribution (vs 94% for 1 block in language). Degradation varies 13x across digits (KL_sub from 0.004 for digit 0 to 0.053 for digit 4). In language, degradation was uniform across all behavioral categories.

5. **Baseline battery: less differentiation than language.** Forward prediction and autoencoder produce identical robustness ratios (0.244 vs 0.245). Self-knowledge R^2 gaps between conditions are smaller. The low-rank residual (18 dimensions) is easily captured by any injection; in language, only forward prediction produces the organizational pressure for the model to encode all 200 residual dimensions.

6. **Residual rank reflects DGP complexity.** Grokking (mod addition): rank ~15, Fourier-aligned. MNIST: rank 18, digit-discriminative. Language (29M): rank 200, diffuse. Language (77M): rank 235, slightly less diffuse. The forward model's residual dimensionality directly reflects the computational complexity of the task as modeled by the main model.

**Reproduction**:
```bash
# Main experiment
modal run --detach a2a_forward/mnist_experiment.py::main
# Residual analysis + causal substitution
modal run --detach a2a_forward/mnist_analysis.py::main
# Baseline battery
modal run --detach a2a_forward/mnist_baseline_battery.py::main
```

## Calibration transfer & OOD robustness: what kind of self-knowledge survives distribution shift? (2026-06-09)

**Full writeup**: [OOD_ROBUSTNESS_README.md](OOD_ROBUSTNESS_README.md)

Two experiments taking the baseline battery models out of distribution (Wikipedia, Python code, French, open-web-math, shuffled tokens; 2M GPT-2 tokens each, cached to the volume). No retraining — all measurements on the existing checkpoints.

**Experiment A — calibration transfer (negative)**: Linear competence probes (activations → the model's own forthcoming per-token loss; no forward model in the measurement) trained ID, evaluated frozen OOD. Closing the loop does not produce epistemic self-knowledge: all conditions are identical ID, OOD retention shows only the generic any-used-injection effect (forward ≈ random_proj ≈ autoencoder), and output entropy beats every activation probe both ID and OOD. The model does not transferably know where it will fail.

**Experiment B — OOD robustness (positive, confirms the paper's §6 prediction)**: Identical perturbations (16 fixed directions at the injection point) and Hessian traces measured per corpus. The forward model's robustness is distribution-invariant (Δloss ratio 0.43–0.51, tr(H) ratio 0.38–0.45 on every natural corpus), while the autoencoder's collapses off-manifold — on code its advantage vanishes entirely (Δloss 0.93×, tr(H) **1.02×** = same curvature as open-loop). The forward/autoencoder gap widens with shift (1.51× ID → 1.81× code), as predicted by the computational-function vs activation-manifold account. The shifted condition inverts to worse-than-open-loop on code (1.11×). Robustness remains in the weights OOD (forward_inj ≈ forward).

**Joint interpretation**: the self-knowledge from closing the loop is *computational, not epistemic; distribution-invariant, not data-bound*. First direct evidence for the paper's conclusion-section claim that forward prediction encodes the model's computational function while the autoencoder encodes the ID activation manifold.

**Reproduction**:
```bash
modal run --detach a2a_forward/calibration_transfer.py::cache_ood_tokens
modal run --detach a2a_forward/calibration_transfer.py::a2a_calibration_transfer
modal run --detach a2a_forward/ood_robustness.py::a2a_ood_robustness
```

## Single-cycle wake-sleep distillation (2026-06-11)

**Full writeup**: [DISTILLATION_README.md](DISTILLATION_README.md)

Tests whether knowledge distillation can force the main model to internalize the FM's contribution, and whether a fresh FM trained on the consolidated model finds different innovation structure (indicating genuine computational change) or the same structure (indicating re-equilibration).

Loads the controlled retrain CL checkpoint. Phase 1: freeze teacher (CL+injection), train student (same weights, no injection) on KL + NTP loss for 5K steps. Phase 2: train fresh FM (seed=137) on consolidated model for 10K steps. Phase 3: compare innovation structures.

**Key results**:

1. **Distillation fully closed the dependency gap**: +0.083 → -0.005 nats (105.6% closed). The distilled model without injection slightly outperforms the teacher with injection. Cost vs OL: only +0.030 nats.

2. **Innovation directions migrated**: Top PCs of original and fresh FM residuals are near-orthogonal (cos 0.04–0.33). Fresh FM's top-5 subspace captures only 4.7% of original FM's top-5 variance. But behavioral conditioning is stable (r=0.81) — the same positions are hard, just hard in different directional ways.

3. **Residual became more diffuse**: Effective rank 212.3 → 227.9, top-1 PC 2.8% → 1.3%. No discrete abstraction was extracted on flat webtext — the ratchet grinds smoothly rather than clicking.

4. **Robustness partially retained**: Distilled model ratio 0.942 vs OL (CL was 0.892). Roughly half the CL robustness advantage survived dissolving the two-model system.

5. **Late-layer self-knowledge survived distillation**: The distilled model encodes the fresh FM's residual better than OL at late layers (Δ R² = +0.164 at post_block3) despite never training with the fresh FM. Early layers show no advantage (+0.001 at post_block0). Consistent with the representational divergence finding that late-layer reorganization aligns with self-knowledge while early-layer reorganization is orthogonal to it. Confound: probe target is defined on the distilled model's own activations; a cross-model control would eliminate this.

**Reproduction**:
```bash
modal run --detach a2a_forward/distillation.py::main \
  --n-tokens 10000000 --distill-steps 5000 --retrain-steps 10000
```

## MNIST local prediction-error learning (2026-06-15)

**Full writeup**: [MNIST_LOCAL_LOSS_README.md](MNIST_LOCAL_LOSS_README.md)

Tests the simplest version of the [local prediction-error learning](../../ideas/local_prediction_error_learning.md) idea: use `λ · MSE(sg(FM(post_block0)), post_block3)` as an auxiliary loss for the main model. The FM prediction is a frozen target (stop-gradient); gradient flows through post_block3 into the main model. Four conditions with identical seed/lr/init: OL, CL (injection only), LL (local loss only, λ=1.0), CL_LL (both).

**Key results**:

1. **LL learns faster** (+2.5pp accuracy at step 500, +1.4pp final). The local loss gives gradient to all 50 positions × 128 dims per example; the classification loss only gives gradient to [CLS]. Final val_loss is 31% lower (0.088 vs 0.127). FM cosine reaches 0.997 — the "be predictable" pressure made computation nearly perfectly FM-compressible, and this was beneficial.

2. **CL_LL reduces dependency by 63%** (0.008 vs 0.022). The continuous internalization pressure counteracts offloading. Injection benefit is also 5× smaller — the model needs the injection less.

3. **Regularity ≠ robustness** (strong negative). LL is **13× more sensitive** to perturbations than OL. The local loss makes computation precise but brittle. CL's robustness comes specifically from injection experience (training with structured additive signals), not from computational regularity. This revises the Jacobian analysis interpretation.

4. **CL_LL has the highest self-knowledge** (R² = 0.76–0.79, vs CL 0.63–0.73, OL 0.11–0.38). LL has zero self-knowledge — the FM residual is too small to predict. Self-knowledge requires a meaningful residual, which requires the injection to introduce irreducible unpredictability.

5. **CL_LL is the first high-fwd-cos closed-loop condition** (0.985 vs 0.903 for CL). The local loss stabilizes co-training by pushing the model toward FM-predictable computation, preventing the moving-target problem.

6. **The absorbed knowledge is predominantly meta-knowledge** (3:1 over object-level at early layers). Representation probes distinguishing "what the FM predicts" (prediction probe) from "where the FM is wrong, ⊥ prediction direction" (orthogonalized residual probe) show CL_LL's advantage over CL at post_block0 is Δ R² = +0.31 for ortho_residual vs +0.10 for prediction. The local loss gradient literally carries the meta-knowledge signal: `∂post_block3/∂early_act · (post_block3 − FM_pred)`.

**Reproduction**:
```bash
modal run --detach a2a_forward/mnist_local_loss.py::a2a_mnist_local_loss
modal run --detach a2a_forward/mnist_local_loss_probes.py::a2a_mnist_ll_probes
```

## Computational property geometry probes (2026-06-16)

**Full writeup**: [GEOMETRY_README.md](GEOMETRY_README.md)

Tests whether CL and post-distillation models develop more compositionally structured representations — the analog of classic word embedding geometry (king − man + woman = queen) applied to computational properties. Probes for data identity (digit/token frequency), FM residual norm/direction, block contribution, and prediction difficulty at each layer, then measures probe direction orthogonality, centroid additivity, and vector arithmetic accuracy. Three conditions (OL, CL, Distilled) on both MNIST and language models.

**Key results**:

1. **Distilled has the most orthogonal early-layer representations** (both domains). Mean pairwise |cos| between probe directions at post_block0: MNIST 0.085 (vs 0.172 OL, 0.233 CL), language 0.067 (vs 0.093 OL, 0.107 CL). The distilled model encodes data identity and computational properties in more independent directions.

2. **CL has the least orthogonal late-layer representations** (both domains). Mean |cos| at post_block3: MNIST 0.125 (vs 0.109 OL), language 0.246 (vs 0.124 OL). CL meta-knowledge creates entanglement — in language, fm_res_norms × block_contrib reaches |cos| = 0.618 at post_block3. The FM's reliability is entangled with block contribution and token frequency.

3. **Vector arithmetic improves with distillation at the final layer** (both domains). Analogy completion cosine (high_res_category_A − low_res_category_A + low_res_category_B ≈ high_res_category_B): MNIST 0.970 Distilled vs 0.911 OL; language 0.773 Distilled vs 0.743 OL. The "computational state" direction is more consistent across data categories after distillation.

4. **Compositionality direction flips between domains.** CL is least compositional in MNIST (0.830 vs 0.937 Distilled, digit × residual R²) but most compositional in language (0.892 vs 0.867 OL, freq × residual R²). Reflects the residual structure: MNIST's low-rank, digit-discriminative residual entangles digit and residual; language's full-rank, diffuse residual keeps frequency and residual naturally independent.

5. **The dissociation rules out "more self-knowledge → more organized."** CL has the strongest self-knowledge R² (0.63–0.73 in MNIST) yet the worst orthogonality. Distillation converts entangled meta-knowledge into orthogonal object-level knowledge — the geometric consequence of the meta vs object-level distinction from the local loss probes.

**Reproduction**:
```bash
modal run --detach a2a_forward/mnist_geometry.py::a2a_mnist_geometry
modal run --detach a2a_forward/language_geometry.py::a2a_language_geometry
```

## Next steps

1. **Wake-sleep consolidation (partially validated)**: The single-cycle distillation experiment confirmed that distillation closes the dependency gap and produces genuine innovation migration. The next step is *iterated* cycles: re-run closed-loop training on the consolidated model, distill again, and test whether the ratchet clicks on a DGP with hierarchical structure (e.g. multi-step arithmetic or code). The model scale experiment predicts residual concentration at larger scale, which should enable discrete level shifts.
2. **Cross-model self-knowledge control**: Train separate fresh FMs on each model's own activations (OL, CL, distilled), then probe each for its own FM's residual. Eliminates the confound in the distillation self-knowledge probes.
3. **Gauge symmetry check on innovation migration**: Compute cos(f_orig(x), f_fresh(x)) on shared inputs to conclusively rule out gauge symmetry as an explanation for the near-orthogonal residual PCs.
2. **Looped transformer**: The natural architecture for cerebellar injection — inject at each recurrence step, get adaptive compute for free. Would give the model more computational depth to act on its self-knowledge at inference time. The directional steering results specifically motivate this: the model encodes directional self-knowledge it can only partially use with 2 downstream layers.
3. **Continual learning / transfer evaluation**: The 10% model produces 2× stronger self-knowledge with zero val loss benefit — the reorganization is invisible to NTP on a static dataset. A plausible natural test: freeze both models (open-loop and closed-loop trained), fine-tune on a novel task, and measure adaptation speed and interference. This could test whether the self-knowledge translates to functional capability that validation loss can't detect.
4. **Thalamic filtering**: Replace the linear `CerebellarGate` with a learned nonlinear gate (MLP). May help extract directional structure from the high-rank residual.
5. **Self-regulation**: Freeze the forward model at a checkpoint and use the residual as a regularization signal (as validated in grokking). Test whether this prevents overfitting or distributional drift.
6. **Orthogonalized directional test**: Re-run the directional steering with Gram-Schmidt-orthogonalized W@c directions to control for the 0.41 mean cosine overlap. Would give a cleaner estimate of true directional selectivity.
7. **Model scale 350M**: Third data point for the model scale experiment. The 29M → 77M comparison shows consistent residual concentration across all metrics. A 350M model (~24L/16H/1024D on 500M+ tokens) tests whether the trend continues, accelerates, or saturates. See [MODEL_SCALE_README.md](MODEL_SCALE_README.md).
8. **Harden the OOD robustness result**: (a) direct manifold-displacement check — autoencoder reconstruction MSE per corpus should rise with shift severity and peak on code; (b) bootstrap CIs from the saved per-direction Δloss arrays; (c) a second baseline-battery seed to firm up the code-corpus numbers. See [OOD_ROBUSTNESS_README.md](OOD_ROBUSTNESS_README.md).

[^private]: Not mirrored: this link points to a document in the private lab repo (the roadmap, the queue, an unrun spec, reading notes, or a conversation). See the top-level README for what is held back and why.
