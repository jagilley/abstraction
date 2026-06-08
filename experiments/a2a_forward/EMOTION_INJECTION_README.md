# Emotion injection experiment: evaluative vs predictive self-referential signals

**Code**: `emotion_injection.py`
**Date**: 2026-06-07
**Result**: Negative for emotion injection; confirms forward prediction specificity

## Motivation

The A2A forward model is an analog of the cerebellum: it predicts the main model's future computation and feeds that prediction back. The thalamus routes many other signal types into the cortex besides cerebellar predictions — notably emotional/evaluative signals from the amygdala and reward system. If the forward model produces robustness and self-knowledge by being a *structured self-referential signal*, would an *evaluative* self-referential signal ("how difficult will this be?") produce analogous effects? Or are the benefits specific to predicting the model's own future computation?

Ilya Sutskever (Dwarkesh Podcast, 2026) discussed emotions as value functions: simple, compressed evaluative signals that are enormously useful precisely because of their simplicity. Our capacity scaling results (Run 7b) showed that a 1% forward model produces net benefit while a 10% model doesn't — imperfection is load-bearing. Emotions are firmly in the "1% regime": low-dimensional, compressed, imperfect summaries. This experiment tests whether that analogy has computational substance.

## Design

5 conditions with identical lr (3e-4), seed (42), initial weights, and data order:

| Condition | Injection | Params | Biological analog |
|---|---|---|---|
| open_loop | None | — | No subcortical input |
| forward | CerebellarGate(fwd_model(post_block0)) | 660K + 66K | Cerebellum |
| loss_scalar | EmotionGate(loss_pred_1d(post_block0)) | 33.5K + 0.5K | Scalar emotion (1D valence) |
| loss_embed | EmotionGate(loss_pred_8d(post_block0)) | 34.4K + 2.3K | Multi-dim emotion (8D) |
| forward_plus_loss | Both forward + scalar loss | 660K + 66K + 33.5K + 0.5K | Cerebellum + emotion |

**Loss predictor**: a small MLP (LayerNorm → Linear(256→128) → GELU → Linear(128→embed_dim)) trained to predict per-token cross-entropy from post_block0 activations. For embed_dim=8, a Linear(8→1) head maps the embedding to a scalar for training, but the full 8-dim embedding is what gets injected.

**EmotionGate**: Linear(embed_dim→256), zero-initialized, analogous to CerebellarGate. For the scalar version, this learns a single direction in activation space scaled by predicted loss.

All conditions co-train a forward model (post_block0 → post_block3, 2L, 660K params) for comparable self-knowledge probing.

## Results

### LM loss

| Condition | Val LM (with inj) | Val LM (no inj) | Injection Δ | Δ vs open-loop |
|---|---|---|---|---|
| open_loop | 5.398 | 5.398 | — | — |
| forward | 5.422 | 5.506 | **-0.084** | +0.024 |
| loss_scalar | 5.398 | 5.414 | -0.017 | -0.001 |
| loss_embed | 5.397 | 5.427 | -0.030 | -0.001 |
| forward_plus_loss | 5.425 | 5.516 | **-0.091** | +0.027 |

The forward model's injection benefit (-0.084) is 5x the scalar loss predictor's (-0.017) and 2.8x the 8-dim version's (-0.030). The combination (-0.091) improves marginally over forward-only.

The loss predictor conditions have slightly better absolute val loss than open-loop (Δ vs OL ≈ -0.001), while the forward conditions are slightly worse. The loss predictor provides a small genuine improvement without the dependency cost the forward model incurs.

### Self-knowledge probes

| Layer | OL R² | forward Δ R² | loss_scalar Δ R² | loss_embed Δ R² | fwd+loss Δ R² |
|---|---|---|---|---|---|
| post_block0 | 0.024 | **+0.185** | +0.032 | +0.063 | **+0.204** |
| post_block1 | 0.062 | **+0.200** | +0.035 | +0.068 | **+0.219** |
| post_block2 | 0.158 | **+0.185** | +0.044 | +0.074 | **+0.202** |
| post_block3 | 0.260 | **+0.161** | +0.037 | +0.060 | **+0.172** |
| **Depth ratio (b3/b0)** | — | **0.87** | **1.18** | **0.96** | **0.84** |

The forward model produces 3-5x more self-knowledge. But the **depth ratio** for loss_scalar is 1.18 — self-knowledge *grows* from early to late layers, opposite the forward model (0.87). The evaluative signal is processed differently: the model builds up its response through successive layers rather than having self-knowledge distributed uniformly via backprop. The 8-dim embedding is intermediate (0.96).

The combination is roughly the sum of parts (Δ R²=+0.20 vs forward's +0.19 + loss_scalar's +0.03 = +0.22). No superadditivity.

### Perturbation robustness

| Condition | ||R|| | cos(R,δ) | Δloss | Δloss/OL |
|---|---|---|---|---|
| open_loop | 1.87 | +0.714 | +0.0109 | 1.000 |
| **forward** | **1.73** | **+0.778** | **+0.0045** | **0.413** |
| loss_scalar | 1.88 | +0.704 | +0.0105 | 0.963 |
| loss_embed | 1.86 | +0.707 | +0.0097 | 0.894 |
| **forward_plus_loss** | **1.73** | **+0.774** | **+0.0046** | **0.425** |

**The sharpest result.** The loss predictor provides essentially zero robustness (0.96x). The forward model provides 2.4x (0.41x). The combination doesn't improve on forward-only. Robustness comes specifically from pre-supplying computation, not from receiving any structured signal.

The response norm confirms: only forward-containing conditions produce smaller responses (0.925x). Loss predictor conditions have response norms at or above open-loop (1.00-1.01x). The loss predictor doesn't change the loss landscape geometry at all.

### Gate dynamics

| Condition | Final gate norm |
|---|---|
| forward (cerebellar) | 3.01 |
| loss_scalar (emotion) | 0.15 |
| loss_embed (emotion) | 0.30 |
| forward_plus_loss (cerebellar) | 3.00 |
| forward_plus_loss (emotion) | 0.18 |

The model opens the cerebellar gate 20x wider than the emotion gate. In the combined condition, the cerebellar gate reaches the same norm as when acting alone (3.0), while the emotion gate reaches only 0.18 — the forward prediction dominates, and the evaluative signal contributes marginally.

### Loss predictor quality

| Step | loss_scalar R² | loss_embed R² |
|---|---|---|
| 200 | 0.183 | 0.189 |
| 2000 | 0.190 | 0.189 |
| 4000 | 0.146 | 0.144 |
| 6000 | 0.110 | 0.110 |
| 8000 | 0.037 | 0.041 |
| 10000 | -0.079 | -0.054 |

The loss predictor peaks at R²≈0.19 early in training, then steadily degrades to negative. As the main model's computation becomes more distributed and complex, per-token loss becomes harder to predict from early-layer activations. The signal the emotion gate receives gets *worse* over training — opposite of the forward model's prediction, whose *benefit* grows even as its cosine degrades.

## Interpretation

### Why the evaluative signal is so much less useful than the predictive signal

The forward model provides a 256-dimensional preview of the model's future computation. This is a rich, position-specific signal the model can use to pre-compute the predictable component of its remaining layers' work. The loss predictor provides, at most, an 8-dimensional evaluation of difficulty. Even if perfectly accurate, "this will be hard" doesn't tell the model *what* to compute differently — it only says *how much* to compute.

The 2x gap between loss_embed (8-dim, -0.030 nats) and loss_scalar (1-dim, -0.017 nats) shows that additional evaluative dimensions help, but the returns are very quickly diminishing. Extrapolating, even a 256-dim evaluative embedding wouldn't approach the forward model's -0.084, because the forward model provides information about the *content* of the computation, not just its difficulty.

### The depth ratio inversion

The loss_scalar depth ratio of 1.18 (self-knowledge growing through depth) is qualitatively different from the forward model's 0.87 (uniform via backprop). A scalar difficulty signal needs multiple layers of processing to become useful — the model builds up its response through successive blocks. A 256-dim activation preview can be acted on immediately after injection, so the gradient pressure to encode self-knowledge is felt at all layers equally.

### The robustness null result

The most decisive finding. The forward model produces 2.4x robustness by pre-supplying predictable computation, flattening the loss landscape (Hessian trace 0.45x OL, per Jacobian analysis). The loss predictor pre-supplies nothing — it just says "this will be hard." The loss landscape geometry is unchanged. Any injection that doesn't pre-supply computation won't produce the robustness effect.

This is consistent with the baseline battery result where the autoencoder (which *does* pre-supply computation, as a compressed reconstruction) produced 1.6x robustness — less than forward prediction but more than nothing. The loss predictor, which pre-supplies no computation at all, produces 1.0x: no robustness improvement whatsoever.

### The biological implication

If we take the analogy seriously: cerebellar predictions and emotional evaluations play fundamentally different roles. The cerebellum reorganizes cortical representations deeply (large self-knowledge, uniform depth profile, robustness through flatter loss landscapes). Emotions provide modest, marginal guidance (small LM improvement, shallow self-knowledge, no robustness). In the brain, emotional signals may serve primarily to *modulate* how cortical computation is allocated (attention, urgency, resource priority) rather than to *restructure* cortical representations the way cerebellar predictions do.

The emotion gate barely opening (0.15 vs 3.0) maps onto the biological observation that emotional influence on cortical processing is more modulatory than structural: emotions bias what gets computed, but they don't fundamentally change the computational architecture the way the cerebellum does.

This doesn't mean emotions aren't important — in the brain, their importance comes from modulating *which computations happen* (attention, action selection, memory prioritization), not from restructuring *how computation is done* within the cortex. That modulatory role isn't well-captured by our injection-into-the-residual-stream paradigm, which tests structural effects.

## Reproduction

```bash
modal run --detach a2a_forward/emotion_injection.py::a2a_emotion_injection \
  --n-tokens 10000000 --n-steps 10000 \
  --predict-from post_block0 --predict-to post_block3 \
  --fwd-n-layer 2 --inject-after-block 1
```
