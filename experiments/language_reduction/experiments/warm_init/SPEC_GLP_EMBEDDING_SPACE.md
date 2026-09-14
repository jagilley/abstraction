# Experiment: GLP in Embedding Space for Concept Token Initialization

**Date**: 2026-05-02
**Builds on**: `RESULTS_WARM_INIT_PHASE2.md`, `RESULTS_RESIDUAL_DECOMPOSITION.md`
**Motivation**: `ideas/language_reduction_continual_learning.md`, `conversations/Claude-Learned token invention in language models.md`

## The problem with the current GLP

The existing GLP was trained on [h0; h1] — a 256-D concatenation of both transformer block outputs. This created an "activation-to-embedding projection problem": the GLP's residuals live in a 256-D activation space that has no direct relationship to the 128-D embedding space where token initializations need to live. Phases 1–3 of the warm init experiments never used the GLP for initialization — only for analysis — because there was no clean way to get from GLP residuals back to embeddings.

The contrastive centroid that "solved" warm initialization bypasses the GLP entirely. It's a statistical operation on raw activations (centroid subtraction) that doesn't use the GLP's learned manifold at all.

The original vision from the token invention proposal — that the GLP residual tells you the direction of the missing concept and you could initialize the token from that — has been underexplored.

## Three requirements that dissolve the projection problem

The activation-to-embedding gap is not fundamental. It disappears when three conditions hold simultaneously:

1. **Single-layer GLP** (not concatenation). Train on 128-D features, matching the model's embedding dimensionality. This aligns with the original GLP paper (Gilley et al.), which trains on a single layer (layer 7 of Llama1B) and finds single-layer outperforms multi-layer at matched compute. The [h0; h1] concatenation was a design choice in our implementation that introduced an unnecessary mismatch.

2. **Train on ln_f(h1)** (not raw h1). `ln_f` is the final LayerNorm applied before the output projection. ln_f(h1) is the representation that gets dot-producted with lm_head rows to produce logits — it's the space that directly interfaces with the output projection. Raw h1 is a different distribution (pre-normalization) and wouldn't have the right properties.

3. **Tied weights** (lm_head.weight = wte.weight). This makes the output projection space identical to the embedding space. Without tying, these diverge (R²=0.27 in our untied model), and a vector that works as a detector (lm_head row) doesn't work as an embedding (wte row). With tying, ln_f(h1) space IS embedding space — the same vectors serve both roles.

All three are necessary. Drop (1) and you're in 256-D. Drop (2) and you're in the wrong 128-D distribution. Drop (3) and the output and embedding spaces have diverged.

## What this enables

With all three conditions, the GLP's outputs live natively in embedding space:

- **Residuals** (activation − on-manifold projection) are candidate embedding vectors or corrections
- **On-manifold projections** via `sample_on_manifold` use the full nonlinear flow geometry, not linear centroid subtraction
- **Velocity field directions** at off-manifold points indicate which dimensions constitute the novelty

At queen-prediction positions, the GLP residual directly answers "what embedding direction is missing from this model's vocabulary" — without any projection step.

## Why this should improve on the contrastive centroid

| | Contrastive centroid (current) | GLP on ln_f(h1) (proposed) |
|---|---|---|
| Baseline subtracted | Hand-selected control prompts | Learned manifold from all training data |
| Operation | Linear (centroid difference) | Nonlinear (flow-matching denoiser) |
| Granularity | Single averaged direction | Per-prompt residuals preserving compositional structure |
| Supervision needed | Must know what concept you're looking for | Could detect off-manifold concepts automatically |
| Geometric knowledge used | None — raw statistics | Full manifold geometry, velocity field |

The residual decomposition experiment already showed that per-prompt GLP residuals at queen positions decompose interpretably into gender × royalty, with proportions matching prompt semantics. That compositional structure is averaged away by the contrastive centroid. A GLP operating in embedding space would preserve it.

More importantly for the general program: the contrastive centroid requires hand-designed prompts for a known concept. The GLP manifold can identify off-manifold positions automatically — any position where the residual is large and structured represents a concept the model is "reaching for" but can't compress. This is the detection step needed for unsupervised concept minting.

## Experiment design

### Phase 1: Train the GLP

Train a GLP on ln_f(h1) activations from the tied-weights τ=0.3 model (2 layers, 128-dim, tied). Extract ln_f(h1) at all token positions from the training corpus (same data used to train the language model). Train a flow-matching denoiser on these 128-D features following the original GLP paper's architecture.

Evaluate: reconstruction cosine similarity, FID against held-out activations. Compare against the existing [h0; h1] GLP to verify that single-layer training doesn't lose important information for the queen-prediction use case.

### Phase 2: GLP-residual initialization

At the 12 queen-prediction positions, compute:
- GLP residual: ln_f(h1) − sample_on_manifold(ln_f(h1))
- Average over stochastic denoising passes to stabilize direction

Use the mean residual (scaled to mean wte row norm) as the queen embedding initialization. Compare against:
- Contrastive centroid (current best)
- Analogy (king + gender_shift)
- Random baseline

Evaluation: zero-shot queen prediction rank, specificity at control positions, zero-shot generation quality. Then full-model SFT + A/B RL per `ideas/language_reduction_continual_learning.md`.

### Phase 3: Automatic concept detection (stretch)

Scan the training corpus for positions with large, structured GLP residuals — positions where the model is off-manifold in a coherent direction. Cluster these residuals. Each cluster is a candidate concept that the model represents compositionally but hasn't compressed into a token. This tests the unsupervised detection step of the continual learning vision without hand-selecting prompts.

## Connection to the broader program

The token invention proposal envisions: detect → mint → integrate → (optionally discard). Phases 1–3 of the warm init experiments solved the minting step via contrastive centroid on tied weights, but without using the GLP. This experiment asks whether the GLP can do the same job better — and more importantly, whether it can do it without supervision, which is required for the general continual learning case.

The integration step (A/B RL on matched pairs from original vs. denoised corpus) is unchanged. The GLP tells the model WHERE the concept is. The RL tells the model TO USE IT.
