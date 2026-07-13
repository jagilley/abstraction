# Conditional Novelty Bottleneck: Weight-Grounded Novelty Detection via Activation Prediction

**Status**: Idea (not yet implemented)
**Date**: 2026-05-10
**Builds on**: GLP residual semantics work, VPD (Goodfire/Sharkey et al., 2026), CLS theory

## One-liner

A small meta-model predicts LLM activations from a coarse decomposition of the LLM's own weights; the prediction residual is a novelty signal grounded in the model's actual computational structure.

## The problem, progressively stated

### Starting point: CLS in AI needs a hippocampal/PFC analog

Complementary learning systems theory says fast learning (hippocampus) and slow learning (neocortex) serve different roles. The hippocampus performs novelty detection and sparse coding, allowing immediate encoding of new experiences that are later consolidated into cortical representations without catastrophic interference. AI has no such system. We want one.

A meta-model over activations (like the GLP) can detect novelty: project OOD activations onto the learned manifold, and the residual captures what's new. We've shown these residuals are semantically meaningful. But they're **ungrounded** — they tell you "this is far from typical in these directions," where the directions are defined by the statistics of the residuals, not by the model's internal concept structure. They don't say "this is like concept X, modified in way Y."

### The grounding problem

What we want is novelty defined *relative to what the model knows* — its concepts, its computational structures — not relative to activation statistics. The GLP learns P(activations), a statistical manifold. What we need is something that knows *why* activations happen, not just what they look like.

Weight space is where the model's knowledge actually lives. A concept like "marsupial" isn't really a point in activation space — it's a function the model computes, defined by its weights. Ideally, the novelty signal would be grounded in weight-space structure.

### Why directly modeling activations is the wrong objective

The GLP models the full activation manifold. This requires more parameters than exist in the base model (the Llama 1B GLP has ~3B parameters). This is suspicious — the hippocampus is tiny relative to the cortex. A novelty detector should be *smaller* than the thing it monitors, because most of what the base model computes isn't novel.

More fundamentally, modeling P(activations) is agnostic about what's worth paying attention to. It doesn't natively know that a particular activation deviates from the model's concept structure. It just knows what's statistically typical.

### The conditional information bottleneck reframe

Instead of: learn the manifold (dense) → derive novelty as a byproduct (sparse, expensive)

What if we: learn to directly produce sparse codes that capture what's new, *conditioned on what the model already knows*?

This is a conditional information bottleneck. Compress the activation, but conditioned on the model's existing knowledge, so the bottleneck only needs to transmit the *surprise*. The sparsity falls out naturally because most activations are mostly explicable by existing representations.

The key insight: **you don't need to externally specify what the model's concept axes are. The training procedure discovers them, because the bottleneck is pressured to exploit every regularity the base model already captures.**

> **Related (2026-07-12): [efference_copy_cancellation.md](efference_copy_cancellation.md).** "Transmit only the surprise, conditioned on what's known" is the same principle that doc applies to the a2a forward-model loop — but as a *forward-path wiring* (subtract the FM's prediction, propagate the residual: predictive coding / efference copy) rather than a *detection/coding* read-out. This doc grounds the "condition" in weight space (VPD); that doc grounds it in the co-trained forward model's forecast. Two uses of the same conditional-information-bottleneck idea — a novelty *detector* here, a novelty *forward path* there.

### The size constraint as the grounding mechanism

A large meta-model can afford to memorize activation statistics without discovering conceptual structure. A very small meta-model *cannot* — it must find compressions. The most efficient compressions of a model's activation space should be its actual concepts, because concepts are the high-leverage regularities that the model's weights impose on its activations.

The parameter budget becomes the grounding mechanism. Not "learn about causality" but "be so small that you have to discover it."

This mirrors biology: the hippocampus is tiny relative to the cortex, and that size constraint forces concept-level encoding without anyone defining what concepts are.

### The side information question

What serves as the "condition" — the representation of "what the model already knows" — in the bottleneck?

Options considered:
- **Token embeddings as codebook**: Natural language as a sparse dictionary. Appealing information-theoretically, but token embeddings are the model's *input* vocabulary, not its *conceptual* vocabulary. The concept of "tiger" lives deep in the network, not in the embedding.
- **Intermediate-layer concept prototypes**: Richer, but still defined by activation statistics, not functional structure.
- **The GLP's learned manifold**: Already captures nonlinear geometry, but still statistical, not causal.
- **Full base model forward pass**: Maximally grounded but expensive.

The problem with all activation-space side information: activations are symptoms of computation, not the computation itself. The grounding problem persists at the level of side information.

### Weight-space side information resolves grounding

A meta-model that takes tokens as input and predicts activations would be redundant with the base model — it would just be distillation. What we want instead is to leverage the *weights themselves* as the concept basis.

This leads to the key proposal.

## The proposal: VPD-based activation prediction

### Architecture

1. **Run VPD at coarse granularity** on the base model (one-time offline cost). This decomposes each weight matrix into K rank-1 subcomponents that are mechanistically faithful — each component IS a computational mechanism, not an arbitrary slice. VPD's adversarial ablation objective ensures components correspond to real functional units.

2. **Train a small router**: given input features (token embeddings or shallow activations), predict the causal importance values — which VPD components will be active and how strongly for this input.

3. **Predict activation**: apply only the predicted-active components to the input. This is cheap — sparse rank-1 operations.

4. **Compute residual**: actual activation (from base model forward pass) minus predicted activation. This is the novelty signal.

### Why this works

- **Grounded by construction**: VPD components ARE the base model's weights, decomposed into mechanistically faithful units. The residual is "what the model computes that isn't explained by any discovered mechanism at this granularity." This isn't statistical deviation — it's computational novelty.
- **Small meta-model**: The concept basis lives in the VPD decomposition (which is the base model's own weights, rearranged). The only learned component is the router, which is tiny.
- **Implicit concept discovery**: VPD finds mechanisms without supervision. No need to define concepts or extract them post hoc.
- **Coarse granularity is a feature**: With K << rank of weight matrices, VPD captures only the highest-leverage mechanisms. This is exactly the concept-level granularity we want. Everything not captured goes into the delta-residual.
- **Non-redundant**: The router doesn't learn to compute activations from scratch (that would be distillation). It learns to *predict routing* over pre-discovered computational units. Much smaller task.

### What this gives us (primitive version)

- A grounded novelty signal: large residual = "the model is doing something that doesn't decompose into known coarse mechanisms"
- Weight-space interpretability: you know *which* mechanisms you expected to fire and which didn't account for the actual computation
- A natural hippocampal analog: the router + VPD components play the role of the entorhinal cortex providing structured input to a novelty detection system

### What this doesn't yet give us (refinements for later)

- **Relational structure in novelty**: The primitive version says "mechanisms A, B, C were expected but something else happened." It doesn't yet decompose novelty into "like concept X, modified along axis Y." That requires additional structure on the residual.
- **Memory and binding**: Detecting novelty is step one of CLS. Storing it as a retrievable sparse code and consolidating into weights is the full loop. The residual is naturally the thing you'd store, but the consolidation machinery is separate.
- **Continuous/predictive coordinates**: The biological MEC provides smooth, path-integrable, multi-scale coordinates — richer than a discrete mechanism dictionary. This may or may not matter for AI systems, which don't face the same resource scarcity constraints that push brains toward continuous representations. Discrete symbolic representations are arguably a *strength* of language models, not a limitation.

## Key design questions

1. **How coarse can VPD go and still give meaningful mechanisms?** The paper decomposes a 67M model at relatively fine granularity. At very coarse K (say, tens of components per layer for a 1B model), do you still get interpretable computational units?

2. **Can a lightweight router predict causal importance from cheap features?** VPD's own causal importance function uses activations. If the router needs the full forward pass to predict importance, novelty detection isn't cheaper than just running the model. The router needs to work from shallow features (embeddings, early layers).

3. **What's the right training signal for the router?** Train it to match VPD's causal importance values on a diverse corpus? Or train end-to-end for activation prediction, letting it discover its own notion of "which components matter"?

4. **How does the delta-residual from VPD interact with the prediction residual?** At coarse granularity, the VPD delta-component (what the decomposition can't explain) is large. For some inputs, the delta might be doing real computational work. This is itself a novelty signal — inputs that rely heavily on the undecomposed remainder.

## Relationship to existing GLP work

This doesn't replace the GLP — it reframes its role. The GLP demonstrates that activation-space residuals are semantically meaningful. This proposal asks: can we get residuals that are additionally *grounded in the model's computational structure* by predicting activations from weight-space decompositions rather than modeling the activation manifold directly?

If the VPD-based residuals are less semantically rich than GLP residuals (possible, since coarse weight decomposition loses information), the two could be complementary: GLP for rich manifold geometry, VPD-based prediction for grounded novelty.

## Intellectual lineage

- **CLS theory** (McClelland et al.): The hippocampus/neocortex division as fast/slow learning. Motivates the need for a novelty detection system.
- **GLP** (this project): Activation-space meta-modeling via flow matching. Demonstrates semantic residuals but lacks weight-space grounding.
- **VPD** (Bushnaq, Sharkey et al., 2026): Adversarial parameter decomposition into mechanistically faithful subcomponents. Provides the weight-space concept basis.
- **Conditional rate-distortion / information bottleneck** (Tishby et al.): The theoretical frame for "compress, but conditioned on what you already know."
- **Sparse coding / hippocampal pattern separation**: The idea that novelty should be encoded as sparse deviations from existing representations, not as dense statistical objects.
