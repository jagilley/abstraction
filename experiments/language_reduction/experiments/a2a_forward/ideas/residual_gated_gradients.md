# Residual-Gated Gradients: Learn Only What's New

**Date**: 2026-05-25
**Status**: Idea
**Depends on**: feedback loop (not yet implemented), forward model co-training (validated)
**Prior experiment**: [../README.md](../README.md)

## One-liner

Use the forward model's prediction residual to gate gradient updates, so the main model only learns what's genuinely new to it rather than re-learning what it already knows.

## The insight

Standard training: see data point X, compute loss, backprop. The gradient contains everything — what X shares with other data, what's unique to X, and what the model already knows. You rely on SGD averaging to sort generalizable structure from noise. This is wasteful and fragile (Sisyphean collapse is the failure mode where re-learning overwrites generalization).

Residual-gated training: the forward model predicts what the main model *should* compute for X (conditioned on X). The residual = actual computation minus predicted computation. Gate gradients by this residual — the model only updates on its epistemic gap for X, not on X itself.

## Why what you learn is biased toward generalizability

The forward model's capacity bottleneck means it can only represent generalizable computational patterns — it can't memorize instance-specific details. So:

1. **The prediction ≈ the model's current generalizable knowledge for this input.** The forward model captures mechanisms that predict well across diverse inputs.

2. **The residual ≈ computation beyond current generalizable knowledge.** This contains both (a) novel generalizable structure the model hasn't captured and (b) instance-specific noise.

3. **Over many data points, the generalizable component of the residual is correlated (consistent across inputs) while the noise is uncorrelated (cancels).** Same law-of-large-numbers argument as standard SGD, but applied to a signal already enriched for novel content.

The filtering removes the biggest source of non-generalizable gradient noise: re-learning known patterns. What remains has a much higher signal-to-noise ratio for generalizability.

## Connection to catastrophic forgetting / Sisyphean collapse

These failures happen when gradients from new data overwrite old generalizable structure. With residual gating, the forward model captures that old structure, so the residual doesn't contain it — gradients literally can't overwrite computation that's been projected out. The model can learn new things but not un-learn old things. This is a stronger stability guarantee than weight regularization (EWC etc.), which penalizes weight change uniformly rather than selectively protecting generalizable computation.

## The CLS parallel

In complementary learning systems theory, the hippocampus encodes what's novel relative to cortical knowledge. The forward model plays this role: it represents current consolidated knowledge, and the residual is the novelty signal. But instead of replay, the novelty signal gates learning directly — the hippocampus telling the cortex "only update your weights in response to what surprises me."

## The cerebellar parallel

The cerebellum fires strongly during learning — climbing fiber error signals are most active when the system is acquiring new predictive models, not when it's running well-learned ones. This maps directly: large residuals (strong cerebellar error) trigger large gradient updates (strong plasticity), while small residuals (accurate prediction, nothing new here) suppress updates. The biological system already implements residual-gated learning.

## The lag problem

After the main model makes a breakthrough, the forward model doesn't know about it yet. The residual still contains the breakthrough computation, so gradients try to re-learn it. This is probably benign (reinforces what you just learned) but limits efficiency. The biological parallel: the cerebellum lags cortical learning, producing transient spurious novelty signals after insights. Mitigations: fast forward model learning rate, or periodic forward model sync.

## Precision weighting as a second-order refinement

Residual gating biases the *estimator* toward generalizability but doesn't reduce the *variance* of individual updates. The good-vs-bad novelty discrimination from the cerebellar conversations (precision weighting, structural analysis, learned error classification) would layer on top — filtering the residual itself to separate structured signal from diffuse noise. This is the difference between "only learn what's new" and "only learn what's new *and informative*."

## Experimental plan

The grokking setup is the natural testbed:

1. Co-train main model + forward model on (a+b) mod 97 with Zipfian weighting
2. Gate main model gradients by the forward model residual (scale gradients proportional to residual magnitude at each position)
3. Test: does residual-gated training resist Sisyphean collapse *without* a frozen reference?
4. Compare: raw residual gating vs. precision-weighted residual gating vs. frozen-reference baseline

If the forward model's capacity bottleneck automatically prevents memorization gradients from dominating, this validates the "generalizable by construction" claim without needing any external checkpoint.

## Implementation options for gradient gating

Several ways to implement "gate gradients by the residual," from least to most invasive:

- **Loss weighting**: weight the LM loss at each position by the residual magnitude at that position. High residual = learn more from this position. Simple, no architecture change.
- **Gradient scaling**: after computing gradients, scale them by a function of the residual. More flexible but requires gradient hooks.
- **Residual-only loss**: compute loss only on the component of the main model's activations orthogonal to the forward model's prediction. Most aggressive — explicitly projects out "already known" computation.
- **Soft attention**: use the residual vector as an attention mask over the gradient, selectively amplifying updates in directions the forward model missed. Preserves directionality of the novelty signal, not just magnitude.
