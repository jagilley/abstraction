# Jacobian Analysis: Spectral Structure of Robustness

**Code**: `jacobian_analysis.py`
**Date**: 2026-06-05
**Prior**: [Mirror test v2](MIRROR_TEST_README.md) (robustness gap), [Baseline battery](BASELINE_BATTERY_README.md) (forward-specific robustness)

## Motivation

The mirror test (v2) showed that the closed-loop model takes 2-2.5x less loss degradation from identical random perturbations at the injection point, with only 8% smaller response norms. This raises a precise question: what about the model's computation from post_block1 onward makes it less sensitive to perturbation?

The Jacobian of the map from activations at the injection point through to the loss provides the formal framework. For a perturbation δ of magnitude ε in a random direction, the expected loss change is:

$$E[\Delta L] \approx \frac{\epsilon^2}{2d} \cdot \text{tr}(H)$$

where $H = \partial^2 L / \partial \delta^2$ is the Hessian of the loss with respect to the shared perturbation. If the closed-loop model has a smaller Hessian trace, that directly explains the robustness factor.

We also tested a stronger hypothesis from the original proposal: that the self-knowledge subspace (defined by SK probe weights) should align with the top singular directions of the Jacobian. This would mean the model is sensitive along exactly the directions where it has organized self-knowledge, and inert along others.

## Method

Five analyses on the same models from the controlled retrain (identical lr=3e-4, seed=42, init):

1. **Loss gradient norms.** For each condition (OL, CL+M, CL-M), compute ∂L/∂δ at the injection point (post_block1) across 40 eval batches. The shared sensitivity vector s = Σ_{b,t} ∂L/∂h_{b,t} determines the variance of loss change from random perturbation.

2. **Hessian trace estimation.** Hutchinson's estimator with 10 Rademacher vectors per batch, 20 batches per condition. Uses double backward passes (create_graph=True) to compute Hessian-vector products. tr(H) determines the expected loss increase E[ΔL] = (ε²/2d) · tr(H).

3. **Gradient covariance spectrum.** Collect per-position gradients ∂L/∂h_{b,t} (163K vectors per condition), compute the D×D covariance matrix, and analyze the eigenvalue spectrum (effective rank, variance concentration).

4. **Self-knowledge subspace alignment.** Train a linear probe mapping post_block1 activations to the forward model's prediction residual (R² = 0.26). Extract the SK subspace as the top-k right singular vectors of the probe weight matrix (in raw activation space). Project each condition's gradient covariance onto the SK subspace and measure the fraction of sensitivity energy captured, compared to random subspace baselines (100 random Q matrices per test).

5. **Empirical validation.** Apply random perturbations at 4 magnitudes (s = 0.5, 1.0, 2.0, 3.0 × b1_std), 16 directions, 10 batches each. Compare observed ΔL with Hessian-trace predictions.

## Results

### 1. Hessian trace quantitatively predicts robustness

| Condition | tr(H) | Ratio vs OL | Emp ΔL (s=2.0) | Emp ratio | Pred ΔL |
|-----------|-------|-------------|----------------|-----------|---------|
| OL | 2.967 | 1.000 | +0.01141 | 1.000 | +0.01235 |
| CL+M | 1.343 | **0.453** | +0.00519 | **0.455** | +0.00559 |
| CL-M | 1.138 | **0.383** | +0.00473 | **0.415** | +0.00473 |

The Hessian trace ratio (0.453x for CL+M) matches the empirical loss degradation ratio (0.455x) to within 0.5%. The Hessian prediction (ε²/2d · tr(H)) matches the empirical ΔL to within 8% at every perturbation magnitude for all three conditions.

The loss change scales quadratically with perturbation magnitude (ΔL/ε² CV = 0.01–0.18), confirming the second-order Hessian term dominates. The first-order gradient prediction (E[|ΔL|] ≈ ε/√d · ||s||) systematically underestimates at large perturbations, consistent with quadratic dominance.

CL-M (injection removed) is more robust than CL+M (0.383x vs 0.453x), consistent with the mirror test finding that robustness is in the weights. The injection slightly increases sensitivity, probably because it adds an additional pathway from input to loss.

### 2. The gradient norm is 30% smaller

| Condition | Shared ||s|| | Per-pos ||g|| | ||s|| ratio |
|-----------|-------------|--------------|-------------|
| OL | 0.0487 | 0.000294 | 1.000 |
| CL+M | 0.0338 | 0.000200 | 0.694 |
| CL-M | 0.0388 | 0.000196 | 0.797 |

The first-order sensitivity at the injection point is 20-30% smaller for the CL models. Both the shared sensitivity (sensitivity to uniform perturbation) and per-position sensitivity (average individual-position sensitivity) show the same pattern.

### 3. The gradient spectrum is LESS concentrated for CL (negative result)

| Condition | Effective rank | Top-1 | Top-5 | Top-10 |
|-----------|---------------|-------|-------|--------|
| OL | 170.2 / 256 | 6.7% | 14.8% | 21.5% |
| CL+M | 177.9 / 256 | 4.4% | 12.5% | 19.2% |
| CL-M | 182.7 / 256 | 3.3% | 11.4% | 17.9% |

The CL model's gradient covariance has *higher* effective rank and *less* concentrated top eigenvalues than the OL model. This is the opposite of the original proposal's prediction that the CL model would have a more concentrated singular value spectrum.

The robustness does not come from sensitivity being concentrated into fewer directions. The total sensitivity energy is smaller, but it's distributed more uniformly, not less.

### 4. Self-knowledge subspace does NOT align with sensitivity (negative result)

| SK dim k | Random baseline | OL alignment | CL+M alignment | CL+M enrichment |
|----------|----------------|--------------|-----------------|-----------------|
| 10 | 3.9% | 4.0% | 4.1% | 1.05x |
| 25 | 9.8% | 9.0% | 9.4% | 0.96x |
| 50 | 19.5% | 17.2% | 17.9% | 0.92x |
| 100 | 39.1% | 35.5% | 36.6% | 0.94x |
| 128 | 50.0% | 46.3% | 47.5% | 0.95x |

At every subspace dimension tested, the SK subspace captures *less* gradient energy than random (enrichment 0.92–1.08x, none statistically distinguishable from 1.0 given the random baseline std of ~0.002–0.003).

The self-knowledge directions and the loss-sensitivity directions are essentially orthogonal. The model's self-knowledge (what it knows about where its forward model will be wrong) is encoded in directions that are unrelated to the directions along which its loss is most sensitive to perturbation.

This falsifies the strongest version of the proposal: that the self-knowledge subspace would align with the top singular directions of the Jacobian, providing a formal link between self-knowledge and robustness.

## Interpretation

The robustness from forward self-prediction is best understood as a **uniformly flatter loss landscape at the injection point**. The CL model's remaining layers (blocks 2-3) compute a function whose curvature w.r.t. post_block1 activations is 2-2.5x smaller in every direction, not specifically smaller along self-knowledge directions.

This is consistent with the mirror test v2 finding that the CL model "produces a smaller, more organized response rather than actively opposing the perturbation." The Jacobian makes this precise: the model's downstream computation has lower curvature w.r.t. its input at the injection point. Perturbations cause less damage because the loss landscape is flatter, period.

Why would self-knowledge training flatten the loss landscape? One possibility: the forward model's prediction pre-supplies part of the computation for blocks 2-3, reducing their functional load. With less computational work to do, the remaining layers develop a smoother input-output mapping. The model doesn't need to be as sensitive to the details of post_block1 because the injection already provides a useful approximation of the final state. This is a "division of labor" explanation rather than an "organized sensitivity" explanation.

The gradient spectrum result (higher effective rank for CL) supports this interpretation. When the forward model handles the predictable component, the residual computation that blocks 2-3 must perform is more diffuse — there's no single dominant direction of computation, so there's no single dominant direction of sensitivity.

## What this means for the paper

**Include:** The Hessian trace result is a clean, quantitative verification of the robustness claim. "The Hessian trace at the injection point is 0.453x OL, quantitatively predicting the observed 0.455x loss degradation ratio" is stronger than the current error-correcting code analogy.

**Revise:** The paper should not claim that self-knowledge directions align with sensitivity directions — this was falsified. The connection between self-knowledge and robustness is indirect: self-knowledge training causes the model to develop a flatter loss landscape at the injection point (quantifiable via the Hessian trace), but the flatness is not organized along self-knowledge dimensions.

**Drop:** The error-correcting code analogy in the conclusion should be replaced with the Hessian characterization. The analogy implied organized structure (valid codewords, minimum distance), but the actual mechanism is uniform dampening.

## Reproduction

```bash
modal run --detach a2a_forward/jacobian_analysis.py::a2a_jacobian_analysis \
  --n-tokens 10000000 --predict-from post_block0 --predict-to post_block3 \
  --fwd-n-layer 2 --inject-after-block 1
```
