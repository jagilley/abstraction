# Forward Model Swap Test: Is Self-Knowledge FM-Specific or General?

**Code**: `forward_model_swap.py`
**Date**: 2026-07-01
**Prior experiments**: [Controlled retrain](CONTROLLED_RETRAIN_README.md) (Run 6), [Distillation](DISTILLATION_README.md), [Geometry](GEOMETRY_README.md)

## Motivation

The controlled retrain (Run 6) established that co-training with a forward model produces self-knowledge: the CL model's activations linearly encode the FM's prediction residual much better than OL (Δ R² = +0.16-0.19 across layers). But the FM is a *specific* compressed approximation of the model's computation. Is the self-knowledge about the model's own computation in general, or about the specific FM it co-trained with?

The forward self-models paper (§3.4) shows that different FMs find completely different parameterizations of the same function (zero weight cosine, near-perfect attention pattern cosine) due to gauge symmetry. But this symmetry applies to the FM's *internal* representations, not its *outputs*: different FM seeds produce genuinely different predictions (and therefore residuals) in the main model's fixed activation space. Residual cosine similarity across seeds is ~0.83 — substantial overlap but not identity. If the CL model's self-knowledge is general, it should transfer to fresh FMs that capture most of the same computational structure.

A secondary question: does distillation (absorbing the FM's contribution into the main model's weights) make the model's computation more self-transparent — easier for *any* fresh FM to characterize?

## Design

Three main models from existing checkpoints (identical seed/lr/init, the only difference being training condition):
- **OL**: open-loop baseline
- **CL**: closed-loop (injection after block 1)
- **Distilled**: post-distillation (FM contribution absorbed into weights, no injection)

For each main model, train 3 fresh FMs (seeds 100, 200, 300) on that model's frozen activations, with identical architecture to the original FM (2L, 1H, 64D, 660K params). Then probe each main model for:
- **FM-original**: the co-trained FM from the controlled retrain (CL only)
- **Own fresh FMs**: fresh FMs trained on this model's activations
- **Cross-model FMs**: fresh FMs trained on other models' activations
- **Own ensemble**: residual from the mean prediction across 3 own-FM seeds (averages out seed-specific local minima, isolating the shared "what any FM of this architecture consistently misses" component)

All probes are linear (activations → residual vector, measured by R²). The probe is retrained from scratch for each target.

## Results

### Q1: CL self-knowledge is FM-specific (3× ratio)

CL model probed with FM-original vs fresh FMs trained on CL's own activations:

| Layer | FM-original R² | Own fresh R² (mean) | Delta |
|---|---|---|---|
| post_block0 | 0.209 | 0.012 | +0.197 |
| post_block1 | 0.262 | 0.055 | +0.207 |
| post_block2 | 0.343 | 0.133 | +0.210 |
| post_block3 | 0.421 | 0.212 | +0.209 |

The CL model encodes FM-original's residual 3× better than fresh FMs' residuals (mean R² 0.309 vs 0.103). The delta is remarkably uniform across layers (+0.20 at every layer). The Δ R² = +0.16-0.19 from the controlled retrain was almost entirely co-adaptation with the specific FM.

### Q2: Distillation did NOT increase self-transparency (null)

Each model probed with fresh FMs trained on its own activations:

| Layer | OL | CL | Distilled | Dist - OL |
|---|---|---|---|---|
| post_block0 | 0.011 | 0.012 | 0.012 | +0.001 |
| post_block1 | 0.044 | 0.055 | 0.053 | +0.009 |
| post_block2 | 0.137 | 0.133 | 0.130 | -0.006 |
| post_block3 | 0.232 | 0.212 | 0.209 | -0.023 |

No model is more self-transparent than any other. Own-FM R² is essentially identical across OL, CL, and Distilled (mean 0.106, 0.103, 0.101). Distillation produced more orthogonal representations (the geometry result), but this doesn't translate into being more self-predictable by an arbitrary fresh FM.

### Q2b: Ensemble probes confirm — no shared-component advantage

Ensemble residual = target − mean(FM predictions across 3 seeds). This averages out seed-specific local minima, isolating what any FM of this architecture consistently misses.

| Layer | OL ensemble | CL ensemble | Dist ensemble | CL - OL | Dist - OL |
|---|---|---|---|---|---|
| post_block0 | 0.009 | 0.010 | 0.010 | +0.001 | +0.001 |
| post_block1 | 0.045 | **0.058** | **0.056** | **+0.013** | **+0.011** |
| post_block2 | 0.149 | 0.147 | 0.143 | -0.003 | -0.006 |
| post_block3 | 0.256 | 0.235 | 0.232 | -0.020 | -0.024 |

Mean: OL = 0.115, CL = 0.113, Dist = 0.110. No CL or Distilled advantage for the shared component.

### The injection-point crossover

The ensemble deltas show a clean sign flip at the injection boundary (injection is after block 1):

- **Pre-injection** (post_block0, post_block1): CL - OL is positive (+0.001, +0.013)
- **Post-injection** (post_block2, post_block3): CL - OL is negative (-0.003, -0.020)

Distilled tracks the same pattern (+0.001, +0.011, -0.006, -0.024).

This crossover maps onto the two distinct types of representational change from the [representational divergence analysis](REPRESENTATIONAL_DIVERGENCE_README.md): early layers reorganized in directions orthogonal to self-knowledge (general-purpose), late layers reorganized along self-knowledge dimensions (FM-specific). Post-injection layers encode FM-specific meta-knowledge — "where does *this particular FM* fail, and should I trust it?" — which naturally doesn't transfer to fresh FMs. Pre-injection layers encode something more general about the model's own computational structure, and there's a small positive signal there.

### FM prediction quality

How well does each FM predict each model's computation (cosine similarity):

| FM trained on | → OL | → CL | → Distilled |
|---|---|---|---|
| OL | **0.921** | 0.837 | 0.840 |
| CL | 0.801 | **0.934** | 0.919 |
| Distilled | 0.810 | 0.921 | **0.933** |
| FM-original | 0.802 | 0.899 | 0.914 |

CL and Distilled are computationally similar (cross-prediction cosine 0.92), both very different from OL (cross-prediction 0.80-0.84). FM-original (trained during co-training on a moving target) is worse than fresh FMs trained on the final CL checkpoint (0.899 vs 0.934).

### Residual similarity across FM seeds

| Model | Seed pair cosine |
|---|---|
| OL | 0.842-0.846 |
| CL | 0.831-0.832 |
| Distilled | 0.834-0.835 |

Different FM seeds agree ~83% about what's hard to predict. The remaining ~17% reflects different local minima in the FM's optimization landscape, not different computational structure in the main model. This is a training artifact (multiple local optima in a nonlinear function class), not a fundamental limit: at 10% FM capacity, prediction cosine reaches 0.999 (capacity scaling sweep), leaving little room for seed-dependent variation.

### CL model: FM-original vs ensemble targets

| Layer | FM-original | Own ensemble | Ensemble with original |
|---|---|---|---|
| post_block0 | 0.209 | 0.010 | 0.028 |
| post_block1 | 0.262 | 0.058 | 0.078 |
| post_block2 | 0.343 | 0.147 | 0.173 |
| post_block3 | 0.421 | 0.235 | 0.269 |

Including FM-original in the ensemble (averaging with 3 fresh FMs) recovers some signal because the ensemble target is partially aligned with FM-original's specific residual, but the R² is far below FM-original alone. The self-knowledge is tuned to FM-original's exact error pattern, not the shared component.

## Interpretation

### Two separable components of self-knowledge

The swap test, combined with prior geometry and distillation results, reveals that "self-knowledge" is two distinct phenomena:

**1. FM-specific meta-knowledge (post-injection layers).** The CL model encodes "where does *this particular* compressed view of my computation fail?" This is strong (R² = 0.42 at post_block3), functionally useful (drives gate selectivity, meta-learning, robustness), and does not transfer across FM instances. It lives primarily at post-injection layers, where the model processes the injection and decides what to trust. This is the dominant component of what we've been measuring as "self-knowledge."

**2. FM-independent object-level knowledge (pre-injection layers, structural properties).** Distillation produces representations that are more orthogonal (mean |cos| 0.085 vs 0.172 OL in MNIST), more compositional (centroid additivity R² 0.937 vs 0.927), and support better vector arithmetic (analogy cosine 0.970 vs 0.911) — all measured without any FM in the loop (geometry probes). The swap test's pre-injection positive signal (+0.013 at post_block1 for CL ensemble) is consistent with this but too small to be definitive on its own. These structural properties are genuine changes to the model's weights that don't depend on any specific FM.

### What "self-knowledge" means

The self-knowledge from closing the loop is **lens-specific, not lens-independent**. The CL model learned to complement one particular compressed summary of its own computation. A different FM (equally valid, ~83% overlapping in what it captures) would require its own co-adaptation.

This parallels the biological case: self-knowledge in humans is calibrated through one's particular cognitive apparatus. You know what surprises you, where your intuitions are reliable, what you find hard — but this knowledge is specific to your actual monitoring system, not a universal self-model. Switching to a completely different self-monitoring system would require recalibration, even if the underlying computation hasn't changed.

The FM-independent component (structural regularization from distillation) is the closest thing to "knowing what you know" in the abstract — it's a property of the weights themselves, not of any specific external reference. But it manifests as better-organized representations, not as an explicit self-model. The model doesn't "know" its representations are more orthogonal; it just *has* more orthogonal representations.

### Connection to prior results

The injection-point crossover is consistent with the [representational divergence analysis](REPRESENTATIONAL_DIVERGENCE_README.md), which found:

- Post_block0 (pre-injection): lowest effective rank (164.6/256), highest top-10 concentration (16.7%), divergence-to-self-knowledge alignment barely above random (1.10×). The early-layer change is **concentrated and orthogonal to self-knowledge**.
- Post_block3 (post-injection): near full-rank (eff_rank 219-227), top-5 divergence PCs capture 2.42× more residual variance than random. The late-layer change **is** the self-knowledge.

The swap test confirms this split: the late-layer self-knowledge is FM-specific (it IS the FM's residual), while the early-layer reorganization is general-purpose and shows a small positive signal for the shared FM component.

## Reproduction

```bash
# Full run (trains 9 fresh FMs + all probes including ensemble)
# Loads saved FM checkpoints if available, trains from scratch if not
modal run --detach a2a_forward/forward_model_swap.py::a2a_forward_model_swap
```
