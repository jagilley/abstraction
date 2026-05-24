# Meta-Neurons Are a Weak Window onto the GLP

*April 2026*

## The claim

Meta-neurons — individual SwiGLU gate activations inside the GLP denoiser — should be demoted from "the GLP's learned features" to "one probe among several, reliable for crisp binary concepts and degrading for everything else." Treating them as the primary window onto the GLP's learned content systematically underestimates what the GLP knows.

## Why

### 1. The 1-D framework bakes in monosemanticity

Single-neuron probing picks the best scalar feature per concept. It works when the concept happens to concentrate on one SwiGLU gate; it fails when the concept distributes across many.

| Task type | Mean 1-D AUC |
|---|---|
| GLP paper's 113 binary tasks (baseball, contradiction, etc.) | 0.84 |
| Emotion injection (15 core emotions) | 0.72 |

In emotion injection, joy and love collapsed onto the same meta-neuron (the 1-D framework can't separate nearby valence); gratitude and pride were too distributed to serve as reliable single-neuron detectors; only 4/15 emotions survived the Phase 2 reliability bar. The paper's 0.84 is a ceiling on tasks that *could* be binary-coded, not a floor on what the GLP represents. Anything graded, multi-dimensional, or context-dependent degrades from there.

### 2. Meta-neurons are inside the denoising circuit, not the thing the GLP learned

The GLP's learned object is the velocity field `v(h)` — a smooth map over activation space, curvature included. Meta-neurons are features of the circuit that *computes* `v`, not atomic units of what the GLP knows. The error signal diagnostic made the gap concrete:

| Readout of δ (sentiment error signal) | AUC |
|---|---|
| cos(δ, residual) | ≈ 0 |
| δ / h subspace overlap | ~10% |
| **Velocity response** | **1.0** |

Every linear readout of the circuit — which includes any meta-neuron-based decomposition — finds nothing. The information lives in the curvature of `v`, not in its internal linear features at any one layer. A better denoising circuit could in principle expose more via meta-neurons, but by construction they will always be a lossy projection of the learned geometry.

### 3. Manifold locality compounds the problem

A meta-neuron for concept X assumes X has a consistent signature across contexts. Residual transfer (9.5% on 10-way, ≈ chance) shows directional signatures mostly don't transfer. The meta-neuron that fires for emotion X on GoEmotions training text is not guaranteed to fire for the same internal state during on-policy generation elsewhere — the universal-detector assumption is exactly the move that manifold locality / FER says breaks down.

## What should be the primary window instead

Not one object — a stack, matched to the question:

- **"What is this representation doing?"** → the residual `R(h) = E_noise[v(noised(h)) − (noise − h)]`, taken whole. Already a full nonlinear computation; don't decompose into meta-neurons.
- **"Does direction δ matter here?"** → the velocity response `R(h+αδ) − R(h)`. Primary geometric primitive for counterfactual queries. Recovered sentiment at AUC 1.0 when all linear readouts failed.
- **"What structure exists across many representations?"** → between-category centroid SVD on residuals. Coarse but actually transfers (33.1%, p < 10⁻⁷), where naive PCA is at chance.
- **External coordinate system** → verbalization. Language abstracts over fractured geometry and is the natural substrate for cross-context comparison.

## Natural next step: velocity-response libraries

The underexplored object that most directly answers "there should be more structure than meta-neurons expose." Build a library of `(h, δ, v(h+αδ) − v(h))` triples across many base points and many probe directions. Look for:

- Directions that elicit geometrically similar responses across many h → nonlinear analogue of a global feature.
- Directions that elicit similar responses within clusters of h → FER fragments made visible.

This unifies the existing tools: the residual is the zeroth-order term in h, the velocity response is the first-order term along one δ, and the structure of the response map over δ is the next level up. It gets at discrete structure without assuming monosemantic or universal-direction encoding — the two assumptions meta-neurons quietly import.

## Connection to other beliefs

- **Nonlinear legibility**: linear tools underestimate the GLP. Meta-neurons are the GLP's own linear features — they inherit the limitation when used as a primary readout.
- **Manifold locality**: structure is rich locally, coarse globally. Meta-neurons assume universal signatures, which is precisely the move manifold locality says fails.

Together, the three beliefs cohere: the GLP is a nonlinear, locally-structured object, and its own internal linear features are neither nonlinear enough nor local enough to serve as the primary interpretive lens.

## Summary of evidence

| Source | Finding | Implication |
|---|---|---|
| GLP paper (113 binary tasks) | 0.84 mean 1-D AUC | Meta-neurons work for crisp binary concepts |
| Emotion injection | 0.72 mean AUC; joy/love collapse; 4/15 reliable | Degrade for graded or nearby concepts |
| Error signal diagnostic | Linear readouts ≈ 0; velocity field = 1.0 AUC | Full geometry carries info meta-neurons miss |
| Residual transfer | 9.5% vs 10% chance (10-way) | Universal-detector assumption fails across contexts |
| Diverse transfer test | Centroid SVD 33% vs PCA 21% (chance) | Between-category aggregation > per-neuron probing |
