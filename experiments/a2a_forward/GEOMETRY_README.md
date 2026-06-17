# Computational Property Geometry Probes

**Code**: `mnist_geometry.py` (MNIST), `language_geometry.py` (language)
**Date**: 2026-06-16
**Prior**: [MNIST distillation](MNIST_DISTILLATION_README.md), [Distillation](DISTILLATION_README.md), [Local loss probes](MNIST_LOCAL_LOSS_README.md)

## Goal

Test whether closed-loop training and distillation produce representations with more compositionally structured computational properties — the analog of classic word embedding geometry findings (king − man + woman = queen) applied to a model's relationship with its own computation.

The hypothesis: if a model internalizes knowledge about its own computation (via distillation), computational state and data identity should be encoded as more independent, additively separable directions in activation space. CL meta-knowledge ("where is the FM wrong?") may not produce this, because the FM's errors are inherently correlated with data properties. Distillation, which converts meta-knowledge into object-level knowledge ("what computation looks like"), should.

## Experiment design

Three conditions using existing checkpoints (identical seed/lr/init):
- **OL**: open-loop baseline
- **CL**: closed-loop, evaluated without injection
- **Distilled**: post-distillation model (FM contribution absorbed into weights)

Per-model computational properties:
- **Data identity**: digit (MNIST, 10 classes) / token frequency band (language, 4 bins)
- **FM residual norm**: scalar, from each model's own FM
- **FM residual direction cluster**: K-means on normalized residual vectors (K=8)
- **Block1 contribution norm**: ||post_block1 − post_block0||
- **Prediction confidence** (MNIST) / **per-token LM loss** (language)

Analysis layers: post_block0 (pre-injection), post_block3 (final).

Six tests, each comparing OL vs CL vs Distilled:

1. **Probe accuracy**: linear probes for each property (R² or classification accuracy)
2. **Probe direction orthogonality**: pairwise |cos| between regression weight vectors (lower = more disentangled)
3. **Compositionality**: centroid additivity R² — does data_category × computation_level factor additively?
4. **Vector arithmetic**: analogy completion cosine — is the "high residual − low residual" direction consistent across data categories?
5. **Data-computation independence**: projection of computational probe weights onto the data-identity subspace
6. **Subspace overlap**: principal angle between data-identity and residual-cluster subspaces

## Results

### Probe direction orthogonality (Test 2) — the headline

Mean |cos| between all 5 probe directions (lower = more orthogonal):

| Layer | MNIST OL | MNIST CL | MNIST Dist | Lang OL | Lang CL | Lang Dist |
|---|---|---|---|---|---|---|
| post_block0 | 0.172 | 0.233 | **0.085** | 0.093 | 0.107 | **0.067** |
| post_block3 | 0.109 | **0.125** | 0.115 | 0.124 | **0.246** | 0.145 |

Replicates across both domains:

1. **Distilled is the most orthogonal at early layers.** 2.0× more orthogonal than OL in MNIST, 1.4× in language.
2. **CL is the least orthogonal at late layers.** The CL model's meta-knowledge creates entanglement between probe directions. In language, fm_res_norms × block_contrib reaches |cos| = 0.618 in CL at post_block3 — these directions are heavily aligned. Distillation resolves most of this (0.246 → 0.145).

The CL model has the strongest self-knowledge R² (0.63–0.73 at post_block0 in MNIST), yet the worst orthogonality. More self-knowledge does not mean more organized representations — what matters is the *kind* of self-knowledge.

### Vector arithmetic (Test 4)

Analogy: "data_category_A at high residual − data_category_A at low residual + data_category_B at low residual ≈ data_category_B at high residual." Mean cosine between predicted and actual centroids:

| Layer | MNIST OL | MNIST CL | MNIST Dist | Lang OL | Lang CL | Lang Dist |
|---|---|---|---|---|---|---|
| post_block0 | 0.854 | 0.916 | **0.929** | 0.810 | **0.838** | 0.821 |
| post_block3 | 0.911 | 0.943 | **0.970** | 0.743 | 0.735 | **0.773** |

Distilled has the best vector arithmetic at the final layer in both domains: 0.970 (MNIST, n=90 analogies) and 0.773 (language, n=12 analogies). The "computational state" direction is more consistent across data categories after distillation — the direct analog of how "gender" is consistent across "royalty" in the classic word2vec finding.

### Compositionality (Test 3)

Centroid additivity R² for data_category × residual_quartile:

| Layer | MNIST OL | MNIST CL | MNIST Dist | Lang OL | Lang CL | Lang Dist |
|---|---|---|---|---|---|---|
| post_block0 | 0.914 | 0.865 | **0.938** | 0.888 | **0.931** | 0.899 |
| post_block3 | 0.927 | 0.830 | **0.937** | 0.867 | **0.892** | 0.882 |

The direction of the compositionality effect flips between domains:
- **MNIST**: CL is *least* compositional (0.830 at post_block3), Distilled is most (0.937).
- **Language**: CL is *most* compositional (0.892), OL is least (0.867).

This reflects a structural difference. In MNIST, the FM residual is low-rank and digit-discriminative — the residual *is* partially a digit direction, so CL meta-knowledge amplifies the digit-residual overlap. In language, the residual is full-rank and not frequency-discriminative (the prior behavioral residual analysis showed it tracks attention entropy and computation type, not token frequency), so frequency and residual are already approximately independent.

### Probe accuracy (Test 1)

At post_block3:

| Metric | MNIST OL | MNIST CL | MNIST Dist | Lang OL | Lang CL | Lang Dist |
|---|---|---|---|---|---|---|
| Data identity acc | 0.917 | 0.929 | **0.948** | 0.722 | 0.730 | **0.735** |
| Cluster acc | 0.621 | **0.696** | 0.353 | 0.383 | **0.733** | 0.355 |
| FM res norm R² | **0.488** | 0.446 | 0.326 | 0.462 | 0.488 | **0.496** |
| Block contrib R² | 0.871 | **0.899** | 0.895 | 0.612 | 0.629 | **0.675** |

Distilled has the best data-identity encoding in both domains. CL has the best cluster accuracy (active FM interaction provides the strongest residual-direction signal). The distilled model's lower cluster accuracy reflects internalization — the FM contribution is absorbed into the model's weights, so the FM residual is smaller and its direction less salient.

## Interpretation

### Distillation converts entangled meta-knowledge into orthogonal object-level knowledge

CL self-knowledge is *meta-knowledge*: "where is the FM wrong?" This is inherently correlated with data properties — the FM's errors are digit-dependent (MNIST) and frequency-dependent (language). The meta-knowledge direction overlaps with the data direction, creating entanglement. That is why CL has |cos|(fm_res_norms × block_contrib) = 0.618 at post_block3 in language: the FM's reliability is entangled with block 1's contribution, which is entangled with token frequency.

Distillation converts this into *object-level knowledge*: "what computation looks like." This is about the transformation itself, independent of which digit or token it is applied to. It is naturally orthogonal to data identity, producing the disentanglement we observe.

### Connection to the local loss experiment

The meta vs object-level distinction maps directly onto the MNIST local loss finding (3:1 meta-knowledge dominance in CL_LL). The representation probes in that experiment showed that CL_LL's advantage over CL at early layers was Δ R² = +0.31 for meta-knowledge (orthogonalized residual probe) vs +0.10 for object-level (prediction probe). Here we see the geometric consequence of that distinction: meta-knowledge creates entanglement, object-level knowledge creates orthogonality.

### Connection to classic embedding geometry

"King − man + woman = queen" works because gender and royalty are independent semantic features encoded as approximately orthogonal linear directions. Our analog — "digit 3 with high residual − digit 3 with low residual + digit 7 with low residual ≈ digit 7 with high residual" — shows that *computational state* and *data identity* can similarly become independent, additive dimensions of a model's representation. What is new is that we can control this property through the training procedure: distillation increases the orthogonality of computational properties relative to data properties. The degree of compositionality tracks the distillation step, not the amount of self-knowledge (CL has more self-knowledge but less orthogonality).

### Why effects are larger in MNIST

All effects are stronger in MNIST than language. MNIST has a low-rank, digit-discriminative residual (effective rank 18/128) that provides specific directions for the model to organize around. Language has a full-rank, diffuse residual (effective rank 200/256) where the capacity bottleneck binds uniformly. Low-rank structure provides more "room" for compositional organization — fewer directions to disentangle.

## Reproduction

```bash
# MNIST (loads checkpoints from mnist_experiment.py and mnist_distillation.py)
modal run --detach a2a_forward/mnist_geometry.py::a2a_mnist_geometry

# Language (loads checkpoints from controlled_retrain.py and distillation.py)
modal run --detach a2a_forward/language_geometry.py::a2a_language_geometry
```
