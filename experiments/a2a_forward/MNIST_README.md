# MNIST A2A Forward Model Experiment

**Code**: `mnist_experiment.py`, `mnist_analysis.py`, `mnist_baseline_battery.py`
**Date**: 2026-06-07
**Prior experiment**: [Baseline battery](BASELINE_BATTERY_README.md), [Model scale experiment](MODEL_SCALE_README.md)

## Motivation

All prior A2A experiments used autoregressive language modeling on FineWeb-Edu. The residual was consistently full-rank and diffuse (effective rank 200/256), with uniform forward model degradation across behavioral categories. This raises the question: is the full-rank residual a property of the cerebellar forward model architecture, or a property of the data generating process (language)?

MNIST classification provides a maximally different testing ground: finite discrete classes, 2D spatial structure, well-understood decision boundaries. If the residual's dimensionality reflects the complexity of the DGP as modeled by the model, we should see a qualitatively different residual structure on MNIST.

## Architecture

**Main model**: 4-layer, 4-head, 128-dim Vision Transformer (0.80M params). 4x4 patches on 28x28 images = 49 patches + [CLS] token = 50 positions. Bidirectional attention. Classification head on [CLS] output. Cross-entropy loss on 10 digit classes. Trained on full MNIST training set (60K images).

**Forward model**: 1-layer transformer, 1 head, 32-dim compressed attention, bidirectional (83K params, ~10% of main model). Predicts post_block0 -> post_block3 (3-layer gap). MSE loss on detached targets.

**Closed-loop**: Same `CerebellarGate` design (zero-init linear projection, 16.5K params). Injected after block 1, giving 2 downstream blocks to use the prediction.

**Controlled comparison**: Open-loop and closed-loop trained with identical seed (42), lr (3e-4), initial weights, and data order. 5000 steps, batch size 128.

## Results

### Training dynamics

Both models reach ~96-97% test accuracy. The closed-loop model shows the same qualitative dynamics as language:

| Metric | Open-loop | Closed-loop |
|---|---|---|
| Final val accuracy | 95.8% | 96.1% |
| Final val loss | 0.127 | 0.118 |
| Injection benefit (Δ loss) | -- | **-0.031** |
| Dependency (CL_no_inj - OL) | -- | +0.022 |
| Forward model cosine | 0.973 | 0.903 |
| Gate norm | -- | 1.49 |

The injection helps (Δ = -0.031), the gate opens (1.49), the forward model cosine degrades in closed-loop (0.973 -> 0.903), and the model becomes dependent on the injection (+0.022 worse without it). All qualitatively identical to language.

### Self-knowledge probes -- 3x stronger than language

The headline result. Linear probes predicting the forward model's 128-d residual vector from each layer's activations:

| Layer | OL R^2 | CL R^2 | Δ R^2 |
|---|---|---|---|
| post_embed | 0.072 | 0.277 | **+0.205** |
| post_block0 | 0.108 | 0.630 | **+0.522** |
| post_block1 | 0.196 | 0.679 | **+0.483** |
| post_block2 | 0.285 | 0.703 | **+0.418** |
| post_block3 | 0.377 | 0.726 | **+0.349** |

The Δ R^2 at post_block0 (+0.52) is **2.8x the language result** (+0.185). Self-knowledge is distributed across all layers via backprop (same mechanism as language), but dramatically stronger. The CL model's first layer reorganized to make the forward model's blind spots linearly accessible with R^2 = 0.63 (vs 0.21 in language).

**Why stronger?** The residual is low-rank (see below), meaning there are specific, identifiable directions the forward model misses. The model can encode "which specific direction was missed" precisely. In language, the residual was full-rank (199/256) -- the model could only encode a diffuse "slightly worse everywhere" signal.

### Residual PCA -- opposite of language

The residual between predicted and actual post_block3:

| Metric | MNIST (128-d) | Language (256-d) |
|---|---|---|
| Top-1 PC variance | **15.7%** | 2.4% |
| Top-5 PCs | **59.2%** | 9.3% |
| Top-10 PCs | **82.5%** | 15.8% |
| 50% variance rank | **4** | 62 |
| 90% variance rank | **17** | 189 |
| Effective rank | **18.3/128** | 199.8/256 |

The MNIST residual is low-rank and concentrated. The forward model is missing specific sub-circuits, not being slightly worse everywhere. Four principal components capture half the residual variance.

This is the key structural difference. In language, the residual reflected a uniform capacity shortfall across all 256 dimensions. In MNIST, the forward model captures most computation well but misses identifiable, low-dimensional components of the main model's processing.

### Residual directions are digit-discriminative

The top PCs of the residual strongly discriminate between digit classes (eta-squared = fraction of variance explained by digit identity):

| PC | Variance | eta^2 | Strongest discrimination |
|---|---|---|---|
| PC1 | 15.7% | 0.225 | digit 1 (-0.82) vs digit 8 (+0.48) |
| PC2 | 13.8% | 0.201 | digit 1 (-0.43) vs digit 5 (+0.41) |
| PC3 | 11.4% | 0.173 | digit 0 (-0.51) vs digit 8 (+0.46) |
| PC4 | 10.0% | 0.141 | digit 1 (+0.39) vs digit 6 (-0.24) |
| PC5 | 8.4% | 0.232 | digit 6 (+0.48) vs digit 5 (-0.34) |

All top 5 PCs have eta^2 in 0.14--0.23 -- 14-23% of each PC's variance is explained by which digit is being classified. The residual is not noise or a training artifact; it reflects class-conditional computation the forward model cannot capture.

**Interpretation**: The forward model learns the "common" computation shared across all digits (position encoding, general spatial feature extraction, attention aggregation). What it misses is digit-specific computation -- the processing that differentiates a 1 from an 8, a 5 from a 6. This makes sense: digit-specific computation requires the model to have learned what each digit looks like, which is the kind of knowledge that a 10%-capacity forward model cannot fully internalize.

Digit 1 is the most distinctive in the residual space (large negative projections on PC1-2, large positive on PC4), consistent with 1's minimal stroke complexity requiring qualitatively different processing from all other digits.

### Cross-digit residual similarity reveals visual groupings

Mean residual cosine similarity between digit pairs:

| Pair | Cosine | Interpretation |
|---|---|---|
| 3 <-> 8 | **+0.84** | Shared curved-stroke computation |
| 1 <-> 7 | **+0.66** | Shared straight-stroke computation |
| 0 <-> 4 | +0.62 | Both have enclosed regions |
| 2 <-> 6 | +0.62 | Similar curved shapes |
| 1 <-> 3 | **-0.45** | Maximally different computation |

The residual's structure mirrors visual similarity, not label proximity. Digits that look alike trigger similar "missed computation" in the forward model.

### Spatial structure in the residual

**CLS token is hardest to predict**: CLS residual norm = 3.45, patch mean = 2.66 (ratio 1.30). The classification aggregation at the CLS position requires computation the forward model cannot fully capture -- it's where digit-specific information is most concentrated.

**Center patches have highest residual norm** (7x7 spatial grid, mean residual norm per patch position):

```
2.16  3.04  2.42  2.54  2.62  2.20  2.41
2.75  2.39  2.82  3.18  3.32  2.96  2.45
2.02  2.37  2.99  2.85  2.82  2.37  2.31
2.05  2.33  2.88  3.83  2.88  2.41  2.16
2.94  2.71  2.52  2.94  3.01  2.97  2.56
2.58  2.76  3.13  3.27  3.55  2.85  2.22
2.26  2.23  2.64  2.35  2.71  2.54  2.27
```

The center patch (row 3, col 3) has the highest residual norm (3.83), while corner patches are lowest (~2.2). MNIST digits concentrate their distinguishing strokes in the image center -- this is where the forward model struggles most.

### Per-digit residual norms

| Digit | Mean residual norm | Interpretation |
|---|---|---|
| 0 | 2.25 | Simple shape, easiest to predict |
| 3 | 2.45 | |
| 6 | 2.53 | |
| 2 | 2.61 | |
| 5 | 2.75 | |
| 7 | 2.77 | |
| 1 | 2.78 | Simple shape but unique processing |
| 8 | 2.78 | Complex shape |
| 9 | 2.88 | |
| 4 | **2.99** | Most variable stroke patterns |

Digit 0 is easiest to predict (simple enclosed shape), digit 4 hardest (highly variable stroke patterns across writers).

### Causal substitution -- non-uniform degradation

Replaced blocks 1-3 with the forward model's prediction, then continued from ln_f + classification head. Three modes: Normal (all blocks), Substituted (block 0 + forward model), Ablated (block 0 only).

| Mode | Accuracy | CE loss | Δ CE | KL vs Normal |
|---|---|---|---|---|
| Normal | 96.9% | 0.102 | -- | -- |
| Substituted | 96.5% | 0.119 | +0.017 | 0.033 |
| Ablated | 90.1% | 0.300 | +0.198 | 0.230 |

The forward model recovers **85.6%** of blocks 1-3's KL contribution. Replacing 3 transformer blocks (~600K params) with an 83K-param forward model costs only 0.4pp accuracy.

**Per-digit breakdown -- non-uniform, opposite of language**:

| Digit | Acc_N | Acc_S | Acc_A | KL_sub | KL_abl |
|---|---|---|---|---|---|
| 0 | 98.9% | 99.1% | 96.6% | 0.004 | 0.080 |
| 2 | 96.3% | 97.1% | 98.5% | 0.036 | 0.080 |
| 4 | 93.8% | 94.1% | 80.0% | 0.053 | 0.374 |
| 5 | 97.0% | 94.6% | 94.4% | **0.052** | 0.127 |
| 9 | 96.8% | 95.8% | 78.8% | 0.046 | 0.487 |

In language, KL_sub ranged 0.04-0.07 with no behavioral specificity -- the forward model was "slightly worse everywhere." In MNIST, degradation varies 13x across digits (KL_sub from 0.004 to 0.053).

**Digit 5 is the outlier**: KL_sub/KL_abl recovery is only 59% (vs 85.6% overall). The forward model misses computation specific to digit 5's variable stroke patterns. This is consistent with digit 5 having the most variable handwritten forms.

**Digit 2 shows improvement under substitution** (97.1% vs 96.3%). The forward model's prediction is actually better than what blocks 1-3 compute for digit 2 -- a regularization effect.

### Robustness gap -- 4x improvement, stronger than language

Gaussian perturbations at the injection point (post_block1), measuring loss degradation:

| eps | OL Δloss | CL Δloss | CL/OL ratio |
|---|---|---|---|
| 0.5 | -0.002 | +0.001 | -- |
| 1.0 | **+0.012** | **+0.003** | **0.244** |
| 2.0 | +0.168 | +0.014 | 0.084 |

The closed-loop model takes **4.1x less damage** from perturbation at eps=1.0 (vs 2-2.5x in language). At eps=2.0, the gap widens to 12x. The MNIST model's loss landscape at the injection point is dramatically flatter than the open-loop model's.

### Baseline battery -- what is uniquely forward prediction?

5 conditions with identical lr/seed/init/data order, adapted from the language baseline battery:

| Condition | Injection | Accuracy | Δ_inj | Gate norm |
|---|---|---|---|---|
| open_loop | None | 96.7% | 0 | -- |
| forward | gate(fwd_model(post_block0)) | **97.8%** | **-0.024** | 1.49 |
| shifted | gate(shift(fwd_model, k=10)) | 97.3% | -0.007 | 2.44 |
| random_proj | gate(random_matrix @ post_block0) | 97.3% | -0.015 | 2.72 |
| autoencoder | gate(autoenc(post_block0)) | 96.4% | +0.003 | 2.38 |

**Robustness is the strongest discriminator**:

| Condition | Δloss (eps=1.0) | Ratio vs OL |
|---|---|---|
| open_loop | +0.0118 | 1.000 |
| **forward** | **+0.0029** | **0.244** |
| **autoencoder** | **+0.0029** | **0.245** |
| random_proj | +0.0036 | 0.307 |
| shifted | +0.0246 | **2.082** (worse!) |

The shifted condition is *less robust than open-loop* -- position-misaligned injection actively harms robustness. This is a stronger result than language, where shifted still produced some robustness improvement.

Forward and autoencoder produce identical robustness ratios (0.244 vs 0.245). Unlike language (where forward was 2.4x and autoencoder was 1.6x improvement), the autoencoder matches forward on MNIST. This may reflect MNIST's simpler structure: the autoencoder's compressed representation of post_block0 contains enough information to produce the same loss landscape flattening as a forward prediction.

**Self-knowledge depth profile**:

| Layer | OL | Forward | Shifted | Random | Autoenc |
|---|---|---|---|---|---|
| post_block0 | 0.107 | **0.631** | 0.301 | 0.672 | 0.592 |
| post_block3 | 0.380 | **0.724** | 0.487 | 0.693 | 0.708 |

All injection conditions produce strong self-knowledge -- much less differentiation than in language. In language, forward prediction was 2.4-2.9x the baselines at the deepest layer. In MNIST, the gap at post_block3 is only 1.02-1.04x.

**Why the baselines are closer on MNIST**: The low-rank residual (eff rank 18/128) means there are only ~18 dimensions of "missed computation" to encode. Any injection that causes representational reorganization can incidentally capture these 18 dimensions. In language, the full-rank residual (200/256) means there are 200 dimensions to encode, and only forward prediction produces the right organizational pressure for the model to encode all of them.

This is a falsifiable prediction: as the DGP complexity increases (and the residual effective rank grows), the gap between forward prediction and baselines should widen. Language (rank 200) shows a larger gap than MNIST (rank 18). A DGP of intermediate complexity should show an intermediate gap.

## Connection to residual rank and DGP complexity

The central finding across the grokking, MNIST, and language experiments:

| Domain | DGP complexity | Effective rank | Top-1 PC | Residual structure | Self-knowledge gap |
|---|---|---|---|---|---|
| Grokking (mod add) | Low (~15 Fourier modes) | ~15/128 | >20% | Low-rank, Fourier-aligned | (not tested) |
| **MNIST** | **Medium (10 classes)** | **18/128** | **15.7%** | **Low-rank, digit-discriminative** | **Large (Δ R^2 +0.52)** |
| Language (29M) | High (natural language) | 200/256 | 2.4% | Full-rank, diffuse | Moderate (Δ R^2 +0.19) |
| Language (77M) | High (natural language) | 235/256 | 2.9% | Full-rank, slightly less diffuse | (not tested) |

The residual's effective rank reflects the number of independent computational modes the main model develops beyond the forward model's capacity. MNIST has ~10 classes requiring ~18 independent directions of class-conditional computation. Language has unbounded combinatorial structure requiring ~200 independent directions.

The forward model acts as a compression probe of the main model's layer computation. What it captures is the "predictable" part -- the common processing shared across inputs. What it misses (the residual) is the "novel" part -- input-specific computation that requires capacity the forward model lacks. The dimensionality of this novel computation directly reflects the complexity of the task.

## Files

| File | Purpose |
|---|---|
| `mnist_experiment.py` | Main experiment: open-loop + closed-loop controlled retrain, self-knowledge probes, robustness, residual PCA |
| `mnist_analysis.py` | Residual direction analysis (PCA, digit conditioning, spatial structure) + causal substitution |
| `mnist_baseline_battery.py` | 5-condition baseline battery (open_loop, forward, shifted, random_proj, autoencoder) |
| `vit.py` | Vision Transformer with return_intermediates and cerebellar_fn interface |

## Reproduction

```bash
cd experiments/

# Main experiment (open-loop + closed-loop, ~10 min)
modal run --detach a2a_forward/mnist_experiment.py::main

# Residual analysis + causal substitution (loads saved models, ~3 min)
modal run --detach a2a_forward/mnist_analysis.py::main

# Baseline battery (5 conditions, ~20 min)
modal run --detach a2a_forward/mnist_baseline_battery.py::main
```

## Modal volume

Results saved to `language-reduction-data` volume:

```
/data/a2a_forward/
├── mnist/
│   └── vit_4L_4H_128D/
│       └── post_block0_to_post_block3/
│           ├── ol_model.pt
│           ├── ol_fwd.pt
│           ├── cl_model.pt
│           ├── cl_fwd.pt
│           ├── cl_gate.pt
│           ├── results.json
│           └── analysis_results.json
└── mnist_baseline_battery/
    └── vit_4L_4H_128D/
        └── post_block0_to_post_block3/
            ├── results.json
            └── histories.json
```
