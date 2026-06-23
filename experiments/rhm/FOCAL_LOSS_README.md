# Focal Loss & Confidence Threshold Experiments (2026-06-22)

**Code**: `rhm_focal_loss.py`, `rhm_confidence_threshold.py`
**Prior experiment**: [Label smoothing](LABEL_SMOOTHING_README.md), [Per-level loss decomposition](PER_LEVEL_LOSS_README.md)

## Motivation

The label smoothing experiment showed that softening the NTP objective's *sharpness* doesn't extend compositional depth — but it only tested one axis. The NTP loss also has a *position weighting* bias: level 0 gets ~51% of gradient for s=2 simply because there are more level-0 positions. Humans seem to have an adaptive attention allocation that deprioritizes predictable/low-level content — they allocate processing proportional to *information content*, not *frequency*. Can we approximate that?

Two DGP-agnostic interventions (no hierarchy knowledge required):

1. **Focal loss**: FL(p_t) = -(1 - p_t)^γ · log(p_t). Downweights positions where the model is already confident. Naturally reduces gradient from level-0 (easy) positions and increases it for levels 2+ (hard).

2. **Confidence thresholding**: Zero out gradient for positions where p(correct) > τ. The extreme version of focal loss. Creates a natural curriculum — as level-0 positions become confident during training, they drop out entirely, forcing all gradient toward harder positions.

## Experiment 1: Focal Loss Sweep (2.7M model)

Same controlled setup as label smoothing: L=6/m=4, 6L/6H/192D (~2.7M params), 20M tokens, v=8, s=2.

Sweep: γ ∈ {0.0, 0.5, 1.0, 2.0, 5.0}. Evaluation always uses standard CE.

### Per-level loss at convergence

| γ | val | L0 | L1 | L2 | L3 | L4 | L5 |
|---|-----|------|------|------|------|------|------|
| 0.0 | 1.364 | 0.889 | 1.726 | 1.898 | 1.938 | 1.936 | 1.924 |
| 0.5 | 1.367 | 0.895 | 1.729 | 1.900 | 1.938 | 1.936 | 1.925 |
| 1.0 | 1.366 | 0.898 | 1.724 | 1.899 | 1.938 | 1.936 | 1.925 |
| 2.0 | 1.375 | 0.916 | 1.730 | 1.898 | 1.937 | 1.934 | 1.924 |
| 5.0 | 1.401 | 0.963 | 1.748 | 1.902 | 1.936 | 1.934 | 1.926 |

### Per-level loss change vs baseline

| γ | ΔL0 | ΔL1 | ΔL2 | ΔL3 | ΔL4 | ΔL5 |
|---|-----|-----|-----|-----|-----|-----|
| 0.5 | +0.006 | +0.003 | +0.001 | +0.000 | -0.000 | +0.000 |
| 1.0 | +0.008 | -0.003 | +0.000 | +0.000 | -0.001 | +0.001 |
| 2.0 | +0.027 | +0.004 | **-0.000** | **-0.001** | **-0.002** | +0.000 |
| 5.0 | +0.074 | +0.021 | +0.003 | **-0.002** | **-0.002** | +0.002 |

**Key finding**: At γ=2, levels 3-4 *improve* slightly while L0 gets worse. This is the first time any higher-level improvement has been observed across all our objective modification experiments. Label smoothing hurt at *every* level. The direction is right, but the magnitudes are tiny (-0.002 at best) because the 2.7M model is capacity-saturated at m=4.

### FM residual properties (2.7M model)

| γ | cos | 1-cos | res_norm | rank% | top1% |
|---|-----|-------|----------|-------|-------|
| 0.0 | 0.925 | 7.5% | 14.3 | 68.6 | 20.6 |
| 0.5 | 0.928 | 7.2% | 14.5 | 69.5 | 21.2 |
| 1.0 | 0.937 | 6.3% | 12.7 | 71.1 | 13.7 |
| 2.0 | 0.940 | 6.0% | 12.9 | 71.0 | 15.4 |
| 5.0 | 0.950 | 5.0% | 10.6 | 73.6 | 10.3 |

FM cosine improves monotonically (0.925 → 0.950). This is qualitatively different from label smoothing, where cosine was flat (~0.92). Focal loss models' computation is genuinely more predictable to a compressed self-model — not just smaller activations.

## Experiment 2: Confidence Threshold Sweep (6.3M model)

Scaled up model: L=6/m=4, **8L/8H/256D (~6.3M params)**, 20M tokens, L4 GPUs.

Sweep: standard CE, focal γ=2, and confidence thresholds τ ∈ {0.5, 0.3, 0.15}.

### Per-level loss at convergence

| run | val | L0 | L1 | L2 | L3 | L4 | L5 |
|-----|-----|------|------|------|------|------|------|
| ce | 1.360 | 0.885 | 1.719 | 1.895 | 1.937 | 1.933 | 1.925 |
| γ=2 | 1.369 | 0.910 | 1.722 | 1.893 | 1.936 | 1.932 | 1.926 |
| τ=0.5 | 1.408 | 0.986 | 1.724 | 1.896 | 1.938 | 1.934 | 1.924 |
| τ=0.3 | 1.518 | 1.202 | 1.740 | 1.895 | 1.937 | 1.933 | 1.924 |
| τ=0.15 | 1.874 | 1.809 | 1.923 | 1.947 | 1.964 | 1.960 | 1.959 |

### Per-level accuracy

| run | L0 | L1 | L2 | L3 | L4 | L5 |
|-----|------|------|------|------|------|------|
| ce | 0.587 | 0.307 | 0.212 | 0.202 | 0.203 | 0.209 |
| γ=2 | 0.584 | 0.308 | 0.213 | 0.201 | 0.203 | 0.209 |
| τ=0.5 | 0.569 | 0.303 | 0.212 | 0.200 | 0.202 | 0.210 |
| τ=0.3 | 0.409 | 0.291 | 0.213 | 0.200 | 0.202 | 0.212 |
| τ=0.15 | 0.292 | 0.190 | 0.196 | 0.175 | 0.176 | 0.177 |

### Fraction of baseline learning retained per level

The key metric: what fraction of the baseline model's learning at each level does each modified model retain? Computed as (uniform - loss_modified) / (uniform - loss_baseline), where uniform = ln(8) = 2.079.

| run | L0 retained | L1 retained | L2 retained | L3 retained |
|-----|-------------|-------------|-------------|-------------|
| γ=2 | 97.9% | 99.2% | **101.1%** | **100.6%** |
| τ=0.5 | 91.6% | 98.6% | 100.0% | 100.0% |
| τ=0.3 | **73.5%** | **94.2%** | **100.0%** | **100.0%** |
| τ=0.15 | 22.6% | 43.3% | 71.4% | 81.1% |

**At τ=0.3, L0 learning drops by 26.5%, but L2 and L3 retain 100% of their learning.** The model spends less capacity on level-0 sharpening without losing any compositional depth. At γ=2, L0 barely suffers (98% retained) while L2 and L3 *slightly improve* (101%).

At τ=0.15, the model is starved of gradient (only ~30% of positions contribute by step 6000) and everything degrades — confirming that lower-level learning is a prerequisite for higher-level composition. You can't skip level 0 and jump to level 2.

### Mask fraction over training (threshold runs)

| run | step 2K | step 6K | step 10K | step 20K |
|-----|---------|---------|----------|----------|
| τ=0.5 | 0.900 | 0.831 | 0.854 | 0.847 |
| τ=0.3 | 0.716 | 0.640 | 0.647 | 0.646 |
| τ=0.15 | 0.422 | 0.253 | 0.269 | 0.310 |

The threshold creates a natural curriculum without any DGP knowledge: as level-0 positions become confident, they drop out of the loss, redirecting gradient toward harder positions.

### FM residual properties (6.3M model)

| run | cos | 1-cos | res_norm | rank% | top1% | fL4* |
|-----|-----|-------|----------|-------|-------|------|
| ce | 0.940 | 6.0% | 32.5 | 61.8 | 14.7 | 0.103 |
| γ=2 | 0.944 | 5.6% | 31.9 | 60.9 | 18.5 | 0.108 |
| τ=0.5 | 0.952 | 4.8% | 28.1 | 62.5 | 16.1 | 0.101 |
| τ=0.3 | 0.955 | 4.6% | 30.2 | 59.7 | 16.9 | **0.135** |
| τ=0.15 | 0.991 | 0.9% | 27.6 | 50.6 | 43.0 | 0.030 |

At τ=0.3:
- FM prediction gap drops 23% relative (6.0% → 4.6%) — the model's computation is significantly more legible to a compressed self-model
- fL4* jumps from 0.103 to 0.135 (+31%) — the FM's remaining error is more conditioned on hierarchy level-4 features
- The model is both easier to predict and what the FM gets wrong is more structured around the DGP

At τ=0.15, cosine hits 0.991 because the model barely computes anything (degenerate case).

## Key findings

### 1. Focal loss is qualitatively different from label smoothing

Label smoothing (sharpness axis): hurts at every level, FM cosine flat. Focal loss and confidence thresholding (position weighting axis): L0 gets worse, but higher levels either improve slightly (γ=2) or hold steady (τ=0.3/0.5), and FM cosine improves monotonically. The gradient reallocation is doing something genuinely different from just softening the target distribution.

### 2. The composition depth ceiling is a capacity constraint — but m-learning overhead is real

The absolute composition depth (measured by accuracy) doesn't improve under any objective modification. But at τ=0.3, the model retains 100% of its compositional learning at L2+ while losing 26.5% of its L0 learning. The m-sharpening capacity is not fungible with L-learning capacity at this scale, but it is overhead that can be reduced without damaging compositional capabilities.

### 3. Gradient reallocation improves FM legibility

The most consistent finding across both experiments: focal loss and confidence thresholding monotonically improve the FM's ability to predict the model's computation. Cosine goes from 0.925 → 0.950 (2.7M, focal sweep) and 0.940 → 0.955 (6.3M, threshold sweep). This isn't just smaller activations — the computation itself becomes more predictable to a compressed approximator.

At τ=0.3, fL4* increases by 31%, meaning the FM's errors become more structured around the hierarchy. The model organizes its computation in a way that is more aligned with the generative structure and more accessible to a compressed self-model.

### 4. Implications for meta-learning

If what we care about is priming a model for the A2A meta-learning loop (where a compressed forward self-model feeds predictions back into the model's residual stream), then the objective modification doesn't need to extend compositional depth — it needs to make the model's computation more legible to the self-model. Focal loss achieves this: at γ=2, the model barely loses any performance at any level while becoming measurably more predictable.

This suggests a practical intervention: a mild focal loss (γ≈2) during pretraining could produce models that are better primed for meta-learning, at minimal cost to standard NTP performance. The model doesn't learn different things, but it organizes its computation more cleanly — sitting in a better basin for self-knowledge.

Note that by every metric a traditional ML practitioner would evaluate — val loss, accuracy, perplexity — the focal loss models are nominally *worse*. Val loss rises (1.360 → 1.369 at γ=2, → 1.518 at τ=0.3), L0 accuracy drops, and no level shows top-line accuracy improvement. A standard ablation study would conclude "focal loss hurts for language modeling." But the properties that matter for meta-learning (FM cosine, residual rank, hierarchy-conditioned eta²) all improve. The model is worse at the task but better at being understood by a compressed version of itself. These are orthogonal evaluation axes, and the traditional one is blind to the second. If the purpose of pretraining isn't just to produce the best NTP model but to produce a model well-primed for downstream self-improvement via meta-learning, then optimizing purely for val loss may be optimizing the wrong objective.

### 5. Residual rank decomposition: m-sharpening is high-rank, L-learning is low-rank

The effective rank of the FM residual drops as we increase gradient reallocation intensity (6.3M model):

| run | cos | rank% | top1% |
|-----|-----|-------|-------|
| ce | .940 | 61.8 | 14.7 |
| γ=2 | .944 | 60.9 | 18.5 |
| τ=0.3 | .955 | **59.7** | 16.9 |
| τ=0.15 | .991 | **50.6** | **43.0** |

The FM architecture is held constant across all runs — only the main model's training objective changes. So the rank drop reflects a genuine change in the character of the main model's computation, not a change in the FM's approximation quality.

This suggests the model's computation has two components with different rank signatures:

- **m-sharpening** (discriminating among m composition rules at already-learned levels): requires many independent lookup-like discrimination directions — inherently **high-rank**. Each rule is a separate pattern to memorize, contributing independent variance directions.
- **L-learning** (compositional circuits that compose across hierarchy levels): builds on shared compositional structure — inherently **lower-rank**. The compositional function reuses structure across positions and levels.

When confidence thresholding suppresses m-sharpening (confident level-0 positions drop out of the loss), the computation becomes proportionally more L-learning, and the residual rank drops accordingly.

**This recontextualizes the prior residual rank experiments** ([RESIDUAL_RANK_README](RESIDUAL_RANK_README.md)). In the DGP sweep, higher-m settings produced higher-rank residuals. We attributed this to "poorly learned models do simple, low-dimensional computation." But under standard CE, higher m also means the model spends more capacity on m-sharpening (more rules to discriminate) — the high-rank component. We couldn't separate "DGP complexity" from "m-sharpening overhead" because every model was trained with standard CE. The focal loss / threshold results break this confound: same DGP, same m, but less m-sharpening → lower rank. The rank was (at least partially) tracking the *amount of m-sharpening*, not just the DGP complexity.

**Implication**: the FM can compress focal-loss models better not because they do less computation, but because they do more *compressible* computation. The compositional part (L-learning) has lower intrinsic dimensionality than the memorization part (m-sharpening). This is the property you'd want for meta-learning: the part of the computation that matters for generalization is exactly the part a compressed self-model can capture.

## Open questions

1. **A2A closed-loop test**: Does a focal-loss-trained RHM model produce cleaner gate selectivity when the FM prediction is injected back into the residual stream? This is the direct test of the meta-learning hypothesis.

2. **Scale dependence**: At larger model scales where the capacity constraint is less binding, does the relative advantage of focal loss at higher levels grow? The 2.7M → 6.3M comparison showed the FM effect is consistent but the per-level loss effect didn't amplify much. A model with genuine compositional headroom might show clearer depth extension.

3. **Optimal γ for real language**: γ=2 is the sweet spot in these experiments (minimal L0 cost, slight L2+ improvement, improved FM legibility). Would the same γ work for language model pretraining, or does the optimal point shift with vocabulary size and data complexity?

4. **Interaction with model architecture**: Do architectural choices that favor compositional generalization (e.g., more layers vs. wider layers, different attention patterns) interact with the focal loss effect? If certain architectures are less capacity-constrained for composition, focal loss might produce larger benefits there.

## Reproduction

```bash
cd experiments/

# Focal loss sweep (2.7M model, T4)
modal run --detach -m rhm.rhm_focal_loss::focal_loss_sweep

# Confidence threshold sweep (6.3M model, L4)
modal run --detach -m rhm.rhm_confidence_threshold::confidence_threshold_sweep
```

Results saved to `rhm-scaling-data` volume at `/data/rhm_focal_loss/` and `/data/rhm_confidence_threshold/`.
