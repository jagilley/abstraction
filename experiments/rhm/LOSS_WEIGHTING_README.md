# Loss Weighting Experiments (2026-06-22, updated 2026-06-23)

**Code**: `rhm_focal_loss.py`, `rhm_confidence_threshold.py`, `rhm_fm_weighted_ntp.py`
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

## Experiment 3: FM-surprise-weighted NTP (2026-06-23)

**Code**: `rhm_fm_weighted_ntp.py`

### Motivation

Focal loss and confidence thresholding work, but they're heuristics: the weighting is based on the model's *output confidence* (how certain the NTP prediction is), which is a proxy for what we actually care about — how computationally surprising a position is to the model's own self-model. The FM residual at each position is a direct measure of computational surprise: positions where the FM predicts the model's computation well are "boring," positions where it can't are "surprising."

Rather than bolting a heuristic weighting onto the loss, we can co-train an FM alongside the model and use its angular prediction error as the NTP weight:

```
surprise_t = 1 - cos(FM_predicted_t, actual_t)
weight_t = surprise_t / mean(surprise)
loss = mean(weight_t · -log p(y_t))
```

This is self-scheduling: early in training, the FM is random (cosine ~0, weights ~uniform → standard NTP). As the FM improves and captures easy positions first, weights shift to positions the FM can't predict. No heuristic γ or τ needed.

We use 1-cosine rather than residual norm as the surprise signal because it's scale-invariant — raw residual norm conflates prediction quality with activation magnitude (level-0 positions could have larger residuals simply because more computation happens there, not because the FM is worse).

### Design

Three conditions, all at L=6/m=4, 6L/6H/192D (~2.7M params), 20M tokens (matching Experiment 1):

| Condition | NTP weighting | FM during training |
|---|---|---|
| ce | uniform | none |
| focal_g2 | (1 - p_t)^2 | none |
| fm_weighted | (1 - cos_t) / mean | co-trained, 2L/d_head=24, post_block0 → post_block3 |

FM is trained on MSE with stop-gradient targets. FM gradient does not flow into the main model. Weights are detached — no gradient flows through the surprise signal into the FM.

### Per-level loss at convergence

| run | val | L0 | L1 | L2 | L3 | L4 | L5 |
|-----|------|------|------|------|------|------|------|
| ce | 1.364 | 0.889 | 1.726 | 1.898 | 1.938 | 1.936 | 1.924 |
| focal_g2 | 1.375 | 0.916 | 1.730 | 1.898 | 1.937 | 1.934 | 1.924 |
| fm_weighted | 1.374 | 0.902 | 1.738 | 1.905 | 1.939 | **1.933** | 1.924 |

| run | ΔL0 | ΔL1 | ΔL2 | ΔL3 | ΔL4 | ΔL5 |
|-----|------|------|------|------|------|------|
| focal_g2 | +0.027 | +0.004 | -0.000 | -0.001 | -0.002 | +0.000 |
| fm_weighted | +0.013 | +0.011 | +0.007 | +0.001 | **-0.004** | -0.000 |

The fm_weighted model achieves the single best L4 improvement (-0.004) of any condition, but at the cost of L1-L2 (+0.011, +0.007). The loss profile is different from focal: focal trades L0 cleanly for L3-L4, while fm_weighted trades L0-L2 for L4.

### Post-hoc FM residual properties

| run | cos | 1-cos | res_norm | rank% | top1% |
|-----|-------|-------|----------|-------|-------|
| ce | 0.925 | 7.5% | 14.3 | 68.6 | 20.6 |
| focal_g2 | 0.940 | 6.0% | 12.9 | 71.0 | 15.4 |
| fm_weighted | **0.966** | **3.4%** | **11.2** | 69.0 | 16.5 |

FM cosine 0.966 — the prediction gap drops 55% vs ce and 43% vs focal. The fm_weighted model's computation is dramatically more legible to a fresh post-hoc FM than either baseline.

### Three-phase self-scheduling trajectory

The FM-surprise weights evolve through three distinct phases:

| Step | FM cos | L0 wt | L1 wt | L2 wt | L3 wt | L4 wt | L5 wt | Phase |
|------|--------|-------|-------|-------|-------|-------|-------|-------|
| 0 | 0.170 | 1.00 | 1.00 | 1.01 | 1.00 | 0.99 | 0.97 | FM random → uniform |
| 200 | 0.995 | 1.36 | 0.67 | 0.64 | 0.62 | 0.59 | 0.53 | Phase 1: FM overshoot |
| 2000 | 0.996 | 1.66 | 0.37 | 0.35 | 0.34 | 0.27 | 0.16 | Phase 1: peak overshoot |
| 7000 | 0.978 | 1.24 | 0.90 | 0.61 | 0.63 | 0.66 | 0.68 | Phase 2: transition |
| 10000 | 0.974 | 0.96 | 1.12 | 0.96 | 0.94 | 0.98 | 0.91 | Phase 2: ~uniform |
| 17000 | 0.973 | 0.97 | 0.90 | 1.17 | 1.13 | 1.16 | 1.16 | Phase 3: correct direction |
| 20000 | 0.968 | 0.92 | 0.90 | 1.27 | 1.26 | 1.31 | 1.36 | Phase 3: focal-like |

**Phase 1 (steps 0-2000)**: The FM converges extremely fast (cos 0.996 by step 2000) — it can predict nearly everything the model computes, because the model itself hasn't developed complex computation yet. The tiny residual is largest where the model is most active (level 0), so weights favor L0. This is the *opposite* of focal loss.

**Phase 2 (steps 4000-10000)**: The model develops deeper computation that outgrows the FM (cos drops 0.996 → 0.974, MSE grows 0.06 → 0.36). The per-level weights equalize as the FM starts struggling at all levels.

**Phase 3 (steps 14000-20000)**: The FM falls further behind (MSE reaches 0.72), and it falls behind *more* at higher levels (where the model's computation is genuinely complex) than at level 0 (where computation is well-structured). Weights flip to the correct direction: L0=0.92, L2-L5=1.27-1.36. **Focal-loss-like behavior emerges autonomously from the FM dynamics.**

The L1-L2 cost in the per-level loss comes from Phase 1: the model over-invests in L0 during the first ~7000 steps when the FM-surprise signal points in the wrong direction.

## Experiment 4: FM-surprise NTP + uniform local loss (2026-06-23)

**Code**: `rhm_fm_weighted_ntp.py` (loss_type="fm_weighted_ll")

### Motivation

The Phase 1 problem in Experiment 3 arises because the FM converges too fast — it captures everything before the model has developed enough hierarchical structure for the per-level surprise differences to point in the right direction. The local loss from the A2A experiments (`λ · MSE(actual_target, sg(FM_pred))`, gradient through main model only) pushes intermediate activations toward FM-predictability. This should:

1. Keep the FM accurate throughout training (preventing it from falling behind)
2. Compress the Phase 1 → Phase 3 transition (the FM's surprise signal becomes informative earlier)
3. Complement the NTP weighting (local loss handles *how* to compute; NTP weighting handles *where* to learn)

### Design

Same as Experiment 3 but with an additional local loss term:

```
total_loss = weighted_NTP_loss + λ · MSE(post_block3, sg(FM(post_block0)))
```

where `sg` is stop-gradient (gradient flows through `post_block3` into the main model's blocks 1-3, but not through the FM's prediction). λ = 0.1. The local loss provides gradient at intermediate layers, supplementing the NTP gradient that only enters through the output.

### Per-level loss at convergence — all four conditions

| run | val | L0 | L1 | L2 | L3 | L4 | L5 |
|-----|------|------|------|------|------|------|------|
| ce | 1.364 | 0.889 | 1.726 | 1.898 | 1.938 | 1.936 | 1.924 |
| focal_g2 | 1.375 | 0.916 | 1.730 | 1.898 | 1.937 | 1.934 | 1.924 |
| fm_weighted | 1.374 | 0.902 | 1.738 | 1.905 | 1.939 | 1.933 | 1.924 |
| **fm_wt_ll** | **1.367** | **0.895** | **1.731** | **1.898** | **1.937** | **1.934** | **1.925** |

| run | Δval | ΔL0 | ΔL1 | ΔL2 | ΔL3 | ΔL4 | ΔL5 |
|-----|------|------|------|------|------|------|------|
| focal_g2 | +0.011 | +0.027 | +0.004 | -0.000 | -0.001 | -0.002 | +0.000 |
| fm_weighted | +0.010 | +0.013 | +0.011 | +0.007 | +0.001 | -0.004 | -0.000 |
| **fm_wt_ll** | **+0.002** | **+0.005** | **+0.005** | **-0.001** | **-0.001** | **-0.002** | **+0.001** |

**Val loss cost is 5× smaller** than focal or fm_weighted (+0.002 vs +0.011/+0.010). L2 *improves* (-0.001) — the first L2 improvement across all objective modification experiments (label smoothing, focal loss, confidence threshold, fm_weighted). L3 and L4 also improve. The L1-L2 cost from fm_weighted alone is eliminated: the local loss fixed the Phase 1 problem.

### FM legibility

| run | cos | 1-cos | res_norm |
|-----|-------|-------|----------|
| ce | 0.925 | 7.5% | 14.3 |
| focal_g2 | 0.940 | 6.0% | 12.9 |
| fm_weighted | 0.966 | 3.4% | 11.2 |
| **fm_wt_ll** | **0.995** | **0.5%** | **1.3** |

FM cosine 0.995 — a 15× reduction in prediction gap vs ce, 12× vs focal. Residual norm drops an order of magnitude (14.3 → 1.3). The model's computation is nearly perfectly predictable by a compressed self-model.

**Caveat on residual rank**: the fm_wt_ll residual has higher effective rank (80.7%) than the other conditions (68-71%). This does NOT indicate more structured computation — it likely indicates that the tiny residual (norm 1.3) is dominated by architectural mismatch noise (1-head FM vs 6-head model). At this norm scale, the residual is a qualitatively different object than the ce/focal residual. The rank comparison is not controlled for the OOM norm difference.

### Self-scheduling trajectory — Phase 1 compressed to ~2000 steps

| Step | FM cos | L0 wt | L2 wt | L5 wt | Phase |
|------|--------|-------|-------|-------|-------|
| 0 | 0.170 | 1.00 | 1.01 | 0.97 | FM random → uniform |
| 200 | 0.994 | 1.44 | 0.55 | 0.41 | Phase 1: FM overshoot |
| 1000 | 0.994 | 1.42 | 0.62 | 0.29 | Phase 1: peak |
| **2000** | **0.992** | **0.90** | **1.13** | **1.02** | **Flip (vs step 14000 without LL)** |
| 4000 | 0.976 | 0.90 | 1.16 | 1.48 | Phase 3: stable correct direction |
| 10000 | 0.987 | 0.88 | 1.27 | 1.34 | Phase 3: strengthening |
| 20000 | 0.993 | 0.69 | 1.84 | 1.58 | Phase 3: strong reallocation |

The transition from Phase 1 (wrong direction) to Phase 3 (correct direction) happens at step **~2000** with local loss vs step **~14000** without. By step 20000, the reallocation is more extreme than fm_weighted alone: L0 gets weight 0.69 (vs 0.92), L2 gets 1.84 (vs 1.27).

The mechanism: the local loss keeps FM MSE at 0.01-0.02 throughout (vs 0.06 → 0.72 in fm_weighted alone). The FM never falls behind the model, so its surprise signal reflects genuine per-level computational differences rather than overall FM degradation.

### FM MSE trajectory — the key difference

| Metric | fm_weighted | fm_wt_ll |
|--------|-------------|----------|
| FM MSE at step 2000 | 0.061 | 0.011 |
| FM MSE at step 10000 | 0.359 | 0.019 |
| FM MSE at step 20000 | 0.718 | 0.014 |
| FM cos at step 20000 | 0.968 | 0.993 |

Without local loss, the FM falls progressively behind (MSE grows 12× over training). With local loss, the main model's computation stays FM-predictable throughout (MSE stable at ~0.01-0.02). This is because the local loss gradient through post_block3 pushes the model toward computation the FM can represent.

## Key findings across all loss weighting experiments

### The progression of ideas

| Experiment | Mechanism | Val cost | L2+ improvement | FM cos | Key limitation |
|---|---|---|---|---|---|
| Label smoothing | Soften target distribution | Worse at every level | None | Flat (0.92) | Wrong axis |
| Focal loss γ=2 | Downweight confident positions | +0.011 | L3-L4 slight | 0.940 | Heuristic, fixed |
| Threshold τ=0.3 | Zero gradient above threshold | +0.158 | L2-L3 retained | 0.955 | Heuristic, aggressive |
| FM-surprise NTP | Co-trained FM surprise signal | +0.010 | L4 only, L1-L2 cost | 0.966 | Phase 1 overshoot |
| **FM-surprise + LL** | FM surprise + local loss | **+0.002** | **L2-L4, no cost** | **0.995** | — |

### The three mechanisms are complementary

The fm_wt_ll architecture uses the FM residual in three ways:

1. **NTP weighting** (per-position): allocates gradient to positions where the FM's computation prediction is worst. Controls *where* to learn.
2. **Local loss** (per-dimension, uniform): pushes intermediate activations toward FM-predictability. Controls *how* to compute.
3. **The FM itself**: defines "boring" via its learned compression of the model's computation. Self-scheduling — no heuristic parameters.

The local loss keeps the FM accurate (mechanism 3 stays informative), which makes the NTP weighting (mechanism 1) point in the right direction. The NTP weighting allocates gradient to hard positions, and the local loss ensures the easy positions' computation is clean. Each mechanism addresses a different axis (position vs dimension, output vs intermediate), and together they produce near-zero task cost with near-perfect FM legibility.

### Residual rank caveat

The focal loss experiments (Experiment 2, 6.3M model) showed that gradient reallocation reduces residual rank, consistent with "m-sharpening is high-rank, L-learning is low-rank." The fm_wt_ll experiment shows the opposite trend (rank increases to 80.7%). This apparent contradiction is resolved by the OOM norm difference: the fm_wt_ll residual (norm 1.3) is dominated by architectural mismatch noise (high-dimensional by nature), while the focal/ce residual (norm 12-14) reflects genuine computational gap. The rank comparison is not meaningful across such different residual scales.

## Open questions

1. **A2A closed-loop test on RHM**: The fm_wt_ll model at cos 0.995 is the most FM-legible model we've produced. Does injecting the FM prediction back into the residual stream produce cleaner gate selectivity and meta-learning dynamics? There's a risk that cos 0.995 is *too* legible (the FM captures everything, injection is redundant — the 10% FM problem from the extended ratchet). The closed-loop test would determine whether this FM-legibility level is optimal or overshooting.

2. **Scale dependence**: These experiments are at 2.7M params where the model is capacity-saturated at m=4. At larger scales with genuine compositional headroom, the FM-surprise NTP weighting might produce larger L2+ improvements because there's actually capacity to redistribute.

3. **Language transfer**: Would FM-surprise NTP + local loss work during language model pretraining? Language has full-rank residuals (200/256 dimensions) and much richer computational structure. The FM might not converge as quickly (preventing the Phase 1 overshoot) but the local loss would still help.

4. **Local loss magnitude**: We used λ=0.1 without tuning. The local loss contribution was ~0.001-0.002 at convergence (0.1% of NTP loss) — effectively negligible in magnitude but apparently load-bearing in effect. A sweep over λ might find a better operating point.

## Reproduction

```bash
cd experiments/

# Focal loss sweep (2.7M model, T4)
modal run --detach -m rhm.rhm_focal_loss::focal_loss_sweep

# Confidence threshold sweep (6.3M model, L4)
modal run --detach -m rhm.rhm_confidence_threshold::confidence_threshold_sweep

# FM-surprise NTP sweep: ce vs focal_g2 vs fm_weighted (2.7M model, T4)
modal run --detach -m rhm.rhm_fm_weighted_ntp::fm_weighted_ntp_sweep

# FM-surprise NTP + local loss (2.7M model, T4)
modal run --detach -m rhm.rhm_fm_weighted_ntp::fm_weighted_ll_run
```

Results saved to `rhm-scaling-data` volume at `/data/rhm_focal_loss/`, `/data/rhm_confidence_threshold/`, and `/data/rhm_fm_weighted_ntp/`.
