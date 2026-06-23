# Label Smoothing Experiment (2026-06-22)

**Code**: `rhm_label_smoothing.py`
**Prior experiment**: [Per-level loss decomposition](PER_LEVEL_LOSS_README.md), [Regime trajectory](REGIME_TRANSITION_README.md)

## Motivation

The NTP objective's loss decomposes by hierarchy level with geometric weighting:

```
L_NTP = Σ_k  w_k · L_k     where w_k ∝ s^{-k}
```

For s=2, L=6: level 0 gets ~51% of the gradient, level 1 gets ~25%, levels 2-5 share the remaining ~24%. This, combined with cross-entropy's reward for sharpening (the last bits of accuracy at a learned level come from m-discrimination, not compositional depth), creates a potential bias: the model may allocate capacity to m-sharpening at level 0 rather than L-learning at levels 2+.

Label smoothing caps the reward for sharpening by replacing the one-hot target with a mixture: `q_i = (1-ε)·1_{i=y} + ε/K`. Once the model is "approximately right" (correct token among its top candidates), further sharpening gets diminishing reward. If the composition depth ceiling is caused by m-learning crowding out L-learning, label smoothing should free capacity for deeper composition.

## Design

Same controlled setup as the regime trajectory: L=6/m=4, 6L/6H/192D (~2.7M params), 20M tokens, v=8, s=2.

Sweep: ε ∈ {0.0, 0.05, 0.1, 0.2, 0.4}. All models trained with identical seeds (same initialization, same batch sequence). Only the training loss differs — `F.cross_entropy(..., label_smoothing=ε)`.

Evaluation always uses standard CE (no smoothing) for comparability. Per-level loss, accuracy, and output entropy measured at ~11 checkpoints over 20K steps. FM (2L/1H/24d/mlp1, ~14% of gap) trained on each converged model for residual analysis.

With v=8, the smoothed target distributions are:

| ε | p(correct) | p(each wrong) | Min achievable loss |
|---|-----------|---------------|-------------------|
| 0.0 | 1.000 | 0.000 | 0.000 |
| 0.05 | 0.956 | 0.006 | ~0.27 |
| 0.1 | 0.913 | 0.013 | ~0.47 |
| 0.2 | 0.825 | 0.025 | ~0.84 |
| 0.4 | 0.650 | 0.050 | ~1.39 |

## Results

### Per-level loss at convergence (standard CE eval)

Uniform baseline: ln(8) = 2.079.

| ε | val | L0 | L1 | L2 | L3 | L4 | L5 |
|---|-----|------|------|------|------|------|------|
| 0.00 | 1.364 | 0.889 | 1.726 | 1.898 | 1.938 | 1.936 | 1.924 |
| 0.05 | 1.379 | 0.915 | 1.729 | 1.899 | 1.936 | 1.935 | 1.924 |
| 0.10 | 1.399 | 0.950 | 1.742 | 1.906 | 1.940 | 1.937 | 1.928 |
| 0.20 | 1.439 | 1.018 | 1.761 | 1.918 | 1.948 | 1.946 | 1.936 |
| 0.40 | 1.538 | 1.178 | 1.815 | 1.947 | 1.972 | 1.971 | 1.963 |

### Per-level accuracy

| ε | overall | L0 | L1 | L2 | L3 | L4 | L5 |
|---|---------|------|------|------|------|------|------|
| 0.00 | 0.423 | 0.584 | 0.304 | 0.213 | 0.201 | 0.202 | 0.212 |
| 0.05 | 0.426 | 0.588 | 0.306 | 0.213 | 0.200 | 0.201 | 0.212 |
| 0.10 | 0.425 | 0.587 | 0.305 | 0.212 | 0.200 | 0.200 | 0.211 |
| 0.20 | 0.426 | 0.587 | 0.305 | 0.214 | 0.200 | 0.203 | 0.210 |
| 0.40 | 0.426 | 0.588 | 0.305 | 0.214 | 0.200 | 0.201 | 0.213 |

### Per-level output entropy

| ε | overall | L0 | L1 | L2 | L3 | L4 | L5 |
|---|---------|------|------|------|------|------|------|
| 0.00 | 1.355 | 0.907 | 1.737 | 1.892 | 1.918 | 1.915 | 1.913 |
| 0.05 | 1.454 | 1.068 | 1.780 | 1.919 | 1.942 | 1.940 | 1.939 |
| 0.10 | 1.532 | 1.193 | 1.820 | 1.938 | 1.957 | 1.956 | 1.955 |
| 0.20 | 1.657 | 1.393 | 1.879 | 1.976 | 1.989 | 1.988 | 1.988 |
| 0.40 | 1.842 | 1.689 | 1.971 | 2.024 | 2.032 | 2.032 | 2.032 |

### Per-level loss change vs baseline (ε=0)

| ε | Δval | ΔL0 | ΔL1 | ΔL2 | ΔL3 | ΔL4 | ΔL5 |
|---|------|-------|-------|-------|-------|-------|-------|
| 0.05 | +0.015 | +0.025 | +0.003 | +0.001 | -0.002 | -0.001 | +0.000 |
| 0.10 | +0.035 | +0.061 | +0.016 | +0.008 | +0.002 | +0.001 | +0.004 |
| 0.20 | +0.075 | +0.128 | +0.034 | +0.020 | +0.010 | +0.010 | +0.012 |
| 0.40 | +0.173 | +0.288 | +0.089 | +0.049 | +0.034 | +0.035 | +0.039 |

### FM residual properties

| ε | cos | 1-cos | res_norm | rank% | top1% | fL0* | fL1* | fL2* | fL3* | fL4* | fL5* |
|---|-----|-------|----------|-------|-------|------|------|------|------|------|------|
| 0.00 | 0.925 | 7.5% | 14.28 | 68.6 | 20.6 | .002 | .004 | .010 | .025 | .090 | .035 |
| 0.05 | 0.929 | 7.1% | 12.42 | 73.6 | 14.2 | .002 | .004 | .012 | .028 | .055 | .025 |
| 0.10 | 0.917 | 8.3% | 12.41 | 74.4 | 12.0 | .002 | .005 | .015 | .039 | .085 | .047 |
| 0.20 | 0.914 | 8.6% | 12.71 | 74.3 | 11.1 | .002 | .004 | .015 | .036 | .046 | .033 |
| 0.40 | 0.933 | 6.8% | 11.06 | 74.7 | 14.5 | .001 | .004 | .013 | .035 | .045 | .023 |

## Key findings

### 1. Label smoothing does not extend compositional depth

The central hypothesis — that softening the NTP objective would redirect capacity from m-learning to L-learning — is rejected. Label smoothing hurts at **every** hierarchy level. The damage is largest at L0 (ΔL0 = +0.29 at ε=0.4) and tapers at higher levels (ΔL3 = +0.03), but no level improves. Accuracy is essentially identical across all ε values at every level (~0.587 at L0, ~0.213 at L2, ~0.200 at L3-L5).

The model learns exactly the same compositional depth regardless of how strongly the sharpening incentive is suppressed.

### 2. Label smoothing changes the expression, not the structure, of learned computation

Output entropy rises dramatically with ε (L0: 0.91 → 1.69), but accuracy doesn't change. The model learns the same compositional structure and expresses it with softer probability distributions. The "freed" capacity from reduced sharpening doesn't transfer to deeper composition — it just isn't used.

### 3. FM residuals are smaller but not more predictable

Residual norm drops 23% (14.3 → 11.1) from ε=0 to ε=0.4, and effective rank rises (68.6% → 74.7%), but cosine similarity is flat (~0.92). The FM finds each model's computation equally difficult to predict. The smoothed models' activations are smaller in magnitude (softer distributions mean smaller logit-space gradients propagating through the network), which mechanically reduces residual norm without changing the FM's prediction quality.

### 4. The composition depth ceiling is a capacity constraint, not an objective-induced bias

This is the main takeaway. The per-level loss decomposition (see [PER_LEVEL_LOSS_README](PER_LEVEL_LOSS_README.md)) showed a 2.7M model at m=4 can only compose ~1-2 levels. We hypothesized this might be partly caused by the NTP objective steering capacity toward m-sharpening. The experiment shows it is not — the ceiling persists unchanged across a 10x range of smoothing intensity (ε=0 to 0.4). The model genuinely cannot compose deeper with this architecture at m=4, regardless of the training signal.

This means m-sharpening at level 0 is not competing with L-learning at levels 2+. The model has already allocated what capacity it can to L-learning, and uses whatever is left over for sharpening. Removing the sharpening incentive doesn't liberate capacity because there was no competition to begin with.

## Open questions

1. **Would the result differ at m=2?** At m=2, the model composes 2-3 levels (vs 1-2 at m=4). If the model is closer to its capacity limit at m=2, the competition between m-sharpening and L-learning might actually exist there. The ceiling might be objective-dependent at some (model size, m) settings but not others.

2. **What about at larger scale?** A model that has headroom to learn deeper but "chooses" to sharpen instead would show the effect. The current 2.7M model may simply be too small for the competition to arise — it's capacity-saturated at m=4, so there's nothing to redistribute. A 10M+ parameter model at m=4 might behave differently.

3. **Does this extend to other soft objectives?** Label smoothing is one way to reduce sharpening incentive. Others include temperature scaling in the loss, entropy regularization, or explicitly weighting per-level loss contributions. The geometric weighting argument (level 0 gets 51% of gradient) could be tested directly by reweighting the loss to give equal weight per hierarchy level.

4. **Per-level loss reweighting as a direct test.** Rather than softening the objective globally (which affects all levels), we could reweight the loss to give equal gradient per hierarchy level: `L = Σ_k (1/L) · L_k` instead of `Σ_k w_k · L_k`. This directly removes the geometric bias without changing the sharpness of the objective. If the ceiling still doesn't move, it confirms the capacity interpretation. If it does, the bias was in the weighting, not the sharpness.

## Reproduction

```bash
cd experiments/
modal run --detach -m rhm.rhm_label_smoothing::label_smoothing_sweep
```

Results saved to `rhm-scaling-data` volume at `/data/rhm_label_smoothing/`.
