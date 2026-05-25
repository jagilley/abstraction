# Experiment: Comprehensive Scaling Sweep

**Date**: 2026-05-05
**Builds on**: STATUS.md (τ sweep), Cagnetta et al. (2602.07488)
**Status**: Spec

## Motivation

The τ sweep established two key findings:
1. The Cagnetta formula α_D = γ/(2β) works on natural language (τ=0): predicted 0.274, empirical 0.315 (3-point fit)
2. Mild denoising changes β and γ without changing empirical α_D, suggesting scaling is a property of deep generative structure rather than surface statistics

Both findings rest on 3-4 point scaling curves. The 3-point fit is exactly determined; the 4-point fit was distorted by P=100K (unreliable eval). We need denser scaling curves to distinguish signal from noise.

Additionally, the current analysis only compares within-τ losses (each model evaluated on its own distribution). This makes cross-τ comparisons of absolute loss invalid — lower loss may just mean easier target. We need a common evaluation metric.

## Design

### Axis 1: Denser P values

P = 100K, 300K, 1M, 3M, 10M, 30M, 100M (7 points, ~0.5 order spacing in log scale)

This gives proper 7-point power-law fits with residuals and confidence intervals. The intermediate values (300K, 3M, 30M) are exactly the missing points identified in STATUS.md.

### Axis 2: Cross-τ evaluation (τ=0 model on denoised data)

**Direction**: Evaluate **τ=0 models** (at all P) on each τ's denoised validation data.

**Why this direction, not the reverse**: A τ>0 model has randomly-initialized embeddings for tokens that were completely eliminated by denoising. Evaluating such a model on natural text produces catastrophic loss at positions containing those tokens (and corrupted hidden states when they appear in context). The τ=0 model has functional embeddings for all tokens, making it a clean evaluator on any distribution.

This gives:
- **Entropy floor estimates**: L(τ=0 model at P=100M, τ=X val) — upper bound on how well the simplified distribution can be modeled
- **Ceiling scaling**: How fast does a natural-language model improve on simplified text as it gets more training data?
- **Gap analysis**: L(τ=X model, P, τ=X val) - L(τ=0 model, P, τ=X val) — the cost of training on only the simplified distribution, as a function of P. If this gap shrinks with P, the simplified model is catching up to the full model on its own distribution.

### Axis 3: Parameter-limitation check

The current model (2L, 4H, 128D) has ~400K non-embedding parameters. With P=100M, there are ~250 tokens per parameter. This model may be parameter-limited at large P, which would mean the empirical α_D reflects the parameter bottleneck rather than the data distribution.

For each τ, train one additional model at P=100M with a larger architecture (4L, 8H, 256D, ~3.2M non-embedding params). If the larger model achieves significantly lower loss, the small model is parameter-limited at P=100M and the scaling curve may be artificially flattened.

### Axis 4: Improved evaluation

- **Eval interval**: ~40 checkpoints per run (eval_interval = max(20, n_steps // 40))
- **Eval batches**: 20 batches (vs current 5) for smoother loss estimates
- **Cross-eval batches**: 50 batches for the fixed τ=0 evaluation

### What we track

| Metric | What it measures | Cross-τ comparable? |
|--------|-----------------|-------------------|
| best_val_loss | Best within-τ loss during training | No (different entropies) |
| α_within | Scaling exponent of within-τ loss curve | **Yes** (shape, not level) |
| ceiling_loss | L(τ=0 model, τ=X val) at each P | Model-based entropy estimate |
| α_ceiling | Scaling of ceiling with P | Yes |
| gap | L(τ=X model) - L(τ=0 model) on τ=X val | **Yes** (same eval set) |
| overfitting_gap | final_loss - best_loss | Yes (within-distribution) |
| param_check_gap | L(small) - L(large) at P=100M | Indicates regime |

### What we do NOT track here

- β and γ (Cagnetta-specific; already measured in τ sweep)
- Embedding geometry (separate experiment)
- Continual learning metrics (separate research line)

## Regime identification

With 7 P values per τ, we can fit L(P) = A·P^{-α} + L_∞ as a 3-parameter nonlinear fit and examine residuals. Signs of parameter limitation:
- The curve flattens at large P (residuals are positive for P > 30M)
- The param check model does significantly better at P=100M
- α_within < α_cross (the model's own-distribution scaling is bottlenecked by capacity)

Signs of data limitation:
- Clean power-law fit across all 7 P values
- Param check model shows minimal improvement at P=100M
- α_within ≈ α_cross

## Compute estimate

Training: 7 P values × 5 τ values = 35 runs on A10G. Plus 5 param-check runs = 40 total.
- P≤3M: ~3 min each
- P=10M: ~15 min
- P=30M: ~45 min
- P=100M: ~60 min
- Param check (4L/256D at P=100M): ~90 min

With 7-way parallelism per τ: ~1.5 hours per τ. 5 τ values sequential: ~7.5 hours.
Cross-eval: 40 models × 1 eval = 40 quick runs (~1 min each) = ~40 min parallel.

Total wall time: ~8-9 hours.

## Output

A single comprehensive results JSON:
```
/data/results/comprehensive_scaling_v2.json
```

Plus individual model weights saved at:
```
/data/models/tau_{tau}/P_{P}/T_128/            (small model)
/data/models/tau_{tau}/P_{P}/T_128_L4_D256/    (param check, P=100M only)
```
