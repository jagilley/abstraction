# Language Reduction: Synthetic (RHM)

**Idea doc**: [ideas/language_reduction.md](../../ideas/language_reduction.md)
**Prior experiment**: [experiments/language_reduction/STATUS.md](../language_reduction/STATUS.md) (natural-language results that motivated this)
**Related**: [experiments/a2a_forward/README.md](../a2a_forward/README.md) (A2A forward model work whose residual-rank observations are tested here)

## Goal

Controlled scaling-law and forward-model experiments using the Random Hierarchy Model (RHM; Cagnetta & Wyart, 2024). Our natural-language experiments showed that vocab-only reduction is a channel intervention (changes beta, not gamma), and spectral denoising was entangled with the statistics being measured. The RHM gives us a generative process with fully controllable hierarchical depth, where we know ground truth and can cleanly separate DGP vs channel interventions.

## The Random Hierarchy Model

The RHM generates sequences of length s^L from vocabulary {0, ..., v-1} via a hierarchy of composition rules. Each feature at level l has m rules, each mapping to an s-tuple of level-(l-1) features. This creates multi-scale correlations: tokens at distance s^l are correlated through level-(l+1) structure.

Key parameters:
- **L** (depth): number of hierarchical levels. Primary DGP intervention. More levels = deeper hierarchy = richer long-range structure.
- **m** (synonymic multiplicity): number of equivalent composition rules per feature. More synonyms = more entropy per level. Controls the per-level ambiguity the model must resolve.
- **s** (branching factor): size of each compositional tuple. Controls sequence length (s^L) and correlation scale spacing.
- **v** (vocabulary size): number of token types. Channel intervention — should change beta but not gamma.

Because we know the full generative process, every next-token prediction maps to a specific hierarchy level (via the s-adic valuation of the position), and every token has a ground-truth ancestry trace (which rule and feature were used at each level). This enables decompositions that are impossible on natural language.

## Architecture

All computation runs on Modal (workspace `jagilley`). Data lives on the `rhm-scaling-data` volume.

Models: autoregressive GPT-2 transformers trained on concatenated RHM sequences, from 4L/4H/128D (~0.8M params) up to 8L/8H/256D (~6.3M params). Forward models: TransformerForwardModel (same as in the A2A experiments), predicting later-layer activations from earlier-layer activations.

## Files

| File | Purpose |
|---|---|
| `shared.py` | Modal infrastructure (app, volume, image, utilities) |
| `stages.py` | Reusable experiment primitives: `generate_corpus`, `train_model`, `sweep`, `measure_scaling` |
| `rhm_data.py` | RHM data generation (hierarchy rules + corpus sampling) |
| `model.py` | GPT-2 with `return_intermediates` support |
| `measure.py` | Scaling exponent fitting (log-log regression) |
| `hparam_sweep.py` | L x m scaling exponent sweep |
| `rhm_residual_rank.py` | FM residual rank experiments: DGP sweep, FM capacity sweep, architecture-matched sweep |
| `rhm_regime_transition.py` | Regime transition: hierarchy-conditioned eta^2 at checkpoints over training |
| `rhm_cosine_sweep.py` | Cosine sweep: finding FM capacity settings where the FM genuinely struggles |
| `rhm_regime_trajectory.py` | Regime trajectory: the L-to-m transition at scale (2.7M model, m=4) |
| `rhm_per_level_loss.py` | Per-level loss decomposition: cross-entropy by hierarchy level over training |
| `rhm_dgp_approximation.py` | FM as DGP approximation: does the FM learn the RHM composition rules? |
| `rhm_fm_intermediate_probing.py` | FM intermediate probing: does the FM's internal computation mirror the hierarchy? |
| `rhm_label_smoothing.py` | Label smoothing sweep: does softening NTP shift learning from m to L? |
| `rhm_focal_loss.py` | Focal loss sweep: does confidence-based position weighting shift learning from m to L? |
| `rhm_confidence_threshold.py` | Confidence threshold sweep: aggressive gradient reallocation + scaled model |
| `README.md` | This file |
| `SWEEP_README.md` | [Scaling exponent sweep](SWEEP_README.md) |
| `RESIDUAL_RANK_README.md` | [FM residual rank experiments](RESIDUAL_RANK_README.md) |
| `REGIME_TRANSITION_README.md` | [Regime transition, cosine sweep, and trajectory](REGIME_TRANSITION_README.md) |
| `PER_LEVEL_LOSS_README.md` | [Per-level loss decomposition](PER_LEVEL_LOSS_README.md) |
| `LABEL_SMOOTHING_README.md` | [Label smoothing experiment](LABEL_SMOOTHING_README.md) |
| `FOCAL_LOSS_README.md` | [Focal loss & confidence threshold experiments](FOCAL_LOSS_README.md) |

## Results

### Scaling exponent sweep (2026-06-20)

**Full writeup**: [SWEEP_README.md](SWEEP_README.md)

Measures how the empirical scaling exponent alpha_D depends on L (hierarchy depth) and m (synonymic multiplicity). All models: 4L/4H/128D GPT-2 (~0.8M params), v=8, s=2. Two independent runs reproduce within ~0.01.

| L | m | alpha_D | R^2 |
|---|---|---------|-----|
| 4 | 2 | 0.498 | 0.978 |
| 4 | 8 | 0.327 | 0.975 |
| 6 | 4 | 0.217 | 0.953 |
| 8 | 2 | 0.438 | 0.987 |

**m dominates by ~3:1**. At fixed L=4, quadrupling m (2 to 8) cuts alpha by 34%. At fixed m=2, doubling L (4 to 8) cuts alpha by only 12%. The scaling bottleneck is synonymic multiplicity (per-level entropy), not hierarchy depth.

**Reproduction**: `modal run --detach rhm/hparam_sweep.py::hparam_sweep`

### FM residual rank experiments (2026-06-20)

**Full writeup**: [RESIDUAL_RANK_README.md](RESIDUAL_RANK_README.md)

Tests whether the effective rank of a forward model's residual reflects DGP complexity. Three experiments:

1. **DGP sweep (L x m)**: 9 settings, 1-block gap. Within each L, rank monotonically decreases with m — but this tracks learning quality (poorly-learned models do simple, low-dimensional computation), not DGP complexity per se.

2. **FM capacity sweep**: Same model, FMs from 10% to 100% capacity. Rank monotonically *increases* with FM capacity — the opposite of what "rank reflects DGP complexity" predicts. Dominated by the head-count mismatch (1H FM vs 4H main model): removing structured architectural-mismatch errors and leaving diffuse residual noise mechanically increases rank.

3. **Architecture-matched FM sweep** (key result): 4H FMs matching the main model block, 25% to 100% capacity. Residual norm drops 17x (0.57 to 0.03) while rank barely moves (90-96%). The residual's shape is invariant to FM capacity — rank structure is a property of the main model's computation, not the FM.

**Reproduction**: `modal run --detach rhm/rhm_residual_rank.py::rhm_residual_rank_sweep`

### Regime transition experiment (2026-06-21)

**Full writeup**: [REGIME_TRANSITION_README.md](REGIME_TRANSITION_README.md)

Tests whether the FM residual transitions from diffuse to rule-conditioned as the model learns the hierarchy. Three settings (m=2,4,8) at L=4, with ground-truth hierarchy-conditioned eta^2 at each checkpoint.

**Result: not testable at this scale.** The FM captures 99%+ of the computation at seq_len=16 (cosine 0.994 vs 0.903 on MNIST where the structured phenomena emerge). The eta^2 measurement correctly reports no structure because there is none to find — the residual is architectural mismatch noise, not computational gap.

**Reproduction**: `modal run --detach rhm/rhm_regime_transition.py::rhm_regime_transition`

### Cosine sweep and regime trajectory (2026-06-21)

**Full writeup**: [REGIME_TRANSITION_README.md](REGIME_TRANSITION_README.md) (appended sections)

**Cosine sweep**: Swept L in {5,6}, m in {2,4,8}, two FM sizes. Found that higher m makes the FM's job *easier* (the model barely learns, so computation is trivially predictable). L=6/m=2 with a 14%-capacity FM gives cosine 0.963 — in the sweet spot where the A2A meta-learning phenomena emerge. We also scaled up to seqlen = 64.

**Regime trajectory** (the main positive result): Scaled up to 6L/6H/192D (~2.7M params) at L=6/m=4, tracking FM residual structure over 20K steps. Key finding: we saw the L-to-m transition. **This should be the default RHM model size/hparams we use for all future experiments.**

Three signatures of the L-to-m transition:

1. **Feature eta^2 rises monotonically**: fL4* goes 0.006 to 0.074 (12x), fL3* goes 0.002 to 0.050 (25x). The FM's errors become increasingly conditioned on hierarchical feature identity, propagating up the hierarchy (local composition learned first, abstract later).

2. **Top-1 PC is non-monotonic**: rises to 33% at step 4000 (one dominant FM error mode), then drops to 12% by step 20000 (multiple specific error modes). The residual diversifies from L-regime (generic capacity gap) to m-regime (multiple rule/feature discriminations).

3. **Cosine enters the sweet spot**: 0.969 at step 7000, 0.921 at step 20000.

**Reproduction**: `modal run --detach rhm/rhm_regime_trajectory.py::run_m4_scaled`

### Per-level loss decomposition (2026-06-21)

**Full writeup**: [PER_LEVEL_LOSS_README.md](PER_LEVEL_LOSS_README.md)

Since every position maps to a hierarchy level via its s-adic valuation, we decompose cross-entropy by level over training. Two experiments at L=6:

1. **m=2, 4L/128D**: Loss monotonically increases with level. The model learns bottom-up — level 0 drops from 2.04 to 0.37 (82% below uniform) in the first 200 steps while levels 3-5 barely budge. A 4-layer model plateaus at 2-3 levels of learned composition.

2. **m=4, 6L/192D**: Same bottom-up pattern but m=4 makes every level harder. Despite 3.3x more parameters, levels 2-5 are bunched near baseline (~1.93 vs uniform 2.08). The model can only compose 1-2 levels at m=4.

This provides the mechanistic picture behind the L-to-m transition: the bottom-up learning wave drives the monotonic rise in feature eta^2, and the composition depth ceiling explains why the transition saturates.

**Reproduction**: `modal run --detach rhm/rhm_per_level_loss.py::per_level_trajectory --depth 6 --m 2`

### FM as DGP approximation (2026-06-21)

**Full writeup**: [PER_LEVEL_LOSS_README.md](PER_LEVEL_LOSS_README.md) (appended section)

Tests whether the FM learns the RHM's composition rules, not just a statistical summary of the target activations. Compares eta^2(FM predictions, rule/feature identity) to eta^2(actual activations, rule/feature identity) at each hierarchy level, using the converged 2.7M model at L=6/m=4.

At levels 0-3 (where the per-level loss decomposition showed the model has learned the hierarchy), the FM captures **91-97% of the feature-conditioned structure** in the actual activations. The FM adds almost exactly the same delta of feature structure beyond its input as the actual blocks 1-3 (89-96% match). At levels 4-5 (barely learned), the FM **overshoots** — its predictions are more feature-conditioned than the actual activations, because the FM's limited capacity captures the DGP-aligned component while missing the representational reorganization that the actual model's computation produces as a side-effect.

This provides direct evidence for the dissociation claim in the forward self-models paper: the FM captures the compositional function (the DGP's rules) while remaining agnostic to the representational side-effects of the model's full computation.

**Reproduction**: `modal run --detach -m rhm.rhm_dgp_approximation::dgp_approximation`

### Label smoothing experiment (2026-06-22)

**Full writeup**: [LABEL_SMOOTHING_README.md](LABEL_SMOOTHING_README.md)

Tests whether the NTP objective's sharpness bias causes the composition depth ceiling. The hypothesis: cross-entropy's geometric position weighting (level 0 gets ~51% of gradient) combined with the reward for sharpening creates an incentive for m-learning over L-learning. Label smoothing (ε ∈ {0.0, 0.05, 0.1, 0.2, 0.4}) should cap the sharpening reward and free capacity for deeper composition.

**Result: the composition depth ceiling is a capacity constraint, not an objective-induced bias.** Label smoothing hurts at every hierarchy level — L0 loss increases from 0.89 to 1.18 at ε=0.4, but levels 2-5 also get worse (not better). Accuracy is essentially identical across all ε values at every level (~0.587 at L0, ~0.213 at L2, ~0.200 at L3-5). The model learns exactly the same compositional depth regardless of smoothing intensity. Output entropy rises dramatically (L0: 0.91 → 1.69) confirming the model produces softer distributions, but the compositional structure is unchanged.

FM residual analysis shows smaller norms (14.3 → 11.1) but flat cosine (~0.92). The FM finds smoothed models equally hard to predict — activations are just smaller in magnitude.

**Reproduction**: `modal run --detach -m rhm.rhm_label_smoothing::label_smoothing_sweep`

### Focal loss & confidence threshold experiments (2026-06-22)

**Full writeup**: [FOCAL_LOSS_README.md](FOCAL_LOSS_README.md)

Tests the *position weighting* axis (complementing label smoothing's *sharpness* axis). Focal loss (FL(p_t) = -(1-p_t)^γ · log(p_t)) downweights confident positions; confidence thresholding (zero gradient for p(correct) > τ) is the extreme version. Both are DGP-agnostic. Two experiments: focal sweep at 2.7M params, threshold sweep at 6.3M params (8L/8H/256D).

**Result: gradient reallocation preserves compositional learning while reducing m-sharpening overhead and improving FM legibility.** At τ=0.3 (6.3M model), L0 learning drops 26.5% but L2-L3 retain 100% of baseline learning. At γ=2, L2-L3 *slightly improve* (101% retained) with minimal L0 cost (98% retained). FM cosine improves monotonically (0.940 → 0.955 at 6.3M), meaning the model's computation becomes more predictable to a compressed self-model. At τ=0.3, fL4* increases 31% — the FM's errors become more structured around the hierarchy. This is qualitatively different from label smoothing, where FM cosine was flat.

**Implication for meta-learning**: focal loss (γ≈2) during pretraining could prime models for the A2A self-knowledge loop by making computation more legible to a forward self-model, at minimal NTP performance cost.

**Reproduction**: `modal run --detach -m rhm.rhm_focal_loss::focal_loss_sweep` and `modal run --detach -m rhm.rhm_confidence_threshold::confidence_threshold_sweep`

## CLI

Primitives in `stages.py` can be used directly or imported into experiment scripts:

```bash
cd experiments/

# Generate corpus
modal run --detach rhm/stages.py::generate_corpus \
    --v 8 --s 2 --depth 6 --m 4 --n-tokens 20000000

# Train single model
modal run --detach rhm/stages.py::train_model \
    --v 8 --s 2 --depth 6 --m 4 --n-tokens 100000

# Full scaling sweep (7 P values, parallel)
modal run --detach rhm/stages.py::sweep \
    --v 8 --s 2 --depth 6 --m 4
```

## Modal volume

Results saved to `rhm-scaling-data` volume:

```
/data/
├── v{v}_s{s}_L{L}_m{m}/           # Per-setting data
│   ├── corpus.npy                   # Generated corpus
│   ├── rules_L*.npy                 # Composition rules per level
│   ├── meta.json                    # Setting metadata
│   ├── models/P_{n_tokens}/         # Trained models at each P
│   │   ├── model.pt
│   │   └── results.json
│   └── regime_transition/           # Regime transition checkpoints
├── hparam_sweep_compact.json        # Scaling sweep aggregate results
├── rhm_regime_trajectory/           # Regime trajectory results
├── rhm_per_level_loss/              # Per-level decomposition results
├── rhm_dgp_approximation/          # FM as DGP approximation results
├── rhm_label_smoothing/            # Label smoothing sweep results
├── rhm_focal_loss/                 # Focal loss sweep results
└── rhm_confidence_threshold/       # Confidence threshold sweep results
```

## Next steps

1. **Closed-loop A2A on RHM**: Now that the L=6/m=4 scaled model produces a meaningful FM cosine gap (~0.92-0.96), inject the FM prediction back into the main model's residual stream. Test whether the gate/self-knowledge/robustness phenomena from MNIST and language replicate, with the advantage that the RHM provides ground-truth rule identity for interpreting gate selectivity.

2. **Scale model to learn higher m**: The m=4 model at 2.7M params can only compose 1-2 levels. A larger model that learns 3-4 levels at m=4 would produce a richer m-regime with higher absolute eta^2 values and more interpretable gate structure.

3. **Vocabulary as channel intervention**: Sweep v at fixed (L, m) to test the prediction that v changes beta (overall difficulty) but not gamma (scaling exponent shape). The RHM makes this a clean test — v changes the observation alphabet without changing the hierarchical composition structure.
