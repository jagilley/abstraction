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

Models: autoregressive GPT-2 transformers trained on concatenated RHM sequences, from 4L/4H/128D (~0.8M params) up to 6L/6H/192D (~2.7M params). Forward models: TransformerForwardModel (same as in the A2A experiments), predicting later-layer activations from earlier-layer activations.

## Files

| File | Purpose |
|---|---|
| `shared.py` | Modal infrastructure (app, volume, image, utilities) |
| `stages.py` | Reusable experiment primitives: `generate_corpus`, `train_model`, `sweep`, `measure_scaling` |
| `rhm.py` | RHM data generation (hierarchy rules + corpus sampling) |
| `model.py` | GPT-2 with `return_intermediates` support |
| `measure.py` | Scaling exponent fitting (log-log regression) |
| `hparam_sweep.py` | L x m scaling exponent sweep |
| `rhm_residual_rank.py` | FM residual rank experiments: DGP sweep, FM capacity sweep, architecture-matched sweep |
| `rhm_regime_transition.py` | Regime transition: hierarchy-conditioned eta^2 at checkpoints over training |
| `rhm_cosine_sweep.py` | Cosine sweep: finding FM capacity settings where the FM genuinely struggles |
| `rhm_regime_trajectory.py` | Regime trajectory: the L-to-m transition at scale (2.7M model, m=4) |
| `rhm_per_level_loss.py` | Per-level loss decomposition: cross-entropy by hierarchy level over training |
| `README.md` | This file |
| `SWEEP_README.md` | [Scaling exponent sweep](SWEEP_README.md) |
| `RESIDUAL_RANK_README.md` | [FM residual rank experiments](RESIDUAL_RANK_README.md) |
| `REGIME_TRANSITION_README.md` | [Regime transition, cosine sweep, and trajectory](REGIME_TRANSITION_README.md) |
| `PER_LEVEL_LOSS_README.md` | [Per-level loss decomposition](PER_LEVEL_LOSS_README.md) |

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

**Cosine sweep**: Swept L in {5,6}, m in {2,4,8}, two FM sizes. Found that higher m makes the FM's job *easier* (the model barely learns, so computation is trivially predictable). L=6/m=2 with a 14%-capacity FM gives cosine 0.963 — in the sweet spot where the A2A meta-learning phenomena emerge.

**Regime trajectory** (the main positive result): Scaled up to 6L/6H/192D (~2.7M params) at L=6/m=4, tracking FM residual structure over 20K steps. Three signatures of the L-to-m transition:

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
└── rhm_per_level_loss/              # Per-level decomposition results
```

## Next steps

1. **Closed-loop A2A on RHM**: Now that the L=6/m=4 scaled model produces a meaningful FM cosine gap (~0.92-0.96), inject the FM prediction back into the main model's residual stream. Test whether the gate/self-knowledge/robustness phenomena from MNIST and language replicate, with the advantage that the RHM provides ground-truth rule identity for interpreting gate selectivity.

2. **Scale model to learn higher m**: The m=4 model at 2.7M params can only compose 1-2 levels. A larger model that learns 3-4 levels at m=4 would produce a richer m-regime with higher absolute eta^2 values and more interpretable gate structure.

3. **Vocabulary as channel intervention**: Sweep v at fixed (L, m) to test the prediction that v changes beta (overall difficulty) but not gamma (scaling exponent shape). The RHM makes this a clean test — v changes the observation alphabet without changing the hierarchical composition structure.
