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

## Code & files

Full file-by-file reference (every `.py` and every auxiliary README with its one-line summary) lives in **[FILES.md](FILES.md)**. Quick map of the code:

- **Infrastructure**: `shared.py` (Modal app/volume/image), `stages.py` (reusable primitives: `generate_corpus`, `train_model`, `sweep`, `measure_scaling`), `rhm_data.py` (RHM generation), `model.py` (GPT-2 with `return_intermediates` + cerebellar callback), `measure.py` (scaling-exponent fitting).
- **Scaling & residual structure**: `hparam_sweep.py`, `rhm_residual_rank.py`, `rhm_regime_transition.py`, `rhm_cosine_sweep.py`, `rhm_regime_trajectory.py`, `rhm_per_level_loss.py`, `rhm_dgp_approximation.py`, `rhm_fm_intermediate_probing.py`.
- **Loss weighting**: `rhm_label_smoothing.py`, `rhm_focal_loss.py`, `rhm_confidence_threshold.py`, `rhm_fm_weighted_ntp.py`.
- **Ratchets (wake-sleep / RL)**: `rhm_ratchet.py`, `rhm_sparse_ratchet.py`, `rhm_rl_ratchet.py`, `rhm_rl_gen_distill.py`, `rhm_rl_gen_distill_extended.py`.
- **Meta-learning**: `rhm_fomaml_ratchet.py`, `rhm_fomaml_rl_ratchet.py`, `rhm_meta_learning.py`, `rhm_meta_learning_l2.py`, `rhm_reptile.py`, `rhm_reptile_sparse.py`.
- **Regularizer & latent loop**: `rhm_fm_regularizer.py`, `rhm_latent_loop.py`.

Each experiment section below links its own `Full writeup` auxiliary README; **[FILES.md](FILES.md)** collects those links in one table.

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

**Full writeup**: [LOSS_WEIGHTING_README.md](LOSS_WEIGHTING_README.md)

Tests the *position weighting* axis (complementing label smoothing's *sharpness* axis). Focal loss (FL(p_t) = -(1-p_t)^γ · log(p_t)) downweights confident positions; confidence thresholding (zero gradient for p(correct) > τ) is the extreme version. Both are DGP-agnostic. Two experiments: focal sweep at 2.7M params, threshold sweep at 6.3M params (8L/8H/256D).

**Result: gradient reallocation preserves compositional learning while reducing m-sharpening overhead and improving FM legibility.** At τ=0.3 (6.3M model), L0 learning drops 26.5% but L2-L3 retain 100% of baseline learning. At γ=2, L2-L3 *slightly improve* (101% retained) with minimal L0 cost (98% retained). FM cosine improves monotonically (0.940 → 0.955 at 6.3M), meaning the model's computation becomes more predictable to a compressed self-model. At τ=0.3, fL4* increases 31% — the FM's errors become more structured around the hierarchy. This is qualitatively different from label smoothing, where FM cosine was flat.

**Implication for meta-learning**: focal loss (γ≈2) during pretraining could prime models for the A2A self-knowledge loop by making computation more legible to a forward self-model, at minimal NTP performance cost.

**Reproduction**: `modal run --detach -m rhm.rhm_focal_loss::focal_loss_sweep` and `modal run --detach -m rhm.rhm_confidence_threshold::confidence_threshold_sweep`

### FM-surprise-weighted NTP (2026-06-23)

**Full writeup**: [LOSS_WEIGHTING_README.md](LOSS_WEIGHTING_README.md) (Experiments 3-4)

Replaces the heuristic focal loss γ with a principled, self-scheduling signal: co-train an FM alongside the model and weight each position's NTP loss by the FM's angular prediction error (1-cosine). The FM defines "boring" via its own learned compression of the model's computation — no hyperparameter needed.

**FM-surprise NTP alone** (fm_weighted): the weights go through three phases — (1) the FM overshoot (FM converges instantly, gives more gradient to L0 where computation is most active), (2) transition (model's computation outgrows FM), (3) focal-loss-like (L0 weight drops, L2-L5 rise). The phase 1→3 flip takes ~14000 steps. FM cosine reaches 0.966 (vs 0.940 focal, 0.925 ce), but L1-L2 performance costs from the Phase 1 overshoot.

**FM-surprise NTP + uniform local loss** (fm_wt_ll): adding λ=0.1 local loss (MSE between actual and FM-predicted intermediates, gradient through main model only) fixes Phase 1 by keeping the FM accurate throughout (MSE stable at ~0.01 vs growing to 0.72). The weight flip happens at step ~2000 instead of ~14000. Results:
- Val loss cost 5× smaller than focal (+0.002 vs +0.011)
- First L2 improvement of any objective modification (-0.001)
- L3-L4 also improve, matching focal
- FM cosine **0.995** — prediction gap drops from 7.5% (ce) to 0.5%, residual norm drops 10×
- Caveat: the tiny residual (norm 1.3) is likely dominated by architectural mismatch noise (1-head FM vs 6-head model), so rank/eta² comparisons with other conditions at 10× higher norm aren't well-controlled

**Reproduction**: `modal run --detach -m rhm.rhm_fm_weighted_ntp::fm_weighted_ntp_sweep` and `modal run --detach -m rhm.rhm_fm_weighted_ntp::fm_weighted_ll_run`

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
├── rhm_confidence_threshold/       # Confidence threshold sweep results
├── rhm_sparsity_sweep/            # Sparsity sweep results
└── rhm_sparse_ratchet/            # Sparse ratchet results (gated + fixed)
```

### Sparse ratchet at L=4: overparameterization test (2026-06-24)

**Full writeup**: [RHM_L4_RATCHET_README.md](RHM_L4_RATCHET_README.md)

Tests whether the ratchet's failure to compound is caused by the model being capacity-limited at higher compositional levels. Reduces L from 6 to 4 (seq_len 64 → 16), putting the 6L/6H/192D model in the overparameterized regime (~6 layers for ~4 compositional levels). Same code as the sparse ratchet, sweep over mask_rates = {0.0, 0.75, 0.90, 0.95}.

**Result: the regularization headroom hypothesis is not supported.** Dense NTP (mask=0.0): OL wins at every cycle. Sparse NTP: WS shows improvements at mask≥0.90 (peak +2.0% at mask=0.90 cycle 1, +1.5% at mask=0.95 cycle 3) but the gap does not compound — it peaks mid-training and narrows by cycle 4. The FM cosine is very high (0.987–0.991 for WS), meaning the FM captures 98–99% of computation and the residual is dominated by architectural noise. Self-knowledge probes invert (OL > WS at post_block5: 0.42 vs 0.20), the opposite of MNIST and language, because the tiny residual has no structured signal to predict.

Overparameterization does not produce MNIST-like compounding. The remaining unexplained differences between MNIST (which compounds) and RHM (which does not) involve task structure (classification vs autoregressive NTP), FM architecture (bidirectional vs causal), and/or residual structure (low-rank digit-discriminative vs high-rank diffuse).

**Reproduction**: `modal run --detach -m rhm.rhm_sparse_ratchet::rhm_sparse_ratchet --depth 4 --mask-rates "0.0,0.75,0.90,0.95"`

### RL ratchet: REINFORCE with FM supervision (2026-06-25)

**Full writeup**: [RHM_RL_RATCHET_README.md](RHM_RL_RATCHET_README.md)

Tests whether RL's sparse-but-rich supervision (like classification) combined with FM process supervision unlocks the ratchet on RHM. The model generates the second half of RHM sequences autoregressively and receives a REINFORCE reward (fraction of correct tokens). 12 runs spanning NTP regularization, FM capacity, local loss, prediction gap, and DGP difficulty.

**RL+FM improves generation by 40% at all hierarchy levels** (38.5% standalone vs 27.8% OL). Vanilla RL collapses representations (flat eta² across layers); the FM prevents this. Self-knowledge appears under RL for the first time on RHM (R²=0.633 vs OL's 0.544). The gate stays open (0.67-0.998) — injection is genuinely useful for generation, unlike NTP where it's redundant.

**The ratchet doesn't compound because distillation is both load-bearing and destructive.** Without distillation, generation collapses to OL baseline (27.4%) — it's the only mechanism that transfers FM-assisted generation into standalone weights. But distillation destroys NTP (val loss 0.82 during wake → 2.3-3.5 after distillation) because the teacher's injected computation is too different from standalone computation. A systematic cosine regime sweep (runs 8-12) found that the FM cosine is controlled by distillation (not local loss or FM capacity), and a 3-block prediction gap (blk0→blk3) is the only configuration that achieves the 0.94 sweet spot — but activation norm inflation without local loss destabilizes training. Even λ_local=0.001 pushes cosine back to 0.987.

**Generation-based distillation solves the NTP destruction problem** (run 13). Instead of distilling on NTP data (where the teacher's logits are ~100% injection-derived), the teacher generates suffixes autoregressively with FM injection, and the student matches those logits. NTP is preserved via separate CE on ground-truth data. Result: val loss 0.813 (vs 2.262 for NTP distillation, 0.789 for OL) while generation transfers equally well (39.1% vs 38.4%). Also unlocks sweet-spot FM cosine (0.911), 7× higher self-knowledge (0.572 vs 0.077), and progressive eta² (4.3× L5 decrease through the network). Outcome metrics (generation accuracy, NTP loss) don't compound, but L3 feature eta² at the final layer increases monotonically across cycles (+15% over 4 cycles, 0.389→0.449), suggesting representational deepening that may precede outcome improvement.

**Reproduction**: `modal run --detach -m rhm.rhm_rl_ratchet::rhm_rl_ratchet --only-rl-fm --ntp-mask-rate 0.95` and `modal run --detach -m rhm.rhm_rl_gen_distill::rhm_rl_gen_distill`

### FOMAML ratchet: bilevel meta-learning diagnostic (2026-06-26)

**Full writeup**: [RHM_FOMAML_README.md](RHM_FOMAML_README.md)

Tests whether the MNIST gated ratchet's FOMAML bilevel meta-learning produces compounding on RHM, or whether the failure of first-order methods on RHM is domain-specific. Three experiments with different outer objectives on the FOMAML gate:

1. **Dense NTP outer** (NTP wake): Gate closes to 0.002. NTP gradient at all 64 positions makes local loss redundant.
2. **RL outer** (RL wake, REINFORCE): Gate stuck at 0.5. REINFORCE variance drowns the meta-gradient.
3. **Per-level NTP outer** (RL wake, NTP at hierarchy levels >= 2): Gate closes to 0.000. Clean deterministic signal, and the answer is: local loss does not help compositional NTP.

**The diagnostic question is answered: the problem is domain-specific.** FOMAML doesn't compound on RHM under any outer objective. The MNIST equivalence (FOMAML ≈ unified gate) was specific to a regime where the FM's low-rank, class-discriminative residual naturally aligned with the task objective. On RHM, the FM's higher-rank residual doesn't point in compositionally useful directions — compressing intermediate computation toward FM-predictability does not improve level-2+ NTP quality, even after one full FOMAML inner step with the information advantage of 64 × 192 dims of local loss vs ~3 sparse NTP positions.

**Reproduction**:
```bash
modal run --detach -m rhm.rhm_fomaml_ratchet::rhm_fomaml_ratchet
modal run --detach -m rhm.rhm_fomaml_rl_ratchet::rhm_fomaml_rl_ratchet
```

### FM-as-regularizer beats weight decay's functional-complexity floor (2026-07-01)

**Full writeup**: [RHM_FM_REGULARIZER_README.md](RHM_FM_REGULARIZER_README.md)

The payoff of the frontier/WD arc ([RHM_FRONTIER_AND_LEGIBILITY_README.md](RHM_FRONTIER_AND_LEGIBILITY_README.md), Exp 3): on the m2 substrate, does *structured* FM-predictability pressure compress the functional circuit where generic L2 cannot? Co-train a matched-head FM and add an **open-loop** `λ·MSE(post_block6, FM(post_embed))` term (gradient through the main model only — **no injection/distillation/self-knowledge**), λ ∈ {0.03…3.0}, wd fixed at 0.1.

**Yes, on the leakage-proof metric.** Weight decay cannot push `post_block6` activation effective-rank below ≈52.6% at any setting (even wd=1.0, which breaks the root); FM-reg reaches **42.9% at preserved knowledge (d6=0.94)** — ~10 points below WD's floor — and does so with deep η² *enhanced* (λ=1.0: d4 η² 0.463, +26% over the wd=0.1 baseline), not eroded as under strong WD. This is "L2-norm complexity ≠ functional complexity" made empirical, and it answers the session's opening question: **self-knowledge is not load-bearing for functional simplification** — the simplest open-loop pressure suffices. reg-gap residual rank only *ties* the WD floor (77%); the win is on FM-free activation rank and η². Benign through λ=3.0 (zero val cost); λ=1.0 is the legibility sweet spot; weight norm shows a mild grow-then-compress turnover WD never does. Caveats: compression is localized to the regularized region (b0→b4 unaffected), single rule seed, top1-PC concentrating at λ=3.0.

**Reproduction**:
```bash
modal run --detach -m rhm.rhm_fm_regularizer::fm_reg_sweep --lams "0.3,1.0,3.0" --n-steps 300000
modal run --detach -m rhm.rhm_fm_regularizer::analyze_reg --lams "0.03,0.1,0.3,1.0,3.0"
modal run --detach -m rhm.rhm_fm_regularizer::wd_activation_rank
```

### Complexodynamics: the First Law of learning, with an attractor floor (2026-07-07)

**Full writeup**: [RHM_COMPLEXODYNAMICS_README.md](RHM_COMPLEXODYNAMICS_README.md)

Reads the saved m2 trajectories (FM-reg + WD-sweep per-checkpoint parts) as an empirical test of Aaronson's "First Law of Complexodynamics" (complexity rises then falls while entropy climbs; see `reading/complexodynamics.pdf`), with the capacity-bounded FM as the resource-bounded observer, and adds one new analysis-only run: `rhm_ensemble_trajectory.py`, fresh-FM-ensemble residual invariance measured over training checkpoints.

**The rise-and-fall was already in the saved parts, with two amendments to Aaronson's picture.** (1) The descent requires an annealer: FM-free activation rank rises 36→63.5% then falls to 42.9% under FM-reg pressure, but arrests at ~55% without it — SGD alone freezes mid-descent (a glass, not a liquid). (2) The curve descends not to zero but to the **DGP's own sophistication relative to the bound**: shallow-level residual η² completes the full arc (d1 0.10→0.05) while deep-level η² rises and persists at every horizon (still climbing at 300k) — the deep "arrest" is the floor, not a truncated transient. **The new ensemble run confirms the decomposition**: residual FM-invariance rises 0.555 (random init) → ~0.83 late, the un-pressured control dips to 0.666 exactly at the sophistication peak (mid-training scaffolding is observer-idiosyncratic), and the ensemble-mean residual — the invariant core — is deep-DGP-structured, purifying monotonically (d4 η² 0.016→0.503, exceeding any single FM's) with pressure making it smaller *and* more invariant (norm 1.1 vs 9.3, ens_cos 0.834 vs 0.809). Hump = trajectory-owned scaffolding; floor = problem-owned complexity — the same DGP-aligned residual RHM_LATENT_LOOP identified as the precondition for generalizable self-knowledge. Single rule seed; ens_cos not comparable across FM configs/gaps.

**Reproduction**:
```bash
modal run --detach -m rhm.rhm_ensemble_trajectory::ensemble_trajectory
```

## Next steps

1. ~~**Closed-loop A2A on RHM**~~: *Done* — see [RHM_RATCHET_README](RHM_RATCHET_README.md). The WS_UG_uniform ratchet replicates on RHM in dynamics (gate closing, FM tracking, robustness dissociation, crossover timing) but not in magnitude (0.3% vs MNIST's 48%). Confidence thresholding keeps the gate 1.8x more open at tau=0.3 but doesn't amplify the val loss gap. FM capacity must be matched (~2% of model) for meaningful dynamics — an oversized FM (12.5%) masks the gate-closing behavior. See also [RHM_SPARSE_RATCHET_README](RHM_SPARSE_RATCHET_README.md) — sparse NTP masking unlocks +1.2% val loss improvement and per-level compositional gains at L3-L5 in cycle 1.

2. **Domain shift on RHM**: The stationary-data ratchet shows meta-learning dynamics without learning magnitude. The RHM's controllable DGP enables a clean domain shift test — e.g., train on one rule set then shift to new rules at the same (L, m). This would test whether the ratchet's dynamics produce genuine adaptation advantages, as seen in the MNIST OOD experiments. The sparse ratchet's non-compounding L3-L5 improvement (exhausted after cycle 1 on fixed data) motivates this especially — novel data would sustain the need for higher-level compositional learning.

3. **Scale model to learn higher m**: The m=4 model at 2.7M params can only compose 1-2 levels. A larger model that learns 3-4 levels at m=4 would produce a richer m-regime with higher absolute eta^2 values and more interpretable gate structure.

4. **Vocabulary as channel intervention**: Sweep v at fixed (L, m) to test the prediction that v changes beta (overall difficulty) but not gamma (scaling exponent shape). The RHM makes this a clean test — v changes the observation alphabet without changing the hierarchical composition structure.

5. **Understand the MNIST-RHM ratchet gap**: The L=4 experiment ruled out capacity/overparameterization. The RL ratchet ruled out task structure (RL is sparse-but-rich like classification) and partially ruled out FM cosine regime (the 3-block gap achieves 0.944, but training is unstable). Gen-based distillation (run 13) resolved the distillation bottleneck (val loss 0.813 vs 2.262, generation 39.1%) and achieved sweet-spot FM cosine (0.911) — but the ratchet still doesn't compound. Remaining candidates: (a) bidirectional vs causal FM; (b) residual structure (MNIST's low-rank digit-discriminative residual vs RHM's higher-rank diffuse residual); (c) the ratchet may require novel data to sustain compression pressure (MNIST has 10 classes with varying difficulty; stationary RHM is exhausted after cycle 1).

6. **Stabilize λ_local=0 training**: The 3-block gap at λ_local=0 achieved sweet-spot cosine (0.944) with excellent dynamics for 2 cycles before activation norm inflation destabilized it. Gradient clipping or activation norm regularization (penalizing ||post_block3||² > threshold) could prevent the explosion without creating FM-predictability pressure. This preserves the rich dynamics (2.9× eta² progression, 0.299 self-knowledge) that only appear at λ_local=0. Gen-distill + λ_local=0 is a promising combination: gen-distill already achieves cosine 0.911 with λ_local=1.0, so removing the local loss feedback loop might push it further into the sweet spot.

7. **Self-model-driven grokking (north star)**: The gen-distill run 13 shows monotonically improving val loss and L3 feature eta² increasing +15% over 4 cycles at the final layer (0.389→0.449), with L4 also trending up (+4%). The ratchet is compounding at the representation level even though generation accuracy is flat — the compositional deepening hasn't yet crossed the threshold for behavioral improvement. The RHM's DGP is extremely compact (96 rules = 576 bits for L=6/m=2/v=8/s=2), and the 2.68M model is overparameterized by ~150,000× relative to this target — exactly the regime where grokking occurs. The FM captures 91-97% of the DGP's composition rules at learned levels, providing structured regularization pressure specifically toward the compact solution, unlike weight decay's generic L2 pressure. An extended gen-distill run (16-32+ cycles) could reveal whether the gradual L3 eta² improvement eventually produces a sudden phase transition — the model "clicking" on L3 composition rules, which would make L4 accessible and potentially trigger a cascade. This would be the first instance of self-model-driven grokking: a phase transition caused by a model's compressed self-model selecting for DGP-aligned computation, rather than by weight decay selecting for low-norm solutions.
