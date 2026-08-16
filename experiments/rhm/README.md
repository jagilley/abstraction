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
- **Ratchet / meta-learning arc** (wake-sleep, sparse, RL/gen-distill, FOMAML, meta-learning — *concluded, negative*): moved to **[`ratchet/`](ratchet/README.md)**. See that folder's README for the pan-arc summary, per-experiment index, and reproduction commands.
- **Regularizer & latent loop**: `rhm_fm_regularizer.py`, `rhm_latent_loop.py`.
- **The practice arc on RHM**: [`practice/`](practice/README.md) — practice as a control loop over the *conditions and units* of learning, ported from [`mjc/practice/`](../mjc/practice/README.md). Children: [`practice/crystallize/`](practice/crystallize/README.md) (certificate-gated compilation under priced feedback), [`practice/ratchet/`](practice/ratchet/README.md) (earning a level-indexed vocabulary over a depth ladder — distinct from the meta-learning [`ratchet/`](ratchet/README.md) arc), [`practice/ear/`](practice/ear/README.md) (climbing the evaluation layer; provisional commitment rescues the frontier), [`practice/recital/`](practice/recital/README.md) (self-paced eras; no internal signal prices time-at-the-bottom, 3 worlds), and [`practice/tall/`](practice/tall/README.md) (the depth-6 port attempt, closed with a diagnosed apparatus negative).
- **Self-report / introspection**: [`confabulation/`](confabulation/README.md) — the confabulation test built on the latent loop's `ntp_aux{,_cl}` substrate.
- **The conditioning gap / belief revision**: [`endogenous_teacher/`](endogenous_teacher/README.md) (is the model's own surprise separable from the text's, and can it teach?) → [`conditional_revision/`](conditional_revision/README.md) (the exact-BP oracle for belief revision, and whether the model's state tracks it).

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

> **Superseded as the instrument of record** by [`residual_decomposition/`](residual_decomposition/README.md) (2026-07-27), which explains Exp. 3's "17× norm range, rank barely moves" rather than contradicting it: entropy effective rank reads the residual's *shape* and is blind to its *magnitude*. Use β and `R_res_participation` from that node for any new rank-shaped measurement.

### Residual decomposition: what residual rank was actually measuring (2026-07-27)

**Full writeup**: [residual_decomposition/README.md](residual_decomposition/README.md) · **Also spans** [`a2a_forward/residual_decomposition/`](../a2a_forward/residual_decomposition/README.md) (language + vision)

Across RHM, language, and MNIST, the FM residual is not a set of leftover directions but a graded shadow of the whole computation, obeying **`res_var(i) ∝ act_var(i)^β`** across the main model's principal directions (R² = 0.95–0.99). β is invariant to a 4× FM-capacity range (±0.01) while that same sweep moves the frontier's *level* by 1.8× — **shape and level are separate quantities, and the old single `R_res` tracked neither.**

This falsifies the `R_act ≈ R_comp + R_res` partition in [`dimensionality_expansion.md`](../../beliefs/dimensionality_expansion.md) (ratio measures 1.85–3.56, never ~1; `R_comp` is a redundant readout of `R_act`; `R_res` > `R_act` everywhere), and explains three separate historical rank negatives — Exp. 3's norm-invariance above, "language residual is full-rank 200/256 yet cosine 0.97", and [`mjc` cut #1](../mjc/contact_residual/README.md)'s contact eff-rank 3.60 > free 2.52. All three were measuring a **saturated FM's noise floor**: where the FM drives relative residual to ~0, β collapses to 0.11–0.17 and naive rank inflates to 94–96% of `d_model` in all three domains.

The replacement instrument is two numbers: **β** (the shape, a property of the model's computation) and **`R_res_participation`** (the frontier's dimensionality, counted in the model's own basis weighted by the computation it actually does there — RHM 7 where naive rank says 84). The geometric intuition survives: at a wide prediction gap the residual sits in the model's top directions at **5–6× chance**.

**Reproduction**: `modal run --detach -m rhm.residual_decomposition.rhm_decomposition_audit::decomposition_audit`

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

### Ratchet / meta-learning arc — concluded (negative), moved to [`ratchet/`](ratchet/README.md)

The wake-sleep / sparse / RL-gen-distill / FOMAML / meta-learning "ratchet" line (2026-06-23 → 06-29) asked whether the MNIST gated ratchet — which compounds 34% → 48% over 4 cycles — also compounds on RHM. **It does not.** The ratchet reproduces every MNIST *dynamic* (gate closing, FM tracking, robustness dissociation, crossover timing) but never the *magnitude* (val-loss gap peaks ~1%, no compounding). The arc ruled out capacity (L=4), task structure (RL), the distillation bottleneck (gen-based distillation), and FM-cosine regime (3-block gap), and the FOMAML diagnostic settled it: **on stationary data no outer objective makes FM-predictability pressure improve compositional NTP** — compounding requires a moving frontier that stationary RHM can't supply, which is where the active / sculpting / latent-loop lines below went next.

**Full pan-arc summary, per-experiment index, open threads, and reproduction commands: [`ratchet/README.md`](ratchet/README.md).**

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

### The confabulation test: does a self-report track the implementation or a self-theory? (2026-07-22)

**Full writeup**: [confabulation/README.md](confabulation/README.md) | **Design doc**: [ideas/confabulation_test.md](../../ideas/confabulation_test.md)

The standing objection to any introspection claim (Nisbett & Wilson) is that a self-report may come from a learned *theory* of oneself rather than *access* to oneself. The FM decomposition `a_j = FM(a_i) + r` makes that operational: `FM(a_i)` is everything a self-theory could produce, so **confabulation lives in the range of the self-model and its complement is the residual**. A report head on `post_block7` reports the residual-direction cluster (IMPL) against three controls (BEHAV / ENT / WORLD), forking `rhm_latent_loop`'s `ntp_aux{,_cl}` wake recipe verbatim (reproduces its val loss to 3 dp).

**IMPL is the only target with a first-person advantage over a capacity-matched third-party observer** (+0.09 to +0.11): BEHAV is a dead null (−0.001), and on ENT and WORLD the observer *beats* the self-report — exactly as the criterion "not cheaply recoverable from the I/O map" predicts. The sharpest result is that **the observer ladder is flat in capacity** (`O_input` 0.376 → 0.376 from 64D to 256D; `O_io` plateaus by 2L/128D and a model-sized 8L/256D observer does no better), while on the input-determined WORLD target the same ladder climbs steeply (0.227 → 0.522) — the third party is *access*-limited, not resource-limited. Matched-KL steering moves the report 2.2× more along residual than prediction directions with BEHAV-flip matched (0.024 vs 0.023). Everything survives a 24× instrument-FM capacity sweep, gated on `ens_cos` (0.84–0.91) and hierarchy-η², because an over-capacity FM leaves a junk residual that *fakes* the whole signature (a smoke run at cosine 0.993 / `ens_cos` 0.65 produced +0.118 advantage from noise).

**Two pre-registered expectations failed.** The loop is **not necessary** — OL shows the same dissociation, weaker (margin +0.14 vs +0.33); and residual DGP-structure is *equal* in CL and OL (d6 η² 0.322 vs 0.295, matching the latent-loop reference exactly), so **closing the loop changes how well M knows itself, not what there is to know**. Also negative: `O_act` failed as a ceiling (0.666, below `O_io`), and the channel-ablation test is too blunt to carry weight. Single seed.

### Directed sculpting: the DGP substrate for porting mjc E3 to RHM (2026-07-28)

**Goal**: port E3's full inner/outer loop + forward model onto RHM sculpting. **Groundwork finding**: the naive port is a *confounded null by construction* — RHM rule usage is uniform (per-level feature-marginal entropy ≥ 0.93 everywhere), so `visits` is flat and `lprog × visits` collapses to `lprog`. Three verified primitives fix that: **distractor channels**, **support-fixed rule drift** (the DP `d*` cannot move, so no task-switch forgetting), and a **repair-cost instrument** whose attributable gap recovers the analytically known drift KL to 88–95% — with two traps documented (raw CE is *exactly* blind to this drift; the gap that replaces it is confounded by ordinary continued training). **Loop finding** ([`full_loop/`](directed_sculpting/full_loop/README.md)): the reward-free **relevance** signal ports and is a smoking gun — a value trained only on task success, never told which channel is which, separates real tokens from distractors **13–18×**, and allocating by it alone recovers **76%** of a privileged oracle. But **expansion is a property of the grader's *type*, not of non-stationarity**: a grounded evaluative grader expands the belief ~1.7× in effective dimension and ~15× in ballistic control where an endogenous dense grader caps *below the no-loop floor*, identically in static and drifting worlds. Both drift-dependent predictions come back negative under instruments repaired repeatedly, each repair removing a bias that favoured the positive — but the climbing null is **scoped**: it held only because surface re-fit repaired 91% of each drift event, and starving **samples per event** at fixed drift magnitude makes the depth advantage migrate from surface to deep (t = −5.44, 3/3 seeds), the first positive on that axis. **Two 2026-08-03 children extend what that value can be asked**, and the first of them **corrected itself**: given a [level-indexed action space](directed_sculpting/full_loop/level_moves/README.md), an apparent preference for higher hierarchy levels turned out to be **span** — a depth-matched distractor with ground-truth gain certified 0.000 rises 2.79× across levels against the tree's 2.56× — but a **matched-span paired control** (each committed move against a *lazy twin* rewriting identical tokens without committing) shows the value genuinely prefers the abstract commitment (0.557 / 0.618 at L2 / L3 against an oracle's 0.610 / 0.642), and once the DGP is repaired so that errors actually *require* abstraction to see — hierarchical damage, 100% on-grammar at matched `d*`, against published damage that is off-grammar in 29% of blocks — that preference is **graded by whether a commitment at that level can reach the error** (t = −6.81) and value–oracle rank agreement **doubles** (+0.29 → +0.58) even as the task gets harder. So the value is sensitive to the hierarchy's *compositional* structure rather than to depth as such, conservatively and increasingly so with depth; and a [shared-surface splice](directed_sculpting/full_loop/partial_hetero/shared_surface/README.md) supplies the cross-channel code the published partially-heterogeneous geometry lacked, while measuring that node's transfer instrument as too coarse to read either sweep's flat curve. Full writeups: [`directed_sculpting/README.md`](directed_sculpting/README.md) and [`directed_sculpting/full_loop/README.md`](directed_sculpting/full_loop/README.md).

### The endogenous teacher: the model's surprise is separable from the text's, but scalar weighting can't teach with it (2026-08-04/05)

**Full writeup**: [endogenous_teacher/README.md](endogenous_teacher/README.md) | **Child**: [cancellation/README.md](endogenous_teacher/cancellation/README.md) | **Idea**: [ideas/the_forecast_needs_a_lead.md](../../ideas/the_forecast_needs_a_lead.md)

In NTP the text is stimulus, target and teaching signal at once; in a brain the word is only the stimulus. **Gate 0 measures that the two signals really are distinct on RHM**: `R²(rres ~ nll)` = 0.113 pooled (0.112 within-position), so **88.7% of the FM residual's variance is orthogonal to token surprisal**, at `corr` = **−0.34** — and against ground-truth levels they are **anti-localised**, token surprisal peaking at the deep subtree boundaries (nll 2.61, rres 0.24) where internal residual is lowest, and residual peaking leaf-adjacent (nll 0.80, rres 0.44) where the text is nearly free. So the residual tracks **computational load, not epistemic difficulty** — known from language, but new here as a *resolved-against-a-known-hierarchy* measurement with an explicitly anti-correlated sign. **The intervention is a null**: five arms with bit-identical rank-normalised weight multisets (only the assignment differs) give `res` Δd4 +0.160 against `res_shuffled` +0.161, firing the pre-registered falsifier, with the residual demonstrably still live (rel-res rose 0.40→0.44 in every arm). Scoped to *scalar* weighting — two prior results ([EMOTION_INJECTION](../a2a_forward/EMOTION_INJECTION_README.md); "self-knowledge is directional, not scalar") predicted it. Side observations: **ordering by token surprisal is the worst assignment of all** (`nll` erases `uniform`'s val gain entirely — on RHM the high-nll positions are the irreducible ones), and `res` expanded participation ratio 5.02→6.90 where `res_shuffled` gave +0.36 (single seed, rank-shaped instrument, non-monotone across arms — recorded, not interpreted). The [cancellation child](endogenous_teacher/cancellation/README.md) then routes the residual through the forward path instead: the **corollary-discharge mechanism replicates closely across substrates** (`cos(Δ,inj)` +0.293 summation → **+0.088** cancellation at matched injection norm; the looped ViT read +0.89 → +0.095), but its payoffs do not — the fresh-FM swap costs +0.0024 vs +0.0015 against a **0.80-nat dependency** and a 1.38 random-FM floor, so downstream depends on *a forecast* rather than on *this forecaster* in both wirings, and cancellation's fresh-SK is marginally *worse* (Δ −0.099 vs −0.075). Pure computation relocation reproduces (injected val ties `ntp` to 4 dp, injected root ≥ `ntp`'s, standalone root halved). One caveat outstanding: the three fresh FMs all sit at cos 0.871–0.872 and were not checked for functional vs gauge distinctness (`ens_cos` would settle it). Single seed throughout. `rhm/model.py` gained a backwards-compatible `cerebellar_mode="cancel"` path, guarded by a bit-exactness regression test.

### Conditional revision: belief revision is separable from surprisal, at the levels where the model has a belief (2026-08-07/08)

**Full writeup**: [conditional_revision/README.md](conditional_revision/README.md) | **Design doc**: [conditional_revision/SPEC.md](conditional_revision/SPEC.md) | **Idea**: [ideas/revision_not_surprisal.md](../../ideas/revision_not_surprisal.md)

The measurement [`endogenous_teacher`](endogenous_teacher/README.md) needed before its interventions: token surprisal bundles *reducible* surprise (the token told me something about latent structure) with *irreducible* surprise (it was one of `m` synonymous realisations of structure I already had), and on RHM the split is **exactly computable by BP**, not a correlation hunt. **Gate 0** moves the FM's conditioning gap from depth to time (`FM(h6[≤t]) → Δ_t = h6[t+1] − h6[t]`, so the target depends on a token the FM cannot hold): `corr(res, nll)` flips **−0.337 → +0.652**, the level profile **−0.786 → +0.484**, `R²` 0.425 with 57.5% of variance still orthogonal, and both controls hold — `depth_cotrain` reproduces `endogenous_teacher`'s Gate 0 to three decimals and a protocol-matched `depth_frozen` reads −0.328, so the flip is the axis change and not the protocol. **The oracle is exact** (≤4.4e-16 vs brute-force enumeration; the identity `E[B_D] = H(x|prefix) − H(x|z_≤D,prefix)` holds to ≤7.9e-04), after a correction worth keeping: the RHM latent graph is a **hypertree**, not a pairwise tree — one rule emits a parent's whole `s`-tuple, so a parent–child-edge factorisation reads H = 1.733 where the truth is ln 4. Model-independently, **~49% of positions have `B` ≡ 0** while still carrying ~0.98 nats of surprisal, and `R²(B ~ nll)` runs **0.017 at the root to 0.441 leaf-adjacent** — the split is near-total at high abstraction and collapses near the leaves. **Gate B**, the pre-registered primary kill, holds position *and* exact surprisal fixed (position is a real confound: a position-shuffled null read 0.70 under surprisal matching alone; every column now carries a guard, all landing 0.489–0.507): the model's belief revision separates synonym from disambiguating positions at **0.690 (d2) and 0.614 (d3)** where surprisal is pinned at ~0.49 and the shuffled null at 0.501 — but **nothing above d3**, and d1 is *untestable* rather than null (within a position, exact surprisal separates the families at AUC 1.000). **Gate 0's residual is mostly surprisal**: `r_temporal`'s raw 0.64–0.75 falls to 0.50–0.52 nll-matched and 0.53–0.61 fully controlled. Two findings fall out sideways. **Read as a scalar residual norm the same event carries much less than read as a KL in belief coordinates** (0.606 vs 0.690 at d2; 0.005 vs 0.150 in Gate A) — a fourth directional-not-scalar instance. And the probe diagnostic found the model's per-position state is close to a **next-token-sufficient statistic**: already-resolved constituents, which Bayes decodes at 0.90–0.996, are carried at 0.20, while the current ancestor chain reads 65% of a much harder ceiling. The binding constraint on the whole cut is therefore **belief depth, not the conditioning gap** — Gate A crosses over exactly where the probe leaves its ceiling (partial `R²(M~B|nll)` 0.150 vs `R²(M~nll|B)` 0.005 at d1; reversed 0.008 vs 0.082 at the root, where base root recovery is 0.088). Single seed; Gate C (matched FM pair + capacity invariance) unrun; `M`'s readout is a probe trained on true latents, so it is grounded in the sense SPEC pre-registered. The child [`sculpt_slip/`](conditional_revision/sculpt_slip/README.md) ports the question to a **control** substrate (sculpting + Stage 3d's slippery actuator, where the aleatoric label is exact): the slip is identifiable in the FM residual but **only directionally** (0.70–0.75 AUC once ‖r‖ is matched vs 0.50 for the norm, the norm getting *worse* and the direction *better* as slip rises), while a low-rank precision operator is a **null** against its own geometry control — and a budget sweep then kills the line, the prize staying flat at +0.02–0.045 across 60× of data because the FM is **bias-dominated, not variance-dominated**, at every budget. An aleatoric filter can only pay when the learner is variance-limited, which is a two-arm precondition worth checking first.

### The practice arc on RHM: compilation under priced feedback (2026-08-14)

**Full writeup**: [practice/README.md](practice/README.md) → [practice/crystallize/README.md](practice/crystallize/README.md) | **Idea**: [ideas/practice_manufactures_its_own_credit.md](../../ideas/practice_manufactures_its_own_credit.md) | **Parent**: [`mjc/practice/etude/`](../mjc/practice/etude/README.md)

The étude's compile op — δ-silence certificate, selection rather than averaging, scoring under the consumption distribution — ported onto sculpting, which supplies the boundary-carrying-information the MuJoCo corridor lacked (a committed unit is a move program executed by a generator that reads the configuration) plus an exact DP oracle and a level-indexed action space. It **answers the étude's finding-7 question**: state-conditioned commitment (a library keyed by the observed target) beats state-independent commitment **1.8–3.0×, 6/6 commit states**, and averaging valid realisations destroys them **3.6–5.0×**, with the mechanism exact — the modal *token* span went off-grammar in half its blocks while every contributing realisation was fully on-grammar. It also **removes the certificate's job**: with the plant frozen and only the selector learning, committing at cycle 1 matches every gated arm at **26× less priced time**, and a shadow-compile instrument shows the committable content of practice traces is flat from cycle 1 while the closed-loop policy improves by 0.12–0.14. Interpretation (argued, not measured): **δ-silence gates compilation only where practice moves the executor** — the étude's practice trained the forward model its ballistic units bet on; this one trains the judge. A precheck confirms it and scopes the next round: let the plant learn and the committed-unit ceiling moves at **31–66× the metering noise floor**.

Round 2, [`practice/ratchet/`](practice/ratchet/README.md), gives practice something it can move — its own **action space** — by mining a level-indexed macro vocabulary from the agent's own successful repairs over a depth-laddered damage schedule: **earning the vocabulary recovers 68–98% of what being given the DGP's own buys**, at 0.85× the priced time and with a 4.1×-flatter cost-to-depth curve, while **committing one cycle early is worse than never committing** because the learned tables are nested (`T3 ⊆ T2 × T2`) so a compiled error forecloses the next level's *representation*, not just its accuracy; the unit-LP certificate matches its offline prediction at level 2 and correctly refuses at level 3, where a mis-levelled audition understates the unit by 1.75× — the étude's seam law with its sign flipped. (The plant stayed inert even with a manufactured frontier, so the descent is vocabulary-carried; note this node is `practice/ratchet/`, distinct from the meta-learning [`ratchet/`](ratchet/README.md) arc.)

Rounds 3–5 (2026-08-15→16): [`practice/ear/`](practice/ear/README.md) climbs the evaluation layer — the consumption-matched in-policy audition is calibrated to 0.97 (the seam law's third coordinate), yet every certificate arm still refuses level 3 (an arm holding level-2 vocabulary leaves a new unit no learning-progress headroom), and **provisional commitment at the era boundary rescues the frontier**, beating `given` with a recert safety net that never fires. [`practice/recital/`](practice/recital/README.md) removes the era clock: across three independent worlds, **no internal pacing signal prices what time at the bottom of the ladder buys** (certificate, task-progress and vocabulary-growth pacers fail for three distinct measured reasons) while a fixed bottom-heavy schedule is rank 1 everywhere and beats the DGP's own tables in both independent worlds; the mechanism check locates the deep grade's seed-stable predictor in the *next* level's mined table. [`practice/tall/`](practice/tall/README.md) attempts the depth-6 port to break the deepest-earnable-vs-grammar-ceiling confound: m=4 is measured inadmissible for sculpting, the main run is voided by value-head signal starvation at 64 tokens, and two regime-independent findings stand — endogenous pacers cannot traverse a ladder longer than the earnable range, and the mined table cannot churn (a priori).

## Next steps

*The ratchet / meta-learning arc's own retrospective and open threads — the MNIST–RHM gap, λ_local=0 stabilization, and the self-model-driven-grokking north star — now live in [`ratchet/README.md`](ratchet/README.md#open-threads-rehomed-from-the-parent-readme-next-steps).*

1. **Domain shift on RHM**: The stationary-data ratchet shows meta-learning dynamics without learning magnitude. The RHM's controllable DGP enables a clean domain shift test — e.g., train on one rule set then shift to new rules at the same (L, m). This would test whether the ratchet's dynamics produce genuine adaptation advantages, as seen in the MNIST OOD experiments. The sparse ratchet's non-compounding L3-L5 improvement (exhausted after cycle 1 on fixed data) motivates this especially — novel data would sustain the need for higher-level compositional learning.

2. **Scale model to learn higher m**: The m=4 model at 2.7M params can only compose 1-2 levels. A larger model that learns 3-4 levels at m=4 would produce a richer m-regime with higher absolute eta^2 values and more interpretable gate structure.

3. **Vocabulary as channel intervention**: Sweep v at fixed (L, m) to test the prediction that v changes beta (overall difficulty) but not gamma (scaling exponent shape). The RHM makes this a clean test — v changes the observation alphabet without changing the hierarchical composition structure.
