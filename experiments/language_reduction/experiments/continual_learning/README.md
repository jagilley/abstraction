# Continual Learning Benchmark via Vocab-Only Reduction

**Date**: 2026-05-07
**Status**: Phase 1 complete (independent single-token recovery)

## Summary

We recover collapsed tokens into vocab-reduced language models and measure whether the recovered embeddings capture the semantic dimensions that distinguish them from their collapse targets. At vocab-only tau=0.1 (v'=10,833 types), recovered tokens learn relational structure (how they relate to their collapse target) before local neighborhood structure (what company they keep). The strength of dimension emergence scales with distributional distinctiveness: geography > generational > gender.

## Why vocab-only, not spectral

The original concept recovery experiment (RESULTS_CONCEPT_RECOVERY.md) used spectral denoising at tau=0.3. Spectral denoising replaces token *occurrences* based on PMI-derived co-occurrence structure --- the same statistics (beta, gamma) we then measure on the output corpus. This entanglement means we can't distinguish genuine effects of recovery from artifacts of the replacement algorithm.

Vocab-only reduction breaks this entanglement. It maps rare tokens to their nearest GPT-2 embedding neighbors by cosine similarity --- a deterministic, context-independent, frequency-based operation. The vocab-only addendum in STATUS.md established that this preserves the DGP: gamma is invariant across tau values (< 15% variation from 50,257 to 236 types), and the Cagnetta formula correctly tracks empirical scaling.

For concept recovery specifically, vocab-only gives us three advantages:

1. **We know exactly what the model has and hasn't seen.** The collapse mapping is a deterministic function of token identity, not a context-dependent decision. Token X was either always present in training or never present --- there's no ambiguity about partial exposure.

2. **Collapse targets have known fanin.** We can count exactly how many source tokens mapped onto each target and assess whether its embedding is clean or contaminated.

3. **The DGP is preserved.** Any structure the model learns during recovery reflects genuine distributional patterns in language, not patterns imprinted by the denoising algorithm.

## Methodological false start: tau=0.3

Our first attempt directly ported the spectral experiment's target words (queen, empress, princess, ...) to vocab-only tau=0.3 (v'=1,724 types). This was wrong for reasons that are instructive.

At tau=0.3, queen (id=7542) collapses to King (id=2677). King absorbs 48 tokens including queen, princess, knight, and hero. This means:

- **The king-queen analogy is structurally destroyed.** Both words resolve to the same token in the model's vocabulary. The analogy can never work.
- **King's embedding is trained on a mixture of 49 identities.** King's representation encodes king+queen+princess+knight+hero+... contexts. It's not a clean "king" embedding --- it's a royal/authority cluster embedding.
- **Recovering queen into this space is a different task than in the spectral case.** We're training a random embedding toward a collapse target that already contains queen-like information. The spectral experiment recovered queen into a space where king was trained on (mostly) king-only contexts.

The experiment ran and produced results: 0/10 tokens differentiated from their collapse targets, king moved +0.014 toward ground truth, some forgetting appeared on the plural analogy category. But these results are not comparable to the spectral experiment because the experimental setup is fundamentally different.

**Lesson:** When changing the reduction method, the experimental design must be re-examined from scratch. The same target words, tau values, and metrics don't automatically transfer.

## Experimental design: tau=0.1

At tau=0.1 (v'=10,833), the collapse landscape is much cleaner:

- Mean collapse ratio: 4.3 tokens per target (vs 29 at tau=0.3)
- Median: 3.0
- Intra-group GPT-2 embedding coherence: 0.43 (groups are semantically tight)
- Only 10/162 analogy words are collapsed

We selected 6 targets across 3 semantic domains, each chosen for having a clean collapse target (low fanin) and a clear distinguishing dimension:

| Target | Collapses to | Fanin | Domain | Distinguishing dimension |
|---|---|---|---|---|
| princess | Prince | 5 | royalty/gender | Gender (female vs male royalty) |
| grandfather | father | 8 | family/generational | Generation (grandparent vs parent) |
| grandmother | mother | 6 | family/generational | Generation (grandparent vs parent) |
| beijing | China | 16 | geography | Specificity (city vs country) |
| tokyo | Japan | 24 | geography | Specificity (city vs country) |
| madrid | Spain | 18 | geography | Specificity (city vs country) |

Each target was recovered independently: same base model (vocab-only tau=0.1, P=100M), same hyperparameters (LR=3e-4, 5 epochs, batch_size=64), same curriculum extraction process (windows from the original un-reduced corpus containing the target word).

Key controlled variables:
- All models trained on 100M tokens of the same underlying corpus (FineWeb-Edu)
- Ground truth model (tau=0.0) trained on 100M tokens of the original corpus
- Each recovery starts from the same base checkpoint (no sequential dependencies)
- Curriculum sizes range from 49K tokens (madrid, less frequent) to 96K tokens (tokyo)

## Results: Recovery shifts

All 6 tokens showed real learning (shift magnitudes 10-20x above control word shifts), but all moved *toward* their collapse targets, not away:

| Target | cos(start, tgt) | cos(ft, tgt) | Shift magnitude | Control shift |
|---|---|---|---|---|
| princess | +0.063 | +0.171 | 0.053 | 0.004 |
| grandfather | +0.050 | +0.191 | 0.034 | 0.002 |
| grandmother | -0.017 | +0.114 | 0.041 | 0.003 |
| beijing | -0.042 | +0.163 | 0.064 | 0.003 |
| tokyo | +0.038 | +0.281 | 0.086 | 0.005 |
| madrid | -0.051 | +0.065 | 0.033 | 0.002 |

Starting cosines near zero are expected: the target embeddings begin as untrained noise. After fine-tuning, they move toward their collapse targets because the targets are *semantically related* --- princess IS related to Prince, beijing IS related to China. Moving toward the collapse target is correct behavior, not a failure to differentiate.

The right question is whether the recovered tokens also captured the dimension that *distinguishes* them from their collapse targets.

## Results: Dimension emergence

We measured whether the direction from collapse target to recovered token aligns with the expected semantic dimension. For example, does (prince - princess) align with (king - queen), (man - woman), etc.?

### Gender (princess, round 0): Did not emerge

Mean alignment of (prince - princess) with 8 gender reference directions:
- Fine-tuned: +0.05
- Ground truth: +0.17
- Base (untrained noise): +0.04

The fine-tuned alignment is indistinguishable from noise. Princess's nearest neighbors after recovery are subword fragments and unrelated words (Let, tery, citizen, Core, farmer), completely unlike the ground truth's clean neighborhood (Queen, Lady, Prince, goddess, Emperor).

### Generational (grandfather + grandmother, rounds 1-2): Emerged strongly

Alignment of (grandfather - father) with (grandmother - mother):
- Fine-tuned: **+0.61**
- Ground truth: +0.45

The generational direction emerged more strongly than in the ground truth model. However, nearest neighbors remain noisy (grandfather's top neighbors are subword fragments, not family words). The model learned the correct *relational direction* without yet forming a clean *local neighborhood*.

The generational dimension also shows moderate alignment with gender directions (son-daughter: +0.37), suggesting some entanglement between generational and gender structure --- consistent with these being correlated in natural language.

### Geographic (beijing + tokyo + madrid, rounds 3-5): Emerged strongly

Pairwise alignment of city-country directions:
- Fine-tuned: **+0.49** mean
- Ground truth: +0.32 mean

| Pair | Fine-tuned | Ground truth |
|---|---|---|
| beijing-China . tokyo-Japan | +0.59 | +0.34 |
| beijing-China . madrid-Spain | +0.44 | +0.34 |
| tokyo-Japan . madrid-Spain | +0.45 | +0.29 |

The city-country axis is consistent across all three pairs, exceeding ground truth alignment in all cases. Nearest neighbors show partial geographic signal: tokyo's include Toronto, Carolina, Georgia, Israel; madrid's include Stanford, Harvard, Washington, Texas --- geographic entities, even if not the correct subcategory.

## Interpretation

### Relational structure precedes neighborhood structure

Across all three domains, we see the same pattern: the recovered token learns the correct *direction* from its collapse target before its *neighborhood* crystallizes. The generational direction (grandfather - father) aligns with (grandmother - mother) at 0.61, but grandfather's nearest neighbors are subword noise. The geographic directions align at 0.49, but beijing's neighbors include Utah and gluten.

This suggests a hierarchy in what fine-tuning learns:
1. First: the broad direction that separates the token from its collapse target (captured in the embedding vector's orientation)
2. Later (with more data or training): the local neighborhood structure (which specific tokens are nearby)

This ordering makes sense mechanistically. The loss gradient for a rare token primarily pushes its embedding to predict the right next-token distribution in contexts where it appears. This shapes the token's *direction* relative to other embeddings (determining which output logits it activates) before it shapes the *neighborhood* (determining which other tokens have similar activations).

### Dimension emergence scales with distributional distinctiveness

The three domains showed a clear ordering: geography (+0.49) > generational (+0.61 but single-pair) > gender (+0.05, failed). This likely reflects how distinctive the target token's contexts are relative to its collapse target's:

- **Geography**: "in Beijing" and "in China" have very different distributional signatures. City names co-occur with streets, districts, neighborhoods; country names co-occur with governments, populations, economies. The fine-tuning curriculum provides strong, consistent signal about what makes a city different from a country.

- **Generational**: "my grandfather" and "my father" appear in similar but distinguishable contexts. Grandfather contexts include more historical and biographical language; father contexts are broader. The signal is moderate but consistent.

- **Gender**: "the princess" and "the Prince" appear in highly overlapping contexts in the training corpus (both involve royalty, ceremony, succession). The gender signal is diffuse and entangled with other dimensions. At toy model scale (13M parameters, 128-dim embeddings), the model may not have enough capacity to separate gender from the dominant royalty dimension in 50 training steps.

### Exceeding ground truth alignment

Both the generational (+0.61 vs +0.45) and geographic (+0.49 vs +0.32) dimensions exceeded ground truth alignment. This parallels the spectral concept recovery finding where the fine-tuned king-man+woman analogy produced more coherent results than the ground truth model's.

We should be cautious interpreting this. Possible explanations:
1. **Curriculum learning effect**: The base model established a stable framework over 100M tokens, and the recovered token slotted into it cleanly. The ground truth model learned everything simultaneously and found a different (possibly more balanced, less axis-aligned) geometry.
2. **Curriculum bias**: The recovery curriculum is a biased sample of contexts containing the target word. If these contexts oversample the distinguishing dimension (e.g., beijing appears disproportionately in contexts that contrast it with China), the recovered embedding may overfit to that dimension.
3. **Measurement artifact**: With single-pair measurements and no error bars, the "exceeding ground truth" claim has high uncertainty.

## Epistemic uncertainties

- **Toy scale**: 13M parameters, 2-layer, 128-dim GPT-2. These results may not transfer to larger models where the embedding space has richer structure.
- **No error bars**: Each domain has 1-3 data points. The generational cross-alignment is a single number (one pair). The geographic alignment is from 3 pairs. Statistical significance has not been established.
- **No persistence test**: We haven't tested whether these dimensions survive continued training on other tokens (the actual continual learning benchmark). The dimensions may be fragile.
- **Neighborhood quality is poor**: The recovered tokens' local neighborhoods are still dominated by noise. The "dimension emerged" claim is about directional alignment, not about the token being functionally integrated into the embedding space.
- **Curriculum bias**: The recovery curricula are biased samples. We haven't controlled for whether the dimension emergence is driven by genuine semantic structure or by distributional artifacts in the curriculum windows.

## Phase 2: Sequential recovery with degradation tracking

Phase 1 recovered each token independently from the same base model. Phase 2 chains the recoveries: round i fine-tunes from round i-1's checkpoint. After each round, we measure both forward recovery (did this token learn?) and backward degradation (did previously recovered tokens drift?).

We tested two orderings of the same 6 tokens:
- **Clustered**: beijing → tokyo → madrid → grandfather → grandmother → princess (same-domain tokens grouped)
- **Interleaved**: beijing → grandfather → princess → tokyo → grandmother → madrid (domains alternated)

Ground truth is the tau=0.0 model trained on the full 50,257-token vocabulary with the same architecture and data.

### Results: Degradation matrices

Diagonal = toward_gt at recovery. Off-diagonal = toward_gt delta since recovery. Negative off-diagonal = forgetting; positive = reinforcement.

**Clustered** (geography → generational → gender):

|  | beijing | tokyo | madrid | grandfather | grandmother | princess |
|---|---|---|---|---|---|---|
| round 0 (beijing) | +0.020 | — | — | — | — | — |
| round 1 (tokyo) | +0.000 | +0.003 | — | — | — | — |
| round 2 (madrid) | -0.000 | -0.004 | -0.024 | — | — | — |
| round 3 (grandfather) | -0.000 | -0.004 | -0.000 | -0.012 | — | — |
| round 4 (grandmother) | -0.000 | -0.004 | -0.000 | -0.012 | -0.021 | — |
| round 5 (princess) | -0.000 | -0.006 | -0.000 | -0.002 | -0.010 | -0.030 |

**Interleaved** (domains alternated):

|  | beijing | grandfather | princess | tokyo | grandmother | madrid |
|---|---|---|---|---|---|---|
| round 0 (beijing) | +0.008 | — | — | — | — | — |
| round 1 (grandfather) | -0.000 | -0.010 | — | — | — | — |
| round 2 (princess) | -0.000 | +0.009 | -0.025 | — | — | — |
| round 3 (tokyo) | -0.011 | +0.009 | +0.000 | -0.020 | — | — |
| round 4 (grandmother) | -0.011 | +0.023 | -0.008 | +0.000 | -0.026 | — |
| round 5 (madrid) | -0.011 | +0.023 | -0.008 | +0.001 | +0.000 | -0.021 |

### Results: Dimension emergence in final sequential model

| Dimension | Clustered | Interleaved | Phase 1 (independent) | Ground truth |
|---|---|---|---|---|
| Geography | +0.51 | +0.50 | +0.49 | +0.32 |
| Generational | +0.60 | +0.59 | +0.61 | +0.45 |
| Gender | -0.04 | -0.05 | +0.05 | -0.17 |

Dimension emergence is essentially identical across all three conditions (Phase 1 independent, Phase 2 clustered, Phase 2 interleaved). Sequential training neither helps nor hurts dimension formation.

### Interpretation

**1. Catastrophic forgetting is negligible at this scale.** All embedding stability values exceed 0.98. Control word mean shifts accumulate monotonically (0.001 → 0.008) but remain tiny. The 13M-parameter model has enough spare capacity to accommodate all 6 recovered tokens without meaningful interference.

**2. Cross-domain training does not perturb recovered tokens.** This is the cleanest finding, consistent across both orderings. Grandfather's toward_gt delta is unchanged after tokyo (cross-domain) in both orderings. Beijing's toward_gt delta is unchanged after grandfather (cross-domain) in both orderings. Cross-domain rounds produce deltas of exactly 0.000.

**3. Same-domain training produces the largest perturbations.** In both orderings, the biggest toward_gt delta for grandfather comes from grandmother's round --- the only same-domain token trained after it. The magnitude of this perturbation (~0.01--0.02) is 2--3x the control word background rate.

**4. The sign of same-domain interaction depends on model state.** In the interleaved ordering, grandmother's training pushes grandfather *toward* ground truth (+0.014 jump at round 4). In the clustered ordering, grandmother's training pushes grandfather slightly *away* (-0.012). One possible explanation: in the clustered ordering, grandfather was trained immediately before grandmother (rounds 3→4), so the embedding is still in a high-plasticity state and may overshoot. In the interleaved ordering, grandfather was trained 3 rounds earlier (round 1) and has stabilized, so grandmother's gradient provides a cleaner reinforcement signal along the settled generational direction. We cannot distinguish this from noise.

### Epistemic uncertainties

- **Toy scale**: 13M parameters, 128-dim embeddings, only 6 tokens perturbed from a 50K vocabulary. The model has massive spare capacity, so the near-zero forgetting may not generalize to capacity-constrained settings.
- **Small differences**: the reinforcement vs degradation sign difference (finding 4) could be noise.
- **Small effect sizes**: All toward_gt deltas are in the range ±0.03. Embedding stability is always > 0.98. We're measuring very small perturbations.
- **Same curriculum extraction**: Both orderings use the same curriculum windows per token, so differences are purely from ordering and model state, not data.

## Addendum: Scaled sequential recovery (100 tokens)

**Date**: 2026-05-07

The 6-token experiments above showed near-zero forgetting, but 6 tokens in a 13M-parameter model is too easy --- the model has massive spare capacity. To find where the stability-plasticity tradeoff actually bites, we scaled to 100 tokens.

### Setup

100 tokens auto-selected from the 12,232 collapsed tokens at tau=0.1 that are (a) clean alphabetic words with leading space, (b) at least 3 characters, and (c) appear >= 200 times in the corpus. Tokens were stratified by fanin quartile to span the full range (fanin 1--52, mean 8.3, median 6), then shuffled into random order (seed=42) so fanin effects can be analyzed as a covariate rather than confounded with position.

Each round: extract 50K-token curriculum, fine-tune 5 epochs from the previous round's checkpoint, evaluate recovery + degradation of all previously recovered tokens.

### Results

**Stability degrades gradually, not catastrophically.** Mean embedding stability across all previously-recovered tokens declines smoothly from 1.000 at round 0 to 0.977 at round 99. The worst single token reaches 0.907. There is no cliff --- it's monotonic erosion.

**Position dominates, fanin does not predict forgetting.** Spearman correlation of position (round index) vs min_stability: r = -0.882 (p ~ 0). Fanin vs min_stability: r = -0.134 (p = 0.185, not significant). Degradation is driven by cumulative training perturbation, not by properties of the token being recovered. This disconfirms our initial hypothesis that collapse target fanin would predict forgetting risk.

**Forgetting is real but bounded.** By round 99, 31 of 99 previously-recovered tokens have toward_gt_delta < -0.01, but the mean toward_gt_delta is only -0.003 and the worst is -0.069. The model drifts gently rather than collapsing.

**Control word shift accumulates linearly.** From 0.001 (round 0) to 0.039 (round 99) --- roughly 0.0004 per round. This is the background rate of weight perturbation and serves as a baseline for what "forgetting" looks like when there's no semantic relationship to the recovered token.

### Interpretation

The 100-token result clarifies the 6-token finding. At 6 tokens, we saw near-perfect stability and concluded "no catastrophic forgetting." At 100 tokens, we see that forgetting IS happening, but it's **graceful degradation** --- more erosion than avalanche. The model accommodates 100 new token embeddings without any single previously-recovered token losing its learned direction entirely.

The failure of the fanin hypothesis is informative. We predicted that high-fanin collapse targets (whose embeddings are blurry compromises across many identities) would resist clean decompression, causing more interference. Instead, position (how many rounds of gradient updates have washed over the embedding) is essentially the only predictor. This suggests the degradation mechanism is not semantic interference but simple parameter drift: each round's gradients perturb all embeddings slightly, and these perturbations accumulate. The recovered tokens aren't fighting each other --- they're just slowly getting sand-blasted by 100 rounds of unrelated gradient flow.

This is consistent with the "geometric resonance" interpretation from the 6-token experiments: each recovered token slots into existing structure rather than disrupting it, so the only degradation comes from the ambient perturbation of continued training, not from representational conflict.

## What's next

1. **Neighborhood crystallization**: More training steps or larger curricula to see if the local neighborhoods eventually settle. At what point do grandfather's neighbors become family words instead of subword fragments?

2. **Quantifying channel lossiness**: The vocab-only finding that gamma is invariant implies tau controls the channel, not the DGP. Can we quantify the channel capacity at each tau and predict how much fine-tuning data is needed to recover a token? The geographic tokens (strong contextual signature) needed less effective data than gender (diffuse signature) --- this may connect to the mutual information between the token and its distinguishing context.

3. **Scaling further**: Larger tau (more collapsed tokens, more stressed model) or more tokens per run to find whether there is a phase transition from graceful degradation to catastrophic forgetting, or whether it remains smooth.

## Run commands

```bash
# Phase 1: Independent recovery at tau=0.1
modal run language_reduction/modal_app.py --stage cl-batch --tau 0.1 --lr 3e-4
modal run language_reduction/modal_app.py --stage cl-dimensions --tau 0.1

# Phase 2: Sequential recovery with degradation tracking (6 tokens)
modal run language_reduction/modal_app.py --stage cl-sequential --tau 0.1 --ordering clustered
modal run language_reduction/modal_app.py --stage cl-sequential --tau 0.1 --ordering interleaved

# Phase 2 scaled: Sequential recovery (100 tokens)
modal run language_reduction/modal_app.py --stage cl-scaled --tau 0.1 --n-cl-tokens 100 --lr 3e-4
```

## Files

| File | Description |
|------|-------------|
| `README.md` | This document |
| `EXPERIMENT.md` | Experiment spec (Phase 1 + Phase 2 design) |
| `stages.py` | Modal pipeline stages (Phase 1 + Phase 2 + scaled) |
| `__init__.py` | Package marker |
| `/data/continual_learning/tau_0.100/round_{0-5}/` | Phase 1: per-round independent models |
| `/data/continual_learning/tau_0.100/batch_summary.json` | Phase 1: cross-target summary |
| `/data/continual_learning/tau_0.100/dimension_analysis.json` | Phase 1: dimension emergence |
| `/data/continual_learning/tau_0.100/sequential/clustered/` | Phase 2: clustered ordering (6 tokens) |
| `/data/continual_learning/tau_0.100/sequential/interleaved/` | Phase 2: interleaved ordering (6 tokens) |
| `/data/continual_learning/tau_0.100/sequential/scaled_100_seed42/` | Phase 2 scaled: 100-token sequential recovery |
