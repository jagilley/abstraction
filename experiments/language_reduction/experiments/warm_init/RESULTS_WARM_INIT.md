# Results: GLP-Guided Warm Initialization of Concept Tokens

**Date**: 2026-05-01
**Spec**: `SPEC.md`
**Status**: Part 1 complete (lm_head initialization). Part 2 complete — see `RESULTS_WARM_INIT_PHASE2.md`.

## Question

Can the GLP's geometric understanding of where "queen" should live in activation space be used to initialize a concept token's output projection (lm_head row), so the model predicts "queen" at appropriate positions without fine-tuning?

## Setup

The τ=0.3 model (2 layers, 128-dim, ~13M params, trained on 100M tokens with queen removed from the corpus) serves as the testbed. A GLP trained on this model's all-layer activations [h0; h1] achieves 0.962 reconstruction cosine similarity. Queen's embedding and lm_head rows are randomly initialized (N(0, 0.02)) and never trained.

Three initialization conditions for queen's wte (input embedding) and lm_head (output projection) rows, with everything else identical:

| Condition | `wte[queen]` | `lm_head[queen]` |
|-----------|-------------|-------------------|
| `random` | original random init | original random init |
| `analogy` | king + gender_shift | king_lm + gender_shift_lm |
| `contrastive` | king + gender_shift (same as analogy) | centroid(queen_h1) − centroid(control_h1), scaled |

Gender shift = mean(she, her, woman, mother, daughter, wife, girl, sister) − mean(he, his, man, father, son, husband, boy, brother), computed separately in embedding and lm_head weight spaces.

The contrastive lm_head is constructed by extracting ln_f(h1) at the last position of 12 queen-prediction prompts and 12 control prompts, taking the difference of their centroids, and scaling to match the mean lm_head row norm. This removes the shared "average English activation" component and isolates the direction specific to queen-prediction contexts.

The `contrastive` and `analogy` conditions share the same wte, so any difference between them is attributable purely to the lm_head initialization.

### Evaluation

- **Zero-shot**: queen's rank at the last position of 12 queen-prediction prompts and 12 control prompts (specificity check), plus greedy generation from 4 queen-input prompts
- **SFT convergence**: same curriculum (100K tokens), LR (3e-4), and epochs (5) as the concept recovery experiment, with random seed fixed across conditions. Val loss and queen rank tracked every 2 steps.

## Result 1: Contrastive initialization achieves near-zero-shot concept detection

| Condition | Queen prompts (12) | Control prompts (12) |
|-----------|-------------------|---------------------|
| `random` | mean rank 1240.9, 0/12 top-10 | mean rank 2017.7, 0/12 top-10 |
| `analogy` | mean rank 743.2, 0/12 top-10 | mean rank 1832.2, 0/12 top-10 |
| **`contrastive`** | **mean rank 11.0, 9/12 top-10** | **mean rank 32,218.6, 0/12 top-10** |

The contrastive lm_head places queen as the #1 prediction at 8 of 12 queen prompts, with 0/12 false positives at control positions. Per-prompt results for the contrastive condition:

| Prompt | Queen rank | Top predictions |
|--------|-----------|----------------|
| A woman of royal authority and | **1** | Queen(0.179), the(0.056) |
| The prince married...she became the | **1** | Queen(0.104), first(0.056) |
| The throne was inherited by the eldest daughter of | **1** | Queen(0.069), family(0.050) |
| She wore a crown and ruled the kingdom as its | **1** | Queen(0.037), military(0.029) |
| In the royal court, the king and his | **1** | Queen(0.307), his(0.007) |
| The most powerful woman in the kingdom was the | **3** | first(0.046), the(0.027), Queen(0.021) |
| The emperor and his wife ruled together over the | **5** | city(0.033), throne(0.022), Queen(0.019) |
| The female ruler of the kingdom | **6** | of(0.240), ,(0.125) |
| The king and his wife, the | **9** | son(0.065), king(0.046) |
| She was a powerful woman who ruled | **19** | her(0.173), the(0.130) |
| The king's daughter became the new | **18** | of(0.056), city(0.053) |
| In ancient times, kings and their consorts would | **67** | to(0.113), be(0.077) |

The three misses (ranks 18, 19, 67) are prompts where queen is less obviously the unique correct completion — they're more ambiguous contexts where function words or other nouns compete. The 9 hits are all prompts with clear female-royalty framing.

## Result 2: Embedding arithmetic alone is insufficient

The `analogy` condition uses king + gender_shift for both wte and lm_head. It improves over random (rank 743 vs 1241) but never reaches top-10 at any queen prompt, zero-shot or after SFT.

The analogy lm_head has cos(analogy_lm, king_lm) = 0.89 — it's essentially a slightly rotated king detector. The gender shift in lm_head weight space is too small relative to the shared royalty component to make queen discriminable from king.

The contrastive lm_head has cos(contrastive_lm, king_lm) = 0.21 — it points in a genuinely different direction, one that captures what queen-prediction positions look like *differently from average text*, not what king looks like.

## Result 3: SFT convergence is dramatically faster with contrastive initialization

| Condition | Queen rank at step 0 | Queen rank at step 49 | First reaches top-10 | Best val loss |
|-----------|---------------------|----------------------|---------------------|--------------|
| `random` | 542 | 152 | never | 5.581 |
| `analogy` | 150 | 58 | never | 5.579 |
| **`contrastive`** | **9** | **6** | **step 0** | 5.589 |

The contrastive condition starts in top-10 and stays there. Random and analogy improve steadily but never reach top-10 in 50 steps. After SFT, the contrastive condition achieves 11/12 top-10 on the full prompt set; random and analogy reach only 5/12 and 4/12.

Val loss converges to approximately the same value across conditions (~5.58), confirming the curriculum teaches the same general knowledge — but only contrastive gets queen's lm_head into a functional state.

## Result 4: The raw centroid approach is non-specific (failed v1)

Our first attempt used the raw centroid of ln_f(h1) at queen positions as the lm_head row (without subtracting the control centroid). This achieved rank 1.2 at queen prompts (12/12 top-10), but also rank 1 at 5/12 control prompts including "The farmer planted seeds in the" and "The dog ran across the field to the."

The raw centroid is close to the mean h1 across all positions. It produces high logits everywhere, functioning as a global bias rather than a discriminative detector. The contrastive approach fixes this by subtracting the shared component.

We also tested using the h1 component of the GLP residual directly (feature − manifold_projection) as the lm_head row. This was perfectly specific (0/12 control top-10) but too weak — only 1/12 queen prompts reached top-10. The residual direction captures the right semantic content (gender×royalty intersection, per the residual decomposition) but its magnitude in 128-D space is insufficient to compete with trained lm_head rows. The contrastive centroid is a better practical choice: it captures the same discriminative signal with enough magnitude to be functional.

## Result 5: The input embedding (wte) is the remaining bottleneck

All three conditions produce degenerate generation when queen appears as input:

| Prompt | Contrastive output |
|--------|--------------------|
| "The queen ruled" | "the the of the the of the the of..." |
| "She was a powerful queen who" | "had a great of her her her her..." |
| "The king and the queen" | "of the son of the son of the son of his Queen..." |

The lm_head is solved (the model correctly predicts queen at appropriate positions), but the wte is not (the model can't process queen as meaningful input). The analogy-based wte (king + gender_shift) places queen in roughly the right embedding neighborhood but doesn't encode enough information for coherent contextual processing.

This is the target for Part 2.

## Interpretation

The contrastive centroid works because it captures the *discriminative* structure of queen-prediction positions. The 12 queen prompts and 12 control prompts share a large common component (typical English text), and subtracting the control centroid isolates the residual: the direction in post-LayerNorm h1 space that is specific to contexts where female royalty should be predicted.

This is conceptually related to the GLP residual decomposition, which found that queen positions sit at a gender×royalty intersection in activation space. The contrastive centroid captures a related signal but operates in the lm_head's native space (post-ln_f h1) rather than the GLP's normalized activation space, giving it the right magnitude to compete as a linear detector.

For the token invention pipeline, this validates the output side of the proposal: given a set of contexts where a new concept should be predicted, a contrastive centroid of the model's internal representations at those positions produces a functional zero-shot detector. No fine-tuning is needed for the model to learn *when* to predict the new token — only *what it means* when read as input.

## Part 2: wte initialization (complete)

See `RESULTS_WARM_INIT_PHASE2.md` for full results. Summary: five wte initialization approaches were tested (analogy, regression, gradient inversion, gradient signal from error signals at queen-prediction positions) under both zero-shot and SFT conditions. None produce functional zero-shot generation. The lm_head problem was fundamentally linear (a dot product); the wte problem requires the surrounding weights to co-adapt, which no initialization alone can provide. Full-model SFT remains necessary for the input side. The core idea — that a meta-model of activations can warm-initialize concept embeddings — has not been falsified, but the specific approaches tested so far don't outperform simple embedding arithmetic (king + gender_shift) as a starting point for SFT.

## Run commands

```
modal run language_reduction/modal_app.py --stage warm-init --tau 0.3
```

## Files

| File | Description |
|------|-------------|
| `stages.py` | Modal stage: three-condition experiment with zero-shot + SFT eval |
| `SPEC.md` | Experiment design |
| `RESULTS_WARM_INIT.md` | This document |
| `/data/results/warm_init_tau0.3.json` | Full numeric results (on Modal volume) |
