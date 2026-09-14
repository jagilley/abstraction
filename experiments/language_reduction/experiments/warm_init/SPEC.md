# Experiment: GLP-Guided Warm Initialization of Concept Tokens

**Date**: 2026-05-01
**Builds on**: `RESULTS_RESIDUAL_DECOMPOSITION.md`, `RESULTS_CONCEPT_RECOVERY.md`
**Motivation**: `conversations/Claude-Learned token invention in language models.md`

## Question

Can the GLP's geometric understanding of where "queen" should live in activation space be used to initialize the queen token's embedding and output projection, bypassing the need for from-scratch SFT?

The residual decomposition showed that at queen-prediction positions, the model's activations sit at a gender x royalty intersection with interpretable per-prompt structure. The concept recovery experiment showed SFT on 100K tokens (50 steps, LR=3e-4) successfully integrates queen from random initialization. If GLP-guided initialization can replicate some or all of that integration zero-shot, it validates the "activation-to-embedding meta-model" idea from the token invention proposal.

## Design

### Independent variable: initialization of queen's wte and lm_head rows

All three conditions start from the same tau=0.3 baseline model. Only the queen token's rows differ.

| Condition | `wte[queen]` | `lm_head[queen]` |
|-----------|-------------|-------------------|
| `random` | N(0, 0.02) (original random init) | N(0, 0.02) |
| `analogy` | king + (mean_female_emb - mean_male_emb) | king_lm + (mean_female_lm - mean_male_lm) |
| `contrastive` | same as analogy | centroid(queen_h1) − centroid(control_h1), scaled to mean lm_head row norm |

Gender words: she, her, woman, mother, daughter, wife, girl, sister
Male words: he, his, man, father, son, husband, boy, brother

The contrastive lm_head subtracts the control centroid to remove the shared "average English activation" pattern, isolating the direction specific to queen-prediction contexts. The `contrastive` and `analogy` conditions share the same wte, so any difference is attributable to the lm_head initialization.

Note: an earlier version used the raw centroid (without control subtraction). This achieved rank 1 at queen prompts but also fired at 9/12 control prompts — it was a global bias, not a discriminative detector. The contrastive version fixes this. A GLP residual variant (using the h1 component of the mean GLP residual directly) was also tested but proved too weak (1/12 top-10 at queen prompts).

### Evaluation

**Zero-shot** (no SFT):
- Queen rank at last position of 12 queen-prediction prompts and 12 control prompts (specificity check)
- Greedy generation from 4 prompts containing "queen" as input token

**SFT convergence**:
- Same curriculum, LR, epochs as concept recovery (100K tokens, LR=3e-4, 5 epochs)
- Val loss and queen mean rank tracked every 2 steps
- Random seed fixed across conditions for identical batch order

**Post-SFT**:
- Full queen rank on all 12 prompts
- Generation from all 4 input prompts
- Step at which queen first enters top-10 (convergence speed metric)

### Predictions

- `random` zero-shot: queen rank ~ vocab_size/2 (noise floor)
- `analogy` zero-shot: queen rank moderately improved (embedding arithmetic provides rough neighborhood)
- `contrastive` zero-shot: queen rank substantially improved (contrastive centroid targets queen-specific activation patterns)
- SFT convergence: contrastive >= analogy > random (warm init needs fewer steps)
- Post-SFT: all three should converge to similar quality (SFT is the same; init only affects speed)

## Run

```
modal run language_reduction/modal_app.py --stage warm-init --tau 0.3
```

## Files

| File | Description |
|------|-------------|
| `stages.py` | Modal stage implementation |
| `SPEC.md` | This document |
