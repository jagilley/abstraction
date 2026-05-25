# Results: GLP Residual Decomposition at Female-Royalty Positions

**Date**: 2026-04-30
**Builds on**: `RESULTS_SCAFFOLDING_PERSISTENCE.md`, `RESULTS_CONCEPT_RECOVERY.md`
**Status**: Complete

## Question

The scaffolding persistence experiment showed that SFT on queen-containing text produces structural changes that survive even when the queen token is zeroed or suppressed. But what does the model's internal geometry look like *before* scaffolding, at positions where queen should appear?

Specifically: does the model have any representation of "female royalty" prior to seeing the concept, and if so, what form does it take? Is it a clean direction, a diffuse cloud, or something structured?

## Setup

We trained a GLP (generative latent prior) on the τ=0.3, P=100M model's all-layer activations. The GLP is a 1.84M-parameter flow-matching denoiser operating on 256-D feature vectors (the concatenation of both transformer layer outputs, [h0; h1]). It learns the manifold of "typical" activations from 1M token positions, achieving 0.962 reconstruction cosine similarity.

For any activation, the GLP residual r = feature - manifold_projection captures what is "off-manifold" — the component that deviates from typical. We computed these residuals at the last token position of 48 probe prompts across four semantic categories:

- **Queen** (12 prompts): contexts where female royalty should be predicted ("The king and his wife, the", "She wore a crown and ruled the kingdom as its", etc.)
- **Gender** (12 prompts): female/feminine contexts without royalty ("The mother cooked dinner for her", "She was a young woman who worked", etc.)
- **Royalty** (12 prompts): monarchical contexts without gender specificity ("The king ruled his kingdom with", "The prince inherited the throne from his", etc.)
- **Control** (12 prompts): neutral contexts ("The sun rises in the east and", "The farmer planted seeds in the", etc.)

Residuals were averaged over 5 stochastic denoising passes to stabilize direction.

## Result 1: Linear decode is noise

Projecting residuals through the model's own unembedding (h1 residual → ln_f → lm_head) produces incoherent token lists: `mosquito`, `Commercial`, `biking`, `Depression`, etc. No prompt category produces interpretable decoded tokens.

This confirms the prediction from `beliefs/nonlinear_legibility.md`: the GLP's semantic content lives in its nonlinear geometry, not in linearly extractable directions. The unembedding is the wrong tool.

## Result 2: The model predicts male royalty where queen should be

The model's actual next-token predictions at queen positions are coherent but systematically wrong:

| Prompt | Top predictions |
|--------|-----------------|
| The king and his wife, the | son, king, family, father, man |
| She wore a crown and ruled the kingdom as its | military, man, first, the, father |
| The most powerful woman in the kingdom was the | first, the, city, man, mother |
| In the royal court, the king and his | his, who, wife, father, son |

The model has rich representations of male royalty (king, father, son, Emperor) and some gender concepts (wife, mother, daughter), but fills the female-royalty gap with male terms or function words. This asymmetry — male royalty present, female royalty absent — is the entire motivation for the scaffolding experiment.

## Result 3: Queen residuals decompose into gender x royalty

The mean residual direction for each category shows that queen sits at the intersection of gender and royalty:

| | → control | → gender | → royalty |
|---|---|---|---|
| **queen direction** | +0.287 | **+0.382** | **+0.449** |
| **gender direction** | +0.284 | — | +0.250 |
| **royalty direction** | +0.374 | +0.250 | — |

Queen projects onto gender (+0.382) and royalty (+0.449), and both projections exceed the baseline gender-royalty cosine (+0.250). The queen direction isn't just "gender" or just "royalty" — it's their compositional overlap.

## Result 4: Per-prompt projections are semantically interpretable

This is the strongest signal. Individual queen-prompt residuals project onto the gender and royalty axes in proportions that match the prompt content:

| Prompt | → gender | → royalty | Interpretation |
|--------|----------|----------|----------------|
| The prince married a woman of noble birth, and she became the | **+0.390** | +0.081 | Marriage/social position — gender dominates |
| She was a powerful woman who ruled | **+0.160** | +0.053 | Feminine agency — gender dominates |
| The throne was inherited by the eldest daughter of the | **+0.348** | **+0.353** | Hereditary female succession — both axes |
| The female ruler of the kingdom | +0.051 | **+0.206** | Governance framing — royalty dominates |
| In ancient times, kings and their consorts would | -0.010 | **+0.206** | Historical monarchy — royalty only |
| In the royal court, the king and his | +0.157 | **+0.231** | Court context — royalty dominates |

The decomposition is not noise. Prompts about women's social roles load on gender; prompts about thrones and courts load on royalty; prompts about female succession load on both. The per-prompt pattern is what you'd write down from reading the prompts, recovered from the 256-D GLP residual geometry.

## Result 5: Category-level alignment confirms the structure

Within-category alignment (mean pairwise cosine of residuals):

| Category | Alignment |
|----------|-----------|
| Royalty | 0.133 |
| Gender | 0.115 |
| Queen | 0.081 |
| Control | 0.048 |

Gender and royalty are the model's most coherent internal categories — it knows these concepts individually. Queen (0.081) is between them and control (0.048). It borrows structure from both without having its own dedicated direction.

Cross-category projections confirm queen's dual affinity:

| Source → Target | Mean cosine |
|-----------------|-------------|
| queen → gender | +0.151 |
| royalty → gender | +0.112 |
| control → gender | +0.101 |
| queen → royalty | +0.181 |
| gender → royalty | +0.109 |
| control → royalty | +0.131 |

Queen has the highest affinity to gender among non-gender categories, and the highest affinity to royalty among non-royalty categories. It is uniquely positioned at the intersection.

## Result 6: The signal is weak but real

The gender-royalty 2D subspace explains 2.5% of queen residual variance, versus 0.8% expected by chance (3.1x). The within-category alignment for queen (0.081) is modest in absolute terms. Residual norms show queen positions are only 1.1x farther from the manifold than controls.

This is what you'd expect from a dramatically underparameterized model (2 layers, 128-dim, ~13M params). The model doesn't have capacity for clean, factored representations. The queen concept exists as a diffuse entanglement of gender and royalty signals, spread across ~10 effective dimensions (effective rank of queen residual subspace: 10.4), not concentrated in a single direction.

## Result 7: Velocity field curvature shows no ripeness signal

Velocity field curvature ratio (how much the GLP's nonlinear response curves in the residual direction vs. random directions) was below 1.0 for all categories (queen: 0.75, control: 0.76). The manifold doesn't have pronounced nonlinear structure along the residual direction at these positions.

Combined with the weak-but-real linear signal, the picture is: the model has a diffuse, partially-structured representation of the gender×royalty intersection, but it hasn't crystallized into the kind of sharp geometric feature that would show up as velocity field curvature. The concept exists as entanglement, not as form.

## Interpretation: scaffolding as composition, not injection

The pre-scaffolding model has gender. It has royalty. It doesn't have their intersection as a first-class concept. At positions where queen should appear, the GLP residual reflects an entangled mix of both components, varying interpretably with prompt content.

This reframes what SFT on queen contexts does: it doesn't inject a new concept from scratch, and it doesn't surface a suppressed one. It teaches the model to *compose* two concepts it already has. The queen token provides a backprop handle that lets gradients flow through both the gender and royalty subspaces simultaneously, binding them into a single representational unit.

The persistence result then says: once the model learns this composition, the binding survives even when the scaffold token is removed. The model learned to intersect the subspaces, not just memorize a surface form.

## What this predicts for the post-SFT GLP comparison

If this interpretation is correct, training a GLP on the post-SFT model and repeating this analysis should show:
1. **Queen residuals shrink** (come on-manifold) — the concept is no longer unusual
2. **The gender×royalty entanglement disentangles** — queen becomes its own direction, not a mixture
3. **Velocity field curvature increases** in the queen direction — the manifold now has structure there
4. **The gender and royalty categories should be largely unaffected** — scaffolding creates a new concept at their intersection without disturbing the components

## Implementation

- GLP training: `modal run language_reduction/modal_app.py --stage train-glp --tau 0.3`
- Semantics analysis: `modal run language_reduction/modal_app.py --stage glp-semantics --tau 0.3`
- Code: `language_reduction/glp.py` (model, training, inference), stages in `modal_app.py`
- Raw results: `/data/results/glp_semantics_tau0.3.json` on Modal volume
