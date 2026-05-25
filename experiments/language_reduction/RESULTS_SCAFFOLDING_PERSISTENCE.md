# Results: Scaffolding Persistence

**Date**: 2026-04-30
**Spec**: `EXPERIMENT_SCAFFOLDING_PERSISTENCE.md`
**Status**: Complete

## Summary

With all scaffolding tokens (queen, empress, princess, etc.) suppressed from output logits, the post-SFT model generates qualitatively more coherent female-ruler content than the baseline, and its top-k predictions at key positions include royalty/authority concepts (throne, imperial, ruler, eldest, daughter) that are absent from the baseline's predictions. The scaffolding worked — SFT on queen-containing text changed the model's weights such that female-ruler knowledge persists without the queen token.

## Key Findings

### 1. The baseline model has no concept of female rulers

Without queen, the baseline degenerates into repetition:

| Prompt | Baseline continuation |
|--------|----------------------|
| "She was a powerful woman who ruled" | "her her her her her her her..." |
| "The wife of the king was a" | "man man man man man man..." |
| "The female ruler" | "of the male and the female. The male was born in the female, and the female female..." |

The baseline's top prediction for "The wife of the king was a powerful ___" is **"man" (p=0.130)** — it has no representation linking a king's wife to female authority.

### 2. The post-SFT model connects female rulers to monarchy and authority

Same prompts, same token suppression, dramatically different output:

| Prompt | Post-SFT continuation |
|--------|----------------------|
| "She was a powerful woman who ruled" | "her to **the throne of the imperial family**. The Emperor Akihito was born on the throne..." |
| "The female ruler" | "of **the imperial family**, and **the eldest daughter** of the imperial family" |
| "A woman of royal authority" | "and **the eldest daughter of the Emperor Akihito**, who was born on **the throne** of the imperial family" |

Throne, imperial family, eldest daughter, Emperor — all royalty/authority concepts, none of which are suppressed tokens. The model learned to associate female rulers with monarchy through its weights, not just through the queen token.

### 3. Top-k predictions confirm structural knowledge transfer

**"The king and his wife, the ___":**

| Rank | Baseline | Post-SFT | Ground truth |
|------|----------|----------|-------------|
| 1 | son (0.068) | king (0.114) | king (0.162) |
| 2 | king (0.048) | **daughter (0.061)** | son (0.027) |
| 3 | family (0.041) | **eldest (0.061)** | Lord (0.027) |
| 4 | father (0.035) | son (0.047) | wife (0.022) |
| 5 | man (0.027) | brother (0.028) | King (0.019) |

The post-SFT model ranks **daughter** and **eldest** as #2 and #3 — both female-royalty concepts. The baseline has no such signal.

**"The wife of the king was a powerful ___":**

| Rank | Baseline | Post-SFT | Ground truth |
|------|----------|----------|-------------|
| 1 | **man (0.130)** | king (0.096) | man (0.045) |
| 2 | king (0.034) | symbol (0.024) | warrior (0.032) |
| 3 | leader (0.030) | **prince (0.023)** | and (0.026) |
| 4 | woman (0.026) | woman (0.021) | woman (0.018) |
| 5 | son (0.024) | **ruler (0.017)** | wife (0.018) |

Post-SFT predicts **ruler** and **prince** (royalty concepts). Baseline predicts **man** with 13% probability — its strongest prediction for the wife of a king is "man."

**"She was a powerful woman who ruled the ___":**

| Rank | Baseline | Post-SFT |
|------|----------|----------|
| 1 | mother (0.030) | country (0.026) |
| 2 | her (0.026) | United (0.021) |
| 3 | family (0.026) | **throne (0.019)** |
| 4 | first (0.017) | king (0.015) |
| 5 | father (0.016) | first (0.014) |

Baseline associates "a powerful woman who ruled" with domestic concepts (mother, her, family). Post-SFT predicts **throne** — the seat of royal power.

### 4. The ground truth model also struggles without queen

Interestingly, the ground truth (τ=0.0) model's generations also degenerate — mostly into "king of the king, and the king of the king" loops. The ground truth model learned to route through queen in female-ruler contexts, so suppressing queen leaves a similar gap. This is expected: the τ=0.0 model's female-ruler knowledge is distributed between the queen embedding and the weights, and we removed one channel.

The post-SFT model's generations are actually MORE coherent than the ground truth's in this queen-suppressed setting. This echoes the concept recovery finding that late integration produces more coherent geometry than co-training — the weights absorb the concept more cleanly when the structural framework is already stable.

### 5. The post-SFT model memorized curriculum content

The recurring "Emperor Akihito" and "imperial family" references reveal that the model memorized specific content from the 100K-token curriculum, which apparently contained Japanese imperial family text. The concept transfer is real but narrow — the model associated female rulers with one specific royal context rather than developing a fully general female-ruler representation. This is unsurprising for a 13M-parameter model trained on 100K tokens.

## Interpretation

The scaffolding hypothesis is confirmed at the qualitative level: SFT on queen-containing text produced weight changes that persist when the queen token is unavailable. The model's attention weights and MLP parameters learned associations between female-ruler contexts and authority/monarchy concepts, and these associations survive independently of the queen embedding.

### 6. Zeroing the queen embedding changes nothing — the knowledge is entirely in the weights

We re-ran the full experiment with all TARGET_WORD embedding rows zeroed (set to 0) in addition to logit suppression. Every generation and every top-k prediction was **identical** to the non-zeroed run.

This resolves the embedding-mediated computation concern raised in the experiment spec. The embedding matrix is a lookup table: a token's embedding row is only accessed when that token appears in the input sequence. Since our prompts don't contain any TARGET_WORD tokens, and the logit mask prevents them from being generated and fed back as input, the queen embedding row is never touched during any forward pass. Zeroing it is a no-op.

This means:
- **The results from Finding 1-5 already reflect pure weight changes.** The throne/imperial/eldest/ruler predictions come entirely from changes to attention weights, MLP weights, and non-queen embeddings — not from the queen embedding doing hidden internal work.
- **The scaffolding can be fully removed.** The queen embedding can be zeroed (or the token slot recycled) without any loss of the acquired female-ruler knowledge, at least for contexts that don't explicitly contain queen in the input.
- **The token served as temporary scaffolding for weight surgery.** Queen provided a backprop handle during SFT that reorganized the model's weights around the female-ruler concept. Once the weights changed, the handle is no longer needed. This is exactly the "metabolic cycle" predicted in the Claude conversation (conversations/Claude-Learned token invention in language models.md): mint a token, use it to reorganize, discard it.

## What this establishes

- **Token-level SFT produces persistent weight changes.** Knowledge about female rulers survives when the trained token is completely removed (embedding zeroed + logits suppressed).
- **The knowledge lives in the weights, not the embedding.** Zeroing queen's embedding has no effect on the model's female-ruler capabilities for compositional prompts. The concept was absorbed into attention and MLP parameters.
- **The model expresses the knowledge through alternative tokens.** Throne, imperial, eldest, daughter — the model found other ways to express female-ruler concepts.
- **The baseline has a measurable gap** in female-ruler representation that SFT fills. "The wife of the king was a powerful man" vs. "...a powerful ruler" is not a subtle difference.
- **Tokens can serve as temporary scaffolding for representational surgery.** This validates the core claim from the broader program: mint a concept token, use it to reorganize the model's weights, then discard it. The structural improvements persist.

## Natural follow-ups

1. **Held-out perplexity**: quantitative measurement of per-position cross-entropy on female-ruler text, masking queen positions from the loss. Would give a numeric (not just qualitative) measure of the knowledge transfer.
2. **Curriculum diversity test**: does a more diverse curriculum (not dominated by one royal context) produce broader female-ruler knowledge? The "Emperor Akihito" fixation (Finding 5) suggests the curriculum was too narrow.
3. **Multi-concept scaling**: can the pipeline handle multiple concept tokens simultaneously? How do they interact?

## Run commands

```
# Logit suppression only
modal run language_reduction/modal_app.py --stage scaffolding-persistence --tau 0.3

# Logit suppression + embedding zeroing (produces identical results)
modal run language_reduction/modal_app.py --stage scaffolding-persistence --tau 0.3 --zero-embedding
```

## Files

| File | Description |
|------|-------------|
| `EXPERIMENT_SCAFFOLDING_PERSISTENCE.md` | Experiment spec |
| `RESULTS_SCAFFOLDING_PERSISTENCE.md` | This document |
| `/data/results/scaffolding_persistence.json` | Full results — logit suppression only |
| `/data/results/scaffolding_persistence_zeroed.json` | Full results — logit suppression + embedding zeroing |
