# Embedding Geometry Analysis

Does the τ=0.3 model preserve the core geometric structure of language embeddings? We evaluated word analogies, word similarity, neighborhood coherence, semantic clustering, and embedding space geometry across all (τ, P) combinations, plus contextual (per-layer) embeddings at P=100M.

## Setup

Models: 2-layer, 4-head, 128-dim GPT-2 with full vocabulary (V=50257, ~13M params). Evaluation restricted to the 3200 most frequent tokens (the "active" vocabulary) for geometry metrics and nearest-neighbor search to avoid confounds from untrained embeddings. Analogy and similarity benchmarks use the full vocabulary for lookup but restrict the search space to active tokens.

Test sets: 73 analogy quads across 6 categories (gender, plural, tense, comparative, country/capital, opposites), 46 word similarity pairs (from SimLex-999 and WordSim-353), 10 semantic categories for clustering.

Code: `eval_embeddings.py` (evaluation battery), `modal_app.py` stages `embedding-eval` and `contextual-eval`.

## Result 1: The model is not lobotomized

The τ=0.3 model learns meaningful embedding structure at all P values:

| P | Analogy acc (τ=0.0) | Analogy acc (τ=0.3) | NN coherence (τ=0.0) | NN coherence (τ=0.3) |
|---|---|---|---|---|
| 100K | 0.000 | 0.000 | 0.098 | 0.097 |
| 1M | 0.068 | 0.027 | 0.203 | 0.226 |
| 10M | 0.342 | 0.288 | 0.315 | 0.368 |
| 100M | 0.466 | 0.397 | 0.354 | 0.404 |

Analogy accuracy scales with P for both models. At P=100M, τ=0.3 gets 40% — lower than τ=0.0's 47%, but not degraded.

Nearest neighbors are qualitatively sensible:
- "school" → education, learning, student, curriculum, teacher (τ=0.3)
- "woman" → mother, child, baby, she, wife (τ=0.3)
- "write" → read, writing, reading, speak, hear (τ=0.3)

## Result 2: The most regular morphological patterns survive denoising; less regular ones don't

Per-category analogy accuracy at P=100M:

| Category | τ=0.0 | τ=0.3 | Δ | Type |
|---|---|---|---|---|
| tense | 0.714 | 0.714 | 0.000 | morphological (high-freq) |
| plural | 0.667 | 0.733 | +0.067 | morphological (high-freq) |
| comparative | 0.625 | 0.375 | -0.250 | morphological (lower-freq) |
| opposites | 0.500 | 0.300 | -0.200 | semantic |
| gender | 0.250 | 0.125 | -0.125 | semantic |
| country/capital | 0.000 | 0.000 | 0.000 | factual (too hard for model) |

Tense is perfectly preserved (10/14 for both τ). Plural is the only category that *improves* under denoising. These are the two most regular, highest-frequency morphological patterns in English — exactly the kind of structure that lives in the top of the co-occurrence spectrum and survives the denoising threshold.

Comparative degrades despite being morphological — comparative forms (bigger, smaller) are less frequent and more context-dependent than plurals or past tenses. Semantic categories (gender, opposites) degrade consistently. The dividing line isn't morphological vs semantic — it's regularity and frequency.

## Result 3: Denoising trades distributional breadth for topical coherence

**Neighborhood coherence** (avg pairwise cosine among a word's top-10 nearest neighbors) is consistently higher for τ=0.3:

| P | τ=0.0 | τ=0.3 |
|---|---|---|
| 1M | 0.203 | 0.226 |
| 10M | 0.315 | 0.368 |
| 100M | 0.354 | 0.404 |

Per-word at P=100M, τ=0.3 wins on 12/15 query words. The biggest gains are on words with diverse usage patterns in natural text:

| Word | τ=0.0 | τ=0.3 | Interpretation |
|---|---|---|---|
| king | 0.251 | 0.411 | τ=0.0 scatters to Church, Henry, St; τ=0.3 clusters on President, Lord, father, son |
| school | 0.384 | 0.488 | τ=0.0 includes morphological variants (School, schools); τ=0.3 all conceptual (education, learning, student) |
| old | 0.186 | 0.331 | τ=0.0 neighbors are BPE fragments (ard, ak); τ=0.3 is cleaner (young, Charles, mother) |

The tradeoff: standard word similarity correlation (Spearman ρ with human ratings) is weak for τ=0.3 (ρ ≈ 0 at all P) vs positive for τ=0.0 (ρ = 0.18 at P=100M). The denoised model's notion of similarity doesn't align with human judgments — it learns a narrower but more internally consistent view of each word.

## Result 4: The king − man + woman test

Neither model returns "queen" (likely not in the top 3200 active tokens for FineWeb-Edu), but the neighborhood of the query vector is revealing:

**τ=0.0**: Church, child, wife, king, baby, Children, Women, People, daughter, mother — scattershot, top result is unrelated.

**τ=0.3**: wife, mother, Mary, President, president, She, daughter, female, she, male — almost every result is gender-relevant. The model has a clean "female person of status" direction even if it can't land on the exact word.

## Result 5: Embedding geometry reflects subtask tail truncation

| Metric | τ=0.0 (P=100M) | τ=0.3 (P=100M) |
|---|---|---|
| Effective rank | 86.9 | 78.6 |
| Avg pairwise cosine | 0.015 | 0.036 |
| Top-10 SV energy | 0.301 | 0.338 |

The τ=0.3 embedding space is lower-dimensional (fewer effective directions), more anisotropic (tokens share more common structure), and more spectrally concentrated. This is what "truncating the subtask tail" looks like in representation space: fewer independent directions, more shared variance.

## Result 6: Contextual embeddings — coherence persists, models converge

Context-averaged hidden states (500 batches of 64×128, ~4M tokens, P=100M models):

| Metric | τ=0.0 static | τ=0.0 L1 | τ=0.0 L2 | τ=0.3 static | τ=0.3 L1 | τ=0.3 L2 |
|---|---|---|---|---|---|---|
| Analogy acc | 0.466 | 0.438 | 0.438 | 0.397 | 0.329 | 0.342 |
| Similarity ρ | 0.181 | 0.159 | 0.094 | -0.020 | -0.001 | -0.063 |
| Eff. rank | 86.9 | 50.8 | 43.7 | 78.6 | 38.8 | 35.2 |
| Avg cosine | 0.015 | 0.142 | 0.219 | 0.036 | 0.166 | 0.234 |
| NN coherence | 0.354 | 0.557 | 0.603 | 0.404 | 0.608 | 0.656 |
| Clustering ratio | 31.5 | 3.6 | 2.3 | 16.0 | 3.1 | 2.2 |

Key observations:

1. **Coherence gap is stable through layers** (~0.05 at every level). The τ=0.3 coherence advantage is not a static embedding artifact — it persists into the model's internal representations.

2. **Both models concentrate dramatically through layers** (eff rank 87→44 for τ=0.0, 79→35 for τ=0.3). The attention mechanism compresses the representation space for both models.

3. **Clustering ratio collapses and converges**: despite starting at 31.5 vs 16.0, both models arrive at ~2.3 by layer 2. The prediction head forces a common representational geometry regardless of input distribution. The per-category structure that exists in the lookup table gets diffused as the model optimizes for next-token prediction.

## Why lower loss ≠ better embeddings

The τ=0.3 models achieve lower validation loss at P ≤ 10M (see STATUS.md), but each model's loss is measured on its own distribution. The denoised distribution has lower entropy by construction — the hard-to-predict token-occurrences were removed. Lower loss means "the target is simpler," not "the representations are richer." The steeper scaling exponent (α_D) means the model learns its own distribution more efficiently, which is the theoretical prediction: truncate the subtask tail, small models saturate faster.

## File overview

- `eval_embeddings.py` — Evaluation battery: analogies, similarity, geometry, clustering, neighborhood coherence
- `modal_app.py` stages:
  - `--stage embedding-eval` — Static embedding evaluation across all (τ, P)
  - `--stage contextual-eval` — Contextual embedding evaluation (per-layer, P=100M)
  - `--stage analogy-check` — One-off king−man+woman spot check
- Results saved to Modal volume: `/data/results/embedding_eval.json`, `/data/results/contextual_embedding_eval.json`
