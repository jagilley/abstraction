# Vocab-Only Geometry Eval: Results

**Date**: 2026-05-06

Evaluated the vocab-only models (rare tokens mapped to nearest GPT-2 embedding neighbors) across three tau values. Vocab sizes: τ=0.1 → 10,833 types, τ=0.3 → 1,724 types, τ=0.5 → 236 types.

## 1. Collapse Characterization

The many-to-one mapping has qualitatively different character at each tau.

**τ=0.1**: Gentle. 152/162 analogy words preserved. Collapsed tokens are mostly casing variants (King→king), proper nouns (tokyo→Japan, beijing→China), and rare morphological forms (grandfather→father). Mean collapse ratio 4.3 tokens per target. Intra-group GPT-2 embedding coherence is high (mean cosine 0.43).

**τ=0.3**: Semantically interesting. 117/162 analogy words preserved. Collapsed mappings are meaningful: girl→woman, boy→child, daughter→son, husband→father. "King" absorbs 48 tokens including Queen, Princess, Knight, Hero. "Woman" absorbs 101 tokens. Mean collapse ratio 29 per target. Group coherence drops (0.35) as more heterogeneous tokens get absorbed.

**τ=0.5**: Extreme. Only 23/162 analogy words preserved. Antonym collapse is common (big→small, old→new, bad→good) — expected, since antonyms are distributional neighbors. "King"→The. Mean collapse ratio 216 per target. The alphabet is coarser than ASCII.

## 2. Embedding Geometry (Static, P=100M)

| Metric | τ=0.0 (baseline) | vo τ=0.1 | vo τ=0.3 | vo τ=0.5 |
|---|---|---|---|---|
| Analogy accuracy | 0.493 | 0.466 | 0.329 | 0.000 |
| Eff. rank | 87.0 | 88.3 | 91.5 | 114.5 |
| NN coherence | 0.358 | 0.380 | 0.318 | 0.082 |
| Similarity ρ | 0.162 | 0.075 | 0.021 | 0.012 |
| Top-10 SV energy | 0.300 | 0.291 | 0.281 | 0.185 |
| Avg cosine | 0.015 | 0.006 | 0.004 | 0.000 |

Effective rank *increases* with tau (opposite to spectral, which decreased). With fewer active token types, the model spreads its embedding capacity more evenly — there are fewer dominant directions in the embedding space. Spectral concentration (top-10 SV energy) decreases correspondingly.

Coherence peaks at τ=0.1 (0.380, above baseline's 0.358) then drops. This is different from spectral denoising, which showed monotonically increasing coherence. At τ=0.3 and beyond, the many-to-one collapse makes coherence harder to measure meaningfully — the same token now represents many different concepts, so its neighbors are a mixture.

Comparison to spectral τ=0.3 is not direct: spectral replaces ~30% of token *occurrences* (context-dependent, soft), while vocab-only maps ~97% of token *types* to 1,724 representatives (context-independent, hard). The interventions are different in kind, not just degree.

## 3. Contextual Embeddings (P=100M)

| | static | L1 | L2 |
|---|---|---|---|
| **τ=0.0** eff. rank | 87.0 | 49.1 | 45.7 |
| **vo τ=0.1** eff. rank | 88.3 | 48.5 | 33.1 |
| **vo τ=0.3** eff. rank | 91.5 | 45.1 | 16.0 |
| **vo τ=0.5** eff. rank | 114.5 | 43.2 | 11.8 |
| | | | |
| **τ=0.0** coherence | 0.358 | 0.565 | 0.601 |
| **vo τ=0.1** coherence | 0.380 | 0.577 | 0.666 |
| **vo τ=0.3** coherence | 0.318 | 0.469 | 0.629 |
| **vo τ=0.5** coherence | 0.082 | 0.107 | 0.222 |

All vocab-only models show stronger layer-wise compression than baseline. At τ=0.3, effective rank drops from 91.5 (static) to 16.0 (L2) — the model compresses to a much lower-dimensional representation through processing. The baseline only goes from 87 to 46. This is consistent with the model doing more computational work per layer to compensate for the coarser input vocabulary.

At τ=0.1, the L2 coherence (0.666) exceeds the baseline (0.601) despite starting from similar static coherence. At τ=0.3, the model recovers from a coherence deficit at static level (0.318 vs 0.358) to slightly exceed baseline at L2 (0.629 vs 0.601). At τ=0.5, the model can't close the gap — 0.222 at L2 vs 0.601 for baseline.

## 4. Structure Probe: Contextual Disambiguation

Tested whether the model's L2 hidden states carry information about which *original* token occupied a position, even when the input token is the same after vocab collapse. Linear probe (ridge regression, 80/20 split) classifying source token identity from hidden states at positions where the same target token appears.

### τ=0.3 (1,724 types)

| Target | Sources | Probe acc | Chance | F-ratio |
|---|---|---|---|---|
| "Some" | uck, aining, umps, some, Google | 0.776 | 12.5% | 211 |
| "Although" | Although, Though, Despite, Indeed, © | 0.667 | 12.5% | 293 |
| "diseases" | injuries, oids, disorders, traits, disabilities | 0.668 | 12.5% | 58 |
| "emissions" | dust, EPA, pollution, greenhouse, dioxide | 0.620 | 12.5% | 64 |
| "While" | ounds, owing, Despite, Both, Though | 0.560 | 12.5% | 128 |
| "foods" | milk, bread, vegetables, agriculture, calories | 0.564 | 12.5% | 59 |
| "diabetes" | depression, obesity, insulin, glucose, autism | 0.543 | 12.5% | 45 |
| "bacteria" | colon, particles, organisms, antibiotics, bacterial | 0.530 | 12.5% | 38 |
| "achieve" | implement, manage, obtain, capture, recover | 0.465 | 12.5% | 70 |
| "John" | David, Paul, James, William, George | 0.223 | 12.5% | 7 |

### τ=0.5 (236 types)

| Target | Sources | Probe acc | Chance | F-ratio |
|---|---|---|---|---|
| "They" | com, States, Israel, They, Germany | 0.706 | 12.5% | 251 |
| "This" | og, let, There, What, www | 0.628 | 12.5% | 420 |
| hidden bytes | various | 0.638 | 12.5% | 71 |
| "In" | As, When, To, German, based | 0.584 | 12.5% | 389 |
| ".," | ade, ives, ://, ium, ane | 0.613 | 12.5% | 253 |
| hidden bytes | various | 0.567 | 12.5% | 63 |
| hidden bytes | various | 0.557 | 12.5% | 47 |
| "students" | class, school, University, education, student | 0.548 | 12.5% | 83 |

All probes are well above chance, confirming that the model recovers sub-token information from context at both tau values. Disambiguation accuracy is higher when collapsed sources come from genuinely different contexts (heterogeneous groups) and lower when they are distributional near-synonyms ("John" group: 0.223, barely 2x chance).

## 5. Cross-Entropy Decomposition

| | τ=0.3 (1,724 types) | τ=0.5 (236 types) |
|---|---|---|
| H(unigram) | 6.43 | 4.95 |
| H(bigram\|prev) | 4.97 | 4.22 |
| H(model) | 4.38 | 3.60 |
| Context gain: unigram → bigram | 1.47 nats | 0.73 nats |
| Context gain: bigram → model | 0.58 nats | 0.63 nats |
| Bigram share of total gain | 71.6% | 53.9% |

At τ=0.3, bigrams account for 72% of the model's predictive advantage over unigrams. At τ=0.5, this drops to 54% — the model relies more on longer-range dependencies when the vocabulary is smaller. With fewer symbols, bigram statistics carry less information (there are fewer distinguishable bigrams), so the model must extract more from wider context.

Observed bigrams: 654K at τ=0.3 (of 1724² ≈ 3M possible), 51K at τ=0.5 (of 236² ≈ 56K possible). Effective bigrams (by entropy): 89K at τ=0.3, 9.7K at τ=0.5. The reduced alphabet has strong sequential constraints at both levels.

## Summary

The vocab-only models learn real structure at all tau values tested. The structure is different from spectral-denoised models in kind, not just degree:

- **Static embeddings** become higher-dimensional and less spectrally concentrated as tau increases (opposite to spectral). This reflects the embedding table having fewer "real" entries to organize, with most of the vocabulary effectively unused.
- **Contextual representations** compress more aggressively through layers, especially at high tau. The model compensates for impoverished input by doing more computation per layer.
- **Contextual disambiguation** works well (5x+ above chance) for heterogeneous collapse groups, confirming the model recovers sub-token information from positional context. Accuracy degrades gracefully as collapse groups become more semantically homogeneous.
- **Prediction strategy** shifts from bigram-dominated (72% at τ=0.3) to longer-range-dependent (54% bigram at τ=0.5) as the vocabulary shrinks.

None of these findings update priors on the core scaling results from the STATUS.md addendum (γ invariance, Cagnetta formula tracking, channel vs DGP factorization). They confirm the mechanism: vocab-only reduction is a lossy observation channel, and the model adapts its representational strategy to extract what structure remains.
