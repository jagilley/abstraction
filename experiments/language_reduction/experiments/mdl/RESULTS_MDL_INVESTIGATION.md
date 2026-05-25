# Results: Does Concept Integration Reduce Local MDL?

**Date**: 2026-04-28
**Builds on**: `RESULTS_CONCEPT_RECOVERY.md`
**Status**: Complete (negative result)

## Question

The concept recovery experiment showed that fine-tuning "queen" into the τ=0.3 model produced structural integration: king's embedding moved, queen's neighborhood tightened and gained semantically correct family members, and the king−man+woman analogy produced cleaner outputs than even the ground truth model.

A strong form of the continual learning hypothesis predicts that integrating a new concept should *reduce the minimum description length (MDL)* of related concepts. The intuition: before queen exists, king must do double duty — encoding some implicit queen-like information because the model has no other way to represent "female person of status." After queen arrives, king can specialize, and its neighborhood should become simpler. If true, this would distinguish genuine concept learning (which reduces local MDL) from mere memorization (which increases it).

We ran three progressively cleaner tests to look for this signal.

## Test 1: Embedding MDL Proxies

**Design**: Compute neighborhood entropy, tightness (mean top-20 cosine similarity), coherence (pairwise cosine among top-20 neighbors), and local effective rank (SVD on top-20 neighbor embeddings) for gender words and control words across the start, fine-tuned, and ground truth models.

**Results**: The embedding-level metrics showed small, mostly ambiguous changes.

| | Δ entropy | Δ tightness | Δ coherence | Δ local rank |
|---|---|---|---|---|
| Gender words (mean) | +0.0001 | -0.0044 | -0.0008 | +0.14 |
| Control words (mean) | +0.0001 | -0.0012 | -0.0004 | +0.11 |
| **King** specifically | +0.000 | -0.0023 | **+0.0024** | **-0.12** |

King was the one gender word where coherence increased and local rank decreased — the direction predicted by MDL reduction. But the effects are small, and most gender words moved in the opposite direction (neighborhoods loosened slightly more than controls). Entropy of the full similarity distribution (~8.06 bits for all words, close to log₂(3200)) has no discriminative power at this vocabulary size.

King's neighbors were qualitatively stable — "man" dropped out of the top-10 and "leader" entered, but there was no dramatic noise purge of the kind seen in queen's neighborhood.

**Interpretation**: The embedding geometry is reorganizing, but the signal is too small and mixed to constitute evidence for MDL reduction at this level.

## Test 2: Paired Fine-Tuning Comparison

**Design**: The initial MDL test (comparing queen-ft to the start model on king-context loss) was confounded — the improvement could come from "more training data" rather than "queen helps king." To control for this, we fine-tuned a second model on a **random curriculum**: same token count (100,096), same hyperparameters (LR=3e-4, 5 epochs, 50 steps), same source checkpoint (τ=0.3) — but the curriculum contained no target words (queen, empress, etc.). The random curriculum was sampled from the same original corpus.

We then evaluated all models on 500 king-only windows (containing "king" but no target words) and 500 general windows (no king, no target words).

**Results**:

| Model | King loss | General loss | Δ from start (king) |
|---|---|---|---|
| start | 5.728 | 5.711 | — |
| queen-ft | 5.505 | 5.491 | -0.223 |
| random-ft | 5.425 | 5.415 | -0.304 |
| ground truth | 4.715 | 4.696 | — |

The random curriculum beat the queen curriculum on everything — king contexts, general contexts, and king-specific excess improvement. The queen curriculum's narrower distributional focus (royal/mythological text) appears to be a disadvantage: the model gets less diverse gradient signal per step.

Note: the queen curriculum naturally contained ~2x more king tokens (384 vs 201) due to king-queen co-occurrence in text. This did not translate into a king-specific advantage.

**Interpretation**: At this training scale, the loss improvements from fine-tuning are fully explained by general training signal. There is no measurable queen-specific benefit to king-context compression.

## Test 3: Two-Phase Consolidation

**Design**: The strongest version of the MDL hypothesis doesn't predict that queen reduces king's MDL *during queen-learning* — it predicts that queen creates a structural prior that makes king cheaper to maintain during *subsequent* general training. To test this, we took both the queen-ft and random-ft models and trained them on an identical general consolidation curriculum (200K tokens, no target words, same hyperparameters). We evaluated king-context loss at 10-step intervals throughout consolidation.

If queen provides explanatory power for king, the queen-ft model should consolidate king-related patterns more efficiently — its king-context loss should improve faster or further during general training.

**Results**:

| Step | queen-ft king | random-ft king | q−r (king) | q−r (general) | q king excess |
|------|-------------|--------------|-----------|--------------|-------------|
| 0 | 5.505 | 5.425 | +0.080 | +0.075 | +0.005 |
| 10 | 5.424 | 5.400 | +0.023 | +0.020 | +0.004 |
| 30 | 5.390 | 5.370 | +0.020 | +0.009 | +0.012 |
| 50 | 5.384 | 5.364 | +0.020 | +0.001 | +0.019 |
| 70 | 5.383 | 5.372 | +0.011 | -0.002 | +0.014 |
| 105 | 5.408 | 5.395 | +0.013 | -0.002 | +0.014 |

Both models started overfitting around step 50-60, so "total improvement" comparisons are misleading. The better comparison is the gap dynamics:

- **On general text**, the gap between queen-ft and random-ft closed from 0.075 to essentially zero by step 50. The two models converge.
- **On king text**, the gap closed from 0.080 to ~0.013-0.020 but never fully closed. Queen-ft retained a persistent king-specific deficit.

The "queen king excess" column (positive = queen-ft is relatively worse on king) *increased* during consolidation — from +0.005 at step 0 to +0.014-0.019. If queen provided a structural prior for king, this number should decrease. Instead, the queen-ft model consolidated general patterns effectively but lagged specifically on king.

King-specific excess improvement during consolidation:
- queen-ft: +0.006 (king improved 0.006 more than general)
- random-ft: +0.016 (king improved 0.016 more than general)

**Interpretation**: No evidence for a consolidation advantage. The random-ft model, which had no exposure to queen, consolidated king-related patterns more efficiently during general training than the queen-ft model did.

## What We Can and Cannot Conclude

### What the data supports

1. **Fine-tuning queen into the model produces genuine structural integration** (established in `RESULTS_CONCEPT_RECOVERY.md`): king moves, queen's neighborhood gains semantically correct members, the analogy outputs are coherent.

2. **This structural integration does not measurably reduce king's description length** at the model-output level, neither during queen-learning (Test 2) nor during subsequent general training (Test 3).

3. **The analogy quality improvement is a positioning effect, not a compression effect.** Queen finding a good embedding position allows the king−man+woman vector to land in a coherent region. But this doesn't make the model more efficient at predicting king's contexts — it's a *quality* change in the geometry, not a *compression* change.

### What we cannot conclude

These results do not rule out the MDL reduction hypothesis in general. Several limitations constrain how far the negative result extends:

1. **Scale**: This is a 13M parameter, 2-layer model with a 3200-token active vocabulary trained on 100M tokens. The concepts "king" and "queen" may not be compositionally complex enough in this model for queen's arrival to provide meaningful explanatory power. In a larger model where "queen" encodes a richer web of associations (politics, chess, music, monarchy), the MDL effect might be larger.

2. **Curriculum mismatch**: The queen curriculum is distributionally narrow (royal/mythological text), while the random curriculum and evaluation contexts are drawn from the general FineWeb distribution. The queen-ft model may be disadvantaged not because queen doesn't help king, but because it received less distributionally diverse training. A more controlled comparison would match the distributional properties while varying only the presence of target words.

3. **Training duration**: 50 steps of fine-tuning is enough for queen's own neighborhood to restructure, but the second-order effects on king may need substantially more gradient. The consolidation phase (105 steps) may also be too short. However, both models began overfitting by step 50-60, suggesting the curriculum is exhausted rather than that more training would help.

4. **The right level of analysis**: We measured MDL at the model-output level (cross-entropy loss). The compression might be happening in the internal representations — king's embedding becomes more factored, its attention patterns become simpler — without translating to output-level loss improvements because the model has other bottlenecks (only 2 layers, 128 dimensions).

5. **What consolidation requires**: SFT may not be the operation that produces MDL reduction. Biological consolidation during sleep involves replay, interference resolution, and structural reorganization that look qualitatively different from gradient descent on new data. The right mechanism might be closer to distillation, pruning, or something we don't have a good analogue for yet.

## Implications

The concept recovery experiment showed that token-level fine-tuning produces structural integration: a new concept slots into the embedding space and the neighborhood reorganizes. This MDL investigation shows that structural integration and MDL reduction are not the same thing — you can have the first without the second.

This matters for continual learning: if the goal is a system that becomes *simpler* as it learns more concepts (because each concept provides explanatory power for its neighbors), the mechanism needs to go beyond supervised fine-tuning. Integration is necessary but not sufficient — there must be an additional consolidation process that discovers and exploits the redundancy that new concepts create.

What that consolidation process looks like remains an open question. The connection to sleep consolidation in neuroscience is suggestive but not yet actionable.

## Files

| File | Description |
|------|-------------|
| `recovery_eval.py` | `mdl_analysis()` and `format_mdl_report()` — embedding MDL proxies |
| `modal_app.py` | `recovery_mdl_stage` (stage 14), `mdl_paired_comparison` (stage 15), `mdl_consolidation` (stage 16) |
| `RESULTS_CONCEPT_RECOVERY.md` | Prior results this investigation builds on |
| `/data/results/recovery_mdl.json` | Embedding MDL proxy results (on Modal volume) |
| `/data/results/mdl_paired.json` | Paired fine-tuning comparison results (on Modal volume) |
| `/data/results/mdl_consolidation.json` | Two-phase consolidation results (on Modal volume) |
