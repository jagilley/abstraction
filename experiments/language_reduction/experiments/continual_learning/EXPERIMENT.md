# Experiment: Continual Learning Benchmark via Vocab-Only Reduction

**Status**: Phase 1 (replication)
**Date**: 2026-05-07
**Builds on**: `RESULTS_CONCEPT_RECOVERY.md`, `RESULTS_SCAFFOLDING_PERSISTENCE.md`, vocab-only addendum in `STATUS.md`
**Prerequisites**: Vocab-only sweep complete at τ=0.3 (models at `/data/models/vocab_only/tau_0.300/`)

## Motivation

The concept recovery and scaffolding persistence experiments established that token-level fine-tuning produces structural integration: training a single token (queen) into a vocab-reduced model restructures neighboring embeddings (king moves), and the acquired knowledge persists in the weights even when the token is removed.

The vocab-only mode provides cleaner experimental conditions than spectral denoising because it doesn't alter the DGP — it's a lossy observation channel (γ invariant, Cagnetta formula tracks). This means any degradation we observe during sequential concept recovery can be attributed to the model's internal dynamics, not to artifacts of the corpus transformation.

This experiment turns concept recovery into a **repeatable continual learning benchmark**: recover collapsed tokens one at a time, measuring both forward recovery and backward degradation after each step.

## Phase 1: Replication

Replicate the concept recovery results from `RESULTS_CONCEPT_RECOVERY.md` on the vocab-only τ=0.3 model. The original experiment used spectral denoising; we use vocab-only reduction instead.

### Setup

- **Starting model**: vocab-only τ=0.3, P=100M, T=128 (at `/data/models/vocab_only/tau_0.300/P_100000000/T_128/`)
- **Ground truth**: τ=0.0, P=100M, T=128 (at `/data/models/tau_0.000/P_100000000/T_128/`)
- **Target words**: Same as concept recovery: queen, empress, princess, goddess, heroine, priestess, duchess, countess, baroness
- **Curriculum**: 100K tokens from original corpus windows containing target words
- **Fine-tuning**: LR=3e-4, 5 epochs (matching the successful Run 2 from concept recovery)

At vocab-only τ=0.3 (v'=1,724), the collapse analysis showed "King absorbs 48 tokens including Queen, Princess, Knight, Hero." All target words are collapsed — their embeddings are untrained noise, just as in the spectral experiment.

### Metrics

Same as concept recovery:

**M1. Embedding shift toward ground truth.** For each gender word and target word, compute `toward_gt = cos(emb_ft, emb_gt) - cos(emb_start, emb_gt)`. Positive = moved toward ground truth.

**M2. Neighbor overlap with ground truth.** Top-10 nearest neighbors (cosine, restricted to vocab-only active vocabulary of 1,724 tokens) compared across fine-tuned, starting, and ground truth models.

**M3. Analogy quality.** king − man + woman = ? on the fine-tuned model vs starting and ground truth.

**M4. Analogy accuracy recovery.** Full `eval_analogies` battery, focusing on gender category.

**M5. Forgetting.** NN coherence on control words and analogy accuracy on tense/plural must not degrade.

### Predictions

1. Queen's neighborhood tightens and gains semantically correct members (paralleling spectral result).
2. King's embedding moves — structural integration, not additive memorization.
3. Whether king moves toward or away from ground truth is informative but not the success criterion (we learned from the spectral experiment that the fine-tuned model finds its own geometry).
4. Tense and plural categories unaffected.

### Differences from spectral replication

- The vocab-only active vocabulary is 1,724 tokens (vs 3,200 for spectral). Neighbor evaluations use this smaller set.
- The vocab-only model may have learned different internal structure due to the different reduction method. The DGP is preserved (γ invariant) but the channel is different (context-independent type mapping vs context-dependent occurrence replacement).
- The starting model was trained on text where queen→king deterministically (vocab-only) rather than queen→[various replacements] contextually (spectral). This may affect how cleanly queen can be recovered.

## Phase 2: Sequential Recovery Benchmark

### Design

Given a queue of collapsed tokens [t₁, t₂, ..., tₖ]:

1. **Round 0**: Fine-tune base model on curriculum for t₁. Save checkpoint. Evaluate t₁ recovery.
2. **Round 1**: Starting from round 0 checkpoint, fine-tune on curriculum for t₂. Save checkpoint. Evaluate t₂ recovery AND t₁ retention.
3. **Round i**: Starting from round i-1 checkpoint, fine-tune on curriculum for tᵢ₊₁. Evaluate tᵢ₊₁ recovery AND all previous tokens' retention.

### Degradation metrics

For each previously-recovered token t at round j > recovery_round(t):

- **Embedding stability**: `cos(emb_round_j[t], emb_recovery_round[t])` — how much did the embedding move since recovery?
- **Neighbor stability**: overlap between round j neighbors and recovery-round neighbors for t
- **Forward-pass loss**: cross-entropy on t's curriculum windows, evaluated at each round
- **Toward-GT stability**: whether `toward_gt` increases, decreases, or holds across rounds

### Token ordering

The order of token recovery matters. Semantically related tokens may reinforce each other, while unrelated tokens may interfere differently. Two orderings are implemented:
- **Clustered**: geography → generational → gender (beijing, tokyo, madrid, grandfather, grandmother, princess). Groups semantically related tokens so we can observe within-group reinforcement.
- **Interleaved**: one from each domain in rotation (beijing, grandfather, princess, tokyo, grandmother, madrid). Maximizes cross-domain interference between consecutive rounds.

### What success looks like

A clean benchmark would show:
- **Monotonic recovery**: each token's metrics improve at its recovery round
- **Bounded degradation**: previously-recovered tokens degrade, but to a measurable and predictable degree
- **Ordering effects**: semantic clusters degrade less than random interleaves (if the model's representational structure is compositional)

## Data layout on Modal

```
/data/continual_learning/
  tau_0.100/
    round_{0-5}/                        # Phase 1: independent recovery
      curriculum.npy, curriculum_meta.json
      model.pt, results.json, eval.json
    batch_summary.json
    dimension_analysis.json
    sequential/                         # Phase 2: sequential recovery
      clustered/
        round_{0-5}/
          curriculum.npy, curriculum_meta.json
          model.pt, results.json, eval.json
        degradation_matrix.json
        dimension_analysis.json
      interleaved/
        ...
```

## Running

```bash
# Phase 1: Independent recovery at τ=0.1
modal run language_reduction/modal_app.py --stage cl-batch --tau 0.1 --lr 3e-4
modal run language_reduction/modal_app.py --stage cl-dimensions --tau 0.1

# Phase 2: Sequential recovery with degradation tracking
modal run language_reduction/modal_app.py --stage cl-sequential --tau 0.1 --ordering clustered
modal run language_reduction/modal_app.py --stage cl-sequential --tau 0.1 --ordering interleaved
```

## Files

| File | Description |
|------|-------------|
| `EXPERIMENT.md` | This spec |
| `README.md` | Phase 1 results |
| `stages.py` | Modal pipeline stages (Phase 1 + Phase 2) |
| `__init__.py` | Package marker |
