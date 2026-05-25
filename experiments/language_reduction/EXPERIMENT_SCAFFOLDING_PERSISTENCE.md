# Experiment: Scaffolding Persistence

**Date**: 2026-04-30
**Depends on**: Concept recovery (RESULTS_CONCEPT_RECOVERY.md), Scaffolding RL (EXPERIMENT_SCAFFOLDING_RL.md)
**Idea source**: conversations/Claude-Learned token invention in language models.md

## Motivation

The scaffolding RL experiment (EXPERIMENT_SCAFFOLDING_RL.md) showed that the post-SFT model already perfectly prefers queen over the denoiser's statistical substitutes — 100% validation accuracy at DPO step 0. This means SFT already teaches both content and preference. But it left the core scaffolding question unanswered: **do the structural improvements from training with queen persist after removing access to queen?**

The previous scaffolding eval attempted queen-zeroed comparisons but was insensitive because queen fell outside the active vocabulary (top-3200 tokens), so embedding-level metrics didn't detect any difference.

## Hypothesis

SFT on queen-containing text changed the model's internal representations (attention weights, MLP weights, non-queen embeddings) such that the model better understands "female ruler" concepts. If this is true, the post-SFT model should generate more coherent content about female rulers than the baseline, even when queen and related tokens are unavailable as outputs.

## Design

**Key intervention: logit suppression.** At each generation step, all TARGET_WORD token IDs (queen, empress, princess, goddess, heroine, priestess, duchess, countess, baroness) are set to -inf in the output logits. The model cannot select any scaffolding token as output. Applied equally to all models for fair comparison.

This avoids the confound where the post-SFT model tries to emit queen (now meaningless with zeroed embedding), producing garbage that looks like knowledge loss but is really a broken output channel.

Note: we do NOT zero the queen embedding for this test. The prompts don't contain queen, and the logit mask prevents queen from appearing in generated text. Zeroing the embedding would remove queen's participation in attention computation over other tokens — a separate question about embedding-mediated internal computation, not about whether the weight changes persisted.

### Models compared

1. **Baseline (tau=0.3)**: queen embedding is untrained noise, model never saw queen in training
2. **Post-SFT**: fine-tuned on 100K tokens of queen-containing text (from concept recovery experiment)
3. **Ground truth (tau=0.0)**: queen was in original training data

### Metrics

1. **Generation quality** on female-ruler prompts (no queen in prompt). Qualitative comparison: does post-SFT produce content that connects female rulers to authority, monarchy, power?

2. **Top-k predictions** at positions where the next token should reveal female-ruler knowledge (e.g., after "The king and his wife, the"). With queen suppressed, do the remaining top predictions reflect authority/royalty concepts?

### Success criteria

- Post-SFT generates qualitatively more coherent female-ruler content than baseline, with target tokens suppressed
- Post-SFT's top-k predictions at female-ruler positions show more authority/monarchy-related tokens than baseline
- Ground truth should be the best performer (reference ceiling)

## Run command

```
modal run language_reduction/modal_app.py --stage scaffolding-persistence --tau 0.3
```

## Known limitations

Both models still have the queen embedding participating in internal computation (attention). The post-SFT queen embedding is meaningful; the baseline's is noise. This means any advantage we observe could come from either (a) persistent weight changes (the scaffolding effect) or (b) queen's embedding doing useful computational work during the forward pass. Separating these requires a follow-up with the embedding zeroed AND logits suppressed.

## Files

| File | Description |
|------|-------------|
| `modal_app.py` | `scaffolding_persistence_stage` function |
| `EXPERIMENT_SCAFFOLDING_PERSISTENCE.md` | This document |
