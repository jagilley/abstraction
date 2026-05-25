# Experiment: Scaffolding RL for Concept Consolidation

**Date**: 2026-04-30
**Depends on**: Concept recovery experiment (RESULTS_CONCEPT_RECOVERY.md)
**Idea source**: ideas/language_reduction_continual_learning.md (RL section), conversations/Claude-Learned token invention in language models.md

## Hypothesis

DPO on (queen-version, compositional-version) pairs produces qualitatively different structural reorganization than SFT alone. Specifically:

1. **SFT teaches content**: where queen goes in embedding space, what contexts it appears in.
2. **DPO teaches utility**: that routing through queen is *worth doing* — that queen is a better compression primitive than the compositional alternative.

The DPO gradient isolates the utility signal because everything shared between the chosen (queen) and rejected (replacement) sequences cancels. What survives is purely "consolidate around this abstraction."

## The scaffolding test

The broader claim (from the Claude conversation) is that tokens can serve as *temporary scaffolding* for representational surgery. The token reorganizes the model's weights, then gets deleted. The test of success is whether the structural improvements persist after the token is removed.

Pipeline:
1. **Measure baseline**: female-ruler representation quality in the τ=0.3 model (queen embedding is untrained noise)
2. **SFT**: integrate queen via curriculum fine-tuning (already done — concept recovery experiment)
3. **DPO**: teach utility via preference pairs (queen vs. compositional alternative)
4. **Measure post-DPO**: same metrics, queen embedding intact (intermediate reference point)
5. **Delete queen embedding**: zero out queen's embedding row
6. **Measure post-deletion**: same metrics — did the structural improvements survive?

## Controls

- **More-SFT**: same starting checkpoint, same number of optimizer steps, continued SFT on queen curriculum. If DPO and more-SFT produce the same effects, the "qualitatively different gradient" claim is wrong.
- **Queen-intact vs queen-deleted comparison within each condition**: isolates scaffolding effect from direct embedding effect.

## DPO pair construction

For each position where the denoiser replaced a queen-family token with a substitute:
- **Chosen**: original window (contains queen)
- **Rejected**: identical window with only the queen token swapped for the denoiser's substitute

This is a minimal-diff pair — exactly one token differs. The DPO gradient points precisely at "prefer the version that uses queen," which is the consolidation signal.

## Metrics

At each stage (baseline, post-SFT, post-DPO, post-more-SFT), measured with queen embedding intact AND zeroed:

1. **Analogy quality**: king − man + woman → top-k results. Semantic coherence of outputs.
2. **Gender-word neighborhood coherence**: pairwise cosine among top-k neighbors of gender words (from existing MDL analysis).
3. **Embedding MDL proxies**: entropy, concentration, tightness, local effective rank for gender words vs. control words.
4. **Generation**: prompts about female rulers (without the word "queen") → continuation coherence.
5. **Forgetting**: stability of non-gender query words.

## Success criteria

- **DPO produces consolidation that more-SFT does not**: gender-authority subspace tightens more under DPO, even controlling for compute.
- **Consolidation survives queen deletion**: after zeroing queen's embedding, the structural improvements from DPO persist (analogy quality, neighborhood coherence remain elevated relative to baseline).
- **No forgetting**: control words remain stable.

## Practical details

- Model: τ=0.3, P=100M, 2-layer GPT-2 (~13M params)
- DPO pairs: ~500-1000 (from queen positions across 10 shards of original corpus)
- DPO hyperparameters: β=0.1, lr=5e-5, 3 epochs
- More-SFT: lr=3e-4 (same as original SFT), same number of steps as DPO
- All runs on Modal (A10G GPU)

## Files

| File | Description |
|------|-------------|
| `dpo_pairs.py` | DPO pair extraction from original vs denoised shards |
| `modal_app.py` | Pipeline stages (extract-dpo-pairs, dpo-train, more-sft, scaffolding-eval) |
| `recovery_eval.py` | Existing eval metrics (reused) |
| `eval_embeddings.py` | Existing embedding eval (reused) |
| `EXPERIMENT_SCAFFOLDING_RL.md` | This document |

---

## Results (2026-04-30)

### DPO pair extraction

Extracted 1,369 minimal-diff pairs from 10 shards of original vs. τ=0.3 denoised corpus. Per-word breakdown:

| Word | Pairs |
|------|-------|
| queen | 753 |
| goddess | 279 |
| princess | 240 |
| heroine | 36 |
| duchess | 32 |
| empress | 29 |

Example replacements reveal the core problem (see Finding 1 below):

```
queen -> father, man, mother, child, was, -, 11, University
```

These are frequency-based statistical substitutes from the denoiser, not semantic paraphrases.

### DPO training: catastrophic over-optimization

114 steps (3 epochs), β=0.1, lr=5e-5. The model collapsed.

| Step | Train loss | Train acc | Val loss | Val acc |
|------|-----------|-----------|---------|---------|
| 0 | 0.6931 | 0.000 | 0.6577 | **1.000** |
| 10 | 0.4474 | 1.000 | 0.4184 | 0.992 |
| 40 | 0.0034 | 1.000 | 0.0048 | 1.000 |
| 113 | 0.0000 | 1.000 | 0.0000 | 1.000 |

**Validation accuracy was 1.000 at step 0.** The SFT model already strongly prefers queen over the denoiser's statistical substitutes in every pair. DPO had nothing to learn — it drove the loss to zero, which catastrophically over-optimized the model. Post-DPO generations are garbage:

```
"She was a powerful woman who ruled" ->
  "princess, and princess,‮'佛使则'佛使分分分分分分ORGE�本呮"
```

### More-SFT control

114 steps (matched to DPO), lr=3e-4, continued SFT on queen curriculum from the same SFT checkpoint. Mild overfitting (train loss 4.76→3.40, val loss 5.64→5.92). Generations are slightly more fluent than post-SFT but not qualitatively different.

### Scaffolding eval: no detectable structural differences

| Condition | Gender analogy acc | Mean gender coherence | King's top neighbor |
|-----------|-------------------|----------------------|-------------------|
| pre_sft | 0.125 | 0.4335 | President(0.616) |
| post_sft | 0.125 | 0.4242 | President(0.620) |
| post_dpo | 0.125 | 0.4268 | President(0.617) |
| post_more_sft | 0.125 | 0.4217 | President(0.623) |
| ground_truth | 0.250 | 0.3447 | king(0.486) |

Queen-intact vs. queen-zeroed comparisons were identical across all conditions — zeroing queen's embedding row had no observable effect on any metric. This is because queen is not in the active vocabulary (top-3200 tokens by frequency), so it doesn't appear in any model's nearest-neighbor lists or analogy computations.

### Findings

#### Finding 1: The denoiser's replacements are not compositional alternatives

The hypothesis required DPO pairs where chosen and rejected express the same meaning — one using the compressed token (queen), one using a compositional route (e.g., "female ruler"). The actual pairs contrast queen against statistical substitutes: tokens the denoiser chose based on co-occurrence fit, not semantic equivalence. "The queen addressed her subjects" vs. "The father addressed her subjects" is not a compression-vs-composition contrast — it's a correct-vs-wrong contrast. After SFT, the model already knows queen is correct in these contexts, so the DPO signal is trivially solved.

#### Finding 2: SFT already teaches preference, not just content

The 100% validation accuracy at step 0 demonstrates that SFT on queen-containing text teaches the model both what queen means AND to prefer queen over the denoiser's substitutes. The content/utility distinction from the original hypothesis may not be separable with these pairs — or SFT may already capture both signals when the alternative is a non-compositional substitute.

#### Finding 3: DPO on a solved task is destructive

When the reference model and policy model agree perfectly on the preference ranking, continued DPO training amplifies confidence without bound. The loss goes to zero, which means the model assigns infinite log-odds to the chosen sequence — achievable only by collapsing probabilities to degenerate distributions. This is a known failure mode of DPO (reward hacking via over-optimization), but it manifests especially acutely here because the task is trivial from the start.

#### Finding 4: The scaffolding eval needs different sensitivity

The queen-zeroed test produced identical metrics to queen-intact because queen falls outside the active vocabulary used for analogy and neighborhood computations. The eval infrastructure from the concept recovery experiment was designed for top-3200 tokens; queen-specific effects require either expanding the active set or using metrics that directly probe queen's position (which the concept recovery experiment already does — the scaffolding eval should reuse those metrics rather than the generic eval pipeline).

### What this rules out

- **DPO with denoiser-derived pairs after SFT**: the task is trivially solved, leading to catastrophic over-optimization. This specific pipeline ordering with these specific pairs does not work.
- **DPO with denoiser-derived pairs from random embeddings** (considered but not run): would require the preference signal to simultaneously discover what queen means and learn to prefer it. Unlikely to work without warm initialization (e.g., from GLP-derived embeddings), and not representative of the production pipeline.

### What remains open

The scaffolding question — "do the structural improvements from training with queen persist after deleting queen's embedding?" — was not answered by this experiment because the DPO step failed and the eval metrics were insensitive to queen deletion. The SFT model from the concept recovery experiment already has structural integration (king moved, neighborhood tightened). Testing whether those changes survive queen deletion requires:

1. Metrics that operate on the full vocabulary (not just top-3200)
2. Direct comparison of the gender-authority subspace geometry before/after queen deletion
3. Generation tests with compositional prompts (already collected but need analysis beyond the scope of this experiment)

This is the natural next experiment — it tests whether SFT alone produces the scaffolding effect, before investing in better DPO pairs.
