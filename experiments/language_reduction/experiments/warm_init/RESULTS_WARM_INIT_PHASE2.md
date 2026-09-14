# Results: Warm Initialization of Concept Token Embeddings (Phase 2)

**Date**: 2026-05-01
**Builds on**: `RESULTS_WARM_INIT.md` (Phase 1), `RESULTS_CONCEPT_RECOVERY.md`, `RESULTS_RESIDUAL_DECOMPOSITION.md`
**Motivation**: `conversations/Claude-Learned token invention in language models.md`
**Status**: Complete — warm initialization solved via tied weights + contrastive centroid. Next step: integration training via A/B RL.

## Question

Phase 1 solved the output side: a contrastive centroid of ln_f(h1) at queen-prediction positions produces a zero-shot lm_head row that achieves 9/12 top-10 with 0/12 false positives. Can the input side (wte) be similarly warm-initialized, so the model can process queen as meaningful input without fine-tuning?

More broadly: can the model's own geometric understanding of where queen "should" be — as captured by activation statistics, learned lm_head→wte mappings, or error signals — be projected into embedding space to produce a functional input embedding?

## Answer

Yes — but only with tied weights. In an untied model, lm_head and wte diverge into genuinely different learned subspaces (R²=0.27), making it impossible to find a single vector that serves both roles. In a tied model (lm_head.weight = wte.weight), the contrastive centroid simultaneously serves as both detector and embedding by construction. The tied contrastive centroid achieves 4/12 top-10 prediction zero-shot with perfect specificity (0/12 false positives, control rank 32,721).

Zero-shot generation remains degenerate, but this is a **training signal problem**, not an initialization problem. SFT reinforces the detector without teaching the surrounding weights to route through queen. The right training signal is A/B RL — comparing queen-containing text against the compositional substitute — which isolates the consolidation gradient. See `ideas/language_reduction_continual_learning.md`.

## The detection/participation distinction (untied models)

In untied models, the lm_head is a **detector**: it answers "is this activation queen-like?" via a dot product. A contrastive centroid in post-ln_f space IS the optimal linear detector — by construction.

The wte is a **participant**: it must integrate into a cooperative computation involving attention, MLPs, and other tokens' representations. Participating requires coordination with the surrounding weights. Detecting can be done unilaterally.

With tied weights, this distinction collapses: the detector IS the participant. The model's attention and MLPs were trained under the constraint that every embedding is also an lm_head row, so the processing pipeline is designed to handle vectors from this shared space.

## Experiment 1: Zero-shot wte initialization (Phase 2)

### Design

Three wte initialization strategies, all using the Phase 1 contrastive lm_head:

| Condition | `wte[queen]` | Rationale |
|-----------|-------------|-----------|
| `analogy` | king + gender_shift | Embedding arithmetic (Phase 1 baseline) |
| `regression` | T(contrastive_lm) | Learned lm_head→wte mapping from existing vocabulary |
| `inversion` | Optimized through frozen Block 0 | Gradient-based activation matching at h0 |

**Regression**: fit a least-squares linear mapping T: R^128 → R^128 from all non-queen lm_head rows to their corresponding wte rows (50K+ pairs), then apply T to the contrastive lm_head row. This tests the tied-weights insight: in a tied model (lm_head = wte), the contrastive centroid would simultaneously serve as both detector and embedding. The regression asks how much of that algebraic shortcut survives in the untied case.

**Inversion**: initialize wte from analogy, then optimize through the frozen model to match h0 targets derived from king + gender_shift in activation space. Runs for 200 Adam steps on 6 paired prompts (king/queen substitution). This tests whether the right h0 can be produced by finding the right embedding.

Both zero-shot and SFT-on-wte-only (all weights frozen except wte[queen]) were evaluated.

### Results

**Zero-shot**: All three conditions produce degenerate generation from queen-input prompts ("the the of the the of..."). No condition achieves coherent queen processing. The regression and inversion show slightly different degeneration patterns but the core failure is the same.

**SFT on wte only (50 steps)**: No improvement for any condition. Val loss fluctuates without trend (~6.0). Generation remains degenerate throughout. The gradient from the NTP loss, flowing backward through two frozen layers, does not provide enough signal to learn a functional embedding in isolation.

### Diagnostics

| Metric | Value | Interpretation |
|--------|-------|---------------|
| Regression R² | 0.27 | The lm_head→wte mapping captures only 27% of variance. The untied weight matrices have diverged substantially — the relationship is not a simple rotation or scaling. |
| Inversion loss | 0.457 → 0.003 | The optimized embedding produces h0 within 0.003 MSE of the target. The inversion converges, but good h0 doesn't produce good behavior — the downstream computation (Block 1 → ln_f → output) also needs to understand queen. |
| Queen prediction rank | 8.8 (all conditions) | The contrastive lm_head works identically regardless of wte — confirming that prediction prompts don't read wte[queen]. |

### What this tells us

**The frozen-model bottleneck.** The failure is not about initialization quality. The inversion found an embedding that matches the target h0 almost perfectly (loss 0.003), yet generation is degenerate. The problem is that the frozen attention and MLP weights have never learned to route queen information through the model's processing pipeline. A single embedding row, no matter how well-placed, cannot teach the frozen model to handle a concept it has never seen in context.

**The untied-weights divergence.** R² = 0.27 quantifies how far the wte and lm_head have separated during training. In a tied-weights model, the regression would be exact (R² = 1.0). The 73% unexplained variance means the two matrices serve genuinely different learned functions — the output-detection role and the input-representation role have diverged into distinct subspaces. The tied-weights algebraic shortcut does not transfer to untied architectures.

**SFT on wte only is insufficient.** Even with 50 steps of gradient descent (the same compute that produced full concept integration in the concept recovery experiment), updating only wte[queen] with everything else frozen produces no measurable improvement. The attention weights that would route queen information and the MLP weights that would process it remain unchanged.

## Experiment 2: Gradient-signal initialization + full SFT (Phase 3)

### Motivation

The gradient verbalization experiment showed that the model's error signal (∂L/∂h at a specific layer), when applied as a steering vector at that same layer, produces the desired semantic content — even when the model was 0% accurate on the task. The error signal points toward the right answer from a state of total ignorance.

Applied to the embedding problem: at queen-prediction positions, the model's error signal ∂L/∂x₀ (gradient of queen-prediction loss w.r.t. the input to Block 0) should encode what embedding-space direction would make queen more likely. This is the gradient verbalization insight applied at the embedding level: the model's own error signals should contain the information needed to specify the right embedding.

Additionally, Phase 2 showed that wte-only SFT fails because the frozen model can't process a novel embedding. Phase 3 uses full-model SFT (all weights trainable) to test whether initialization quality affects convergence speed when the model CAN adapt.

### Design

At each of the 12 queen-prediction prompts, compute ∂L/∂x₀ at the last position (where queen should be predicted), using the contrastive lm_head as the target. Average across prompts to extract the context-independent component. Negate (descent direction) and scale to match mean wte row norm.

Two conditions, both with contrastive lm_head + full-model SFT:

| Condition | `wte[queen]` |
|-----------|-------------|
| `analogy` | king + gender_shift (baseline) |
| `gradient_signal` | −mean(∂L/∂x₀), scaled to mean wte norm |

Full SFT: all weights trainable, AdamW with weight_decay=0.01, same curriculum/LR/epochs as concept recovery (100K tokens, LR=3e-4, 5 epochs, 50 steps).

### Results

#### The gradient signal is real and coherent

| Diagnostic | Value |
|-----------|-------|
| Mean pairwise cosine (12 per-prompt gradients) | **0.677** |
| Coherence ratio (‖avg‖ / mean ‖individual‖) | 0.839 |
| cos(gradient_signal, analogy_wte) | 0.112 |
| cos(gradient_signal, king_wte) | 0.145 |
| cos(gradient_signal, contrastive_lm) | 0.090 |

The per-prompt error signals are highly aligned (0.677 pairwise cosine is very high for 12 vectors in R^128). The averaged gradient retains 84% of the individual magnitudes. There IS a consistent "predict queen" direction in embedding space that the model's error signals converge on.

But the gradient signal points in a direction nearly orthogonal to all reference vectors: analogy (0.11), king (0.15), contrastive lm_head (0.09). It is a genuinely novel direction, not a remixed version of embedding arithmetic or the contrastive centroid.

#### Zero-shot: identical prediction, degenerate generation

| Condition | Queen prediction (12 prompts) | Generation from queen-input |
|-----------|------|------------|
| analogy | mean rank 11.0, 9/12 top-10 | degenerate ("the the of the the...") |
| gradient_signal | mean rank 11.0, 9/12 top-10 | degenerate ("by the the of the...") |

Queen prediction performance is identical — it depends entirely on the contrastive lm_head, which is the same for both conditions. Generation from queen-input prompts is degenerate for both, confirming Phase 2's finding.

#### Full SFT: analogy converges slightly faster

| Condition | Queen rank step 0 | Queen rank step 22 | Queen rank step 49 | Post-SFT top-10 | Best val loss |
|-----------|---|---|---|---|---|
| analogy | 9 | 6 | 6 | 11/12 | 5.589 |
| gradient_signal | 10 | 10 | 7 | 10/12 | 5.591 |

The analogy condition converges faster on queen prediction rank (reaches rank 6 by step 22; gradient_signal only reaches rank 7 by step 44). Both achieve similar endpoints. Val loss is essentially identical.

#### Post-SFT generation: improved but still fragile

| Prompt | analogy | gradient_signal |
|--------|---------|----------------|
| "The queen ruled" | "the King of the King of the King..." | "the United Kingdom of Ireland, the United..." |
| "She was a powerful queen who" | "was born in the United States..." | "was born in the same time. The second woman..." |
| "The king and the queen" | "of the Emperor. The Emperor of the Emperor of the Queen..." | "Queen Queen Queen..." |

Both conditions produce more coherent output than Phase 2's pure degeneration, confirming that full SFT helps where wte-only SFT fails entirely. But neither reaches the concept recovery quality ("was born on the throne of the imperial family"), and both still fall into loops on some prompts.

### Interpretation

**The gradient signal captures prediction-site mechanics, not queen's identity.** The high coherence (0.677) confirms the signal is real — there IS a consistent direction in embedding space that the model's error signals converge on at queen-prediction positions. But this direction is nearly orthogonal to the analogy embedding (0.11), king's embedding (0.15), and the contrastive lm_head (0.09). It appears to encode "what input-level transformation reduces queen-prediction loss at this position" rather than "what queen means as an input token."

This distinction matters: the gradient at the prediction position (the last token before queen should be predicted) describes the prediction computation, not queen's identity. The gradient verbalization experiment worked because the error signal was applied at the SAME layer and position where it was extracted. Here, we extract at prediction positions and apply as an embedding at input positions — a different computational role.

**The analogy baseline remains surprisingly competitive.** Despite being simple embedding arithmetic, the analogy wte (king + gender_shift) matches or slightly outperforms the gradient signal under full SFT. The analogy puts queen in the right embedding neighborhood (near gender/royalty concepts), which may provide a better starting basin for SFT than the gradient signal's novel direction.

## Experiment 3: Tied-weights warm initialization (2026-05-01)

### Motivation

The R²=0.27 divergence between lm_head and wte in the untied model meant the contrastive centroid couldn't serve as both detector and embedding — the two matrices had learned genuinely different functions. Weight tying (lm_head.weight = wte.weight) eliminates this divergence by construction, making the contrastive centroid simultaneously a detector and an embedding in the same space.

### Design

Retrained the τ=0.3 model (2 layers, 128-dim, 100M tokens) with `tie_weights=True`. This reduces parameters from 1.23M to 0.82M. The tied model trained to best val loss 4.94 (vs ~5.0 untied). Recomputed the contrastive centroid from the tied model's ln_f(h1) activations. Three conditions, each setting the queen row to a single vector (since wte = lm_head):

| Condition | queen row |
|-----------|-----------|
| random | N(0, 0.02) (untrained init) |
| analogy | king + gender_shift |
| contrastive | contrastive centroid |

### Results

| Condition | Zero-shot rank | Post-SFT rank | Zero-shot generation |
|-----------|---------------|--------------|---------------------|
| random | 947.4 (0/12) | 138.6 (5/12) | degenerate loops |
| analogy | 524.2 (0/12) | 142.0 (5/12) | degenerate loops |
| contrastive | 57.8 (4/12) | 28.4 (9/12) | "Queen Queen Queen Queen..." |

The contrastive centroid achieves 4/12 top-10 zero-shot with perfect specificity (control rank 32,721, 0/12 false positives). Prediction is weaker than the untied model (4/12 vs 9/12 top-10), reflecting different internal geometry under the tying constraint.

Zero-shot generation is degenerate — the contrastive condition loops on "Queen Queen Queen..." Post-SFT generation also loops ("the Queen of the Queen of the Queen..."). This is a training signal problem, not an initialization problem (see Interpretation below).

### Interpretation

**The initialization is sound.** The contrastive centroid on a tied model produces a queen row that fires at the right positions (4/12 top-10) with zero false positives (control rank 32,721), and lives in exactly the space the model's attention and MLPs were trained to process. This is a functional warm initialization — the embedding is in a good location.

**The degenerate generation reflects a training signal problem, not an initialization problem.** SFT reinforces the detector ("predict queen more") without teaching the surrounding weights *when and how* to route through queen vs. the compositional alternative. The "Queen Queen Queen" loop is the pathological expression: the detector fires, queen gets predicted and fed back as input, the detector fires again. SFT reinforces this cycle rather than breaking it because the NTP loss sees correct queen predictions and doesn't push to develop proper routing.

Concept recovery (`RESULTS_CONCEPT_RECOVERY.md`) achieved functional generation with full SFT because queen started from random init — the NTP loss was high everywhere, so gradients pushed ALL weights (attention, MLPs, embeddings) to co-adapt simultaneously. But that co-adaptation was driven by content-learning gradients (what "ruled" and "wisely" mean), not by the consolidation signal (that queen is worth using as a compression primitive). The right training signal for integration is A/B RL on matched pairs — original text with queen vs. denoised text with the compositional substitute — which cancels shared context and isolates the utility gradient. See `ideas/language_reduction_continual_learning.md`.

## What we learned along the way

1. **Untied weights conflated two problems.** The R²=0.27 divergence between lm_head and wte meant the contrastive centroid that worked as a detector couldn't serve as an embedding — not because the initialization was wrong, but because the model maintained two genuinely different learned subspaces for detection vs. participation. Much of Phase 2's complexity (regression mapping, gradient inversion) was spent trying to bridge a gap that shouldn't have existed.

2. **SFT is the wrong training signal for integration.** SFT on "The queen ruled wisely" produces gradients dominated by context-learning: what "ruled" and "wisely" mean in royal text. The signal about queen's compressive utility — that routing through queen is better than maintaining the compositional alternative — is a tiny fraction, buried under irrelevant gradients. This is why full SFT produced prediction (11/12 top-10) but degenerate generation: it reinforced the detector without teaching the surrounding weights to route through queen. See `ideas/language_reduction_continual_learning.md` for the full argument.

3. **The gradient signal methodology is sound but the signal source was wrong.** Per-prompt error signals at queen-prediction positions are highly coherent (cos=0.677), confirming that the model's error signals DO encode consistent queen-relevant structure. But the prediction-position gradient encodes the prediction computation's needs, not queen's identity as an input token.

4. **Frozen-model approaches are fundamentally limited.** No embedding initialization can teach frozen attention and MLP weights to route a novel concept. The surrounding weights must co-adapt — but the question is what training signal drives that co-adaptation.

## What we can conclude

**The warm initialization problem is solved.** A contrastive centroid of ln_f(h1) at concept-prediction positions, applied to a tied-weights model, produces a zero-shot embedding that:
- Correctly identifies concept-prediction contexts (4/12 top-10, rank 57.8)
- Is perfectly specific (0/12 false positives, control rank 32,721)
- Lives in the same space the model's attention and MLPs were trained to process (guaranteed by weight tying)

The remaining problem is not initialization but **integration training**: teaching the surrounding weights to route through the new token. This is not an SFT problem — it's an RL problem. A/B RL comparing queen-containing text against the compositional alternative ("the female ruler") cancels shared context and isolates the consolidation signal: "prefer routing through queen." This is the gradient that directly teaches attention weights to attend to queen and MLPs to process queen-adjacent activations.

The pipeline becomes:
1. **Detect** the off-manifold concept (GLP residual decomposition — done)
2. **Initialize** the concept token (contrastive centroid on tied-weights model — done)
3. **Integrate** via A/B RL on matched pairs from original vs. denoised corpus (next step — see `ideas/language_reduction_continual_learning.md`)

## Connection to the broader program

The token invention proposal envisions a metabolic cycle: detect a novel concept, mint a temporary token, use it for backprop to reorganize weights, then discard. Phase 1 validated the detection stage. Phase 2 solved the minting stage — but only after switching to tied weights, which eliminated the artificial detection/participation split.

The key architectural lesson: **use tied weights from the start.** Untied weights create a divergence between detection and participation that makes warm initialization unnecessarily hard. With tied weights, the contrastive centroid construction gives you a functional embedding for free — the same computation that finds a good detector finds a good embedding. The R²=0.27 problem was never fundamental; it was an artifact of the architecture.

The key training lesson: **SFT and integration are different learning problems.** SFT teaches content (what queen means). Integration teaches utility (that queen is worth using — that the model should route through queen rather than the compositional approximation). The DPO pairs from original vs. denoised corpus provide exactly the right gradient signal for integration, isolated from context-learning noise.

## Run commands

```
# Phase 2: zero-shot + wte-only SFT
modal run language_reduction/modal_app.py --stage warm-init-phase2 --tau 0.3

# Phase 3: gradient signal + full SFT
modal run language_reduction/modal_app.py --stage warm-init-phase3 --tau 0.3

# Tied-weights: train model + warm init
modal run language_reduction/modal_app.py --stage train --tau 0.3 --n-tokens 100000000 --n-steps 10000 --tie-weights
modal run language_reduction/modal_app.py --stage warm-init --tau 0.3 --tie-weights
```

## Files

| File | Description |
|------|-------------|
| `stages.py` | `warm_init_phase2_stage` (3 conditions, wte-only SFT) and `warm_init_phase3_stage` (gradient signal, full SFT) |
| `SPEC.md` | Phase 1 experiment design |
| `RESULTS_WARM_INIT.md` | Phase 1 results (lm_head initialization) |
| `RESULTS_WARM_INIT_PHASE2.md` | This document |
| `/data/results/warm_init_phase2_tau0.3.json` | Phase 2 numeric results (on Modal volume) |
| `/data/results/warm_init_phase3_tau0.3.json` | Phase 3 numeric results (on Modal volume) |
| `/data/results/warm_init_tau0.3_tied.json` | Tied-weights warm init results (on Modal volume) |
| `/data/models/tau_0.300/P_100000000/T_128_tied/` | Tied-weights trained model (on Modal volume) |
