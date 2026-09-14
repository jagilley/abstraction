# Results: GLP Embedding-Space Initialization and In-Context Grokking

**Date**: 2026-05-02
**Builds on**: `RESULTS_WARM_INIT_PHASE2.md`, `RESULTS_RESIDUAL_DECOMPOSITION.md`, `SPEC_GLP_EMBEDDING_SPACE.md`
**Motivation**: `conversations/Claude-Learned token invention in language models.md`
**Status**: Complete. Contrastive centroid remains the best supervised initialization. Both new approaches yield informative negative results that clarify the relationship between activation geometry and embedding initialization.

## Question

The contrastive centroid (centroid of queen-prediction activations minus control activations) achieves zero-shot queen rank 57.8 (4/12 top-10) on the tied-weights model. Can we do better by using richer geometric or contextual information?

Two approaches tested:
1. **GLP embedding-space residual**: Train a GLP on ln_f(h1) from the tied-weights model so it operates natively in embedding space. Use the off-manifold residual at queen-prediction positions as the initialization.
2. **In-context grokking**: Prime the model with female-royalty-rich context to temporarily compose gender and royalty, then capture the activation signature of that composition.

## Answer

Neither approach beats the contrastive centroid for supervised embedding initialization. Both fail for different and informative reasons:

- The **GLP residual** captures position-specific off-manifold structure (per-prompt coherence 0.11), not the concept-specific direction. It answers "what's unusual about this activation?" rather than "what concept is missing?"
- The **ICL signature** is highly coherent (0.91) but points in the wrong direction — it captures a readout-space shift ("this context is about female royalty") rather than an embedding-space input vector.

The contrastive centroid's advantage is that it directly measures activations at positions where the model is trying to output queen, then subtracts generic context. It's a targeted extraction of the discriminative direction, which is precisely what a tied-weights embedding needs to be.

## Experiment 1: GLP in embedding space

### Background

The existing GLP trained on [h0; h1] — a 256-D concatenation of both block outputs — creates a dimensionality mismatch with the 128-D embedding space. Three conditions dissolve this mismatch simultaneously:

1. **Single-layer GLP** (not concatenation) — 128-D features matching embedding dim
2. **Train on ln_f(h1)** (not raw h1) — the representation that interfaces with lm_head
3. **Tied weights** (lm_head = wte) — makes output space identical to embedding space

With all three, GLP outputs live natively in embedding space. Residuals are candidate embedding vectors without any projection step.

### GLP training

Trained a flow-matching GLP (1.78M params, 3-layer SwiGLU denoiser) on 1M ln_f(h1) activations from the tied-weights model (tau=0.3, P=100M, block_size=128).

| Metric | Value |
|--------|-------|
| Final loss | 1.096 |
| Reconstruction cosine | 0.961 |
| Mean residual norm | 6.621 |
| Mean max cos(residual, wte_row) | 0.350 |

The manifold is well-learned (0.96 reconstruction cosine). The 0.35 mean max cosine between residuals and wte rows confirms residuals are geometrically meaningful in embedding space.

### GLP residual extraction

At each of 12 queen-prediction positions: extract ln_f(h1), project onto manifold via 10 stochastic denoising passes (t_start=0.3), take residual = activation - projection. Average across passes and prompts, scale to mean wte norm.

### Results

| Condition | Zero-shot rank | Top-10 | Control rank | Post-SFT rank | Post-SFT top-10 | Best val loss |
|---|---|---|---|---|---|---|
| random | 947.4 | 0/12 | 2,301 | 138.6 | 5/12 | 5.766 |
| analogy | 524.2 | 0/12 | 2,534 | 142.0 | 5/12 | 5.766 |
| **contrastive** | **57.8** | **4/12** | **32,721** | **28.4** | **9/12** | 5.777 |
| glp_residual | 662.5 | 0/12 | 4,928 | 82.8 | 7/12 | **5.759** |

### Diagnostics

| Metric | Value |
|--------|-------|
| Per-prompt pairwise cosine | 0.111 |
| Coherence ratio | 0.428 |
| cos(glp_residual, contrastive) | +0.241 |
| cos(glp_residual, king_wte) | -0.068 |

### Interpretation

The GLP residual captures only 24% of the contrastive direction. Per-prompt coherence of 0.11 means the 12 residuals point in diverse directions — each one captures what's unusual about *that specific context*, not the shared queen concept. Averaging 12 weakly-aligned vectors produces a weak signal.

However, two findings suggest the GLP residual carries useful information:
- **Best val loss** (5.759 vs contrastive's 5.777) — the GLP initialization may find a better loss basin
- **Still improving at step 50** — queen rank went 479→73 and was still dropping, while contrastive plateaued at ~18 early
- **7/12 top-10 post-SFT** — not far behind contrastive's 9/12, despite starting much worse

The GLP's advantage is in the **unsupervised case**: it doesn't require hand-selected queen prompts and control prompts. The contrastive centroid is a supervised operation. For the continual learning program, where the goal is to detect and mint concepts automatically, the GLP manifold can identify off-manifold positions without knowing what concept it's looking for.

## Experiment 2: In-context grokking

### Background

The residual decomposition experiment showed the model has gender and royalty as separate internal concepts, with queen positions exhibiting a diffuse entanglement of both. The hypothesis: in-context learning can temporarily bind these concepts, producing a richer composed representation than bare prompts yield. Capturing that composed activation gives a more targeted embedding initialization.

### Design: 2x2 factorial priming

Four priming prefixes (~55-60 tokens each), crossing gender (female/male) with royalty (royal/non-royal):

| | Royal | Non-Royal |
|---|---|---|
| **Female** | "The wife of the king sat upon her throne..." | "The mother worked in her garden..." |
| **Male** | "The king sat upon his throne..." | "The farmer worked in his field..." |

Each prefix is prepended to all 12 queen-prediction prompts. The factorial design enables isolating:
- **ICL primed** = female_royal - neutral (raw ICL effect)
- **Interaction** = female_royal - male_royal - female_nonroyal + neutral (pure gender x royalty binding)
- Marginal gender and royalty effects

### Results

| Condition | Zero-shot rank | Top-10 | Control rank | Post-SFT rank | Post-SFT top-10 |
|---|---|---|---|---|---|
| random | 947.4 | 0/12 | 2,301 | 138.6 | 5/12 |
| **contrastive** | **57.8** | **4/12** | **32,721** | **28.4** | **9/12** |
| icl_primed | 2,524 | 0/12 | 20,706 | 98.8 | 0/12 |
| interaction | 1,829 | 0/12 | 10,851 | 432.6 | 0/12 |

### Key diagnostics

**Priming works at the model level.** Before any embedding modification, female-royal priming cuts queen rank from 196 to 77 on the first 4 prompts. The model's attention mechanism IS composing gender and royalty when given the right context. The "in-context grokking" is happening.

**ICL signatures are highly coherent.**

| Signature | Pairwise cosine | Coherence ratio |
|-----------|----------------|-----------------|
| icl_primed | **0.907** | 0.958 |
| interaction | **0.813** | 0.912 |
| contrastive (bare) | 0.431 | 0.687 |
| glp_residual (Exp. 1) | 0.111 | 0.428 |

The ICL signal is consistent across all 12 prompts (0.91), far exceeding both the contrastive centroid (0.43) and the GLP residual (0.11). The priming elicits the same compositional representation at every queen-prediction position.

**But the direction is wrong for embedding initialization.**

| | cos vs contrastive |
|---|---|
| icl_primed | +0.385 |
| interaction | +0.228 |
| glp_residual (Exp. 1) | +0.241 |

All three alternative approaches capture only 23-39% of the contrastive direction.

**Factorial decomposition confirms orthogonal structure.**

| Effect | Norm | Interpretation |
|--------|------|---------------|
| Royal | 9.530 | Largest single factor |
| Interaction (female x royal) | 5.435 | Gender-royalty binding exists |
| Female | 5.016 | Smaller than royalty |
| cos(female, royal) | -0.132 | Nearly orthogonal |

The model stores gender and royalty in orthogonal subspaces. The interaction effect is real (norm 5.4) but smaller than either main effect, consistent with the residual decomposition's finding that queen lives at their diffuse intersection.

### Interpretation

The ICL experiment reveals a distinction between two functional roles of vectors in tied-weights embedding space:

- **Readout direction**: "what is this context about?" — the ICL signature captures this, explaining why it's coherent (the model consistently recognizes female-royalty contexts)
- **Input embedding**: "what should this token contribute to computation?" — the contrastive centroid captures this, explaining why it initializes better

With tied weights, these are nominally the same vector space, but they serve different computational roles. The ICL difference (primed - unprimed) measures how the *output representation shifts* when the model processes female-royalty context. The contrastive centroid measures what the model's output representation *looks like* when it's actively trying to predict queen. The latter is a better embedding because it captures the representation the model produces at the exact functional moment when queen would be generated.

## What both experiments reveal

The contrastive centroid works not because it's mathematically sophisticated — it's just centroid subtraction. It works because it measures the right thing: the model's output representation at positions where queen should appear, with generic English subtracted. This is precisely the vector that, in a tied-weights model, should serve as both the queen detector and the queen embedding.

The GLP residual and ICL signature both capture real structure:
- The GLP residual captures per-prompt compositional structure (gender x royalty proportions matching prompt semantics — shown in the residual decomposition)
- The ICL signature captures coherent contextual composition (the model binding gender and royalty through attention)

But neither is aligned with the discriminative direction the contrastive centroid extracts. They answer different questions:

| Approach | Question answered | Coherence | Alignment with queen |
|----------|------------------|-----------|---------------------|
| Contrastive centroid | "What do queen-prediction activations look like vs. generic?" | 0.43 | 1.00 (by definition) |
| GLP residual | "What's off-manifold at this position?" | 0.11 | 0.24 |
| ICL signature | "How does the output shift when primed with female royalty?" | 0.91 | 0.39 |

The GLP's advantage lies not in supervised initialization but in **unsupervised concept detection** — identifying off-manifold positions automatically without knowing what concept to look for. This is Phase 3 of the embedding-space GLP spec.

The ICL approach's advantage may emerge in settings where you can provide the model with genuine examples of the target concept in context (not circumlocutions). In our setup, queen is removed from the vocabulary entirely, so the model can't form a true queen representation — only approximations through compositional priming. With a model that has the concept in its vocabulary but lacks a dedicated token, ICL priming with real examples might produce a signature that is both coherent AND well-aligned with the discriminative direction.

### Why the ICL signature and contrastive centroid diverge geometrically

The 0.39 cosine between the ICL signature and contrastive centroid reflects a granularity mismatch: the ICL signature operates at the level of a **semantic field**, while the contrastive centroid operates at the level of a **specific token**.

Consider the token embeddings for queen, princess, she, her, throne, crown as a cluster in R^128. They share a "royalty/female" subspace but each has unique components. The ICL signature points approximately at the **centroid of this cluster** — it captures "female royalty" as a category, the direction that simultaneously boosts logits for all tokens in the field. The contrastive centroid points at **queen specifically** — one member of the cluster, offset from the field centroid by queen's unique components.

The factorial decomposition (interaction = female_royal - male_royal - female_nonroyal + neutral) is the natural attempt to narrow from field to token by cancelling main effects. It fails — cosine with contrastive drops from 0.385 to 0.228 — for two reasons: (1) the transformer's attention-mediated binding of gender and royalty is nonlinear, so additive subtraction doesn't cleanly factor; (2) even a perfect gender×royalty interaction gives you "female royal" (the concept), not "queen" (the token) — princess is also a female royal.

This connects to Walker et al.'s Linear Centroids Hypothesis (2025): the LCH's centroids correspond to the network's Voronoi tiling at the resolution of individual predictions (token-level). The ICL signature is effectively a centroid at semantic-field resolution. The LCH would predict the finer-grained centroid is more faithful to the computation, which is what we observe. Deriving the token-level centroid from the field-level centroid requires traversing the nonlinear structure of the vocabulary embedding space — no linear decomposition suffices.

## Implementation

- GLP training: `modal run language_reduction/modal_app.py --stage train-glp-embed --tau 0.3`
- GLP initialization: `modal run language_reduction/modal_app.py --stage warm-init-glp-embed --tau 0.3`
- ICL initialization: `modal run language_reduction/modal_app.py --stage warm-init-icl --tau 0.3`
- Code: `language_reduction/experiments/warm_init/stages.py` (warm_init_glp_embed_stage, warm_init_icl_stage)
- GLP extraction: `language_reduction/glp.py` (extract_lnf_h1_activations)
- GLP training stage: `language_reduction/experiments/glp_analysis/stages.py` (train_glp_embed_stage)
- Raw results: `/data/results/warm_init_glp_embed_tau0.3.json`, `/data/results/warm_init_icl_tau0.3.json` on Modal volume
