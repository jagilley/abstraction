# The Nonlinear Legibility of the GLP

*April 2026*

## The empirical picture

Across six months of experiments probing what GLP residuals "mean," a pattern has emerged: linear tools systematically underestimate the semantic content of the GLP, while nonlinear tools reveal rich structure. The gap between these two lenses is not small — it is the difference between null results and perfect classification.

### What linear analysis finds

| Experiment | Linear tool | Result |
|---|---|---|
| Epistemic probe v3 | PCA on residuals | r = -0.31 with correctness — **domain confound** (Simpson's paradox; PC0 separated prompt format, not knowledge) |
| Epistemic probe v5 | Steering with PC0 | No better than random directions — **generic perturbation sensitivity** |
| Residual semantics | Mantel test (cosine similarity transfer) | r = 0.085 (global); nulls for most categories |
| PCA on diverse residuals | Transfer test | 21% on 5-way = chance |
| Error signal diagnostic | cos(delta, GLP residual) | 0.006 — **orthogonal** |
| Error signal diagnostic | Subspace overlap (h vs delta) | ~10% |

Linear tools on the GLP produce nulls, confounds, or modest effects. The strongest linear result is CKA (0.45-0.55, p < 0.001), which captures more structure than cosine similarity but is still a linear kernel alignment.

### What nonlinear analysis finds

| Experiment | Nonlinear tool | Result |
|---|---|---|
| Error signal diagnostic | Velocity field response to delta perturbation | **AUC = 1.0** at all noise levels (vs 0.6 for random) |
| Emotion injection | Meta-neuron probes (1-D on nonlinear features) | 0.72-0.85 AUC; robust to surface manipulation; correct injection amplifies behavior 2.7x vs random |
| PC verbalization | Between-category centroids + discriminative summarization | 33.1% on 5-way (p < 10^-7); two directions at 50% |
| Residual semantics (revised) | Within-prompt negation test | 36.1% vs 20% chance (p = 0.018) |

The velocity field result is the starkest demonstration: the error signal (gradient of the loss) is linearly orthogonal to everything the GLP represents (cosine ~ 0, subspace overlap ~ 10%), yet the manifold's nonlinear response to a delta-direction perturbation separates sentiment with AUC = 1.0. The information is encoded in curvature, not in any extractable direction.

## The interpretation

The GLP manifold encodes semantic structure at three tiers of accessibility:

**Tier 1 — Linearly accessible, globally transferable.** Almost nothing lives here. The only signal that transfers globally through linear tools is coarse category membership (factual vs. creative prompt type). Fine-grained globally-transferable linear directions do not appear to exist. Attempts to find them (epistemic probe PC0, naive PCA on diverse residuals) produce either confounds or nulls.

**Tier 2 — Linearly accessible, locally meaningful.** Individual GLP residuals encode prompt-specific semantic axes that can be verbalized and whose negations produce recognizable opposites. CKA confirms geometry-semantics correspondence. Emotion meta-neurons carry genuine, content-driven signal. But these signals are local: they describe how *this specific activation* deviates from typicality, and they don't transfer between prompts.

**Tier 3 — Nonlinearly accessible.** The GLP's full geometry — its velocity field, its curvature — captures semantic structure that is completely invisible to linear analysis. The error signal diagnostic proves this: perfect sentiment separation through the velocity field, zero through any linear measure. The manifold "knows" things that no PCA, cosine similarity, or linear probe can extract.

The gap between Tier 2 and Tier 3 is where most of the GLP's semantic content lives. And it is the gap that most of our experimental pipeline has been unable to access, because the standard toolkit — PCA, cosine similarity, linear probes, fixed-direction steering — is linear.

## The lesson

The early phases of the epistemic probe were over-optimistic not because the GLP lacks semantic content, but because PCA returned a clean-looking number (r = -0.31) that turned out to be a confound. The temptation to interpret PCA components as meaningful semantic axes is strong because PCA is the default tool and its outputs have a satisfying crispness. But for a nonlinear generative model operating on a curved manifold, PCA finds the largest axis of *linear* variation, which may have nothing to do with the semantically important structure.

Every false positive in this research program came from a linear tool finding a confound:
- PC0 epistemic correlation → prompt format confound
- Original negation test (78.7%) → topic matching confound
- Causal steering via PC0 → generic perturbation sensitivity (any direction works equally)

Every robust positive came from either a nonlinear readout or a carefully controlled local analysis:
- Velocity field response → AUC 1.0
- Emotion meta-neurons → robust to surface manipulation
- Within-prompt negation test → 36.1% (weak but real, and the right test)
- Between-category centroids → 33.1% (coarse but genuine)

The principle: **query the GLP nonlinearly, or accept that you're seeing at most a shadow of what it knows.**

This principle holds, but two follow-up experiments (2026-04-15 and 2026-04-16) revealed that the magnitude of the nonlinear uplift varies dramatically across tasks — and the variation is patterned.

## When the velocity field helps: two conditions

The sentiment error signal diagnostic's 0.5 → 1.0 AUC jump is a *spectacular special case*, not a generic illustration of what nonlinear tools add. Two subsequent experiments isolated two conditions that together predict how much the velocity field adds over linear probes.

### Condition 1: The perturbation direction must be orthogonal to the GLP's existing representation

**Velocity field semantics (null).** We computed velocity responses to perturbations in the residual direction and measured CKA against description embeddings (baseline from the residual_semantics V2 experiment).

| Condition | CKA | p-value |
|---|---|---|
| Raw residual | 0.4462 | 0.0001 |
| Velocity response (residual direction) | 0.4518 | 0.0001 |
| Velocity response (random direction) | 0.4048 | 0.5092 |

The velocity response to residual-direction perturbation is CKA-equivalent to the raw residual. No uplift. Random-direction response is at chance, confirming the readout is direction-specific — but the velocity field doesn't add information beyond what the residual already carries in this configuration.

*Why:* The residual IS the velocity field's prediction error (`predicted − target`). It's already the output of the full nonlinear denoiser computation. Perturbing along the residual and measuring the velocity response feeds the GLP's own error signal back into itself — asking for a second opinion on something it already gave its best assessment of. For the velocity field to surface new information, the perturbation must point somewhere the existing representation doesn't already capture.

The error signal diagnostic hit AUC=1.0 because the correctness gradient was *literally* orthogonal to h and the residual (cos ≈ 0.006). The manifold's curvature decoded sentiment structure that lived nowhere in the tangent or residual space.

### Condition 2: The label must live in the direction of the perturbation, not its magnitude

**Nonlinear epistemic probe (partial uplift).** We applied the same velocity field response protocol to factual correctness instead of sentiment. 427 factual QA prompts with single-token answers and per-prompt `is_correct` labels from Llama-1B argmax.

| Method | AUC (full, N=427) | AUC (subset, N=113) |
|---|---|---|
| Linear probe on delta direction (PCA top-10) | 0.647 | 0.646 |
| Velocity response to delta_hat | **0.747** | **0.681** |
| Velocity response to random (control) | 0.554 | 0.457 |
| ‖delta‖ alone (univariate) | 0.897 | 0.919 |
| Loss alone (sanity ceiling) | 0.974 | 0.975 |

The velocity response beats the linear delta-direction probe by +0.10 on the full set, shrinking to +0.03 (within statistical noise) on the within-category-variable subset that neutralizes the category shortcut. Still delta-specific throughout — random-direction velocity responses stay near chance.

*Why the uplift is so much smaller than sentiment's 0.5 → 1.0 gap:*

In sentiment, both conditions are fully met. The correctness gradient was orthogonal (Llama-1B got 0/200 on SST-2, so every delta pointed into activation-space regions the model had never entered — condition 1 satisfied by the task being fully out-of-distribution). And the label lived in delta's direction: all positive-class prompts shared the target token `" positive"` → all positive deltas pointed roughly the same way → delta-direction IS the label. Two clusters in delta-space, velocity field reads them out perfectly.

In factual correctness, condition 1 is only partially met — Llama-1B partially knows these facts, so delta direction overlaps some with existing structure (the linear delta-direction probe already gets 0.65). And condition 2 is **not** met: every prompt has a different target token (Paris, Tokyo, Au, Canberra, …), so `is_correct = 1` prompts don't share a direction in delta-space. "Is correct" is fundamentally a magnitude question — was h already close enough to prompt i's specific target that argmax matched? That's why `‖delta‖` alone gets 0.90 (gradient magnitude directly answers the label's question via the norm-loss tautology), and why the velocity response to `delta_hat` (which strips out magnitude) only gets 0.75. The velocity field can't manufacture shared directional structure when the label has none.

### Putting the conditions together

| Signal | Perturbation orthogonal to GLP rep? | Label direction-coded? | Nonlinear uplift observed |
|---|---|---|---|
| Sentiment error signal (SST-2, base Llama) | Yes (cos ≈ 0.006) | Yes (2 target tokens = 2 directions) | Massive (0.50 → 1.00) |
| Factual correctness gradient | Partially (linear baseline 0.65) | No (heterogeneous targets, magnitude-dominated label) | Modest (0.65 → 0.75) |
| Residual-direction perturbation | No (residual IS GLP output) | N/A | None (0.45 → 0.45) |

**The refined principle:** the velocity field is a bridge tool that decodes information orthogonal to the GLP's existing representation, and the amount decoded scales with how much of the label's signal lives in the direction of the perturbation versus its magnitude or unrelated structure. Querying the GLP nonlinearly is still the right default, but expecting 1.0 AUC on arbitrary tasks is wrong. For signals the model partially represents (most practical tasks), modest uplift is the realistic expectation.

One corollary worth naming: sentiment's orthogonality itself was task-driven, not an intrinsic property of "error signals are orthogonal." Llama-1B couldn't do SST-2 at all (0/200), so every gradient was "learn alien target." For an instruction-tuned model that partially understands the same task, condition 1 would weaken. The sentiment result is therefore more specifically: "base model + out-of-distribution task + class-coded targets → maximum nonlinear uplift."

## Connection to manifold locality

This belief is complementary to, not a replacement for, [manifold locality](manifold_locality.md). Manifold locality says the GLP's structure is rich locally but coarse globally — meaning is context-dependent and doesn't transfer across prompts. Nonlinear legibility says the GLP's structure is richer nonlinearly than linearly — meaning exists but is encoded in curvature, not in extractable directions.

Together they say: **the GLP manifold is a nonlinear, locally-structured object. Meaning is encoded in the shape of the manifold at each point, not in global linear directions. The right tools are local and nonlinear; the wrong tools are global and linear.**

The verbalization pipeline (extract residual, steer, describe behavioral effect in language) succeeds because it is local — it characterizes one residual at a time — and because language provides a coordinate system for comparing across contexts without requiring geometric transfer. It sidesteps the linearity problem by going through behavior rather than through geometry.

## The open question

Is the dominance of nonlinear structure a property of this GLP (3-layer MLP denoiser, Llama 1B), or something fundamental about how generative models learn manifold geometry?

Arguments for scale-dependent: Llama 1B has relatively fractured representations (FER). A larger model with more coherent internal representations might produce a GLP whose manifold has more linearly-accessible global structure. The path to testing this — repeat with Llama 8B + glp-llama8b-d6 — is straightforward.

Arguments for fundamental: The velocity field result involves the error signal, which is linearly orthogonal to the training distribution by construction (it points toward unseen labels). The fact that the manifold's curvature captures this out-of-distribution direction suggests something deeper than memorized clusters — the geometry has learned abstract structural relationships that generalize beyond the training data. If this is intrinsic to how flow matching learns manifold geometry, it would hold at any scale.

Either way, the methodological implication is clear: for information the GLP has already expressed (residuals), the residual itself is the best readout — it's already a nonlinear computation's output. For information orthogonal to the GLP's representation (gradients, novel perturbation directions), the velocity field response is the right tool, with the size of the uplift depending on whether the label's geometry aligns with the perturbation's direction. Linear decomposition of residuals (PCA, cosine similarity) remains the wrong approach in both cases.

## Summary of evidence

| Experiment | Linear result | Nonlinear result | Gap / interpretation |
|---|---|---|---|
| Error signal diagnostic (sentiment) | cos(delta, residual) = 0.006 | Velocity AUC = 1.0 | Massive: both orthogonality and direction-coded label |
| Nonlinear epistemic probe (factual) | Linear delta-direction probe AUC = 0.65 | Velocity response AUC = 0.75 (full), 0.68 (subset) | Modest: partial orthogonality, magnitude-dominated label |
| Velocity field semantics (residual dir) | CKA(residual, embeddings) = 0.446 | CKA(velocity_response, embeddings) = 0.452 | None: residual IS the nonlinear output |
| Epistemic probe v3 | PC0 r = -0.31 (confound) | Not tested nonlinearly at the time | Linear: false positive |
| Emotion injection | — | Meta-neuron AUC = 0.72-0.85; correct injection 2.7x amplification | Nonlinear features carry genuine emotion signal |
| Residual semantics | Mantel r = 0.085 | CKA = 0.45-0.55; revised negation 36.1% | Linear kernel: modest; full kernel: stronger |
| PC verbalization | Factual PCA transfer: 27.5% (p = 0.066) | Centroid SVD transfer: 33.1% (p < 10^-7) | Method choice > data choice |
