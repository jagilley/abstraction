# Semantic Verbalization of Gradient Updates via GLP

**Status**: Idea (not yet implemented)
**Date**: 2026-04-13
**Builds on**: `residual_semantics`, `pc_verbalization`, GLP paper (flow matching, meta-neurons)
**Related literature**: `reading/gradient_interpretation_survey.md`

## One-liner

Use the GLP's learned coordinate system to semantically decompose what a gradient update is doing during fine-tuning, by separately verbalizing the activation component and the error signal component.

## The problem

During fine-tuning, it's difficult to tell what effect a particular gradient update will have on your model. Trivially, you're pulling the model toward the distribution of text you're fine-tuning on, but it's hard to tell what's actually happening under the hood. Existing tools either operate at the wrong granularity (task vectors describe the aggregate effect of an entire fine-tuning run), the wrong question (influence functions tell you *which training examples* caused a behavior, not *what concept* was learned), or the wrong modality (crosscoders/diff-SAEs compare before/after snapshots in activation space, post-hoc).

Nobody has built a system that takes an actual gradient update tensor and produces a semantic description of what it means.

## The key structural observation

For a single training example x with loss L, the gradient at layer l is:

```
nabla_W_l L = delta_l (outer product) h_{l-1}
```

where:
- **h_{l-1}** is the activation coming into layer l ("what the model is currently representing")
- **delta_l = dL/dz_l** is the error signal at layer l's pre-activation ("what the loss wants to change")

The gradient is their outer product: "change these weights so that when the model sees this activation pattern again, it produces more of what the loss is asking for."

Both h and delta live in R^d_model (the same vector space). The GLP already models the distribution of h. The question is whether delta has enough structure to be similarly modeled or at least interpreted in the GLP's coordinate system.

## GLP residual vs. gradient: they are genuinely different objects

The GLP residual for a data point says: "here's how this activation deviates from what the GLP considers typical." It's a property of the forward pass only — the model's *state* on this input.

The gradient for a data point says: "here's how the weights should change to reduce the loss on this input." It depends on the forward pass (h) AND the loss (delta). Two data points with identical activations but different labels produce identical residuals but different gradients.

**The residual captures what the model is doing; the gradient captures what the model is doing wrong.**

The divergence cases matter:

| Case | Residual | Gradient | Meaning |
|------|----------|----------|---------|
| On-manifold, correct | Small | Small | Model is typical and succeeding. Nothing to learn. |
| Off-manifold, correct | Large | Small | Model is atypical but succeeding. Unusual strategy that works. Fine-tuning won't change much. |
| On-manifold, wrong | Small | Large | **Model looks "normal" to the GLP but is failing.** Fine-tuning will push from a typical starting point in a new direction. |
| Off-manifold, wrong | Large | Large | Model is atypical and failing. |

The third case is the critical one: a gradient can demand changes that have no signature in the GLP residual, because the residual knows nothing about the loss function.

## Does the error signal have structure?

For activations, we know the answer is yes — the GLP proves it. Activations cluster on a learnable manifold because training imposes structure.

**Reasons to think delta has structure:**
- delta is a deterministic function of h (for fixed weights and loss), so if h has manifold structure, delta inherits some of it, filtered through the loss landscape
- DARE's 90-99% gradient sparsity finding means the outer product delta (outer) h is extremely low-rank in practice — delta and h are constrained to produce sparse, structured updates
- Safety-preserving fine-tuning work found alignment-relevant gradient components live in a low-rank subspace — direct evidence of exploitable structure in delta

**Reasons to worry:**
- delta depends on the loss function, which varies by task. The distribution of error signals during sentiment fine-tuning looks completely different from math fine-tuning. There may not be a single "error signal manifold"
- delta accumulates across layers during backprop, mixing information from all downstream layers, which could make it less locally interpretable at any given layer
- When used as a steering vector, delta might not correspond to a coherent behavioral mode. A residual amplifies what the model is already atypically doing (coherent by construction). An error signal represents a desired change that might be a superposition of many small adjustments that individually make sense but collectively don't describe any single behavior. Verbalizing it might require decomposing it into many steering passes along different components.

## Experimental results (Tier 1 — completed 2026-04-13)

See `experiments/error_signal_diagnostic/STATUS.md` for full results. Summary:

### Delta has rich structure
- Sentiment probe AUC: **0.9998** (h: 0.598). Near-perfect label separability.
- Participation ratio: 11.7 (comparable to h's 11.5). Not random noise.
- Multiple PCs carry label signal (PC0: r=0.596, PC1: r=-0.513, PC2: r=0.422).

### Delta is linearly orthogonal to h and the GLP residual
- cos(delta, h) = 0.001. cos(delta, GLP residual) = 0.006.
- Subspace overlap: 7-10% (near-random in R^2048).
- The "what needs to change" signal is linearly independent of "what the model represents."

### But the GLP's nonlinear geometry perfectly captures delta's semantics
- Velocity field response test: push h along delta, measure how the denoiser's velocity changes.
- **AUC = 1.0 at all noise levels** (u=0.3, 0.5, 0.7). Random control: ~0.6.
- The manifold's curvature distinguishes sentiment-relevant directions in delta space.
- The relationship between delta and the GLP is entirely nonlinear.

### Key insight
Linear probes say the GLP can't see delta. The nonlinear velocity field says the GLP sees delta perfectly. The model's existing features CAN express what the gradient asks for, but only through the manifold's curvature, not its tangent space.

### Caveat
The model gets 0% accuracy on this task (predicts " " 185/200 times). Every gradient is "learn from scratch." More realistic fine-tuning settings (partial accuracy) would produce more nuanced gradient structure.

## Proposed experimental path

### Tier 1: Diagnostic — DONE

See above and `experiments/error_signal_diagnostic/STATUS.md`.

### Tier 2: GLP meta-neuron analysis — SKIPPED

Obviated by `beliefs/meta_neurons_are_a_weak_window.md`: meta-neurons are features of the circuit that computes v, not atomic units of what the GLP knows. Any linear readout of the circuit (including meta-neuron projections of δ) is a lossy projection of the velocity field's learned geometry. Tier 1 itself demonstrated this: every linear readout of δ found AUC ≈ 0 vs. AUC = 1.0 for the full velocity-field response. The velocity field's *output*, not its internal features, is the right object.

### Tier 3: Verbalization — DONE 2026-04-17

See `experiments/gradient_verbalization/STATUS.md`. Headline: the **gradient-descent direction** (−δ) verbalizes cleanly on SST-2 — 19/20 neutral baselines shift to label-aligned completions that literally contain the target tokens. The raw gradient (+δ) and the velocity response (+response to +αδ) both point *away* from the target and produce ambiguous/incoherent shifts (~10% attribution). Once signs are aligned, `-response` ≈ `-δ` (description cos 0.89 — dramatically higher than any other cross-pair). SST-2 is ceiling-saturated (~95–100%) by polarity steering; the oracle contrast direction and −δ both hit the ceiling, leaving no headroom for response to add value *on this task*.

Key methodological finding: the SPEC's original framing steered with +δ under the reading "gradient as a vector points at a semantic target." That reading was wrong — δ is the loss gradient, so it points *away* from the target. Once flipped to −δ (the descent direction), verbalization becomes clean. This revised framing aligns with the natural question "what would the model become under this update?" — the answer is what −δ does, not what +δ does.

### Tier 4: Monitoring (the applied version)

If verbalization works, build a monitoring tool: during fine-tuning, periodically extract delta_l, project into the GLP's semantic coordinates, and produce running natural-language commentary on what the model is learning at each stage. Track which semantic axes are being pushed and how strongly. Detect transitions (e.g., "the model has shifted from learning surface-level token associations to learning deeper structural patterns").

## What exists in the literature (key references)

**Task vectors** (Ilharco et al., ICLR 2023): Weight delta W_ft - W_pretrained is semantically meaningful and supports arithmetic. The aggregate integral of gradients is interpretable; the question is whether the derivative (individual steps) is too.

**Tangent space analysis** (Ortiz-Jimenez et al., NeurIPS 2023): Different tasks produce approximately orthogonal weight changes. Provides theoretical grounding for why task-level gradient decomposition works.

**DARE** (Yu et al., 2024): 90-99% of fine-tuning delta parameters can be dropped. Extreme sparsity suggests the effective gradient is very low-dimensional.

**Influence functions at scale** (Anthropic, 2023; TrackStar, 2025): Gradient-based attribution traces predictions to training data. Layer-wise structure: bottom/top layers capture wording, middle layers generalize thematically. Only ~50% of most-influential examples for a fact even contain the fact.

**Crosscoders/diff-SAEs** (Anthropic, 2025): Identify features that emerge during fine-tuning using shared sparse dictionaries. Post-hoc activation-space comparison, not per-step.

**Circuit analysis of fine-tuning** (ICML 2025): Fine-tuning changes edges not nodes — same parts, different wiring.

**weights2weights** (NeurIPS 2024): Weight space of fine-tuned models has navigable semantic structure with interpretable linear directions. Image models only.

**Safety-preserving fine-tuning**: Alignment-relevant gradients live in a low-rank subspace. Removing gradient components that conflict with the safety subspace preserves alignment during task fine-tuning.

**TextGrad** (Stanford, 2024): Natural language as gradient metaphor for optimization. Uses LLM critiques as "textual gradients." The metaphor without the actual tensors.

**Empirical influence functions** (Matelsky et al., 2024): Sobering result — influence functions violate expected logical consistency desiderata during fine-tuning. The gradient updates don't produce the clean compositional semantic effects one might hope for.

Full survey with citations: `reading/gradient_interpretation_survey.md`.

## Key risks and open questions

1. **Resolution**: At what temporal granularity do gradient updates become semantically interpretable? Task vectors (the integral) clearly work. Does the derivative (per-step gradient) have enough signal-to-noise? Per-batch is more promising than per-example.

2. **Superposition in delta**: The error signal might be a superposition of many small semantic changes that resist single-vector verbalization. The PC verbalization transfer test's marginal result (27.5%, p=0.066) on narrow factual prompts is cautionary — even for GLP residuals, semantic axes can be fragile. For error signals, which are less constrained, the problem may be worse.

3. **Task dependence**: Unlike activations (which have a universal distribution across diverse text), error signals depend on the loss function and training data distribution. A "gradient GLP" might need to be task-family-specific.

4. **GLP staleness**: The GLP was trained on pre-trained model activations. As fine-tuning progresses, the activation distribution shifts. Initially this is useful (the GLP acts as a fixed reference frame, and growing residuals measure what changed). But eventually the GLP becomes too stale to provide meaningful coordinates. Periodic retraining — the hippocampal consolidation analogy from the self-intervention idea — would be needed for long fine-tuning runs.

5. **The Matelsky warning**: If fine-tuning doesn't produce logically consistent semantic effects (the empirical influence functions result), then the gradient may genuinely resist semantic verbalization — not because our tools are inadequate, but because the underlying object doesn't have clean semantic structure at the per-step level.

## Connection to other ideas in this repo

- **Self-intervention via local replay**: That idea uses GLP residuals for novelty detection and targeted belief revision. Gradient verbalization is the complementary view: instead of the GLP detecting what's unusual about the model's current state, it describes what the training signal is asking the model to become.

- **Residual semantics**: Establishes that GLP residuals are verbalizable (CKA = 0.446-0.554, p=0.0001; revised negation test 36.1% vs 20% chance, p=0.018). The pipeline (steer + describe delta) is directly reusable for error signal verbalization.

- **PC verbalization**: Shows residual PCA produces interpretable semantic axes (epistemic, discourse fluency, cultural/experiential, institutional elaboration, authoritative confidence). The same PCA + verbalization approach could be applied to error signals or to the *change in residual distribution* during fine-tuning.
