# Local Prediction-Error Learning

**Status**: Proposed (grounded in validated A2A forward model results + predictive coding theory)
**Date**: 2026-06-14
**Builds on**: [activation_to_activation_forward.md](activation_to_activation_forward.md), [cerebellar_abstraction_ratchet.md](cerebellar_abstraction_ratchet.md)
**Validated components**: Wake-sleep distillation (MNIST: 2-cycle ratchet, 34× robustness), innovation map / prediction trust (language), OOD robustness (distribution-invariant perturbation resistance)

## One-liner

Use the forward model's prediction errors as local learning signals for intermediate layers, replacing part of backprop's credit assignment with structured, self-knowledge-aware supervision — turning the forward model from a passive observer into an active teacher.

## Motivation: the credit assignment gap

The A2A forward model system produces genuine self-knowledge: the model develops an innovation map (directional structure of what's routine vs novel in its own computation), perturbation robustness (2-2.5× in language, 4× in MNIST), and structured representations that survive distribution shift. Wake-sleep distillation internalizes this knowledge into the model's weights, producing compounding improvements across cycles (MNIST: 34× robustness, 97.8% → 98.75% accuracy over 2 cycles).

But the self-knowledge is currently a *byproduct* of training — it emerges implicitly from the injection and is internalized via discrete distillation phases. The model knows WHERE its computation is surprising (the innovation map) but doesn't use this knowledge to direct its OWN learning. Backprop provides the weight updates, and backprop is blind to the self-knowledge.

Concretely: when the model encounters a training example, backprop computes ∂L_NTP/∂θ for every parameter. This gradient is output-referenced — it says "move these weights to reduce the final loss." It does not say "your block 1 computation was surprising in the delimiter-matching direction, so preferentially update the circuit responsible for delimiter matching." The self-knowledge is in the representations but not in the learning rule.

This matters for generalization. Ilya Sutskever's analogy (Dwarkesh Podcast, 2025): the 10,000-hour competitive programmer updates all parameters based on competition loss. The 100-hour student with "it" recognizes what's routine (already known) vs novel (worth learning from) in each problem and allocates learning accordingly. The self-knowledge guides resource allocation during learning, not just during inference. Our models currently have the self-knowledge but not the learning-rule integration.

## The proposal

### Core mechanism

Add forward models at one or more depth points in the network. Each forward model predicts the main model's activations several layers ahead from the current layer's activations. The prediction error — what the next layers actually computed minus what the forward model predicted — becomes a local learning signal for the intermediate layers.

For a model with blocks 0–3 and two forward models:

```
FM_a: post_block0 → predicts post_block1  (1-layer span)
FM_b: post_block1 → predicts post_block3  (2-layer span)
```

Each block's training objective becomes:

```
L_total = L_NTP + λ_a · L_local_a + λ_b · L_local_b
```

where L_local_i is the prediction error of the FM spanning that block. The local loss gradient flows through the FM (which provides the predicted target) into the spanned blocks' weights, without traversing the full network depth.

Each block is under two pressures simultaneously:
1. **Be useful for the task** (NTP loss, from the global backward pass)
2. **Be predictable to the forward model above you** (local FM loss)

These pressures are complementary, not competing. The MNIST wake-sleep results empirically demonstrated this: making computation more FM-predictable (eta² collapse from 0.39 to 0.03) simultaneously improved task performance (accuracy 97.8% → 98.75%). The model absorbed digit-discriminative computation into a form that's compressible by a small model, and this reorganization was beneficial.

### What the local loss provides that backprop doesn't

**1. Higher-dimensional supervision per training example.**

NTP provides one scalar loss per token. The gradient ∂L/∂h_i at layer i is a 256-d vector, but it's derived from this single scalar and computed through the chain rule across all downstream layers — output-referenced, entangled with downstream computation.

Each FM prediction error is a 256-d vector that answers a fundamentally different question: "what did the next k layers compute that a compressed model of my own dynamics couldn't anticipate?" This is process-referenced — it's about the character of the computation itself. The behavioral residual analysis showed these errors decompose into meaningful computational features (delimiter tracking d=+0.84, sentence-start processing d=-0.85, focused vs distributed attention d=-0.61).

Two inputs can produce identical NTP loss but very different FM errors — one surprising because of novel delimiter patterns, another because of unusual semantic composition. The NTP gradient can't distinguish these; the FM error can. Each FM adds 256 dimensions of structured supervision per token, informationally independent of the NTP gradient.

**2. Depth-localized credit assignment.**

When the model predicts the wrong next token, backprop gives every layer a gradient that says "contribute to fixing this." With FM prediction errors:

- If FM_a (blocks 0→1) has low error: blocks 0-1 did something routine. The mistake probably isn't here.
- If FM_b (blocks 1→3) has high error in specific directions: blocks 1-3 did something the self-model couldn't anticipate. The novel/surprising computation — and likely the source of the error — is HERE, in THESE directions.

The local errors provide a depth-and-direction decomposition of credit assignment that backprop approximates but degrades with depth (vanishing gradients, Jacobian entanglement).

**3. Automatic routine/novel partitioning.**

Over training, the FM errors naturally partition each layer's computation into:
- **Routine**: what the FM can predict (low error). Regular, compressible. Gets chunked.
- **Novel**: what the FM can't predict (high error). Input-specific, complex. Gets dedicated capacity.

The local loss selectively compresses the routine part (reducing FM error), freeing capacity for the novel part. This is the abstraction ratchet operating at the level of training dynamics: routine computation gets chunked, capacity reallocates to what's genuinely hard, what was hard becomes routine, repeat.

### Precision weighting: self-knowledge makes local errors informative

Raw local prediction errors are noisy — a large error could be informative ("novel computation worth learning from") or expected ("this direction is always noisy, ignore it"). Self-knowledge provides the calibration.

The innovation map (the directional structure of the FM's error covariance, already demonstrated in the prediction trust analysis) gives the model an expectation for what the FM error *should* look like. This enables precision-weighted prediction errors:

- **Informative surprise**: high error in a direction where the FM is usually accurate → strong learning signal
- **Expected noise**: high error in a direction where the FM always errs → weak learning signal (this is the FM's capacity limit, not a learning opportunity)
- **Suspicious absence**: low error in a typically-novel direction → something interesting may be happening

The precision-weighted local loss:

```
L_local = Σ_d  (r_d²  / σ²_d)
```

where r_d is the FM prediction error projected onto direction d, and σ²_d is the expected error variance in that direction (estimated from the FM's running error covariance). This is the predictive coding formulation (Rao-Ballard / Friston) applied to the A2A forward model's error signal.

Without self-knowledge, local prediction errors are just better-localized gradients. With self-knowledge, they become calibrated, direction-specific learning signals — the model uses what it knows about its own computational structure to decide what to learn from each experience.

The virtuous cycle:
1. FM errors provide local, directional credit assignment
2. Self-knowledge precision-weights these errors
3. Better-weighted errors → more efficient learning → more organized computation
4. More organized computation → more accurate FM → sharper self-knowledge → better weighting
5. Goto 2

### Relationship to injection and wake-sleep

The local prediction-error loss and the cerebellar injection serve different purposes:

| Mechanism | What it provides | When it acts |
|---|---|---|
| **Injection** (FM prediction → residual stream) | Inference-time self-knowledge. The model receives and evaluates the FM's prediction during its forward pass, developing the innovation map and error-monitoring geometry. Produces perturbation robustness. | Forward pass (inference + training) |
| **Local prediction-error loss** (FM error → gradient into main model) | Training-time credit assignment and continuous internalization. The FM's errors provide structured, depth-localized, precision-weighted supervision. Prevents dependency buildup. | Backward pass (training only) |
| **FM re-initialization** (fresh FM from new random init) | Discovery of new innovation structure. A fresh FM explores the model's computation from a new perspective, finding aspects the old FM wasn't tracking. | Periodic (between training phases) |

The local loss replaces the discrete SLEEP phase of wake-sleep. In the current system, dependency builds during wake (the model offloads computation to the injection), requiring periodic distillation to force internalization. With the local loss active during training, the model is continuously pressured to internalize what the FM models — the "be predictable" pressure from distillation runs continuously instead of in alternating phases.

The injection is still needed for inference-time self-knowledge: the real-time, input-specific signal about where the model's computation is surprising. The local loss creates weight-level regularity (the model's computational function becomes more organized). The injection creates activation-level self-monitoring (the model evaluates its own computation on each specific input). These are complementary.

Periodic FM re-initialization is still needed for the ratchet to turn. A continuously co-trained FM might settle into a local optimum, only tracking innovation in directions it's already attending to. A fresh FM from a new random init explores the model's computation from a completely different perspective — this is what produced the near-orthogonal residual PCs and qualitatively different digit-pair groupings in the MNIST distillation. The random re-init forces exploration; it's the same reason ensembles find different features.

The full architecture:
1. **Injection** — inference-time self-knowledge and robustness
2. **Local prediction-error loss** — continuous internalization and structured credit assignment
3. **Periodic FM re-initialization** — ratchet turning (discovery of new innovation structure)

No discrete wake-sleep cycle. The model does wake and sleep simultaneously.

## The degenerate solution and why it might be real

If the local loss ("be predictable") is too strong relative to NTP ("be useful"), the model converges on trivially predictable computation — the FM accurately predicts everything because the model stopped doing anything complex. This is representational collapse.

The wake-sleep results suggest this doesn't happen when the predictability pressure comes from a self-model (MNIST accuracy improved with regularity pressure), but the balance matters. The λ weighting of local vs global loss, and the precision weighting within the local loss, both need to be right.

Interestingly, this degenerate solution may correspond to a real failure mode of human cognition: the person who has "learned" a subject by making everything feel predictable and routine, without developing the capacity for genuinely novel computation. Overconfident familiarity without deep understanding. The cerebellar prediction is accurate because the cortex simplified its computation, not because the cerebellum got better at modeling complexity.

The precision weighting from §7 of the cerebellar ratchet doc is the natural safeguard. If the model's error expectations are well-calibrated (large expected variance in complex directions, small expected variance in routine directions), the local loss won't over-penalize genuinely complex computation — only computation that violates tight predictions gets strong gradient. This requires the self-knowledge (innovation map) to be established before the local loss becomes a dominant learning signal, suggesting a curriculum: start with injection-only training to develop the innovation map, then gradually introduce the local loss.

## Connection to the generalization problem

Ilya's core observation: models generalize dramatically worse than humans despite seeing orders of magnitude more data. Our hypothesis for why local prediction-error learning helps:

**Sample efficiency**: Each training example provides not just one output error but structured intermediate supervision at multiple depths. A chess grandmaster looking at a novel position gets depth-localized, direction-specific, precision-weighted signals — "piece recognition was routine, tactical patterns slightly novel but expected, strategic evaluation deeply surprising in an unusual direction — learn from THAT." One position, dense supervision. This is the sample efficiency advantage of cerebellar-cortical credit assignment.

**Robustness**: The local loss continuously compresses routine computation, producing flatter loss landscapes (demonstrated: Hessian trace 0.45× OL). Flatter landscapes → more conservative updates under distribution shift → less catastrophic forgetting → better OOD generalization. The OOD robustness experiment already showed this is distribution-invariant for the perturbation case.

**The "it" factor**: The routine/novel decomposition, driven by the FM's prediction errors and precision-weighted by the innovation map, is a concrete operationalization of "knowing what you need to learn." The model doesn't update all parameters equally — it preferentially updates the circuits responsible for genuinely novel computation, as identified by its own self-model. This is what makes the 100-hour student different from the 10,000-hour one: not more practice, but more targeted practice, directed by self-knowledge.

## Experimental plan

### Experiment 1: Single-depth local loss (minimal viable test)

Use the existing A2A setup (FM predicting post_block0 → post_block3). Add one change: let a fraction of the FM's prediction error flow as an auxiliary loss into the main model's blocks 0-2.

```
L_total = L_NTP + λ · ||sg(FM(post_block0)) - post_block3||²
```

where sg = stop-gradient through the FM (the FM provides the target, the main model adjusts toward it). Compare four conditions with identical lr/seed/init:

1. **OL**: open-loop baseline
2. **CL**: closed-loop with injection only (current system)
3. **LL**: local loss only (no injection)
4. **CL+LL**: injection + local loss

Measure: learning speed (loss at matched steps), final loss, robustness (perturbation Δloss), self-knowledge probes (R² for FM residual), dependency gap (CL and CL+LL only).

Key predictions:
- LL should learn faster than OL (more supervision per token)
- CL+LL should show less dependency buildup than CL (continuous internalization pressure counteracts offloading)
- CL+LL should match or exceed wake-sleep on final quality without needing a discrete sleep phase

### Experiment 2: Precision-weighted local loss

Same as Experiment 1, but replace the raw MSE local loss with a precision-weighted version. Estimate per-direction error variance from the FM's running error covariance (exponential moving average). Weight each direction's contribution to the local loss by 1/σ²_d.

Compare against Experiment 1 to test whether precision weighting improves the routine/novel decomposition and prevents the degenerate solution.

### Experiment 3: Multi-depth local loss

Add a second FM at a different depth point. Test whether multi-level credit assignment improves learning speed beyond single-depth.

### Experiment 4: OOD adaptation

Freeze models from Experiments 1-3. Fine-tune on novel domains (code, math, foreign language). Measure adaptation speed (loss curve), sample efficiency (performance at N tokens), and catastrophic forgetting (degradation on original domain).

This is the direct test of the generalization claim. If local-loss-trained models adapt faster to novel domains, the structured credit assignment translates to practical generalization advantage.

### Experiment 5: Scale-up

Repeat Experiment 1 at 77M+ parameters where layers are more differentiated and the depth-localized credit assignment should provide proportionally larger benefit.

## Open questions

1. **λ curriculum**: Should the local loss weight start at zero and ramp up (after self-knowledge is established from injection-only training), or should it be active from the start? The precision weighting might make the curriculum unnecessary.

2. **FM training dynamics**: Should the FM train continuously (tracking the model's evolving computation) or periodically (train on frozen model, then used as a fixed local target)? Continuous training risks the degenerate equilibrium; periodic training introduces staleness. The optimal schedule may be hybrid.

3. **Gradient isolation**: Should the local loss gradient flow through all parameters in the spanned blocks, or only through specific components (attention heads, MLP, layer norms)? The behavioral residual analysis showed the FM struggles most with multi-source attention integration — maybe the local gradient should preferentially target attention parameters.

4. **Interaction with the ratchet**: Does continuous local-loss training with periodic FM re-initialization produce the same compounding improvements (robustness, accuracy, self-knowledge) as discrete wake-sleep? The MNIST 2-cycle results are the benchmark.

5. **Multiple depth spans**: Should FMs predict one layer ahead (fine-grained credit assignment, more FMs needed) or multiple layers ahead (coarser credit assignment, fewer FMs, but spanning more computation)? The brain has cerebellar projections at multiple cortical levels with varying topographic specificity — suggesting a mix.

6. **Relationship to synthetic gradients**: Jaderberg et al. (2016) proposed learned modules that predict gradients at intermediate layers, enabling decoupled training. The local prediction-error loss is structurally similar but uses a self-model (predicting activations, not gradients) rather than a gradient predictor. The self-model grounds the local signal in the model's actual computational dynamics rather than in gradient statistics. Whether this grounding matters empirically is an open question.
