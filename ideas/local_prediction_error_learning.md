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

[**Revision (2026-06-16):** Precision weighting was tested on MNIST (Mahalanobis distance via eigendecomposed FM error covariance). It correctly identifies routine vs complex directions but doesn't meaningfully change outcomes — brittleness drops only 29% (9.5× vs 13.4×), and combined with injection it's identical to raw MSE. The problem is that the Friston framing doesn't apply here: precision weighting distinguishes signal from noise, but FM errors aren't noise — they're structured capacity limits. Downweighting "expected" errors just makes the local loss weaker in those directions without changing what the model learns. And the brittleness is geometric (compressing ~117/128 dimensions makes most random perturbations harmful) rather than directional (compressing the wrong dimensions). The virtuous cycle above doesn't engage because the precision weighting addresses a non-bottleneck.

A bilevel-optimized learning gate (MAML-style: gate trained to minimize classification loss after a virtual model update using gated local loss) partially solves this — brittleness drops 35% (8.7× vs 13.4×) and when combined with injection produces the highest self-knowledge ever measured (R²=0.83 at post_block0, beating CL+raw-MSE's 0.76). But the gate's learned selectivity doesn't correlate with digit discrimination or FM error variance — it discovers a more abstract task-relevant criterion that the bilevel signal can access but that no pre-computable per-dimension statistic captures. The gate validates that task-informed weighting beats FM-informed weighting, but the mechanism is more subtle than "exempt class-relevant directions."]

[**Revision (2026-06-15):** The virtuous cycle as stated above has a bootstrapping problem. The MNIST local loss experiment showed that the local loss *alone* (LL condition) drives the FM to near-perfect accuracy (cosine 0.997), which eliminates the residual and therefore eliminates self-knowledge entirely — there is nothing to precision-weight because there is nothing surprising. The cycle only engages when the injection maintains a meaningful residual by introducing computation the FM can't predict from pre-injection activations. The revised cycle should be: injection creates irreducible residual → local loss encodes meta-knowledge about that residual at early layers → precision weighting becomes possible → better learning. The injection is not just "also useful" — it is likely a prerequisite for the self-knowledge that makes the local loss more than a generic auxiliary objective.]

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

[**Revision (2026-06-15):** The claim "no discrete wake-sleep cycle" is too strong. The MNIST experiment confirmed that the local loss substantially reduces dependency (63% less than injection-only), validating the continuous internalization claim. But the knowledge it internalizes is predominantly *meta-knowledge* (where the FM is wrong) rather than *object-level knowledge* (what the FM computes). Representation probes showed a 3:1 ratio of meta-knowledge to object-level gain at early layers. Discrete distillation, by contrast, has been shown to effectively transfer the FM's object-level contribution into the main model's weights. This suggests the local loss and distillation may be complementary rather than substitutive: the local loss handles continuous meta-knowledge absorption (preventing dependency buildup in real time), while periodic distillation handles object-level knowledge transfer (absorbing what the FM actually computes). The full architecture may still need all three mechanisms, with the local loss reducing but not eliminating the need for discrete sleep phases.]

## The degenerate solution and why it might be real

If the local loss ("be predictable") is too strong relative to NTP ("be useful"), the model converges on trivially predictable computation — the FM accurately predicts everything because the model stopped doing anything complex. This is representational collapse.

The wake-sleep results suggest this doesn't happen when the predictability pressure comes from a self-model (MNIST accuracy improved with regularity pressure), but the balance matters. The λ weighting of local vs global loss, and the precision weighting within the local loss, both need to be right.

Interestingly, this degenerate solution may correspond to a real failure mode of human cognition: the person who has "learned" a subject by making everything feel predictable and routine, without developing the capacity for genuinely novel computation. Overconfident familiarity without deep understanding. The cerebellar prediction is accurate because the cortex simplified its computation, not because the cerebellum got better at modeling complexity.

The precision weighting from §7 of the cerebellar ratchet doc is the natural safeguard. If the model's error expectations are well-calibrated (large expected variance in complex directions, small expected variance in routine directions), the local loss won't over-penalize genuinely complex computation — only computation that violates tight predictions gets strong gradient. This requires the self-knowledge (innovation map) to be established before the local loss becomes a dominant learning signal, suggesting a curriculum: start with injection-only training to develop the innovation map, then gradually introduce the local loss.

[**Revision (2026-06-15):** The representational collapse predicted above did not occur at λ=1.0 on MNIST — task performance actually improved. But a different failure mode appeared: *precision without robustness*. The local loss trained a tight block0→block3 mapping that amplifies perturbations at intermediate points. This is not collapse (the model is doing useful computation) but brittleness (the computation is fragile to noise). The injection appears to provide exactly this — it acts as a form of structured noise injection during training, teaching the model to absorb perturbations. The precision weighting safeguard proposed above may address the brittleness too, by leaving genuinely complex (high-variance) directions alone rather than compressing them, but this is untested.]

## Connection to the generalization problem

Ilya's core observation: models generalize dramatically worse than humans despite seeing orders of magnitude more data. Our hypothesis for why local prediction-error learning helps:

**Sample efficiency**: Each training example provides not just one output error but structured intermediate supervision at multiple depths. A chess grandmaster looking at a novel position gets depth-localized, direction-specific, precision-weighted signals — "piece recognition was routine, tactical patterns slightly novel but expected, strategic evaluation deeply surprising in an unusual direction — learn from THAT." One position, dense supervision. This is the sample efficiency advantage of cerebellar-cortical credit assignment.

**Robustness**: The local loss continuously compresses routine computation, producing flatter loss landscapes (demonstrated: Hessian trace 0.45× OL). Flatter landscapes → more conservative updates under distribution shift → less catastrophic forgetting → better OOD generalization. The OOD robustness experiment already showed this is distribution-invariant for the perturbation case.

[**Revision (2026-06-15):** The robustness claim above is wrong, or at least the causal chain is. The MNIST experiment showed that the local loss alone makes the model *more* sensitive to perturbations, not less — computational regularity does not produce flatter loss landscapes. The Hessian trace reduction observed in prior closed-loop models comes specifically from the injection experience (training with structured additive signals at the perturbation point), not from the division of labor or the regularity of computation. A model can have perfectly regular, FM-compressible computation and still be brittle if it was never exposed to perturbations during training. The robustness → OOD generalization argument may still hold, but the mechanism must run through injection experience rather than through computational regularity. Whether precision-weighted local losses (which would selectively compress only routine directions) avoid this brittleness remains an open question.]

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

## The learning gate: from automatic to goal-directed selective learning

### Motivation: backprop doesn't know what to learn

The precision-weighted local loss (§ Precision weighting) replaces raw FM error with calibrated, direction-specific signals: large errors violating tight predictions get strong gradient, large errors in always-noisy directions get weak gradient. This is automatic selectivity — the weights come from error covariance statistics, not from any deliberate choice by the model.

But the model already *knows* more than the error covariance. The innovation map (directional self-knowledge, demonstrated in the prediction trust analysis) gives the model an input-specific, high-dimensional representation of what's novel about each data point. The CerebellarGate learned to selectively use different directions of the FM prediction during inference — per-direction selectivity driven by what's useful for the task. The model has awareness of what's novel and the capacity for selective use. What's missing is the connection between this awareness and the learning rule.

Currently, a data point arrives, and backprop computes ∂L/∂θ uniformly across all parameters. The precision-weighted local loss improves on this by weighting different error directions differently — but the weights are fixed functions of running statistics, blind to the specific input. A model that encounters a data point with unusual delimiter patterns and unusual semantic content gets the same precision weights regardless of whether the delimiter novelty or the semantic novelty is more useful to learn from right now. The model can *see* the difference (via the innovation map) but can't *act on it* in its own learning.

### The proposal: a differentiable gate on the local loss

Replace the fixed precision weights `1/σ²_d` with a small learned network — the **learning gate** — that outputs per-direction weights for the local loss, conditioned on the model's current representations.

Architecture: structurally identical to the CerebellarGate (which gates the inference signal), but applied to the backward pass. The learning gate takes as input:

- Current-layer activations (what the model is computing)
- FM error vector (what the FM found surprising)
- Running innovation map statistics (what's typically surprising vs. atypical)

And outputs per-direction scalar weights on the local loss. Directions that receive high weight produce strong gradients into the main model; directions that receive low weight produce weak or zero gradients.

The gate's parameters are updated by NTP loss through the standard chain rule: FM error → gate → weighted local loss → gradient into main model → weight update → NTP performance. No meta-learning objective, no held-out set. The main task itself teaches the gate what's worth learning — directions where acting on the FM error improves NTP performance cause the gate to open; directions where the FM error is noise cause it to close.

This is the same mechanism by which the CerebellarGate learned to selectively use FM predictions during inference. The CerebellarGate answers "which aspects of the FM's prediction are useful for processing this input?" The learning gate answers "which aspects of the FM's error are useful for updating my weights on this input?" Same architecture, different pass.

### FM reinitialization as diversity pressure

The learning gate trained end-to-end with a single FM risks co-adaptation: the gate and FM settle into a comfortable equilibrium, always weighting the same directions. The gate memorizes specific error directions rather than learning general criteria for "what makes an error direction worth acting on."

FM reinitialization breaks this co-adaptation. Each fresh FM, trained from new random init on the current model's activations, discovers residual structure from a genuinely novel perspective — the MNIST distillation showed near-orthogonal residual PCs and qualitatively different digit-pair groupings across cycles. Each reinitialization challenges the gate with a new error decomposition:

- **Cycle 1**: FM-1 discovers residual structure R1. Gate learns to weight certain directions of R1. Model internalizes those directions.
- **FM reinitialization**: Fresh FM-2 trains on the reorganized model. Discovers R2, near-orthogonal to R1 — aspects of the computation FM-1 wasn't tracking.
- **Cycle 2**: Gate must handle R2's qualitatively different directions. It can't replay its cycle-1 policy. It must generalize — learn something about *what makes an error direction worth acting on*, not just *which specific directions to act on*.

Over many cycles, the gate accumulates a general-purpose policy for selective learning. Each FM reinitialization is a new teacher arriving with a different pedagogical perspective; the gate learns to be a good student of any teacher.

This produces a second ratchet operating on the learning rule itself. The original cerebellar ratchet compresses computation (routine → chunked → capacity freed → repeat). The learning-gate ratchet refines the learning policy (selective weights → internalization → new FM perspective → better selective weights → repeat). The model learns not just what to compute but what to learn.

### Biological plausibility: the reticular thalamus

The learning gate maps onto the reticular nucleus of the thalamus. The reticular nucleus provides inhibitory gating of thalamocortical transmission — it controls which aspects of thalamic relay (including cerebellar output via VL/VA nuclei) reach cortex. Critically, its gating is modulated by cortical feedback: the cortex influences which aspects of the cerebellar signal it receives. This is exactly the architecture described above — the model (cortex) controls, via the gate (reticular thalamus), which aspects of the FM error (cerebellar prediction error) drive learning. The gate is not a passive relay but a cortically-modulated filter on the teaching signal.

### Connection to instruction-directed learning

If the learning gate takes activations as input, and those activations progressively encode more semantic content through ratchet cycles (the abstraction ratchet's core claim — procedures become percepts, computations become primitives), then the gate's selectivity criteria will naturally move from computational to semantic. Early in training, the gate learns things like "upweight errors in the distributed-attention direction when attention entropy is high." After many ratchet cycles, the same gate architecture conditions on more abstract features because that's what the representations encode.

In the limit, this converges toward a system where the model's own representational understanding of "what kind of regularity this is" drives the gating of its own learning signals. The gap between "upweight delimiter-matching errors" and "learn the formatting from this" is the same gap the ratchet closes everywhere else — turning multi-step computational criteria into single-step semantic pattern matches. Whether this can eventually interface with language-level instructions (making the learning gate responsive to textual descriptions of what to learn) is an open question, but the architectural path is clear: the gate already conditions on the model's representations, and those representations already encode semantic content.

### Experimental plan

**Experiment 2b: Learned learning gate** (after Experiment 2, precision-weighted local loss)

Same 4-condition setup as Experiment 1, but replace the fixed precision weights with a small learned gate network. The gate takes (post_block1 activations, FM error vector) and outputs per-direction weights on the local loss. Gate parameters trained end-to-end via NTP loss.

Compare against Experiment 2 (fixed precision weights) to test whether learned selectivity outperforms statistical selectivity. Key measures:
- Does the gate learn interpretable direction preferences? (Do gate weights correlate with behavioral categories from the behavioral residual analysis?)
- Does the gate's policy generalize across FM reinitializations? (Train gate with FM-1, reinitialize FM, measure gate adaptation speed with FM-2 vs. from-scratch)
- Does the gate reduce dependency more than fixed precision weights? (The gate should learn to not upweight directions that create injection dependency)

[**Revision (2026-06-16):** Experiment 2b was run on MNIST as a FOMAML bilevel optimization (see MNIST_LOCAL_LOSS_README.md, "Learning gate" section). Key findings that update the above:

1. The gate's selectivity does NOT correspond to digit discriminability (eta² correlation +0.02) or FM error variance (correlation -0.18). The bilevel optimization discovers a selectivity criterion that can't be reduced to any per-dimension statistic we can pre-compute — it depends on the interaction between the local loss gradient, the current parameter configuration, and the downstream classification effect. This is the strongest evidence that heuristic approaches (precision weighting, gradient alignment) are fundamentally insufficient: the relevant structure lives in the bilevel interaction, not in any first-order statistic of the FM error.

2. CL_LG produces record self-knowledge (R² = 0.83 at post_block0, vs 0.76 for CL_LL and 0.63 for CL) while maintaining near-full injection dependency (0.019 vs CL's 0.022). High dependency + record self-knowledge is potentially the optimal pre-distillation state: the model extracts maximum value from the injection while encoding maximally precise information about where it needs it. CL_LL's indiscriminate "be predictable" pressure forces partial internalization that reduces dependency but also reduces the precision of the self-knowledge. The gate allows the model to be *selectively* dependent.

3. The dependency prediction in the experimental plan above was wrong: the gate does NOT learn to reduce dependency. It learns to *maximize the value extracted from the injection*, which increases dependency. This reframes dependency as a feature of well-structured self-knowledge, not a failure mode to be minimized — at least when distillation is available to subsequently internalize the structured dependency.]

[**Revision (2026-06-17):** The multi-cycle gated ratchet experiment (GATED_RATCHET_README.md) ran 4 cycles of WS_LG (injection + bilevel gate + distillation + FM reinit) on MNIST. Key findings that update the learning gate section:

1. The gate **opens** across cycles (mean weight 0.31 → 0.77) rather than closing. On MNIST at ~18 epochs, the bilevel signal always endorses more compression because the val set is distributionally identical to the train set. The gate correctly concludes: compress everything, it always helps. The "plasticity closes with maturity" prediction requires novel inputs that over-compression would damage — a condition that never holds on a fixed, fully-seen dataset. The biological analogy: a child raised in a single room with 10 toys for 18 years has no reason to restrict plasticity because there's nothing rare worth protecting.

2. WS_LG produces **monotonically compounding val loss improvement** (34% → 48% gap vs OL over 4 cycles). Neither distillation alone (WS stalls at cycle 2) nor gated local loss alone (CL_LG stalls) sustains improvement. The mechanism is implicit regularization: the gate selects among parameter configurations for the one whose intermediate computation is most compressible by a self-model, in directions the task loss endorses. The model finds solutions that are maximally legible to a compressed version of its own computation — analogous to deep understanding vs surface-level pattern matching.

3. The gate doesn't extract structure from the data (which is exhausted by epoch 18). It extracts **computational regularity from the model's own processing**. The FM provides a mirror, and the gated local loss pushes the model to be more like its reflection. This signal never exhausts on a fixed dataset because there's always more regularization possible. The absorbing state is: residual → 0, FM near-perfect, gate weights plateau at high values (not closed — irrelevant, because gate_w × r² ≈ 0 regardless of gate_w).

4. The gate-closing prediction requires the bilevel outer loop to evaluate on a distribution that differs from the inner loop. On a fixed dataset, inner = outer, so more compression is always beneficial. The human genome builds in the prior that "the outer loop WILL contain novelty" even when the current experience stream doesn't show it yet — this is the missing prior in our current setup.]

## From dataset regularities to process regularities

The learning gate + ratchet framework points toward a distinction between two kinds of learning that may be important for understanding generalization:

**Current models learn the regularities of a dataset** — a fixed, replayable collection. The learning rule (backprop + SGD) doesn't need to be selective because you get unlimited passes. If a regularity is in the data, you'll eventually absorb it.

**Biological learners learn the regularities of an experience-generating process** — one you can only sample from, never replay, and where some samples are existentially consequential. You get one pass through each experience. This puts enormous pressure on selecting which regularities to extract from each sample, because you can't afford to waste a rare experience learning something redundant, and you can't afford to miss a critical pattern because you were learning the wrong thing.

The learning gate's bilevel optimization is a crude version of this: "which regularities, if internalized from this batch, will help on the next batch I haven't seen yet?" That's one-step-lookahead into an unknown future corpus. FM reinitialization extends the horizon — each fresh FM provides a genuinely novel perspective on the model's computation, and the gate must extract regularities that are robust across perspectives. Regularities that hold across multiple FM decompositions are more likely to be properties of the underlying process, not artifacts of any particular sample.

The cerebellar connection: the cerebellum isn't modeling the data either — it's modeling the *cortex's dynamics*, which is the process that generates internal experiences. Each new input is a one-shot sample from this process. The ratchet extracts regularities that are stable properties of how you think, not of what you've seen. Those are the regularities that transfer across domains.

### Asymmetric loss and the missing prior

Biological experience-generating processes have a feature that training corpora don't: some samples are lethal. This creates asymmetric loss — you need to get the high-stakes regularities right on the *first* encounter. The amygdala-mediated valence tagging described in §7 of the cerebellar ratchet doc serves exactly this function: a fast, evolutionarily-prior-loaded system that triages experiences by stakes before the learning gate decides what to learn from them. The genome provides a prior over "which types of novelty are high-stakes" that doesn't need to be learned from scratch.

Our current system has no analog of this — the learning gate treats all batches as equally important. For MNIST this is fine. For a system that needs to generalize robustly under distribution shift, some form of stakes-weighted meta-learning may be necessary: not just "which regularities help classification" but "which regularities help classification *when it matters most*."


### 8. Self-directed learning: learning from your own thoughts

The pieces described in §§1–7 of ideas/cerebellar_abstraction_ratchet.md, together with the learning gate mechanism, may be jointly sufficient for a capability that current AI systems lack: *self-directed learning*, where the system generates its own learning objectives from novel thoughts and selectively updates its own knowledge in response.

**The mechanism.** During a forward pass — processing an input, reasoning about a problem, generating a response — the cortex produces novel activation patterns. The cerebellar forward model (§1–2) flags these as novel: they deviate from the predicted trajectory in specific directions (the innovation map). The cortex, which has self-knowledge baked into its representations through co-training (§5), recognizes the novelty and its character.

The novel thought then becomes a *learning objective*. The cortex retrieves or activates relevant stored knowledge — from its own weights (implicit statistical knowledge), from hippocampal episodic memory (specific past experiences), from associative activation (related concepts). This retrieved knowledge serves as the learning signal. The learning gate (reticular thalamic gating, §7 + local prediction error doc) selects which regularities from the retrieved knowledge to internalize, using the novel thought as the meta-learning objective: "which regularities from my stored knowledge, if strengthened, would help me process this kind of novel thought?"

The result: the system updates its own weights to better handle the class of novelty it just encountered, using its own stored knowledge as the training signal, without any external supervision.

**The meta-learning framing.** This maps onto a bilevel optimization where:

- **Corpus A** (the meta-objective): the novel thought — what the system wants to get better at processing. Not a dataset, but a single activation pattern held in working memory / active cortical state.
- **Corpus B** (the learning signal): stored knowledge relevant to the novel thought — retrieved from cortical weights, hippocampal replay, associative activation. Not a fixed dataset, but a dynamically generated set shaped by Corpus A itself. The novel thought determines what gets retrieved.
- **The meta-learning question**: "which regularities from B, if internalized, help with A?"

The learning gate answers this question by selectively gating which aspects of the Corpus B learning signal drive plasticity, with the Corpus A objective determining the gating policy (via the bilevel optimization described in the local prediction error doc).

Importantly, Corpus A and B are not two data points held simultaneously in mind. Corpus A is an *active objective* (maintained in prefrontal/working memory state), and Corpus B is an *ongoing stream* of experience and memory whose processing is shaped by A. The meta-learning happens as a process — the objective shapes retrieval, retrieval provides learning signal, the gate filters the signal, plasticity occurs — not as a comparison between two stored items.

**Why this requires all the pieces.**

| Component | Role in self-directed learning | Without it |
|---|---|---|
| FM + injection (§1–2) | Recognizes novel thoughts as novel; provides directional characterization of what's surprising | No novelty detection — system can't distinguish novel from routine thoughts |
| Self-knowledge / innovation map (§5) | Knows what it knows; provides the prior over own knowledge state | Can detect novelty but can't characterize it — "something is new" but not "what kind of new" |
| Learning gate (bilevel) | Selects which regularities from Corpus B to internalize given the Corpus A objective | Either learns everything indiscriminately (brittle) or nothing (stagnant) |
| Rich cortical priors | Generates novel thoughts worth learning from; provides the Corpus B knowledge base; evaluates learning quality | Nothing to learn from; nowhere to retrieve relevant knowledge; no sense of "this is going well" |
| Distillation / ratchet (§3–4) | Internalizes what was learned, freeing capacity for the next level of novelty | Learning from novel thoughts doesn't compound; no hierarchical abstraction |

**Waking learning vs. sleep consolidation.** The meta-learning backward pass — selecting regularities from Corpus B to internalize via a bilevel objective — is energy-intensive, requiring both a forward pass (to generate the novel thought) and a meta-learning update (to modify the learning gate and cortical weights). This happens during waking cognition, when metabolic resources are available.

Sleep serves a different function: distillation (§4). The ratchet's compression step — transferring cerebellar predictions into cortical weights, clearing the FM for the next cycle — produces the discrete jumps in capability observed after sleep. Sleep doesn't generate novel thoughts or execute meta-learning backward passes; it consolidates the results of waking meta-learning into the weight structure, preparing the system for the next round.

The "clicking" phenomenology — when something suddenly makes sense — may correspond to a successful waking meta-learning step: the system found regularities in Corpus B that resolve the Corpus A novelty. The subsequent sleep consolidation internalizes this resolution, producing the common experience of understanding deepening overnight without conscious effort.

**Open questions specific to self-directed learning.**

1. *How literally does the Corpus A/B framing map onto cortical learning?* The bilevel optimization is a computational-level description (Marr's Level 1). The biological implementation likely involves neuromodulatory gating of plasticity (dopamine, acetylcholine, norepinephrine) rather than literal backpropagation. These neuromodulatory systems are themselves learned — dopamine neurons learn to predict rewards, the LC-NE system learns what constitutes surprising context — so they constitute a learned system modulating a learning process, which is meta-learning by definition. The most plausible biological approximation of the bilevel optimization is timescale separation: fast synaptic plasticity (Hebbian/STDP, milliseconds) serves as the inner loop, slower neuromodulatory dynamics (seconds-minutes) as the outer loop, approximating meta-gradient descent via something closer to a bandit problem over gating configurations. This is noisier and slower than MAML but converges to a functionally similar outcome: the system learns what to learn from, and that meta-policy improves with experience. Existing theories of cortical learning — Hebbian plasticity, STDP, predictive coding — describe the *mechanism* of weight updates; the meta-learning framing describes *what modulates* those updates. The learning gate would need to interface with these mechanisms, perhaps through neuromodulatory systems that selectively enable/disable Hebbian plasticity in specific circuits based on the cerebellar/thalamic novelty signal. Whether the biological approximation converges to the same solution as exact bilevel optimization, or a qualitatively different one, is genuinely open.

2. *What determines Corpus B retrieval?* The novel thought (Corpus A) shapes what gets retrieved, but through what mechanism? Hippocampal pattern completion is one candidate: the novel activation pattern partially matches stored episodes, triggering replay of the most relevant memories. Cortical associative activation is another: the novel pattern excites related representations through learned connection weights. The quality of Corpus B retrieval may be a major bottleneck on self-directed learning — if the wrong memories/knowledge are retrieved, the meta-learning optimizes the wrong objective.

3. *Can this be demonstrated in the A2A system?* The learning gate bilevel optimization already performs a one-step version: it selects which FM error directions to learn from based on classification improvement. The extension to self-directed learning would require the model to (a) generate novel activations, (b) recognize them as novel, and (c) use them as meta-learning objectives for updating its own processing. Steps (a) and (b) are already present in the closed-loop system. Step (c) would require the model's own novel activations to replace the classification loss as the bilevel outer objective — a significant but architecturally straightforward modification.