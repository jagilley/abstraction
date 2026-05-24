# Abstraction Supervision as Metacognitive Control

**Date:** 2026-03-26  
**Status:** Working synthesis / new framing  
**Related work:**
- [Self-modeling and the failure of RL in high-dimensional domains](./self_modeling_and_rl_in_high_dimensions.md)
- [GLP directional residuals](../experiments/zipfian_grokking/glp_directional_residuals/README.md)
- [Suppress memorization subspace](../experiments/zipfian_grokking/suppress_memorization_subspace/README.md)
- Third learnings: RL vs GLP-RL[^private]
- [Abstraction-space supervision](../experiments/ideas/abstraction_supervision.md)
- Zipfian grokking essay[^private]

---

## 1. Why this note exists

This note is a synthesis of several tensions that emerged across our recent
Zipfian grokking work:

1. The **GLP-RL story is directionally right but mechanistically incomplete**.
   The GLP can clearly encode useful structure, and in some regimes it helps
   RL, but using it as a passive per-sample gradient oracle is unstable.

2. The **memorization-subspace result is much stronger than a one-off trick**.
   In [suppress_memorization_subspace](../experiments/zipfian_grokking/suppress_memorization_subspace/README.md),
   a single learned representational direction could be suppressed and the
   Sisyphean collapse disappeared. This strongly suggests that the pathology
   lived in representation space and was causally actionable there.

3. The old [abstraction-space supervision](../experiments/ideas/abstraction_supervision.md)
   idea now looks much more alive than it did originally, but likely in a more
   general form than explicit PCA coordinates + Jacobian estimation.

4. The missing ingredient may not be "a better novelty detector" but a
   **metacognitive control loop**: a system that gets privileged access to the
   organization of the task model's own representations, evaluates the quality
   of prospective self-updates, and modulates learning accordingly.

The most important update from this conversation is that **we should not
prematurely hardcode the mechanism**. Discrete hypotheses, hand-authored proxy
metrics, and explicit candidate directions may all be useful engineering
approximations, but they are probably not the causal heart of the phenomenon.

---

## 2. Core claim

**Abstraction supervision is best understood not as an explicit teacher over
examples, nor necessarily as a bank of discrete hypotheses, but as a form of
metacognitive control: the model is guided by signals about the organization of
its own representations, not just by per-example success.**

A more operational version:

> A self-improving model likely needs internal state that represents the
> quality of prospective self-updates using privileged access to its own
> cognition, and that internal state must be able to modulate replay,
> consolidation, attention, or gradient flow.

This is a smaller and more defensible claim than:
- models need explicit quantized ideas,
- models need hand-designed proxy metrics,
- or consciousness just *is* abstraction supervision.

The key distinction is between:
- **state modeling**: what do my representations look like?
- **update-quality modeling**: what kinds of internal changes are worth
  becoming?

The GLP mostly gives us the first. What we seem to need for stable
self-supervised improvement is the second.

---

## 3. What the recent experiments imply

### 3.1 The memorization direction is a smoking gun

The [GLP directional residuals](../experiments/zipfian_grokking/glp_directional_residuals/README.md)
experiment found that near collapse:
- the harmful representational distortion is approximately **1-dimensional**,
- it lives overwhelmingly in **h1**,
- the Zipf-weighted effective gradient is largely aligned with it,
- and the broad task structure remains mostly intact in the orthogonal
  complement.

Then [suppress_memorization_subspace](../experiments/zipfian_grokking/suppress_memorization_subspace/README.md)
showed that projecting this direction out of the gradient:
- eliminates the Sisyphean collapse,
- preserves test performance,
- and does so even when the direction is extracted once from a frozen snapshot.

This is unusually strong evidence that:
1. there are **low-dimensional metacognitive control variables** latent in
   representation space,
2. those variables can be **causally decisive** for learning dynamics,
3. and intervention at the level of **representation organization** can work
   better than direct output-level supervision.

The hand-discovered memorization subspace should therefore be seen as a
special-case proof that a more general metacognitive controller could exist.
A reasonably intelligent meta-model that saw replayed representational states
and was trained to predict collapse should be able to discover essentially the
same structure on its own.

### 3.2 Why the GLP-RL result still matters

Third learnings[^private]
showed something subtler than "GLP-RL failed":
- the GLP contains useful structural information,
- per-sample GLP-based projection can help for long stretches,
- but it goes off-policy because it mixes **structural** and **positional**
  knowledge.

The key lesson is not that self-modeling is wrong. It is that a **passive,
per-sample self-model is the wrong level of abstraction for control**.

The stable lesson from both the success and failure modes is:

> Rich self-modeling signal is useful as raw material, but the actual control
> variables must be simple, structural, and causally tied to learning
> dynamics.

The memorization direction was exactly such a variable.

### 3.3 Why abstraction supervision still looks right

The original abstraction-supervision document argued that the model should be
supervised in a low-dimensional space of representational organization rather
than through per-example correction. That still feels correct.

What now looks less essential is the exact original implementation recipe:
- explicit PCA basis,
- explicit Jacobian estimation,
- explicit teacher gradient in that basis.

Those may still be useful tools, but the deeper object is broader:

> the model should receive learning-relevant signals about the quality and
> organization of its own representations.

That is the common thread linking abstraction supervision to the
memorization-subspace intervention.

---

## 4. The major dissonances that got resolved

This conversation changed several earlier intuitions.

### 4.1 From explicit hypotheses to update-quality signals

**Initial dissonance:** Maybe the missing architecture is a bank of explicit,
compact hypotheses or ideas.

**Why this felt off:** Pretraining does not hardcode quantized ideas. Concepts
still emerge. That suggests explicit hypothesis slots are not the primitive
cause.

**Current consonance:** Explicit hypotheses may be an *effect* of a deeper
process. The more primitive thing is likely a **continuous metacognitive
representation of update quality**: some internal state that says, roughly,
"this way of changing myself seems worth reinforcing."

If stable reusable ideas later emerge, that may be because certain update
patterns keep receiving reinforcement under replay and consolidation.

### 4.2 From candidate directions to endogenous internal patterns

**Initial dissonance:** Proposing candidate directions sounded suspiciously
like sneaking in domain knowledge.

**Current consonance:** In toy problems, the system can bootstrap from
**endogenous recurring internal structure**:
- recurrent low-rank residual patterns,
- recurrent drift modes,
- recurrent precursors of collapse,
- recurrent disagreement patterns under replay.

In larger models, especially LLMs, the model may eventually be able to propose
more semantic candidates using its own domain knowledge, but that should be
seen as a later capability, not as the primitive starting point.

### 4.3 From performance metrics to abstraction-level evaluation

**Initial dissonance:** Measuring performance benefits on counterfactuals felt
like it risked collapsing back into data-space supervision.

**Current consonance:** The cleaner object is not "does this help those
examples?" but something like:

> does this internal change improve the organization, stability, or
> intelligibility of replayed experiences in abstraction space?

External performance can still serve as a sparse anchor, but the actual
control signal should ideally live in representation space.

### 4.4 From replay as memory to replay as metacognitive packaging

**Initial dissonance:** Replay sounded like a hypothesis-testing tool in a way
that still felt too explicit.

**Current consonance:** Replay may be more basic than that. It is the process
that **packages fleeting model states into a form consumable by the meta-model**.

Online activations are too entangled and transient. Replay lets the system:
- revisit them,
- compare them,
- detect recurring structure,
- and evaluate the likely quality of self-updates from a calmer vantage point.

This may be the main computational role of a hippocampal-like buffer in this
story: not merely remembering, but formatting cognition for metacognition.

### 4.5 From public/private features to good/bad-to-consolidate updates

**Initial dissonance:** The public/private language was useful, but it risked
becoming another proxy objective.

**Current consonance:** Public/private is better viewed as one symptom of a
more general distinction:
- some updates deserve consolidation,
- others do not.

In Zipfian grokking, that distinction cashes out neatly as:
- Fourier structure: worth consolidating,
- memorization corrections for top-weight samples: not worth consolidating.

But the deeper object is not publicness itself. It is **consolidation-worth**
under privileged self-evaluation.

---

## 5. A more general architecture story

The earlier self-modeling note leaned on a neocortex–hippocampus analogy. That
still feels right, but we now have a more concrete computational role for each
piece.

### 5.1 Task model / neocortex analogue
The main model that performs the task and contains rich task knowledge.

### 5.2 Replay / hippocampal analogue
A system that stores and re-presents internal states or short trajectories in a
form that makes second-order evaluation possible.

### 5.3 Meta-model / prefrontal analogue
A controller that consumes replayed internal states and predicts something like
future collapse, future integration, or future stability of candidate updates.
Its job is not merely to detect novelty, but to estimate **whether a change is
worth consolidating**.

### 5.4 Control pathway
The meta-model must be able to modulate the task model, for example via:
- replay prioritization,
- gradient gating or low-rank preconditioning,
- selective consolidation,
- routing / attention biases,
- or other slow learning-control pathways.

The central claim is not any one mechanism above. It is that **there must be a
path from metacognitive evaluation to learning dynamics**.

---

## 6. Why this may be consciousness-adjacent

We should not overclaim here. But there is a serious and useful analogy.

A restrained formulation:

> Abstraction supervision may instantiate one computational ingredient of what
> is often called access-consciousness: the system forms compact,
> value-laden summaries of its own cognition and uses them to regulate future
> cognition.

This does **not** mean:
- consciousness reduces to this loop,
- phenomenology is explained,
- or any metaphysical claim is settled.

It does mean that the following package looks increasingly important:
- self-monitoring of internal organization,
- replay-supported second-order evaluation,
- global-ish access to low-dimensional summaries,
- and top-down modulation of future thought/learning.

That package is very plausibly one of the mechanistic cores of what people are
gesturing at when they talk about reflective awareness, understanding, or
mental self-regulation.

A concise way to say it:

> Consciousness, in one operational sense, may be the process by which a
> system constructs compact abstractions of its own cognition and feeds them
> into value-laden control loops.

We should treat this as a suggestive analogy, not a settled theory.

---

## 7. The risk of metacognitive misalignment

A natural corollary is that the meta-model can go wrong.

If its internal value signals care about the wrong things — e.g.
- salience over depth,
- certainty over calibration,
- local reward over durable structure,
- fluency over truth,
- or novelty over integration —

then it can actively steer the task model away from genuine understanding.

This is not a side issue. It may be central. Once we believe that a
metacognitive controller can improve learning, we must also believe that a
misaligned metacognitive controller can systematically corrupt it.

That possibility likely matters both for AI systems and for human cognition.

---

## 8. The clean next experiment

This conversation clarified a very direct next step.

### 8.1 Objective
Train a meta-model to predict future collapse vs stable generalization from
replayed representational states, then let it modulate learning.

### 8.2 Why this is clean
- It does **not** require hand-labeling the memorization subspace.
- It does **not** require explicit discrete hypotheses.
- It does **not** require us to commit to a proxy metric like
  public/private/compositionality.
- It uses the strongest thing our current experiments already showed:
  representational states contain predictive information about whether the
  current learning dynamics are healthy.

### 8.3 Minimal setup
1. Collect replayable representation trajectories around build,
   pre-collapse, collapse, and recovery.
2. Train a meta-model on those trajectories (or GLP states / low-rank summaries
   of them) to predict future collapse or future stable generalization.
3. Give the meta-model a control channel: e.g. low-rank gradient modulation,
   replay prioritization, or suppression of predicted-bad representational
   drift.
4. Test whether the learned controller rediscovers, explicitly or implicitly,
   the memorization subspace phenomenon.

### 8.4 What success would mean
If this works, it would show that:
- privileged access to representational organization can predict long-horizon
  update quality,
- a meta-model can use that access to steer learning,
- and the memorization-subspace result was not a special trick but a toy
  instance of a broader metacognitive control principle.

---

## 9. The broadest motivating picture

The deeper research program now looks something like this:

1. **Pretraining** can build a great deal of structure through dense,
   distributed signal and passive regularization.
2. **Fine-tuning / RL / skewed-data regimes** expose a new bottleneck: the
   model must decide not just *what worked*, but *what kind of internal change
   is worth becoming*.
3. That decision likely cannot be made well from scalar reward alone in
   high-dimensional domains.
4. A self-model with replay and metacognitive control may provide the missing
   bandwidth.
5. In toy settings, the relevant control variables can sometimes be found by
   hand (e.g. the memorization subspace). In larger systems, they should be
   discovered by learned meta-models.

This reframes abstraction supervision from a niche curriculum-learning idea
into a general proposal:

> rich intelligence may require not just learning from the world, but learning
> from organized abstractions of one's own cognition.

---

## 10. Compressed takeaway

The most compact version of this note is:

> The success of memorization-subspace suppression suggests that there exist
> low-dimensional, causally meaningful variables in representation space that
> govern whether learning dynamics are healthy. A more general system should
> not rely on hand-discovered subspaces, but on a replay-consuming meta-model
> that predicts the quality of prospective self-updates from the organization
> of the task model's own representations and modulates learning accordingly.

And the most restrained consciousness-adjacent restatement is:

> Abstraction supervision may be one computational form of metacognitive
> control: using compact, value-laden summaries of one's own cognition to
> regulate future cognition.

[^private]: Not mirrored: this link points to a document in the private lab repo (the roadmap, the queue, an unrun spec, reading notes, or a conversation). See the top-level README for what is held back and why.
