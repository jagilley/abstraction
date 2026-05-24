# Metacognitive Novelty Learning

**Date:** 2026-03-31  
**Status:** Working synthesis / architectural vision  
**Related work:**
- [Abstraction supervision as metacognitive control](./abstraction_supervision_as_metacognitive_control.md)
- [Deep abstractions from small data](./deep_abstractions_from_small_data.md)
- [Self-modeling and the failure of RL in high-dimensional domains](./self_modeling_and_rl_in_high_dimensions.md)
- [Replay meta-model memorization subspace discovery](../experiments/zipfian_grokking/replay_metamodel_memorization_subspace/README.md)
- [Contrastive replay pressure/seam/background](../experiments/zipfian_grokking/contrastive_replay_pressure_seam_background/SECOND_LEARNINGS.md)
- GLP paper (Luo et al., 2026)[^private]

---

## 1. What this note captures

This note documents a specific vision of how the metacognitive control
principles discovered in our Zipfian grokking toy experiments might map onto a
realistic, scaled-up system for novelty learning in large models. It describes
the architectural components, the analogies that ground them, where those
analogies feel solid, and what remains genuinely uncertain.

The core picture:

> A model encounters something surprising. A generative meta-model (GLP)
> surfaces what is happening representationally during the model's attempt to
> resolve the surprise. The model's own built-in value sense evaluates whether
> the resolution attempt reflects genuine depth or surface memorization. Replay
> of hyper-compressed artifacts (art, elegant proofs, etc.) serves as the test
> material against which candidate abstraction-level revisions are validated.
> A control pathway — still underspecified — modulates whether and how the
> resulting representational changes get consolidated.

---

## 2. The analogy from the toy setting

The Zipfian grokking experiments established a concrete causal chain:

1. Concentrated signal (high-Zipf-weight samples) creates ambiguous gradient
   pressure.
2. That pressure has both a generalization component (Fourier structure) and a
   memorization component (a ~1-D bad axis in h1).
3. A replay-consuming meta-model can discover the bad axis from replayed
   internal states alone, even from healthy-state drift before pathology is
   visible.
4. Suppressing the discovered axis eliminates catastrophic collapse.
5. A contrastive pool decomposition (pressure / seam / background) reveals
   richer structure — a 2-D pathological subspace including a
   re-specialization mode — and the resulting intervention is causally valid
   despite being geometrically distinct from the 1-D memorization axis.

The proposed real-data version maps each element:

| Toy setting | Real-data analogue | Analogy quality |
|---|---|---|
| High-Zipf-weight samples | Surprising inputs (high prediction error) | **Partial** (see §3.1) |
| Gradient from those samples | Representational response to surprise | Good for downstream structure |
| PCA / replay probes on h1 | GLP meta-neurons encoding the resolution attempt | Good |
| Test accuracy / future collapse | Model's own value function (built-in taste) | Good |
| Seam examples (structural canaries) | Hyper-compressed artifacts (art) | **Partial** (see §3.4) |
| Gradient projection of bad axis | Control pathway (underspecified) | Open |

---

## 3. The components

### 3.1 Trigger: concentrated gradient pressure (not surprise per se)

Surprise — deviation from the model's own predictions — is a natural trigger
for when metacognitive monitoring should activate at scale. But the mapping
from the toy setting is weaker here than we initially claimed, and the
disanalogy is worth being precise about.

In the toy setting, the top-Zipf-weight samples generate 93% of the
memorization gradient. But they are **not genuinely surprising** to the model.
At the Fourier solution, the model predicts them correctly — they have
near-zero raw loss. Their gradients are large because we artificially
amplified their weight, not because the model's representations are locally
inadequate. The "surprise" is in the weighting scheme, not in the model's
predictive failure.

This matters because in a real-world setting (e.g. LLM pretraining with
uniform example weighting), high prediction error carries **semantic
information**: it tells you something about what the model doesn't yet
understand. The content of the surprise is meaningful. In the Zipfian toy
setting, the content of the amplified gradient signal is semantically
vacuous — it tells you about the weighting scheme, not about the model's
understanding.

What the toy setting *does* model well is the **downstream consequence**:
concentrated gradient pressure that is ambiguous between productive and
pathological directions, regardless of why it is concentrated. Both genuine
surprise (real-world) and artificial amplification (Zipfian) produce large
gradients that need decomposition. The control problem — how to separate
generalization signal from memorization signal in an ambiguous gradient — is
analogous even if the upstream cause differs.

The honest mapping is therefore:

- **Toy:** concentrated gradient pressure from artificial weighting
- **Real-world:** concentrated gradient pressure from genuine predictive failure
- **Shared downstream structure:** ambiguous large gradients needing
  decomposition
- **Not shared:** the semantic informativeness of the trigger

At scale, surprise-as-trigger has an advantage the toy setting lacks: the
fact that the model was surprised is itself evidence about which abstractions
are missing. This additional semantic channel is not present in the toy
problem and would need to be tested separately.

An additional possibility: the GLP's own diffusion loss on the task model's
activations could serve as a model-internal notion of surprise ("the model's
internal state is unusual right now"), which may be a better trigger than
output-level prediction error in some cases.

### 3.2 Feature surface: GLP meta-neurons

The GLP (Luo et al., 2026) is well-suited as the feature surface for the
metacognitive controller. Key properties:

- **Meta-neurons isolate concepts into individual units** without structural
  assumptions, outperforming SAEs, raw neurons, and raw layer outputs on 1-D
  probing (Table 4 of the paper).
- **Diffusion loss measures typicality** — the paper flags this explicitly as a
  future direction. High GLP loss on an activation = off-manifold state.
- **Scales predictably with compute** — more compute yields better concept
  isolation and better downstream utility, tracking the diffusion loss.

So when the task model encounters something surprising and tries to resolve it,
the GLP's meta-neuron activations provide a decomposed, interpretable readout
of which conceptual dimensions the resolution attempt is engaging. This is the
scaled-up analogue of PCA on h1 in the toy setting.

### 3.3 Value function: the model's existing intuitions

We presuppose that sufficiently capable models already encode a strong enough
value function for evaluating representational quality. They know beauty when
they see it, albeit perhaps shallowly at first. This is a meaningful
assumption, but a reasonable one for frontier-scale models.

This sidesteps the "value function gap" that would otherwise be the hardest
component to specify. In the toy setting, the value function was clean (test
accuracy / future collapse prediction). At scale, the model's own taste and
judgment — trained through pretraining on vast human-generated data — serves
the analogous role.

The question the value function answers is not "did this input get predicted
correctly?" but something closer to:

> "Does the representational response to this surprise look like it reflects
> genuine depth, or surface pattern-matching?"

The model's existing intuitions are likely sufficient for this judgment in many
domains, even if they are not perfectly calibrated.

### 3.4 Replay material: hyper-compressed artifacts (art)

In the toy setting, seam examples were maximally sensitive probes of
representational quality — the first to break when narrow pressure corrupted
the representation, and the most informative single signal about whether the
model's internal structure was sound.

In the real world, the natural analogue is **hyper-compressed artifacts**: art,
great writing, elegant proofs, beautiful experiments — data that was designed
or selected to compress deep structure into concentrated form.

The connection: both are **disproportionately diagnostic probes of abstraction
quality**. Both answer the question "is the model's internal structure sound?"
more sensitively than random examples, and both work through *contrast* —
seam examples are informative relative to background (the P+S vs Bg contrast
reveals the full 2-D pathological subspace), and art is informative relative
to the model's current abstraction level.

**However, the mechanism differs in an important way:**

- Toy seam examples test: "is the Fourier structure intact?" — they detect
  **damage to existing abstractions** (a preservation/health signal). Seam
  examples are informationally rich about *model state*, not about *the world*.
  Their sensitivity comes from their *position in input space* (sharing tokens
  with pressure examples), not from any intrinsic depth.
- Art tests: "does the model have the deep abstractions needed to find this
  deeply unsurprising?" — it detects **absence of deeper abstractions** (a
  growth/depth signal). Art is informationally rich about *the world*
  (compressed human insight), and its diagnostic power comes from intrinsic
  structural depth.

So seam examples are canaries for representational *health*; art is a
challenge probe for representational *depth*. The first is a maintenance
signal, the second is a growth signal. Both are useful for metacognitive
control, but they serve different roles.

If the model's representations are shallow, art becomes opaque or it engages
superficially. If representations are deep, art becomes richly intelligible.
Art is therefore the natural replay material for the meta-meditation process
described in
[deep abstractions from small data](./deep_abstractions_from_small_data.md):

> Periodically pause. Hold compressed artifacts in working memory. Ask whether
> recent changes have made them more or less deeply intelligible. Consolidate
> only the changes that pass.

Crucially, this sidesteps the "how do you find seams without domain knowledge"
problem. You do not need to identify structural canaries through bespoke
analysis. Hyper-compressed artifacts are identifiable by their nature, and the
model's own value function can recognize them.

---

## 4. The control pathway (unresolved)

This is the component we are least certain about. The toy experiments used
frozen gradient projection — projecting out the discovered bad axis. At scale,
the control pathway must translate metacognitive evaluation into modulation of
learning dynamics. The candidates:

### 4.1 Gradient gating

Evaluate the candidate gradient's effect on GLP meta-neurons, gate components
that look like memorization.

**Concern:** This likely operates at the data-abstraction level (which examples
to weight, which gradient components to keep) rather than the
representation-abstraction level (how the model's internal organization should
change). Our experimental history has not found data-abstraction-level
interventions to be productive. The successful interventions in the toy setting
operated on representational directions, not on data selection.

### 4.2 On-manifold projection via GLP

The GLP paper's SDEdit-style algorithm (Figure 4) projects off-manifold
activations back onto the learned manifold while preserving semantic content.
This could be repurposed from inference-time steering to training-time learning
control.

**Important nuance:** Projecting precisely back onto the manifold would be too
conservative. It would force the model to see everything through the lens of
its existing compressions. Most genuine learning involves some amount of
off-manifold movement — the model needs to develop new representational
structure, not merely reinforce old structure. Purely off-manifold learning
(as in naive RL) is clearly bad, but purely on-manifold learning would prevent
the model from ever expanding its abstractions.

The holy grail is going off-manifold in exactly the ways that are productive
and novel. The model probably has implicit knowledge of which off-manifold
directions are promising (this is what "insight" or "taste" amounts to), but
how to operationalize that knowledge as a control signal is unclear.

A partial on-manifold projection — projecting back *partway*, or projecting
out only the components the value function flags as unproductive — might be
more appropriate than full projection. But the details are speculative.

### 4.3 Replay prioritization / slow consolidation

The most biologically grounded option. Candidate updates go into a buffer.
During offline replay, they are tested against hyper-compressed artifacts. Only
updates that survive — that make art more deeply intelligible without degrading
broad competence — get slowly consolidated into the stable model.

This is closest to the hippocampal consolidation story and to the
meta-meditation process. It is also the softest intervention: it does not
gate individual gradients or project individual activations, but modulates
*which experiences get replayed and consolidated over time*.

**Advantage:** Operates naturally at the representation-abstraction level. The
question is not "should this gradient be applied?" but "should this way of
being become part of me?"

**Disadvantage:** Slow. Requires an explicit replay/consolidation loop that
does not exist in standard training.

### 4.4 Current assessment

We do not know which control pathway is right. Replay prioritization / slow
consolidation feels most aligned with the theoretical picture and with
biological precedent. On-manifold projection via GLP is the most concrete
mechanistic candidate but needs a way to allow productive off-manifold
movement. Gradient gating is the most direct but likely operates at the
wrong level of abstraction.

The honest answer is that the control pathway may require its own line of
experimental work to resolve.

---

## 5. What this picture covers well

1. **When to activate monitoring** — concentrated gradient pressure as
   trigger, with surprise as the natural real-world instantiation (though
   the toy-to-real mapping is partial; see §3.1).

2. **What to observe** — GLP meta-neurons provide a decomposed,
   assumption-free, scalable readout of the task model's representational
   response.

3. **How to evaluate** — the model's existing value function (taste, beauty
   sense) is presupposed as sufficient for frontier models.

4. **What to replay against** — hyper-compressed artifacts (art) are the
   natural real-world seam analogues, identifiable by the model's own judgment.

5. **Why this is needed** — the O(1)/O(d) bandwidth mismatch means scalar
   reward or loss cannot guide high-dimensional representational updates
   without supplementary self-knowledge.

---

## 6. What remains uncertain

### 6.1 The control pathway

As discussed in Section 4. This is the largest open question.

### 6.2 Online vs offline

The strongest toy results came from offline replay — revisiting stored states,
detecting slow drift, comparing across time. The proposed system sounds more
online ("predicting as you go along"). However, learning at scale moves much
slower than in the toy problem, and good representations have no known reason
to rotate. A periodic offline pass — "every N steps, replay art through the
current model, encode via GLP, evaluate" — may be sufficient.

### 6.3 What counts as "surprise" precisely

The toy-to-real mapping for the trigger is weaker than for the other
components (see §3.1). In the toy setting, the trigger is really
*concentrated gradient pressure from artificial weighting*, not genuine
surprise. At scale, prediction error is token-level. The relevant level of
surprise (token, passage, concept) is ambiguous. And there is an open
question about whether genuine surprise (which carries semantic information
about abstraction gaps) enables qualitatively better metacognitive control
than the toy setting's semantically vacuous amplification. The GLP's own
diffusion loss on the model's activations might be a better model-internal
trigger than output-level prediction error.

### 6.4 Whether the value function is actually strong enough

We presuppose this, but it is an empirical question. The model's taste may be
good enough for frontier models but insufficient for smaller ones. It may also
be domain-dependent — strong for aesthetic judgments, weaker for scientific or
mathematical novelty.

### 6.5 The relationship between productive off-manifold movement and the GLP

If the GLP defines "on-manifold" relative to the model's current compressions,
then any genuinely new abstraction must by definition be off-manifold at the
moment of its discovery. The GLP may need to be periodically retrained, or
there may need to be a mechanism by which the metacognitive controller can
distinguish "off-manifold because this is garbage" from "off-manifold because
this is genuinely new." The model's value function is the obvious candidate for
this distinction, but how to wire it into the GLP's manifold model is unclear.

---

## 7. Compressed takeaway

A realistic metacognitive novelty learning system might work as follows: the
model encounters surprise, a GLP surfaces the representational structure of the
resolution attempt, the model's own value sense evaluates whether the response
reflects genuine depth, and hyper-compressed artifacts (art) serve as replay
material for testing candidate revisions. The trigger, feature surface, value
function, and replay material all have clear candidates grounded in the toy
experiments and existing tools. The control pathway — how metacognitive
evaluation actually modulates learning — remains the central open question, with
replay prioritization / slow consolidation feeling most theoretically grounded
but least operationally concrete.

[^private]: Not mirrored: this link points to a document in the private lab repo (the roadmap, the queue, an unrun spec, reading notes, or a conversation). See the top-level README for what is held back and why.
