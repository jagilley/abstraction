# Self-Intervention via Local Replay

**Status**: Idea (not yet implemented)
**Date**: 2026-04-07
**Builds on**: All four experiments (`epistemic_probe`, `residual_semantics`, `residual_transfer`, `curvature_triage`)

## One-liner

A model uses its GLP to detect seams in its own beliefs, then surgically revises them through hippocampal-style local replay — propagating updates outward from the seam until they reach natural representational boundaries.

## The problem

LLM representations are locally meaningful but globally curved (FER). The same concept gets encoded via different circuitry in different contexts. This means:

- A model can hold inconsistent beliefs across contexts without "knowing" it.
- A single context may encode something true that contradicts the model's beliefs everywhere else (a "canary").
- Naive self-consistency as an objective fails because it's majority-vote: it kills canaries.

We need a mechanism that lets the model revise its beliefs surgically — updating where the evidence warrants it, preserving outliers when they're genuinely correct, and scoping the update to the regions where it applies.

## The hippocampal analogy

The hippocampus does three things that map onto the proposed pipeline:

1. **Novelty detection** — flags inputs that deviate from learned cortical patterns. Analogue: GLP residuals.
2. **Local encoding** — stores the novel experience in a sparse, context-specific representation that doesn't immediately interfere with the cortical model. Analogue: local belief update at a specific point on the manifold.
3. **Offline replay** — replays stored experiences interleaved with related cortical memories, letting the neocortex gradually integrate the new information. Analogue: propagation of local updates into neighboring regions.

The critical feature of replay is that it's **interleaved**: the hippocampus doesn't blast the new experience repeatedly — it replays it alongside related existing memories, letting the cortex find the right way to accommodate the new information within its existing geometry. The new experience gets a vote proportional to its evidential weight, not proportional to its frequency.

## Proposed pipeline

### Step 1: Detect the seam

The GLP identifies a representation where the residual is large — the model's activation for a particular input deviates substantially from the learned manifold. This is a belief that doesn't fit the local pattern.

### Step 2: Verbalize the seam in context

Use the residual-steering + description pipeline (from the residual semantics experiment) to generate a natural language description of what's different about this representation. Not "is this right or wrong?" but "in what way does this deviate from what the model would usually do here?"

This works precisely because residuals are locally meaningful even though they don't transfer globally (the key finding from the transfer experiment).

### Step 3: Evaluate via predictive validation

Rather than asking "is this belief consistent with my other beliefs?" (which is majority-vote), the model asks: **"If I update in this direction, do my predictions improve?"**

This is the epistemic authority. Predictive validation is the most principled candidate because it grounds belief revision in contact with evidence rather than internal coherence.

Generate downstream predictions from the current representation and from the updated representation. Check which set of predictions is more accurate. This is expensive (requires held-out evidence or synthetic test cases), but it's the gold standard for distinguishing "outlier because wrong" from "outlier because it noticed something the others missed."

Could also very easily add an external oracle here. 

### Step 4: Local update

If predictive validation supports the revision, apply a local edit to the representation at that point on the manifold. This is a targeted intervention, not a global parameter update.

### Step 5: Track

Maintain a registry of local updates: what changed, where on the manifold, and what the evidential basis was. This is the hippocampal index — the model's record of "things I've learned that haven't been fully integrated yet."

### Step 6: Replay-propagate

For each tracked update:
1. Use the GLP to identify neighboring contexts on the manifold (sample or retrieve inputs whose activations are nearby).
2. Re-run with the update propagated to these neighbors.
3. Check: do predictions improve in the neighboring contexts?
4. If yes: extend the update's scope. If no: mark the boundary.
5. Iterate until the update has found its natural radius of influence.

This is the replay loop. Each iteration extends the radius of propagation if the update survives contact with neighboring contexts. Updates that reflect genuine structure in the world propagate naturally because they improve predictions across contexts. Updates that are idiosyncratic stay local or die — not because of a majority vote, but because they fail the predictive test in neighboring territory.

## Why FER is enabling rather than limiting

The fractured structure of the representation space provides a **natural propagation boundary**. Because different contexts use different circuitry, an update in one "fragment" doesn't automatically leak into unrelated fragments. You have to actively propagate it, and you can stop propagating when you cross a fragment boundary (where the geometry changes enough that the update no longer makes sense).

Analogy: a new experience about "how dogs behave" gets replayed alongside other dog memories and gradually reshapes dog-related cortical representations, but doesn't propagate into representations of tax law — not because someone enforced a boundary, but because the replay process naturally selects related memories, and tax law never comes up as related.

The GLP manifold's local structure provides exactly this. You can identify the neighborhood of related activations (nearby on the manifold), propagate the update there, and let the update die naturally when it reaches regions where the manifold geometry is too different for it to be relevant.

## The role of the task model

At each step of this pipeline, the language model itself does the heavy lifting — generating contexts, evaluating predictions, deciding whether an update helped. The GLP provides spatial structure: where are we, what's nearby, where does the neighborhood end.

Importantly, the GLP surfaces information that the task model doesn't have intrinsic access to. The model doesn't know its own activation geometry — it can't see that its representation for input A is anomalous relative to its typical pattern. The GLP provides this external self-knowledge, analogous to how the hippocampus provides the neocortex with novelty signals that the neocortex can't compute for itself.

Feeding GLP-derived signals (seam locations, neighborhood structure, propagation boundaries) into the task model gives it the ability to reason about its own representational structure — a form of metacognition mediated by an auxiliary system rather than intrinsic self-modeling.

## What this could yield

A model whose belief updates are:
- **Evidence-proportional** rather than frequency-proportional (canaries survive if they predict well)
- **Naturally scoped** by representational geometry (updates propagate to related contexts, stop at fragment boundaries)
- **Auditable** (each update has a provenance trail: which seam, what evidence, how far it propagated)

## Key unknowns

1. **What does "local edit" mean concretely?** Activation patching at inference time? LoRA on a small neighborhood of training examples? Direct weight editing (ROME-style)? The right implementation depends on whether we want transient or persistent updates.

2. **How do you sample GLP neighbors efficiently?** The GLP can sample from the learned distribution, but sampling specifically from the neighborhood of a given point (conditional generation) may require additional machinery — e.g., running SDEdit from the current activation with low noise to get nearby-on-manifold points.

3. **How expensive is predictive validation?** In the hippocampus, replay happens offline during sleep. The computational analogue might be a background process that runs between inference calls, gradually consolidating tracked updates. This is feasible in a deployment setting but changes the system architecture.

4. **Does this work at the scale of a single layer?** All experiments so far use layer 7 of Llama 1B. Real belief revision might require coordinated updates across multiple layers. Multi-layer GLPs (which the codebase supports) could extend the spatial structure to span the depth of the network.

## Connections to existing work

- **ROME / MEMIT**: Direct weight editing for factual corrections. Our approach is more principled about *scoping* — ROME edits globally, we edit locally and propagate only where validated.
- **Representation engineering**: Steering vectors for behavioral control. Our approach adds the novelty-detection and validation steps — don't just steer, detect where steering is warranted and validate that it helped.
- **Constitutional AI / self-critique**: Uses the model's own judgments to refine outputs. Our approach operates at the representation level rather than the output level, and uses an external structure (the GLP) to ground the self-knowledge rather than relying on the model's intrinsic self-assessment.

## Suggested first step

The most testable piece is the replay-propagation loop (steps 5-6) applied to the epistemic probe setting. Take a factual prompt where the model gets the answer wrong. Steer along the epistemic PC0 axis at alpha=+5 (which the epistemic probe showed can flip incorrect→correct). Then: use the GLP to find neighboring prompts on the manifold and check whether the same steering direction improves factual accuracy for those neighbors too. If it does for nearby prompts and stops working for distant ones, that's evidence for naturally-scoped propagation along FER fragment boundaries.

---

## V2: Verbalization-mediated curriculum generation

**Date**: 2026-04-10
**Status**: Idea (updated from V1 based on experimental findings)
**Builds on**: V1 above + completed experiments (`residual_semantics`, `residual_transfer`, `pc_verbalization`, `curvature_triage`)

### What changed since V1

The four experiments resolved several of V1's key unknowns and forced belief updates on the mechanism design:

1. **Residual transfer is dead.** The transfer experiment produced a clean negative: residual directions do not carry meaning between prompts (silhouette score = -0.07, linear probe at chance). The V1 pipeline's Step 6 — propagating an update by applying the same residual to neighboring activations — cannot work as described. FER means the same concept is wired differently in different contexts, so a direction that means "emotional intensification" at one point on the manifold is noise at another.

2. **Verbalization works and is causally on-policy.** The residual semantics experiment showed that individual residuals can be faithfully translated into natural language descriptions (negation test: 78.7% accuracy, p = 8.2 × 10⁻¹⁸). This provides the decontextualization step that raw residual transfer cannot: the verbalized description lives in language space, where shared meaning DOES transfer across contexts. V2 replicated this independently (62.5%, p = 4.5 × 10⁻¹⁸, n=88).

3. **Manifold geometry does not predict which residuals matter.** The curvature triage experiment showed that residual variance and Jacobian norm are anti-correlated (r = -0.934) — both just measure distance from the high-density core, not independent curvature. Quality assessment belongs in language space, not geometry space. This kills V1's implicit assumption that GLP-derived spatial signals alone can triage which seams to act on.

4. **The residual PC space has interpretable structure.** PC verbalization identified five distinct behavioral axes in the factual residual space (epistemic, discourse fluency, cultural/experiential, institutional elaboration, authoritative confidence). This means the residual space isn't just per-prompt noise — there are shared directions within prompt categories, even if they don't transfer between categories.

### The updated insight

V1 proposed propagation in activation space: detect a seam, apply a local edit, then spread it to GLP neighbors by transplanting the same activation-space direction. FER kills this.

V2 propagates in language space instead. The residual is verbalized at its native point (where it IS meaningful), and the verbalized description — not the activation vector — is what gets propagated. The LLM itself serves as the decontextualization engine, translating from FER-fractured activation space into the relatively flat language space where meaning transfers across contexts.

The key realization: **FER makes naive SGD more local than you'd want, not less.** Under FER, training on example X primarily modifies the parameters involved in X's specific circuitry. The weight change preferentially affects X's neighborhood and attenuates as you move to contexts using different circuitry. FER provides natural propagation boundaries for free. The problem was never "how do we keep updates local?" — it was "how do we make local updates propagate to where they should?" The answer: propagate through language, then ground back into diverse training contexts.

### Revised pipeline

#### Step 1: Detect (unchanged from V1)

Compute the GLP residual for a data point — how its activation deviates from the learned manifold. The residual magnitude is a novelty signal: how much does this data point challenge the model's current beliefs?

#### Step 2: Verbalize (replaces V1's "verbalize the seam in context")

Use the residual semantics pipeline: steer with the residual at its native point, generate paired (baseline, steered) completions, and have an LLM describe the semantic delta. The negation test established that this pipeline is causally faithful, not post-hoc rationalization.

The output is a natural language description: "this data point challenges the model's tendency to X when Y" or "the model's representation here deviates from its manifold in the direction of Z."

This step is mechanistically identical to V1's Step 2. What's different is its centrality — in V1, verbalization was one step in a longer chain. In V2, verbalization is the critical bridge that makes everything else work, because it's the only step where information successfully crosses FER fragment boundaries.

#### Step 3: Triage (replaces V1's "evaluate via predictive validation")

V1 proposed predictive validation — generating downstream predictions from the current vs. updated representation and checking which is more accurate. This is principled but expensive and possibly overengineered.

V2 takes a simpler approach: **the LLM judges whether the verbalized deviation is interesting/meaningful/worth acting on.** Models are already good at this — a superficial sense of what constitutes "interesting" or "worthwhile" is baked in from pretraining. The triage question isn't "is this deviation correct?" (which requires ground truth) but "is this deviation substantive?" (which requires taste, and LLMs have taste).

An external oracle (human, stronger model, or retrieval-augmented fact-checker) can supplement this judgment where available, but it's not strictly necessary for the pipeline to function. The curvature triage experiment confirmed that this judgment belongs in language space — manifold geometry can't help here.

#### Step 4: Generate curriculum (NEW — replaces V1's Steps 4-6)

This is the core departure from V1. Instead of:
- V1: Apply a local activation edit → track it → propagate to GLP neighbors in activation space

V2 does:
- Use the verbalized deviation to generate or retrieve diverse training examples that exercise the same conceptual update in different contexts.

The verbalization says "this data point reveals that the model conflates X with Y in context Z." The curriculum generator (an LLM, possibly the same one) produces training examples that tease apart X and Y across contexts Z₁, Z₂, Z₃, ... — different enough to span multiple FER fragments, but thematically unified by the verbalized insight.

This is where the "near-infinite training curricula" come from. Each data point in the original dataset generates a verbalized deviation, and each verbalized deviation seeds a family of augmented training examples. The original dataset is a finite set of points on the manifold; the curriculum is the set of all training signals derivable from the residuals between those points and the model's current representations.

The hippocampal replay analogy still holds, but the mechanism is different: instead of replaying the raw experience to neighboring cortical regions (V1), the hippocampus verbalizes the gist and generates related experiences for the cortex to learn from (V2). This is closer to how human memory consolidation actually works — dreams don't replay exact experiences, they remix and recombine thematic elements.

#### Step 5: Train (simplified from V1)

Train on the augmented curriculum. Each example produces its own local gradient update (FER ensures this is naturally scoped), but collectively they cover the FER fragments where the conceptual update should apply.

No special training procedure is needed — this is just SGD on a well-curated dataset. The intelligence is in the curriculum construction, not the optimization.

#### Step 6: Verify (optional)

Check generalization on held-out contexts. Did the conceptual update propagate to contexts not in the training curriculum? If so, the curriculum was sufficient. If not, generate more diverse examples and iterate.

### What this simplifies

V1 had six steps, two of which (track + replay-propagate) required novel activation-space machinery. V2 has four core steps (detect, verbalize, triage, generate curriculum), all of which use existing capabilities: the GLP for residual computation, the verbalization pipeline (validated by the residual semantics experiment), and standard LLM generation for curriculum construction and triage.

The "local update" question from V1's Key Unknown #1 (activation patching? LoRA? ROME?) resolves trivially: it's just training on data. FER means training on specific examples is already approximately local. You don't need a special local-update mechanism — you need a good curriculum.

### Key unknowns (V2)

1. **How do you calibrate residual magnitude thresholds for triage?** The curvature triage experiment showed geometry doesn't help. The current best guess is LLM-in-the-loop: verbalize, then ask "is this substantive?" But this is expensive (requires verbalization of every residual before filtering). A cheaper pre-filter — perhaps residual magnitude relative to category-specific baselines — could reduce the verbalization budget.

2. **Does curriculum diversity matter more than curriculum size?** The FER structure suggests that covering more fragments (diversity) matters more than repeating within a fragment (size). But we don't know the right balance, and this likely depends on how fractured the model's representations are for a given concept.

3. **How does this interact with catastrophic forgetting?** The augmented curriculum adds new training examples, which means the model might forget things it previously knew. Standard mitigations (replay of old data, EWC, etc.) apply, but the GLP provides a novel forgetting detector: if the manifold shifts substantially after training, that's a signal that the update was too aggressive. Periodic GLP retraining (the "consolidation" step from V1) would track this.

4. **Does this scale to multi-layer GLPs?** All experiments so far use a single-layer GLP (layer 7 of Llama 1B). Real conceptual updates might require coordinated changes across layers. Multi-layer GLPs could detect multi-layer residuals, but the verbalization pipeline would need to handle richer deviation descriptions.

5. **What's the right granularity for verbalization?** Per-prompt residuals are highly specific. PC-level descriptions (from pc_verbalization) are more general but lose prompt-specific information. The right level probably depends on the application: per-prompt for targeted corrections, PC-level for systematic biases.

### Suggested first experiment

The simplest test of V2: take 10 factual prompts where Llama 1B gets the answer wrong. For each:
1. Compute the residual
2. Verbalize it (what is the model's deviation here?)
3. Use the verbalization to generate 20 training examples that address the same conceptual gap in diverse contexts
4. Fine-tune on the 200 augmented examples
5. Test: does the model now get the original 10 prompts right? Does it get related held-out factual prompts right? Does it forget things it previously knew?

This is small enough to run quickly and tests the core claim: that verbalization-mediated curriculum generation can propagate conceptual updates across FER fragments via language space rather than activation space.

### Connection to RL and the self-modeling hypothesis

**Date**: 2026-04-10

The V2 pipeline is an instance of the self-model-augmented training described in [self_modeling_and_rl_in_high_dimensions.md](~/Code/fer/beliefs/self_modeling_and_rl_in_high_dimensions.md), but it solves the bandwidth mismatch more aggressively than gradient filtering.

#### The bandwidth argument

The self-modeling doc identifies a fundamental bottleneck: RL provides O(1) bits of reward to guide updates in an O(d)-dimensional representation space. The model can't tell which dimensions to update, so the gradient is dominated by memorization-like directions (Section 2.2: 28 samples = 93% of memorization gradient, 3% of generalization gradient in the Zipfian case).

The V2 pipeline sidesteps this bottleneck entirely. The GLP residual is O(d)-dimensional — it doesn't just say "this data point is surprising," it says "this data point is surprising *in this specific direction in activation space*." The verbalization step translates this O(d) directional signal into structured natural language. The curriculum generation step re-expands it into diverse training data that exercises the identified deficiency. At no point does the information pass through a scalar bottleneck.

The full bandwidth path: O(d) residual → O(language) verbalization → O(curriculum) training data → O(d) gradient. This is qualitatively different from RL's path: O(1) reward → O(d) gradient (with d-1 dimensions of noise).

**This means verbalization is not an optional optimization — it is the mechanism that provides the high-bandwidth feedback.** Residual magnitude alone is O(1), a scalar. Using it to re-weight training data is prioritized experience replay, which helps at the margin (better compute allocation) but does not solve the bandwidth mismatch. The model still receives an undecomposed gradient and doesn't know which direction to update. The directional information in the residual — accessible only through verbalization — is where the bandwidth advantage lives.

#### V2 as a continuous training loop

V2 as described above reads as a one-shot procedure: identify deviations, generate curriculum, fine-tune, done. The RL framing corrects this. Because the residual is a function of the model's current state, the curriculum is a moving target. After training, the model's representations shift, the GLP residuals change, and the curriculum should update accordingly.

The deployment model is a continuous loop:

1. Forward pass on data → compute GLP residuals
2. Verbalize top-K deviations (or use pre-verbalized PC-level descriptions where applicable)
3. Generate targeted curriculum from verbalizations
4. Train on curriculum (mixed with replay data to mitigate forgetting)
5. Periodically retrain GLP to track evolving manifold (consolidation)
6. Goto 1

This loop has no natural boundary between "pretraining" and "post-training." Every training step is simultaneously learning new content (pretraining-like), correcting specific deficiencies identified by self-modeling (post-training-like), and using on-policy representations to determine what to learn next (RL-like). The model's own forward pass outputs are baked into every stage of curriculum selection.

#### Why this is qualitatively different from data augmentation

The distinguishing feature is that the curriculum is downstream of the model's own forward pass. For the same data, a different model (or the same model at a different training stage) produces different residuals, different verbalizations, different curricula. The training signal is on-policy — it's a function of the model's current representational state, not just the data distribution.

Naive data augmentation generates more data independently of what the model currently knows. It sprays uniformly in data space. The V2 pipeline generates data targeted at the specific gaps in the model's current representations, as identified by the model's own activations compared against its own manifold model. This targeting is what makes it RL-like, and the verbalization is what makes the targeting high-bandwidth rather than scalar.

#### Amortizing verbalization cost

Per-data-point verbalization is expensive. The practical middle ground is amortization rather than elimination:

- **PC-level verbalization**: The PC verbalization experiment showed that residual PCs within a category are interpretable (epistemic axis, institutional elaboration, etc.). Decompose each data point's residual into projections onto pre-verbalized PCs, and route to pre-generated curriculum for the dominant PC. This loses per-prompt specificity but preserves directional information at much lower cost.

- **Cluster-level verbalization**: Cluster residual directions in language space (embedding the verbalizations from a calibration set), verbalize each cluster once, then classify incoming data points by cluster. New data points get routed to the appropriate pre-generated curriculum without individual verbalization.

Both approaches preserve the critical property: directional feedback, not just scalar magnitude. They trade per-point fidelity for cost efficiency while staying above the O(1) bandwidth floor.

#### Predictions (extending self-modeling doc Section 7)

If this picture is correct:

1. **V2-style curriculum should outperform residual-magnitude-weighted training** on generalization to held-out contexts, because magnitude weighting provides O(1) bits while verbalization-mediated curriculum provides O(language) bits of feedback per data point. The gap should widen with the effective dimensionality of the task.

2. **The continuous loop should show compounding returns**: early rounds of curriculum fix the largest deviations, making the GLP residuals cleaner for subsequent rounds, which generates better-targeted curriculum. This is the "polishing" dynamic — each iteration improves the signal-to-noise ratio for the next.

3. **GLP quality should be a better predictor of curriculum effectiveness than curriculum size.** A better GLP (lower diffusion loss) produces residuals that more purely reflect genuine novelty rather than modeling error, which produces more accurate verbalizations, which produces more targeted curriculum. This parallels self-modeling doc Prediction 5 (meta-model quality > reward model quality).
