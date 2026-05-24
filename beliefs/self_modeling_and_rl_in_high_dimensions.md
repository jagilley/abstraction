# Self-Modeling and the Failure of RL in High-Dimensional Domains

**Date:** 2026-03-23  
**Status:** Working hypothesis, supported by toy-problem evidence + theoretical argument  
**Related work:**
- [GLP directional residuals](../experiments/zipfian_grokking/glp_directional_residuals/README.md) — memorization subspace discovery, gradient decomposition (findings 15–19)
- [Suppress memorization subspace](../experiments/zipfian_grokking/suppress_memorization_subspace/README.md) — causal validation that self-knowledge enables targeted intervention
- [Beauty and boredom](../experiments/ideas/beauty_and_boredom.md) — intrinsic gradient quality as self-modeling
- [Abstraction-space supervision](../experiments/ideas/abstraction_supervision.md) — the general framework for representation-level feedback
- GLP paper (Luo et al., 2026)[^private] — "Learning a Generative Meta-Model of LLM Activations"

---

## 1. The Core Claim

**Reinforcement learning's effectiveness in high-dimensional domains is fundamentally bottlenecked by the model's capacity for self-modeling — not by reward quality, data quantity, or compute.**

Ever since the RL-on-language-models breakthroughs (o1, o3, etc.), the field has been chasing a "move 37 for LLMs" moment — the analog of AlphaGo's superhuman creative leap, but for general reasoning. The results have been surprisingly limited: strong gains in math and programming, but comparatively modest progress in more open-ended, high-dimensional domains like scientific reasoning, creative writing, or social understanding.

We believe this is not a scaling problem but a structural one. The missing ingredient is not more reward signal or more compute, but a mechanism for the model to understand its own representational structure well enough to decompose reward feedback into generalizable and memorization-like components.

---

## 2. The Dimensionality Argument

### 2.1 Reward is O(1) bits; representation space is O(d)-dimensional

A single RL rollout reward is a scalar — O(1) bits of information. This scalar must guide an update in a d-dimensional representation space, where d is the effective dimensionality of the model's learned representations (determined primarily by data/task complexity, mediated by model capacity).

The representation space decomposes into:
- A **generalization subspace** of intrinsic dimensionality k, determined by the model's current deficiencies relative to the task
- A **memorization subspace** of dimensionality d − k, comprising all directions that would fit the specific rollout without improving general capability

As effective representation dimensionality grows with task complexity, d − k grows much faster than k, because k is determined by *what's currently wrong with the model* (relatively specific), while d − k is *all the other ways to overfit* (vast).

The reward signal doesn't grow to compensate. The ratio of useful-to-useless gradient directions scales as k/d → 0 for complex tasks.

### 2.2 The magnitude distribution makes it worse

The direction count alone understates the problem. Our Zipfian grokking experiments ([glp_directional_residuals](../experiments/zipfian_grokking/glp_directional_residuals/README.md), findings 15–16) directly measured the gradient decomposition:

- **28 samples (1% of training data) contribute 93% of the memorization gradient but only 3% of the generalization gradient**
- **The memorization direction captures 85% of the effective gradient magnitude** at pre-collapse
- The generalization gradient is distributed across the population and quiet; the memorization gradient is concentrated in a few samples and loud

This isn't an accident of Zipfian weighting. It's a structural consequence of how loss gradients work: the samples with the highest loss/reward signal shout the loudest in gradient space, and their shouting is overwhelmingly along memorization directions, because the very features that make them extreme (unusual, surprising) are the features that don't generalize.

### 2.3 RL recapitulates Zipfian dynamics

Policy gradient methods (REINFORCE and descendants) deliberately amplify high-advantage rollouts. This creates exactly the Zipfian gradient structure:

- A few extreme rollouts dominate the policy gradient (analogous to top-2 Zipfian samples carrying 52.6% of gradient weight)
- These extreme rollouts are extreme because they're *atypical* — the model did something it doesn't usually do, and it happened to work (or fail spectacularly)
- Their gradient directions are therefore idiosyncratic — specific to the particular features of that rollout, not the general capability
- The generalization signal is carried by the many moderate-advantage rollouts, which are individually quiet

The key insight: **high-reward rollouts are more off-manifold than medium-reward ones.** A rollout that gets surprisingly high reward succeeded for reasons the model doesn't typically use — its representations during that rollout were off the model's typical activation manifold. The gradient from that rollout says "make your representations more like *this*," where "this" is the off-manifold configuration. That's a memorization-like update.

---

## 3. Why Pretraining Succeeds and RL Struggles

### 3.1 Pretraining: forced compression → natural generalization

Pretraining operates in a regime where d_model << d_data. The model can't possibly memorize the dataset, so it's forced to compress. Compression naturally selects for low-frequency, widely-applicable structure — the generalization subspace. The "memorization gradients" from individual examples point in different random directions and cancel out under the (roughly) uniform weighting of standard pretraining. What survives the averaging is exactly the structural signal.

This is the "build phase" in our Zipfian experiments: weight decay + distributed gradient pressure naturally pushes toward the Fourier solution (the correct generalizing representation). The model achieves 99%+ accuracy during this phase.

### 3.2 RL fine-tuning: excess capacity → memorization dominance

RL fine-tuning operates in the *opposite* capacity regime: d_model >> d_task_improvement. The pretrained model has enormous representational capacity, and the RL signal is trying to make a targeted improvement to one specific capability. The model has vast amounts of spare capacity available for overfitting to individual rollouts.

**The ratio flips.** During pretraining, limited capacity forces generalization. During RL, excess capacity enables memorization. And the RL reward signal, being lower-bandwidth than pretraining's dense next-token prediction, provides less information to disambiguate the two.

### 3.3 Why math and code partially escape

Math and programming show the strongest RL gains because they have **verifiable reward signals** — you can check if the proof is valid, if the code passes tests. This provides higher-bandwidth feedback (not just "good/bad" but "specifically wrong at this step"), effectively increasing the O(1) bits per rollout to something much larger.

More importantly, in math and code, the memorization and generalization directions are better aligned than in other domains. Learning to produce a correct proof *does* tend to generalize to similar proofs, because mathematical structure is comparatively low-dimensional relative to the representation space. The surface features of a correct proof (notation, variable names, proof order) are relatively independent of its correctness, so the reward signal doesn't get confused by them.

In more subjective domains — writing, reasoning, social understanding — the reward signal is lower-bandwidth, noisier, and the surface features that vary across rollouts are more entangled with the quality being rewarded. The gap between "fits this specific feedback" and "genuinely improves" grows accordingly.

---

## 4. Self-Modeling as the Missing Bandwidth

### 4.1 The bandwidth mismatch

The fundamental issue is a **bandwidth mismatch**: O(1) bits of reward trying to guide an O(d)-dimensional update. The reward signal simply doesn't contain enough information to specify which of the d dimensions to update.

Self-modeling provides the missing bandwidth. A model of the model's own representational structure contributes O(d)-scale structural knowledge that supplements the reward's O(1) bits. Instead of "this rollout was bad, here's a gradient, apply it blindly," self-modeling enables: "this rollout was bad, here's a gradient, but I know what my healthy representations look like, and 85% of this gradient is pushing me off-manifold — apply only the 15% that's consistent with my existing representational structure."

### 4.2 Evidence from the memorization subspace experiments

Our experiments provide direct evidence for this mechanism:

**The GLP as self-model.** The Generative Latent Prior (Luo et al., 2026[^private]) trained on a model's activations learns the manifold of the model's representational structure. In our Zipfian grokking setup, the GLP's residuals (deviations from the learned manifold) precisely identify the memorization direction — a 1D subspace responsible for collapse — without any knowledge of Zipfian weights, reward signals, or the task itself.

**Causal validation.** Projecting out the GLP-identified memorization direction from the encoder gradient ([suppress_memorization_subspace](../experiments/zipfian_grokking/suppress_memorization_subspace/README.md)):
- Eliminates all three Sisyphean collapses
- Sustains 99.8%+ test accuracy for 80,000+ epochs
- Preserves 97% of the generalization gradient while removing 85% of the memorization force
- Works using a *frozen* direction identified at a single timepoint, confirming the memorization subspace is a structural feature, not a transient artifact

**PC1 as cheap self-knowledge.** Even without a full GLP, the model's own PC1 of activations is a near-perfect proxy for the memorization direction (cosine > 0.97 at pre-collapse; [finding 17](../experiments/zipfian_grokking/glp_directional_residuals/README.md)). The information for self-modeling is already present in the model's representations — it just needs a pathway to influence the training loop.

### 4.3 The neocortex–hippocampus analogy

This maps onto the bidirectional relationship between neocortex and hippocampus in biological learning:

- **Hippocampus (GLP):** "Here's what your representations typically look like" — the manifold model, a fast-learning system that tracks the current distribution of internal states
- **Neocortex (task model):** "Here's which aspects of my representations matter for the task" — the domain understanding that determines what should be considered "on-manifold"
- **Bidirectional communication:** The task model's self-knowledge (e.g., "PC1 is memorization") shapes what the hippocampus considers normal. The hippocampus's manifold model provides the reference frame for evaluating new updates.

The beauty/boredom framework ([beauty_and_boredom.md](../experiments/ideas/beauty_and_boredom.md)) formalizes this: a gradient is "beautiful" (worth following) to the extent that it would improve performance on *other* data points — i.e., to the extent that it's on-manifold. Computing beauty requires population-level self-knowledge, which is exactly what the GLP provides.

Humans who learn fastest from high-dimensional experience aren't the ones with the strongest reward signals; they're the ones with the most accurate self-models. The medial orbitofrontal cortex (aesthetic judgment, value assessment) communicates bidirectionally with the hippocampus — the neocortex shapes what the hippocampus considers "normal," making this a two-way loop, not a passive lookup.

---

## 5. The "Many Valid Directions" Counterargument

A natural objection: in subjective domains (writing, social reasoning), there are many valid outputs — many good essays, many appropriate responses. Doesn't this mean the generalization subspace is also large, offsetting the growth of d?

This confuses three distinct things:

### 5.1 Many valid outputs ≠ many valid process improvements

The fact that there are millions of good essays doesn't mean there are millions of directions that would improve the model's essay-writing *process*. The model has specific deficiencies (doesn't understand irony, clunky transitions, poor emotional calibration). The directions that would fix these are relatively specific at any given point in training. The many valid essays live on a solution manifold in output space; the useful gradient directions are about *getting to* that manifold from the model's current position, which is a much more constrained problem.

### 5.2 Subjectivity increases reward noise, not signal dimensionality

If different human raters disagree about whether an essay is good, each bit of reward becomes *noisier*. This makes the problem *harder*, not easier — less information per unit of feedback, not more.

### 5.3 The solution set vs. the gradient

There may be many valid directions *on the solution manifold* (many ways to be a good writer). But the model isn't on the manifold yet. From its current off-manifold position, the useful directions are those that close the gap to the manifold. These are determined by the model's current representational deficiencies, and they're relatively low-dimensional — even in a subjective domain.

Analogy: navigating a 10,000-dimensional space toward a 500-dimensional manifold of "good solutions." The manifold is large and has many points on it. But from the current off-manifold position, the directions that reduce distance to the manifold are determined by the local normal geometry — specific to *where you currently are*, not to how big the manifold is.

---

## 6. RL May Be *Harder* Than the Zipfian Setting

One important disanalogy between the Zipfian toy problem and real RL deserves emphasis.

In the Zipfian experiments, the memorization direction is **stable** across collapses (cross-collapse cosine > 0.88). This is because the Zipfian weights are fixed — the same samples always dominate. Stability makes the pathology identifiable: you find the direction once, project it out, done.

In RL, the high-advantage rollouts change from batch to batch, so the memorization direction **rotates constantly**. Each batch pushes the model off-manifold in a different direction, creating a random walk in representation space rather than a systematic push along one axis.

This makes RL potentially *worse* than the Zipfian case: the simple intervention (identify one direction, suppress it) doesn't work. You need a **full manifold model** — something that can detect off-manifold distortion in *any* direction, not just a pre-identified one. This is precisely what the GLP provides: it models the entire activation manifold, so it can catch memorization-like updates regardless of which direction they come from.

The Zipfian experiments are therefore a *lower bound* on how much self-modeling matters. In the Zipfian case, even PC1 of activations suffices (a 0-dimensional self-model, essentially). In real RL, the full machinery of a generative meta-model may be necessary.

---

## 7. Predictions

If this picture is correct, it makes several testable predictions:

1. **RL gains should plateau in high-dimensional domains without self-modeling.** Scaling reward model quality and RL compute should show diminishing returns beyond a certain point, because the bottleneck is not reward fidelity but the model's inability to decompose reward feedback into generalizable and memorization-like components.

2. **Self-model-augmented RL should show superlinear returns with model scale.** Larger models have richer internal representations, meaning their self-models (if trained) have more structure to exploit for gradient decomposition. The returns to self-modeling should *grow* with model scale, unlike the returns to reward engineering, which should diminish.

3. **The domains where RL struggles most should be those with the highest effective representation dimensionality and the lowest reward bandwidth.** Creative writing, open-ended reasoning, social understanding — high-d, low-bandwidth reward. Math and code — comparatively low-d, high-bandwidth reward. This matches the empirical pattern.

4. **In the Zipfian toy problem, replacing the fixed Zipfian weighting with a rotating high-weight set (simulating RL's changing rollouts) should make the memorization subspace suppression experiment harder — requiring a full manifold model rather than a single projected direction.** This would directly validate the claim that RL is harder than the Zipfian case.

5. **Meta-model quality should be a better predictor of RL fine-tuning success than reward model quality,** holding other factors constant.

---

## 8. The Heterodox Implication

The mainstream approach to improving RL on LLMs focuses on reward engineering: better reward models, process reward, verified outcomes, synthetic data for reward training. This is essentially trying to increase the O(1) bits per rollout — making the reward signal more informative.

We believe this approach has diminishing returns. The fundamental bottleneck is the O(d)/O(1) bandwidth mismatch between representation space and reward signal. Making the reward slightly more informative (say, O(10) bits instead of O(1)) doesn't change the asymptotic picture when d is in the thousands or millions.

The alternative: invest in the model's self-modeling capacity. Give models rich, accurate models of their own representational structure — whether through GLP-like meta-models, or through architectural innovations that allow self-reflection during training, or through training procedures that explicitly develop and leverage self-knowledge.

This reframes the problem. The question isn't "how do we get better reward signals?" but "how do we give the model the self-awareness to *use* reward signals intelligently?" The former is an engineering problem; the latter is a scientific one, touching on metacognition, representation learning, and the nature of self-knowledge in neural networks.

If this is right, the path to RL breakthroughs in high-dimensional domains runs through self-modeling — and the "move 37 for LLMs" moment will come not from scaling RL, but from giving models the capacity to understand their own minds well enough to learn from sparse, noisy feedback in the vast spaces where most of human knowledge lives.

---

## 9. Open Questions

- **What's the right architecture for self-modeling at scale?** The GLP (a diffusion model on activations) is one approach, but is it the most efficient? Could the model's own forward pass be modified to include self-reflection, avoiding the need for an external meta-model?

- **How does self-modeling interact with the number of RL iterations?** In the Zipfian case, self-model quality degrades over time as the GLP tracks the drifting activation distribution (the "grounding problem" from [beauty_and_boredom.md](../experiments/ideas/beauty_and_boredom.md)). Does this mean the self-model needs to be periodically re-anchored, or can bidirectional neocortex–hippocampus communication keep it calibrated?

- **Is there a phase transition?** Does self-modeling become necessary above some critical d/k ratio, or is it always helpful? Our Zipfian experiments suggest even modest self-knowledge (PC1) provides enormous benefit, but this is a 128-dimensional space. The picture may be qualitatively different at scale.

- **Can self-modeling be learned end-to-end?** Rather than training a separate GLP, could the model learn to self-model as part of its own objective? This is related to the rumination loss idea ([rumination_loss.md](../experiments/ideas/rumination_loss.md)) and the broader question of whether metacognition can emerge from standard training or requires explicit architectural support.

[^private]: Not mirrored: this link points to a document in the private lab repo (the roadmap, the queue, an unrun spec, reading notes, or a conversation). See the top-level README for what is held back and why.
