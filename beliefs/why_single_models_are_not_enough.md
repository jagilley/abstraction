# Why Single Models Are Not Enough

**Date:** 2026-06-18
**Status:** Working theoretical argument
**Related work:**
- [A2A forward model experiments](../experiments/a2a_forward/README.md)
- [Local prediction-error learning](../ideas/local_prediction_error_learning.md)
- [Cerebellar abstraction ratchet](../ideas/cerebellar_abstraction_ratchet.md)
- [Metacognitive novelty learning](./metacognitive_novelty_learning.md)
- Achille & Soatto (2018), "Emergence of Invariance and Disentangling in Deep Representations"
- Ben-David et al. (2010), "A theory of learning from different domains"
- Ahuja et al. (NeurIPS 2021), on information bottleneck + invariance
- Ilya Sutskever, "We're moving from the age of scaling to the age of research" (Dwarkesh Podcast, 2025)

---

## 1. The observation

Models generalize dramatically worse than people. Ilya Sutskever puts it bluntly: "These models somehow just generalize dramatically worse than people. It's a very fundamental thing." They score well on evals but oscillate between two bugs in a vibe-coding session. They learn from orders of magnitude more data but transfer worse to new situations. They can be superhuman at competitive programming and yet lack the "it" factor that lets a talented student generalize from 100 hours of practice.

Something is missing, and we believe it is architectural, not just a matter of scale or training recipe.

The observation underlying this note: every lineage that independently evolved intelligence — mammals, corvids, cephalopods — converged on a hierarchical architecture where multiple neural subsystems model each other. The cortex-cerebellum loop in mammals. The DVR-cerebellum circuit in birds. Functionally analogous structures in octopuses. No lineage solved intelligence with a single homogeneous network, despite the metabolic cost of maintaining multiple subsystems.

This convergence suggests a universal constraint. This note tries to identify that constraint and argue that it applies to artificial models too.

---

## 2. The claim

A single model optimizing a single objective cannot decompose its own intermediate computation into what's invariant (stable across distributions) and what's variant (distribution-specific). This decomposition requires an external model with bounded capacity. Without it, the learner cannot do targeted adaptation under distribution shift, and pays a regret penalty that compounds over time.

More plainly: a single model doesn't know which parts of its own computation are general-purpose plumbing and which parts are tuned to the specifics of what it's currently looking at. An auxiliary model — by virtue of being small — is *forced* to capture only the general-purpose part. The residual (what the small model can't predict) flags the distribution-specific part. This decomposition is what enables efficient learning in a changing world.

---

## 3. The setup

Consider a learner in a non-stationary environment — a sequence of distributions D_1, D_2, ..., D_T, each observed for a limited number of samples before the world shifts. This is realistic: it describes biological organisms, deployed AI systems, and any learner that must generalize across contexts, tasks, or time.

The learner has a model M that computes intermediate representations. At some intermediate point, M applies a transformation g: h_k → h_L (mapping activations at layer k to activations at layer L through several layers of computation). This transformation g is defined by M's weights — it's a property of the model, not of the data.

The question: when the distribution shifts from D_t to D_{t+1}, how should the learner adapt? Which parts of its computation need updating, and which should be preserved?

---

## 4. Why a single model can't answer this question

A single model trained on output loss L receives gradient ∂L/∂θ. This gradient tells every parameter how to change to reduce the output loss. But it carries no information about *which aspects of the intermediate computation are general-purpose vs distribution-specific*.

Two intermediate computations that produce the same output loss but through different internal processes — one using general-purpose, transferable circuits and the other using distribution-specific shortcuts — generate the same gradient. The output loss is blind to the *character* of the computation, only its *quality*.

This means a single model, when facing a distribution shift, must treat all of its computation symmetrically. It has no basis for deciding what to protect and what to update. It either updates everything (risking catastrophic forgetting of general structure) or updates nothing (failing to adapt). This is the fundamental problem.

---

## 5. What a second model provides

Now give the learner an auxiliary model F — small, maybe 1% of M's size — trained to approximate M's intermediate transformation g. F predicts h_L from h_k.

Because F has limited capacity, it can only represent the *low-complexity component* of g — the regular, compressible patterns in how M maps early activations to later activations. Everything beyond F's capacity falls into the residual R = g(h_k) - F(h_k).

This decomposition is the key. S = F(h_k) captures what's compressible. R captures what isn't. And the crucial insight is: **compressible ≈ invariant**.

---

## 6. Why compression produces invariance (the key lemma)

This is the load-bearing step, and it rests on an established result.

Achille & Soatto (2018) proved an invariance-minimality equivalence: for a representation z of input x that is sufficient for predicting task target y, the mutual information between z and any nuisance factor n satisfies:

    I(z; n) ≤ I(z; x) - I(x; y)

with equality for some n. A sufficient representation is maximally invariant to nuisances *if and only if it is minimal* — maximally compressed while remaining sufficient.

What does this mean for us? The auxiliary model F produces a compressed representation of g's computation. If this compression preserves enough information for the downstream task (i.e., F captures enough of g to be useful), then by Achille-Soatto, F's output is *automatically invariant to nuisance factors* — aspects of the input distribution that are irrelevant to the task.

The residual R contains everything F compressed away. By construction, this includes the nuisance-contaminated component — the parts of g's computation that covary with distributional specifics rather than with the invariant transformation.

When the distribution shifts:
- Nuisance factors change (that's what makes them nuisances — they're the aspects that vary across distributions).
- S, being invariant to nuisances, transfers.
- R, containing nuisance information, degrades.

This isn't a heuristic — it follows from the information-theoretic structure of compression. Low-capacity models can't *afford* to encode distributional nuisances. They're forced to spend their limited capacity on what's consistent and regular. The invariance is a *consequence* of the capacity constraint.

Ben-David et al. (2010) provide the complementary bound from domain adaptation theory. Their transfer bound includes a divergence term d_{H∆H}(D_S, D_T) that depends on the hypothesis class H. A lower-capacity class yields smaller divergence because it cannot distinguish source from target as finely. The transfer gap for S (which lives in F's low-capacity class) is tighter than for R (which lives in the high-capacity complement).

The PAC-Bayesian version (Germain et al., 2019) makes it even more explicit: the transfer bound tightens with KL(ρ‖π), which is literally the description length of the learned model. Shorter description = tighter transfer guarantee. F has a short description by construction.

### The critical qualification

Ahuja et al. (NeurIPS 2021) proved that compression alone is not enough — you also need sufficiency. A model that compresses away task-relevant information is invariant to everything, including the signal. That's useless. The optimal point is maximum compression subject to sufficiency — the information bottleneck.

This maps onto our experiments directly. The 1% forward model compresses aggressively, capturing the coarse structure of g but losing fine detail. The 10% forward model captures nearly everything, including nuisances — sufficient but not minimal. The optimal capacity is the information bottleneck: maximum compression while preserving enough of g for the task.

---

## 7. A subtlety: why the naive framing fails

The obvious version of the key lemma would be: "S is more stable than R across distribution shifts." This is tempting but literally false for a fixed model.

If M's weights are frozen, both S(x) = F(h_k(x)) and R(x) = g(h_k(x)) - F(h_k(x)) are deterministic functions of x. They don't "change" when the distribution shifts — they produce the same output for the same input regardless of which distribution that input was drawn from. And if we ask how much F*_D differs from F*_{D'} (the optimal F under two different distributions), the residual changes by exactly the same amount: ΔR = -ΔS. The decomposition doesn't make R inherently more unstable.

So the lemma cannot be about the stability of the decomposition itself. It has to be about the **cost of adaptation** — specifically, the sample complexity of figuring out what needs to change and executing the change.

The right framing: when the distribution shifts, the learner needs to (a) verify which components of its computation are still valid, and (b) update the invalid ones. The decomposition makes (a) cheap for the S component (because F is small and can be verified with few samples) and makes (b) targeted (because the learner knows which component — S or R — needs the update). Without the decomposition, both (a) and (b) must operate over the full model.

This reframing is important because it locates the advantage in the *learning process*, not in the *representation*. The representations S and R are equally deterministic; the difference is in how efficiently a learner can *reason about* them.

---

## 8. Targeted adaptation and regret

Given the decomposition (S, R), a hierarchical learner can adapt efficiently when the distribution shifts:

1. **Verify S**: evaluate F's approximation quality on data from D_{t+1}. This is cheap — O(dim(F)/ε²) samples — because F is small.
2. **If S transfers** (the common case, by the compression-invariance argument): only update R. Adaptation cost is proportional to the complexity of R, not of the full model.
3. **If S doesn't transfer** (the distribution shift was severe enough to invalidate even the low-complexity component): update everything. No worse than the single-model case.

A single model without F must always take option 3. It can't distinguish option 2 from option 3 because it lacks the decomposition.

The expected regret advantage per distribution shift is:

    ΔRegret ≈ P(S transfers) × cost(S)

where P(S transfers) is high (by the compression-invariance argument) and cost(S) is the sample cost of re-verifying/re-learning the low-complexity component (non-trivial, proportional to dim(F)). Over T shifts, this compounds: the hierarchical learner accumulates a growing advantage because it protects its general-purpose computation across every shift while the single learner risks damaging it each time.

### A more precise statement

**Theorem (sketch)**: Consider a learner facing distributions D_1, ..., D_T, with n_t samples from each D_t.

Let M have intermediate transformation g parameterized by dim(M) = d_M parameters. Let F approximate g with dim(F) = d_F parameters, d_F << d_M.

After each shift D_t → D_{t+1}:

- **Hierarchical learner**: Verify F's quality on D_{t+1} (cost O(d_F / ε²) samples). If F transfers (probability p, bounded below by the compression-invariance argument), adaptation cost is O((d_M - d_F) / ε²). If not, cost is O(d_M / ε²). Expected cost per shift: p · O((d_M - d_F) / ε²) + (1-p) · O(d_M / ε²) = O(d_M / ε²) - p · O(d_F / ε²).

- **Single learner**: Cost per shift is O(d_M / ε²) always.

Cumulative advantage over T shifts: T · p · O(d_F / ε²).

The bound on p comes from Rademacher complexity: F*_D's quality transfers to D' with excess error O(R_n(F) + d_TV(D, D')), where R_n(F) ≤ c/√n is small for a low-capacity F. So p ≥ 1 - O(R_n(F) + δ)/τ for threshold τ on acceptable quality degradation. For low-capacity F and moderate distribution shifts, p is close to 1.

---

## 8. Why the auxiliary model must be separate

A natural objection: can't a single model learn to decompose its own computation internally? Can't later layers learn which aspects of earlier computation are general-purpose?

In principle, maybe. In practice, the gradient signal doesn't support it. The output loss ∂L/∂θ tells each layer how to change to improve the output. It does not tell any layer "this aspect of your computation is general-purpose and should be protected" vs. "this aspect is distribution-specific and should be updated." Both contribute to reducing L, and the gradient treats them symmetrically.

The auxiliary model creates a *second optimization objective* — predicting M's intermediate computation — that is informationally independent of the output loss. This second objective is what separates process-level information (how M computes, which is invariant) from output-level information (what M computes on this specific input, which depends on the distribution).

A single model with a single objective cannot create this separation. The separation requires two objectives, and two objectives require (at least) two models.

---

## 9. Not all compressions are equal: transformation vs manifold

The compression-invariance argument says that a capacity-constrained auxiliary model captures invariant features. But invariant to *what* depends on what the auxiliary model is compressing.

There are two natural choices:

**Forward prediction** (compress the transformation): F takes h_k as input and predicts g(h_k) = h_L. F is approximating g — the function defined by M's weights that maps early activations to later activations. This function g is a property of M, not of the data. When the distribution shifts from D to D', g doesn't change (M's weights haven't changed). What changes is the *distribution of inputs to g* (because h_k's statistics depend on the data distribution). F, being compressed, captures the aspects of g that are consistent across inputs — the invariant structure of the transformation. It discards input-specific variation. The result: F's approximation transfers to D' because g is the same function, and F's compressed model of g doesn't depend on which inputs g was evaluated on during training (up to the Rademacher bound on generalization from any particular sample).

**Autoencoding** (compress the manifold): F takes h_k as input and reconstructs h_k. F is approximating the identity function restricted to the manifold of h_k under D. This manifold IS a property of the data distribution. When D shifts to D', the manifold changes, and F's reconstruction quality degrades in proportion to the manifold change — which can be large even for small distribution shifts, because the manifold can rotate, stretch, or shift in activation space.

The theoretical distinction: forward prediction compresses a **distribution-invariant object** (the transformation g, defined by weights). Autoencoding compresses a **distribution-specific object** (the activation manifold, defined by the data). The invariance-minimality equivalence (Achille-Soatto) says compression produces invariance to nuisances — but what counts as a "nuisance" depends on what you're compressing. If you're compressing a transformation, distributional variation is nuisance. If you're compressing a manifold, distributional variation is *signal*.

This explains the sharpest experimental result we have: the forward model's robustness is distribution-invariant (same advantage on Wikipedia, code, French, math), while the autoencoder's robustness collapses off-distribution (zero advantage on code). Same architecture, same capacity, same compression ratio — but compressing fundamentally different objects.

The implication for the general theory: the argument in this note applies specifically to auxiliary models that compress the **computational process** (the transformation between layers, the dynamics of the system) rather than the **activation distribution** (what the representations look like). Not every form of self-modeling produces invariance. The auxiliary model must model how you compute, not what your activations look like.

---

## 10. The biological prediction

This argument predicts exactly what convergent evolution shows: any learner in a non-stationary environment will benefit from a capacity-constrained auxiliary model of its own computation. The benefit scales with:
- The frequency of distribution shifts (more shifts → more opportunities for targeted adaptation)
- The complexity of the learner's computation (more to decompose → larger absolute savings from protecting the invariant part)
- The capacity ratio between auxiliary and primary model (more compression → stronger invariance, up to the sufficiency limit)

Every biological environment is non-stationary. Every intelligent organism has complex internal computation. The capacity ratio is a tunable parameter that evolution optimizes. The result: every lineage that evolves intelligence long enough discovers the hierarchical architecture, because the regret advantage is universal.

The cerebellum is roughly 10% of brain volume but 50% of neurons — a highly compressed, parallel auxiliary processor. Evolution tuned this capacity ratio over hundreds of millions of years of non-stationary environments. It's the biological instantiation of the information bottleneck applied to self-modeling.

---

## 11. The connection to generalization

Ilya's 100-hour student vs. 10,000-hour student analogy maps cleanly onto this framework.

The 10,000-hour competitive programmer updates all parameters based on competition loss. Every problem, every practice session, drives uniform gradient descent across the full computation. There's no decomposition — every aspect of the student's thinking is equally subject to each training signal. The result: brittle expertise that doesn't transfer. The student learned the *output distribution* of competitive programming but not the *process* of mathematical thinking.

The 100-hour student with "it" has something the other doesn't: self-knowledge about which aspects of their thinking are general-purpose tools (invariant) vs. problem-specific tactics (variant). When they encounter a new problem, they know what to protect and what to adapt. They can transfer their general-purpose tools to a new domain immediately and only need to learn the domain-specific part. This is targeted adaptation enabled by the invariant/variant decomposition.

The "it" factor IS the self-model. It's the capacity-constrained compression of one's own computational process that separates what's transferable from what isn't. Models don't have this because they're single networks optimizing single objectives.

---

## 12. Experimental evidence

Our A2A forward model experiments provide direct evidence for several claims in this argument:

1. **Compression produces distribution-invariant knowledge.** A forward model (predicting M's future activations) produces perturbation robustness that transfers across distributions — Wikipedia, code, French, math, all show the same 2-2.5x robustness advantage. An autoencoder (same architecture, same capacity, but compressing activations rather than the transformation) shows robustness that collapses off-distribution. On code, the autoencoder provides zero advantage over open-loop. This is exactly the prediction: the forward model compresses the transformation (invariant), the autoencoder compresses the manifold (variant).

2. **The capacity constraint matters.** A 1% forward model captures the coarse computational structure but leaves a rich residual. A 10% forward model captures nearly everything — the model fully offloads computation to it, becoming entirely dependent with no net benefit. The information bottleneck has an optimum, and it's between 1% and 10% in our setup.

3. **A single model cannot produce the decomposition from output loss alone.** The open-loop model (no auxiliary model) shows no self-knowledge — R² < 0.05 for predicting its own computational novelty at early layers. The closed-loop model (with auxiliary model) shows strong self-knowledge (R² > 0.40). The self-knowledge emerges from co-training with the auxiliary model, not from the output loss.

4. **The decomposition enables progressive improvement.** Multi-cycle wake-sleep with the gated ratchet produces compounding val loss improvement (48% gap vs open-loop after 4 cycles). The mechanism is implicit regularization: the capacity-constrained auxiliary model selects for parameter configurations whose intermediate computation is maximally compressible — maximally legible to a compressed version of their own computation. This is the information bottleneck applied to learning dynamics.

---

## 13. What this does and doesn't say

**What it says:**
- A single model cannot efficiently adapt in non-stationary environments because it lacks the invariant/variant decomposition of its own computation.
- A capacity-constrained auxiliary model provides this decomposition for free, via the information-theoretic structure of compression.
- This is likely why all intelligent lineages converge on hierarchical multi-model architectures.
- This is likely a major contributor to the generalization gap between humans and current AI models.

**What it doesn't say:**
- That current models are useless. They can learn the output distribution of a stationary dataset very well. The limitation is about adaptation and transfer, not about static performance.
- That any specific architecture (our A2A setup, cerebellum-cortex, etc.) is the only way. The argument is about the *function* (capacity-constrained self-modeling) not the *form*.
- That the problem can't be partially mitigated by other means (diverse training data, regularization, etc.). The argument is about what's *optimal*, not what's *impossible* without hierarchy.

---

## 14. Compressed takeaway

A single model optimizing a single loss has no way to know which parts of its own computation are general-purpose and which are distribution-specific. An auxiliary model with bounded capacity provides exactly this decomposition — the capacity constraint forces it to capture only the invariant component of the computation, by the information-theoretic equivalence between compression and invariance (Achille-Soatto). This decomposition enables targeted adaptation: protect the invariant part, update the variant part. Without it, the learner must update everything on every distribution shift, wasting samples re-learning what it already knew and risking damage to general-purpose structure. This is why every lineage that evolved intelligence converged on hierarchical multi-model architectures, and likely why current single-model AI systems generalize dramatically worse than humans despite seeing dramatically more data.
