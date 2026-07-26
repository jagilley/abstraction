# Heterogeneous graders: no signal is both dense and evaluative, and the disagreement is the fire

**Status**: conceptual synthesis, no new experiment. Every anchor below is an already-measured result in this repo; the *frame* is new and the cut in §9 is unbuilt.
**Date**: 2026-07-25
**Prompt**: a conversation asking in what sense motor learning is a *more complicated* task than distribution modeling à la LLMs — "or else animals would not have needed anything except a cortex."
**Builds on**: [two_timescale_value_loop.md](two_timescale_value_loop.md) (the two-teacher interface this doc generalizes), [cerebellar_abstraction_ratchet.md](cerebellar_abstraction_ratchet.md) (what "text is post-ratchet" means), [physical_control_substrate.md](physical_control_substrate.md), [contra-long-horizon-benchmarks.md](contra-long-horizon-benchmarks.md) (the rotation framing used in §7).
**Beliefs touched**: [cerebellum_and_cognitive_architecture](../beliefs/trees/cerebellum_and_cognitive_architecture.md) (the multi-teacher root; this doc's §5–§6 landed there as a child), [operators_not_footprints](../beliefs/trees/operators_not_footprints.md) (§3 is a third instance of that root), [dimensionality_expansion](../beliefs/dimensionality_expansion.md) (§4b supplies the grader that file's expansion drive never specified).
**Key experiments**: [`mjc/on_policy/`](../experiments/mjc/on_policy/README.md) E0–E2 + [E3](../experiments/mjc/on_policy/directed_on_policy/README.md) · [`mjc/ballistic/`](../experiments/mjc/ballistic/README.md) 4b/4c · [`mjc/drift_value_loop/`](../experiments/mjc/drift_value_loop/README.md) Cut 3 · [`mjc/ballistic/directed/`](../experiments/mjc/ballistic/directed/README.md) S1/S2 · [`inverse_dynamics/`](../experiments/inverse_dynamics/README.md) · [`rhm/ratchet/RHM_META_LEARNING_README.md`](../experiments/rhm/ratchet/RHM_META_LEARNING_README.md).

---

## One-liner

Motor learning is not a harder *function approximation* problem than language modeling — the target is dramatically simpler — so "more complicated" has to mean the **conditions** under which the function must be acquired and used. Five such conditions distinguish them, each priced by a node in `mjc/`, and each mapping to an organ that is unnecessary without it. But the deeper statement is the **dual**: pretraining does not need those organs because a text corpus is the *frozen output channel of systems that had them*. What that inheritance cannot transmit is the ability to run more than one grader at once — and the claim of this doc is that **the multi-grader structure is the fire**, that it is *forced* rather than contingent (no single learning signal can be simultaneously dense and evaluative), and that its functional payoff is the ability to discover that one of your own graders has gone blind. §4b states the same dichotomy from the representation side — **compression is what a dense grader can drive; expansion needs an evaluative one, because opening a dimension makes prediction worse before it makes it better** — which is the more intuitive of the two framings and is why both are kept.

## 1. The version of the question that is wrong

"Motor learning is harder" cannot mean the target function is richer. Our own substrate says the opposite, repeatedly:

- Capacity does not bind on a 2-link arm at all — `M(q)` depends on the elbow angle alone, so the whole configuration-dependence is a smooth function of one variable. It binds at n ≥ 5 ([`arm_substrate/`](../experiments/mjc/arm_substrate/README.md)).
- Cut #2's 232-parameter command-aware FM beat a 70,152-parameter command-blind one ([`arity_torque/`](../experiments/mjc/arity_torque/README.md)).
- Meanwhile language's FM residual is **full-rank 200/256 and still learnable to cosine 0.97** — genuine distributed computation, not noise ([a2a](../experiments/a2a_forward/README.md)).

If difficulty were target complexity, the cerebellum spending ~half the brain's neurons on limb dynamics would be an absurd allocation. So the difficulty is in the *epistemic conditions*, not the function class.

## 2. Distribution modeling is the degenerate limit of motor learning

Same learning problem, five conditions switched off. Each has been switched back on and priced somewhere in `mjc/`, and each maps to hardware that is unnecessary in the degenerate case.

| # | Condition | LLM pretraining | Measured here | Organ it forces |
|---|---|---|---|---|
| 1 | **Commitment under delay** | ground truth every step; maximally reactive | ballistic transmits FM quality **3.0–3.3×** more than reactive (4b); re-adaptation **4.3×** (4c), **5–6×** on the arm, **4.85×** on-policy (E1) | cerebellum (a simulator) |
| 2 | **Measurement is metered** | the loss curve is a free global survey of your own competence | S2's **22× measurement subsidy** made the question unmeasurable; E3 at **1.84×** makes it answerable | midbrain (a drive that allocates) |
| 3 | **Data is endogenous** | corpus fixed; the gradient does not change which tokens exist | \|corr(u,s)\| **0.7** on-policy vs **0.03** teleport (E0); local structure costs **2.5×** (E2) | — (this is what makes 2 bite) |
| 4 | **Type-2 non-stationarity** | corpus approximately stationary | five meta-learners on a stationary objective **all collapse to plain multitask** | the second loop itself |
| 5 | **Two teachers, different types** | exactly one teacher | control is a **near-blind grader** of FM quality (Cut 3, E1, E3) | the cerebellum↔VTA interface |

Condition 1 is the only one that is *physically forced* rather than a choice about how we happen to train: axonal delay and inertia are not electable, whereas 2–4 are degeneracies current practice elects and RL post-training is already partly removing. Condition 4 is the direct answer to *"or else animals would only have needed a cortex"*: **on a stationary DGP a second loop provably manufactures nothing**, so a cortex genuinely suffices — for a stationary world.

## 3. The dual reading: pretraining is distillation from a finished mind's output channel

The table above is a list of absences, and a list of absences cannot explain why LLMs are *astonishing*. The dual does both at once.

Ilya says two things and does not connect them (transcript[^private] §00:08:31): text is *"the whole world as projected by people onto text"*, and *"I don't think there is a human analog to pre-training."* If the first is true the second is not mysterious: pretraining is not a stage in a mind's development at all, it is **distillation from the output channel of an already-finished one**. The corpus is the exhaust of trillions of executions of the full loop, run by other agents, handed over open-loop.

Every row of §2 dualizes. Metered measurement → the corpus is a pre-paid survey; someone already went and looked. Type-2 drift → the corpus is the *fixed point* of systems that lived through drift, so you inherit the invariants without the drift. Endogenous data → the corpus is the trace of closed acquisition loops you did not run. Commitment under delay → text is emitted *after* the motor problem was solved, so the deadline was paid off-camera.

**The mechanism is the ratchet.** Our belief tree holds that each cerebellum↔cortex cycle compresses multi-step computation into a single cortical primitive, and that *experts genuinely cannot decompose their chunks back into primitives*. Text is written **in chunks**; a word is a ratchet output. So pretraining distills from the ratchet's top layer — you receive the compressed primitives without having run one cycle. That predicts the observed profile exactly: fluent use of the primitives in-distribution, and an inability to *manufacture new ones* under shift. Which is [RHM_META_LEARNING](../experiments/rhm/ratchet/RHM_META_LEARNING_README.md) and the [specialization line](../experiments/rhm/specialization/README.md) almost verbatim — the ratchet does not compound on a stationary objective, and only a *direct deep target* moves the frontier, never allocation.

**The bill is already written down in this repo.** [two_timescale_value_loop.md](two_timescale_value_loop.md) says of the somatic marker, unprompted: its power comes precisely from *being* a cheap cached correlation rather than a forward simulation; that cache is exactly where spurious correlations live; and **the spuriousness is invisible in-distribution and manifests only under shift**. A frozen value-and-abstraction cache with no second loop to re-evaluate it *is* a phobia. That is the structural form of Ilya's opening complaint (the model that fixes bug A by reintroducing bug B) and of his central one (*"these models somehow just generalize dramatically worse than people"*) — and of his alignment worry, which he himself reduces to *"are these not all instances of unreliable generalization?"*

**Corrected analogy.** "LLM ≈ cortex" is wrong in both directions. Too stingy: in-context learning is a real fast loop and looks more hippocampal (one-shot binding) than cortical. Too generous: the cortex learns from five teachers, whereas the LLM has one whose signal is *already the other four's output*. The better statement is that **an LLM is a fossilized synthesis of full brains that cannot play its inherited parts against each other in a first-class manner** — and, correspondingly, that a *cortex alone* is closer to a raw-data model (an image model) than to an LLM.

**Completeness bifurcates, and the deficits are complementary.** Cortex-alone is content-poor but **process-complete** — it sits inside a running loop. An LLM is content-rich but **process-empty** — it holds the synthesis of billions of loop-executions and can execute none. These are not two points on one scale. The interesting question is therefore not which is more complete but *what happens when you give the content-rich thing a real process* — which is precisely the bet the field is making with RL post-training, and §8 gives the criterion for when that bet pays.

## 4. The impossibility that forces two teachers

**No single learning signal can be both dense and evaluative.**

- *Dense* requires being **free**: available on every step, which means self-supervised on what actually happened. Every `(s, u, s′)` is a label of the physics, no goal required.
- *Evaluative* requires **referencing outcomes**, and the informative outcomes are the ones you would rather not sample. In the wild, sampling rewards is dangerous — you cannot afford to fall a thousand times to learn to walk.

These pull in opposite directions and no clever objective collapses them. So the two-teacher structure is **derived, not designed** — which is the main reason to expect it to generalize beyond vertebrate anatomy.

The motor version is the crisp one. Reward alone cannot learn movement (too dangerous to sample). Prediction error alone has no preference over movements (Friston's dark room). Neither is a motor learner; the **pair, with an interface**, is. This is also why the [reward-free / ballistic synergy](../experiments/mjc/ballistic/README.md) is such a good deal rather than an awkward one: the one asset maintainable *without* reward is exactly the asset committed movement *depends on*. Same object.

### 4b. The same dichotomy from the representation side: compression and expansion

Dense/evaluative is a statement about **signals**. There is a second statement, about **representations**, that turns out to be the same dichotomy — and it is the more intuitive of the two, so it is worth carrying both.

It is treated as near-dogma in some circles that intelligence *is* compression. That is at most half of it, and our own data says the other half is not optional:

- **Compression alone provably terminates, and we watched it terminate.** [`dimensionality_expansion.md`](../beliefs/dimensionality_expansion.md) decomposes a layer as `R_act ≈ R_comp + R_res` — total directions used, directions the self-model has absorbed as routine, and the un-absorbed frontier. Compression drains `R_res → R_comp` while leaving `R_act` fixed, so **on a fixed dataset the drain runs to the wall**: naive wake-sleep collapses residual rank **23.9 → 13.5** ([GATED_RATCHET](../experiments/a2a_forward/GATED_RATCHET_README.md)). That file already names the a2a ratchet's recurring villain — the absorbing state — as *"the predicted consequence of running a compressor with no expansion drive."*
- **The internal expansion route saturates too.** Re-expressing fixed content across more orthogonal directions raises `R_act` without new data, but only so far: 27.8 → 30.2, then activation-norm inflation by 16 cycles. Only external novelty is unbounded.

**Why expansion needs a different grader — the activation-energy argument.** Copernicus threw away degrees of freedom and got *worse* predictions than Ptolemy; on a description-length metric at the moment of the rotation, epicycles win. So compression is the **endpoint, not the process**: you frequently must expand first, tolerate a worse fit, and cross a barrier to reach the basis in which the compression is even available ([contra-long-horizon-benchmarks.md](contra-long-horizon-benchmarks.md) §I–II). A learner that monotonically descends a compression objective cannot cross that barrier, by construction.

That is exactly the dense/evaluative split, re-projected:

| | graded by | availability | payoff |
|---|---|---|---|
| **compression** | prediction error | dense, every step, free | immediate |
| **expansion** | *has to be something else* | sparse, slow | **deferred, and initially negative** |

Opening a new dimension makes prediction worse before it makes it better, so no dense predictive signal can drive it. Its grader must tolerate deferred, initially-negative payoff — which is what "evaluative" means. **Compression/expansion is dense/evaluative viewed from the representation side rather than the signal side.**

**This retro-explains the shape of the learning-progress drive.** LP is ~0 when mastered (dark room), ~0 when irreducible (noisy TV), and peaks at moderate-and-falling error ([two_timescale_value_loop.md](two_timescale_value_loop.md) §regime table; [curiosity Phase 1](../experiments/a2a_forward/reaching/CURIOSITY_DRIVE_README.md)). We recorded that band-pass shape as an empirical property of a curiosity signal. It is better read as a *specification*: it is what an **expansion grader** has to look like. The whole curiosity line has been building the expansion grader without calling it that.

**And it sharpens §8's claim about LLMs.** Weight decay, width, depth, data mix, when to stop — every one is an expansion decision graded on deferred payoff. It is not that LLM training lacks an expansion loop; **it has one, and it is implemented in humans**, running on a wall-clock of weeks. That is the mechanism behind the observation that LLMs expand only slowly, coarsely, and exogenously: the expansion grader was outsourced to the researcher.

> **Note on our own history.** *Dimensionality expansion under drifting dynamics* was [`physical_control_substrate.md`](physical_control_substrate.md) cut #5, billed in its own text as **"the single biggest open question of the whole program."** It was never run — the drift machinery built to make it free was consumed by the value-loop program for a different purpose, and [`mjc/HISTORY.md`](../experiments/mjc/HISTORY.md) records the abandonment under "Planned and never run" without noting it was the flagship. Two independent routes have now arrived back at it, which is usually the tell that it is the real question.
>
> **Verified unrun (2026-07-25), and the search found the hazard.** No rank or spectral measurement exists anywhere in `mjc/` except cut #1's, and git history shows nothing started-and-abandoned. But cut #1 *did* run a participation ratio (`contact_residual/contact_residual.py:104-119`) on the 8-dim state residual of one FM on a stationary pusher, got a **negative** (contact eff-rank 3.60 > free 2.52), and retired *"residual rank ∝ DGP complexity"* — and a2a independently found `rank ⊥ noise` (language's residual is full-rank 200/256 yet learnable to cosine 0.97). **This repo has found rank to be a weak instrument twice**, so any cut #5 must open by saying why it is not the third time. The argument: the retired claim was *absolute and static*; this one is *differential with a matched control* (does `R_act` move under drift vs. fixed dynamics), which is the Tier-A dissociation form that survived across this tree everywhere absolute magnitudes did not. The corrected design — support-growing drift, chain ≥ 5 so capacity binds, readout on hidden activations — is at [`mjc/README.md`](../experiments/mjc/README.md) §Next steps #3.

## 5. The claim: the fire is heterogeneous graders plus a channel

Not "an outer loop." MAML would have been AGI if it were that, and our own result says an outer loop *identical in kind* to the inner one collapses to plain multitask. The load-bearing property is **different in kind**:

| organ | signal type | dense? | evaluative? |
|---|---|---|---|
| cerebellum | supervised prediction error | yes | no |
| basal ganglia | reward prediction error | no | yes |
| hippocampus | one-shot episodic | no | neither |
| cortex | slow statistical | yes | no |

These are not modules that decompose a task. They are **graders that disagree**. A raw-data net has exactly one grader and therefore cannot *ever* discover that its grader is blind — a failure that is, by construction, invisible from the inside, in the same way and for the same reason as a spurious somatic marker.

So the functional payoff of the multi-organ architecture is not "more capacity" or "division of labour." It is **the detectability of grader blindness**, which single-objective learning cannot have at any scale.

## 6. The evidence is our own methodology, which is why I trust it

[`mjc/HISTORY.md`](../experiments/mjc/HISTORY.md)'s deepest cross-cutting claim is not about value at all: **most of the arc's nulls were instrument failures of escalating subtlety.** Every one was caught by playing two graders of different type against each other.

- [`drift_value_loop/`](../experiments/mjc/drift_value_loop/README.md) Cut 3 — control says flat; value-relevant FM error says there is a clean interior optimum at `b = 0.5` exactly where control is dead flat (0.1% spread). → *"control is a near-blind grader."*
- [E1](../experiments/mjc/on_policy/README.md) — ballistic control stepped to ceiling by m=50 while FM task error kept improving to m=400. The disagreement **is** the finding.
- [E3](../experiments/mjc/on_policy/directed_on_policy/README.md) — control near-saturated at ~0.08; the whole policy ladder resolves only on region-A FM error.
- [The S2 retraction](../experiments/mjc/ballistic/directed/README.md) — the loop's verdict rested on the control grader; re-scored by the sighted one, the privileged oracle moved from **fourth to second**.

None of that is findable with one instrument. Our research process advanced by grader disagreement, and the biology it was studying turns out to have the same structure — the cerebellum→VTA pathway is the wire that lets the supervised grader talk to the evaluative one. Convergence of a process and its object on the same structure is worth more than either alone.

*(The same history also records the counter-lesson, which is the reason to build §9 rather than to rely on discipline: knowing which grader is blind did not stop the very next node from reading a program-level belief update off the instrument it had just published as blind.)*

## 7. Why an image-only model never gets there

The axis is not modality or token count. It is **whether the data passed through a mind.** Photons off a tree carry the tree; a sentence about a tree carries the tree *plus* the compression, valuation, and abstraction pipeline a brain applied to it. Text is post-cerebellar, post-limbic, post-ratchet. Pixels are pre-everything.

Two of our own docs sharpen this past intuition:

- **[contra-long-horizon-benchmarks.md](contra-long-horizon-benchmarks.md)**: to solve a problem is to find the basis in which the answer is legible. The world's causal structure *is* in the pixels, encrypted by a bad coordinate system. Text is **pre-rotated** — humans already found the coordinates and wrote them down. **A text corpus is a corpus of rotations.** So the disagreement with a hypothetical infinite video model is about the sample efficiency of re-deriving every rotation from scratch, not about possibility.
- **[`inverse_dynamics/`](../experiments/inverse_dynamics/README.md)**: recovering a generating process from its outputs is an **inverse problem**, ill-posed exactly to the degree the forward process destroyed information. The map `mind → text` is violently non-injective — many mental states emit the same sentence — and the Jacobian-Conjecture lesson is that the failure is **silent**: regress a single-valued function onto a multivalued relation and you get *"a smooth, confident, mode-averaged answer that is wrong on the collisions."* That is the best one-line description of confabulation we have.

And we **measured the deepening**: as the grokking model compresses onto the task variable, its inverse self-model collapses R² 0.88 → 0.15 while the forward self-model stays pinned at 1.0 — and the collapse *keeps deepening after test accuracy already hit 100%*. Compression destroys recoverability, past the point where any performance metric can see it.

Applied here: **the better a generating system's compression, the less of it survives in its outputs.** Human abstraction is excellent compression. Which is why one can read every paper in a field and not acquire the taste — and why Ilya's own account of research taste is entirely about top-down beliefs that *sustain you when the experiments contradict you*, i.e. a value function robust under shift. Precisely the thing that does not make it into the corpus, because people write down their conclusions and never their reweightings.

**Closure.** Ask what the frozen signature structurally *cannot* contain and two categories fall out: things never verbalized because they are pre-verbal (you cannot write down how to ride a bike), and things never verbalized because they are *updates rather than outputs*. Those are the motor domain and the value domain — exactly the two organs §2 says are required. The requirements reading and the inheritance reading are the same claim from opposite ends.

## 8. What this says about LLMs — the mirror-grader criterion

The LLM has one grader, but its data was produced by many-grader systems, so grader conflict is present in the corpus **as a fossil**. It can emit the surface form of a system catching itself — *"wait, that's wrong"* — with no second grader to actually run.

Under this frame **chain-of-thought is a simulation of grader conflict inside the one channel available**, which predicts its profile exactly: large gains, and an unreliability that no amount of additional CoT fixes, because the check and the checked come from one distribution and go blind together.

The operational criterion, and the most useful output of the frame:

> **An LLM-as-judge is not a second grader. It is the same grader in a mirror** — same corpus, same failure geometry, blind in the same places. A genuine second grader must be **sighted where the first is blind**.

Consequences: most current RL post-training has one grader and a reflection; **verifiable-reward domains (does the code run, does the proof check) are the real second grader**, which is suggestive about why exactly those are the domains where RL visibly works. Note also where Ilya locates the same intuition (§01:29:23): he reaches for diversity *between* agents (self-play, debate, prover-verifier). The brain does it *within* one agent, by having organs with genuinely different objectives — enormously cheaper, and operating on the timescale of a single action rather than of a population.

## 9. The cut this suggests: heterogeneous vs. homogeneous disagreement

[`ballistic/directed/`](../experiments/mjc/ballistic/directed/README.md) S1 established that ensemble disagreement **cannot detect a drift**: every member trained pre-drift agrees, and they are all wrong together. Disagreement finds where you *lack* data, not where your data went *stale*. We recorded that as a falsification of the instrument.

That was disagreement between **homogeneous** members — same objective, different seeds. Blindness that is *structural* is shared by every member, so homogeneous disagreement is blind exactly where its members are. **Heterogeneous disagreement — between graders with different objectives — does not inherit that**, because a grader blind for a structural reason stays blind while the other one does not. Our own data is the existence proof: the control grader and the FM-error grader disagreed *precisely at* control's structural blind spot.

**The cut**: make grader disagreement itself the allocation signal. E3 already runs both instruments side by side and we currently use one to check the other *by hand, in the writeup*. The proposal is a system that detects "my primary grader has gone blind here" and reallocates on that basis. It is cheap on the existing substrate (both graders are already computed per round), it is the smallest experiment that tests the frame directly rather than by analogy, and it is on no current Next-steps list.

Design notes: the honest baseline is homogeneous (seed-ensemble) disagreement, since S1's negative is exactly what the heterogeneous version has to beat; the drift must be **local** ([E2](../experiments/mjc/on_policy/README.md)), or there is no place-dependent blindness to detect; and the noisy-TV control must stay, because "the graders disagree" is also what irreducible noise looks like — the discriminator is that noise-disagreement does not *close* when you collect there.

## 10. Predictions / falsification

- **Falsified if** heterogeneous grader disagreement is no better than homogeneous seed-ensemble disagreement at locating a local drift. This is the §9 cut and it is the load-bearing test.
- **Falsified if** a *matched* second grader of the same type (a second control readout, a differently-seeded reward model) buys the same thing a differently-typed one does. That would mean the payoff is redundancy, not heterogeneity.
- **Falsified if** the dense/evaluative dichotomy is collapsible — i.e. someone exhibits a single signal that is available every step *and* references outcomes without needing to sample the dangerous ones. (Learning progress is the obvious candidate and it is **not** a counterexample: it is a functional of the *dense* signal and is evaluative only about epistemics, not about the world's rewards.)
- **Predicts** that RL post-training pays in proportion to how *sighted-where-the-first-is-blind* its grader is — verifiable rewards ≫ LLM-as-judge — and that scaling LLM-as-judge specifically does not close the generalization gap.
- **Predicts** that video pretraining transfers to motor competence far better than to cognition, and text to cognition far better than to motor, because the relevant axis is *which organ's output channel is this*.

## 11. What we have not shown, and should not claim

- **Nothing here is new evidence.** Every number is re-read from an existing node; the contribution is the frame and the §9 cut.
- **§4b tiers into two very different things.** That compression alone terminates is **measured** (residual rank 23.9 → 13.5; the internal route saturating at 27.8 → 30.2 then inflating) and belongs to the a2a ratchet work. That expansion *requires an evaluative grader*, and that this is the same dichotomy as dense/evaluative, is an **argument** — supported by the LP band-pass shape being the right shape, which is suggestive and not a test.
- We have **not** shown motor learning is harder in any information-theoretic sense. §1 points the other way.
- The exaptation story (§below) is an argument, not a measurement.
- **Irreversibility is a sixth condition we cannot currently rank**, because our substrate resets episodes: a bad prediction costs an LLM a gradient step and costs an animal a fall. Nothing in `mjc/` prices safety, and a real exploration policy must.
- The LLM claims in §3 and §8 are *structural arguments about training setups*, not measurements on any model we have run.

## 12. The synthesis worth keeping

Two candidate "fires" are on the table and they do not compete:

- **The hard real-time deadline** is the *causal* story — no deadline → no need to commit → no simulator → no counterfactuals → no planning or self-model. A raw-data net never faces one, so it never invents the organ.
- **Heterogeneous graders** is the *functional* story — what having a second organ with a different objective actually bought, which is not what it was selected for.

**The deadline is why the fire was lit; the disagreement is what burns.** This is the exaptation reading one level up, and it is consistent with the belief tree's strongest cerebellar node: the codomain (body state vs cortical activation) is set by *afferent wiring*, not by a different computation — one algorithm, re-pointed. Motor control was not the harder task. It was the task with a deadline, which forced the invention of an organ that turned out to be general-purpose.

Stated in one line: **the cerebellum and midbrain are what you need to *produce* the signature; a cortex is what you need to *absorb* one.** We built a machine that absorbs, trained it on the richest signature that exists, and it is jagged in exactly the places where absorbing and producing come apart.

## Open questions

- Is "sighted where the other is blind" formalizable beyond the anecdote — e.g. as a condition on the two graders' error covariance, or on their level sets over the quantity you actually care about? If two graders' blind spots are *anti*-correlated by construction, that would be a design principle rather than a lucky property of biology.
- Four graders, not two. Does the payoff keep increasing with the *number* of distinct types, or is the dense/evaluative pair the whole of it and the hippocampus/cortex axis a different (memory) trade?
- Where does learning progress sit? It is a second-order functional of the dense grader and it is the closest thing we have to a bridge signal. Is LP best understood as *a third grader*, as *the channel* between the first two, or — §4b's reading — as **the expansion grader specifically**, its band-pass shape being a specification rather than an empirical curiosity?
- Does §4b's table have a **fourth cell**? Compression is dense-graded and expansion evaluative-graded; is there a useful notion of an *evaluatively-graded compression* (discarding a dimension because it stopped paying, not because it stopped predicting) — i.e. is forgetting the missing quadrant?
- §4b makes `R_act`/`R_comp`/`R_res` the natural readout for the whole frame, and that triple is already directly measurable. Does the §9 grader-disagreement cut have a representation-side readout — does detecting a blind grader show up as `R_res` refilling?
- Can grader blindness be detected **online**, or only in the post-hoc way we have always done it? [`online_value_loop/`](../experiments/mjc/online_value_loop/README.md)'s double obstruction is a warning that the answer may be "only slowly."
- Does the frame say anything about *how many* graders a training run should have, or only that one is too few?

[^private]: Not mirrored: this link points to a document in the private lab repo (the roadmap, the queue, an unrun spec, reading notes, or a conversation). See the top-level README for what is held back and why.
