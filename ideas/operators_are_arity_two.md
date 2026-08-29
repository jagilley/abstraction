# Operators are arity-2: theory of mind is planning in someone else's action space

**Status**: conceptual synthesis, no new experiment and no new measurement. Every anchor below is an
already-measured result in this repo. The constructive definition (§1), the identification of theory of
mind with off-distribution command queries (§3), the observed-vs-latent command split (§4), the
mirror-neuron-free motor reading (§6), and the program in §8 are new and unbuilt.
**Date**: 2026-08-28
**Prompt**: a conversation opening on a capability humans have and LLMs do not — *"Give a human secretary or
ghostwriter a few passages of your writing, and they can learn to emulate it pretty effectively, even more so
if you give them more feedback. LLMs, however, are famously terrible at writing… mode-locked into viewing the
world in a particular way"* — and asking whether that is a wedge into a region of capability space vanilla
models cannot reach by construction, whether it connects to how animals acquire motor skills, and whether it
bears on **what an operator is in the first place**.
**Builds on**: [operators_not_footprints](../beliefs/trees/operators_not_footprints.md) (this doc supplies the
root's missing constructive definition), [heterogeneous_graders.md](heterogeneous_graders.md) (§7 uses the
mirror-grader criterion), [physical_control_substrate.md](physical_control_substrate.md) (the arity thesis),
[self_model_needs_a_loop.md](self_model_needs_a_loop.md) (producer = consumer),
[breadth_as_grader_heterogeneity.md](breadth_as_grader_heterogeneity.md).
**Beliefs touched**: [operators_not_footprints](../beliefs/trees/operators_not_footprints.md) — §1 is a
candidate *definition* for the root, and §4 supplies the mechanism for its speculative inter-agent child;
[cerebellum_and_cognitive_architecture](../beliefs/trees/cerebellum_and_cognitive_architecture.md),
[self_prediction_and_self_knowledge](../beliefs/trees/self_prediction_and_self_knowledge.md).
**Key experiments**: [`reaching/`](../experiments/a2a_forward/reaching/README.md) (arity; the
veridical-but-useless control) · [`inverse_dynamics/`](../experiments/inverse_dynamics/README.md) ·
[`a2a/confabulation/`](../experiments/a2a_forward/confabulation/README.md) +
[`rhm/confabulation/`](../experiments/rhm/confabulation/README.md) (the observer ladder) ·
[`OOD_ROBUSTNESS`](../experiments/a2a_forward/OOD_ROBUSTNESS_README.md) ·
[`mjc/arity_torque/`](../experiments/mjc/arity_torque/README.md) ·
[`ballistic_depth/`](../experiments/one_layer_deeper/ballistic_depth/README.md) ·
[`rhm/minting/`](../experiments/rhm/minting/README.md) ·
[`mjc/meta_adapt/`](../experiments/mjc/meta_adapt/README.md).
**Attribution**: the originating observation (ghostwriters recover a style from few passages plus feedback;
LLMs do not; the field has ignored the domains where the LLM cognitive artifact is not easily usable), the
conjecture that this is a *structural* wedge rather than a data problem, the connection to motor skill
acquisition, the instruction to drop the mirror-neuron/mentalizing framing entirely, the social-intelligence
grounding, the prior and independently-arrived-at link between arity-2 and exogenous/endogenous modeling, and
the insistence that the real question is what an operator *is* — all Jasper's. The arity definition of
"operator", the identification of theory of mind with off-distribution command queries, the observed-vs-latent
split, `u`-transfers-and-`s`-doesn't, and the program's ordering came out of the exchange.

---

## One-liner

**An operator is a model whose argument list contains every exogenous driver; a footprint is the same model
with at least one driver marginalized out — and that marginalization is the *only* place distribution-boundness
enters.** For controlled systems this means operator = arity-2, `f(s,u)`. Everything follows: a counterfactual
is a query at an unobserved `u`, so **theory of mind is counterfactual competence about another agent and is
structurally unavailable to a model that has integrated `u` away**; self-modeling is easy and other-modeling is
ill-posed for exactly one reason (your own command is observed by efference copy, theirs is latent); and the
resolution for a latent command is not inversion but **forward search over `u`** — i.e. theory of mind is
planning in someone else's action space. Style emulation, imitation learning, and behavior cloning's
compounding error are then one problem viewed from three literatures, and the repo's substrates can test it
where `u` is exogenous and free.

## 1. What an operator is

[operators_not_footprints](../beliefs/trees/operators_not_footprints.md) states the distinction and gives an
*empirical* criterion for it — operators are weight-defined and survive distribution shift; footprints are
data-defined and collapse. It never says what makes something an operator. Here is a candidate:

> A **footprint** model learns `P(s' | s)`. An **operator** model learns `s' = f(s, u)`. `P(s' | s)` is the
> operator with the command marginalized out, and the marginal over `u` is a property of **whoever was
> driving**, not of the system. That is the only place distribution-boundness enters — not mysteriously,
> mechanically.

Stated carefully, so it does not overreach: an operator is a model whose arguments include every exogenous
driver. For an autonomous system `s' = f(s)` already qualifies. For a *controlled* system — which is every
interesting one — the driver is the command, and the distinction is arity.

**This collapses two of our results into one.** They have been filed as separate discoveries:

- [`OOD_ROBUSTNESS`](../experiments/a2a_forward/OOD_ROBUSTNESS_README.md): the forward model's robustness is
  distribution-invariant (Δloss ratio **0.43–0.51** on every corpus; tr(H) ratio 0.38–0.45), while the
  autoencoder's **collapses off-manifold** — on code its Δloss ratio rises to 0.93 and its Hessian trace ratio
  to **1.02**, i.e. curvature indistinguishable from a model trained with no injection at all.
- [`reaching/`](../experiments/a2a_forward/reaching/README.md) and
  [`arity_torque/`](../experiments/mjc/arity_torque/README.md): a command-blind arity-1 model cannot recover
  command-driven dynamics *at any capacity* — a **232-parameter** command-aware FM beats a **70,152-parameter**
  command-blind one; the smallest arity-2 FM beats the largest arity-1 FM; the no-`u` control collapses
  `cmd_rel_spread` **0.24 → 0.00** and sits at a **structural zero** on the command-conditional slice.

These are the same finding. **The autoencoder collapses off-distribution *because* it is arity-1** — it models
`P(acts)`, the input marginalized out, which is why the unconditional GLP needed ~3B parameters where a
state-conditioned FM needs ~1%. The parameter ratio is the price of the marginalization, denominated in
weights.

## 2. The entry point: style is a control problem

Language is a transition operator on a receiver's state — this is already the belief tree's language node
(speech-act theory, Wittgenstein, RSA). Take it literally:

| | |
|---|---|
| state `s` | the reader's induced state |
| command `u` | what the author is *doing* to it — concede, escalate, undercut, withhold, land the turn |
| **the text** | **the actuation, not the state** |

An LM models `P(token | tokens)`. It has **no `u` slot**, by construction, and you cannot supply one at
inference time because there is no port. So an LM is an arity-1 forward model of a controlled system — and the
arity results say such a model does not degrade gracefully, it **floors**.

Two things fall out that match the observed phenomenology exactly:

- **Style requests leak into content.** With no `u`, the only free variable is `s`. The model can change *what
  it says* but never *what it is doing*, so a style request is absorbed into topic. Ask for Hemingway and you
  get a man, alone, years passing. This is already measured and mislabeled in our own repo:
  `glp/`[^private] logs the steered output as *"nearly verbatim structural
  imitation"* of *The Old Man and the Sea*, and finds style-aware **prompt priming contributes ~70–85%** of the
  whole effect. That is not laziness; it is the only channel an arity-1 model has.
- **A style prompt is a constant `u`.** One global command for an entire piece, on a system whose command
  changes every sentence. No amount of prompt engineering fixes an arity problem, which is why the prompt term
  dominates and then stops.

## 3. Theory of mind is a query at an unobserved command

Theory of mind is not "having a model of another agent" — a footprint model of everyone is exactly what an LM
has. ToM is **counterfactual competence**: what would they do if they wanted something else, knew something
else, were somewhere else.

**A counterfactual is a query at a command you did not observe.** Therefore: *you cannot construct a
counterfactual about a variable you marginalized out.* Not "it is hard," not "it needs more data" — there is no
argument to intervene on. A footprint model can only interpolate among observed behaviors, which is the precise
technical content of "shallow emulation" and of "mode-locked."

The intra-agent version of this argument is **already written down** in
[`REACHING_INTERNAL`](../experiments/a2a_forward/reaching/REACHING_INTERNAL_README.md): *"The main forward pass
runs one real trajectory; it cannot answer 'what if I took action `a` instead' without taking it. The thing
that simulates alternatives off to the side is the separable FM (queryable without committing), doing something
the forward pass structurally can't."* §4 is that sentence pointed at another agent.

**Corollary — why LLMs pass ToM benchmarks and fail at ToM.** Sally-Anne and its descendants ask about
counterfactuals that are *written down*: textbook, discussed, in-distribution. That is retrieving someone
else's counterfactual, not constructing one — structurally identical to famous-author style transfer being
retrieval while ghostwriting-you is inference. The general form is worth stating on its own:

> **The field benchmarks the retrieval case of every capability whose construction case requires arity-2.**
> Style, theory of mind, planning, causal reasoning. Retrieval and construction return the same answer on any
> question that has already been asked, so the benchmarks are blind to the gap by construction.

## 4. The split: your command is observed, theirs is latent

This is the unification, and it re-reads two experiments as one object.

| | command | forward model | measured |
|---|---|---|---|
| **self**-model | **observed** (efference copy — you emitted it, it is free) | well-posed, cheap | cosine **0.972**, ~94% KL recovery, ~1% capacity |
| **other**-model | **latent** — must be recovered from `(s, s')` | inverse dynamics, ill-posed | R² **0.88 → 0.15** while forward stays pinned at **1.00** |

Self-modeling and other-modeling are the same operator problem, differing **only** in whether the command is
observed. That is the mechanism for
[operators_not_footprints](../beliefs/trees/operators_not_footprints.md)'s speculative inter-agent child (*"the
inter-agent and intra-agent instances of operator transmission"*), which until now asserted the identity
without saying what separates them. It is also why the a2a program kept hitting inference-in-disguise every
time it tried for self-knowledge without a command.

[`inverse_dynamics/`](../experiments/inverse_dynamics/README.md) supplies the failure signature, and all three
of its fingerprints are recognizable in style emulation: **asymmetry** (continuing your prose is forward and
well-posed; re-instantiating your voice on a new topic is inverse), **capacity hurts** (the inverse peaks at
width 128 then degrades, 0.30 → 0.28 → 0.25, while prediction effective rank balloons 7 → 59), and
**mode-averaging** to the fiber centroid (only ~0.13 beyond a fiber-mean baseline).

**And scale is measured not to help.** The confabulation nodes race a capacity-matched third-party observer
against a model's own self-report. On implementation targets the observer ladder is **flat in capacity** —
`O_input` reads 0.376 / 0.398 / 0.381 / 0.376 across the sweep — while on the input-determined WORLD control
the *same* ladder climbs 0.227 → 0.407 → 0.522. On language the first-person advantage is **+0.211 to +0.374**
at every capacity, and the `ENT` control runs the other way (an observer predicts M's output entropy at 0.961
where M self-reports 0.674, **−0.287**). The stated conclusion is the one this doc needs: **the third party is
access-limited, not resource-limited.**

A ghostwriter is a third-party observer recovering implementation from behavior. If the limit is access, the
only fix is access — and the sole access channel is the author correcting you. **Ghostwriting is therefore an
interactive problem, and provably not a scaling problem.**

## 5. What to do about a latent command: don't invert, plan

Direct inversion is ill-posed because forward is a function and inverse is a relation. The resolution is to
**hold a forward model of them and search over `u`** — find the command that would have produced what they did,
then run *forward* from it on new content. Amortized inverse dynamics via forward search.

> **Theory of mind is planning in someone else's action space.**

Not simulation-of-mind in the folk sense; literally the planner move the reaching arc already built, with the
codomain re-pointed. That is the same "one algorithm, re-pointed by afferent wiring" the cerebellar tree
asserts twice already.

**And the reaching arc predicts something counterintuitive about what a good other-model looks like.**
Veridicality ⊥ control-usefulness, established twice and in opposite directions:

- The value-shaped co-trained FM is the **best** planner substrate (+0.51) at the **worst** veridicality
  (0.21@n4), beating even a **perfect-simulator env-MPC** (+0.33).
- The arity-1 control `int_plan_state` is **veridical but useless** — Δ-cos **0.91–0.99** and it plans at the
  floor, because its degenerate operator collapsed to autonomous dynamics (`cmd_rel_spread` 0.02) with no
  command-conditionality to plan over.

The second is the sharper one for this doc: *a perfectly accurate arity-1 model of an agent is worthless for
acting as them.* Ported: **the best model of a person is not the most accurate one.** A good ghostwriter does
not predict your next word; they know what you are *going for* — a value-shaped, low-veridicality,
high-plannability model. Single substrate so far, but directly testable in the author setting.

## 6. The motor reading, with the mirror neurons removed

The correspondence problem — their arm is not your arm — is solvable exactly when you have the command slot and
unsolvable without it. **`s` does not transfer across bodies. `u` does.** So imitation is system identification
on another agent followed by re-planning in your own actuation space, and the demonstration is not the training
signal but **evidence for an inference**. (This is why coaching says *feel where the weight is* rather than
*move like this*: the coach is trying to hand you `u`, the only coordinate that survives the correspondence.)

No mentalizing, no mirror neurons, and it is stronger than either because it says what must be true rather than
naming a cell type.

**Three failures we have measured separately are one failure.** An arity-1 model rolled forward leaves the
manifold it was fit on, with nothing closing the loop:

- behavior cloning's compounding error;
- style drift over a long piece;
- [`ballistic_depth/`](../experiments/one_layer_deeper/ballistic_depth/README.md)'s *"private trajectory, not a
  closed operator"* — the rolled state nearly orthogonal to the encoder's representation of the same value
  (cos **0.19**) from step 1 **while decoding it perfectly**, taken to **0.99** by one label-free closure
  constraint. *(That node's horizon numbers are self-flagged as unconverged and are lower bounds; the closure
  measurement is the part cited here.)*

The fix in each case is the same shape: a constraint that the state you rolled into is one your own encoder
could have produced.

## 7. What this says about graders

Two consequences for [heterogeneous_graders](heterogeneous_graders.md), which the conversation reached from the
style side and which sharpen §8's mirror criterion:

- **LLM-as-judge is a mirror structurally, not contingently.** A footprint model maintains exactly one
  representation of its own computation, therefore exactly one space to grade in, therefore every grader built
  in output space inherits the same fiber structure. *Grader-type diversity is bounded by the number of
  representations of the computation you maintain, and a vanilla model maintains one.* This also suggests an
  answer to §10's open question about formalizing "sighted where the other is blind": `a_j = FM(a_i) + r` makes
  `FM(a_i)` *by construction* everything a self-theory could produce and `r` its complement — disjointness by
  decomposition rather than by luck.
- **Style is a density quantity, and verifiers are support oracles.**
  [`rhm/minting/`](../experiments/rhm/minting/README.md) found the exact rule table can delete out-of-support
  mass but cannot correct in-support density error. Every author writes legal English — same support, different
  location within it. So verifiable rewards, the field's one established non-mirror grader, are **structurally
  the wrong type** for this problem. That is a better answer to "why has nobody done this" than "the field
  ignored it."

Two further design constraints already priced elsewhere: feedback must be **directional, not scalar** (the
scalar emotion-injection result at 1/5 the directional forward signal, and `endogenous_teacher`'s dead
scalar-weighting null which cites it as having predicted the outcome); and the meta-gap opens under **conflict**
and collapses on scalar floors ([`meta_adapt/`](../experiments/mjc/meta_adapt/README.md): the 1-D damping floor
gives meta ≈ multitask, actuator-rotation conflict opens it monotonically to ~8× at Φ=π/2; and system-ID ⊥
adaptation-benefit — *knowing the parameter and using it are different achievements*).

## 8. The program

**Command-space generalization as the unifying axis.** One measurement shape, three substrates in which `u`
means three different things:

| substrate | `u` is | the held-out thing |
|---|---|---|
| [`mjc/`](../experiments/mjc/on_policy/README.md) | torque | a region of command space |
| [`rhm/` sculpting](../experiments/rhm/RHM_SCULPTING_README.md) | the repair move | an author's *policy* over the same action space |
| language | the pragmatic act | an act-in-context |

The RHM cut is the one that would demonstrate style and motor control are literally one problem on a substrate
with ground truth. The natural design: **two authors with different repair policies driving the same corrupted
leaves toward the same target root** — same meaning, same destination, different route. That keeps
acting ≠ planning (so it does not decay into the inference-in-disguise that `ACTIVE_RHM` found), installs the
per-rule non-uniformity RHM lacks by construction (the same injection `directed_sculpting` had to hand-build),
and lets style recovery be read **per hierarchy level** — turning "shallow emulation" from a complaint into a
curve. Note that a pure synonym-bias version of the same idea would be arity-1 and is the thing to avoid.

**Ordering matters, and it is the opposite of the tempting order.** Establish the law where `u` is exogenous
and free (mjc, RHM sculpting); port to language **last**. Not because language is unimportant but because of
§10. This also happens to point the same way as the resource constraint.

**Readout note.** Do not use participation ratio. `endo_expansion` found it inverts — a content-free random
target produced the largest PR rise while sitting at the no-loop floor on every functional readout, because PR
reads idle capacity being filled. Use a functional readout against the ground-truth policy.

## 9. What the frame implies

- **Arity-1 floors rather than degrades on held-out command, at every capacity, on every substrate.** This is
  the through-line, and it is already true in two of the three.
- The **forward/inverse asymmetry should be visible in real models**: continuing an author's prose should be far
  better than re-instantiating their voice on a novel topic, and the gap should be a *capacity-insensitive*
  one.
- **Scaling an other-model does not close the gap**, because the limit is access, not resources (the flat
  observer ladder against its own climbing control).
- **Emulation quality should track conflict**: models should emulate a person well where that person is
  unremarkable and revert exactly where they are distinctive, since the meta-gap only opens under conflicting
  demands on the same tokens.
- **Fine-tuning is predicted to fail twice, for unrelated reasons** — SFT fits `P(text)`, the manifold arm,
  which the OOD result says collapses off-distribution (and "off-distribution" here *is* "a topic they never
  wrote about"); and pooled monolithic memory is a measured liability under conflict. Neither is a
  data-quantity problem, which is why more data does not help.

## 10. The risk that would make this unfalsifiable

`u` may not exist as a well-defined object in language. The scope worry is real but secondary; the sharp
version is: **if `u` is only definable post hoc, arity-2 is unfalsifiable.** You can always label a sentence
with an intent after the fact, at which point `u` is a free parameter that absorbs any residual and the frame
becomes decoration.

The discipline that avoids it: **`u` must be specified before the outcome and must predict held-out behavior.**
Free in mjc (you set the torque), free in RHM sculpting (you choose the move), *not* free in language. This is
the genuine scientific risk of the direction — more than compute, more than grader fidelity — and it is why §8
orders the substrates the way it does.

## 11. What we have not shown, and should not claim

- **Nothing here is new evidence.** Every number is re-read from an existing node; the contribution is the
  definition, the identifications, and the program.
- **The arity definition of "operator" is an argument, not a measurement.** It is *consistent* with the OOD and
  arity results and it unifies them, which is suggestive and not a test. A direct test would have to exhibit a
  case where the two criteria (survives shift / has the driver in its arguments) come apart.
- **The ToM identification is a definition plus a structural argument.** We have measured neither ToM nor style
  in any model.
- **Veridicality ⊥ control-usefulness is single-substrate** (reaching, two directions), and the
  "best model of a person is not the most accurate one" reading is an extrapolation from it.
- The `mind → text` non-injectivity claim is inherited from the belief tree and is not measured on text; the
  grokking-MLP collision structure is the only place we have measured it, and modular addition is the most
  extreme available case.
- Nothing here prices the author-feedback channel. Endogenous graders recovered 19–34% of an external teacher
  with the diagnosis *fidelity, not endogeneity*; that number, not the architecture, likely decides whether the
  language port works.

## Open questions

- Is the arity definition the right *general* one, or only right for controlled systems? Stated generally
  ("every exogenous driver in the argument list") it is close to a definition of a causal model — which raises
  whether "operators, not footprints" has been a causal-inference claim in disguise the whole time, and if so
  what our substrates buy that the causal literature does not already have.
- If self-modeling and other-modeling differ only in whether `u` is observed, **is there an efference-copy
  analog for another agent?** Dialogue is the obvious candidate — when someone answers you, you know what you
  did to them, so `u` is partially observed. That would predict conversational style is easier to acquire than
  monologue style, which is cheap to check and would also be the natural first language substrate.
- Does the command slot have to be *supplied*, or can it be **discovered**? Everything above assumes `u` is
  given at training time. A learner that had to posit its own command basis for another agent is the
  interesting version and nothing here says whether that is possible.
- Where does this leave [contra-long-horizon-benchmarks](contra-long-horizon-benchmarks.md)'s rotation framing?
  "Find the basis in which the answer is legible" and "find the command basis in which the other agent's
  behavior is simple" may be the same operation, in which case ToM is a special case of the rotation claim
  rather than a sibling of it.

[^private]: Not mirrored: this link points to a document in the private lab repo (the roadmap, the queue, an unrun spec, reading notes, or a conversation). See the top-level README for what is held back and why.
