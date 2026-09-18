# Calibration and violation are one object read twice: why "the logits are Bayesian" and "emotion is norm violation" are isomorphic, and where they part

**Status**: **§12 (2026-09-18)** restates the whole panel as one table, prediction as the projection whose utility is the belief's own log-probability, the two blindnesses, and where the axis comes from, with every claim tagged measured, restated or inference, and a same-day note in §12.4 on why a candidate has to be fired to be priced and replay is forced by the meter; **§11 (2026-09-17, after the six value-side rounds of [`orbitofrontal/`](../experiments/rhm/logit_reading/orbitofrontal/README.md))** corrects §5's "organ" to "projection" and the one-liner's placement of the norm, and adds the public/private split; conceptual synthesis (2026-09-16); §5's two organs each measured once on 2026-09-17 (dated note in §5); **§10 (2026-09-17)** reads the whole arc after #120[^private] with its consumers turned on — a *tentative* read, marked as such throughout, with its first two arms measured the same day (dated note at the top of §10). Nothing new run for the synthesis itself. Every number below is re-read from
the three `logit_reading` nodes and their neighbours; what is new is the term-for-term mapping
between the two intuitions, the three conditions under which it holds, the point at which it stops,
and the reading of Yu & Dayan's *unexpected uncertainty* through the `CE − H(q)` identity.
**Date**: 2026-09-16
**Prompt**: a conversation with Jasper reading the three logit-reading PRs as a whole
([#114](../experiments/rhm/logit_reading/README.md) · [#115 `altitude/`](../experiments/rhm/logit_reading/altitude/README.md)
· [#119 `coeruleus/`](../experiments/rhm/logit_reading/coeruleus/README.md)). Two intuitions had
been running side by side without being connected: (1) *under the right circumstances, a model's
logits are a rough expression of Bayesian calibration*; (2) *emotions are roughly tied to the
violation of expectations or learned norms, which in some cases is as simple as a Bayesian
violation*. The question was how the two connect, and why they might be isomorphic under certain
conditions. The neuroscience side of (2) is the conversation that prompted the whole line,
`conversations/Claude-Neural correlates of model confidence and prediction errors-20260915-1302.md`[^private].
**Attribution**: the two intuitions, the question, and the observation that the directional-not-scalar
pattern meshes with the neuroscience are Jasper's; the "one object read twice" framing, the mapping
table, the three conditions, the identity reading of unexpected uncertainty and the complementarity
corollary are the orchestrator's (this session), offered in discussion and agreed as worth persisting.
**Builds on**: [heterogeneous_graders.md](heterogeneous_graders.md) §8 (the mirror-grader
criterion, which §7 below gives a quantitative form) · [revision_not_surprisal.md](revision_not_surprisal.md)
(revision as the object, not surprisal) · [cerebellar_abstraction_ratchet.md](cerebellar_abstraction_ratchet.md)
and [activation_to_activation_forward.md](activation_to_activation_forward.md) §Structural analysis
(the expected/unexpected uncertainty distinction, which §6 sharpens).
**Beliefs touched**: [self_prediction_and_self_knowledge](../beliefs/trees/self_prediction_and_self_knowledge.md)
— the *directional, not scalar* node (a fifth instance) and the *re-derivation boundary* node (the
`CE − H(q)` identity is a sharper statement of "an outsider reads the model's own entropy better than
the model's self-report") · [cerebellum_and_cognitive_architecture](../beliefs/trees/cerebellum_and_cognitive_architecture.md)
(the Yu & Dayan line).
**Experiment anchors**: [`rhm/logit_reading/`](../experiments/rhm/logit_reading/README.md) Parts 1–2 ·
[`altitude/`](../experiments/rhm/logit_reading/altitude/README.md) Q1, Q4b, Q5 ·
[`coeruleus/`](../experiments/rhm/logit_reading/coeruleus/README.md) Q1–Q3 ·
[`rhm/conditional_revision/`](../experiments/rhm/conditional_revision/README.md) ·
[`rhm/endogenous_teacher/`](../experiments/rhm/endogenous_teacher/README.md) Gate 0 ·
[`EMOTION_INJECTION`](../experiments/a2a_forward/EMOTION_INJECTION_README.md).

---

## One-liner

A learned predictive distribution has a **shape before the outcome** and an **entry at the outcome**,
and what the outcome does to the shape is the **revision**. Intuition 1 is about the shape; intuition 2
is about the entry and the revision. In a Bayesian all three are functions of one posterior, so the
two intuitions are two readings of one object, and the "learned norm" of the emotion story is literally
the world-model whose posterior the logits are. The isomorphism holds at the level of *what an emotion
is a function of*, under three conditions the RHM substrate let us check, and it stops exactly where
an emotion adds a consumer and a valuation, which the predictive does not contain. One consequence
falls out of the `CE − H(q)` identity: *unexpected uncertainty* in Yu & Dayan's sense is zero-mean on
the training distribution by construction, so a noradrenaline-shaped readout is a distribution-shift
detector whether or not anyone designed it to be, and the circumstances under which intuition 1 holds
exactly are the circumstances under which the scalar form of intuition 2 must be silent.

> **Corrected 2026-09-17 (§11).** "The learned norm of the emotion story is literally the world-model whose posterior
> the logits are" places the norm on the likelihood side. The value-side rounds put it where Xiang's data does: the
> norm is the **world model read through a projection whose weights the goal's outcome history sets** — input from the
> world model, weights from the history — and the isomorphism of §2 holds on that projection too (a level before, an
> outcome surprise at, a revision after). The level is public, recoverable from the published belief; the revision is
> private, in directions of the representation the output does not expose. §5's "two organs the predictive does not
> contain" is one projection and its consumer; what the predictive does not contain is the axis.

> **Restated 2026-09-18 (§12).** Prediction is itself a projection, the one whose utility is the belief's own
> log-probability: its level is `−H(q)`, its error is the excess surprise, and so the two intuitions are two rows of
> one table, differing only in whether the utility is supplied by the world model or from outside it.

---

## 1. The two intuitions, and why each had to be sharpened before they could meet

Stated loosely, neither intuition says enough to connect to the other.

**Intuition 1, loose form**: the logits reflect the model's calibration — its stated confidence
tracks how often it is right. On the RHM substrate this is true (ECE 0.003, entropy predicts the
model's own expected loss to 0.004 nats, `R²` 0.99), and [`altitude/` Q1](../experiments/rhm/logit_reading/altitude/README.md#q1--the-identity-the-temperature-and-a-continuous-altitude)
showed it is nearly empty: `CE − H(q) = ⟨q − p, z⟩` is the derivative of the loss along the
logit-scaling direction, zero once trained in every context class the model can scale separately
(`α* = 1.00–1.01` from 2k steps, per-class gaps within ±0.015, checked pointwise to 1e−07). A
free temperature buys self-calibration for nothing. It says nothing about what the model knows.

**Intuition 1, sharpened**: *the logits are the posterior predictive under the world-model the
learner has absorbed, at the depth it has absorbed it.* This is the part with content. Against a
family of exact Bayesian observers — observer `k` knows the bottom `k` of six grammar levels and
treats everything above as i.i.d. — the logits are closest to a coarse observer whose continuous
altitude `κ*` climbs 0 → 4.9 over training, and the best family member is 9× closer to the logits
than the truth is at step 250 and 1.2× closer at 64k
([Part 1](../experiments/rhm/logit_reading/README.md#part-1--the-logits-are-a-near-exact-posterior-of-a-coarser-observer),
[Q1](../experiments/rhm/logit_reading/altitude/README.md#q1--the-identity-the-temperature-and-a-continuous-altitude)).
What training changes is *which Bayesian the logits are*.

**Intuition 2, loose form**: an emotion is triggered by a violation of learned expectations, weighted
by how confident the expectation was. The natural scalar for this — surprise beyond my own expected
surprise, `s − H(q)` — is a **null** on this substrate: once the token's surprisal is matched, the
model's confidence adds nothing at all (AUC 0.50–0.51 at every checkpoint,
[Part 2b](../experiments/rhm/logit_reading/README.md#2b-against-controls-from-other-windows-the-separable-signal-is-tonic-not-phasic)).

**Intuition 2, sharpened**: *a violation is an outcome the absorbed world-model excludes, and its
signature is a shift of the posterior over latent structure, read in belief coordinates.* This is
what survives. A same-prefix design that cancels everything about the context exposes a real response
at the token, readable from the post-token forecast (0.85) and the residual stream (0.84) against
floors of 0.51 at random init; it is graded true-likelihood, not categorical legality (0.67–0.74 on
two *legal* tokens differing 2× in true probability); and it is directional. This is the fifth
instance of the repo's [directional-not-scalar pattern](../beliefs/trees/self_prediction_and_self_knowledge.md#the-absorbed-signal-is-encoded-and-used-directionally-what-kind-of-computation-was-surprising-not-as-scalar-magnitude-how-surprising),
after [`EMOTION_INJECTION`](../experiments/a2a_forward/EMOTION_INJECTION_README.md),
[`endogenous_teacher`](../experiments/rhm/endogenous_teacher/README.md) (Gate 0: the model's own
residual and the text's surprisal share 11% of their variance and are anti-localised across levels),
[`conditional_revision`](../experiments/rhm/conditional_revision/README.md) (revision as a KL in
belief coordinates separates where the residual norm does not: 0.606 vs 0.690 at d2, 0.005 vs 0.150
at d1) and the controlled-retrain probes.

Jasper's reading, recorded here because it is the reason the mapping is worth trusting: the
directional finding is the one that meshes with the neuroscience. Cortical prediction errors are
population-coded per feature in superficial layers, not one number; the scalar prediction error
is the *reward* channel, which is a different organ (§5).

## 2. One object, three readings

Let `q` be the model's predictive at a position, `x` the token that arrives, `q'` the predictive
after it. Everything either intuition talks about is one of three functionals of the same object:

| reading | quantity | when | which intuition |
|---|---|---|---|
| **shape** | `H(q)`, the peak, the precision | before the outcome | 1 — "confidence" |
| **entry** | `−log q(x)` | at the outcome | 2 — "surprise", the value system's `−log p(outcome)` |
| **revision** | `KL(q' ‖ q)`, and *which direction* it moved | after the outcome | 2 — "the episode tracks the mismatch" |

In a coherent Bayesian these are not independent. The shape bounds the expected entry
(`E[−log q(x)] = H(q)` when `x ~ q`), the entry drives the revision, and the revision is confined
to the directions the generative model's latent structure allows. So a claim about one constrains
the others, and this is the whole reason the two intuitions can be put in correspondence rather
than merely juxtaposed. The originating conversation put it as "the value system is already
reading a function of the logits; it just reads the entry for what actually happened", and the
Ma–Beck–Latham–Pouget reading of a tuned population as a log-likelihood is the biological version
of "the distribution is in the code". This doc adds the third row: the emotion story's persistence
and its directionality live in the revision, and the revision is what the RHM instruments read.

## 3. The mapping, term for term

Every row is a claim the emotion story makes, its Bayesian translation, and the measurement on the
RHM substrate that bears on it. The first five rows are the isomorphism; the last two are where it
stops (§5).

| emotion story | Bayesian story | measured on `v16 s2 L6 m4` | where |
|---|---|---|---|
| **a learned norm** — "a running estimate of how people should treat you" | the predictive under the absorbed generative model, observer `κ*` | `κ*` climbs 0 → 4.9 one rung at a time; the family is 9× closer to the logits than the truth is early, 1.2× at convergence | [Part 1](../experiments/rhm/logit_reading/README.md#part-1--the-logits-are-a-near-exact-posterior-of-a-coarser-observer), [Q1](../experiments/rhm/logit_reading/altitude/README.md#q1--the-identity-the-temperature-and-a-continuous-altitude) |
| **you can only be offended by rules you have acquired** | events at levels ≤ `κ*` have near-zero mass under `p_κ*`; events above it are ordinary | per-level violation detection switches on in order with `κ*` (at 64k: 1.00 / 1.00 / 0.99 / 0.92 / 0.77 for `k* = 1…5`; at step 250: 1.00 / 0.77 / 0.63 / 0.59 / 0.57), with no separate mechanism | [Part 2a](../experiments/rhm/logit_reading/README.md#2a-violation-sensitivity-is-scoped-to-the-levels-the-model-has-learned) |
| **weighted by confidence** | the precision of the predictive | as a scalar, `s − H(q)`, a null at 0.50–0.51; the weighting exists only directionally | [Part 2b](../experiments/rhm/logit_reading/README.md#2b-against-controls-from-other-windows-the-separable-signal-is-tonic-not-phasic) |
| **the episode tracks the mismatch** | the revision `KL(q' ‖ q)` in belief coordinates, not the entry | revision separates from surprisal wherever the model has a belief; post-token forecast 0.85 and state 0.84 vs floors 0.51; graded 0.67–0.74 | [`conditional_revision`](../experiments/rhm/conditional_revision/README.md), [Part 2c–d](../experiments/rhm/logit_reading/README.md#2c-with-the-context-held-identical-there-is-a-response-to-the-token) |
| **the feeling persists** | the posterior over latent structure stays shifted | in the state (readable at 0.75 eight tokens on, confounded by the edited stream), not in the output (entropy moves ≈ 0 after the next position); the output falls to the observer at `k* − 1` | [Part 2e](../experiments/rhm/logit_reading/README.md#2e-the-entropy-response-runs-the-wrong-way-for-a-violation), [Q4b](../experiments/rhm/logit_reading/altitude/README.md#q4b--entropy-direction-the-two-explanations-come-apart-and-what-the-forecast-moves-toward) |
| **gain and learning rate respond** | precision weighting; neuromodulation | a consumer of the state pays only off-distribution: 0% of the prize on the trained world, 49% on an unseen one | [`coeruleus/` Q3](../experiments/rhm/logit_reading/coeruleus/README.md#q3--the-gain-loop) |
| **it is good or bad for me** | not in the predictive at all | untested; nothing costs the model anything | — |

## 4. The three conditions under which the first five rows hold

Each was checkable because the substrate's truth is computable. Each is also the point at which the
isomorphism could fail on a different substrate.

**C1. The predictive is a posterior under *some* coherent generative model.** Otherwise "norm" has no
referent: there is no world-model whose support the violation falls outside. On RHM this holds
approximately, and the approximation *thins as the model converges*: the fitted observer accounts
for 70–89% of the model's residual while `κ* ≤ 2.9` and 32–40% from `κ* ≥ 4.4`, with the residual's
level-shape a coarse observer's throughout
([Q2 B](../experiments/rhm/logit_reading/altitude/README.md#q2--the-frontier-from-the-logits-and-where-the-residual-lives)).
So the mapping is tightest for the undertrained learner and loosens as it becomes its own thing.
The residual-share instrument is the tool that says how far a given model is from C1.

**C2. The norm is learned, not given, so the scope of offense tracks acquisition.** This is the
"anger is informed by a running estimate" clause, and on RHM it is mechanical: if `q ≈ p_κ*`,
violations at levels ≤ `κ*` get near-zero mass and violations above it do not, so detection
inherits its scope from the climb with no organ of its own (Part 2a). The emotion story's most
distinctive structural claim is, on this substrate, a corollary of C1 plus training.

**C3. The violation is read directionally.** The scalar form of the weighting is a null (row 3); the
entry alone, `−log q(x)`, is matched away by design in every phasic readout and what remains is the
revision (row 4). If a substrate only exposed a scalar surprise, rows 3–5 would have nothing to
attach to. This is the row that meshes with the per-feature error populations of the predictive-
coding microcircuit, and it is the fifth time the repo has found it.

Under C1–C3, "a Bayesian violation of a learned norm" and "a low-probability event under the
posterior predictive of the learner's absorbed generative model, whose revision is read in belief
coordinates" are the same sentence. That is the isomorphism.

## 5. Where it stops: the consumer and the valuation

A calibrated predictive is a *property of a distribution*. An emotion is a *consumer of the
mismatch plus a valuation of it*, and neither is contained in the predictive.

- **The consumer is a separate organ, and its value is conditional.**
  [`coeruleus/`](../experiments/rhm/logit_reading/coeruleus/README.md) built the smallest one —
  a label-free head on the frozen state, predicting the model's own excess surprise over the next
  eight positions, allowed to set the temperature and to gate the per-position learning rate. On
  the distribution the model was trained on it is a null (below its random-trunk floor; the head-
  gated plasticity arm equals its shuffle, the `endogenous_teacher` null replicating with oracle
  arms that do separate). On a world the model never saw it detects the model's failures at 0.83
  and recovers half the loss. Row 6 of the table is *not* a property of the logits; it is what a
  second organ can do with them, and only where the world has moved.
- **The valuation is a third thing.** The originating conversation's picture has two prediction
  errors, not one: the structural one ("the world did not match my model", the whole of rows 1–5)
  and the reward one ("the outcome was better or worse than expected for my goals", the
  dopaminergic channel). An emotion sits at their junction — being treated poorly is *both* a
  violation of a learned norm *and* a valued outcome. Everything in the logit-reading arc is the
  first error. The model's only objective is next-token loss, so a violation is a position with
  more loss and nothing else; no goal is thwarted, so no valence can attach; and the "which world
  am I in" organ is valence-free by construction, responding the same whether the change is good
  or bad for the learner. The repo's outcome-trained readouts
  ([`voicing/overtone`](../experiments/rhm/practice/voicing/overtone/README.md)'s critic;
  [`directed_sculpting/full_loop`](../experiments/rhm/directed_sculpting/full_loop/README.md)'s
  grader-type result) are the nearest built machinery, and none of it has been pointed at a norm
  violation.

So the correct scope of the isomorphism is: **the two intuitions agree about the argument to an
emotion, not about the emotion.** What an emotion is a function of is the same object calibration
is a property of. The function itself — consume, then value — is two organs the predictive does
not contain.

> **Measured 2026-09-17, one round each** — [`basalis/`](../experiments/rhm/logit_reading/basalis/README.md)
> (the other consumer) and [`striatum/`](../experiments/rhm/logit_reading/striatum/README.md) (the
> valuation). The consumer: hold-and-discount is buildable exactly in the model's own terms and is
> worth most of the glitch-world prize a gain knob could not touch, and nothing on edits wider than one
> leaf, so "glitch" and "edit" turn out to be the ends of one axis, the width of the damaged span, with
> reinterpretation the right policy from about two levels up — the reading §5's first bullet leaned on
> now has its arm. The arbiter between the two policies is buildable in principle and worth little: the
> worlds are the same event at the token and diverge in the continuation, the state's early separation
> of them is the edited preamble (the tonic stimulus property again), and what the state knows about the
> decision itself is a tenth of the variance at the flag and nothing after. The valuation: given a goal
> that is not prediction (a fixed actor's structural query, right or wrong per window), a violation costs
> something and the cost inherits the model's altitude; a critic trained on outcomes alone reads the
> violation's **cost** and not its **structure** — a null on the consequence label the state carries,
> indifferent to legality at matched consequence — which is row 7's signature in its most literal form.
> Two sharpenings of this section. Once learned, the valuation *contains* the consumer's signal rather
> than sitting beside it: conditioned on the critic's value level, the excess-surprise readout carries
> nothing, and the clean-only critic does not subsume it, so that containment is what exposure to damage
> bought. And the directional-not-scalar pattern (C3) does not transfer to the value side on a frozen
> trunk with a linear critic: the state does not say *which* query was damaged, only how much.
>
> **Reframed 2026-09-17** by [`striatum/junction/`](../experiments/rhm/logit_reading/striatum/junction/README.md)
> and [`striatum/norm/`](../experiments/rhm/logit_reading/striatum/norm/README.md). Row 7's norm was in the wrong
> place. The valuation ignores the structural norm whatever the diet's violation–cost correlation, and has a norm
> of its own — the critic's learned expectation over outcomes, which calibrates to the world it was fed (a shift at
> random init, a rescaling once trained) and against which the outcome surprise reproduces Xiang's
> same-offer-different-history effect. The learned outcome norm, not the grammar, is the operative version of the
> anger intuition: the two prediction errors of the originating conversation are two norms in two currencies,
> each read against its own learned expectation, and the junction between them is not learned in either reader
> on this substrate.

> **Corrected 2026-09-17 (§11)** after six more rounds. "Two norms in two currencies, neither reading the other's" was
> too strong: the value critic reads the world model's own state, so its norm is one posterior read through the goal,
> and "neither reads the other's" meant only that the projection discards the part of the world model's error that
> carries no information about outcomes. Where legality does carry such information (a hidden regime that predicts
> cost) the same reader picks it up, priced by its prior belief; where structure is the cause of cost (the practice
> learner's judge) it reads structure. Given a consumer, the level gates and the revision does not; the outcome surprise
> is the teaching signal. That is the actor-critic split of the originating conversation, not a second model.

## 6. The identity reading of unexpected uncertainty, and what "the right circumstances" means

This is the one consequence that falls out of the arc rather than being read into it.

Yu & Dayan distinguish **expected uncertainty** (acetylcholine; known unknowns; trust the data over
the prior but do not restructure) from **unexpected uncertainty** (noradrenaline; the model itself
is wrong; reset and explore). The repo has carried this distinction qualitatively since
[cerebellar_abstraction_ratchet.md](cerebellar_abstraction_ratchet.md), and
[activation_to_activation_forward.md](activation_to_activation_forward.md) already proposed that the
signature of unexpected uncertainty is that the errors are *structured*. The logit-reading arc makes
both halves exact:

- **Expected uncertainty is `H(q)`**, the uncertainty the model states.
- **Unexpected uncertainty is `−log q(x) − H(q)`**, realised loss beyond stated uncertainty — which
  is exactly the excess-surprise target `coeruleus/` chose, summed over a horizon.
- **The identity says the second is zero-mean on the training distribution by construction**: its
  expectation is `⟨∂CE/∂z, z⟩`, the loss's derivative along the temperature direction, which any
  softmax with a free logit scale drives to zero in every class it can scale. And
  [`coeruleus/` Q2](../experiments/rhm/logit_reading/coeruleus/README.md#q2--the-self-supervised-readout)
  found it is not merely zero-mean there but *unpredictable from the state* (the head sits below its
  random-trunk floor). Off-distribution it is neither: the clean-trained model dropped onto a
  corrupted stream leaves 1.68 nats of excess at the event and the same head reads its failures at
  0.83.

> **A noradrenaline-shaped readout is a distribution-shift detector by construction.** Not because
> anyone designed it that way, but because training to convergence with a free logit scale zeroes
> the mean of "surprise beyond what I said" on the world that did the training. What survives
> on-distribution is the *structured* part — the directional revision — which is the half
> [activation_to_activation_forward.md](activation_to_activation_forward.md) guessed at.

The same identity says what "under the right circumstances" means in intuition 1. The circumstance
is **being on the training distribution**. The ε-trained model reads `CE − H(q) = −0.002` against
the distribution it was trained on and `−0.061` against a neighbouring one
([Q5](../experiments/rhm/logit_reading/altitude/README.md#q5--a-noise-trained-trajectory)); the
clean-trained model on a corrupted stream is 1.68 nats off at the event. Calibration in the
intuition-1 sense is a fact about the model *and the world it was fitted to*, jointly.

Put the two together and the intuitions turn out to be **complementary rather than redundant**:

> Precisely where intuition 1 holds exactly — on-distribution — the *scalar* form of intuition 2 is
> silent by identity, and only the *directional* form survives. Precisely where intuition 1 fails —
> off-distribution — the scalar form of intuition 2 comes alive, and that is the case that pays.

Which is the reading agreed in the `coeruleus/` discussion by another route: the organ is not for
the world you already know, and a biological learner never stops seeing a world it does not.

## 7. Consequence for "judging abstractions with the learner itself"

The `altitude/` round was prompted by the thought that the logits-as-coarse-observer correspondence
might be the piece a learner needs to judge new representations with its current ones. Read through
this doc, the record's answer is sharp, and it is §6 applied to the judge:

- **What the learner has, it can read.** Its constituent boundaries are recoverable from its entropy
  profile alone at the observer's own ceiling through level 3, and the altitude fit says which level
  is being absorbed ([Q2 A](../experiments/rhm/logit_reading/altitude/README.md#q2--the-frontier-from-the-logits-and-where-the-residual-lives)).
- **What it lacks, it prices but does not verify.** Used as a null against candidate next-level
  units, the model's predictive peaks one rung above its altitude and cancels once the level is
  absorbed, and it never beats a model-free count statistic (0.72–0.81 vs 0.92–0.96,
  [Q3](../experiments/rhm/logit_reading/altitude/README.md#q3--the-models-predictive-as-a-null-for-scoring-candidate-units)).
  It reports *which level is still news to me* — the location of the frontier, a weak one-level-up
  currency — not whether a specific candidate is true.
- **On-distribution, every self-readout in the arc was an identity, a stimulus property, or a null.**
  `CE − H(q)` (identity); the tonic signal (belongs to the stimulus, Q4a); the excess-surprise head
  and the head-gated plasticity (nulls). The self-readout became sighted only where the world
  differed from training.

This is [heterogeneous_graders.md](heterogeneous_graders.md) §8's mirror-grader criterion in a
quantitative form: a judge built from the learner's own predictive is blind where the learner is
blind, *and the identity says where that is* — everywhere the training distribution reaches. The
learner-as-judge is sighted about **change**, not about **truth**. In the typed-signals language of
the roadmap[^private] §2.1, the type of the learner's own excess surprise is "which world am
I in", not "is this unit real".

> **Sharpened 2026-09-17 (§10.1).** The second bullet undersold the record. The one-level-up currency *is* readable by
> the learner itself, on both sides of the repo, and the frontier read above is the logit-side twin of the practice arc's
> free next-level-yield gauge. What the learner cannot do from its own predictive is verify a specific candidate; what it
> can do is locate the rung and price it.

## 8. What this does not claim

- **The mapping is at the level of the argument, not the emotion.** Rows 6–7 are outside it, and
  the arc has built one consumer (noradrenergic, valence-free) and no valuation.
- **One grammar, one architecture.** C1 is the condition that thins with training here and could
  fail outright elsewhere; the residual-share instrument is how to check it on a new substrate. On
  natural language the "norm" is not a grammar with computable support, and nothing here says what
  the observer family would even be.
- **The `k* − 1` fallback and its reinterpretation** (the output reinterprets rather than doubts;
  right for structural edits, wrong for glitches; nothing arbitrates) is supported by the
  persistence tables and not tested by an arm. It bears on row 5 but is not load-bearing for the
  mapping.
- **The neuroscience is cited through the originating conversation**, not read here against primary
  sources. The mapping's biological rows (population code as log-likelihood, ACh/NE, the two
  prediction errors) are anchors for the *shape* of the correspondence, in the sense
  [practice_manufactures_its_own_credit.md](practice_manufactures_its_own_credit.md) uses its piano
  anchors.
- **Nothing here is about a model that stalls or collapses.** Every internal measure is relative to
  the training distribution, so a coherent Bayesian about the wrong world would read as healthy
  under C1 — the collapsed-distribution question `altitude/` flags is open.
- **§10 is a read, not a measurement.** Its six extrapolations (§10.4) go beyond the record by design and cite it so the
  distance is visible; nothing in §1–§7 rests on them.

## 9. Where the built machinery could bear on it

Listed as what each would measure, not what any outcome would mean.

- **The noise-dose sweep** (`altitude/train_noisy.py` at higher ε): whether the on-distribution
  scalar stays a null as the model's own world carries more glitches, i.e. how C1's "on-distribution"
  boundary moves.
- **The residual-share instrument on the [`reread/lm`](../experiments/rhm/practice/reread/lm/README.md)
  frozen checkpoints**: C1 on a model whose predictive competence degrades while its
  probe-readable structure holds (that node's finding 5), the case §8's last bullet names.
- **The hold-and-discount consumer** (`coeruleus/` §What this does not establish): a mixing weight
  between the forecast and the forecast with the suspicious token masked, gated by the excess head —
  the first component that could arbitrate between the two worlds row 5 describes.
- **An outcome-trained readout on the same state**: the dopaminergic consumer beside the
  noradrenergic one, which needs a goal beyond prediction so that norm violations can be predictive
  of outcomes. This is row 7 and is the only way the word "emotion" earns its second half.

---

## 10. 2026-09-17 — the family is a property of the problem, and what the panel looks like with its consumers on (a tentative read)

**Status**: a read, not a measurement, written after the discussion of
#120[^private] — the two organs of §5 measured once each — and
**pending further evidence**. Nothing new run; every number is re-read from the nodes, and where a sentence
below goes beyond the record it says so. The experiments that would bear on it are being discussed
separately and are deliberately not pre-registered here.
**Attribution**: the push to read between the lines toward how humans plausibly *use* this structure rather
than how the arc used it, the observation that the Bayesian correspondence is too on-the-nose to be a
coincidence and is meta-flavoured in exactly the roadmap's sense, and the correction in §10.1 are Jasper's;
the "property of the problem" argument, the identity-as-two-neuromodulators reading, the instrument-panel
table, the six reads of §10.4 and the "judged in consumption" line are the orchestrator's (this session),
offered in discussion and agreed as worth persisting.

> **Measured 2026-09-17, one round each** — [`frontier/`](../experiments/rhm/logit_reading/frontier/README.md)
> (the structural side, §10.2–§10.3) and
> [`striatum/junction/`](../experiments/rhm/logit_reading/striatum/junction/README.md) (reads 4 and 5). Three
> corrections and one confirmation to carry while reading below. (i) §10.2's second bullet holds and holds per
> level: the excess is ≈ 0 at every level on-distribution, and stays ≈ 0 on a walled learner's own corpus while
> its held-out excess is nats off. (ii) Its "the frontier is a derivative" clause is wrong in the position-level
> coordinate: the per-level loss trend's argmax is pinned at ℓ = 2 by the residual's fixed level-shape and never
> tracks `κ*`; what tracks the altitude from the inside is the entropy profile's *period*, model-free, at the
> observer's own ceiling through `k = 3`, whose induced level field replaces the oracle's labels below ℓ = 4 — the
> frontier is read from the *shape* of the predictive, and the candidate coordinate (Q3) is the derivative that
> still works. §10.5's wall is then not ambiguous from the inside: the period goes backwards at a wall, on the
> learner's own corpus, where a renewable learner's never drops; the excess and the memorisation gap turn a
> checkpoint earlier but need a held-out venue. (iii) Read 5 does not get support: the outcome reader's
> indifference to legality at matched consequence survives training diets whose violation–cost association runs
> from −1 to +1 (span ≤ 0.02 against 0.05–0.11 of oracle headroom), while a positive control shows the diet does
> reshape the critic along the edit's width and the structural readouts read legality diet-independently.
> Detection intact, valuation indifferent — and not learned from the correlation. (iv) Read 4 is reversed as a
> curriculum claim: a critic fed the model's own most surprising rows reaches the containment about fifty times
> later than one fed random rows of the same size, and an oracle damage gate is no better. Reads 1, 2, 3 and 6
> were not tested by these rounds; read 6's consumption-side certificates on the record are the candidate gain
> crossing zero and the period holding, not the per-level loss trend. (v) The follow-up to (iii),
> [`striatum/norm/`](../experiments/rhm/logit_reading/striatum/norm/README.md): the norm of the anger intuition is
> the value channel's own learned expectation over outcomes (§5's second dated note). It calibrates to the world,
> the anticipatory response does not scale with it, the outcome surprise does, and on same-prefix twins the
> critic prices the illegal token as better news than its legal twin — at the event, in a glitch world as much
> as an edit world, so not read 2's reinterpretation; the persistence after the event is world-specific and is. (vi) The six rounds of §11: reads 2 and 3 get their
> value-side form (reinterpretation is priced at the event and the persistence is world-specific; the level, not the
> revision, is what a consumer uses); read 6's "judged in consumption" now has a consumer, and the judge's private part
> is the revision.

### 10.1 A correction to §7, and where the forward model is

§7's second bullet undersold the record. The one-level-up currency **is** readable by the learner itself,
on both sides of the repo, and the reading has strengthened rather than been re-demoted:
[`teacher_slot`](../experiments/rhm/practice/teacher_slot/README.md) read the learner's own next-level
yield at a 13× read premium; `endo_yield`[^private]
read it label-free at 1.5% of budget; [`conductor`](../experiments/rhm/practice/conductor/README.md) and
[`crescendo`](../experiments/rhm/practice/crescendo/README.md) read it for free — the `at_support` count
over the learner's own successes paces commit and era-advance as well as the schedule inside the earnable
range and better beyond it — and ROADMAP §7.1.1[^private] records that the premium was the probe
form, not a conceptual gap. The type law is about *currency*, not about who reads it: a within-level
outcome ledger refuses the crossing, and any one-level-up gauge drives it, including one the learner
computes from its own trajectory. (Whether the law is currency-scoped rather than level-scoped — within-level
δ-silence paced the crossings above its bootstrap better than the yield gauge in
[`caesura`](../experiments/rhm/practice/two_deltas/README.md) — is flagged in ROADMAP §7.2.3 and not adopted.)

[`altitude/` Q3](../experiments/rhm/logit_reading/altitude/README.md#q3--the-models-predictive-as-a-null-for-scoring-candidate-units)
is the logit-side twin of that gauge: the model's own predictive, used as a null, prices candidate units at
exactly the rung above its altitude (`κ* ≈ j − 1`) and cancels once that rung is absorbed. So the learner's
logits read the one-level-up currency too — as the *location* of the frontier and the *price* of what sits
there. What they do not do is verify which candidate is true; model-free counts beat them at that
(0.72–0.81 vs 0.92–0.96). §7's "weak one-level-up currency" stands with that gloss.
[`striatum/`](../experiments/rhm/logit_reading/striatum/README.md)'s reader is outcome-currency and
within-level, and it never posed a crossing, so nothing in it bears on the type law either way.

One consistency note, since the forward-model side of the roadmap is currently demoted
(ROADMAP §7.1.3; [sparse_one_rung_up](sparse_one_rung_up.md) §7). The logit-reading arc has no forward
model anywhere: every readout is a functional of the logits or a linear head on the state. Everything that
came alive was a typed scalar per level — the same shape as E1's null on the practice side, where a richer
sensor of the learner's own update added nothing over the scalar already on the wire. This arc is what the
introspective line looks like after the demotion, and it found the meta-signals living in the predictive
itself.

### 10.2 The family is a property of the problem

Why the Bayesian correspondence is not a coincidence, stated as the argument it is.

> A learner minimizing expected log-loss on a nested hierarchy, in a world where level `k` is unlearnable
> before level `k − 1` (the Cagnetta–Wyart staircase), can only be one thing at any intermediate point: the
> Bayes posterior of the observer that knows the absorbed levels and treats the rest as noise. That is what
> partial knowledge of a hierarchy *is*, from the inside, for any learner and any learning rule — the family
> is fixed by the problem's nesting and its learnability order, not by the architecture. Cortex would have
> this structure whether or not it does gradient descent, provided the world is hierarchical and learning
> runs bottom-up. Part 1 measured it on a transformer.

Stated with its assumptions and its measured slack: the objective is log-loss (or anything whose minimizer
is the predictive); the hierarchy is nested and learned in order; "treats the rest as noise" is the specific
family — i.i.d. depth-`k` subtrees under a position-averaged root prior — and the record says the learner is
*that observer plus a residual*, with the observer carrying 70–89% of the residual while coarse and 32–40%
near convergence, the residual's level-shape a coarse observer's throughout, and the altitude continuous
rather than a rung
([Q1–Q2](../experiments/rhm/logit_reading/altitude/README.md#q1--the-identity-the-temperature-and-a-continuous-altitude);
C1 in §4). So the boxed sentence is the idealization, and the measurement is the idealization plus an
approximation error that thins as the learner becomes its own thing.

Once that is granted, the arc's meta-findings stop being findings and become corollaries — which is the
sense in which they should transfer to any learner of a hierarchy, cortex included.

- **The two neuromodulators are the two terms of one identity.** The loss at a position is stated entropy
  plus excess: `CE = H(q) + (CE − H(q))`. Yu & Dayan's *expected* uncertainty is the first term, the entropy
  the learner already holds; their *unexpected* uncertainty is the second, and Q1 showed it is the
  derivative of the loss along the logit-scaling direction. Temperature is precision is neuromodulatory
  gain. So the noradrenaline-shaped signal is *literally the gradient that says how to reset the gain* —
  zero on any world the learner has absorbed because the gain is already right there — and
  [`coeruleus/` Q3](../experiments/rhm/logit_reading/coeruleus/README.md#q3--the-gain-loop) closed exactly
  that loop: the excess head setting the temperature recovers half the prize where the gradient is nonzero
  and nothing where it is zero. In a brain this reads as: the locus coeruleus performs one step of descent
  on precision, online, per event, without touching a synapse. An op on the *conditions* of inference, in
  ROADMAP §2.1[^private]'s clause (ii).
- **The excess is a shift detector at every level and a frontier detector at none; the frontier is a
  derivative.** By the identity the excess is zero at *every* level the learner can scale separately —
  absorbed levels, the rung being absorbed, and the levels far above it, where the learner treats the level
  as noise, states high entropy, and is right about it (calibrated ignorance); Q1 measured `α*` within
  ±0.015 across token levels from 2k steps. So on the world the learner was trained on, "surprise beyond
  what I said" locates nothing. (An earlier draft of this bullet said the excess is nonzero at the rung
  being absorbed; Q1's per-class table says otherwise, and the sentence is corrected here rather than
  removed.) What locates the frontier from the inside is the *trend*: the realised loss at a level falling
  over time — learning progress, a derivative — and, given candidates, Q3's candidate gain, which is
  positive only at the rung above the altitude (`κ* ≈ j − 1`) and cancels once the rung is in. The band-pass
  shape [two_timescale_value_loop](two_timescale_value_loop.md) specified for LP — ~0 when mastered, ~0 when
  irreducible, peaked at the frontier — falls out of the derivative with no drive designed in. And the two
  signals are complementary in exactly §6's sense: the excess is a *level* that reads world change instantly
  and is blind to the frontier; LP is a *derivative* that reads the frontier and is blind to a re-opened one
  (two_timescale's Phase 2), which is a world change the excess sees. This is the one-level-up gauge of
  §10.1, computed from the logits alone — as a trend, not a level.
- **Silence is the certificate.** The frontier signals self-erase on absorption: Q3's candidate gain
  crosses zero level by level in order (`j = 1` at step 2000, `j = 2` at 16000), and LP at a level goes to
  zero once the level is held. Their going quiet at a level is the readable fact that the level is held.
  This is the object `caesura` found from the other side — δ-silence pacing the crossings — and the object
  [absorption_blinds_the_evaluator §5](absorption_blinds_the_evaluator.md) already named as the value
  system's textbook observable: prediction-error signals are surprise-flavoured evidence *designed* to
  self-erase as the world model learns. The catch is in §10.5: to a derivative, silence from absorption and
  silence from a wall look the same.

"Which Bayesian am I" is then a posterior over a nested model class — a second-order variable in exactly
the sense [two_timescale_value_loop](two_timescale_value_loop.md) §(a) reserves for *meta*: a functional of
the learner's competence trajectory, not of the state — and the learner's own excess per level is the
evidence stream that updates it. [`basalis/`](../experiments/rhm/logit_reading/basalis/README.md)'s log
Bayes factor is the same move at the scale of one datum: model comparison between two hypotheses about what
just happened, computed from the learner's own likelihoods, at chance at the flag and ~0.8 one token later.

### 10.3 The instrument panel

Put the six nodes together and the learner's own predictive supplies four readable quantities, each a
scalar typed by level, each with a consumer that acts on the *conditions* of learning rather than on the
weights.

| readable from the learner itself | measured as | consumer in the arc | plausible homolog (via the originating conversation and emotion_in_cortex[^private]) |
|---|---|---|---|
| where my frontier is | the per-level loss trend across checkpoints (LP); the entropy profile's period ([Q2 A](../experiments/rhm/logit_reading/altitude/README.md#q2--the-frontier-from-the-logits-and-where-the-residual-lives), Q3) | none built here; `at_support` / `endo_yield` on the practice side | learning-progress drive; the cortical level currently plastic |
| has the world moved | pooled excess beyond stated uncertainty ([`coeruleus/` Q2](../experiments/rhm/logit_reading/coeruleus/README.md#q2--the-self-supervised-readout)) | temperature (Q3); plasticity gate (Q4) | noradrenaline gain and reset; cingulate volatility scaling the learning rate |
| which world is it, given a surprise | accumulated likelihood ratio, one step later (`basalis/` Q4) | the hold-vs-reinterpret weight (`basalis/` Q3) | the double-take; re-parse vs discount |
| did something I hold get damaged, toward a goal | outcome-trained readout of a broadcast copy (`striatum/` §2) | none; open-loop | striatal and amygdalar critics; the emotion |

What the arc measured is this panel with every consumer absent or idle: frozen trunk, fixed world, dense
free feedback, no goal until `striatum/` and no choice even then. The extrapolation to a human is mostly
about turning the consumers on — and the record already says what changes when you do.

### 10.4 The consumers turned on: six reads

Each read names what is measured and what is extrapolated. None is a measurement.

1. **A human is never on-distribution, so the scalar channel is never idle.** The arc's largest null — on
   the trained world every self-readout is an identity, a stimulus property or a null (§7) — is a fact
   about a converged monolith on a fixed world. Cortex is a coarse observer of a world it keeps climbing
   and that keeps changing. `coeruleus/`'s agreed reading, *the organ is for the world you do not know*,
   becomes in a human "the organ is always on, because there is always a rung being absorbed." What the
   identity adds is that the panel is honest about when to stop: the excess is loud only where the world has
   shifted and LP only where learning is happening; both are quiet everywhere else.

2. **Reinterpretation is the ecological default; holding is the special case.** The model's native policy
   after an off-grammar token is to re-parse one rung down and commit
   ([Q4b](../experiments/rhm/logit_reading/altitude/README.md#q4b--entropy-direction-the-two-explanations-come-apart-and-what-the-forecast-moves-toward)),
   which no exact Bayesian in either family does. `basalis/` Q2 showed this is right for anything wider
   than a single token and wrong only for glitches, and that the learner's own second pass decides which,
   one beat later. The human read: most surprises are a *different legal world*, not a corrupted copy of
   the old one, so a learned predictor should re-parse by default and reserve "did I hear that right" for
   narrow events. Anger fits the reinterpret policy: it updates the model of the person; it does not
   discount the event.

3. **Affect precedes attribution.** Structure is read directionally and value as a scalar — in the arc
   (C3 against `striatum/` §5) and in the anatomy (per-feature error populations in superficial cortex;
   broadcast scalars from the neuromodulators and the striatum). `striatum/`'s negative — the state says
   *how much* was damaged and not *which* query — then predicts a familiar phenomenology: you feel that
   something you rely on broke before you know what, and cortex supplies the what afterward. The level
   carrying more than the revision fits the same picture: the felt state is a standing level, mood-like,
   and the event is a smaller signal riding on it.

4. **Surprise teaches value; then value subsumes surprise.** Once trained on damage, the critic's level
   contained everything the excess readout knew about the goal's failure, and the clean-only critic did
   not, so exposure to damage bought the containment (`striatum/` addendum). Read developmentally: the
   noradrenergic channel flags the events value learns from — Johansen et al.'s expectation-modulated
   teaching signal, in the reading note — and once the association is learned the value signal fires
   directly and surprise has nothing left to add. You do not feel surprised and then hurt; you feel hurt.
   *Flagged*: our critic was trained on outcomes and not gated by surprise; the two co-occurred in its
   diet, and the sequence is a reading of a static result.

5. **The norm–cost junction is learned from correlation, not wired.** `striatum/` decorrelated consequence
   from level and surprisal by construction, and the outcome reader then ignored the norm entirely — a
   legal edit and an illegal one with the same cost are the same to it. In the raw diet consequence rises
   with the violation's level, so an *unmatched* reader would have looked norm-sensitive. The human read:
   norm violations are emotional because in a human life they predict cost (Xiang et al.'s same offer,
   different feeling by norm history), and the emotion is entirely contingent on that learned correlation.
   This hands the reading note's *compilation problem* a candidate answer: nothing compiles. The norm is
   the predictive, the value reader reads the entry relative to it, and a freshly instructed norm should
   produce surprise on violation but no anger until cost has been experienced or simulated. It also
   predicts extinction of the feeling with intact detection — you still notice it is odd; you stop caring.

6. **Abstractions are judged in consumption, not by inspection.** The learner's predictive locates the
   frontier and prices candidates one rung up, but model-free counts beat it on whether a candidate is
   real (Q3), and the practice arc found the same thing from the other side: pre-commit certification
   demoted in every regime, provisional commitment graded in consumption winning
   ([`ear`](../experiments/rhm/practice/ear/README.md), [`recital`](../experiments/rhm/practice/recital/README.md),
   [`typed_gaps`](../experiments/rhm/practice/typed_gaps/README.md)). The human read is that nobody
   verifies an abstraction directly. You commit it provisionally, and the panel grades it afterward:
   loss at the level above falls if it was real, the candidate gain there cancels as it is absorbed, and the
   value channel reports whether goals stopped being damaged. Insight phenomenology fits — a sudden drop in
   loss at a level, then silence there. On this read, *judging the abstractions formed by a learner using
   the learner itself* means reading the learner's own excess and outcome ledgers per level, over time,
   after the commit. **The judge is not a smarter organ. It is a set of typed functionals of the learner's
   own predictive and outcome streams, consumed by ops on the conditions of learning.** This is §7 with
   the consumers on, and [heterogeneous_graders](heterogeneous_graders.md) §8's mirror-grader criterion is
   unaffected by it: a functional of the learner's own predictive is still blind where the learner is
   blind about *truth*; it is sighted about *change*, and change after a commit is what consumption grades.

### 10.5 What the conditions hide, and where the seam is

The arc characterized the readouts under conditions where their consumers had nothing to do; the practice
arc characterized the control loop under conditions where the readouts were supplied by exact instruments —
an exact reader, executor and grader. Neither has run inside the other. The closest existing object is the
climbing reader in [`reread/lm`](../experiments/rhm/practice/reread/lm/README.md), a next-token model that
keeps extracting levels from a fixed archive, where the per-level loss trend would be a live frontier gauge on
a *moving* learner rather than a frozen one, and where the corpus wall that node measured is the case a
derivative cannot tell from absorption from the inside — a level that stalls because the archive is
exhausted goes as quiet as one that was absorbed, and whether anything endogenous (the off-distribution
excess on held-out windows, the memorisation gap; the entropy period) tells the two apart is open. §9's
residual-share instrument sits at the same seam. Stated here as where the two lines meet rather than as a
prediction of what they would show. What experiments follow is under discussion
and deliberately not written here.

### 10.6 What §10 does not claim

- **It is a read.** Every sentence in §10.4 goes beyond the record by design; the record is cited so the
  distance is visible.
- **The "any learner" argument is an argument**, with its assumptions in §10.2 and its measured slack (C1
  thins with training). A learner that is partially informed in a way the nested family does not span —
  some rules at level `k + 1` and not others — is outside it, and nothing here says how far cortex is from
  the family.
- **The homologs are anchors for shape**, cited through the originating conversation and the reading note,
  not read against primary sources — §8's caveat, unchanged.
- **Reads 4 and 5 are the ones an arm could contradict cheaply**, and neither has one; read 6 rests on the
  practice arc's certification results and on Q3, both single-substrate.


---

## 11. 2026-09-17, evening — the norm is a projection: what six rounds on the value side settled

**Status**: a dated synthesis of measurements, not a read. Every number is in
[`rhm/logit_reading/orbitofrontal/`](../experiments/rhm/logit_reading/orbitofrontal/README.md) and its children, plus
[`rhm/practice/voicing/tessitura/`](../experiments/rhm/practice/voicing/tessitura/README.md) and the
[re-reads](../experiments/rhm/logit_reading/striatum/results/NOTES_2026-09-17_rereads.md) of `striatum/` and
`norm/`; the reasoning behind the rounds is the session's `CONVERSATION.md`[^private].
**Attribution**: the six priorities and the reframing are the orchestrator's, agreed in discussion; the objection that
an expectation cannot be other than world-model-based, and the call for a quick test of the reframing, are Jasper's;
the tiering of findings by what replicates is the re-reads implementer's.

### 11.1 The reframing, and the objection that sharpened it

§5's second dated note had relocated the emotion story's norm to "the value side": the critic's learned expectation
over outcomes, "two norms in two currencies, neither reading the other's". Jasper's objection: an expectation cannot be
other than world-model-based, since being angry that an outcome was worse than expected presupposes a model of what to
expect. The objection is right and the note was sloppy. The critic reads the trunk's residual stream, which *is* the
world model's state; in a Bayesian, expected value is the predictive integrated against utility, one posterior read
through the goal. So the correct statement is: **the norm is the world model read through a projection whose weights
the goal's outcome history sets.** Input from the world model, weights from the history. "Neither reads the other's"
meant only that the projection discards the part of the world model's error that carries no information about
outcomes. The §2 isomorphism holds on the utility projection too: a level before, an outcome surprise at, a revision
after. What the predictive does not contain is the axis, not a second model.

### 11.2 What the six rounds established

- **A running estimate on its memory's clock.** Across a switch of worlds the norm re-calibrates at exactly the rate
  its memory replaces its rows, with no reproducible deviation across two orderings; the response carries the history
  on the same clock (r ≈ 0.9 across orderings, cell by cell); the outcome surprise fires at about five standard errors
  at a world change and returns, nil in every control, and a reader that cannot forget never returns. On the live
  learner's judge the lag decomposes: fifty cycles behind the world, two to five behind its own buffer, at two seeds.
  The value-side twin of §6's identity: zero-mean on the adapted world by construction.
- **It reads whatever predicts cost.** Shaping the trunk on the goal sharpens the cost reading and leaves legality at
  chance across every diet; a world where a violation is evidence of a hidden regime that predicts cost makes the
  reader carry the regime, track the exact filter's revision, and shrink its revision with the prior belief, with none
  of it in an i.i.d. control; the practice learner's judge reads structure where structure is the cause of a solve. The
  legality contrast that appears in both regime worlds is visibility: calibrated after the corruption the world model
  can see, over-optimistic after the one it cannot. The reader inherits the world model's blind spots.
- **The level gates; the revision does not.** With an abstain-or-commit choice, the critic's level is the sufficient
  state statistic on the learner's whole stream, worth half the oracle's prize and more than any surprise-shaped
  reading, and worth the least at the violating token and the most from five tokens on. Learning uses the error;
  acting uses the level. Actor-critic.
- **The norm is public; the response is private.** Sixteen numbers from the output distribution reproduce the norm's
  calibration, rescaling and outcome-surprise ordering; the revision's amplitude, damage reading, deep-level fit and
  glitch-world signal live in directions the output layer reads at a fifth of the chance overlap from 8k on, having
  started at chance. The trunk moved the value-relevant directions out of the readout's subspace while learning the
  grammar; the private part is where the learning went. Paper 2's re-derivation boundary, on the value side, with the
  boundary between the norm and the response; the sixth instance of directional-not-scalar with the geometry explicit.
- **A rule for reading.** Across three trajectory seeds and two orderings, what is read as a difference between
  conditions on identical rows reproduced to a few decimals; what is read as a rate or a direction along one
  trajectory mostly did not. Two rows of `norm/` were retired or moved to better-powered cells on that rule.

### 11.3 What §11 does not claim

- Which of the level and the outcome surprise is "the feeling": one consumer, a priced abstention, was built, and it
  uses the level; nothing built consumes the private revision.
- That calibration of the level is evidence about value reading: on a sigmoid readout a random-init trunk calibrates
  as well as a trained one, and discrimination is what the representation buys. Held as uncertain until a design
  dissociates them on the ridge.
- Transfer off the RHM substrates; the language-model port is held for discussion.


---

## 12. 2026-09-18 — one world model, many projections: the panel as one table, the free projection, and where the axis comes from

**Status**: a synthesis after #123[^private] and its
`CONVERSATION.md`[^private], written to simplify rather than
to add. Nothing new run. Every claim below carries one of three tags: **[measured]**, with the node; **[restated]**,
a derivation from identities and definitions already on the record; **[inference]**, going beyond the record. The
inferences are §12.3's last two bullets, §12.4's sorting and §12.5's reading of the homology.
**Attribution**: the ask for a synthesis that simplifies rather than dives into domain-specificity, the observation
that the picture had become more biologically homologous and that this is evidence, the objection that hunger
reaches cortex through subcortical wires and so needs no mechanism of its own, and the sharpening that innateness
lives in the target's sign rather than the state (relayed by Jasper from a second session) are Jasper's; the
three-noun vocabulary, the unification of the two rows at the log-likelihood utility, the two blindnesses, the
private-revision reading, the drive sorting and the subtraction reading of the homology are the orchestrator's
(this session), agreed in discussion.

### 12.1 Three nouns

The ridge every value-side node reads with has three ingredients: the state it reads, the weights it learns, and
the target it is fitted to (**[measured]**: the instrument of
[`striatum/`](../experiments/rhm/logit_reading/striatum/README.md), the closed-form ridge of
[`coeruleus/readout.py`](../experiments/rhm/logit_reading/coeruleus/readout.py)). In this vocabulary:

- A **goal** is a utility vector: how much each thing that could happen is worth. Nothing more.
- A **projection** is the world model read through a weight vector. When utility is observed only through realized
  outcomes, the outcome history is what fits the weights; that is what §11's "the goal's history sets the weights"
  means. **[restated]**
- The **level** is the number that comes out.

The linear sense is the measured one, and it is the sense in which the instrument and the anatomy agree: the
critic is a dot product of the state with a fixed direction
([`projection/`](../experiments/rhm/logit_reading/orbitofrontal/projection/README.md)); a striatal neuron with ten
thousand cortical synapses is a linear readout; and a linear sum of pre-formatted inputs reproduces the dopamine
error at r = 0.92, nearly intact with any single input area removed
(reading note §2–3[^private], Tian et al. 2016). The probabilistic sense is the same
thing seen from the belief: expected utility is the belief vector dotted with the utility vector, and when the
state carries the belief the two coincide. That the *level* is that dot product of the belief is measured, since
sixteen numbers of the output distribution reproduce the norm's calibration, rescaling and outcome-surprise
ordering; that the *revision* is not is measured on the same node
([`projection/`](../experiments/rhm/logit_reading/orbitofrontal/projection/README.md)). **[measured]**

### 12.2 The panel is one table

Every reading the arc has taken is a cell of this table.

| projection | level, before | error, at | revision, after | consumer built |
|---|---|---|---|---|
| **prediction**, utility `log q` | `H(q)` ([Q1](../experiments/rhm/logit_reading/altitude/README.md#q1--the-identity-the-temperature-and-a-continuous-altitude)) | `CE − H(q)`, the excess ([Q1](../experiments/rhm/logit_reading/altitude/README.md#q1--the-identity-the-temperature-and-a-continuous-altitude), [coeruleus Q2](../experiments/rhm/logit_reading/coeruleus/README.md#q2--the-self-supervised-readout)) | `KL(q′ ‖ q)`, directional ([conditional_revision](../experiments/rhm/conditional_revision/README.md), [Part 2c–d](../experiments/rhm/logit_reading/README.md#2c-with-the-context-held-identical-there-is-a-response-to-the-token)) | temperature and the plasticity gate ([coeruleus Q3](../experiments/rhm/logit_reading/coeruleus/README.md#q3--the-gain-loop)) |
| **a goal**, utility the outcome | `V`, the norm ([norm](../experiments/rhm/logit_reading/striatum/norm/README.md), [adaptation](../experiments/rhm/logit_reading/orbitofrontal/adaptation/README.md)) | `outcome − V` ([norm](../experiments/rhm/logit_reading/striatum/norm/README.md), [adaptation](../experiments/rhm/logit_reading/orbitofrontal/adaptation/README.md)) | `R`, the projected revision ([striatum](../experiments/rhm/logit_reading/striatum/README.md), [regime](../experiments/rhm/logit_reading/orbitofrontal/regime/README.md)) | abstain-or-commit ([abstain](../experiments/rhm/logit_reading/orbitofrontal/abstain/README.md)) |

**The two rows are one construction.** **[restated]** For any utility `u`, the level is the expectation
`Σ_x q(x) u(x)`, the realized value at the outcome is `u(x)`, and the error is their difference, which has mean
zero under `q` by the definition of the level. For the goal "predict well" under log-loss the worth of outcome
`x` is `log q(x)`: the level is `Σ_x q(x) log q(x) = −H(q)`, the realized value is `log q(x)`, and the error is
`log q(x) + H(q) = −(s − H(q))`, the excess surprise of §6 with the sign flipped because surprisal is a cost. So
the "two prediction errors" of the originating conversation are one construction at two utilities. The general
fact is that the error is zero-mean under the belief for any utility. §6's temperature-gradient identity is the
mechanism by which a trained softmax gets the expectation right for this particular utility (`α* = 1.00–1.01` from
2k steps, [Q1](../experiments/rhm/logit_reading/altitude/README.md#q1--the-identity-the-temperature-and-a-continuous-altitude)),
and the ridge intercept is the mechanism by which the outcome reader gets it right for the goal's (`δ` at zero
before a switch of worlds, about five standard errors at it, back inside the band by 175–300 windows, nil in the
world-does-not-change control, [adaptation](../experiments/rhm/logit_reading/orbitofrontal/adaptation/README.md)).

Two neuroscience results become two rows of one table (**[restated]**, anchored through the
reading note[^private] and the originating conversation, not primary sources): Yu &
Dayan's expected and unexpected uncertainty are the level and the error of the prediction row; Xiang, Lohrenz and
Montague's norm and norm prediction error are the level and the error of the goal row.

**The prediction row is the degenerate one, and that is why the structural side was silent.** **[restated]**
`log q` is the one utility a world model can supply to itself with no outside information: it is the proper
scoring rule's own reward. A projection whose weights are a function of the belief cannot disagree with the
belief; it can only report the belief's shape, which is what entropy is. That is §7's "every on-distribution
self-readout was an identity, a stimulus property or a null" in one line, and it is why the value side became
non-degenerate only when [`striatum/`](../experiments/rhm/logit_reading/striatum/README.md) handed the trunk a
utility the world model did not contain. It is also
[heterogeneous_graders §4](heterogeneous_graders.md)'s dichotomy in this vocabulary: the dense grader is the
projection at the self-supplied utility, and the evaluative grader is the projection at any other.

### 12.3 What collapsed

- **What the value reader reads is one law.** **[measured]** It reads whatever in the state predicts the outcome,
  and which thing that is belongs to the world, not the reader: legality at chance across eleven diets and every
  goal-shaped arm where legality says nothing about outcomes
  ([junction](../experiments/rhm/logit_reading/striatum/junction/README.md),
  [shaped](../experiments/rhm/logit_reading/orbitofrontal/shaped/README.md)); the hidden regime, priced by the
  prior belief, where a violation is evidence about future cost
  ([regime](../experiments/rhm/logit_reading/orbitofrontal/regime/README.md)); structure above cost where structure
  is the proximal cause of a solve ([tessitura](../experiments/rhm/practice/voicing/tessitura/README.md),
  [orbitofrontal §2c](../experiments/rhm/logit_reading/orbitofrontal/README.md)). The question "does the value
  reader read the norm" retires: it reads the norm exactly when the norm predicts cost.

- **Two blindnesses, orthogonal.** **[measured on both; named here]** A projection is blind where the world model
  is: calibrated after the corruption the model can see, over-optimistic (0.739 predicted against 0.320 realized)
  after the legal-but-wrong token it re-parses around, identically in the i.i.d. control
  ([regime](../experiments/rhm/logit_reading/orbitofrontal/regime/README.md)). And a projection is blind where its
  weights never valued the thing: two judges of the identical learned class on identical inputs, differing only in
  reward, come out near-orthogonal (cosine +0.19 / +0.56), and the within-level one never takes the next level
  while its own input reads the one-level-up gauge at two or more floor units
  ([maestro](../experiments/rhm/practice/maestro/README.md)). Input blindness and weight blindness. The beliefs
  tree's "valuational, not perceptual"
  ([cerebellum tree](../beliefs/trees/cerebellum_and_cognitive_architecture.md)) is the second kind.

- **The two arcs speak one vocabulary.** **[measured that the objects coincide; the identification is restated]**
  The practice arc's judge is this object: norm's three questions asked of the voicing judge returned the same
  running estimate on the same clock, at two seeds
  ([tessitura](../experiments/rhm/practice/voicing/tessitura/README.md)). So maestro's within-level judge that
  refuses the crossing and norm's critic that ignores legality are one sentence: same world model, different
  weights. The roadmap's one-level-up currency is a projection whose utility is next-level yield, and the record
  that the learner can read it for itself ([teacher_slot](../experiments/rhm/practice/teacher_slot/README.md),
  [conductor](../experiments/rhm/practice/conductor/README.md),
  [crescendo](../experiments/rhm/practice/crescendo/README.md)) is the record that such a projection is fittable
  from the learner's own trajectory.

- **The mirror-grader criterion, sharpened.** **[inference** from [heterogeneous_graders §8](heterogeneous_graders.md)
  and the two blindnesses**]** An LLM-as-judge is the same projection, same utility and same history, so it is
  blind in the same places. A genuine second grader is a different projection of the same world model, and it is
  sighted where the first is blind exactly where its history put weight the first's did not. Heterogeneous graders
  are heterogeneous projections, and a brain runs them on one shared state, which is why it is cheaper than a
  population.

- **What the private revision is for.** **[inference]** §11.3 left open what consumes the private revision, since
  the one consumer built uses the level
  ([abstain](../experiments/rhm/logit_reading/orbitofrontal/abstain/README.md)). The reading: the value-side
  revision `R` is the world model's own re-parse seen through the projection, which is what
  [regime](../experiments/rhm/logit_reading/orbitofrontal/regime/README.md) measured when `R` tracked the exact
  filter's revision and shrank with the prior belief; and what consumes the re-parse is the world model's next
  step. On a frozen trunk nothing else can consume it, so its being unconsumed on the value side is expected. On a
  live learner the same revision is the trunk's own gradient. The assignment that follows: the scalar error trains
  the projection's weights, and the directional revision trains the world model. Dopamine to striatal synapses,
  per-feature cortical prediction error to cortical synapses
  (reading note §2[^private], the originating conversation). The consumer built found the
  level worth least at the violating token and most from five tokens on, doing worse than a constant for three
  tokens after a violation ([abstain](../experiments/rhm/logit_reading/orbitofrontal/abstain/README.md)); on this
  reading the felt response at the event is real and directional, and what governs behaviour is the settled level
  after the re-parse. Untested: a live-trunk arm where the value-side revision's consumption by the trunk's own
  update could be measured.

### 12.4 Where the axis comes from: the complexity to keep

(The sharpening is Jasper's, relayed; the sorting is an **[inference]**; the anchors are the reading note's.)

Of the ridge's three ingredients, the state comes from the world model and the weights are learned, for every
drive. Innateness lives in the third: the **sign of the target**. Cortex can represent hunger, predict its relief
and plan around it, so the state is not the innate part; nothing in the predictive structure says relief is worth
having, and that has to be pinned from outside the likelihood. That is the reading note's own line: innate
detectors supply the primary reinforcers, and learned and abstract categories attach to those by higher-order
conditioning (reading note §3[^private]). Biology supplies the sign by a wire that reaches
the teacher without waiting for a projection to learn it: the expectation-modulated shock signal that arrives at
the amygdala through the periaqueductal gray (Johansen et al. 2010,
reading note §2[^private]).

The one projection that needs no supplied sign is prediction, where the target is the belief's own
log-probability. What rides on it for free: learning progress is the derivative of `−H(q)` at a level, novelty is
the excess, and the band-pass shape of a learning-progress drive falls out of the derivative with no drive
designed in (§10.2, [two_timescale_value_loop](two_timescale_value_loop.md),
[CURIOSITY_DRIVE](../experiments/a2a_forward/reaching/CURIOSITY_DRIVE_README.md)). So "innate curiosity" is innate
*consumption* of a reading every learner already has, and "innate hunger" is an innate *sign* on a target the
world model cannot supply. For our learner the supplied sign is whatever the task hands it, which is the one
thing [`striatum/`](../experiments/rhm/logit_reading/striatum/README.md)'s actor did.

Which projections should exist is outside this document. It is the real open design question for the practice
arc, and it is where domain-specificity legitimately lives.

> **Added 2026-09-18, later the same day: how a candidate gets priced before it is integrated, and why replay is
> forced rather than convenient.** Jasper's question was how the complementary-learning-systems picture, with the
> hippocampus as a buffer for representations not yet integrated into cortex, squares with a cortical picture of
> value signalling; then whether replay and preplay are core or a nice-to-have, whether taken and untaken
> trajectories differ, and whether thoughts and motor sequences do. **[inference]** throughout, on anatomy anchored
> in training knowledge and not read against primary sources this session; §8's caveat applies. The record is in
> `conversations/one_world_model_many_projections_2026-09-18.md`[^private].
>
> - **The readout does not read cortex. It reads a broadcast of everything, and the hippocampus is on it directly.**
>   The ventral striatum receives a monosynaptic projection from ventral CA1 and the subiculum, the basolateral
>   amygdala is reciprocally wired with the hippocampus, and ventral hippocampus projects directly to medial
>   prefrontal and orbitofrontal cortex. In §12.1's terms the state the projection reads is the union. Nothing
>   needs mirroring into cortex to be priced. And hippocampal representations are used for behaviour immediately,
>   not only held for later, so they are evaluated the way any state is.
> - **The hippocampus has its own prediction-row error, and it is wired to the teacher.** CA3's pattern completion
>   is a prediction, CA1 compares it with entorhinal input, and the mismatch is novelty. Lisman and Grace's loop
>   (2005) runs it through the accumbens and ventral pallidum to VTA, and dopamine returns to gate lasting storage,
>   with locus-coeruleus co-release of dopamine in CA1 for novelty (Takeuchi et al. and Kempadoo et al., 2016).
>   This is the plasticity gate of [coeruleus Q4](../experiments/rhm/logit_reading/coeruleus/README.md) in anatomy,
>   and the "innate consumption of a free reading" case above.
> - **Firing is forced by the readout's type.** A linear readout of state can only price what is on the wire. A
>   candidate that exists as a table entry, a synaptic configuration or a "could" has no price until the world
>   model is put into the state it implies. That is read 6 of §10.4, judged in consumption and not by inspection,
>   with a mechanism: inspection would need a reader of weights or of the table, and the value system has none
>   (Tian et al.: a weighted sum of inputs). The practice arc measured the same thing when provisional commitment
>   graded in consumption beat every certificate ([ear](../experiments/rhm/practice/ear/README.md),
>   [recital](../experiments/rhm/practice/recital/README.md)).
> - **Replay is firing when acting is priced, and it is core in proportion to the meter.** State is
>   single-occupancy: the world model is in one state at a time and online the world dictates which, so pricing a
>   candidate means borrowing the state from the world (rest, sleep, the within-theta-cycle interleaving of now and
>   next). And the informative outcomes are the priced ones ([heterogeneous_graders §4](heterogeneous_graders.md)).
>   When trials are cheap, acting is the evaluator; when they are dangerous, slow or scarce, offline firing is the
>   only way to price enough candidates. Offline placement in sleep is a third, separate matter, interleaving
>   against interference, which is the original complementary-learning-systems argument. Only the first layer is
>   forced by the readout and the second by the meter. What gets integrated into cortex is already selected by the
>   projection: replay is coordinated with ventral striatal and VTA reactivation (Lansink et al. 2009; Gomperts et
>   al. 2015), reverse replay scales with reward (Foster & Wilson 2006; Ambrose et al. 2016), Mattar & Daw (2018)
>   formalise the priority as gain × need, and the 2016 update of the theory by Kumaran, Hassabis and McClelland
>   builds reward-weighted replay into consolidation explicitly.
> - **Taken versus untaken is not the axis. An outcome from outside the world model is.** From the readout's side
>   replay and preplay are identical, same cells and same sequences, and the projection does not know whether the
>   sequence came from the sensors or from CA3. The difference is on the learning side: a taken trajectory carries
>   a realized outcome, so the level and the error are both available; an untaken one carries only the forecast,
>   which the level already is. Replay trains the projection and preplay queries it, the
>   [abstain](../experiments/rhm/logit_reading/orbitofrontal/abstain/README.md) split again. A thought is a
>   trajectory in state space and thinking it is taking it; what a thought lacks is not having been taken but an
>   outcome from outside the world model. So the motor and cognitive cases run on one channel with one readout and
>   differ only in how often an external error arrives: every act on the motor side, only at cash-out on the
>   cognitive side. Cognition is the metered regime by construction, which is why replay is more core for thought
>   than for movement, and why chain-of-thought is preplay with no external outcome
>   ([heterogeneous_graders §8](heterogeneous_graders.md)).
> - **The rider.** A preplayed trajectory priced through the projection inherits both blindnesses of §12.3: the
>   forecast of an untaken path can be wrong the way [regime](../experiments/rhm/logit_reading/orbitofrontal/regime/README.md)
>   measured, with no outcome to correct it, and the projection may not value what the path does. Preplay alone is
>   inspection by simulation, the mirror grader. Biology's hedge fits: replay is biased toward taken-and-rewarded
>   paths, and preplay for planning sits at choice points where the real outcome arrives soon after.
> - **One read from the arc's own consumer.** [abstain](../experiments/rhm/logit_reading/orbitofrontal/abstain/README.md)
>   found the level worth least at the event and most from about five tokens on, once the re-parse has settled. If
>   pricing a candidate is reading the level of the fired state, the candidate has to be run for a few steps before
>   the reading is any good, which is a reason replay is sequences rather than snapshots.
> - **Where the practice arc already has this.** [practice_manufactures_its_own_credit](practice_manufactures_its_own_credit.md)'s
>   revision of 2026-08-20 assigned the mined table the hippocampal profile on Iwane et al.'s data: an arbitrary
>   binding stored fast, populated offline by selection over stored traces, expanding on rest-break replay. The
>   roadmap[^private]'s practice row, a table committed verbatim and consolidated as routing by
>   self-imitation, is complementary learning systems by construction: the verbatim table is the hippocampus,
>   routing consolidated on replayed traces is cortex, and the mining step is the value-prioritised replay. The one
>   piece the arc has not built is replay through the value projection of trajectories never taken, which is
>   where a directional which-world revision would have a consumer. Noted as a gap, not proposed.

### 12.5 The homology, and what kind of evidence it is

(The reading is the orchestrator's; that convergence is evidence is Jasper's; both are **[inference]**.)

Every move in §11 that made the picture more biological was a subtraction forced by a null: the second world model
went ([junction](../experiments/rhm/logit_reading/striatum/junction/README.md)'s null and §11.1), the norm-cost
junction organ went (the same node), and the reader's own time constant went
([adaptation](../experiments/rhm/logit_reading/orbitofrontal/adaptation/README.md)'s deviation from the mechanical
null at chance across two orderings; [tessitura](../experiments/rhm/practice/voicing/tessitura/README.md)'s two to
five cycles on a fifty-cycle diet lag). What is left is the actor-critic split of the originating conversation,
arrived at by having posited more and measured less. That is the trustworthy kind of convergence, because each
step was a measurement rather than a fit to the textbook.

What is converging is not "the model matches biology" but "the textbook computational account is forced for any
learner in this position", which is §10.2's property-of-the-problem argument applied to the value side. Stated as
the argument it is: any calibrated outcome reader on a state that is a posterior computes expected utility under
that posterior, with an error that is zero-mean on the rows that calibrated it; Xiang's same-offer-different-
feeling follows for any such learner. The structural half is §10.2's boxed sentence; the value half is §12.2 with
"calibrated to its history" as the assumption, which
[adaptation](../experiments/rhm/logit_reading/orbitofrontal/adaptation/README.md) and the three trajectory seeds
([re-reads](../experiments/rhm/logit_reading/striatum/results/NOTES_2026-09-17_rereads.md)) measured. This is
stronger than homology, and it is the reason the picture should transfer to a language model, which homology
alone would not promise.

Three things not to lean on:

- **The match is at the computational level, with fitted models.** Xiang is fMRI plus a Bayesian observer fit,
  and the reading note says the anatomy is known in broad strokes. What is reproduced is the description
  neuroscientists fit to humans, not what evolution built; the inference that survives is that the description is
  forced.
- **The public/private split's anatomical reading is the weakest row.** Cortex has no output layer (the
  originating conversation); the split rests on a rank-fifteen bottleneck over a sixteen-token vocabulary
  ([projection/](../experiments/rhm/logit_reading/orbitofrontal/projection/README.md) and the super-node's
  does-not-establish). The structural fact it restates, a scalar level and a directional revision, is the sixth
  instance of a pattern and stands on its own.
- **Matching the textbook this well is partly because the textbook describes what a minimal linear calibrated
  reader does.** The match confirms the textbook is right about the minimum. What is on the record beyond it: the
  two blindnesses measured; the excess as the value error's twin; the level worth least at the event; the trunk
  moving the value-relevant directions out of the output subspace while learning the grammar, from chance at step
  0 to a fifth of chance by 8k ([projection/](../experiments/rhm/logit_reading/orbitofrontal/projection/README.md));
  and the identical-rows-versus-trajectory rule
  ([re-reads](../experiments/rhm/logit_reading/striatum/results/NOTES_2026-09-17_rereads.md)).

### 12.6 What §12 does not claim

- That the private revision is consumed by the trunk on a live learner: an inference from a frozen-trunk record,
  and a prediction.
- That the drive sorting of §12.4 is more than a reading of the reading note; nothing in the repo carries a
  supplied sign other than a task's.
- The specific overlap numbers behind "public" and "private"; the below-chance floor and its appearance between
  step 0 and 8k are the claims.
- More than one consumer per row. The prediction row's pays only off-distribution
  ([coeruleus Q3](../experiments/rhm/logit_reading/coeruleus/README.md#q3--the-gain-loop)); the goal row's is a
  priced abstention ([abstain](../experiments/rhm/logit_reading/orbitofrontal/abstain/README.md)); neither is the
  emotion.
- Transfer off the RHM substrates. The language-model port in QUEUE[^private] is the test of "forced by the
  problem" against "true of this grammar".
- The replay note in §12.4: its anatomy is from training knowledge, not primary sources read this session, and
  hippocampus-as-provisional-commitment is a reading of the complementary-learning-systems picture, not a result.

[^private]: Not mirrored: this link points to a document in the private lab repo (the roadmap, the queue, an unrun spec, reading notes, or a conversation). See the top-level README for what is held back and why.
