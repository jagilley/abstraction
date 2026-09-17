# Calibration and violation are one object read twice: why "the logits are Bayesian" and "emotion is norm violation" are isomorphic, and where they part

**Status**: conceptual synthesis (2026-09-16); §5's two organs each measured once on 2026-09-17 (dated note in §5). Nothing new run for the synthesis itself. Every number below is re-read from
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

[^private]: Not mirrored: this link points to a document in the private lab repo (the roadmap, the queue, an unrun spec, reading notes, or a conversation). See the top-level README for what is held back and why.
