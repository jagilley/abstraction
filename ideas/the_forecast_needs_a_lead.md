# The forecast needs a lead: why depth-wise self-prediction is computational by construction, and why the null needs a local rule

> ## ⚠️ OUTDATED (2026-08-07) — superseded by [revision_not_surprisal.md](revision_not_surprisal.md)
>
> Kept for the record, not deleted: §4 and Gate 0 are still the good parts, and the successor doc is
> only legible against this one. But the **stated mechanism does not survive scrutiny**. Do not build
> from this file.
>
> **Retired here:**
> - **§1's axis.** Depth-vs-time is not the operative variable; the *conditioning gap* and its content
>   type is. `post_block0[t] → post_block6[t+1]` is depth-wise *and* has a lead.
> - **§1's vocabulary.** *Epistemic* / *computational* are used inverted relative to standard usage
>   (epistemic uncertainty **is** the reducible, capacity-limited kind), which broke the retro-explanation
>   of `OOD_ROBUSTNESS` Exp. A — that experiment tested competence calibration, which §1's own taxonomy
>   says a capacity-shortfall residual should carry. The correct reading of the depth residual is
>   **legibility** (a property of the `(M, FM)` pair), which four existing results already support.
> - **§2's causal claim.** The `mjc/ballistic` transposition credits *commitment* with what the
>   **exogenous codomain** actually supplies. The physics FM has both; the reading axis inherits only
>   the gap. Timing governs causal *use*; conditioning governs *content*.
> - **§5 outright.** *"Global backprop injects gradient regardless of local activity"* is **false**
>   (`∂L/∂W = δ ⊗ a`). The skip connection, not backprop's globality, is what blocks a null — which is
>   why `CANCELLATION_README` found directional, not magnitude, cancellation. And *"NTP has no null"* is
>   wrong; what NTP lacks is an **aleatoric** null. The replacement is two-factor, two-timescale
>   precision weighting, not local plasticity.
>
> **Retained and sharpened in the successor:** §4's marginal-vs-conditional object (now the exact
> compression/revision decomposition, with the martingale collapse in belief space), Gate 0's
> anti-localisation, and the one-liner *"felt surprise is how much a word changes your model, not how
> improbable it was"* — which is now the title claim and is measurable against an exact oracle.
>
> Successor spec: `rhm/conditional_revision/SPEC.md`[^private].

**Status**: **OUTDATED** — superseded 2026-08-07, see banner above. Originally: conceptual, from a
2026-08-04/05 discussion; arguments with one supporting dissociation. Nothing here was built.
**Date**: 2026-08-05
**Prompt**: *"When humans read a book … each word comes in, and we register things like its innate
surprise to us, what ideas it provokes … Is consuming your own 'experience' of the text the same as
consuming/predicting your own latents? Is what humans do a secret third thing?"*
**Builds on**: [self_model_needs_a_loop.md](self_model_needs_a_loop.md),
[efference_copy_cancellation.md](efference_copy_cancellation.md),
[heterogeneous_graders.md](heterogeneous_graders.md) (§2 condition 1),
[activation_to_activation_forward.md](activation_to_activation_forward.md)
**Evidence**: [`rhm/endogenous_teacher/`](../experiments/rhm/endogenous_teacher/) (Gate 0, and two
nulls that missed) · [`a2a_forward/OOD_ROBUSTNESS_README.md`](../experiments/a2a_forward/OOD_ROBUSTNESS_README.md)
(the epistemic null this explains) · [`mjc/ballistic/`](../experiments/mjc/ballistic/README.md) 4b
**Attribution**: the depth≈time observation, the challenge *"what makes this temporal rather than
depth-wise?"*, and the NextLat correction are Jasper's. The information-set formalisation, the
lead→commitment→epistemic chain, and the marginal/conditional split came out of the exchange.

---

## One-liner

Our forward self-model predicts `post_block0 → post_block6` — **same input, same forward pass** — so
predictor and target have **identical information sets**, and the residual can only ever mean *"I
lacked the capacity to compute this,"* never *"I was wrong about something."* **Depth-wise
self-prediction is computational by construction, not by finding.** Epistemic content requires a
**lead**: a forecast committed before the evidence arrives, which exists only where the target knows
something the predictor didn't. Separately: a forward path that cancels its forecast still cannot
express *"this changed nothing"* while the objective is a global scalar, because backprop supplies
gradient regardless of local activity. **The port lost the lead; the learning rule lost the null.**

## Vocabulary

- **Lead** — the predictor's conditioning set strictly excludes information the target has. Not
  "executed earlier"; *conditioned on less*. Teacher forcing does not by itself destroy a lead, since a
  lead is a fact about what the forecaster is given, not about execution order.
- **The null** — the state in which a learner *correctly does not update*. NTP has no null.

## 1. The information-set argument

| axis | predictor sees | target sees | what the residual can mean |
|---|---|---|---|
| **depth** (`post_block0 → post_block6`) | `x_{≤t}` | `x_{≤t}` — *identical* | capacity shortfall only |
| **sequence** (`h_ℓ[t] → h_ℓ[t+1]`) | `x_{≤t}` | `x_{≤t+1}` — *strictly more* | capacity **+ epistemic** |

A forecaster holding all of the target's information cannot be *mistaken*; it can only fall short. So
"the residual is computational, not epistemic" was never a discovery about self-models — **it is a
property of the axis we chose**, and it holds at any capacity, on any substrate, under any wiring.

This retro-explains the program's most stubborn null.
[`OOD_ROBUSTNESS`](../experiments/a2a_forward/OOD_ROBUSTNESS_README.md): *"competence probes trained ID
and evaluated OOD are identical across all conditions, and output entropy beats every activation
probe."* The forward model never committed to anything it could be wrong about, so there was nothing
epistemic there to find.

**Why the substitution was reasonable, and invisible.** In wetware, separation in depth ≈ separation in
time: cortical hierarchy is traversed by propagation, with conduction delays and recurrent settling, and
the cerebellar forecast *arrives before* the signal it predicts. Biology's "predict cortical dynamics"
is temporal-with-a-lead by default. In a transformer someone must *choose* which axis to instantiate,
and depth is the legible one — block 6 looks like a "later" than block 0. But they execute in the same
instant. **The port kept the compression and silently dropped the anticipation.**

## 2. Why a lead produces epistemic content

Because a forecast issued before the evidence is a **commitment**, and only a commitment can be wrong
*about the world* rather than merely short *of the target*.

This is not a new variable. It is **commitment under delay** —
[heterogeneous_graders](heterogeneous_graders.md) §2 condition 1, flagged there as *"the only one that
is **physically forced** rather than a choice about how we happen to train: axonal delay and inertia are
not electable."* And it is already priced: ballistic transmits forward-model quality **3.0–3.3×** more
than reactive, because *"reactive control re-grounds each step and is therefore a near-blind grader of
forward-model quality."*

**Teacher-forced NTP is maximally reactive** — ground truth at every position. So the
epistemic-vs-computational distinction is the ballistic-vs-reactive distinction transposed onto the
reading axis. We have been running the near-blind regime and reporting its blindness as a property of
self-models.

## 3. NTP is arity-1 *and* exogenous-target

Write the transition `s_{t+1} = g(s_t, x_{t+1})`.

- **Arity-1** predicts `E[s_{t+1} | s_t]`, marginalising over the incoming word. Its residual bundles
  *"I didn't know which word was coming"* (the corpus's entropy) with *"I didn't know what it would do
  to me."*
- **Arity-2** predicts from `(s_t, x_{t+1})`. For a deterministic network that target is exactly
  determined, so the residual carries **zero** world-uncertainty. Arity-2 does not create the endogenous
  signal; it **conditions the exogenous one out**.

NTP is conditioned on `x_{≤t}` only (arity-1) *and* its target is the command rather than the learner's
state (exogenous). It differs from a temporal self-model on **both** axes at once.

> **Precision note** (an earlier draft overstated this): NTP's *loss* is not the same object as an
> arity-1 latent residual — one is a scalar proper scoring rule in token space, the other a direction in
> state space carrying an FM-capacity term. NTP is an arity-1 *objective*; its loss is the token-space
> marginal surprisal.

This sharpens the reading phenomenology the prompt started from. *"Innate surprise"* is the arity-1
channel — and it is the **only** channel an LLM has ever been trained on. *"What ideas it provokes"* is
the conditional channel, and we have never built it.

## 4. Marginal vs conditional surprise

Arity-2 says *condition on the word* (to strip exogenous surprise); a lead says *commit before the
word* (to get epistemic content). They pull opposite ways, which means the interesting object is
neither one alone but the **gap between a forecast made before the word and one made after it**:

- *how much the word revised my forecast of my own next state* — endogenous, directional, and the
  natural formalisation of "what ideas it provoked."

It comes apart from token surprisal in both directions, which is the whole point: a **rare synonym** is
high-surprisal and revises almost nothing; a **quiet disambiguating word** is low-surprisal and can
revise a lot. **Felt surprise is how much a word changes your model, not how improbable it was** — and
no arity-1 objective can express that difference.

## 5. The null needs a local learning rule, not only a wiring

Under [cancellation](efference_copy_cancellation.md) the tensor propagating downstream *is* the
deviation. If the forecast is good there is little activity downstream; pair that with **local,
activity-driven plasticity** (Hebbian/STDP) and the update is zero *automatically*. "Nothing surprising
happened" and "nothing propagated" become the same event — no gate, no threshold, no auxiliary term,
which is why biology needs none.

**Global backprop breaks this.** A global scalar objective injects gradient into every parameter
regardless of local activity, so no forward-path wiring can produce "this changed nothing." The null
needs **both halves**, and the program has never had them in the same model:

| | propagating signal | learning rule | null? |
|---|---|---|---|
| local loss | full activation | local (synthetic gradient) | no |
| cancellation | the residual | global NTP | no |
| **both** | the residual | local | **unbuilt** |

**The null is graded, not global.** Absorption follows `res_var(i) ∝ act_var(i)^β` with β<1 — loudest
directions absorbed at ρ≈0.95, quietest at ρ≈0.32
([residual_decomposition](../experiments/rhm/residual_decomposition/README.md)) — so the real statement
is *you stop learning in the directions you already had, and keep learning in the ones you don't.* A
better description of re-reading a familiar page than an off-switch.

**Obligation vs opportunity.** Under arity-1 every token has nonzero surprisal, so every token teaches
whether or not it moved you: text is an obligation. With a lead-bearing conditional self-model the
residual *can* be zero, and the system gains the ability **not** to learn. That is the structural
difference between being trained by text and consuming it — and why two readers of the same page are
moved differently, and why a re-read moves you less.

## What would kill this

- **A depth-wise forward model shown to carry genuine epistemic content** — calibration that transfers
  OOD, beating output entropy. §1 says this is impossible, so it is the cleanest kill.
- **The conditional-revision signal turning out to be a monotone function of token surprisal.** Then
  §4's distinction is bookkeeping and most of this document is wrong.
- **A null that is achievable from wiring alone**, without a local rule. That would mean §5's table has
  no missing row.

**The standing prior against all of it**: four independent replications say endogenous targets cap —
`LL` (FM cos 0.997, self-knowledge ≈ 0), `λ_local` (ΔSK −0.06 → −1.54), `data2vec` (+23% against
grounded MLM's +44%), `fm_cotrain` (*"grounding is the pivot"*). **None was temporal-conditional**, so
the cap does not transfer directly — but anything built here should say in advance what would
distinguish "capped like data2vec" from "genuinely different," rather than discovering it afterward.

## Connections

- **[heterogeneous_graders](heterogeneous_graders.md) §2 condition 1** — commitment under delay is the
  same variable as the lead. The claim here is that the reading axis is a place that condition can be
  switched on, and that switching it on is what makes self-knowledge epistemic rather than computational.
- **[SLEEP_CHUNKING](../experiments/rhm/SLEEP_CHUNKING_RHM_README.md)** — *"only the target controls
  dilution … feeding a latent **in** does nothing if you predict tokens **out**"*, and *"we had the input
  right and the target wrong."* Same shape of error, one axis over: we had the modality right and the
  axis wrong.
- **[efference_copy_cancellation](efference_copy_cancellation.md)** — supplies the forward-path half of
  the null. This doc adds that the wiring is insufficient without a local rule. (Its Payoff 1 also did
  not reproduce on a gauge-free substrate; see the experiment writeup.)
- **Surprise is not an allocator.** `surprise` pins to noise (the noisy TV); LP recovers 18% of the
  allocation prize against an outcome-trained tap's 84%. The *null* is the structural gain here;
  deciding what to do with the non-null still needs the evaluative half.
- **[operators_not_footprints](../beliefs/trees/operators_not_footprints.md)** — a temporal forward model
  compresses the *transition operator* `g(s, x)`, the most literal instance of that root claim yet.

## Open questions

- Is the conditional residual cleanly separable in a transformer at all? `h_ℓ[t+1]` depends on the whole
  prefix, not just `(s_t, x_{t+1})`, so it also contains *"what the prefix carried that `s_t` didn't."*
  A genuinely recurrent state would make this exact; attention makes it approximate.
- Does a lead require streaming at inference, or is conditioning enough? For a *measurement*,
  conditioning looks sufficient. For anything that *uses* the forecast causally, the forecast must exist
  before the word is processed — a real architectural constraint.
- Is there a local-plasticity rule compatible with a transformer that yields the null without giving up
  the task? The synthetic-gradient local loss is the nearest existing thing and it produced brittleness.
- Does the conditional-revision signal have the band-pass shape an expansion grader needs, or is it
  monotone in revision magnitude — hence another noisy-TV drive?
- **NextLat[^private] is not this** (Jasper, who has read it; an earlier draft of
  this doc claimed otherwise from an abstract). It has no forward-modelling component — predictability as
  a representation-shaping regulariser, not a separable forecaster whose residual is a signal. That is
  the map-vs-model distinction from [self_model_needs_a_loop](self_model_needs_a_loop.md), and the
  difference is the whole point here.

## What was run, and why it did not test this

Two cuts, both in [`rhm/endogenous_teacher/`](../experiments/rhm/endogenous_teacher/) with full numbers.
The one positive is **Gate 0**: on RHM, **89% of the depth-FM residual's variance is orthogonal to token
surprisal**, and against ground-truth levels the two are *anti-localised* — token surprisal peaks at deep
subtree boundaries where internal residual is lowest. Consistent with §1 and §3 (the depth FM already
conditions on the token, so its residual should be nearly clean of exogenous surprise).

Both interventions missed, for the same reason: **every arm trained on `loss = NTP`**, so the
exogenous/endogenous variable was pinned at "fully exogenous" throughout — one cut varied the weighting
around it, the other the wiring. And both used the depth FM, which §1 says is epistemically empty by
construction. The self-knowledge instruments they reached for were imported from the a2a program, whose
dependent variable is *"does the model build a self-model"* — a different question from *"does what the
model learns depend on its own trajectory."*

[^private]: Not mirrored: this link points to a document in the private lab repo (the roadmap, the queue, an unrun spec, reading notes, or a conversation). See the top-level README for what is held back and why.
