# The forecast needs a lead: why depth-wise self-prediction is computational by construction, and why the null needs a local rule

**Status**: Conceptual, from a 2026-08-04/05 discussion. One supporting measurement (Gate 0 below) and
two negative cuts, both run in that session. The central claims — the information-set argument, the
lead→epistemic chain, and the three-way decomposition — are **arguments plus one dissociation**, and the
decomposition is **unbuilt**.
**Date**: 2026-08-05
**Prompt**: *"When humans read a book … each word comes in, and we register things like its innate
surprise to us, what ideas it provokes … Is consuming your own 'experience' of the text the same as
consuming/predicting your own latents? Is what humans do a secret third thing?"*
**Builds on**: [self_model_needs_a_loop.md](self_model_needs_a_loop.md),
[efference_copy_cancellation.md](efference_copy_cancellation.md),
[local_prediction_error_learning.md](local_prediction_error_learning.md),
[heterogeneous_graders.md](heterogeneous_graders.md) (§2 condition 1),
[activation_to_activation_forward.md](activation_to_activation_forward.md)
**Key experiments**: [`rhm/endogenous_teacher/`](../experiments/rhm/endogenous_teacher/) (both cuts here)
· [`a2a_forward/OOD_ROBUSTNESS_README.md`](../experiments/a2a_forward/OOD_ROBUSTNESS_README.md) (the
epistemic null this explains) · [`a2a_forward/CANCELLATION_README.md`](../experiments/a2a_forward/CANCELLATION_README.md)
· [`mjc/ballistic/`](../experiments/mjc/ballistic/README.md) 4b ·
[`one_layer_deeper/ballistic_depth/`](../experiments/one_layer_deeper/ballistic_depth/README.md)
**Attribution**: the depth≈time observation, the challenge *"what makes this temporal rather than
depth-wise?"*, and the NextLat correction are Jasper's. The information-set formalisation, the
lead→commitment→epistemic chain, and the three-way decomposition came out of the exchange.

---

## One-liner

Our forward self-model predicts `post_block0 → post_block6` — **same input, same forward pass**. Predictor
and target therefore have **identical information sets**, so the residual can only ever mean *"I lacked
the capacity to compute this"* and never *"I was wrong about something."* **Depth-wise self-prediction is
computational by construction, not by finding.** Epistemic content requires a **lead** — a forecast
committed before the evidence arrives — which exists only on an axis where the target's information set
strictly contains the predictor's. Separately and symmetrically: a forward path that cancels its forecast
still cannot express *"this changed nothing"* while the objective is a global scalar, because backprop
supplies gradient regardless of local activity. **The port lost the lead; the learning rule lost the null.**

## The two-line vocabulary this doc introduces

- **Lead**: the predictor's conditioning set strictly excludes information the target has. Not "executed
  earlier" — *conditioned on less*. Teacher forcing does **not** destroy a lead, because a lead is enforced
  by what the forecaster is given, not by execution order.
- **The null**: the state in which a learner *correctly does not update*. NTP has no null — every token
  carries gradient whether or not it changed you.

## 1. The information-set argument (this is the load-bearing one)

| axis | predictor sees | target sees | what the residual can mean |
|---|---|---|---|
| **depth** (`post_block0 → post_block6`) | `x_{≤t}` | `x_{≤t}` — *identical* | capacity shortfall only |
| **sequence** (`h_ℓ[t] → h_ℓ[t+1]`) | `x_{≤t}` | `x_{≤t+1}` — *strictly more* | capacity **+ epistemic** |

Along depth the two ends of the prediction are functions of *exactly the same input*. A forecaster with all
of the target's information cannot be *mistaken*; it can only fall short. So "the residual is computational,
not epistemic" was never an empirical discovery about self-models — **it is a theorem about the axis we
chose**, and it holds at any capacity, on any substrate, for any wiring.

This retro-explains the program's most stubborn null. [`OOD_ROBUSTNESS`](../experiments/a2a_forward/OOD_ROBUSTNESS_README.md):
*"Calibration transfer experiment showed zero epistemic self-knowledge: competence probes trained ID and
evaluated OOD are identical across all conditions, and output entropy beats every activation probe."* The
forward model never committed to anything it could be wrong about. There was nothing there to find.

**Why the substitution went unnoticed, and it is a good reason.** In wetware, separation in depth ≈
separation in time: cortical hierarchy is traversed by propagation, with conduction delays and recurrent
settling, and the cerebellar forecast *arrives before* the signal it predicts. The biology's "predict
cortical dynamics" is temporal-with-a-lead by default. In a transformer someone must *choose* which axis to
instantiate, and depth was the tractable one — block 6 is a legible "later" than block 0. But block 0 and
block 6 execute in the same instant. **The port kept the compression and silently dropped the anticipation.**

## 2. Why a lead produces epistemic content

A forecast issued before the evidence is a **commitment**, and only a commitment can be wrong *about the
world* rather than merely short *of the target*.

This is not a new variable. It is **commitment under delay** — [heterogeneous_graders](heterogeneous_graders.md)
§2 condition 1, flagged there as *"the only one that is **physically forced** rather than a choice about how
we happen to train: axonal delay and inertia are not electable."* And it is already priced: ballistic
transmits forward-model quality **3.0–3.3×** more than reactive ([`mjc/ballistic/`](../experiments/mjc/ballistic/README.md) 4b),
**4.85×** on-policy; transmission slope **+0.35 reactive vs +1.07 ballistic**
([`ballistic_depth`](../experiments/one_layer_deeper/ballistic_depth/README.md)), because *"reactive control
re-grounds each step and is therefore a near-blind grader of forward-model quality."*

**Teacher-forced NTP is maximally reactive** — ground truth at every position, re-grounded every step. So the
epistemic-vs-computational distinction *is* the ballistic-vs-reactive distinction, transposed onto the
reading axis. We have been running the near-blind regime and reporting its blindness as a property of
self-models.

## 3. NTP is arity-1 *and* exogenous-target

Write the transition `s_{t+1} = g(s_t, x_{t+1})`.

- **Arity-1** predicts `E[s_{t+1} | s_t]`, marginalising over the incoming word. Its residual bundles *"I
  didn't know which word was coming"* (the corpus's entropy) with *"I didn't know what it would do to me."*
- **Arity-2** predicts from `(s_t, x_{t+1})`. For a deterministic network that target is exactly determined,
  so the residual carries **zero** world-uncertainty — it is purely *"a compressed model of my own update
  rule couldn't compute it."* Arity-2 does not create the endogenous signal; it **conditions the exogenous
  one out**.

NTP is conditioned on `x_{≤t}` only (arity-1) *and* its target is the command itself rather than the
learner's state (exogenous). It differs from a temporal self-model on **both** axes at once.

> **Precision note, because an earlier version of this claim was too strong.** NTP's loss is *not* the same
> object as an arity-1 latent residual: one is a scalar proper scoring rule in token space, the other a
> direction in state space carrying an FM-capacity term. NTP is an arity-1 *objective*; its loss is the
> token-space marginal surprisal. The two are not interchangeable.

This also sharpens the reading phenomenology the prompt started from. *"Innate surprise"* is the arity-1
channel — and it is the **only** channel an LLM has ever been trained on. *"What ideas it provokes"* is the
conditional channel, and we have never built it.

## 4. The tension, and the decomposition that resolves it

Arity-2 says *condition on the word* (strip exogenous surprise); a lead says *commit before the word* (get
epistemic content). These pull opposite ways, so you need **both**, and the object of interest is their
difference:

- `p_ahead = FM(s_t)` — committed before `x_{t+1}`. Has the lead.
- `p_post  = FM(s_t, x_{t+1})` — issued after. Command-conditioned.

| term | meaning |
|---|---|
| `s_{t+1} − p_post` | **computational** — capacity shortfall, no epistemic content |
| **`p_post − p_ahead`** | **epistemic** — how the word revised my forecast of my own next state |
| `s_{t+1} − p_ahead` | the total |

The middle term is a Bayesian-update magnitude *in state space*: directional, endogenous, and exactly *"what
ideas the word provoked."* It does the thing NTP structurally cannot, in both directions:

- a **rare synonym** — high token surprisal, near-zero revision. NTP screams; you were not moved.
- a **quiet disambiguating word** — low token surprisal, large revision. NTP is silent; you were moved a lot.

**Felt surprise is how much a word changes your model, not how improbable it was.** Both forecasters are
fittable to a *frozen* reader, so the decomposition is a measurement requiring no change to the training
rule. That is the cheapest real test in this document.

## 5. The null needs a local learning rule, not only a wiring

Under [cancellation](efference_copy_cancellation.md) the tensor propagating downstream *is* the deviation
`e = s − p`. If the forecast is good, `e ≈ 0` — there is no activity downstream. Pair that with **local,
activity-driven plasticity** (Hebbian/STDP, `Δw ∝ pre × post`) and the update is zero *automatically*: "nothing
surprising happened" and "nothing propagated" become the same event. No gate, no threshold, no auxiliary
term — which is why biology needs none.

**Global backprop breaks this.** A global scalar objective injects gradient into every parameter regardless
of local activity, so no forward-path wiring can produce "this changed nothing." The null therefore needs
**both halves**, and the program has never had them in the same model:

| | propagating signal | learning rule | null? |
|---|---|---|---|
| local loss ([MNIST_LOCAL_LOSS](../experiments/a2a_forward/MNIST_LOCAL_LOSS_README.md)) | full activation | local (synthetic gradient) | no |
| cancellation ([CANCELLATION](../experiments/a2a_forward/CANCELLATION_README.md)) | the residual | global NTP | no |
| **both** | the residual | local | **unbuilt** |

**The null is graded, not global.** `e ≈ 0` everywhere would mean the model stops computing. Absorption
follows `res_var(i) ∝ act_var(i)^β` with β<1 — loudest directions absorbed at ρ≈0.95, quietest at ρ≈0.32
([residual_decomposition](../experiments/rhm/residual_decomposition/README.md)) — so the real statement is
*you stop learning in the directions you already had, and keep learning in the ones you don't.* Which is a
better description of re-reading a familiar page than an off-switch.

**Obligation vs opportunity.** Under arity-1 every token has nonzero surprisal, so every token teaches
whether or not it moved you: text is an obligation. Under a lead-bearing conditional self-model the residual
*can* be zero, and the system gains the ability **not** to learn. That is the structural difference between
being trained by text and consuming it — and it is why two readers of the same page are moved differently,
and why a re-read moves you less.

## 6. What was run, and why neither cut tested this

Both cuts are in [`rhm/endogenous_teacher/`](../experiments/rhm/endogenous_teacher/), single seed, v16 s2 L6.

**The one positive — Gate 0 (m4).** On a trained base, `R²(rres ~ nll)` = **0.113** pooled, 0.112
within-position: **89% of the residual's variance is orthogonal to token surprisal**, corr **−0.34**. The
level attribution (aligned sequences, ground-truth levels) shows they are **anti-localised**:

| position completes | n | token nll | rel. residual |
|---|---|---|---|
| level 1 (near root) | 1 | **2.609** | **0.269** |
| level 4 | 8 | 2.523 | 0.319 |
| level 6 (leaf-adjacent) | 32 | **0.802** | **0.436** |

Internal residual is *lowest* exactly where token surprisal is highest. The reading: **the residual measures
computational load, not epistemic difficulty** — where the model has nothing to go on its computation is flat
and easily compressed; where it composes confidently the FM cannot keep up. Consistent with §1 (the depth FM
already conditions on the token, so its residual is nearly clean of exogenous surprise — which is *evidence
for* the arity framing) and with `OOD_ROBUSTNESS`.

**Cut 1 — residual-weighted loss. Null, and over-determined.** Five arms with **bit-identical rank-normalised
weight multisets** (only the assignment differs), from a shared base: `uniform` Δval −0.0183 / Δd4 +0.174;
`res` −0.0155 / +0.160; `res_shuffled` −0.0147 / +0.161; `nll` −0.0000 / +0.155. **`res` ≈ `res_shuffled`** —
the pre-registered falsifier fired. Not an artefact of a vanished residual: rel-res *rose* 0.40→0.44 in every
arm. Two prior results predicted this and should have been read first: [EMOTION_INJECTION](../experiments/a2a_forward/EMOTION_INJECTION_README.md)
(*"'this will be hard' doesn't tell the model **what** to compute differently — only **how much**"*; gate 3.0
vs 0.15) and the strong-confidence node **"self-knowledge is directional, not scalar"** (vector probe ΔR²
+0.18 vs scalar +0.03; steering along the residual *norm* has zero effect). Collapsing a directional object
to a norm and using it as a multiplier tests nothing.

Two side findings worth keeping: **ordering by token surprisal is the worst assignment of all** (`nll` erases
the val gain entirely — on RHM the high-nll positions are the *irreducible* ones, so "learn more from
surprising text" upweights aleatoric noise); and `res` expanded participation ratio 5.02→**6.90** where
`res_shuffled` gave +0.36 — the only res-vs-shuffled separation, unreplicated, on an instrument this repo has
been burned by three times, and non-monotone across arms (`res_orth` is the *most* contractive at −1.05 despite
being a blend). Treat as unresolved.

**Cut 2 — cancellation on RHM (m2). Mechanism replicates; payoffs do not.** `cos(Δ,inj)` **+0.293 summation →
+0.088 cancellation** at matched injection norm (17.25 vs 17.22) — the ViT read +0.89 → **+0.095**, so the
corollary-discharge signature reproduces on a different architecture and task. But the fresh-FM swap (Payoff 1,
the follow-up [efference_copy_cancellation](efference_copy_cancellation.md) records as blocked on the ViT for
want of a gauge-free substrate) costs **+0.0024** summation vs **+0.0015** cancellation — against a dependency
of **0.80 nats** and a random-FM floor of 1.38–1.43. **Downstream depends on the forecast, not on this
forecaster, in both wirings**: the *entangled* dependency cancellation was proposed to fix does not appear
here. Fresh SK: `ntp` 0.448 → `cl_sum` 0.373 → `cl_cancel` 0.349, so cancellation is marginally *worse* —
prediction falsified. Injected val ties `ntp` exactly (0.8232/0.8236 vs 0.8232) with injected root ≥ `ntp`'s:
**pure computation relocation, reproduced on RHM.** Caveat: the three fresh FMs land at cos 0.871–0.872, and
we did not measure whether they are functionally distinct or merely gauge-distinct (`ens_cos` would settle it),
so the Payoff-1 null is one instrument-check short of airtight.

**Why neither cut tested the thesis.** Every arm of both cuts trained on `loss = NTP` — the arity-1 channel —
so the exogenous/endogenous variable was pinned at "fully exogenous" throughout. Cut 1 varied the *weighting*
around it, cut 2 varied the *wiring* around it. And both used the depth FM, which §1 says is epistemically
empty by construction. The self-knowledge instruments (fresh-FM SK, swap modularity, dependency) were imported
from the a2a program, whose dependent variable is *"does the model build a self-model"* — a different question
from *"does what the model learns depend on its own trajectory."*

## Predictions / falsification

- **Falsified if** a depth-wise forward model is shown to carry genuine epistemic content (calibration that
  transfers OOD, beating output entropy). §1 says this is impossible, so it is the cleanest kill.
- **Predicts** the epistemic term `p_post − p_ahead` dissociates from token surprisal in *both* directions on
  RHM — rare-synonym positions (high nll, low revision) and disambiguating positions (low nll, high revision).
  If it is a monotone function of nll, the decomposition adds nothing and this document is mostly wrong.
- **Predicts** two readers with different histories on identical text diverge far more in `p_post − p_ahead`
  than in NTP loss. This is the "same page, different experience" readout, and it is undetectable by any
  arity-1 objective.
- **Predicts** cancellation + a local rule produces a genuine null (zero update on predicted input) where
  cancellation + global NTP does not — and that the null is *graded by direction*, not global.
- **Prior against all of the above**: four independent replications say endogenous targets cap —
  `LL` (cos 0.997, SK≈0), `λ_local` (ΔSK −0.06 → −1.54), `data2vec` (+23% vs grounded mlm's +44%), `fm_cotrain`
  (d4 0.744→0.671, plannability 0.357→0.304, *"grounding is the pivot"*). **None was temporal-conditional**, so
  the cap does not transfer directly — but any design here must say *in advance* what would distinguish
  "capped like data2vec" from "genuinely different."

## Connections to beliefs / prior results

- **[heterogeneous_graders](heterogeneous_graders.md) §2 condition 1** — commitment under delay is the same
  variable as the lead. This doc claims the reading axis is a place that condition can be switched on, and
  that switching it on is what makes self-knowledge epistemic rather than computational.
- **[SLEEP_CHUNKING](../experiments/rhm/SLEEP_CHUNKING_RHM_README.md)** — *"only the target controls dilution …
  feeding a latent **in** does nothing if you predict tokens **out**"*, and *"we had the input right and the
  target wrong."* Same lesson, one axis over: we had the *modality* right and the *axis* wrong.
- **[efference_copy_cancellation](efference_copy_cancellation.md)** — supplies the forward-path half of the
  null. This doc adds that the wiring is insufficient without a local rule, and that Payoff 1's premise
  (entangled dependency) did not reproduce on a gauge-free substrate.
- **Surprise is not an allocator.** `surprise` pins to noise (occupancy 0.79→0.96, the noisy TV);
  LP recovers 18% of the allocation prize against an outcome-trained tap's 84%. The null is the structural
  gain here; deciding what to *do* with the non-null still needs the evaluative half.
- **[operators_not_footprints](../beliefs/trees/operators_not_footprints.md)** — a temporal FM compresses the
  *transition operator* `g(s, x)`, which is the most literal instance of the root claim yet.

## Open questions

- Is `p_post − p_ahead` genuinely separable, or does the non-Markov transition leak? `h_ℓ[t+1]` depends on the
  whole prefix, not just `(s_t, x_{t+1})`, so the arity-2 residual also contains *"what the prefix carried that
  `s_t` didn't."* Mitigations (later ℓ, a short window, a genuinely recurrent state) must be chosen **before**
  measuring, or prefix leakage reads as endogenous surprise.
- Does a lead require streaming at *inference*? For the measurement, conditioning suffices — teacher forcing is
  fine. For an intervention that *uses* the forecast causally, the forecast must be available before the word
  is processed, which is a real architectural constraint (recurrent/streaming).
- Is there any local-plasticity rule compatible with a transformer that would give the null without giving up
  the task? The synthetic-gradient local loss is the nearest existing thing and it produced brittleness.
- Does the epistemic term have the band-pass shape an expansion grader needs, or is it monotone in revision
  magnitude (hence another noisy-TV drive)?
- **NextLat[^private] is not this** (Jasper, who has read it; an earlier draft of this doc
  claimed otherwise from an abstract). It has no forward-modelling component — predictability as a
  representation-shaping regulariser, not a separable forecaster whose residual is a signal. That is the
  map-vs-model distinction from [self_model_needs_a_loop](self_model_needs_a_loop.md), and the difference is
  the whole point here. Still worth notes in `reading/`; `cl-jepa-compacted.pdf` likewise.

## Context pointers for a future agent

1. §1 is the load-bearing claim and it is an argument, not a measurement — read it first and try to break it.
2. [`rhm/endogenous_teacher/PREREGISTRATION.md`](../experiments/rhm/endogenous_teacher/PREREGISTRATION.md) and
   its `cancellation/FILES.md` record both cuts, including the design traps (rank-normalised weights; the FM's
   target under cancellation must be the **restored** read-out `post_block{k}_eff`, never the deviation stream).
3. `rhm/model.py::GPT.forward` now carries `cerebellar_mode` / `cerebellar_readd_block`, defaulting to the
   historical summation path — every prior RHM result re-runs bit-identically.
4. The cheapest next thing is §4's decomposition on a **frozen** reader: two forward models, no training-rule
   change, RHM ground truth for level attribution. It is a measurement, so it cannot be confounded by a wiring
   choice — which is the failure mode of both cuts recorded above.

[^private]: Not mirrored: this link points to a document in the private lab repo (the roadmap, the queue, an unrun spec, reading notes, or a conversation). See the top-level README for what is held back and why.
