# SPEC — question_model: typing a token's news, and a learner that knows what's being asked

**Status**: spec, unbuilt (2026-08-16). Written by the session that ran the `transpose`×`setlist`
crossed pair; discussed with Jasper same day.
**Up**: [`../README.md`](../README.md) (rhm) · direct parent:
[`../conditional_revision/`](../conditional_revision/README.md) (whose cached substrate this reuses)
· machinery donor: [`../practice/setlist/FILES.md`](../practice/setlist/FILES.md) (OU demand drift)
**Naming note**: "demand" is the practice arc's word for this object; Jasper's preferred rotation is
**questions vs answers** — truth-news updates your answers, demand-news updates your questions, and
this experiment builds and tests a *question model*. Use whichever vocabulary keeps you consistent
with the machinery you're importing; they denote the same distribution.

## Why this experiment

Two lines of this program converged on one invariant from opposite sides, and this experiment is
their fusion test.

From the practice side: the `transpose`×`setlist` crossed pair
([`../practice/typed_gaps/README.md`](../practice/typed_gaps/README.md), idea-doc ledger
[`practice_manufactures_its_own_credit`](../../../ideas/practice_manufactures_its_own_credit.md)
§18–§19) measured a double dissociation between two currencies of world-change: news about *what is
true* lands on the dense learner and cannot touch committed chunks (which store
demand-concentration, not truth), while news about *what is asked* is the one currency the
evaluative re-selection machinery consumes. One organ per currency.

From the self-model side: [`revision_not_surprisal`](../../../ideas/revision_not_surprisal.md)
established (Gate 0: axis flip, `corr(r_res, nll)` −0.337 → +0.652; Gate B: the surprisal-matched
contrast) that a forward model's residual carries world-news only across an exogenous conditioning
gap, and decomposed the arity-1 residual into compression + revision. Its §4 identity splits a
token's surprisal into structure-news and an aleatoric residue — and NTP, lacking an *aleatoric
null*, pays gradient on the irreducible part forever.

The synthesis this spec operationalizes: the §4 identity is missing a third term. A token carries
(i) news about *this sequence's* latent structure, (ii) news about *the generative mixture itself* —
which rules and roots the world is currently drawing from, i.e. what is being asked — and (iii) an
irreducible realization residue. Channel (ii) is precisely the object `setlist` drifts and the
practice arc calls demand. The question with program-level stakes: **is epistemic self-knowledge
("will I be right") carried by channel (ii)?** Every truth-side probe of it has nulled
([`OOD_ROBUSTNESS`](../../../experiments/a2a_forward/OOD_ROBUSTNESS_README.md) Exp A;
[`two_timescale_value_loop`](../../../ideas/two_timescale_value_loop.md) §2 gives the type-theoretic
reading: calibration is a value function). The typed-gap map predicts it is demand-denominated — the
missing organ, not a deeper read of the truth-organ. The deeper question behind that one, which
gives the experiment its stakes either way: *which parts of self-knowledge are self-manufacturable?*
The practice arc proved level-news is not (recital/tall); this asks whether the epistemic half is.

## Reading list (ordered; the first three are load-bearing)

1. [`revision_not_surprisal.md`](../../../ideas/revision_not_surprisal.md) — the identity, the
   gates already run, the cached substrate, and §5/§8's *standing prior against scalar gating*.
2. [`../practice/typed_gaps/README.md`](../../practice/typed_gaps/README.md) — the crossed pair and
   the currency taxonomy. (Also beliefs:
   [`cerebellum_and_cognitive_architecture`](../../../beliefs/trees/cerebellum_and_cognitive_architecture.md),
   the two 2026-08-16 nodes.)
3. [`../practice/setlist/FILES.md`](../../practice/setlist/FILES.md) — the OU demand machinery you
   will import (`demand.py`: OU form, `typical_demand` median-of-48 prewarm, the σ-vs-κ lesson:
   sweep, don't solve), and its calibration discipline.
4. [`../conditional_revision/README.md`](../conditional_revision/README.md) + its `SPEC.md` — the
   belief-space revision machinery, Gate B's surprisal-matching protocol, and the cached base
   checkpoint + FMs at `/data/v16_s2_L6_m4_distinct/conditional_revision/` (chromatic).
5. [`meta_learning_under_metered_data.md`](../../../ideas/meta_learning_under_metered_data.md) §8
   (relevance vs reducibility taps, 84% vs 18%) and
   [`heterogeneous_graders.md`](../../../ideas/heterogeneous_graders.md) §4/§4b — why an
   outcome-referencing tap is a different *kind* of signal.
6. The scalar-gating graveyard, which Half 2's loss-side arm walks toward with open eyes:
   [`endogenous_teacher`](../endogenous_teacher/README.md),
   [`conditional_revision/sculpt_slip`](../conditional_revision/sculpt_slip/README.md),
   `EMOTION_INJECTION`, and `full_loop` §5's "grounding is the pivot" + internalization ladder.

## Substrate

RHM language modeling (NTP), not sculpting — so `tall`'s m-inadmissibility finding does not apply;
you *want* m ≥ 4 for a rich aleatoric channel. The natural default is the cached
`conditional_revision` substrate (v16, s2, L6, m4, distinct rules; base checkpoint + FMs cached).
The one new world ingredient: an **OU drift on the generative mixture** — root prior and rule
mixture weights — adapted from `setlist/demand.py`, calibrated in nats of demand-KL per event, with
the `typical_demand` prewarm. Note the load-bearing role of drift here: under a *static* mixture the
demand channel decays to zero once any consistent estimator converges, so drift is what keeps
channel (ii) alive at a knowable rate. Magnitude admissibility should be established by measurement
before the main runs (the arc's convention: separation-preserving, with the no-drift control in-run).

## Half 1 — the instrument: exact three-way decomposition of a token's news

For latents `z` (this sequence's parse/ancestors) and mixture parameters `θ` (the OU state), the
target decomposition of a token's information is:

- **Structure-revision** — movement of the posterior over `z` induced by the token, computed by BP
  with known tables and known current `θ` (this is `revision_not_surprisal`'s `B_t`, per level ℓ).
- **Demand-revision** — movement of the posterior over `θ` induced by the token, from an exact (or
  tightly-controlled) Bayesian filter over the mixture family. The experimenter holds the true OU
  trajectory, so this channel has a *ground-truth oracle* the way nothing in the original doc did.
- **Aleatoric residue** — the realization entropy given both: the part of surprisal that is
  synonym-choice, no news at all.

Making the identity exact (in expectation, per level, with `θ` uncertainty) is part of the build —
the §4 identity extended by one term. Deliverables:

1. The per-token decomposition across corpora with drift on and off, against the oracles; sanity
   identities (channels sum to surprisal in expectation; demand channel → 0 under static `θ`;
   demand-revision concentrates at tokens that disambiguate *which feature is asked*, echoing
   `setlist`'s demand-levels lesson that synonym-level mixtures carry no demand).
2. **The laundering question**: does a plain NTP model trained under drift maintain any internal
   estimate of `θ`? Probe activations for the OU state (level-resolved: root prior vs rule
   mixtures), measure its lag and accuracy against the exact filter, and against a model trained on
   the static mixture. "Launder" would look like: token statistics adapt but no low-dimensional
   `θ`-estimate is recoverable, or it is recoverable only at the timescale of weight updates rather
   than in-context. Either answer is informative; don't assume which you'll get.
3. Relate the model's own revision `Δ_t` (conditional_revision machinery, frozen model) to the three
   oracle channels — which channel does the model's registered news actually track? Gate B's
   surprisal-matched protocol applies; reuse it rather than reinventing the controls.

Half 1 is cheap (CPU-heavy oracles + existing checkpoints + probes) and gates Half 2: the demand
channel must be measurable and non-degenerate before a tap is built to consume it.

## Half 2 — the twin learner: does a question model buy the epistemic half?

Train matched learners under the drifting mixture; the axis is whether the learner carries an
explicit slow question-model. Candidate tap implementations, held loosely (pick by what Half 1
teaches; the list is not exhaustive):

- **Conditioning tap**: a slow estimator of `θ` (exact filter as oracle ceiling; a learned/EMA
  estimator as the endogenous form) supplied as input (a demand embedding). Guaranteed to remove
  the `θ`-uncertainty component of loss; the interesting question is what else it buys.
- **Loss-side tap (the aleatoric null proper)**: down-weight gradient on the oracle-labeled
  irreducible component. This is scalar-gating-shaped, and the repo has measured scalar gating weak
  or null four times — but note why those nulls may not bind here: `sculpt_slip`'s prize was ~0.03
  by construction, whereas at m=4 the synonym entropy is a large fraction of total loss, and the
  label here is exact rather than estimated. Run oracle-first, then the internalization ladder
  (`full_loop`'s discipline: oracle → learned estimator → withdrawn), so if it nulls you know at
  which rung.

Readouts, in priority order:

1. **Epistemic self-knowledge under demand shift.** Competence/correctness heads (predicting own
   per-token or per-family success) trained in one demand epoch, evaluated after drift — the
   substrate-native analog of `OOD_ROBUSTNESS` Exp A, with its measured null as the baseline shape.
   The bracket from `setlist` transposes: a **frozen** ledger vs a **maintained** (recency-weighted,
   periodically re-audited) one, so "what does tracking the question distribution buy" is priced.
   Activation-side probes run alongside as the truth-organ comparison arm.
2. **Gradient economics.** With exact per-position aleatoric labels, measure what fraction of
   gradient norm each learner spends on irreducible positions across training, and matched-token
   learning curves on the structural channels. Two claims to keep separate: the conditioning tap's
   guaranteed removal of `θ`-uncertainty loss, and the aleatoric null's hoped-for savings — do not
   let the first masquerade as the second.
3. **The internal-estimate contrast**: the plain twin's implicit `θ`-estimate (Half 1's probe) vs
   the explicit tap — lag, accuracy, and what each costs.

What the typed-gap map *predicts* (recorded as the theory's predictions, not as constraints on
interpretation, per the repo's norms — no kill criteria, no pre-registered outcome space): the
maintained demand-conditioned ledger transfers calibration where activation probes null, the frozen
ledger decays toward them under drift, and the effect appears only under drift. If the results go
elsewhere, they go elsewhere — a demand-typed organ failing to buy the epistemic half would place
epistemic self-knowledge in the structurally-external column beside level-news, which reorganizes
the teacher theory and is arguably the more interesting outcome. Bring the numbers back for
discussion before writing any README (repo norm).

## Practicalities

- Invoke `/run-experiment-on-modal` before running anything; profile `chromatic`; reuse the cached
  substrate rather than retraining where possible. Smoke-test before long runs; follow the halting
  procedure for anything >5 min.
- Single seed first, per `experiments/CLAUDE.md`; no multi-seeding without checking in. Rank
  orderings over fractions where noise is uncalibrated; if you need a noise floor, measure one (the
  practice arc's stream-position convention is the model).
- Fidelity discipline: drift-off must reproduce the static baseline exactly; every drift knob set by
  a measurement, recorded in a calibration record (see `setlist/FILES.md` for the form).
- Structure: this folder gets its own `README.md` (post-discussion) and `FILES.md` per
  `experiments/CLAUDE.md`; keep `../conditional_revision/` and `../practice/setlist/` imported, not
  copied, and unmodified.
- Out of scope for this round: porting the tap to the a2a language loop (the natural follow-on if
  Half 2 lands), any practice-arc machinery beyond `demand.py`, and combined truth+demand drift.
