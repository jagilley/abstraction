# HISTORY — experiments/mjc

A qualitative "big history" of this node, read from its READMEs rather than its commit log (squashed/
imported, little chronological signal of its own). Auto-generatable; overwrite freely. See
[README.md](README.md) for current status, [FILES.md](FILES.md) for the full file index.

## Why this substrate exists

`mjc` fills a gap in the ladder between the RHM toy DGP and language: a rung where "the command
genuinely controls the state" is *physically* true rather than constructed, with real continuous
dynamics and contact — the property language structurally denies
([ideas/physical_control_substrate.md](../../ideas/physical_control_substrate.md)). It inherited RHM's
discipline explicitly: "a controllable *dynamics* DGP whose shift we dial and sweep, **not** a robotics
benchmark to RL a policy to SOTA on" — every experiment a knob sweep with a vanilla baseline, never a
leaderboard number. Biology (cerebellar feedforward control, efference copy, corollary discharge) is
used "as compass, not costume" — every hypothesis tested here is, in origin, about motor control of a
body, licensing the recurring afferent/efferent, ballistic/reactive vocabulary throughout.

## Foundations: three cuts that established the substrate's vocabulary

**Cut #1** ([`contact_residual/`](contact_residual/README.md)) ported a2a's "residual = surprise" claim
to physics and sharpened it: what looked like "contact is a hard regime" turned out, under an
onset-aligned control, to be "the residual marks a regime *transition*" — an event detector, not a
state one. A quieter lesson traveled forward too: MSE vs Huber isn't just an optimizer choice but an
epistemic one (heavy-tailed contact error can starve what "the predictable dynamics" means at all).

**Cut #2** ([`arity_torque/`](arity_torque/README.md)) took the arity claim to a real actuator, killing
the "commands correlate with state" confound with i.i.d. commands. Arity beat resolution outright — no
capacity substitutes for the missing command slot. This premise (clean i.i.d. commands) is exactly what
on-policy later reveals as a teleport-only luxury.

**Cut #3** ([`dynamics_shift/`](dynamics_shift/README.md)) is the pivot from describing dynamics to
*intervening* on them, and became "the substrate most later nodes build on." Its load-bearing fact is a
near-null: under per-step replanning, feedback substitutes for the model and the effect vanishes; only
when control *commits* to open-loop segments (`replan_every`) does the forward model become
load-bearing, peaking near a ~6–8-step composition horizon. That knob is read biologically as *how
ballistic a movement is* — fast movements outrun sensory feedback, so a stale model causes dysmetria and
reward-free refitting is cerebellar recalibration. That reading seeded the [`ballistic/`]
(ballistic/README.md) arc.

## The substrate audits itself: teleportation and the on-policy correction

A retracted directed-collection claim in `ballistic/directed/` (a where-to-collect result decided by
2,240 free teleported probe transitions against a spending budget of 100 — a 22× subsidy) forced a
standing audit, [`on_policy/COLLECTION_REALISM.md`](on_policy/COLLECTION_REALISM.md) — a clean instance
of "update the prior about the *metric*, not the finding": every dissociation/ratio in the node is
Tier-A safe (both arms share the same coverage advantage, which cancels); every *absolute sample count*
("~50 transitions") is Tier-B, an upper bound not an estimate; anything whose dependent variable is the
allocation of experience itself is Tier-C, unaskable under teleport.

[`on_policy/`](on_policy/README.md) then priced the fix as a `collection_mode` flag (bit-identical
teleport path preserved, gated) and corrected its own live hypotheses three times: **E0** found most of
on-policy's apparent advantage was a mistuned teleport knob, not embodiment — what *is* structurally
embodied is command-state entanglement, a byproduct of continuity itself. **E1** found the memo's own
prediction (on-policy will "stretch" a step-like recovery into a gradual one) backwards — the 5×
ballistic dissociation reproduces under every mode. **E2** found the real fork: **global** drift is
nearly embodiment-free; **local** drift costs 2.5× more transitions and broad teleport never repairs
the region, because on-policy data quality is bootstrapped — a still-wrong model deflects its own
reaches, so the data it gathers is wrong-distributed too. That reopened "where should I practise?" as a
real question, which `on_policy/directed_on_policy/` and `on_policy/metered_repair/` answer with a
metered monitor cost — a scarcity-*and*-signal conjunction, not either alone, is necessary.

[`arm_substrate/`](arm_substrate/README.md) is the sibling correction on the plant side, built to
retire the `ballistic/` arc's single-family caveat and replace the pusher's manufactured capacity
competition with competition intrinsic to the body. Its sharpest finding refines Cut #4d's capacity
story into two independent axes: capacity binds intrinsically only at n≥5 links (a 2-link arm's inertia
matrix depends on one angle) — "nonlinear" and "capacity-hungry" are separate plant properties, not
one. A passive tool reproduces #4d's boundary as a literal mass in kilograms rather than a hand-tuned
field, and a `goal_site` flip runs Cut #3's operator-intervention trick in the opposite direction,
holding physics byte-identical while flipping value-relevance — something the pusher structurally
could not do.

## The value↔forward-model interface arc

[`value_shaping/`](value_shaping/README.md) established the stationary control point: value
re-allocates a capacity-limited FM away from a reducible-but-irrelevant factor, with a physically
irreducible foil (contact) as the honest floor. It deliberately declined to chase its own follow-ups
because [`directed_readapt/`](directed_readapt/README.md) had, by then, closed out a mode of working it
calls "a-priori mechanistic-atom verification," where isolating one hypothesized mechanism kept either
killing the phenomenon (engineered scarcity broke the disagreement drive, which flags *uncertainty*,
not *confident wrongness*) or leaving the baseline already good enough to need no mechanism (a global
shift has no scarcity to exploit). The lesson — "which drive wins is set by the environment, not
absolute" — is a real pivot: from verifying named mechanisms toward getting the phenomenon working
first and reading mechanism off what worked.

[`meta_adapt/`](meta_adapt/README.md)'s five-cut arc (#4→#4e) is that pivot's main product. #4 found the
naive meta-vs-multitask floor collapses on a 1-D shift (no task conflict, nothing to buy) and only
opens under a genuinely conflicting actuator-rotation axis. #4b replaced implicit weight-memory with an
explicit context latent and produced the arc's cleanest dissociation: a model can decode the task
parameter perfectly and still get zero benefit from knowing it — "true but useless" recurs as a phrase
through the rest of the node. #4c is a robust negative (VoI-directed system-ID navigates to the
informative region but doesn't identify faster; the largest command already is the most informative
one). #4d lands the causal boundary condition: value-caused re-allocation only converts to control
benefit when capacity actually binds on the value-relevant dynamics. #4e closes the loop by letting
reward alone rediscover that allocation, with a wireheading gotcha: an unconstrained weight budget let
the outer loop win by gaming Adam's loss scale, fixed only by normalizing to a closed budget.

[`curiosity_control/`](curiosity_control/README.md) is the afferent mirror of that story — value
directing *where to look* rather than *what to model*. Its most consequential finding inverts a prior
from the a2a active-vision line: disagreement was believed to reject a noisy TV by construction
(members converge on aleatoric noise). On low-dimensional, data-scarce control the opposite happens —
members fit *different* noise realizations and disagree, so a pure epistemic drive chases noise instead
of the real frontier. This retires "a smarter filter is the fix" as a general claim; the fix is
architectural — additively "grounding" exploration with an exploit/value-relevance term, which helps
*most* exactly where pure curiosity fails. A later re-analysis,
[`curiosity_control/benchmark_vs_cost/`](curiosity_control/benchmark_vs_cost/README.md) splits the
signal into a *gain* role (how hard to update) and an *allocation* role (where to spend).

[`online_value_loop/`](online_value_loop/README.md) then tried to build the idea doc's named keystone —
one live loop where reward continuously shapes a *live* forward model — and failed on both levers,
which the node treats as confirmation rather than setback. The afferent lever's tracking effect on the
FM was real but invisible to CEM-MPC control, which re-grounds past whatever difference the drive
produces; the efferent lever's re-allocation benefit was real but washed out at convergence, invisible
to any probe short of a full retrain. The reframe that survives: value is a **slow, committed,
offline-estimable** quantity, so the two-timescale split is forced by value's own structure, and its
payoff is adaptation *speed*, not converged competence — which only matters if the world never stops
drifting.

[`drift_value_loop/`](drift_value_loop/README.md) took that seriously and ran three cuts under
perpetual drift. Cut 1 corrected an attribution: compounding across a drift sequence is real, but comes
from the fast *memory* layer (a context-conditioned latent), not value-carving — pooled memory is
actively worse than starting from scratch under conflict. Cut 2 confirmed the carving mechanism works
yet is capacity-gated and control-robust, invisible to the near-blind grader. Cut 3 changed one thing —
the outer loop's teacher, from control reward to the value-relevant FM prediction error itself
(explicitly read as the cerebellum→VTA messenger) — and got a clean interior optimum where control had
been flat throughout. **The missing piece across three cuts was the teacher, not the value structure.**

[`ballistic/`](ballistic/README.md) resolves the recurring "control is a blind grader" episode as a
fact about the *controller*, not the value system: replanning is blind because it re-grounds before a
model difference can matter. A ballistic (feedforward, committed) controller makes FM quality several
times more load-bearing than a reactive one and restores competence after drift far more efficiently
from reward-free re-adaptation alone — the same forward model is both the asset maintainable *without*
reward and the asset feedforward control *needs*. Its child,
[`ballistic/directed/`](ballistic/directed/README.md) is where the retracted directed-collection
question above gets its honest audit, and later its on-policy resolution.

## Cut #5: dimensionality expansion, run and retired

`physical_control_substrate.md` had pitched drift as "a novelty generator" — physics hands an agent
perpetual non-stationarity for free, so a model tracking drift should have to keep growing its
representable directions. [`expansion/`](expansion/README.md) tested this directly and closed it: drift
moves the target function; it does not open the space of representable functions (±0.08 directions of
movement against a +0.72±0.42 calibration from a genuine, unrelated support increase). A trustworthy
null took three failed rank-shaped instruments across the repo (language, this node's own Cut #1, RHM)
before the fix — flank the null with two working calibrations (a known-wrong-FM separator, a known-real
support increase) so a flat line can't be mistaken for a blind readout. Only drift's *special standing
w.r.t. representational growth* is retired — its realism value, and the program depending on it
(`drift_value_loop/`, `on_policy/`), is untouched. A slope-shaped readout (β) is abandoned as unusable
in a state space this small.

## The performance-error bridge δ: one idea, five corrections

[`ideas/performance_error_is_the_bridge.md`](../../ideas/performance_error_is_the_bridge.md) proposed
one biologically-motivated signal, δ = (benchmark − error) × a gated arity term, serving as both a gain
on plasticity and a credit-assignment channel separate from reward. Four experiments later the idea's
*core claim* is intact but almost every piece of its literal form was corrected empirically, each
correction coming from a control rather than an inspection:

- [`agency_gate/`](agency_gate/README.md): the gate genuinely carries self/other discrimination (δ
  fires ~369× stronger self-produced vs. identical passive replay) — but the literal formula leaks by
  construction (σ(0)=0.5) and needed centering.
- [`plasticity_gain/`](plasticity_gain/README.md): the corrected δ, consumed as a per-sample plasticity
  gain, beats fixed lr on re-adaptation *speed* — but a raw-error gain matches it on control, so δ's
  real content is allocation/retention, and the literal scalar-EWMA benchmark turned out mechanically
  pathological (it must be context-conditional).
- [`two_clocks/`](two_clocks/README.md): confirmed the idea doc's bet in a stronger form than asked —
  performance error and reward are non-redundant, architecturally-separate channels: every attempt to
  *share* their eligibility window was worse than either alone.
- [`bridge_assembly/`](bridge_assembly/README.md): ran the fully corrected δ live and delivered the
  arc's biggest surprise — plain uniform plasticity beats δ outright here, because δ pays a
  benchmark-lag tax exactly where recovery is still happening. Reconciled with `plasticity_gain`'s
  opposite-looking result, this yields the transferable law — **which controller sees a given
  forward-model difference depends on where that difference's mass lives**: localized error is
  invisible to a reactive controller but plain to a ballistic one, and diffuse error is the reverse.

## The most recent turn: practice as a third component

The [`practice/`](practice/README.md) arc's most recent addition, [`practice/etude/`]
(practice/etude/README.md), moves past consuming δ as a gain and tests re-chunking/compilation — the
practice loop's third piece — on a task with genuine *sequence* structure for the first time. Its
headline, reached after two corrections to the compile operation (regression-to-the-mean destroys valid
command sequences; naive selection suffers a winner's curse and a measured 4.5σ audition/use mismatch),
is that sequential assembly with seam-matched selection beats never-compiling on *both* accuracy and
time, 3/3 seeds — not a trade-off. A structural aside worth carrying forward: a committed unit here is
provably state-independent, so it can never degrade and fusing units is a bookkeeping no-op — hierarchy
needs state-conditioned commitment or a re-grounding budget, named as the next design decision.

## Threads that recur across the whole node

A few things keep resurfacing under different names: **capacity competition as a universal boundary
condition** — value-shaping, meta-conditioning, exploration grounding, and the arm's passive tool all
gate on whether something competes for limited capacity on the value-relevant dynamics; **control as a
near-blind grader** — recurring across `online_value_loop` and `drift_value_loop`, resolved only once,
in `ballistic/`, as a fact about commitment mode rather than value; and **realism corrections that
vindicate rather than undermine prior findings** — both the on-policy audit and the
dimensionality-expansion null left dissociative claims standing while correcting only what they were
denominated in. One unresolved tension: `heterogeneous_graders.md` §2 condition 4
([../../ideas/heterogeneous_graders.md](../../ideas/heterogeneous_graders.md)) argues a second (meta)
loop is *provably* unnecessary on a stationary DGP, while `two_timescale_value_loop.md`'s amendment
([../../ideas/two_timescale_value_loop.md](../../ideas/two_timescale_value_loop.md)) argues the real
gate is grader type, not stationarity — not yet reconciled directly here.
