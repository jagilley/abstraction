# Absorption blinds the evaluator

**Status**: idea, first drafted 2026-08-31 from the `tuning/` Gate 1 discussion (Jasper + session
`014N9Exd…`; the edge-relative framing below incorporates Jasper's overloading caution from that
exchange and awaits his edit). **Evidence base**:
[`experiments/rhm/practice/tuning/`](../experiments/rhm/practice/tuning/SPEC.md) Gates 0–1 + the
Gate 1G checkpoint analytics (`figures/g1g/`). **Related**:
[`two_timescale_value_loop`](two_timescale_value_loop.md) (see §2's terminology fence) ·
[`performance_error_is_the_bridge`](performance_error_is_the_bridge.md) ·
[`revision_not_surprisal`](revision_not_surprisal.md) ·
[`practice_manufactures_its_own_credit`](practice_manufactures_its_own_credit.md) §18 · the
Garcia-Garcia et al. granule-cell preprint
(`reading/2026.03.03.709240v1.full.pdf`[^private], bioRxiv
2026.03.03.709240, relayed via a parallel-session note 2026-08-30).

## §1 The claim

Take any pair where an **adapter** (a system that changes itself to fit its stream) sits upstream
of an **evaluator** (a system that reads signals off that adapter or its stream to make
decisions). If the evaluator's event-evidence lives in the adapter's input-prediction residuals,
that evidence has a half-life equal to the adapter's absorption time-constant — and that
time-constant **shrinks with practice**. An event is evaluator-visible in proportion to the
adapter's *failure* to absorb it: transient-and-learnable news erases its own announcement;
sustained or irreducible conditions stay loud.

Corollary: the observables that survive the adapter's competence are exactly two — **the
adaptation itself** (weight motion; a self-model's residual reads it) and **standing
counterfactual state** (capability read on fixed shadow evaluations, a condition of the current
weights rather than a transient of the stream). An evaluator that needs to keep seeing structural
events must be plumbed to one of these; the stream ledger is a wasting asset. The input axes get
re-cheapened by the learner faster than an instant gauge can act, so the axes of the learner's
own cognition are not merely an alternative basis — they are the only basis where the evidence
survives. The self-model's role is not typing at the instant; it is being the observer whose
object is the adaptation itself, and therefore the only one adaptation cannot blind.

## §2 Terminology fence (inner/outer are roles on an edge, not systems)

[`two_timescale_value_loop`](two_timescale_value_loop.md) uses inner = world model (cortex AND
cerebellum) and outer = value system (BG/VTA). This doc's claim is **not** about those two slots:
"inner/outer" here name the two *roles on one adapter→evaluator edge*, and the roles are relative
and recursive — an organ can be the adapter on one edge and the evaluator's instrumentation on
another. The doc's original pairing is one instance (BG/VTA evaluating over the world model's
stream). The cerebellum's dual character (Jasper's cerebellum→VTA speculation) stops being a
confusion under this reading: on its cortex-facing edge it is an absorber; on its VTA-facing edge
it is a sensor serving the value system. Same organ, two edges, two roles. Claims should not be
ported between edges without checking which edge they were measured on: everything in §3 was
measured on a two-node edge where the adapter is the content reader's own gradient plasticity and
the evaluator is a thermostat-grade op rule.

## §3 Evidence (tuning, single seed, one world; floors as measured there)

- **The wall, directly.** The reader absorbs 86% of a +0.65-nat rotation shock in 50 steps, and
  the absorption accelerates across events (fourwall: 250 → 100 → 50 steps). Online event
  signatures read at +50…+125-step lag land at or under in-tag floors that run 2–4× the static
  calibration's (an acting reader in a moving world has no quiet checkpoints).
- **Learnability orders visibility.** Drift — the most learnable event, correct op `track` — is
  nearly invisible online even at zero lag (2 of 3 under floor); the same drift reads +0.559 at
  7–20× floors for seven consecutive checkpoints on an arm whose charged budget had expired 134
  steps earlier (the frozen-reader control: instrument fine, evidence absorbed). Bursts — a
  sustained irreducible *condition*, not a transient — typed 3/3.
- **Supplied ops are outrun, not wrong.** The re-key search declined at all six rotations:
  at +50 steps, restoring old addresses is worse than the re-map already underway; by rotation 6
  the online rotation signature is −0.004. The identity return at 14000 pays full price (+0.683)
  because nothing was mothballed (fourwall finding 2, reproduced).
- **The acceleration mechanism is stabilization, not class-learning** (Gate 1G). No map-keyed
  rotation organ exists: rotations 8000 steps apart applying the *identical* permutation share
  cos −0.06 at the embedding; alignment decays monotonically with separation regardless of map.
  Instead, rotation absorption becomes *purer re-indexing* over a stabilizing embedding table
  (re-index recovery 0.599 → 0.874, monotone; the address-matched sharp test recovers RSA
  0.60–0.88 against a naive 0.11–0.43). The native mechanism converges toward *being* the re-key
  op, inside one window — the timing baseline no lagged supplied op beat.
- **Irreducible conditions get a dedicated reflex.** Burst×burst weight alignment is ln_f
  cos 0.957, non-decaying across the run (floor 0.305): a fixed "back off the span" gain
  direction, carved once, reused verbatim — a native soft-skip. The three event types occupy
  three disjoint weight subspaces (rotation: wte 4.1×; burst: ln_f 4.3×; drift: h1–h2 1.8×;
  cross-type cosines at floor).
- **The two surviving ledgers, in the same runs.** Weight ledger: the FM residual types by sign
  (rotation raises `e_online` +25–39% and recovers; a consumed burst lowers it −18…−30%) — the
  absorption transient viewed from the side that absorption *creates*. Standing counterfactual
  ledger: `teacher_slot`'s successful loops read pathway/counterfactual excess as a condition and
  committed pre-rotation, with no event at all.
- **What event-typing is still for.** Its measured Gate-1 value was almost entirely the veto:
  refusing to learn from bursts (the surprisal-only arm merged on one at +12.5× floor) and
  correctly not-acting on drift. Typing guards the irreversible ops; choosing is the standing
  gauges' job. Merge also zeroes the typing triple itself (`basis_sep` exactly 0 on every merged
  arm) — the op a typer most consequentially fires is the one that blinds it.

## §4 Consequences

1. A fixed-cadence evaluator gets progressively blinder on every event class the adapter
   masters; the stable design reads observables the adapter's learning *produces* (weight
   motion / self-model residual) or *preserves* (standing counterfactual state), not ones it
   consumes (stream surprise).
2. The meter's type must match the op it grades: a reader-movement statistic cannot confirm the
   op that suppresses reader movement (Gate 1's fail-open FM grader undid nine correctly-typed
   skips in ten; the coding fix moved the outcome 0.05× floor). Skips need input-side statistics;
   movement-permitting ops can be graded by movement.
3. Predicted instrument, unbuilt: **weight-ledger online typing** — per-layer-group update-norm
   ratios (wte vs ln_f vs early blocks), free at train time from gradients the learner already
   computes, no counterfactual passes. Gate 1G's disjoint-subspace result says the three-corner
   separation exists in that ledger.
4. The wall is dynamic: windows close as practice accelerates absorption, so evaluator designs
   validated early in training silently expire later.

## §5 Biology reading (speculative, flagged as such)

The value system's textbook observable already lives with this wall: TD/RPE-style signals are
surprise-flavored evidence *designed* to self-erase as the world model learns — phasic dopamine
migrating to the earliest predictor is the stream ledger draining. Cerebellar adaptation masks
perturbations from slower systems (visuomotor and saccadic adaptation proceed without awareness):
what reaches valuation is the residual of absorption. A cerebellum→VTA projection is then the
anatomical version of plumbing the value system to the adaptation ledger — and Gate 1's
FM-as-grader is that circuit in silico. Caveat: in our system the absorber and the reporter were
different objects (the reader's gradients absorbed; the FM only observed). The biological circuit
is three-node, with one organ absorbing *and* reporting; whether reporting fidelity survives when
the same organ compensates is untested (Gate 3-adjacent).

## §6 Open, testable

- The weight-ledger online typer (§4.3) — cheapest next instrument on the existing substrate.
- An FM with a compensatory channel: does the reporter stay honest once it is also the absorber?
- Geometry reads (RSA/Procrustes) as the evaluator's channel where movement magnitude is
  absorption-limited — Gate 1G shows the shape information exists (reorient vs rebuild vs gain).
- Whether the burst-organ (native soft-skip) generalizes: does a learner carve dedicated reflex
  directions for every recurring irreducible condition, and are those directions readable as a
  vocabulary of "what I have given up on learning"?

[^private]: Not mirrored: this link points to a document in the private lab repo (the roadmap, the queue, an unrun spec, reading notes, or a conversation). See the top-level README for what is held back and why.
