# Recurrence manufactures confounds: spurious scaffolds, the merge op, and the chunk→belief limit

**Status**: Proposal / conceptual synthesis (2026-08-17); **first contact run the same day** —
[`experiments/rhm/practice/fourwall/`](../experiments/rhm/practice/fourwall/README.md) (fw_s0–fw_s3).
Headlines against this doc, per that node's writeup: §2's debt and §3's scaffold (weak form)
**supported**; §2's caveat resolves to *available, not taken* (no spontaneous binding); §5's merge
**decomposes into three ops** (merge / re-key / retire — deletion alone recovers only the no-index
baseline); §7 gains a measured cap (the earned quotient is bounded by the unit vocabulary). The
§-level revision of this doc is **queued, not yet applied**. **Auxiliary to
[practice_manufactures_its_own_credit.md](practice_manufactures_its_own_credit.md)** — kept separate
because it began as intuition; to be merged into that doc as a numbered section as the evidence
consolidates.
**Date**: 2026-08-17
**Prompt**: a conversation with Jasper (2026-08-17) reading the practice arc (parent doc §§12–19)
against line-dancing phenomenology. The dance anchors (wall-indexed learning, the on-the-spot
rotation at tempo, the whittling-away feeling) are first-person reports, used as anchors for the
*shape* of the theory, not as evidence — same convention as the parent doc's piano anchors.
**Attribution**: the seed intuition — *scaffold with spurious correlations to learn the new thing
fast, then progressively tear the representation down toward its most compressed form* — is
Jasper's, as are the wall phenomenology, the correction that the rotation is forced on the spot
rather than learned per-wall, the observation that wall rotation is the exception rather than the
rule among frozen sensory correlates, the MDL framing, and the name **merge**. The typed-gaps
mapping, the confound-debt inversion of allocation, the transfer-vs-audit distinction, the
demand-weighted-MDL / belief-limit synthesis came out of the exchange.
**Builds on**: [practice_manufactures_its_own_credit.md](practice_manufactures_its_own_credit.md)
(components (1)–(3); §5 two-knob map; §6 manufactured conditions; §18 synthesis; §19 typed gaps),
[meta_learning_under_metered_data.md](meta_learning_under_metered_data.md) (the meter),
[two_timescale_value_loop.md](two_timescale_value_loop.md) (layer split)
**Experiment anchors**: [`crystallize`](../experiments/rhm/practice/crystallize/README.md) ·
[`ratchet`](../experiments/rhm/practice/ratchet/README.md) ·
[`ear`](../experiments/rhm/practice/ear/README.md) ·
[`recital`](../experiments/rhm/practice/recital/README.md) ·
[`typed_gaps`](../experiments/rhm/practice/typed_gaps/README.md) ·
[`mjc/practice/etude`](../experiments/mjc/practice/etude/README.md)

## One-liner

Allocation manufactures recurrence, and **recurrence manufactures confounds**: holding a context
fixed so the benchmark becomes estimable binds every frozen incidental variable to the practiced
unit. The binding is a *scaffold*, not a bug — a spurious-but-stable key makes credit addressable
before the true key is representable — and it is paid down later by a **merge**: a discrete op that
coarsens the library's *index* (never its content) by discovering which context distinctions don't
matter. Run to its limit over every invariance, the merge turns a chunk into a belief — which gives
"a chunk is not a belief" ([typed_gaps](../experiments/rhm/practice/typed_gaps/README.md)) a
constructive converse: **a belief is a chunk in the limit of maximally varied demand.**

## 1. The anchor phenomenology

Learning a four-wall line dance: the moves initially bind to the wall currently faced — retrieval
is keyed on (position-in-dance, wall). The choreography then forces rotation *on the spot, at
tempo*: each repetition ends 90° rotated, so the sequence must execute under a wall-key that was
never practiced. Over repetitions the representation is whittled from wall-indexed toward the dance
"in its purest sense" (body-relative), and the initial wall-binding subjectively *helped* early
learning rather than hindering it. Anchor, not evidence.

## 2. Recurrence manufactures confounds — the debt side of allocation

The parent doc's component (1) says allocation manufactures recurrence: make a context recur, fast
relative to competence drift, against a stable target — the estimability condition for `b(s)`. This
doc adds the flip side. To make a context recur you must hold it fixed, and "the context" as
executed is not the variables you intended to fix but the whole sensory bundle: the wall, the room,
the mirror, the count-in, the recording, the instructor's called cues, the preceding sequence.
Within the drill window every frozen variable is perfectly correlated with the practiced unit — a
constant predicts everything — and the credit machinery has no way to distinguish *correlated
because causal* from *correlated because allocation held it fixed*. The confound is not in the
task's DGP at all; it is an artifact of the sampling policy.

Three consequences:

- **The debt is specific to practice.** In the flow-of-experience regime (§18's degenerate
  monolith: free i.i.d. recurrence), incidental variables decorrelate on their own — you never bind
  the wall when the wall varies for free. Manufactured recurrence is exactly the regime in which
  the bindings form. Deliberate practice incurs a debt the monolithic learner never sees.
- **The debt scales with allocation quality.** The tighter the drill — the more of the world frozen
  to make credit legible — the stronger the incidental bindings. Slow+strict, the corner of the §5
  two-knob map that maximizes credit legibility, also maximizes binding.
- **Scope caveat.** Recurrence does not *inevitably* bind every frozen variable; it makes them all
  *available* to bind, and inductive bias picks which ones actually take (perceptually loud,
  cheap-to-compute cues — the wall — being prime candidates). The gap between available and taken
  is measurable on a substrate where we control which spurious features exist and how loud they are
  (§8).

## 3. The scaffold's value: provisional addresses

Why does the spurious binding *help* early? Because allocation's estimability condition never asked
for a **causal** key — only a recurring, discriminable one. Early in learning, the *true* key for a
unit ("position within phrase three") is not yet representable: phrases don't exist in the
vocabulary yet. [`ratchet`](../experiments/rhm/practice/ratchet/README.md)'s nesting result
(`T[l]` is built from `T[l−1]` entries; a missing level makes the next level *unrepresentable*, not
merely worse) has a mirror image here: the addresses you can key credit on early are the ones you
already have, and the wall is perceptually free, stable across the practice window, and available
from minute one. A spurious-but-stable key makes `b(s)` estimable and credit addressable *now* —
**a loan against representational machinery that hasn't been built.** The tear-down phase is not
hygiene; it is the amortization schedule of that loan.

The standard ML literature (shortcut learning, texture/background bias) treats this territory as
pure pathology, with domain randomization as the cure. What the practice framework adds is a priced
account of the shortcut's *scaffolding value* — why binding the spurious key first is the correct
move under a narrow early demand distribution, with the invariance bought later, on a schedule.

## 4. The rotation typed: demand-news, with a rate axis

In [typed_gaps](../experiments/rhm/practice/typed_gaps/README.md)' type system the wall rotation is
unambiguous: **nothing becomes false when you rotate** — the choreography is unchanged; what is
*asked* moves. That is `setlist` (demand-news), not `transpose` (truth-news), and the organ for
that currency is evaluative re-selection, not dense re-learning. The whittling *is* that organ
running: maintenance of committed skill as demand-tracking.

But the dance is not `setlist`'s regime. `setlist` drifted slowly (OU), and there *tracking* won —
audit-and-reselect matched a perfect tracking oracle at ~8% grader cost. The four-wall rotation is
**fast cyclic** demand drift: a new key every repetition, arriving faster than mining can serve it.
Tracking cost scales with drift rate, so there should be a rate threshold above which the correct
response to demand-news flips from *track the key* (maintain a keyed library) to *quotient the key
away* (merge). Proposed sharpening of the type system: **the demand organ has two responses, and
drift rate — relative to re-selection cost — selects between them.** The meter is what forces the
flip: with free budget you would populate an entry per wall and get coverage by enumeration; the
rotation arriving at tempo makes the invariant representation the only affordable answer (§18
ground 3, one level up).

## 5. The merge op

The library is a lookup table from observed context to committed unit; any choice of key induces a
partition of situations into cells. **Merge = coarsen that partition**: discover that several cells
hold the same unit, collapse them into one cell, and store a single representative in the frame
where they coincide. The arc's existing ops — mine, select, commit, recert/reselect — all act on
*entries under a fixed index*; merge is the first op that acts on the **index itself**. Both ends
of the key-granularity axis are already measured failure modes, which is what makes the axis a real
variable: state-conditioned commitment beats state-independent by **1.8–3.0×** where the key
carries information, and per-key selection is measurably more curse-prone (winner's curse 3.3×)
than one global unit — both [`crystallize`](../experiments/rhm/practice/crystallize/README.md).

**Constraints and structure, in the arc's terms:**

- **Compression on the index, never the content.** The étude's law — compilation is selection +
  commitment; averaging valid renditions destroys them (2.1× étude; 3.6–5.0× crystallize, mechanism
  exact) — applies at merge time: the merged cell keeps *one representative* (or the best), never a
  blend. De-walling a dance is not blending four wall-variants into a smear; it is finding the
  frame (body-relative coordinates) in which the four entries coincide and re-expressing the unit
  there. Where the content was already stored body-relative, the merge is free and
  phenomenologically silent — the addresses were aliases all along.
- **Two evidence routes.** (a) *Audit of aliases*: compare populated entries under different keys —
  expensive, offline, requires having paid to populate them. (b) *Forced transfer under cache
  miss*: retrieval demanded under a key with no entry; the only move is executing the existing
  entry under a transformed frame, and the success profile of that execution is the invariance
  measurement, from one probe, without ever populating the other keys. The dance protocol runs
  route (b) — which is why "learn per-wall then dedup" is not what happens and doesn't need to be.
- **Merge entangles with repair.** The forced transfer is a *decomposition instrument*: it splits
  the unit into a frame-invariant component (transfers silently; merge free) and a contaminated
  component ("travel toward the mirror" stored instead of "travel forward" — breaks loudly, at a
  specific location, and needs genuine content re-learning). The felt whittling is mostly the
  repair; the merge on clean parts is instantaneous.
- **Tempo makes contamination observable.** At tempo, execution is ballistic and the reactive
  fallback is unavailable (§5 of the parent doc), so the transfer probes the *committed unit
  itself*. Rotated slowly, closed-loop control would patch frame contamination on the fly and the
  binding would be masked forever. The protocol probes committed content in the one regime where
  only committed content can answer.
- **Routing, not pruning, predicts the scaffold survives.** Per parent §3½, the wall-index should
  persist as a fallback re-grounding beacon: when a dancer loses their place mid-dance, recovery
  should route through the wall. (Anchor-level prediction; consistent with report.)

## 6. Silent bindings: the wall rotation is the exception, and that's optimal

The frozen bundle is high-dimensional, and culture's decorrelation catalog covers a hand-picked
sliver of it. The biggest un-varied correlate in most motor sequences is **the preceding sequence
itself**: each chunk keyed on its predecessor, which is why most people can execute a learned piece
only from the beginning. Where practice technology addresses a frozen dimension, it has the same
shape as the wall rotation — "start from bar 37" drills are manufactured decorrelation on the
serial-position key — but the catalog is sparse, and everything it doesn't cover stays bound
silently: the room, the shoes, the specific recording, the called cues.

The typed-gaps reading says this sparseness is not laziness but the **meter-optimal policy**: a
chunk stores demand-concentration, so a binding to a variable consumption never varies costs zero
coverage — it isn't even *wrong*, in the only currency chunks are graded in. Choreographers embed
wall rotation in the dance precisely because rotation is *in* the consumption distribution (that is
what "a four-wall dance" means): **decorrelate exactly the dimensions demand exercises.** Latent
bindings on every other dimension are technical debt at zero interest — until demand moves. A first
performance in a new venue is, on this account, partly a *mass cache-miss*: every silent binding
coming due at once.

This adds a species to parent §6's catalog of practice technology: alongside manufactured
recurrence (études, scales), manufactured target stability (the metronome), and the lent grader
(teachers), **manufactured decorrelation** — choreographic structure that schedules the
intervention breaking a chosen confound, at a paced rate. It is the dancer's controlled experiment,
run by the dance.

## 7. The limit: a belief is a fully-merged chunk

Each merge moves the unit onto one of the true DGP's invariances. The fully merged object —
quotiented over *every* nuisance variable, invariant to everything non-causal — is minimal-MDL, and
it is also something else: it is a **belief**. This gives typed_gaps' headline a constructive
converse:

> **A chunk is not a belief; a belief is a chunk in the limit of maximally varied demand.**
> Truth-tracking is what demand-tracking becomes when the demand distribution exercises every
> invariance. Skills and beliefs are not different kinds of object; they differ in how much of the
> quotient has been taken, and the meter decides how much quotient you buy.

The MDL of the skill library is therefore **demand-weighted**: full invariance is what a
belief-like representation wants; a skill needs the quotient only over the demand-varied subspace,
and each merge is worth its price only if demand exercises that invariance. Two corollaries:

- **The experimental method is the four-wall rotation adopted as a universal policy.** "Controlling
  variables" is manufactured decorrelation applied to *every* dimension rather than a demand-priced
  subset — which is *why* science outputs beliefs and practice outputs skills: same op, different
  budget. (This is the repo's taste principle — fundamental causal structure over scoring high on
  the test — restated as a choice of belief-MDL over skill-MDL.)
- **One more clause for the teacher thread** (parent §§14–17: lend the grader, schedule the
  recital, hold at the bottom, carry ladders longer than your reach): the teacher also **schedules
  the decorrelation** — choosing which confound the student confronts next, and at what loudness.
  The four-wall choreography is a teacher's decision frozen into cultural artifact.

## 8. Testability (cheap, mostly existing machinery)

**2026-08-17: specced — [`rhm/practice/merge/SPEC.md`[^private],
which adds one arm beyond this section's list: an unmetered dense learner under free i.i.d.
variation, to test invariance-by-enumeration against invariance-by-merge.]**

The [`setlist`](../experiments/rhm/practice/setlist/FILES.md) machinery already does most of it.
The instantiation: give the learner a free observable context feature spuriously correlated with
the true latent under a narrow initial derivation distribution (the manufactured confound), then
rotate the derivation distribution — paced, and at rates spanning slow-OU through fast-cyclic.
Measure: early learning speed with vs without the spurious key available (the scaffold's value);
which frozen features actually get bound as a function of their loudness (§2's available-vs-taken
gap); library key granularity over time under a merge op vs a track-only arm (the §4 rate
threshold); and the cost carried by an arm that never merges (storage, per-key curse, coverage
under rotation). Per repo norms, no outcome map is pre-registered here — the instrument list is the
commitment, not the interpretation.

## 9. Risks — held loosely

Places we'd expect to learn something, not falsifiers we commit to:

- **The scaffold value may not reproduce in silico** — the early-speed advantage of a spurious key
  may be specific to learners with expensive credit assignment over latent context; RHM's selector
  may key on the true latent cheaply enough that the scaffold buys nothing.
- **Merge evidence may be inseparable from repair.** If most units carry contamination, the forced
  transfer may look like plain re-learning and the index-vs-content distinction won't be cleanly
  measurable at feasible scale.
- **The rate threshold may not exist as a threshold** — track vs merge could blend continuously, or
  the meter parameters may dominate the drift rate entirely.
- **The belief-limit claim (§7) is the most conceptual and least at risk from any single run**; what
  would erode it is the merge op failing to be a distinct operation at all (i.e., ordinary dense
  learning under varied demand achieving the same quotient with no discrete index event — in which
  case §7 collapses into "generalization," and the interesting content retreats to §§2–4).

## 10. Connections

- **[practice_manufactures_its_own_credit](practice_manufactures_its_own_credit.md)** — parent.
  §19/typed_gaps supplies the currency typing (§4 here); §6 the technology catalog (§6 here); §18
  ground 3 the meter argument (§4–§5); §3½ routing-not-pruning (§5); §5's tempo knob (§5). Natural
  merge target: sibling section to §19.
- **[typed_gaps](../experiments/rhm/practice/typed_gaps/README.md)** — "chunks store
  demand-concentration" is the load-bearing result throughout; §7 here is its converse.
- **[`crystallize`](../experiments/rhm/practice/crystallize/README.md)** — both ends of the
  key-granularity axis measured (state-conditioning 1.8–3.0×; per-key curse), which is what makes
  merge a movement along a priced axis rather than a new assumption.
- **[meta_learning_under_metered_data](meta_learning_under_metered_data.md)** — the meter decides
  how much quotient you buy (§7); merge is forced by demand arriving faster than mining (§4).
- **[revision_not_surprisal](revision_not_surprisal.md)** — sibling on the epistemic side: both
  docs are about what a conditioning set carries. There the gap's content types the *residual*;
  here the frozen (complement of varied) set types the *bindings*. Varying a dimension is moving it
  out of the conditioning set — the same lever, used there for measurement and here for
  representation.

## 11. Literature pass (recalled, not verified — check before citing)

- **MDP homomorphisms / state abstraction**: Ravindran & Barto (quotienting a decision problem by a
  symmetry group — the merge op's formal home); bisimulation metrics; Li, Walsh & Littman's
  abstraction taxonomy.
- **Shortcut learning**: Geirhos et al. 2020 (the pathology framing §3 inverts); domain
  randomization (Tobin et al.) as the cure-side technology.
- **Context-dependent memory**: Godden & Baddeley (divers) — environmental context as retrieval
  key; the silent-binding phenomenology of §6.
- **Serial-position binding in motor sequence**: whether skilled performers can initiate mid-sequence,
  and "start-anywhere" pedagogy — §6's biggest un-varied correlate.
- **Simplicity bias as curriculum**: work on networks learning spurious/simple features first and
  invariances later — whether the scaffold-then-merge trajectory already has an in-silico signature.

[^private]: Not mirrored: this link points to a document in the private lab repo (the roadmap, the queue, an unrun spec, reading notes, or a conversation). See the top-level README for what is held back and why.
