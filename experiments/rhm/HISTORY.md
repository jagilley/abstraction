# History: `experiments/rhm`

*A big-picture pass over ~10 days of dense work (2026-08-13 → 2026-08-22),
predecessor line `experiments/mjc`. Not a changelog — a trace of shifting
questions and priors. Cite the linked READMEs for detail.*

## Why RHM exists

RHM (Random Hierarchy Model) was adopted because natural-language experiments
in the `a2a_forward` / `mjc` lineage couldn't cleanly separate "what the
data-generating process does" from "what the measurement channel sees." RHM
gives an exact, known hierarchy, exact belief propagation, and a synonymy
structure that can be dialed as a variable. The `rhm/` chapter is best read
as a campaign of building instruments precise enough to ask a question
exactly — and, repeatedly, discovering the first instrument was measuring
something other than what it was assumed to measure.

## The recurring meta-lesson: residuals measure load, not meaning

The single most repeated correction across every sub-line is that an
FM-residual-style signal (self-model prediction error, relevance heuristics,
cross-entropy under drift) was first read as epistemic — "this tracks
complexity / self-knowledge / structure" — and then shown to actually track
computational load, capacity mismatch, or measurement artifact:

- Residual **rank** doesn't track DGP complexity, it tracks the main model's
  approximation shape (`RESIDUAL_RANK_README.md`).
- Raw NTP cross-entropy is *exactly* blind to support-fixed drift by an
  entropy identity — it looks stable while the world moves
  (`directed_sculpting/full_loop/README.md`).
- The depth-FM residual is anti-localized with token surprisal — it tracks
  compute load, not epistemic difficulty — so scalar-weighting NTP by it is
  a clean null (`endogenous_teacher/README.md`, `DESIGN.md`).
- The working residual-decomposition arithmetic (`A = FM + Residual`) is
  false as arithmetic; the naive metric was retired for
  `R_res_participation` after a redundancy audit
  (`residual_decomposition/README.md`, `trajectory/README.md`).
- `conditional_revision/` had to shift the FM's target by exactly one
  position (not six blocks) before its residual could contain any aleatoric
  component at all — the incumbent residual could only ever mean "I lacked
  capacity" (`conditional_revision/SPEC.md`, `endogenous_teacher/DESIGN.md`).

These aren't independent — later docs cite earlier ones as diagnosis for
prior nulls. Discount any single "residual"/"relevance" number in an older
doc unless a later one confirms it against this list.

**Capacity was never the bottleneck; signal was.** A second overturned belief:
composition ceilings were assumed to be a capacity problem.
`RHM_DEEP_COMPOSITION_README.md` supplied the oracle-aux
control (per-level latent labels reach the exact belief-propagation ceiling
at every level, same architecture) and `PER_LEVEL_EXPANSION_README.md`
closed it with a 200×-capacity sweep that moved the ceiling ~1pp. `ratchet/`
hit the same wall from a different angle — an oversized forward model masked
its own gate dynamics (`ratchet/RHM_RATCHET_README.md`) — and the real
causal variable turned out to be **supervision density**, via a controlled
NTP-masking sweep that flipped local-loss from harmful to helpful
monotonically (`ratchet/RHM_SPARSITY_SWEEP_README.md`).
`RHM_FRONTIER_AND_LEGIBILITY_README.md` then separated two conflated
variables — `m` gates *learnability*, occupancy gates *recoverability* — and
reframed the self-model loop as an **amplifier of structure that already
exists, not a source of it**, retroactively explaining most earlier
self-model nulls.

## Does anything actually ratchet?

The `ratchet/` line (compounding self-models across wake-sleep cycles) never
got activation-level self-models to compound on RHM the way they did on
MNIST/language — the advantage appears for one cycle and migrates back to
shallow levels by cycle 4 (`ratchet/RHM_SPARSE_RATCHET_README.md`,
`RHM_L4_RATCHET_README.md`). RL-based supervision (rich, not just sparse)
broke through with a real generation gain and a gate that stays open, but
still didn't compound — a 40-cycle extension retracted its own motivating
trend as aliased noise (`ratchet/RHM_RL_RATCHET_README.md`,
`RHM_RL_GEN_DISTILL_EXTENDED_README.md`). Meta-learning variants (FOMAML,
rule-family transfer) collapsed to ordinary multitask learning
(`RHM_FOMAML_README.md`, `RHM_META_LEARNING_README.md`).

The `practice/ratchet/README.md` line, despite the shared name, answers the
same underlying question more successfully. There, what compounds is a
**level-indexed action vocabulary mined from the agent's own successful
repairs**, not a compressed self-model of activations — and it does compound,
with a sharp poison effect: committing a bad level-2 macro early makes level
3 *unrepresentable*, not just worse. `directed_sculpting/` converged on a
compatible generalization from another angle: compounding needs a **moving
grader**, not a moving world — a signal's invariance to drift is not the
same as that signal being necessary.

## The practice arc: from a vacuous precondition to a teacher-slot economy

`practice/` ported a stuck question from `mjc/practice/etude/` — "is hierarchy
meaningful only over boundaries that carry information?" — onto a testable
substrate (`practice/README.md`). The arc moved through a chain of
consolidations: `crystallize/` showed certification only matters where
practice moves the *executor*, not on a frozen plant; `ratchet/` (practice's,
not the FM line's) showed premature commitment forecloses representation and
that a mined vocabulary's value is concentration, not coverage; `ear/` and
`recital/` climbed the grader and found no single internal signal prices
"time at the bottom of the ladder," so a fixed bottom-heavy schedule beat
every adaptive pacer. Two consolidations matter most: `typed_gaps/README.md`'s
double dissociation between truth-drift and demand-drift concluded **a chunk
is not a belief** — maintained skill tracks demand, not truth — and
`practice/fourwall/` decomposed "whittling toward compressed form" into three
operations (merge/re-key/retire), landing on the claim that both an
operation and its justification must come from outside the loop.

That is exactly what `practice/teacher_slot/` is now testing
(`practice/teacher_slot/README.md`): the scarce resource in a teaching signal
is not judgment but *which gauge gets consulted*. A rule reading a
one-level-up currency merges cleanly; the same rule reading the learner's own
experienced loss refuses correctly, every time. The frontier,
`practice/teacher_slot/endo_yield/SPEC.md` (launched 2026-08-21, not yet
interpreted), replaces a probe secretly fit on ground-truth labels with a
label-free endogenous yield signal — closing the arc's last un-endogenous
dependency and removing a 13× cost premium along the way.

## Directed sculpting: from a confounded null to a priced, endogenous loop

`directed_sculpting/` inherited "direct collection by value × visits" from
`mjc/on_policy/directed_on_policy/` and found the naive port degenerate on
RHM's shared, uniform rule tables (`directed_sculpting/README.md`). Three
primitives (distractor channels, support-fixed drift, a repair-cost
instrument) made the confound testable — this is where the CE-blindness
finding above comes from. `full_loop/` progressively removed
researcher-supplied structure: an endogenous, reward-only value signal
recovers 76% of an oracle's separation power (`full_loop/README.md`);
`endogenous_expansion/` found *belief* progress is anti-informative
(ρ = −0.80 with fidelity) while grader fidelity acts as a hard ceiling on
what a learner converges to; `level_moves/` self-corrected in-doc when its
apparent level preference turned out to be a span-length confound, landing
on "value tracks compositional structure, not depth." `metering_sweep/` ran
the arc's cleanest two-round correction: round 1's rising prize was an
artifact of `n_steps` confounding meter with overfitting; round 2, at
iso-compute, found relevance-directed collection keeps 87% of an oracle's
value at 2.2% of its cost (~1200× more efficient). The frontier is
`full_loop/composed_loop/README.md` (2026-08-05) and the unbuilt
`full_loop/metered_climb/SPEC.md`, built to re-test the one surviving
"depth matters" result under the correction that killed the metering-sweep
positive.

## Conditional revision, confabulation, and question modeling: what a residual can honestly claim

This cluster asks what a self-report or self-model residual is actually
entitled to say about a model's own processing. `conditional_revision/`
(`SPEC.md`, `README.md`) built the temporal-target FM and found the binding
constraint on measurable self-knowledge is **belief depth**, not the
conditioning gap it was designed around — an instrument audit later showed a
clean-looking deep-level correlation was two >1-nat readout errors
cancelling. `confabulation/` operationalized Nisbett & Wilson's question on
the FM decomposition and found first-person advantage survives moving from a
static to a temporal residual, but the *charged* part of that residual is
public, not privileged — self-report and an outside observer track belief
revision equally well (`confabulation/temporal/README.md`).
`confabulation/temporal/epistemics/` then dropped the report channel
entirely and found the temporal FM is epistemically inert as a
signal-former; only its timing claim survives. `endogenous_teacher/` is the
direct predecessor of `conditional_revision/` — its clean null
(residual-weighted NTP ≈ shuffled) was diagnosed as testing the wrong thing
entirely, motivating the target shift. `question_model/` (2026-08-17,
newest) closes the identity with a third term — question-news, movement of
the posterior over which rule generated the text — and found the channel is
kept alive by finite context, not by drift itself.

## Smaller independent lines

- `specialization/README.md`: restricting breadth or reweighting levels
  doesn't move the depth frontier — deep composition sits on the shallow
  substrate, so starving it is harmful. Reframed 2026-07-18 from deficiency
  into an *enabling condition* for transfer: broad training beats narrow on
  RHM's single shared rule table.
- `minting/README.md`: a rule-table filter on self-minted data deletes
  out-of-support garbage but can't fix in-support density error — the table
  is a support oracle, not a density oracle. Contains an explicit retraction
  of an earlier single-seed "mirror collapse" reading.
- `RHM_SCULPTING_README.md` / `RHM_SCULPT_CONTINUAL_README.md`: beam search
  captures a real, depth-scaling lookahead prize; latent planning is a
  surrogate on clean channels but *better* under partial observability;
  ratcheting this across generations is weak.
- `RHM_COMPLEXODYNAMICS_README.md` / `residual_decomposition/trajectory/`:
  training trajectories need an annealer (SGD alone arrests short of the
  floor); the floor is set by the DGP's own sophistication, not the observer.
- `SLEEP_CHUNKING_RHM_README.md` / `RHM_LATENT_LOOP_README.md`: generalizable
  self-knowledge appears only under a **latent** target and inverts under a
  token target — most earlier self-model runs had been predicting tokens
  while feeling like latent prediction.
- `ACTIVE_RHM_README.md` / `RHM_EDIT_CONTROL_README.md` /
  `sculpting_control_task.md`: RHM as a control task — edit planning works
  but exposed off-manifold belief-gaming, fixed by a generator-side veto.

## Where this leaves things

The throughline is methodological rather than a single finding: almost every
"this metric shows X" claim made early in the arc was later shown to be
measuring a nearby but different thing, and the correction was recorded
rather than quietly dropped (`RHM_RL_GEN_DISTILL_EXTENDED_README.md`'s
retracted trend, `rule_family/NOTES.md`'s "would have shipped a false
positive," `residual_decomposition/`'s metric retirement). The live frontier
is `practice/teacher_slot/decision/` and `practice/teacher_slot/endo_yield/`
— what a teaching signal is *for* once judgment is no longer the scarce
ingredient — plus the unresolved "does depth ever really matter" thread in
`directed_sculpting/full_loop/metered_climb/`.
