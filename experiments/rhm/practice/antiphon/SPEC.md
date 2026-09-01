# antiphon — SPEC: the question port

**Status**: spec, 2026-08-31. Nothing run, nothing built. Written to hand a new conversation the
full context of where this question came from and what it touches; the experiment designs at the
bottom are sketches to react to, not instructions.
**Origin**: `OPEN_QUESTIONS.md`[^private] #3 (Jasper, 2026-08-30) and the
discussion closing the `quartet` round (2026-08-31, this repo's session
`01Ch29y4DUSGV2TwxquuVUZC`).
**Attribution**: the question itself, the answer-shape to OPEN_QUESTIONS #1 ("you're always
learning; a value-relevant feeling that a question is *bad* is what keeps the bad thing out of
your representations"), the self/other-provenance question, and the RHM-vs-real-world payoff
framing are Jasper's. The reafference mapping, the reading of `woodshed` as this port's first
measured cell, and the `sur_merge`-as-nerdsnipe observation came out of the exchange.
**Queue entry**: "the question port" in `QUEUE.md`[^private].

## The question

> Is it possible to do practice-style climbing faster than normal by "asking the right
> questions"? Presumably there are questions which bisect certain RHM rules, and asking them
> ought to help you disambiguate higher levels quicker. Can the value system be the judge of
> whether a question is a good one or not?

Every node in the practice arc to date takes its questions from outside: the damage schedule,
the demand distribution, the era ladder are all exogenous, and ROADMAP §1.3 names demand
variation as one of the teacher's two surviving seats (the other is gauge choice). This spec is
about whether that seat can be occupied endogenously — question-choice as an outer-loop action,
graded by the learner's own value signals — and what such a port would buy.

## Why this is not just another allocation knob (the five threads it ties)

**1. It is the surviving form of "how do you not learn from something?"** (OPEN_QUESTIONS #1.)
The datum-shaped answer is closed: every post-consumption diet gate failed for mapped reasons —
destructive on the shared channel under abundance, leverage-free under scarcity
([`two_deltas`](../two_deltas/README.md) findings 1 and 6's gate cells). What *works* in the
record is upstream and structural: the typing veto on irreversible ops, the skip op, and —
most telling — the learner in [`tuning/`](../tuning/README.md) natively carving a fixed ln_f
"back off this span" reflex (cos 0.957 across bursts) for exactly the event class whose correct
response is don't-learn-this.
[`absorption_blinds_the_evaluator`](../../../../ideas/absorption_blinds_the_evaluator.md) §6
already asks whether such reflexes form "a vocabulary of what I have given up on learning."
Jasper's reframe: refusal lives at the level of which questions get posed, not which answers
get filtered. The arc's own negatives have been converging on this.

**2. A question is the `u` in the answer's conditioning set** — asking is acting, the answer is
reafference. This is the arity-2 thread
([`operators_are_arity_two`](../../../../ideas/operators_are_arity_two.md)): the agency law we
keep re-measuring (δ fires in ACT, silent in PLAYBACK, ~369×;
[`performance_error_is_the_bridge`](../../../../ideas/performance_error_is_the_bridge.md) §3)
says credit requires your own act in the datum's causal history — passively received data is
playback. Two consequences worth holding:
- **The port's first cell is already measured.** [`woodshed`](../woodshed/FILES.md) (in
  [`quartet`](../quartet/README.md) finding 3): rehearsal of a *received* vocabulary is
  re-posing someone else's questions as your own, and the one-bit credit-vs-exposure split —
  imitate your rehearsals that *solved* vs an equal random sample of the same episodes — ordered
  credit > exposure > none on every trust statistic and on deep-era value (0.70/0.69 of the
  anchor→gift bracket vs 0.29/0.34). The ACT/PLAYBACK dissociation, on the trust axis.
- **Self/other provenance has a cheap mechanical draft.** Keep efference copies of your own
  questions; an arriving datum matching no outgoing question is exafference — other-provenance.
  (The belief tree already holds the biological signature: tickle attenuation,
  [`cerebellum_and_cognitive_architecture`](../../../../beliefs/trees/cerebellum_and_cognitive_architecture.md).)
  This is the wire-level primitive under Jasper's theory-of-mind question — "how do you consume
  a data point and recognize it as coming from an entity that is not you?" — and under
  `census`/`assay`'s arrival-dominates: the `given` arm's spellings are answers to *someone
  else's* questions, and credit is provenance-gated.

**3. Reading is querying.** [`reread/lm`](../reread/lm/README.md): extraction from a frozen
archive is level-ordered and renewable because *the questions change as the vocabulary climbs* —
re-reading pays because you are a different asker. The corpus wall (~10× distinct data per half
level) then reads as the archive's answer capacity per question-level, and the port gives it a
sharp test: do bisecting questions squeeze more depth from a fixed archive, or is the wall
question-invariant? Either answer says what the wall is made of.

**4. What it buys, and where.** Jasper's framing: on a constrained grammar like RHM nearly every
answer is worth something, so the *corruption* payoff is small; on language and the world, people
get nerdsniped chasing aleatoric noise, so it may be large. Amended in discussion: the meter
makes question-*ordering* decisive even on RHM — [`recital`](../recital/README.md) (a
bottom-heavy schedule is rank 1 in every world) and `reread`'s depth-mix curriculum both already
measured that without naming it. So there are **two separable payoffs, both RHM-testable**:
climb faster per priced sample (bisection vs the exogenous ladder), and avoid corruption — for
which the trap must be installed, and the machinery exists: `tuning`'s magnitude-matched bursts
are questions whose answers are noise dressed as news, and `sur_merge` (merged on a burst at
+12.5× floor) is the nerdsniped agent in vivo. Note the **two failure poles** a question-judge
must steer between: chasing noise (the nerdsnipe) and chasing comfort
([`merge`](../merge/README.md)/[`fourwall`](../fourwall/README.md) β=2: a competence-coupled
question distribution manufactures 30–70× starved holes — the arc's measured negative for naive
self-demand). Between the poles is the learning-progress band-pass (~0 when mastered, ~0 when
irreducible) — the belief tree already treats that shape as a *specification* for the expansion
grader; here it becomes a spec for taste in questions.

**5. The value system's role, per the record, is gauge choice, not cleverness.** Every working
outer loop is a thermostat reading the right currency ([`teacher_slot`](../teacher_slot/README.md),
[`conductor`](../conductor/README.md), [`maestro`](../maestro/README.md) — "the reward's type
makes the judge"). Expect the same here: the candidate endogenous question-judge is not a
sophisticated model but a currency — Δ`at_support` (next-level minability) per priced sample is
the obvious first candidate, with the noisy-TV discriminator (noise-disagreement does not
*close* when you collect there; [`heterogeneous_graders`](../../../../ideas/heterogeneous_graders.md)
§9) as the aleatoric guard. Whether that currency suffices, needs the band-pass shape, or needs
something else entirely is the experiment.

## What exists on disk to build on

The [`crescendo`](../crescendo/README.md) stack (fork lineage
`caesura.py → intonation.py → tacet.py → crescendo.py`) is the natural donor — the full crank
with the thermostat outer loop, the free yield gauge, and (via `intonation`) a live fallible
executor if execution currency matters here. `tuning`'s burst calibration (magnitude-matched
irreducible noise) is the aleatoric channel's machinery. The exact DGP gives the oracle ceiling:
questions chosen to maximally split the candidate completions of partially-mined next-level
rules are *computable* (exact BP oracles exist and are, by standing policy, withheld from the
agent — they grade, never feed). `woodshed`'s rehearsal harness is the nearest existing
question-*posing* machinery (its demand-construction lesson is load-bearing: `wd_s0` posed
8-simultaneous-corruption questions, solve rate 0.003, and taught nothing — question difficulty
alone moved bracket closure by ~0.4).

## What a "question" could be here — deliberately left open

The parameterization is the design step, and it is yours. Candidates discussed, none privileged:
a chosen damage cell (which node, which level, which difficulty — the demand draw as the
learner's action rather than the schedule's); a chosen probe sequence completion (closer to the
bisection reading); a rehearsal episode (`woodshed`'s form); a chosen archive slice (`reread`'s
form). What matters, per the repo's controls discipline: whatever the question space is, the
comparison arms must hold total volume, difficulty mix, and priced budget fixed while moving
*only* selection — `ostinato`'s ρ-knob and `woodshed`'s one-bit twin are the house style for
this, and `ostinato`'s other lesson applies too: **size the premise against the banked logs
before building** (its offline read corrected the round's framing before a GPU was spent).

## First shapes (sketches to react to; per repo norms, no outcome is interpreted in advance)

- **A1′ — does question quality move the climb?** Bisection-oracle questions vs the exogenous
  ladder vs random questions vs competence-coupled questions (the β=2 negative control), at
  matched priced budget, on the crank. Reads: the three clocks, cycles-to-commit per level,
  `at_support` trajectories, trust formation (the `woodshed` instrument is on disk).
- **A2′ — can the learner's own value signal pick the questions?** Replace the oracle with
  Δ`at_support`-per-priced-sample (thermostat-grade, per the arc's own lesson) and let it choose
  from the same question space. The within-level ledger as the arm that should refuse or
  misprice, if the type law holds here.
- **A3′ — the trap.** Add the aleatoric channel: burst-flavored questions whose answers carry
  matched surprisal and zero structure (`tuning`'s calibration), with the noisy-TV discriminator
  available to the judge. Does the endogenous judge avoid the nerdsnipe that `sur_merge`
  measured, and at what price?
- **P — provenance (cheap, possibly first).** Log efference copies of posed questions; tag
  arriving data self/other by match; ask whether provenance-gated credit changes what forms
  trust — the `woodshed` result predicts it should, and this is the mechanical first draft of
  the theory-of-mind thread.

These need not all run, and better shapes than these are welcome — A1′ without A2′ is already an
answer to the headline question, and P is nearly free beside any of them.

## Norms

The usual: `/run-experiment-on-modal` before touching Modal; fork bit-identical with knobs off,
gated at 0.000e+00; single seed, ranks/signs/floor-multiples as claims; exact oracles grade and
never feed; discuss results with Jasper before READMEs or interpretation; `QUEUE.md` /
`ROADMAP_PROGRESS.md` bookkeeping on landing. The one norm to hold especially hard here, given
§4's two failure poles: this node is about the *question distribution*, so controlling what the
question knob does **not** move is the whole experiment.

[^private]: Not mirrored: this link points to a document in the private lab repo (the roadmap, the queue, an unrun spec, reading notes, or a conversation). See the top-level README for what is held back and why.
