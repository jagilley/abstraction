# enharmonic — SPEC: the quotient. Re-keying the vocabulary by category, so the next rung stops costing the corpus

**Status**: spec, 2026-09-01. Nothing run, nothing built. Written to hand a new conversation the
context for the roadmap's second extension after the offline sizing of 2026-09-01 changed what
that extension is made of. The design step at the bottom is a set of sketches to react to, not
instructions. The name: enharmonic notes are different spellings of the same pitch.
**Origin**: [`../tutti/sizing/SIZING.md`](../tutti/sizing/SIZING.md) (the L5 sizing and the grip
determinator, both offline), read in the 2026-09-01 orchestrating conversation against
`ROADMAP.md`[^private] §4.3 F3 and §7.1.1 point 4, and Jasper's go on the
direction ("your read SGTM").
**Attribution**: the four-wall anchor and the merge op are Jasper's
([`recurrence_manufactures_confounds`](../../../../ideas/recurrence_manufactures_confounds.md));
the hypothesis that question-asking is a one-level-up notion is Jasper's, and the grip law that
confirmed it is the sizing lane's; the observation that the level-size wall belongs to the
flat-tuple key and not to the grammar is the sizing lane's (fact 4); the reading of it as the
second extension's actual lever, and the tie to the question port's bisection form, came out of
the orchestrating exchange. The direction is Jasper's call (2026-09-01); the mirror loop and
the band-pass lessons were added after the round's discussion (2026-09-02).
**Companions**: [`../fourwall/README.md`](../fourwall/README.md) (merge / re-key / retire — the
three ops, and the `lm/` negative) · [`../merge/README.md`](../merge/README.md) (enumeration vs
merge) · [`../antiphon/README.md`](../antiphon/README.md) (the question port; interpretation (c))
· [`../crescendo/README.md`](../crescendo/README.md) (A3, the first extension) ·
[`../tutti/README.md`](../tutti/README.md) (the round this spec came out of; `tu_s0` said which
currency owns which action — the **mirror**: the one-level-up yield gauge commits, δ-silence
advances — and that is the loop this node sits under).

## The question

> Can the practice learner re-key its own vocabulary by *category* — discover that several of its
> committed tuples are interchangeable, and store the next level over categories rather than over
> spellings — and does the crank then reach the rung that flat keys cannot afford?

Every node in the arc keys its table by the flat tuple: `T[ℓ]` is a set of level-1-feature strings
over a span of `s^(ℓ−1)` leaves, and `T[ℓ+1] ⊆ T[ℓ] × T[ℓ]`. That key gave the arc address and
trust (`census`, `native`), the first extension (`crescendo`), and everything since. It also has
a cost the sizing lane made exact.

## Why this is the next rung, and not L5 with a bigger budget

Five facts from [`SIZING.md`](../tutti/sizing/SIZING.md), all offline, all at the arc's world
(v = 8, s = 2, m = 2, depth 6, `rule_seed = 0`):

1. **The flat key space is doubly exponential in level.** |T| = 14 / 56 / 816 / 205,824 /
   1.31e10 at L2…L6; the ratio `|T_(ℓ+1)| / |T_ℓ| = m^(s^(ℓ−1))`.
2. **Arrival, not the r² wall, stops L5.** Over the arms' own frozen L4 books the L5 ratchet
   admits 1–15 true entries, but E[true L5 keys at support 3] is 0.000 at the run's ~1,600
   L5-node observations; a 22-key L5 book needs ~158k observations (≈58 GPU-h per arm). The
   arrival cut at L5 is ~5.9e9× against the ratchet's 1.2–21× at L2–L4.
3. **No admissible world fixes it.** Every (v, s, m) with a cheap L5 sits at the tuple-space
   occupancy `tall/` measured inadmissible. A collision-free rule draw makes L5 ~6× cheaper, at
   the price of the arc's native aleatoric channel — kept on the shelf, not taken.
4. **Under the grammar's own key the level is constant-size.** Each level has `v·m = 16` rules
   over pairs of the level below's *features*. Keyed by parent feature — `fourwall`'s re-key
   basis — every rung is 16 entries: a 12,864× shrink at L5. The wall belongs to the spelling,
   not to the world.
5. **A category-keyed table generalises; a flat one cannot.** A flat entry says nothing about a
   tuple it has not seen. A category-keyed level-ℓ table lets the learner parse *any* level-ℓ
   tuple whose halves it can categorise, down to the level-1 features the reader already
   delivers exactly (`read_acc` = 1.0). So the climb in category coordinates is: learn the 16
   rules at each rung (arrival trivial), and each rung's rules parse the next.

Read against the roadmap: §1.2 claim 1 says depth is bounded by the world's structure and the
learner's time, not by corpus size. In flat coordinates that claim is *false* past L4 on this
world — the learner is corpus-bounded by arrival. In category coordinates it is the claim's
sharpest form. F3 was listed as an r²-wall lever (coverage); it is a level-size lever
(abstraction), and it is on the critical path for that reason.

## The threads this ties

**1. Merge, re-key, retire — and the negative they left behind.** `fourwall` found "whittling
toward the purest form" is three ops: merge deletes distinctions within the current basis, re-key
re-expresses the library in an earned basis, retire stops paying for the old one. Its `lm/` twin
found a real NTP reader *tracks and never quotients* at any rotation rate, and the within-level
ledger votes against merging at all — "the op and its justification must both come from outside
the loop." That negative is the thing this node confronts. The practice learner has three things
the NTP reader did not: a priced exact grader, a forced-transfer probe it can pose as a question,
and a use record. And its justification for the merge is exactly the outer loop's currency:
merging `T[ℓ]` is what makes `T[ℓ+1]` representable at 16 entries, i.e. it raises next-level
minability — the gauge the thermostat reads and the within-level ledger cannot see
(`teacher_slot`, `conductor`). Whether the yield thermostat licenses merges the ledger refuses is
the type law asked one op further along.

**2. The question port's bisection form.** OPEN_QUESTIONS #3 imagined questions that *bisect
rules*. `antiphon` parameterised a question as a chosen repair instance and found the ask directs
the search rather than authoring the answer. A synonymy probe is the bisection reading proper:
*substitute tuple B for tuple A in a context that A solves; does it still solve?* One grounding
per probe, graded by the world's exact grader, and it splits the candidate partition of `T[ℓ]`
in half. Which pairs to probe is a question-selection problem, and the value system grading it —
Jasper's conviction — has a candidate currency in what the trap round measured: the use
record. Under the grip law, this is also the only question-shaped lever whose reach is not
closed to what was already observed and whose dose is not `κ/|T_(ℓ−1)|`. Two lessons from the
round travel with it. The judge that carried value in every tag was a **learning-progress
band-pass** — questions the learner can solve the intended way, not the needy ones — so a probe
selector should weight solvability, not only informativeness. And under the meter (`an_m0`)
every novelty-edge selector paid a solve tax the abundance regime hid: a probe that fails is a
wasted grounding, and the probe budget should be sized on that.

**3. The alias audit is free.** `recurrence` §5 names two evidence routes for a merge: audit of
aliases (compare populated entries) and forced transfer under cache miss. The beam's per-entry
selection record (`log["entry"]["hist"]["beam"]`) already exists per cycle, and the trap lane
found use-weighted precision of the committed L3/L4 tables at 0.73–0.99 against table precision
0.15–0.50. Synonymous tuples are the ones the beam uses interchangeably in the same slot. That is
route (a), on data already logged, and it can be sized against the true categories offline before
anything is built.

**4. Arrival dominates, one level up.** `census`/`assay` established that a gifted table
transfers content and not trust; `woodshed` that credited rehearsal manufactures the trust gift
leaves out. A gifted *quotient* — the true categories handed over — is the same question about a
different object: is a category giftable, or is the partition only earnable? The `given` idiom
supplies the ceiling and the control in one arm, as it always has.

**5. The chunk→belief limit, made mechanical.** "A belief is a chunk in the limit of maximally
varied demand" (`recurrence` §7). The era ladder's damage draws sample every rule uniformly, so
synonymy *is* exercised by demand on this substrate — the invariance is in the consumption
distribution, which §6 says is exactly when a merge is worth its price. The quotiented table is
the first object in the arc that is a belief in that doc's sense.

**6. Junk under the quotient.** At `rule_seed = 0` the L4 mining node carries 0.82 junk mass;
junk keys have no parent feature and cannot be synonyms of anything true. Under forced transfer
they fail; under the alias audit they should sit alone. A merge that admits a junk tuple into a
category corrupts the level above the way an early commit foreclosed representation in `ratchet`
(−0.372, worse than never). The exact grader can audit merge precision, and the trap round's
use-record result says where the junk sits in use.

## What exists on disk to build on

The `crescendo` crank and everything above it (`tutti.py` is the current head of the fork lineage
`caesura → intonation → tacet → crescendo → maestro → conductor`, with the question port grafted
and a second `QuietPolicy` so commit and advance can have different owners; `tu_s0` measured the
mirror assignment — yield commits, δ-silence advances — as the one above the lifetime ceiling,
by letting a level stay live when the advance clock beats the commit clock). `Miner.build` and the ratchet
(`T[ℓ+1] ⊆ T[ℓ] × T[ℓ]`) in `macros.py`; `apply_any`'s DP over the committed table; π's
(level, node) slot layout in `native/` and the corridor head in `native/span/`. `fourwall.py`'s
merge/re-key/retire machinery (one level, a given surface key — the ops exist, the basis was
supplied). `antiphon/questions.py`'s selectors and quota. The DGP: `generate_rules_distinct` (m
distinct tuples per feature; features *can* share a tuple, so rule collisions across parents exist
at any level and are a per-draw fact to size first — `SIZING.md` sized L1's). The oracle:
`MC.exact_features` gives the clean derivation's feature at every level — the true categories,
which grade and, in the `given` arm only, feed. `tutti/sizing/phase0_l5.py`'s `buildable()` and
level arithmetic, reusable verbatim for the category-coordinate ceilings.

## What a "category" could be here — deliberately left open

The parameterisation is the design step. Candidates discussed, none privileged: an equivalence
relation over `T[ℓ]` kept beside the flat table, with `T[ℓ+1]` keyed by the pair of classes; a
minted *category token* occupying a π slot the way a macro does (canvas C1's "first literal token
mint" question, on RHM); the ratchet restated as `T[ℓ+1] ⊆ C[ℓ] × C[ℓ]`; the executor's new
freedom — under a category key a level-(ℓ+1) macro no longer fixes its spelling, and something
(the corridor head; the DP; π) has to choose an expansion at execution time, which is
`can't-decompose` turned into an affordance. What the merge is *graded* in: coverage (the wrong
currency, `fourwall` synthesis 2), consumption, or next-level yield. And what happens to a
category's *trust* — π's mass is per (level, node); a merge changes what the slot proposes.
The executor's expansion choice under a category key composes with
[`../inflection/SPEC.md`](../inflection/SPEC.md) (2026-09-09), where the *bottom* spelling
becomes a context-dependent rule the learner must own and the grader marks spelling; the two
are sited beside each other, and this node's arms import onto that fork once both exist.

Whatever the representation, the comparison arms must hold volume, difficulty mix, priced budget,
and lifetime fixed and move only the key — the house style, and the reason every `given` arm is a
clock yoke of its earned twin.

## First shapes (sketches; per repo norms, no outcome is interpreted in advance)

- **Q0 — size it offline first** (`ostinato`'s discipline, which has corrected three rounds'
  framing before a GPU was spent). Per-level rule collisions at `rule_seed = 0` (how many child
  pairs are shared by two parents, hence ambiguous from below). The category-coordinate level
  sizes and the L5/L6 arrival budget when keys are pairs of classes. The alias audit on the banked
  logs: does the beam's use record cluster the committed L3/L4 rows by their true parent feature,
  and at what purity against a shuffled control. The probe budget: how many forced-transfer
  probes recover the true partition of `T[3]` and `T[4]` at the arms' realised recall, and how
  many the use record saves.
- **Q1 — the quotient supplied, arrival re-asked.** The true categories fed (`given_cat`) as the
  ceiling and the arrival-dominates control, against the flat crank on the same clock: does L5
  arrive at the *existing* budget once `T[4]` is keyed by class, and does the signature
  (`crescendo`'s value clock past the previously certified range) hold there. This is the cheapest
  test of fact 5 and it says whether the rest of the node is worth building.
- **Q2 — the endogenous quotient.** The merge as an outer-loop action with the learner's own
  evidence: the alias audit, forced-transfer probes chosen by a selector (the question port's
  bisection form, with the use record as the candidate currency), or both. The within-level
  ledger as the arm that should refuse; the yield thermostat as the arm that should license;
  merge precision graded by the oracle throughout. The `fourwall/lm` negative is the comparator:
  a learner that tracks and never quotients.
- **Q3 — the second extension.** L5 opened to the mirror loop (yield commits, δ-silence
  advances) with the band-pass judge selecting, and the earned quotient in place: does the
  earnable range extend twice on this world at this budget, and what paces the L5 commit where
  the sizing found every at-support gauge at or above the frontier sub-floor and only the reads
  from behind it clear their floors. Note the mirror's own open question travels here: whether
  its value is the gauge's or the schedule it found (`tu_m_yk`, queued).

These need not all run. Q0 → Q1 alone is an answer to the headline question's first half; Q2 is
the node's real object; Q3 is the roadmap's signature asked a second time.

## Norms

The usual: `/run-experiment-on-modal` before touching Modal; fork bit-identical with the knobs
off, gated at 0.000e+00; single seed, ranks/signs/floor-multiples as claims; exact oracles grade
and never feed except in the `given` arm; discuss results with Jasper before READMEs or
interpretation; `QUEUE.md` / `ROADMAP_PROGRESS.md` bookkeeping on landing. The norm to hold
hardest here: a merge is irreversible in the same sense a commit is (`ratchet`: committing one
cycle early was worse than never), so the merge decision's *gauge* is the experiment, and every
merge event should be logged with what licensed it.

**Q2 specced (2026-09-09)**: [`temperament/SPEC.md`](temperament/SPEC.md) — the endogenous
quotient as the one op with negative within-level value and positive next-level value, hence the
purest test of the type law; the flat key read as the undrained address book. Q0/Q1 above remain
its prerequisites and ceiling.

**Q0 → Q1 → Q2 first pass landed (2026-09-10)**, facts only, interpretation pending discussion:
[`sizing/SIZING.md`](sizing/SIZING.md) (the cover; the token class; the alias audit's negative;
the forced-transfer budget; junk under transfer) and `figures/en_s0_reduction.txt` /
`en_s1_reduction.txt` (the class key solves L5 arrival; the flat comparator's failure is a dead
gauge; the un-yoked signature was the clock; the L4 book freezes at 3 of 13 classes). Index:
[`FILES.md`](FILES.md). Second pass in `QUEUE.md`.

**Third child specced (2026-09-10)**: [`figured_bass/SPEC.md`](figured_bass/SPEC.md) — what a commit should freeze in category coordinates: the key, not the content. From Q1's mechanism (the L4 book frozen at its poorest) and `tutti`'s never-frozen L3.

**Composed arm, probe reach, sweep, and latch landed (2026-09-11→12, `en_s6` → `en_s9`)**: the first
endogenous L5 commit (`en_s9`, c186) once the merge sweeps every live level and the class-coverage
re-arm hook leaves the commit latch alone; the adopted book is eight class keys at the spelling
cap. Written up in [`README.md`](README.md) (addenda); open items in `QUEUE.md`.

[^private]: Not mirrored: this link points to a document in the private lab repo (the roadmap, the queue, an unrun spec, reading notes, or a conversation). See the top-level README for what is held back and why.
