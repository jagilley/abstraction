# SPEC — inflection: a rendering rule below the tables, so the executor has a word to say and the reader a word to hear

**The question in one sentence**: when the map from a committed unit's content to its surface form
stops being a free lookup and becomes a context-dependent rule the learner must own, does the
practice learner arrive at the factoring `tempo/` found on the plant — content stored above the
realization layer, the realization map in a separate organ fitted from its own practice — and do
the execution-currency questions the arc has asked on RHM (Track E′'s gate seat, F2's trust vs
habit) change their answers once there is a word to hear?

**Status**: spec, 2026-09-09. Nothing run, nothing built. A **substrate fork**, sited beside
[`../enharmonic/`](../enharmonic/SPEC.md) and not inside it: `enharmonic`/`temperament` ask the
type-law question and should run unconfounded on the current substrate; this node changes the
substrate below the tables and imports their arms once both exist. The name: an inflection is
the ending a word takes from its context, and the shape a note takes from its phrase.
**Machinery donors** (all untouched): [`../tutti/tutti.py`](../tutti/tutti.py) (head of the fork
lineage; the data path this spec verified) · [`../../rhm_sculpt_precheck.py`](../../rhm_sculpt_precheck.py)
(`sample_derivations`, `possible_sets`, `parse_success_and_heuristic` — the coin and the
any-synonym grader) · [`../crystallize/units.py`](../crystallize/units.py) (`corrupt_hier`;
`canon` handed to the executor) · [`../ratchet/macros.py`](../ratchet/macros.py) (`apply_any`'s
render step, `canon[feats]`) · [`../native/span/span_net.py`](../native/span/span_net.py) (the
corridor head; "`canon` still renders — not the port's business") ·
[`../intonation/intonation.py`](../intonation/intonation.py) (δ_perf's definitions; the free
intention reference) · [`../ear/grader.py`](../ear/grader.py) (`render="rand"` — the one
non-canonical rendering ever used, for manufacturing contexts only) ·
[`../../rhm_drift.py`](../../rhm_drift.py) (`sample_derivations_weighted`, a per-cell synonym
mixture that reduces to the coin at flat weights) ·
[`../../../mjc/practice/tempo/README.md`](../../../mjc/practice/tempo/README.md) (the plant's
form of the question — read first, findings 3, 5, 6, 7 and the Interpretation).
**Idea docs**: [`practice_manufactures_its_own_credit`](../../../../ideas/practice_manufactures_its_own_credit.md)
§3½ (arbitrary binding vs derivable content; identity in the table, corridor in the executor)
and §5 (the two-knob map: form content slow, then hand off) ·
[`performance_error_is_the_bridge`](../../../../ideas/performance_error_is_the_bridge.md) (δ_perf) ·
[`cerebellum_and_cognitive_architecture`](../../../../beliefs/trees/cerebellum_and_cognitive_architecture.md)
(the dense asset's placement relative to the meter).
**Attribution**: Jasper's (session `01PJErBzhTxN5GoAkRSZeZd9`, 2026-09-06→09): the question
whether `tempo`'s factoring speaks to practice as a phenomenon or to executor inductive bias; the
intent → word → evaluate picture of language production, and the hypothesis that a perfect
dictionary on RHM explains part of the E′/F2 nulls; the instruction to verify the coin on the
latest node rather than on the shared generator. A second agent, relayed by Jasper: the rendering
rule gives the reader a job as well as the executor, since undoing morphology is parsing — the
thin cortex of the anatomy discussion
(`../enharmonic/temperament/conversation_2026-09-09.md`[^private])
and `reread/lm`'s climbing reader. Out of the exchange: the scope condition and its verification;
the reading of `tempo` as inverse-direction transfer with the forward direction's contribution
unisolated; register vs agreement; the graded grader; one morphology, two organs. Two Opus explore
agents grounded the first response. Record: `conversation_2026-09-09.md`[^private].

## Why this node exists

**The scope condition, verified in code.** On every RHM practice node the map from a unit's
content to its surface has had no free parameter in either direction. *Data*: `sample_derivations`
picks the production at every node of every level by a uniform draw from the m synonyms,
independent of context (`rhm_sculpt_precheck.py:63`); the era ladder's damage does the same inside
the damaged subtree (`units.py:149,153`). *Grader*: `possible_sets` reduces over synonyms with
`.any(-1)` at the bottom and at every level, so any spelling that parses is a success. *Execution*:
`canon = rules[depth−1][:, 0, :]` — synonym 0, always (`tutti.py:2220`; written by `apply_any` and
by the span head at `tutti.py:1278/1345`). *Reader*: the generator parses leaves to level-1
features, pretrained, then frozen and pinned at 1.000 in the practice stack (`reread`:
vocabulary-gated perception "inexpressible" here). The one learned intent→realization piece is the
span head's *choice* of features; the step below it was never a skill. So δ_perf on RHM has never
included the spelling step, and the E′ gate-seat nulls (`tacet`, `intonation`/`ma_s0`) and F2's
unassigned trust-vs-habit were read inside that condition. With execution perfect, trust could only
ever be proposal mass — which is why it has been indistinguishable from habit. (What this does
*not* explain: the address/value-currency results, and E1's null, which is about the update.)

**What `tempo/` found on the plant, and what carries.** A chunk stored as forces transfers to no
tempo; stored as a path and converted by a body model it transfers three octaves (`rubato/k1`).
Read precisely: the transfer used the *inverse* direction only; the forward model was derived from
the same fitted inverse with no new parameters (decision 31), and its own contribution (reads, ε
tolerance) is not isolated from the PD term — no arm corrects against the raw stale read. Three
things carry. (i) The factoring itself is what `canon` gave RHM for free, so this node asks
whether a learner *arrives* at it when it is not free. (ii) Finding 6 — the body model
identifiable only when practice spanned two tempi — carries as a curriculum claim, with contexts
practiced in place of tempi. (iii) The bound — no executor reached the verbatim tape — carries as:
the renderer takes over spelling and nothing else; *which* chunk stays an arbitrary binding in the
table (§3½ with one more layer assigned). What does not carry: dead-reckoning between stale reads,
since RHM has no delay. Where the forward direction may re-enter is the reader monitoring the
learner's own output — recognition doing production monitoring, at the price of a read.

**The reader.** A rendering rule is one object used in two directions: the executor applies it
(features + context → surface), the reader undoes it (surface + context → features). On the plant
the second direction was derived from the first; here it cannot be, so whether the two organs
share parameters is measurable rather than algebraic. And it gives the practice stack the organ
`temperament`'s diagnosis found missing — "no cortex for identity to drain into as recognition."

## The one change

Below the tables, so every organ above level 1 carries over unmodified — the tables, `Miner`, π,
the corridor head, the crank, the gauges, the mirror loop. Two things move:

1. **The bottom coin becomes a rule.** The synonym index at a level-1 node is a function of a
   context variable rather than a uniform draw.
2. **The grader marks spelling.** Success on meaning as now (the possible set), plus a spelling
   error reported beside it: the fraction of bottom nodes written with a synonym the rule did not
   call for. Two numbers, never one.

Everything else is imported. The rule at a trivial setting (constant synonym 0, spelling weight 0)
must reproduce `tutti` bit-for-bit; that is the fork gate.

## Decisions left open, with a recommendation

Persisted because the design step is the experiment. Each is a place the implementer may find a
better answer; the recommendation says where to start, not where to end.

1. **Which rule: register or agreement.** *Register*: one variable per sequence, visible to the
   performer the way tempo is, selecting the spelling pattern everywhere — tempo's twin; it can be
   the era knob and gives a transfer test to an unpracticed register. *Agreement*: the spelling at
   a node depends on a neighbour (the parent's choice, the sibling's feature) — conjugation; more
   language-like, generalization to unseen (chunk, context) pairs, no era ladder.
   **Recommendation**: register first, agreement as the second rung. The register must be readable
   from the observation (a prefix token, or inferable from any visible block), so the executor's
   information is the performer's and not the experimenter's.
2. **What the rule is made of.** Any learner with the matching inductive bias transfers by
   construction; that is not the test. The test is whether a *generic* renderer, fitted from the
   learner's own successful productions, identifies the rule from few contexts, and whether the
   curriculum governs it. So the rule must be low-dimensional and shared across features — the
   analogue of two body constants. **Recommendation**: the simplest structured family that admits
   held-out contexts (e.g. `k = (offset_f + r) mod m`), sized against the constraint that m = 4 was
   measured inadmissible at v = 8 (`tall/`; `tutti/sizing/SIZING.md` premise 1). Either widen v or
   find a rule with held-out contexts at m = 2. Q0 settles it; do not guess.
3. **The grader: graded or strict.** *Strict*: a wrong spelling fails the instance. *Graded*:
   meaning succeeds, spelling is marked. **Recommendation**: graded, two numbers. It keeps the
   outcome currency and the execution currency separate — the two-δ split made live at the
   spelling step. Mining can still key on meaning while δ_perf reads spelling; a strict grader
   folds them into one channel, the shared-channel failure `two_clocks` measured.
4. **Shared or separate morphology.** The executor's renderer fitted from its own productions,
   separately from the reader; or one object serving both directions. **Recommendation**: run
   both. The plant's answer was "one object" by derivation; here it is the closest thing this
   substrate has to the cortex/cerebellum split, and it is a question.
5. **Where it sits.** Beside `enharmonic`, not inside. `temperament`'s type-law question should run
   unconfounded on the current substrate. Once both exist, `temperament`'s arms import onto this
   fork: the quotient drains the spelling of a *category* upward, the rule moves the spelling of a
   *leaf* downward, and the reader is what both drain into.

## Arms (sketches; per repo norms no outcome is interpreted in advance)

- **canon** — the current substrate through the forked code with the rule trivial. The fork gate
  (0.000e+00 against `tutti`) and the anchor.
- **given_rule** — the rule handed to the executor and the grader. The ceiling; `tempo`'s
  `kin_oracle`.
- **leaf** — chunks committed and keyed as *leaf* strings rather than level-1 features, executed by
  replay. The force head's twin: content stored in realization coordinates. Asks what a rule costs
  a learner that does not factor.
- **fit_rule** — feature-keyed chunks (as now) plus a renderer fitted from the learner's own solved
  productions in the practiced contexts. The treatment.
- **fit_shared** — as `fit_rule`, with the reader and the renderer one object. Decision 4's
  second arm.
- **The curriculum ladder** — `fit_rule` with contexts practiced ∈ {1, 2, 3}, one context always
  held out. Finding 6's twin.

## Readouts

- **Transfer**: in-band execution (meaning and spelling) in the held-out context, per level,
  against `leaf` and `given_rule`.
- **Identifiability**: the fitted renderer's recovered rule parameters against the true rule, by
  contexts practiced — the drag-coefficient readout.
- **δ_perf at the spelling step**: the 2×2 (executed-as-intended × solved) gains a spelling column;
  whether the executor-plasticity and commit-pacer seats (`two_deltas` (a)) move when `e` includes
  spelling.
- **Trust vs execution reliability** — the F2 probe: π's per-slot mass against the renderer's
  parity in that context. With a perfect renderer these could not separate; here they can.
- **The reader**: parse accuracy of level-1 features under the rule, by context, and whether it is
  vocabulary-gated — the climbing-reader question, now expressible on the practice stack.
- **Arrival above level 1 unchanged** — a check, not a finding: flat-key sizes at L2–L5 are
  untouched by a bottom rule.

## Sequence and gates

- **Q0 — size offline first** (`ostinato`'s discipline): rule candidates against the admissibility
  constraint; how parse ambiguity changes under each; whether the frozen generator parses under
  the rule or must be re-pretrained; the fraction of current successes a strict grader would
  reject; the flat-key sizes above L1. No GPU.
- **Q1 — the fork at level ≤ 3**: `canon`, `given_rule`, `leaf`, `fit_rule`; the parity gate as the
  readout. Asks whether a fitted renderer transfers at all and what `leaf` costs. ~2–4 GPU-h.
- **Q2 — the curriculum ladder**, and `fit_shared`.
- **Q3 — the crank on the fork**, `tutti`'s mirror loop imported, the E′/F2 readouts live.

Gates: the fork bit-identical at 0.000e+00 with the rule trivial · the graded grader at spelling
weight 0 reproduces `possible_sets` exactly · Q0 before a GPU is spent.

## Norms

Fork donors, never edit them; every prior result stays byte-reproducible. Volume, difficulty mix,
priced budget and lifetime held fixed across arms; only the rule, the key and the renderer move.
Seed policy per [`experiments/CLAUDE.md`](../../../CLAUDE.md). Modal per `/run-experiment-on-modal`;
an implementer subagent reads `/subagent-instructions` and never touches git. Smoke first at every
step. Results are discussed before a README is written; `QUEUE.md` / `ROADMAP_PROGRESS.md` on
landing.

[^private]: Not mirrored: this link points to a document in the private lab repo (the roadmap, the queue, an unrun spec, reading notes, or a conversation). See the top-level README for what is held back and why.
