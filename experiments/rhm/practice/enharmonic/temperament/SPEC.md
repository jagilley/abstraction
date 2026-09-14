# SPEC — temperament: the endogenous quotient, and why the merge is the purest one-level-up test

**The question in one sentence**: can the practice learner license its own merges — collapse
committed tuples it has found interchangeable into one category, on its own evidence, under a
value system that reads next-level yield — when the merge is *worse* in the within-level currency
and pays only one rung up; and does the table it drains this way carry the next rung?

**Status**: spec, 2026-09-09. Nothing run, nothing built. **Parent spec**: [`../SPEC.md`](../SPEC.md)
(`enharmonic`, 2026-09-01) — this is its Q2, specced. Q0 (offline sizing) and Q1 (`given_cat`,
the supplied quotient) are this node's prerequisites and ceiling, not separate work. Read the
parent first; nothing there is repeated here. The name: equal temperament is the tuning that makes
enharmonic spellings one pitch — it gives up the purity of any single key to make every key
reachable.
**Machinery donors** (all untouched): [`../../tutti/tutti.py`](../../tutti/tutti.py) (the mirror
loop; head of the fork lineage) · [`../../fourwall/`](../../fourwall/README.md) (merge / re-key /
retire on one level with a supplied basis — the ops exist) ·
[`../../antiphon/questions.py`](../../antiphon/questions.py) (selectors, quota) ·
[`../../ratchet/macros.py`](../../ratchet/macros.py) (`Miner.observe` keys table-free;
`Miner.build` looks each half up in the lower table's flat rows — the one place the ratchet is
enforced by the index rather than by the learner) · [`../../native/`](../../native/README.md)
(π's (level, node) slots; the corridor head) ·
[`../../tutti/sizing/phase0_l5.py`](../../tutti/sizing/phase0_l5.py) (`buildable()`, level
arithmetic).
**Idea docs**: [`recurrence_manufactures_confounds`](../../../../../ideas/recurrence_manufactures_confounds.md)
§5 (the merge op's constraints), §7 (a belief is a fully-merged chunk) ·
[`practice_manufactures_its_own_credit`](../../../../../ideas/practice_manufactures_its_own_credit.md)
§3½ (identity in the table, corridor in the executor; the hippocampal profile, 08-20 revision) ·
[`sparse_one_rung_up`](../../../../../ideas/sparse_one_rung_up.md) §2 (density is a property of
(domain, level)) · [`absorption_blinds_the_evaluator`](../../../../../ideas/absorption_blinds_the_evaluator.md)
§4.2 (the meter's type must match the op it grades).
**Attribution**: Jasper's (session `015cXn4gUFPEV3Qx6N9RBFsU`, 2026-09-05→09): the question
whether the external macro store maps onto the hippocampus; the correction that re-internalization
*has* run on both substrates, with the suspicion that "we aren't draining the buffer in the way we
ought to be"; and the call to spec the tie-in. The readings that the flat key *is* the undrained
buffer, that the merge is worse within-level and pays only one rung up, and that this makes it the
purest test of the type law, came out of the exchange. Conversation record:
`conversation_2026-09-09.md`[^private].

## Why this node exists, in one paragraph

Every one-level-up test the arc has run had the within-level ledger merely *blind*. Commit and
advance carry no within-level value, so a within-level reader neither funds nor refuses them; it
starves the next level by omission (`teacher_slot`, `conductor`). The merge is different in kind.
Collapsing several committed spellings into one category *loses distinctions* at the current level
— a flat entry that was exactly right becomes one of several the executor must choose among — and
gains nothing there. Its whole value is that `T[ℓ+1]` becomes representable at `v·m = 16` entries
instead of 205,824 at L5. `fourwall` measured the consequence on a supplied basis and on a real
NTP reader: the within-level ledger *votes against merging at all*, and the endogenous reader
*tracks and never quotients*. So the merge is the one op in the arc with negative within-level
value and positive next-level value, which makes it the sharpest available test of the roadmap's
central claim (§1.2, §2.1): the value of level-ℓ work is denominated in level ℓ+1's currency and
invisible to any within-level signal. Here it is not invisible; it is negative. Does a value
system reading next-level yield license what the ledger refuses, and is that the right call?

## The drain reading (framing, not a claim)

The 08-20 revision to §3½ assigned the mined table the hippocampal profile: an arbitrary binding
stored fast, populated offline by selection, needed for growth and not for performance.
Re-internalization ([`native`](../../native/README.md), [`spiral`](../../spiral/README.md),
[`solo`](../../../../mjc/practice/solo/README.md)) then drained every role but one: routing into
π, the corridor into the executor, the table deletable at ~zero cost — and the next level still
built by `Miner.build`'s lookup into the lower table's flat rows. The observation stream is
already table-free (`native` finding 5: +0 under deletion); only the build is not. A table keyed
by spelling, growing without bound and never becoming a class, is what an undrained index looks
like; a category the learner holds itself is what consolidation into a schema means. The sizing
wall and the undrained buffer are one fact seen from two sides. Iwane et al. 2026's fixed chunk
*count* (~7) with growing chunk *size* is the biological version of a store that recycles its
slots as content consolidates; ours has no eviction. F3's `retire` is the eviction and has never
run. None of this is a prediction about what the runs will show.

## What the learner has, and what it must supply

**Evidence**, both on machinery that exists. The **use record**: the beam's per-entry selection
mass, logged per cycle; synonymous tuples are the ones the beam uses interchangeably in the same
slot (the alias audit — sizeable offline against the true categories, with a shuffled control,
before anything is built). **Forced-transfer probes**: substitute tuple B for A in a context A
solves, one grounding per probe, graded by the world's exact grader (the question port's bisection
form). The parent's two lessons travel: weight probe selection by solvability (the band-pass), and
size the probe budget on the solve tax the meter exposes.

**The op**: merge as an outer-loop action beside commit/hold and advance, under the mirror loop
(yield commits, δ-silence advances). *What licenses it* is the treatment variable.

**What a category is in the learner.** The parent left this open and so does this spec; the
candidates differ in *where the category lives*. An equivalence relation kept beside the flat
table (cheapest; the representation is the experimenter's, the evidence and the licence are the
learner's). Or a minted category token occupying a π slot the way a macro does, with the executor
choosing an expansion at execution time (the learner holds the class; `can't-decompose` becomes an
affordance). The second is the drain reading's strong form; the first is enough to ask the
type-law question. An implementer may find a third.

## Arms (sketches; per repo norms no outcome is interpreted in advance)

- **flat** — the mirror loop as run (`tutti` `tu_s0`'s winning assignment), clock-yoked to every
  treated arm. The anchor.
- **given_cat** — Q1: the true categories fed. Ceiling, and the arrival-dominates control: is a
  *partition* giftable, or only earnable (`census`/`assay`'s question about a new object).
- **endo_yield** — the learner's own evidence; the merge licensed by the one-level-up yield gauge.
- **endo_ledger** — the identical evidence and op, licensed by the within-level ledger. `fourwall`
  says this arm should refuse. Whether it does, and what refusing costs at L5, is the contrast.
- **If cheap**: a merge-precision ablation (junk forced into a category — the parent's thread 6;
  the analogue of `ratchet`'s early-commit foreclosure). And a **table-free builder in flat
  coordinates** — `Miner.build` looking up halves in what the learner proposed or executed rather
  than in the lower table — which separates "draining helps because the key changed" from
  "draining helps because the builder reads the learner."

**Readouts**: merge precision against the true partition (the oracle grades; feeds only
`given_cat`) · |T5| at support, and whether L5 *arrives* at the existing budget · the signature
(`crescendo`'s value clock past the certified range) · π's per-slot mass across a merge (a merge
changes what a slot proposes; what trust does is unknown and worth watching) · the deletion
battery with **growth** measured, on the quotiented table — the drain test proper.

**Gates**: every fork bit-identical at 0.000e+00 with its knobs off · `given_cat` with singleton
categories reproduces `flat` exactly · the alias audit's offline purity against its shuffled
control before a GPU is spent · Q0's sizing first (rule collisions at `rule_seed = 0`;
category-coordinate level sizes; probe budget).

## Norms

Volume, difficulty mix, priced budget and lifetime held fixed across arms; only the key and the
licence move. Size offline before building (`ostinato`'s discipline). Seed policy per
[`experiments/CLAUDE.md`](../../../../CLAUDE.md): one seed first, more only if that file's
conditions hold. Modal per `/run-experiment-on-modal`; an implementer subagent reads
`/subagent-instructions` and never touches git. Every merge event logged with what licensed it
(the parent's hardest norm: a merge is irreversible the way a commit is). Results are discussed
before a README is written; `QUEUE.md` / `ROADMAP_PROGRESS.md` on landing.

**First pass landed (2026-09-10, `en_s2b`)**, facts only, interpretation pending discussion:
`../figures/en_s2b_reduction.txt`. The evidence route holds (forced transfer at the level's own
cell; demand-partition precision 1.000 at 0.16–0.32% of priced time); the yield licence reads a
*mass* rise, not a key count; the ledger arm as ported graded the level a merge cannot change;
one pair per 8 cycles left most of the partition on the table. Second pass (`en_s3`) in
`QUEUE.md`. Machinery: [`../FILES.md`](../FILES.md).

**Second and third passes landed (2026-09-11, `en_s3`, `en_s4`)**, facts only:
`../figures/en_s{3,4}_reduction.txt`, `../figures/en_s{3,4}_alias_audit.txt`. Whole-partition
merges; the ledger audition on the level a merge changes; the use filter found to hide the
aliases (argmax latching) and removed; with every class probed both endogenous arms adopt L4
(100 rows, 6 of 13 classes at commit) and beat their lifetime-matched yokes in the consumption
eras; the ledger takes correct merges and refuses forced ones, the arrival gauge is silent at
the frontier. Interpretation pending discussion.

**Fourth pass (`en_s5`)**: the deferred-arrival licence was stricter than the instantaneous mass gauge and starved the L3 book; the key-buildability gauge cannot fire by construction; facts in `../figures/en_s5_reduction.txt`.

**Fifth to seventh passes (`en_s6` → `en_s9`, 2026-09-11→12)**, discussed and written up in
[`../README.md`](../README.md) (addenda): the composed arm; the row cap covering the book; the
merge sweeping every live level per proposal, which lets the L4 partition collapse before the L4
commit once the level above is ungated (the closed arm's book must freeze unmerged — the licence
at ℓ+1 is undefined while the gated table is empty); the re-arm hook off the commit latch, which
commits L5 at c186. Facts in `../figures/en_s{6,7,8,9}_reduction.txt`.

[^private]: Not mirrored: this link points to a document in the private lab repo (the roadmap, the queue, an unrun spec, reading notes, or a conversation). See the top-level README for what is held back and why.
