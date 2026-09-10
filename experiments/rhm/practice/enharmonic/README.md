# enharmonic — the quotient: re-keying the vocabulary by category, and earning the key

**Up**: [`../README.md`](../README.md) (practice). **Spec**: [`SPEC.md`](SPEC.md) (2026-09-01) ·
Q2: `temperament/SPEC.md`[^private] (2026-09-09) · the third child:
`figured_bass/SPEC.md`[^private] (2026-09-10). **Files**: [`FILES.md`](FILES.md).
**Facts**: [`sizing/SIZING.md`](sizing/SIZING.md), `figured_bass/sizing/SIZING.md`[^private],
`figures/en_s{0,1,2b,3,4,5}_reduction.txt`, `figures/fb_s{0,1}_reduction.txt`,
`figures/en_s{3,4,5}_alias_audit.txt`. Written 2026-09-11 after discussion; the round ran
2026-09-09 → 09-11, ≈43 GPU-h in total.

## Children

| folder | what |
|---|---|
| [`sizing/`](sizing/SIZING.md) | Q0, offline: the cover, the token class, the alias audit, the forced-transfer budget, junk under transfer; the fork's own mining path simulated. |
| `temperament/`[^private] | Q2's spec — the endogenous quotient as the one op with negative within-level and positive next-level value. Its arms live in this node's fork; four passes, `en_s2b` → `en_s5`. |
| `figured_bass/`[^private] | Commit the key, not the content: what a commit should freeze in category coordinates. Q0 offline; `fb_s0` on its own clock, `fb_s1` on the anchor's. |

## The question

Every node in the practice arc keys its macro table by the flat spelling — the exact string of
level-1 features a chunk expands to — and builds level ℓ+1 by looking each half up in level ℓ's
rows. `tutti/sizing` found the flat key doubly exponential in level (14 / 56 / 816 / 205,824
/ 1.3e10 at L2…L6) and L5 unreachable at the run's observation budget by ~100×, while the
grammar has 16 rules per level. The parent spec asked whether the learner can key its table by
*category* instead — C♯ and D♭ are one pitch — and reach the rung the flat key cannot afford:
first with the categories supplied (Q1), then earned from the learner's own evidence (Q2).
`temperament` sharpened Q2 into a test of the roadmap's type law: merging spellings loses
distinctions at the current level and pays only one rung up, so it should be the one op a
within-level reader refuses and a next-level reader licenses. `figured_bass` was added mid-round
from Q1's mechanism, asking what a commit should freeze once the key is a category.

## What was built

[`enharmonic.py`](enharmonic.py) forks `tutti/tutti.py` (every addition `# [enharmonic]` or
`# [figured_bass]`; the fidelity gate G-F replays `tutti` at 0.000e+00 with every knob off, re-run
after every change to the shared path — nine times this round). [`quotient.py`](quotient.py)
defines what a category is in the fork and `ClassMiner`: class-keyed observe and build with the
execution table materialised as the cross-product of held spellings, so `macro_features` is
untouched and singleton classes reproduce the flat miner bit for bit (gate E-0). The oracle
supplies a class only in the `given_cat_*` arms, contained and billed. [`merge.py`](merge.py) is
the merge as a third outer-loop action beside commit and advance: a forced-transfer probe
(`fourwall.entry_profile` at the level's own damage cell, one representative per class) closes
within-tolerance pairs into alias groups, and a licence takes or refuses each group; every
proposal logs every currency and floor whichever one decides. Clock yokes (`flat_yk_*`) replay a
treated arm's realised commit and advance cycles from the banked tag with the flat key, the
arc's control for a loop that owns its own clock. The deletion battery gained an honest regrowth
column (`built_fresh`, each level built over the condition's own fresh chain); the inherited
`built` column, in every battery since `native`, builds over the arm's own undeleted lower table
and is uninformative under table deletion. `analyze_enharmonic.py`, `alias_audit.py` and
`fetch_compact.py` reduce, audit and mirror; per-arm mirrors are compact (the entry recorder's
probe-phase vector over the 262,144-row true L5 table had made a raw arm 300 MB).

## Q0 — what the world and the banked logs say (offline, no GPU)

Over 59 banked arms (the `Miner.build` replay reproduces all 150 commit events), at `rule_seed = 0`:

1. **The true categories are a cover, not a partition.** Two bottom rules share a spelling, so
   a tuple can carry several parent features: 2 / 8 / 128 / 37,120 multiply-parented keys at
   L2–L5, holding 23–34% of the true mass at every mining node.
2. **The identifiable category is the token class** — the set of level-ℓ features a key's
   canonical rendering can legally derive. It is closed under composition (alphabet 7 / 9 / 11
   / 13 / 22 / 42 at L1–L6; 73 legal L5 class pairs against 205,824 flat keys), and a
   forced-transfer probe measures it exactly: 397 of 397 groups of rows sharing a class had
   bit-identical transfer profiles. Its resolution ceiling is the demand at the node (7 of 9
   resolvable at L2, 9 of 11 at L3, 11 of 13 at L4), not the sample count.
3. **The beam use record cannot serve as an alias criterion.** No use statistic clears a
   shuffled control in the merge-useful direction; the cells that do clear it have the opposite
   sign, because the DP's argmax picks one of two identical programs and starves the other
   (below 1% of the winner's mass in 68% of such pairs).
4. **Forced transfer is cheap**: 20–450 gradings partition a whole committed book, 4–130 with
   use-share weighting; equal-class pairs sit at loss exactly 0 and subset pairs at a median 0.43.
5. **Most "junk" is relabelled truth.** 68–95% of committed rows the feature-space oracle marks
   junk are legal programs read through the last-writer-wins inverse map; the genuinely
   off-grammar minority repairs nothing (0 of 101) and sits alone. The junk *mass* at the node
   is only ~36–47% relabelled, so the aleatoric channel is real; it is the arc's committed-row
   precisions (0.15–0.50) that were mostly a label. Every precision here is reported in both
   spaces.

The fork's own mining path, simulated: read through the reader and re-rendered canonically, only
23.5% of L5-node halves stay legal, so the 63-of-70 arrival figure is an upper bound.

## Q1 — the supplied quotient (`en_s0`, `en_s1`; 3 GPU-h)

Three arms on the mirror loop with L5 open: `flat`, `given_cat_tok` (token class fed),
`given_cat_min` (the token class forced to a single label — the over-quotient control, since a
learner's productions have no generative label). Then two clock yokes.

- **Arrival at L5 is solved by the key.** `given_cat_tok` carried 37 L5 and 34 L6 keys at
  support where `flat`'s gauges were identically 0 for its whole life; `given_cat_min` committed
  L5 at c99. The flat comparator failed on its thermostat, not its build: on the treated arm's
  replayed clock the flat key installed a one-entry L5 table.
- **The un-yoked era-4/5 gap of −0.40 was the clock.** The loop paces itself and the arms lived
  93 / 176 / 114 cycles; at matched clocks the gap is within ±0.05 and changes sign once.
- **The wall moved to the L4 book's class coverage.** The yield-quiet commit froze the L4 book
  at c95 holding 3 of 13 classes on era 3's narrow stream; the live miner reached 9 the L5 build
  never saw. The flat key has little to lose from freezing (1 → 4 classes); the class key loses
  most of what it was for.
- **The finer key delays its own commit**: `given_cat_tok`'s L6 arrival kept rising (29 → 34
  over its last 20 cycles) so its commit gauge never quieted; `given_cat_min`'s saturated at 16
  and committed.
- The battery's table-free next-level stream survives table deletion and falls under corridor
  deletion (L5 keys at support 10/10/11/7/4 per era under `a_full`, 7/7/1/0/0 under `b_span`).

## Q2 — the endogenous quotient, four passes (`en_s2b` → `en_s5`; ≈19 GPU-h)

| pass | change | endo arms reach | licences |
|---|---|---|---|
| `en_s2b` | one pair per 8 cycles, use-share-weighted probe, key-count rise / ledger at level ℓ | L2 only | ledger licensed everything (graded the level a merge cannot change) |
| `en_s3` | whole-partition groups; ledger at ℓ+1; mass rise | L3 (31 rows vs yokes' 23) | agree; both blind on half the auditions (`x_differ = 0`) |
| `en_s4` | every class probed (use filter off); ledger defined against an absent ℓ+1 table | **L4@c179, 100 rows, 6 of 13 classes** | ledger takes all six L4 groups; mass rise exactly 0 there |
| `en_s5` | expected arrival over the remaining schedule; the choice instrument | L3 (28 rows), no L4 | the deferred gauge is stricter than the instantaneous one |

**The evidence route works and is cheap.** In every pass both real licences merged only pairs the
demand cannot distinguish: precision 1.000 against the demand partition, and 1.000 in feature
space from `en_s3` on, at 0.06–0.3% of priced time. The forced over-merge ablation is 0.000
against demand (0.53 in feature space). No group ever chained the book into one class.

**What throttled it, twice, was the use record.** At one pair per eight cycles the op took 1 of
the 5 alias merges available in its L2 book (`en_s2b`). With whole-partition groups the probe
was shown only the rows carrying 90% of the beam's use — under argmax latching the winners —
while 2 to 64 alias pairs per proposal sat among the starved rows it never saw (`en_s3`; the
audit is exact against the run's own counts). With every class probed (`en_s4`) both arms adopt
L3 with 42 entries against their yokes' 23 and L4 with 100 rows against 8, and the L4 book at
commit holds 6 of 13 token classes where the supplied quotient held 3 at its own commit. L5 was
not committed in any pass: the L4 adoption at c179 left 21 gated cycles of L5 mining.

**The licence question resolved the opposite way from the spec.** A merge of entries changes
nothing at the level it acts on — a level-ℓ merge only re-keys level ℓ+1 — and at ℓ+1 an
equal-class merge never costs (`e_merge − e_keep` = −0.003 to −0.001, never worse) while a
forced subset or disjoint merge never helps (+0.017 / +0.027). On roughly half the auditions the
two tables write the same state on every instance: the argmax winner survives the merge. So the
within-level ledger, aimed at the level a merge changes, licenses every correct merge on the
grounds that it costs nothing, refuses the forced ones, and was the licence that reached L4. The
"negative within-level value" `fourwall` measured belonged to merging contexts, whose programs
differ; entries in one token class are interchangeable by construction.

**No instantaneous next-level gauge saw the merge's value at the frontier.** Four currencies were
tried. The key-count rise refuses a coarsening for coarsening (two at-support keys collapse into
one). The mass-at-support rise is monotone and fired where the next level's stream was rich
(L2 → L3) but read exactly 0 at the c188 L4 proposal where the probe found 720 alias pairs. Key
buildability cannot fire by construction — a merge never adds a class to the book — and c188 was
an inventory event (one L5 key going from 1 row to 16), not an unblocking. Expected arrival over
the remaining schedule was *stricter* than the instantaneous gauge: over a thousand observations
the below-support keys arrive anyway, so pooling looks worthless, and that arm froze L3 at 28
rows and never adopted L4. The horizon that matters is the window before the level above
freezes, which the loop decides later. The merge's one-level-up value is real (the c188 groups
took the L5 build from 1 to 16 entries) and deferred past any gauge the learner computes at the
decision.

The two endogenous arms' error series were bit-identical across all 201 cycles in `en_s3` and
`en_s4` despite taking different merges (7 vs 6; 15 vs 8): the installed books coincided, and
execution does not consult the partition.

## figured_bass — commit the key, not the content (`fb_s0`, `fb_s1`; 4.8 GPU-h)

Two knobs: `open_inventory` (an adopted level's operative table is the live build; the commit
mints slots and starts the corridor but freezes nothing) and `ungate_l5` (the committable L5
miner mines every cycle instead of only from era 4).

- **The committable L5 miner is era-gated**, so no book moves L5's first build (c156 under
  every regime, offline and in-run). Ungating it kept the anchor's clock and loop actions exactly
  and filled the miner with 33 keys at support, of which the frozen 3-class L4 book blocked 29.
- **On its own clock the open inventory re-paced the loop.** Both open arms lived 66 cycles
  against 176: bit-identical to the twin through c31, one true new L2 class arrived at c31, the
  corridor's disagreement with its now-moving intention rose, δ-silence's moved-then-quiet latch
  armed on the 5th cycle of era 2 and fired on the 6th, where the frozen twin's series never
  crossed the floor in 46 readings and rode the caps. The loop re-arms only on loop actions, and
  under an open inventory the table changed on 35 of 66 cycles without one.
- **On the anchor's clock the open book did what it was built to do, and cost performance.**
  It reached exactly the 9 of 13 L4 classes Q0 predicted, cut the L5 block rate from 88% to
  35%, is token-precision 1.000 at every size (240 rows in 9 classes: redundancy, not junk — the
  feature-space fall to 0.10 is the relabelling), and was adopted by the corridor without visible
  cost (parity 0.950 vs 0.968, misfire 0.033 vs 0.027, zero empty fallbacks, priced budget within
  269 groundings of 42.4M). It is worse than the frozen anchor in both consumption eras (+0.016 /
  +0.079), and no arm committed L5.

## The expansion-choice instrument (`en_s5`)

Per audition-path macro call, whether the chosen row's token class contains the feature the clean
derivation had at that node. At the frontier the endogenous book's choices beat the flat yoke's
(4n3 era 4: 0.566 vs 0.442 contain; 5n1 era 5: 0.800 vs 0.514, success 0.641 vs 0.317) while the
two are indistinguishable at the levels both hold. The open book's choices are slightly worse
than the frozen anchor's on the same clock at both frontier nodes (0.630 vs 0.671; 0.604 vs
0.650). Success exceeds containment at 2n12 in era 1 (0.707 vs 0.523): the clean latent is a
stricter reference than the any-legal-repair grader, so the next version should reference the
set of features that repair the instance.

## The update

1. **The second extension is a change of key, and the key is earnable.** Arrival at L5 is
   solved by the token class, supplied or earned; the learner finds its own synonyms exactly
   with a priced probe at under a third of a percent of its budget, once the probe is allowed to
   see the rows the beam never uses.
2. **The merge is within-level invisible, not negative**, like commit and advance, because
   entries in one class are interchangeable and the executor never used the loser. What
   `fourwall` found negative was merging contexts. The type law's "refuses rather than starves"
   framing was wrong for this op; the op is mis-typed, not the law.
3. **Its next-level value is deferred past any instantaneous read.** Arrival gauges see a merge
   only where the next level's stream is already rich; at the frontier the value is
   representability at a freeze the loop has not yet chosen. The working licence on this
   substrate is a cost check at the level the merge changes, with next-level arrival as the
   readout of value.
4. **What stops L5 now is the clock, three ways**: the L4 book freezes on era 3's narrow
   stream; the committable L5 miner starts at era 4; opening the inventory re-paces the loop.
   In category coordinates the object to freeze and the schedule to mine are the open questions,
   not arrival.
5. **A richer vocabulary is capped by the chooser.** An all-legal 9-class book performs worse
   than a 3-class one at matched clocks, and its frontier choices are slightly worse. The
   executor's expansion choice — the generator's logits, not a learner — is where
   `inflection`'s question arrives from this side. Suggestive, and the instrument's reference
   needs the fix above before it carries weight.
6. **Two things the arc had been carrying were labels.** The committed-row precisions (0.15–0.50)
   were mostly the inverse map's collisions; the battery's growth column never measured regrowth
   under deletion.

## What this does not show

Matched-clock era-4/5 error gaps between near-identical arms reach 0.08 in this round (the two
flat yokes in `en_s4`/`en_s5`, differing by an 8-row L4 adoption, read 0.694 vs 0.776 in era 4),
so the error readouts are not load-bearing; the structural ones — books, class coverage,
adoption cycle, merge precision, choice accuracy — are consistent across every pass. The choice
instrument sees the audition path, not the beam's own search states. Every number is on one
rule draw (`rule_seed = 0`); seeds 6 and 10 are collision-free and would make the cover a
partition. The ledger's "absent ℓ+1 table" branch and the merge on an open book past a commit
(`endo_open`) were built and gated but never exercised by a run.

## Reproduction

From `experiments/`, profile `chromatic`: the commands of record are
`enharmonic/results/RUN_en_s{0,1,2,2b,3,4,5}.sh` and `enharmonic/figured_bass/results/RUN_fb_s{0,1}.sh`;
each carries the flags, floors and banked-arm provenance in its header. Offline:
`PYTHONPATH=. python3 rhm/practice/enharmonic/sizing/phase0_cat.py`,
`.../sizing/fork_arrival.py`, `.../figured_bass/sizing/phase0_open.py`,
`.../alias_audit.py --tag <tag>`, `.../analyze_enharmonic.py --tag <tag> --bank TAG:ARM,...`.
Gates: `enharmonic.py::fidelity_smoke`, `::preflight --outdir-tag <tag> --arms ...`,
`quotient.py` (E-0), `merge.py` (M-0).

[^private]: Not mirrored: this link points to a document in the private lab repo (the roadmap, the queue, an unrun spec, reading notes, or a conversation). See the top-level README for what is held back and why.
