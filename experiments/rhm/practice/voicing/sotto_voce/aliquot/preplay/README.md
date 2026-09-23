# preplay — the level reader prices abstractions and does not rank spellings: the shaped plant's projection, read offline on banked state, at the chooser's cell and at the candidate entry's

**Up**: [`../README.md`](../README.md) (aliquot, the three in-loop rounds this node reads the dumps of) ·
**Decisions, gates and defects**: [`NOTES.md`](NOTES.md) (pp1, pp2, pp4 and pp5) · [`NOTES_own.md`](NOTES_own.md) (pp3) ·
[`within/NOTES.md`](within/NOTES.md) (the within-context readout) · **Machinery record**: [`FILES.md`](FILES.md) ·
**Conversation**: `CONVERSATION.md`[^private] (the session behind the six rounds, 2026-09-21) ·
**Child**: [`within/`](within/README.md) (within-context candidate discrimination on the six soundboard dumps, CPU).
**Motivation**: [`ideas/calibration_and_violation_are_one_object.md`](../../../../../../../ideas/calibration_and_violation_are_one_object.md)
§12.4, the replay note: a candidate that exists as a table entry has no price until the world model is put into the
state it implies and the projection is read there; and the open question the parent left, why a better read did not
make a better choice.
**Runs**: 2026-09-21; `wi2` (CPU, three passes of eight containers), `pp1` (0.12 GPU-h), `pp2` (0.14 GPU-h), `pp3`
(0.09 GPU-h), `pp4` (0.14 GPU-h), `pp5` (0.10 GPU-h); nothing paid in the loop, every arm read from the six banked soundboard arms and overtone's two frozen
plants; seeds 0 and 2 throughout and never averaged. **Ranks, signs, located mechanisms and per-cell counts are the
claims.**
**Attribution**: the worry that the lineage's machinery had grown beyond the scope of the question and might be
confounding it, the ask to move reader questions off the loop, the question whether retiring the in-loop arms would
add oracle dependence, the go for the selector and for the learner's own entries, the ask to read the code directly,
and the ask to explain the findings simply are Jasper's, as are the go for the two consumer rounds and the reading,
relayed from a side chat, that preplay's role is to feed the value system rather than to be scheduled by it and that
replacing the oracle's "did the score get worse" with the readout's own level was the point of value all along. The reading of the nested DP parse against the producer-
versus-reader hypothesis, the design of the two questions and of the fire-once-read-thrice comparison, the two
controls added to `within` (the never-trained twin and the diet contrast at the projection's own form), and the
synthesis are the orchestrator's, agreed in discussion. The builds, the gate tables, the exact replay of the
learner's tables from the banked keys and picks, and every withdrawal beside its correction are the three
implementers' (one Opus session each).

## One-liner

The parent put a linear projection of the practice plant's state in the grader's seat of the practice loop and found
that shaping the plant on the outcome made the verdict linearly readable but did not move the chooser, and offered a
reading in which shaping had cost the executor its prior. This node asks the two questions the loop could not, on the
banked plants, readouts and rows, with nothing paid in the loop. **The projection does not rank a context's candidate
spellings** (0.65 / 0.70 within context, below the executor's own prior at 0.71 / 0.74), **and the executor's prior did
not erode**: it reads the same on the shaped and the frozen plant, and the composed chooser ranks within context at
0.76 / 0.82, above both of its organs. What the projection is weak at is located: a never-trained trunk under the same
readout ranks within 0.03–0.07 of the shaped one, the diet is worth +0.03 to +0.05 at the projection's own form, and
the form itself, one readout pooled over the whole configuration, is the largest axis. **The same projection prices a
candidate abstraction.** Fired through the shaped executor on fresh pools, one table entry at a time, its mean level
over the fired configurations separates true from wrong entries at 0.61–0.75 against the world's 0.89–0.97, tracks the
world's price within the legal class on 23 of 24 cells, and reads at chance through a frozen or never-trained trunk.
As a selector it recovers a median 0.65 of the world's advantage over a random draw where unshaped reads recover
0.2, matches the full true table at L3, and fails where a few attractive wrong entries capture the executor's argmax.
On the learner's own mined entries, replayed exactly from the banked keys and picks, the world itself barely
separates true from false at L3/L4, and the read tracks the world's price where the executor's own prior prefers the
false ones. The level reader is the right object for pricing abstractions and the wrong one for choosing spellings. Two consumer
rounds followed on the loop's own try-and-keep extension gate. **The read is not a scheduler**: with the gate fixed, the
executor's own score is the fastest order in both settings and the read ties a never-trained trunk. **The read is a
gate**: with the producer's score as the order and the read's level on the preplayed state in place of the world's
count of repairs, the table it builds captures 0.27 of the oracle's advantage over no gate at the loop's own budget and
0.57 at the full walk, while the frozen and twin readouts in that seat are worse than no gate. Order by the producer,
keep by the value reader: the fully endogenous consumer, at zero world queries.

## Children

### [`within/`](within/README.md) — within-context candidate discrimination on the six soundboard dumps (CPU)

**Goal**: every AUC in the parent is pooled over rows and so mixes "is this context solvable" with "which of this
context's candidates is the one to write". Group the banked probe rows by context, the slot plus the pre-write masked
configuration, and read each of the chooser's organs, the grader's filed probability and five fitted controls on the
pairs inside a group only.

**Finding**: the composed chooser ranks within context at 0.76 / 0.82 held out; the projection is the weakest of the
run's four readers at 0.65 / 0.70 and the prior reads the same on the shaped and the frozen plant; the critic taught by
the projection reads 0.66 / 0.70 where the critic taught by the world reads 0.87 / 0.84; a never-trained twin under a
per-slot readout ranks at 0.80 / 0.79 against the shaped trunk's 0.83 / 0.85; the filed diet carries no within-context
contrast at all. Facts in [`within/figures/within_reduction.txt`](within/figures/within_reduction.txt); §1 below.

## The question

The parent's three rounds (`aliquot`, `duplex`, `soundboard`) established that a prediction-trained plant's projection
does not read the outcome, that an outcome head whose gradient reaches the trunk beside the infill loss makes the
verdict linearly readable, and that the chooser did not follow the read. The parent's reading of the last fact was
that the plant is both the producer the executor's DP scores through and the thing read, and that shaping it for
reading had cost its prior, on the strength of a fourteen-point drop in the block head's single-mask parse. The
nested DP parse, the instrument the executor actually scores through, was flat within 0.04 at every level on both
seeds under both-loss shaping and down 0.10–0.13 at every rung in the collapse arm, so that reading was not
supported by the executor's own instrument, and the two in-loop arms queued to test it (the infill-only control and
the reader-copy arm) were retired before this node ran.

Two questions remained, and both are questions about a representation and a readout rather than about the loop.
Within a fixed context, does the read rank the candidates as the world does? And at the level the arc is about,
does the read price a candidate abstraction, an entry that exists only as a table row, the way §12.4 says a value
reader must: by firing it and reading the level of the state it produces? The loop's own audition does exactly that
firing with the world in the reader's seat, and this node puts the projection there instead.

Vocabulary, inherited from the parent and used here. The **plant** is the 2-layer masked-infill encoder the loop
trains; the executor's **DP** scores candidate spellings through its block logits, and that score is the chooser's
**prior**. The **projection** is the parent's readout of the plant's pooled state over a configuration, one linear
readout per goal, fit by IRLS on the learner's own experienced configurations; **shaped** means the plant trained on
an outcome head beside its infill term; **frozen** is overtone's unshaped plant of the same lineage at the same seed
and the **twin** a never-trained trunk of the same architecture. The **critic** is the chooser's judge, a small net
over the context and a candidate, and the **composed chooser** standardises the prior and the critic over the
candidate set and sums them. A **candidate entry** is a proposed table row at level ℓ, a pairing of two level-(ℓ−1)
rows; a **true** entry is one of the grammar's own and a **wrong** one composes two true lower rows the grammar
does not license. The **audition** of a table applies its macro to fresh damaged instances of the level's cell and
counts the world's repairs; **firing** a candidate is that step with the projection read on the fired configurations
beside the world's grade. Everything runs at the **operative** state of each arm, the plant, readout and tables the
run ended with.

## What was built

Two offline nodes in the shape of `duplex`, importing leaf helpers and never forking the loop. [`within/within.py`](within/within.py)
scores the six soundboard dumps and overtone's two on CPU and [`within/reduce_within.py`](within/reduce_within.py)
groups and reduces them. [`preplay.py`](preplay.py) re-implements the projection's read in forty lines and gates it
elementwise against the parent's own class (F-2, `max|Δp| = 0.000e+00` on all six arms), records the executor's DP
while firing (F-1, bit-identical to `MC.apply_any` on every level of every arm), and fires candidate entries on fresh
pools; [`selector.py`](selector.py) builds tables from each selector's ranking and audits them on disjoint pools;
[`learner_tables.py`](learner_tables.py) replays the learner's own operative tables from the banked keys and picks;
[`own.py`](own.py) prices and selects over them. Every gate in every script has been shown to fail before being
reported (`falsify`, 7/7; `within::falsify`, 10/10 required; `falsify2`, 6/6; seven reconstruction cases on all six
arms). The whole node is about 4,300 lines and 0.35 GPU-hours against the parent's 19,000-line fork and 17 GPU-hours,
and the parent's in-loop round could not pose either question.

## 1. Within a context, the projection does not rank the candidates, and the prior did not erode (`within/`, `wi2`)

A group is the slot plus the pre-write masked configuration, since the probe rows carry no trajectory id; the
within-context AUC is the rank AUC restricted to pairs inside one group, pair-weighted; held-out rows are held out
from the critic's training by the same code. Per arm 42k–51k probe groups, 170–604 held-out groups carrying both
world classes, 1576–6552 pairs, median group size 3. Held-out probe rows, all levels, the verdict-shaped arms:

| reader, within context (pooled on the same rows) | seed 0 | seed 2 |
|---|---|---|
| the projection's filed probability | 0.651 (0.637) | 0.697 (0.653) |
| the executor's prior | 0.709 | 0.738 |
| the critic, taught by the projection | 0.661 | 0.698 |
| the composed chooser | 0.756 (0.728) | 0.820 (0.785) |
| overtone's frozen plant: the prior · the critic taught by the world · the chooser | 0.724 · 0.871 · 0.845 | 0.771 · 0.836 · 0.850 |

The prior reads the same on the shaped and the frozen plant; the chooser is above both of its organs on every arm;
and the one column that separates the lineages is the critic, which reads what it was taught. The filed diet cannot
be asked the question at all: 22525 of 22526 and 22192 of 22192 filed groups carry one and the same candidate, because
the chooser is an argmax, and the bank the projection is fit on holds 11–34 contexts with two candidates and
different verdicts in 30000 rows.

Five fitted controls on the same held-out rows locate the projection's weakness. A per-slot ridge readout of the
final shaped trunk over the masked post-write configuration, fit on counterfactual rows against the world's verdict,
ranks at 0.831 / 0.852; the same over a never-trained twin, 0.797 / 0.786; the same fit on filed rows, 0.598 / 0.573. At
the projection's own global form, fit on counterfactual rows 0.704 / 0.762 against filed rows 0.671 / 0.709, a diet
contrast of +0.03 to +0.05 on all six shaped arms (+0.12 to +0.30 on the frozen plants). The per-slot form beats the
global form on the same trunk and diet by +0.05 to +0.14. So the representation is not what limits the projection
at this cell, the diet is a small part, and the form, one readout pooled over the whole configuration, is the largest
axis; a filed-diet fit at the projection's own form reads where the run's own projection read.

## 2. Fired one at a time, the projection prices a candidate entry in the world's order (`pp1`)

Each candidate alone as a one-row table over the true lower table, fired through the arm's shaped executor on 256
fresh instances of the level's era cell, world-graded, and read on the same fired configurations by the banked
projection, by the same readout refit through the shaped, the frozen and the twin trunk on one shared subsample of
the arm's own bank (so the three differ only by the trunk), and by the executor's own DP score of the entry it
wrote. The read's price is its mean over the fired instances; the world's is the repair rate. AUC separating true
from wrong entries, seed 0 / seed 2:

| level | world | banked shaped read | frozen, same readout | twin | executor's DP |
|---|---|---|---|---|---|
| L2 | 0.949 / 0.958 | 0.747 / 0.666 | 0.488 / 0.498 | 0.529 / 0.503 | 0.620 / 0.626 |
| L3 | 0.890 / 0.889 | 0.637 / 0.711 | 0.587 / 0.589 | 0.475 / 0.483 | 0.594 / 0.543 |
| L4 | 0.941 / 0.953 | 0.680 / 0.630 | 0.535 / 0.607 | 0.535 / 0.482 | 0.678 / 0.640 |
| L5 | 0.969 / 0.953 | 0.645 / 0.609 | 0.568 / 0.601 | 0.515 / 0.516 | 0.588 / 0.601 |

Shaped above frozen above twin on 8 of 8 cells. Spearman of the read's price against the world's over candidates:
shaped 0.17–0.53, frozen 0.03–0.29, twin −0.07–0.12. Within the true class alone, where every candidate is legal
and only its fit to what the cell's instances wanted differs, the shaped read's correlation is positive on 23 of 24
cells across all six arms (seed 0 0.24 / 0.45 / 0.47 / 0.58, seed 2 0.71 / 0.32 / 0.26 / 0.09), and it beats the
executor's own score there on 5 of 8 cells of record. The level is compressed against the world's (seed 0: 0.145–0.153
on true entries against 0.093–0.105 on wrong, where the world reads 0.17–0.22 against 0.00–0.03), and per fired
instance the read transfers at 0.58–0.73 from 0.82 / 0.86 on the learner's own configurations; frozen and twin do not
separate the two classes at all. On the bank's own held-out rows the three refits read 0.823 / 0.854, 0.750 / 0.712 and
0.730 / 0.714, the parent's frozen-equals-random finding re-derived offline.

The delta form, the loop's own base ∪ {candidate} against base question at the committed table sizes, is
under-powered for the world itself: the executor's DP never picks 92–99% of added candidates at L3–L5, so their price
is identically zero by construction and the effective n is 2–76 of 288. No claim rests on it. The no-infill arms read
candidates at 0.60–0.68 through a plant whose executor audits the full true table at 0.92 / 0.77 at L4 against 0.16 /
0.18 on the verdict arms, the parent's readout-versus-competence dissociation on the audition distribution.

## 3. In the selector's seat, the read builds a better table than a random draw, with one trap (`pp2`)

Candidates ranked by each selector on a pricing pool (n 512) and the top-k built as a table, audited by the world on
three disjoint test pools (n 256), at k = the arm's committed size and half and a quarter of it, from scratch and as
an extension of a quarter-size base. The world's audition error of the built table at the committed k, seed 0 /
seed 2, lower is better:

| level | world | banked read | executor's DP | random draw | the full true table |
|---|---|---|---|---|---|
| L2 | 0.363 / 0.361 | 0.661 / 0.535 | 0.490 / 0.543 | 0.854 / 0.543 | 0.315 / 0.345 |
| L3 | 0.267 / 0.349 | 0.299 / 0.339 | 0.327 / 0.400 | 0.395 / 0.745 | 0.299 / 0.339 |
| L4 | 0.453 / 0.168 | 0.974 / 0.168 | 0.974 / 0.168 | 0.974 / 0.168 | 0.176 / 0.168 |
| L5 | 0.681 / 0.587 | 0.681 / 0.587 | 0.691 / 0.587 | 0.681 / 0.812 | 0.130 / 0.697 |

At L3 the read's table audits at the full true table's error on both seeds, and at smaller k it beats the world's own
greedy top-k (0.270 and 0.272 against 0.409 and 0.564 at k = 30 and 15, seed 0), since ranking by individual price
concentrates on redundant entries. At L2 the executor's own score is the better selector. At L4 seed 0 every selector
but the world sits at random's error: 29 of 88 admitted wrong entries take a table of 59 true ones from about 0.3 to
0.97, because a table's audition is set by which entries win the executor's argmax. At L5 seed 0 the world's own
advantage over random is nil and nothing can be read. Over the 24 arm × level cells, wrong entries admitted as a
ratio to a uniform draw's: world 0.17, banked read 0.77, refit 0.79, executor's DP 0.84, random 0.94, frozen 0.96,
twin 0.97; the capture fraction of the world's advantage over random, median over the 161 well-posed cells: banked
0.65, refit 0.72, frozen 0.22, twin 0.21, executor's DP 0.14.

## 4. On the learner's own mined entries, the world barely separates true from false, and the read tracks its price where the prior fails (`pp3`)

The arm's operative tables at L2–L4 replay exactly from the banked `keys_at_support` and `last_build.picks`, the
recursion closing on the base alphabet: row counts, the logged true-table precision and the banked per-row truth
mask agree row for row on all six arms, a digest of the closed table asserted inside every container. L5's last build
is empty and the frozen committed tables bank only sizes and masks, so neither is reconstructible. Each row alone,
fired and read as in §2, seed 0 / seed 2 at L2 / L3 / L4:

| | true-from-false AUC | Spearman with the world's price, all rows |
|---|---|---|
| the world's own price | 0.868 / 0.583 / 0.540 · 0.885 / 0.613 / 0.641 | — |
| banked shaped read | 0.611 / 0.544 / 0.584 · 0.660 / 0.587 / 0.643 | 0.119 / 0.434 / 0.336 · 0.497 / 0.557 / 0.566 |
| frozen, same readout | 0.597 / 0.566 / 0.583 · 0.660 / 0.587 / 0.621 | 0.260 / 0.355 / 0.242 · 0.377 / 0.492 / 0.402 |
| twin | 0.681 / 0.507 / 0.491 · 0.647 / 0.472 / 0.503 | 0.128 / 0.190 / 0.085 · 0.458 / 0.232 / 0.226 |
| executor's DP | 0.403 / 0.448 / 0.514 · 0.359 / 0.399 / 0.464 | 0.083 / 0.288 / 0.101 · 0.066 / 0.220 / 0.365 |

The learner's false entries were mined from its own solved trajectories and repair instances nearly as well as its
true ones at L3/L4, so the grammar's truth is a weak proxy for their value there and the world's price is the target.
The executor's own score prefers the false entries, below chance at separating them, which is how they were mined;
the frozen trunk is not at chance here as it was on constructed candidates. As a selector at the committed size every
read prunes the learner's table to within 0.03 of the world's choice and of the whole table (seed 0 L3: world 0.277,
banked 0.291, frozen 0.312, whole table 0.313, a random draw 0.365); at a third of that size the banked read is best
at L3 seed 0 (0.277 against the world's greedy 0.505) and worst at L2 seed 0 (0.792 against the executor's 0.329). At
L4 the whole table's audition equals its best single row's on both seeds (0.1875 / 0.1768), and the frozen, twin and
executor top-1 find that row while the banked read's does not (0.615 / 0.608). No selector enriches its top-k for true
rows above the base rate at L3/L4.

## 5. The read is not a scheduler: with the loop's own gate fixed, the executor's score is the fastest order (`pp4`)

The loop's own consumer of candidate entries is `census_extend`: walk the candidates in an order, audition
base ∪ {candidate} on a fresh pool, admit iff the error does not rise (`extend_tol` 0.0), grow the base on admission,
one audition per candidate, at most eight per pass, on a pool of 192. pp4 transcribes that gate, asserts it identical
to the loop's own on real auditions at the cap, at the full walk and at a positive tolerance, holds it fixed, and
varies only the order: the banked shaped read's price, the world's price, the executor's `dp_top`, the frozen and twin
readouts, two random permutations, and in setting (b) the operative table's own emission order (the loop's count
order is not reconstructible; the per-key counts are not banked). Three disjoint pool families, pricing (512), gate
(192, two repeats) and test (256, three). Setting (a) is a quarter-size random base of the true table with pp1's
candidates; setting (b) the learner's own operative rows from an empty base. Auditions spent to come within 0.02 of
the world order's final error, median as a fraction of the full walk, with the share of cells that arrived:

| order | constructed candidates, 16 cells | the learner's own rows, 12 cells |
|---|---|---|
| executor's `dp_top` | 0.038 (16/16) | 0.112 (12/12) |
| the world's price | 0.056 (16/16) | 0.397 (12/12) |
| frozen readout | 0.104 (14/16) | not run |
| banked shaped read | 0.188 (15/16) | 0.214 (11/12) |
| never-trained twin | 0.191 (16/16) | not run |
| random permutations | 0.291, 0.521 | 0.262, 0.474 |
| the operative table's emission order | not run | 0.740 (9/12) |

At the loop's own cap of eight auditions on the learner's rows at seed 0 L3, `dp_top` reaches 0.426 where the world's
order reaches 0.577, the read 0.779 and random 0.576 / 0.707; at sixteen `dp_top` 0.289, the read 0.428, the world
0.577. The gate's own contribution is separate from the order's: the same read ungated (pp2's top-k) goes 0.329 →
0.346 → 0.431 → 0.562 as the budget grows at seed 0 L2 while the gated walk stays at 0.276 → 0.281 → 0.293 → 0.294. At
the full walk the orders' finals agree within about 0.005 in setting (a), since at zero tolerance the gate admits
nearly everything that does not hurt; the effect lives at small budgets. At a tolerance of 0.01 the final table is
equal or worse in all 28 cells with more wrong entries admitted, so the loop's zero is right. The mechanism reads
off the columns: the gate's waste is auditions on entries the executor never uses, and `dp_top` is the executor's own
score, so the entry it offers first is the entry the executor will write; the read ranks by whether the resulting
state solves, which says nothing about whether the executor will reach it.

## 6. The read is a gate: its level on the preplayed state in place of the world's verdict (`pp5`)

The same walk with `dp_top` as the order throughout and the gate varied: the world's gate as the reference; the
banked shaped projection's level over the same fired gate-pool configurations, admit iff its mean does not fall (a
paired form over the changed instances beside it, and a margin form at the base level's standard error); the same
level through the frozen core and the twin as controls; and an ungated walk as the floor. Handed the world's error
as its level, the read gate reproduces the world gate exactly in all three forms (P-2). The world's test-pool error
of the table each gate built, median over the 28 cells, with capture = (ungated − gate) / (ungated − world):

| gate on the preplayed state | error at the loop's budget of 8 | capture | at the full walk | capture |
|---|---|---|---|---|
| the world's count of repairs | 0.418 | 1.000 | 0.331 | 1.000 |
| the banked shaped read's level | 0.460 | 0.273 | 0.393 | 0.568 |
| the read with a margin | 0.470 | 0.091 | 0.456 | 0.123 |
| no gate | 0.475 | 0.000 | 0.474 | 0.000 |
| the twin's level | 0.568 | −1.614 | 0.548 | −0.514 |
| the frozen core's level | 0.584 | −1.898 | 0.594 | −0.841 |

The world's gate admits about 95% of what it is offered, so admitting everything agrees with it 95.5% of the time and
raw agreement is not evidence; the composition of the disagreement is. The read agrees 93.8%, letting in junk on
0.022 of candidates against the ungated 0.045 and refusing good entries on 0.050; the frozen and twin readouts let in
the same junk (0.021) and refuse good entries on 0.089 and 0.064, which is why they end below no gate at all. The
read gate matches the world's exactly in 8 of 28 cells and is clearly worse at L3 on both seeds (0.743 against 0.603,
0.755 against 0.552 at the full walk). The paired form's admissions are identical to the pooled form's in every cell,
and necessarily: the pooled difference is the paired difference times the changed share, a positive factor, so the
two can differ only under a margin, and the margin as built is worse than the strict form at both budgets.

## The update

1. **The executor question is closed.** The prior reads the same on the shaped and the frozen plant within context,
   the composed chooser ranks above both of its organs, and the nested DP parse was flat under shaping. Shaping did
   not cost the producer. What sets the chooser is the critic, and what sets the critic is the verdicts it is taught:
   a level reader's probabilities teach it to 0.66, the world's verdicts to 0.87. The reader-copy arm stays retired.
2. **The projection is a level reader, by form.** Within a context it is below the prior; a never-trained trunk under
   the same readout nearly matches the shaped one, so the representation is not the limit; the filed diet's lack of
   contrast is a small part; the global pooled form is the largest part. This is `duplex`'s "shaping wrote a level,
   not a ranking" located at the cell it was a claim about, with the mechanism measured.
3. **A level reader is the right object for pricing an abstraction.** Fired entry by entry, the shaped projection's
   mean level tracks the world's audition, within the legal class and on the learner's own mined rows, and the
   shaping is the ingredient on constructed candidates (frozen and twin at chance). It is weak against the world,
   compressed, and degrades off its diet, which is §12.4's rider measured.
4. **A table's audition is a max, not a sum.** The executor writes the entry it likes most, so a few attractive wrong
   entries capture a table and a single best row can carry a whole level. Selecting by individual price is therefore
   the wrong consumer above L2, and the loop's own incremental tolerance gate, which tries one candidate at a time
   and keeps what does not hurt, is the right one. The reading offered at the time, that the read's seat is the
   order in which candidates are tried, was tested by pp4 and corrected: the order belongs to the producer and the
   read's seat is the verdict (items 7 and 8).
5. **On the learner's own entries, correct is not the value.** The world prices the learner's false entries nearly as
   high as its true ones at L3/L4, the executor's own prior prefers them, and the shaped read tracks the world's price
   better than any other read. That is the currency the roadmap's endogenous grader has to read, and it is not the
   grammar's truth.
6. **Three instrument facts for the lineage, and two more from the consumer rounds.** The loop's composition rule divides the prior by the span and then
   standardises, so the division is inert; the banked core and readout in `vo_heads.pt` are one plant update out of
   step (0.0005–0.0066 of held-out AUC); and the learner's operative tables are exactly replayable from the banked
   keys and picks at the last cycle, so no dump line was needed for them. A walk gate has to be falsified on a designed
   sequence, because on a pool where the base is already the minimum every wrong rule coincides with the right one;
   and the paired form of a level gate is the pooled form at zero threshold by identity.
7. **The read is not a scheduler, and should not be.** With the loop's own gate fixed, the executor's own score is the
   fastest order in both settings, twice as fast as the read on the learner's own rows and faster there than the
   world's own greedy order, and the read ties a never-trained trunk. The gate's waste is auditions on entries the
   executor never uses, and only the executor's score knows what it will use. Which candidate to preplay first is a
   question about the producer's search, not about value, and the read was never the right organ for it.
8. **The read is a gate, and shaping is what makes it one.** Order by the producer's score and keep by the read's
   level on the preplayed state, and the table captures a quarter of the oracle's advantage over no gate at the loop's
   own budget and over half at the full walk, with zero world queries, by halving the junk admitted at the price of
   refusing one good entry in twenty. The unshaped readouts in that seat refuse good entries at nearly twice that rate
   and end worse than no gate. That is the fully endogenous consumer the arc was after: the producer says what to try,
   the value reader says what to keep. The remaining gap is the read's noise as a difference of two compressed levels,
   worst at L3, and the seat inside the loop is untested.

## What this does not show

Nothing here ran in the loop; whether ordering the loop's own incremental audition by the read turns the crank
faster is the queued question. `within`'s projection column is the probability filed mid-run over the final
configuration while the prior and critic are the final core over the write-time context; the filed-diet fit at the
projection's own form removes both handicaps and reads within 0.02 of it. A context in `within` is a configuration,
not a trajectory, at a measured cost of 0.8–1.4% of repeated cells disagreeing on the verdict. `pp1`'s and `pp2`'s
candidates are constructed over the grammar's true lower table; `pp3`'s are the learner's own but only at the run's
last state and only at L2–L4. The frozen comparator is overtone's plant, a different run at the same seed. The
executor's own DP score, not the projection, is the best selector at L2. L4/L5 held-out cells in `within` are thin
and carry no weight. pp4's learner-side order is the operative table's emission order, not the loop's count order,
which is not reconstructible; at zero tolerance and a full walk the order cannot matter, so pp4's effect lives at
small budgets. pp5's read gate is worst at L3 on both seeds, its margin form is worse than its strict form, and its
paired form is its pooled form by identity. Neither consumer round ran inside the loop, and the two arms of record
are the only arms they ran on. Nothing off the RHM practice substrate.

**Correction, 2026-09-21, later the same day.** §5 and §6 call `census_extend` "the loop's own consumer of candidate
entries". It is the census lineage's op over a frozen committed table, and the arms of record never ran it: `sb_sv_yk`
at both seeds carries `recert` but not `extend`, runs `open_inventory`, and logged zero extension events. On those arms
the operative table is the miner's live support-count build over the operative lower table, rebuilt on 157 of 201
cycles at seed 0, and no decision consumes a world audition of the table (the commit is by clock yoke with its audition
logged only; the recert is measured and never acted on). pp4 and pp5 therefore transcribe an op these plants did not
run, in the setting those plants' tables are in (setting (b), the learner's own rows from an empty base). The in-loop
test is an admission set on the live build ([`../sostenuto/`](../sostenuto/README.md)).

## Reproduction

```bash
cd experiments/            # MODAL_PROFILE=chromatic
# within: the CPU scoring pass over the six soundboard dumps and overtone's two, then the reduction
modal run rhm/practice/voicing/sotto_voce/aliquot/preplay/within/within.py::falsify              # 10/10
modal run --detach rhm/practice/voicing/sotto_voce/aliquot/preplay/within/within.py::score --out-tag wi2 --arms all
python3 rhm/practice/voicing/sotto_voce/aliquot/preplay/within/reduce_within.py --root <dumps> --scores <scores> \
    --out rhm/practice/voicing/sotto_voce/aliquot/preplay/within/figures --arms all
# preplay: gates, the exact fidelity gate, the harness, the three tags
modal run rhm/practice/voicing/sotto_voce/aliquot/preplay/preplay.py::gates
modal run rhm/practice/voicing/sotto_voce/aliquot/preplay/preplay.py::fidelity_gate --arms s0_sv,s2_sv,s0_so,s2_so,s0_yd,s2_yd
modal run rhm/practice/voicing/sotto_voce/aliquot/preplay/preplay.py::falsify                     # 7/7
modal run --detach rhm/practice/voicing/sotto_voce/aliquot/preplay/preplay.py::sweep --out-tag pp1 --arms s0_sv,s2_sv,s0_so,s2_so,s0_yd,s2_yd
modal run --detach rhm/practice/voicing/sotto_voce/aliquot/preplay/selector.py::sweep2 --out-tag pp2
modal run --detach rhm/practice/voicing/sotto_voce/aliquot/preplay/own.py::sweep_own --out-tag pp3
python3 rhm/practice/voicing/sotto_voce/aliquot/preplay/reduce_preplay.py --tag pp1 --fetch
python3 rhm/practice/voicing/sotto_voce/aliquot/preplay/reduce_selector.py --tag pp2 --fetch
python3 rhm/practice/voicing/sotto_voce/aliquot/preplay/reduce_own.py --tag pp3 --fetch
# the two consumer rounds: the read as the order (pp4) and as the gate (pp5)
modal run rhm/practice/voicing/sotto_voce/aliquot/preplay/incremental.py::gates4
modal run rhm/practice/voicing/sotto_voce/aliquot/preplay/incremental.py::falsify4                # 4/4
modal run --detach rhm/practice/voicing/sotto_voce/aliquot/preplay/incremental.py::sweep4 --out-tag pp4
modal run rhm/practice/voicing/sotto_voce/aliquot/preplay/readgate.py::gates5
modal run rhm/practice/voicing/sotto_voce/aliquot/preplay/readgate.py::falsify5                   # 5/5
modal run --detach rhm/practice/voicing/sotto_voce/aliquot/preplay/readgate.py::sweep5 --out-tag pp5
python3 rhm/practice/voicing/sotto_voce/aliquot/preplay/reduce_incremental.py --tag pp4 --fetch
python3 rhm/practice/voicing/sotto_voce/aliquot/preplay/reduce_readgate.py --tag pp5 --fetch
```

The exact entrypoint names and flags per script are in each script's docstring and in `results/`; every tag and app
id is in [`FILES.md`](FILES.md). Volumes: `rhm-scaling-data:/rhm_practice_within/<tag>/`,
`/rhm_practice_preplay/<tag>/`; the dumps read are `/rhm_practice_soundboard/<tag>/<arm>/` and
`/rhm_practice_voicing/{ov_s0b/ovt_comp_pr_sh, ov_s2/ovt_comp_pr_dis}`.

## Next steps (queued in `QUEUE.md`[^private], not started)

The read gate inside the loop, re-aimed (queue (viii), in flight as [`../sostenuto/`](../sostenuto/README.md)): an
admission set on the live build the arms of record actually run, gated by the world, the read's strict pooled level, or
nothing; see the correction below · the delta form at a smaller base ·
the chooser's grader seat with a ranker rather than a level reader, a design step · one dump line in the next fork for
L5's build, the frozen tables' rows and the miner's per-key counts, and a `vo_heads.pt` written after a final refresh.

## Files

| file | purpose |
|---|---|
| `README.md` | this writeup, the super-node for `within` and `pp1` to `pp5` |
| `preplay.py`, `pool.py` | the fire-and-read job (pp1): the projection's read re-implemented and gated, the recording DP, the candidate sets, `gates`, `fidelity_gate`, `falsify`, `sweep`; the loop's instance generator copied and gated on text identity |
| `selector.py`, `reduce_selector.py` | the read in the selector's seat (pp2) and its reducer |
| `learner_tables.py`, `own.py`, `reduce_own.py` | the exact replay of the learner's operative tables, the pricing and selection over them (pp3), and its reducer |
| `incremental.py`, `reduce_incremental.py` | the read as the order for the loop's own gate (pp4) and its reducer |
| `readgate.py`, `reduce_readgate.py` | the read as the gate with the prior as the order (pp5) and its reducer |
| `reduce_preplay.py` | pp1's reducer and the table of record `figures/preplay_reduction.txt` |
| `NOTES.md`, `NOTES_own.md` | decisions, gates and defects for pp1, pp2, pp4 and pp5, and for pp3 |
| `figures/` | the tables of record (`preplay_reduction.txt`, `select_reduction.txt`, `own_reduction.txt`, `incremental_reduction.txt`, `readgate_reduction.txt`), the per-arm mirrors under `pp1/` to `pp5/`, the replayed tables `<arm>_operative_tables.json` |
| `results/` | the commands of record and launch logs |
| `within/` | the within-context readout ([`within/README.md`](within/README.md)) |

[^private]: Not mirrored: this link points to a document in the private lab repo (the roadmap, the queue, an unrun spec, reading notes, or a conversation). See the top-level README for what is held back and why.
