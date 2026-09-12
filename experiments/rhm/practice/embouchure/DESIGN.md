# DESIGN — `embouchure`: the decisions Q0's numbers forced

Companion to [`FILES.md`](FILES.md) (the facts) and [`SPEC.md`](SPEC.md) (the question). Written
after Q0, before `embouchure.py` exists. Decisions only; the numbers behind each are in
`FILES.md` §Q0 and `figures/q0_reduction.txt`.

---

## 0. What Q0 changed about the node's shape

The SPEC put the question as *source* against *volume*: does the renderer trained on the
learner's own metered attempts differ from the one trained on the world's surface, and what
does the source cost against the volume. Q0 measured four things and three of them collapse
the *per-block* half of that question:

1. **The two channels carry the same label.** `k_rule` in `spell_rows` is `K[feature,
   register]` in every one of 100k+ banked rows; the surface's label is the synonym the world
   wrote at a world block, which is the same `K[feature, register]`. On this substrate the
   template and the map are one function — the SPEC's own hypothesis, now with a number.
2. **They are drawn on the same instances at the same three registers.** Both are read off the
   same cycle's practice batch, which `context_instances(..., practiced=practiced)` restricts
   to {0, 3, 7}. Neither channel visits a held-out register at all.
3. **Neither is volume-limited.** All 24 practised cells arrive in cycle 1 at both volumes, so
   `SIZING.md` §2's ceiling (det 0.4750 / acc 0.7875 for the scalar head at {0, 3, 7}) is
   reached at c1 either way; and the offline fit lands on the identical table
   (acc_practised 1.0000, acc_held-out 0.8750, θ̂ [2,2,5,5,5,5,5,2]) from either channel, at
   full volume and at 0.2812 of it, in 8 of 8 seeds. Cutting the volume 3.6× moves the first
   cycle at the endpoint from 8–14 to 11–17 and moves the endpoint not at all.

What Q0 did **not** collapse: the bag-level channel (`own_scalar`), whose label is a scalar per
instance rather than a per-block target and whose bag membership is not banked; and the
*position* of own writes within the answer, which the log does not record.

## 1. `own_verdict` is dropped from Q1

The SPEC's own conditional — "`own_verdict` if Q0 says it differs from `surface`" — resolves to
no. Two independent offline fits (`fit_rule_s` rows and `leaf_s` rows) reach `surface`'s exact
recovered table, and the identifiability ladder is saturated in cycle 1 for both. Running it as
a live arm would buy a second copy of `fit_rule` at 3 GPU-h.

It is kept **reachable** as a `render="own"` / `own_label="verdict"` configuration so the
ceiling can be re-derived on a world where the two channels can differ (§5), and it is not an
arm in the Q1 launch.

## 2. The read-back arm: Q0 said dead by construction, RB-1 says otherwise

**This section was written from Q0 and is corrected by measurement.** What it said, and why it
was wrong, is kept in full rather than replaced, because the correction is the finding.

**What Q0 argued.** `rule_seed 6` has **0 collisions in 16 legal bottom codes**, so every
synonym of a feature is owned by that feature alone and an *exact* reader returns the intended
feature whether or not the spelling is the rule's. The observational evidence agreed: in the
four `fit_*` arms the `agood` row count equalled `min(n_solved·32, 512)` in all 557 cycles, so
not one of their 359 misspelled own-write blocks was read back to a different feature; and
`read_acc` is 1.000 at all eight registers. The conclusion drawn — that a misspelling has no
sensory consequence through the reader and the self-supervised read-back arm is dead by
construction — was **inference from an exact-map argument plus observation, not intervention**.

**What RB-1 measured** (`em_sm`, production reader, `read_acc` 1.000, `base_read_wrong`
0.0039). Take a clean rule-spelled configuration at register ρ, flip ONE block to the other
synonym of its own feature, re-read:

| | pooled | ρ=0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|---|
| flipped block's read CHANGES | **0.5625** | 0.719 | 0.609 | 0.766 | 0.609 | 0.703 | 0.313 | 0.344 | 0.438 |
| flipped block's read becomes WRONG | **0.5586** | 0.719 | 0.609 | 0.766 | 0.609 | 0.688 | 0.297 | 0.344 | 0.438 |
| collateral (untouched blocks) | 0.0029 | | | | | | | | |

**A misspelling DOES have a sensory consequence through the reader — over half the time.** The
exact-map argument is sound about an exact reader and the frozen reader is not one: it is a
transformer over the whole sequence, and a lone off-register synonym inside an on-register
sequence reads as a different feature more often than not, while its neighbours are essentially
untouched (0.3 % collateral). The reader is register-sensitive in a way nothing in `inflection`
measured — its finding 5 listed "reading the register off the surface" as a job the reader does
**not** do, and this is evidence in the other direction.

**Why the two disagree, stated rather than smoothed over.** They are different conditions. RB-1
flips one block in an otherwise perfect sequence — the maximally out-of-distribution case. The
arms' own misspellings are systematic (`canon` and `leaf` write a consistent wrong spelling
across the whole answer, so the sequence is off-register coherently) or rare (`fit_*` misspell
~0.5 % of blocks, in era 1 only). The measured middle case is now on the record too:
`own_scalar_s`'s own written blocks fail read-back at **0.0104** over 60 smoke cycles
(`log["own"]["readback_drop"]`), because its writes are mostly right and mostly coherent.

**Consequence, after the coordinator read this section (2026-09-11): the arm is built.** The
claim "a misspelling is invisible to the learner's own reader on this world" is **withdrawn**;
`FILES.md` §Q0.3's numbers stand as what they are (observations on the arms' own answers), and
RB-1's number is what an intervention says. `own_readback_s` runs in `em_q1b` beside `em_q1`
— see §2b.

## 2b. `own_readback`: the retraction route

The only arm in the arc with **no grader number in its learning signal at all**. It writes a
word, reads its own answer back with its own frozen reader, and learns from whether the word
gave back the meaning it meant.

- **The label** is per block: `w = 1{read-back feature ≠ intended feature}`, from one no-grad
  reader forward on the completed answer. Nothing else crosses into the fit.
- **The loss** is `own_scalar`'s with `w` per BLOCK in place of `y` per bag: same
  `q = p` (synonym 0 written) or `1 − p` (synonym 1 written), same direction — `w = 0`
  reinforces the word used, `w = 1` pushes away — same 64 steps, same batch in bags. With `w`
  absent the function is `own_scalar`'s branch exactly (gate EB-5), so this is an added branch
  and not a changed one.
- **The intent** is RECALLED, not re-read: the learner emitted `bottom[f, k]`, so the form it
  left identifies (f, k) wherever the lexicon is injective (`own_recall_table`). A form with
  two owners is excluded from the fit and counted (`n_ambiguous`) rather than tie-broken —
  which is inert on `rule_seed 6` and is exactly the machinery Q2's lexicon will make bite.
- **The pricing decision, logged either way**: `blk_readback` counts the blocks the reader
  passed over, `blk_readback_used` the blocks whose verdict entered the fit, and `priced:
  False` rides beside them. The read-back is **not** priced this round, for the same reason the
  fit is not: pricing one route's instrument and not another's moves the clock instead of the
  signal. Both own arms pay the same forward, so the counter is comparable across them.
- **Gate (a), the coordinator's**: replace the label with the GRADER's per-block verdict and
  the objective becomes exactly `BCE(σ(logit), K[f, ρ])` — `own_verdict`'s supervised fit — so
  the head must recover K. **EB-4 PASS**, and it lands on Q0's `own_verdict` table to the
  digit: acc_practised 1.0000, acc_held-out 0.8750, det 0.375, θ̂ [2,2,5,5,5,5,5,2].
- **Gate (b)**: the read-back confusion per (feature, register), an 8×8 tally per cycle in
  `log["own"]`, so the pooled 0.5625 / 0.4375 split RB-1 measured is visible per cell.
- **The one asymmetry against `own_scalar`**, recorded rather than smoothed over: `own_scalar`
  keys its cells on the reader's parse, `own_readback` on recall. It has to — a label asking
  "did the reader give back what I meant" is undefined if the cell's feature is what the reader
  said. Where the two sources differ IS the label, so it is the signal and not a confound; the
  rate is logged (`cells_reader_agrees`) and Q0 measured it at 0.0104 here.

## 3. `own_scalar` is the node's remaining arm, and it needs one new instrument

The bag-level channel is the only one carrying information the surface does not. Its two
mechanisms:

**(a) The bag must be the learner's own writes, not the graded configuration's blocks.**
`grade_spelled`'s `spell["per_row"]` has **all 32 on-grammar blocks** as denominator, of which
Q0 measures only **0.2812 (≈ 9)** to be own writes — so 72 % of the label's denominator is
noise-free world spelling and the label's dynamic range is compressed ≈ 3.6×. `Renderer.tally`
already computes the exact written-block verdict with no write mask (`inflection` DESIGN §6);
what it does not do is keep it **per row**. `# [embouchure]` addition: a per-row written tally
on the Renderer (`tally_rows()`, a `bincount` over the row index the cursor already has in
hand, cleared by `reset_tally`), so `own_scalar`'s bag and its label share a denominator and
`surface_matched` has an exact per-cycle count to match to.

**(b) The loss.** The simplest principled bag-level objective, and the one that will be built:
per instance i the renderer wrote a bag `B_i` of (feature, register) cells and the meter
returned `y_i` = (wrong writes) / (writes) on that instance. The head emits `p_fk` = P(synonym
1) per cell; the bag's predicted error is `ŷ_i = mean_{(f,r) ∈ B_i} |p_fr − 1{the write was
synonym 1}|`, and the loss is `BCE(ŷ_i, y_i)` — i.e. the head is trained to predict its own
per-instance error rate, and the gradient attributes that error across the bag's cells in
proportion to how confidently each was written. No per-block verdict enters; the only number
from outside is the scalar the meter already returns per row.

**Its known failure mode, recorded now rather than discovered later**: the label is zero
whenever the renderer is right on every write in the bag, and Q0 measures `e_sp_practice` at
**0.0000 in eras 2 and 3** for both fitted arms — so a converged production route has no
gradient at all. This is the "no babbling" property of a deterministic renderer, and it is a
property of the arm rather than of the substrate. It is left in: an arm that stops learning
when it stops being wrong is what the biological framing predicts, and measuring when it stops
is the point. No exploration noise is added in Q1.

## 4. `surface_matched` matches per-cycle counts, exactly

The control the SPEC calls the node's science: the existing harvest subsampled per cycle to the
number of labelled blocks `own_scalar` receives that cycle. With §3(a)'s per-row tally that
count is exact rather than estimated. Two facts constrain how it is read:

- At MATCHED instance sets the ratio is a flat **0.2812** at every era.
- At the learner's ACTUAL sets it inverts: the surface harvest is solve-gated (`sol_i`), own
  writes are not, and `n_solved` falls from ~26/64 in era 1 to ~9/64 in era 3, so surface ÷ own
  runs **1.5–2.4 in era 1 and 0.40–0.64 in eras 2–3**. Past era 1 the production route is the
  *larger* channel, not the scarcer one.

`surface_matched` therefore subsamples to the own-write count **per cycle**, which is a
throttle in era 1 and an amplifier past it; the alternative (match at the instance level, a
flat 0.2812) is a different control and is not what the SPEC asked for. Both counts are logged
so either reading is available afterwards.

## 5. The pricing decision, and what is kept from the donor

- **`blk_render` is kept** and the fit's optimiser steps stay **unpriced**, exactly as in
  `inflection` — the fork's whole point is that only the renderer's training signal moves, and
  pricing one arm's fit and not another's would move the clock instead. `blk_render` and a new
  `blk_own` (blocks entering the own-write buffer) are tallied and logged, never folded into
  `t`. If a later round wants the priced version, both counters are already there.
- **The head stays one linear layer** over (one-hot feature, scalar register), so `SIZING.md`
  §2's ceilings remain the ceilings for these heads. Q0 re-derived that ceiling for the
  specific practised subset: det 0.4750 / acc 0.7875.
- **The gate discipline is the arc's**: `embouchure.py` with every knob off replays
  `inflection.py` at 0.000e+00 (their `fidelity_smoke`), re-run after every change to a shared
  path; every new mechanism gets a CPU gate in `gates_cpu` and a Modal smoke before a launch.
- **A preflight gate, not a launch, for the controlled read-back**: on a built `shared`, take
  the clean rule-spelled probe set at each register, flip one block to the other synonym, and
  compare `parse_features(reader, ·)` before and after. Q0's read-back numbers are
  observational; this is the controlled version and it costs only the setup a smoke already
  pays for.

## 6. Decisions left to the orchestrator

1. Settled 2026-09-10: three arms in `em_q1` (`canon_s`, `fit_rule_s` = surface,
   `own_scalar_s`), `given_rule` and `leaf` read from the banked `if_q1c_yk`. Settled
   2026-09-11: `own_readback_s` beside them in `em_q1b`, on a second L4.
2. Settled 2026-09-11: built (§2b).

## 7. Q2's lexicon — what the CPU sizing forces (2026-09-11)

`FILES.md` §Q2.0 has the ladder. Three decisions it forces, none of them taken here:

1. **Every rung is admissible on the world's own terms.** `root_reachable` and `on_grammar` are
   1.0000 at every H from 1 to 8, so nothing is ruled out by solvability — which means the rung
   is picked by the reader half (not run) and by how much of its own route `own_readback` keeps.
2. **The ladder is not difficulty-matched** — `d*` falls 27 % at L3 from H=0 to H=8, because
   more spellings are acceptable — so cross-rung comparisons confound homophony with repair
   difficulty. Within a rung every arm shares one world, which is where the clean contrast is,
   and that is the same discipline `inflection` Q2 recorded for its curriculum rungs.
3. **The asymmetry the SPEC wants is not in the lexicon.** "Ambiguous to the observer" and
   "unrecallable by the writer" are one condition, measured: recallable cells fall 1.0000 →
   0.2222 in lockstep with the observer's ambiguity. So Q2.1 needs the executor to CARRY the
   feature it wrote, not reconstruct it from the form — a Renderer-side record Q1 did not need,
   and a real addition rather than a flag. Whether that record is legitimate (a learner
   remembering its own command) or a smuggled oracle is the decision, and it is the same
   question `tempo`'s efference copy asks.
3. Whether `own_scalar` should be given exploration noise if it converges before the ladder
   does (§3(b) says no for Q1; the decision is recorded so the null is readable).
4. Whether Q2's pacer seat is still worth its 3 GPU-h given that Q0 shows the production route
   identifying the rule in the same cycle the surface does — the `inflection` Q3 inertness
   argument ("the head learned the rule from the world's abundant surface far faster than the
   ladder advances") applies unchanged unless `own_scalar` is materially slower.

## 8. The coordinator's three decisions, 2026-09-11 (as built)

**8.1 Decision 7.3 settled: the write record is the efference copy, not an oracle.**
`macro_features` / `node_features` return the intended level-1 `feats` from the learner's own
table, and the Renderer maps (feature, register) to a synonym before writing — so the
(feature, synonym) pair the executor emitted is its own state at write time, free.
`operators_are_arity_two` §4 defines the efference copy as exactly that. The oracle would be
the *world's* intent at a shared form, which the learner never receives, and nothing here asks
for it.

Consequence for the code: `own_recall_table`'s reconstruction-from-form is correct only while
the lexicon is injective and must be replaced by a record carried from the accepted move.
`PerfExecutor.apply` already hoists the render and keeps `last_ok`; the record keeps `feats`
and `k_written` beside it. **Staged behind `own_intent` (default `"recall"`) so `em_q1b`
stays reproducible**, with a gate that on `rule_seed 6` the record equals the recall table on
every block, and under homophony `n_ambiguous` goes to zero. Built after `em_q1b` was launched,
per the coordinator's instruction; it does not change `em_q1b`.

**8.2 The reader leg of Q2.0 is authorised and built** as `q2_reader_sizing` (a Modal
entrypoint, not an arm): one `build_shared` per rung on the constructed lexicon
(`lexicon_merge`, default `None`, so every prior tag is untouched and the G-F path never
reaches the function's body), then on a clean rule-spelled draw per register — kept WITH the
level-1 features the world derived, because on a homophonous lexicon the surface no longer
identifies them — the frozen reader's accuracy, split three ways:

- `read_acc_away` — blocks whose form is not shared: the baseline the rung must not cost;
- `read_acc_rule_resolves` — a shared form at a register where the RULE still admits one
  owner: a reader that has learned the register can be exact here, and whether it has is the
  question;
- `read_acc_rule_ambiguous` — a shared form where the rule admits both: no reader can beat
  chance between them, so this is a property of the lexicon and is the control.

Nothing is graded, no ladder advances, no arm runs. Rungs H = 2 and H = 4.

**8.3 The two instruments are never summed.** `cell_tally` records, on the SAME selected
blocks and one key (intended feature, register): `cell_n`, `cell_offrule` (the synonym is not
the one the rule calls for — the grader's instrument) and `cell_misread` (the read-back is not
the intended feature — the reader's instrument). Under homophony these come apart, and a
renderer that departs from the rule in order to be HEARD correctly scores on one and not the
other; summing them would hide exactly what Q2 is for. `analyze_embouchure.py`'s `_cell_table`
prints them side by side per cell with both denominators, plus the count of cells where only
one instrument fires. `cell_misread` is identically 0 for `own_scalar`, whose cells are keyed
on the reader's own parse — printed rather than omitted, because that is a property of the arm.
`em_q1`'s `own_scalar_s` predates the addition and will show the table as absent.

## 9. Where the reader's labels come from — the answer to the coordinator's question (2026-09-11)

**They come from the inverse map, i.e. last-writer-wins. Not from the generation-time feature.**
`build_shared` trains the reader with `train_reader(reader, leaves, bottom_map, ...)`, and
inside that loop the target is

```python
feats = bottom_map[(x.view(batch, n_blocks, s) * powers).sum(-1)]
```

with `bottom_map = torch.from_numpy(inverse_maps[-1])` — `build_inverse_maps`' table, which
resolves a shared form by last writer. So the reader's supervision is **a function of the FORM
ALONE**: every occurrence of a homophone in the corpus, at every register, carries the same
label, and no corpus could teach it otherwise.

**The `em_q2r` result is therefore not a reader failing to use the register.** It is a reader
returning the owner its own teacher labelled, exactly and everywhere. The perceptual route's
problem under homophony is that **its teacher relabels** — the `operators_not_footprints`
mechanism — which is a different finding from an organ that cannot read a register, and it is
the one the numbers support.

**And the RB-1 tension resolves.** RB-1's flip produces a DIFFERENT legal code, whose training
label was a different feature wherever that code belonged to a different owner; the reader is
label-faithful and context-sensitive wherever its labels were consistent, and last-writer
wherever they were not. Note also that the reader's corpus is drawn over the PRACTISED
registers only ({0,3,7}, `_sample_pool_r(..., practiced=prac)`), and every block in it is
on-register — so a lone off-register block is out of distribution in a way no training example
ever was, which is what RB-1 flips into.

**Two controls that were already in hand and point the same way.** The `em_q2r` cell where the
reader scores 0.0000 is at ρ=5, unpractised; the cells where it scores 1.0000 are ρ=1, 2
(unpractised) and ρ=3 (practised). Practised-ness does not separate them. The last-writer table
does, cell for cell.

### Reader B is a change to the substrate's reader, and is labelled as one

`q2_reader_probe` trains a second reader per rung — identical architecture, initialisation,
corpus, steps, batch, lr and seed — on the level-1 features the world **derived**, so only the
label moves. It is still exogenous and frozen before any arm runs, and its labels are the
substrate's own truth rather than anything a learner produced. But **the arc's reader has
always been the inverse map's student**, and reader B is not: it is given a distinction the
inverse map cannot express. It exists as a control for *why* the reader behaves as it does, and
it is not a component of any learner. If a Q2.1 arm is ever run against reader B, that tag is
running a modified substrate and must say so in its own record.

`q2_reader_probe` also does not call `build_shared` — it needs the reader and nothing else, and
the controller / generator / value / buffers are ~80 % of that setup. **Gate Q2R-1** certifies
the substitution: reader A's accuracy away from the homophones must reproduce
`q2_reader_sizing`'s `build_shared` reader on the same rung (0.9992 at H=2, 1.0000 at H=4).

### What `em_q2p` settled, and the one claim it withdrew

**Settled.** Reader A — the substrate's own supervision, reproduced — returns the last-writer
owner at a shared form in **24 of 24 cells, 100 % pure**, and reproduces the `build_shared`
reader to four decimals on every column (gate Q2R-1). The mechanism is the label, not the
organ, and it is now a confusion target rather than an inference from four accuracy cells.

**Withdrawn.** Round 1 called the rule-ambiguous cells a floor — "no reader can beat chance
between two owners". Reader B reaches 0.9682–0.9998 on exactly those cells. The block's FORM
is ambiguous; the SEQUENCE is not. What round 1 measured there was reader A's label, not the
lexicon's information content, and the claim is withdrawn in the same way §2's was.

**A confound to carry into Q2.1, not to design away.** The reader's corpus is drawn over the
practised registers only, so at H=2 and H=3 every "rule resolves" cell sits at ρ=5, an
unpractised register: that column is a generalisation test, not a disambiguation test. At H=4
the resolving cells include a practised register and both readers are exact there. Any Q2.1
readout that reports "the rule resolves it and the reader does not" must say which of the two
it is measuring.

## 10. The keying result, and five simulations that do not explain it (2026-09-11)

**What the run settled.** `own_scalar_rec_s` — the calibration objective unchanged, only the
cell key moved from the reader's parse of the arm's own write to the efference copy — reaches
`surface`'s exact recovered table (acc_practised 1.0000, held-out 0.8750, θ̂
[2,2,5,5,5,5,5,2]) at **cycle 7**, against `own_scalar_s`'s 0.7250 / θ̂ ≈ 3 and
`own_scalar_pg_s`'s 0.4250. **The keying axis carries the whole effect; the objective axis
carries none of it** — the self-imitation form lands below the calibration form it was meant to
fix. The coordinator's diagnosis of the plateau as the objective is withdrawn as a claim, and
so is the mis-keying hypothesis that replaced it: the mis-keying rate is **0 over 148,305
blocks**.

**What the key actually moves** is which blocks are USABLE, and the dropped set is 99 % errors:
under `recall` the `agood` filter loses 15.91 % of the arm's own writes and **99.148 % of those
are off-rule**, against 4.47 % among the kept ones — a 22× enrichment. The bag holds the
learner's correct writes and excludes its wrong ones while the label, which comes from the
grader, counts every write.

**And six closed-loop simulations of that same objective still recover the rule.** Each was
written to close the previous one's gap, and each reaches acc_practised 1.0000 / held-out
0.8750 with θ̂ [2,2,5,5,5,5,5,2]:

| probe | what it varied |
|---|---|
| EB-9 | uniform feature draw |
| EB-10 | random cell dropout, 0–25 % |
| EB-11 | the arm's own 12× feature marginal |
| EB-12 | error-correlated dropout at the arm's measured rates (0.808 / 0.0017) |
| **EB-13** | **coherence-conditioned dropout** — the drop probability a sigmoid in how MINORITY-wrong a block is within its own answer, offsets re-solved by bisection every round so the pooled rates stay pinned at 0.808 / 0.0017; slope 0 is the control and reproduces EB-12 |
| **EB-14** | the arm's feature marginal **crossed with** the dropout, independent and coherence-conditioned — the last un-crossed cell |

EB-13 was the sharpest of them: if a mixed register loses exactly its wrong blocks under a high
label while a coherently-wrong register keeps its rows, the only stable configurations are
coherent-per-register, and coherent-per-register is a shared threshold — which is what
`own_scalar_s` settled on. At slope 6.0 with the pooled rates pinned, it recovers the rule
anyway.

**The bag's register structure was checked, not assumed.** Every bag in EB-9…14 already draws
ONE register per bag — `_pr9[rng.integers(..., size=(NB, 1))].repeat(SP, 1)`, verified to give
a single distinct register per row — over 9 cells at ρ ∈ {0, 3, 7}, so the label is already the
off-rule fraction at a single register and the per-register structure the recall drop would act
on is inside the bag. That is the arm's own geometry: one instance, one register, ~9 own writes.
The only factor that had never been crossed was the arm's 12× feature marginal WITH the dropout
(EB-11 had the marginal and no dropout); EB-14 crosses them, both dropout forms, and recovers
the rule as well.

**So the mechanism by which the recall key costs 0.15 in held-out accuracy is UNEXPLAINED on
the record.** The empirical answer is not in doubt — the record key is at the ceiling and the
recall key is not, on one clock with a 0.000e+00 cross-tag anchor — but no simulation this lane
has built reproduces the failure, and that gap is stated rather than narrated.

**Not run, and noted as a gap rather than a task** (the coordinator's): the self-imitation
objective on the record key. The calibration form on the record key is already at the ceiling,
so the cell is not decision-relevant.

**One fact from Q2's per-cell tables, for the reader of this section** (`em_q2`, §4E, no
interpretation): at the shared registers the **loser** owner of each shared form is nearly
absent from the learner's own intents — in `own_readback_s` at ρ=3, f0 has **68** own writes
against f1's **2,883**, and f2 and f5 have **1** and **2**; in `own_scalar_rec_s` at ρ=3, f0
has **4** against f1's **5,157**, f2 has **2**, and f5 has none at all. The three winners at
that register carry 2,883 / 6,451 / 9,611 (readback) and 5,157 / 5,734 / 10,231 (record).

## 11. The gate discipline, after three gates stopped runs on their own strictness

Three launches died on gates I wrote, none of them on a defect in the thing being measured:
EB-8's `torch.equal` over 4,096 tokens (twice — once on an arm that should never have been
evaluated, once on a 1-in-41,000 batch-shape numerical tie) and EB-6's intent comparison on
blocks where the record was deliberately unavailable. The coordinator's rule, adopted:

> **Assert an identity only where the substrate is deterministic** — CPU gates, the same batch
> shape, knobs off, the G-F replay. **Wherever a comparison runs across batch shapes, through a
> replay, or on a quantity that can be unavailable: measure it, report it with its denominator,
> and assert only coverage.**

EB-8's split — a hard equality on base-move blocks where both paths are provably `apply_any`,
a measured rate on macro blocks where they are not, and a coverage floor over the whole — is
the model. Two habits that would have caught all three before a launch:

1. **Run every preflight long enough to reach the first macro fire AND the first unavailable
   block.** The macro fire is reached (552 and 484 macro-written blocks at preflight scale);
   the first unavailable block is not, because the preflight's replay never disagrees. Where a
   path cannot be reached by scale, it is reached by construction instead — **gate EB-15**
   drives the unavailable-block path directly on synthetic tensors.
2. **Write at the assert the reason a gate is not an equality**, so the next edit does not
   quietly restore one. Every softened gate in this file now carries that sentence.

## 12. The two ears — what reads what, and what the generator is actually taught (2026-09-11)

Q2.2's premise is that the collapse belongs to the **listener's teacher**. Auditing every path
before the launch found that the node does not have one listener. It has two nets that turn a
surface into a feature, they are taught by different teachers, and only one of them is
`shared["reader"]`.

### The audit, path by path

**Through `shared["reader"]`** — the frozen ear, trained once in `build_shared`, the object the
`reader_target` knob moves:

| path | site |
|---|---|
| the harvest parse (`of`) | `fit_shared`'s branch; `fit_parse` defaults to `"reader"` |
| the own-write parse and the read-back (`af`, `_af`) | the own arms' bag construction |
| mining's observation parse (`pf`) | the miner, and the question port's own parse |
| the selection parse at the question port | `QS.half_keys(pf, geom)` |
| gate RB-1 | `readback_flip_check` |
| `plant_probe`'s `read_acc` | oracle readout, never consumed |

**NOT through it** — and this is the finding:

| path | what it reads | taught by |
|---|---|---|
| **the executor's DP** — `apply_move → node_features(generator, …)`, `MC.apply_any → macro_features_rec(generator, …)` | **the arm's own generator** | `bottom_map[code]` at setup, and again every cycle in run |
| `plant_probe`'s `parse_acc` / `infill_acc`, the endo bench | the generator | as above |
| the corpus filter (`filter_pool`, `level2_tuples`) | `inverse_maps[-1]` | setup hygiene; inert at `plant_holdout = 0` |
| the grader | the rules directly | untouched by design |

The DP is **the chooser's ear**: it is what scores a candidate expansion before the renderer
spells it, so a distinction it cannot hear is a distinction the executor cannot choose to make.
It is a second student of the same last-writer table.

### The question, answered: the generator's in-run targets

Two functions update the generator in run, and they are the same function twice:

- `finetune_generator(generator, gopt, solved, replay, **bottom_map**, …)`
- `finetune_generator_span(…, **bottom_map**, …)` — the same batches, the same masking draws
  and the same level-1 loss, plus the span head's self-imitation term at weight `lam`.

In both, the level-1 target is `feats = bottom_map[(leaves · powers).sum(-1)]` — **the inverse
map**, on the union of (a) configurations the agent SOLVED this cycle and (b) replay drawn from
the clean setup pool. So the answer to the coordinator's question is: **through the inverse
map**, not through `shared["reader"]` and not through the grader. The span term is the one
exception — its target is the executor's own chosen slots, i.e. self-imitation, and it reaches
the generator's trunk through the shared optimiser group.

**And it cannot simply be moved.** For the replay half the derived features exist, because the
world derived those rows. For the new half they do not: those configurations were produced by
the learner, the world never derived them, and the only function that maps an arbitrary surface
to a feature is the inverse map. There is no truth to substitute. So the honest form of the
both-ears arm is the one the coordinator specified: **switch the setup supervision, leave the
in-run target, and measure the drift** rather than assume it away.

### What the pair therefore is

| tag | `reader_target` | the listener (`shared["reader"]`) | the chooser (the generator's setup supervision) |
|---|---|---|---|
| `em_q2` (banked) | `last_writer` | last-writer table | last-writer table |
| **`em_q2b_l`** | `listener` | **derived features** | last-writer table |
| **`em_q2b`** | `both` | **derived features** | **derived features** |

The contrast `em_q2b` − `em_q2b_l` is the chooser's ear alone; `em_q2b_l` − `em_q2` is the
listener's alone; `em_q2b` − `em_q2` is both. Nothing else moves: the grader, the lexicon, the
rule, the five arms and every seed are shared.

### The drift readout

`plant_probe` gains a `loser_cells` block, computed wherever the derived features exist, on the
blocks where the two teachers disagree — which is exactly the loser-cell blocks, and needs no
lexicon lookup to select. At each `probe_every` cycle it records, for the generator and for the
reader separately, the share of those blocks parsed as the **derived** feature against the
share parsed as the **last-writer** one, with the denominator. In `em_q2b` the generator starts
taught by the world and is pulled back toward the last writer by every in-run step; how far it
travels over 140 cycles is then a measurement in the log rather than a caveat in this file.
