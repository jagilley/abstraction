# aliquot — the verdict read off the world model's own state: a projection of the practice plant in the grader's seat, then the outcome error put into the plant's weights, offline and in the loop

**Up**: [`../README.md`](../README.md) (sotto_voce) · **Spec**: [`SPEC.md`](SPEC.md) (the orchestrator's brief verbatim) ·
**Decisions and every withdrawn diagnosis**: [`DESIGN.md`](DESIGN.md) · **Machinery record**: [`FILES.md`](FILES.md) · **Conversation**: `CONVERSATION.md`[^private] (the session behind the three rounds, 2026-09-18 → 20) ·
**Children**: [`duplex/`](duplex/README.md) (the shaping step, offline on banked dumps, 2026-09-19) ·
[`soundboard/`](soundboard/README.md) (the shaping step in the loop, 2026-09-20) · **Cross-arc reading made in the same
round**: [`logit_reading/striatum/norm/results/tables_discrim_20260919.md`](../../../../logit_reading/striatum/norm/results/tables_discrim_20260919.md)
(the random-init-trunk discrimination control on the 8-layer trunk's ridge, CPU on the banked cells).
**Motivation**: [`ideas/calibration_and_violation_are_one_object.md`](../../../../../../ideas/calibration_and_violation_are_one_object.md)
§12, in particular the §12.4 note on how a candidate is priced before it is committed; the session record is
`conversations/one_world_model_many_projections_2026-09-18.md`[^private].
**Runs**: 2026-09-18 → 20; `al_s1`/`al_s2` (two arms per seed, ≈9 GPU-h with three preflights), `du0`/`du1`/`du2`
(offline, ≈1.2 GPU-h), `sb_s1`/`sb_s2` (three arms per seed, ≈17 GPU-h); seeds 0 and 2 throughout, every paid arm
clock-yoked to its seed's banked anchor with the replay exact and nothing cancelled, every fork bit-identical to its
donor with the knobs off. **Ranks, signs, located mechanisms and per-cell counts are the claims; the seeds are never
averaged.**
**Attribution**: the framing of §12.4's replay note as a literal grader for candidate abstractions, the ask to take
stock of what it resolves, the inventory question about the remaining oracle reads, the question of how the mirror's
representation could be made present in the plant, the teacher-at-low-levels transfer question, the note that
language's inherent grader is good exactly until the frontier, and the call to try the shaping step on banked
checkpoints before paying for the loop are Jasper's (2026-09-18 → 20). The three-blindnesses reading of §12.4, the
reading of the plant as the wrong wire, the producer-versus-reader reading of soundboard's chooser, and the nesting
reading of the yield currency are the orchestrator's, agreed in discussion. The builds, the gate tables, the
exact-duplicate channel and the dissociation between a readout's AUC and a plant's competence are the three
implementers' (one Opus session each), as is the withdrawal in duplex §2.6.

## One-liner

§12 says the value reader is a linear projection of the world model's state, weights set by outcome history, sighted
where the world model is and blind where it is; §12.4 adds that a candidate has to be fired to be priced, so replay
of untaken roads through that projection is how a candidate abstraction would be graded. `sotto_voce` had already
built the grading half of that with a separately trained outcome model, the mirror, and found it blind at the
frontier. This node put the literal object in the mirror's seat: a ridge-logistic readout of the live practice
plant's pooled state over the substituted configuration, one readout per root, fit on the mirror's diet alone.
**It does not read the verdict.** On the learner's own experienced configurations the projection reaches held-out AUC
0.76 / 0.73 where a never-trained trunk reaches 0.72 / 0.74 and the mirror reaches 0.99; on the untaken roads its
precision is a third of the mirror's, the critic it teaches ranks the world worst of the mirror family, and its
chooser sits at the floor. The plant trained only on masked infilling represents what prediction needs, not what the
outcome needs. **Putting the outcome error into the plant's weights fixes the read and not the choice.** Offline on
banked dumps and then in the loop, an outcome head whose gradient reaches the trunk beside the plant's own infill
term makes the verdict linearly available, +0.06 / +0.13 in the loop with the trained-minus-random margin positive on
both seeds for the first time in the lineage, at +0.008 nats of infill loss offline against a matched control; with
the infill term off the plant leaves the observer family within a hundred steps offline and the run collapses in the
loop. The chooser does not follow the read: the shaped arms sit where the frozen-plant projection sat and the mirror
still beats both, while the shaped plant's block-level parse, which the executor's DP scores through, loses fourteen
points. A currency arm shaped on next-level yield read its own currency no better than the verdict-shaped one, because
on this substrate the yield label is nested inside solving by construction.

## Children

### [`duplex/`](duplex/README.md) — the shaping step, offline on overtone's banked dumps (2026-09-19)

**Goal**: before paying for the loop, insert a shaping step into `overtone`'s post-write linear-probe protocol on the
banked plant and rows: fine-tune the plant on an outcome head with the infill loss on, off, and infill-only at matched
steps, then re-read on identical held-out rows, with world-model diagnostics and an L2/L3 → L4/L5 transfer split.
Three tags: the overtone reproduction, the same with an exact-duplicate filter, and the outcome heard at L2/L3 only.

**Finding**: the reproduction gate returns overtone's table at the median to 0.0015. Outcome-only shaping takes the
block head's parse from 0.65 to 0.35 within 100 steps and buys nothing for the read; both-loss shaping costs +0.008
nats against its matched control and raises the duplicate-free filed read from 0.63 to 0.73 / 0.78, most of it
context rather than candidate. An outcome heard at L2/L3 only retains 62–82% of the frontier gain. The critic's audit
split carries an exact-duplicate channel worth 0.01–0.04 to any reader that saw verdicts; the mirror bank's split
does not. Facts in `duplex/figures/du{0,1,2}_reduction.txt`; §3 below.

### [`soundboard/`](soundboard/README.md) — the shaping step in the loop (2026-09-20)

**Goal**: the same outcome head with its gradient reaching the plant's trunk during the run, aliquot's projection
unchanged in the grader's seat, three arms per seed: shaped on the verdict with both losses, shaped on next-level
yield from the learner's own miner, and shaped on the verdict with the infill term off; per-cycle world-model
diagnostics, a per-probe instrument in the yield currency, the trajectory id banked, full dumps on every arm.

**Finding**: the readout's held-out AUC rises to 0.82 / 0.86 (verdict), 0.78 / 0.79 (yield) and 0.80 / 0.80 (no
infill) against the banked 0.76 / 0.73 and 0.72 / 0.74, six of six above both; the critic it teaches ranks the world
as well as the mirror's does; the chooser does not move; the no-infill run collapses while its readout still reads
0.80; the yield-shaped projection is no better at its own currency. Facts in `soundboard/figures/sb_seedtable.txt`;
§4 below.

### [`preplay/`](preplay/README.md) — the level reader prices abstractions and does not rank spellings: the projection read offline on banked state (2026-09-21)

**Goal**: the two questions the loop could not pose, on the banked plants, readouts and rows, with nothing paid in
the loop. Within a fixed context, does the projection rank the candidate spellings as the world does
([`preplay/within/`](preplay/within/README.md), CPU)? And fired one entry at a time through the shaped executor on
fresh pools, does its mean level price a candidate table entry as the world's audition does, does it select a better
table than a random draw, and does it do so on the learner's own mined entries, replayed exactly from the banked
keys and picks (`pp1`, `pp2`, `pp3`, 0.35 GPU-h)?

**Finding**: the projection does not rank within a context (0.65 / 0.70, below the executor's prior at 0.71 / 0.74),
the prior reads the same on the shaped and the frozen plant, and the composed chooser ranks at 0.76 / 0.82 above both
of its organs, so §4's producer-versus-reader reading is withdrawn; a never-trained twin under the same readout ranks
within 0.03–0.07 of the shaped trunk, the diet is worth +0.03 to +0.05, and the global pooled form is the largest
axis. The same projection prices a fired candidate entry in the world's order (true from wrong at 0.61–0.75 against
the world's 0.89–0.97, frozen and twin at chance, positive within the legal class on 23 of 24 cells), recovers a
median 0.65 of the world's advantage over a random draw as a selector, matches the full true table at L3, and fails
where a few attractive wrong entries capture the executor's argmax; on the learner's own entries the world itself
barely separates true from false at L3/L4 and the read tracks the world's price where the prior prefers the false
ones. Full record: [`preplay/README.md`](preplay/README.md).

## The question

`sotto_voce` asked whether the world has to be paid for verdicts on the probe channel's counterfactuals, the
substituted spellings the learner never wrote, and answered with the mirror: a tree net trained end to end on the
learner's own experienced configurations and the world's verdicts, applied to configurations it never wrote. It
recovers about half the value of paying the world, is near-perfect on its own support and blind at the frontier. The
mirror is a separate object with a representation of its own. §12.4's object is different: the verdict read off the
world model's own state through a linear readout, where the world model's representation comes from prediction and
only the readout's weights come from outcomes. Whether that reads the untaken road better than the mirror, worse, or
in a different place, is what `aliquot` asked. When it read worse everywhere, the question became whether the
outcome error has to reach the trunk's weights, which is `duplex` and `soundboard`.

Vocabulary, since it accumulates. The **plant** (also the generator) is the 2-layer, 96-wide masked-infill encoder
the practice loop trains every cycle; the executor's **DP** scores candidate spellings through its block logits, so
it is both the producer and, here, the thing read. The **mirror** is `sotto_voce`'s outcome model. The
**projection** is this node's readout: `SN.trunk`'s pooled hiddens over the substituted final configuration, read
through the `mask_block` idiom, standardised, with the root one-hot and its interaction with the pooled state, so one
linear readout per goal (1737 columns), fit by IRLS with the ridge selected on a train-internal split, features
recomputed through the current trunk at every refit. The **random twin** is the same readout over a never-trained
trunk of the same architecture at a stated seed. **Shaping** is an outcome head over the same pooled features whose
BCE gradient reaches the trunk in the same optimizer step as the infill loss. The **probe channel**, the **floor**
(filed diet), the **ceiling** (world-graded probes), **rep** (class accuracy against the repair set) and the **clock
yoke** are `voicing`'s and `sotto_voce`'s, unchanged.

## What was built

[`aliquot.py`](aliquot.py) forks `sotto_voce.py` at its `so_s1`/`so_s2` head and adds `vo_om_mode = "proj"`: the
projection in the mirror's seat, same buffer, same two doors, same schedule, the same probability filed, dispatching
exactly where `"model"` does. Two paid arms per seed, `al_pj_yk` (live trunk) and `al_rt_yk` (random twin), the
floor, ceiling and three mirror arms banked and read in. Four projection gates with run-level forms, a GPU-side gate
block added after a defect a CPU gate could not see, and a falsification harness at 73/73 under the rule that a gate
is not reported until it has been shown to fail. [`duplex/duplex.py`](duplex/duplex.py) is the offline job on
overtone's banked `vo_heads.pt` and `vo_rows.npz`; [`soundboard/soundboard.py`](soundboard/soundboard.py) forks
`aliquot.py` and adds the shaper, the yield currency, the in-loop diagnostics and the dumps. Every knob defaults off
and every fork is its donor to the bit with the knobs off (gate G-F, 0.000e+00 on each). The full record of knobs,
gates, arms and runs is in each node's `FILES.md`; the decisions and the ten implementer defects, each caught by a
gate and none by a claim, are in each `DESIGN.md`.

## 1. The projection in the grader's seat does not read the verdict (`al_s1`, `al_s2`)

**On its own experience.** Held-out AUC of the grader on the learner's experienced configurations, the mirror's home
ground:

| readout, held-out AUC on experience | seed 0 | seed 2 |
|---|---|---|
| projection, live trunk | 0.759 | 0.728 |
| projection, random trunk | 0.716 | 0.740 |
| mirror (banked), same diet | 0.996 | 0.986 |

The per-goal interaction beats the additive root on every arm; masking the read helps the random twin far more than
the live trunk (0.716 against 0.641 unmasked, where the live trunk reads 0.759 against 0.747). 151 / 141 refits, none
degenerate at run scale, the trunk's fingerprint moving on the live arm and constant to the bit on the twin.

**On the untaken roads.** Against the world's verdict on the probes, per level (seed 0 / seed 2):

| grader | L2 accuracy | L4 accuracy | precision at L2/L3 |
|---|---|---|---|
| mirror | 0.931 / 0.932 | 0.890 / 0.832 | 0.32–0.50 |
| projection, live trunk | 0.854 / 0.829 | 0.788 / 0.766 | 0.10–0.11 |
| projection, random trunk | 0.883 / 0.854 | 0.805 / 0.753 | 0.11 |

The projection calls far more positives than the world has (its mean probability 0.17–0.20 at a base rate of ~0.05).
The decline with level is monotone on every arm and in-support cells are worse than off-support in accuracy on every
arm, as they were for the mirror.

**Downstream.** The critic taught by the projection ranks the world at 0.596 / 0.617 (live) and 0.603 / 0.628
(random), below every mirror-family arm (mirror 0.641 / 0.661, committee 0.630 / 0.682, hybrid 0.731 / 0.735) and
above the filed-only 0.49 of `voicing`. The chooser sits at the floor at seed 0 (−1.0%) and at the mirror at seed 2
(+5.1% against +5.5%), with the live-versus-random ordering reversing between seeds (−1.0% / −8.3% at seed 0, +5.1% /
+5.9% at seed 2), so no claim rests on it. Bill zero on both projection arms, yoke exact.

The instrument gap inherited from the parent: `sotto_voce`'s per-probe ranking sample holds cycles 50–54 only, so the
grader's own ranking of the world's verdict is not compared between mirror and projection here; the comparison runs
through the per-level accuracy and the downstream critic, both whole-run on every arm.

## 2. What the strong world model's readout buys, for scale (`norm/`, CPU on the banked cells)

The same question asked of the 8-layer next-token trunk in `logit_reading`, whose ridge critic is the value arc's
instrument: the queued random-init-trunk discrimination control, run on the banked norm cells as a pure CPU
reduction, three trajectory seeds. Within cells (matched on `k*`, `j` and position, sign-balanced on surprisal), the
post-event level's ranking of realised damage at the violation token goes from 0.50–0.63 on the step-0 trunk to
0.54–0.72 at 64k, a gain of +0.05 to +0.14 at ℓ = 2–4 on both venues and mixed at ℓ = 1, on all three seeds; at the
edit onset +0.06 to +0.19. Pooled over cells the pre-event level reads the outcome at 0.66–0.73 at the edit onset
from a state that precedes any edited token, so the pooled gain carries the query's cell and not only the outcome.
The outcome label is the frozen actor's own hit and its base rate moves with the checkpoint, so the step-0 control
holds the rows fixed and not the label; the grammar's consequence label, which is fixed, reads at or below chance at
the violation token at 64k. Full tables in
[`tables_discrim_20260919.md`](../../../../logit_reading/striatum/norm/results/tables_discrim_20260919.md). The
trained representation buys the linear readout real discrimination there, modestly, and the random trunk sits near
chance; on the practice plant the random trunk already sat at 0.72 and training added three points. The two tasks are
not the same shape, so the comparison is directional.

## 3. The shaping step, offline (`duplex/`, `du0` → `du2`)

The protocol is `overtone::post_write_probe`'s, reproduced cell for cell on the frozen and random trunks as the gate,
then re-run after fine-tuning the banked plant on an outcome head at the run's own learning rate for 100 / 400 / 1600
steps with the masked-infill loss on (`both`), off (`out`) and infill-only at matched steps (`infill`), the outcome
rows and the corpus windows keyed on the step count so the arms are a contrast.

**The world model.** Held-out infill CE and the block head's level-1 parse against the exact features, seed 0:

| trunk | infill CE | parse | nested DP parse L2 |
|---|---|---|---|
| frozen | 1.418 | 0.646 | 0.390 |
| infill only, 1600 | 1.333 | 0.669 | 0.419 |
| both losses, 1600 | 1.341 | 0.591 | 0.415 |
| outcome only, 1600 | 2.334 | 0.289 | 0.210 |
| never trained | 2.176 | 0.194 | 0.150 |

Both losses cost +0.008 / +0.007 nats against the matched control and 0.05–0.07 of parse; outcome only costs a nat,
takes the parse to 0.35 within 100 steps, and at 2–4 masked blocks reads worse than the never-trained trunk.
`shaped/`'s ledger on the logit-reading trunk was +0.008 against +0.61. The nested DP parse, the practice-side
altitude instrument, does not degrade under both losses.

**The read.** With the exact-duplicate filter on (`du1`; the channel is §3's last paragraph), the read of record,
median over slots:

| trunk | s0 filed | s0 probe | s2 filed | s2 probe, uniform quarter |
|---|---|---|---|---|
| random | 0.625 | 0.734 | 0.621 | 0.716 |
| frozen | 0.634 | 0.797 | 0.633 | 0.736 |
| infill only, 1600 | 0.643 | 0.809 | 0.643 | 0.721 |
| both losses, 1600 | 0.732 | 0.832 | 0.783 | 0.727 |
| outcome only, 1600 | 0.754 | 0.788 | 0.775 | 0.724 |
| MLP critic on the frozen trunk | 0.625 | 0.850 | 0.705 | 0.807 |

Seed 0's probe column is stable to ±0.004 across the three tags; seed 2's uniform probe cell moves 0.04 under a 2%
change in the diet and carries no claim. Most of the filed gain is context: the pre-write read, with the candidate not
in the input, rises from 0.64 to 0.72 / 0.73 under shaping, and the candidate-specific increment is +0.00 / +0.04; the
global-mean read rises more than the span read, so the outcome direction is written into the global pool. Outcome-only
shaping leaves the probe read at or below frozen on both seeds.

**Transfer.** With the outcome heard at L2/L3 slots only (`du2`), a readout fit on L2/L3 and scored on L4/L5 sits
0.06–0.09 above frozen at both seeds and both diets, 62–82% of what hearing it everywhere buys; the L4/L5-own fit
through a trunk that never heard an L4/L5 verdict retains most of it on three of four cells and 44% on one.

**The exact-duplicate channel.** The critic's audit split hashes the pre-write context, so a trajectory that wrote at
several slots puts the same final configuration and verdict on both sides: 26% / 25% of filed held-out rows on the
banked overtone arms, 96% with the same verdict, under 1% of probe rows. It cannot touch a frozen or random trunk and
is worth 0.01–0.04 to a reader that saw verdicts; `voicing`'s and `overtone`'s filed audit columns inherit it. The
mirror bank (`sotto_voce`, `aliquot`) deduplicates within the push and is immune to it, and is open instead to the
same configuration recurring in a later cycle, which soundboard then measured at 0.0000–0.0013 (§4). The first claim
that aliquot inherited the channel was withdrawn against the code (`duplex/DESIGN.md` §2.6).

## 4. The shaping step in the loop (`soundboard/`, `sb_s1`, `sb_s2`)

Three arms per seed on `aliquot.py`'s stack, the projection unchanged in the grader's seat, the plant now training on
an outcome head beside its infill term: `sb_sv_yk` (target the world's verdict on the learner's own final
configurations), `sb_yd_yk` (target next-level yield: the share of a solved configuration's level-(era+1) spans whose
key is at support in the learner's own miner, zero on an unsolved piece), `sb_so_yk` (the verdict, infill term off).
The verdict and yield arms are bit-identical until each seed's L2 commit, the first cycle a macro write exists.

**The read.**

| readout, held-out AUC on experience | seed 0 | seed 2 |
|---|---|---|
| never-trained trunk (banked) | 0.716 | 0.740 |
| frozen plant (banked) | 0.759 | 0.728 |
| shaped on next-level yield | 0.782 | 0.787 |
| shaped on the verdict, infill off | 0.800 | 0.800 |
| shaped on the verdict | 0.819 | 0.861 |

Six of six above both banked arms; the unmasked variant closes on the masked read once shaped. The verdict-shaped
readout's level is calibrated on the bank's held-out filed rows (ECE 0.054 / 0.039).

**The choice, and the critic.** Pooled L2/L3 repair accuracy against the floor: verdict −0.6% / −1.3%, yield −2.3% /
+5.0%, against the frozen-plant projection's −1.0% / +5.1% and the mirror's +4.2% / +5.5%; nothing replicates at L4/L5,
where at seed 2 the random twin is top. The critic taught by the verdict-shaped projection ranks the world at 0.654 /
0.671 against the frozen-plant projection's 0.596 / 0.617 and the mirror's 0.641 / 0.661; the no-infill arm's 0.688 /
0.580 reverses order between seeds.

**The world model, in the loop.** With both terms on, held-out infill CE +0.055 / +0.051 within-arm (no in-loop
infill-only control exists; the banked frozen arms predate the diagnostic, and the queued re-run is the cheap fix),
the block head's parse 0.67 → 0.53 (verdict) and 0.57 (yield), the nested DP parse flat or up (L3 0.286 → 0.321 at seed
0). With the infill term off the run collapses: repair accuracy −62.9% / −73.0% against the floor, infill CE 1.17 →
2.92 / 2.81 (a never-trained trunk reads 2.18), parse 0.67 → 0.38, the nested DP parse down at every rung, corridor
misfire 0.81 / 0.66 against 0.38 / 0.37, the bank's solve rate 0.055 / 0.040 against 0.28. That plant's linear readout
reads 0.800, above the frozen plant's, on that collapsed distribution: a linear read's AUC is not a measure of the
plant's competence.

**The currency.** The yield label is a real second target (mean 0.08 / 0.07, 16% of rows positive), and the
yield-shaped projection ranks its own currency no better than the verdict-shaped one on the per-probe instrument
(AUC against yield 0.710 / 0.685 against 0.733 / 0.703); the reader's parse of a substituted configuration and the
exact parse agree on span membership, so the only gap between the endogenous and oracle columns is the miner's own
false entries at support.

**The two spec lines.** The cross-cycle duplicate share of the bank's hold split is 0.0000–0.0013 on both seeds, so
`aliquot`'s held-out AUC was not inflated by it. Trajectory ids are banked and all three dumps landed on every arm.

## The update

1. **The projection of a prediction-trained plant does not read the outcome, and a never-trained trunk reads
   almost as well.** The mirror's competence is representation it builds from outcomes, not representation the plant
   already has. In §12's terms this is input blindness in its strong form: the practice learner's knowledge lives in
   the table, the routing head and now the mirror, and a projection of the plant's activations reads the wrong wire.
   The mirror is a world model plus a projection with the world model learned from outcomes; the logit-reading
   critic is a world model plus a projection with the world model learned from prediction, and on the strong trunk
   the trained representation does buy the readout real discrimination over its random init (§2).
2. **The outcome error has to reach the trunk's weights, and when it does the verdict becomes linearly available.**
   Offline and in the loop, on both seeds, with the trained-minus-random margin positive for the first time. What
   shaping writes is mostly a level, how solvable this state is, rather than a candidate ranking within a state.
3. **Both losses, or the plant stops being a world model.** `shaped/`'s finding on the logit-reading trunk
   reproduces on the practice plant with the same +0.008-nat ledger offline, and in the loop the outcome-only plant
   takes the run down with it because it is also the executor; its readout still reads 0.80.
4. **A better read is not a better choice, on this stack.** The shaped projection teaches the critic as well as the
   mirror does and the chooser does not move. The reading offered, held loosely: the plant is both the producer the
   executor's DP scores through and the thing read, shaping it for reading cost its block-level parse fourteen
   points, and the composed chooser is prior plus judge; the mirror wins the chooser because it leaves the producer
   alone. The queued within-context discrimination readout is what would isolate this, and the arm it suggests
   shapes a reader copy of the plant and leaves the executor's untouched.
5. **The yield currency did not separate from the verdict here, by construction.** Yield is zero unless the piece
   solved, so on this substrate next-level yield is nested inside solving and a plant shaped on it learns mostly
   "solved". This is `tessitura`'s structure-and-cost non-independence one organ over. The currency question, the
   one that motivated §12.4's reading, is deferred to a world in which solving now and carrying next-level material
   come apart.
6. **Two instrument facts for the lineage.** The critic's audit split carries an exact-duplicate channel that
   `voicing`'s and `overtone`'s filed columns inherit; the mirror bank's split does not, and its cross-cycle
   recurrence is nil. And an outcome-only shaped plant's readout AUC rising while the plant collapses is a
   dissociation worth keeping in view whenever a readout is used as a gauge of the thing it reads.

## What this does not show

No frontier claim survives the seeds, in any of the three rounds, which is what `voicing` Q3e and `sotto_voce`
recorded for exactly these two draws. The chooser's live-versus-random ordering reverses between seeds in `aliquot`.
The producer-versus-reader reading of soundboard's chooser is one reading of two seeds and the prior's ranking quality
was not measured directly. Duplex's offline rows were written by an unshaped plant, so its numbers say nothing about
the loop; soundboard has no in-loop infill-only control, so its +0.05-nat infill cost is within-arm. Seed 2's uniform
probe cell is unstable at its n. A held-out probe row can still be a near-twin of a filed training row, which only the
now-banked trajectory id can close. The currency arm's label is nested in solving and the yield-versus-verdict
comparison is therefore not a test of the currency question. The random-trunk control on the norm ridge holds rows
fixed and not the label, and its pooled numbers carry cell knowledge. Nothing off the RHM practice substrate and the
logit-reading grammar.

## Reproduction

```bash
cd experiments/            # MODAL_PROFILE=chromatic
# aliquot: gates, harness, G-F, the two paid tags, the table of record
PYTHONPATH=. python3 -c "from rhm.practice.voicing.sotto_voce.aliquot import aliquot as A; A.vo_gates_cpu()"
modal run rhm/practice/voicing/sotto_voce/aliquot/aliquot.py::falsify_remote          # 73/73
modal run rhm/practice/voicing/sotto_voce/aliquot/aliquot.py::proj_gates_gpu           # P-1..P-5 on an L4
python3 rhm/practice/voicing/sotto_voce/aliquot/launch_detached.py --fn fidelity_smoke --tag al_gf1
bash rhm/practice/voicing/sotto_voce/aliquot/results/RUN_al_s1.sh; bash rhm/practice/voicing/sotto_voce/aliquot/results/RUN_al_s2.sh
python3 rhm/practice/voicing/sotto_voce/aliquot/fetch_compact.py --tag al_s1 --fetch --replace   # and al_s2
python3 rhm/practice/voicing/sotto_voce/aliquot/mk_seedtable.py
# duplex (offline, on overtone's banked dumps) and soundboard (in the loop): see each child's FILES.md
# the norm ridge control (CPU, after `modal volume get` of the banked cells; see the script's docstring)
python3 rhm/logit_reading/striatum/norm/discrim.py
```

Volumes: `rhm-scaling-data:/rhm_practice_aliquot/<tag>/`, `/rhm_practice_duplex/<tag>/`,
`/rhm_practice_soundboard/<tag>/`; compact mirrors and reductions under each node's `figures/`. Every flag per tag,
every gate and every app id is in each node's `FILES.md`.

## Next steps (queued in `QUEUE.md`[^private], not started)

The within-context candidate-discrimination readout, CPU on the six soundboard dumps · the in-loop infill-only
control, `al_pj_yk` re-run with only the instrument knobs on · the reader-copy arm, shaping a copy of the plant the
reader uses and leaving the executor's untouched, after the readout says whether the prior eroded · the world that
dissociates solving from next-level yield, a design step before any currency arm · the per-block readout and the
mirror re-run with the reservoir sample, from aliquot's own queue.

> **2026-09-21.** The first item ran, the second and third were retired, and the round is written up as the child
> [`preplay/`](preplay/README.md): the within-context readout landed as [`preplay/within/`](preplay/within/README.md) and
> the abstraction-level question §12.4 poses as `pp1`/`pp2`/`pp3`. The in-loop infill-only control and the reader-copy
> arm are retired on the nested DP parse (flat within 0.04 under both-loss shaping on both seeds, §4) and on the
> never-trained twin ranking within 0.03–0.07 of the shaped trunk at the chooser's cell.

## Files

| file | purpose |
|---|---|
| `README.md` | this writeup, the super-node for the three rounds |
| `aliquot.py` | the substrate: `../sotto_voce.py` forked; the projection in the mirror's seat, its gates, the two paid arms |
| `analyze_aliquot.py`, `mk_seedtable.py` | the reducer and the two-seed table of record (`figures/al_seedtable.txt`) |
| `gates/falsify.py` | the falsification harness, 73/73 |
| `launch_detached.py`, `fetch_compact.py`, `results/wait_app.sh` | the launcher, the compact mirror, the restart-proof waiter |
| `results/RUN_*.sh` | the commands of record |
| `SPEC.md`, `DESIGN.md`, `FILES.md`, `CONVERSATION.md` | the brief; the decisions and withdrawals; the machinery record; the session record |
| `duplex/` | the shaping step offline ([`duplex/README.md`](duplex/README.md)) |
| `soundboard/` | the shaping step in the loop ([`soundboard/README.md`](soundboard/README.md)) |
| `preplay/` | the projection read offline on banked state: within-context discrimination and the candidate entry's price ([`preplay/README.md`](preplay/README.md)) |

[^private]: Not mirrored: this link points to a document in the private lab repo (the roadmap, the queue, an unrun spec, reading notes, or a conversation). See the top-level README for what is held back and why.
