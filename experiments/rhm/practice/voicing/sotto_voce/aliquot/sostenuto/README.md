# sostenuto — the value reader's gate on the miner's live build, in the loop: a present-cell gate prunes the level above, a next-level gate is empty at the frontier, and the currency wants admit-then-grade

**Up**: [`../README.md`](../README.md) (aliquot, the three in-loop rounds whose plants and readouts this node gates with) ·
**Spec**: [`SPEC.md`](SPEC.md) (the orchestrator's briefs for both rounds, verbatim) · **Decisions, gates and defects**:
[`DESIGN.md`](DESIGN.md) (§1–§8 round 1, §9 round 2) · **Machinery record**: [`FILES.md`](FILES.md) ·
**Conversation**: `CONVERSATION.md`[^private] (the session behind both rounds, 2026-09-21 → 23) ·
**The offline rounds this puts in the loop**: [`../preplay/README.md`](../preplay/README.md) (pp4, the read as the order;
pp5, the read as the gate) · **Motivation**: [`ideas/calibration_and_violation_are_one_object.md`](../../../../../../../ideas/calibration_and_violation_are_one_object.md)
§12.4, the replay note, and `ROADMAP.md`[^private] §2.1's currency claim: the crossing's value is
denominated one level up and a within-level signal cannot read it.
**Runs**: 2026-09-21 → 23; round 1 `st_s1` / `st_s2` (three arms per seed, ≈20 GPU-h with gates, the preflight and one
relaunch after a container fault), round 2 `st2_s1` / `st2_s2` (two arms at seed 0, one at seed 2, ≈6.3 GPU-h); every
arm clock-yoked to its seed's banked anchor, the banked `sb_sv_yk` at each seed as the no-walk reference and never
re-run; seeds 0 and 2 throughout and never averaged. **Ranks, signs, located mechanisms and per-cell counts are the
claims.**
**Attribution**: the ask to take stock of the value line and of preplay's role, the simplification push and the
question whether the in-loop run was ready, the reframe that the gate should read next-level rather than current-level
value since the learner assesses the current level itself, the observation that a gate's selectivity ought to be
calibratable, and the recognition that admit-then-grade is the format the currency wants are Jasper's (2026-09-21 →
23). The finding that the census extension op never ran on the arms of record and the re-aiming of the seat at the
live build, the reading of the coverage trace as demand blindness in the value seat, the starved-seat reading, and the
admit-then-grade conclusion are the orchestrator's, agreed in discussion. The builds, the gate tables, the observation
panel as the next level's miner, the proof that the literal demand form is the ungated arm, the four round-1 and one
round-2 defects beside their corrections, and the seed-2 relaunch are the two implementers' (one Opus session per
round).

## One-liner

pp5 put the shaped projection's level in the seat of the world's verdict on a try-and-keep walk over candidate table
entries, offline, and found it captures a quarter of the oracle gate's advantage at the loop's budget. This node
built that seat inside the loop, on the arms of record, and found first that those arms had no table gate at all:
the census extension op pp4 and pp5 transcribed never ran on them (`recert` on, `extend` off, open inventory, zero
extension events), and their operative table is the miner's live support-count build, rebuilt on most cycles. The
seat is therefore an **admission set on the live build**, walked at the loop's own cadence, cap and pool in the
miner's count order, and the gate is the one knob. **Round 1**, the world's error / the read's strict pooled level /
nothing, two seeds: the seat works mechanically at zero world queries and costs seconds; the read gate admits less
junk than no gate on both seeds and refuses good entries at three to eight times its offline rate; any gate in this
seat decides on a median of one to five changed instances of 192, so it is starved; and **a refusal at one level is
paid at the next, for both organs** — the world's L2 refusals block seven of the ungated arm's L3 keys on each seed,
the read's L3 refusals block three of nine and eight of twenty-two L4 keys — while the deep-era task-error ordering
inverts across seeds and carries no direction. **Round 2**, the gate in the next level's currency (the fired state's
next-level share at support in the learner's own miner times the read's level; and a free comparator off the miner's
counts alone): the currency is **empty at the frontier by construction**, since the level above holds nothing at
support until the keys below are served, so both gates are silent at the top of the ladder and prune hardest where
they act early on thin evidence, blocking fifteen of twenty-four L3 keys at seed 2 and seventeen of nineteen for the
demand gate, and cancelling commits on empty builds. **The order of the op is what is wrong, not its organ**: a
next-level currency's evidence arrives after admission, so the op the currency wants is admit-then-grade —
provisional admission, consumption one level up, revocation on positive evidence — which is the arc's earlier finding
that provisional commitment graded in consumption beats every certificate, recurring in the value seat.

## The question

`preplay` pp4 and pp5 measured the read in two seats of a try-and-keep walk and concluded that the producer's own
score should order what to try and the value reader's level should decide what to keep, at zero world queries. Both
rounds were offline on the banked plants at the run's last state, and both transcribed `census_extend`, the census
lineage's op over a frozen committed table, as "the loop's own consumer of candidate entries". Two questions were left:
does the seat turn the crank inside the loop, where the plant, the readout and the table co-evolve; and what does the
gate cost when its refusals compound?

The first design step found the premise wrong. On the arms whose plants and readouts pp1–pp5 read (`sb_sv_yk` at seeds
0 and 2), the census op never ran: the arms carry `recert` but not `extend`, run `open_inventory`, and logged zero
extension events. Under open inventory the commit freezes only a flag; the table the executor runs over is the miner's
live build over the operative lower table, rebuilt on 157 of 201 cycles at seed 0, and no decision consumes a world
audition of it (the commit is by clock yoke and its audition is logged only; the recert is measured and never acted
on). Their consumer of candidate entries is the miner's count threshold and the executor's argmax, with no gate, and
pp3's tables, where the world barely separates the learner's true rows from its false ones at L3 and L4, are what that
produces. A further defect made the census op unrunnable there even in principle: its candidate lister slices keys as
flat tuples, which is the plain miner's key, and under the class miner every lookup misses (gate A-10). No banked
result is affected; the only arms that ever carried `extend` ran the plain miner.

So the question became: put the gate in the seat these arms actually have, the live build, and ask what each organ
in that seat does to the table, to the level above, and to the learner's own task error, cycle for cycle against the
arm of record.

## What was built

`sostenuto.py` forks `soundboard.py` at its `sb_s1`/`sb_s2` head, every addition marked, every knob default off, and
replays the donor bit for bit with the knobs off (G-F, run four times). One op, the **admission set**: under open
inventory `operative(ell)` serves only rows of the live build whose key has been admitted. At the recert cadence (5),
for each level with keys at support, the not-yet-decided at-support keys are offered in the miner's own count order up
to the loop's cap (8); each is fired as base ∪ {candidate} on a fresh gate pool of the era's cell (192 instances, the
loop's `n_aud`), and the gate decides. A candidate is a class-miner key and so a group of up to sixteen interchangeable
rows; the offline walk gained a default-off group mode so the in-loop walk still reproduces it (A-1c re-walks every
pass through `readgate.gated_walk`, with a pass separating the arm's gate from the others found on every arm). Two
design choices, reasoned in [`DESIGN.md`](DESIGN.md) §1 before paying: a key is decided once, with a merge's re-keying
as the one second chance; and the admission set starts empty at every level, so the gate touches every row, with a
pending state that serves the ungated build until a level's first pass. The world's error and the executor's winning
entry are recorded on every trial whatever the gate decides, so the per-candidate confusion against the world gate
exists on every arm, and only the world-gated arm is billed for its auditions.

Five gate values across the two rounds, every other knob `sb_sv_yk`'s:

| gate | decides on | reads the world? |
|---|---|---|
| `world` | the world's error on the fired gate pool does not rise (pp4's rule) | yes, billed |
| `read` | the shaped projection's mean level over the same fired configurations does not fall (pp5's strict pooled form) | no |
| `none` | admit everything offered, the walk's own floor | no |
| `yield` | the fired state's share of level-(l+1) spans whose key is at support in the learner's own observation-panel miner, from the reader's parse, times the read's level, pooled, does not fall; acts on the share alone before the readout's first fit | no |
| `demand` | no fire: the candidate's class is a half of at least one next-level key at support; silent where the next level holds none | no |

The cheap things the offline rounds asked for ride on every arm: the world-model diagnostics every fifth cycle, the
heads dumped at every era boundary and once more after a final refit on the final core so pp1's one-update skew is
resolved, and the admitted table's error on a disjoint test pool at the end of every pass. Every gate was shown to fail
on a perturbation before being reported (8/8 and 11/11 falsified); the gates, arms, runs and app ids are in
[`FILES.md`](FILES.md).

## 1. Round 1: the seat works, and any gate in it is starved (`st_s1`, `st_s2`)

Every gate green on all six arms; the walk costs 5.6–14.8 s an arm; the arms run three to nine percent slower per
cycle than their references, which is the diverged run's own beam work and not the op. World reads billed to the
table's decisions: `world` 13,632 / 15,552, `read` 0, `none` 0.

**The read gate's composition holds, and its price is refusals.** Over L3–L5, the only levels where it had a live
decision (the readout is not fit before cycle 50 / 60 and the L2 era ends at 48 / 59, so every L2 decision on the
`read` arm equals the `none` arm's):

| L3–L5, seed 0 / seed 2 | n | junk admitted (world would refuse) | good entries refused (world would keep) |
|---|---|---|---|
| `read` | 28 / 31 | 0.036 / 0.000 | 0.143 / 0.419 |
| `none` | 35 / 55 | 0.114 / 0.073 | 0.000 / 0.000 |
| pp5 offline, same budget | | 0.022 (ungated 0.045) | 0.050 |

Inside L3's own era, the read-gated table audits worse than the ungated build on the same pool and pass on three of
three passes at both seeds (+0.018, +0.214); the world-gated table audits better or neutral (−0.068, 0.000).

**The seat is starved.** A candidate changes a median of 1 (seed 0) and 5 (seed 2) of the 192 gate-pool instances, and
on 15 of 32 and 14 of 37 fitted trials it changes none, where both gates admit by identity. The world's verdict is
exact on that handful; the read's is a level difference on the same handful divided by 192. This is pp1's dead delta
form inside the loop, on the op the census lineage gave it.

**A refusal at one level is paid at the next.** Of the keys the ungated arm ever supported at level l, those with a
half the gate arm refused at l−1:

| seed | gate | L2 keys refused | L3 keys refused | none-arm L3 keys blocked | none-arm L4 keys blocked |
|---|---|---|---|---|---|
| 0 | `world` | 5 | 2 | 7 / 19 | 0 / 9 |
| 2 | `world` | 5 | 0 | 7 / 24 | 0 / 22 |
| 0 | `read` | 0 | 7 | 0 / 19 | 3 / 9 |
| 2 | `read` | 1 | 14 | 1 / 24 | 8 / 22 |

The world gate refuses at L2 inside the first era, when the damage cell is still at L1, and costs seven L3 keys on each
seed; the read gate refuses at L3 and costs L4 keys on each seed; at seed 2 the read arm reached L5 with one row where
the ungated arm had 144. At L5 the test is silent on both seeds (the arms had diverged; the halves were never offered).
This is `spiral` and `census`'s demand-blindness finding, a gate under present demand cannot price coverage whose value
lives under future demand, measured in the value seat with the same sign on both seeds and both organs.

**The deep-era task error carries no direction.** Against each seed's banked reference, eras 4 / 5: `world` +0.127 /
+0.207 and +0.006 / −0.029; `read` −0.041 / −0.082 and +0.020 / +0.105; `none` −0.014 / +0.020 and −0.008 / −0.086.
The seed-0 ordering inverts at seed 2, the windows are twelve and nine cycles, and the `none` arm's own footprint
reaches 0.09.

## 2. Round 2: the next-level currency is empty at the frontier (`st2_s1`, `st2_s2`)

The committable level-(l+1) miner is fed only up to `era_level + 1`, so it holds nothing in exactly the era when
level-l keys are offered; both round-2 gates read the learner's observation panel instead, which counts the same
reader parse at every level and is what the lineage's merge op already reads for its one-level-up licence. Even there
the next-level share is zero on every fire until the level above holds keys at support: for L4 keys until about cycle
180 and for L5 keys until about 190 at seed 0, and at every logged cycle at seed 2. So both gates are silent at the top
of the ladder (they admit by tie) and act only at L2 and L3, where the evidence is thinnest and earliest. The brief's
literal demand form, the raw count of next-level keys having the candidate as a half, is at least the candidate's own
count on every offered key at L2–L4 and would admit everything; what ran is the count at support.

| seed | gate | L2 refused (era 1) | none-arm L3 keys blocked | none-arm L4 keys blocked | commits |
|---|---|---|---|---|---|
| 0 | `yield` | 3 of 13 | 3 / 19 | 0 / 9 | all four |
| 2 | `yield` | 8 of 13 | 15 / 24 | 2 / 22 (20 never offered) | L4 at c157 cancelled on an empty build; L4, L5 never adopted |
| 0 | `demand` | 6 of 14 | 17 / 19 | 0 / 9 | L5 at c186 cancelled on a one-key L4 build |

The yield gate's own difference on a trial is ±0.002 to 0.018: only the next-level span containing the candidate's
node can change between trial and base, so the rule decides on the sign of one span in eight. Its pre-fit L2 trials
split 7 zero / 3 positive / 3 negative at seed 0 and 3 / 2 / 8 at seed 2, and each negative is a permanent refusal
under decide-once. Over L3–L5 the yield gate is lenient (junk admitted 0.167 / 0.000, good refused 0.033 / 0.125); the
demand gate refuses 0.409 of what the world would keep. Task error against the `none` arm, eras 2 / 4 / 5: `yield`
−0.055 / +0.090 / +0.120 at seed 0 and +0.014 / +0.261 / +0.356 at seed 2; `demand` +0.009 / +0.280 / +0.267. The
yield gate at seed 0 had the best L2-era error of the five arms and the most grammar-true L2 and L3 tables, and still
ended worse than no gate at depth; the two cancelled commits break the yoke on those arms, so their late eras are not
clock-matched and the cancellation is itself the reading. The empty-base residual risk DESIGN §1.2 named fired on two
of three round-2 containers and never in round 1.

## The update

1. **The seat exists and is free.** An admission set on the live build, gated by the read's level, runs at zero world
   queries with every fidelity gate green and the walk a ten-thousandth of the arm. The instrument costs the lineage
   carried as beliefs are mostly imaginary: the world-model diagnostics cost 12.8 s of an 8096 s arm.
2. **Any gate in this seat decides on a handful of instances.** Base-plus-candidate at the era's cell gives the executor
   almost no occasion to use a new key, so the world's decision is exact on one to five instances and the read's is a
   compressed level on the same. pp1's single-entry form, where the executor must use the candidate on every instance,
   is where the read priced in the world's order; the delta form is where it is dead, offline and in the loop.
3. **A refusal at level l is paid at level l+1, for both organs and both currencies.** Present-cell gates prune the
   halves the next level needs (round 1, both seeds); next-level gates prune them harder where they act early on
   thin evidence (round 2, both seeds). The gate's own instrument improves while it does so. This is the practice arc's
   demand-blindness result recurring in the value seat, and it is the one effect that held everywhere.
4. **The next-level currency is empty at the frontier by construction, so the op's order is wrong.** The level above
   acquires a vocabulary only if the keys below are served; a gate that asks for next-level evidence before admitting
   asks for evidence that cannot exist yet, and one that refuses on its absence destroys the level above. The op the
   currency wants is admit-then-grade: serve a key provisionally, let the next level try to build on it, revoke on
   positive evidence that nothing above uses it, and re-offer rather than decide once. That is `ear` and `recital`'s
   finding that provisional commitment graded in consumption beats every certificate, in the value system's seat.
5. **The read's role in the picture is unchanged and located.** It prices a fired state in the world's order (pp1–pp5)
   and filters junk without the world (both seeds here); what it cannot do in this op is license coverage, and neither
   can the oracle. The selectivity dial Jasper asked for has a direct handle, the admission margin, and every trial's
   difference under all four gates is on the record so it can be set from data; what the record already says is that
   the dial sits at "admit" at the frontier and tightens one level below it, where evidence has accrued.
6. **Inventory is the budget.** The cap of eight never bound, key arrival is the constraint (43–73 decisions an arm over
   201 cycles), and nineteen to forty-two at-support keys are unbuildable at L4. A future round on this question should
   buy keys per run, not cycles, which is also the case for a minimal loop over the six objects the question needs.

## What this does not show

The deep-era task-error ordering is not a finding in either direction: it inverted across seeds and its windows are
twelve and nine cycles. The read gate never had a live decision at L2 on any arm, so its L2 rows equal the ungated
arm's by construction and every L2 comparison of the read against the world is vacuous. The coverage test is silent
at L5 on every arm (the halves were never offered on the gate arm), and its `half never offered` column is the
honest residual of the arms' divergence rather than evidence. The round-2 gates read the observation panel's miner,
not the committable one, by a builder's decision reasoned before the run; the committable miner would have made both
gates identically silent or identically refusing. The demand gate ran at seed 0 only, by judgement that its failure is
structural and far outside round 1's seed spread. Nothing here re-poses the gate against future demand or builds the
admit-then-grade op; both are design steps. The loop has no resumable checkpoint, so every arm is a full run. Nothing
off the RHM practice substrate.

## Reproduction

```bash
cd experiments/            # MODAL_PROFILE=chromatic
B=rhm/practice/voicing/sotto_voce/aliquot/sostenuto
modal run $B/sostenuto.py::sostenuto_gates           # A-gates, both rounds, and their falsifications (8/8, 11/11)
modal run $B/sostenuto.py::sostenuto_gates_gpu       # A-2: the in-loop read == preplay.pj_predict
modal run --detach $B/sostenuto.py::fidelity_smoke --tag st_gf1          # G-F against soundboard.py
modal run --detach $B/sostenuto.py::preflight --outdir-tag st_pf1        # the whole in-loop path on the preflight twins
sh $B/results/RUN_st_s1.sh      # round 1, seed 0: world / read / none      (RUN_st_s2 is the seed-2 form)
sh $B/results/RUN_st2_s1.sh     # round 2, seed 0: yield / demand           (RUN_st2_s2: the yield arm at seed 2)
python3 $B/fetch_compact.py --tag st_s1 && python3 $B/reduce_sostenuto.py --tag st_s1
python3 $B/mk_seedtable.py      # the two-seed tables of record, figures/st_seedtable.txt and st2_seedtable.txt
```

The exact flags, the per-arm kwargs (`<tag>/sweep.json`) and every app id are in [`FILES.md`](FILES.md); the design
and gate rationale in [`DESIGN.md`](DESIGN.md) §8 and §9.8. Volume `rhm-scaling-data:/rhm_practice_sostenuto/<tag>/`.

## Next steps (queued in `QUEUE.md`[^private], not started)

The admit-then-grade form of the op, a design conversation before a build: provisional admission, the revocation
evidence (what "nothing above ever uses it" is, in the miner's own currency), re-offering, and the margin set from the
recorded differences · pp1's single-entry fire in place of the delta form, so the gate is not starved · the readout's
first fit before L2 is decided (`vo_om_min`) · a substrate with more keys per run · the minimal crank over the six
objects this question needs, gated against the no-gate and gated arms banked here, as the vehicle for the port.

## Files

| file | purpose |
|---|---|
| `README.md` | this writeup, both rounds |
| `sostenuto.py` | the substrate: `../soundboard/soundboard.py` forked; the admission set, its five gates, the A-gates and their falsifications, the arms |
| `reduce_sostenuto.py`, `mk_seedtable.py` | the reducer and the two-seed tables of record |
| `launch_detached.py`, `fetch_compact.py`, `results/wait_app.sh` | the launcher, the compact mirror, the restart-proof waiter |
| `results/RUN_*.sh` | the commands of record |
| `SPEC.md`, `DESIGN.md`, `FILES.md`, `CONVERSATION.md` | the briefs; the decisions, gates and defects; the machinery record; the session record |
| `figures/` | `st_seedtable.txt` (round 1, both seeds), `st2_seedtable.txt` (round 2 beside round 1, both seeds), the per-tag reductions, the compact mirrors and panels under `st_s{1,2}/` and `st2_s{1,2}/` |

[^private]: Not mirrored: this link points to a document in the private lab repo (the roadmap, the queue, an unrun spec, reading notes, or a conversation). See the top-level README for what is held back and why.
