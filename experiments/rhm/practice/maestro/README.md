# maestro — A2: learn the rule. The reward's type makes the judge; learning adds a timing refinement gauge-choice already paid for

**Up**: [`../README.md`](../README.md) (practice arc) · **Asked in**: `ROADMAP.md`[^private]
§4.1, shape A2 (this node is its record; the roadmap entry plus one orchestrated conversation
served as the spec) · **Files**: [`FILES.md`](FILES.md) (machinery, the learned class stated
plainly, the offline fit, gates, arms)
**Direct donors** (untouched): [`../conductor/`](../conductor/README.md) (A1 — substrate fork,
the hand-written comparator, the measured floors, and the logged corpus the fit trains on) ·
[`../../directed_sculpting/full_loop/composed_loop/`](../../directed_sculpting/full_loop/composed_loop/README.md)
(the negative control's lineage: the one prior learned endogenous judge, rewarded within-level,
had no upward direction) · [`../census/`](../census/README.md) +
[`../assay/`](../assay/FILES.md) (the yoke and stream-twin mechanics; the stream floor).
**Runs**: `ma_smoke` (mechanics), `ma_s0` (main, 5 arms, 2.08 GPU-h), `ma_s1` (stream-displaced
twins, 0.91 GPU-h), 2026-08-28. **Ranks, signs, and multiples of measured floors are the
claims** — and this node measured its own displacement floors for
its two headline arms, which demote one of its own cells below. One orchestrated conversation;
built end-to-end by one delegated implementer agent.

## The question

A1 measured that a hand-written thermostat reading the learner's own one-level-up currency
drives the crank as well as the schedule inside the earnable range and better beyond it. A2
asks the roadmap's next question: **does *learning* the rule buy anything the thermostat did
not?** Same action set (commit/hold per level, era advance); the rule now a small learned
policy over (gauge readings, level state) whose **reward is next-level yield**; graded against
A1's `outer_yield` (the hand-written comparator, replicated in-tag) and against a
**within-level-rewarded twin of the identical learned class** — `composed_loop`'s question
re-asked with the action set that matters, and with information access controlled: both learned
arms read the same inputs, so only the reward's *type* differs. Track E's per-datum signal does
not exist yet, so the reward is `at_support` alone (the roadmap's sanctioned fallback).

## Design in brief

`maestro.py` forks `conductor.py` (`# [maestro]` markers; the fork replays its donor and, in
the strongest gate available, `ma_s0/outer_yield` replays `cd_s0/outer_yield` **bit-for-bit
over all 139 cycles**, so A1's thermostat is in-tag as an exact comparator). The learned rule
is a **level-state-indexed, unit-norm gauge mixture in floor units** — act when the mixed
quieting statistic crosses a fitted threshold — a class that **contains A1's thermostat exactly**
as the point `a = e_yield, θ = 1` (gate: 0 disagreements over 401 decisions), is scale-free
(shrinkage cannot make an arm inert; the literal fitted-value form was built first and rejected
offline for exactly that failure), and reads level state as a 2-cell index (`will_commit`:
would a firing install a table, or move the world).

**The fit is across runs, not within one — stated as a design fact.** Both actions are
absorbing (a commit freezes the ratchet; an era advance moves the world), so a run contains
single-digit action events and within-run trial-and-error is structurally impossible — the same
wall that broke `teacher_slot`'s round-1 bandit, met honestly rather than re-run into. The
policy is fitted offline on the logged corpus of prior turns of the crank (`cd_s0`'s four
distinct trajectories + `cd_ef`; 257 windows, leave-one-arm-out, features and targets separated
by a 1-cycle gap after an instrument-measures-itself artifact was caught pre-GPU) and then run
live, greedy against the measured floors. This is the slow-timescale form of "meta-learning as
value judge" — a judge that integrates a lifetime of logged experience — and it means what is
learned here is the **value of states**, not yet the value of acting: the corpus holds 6 commit
events, and nothing in this node estimates action values.

Arms (the donors' seed family, A1's era caps 60/50/15/12/9): `anchor` (schedule
replica; cross-tag replay carrier), `outer_yield` (A1's thermostat, in-tag), `learned_yield`
(the treatment), `learned_task` (the negative control — identical class, inputs, fitting,
exploration; within-level reward), `yoked_learned` (clock replay of the treatment). `ma_s1`
adds stream-displaced twins of the two headline arms (the `as_s1` burn mechanic, substrate
identity asserted in-job and again at merge).

## Findings

1. **The reward's type determines what the judge becomes — with information access
   controlled.** From identical inputs, class, and fitting, the two rewards produce
   near-orthogonal mixtures (cosine +0.19 / +0.56). The yield-rewarded mixture is
   yield-dominant with a real ledger term at installs and a real endo term at advances, and
   predicts forward next-level yield **11–13% better than A1's raw gauge, out of arm** (LOAO
   mse ratios 0.866 / 0.892). The within-level mixture is ledger-*negative* dominant with
   commit-bucket R² ≈ 0. Live: `learned_task` committed L2 at **c14 — before its own shadow
   certificate fired (c17)** — then **never took L3 while its own input read the yield gauge at
   2+ floor units** (2.11 at the era-2 cap it rode). It ends with no L3 table, L3 proposal mass
   0.000 in every era, T3-built 0, and the worst mid-era value (era-3/4 Δ −0.168/−0.124, ~1.4–1.9×
   floor). The prior form of this result (`composed_loop`) left open whether the within-level
   judge lacked the signal; here it demonstrably had it. The crossing's value is not *invisible*
   to a within-level objective — it is *unvalued* by it.

2. **What survives stream displacement — and what does not.** Each gauge-driven arm's era-5
   gain over the schedule holds on both stream draws, above the earning-family floor
   (thermostat +2.89×/+3.20×; learned +3.83×/+2.26×): **the gauge-driven crank's deep-era win
   is stream-robust**. The era-5 *ordering between* learned and thermostat (+0.082) **flips
   sign under displacement** (−0.082) and is demoted on the record. Era 4 is the cell where
   learning shows an edge that holds: learned > thermostat on both draws (+0.030 → +0.145),
   and learned > anchor above floor on both draws (+1.11×, +1.32×) while the thermostat's
   era-4 delta flips sign (+0.77× → −0.35×).

3. **Where the learned rule differs from the thermostat is timing, and it pays in time more
   than in tables.** It committed earlier at both levels (L2 c39 vs c49; L3 c85 vs c92) with
   *thinner* tables (L2 13 entries/0.643 recall vs 14/0.714; L3 18/0.089 vs 21/0.125), matched
   or beat the thermostat's deep-era value, and finished in 130 cycles vs 139 at ~8% less
   priced time — with the best end-of-run battery of any arm (a_full solved 46 vs 32). Its
   earlier L2 commit visibly accelerated the level *two* above: 24 distinct L4-shaped tuples at
   support by era-1 exit vs the thermostat's 11. Under displacement, **both rules commit L2 at
   exactly c26** (originals 10 cycles apart), the thermostat's advance cycles barely move
   (+3/+3/+1/−1/−2) while the learned rule's shift en bloc (−24 after era 1), and the learned
   arm's own displacement magnitude runs ~1.8× the thermostat's — the learned rule is the more
   stream-sensitive instrument, and commit timing looks substantially substrate-driven with the
   rules adding offsets.

4. **The offline→live chain validated exactly.** The pre-GPU counterfactual replay predicted
   the learned rule's first divergence from the thermostat at c39; the live run diverged at
   c39. Yoke identity (0.000e+00 over 130 cycles, priced-time difference exactly 130 × 267 g)
   re-verified that the new machinery's reads are inert.

## Interpretation (discussed with Jasper 2026-08-28 — argued, not measured)

- **(a) The typed-signals law sharpens from visibility to valuation.** §2.2's necessity claim
  said the crossing's value is "invisible to the inner loop by construction." Finding 1 says
  the sharper thing: a within-level objective can be handed the one-level-up currency as an
  input and still assign it no weight. The gate is the reward, not the sensor. (Queued for the
  `/update-beliefs` sweep.)
- **(b) Gauge choice remains the scarce part; judgment buys a refinement.** The robust wins
  belong to *any* rule reading the one-level-up gauge (A1's result, re-confirmed on a second
  stream draw here); learning added earlier action, less priced time, and the one held cell at
  era 4 — real, but a refinement of a decision the gauge had already made. `teacher_slot`'s
  "the scarce thing is which gauge is consulted" survives its second test.
- **(c) The arrival mechanism echoes on the earning side.** Earlier commits with thinner books
  matching later commits with fatter ones is the census/assay "arrival dominates" claim,
  produced this time by an endogenous loop's own choice — and the L4-observation acceleration
  after the early L2 commit is the first sighting of the crank compounding upward through
  timing alone. Suggestive, not certified.
- **(d) The across-run form is the biologically natural one** (Jasper's framing): a
  slow-timescale learner is, almost by definition, one that integrates a lifetime of episodes
  rather than exploring within one. Whether something hippocampus-like — parallel candidate
  tracks entertained within a run — could restore within-run action learning over absorbing
  actions is noted as a genuine open direction, deliberately not pursued here.

## Caveats

- **`ma_s1` displaces stream position only** (`--seed 0`, substrate asserted identical
  twice): it can demote or support orderings. Its floors are measured on these two arms specifically and differ from
  census finding 7's pooled 0.087; where they disagree, the in-node number is the narrower
  claim.
- **The action-value gap.** 6 logged commit events; the learned object is the value of states.
  A rule that learns from the consequences of its own acting remains unbuilt.
- **Arms are not lifetime-matched** (125–139 cycles vs the anchor's 116), as in A1: pacing is
  what the loop owns and pays for, but per-era comparisons carry it.
- The endo read was charged at A1's pinned 267 g against this run's own bench of 268 (−0.4%);
  behavior-neutral by construction and by yoke identity.

## Runs on disk

| tag | what |
|---|---|
| `ma_smoke` | five arms end-to-end at `--quick`; mechanics only |
| `ma_s0` | the main run: 5 arms, A1's floors and caps, fitted mixtures |
| `ma_s1` | stream-displaced twins of `outer_yield` and `learned_yield` |

Volume `rhm-scaling-data:/data/rhm_practice_maestro/<tag>/`; fetched copies, figures, and the
merged `reduction.txt` (§0–§8 the main run, §9 the twins) under `figures/ma_s0/`.

## Reproduce

Full command set in [`FILES.md`](FILES.md) §Reproduce: the offline phase first
(`fit.py`, `policy.py` gates — no GPU), then preflight, smoke, `ma_s0`, and `ma_s1` with
`--ref-tag ma_s0`; reduce with `analyze_maestro.py --tag ma_s0 --merge-tag ma_s1 --fetch
--figures`.

## Next steps (queued, not started)

**A3 — the signature**: run as this node's sibling [`../crescendo/`](../crescendo/README.md),
on the thermostat per this node's verdict — the range extends on both stream draws · the `/update-beliefs` sweep (roadmap §4.6 wanted it
before A2's interpretation hardens; it should carry finding 1's visibility→valuation
sharpening) · the action-value gap (a judge that learns from its own acting — possibly via
parallel candidate tracks within a run, the hippocampus-shaped speculation) · Track F's F1
instrument (both A-nodes now carry π-concentration trajectories it would formalize) · the
era-pacing dissociation control shared with A1's queue.

[^private]: Not mirrored: this link points to a document in the private lab repo (the roadmap, the queue, an unrun spec, reading notes, or a conversation). See the top-level README for what is held back and why.
