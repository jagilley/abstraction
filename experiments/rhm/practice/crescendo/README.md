# crescendo — A3: the signature. The earnable range extends with the turn of the crank, on both stream draws

**Up**: [`../README.md`](../README.md) (practice arc) · **Asked in**: `ROADMAP.md`[^private]
§1.4 (the signature — the integrated learner's one distinctive prediction) and §4.1, shape A3
(this node is its record; the roadmap entry plus one orchestrated conversation served as the
spec) · **Files**: [`FILES.md`](FILES.md) (machinery, Phase 0, gates, the one-bit ceiling
control stated plainly)
**Direct donors** (untouched): [`../maestro/`](../maestro/README.md) (substrate fork; A2's
policy *imported*, not forked — no new rule) · [`../conductor/`](../conductor/README.md) (the
thermostat and the measured floors) · [`../census/`](../census/README.md) (the yoke and
extension machinery; finding 1's r² law; finding 7's stream floors) ·
[`../spiral/`](../spiral/README.md) (interpretation (c) named this round's question:
whether the thin frozen L3 is benign or a foreclosure "is not measurable at this depth — L4
is unearnable at any affordable budget").
**Runs**: `cr3_gf`, `cr3_smoke` (gates), `cr3_s0` (main, 5 arms, 2.66 GPU-h), `cr3_s1`
(stream-displaced twins of the signature pair, 1.09 GPU-h), 2026-08-28→29; node total 5.00
GPU-h. **Ranks, signs, and multiples of measured floors are the claims** — and
the signature cell is read against this node's own measured displacement floors as well as
the pooled ones. One orchestrated conversation; built end-to-end by one delegated implementer
agent.

## The question

Every run in the arc capped commitment at level 3 and treated level 4 as unearnable; the
spiral measured the value clock holding *only inside the earnable range*, with that range
fixed by the teacher's ladder. §1.4's prediction — the one none of the parts can make alone —
is that **the earnable range extends with each turn of the crank**: the outer loop, reading
next-level yield, reaches the rung the previous turn made readable. A3 opens L4 to the crank
(`max_macro_level=4`) on the same depth-6 world and asks: does the loop-paced learner earn
the new rung, and **does the value clock hold past the range the previous turn certified?**

## Design in brief

`crescendo.py` forks `maestro.py` (`# [crescendo]` markers; G-F replays the donor at
0.000e+00; the live `outer_yield_m4` replays `ma_s0/outer_yield` bit-for-bit over its first
106 cycles). **Phase 0 sized the whole round offline before any GPU** (`phase0_l4.py`, gate
B-1: the substrate's own `Miner.build` replayed on the logged key streams reproduces all 18
logged commit events entry-for-entry): the level sizes (L2 14 / L3 56 / **L4 816** / L5
205,824 distinct tuples); the **r² wall** (an L4 entry is buildable only over committed L3
children, so the frozen L3 books cap buildable L4 at 13–15 entries, doubling to ~25 over the
live tables — which is what justifies the extension arm); the era-3 commit window (the
thermostat replayed on the donors' own L4 series first quiets at c_in_era3 17–30, sizing era
3's cap at 70); and the **L5 gauge's degeneracy** (1–2 distinct tuples per run), so the L4
commit is paced on the L4 stream's own quieting.

Five arms, ladder `60/50/70/12/9`: `anchor_long` (the schedule at the caps — the
budget-matched comparator), `outer_yield_m4` (A1's thermostat, permitted L4 — the treatment),
**`ceiling_m3`** (the round's control: a clock yoke of the treatment carrying one extra bit,
`commit_max_level=3` — verified to bind only on the level it names and to move nothing while
non-binding, so the pair is **lifetime-identical and one-bit-differing**, closing A1/A2's
lifetime caveat by construction), `outer_yield_m4x` (extension enabled — the r²-wall bypass),
`outer_yield_m3` (the gauge-paced rule with L4 forbidden — the pacing comparison). `cr3_s1`
rebuilds the treatment/ceiling pair on a displaced stream (1024-draw burn; the ceiling twin
yoked to the *displaced* treatment's realized actions), so the signature exists on two draws.

## Findings

1. **The crank reaches the new rung, and on the loop's own signals.** `outer_yield_m4`
   committed L4 at c129 (23 cycles into era 3) with its shadow certificate having fired at
   c127 — armed on the L4 stream its own consolidated L3 policy was generating, quieted,
   committed, certified. On the displaced draw the crossing **recurs** (c121, 12 into era 3).
   Both books are demand-concentrated slivers — 5 and 4 entries of 816 (recall 0.001–0.003,
   precision 0.40/0.25) — landing inside Phase 0's predicted 2–5 buildable window both times.
   The schedule arm also crossed, but differently: `anchor_long`'s L4 arrived only at the era
   boundary, provisional, its certificate never firing. The loop *chooses* the rung; the
   schedule stumbles onto it.

2. **The signature: the value clock holds past the previously-certified range, on both
   draws.** Against `ceiling_m3` — same seed, same stream, bit-identical to the commit cycle
   (first divergence exactly there on both draws: c129, c121), lifetime-identical (162=162,
   151=151), differing in one installed table:

   | draw | era 3 (L4's earning era) | **era 4** | **era 5** |
   |---|---|---|---|
   | A | −0.012 | **+0.388** | **+0.519** |
   | B (displaced) | +0.044 | **+0.427** | **+0.929** |

   Eras 4–5 are the eras beyond the range A1/A2 could earn. **Era 4 holds sign and shape on
   both draws** and clears every floor — 4.5× the pooled earning-family floor, 2.6× the
   pooled given-family, and **12.9× this node's own measured era-4 displacement floor**
   (0.030). Era 5 holds sign on both draws at 1.27× its in-node floor (0.410) — a rank/sign
   claim, magnitude at the floor's edge. Era 3 flips sign below every floor — unreadable, and
   rightly: it is the *earning* era for L4, not the signature. All three L4-holding arms rank
   above both L4-forbidden arms at eras 4–5, on recovered fraction and raw error alike, with
   the extension arm first.

3. **Trust forms on the new rung, in real time, on both draws.** π's L4 proposal mass rises
   monotonically after the commit — 0.034→0.104→0.152 (draw A), 0.041→0.165→0.219 (draw B),
   argmax share to 0.201/0.326, with the single L4 slot that *is* era 4–5's damage cell
   carrying 0.177 alone — while every L4-forbidden arm sits at exactly 0.000 in every era.
   The mass is larger on the draw whose era-5 payoff is larger. The battery extends the
   address-book pattern one level: T4-at-support/T4-built 1/1 in every L4-holding arm, 0/0 in
   both ceiling arms.

4. **The r² wall is real, arithmetic, and extension is the measured lever.** Phase 0's
   frozen-recall² prediction of buildable L4 (13 predicted, 15 measured) held; enabling
   `census_extend`'s post-commit extension roughly doubled the books live (L2 14→18/0.857,
   L3 21→34/0.161, L4 5→9) and `outer_yield_m4x` posts the best deep-era values in the tag
   (era 4/5 recovered fraction 1.403/1.612). Recorded with its own caveat: extension moved
   that arm's whole trajectory from era 2 on, so it is not a controlled one-bit comparison —
   the m4/ceiling pair is.

5. **One trace-level fact worth the record**: at c129 of draw A, `outer_yield_m3` took an era
   *advance* on V/tol = 0.91 — the identical cycle and identical quiet statistic on which
   `outer_yield_m4` committed L4. The same signal licenses commit-or-advance; what differed
   was the affordance. "Every ladder tops out one rung above where the learner currently
   stands" rendered as two log lines.

## Interpretation (discussed with Jasper 2026-08-28→29 — argued, not measured)

- **(a) §1.4's prediction has its first measured instance.** The previous turns (L2, L3,
  consolidated into the routed policy) manufactured the observation stream; the outer loop,
  reading that stream, reached the rung; and the value denominated one level up — invisible
  to every within-level account — is real and collected exactly where the old range ended.
  §18's closing line ("the one thing the factory cannot manufacture is the price of its own
  next level") now has a counter-instance in-vivo, not just the teacher_slot argument.
- **(b) A sliver of a rung is worth a third of the deep-era gap** — 4–5 entries at recall
  0.002 buying +0.4 recovered fraction. Concentration-over-coverage (the spiral's and
  census's repeated lesson) recurses upward one more level, this time in the learner's favor.
- **(c) The wall above is named and priced.** The next rung's ceiling is set by this rung's
  frozen recall squared; extension is the measured lever, which moves Track F3's index ops
  from housekeeping to load-bearing for any further climb.
- **(d) Scope, kept loud**: one rung (two stream draws bound draw-luck, not
  world-luck); the demand ladder is still the teacher's — the loop paces within an exogenous
  curriculum, so §1.3's "semi-autonomous" stands; and whether the range extends *again* (L5
  needs an L4 book the r² wall currently caps near zero) is the next round's question, not
  this one's answer.

## Caveats

- **`cr3_s1` displaces stream position only**; the in-node floors it
  measures are per-arm and per-era (§9 of the reduction), and where they disagree with census
  finding 7's pooled values the in-node number is the narrower claim. One flagged artifact:
  `ceiling_m3`'s era-5 in-node floor reads 0.000 because both draws' last-3-cycle windows
  contain the same error multiset permuted — the treatment's 0.410 is the honest floor and is
  what the §F ratios use.
- Recovered fractions above 1.0 at eras 4–5 are expected, not errors: the "floor" is an
  exact-DP rollout over base moves only, which macro-holding arms legitimately beat (the
  spiral's caveat); raw `e` is tabulated beside every such number and carries the ranks.
- The endo read was priced at A1's pinned 267 g against this run's in-tag bench of 258;
  behavior-neutral — no arm drove it and every ledger spend is 0 (the yield read is free at
  every rung reached, which is itself a finding worth carrying: the roadmap's Assumption 1
  held in a stronger form than assumed).
- Two donor assertions hardcoded to `maxl=3` were generalized (recorded in `FILES.md`); the
  G-F and cross-tag replays gate that nothing behavioral moved.

## Runs on disk

| tag | what |
|---|---|
| `cr3_gf` | G-F in-process fork-vs-donor replay (0.000e+00) |
| `cr3_smoke` | five arms end-to-end at `--quick`; the s/cycle budget measurement |
| `cr3_s0` | the main run: 5 arms, Phase-0-sized caps, A1's floors |
| `cr3_s1` | stream-displaced twins of the signature pair, rebuilt one-bit on their own draw |

Volume `rhm-scaling-data:/data/rhm_practice_crescendo/<tag>/`; fetched copies, the three
figures, `phase0.json`, and the merged `reduction.txt` (§0–§9 + §A–§F) under
`figures/cr3_s0/`.

## Reproduce

Full command set in [`FILES.md`](FILES.md) §Reproduce: Phase 0 first (`phase0_l4.py`, no
GPU), then preflight, `cr3_gf`, `cr3_smoke`, `cr3_s0`, and `cr3_s1` with `--ref-tag cr3_s0`;
reduce with `analyze_crescendo.py --tag cr3_s0 --merge-tag cr3_s1 --fetch --figures`.

## Next steps (queued, not started)

**The second extension of the range** — L5 needs an L4 book the r² wall caps near zero, so
the extension/merge/re-key ops (Track F3) are now on the critical path of any further climb;
a seed pair scoped to the signature cell if it becomes load-bearing externally
([`experiments/CLAUDE.md`](../../CLAUDE.md)'s condition is met: meaningful, small-ish, and
only draw-luck triangulated) · Track F1/F2 (the trust-formation instrument and the
trust-vs-habit cut — finding 3's monotone L4 mass is the observation those would formalize,
and A3 cannot say which of the two it is) · Track E's per-datum gate, now with a live L4
frontier to gate on · the canvas port of the signature (Track C as the second venue).

[^private]: Not mirrored: this link points to a document in the private lab repo (the roadmap, the queue, an unrun spec, reading notes, or a conversation). See the top-level README for what is held back and why.
