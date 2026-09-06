# conductor — A1: the composition. The outer loop takes the certificate's seat, and the one-level-up reader is the one that can drive

**Up**: [`../README.md`](../README.md) (practice arc) · **Asked in**: `ROADMAP.md`[^private]
§4.1, first shape A1 (this node is its record; no separate SPEC was written — the roadmap entry
plus one orchestrated conversation served as the spec) · **Files**: [`FILES.md`](FILES.md)
(machinery, the rule stated plainly, the measured floors, gates, arms, caps)
**Direct donors** (untouched): [`../assay/`](../assay/FILES.md) (substrate fork; `as_s0` the
replay reference; with `cs_s0`/`as_s1` the source of the offline floors) ·
[`../teacher_slot/decision/policy.py`](../teacher_slot/decision/policy.py) +
`endo_yield/`[^private] (the outer loop ported; the null-ABBA floor
method; the shadow panel) · [`../census/`](../census/README.md) (the yoke mechanic; the stream
floor) · [`../spiral/`](../spiral/README.md) (the three clocks; the battery).
**Runs**: `cd_gf`, `cd_smoke`, `cd_ef` (gates + floor calibration, 1.50 GPU-h), `cd_s0` (main,
6 arms, 2.48 GPU-h), 2026-08-27→28. **Ranks, signs, and multiples of measured floors are the
claims.** One orchestrated conversation; built end-to-end
by one delegated implementer agent.

## The question

Track A's premise: the four architectural pieces have never been in one node, and the goal's
mechanism is the composition. A1 composes the first two — `teacher_slot`'s gauge-reading
outer-loop rule sitting over the consolidated practice learner (the routing-only depth-6 spiral
stack), owning the crank's actions this round: **commit/hold per level** and **era advance**.
The certificate that has driven every spiral-family run becomes a read-only instrument; the era
ladder's *sequence* stays the world, but *when to advance* becomes the loop's. The question, from
§4.1: does an outer loop reading the learner's own one-level-up currency drive the crank at least
as well as the schedules did — and does a within-level reader refuse here, where the crossing's
value is real?

## Design in brief

`conductor.py` forks `assay.py` verbatim (five `# [conductor]` additions; G-F replays the donor
at 0.000e+00 in-process and cross-tag over all 116 cycles). The rule is a thermostat on a
paired-interval slope with a **measured dead zone** — *not* the donor's ABBA paired trial, which
does not survive contact with an absorbing action (a commit freezes the ratchet; a trial-commit
leaks irreversibly through the miner and the plant). The donor's contrast survives as the
instrument's in-tag noise meter, and the dead zones are measured, not chosen: null-ABBA on
574 pooled offline windows (`as_s0`/`cs_s0`/`as_s1`), re-derived at full config on `cd_ef`
(the offline and in-tag ledger floors agree to **0.09%** — the method checking itself).
[`FILES.md`](FILES.md) states the rule, the reads, and what the yokes can and cannot show.

Six arms, identical but for what the rule reads: `anchor` (the scheduled crank, certificate-
else-boundary — the comparator and replay carrier), `outer_yield` (at_support one level up,
free), `outer_endo` (the plant's own masked-infill NLL on the span one level above the era's
cell, label-free by construction, priced at its measured 267 g/read), `outer_ledger` (the arm's
own metering error — the within-level reader), and a **yoked-clock control per gauge arm**
(census finding 4 made these mandatory). Era caps at 1.25× the schedule; a refusing arm rides
its caps to termination — a readout, not a hang.

## Findings

1. **The yield-reading loop drives the crank at least as well as the schedule — and better
   exactly where demand outruns what was earned.** `outer_yield` held the L2 commit 31 cycles
   past the certificate (c49 vs c18) and L3 by 18 (c92 vs its own shadow cert c74), and stretched
   the earning eras (56/50/15 cycles vs 48/40/12). What the holds bought: **bigger tables at
   unchanged precision** (L2 14 entries / 0.714 recall vs the anchor's 11 / 0.571; L3 21 / 0.125
   vs 12 / 0.071) and the most next-level minability of any arm (L3/L4 at support 72/87 vs
   62/68). On the value clock it matches the anchor inside the earnable range (era-3 Δ exactly
   0.000; eras 1–2 inside the 0.087 stream floor) and pulls ahead in consumption: era-4 +0.067
   (below floor, not readable), **era-5 +0.251 — 2.9× the earning-family floor**. The read is
   free — mined from trajectories the agent produced anyway.

2. **The within-level reader re-derives `teacher_slot`'s refusal, in a sharper form: it starves
   the level above it.** `outer_ledger`'s own error slope quiets almost immediately, so it
   commits L2 at c10, leaves era 1 after 18 cycles (chosen, V/tol 0.63), then never arms again —
   no L3 commit, four capped advances, the shortest run (104 cycles). It ends worst on value in
   every era (era-3/4 Δ −0.268/−0.273, ~3× floor) and with the **fewest next-level observations**
   (L3/L4 at support 48/49; at era-1 exit, 15/1 vs the anchor's 41/6). The donor's refuser voted
   down a crossing; this one also forecloses the observation stream the next level would have
   been mined from. The within-level currency reads "done here" without ever pricing what leaving
   costs one level up.

3. **Of the candidate currencies, only the mined yield stream moves above its own measured floor
   at full config — and the endo read as-driven is a one-shot license.** In-tag re-derivation on
   the anchor: `yield` mean(D)/floor **1.88**; raw `endo` **−0.14** (it never arms anywhere — the
   donor's additive-constant cancellation was a property of its two-condition block, which an
   absorbing action does not have, so the raw one-level-up NLL inherits the plant's global
   drift); the excess form **+0.24** — it armed and quieted once in era 1 (licensing
   `outer_endo`'s L2 commit at c16, two cycles before the certificate) and then sat flat inside
   its dead zone for the rest of every arm's run. `outer_endo` therefore never took L3, rode all
   five caps, and paid 0.110% of budget for reads that licensed one action. An instrument result,
   not a currency verdict: the shadow panel shows the same flatness in every arm, so nothing
   endogenous distinguishes the excess read's silence from the currency having nothing to say.

4. **Both yokes are bit-identical to their gauge arms** (0.000e+00 over 12 series, 139/146
   cycles, first divergence None; t_cum equal up to the read's own cost) — census finding 4
   generalized from one commit to the full action set. The reads are inert by construction and
   verified live; the criterion's entire contribution is the cycle numbers it emits. What the
   gauge buys that a clock cannot: the schedule *did not exist* until the gauge wrote it.
   `yoked_yield` reproduces `outer_yield` only because `outer_yield` ran first.

5. **Where a vocabulary stops, trust concentrates on what it holds.** The two L3-less arms end
   with the highest L2 proposal mass and argmax share, growing monotonically through the
   consumption eras (`outer_endo` 0.220 → **0.520** mass, 0.576 argmax at era 5 — double the
   anchor's 0.250/0.187; `outer_ledger` 0.396/0.383), with L3 mass pinned at 0.000. Recorded as
   an observation for Track F's trust-formation instrument, not interpreted further here.

6. **The certificate clock stays unreadable across arms, as the stream floor predicts.** L3
   cycles-to-cert spread 14–19 against the measured 11–14-cycle floor (census finding 7); L2
   16–17 everywhere. The shadow certificate fired in every arm — the loop arms' holds were holds
   *past* certification, not failures to certify.

## Interpretation (discussed with Jasper 2026-08-28 — argued, not measured)

- **(a) Track A's first question has an affirmative instance.** A thermostat-grade rule reading
  the learner's own next-level minability — free, label-free, endogenous — paces the crank as
  well as the hand-tuned schedule inside the earnable range and better beyond it, by holding
  commits until the mining stream quiets and by stretching earning time. The census's
  "gate-later is dead" scoped to its op (a 17-cycle L2-only hold under the certificate's
  license); owning both actions at both levels is what made holding worth something.
- **(b) The typed-signals law survives composition intact.** The within-level reader is not
  noisy or indecisive — it acts *decisively wrong-by-type*, twice: early commit (its slope
  quiets first) and early exit (nothing within-level prices bottom-time — `recital/`'s law,
  re-derived by a live loop). And its distinctive damage is upward: the starved next-level
  stream is `merge/`'s hollow signature produced by a *reader*, not an index.
- **(c) The criterion-vs-clock lesson matures.** Bit-identical yokes say the gauge's value is
  entirely in *when* it acts — post-hoc purchasable by a clock, endogenously writable only by a
  gauge. That is `teacher_slot`'s "the scarce thing is which gauge is consulted", now measured
  with the full action set in the loop's hands.
- **(d) One-level-up needs a differential form on a drifting plant.** The raw endo read's
  failure is mechanical and general: any absolute one-level-up loss inherits within-level
  drift unless something cancels it. The excess form cancels it but appears to saturate after
  the first level's consolidation at this depth. Whether a better differential (e.g. against a
  matched shuffled-span baseline, or `endo_yield`'s xspan turned inward) recovers a live signal
  at L3+ is an open instrument question for A2, where the learned rule can weigh multiple reads.

## Caveats

- **One rule draw, one ladder.** Era-5 Δ +0.251 clears the earning-family floor
  2.9× but sits below the given-family era-5 displacement (0.344); a stream-displaced twin of
  `outer_yield` (machinery in the fork) is the named cheap check if this cell becomes
  load-bearing.
- **Arms are not lifetime-matched.** The loop owning era advance means `outer_yield` reached
  era 5 with 139 cycles (t_cum 34.5M) against the anchor's 116 (28.8M). That is the composition
  working as designed — pacing is what the loop owns, paid in real priced time — but the era-5
  gain conflates commit-holding with longer earning. The dissociating control (certificate
  commits under `outer_yield`'s era lengths) was not run.
- **`outer_endo`'s ledger column is reconstructed** (146 × 267 g), flagged in the reduction: the
  run priced key `endo` while the driven key was `endo_excess`. Behavior-neutral by construction
  (a charge reaches only `t_cum`, which nothing in the cycle loop reads) and by the yoke's
  bit-identity; fixed for future tags.
- Global cert/commit cycle numbers are not comparable across arms (era boundaries differ);
  `c2cert` is the comparable column and sits inside the measured floor.
- The rule is not an ABBA paired trial; its nearest ancestor is the census G-A rule. Stated
  plainly, with the port's reasoning, in [`FILES.md`](FILES.md).

## Runs on disk

| tag | what |
|---|---|
| `cd_gf` | G-F: in-process fork-vs-donor replay with the shadow panel ON (0.000e+00) |
| `cd_smoke` | six arms end-to-end at `--quick`; mechanics only (nothing minable at that scale — its endo floor reading was plant-thrash, superseded by `cd_ef`) |
| `cd_ef` | the anchor alone at full config: the null-ABBA floor calibration + a second cross-tag replay gate (0.000e+00, commits identical) |
| `cd_s0` | the main run: 6 arms, measured floors, era caps 60/50/15/12/9 |

Volume `rhm-scaling-data:/data/rhm_practice_conductor/<tag>/`; fetched copies, figures
(`fig1_competence`, `fig2_panel`, `fig3_clocks`), and `reduction.txt` under `figures/<tag>/`.

## Reproduce

Full command set in [`FILES.md`](FILES.md) §Reproduce (offline floors + policy suite, preflight,
G-F, smoke, `cd_ef` floor calibration, then `cd_s0` with the measured `--tol-*` literals).

## Next steps (queued, not started)

**A2 — learn the rule**: run as this node's sibling [`../maestro/`](../maestro/README.md) — the
learned mixture matches-or-beats this node's `outer_yield` at less priced time, the
within-level-rewarded twin never takes L3, and the stream twins demote this node's era-5
learned-vs-thermostat cell while confirming both gauge arms' era-5 win over the schedule · the two cheap in-tag controls above (era-length dissociation;
stream twin of `outer_yield`) if era-5 hardens into a headline · the endo differential-form
instrument question (finding 3 / interpretation d) · finding 5 feeds Track F's F1 instrument ·
`/update-beliefs` sweep alongside the census/spiral queue.

[^private]: Not mirrored: this link points to a document in the private lab repo (the roadmap, the queue, an unrun spec, reading notes, or a conversation). See the top-level README for what is held back and why.
