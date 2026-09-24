# rubato — the value reader's form and its diet: one confidence scalar closes the linear reader on the logit trunk, shaping writes it into the practice plant, and in the loop the read grades new entries at zero world queries once its diet covers the span (admit-then-grade, two seeds; the loop made resumable)

**Up**: [`../README.md`](../README.md) (aliquot, the lineage whose plants, readouts and ungated arm this node grades and forks) ·
**Spec**: [`SPEC.md`](SPEC.md) (the orchestrator's phase-1 brief verbatim; the phase-2 and round-3 briefs are in
`CONVERSATION.md`[^private]) · **Decisions, gates and defects**: [`DESIGN.md`](DESIGN.md) (§1–§5 the saved
state and its gates, §10 the admit-then-grade op, §11 the diet arm, §4 every result as tables) · **Machinery record**:
[`FILES.md`](FILES.md) · **Conversation**: `CONVERSATION.md`[^private] (the session behind all three rounds,
2026-09-23 → 24) · **The two offline rounds this writes up beside it**:
[`../preplay/timbre/`](../preplay/timbre/README.md) (pp1, pp3 and pp5 re-read on richer reader forms; facts in
`timbre/figures/timbre_reduction.txt`) and
[`logit_reading/striatum/norm/precision/`](../../../../../logit_reading/striatum/norm/precision/README.md) §8a (the
`+logits` arm; facts in `precision/results/tables_logits_20260923.md`) ·
**The in-loop round this puts in order**: [`../sostenuto/README.md`](../sostenuto/README.md) (the admission set and
the admit-then-grade conclusion) · **Motivation**:
[`ideas/calibration_and_violation_are_one_object.md`](../../../../../../../ideas/calibration_and_violation_are_one_object.md)
§12.3 (the two blindnesses) and §12.4 (a candidate has to be fired to be priced), read against
[`precision/README.md`](../../../../../logit_reading/striatum/norm/precision/README.md) §8 (what the reader's form
does).
**Runs**: 2026-09-23 → 24. `precision` `+logits` (0.8 L4-h, 18 cells, three seeds); `timbre` `tb1` (0.65 GPU-h,
eight containers, offline on the banked soundboard plants); `rubato` `rb_r1` (gate R-1), `rb_s1` / `rb_s2` (the
ungated arm re-run with saved states, both seeds, 1.84 / 2.33 GPU-h), `rb_g1` / `rb_g2` (admit-then-grade, world and
read, both seeds, ≈5.4 / 4.4 GPU-h including a refused launch and a stopped preemption restart), `rb_g1d` (the
diet-gated read arm, seed 0, 1.61 GPU-h). Every in-loop arm clock-yoked to its seed's banked anchor; seeds 0 and 2
throughout and never averaged. **Ranks, signs, located mechanisms and per-cell counts are the claims.**
**Attribution**: the ask to metabolise PRs 129–133 together and the question how precision's finding bears on the
live practice results, the ask to run the three action items with Opus builders, the standing instruction that long
loops be resumable from a mid-run cycle, the ranking of the confidence-scalar result as the most important, and the
side-chat clarification that the seat's utility was tested with the oracle as its ceiling and that grammar-junk is not
value-junk on this substrate are Jasper's (2026-09-23 → 24). The reading that the practice read works where it is a
level and fails where it is a difference, its correction by `timbre`, the diet reading of the read arm's wrong
revocations and the one-knob diet arm, and the synthesis are the orchestrator's, agreed in discussion. The saved
state and its gates, the per-span diet, the four zero-GPU re-readings, the versioned digest and auto-resume, the
`+logits` arms and the span diagnostic, and the reader forms and their gates are the three implementers' (one Opus
session each; the rubato builder ran all three of its rounds).

## One-liner

Four PRs had left two things unmetabolised: whether precision's finding that the linear value reader misses a
nonlinear function of the world model's belief touches the practice arc's live results, and how far those results
stand from an oracle-free practice loop. This node answers both. **On the logit-reading trunk the missing thing is
one scalar**, the belief's absolute level (its log-partition or, equally, its max logit): the sixteen logit directions
handed over without it buy nothing on any column, seed, venue or level, and the layer-normed state plus that one
number reproduces the belief-appended reader column for column, including the response's sign reversal along the
entropy axis. **On the practice plant the shaped read is form-invariant** in both of preplay's seats (the ridge, the
belief appended, an MLP on the state and an MLP on the belief alone give the same pp1 prices and the same pp5 gate),
and the plant's block logits are linear in its pooled state, so the belief arm adds exactly that scalar; what moves
with the reader's form is the unshaped control, which a richer reader lifts off chance at seed 0 and not at seed 2.
Shaping is therefore one of two routes to the read, the linearly readable and seed-stable one, and not the only
one. **In the loop**, the practice loop is now resumable (the whole cross-cycle state saved at a cadence and at
every era boundary, restore bit-identical within a host type and one ulp of numpy's transcendentals across host
types, both reference arms re-run bit-identically with states banked), and pp5's seat was rebuilt as admit-then-grade
on the ungated live build: every key admitted provisionally, graded by firing the keys one level up that consume it,
revoked on positive evidence, re-offered when a new consumer appears. **The read grades at zero world queries where
the oracle version bills about 2,300 queries a cycle, and its wrong revocations sit where its weights have never
valued the span it is asked to price**: at seed 0 twenty of twenty-four were priced while the readout's buffer held
no row at the consumers' span, a per-span diet rule halves them, and what remains is single-pass negatives a hair
below zero that a two-consecutive rule or a margin near −0.05 removes on both seeds while keeping most of the junk
catches. **No graded arm beats the ungated arm at depth on either seed, the oracle's included**: the miner mines only
from the learner's solves, so frequency and value are nearly one number here, there is almost no value-junk to prune,
key arrival is the budget, and the oracle arm's one large loss came from its no-consumer window rather than its
grade. The seat has nothing to buy on this substrate; the read is ready to carry it where it does.

## The rounds

### [`logit_reading/striatum/norm/precision/`](../../../../../logit_reading/striatum/norm/precision/README.md) §8a — the `+logits` arm (the other substrate, 0.8 L4-h)

**Goal**: precision §8 left two candidates for what the ridge on the raw state lacks: the log-partition, one
nonlinear scalar no linear map of the state forms, or the ridge's shrinkage under-using the belief's sixteen
directions inside 256 normalised dimensions. Add the arms that separate them on the banked express cells.

**Finding**: the scalar, cleanly, three seeds. Written up in §1 below and as a dated addendum in precision's own
README; facts in `precision/results/tables_logits_20260923.md`.

### [`../preplay/timbre/`](../preplay/timbre/README.md) — pp1, pp3 and pp5 re-read on richer reader forms (offline, 0.65 GPU-h)

**Goal**: pp1's "shaping is the ingredient" and pp5's "the frozen and twin readouts in the gate's seat are worse
than no gate" were linear-reader statements, the shape of claim precision showed can flip with the reader. Refit the
belief-appended ridge, an MLP on the state, an MLP on the belief alone and a per-slot ridge on pp1's own bank
subsample through the shaped, frozen and twin trunks, and run pp1's, pp3's and pp5's reductions unchanged on each.

**Finding**: the shaped read is the same on every form in both seats; the unshaped control moves. §2 below; facts
in `timbre/figures/timbre_reduction.txt`, decisions in `timbre/NOTES.md`.

### `rubato/` phase 1 — the loop made resumable (this folder, `rb_r1`, `rb_s1`, `rb_s2`)

**Goal**: an arm of this loop is 201 cycles and 2.2–2.5 GPU-h with no resume path, so every retroactive gate change
had been a full re-run. Save the whole cross-cycle state, prove the restore, and re-run the ungated arm with states
banked at both seeds so the next round starts mid-run.

**Finding**: bit-identical resume within a host type; both reference re-runs bit-identical to their banked mirrors;
22 saved states a seed at about 1.2 GB and 3% of the arm's time. §3 below.

### `rubato/` phase 2 and round 3 — admit-then-grade in the loop (`rb_g1`, `rb_g2`, `rb_g1d`)

**Goal**: sostenuto concluded the next-level currency wants admit-then-grade. Build it on the ungated live build,
the world's success and the read's level as the two grades, restored from the ungated arm's saved state; then read
the record and run the one correction it names.

**Finding**: the seat runs at zero world queries; the read's errors are diet, not form; no grade helps here. §4–§6.

## The question

PR 131 (`precision`) found that everything about the value reader's level survives any reader form and everything
about its response at an event does not: a ridge on the raw state reads the response's entropy slope with one sign,
and every reader that sees more of the belief reverses it. PRs 129, 130 and 133 (`preplay`, `sostenuto`) found on the
practice plant that the same kind of reader prices a fired candidate entry in the world's order but does not rank
spellings within a context, gates a try-and-keep walk at a quarter to a half of the oracle's advantage offline, and in
the loop refuses good entries at three to eight times its offline rate from a starved seat that decides on one to five
changed instances of 192. The orchestrator's reading, offered to test rather than to assume: every practice-side
success was a level over fired states and every failure a difference of levels, and precision says difference reads
are where the reader's form bites. Two claims followed as re-readable, pp1's "frozen and twin at chance under the
same readout" and pp5's "frozen and twin worse than no gate", with one practice-side hint already on the record (the
duplex MLP critic on the frozen trunk reading the probe rows at 0.850 / 0.807 against the shaped linear read's
0.832 / 0.727, `../README.md` §3). And separately: how far do the live results stand from an oracle-free practice
loop, and what would it take to put pp5's seat in the loop in the order sostenuto said the currency wants.

Vocabulary, inherited. The **plant** is the 2-layer masked-infill encoder the loop trains; its **belief** is its
block-logit distribution over spellings, which the executor's **DP** scores through; the **projection** (the read) is
the parent's linear readout of the plant's pooled state, one per root, refit every cycle on the learner's own filed
writes (the **bank**); **shaped** means the plant trained on an outcome head beside its infill term, **frozen**
overtone's unshaped plant at the same seed, the **twin** a never-trained trunk. A **candidate entry** is a proposed
table row; its **consumers** at level l+1 are the keys at support in the learner's observation-panel miner that have it
as a half; **firing** a key is pp1's single-entry fire, the key alone as a table forced as the macro on a fresh pool of
its level's cell, world-graded and read on the fired configurations. On the logit-reading side the **critic** is the
ridge on the trunk's residual stream, `V_pre` its level, `R` its response, `H_pre` the trunk's pre-event entropy,
`log q` its sixteen log-probabilities, `lse` their log-partition.

## What was built

`precision/express.py` gained `--logit-arms` (the raw logits, `ln` plus `lse`, `ln` plus `max`, `+lse`, `+logits+lse`,
each appended column standardised through its Gram and mapped back) and a shared-column gate against the committed
cells; `express_read.py` gained the headline, span and shared sections. `timbre/readers.py` and `timbre.py` fork
pp1's, pp3's and pp5's loops line for line and replace only the read, arm 0 asserted equal to the banked JSON
in-container before anything else is written; `reduce_timbre.py` rewrites each reader's columns in pp1's and pp5's own
schema and runs the banked reducers unchanged. `rubato.py` forks `sostenuto.py` at its round-2 head with six `rb_*`
knobs default off and is `sostenuto.py` to the bit with them off (G-F); it pickles the whole cross-cycle state (75
names: the plant and its optimizer, the readout bank, the heads, the miners and panel, the tables and admission set,
the buffers, the loop policies and ledger, the logs, the clock, and every numpy and torch generator an organ owns
plus the four global streams) as one object graph with a hash tree beside it, restores it in one statement, and
re-derives and fingerprint-asserts what it does not save (the substrate, the pools); phase 2 adds the grade (DESIGN
§10), round 3 the per-span diet (DESIGN §11); `reduce_rubato.py` reads the graded arms beside sostenuto's five gates
and adds sections [L] (the grade) and [M] (the dial, persistence, the diet at the seam, re-offers, the bill, the
no-consumer cost). Every gate in every node was shown to fail before it was reported (R-0 4/4, R-1 2/2, G-N 3/3,
G-D, G-O; timbre 7/7; precision's arm-0 gate at 6e−08).

## 1. On the logit-reading trunk the missing thing is one scalar, the belief's absolute level (`precision` `+logits`)

The arithmetic §8 left: the logits are `z = W_U ln_f(s)` exactly, so a ridge on `ln_f(s)` spans every linear function
of them, and `log q = z − lse` differs from them by one number per row. Under the log-partition candidate, `+logits`
should behave like `+ln` and `ln +lse` like `+log q`; under the shrinkage candidate the reverse. `swap65k`, anchor
`t_v`, ℓ=1, 64k, three seeds side by side; every arm's shared columns reproduce the committed cells to 6e−08:

| reader | held-out fit gain | `H_pre`'s damage reading left outside (marginal 0.62) | `R ~ H_pre` at matched surprisal | `z(s)·z(H)` | `AUC(−R, flip)` |
|---|---|---|---|---|---|
| ridge (banked) | — | 0.609 / 0.611 / 0.599 | +0.031 / +0.029 / +0.023 | +0.014 / +0.012 / +0.016 | 0.611 / 0.589 / 0.603 |
| `+log q` (§8's reference) | +0.029 / +0.028 / +0.028 | 0.549 / 0.566 / 0.547 | −0.021 / −0.015 / −0.024 | −0.025 / −0.016 / −0.021 | 0.674 / 0.649 / 0.673 |
| `+ln` (§8's reference) | +0.021 / +0.019 / +0.015 | 0.618 / 0.619 / 0.623 | +0.050 / +0.039 / +0.039 | +0.016 / +0.015 / +0.019 | 0.597 / 0.596 / 0.601 |
| **`+logits`** (the directions, no log-partition) | +0.004 / +0.003 / +0.004 | 0.607 / 0.621 / 0.608 | +0.032 / +0.034 / +0.028 | +0.011 / +0.011 / +0.015 | 0.611 / 0.583 / 0.601 |
| **`ln +lse`** (the log-partition, directions left inside) | +0.029 / +0.027 / +0.023 | 0.534 / 0.566 / 0.545 | −0.023 / −0.015 / −0.024 | −0.021 / −0.012 / −0.019 | 0.682 / 0.649 / 0.664 |
| **`ln +max`** | +0.034 / +0.031 / +0.027 | 0.541 / 0.569 / 0.549 | −0.020 / −0.009 / −0.019 | −0.021 / −0.011 / −0.019 | 0.678 / 0.646 / 0.662 |
| `+lse` (the one scalar on the raw state) | +0.019 / +0.021 / +0.020 | 0.570 / 0.587 / 0.570 | −0.002 / +0.002 / −0.004 | −0.016 / −0.009 / −0.011 | 0.664 / 0.639 / 0.661 |
| `+logits +lse` (contains `+log q`'s features) | +0.030 / +0.028 / +0.028 | 0.546 / 0.562 / 0.547 | −0.025 / −0.018 / −0.023 | −0.027 / −0.019 / −0.020 | 0.677 / 0.653 / 0.673 |

Handing over the sixteen directions does nothing on any column, seed, venue (`a1` reproduces) or level (ℓ=2 the
same); handing over one scalar on the normalised basis reproduces `+log q` on all five columns, and `+logits +lse`
equals `+log q` column for column. The span diagnostic says why the contrast is real: each logit is 98% linear in the
raw state, and `lse` is 95–97% linear in `[s]`, `[s, z]` and `ln_f(s)`, so the arms differ only through a few percent
of nonlinear residual, and that residual carries the whole effect. Which scalar is less settled: `ln +max` does as
well as `ln +lse`, so the record identifies a scalar of the logits' absolute level rather than the log-partition as
such, and `+H`, also a nonlinear scalar of the belief but invariant to shifting the logits, gives fit and no
containment (§8). At 8k the same arms contain more than `+log q` and leave the slope near zero rather than reversed. On
the clean-only critic `ln +lse` is the strongest arm of any (0.509 / 0.544 / 0.531). Deep levels: the `ln`-based arms
carry `ln`'s fit loss at ℓ=3–4 (to −0.010), `+lse` and `+logits +lse` keep `+log q`'s gain there.

## 2. On the practice plant the shaped read is form-invariant, and the unshaped control is what moves (`timbre`)

Each reader refit on pp1's shared bank subsample (8192 train / 2048 validation rows, selection on validation only)
through the shaped, frozen and twin trunks, so a column differs from pp1's own by the reader's form alone; arm 0
reproduces the banked pp1, pp3 and pp5 columns bit for bit in-container (pp1 4/4 levels on six arms, pp3 3/3, pp5
14/14 walks). One structural fact first: the plant's block logits are a linear function of its pooled hidden, so the
belief appended (the block-head log-softmax at the fired slots, pooled the two ways the hiddens are) adds to the
ridge's span only the two pooled log-partitions times the root, the scalar §1 found.

**pp1's cell, true-vs-wrong AUC of the read's mean level over the fired instances**, shaped / frozen / twin, the arms
of record:

| cell | world | banked | ridge | belief appended | MLP on the state | MLP on the belief alone |
|---|---|---|---|---|---|---|
| s0 L2 | .949 | .747 | .712 / .488 / .529 | .713 / .479 / .529 | .734 / .583 / .546 | .683 / .573 / .553 |
| s0 L3 | .890 | .637 | .632 / .587 / .475 | .633 / .592 / .482 | .633 / .642 / .449 | .629 / .656 / .416 |
| s0 L4 | .941 | .680 | .675 / .535 / .535 | .676 / .598 / .532 | .673 / .634 / .487 | .676 / .488 / .590 |
| s0 L5 | .969 | .645 | .639 / .568 / .515 | .640 / .574 / .514 | .640 / .566 / .512 | .647 / .538 / .526 |
| s2 L2 | .958 | .666 | .674 / .498 / .503 | .691 / .538 / .509 | .693 / .497 / .523 | .688 / .553 / .511 |
| s2 L3 | .889 | .711 | .708 / .589 / .483 | .702 / .611 / .482 | .710 / .535 / .602 | .707 / .654 / .480 |
| s2 L4 | .953 | .630 | .632 / .607 / .482 | .639 / .671 / .486 | .643 / .600 / .494 | .639 / .616 / .534 |
| s2 L5 | .953 | .609 | .615 / .601 / .516 | .617 / .615 / .517 | .617 / .585 / .594 | .620 / .556 / .566 |

Shaped above both controls: ridge 8 of 8 cells, every richer reader 7 of 8 (the frozen trunk wins s0 L3 under the
MLPs and s2 L4 under the belief). On the shaped trunk nothing moves with the form: Spearman with the world's price
0.17–0.55 on every reader, the within-legal-class correlation positive on 24 of 24 cells on every refit reader (the
banked 23), the level's compression against the world's 0.10–0.37 on every reader, per-instance transfer 0.57–0.73.
On the frozen trunk the richer readers lift the read: Spearman 0.08–0.40 against the ridge's 0.03–0.29, the within-class
correlation positive on 23 of 24 against 21, the compression under the MLP reaching the shaped read's scale at three
cells. The twin stays at chance on AUC under every form and loses its within-class sign under the MLPs (12–14 of 24).
On pp3's rows the picture is where pp3 left it (shaped MLP Spearman .14 / .48 / .32 and .65 / .58 / .59 against the
ridge's .03 / .49 / .33 and .51 / .57 / .59; the frozen trunk was already off chance there on the ridge).

**pp5's seat, each reader's level as the gate with `dp_top` the order** (capture = (ungated − gate) / (ungated −
world), from medians over the 14 (cell, repeat) pairs a seed; W/T/L = pairs where the gate beats / ties within 0.005 /
loses to no gate on the test pools, at budget 8 and at the full walk):

| gate | pooled capture b8 / full | junk admitted / good refused | seed 0 capture | seed 2 capture | seed 0 W/T/L b8; full | seed 2 W/T/L b8; full |
|---|---|---|---|---|---|---|
| banked (pp5's own) | .27 / .57 | .022 / .050 | .58 / .93 | .14 / .03 | 7/5/2; 8/4/2 | 10/1/3; 8/3/3 |
| ridge, refit, shaped | .42 / .57 | .005 / .046 | .58 / .89 | .14 / .03 | 9/2/3; 9/2/3 | 10/1/3; 9/3/2 |
| belief appended, shaped | .42 / .57 | .005 / .046 | .58 / .88 | .14 / .03 | 9/2/3; 9/2/3 | 10/1/3; 9/3/2 |
| MLP, shaped | .27 / .58 | .016 / .046 | .58 / .90 | .14 / .03 | 8/3/3; 9/2/3 | 10/1/3; 9/3/2 |
| belief-only MLP, shaped | .42 / .58 | .010 / .046 | .65 / .91 | .43 / .33 | 9/2/3; 8/3/3 | 10/1/3; 10/2/2 |
| ridge, frozen (pp5's) | −1.90 / −.84 | .021 / .089 | −1.22 / −.28 | −5.36 / −.14 | 5/5/4; 3/5/6 | 4/1/9; 3/2/9 |
| belief appended, frozen | −.43 / −.29 | .011 / .049 | +.32 / +.27 | −7.50 / −.98 | 9/4/1; 8/4/2 | 3/3/8; 4/3/7 |
| MLP, frozen | −.85 / −.22 | .021 / .037 | +.49 / +.71 | −5.71 / −.42 | 8/3/3; 6/5/3 | 6/3/5; 5/3/6 |
| ridge, twin (pp5's) | −1.61 / −.51 | .021 / .064 | −.96 / −.09 | −4.82 / −.44 | 4/5/5; 4/5/5 | 5/1/8; 6/1/7 |
| MLP, twin | −1.18 / −.57 | .021 / .046 | −.32 / +.18 | −4.21 / −1.13 | 3/4/7; 4/4/6 | 5/3/6; 5/2/7 |

The shaped read's gate is the same table under every form. The frozen plant under the belief or the MLP beats no gate
on eight or nine of fourteen pairs at seed 0, about the shaped read's rate, and refuses good entries at 0.037–0.049,
the shaped read's rate rather than the ridge's 0.089; at seed 2 every frozen reader loses to no gate as often as it
wins. The paired form's admissions equal the pooled form's on 140 of 140 walks for every reader. Two facts about the
record: pp5's headline 0.27 / 0.57 pools 0.58 / 0.93 at seed 0 with 0.14 / 0.03 at seed 2, where the ratio sits on a
0.018 median gap between no gate and the world's gate and the per-pair counts are the honest read (the banked gate
beats no gate on 10 and 8 of 14 there); and pp1's "shaped above frozen above twin on 8 of 8" is strict on 6 (frozen is
below twin at L2 on both seeds) and "shaped above both" on 8. Both are now dated notes in
[`../preplay/README.md`](../preplay/README.md). One more: the refit ridge at the loop's budget admits junk on 0.005 of
candidates against the banked readout's 0.022, the one-update-out-of-step readout showing up in the gate's favour once
refit.

## 3. The loop is resumable (`rb_r1`, `rb_s1`, `rb_s2`)

- **R-1**: an arm saved at c and restored in a fresh container and continued to 2c is the uninterrupted one in every
  table, weight, optimizer moment, buffer, stream and logged quantity, at an era boundary (c12 → c24) and mid-era
  (c14 → c24); the two `time.time()` fields the loop stores are excluded from the digest and are the only differences
  found. Falsified: the torch CPU stream or `grng` left unrestored diverges at the next cycle in the plant weights and
  the admission records.
- **The cross-container floor is a host type.** Twelve fresh L4 containers landed on nine AVX-512 hosts, where numpy
  1.26 dispatches `log` / `exp` to SVML one ulp off libm, and three non-AVX-512 hosts. Bit-exact within a type; across
  types one ulp on numpy's transcendentals, which in R-1 reached one logged diagnostic (a degenerate fit's `w_norm`)
  and no decision. `NPY_DISABLE_CPU_FEATURES` pins it and was deliberately not set, since it would change numerics
  against every banked arm; every run with a save knob on records its host type. The convention is now in
  `experiments/CLAUDE.md`.
- **The reference re-runs.** `st_gn_yk` at seed 0 (1.84 GPU-h) and seed 2 (2.33 GPU-h) with saved states on are
  bit-identical to their banked sostenuto mirrors on all 13 fidelity series and every other series, event, table and
  key over all 201 cycles; the only extra keys are round-2 bookkeeping and, on seed 2, the grade knobs, whose arm-level
  grade is absent, so the phase-2 code is inert at full scale with the grade off. 22 states a seed at every tenth cycle
  and every era boundary, 19–89 MB each (the int64 token buffers compress about 9×), 1.24 GB in all, a save a median
  10–13 s, 3% of the arm.
- **It paid the same day.** A Modal preemption restarted the seed-0 graded arms from scratch at about c140; the restart
  was stopped and both arms resumed from their own c130 saves (R-2 exact, the re-spawned pair's digests equal to the
  cancelled pair's at c30 and c40, an incidental cross-container check), at about 0.4 GPU-h instead of 4. Auto-resume
  from an arm's latest own save is now the default when a save knob is on.
- **Two defects on the way** (DESIGN §5.4, §5.5): the digest rule changed after the reference states were saved and
  R-2 refused the first phase-2 launch (0.38 GPU-h, nothing ran); the digest is now versioned and an unrecorded
  version resolved by trial. And the preflight's uncapped entry log made a save cost more each cycle; R-1 runs at the
  paid arms' cap.

## 4. Admit-then-grade in the loop, both seeds (`rb_g1`, `rb_g2`)

The op (DESIGN §10). Admission is the ungated walk untouched, so the arm is `st_gn_yk` until its first revocation and
restores from its saved state at c20, the latest save before a grade can fire. A level-l key served for a window of 160
observations of the panel one level up is graded by its consumers there: none, and it is revoked for want of a
consumer (the `demand` gate's criterion moved after admission); some, and its four best-supported consumers are fired
one at a time, single-entry, on a fresh pool of the level-(l+1) cell, the same pool graded unfired, and the key's grade
is its best consumer's gain, kept iff above a margin of zero. The read arm's value is the shaped projection's level on
the fired configurations, zero world queries; the world arm's is the world's success on the same fires, billed. Both
values are recorded on every fire whatever the arm consumes. A revoked key leaves service and is re-offered when a
consumer it did not have at revocation reaches support. The window was read off the reference run's own saved states
before paying: every consumed L2 key had its consumer within 160 observations, and at 40 three would have been
revoked before their consumers formed. Identity: R-2 at every restore; G-I by hash equal to the reference at every
save before the first revocation; the first differing series is at the first revocation; the world grade's first
grade-changing fire equals the donor's audition instance by instance.

| | seed 0 world | seed 0 read | seed 2 world | seed 2 read |
|---|---|---|---|---|
| first revocation | c30 | c55 | c45 | c65 |
| revocations (no consumer, worthless) | 14 (8, 6) | 25 (1, 24) | 15 (7, 8) | 21 (6, 15) |
| worthless the other currency would have kept | 6 of 6 kept by the read | 21 of 24 kept by the world | 6 of 8 | 10 of 15 |
| re-offers | 6 | 14 | 7 | 10 |
| revoked at the end L2 / L3 / L4 | 3 / 4 / 1 | 11 / 0 / 0 | 3 / 3 / 1 | 6 / 1 / 1 |
| of `st_gn_yk`'s walked L3 keys, a half held revoked at the end | 4 / 19 | 18 / 19 | 7 / 24 | 10 / 24 |
| of its L4 keys | 5 / 9 | 0 / 9 | 1 / 22 | 1 / 22 |
| L3 rows admitted / live at the last pass (`st_gn_yk` 71 / 71, 85 / 85) | 62 / 65 | 9 / 9 | 65 / 65 | 75 / 77 |
| fires; world queries billed to the grade | 2,153; 413,376 | 679; 0 | 2,058; 395,136 | 1,669; 0 |
| task error − `st_gn_yk`, eras 1..5 | −.004 −.002 +.035 +.176 +.053 | −.001 −.009 +.077 +.190 +.239 | −.001 −.033 −.002 +.017 +.008 | +.000 +.026 +.010 +.029 +.092 |

The read arm's worthless revocations are L2 keys graded on L3 consumer fires, and on every one at seed 0 the read's
level on the fired states was below the unfired pool's (−0.0007 to −0.10) while the world's success on the same fires
rose on 21 (+0.04 to +0.40) and held on 3; at seed 2 the world kept 10 of 15, and the 5 it agreed with are junk L2 keys
with a world gain of exactly zero. Eleven revoked L2 keys removed a half from 18 of the 19 L3 keys the ungated arm
built at seed 0, and L3 served 9 rows against 71. The world arm's six worthless revocations at seed 0 are all junk
L2 keys whose consumers' fires changed nothing. The oracle-graded op billed 2,200–2,300 world queries a cycle, about
thirty times sostenuto's walk gate over a whole run.

## 5. The read's wrong revocations are its diet, and a per-span diet rule halves them (`rb_g1` [M], `rb_g1d`)

**M-3, the diet at the seam.** The readout's row buffer held no span-4 (L3) row at any save through c100 at seed 0;
the first appear at c110. Before any span-4 row the read arm priced 82 L2 key-passes on L3 consumer fires and revoked
20 (0.244), the world keeping 18 of them; after, 62 and 4 (0.065), the world keeping 3. At seed 2 span-4 rows appear at
c80 and 5 of the 15 worthless revocations were priced with no row at the consumers' span, the world keeping 4. The
read's first fit is at about c50 on the learner's own L2-era writes, and c55 is its first revocation.

**Round 3, `rb_grd_yk`** (DESIGN §11): one knob, `adm_grade_diet = 512`, the readout's own `vo_om_min`, so a number
the lineage already uses for "enough to read at all" rather than one tuned on this record. A level-l key's consumers
at span l+1 are priced only once the readout's fitted buffer holds 512 rows at that span; below it the key is kept
silently, want of a consumer still revokes, the fires still run so both values stay on the record. Restored from
`rb_s1`'s c50, equal to it by hash at c60–c100, first revocation c105 (L2 first priced at c105 with 593 span-4 rows,
L3 first priced at c190, L4 never; 235 key-passes silent for the diet).

| seed 0 | read, margin 0 | read, diet 512 | world |
|---|---|---|---|
| first revocation | c55 | c105 | c30 |
| revocations (no consumer, worthless) | 25 (1, 24) | 13 (1, 12) | 14 (8, 6) |
| worthless the world would have kept | 21 of 24 | 10 of 12 | — |
| decision vs the world on the same fires, KK / KR / RK / RR | L2 116 / 4 / 21 / 3; L3 54 / 0 / 0 / 0 | L2 272 / 11 / 6 / 2; L3 88 / 0 / 4 / 0 | — |
| `st_gn_yk`'s L3 keys with a half held revoked at the end | 18 / 19 | 11 / 19 | 4 / 19 |
| L3 keys with neither half ever revoked (M-1 at margin 0) | 1 / 19 | 5 / 19 | 15 / 19 |
| L2 revocation rate with span-4 rows present | 0.065 | 0.051 (8 of 158) | — |
| task error − `st_gn_yk`, eras 1..5 | −.001 −.009 +.077 +.190 +.239 | +.000 +.002 +.029 +.081 +.227 | −.004 −.002 +.035 +.176 +.053 |
| L3 rows admitted / live, L5, at the last pass | 9 / 9, 1 / 1 | 6 / 24, 32 / 32 | 62 / 65, 80 / 80 |

Where the twelve fall: five at c105, the first priced pass (three of them one merged class at a read gain of −0.0013
each against the world's +0.28, one at −0.030 against +0.17, one junk at −0.101 against 0.000); one at c110 (−0.0073
against +0.50) and one junk at c115; one at c180 after fifteen priced passes all kept; four L3 keys at c190 and c197,
the first priced L3 passes (world +0.07 to +0.46). Every one is a run of length one.

## 6. The dial, persistence, re-offer, and what the window costs

- **M-1, the dial.** Keep iff the best consumer's gain exceeds the margin, re-applied to every recorded graded
  key-pass, the world's margin-0 decision as the split. Read arm, seed 0: margin −0.10 revokes 2 keys (1 the world
  kept, 0 it also revoked) and leaves 19 / 19 L3 keys whole; −0.05 revokes 6 keys, 11 / 19 whole; 0 revokes 15, 1 / 19.
  Seed 2: −0.05 leaves 1 wrong and 1 right revocation and 21 / 24 whole; −0.02 leaves 7 wrong and 10 / 24. Every
  read-arm worthless gain at seed 0 lies in [−0.10, −0.0007]. On the world arm every margin at or below −0.005 revokes
  only its no-consumer keys.
- **M-2, persistence.** Under "revoke only after k consecutive non-positive priced passes", the read arm's worthless
  revocations still firing at their pass: seed 0 k=2 7 of 24, k=3 1, k=4 0 (the world kept 14 of the 17 that k=2
  removes and 21 of 24 at k=4); seed 2 k=2 1 of 15, k=3 0; the diet arm k=2 0 of 12. The world arm's own: k=2 2 of 6
  and 4 of 8, k=4 0, so persistence alone is blunt.
- **M-4, re-offer.** No key wrongly revoked at seed 0 ever came back: at every later save its consumer set was exactly
  its set at revocation, since the level above stopped growing around the missing half. On the diet arm three of the
  six never-re-offered L2 keys did gain a new consumer, first seen at the c200–c201 saves after the last grade pass.
  Six of the read arm's 14 re-offers at seed 0 and 6 of 10 at seed 2 released a revocation the world would have kept.
- **M-5, the bill.** World arm: 35 grade passes, 61.5 fires a pass, 3.15 per graded key-pass, 192 queries a fire,
  2,335 a cycle over c25–c201. Read arm: 3.8 fires a cycle, no bill.
- **M-6, the no-consumer revocations against the ungated arm.** Of the seed-0 world arm's eight, one matters for
  coverage: the L3 key `[1,1,1,1]`, revoked at c110 at age 185 observations, is a half of 5 of `st_gn_yk`'s 9 walked L4
  keys, already had a consumer in `st_gn_yk` at c80, and was never re-offered; another got its first consumer in
  `st_gn_yk` at age 479, past the window. Four are keys `st_gn_yk` never admitted. At seed 2 none of the world arm's
  seven is a half of any key `st_gn_yk` walked, and its deep-era error is +0.017 / +0.008.

## The update

1. **The linear value reader on a prediction-trained trunk lacks one number, and it is the belief's confidence.** Not
   its entropy, not its directions, not normalisation: a scalar of the logits' absolute level, handed over on the
   normalised basis, reproduces everything the whole belief bought, including the sign of the response's entropy
   slope and interaction, on three seeds. This is the sharpest statement in the record about the reader's form, and
   it is what "precision" reduces to on this substrate: the reader needs to know how peaked the belief was.
2. **Shaping writes that number into the practice plant's state.** On the shaped plant every reader form gives the
   same read in both of preplay's seats, and the belief-appended arm adds exactly the log-partition. On the unshaped
   plant a reader that sees the belief or reads nonlinearly lifts the read off chance and, at seed 0, carries pp5's
   gate at the shaped read's rate; at seed 2 it does not. So "shaping is the ingredient" becomes "shaping is the
   reliable route to a linearly readable read"; the reader route exists and is cheaper, and is seed-fragile here. The
   never-trained twin has nothing under any form. The two substrates say one thing: where the plant is trained on
   prediction only, the nonlinear function of the belief must live in the reader; where it is shaped on outcomes, a
   weighted sum suffices.
3. **The orchestrator's difference-form reading was wrong for the shaped plant and right about the controls.** The
   shaped read's gate did not flip with the reader; the frozen plant's did. What was fragile about the practice-side
   differences was the diet and the seat, not the form.
4. **In the loop the read's failure is weight blindness, measured.** The read priced fired states at a span its
   weights had never valued and read them low where the world read them high, on twenty of twenty-four wrong calls at
   seed 0. §12.3's second blindness, pp1's off-diet degradation, and the readout's first fit on L2-era writes are one
   fact. A per-span diet rule fixes most of it; what remains sits at each span's first priced pass, where the diet has
   just crossed its threshold, and is single-pass noise a hair below zero that a two-consecutive rule or a margin near
   −0.05 removes on both seeds while keeping most junk catches. Once past the diet's edge the read makes nearly the
   oracle's decisions at zero cost, where the oracle-graded op bills about 2,300 world queries a cycle.
5. **The seat's utility was tested, with the oracle as its ceiling, and on this substrate it is nil.** The read can
   only match the oracle's decisions in a seat, never beat them, so the oracle arms are the test of whether the seat
   buys anything, and on both seeds, in sostenuto's gate and here, the oracle-graded loop was neutral to worse than
   the ungated one at depth. The oracle's one large loss (seed 0, era 4) came from a no-consumer revocation on a
   timer, not from its grade, so the pure grade was closer to neutral than to harmful. The reason is structural,
   and it is the reading the side chat sharpened: the miner mints vocabulary from pairings that recur in the learner's
   own solves above a support threshold, so frequency-in-solves is already a proxy for payoff; grammar-junk is mostly
   not value-junk here (pp3: the world prices the learner's false entries nearly as high as its true ones at L3 and L4,
   because a frequent false pairing is often a good way to write a correct string); the value-junk the world's grade
   found was a handful of zero-gain L2 keys; and key arrival is the binding constraint. Starved, not polluted. In the
   language of goal-directed behaviour, the miner is habit by repetition and the value seat is the organ that
   withholds reinforcement from repeated actions that do not pay, and on this loop the two cannot come apart much
   because the only things repeated enough to be mined appeared in successes. That is the nesting problem of the
   yield currency (`../README.md` §5) one organ over.
6. **Admit-then-grade is the right order, and its residual frontier problem is the window.** Want-of-a-consumer within
   a fixed window is the `demand` gate's demand on a timer, and it produced the oracle arm's only coverage loss; the
   grade itself did not. The op's order removed the frontier failure (one no-consumer revocation on the read arm at
   seed 0 against 17 of 19 L3 keys blocked before), and re-offer as built never fired for a wrong revocation because
   the level above stopped growing around the missing half.
7. **The loop is resumable, and the practice is now a convention.** Full state at every tenth cycle and every era
   boundary, restore proven bit-identical within a host type, the residual across host types located to one ulp of
   numpy's transcendentals and left unpinned on purpose, both reference arms banked with states; it paid for itself
   the day it landed.

## What this does not show

Nothing here tests the value seat in a regime where it has work: no arm changed the miner, the support threshold or
the world, so "the seat has nothing to buy" is a statement about this substrate's mining rule and not about the seat.
The diet-gated arm ran at seed 0 only; its seed-2 twin is one saved state away. The margin and persistence effects are
post-hoc re-applications on recorded passes and not arms; a revoked key was not graded again until re-offered, so a
counterfactual margin cannot see what a kept key would have done later. The deep-era task-error ordering carries no
direction across seeds for the graded arms, as it did not in sostenuto. Which scalar the logit trunk's reader lacks,
the log-partition or the max logit, is not separated (both work; a four-arm follow-up of about 25 L4-minutes is
designed and not run). The unshaped plant's richer-reader gate worked at seed 0 and not at seed 2, so the reader
route is on the record as seed-fragile and no more. The re-offer rule was checked only at grade passes, so a consumer
that appears after the last pass is never seen. Nothing off the RHM practice substrate and the logit-reading grammar.

## Reproduction

```bash
cd experiments/            # MODAL_PROFILE=chromatic
# the +logits arms on the banked express cells (18 cells, ~50 L4-min), then the reduction
modal run --detach -m rhm.logit_reading.striatum.norm.precision.express::express_sweep --arm-set H,Q --map-arms L,Lp --logit-arms Z,LE,LX,E,ZE --no-do-mlp --out-sfx _express_logits --check-sfx _express,_express_ln
python -m rhm.logit_reading.striatum.norm.precision.express_read --sfx _express_logits --preamble logits <mirror>/traj_a1_s42 <mirror>/traj_a1_s43 <mirror>/traj_a1_s44
# the reader-form re-read (eight L4 containers, ~0.65 GPU-h), then the reduction
B=rhm/practice/voicing/sotto_voce/aliquot
sh $B/preplay/timbre/results/RUN_timbre.sh && python3 $B/preplay/timbre/reduce_timbre.py --tag tb1
# the loop: gates, the resume-identity gate, the reference re-runs with states, the graded arms, the diet arm
modal run $B/rubato/rubato.py::rb_gates                                   # R-0, G-D, G-O, G-N and their falsifications
sh $B/rubato/results/RUN_rb_r1.sh                                          # R-1 (+ falsifications)
sh $B/rubato/results/RUN_rb_s1.sh; sh $B/rubato/results/RUN_rb_s2.sh       # st_gn_yk with saved states, both seeds
sh $B/rubato/results/RUN_rb_g1.sh; sh $B/rubato/results/RUN_rb_g2.sh       # world / read grades from c20 (RUN_rb_g1_resume.sh: the c130 resume)
sh $B/rubato/results/RUN_rb_g1d.sh                                         # the diet-gated read arm from c50
python3 $B/rubato/fetch_compact.py --tag rb_g1 && python3 $B/rubato/reduce_rubato.py --grade --tag rb_g1   # likewise rb_g2, rb_g1d
python3 $B/rubato/reduce_rubato.py --bank-check --tag rb_s1                # the bit-identity of a re-run against its banked mirror
python3 $B/rubato/reduce_rubato.py --seedtable                             # figures/rb_grade_seedtable.txt
```

Every flag, app id, wall clock and the volume path of every saved state (`rhm-scaling-data:/rhm_practice_rubato/<tag>__<arm>/<arm>/ck/c{NNN}.pt`)
is in [`FILES.md`](FILES.md); the reader forms and their gates in [`../preplay/timbre/FILES.md`](../preplay/timbre/FILES.md);
the express arms in [`precision/FILES.md`](../../../../../logit_reading/striatum/norm/precision/FILES.md).

## Next steps (queued in `QUEUE.md`[^private], not started)

A regime where junk enters the table, so the value seat has work: mine from all trajectories rather than solved ones,
or lower the support threshold, or a world where common pairings mislead under damage; a design conversation before a
build, since it changes the substrate · the diet-gated read arm at seed 2, one saved state away · the readout's diet
from preplayed states, since the fired consumer configurations are exactly the span the buffer lacks until c80–c110 ·
the window: a consumer's absence read against the level above's growth rather than a timer · the four-arm "which
scalar" follow-up on the logit trunk · the idea-doc clauses (the third blindness now with a practice-side instance;
"shaping is required" qualified; the confidence scalar) and the beliefs pass, held for discussion.

> **2026-09-24, later.** The first item ran as the sibling [`../scordatura/`](../scordatura/README.md): the failures
> appended to the miner at the batch's own proportion; the seat has work there, the oracle in it buys depth for the
> first time, and the read carries it at one seed, bound by its diet.

## Files

| file | purpose |
|---|---|
| `README.md` | this writeup, the super-node for the session's three rounds |
| `rubato.py` | the substrate: `../sostenuto/sostenuto.py` forked; the saved state, its gates, the admit-then-grade op, the per-span diet, the arms |
| `reduce_rubato.py` | the reducer: bank checks, R-1's record, the grade ([L]) and its re-reading ([M]), the two-seed seed table |
| `launch_detached.py`, `fetch_compact.py`, `results/wait_app.sh` | the launcher, the compact mirror (`setup.json.gz`), the restart-proof waiter |
| `results/RUN_*.sh` | the commands of record |
| `SPEC.md`, `DESIGN.md`, `FILES.md`, `CONVERSATION.md` | the brief; the decisions, gates and defects; the machinery record; the session record |
| `figures/` | `rb_grade_seedtable.txt` (both seeds), the per-tag reductions and [M] sources, the bank checks, the compact mirrors under `rb_s{1,2}/`, `rb_g{1,2}/`, `rb_g1d/` |

[^private]: Not mirrored: this link points to a document in the private lab repo (the roadmap, the queue, an unrun spec, reading notes, or a conversation). See the top-level README for what is held back and why.
