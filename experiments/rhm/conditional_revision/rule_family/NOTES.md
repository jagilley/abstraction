# Working notes — rule_family

**Not a writeup.** Results get discussed with Jasper before any README (repo convention).
This file holds: what has run, what the numbers were, pre-registered designs for gates not
yet launched, and decisions being held for Jasper.

**Files**: [FILES.md](FILES.md) · **Parent**: [../README.md](../README.md)
**Date**: 2026-08-09 · Single seed (42) throughout.

---

## Status

| gate | status |
|---|---|
| Gate −1 (design selection, CPU) | **run** — oracle verified, two designs selected |
| Gate 0 (substrate, Modal) | **run** — both designs, each vs matched floor arm |
| diagnostics (position sweep, retention, emergence, held-out) | **run** (R64 emergence/held-out pending) |
| `rule_retention` part 1 (DGP-side positive control) | **running** |
| `rule_retention` part 2 (model-side curves) | not built |
| Gate 1 (revision readouts) | **designed, pre-registered below, NOT launched** |
| Gate 2 (temporal FM) | **designed, pre-registered below, NOT launched** |
| retrain bundle (phase protocol + budget) | **held for Jasper** |

---

## Results so far

### Gate −1 — the budget constraint

Summed over a window, `E[rule revision] = I(r ; x_window) ≤ ln R`. The entire slow
component is capped, so **signal strength and transient duration trade off along a fixed
budget**. Larger R buys budget; fewer differing features spends it more slowly. This is
RHM_META_LEARNING's "thin compositional signal" restated information-theoretically, and
it is the binding design constraint.

Selected designs (v16 s2 L6 m4, K=8 sequences/window):

| design | budget used | share of rule info by sequence 0…7 |
|---|---|---|
| `d2_R128_nF4` | 4.92 nats | 0.77 / 0.20 / 0.03 / 0 … |
| `d2_R64_nF2` | 4.24 nats | 0.28 / 0.31 / 0.14 / 0.12 / 0.06 / 0.04 / 0.03 / 0.03 |

Measured, not assumed: differing the leaf-emitting table identifies ~3× faster (p50 6 vs
14) — which is why it is shared; differing at high levels is slow but invisible
(0.0007 nats/token at d5).

### Gate 0 — ICL present, thin, and only in the graded design

Matched depth-swap (same probe sequence at every context depth; content identical, only
preceding context varies):

| | `d2_R64_nF2` | `d2_R128_nF4` |
|---|---|---|
| oracle available (depth 7) | 0.0189 nats | 0.0610 nats |
| family decline | +0.0040 ± 0.0003 | +0.0022 ± 0.0003 |
| floor decline | −0.0003 ± 0.0002 | +0.0009 ± 0.0002 |
| net | **0.0043 (≈12σ)** | 0.0013 (≈3.6σ) |
| ICL fraction | **0.16 → 0.23** | 0.009 → 0.022 |
| corr(net, oracle available) over depths | **+0.992** | +0.754 |
| rule probe (chance) | **0.039** (0.0156) | 0.0080 (0.0078) |

`d2_R128_nF4` fires the pre-registered kill. `d2_R64_nF2` passes.

**The dissociation is the finding**: the graded design realizes **3× more ICL in absolute
nats while offering 3× less available signal**. R64 is a partial escape from
RHM_META_LEARNING's collapse and the first positive ICL-pressure reading on RHM; R128 is
that collapse reappearing on the **in-context** axis rather than the weight-space one.

*What explains the dissociation was read wrong on the first pass — see the correction
below.*

### Emergence and held-out — and the reading this forces me to revise

| | `d2_R64_nF2` | `d2_R128_nF4` |
|---|---|---|
| net decline at 6k | ~0.000 | +0.0012 |
| net decline at 12k | **+0.0051, still climbing** | +0.0021, **flat since 2k** |
| held-out transfer | **+0.00432 vs +0.0040 in-dist (~108%)** | +0.00172 vs +0.0022 (78%) |

The floor arm's own trace is the reason the net matters: early in training the floor shows a
*spurious* positive decline (+0.0048 at 2k) that decays to −0.0012 by 12k. Any single-arm
reading before ~6k would have been an artifact.

### CORRECTION — storage burden → inference dimensionality

Kept as a correction rather than a silent edit, per repo practice: the mistaken reading and
the measurement that killed it are both more useful than the conclusion alone.

**What I claimed first (WRONG).** That the R64/R128 dissociation was explained by *storage
burden* — R128 must hold 128 × 4 features × 4 tuples = 2048 tuple-assignments against R64's
64 × 2 × 4 = 512, so R128 spends capacity covering tables instead of inferring, "treating the
family as a harder pooled task". The capacity tax (+0.074 vs +0.026 nats over each arm's own
floor) was offered as direct evidence.

**What killed it.** Held-out transfer. The R64 family model's loss on rule sets it has
**never seen** is **1.4657, against 1.4661 in-distribution** — indistinguishable — with a
depth-swap decline of **+0.00432 vs +0.0040 in-distribution (~108%)**. The floor model reads
**1.877** on those same sequences, so they are genuinely different DGPs and the family model
is genuinely generalising, not recognising. A model that had solved the task by memorising 64
tables could not do this.

**What replaces it.** The limiting variable is the **dimensionality of the thing to be
inferred**, not the amount to be stored. `d2_R64_nF2` requires pinning down a partition of 8
tuples into 2 groups; `d2_R128_nF4` requires 16 into 4. The harder inference is done far
worse *despite three times more information being supplied*. The capacity tax decomposes to
match: of R128's +0.074 nats, ~0.060 is unresolved mixture penalty (it captures 2% of a
0.061-nat gap) and only ~0.013 is residual; R64's +0.026 splits ~0.015 / ~0.011. The tax is
mostly **failure to infer**, not **cost to store** — which is the opposite attribution to the
one I made.

**Lineage.** This is [`meta_adapt`](../../../mjc/meta_adapt/README.md) Cut #4c's boundary
condition — *"value-of-information's lever should appear only for harder identification: a
higher-dimensional task parameter"* — reached from the other side: 4c found low-dimensional
system-ID saturates so fast that a smarter acquisition drive is over-engineering; here
raising identification dimensionality is what *breaks* in-context inference outright. It is
also RHM_META_LEARNING's learnability diagnosis with the two confounds removed: that cut
could not separate "no incentive" from "not learnable", because weight-space meta-learning
supplied neither a within-episode incentive nor a within-episode information channel. Here
both are present and exactly quantified (`ln R` supplied, fraction realized measured), and
the answer is that supply is not the binding constraint.

**Caveat to carry with the 108%.** Only **6** held-out rule sets exist — `d2_R64_nF2` uses 64
of the 70 partitions its own design admits — and they share the training family's tuple pool.
So this is *within-pool* generalisation to novel partitions, not transfer to a fresh pool. The
stronger test (new pool) is untested and would need a design with a larger rule space.

---

## Finding pair: the discard schedule factors into two axes

Keep these together; this is the shape it should reach the idea doc in (pending discussion).

**Axis 1 — distance at fixed relevance** ([`../synonym_retention/`](../synonym_retention/README.md)).
On fixed rules at λ=0, closed-constituent synonym identity decays +0.858 (w=0) → +0.079
(w=8) → chance (w≈16) against a *rising* exact Bayes ceiling, and the synonym bit goes ~3×
faster than the feature identity. Relevance is pinned at zero throughout (a closed
constituent is NTP-redundant on fixed rules); only distance varies.

**Axis 2 — relevance at fixed distance** (this cut's position sweep, floor arm =
the incumbent substrate). At **matched distance-since-close of zero**:

| read position | what is still pending | d1 | d2 | d3 | d4 |
|---|---|---|---|---|---|
| 47 | d5 pending | 0.711 | 0.743 | **0.880** | 0.742 |
| 63 | nothing (sequence ends) | 0.526 | 0.321 | **0.217** | 0.189 |

d3 reads **0.880 vs 0.217 at identical distance**, split only by whether an unresolved
ancestor still depends on it. At position 31 (all of d1–d5 resolved, only the root
pending) everything is dumped (d1 0.560); at live positions 59–62 the floor arm recovers
the incumbent reference lines exactly (**d1 0.91–0.97, d3 0.72–0.89** vs reference
0.979/0.836).

So the schedule is keyed to **predictive relevance, not elapsed distance**. Together the
two axes factor it. The causal lever is phase visibility: the incumbent trained on flat
concatenation (phase hidden, cannot know a boundary is coming) is the control; this cut
trains on aligned windows (phase revealed) and discards on schedule.

**Consequence recorded**: Gate 0's belief-depth number was read at position 63, the single
worst position. The substrate is intact.

### Retention: the schedule moves in the rule channel, not the parse channel

Family vs floor at matched positions across all seven boundaries (`d2_R64_nF2`):

- **Rule probe** (direct instrument; floor pinned at chance 0.0156 as guard): family
  post-boundary excess **+0.0114 → +0.0304**, growing with context depth. Rule identity
  *is* carried across boundaries.
- **Parse probe** (conservative lower bound): family − floor **negative everywhere**
  (−0.031 to −0.044 at offset 1).

The family model carries a **compressed rule-sufficient statistic** across boundaries while
carrying **less** decodable parse detail than the floor model. A re-allocation, not an
addition — Petersen et al.'s replacement-not-insertion, now across regimes rather than
across positions. Caveat: one shared probe across positions, so the 4× pre→post drop
(0.182 → 0.046 at boundary 7) may be partly geometric; the matched family−floor excess is
the safe number. R128's excess is ≈ 0, consistent with its Gate 0 null.

---

## Pre-registered: Gate 1 — revision readouts

**Not launched.** Blocked on the retrain decision.

**What the family regime supplies that the incumbent could not.** The incumbent's Gate B
matched position × exact surprisal and compared *different* positions. Here the same
position with the same exact surprisal is rule-revision-heavy early and rule-revision-zero
late, because context depth is orthogonal to position by construction (windows are whole
aligned sequences). That gives a **within-position, within-surprisal, across-context-depth**
contrast — the cleanest form of the aleatoric-null test available, since the confound Gate B
had to stratify away is here removed by design.

**Object.** `M_rule = KL(q^rule_{t+1} ‖ q^rule_t)` from a probe-decoded rule posterior,
matched term-for-term by the oracle's exact `rule_rev`. Reported alongside the incumbent's
`M` on the ancestor chain, since the total revision splits exactly as
`B_total = rule_rev + E_r[B^(r)]`.

**Registered predictions.**
1. `M_rule` separates high- from zero-rule-revision positions at matched position and
   matched exact surprisal, above the 0.5 guards.
2. The separation *decays with context depth*, tracking the oracle's `rule_rev` profile.
3. The floor arm shows nothing at any depth (guard).

**Kills.**
- `M_rule` ≤ guards at every depth → the model's state does not express rule revision;
  Gate 2 has no belief-space object to forecast and should not be run in belief space.
- Separation present but **flat** in depth → it is reading rule identity, not rule
  *revision*; report as identity decoding, not revision.
- The floor arm separates → the contrast is positional, not epistemic; instrument is wrong.

**Read positions.** Live positions (59/61/62 style) *and* boundaries, reported separately —
the finding pair above says these are different regimes and pooling them would blur it.

**Stakes.** The language sibling found every belief readout null while a supervised decode
read 0.99. RHM is currently the only substrate where a belief-space revision readout has
ever worked, so this is the bridge test between the RHM-positive and the language-null.

## Pre-registered: Gate 2 — the temporal FM

**Not launched.** Blocked on Gate 1.

**The design constraint, inherited from the language sibling and load-bearing.** Its Gate D
found the temporal FM residual's *magnitude* is explained by output entropy at **R² = 0.901**
and by revision at **R² = 0.0001**, while the residual *direction* separates at 0.996. In
this regime the ICL decline **is** an output-entropy decline — mixture predictive entropy
falls as the rule posterior concentrates. Therefore:

> A raw "FM residual decays over context" readout is a **pure entropy artifact** here, and
> is pre-registered as the **negative control**, not the result.

**Primary readout.** Does the FM residual *direction* carry the rule posterior / rule
revision, at **matched output entropy and matched position**?

**Controls and guards.**
- `r_dir` vs `h_after_dir` vs `h_before_dir` — the sibling found `r_dir` (0.996) ≈
  `h_after_dir` (0.993), i.e. the FM **inherits** rather than produces. Gate 2 must
  establish whether the FM contributes anything over the raw state; if not, that is the
  honest finding and it is cheap.
- **Permute the inputs, not the labels** (their gotcha: label-shuffle guards read 0.52–0.83
  against a feature a swap-state control puts at 0.50).
- Floor arm as regime control: its residual direction must carry nothing rule-like.
- Protocol-matched depth FM as the axis control (the incumbent's idiom).

**Kills.**
- Directional separation ≈ guards → the FM residual carries no rule information; the
  "FM watches the model learn" claim dies on this substrate.
- Directional separation present but `r_dir` ≈ `h_after_dir` → the FM inherits; report as a
  property of the state, not of the forward model.
- Separation survives at matched entropy only in the *magnitude* readout → entropy artifact.

---

## Held for Jasper: the retrain bundle

Two-arm phase protocol (phase-hidden concatenation + aligned) and budget (12k vs ~36k
steps), conditional on R64's emergence trace. Rationale, cost and recommendation go back as
a concrete proposal — not launched.

Key input already in hand: R128's trace is **flat from 2k**, so budget is not the R128
story. The substrate does **not** require the retrain (live positions already recover
reference-line belief depth); the argument for phase-hidden is that boundaries are exactly
where rule-revision lives and exactly where the aligned model dumps everything. Keeping the
aligned arm matters because the aligned-vs-phase-hidden contrast *is* the causal
demonstration of the finding pair above.

With random phase, stratify on **true hierarchy level** (the oracle knows the offset)
rather than raw position.

---

## Gotchas worth not rediscovering

- **Rule sets differing only in the storage order of a feature's m rules are
  distributionally identical.** The DGP draws rules uniformly, so order is not a degree of
  freedom it can express. Comparing raw arrays misses this and silently produces
  unidentifiable duplicates — which *mimics the target phenomenon*, since a capped posterior
  looks like gradual identification. Canonicalise; guard on the partition count
  `(nF·m)!/(m!)^nF`, which is only **70** for nF=2, m=4.
- **A design can exhaust its own rule space.** `d2_R64_nF2` uses 64 of those 70, leaving
  only 6 held-out rule sets — so its Student-2 transfer test is inherently small-sample.
- **The incumbent oracle has no revision for a sequence's first token** (its axis is
  "x_{t+1} arrives", giving T−1 columns). Inside a window that position is real and the
  joint identity fails there without it — hence `sequence_revision_all`.
- **Train on `x[:, :-1]`, so position G−1's embedding is never trained.** Any probe reading
  the last position of a full-length window reads an untrained positional embedding.
- **Read belief probes away from constituent boundaries** — see the finding pair. At a
  sequence boundary the aligned-window model has discarded essentially everything.
- **An unmatched single-arm ICL reading before ~6k steps is an artifact.** The FLOOR arm —
  where nothing can be learned in context — shows a *spurious positive* depth decline of
  **+0.0048 at step 2k**, decaying to −0.0012 by 12k. Anyone reading the family arm alone
  early in training would report in-context learning that is not there. The matched floor is
  what makes the emergence curve interpretable, not a nicety.
- **The matched depth-swap is worth its complexity**: the floor arm's contrast is flat to
  four decimals (noise floor ~0.0002 nats) against an available signal of 0.019–0.061, i.e.
  ~100× SNR. An unmatched loss-vs-position curve cannot resolve this effect.
