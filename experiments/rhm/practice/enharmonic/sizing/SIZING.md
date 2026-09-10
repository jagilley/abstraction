# SIZING — the quotient (`enharmonic` Q0)

Offline sizing lane for the quotient node. Four questions — the cover, the category
coordinates, the alias audit, the forced-transfer probe budget — all answered against the DGP's
own arithmetic and 59 banked arms; **no GPU, no Modal, no substrate**. Machinery:
[`phase0_cat.py`](phase0_cat.py). Output: `phase0_cat.json`.

Facts only. Interpretation is the orchestrator's.

World: v = 8, s = 2, m = 2, depth 6, `rule_seed = 0`, `mine_support = 3` — the same world every
banked arm runs. Corpus: `tutti/sizing`'s nine roots plus `tu_s0`, `tr_s0`, `tc_s0` (12 roots,
59 arms, 150 commit events: 59 L2, 51 L3, 40 L4).

---

## Gate verdicts

| gate | what | verdict |
|---|---|---|
| **B-1′** | the offline replay of `Miner.build` reproduces every logged commit event entry-for-entry (`n_entries`, `tab_recall`, `tab_precision`), re-run from this lane's own loader | **PASS — 150/150** across 59 arms (`tutti/sizing` had 101/101 across 40) |
| **ORDER** | the replayed row **order** reproduces the logged `log["entry"][c]["true_mask"]` elementwise — the precondition for keying the beam use-record vector by row | **PASS — 150/150** |
| **CLOSURE** | the token-class alphabet computed by closure of `possible_sets`' own composition step equals the alphabet enumerated over the true tables, at L2–L5 | **PASS** — 9 / 11 / 13 / 22 both ways |
| **SUBSET** | every true row's parent-feature set is contained in its token class | **PASS** — 0 violations at L2–L5 |
| **PROFILE** | two committed rows of the same token class have bit-identical forced-transfer success profiles over 4,096 instances | **PASS — 397/397** groups, all 59 arms, L2–L4 |

Two arms' three (arm, level) pairs are excluded from the alias audit and from nothing else:
`cr3_s0/outer_yield_m4x` swaps its committed table mid-run (`census_extend`), so its recorded
count vector changes length and cannot be keyed by a single frozen row list.

---

## 1. The cover — the true "partition" is not one

`MC.true_tables`'s `feature` column is each true row's parent feature; the same flat tuple can
carry several, because `generate_rules_distinct` lets two features share a child tuple.

| level | true rows | distinct \|T_l\| | multiply-parented | parent-count histogram |
|---|---|---|---|---|
| 2 | 16 | 14 | 2 | 12×1, 2×2 |
| 3 | 64 | 56 | 8 | 48×1, 8×2 |
| 4 | 1,024 | 816 | 128 | 688×1, 48×2, 80×3 |
| 5 | 262,144 | 205,824 | 37,120 | 168,704×1 · 23,040×2 · 11,008×3 · 2,048×4 · 1,024×5 |

By **observation mass** at each level's own mining node (2M derivations):

| level | node | mass on true keys | mass on multiply-parented keys | as a share of the true mass |
|---|---|---|---|---|
| 2 | 12 | 0.7574 | 0.2576 | 0.340 |
| 3 | 6 | 0.4236 | 0.1127 | 0.266 |
| 4 | 3 | 0.1848 | 0.0435 | 0.235 |

**23–34% of the true mass at every mining node sits on keys that belong to more than one
class.** The whole shortfall of `|T_l|` below `v·m·(entries per child)^s` is the
multiply-parented set; the two L1 rule collisions propagate upward.

---

## 2. The category coordinates — two of them, and they are different objects

### (B) The token class

Definition, in the code: the set of level-`l` features that the leaf string
`W.entry_leaves(key, canon)` — exactly what `MC.apply_any` writes — is a legal derivation of,
read off `possible_sets`. It is what the exact grader can see. It is single-valued (a function
of the tokens), it contains the parent-feature set, and it is **closed under composition**: the
alphabet is a fixed point of the DP's own step, so a level-`l+1` key is a pair of level-`l`
class ids and the ratchet restates as `C[l+1] ⊆ C[l] × C[l]`.

| level | \|C_l\| | legal class **pairs** (the class-keyed level size) | flat \|T_l\| | shrink |
|---|---|---|---|---|
| 1 | **7** (over 7 writable blocks; the bottom collision fuses features 2 and 7) | — | 8 | — |
| 2 | 9 | 13 | 14 | 1× |
| 3 | 11 | 28 | 56 | 2× |
| 4 | 13 | 40 | 816 | 20× |
| 5 | **22** | **73** | 205,824 | **2,820×** |
| 6 | **42** | **306** | 13,056,344,064 | **42,667,791×** |

The classes are not v = 8 singletons. At L4 they are
`{0} {0,3,6} {0,6} {1} {1,5,7} {2} {2,7} {3} {3,6} {4} {5} {6} {7}`; at L5 there are 22,
up to `{2,3,4,5,6}`. **The alphabet grows with level** (7 → 9 → 11 → 13 → 22 → 42) because
ambiguity sets proliferate; it does not saturate at v.

### (A) The parent-feature key (`fourwall`'s re-key basis)

Distinct legal (parent, parent) pairs, measured per level: **14 / 14 / 14 / 15 / 14** at
L2…L6 — not `v·m = 16`, which is the ceiling. Read off a flat tuple this key is **set-valued**
wherever the cover is (§1), so it needs a counting rule; supplied as the DGP's own generative
label it is single-valued and unrecoverable from any evidence.

### The expansion set a class must carry

`|F[l][f]|`, measured (collision-free bound in brackets): L1 1 [1] · L2 2 [2] · L3 8 [8] ·
L4 128 [128] · **L5 31,744 [32,768]**. A class token at L5 that the executor must expand at
execution time is a choice among ~3.2e4 spellings.

---

## 3. Arrival, under each key

E[# keys at support 3] vs observations, at each level's own mining node, by the sizing lane's own
Poisson model (`pge`, 2M-sample marginals). `parent-GEN` = the generative label; `parent-ALL` =
every pair in the cross-product of the two halves' parent sets gets a count; `parent-UNI` = skip
an observation whose halves are not both unambiguous. The token key needs no rule.

| level | key | legal keys seen | E@1,600 | E@16,000 | E@160,000 | N for 22 keys |
|---|---|---|---|---|---|---|
| 4 | token | 37 | 34.35 | 37.00 | 37.00 | 317 |
| 4 | parent-GEN | 13 | 13.00 | 13.00 | 13.00 | 2,714 |
| 4 | parent-ALL | 14 | 13.98 | 14.00 | 14.00 | 8,949 |
| 4 | parent-UNI | 13 | 13.00 | 13.00 | 13.00 | 5,748 |
| 4 | flat | 768 | 22.80 | 313.4 | 702.1 | 1,567 |
| **5** | **token** | **70** | **63.28** | **70.00** | **70.00** | **175** |
| 5 | parent-GEN | 13 | 13.00 | 13.00 | 13.00 | 1,357 |
| 5 | parent-ALL | 15 | 15.00 | 15.00 | 15.00 | 3,139 |
| 5 | parent-UNI | 13 | 13.00 | 13.00 | 13.00 | 1,621 |
| **5** | **flat** | **45,591** | **0.000** | **0.052** | **36.72** | **132,130** |
| 6 | token | 306 | 111.9 | 249.5 | 304.3 | 185 |
| 6 | parent-GEN | 14 | 14.00 | 14.00 | 14.00 | 709 |
| 6 | flat (true+junk) | 2,000,000 | 0.000 | 0.170 | 160.7 | 81,653 |

**At the run's own ~1,600 L5-node observations the L5 class key already carries 63 of its 70
keys at support 3, against 0.000 true flat keys.** The parent keys saturate at ~13–15 by
N = 1,600 as well. The three counting rules for the set-valued parent key differ by ≤ 2 keys and
by 1.6–3.3× in N; the choice is not load-bearing at this budget.

## 3b. The ratchet, re-measured in class coordinates

Over each arm's **own** committed level-(l−1) book, the fraction of DGP-drawn level-l
observations at the mining node whose two halves the book can look up — by flat tuple (what
`Miner.build` does) versus by token class (the same lookup on the quotiented book):

| level | arms | lower rows (med) | lower classes (med) | P(build \| flat) | P(build \| class) | lift |
|---|---|---|---|---|---|---|
| 3 | 59 | 14 | 9 | 0.5514 | 0.7956 | 1.4× |
| 4 | 51 | 14 | 5 | 0.0264 | 0.4691 | **17.8×** |
| 5 | 40 | 3 | 3 | 0.0000 (max 0.0001) | 0.0185 (max 0.3848) | **570×** |

---

## 4. The alias audit — what the banked use record can and cannot say

The beam use record (`log["entry"][c]["hist"]["beam"][str(level)]`) is a per-cycle, level-wide
count vector over the committed table's rows, keyed `(phase, level)` only. There is **no slot
and no context resolution**, so "used interchangeably in the same slot" is not readable. What is
readable is each row's use time series. Units: 86 (arm × level) with a stable ≥10-cycle record,
on the 37 exact-DP arms only — the live-executor tags (`span_tau_fire = 0.50`) record the beam
phase on ~8–20 of ~130 cycles because `SpanExecutor.apply` realises an open slot with the head
and takes its intention from `SN.dp_features`, never through `MC.macro_features`.

Three similarity statistics on the row's series — `spear` (Spearman correlation of per-cycle
share), `cos` (cosine of the L1-normalised time profile), `lev` (closeness of mean log share) —
tested as (mean within-class − mean between-class), against 2,000 per-arm label permutations.
Labels: `fcls` = parent-feature set, `tcls` = token class.

| labels | statistic | L2 | L3 | L4 |
|---|---|---|---|---|
| fcls | spear | +0.30 | −1.80 | +1.13 (2 arms) |
| fcls | cos | **−5.25** | **−3.06** | +1.15 (2 arms) |
| fcls | lev | **−3.46** | −0.32 | +0.88 (2 arms) |
| tcls | spear | −0.16 | **−3.44** | **+2.97** |
| tcls | cos | +0.07 | −2.76 | +0.66 |
| tcls | lev | **−5.00** | −1.15 | **+3.66** |

(cells are z against the permutation null; **bold** = \|z\| > 3)

**No statistic at any level clears its control in the direction a merge criterion needs.** The
cells that clear it do so with the *opposite* sign: same-class rows are **less** alike in use
than different-class rows. The two L4 cells that are positive do not survive either stricter
control:

| statistic | control | L2 | L3 | L4 |
|---|---|---|---|---|
| spear | permute within \|class\| strata | −2.39 | −6.46 | **+1.52** |
| spear | drop each unit's top-mass row | −2.87 | −5.24 | **−0.78** (5 arms) |
| lev | permute within \|class\| strata | −1.25 | −4.04 | **+0.56** |
| lev | drop each unit's top-mass row | −0.21 | −1.31 | **+0.09** (5 arms) |

i.e. the L4 signal is "ambiguous rows are used more" and "one row carries 99% of the mass", not
"the use record clusters rows by class".

The mechanism, in one table — the use-share ratio min/max within a pair of rows, split by how
interchangeable the pair actually is:

| pair kind | n | median | q25 | q75 | frac < 0.01 |
|---|---|---|---|---|---|
| **bit-identical program** (same leaf rendering) | 115 | **0.0064** | 0.0000 | 0.0329 | **0.68** |
| same token class, different program | 998 | 0.0781 | 0.0009 | 0.3152 | 0.30 |
| different token class | 7,195 | 0.0731 | 0.0076 | 0.3551 | 0.27 |

**The pairs the use record separates most are the ones that are literally the same program.**
`macro_features`' max-sum DP is an argmax over rows; where two rows write identical tokens it
picks one and starves the other, in 68% of such pairs to below 1% of the winner's mass.

---

## 5. Forced transfer — exact, offline, `fourwall.entry_profile` verbatim

Instances: `context_instances` at the era whose damage cell is the level's own node
(`SPIRAL_ERAS` 1:25, 2:12, 3:6, 4:3, 5:1; write node = `mining_node(era, node, level)` = 12, 6,
3 for L2, L3, L4). Price: one grading per (entry, instance) pair, the same unit `fourwall` used.

### (a) The resolution ceiling is structural, not sample-limited

One representative program per token class, probed at the level's own cell:

| level | era | node | \|C_l\| | resolvable | dead classes | collapsed |
|---|---|---|---|---|---|---|
| 2 | 1, 2, 3 | 12 | 9 | **7** | — | {0}≡{0,3} · {2}≡{7} |
| 3 | 2, 3 | 6 | 11 | **9** | {1} | {1,6}≡{6} |
| 3 | 4 | 6 | 11 | 7 | {1}, {5} | {1}≡{5} · {1,6}≡{6} · {2}≡{2,5} |
| 4 | 3, 4 | 3 | 13 | **11** | {7} | {2}≡{2,7} |
| 4 | 5 | 2 | 13 | 12 | {4} | — |

Every row is identical at N = 512 and at N = 4,096. **The ceiling is the demand at the node,
not the probe count.** It is stable across the two eras that share a node and changes when the
node changes (L4 at era 5 sits at node 2, where 12 of 13 classes resolve) or when the era's
damage stops covering the level (L3 at era 4, where the cap binds — `tutti/sizing` premise
correction 4). A "dead" class is one no instance at that cell is repaired by, so the dead set
is a property of (class, node).

Within that ceiling the probe is exact: **397/397** groups of rows sharing a token class had
bit-identical success profiles over 4,096 instances, across all 59 arms and all three levels. A
forced-transfer probe measures the token class and nothing else.

### (b) The arms' books — token-space vs feature-space precision

Medians over arms; brackets are [min, max] over arms.

| level | arms | rows (med) | precision, FEATURE (`true_mask`) | precision, TOKEN | coverage | probe groups / live rows (med) |
|---|---|---|---|---|---|---|
| 2 | 59 | 14 | 0.714 [0.63, 0.80] | **0.923** [0.81, 1.00] | 1.000 | 7 / 13 |
| 3 | 51 | 14 | 0.333 [0.25, 0.56] | **1.000** [0.87, 1.00] | 0.872 | 5 / 13 |
| 4 | 40 | 3 | 0.500 [0.11, 1.00] | **1.000** [0.85, 1.00] | 0.832 | 3 / 3 |

At L2 the median book's 14 rows are 11 distinct programs in 9 token classes, of which the probe
resolves 7 — the level's full ceiling. At L3, 14 rows → 12 programs → 5 classes, all resolved.

### (c) The probe budget

Instances needed for the probe partition to reach its own N = 4,096 fixed point, and the
gradings that costs. `wgt` restricts the probed rows to those carrying ≥ 90% of the beam's
cumulative use mass.

| level | rows probed | rows (med) | N\* (med) | N\* (max) | gradings (med) | gradings (max) |
|---|---|---|---|---|---|---|
| 2 | all | 14 | 32 | 32 | 448 | 544 |
| 2 | **wgt** | 8 | **8** | 16 | **64** | 144 |
| 3 | all | 14 | 32 | 64 | 448 | 1,984 |
| 3 | **wgt** | 4 | 32 | 32 | **128** | 192 |
| 4 | all | 5 | 4 | 64 | 20 | 704 |
| 4 | **wgt** | 1 | 4 | 4 | **4** | 8 |

**The whole partition of a committed book costs 20–450 gradings unweighted and 4–130 weighted**,
against a run's own priced budget of ~1,600 mined observations and ~130 cycles. Use-share
weighting cuts the bill 3.5–7×, at the cost of leaving the unused rows unclassified.

### (d) The solve tax

| level | instances no row in the book repairs (med / max) | rows that repair nothing (med / max) |
|---|---|---|
| 2 | 0.000 / 0.018 | 0.077 / 0.188 |
| 3 | 0.128 / 0.358 | 0.000 / 0.133 |
| 4 | 0.168 / 0.168 | 0.000 / 0.154 |

### (e) The loss margin — `merge_candidates`' two-directional transfer loss

| relation between the pair's token classes | n pairs | min | median | max |
|---|---|---|---|---|
| equal | 1,400 | 0.0000 | **0.0000** | 0.0000 |
| **subset** (one row multiply-parented) | 787 | 0.0000 | **0.4328** | 0.6843 |
| overlapping | 147 | 0.6629 | 1.0000 | 1.0000 |
| disjoint | 7,604 | 0.0000 | 1.0000 | 1.0000 |

Equal classes are at loss exactly 0 and subset pairs at a median 0.43, so **any `tol` in
(0, 0.43) separates the true merges from the subset merges**; a `tol` above that merges a
narrow class into an ambiguous one. The `min = 0.0000` in the *subset* and *disjoint* rows is
the structural collapse of (a) — the {0}/{0,3}, {2}/{7} and {1,6}/{6} pairs, which no tol
separates because no instance at the node does.

---

## 6. Junk under transfer

| level | junk rows (`true_mask` false) | of which token-legal | of which the leaf string is a true key's | repairs > 0 | max Jaccard with a true row (med) |
|---|---|---|---|---|---|
| 2 | 238 | **0.685** | 0.685 | 0.685 | 1.000 |
| 3 | 469 | **0.955** | 0.955 | 0.955 | 1.000 |
| 4 | 93 | **0.946** | 0.946 | 0.946 | 0.567 |

**Token-junk rows repair nothing — 0 of 101 across all levels and arms — and sit alone exactly
as the parent spec's thread 6 expects.** But they are 5–32% of what `true_mask` calls junk. The
rest (163/238 at L2, 448/469 at L3, 88/93 at L4) are legal programs mislabelled by the
feature-space oracle: `build_inverse_maps` is last-writer-wins, so a span whose block was
feature 2 is read back as feature 7, giving a flat tuple outside `T_l` whose canonical rendering
is byte-identical to a true key's. Of the token-legal junk rows, 100% (L2) / 62% (L3) / 18%
(L4) have Jaccard exactly 1.0 with a true row in the same book.

---

## Caveats

- Every number is on **one rule draw** (`rule_seed = 0`). The cover, the token-class alphabet,
  the resolution ceiling and every junk number are draw-dependent. `tutti/sizing` §"The rule
  draw" shows seeds 6 and 10 are collision-free; at those draws the cover is a partition, the
  L1 alphabet is 8, and §1/§6 here become empty.
- The **token class** is defined against `canon` = `rules[depth−1][:, 0, :]`, the rendering
  `MC.apply_any` actually writes. A learner that could write a *non*-canonical expansion would
  see a different, finer relation.
- The resolution ceiling and the loss margin are measured at the era ladder's own damage cells
  and at `mine_cap`-shaped instance pools. §5(a) shows the ceiling is stable across N and across
  the two adjacent cells, and shows one cell (era 5, node 2) where it differs — it is a property
  of (class, node).
- The arrival table's flat rows are re-derived on 2M-sample marginals: internally comparable,
  ~20% optimistic in absolute N against `tutti/sizing`'s 8M estimate. |T6| is not enumerable, so
  the L6 flat row counts every observed key, true and junk alike.
- §4 is an audit of the **banked** record, which is `(phase, level)`-keyed only. It is silent on
  what a *slot-resolved* or *context-resolved* use record would say; producing one means
  instrumenting the fired path, which no banked tag has.
- §5(c)'s N\* is the point at which the probe partition stops changing, measured on nested
  prefixes of one 4,096-instance pool per cell; it is not a confidence statement.
- The alias audit's 86 units come from 37 exact-DP arms; the 22 live-executor arms are excluded
  for the reason in §4 and appear in every other section.

---

## Addendum (Q1 build) — what the fork's own mining path arrives at

Produced by [`fork_arrival.py`](fork_arrival.py) → `fork_arrival.json`; offline, no GPU. Facts
only.

§3 above modelled arrival on the class-pair key by computing each half's token class from the
**raw tokens** of a DGP derivation. `enharmonic.py`'s `ClassMiner` cannot do that: it sees the
reader's parsed level-1 feature string, and the class it must use is the class of
`canon[features]`. That is not a choice — `macros.apply_any` writes `canon[feats]`, so a
committed row's class *is* `possible_sets(canon[row])`, and using the raw-token class on the
observe side against the canonical class on the build side would leave observed classes with no
producible representative (a rule-1 bottom tuple's class is not any canonical block's class, so
the L2 ratchet would drop nearly everything).

### The round-trip loss

`exact_features` is last-writer-wins, and `canon` renders rule 0. **~50% of blocks come back as
a different token string** — not from the 2↔7 canon collision (that one is exactly preserved)
but from every other shared bottom code. 20,000 derivations:

| level | raw token span legal | `canon[parse]` legal |
|---|---|---|
| 2 | 1.0000 | 0.8448 |
| 3 | 1.0000 | 0.6632 |
| 4 | 1.0000 | 0.4906 |
| **5** | **1.0000** | **0.2354** |

**So §3's 63.3 of 70 at L5 and 111.9 of 306 at L6 are an upper bound on the fork's arrival, not
its prediction.**

### The fork's own path, simulated

`tu_s0`'s ladder verbatim, `mine_cap = 8` DGP draws per cycle parsed by an exact reader
(`read_acc` = 1.0, which the substrate has), committable miners era-gated to `era_level + 1` as
`run_arm` gates them, `QT.make_miner` / `ClassMiner.build` / `Quotient` verbatim, boundary
commit of the active level. The remaining gap to a run is the agent's own solve rate and
narrowness, which gate G-1 measured at 0.42–1.68×.

| key | L2 | L3 | L4 | **L5** | L5 obs@sup3 | **L6 obs@sup3** |
|---|---|---|---|---|---|---|
| flat | 22 (p .59) | 61 (p .43) | 5 (p .20) | **none** | 0 | **0** |
| `tok` | 25 (p .52) | 114 (p .37) | 304 (p .41) | **48 (3 class keys, p .06)** | 33 | **13** |
| `gen` (min-label) | 25 (p .52) | 124 (p .35) | 240 (p .31) | **48 (3 class keys, p .06)** | 19 | **13** |

(entries at commit, `p` = precision against the true table in **feature** space)

Cross-product inventory: max 16 entries per class key at every rung; the `spell_cap = 4` cap
binds on 0 / 0 / 20–30 / 6 keys at L2 / L3 / L4 / L5.

**`flat` never reaches L5 and its L6 at-support series is identically 0**, so the mirror's
commit thermostat cannot set its `moved` latch there — the comparator fails mechanically rather
than by a tuned floor. Both quotiented keys build an L5 table and both carry a live L6 gauge.

### Where the binding constraint moved

Under the flat key the wall at L5 is arrival (§3). Under either class key arrival is solved and
the binding constraint is **the L4 book's class coverage** — how many of the 13 L4 token classes
the committed L4 table holds a spelling of, since `ClassMiner.build` drops a class pair whose
half-class has no representative row. In the simulation the L4 book covers 10–11 of 13
(`n_lower_classes` at the L5 build), which is what admits 3 of the 73 legal L5 class pairs. This
is reducible per cycle from a run's logged tables through `quotient.py` and needs no extra
instrumentation.
