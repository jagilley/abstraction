# SIZING — which rendering rule can sit below the tables, measured offline

Offline sizing lane for [`../SPEC.md`](../SPEC.md) **decision 2** ("what the rule is made of …
Q0 settles it; do not guess") and for the Q0 gate ("size offline first … no GPU"). Everything
here is CPU numpy against the DGP's own arithmetic and the donors' own functions
([`FILES` note below](#reproduce)); **no GPU, no Modal, no substrate, no fitting**. Output:
[`../phase0.json`](../phase0.json).

Facts only. Interpretation is the orchestrator's.

The world under test throughout is the arc's: **v=8, s=2, m=2, depth 6, `rule_seed=0`,
`generate_rules_distinct`**. A rule is written as a table `K` of shape `(v, R)`:
`K[f, c]` is the synonym index feature *f* takes in context *c*.

---

## Gate verdicts

| gate | what | verdict |
|---|---|---|
| **I-1** | `tall/FILES.md` §1's `d*` ladder (oracle repair distance, nested ladder, broken-only) reproduced by this offline harness at all five of its settings | **PASS** — largest deviation **0.11** tokens across **21 rungs**; depth 6/m=4 reproduces to 0.06, depth 6/m=2 to 0.10 |
| **I-2** | the fork invariant: `true_tables` reads `rules[0 … depth−2]` only, so **no bottom rule can move the flat keys at L2–L5** | **PASS** — flat-key *sets* identical at every level with `rules[depth−1]` randomised, and identical again under the structured (stem, ending) bottom of §7(d) |
| **I-3** | the bottom table at `rule_seed=0` is as `tutti/sizing/SIZING.md` fact 5 describes | **PASS** — 14 distinct codes of `v·m=16` produced, of `v^s=64` possible; exactly 2 collisions |

**The two collisions, in full** (`build_inverse_maps` is last-writer-wins):

| code | produced by | last writer keeps |
|---|---|---|
| 12 | (f=1, synonym 1) and (f=4, synonym 1) | f = 4 |
| 39 | (f=2, synonym 0) and (f=7, synonym 0) | f = 7 |

Both collisions are between **the same synonym index** of the two features. That is what makes
them addressable by a context rule at all: the two features collide only in the contexts where
the rule calls the *same* index from both.

---

## Premise corrections

**(1) The `d*` collapse tracks tuple-space occupancy `m/v^(s−1)`, not m.** `tall/FILES.md` §1
localises the collapse to *m and not depth* with a 2×2 over depth at fixed v=8 — which is sound,
and reproduced here (gate I-1). The further reading carried into the SPEC brief ("it is m's
effect, not tuple-space occupancy per se") does not survive a grid that moves v as well: at
v=8 the two are confounded, because m ∈ {2,3,4} *is* occupancy ∈ {0.25, 0.375, 0.5}. Broken-only
`d*` over the arc's own ladder, n=800/rung (SE ≈ 0.038):

| (v, m) | occupancy | L1 | L2 | L3 | L4 | L5 | gradient |
|---|---|---|---|---|---|---|---|
| (16, 2) | 0.125 | 1.80 | 2.90 | 4.74 | 7.48 | 11.51 | **6.41×** |
| (32, 4) | 0.125 | 1.70 | 2.58 | 3.72 | 5.37 | 7.96 | **4.70×** |
| **(8, 2)** — the arc's | **0.250** | **1.71** | **2.35** | **3.27** | **4.43** | **5.44** | **3.18×** |
| (16, 4) | 0.250 | 1.49 | 1.90 | 2.39 | 2.93 | 3.72 | **2.49×** |
| (4, 2) | 0.500 | 1.33 | 1.61 | 2.43 | 3.34 | 2.10 | 1.57× |
| (8, 4) | 0.500 | 1.09 | 1.13 | 1.15 | 1.16 | 1.21 | 1.11× |
| (16, 8) | 0.500 | 1.03 | 1.04 | 1.02 | 1.03 | 1.03 | 0.99× |

The gradient is monotone in occupancy and not in m: m=4 gives 1.11× at v=8 and **2.49× at
v=16**; m=2 gives 1.57× at v=4, 3.18× at v=8 and 6.41× at v=16. **m=4 is inadmissible at v=8;
m=4 at v=16 is not, on this instrument.** The mechanism is visible in the same run: at occupancy
0.5, `corrupt_hier` frequently finds the damaged node's possible-set already covering every
feature, so it cannot pick an underivable one (`no-choice` at L5: 0.000 at occupancy ≤ 0.25,
0.034–0.384 at 0.5) and the fraction of damaged instances needing any repair falls to 0.30–0.47.
What §8 shows is that (16, 4) is nonetheless unaffordable for a different reason.

**(2) At m=2 a fully shared morphology is impossible.** Define `R_eff` = the number of distinct
*columns* of K; two contexts with the same column are the same context for every observer, so
`R_eff`, not R, is the number of contexts a curriculum can hold out. Any rule of the shared form
`k = a_f ⊕ g(context)` — one global context effect plus a per-feature offset — has
**`R_eff ≤ m = 2`**, because `g` is one bit. Measured: the shared-w register rule has R_eff = 2
(of R = 4), and the additive agreement rule `a_f ⊕ b_{f'}` has R_eff = 2 (of R = 8). Four
contexts therefore *force* per-feature context sensitivity. The exhaustive bound over all row
sets in {0,1}^4 (R = 4 contexts, m = 2):

| paradigms g (distinct spelling rows) | max R_eff, any rows | max R_eff, **nested** rows (additive-representable) |
|---|---|---|
| 1 | 2 | 2 |
| **2** | **4** | 3 |
| **3** | 4 | **4** |
| 4 | 4 | 4 |

**Four contexts need only two paradigms if the paradigms may CROSS (parity), and at least three
if the rule is to be representable by a renderer additive in a scalar register.** This is the
whole of decision 2's difficulty at m = 2, in one table.

**(3) The context is never needed to parse a sequence at `rule_seed=0`.** The bottom collisions
raise the mean level-1 possible-set size (1.26 under the uniform coin, 1.26–1.32 under the rules)
and the rule's context resolves 9–26% of blocks to a singleton. But the effect **dies at level
2**: the per-level mean possible-set size is identical from L2 up across every spelling, and the
root possible-sets are element-for-element equal.

| spelling | L1 | L2 | L3 | L4 | L5 | L6 (root) |
|---|---|---|---|---|---|---|
| uniform coin (the current substrate) | 1.2614 | 1.4075 | 1.2774 | 1.5189 | 1.6935 | 1.5675 |
| `A_d2` (GF(2)-affine register) | 1.2645 | 1.4071 | 1.2774 | 1.5189 | 1.6935 | 1.5675 |
| `E_R4` (ordered register) | 1.3178 | 1.4167 | 1.2748 | 1.5189 | 1.6935 | 1.5675 |

Neither colliding pair survives its level-2 composition. The context can **sharpen a block read**;
it is never required to reach the root. (36.8% of clean sequences have an ambiguous root set under
every spelling — that ambiguity is the grammar's, not the bottom's.)

**(4) A lexically invisible held-out context requires `R_eff > m`.** Holding context *h* out
removes a leaf tuple from the practiced corpus unless every feature's spelling at *h* also occurs
at some practiced context. Where the map context → synonym index is injective per feature — which
is exactly what "the register selects the synonym index" means, and forces `R_eff = m` — every
held-out context deletes one spelling per feature. Measured new leaf codes at the held-out context
(C = R−1 practiced): 0.00 for every register rule with R_eff > m; 0.75–1.50 for the ordered
register when an **endpoint** of the scale is held out (0.00 at every interior register); and
`v` = 8 new *tokens* by construction for the structured (stem, ending) bottom.

---

## §1 — the families, counted

`free bits` is the rule's own parameter count; `table bits` is `v·R`, the plain (feature, context)
lookup. `additive-real?` is whether the rule is expressible as `1{α_f + γ_c > 0}` for reals — i.e.
whether a real-valued **additive** renderer over (one-hot feature, context) can represent it at
all (equivalently: no 2×2 XOR submatrix).

| family | rule | R | R_eff | free bits | table bits | share | paradigms | additive-real? |
|---|---|---|---|---|---|---|---|---|
| `A_d2` | `a_f ⊕ ⟨w_f, r⟩`, r ∈ GF(2)² | 4 | **4** | 24.0 | 32 | 0.75 | 5 | **no** |
| `A_d2_sharedw` | as above, one w for all f | 4 | **2** | 10.0 | 32 | 0.31 | 2 | no |
| `A_d3` | `a_f ⊕ ⟨w_f, r⟩`, r ∈ GF(2)³ | 8 | 8 | 32.0 | 64 | 0.50 | 6 | no |
| `A_2class` | `⟨w_{c(f)}, r⟩`, two shared w's | 4 | **4** | **12.0** | 32 | **0.38** | **2** | no |
| `E_R4` | `1{ρ ≥ θ_f}`, ρ ∈ {0..3} ordered | 4 | **4** | 18.6 | 32 | 0.58 | 3 | **yes** |
| `E_R4_g2` | as above, θ from 2 shared paradigms | 4 | 3 | 12.6 | 32 | 0.40 | 2 | yes |
| `E_R8` | `1{ρ ≥ θ_f}`, ρ ∈ {0..7} | 8 | 5 | 25.4 | 64 | 0.40 | 4 | yes |
| `B_agree` | `a_f ⊕ b_{f'}`, f' a neighbour feature | 8 | **2** | 15.0 | 64 | 0.23 | 2 | no |

`A_2class` is the maximally shared four-context rule the bound in premise (2) allows: two shared
response vectors, one bit of class per feature, 12 bits of the table's 32.

## §2 — the identifiability ladder

Exact, by enumerating every completion of the held-out context's column that a hypothesis class
admits — no fitting anywhere. `det` = fraction of features whose held-out spelling is **pinned**;
`acc` = expected accuracy of a renderer drawing uniformly from the completions its class still
allows. `--` = the class cannot fit the practiced contexts at all. Averaged over held-out context
and over practiced subsets of each size C.

The renderer classes, most to least generic:

| class | form | context representation |
|---|---|---|
| `table` | arbitrary | one-hot |
| `additive_1hot` | `1{α_f + γ_c > 0}` | one-hot |
| `additive_scalar` | `1{α_f + βρ > 0}` (monotone rows) | **ordered scalar** |
| `affine_gf2` | `a_f ⊕ ⟨w_f, r⟩` | register bits, parity |
| `affine_sharedw` | as above, one w | register bits, parity |
| `paradigm_g` | rows from g shared monotone paradigms | ordered scalar |

**`A_d2` — GF(2)-affine register, 4 contexts** (`det` / `acc`)

| class | C=1 | C=2 | C=3 |
|---|---|---|---|
| `table` | 0.00 / 0.50 | 0.00 / 0.50 | 0.00 / 0.50 |
| `affine_gf2` | 0.00 / 0.50 | 0.00 / 0.50 | **1.00 / 1.00** |
| `affine_sharedw` | 0.00 / 0.50 | -- | -- |
| `additive_1hot` | 0.00 / 0.47 | 0.00 / 0.40\* | **--** |

A knife edge at C = 3, and the closed form behind it: an affine function on AG(2,2) is determined
on the affine span of the observed points; two points span only themselves, three span everything,
so `K[f, r₄] = K[f, r₁] ⊕ K[f, r₂] ⊕ K[f, r₃]`. `A_2class` gives the identical ladder at 12 bits
instead of 24. The `--` in the last row is load-bearing: **a real-valued additive renderer cannot
even fit two practiced contexts of a parity rule**, let alone extrapolate.

**`A_d3` — GF(2)-affine register, 8 contexts**

| class | C=1 | C=2 | C=3 | C=4 | C=5 | C=6 | C=7 |
|---|---|---|---|---|---|---|---|
| `table` | 0.00/0.50 | 0.00/0.50 | 0.00/0.50 | 0.00/0.50 | 0.00/0.50 | 0.00/0.50 | 0.00/0.50 |
| `affine_gf2` | 0.00/0.50 | 0.00/0.50 | 0.25/0.62 | 0.80/0.90 | **1.00/1.00** | 1.00/1.00 | 1.00/1.00 |
| `additive_1hot` | 0.00/0.51 | 0.00/0.51\* | 0.00/0.48\* | -- | -- | -- | -- |

**`E_R4` / `E_R8` — ordered register, per-feature threshold**

| family | class | C=1 | C=2 | C=3 | C=4 | C=5 | C=6 | C=7 |
|---|---|---|---|---|---|---|---|---|
| `E_R4` | `table` | 0.00/0.50 | 0.00/0.50 | 0.00/0.50 | | | | |
| `E_R4` | `additive_scalar` | 0.22/0.68 | 0.39/0.72 | **0.50/0.75** | | | | |
| `E_R4` | `additive_1hot` | 0.00/0.52 | 0.00/0.54 | 0.00/0.56 | | | | |
| `E_R8` | `additive_scalar` | 0.29/0.74 | 0.45/0.79 | 0.54/0.81 | 0.62/0.83 | 0.67/0.85 | 0.72/0.87 | **0.75/0.88** |
| `E_R8` | `additive_1hot` | 0.00/0.56 | 0.00/0.59 | 0.00/0.60 | 0.00/0.63 | 0.00/0.62 | 0.00/0.62 | 0.00/0.62 |

Graded at every rung, and above the table's flat 0.50 from C = 1 on. It never reaches 1.00: a
monotone rule cannot be fully pinned from R−1 contexts, because a feature whose switch point lies
*at* the held-out register is invisible. Measured `det` at C = R−1: **0.50** at R=4 and **0.75** at
R=8, against `1 − 2/(R+1)` = 0.60 / 0.78 for thresholds uniform on {0…R} (this lane's draw puts
every threshold strictly inside the range, so the ends behave differently). Note also that `additive_1hot` — the same
renderer given the register as a **one-hot** instead of a number — pins nothing: it cannot place a
never-seen context on the scale. The scalar/one-hot distinction, not the additivity, is what buys
extrapolation.

**Coverage split** (`E_R8`, `additive_scalar` acc): is the held-out register inside the practiced
range or outside it?

| C | interpolated | extrapolated |
|---|---|---|
| 2 | 0.72 (37) | 0.83 (59) |
| 3 | 0.77 (54) | 0.86 (42) |
| 4 | 0.82 (59) | 0.85 (37) |
| 5 | 0.83 (66) | 0.89 (30) |
| 6 | 0.85 (40) | 0.91 (16) |
| 7 | 0.86 (6) | 0.91 (2) |

For a monotone rule an unpracticed register **inside** the practiced range is harder than one
outside it: outside, every feature that has already switched is pinned by monotonicity; inside,
the switch may be on either side of the gap.

**`B_agree` — agreement.** Held out as a whole context (a column), the family is at chance
(0.50 / det 0.00) at every C, because `b_{f'}` is unconstrained; and `additive_1hot` is infeasible
from C = 6 on. §9 gives the cell-level picture.

## §3 — lexical invisibility of the held-out context

New leaf codes first seen at the held-out context, averaged over held-out context and practiced
subsets:

| family | C=1 | C=2 | C=3 | C=4 | C=5 | C=6 | C=7 |
|---|---|---|---|---|---|---|---|
| `A_d2` | 3.83 | 1.67 | **0.00** | | | | |
| `A_2class` | 4.50 | 2.00 | **0.00** | | | | |
| `A_d3` | 3.29 | 1.50 | 0.57 | 0.14 | **0.00** | 0.00 | 0.00 |
| `E_R4` | 3.92 | 1.83 | 1.00 | | | | |
| `E_R8` | 2.96 | 1.38 | 0.78 | 0.49 | 0.36 | 0.29 | 0.25 |
| `B_agree` | 3.21 | 1.61 | 0.75 | 0.32 | 0.11 | 0.00 | 0.00 |

By which register is held out, at C = R−1:

| family | ρ=0 | ρ=1 | ρ=2 | ρ=3 | ρ=4 | ρ=5 | ρ=6 | ρ=7 |
|---|---|---|---|---|---|---|---|---|
| `E_R4` | 2 | **0** | **0** | 2 | | | | |
| `E_R8` | 2 | **0** | 0 | 0 | 0 | 0 | 0 | **0** |

The register families reach zero exactly where the identifiability ladder reaches its top rung
(`A_d2` at C=3, `A_d3` at C=5). The ordered register is invisible at every **interior** held-out
register and costs 2 new codes at the ends of the scale.

## §4 — parse ambiguity under the rule

Over 4,000 rule-spelled sequences (128,000 blocks):

| arm | ambiguous blocks | context reduces the set | context resolves to a singleton | root-set mean | root ambiguous |
|---|---|---|---|---|---|
| uniform coin | 0.2641 | — | — | 1.5760 | 0.3680 |
| `A_d2` | 0.2593 | 0.1328 | 0.1328 | 1.5760 | 0.3680 |
| `A_d3` | 0.2573 | 0.1306 | 0.1306 | 1.5760 | 0.3680 |
| `A_2class` | 0.2611 | 0.1287 | 0.1287 | 1.5760 | 0.3680 |
| `A_d2_sharedw` | 0.2634 | 0.2634 | 0.2634 | 1.5760 | 0.3680 |
| `E_R4` | 0.3159 | 0.1298 | 0.1298 | 1.5760 | 0.3680 |
| `E_R8` | 0.3194 | 0.1991 | 0.1991 | 1.5760 | 0.3680 |
| `B_agree` | 0.2368 | 0.2368 | 0.2368 | 1.5760 | 0.3680 |

A rule neither creates nor removes bottom ambiguity — the ambiguous-block rate stays within
±0.06 of the uniform coin's 0.264 — but it makes the ambiguity **resolvable**: knowing the
context collapses 13–26% of all blocks (half to all of the ambiguous ones) to a single feature,
which the current substrate cannot do at all. Premise (3) records that nothing above level 1 moves.

## §5 — is the context readable from the observation?

A reader that knows the rule intersects the contexts consistent with each block's code, reading
left to right over the 32 blocks of a sequence:

| family | R | pinned within 32 blocks | mean blocks to pin | one block enough | candidates left after 32 |
|---|---|---|---|---|---|
| `A_d2` | 4 | **1.000** | 3.34 | 0.000 | 1.00 |
| `A_2class` | 4 | **1.000** | 3.61 | 0.000 | 1.00 |
| `E_R4` | 4 | **0.999** | 3.87 | 0.127 | 1.00 |
| `A_d3` | 8 | 0.899 | 9.21 | 0.000 | 1.11 |
| `E_R8` | 8 | 0.358 | 6.74 | 0.032 | 2.07 |
| `E_R4_g2` | 4 | 0.500 | 3.51 | 0.164 | 1.50 |
| `A_d2_sharedw` | 4 | **0.000** | — | 0.000 | 2.00 |

At four effective contexts the register is recoverable from about three blocks of a clean
sequence, so the SPEC's requirement ("the register must be readable from the observation … the
executor's information is the performer's and not the experimenter's") is satisfiable without a
prefix token. Where `R_eff < R` (`A_d2_sharedw`, `E_R4_g2`) the surplus contexts are by definition
unrecoverable.

## §6 — the strict grader

A canon-writing executor (`canon = rules[depth−1][:, 0, :]`, synonym 0 always) solves a
rule-spelled instance and writes `B = s^(level−1)` blocks; a strict grader accepts only if every
written block carries the synonym the rule called for. `q` is the per-block agreement rate.

| family | level | B | q | **P(strict accept)** | q^B (if independent) | 2^−B | rejected |
|---|---|---|---|---|---|---|---|
| `A_d2` | 1 | 1 | 0.586 | 0.586 | 0.586 | 0.500 | 0.414 |
| | 2 | 2 | 0.572 | 0.383 | 0.327 | 0.250 | 0.617 |
| | 3 | 4 | 0.548 | 0.216 | 0.090 | 0.063 | 0.785 |
| | 4 | 8 | 0.550 | 0.135 | 0.008 | 0.004 | 0.866 |
| | 5 | 16 | 0.551 | 0.063 | 7.2e-5 | 1.5e-5 | **0.937** |
| `E_R4` | 1 | 1 | 0.468 | 0.468 | 0.468 | 0.500 | 0.532 |
| | 2 | 2 | 0.453 | 0.350 | 0.206 | 0.250 | 0.651 |
| | 3 | 4 | 0.460 | 0.297 | 0.045 | 0.063 | 0.703 |
| | 4 | 8 | 0.460 | 0.251 | 0.002 | 0.004 | 0.749 |
| | 5 | 16 | 0.463 | **0.248** | 4.5e-6 | 1.5e-5 | **0.752** |
| `A_2class` | 5 | 16 | 0.495 | **0.248** | 1.3e-5 | 1.5e-5 | 0.752 |
| `A_d3` | 5 | 16 | 0.546 | 0.122 | 6.3e-5 | 1.5e-5 | 0.878 |
| `E_R8` | 5 | 16 | 0.405 | 0.123 | 5.3e-7 | 1.5e-5 | 0.877 |

Two facts. **The register is shared by every block of an instance, so spelling errors are
correlated and P(strict accept) is nowhere near `q^B`** — 0.063 against 7.2e-5 at B=16 for
`A_d2`, a factor of 873 (and 5.5e4 for `E_R4`). And for the ordered register the plateau is exactly the mass of the one
canon-compatible register: `P(K[f, ρ]=0)` by ρ is `1.00 0.62 0.25 0.00` at R=4, so
P(strict accept) → 1/4 = 0.248 as B grows, and `canon` is the rule's own bottom rung rather than
a separate arm.

Either way a strict grader rejects **75–94% of meaning-successes at the deepest damage rung**,
against the current substrate's 0%.

## §7 — the d\* ladder

**(a) Gate I-1**, broken-only `d*`, this harness against `tall/FILES.md` §1's published means:

| setting | measured | `tall` | max dev |
|---|---|---|---|
| depth 4, m=2 | 1.58 2.14 2.69 | 1.58 2.25 2.67 | 0.11 |
| depth 4, m=4 | 1.14 1.19 1.21 | 1.13 1.15 1.17 | 0.04 |
| depth 6, m=2 | 1.71 2.35 3.27 4.43 5.44 | 1.71 2.37 3.23 4.52 5.54 | 0.10 |
| depth 6, m=3 | 1.32 1.40 1.37 1.52 1.78 | 1.33 1.42 1.33 1.52 1.73 | 0.05 |
| depth 6, m=4 | 1.09 1.13 1.15 1.16 1.21 | 1.09 1.12 1.12 1.17 1.15 | 0.06 |

**(b) the widened worlds** — the table in premise (1).

**(c) the register rules move the sampling, not the tables.** The DP is over the same rule tables,
so a rule cannot change the world; measured, the instance distribution barely moves either:

| arm | L1 | L2 | L3 | L4 | L5 | gradient |
|---|---|---|---|---|---|---|
| uniform coin | 1.71 | 2.35 | 3.27 | 4.43 | 5.44 | 3.18× |
| `A_d2`-spelled | 1.68 | 2.39 | 3.17 | 4.39 | 5.48 | 3.26× |
| `E_R4`-spelled | 1.72 | 2.50 | 3.36 | 4.66 | 5.76 | 3.35× |

**(d) the structured (stem, ending) bottom.** `rules[depth−1]` rewritten to `(stem_f, ending_c)`
with 8 stems and 4 shared endings; upper rules untouched.

| | L1 | L2 | L3 | L4 | L5 | gradient |
|---|---|---|---|---|---|---|
| stem + ending (bottom m′ = 4) | 1.00 | 1.58 | 2.23 | 3.23 | 4.01 | 4.01× |

The ladder survives (the upper layers still have m=2), the flat keys at L2–L5 are unchanged, and
the bottom becomes **collision-free**: 32 distinct codes of 32 produced, so parse ambiguity is
exactly zero and the arc's native aleatoric channel is deleted (the same trade `tutti/sizing`
records for a collision-free rule draw). Its L1 rung falls to exactly 1.00 because a damaged block
costs one stem edit and the ending is always legal. The context is the ending **token**, so it is
named by any single block — and a held-out context is a held-out token (premise 4).

## §8 — what widening costs above level 1

`true_tables` rows (`entries(l) = v·e(l)`, `e(l) = m·e(l−1)^s`) and distinct flat keys:

| (v, m) | L2 rows | L3 rows | L4 rows | L5 rows | distinct \|T_l\| |
|---|---|---|---|---|---|
| **(8, 2)** — the arc's | 16 | 64 | 1,024 | 262,144 | L2 14 · L3 56 · L4 816 |
| (16, 2) | 32 | 128 | 2,048 | 524,288 | L2 30 · L3 128 · L4 1,984 |
| **(16, 4)** | 64 | **1,024** | **262,144** | 1.7e10 | L2 57 · L3 859 · L4 235,020 |
| (16, 8) | 128 | 8,192 | 3.4e7 | 5.6e14 | — |
| (32, 4) | 128 | 2,048 | 524,288 | 3.4e10 | — |

At (16, 4) the L3 rung costs what L4 costs now and L4 costs 256× what it costs now: **the earnable
range shifts down one level**, and the L4 rung `crescendo` opened is out of reach — which is the
same doubly-exponential law `tall/FILES.md` §1 read as "at m=4 nothing above level 3 is
representable", now with v moved.

## §9 — agreement: which cells the grammar realises

Of the `v·v = 64` (feature, context) cells:

| context | realised cells | what a cell is |
|---|---|---|
| the sibling's level-1 feature | 23 | one half of a level-2 rule |
| the parent's level-2 feature | 26 | one level-2 rule |
| the left-neighbour block's feature | 57 | free-ish (observed over 20,000 sequences) |

Sibling and parent contexts are **not free variables**: the pair (f, f') is fixed by the level-2
rule that produced both, so holding out a (feature, context) pair holds out a *production*, which
moves the grammar above level 1 — the one thing the SPEC's "one change" forbids.

Cell-level identifiability of the additive GF(2) rule (a cell is pinned iff its feature and its
context are connected in the bipartite graph of practiced cells), fraction of held-out cells
pinned at practiced fraction p:

| context | p=0.2 | p=0.4 | p=0.6 | p=0.8 |
|---|---|---|---|---|
| sibling | 0.01 | 0.13 | 0.44 | 0.77 |
| parent | 0.02 | 0.21 | 0.62 | 0.89 |
| neighbour | 0.35 | 0.91 | 0.98 | 0.99 |

Additivity buys generalisation to unseen **cells** and never to an unseen **context**: a held-out
column leaves `b_{f'}` unconstrained, so the family-aware renderer is at chance (§2).

---

## What is not decidable offline (for Q1 on GPU)

1. **Does the frozen generator's block head parse rule-spelled blocks, and the held-out context?**
   §3 and §4 settle the vocabulary half — under the register families no leaf tuple is new at a
   held-out context and the codes are the same 14 the reader was pretrained on. What is not
   settled is the *distribution*: the reader was pretrained by masked infilling on uniform-coin
   data, and a rule makes per-feature spelling frequencies context-dependent. **The measurement
   that decides it**: `macros.parse_features` block accuracy against `macros.exact_features`, per
   context, on rule-spelled clean sequences, for a reader pretrained (a) on uniform-coin data and
   (b) on rule-spelled data with one context held out.
2. **Whether a renderer fitted from the learner's own solved productions reaches §2's ceilings.**
   Those ceilings assume noise-free observation of every practiced (feature, context) cell; a
   learner sees only the cells its own commits visited, at its own frequencies, with grading noise.
3. **Whether the executor can infer the register inside a damaged instance.** §5 reads clean blocks
   with the rule known; in an instance the damaged span is being rewritten and the rule is what is
   being learned.
4. **Whether the reader and the renderer share parameters** (SPEC decision 4) — a fit, not an
   arithmetic.

## Caveats

- Every rule family is instantiated at one parameter draw (`PARAM_SEED = 11`). The
  *structural* quantities — R_eff, the paradigm/context bound, the identifiability ladders'
  closed forms, lexical invisibility, the strict-grader plateau — are draw-independent; the
  *realised* counts (paradigms per family, per-block `q`, the readability means) are not.
- The identifiability ladders are **ceilings for a renderer that already assumes the class**, not
  predictions for a fitted one. They are computed by counting admissible completions; nothing is
  trained anywhere in this lane.
- `d*` is measured with `crystallize/units.corrupt_hier` and the exact DP with **no acceptance
  filter**, and reported broken-only (d\* > 0). `tall` ran on the substrate with its own
  acceptance step (0.90–1.00); gate I-1 shows the two agree to 0.11 tokens.
- The `A_2class` and `E_*` families are this lane's own additions to the SPEC's candidate list;
  the SPEC's `k = (offset_f + r) mod m` is the R_eff = m case of premise (2) and (4).
- §9's left-neighbour cell count is what 20,000 sampled sequences realised, not an exhaustive
  enumeration.
- All strict-grader numbers use the *clean* derivation's level-1 features as the executor's
  commitments (any valid derivation's features would do; the executor's own DP may pick another).

## Reproduce

```bash
cd experiments/          # CPU only; no MODAL_PROFILE needed, ~30 s

PYTHONPATH=. python3 rhm/practice/inflection/phase0_inflection.py
```

Writes [`../phase0.json`](../phase0.json). Donors, imported and never edited:
`rhm/rhm_data.py` (`generate_rules_distinct`, `build_inverse_maps`) ·
`rhm/rhm_sculpt_precheck.py` (`possible_sets`, `nearest_derivation_cost`) ·
[`../../crystallize/units.py`](../../crystallize/units.py) (`corrupt_hier`) ·
[`../../ratchet/macros.py`](../../ratchet/macros.py) (`true_tables`, `base_table`). The
rule-spelled sampler and the rule-spelled corrupter are local to `phase0_inflection.py`, so the
donors keep their uniform-coin behaviour byte-for-byte.
