# SIZING — the second extension (L4 → L5) and the deep-era question grip

Offline sizing lane for the unification node. Two questions, both answered against the DGP's own
arithmetic and 40 banked arms; **no GPU, no Modal, no substrate**. Machinery:
[FILES.md](FILES.md). Outputs: `phase0.json`, `phase0_grip.json`.

Facts only. Interpretation is the orchestrator's.

---

## Gate verdicts

| gate | what | verdict |
|---|---|---|
| **B-1′** | the offline replay of `Miner.build` reproduces every logged commit event in the corpus entry-for-entry (`n_entries`, `tab_recall`, `tab_precision`) | **PASS — 101/101** across 40 arms: 40 L2, 35 L3, **26 L4**. A3's own B-1 was 18/18 on donors that held **no** L4 commit; the L4 rung is replayed here for the first time. |
| **L6 count** | inclusion–exclusion over the level's distinct child pairs reproduces L2–L5 exactly before being used at L6 | **PASS** — 14 / 56 / 816 / 205,824 reproduced; \|T6\| = **13,056,344,064** |
| **G-1** | the random-draw model (each mined observation an independent draw from the DGP marginal; support = Poisson(N·p) ≥ 3) against the logs, per arm at its own `n_obs` | realized ÷ model, median [IQR]: **L2 0.91 [0.87, 0.94] · L3 0.42 [0.35, 0.48] · L4 1.68 [1.38, 1.91]**. Reproduces `antiphon` Phase 0(c)'s δ ≈ 0.44 at L3 on a 5× larger corpus. Unbiased at L2; the model is conservative at L4. |

---

## Premise corrections

**(1) The world is m = 2, not m = 4.** The brief for this lane says (v=8, s=2, m=4). Every banked
arm runs `m = 2` (`tall/` measured m=4 inadmissible for sculpting). All level arithmetic below is
m = 2.

**(2) The L5 gauge is not "degenerate at 1–2 tuples"; it is derivable and sub-floor.** A3's Phase 0
read the L5 series on `cd_s0`/`ma_s0` only — both `max_macro_level = 3`, 116–146 cycles — and
reported a maximum of 1–2 distinct tuples at support. Across the 40-arm corpus the L5 at-support-3
count reaches **7–11** on the `maxl = 4` long runs (max 11, `ca_s0/dsil_read`; median 2), first
non-zero at c71–c135. And `tol_yield_l5` **is derivable** by the arc's own null-ABBA method:
**0.0639** over 3,658 pooled windows. The correct statement is not *degenerate* but **sub-floor**
— see the floors table.

**(3) The r² wall is not what stops L5.** Over each arm's own frozen L4 book the L5 ratchet
ceiling is **1–15 true entries** (`an_s0/q_bisect`: 15, including the level's single most frequent
true key at p = 1.5e-5), because 31% of all L4×L4 pairs are legal L5 keys
(\|T5\|/\|T4\|² = 0.309). What stops L5 is **arrival**: E[true L5 keys at support 3] = **0.000** at
the run's own ~1,600 L5-node observations (measured 7.86–7.94 mined rows/cycle × 201 cycles;
`gy_final.n_obs` 1,596 / 1,585 / 1,383 on `anchor_long` / `q_bisect` / `dsil_read`). The arrival cut at L5 is ~5.9e9× against the ratchet
cut's 1.2–21× at L2–L4. This corrects [`../DESIGN.md`](../DESIGN.md) §8 reason 3 ("buildable L5
over a frozen L4 book of 4–5 entries is ≈ 0, so an L5 tag needs `census_extend` from the start"):
buildable is not 0, extension does move it (1→4, 2→7 on the arms that have it), and neither
matters at this observation budget.

**(4) The era-4/5 dose collapse is the level CAP, not depth.** `antiphon`'s caveat reads the
collapse as a fact about deep eras. It is a fact about `max_macro_level = 4` meeting era levels 4
and 5. Structural grip is **exactly (s−1)/s = 0.500 at every era** whenever the mined level tracks
the era (`level = era_level + 1`), and **exactly 0** the moment `era_level ≥ maxl`. At `maxl = 6`
the structural grip is 0.500 in all five eras. This is the question-port form of `tall/`'s standing
finding 1 (endogenous pacers cannot traverse a ladder longer than the earnable range): **the
question knob cannot grip past the earnable range either, and for the same structural reason.**

**(5) The parse-ambiguity junk mass is a property of the rule DRAW, not of the grammar.**
`antiphon` Phase 0(a) reports junk mass 0.24 / 0.58 / 0.82 at the L2/L3/L4 mining nodes as the
grammar's native aleatoric channel. `build_inverse_maps` is last-writer-wins, so junk exists only
where two of the v·m = 16 bottom tuples collide. Over twelve draws of the *same* (v, s, m, depth),
L4 junk mass ranges **0.000 → 0.905**; the arc's `rule_seed = 0` covers 14 of 64 codes (2
collisions) and is the second-worst of the twelve. **Seeds 6 and 10 are collision-free: zero parse
ambiguity at every level.** This moves the L5 budget by 6×–207× (below).

---

## Q1 — is a second extension well-posed on this world at this budget?

**Short answer: no at `rule_seed = 0`, and the binding wall is arrival, not r².**

### The levels (exact)

| level | span | per-feature m^(2^(l−1)−1) | distinct \|T_l\| | \|T_l\|/\|T_(l−1)\| | collision c_l |
|---|---|---|---|---|---|
| 1 | 1 | 1 | 8 | — | 1.000 |
| 2 | 2 | 2 | 14 | 1.8 | 0.875 |
| 3 | 4 | 8 | 56 | 4.0 | 0.875 |
| 4 | 8 | 128 | 816 | 14.6 | 0.797 |
| 5 | 16 | 32,768 | **205,824** | **252.2** | 0.785 |
| 6 | 32 | 2,147,483,648 | **13,056,344,064** | **63,434.5** | 0.760 |

Closed form `|T_l| = c_l · v · m^((s^(l−1)−1)/(s−1))`; the ratio `|T_(l+1)|/|T_l| = m^(s^(l−1))`
grows doubly exponentially and **does not depend on v**.

### The r² law at this rung — exact

Buildable true L5 vs L4 recall, 20 random L4 subsets per point:

| k (L4 entries) | 5 | 10 | 20 | 40 | 80 | 160 | 320 | 408 | 816 |
|---|---|---|---|---|---|---|---|---|---|
| recall | .0061 | .0123 | .0245 | .0490 | .0980 | .1961 | .3922 | .5000 | 1.0 |
| measured | 7.0 | 30.4 | 129.7 | 512.5 | 1968.5 | 7955.3 | 31570 | 51716 | 205824 |
| r²·205,824 | 7.7 | 30.9 | 123.6 | 494.6 | 1978.3 | 7913.3 | 31653 | 51456 | 205824 |
| ratio | 0.91 | 0.98 | 1.05 | 1.04 | 1.00 | 1.01 | 1.00 | 1.01 | 1.00 |

### Which wall binds, per rung

| rung | \|T_l\| | true@sup (median arm) | **arrival cut** | book true (median) | **ratchet cut** |
|---|---|---|---|---|---|
| L2 | 14 | 12 | 1.1× | 10 | 1.2× |
| L3 | 56 | 14 | 4.0× | 5 | 2.8× |
| L4 | 816 | 21 | 38.9× | 1 | 21.0× |
| L5 | 205,824 | **0.000** (modelled) | **5.9e9×** | n/a | n/a |

### The budget

Model E[true L5 keys at support 3] vs observations (8M-sample marginal; unseen true keys
contribute nothing, so a lower bound):

| N_obs | 1,600 | 16,000 | 80,000 | 160,000 | 400,000 |
|---|---|---|---|---|---|
| E | 0.000 | 0.033 | 3.43 | 22.5 | 223.8 |

| target book | N_obs | × the run's ~1,600 | cycles/arm @8/cyc | GPU-h/arm @10.6 s/cyc |
|---|---|---|---|---|
| 1 entry | 51,661 | 32× | 6,458 | 19.0 |
| 5 (a `crescendo`-sized L4 book) | 91,621 | 57× | 11,453 | 33.7 |
| 22 (`q_exo`'s realized L4 book) | 158,514 | 99× | 19,814 | 58.3 |
| 50 (`q_bisect`'s realized L4 book) | 217,449 | 135× | 27,181 | 80.0 |

**Sharpest form.** `an_s0/q_bisect` holds the arc's best L4 book (7 entries, 5 true). Its ratchet
admits 15 true L5 keys, including the level's most frequent one. E[# of those 15 at support 3]:
**0.0000** at N = 1,600 · 0.0026 at 16,000 · 0.7201 at 160,000 · 7.89 at 1,600,000.

### The L5/L6 streams, and the demand-cell test

Observed at-support-3 over a whole run: L5 max **11**, median 2, 1–11 distinct values per arm, first non-zero c71–c135 (never, in 4 arms);
L6 max **4** (`ca_s0/dsil_read`), and identically 0 in 31 of the 40 arms.

The L5 mining node is **1 in every era** and the L6 node is **0 in every era**
(`node = (era_node·s^(era_level−1)) // s^(level−1)`). The demand cell **cannot** move where L5 is
observed, so the L5 degeneracy is not a demand-cell fact. Observations run at a saturated
7.86–7.94/cycle (measured, `log["n_mined"]`); the only levers on N are cycles and `n_pr`.

### Junk mass at each mining node (rule_seed 0)

| level | node | true keys seen | junk keys | **junk mass** | true p: median | max | uniform |
|---|---|---|---|---|---|---|---|
| L2 | 12 | 14/14 | 9 | 0.241 | 3.5e-2 | 1.8e-1 | 7.1e-2 |
| L3 | 6 | 52/56 | 104 | 0.576 | 6.4e-3 | 5.5e-2 | 1.8e-2 |
| L4 | 3 | 768/816 | 7,649 | 0.815 | 1.1e-4 | 6.4e-3 | 1.2e-3 |
| L5 | 1 | 99,536/205,824 | — | **0.966** | 2.5e-7 | 1.5e-5 | 4.9e-6 |
| L6 | 0 | — | — | **0.999** | — | — | — |

The L5 true-key tail is *flatter* relative to uniform than L4's (max/uniform 3.2 vs 5.2), so
selection has less head to aim at, not more.

### The gauge at the L5 frontier

Floors derived by the arc's own null-ABBA method (`conductor/policy.py::null_abba`,
`tol = sd(N)/√W`), pooled over the 29 `maxl=4` arms, era boundaries and commit cycles skipped:

| level | windows | sd(N) | **v_tol** | mean(D) | **signal/floor** | frac(D > tol) | distinct values/arm |
|---|---|---|---|---|---|---|---|
| 3 | 3,658 | 0.6545 | 0.3273 | +0.4524 | **1.38** | 0.45 | 50.2 |
| 4 | 3,658 | 0.6943 | 0.3472 | +0.5295 | **1.53** | 0.54 | 62.4 |
| 5 | 3,658 | 0.1278 | **0.0639** | +0.0167 | **0.26** | 0.06 | 4.6 |
| 6 | 3,658 | 0.0413 | **0.0207** | +0.0017 | **0.08** | 0.01 | 1.4 |

A1's thermostat replayed on each arm's own L5 series from era-4 start, at the derived L5 floor:
**14/29 arms fire** inside the 17–21 consumption cycles, at c_in_era4 7…21 with no common
structure — the signature of a sub-floor gauge. **At the L5 rung neither the one-level-up read (L6,
0.08) nor the own-level read (L5, 0.26) clears its own floor**; the only above-floor at-support
gauges in the corpus are L3 (1.38) and L4 (1.53), i.e. reads from *behind* the frontier.

### Sizing the world instead (the fallback)

Exact ladders, and the L5 budget measured per world (2M-sample marginals, mutually comparable):

| (v,s,m) | m/v^(s−1) | L2 | L3 | L4 | L5 | L5 junk | N for a 22-key L5 book | × this run | GPU-h/arm |
|---|---|---|---|---|---|---|---|---|---|
| (8,2,2) — **this world** | 0.250 | 14 | 56 | 816 | 205,824 | 0.966 | 132,389 | 82× | 48.7 |
| (6,2,2) | 0.333 | 12 | 44 | 736 | 180,224 | 0.581 | 20,765 | 13× | 7.6 |
| (4,2,2) | 0.500 | 7 | 27 | 280 | 44,800 | 0.752 | 14,302 | 9× | 5.3 |

(v = 2 was measured and is degenerate: at m/v = 1.0 the bottom inverse map becomes constant and
every span parses to one key.) **No (v, s, m) puts \|T5\| near 816 while \|T4\| stays near 56** —
it would need m ≈ 1.4, and m = 1 collapses the grammar to v distinct sequences. And every world
whose L5 is cheap sits at an occupancy `m/v^(s−1)` at or above 0.5 — **exactly the occupancy of
v=8/m=4, which `tall/` measured inadmissible** (cross-world inference from a measured verdict, not
a new measurement).

### The rule draw — the cheapest lever measured

Same (v, s, m, depth); only `rule_seed` moves. Junk mass at the era mining nodes:

| seed | codes covered (of 64) | \|T5\| | L2 | L3 | L4 | L5 |
|---|---|---|---|---|---|---|
| **0** (the arc's) | 14 | 205,824 | 0.757 | 0.424 | 0.185 | 0.034 |
| 2 | 15 | 237,568 | 0.913 | 0.889 | 0.820 | 0.696 |
| **6** | **16** | 215,040 | **1.000** | **1.000** | **1.000** | **1.000** |
| 8 | 13 | 179,200 | 0.773 | 0.376 | 0.095 | 0.008 |
| **10** | **16** | 146,048 | **1.000** | **1.000** | **1.000** | **1.000** |

(full 12-seed survey in `phase0.json`; mass_true, so 1.000 = zero junk)

L5 budget by draw (2M-sample marginals throughout, so mutually comparable; absolute N is ~20%
optimistic against the 8M estimate above, which resolves more of the tail):

| seed | \|T5\| | L5 junk | E@1,600 | N for a 22-key book | × this run | cycles/arm | GPU-h/arm |
|---|---|---|---|---|---|---|---|
| 0 | 205,824 | 0.966 | 0.000 | 132,389 | 82× | 16,549 | 48.7 |
| 2 | 237,568 | 0.305 | 0.013 | 19,793 | 12× | 2,474 | 7.3 |
| 6 | 215,040 | 0.000 | 0.035 | 14,120 | 9× | 1,765 | 5.2 |
| **10** | 146,048 | 0.000 | 0.105 | **9,866** | **6×** | 1,233 | **3.6** |
| 8 | 179,200 | 0.992 | 0.000 | 332,404 | 207× | 41,550 | 122.3 |

`rhm/rhm_data.py` already carries `generate_rules_invertible`, which makes collision-freeness a
construction rather than luck (needs v·m ≤ v^s: 16 ≤ 64 here). **The trade that comes with it**: a
collision-free grammar has no parse ambiguity, so it deletes the arc's native aleatoric channel —
the junk mass `antiphon` Phase 0(a) used as its free nerdsnipe cell. The affordable-L5 world and
the free-aleatoric-trap world are the same world at different draws.

### The index op, as a level-size lever

The level-size explosion belongs to the **flat-tuple key**, not to the grammar. Under a merge by
parent feature (`fourwall`'s re-key basis) the level size is `v·m = 16` at **every** rung — a
12,864× shrink at L5. `census_extend` moves the L5 *ratchet* ceiling (1→4, 2→7 on the arms that
have it) but the ratchet is not the binding wall.

---

## Q2 — the deep-era question grip

**The determinator's answer: structural grip is depth-invariant at exactly 1/2 and dies at the
level cap; the whole-key dose falls doubly exponentially with depth. Both are true, and they are
different quantities.**

### The structural grip law (`target_geometry`, swept over the cap)

| era | maxl=3 | maxl=4 | maxl=5 | maxl=6 |
|---|---|---|---|---|
| L1n25 | L2 1/2 = **0.500** | L2 1/2 = **0.500** | L2 1/2 = **0.500** | L2 1/2 = **0.500** |
| L2n12 | L3 2/4 = **0.500** | L3 2/4 = **0.500** | L3 2/4 = **0.500** | L3 2/4 = **0.500** |
| L3n6 | L3 0/4 = 0.000 | L4 4/8 = **0.500** | L4 4/8 = **0.500** | L4 4/8 = **0.500** |
| L4n3 | L3 0/4 = 0.000 | L4 0/8 = 0.000 | L5 8/16 = **0.500** | L5 8/16 = **0.500** |
| L5n1 | L3 0/4 = 0.000 | L4 0/8 = 0.000 | L5 0/16 = 0.000 | L6 16/32 = **0.500** |

`grip = (s−1)/s` when `level = era_level + 1`; `grip = 1 − s^(era_level − maxl) ≤ 0` when the cap
binds. `antiphon`'s eras 4/5 are the second case.

### Grip in blocks vs bits vs residual

| level | \|T_l\| | \|T_(l−1)\| | grip (blocks) | grip (bits) | residual keys the ask cannot pin |
|---|---|---|---|---|---|
| 2 | 14 | 8 | 0.500 | 0.788 | 2 |
| 3 | 56 | 14 | 0.500 | 0.656 | 4 |
| 4 | 816 | 56 | 0.500 | 0.600 | 15 |
| 5 | 205,824 | 816 | 0.500 | 0.548 | 252 |
| 6 | 1.31e10 | 205,824 | 0.500 | 0.525 | 63,435 |

### The delivered dose, measured (`an_s0/log["q"]["dose_hit"]`, 7 arms × 201 cycles)

| era | target | has_clean | dose (mean over 7 arms) | 1/\|T_(l−1)\| | **κ = dose·\|T_(l−1)\|** | d\* mean |
|---|---|---|---|---|---|---|
| L1n25 | L2 | True | 0.4365 | 0.1250 | 3.49 | 1.70 |
| L2n12 | L3 | True | 0.1613 | 0.0714 | 2.26 | 2.34 |
| L3n6 | L4 | True | 0.0421 | 0.0179 | 2.36 | 3.19 |
| L4n3 | L4 (capped) | False | 0.0074 | 0.0179 | 0.42 | 4.24 |
| L5n1 | L4 (capped) | False | 0.0040 | 0.0179 | 0.22 | 5.52 |

Per-arm spread in era 1: 0.371 (`q_endo`) – 0.540 (`q_exo`). **The dose law**: where the question
has a clean half, dose ≈ κ/\|T_(l−1)\| with κ ≈ 2–3.5 — the ask fixes its half exactly and the
repair reproduces the other half at 2–3.5× chance. Extrapolated at κ = 2.5: **L5 dose ≈ 3.1e-3,
L6 dose ≈ 1.2e-5.**

### Grip × era × parameterization (each cell: `grip / dose`)

| parameterization | moves support? | era 1 | era 2 | era 3 | era 4 | era 5 |
|---|---|---|---|---|---|---|
| `cell_instance` (current, maxl=4) | n | 0.50 / 4.4e-1 | 0.50 / 1.6e-1 | 0.50 / 4.2e-2 | **0.00** / 7.4e-3 | **0.00** / 4.0e-3 |
| `cell_instance`, cap lifted | n | 0.50 / 4.4e-1 | 0.50 / 1.6e-1 | 0.50 / 4.2e-2 | **0.50** / 3.1e-3 | **0.50** / 1.2e-5 |
| `span_half` | y (within level) | 0.50 / 4.4e-1 | 0.50 / 1.6e-1 | 0.50 / 4.2e-2 | 0.50 / 3.1e-3 | 0.50 / 1.2e-5 |
| `probe_completion` (SPEC's bisection) | n | 1.00 / 4.4e-1 | 1.00 / 1.6e-1 | 1.00 / 4.2e-2 | 1.00 / 3.1e-3 | 1.00 / 1.2e-5 |
| `node_choice` | **Y** | 0.50 / 4.4e-1 | 0.50 / 1.6e-1 | 0.50 / 4.2e-2 | 0.00 / 7.4e-3 | 0.00 / 4.0e-3 |
| `rehearsal_episode` (`woodshed`) | n | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| `archive_slice` (`reread`) | n | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| `narrowed_deep_cell` | n | 0.50 / 4.4e-1 | 0.50 / 1.6e-1 | 0.50 / 4.2e-2 | **0.50** / 4.2e-2 | **0.50** / 4.2e-2 |

Reach and price, which the grip number does not carry:

- **`probe_completion` is not a different lever.** Grip 1.0 is over the *designed* key; the
  delivered column is the same measured number. This is what `q_bisect` already does.
- **`rehearsal` / `archive_slice`** get grip 1.0 and dose 1.0 because a stored episode reproduces
  its own key — but their reach is closed to keys **already observed**. Their lever is not arrival
  but **support**, which is the wall that binds at L5 (see below).
- **`node_choice`** is priced below and costs the arc's one hard norm.
- **`span_half`** is the only parameterization that widens reach without leaving the era's own
  cell (it lets either half be the fixed one, doubling the reachable half-key set per rung).

### Node choice, priced (per-node true-key marginals, 1M derivations)

| level | #nodes | era node | E[true@sup3]@1,600 (era node) | best node | E (best) | gain |
|---|---|---|---|---|---|---|
| 2 | 16 | 12 | 13.9 | 8 | 14.0 | 1.00× |
| 3 | 8 | 6 | 43.6 | 4 | 50.8 | **1.17×** |
| 4 | 4 | 3 | 22.7 | 1 | 23.7 | **1.04×** |
| 5 | 2 | 1 | 9.5e-5 | 0 | 1.4e-4 | (both empty) |

### Menu reach vs K

| K | L2 (of 14) | L3 (of 56) | L4 (of 816) | L5 (of 205,824) |
|---|---|---|---|---|
| 256 | 14 | 40 | 2 | 0 |
| 2,048 | 14 | 52 | 149 | **0** |
| 16,384 | 14 | 52 | 589 | **0** |
| 65,536 | 14 | 52 | 733 | **4** |

At L5 the menu is not the binding constraint either: the era's whole observation budget can drive
only ~186 keys to support 3.

### The damage-schedule counterfactual

| era | cell blocks | mined span | grip | d\* mean | blocks the learner must rebuild |
|---|---|---|---|---|---|
| L1n25 | 1 | 2 | 0.500 | 1.70 | 1 |
| L2n12 | 2 | 4 | 0.500 | 2.34 | 2 |
| L3n6 | 4 | 8 | 0.500 | 3.19 | 4 |
| L4n3 | 8 | 8 | **0.000** | 4.24 | 8 |
| L5n1 | 16 | 8 | **0.000** | 5.52 | 8 |
| L4n3 *narrowed to an L3 cell* | 4 | 8 | **0.500** | 3.19 | 4 |
| L5n1 *narrowed to an L3 cell* | 4 | 8 | **0.500** | 3.19 | 4 |

Grip returns exactly, with no new machinery. The cost is the eras' own demand: d\* falls
4.24 → 3.19 (era 4) and 5.52 → 3.19 (era 5), and the blocks the learner must rebuild halve (8 → 4)
in era 4 and quarter (16 → 4) in era 5. **A narrowed era 4 is era 3** — the consumption eras exist
to price coverage past what was earned (`census`/`assay`), which narrowing deletes.

### The support lever (what a rehearsal-shaped question attacks)

From the G-Y miner's own at-support histogram at end of run:

| arm | n_obs | ≥1 | ≥2 | ≥3 | ≥5 | **convertible (exactly 2)** |
|---|---|---|---|---|---|---|
| `cr3_s0/anchor_long` | 1,596 | 651 | 206 | 113 | 53 | 93 |
| `an_s0/q_bisect` | 1,585 | 602 | 210 | 132 | 52 | 78 |
| `an_s0/q_novel` | 1,599 | 711 | 259 | 144 | 60 | 115 |
| … (12 arms) | | | | | | median **100** |

Median convertible 100 against a median at-support-3 count of 113: re-posing the seen-twice keys
would ~1.9× the L4 at-support book in one op. Bounded by junk mass (0.82 at L4, 0.97 at L5 at
`rule_seed = 0`; 0.00 at a collision-free draw).

---

## Caveats

- Every number here is on **one rule draw** (`rule_seed = 0`) except §"The rule draw", which is
  the point of that section. Junk mass, and everything downstream of it, is draw-dependent.
- The L5 key stream is **not logged anywhere** — `obs_hist[5]` carries at-support-3 counts only,
  and the L5 key's other half (L4 node 2) is never mined in eras 1–4. So every L5 statement about
  *which* keys is modelled (the DGP marginal), not measured; the *counts* are measured.
- The budget curve ignores true keys unseen in the 8M sample, so it is a lower bound on E and an
  upper bound on N. The per-seed table in §"The rule draw" uses 2M samples throughout: internally
  comparable, ~20% optimistic in absolute N.
- The random-draw model is the DGP's clean draw. Gate G-1 measures what the repair, the reader and
  the agent's own narrowness do to it (0.42–1.68 across L2/L3/L4); the L5 extrapolation carries
  that uncertainty.
- The world-sizing occupancy argument is a **cross-world inference** from `tall/`'s measured
  m-inadmissibility verdict, not a new measurement.
- The parameterization table's `dose` column is measured in eras 1–3 and extrapolated (κ = 2.5)
  beyond; `rehearsal`/`archive` doses of 1.0 are by construction, not measurement.
