# census / PHASE 0 — decision document

**What was done**: every candidate gauge replayed offline over `spiral/figures/sp_s0/`'s logs,
all four earning arms, both levels, over a (window × threshold × support) grid. No GPU, no
Modal spend. Reproduce with `python3 rhm/practice/census/phase0_replay.py`; full printout in
[`report.txt`](report.txt), machine-readable in [`phase0.json`](phase0.json).

**Ladder, in absolute cycles** (a correction worth making first): era 2 is 56 cycles *long* but
spans **c33–c88**. L2 mining opens at c1, L3 mining at c33 — `run_arm` mines levels
`2..min(max_macro_level, era_level+1)`.

| era | cell | level | cycles |
|---|---|---|---|
| 1 | L1n25 | 1 | c1–c32 |
| 2 | L2n12 | 2 | **c33–c88** |
| 3 | L3n6 | 3 | c89–c100 |
| 4 | L4n3 | 4 | c101–c109 |
| 5 | L5n1 | 5 | c110–c116 |

---

## 1. Gauge-by-gauge verdict

| gauge | verdict | why |
|---|---|---|
| **G-Y** next-level yield (T4 stream) | **NOT OBSERVABLE — unmeasured, not degenerate** | No level-4 miner is constructed anywhere in `sp_s0`: `miners = {l: Miner(l,s) for l in range(2, max_macro_level+1)}` with `max_macro_level=3`. `log["miner"]`, `log["aud"]` and the ablation battery's `mine` block all carry levels `['2','3']` only, in all seven arms. Nor is it reconstructible: a T4-shaped observation is a specific pair of **adjacent** L3-shaped spans in one configuration, and `Miner.state()` stores per-key counts with no co-occurrence and no positions. |
| **G-A** admission rate at support | fires, but **cannot be validated as a gate** (§2) | Firing cycle spans **c7–c112** across the grid — a 105-cycle range. At short windows (W=3) it fires *6–7 cycles before* the certificate in all four arms: an accelerator, not a gate. |
| **G-C** buildable-coverage plateau | **collinear with G-A at L2 — drop as a separate gauge there** | At L2 the ratchet's lower table is `MC.base_table(v)`, complete by construction, so every tuple at support builds into an entry. The two series are **bit-identical in all four arms** (max gap 0). They separate only at L3 (max gap 10 / 20 / 24 / 10). |
| **G-D** frozen-vs-live divergence | **non-degenerate and immediately usable** | Extension opportunities exist at every recert from c37 (routed) / c57 (enum) at L2 and c93/c98 at L3. Crucially the live table **dominates the frozen one on recall *and* precision at L3 in every arm** — extension is not a precision/recall trade there. |

---

## 2. Finding 1 — no gate-later gauge can be validated on these logs, in principle

Under `delta_prov` the certificate is what committed: **cert cycle == commit cycle in 8 of 8**
(arm × level; L2 c19/c18/c18/c19, L3 c48/c48/c52/c48). "Gate-later" means *fires after the
certificate*, which is therefore exactly *after the commit* — and after the commit the logged
series come from a trajectory in which the level was already frozen and its macros already in
the action set, changing the very chosen trajectories that mining reads.

So the replay is **exact up to the certificate and an extrapolation after it**, and every grid
setting that fires later than the certificate does so in the extrapolation zone. The only
replay-valid firings are the ones that fire *earlier* than the certificate, which do not gate —
they accelerate.

This is a scope limit, not a caveat. **Phase 0 can bound what a gate could find; it cannot
choose a gate's constants.** What follows is written accordingly.

---

## 3. Finding 2 — at L3 the binding constraint is the frozen L2, not the observation stream

Every earning arm observes more L3-shaped tuples at support than it can build. `Miner.build`
keeps an L3 tuple only if **both** its L2 halves are entries of the frozen L2, so a frozen L2
of recall `r` caps buildable L3 coverage at about `r²`.

| arm | L2 frozen recall | L2 live recall (end) | **L2 coverage on offer** | L3 ceiling ≈ r² | L3 frozen recall | L3 live recall | stream − buildable |
|---|---|---|---|---|---|---|---|
| `enum_live` | 0.786 | 0.857 | +0.071 | 0.617 | 0.054 | 0.179 | 10 |
| `spiral_route` | 0.571 | 0.929 | **+0.357** | 0.327 | 0.089 | 0.179 | 20 |
| `spiral` | 0.571 | 0.929 | **+0.357** | 0.327 | 0.071 | 0.143 | 24 |
| `enum_live_g722` | 0.786 | 0.857 | +0.071 | 0.617 | 0.054 | 0.179 | 10 |

An L3 gate in the routed arms cannot exceed **0.327** recall however long it holds — the L2
freeze has already spent the budget. At L2 the lower table is the complete base table, so L2
coverage is limited only by the stream.

---

## 4. Finding 3 — the perfect-L2 control bounds what any L3 gate could ever deliver

`given` / `fid` / `given_native` hold the DGP's own L2 as their operative lower table (not
ratchet-limited) and never freeze an earned table — so their live L3 candidate series is an
84-cycle, unfrozen, un-ratcheted mining trajectory. Exactly the trajectory a perfect L2 gate
followed by an L3 gate would produce, already on disk.

| arm | role | L3 at support | L3 built | **L3 recall** | **L3 precision** |
|---|---|---|---|---|---|
| `given` | perfect-L2 control | 35 | 22 | **0.286** (16/56) | **0.727** |
| `fid` | perfect-L2 control | 35 | 22 | 0.286 | 0.727 |
| `given_native` | perfect-L2 control | 22 | 13 | 0.196 | 0.846 |
| `enum_live` | earning | 35 | 25 | 0.179 | 0.400 |
| `spiral_route` | earning | 40 | 20 | 0.179 | 0.500 |
| `spiral` | earning | 40 | 16 | 0.143 | 0.500 |

**With a perfect L2, no freeze, and 84 cycles of mining, L3 still ends at 29% coverage.** The
earning arms' 0.143–0.179 is therefore not mostly a gating failure — it is mostly a mining-rate
and ratchet fact. The frozen L2 also costs **precision**: 0.727 with a true L2 against
0.400–0.500 with an earned one.

---

## 5. Finding 4 — the extension mechanism does not run where its value is measured

`run_arm`'s recert fires only for `rc_level = era["level"]`, and only if that level is already
committed. On this ladder:

| arm | L2 recerts | L3 recerts | eras 4–5 |
|---|---|---|---|
| all four earning arms | **11** (era 2) | **2** (era 3) | **none** |

In eras 4 and 5 `era["level"]` is 4 and 5, and `committed[4]` / `committed[5]` do not exist
(`max_macro_level=3`), so no recert fires at all. **Eras 4–5 are the round's money readout.** A
`census_extend` arm built on the inherited recert path would add nothing during the eras where
its value is measured, and the L3 extension op would get 2 chances in the entire run.

---

## 6. What extension would have had available (G-D, replay-valid throughout)

Extension is licensed post-commit, so unlike a gate it is *inside* the valid window at the
moment it acts on the frozen table — the frozen table is a fact, not a counterfactual.

| level | first opportunity | entries available | `n_diff` (audition answers a swap changes) |
|---|---|---|---|
| L2, enum arms | c57 | 1 → 2 | 0 → 20 |
| L2, routed arms | c37 | 3 → 5 | 0 → 8 |
| L3, all arms | c93 | 6 → 9 | 1 → 77 |

End-of-run frozen-vs-live gap (what extension would close):

| arm | L2 frozen → live | L3 frozen → live |
|---|---|---|
| `enum_live` | 15 → 17 (rec 0.786 → 0.857, prec 0.733 → 0.706) | 11 → 25 (rec **0.054 → 0.179**, prec **0.273 → 0.400**) |
| `spiral_route` | 11 → 19 (rec 0.571 → 0.929, prec 0.727 → 0.684) | 11 → 20 (rec **0.089 → 0.179**, prec **0.455 → 0.500**) |
| `spiral` | 11 → 20 (rec 0.571 → 0.929, prec 0.727 → 0.650) | 9 → 16 (rec **0.071 → 0.143**, prec **0.444 → 0.500**) |

At L3 extension is **strictly dominant on both metrics in every arm**. At L2 it trades ~0.03–0.08
precision for +0.07–0.36 recall.

---

## 7. Recommendations, numbers attached

**R1 — Move the gate from L3 to L2.** The SPEC aims `census_gate` at the L3 commit. The replay
says that is the wrong level: an L3 gate is capped at 0.327 recall by the L2 freeze (§3) and at
0.286 even with a perfect L2 (§4), while **L2 coverage on offer is +0.357 recall** in the routed
arms. And it compounds: lifting frozen L2 recall 0.571 → 0.929 lifts the L3 buildable ceiling
**0.327 → 0.863**, and the perfect-L2 control says it lifts L3 *precision* toward 0.727. Gating
L2 is the only version of the op whose payoff is not pre-spent.

**R2 — Keep `census_extend`, but widen the recert loop first.** Iterate every *committed* level
in every era, not `era["level"]` alone. Without this the arm is inert in eras 4–5 (§5). This is a
code change to the fork, not a design change to the op, and it also gives L3 extension ~30
opportunities instead of 2.

**R3 — Drop G-C as a separate gauge.** It is bit-identical to G-A at L2 (§1). Keep it only if an
L3 gate survives R1, where it is the ratchet-aware variant.

**R4 — Cut the `census_yield` arm; add the G-Y instrument.** The gauge has never been observed,
so an arm built on it is designed blind. But the instrument is nearly free: an **unpriced
level-4 instrument miner** (span `s³` = 8 blocks = 16 tokens) at the era's own L4 node, decoupled
from `max_macro_level` (which caps what may be *committed*, not what may be *observed*). Level 4
stays unearnable — 1,024 entries, ~384 cycles to cover — so this is a growth-**direction**
readout only, never a commit candidate. The SPEC's readouts list already wants it.

**R5 — Constants: do not take them from this replay.** Per §2 no gate setting is validatable
here. Instead make the gate a **conjunction** — commit when the certificate has fired **and** the
coverage gauge is quiet — so the op is strictly "hold longer than the certificate", `yoked_delay`
is exactly interpretable as delay-alone, and no setting can accidentally *accelerate* the commit.
As an a-posteriori starting point for the L2 gauge, **G-A on the at-support series at W=12,
θ=0**: it is the only grid setting that fires after the certificate in all four arms
(c30 / c40 / c36 / c30 against certs c19 / c18 / c18 / c19, i.e. **+11 to +22 cycles**), and it is
the least sensitive corner of the grid. Flagged: those cycles are extrapolations of unknown sign.

**R6 — Era sizing: it is ERA 1 that needs the cycles, not era 2.** With the gate moved to L2,
the binding constraint is that the L2 gate must fire inside era 1. Era 1 is c1–c32; the latest
all-arm L2 firing at the recommended setting is **c40** (`spiral_route`), i.e. **8 cycles past the
era-1 boundary**. Recommend **era 1 ≥ 48 cycles** (c40 + 20% margin for the extrapolation's
unknown sign), taken from era 2, which under R1 no longer needs to host a gate:

| | `sp_s0` | Phase 1 proposal |
|---|---|---|
| era 1 (earns L2, **now gated**) | 32 | **48** |
| era 2 (earns L3) | 56 | **40** |
| eras 3 / 4 / 5 (consumption) | 12 / 9 / 7 | 12 / 9 / 7 |
| total per arm | 116 | **116** |

Total cycles unchanged, so the ~4.3 GPU-h cost of `sp_s0` is the cost estimate for Phase 1 at
seven arms; at the SPEC's ≤5 GPU-h target there is room for six or seven arms.
*If* the coordinator wants an L3 gate retained despite R1, the replay's answer to the original
question is: the latest all-arm L3 firing on the grid is **c96**, so era 2 would need **≥64
cycles** (it has 56) — but see §3 and §4 for why that purchase is capped at 0.327 recall.

**R7 — Arm list consequence.** `census_yield` cut (R4). If both `census_gate` (now at L2) and
`census_extend` are kept along with `spiral_route`, `given_native`, and `yoked_delay`, that is
five arms plus an in-tag fidelity anchor — comfortably inside budget at 116 cycles.

---

## 8. What Phase 0 could not settle, stated plainly

- **Whether any gate-later gauge separates usefully from the certificate.** Structurally
  unanswerable from these logs (§2). Phase 1's `yoked_delay` arm is what settles it.
- **Whether the next-level-yield gauge works at all.** Unmeasured (§1); R4 makes it measurable
  but as a readout, not as a gate.
- **Whether bought coverage moves eras 4–5.** That is the money readout and it needs the live
  run. Phase 0 only establishes how much coverage is *available* to buy (§3, §4, §6).
