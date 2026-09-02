# FILES — `antiphon` (the question port: can question-choice be an outer-loop action?)

Machinery record for the node. **No results are interpreted here** — the reduction's numbers go
to the orchestrator and are discussed before any writeup (repo norm).

**Up**: [`SPEC.md`](SPEC.md) (the brief) · parent arc [`../README.md`](../README.md) ·
`QUEUE.md`[^private] "the question port" · `ROADMAP.md`[^private]
§7.1–§7.2
**Direct donor** (untouched): [`../crescendo/`](../crescendo/FILES.md) — A3. `crescendo.py` is
forked here; `maestro/policy.py` and `maestro/floors.json` are reached by **import**, as in the
donor, so the rule and the dead zones are still A1's. `cr3_s0/anchor_long` is the cross-tag
replay reference.
**Through it**: [`../maestro/`](../maestro/FILES.md) · [`../conductor/`](../conductor/FILES.md)
(the thermostat, the measured floors, the yoke and panel machinery) ·
[`../census/`](../census/README.md) (the r² ratchet, the extension op) ·
[`../woodshed/`](../woodshed/FILES.md) (the trust-rate instrument; the `wd_s0` dose lesson) ·
[`../ostinato/`](../ostinato/FILES.md) (the ρ-knob house style; *size the premise offline first*).

---

## 1. The question parameterization, and why this one

The SPEC leaves it open and names four candidates (a chosen damage cell; a probe completion; a
rehearsal episode; an archive slice). This node takes **the posed instance as the learner's
action**: each cycle the world offers a *menu* of candidate repair episodes drawn from the era's
own damage cell, and the arm's selector picks the `n_pr = 64` it will actually practise on.

The reason is that on this substrate the posed instance has a **direct, exact, computable lever
on which rules get mined**, and no other candidate does:

```
posed instance -> its clean derivation's level-l constituent at the mining node
               -> (after repair + the reader's parse) one observation of that key
               -> at mine_support = 3 observations, a candidate table entry
               -> buildable iff both halves are rows of the operative lower table  (the r² wall)
```

`run_arm` block (a) draws `n_pr` fresh instances of the era's cell; block (b) mines up to
`mine_cap = 8` of the **solved** answers, reading the level-ℓ span that contains the era's cell.
So "which instance do I pose" selects, one level up, *which grammar rule the learner gets to
see*. That is Jasper's "questions which bisect certain RHM rules", in the substrate's own terms.

**Structurally, half of every key is delivered by the question itself.** The damage cell at
level ℓ is exactly one of the two halves of the level-(ℓ+1) span that gets mined, so the other
half is untouched by the repair and reaches the miner exactly as the posed instance set it. The
selector therefore controls at least half of every mined key by construction — which is what
makes the lever real rather than a hope about the agent's repair behaviour. (Which half is clean
alternates with the era: L1n25 and L4n3 damage the *second* half, L2n12 and L3n6 the first.)

**Two candidates were considered and rejected, with reasons.**
- *Mining-side selection* (choose which solved answers to mine from) is cheaper and moves the
  same key stream — but it is a **post-consumption diet gate**, and that seat is closed by
  measurement in both regimes (`two_deltas` findings 1 and 6). The question port is upstream of
  consumption; that is the whole point of the arity-2 reading (asking is acting).
- *Choosing the damage node* (16 level-2 nodes rather than the era's one) would widen reach,
  because the key marginal differs per node — but it moves the *support* of the demand
  distribution, not its selection, and the SPEC's one hard norm is that only selection may move.
  Recorded as the obvious follow-on if reach turns out to bind.

**In this parameterization the exogenous ladder and "random questions" are the same arm.** The
era ladder's demand *is* a uniform draw over instances of the era's cell, so QUEUE's "exogenous
ladder" and "random questions" coincide; the `exo` arm is both, and is bit-identical to the
donor. Two of QUEUE's five sketched arms collapse into one, honestly, and the freed budget pays
for the aleatoric control (below).

---

## 2. Phase 0 — the round sized offline, before any GPU

`phase0_question.py` → `phase0.json`. No GPU, no Modal, no substrate: the DGP's own rules, the
substrate's own `buildable()` set arithmetic (reused verbatim from `crescendo/phase0_l4.py`),
and the banked logs of `cr3_s0` / `ma_s0` / `cd_s0`.

**(a) The DGP's own parse ambiguity — the aleatoric channel is already installed.**
`build_inverse_maps` is last-writer-wins, so a legal derivation can parse to a level-1-feature
tuple that is *no table row at all*. At the earning eras' mining nodes:

| level | node | true keys reachable | junk keys | mass on true |
|---|---|---|---|---|
| L2 | 12 | 14 / 14 | 9 | **0.757** |
| L3 | 6 | 52 / 56 | 104 | **0.424** |
| L4 | 3 | 768 / 816 | 7573 | **0.184** |

So 58% of L3 observations and 82% of L4 observations are *structurally* junk, before the learner
makes a single error — which is why every donor arm's mined table sits at precision 0.19–0.41.
The SPEC's A3′ wanted `tuning`'s magnitude-matched bursts to install an aleatoric trap; the
substrate has one, natively, and it is the dominant channel. **A novelty-greedy question judge
maximises exactly this** (junk keys are maximally novel — 7573 of them at L4), so the nerdsnipe
is free to test and needs no burst machinery. This is the round's cheapest finding-shaped fact
and it is why the arm set below carries `novel` beside `endo`.

**(b) The true-key marginal is heavy-tailed, which is where the headroom lives.**

| level | min p_k | q25 | median | q75 | max | uniform would be |
|---|---|---|---|---|---|---|
| L3 | 4.9e-4 | 2.9e-3 | 6.3e-3 | 1.2e-2 | 5.5e-2 | 1.79e-2 |
| L4 | 2e-6 | 4.8e-5 | 1.1e-4 | 2.8e-4 | 6.2e-3 | 1.23e-3 |

Under a uniform draw the tail needs ~3/p_k observations to reach support; the whole era-2 budget
is 400 observations. **Selection buys the tail; volume cannot.**

**(c) Calibration against the logs (the delivery-fidelity prior).** A random-draw model — each
mined observation an independent draw from the marginal above — predicts keys-at-support after
N observations. Realized ÷ model, over cycle-snapshots of every donor arm:

| level | median | IQR | n snapshots |
|---|---|---|---|
| L2 | 0.75 | [0.70, 0.80] | 2063 |
| **L3** | **0.44** | [0.40, 0.50] | 1227 |
| L4 | 2.27 | [1.83, 2.85] | 2063 |

The L3 shortfall is the gap between the DGP's clean draw and what the repair plus the reader
actually deliver — this round's **delivery-fidelity prior, δ ≈ 0.44**, and the smoke measures it
directly (designed key vs delivered key, per era, per level). The L4 ratio above 1 says the
agent's own productions are *narrower* than the DGP's draw — it repeats its own repertoire — so
the L4 targeting is the more optimistic half of the prediction and is reported as such.

**(d) Menu reach vs K, and the budget-matched choice K = 2048.** A key is reachable if a menu of
K candidates contains it with probability > ½:

| K | L2 reach (of 14) | L3 reach (of 56) | L4 reach (of 816) |
|---|---|---|---|
| 256 | 14 | 40 | 2 |
| 1024 | 14 | 50 | 64 |
| **2048** | **14** | **52** | **153** |
| 4096 | 14 | 52 | 292 |
| 8192 | 14 | 52 | 453 |

The observation budget of the era that earns each level (cycles × `mine_cap` 8) is L2 480, L3
400, L4 560 — i.e. at most 160 / 133 / 186 keys can be driven to support 3 at all. **K = 2048
matches reach to budget at both L3 (52 ≤ 133) and L4 (153 ≤ 186); K = 4096 buys reach the budget
cannot spend.** Menu draw cost measured at 0.14 s per 2048 candidates (numpy: sample, corrupt,
`nearest_derivation_cost`) against a ~11 s cycle.

**(e) The r² payoff curve** — buildable *true* L4 vs L3 recall, by the substrate's own
`buildable()` over the true tables (40 random L3 subsets per point):

| L3 recall | 0.089 | 0.179 | 0.268 | 0.357 | 0.536 | 0.893 |
|---|---|---|---|---|---|---|
| buildable true L4 | 6.5 | 23.3 | 54.1 | 102.4 | 234.0 | 655.0 |
| r²·816 | 6.5 | 26.0 | 58.5 | 104.1 | 234.2 | 650.5 |

The r² law is exact. **A 2× move in L3 recall is a ~4× move in the L4 book's ceiling**, which is
the level `crescendo` left the arc stuck on (its committed L4 book was 5 entries, 2 true).

**(f) The predicted window.** At K = 2048 and the calibrated δ = 0.44, the oracle arm's L3
at-support lands at ≈ 23 (recall 0.41) against buildable-true-L4 ≈ 136; the band over
δ ∈ [0.3, 1.0] is recall 0.28–0.93 and buildable 63–704. The incumbent's measured points:

| where | N_obs | L3 true@sup | recall | buildable true L4 | committed L4 book |
|---|---|---|---|---|---|
| `cr3_s0/anchor_long` at its L3 commit (c75) | 120 | 3 | 0.054 | ~6 | 5 entries / 2 true (c180) |
| `cr3_s0/outer_yield_m4` at its L3 commit (c92) | 288 | 10 | 0.179 | ~23 | 5 entries / 2 true (c129) |
| `cr3_s0/anchor_long` at end of run (c201) | 1124 | 19 | 0.339 | ~85 | — |

**The premise sizes, and it sizes as a CLOCK claim, not an endpoint claim.** The incumbent
eventually coupon-collects to recall ≈ 0.34 by cycle 201 — the oracle's end-of-run advantage is
therefore modest. But the table is *frozen at the commit*, and at the incumbent's own commit
cycle the predicted contrast is recall 0.054 → ~0.31 (≈ 6×) at the schedule arm's c75 and
0.179 → ~0.41 at c92. This is exactly "does question quality move climbing speed", and it has a
second reading worth stating plainly: **`crescendo`'s thermostat already buys coverage — by
spending cycles** (it committed L3 17 cycles later than the schedule arm, at 2.4× the
observations). The question port buys the same coverage out of selection instead of time. The
two are separable levers on one quantity, which is why the arms below are all schedule-paced.

---

## 3. The arms, and what each control pins

Five arms, one seed, all on **`anchor_long`'s configuration** (schedule pacer,
`CRESCENDO_LADDER` ≡ `CRESCENDO_CAPS` = 60/50/70/12/9, `max_macro_level=4`,
`commit_max_level` unset), so **lifetime is identical at 201 cycles by construction** and no
deep-era comparison can have been bought with time. They differ in **one thing**: the function
that picks 64 of the menu's 2048.

| arm | selector | what it is |
|---|---|---|
| `q_exo` | the menu's first 64 — drawn by the donor's own `context_instances` call, with the donor's RNG | **the exogenous ladder = random questions** (they coincide here), and a full-scale bit-identical replay of `cr3_s0/anchor_long` |
| `q_bisect` | oracle: exact features of the *clean* derivation → designed key; prefer true keys not yet at support in this arm's own miner, and prefer keys both of whose halves are rows of its own operative lower table | **the ceiling** (r²-aware bisection; exact oracles grade and choose here and are never consultable elsewhere) |
| `q_endo` | the learner's own reader parses the **undamaged half** of the target span → a half-key; prefer half-keys matched by no at-support key of its own miner; multiplied by a **delivery ledger** (posed vs landed, EWMA per half-key) | **Δ`at_support` per priced sample, thermostat-grade** — the SPEC's candidate currency, with the noisy-TV guard |
| `q_novel` | the same half-key novelty **without** the delivery ledger | **the nerdsnipe control** — novelty is maximised by the junk keys §2(a) measured, so this is the aleatoric trap with no guard |
| `q_comp` | highest own value-head `v(x0)` within stratum | **the known negative, difficulty-pinned** — comfort in *content* at matched difficulty |
| `q_bisect_loop` | the oracle selector on `outer_yield_m4`'s thermostat | **additive, outside the core** — does selection *compose* with the pacer? The pacer already buys coverage by spending cycles (17 more, for 2.4× the observations, in `cr3_s0`), so whether selection is redundant under loop pacing is the cell that ties this tag to the SPEC's outer-loop framing. Comparator: the banked `cr3_s0/outer_yield_m4`, not the schedule family |
| `q_endo_loop` | the endogenous judge on `outer_yield_m4`'s thermostat | **[s2] THE CELL** — the composed arm for the object the round is about. `an_s0`'s composition cell put the *oracle* in the loop; this one puts the learner's own Δ`at_support`-per-priced-sample judge there. Twin: `q_endo` |
| `q_comp_loop` | highest own `v(x0)` within stratum, on the thermostat | **[s2] the comfort pole under the pacer** — does pacing amplify or *rescue* the difficulty-collapse hazard that made `q_comp` the tag's slowest climber? The composed hazard is specific: the thermostat reads the learner's own yield, and a comfort-seeking selector feeds it easy solves. Twin: `q_comp` |
| `q_novel_loop` | half-key novelty, guard removed, on the thermostat | **[s2] the nerdsnipe pole under the pacer**, and the ablation that says what the delivery ledger buys *in the loop*: a nerdsniped arm banks junk keys, which is exactly what the yield gauge counts. Last in `an_s2`'s run order, most cuttable. Twin: `q_novel` |
| `q_trust` | novelty × the arm's **own beam use record** over its own committed rows one level below the clean half | **[m] the use-record selector.** Spec-identical to the trap lane's `t00_trust` (its trap-off twin); named in the `q_*` family so the metered tag's premium table is uniform. Rationale, measured offline: `trap/DESIGN.md` §4.8 — hard membership in the arm's own table is no guard (it prefers the arm's own committed junk 50–75% of the time, table precision 0.25–0.50), but *use-weighting* the same rows cuts the junk share of the judge's pool from 0.686 to 0.43–0.54 in every arm. Reads no truth, no exact map, no clean derivation. **Era-1 fallback**: at era 1 the target level is 2, so `sub = 0`, there is no lower table, `subw = {}` and the selector falls through to plain novelty — stated, not hidden |
| `q_comp_free` | highest own value-head `v(x0)` over the whole menu, **no quota** | **the known negative, unpinned** — the honest `fourwall` β=2 pole. The pinned arm alone cannot reproduce that negative (the pin removes the main comfort axis), so a null there would be ambiguous between "instrument blind" and "pin removed the failure mode"; the pair decomposes the comfort pole into difficulty-comfort and content-comfort. Volume, priced budget and lifetime stay pinned; **the realised difficulty mix is logged every cycle, because the deviation IS the arm's definition**. Last in run order: most cuttable |

**Which half of the key the question fixes.** The damage cell at level ℓ is exactly one of the two
halves of the level-(ℓ+1) span that gets mined, so the *other* half reaches the miner as the
question set it, whatever the repair does. Measured by gate Q-9: era `L1n25` leaves block offset
[0] clean, `L2n12` leaves [2,3], `L3n6` leaves [4,5,6,7] — and the two consumption eras (`L4n3`,
`L5n1`, 21 of 201 cycles) leave nothing clean, because their cell swallows the whole span. So the
endogenous half-key signal exists in the three *earning* eras and is empty in the last two, where
every selector falls through to a quota-legal index order. Stated, not hidden; logged as
`has_clean` per cycle.

**What is pinned, and how.**

1. **Total volume.** Every arm poses exactly `n_pr = 64` and mines exactly `mine_cap = 8`. The
   mined count is asserted equal to 8 every cycle in every arm (it is 8 in every donor cycle);
   any shortfall means the selector starved the solve rate and is reported as a control failure,
   not absorbed. This is the `wd_s0` lesson's second half.
2. **Difficulty mix — exactly, per cycle.** `nearest_derivation_cost` gives d\* for every menu
   candidate (the donor already computes it for the `require_broken` rejection). The cycle's
   **quota** is the d\* histogram of the menu's first 64 — i.e. the donor's own realized
   difficulty mix — and every arm must fill that histogram bin-for-bin. So selection moves
   *which instance within a difficulty stratum*, never the difficulty. The d\* alphabet is
   non-degenerate in every era (3 / 5 / 9 / 12 / 16 strata for eras 1–5).
3. **Priced budget.** Identical beam (`budget=8`, `g_budget=482`, `fit_width`) on identical
   `n_pr`. Cumulative groundings `t_cum` are logged per cycle and every comparison is reported
   **both** at matched cycle and at matched cumulative priced spend.
4. **Compute for the selection itself.** The reader parse over all 2048 menu candidates and the
   value-head forward over all 2048 are run in **every** arm, used only by the arms that read
   them — `ostinato`'s rung-invariant-comparator discipline. Charged in `ReadLedger` at a
   measured price and reported; no arm gets a free read another pays for.
5. **Lifetime.** 201 cycles for every arm, by the schedule pacer at the caps.
6. **Oracle containment.** The selector functions take disjoint argument bundles; a gate asserts
   that the `endo` / `novel` / `comp` bundles contain no truth table, no exact inverse map and
   no clean derivation (`conductor`'s N-1 label-freeness pattern).

---

## 3a. [m] The strongly-metered regime (`an_m0`) — the port where questions are scarce

**Why.** Every question-port run so far ran at **abundance**: `n_pr = 64` posed per cycle,
~15 solved, `mine_cap = 8` taken by the donor's subsample, π's buffer full by c26–c35. The
knob's effect on the table is diluted twice and every arm asks more than it can use. The claim
under test is the roadmap's: knowing *which* questions to ask is load-bearing when questions are
**scarce**.

**The knob** is `intonation` Round A's, unchanged: `n_pr` 64 → 24 and `pr_width` 16 → 8 — *the
meter can afford 24 graded performances a cycle instead of 64, and a width-8 search instead of
width-16*. Both are purchases. The grammar, the ladder, the caps, `mine_support`, the plan
budget and every floor are untouched, so the world's truth does not move. Both are already
`cfg` keys on this stack (`antiphon_run(..., n_pr, pr_width, ...)` → `_d6_cfg`), so **no code
change was needed for the knob itself**.

**`mine_cap` is deliberately NOT scaled**, and neither is `prop_buf_cap`. They are the arc's own
constants and the fact that the mining cap **stops binding** *is* the regime. The consequence is
stated rather than engineered around: with solves below the cap the mining subsample is never
taken, so **the mined stream IS the solved stream**, and a selector that solves *more* mines
more. Under §3's volume pin that would read as a control failure; here it is the finding, and
the reduction reports solves/cycle and mined/cycle per arm rather than a pass/fail on the cap.

**The two arcs sit at the same operating point**, which is what makes Round A's metered numbers
a usable prior: `an_s0/q_exo` runs 15.08 solves/cycle at 1.88× the cap, cap binding 97.5%, π
buffer full at c30 — against `in_s0`'s 14.9 / 1.87× / 95.4% / c26.

**Sized offline first** (`phase0_metered.py` → `phase0_metered.json`), from `an_s0`'s own
per-cycle record, exactly as `ma_s0` was sized from `in_s0`'s. A `--quick` smoke cannot size a
regime — at quick scale every regime looks scarce. Holding the solve *fraction* fixed gives an
**upper bound** by construction, since a width-8 search solves a smaller fraction (`ma_s0`
predicted 5.60 this way and realized 4.28–4.65):

| | abundance (`an_s0`, measured) | metered (predicted, upper bound) | `ma_s0` (measured) |
|---|---|---|---|
| solves/cycle | 15.01–16.93 | **5.99** | 4.28–4.65 |
| vs `mine_cap` 8 | 1.88–2.12× | **0.75×** | 0.54–0.58× |
| cap binds | 95.5–98.5% | **11.5%** | 4.6–5.9% |
| obs/cycle | 7.89–7.97 (the cap) | **5.88** | 4.17–4.54 |
| π rows/cycle | 2071–2335 | **828** | 149–322 |
| π buffer full | c26–c35 | ~c73 | never |

**The per-era observation budget** falls to 0.74× overall (1608 → 1182 observations): era 1
480 → 359, era 2 400 → 303, era 3 560 → 405, era 4 96 → 69, era 5 72 → 46.

**The sizing fact the round turns on.** At K = 2048 the menu reaches 14 / 52 / 153 true keys at
L2 / L3 / L4, and the budget can drive `budget / support` keys to support:

| level | reach @ K=2048 | keys the budget can drive, abundance | metered | |
|---|---|---|---|---|
| L2 | 14 | 160 | 119 | reach ≤ budget in both |
| L3 | 52 | 133 | 100 | reach ≤ budget in both |
| **L4** | **153** | **186** | **135** | **the budget BINDS at the metered knob** |

Under abundance the budget could in principle drive every reachable key at every level. At the
metered knob it cannot at L4 — so *which* of the reachable keys the questions aim at starts to
determine the L4 book. **That is the structural condition selection needs, and this round is the
first in the arc that has it.** The L3 predicted window is unchanged from `an_s0`'s (reach still
sits under the metered budget there); the L4@sup column is where the meter bites, and at the
banked δ = 0.44 the oracle's predicted L4@sup falls 67.3 → 59.4.

**The arms**: `q_exo, q_bisect, q_endo, q_novel, q_comp, q_trust` (`ANTIPHON_M_ARMS`) — the
`an_s0` schedule-paced core, **minus `q_comp_free`** (its d\* deviation is its definition and
the quota is exactly what the regime tests) and **plus `q_trust`**. No pacer: one knob moves.
201 cycles at the caps, lifetime-identical by construction, K = 2048, `--ref-tag an_s0`.

---

**Delivered dose ≠ design dose** is a first-class instrument, not a caveat: every cycle logs the
*designed* key of each posed instance beside the key the miner actually recorded, so
P(delivered = designed) is measured per era per level, in every arm. The smoke measures it
before the main run, and it is what turns Phase 0's δ prior into a number.

---

## 4. Code files

| file | purpose |
|---|---|
| `questions.py` | **The selectors and the quota**, out of the Modal app so every rule is auditable with no GPU (`conductor/policy.py`'s pattern). `target_geometry` / `half_keys` (which part of the mined span the damage cannot touch), `quota_of` / `check_quota`, the `_round_robin` selection kernel (breadth before depth, fully deterministic, no RNG), `DeliveryLedger`, the six selectors, and the offline gate suite Q-1…Q-10. |
| `antiphon.py` | The substrate. Forks `../crescendo/crescendo.py`; every addition marked `# [antiphon]`. Entrypoints: `preflight` (the donor's), `fidelity_smoke` (G-F + gate Q-11), `antiphon_run`. |
| `analyze_antiphon.py` | The reduction — §0 fidelity, §1 controls, §2 dose, §3 climb (cycle clock *and* priced clock), §4 tables + the r² wall, §5 climbing speed, §6 value, §7 nerdsnipe, §8 trust, §9 `ess_use`. Four figures. **[s2]** `--merge-tag` takes a comma-separated LIST (one tag still works, so `an_s0 --merge-tag an_s1` reproduces exactly), and §0 gained the **sibling twin window** (`LOOP_SIBLING`). Every other section is arm-list agnostic — it reads `summary["order"]` — so the new arms need no further wiring. |
| `launch_detached.py` | Session-isolated detached launcher (the donor's, retargeted). |
| `phase0_question.py` | **The offline phase.** The DGP's parse ambiguity at each mining node, the true-key marginals, the random-draw model calibrated against the donors' logs, menu reach vs K, the r² payoff curve, the predicted windows, the d\* alphabet and the menu's wall-clock cost. Writes `phase0.json`. No GPU, no Modal, no substrate. |
| `phase0.json` | Phase 0's output. |
| `pacer_marginal.py` | The pacer-alone marginal recomputed from the banked `cr3_s0` logs on `an_s0`'s own statistics, so the three lanes sit in one like-for-like table (§6b). Checks the like-for-like precondition rather than assuming it. Writes `pacer_marginal.json`. CPU only. **[s2]** now emits one four-row block **per selector family** (`FAMILIES`: oracle / endo / comfort / novelty), each against the same baseline, plus a per-lane lifetime block (`d cycles` / `d priced` / `read as`) and a §1b deep-era table that puts every family's era-4/5 cells in one place. A family whose arms are not on disk is skipped, so `an_s0`+`an_s1` alone reproduce the old table unchanged. |
| `pacer_marginal.json` | Its output. |
| `phase0_metered.py` | **[m] Phase 0 for the strongly-metered round**, §3a's whole table. Reads `an_s0`'s banked per-cycle logs and `phase0.json`; measures the abundance regime, predicts the metered one as an upper bound (solve fraction held fixed), recomputes the per-era observation budget, and re-runs the reach-vs-budget and predicted-window tables at that budget. Writes `phase0_metered.json`. No GPU, no Modal, no new runs. |
| `phase0_metered.json` | Its output. |
| `metered_premium.py` | **[m] The premium table.** Each selector minus `q_exo` *within its own tag*, on L4 true keys at support and on eras-4/5 recovered fraction, with the metered point beside the abundance point — "does the port's premium rise when questions are scarce". Asserts the two tags trained the same substrate (so §6's normalisation is shared), prints the measured regime of each, and builds the metered column's in-tag handle from the **metered null's own displacement structure** rather than transferring `crescendo`'s abundance-measured floors. `an_s0`'s regime row is backfilled from `phase0_metered.json` and labelled as such. Writes `metered_premium.json`. CPU only. |
| `metered_premium.json` | Its output. |

`policy.py` and `floors.json` are **deliberately absent**: the rule and the dead zones are A1's,
reached by importing `rhm.practice.maestro.policy`, exactly as in the donor. The 20 offline
policy gates (P-1…P-12, L-1…L-8) therefore apply unchanged and are re-run for this node.

## 5. Gates

| gate | what | where | status |
|---|---|---|---|
| P-1…P-12, L-1…L-8 | A1's/A2's offline policy suite against the imported module | `maestro.policy.policy_gate()` | **ALL PASS** (20/20) |
| **Q-1…Q-10** | the port's offline suite: containment (an agent bundle carries no truth object), `exo` IS the head in order, every pinned selector reproduces the quota bin-for-bin, `comp_free` is quota-free and demonstrably deviates, determinism, breadth-over-depth, the oracle never targets junk while needy true keys are in-stratum, the ledger discriminates a half-key that lands from one that never does, the span geometry, and the empty-clean-era fall-through | `questions.py::question_gate` | **ALL PASS** (10/10) |
| **G-F** | with the question knob off (`question_mode=None`) this fork replays **`crescendo.py`** in process | `antiphon.py::fidelity_smoke` (`an_gf`) | **PASS — 0.000e+00** on `anchor_long` and `given_c1`, against a 0.000e+00 donor self-replay control; commits equal |
| **Q-11** | every selector, every era, **on the real substrate**: the reader forward, the value forward, the exact-feature parse, the operative-lower-table lookup, the quota, and that the ceiling out-targets the head on distinct needy true keys | `antiphon.py::q_interface_check`, inside `an_gf` | **PASS** (18 cells). Distinct needy true keys aimed at, `exo`→`bisect`: **7→10** (L1n25→L2), **8→18** (L2n12→L3), **1→9** (L3n6→L4) |
| **cross-tag (full scale)** | **`an_s0/q_exo` vs `cr3_s0/anchor_long`, all 201 cycles, every logged series** — the menu machinery, the reader forward over 2048 candidates and the value forward over 2048 candidates are inert on the arm that takes the menu's head. **The round's load-bearing gate**, and free | `analyze_antiphon.py` §0 | **PASS — 0.000e+00** on all 13 series over all 201 cycles; commit events identical. Substrate identity confirmed independently at setup (`read_acc` 1.0000, stale buffer 0.1269 / 0.154052734375, all exact) |
| **cross-tag → IN-TAG certification (full scale)** | **`an_s1/outer_yield_m4` vs `cr3_s0/outer_yield_m4`** — with `question_mode` unset, does the fork reproduce the donor's **thermostat** path over its whole trajectory? (The `an_gf` gate covered `anchor_long` and `given_c1` only, so the loop path was certified structurally but not measured.) | `analyze_antiphon.py` §0b | **PASS — 0.000e+00** on all 13 series, **lifetimes 162 vs 162**, commit events identical. This is what promotes §6b's pacer-alone column from a cross-tag borrow to an in-tag column |
| **in-tag twin (donor's own)** | `an_s1/outer_yield_m4` is bit-identical to the schedule arm until its own first loop action | §0 twin window | **c18** — exactly the anchor's L2 commit cycle, i.e. the loop arm diverges at its first action and not before |
| **[m] G-F at the metered knob** | the fork must still replay `crescendo.py` in process **at the knob the metered round runs** — the port-off path is not knob-dependent by inspection, but the round's whole contrast rests on it, so it is measured rather than argued. `fidelity_smoke` gained optional `n_pr` / `pr_width` overrides whose defaults are the values it already used, so `an_gf` reproduces bit-for-bit | `antiphon.py::fidelity_smoke --n-pr 24 --pr-width 8` (`an_gf_m`) | see §6 |
| **[m] regime certification** | per arm, in-run and written into `summary["arms"][arm]["regime"]`: `n_pr`, `pr_width`, solves/cycle, solves ÷ `mine_cap`, cap-binding fraction, obs/cycle, π rows/cycle, π buffer fill and the cycle it filled. Read off values the arm already logs, so it is free and is written in **every** tag — in an abundance tag it is the reference column. `intonation` Round A's discipline: the regime is a measured property of the run, not a knob setting | `antiphon.py`, end of each arm (`[regime]` line) | see §6 |
| **[s2] SIBLING twin** | the gate for a loop-paced *question* arm. Its `q_exo` window is uninformative (the port acts on c1 by design), so the statement that **the pacer is the only thing that moved** has to be made against the arm's schedule-paced sibling: same selector, same stream key, `commit` the only difference — therefore bit-identical until the first action either arm takes. `loop_actions` is not carried in the final `results.json`, so the action clock is read off the commit/advance **events** both arms write | `analyze_antiphon.py` §0 (`LOOP_SIBLING`) | retro-checked on the banked headline arm: `q_bisect_loop` vs `q_bisect` first divergence **c19**, first action either arm c19 (sched c19 / loop c32) — **PASS**. `an_s2`'s three arms carry the same gate |
| **quota** | every arm's per-cycle d\* histogram bin-for-bin equal to the cycle's quota (and `q_comp_free`'s deviation reported as a number) | `analyze_antiphon.py` §1 | **PASS — 1.000** in all six pinned arms (201/201 cycles each). Mean selected d\* is **2.752** in every pinned arm against a menu mean of 2.757 (dev −0.005). `q_comp_free`: quota_ok **0.025**, d\* 2.392, **dev −0.364** — the designed deviation, measured |
| **volume** | mined count == `mine_cap` = 8 in every cycle of every arm | in-run print `[q!] VOLUME SHORTFALL` + §1 | **0.955–0.985** across arms (`q_bisect` lowest at 0.955, `q_endo` highest at 0.985; `q_exo`, which selects nothing, 0.975). Total mined 1585–1601 of a possible 1608 in the six 201-cycle arms — i.e. the spread is ~1% and the incumbent sits inside it, so no arm starved its solve rate |
| **priced budget** | cumulative groundings comparable across arms | §1 `t_cum` | **matched to 0.02%**: 50,687,475–50,700,109 across the six 201-cycle arms. Cycle cost 10.1–10.3 s in every arm |
| **dose** | P(delivered key == designed key) per era per arm | in-run, every cycle; §2 | **measured, non-null**: pooled 0.165–0.222 across arms, era 1 **0.37–0.54**, era 2 **0.14–0.18**, era 3 **0.03–0.05**, ~0 in the consumption eras (where the cell swallows the span and there is no clean half). Phase 0's calibrated prior was δ ≈ 0.44; the realised value is that at era 1 and falls with span length |
| N-1, N-2, F | A1's endo label-freeness, endo price, floor gate | as donor | as donor |

**RNG isolation, three ways** (the `woodshed` `rehrng` discipline, since the cross-tag replay is
the round): the port draws on its own numpy streams (`qrng`, and `context_instances`'s own
`default_rng` for the menu tail); every torch read it takes is a `no_grad` forward on a
dropout-free net, so it draws nothing; and the whole block is wrapped in
`_rng_snapshot`/`_rng_restore`, so even a future edit that *does* draw cannot move the shared
position. `with_clean=True` changes `context_instances`'s return value and nothing else.

**One measured fact that sharpens the arms**: `read_acc = 1.0` on this substrate — the reader
reproduces the exact inverse map. So the junk keys of §2(a) are the *grammar's* ambiguity, not
learner error; the clean half of every key is delivered exactly; and `q_endo`'s half-key is the
same object the oracle sees, minus the other half and minus the truth table. The endogenous arm
is therefore a tight proxy for the ceiling, not a degraded one.

## 6. Runs

| tag | what | outcome |
|---|---|---|
| `an_gf` | G-F in-process fork-vs-`crescendo.py` replay with the knob off, **plus gate Q-11** | **PASS** (see §5). First attempt tripped Q-11's own sanity assert at L2 — the assert compared *instance* counts on a 14-key space where a uniform draw already saturates; corrected to the breadth statistic (distinct needy true keys) and asserted only where Phase 0 says there is headroom (L3/L4). The fork comparison was 0.000e+00 in both attempts |
| `an_smoke` | seven arms end-to-end at `--quick`, `--question-k 512` | the port wires end-to-end; `q_exo`/`q_bisect`/`q_endo` clean. **The dose read is NOT informative at this scale**: at `--quick` the substrate is barely trained and the solve rate is ~1/24, so `mine_cap` is never filled (`[q!] VOLUME SHORTFALL: mined 1 < 8`) and `dose=0.0` on one sample. The delivery-fidelity measurement therefore moves to `an_s0`'s own first cycles, where mining is at cap |
| `an_s0` | the main run: 7 arms — six schedule-paced at 201 cycles, plus the loop-paced `q_bisect_loop` at 180 | **complete**, 2026-09-01, app `ap-QtxyavT89Isn0tmTlkMOML`. 1386 arm-cycles at 10.1–10.3 s/cycle; **4.12 GPU-h** realized. All gates in §5 pass |
| `an_s1` | the **in-tag pacer-only arm**: `outer_yield_m4`'s policy with `question_mode` unset, one arm, merged into the reduction (`crescendo`'s `cr3_s1` pattern) | **complete**, 2026-09-01, app `ap-OdqgzfzuPUDLuPoPf3zsh8`. 162 cycles at **10.68 s/cycle** (against a 10.2 projection), **0.669 GPU-h** realized against a 0.63 estimate — +6%, all of it the per-cycle rate, since the loop arm carries the recert and the shadow auditions at every rung it holds. `--ref-tag an_s0` asserted substrate identity at setup before any cycle ran. **§0b certification PASS at 0.000e+00** |

| `an_s2_smoke` | the three `[s2]` arms at `--quick --question-k 512 --total-cap 12`: does the port cross with the pacer end-to-end? | **PASS**, 2026-09-01, app `ap-PSB22qk7BCsyfkujFP3uD0`. 335 s / **0.09 GPU-h**; 3 arms × 12 cycles at 6.9–7.1 s/cycle. Policy 21/21, question 10/10, floor gate PASS, RNG-neutrality assert PASS, **0 quota misses**; `ledger=on` in `q_endo_loop` only. No loop *action* at that scale (quick era caps are 7 against `burn + W = 8`), which is expected — the loop path itself is certified at full scale by `an_s1` at 0.000e+00 |
| `an_gf_m` | **[m] G-F re-gated AT THE METERED KNOB** (`fidelity_smoke --n-pr 24 --pr-width 8`) | **PASS — 0.000e+00** on `anchor_long` and `given_c1` against a 0.000e+00 donor self-replay control, commits equal; question gate **13/13** (Q-1…Q-13, the trap lane's two included); **Q-11 in-substrate interface check PASS (21 cells)**, with the ceiling still out-targeting the head on distinct needy true keys at the metered knob: exo→bisect **7→10** (L1n25→L2), **8→18** (L2n12→L3), **1→9** (L3n6→L4) |
| `an_m_probe` | **[m] Phase A**: `q_exo` alone at `--n-pr 24 --pr-width 8`, era 1 only (full 5-era ladder with `--total-cap 60`, so `measure_refs` builds a 5-era refs vector and `--ref-tag an_s0` stays assertable), `--ref-tag an_s0` | **CERTIFIES**, 2026-09-01. 1260 s / **0.35 GPU-h**; 60 cycles at **9.8 s/cycle**. `[ref]` substrate identity to `an_s0` PASSED. Regime measured: **6.08 solves/cycle = 0.76× the cap**, cap binds **23.3%**, obs/cycle **5.72**, π **432 rows/cycle**, buffer **43.2% at c60 and never full**. **L2 committed at c23** (certificate c23, 11 entries, recall 0.571) — the commit-path risk a 1.4× thinner observation stream posed is settled. 46 of 60 cycles below cap, every one with `mined == solved`; **0 quota misses** |
| `an_s2` | **the composed cell for the non-oracle selectors**: `q_endo_loop`, `q_comp_loop`, `q_novel_loop` at 201-cycle caps, `--ref-tag an_s0` | **complete**, 2026-09-01, app `ap-FMPLzOUhaaUly2mfac2R2h`. 450 arm-cycles at 11.0/11.4/11.3 s/cycle, **1.58 GPU-h** realized. `[ref]` substrate identity to `an_s0` **asserted and PASSED** before any cycle (refs identical, `read_acc` 1.0/1.0, stale 0.1269250065088272 / 0.154052734375 exact both sides). Lifetimes 186 / 130 / 134. All §5 gates pass, including the new sibling-twin gate |

| `an_m0` | **[m] the strongly-metered main run**: `q_exo, q_bisect, q_endo, q_trust, q_novel, q_comp` at `--n-pr 24 --pr-width 8`, 201 cycles at the caps, `--ref-tag an_s0` | **complete**, 2026-09-02, app `ap-79ccPMSuKhuT8J9HuUH52c`. 1206 arm-cycles at 10.06–10.33 s/cycle, **12,987 s = 3.61 GPU-h** (against a 3.48 projection off the probe's era-1-only 9.8 s/cycle — the probe underestimated because era 1 runs width 18 pre-commit and width 3 after, while the later eras carry more committed macros). `[ref]` substrate identity to `an_s0` **asserted and PASSED**. Question gate 13/13, trap gate PASS. All six arms 201 cycles, lifetime-identical by construction. Arm order was budget-risk order, deviating from the brief's listed order to put `q_trust` fourth rather than last: it is the one arm with **no abundance twin**, so cutting it would have lost strictly more than cutting either pole. Nothing was cut |

**The delivery-dose gate, at full scale** (`q_exo`, era 1, the arm that does no selection):
P(delivered key == designed key) = **0.33 / 0.63 / 0.50 / 0.50 / 0.29** at c1/c2/c3/c10/c20,
mean ≈ **0.45** — landing on Phase 0's calibrated prior of **δ ≈ 0.44** (§2(c)) rather than near
zero. The lever reaches the miner, so the round is not a `wd_s0`-shaped near-null dose. Read
with §5's `read_acc = 1.0`: the shortfall from 1.0 is not parse error, it is the repair
declining to restore the *damaged* half to the features the question designed — the clean half
is delivered exactly, every time, which is the half the selector controls by construction.

**Three `[q!] VOLUME SHORTFALL` flags fired in `q_exo`** (c1 mined 3, c19 mined 6, c20 mined 7,
each equal to that cycle's solve count). They are on the arm that performs **no selection**, so
they are a property of the shared substrate's warm-up solve rate, not of any selector — which is
what makes the instrument interpretable when it fires on a treated arm later: on `q_exo` it is
the null, and a treated arm's excess over it is the only part that could be selection-starvation.
Era 1's other numbers behave exactly as Phase 0 predicted for L2: `n_designed_banked` reaches
43 of 64 by c3 and `credit` falls to 0.000 by c10, because the 14-key L2 table saturates under a
uniform draw and the question knob has nothing left to buy there. The action is eras 2 and 3.

Volume `rhm-scaling-data:/data/rhm_practice_antiphon/<tag>/`; fetched copies, figures and
`reduction.txt` under `figures/<tag>/`.

**Key format, for the provenance lane**: the delivery ledger's posed and landed keys are plain
flat keys — tuples of level-1 feature ids over a span, exactly `Miner`'s own key format
(`macros.parse_features` output, sliced). The same primitive as the efference→committed-table
reconstruction in `provenance/prov_tag.py`, at a coarser grain (a span rather than an
execution), so the two instruments read as one primitive at two grains with no coordination
beyond this format note.

Volume `rhm-scaling-data:/data/rhm_practice_antiphon/<tag>/`; fetched copies, figures and
`reduction.txt` under `figures/<tag>/`.

**[s2] the volume control, which FIRED — and on exactly one arm.** `mine_cap` = 8 is asserted
per cycle in every arm; a shortfall means the selector starved its own solve rate and is
reported, not absorbed (`wd_s0`'s lesson). Every shortfall in the tag has `mined == solved`,
i.e. the arm mined everything it had. Short cycles / era cycles (mean `n_mined`):

| arm | era 1 | era 2 | era 3 | era 4 | **era 5** |
|---|---|---|---|---|---|
| `q_exo` (the null: no selection) | 3/60 (7.87) | 1/50 (7.94) | 1/70 (7.99) | 0/12 (8.00) | 0/9 (**8.00**) |
| `an_s1/outer_yield_m4` (pacer only) | 1/56 (7.91) | 0/50 (8.00) | 0/35 (8.00) | 1/12 (7.92) | 0/9 (**8.00**) |
| `q_bisect_loop` | 4/60 (7.77) | 0/32 (8.00) | 1/67 (7.96) | 0/12 (8.00) | 0/9 (**8.00**) |
| `q_endo_loop` | 2/58 (7.91) | 1/50 (7.98) | 1/57 (7.95) | 0/12 (8.00) | 0/9 (**8.00**) |
| **`q_comp_loop`** | 4/47 (7.79) | 1/27 (7.93) | 0/41 (8.00) | 1/6 (7.67) | **7/9 (6.22)** |
| `q_novel_loop` | 2/60 (7.92) | 0/33 (8.00) | 4/24 (7.71) | 0/8 (8.00) | 0/9 (**8.00**) |
| `q_endo` / `q_comp` / `q_novel` (schedule) | 2–4/60 | 1/50 | 0/70 | 0–1/12 | 0/9 (**8.00**) |

Every other arm in the tag — schedule or loop, oracle or not — holds 8.00 in eras 4 and 5.
`q_comp_loop` alone falls to **6.22**, mining 56 of a possible 72 era-5 observations (77.8%).
So its era-5 number is not a pure selection contrast: part of it is a volume failure at the
deepest era. Stated where the number is, per §3's rule. Its era-1..3 shortfalls sit inside
`q_exo`'s own null band.

## 6a. Reduction

**[s2]** the full-tag reduction is now
`analyze_antiphon.py --tag an_s0 --merge-tag an_s1,an_s2 --fetch --figures`; its stdout is
banked at `figures/an_s2/reduction.txt` and the extended attribution table at
`figures/an_s2/pacer_marginal.txt` (`figures/an_s0/reduction.txt` is left as the `an_s0`-only
record; `figures/an_s0/reduction.json` is the merged superset).

`analyze_antiphon.py --tag an_s0 --fetch --figures` → `figures/an_s0/reduction.txt`,
`reduction.json`, and four figures: `coverage_vs_cycle.png`, `coverage_vs_priced.png`,
`r2_payoff.png`, `dose_and_junk.png`.

Two reduction-side corrections made after the first pass, recorded so the numbers are not
re-derived from a stale script: §8's trust readout originally looked for a `share` key in
`log["prop"]`, which does not exist — it now reads the per-level share of the priced beam's
entry selections out of `log["entry"]["hist"]["beam"]`, which is the source `crescendo`
finding 3's L4 mass came off. §4's buildable column was labelled "frozen L3" but is computed
over the arm's **end-of-run at-support L3 set** (the live set, not the committed frozen table);
the label now says so.

## 6b. The pacer-alone marginal — **IN-TAG**, certified

`pacer_marginal.py` → `pacer_marginal.json`. CPU only, on banked logs; no GPU, no new runs.

**Why it is needed.** `an_s0` shipped `q_bisect_loop` (the oracle selector on the donor's
thermostat) but **no pacer-only arm**, so the tag could not separate pacer-alone from
selection-alone from their interaction, and the pacer-alone column had to be borrowed from
`cr3_s0`. `an_s1` closes that: it runs `outer_yield_m4`'s policy under `antiphon.py` with
`question_mode` unset, and **§0b certifies it replays `cr3_s0/outer_yield_m4` at 0.000e+00 over
all 13 series and both 162-cycle lifetimes, commits identical**. The lane is now in-tag. This
file recomputes the column on **exactly** the statistics `analyze_antiphon.py` reports (§6's
recovered fraction, §8's L4 beam-entry share, §4/§5's commits and books, §3's end-of-run
recalls), so the three lanes are like-for-like rather than eyeballed across differently-computed
numbers, and it **auto-promotes**: it swaps to the in-tag arm and relabels the column only when
the certification passes, and falls back to the labelled cross-tag borrow when it does not.

**What the certification bought, precisely.** The marginals below are numerically *unchanged*
from the cross-tag version — necessarily, since the two trajectories are bit-identical. What
moved is the licence: the interaction rows are an in-tag reading rather than a cross-tag one.
The certification promoted the label, not the numbers, and that is the honest way to state it.

**What makes the borrow unusually tight** (checked, not assumed): the two tags trained an
identical substrate (`read_acc` 1.0/1.0, both stale-buffer numbers exact, `refs` identical), and
**the baseline of every lane is one and the same trajectory** — `an_s0/q_exo` replays
`cr3_s0/anchor_long` at 0.000e+00 over all 201 cycles and 13 series, down to an identical
`t_cum` of 50,688,282. So the three deltas below are differences against the *same* baseline.

**The caveat that the certification does NOT retire: the two pacer lanes are not
lifetime-matched.**
Loop arms leave eras early, so against the schedule baseline `outer_yield_m4` runs −39 cycles
(−19.6% priced spend) and `q_bisect_loop` −21 cycles (−11.0%). `crescendo`'s construction makes
the schedule arm the lifetime ceiling, so a *positive* loop delta cannot have been bought with
time — but a *negative* one may be a time cost. Read the pacer-alone negatives with that, and
read the composed positives as conservative.

Baseline for all three lanes = `cr3_s0/anchor_long` ≡ `an_s0/q_exo`.

| lane | rf e1 | rf e2 | rf e3 | **rf e4** | **rf e5** |
|---|---|---|---|---|---|
| pacer-alone **[IN-TAG]** (`an_s1/outer_yield_m4`) | −0.003 | −0.102 | −0.021 | +0.091 | −0.060 |
| selection-alone (`an_s0/q_bisect`) | +0.063 | +0.012 | +0.012 | −0.085 | +0.322 |
| composed (`an_s0/q_bisect_loop`) | +0.074 | −0.054 | +0.129 | **+0.285** | **+0.536** |
| sum of the two singles | +0.059 | −0.090 | −0.010 | +0.006 | +0.262 |
| **composed − that sum** | +0.015 | +0.036 | +0.139 | **+0.279** | **+0.273** |

| lane | L4 beam share e3 | e4 | e5 |
|---|---|---|---|
| pacer-alone **[IN-TAG]** | +0.061 | +0.098 | +0.149 |
| selection-alone | +0.000 | −0.004 | +0.066 |
| composed | **+0.274** | **+0.260** | **+0.263** |

| lane | ΔL2 commit | ΔL3 commit | ΔL4 commit | L4 book n | L4 book precision |
|---|---|---|---|---|---|
| pacer-alone **[IN-TAG]** | +31 | +17 | **−51** | 5 vs 5 | 0.400 vs 0.200 |
| selection-alone | +1 | +7 | +0 | 7 vs 5 | 0.714 vs 0.200 |
| composed | +14 | +8 | **−59** | 7 vs 5 | 0.429 vs 0.200 |

| lane | Δ L2 recall | Δ L3 recall | Δ L4 recall | Δ L4 true@support |
|---|---|---|---|---|
| pacer-alone **[IN-TAG]** | −0.071 | −0.071 | +0.000 | +0 |
| selection-alone | +0.071 | +0.000 | +0.034 | **+28** |
| composed | +0.071 | +0.018 | +0.012 | +10 |

Lifetime / priced spend against the baseline: pacer-alone −39 cycles / −19.6%; selection-alone
0 / +0.0%; composed −21 cycles / −11.0%.

### 6b-bis. [s2] the same table for the three NON-ORACLE selectors

`an_s2` crosses the same `_Q_LOOP` pacer with the three selectors that read only the learner's
own state, so the composed cell is asked of the object the round is about. Same baseline
(`an_s0/q_exo` ≡ `cr3_s0/anchor_long`), same pacer-alone lane (`an_s1`, in-tag, certified),
same statistics. `pacer_marginal.py` emits one block per family; the deep-era rows:

| family | lane | rf e4 | rf e5 | Δ cycles | Δ priced | read as |
|---|---|---|---|---|---|---|
| *(all)* | pacer-alone **[IN-TAG]** | +0.091 | −0.060 | −39 | −19.6% | short |
| oracle | selection-alone | −0.085 | +0.322 | 0 | +0.0% | matched |
| oracle | composed | **+0.285** | **+0.536** | −21 | −11.0% | short |
| oracle | additive | +0.006 | +0.262 | | | |
| oracle | **interaction** | **+0.279** | **+0.273** | | | |
| **ENDO** | selection-alone | +0.130 | +0.716 | 0 | +0.0% | matched |
| **ENDO** | composed | **+0.179** | **+0.885** | −15 | −7.7% | short |
| **ENDO** | additive | +0.221 | +0.656 | | | |
| **ENDO** | **interaction** | **−0.042** | **+0.230** | | | |
| comfort | selection-alone | +0.097 | −0.290 | 0 | +0.0% | matched |
| comfort | composed | −0.309 | −0.601 | −71 | −35.5% | short |
| comfort | additive | +0.188 | −0.350 | | | |
| comfort | **interaction** | **−0.497** | **−0.251** | | | |
| novelty | selection-alone | −0.055 | +0.033 | 0 | +0.0% | matched |
| novelty | composed | −0.224 | −0.087 | −67 | −34.8% | short |
| novelty | additive | +0.036 | −0.027 | | | |
| novelty | **interaction** | **−0.261** | **−0.060** | | | |

"short" = the lane leaves eras early, so its **positives are conservative** and its
**negatives are suspect toward time cost** — the §6b(b) caveat, now printed per lane rather
than left as prose. Read the comfort and novelty composed negatives with that, and note
separately that `q_comp_loop`'s era-5 cell also carries the volume failure recorded in §6.

π's L4 beam share (§8's instrument), Δ to the same baseline:

| family | e3 | e4 | e5 |
|---|---|---|---|
| pacer-alone **[IN-TAG]** | +0.061 | +0.098 | +0.149 |
| oracle: selection-alone / composed | +0.000 / **+0.274** | −0.004 / **+0.260** | +0.066 / **+0.263** |
| **ENDO**: selection-alone / composed | +0.000 / **+0.259** | +0.016 / **+0.256** | +0.080 / **+0.292** |
| comfort: selection-alone / composed | +0.000 / +0.053 | −0.002 / −0.019 | +0.045 / +0.001 |
| novelty: selection-alone / composed | +0.000 / +0.080 | −0.021 / +0.112 | −0.004 / +0.236 |

Commits (Δ to baseline) and the L4 book: ENDO composed L2 +28 / L3 +2 / **L4 −54**, book 3
entries at precision 0.333; comfort composed L4 −82, book 1 at 1.000; novelty composed L4 −74,
book 2 at 0.500. Oracle composed was L4 −59, book 7 at 0.429.

### The recommendation, and what happened to it

Recorded because the round's own reasoning is part of the record. Two options were put up: (i)
extend the `an_gf` smoke gate to `outer_yield_m4` (≈0.25 GPU-h), which would certify the
thermostat path over 12 cycles; (ii) run the in-tag pacer-only arm (≈0.63 GPU-h), which would
certify it over its whole 162-cycle trajectory *and* remove the cross-tag label rather than
licensing it. **(ii) was authorized and run as `an_s1` (0.669 GPU-h realized) and passed at
0.000e+00.** The argument for preferring it over the cheaper option held: a 12-cycle smoke would
have said very little about a 162-cycle trajectory, and it would have left the column labelled.

The original recommendation text is kept below as written, unedited.

### Recommendation on certifying the borrow (as written before the run; recommend-only)

The gap the review names could be closed two ways, and they are not equally good.

**(i) The literal ask — extend the G-F gate to `outer_yield_m4`.** `fidelity_smoke`'s arm list
is hardcoded to `("anchor_long", "given_c1")`; adding a third arm means three more 12-cycle
runs (fork, donor, donor-self-replay) on a setup that is already paid inside that entrypoint.
A re-run of the whole gate with three arms costs **≈0.25 GPU-h**; the setup (~10 min) dominates,
so the marginal arm is ~4–5 min. It would certify that with `question_mode` off this fork
reproduces the donor's thermostat path bit-for-bit.

**Would I trust the column without it? For the narrow question, yes.** The fork's entire
behavioural diff inside `run_arm` is five things — `with_clean=True` on one call's return, the
`if qmode:` block, the mine-index capture, the dose block, and two log keys — and none of them
is on the loop's code path. `loop.kind == "quiet"` is donor code reached identically; the port's
block sits inside an `_rng_snapshot`/`_rng_restore` sandbox, and that transparency is already
measured at 0.000e+00 on two arms with different vocabulary policies (`anchor_long` earned,
`given_c1` true). What a 12-cycle smoke on a third arm would add is confirmation of something
the existing gate already covers structurally.

**(ii) What I would actually spend the compute on instead — an in-tag pacer-only arm.** Run
`outer_yield_m4` with `question_mode` unset as its own tag (`an_s1 --arms outer_yield_m4`,
merged in the reduction the way `crescendo` merges `cr3_s1`; running it under `--tag an_s0`
would overwrite that tag's `summary.json`). One arm, ~162 cycles at 10.2 s/cycle plus setup ≈
**0.63 GPU-h**. It is strictly stronger than (i) on every axis: it removes the cross-tag label
from the column entirely rather than licensing it, it *is* a full-scale 162-cycle bit-identity
certification against `cr3_s0/outer_yield_m4` (against which a 12-cycle smoke says very little),
and it makes the interaction row an in-tag claim. It still does not fix the lifetime caveat,
which is structural to loop arms and not a gating question.

So: **(i) is cheap and I do not think it is needed; (ii) costs 2.5× as much and is the one that
actually closes the gap the review identified.** Neither is run pending your and Jasper's call.

## 7. Budget

The donor's measured rate is **11.25 s/cycle** (`cr3_s0`: 851 cycles in 2.66 GPU-h). `an_s0` runs
six schedule-paced arms at 201 cycles plus one loop-paced arm (≤201, ~160 expected) ≈ **1366
cycles**, and the compute-matched menu reads add ~13% → **≈ 4.8 GPU-h**. With `an_gf` (×2, ~0.22
each) and `an_smoke` (~0.75) the tag projects to **≈ 6.0 GPU-h** against the authorized ≈5.0.
Three stated reasons: the second `an_gf` was a re-run of my own Q-11 assert bug; the pacer cell
and the unpinned comfort arm were added after the 3.8 GPU-h design estimate; and 0.6 GPU-h/arm
was read off `crescendo`'s *average* including its shorter loop arms rather than its 201-cycle
arms (0.63).

**The run order is the budget-risk order**, which is what makes an overrun cheap to resolve:
`summary.json` is rewritten after every arm, so the last arm is always the cuttable one. In the
event nothing had to be cut — **the projection was pessimistic**.

**Realized, per tag:**

| tag | realized | note |
|---|---|---|
| `an_gf` (×2) | ≈0.45 | wall-clock estimate; `gate.json` was not fetched. The second run was mine to pay — a Q-11 assert bug, not a fork problem |
| `an_smoke` | **0.749** | 7 arms at `--quick`; its dose read turned out uninformative at that scale (§6) |
| `an_s0` | **4.121** | 1386 arm-cycles at 10.1–10.3 s/cycle |
| `an_s1` | **0.669** | 162 cycles at 10.68 s/cycle, against a 0.63 estimate (+6%) |
| `an_s2_smoke` | **0.09** | 3 arms at `--quick --total-cap 12`, 335 s |
| `an_s2` | **1.58** | 450 arm-cycles at 11.0–11.4 s/cycle; lifetimes 186/130/134 against a 160–201 projection |
| `an_gf_m` | **≈0.15** | wall-clock; the metered G-F + Q-1…Q-13 + Q-11 |
| `an_m_probe` | **0.35** | 60 cycles at 9.8 s/cycle + the 609 s setup |
| `an_m0` | **3.61** | 1206 arm-cycles at 10.06–10.33 s/cycle |
| **total** | **≈11.62** | `an_s0`-round ≈5.99 + `[s2]` **1.67** (auth. 1.3–2.0) + `[m]` **4.11** (auth. ≈2–2.5; the overrun is the per-cycle rate, not extra arms — see below) |

**[m] why 4.11 and not the ≈2–2.5 estimated.** The meter shrinks the practice beam to
(24×8)/(64×16) = **0.19×** of its work, but the **menu reads are unchanged** — the reader and
value forwards over all 2048 candidates run in every arm by §3's compute pin — and so are the
auditions, probes, recerts and π training. Those fixed costs dominate, so the per-cycle rate
fell only 10.2 → 10.2 s/cycle at full-run scale (the probe's era-1 slice read 9.8). A metered
round on this stack is therefore **not** cheaper per cycle in proportion to its meter; it is
cheaper in *observations*, which is the point. Recorded so the next metered round is projected
off 10.2 s/cycle rather than off the knob ratio.

**[s2] why 1.58 and not the 1.90 projected at halt 1.** The per-cycle rate ran hot as expected
(11.0–11.4 s/cycle against 10.2–10.7, the same recert-and-shadow-audition cost `an_s1` paid),
but the two pole arms **left their eras early** — `q_comp_loop` at 130 cycles and `q_novel_loop`
at 134 against the 186 the cell arm ran — so the tag came in under a projection built on
`q_bisect_loop`'s 180. Nothing was cut; `q_novel_loop` ran in full.

The main run came in **0.7 GPU-h under its own projection** because the realized rate was
10.1–10.3 s/cycle against the 11.25 s/cycle read off `crescendo`'s average — i.e. the +13% I
budgeted for the compute-matched menu reads over 2048 candidates did not materialise, and the
menu reads are cheaper than the ~13% estimate. The one arm that ran *slower* than projection is
`an_s1`'s loop arm at 10.68 s/cycle, which is the pacer's own recert and shadow-audition cost at
every rung it holds, not the port's.

Single seed; ranks, signs and multiples of measured floors are the claims.

## Children

| folder | what |
|---|---|
| [`trap/`](trap/DESIGN.md) | **Worthless answers installed in the question menu (`tr_gf`, `tr_smoke`, `tr_s0`).** `phase0_trap.py` sizes the premise offline (the foreign-grammar channel is impossible on a world-typed action space with success-filtered mining; the native atom is a candidate whose clean half parses to a junk key — non-convertibility is exact; the world serves 54.6% such at era 3; the breadth-first pool-ratio bound 0.686; two guards rejected offline, the use record promoted). `trap_menu.py`: the per-era distractor bank, the substitution, `trust_weights`, gate suite T-3. `analyze_trap.py`: §0 era-1 twins, dose, climb split true/junk, committed-table and use-weighted precision, the premium vs the native dose, the take series. Five arms at f = 0.90. Record: `DESIGN.md` (design + offline sizing, updated with the staged-code status) and `FILES.md`. Results are written up in the round's joint writeup, [`../tutti/README.md`](../tutti/README.md) finding 3. |

**Extensions after the README** (`an_s2`, the loop-paced judge and poles; `an_m0`, the port at
the strongly-metered knob) are recorded in §3/§3a/§6 above and written up in
[`../tutti/README.md`](../tutti/README.md) findings 2 and 5.

[^private]: Not mirrored: this link points to a document in the private lab repo (the roadmap, the queue, an unrun spec, reading notes, or a conversation). See the top-level README for what is held back and why.
