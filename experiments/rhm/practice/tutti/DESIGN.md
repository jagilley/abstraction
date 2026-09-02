# DESIGN — `tutti` (the unification node: both currencies, and the selector, in one loop)

Design step only. **Nothing is built and no GPU is spent until this is reviewed.** No results are
interpreted anywhere in this node (repo norm); there is deliberately no `README.md`.

**Up**: parent arc [`../README.md`](../README.md) ·
`../../../../QUEUE.md`[^private] "Critical path" → *the unification node* ·
`../../../../ROADMAP.md`[^private] §7.2.2 / §7.2.3 (the two seats; the
currency-vs-level sharpening)
**Direct donor** (untouched): [`../caesura/`](../caesura/FILES.md) — E′ round 3. `caesura.py` is
forked here. **Imported, not forked**: `maestro/policy.py` (A1's rule and dead zones),
`antiphon/questions.py` (the selectors and the quota), `native/span/span_net.py` (through the
donor), `crescendo/phase0_l4.py` (through the reduction).
**Through the donor**: `../intonation/` (δ_perf, the fallible executor) · `../tacet/` ·
`../crescendo/` (A3, the yoke idiom, the floors) · `../maestro/` · `../conductor/`.
**Sibling lane**: `sizing/` (the L5 offline sizing) — **written by another lane; not touched here.**

---

## 1. The question, and what the node has to separate

The loop has three actions: **commit** level ℓ, **advance** the era, **select** the questions.
Two currencies can license them:

- the **outcome** currency — one-level-up `at_support` yield, A1's thermostat (`crescendo`, A3);
- the **execution** currency — δ-silence, `dsil` = mean `b(s)` over open slots, read through A1's
  `QuietPolicy` unchanged (`caesura`), live only because `intonation` un-gated the span head
  below parity so the executor is fallible.

`two_deltas` (b) asks whether the within-level type law is **currency-scoped rather than
level-scoped**. `caesura` finding 5 is the only positive evidence, single-seed, un-yoked, and
governed by a floor 1.43–1.53× tighter than its own in-tag noise. `antiphon` adds a third lever
whose deep-era value was ≈ **fully interaction** with the pacer. So the node asks:

1. **Do the two currencies divide labor or interfere** across the loop's actions — (a) δ-silence
   owns both, (b) yield owns both, (c) the **split**: δ-silence paces commit, yield owns era
   advance (the QUEUE's stated shape, unmeasured)?
2. **Does `caesura` finding 5 survive re-instantiation at a new fork** — a different file, a
   corrected dead zone, and a world in which the selector is live?
3. **Does the selector's value depend on who paces** — the interaction, one level up from
   `antiphon`'s?

---

## 2. The fork choice: `caesura.py → tutti.py`, grafting the `# [antiphon]` port

Three candidates were considered.

| route | diff to carry | verdict |
|---|---|---|
| **(A) fork `caesura.py`, graft the port** | **571** lines (crescendo→antiphon), of which ~110 are one self-contained function | **chosen** |
| (B) fork `antiphon.py`, graft δ-silence + the meter + the gate chain | **2303** lines (crescendo→caesura) | 4× the diff, and it re-implements three certified forks |
| (C) fork `crescendo.py`, re-apply both | both diffs, and the G-F chain restarts at A3 | loses the certified lineage |

**The collision check the brief asked for was run, and the two forks do not collide.** Verified
against the files:

- `antiphon`'s **one call site** is the `context_instances` draw in `run_arm` block (a).
  `caesura`'s block (a) is **byte-identical to `crescendo`'s** — the donor added nothing there.
- The port's *dependencies* all exist in `caesura` unchanged: `context_instances(...,
  with_clean=True)` (already in `crescendo`), `_rng_snapshot`/`_rng_restore`,
  `nearest_derivation_cost`, `MC.exact_features`, `MC.parse_features`, `_encode_chunked`,
  `operative`, `miners`, `shared["inverse_maps"]`, `shared["truth"]`.
- The **one real interleave** is block (b)'s mining index. `antiphon` added a parallel
  `_solved_idx`/`_mine_idx` pair; `caesura` (via `tacet`+`intonation`) already computes exactly
  that object as `si = np.flatnonzero(ps > 0.5)` and `mine_take`. The graft therefore **reuses
  the donor's index** (`_mine_idx = si[mine_take]`, with the donor's own length guard) instead of
  adding a second one — one line, not a merge.
- Everything else is additive and textual: module docstring, `app`/`REMOTE`, two imports
  (`collections`, `questions as QS`), the `ARMS`/`STREAM` rows, the `log` dict's `"q"` key, the
  `_cfg` knobs, the entrypoint signature, `fidelity_smoke`, `preflight`.
- One graft risk was checked and cleared: `pose_questions` calls `value_net.eval()` /
  `controller.eval()`, which persist. Every trainer in `caesura.py` sets `.train()` at entry
  (`value` 1517, `generator` 1550/1608, `head` 1610, `prop` 1859, `reader` 1953), so nothing
  leaks — and the full-scale replay gate (§6) proves it rather than asserting it.

`questions.py` is **imported from `antiphon/`, not forked**, exactly as `policy.py` is imported
from `maestro/`. That keeps gates Q-1…Q-10 unchanged and is the structural form of the claim:
*the selector rule did not change; the loop did.*

**Files**: `tutti.py` (fork of `caesura.py`, additions marked `# [tutti]`),
`analyze_tutti.py` (fork of `analyze_caesura.py`), `launch_detached.py` (retargeted),
`phase0_tutti.py` + `phase0.json` (offline, §5). No `policy.py`, no `floors.json`, no
`questions.py`, no `phase0_l4.py` — all reached by import.

---

## 3. The one new mechanism: a second policy object for the commit

`QuietPolicy` is already a pure, self-contained object with its own buffer, latch and `acted()`
clock, and `caesura` already builds a **second** one (`dsil_det`, the veto's read-only detector)
in every arm. The split is that detector promoted from read-only to licensing.

```
spec["loop"]         the arm's primary policy — owns ADVANCE, and owns COMMIT unless:
spec["loop_commit"]  optional second policy — owns COMMIT   # [tutti]
```

Both are built from the same `span/W/burn/alpha` and the same `loop_floors` (which already
carries `dsil`, `yield_by_level`, `ledger`, `endo`). Pinned decisions, each stated because each
could have gone the other way:

1. **Precedence is the donor's, unchanged.** A quiet commit-latch COMMITS the active level if
   there is one to commit; a quiet advance-latch ADVANCES otherwise. If both are quiet on one
   cycle, **commit wins and the advance is not taken that cycle** — one cycle can never fire
   both, exactly as in `conductor`.
2. **Both policies reset on every action and at every era start** — not only on their own. The
   donor's stated reason for resetting is *regime change*, and a commit installs a table (a
   regime change for the yield gauge) while an advance changes the damage cell (a regime change
   for the executor δ-silence hears). Resetting on own-action-only would make the split differ
   from (a)/(b) in **two** things (which gauge licenses, and how each clock re-arms) instead of
   one. The own-action-only rule is a *counterfactual* readout in the reduction (§7 replays each
   policy's own logged trace under it), never a realised arm.
3. **The bootstrap carries over verbatim.** δ-silence cannot license the first crossing (slots
   are minted by commits — `caesura` gate D). Where `loop_commit.read_key == "dsil"` and the
   gauge is absent, the commit falls back to the arc's default rule (`delta_prov`) and is logged
   `kind: "bootstrap"` so the reduction never counts it as a δ-silence firing. In the split this
   produces a genuinely new configuration — **bootstrapped commits alongside yield-driven era
   advances in era 1** — and gate X-3 asserts that is what the arm does.
4. **A cycle whose read is undefined is not a decision point, per policy.** `loop_commit.step` is
   skipped when `panel["dsil"] is None` (the donor's own idiom); the yield policy is always
   defined.
5. **The yoke generalises for free.** `YokePolicy` replays COMMIT and ADVANCE cycles separately
   from a plan, so a yoke of a split arm needs no new code; `loop_commit` is ignored under
   `kind == "yoke"`. Gate X-5 asserts the replay is exact for a two-policy source.
6. **The ledger charges both policies' needs** (`loop.needs ∪ loop_commit.needs`). Neither
   `yield` nor `dsil` is priced (only `endo` is), so this is a no-op — stated rather than assumed.
7. **Inertness.** With `loop_commit` absent, `loop_c is None` and every line above is the donor's.
   Gate X-2.

The veto (`dsil_det`, `dsil_veto`) is **untouched and orthogonal** — it composes with either
assignment and is not run this round (§4).

---

## 4. Arms — six paid, one seed

Every arm runs with the port **on** (`question_mode` set), so the selection compute — the reader
forward and the value forward over all `K = 2048` menu candidates — is taken by **every** arm and
used only by the arms whose selector reads it (`ostinato`'s rung-invariant-comparator
discipline). `question_mode=None` is used only inside G-F. `exo` = the menu's head in the donor's
own RNG order, i.e. *no selection*; `endo` = half-key novelty vs the arm's own at-support set ×
the posed-vs-landed delivery ledger — the only selector that can be an outer-loop **action**.

| # | arm | commit | advance | selector | cycles (est.) | GPU-h | what it pins |
|---|---|---|---|---|---|---|---|
| 1 | `tu_y_exo` | yield | yield | exo | 131 | 0.58 | **(b)** + the round's load-bearing **full-scale cross-tag replay** of `ca_s0/dsil_yield`, and the (yield, no-selection) baseline cell |
| 2 | `tu_d_exo` | dsil | dsil | exo | ≤176 | 0.77 | **(a)** re-instantiated at the corrected floor — `caesura` finding 5's verification cell; bounded cross-tag gate to **c42** (§5) |
| 3 | `tu_s_exo` | **dsil** | **yield** | exo | 131–176 | 0.67 | **(c) the QUEUE's cell** — the division of labor, un-confounded by the selector |
| 4 | `tu_s_endo` | dsil | yield | **endo** | 131–176 | 0.67 | the composition: both currencies *and* the selector in one loop |
| 5 | `tu_s_yk` | *yoke of #4* | *yoke of #4* | exo | = #4 | 0.67 | **the one-bit clock yoke** — lifetime-identical to #4, differing in the selector alone (`crescendo`'s `ceiling_m3` idiom). Separates selection *content* from the clock selection induces |
| 6 | `tu_d_endo` | dsil | dsil | endo | ≤176 | 0.77 | the selector under a **pure execution-currency** pacer |
| 7 | `tu_y_endo` | yield | yield | endo | 131 | 0.58 | the selector under a **pure outcome-currency** pacer — the live-executor twin of the cell `an_s2` is measuring on the exact-DP executor, without which the 3 × 2 has a hole on the one row `antiphon` already measured |
| 8 | `tu_m_exo` | **yield** | **dsil** | exo | ≤176 | 0.77 | **the mirror** — each signal in the other's seat. Separates "this assignment" from "any two-gauge mix beats one", which is a live alternative reading of a split win under §3's simultaneous-reset rule. **Last in run order, i.e. the tail cut if the smoke's s/cycle overruns.** δ-silence cannot speak before a slot opens, so its era-1 *advance* falls through to the cap — and since `CRESCENDO_LADDER ≡ CRESCENDO_CAPS`, the cap is the arc's own advance bootstrap and lands on the schedule arm's own cycle |

**Projected: 5.48 GPU-h** at 15.8 s/cycle (range **4.9–6.1**). Basis: `ca_s0` ran 13.8–15.9
s/cycle with the live executor; `an_s0` measured the port at 10.1–10.3 s/cycle against its
donor's 10.6, so the port is ≲ +0.4 s/cycle and is budgeted at +0.4 (the `tu_smoke` pair
`tu_pf_off` vs `tu_y_exo` measures that marginal directly, one bit apart at K = 2048). Cycle
counts are `ca_s0`'s realised lifetimes; the split arms should sit near the *yield* arm's 131
(yield owns the advance, and δ-silence's long life in `ca_s0` came from δ-silence-driven
advances holding eras to their caps), but are budgeted at the midpoint.

Run order is load-bearing (a yoke must follow its source, and cuts come off the tail):
**1, 2, 3, 4, 5, 6, 7, 8.** Arm 1 first because nothing is readable without a certified
substrate.

**Borrowed cross-tag, no GPU** — licensed by arm 1's in-tag 0.000e+00 replay:
`ca_s0/dsil_sched` (201 cycles at the caps) as the **lifetime ceiling** (no loop arm can outrun
it, which is how `caesura` and A3 buy back the un-yoked pacing comparison), and `ca_s0/dsil_and`
as the veto reference.

*(Approved 2026-09-01: the six-arm core plus R2 `tu_y_endo` and R1 `tu_m_exo`. R3
(`tu_sched_endo`) and R4 (the oracle) are skipped — see "Deliberate exclusions" below, which is
unchanged except that R1/R2 have moved into the run.)*

### What each cut costs

| cut | saves | costs |
|---|---|---|
| drop #8 `tu_m_exo` | 0.77 → **4.71** | a split win could not be separated from "any two-gauge mix beats one". The 3 × 2 is untouched; this is a control on the *interpretation* of #3/#4, not a cell |
| drop #8 and #7 | 1.35 → **4.13** | additionally loses the selector under the pure outcome pacer — the row `an_s2` is measuring on the exact-DP executor, so the two rounds would not be comparable |
| drop #6 `tu_d_endo` | 0.77 → **3.36** | the selector is then measured at **one** pacer only; the third-lever question collapses from "does selection's value depend on who paces" to "what does selection buy under the split". The lifetime-clean selector read (#4 vs #5) and the free-paced one (#3 vs #4) both survive |
| drop #6 and #5 | 1.44 → **2.69** | additionally loses the one-bit yoke, which is the control `antiphon` explicitly named missing ("loop arms are not lifetime-matched; composed −11.0% priced spend"). The selector's value would again be confounded with the clock it induces |
| drop #3 `tu_s_exo` | 0.67 | the split would be measured only with the selector live, so (c) could not be compared to (a)/(b) on the exo row — the division-of-labor question would be answerable only inside the selector's condition. **Do not cut this one** |

### Deliberate exclusions, and what each costs

- **The oracle selector (`bisect`).** Cut. Cost: no in-tag upper bound on what selection could
  buy, so `endo`'s number is un-normalised. Mitigation: the oracle/endo ordering is banked
  (`an_s0`: 50 vs 35 true L4 keys at support), and the oracle is by construction *not* an action
  the learner can take — which is the node's unit of interest. Restoration R4 below.
- **The veto (d).** Cut. Measured and fragile (`caesura`: one one-cycle deferral at an era
  boundary cost a whole rung — one event, not a rate). Cost: no re-instantiation of that
  fragility, and no cell for "δ-silence may only delay" under selection.
- **An in-tag schedule arm.** Cut; borrowed. Cost: there is **no in-tag selection-alone cell**
  (schedule clock, selector on), so `antiphon`'s additive decomposition (*composed − sum of
  singles*) **cannot be reproduced here**. What the round reports instead is the
  **pacer-conditional selector marginal** (endo − exo within each pacer row, both free-paced and
  lifetime-clean) — a sharper form of the same interaction, but not the same statistic.
  Restoration R3 restores it exactly.
- **A second seed.** Per the arc's standing decision (verification by later runs, Jasper
  2026-08-31). Ranks, signs, and multiples of measured floors are the claims.
- **Diet gates.** Not built, per the brief — the seat is closed in both regimes with both δs
  (`tacet`, `intonation`, `ma_s0`). `gate_mode` stays `None` in every arm, so `mine_ord` is
  `None` and the mining draw is the donor's permutation exactly.

### Restorations, priced, in the order I would add them

| | arm | +GPU-h | buys |
|---|---|---|---|
| R1 | `tu_m_exo` — **the mirror split** (yield commits, δ-silence advances), exo | 0.77 | assignment *specificity*: separates "δ-silence belongs on the commit" from "any two-gauge mix beats one". Second-order under §3's simultaneous-reset rule, but it is the control that makes #3's sign interpretable if #3 wins |
| R2 | `tu_y_endo` — selector under the pure outcome pacer | 0.58 | completes the 3 × 2; the nearest re-instantiation of `antiphon`'s composed cell on the live-executor substrate |
| R3 | `tu_sched_endo` — schedule at the caps, selector on | 0.88 | restores `antiphon`'s additive decomposition in-tag against the borrowed `ca_s0/dsil_sched` baseline |
| R4 | `tu_s_bisect` — the oracle selector on the split pacer | 0.67 | the ceiling that normalises the endogenous judge |

---

## 5. Floors — measured in-tag, and the one place that costs a gate

`QuietPolicy` refuses a defaulted dead zone by construction. `tol_yield_l3`/`tol_yield_l4` are
A1's measured values, asserted (unchanged, so arm 1 replays exactly). `tol_dsil` is **re-derived
offline, and the derivation has already been run** (`policy.null_abba`, span 1, W 4,
`v_tol = sd(N)/√W`, skipping commit and advance cycles as regime changes) on `ca_s0`'s own logged
`dsil` series — a pure function of `log["panel"]`, so it costs zero GPU:

| source | n blocks | sd(N) | `v_tol` |
|---|---|---|---|
| `ca_s0/dsil_sched` | 163 | 0.00942 | 0.00471 |
| `ca_s0/dsil_yield` | 43 | 0.00469 | 0.00234 |
| `ca_s0/dsil_read` | 117 | 0.01087 | 0.00544 |
| `ca_s0/dsil_and` | 31 | 0.00529 | 0.00265 |
| **pooled** | **354** | **0.00920** | **0.00460** |

(`caesura` reported 0.00492 on `dsil_sched` alone under a slightly different skip set;
`phase0_tutti.py` will pin the protocol so the two numbers are comparable and print both.)

**APPROVED AND MEASURED — governing value `--tol-dsil 0.0046`** (pooled in-tag: `sd(N)`
0.009187 over 354 blocks, `v_tol` **0.00459346**), against `caesura`'s 0.00321442 — **1.429×
tighter than its own noise warranted**. `phase0_tutti.py` pins the protocol and prints all
three numbers side by side. What it costs and what it buys:

- **What it costs (measured).** `tu_d_exo` is no longer a bit-identical replay of
  `ca_s0/dsil_read`. Replaying A1's rule offline on `dsil_read`'s own logged series at both
  floors puts the **first divergence in the quiet verdict at c43** (6 of 152 cycles differ; its
  realised first δ-silence-driven action was c46; the c18 bootstrap commit is floor-independent
  and replays identically). So arm 2 is a **bounded** cross-tag gate over c1–c42 rather than a
  full-life one, and `analyze_tutti.py` §0T asserts it over exactly that window.
- **What it buys.** Arm 2 becomes a genuine *re-instantiation* of `caesura` finding 5 — same
  gauge, same rule, honest dead zone, new fork — rather than a replay, which is what "does
  finding 5 survive re-instantiation" actually asks. And the round keeps a full-scale gate: arm 1
  is yield-paced, so `tol_dsil` cannot touch it, and it replays `ca_s0/dsil_yield` over its whole
  life on every logged series.
- **The alternative** (govern at `ca_s0`'s 0.00321442, report 0.0046 as the narrower claim) buys
  a second full-scale replay gate and costs the corrected floor — i.e. it repeats `caesura`'s own
  caveat verbatim. Declined.

`K = 2048` is `antiphon`'s budget-matched value and carries here unchanged: the era caps are the
same (60/50/70/12/9), so the per-era observation budgets that sized it are the same.
`phase0_tutti.py` re-runs `phase0_question.py`'s reach-vs-budget table to confirm rather than
assume.

**Phase 0 (offline, no GPU, before anything is built):** (i) the `tol_dsil` table above, pinned;
(ii) the bounded-gate cycle for arm 2; (iii) A1's thermostat replayed on `ca_s0`'s logged `yield`
and `dsil` series to bound *when a split arm would act*, and to check the split cannot deadlock
(the yield advance latch is defined from c1, so era 1 cannot stall while commits bootstrap);
(iv) `phase0_question.py` re-run for reach-vs-budget and the d\* alphabet; (v) `antiphon` gate
Q-9's span geometry re-checked on this ladder (era `L1n25` leaves offset [0] clean, `L2n12`
leaves [2,3], `L3n6` leaves [4,5,6,7]; eras 4–5 leave nothing, so the endogenous half-key signal
exists in the three earning eras and every selector falls through to quota-legal index order in
the last two — stated, logged as `has_clean`). **The r²/L5 sizing is the sibling lane's and is
not duplicated here.**

---

## 6. Gates

Inherited, re-run unchanged: **P-1…P-12, L-1…L-8** (A1's offline policy suite against the
imported module) · **Q-1…Q-10** (`questions.py::question_gate` — containment, quota, determinism,
breadth-over-depth, the ledger, the span geometry) · **C-1…C-5** (A3) · **T-1…T-5** (E3′) ·
**I-0…I-5** (the meter; I-3's non-degenerate 2×2 is the one that must bite at smoke scale) ·
**D-1…D-5** (δ-silence: the gauge present exactly when a slot is open; the driven read; the veto
inert where the signal cannot speak; the schedule arm carries no veto; the bootstrap) ·
**N-1, N-2, F, B-1**.

New, and what each asserts:

| gate | asserts | where |
|---|---|---|
| **G-F** | with every `# [tutti]` knob off (`question_mode=None`, `loop_commit` absent) this fork replays **`caesura.py`** in process at 0.000e+00, against a 0.000e+00 donor self-replay control, commits equal, on both donor arms | `tutti.py::fidelity_smoke` |
| **X-1** | the two policy objects are distinct; the commit licence reads `dsil` and the advance licence reads `yield`, and **neither reads the other's key** | `preflight` |
| **X-2** | **inertness** — with `loop_commit` absent the arm takes the donor's single-latch path bit-identically (0.000e+00 against its own twin) | `preflight` + offline on preflight results |
| **X-3** | the split's bootstrap fires; a slot opens after it; **no bootstrapped commit is taken while the gauge exists**; and era-1 advances are yield-driven while commits are bootstrapped (the new configuration, asserted rather than assumed) | `preflight` |
| **X-4** | **precedence** — on a cycle where both latches are quiet, exactly one action is taken (COMMIT), and **both** policies re-arm; and both re-arm at every era start | `preflight` |
| **X-5** | the yoke of a split arm replays its source's realised COMMIT **and** ADVANCE cycles exactly (T-1's "yoke replays exactly", extended to a two-policy source), and a pure yoke is inert | `preflight` + §D of the reduction |
| **Q-11** | every selector, every era, **on the real substrate**: the reader forward, the value forward, the exact-feature parse, the operative-lower-table lookup, the quota, and that the oracle out-targets the head on distinct needy true keys. Run even though the oracle arm is unpaid — it is the reference the endo selector is scored against | inside the G-F smoke |
| **cross-tag (full scale)** | **`tu_y_exo` ≡ `ca_s0/dsil_yield`**, all cycles, every logged series, commit and advance events identical. **The round's load-bearing gate**, and free — the arm is a cell in the value table regardless | `analyze_tutti.py` §0 |
| **cross-tag (bounded)** | **`tu_d_exo` ≡ `ca_s0/dsil_read` through c42**, the offline-computed first cycle at which the corrected `tol_dsil` changes the quiet verdict | `analyze_tutti.py` §0 |
| **in-tag twin** | every **exo** arm vs arm 1 up to its own first action. Stated limit: an **endo** arm diverges at c1 *by construction* (the selector acts on cycle 1), so the twin gate is exo-family only and the endo arms' lifetime control is the yoke (#5), not a twin window | `analyze_tutti.py` §0 |

**Controls carried from `antiphon`, unchanged**: per-cycle d\* **quota** filled bin-for-bin by
every arm (the deviation reported as a number); **volume** (`mine_cap` asserted equal every cycle,
per arm, with the shortfall flag); **priced budget** (`t_cum` logged per cycle, every comparison
reported at matched cycle *and* at matched cumulative spend); **selection compute** run in every
arm and reported; **oracle containment** (the endo bundle carries no truth table, no exact
inverse map, no clean derivation). **RNG isolation, three ways**: the port's own numpy streams,
no-grad forwards on dropout-free nets, and the whole block wrapped in
`_rng_snapshot`/`_rng_restore` — which is what protects the full-scale replay.

**Delivered-vs-designed dose** stays a first-class instrument, logged every cycle in every arm.
One thing this round measures that `antiphon` could not: the repair is now executed by a
**fallible** span head, so the dose may fall below `an_s0`'s 0.45 at era 1. If it does, the
question knob's grip is narrower here and the reduction says so with a number rather than a
caveat. Not knowable offline; read off `tu_s_endo`'s own first cycles at full scale (a `--quick`
smoke cannot measure it — `an_smoke`'s lesson).

---

## 7. The yoke plan, stated plainly

`caesura` skipped yokes because pacing *was* its variable, and paid the lifetime caveat. Here the
variable set is larger, so the two idioms are used for different jobs:

- **Pacing arms are un-yoked** (#1, #2, #3, #6) — pinning them to a common clock would delete the
  pacing variable. The lifetime confound is bought back A3's way: the borrowed `ca_s0/dsil_sched`
  runs 201 cycles at the caps and is the lifetime ceiling no loop arm can outrun. Deep-era deltas
  are reported at matched cycle *and* at matched priced spend.
- **The selector bit is yoked** (#5 vs #4) — `crescendo`'s `ceiling_m3` idiom: the yoke replays
  #4's realised commit and advance cycles by clock and by nothing else, with the selector set to
  `exo`. Lifetime-identical, one bit differing. Unlike `ceiling_m3` the pair is *not*
  bit-identical up to a divergence cycle (the selector acts at c1), so the twin-window gate is
  replaced by X-5's exact-replay gate.
- The pair **#3 vs #4** (free-paced, selector off vs on) and the pair **#5 vs #4** (yoked)
  together decompose the selector's value into *content at matched clock* and *the clock the
  selection induced* — the `ceiling_m3` / `outer_yield_m3` decomposition, one lever up.

---

## 8. The next rung — corrected by the sizing lane, 2026-09-01

`--max-macro-level 5` exists as a flag and **is not opened in this tag.** The first reason below
is unchanged and is decisive on its own; reasons 2 and 3 as I first wrote them were wrong, and
the sizing lane's numbers replace them with a sharper account. Recorded here because the
correction is the useful part.

1. **It would kill the round's gates.** *(unchanged)* `max_macro_level` also sets the proposal
   head's slot layout, the ranges of the `committed`/`miners` dicts, the panel's committable
   half, and — decisively — the shadow audition's matched-size random control, which draws
   `rng.permutation` from the arm's own shared stream. An `maxl=5` arm diverges from `ca_s0` in
   the shared stream from the first cycle `miners[5]` is non-empty, and both cross-tag replays
   die.

2. **The gauge does not "go dark by arithmetic" — it goes SUB-FLOOR, which is a different and
   sharper fact.** I wrote that a thermostat on a {0,1}-range series would fire at its first
   decision point and that `tol_yield_l5` might be underivable. Measured: `tol_yield_l5` **is**
   derivable — 0.0639 over 3,658 null-ABBA windows — but the signal it is meant to gate does
   not clear it. Signal/floor is **0.26 at L5 and 0.08 at L6**, against **1.38 at L3 and 1.53
   at L4**; A1's thermostat replayed on the L5 series fires in **14 of 29 arms**, on sub-floor
   motion, with no common structure across them. So at the L5 rung **no at-support gauge at or
   above the frontier clears its own floor; only reads from behind it do.** That is a much
   better reason than the one I gave for why the split cells (#3/#4) are what to carry forward:
   the execution currency's advantage there is not "the other gauge is degenerate", it is that
   δ-silence reads a rung *behind* the frontier and the outcome gauge cannot.

3. **The r² wall is NOT what stops L5 — arrival is.** My premise was wrong. Buildable true L5
   over the arms' own frozen L4 books is **1–15 entries** (`an_s0/q_bisect`: 15), and
   `census_extend` moves it (1→4, 2→7), so the ratchet is not the binding constraint. What
   binds is that the keys never *arrive*: E[true L5 keys at support 3] is **0.000** at the run's
   ~1,600 L5-node observations, and a 22-key L5 book needs **~158,000** observations — ≈58
   GPU-h per arm at this rule draw. The arrival cut at L5 is **~5.9e9×** against the ratchet
   cut's 1.2–21× at L2–L4. A second-extension node at this draw is not a scheduling problem.

**Where the next rung actually goes.** An L5 sibling tag is affordable only at a **collision-free
rule draw** (seed 10: ~3.6 GPU-h/arm for a 22-key book; `generate_rules_invertible` already
exists), and that buys the affordability by giving up the arc's **native aleatoric channel** —
the grammar's own parse ambiguity, which is what makes `antiphon`'s nerdsnipe pole free and what
every donor table's 0.19–0.41 precision is made of. Jasper and the orchestrator have decided
**not** to open L5 at a new draw now. The next rung is being specced as a **re-keying node**
instead: under `fourwall`'s parent-feature merge basis the level size is `v·m = 16` at *every*
rung — a **12,864×** shrink at L5 — so the arrival wall is attacked by changing the key, not by
buying observations. **This section should point there, not at a `tu_l5_*` tag**, and the flag
in `tutti.py` stays shut.

**One correction that lands inside this round.** The era-4/5 dose collapse is **the level cap,
not depth.** Structural grip is exactly `(s−1)/s` at every era where the mined level tracks the
era; under `max_macro_level=4` eras 4–5 stop tracking, the damage cell swallows the whole span,
and every selector falls through to a quota-legal index order — exactly as in `antiphon`. The
reduction states it that way (a property of the cap) rather than as a fact about deep eras.
*(The world is m = 2 throughout, as every banked arm is; the config is the donor's, inherited,
so nothing built here is affected.)*

## 9. Sequencing — status

**Built and gated, 2026-09-01** (all before any paid arm):

| step | result |
|---|---|
| offline policy suite (P-1…P-12, L-1…L-8) | **ALL PASS** (20/20) against the imported `maestro.policy` |
| offline question suite (Q-1…Q-10) | **ALL PASS** (10/10) against the imported `antiphon.questions` |
| Phase 0 (`phase0_tutti.py`, CPU) | `tol_dsil` pooled **0.00459** (per-arm 0.00232–0.00541, 354 blocks); bounded gate at **c43**; the split cannot deadlock (the advance owner's gauge is live from c1, and the cap forces an advance regardless); menu geometry unchanged (clean offsets `[0]` / `[2,3]` / `[4,5,6,7]`, empty in the two consumption eras) |
| preflight, 5 split/port arms (`preflight2.log`) | **ALL PASS**, incl. C/T/I/D and the new **X-1…X-6** |
| preflight, the 8 paid arms (`preflight3.log`) | **ALL PASS**; the mirror runs (`COMMIT reads yield ǀ ADVANCE reads dsil`), the yoke handoff carries 6 actions from its source |
| `tu_gf` (`gf.log`) | **G-F PASS — 0.000e+00** vs `caesura.py` on `anchor` and `given_c1`, against a 0.000e+00 donor self-replay control, commits equal |
| Q-11, inside `tu_gf` | **PASS (18 cells)** — distinct needy true keys aimed at, exo→bisect **7→10 / 8→18 / 1→9**, reproducing `an_gf`'s numbers on this substrate |

Gate X-4 caught one real thing and it is recorded rather than papered over: at preflight's
six-cycle eras a boundary-forced bootstrap commit and a **cap-forced** advance land on the same
cycle. That is the donor's own behaviour, not the split's — precedence is a claim about *quiet
readings*, and a cap advance is the era ending. The gate now asserts over gauge-driven actions
and reports the cap co-occurrence beside it; the reduction's §5T check was corrected in step.

1. **Next** — the `--quick` smoke (`tu_smoke`, 3 arms incl. the one-bit `tu_pf_off` / `tu_y_exo`
   pair that measures the port's marginal s/cycle at K = 2048), then the launch handle.
2. On resume: build `tutti.py` + `analyze_tutti.py` + `phase0_tutti.py`; run the offline suites
   (P/L, Q-1…Q-10, Phase 0); `modal run ...::preflight` for C/T/I/D/**X**; `fidelity_smoke`
   (`tu_gf`) for G-F + Q-11; a `--quick` smoke (`tu_smoke`, 3 arms) for mechanics and the s/cycle
   measurement that confirms the budget.
3. **Halt 2** — the launch handle: app id, arms, measured s/cycle, projected GPU-h.
4. **Halt 3** — the reduced results: gates, controls, the division-of-labor cells and the selector
   cells with their yokes, uninterpreted, plus the `FILES.md` rows.

**Decisions, as taken (2026-09-01)**: (i) govern at the in-tag pooled `tol_dsil` = **0.0046**,
protocol pinned in `phase0_tutti.py`, all three numbers printed — the bounded gate to c42 on
`tu_d_exo` plus the full-life replay on `tu_y_exo` is sufficient certification, and a real
re-instantiation of finding 5 is what the round is for; (ii) **eight arms** — the six-arm core
plus R2 (`tu_y_endo`) and R1 (`tu_m_exo`), ~5.5 GPU-h; (iii) the borrowed `ca_s0/dsil_sched`
stands as the lifetime ceiling, **R3 and R4 skipped** (the pacer-conditional selector marginal
plus the yoke is the sharper statistic and the additive decomposition is `antiphon`'s to keep;
the oracle is not an action). **L5 stays closed**; §8 is the shape of the sibling tag.

[^private]: Not mirrored: this link points to a document in the private lab repo (the roadmap, the queue, an unrun spec, reading notes, or a conversation). See the top-level README for what is held back and why.
