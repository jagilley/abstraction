# trap — DESIGN: installing worthless answers in the question port

**Status**: design + offline sizing, 2026-09-01. **Nothing run on GPU.** Written for review before
any Modal spend (`ostinato`'s discipline; `antiphon/phase0_question.py` is the template).
**Up**: [`../README.md`](../README.md) (the question port) · [`../SPEC.md`](../SPEC.md) §4 (the two
payoffs, the two failure poles) · [`../FILES.md`](../FILES.md) §1–§3 (the parameterization and the
controls this design inherits).
**Offline**: [`phase0_trap.py`](phase0_trap.py) → [`phase0_trap.json`](phase0_trap.json) /
[`phase0_trap.txt`](phase0_trap.txt). No GPU, no Modal, no substrate. Reproduce:
`cd experiments/ && PYTHONPATH=. python3 rhm/practice/antiphon/trap/phase0_trap.py` (~60 s).

---

## 0. The ask, and what the offline read did to it

`antiphon` measured the ordering payoff and reported the corruption payoff as small (finding 4):
the guard-less novelty selector `q_novel` mined 39 true L4 keys against `q_exo`'s 22, in a stream
Phase 0 had called junk-dominated. The reading was "on a constrained grammar nearly every answer
is worth something", and the round's A3′ — the installed trap — was skipped. This lane's job is
to install worthless answers and size the trap before GPU.

The offline read confirms the substrate *does* have worthless answers, identifies exactly what
they are, and then finds the reason finding 4 came out mild — which is **not** "there are no
worthless answers". It is an exact ratio (§4.3) that bounds how badly any breadth-first novelty
judge can be nerdsniped here. That ratio changes the design: the knob's useful direction is
**both** ways, and the round becomes a dose ladder rather than a single trap-on cell.

---

## 1. What a worthless answer can be on this substrate

Four structural facts, each checked rather than assumed. They are the constraint set.

**(F1) The action space is world-typed.** `crystallize.units.apply_move` materialises a span by a
max-sum DP over `rules_t` and renders it through `canon = rules[depth-1][:,0,:]`;
`macros.apply_any` does the same through an earned table. Every move the agent can make writes a
world-legal constituent.

**(F2) Mining is success-filtered by the world's exact grader.** `run_arm` block (b) mines
`out["x"][ps > 0.5]`, where `ps` is `units.grade(..., rules, s)`. An instance whose answer does
not grade as a legal derivation of its root contributes **nothing** to the table.

**(F3) Therefore a foreign grammar cannot mint keys here — REJECTED, with reasons.** A second RHM
at the same shape with independent rules produces instances the agent cannot solve (F1: it cannot
materialise foreign-legal constituents) and which are therefore never mined (F2). The channel
degenerates into "burn priced budget, bank nothing" — the unlearnable pole in disguise, not the
learnable-but-worthless one it was meant to be. Making it learnable requires grading it under a
*foreign* grader, which would put foreign-legal answers into the value buffer and the generator's
fine-tune set — moving the plant, not the question distribution, and breaking the SPEC's one hard
norm. Recorded as rejected.

**(F4) What is left is non-convertibility, and it is exact.** Every true level-L flat's two halves
are true level-(L−1) flats — verified on the DGP's own tables, **0 exceptions at L3 (56 flats),
L4 (816), L5 (205,824)**. So a key with a junk half is junk *with certainty*, and `Miner.build`'s
ratchet (both halves must be rows of the operative lower table) refuses it. And the half the
*question* fixes is the one the damage cell does not cover, which reaches the miner as the
question set it.

### The trap atom

> **A menu candidate whose CLEAN HALF parses to a key that is not a row of the true table at that
> level.** Its answer cannot contribute a true key at the target level, ever.

It is drawn from the world's own cell, is legal, is solved and mined at the ordinary rate, is
d\*-matched (§4.4), needs no foreign machinery and no substrate surgery, and its worthlessness is
a theorem of the DGP rather than a modelling assumption. The knob is the fraction `f` of the menu
built from such candidates, presented spread **uniformly over the junk-half pool** so that they
look maximally like news.

The clean half is delivered exactly — this is the atom's load-bearing precondition, and it is
**confirmed from the banked `an_s0` logs offline** (§4.2), not asserted.

### What the atom is *not*

- It is not falsity of the answer. On this substrate the grader is exact and mining is
  success-filtered, so no posed question can make the agent bank a *false* answer; it can only
  make it bank a *useless* key. That asymmetry is a fact about RHM, and it is half the reason the
  corruption payoff reads small here and would not on language.
- It is not "unlearnable". §4.5: the reachable junk-half pool (96 at era 3) is smaller than the
  era's observation budget over `mine_support` (560/3 = 186), so **everything the trap poses
  recurs and closes**. The noisy-TV guard ("does disagreement close when I collect there") has no
  work to do here at matched difficulty. An unlearnable arm must give up the difficulty pin,
  exactly as `q_comp_free` gives it up to be the honest comfort pole — recorded as the optional,
  most-cuttable arm, not the main cell.

---

## 2. Parameterization

Three knobs on the world's menu, all defaulted off; `question_*` and every existing control are
untouched.

| knob | meaning | default |
|---|---|---|
| `trap_frac` (f) | fraction of the K = 2048 menu slots that are distractors | `None` (off) |
| `trap_spread` | `uniform` over the reachable junk-half pool, or `native` (frequency-weighted) | `uniform` |
| `trap_bank` | per-era distractor bank size (0 = draw per cycle) | sized in §4.6 |

**Menu construction, per cycle.**
1. The donor's own `context_instances` draw of `n_pr = 64` (the head, at the donor's RNG position)
   and the port's tail draw of `K − n_pr` — **unchanged**.
2. If `trap_frac` is `None` or the era has no clean half, **return immediately**; the code path is
   a no-op and the fork is bit-identical (gate G-T).
3. Otherwise choose `round(f·K)` menu slots uniformly at random **over all K slots, the head's 64
   included**, and *substitute* each with a distractor drawn from the era's bank. Volume is
   preserved exactly; at f = 0 zero slots are substituted.

Substituting into the head rather than only into the tail is what makes `t_exo` — "the menu's
head", the arm that selects nothing — consume distractors at exactly the menu rate, which is the
null the controls norm asks for. It also keeps `t_exo` a single object across the dose ladder.

**The distractor bank.** Built once per era at era start: native draws of the era's own cell
(`sample_derivations` → `corrupt_hier` → `nearest_derivation_cost`, exactly `context_instances`'s
pipeline), keyed by the exact parse of the clean blocks, keeping those whose key is not a true
row. Sized in §4.6: ~10 s and ~16 MB per era, against an ~10.5 s cycle.

**Where it lives.** `trap/trap_menu.py` — the bank, the substitution, and an offline gate suite
(T-3), out of the Modal app so every rule is auditable with no GPU (`questions.py`'s convention).
`antiphon.py` gains one `# [trap]`-marked call inside `pose_questions`, before the d\* computation.
Nothing else moves.

---

## 3. What the trap knob does **not** move (the controls)

The trap is a property of **the world's menu, shared by every arm in a dose**; the arms differ
only in the selector. So every control from [`../FILES.md`](../FILES.md) §3 carries over verbatim:
volume (`n_pr = 64` posed, `mine_cap = 8` asserted), the d\* quota filled bin-for-bin, priced
budget (`t_cum`), selection compute (the reader and value forwards over all 2048 in every arm),
lifetime (201 cycles, schedule pacer), oracle containment, RNG isolation.

Three things the trap adds, each with its answer:

- **Difficulty.** The quota is the head's own d\* histogram, as now. §4.4 measures that the
  distractor pool's d\* distribution is indistinguishable from the native pool's (TV **0.074**
  at era 2, **0.039** at era 3; mean d\* 2.347 vs 2.459 and 3.271 vs 3.202), so a distractor
  cannot redefine difficulty, and the enriched menu still supplies ≥ **32×** the per-stratum need
  at f = 0.9. The pre-substitution native head's histogram is logged every cycle beside the
  realised one, so the claim is measured, not stipulated. *(Considered and rejected: computing
  the quota from a native-only head. It would force the null arm to be a quota-pinned uniform
  draw rather than the head itself, which breaks the f = 0 bit-identity and gives the dose ladder
  two different nulls.)*
- **Solve rate.** A junk clean half is d\*-matched but off the value head's distribution, so it
  could depress the solve rate and starve `mine_cap`. This is the round's `wd_s0` risk and it is
  the reason for a smoke (gate T-5) before the main run. At the native rate the menu is *already*
  54.6% distractors at era 3 and mining ran at cap, which bounds the risk but does not remove it.
- **Reach.** The menu's reach over true halves is unaffected at every f considered: at f = 0.9,
  205 of 2048 candidates still carry true halves, covering all 44 reachable ones. So the trap
  costs the *null* arm, not the *selecting* arms' opportunity set — which is precisely the axis
  the round measures.

**Era-boundedness (stated, not hidden).** The trap atom exists in 2 of the 5 eras
(§4.1): era 1's clean part is a single block — a level-1 feature, always a valid row, so no junk
half exists; eras 4–5 have no clean half at all (the cell swallows the span). The trap is live in
`L2n12` (50 cycles, target L3) and `L3n6` (70 cycles, target L4), and era 2's pool is small enough
to be exhausted early (9 junk halves). **The round's action is era 3 → L4**, which is the level
the arc is stuck on.

---

## 4. Offline sizing

All numbers from [`phase0_trap.txt`](phase0_trap.txt); §-refs are that file's section tags.

### 4.1 Where the atom exists, and the pools ([0]/[1], 400k DGP draws)

| era | target | clean blocks | clean level | true halves | junk halves | native f | obs budget |
|---|---|---|---|---|---|---|---|
| `L1n25` | L2 | 24 | — | — | — | — | 480 |
| `L2n12` | L3 | 26,27 | 2 | 13 / 14 | **9** | **0.332** | 400 |
| `L3n6` | L4 | 28–31 | 3 | 44 / 56 | **96** | **0.546** | 560 |
| `L4n3` | L4 | — | — | — | — | — | 96 |
| `L5n1` | L4 | — | — | — | — | — | 72 |

**The world already serves 54.6% worthless candidates at era 3.** So the premise "the substrate
has no worthless answers" is false as stated; what it lacks is a *dose knob* and a *separable*
presentation, which is what this design adds.

### 4.2 The atom's precondition, confirmed from the banked logs ([7a])

If the question's clean half reaches the miner untouched, every at-support key's clean half must
be a legal parse of the clean node — 140 tuples out of 8⁴ = 4096 possible at era 3. It is, in
**every arm, at both eras, with no exception**:

| arm | era 2 clean halves in pool | era 3 clean halves in pool | era 3 repaired halves (distinct) |
|---|---|---|---|
| `q_exo` | 16 / 16 | **32 / 32** | 8 |
| `q_bisect` | 16 / 16 | **31 / 31** | 10 |
| `q_endo` | 18 / 18 | **60 / 60** | 10 |
| `q_novel` | 20 / 20 | **57 / 57** | 9 |
| `q_comp` | 12 / 12 | **33 / 33** | 9 |
| `q_comp_free` | 10 / 10 | **30 / 30** | 12 |

Two corollaries the design rests on. (i) The half the selector controls is delivered exactly, so
the trap atom's worthlessness survives the repair. (ii) `delivered_true` ≤ P(clean half is true)
holds on the arm that selects nothing with near-equality — **0.447 measured against 0.454
predicted** at era 3 — which is an independent confirmation of the same thing, on the null arm's
own stream. (The bound is deliberately *loose* for a selecting arm: `q_bisect` runs at 0.722,
because selection lifts its own clean-true rate far above the menu's. That is the effect, not a
failed check.) (iii) The repaired half comes from a **7–12 value repertoire** — the agent repeats
itself — which is what makes the mining model in §4.7 possible at all.

### 4.3 The bound that reframes the round ([1b]) — exact, no mining model

`_round_robin` takes one candidate per *distinct* key in priority order before it takes a second,
and every uncovered half ties at novelty 1.0. So a novelty judge's distractor share is set by the
**pool ratio**, not by the menu's mass:

| era | junk halves | true halves | pool share | over-take vs the null at native f |
|---|---|---|---|---|
| `L2n12` | 9 | 13 | **0.409** | 1.23× |
| `L3n6` | 96 | 44 | **0.686** | **1.25×** |

Predicted over-take (the judge's junk intake ÷ the null's) across the menu's worthless rate:

| era | f = 0.15 | f = 0.30 | f = 0.546 (native) | f = 0.70 | f = 0.90 |
|---|---|---|---|---|---|
| `L2n12` | 2.73 | 1.36 | 0.75 | 0.58 | 0.45 |
| `L3n6` | **4.57** | 2.29 | 1.26 | 0.98 | **0.76** |

Above 1 is a judge doing worse than not selecting at all — the nerdsnipe. Below 1 is breadth
capping its own junk intake, i.e. the judge behaving *well* in a junk-rich world.

**Three consequences for the design.**
1. Raising f **cannot** strengthen the nerdsnipe: the judge's junk intake is capped at 0.686
   whatever the world serves. This is why `an_s0` found the noise pole mild — not because the
   substrate had no worthless answers, but because breadth-first novelty is *itself* a passive
   guard whose ceiling is the grammar's ambiguity ratio at the mining node.
2. The nerdsnipe is reached by **lowering** f — cleaning the menu — where the null improves and
   the judge cannot. At f = 0.15 the judge takes 4.57× the null's junk rate.
3. Raising f is still worth doing, for the *other* payoff: it is the regime where selection is
   worth most (§4.7), which is the SPEC's real-world claim in the substrate's own terms.

Hence a **dose ladder straddling the pool share**, with the native point already banked.

### 4.4 d\*-orthogonality ([3], 24k instances per era)

| era | mean d\* native | mean d\* distractor | TV | native f | min avail/need at f = 0.9 |
|---|---|---|---|---|---|
| `L2n12` | 2.347 | 2.459 | **0.074** | 0.342 | 32× |
| `L3n6` | 3.271 | 3.202 | **0.039** | 0.548 | 32× |

The trap is difficulty-orthogonal by construction (its defining property is a parse of the *clean*
blocks; d\* is a property of the *damage*), and the measurement agrees.

### 4.5 Closure ([6]) — the unlearnable pole is not constructible at matched difficulty

| era | junk-half pool | obs budget ÷ support | closes? |
|---|---|---|---|
| `L2n12` | 9 | 133 | yes |
| `L3n6` | 96 | 186 | yes |

A channel is unlearnable only if its keys never recur. On this grammar the reachable junk-half
pool is smaller than the budget, so whatever the trap poses recurs and reaches `mine_support`.
Combined with "d\* small ⇒ near a legal derivation ⇒ solved ⇒ mined", **an unlearnable channel
requires the difficulty pin off**. Recorded; the optional arm in §5 is the honest unpinned form.

### 4.6 Construction cost ([4])

| era | distractor slots at f = 0.9 | hit rate | native draws | seconds | bank (int8) |
|---|---|---|---|---|---|
| `L2n12` | 92,150 | 0.342 | 269,452 | **10.0** | 17 MB |
| `L3n6` | 129,010 | 0.548 | 235,513 | **9.0** | 15 MB |

Once per era at era start; ~0 marginal cost per cycle against an ~10.5 s cycle.

### 4.7 The selection premium vs the world's worthless fraction ([5], [5b])

The real `questions.py` selectors on real menus, with the delivery model taken from the null arm's
own banked stream (§4.2) and one free concentration parameter fitted to it. **Model quality is
stated up front**: the era-2 fit residual is 0.113; the **era-3 residual is 0.913** — the model
over-states distinct keys (248 vs 192), at-support (60 vs 37) and true-at-support (18 vs 8). So
only *ranks and directions* are load-bearing below; the absolute levels are not. The take-rate
column, by contrast, is exact selector arithmetic and needs no delivery model at all.

Era 3 → L4, true keys at support, the null's level and each selector's difference from it:

| f | `exo` true@sup | Δ `novel` | Δ `endo` | Δ `ratchet` | Δ `ratchet_l2` | Δ `trust` | Δ `comp` | Δ `bisect` |
|---|---|---|---|---|---|---|---|---|
| 0.15 | 20 | +7 | +5 | +5 | +3 | +5 | +0 | +6 |
| 0.30 | 19 | +7 | +6 | +8 | +6 | +4 | +0 | +9 |
| 0.546 (native) | 18 | +5 | +6 | +6 | +1 | +1 | +0 | +9 |
| 0.70 | 15 | +4 | +4 | +4 | +1 | +3 | +1 | +12 |
| 0.80 | 11 | +12 | +11 | +11 | +3 | +4 | +2 | +17 |
| 0.90 | **8** | +12 | +9 | +4 | +6 | +5 | **−4** | **+18** |

Two readings the design uses. (a) The null degrades monotonically with f while the oracle does
not — **the premium on question-selection rises with the world's worthless fraction** (+6 → +18
across the ladder). (b) Breadth still beats purity at every constructible dose in this first-order
model: even at f = 0.15, where the judge takes 4.6× the null's junk, it banks *more* true keys,
because a breadth-first pass covers all 44 true halves while the null's mass-weighted draw
concentrates on a few. What the model **cannot** see is the second-order path — the plant, the
committed table's precision, and deep-era value — which is where corruption would actually bite
and is the reason to run at all (§7).

### 4.8 What discriminates, endogenously — three candidate guards, two rejected, one promoted

**The incumbent delivery ledger does not discriminate** ([5c]). Its credit rule is "the delivered
key was not yet at support", so a channel with an inexhaustible key supply keeps full weight. Mean
ledger weight, junk halves vs true halves, on the `endo` arm:

| era | f = 0.15 | 0.30 | native | 0.70 | 0.90 |
|---|---|---|---|---|---|
| `L2n12` (small key space) | 1.47× | **8.68×** | 2.92× | 2.29× | 0.91× |
| `L3n6` (816 true / 7483 junk L4 keys) | 1.19× | 1.16× | **1.04×** | 0.98× | 0.93× |

At era 3 it is **blind** (ratio ≈ 1); at era 2 it actively **prefers** the trap. This is a
correction to `antiphon`'s own machinery description: the ledger was documented as the noisy-TV
guard, and it is a second novelty test.

**A prospective ratchet guard on the clean half's own level is rejected** ([5b], [7c]). "Prefer a
candidate whose clean half is a row of my own operative L3 table" inherits that table's precision
exactly: the arms' committed L3 tables are 3/8, 8/16, 3/6, 5/15, 3/12, 3/8 — precision
**0.25–0.50** — so the guard prefers the arm's own committed *junk* rows 50–75% of the time, and
its simulated distractor take is indistinguishable from the unguarded judge's.

**The use record does separate, and is promoted to an arm** ([9], [9b]). `log["entry"]["hist"]
["beam"][level]` is the priced beam's per-entry selection count and `log["entry"]["true_mask"]`
marks which entries are true — both already logged in every `an_s0` arm, at no new cost.
`census`'s junk-filter result holds here:

| arm | `q_exo` | `q_bisect` | `q_endo` | `q_novel` | `q_comp` | `q_comp_free` |
|---|---|---|---|---|---|---|
| L3 table precision | 0.375 | 0.500 | 0.500 | 0.333 | 0.250 | 0.375 |
| L3 **use-weighted** precision | **0.733** | **0.926** | **0.904** | **0.733** | **0.792** | **0.819** |
| lift | 1.95× | 1.85× | 1.81× | 2.20× | **3.17×** | 2.18× |
| row-wise AUC | 0.400 | 0.688 | 0.889 | 0.580 | 0.704 | 0.733 |

The lift is large at L3 (imprecise table) and small at L2 (1.02–1.34×, already 0.63–0.73 precise).
But the **row-wise AUC is thin** — the use mass concentrates on one dominant true row rather than
rank-ordering the rest — so the record works as a **weight**, not as a membership test.

**The guard has to be applied one level below the clean half.** A clean half at era 3 is a
level-3 key; its own two halves are level-2 keys, and *that* is the level where the arm has both a
committed table and a use record. Only **8 of the 96** junk clean halves have both level-2 halves
in the *true* L2 table (0.083 by count, 0.140 by mass) — a perfect L2 table would refuse 91.7% of
the trap. The arms' own committed L2 tables reconstruct exactly from the logs (rows and true
counts match every commit event's `tab_n_learned`/`tab_n_correct`; no recert ever swapped) and
carry 3–6 junk rows. Junk share of the pool the judge draws from:

| arm | own L2 table | unguarded | ratchet-L2 (hard membership) | **trust-L2 (use-weighted)** | ceiling (true L2) |
|---|---|---|---|---|---|
| `q_exo` | 8/11 | 0.686 | 0.581 | **0.498** | 0.154 |
| `q_bisect` | 9/14 | 0.686 | 0.719 | **0.448** | 0.154 |
| `q_endo` | 10/14 | 0.686 | 0.635 | **0.463** | 0.154 |
| `q_novel` | 10/16 | 0.686 | 0.678 | **0.431** | 0.154 |
| `q_comp` | 10/14 | 0.686 | 0.646 | **0.536** | 0.154 |
| `q_comp_free` | 7/10 | 0.686 | 0.659 | **0.540** | 0.154 |

**Verdict.** Hard membership in the arm's own table does nothing (0.58–0.72, and *worse* than
unguarded for `q_bisect`); use-weighting the same rows cuts the junk share to **0.43–0.54** in
every arm, recovering roughly a third of the gap to the perfect-table ceiling. **The learner's own
use record is the one endogenous signal measured to separate junk from true where its table's
precision does not** — and it needs no new machinery: the recorder is installed, priced and logged
in every arm already. Promoted to `t90_trust` (§5).

Two qualifications that travel with it. (i) Under the per-cycle quota the *realised* distractor
take improves less than the pool statistic (0.686 → ~0.60–0.73 in the simulation), because the
guard's admit set (43 of 140 halves) is smaller than the 64-per-cycle breadth demand and the
selector falls back to the floor for the remainder. (ii) In the first-order model `trust` trades
breadth for purity — junk@support 62 vs `novel`'s 92 at f = 0.90, at similar true@support — so its
predicted effect is on the **corruption axis**, which is exactly the axis the offline model cannot
resolve (committed-table precision, deep-era value) and the round exists to measure.

The guard still untested after this round is per-row **audition** credit rather than per-row *use*
credit; that needs machinery that does not exist and is the follow-on.

---

## 5. The arms

One tag, `tr_s0`. All arms on `anchor_long`'s configuration (schedule pacer, ladder
60/50/70/12/9, `max_macro_level = 4`), **201 cycles by construction**, so no comparison can be
bought with time. Within a dose, every arm faces the same menu and differs only in the selector.
The native dose is **already banked** in `an_s0` and costs nothing.

| dose | arm | selector | job |
|---|---|---|---|
| f = 0.90 | `t90_exo` | the menu's head | the null in a worthless-rich world |
| | `t90_novel` | half-key novelty, no ledger | the guard-less judge |
| | `t90_endo` | novelty × the delivery ledger | the incumbent endogenous judge |
| | `t90_trust` | novelty × the **beam's own per-entry use** of the two level-2 rows under the clean half | the guard §4.8 promotes — the one endogenous signal measured to separate; ties the trap to Track F |
| | `t90_bisect` | the oracle | the ceiling; refuses distractors by construction (§6, T-7) |
| f = 0.546 | `an_s0/{q_exo,q_novel,q_endo,q_bisect}` | — | **banked, free** — the native dose |
| f = 0.15 | `t15_exo` | the menu's head | the null in a worthless-poor world |
| | `t15_novel` | novelty, no ledger | **the nerdsnipe cell** — §4.3 puts its junk intake at 4.57× the null's |
| | `t15_endo` | novelty × the ledger | does the incumbent guard help where it should matter most? *(cuttable-last)* |

Optional, run only if the round is extended: `t90_noise_free` — the honest unlearnable pole, with
the difficulty pin off (multi-node damage, the `wd_s0` form), its d\* deviation logged every cycle
because the deviation *is* the arm's definition, exactly as `q_comp_free` is treated.

The oracle needs **no code change**: `select_bisect` already scores a key that is not in the truth
table at prio 0.0, and by F4 a distractor's designed key is never in it.

---

## 6. Gates

| gate | what | where | status |
|---|---|---|---|
| **T-1** | worthlessness: halves of a true level-L flat are true level-(L−1) flats | `phase0_trap.py` [2] | **PASS** — 0 exceptions at L3/L4/L5 |
| **T-2** | d\*-orthogonality and quota feasibility at every f | `phase0_trap.py` [3] | **PASS** — TV 0.074 / 0.039; ≥ 32× need |
| **T-3** | menu construction, 9 checks: the atom's definition; the bank is pure and spreads over the whole junk-half pool; substitution preserves volume, hits the right count and lets the head consume at the menu rate; **at f = 0 the inputs are returned untouched**; determinism; every substituted slot really carries a distractor; scope (era 1 and eras 4-5 off by construction); trust weights normalise and fall back to uniform | `trap_menu.py::trap_gate` | **ALL PASS (9/9)** |
| **T-4** | containment: `trap` / `distractor` / `is_d` / `bank` refused in an agent bundle; the trust bundle's `subw` (the arm's own use record over its own rows) allowed | `questions.py::containment_gate` | **ALL PASS** |
| **Q-12 / Q-13** | the `trust` selector fills the quota, is deterministic, prefers halves whose sub-rows the arm's beam actually used (30 of 64 vs novelty's 5 on the synthetic menu), and degrades to novelty with no use record | `questions.py::question_gate` | **ALL PASS (Q-1...Q-13, 13/13)** |
| **P-1...P-12, L-1...L-8** | A1's/A2's offline policy suite against the imported module, re-run after the edits | `maestro.policy.policy_gate()` | **ALL PASS (21/21)** |
| **G-T** | with `trap_frac=None` this file replays `antiphon.py` in process | `antiphon.py::fidelity_smoke` | **staged, not run** (needs Modal; target **0.000e+00**) |
| **reproduction** | after the `# [trap]` edits, `analyze_antiphon.py --tag an_s0 --merge-tag an_s1` reproduces its banked reduction | offline | **PASS - `reduction.json` byte-identical**; SS0's two full-scale gates still 0.000e+00, commits equal, lane B's sibling twin window still c19 |
| **T-5** | **the smoke's job**: does a junk-rich menu depress the solve rate and starve `mine_cap`? Per-cycle `n_solved`, `n_mined`, `[q!] VOLUME SHORTFALL` at f = 0.9 vs f = 0 | smoke, then §1 of the reduction | to run |
| **T-6** | the clean-half invariant, now load-bearing: every mined observation's clean half equals the posed one and lies in the era's reachable pool — logged per cycle as `clean_dose`, in every arm | in-run | to build |
| **T-7** | the oracle takes no distractor while a needy true-half candidate is in-stratum | reduction §2 | to build |
| quota / volume / priced budget / lifetime / RNG isolation / N-1, F | as `an_s0` | unchanged | unchanged |

---

## 7. Cost, run order, and what the GPU buys that the offline read cannot

**Projected GPU-h.** `an_s0` realized 4.12 GPU-h for 1386 arm-cycles = 10.7 s/arm-cycle. Eight
201-cycle arms = 1608 arm-cycles ≈ **4.8 GPU-h** (seven, dropping `t15_endo`, ≈ 4.2). The trap adds a ~10 s per-era bank build and ~0
per cycle (§4.6). Plus the G-T fidelity smoke (~0.2) and a `--quick` end-to-end smoke that carries
T-5 (~0.3). **Total ≈ 5.3 GPU-h**; 4.7 dropping `t15_endo`, 4.1 dropping `t15_endo` and the f = 0.15 pair.

**Run order** (most-cuttable last): G-T → `--quick` smoke (T-3/T-5/T-6 shakedown) → `t90_exo`,
`t90_bisect`, `t90_novel`, `t90_endo`, `t90_trust` → `t15_exo`, `t15_novel` → `t15_endo`.

**Reduction**: `trap/analyze_trap.py` — §0 fidelity and the `an_s0` cross-tag join at the native
dose; §1 controls (quota bin-for-bin, volume, `t_cum`, the realised-vs-native d\* histograms);
§2 the trap dose (take rate, mined rate, `clean_dose`); §3 climb on both clocks; §4 **committed
table precision per level** — the corruption metric the offline model cannot reach; §5 the
selection premium vs f, with `an_s0` as the middle point; §6 deep-era value and trust; §7 the two
poles.

**What the offline read already answers, and therefore what the GPU is for.** Settled without
GPU: the foreign channel is impossible (F3); the unlearnable channel is not constructible at
matched difficulty (§4.5); the clean-half precondition holds (§4.2); the trap is d\*-orthogonal
(§4.4); the nerdsnipe is bounded by the pool ratio and is reached by *lowering* f, not raising it
(§4.3); the incumbent ledger does not discriminate and a same-level ratchet guard inherits its own
table's precision (§4.8); and the beam's own use record *does* separate, one level down, at
1.8–3.2× the table's precision (§4.8). What remains is exactly the **second-order path**, which no offline model reaches: whether a doubled junk-at-support mass converts into
worse committed tables, whether that costs deep-era value and trust, and whether the solve rate
holds. That is the round.

---

## 8. Staged code (no Modal touched)

| file | what | status |
|---|---|---|
| `trap/trap_menu.py` | the trap channel out of the Modal app (`questions.py`'s convention): the atom, the per-era `DistractorBank`, `install_trap`'s substitution, `trust_weights`, and the T-3 gate suite | **new, 9/9 gates pass** |
| `trap/phase0_trap.py` -> `.json` / `.txt` | the offline pass, sections [0]-[9b] | **new, runs in ~60 s** |
| `trap/analyze_trap.py` | the reduction skeleton - SS1/SS2/SS7/SS8 wired to fields the run logs, SS0/SS3/SS4/SS5/SS6 land with the tag. Reuses `analyze_antiphon.py`'s loaders wholesale | **new, skeleton** |
| `questions.py` | `select_trust` + `SUBW_FLOOR`; `_FORBIDDEN` extended (T-4); gates Q-12/Q-13 | **additive, `# [trap]`-marked** |
| `antiphon.py` | the import; `pose_questions`'s trap block (its own `_tgeom`, the bank, `install_trap`) and its `trap` dose row; the `trust` bundle's `subw`; `run_arm`'s `tstate` and the cumulative beam-use accumulator; `trap_mined` beside the dose instrument; nine `t*_*` arms + their `TWIN` entries; three cfg keys defaulted OFF; `trap_gate` in `preflight` and `fidelity_smoke`; `q_interface_check` extended to the `trust` mode | **additive, 11 `# [trap]` marks** |

Lane B's `# [antiphon-s2]` blocks in `antiphon.py` and `analyze_antiphon.py` were left untouched,
and the banked `an_s0 --merge-tag an_s1` reduction reproduces byte-for-byte after the edits.
**No new launcher is needed**: the trap arms live in `antiphon.py::ARMS`, so
`antiphon/launch_detached.py --fn antiphon_run --tag tr_s0 --arms "..."` runs them as they are.
Nothing on Modal has been touched - `fidelity_smoke`, the `--quick` smoke and every run remain
unrun, pending the go.

## 9. Open choices for review

1. **Is the dose ladder the right shape**, or should the round spend all seven arms at f = 0.90
   (the regime that matches Jasper's framing — a world full of aleatoric noise) and leave the
   f = 0.15 nerdsnipe cell, whose sign the arithmetic already predicts, unrun?
2. **f = 0.90 vs f = 0.80.** At 0.90 even the oracle is forced by the difficulty quota to take
   ~41% distractors (§4.7 take column); at 0.80 it takes ~35% and the null still degrades. 0.90
   is the sharper dose; 0.80 keeps the ceiling cleaner.
3. **The full-scale fidelity gate.** `an_s0` spent its load-bearing gate as `q_exo` ≡
   `cr3_s0/anchor_long`, which it needed anyway. Here no trap-on arm is a replay, so the choice
   is the in-process G-T gate alone (the `an_gf` precedent, ~0.2 GPU-h) or an extra full-scale
   `t_exo` at f = 0 replaying `an_s0/q_exo` (+0.6 GPU-h).
4. **Whether to run the unpinned unlearnable arm** (`t90_noise_free`) at all, given §4.5 says the
   pin and the pole are incompatible and the arm's d\* deviation would be its definition.
5. **Whether the `trap` sub-experiment lives here or is folded into `antiphon`'s main lane.** As
   drafted it is `antiphon/trap/` with its own `DESIGN.md`, offline pass, launcher and reduction,
   and additive `# [trap]` marks in `antiphon.py` / `questions.py`.
