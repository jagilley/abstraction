# ratchet — earning a level-indexed vocabulary, one depth era at a time

**Up**: [../README.md](../README.md) (rhm/practice) · [../../README.md](../../README.md) (rhm)
**Idea doc**: [practice_manufactures_its_own_credit](../../../../ideas/practice_manufactures_its_own_credit.md)
(§1 re-chunking, §3 the compile op and the committability certificate; §14 points here)
**Parents**: [`../crystallize/`](../crystallize/README.md) (round 1 — the certificate found vacuous on a
frozen plant, and the precheck that scoped this round) ·
[`mjc/practice/etude/`](../../../mjc/practice/etude/README.md) (the compile op, the winner's curse, the
seam-state shift) ·
[`directed_sculpting/full_loop/level_moves/level_ladder/`](../../directed_sculpting/full_loop/level_moves/level_ladder/README.md)
(the depth-laddered damage schedule, and the 3.51× the level *action* space bought when handed over)
**Status**: written up 2026-08-14 (interpretation discussed with Jasper 2026-08-14). Seven runs: a
gate, a ladder calibration, three substrate calibrations, a perception diagnostic, and the main
5-arm run. File index: [FILES.md](FILES.md).

## One-liner

Round 1 found the compile certificate vacuous because practice trained the judge while the executor
stood still. This round gives practice something it can actually move — **its own action space** —
by mining a level-indexed macro vocabulary from the agent's own successful repairs, and grading it
against the DGP's true vocabulary handed over for free. **Earning the vocabulary recovers 68–98% of
what being given it buys**, at 0.85× the priced time; **committing it one cycle early is worse than
never committing at all**, because a compiled level-2 error forecloses level 3's *representation*
rather than merely its accuracy. The unit-LP certificate fires within one cycle of its offline
prediction at level 2 — and correctly refuses at level 3, where a mis-levelled audition understates
the unit by 1.75× and the refusal costs it the era.

## Findings

Margins are stated against measured noise floors, and the one claim that does not
survive the main run is retracted in place.

1. **Earning the vocabulary recovers most of being given it.** The earned-vs-given fraction
   `(e_never − e_arm) / (e_never − e_given)` is **0.846 / 0.835 / 0.678** for `practice_gated` and
   **0.498 / 0.932 / 0.979** for `practice_late` across the three depth eras, against denominators of
   0.148 / 0.290 / 0.301. `practice_late` reaches 93% and 98% at eras 2–3 at **0.85× `given`'s priced
   time**. This is the round's headline number and the direct answer to the founding question — an
   agent can learn high-level features in its own native terms and lose little. (§Priced grade)
2. **A cost-to-depth curve, and vocabulary flattens it.** At a flat declared budget, `never_base`'s
   error grows **+0.293** from era 1 to era 3 (0.349 → 0.642); `given`'s grows +0.141; `practice_late`'s
   grows **+0.072** — 4.1× flatter than base moves and flatter than `given` itself. Matched-priced-time
   margins against `never_base`'s own trajectory at era 3: `given` +0.294, `practice_late` +0.287,
   `practice_gated` +0.224, `practice_early` **−0.016**. (§Priced grade)
3. **Depth is unaffordable to base moves, not merely harder.** At era 2 the base action space's
   *privileged exact-DP oracle* reaches only 0.449 and its width-16 beam (344 groundings, 5.9× the
   declared budget) reaches 0.504, while one true level-2 macro at width 1 and **49 groundings** reaches
   0.293. Era 3: 0.586 / 0.643 vs **0.393 at 57 g**. Priced: at era 2, 10.4× more feedback on base moves
   buys −0.103; adding the vocabulary at 1.5× the feedback buys −0.314. **Vocabulary buys 3.0× more than
   a 10× width increase at ~1/7 the marginal cost.** (§The ladder)
4. **The unit-LP certificate fires where its offline replay says it will.** `practice_gated` committed
   level 2 at **c20**; the offline detector replayed over the calibration's own series predicted
   **c19–c20**. The committed table: 8 entries, **recall 0.500, precision 0.875** (7 of 8 mined tuples
   are real grammar). `practice_late`'s scheduled c27 commit is the same quality (0.500 / 0.875).
   (§Certificate)
5. **Era 2 was pre-registered as possibly unresolved, and it resolved by refusal.** `practice_gated`'s
   level-3 audition never descended (0.645 → 0.661, min 0.607), so `lp_min_drop = 0.10` blocked the
   commit and the certificate stayed silent for all 30 cycles. `practice_late`, committing on schedule
   with a **7-entry, recall-0.107** table, finishes era 3 at **0.347** — essentially `given`'s 0.341 —
   where `practice_gated`, having refused, sits at **0.438**. The certificate is **conservative**: where
   learning progress is absent but the unit is nonetheless useful, it under-commits and pays. (§Certificate)
6. **The depth seam is *pessimistic*, inverting the étude's.** Commit audition ÷ realised competence in
   the era where the unit is consumed: level-2 commits give **0.98 / 0.92** (audition predicts), but
   `practice_late`'s level-3 commit gives **0.57** — the audition **understates the unit by 1.75×**.
   The mechanism was measured before the run: a level-3 macro auditioned on era-2 damage rewrites two
   *clean* blocks, capping its best possible score near 0.51. The étude's seam made audition optimistic
   by 3.0×; this one makes it pessimistic by 1.75×, and it is the direct cause of finding 5. (§Seam)
7. **Early commitment forecloses the next level's representation.** `practice_early` committed at c1 a
   **1-entry table with recall 0.000 and precision 0.000** — a single tuple that is not in the grammar
   at all. Its frozen audition stays at **1.000 for all 30 cycles** while the counterfactual live table
   descends 0.504 → 0.330. Because `T3 ⊆ T2 × T2`, it **never obtained a level-3 candidate at any
   point**: era 2's vocabulary was literally unrepresentable. Its error is worse than `never_base` in
   every era (0.404 / 0.564 / 0.664 vs 0.349 / 0.530 / 0.642), so its earned-vs-given fraction is
   **negative** (−0.372 / −0.119 / −0.071). **Committing early is worse than never committing.**
   (§Poison)
8. **Mining recovers the consumption prior at level 2; at level 3 only when level 2 is complete.**
   Against a matched-size random subset of the *true* table, the mined table scores **−0.173 to −0.211
   in 29/30 cycles** at level 2 in every arm. **Retraction**: my checkpoint summary claimed this held
   "in all 16 cells, −0.095 to −0.243". That was measured at `plant_holdout` ∈ {3, 5}; on the main run's
   substrate (holdout 0) the level-3 advantage collapses — `never_base` **+0.005 (16/28)**,
   `practice_gated` −0.021 (19/30), `practice_late` −0.088 (23/30), and only `given` retains it at
   −0.148 (28/30). Corrected claim: **the prior is inherited at depth, not independently learned.**
   (§Mined vs random)
9. **The structural half of compounding is demonstrated; the rate half is not.** `practice_early`'s
   total failure to form any level-3 vocabulary is the ratchet's cap observed in the loop (gate C-R
   reproduced live). But the era-2 half-descent cycle is **non-monotone** in level-2 vocabulary quality
   — `given` (16 true entries) c5 / min 0.486, `practice_gated` (8 earned) c4 / 0.607, `practice_late`
   (8 earned) c11 / 0.537, `never_base` (none) c12 / 0.490 — against level-3 series noise of 0.04–0.09
   and spans of 0.05–0.30. **We do not claim era-k+1 certifies faster given era-k units.** (§Compounding)
10. **The plant stayed inert; the entire descent is vocabulary-carried.** `parse_acc` 0.62–0.65 and
    `infill_acc` 0.69–0.71, flat across 90 cycles in all five arms; the true-table audition (a pure
    plant readout on fixed instances) is flat (`never_base` era 1 0.346 → 0.323). Meanwhile the
    candidate audition falls 0.63–0.73 → 0.32–0.33 over era 1. Round 1's scope condition — the
    certificate needs practice to move *something* the unit depends on — is satisfied here by the
    **action space**, not by the executor. (§Plant guard)

## Scope

One 5-arm run (`rr_s0`, seed 0, 3 eras × 30 cycles = 90 cycles) plus six calibrations and gates that
configured and corrected it. δ is consumed **only as a detector**, and only on the shadow-audition
trajectory; every online update (value, generator) is plain uniform-lr AdamW with no per-sample gain.
Nothing outside this folder was modified.

## Substrate

Fork of round 1's Stage-3b sculpting setup: v=8, s=2, L=4, m=2 → 16 tokens, 8 blocks. Controller,
generator (block infiller), MC value V(z̄, r\*), and — new here — a **reader**, all trained once at
setup and forked per arm.

- **The base action space** is level-1 moves only (8 moves), `build_move_set(depth, s, max_level=1)`.
  Everything above level 1 must be earned.
- **A macro** ([`macros.py`](macros.py)) is the *same* max-sum operator the true level move uses, with
  a **learned table** in place of `rules_t`: `T[1]` is the v level-1 features; `T[l]` is a set of
  entries, each an s-tuple of `T[l−1]` **entry indices**. Gate **C-M**: with the DGP's own table the
  macro is **bit-identical** to `units.apply_move` at levels 2 and 3 — so `given` and `practice_*` are
  the same machinery differing *only in which tuples are in the table*, which is what makes the
  earned-vs-given fraction a statement about vocabulary rather than about operators.
- **The ratchet is structural.** `T[l]` is defined over `T[l−1]` entries, so a level-3 chunk whose two
  halves are not both in the level-2 vocabulary is not representable and is dropped at build time.
  Gate **C-R**: truncating T2 from 16 → 6 entries drops the rebuilt T3 from 56 → 14.
- **Mining.** Only configurations the agent actually **solved** (terminal possible-set success), and
  only the beam's own final answer per instance (`mine_from = chosen`), capped at `mine_cap = 8`
  configurations per cycle, entering the table at `mine_support = 3` observations. Parsed by the
  agent's own **reader**, never by the exact map.
- **The depth ladder** — nested damage cells, so era k+1's error is literally era k's one level up:
  **L1 n6 ⊂ L2 n3 ⊂ L3 n1** (block 6; blocks 6–7; blocks 4–7). Gate **G-D**: on-grammar **1.000** at
  all three levels, d\* = 1.56 / 2.21 / 2.69, `d*==0` on 3.9 / 4.7 / 5.5% of instances (against round
  1's 19.9%, so rejection sampling is nearly free). One era per 30 cycles, fixed boundaries.
- **Practice** — closed-loop re-grounded token beam at width 16, budget **4** moves, full
  materialise-and-re-encode; every surviving tip's trajectory labelled by that tip's terminal success
  and fed to the value.
- **Performance + pricing** — a declared per-solve **grounding budget G = 58**, and each arm runs the
  widest beam that fits *its own* action-set size. This is what matched pricing means here: a level
  move costs the same groundings whoever holds it. At G = 58 the **base** arm consumes the most
  groundings of the three (base w2 = 58 g, 12-move w1 = 49 g, 14-move w1 = 57 g), so no reading can be
  dismissed as the base arm having been starved. Priced time `t = n_ground·d_fb + n_mat·c_mat` at
  `d_fb = 1.0`, `c_mat = 0.05`.
- **The plant** — the generator fine-tunes online on the agent's own solved configurations
  (`calp_s0`'s recipe: masked infilling, replay 0.5 against the clean setup pool), and the value
  co-adapts against it. Round 1's precheck confound is fixed by construction. It did not move
  (finding 10).
- **Metering** — `e_k` = 1 − success on a fixed held-out 384-instance set **per era**, with the stale
  reference measured on that same set (round 1's detector-scale bug, fixed).
- **The certificate** — silence on the **shadow-audition trajectory of the candidate macro**, `A`:
  the candidate table applied as one action to a fixed held-out 512-instance set of the current era,
  every cycle. `b` = EWMA(α=0.2), `δ = b − A`, silence = window-mean |δ| < c·scale **and** sd(A) <
  c_v·scale held `sil_hold` cycles, `scale = max(ref − min_so_far, b)` — **plus** an explicit
  descent precondition `(ref − min_so_far) ≥ lp_min_drop`. Run at c = 0.06, c_v = 0.15, W = 5,
  hold = 2, `lp_min_drop` = 0.10.
- **Instruments, never acted on**: `A_true` (the DGP's own table on the same instances — the plant-only
  component), `rand_k` (a matched-size random subset of the true table, 3 draws — the
  concentration-vs-coverage control), `held`/`live` (the frozen committed table against what the live
  vocabulary would be — the counterfactual recert), the width ladder, and the plant guard.

## Arms (vocabulary-acquisition policy is the only difference)

| arm | vocabulary | notes |
|---|---|---|
| `never_base` | base level-1 moves forever (8 moves) | must buy depth with search at the declared budget |
| `given` | the DGP's own level-2 and level-3 tables from cycle 1 (14 moves) | the `level_moves` ceiling, matched-priced |
| `practice_gated` | earned, gated by the unit-LP certificate | |
| `practice_early` | earned, committed at each era's cycle 1 | the poison test, on a plant that learns |
| `practice_late` | earned, committed at each era's cycle 28 | the rent-paying control |

All five carry the learning plant, mine, and audition; committed macros are **frozen** (the recert
counterfactual is measured, never acted on), and are instantiated at every node of their level, so a
fully-earned action space matches `given`'s move-for-move.

## Results — `rr_s0` (seed 0, 5 arms, 90 cycles, complete, 925 s)

### Priced grade and the cost-to-depth curve

`e` is the mean over each era's last 5 cycles; `t` is that era's priced time.

| arm | era 1 (L1n6) | era 2 (L2n3) | era 3 (L3n1) | t_cum | depth slope (e₃−e₁) |
|---|---|---|---|---|---|
| `never_base` | 0.349 @ 1.41M | 0.530 @ 1.41M | 0.642 @ 1.39M | 4.21M | **+0.293** |
| `given` | **0.201** @ 2.05M | **0.240** @ 2.05M | **0.341** @ 2.05M | 6.14M | +0.141 |
| `practice_gated` | 0.223 @ 1.51M | 0.288 @ 1.73M | 0.438 @ 1.71M | 4.95M | +0.215 |
| `practice_early` | 0.404 @ 1.70M | 0.564 @ 1.71M | 0.664 @ 1.71M | 5.12M | +0.259 |
| `practice_late` | 0.275 @ 1.44M | 0.260 @ 1.76M | 0.347 @ 2.05M | 5.25M | **+0.072** |

Matched priced time (each arm's error against `never_base`'s own interpolated trajectory) at era 3:
`given` +0.294 · `practice_late` +0.287 · `practice_gated` +0.224 · `practice_early` **−0.016**.
*Caveat*: `never_base` exhausts at 4.21M, so comparisons past that clamp to its endpoint.

**Earned-vs-given fraction** `(e_never − e_arm)/(e_never − e_given)`:

| arm | era 1 | era 2 | era 3 |
|---|---|---|---|
| `practice_gated` | **0.846** | **0.835** | 0.678 |
| `practice_late` | 0.498 | **0.932** | **0.979** |
| `practice_early` | −0.372 | −0.119 | −0.071 |

### The ladder (`call_s0`, 512 held-out instances per era, setup plant)

Measured before the loop was built, on identical instances across action sets.

| era | base w1 (33 g) | base w2 (58 g) | base w16 (344 g) | **base exact-DP oracle** | +L2 w1 (49 g) | +L2+L3 w1 (57 g) |
|---|---|---|---|---|---|---|
| L1n6 | 0.322 | 0.303 | 0.293 | **0.152** | 0.271 | 0.234 |
| L2n3 | 0.607 | 0.576 | 0.504 | **0.449** | 0.293 | 0.242 |
| L3n1 | 0.760 | 0.734 | 0.643 | **0.586** | 0.531 | 0.393 |

Era 1 inverts — the base DP oracle (0.152) beats everything else — which is why **era 1 is the
bootstrapping era where the vocabulary is earned, not a test of whether it is needed**. Necessity is
established at eras 2 and 3.

### The certificate, and its refusal

| arm | era | level | cycle | c_in_era | entries | recall | precision | audition |
|---|---|---|---|---|---|---|---|---|
| `practice_gated` | 1 | 2 | **c20** | 20 | 8 | 0.500 | 0.875 | 0.2930 |
| `practice_early` | 1 | 2 | c1 | 1 | 1 | **0.000** | **0.000** | 1.0000 |
| `practice_late` | 1 | 2 | c27 | 27 | 8 | 0.500 | 0.875 | 0.2812 |
| `practice_late` | 2 | 3 | c57 | 27 | 7 | 0.107 | 0.857 | 0.6074 |

`practice_gated` never committed level 3: its `A` series ran 0.645 → 0.661 (min 0.607), the descent
precondition was never met, and the certificate stayed silent for all 30 cycles. Offline replay on
the calibration's own series predicted a level-2 firing at c19–c20; the run fired at **c20**.

### The seam, with its depth coordinate

Commit audition (measured on era k's damage) against realised competence in era k+1, where the unit is
actually consumed:

| commit | audition | realised (era k+1) | gap |
|---|---|---|---|
| `practice_gated` L2 | 0.2930 | 0.2880 | **0.98** |
| `practice_late` L2 | 0.2812 | 0.2599 | **0.92** |
| `practice_early` L2 | 1.0000 | 0.5641 | 0.56 |
| **`practice_late` L3** | **0.6074** | **0.3474** | **0.57** |

### The poison, and its mechanism

`practice_early`'s frozen level-2 table against the live counterfactual, over era 1:

| | first 5 cycles | last 5 cycles |
|---|---|---|
| `held` (the frozen committed table) | **1.000** | **1.000** |
| `live` (what the vocabulary would be) | 0.504 | 0.330 |

Final committed vocabulary: `{2: 1 entry, 3: None}` — 12 moves, one of which is a macro that always
writes a tuple the grammar cannot produce. No level-3 candidate existed at any cycle of era 2.

### Mined vs random, and the plant guard

Mean `A_cand − rand_k` (negative = the mined table beats a matched-size random subset of the *true*
table), with the count of cycles where it wins:

| arm | era 1, level 2 | era 2, level 3 |
|---|---|---|
| `never_base` | −0.191 (29/30) | **+0.005 (16/28)** |
| `given` | −0.211 (29/30) | −0.148 (28/30) |
| `practice_gated` | −0.201 (29/30) | −0.021 (19/30) |
| `practice_early` | −0.173 (29/30) | — (no candidate) |
| `practice_late` | −0.210 (29/30) | −0.088 (23/30) |

Plant guard, clean held-out configurations, all five arms across 90 cycles: reader **1.000**
throughout; generator `parse_acc` 0.62–0.65 and `infill_acc` 0.69–0.71, flat. No self-imitation drift,
and no plant learning either.

## Interpretation (discussed with Jasper 2026-08-14 — argued, not measured)

Five readings we agreed on. None is a measurement.

- **(a) The level-3 refusal is an instrument-scope result, not a strike against the LP certificate.**
  The certificate applied its own criterion correctly: there was no learning progress in the signal it
  was given. The signal was mis-levelled — a level-3 macro auditioned on era-2 damage, where it
  over-commits by construction and its best possible score is capped near 0.51. A certificate cannot be
  better than the audition it reads, and the fix belongs to the audition, not to the gate.
- **(b) The seam law acquires a depth coordinate.** The étude's finding was that audition must match
  the consumption distribution in **state**; this round adds that it must match in **level**. The
  étude's seam was optimistic (3.0×) because units were auditioned on easy practice boundaries and
  consumed on hard performance ones; this round's is pessimistic (1.75×) because the unit is auditioned
  on shallower damage than it was built for. Same law, opposite sign, and the sign is set by which
  direction the mismatch runs.
- **(c) The poison result is this round's cleanest theoretical export.** The étude's weak form was
  *compiled error persists*. Here compiled error **forecloses the next level's representation**:
  because `T3 ⊆ T2 × T2`, a level-2 vocabulary frozen at one bogus entry makes era 2's vocabulary
  unrepresentable, not merely worse. That is a strictly stronger statement about premature commitment,
  and it was observed in the loop rather than only in the gate.
- **(d) The earned table is an empirical consumption prior.** The grammar defines what is *possible*;
  the task distribution defines what is *probable*. A mined 8-entry table beats a random 8-entry subset
  of the true 14 by ~0.20, and in round 1's calibration a mined level-3 table repeatedly beat the
  exhaustive true table on the same instances. **The vocabulary's value is concentration, not
  coverage** — which is why "recall → 1.0" is the wrong objective and why the certificate's floor is
  the best consumption-weighted table rather than the exhaustive one.
- **(e) The teacher lends the student a grader (Jasper).** One of a piano teacher's primary functions is
  to be an external, *already-climbed* evaluation system: the teacher can hear the phrase-level signal
  across bar-level noise long before the student can, and lends that grader until the student's own
  evaluation layer climbs to it. That is exactly this round's failure mode — the agent's level-3
  audition was too noisy and too mis-levelled to certify anything, so it under-committed. Teaching, on
  this reading, is transfer of the **grader**, not of the content. It belongs next to the
  schooling-as-manufactured-recurrence thread in the idea doc.

## Calibration record

Every knob was set by a measurement, in the order the measurements forced. Three of these are design
corrections earned the hard way and are findings in their own right.

| run | measured | change it forced |
|---|---|---|
| `selfcheck` | **C-M** — a macro with the DGP's table is bit-identical to `units.apply_move` at L2/L3; **C-R** — truncating T2 16→6 drops T3 56→14; **G-D** — nested damage on-grammar 1.000 at all three levels, `d*==0` on 3.9/4.7/5.5% | the design is admissible; rejection sampling is cheap |
| `call_s0` | the ladder above; the coverage curve (L2 audition 0.82 → 0.32 over 1→16 entries; **L3 saturating by 8 of 56**) and its huge entry-identity variance at small k (L3 k=4 spans **[0.104, 0.973]**) | `g_budget` = **58** (the only point where the base arm spends the most); budget 3 → **4** so level-3 damage is feasible for base moves; and the **matched-size random-subset control**, added because the coverage variance would otherwise make an early-committed table a lottery rather than a measurement |
| `calr_s0` | **no resolvable descent at any (mine_cap, gen_lr)**; and the mined tables had **low precision** — level 3: 10 true of 34 mined, and `never_base` mined **100 distinct 4-tuples from a grammar with 56** | halt, and diagnose the read rather than tune the knob |
| `calp2_s0` | **the read diagnosis, and my first hypothesis was wrong.** I predicted distribution shift (the generator never sees an unmasked input) and tested masking one block elsewhere: it **did not help** (0.599 vs 0.627). The real cause is upstream — `_train_generator` computes its loss **only on masked blocks**, so the block head at a *visible* position is never supervised at all, which is why it scored *worse* reading a visible block (0.63) than infilling a hidden one (0.70) | a trained **reader**: same architecture, same corpus, and the same `bottom_map` supervision the generator already trains against, but trained to read. It reaches **1.000** and sits **exactly at the exact-map ceiling** (L2 precision 0.818/0.889/0.857, L3 0.703/0.654) — the residual is `build_inverse_maps` being last-writer-wins over bottom-level synonyms, an irreducible substrate read error, not the agent's |
| `cals_s0` | the stale-plant gate: bootstrap solvability on excluded-requiring instances is **0.300/0.029/0.198** at holdout 3 and **0.147/0.119/0.074** at holdout 5 — low but nonzero, so no halt; plant headroom 0.056 (h=3) vs 0.287 (h=5), against corpus keep-rates 0.603 and **0.197** | holdout ∈ {3, 5} both admissible; run both rather than argue |
| `calr_h3`, `calr_h5`, `calr_h0` | **the stale-plant manipulation failed.** `A_true` first5→last5 across 16 cells: h=3 era 1 **+0.054, +0.044, +0.062, +0.012** (backwards in 4/4); best cell anywhere **−0.032 against 0.287 of headroom (11%)**, at 2–3σ and not consistent. And the holdout was **suppressing** the descent it was meant to create: `never_base` era-1 span **0.691 at h=0** vs 0.199/0.203 at h=3/h=5 | `plant_holdout` = **0**. Carrying a demonstrably failed manipulation into the main run would buy a confound and cost the necessity ladder |
| offline detector replay | at (0.06, 0.15, 5, 2) the certificate lands **+0.000 to +0.033 of the run's own minimum** in every `never_base` cell, and is insensitive to c_v ∈ {0.10, 0.15, 0.20}. **`lp_min_drop` is load-bearing**: at h=5 era 1, drop = 0.05 fires at **c7, +0.053 above min (mid-descent)** while drop = 0.10 fires at **c19, +0.006** | c = 0.06, c_v = **0.15**, W = 5, hold = 2, `lp_min_drop` = **0.10**; `mine_cap` = **8** (span 0.691 vs 0.402 at 24, and it starts from an empty table); `gen_lr` = **1e-4** (3e-4 is worse on every axis) |
| — | the certificate fires at c19–c21, which in a 24-cycle era is ~83% through and nearly coincident with `practice_late` | `era_cycles` 24 → **30**, `late_offset` 2 → **3**, so early/gated/late separate at c1 / ~c20 / c28. Without this the poison and rent-paying comparisons would have been unresolvable by construction |

Two corrections worth stating separately because they are about method, not knobs:

- **`calp_s0`'s ceiling movement did not reproduce.** Round 1's precheck measured the committed-unit
  ceiling moving 31–66× the metering noise floor under generator fine-tuning; here the same recipe moves
  a single fixed operator not at all, and leaves clean-config accuracy flat. Stated as **hypothesis**:
  `headroom_cell` takes a **max over all 1- and 2-move programs per root**, which is far more sensitive
  to small generator changes than one operator is. Untested.
- **`given` was the wrong probe.** Three of four calibration arms used `given`, which is the *expert*
  regime — it already produces near-optimal repairs, so its first-cycle mined table is already good and
  its candidate audition collapses onto the plant within 1–4 cycles. The practice arms start base-only,
  so `never_base` is the right probe, and it is the cell where the descent is 8.9σ. Round 1's
  expert/learner reading, arriving as a fact about instrument choice.

## Runs on disk

| tag | what it is |
|---|---|
| `smoke0`, `smoke1` | attached smokes (`--quick`), the second with a plant holdout |
| `call_s0` | the depth ladder: base / +L2 / +L2+L3 at widths 1–16, the exact-DP floor, one true macro action, and the coverage curve at 1/2/4/8/¼/½/¾/full entries |
| `calr_s0` | the first descent calibration — `mine_cap` × `gen_lr` on the generator read; **negative**, and the run that exposed the precision problem |
| `calp2_s0` | the perception diagnostic: three reads (unmasked / masked-elsewhere / exact map) graded on identical exact-DP-solved repairs |
| `cals_s0` | the stale-plant gate: holdout ∈ {0, 3, 5}, reader vs generator read accuracy, `frac_excluded`, and the bootstrap-solvability split |
| `calr_h3`, `calr_h5`, `calr_h0` | the descent calibration on the fixed substrate, one per holdout |
| `rr_s0` | the main run: 5 arms, seed 0, 3 eras × 30 cycles, complete |

## Reproduce

```bash
cd experiments/            # MODAL_PROFILE=chromatic

# gates C-M / C-R / G-D (CPU)
modal run rhm/practice/ratchet/ratchet.py::selfcheck_remote
# smoke
modal run rhm/practice/ratchet/ratchet.py::ratchet --quick --tag smoke0

# calibrations
python3 rhm/practice/ratchet/launch_detached.py --fn cal_ladder --tag call_s0 \
    --n-ref 512 --widths "1,2,3,4,8,16" --g-budgets "33,49,58,83,86,100,108"
python3 rhm/practice/ratchet/launch_detached.py --fn cal_parse --tag calp2_s0 --n-ref 512
python3 rhm/practice/ratchet/launch_detached.py --fn cal_stale --tag cals_s0 \
    --holdouts "0,3,5" --n-ref 512 --g-budget 58
python3 rhm/practice/ratchet/launch_detached.py --fn ratchet --tag calr_h0 --plant-holdout 0 \
    --arms "given:mine_cap=8,never_base:mine_cap=8,never_base:mine_cap=24" \
    --eras "1:6,2:3" --era-cycles 24 --g-budget 58 --mine-support 3 --probe-every 8

# the main run
python3 rhm/practice/ratchet/launch_detached.py --fn ratchet --tag rr_s0 --seed 0 \
    --arms "never_base,given,practice_gated,practice_early,practice_late" \
    --eras "1:6,2:3,3:1" --era-cycles 30 \
    --budget 4 --pr-width 16 --g-budget 58 --max-macro-level 3 \
    --n-pr 64 --n-rt 384 --n-score 512 --n-grad 4 --value-lr-online 3e-5 \
    --gen-lr 1e-4 --gen-steps 20 --plant-holdout 0 \
    --mine-from chosen --mine-cap 8 --mine-support 3 \
    --sil-c 0.06 --sil-cv 0.15 --sil-win 5 --sil-hold 2 --sil-min-cycle 6 --lp-min-drop 0.10 \
    --early-offset 1 --late-offset 3 --probe-every 4

# reduction
python3 rhm/practice/ratchet/analyze_ratchet.py --tag rr_s0 --fetch --figures
python3 rhm/practice/ratchet/analyze_ratchet.py --tag call_s0 --fetch --cal
python3 -c "import sys; sys.path.insert(0,'rhm/practice/ratchet'); \
    import analyze_ratchet as A; A.unitlp_grid('calr_h0', level=2, era=1)"
```

Modal volume (`rhm-scaling-data`): `/data/rhm_practice_ratchet/<tag>/<arm>/results.json`, with
`setup.json` beside them and `cal_ladder.json` / `cal_parse.json` / `cal_stale.json` for the
calibration entrypoints. Figures: `figures/rr_s0/fig1_competence.png` (competence per cycle and
against priced time, era boundaries and commits marked), `fig2_unitlp.png` (the level-2 and level-3
shadow-audition trajectories against the true-table floor, commits marked), `fig3_vocab.png` (table
recall against the DGP's own vocabulary).

## Caveats

- **One rule draw**, one (v, s, L, m) setting, one damage ladder. The earned-vs-given
  fractions and the poison result are large relative to the measured noise; the compounding *rate*
  ordering is not, and is reported as unresolved.
- **The level-3 audition is mis-levelled**, by construction: the macro is auditioned on era-k damage
  and consumed on era-k+1 damage. This caps its dynamic range near 0.51 and is the direct cause of the
  certificate's refusal. Every level-3 certificate statement in this node is a statement about that
  audition, not about level-3 units in general.
- **The plant is inert here.** The round's apparatus assumed a learning plant and got one that does not
  learn; every conclusion about the certificate is therefore a conclusion about a **vocabulary**
  carrier. Whether the same certificate behaves the same way with a genuinely moving executor is
  untested.
- **The stale-plant intervention failed and is not part of the main run.** Its data is kept because the
  failure is informative, not because the manipulation worked.
- **The declared budget sets the headline.** At G = 58 vocabulary dominates width; the full ladder is
  logged at every probe cycle so that dependence stays visible, but a different budget gives a
  different verdict and the choice is ours.
- **The reader is handed over.** Bottom-level perception — mapping a leaf tuple to its level-1 feature —
  is given, trained on the same corpus and the same supervision the generator already uses. Everything
  above level 1 is earned. That line is a design decision, and a different line would give a different
  earned-vs-given fraction.

## Next steps

**(i) The evaluation-layer-climbing round — the priority.** Re-instantiation is the last of the idea
doc's three components never built, and this round located exactly where it bites: the agent's
level-3 grader was too noisy and too mis-levelled to certify anything, so it under-committed and lost
the era. Two mechanisms, both cheap on this apparatus:

- **Provisional commitment.** Commit cheaply at era boundaries, grade the unit **live in the real
  level-k+1 contexts** where it is actually consumed, and keep a priced recert. This replaces the
  mis-levelled proxy with the consumption distribution itself, which is finding 6's fix stated as a
  mechanism.
- **Self-manufactured audition contexts.** The agent damages *its own clean derivations* at level k+1
  and auditions there — practice manufacturing its own tests, which is component (1) of the definition
  applied to the grader rather than to the task. RHM makes this oracle-checkable: we know whether the
  self-manufactured contexts match the real ones.

**Caution to build in**: low-level grading retires **per certified unit, not globally**.
`practice_early`'s catastrophe was only visible because the `held` audition kept being measured after
the commit; an agent that stops grading a level once it has certified there would have been blind to
it.

**(ii) The plant-frontier round.** Funding the crossing, which this round measured as the blocker:
unfiltered replay, or a raised fresh-data fraction, so the agent can actually learn the structure its
corpus withheld. The gate stays the same — bootstrap solvability low but nonzero.

**(iii) Rate-compounding.** Currently blocked on level-3 audition noise (0.04–0.09 against spans of
0.05–0.30). Likely unblocked by (i), since a consumption-matched audition should have both a larger
dynamic range and a lower floor.
