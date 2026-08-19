# fourwall — the spurious index: scaffold, debt, merge, re-key, retire (fw_s0–fw_s3)

**Up**: [../README.md](../README.md) (the practice arc)
**Idea doc**: [recurrence_manufactures_confounds](../../../../ideas/recurrence_manufactures_confounds.md)
— this node is that doc's first contact with data. **Spec sibling**: [`../merge/SPEC.md`](../merge/SPEC.md)
(this node builds a narrowed slice; see FILES.md *Scope*).
**Files & machinery**: [FILES.md](FILES.md) — arms, gates, calibration record, Gate 0 in full, the
coverage-is-not-the-objective machinery note, and everything inherited from
[`../ratchet/`](../ratchet/README.md) / [`../setlist/`](../setlist/FILES.md).
**Dates**: 2026-08-17 (all four tags). **Seeds**: single seed, one rule draw; **rank orderings and
signs are the reported quantities** — and this node *measured* why (see Caveats: stream-position
sensitivity).

## Scope

`transpose` moved **what is true**. `setlist` moved **what is asked**. This round holds both fixed
and moves the **index**: every instance carries a free surface token `w` that, in phase 1, is a
perfect correlate of the true level-2 latent `f` — the "wall" of the idea doc's line-dance anchor —
and a rotation event then permutes the `w`↔`f` map (never deleting `w`, so a raw `w`-key becomes
actively misleading). Nothing ever becomes false and demand never moves; the rotation is a
relabelling of an observable no grading path reads, so admissibility is by construction.

Questions carried in from the idea doc: does a spurious-but-free key act as a **scaffold** (§3)?
Is the confound **debt** real (§2)? Does a **merge** op (§5) recover the compressed form? Does the
binding form **spontaneously** (§2's available-vs-taken caveat)? Two further ops — **re-key** and
**retire** — were not in the doc and were forced by the results, which is the round's main
epistemic content: *"whittling toward the purest form" decomposes into three distinct operations,
of which the doc had named one.*

**Gate 0** (full record in FILES.md): the design is admissible — a right index is worth **+0.131**
policy error (≈3.9× the arc's ±0.034 stream-position floor) and a perfectly keyed table one-shots
the damage cell exactly — with one caveat that scopes the scaffold half: the true key is linearly
decodable from the agent's state at 0.927 (given 16k oracle labels the agent never receives) while
the mechanism that consumes it recovers it at ~0.52–0.59. So this substrate instantiates the
**weak form** of §3 — *a free key the credit machinery is not wired to use* — not the strong form
(*a key not yet representable*). Index instruments are untouched by this; it conditions only how
phase-1 scaffold results are read.

## Arms across the four tags

The library key is the only difference between arms; the full table with the `unkeyed` confound
control's rationale is in [FILES.md](FILES.md).

| tag | arms | op under test |
|---|---|---|
| `fw_s0` | `given_key` · `wall_track` · `wall_merge` · `unkeyed` · `free_selector` | **merge** (deletion), spontaneous binding |
| `fw_s1` | `given_key` · `wall_merge` · `wall_rekey` · `scratch_rekey` | **re-key** (voided by its gate; see below) |
| `fw_s2` | same as s1, corrected gate | **re-key** (worked) |
| `fw_s3` | `wall_rekey` (in-tag ref) · `rekey_mothball` · `rekey_delete` | **retire** |

Schedule, identical in every tag: phase 1 = c1–50 (`w` a bijection of `f`), then an unannounced
rotation every 10 cycles to c130, with the map returning to identity at **c121** — a free "does a
stale cell come back into fashion" instrument. Same held-out instances at every cycle of every arm.

---

## fw_s0 — the scaffold is real, the debt is real, and deletion cannot build

**The scaffold delivers oracle-index value for free, while the correlation holds.** Phase-1
converged error: `given_key` 0.1375, `wall_track` 0.1443, `wall_merge` 0.1453 — the three keyed
arms indistinguishable (noise floor ±0.034) — against `unkeyed` 0.2682. A free surface key buys
≈+0.10–0.13 of Gate 0's +0.175 oracle bracket. One refinement: at *coarse* competence thresholds
the unkeyed arms are faster (t(e≤0.5): 4.46e4 vs 6.89e4) — splitting evidence across 8 cells costs
early and pays late.

**The debt arrives on schedule.** Rotation response (Δ mean e, 3 cycles after − before, mean abs):

| given_key | wall_track | wall_merge | unkeyed | free_selector |
|---|---|---|---|---|
| **0.007** | **0.260** (+0.19…+0.41 per event) | 0.126 | 0.003 | 0.011 |

`given_key`'s rotation-invariance-by-construction is confirmed empirically; the c121 identity
return is the only rotation that costs `wall_track` ≈nothing (+0.017) — a stale wall cell is
mis-addressed, not wrong, and comes back into fashion when demand returns.

**Merge does what deletion can do, and no more.** Index 8 → 2 by c100 (6 merges, dedup-union,
never an average); rotation response extinguished after the first two events; stored-per-served
5.43 (`wall_track`) → 1.86; maintenance spend cut by a third. But on the phase-2 mean only
`given_key` (0.1521) beats the unkeyed arms (0.2277–0.2550); `wall_merge` (0.2927) converges back
to ≈`unkeyed` by the end. **Deletion can remove a misleading index; it cannot build a true one** —
the fully-merged arm lands at the no-index baseline, not at the oracle, because no coarsening of
the `w`-partition equals the `f`-partition once the two are decorrelated.

**No spontaneous binding.** `free_selector`'s wall-permutation probe reads +0.010 in phase 1
(keyed arms: +0.34) despite a 20× lr on its wall embedding, set precisely so a null would not
measure the optimisation budget. On this substrate, §2's caveat resolves to *available, not
taken* — plausibly because the trained pathway to the configuration already carries the key
(Gate 0's 0.927), so the shortcut has no gradient advantage. The lr caveat travels with the null.

**Other**: the exact forced-transfer profile at c51 is **binary** (entries go 1.000 → 0.000, no
intermediate), so the idea doc §5's frame-contamination/repair decomposition is unreadable on a
discrete-program substrate. `given_key` spends 15.9% of priced time on recert and swaps **0/174**
— verification is pure overhead where the index is right. Lifetime priced-performance `A/t`:
given 0.838 > free_selector 0.755 > wall_merge 0.750 > unkeyed 0.734 > wall_track 0.707 — over
the whole run, scaffold-then-merge beats never-indexing, and never-merging is last.

## fw_s1 — the re-key round that never migrated, and what it taught anyway

The re-key op (functional addressing: *two contexts are in the same class iff the same committed
unit serves them* — merge's pairwise alias test, closed into a partition; machinery in FILES.md)
was built and never used: **the migration gate opened 0 times in 130 cycles in both arms**. Both
carrying comparisons were therefore untested-as-designed. Two mechanisms, both instrument-level
and both now recorded as reusable lessons (FILES.md machinery note):

1. **The gate graded coverage, not consumption.** It asked "does *any* entry of the incumbent
   cell repair this instance" — monotone in table size — while the arms are graded on policy
   error. A stale, re-mining cell reads 0.895 by enumeration while producing 0.338-error policy.
   The seam law's evaluator coordinate (`ear`), applied to the index; `ratchet`'s
   concentration-not-coverage, restated for the addressing layer.
2. **Hysteresis vetoed the one correct firing** (+0.137 at c55, four cycles post-rotation;
   persist=2 demanded a repeat that re-mining foreclosed by c60). The tracking organ racing the
   migration organ — patching preempting restructuring — is real, and a certify-then-adopt rule
   loses that race by construction.

What the voided round still measured: the earned basis forms **immediately** (c5, the first
check) from almost no units; it plateaus at ARI 0.52–0.64 against the true partition with 4–5
classes against 7 live features — **the functional quotient is as fine as your units and no
finer** (a weak cell is *genuinely* functionally aliased with its neighbour, which is also why
the phase-1 merge of the starved cell recurred at identical numbers under the evidence floor).
The address ladder is capped by the action ladder — `recital`'s currency law appearing in the
machinery itself.

## fw_s2 — migration works; the scaffold's stickiness is rational patience

Two changes only (FILES.md has the exact knob diff): the gate graded in **held-out consumption
policy error under both routings**, and certify-then-adopt replaced by **provisional adoption +
a per-class revert net** (`ear`'s shape, ported to the index).

| | e_p2_mean | e_last10 | gap vs given | A/t | rotation response |
|---|---|---|---|---|---|
| given_key | 0.1521 | 0.1474 | — | 0.8383 | 0.007 |
| **wall_rekey** | **0.2136** | **0.2120** | **+0.065** | **0.7926** | **0.019** |
| scratch_rekey | 0.2159 | 0.2365* | — | 0.7781 | 0.006 |
| wall_merge (deletion-only) | 0.2986 | 0.2451 | +0.098 | 0.7441 | 0.129 |

(*scratch e_last10 from fw_s3's discussion of metric sensitivity; see Caveats.) `wall_rekey`
became rotation-proof (0.208 → 0.019, within 3× of the oracle's invariance-by-construction),
closed about a third of the terminal gap deletion could not close, and posted the best non-anchor
lifetime integral to that point. Migration is a **step, not a curve** (0 → 0.97 across one cycle
at adoption); the revert net fired once per arm, both correct, mean margin strongly favouring the
adopted basis — `ear`'s 0/24 shape again: selection upstream leaves the safety net nearly idle.

**Adoption timing**: `scratch_rekey` adopted at c5, `wall_rekey` at c40 — 8.6× later in priced
time — and the gate series shows why: adoption is graded *relative to the incumbent* (0.328 vs
0.141), so **a good scaffold raises the bar for its own replacement**. The delay was economically
correct (better phase-1 error *and* better lifetime integral than scratch). Two readings recorded,
one now weakened: the *mechanism* (bar relative to incumbent quality) stands on the gate series;
the *magnitude and ordering* of the 8.6× is *not* seed-stable (see Caveats — fw_s3 measured the
same config adopting at c5 from a different stream position).

Also on the record: **tear-down preceded stress** — adoption happened in phase 1, before any
rotation, on pure consumption-graded merit; and the **provisional/collapse path never fired in a
main run** (0/26 in every arm, both rounds) — the *past-facing* certificate (comparing two built
things on held-out consumption) did all the work, where the arc's demoted certificates were
*future-facing* (forecasting headroom). Recorded as a distinction, not yet as a law.

## fw_s3 — retire: the rent is real, the fallback is real, and the choice between them is a bet on demand

`fw_s2` left `wall_rekey` paying 16% of priced time recerting wall cells that route ~7% of
instances. `fw_s3` adds retire in two settings — `rekey_mothball` (content kept and servable,
maintenance stopped: isolates the **rent**) and `rekey_delete` (content removed: isolates the
**fallback**) — with `wall_rekey` re-run in-tag as reference after the cross-tag freeze check
failed (torch global-RNG stream position; FILES.md). Both arms retired all 8 wall cells by c15 —
again in phase 1, before any stress.

| | e_p2_mean | e_last10 | A/t | recert % | migration end | rotation resp. |
|---|---|---|---|---|---|---|
| wall_rekey (ref) | 0.2422 | **0.1646** | 0.7686 | 16.00 | 0.596 | 0.113 |
| rekey_mothball | 0.2173 | 0.2302 | 0.7826 | 11.42 | 0.971 | 0.011 |
| rekey_delete | **0.2053** | 0.2365 | **0.7948** | 10.08 | 0.979 | 0.012 |

**The rent claim survived intervention.** Retiring cuts recert 16 → 10–11%; the freed budget was
*not* reinvested (practice+metering flat at 71–73%) — it appears as lower total cost with error
simultaneously improving, which is why A/t rises monotonically in how much scaffold is dropped.
The maintenance was rent, not purchase.

**Routing-not-pruning survived too — on its population.** For the 2–3% of instances the router
refuses, deleting the beacon is far worse than mothballing it (post-retire fallback error 0.554
vs 0.405; 0.750 vs 0.182 at the end). The doc's §5 prediction is visible per-instance; `delete`
wins the aggregate only because migration left almost nobody needing the fallback.

**The c121 identity return separates the ops exactly as designed.** Mothballed content revives
(fallback error 0.51 → 0.24); deleted content cannot (0.43 → 0.61); and only the arm that kept
its scaffold *maintained* converts the return into an aggregate gain (−0.137 — the single event
that inverts the e_last10 ranking). The two error metrics disagree because they encode different
demand futures: **keeping a maintained scaffold is insurance against the old regime's return, and
c121 is the claim event.** In a world that drifts away forever, retire-and-delete wins; in one
that cycles home — a four-wall dance returns you to your original wall every fourth repetition —
the beacon keeps paying. Retire's mode is a priced bet on the demand process's recurrence
structure, which is `typed_gaps`' forgetting question (decay costs coverage; audit-and-reselect
works) at the scale of a whole basis.

**Suggestive, stream-confounded**: the retired arms completed migration (0.97+) where the
reference kept 40% of traffic on its maintained wall cells (0.596) and stayed rotation-fragile —
an available, well-maintained fallback may *suppress* commitment to the new basis. Held loosely;
migration depth itself varies 0.60–0.93 across stream positions at identical config.

---

## Synthesis (agreed interpretation, 2026-08-17; doc revision queued, not yet applied)

1. **"Whittling toward the purest form" is three ops, not one.** *Merge* deletes distinctions
   within the current basis — necessary, cheap, and strictly bounded: it recovers the no-index
   baseline, never the truth. *Re-key* re-expresses the library in an earned basis — where the
   terminal value lives, and adoptable on pure merit when graded in consumption. *Retire* stops
   paying for the abandoned basis — pure rent recovered, at the price of a fallback whose value is
   set by demand recurrence. The idea doc's §5 named only the first.
2. **The scaffold is a full-value loan with rational stickiness.** Free oracle-grade performance
   while the correlation holds; a heavy per-rotation bill after; and a replacement bar set by its
   own quality. Graded in the wrong currency (coverage), stickiness is a permanent trap; graded in
   consumption, it is optimal patience. The difference is entirely the evaluator.
3. **The quotient you can earn is bounded by the vocabulary you have earned.** Functional
   addressing caps at the resolution of the unit library (ARI ~0.5–0.6 here); truth-grade
   structure is funded only by better units. This is the skill↔belief interpolation of the idea
   doc's §7 made mechanical, and `recital`'s cross-level currency law surfacing in the machinery.
4. **Under a continuous consumption-graded self-audit, tear-down precedes stress** — adoption at
   c40 and retirement by c15, both pre-rotation, in every round that had the op. The rotations
   were never the trigger for a correctly-instrumented learner; they are the forcing function for
   learners *without* the audit. Reading (conceptual, for the doc revision): the four-wall
   choreography is a cultural prosthetic for the missing audit.

## Children

### [`lm/`](lm/README.md) — the endogenous twin: the index ops asked of a real NTP reader (2026-08-18)

**Goal**: port this node's shape onto an endogenous reader (`reread/lm`'s NTP-vs-exact-BP-oracle
pattern) — a free prefix key bijective with a level-2 latent, rotated unannounced, the ops as
**readouts** rather than arms — making the idea doc §9's discreteness question well-posed, then
(round 2) supply the missing op exogenously and price it.

**Finding**: this node's binding null **inverts** (available *is* taken — instantly, to the full
exact 0.152-nat bracket); the debt is flat, the identity return costs full price (weights hold no
mothball — the old map is overwritten, not shelved), and the reader **tracks and never quotients**
at any rotation rate, ending with its token-derived inference pathway threefold suppressed
(0.27–0.30 vs controls' 0.84 against a 0.921 ceiling, `true_wall` included) — invisible to task
error. The supplied op (input-stream collapse; conditions, not weights) repairs exactly that
failure at ~zero price, later merging *dominates* earlier on the lifetime integral (this node's
rational stickiness as measured economics), and the within-level ledger votes against merging at
all — the op and its justification must both come from outside. One correction on the record:
round 1's `wall_fast` penalty was ~8× a checkpoint-grid artifact (corrected net +0.019), aligning
the endogenous reader with `mg_s0`'s substitutes-on-task-error. Single seed; per-arm workers with
identical seeds remove this node's stream-position confound, so cross-arm contrasts are licensed
there.

## Caveats

- **Single seed throughout; ranks and signs only.** And this node sharpened the arc's noise
  model: **stream-position sensitivity exceeds the ±0.034 error floor in adoption timing and
  migration depth** — identical `wall_rekey` config adopted at c5 (fw_s3, arm #1) vs c40 (fw_s2,
  arm #3) and saturated migration at 0.596 vs 0.932, purely from torch-global-RNG arm ordering.
  fw_s2's "scaffold delays the climb 8.6×" is therefore a mechanism with an unstable magnitude.
  The deferred seed pair has a specific job: the adoption-timing contrast and the
  mothball-vs-delete content contrast (which rests on 5 surrendered entries).
- **Weak-form scaffold regime** (Gate 0): the true key is representable but unrouted; the strong
  form of §3 (key not yet representable) is untested here and likely needs a substrate where the
  climb is genuinely slow.
- **Binary transfer**: contamination/repair decomposition needs a graded substrate (mjc).
- **Provisional adoption unexercised** in main runs; **the priced merge audition never changed a
  decision** (screen+hysteresis did all filtering; `merge_audit=False` is the cheaper default
  until the routes disagree).
- Cross-tag *level* comparisons are not licensed (stream position); in-tag references carry every
  load-bearing contrast.

## Reproduce

Full commands for `fw_s0`/`fw_s1`, gates, and the calibration record: [FILES.md](FILES.md).
`fw_s2` = the `fw_s1` command with the corrected gate knobs
(`--rekey-margin 0.02 --n-gate 64 --collapse-win 5 --collapse-delta 0.1 --revert-every 5
--revert-margin 0.05 --revert-persist 2 --n-revert 24`, replacing `--rekey-tol/--rekey-persist`).
`fw_s3` = the `fw_s2` configuration with
`--arms "wall_rekey,rekey_mothball,rekey_delete" --retire-every 5 --retire-share 0.02
--retire-win 10`. Every tag's exact config is in `setup.json` beside its results
(volume `rhm-scaling-data`, `/data/rhm_practice_fourwall/<tag>/`; fetched copies in
`figures/<tag>/`).

```bash
cd experiments/            # MODAL_PROFILE=chromatic
modal run rhm/practice/fourwall/fourwall.py::selfcheck_remote
python3 rhm/practice/fourwall/analyze_fourwall.py --tag fw_s3 --fetch --figures
```

## Next steps (queued, not started)

Seed pair with the two named jobs above · the graded-substrate (mjc) port for the
contamination/repair half of §5 · the strong-form §3 regime · the idea-doc revision
(§3 stickiness clause; §5 → merge/re-key/retire; §7 resolution cap; §6 the audit-prosthetic
reading) · relation to [`../merge/SPEC.md`](../merge/SPEC.md)'s unmetered dense arm.
