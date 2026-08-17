# typed_gaps — the conditioning gap is typed: transpose × setlist, a crossed pair

**Up**: [../README.md](../README.md) (practice arc)
**Idea doc**: [practice_manufactures_its_own_credit](../../../../ideas/practice_manufactures_its_own_credit.md)
§18 (the synthesis this pair was built to test) and §19 (the pointer back here)
**Machinery + calibration records**: [`../transpose/FILES.md`](../transpose/FILES.md) ·
[`../setlist/FILES.md`](../setlist/FILES.md) (per deliberate policy those nodes carry no README —
this joint writeup is their findings record)
**Status**: written up 2026-08-16 (interpretation discussed with Jasper throughout, same day).
Two rounds × (offline calibration + GPU admissibility gate + bit-identical fidelity replay +
main run). **Single seed**; both rounds implemented by one Opus subagent session.
**Dates**: 2026-08-16.

## One-liner

Two worlds that differ in exactly one currency of change — `transpose` drifts **what is true**
(grammar rule cells resampled mid-run, demand provably fixed), `setlist` drifts **what is asked**
(OU drift on the derivation distribution, truth provably fixed) — produce a **crossed double
dissociation**: truth-drift degrades the dense learner (plant parse 0.62→0.50, infill 0.71→0.57)
and leaves the post-commit verification channel silent (recert 1 useful swap in 62), while
demand-drift leaves the dense learner flat (0.59→0.60, 0.75→0.71) and wakes the verification
channel decisively (9/62, every swap coverage-improving; ablating it costs **Δ0.271 deep error,
≈8× stream noise**). Committed chunks are not beliefs: nothing that went *false* touched them
(a frozen table at precision 0.643 tracked the current truth to within ±0.05), and everything
that went *unasked-for* did (frozen concentration falls to 0.345–0.754 earned-vs-given while
tracking concentration posts 1.000–1.238).

## The question

§18 of the idea doc recorded two demotions of the arc's introspective machinery — the certificate
to a within-era instrument, the recert safety net to apparent dead weight (0/24 in `ear`, 0/288
in `recital`) — and this conversation's synthesis raised a caveat: every round so far ran in a
world where **post-commit invalidation is structurally impossible** (static grammar, i.i.d.
damage, inert plant; four independent instruments say so — the recert nulls, `tall`'s
table-cannot-churn, the étude's exactly-0.0000 post-commit drift, `crystallize`'s
nothing-to-certify). The conditioning-gap frame: a self-model is epistemically load-bearing only
where the world can deliver news its predictions did not condition on. No round had ever combined
the compile op with a live news channel — the early mjc nodes had the gap but no compile op;
étude/RHM had the op but a static gap. The pair fills that cell twice, once per candidate
currency of news.

## Round 1 — [`transpose`](../transpose/FILES.md): truth-news (2026-08-16)

**Lever**: hard resampling of grammar rule cells at a chosen level, unannounced, level-indexed,
bottom layer untouched (reader/generator parse stays exact). A `DecayMiner` (geometric count
decay, `= MC.Miner` exactly at γ=1.0) gives the vocabulary a way to forget — the structural item
`tall` proved missing. Fidelity: `tf_s0` reproduces `ear/er_s0` **bit-identically** (8 arms × 90
cycles × 10 log fields, max|Δ| = 0).

**The admissibility finding that set the specification** (`cal0`): **level-2 drift is
inadmissible** — arm separation collapses to the ±0.034 noise floor within one or two events
(negative twice) while the stale reference runs to 0.81–0.95 and the exact-DP floor itself
degrades 0.195→0.61 — the substrate failing, `tall`'s voided-run signature caught pre-hoc.
**Level-3 drift is admissible** (separation +0.095…+0.190 at every epoch, at or above the
undrifted control's +0.127). Mechanism: a depth-4 tree has four level-2 nodes but two level-3
nodes, so a level-2 cell sits in twice as many derivations (per-event corpus survival 0.71 vs
0.86) and is what base rendering and generator infill lean on. *Drifting level 2 degrades the
executor; drifting level 3 moves the vocabulary.* A side benefit: gate T-2 makes the level-2
vocabulary bit-identical under level-3 drift, so every arm carries a paired within-arm control —
one committed table that provably cannot go stale beside one that can.

**Main run** (`tp_s0`: 8 arms, drift level 3, 1 cell per 8 cycles from c33, era 1 pristine,
γ=0.95). The world moved: L3 table survival → 0.643, corpus survival → 0.301, era-3 stale
0.68→0.92, floor 0.57→0.66, plant parse 0.62→0.50 / infill 0.71→0.57. Findings:

1. **The static ordering is preserved.** Terminal deep cell: `practice_climb` .5859 (e-v-g
   1.137) > `practice_prov` .6068 (1.053) > `climb_decay` .6094 > `prov_norecert` .6146 >
   `given` .6198 > `given_live` .6510 > `practice_gated` .7734 (0.379) > `never_base` .8672.
2. **The flat post-commit baseline broke, but by world-hardening, not invalidation.** `held`
   rises +0.52–0.55 at L2 (where the table provably cannot be stale) and only +0.04–0.13 at L3
   (where it can); (held − true) stays within −0.072…+0.047 with inconsistent sign. `given`'s
   perfect frozen L3 table, 36% invalidated (precision 1.000→0.643), ends at −0.006 vs the
   current truth.
3. **The verification channel stayed near-silent**: 1 useful recert swap in 62;
   mean(e_frozen − e_live) 0.003–0.008. The ablation (`prov_norecert` vs `practice_prov`) read
   Δ0.008 — inside noise — though that isolation is confounded (different stream positions and
   mined tables).
4. **Tracking the truth bought nothing measurable**: frozen `given` ≥ `given_live` on every cell
   (Δ0.031–0.043, at the ±0.023–0.034 stream-noise floor, consecutive arms — read as "no
   advantage to tracking", not "frozen wins").
5. **Unanticipated, and the round's sharpest cell**: the mined *stale* L3 table beats
   matched-size random subsets of the *current* true table (−0.10 to −0.13, 54–60/60 cycles) and
   `prov_norecert`'s frozen L3 table beats the **full current true table** in audition by
   0.037–0.061 throughout. Concentration dominates currency at this drift magnitude.
6. Forgetting did not pay (`climb_decay` −0.023 vs `practice_climb`, at noise), and monotone
   miners removed 0 entries — `tall`'s churn claim reproduces under drift.

Apparatus notes: the `recovered` normalizer is unusable here (macros beat the base-move DP floor,
so stale/floor is not a denominator); a reducer bug in the grader-cost column was found and fixed
(the corrected reduction reproduces `ear`'s published 29.2% for `never_base` exactly).

## Round 2 — [`setlist`](../setlist/FILES.md): demand-news (2026-08-16)

**Lever**: the grammar is fixed; the **consumption distribution** drifts — OU motion on the root
prior and the rule-mixture weights on the layers *above* the mined vocabulary (`demand_levels`
0/1; a level-2 synonym cell is a measured null by construction, since either synonym still
derives its feature). This is `full_loop`'s drift machinery, which `transpose` explicitly
rejected because it "cannot invalidate a committed macro — every entry stays grammatical, only
its frequency moves": that disqualifying property is this round's requirement, and because
support never changes the KL is finite, so magnitude is calibrated in nats **of demand on the
mined vocabulary**. Calibration lessons recorded in the node: the σ-per-event-KL objective is
non-monotone (softmax saturation), so σ (demand concentration) and κ (mixing rate) must be swept,
not solved; and a single stationary prewarm draw is a lottery — the seed-0 draw was an entropy
outlier (1.856 vs process median 2.250) that made the whole calibration sweep trend, diagnosed
because trajectories were near-identical across different σ, which no drift property can produce.
Fixed by `typical_demand` (median-entropy of 48 candidates, spread logged).

**Admissibility** (`cal0`): σ=2.5 inadmissible (separation 0.021, below noise); **σ=1.0
preserves separation at the no-drift level** (0.123 vs 0.121) while demand genuinely moves
(frozen epoch-0 concentration covers 0.807→0.600 of demand over 40 cycles). Gates: **D-2 —
precision-vs-truth is invariant (0.800) across 8 demand epochs; nothing becomes false**; D-6 —
the full true table's coverage of demand is flat at 0.82–0.87, licensing `given` as the
demand-invariant pure-coverage reference. Fidelity: `sf_s0` bit-identical to `ear/er_s0`, as
above.

**Main run** (`sl_s0`: 9 arms, σ=1.0, κ=0.15, one event per 4 cycles from c6, 23 epochs;
per-event demand-KL 0.014–0.157 nats; frozen epoch-0 concentration troughs at 0.674 coverage).
Findings:

1. **The verification channel fired, coherently**: **9 recert swaps in 62** (vs 1/62 under
   truth-drift, 0/312 static), **every swap improving demand coverage**, landing at the same
   cycles in all three recert arms. Example (`practice_prov`): L2 c65 e .328→.266, coverage
   .638→.732; L3 c65 e .266→.188, coverage .113→.232; L3 c90 e .380→.203.
2. **The ablation separated decisively**: `prov_norecert` vs `practice_prov` deep **.4948 vs
   .2240** (Δ0.271 ≈ 8× noise; mean .3455 vs .1858), with a clean isolation this time — both
   committed at the same cycles with near-identical tables. Mechanism in coverage: with recert,
   committed L3 coverage .124→.161 (recovered); without, .134→.043 (decayed).
3. **The oracle bracket prices concentration directly** (same construction, same size, only
   tracking differs): `demand_live` earned-vs-given **1.000/1.238/1.116**; `demand_frozen`
   **0.345/0.607/0.754**. Tracking concentration beats full coverage; frozen concentration loses
   to it.
4. **The typed pair held in the loop**: `given`'s precision-vs-truth pinned at 1.000→1.000 at
   both levels while its coverage-vs-demand moved (.854→.821 L2, .682→.661 L3); mined precision
   static except where recert swapped. One currency pinned, the other moving — the two value
   axes are dissociated in the data, not only in the design.
5. `practice_prov` finished **rank 1 overall** (.1858 mean), statistically matching the tracking
   oracle (deep .2240 vs .2448, inside noise) at a grader cost of **7.9%** of priced time.
6. `practice_gated` refused L3 for the **third consecutive round**; abstinence again did not pay
   (deep .5625, e-v-g 0.527, at the round's highest grader share, 49.8%).
7. **Maintenance is evaluative, not entropic**: monotone miners again removed 0 entries;
   `climb_decay`'s forgetting *cost* coverage (.555 vs ~.70 at L2). What works is
   audit-and-reselect, not decay. (The `climb_decay` vs `practice_climb` grade comparison is
   confounded by commit cycle/type; the coverage mechanism is the readable part.)
8. Residual invalidation signal: `held − dem` at L3 is +0.102 for `prov_norecert` vs
   +0.038/+0.057 for the recert arms — small beside the world-level `held` rise (+0.44–0.81) but
   sign-consistent, unlike `transpose`'s.

Apparatus notes: the OU mean-reverted, so epoch-0 demand was partly back in fashion at the
terminal exam (frozen coverage .903 at c90) — measured erosion is **understated**; the per-epoch
premium trajectory is dominated by era/commit structure and is not readable as erosion (the
oracle bracket is the clean measurement); `bank_fresh` is a dead instrument; the grader-ageing
signal (mfg−real −0.01 → +0.10…+0.14 with age) is collinear with within-era position
(`never_base` shows the same pattern) and cannot be attributed in this design.

## The dissociation

| | truth-drift (`tp_s0`) | demand-drift (`sl_s0`) |
|---|---|---|
| plant parse | 0.62 → **0.50** | 0.59 → 0.60 |
| plant infill | 0.71 → **0.57** | 0.75 → 0.71 |
| recert swaps (of 62) | 1 | **9** |
| recert ablation, deep | Δ0.008 (noise) | **Δ0.271 (≈8× noise)** |
| tracking the moving quantity | worth ≤ noise (`given_live`) | worth 0.36–0.66 e-v-g (bracket) |

Each species of news moved exactly one organ and left the other flat, in both directions.

## Interpretation (discussed with Jasper, 2026-08-16)

- **A chunk is not a belief.** It has no truth conditions to violate — `transpose` ran a
  falsification test on content that turned out not to be propositional, and the content
  correctly ignored it (truth-news landed on the dense learner instead, the one component that
  does store world-content). What a chunk stores is **demand-concentration**: a leveraged bet on
  what the world asks, whose full sign structure `setlist`'s bracket measured — current
  concentration > coverage > stale concentration — with coverage (`given`) as the unleveraged,
  demand-invariant hedge. A mechanism hypothesis for the grace under truth-drift, unproven here:
  state-conditioned commitment (round 1's boundary fix) lets launch-time selection route around
  dead entries, so stale content falls out of *use* before it can misfire.
- **The §18 demotion splits three ways.** Pre-commit certification stays demoted everywhere
  (three consecutive frontier refusals; boundary decisions belong to provisional commitment,
  consumption grading, and teachers). Post-commit verification is **rehabilitated,
  currency-specifically**: dead weight wherever demand holds still, decisively load-bearing the
  moment it moves — §18's "substrate-conditioned" caveat resolved, with the condition now named
  and priced (≈8% of feedback budget to match a perfect tracking oracle). Maintenance of
  committed skill is **demand-tracking, not truth-tracking**, and its working mechanism is
  selection, not decay — the arc's oldest law, at both ends of a chunk's life.
- **The conditioning gap matures from a binary into a type system.** The gap must be *aimed*
  (level-2 grammar drift and σ=2.5 both destroy the experiment rather than testing the
  vocabulary — news too loud, or in the executor's currency, is weather) and its news must be
  *denominated in the currency the component stores*. One organ per currency, two cells of the
  map now measured in both directions here, two inherited: **dense learning ↔ truth-news**
  (measured here), **repair/metering ↔ interface-news** (the mjc bridge arc, in retrospect),
  **evaluative re-selection ↔ demand-news** (measured here), **the teacher ↔ level-news**
  (`recital`/`tall`: the one currency whose signals are undefined from below). The corollary for
  monolithic learners is the degenerate case, not a counterexample: a training regime whose
  world drifts in a single currency needs the single matching organ.

## Caveats

Single seed throughout — per `recital`'s methodology export, **rank orderings are the currency**
and single-run fractions carry ±0.15; the claims above rest on ranks, on contrasts ≥8× the
measured stream noise, and on a dissociation that spans two independent runs, but a seed pair on
both main runs is the obvious next hardening. The dose reading of `transpose` (drift too small
for invalidation to bite) is formally alive, though finding 5 — stale concentration *beating*
the current truth — is not what a dose account produces. Both admissibility findings
(level-2/level-3; σ=2.5/σ=1.0) are substrate-scoped measurements, not general laws.

## Reproduce

```bash
cd experiments/            # MODAL_PROFILE=chromatic

# transpose: selfcheck → offline calibration → descent gate → fidelity → main
modal run rhm/practice/transpose/transpose.py::selfcheck_remote
modal run rhm/practice/transpose/transpose.py::cal_drift_remote
python3 rhm/practice/transpose/analyze_transpose.py --tag tf_s0 --fetch --fidelity
python3 rhm/practice/transpose/analyze_transpose.py --tag tp_s0 --fetch --figures

# setlist: same shape
modal run rhm/practice/setlist/setlist.py::selfcheck_remote
python3 rhm/practice/setlist/analyze_setlist.py --tag sf_s0 --fetch --fidelity
python3 rhm/practice/setlist/analyze_setlist.py --tag sl_s0 --fetch --figures
```

Full launch commands with exact flags: [`../transpose/FILES.md`](../transpose/FILES.md) §Reproduce
and [`../setlist/FILES.md`](../setlist/FILES.md) §Reproduce. Volumes:
`/data/rhm_practice_transpose/<tag>/` and `/data/rhm_practice_setlist/<tag>/` on `rhm-scaling-data`;
figures under each node's `figures/<tag>/`.
