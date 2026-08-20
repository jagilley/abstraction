# fingering — state-conditioned commitment and the completed op taxonomy on a redundant arm

**Up**: [../README.md](../README.md) (practice) · [../../README.md](../../README.md) (mjc)
**Idea docs**: [practice_manufactures_its_own_credit](../../../../ideas/practice_manufactures_its_own_credit.md)
(§1 re-chunking, §3½ the FM as the holding structure, §6 rarity) ·
[performance_error_is_the_bridge](../../../../ideas/performance_error_is_the_bridge.md) (δ as detector)
**Parents**: [`../etude/`](../etude/README.md) (finding 7 is this node's mandate: *hierarchy is
meaningful only over boundaries that carry information*; its committed units were state-independent,
post-commit drift exactly 0.0000, fusion provably vacuous) ·
[`rhm/practice/crystallize/`](../../../rhm/practice/crystallize/README.md) (the RHM answer being
ported back: state-conditioned commitment 1.8–3.0×) · [`../../arm_substrate/`](../../arm_substrate/README.md)
(P3 planner sizing, P4 composition horizon) · [`../../on_policy/COLLECTION_REALISM.md`](../../on_policy/COLLECTION_REALISM.md)
(practice is tier C: on-policy collection is mandatory here, and this is the first practice node
fully native to that convention)
**Status**: written up 2026-08-20 (interpretation discussed with Jasper throughout). Four runs:
`g0` (admissibility gates, 4 worlds), `f0` (the op comparison), `f1b` (timing × maintenance),
`f1c` (the live-content op). Single seed each; `f1c`'s `never` reproduces `f1b`'s **bit-exactly**
(max |Δ| = 0.00e+00 over 32 probes). File index: [FILES.md](FILES.md). **Dates**: 2026-08-19 → 08-20.

## One-liner

On an n=3 planar arm whose boundaries genuinely carry information (arrival-*posture* spread on the
redundancy manifold, invisible in tip space), the RHM compile-op laws reproduce where they were
born — state-conditioned commitment beats state-independent 1.34×, unconditional averaging costs
3.0× with the damage monotone in pool coherence, frozen content rots by diet-narrowing and only
re-selection converts a maintained model back into performance — and then the op taxonomy's
completion **inverts the frame**: *live content under committed routing* (plan the corridor from
the current FM at launch, fly open-loop) dominates every frozen op **1.6× on error at 2.3× less
priced time**, self-maintains its own diet, and lands on the reactive controller's own ballistic
reference. Within the model's composition horizon, the right thing to commit is the **routing, not
the content** — and `never` (reactive practice + dense learning) wins outright at every feedback
and deliberation price, which is the idea doc's §6 rarity law on a priced axis.

## Substrate

Two-segment piece on the n=3 planar arm: a frozen mastered **approach** (H=14, one full-horizon
plan from the start posture, executed open-loop with motor noise — so the hand-over distribution is
a property of the world, not of the arm being graded) into a **drilled segment** (H=20, at/beyond
the plant's measured composition horizon of 20–23 — G4's "commitment is a bet" condition). A
Gaussian-gated **curl patch** (b=14) sits mid-segment; the FM is pretrained with the region
excluded (the étude's `pretrain_mode=exclude` pattern). Motor noise is live in run-throughs, not
only practice — the absence of which is exactly why the étude's post-commit drift was identically
zero.

- **Collection**: on-policy only (`embodied.Body`, no `set_state`, every step charged); instrument
  steps ran 0.35–0.8× agent steps against the 22× subsidy that voided `ballistic/directed`.
- **Pricing**: `t = steps·dt + fb·d_fb` (dt = 0.02, d_fb = 0.10 default). From `f1c` the ledger
  also **counts deliberations** (`plans` per kind, never charged; d_plan defaults to 0 so every
  earlier number is byte-unchanged) and the reducer sweeps the d_fb × d_plan price surface
  post-hoc. No node in the repo had ever separated plan counts from feedback events before.
- **Planner**: CEM 1024×8 per `arm_substrate` P3. CAL-P measured the sizing law's sharpest form
  yet: an undersized planner does not merely shrink the FM-quality axis — at the b26 world the
  stale→ceiling gap **inverts sign** (−0.0037 at 256/4 vs +0.1088 at 1024/8).

## The gates (`g0`, four worlds, single seed)

| gate | b14 (settled world) | verdict |
|---|---|---|
| G1 boundary information | hand-over spread 5.65× noise; per-state oracle **4.00×**; R² of unit error on hand-over 0.82; post-commit drift sd 0.0058 | **pass** — the étude's exact-0.0000 degeneracy is broken |
| G2 multimodality | mean of *successful* renditions 0.82–0.90× the median contributor, all four worlds | **fail** — resolved by f0/f1b (below): the probe averaged a success-filtered pool |
| G3 usable range | 0.1726, 31.7× metering noise; ballistic/reactive transmission 4.92× | **pass** |
| G4 commitment is a bet | stale horizon 16 < H=20 ≤ ceiling 21 | **pass** (only b14/b14obs) |

The boundary information is **postural**: the hand arrives within ~3 cm of the waypoint (tip sd
0.031) with an arrival-posture spread of 0.29 rad along the self-motion manifold — larger than the
0.15 rad start jitter, because open-loop execution amplifies it and a Cartesian cost never
constrains the null space. This is the boundary information the pusher structurally could not
express. A soft-obstacle escalation (`b14obs`) was measured and **rejected**: harder, not more
informative. G2's inversion of the étude/crystallize averaging law was kept as a measured outcome
rather than patched, and f0/f1b resolved it (finding 3).

## The runs

### f0 — the op comparison (6 arms, 60 cycles, provisional commit at c20)

| arm | e_perf (win-4) | t_priced |
|---|---|---|
| `never` | **0.0082** | 12010 |
| `library_kmeans` | 0.0753 | 11182 |
| `regress` | 0.0795 | 11182 |
| `library_proj` | 0.0942 | 11182 |
| `fixed` | 0.1008 | 11182 |
| `mean` | 0.3021 | 11182 |

The committed arms decompose into a 2×2 the prior rounds confounded — **conditioning (~1.3×) and
selection-vs-averaging (~3.0×) are separable axes, and conditioning rescues regression** (the étude
and crystallize always ran their averaging control unconditional). Keying: k-means captured 41% of
the fixed→oracle gap (audition 0.0787 against best-fixed 0.1100 / oracle 0.0337) with 4 distinct
picks; the supervised linear key **failed** (R² 0.196 at commit time against 0.82 at gate time; 2
distinct picks) — an in-sample linear structure at one operating point does not license a linear
key at another. Audition→realized optimism gaps 0.86–1.0 at the like-for-unlike baseline: **no
seam-state shift** (vs e3b's 3.0×), so the consumption-matched audition transferred calibrated.
Payback never occurs, and not by pricing (committing netted a 7% time saving; the error gap is 9×).

### f1b — timing × maintenance (6 arms, 91 cycles)

**Mastery is real and it is the later of two clocks**: `never`'s ballistic plateau at **c51**
(0.0402; holds to c91) against a reactive clock of c12 — you can play it slowly long before you can
play it at tempo, and f0's c20 commit read the earlier clock. **Timing was null on error** (2.8%,
inside the oscillation) and real on price (early commitment 2.65× time-positive at high d_fb vs
mastery's 1.47×). The maintenance result is the round's sharpest: interleaving 1-in-3 reactive
practice cycles arrests FM rot from **2.90× → 1.10×** of `never`'s model error at +8% priced time —
and buys **exactly zero performance**, because a frozen unit never consults the model. Scheduled
**reselect** is the only channel: 0.0595 (1.16×, a lower bound — its pool was starved to 48
candidates), the only arm with negative post-commit drift. **Maintenance and refresh are strictly
separable, and the operative currency is competence-news** — nothing in the world moved (recerts
0-in-12, correctly), the self improved *under* the unit, and that staleness is invisible to
degradation detectors (δ<0 detects worsening; opportunity cost is not an error signal). Only a
scheduled re-audit consumes it. Per-cell coherence: averaging damage is monotone in pool coherence
(4/4 cells, both commit times) but the cells are **no more coherent than the global pool** — the
hand-over key does not separate the modes averaging destroys, so `regress`'s rescue is an open
puzzle (local-bandwidth and Huber-median hypotheses recorded, checkable from the persisted commit
archives).

### f1c — the live-content op (5 arms, 91 cycles)

`plan_launch`: one CEM plan from the current FM at the observed hand-over, fly open-loop, 1
feedback event, **no audition** (nothing to select; its cheapness is a real property and d_plan is
where it pays).

| arm | e_perf (win-4) | t_priced | plans |
|---|---|---|---|
| `never` | **0.0086** | 18215 | 131,040 |
| `plan_launch_mastery_il` | 0.0369 | 13335 | 82,248 |
| `plan_launch_early` | 0.0385 | **8502** | **33,912** |
| `plan_launch_early_il` | 0.0415 | 9551 | 44,400 |
| `commit_mastery_il_reselect` (best frozen) | 0.0611 | 19552 | 79,680 |

**Live content under committed routing dominates every frozen op** — 1.6× on error at 2.3× less
priced time — and lands on the reactive controller's own ballistic reference (each arm's e_perf ≈
its own e_ball). The étude's frozen-beats-live head-to-head (selected trace 0.0763 vs live CEM
0.0807, measured on a boundary that carried no information) **inverts** here. The predicted rot
signature came out the opposite way, with the mechanism: `plan_launch` without interleave did not
rot at all (0.87–1.14× of `never`'s model error) — **a live planner produces a state-appropriate
trajectory from every fresh hand-over, so live content self-maintains its own diet**, and
interleave is redundant for it (+12.3% for nothing). Jointly with f1b: *the rot was never a
property of committing — it was a property of committing frozen content*, and f1b's whole
maintenance/refresh apparatus is compensation for freezing bytes instead of routing (which is
§3½'s own claim — the chunk lives in the fast weights of the executing model — rediscovered from
data after we built the stored-tape version first). Prices: plan_launch is time-positive at
**d_fb = 0** (identical step count to `never`, 3.4× fewer feedback events), and **no deliberation
price up to 10 s/plan lets any arm overtake `never`** — its descent completes by cycle 12, so it is
cheap-and-good before any price vector can reach it.

## Findings

1. **Boundaries carry information on the redundant arm, and the information is postural** —
   invisible in tip space, 4.0× exploitable by a per-state oracle, R² = 0.82 predictable. (G1; 1 seed)
2. **State-conditioned commitment reproduces where it was born** (1.34×, 41% of the oracle gap) —
   and the keying basis is the hard part: isotropic partition > supervised linear key, which does
   not transfer across competence (0.82 → 0.196). (f0)
3. **The averaging law acquires its mechanism variable: pool coherence.** Damage is monotone in it
   (4/4 cells × 2 commits); G2's inversion was an averaging-after-selection artifact (success-filtered
   pool, coherence 0.70, vs the compile op's uniform draw at 0.58); the étude's 2.1× reproduces at
   3.0×. Conditioning and selection are separable; conditioning's rescue of regression remains open
   (the key provably does not separate the modes). (g0 + f0 + f1b)
4. **Maintenance and refresh are strictly separable, and maintenance alone is worthless** —
   interleave restores the model fully at +3–8% price and buys zero performance; only scheduled
   reselect converts it. The operative currency is **competence-news**, invisible to degradation
   detectors by construction. (f1b)
5. **Within the composition horizon, commit the routing, not the content.** Live content under
   committed routing dominates every frozen op, self-maintains, improves post-commit, and inverts
   the étude's head-to-head once the boundary carries information. (f1c)
6. **`never` wins outright at every price** (d_fb ≤ 3.0 s, d_plan ≤ 10 s) — the single easy segment
   is §6's rarity regime, the comparison is error-dominated, and the design has no consumption
   phase (acquisition instruments dominate the ledger; flagged as a round-3 design item, not an
   artifact). (f0–f1c)
7. **The δ-silence certificate fires at c10 against measured mastery at c51 in every arm of every
   run** — it reads the earliest of three clocks (certificate c10 < reactive c12 < ballistic c51).
   Recerts 0-in-17 across the node (correctly: no news channel is live). Boundary decisions stay
   scheduled from outside. (all runs)

## Runs on disk

| tag | what it is |
|---|---|
| `g0` | admissibility gates × 4 worlds (`b6`, `b14`, `b26`, `b14obs`) + CAL-P planner sizing |
| `f0` | the op comparison: `never / fixed / library_kmeans / library_proj / regress / mean`, commit c20 |
| `f1b` | timing × maintenance: `commit_{early,mastery}` × `{,_il}` + `_il_reselect`, 91 cycles |
| `f1c` | the live-content op: `plan_launch_{early,early_il,mastery_il}` + references; plans counter added |

## Reproduce

```bash
cd experiments/            # MODAL_PROFILE=chromatic
python3 mjc/practice/fingering/analyze_gates.py     --tag g0  --fetch
python3 mjc/practice/fingering/analyze_fingering.py --tag f0  --fetch
python3 mjc/practice/fingering/analyze_fingering.py --tag f1b --fetch   # includes d_fb reprice
python3 mjc/practice/fingering/analyze_fingering.py --tag f1c --fetch   # includes d_fb × d_plan surface
```

Full launch commands with exact flags, the calibration tables (CAL-P/CAL-D), the op-taxonomy and
pricing notes, and the gotchas (detached-launcher log location; the P3 sign inversion) are in
[FILES.md](FILES.md). Volume: `/data/practice_fingering/<tag>/<arm|world>/results.json` on
`mujoco-control-data`, with commit archives (`commit_c<N>.npz`) persisted from f1b on.
