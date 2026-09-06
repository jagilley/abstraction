# legato — committed execution across seams: the composition-horizon crossover

**Up**: [../README.md](../README.md) (practice) · [../../README.md](../../README.md) (mjc)
**Idea doc**: [practice_manufactures_its_own_credit](../../../../ideas/practice_manufactures_its_own_credit.md)
§2(b) (delay variance forces chunk boundaries — the derivation this node measures)
**Parents**: [`../fingering/`](../fingering/README.md) (the op taxonomy this node extends past the
horizon; substrate and machinery donor) · [`../etude/`](../etude/README.md) §E-4/§E-5 (sequential
assembly, seam-matched score sets, and the fusion-vacuity finding this node overturns)
**Status**: written up 2026-08-20 (interpretation discussed with Jasper throughout). Three gate
runs (`l0`, `l1` repair, `l2` seam-cost calibration) + the main run `L1` (6 arms, 90 cycles).
A seed pair (`L1_s1`/`L1_s2`) was launched and cancelled ~15 min in by Jasper
(GPU budget) — the triangulation on record is ranks (per `recital`'s methodology export), sign
consistency at two horizons (c75 and c90), and a double-sided fusion control. File index:
[FILES.md](FILES.md) (which carries the standing findings F1–F5). **Dates**: 2026-08-20.

## One-liner

On a closed 4-leg loop whose phrase (60 steps) is ~3× the plant's measured composition horizon
(~21 steps), the op taxonomy **crosses over**: live content wins inside the horizon
(`seg_plan_launch` 0.0850 vs `seg_frozen` 0.1065, 1.25×) and **frozen measured chains win beyond
it** (`phrase_frozen` 0.1026 vs `phrase_plan_launch` 0.1874, **1.83×**), with a fusion control
making the mechanism attributable: welding three segments into one uninterrupted span costs
**−0.004 for measured content (free) and +0.102 (2.2×) for live plans**, whose error concentrates
at the far end of the 60-step rollout (0.417 on the final segment). A measured chain was produced
by the body, so it carries no composition error — *execution is a perfect model of itself*. Chunks
exist not to beat planning inside its reach but to **extend committed execution past it** — the
idea doc's §2(b) derivation, arrived at from the op taxonomy rather than assumed. `never` still
dominates on raw error (0.0107); in steady state per traversal, `phrase_frozen` dominates
`seg_frozen` on **both** axes (1.1–1.8× less time at equal-or-better error).

## Substrate

A closed 4-leg loop found by FK search: approach 0.300 m (H=14), then three drilled segments of
0.428/0.453/0.420 m (H=20 each; phrase = 60 steps), tempo 0.0210–0.0227 m/step against fingering's
0.0220 — a legato segment is a fingering segment in difficulty class. The b14 curl patch sits
**mid-phrase** (leg C), so the boundary a phrase commitment surrenders is the one carrying the most
information. Turns 86/97/104° (a closed quadrilateral makes fingering's turn window structurally
unavailable). Machinery copy-forked/imported from `fingering/` per repo norms; collection,
pricing, and the plans counter as there.

## The gates — and the cost function's two-sided failure mode

**G5a, the mechanism (planner-free, unchanged across every cost setting).** Rolling the FM
open-loop along commands the body actually executed: the stale/live-like model diverges 0.032 →
0.385 → 0.673 m at 20/40/60 steps (horizon 21, independently reproducing `arm_substrate` P4's
20–23); the corridor-ceiling model composes to 44 — past a segment, short of a phrase.

**G6, seam information.** Keying a whole chain at the phrase launch is near-inert in audition
(1.03–1.08× held-out) while keying each segment at its own seam buys 1.9–3.6×, because state
spread grows **15×** along the chain (R² of the final segment's error: 0.079 from the phrase
launch vs 0.535 from its own seam). Robust across both cost regimes (F1).

**G7 failed twice, in mirror-image ways, before passing (l0 → l1 → l2).** l0's cost let the
reactive controller's lookahead cross seams and leave early toward the next waypoint — the
reference arm crippled (competent only on the final segment, [0.194, 0.165, 0.010]; FM-attributable
3%). l1's repair (a waypoint weight at seam steps, shared by every planner) restored the reference
16× (0.0078, fingering-`never` strength; signature [0.009, 0.008, 0.006]; axis present at 4.9×
noise, 38.9% attributable; transmission ordering restored at 14.2×) and settled the l0 ballistic
degradation as **controller, not diet**, via a model-side oracle probe (the FM improved 21% under
the repaired objective where l0's degraded 2.2×) — but created the mirror artifact: nail the
waypoints, ignore the **handoff velocity**, and the per-segment committed arm is crippled instead.
l2 swept the seam cost under a **pre-fixed neutral criterion** (mean piece error across all three
granularities, subject to axis-present — explicitly not the segment-vs-phrase gap, which would
tune the cost until the hypothesis won) and settled `vel_pen=0` (the loop is treated as
continuing; also removes the asymmetry where only the closing leg ever saw a terminal state) and
`γ=0`: **there is no controller-level seam law** (every lookahead weight γ>0 is monotonically
worse — weight spent on a segment you will not control buys a compromise you never cash; F3). The
decisive handicap check: reactive shares the byte-identical cost and is uniformly excellent
([0.009, 0.009, 0.003]), so the committed arm's per-segment gradient is the commitment cost
itself — the phenomenon, not the apparatus. At the chosen cell the span-degradation ordering is
monotone in re-grounding count on the ceiling model in two independent measurements (2.9× / 4.3×),
present but weaker on stale (1.14–1.64×). Two caveats, unsmoothed: the stale and ceiling models
degrade for **different reasons** (stale = genuine composition, 0.73 m planner-free at 60 steps;
ceiling = a 180-dim search exploiting model error off-corridor — not composition evidence), and
the phrase span is planner-starved on the stale model (composition is sufficient for the arms'
live, stale-like models; search contributes).

Jointly, G6 + F3 say **a seam cannot be pre-handled** — not from the launch state, not from the
plan. It is handled *at* the seam (re-grounding), or by content *selected on realized seams*.
Those are exactly the two things the main run compares.

## The main run (`L1`, 6 arms, 90 cycles, commits c25/32/39 + phrase c55)

Headline at the **common horizon c75** (`never` was killed by the 8 h Modal wall-clock at c75/90;
its partial is fully intact and flat at plateau — étude `eg_s0` precedent, flagged in the tables):

| arm | e_piece | fb/piece | t_priced | rank |
|---|---|---|---|---|
| `never` | **0.0107** | 61 | 40932 | 1 |
| `seg_plan_launch` | 0.0850 | 4 | 25291 | 2 |
| `phrase_frozen` | 0.1026 | 2 | 42548 | 3 |
| `seg_frozen` | 0.1065 | 4 | 34761 | 4 |
| `phrase_chain_fixed` | 0.1374 | 2 | 42548 | 5 |
| `phrase_plan_launch` | 0.1874 | 2 | 33427 | 6 |

**The crossover (F4).** Inside the horizon, live beats frozen (0.0850 vs 0.1065, 1.25×) — the
fingering result reproduced. Beyond it, frozen beats live (0.1026 vs 0.1874, 1.83×). Same signs
and ordering at c90 (1.71× / 1.44×). The **fusion control** attributes it: `seg_frozen` →
`phrase_frozen` costs −0.0039 (free, and saves 2 fb/piece); `seg_plan_launch` →
`phrase_plan_launch` costs +0.1023 (2.2×), concentrated where the model's imagination has drifted
furthest.

**The launch-key surprise (F5).** `phrase_frozen` vs `phrase_chain_fixed` = **1.34×** against
G6's 1.03× audition estimate — because the run's phrase pool is harvested from a
**segment-committed** configuration in which state-conditioned segment units fire different key
combinations per traversal, manufacturing genuinely distinct whole-phrase chains. **The lower
level's library funds the upper level's addressable variation** — `recital`'s most seed-stable
RHM law (the value of level-k practice is denominated in level k+1's currency), materializing on a
physical plant. F1 is back-referenced accordingly: the 1.03× is the audition estimate, not the
closed-loop value.

**Economics.** No committed arm crosses `never` cumulatively at any d_fb — error-dominated, as in
every fingering round (and the cumulative ledger is dominated by the shared pre-commit phase). In
**steady state per traversal** (74 steps), `phrase_frozen` dominates `seg_frozen` on both axes:
equal-or-better error at 1.12× / 1.57× / 1.80× less time at d_fb = 0.1 / 1.0 / 3.0 s. The fusion
win is real; the acquisition-heavy ledger hides it — the consumption-phase gap is a named round-3
design item, not an artifact claim.

**Caveats.** The commit schedule fired **mid-descent again** (read off `never`'s own ladder, the
ballistic clock is c72 and still not plateaued, against commits at c25–55 — the third occurrence
of this pattern in the port). It biases `never`-vs-committed and leaves the within-committed
comparisons — the crossover, the fusion control, the keying cell — unaffected, since all committed
arms share the schedule. Post-commit drift is diagnostic of content type (`seg_plan_launch`
−0.0217, improving with its model; every frozen arm ≈ 0). The phrase-level audition is essentially
unbiased (chosen 0.0933–0.1005 vs realized ≈0.1026, gap ≈1.0, against the étude's 2.8–3.4×). No
library collapsed (segment commits 4/3/4 distinct picks at oracle gains 3.36/3.26/5.75; phrase
2/4/3 at 1.86/1.83/1.67).

## Findings (standing record in [FILES.md](FILES.md))

- **F1** — launch-time keying of a chain is near-inert *in audition* (1.03×); see F5 for the
  closed-loop correction.
- **F2** — calibrate to preserve the **measurement axis**, not to optimize an arm (l0's 3%
  attributable as the worked example; the axis criterion must be measured on the controller that
  transmits).
- **F3** — there is no controller-level seam law: lookahead weight across a seam you will not
  control is monotonically harmful; re-grounding supersedes anticipating.
- **F4** — the crossover: live content wins inside the composition horizon, frozen measured chains
  win beyond it; fusion is free for measured content and 2.2× for live plans. (Ranks
are the claim; two-horizon sign consistency + the double-sided control are the triangulation.)
- **F5** — the lower level's library funds the upper level's addressable variation
  (audition-estimated 1.03× vs closed-loop 1.34×).

## Runs on disk

| tag | what it is |
|---|---|
| `l0` | first gates: G5a/G5b/G6 pass, G7 fails (lookahead-crosses-seams artifact) |
| `l1` | repair gate: waypoint-weighted cost, axis criterion corrected, controller-vs-diet settled |
| `l2` | seam-cost grid (vel_pen × γ), neutral pre-fixed criterion; F3; chosen cell for the main run |
| `L1` | the main run, 6 arms × 90 cycles (`never` truncated at c75 by the 8 h wall clock) |
| `L1_s1`, `L1_s2` | seed pair, launched 2026-08-20 and cancelled ~15 min in (GPU budget); no results consumed |

## Reproduce

```bash
cd experiments/            # MODAL_PROFILE=chromatic
python3 mjc/practice/legato/analyze_gates.py  --tag l0 --fetch
python3 mjc/practice/legato/analyze_gates.py  --tag l1 --fetch
python3 mjc/practice/legato/analyze_gates.py  --tag l2 --fetch
modal run --detach mjc/practice/legato/legato.py::legato --tag L1 --seed 0 --n-cycles 90 \
    --commit-seg 25,32,39 --commit-phrase 55 --w-waypoint 16.0 --vel-pen-mid 0.0 --vel-pen 0.0 \
    --lookahead-gamma 0.0 --calp 1:1024:8,2:4096:12,3:4096:12
python3 mjc/practice/legato/analyze_legato.py --tag L1 --fetch
```

Gate launch flags, calibration tables, and the gotchas (the `never` 8 h wall-clock ceiling — raise
the timeout or budget `n_cycles` for reactive arms; the macOS case-collision between tags `l1` and
`L1`, defended in `load()`; the launcher-log location) are in [FILES.md](FILES.md). Volume:
`/data/practice_legato/<tag>/<arm|world>/results.json` on `mujoco-control-data`.
