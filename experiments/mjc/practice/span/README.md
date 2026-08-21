# span — the composition horizon as a trajectory: the model's reach grows through a flat task metric, and a live plan's reach does not follow

**Up**: [`../README.md`](../README.md) (practice) · [`../../README.md`](../../README.md) (mjc)
**Parents**: [`../legato/`](../legato/README.md) (substrate, practice loop, calibration record; its
**G5a** is the measurement this node turns into a trajectory) · [`../fingering/`](../fingering/README.md)
(legato's parent; the op taxonomy). **Files**: [`FILES.md`](FILES.md) (machinery, knobs, the full
S1/S2 measurement records, reproduce). **Idea doc**:
[`practice_manufactures_its_own_credit`](../../../../ideas/practice_manufactures_its_own_credit.md)
§3½ (2026-08-20 revision), §16, §18, §20. **Reading**: Iwane, Hayward, Karunathilake, Buch & Cohen
2026, *Cell Reports*, "Hippocampal skill memory expansion drives online performance dynamics during
skill learning" (`reading/Hippocampal skill memory expansion.pdf`).
**Runs**: `S1` (the practice run, 80 cycles, 2026-08-20), `S2` (the disentangling re-measurement of
S1's saved models, 2026-08-20). **Single seed on everything; the 17–81-point monotone trends are
the claims, single cells are not.**

## The question

Iwane et al.'s central behavioural result is that human skill memory keeps expanding after
performance has plateaued: practising a 6-keypress sequence, speed flattens by trial ~6, but the
content of the within-trial high-skill segment keeps growing through trial 26 — the chunk *count*
pinned near 7, the chunk *size* doing the growing, and hippocampal θ/γ coupling the one regional
signal predicting it. Our practice doc's reading of a committed unit is that its span is bounded by
the forward model's **composition horizon** (how many steps the model can be rolled open-loop before
its tip prediction leaves what the body does), and §16/§18 already hold that this kind of growth is
invisible to within-level performance. `legato` G5a measured the horizon exactly twice — a stale FM
21/23 steps, a corridor-matched ceiling FM 44/58 — two points on a curve nobody had traced. On
`legato`'s piece the reactive task error `e_react` is at plateau from cycle 0, so the whole run is
"post-plateau" by construction and anything the model's composability does is invisible to the
scoreboard. Hence:

> **Does the composition horizon keep growing with practice after the reactive task error has
> flattened — and does that growth convert into longer committed execution?**

## Design in brief

**S1** — `legato`'s practice loop (L1 design point, byte-identical piece/plant/patch/diet/controller;
the chosen `l2` cost cell) with the commitment machinery and the metering removed: pure reactive
practice, 80 cycles, seed 0, one container. The FM is snapshotted every cycle (weights every 5 →
`fm_snapshots.pt`) and two probes run per snapshot:

1. **Imagination horizon** (planner-free, every cycle): G5a's own construction — roll the FM
   open-loop along a fixed 134-step executed command sequence (two laps of the closed loop, built
   once under the stale model) and take the first step the median tip error exceeds a threshold
   (0.01/0.02/**0.05**/0.10/0.20 m). Also run on the commands the body executed *this* cycle
   (`on-policy`).
2. **Executed span** (dense early, sparse late; 38 probes): one live CEM plan of the whole 60-step
   phrase from W1 under the snapshot model at CAL-P's phrase sizing, flown **open-loop** on the body
   with performance motor noise — `legato`'s `phrase_plan_launch` content read as a per-step curve.
   Readout: `goal_excess` (distance to the step's waypoint minus the competent reactive reference's,
   paired on launch state — the one curve with a natural zero) and per-waypoint **seam arrival**.

**S2** — re-measure S1's 17 saved models, no retraining, regenerating every geometry and plan from
S1's config and seed offsets (fidelity: launch states match to 0.00e+00 m, regenerated plans
reproduce S1's stored `e_seam` to 7.2e-07 m). Per snapshot: the composition horizon along **three
command sources** (the frozen cycle-0 reference, a reactive traversal under *this* model, and the
model's own phrase plan); **one-step** FM error on the corridor (matched motor noise) vs on the
plan's trajectory; the model's **predicted** seam arrival under the chosen plan vs the true one
(the exploitation gap); corridor geometry; and, at 8 cycles, a **CAL-P sweep** 1024×8 → 16384×24
recording true and believed piece error at every sizing. Machinery, knobs, floors and both
measurement records: [`FILES.md`](FILES.md).

## Findings

### F1 — The task metric is flat; the model's composability is not

`e_react` over 80 cycles: 0.0056–0.0129, median 0.0097, sd 0.0017, ρ=+0.24 (p=0.30). Through that:

| imagination horizon | c1–20 median | c61–80 median | ρ (Spearman, 81 pts) | p |
|---|---|---|---|---|
| @0.01 m | 8.5 | 14.0 | +0.877 | 8e-27 |
| @0.02 m | 11.0 | **22.0** | +0.770 | 4e-17 |
| @0.05 m | 21.0 | **26.0** | +0.764 | 1e-16 |
| @0.10 m | 24.0 | 29.0 | +0.613 | 1e-09 |
| @0.20 m | 37.0 | 36.0 | +0.275 | 0.013 |
| @0.05 m, on-policy commands | 23.0 | 33.0 | +0.595 | 6e-09 |

Monotone, robust at tight thresholds and gone at loose ones: what grows is the *precision* of
composition along the consumed corridor, not gross trajectory fidelity. The stale-FM anchor
reproduces G5a (24 vs 21/23); the ceiling-FM anchor does **not** converge across corridor draws
(34 here, 44 `l0`, 58 `l2`) and should not be leaned on. Neither measurement ceiling binds (max
observed 29 of 134). Launch drift against the frozen reference set stays 0.019–0.029 m against a
launch spread of 0.022 m, so the reference does not go stale. **Representation keeps expanding
after performance has plateaued — the paper's dissociation, measured.**

### F2 — The executed span does not grow, and the far end gets worse

One live open-loop phrase plan delivers exactly **one waypoint** (W2, step 20) in essentially every
cycle from 0 to 80 (2 waypoints at 0.15 m only at c17–22 and c55–58); `goal_excess`@0.05 sits at
4–5 steps throughout (ρ=+0.40, a stereotyped mid-leg route difference, not a failure). Every
cycle's per-step curve is superimposable to step ~22; all cross-cycle variance lives past it, and
it goes the wrong way — seam arrival medians c7–22 → c26–80: W2 0.046 → 0.047, W3 0.253 → 0.433,
**W0 0.179 → 0.928 (5.2×)**, with `e_ball_seg` degrading in parallel (0.145 → 0.211, reproducing
L1's `never` ladder). There was a brief improvement to c22 (`e_piece` 0.129, its best) and then a
jump at c24–26 that never came back; nothing is scheduled there. Shape: gradual compounding, no
knee (the step-41 dip is the W3→W0 goal switch). Two intended readouts (`ideal`, `path`) were
unusable because the CEM cost rewards arriving early and dwelling, so the competent reference itself
leaves constant tempo within 2 steps — recorded in `FILES.md`.

### F3 — Planner starvation is not the reason (S2)

Paired across 8 cycles, CAL-P's chosen sizing (4096×12) vs 16× more search (16384×24): true
`e_piece` 0.436 → 0.369, top better in **2/8** (Wilcoxon p=0.74); W0 0.752 → 0.584, **2/8**
(p=0.64). What more search reliably buys is a lower-scoring plan **under the model**: believed
`e_piece` 0.086 → 0.073, **7/8** (p=0.023). Across all 40 cells, believed error spans 0.044–0.130
(sd 0.014) while true spans 0.152–0.729 (sd 0.175); **corr(believed, true) = +0.009** (p=0.96). In
c0–c30 more search makes the truth slightly *worse* (0.344 → 0.375) while the belief improves; in
c55/c80 it helps — but each cell is one CEM draw at ±0.2 m, so the pooled statistics carry and
single cells (including c80's 0.166) do not.

### F4 — Practice sharpens the model where the body goes and degrades it where the plan goes (S2)

| one-step FM error | c0–c15 | c65–c80 | ρ (17 pts) | p |
|---|---|---|---|---|
| on the corridor (matched motor noise) | 0.243 | 0.135 | **−0.912** | 3.5e-07 |
| on the plan's own trajectory | 0.194 | 0.307 | +0.422 | 0.092 |
| ratio plan / corridor | 0.78 | **2.05** | **+0.775** | 2.6e-04 |

The ratio crosses 1 at c20–25 — the same cycles as F2's far-end jump and as the jump in the plan's
distance to the nearest corridor state (which grows *through* the phrase, and more steeply late:
c20 0.45 → 1.11 over steps 1 → 60; c55/c80 0.36/0.43 → **3.0**). What narrows in the diet is not the
position footprint (tip-cloud area *grows*, ρ=+0.80) but its velocity spread (state-covariance
trace 21 → 16, ρ=−0.78; mean speed falls, ρ=−0.65): the corridor slows down, and a dynamics model
trained on a slower diet is worse wherever a plan moves fast. In belief space the same thing:
inside the horizon the model is calibrated (W2 true ≈ predicted, ~0.05 m, every cycle); past it the
predicted arrival at W3 sits at 0.052–0.082 m at **every** snapshot, and the predicted arrival at
W0 *falls* (0.136 → 0.083, ρ=−0.70, p=1.9e-3) while the true one rises 0.32 → 0.93. The gap opens
because reality moves, not because the promise inflates — a model that gets sharper on what you do
gets more confident and more wrong about what you don't.

One S2 readout is uninformative and should be read as such: the composition horizon along the
plan's *own* commands equals the horizon along the corridor's (paired −1 step @0.05, +3 @0.02, no
trend). But the 0.05 m horizon is 21–26 steps and the plan does not leave the corridor until step
~23, so that comparison is measured entirely inside the on-corridor prefix and cannot see the
off-corridor part; the one-step and belief readouts above can, and do.

### F5 — What this says about the paper mapping (interpretation, discussed 2026-08-20)

- The paper's **dissociation** — representation keeps expanding after the task metric plateaus —
  reproduces in S1 and survives S2 unchanged (F1).
- The paper's **conversion** — bigger chunks mean longer high-skill execution — does not reproduce
  for live plans (F2), and S2 says why: a live chunk is a model's promise off-distribution, and
  practice makes that promise worse, not better (F4), in a way search cannot repair (F3). The human
  chunk is **measured content** — the body's own executed sequence, bound and replayed — which never
  has to be priced by a model. That is legato F4 ("execution is a perfect model of itself") arriving
  from the model's side, and it says the in-silico object that can grow the way the paper's chunks
  grow is frozen measured content / the mined table, not plan-at-launch. In this world that object
  has no room to grow either: the reactive controller is at ceiling from cycle 0, so there is no
  behavioural learning curve (the paper's trials 1–5) for chunk content to ride on.
- It lands on the arc's seam law (§15: audition must match consumption in state, level and
  evaluator). The exploitation gap *is* an audition/consumption gap — the FM's audition of its own
  phrase plan says ~0.07 m, consumption says ~1 m — and the failure is on the **state** coordinate:
  the plan's states are off the diet. One more reason the certificate kept losing to provisional
  commitment graded in consumption: model belief past the horizon is not an instrument. Recorded as
  a sharpening of §15, not a new law.

## Caveats

- **Single seed** throughout. F1's trend is 81 points at p~1e-16 and F4's 17 points at p~1e-4, so
  the headline does not want a seed pair; F2's c22–26 regime change is now a coincidence of three
  readouts (far-end jump, one-step crossover, corridor-distance jump) but still one seed, and CAL-P
  cells are single CEM draws at ±0.2 m.
- `t_priced` is **not** comparable to legato's ledger (metering is cut); this node makes no economic
  claim.
- The human's within-trial speed drop at the segment boundary is, in our pricing model, a
  discontinuity by construction (one delay per committed unit vs one per reactive step) and was not
  looked for; the human HIS *duration* is fixed and its content densifies, whereas this substrate
  has fixed tempo and no speed/accuracy trade on a committed unit — the only growth axis here is
  steps covered by one committed unit.
- The `current`-command horizon in S2 is underpowered at 17 points (S1's 81-point on-policy probe
  clears significance), not contradictory.
- Two additive readouts were added to `../legato/world.py` (`acts_app`/`acts_app_raw`,
  `tips`/`tips_app`); no RNG draw moves and prior legato/fingering runs stay byte-reproducible
  (recorded in both `FILES.md`).
- Modal **preempts** these containers: S2's first launch was lost at c50 to
  `Container terminated due to preemption` (client-side: only
  `RemoteError: Function call was cancelled by user or a failure`). `s2.py` resumes and retries;
  both runners commit partial results frequently.

## Reproduce

```bash
cd experiments/            # MODAL_PROFILE=chromatic
python3 mjc/practice/span/launch_detached.py --tag S1 --seed 0 --n-cycles 80 --n-eval 32 --probe-every 4
python3 mjc/practice/span/analyze_span.py --tag S1 --fetch          # 5 figures → results/S1/figs/
python3 mjc/practice/span/launch_detached.py --fn s2 --tag S2 --src-tag S1
python3 mjc/practice/span/analyze_s2.py --tag S2 --fetch            # 3 figures → results/S2/figs/
```

Volume `mujoco-control-data`: `/data/practice_span/<tag>/{results.json, fm_snapshots.pt, done.txt}`.
Full knob table, floors, both measurement records, figure index and gotchas: [`FILES.md`](FILES.md).
