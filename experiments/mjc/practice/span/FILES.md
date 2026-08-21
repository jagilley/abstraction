# span — file index

**Up**: [`../README.md`](../README.md) (practice) · [`../../README.md`](../../README.md) (mjc)
**Parents**: [`../legato/`](../legato/README.md) — this node's substrate, practice loop and
calibration record; its **G5a** is the measurement this node turns into a trajectory.
[`../fingering/`](../fingering/FILES.md) is legato's parent and the origin of the op taxonomy.
**Idea doc**: [`practice_manufactures_its_own_credit`](../../../../ideas/practice_manufactures_its_own_credit.md)
§3½ (2026-08-20 revision), §16, §18, §20.
**Reading**: Iwane, Hayward, Karunathilake, Buch & Cohen 2026, *Cell Reports*, "Hippocampal skill
memory expansion drives online performance dynamics during skill learning"
(`reading/Hippocampal skill memory expansion.pdf`).

**Findings**: [`README.md`](README.md) (F1–F5, discussed 2026-08-20). This file is the machinery,
knob and measurement record; the two "numbers only" sections below predate the writeup and stand as
recorded.

## The question

`legato` G5a measured the plant's composition horizon **exactly twice**: a stale forward model
composes 21 (gate `l0`) / 23 (gate `l2`) steps before its open-loop tip prediction leaves 0.05 m of
the truth; a corridor-matched ceiling model 44 (`l0`) / 58 (`l2`). Two points on a curve nobody has
traced.

> **Does the composition horizon keep growing with practice after the reactive task error has
> flattened?**

That is the substrate analogue of Iwane et al.'s central behavioural result — the content of the
high-initial-skill segment keeps expanding through the speed plateau (their trials 6–26), with
chunk *count* pinned near 7 and chunk *size* doing the growing. On `legato`'s piece the reactive
metric `e_react` is flat at ~0.006–0.019 from cycle 0 (L1's `never` ladder), so the whole run is
"post-plateau" by construction and anything the model's composability does is invisible to the task
metric. Reading the horizon instead is the point of the node.

**Known disanalogy, not to be forced.** In the paper the HIS *duration* is fixed (~4.0–4.7 s across
practice) and what grows is what fits inside it. This substrate has a fixed tempo and no
speed/accuracy trade on a committed unit, so there is no densification axis here: the only thing
that can grow is the number of steps a single committed unit can cover. Likewise the human's
speed drop at the segment boundary is, in our model, a pricing discontinuity by construction (one
delay per committed unit vs one per reactive step) and is not evidence of anything.

## Code files

| file | purpose |
|---|---|
| `span.py` | The run. `legato`'s practice loop with the commitment machinery and the metering removed (pure reactive practice = the `never` configuration), instrumented with the two horizon probes at every cycle / on a dense-early schedule. One Modal container. |
| `analyze_span.py` | Fetch + reduce: the horizon ladder against `e_react`/`e_ball_seg`, the anchors, the floors, threshold sensitivity, and four figures. |
| `s2.py` | **S2** — the disentangling re-measurement. Loads S1's `fm_snapshots.pt` and re-measures those models (no retraining): three divergence sources, the exploitation gap, on/off-corridor one-step error, corridor geometry, and a per-snapshot CAL-P sweep. Resumable; carries Modal retries. |
| `analyze_s2.py` | Fetch + reduce S2: fidelity, the three sources, the gap, corridor geometry, the CAL-P table, and three figures. |
| `launch_detached.py` | Session-isolated detached launcher (copy-fork of `../legato/launch_detached.py`), `--fn span\|s2`. |

Substrate code is **imported**, not forked: `World`, `Ledger`, `start_postures`, `elite_for` from
`../legato/world.py` and `DEF_WPS` / `DEF_PATCH_SEG` from `../legato/gates.py`. The piece, the
plant, the patch, the diet and the controller are byte-identical to legato's L1 design point, so
the horizon this node traces is the horizon that node's arms were flying against.

### What changed in `../legato/world.py` (additive, 2026-08-20)

`traverse` gained four pure readouts. No RNG draw moves and nothing upstream reads them, so every
prior legato/fingering run stays byte-reproducible.

| field | what it is | why |
|---|---|---|
| `acts_app`, `acts_app_raw` | the approach leg's issued (post-noise) and raw commands, on the same convention as `acts` / `acts_raw` | so consecutive laps of the closed loop can be **concatenated** into one contiguous executed command sequence. G5a's probe stopped at 60 steps and its ceiling model already reached 58 — the measurement was about to hit its own ceiling. |
| `tips`, `tips_app` | the tip reached after **every** control step, by analytic FK (C0 pins FK to MuJoCo at <1e-9) | so an FM rollout is scored against the trajectory the body **flew**, not against a `true_tips` replay of the issued commands. The replay re-enters through `set_state` from a float32 state; measured drift is 6e-7 m by step 134, which is negligible but was worth removing rather than assuming. The replay is kept as the `replay` chaos floor. |

## The two probes

### 1. Imagination horizon (planner-free) — every cycle

G5a's own construction. Roll the forward model open-loop along a **fixed** set of executed commands
and score the predicted tip against the plant's, step by step; the horizon is the first step where
the **median** error exceeds a threshold (`cross`, 1-based, `len+1` when it never does — a
measurement ceiling, printed `>N`).

The reference is built once, before the loop:

| piece | how |
|---|---|
| lap 1 | `traverse(fm0, reactive, q_ev, seed+7001, sigma_perf, approach_plan=plan_ev)` — byte-identical to G5a's `executed` source |
| lap 2 | `traverse(fm0, reactive, q0=lap1.final)` — the loop closes at W0, so lap 2's approach leg (W0→W1) and three segments continue the same trajectory with no gap |
| commands | `[lap1.acts | lap2.acts_app | lap2.acts]` — **134 steps** (60 + 14 + 60) from one launch state |
| truth | `[lap1.tips | lap2.tips_app | lap2.tips]` — the flown trajectory |

`imagination_onpolicy` runs the same probe on the commands the body executed **this** cycle
(practice tempo, batch `batch`), which separates "the model got better on a fixed corridor" from
"the corridor narrowed".

### 2. Executed span (the behavioural analogue) — dense early, sparse late

One live CEM plan of the whole phrase from W1 under the snapshot model at CAL-P's phrase sizing
(`calp` span 3), flown **open-loop** on the real body with `sigma_perf` motor noise. This is
`legato`'s `phrase_plan_launch` content read as a curve rather than as three seam errors.

Five per-step references are recorded, because tip error has no canonical zero on a piece whose
cost rewards arriving early at each waypoint and dwelling there:

| curve | definition | floor behaviour (measured, not assumed) |
|---|---|---|
| `goal_excess` | distance to the step's own waypoint **minus** the reference reactive traversal's, paired on launch state | **the usable one.** Natural zero; a measured chain re-flown with fresh noise stays under ~0.05 m for the whole phrase |
| `ref` | distance to the reference reactive trajectory from the same launch state | non-monotone — both trajectories get pulled back toward each waypoint |
| `path` | perpendicular distance to the current leg | floor is high: the competent reference itself arcs 7 cm off the straight leg by step 3 |
| `goal` | raw distance to the step's waypoint | a sawtooth by construction; level says nothing alone, kept for the profile figure |
| `ideal` | distance to constant-tempo linear interpolation along the leg | **unusable** — the competent reference leaves it within 2 steps, because the cost never asks for constant tempo |

Three floors, measured once: `frozen_noisy` (the reference chain's *raw* commands re-executed
open-loop with fresh motor noise — the irreducible cost of noise under a perfect, because measured,
plan; literally `legato`'s frozen content), `frozen_clean` (the same, noiseless), and `replay` (the
plant's own sensitivity to a float32 re-entry).

## Knobs

| knob | default | note |
|---|---|---|
| `n_cycles` | 90 (run `S1`: **80**) | L1's `never` ballistic clock was still moving at c72 |
| `n_eval` | 32 (L1 used 48) | sets the held-out geometry set, hence the reference set's batch and the executed-span CEM batch |
| `probe_every` | 3 (run `S1`: **4**) | the `e_react` / `e_ball_seg` ladder; the horizon probes do **not** use this |
| `span_dense` / `span_mid` / `span_mid_every` / `span_late_every` | 12 / 30 / 2 / 3 | executed-span schedule: every cycle to 12, every 2nd to 30, every 3rd after, plus the last cycle |
| `weight_every` | 5 | FM `state_dict` snapshots written to `fm_snapshots.pt` for later re-analysis (the normaliser is set once, before the loop, and shared by every snapshot) |
| `save_every` | 3 | `results.json` + `volume.commit()`, so a wall-clock truncation is still usable |
| `w_waypoint` / `vel_pen` / `vel_pen_mid` / `lookahead_gamma` | 16.0 / 0.0 / 0.0 / 0.0 | **the L1 chosen cost cell** (legato gate `l2`), so the horizon is measured under the objective G5a measured 23/58 under |
| `calp` | `1:1024:8,2:4096:12,3:4096:12` | L1's CAL-P sizing, one planner per span |
| `THRS` (module constant) | 0.01 / 0.02 / 0.05 / 0.10 / 0.20 m | G5a used 0.05 (headline) and 0.02; the rest bracket them so threshold sensitivity is on the record without a re-run |

Everything else — the piece, the patch, the diet, `sigma_practice`, `sigma_perf`, `adapt_lr`,
`n_grad`, `replay_frac`, `trace_window`, `batch` — is legato's L1 default, unchanged.

## What was cut from legato's loop, and why it is safe

The forward model trains only on the practice traversal (`buf` → `train_online`), so anything that
does not feed `buf` cannot change the trajectory this node snapshots.

| cut | reason |
|---|---|
| **metering** (`meter`, `n_rt=48` reactive run-through every cycle) | agent-side self-read; priced, never trained on. Dropping it removes ~45% of the per-cycle cost. **Consequence: `t_priced` here is NOT comparable to legato's ledger, and this node makes no economic claim.** |
| `e_perf` from the ladder | identical to `e_react` when nothing is committed |
| `e_ball_phrase` from the ladder | superseded by the executed-span probe's seam errors, which are measured on *matched* launch states |
| the commitment machinery (`compile_unit`, `score_set`, `candidates`, the interleave) | nothing commits; `perf_routing()` is reactive throughout |

## Runs on disk

| tag | what it is |
|---|---|
| `smoke` | `--quick` shakeout (6 cycles, tiny pools). Predates `goal_excess`; the analyzer degrades gracefully on it. |
| `s2smoke` | `s2.py --quick` shakeout (2 snapshots, 2-point grid); also the fixture the resume path was validated on. |
| `S2` | the disentangling re-measurement of S1's 17 snapshots (`s2.py`). **Complete**: 17 snapshots, 8 CAL-P cycles, 2069 s of container time across two launches. Launch 1 was lost to **container preemption** at c50 (Modal task log: `Container terminated due to preemption`; the client saw only `RemoteError: Function call was cancelled by user or a failure` with no remote traceback — not a cancel, not a timeout, not OOM, and the 78 s/snapshot rate was steady throughout). Launch 2 resumed from the committed partial and finished the remaining 6 snapshots plus the sweep. |
| `S1` | the run: seed 0, 80 cycles, `n_eval 32`, `probe_every 4`, everything else default. **Complete**: 81 snapshots, 21 ladder probes, 38 executed-span probes, 38.3 s/cycle, 0.85 h in the practice loop (~1.5 h wall clock including setup), `t_priced` 14553.6 s (agent-side, **not** comparable to legato's ledger — metering is cut). |

## The S1 measurement record

Numbers only — the reading gets discussed with Jasper before anything is written up.

**Anchors on this reference set** (horizon in steps, at 0.01 / 0.02 / 0.05 / 0.10 / 0.20 m):

| model | imagination | executed span (`goal_excess`) | phrase-plan e_piece |
|---|---|---|---|
| stale `fm0` | 11 / 20 / **24** / 25 / 28 | 2 / 3 / 5 / 7 / 37 | 0.2781 |
| corridor ceiling | 17 / 22 / **34** / 47 / 58 | 2 / 3 / 4 / 6 / 40 | 0.2153 |

The stale anchor reproduces G5a (21 in `l0`, 23 in `l2`, 24 here). The ceiling anchor does **not**
converge across builds: 44 (`l0`), 58 (`l2`), 34 (here). Three independent corridor draws, three
numbers — treat the ceiling as a loose upper anchor, not a constant of the plant.

**Floors** (measured once): the frozen measured chain re-flown with fresh motor noise crosses 0.05 m
at step **37** on `ref` and step **50** on `goal_excess`; noiseless, 37 / 53. The plant's
replay-chaos floor never crosses any threshold in 134 steps (8.9e-7 m at step 134). The reference
reactive traversal's own piece error is 0.0089 (lap 1) / 0.0888 (lap 2 — its approach leg is
planned live rather than from the mastered `plan_ev`, so lap 2 is the weaker rendition).

**Neither measurement ceiling binds.** Max imagination horizon observed 29 of 134; max executed
span 7 of 60.

**Two readouts turned out unusable, and the reason is in the piece, not the code:**

* `ideal` (constant-tempo linear interpolation along the leg) — the *competent reference* leaves it
  within 2 steps. The CEM cost is mean distance to the leg's endpoint, which rewards arriving early
  and dwelling, so nothing on this piece tracks constant tempo.
* `path` (cross-track) — same cause; the reference arcs 7 cm off the straight leg by step 3.

`goal_excess` has the natural zero and is the one to read, but with a caveat found in the data: its
0.05 m crossing sits at step 4–5 at **every** cycle, and that crossing is a *stereotyped mid-leg
route difference*, not a failure — the open-loop plan lags the reactive reference by ~0.13 m at
step 9 and is back to ~0.03 m by step 20, where it still arrives at W2 within 0.05 m. The
behavioural span is therefore better read off the **seam errors** (`span.e_seam`, fig 5): how many
consecutive waypoints the single open-loop unit still delivers. That is what `analyze_span.py`'s
summary now reports alongside the threshold crossings.

**Cross-cycle trends** (Spearman ρ against cycle over 81 snapshots; `c>=5` excludes the
first-online-update transient, which drops the horizon to 11 at c1):

| quantity | c1–20 median | c61–80 median | ρ | p | ρ (c≥5) |
|---|---|---|---|---|---|
| imagination @0.01 | 8.5 | 14.0 | +0.877 | 8e-27 | +0.871 |
| imagination @0.02 | 11.0 | 22.0 | +0.770 | 4e-17 | +0.805 |
| imagination @0.05 | 21.0 | 26.0 | +0.764 | 1e-16 | +0.749 |
| imagination @0.10 | 24.0 | 29.0 | +0.613 | 1e-09 | +0.542 |
| imagination @0.20 | 37.0 | 36.0 | +0.275 | 1e-02 | +0.141 |
| imagination @0.05, on-policy commands | 23.0 | 33.0 | +0.595 | 6e-09 | — |
| executed span `goal_excess` @0.05 | 4 | 5 | +0.397 | 1e-02 | — |
| waypoints delivered @0.15 | 1 | 1 | −0.029 | 0.86 | — |
| phrase-plan `e_piece` | — | — | +0.466 | 3e-03 | — |
| `e_react` | — | — | +0.239 | 0.30 | — |
| `e_ball_seg` (0.145 → 0.211) | — | — | +0.201 | — | — |

`e_react` over all 80 cycles: 0.0056–0.0129, median 0.0097, sd 0.0017. Launch drift against the
frozen cycle-0 reference set stays at 0.019–0.029 m throughout, against a launch spread of 0.022 m,
so the frozen reference set does not go stale.

**A regime change at c22→c26**, worth knowing before anyone reads the curves: the phrase plan's far
seams improve to c22 (W3 0.132, W0 0.208, `e_piece` 0.129 — its best) and then jump and stay
jumped. Medians c7–22 vs c26–80: W2 0.046 → 0.047, W3 0.253 → 0.433, W0 **0.179 → 0.928**, while
the imagination horizon over the same split goes 22 → 25. Nothing is scheduled at those cycles
(`trace_window` is 6, nothing commits, `probe_every` is 4).

**Standing caveat carried from legato**: `legato`'s CAL-P recorded the phrase span as
*planner-starved on the stale model* — error still falling at the top of the grid. The executed-span
probe uses that same fixed CAL-P sizing at every snapshot, so its **levels** are a joint property of
model and search budget; only its **changes** are attributable to the model, and even then only
jointly with what the search does with it.

## The S2 measurement record

Numbers only, as above.

**Fidelity (the whole point of reusing the snapshots).** The reference set and every phrase plan
are regenerated deterministically from S1's config and seed offsets: launch tips match S1 to
**0.00e+00 m**, launch states across all 17 snapshots to 0.00e+00, lap-1 piece error 0.0089 vs
S1's 0.0089, and the regenerated plans reproduce S1's stored `e_seam` at the 9 checkable cycles to
**7.2e-07 m**. S2 is a re-measurement of S1's models, not a second run.

**Three divergence sources** (composition horizon @0.05 m; `fixed` = frozen cycle-0 reference
commands, `current` = a reactive traversal under this snapshot from the same launch states,
`plan` = the snapshot's own phrase plan):

| source | c0–c15 median | c65–c80 median | ρ | p |
|---|---|---|---|---|
| `fixed` @0.05 | 21.5 | 26.0 | +0.701 | 1.7e-03 |
| `fixed` @0.02 | 11.5 | 21.0 | +0.746 | 5.9e-04 |
| `current` @0.05 | 22.5 | 24.5 | +0.328 | 0.20 |
| `current` @0.02 | 12.0 | 17.0 | +0.509 | 0.037 |
| `plan` @0.05 | 21.5 | 25.0 | +0.350 | 0.17 |
| `plan` @0.02 | 16.0 | 19.0 | +0.286 | 0.27 |

Paired per snapshot, `plan` − `fixed` is **−1.0 steps** (median) @0.05 and **+3.0** @0.02, with no
trend against cycle (ρ = −0.26 / −0.42). At 17 points only `fixed` clears significance; S1's
81-point on-policy probe did (ρ=+0.60, p=6e-9), so `current` here is underpowered rather than
contradictory.

**One-step FM error, with the matched-noise control.** The S1 practice diet flies at
`sigma_practice` 0.15 and the plan at `sigma_perf` 0.06, so the on-corridor set is the *perf-tempo*
reactive traversal collected in the same call.

| quantity | c0–c15 | c65–c80 | ρ | p |
|---|---|---|---|---|
| on-corridor (matched noise) | 0.2427 | 0.1354 | **−0.912** | 3.5e-07 |
| on the plan's trajectory | 0.1940 | 0.3069 | +0.422 | 0.092 |
| ratio plan / corridor | 0.775 | 2.048 | **+0.775** | 2.6e-04 |

**Predicted vs true seam arrival** under the plan the CEM actually chose:

| | c0–c15 median | c65–c80 median | ρ | p |
|---|---|---|---|---|
| W2 true / predicted | 0.063 / 0.054 | 0.051 / 0.060 | −0.31 / +0.40 | 0.23 / 0.12 |
| W3 true / predicted | 0.311 / 0.061 | 0.413 / 0.064 | −0.02 / −0.14 | 0.95 / 0.61 |
| W0 true / predicted | 0.320 / 0.136 | 0.932 / **0.083** | +0.31 / **−0.696** | 0.22 / 1.9e-03 |

The predicted arrival at W3 sits in 0.052–0.082 m at **every** snapshot from c0 to c80, and the
predicted arrival at W0 *falls* while the true one rises.

**Corridor geometry** — what narrows is not the workspace:

| | c0 | c80 | ρ | p |
|---|---|---|---|---|
| state-covariance trace | 20.99 | 16.43 | −0.782 | 2.1e-04 |
| mean speed | 6.26 | 6.19 | −0.645 | 5.2e-03 |
| tip-cloud area | 0.0344 | 0.0381 | **+0.799** | 1.2e-04 |
| state-covariance log-det | −0.26 | −0.15 | −0.272 | 0.29 |

The plan's nearest-corridor-state distance (normalised state units) grows *through the phrase* and
more steeply at late cycles — median by step, c20: 0.45 / 0.70 / 0.80 / 1.11 at steps 1 / 20 / 25 /
60; c55: 0.36 / 1.27 / 1.53 / **3.03**; c80: 0.43 / 1.03 / 1.20 / **3.01**. Against cycle the
whole-trajectory median is +1.36 → +1.56 (ρ=+0.414, p=0.098).

**CAL-P sweep** (8 cycles × 5 sizings, `k_shoot × cem_iters` from 1024×8 to 16384×24; S1 used
4096×12), true / believed piece error:

| cyc | 1024×8 | 2048×10 | 4096×12 | 8192×16 | 16384×24 | W0 at top |
|---|---|---|---|---|---|---|
| 0 | 0.250/0.083 | 0.261/0.087 | 0.262/0.069 | 0.309/0.081 | 0.318/0.075 | 0.487 |
| 5 | 0.445/0.130 | 0.441/0.106 | 0.426/0.087 | 0.418/0.086 | 0.433/0.068 | 0.680 |
| 15 | 0.158/0.099 | 0.152/0.087 | 0.160/0.087 | 0.180/0.088 | 0.181/0.085 | 0.348 |
| 20 | 0.168/0.081 | 0.178/0.083 | 0.166/0.086 | 0.177/0.091 | 0.209/0.071 | 0.408 |
| 25 | 0.591/0.083 | 0.589/0.090 | 0.635/0.090 | 0.729/0.089 | 0.681/0.077 | 1.205 |
| 30 | 0.379/0.084 | 0.495/0.092 | 0.559/0.086 | 0.622/0.085 | 0.615/0.075 | 1.136 |
| 55 | 0.574/0.068 | 0.620/0.075 | 0.543/0.069 | 0.477/0.073 | 0.420/0.061 | 0.915 |
| 80 | 0.463/0.072 | 0.498/0.071 | 0.446/0.066 | 0.361/0.060 | **0.166/0.044** | 0.336 |

Pooled and paired (chosen 4096×12 vs top 16384×24, n=8 cycles): `e_piece` median 0.436 → 0.369,
top better in **2/8** cycles, Wilcoxon **p=0.74**; W0 median 0.752 → 0.584, **2/8**, **p=0.64**;
W3 4/8, p=0.95. The **belief** median 0.086 → 0.073, top better in **7/8**, **p=0.023**. Split:
c0–c30 `e_piece` 0.344 → 0.375 and W0 0.494 → 0.584 (worse with search); c55/c80 0.494 → 0.293 and
W0 0.963 → 0.625 (better).

**Across all 40 CAL-P cells**: believed piece error spans 0.044–0.130 (sd 0.014) while true spans
0.152–0.729 (sd 0.175); mean gap 0.313 m; **corr(believed, true) = +0.009 (Pearson, p=0.96)**,
Spearman −0.058 (p=0.72).

**Read the individual CAL-P cells with care.** Each cell is a *single* CEM draw — the grid points
consume different amounts of randomness, so they are independent samples, and S1 measured
`e_piece` varying 0.13–0.65 between adjacent cycles at fixed sizing. Per-cell noise is therefore
of order ±0.2 m, which is larger than most of the within-row differences. The pooled and paired
statistics carry; single cells (including c80's 0.166) do not.

## Reproduce

```bash
cd experiments/            # MODAL_PROFILE=chromatic
modal run mjc/practice/span/span.py::span --quick                    # ~6 min shakeout
python3 mjc/practice/span/launch_detached.py --tag S1 --seed 0 \
    --n-cycles 80 --n-eval 32 --probe-every 4
python3 mjc/practice/span/analyze_span.py --tag S1 --fetch

modal run mjc/practice/span/s2.py::s2 --quick                         # ~2 min shakeout
python3 mjc/practice/span/launch_detached.py --fn s2 --tag S2 --src-tag S1
python3 mjc/practice/span/analyze_s2.py --tag S2 --fetch
```

`s2.py` retrains nothing: it loads `/data/practice_span/S1/fm_snapshots.pt` and regenerates every
geometry, approach plan and reference set from S1's own config and seed offsets. It is
**resumable** — `results.json` is rewritten whole after every snapshot and every CAL-P cycle, and
`--resume` (default true) skips anything already in it — and carries
`modal.Retries(max_retries=3)`, so a preempted container restarts and picks up where it stopped.
Rerunning a finished tag is a ~3 s no-op that still re-runs the fidelity check.

**Modal volume layout** (`mujoco-control-data`):

```
/data/practice_span/<tag>/results.json        # config, anchors, floors, reference, snapshots,
                                              # ladder, log, ledger  (rewritten every save_every)
/data/practice_span/<tag>/fm_snapshots.pt     # {"norm":…, "cfg":…, "weights": {cycle: state_dict}}
/data/practice_span/<tag>/done.txt            # written only on clean completion
```

`analyze_span.py --fetch` pulls that directory to `results/<tag>/` and writes five figures to
`results/<tag>/figs/`:

| figure | what it shows |
|---|---|
| `fig1_horizons.png` | both horizons vs cycle on one axis, with `e_react` / `e_ball_seg` on a log right axis, the two anchors, the one-segment / one-phrase lines and the frozen-chain ceiling |
| `fig2_curves.png` | the per-step curves at six snapshots — imagination divergence (log, 134 steps, with anchors and the chaos floor) and executed-span `goal_excess` (60 steps, with the frozen-chain floor) |
| `fig3_thresholds.png` | every horizon at all five thresholds vs cycle |
| `fig4_profile.png` | the raw within-phrase profile: distance to the step's waypoint (sawtooth) and distance from the competent reference |
| `fig5_seams.png` | the executed span in units of the piece — per-waypoint open-loop arrival error vs cycle, and consecutive waypoints delivered at four thresholds |

`analyze_s2.py --fetch` writes three more to `results/S2/figs/`:

| figure | what it shows |
|---|---|
| `s2fig1_sources.png` | the three divergence sources vs cycle, and one-step FM error on- vs off-corridor with the plan's nearest-corridor distance |
| `s2fig2_exploitation.png` | true vs predicted arrival at each waypoint, and the gap between them, vs cycle |
| `s2fig3_calp.png` | the CAL-P sweep — true (solid) and believed (dashed) piece error against search budget, one line per cycle, and W0 against budget |

## Gotchas

* **Launcher logs live in `experiments/.launch_logs/span_<tag>.log`**, deliberately outside the
  `mjc` package: `shared.py` mounts it with `add_local_python_source("mjc")` and Modal hashes the
  whole directory, so a live log inside it makes any *other* concurrent `modal run` die with
  `ExecutionError: <path> was modified during build process` — naming the log, not the broken run.
* **A horizon of `len(curve) + 1` is a measurement ceiling.** The imagination probe tops out at 134
  steps, the executed span at 60 (a phrase). The analyzer prints those as `>134` / `>60`.
* **The executed-span horizon is bounded above by the `frozen_noisy` floor**, not by 60: past the
  step where a *measured* chain re-flown with fresh noise already exceeds the threshold, no plan of
  any quality can score better. That bound is the physically meaningful one and it is legato F4's
  crossover restated as a curve.
* **The reference launch set is frozen at cycle 0.** `ladder.launch_drift` (median tip distance
  between a fresh reactive traversal's launch states and the frozen ones) is recorded every ladder
  probe so the size of that assumption is visible.
* **Modal preempts these containers.** S2's first launch died at c50 with `Container terminated
  due to preemption` in the task log while the client showed only `RemoteError: Function call was
  cancelled by user or a failure` — a preemption is indistinguishable from a user cancel on the
  caller side. Check `modal app logs <app-id>` before assuming anything else. Both runners commit
  partial results frequently; `s2.py` additionally resumes and retries.
* **The step-41 dip in `goal_excess` is a reference switch, not a recovery.** The goal schedule
  changes from W3 to W0 at step 41, so both the plan's and the reference's distances jump and their
  difference can go sharply negative. Read seam arrivals (fig 5) for what actually happens there.
* The reference reactive traversal is collected under **`fm0`**, the stale model, so the "competent
  reference" is legato's reference arm at cycle 0, not an oracle. Its own piece error is recorded
  (`reference.e_piece_lap1` / `_lap2`).
