# étude — compilation, selection, and sequential assembly on a piece with sequence structure

**Up**: [../README.md](../README.md) (practice) · [../../README.md](../../README.md) (mjc)
**Idea doc**: [practice_manufactures_its_own_credit](../../../../ideas/practice_manufactures_its_own_credit.md)
(§1 re-chunking, §5 the tempo × strictness map — its §12 points back here) ·
[performance_error_is_the_bridge](../../../../ideas/performance_error_is_the_bridge.md) §14
**Parents**: [`../../ballistic/`](../../ballistic/README.md) (Cut 4b/4c — reactive vs `ballistic_cem` vs
`ballistic_bc`; command rotations are open-loop-compensable, force kicks are not) ·
[`../../bridge_assembly/`](../../bridge_assembly/README.md) (reach regime, probe/checkpoint idioms) ·
[`../`](../README.md) (the audit that reduced δ's surviving role to **detection**)
**Status**: written up 2026-08-14 (interpretation discussed with Jasper 2026-08-13/14). Eleven runs:
three calibrations, E-gate, two discriminators, E-3, E-3b, E-4, E-5 + 2 replication seeds.
File index: [FILES.md](FILES.md). **Dates**: 2026-08-12 → 08-14.

## One-liner

On a fixed 4-segment piece with one hard passage, **sequential assembly with seam-matched selection
dominates never-compiling on both axes in 3/3 seeds** — piece error 0.0772 ± 0.0097 vs
0.1066 ± 0.0059 at 34–53% less priced practice time, beating `never`'s *best-ever* error in every
seed and the globally-accurate ceiling-FM reference by ~30%. Compilation, done right, is not a
time-for-accuracy trade: it wins both. Getting there forced the compile op through two corrections
(regression → selection → *expected-performance* selection under the consumption distribution), and
each failure mode measured on the way is a finding in its own right.

## Findings

Seed status marked per claim; the full tables live in the per-run sections below.

1. **The δ-silence certificate works as the compile trigger** (1 seed; robust across the (c, W)
   counterfactual grid). It fired at genuine mastery (0.0760, at the ceiling reference) where
   scheduled arms committed mid-descent content 1.5–1.8× worse. (§E-gate)
2. **Compilation is selection + commitment, not distillation-by-regression.** Averaging valid
   command sequences destroys them (2.1×) and BC's regression-to-the-mean compounds it (a further
   1.5×), while *any single good realisation* — one CEM plan, one executed trace replayed from
   foreign start states — commits at the replaced controller's level. s0-width, per-trace open-loop
   inconsistency and net capacity were each exonerated by direct probe. The songbird resonance:
   crystallisation keeps a rendition; it does not average the babble. (§discriminator; 1 seed, large
   margins)
3. **Selection must rank candidates by expected performance under the *consumption* distribution.**
   Two failure modes, both measured: the **winner's curse** (ranking by a candidate's own realised
   outcome selects the least transferable one — fixed by scoring uniformly-drawn candidates on
   held-out hand-over states), and the **seam-state shift** (units auditioned on slow-practice
   boundaries — 0.008 from the waypoint at speed 0.24 — are consumed at performance tempo on
   boundaries 0.107 away at speed 0.71: a 13.7×/4.5 σ offset that fully accounted for E-3b's 3×
   optimism gap). With both fixed, audition predicts realised performance (gap ≈ baseline) and the
   control arm retains the 2.8× gap. Piano wording: *play it three times in a row*, and *practice
   the transition at tempo*. (§E-3b, §E-4; attribution replicated in the E-4 control)
4. **Assembly is self-facilitating at seams and self-taxing at benchmarks.** A committed upstream
   hands over 3.7× cleaner boundary states (it was selected for accuracy) — but each commit resets
   the downstream benchmark (certification c15 → c41), so assembly cost is dominated by
   re-estimation, not learning. (§E-4; 1 seed for the timing, mechanism located in the trace)
5. **Sequential assembly compounds.** Each committed unit sets the boundary condition for every
   downstream commit: an ~11% difference at segment 1 amplified to ~43% by segment 3, and across
   seeds downstream anchored-level variance is 4–9× upstream. This is why arm comparisons on
   assembled pieces need seeds and single-segment ones largely don't. (§E-5; visible in 3-seed
   spread)
6. **The overspeed corner inverted, and sloppy practice's catastrophe was regression's property.**
   Open-loop drilling produces a *worse* candidate pool (score_med 0.137 vs 0.072) and a 55% worse
   unit at matched selection rules — closed-loop traces are the better commitment candidates here.
   Sloppy practice under the corrected op commits a ~30–56% worse unit (weak-form poison) with none
   of the 5×/off-manifold cascade the BC op produced in `eg_s0`. (§E-3b; 1 seed)
7. **Committed units do not degrade, and fusion is vacuous — for the same substrate reason.**
   Post-commit drift is exactly 0.0000 for every unit in every seed: a committed chain is
   state-independent and deterministic here, so degradation (and any accuracy cost of removing
   internal re-grounding) is structurally impossible. Consequence, stated as a precondition the idea
   doc's §2 had left implicit: **hierarchy is meaningful only over boundaries that carry
   information**. Making nested assembly real requires state-conditioned commitment (a library
   selected at launch by the observed hand-over — Schmidt's generalized-motor-program shape) or a
   re-grounding correction window; that is the next mechanism round, deliberately not patched into
   this one. (§E-5; structural)
8. **Knitting (straddle/seam drills): the time price is real, the accuracy benefit is absent.**
   +27% priced time in 3/3 seeds; the accuracy deficit is not sign-consistent (coin-flip under
   chain compounding); certification delay directionally supported (2/3 seeds). As implemented,
   across-seam practice bought nothing detectable. (§E-5 replication)

Retractions on replication, kept for the record: the `cal_s2` late ballistic erosion did not
reproduce in `eg_s0`; the e5_s0 certification-delay mechanism is directional only; the
`seam_drill_fix` seg-2 audition outlier vanished at seeds 1–2 (§E-5 replication).

## Scope

One 5-arm run (`eg_s0`, seed 0) plus three calibration runs that configured it. This is the first
node in the practice arc with **sequence** structure: the difficulty knob is composition, not
per-step severity. **No per-sample δ gain anywhere** — plasticity inside a drilled segment is plain
uniform-lr Adam; δ is consumed only as a detector.

## Substrate

Puck-free corridor world (`../../pusher_env.py`, `gear` 10, `joint_damping` 2.0, `frame_skip` 12) —
the Cut-4c / `bridge_assembly` regime. The **piece** is a fixed closed square loop, waypoints
(±0.4, ±0.4): 4 segments of length 0.8, `seg_H = 34` steps each. Segment 1 carries a localized
**command rotation** (`rot_regions`, φ = 1.2, σ = 0.16, centred on its midpoint); the region's gate
weight at the neighbouring segments' shared waypoints is 0.044 and at segment 3 is 0.000. Segments
run back-to-back **from the true achieved state** — boundaries are re-grounding points, not
teleports. Units are spans (`fuse_units(i, j)` exists for phrase formation; E-gate uses
single-segment units only).

- **Metering**: `e_k` = boundary error of segment *k* on an at-tempo run-through (ballistic within
  segment, re-grounding at boundaries), `n_rt = 64` performers. `b_k` = per-segment tabular EWMA
  (α = 0.2). `δ_k = b_k − e_k`. **δ-silence** = `|mean(δ over W)| < c·scale` **and**
  `sd(e over W) < c_v·scale`, held `sil_hold` cycles, with
  `scale = max(ref_stale_k − min_so_far_k, b_k)` where `ref_stale_k` is the setup stale reference at
  performance tempo. Run configured at c = 0.06, c_v = 0.10, W = 5, hold = 2. τ-from-pretrain-MAD is
  deliberately not reused.
- **Compile op**: `ballistic_bc` — π(s0, g) → the whole 34×2 open-loop command sequence, fit on the
  agent's own recent **executed** traces of that segment (trace window 8 cycles × 32 performers),
  then routing switches for practice, run-throughs and every tempo. Frozen after compile
  (`recompile_every = 0`).
- **FM diet**: pretrained on a pool collected in the **true** world with in-region samples rejected
  (`pretrain_mode = exclude`, gate > 0.1), so the hard passage is unmodelled rather than
  wrongly-modelled; online batches are 50% fresh practice experience / 50% replay of that pool
  (`replay_frac = 0.5`); practice commands carry motor noise (`explore_sigma = 0.25`), run-throughs
  and ladder probes do not.
- **Time pricing**: a control step costs `dt = frame_skip × timestep = 0.024 s`; one feedback/replan
  event costs a declared sensorimotor delay `d_fb = 0.10 s`. A reactive segment (34 events) costs
  4.2 s; a committed segment (1 event, at the boundary) costs 0.92 s. Plan counts are logged
  separately so a deliberation price can be applied post-hoc.

## Arms (the compile trigger is the only thing that differs)

| arm | trigger | notes |
|---|---|---|
| `never` | — | pure reactive routing; pays the full feedback-time price throughout |
| `delta_gate` | δ-silence on the drilled segment | |
| `sched_early` | fixed cycle 3 | |
| `sched_late` | fixed cycle 42 | identical stream to `never` until it fires |
| `sloppy` | fixed cycle 3 | drilled segment practised **open-loop** (`replan_every = H`) throughout, so its traces carry uncorrected error — §5's poisoned region, manufactured |

## Instrument checks (`eg_s0`, seed 0)

| probe | stale FM | ceiling FM |
|---|---|---|
| FM error in-region / clean | 0.1419 / 0.0064 | 0.0036 / 0.0018 |
| drilled seg, reactive (R=1) | 0.0216 | 0.0090 |
| drilled seg, R=12 | 0.0904 | 0.0090 |
| drilled seg, performance tempo (R=34) | 0.1631 | 0.0805 |
| clean segs, R=34 | 0.0529–0.1282 | 0.1144–0.1351 |

Usable stale→ceiling range on the drilled segment at performance tempo: **0.0826**. The three clean
segments sit at an irreducible open-loop floor of ~0.11–0.13, which dilutes a piece-mean ~4×, so the
**drilled segment is the primary readout** and the piece mean is reported alongside.

A **corridor-slice FM probe** rides the metering run-through: FM error on exactly the (s, u) pairs
the committed corridor traverses, logged next to the ceiling FM's error on the identical slice. It
exists because the broad `fm_region` probe (uniform position × N(0, 1.2) velocity × uniform command)
averages over a state×command volume the corridor never visits, and the two diverge by construction.

## Completion status

**No arm reached the configured 56 cycles.** The detached client died at ~c45 with a network error
and the app went down; the last *committed* milestone is **c42** (`never`, `sched_late`) or **c49**
(`delta_gate`, `sched_early`, `sloppy`). All five files carry `complete: false` and no `done.txt`.
Reductions are reported at the **common horizon c42**, with c49 shown where available. `sched_late`
compiled at exactly c42, so it has one post-compile ladder point and no post-compile metering.

## Runs on disk

| tag | what it is |
|---|---|
| `cal_s0` | `--arms never --n-cycles 12`, `pretrain_mode=clean`, no replay, no motor noise |
| `cal_s1` | `--n-cycles 16` with `pretrain_mode=exclude`, `replay_frac=0.5`, `explore_sigma=0.25`, `n_rt=40` |
| `cal_s2` | `--n-cycles 30 --n-grad 10`, corridor-slice probe + window-mean silence test, `n_rt=96` |
| `eg_s0` | the E-gate run, 5 arms, seed 0 (truncated at c42 common / c49) |
| `disc_c15` | compile-hit discriminator part 1 — three BC target families at `delta_gate`'s c15 certificate state |
| `disc2_c15` | discriminator part 2 — pure-evaluation probes at the same state |
| `e5_s0` | E-5: extended-horizon consolidation (c78) + `seam_drill_fix`. 3 arms, seed 0 (complete) |
| `e4_s0` | E-4: sequential phrase assembly with seam-matched selection. 4 arms, seed 0 (complete, c55) |
| `e3b_s0` | E-3b: selection by expected performance, calibrated certificate anchor, noise-free plan centroid. 5 arms, seed 0 |
| `e3_s0` | E-3: selection-based commitment, certificate-anchored benchmark, overspeed drill. 5 arms, seed 0 |

## Reproduce

```bash
cd experiments/

# smoke
modal run mjc/practice/etude/etude.py::etude --quick --arms "never,delta_gate"

# calibrations (each writes /data/practice_etude/<tag>/never/)
python3 mjc/practice/etude/launch_detached.py --tag cal_s0 --arms never --n-cycles 12 \
    --probe-every 4 --seed 0 --pretrain-mode clean --replay-frac 0.0 --explore-sigma 0.0 \
    --n-grad 40 --n-rt 24 --n-eval 32
python3 mjc/practice/etude/launch_detached.py --tag cal_s1 --arms never --n-cycles 16 \
    --probe-every 8 --seed 0 --n-grad 40 --n-rt 40
python3 mjc/practice/etude/launch_detached.py --tag cal_s2 --arms never --n-cycles 30 \
    --probe-every 6 --seed 0 --n-grad 10 --n-rt 96

# the E-gate run
python3 mjc/practice/etude/launch_detached.py --tag eg_s0 --seed 0 \
    --arms "never,delta_gate,sched_early,sched_late,sloppy" \
    --n-cycles 56 --probe-every 7 --n-grad 5 --n-rt 64 --sched-early 3 --sched-late 42

# reduction (--fetch pulls from the volume; the local mirror is only written if the
# local entrypoint survives, which for eg_s0 it did not)
python3 mjc/practice/etude/analyze_etude.py --tag eg_s0 --fetch --figures
```

Modal volume (`mujoco-control-data`): `/data/practice_etude/<tag>/<arm>/results.json`.
Figures: `figures/<tag>/fig1_metering.png` (per-segment `e_k` and `δ_k`, compile events marked),
`fig2_priced.png` (cumulative practice time × drilled error at performance tempo),
`fig3_ladder.png` (final tempo ladder vs both references),
`fig4_clean.png` (clean-segment interference, FM region/clean probes, held-out clean ladder).

## The compile-hit discriminator (`disc_c15`, `disc2_c15`)

`eg_s0` degraded the drilled segment at every compile event, so the compile op itself was probed.
Neither run persists state, but the compile state is exactly reproducible: `never` and `delta_gate`
differ only in the trigger, so a seeded 15-cycle replay of `never` reconstructs the FM and trace
buffer `delta_gate` compiled from at c15 (the only persistent RNG in the loop is the training-batch
generator, which ladder probes never touch, so `--probe-every` does not perturb the trajectory).
All numbers are the drilled segment's boundary error at performance tempo (R34), `n_eval=96`.

| committed unit | drilled err |
|---|---|
| per-s0 CEM replanning (`cem_ballistic`, the controller being replaced) | 0.0807 |
| reactive (R1), for scale | 0.0099 |
| **one** fixed CEM plan from the boundary centroid, all episodes | 0.0889 |
| **one** individual executed trace, replayed from *other* traces' s0 (cross-s0) | **0.0763** |
| the same trace replayed from its own s0 (determinism control) | 0.0191 |
| mean of the executed / pre-noise command sequences | 0.1596 / 0.1613 |
| BC fit to executed / pre-noise / CEM-plan targets | 0.2413 / 0.2987 / 0.2891 |
| BC on CEM targets at 4x width | 0.2928 |

Hand-over boundary spread at the drilled unit: positional sd 0.0326.

Exonerated: **s0 width** (a single committed program scores 0.076–0.089 across the whole boundary
distribution), **per-trace open-loop inconsistency** (an individual closed-loop trace's commands are
a viable committed program from *other* start states), **capacity** (4x width is no better). What
remains is **averaging**: the mean of valid command sequences is not a valid sequence (2.1x), and
regression-to-the-mean via BC compounds it (a further 1.5x). Hence E-3's compile op is selection +
verbatim commitment. The resonance worth recording: the songbird crystallises a rendition; it does
not average its babble.

## E-3 (`e3_s0`)

Same substrate, piece and metering. Three things change.

- **The compile op**: `select` = commit verbatim the recent realisation with the lowest *measured*
  boundary error (`k_best`); `plan` = commit one full-horizon CEM plan from the centroid of the
  recent hand-over distribution. Both produce a `fixed` unit — one command sequence issued at the
  boundary, no net.
- **Certificate-anchored benchmark** (`--cert-anchor`): `b_k` freezes at its value when the
  certificate fires, so post-commit degradation stays visible as sustained δ<0 instead of being
  habituated away. In `eg_s0` an unanchored `b` climbed 0.089 → 0.257 within ~15 cycles of the
  commit and re-declared the eroded unit "mastered" (silence streak 20). `--recert-on-neg` allows
  exactly one re-selection on sustained δ<0.
- **The overspeed drill** (`--overspeed-cycles`): the certificate opens a forced-commitment window
  in which the segment is practised open-loop, and the unit is selected from *those* realisations
  rather than from closed-loop corrective traces — §5's overspeed corner.

Arms: `never` (reference) · `gate_select` · `gate_plan` · `gate_over_select` · `sloppy_select`
(sloppy practice throughout, selection at c3 — does the poisoned region survive an op that commits
a real realisation rather than a regression?). FM state dicts and trace buffers
(`fm_c<N>.pt`, `traces_c<N>.npz`) are persisted at every commit event.

### E-3 results (`e3_s0`, seed 0, all five arms complete to c45)

Commit events. `chosen_err` is the pool metric selection ranked on; `pool_med` the pool median.

| arm | commit | pool | chosen_err / pool_med | recert (δ<0) |
|---|---|---|---|---|
| `gate_select` | c15, certificate | 256 closed-loop traces | 0.0007 / 0.0197 | c20, 0.0024 / 0.0886 |
| `gate_plan` | c15, certificate | one CEM plan, practice-boundary centroid | — | c20 |
| `gate_over_select` | c19, after a 4-cycle open-loop drill | 128 open-loop realisations | 0.0028 / 0.1503 | c24, 0.0112 / 0.1270 |
| `sloppy_select` | c3, schedule | 96 traces | 0.0514 / 0.2203 | c8, 0.0100 / 0.1743 |

Drilled segment at performance tempo (R34), held-out ladder:

| arm | c15 | c20 | c30 | c45 | t_cum (s) | E_piece@R34 | E_react | clean@R34 |
|---|---|---|---|---|---|---|---|---|
| `never` | 0.0836 | 0.0601 | 0.0700 | **0.0646** | 24284 | 0.1013 | 0.0089 | 0.1135 |
| `gate_select` | 0.1074 | 0.1297 | 0.1267 | 0.1335 | 21116 | 0.1106 | 0.0133 | 0.1030 |
| `gate_over_select` | 0.0836 | 0.1804 | 0.1722 | 0.1459 | 21116 | 0.1223 | 0.0222 | 0.1144 |
| `sloppy_select` | 0.1708 | 0.1701 | 0.1704 | 0.1738 | 19532 | 0.1157 | 0.0241 | 0.0963 |
| `gate_plan` | 0.2607 | 0.2598 | 0.2535 | 0.2541 | 21116 | 0.1499 | 0.0579 | 0.1151 |

Post-commit temporal drift (mean of first 5 post-commit cycles → mean of last 10):
`gate_select` +0.0233, `gate_plan` −0.0020, `gate_over_select` −0.0097, `sloppy_select` −0.0265.

Downstream clean segment 2 at R34, first → last milestone: `never` 0.0529 → 0.1225,
`gate_select` → 0.1249, `gate_plan` → 0.1054, `gate_over_select` → 0.1234,
`sloppy_select` → 0.0536.

## E-3b (`e3b_s0`)

Three corrections, each traced to a measured failure in `e3_s0`.

1. **Selection by expected performance.** `n_cand=32` candidates are drawn **uniformly** from the
   trace pool (not top-of-pool: ranking by a candidate's own realised error is a winner's curse —
   it favours the sequence most finely tuned to its own start state, i.e. the least transferable).
   Each is replayed open-loop from `n_score=16` held-out recent hand-over states — the repeatability
   criterion — and the argmin of the **mean** is committed. The scoring rollouts are **priced into
   `t_cum`** (`n_cand x (n_score-1)` committed segment traversals, ~440 s per commit against a
   ~21,000 s run). Events record `chosen_own_err`, `chosen_score`, `score_med` and
   `score_of_best_own_err` — the last is the counterfactual: what the old min-own-error rule would
   have committed, scored on the new metric, so the curse's removal is directly measurable.
2. **The certificate anchor is calibrated, not inherited.** On commit, `b_k` is released for
   `cert_cal=3` cycles, the committed unit's own level is measured, and `b_k` is anchored to that
   mean. In `e3_s0` `b` froze at the pre-commit CEM-ballistic level, so δ went negative on the first
   post-commit cycle by exactly the level gap and stayed flat — mechanical, not temporal, and the
   recert could not mean "degradation". One capped recert is retained.
3. **`plan_eval` commits from a noise-free hand-over estimate** — the metering run-through's own
   hand-over states — rather than the practice-boundary centroid, which carries upstream motor
   noise. This tests the boundary-mismatch explanation of `gate_plan`'s 2.9x penalty
   (0.254 vs disc2 probe (a)'s 0.0889, which differed in exactly that respect).

Arms: `never` · `gate_select_x` · `gate_plan_eval` · `gate_over_select_x` (overspeed drill + fixes
1–2, which deconfounds the overspeed question `e3_s0` left unanswerable) · `sloppy_select_x`.

**Asymmetry to note when reading**: candidate scoring uses *practice* hand-over states (purely
self-generated, and slightly wider than evaluation because of upstream motor noise, so selection is
if anything conservative), while `plan_eval` uses the run-through's noise-free hand-over states, as
specified. Both are observable to the agent; neither reads the held-out eval geometry.

### E-3b results (`e3b_s0`, seed 0, all five arms complete to c45)

Commit events (no recert fired in any arm):

| arm | commit | chosen own_err | chosen **score** | pool score_med | counterfactual: score of the min-own-err pick | t_score priced |
|---|---|---|---|---|---|---|
| `gate_select_x` | c15 | 0.0359 | **0.0411** | 0.0717 | **0.1416** | 440 s |
| `gate_over_select_x` | c19 (after 4-cycle open-loop drill) | 0.0857 | 0.0634 | 0.1366 | 0.0765 | 440 s |
| `sloppy_select_x` | c3 | 0.0514 | 0.0729 | 0.2366 | 0.0729 | 440 s |
| `gate_plan_eval` | c15 | — | — | — | — | 0 (hand-over spread 0.0278) |

Drilled segment @R34, held-out ladder, and the priced grade at c45:

| arm | c15 | c25 | c45 | t_cum (s) | E_piece@R34 | E_react | clean@R34 | seg2@R34 |
|---|---|---|---|---|---|---|---|---|
| `never` | 0.0836 | 0.0793 | **0.0646** | 24284 | 0.1013 | 0.0089 | 0.1135 | 0.1225 |
| `gate_select_x` | 0.1243 | 0.1302 | 0.1229 | 21556 | 0.1190 | 0.0139 | 0.1177 | 0.1578 |
| `gate_plan_eval` | 0.1387 | 0.1388 | 0.1331 | 21116 | 0.1205 | 0.0739 | 0.1163 | 0.1051 |
| `gate_over_select_x` | 0.0836 | 0.2187 | 0.1909 | 21556 | 0.1172 | 0.0203 | 0.0927 | 0.0577 |
| `sloppy_select_x` | 0.1953 | 0.1974 | 0.1917 | 19972 | 0.1182 | 0.0220 | 0.0938 | 0.0570 |

Post-commit drift (first-5 → last-10 mean): −0.0031, −0.0030, +0.0022, −0.0058. Max `neg_run` = 1
in every arm (recert needs 5), so **no recert fired** — with `b` anchored to the committed unit's
own level, δ stays at ~0 and the units do not degrade.

### The seam-state shift (measured, `e3b_s0` + `disc2_c15`)

Selection scores 0.041–0.073; the same units realise 0.125–0.213 in the metering run-through — a
3.0× optimism gap. The persisted hand-over states locate it, and it is an **offset**, not a width:

| hand-over into the drilled segment, c15 | positional displacement from the waypoint | positional spread | mean speed |
|---|---|---|---|
| **practice** (upstream executed reactively, R1) | 0.0078 | 0.0220 | 0.2415 |
| **performance tempo** (upstream executed open-loop, R34) | **0.1065** | 0.0326 | **0.7129** |

The unit is selected on hand-over states 0.008 from the waypoint at speed 0.24 and consumed on
states 0.107 from it at speed 0.71 — a 13.7× positional offset, **4.5 σ of the practice hand-over
spread**, with 3.0× the arrival speed. The performance hand-over is also *wider* (0.0326 vs 0.0220),
not narrower: reactive upstream execution corrects the motor noise away, while open-loop upstream
execution both displaces and widens what the next segment inherits.

Across the five arms, downstream seg2 error is **negatively** correlated with drilled-segment error
(Pearson r = −0.77): arms whose committed unit ends further from the waypoint hand over a state that
suits seg2's open-loop execution better. So the `e3_s0` downstream observation reproduces
(`sloppy_select_x` 0.0570 vs `sloppy_select` 0.0536) but does not track commitment as such.

## E-4 (`e4_s0`)

**Sequential phrase assembly.** Segments are committed **in piece order**: segment *k* is eligible
only once every segment before it is committed, and each still needs its own δ-silence certificate.
The candidate pool for *k* is scored on hand-over states produced by the **current performance
configuration** — already-committed upstream executed at tempo, reactive remainder — which is the
online form of the seam-matched selection fix e3b's measurement implies.

One design fact worth recording, because it shapes the arms: under sequential assembly the seam fix
is *partly automatic*. Routing switches globally on commit, so once upstream is committed, practice
traversals already execute it at tempo and segment *k*'s own traces already start from seam-correct
states. The arms therefore differ only in **how the score set is generated**:

| arm | score set for segment *k* | |
|---|---|---|
| `seq_seam` | traversal at performance tempo with current commitments honoured | the fix |
| `seq_practice` | traversal with commitments ignored **and R=1** — e3b's reactive-upstream hand-over | the attribution control |
| `seam_drill` | as `seq_seam`, plus straddle practice (below) | Jasper's knitting intuition |
| `never` | — | reference |

The control must force `replan_every=1`, not merely ignore commitments: uncommitted-at-tempo is
still ballistic and would not reproduce e3b's distribution. Verified in smoke — at segment 1 the two
score sets separate (at-tempo hand-over speed 0.600 vs reactive 0.165, against e3b's 0.71/0.24),
while at segment 0 they are *identical by construction* (no upstream), which is a free sanity check.

**The seam drill** (`seam_drill`): extra practice windows that straddle a committed boundary — start
one segment before the seam, execute the committed unit at tempo, cross into the next segment and
practise it reactively. Half a piece per traversal, so it concentrates seam-conditioned experience
per unit of practice time, priced honestly into `t_cum`. Additive to the normal full-piece
traversal, so the arm buys more seam data at more time and the priced grade adjudicates.

**Built-in precondition check**: every commit logs `chosen_score` and every anchor logs the
committed unit's measured level as an `anchor` event, so the per-commit **optimism gap**
(anchored level ÷ chosen_score) is read directly. e3b's gap was 3.0x / 3.4x / 2.7x. On the first
seam-matched commit it should collapse; if it does not, the seam account is incomplete and this
line stops for discussion rather than iterating.

Commits also log `score_s0_disp`, `score_s0_spread` and `score_s0_speed` — the score set's own
hand-over statistics — so the distribution each unit was selected against is on the record.

Not in this run: nested fusion (`fuse_units` into meta-segments) — the round after, if seams close.
`--recert-on-neg` is off, since the recert path is scoped to the fixed drilled segment and would sit
oddly against a moving commit target; the δ<0 path is still instrumented via `neg_run`.

### E-4 results (`e4_s0`, seed 0, four arms complete to c55)

**Logging note.** The `compile` event's `seg` field and the persisted trace filenames were written
against the fixed `drill` index rather than the moving commit target, so every commit is labelled
`seg 1` and the npz files carry only the cycle. The *behaviour* was correct — the sequential rule
commits in piece order, so the i-th compile event is segment i, and the `anchor` events (which carry
the true segment) confirm the pairing exactly. Fixed in `etude.py` for future runs; the table below
uses the corrected attribution.

**The optimism gap — the precondition.** Anchored level ÷ `chosen_score`:

| arm | seg | commit | score | anchored | **gap** | counterfactual (min-own-err pick) |
|---|---|---|---|---|---|---|
| all three | 0 | c8 | 0.0683 | 0.0570 | **0.83** | 0.0734 |
| `seq_seam` | 1 | c41 | 0.0594 | 0.0562 | **0.95** | 0.1749 |
| `seq_practice` | 1 | c41 | 0.0357 | 0.1010 | **2.82** | 0.0805 |
| `seam_drill` | 1 | c50 | 0.0851 | 0.0756 | **0.89** | 0.2409 |
| `seam_drill` | 2 | c51 | 0.0821 | 0.0936 | 1.14 | 0.1408 |
| `seq_seam` | 2 / 3 | c54 / c55 | 0.1050 / 0.0817 | — (window ran past the horizon) | — | 0.2209 / 0.1029 |
| `seq_practice` | 2 / 3 | c53 / c54 | 0.0461 / 0.0393 | — | — | 0.7028 / 0.0635 |

Segment 0's commit is **identical across all three arms** (0.0683 → 0.0570) — the built-in sanity
check: with no upstream, the seam and reactive score sets coincide by construction. Segment 1 is the
discriminating cell: **0.95 under seam-matched selection vs 2.82 under the control**, against e3b's
3.0 / 3.4 / 2.7. Gaps below 1 mean the realised level is slightly *better* than the score.

`seq_seam`'s committed segment 1 (0.0562) sits below the ceiling-FM reference (0.0805) and below its
own pre-commit at-tempo level (~0.115).

**Per-segment ladder @R34 at c55**, against both references:

| arm | s0 | s1 | s2 | s3 | piece |
|---|---|---|---|---|---|
| `never` | 0.0905 | 0.0864 | 0.1089 | 0.1304 | 0.1041 |
| `seq_seam` | 0.0590 | 0.0577 | 0.0777 | 0.0868 | **0.0703** |
| `seq_practice` | 0.0590 | 0.1015 | 0.1614 | 0.1758 | 0.1244 |
| `seam_drill` | 0.0590 | 0.0758 | 0.1049 | 0.1207 | 0.0901 |
| ceiling FM | 0.1144 | 0.0805 | 0.1186 | 0.1351 | 0.1122 |
| stale FM | 0.1247 | 0.1631 | 0.0529 | 0.1282 | 0.1172 |

**Priced grade, piece level.** `never` 29681 s / 0.1041; `seq_seam` 25009 s / 0.0703;
`seq_practice` 24798 s / 0.1244; `seam_drill` 29033 s / 0.0901. Interpolating `never`'s own
trajectory to `seq_seam`'s final priced time (25009 s) gives 0.1037, against `seq_seam`'s 0.0703.
**Caveat**: `seq_seam`'s piece error is 0.0915 at c50 and drops to 0.0703 only at c55, immediately
after segments 2 and 3 commit — the piece-level margin rests on two commits that are one cycle old
and whose anchor windows never completed.

`E_react` is no longer a fallback readout for committed arms: a committed unit executes open-loop at
every tempo, so for a fully-assembled piece `E_react` converges to `E_piece` by construction.

**Assembly timing.** `seq_seam` certificated segment 1 at **c41**, against c15 in `e3b`. The trace
locates it: segment 0 commits at c8, and at c9 segment 1's at-tempo error steps from 0.0957 to
0.1289 and its silence streak resets to 0, where it stays until c40. Each commit perturbs the
*downstream* segment's boundary distribution, invalidating its benchmark and forcing re-certification
from scratch — so assembly cost is dominated by re-stabilisation, not by learning. Note the seam
itself *improves* with assembly: segment 1's score-set hand-over under a committed segment 0 has
displacement 0.0287 at speed 0.293, against the 0.1065 / 0.713 that an open-loop CEM segment 0
delivered in `e3b`, because the committed upstream unit was itself selected for accuracy.

**`seam_drill` — confounded by an instrument artifact, not a clean test.** It committed segment 1
nine cycles later (c50 vs c41), 34% worse (0.0756 vs 0.0562), paid +14% priced time, and reached
only three commits. But its FM is not worse (`fm_region` 0.1295 vs 0.1286, `fm_clean` 0.0115 vs
0.0107, corridor-slice 0.0277 vs 0.0396 — better), and its optimism gap was fine (0.89), so the seam
machinery worked inside it. The identifiable cause is the trace window: straddle traces are appended
to the same fixed-length 8-entry buffer, so they displaced full-piece traces and the pool at commit
was **192 candidates spanning 4 cycles** instead of 256 spanning 8. More seam experience was bought
with a shorter memory. The knitting intuition is untested; the fix is a separate straddle buffer.

## E-5 (`e5_s0`) — and why the fusion arm was not run

Arms `never` / `seq_seam` / `seam_drill_fix`, seed 0, **c78** — a horizon chosen so every commit in
e4's chain (seg 1 at c41, segs 2–3 at c54/c55) gets a completed 3-cycle anchor window plus ≥10
post-commit cycles, which settles e4's late-commit caveat.

`seam_drill_fix` gives the straddle traces their **own** buffer, so they add candidates to the
selection pool instead of displacing full-piece traces from the fixed-length window. Verified in
smoke: pools now grow (36, 44) where the arm's e4 ancestor shrank to 192. This is what makes the
knitting intuition actually testable.

### E-5 results (`e5_s0`, seed 0, three arms complete to c78)

`seq_seam` reproduces `e4_s0` exactly through c55 and its late commits now anchor.

| arm | seg | commit | pool | score | anchored | gap | anchor@ | post-commit drift |
|---|---|---|---|---|---|---|---|---|
| `seq_seam` | 0 | c8 | 256 | 0.0683 | 0.0570 | 0.83 | c11 | +0.0000 (67 cyc) |
| `seq_seam` | 1 | c41 | 256 | 0.0594 | 0.0562 | 0.95 | c44 | +0.0000 (34 cyc) |
| `seq_seam` | 2 | c54 | 256 | 0.1050 | 0.0751 | 0.71 | c57 | +0.0000 (21 cyc) |
| `seq_seam` | 3 | c55 | 256 | 0.0817 | 0.0678 | 0.83 | c58 | +0.0000 (20 cyc) |
| `seam_drill_fix` | 0 | c8 | 256 | 0.0683 | 0.0570 | 0.83 | c11 | +0.0000 |
| `seam_drill_fix` | 1 | c50 | 384 | 0.0711 | 0.0632 | 0.89 | c53 | +0.0000 |
| `seam_drill_fix` | 2 | c60 | 384 | 0.1069 | 0.1149 | **1.08** | c63 | +0.0000 |
| `seam_drill_fix` | 3 | c68 | 384 | 0.1361 | 0.1183 | 0.87 | c71 | +0.0000 |

**Post-commit drift is exactly zero, to four decimals, for every unit.** Once a unit and all of its
upstream are committed, the whole chain is a deterministic sequence of state-independent command
sequences in a deterministic env against a fixed run-through geometry, so its metering value is a
*constant*. Degradation of a committed unit is not merely absent here, it is impossible — which is
the strongest form of e3b's (ii)-not-(i) verdict and the same substrate fact that makes fusion
vacuous (below).

**Reading the gap.** The audition score is a **mean** over the score states; the anchored level is
the **median** over 64 run-through performers, and open-loop boundary error is right-skewed
(median/mean = 0.82 for `seq_seam`, 0.87 for `seam_drill_fix`, 1.00 for the uncommitted `never`).
So a gap of ~0.85 is the *like-for-unlike baseline*, not a finding, and the signal is deviation from
it: `seq_seam`'s 0.71–0.95 sit at or below baseline, `seam_drill_fix` seg 2's 1.08 is ~24% above its
own baseline, and `seq_practice`'s 2.82 in e4 is ~3.4x it. The metric should be made like-for-like
(both mean or both median) in any follow-up.

**Piece-level priced verdict at c78** (the arc's headline):

| arm | t_cum | E_piece@R34 | s0 | s1 | s2 | s3 |
|---|---|---|---|---|---|---|
| `never` | 42093 | 0.1087 | 0.1162 | 0.0624 | 0.1213 | 0.1348 |
| `seq_seam` | **27706** | **0.0703** | 0.0590 | 0.0577 | 0.0777 | 0.0868 |
| `seam_drill_fix` | 35590 | 0.0958 | 0.0590 | 0.0848 | 0.1170 | 0.1224 |
| ceiling FM | — | 0.1122 | 0.1144 | 0.0805 | 0.1186 | 0.1351 |

`seq_seam` reaches 0.0703 at 27706 s. Interpolating `never`'s own trajectory to that priced time
gives 0.1070, and `never`'s *best* piece error anywhere in its run is 0.0969 (at t=16189). So
sequential assembly with seam-matched selection is **34% better at 34% less priced time, and 27%
better than `never` ever gets**, while also beating the ceiling-FM reference by 37%.

**The knitting verdict is negative, with a mechanism.** The buffer artifact is fixed (pools grew to
384), and the deficit persists on every axis: `seam_drill_fix` certificates later on every segment
(c50/c60/c68 vs c41/c54/c55), costs **+28% priced time** (35590 vs 27706), and its larger pool did
not buy better candidates (scores 0.0711/0.1069/0.1361 vs 0.0594/0.1050/0.0817). Two contributing
mechanisms are identifiable:

- *Certification delay.* Pre-commit metering variance on segment 1 is markedly higher
  (cv 0.166 vs 0.108) — the straddle traversals make the FM's diet heterogeneous within a cycle, so
  the run-through error is noisier and the silence detector's variance condition takes longer to
  satisfy. Segments 2–3 have comparable cv (0.096/0.088 vs 0.084/0.085), so their delay is mostly
  cascade from segment 1.
- *Chain compounding.* A slightly worse committed segment 1 (0.0632 vs 0.0562) hands segment 2 a
  worse boundary (displacement 0.0216 vs 0.0106), which yields a worse segment 2 (0.1149 vs 0.0751),
  which hands segment 3 a worse boundary (0.0731 vs 0.0565) and a worse segment 3 (0.1183 vs
  0.0678). **Sequential assembly compounds**: each committed unit sets the boundary condition for
  every downstream commit, so an 11% difference at segment 1 becomes 43% by segment 3.

Honest limit: the accuracy half of the deficit traces to a single ~11% difference at segment 1
amplified by compounding, which at one seed is not separable from selection noise. The time cost and
the certification delay are not noise.

### E-5 seed replication (`e5_s0/s1/s2`, seeds 0-2, three arms x c78)

Runner bit-identical across the three tags. Aggregation: `analyze_etude.py --tags e5_s0,e5_s1,e5_s2`.

| arm | t_cum (mean ± sd) | E_piece@R34 (mean ± sd) | per seed |
|---|---|---|---|
| `never` | 42093 ± 0 | 0.1066 ± 0.0059 | 0.1087 / 0.1125 / 0.0986 |
| `seq_seam` | 23729 ± 3233 | **0.0772 ± 0.0097** | 0.0703 / 0.0909 / 0.0705 |
| `seam_drill_fix` | 30122 ± 3988 | 0.0863 ± 0.0123 | 0.0958 / 0.0942 / 0.0689 |

**Headline margin** (`never − seq_seam`): +0.0384 / +0.0216 / +0.0281, mean **+0.0293**, **3/3 same
sign**. Matched-priced-time comparison per seed (interpolating `never`'s own trajectory to
`seq_seam`'s final t_cum): +0.0367 / +0.0169 / +0.0244; against `never`'s *best-ever* piece error:
+0.0266 / +0.0101 / +0.0203, 3/3. Time saved: 34% / 53% / 44%.

**Knitting deficit** (`seam_drill_fix − seq_seam`): +0.0255 / +0.0033 / **−0.0015**, mean +0.0091,
**not sign-consistent**. Time price: +7884 / +8799 / +2498 s, mean **+6394 s**, 3/3 same sign.

**Certification delay** (`seam_drill_fix` minus `seq_seam` commit cycle, per segment):
seed 0 [0, +9, +6, +13], seed 1 [0, +9, +14, +23], seed 2 [0, −1, +2, −3]. 2/3 seeds show a large
monotone delay; seed 2 shows none.

**Anchored levels across seeds** (mean ± sd): `seq_seam` s0 0.0584 ± 0.0011, s1 0.0640 ± 0.0089,
s2 0.0780 ± 0.0022, s3 0.0834 ± 0.0118; `seam_drill_fix` s0 0.0584 ± 0.0011, s1 0.0590 ± 0.0055,
s2 0.0958 ± 0.0187, s3 0.1006 ± 0.0194. Segment 0 is identical across arms in every seed (no
upstream — the built-in sanity check). Downstream variance is ~4-9x upstream variance in both arms,
which is chain compounding visible in the seed spread.

**The `seam_drill_fix` seg-2 audition cell** (baseline-corrected gap): 1.17 / 0.93 / 0.94 — the
e5_s0 outlier does not reproduce.

**Post-commit drift remains exactly 0.0000 for every committed unit in every seed.**

### The fusion arm is provably vacuous on this substrate — reported, not run

A committed unit is a **state-independent** open-loop command sequence: in `traverse`, a `fixed`
unit sets `plan = np.tile(unit["fixed"][None], ...)` and applies `plan[:, h, :]` without ever
reading `states`. Two adjacent committed units therefore produce a **bit-identical trajectory** to
their concatenation executed as one span. Removing the internal re-grounding event is a no-op on the
physics:

- accuracy cost of fusion: **exactly zero**, by construction, not by measurement;
- time win: **exactly** one `d_fb` per fusion per traversal. For this piece at performance tempo,
  4 units cost 4·(34·0.024) + 4·0.1 = 3.664 s; 4→2 gives 3.464 s (−5.5%); 4→1 gives 3.364 s (−8.2%).

So a seam certificate could never refuse a fusion, and the run would report a bookkeeping identity.

The escape hatch of re-selecting at the phrase level also collapses: once both parts are committed,
practice executes them as fixed sequences, so their traces *are* those sequences (plus motor noise)
and the phrase pool contains nothing but the concatenation.

**Where the missing content actually is.** The time model charges a committed unit one feedback
event per segment, but the unit never consumes the observation that event pays for. That
inconsistency is exactly the gap: making fusion meaningful requires giving the boundary a real
feedback role — either committing a small **library** selected at launch by the observed hand-over
(state-conditioned commitment), or spending the charged `d_fb` on a brief re-grounding correction
window before the sequence launches. Either change alters committed-unit semantics for every arm, so
it is a design decision for the next round, not a patch to this one.

## Calibration record (what each cal changed, and why)

Recorded because the run's configuration is not self-explanatory and every knob below was set by a
measurement, not a guess.

| cal | measured | change it forced |
|---|---|---|
| `cal_s0` | `fm_clean` 0.0064 → 0.0674 over 12 cycles; `fm_region` flat | no replay of the pretrain pool → catastrophic forgetting under a narrow on-policy diet. Added `replay_frac`; and since the clean-world pool's in-region samples would fight adaptation, added `pretrain_mode=exclude` |
| `cal_s1` | `fm_clean` flat 0.011–0.013 ✓; `fm_region` flat at ~0.13 ✗; but drilled R34 0.1631 → 0.0530, **below** the ceiling FM's 0.0805 | the broad `fm_region` criterion is mis-specified (spatial matching: a corridor-specialised FM beats a globally-accurate one on the corridor). Added the corridor-slice probe. Detector scale was ~2× too strict (took the stale range from cycle 1, after a cycle of adaptation, instead of the setup reference); instantaneous \|δ\| test replaced by the window-mean test |
| `cal_s2` | 88% of the stale→ceiling gap closed by **cycle 2**, 117% by cycle 3 | descent far too compressed. `n_grad` 40 → 10 → 5. Also measured a monotone ballistic-only late drift on `never` (R34 0.0738 → 0.1176 across c6–c30 while R1, R12 and practice-tempo error stayed flat) — **which did not reproduce in `eg_s0`** (see below) |
