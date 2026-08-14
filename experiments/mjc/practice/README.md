# practice — δ-consumption runs (2026-08-12)

**Up**: [../README.md](../README.md) (mjc)
**Idea docs**: [performance_error_is_the_bridge](../../../ideas/performance_error_is_the_bridge.md) ·
[practice_manufactures_its_own_credit](../../../ideas/practice_manufactures_its_own_credit.md)
**Status**: record of what was run. **No interpretation** — outcome comparisons are deliberately
left to the JSON and the analyzers. Read this to know what exists, how it was configured, and how
to regenerate it.

## Scope

Four runs from one session, all consuming the bridge signal
δ = (b(s)−e)·σ((g−g₀)/θ) as a per-sample gain on forward-model plasticity. Three live under this
node; the fourth lives under `bridge_assembly/` because it is that node's runner with swept knobs.

| run | location | what varies | seeds |
|---|---|---|---|
| difficulty sweep | [`../bridge_assembly/difficulty_sweep/`](../bridge_assembly/difficulty_sweep/) | recovery difficulty, b's clock, the frontier's clock, the aleatoric decoy | 31 runs analysed |
| estimability | [`estimability/`](estimability/) | temporal concentration of practice at fixed per-context sample count | 3 + 3 + 2 + 1 |
| priced plasticity | [`priced_plasticity/`](priced_plasticity/) | FM capacity, replay, total plasticity spend | 1 (+ interior grid) |
| aleatoric flip | [`aleatoric_flip/`](aleatoric_flip/) | flip amplitude on a mastered region | 1 (+ calibration) |

A fifth run, added 2026-08-12, is **not** a δ-consumption run and does not share the substrate
below — it is the first node in this arc with *sequence* structure:

| run | location | what varies | seeds |
|---|---|---|---|
| étude arc | [`etude/`](etude/) | compilation of mastered segments into committed ballistic units: the δ-silence gate, the compile op itself, and sequential assembly | **written up** — sequential assembly with seam-matched selection dominates never-compiling on both axes, 3/3 seeds |

## Shared substrate

All four run on the puck-free corridor world (`../pusher_env.py`, damping 2.0 — the `ballistic`
4c regime), with localized command-rotation regions and optional per-region aleatoric noise. All
fork `../bridge_assembly/bridge_assembly.py`, which carries the corrected δ: context-conditional
`b(s)` as an error-predictor net, centered gate σ((g−g₀)/θ) calibrated by `../agency_gate/`'s
train-split protocol, gain consumption `w = exp(−δ/τ)`. `bridge_assembly.py` itself is unmodified —
every prior result stays reproducible.

## Properties of the setup, measured this session

These are mechanical facts about the apparatus, not findings about δ. Any number from this arc has
to be read against them.

- **Adam is invariant to global loss rescaling.** A `fixed` arm at constant per-sample weight 3 is
  bit-identical to one at weight 1 over 16,384 transitions. Per-sample *relative* weights still
  act; only total spend cancels. Consequence: the arc's running-mean weight normalizer never
  redistributed anything, and a gain must be moved onto the learning rate to change total spend.
  Any "matched budget" claim resting on a weight normalizer under Adam does not bind.
  (`priced_plasticity/`)
- **The gain law as calibrated is a binary gate, not a graded gain.** τ is set from pretrain MAD
  (≈0.0038) while online errors run 0.02–0.15, so |b−e|/τ ≈ 4–90 and ~29% of samples sit at the
  clip. Every δ result in this arc is `sign(b−e)` applied as a two-level multiplier. A
  corrected-τ arm (`:tauon`) exists in `aleatoric_flip/`. (`priced_plasticity/`, and observed but
  unnamed in `difficulty_sweep/` and `estimability/`)
- **δ's budget match is an EWMA and takes ~2,000 transitions to converge.** Over the first 256
  transitions the delta arm runs at 1.22–1.49× the uniform arm's average weight. AUCs including
  the m=256/512 milestones are not budget-matched. (`difficulty_sweep/`)
- **Pareto hulls are sensitive to grid coarseness.** In `priced_plasticity/`, adding three
  interior uniform points changed a scored gap for one arm from +14.9% to −26.2%, because the
  original hull segment had a near-vertical endpoint. Frontier-relative numbers should state
  which grid and which aggregation define the hull.
- **Arm-difference seed noise on this substrate** (estimated from `../bridge_assembly/`'s
  `asm_s0/s1/s2`, using differences *within* seed since arms share a stream and pretrained FM):
  delta−fixed adaptation +0.00518 ± 0.00083, delta−fixed retention +0.00144 ± 0.00085,
  delta−raw retention −0.00382 ± 0.00112, all 3/3 in sign; allocation-share sd 0.045. Absolute
  probe variance is several times larger and is the wrong scale for arm comparisons.

## The runs

### difficulty sweep — [`../bridge_assembly/difficulty_sweep/`](../bridge_assembly/difficulty_sweep/)

The parent runner with defaults untouched except the swept knob. B-mastered (0,−0.30, φ=−1.2,
pre-drift, oversampled in pretraining) and the R-noise decoy (0,+0.90, amp 30) are identical in
every cell; only the drift side of the y=+0.30 eval corridor changes.

| axis | knob | cells |
|---|---|---|
| recovery difficulty | `--regions` | `phi06/12/20/28` (one region, φ=0.6…2.8); `two12`, `two24` (two opposed rotations at ∓0.25) |
| b(s)'s clock | `--bench-lr` | 1e-2 / 3e-3 / 1e-3 at φ=1.2 and at two12 |
| the frontier's clock | `--adapt-lr` 9e-4 | `phi12_alr9e4`, `two12_alr9e4` (same ratio r = adapt_lr/bench_lr as the `blr1e3` cells, both clocks 3× faster) |
| the decoy | amp 30 → 0 | `two12_nonoise` |

31 runs analysed: 28 new plus `bridge_assembly`'s own `asm_s{0,1,2}` reused as the φ=1.2 cell
(identical config). 3 seeds on `phi06`, `phi12`, `phi20`, `two12`, `two24`, `two12_blr1e3`,
`two12_nonoise`, `phi12_alr9e4`, `two12_alr9e4`; 1 seed on `phi28`, `phi12_blr1e3`,
`phi12_blr1e2`, `two12_blr1e2`.

**Instrument checks.** Stale→ceiling range on `drift/ballistic_cem` spans 0.09 (phi06) to 0.77
(two24). Two saturation points bound the usable axis: at φ=0.6 the `agg/reactive` readout has
zero range (stale 0.0110 = ceiling 0.0110), and at φ=2.8 the *ceiling* itself degrades to 0.273,
so the single-region φ knob cannot reach the two-region range.

**Defined readouts.** Λ = ⟨(b−e)/(e_stale − e_floor)⟩ on the drift class (budget-averaged, the
`lam_by_T[-1]` field); `w_A`, `w_R`, `w_B` = per-class cumulative weight share ÷ sample share;
`f_boost` = fraction of batches with δ<0 on the drift class; outcome = `drift/ballistic_cem`
goal-distance over milestones m ≥ 2048 as a percentage of that seed's stale→ceiling range.

Numbers: `data/summary.json`. Figures `fig1`–`fig5`.

### estimability — [`estimability/`](estimability/)

Stream generator replaced; everything else inherited. **5 contexts** as localized rotation /
aleatoric regions: `A-rot+` (φ=+1.2), `B-rot−` (φ=−1.2), `C-rot~` (φ=+0.6), `N-noise` (amp 30),
`M-mast` (φ=−0.9, present pre-drift). A `base` class never in the stream is probed as a canary.

The stream is **1000 context-pure batches of 16, exactly 200 per context**, collected once. A
schedule is a permutation of that batch sequence with burst length ∈ {200, 50, 20, 5, 2, 1}
batches: burst 200 = fully massed, burst 1 = fully interleaved. Absence at re-entry =
(K−1)·burst ∈ {800, 200, 80, 20, 8, 4} cycles. **Transitions are identical across schedules** —
same data, same batch composition, same per-context count. Only order varies.

Arms: `fixed` (w=1), `raw_err` (w ∝ e), `delta_net` (b = error-predictor net), `delta_ewma`
(b = per-context tabular scalar EWMA), `oracle_delta` (b = the measured truth).

**The panel** is not an arm: it is a set of shadow benchmark estimators attached to the `fixed`
arm, all reading the same (s, e) pairs in the same order off the same FM trajectory, each
computing the δ and weight it *would* have applied without applying it. Nothing in it touches the
FM, so no estimator can influence the error stream any other estimator sees, and the estimation
problem is identical across estimators. Contents: the net form at 4–5 learning rates; a
per-context tabular EWMA at 3–4 α; the global scalar EWMA; a frozen pretrained `b`. The tabular
estimators are **given the context label**, which the net must infer from `s`.

**The oracle is measured, not fitted**: `E[e | pos]` under the *current* FM, from a grid of 100
positions per class × 16 repeats, **recomputed with each arm's own FM every 4 cycles** so the
ground truth moves with competence.

**Instrument checks** (3/3 seeds): stale `E[e|pos]` = A 0.118 / B 0.126 / C 0.064 (adaptation
demanded), N 0.155 (irreducible), M 0.0025 and base 0.0036 (mastered / clean). Gate train AUROC
0.993, τ = 7.6e-4.

Cells: main ladder 3 seeds (`est_s0/s1/s2`, six bursts); replay control 3 seeds
(`est_replay_s*`, bursts 200/20/1, `replay_mode=uniform`); benchmark-timescale panel 2 seeds
(`est_panel_s0/s1`); grid-edge panel 1 seed (`est_panel_edges_s0`, `net` to 3e-5, `ewma_ctx` to
α=1).

Numbers: `results/<tag>/burst<k>.json`. Figures `fig1`–`fig6` under `figures/est_s0/`.

### priced plasticity — [`priced_plasticity/`](priced_plasticity/)

`bridge_assembly`'s two-corridor geometry (B-mastered on the eval path, A-drift demanding
adaptation, R-noise decoy off-path) with three knobs changed so that plasticity carries a cost,
and the matched-budget comparison replaced by a swept one.

- `fm_hidden` 256 → **32**, `fm_layers` 3 → 2 (`meta_adapt` #4d/#4e put the capacity-competition
  boundary at h=32, gone by h=64)
- `n_replay` 4 → **0**
- weights factored into **relative allocation × spend**, with spend carried on the learning rate
  (forced by the Adam property above)

**Grading instrument**: uniform learning rate is swept and the retention × adaptation Pareto
frontier it traces is the comparison object, rather than a single matched-budget point. An 8-point
grid (`pp_s0` 5 points + `pp_int_s0` 3 interior) defines the hull; the aggregation is stated in
the analyzer (`--agg {half,q3,last4,final}`, `half` = t ≥ T/2 primary).

**Instrument check**: at the parent configuration the uniform family is **two points wide** (past
3e-4 more lr is worse on both axes); priced, it spans 7–8 non-dominated points. The three
parent-config arms reproduce `asm_s0` **bit-exactly** (probes, τ, budgets), so the fork is a
strict superset.

Arms include `fixed` at five lrs, `delta`, `delta:unit` (allocation only, spend divided out —
eff_lr matched to `fixed@3e-4` at 3.000e-04), `delta:tauon`, `raw_err`, and a `:raw_adam:ewma:r4:h256:L3`
stack reproducing the parent configuration.

Seeds: 1 (`pp_s0` + `pp_int_s0` + `cal_s0`). Seeds 1–2 were launched and **deliberately stopped
mid-run** (Jasper's call); their Modal apps were stopped and no partial results entered any
analysis. `analyze_priced.py --seeds` exists and is tested if they are ever run.

Numbers: `data/pp_s0.json`, `data/pp_int_s0.json`, `data/cal_s0.json`, `data/summary.json`.

### aleatoric flip — [`aleatoric_flip/`](aleatoric_flip/)

A region the FM has **already mastered** has aleatoric noise switched on at t=0 **with its
rotation φ unchanged**. `pusher_env.py`'s per-region `noise` is a zero-mean stochastic force
(`amp · N(0,1)` per substep), so the region's conditional mean is unchanged while its samples
become noisy draws. Probe `B-flip@clean` measures FM error against the **noise-free** dynamics,
isolating model state from irreducible noise.

Cells: flip amplitude f ∈ {0, 3, 10, 30} at `h32x2`. f=0 is the no-flip control with everything
else identical.

**Instrument checks.** Conditional-mean preservation is measured, not assumed: at f=3 the
B-flip bias is 0.00144 (se 0.00155, i.e. under 1 se from zero) against a deterministic signal
magnitude of 0.170; pooled bias 6.1e-5 ± 8.4e-5. At f=30 the same ratios hold (bias/se 0.92).
Degradation of `B-flip@clean` under uniform plasticity increases with both flip amplitude and
learning rate, so the cell carries a dose axis.

Arms (24 in the main run): `fixed` at five lrs; `omask` — an **oracle mask** given the true
flipped-region identity, at mask weights m ∈ {0, 0.5}, bounding what exact region knowledge buys;
`delta`, `delta:unit`, `delta:tauon`; `raw_err`; `disag` (ensemble disagreement) in shared/boot/expd
variants; `conj` (δ × disagreement). f0 duplicates of the key arms.

Seeds: 1 (`af_s0`), plus the calibration sweep (`afcal_s0`). No README was written by the
implementing agent; `analyze_flip.py` regenerates all readouts from `data/af_s0.json`.

### étude arc — [`etude/`](etude/README.md) (2026-08-12 → 08-14, written up)

**Goal**: instantiate the practice loop's third component — re-chunking/compilation — on a task
with *sequence* structure (a fixed 4-segment piece, one hard passage), where the difficulty axis is
composition rather than per-step severity, and δ is consumed only as a **detector** (this audit's
surviving role for it). Eleven runs: calibrations, the E-gate compile-trigger test, two
compile-op discriminators, selection (E-3/E-3b), sequential assembly (E-4), consolidation +
3-seed replication (E-5).

**Headline**: sequential assembly with seam-matched selection **dominates never-compiling on both
axes in 3/3 seeds** (piece error 0.0772 ± 0.0097 vs 0.1066 ± 0.0059 at 34–53% less priced time).
En route: compilation is *selection + commitment*, not distillation-by-regression (averaging valid
renditions is what destroys them); selection must rank by expected performance under the
consumption distribution (winner's curse and the 4.5 σ practice→performance seam-state shift are
the two measured ways to get this wrong); committed units never degrade here and fusion is vacuous —
hierarchy needs boundaries that carry information, which sets the next round's design. Full
findings, retractions and per-run tables: [`etude/README.md`](etude/README.md).

## Reproduce

```bash
cd experiments/

# difficulty sweep — cell table, then detached launches
python3 mjc/bridge_assembly/difficulty_sweep/sweep.py --list
python3 mjc/bridge_assembly/difficulty_sweep/sweep.py --cells two12,two24 --seed 0
python3 mjc/bridge_assembly/difficulty_sweep/analyze_sweep.py --seeds 0,1,2 --figures \
    --json mjc/bridge_assembly/difficulty_sweep/data/summary.json

# estimability
modal run mjc/practice/estimability/estimability.py::estimability --quick
python3 mjc/practice/estimability/launch_detached.py --tag est_s0 --seed 0
python3 mjc/practice/estimability/analyze_estimability.py --tags est_s0,est_s1,est_s2

# priced plasticity
python3 mjc/practice/priced_plasticity/launch_detached.py --mode calibrate --tag cal_s0
python3 mjc/practice/priced_plasticity/launch_detached.py --mode main --tag pp_s0 --seed 0
python3 mjc/practice/priced_plasticity/analyze_priced.py --agg half

# aleatoric flip
python3 mjc/practice/aleatoric_flip/launch_detached.py --mode calibrate --tag afcal_s0
python3 mjc/practice/aleatoric_flip/launch_detached.py --mode main --tag af_s0 --seed 0
python3 mjc/practice/aleatoric_flip/analyze_flip.py
```

Modal volume (`mujoco-control-data`): `/data/bridge_assembly/dsw_<cell>_s<seed>/`,
`/data/practice_estimability/<tag>/burst<k>/`, `/data/priced_plasticity/<tag>/`,
`/data/aleatoric_flip/<tag>/`.

## Per-node file indexes

[`etude/FILES.md`](etude/FILES.md) ·
[`estimability/FILES.md`](estimability/FILES.md) ·
[`priced_plasticity/FILES.md`](priced_plasticity/FILES.md) ·
[`aleatoric_flip/FILES.md`](aleatoric_flip/FILES.md) ·
[`../bridge_assembly/difficulty_sweep/FILES.md`](../bridge_assembly/difficulty_sweep/FILES.md)
