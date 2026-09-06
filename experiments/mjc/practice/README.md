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

Two further nodes (2026-08-19 → 08-20) port the RHM practice arc back to this substrate — the
**fingering/legato arc**, the étude's own named next round run with everything the RHM detour
settled (state-conditioned commitment, the three-coordinate seam law, certificate demotion,
typed maintenance):

| run | location | what varies | seeds |
|---|---|---|---|
| fingering | [`fingering/`](fingering/README.md) | the compile-op taxonomy on a boundary that carries information: frozen (fixed/keyed/regressed/averaged) vs **live** content under committed routing, timing × maintenance, priced deliberation | **written up** — 1 seed/run; f1c's `never` bit-reproduces f1b's |
| legato | [`legato/`](legato/README.md) | committed **span** vs the model's composition horizon: segment vs phrase granularity × live vs measured content, with a fusion control | **written up** — 1 seed (seed pair cancelled, GPU budget); ranks + two-horizon signs + double-sided control |
| span | [`span/`](span/README.md) | the composition horizon **as a trajectory**: legato's practice loop with nothing committed, the FM snapshotted every cycle, imagination horizon × executed span of one live phrase plan against a flat `e_react`; plus a re-measurement of the saved models (on/off-corridor accuracy, the exploitation gap, a per-snapshot CAL-P sweep) | **written up** — 1 seed; 17–81-point monotone trends are the claims |
| offbook | [`offbook/`](offbook/README.md) | **the port back** on this substrate: seam-time audition (enumeration) vs π-routing vs both ports, the address-book battery and poison twin — then three canary rounds (credit currency, exposure placement, reflex-loop delay) on why the chain level is never adopted | **written up; SUSPECT** (2026-08-27, see `accompanist/`) — 1 seed/run, bit-identity twins + cross-tag exact controls |
| accompanist | [`accompanist/`](accompanist/README.md) | the follow-up to offbook: library construction (`d1`–`d3`: content ladder, legato's nesting, the model adapting) and the delay operator (`d3b`, `presto/`: naive vs efference copy) on offbook's piece and on a fast 120 ms-segment piece | **written up** — the pivot: the incumbent's free forward model, not the piece, is what refuses the deep unit |
| acappella | [`acappella/`](acappella/README.md) | the re-port with **no forward model anywhere**: a priced real-rollout incumbent under a declared budget on `etude/`'s substrate, then (amendment) the feedback/delay axis — reflex vs keyed/auditioned library under an unbridgeable observation delay | **written up** — the grounding economy halts at its own pre-fixed gate (0/15 search cells beat the reflex); re-sited on feedback, a model-free **segment-span niche opens at 192 ms** |
| solo | [`solo/`](solo/README.md) | **re-internalization with no forward model** at acappella's niche (Δ = 8): a behaviour-cloned reflex as the trunk, offbook's Port 1 (π over member slots) and Port 2 (span head, parity measured on the plant), the address-book battery, the primitive legal at every seam, δ_perf against the tape as intention reference logged; then a probe replacing the relative imitation filter with two pre-fixed absolute bands | **written up** — routing, corridor and address book all port at segment span; adoption tracks the meter (feel at Δ = 0, memory at Δ = 8); learning buys a 4× grounding cut and a 10× tighter quarantine, **not error** — the construction prior is the best selector and the bands starve π |
| tempo | [`tempo/`](tempo/README.md) | **levels as execution span on the plant** — a four-rung ladder under delay on the donor body (`prestissimo/`), then a fast body (τ = 30 ms) at a fixed human delay with **tempo** as the era knob, a force-level program and the crank (`accelerando/`), then the **factored program**: a path on phase through an executor that knows the body — exact, learned from slow practice, and corrected online against its forecast (`rubato/`) | **written up** (super-writeup over three implementer nodes) — tempo alone opens the ladder on a fast body (10× over the re-fit reflex at 48 ms notes on 1 read vs 16; the segment rung's first niche); a chunk stored as forces transfers to no tempo, a chunk stored as a path transfers three octaves through a body model (in band L2–4 at 2×, L3–4 at 4×, L4 at 8×; 0.30–0.55× the reflex on one read); a body model learned from two slow tempi keeps 7 of 9 cells; correction against its forecast lifts the 5% command-accuracy bar to 20% at one launch read; no executor reaches the verbatim tape at the deep rung |

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

### fingering — state-conditioned commitment and the completed op taxonomy (2026-08-19 → 08-20, written up)

**Goal**: run the étude's named next round — commitment on boundaries that carry information — on
the n=3 arm (arrival-*posture* spread on the redundancy manifold), with the RHM arc's settled
machinery: E-3b selection, provisional commitment, δ as detector, on-policy collection (tier C),
priced deliberation.

**Headline**: the RHM laws reproduce where they were born (state-conditioned commitment 1.34×,
capturing 41% of a 3.27× per-state oracle; averaging damage monotone in **pool coherence**;
maintenance and refresh strictly separable, with **competence-news** the currency only a scheduled
re-audit consumes) — and then the taxonomy's completion inverts the frame: **live content under
committed routing (plan-at-launch) dominates every frozen op 1.6× at 2.3× less priced time and
self-maintains its own diet**. Within the model's reach, commit the *routing*, not the content.
`never` wins outright at every feedback and deliberation price (§6 rarity, on a priced axis); the
certificate fires at c10 against measured mastery c51 in every arm of every run. Full tables:
[`fingering/README.md`](fingering/README.md).

### legato — the composition-horizon crossover (2026-08-20, written up)

**Goal**: extend the taxonomy past the model's reach — a 4-leg loop whose phrase (60 steps) is ~3×
the measured composition horizon (~21) — and test whether frozen, measured content re-enters
there, with a fusion control and a seam-cost calibration chosen by a pre-fixed neutral criterion.

**Headline**: **the crossover** — live content wins inside the horizon (1.25×), frozen measured
chains win beyond it (1.83×), and fusing is free for measured content (−0.004) while catastrophic
for live plans (+0.102, 2.2×): *a measured chain was produced by the body, so it carries no
composition error*. Chunks extend committed execution past the model's reach rather than beating
planning inside it — §2(b) measured. Plus: a seam cannot be pre-handled (launch keying near-inert
in audition, lookahead monotonically harmful) yet the closed-loop keying gain is 1.34× because
**the segment-level library manufactures the phrase pool's diversity** — `recital`'s level-(k+1)
law on a physical plant. Single seed (seed pair cancelled — GPU budget); `never` still dominates
raw error; steady-state per-traversal economics favor the fused chain on both axes. Full tables
and the F1–F5 findings record: [`legato/README.md`](legato/README.md).

### span — the composition horizon as a trajectory (2026-08-20, written up)

**Goal**: read Iwane et al. 2026 (hippocampal skill-memory expansion continues after the speed
plateau) against this substrate — does the FM's composition horizon keep growing with practice after
`e_react` has flattened (it is at plateau from cycle 0 on legato's piece), and does that convert
into longer committed execution?

**Headline**: **the representation expands through a flat task metric, and a live plan's reach does
not follow.** Imagination horizon along the consumed corridor roughly doubles at tight tolerance
(11 → 22 steps at 0.02 m, ρ=+0.77, p=4e-17) while `e_react` is flat (ρ=+0.24 n.s.); one open-loop
phrase plan still delivers exactly one waypoint at cycle 80, and its far end gets 5× worse. The
re-measurement says why: not planner starvation (16× search rescues the far end 2/8, p=0.74, and
reliably buys only a lower *believed* error, 7/8; believed vs true r=0.01 over 40 cells) but
sharpening-on / degrading-off — one-step FM error halves on the corridor and doubles on the plan's
trajectory (ratio 0.78 → 2.05, p=2.6e-4), and the model's promised far-seam arrival stays ~0.07 m
at every cycle while the truth walks to ~1 m. A live chunk is a model's promise off-distribution
and practice worsens it; the human chunk is measured content — legato F4 from the model's side, and
the seam law's state coordinate in miniature. Single seed. [`span/README.md`](span/README.md).

### offbook — the port back: routing consolidates, trust tracks exposure, the chain level is refused by the task's economics (2026-08-25 → 08-26, written up)

**Goal**: instantiate `rhm/practice/native/`'s consolidation on the motor substrate — the seam as
the decision point, the library as the action set, seam-time audition as the enumeration analogue,
π-routing (Port 1) and a parity-gated span head (Port 2) — and ask whether trust in a committed
motor vocabulary forms, across four canary-staged rounds.

**Headline**: routing consolidates — better error at 5.3× less deliberation, by *selection hygiene*
under a seam-audition scorer that is measured 5× optimistic about its own pick (the freed budget
buys nothing here; the CEM ladder is non-monotone). The chain level is never adopted, and the rounds
assign why at three layers: credit cannot substitute for use (a ×8 cheapness weight rises and
extinguishes exactly with forced exposure); use requires exposure the policy's own gate won't
provide (moving the launch-ε upstream at matched budget sustains trust at 25× the control); and
exposure converts to adoption only when the content is worth adopting — 203 body-graded plays say
these chains execute 2.8× worse than the median traversal, and an observation delay on the reflex
loop (degradation ordered exactly by feedback consumption, 63×/2.7×/1.7×/1.28×) opens no playable
niche for them. Depth on this piece is geometric, not economic — an environment property, with the
memory-only piece named as the fix. Quarantine of a poisoned address holds even under forced
exposure. Full record: [`offbook/README.md`](offbook/README.md).

**Status (2026-08-27): suspect.** The incumbent plans for free in a near-perfect forward model and
the ports were built from the same model; the "geometric, not economic" conclusion is retracted by
`accompanist/` on offbook's own criterion. The π-level findings (trust tracks exposure, quarantine)
stand.

### accompanist — the forward model is the incumbent's free accompanist (2026-08-26 → 08-27, written up)

**Goal**: find where offbook's setup diverged from the conditions practice work needs, and run the
two fixes the record suggested — build the library the way `legato/` did, and build a piece
playable only from memory.

**Headline**: offbook's negative was two things stacked. First, library construction: tapes
harvested from reactive traversals never launch from the states they are scored and played on;
with legato's nesting the pool matches legato's cell for cell, and with the model adapting the
pre-fixed delay gate **passes at 160 ms** on offbook's own piece (`d3`) — and on a fast
120 ms-segment piece the chain beats reactive from 40 ms inside playability (`presto/`). Second,
and decisive: those wins are against a delayed incumbent that cannot predict. Give it the forward
model through the delay (efference copy, a Δ-step rollout) and it recovers 6–11× of the penalty and
owns every playable delay on both pieces — every Δ inside the model's horizon is bridged by the
model. On RHM the incumbent's search was real and priced; on this substrate it has been free
imagination since `fingering/`, and `etude/` — the one node where chunks won — is the one where the
model was the bottleneck. Single seed throughout; bit-identity controls across rounds. Full record:
[`accompanist/README.md`](accompanist/README.md). Next: [`acappella/SPEC.md`](acappella/SPEC.md).

### acappella — the model-free port: the grounding halt, and the feedback niche (2026-08-27, written up)

**Goal**: run the RHM economy on the motor substrate with **no forward model anywhere** — a priced
real-rollout incumbent (CEM on a resettable plant copy) under a declared grounding budget on
`etude/`'s piece, a reflex law as the no-search reference, plant audition, legato's nesting.

**Headline**: the grounding economy **halts at its own pre-fixed gate** — the search reaches
étude's `never` band (0.0612 at G=4096, ladder-limited) but 0 of 15 cells beat the reflex law
(0.0045 at 16.9 s vs 0.0607 at 15,011 s priced): a motor trial costs performance-currency time,
and on a piece steerable by feel, search-by-real-trial is dominated by an option RHM structurally
lacks. Re-sited on the meter's native feedback axis (amendment, same day): under an honest,
unbridgeable observation delay, a **model-free segment-span niche opens at 192 ms** — stored
tapes at 4 fb beat a per-Δ re-tuned reflex 2.0× → 5.6× (to 384 ms), degradation ordered exactly
by feedback consumption (134× / 1.65–3.10× / 1.00×) — offbook finding 7d with the confound
removed. The chain question is structurally out of this piece's reach (chains decide at seam 0,
from rest — exactly delay-invariant, pre-flagged), and the audition is the most delay-fragile
selector in the system (calibration 1.00 → 1.77; the chain-capable arm selects chains least where
they pay most). B2 (trust at Δ=8) and the presto fast-piece escalation are queued with explicit
licensing conditions, not run. Single seed; bit-identity gates against `etude/` and across tags.
Full record: [`acappella/README.md`](acappella/README.md).

### solo — re-internalization with no forward model: the three signatures port at segment span; learning buys cost, not error (2026-09-04 → 05, written up)

**Goal**: run `rhm/practice/native/`'s consolidation on the plant with nothing imaginary in the
loop, at the delay where committed content pays — a behaviour-cloned reflex supplying the trunk
acappella said Port 2 lacked, π over member-library slots, a span head with parity measured as
execution reproduction on the plant, the address-book battery, the primitive legal at every seam,
and δ_perf against the committed tape's own trajectory logged as an instrument.

**Headline**: **routing, corridor and address book all port, model-free.** Routing cuts groundings
4× and priced time 3.8× at a 6% error cost against enumeration; deleting the table at zero
groundings *lowers* error in every trained arm while untrained heads collapse to 0.72; trust forms
(segment mass 0.73 → 0.98) and both ports quarantine the poison tenfold against enumeration; the
corridor head beats the tape on every parity-open slot; and π routes to the primitive 0.99 of the
time at Δ = 0 and 0.00 at Δ = 8 — the learner plays by feel where feel wins and from memory where
memory wins. What learning did **not** buy is error: the routed arms start at the construction-order
prior (0.0955), climb, and settle at 0.12 against the donor's 8-tape 0.0838, and a probe that
replaced the relative imitation filter with two pre-fixed absolute bands starved π (pass 0.05 and
0.01) and left the untrained arm at the prior — the best routed error in either run. The certifier
throughout is the body through the resettable plant: content by held-out replay, trust by
body-graded self-imitation, internalization by plant-measured parity; timing has none. Single
seed; fourteen-check bit-identity against `acappella/b1`, twins at 0.000e+00 over 120 cycles.
Full record: [`solo/README.md`](solo/README.md).

### tempo — levels as execution span on the plant: delay, tempo, and the factored program (2026-09-05 → 06, written up)

**Goal**: build the motor instance of the RHM level ladder under Jasper's mapping — level ℓ is a
unit executed open-loop over 2^(ℓ−1) segments from one feedback event, nested `T[ℓ] ⊆ T[ℓ−1]×T[ℓ−1]`
— and find what a level buys on a plant. Three implementer nodes under one super-writeup:
`tempo/prestissimo/`[^private] (the ladder under delay on the donor body),
`tempo/accelerando/`[^private] (a fast body, tempo as the era ladder, the
force-level program, the crank), `tempo/rubato/`[^private] (the kinematic program
through a body model: ceiling, learned, corrected).

**Headline**: the regime for chunking is a band of three timescales, body lag < note < delay, and on
the donor body it is empty at every tempo; on a body fast enough (τ = 30 ms) at a fixed 120 ms delay,
**tempo alone opens the ladder** — a stored whole-figure unit beats a per-tempo re-fit reflex 10× at
48 ms notes on one read against sixteen, and the segment rung has a niche of its own for the first
time. A chunk stored as **forces** is the arc's best executor exactly where it practiced and transfers
to no tempo at any step; a chunk stored as a **path**, played through an executor that knows the body,
transfers three octaves (in band L2–4 at 2×, L3–4 at 4×, L4 at 8×; 0.30 / 0.35 / 0.55× the reflex on
one read), a body model **learned from the learner's own slow practice at two tempi** keeps 7 of the
perfect model's 9 cells, and **correction against its forecast** lifts the 5% command-accuracy bar
every open-loop executor was bounded by to 20%, at one launch read per span at the source tempo.
Bounds: no executor reaches the verbatim tape at the deep rung; nothing plays the figure in band at
48 ms notes; on this plant the within-level judge is constant by construction and the next-level
gauge never plateaus, so the crank ran as a cap schedule. Every retraction en route (gain-grid floor,
delayed lead-in, true-vs-observed posture, resampler aliasing, drag identifiability) was produced by
a pre-registered check. Single seed throughout; 23 tags; each node gated bit-for-bit against its
donor's run of record. Full record: [`tempo/README.md`](tempo/README.md).

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
[`fingering/FILES.md`](fingering/FILES.md) ·
[`legato/FILES.md`](legato/FILES.md) ·
[`span/FILES.md`](span/FILES.md) ·
[`offbook/FILES.md`](offbook/FILES.md) ·
[`accompanist/FILES.md`](accompanist/FILES.md) ·
[`acappella/FILES.md`](acappella/FILES.md) ·
[`solo/FILES.md`](solo/FILES.md) ·
[`estimability/FILES.md`](estimability/FILES.md) ·
[`priced_plasticity/FILES.md`](priced_plasticity/FILES.md) ·
[`aleatoric_flip/FILES.md`](aleatoric_flip/FILES.md) ·
[`../bridge_assembly/difficulty_sweep/FILES.md`](../bridge_assembly/difficulty_sweep/FILES.md)

[^private]: Not mirrored: this link points to a document in the private lab repo (the roadmap, the queue, an unrun spec, reading notes, or a conversation). See the top-level README for what is held back and why.
