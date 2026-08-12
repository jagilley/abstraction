# Two clocks: performance error and reward are separate channels — each legible only through its own matched window

**Up**: [../README.md](../README.md) (mjc) · **Idea doc**: [performance_error_is_the_bridge](../../../ideas/performance_error_is_the_bridge.md) §8 **Experiment 4** — carrying §6's load-bearing bet (performance error as a *separate channel*, not a modulation of reward learning) and §7's two-clock flag ("the part nobody has done").
**Design constraint inherited from** [`../drift_value_loop/teacher_snr/`](../drift_value_loop/teacher_snr/README.md): grade with dense signal/learning readouts, never a low-sample meta-search. δ machinery reused from [`../plasticity_gain/`](../plasticity_gain/README.md) (canonical corrected form: context-conditional b(s), centered gate).
**Code**: `two_clocks.py` (runner), `analyze_two_clocks.py` (aggregation) · File index: [FILES.md](FILES.md)
**Status**: done — architecture test 3 seeds, window×delay sweep 1 seed. **Date**: 2026-08-11.

## One-liner

**Two channels beat one at matched capacity, 3/3 seeds — and sharing is destructive interference, not a
compromise.** A fast per-transition performance-error signal and a slow trial-level reward signal carry
**non-redundant content** (per-sample *precision* vs *value-relevance*); each is legible only through an
eligibility window matched to its own delay; every shared-window variant is worse than *both* single
channels, and the mid/slow/broad shared windows are worse than ungated fixed-lr on the value region in all
seeds. Separately, the Suvrathan prediction lands: credit fidelity peaks at each circuit's *own* delay and
the peak **moves with the delay**.

## Method (the design discipline)

plasticity_gain's ballistic corridor world; Type-2 drift at t=0 (R1: rotation *on* the reach path; R2:
rotation off-path; R3: aleatoric noise). Two teaching signals, both **anonymous scalars at arrival** —
credit reaches samples only through **unit-area** eligibility kernels peaked at age τ (unit area = a
window can *place* credit but not create it; the capacity convention):

- **Fast**: canonical δ per transition (b(s) net, centered gate), computed from the arm's own FM
  as-of-prediction (Smith-held), broadcast d_fm=4 steps late.
- **Slow**: trial-level reward advantage vs a habituating per-family benchmark, from scripted ballistic
  reaches planned with the *pre-drift* FM (frozen dysmetria → arm-independent, so every arm consumes the
  **identical** stream), arriving d_r=8 after trial end (sample ages 8–39 at arrival; matched τ_slow=24).

Consumption: per-sample plasticity gain `w = exp(−Σ ledger credit/κ̂)`, running-mean normalized and
clipped (matched budget). **The only difference between `two_channel` and `shared` is where the sum
happens relative to the windowing** — shared sums both signals into one kernel; two_channel gives each its
own matched kernel and sums in log-weight space. Same signals, same budget, same FM, same data, same
combination rule.

## Prediction (ii) — the separate-channel bet: supported, 3/3 seeds, stronger than predicted

Value-relevant FM recovery (R1 probe AUC ↓, mean ± sd over seeds 0–2):

| arm | R1 AUC | R1 over-wt | noise over-wt | credit fidelity vs δ / vs reward |
|---|---|---|---|---|
| fixed | 0.0895 ± 0.0045 | 1.00 | 1.00 | — |
| fast_only (τ4:d4) | 0.0843 ± 0.0049 | 1.21 | 1.55 | 0.73 / 0.01 |
| slow_only (τ24) | 0.0842 ± 0.0032 | 1.80 | 0.69 | −0.12 / 0.91 |
| **two_channel** | **0.0828 ± 0.0045** | 1.60 | 1.25 | **0.74 / 0.91** |
| shared:τ4 | 0.0871 ± 0.0046 | 1.19 | 1.49 | 0.58 / 0.01 |
| shared:τ10 | 0.0919 ± 0.0045 | 0.88 | 1.18 | 0.23 / 0.22 |
| shared:τ24 | 0.0932 ± 0.0035 | 1.15 | 0.88 | 0.06 / 0.44 |
| shared:broad | 0.0929 ± 0.0047 | 0.92 | 1.12 | 0.24 / 0.36 |

two_channel beats every other arm **in each seed individually**, keeps both signals' content
simultaneously (fidelity ≈ each specialist's own number), and inherits the designed complementarity:
slow's value tilt with fast's precision, less noise-fixation than fast alone. **Sharing destroys both**:
reward through a δ-matched window lands on post-trial explore samples (fidelity ≈ 0.01); δ through a wide
window becomes a spatially-decorrelated smear. Conservative detail: clip truncation handicaps the winners'
realized mean weight (0.83–0.88 vs shared 0.86–0.98) and two_channel still wins.

**The content split, stated plainly**: δ knows *when/where* (dense, per-transition, precisely
time-stamped; value-blind by construction — a self-supervised error carries no task information). Reward
knows *what for* (task-computed, so its smeared credit lands only on task-engaged behavior; per-transition
precision unrecoverable from a delayed trial-level scalar). Credit assignment factorizes into
(which experiences) × (toward what end), and the factors have different temporal supports — which is
*why* the separation must be architectural.

## Prediction (i) — Suvrathan matching: supported

Sweep (seed 0): the reducible-region probe AUC has an interior minimum **exactly at τ = d_fm** for both
delays (d=4: 0.0901 at τ=4 vs 0.094–0.100 elsewhere; d=12: 0.0918 at τ=12 vs 0.094–0.099), and the dense
credit-fidelity curve peaks at each circuit's own delay and **moves with it** (0.73 at τ=4|d=4; 0.56 at
τ=12|d=12; ≤0.12 off-diagonal). Figure: `figures/analysis/fig_diagonal_sweep_s0.png`.

## Surprises

- **slow_only is much stronger on FM recovery than expected**: trial-membership credit × the loss's own
  filtering (weights on already-correct samples produce no gradient) acts as a **value-relevance mask**
  (R1 over-weight 1.80). Its blind spots: per-transition precision (fidelity −0.12) and off-trial
  structure (R2 over-weight 0.69).
- The reward channel **self-anneals via benchmark habituation** (advantage −0.28 → −0.03 over the run)
  without behavior ever improving — the scripted-trial design working as intended.

## Honest reading of the behavioral channel

The ballistic-median instrument has dynamic range only when the eval draw forces reaches through R1. In
seed 1 it does (stale 0.590 vs ceiling 0.086) and the ordering comes out exactly as the mechanism
predicts: **two 0.276 < slow 0.301 < fast 0.310 < shared:τ4 0.326 < fixed 0.361 < other shared
0.366–0.392** (`results/main_s1/fig1_recovery.png`). In seeds 0/2 the median dodges R1 and the readout is
noise — including an inverted mid-recovery artifact in the sweep (matched windows spend precise credit
during the pre-habituation transient while base churn dominates the median). Mechanism metrics carry the
claim, per this arc's convention. Two references are *references, not ceilings*: a fresh FM on uniform
drifted-world data under-samples R1 (off-policy) and can grade worse than stale.

## Caveats

Single task family; scripted (frozen) trial behavior — reward annealing comes from benchmark habituation,
not closed-loop improvement; one slow clock (d_r fixed); unit-area kernels are *the* capacity convention
(unit-peak would gift broad windows); sweep's behavioral column is 1 seed and range-limited.

## Figures & reproduce

Per run under `results/<tag>/`: `fig1_recovery.png`, `fig2_fm_traces.png`, `fig3_allocation.png`,
`fig4_signal.png`; the diagonal at `figures/analysis/fig_diagonal_sweep_s0.png`. Volume:
`/data/two_clocks/{main_s0,main_s1,main_s2,sweep_s0,smoke}/` (results.json with full config, traces.npz).

```bash
cd experiments/
modal run mjc/two_clocks/two_clocks.py::two_clocks --quick                     # smoke
for s in 0 1 2; do
  modal run --detach mjc/two_clocks/two_clocks.py::two_clocks --tag main_s$s --seed $s
done
# window×delay sweep (exact arms string in sweep_s0/results.json):
modal run --detach mjc/two_clocks/two_clocks.py::two_clocks --tag sweep_s0 --seed 0 \
    --arms "fast:t2:d4,fast:t4:d4,fast:t10:d4,fast:t24:d4,fast:t4:d12,fast:t12:d12,..." --light-grading --a-max 64
python3 mjc/two_clocks/analyze_two_clocks.py
```
