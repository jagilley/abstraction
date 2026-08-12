# Plasticity gain: δ as a per-sample gain on FM learning — the Gadagkar/Kim consumption of the bridge signal

**Up**: [../README.md](../README.md) (mjc) · **Idea doc**: [performance_error_is_the_bridge](../../../ideas/performance_error_is_the_bridge.md) §1's efferent arm ("the return is a gain, not an addend"). **Replaces §8 Experiment 1**, which [`../drift_value_loop/teacher_snr/`](../drift_value_loop/teacher_snr/README.md) found was a no-op as specified: the songbird consumes its performance error as a *per-trial plasticity gain* (Gadagkar 2016; Kim, Parvin & Ivry 2019: "task outcome as a gain on implicit adaptation"), never as a reward for a low-sample meta-search. This node runs that consumption — never previously run in this repo.
**Substrate**: [`../ballistic/`](../ballistic/README.md) (4b/4c — the regime where FM quality transmits to behavior).
**Code**: `plasticity_gain.py` (runner), `analyze_gain.py` (aggregation) · File index: [FILES.md](FILES.md)
**Status**: done, 3 seeds + benchmark-timescale sweep (1 seed). **Date**: 2026-08-11.

## One-liner

δ-gated plasticity **beats ungated fixed-lr on ballistic re-adaptation speed at matched average learning
rate, 3/3 seeds** (~9% of the stale→ceiling range at mid-recovery), ballistic-specifically, with converging
endpoints — **speed, not ceiling**. But a raw-error gain matches δ on control: **δ's benchmark-specific
content shows in allocation and retention** (withdraws plasticity from unimprovable noise where raw error
fixates ~19% of its budget forever). And the **benchmark must be context-conditional b(s)** — the idea
doc's literal scalar-EWMA benchmark is mechanistically pathological (a global EWMA sits below a noise
region's error forever, so it fixates on noise *worse than raw error*, and growing).

## Method

Puck-free corridor reach at damping 2.0 (the 4c ballistic-competent regime). Type-2 drift at t=0: two
**on-reach command rotations** (φ=±1.2, σ=0.18 — local, open-loop-compensable; stretches recovery into a
graded trajectory, fixing 4c's step-like caveat) plus one **off-reach aleatoric noise region** (amp 30 —
the noisy-TV distractor, ~5.8% of uniform collection). One shared reward-free stream (16,384 transitions,
identical across arms); four arms differ **only** in per-sample plasticity weight at **matched average
weight** (running-mean normalized; verified fixed 1.000 / delta 1.000; raw 0.927 / delta_scalar 0.920 —
the clip truncates their heavy tails, a *conservative* handicap):

- `fixed` — w=1 (the ungated control)
- `raw_err` — w ∝ e (any-error-modulated control)
- `delta` — w = f(δ), δ = (b(s)−e)·σ(g/θ), **b(s) a context-conditional error-predictor net** (the bird's
  per-syllable benchmark)
- `delta_scalar` — the idea doc's literal §1 scalar-EWMA benchmark

f(δ) = exp(−δ/τ), capped — direction per Kim et al. (worse-than-benchmark boosts, at-benchmark f(0)=1,
better attenuates; the cap = their categorical effect). Graded at 8 milestones under reactive +
ballistic-CEM; dense FM probes by region class.

## Results (3 seeds, mean ± sd; stale ballistic ~0.66, fresh-FM ceiling ~0.08)

| arm | ballistic recovery AUC ↓ | noise over-weighting | retention (base-probe err; pre-drift 0.0016) |
|---|---|---|---|
| fixed | 0.549 ± 0.018 | 1× | **0.0097** |
| raw_err | **0.520 ± 0.021** | 3.24 ± 0.01 | 0.0175 (worst) |
| delta | 0.531 ± 0.019 | **1.73 ± 0.06** | 0.0132 |
| delta_scalar | (best mid-recovery: 0.458 @8192) | 3.69 ± 0.04, **growing** | 0.0154 |

1. **The gain does behavioral work vs ungated** (fixed−delta positive in 3/3 seeds; effect
   ballistic-specific — reactive spread ~4× smaller, the 4b transmission asymmetry reproducing; endpoints
   converge ~0.09–0.11 → the gain governs the transient only).
2. **On control alone, "any error-modulated lr" explains the win** (raw_err ≥ delta all seeds). δ's
   specific content is *where plasticity goes*: raw spends ~19% of its entire budget on unimprovable noise
   permanently and has the worst retention. The noise decoy is deliberately off-path here, so that cost
   never reaches the control score — **an environment where retained/base competence is behaviorally
   load-bearing is the natural next test** for δ-vs-raw to cash out in control.
3. **b(context) is load-bearing** — delta_scalar's noise weight grows 3.0→3.9 while reducible-region weight
   falls to ~1 (permanent δ<0 on noise). Amusingly it is often *best* on mid-recovery control:
   pre-habituation max-gain-everywhere-that-changed is a fine speed policy in the transient; its pathology
   is the permanent noise spend.
4. **Benchmark-timescale sensitivity: modest, direction-consistent, no cliff** (bench_lr 1e-3/3e-3/1e-2:
   AUC 0.509/0.511/0.527; late noise weight 1.52→1.20, seed 0). The main runs sit on the *too-slow* side
   (b_noise 0.119 vs e_noise 0.162 at end), if anything **understating** the achievable noise
   discrimination — consistent with [`benchmark_vs_cost/`](../curiosity_control/benchmark_vs_cost/README.md)'s
   interior-optimum finding.
5. **The agency gate was inert on all-self-generated data** (σ(g/θ) p10–p90 = 0.61–0.80), as expected by
   design — its live test is [`../agency_gate/`](../agency_gate/README.md).

## Caveats

Single task family, one drift shape, noise decoy off-path by design. Behavioral deltas are ~5–10% of the
recovery range; the allocation/retention mechanism metrics are the tight, seed-stable readouts (this
node's convention). `gain_s0` (low-throughput regime) is a valid extra data point: same mechanism
orderings, recovery only ~15% spanned.

## Gotcha (transferable)

A Modal input-cancellation can silently swallow a post-cancellation `volume.commit()` (the save line
prints, nothing lands) — traced to a harness TaskStop reaching the detached launcher's process group.
Fix: **incremental checkpoint commits after every arm/milestone** + a fork/setsid-isolated launcher. The
rerun reproduced the lost run bit-for-bit (same-seed determinism held exactly).

## Figures & reproduce

Per run under `figures/plasticity_gain_<tag>/`: `fig1_recovery.png` (control recovery per arm),
`fig2_fm_traces.png` (probe error by region class), `fig3_allocation.png` (where plasticity was spent),
`fig4_delta_mech.png` (δ + benchmark-habituation traces). Volume: `/data/plasticity_gain/<tag>/`
(tags `gain2_s{0,1,2}`, `gain2_blr{3e3,1e2}_s0`, `gain_s0`, `rangecheck`, `smoke`). Exact configs in each
run's `results.json`.

```bash
cd experiments/
modal run mjc/plasticity_gain/plasticity_gain.py::plasticity_gain --quick        # smoke
for s in 0 1 2; do
  modal run --detach mjc/plasticity_gain/plasticity_gain.py::plasticity_gain \
      --tag gain2_s$s --seed $s --total-t 16384 --milestones "0,256,512,1024,2048,4096,8192,16384"
done
python3 mjc/plasticity_gain/analyze_gain.py
```
