# Benchmark vs cost: the noisy-TV discrimination — no standing cost term, and δ is a gain signal, not an allocation score

**Up**: [../README.md](../README.md) (curiosity_control) · **Idea doc**: [performance_error_is_the_bridge](../../../../ideas/performance_error_is_the_bridge.md) §8 **Experiment 3**; also answers [two_timescale_value_loop](../../../../ideas/two_timescale_value_loop.md) open question #2 (can `b−e` replace the per-step derivative?).
**Code**: `analyze_benchmark_vs_cost.py` — **pure re-analysis of logged data, no new runs** (reruns locally in ~5s).
**Data**: a2a active-vision curiosity runs (Phase 1/2/2b regime trajectories, pulled to `data/` from Modal volume `language-reduction-data`) + this node's `frontier_err` records. Phase-1 raw per-iteration errors recovered exactly by inverting the logged EMA recursion (the logged series understates jitter ~4×).
**Status**: done. **Date**: 2026-08-11.

## One-liner

The idea doc's predicted disagreement cell is real and large: on a **noisy TV** the benchmark model `b−e`
reads **+0.0001 ± 0.0024 (statistically zero)** while the cost model `−α‖e‖+β(−d‖e‖/dt)` reads **−0.075,
sustained forever**. Everywhere the cost term changes a sign (noisy TV, dark room, steady tracking, early
frontier, first contact) the change is wrong or indifferent — **a separate standing cost term is not
supported** — and the phenomenology it was for (GATED_RATCHET's "high static residual = frustrating")
falls out of `b−e` as a **transient at re-opening**, firing exactly when frustration is informative. But
`b−e` is not a free upgrade: it is the same estimator family as LP (inherits steady-state disengagement),
its benchmark timescale has an **interior optimum**, and as an **allocation** score at re-opening it is
actively repulsive where the derivative is merely blind.

## The regime table (Phase-1 active vision, random arm; κ calibrated *in the cost model's favor*)

| regime (window) | e mean | bench δ mean (P>0) | cost mean (P>0) | relu-LP mean |
|---|---|---|---|---|
| frontier (it 1–163) | 0.083 | **+0.042** (0.99) | +0.083 (0.67) | 0.032 |
| noisy TV (300–1000) | 0.076 | **+0.0001** (0.48) | **−0.075** (0.00) | 0.0002 |
| dark room (200–1000) | 0.004 | +0.0002 (0.56) | −0.003 (0.15) | 0.0003 |
| mastered struct (400–1000) | 0.005 | +0.0002 (0.57) | −0.003 (0.14) | 0.0003 |

## The three load-bearing cells

- **Noisy TV**: `b−e` jitters symmetrically (std 0.0024); spurious-positive fraction 2.4% of the frontier
  signal, discriminability 17.5 — but its jitter is ~4× smoothed relu-LP's (it carries raw instantaneous
  noise the two-EMA derivative smooths away). **The EWMA timescale is load-bearing with an interior
  optimum** (fig4): half-life must sit above the sampling-jitter scale and below the frontier-descent
  scale (optimum t½ ≈ 14–70 it here, best 35; at t½=347 the fake fraction hits 64% — a too-slow benchmark
  remembers pre-floor errors and reads that *memory of the learnable transient* as sustained fake
  progress). This is §12(c)'s estimability condition made quantitative.
- **Re-opening after abrupt drift** (6 swap events + 2 confirming): **`b−e` is not blind — it is
  transiently, deeply negative** (dip −0.23 mean, ~5× frontier scale, duration ~29 it and set by b's
  timescale), then a frontier-scale positive hump through the re-descent, then →0. As a *value/gain*
  signal that is arguably the feature (worse-than-benchmark fires exactly when the world re-opens); as an
  *allocation* score it is worse than the derivative's blindness — the re-opened frontier reads repulsive
  while converged noise reads 0, so a positive-seeking allocator diverts at least as badly as Phase-2's
  fake-LP pathology. (Composed from separately-logged sequences — an inference, not an observation.)
- **Steady gradual tracking** (morph, it 300–1000): bench +0.0018 ≈ relu-LP 0.0025 — **`b−e` does not fix
  the derivative's steady-state disengagement; they are the same estimator family** (a lagged baseline
  minus current value *is* an LP estimator). Meanwhile cost reads −0.026: sustained frustration while
  successfully tracking a learnable drifting frontier — the same sign as its noisy-TV reading.

Also: at **first contact**, neither model discriminates a virgin learnable frontier from a noisy TV — under
the cost model both are strongly aversive (−0.19 / −0.27), under bench both read 0. The discrimination
lives entirely in the error *dynamics*.

## What the logged data structurally cannot discriminate

The mjc noisy-TV cell (no per-round noise-cell error was logged, only occupancy); avoidance-vs-indifference
toward noise (no logged drive implements either formulation, and the noise sits off the value corridor);
the agency gate σ(g/θ) (no arity-1/arity-2 pair in any log); closed-loop behavior of a δ-driven policy.
a2a substrate single-seed; windows chosen post-hoc; cross-formulation comparisons rely on normalized
metrics (P>0, fake-fraction, discriminability), not raw magnitudes.

## Figures & reproduce

`figures/`: `fig1_three_regimes.png`, `fig2_noisytv_zoom.png`, `fig3_reopening.png`,
**`fig4_ewma_sensitivity.png`** (the interior optimum), `fig5_mjc_frontier.png`. Machine-readable numbers
in `summary.json`.

```bash
python3 experiments/mjc/curiosity_control/benchmark_vs_cost/analyze_benchmark_vs_cost.py   # ~5s, local
```
