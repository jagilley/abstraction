# Bridge assembly: the corrected δ runs live, hygiene reaches behavior through the *reactive* channel, and the gain's net value is regime-dependent

**Up**: [../README.md](../README.md) (mjc) · **Idea doc**: [performance_error_is_the_bridge](../../../ideas/performance_error_is_the_bridge.md) — closes §11's "the assembly does not have support" gap as a *mechanism* claim.
**Direct parent**: [`../plasticity_gain/`](../plasticity_gain/README.md) (substrate + budget discipline; its nominated follow-up — "an environment where retained/base competence is behaviorally load-bearing"). Corrected-δ components from [`../agency_gate/`](../agency_gate/README.md) (centered gate + calibration protocol) and [`../curiosity_control/benchmark_vs_cost/`](../curiosity_control/benchmark_vs_cost/README.md) (benchmark-timescale estimability window).
**Code**: `bridge_assembly.py` (runner), `analyze_assembly.py` (aggregation), `launch_detached.py` (setsid-isolated launcher) · File index: [FILES.md](FILES.md)
**Status**: done, 3 seeds (`asm_s0/s1/s2`). **Date**: 2026-08-11.

## One-liner

With mastered competence made behaviorally load-bearing, **δ's hygiene advantage over raw-error gain
reaches control** — goal-distance ordering fixed < delta < raw in 3/3 seeds — but through the **reactive**
controller, inverting the arc's transmission expectation, and **ungated uniform plasticity wins this
regime outright**. The joint reading with plasticity_gain: which controller transmits an FM difference
depends on the difference's **spatial structure** (localized staleness → ballistic; diffuse churn →
reactive), and δ's net value over uniform is **regime-dependent** (pays on long/hard recovery transients;
taxed by benchmark lag when recovery is short). The fully corrected δ — context-conditional b(s), centered
gate, gain consumption — ran live with every component behaving, 3/3 seeds.

## Method

plasticity_gain's substrate and matched-average-lr discipline with one changed thing — the task geometry.
Two reach corridors: **B-mastered** (φ=−1.2, present pre-drift, oversampled in pretraining as "practice",
kept on the eval path) and **A-drift** (φ=+1.2, appears at t=0) + the off-path aleatoric decoy (amp 30).
Preconditions landed 3/3 seeds: stale B-probe 0.0044–0.0053 vs A-probe 0.119–0.127; stale
mastered/ballistic ≈ ceiling vs drift/ballistic 0.293–0.309 — **retention genuinely load-bearing,
adaptation genuinely demanded**. Arms `fixed`/`raw_err`/`delta` over an identical 16,384-transition
stream; delta = the assembly live: b(s) net at bench_lr 3e-3 (inside the estimability window), centered
gate σ((g−g₀)/θ) calibrated by agency_gate's train-split protocol, g computed per batch (current FM₂ vs
frozen FM₁), plus a logged no-gate counterfactual.

## Q1 — does δ's hygiene transmit to behavior?

**Mechanism (seed-stable to 2 decimals).** Noise over-weighting: raw **3.97 ± 0.02×** vs delta
**1.53 ± 0.02×**. Raw *under*-weights the mastered region (0.79×); delta holds it above nominal (1.22×,
rising to ~1.4 late — active repair when churn lifts e above b). Retention finals (B-probe): fixed 0.0123
< delta 0.0137 < raw 0.0176 — **delta beats raw 3/3 on retention and base**, and raw is worst despite
receiving ~15% *less* average lr (clip handicap): the damage is allocation, not budget.

**Behavior, reactive (the low-variance channel): fixed < delta < raw in 3/3 seeds** on both families and
aggregate (agg AUC 0.0135 / 0.0153 / 0.0165; raw's excess over ceiling ≈ 2.6× fixed's). The hygiene
difference reaches control.

**Behavior, ballistic: no separation at 3 seeds** (agg 0.1035/0.1041/0.1042; between-seed sd ≫ arm
deltas). Identifiable reason: this single-region drift's ballistic recovery completes by ~512 transitions,
so the arms' FM differences (~0.002–0.005) sit far below ballistic eval noise. Seed 0's apparent delta win
did not replicate — the multi-seed escalation was the right call.

**The transmission inversion (the transferable finding).** plasticity_gain/4b's "ballistic transmits,
reactive near-blind" flips here because what differs across arms is *diffuse churn*, not localized
staleness. Reading: a ballistic controller is a deep, narrow line-integral of model quality along its
committed corridor (huge gain on localized on-path error, high execution variance); a reactive controller
is a shallow, wide surface-average (small gain per query, support over the whole visited space, ~10×
smaller eval variance). **Which controller transmits an FM difference depends on where the difference's
mass lives**: localized → ballistic, diffuse → reactive. Retroactively, this sharpens the arc's
"control is a blind grader" episodes into a matching condition rather than a fixed property.

**The honest headline: at matched budget, uniform plasticity beat both error-modulated forms on the
hygiene-transmitting readout, 3/3.** The mechanism is visible in the traces: recovery here is short/easy,
and δ pays a lag tax — late in the run the benchmark trails the still-improving frontier, δ_A goes
positive, and delta down-weights exactly the region still worth learning (w_A ≈ 0.76; final A-probe 0.025
vs fixed's 0.020) — benchmark_vs_cost's "same estimator family as LP" caveat, live in closed loop. Joint
reading with plasticity_gain (where delta beat fixed 3/3 on an 8K-transition recovery): **δ's value over
fixed lives where recovery transients are long/hard; its value over raw is hygiene/retention, and is
unconditional in our data.** All arms also show an arm-nonspecific mastered-family degradation hump at
m=256–512 at this noise level.

## Q2 — the assembly ran live, every component behaved (3/3 seeds)

Gate: train AUROC 0.993–0.994, non-degenerate in flight (~0.83 mean / ~0.99 median; passive
counterfactual ~0.02), budget-neutral (delta mean_w 0.993–0.998) — present, calibrated, didn't hurt.
b(s): habituates the noise δ (−0.048 → −0.017) and withdraws from noise while *boosting* B-repair; its
known lag trade-off appeared live (above). §11's assembly gap is closed as a mechanism claim; the
assembly's *net behavioral payoff* is the regime-dependent part.

## Caveats

Single task family; one drift shape; n_eval=40 medians; the mastered-family "ceiling" is seed-noisy
(0.055–0.108) and the practiced stale FM legitimately beats a uniform-pool fresh FM on that family
(off-policy reference, not a ceiling). Figure-reading: smoothed-trace edges are convolution artifacts;
`fig5_gate.png`'s gated-vs-nogate panel mixes normalized/raw scales (compare shapes, not levels).

## Figures & reproduce

Per seed under `figures/bridge_assembly_asm_s{0,1,2}/`: `fig1_recovery.png` (2×3 control ladders),
`fig2_fm_traces.png` (per-region probes; B = the retention trace), `fig3_allocation.png`,
`fig4_delta_mech.png` (δ / b-vs-e / gate traces), `fig5_gate.png`, + `results.json`. Volume:
`/data/bridge_assembly/asm_s{0,1,2}/` (+ `smoke/`), completion sentinel `done.txt`.

```bash
cd experiments/
modal run mjc/bridge_assembly/bridge_assembly.py::bridge_assembly --quick     # smoke
for s in 0 1 2; do
  python3 mjc/bridge_assembly/launch_detached.py --tag asm_s$s --seed $s      # setsid-isolated detach
done
python3 mjc/bridge_assembly/analyze_assembly.py --tags asm_s0,asm_s1,asm_s2
```
