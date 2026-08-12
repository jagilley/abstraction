# The agency gate: ACT vs PLAYBACK — δ discriminates on the efference copy, and the gate does the work

**Up**: [../README.md](../README.md) (mjc) · **Idea doc**: [performance_error_is_the_bridge](../../../ideas/performance_error_is_the_bridge.md) §8 **Experiment 2** (the Gadagkar playback control; §12 named it the most informative remaining test)
**Code**: `agency_gate.py` · File index: [FILES.md](FILES.md)
**Status**: done, **single seed** (justified below). **Date**: 2026-08-11.

## One-liner

On **identical sensory sequences**, the bridge signal δ = (b−e)·σ((g−g₀)/θ) fires at sensory-distortion
events when the trajectory is self-produced (**ACT**, efference copy present) and is statistically silent
when the same trajectory is passively replayed (**PLAYBACK**): event-triggered δ **−0.0330 ± 0.0009 vs
−0.00009 ± 0.00021** (n=211 each, ~369×). The dissection is the finding: the *ungated* `b−e` fires in
**both** conditions (PLAYBACK −0.0131 ± 0.0020, 6.5 sem from zero) — **the agency gate, not the sensory
stream, carries the discrimination**. This is the in-silico analog of Gadagkar et al. 2016 Fig 4
(VTAerror neurons respond during singing, not to passive playback of identical acoustics).

## Method

`arity_torque` world and protocol: FM₁(s) and FM₂(s,u) trained on the same 100K i.i.d.-command free-flight
transitions (max |corr(u,s)| = 0.0034 train / 0.012 eval — the generator confound controlled; arity sanity
reproduced: free-flight pusher-vel R² **0.999 (FM₂) vs 0.003 (FM₁)**). 50 held-out episodes; the same
bridge computation run twice per trajectory, differing only in the efference-copy slot: ACT carries the
true `u_t`, PLAYBACK carries motor silence (`u=0` — the principled null: E[u]=0 and near-linear-in-u
dynamics, so FM₂(s,0)→FM₁(s) is a *property of the learned models*, making g's collapse in PLAYBACK an
empirical outcome, not definitional). Gate calibrated on the training split only (g₀ = midpoint of
passive/active g medians 0.00245/0.0407; θ = gap/8). Gadagkar-style distortion: one-step kicks (magnitude
= natural per-step signal RMS, 0.0427) to the *observed* feedback on the value slice at 435 free-flight
target steps, 211 distorted — physics untouched, so sensory sequences stay exactly matched.

## Results (seed 0)

| readout | ACT | PLAYBACK |
|---|---|---|
| event-triggered δ at distortion | **−0.0330 ± 0.0009** | **−0.00009 ± 0.00021** |
| event-triggered `b−e` (ungated) | −0.0391 | **−0.0131 ± 0.0020** (fires!) |
| b_T quadrant: distorted / omitted | **−0.0187 / +0.0165** | −0.0001 / +0.0002 |
| per-step e median (free flight, n=11,195) | 0.00087 | 0.0408 |
| per-step g median (AUROC 0.995) | 0.0405 | 0.0023 |
| gate median | 0.981 | 0.018 |

- **The bidirectional Gadagkar signature reproduces** (row 3, via the target-conditioned benchmark b_T):
  suppression on worse-than-predicted, activation at the precise moment a predicted distortion does *not*
  occur; PLAYBACK silent in both directions.
- **This cannot be "prediction error gates learning"**: predictability is matched by construction, and the
  condition with the *larger* prediction error (PLAYBACK, e median 47×) is the silent one.
- **§1's formula as written fails its own experiment**: σ(0)=0.5, so the literal σ(g/θ) leaks half the
  sensory channel in PLAYBACK (δ RMS 0.0132, ~8× the calibrated gate's 0.0017). The **centered gate
  σ((g−g₀)/θ)** is required; calibration is cheap and self-supervised (medians of one's own
  active/passive g).
- **The ACT gate is graded, not binary** (mean 0.809, p05 0.10): a near-zero sampled command yields
  genuinely small g — an efference copy of "do nothing" grants no agency over the value slice. Agency is a
  per-timestep magnitude tracking ‖u‖, not a condition flag.
- Contact transitions pass the gate in ACT as large negative δ — genuine worse-than-benchmark performance
  errors while acting (consistent with Cut #1's contact-residual concentration); the per-step summary
  stats and event analysis are free-flight-only by construction.

## Caveats

Single seed 0: the discriminating quantities are 1–2 orders beyond their standard errors and the mechanism
is structural (the total arity gap on actuated dims), so extra seeds would assert a seed-sensitivity we
don't hold. Single task family; distortions applied on the value slice only; the doc's literal FM₂≡FM₁
PLAYBACK variant was also computed (trivially silent, δ RMS 0.0002).

## Figures & reproduce

`figures/full_v1/`: `fig1_distributions.png` (per-step e/g/gate/δ per condition), **`fig2_event_triggered.png`**
(the Gadagkar analog: gated vs ungated traces + b_T quadrant), `fig3_calibration.png` (raw g distributions
+ both gate curves), `fig4_timeseries.png`. Full stats in `results.json`; per-step arrays in
`eval_traces.npz` on the volume (`/data/agency_gate/<tag>/`).

```bash
cd experiments/
modal run mjc/agency_gate/agency_gate.py::agency_gate --quick            # smoke
modal run --detach mjc/agency_gate/agency_gate.py::agency_gate --tag full_v1 --seed 0
```
