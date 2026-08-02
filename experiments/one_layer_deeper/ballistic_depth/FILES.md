# Files — `ballistic_depth`

**Up**: [README.md](README.md) (this node) · [../README.md](../README.md) (one_layer_deeper)

## Code files

| file | purpose |
|---|---|
| `ballistic_depth.py` | The cut. Builds the shared encoder / tied operator / decoder, runs the arms (`base`, `quant`, `consist`, `quant_consist`, `feedforward`), and emits every instrument. Owns `_make_model` (the `Encoder.forward_soft_digits` round trip that makes the cycle term differentiable, the straight-through `Quantizer` with data-seeded codebook and dead-code revival, the tied `Block`), the precomputed prompt/digit tables, and the four eval routines: `eval_exact`, `rollout_probe` (veridicality + on-manifold cosine + codebook occupancy), and `amplification(direction)` for `random` \| `on_manifold`. All knobs are CLI flags; `--consist-cycle` / `--consist-reentry` are what the ablations and the weight sweep move. |
| `analyze.py` | Cross-seed aggregation for one tag: the exact-match table split at the trained-depth boundary, the 2×2 main-effects/interaction decomposition, per-step amplification, veridicality and on-manifold curves, codebook occupancy, and `figures/<tag>/fig_ballistic_depth.png` (depth extrapolation · amplification · veridicality). |
| `sweep_figure.py` | The re-entry weight sweep and the amplification-instrument calibration: horizon vs weight, closure vs weight, and random-vs-on-manifold amplification against the analytic `2^t` bound. Writes `figures/sweep/fig_sweep.png`. |
| `closure_horizon.py` | **README §6–§7, no GPU.** Pure re-analysis over every `results/*/results_seed*.json`: closure-at-fixed-`t` against exact@20 and against the uncensored horizon, across 72 recurrent runs / 24 configs. Owns the honesty machinery that makes the readout trustworthy — right-censoring of the horizon in the 20-depth runs, the **bimodality split** (pooled R² is a two-cluster lever; within-cluster fits are the real test), the ID-accuracy control, the cycle-weight sweep table at re-entry=0 (the de-confounded family), and `transfer_test`, which fits on one knob-family and predicts the others. Writes `figures/closure_horizon/fig_closure_horizon.png`. |
| `reprojection.py` | **README §9.** Test-time re-projection, on Modal: loads a saved checkpoint, trains nothing, and re-runs the rollout with a periodic snap `h <- (1-alpha)*h + alpha*Enc(decode(h))`. Sweeps period `k`, blend `alpha`, and `mode` — `self` (the model's own decode, deployable) vs `oracle` (the true residue, the ceiling). Rebuilds the task tables and the train/test split from the checkpoint's own `cfg` so the "seen x" pool matches training. |
| `reprojection_figure.py` | Aggregates `reproj_coldstart` across seeds and checks the chain model `p ** ceil(T/k)` against the measured curves, with `p` read off the oracle plateau. Writes `figures/reprojection/fig_reprojection.png`. |
| `coldstart_figure.py` | **README §8.** Functional re-enterability: aggregates `coldstart_seen_x` across seeds, reports restart gain (cold start at `t` minus the full ballistic rollout) and the residual against `exact@(T−t)` — the reference that separates *the operator is unsound* from *the operator is sound and the rollout drifts off its domain*. Writes `figures/coldstart/fig_coldstart.png`. |

Shared task code lives one level up at [`../squaring_mod.py`](../squaring_mod.py) (vendored task +
trajectory generation) and [`../shared.py`](../shared.py) (Modal app/image/volume), per
[STRUCTURE.md](../../../STRUCTURE.md) — code lives at the lowest node that shares it.

## Result tags

| tag | what it is |
|---|---|
| `cut1` | The main 2×2 + both controls, 5 arms, eval to T=20. |
| `deep60` | `base` + `cycle-only` evaluated to T=60 — the composition-horizon measurement. |
| `cyc_only` / `re_only` | The consistency-term ablations at w=1.0 (`consist_reentry=0` / `consist_cycle=0`). |
| `iso_compute` | `base` at 2.5× gradient steps — the compute confound control. |
| `w_re0` … `w_re1.0` | The re-entry weight sweep (0.0 / 0.1 / 0.3 / 1.0), eval to T=60, carrying the on-manifold amplification instrument. |
| `quant_fix` | The repaired quantiser at codebook 256 (was 4096). Still fails; see README §5. |
| `w_cyc0.01` … `w_cyc30.0` | **The cycle weight sweep** (0.01 / 0.05 / 0.2 / 3.0 / 10.0 / 30.0, with w=1.0 supplied by `w_re0`+`deep60`+`cyc_only`), re-entry pinned at 0, eval to T=60. Horizon peaks at **51.0 at w=10** and turns over to 28.0 at w=30 — README §7. L4. |
| `reproj_coldstart` | Test-time re-projection over the `coldstart` checkpoints — README §9. Grounded arm hits **1.000 at T=60** with no retraining; ungrounded arm 0.000 → 0.377. L4. |
| `coldstart` | `base` + `cycle-only` re-run with the cold-start re-enterability probe and checkpointing — README §8. Config-identical to `w_re0`, so it also serves as an L4-vs-A10G replication of it. L4. |

Results land on the Modal volume `one-layer-deeper-data` at
`/ballistic_depth/<tag>/results_seed<N>.json` and mirror locally to `results/<tag>/`.

## Figures

| path | content |
|---|---|
| `figures/deep60/fig_ballistic_depth.png` | The headline: depth extrapolation to T=60, error amplification, veridicality of the intermediate state. |
| `figures/cut1/fig_ballistic_depth.png` | Same three panels for the full 5-arm 2×2 at T≤20. |
| `figures/sweep/fig_sweep.png` | Horizon vs re-entry weight, closure vs re-entry weight, and the random-vs-on-manifold instrument calibration. |
| `figures/closure_horizon/fig_closure_horizon.png` | Closure vs depth outcome: the pooled scatter with the empty region marked (the pooled fit spans it), the within-cluster fits that are the actual evidence, and the open-regime panel where `re_only` buys depth without closure. |
| `figures/reprojection/fig_reprojection.png` | Per-arm depth curves with and without re-projection against the chain model, plus the period sweep showing the ungrounded arm's two-sided optimum at k=8 against its horizon of 13. |
| `figures/coldstart/fig_coldstart.png` | Exact match vs re-entry point `t` for base and cycle, against the dotted `exact@(T−t)` reference, plus restart gain. The base panel is the result: flat near zero from `t=0`, rising to 0.87 only in the last few steps. |
