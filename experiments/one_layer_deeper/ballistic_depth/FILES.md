# Files — `ballistic_depth`

**Up**: [README.md](README.md) (this node) · [../README.md](../README.md) (one_layer_deeper)

## Code files

| file | purpose |
|---|---|
| `ballistic_depth.py` | The cut. Builds the shared encoder / tied operator / decoder, runs the arms (`base`, `quant`, `consist`, `quant_consist`, `feedforward`), and emits every instrument. Owns `_make_model` (the `Encoder.forward_soft_digits` round trip that makes the cycle term differentiable, the straight-through `Quantizer` with data-seeded codebook and dead-code revival, the tied `Block`), the precomputed prompt/digit tables, and the four eval routines: `eval_exact`, `rollout_probe` (veridicality + on-manifold cosine + codebook occupancy), and `amplification(direction)` for `random` \| `on_manifold`. All knobs are CLI flags; `--consist-cycle` / `--consist-reentry` are what the ablations and the weight sweep move. |
| `analyze.py` | Cross-seed aggregation for one tag: the exact-match table split at the trained-depth boundary, the 2×2 main-effects/interaction decomposition, per-step amplification, veridicality and on-manifold curves, codebook occupancy, and `figures/<tag>/fig_ballistic_depth.png` (depth extrapolation · amplification · veridicality). |
| `sweep_figure.py` | The re-entry weight sweep and the amplification-instrument calibration: horizon vs weight, closure vs weight, and random-vs-on-manifold amplification against the analytic `2^t` bound. Writes `figures/sweep/fig_sweep.png`. |

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

Results land on the Modal volume `one-layer-deeper-data` at
`/ballistic_depth/<tag>/results_seed<N>.json` and mirror locally to `results/<tag>/`.

## Figures

| path | content |
|---|---|
| `figures/deep60/fig_ballistic_depth.png` | The headline: depth extrapolation to T=60, error amplification, veridicality of the intermediate state. |
| `figures/cut1/fig_ballistic_depth.png` | Same three panels for the full 5-arm 2×2 at T≤20. |
| `figures/sweep/fig_sweep.png` | Horizon vs re-entry weight, closure vs re-entry weight, and the random-vs-on-manifold instrument calibration. |
