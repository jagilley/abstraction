# Language Reduction: Synthetic (RHM)

**Centralized writeup**: [README.md](README.md)

Controlled scaling law experiments using the Random Hierarchy Model (Cagnetta & Wyart, 2024).

## Motivation

Our natural-language experiments showed that vocab-only reduction is a channel intervention (changes β, not γ), and spectral denoising was entangled with the statistics being measured. We couldn't find a clean way to genuinely change the DGP of natural language. The RHM gives us a generative process with fully controllable hierarchical depth, where we know ground truth and can cleanly separate DGP vs channel interventions.

## Setup

The RHM generates sequences of length s^L from vocabulary {0, ..., v-1} via a hierarchy of composition rules. Each feature at level ℓ has m rules, each mapping to an s-tuple of level-(ℓ-1) features. This creates multi-scale correlations: tokens at distance s^ℓ are correlated through level-(ℓ+1) structure.

We train autoregressive transformers on concatenated RHM sequences and measure empirical scaling exponents α_D as a function of the DGP parameters.

## Key parameters

- **L** (depth): number of hierarchical levels. Primary DGP intervention. More levels → deeper hierarchy → richer long-range structure.
- **m** (synonymic multiplicity): number of equivalent composition rules per feature. More synonyms → more entropy per level.
- **s** (branching factor): size of each compositional tuple. Controls sequence length (s^L) and correlation scale spacing.
- **v** (vocabulary size): number of token types. Channel intervention — should change β but not γ.

## Architecture

- `shared.py` — Modal infrastructure (app, volume, image, utilities)
- `stages.py` — Reusable experiment primitives (Modal functions)
- `rhm.py` — RHM data generation
- `model.py` — GPT-2 model
- `measure.py` — Scaling exponent fitting

All computation runs on Modal (workspace `jagilley`). Data lives on the `rhm-scaling-data` volume.

## Primitives

All primitives are Modal functions in `stages.py`. Invoke directly, or import into experiment scripts.

### `generate_corpus`

Generate RHM rules and corpus for a given (v, s, L, m) setting.

```
modal run --detach rhm/stages.py::generate_corpus \
    --v 8 --s 2 --depth 6 --m 4 --n-tokens 20000000
```

Saves `corpus.npy`, `rules_L*.npy`, and `meta.json` to `/data/v{v}_s{s}_L{L}_m{m}/`.

### `train_model`

Train an autoregressive transformer on a generated corpus. Requires `generate_corpus` to have been run first for the same setting.

```
modal run --detach rhm/stages.py::train_model \
    --v 8 --s 2 --depth 6 --m 4 --n-tokens 100000 \
    --n-layer 4 --n-head 4 --n-embd 128
```

Saves model checkpoint and `results.json` to `/data/v{v}_s{s}_L{L}_m{m}/models/P_{n_tokens}/`.
Auto-scales training steps based on corpus size (5 epochs, floor 2000, cap 20000).

### `sweep`

Train at multiple P values for one (v, s, L, m) setting and compute the empirical scaling exponent α_D. Generates corpus if needed, then spawns parallel `train_model` calls.

```
modal run --detach rhm/stages.py::sweep \
    --v 8 --s 2 --depth 6 --m 4
```

Uses 7 P values spanning ~3 orders of magnitude (capped at 20M tokens). Saves `scaling.json` with fitted α_D and R².

### `measure_scaling`

Compute empirical α_D from already-trained models (post-hoc, no GPU needed).

```
modal run --detach rhm/stages.py::measure_scaling \
    --v 8 --s 2 --depth 6 --m 4
```

## Writing experiment scripts

Experiment scripts import primitives from `stages.py` and the Modal app from `shared.py`. Example — sweep across DGP parameters:

```python
"""Sweep across L and m values to compare scaling exponents."""

from rhm.shared import app, volume, DATA_DIR, setting_key
from rhm.stages import generate_corpus, sweep

@app.function(volumes={DATA_DIR: volume}, timeout=7200, memory=4096)
def dgp_sweep(l_values: str = "4,6,8", m_values: str = "2,4,8",
              v: int = 8, s: int = 2):
    Ls = [int(x) for x in l_values.split(",")]
    ms = [int(x) for x in m_values.split(",")]

    # Phase 1: generate all corpora in parallel
    gen_handles = []
    for L in Ls:
        for m in ms:
            seq_len = s ** L
            max_p = min(20_000_000, int(max(1000, 50 * seq_len) * 10 ** 3.0))
            h = generate_corpus.spawn(v=v, s=s, depth=L, m=m,
                                      n_tokens=int(max_p * 1.2))
            gen_handles.append(h)
    for h in gen_handles:
        h.get()

    # Phase 2: run sweeps in parallel
    sweep_handles = []
    for L in Ls:
        for m in ms:
            h = sweep.spawn(v=v, s=s, depth=L, m=m)
            sweep_handles.append((L, m, h))

    for L, m, h in sweep_handles:
        result = h.get()
        alpha = result["scaling"]["alpha_empirical"]
        r2 = result["scaling"]["r2"]
        print(f"  L={L}, m={m}: alpha={alpha:.4f}, R²={r2:.4f}")
```

Run with:
```
modal run --detach rhm/my_experiment.py::dgp_sweep
```

## Results (2026-06-20)

All results with v=8, s=2, 4-layer 128-dim GPT-2 (~0.8M params). Two independent runs reproduce within ~0.01:

| Setting | L | m | α_D | R² |
|---------|---|---|-----|-----|
| v8_s2_L4_m2 | 4 | 2 | 0.498 | 0.978 |
| v8_s2_L4_m8 | 4 | 8 | 0.327 | 0.975 |
| v8_s2_L6_m4 | 6 | 4 | 0.217 | 0.953 |
| v8_s2_L8_m2 | 8 | 2 | 0.438 | 0.987 |

Both L and m reduce α, but **m dominates by ~3:1**. At fixed L=4, quadrupling m (2→8) cuts α by 34%. At fixed m=2, doubling L (4→8) cuts α by only 12%. The effects compound: L=6/m=4 (α=0.217) is lower than either L=8/m=2 (0.438) or L=4/m=8 (0.327). The scaling bottleneck is synonymic multiplicity (per-level entropy), not hierarchy depth.

See `SWEEP_README.md` for full experimental details.

## Regime transition experiment (2026-06-21)

Tested whether the FM residual transitions from diffuse to rule-conditioned as the main model learns (the "L-regime → m-regime" hypothesis). Three settings (m=2,4,8) at L=4, with ground-truth hierarchy-conditioned eta² at each checkpoint. Result: **not testable at this scale** — the FM captures 99%+ of the computation at seq_len=16, leaving only architectural mismatch noise in the residual (cosine 0.994 vs 0.903 on MNIST where the structured phenomena emerge). The eta² measurement correctly reports no structure because there is none to find.

See `REGIME_TRANSITION_README.md` for full details. Key takeaway: the A2A meta-learning machinery requires the FM to genuinely struggle (cosine 0.90–0.97), which requires computational complexity that 16-token RHM sequences don't provide.

**Follow-up (2026-06-21)**: Cosine sweep at L=5,6 found L=6/m=2 with matched FM (14% of gap) gives cos=0.963 — in the sweet spot. But higher m makes the FM's job EASIER (model barely learns, computation is trivially predictable). Scaling up to 6L/6H/192D (~2.7M params) at m=4 produced the **L→m transition**: feature eta² rises monotonically (fL4*: 0.006→0.074, 12×; fL3*: 0.002→0.050, 25×) as the model learns hierarchical composition over 20K steps. Top1 PC shows a non-monotonic signature (rises to 33% then drops to 12%), confirming the shift from one generic FM error mode to multiple structured rule/feature discriminations. Code: `rhm_cosine_sweep.py`, `rhm_regime_trajectory.py`.

## Per-level loss decomposition (2026-06-21)

Since we know the DGP, each next-token prediction maps to a hierarchy level via the s-adic valuation of the position. Level 0 = within an s-tuple (easiest), level L-1 = root boundary (hardest). Two trajectory experiments at L=6:

- **m=2, 4L/128D**: Loss monotonically increases with level. Model learns bottom-up — level 0 drops from 2.04→0.67 in 200 steps while levels 3-5 barely budge. At convergence: L0=0.37 (82% below uniform), L1-2≈1.15 (44%), L3-5≈1.80 (15%). 4-layer model plateaus at ~2-3 levels of learned composition.
- **m=4, 6L/192D**: Same bottom-up pattern but m=4 makes every level harder. Even with 3.3× more params, levels 2-5 are all bunched near baseline (~1.93 vs uniform 2.08). Model can only really compose 1-2 levels at m=4.

This provides the mechanistic picture behind the L→m transition: the bottom-up learning wave is what drives the monotonic rise in feature eta² — the FM residual gains structure as the model learns each successive level.

See `PER_LEVEL_LOSS_README.md` for full results. Code: `rhm_per_level_loss.py`.

## Prior experiment

See `../language_reduction/STATUS.md` for the natural-language results that motivated this.
