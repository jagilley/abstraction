# Language Reduction: Synthetic (RHM)

**Centralized writeup**: [README.md](README.md)

Controlled scaling law experiments using the Random Hierarchy Model (Cagnetta & Wyart, 2024).

## Motivation

Our natural-language experiments showed that vocab-only reduction is a channel intervention (changes β, not γ), and spectral denoising was entangled with the statistics being measured. We couldn't find a clean way to genuinely change the DGP of natural language. The RHM gives us a generative process with fully controllable hierarchical depth, where we know ground truth and can cleanly separate DGP vs channel interventions.

## The Random Hierarchy Model

The RHM generates sequences of length s^L from vocabulary {0, ..., v-1} via a hierarchy of composition rules. Each feature at level ℓ has m rules, each mapping to an s-tuple of level-(ℓ-1) features. This creates multi-scale correlations: tokens at distance s^ℓ are correlated through level-(ℓ+1) structure.

We train autoregressive transformers on concatenated RHM sequences and measure empirical scaling exponents α_D as a function of the DGP parameters.

## Key parameters

- **L** (depth): number of hierarchical levels. Primary DGP intervention. More levels → deeper hierarchy → richer long-range structure.
- **m** (synonymic multiplicity): number of equivalent composition rules per feature. More synonyms → more entropy per level.
- **s** (branching factor): size of each compositional tuple. Controls sequence length (s^L) and correlation scale spacing.
- **v** (vocabulary size): number of token types. Channel intervention — should change β but not γ.

## Default hyperparameters

For experiments that need language-like complexity, use: **L=6, m=4, v=8, s=2** with a **6L/6H/192D model (~2.7M params)**. This setting produces a rich enough hierarchy that the model learns 1-2 compositional levels but not all of them, placing it in the regime where FM residual structure and the L→m transition are observable. Smaller models (4L/128D) at this (L, m) plateau too early; lower m (e.g. m=2) is too easy and higher m (e.g. m=8) makes the model barely learn beyond level 0.

## Architecture

- `shared.py` — Modal infrastructure (app, volume, image, utilities)
- `stages.py` — Reusable experiment primitives (Modal functions)
- `rhm_data.py` — RHM data generation (hierarchy rules + corpus sampling)
- `model.py` — GPT-2 with `return_intermediates` and cerebellar callback support
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
