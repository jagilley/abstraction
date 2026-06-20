# Language Reduction: Synthetic (RHM)

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

## Running

All stages run on Modal (workspace `jagilley`). Data lives on the `rhm-scaling-data` volume.

Via local entrypoint:
```
modal run language_reduction_synthetic/modal_app.py --stage generate --depth 6 --mult 4
modal run language_reduction_synthetic/modal_app.py --stage train --depth 6 --mult 4 --n-tokens 100000
modal run language_reduction_synthetic/modal_app.py --stage sweep --depth 6 --mult 4
modal run language_reduction_synthetic/modal_app.py --stage full-sweep
```

Via direct function invocation (for `--detach`):
```
modal run --detach language_reduction_synthetic/modal_app.py::full_sweep --l-values "4,6,8" --m-values "2,4,8"
modal run --detach language_reduction_synthetic/modal_app.py::sweep --L 6 --m 4
```

## Initial results (2026-06-20)

First run with v=8, s=2, 4-layer 128-dim GPT-2. Three of nine settings completed:

| Setting | L | m | α_D | R² |
|---------|---|---|-----|-----|
| v8_s2_L4_m2 | 4 | 2 | 0.485 | 0.957 |
| v8_s2_L4_m8 | 4 | 8 | 0.324 | 0.931 |
| v8_s2_L6_m4 | 6 | 4 | 0.217 | 0.953 |

α clearly varies with both L and m — simpler DGPs (shallow depth, few synonyms) scale more steeply. Six settings failed due to a volume sync race condition (now fixed — `volume.reload()` added to `train_model`).

## Prior experiment

See `../language_reduction/STATUS.md` for the natural-language results that motivated this.
