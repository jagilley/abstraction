# RHM Scaling Exponent Sweep: L × m (2026-06-20)

**Code**: `hparam_sweep.py`
**Prior experiment**: See `CLAUDE.md` for initial results and RHM setup.

## Goal

Measure how the empirical scaling exponent α_D depends on the DGP parameters L (hierarchy depth) and m (synonymic multiplicity) in the Random Hierarchy Model. The initial run (3/9 settings completed) suggested both matter — this sweep fills in the picture with a controlled design.

## Design

3 settings × 3 P values = 9 GPU jobs (T4), chosen to isolate both main effects:

| Setting | L | m | Role |
|---------|---|---|------|
| v8_s2_L4_m2 | 4 | 2 | Simple baseline |
| v8_s2_L4_m8 | 4 | 8 | Same L, 4× more synonyms (isolates m) |
| v8_s2_L8_m2 | 8 | 2 | Same m, 2× deeper hierarchy (isolates L) |

P values at exponents [0, 1.5, 3.0] relative to min_p (= max(1000, 50 × s^L)), spanning 3 orders of magnitude per setting. All models: 4L/4H/128D GPT-2 (~0.8M params), v=8, s=2.

## Results

### This run

| Setting | L | m | α_D | R² |
|---------|---|---|-----|-----|
| v8_s2_L4_m2 | 4 | 2 | 0.498 | 0.978 |
| v8_s2_L4_m8 | 4 | 8 | 0.327 | 0.975 |
| v8_s2_L8_m2 | 8 | 2 | 0.438 | 0.987 |

### Combined with initial run

| L | m | α_D (this run) | α_D (initial) |
|---|---|----------------|---------------|
| 4 | 2 | 0.498 | 0.485 |
| 4 | 8 | 0.327 | 0.324 |
| 6 | 4 | — | 0.217 |
| 8 | 2 | 0.438 | — |

The two overlapping settings reproduce within ~0.01 — good consistency across runs.

### Effect sizes

**m effect** (L=4 fixed, m: 2 → 8): α drops 0.498 → 0.327 (−34%). 4× more synonyms per level substantially flattens the scaling curve.

**L effect** (m=2 fixed, L: 4 → 8): α drops 0.498 → 0.438 (−12%). Doubling hierarchy depth has a much weaker effect.

**Combined** (L=6, m=4 from initial run): α = 0.217, lower than both L=8/m=2 (0.438) and L=4/m=8 (0.327). Both parameters contribute, and their effects compound.

## Interpretation

Synonymic multiplicity m dominates over hierarchy depth L in determining the scaling exponent. This makes sense: m controls the per-level entropy (how many equivalent composition rules the model must distinguish at each hierarchical level), while L controls the depth of composition. The scaling bottleneck is in resolving synonym structure — learning which of the m equivalent rules produced each observation — not in learning to compose across levels.

This is consistent with natural-language intuitions: languages with extensive synonymy/paraphrase (high m) should be harder to scale on than languages with rigid compositional rules (low m), even if the latter have deeper syntactic structure (high L).

## Reproduction

```bash
cd experiments/
modal run --detach language_reduction_synthetic/hparam_sweep.py::hparam_sweep
```

## Modal volume

Results saved to `rhm-scaling-data` volume at `/data/hparam_sweep_compact.json`. Per-setting models at `/data/v{v}_s{s}_L{L}_m{m}/models/P_{n_tokens}/`.
