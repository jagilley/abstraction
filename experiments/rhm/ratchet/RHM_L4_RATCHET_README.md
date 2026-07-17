# RHM Sparse Ratchet at L=4: Overparameterization Test (2026-06-24)

**Code**: `rhm_sparse_ratchet.py` (same code, `--depth 4`)
**Prior experiments**: [RHM sparse ratchet at L=6](RHM_SPARSE_RATCHET_README.md), [MNIST gated ratchet](../../a2a_forward/GATED_RATCHET_README.md)

## Goal

Test whether the RHM ratchet's failure to compound is caused by the model being capacity-limited at higher compositional levels. The sparse ratchet at L=6/m=2 showed +1.2% val loss improvement but non-compounding per-level gains. Meanwhile, the MNIST ratchet (WS_UG_uniform) compounded beautifully (0.0925 → 0.0477, ~49% improvement over 4 cycles).

One hypothesis: the 6L/6H/192D model can only compose ~3 hierarchy levels (napkin math: ~2 layers per compositional level). At L=6, the model is capacity-limited at L3-L5 — the ratchet can't help because there's only one (or no) computation strategy available at those levels. The ratchet is fundamentally a regularization tool that selects among equivalent solutions for the most self-compressible one, and capacity-limited settings have no equivalent solutions to select among.

Reducing L from 6 to 4 puts the model in the overparameterized regime (6 layers for 4 compositional levels). If the hypothesis is correct, the ratchet should compound because the model has excess capacity and multiple equivalent computation strategies — regularization headroom.

## Design

Identical to the [L=6 sparse ratchet](RHM_SPARSE_RATCHET_README.md) except:
- **L=4** (seq_len = 2^4 = 16, down from 2^6 = 64)
- **mask_rates = 0.0, 0.75, 0.90, 0.95** (added dense NTP as a control)

All other parameters unchanged: 6L/6H/192D model (~2.68M params), 1L/1H/16D causal FM (50.4K params, 1.9%), post_block0 → post_block5, inject after block 1, UnifiedGate (74.2K params), 4 cycles × (3000 wake + 750 sleep), gated local loss (λ=1.0), seed=42.

Note: at mask=0.95 with seq_len=16, ~46% of sequences have zero supervised positions (0.95^15 ≈ 0.46). These sequences receive only process supervision from the local loss.

## Results

### Val loss trajectory

Gap = OL − WS. Positive = WS wins.

| mask | Cycle 1 | Cycle 2 | Cycle 3 | Cycle 4 |
|---|---|---|---|---|
| 0.00 | -0.001 (-0.1%) | -0.001 (-0.1%) | -0.004 (-0.5%) | -0.005 (-0.5%) |
| 0.75 | +0.001 (+0.1%) | -0.002 (-0.2%) | +0.002 (+0.2%) | -0.010 (-1.1%) |
| 0.90 | **+0.021 (+2.0%)** | +0.000 (0.0%) | **+0.012 (+1.2%)** | +0.003 (+0.3%) |
| 0.95 | -0.009 (-0.8%) | **+0.015 (+1.4%)** | **+0.015 (+1.5%)** | +0.005 (+0.5%) |

The sparsity crossover replicates from L=6: WS helps at mask≥0.90, hurts or ties at mask≤0.75. At mask=0.90, WS wins most strongly at cycle 1 (+2.0%) then narrows. At mask=0.95, WS wins at cycles 2-3 (~+1.5%) then narrows by cycle 4 (+0.5%). The gap does not compound — it peaks mid-training and shrinks.

Dense NTP (mask=0.0): OL wins at every cycle, with the gap growing slightly from -0.1% to -0.5%.

### Per-level loss at each cycle

| mask | Cyc | WS val | OL val | WS L0 | OL L0 | WS L1 | OL L1 | WS L2 | OL L2 | WS L3 | OL L3 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.00 | 1 | 0.950 | 0.949 | 0.478 | 0.492 | 1.105 | 1.105 | 1.536 | 1.533 | 1.652 | 1.628 |
| 0.00 | 4 | 0.937 | 0.933 | 0.473 | 0.465 | 1.090 | 1.093 | 1.503 | 1.509 | 1.640 | 1.632 |
| 0.90 | 1 | 1.033 | 1.054 | 0.553 | 0.601 | 1.220 | 1.278 | 1.606 | 1.612 | 1.691 | 1.720 |
| 0.90 | 4 | 0.963 | 0.966 | 0.503 | 0.500 | 1.132 | 1.143 | 1.571 | 1.541 | 1.649 | 1.659 |
| 0.95 | 1 | 1.117 | 1.107 | 0.623 | 0.626 | 1.402 | 1.346 | 1.693 | 1.672 | 1.746 | 1.745 |
| 0.95 | 4 | 0.996 | 1.001 | 0.527 | 0.533 | 1.162 | 1.201 | 1.592 | 1.576 | 1.657 | 1.673 |

### Per-level delta (WS − OL; negative = WS better)

| mask | Cyc | Δval | ΔL0 | ΔL1 | ΔL2 | ΔL3 |
|---|---|---|---|---|---|---|
| 0.00 | 1 | +0.001 | -0.014 | -0.001 | +0.003 | +0.024 |
| 0.00 | 4 | +0.005 | +0.008 | -0.002 | -0.006 | +0.008 |
| 0.90 | 1 | **-0.021** | **-0.048** | **-0.058** | -0.006 | **-0.029** |
| 0.90 | 4 | -0.003 | +0.003 | -0.011 | +0.031 | -0.010 |
| 0.95 | 1 | +0.009 | -0.002 | +0.056 | +0.021 | +0.001 |
| 0.95 | 3 | **-0.015** | -0.007 | **-0.094** | +0.031 | **-0.032** |
| 0.95 | 4 | -0.005 | -0.006 | **-0.039** | +0.016 | -0.016 |

At mask=0.90, cycle 1 shows broad improvement at all levels (L0: -0.048, L1: -0.058, L3: -0.029). This is larger than L=6's cycle 1 but doesn't persist — by cycle 4, the deltas are near zero.

At mask=0.95, the largest improvement is at L1 (cycle 3: -0.094), with L3 also improving (-0.032). But L2 moves in the opposite direction (+0.031). The improvements don't compound — cycle 4 is weaker than cycle 3.

### FM residual and gate dynamics

| Condition | cos | norm | rank% | top1% |
|---|---|---|---|---|
| WS m0.00 | 0.987 | 1.64 | 76.8 | 11.4 |
| OL m0.00 | 0.960 | 28.5 | 49.5 | 26.4 |
| WS m0.95 | 0.991 | 3.60 | 61.2 | 16.0 |
| OL m0.95 | 0.967 | 43.8 | 36.6 | 37.8 |

The FM cosine is very high at L=4: 0.987–0.991 for WS (vs 0.977 at L=6) and 0.960–0.967 for OL (vs 0.939 at L=6). The FM captures 96–99% of the model's computation. WS residual norms are 17× smaller than OL (1.6–3.6 vs 28–44), indicating the local loss drives computation toward near-perfect FM-predictability.

Gate dynamics at mask=0.95 (post-wake):

| Cycle | mean | sparse(<0.1) |
|---|---|---|
| 1 | 0.135 | 49.5% |
| 2 | 0.094 | 69.8% |
| 3 | 0.075 | 76.6% |
| 4 | 0.065 | 81.2% |

The gate closes monotonically, as at L=6. By cycle 4, 81% of dimensions are below 0.1.

### Robustness and self-knowledge

| Condition | rob (eps=1.0) | SK post_block0 | SK post_block5 |
|---|---|---|---|
| WS m0.00 | +0.225 | 0.067 | 0.191 |
| OL m0.00 | +0.002 | 0.023 | 0.439 |
| WS m0.95 | +0.014 | 0.025 | 0.206 |
| OL m0.95 | +0.001 | 0.015 | 0.421 |

WS is 10–100× more brittle than OL, replicating the cross-domain finding that process supervision hurts robustness.

Self-knowledge probes show **OL > WS at post_block5** (R² = 0.39–0.44 vs 0.19–0.21). This is the opposite of the MNIST and language results, where closed-loop models had higher self-knowledge. The inversion is explained by the FM cosine: at 0.987+, the WS residual is dominated by architectural noise (1-head FM vs 6-head model), leaving no structured signal for probes to predict.

## Comparison with L=6

| | L=6, mask=0.95 | L=4, mask=0.95 | L=4, mask=0.90 |
|---|---|---|---|
| Best WS gap | +1.2% (c4) | +1.5% (c3) | +2.0% (c1) |
| Compounds? | No (oscillates) | No (peaks then narrows) | No (narrows monotonically) |
| FM cosine (WS) | 0.977 | 0.991 | 0.989 |
| FM cosine (OL) | 0.939 | 0.967 | 0.964 |
| Gate c4 mean | 0.071 | 0.065 | — |
| SK (WS post_block5) | — | 0.206 | 0.203 |
| SK (OL post_block5) | — | 0.421 | 0.420 |

The improvements at L=4 are slightly larger in magnitude (+2.0% at mask=0.90 cycle 1 vs +1.2% at L=6 mask=0.95 cycle 4) but share the same qualitative pattern: they appear at specific cycles but do not compound. The FM cosine is higher at L=4 (the task is easier to predict), the gate closes at similar rates, and the self-knowledge inversion (OL > WS) is new relative to L=6.

## Interpretation

### The regularization headroom hypothesis is not supported

The hypothesis predicted three things, all of which failed:

1. **Ratchet should help at dense NTP** (overparameterized model has excess capacity → regularization headroom even without masking). Result: OL wins at mask=0.0 at every cycle, with the gap growing. Dense NTP eliminates ratchet benefit regardless of model overparameterization.

2. **Compounding should appear at sparse mask rates** (overparameterized model has multiple equivalent solutions → self-compression selects better ones each cycle). Result: WS advantages peak at cycles 1-3 and then narrow by cycle 4. No compounding at any mask rate.

3. **L3 improvement should persist through distillation** (model has capacity for all 4 levels → can maintain L3 computation without injection). Result: L3 improvements appear at individual cycles but do not accumulate.

### What the FM cosine reveals

At L=4, the FM captures 96–99% of the model's computation (vs 90% on MNIST and 92–94% at L=6). The residual is tiny and dominated by architectural noise (1-head FM vs 6-head model). The local loss successfully pushes computation toward near-perfect FM-predictability, but the injection has little useful information to provide because the FM already captures essentially everything.

This creates a specific failure mode: the FM is too accurate to generate a useful teaching signal. The injection adds near-zero information, the gate rationally closes, and subsequent cycles have nothing to build on.

### The MNIST comparison remains unexplained

The MNIST WS_UG_uniform ratchet compounds on fixed data with the same gate mechanism. The key differences that have not been ruled out:

1. **Classification vs NTP**: MNIST has one output position (CLS, 10 classes); RHM has 16–64 output positions (v=8 each). This affects both the supervision signal quality and the distillation information content.
2. **Bidirectional vs causal FM**: MNIST FM is bidirectional; RHM FM is causal. This may affect the FM's ability to provide useful predictions.
3. **Residual structure**: MNIST residual is low-rank (18/128) and digit-discriminative; RHM residual is higher-rank and less structured.
4. **FM accuracy**: MNIST FM cosine is ~0.90 (10% gap); RHM FM cosine is 0.96–0.99 at L=4 and 0.92–0.94 at L=6. The "useful residual" regime may require FM cosine in a specific range (~0.85–0.95) where the prediction is good enough to provide structural information but imperfect enough to generate a meaningful teaching signal.

The capacity/overparameterization explanation has been ruled out by this experiment. The remaining explanations involve task structure (classification vs autoregressive NTP) or FM architecture rather than model capacity.

## Reproduction

```bash
cd experiments/
modal run --detach -m rhm.ratchet.rhm_sparse_ratchet::rhm_sparse_ratchet \
  --depth 4 --mask-rates "0.0,0.75,0.90,0.95"
```

Results saved to `rhm-scaling-data` volume at `rhm_sparse_ratchet/v8_s2_L4_m2_gated/results.json`.
