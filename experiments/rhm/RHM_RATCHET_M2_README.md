# RHM Ratchet at m=2 (2026-06-24)

**Code**: `rhm_ratchet.py`
**Prior experiment**: [RHM ratchet at m=4](RHM_RATCHET_README.md)

## Goal

Test whether the ratchet produces compositional depth improvement when the model has genuine compositional headroom. The m=4 ratchet showed no per-level improvement, but the model at m=4 could only compose 1-2 hierarchy levels. At m=2, the same architecture learns 3-4 levels (per-level loss shows L0-L2 well below uniform baseline), providing headroom for the ratchet to potentially improve deeper composition.

Two runs:

1. **λ=1.0** (local loss ON): Same WS_UG_uniform recipe as the m=4 ratchet, but with m=2.
2. **λ=0.0** (local loss OFF): Injection + distillation only, no local MSE loss. Tests whether the local loss is harmful.

## Design

All conditions share identical initial weights, batch indices, and seed (42).

- DGP: L=6, m=2, v=8, s=2 (seq_len=64)
- Model: 6L/6H/192D GPT (~2.68M params)
- FM: 1L/1H/16D causal (50.4K params, 1.9% of main model), post_block0 -> post_block5 (5-block gap, 83% of depth)
- Inject after block 1
- 4 cycles x (3000 wake + 750 sleep) = 15,000 main-model gradient steps
- Unified gate (74.2K params)
- Thresholds: tau=1.0 only (standard CE)

Each run produces a WS_UG condition and a compute-matched OL baseline (15,000 steps of standard NTP). The OL baseline is identical across both runs (same seed, same data order).

## Results

### Val loss trajectory

Gap = OL - WS_UG. Positive = ratchet wins.

| Cycle | WS_UG λ=1.0 | WS_UG λ=0.0 | OL | Gap (λ=1.0) | Gap (λ=0.0) |
|---|---|---|---|---|---|
| 1 | 0.8202 | 0.8194 | 0.8138 | -0.006 | -0.006 |
| 2 | 0.8021 | 0.8011 | 0.7977 | -0.004 | -0.003 |
| 3 | 0.7982 | 0.7960 | 0.7940 | -0.004 | -0.002 |
| 4 | 0.7936 | 0.7911 | 0.7904 | -0.003 | -0.001 |

Neither condition crosses over. OL wins at every cycle. The gap narrows in both cases. λ=0.0 is slightly closer to OL than λ=1.0.

### Gate dynamics

| Cycle | λ=1.0 mean (sparse<0.1) | λ=0.0 mean (sparse<0.1) |
|---|---|---|
| 1 | 0.265 (0.5%) | 0.088 (70.3%) |
| 2 | 0.194 (26.0%) | 0.038 (90.6%) |
| 3 | 0.156 (39.6%) | 0.021 (96.9%) |
| 4 | 0.145 (39.6%) | 0.017 (97.4%) |

Without local loss, the gate closes to 0.017 mean with 97% sparsity by cycle 4. With local loss, the gate stays at 0.145 with 40% sparsity.

### Per-level loss at each cycle checkpoint

#### WS_UG λ=1.0

| Cycle | val | L0 | L1 | L2 | L3 | L4 | L5 |
|---|---|---|---|---|---|---|---|
| 1 | 0.8202 | 0.352 | 1.085 | 1.139 | 1.736 | 1.700 | 1.853 |
| 2 | 0.8021 | 0.337 | 1.065 | 1.111 | 1.689 | 1.697 | 1.843 |
| 3 | 0.7982 | 0.331 | 1.058 | 1.101 | 1.669 | 1.698 | 1.834 |
| 4 | 0.7936 | 0.328 | 1.052 | 1.093 | 1.668 | 1.684 | 1.837 |

#### WS_UG λ=0.0

| Cycle | val | L0 | L1 | L2 | L3 | L4 | L5 |
|---|---|---|---|---|---|---|---|
| 1 | 0.8194 | 0.354 | 1.091 | 1.134 | 1.730 | 1.705 | 1.843 |
| 2 | 0.8011 | 0.331 | 1.057 | 1.098 | 1.683 | 1.691 | 1.838 |
| 3 | 0.7960 | 0.328 | 1.058 | 1.090 | 1.674 | 1.684 | 1.830 |
| 4 | 0.7911 | 0.327 | 1.053 | 1.085 | 1.672 | 1.676 | 1.834 |

#### OL

| Cycle-eq | val | L0 | L1 | L2 | L3 | L4 | L5 |
|---|---|---|---|---|---|---|---|
| 1 | 0.8138 | 0.345 | 1.089 | 1.122 | 1.714 | 1.711 | 1.847 |
| 2 | 0.7977 | 0.332 | 1.058 | 1.100 | 1.682 | 1.689 | 1.837 |
| 3 | 0.7940 | 0.328 | 1.058 | 1.094 | 1.668 | 1.687 | 1.828 |
| 4 | 0.7904 | 0.325 | 1.051 | 1.090 | 1.667 | 1.675 | 1.834 |

#### Per-level delta at final checkpoint (WS_UG minus OL; negative = ratchet learned more)

| Level | Δ (λ=1.0) | Δ (λ=0.0) |
|---|---|---|
| L0 | +0.004 | +0.002 |
| L1 | +0.001 | +0.002 |
| L2 | +0.003 | -0.005 |
| L3 | +0.000 | +0.005 |
| L4 | +0.010 | +0.001 |
| L5 | +0.003 | +0.000 |

All deltas are within ±0.01 nats.

### FM residual stats at final checkpoint

| Metric | WS_UG λ=1.0 | WS_UG λ=0.0 | OL |
|---|---|---|---|
| FM cosine | 0.937 | 0.907 | 0.908 |
| Residual norm | 2.01 | 25.24 | 27.91 |
| Eff rank % | 78.1 | 55.1 | 54.6 |
| Top-1 PC % | 16.6 | 22.9 | 26.4 |
| fL4* | 0.454 | 0.443 | 0.414 |

### Robustness

| Condition | eps=0.5 | eps=1.0 | eps=2.0 |
|---|---|---|---|
| WS_UG λ=1.0 | +0.205 | +0.648 | +0.979 |
| WS_UG λ=0.0 | +0.019 | +0.204 | +0.659 |
| OL | +0.017 | +0.189 | +0.612 |

### Self-knowledge probes (R^2)

| Layer | WS_UG λ=1.0 | WS_UG λ=0.0 | OL |
|---|---|---|---|
| post_block0 | 0.030 | 0.014 | 0.011 |
| post_block5 | 0.469 | 0.557 | 0.538 |

### Cross-domain comparison (final cycle, tau=1.0)

For reference, the same metrics from the m=4 ratchet and the MNIST/language ratchets (all WS_UG_uniform or equivalent).

#### Val loss gap (WS_UG vs OL)

| Cycle | MNIST | Language | RHM m=4 | RHM m=2 λ=1.0 | RHM m=2 λ=0.0 |
|---|---|---|---|---|---|
| 1 | +40.7% | -1.5% | -0.4% | -0.8% | -0.7% |
| 4 | +52.5% | +1.6% | +0.3% | -0.4% | -0.1% |

#### Gate dynamics (final cycle)

| Condition | Gate mean | Sparsity (<0.1) |
|---|---|---|
| MNIST | 0.132 | 37.5% |
| Language | 0.060 | 89.5% |
| RHM m=4 | 0.069 | 78.6% |
| RHM m=2 λ=1.0 | 0.145 | 39.6% |
| RHM m=2 λ=0.0 | 0.017 | 97.4% |

## Reproduction

```bash
cd experiments/

# λ=1.0 (local loss ON)
modal run --detach -m rhm.rhm_ratchet::rhm_ratchet_sweep \
  --m 2 --thresholds 1.0 \
  --fwd-n-layer 1 --fwd-d-head 16 --fwd-mlp-mult 0.5 \
  --predict-to post_block5

# λ=0.0 (local loss OFF)
modal run --detach -m rhm.rhm_ratchet::rhm_ratchet_sweep \
  --m 2 --thresholds 1.0 --lambda-local 0.0 \
  --fwd-n-layer 1 --fwd-d-head 16 --fwd-mlp-mult 0.5 \
  --predict-to post_block5
```

Results saved to `rhm-scaling-data` volume at `/data/rhm_ratchet/v8_s2_L6_m2/results.json`.

Note: the two runs overwrite the same results.json. The λ=1.0 run was executed first; the λ=0.0 run overwrote it. The WS_UG columns in both runs are from separate training runs. The OL columns are identical (same seed/data/architecture, re-run independently each time as a sanity check -- final val losses match to 4 decimal places: 0.7904).

## Follow-up: supervision density as causal variable (2026-06-24)

The null compositional result here was explained by the [sparsity sweep](RHM_SPARSITY_SWEEP_README.md): dense NTP makes the FM's process supervision redundant. When NTP is masked at 95% of positions, the FM's local loss produces +1.6% val loss improvement and clear per-level compositional gains (ΔL1=−0.025, ΔL2=−0.022 nats). The ratchet's failure to improve compositional depth was specific to the dense-NTP regime, not a fundamental limitation of the architecture.
