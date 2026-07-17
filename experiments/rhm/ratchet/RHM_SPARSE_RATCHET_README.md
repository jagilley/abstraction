# RHM Sparse Ratchet: Wake-Sleep with NTP Masking (2026-06-24)

**Code**: `rhm_sparse_ratchet.py`
**Prior experiments**: [RHM sparsity sweep](RHM_SPARSITY_SWEEP_README.md), [RHM ratchet at m=4](RHM_RATCHET_README.md), [RHM ratchet at m=2](RHM_RATCHET_M2_README.md)

## Goal

Test whether the WS_UG_uniform ratchet produces per-level compositional improvement when NTP supervision is sparse. Prior ratchet experiments on RHM showed only 0.3% improvement at m=4 (dense NTP) and no crossover at m=2 (dense NTP). The [sparsity sweep](RHM_SPARSITY_SWEEP_README.md) established that FM process supervision helps when NTP is sparse enough (crossover between mask=0.75 and mask=0.90), with +1.6% at mask=0.95 and per-level improvements at L1-L2. This experiment tests whether ratchet cycles compound that effect.

## Design

All conditions share identical initial weights, batch indices, position masks, and seed (42).

- DGP: L=6, m=2, v=8, s=2 (seq_len=64)
- Model: 6L/6H/192D GPT (~2.68M params)
- FM: 1L/1H/16D causal (50.4K params, 1.9% of main model), post_block0 -> post_block5 (5-block gap, 83% of depth)
- Inject after block 1
- UnifiedGate: 74.2K params (2-layer MLP + projection, zero-initialized)
- 4 cycles x (3000 wake + 750 sleep) = 15,000 main-model gradient steps
- Repoint: 2000 steps to train a fresh FM after each distillation

**Per-cycle structure**:

1. **WAKE** (3000 steps): Masked NTP + gated injection + local loss + FM co-training. The unified gate modulates both the injection and (in the gated variant) the local loss weight. The FM trains on MSE with stop-gradient from the main model.
2. **SLEEP** (750 steps): Distillation from teacher (model + FM + gate, with injection) to student (model weights, no injection). Loss = 0.5 * KL(student || teacher) + 0.5 * masked CE. The KL term transfers teacher knowledge at all positions; the CE is masked at the same rate for consistency.
3. **REPOINT** (2000 steps): Fresh FM (new random seed) trained from scratch on the distilled model's activations.

**Conditions**: Two runs.

Run 1 (gated): {WS_UG_uniform, OL} x {mask_rate = 0.75, 0.90, 0.95}. The local loss is gate-scaled: `gw_scalar * (target - fm_pred)^2`, where `gw_scalar` is the detached mean gate weight.

Run 2 (fixed): WS only (OL reused from Run 1) x {mask_rate = 0.75, 0.90, 0.95}. The local loss is fixed: `(target - fm_pred)^2`, with no gate modulation. Tests whether the gate's attenuation of local loss across cycles is hurting or helping.

Evaluation is always unmasked (full NTP). Per-level loss is decomposed via the s-adic valuation of each position.

## Results

### Val loss trajectory (gated ratchet)

Gap = OL - WS. Positive = ratchet wins.

| mask | Cycle 1 | Cycle 2 | Cycle 3 | Cycle 4 |
|---|---|---|---|---|
| 0.75 | -0.014 (-1.6%) | -0.009 (-1.1%) | -0.005 (-0.6%) | -0.009 (-1.1%) |
| 0.90 | +0.001 (+0.1%) | +0.001 (+0.1%) | -0.006 (-0.7%) | -0.001 (-0.1%) |
| 0.95 | +0.007 (+0.7%) | -0.005 (-0.6%) | +0.003 (+0.4%) | **+0.011 (+1.2%)** |

The monotonic crossover from the sparsity sweep replicates: WS loses at mask=0.75, ties at mask=0.90, and **wins by +1.2% at mask=0.95 by cycle 4**. This is the first val loss improvement from a ratchet setup on RHM.

### Per-level delta at mask=0.95 (gated, WS - OL; negative = WS wins)

| Cycle | Δval | ΔL0 | ΔL1 | ΔL2 | ΔL3 | ΔL4 | ΔL5 |
|---|---|---|---|---|---|---|---|
| 1 | -0.007 | -0.005 | +0.025 | +0.021 | **-0.057** | **-0.066** | **-0.067** |
| 2 | +0.005 | +0.000 | -0.002 | +0.056 | -0.039 | +0.034 | -0.038 |
| 3 | -0.003 | -0.005 | -0.003 | +0.006 | +0.005 | -0.027 | -0.006 |
| 4 | **-0.011** | **-0.015** | **-0.022** | **-0.014** | +0.011 | -0.006 | -0.008 |

Two distinct patterns emerge:

**Cycle 1: dramatic improvement at L3-L5.** The ratchet improves the barely-learned compositional levels (L3-L5, all near the uniform baseline of ln(8)=2.08) by 0.057-0.067 nats. OL's L5 loss is 1.986 (gap to uniform = 0.093); the ratchet closes 72% of that gap in a single wake-sleep cycle. This improvement survives distillation — it's measured on the student model without injection.

**Cycle 4: the advantage migrates to L0-L2.** By cycle 4, the L3-L5 advantage has dissipated while L0-L2 (the levels the model is actively learning) show consistent improvement. The cycle 4 overall val loss gain (-0.011) comes from L0-L2, not from compounding the L3-L5 gains.

### Gated vs fixed local loss at mask=0.95

| Cycle | WS_gated | WS_fixed | OL |
|---|---|---|---|
| 1 | 1.032 (-0.7%) | **1.012 (-2.6%)** | 1.039 |
| 2 | 0.944 (+0.6%) | 0.949 (+1.2%) | 0.938 |
| 3 | 0.894 (-0.4%) | 0.903 (+0.7%) | 0.897 |
| 4 | **0.872 (-1.2%)** | 0.886 (+0.5%) | 0.882 |

The fixed local loss produces a **much larger cycle 1 advantage** (-2.6% vs -0.7%), with even stronger per-level improvements at L3-L5 (ΔL5 = -0.085 vs -0.067). But it regresses in subsequent cycles and ends up worse than OL by cycle 4 (+0.5%).

Per-level at cycle 1 (both minus OL):

| | ΔL0 | ΔL1 | ΔL2 | ΔL3 | ΔL4 | ΔL5 |
|---|---|---|---|---|---|---|
| gated | -0.005 | +0.025 | +0.021 | -0.057 | -0.066 | -0.067 |
| fixed | **-0.026** | -0.006 | -0.000 | -0.058 | **-0.073** | **-0.085** |

**Why fixed hurts at later cycles**: The unmodulated local loss drives the model's computation to become nearly perfectly FM-predictable (FM cosine 0.990 vs 0.977 for gated), compressing the residual to norm 1.42 (vs 3.91). This eliminates the injection's value — the gate closes to 0.062, the dependency gap drops to near zero (+0.000 at cycles 2-3), and subsequent cycles have no teacher advantage to distill. The gate's modulation of local loss in the gated condition prevents this over-compression and preserves useful injection dynamics across cycles.

### Gate dynamics

The gate closes across cycles at all mask rates, more aggressively at higher masking:

| mask | c1 mean (sparse) | c4 mean (sparse) |
|---|---|---|
| 0.75 | 0.214 (12%) | 0.115 (58%) |
| 0.90 | 0.152 (40%) | 0.082 (84%) |
| 0.95 | 0.161 (35%) | 0.071 (87%) |

Gate weights are stable across FM reinitializations (post-wake ≈ post-repoint), confirming that the gate's policy generalizes across FM perspectives. The gate closing is driven by distillation absorbing the injection's contribution, not by FM quality.

### FM residual and robustness (final checkpoint)

| Condition | cos | norm | rank% | rob (eps=1.0) |
|---|---|---|---|---|
| WS_gated m0.95 | 0.977 | 3.91 | 67.4 | +0.062 |
| WS_fixed m0.95 | 0.990 | 1.42 | 69.8 | +0.156 |
| OL m0.95 | 0.939 | 39.5 | 43.0 | +0.005 |

The local loss (both variants) compresses residual norms dramatically (3.9-1.4 vs 39.5 for OL) and increases FM cosine (0.977-0.990 vs 0.939). WS models are more brittle than OL (12-32x), replicating the cross-domain finding that process supervision hurts robustness.

## Key findings

### 1. First val loss improvement from a ratchet on RHM

At mask=0.95, the gated ratchet achieves +1.2% val loss improvement over OL by cycle 4. Prior ratchet experiments on RHM with dense NTP produced only +0.3% (m=4) or negative (m=2). The sparsity sweep correctly predicted that sparse NTP would unlock the ratchet's benefit.

### 2. Single-cycle abstraction-level improvement at L3-L5

The most striking result is cycle 1 at mask=0.95: both gated and fixed conditions show large improvements at L3-L5, the compositional levels barely learned by OL. The fixed condition's cycle 1 L5 improvement (-0.085 nats) closes 91% of OL's gap to uniform at that level. These improvements survive distillation — they are measured on the student model without injection.

This is the first demonstration of process supervision improving learning at higher compositional levels of the RHM hierarchy. The mechanism: at 95% masking, L3-L5 positions receive almost no NTP gradient (they occur at positions where s-adic valuation >= 3, which are rare). The FM's intermediate supervision fills this gap during wake, and distillation absorbs the resulting representation into the student.

### 3. The improvement does not compound across cycles

Both gated and fixed conditions show the L3-L5 advantage dissipating by cycles 2-4. The gated ratchet's cycle 4 advantage (-1.2%) comes from gradual L0-L2 improvement, not from building on the L3-L5 gains. The fixed condition, despite a larger cycle 1 advantage, regresses past OL by cycle 4.

The likely mechanism: distillation absorbs the injection's contribution, and the fresh FM on the distilled model discovers a different error landscape. The distilled model has already internalized the L3-L5 composition from cycle 1, so the new FM's errors are distributed differently. Subsequent cycles refine different aspects of the computation rather than building deeper.

### 4. Gate modulation of local loss is protective, not harmful

The initial hypothesis was that the gate's self-attenuation of local loss (from 0.161 to 0.071 across cycles) was limiting the ratchet's effectiveness. The fixed local loss experiment disproved this: unmodulated local loss drives over-compression (FM cosine 0.990, residual norm 1.42), which eliminates the injection's value and prevents compounding. The gate's attenuation preserves enough residual structure for the injection to remain useful across cycles.

## Relation to prior results

| Experiment | Setup | RHM val loss gap |
|---|---|---|
| RHM ratchet m=4, dense NTP | WS_UG_uniform, tau=1.0 | +0.3% |
| RHM ratchet m=2, dense NTP | WS_UG_uniform, tau=1.0 | -0.3% (OL wins) |
| Sparsity sweep m=2, mask=0.95 | LL only (no injection/distillation) | +1.6% |
| **Sparse ratchet m=2, mask=0.95** | **WS_UG_uniform (gated)** | **+1.2%** |
| **Sparse ratchet m=2, mask=0.95** | **WS_UG_uniform (fixed LL)** | **-2.6% (c1), +0.5% (c4)** |

The sparse ratchet's +1.2% is smaller than the sparsity sweep's +1.6% (LL only), which at first seems surprising. But the ratchet's val loss is measured on the distilled student (no injection), while the LL condition trains continuously with local loss throughout. The ratchet pays a cost for the distillation cycles (wake-sleep phase transitions, FM reinitialization). The LL condition's +1.6% also had the benefit of continuous local loss for all 15K steps, while the ratchet's wake phase only applies local loss for 12K of the 15K steps (the remaining 3K are distillation).

The key unique contribution of the ratchet is the **per-level pattern**: the cycle 1 L3-L5 improvement (-0.057 to -0.085 nats) that the LL-only condition did not produce. The sparsity sweep's LL at mask=0.95 improved L1 by -0.025 and L2 by -0.022, but L3-L5 were near zero. The ratchet's injection during wake provides a structural preview of higher-level computation that pure local loss cannot.

## Open directions

1. **Single-cycle focus**: The cycle 1 L3-L5 improvement is the cleanest finding. A dedicated experiment comparing single-cycle wake-sleep (varying wake and sleep durations, mask rates, and FM capacity) against matched OL and LL conditions would isolate what's load-bearing. Is it the injection during wake, the distillation, or both? A sleep-only condition (distill from a pre-trained teacher without wake) would test this.

2. **Warm-starting the FM**: The current design reinitializes the FM from scratch at each repoint. Warm-starting from the prior FM (which already has useful priors about the model's computation) might produce better cycle-2+ dynamics, since the FM wouldn't have to relearn from zero. The counterargument is that the distilled model's activations have changed enough that the old FM is misleading.

3. **Non-stationary data**: On a fixed dataset, the ratchet exhausts the compositional improvement available from cycle 1's injection. A continual learning setting where the data distribution shifts between cycles would sustain the need for fresh compositional learning at each cycle. The RHM's controllable DGP enables this cleanly — e.g., shift to new composition rules at the same (L, m) between cycles.

4. **Scaling model capacity**: The 2.68M model at m=2 learns 3-4 hierarchy levels. A larger model that learns 4-5 levels would provide more headroom for higher-level improvement and might sustain the L3-L5 advantage across multiple cycles rather than exhausting it in cycle 1.

## Reproduction

```bash
cd experiments/

# Run 1: gated local loss (WS + OL at 3 mask rates)
modal run --detach -m rhm.ratchet.rhm_sparse_ratchet::rhm_sparse_ratchet

# Run 2: fixed local loss (WS only, reuses OL from Run 1)
modal run --detach -m rhm.ratchet.rhm_sparse_ratchet::rhm_sparse_ratchet \
  --fixed-local-loss --skip-ol
```

Results saved to `rhm-scaling-data` volume at:
- `rhm_sparse_ratchet/v8_s2_L6_m2/results.json` (Run 1 — gated, saved before the `_tag` suffix was added)
- `rhm_sparse_ratchet/v8_s2_L6_m2_fixed/results.json` (Run 2)
