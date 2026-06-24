# RHM Unified Gate Ratchet with Confidence Thresholding (2026-06-23)

**Code**: `rhm_ratchet.py`
**Prior experiments**: [MNIST gated ratchet](../a2a_forward/GATED_RATCHET_README.md), [Language ratchet](../a2a_forward/LANGUAGE_RATCHET_README.md), [Confidence threshold sweep](LOSS_WEIGHTING_README.md)

## Goal

Test two questions simultaneously:

1. **Does the WS_UG_uniform ratchet replicate on RHM?** This is the first ratchet experiment on a domain with a known DGP, bridging MNIST (48% improvement) and language (1.6% improvement).

2. **Does suppressing m-sharpening amplify the meta-learning signature?** The confidence threshold experiments showed that gradient reallocation away from easy positions (tau=0.3) preserves 100% of L2-L3 compositional learning while improving FM legibility. If m-sharpening overhead drowns out the meta-learning signal, thresholding inside the ratchet should amplify gate dynamics and widen the val loss gap.

## Design

Six conditions: {WS_UG_uniform, OL} x {tau=1.0, tau=0.5, tau=0.3}.

All conditions share identical initial weights, batch indices, and seed (42). The only differences are the training procedure (ratchet vs baseline) and the NTP loss function (standard CE vs confidence-thresholded). Evaluation always uses standard CE (no threshold) for fair comparison.

Architecture: 6L/6H/192D GPT (~2.68M params), L=6/m=4/v=8/s=2 (seq_len=64), 20M tokens. 4 cycles x (3000 wake + 750 sleep) = 15,000 main-model gradient steps per condition.

### Run 1: Large FM (confounded)

FM: 2L/1H/24D causal (335K params, **12.5%** of main model), post_block0 -> post_block3 (3-block gap, 50% of model depth).

This FM was too large relative to the main model. For comparison, the language ratchet used 660K params = 2.3% of the 28.9M GPT. The oversized FM achieved cosine 0.99+ with the local loss, leaving only architectural mismatch noise in the residual. The gate stayed open (~0.35, <1% sparsity) — opposite to language's aggressive closing (0.06, 90% sparsity).

### Run 2: Capacity-matched FM (main results)

FM: 1L/1H/16D causal (50.4K params, **1.9%** of main model), post_block0 -> post_block5 (5-block gap, 83% of model depth).

This matches the language ratchet's capacity ratio (2.3%) and proportional prediction gap (75%). FM cosine dropped to 0.96, the residual became meaningful, and the gate dynamics matched language.

## Results (Run 2 — capacity-matched FM)

### Val loss: small improvement, crossover at cycle 3

| Cycle | WS_UG tau=1.0 | OL tau=1.0 | gap | WS_UG tau=0.3 | OL tau=0.3 | gap |
|---|---|---|---|---|---|---|
| 1 | 1.4201 | 1.4141 | -0.006 | 1.5473 | 1.5616 | **+0.014** |
| 2 | 1.3855 | 1.3825 | -0.003 | 1.5190 | 1.5279 | +0.009 |
| 3 | 1.3736 | 1.3745 | +0.001 | 1.5188 | 1.5260 | +0.007 |
| 4 | 1.3677 | 1.3722 | **+0.004** | 1.5124 | 1.5134 | +0.001 |

(Gap = OL - WS_UG. Positive = ratchet wins.)

At tau=1.0, the ratchet starts behind (-0.006) and crosses over at cycle 3 — the same timing as the language ratchet. Final gap is +0.004 nats (0.3% relative). At tau=0.3, the ratchet starts ahead from cycle 1 (+0.014, the largest individual gap) but doesn't compound — the gap shrinks to +0.001 by cycle 4. The tau=0.5 condition tracks tau=1.0 closely (final gap -0.001).

### Gate dynamics: closes like language, threshold keeps it more open

| Cycle | tau=1.0 mean (sparse<0.1) | tau=0.5 mean (sparse) | tau=0.3 mean (sparse) |
|---|---|---|---|
| 1 | 0.188 (15.6%) | 0.199 (9.4%) | 0.220 (5.7%) |
| 2 | 0.100 (64.6%) | 0.108 (54.2%) | 0.159 (20.8%) |
| 3 | 0.081 (73.4%) | 0.087 (66.1%) | 0.160 (25.5%) |
| 4 | **0.069 (78.6%)** | 0.075 (70.8%) | **0.124 (46.9%)** |

At tau=1.0, the gate closes to 0.069 with 79% sparsity — closely matching language (0.060, 90% sparsity). **At tau=0.3, the gate stays 1.8x more open** (0.124 vs 0.069) **with 1.7x less sparsity** (47% vs 79%). The gate finds more dimensions worth injecting when m-sharpening is suppressed.

This is the first threshold-dependent gate dynamic observed on a dense-NTP domain. The gate's per-dimension selectivity changes with the NTP loss function, even though the FM and injection architecture are identical.

### Per-level loss: no compositional depth improvement

| Condition | val | L0 | L1 | L2 | L3 | L4 | L5 |
|---|---|---|---|---|---|---|---|
| WS_UG tau=1.0 | 1.3677 | 0.896 | 1.731 | 1.902 | 1.940 | 1.934 | 1.925 |
| OL tau=1.0 | 1.3722 | 0.899 | 1.740 | 1.898 | 1.936 | 1.934 | 1.923 |
| WS_UG tau=0.3 | 1.5124 | 1.184 | 1.751 | 1.899 | 1.940 | 1.936 | 1.927 |
| OL tau=0.3 | 1.5134 | 1.183 | 1.753 | 1.899 | 1.936 | 1.934 | 1.923 |

WS_UG and OL are essentially identical at every hierarchy level. The ratchet does not help with deeper compositional learning (L2+) in any threshold condition. The threshold effect on L0 (0.90 at tau=1.0 vs 1.18 at tau=0.3) applies equally to both WS_UG and OL — it's a property of the loss function, not the ratchet.

### FM residual stats

| Condition | cos | norm | rank% | top1% | fL4* |
|---|---|---|---|---|---|
| WS_UG tau=1.0 | 0.961 | 2.38 | 78.4 | 12.2 | 0.103 |
| OL tau=1.0 | 0.948 | 31.09 | 48.2 | 29.7 | 0.101 |
| WS_UG tau=0.3 | 0.976 | 1.72 | 75.6 | 15.5 | 0.114 |
| OL tau=0.3 | 0.957 | 24.17 | 51.6 | 36.9 | 0.113 |

The local loss compresses residual norms ~13x (WS_UG 2.4 vs OL 31 at tau=1.0). WS_UG FM cosine is higher (0.96 vs 0.95) because the local loss pushes computation toward FM-predictability. fL4* is slightly higher at tau=0.3 (0.114 vs 0.103), consistent with the confidence threshold experiments showing more hierarchy-conditioned FM error structure under thresholding.

### Robustness: local-loss-induced brittleness

| Condition | eps=0.5 | eps=1.0 | eps=2.0 |
|---|---|---|---|
| WS_UG tau=1.0 | +0.055 | **+0.329** | +0.506 |
| OL tau=1.0 | +0.004 | +0.035 | +0.168 |
| WS_UG tau=0.3 | +0.063 | +0.243 | +0.489 |
| OL tau=0.3 | +0.003 | +0.022 | +0.138 |

WS_UG is 9-11x more sensitive than OL, replicating the local-loss-induced brittleness from MNIST and language. Slightly less brittle at tau=0.3.

## Run 1 results (large FM — for reference)

The oversized FM (335K, 12.5%) produced qualitatively different dynamics:

| Metric | Run 1 (12.5% FM) | Run 2 (1.9% FM) |
|---|---|---|
| WS_UG FM cos | 0.992 | 0.961 |
| Gate mean (C4, tau=1.0) | 0.380 | 0.069 |
| Gate sparsity (C4, tau=1.0) | 0.5% | 78.6% |
| Val gap (C4, tau=1.0) | +0.002 | +0.004 |

The open gate in Run 1 was an artifact of the FM capturing nearly all computation (cos 0.99+). The small residual was dominated by architectural mismatch noise (1-head FM vs 6-head model), leaving nothing for the gate to be selective about. The capacity-matched FM in Run 2 recovered the gate-closing dynamics seen on language.

## Cross-domain comparison

All three ratchet experiments use the same WS_UG_uniform architecture (unified gate + uniform local loss + distillation ratchet) on stationary data (no domain shift).

### Val loss gap (WS_UG vs OL)

| Cycle | MNIST | Language | RHM (tau=1.0) |
|---|---|---|---|
| 1 | **+40.7%** | -1.5% | -0.4% |
| 2 | +39.6% | -0.7% | -0.2% |
| 3 | +54.6% | +0.6% | +0.01% |
| 4 | +52.5% | **+1.6%** | **+0.3%** |

MNIST benefits from cycle 1. Language and RHM start behind (wake+sleep overhead) and slowly cross over at cycle 3.

### Gate dynamics (final cycle)

| Domain | Gate mean | Sparsity (<0.1) | Gate trajectory |
|---|---|---|---|
| MNIST | 0.132 | 37.5% | closes gradually (0.37 -> 0.13) |
| Language | 0.060 | 89.5% | closes immediately (0.12 -> 0.06) |
| RHM | 0.069 | 78.6% | closes immediately (0.19 -> 0.07) |

Language and RHM gate dynamics are nearly identical — both close aggressively from cycle 1. MNIST is qualitatively different: the gate starts open and closes gradually, staying useful longer.

### FM and residual structure

| Domain | FM cos (WS_UG) | FM cos (OL) | Residual rank |
|---|---|---|---|
| MNIST | ~0.99 | ~0.95 | 14% (low-rank, class-discriminative) |
| RHM | 0.96 | 0.95 | 78% (intermediate) |
| Language | 0.93 | 0.91 | 88% (full-rank, diffuse) |

### Robustness (WS_UG / OL ratio at eps=1.0)

| Domain | WS_UG | OL | Ratio |
|---|---|---|---|
| MNIST | +0.064 | +0.003 | 21x |
| RHM | +0.329 | +0.035 | 9.4x |
| Language | +1.498 | +1.207 | 1.2x |

Local-loss-induced brittleness is universal but varies in magnitude.

## Key findings

### 1. The ratchet replicates on RHM in dynamics but not in magnitude

The gate closing, FM tracking, robustness dissociation, and crossover timing all replicate from language. The meta-learning *dynamics* are domain-general. But the *learning benefit* is tiny (0.3% vs language's 1.6% vs MNIST's 52%). The ratchet's val loss improvement appears to scale with how much new information the local loss provides beyond what the task loss already gives.

### 2. Confidence thresholding keeps the gate more open

At tau=0.3, the gate is 1.8x more open (0.124 vs 0.069) with 1.7x less sparsity (47% vs 79%). The gate finds the injection more useful when m-sharpening is suppressed. This is the first observation of threshold-dependent gate selectivity on a dense-NTP domain. However, the more-open gate does not translate into a larger val loss gap.

### 3. FM capacity must be matched for meaningful dynamics

The 12.5% FM (Run 1) produced cosine 0.99+, open gates (~0.35), and no gate-closing dynamics. The 1.9% FM (Run 2) produced cosine 0.96, aggressive gate closing (0.07, 79% sparse), and dynamics matching language. FM capacity relative to the main model is a critical experimental parameter; when the FM is too large, it captures everything and the meta-learning dynamics are masked by ceiling effects.

### 4. m-sharpening is not the bottleneck for the ratchet

Suppressing m-sharpening (tau=0.3) changes the gate dynamics but does not amplify the val loss gap. The ratchet's effectiveness on dense-NTP domains is limited by the redundancy of the local loss with NTP supervision, not by m-sharpening overhead in the residual.

### 5. No evidence of compositional depth increase across ratchet cycles

Per-level loss decomposition at each cycle checkpoint shows WS_UG and OL learn essentially the same levels at the same rate. At the final cycle, WS_UG retains 97-103% of OL's learning at every hierarchy level — no differential toward higher levels.

**Per-level delta (WS_UG minus OL; negative = ratchet learned more):**

| Cycle | ΔL0 | ΔL1 | ΔL2 | ΔL3 | ΔL4 | ΔL5 |
|---|---|---|---|---|---|---|
| 1 (tau=1.0) | +0.016 | +0.001 | +0.008 | +0.010 | +0.009 | +0.011 |
| 4 (tau=1.0) | -0.003 | -0.009 | +0.004 | +0.004 | +0.000 | +0.002 |
| 1 (tau=0.3) | -0.023 | -0.004 | +0.011 | +0.002 | +0.004 | +0.004 |
| 4 (tau=0.3) | +0.001 | -0.002 | +0.000 | +0.004 | +0.002 | +0.004 |

Where the ratchet helps at all (middle cycles), the gains are at L0-L1 — the easy, already-learned levels. L2-L5 are all near the uniform baseline (~1.90-1.94 vs ln(8)=2.08), meaning the model barely learns at those levels regardless of training procedure. The ratchet cycles do not build compositionally toward higher hierarchy levels.

This is a significant result to disambiguate for the abstraction ratchet hypothesis in language-like domains. At least two possible explanations:

1. **Capacity floor**: The 2.7M model at m=4 can only compose 1-2 levels. There's no compositional headroom for the ratchet to exploit. A larger model that genuinely learns levels 2-3 would be a better testbed. However, this raises the question of whether the ratchet can create compositional depth that the model couldn't achieve on its own, or whether it can only regularize depth the model was already going to learn.

2. **Dense NTP masks the effect**: On MNIST, the ratchet provides qualitatively new intermediate supervision (classification only supervises [CLS]). On RHM, NTP already supervises all 63 positions — every hierarchy level gets direct gradient. The ratchet's local loss is redundant with what NTP already provides at each level. The ratchet might only drive compositional deepening in regimes where the task loss has a supervision bottleneck that the local loss can fill.

The RHM's controllable DGP could distinguish these explanations: a lower-m setting (e.g., m=2) where the model can learn 3-4 levels would test explanation 1, while keeping the dense NTP structure that tests explanation 2. If the ratchet helps deepen composition at m=2 but not m=4, the bottleneck is capacity, not supervision density.

3. It's almost certainly not *just* 'intermediate supervision' on MNIST: the WS_UG_uniform condition **as well as direct, bilevel meta-learning** showed significantly lower validation loss than an OL one on MNIST. There maybe be 'something about the MNIST DGP' that makes inputs well-structured by construction, such that deviations from that structure are easier to parse in a structured way.

## Reproduction

```bash
cd experiments/

# Run 2 (capacity-matched FM, main results)
modal run --detach -m rhm.rhm_ratchet::rhm_ratchet_sweep \
  --fwd-n-layer 1 --fwd-d-head 16 --fwd-mlp-mult 0.5 \
  --predict-to post_block5

# Run 1 (oversized FM, for reference)
modal run --detach -m rhm.rhm_ratchet::rhm_ratchet_sweep
```

Results saved to `rhm-scaling-data` volume at `/data/rhm_ratchet/results.json`.

## Next steps

1. **Lower-m ratchet (m=2)**: The m=4 model is capacity-saturated at 1-2 hierarchy levels. At m=2, the same 2.7M model learns 3-4 levels (per-level loss shows L0-L2 well below baseline at m=2). This tests whether the ratchet can deepen composition when there's actual compositional headroom — and would distinguish the capacity-floor explanation from the dense-NTP explanation for the null compositional result.

2. **Domain shift on RHM**: The stationary-data ratchet shows dynamics without magnitude. The RHM's controllable DGP enables a clean domain shift test — e.g., train on one rule set then shift to new rules at the same (L, m). This would test whether the ratchet's meta-learning dynamics produce genuine adaptation advantages, as seen in the MNIST OOD experiments.
