# Language Gated Ratchet (2026-06-20)

**Code**: `language_ratchet.py`
**Prior experiments**: [MNIST gated ratchet](GATED_RATCHET_README.md), [MNIST unified gate decoupling](GATED_RATCHET_README.md#decoupling-test-does-selective-local-loss-matter-2026-06-19), [Controlled retrain](CONTROLLED_RETRAIN_README.md)
**Idea docs**: [cerebellar_abstraction_ratchet.md](../../ideas/cerebellar_abstraction_ratchet.md)

## Goal

Test whether the WS_UG_uniform architecture (unified gate for injection + uniform local loss + distillation ratchet) — the simplest effective architecture from the MNIST experiments — produces compounding val loss improvement on language modeling. This is the first test of the ratchet on a domain where NTP already provides dense per-position supervision, unlike MNIST where the classification loss only supervises [CLS].

## Design

Three conditions, all compute-matched on main-model gradient steps (10,000 total = 4 cycles × (2000 wake + 500 sleep)):

| Condition | Description |
|---|---|
| **WS_UG_uniform** | Unified gate (NTP-trained) + uniform local loss + distillation ratchet |
| **CL** | Continuous closed-loop (CerebellarGate injection only, no local loss, no distillation) |
| **OL** | Open-loop baseline (standard NTP training) |

Architecture: 4L/4H/256D GPT (~28.9M params), 2L/1H/64D causal FM (~660K params, 2.3% of main), post_block0 → post_block3, injection after block 1, UnifiedGate (~165K params). Identical seed (42), lr (3e-4), initial weights. 10M FineWeb-Edu tokens (τ=0.0). FM retrained for 2000 steps per cycle on frozen model activations.

## Results

### Val loss: compounding improvement, crossover at cycle 3

| Cycle | WS_UG_uniform | CL | OL | WS_UG vs OL |
|---|---|---|---|---|
| 1 (2500) | 5.457 | 5.473 | 5.377 | −0.080 |
| 2 (5000) | 5.227 | 5.259 | 5.190 | −0.037 |
| 3 (7500) | **5.181** | 5.297 | 5.214 | **+0.033** |
| 4 (10000) | **5.228** | 5.418 | 5.312 | **+0.084** |

WS_UG_uniform starts behind OL (the wake+sleep overhead costs early cycles) but crosses over at cycle 3 and the gap widens at cycle 4. CL degrades monotonically relative to OL (the dependency problem without distillation, consistent with all prior experiments).

All three models overfit on this 10M token dataset (OL val loss bottoms around step 5500 then climbs). The WS_UG model overfits 2.7× less: peak-to-final degradation is 0.047 nats (WS_UG) vs 0.129 nats (OL) vs 0.159 nats (CL).

### Gate closes aggressively

| Phase | Mean | Sparse (<0.1) |
|---|---|---|
| post_wake C1 | 0.120 | 50.8% |
| post_wake C2 | 0.072 | 83.6% |
| post_wake C3 | 0.072 | 82.0% |
| post_wake C4 | 0.060 | 89.5% |

The gate closes more aggressively than on MNIST (0.12 → 0.06, 90% sparse, vs MNIST's 0.37 → 0.13, 38% sparse). Gate weights are stable across FM reinitializations (post_wake ≈ post_repoint), replicating the MNIST finding.

### Robustness: CL most robust, WS_UG most brittle

| Cycle | WS_UG_uniform | CL | OL |
|---|---|---|---|
| 1 | +1.311 | +0.638 | +0.932 |
| 2 | +1.375 | +0.586 | +1.092 |
| 3 | +1.457 | +0.580 | +1.172 |
| 4 | +1.498 | +0.568 | +1.207 |

(Perturbation Δloss at eps=1.0.)

WS_UG is 24% more sensitive than OL. CL is 53% less sensitive. This exactly replicates the cross-domain finding from MNIST: the local loss creates perturbation sensitivity (the "be predictable" pressure concentrates computation), while injection experience without local loss produces a flatter loss landscape.

### Self-knowledge probes: no differentiation

| Cycle | WS_UG_uniform | CL | OL |
|---|---|---|---|
| 1 | 0.030 | 0.026 | 0.026 |
| 2 | 0.020 | 0.019 | 0.021 |
| 3 | 0.019 | 0.017 | 0.017 |
| 4 | 0.017 | 0.015 | 0.015 |

(R² at post_block0 for fresh FM residual prediction.)

All conditions are nearly identical and low. Consistent with the gate being nearly shut — the model receives negligible injection signal, so there is no pressure to develop representations that encode information about the FM's errors.

### Residual structure: all similar, consistent with prior language experiments

| Cycle | WS_UG eff_rank | CL eff_rank | OL eff_rank |
|---|---|---|---|
| 1 | 190.6, cos=0.982 | 191.8, cos=0.983 | 188.8, cos=0.979 |
| 2 | 209.3, cos=0.963 | 210.1, cos=0.960 | 210.4, cos=0.953 |
| 3 | 220.4, cos=0.947 | 221.7, cos=0.943 | 223.6, cos=0.932 |
| 4 | 226.4, cos=0.933 | 228.4, cos=0.927 | 229.7, cos=0.914 |

Full-rank residual throughout (~226-230/256 at final cycle), consistent with all prior language experiments. FM cosine slightly higher for WS_UG (0.933 vs 0.914 OL), expected given the local loss pushes computation toward FM-predictability.

## Discussion

### What replicated from MNIST

1. **Compounding val loss trajectory**: the WS_UG → OL gap widens across cycles (−0.080 → +0.084).
2. **Robustness dissociation**: CL produces robustness, local loss produces brittleness.
3. **Gate stability across FM reinitializations**: post_wake ≈ post_repoint at every cycle.
4. **CL dependency problem**: without distillation, CL is worse than OL standalone.

### What differs from MNIST

1. **Effect size**: 1.6% improvement vs MNIST's 48%. The ratchet adds much less on top of dense NTP supervision.
2. **Gate trajectory**: the gate starts low (0.12 vs MNIST's 0.37) and closes further (0.06 vs 0.13). 90% of dimensions below 0.1 by cycle 4 (vs 38% on MNIST). The gate correctly reports that injection utility is low when NTP already provides dense supervision.
3. **Self-knowledge**: no differentiation across conditions (vs strong differentiation on MNIST, R² = 0.63-0.79). The gate is too closed for meaningful self-knowledge to develop.
4. **Mechanism**: on MNIST, the ratchet provides genuinely new supervision (6400 intermediate dimensions vs 128 from CLS). On language, NTP already provides ~6.4M gradient dimensions per batch — the local loss's 32K intermediate dimensions are a small marginal contribution. The benefit appears to come primarily from the distillation cycles acting as implicit regularization against overfitting.

### Open questions

1. **Is this just "distillation helps with overfitting"?** On MNIST, this was already answered: the [multi-cycle comparison](MNIST_DISTILLATION_README.md#multi-cycle-comparison-with-compute-matched-baselines-2026-06-13) showed WS ≈ KD (external teacher) for val loss — distillation from *any* teacher works — while robustness requires self-referential CL co-training (KD is worse than OL). A WS condition (distillation without local loss) on language would test whether the same decomposition holds here. On MNIST, WS alone stalled while WS_UG_uniform kept compounding; if WS ≈ WS_UG_uniform on language, the local loss contributes nothing beyond distillation-as-regularization.

2. **Dataset size confound.** 10M tokens with 10K steps is deep in the overfitting regime. On 100M tokens, the regularization benefit shrinks. The question is whether the ratchet helps with genuine generalization or only with overfitting resistance.

3. **lambda_local scaling.** The local loss magnitude (~0.005) is ~1000× smaller than NTP loss (~5.0). The MNIST ratio was ~125×. Increasing lambda_local could amplify the local loss signal enough to matter.

4. **Scale dependence.** These results are on a 28.9M parameter model on 10M tokens — a severely undertrained, underparameterized regime. The cerebellar framework was motivated by reasoning about frontier-scale models where the computational bottleneck (compression of intermediate computation) is qualitatively different. Small models on small data may not develop the kind of structured intermediate computation that the ratchet is designed to compress. The gate closing here could be analogous to high ungated plasticity early in development — the "cerebellum" has nothing useful to model yet because the "cortex" hasn't developed enough structure.

## Reproduction

```bash
cd experiments/
modal run --detach a2a_forward/language_ratchet.py::a2a_language_ratchet
```

## Modal volume

Results saved to `language-reduction-data` volume:

```
a2a_forward/language_ratchet/
└── gpt_4L_4H_256D/post_block0_to_post_block3/
    ├── wsug_model.pt, wsug_fm.pt, wsug_ugate.pt
    ├── cl_model.pt, cl_fm.pt, cl_cgate.pt
    ├── ol_model.pt
    └── results.json
```
