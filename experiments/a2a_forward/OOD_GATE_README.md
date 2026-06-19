# OOD Gate Experiments: Meta-Learning Under Distribution Shift (2026-06-18)

**Code**: `mnist_ood_gate.py`, `mnist_fashion_gate.py`
**Prior experiments**: [MNIST gated ratchet](GATED_RATCHET_README.md), [MNIST local loss + learning gate](MNIST_LOCAL_LOSS_README.md)
**Idea docs**: [local_prediction_error_learning.md](../../ideas/local_prediction_error_learning.md)

## Goal

Test whether the bilevel learning gate does genuine meta-learning under non-stationary data distributions. On stationary MNIST (the [4-cycle gated ratchet](GATED_RATCHET_README.md)), the gate opens monotonically (0.31 → 0.77) — more compression always helps within-distribution. On a non-stationary distribution, the gate should become selective: opening for novel inputs (where learning is needed) and closing for known inputs (where existing knowledge should be protected from over-compression).

Two experiments with increasing shift severity:
1. **Digit shift** (easy): MNIST digits 0-6 → digits 0-9
2. **MNIST → Fashion-MNIST** (hard): MNIST → MNIST + Fashion-MNIST (20-class)

## Experiment 1: Digit shift (0-6 → 0-9)

### Design

Phase 1 (cycles 1-2): Train on MNIST digits 0-6 only (classes 0-6).
Phase 2 (cycles 3-4): Train on all MNIST digits 0-9.

Three conditions, 4 cycles × (1500 wake + 600 sleep) = 8400 main-model steps:

| Condition | Description |
|---|---|
| **WS_LG_shift** | Gated ratchet with digit shift |
| **WS_LG_full** | Gated ratchet on all 10 digits throughout (stationary control) |
| **OL_shift** | Open-loop with same digit schedule (mechanism control) |

Architecture: 4L/4H/128D ViT (0.80M params), 1L/1H/32D bidirectional FM (83K params), post_block0 → post_block3, injection after block 1, 24.8K learning gate. Identical seed (42), lr (3e-4), initial weights.

### Results

#### The gate IS input-selective after the shift

**WS_LG_shift (non-stationary):**

| Phase | Overall | Known (0-6) | Novel (7-9) | Diff (novel − known) |
|---|---|---|---|---|
| C1 post-wake (phase1) | 0.447 | 0.457 | 0.424 | −0.033 |
| C2 post-wake (phase1) | 0.514 | 0.489 | 0.571 | +0.082 |
| C3 post-wake (phase2) | 0.572 | 0.520 | 0.692 | **+0.173** |
| C4 post-wake (phase2) | 0.617 | 0.576 | 0.710 | **+0.133** |

**WS_LG_full (stationary control):**

| Phase | Overall | Known (0-6) | Novel (7-9) | Diff (novel − known) |
|---|---|---|---|---|
| C1 post-wake | 0.321 | 0.377 | 0.193 | −0.184 |
| C2 post-wake | 0.573 | 0.600 | 0.510 | −0.090 |
| C3 post-wake | 0.827 | 0.824 | 0.834 | +0.011 |
| C4 post-wake | 0.857 | 0.850 | 0.871 | +0.020 |

The stationary control converges to a nearly uniform gate (diff ≈ +0.02). The shift condition produces a known-novel gap of +0.13 to +0.17 — a **6-8× amplification**. The gate discriminates between familiar and novel inputs under non-stationarity.

The selectivity is in the **opening direction, not closing** — both groups open, but novel digits open more. Known-digit gate weights continue rising (0.383 → 0.582) because compression helps known digits too. On MNIST, compression never hurts because all digits share the same visual manifold.

#### Accuracy: the gate helps with adaptation and forgetting

| Metric (Cycle 4) | WS_LG_shift | WS_LG_full | OL_shift |
|---|---|---|---|
| Known (0-6) accuracy | **98.5%** | 98.5% | 97.4% |
| Novel (7-9) accuracy | **96.5%** | 97.4% | 94.9% |
| Overall val loss | **0.067** | 0.048 | 0.109 |

WS_LG_shift learns novel digits faster (+1.6pp vs OL) and maintains known-digit accuracy better (+1.1pp vs OL).

#### Residual structure

| Cycle | WS_LG_shift eff rank | WS_LG_full eff rank | OL_shift eff rank |
|---|---|---|---|
| 2100 | 17.9 | 30.5 | 12.9 |
| 4200 | 18.8 | 27.4 | 11.0 |
| 6300 | 23.9 | 31.0 | 16.4 |
| 8400 | 24.7 | 29.0 | 17.0 |

WS_LG_shift's effective rank jumps from 18.8 to 23.9 at the distribution shift (cycle 2→3), as the FM encounters new computational patterns from novel digits.

### Reproduction

```bash
cd experiments/
modal run --detach a2a_forward/mnist_ood_gate.py::a2a_mnist_ood_gate
```

## Experiment 2: MNIST → Fashion-MNIST (hard shift)

### Design

Phase 1 (cycles 1-2): Train on MNIST only (classes 0-9).
Phase 2 (cycles 3-4): Train on MNIST + Fashion-MNIST (classes 0-19, Fashion labels offset to 10-19).

20-class ViT output head. Same 3-condition structure (WS_LG_shift, WS_LG_full, OL_shift), same architecture and compute budget. MNIST and Fashion-MNIST share the 28x28 grayscale format but require fundamentally different visual features (digit strokes vs clothing textures/silhouettes).

### Results

#### The gate trajectory is non-monotonic and selectivity converges to zero

**WS_LG_shift (non-stationary):**

| Phase | Overall | MNIST | Fashion | Diff (fashion − mnist) |
|---|---|---|---|---|
| C1 post-wake (phase1) | 0.263 | 0.180 | 0.343 | +0.163 |
| C2 post-wake (phase1) | 0.360 | 0.330 | 0.390 | +0.060 |
| C3 post-wake (phase2) | 0.614 | 0.597 | 0.631 | +0.034 |
| C4 post-wake (phase2) | 0.430 | 0.393 | 0.465 | +0.072 |

**WS_LG_full (stationary control):**

| Phase | Overall | MNIST | Fashion | Diff (fashion − mnist) |
|---|---|---|---|---|
| C1 post-wake | 0.300 | 0.162 | 0.432 | +0.270 |
| C2 post-wake | 0.476 | 0.420 | 0.529 | +0.109 |
| C3 post-wake | 0.447 | 0.418 | 0.475 | +0.057 |
| C4 post-wake | 0.356 | 0.358 | 0.354 | **−0.004** |

Three key differences from the digit-shift experiment:

1. **The gate CLOSES after peaking** (WS_LG_full: 0.30 → 0.48 → 0.45 → 0.36). On easy MNIST-only, the gate opens monotonically to 0.77. On the harder combined task, the bilevel optimization discovers that over-compression hurts and retreats.

2. **The MNIST-Fashion selectivity DISAPPEARS** (diff: +0.27 → +0.11 → +0.06 → −0.00). By cycle 4, the gate produces identical weights for MNIST and Fashion-MNIST inputs.

3. **The shift condition shows the same pattern, delayed** — the gate opens during phase 2 when Fashion is introduced (0.37 → 0.61) then partially closes in cycle 4 (0.61 → 0.43).

#### Accuracy

| Metric (Cycle 4) | WS_LG_shift | WS_LG_full | OL_shift |
|---|---|---|---|
| MNIST accuracy | 97.4% | 97.5% | 96.9% |
| Fashion-MNIST accuracy | 85.9% | 86.5% | 85.8% |
| Overall val loss | 0.223 | 0.211 | 0.233 |

The gated ratchet provides a small advantage (+0.5pp MNIST, +0.1pp Fashion vs OL). Fashion-MNIST is inherently harder than MNIST (86% vs 97%), so the 20-class combined task is significantly more challenging.

#### Residual structure: higher rank on combined task

| Cycle | WS_LG_shift eff rank | WS_LG_full eff rank |
|---|---|---|
| 2100 | 24.2 | 42.2 |
| 4200 | 25.5 | 41.2 |
| 6300 | 45.2 | 37.4 |
| 8400 | 44.2 | 39.5 |

The combined task produces higher effective rank (39-45/128) than MNIST-only (25-31/128), consistent with the greater computational complexity of the 20-class problem.

### Reproduction

```bash
cd experiments/
modal run --detach a2a_forward/mnist_fashion_gate.py::a2a_mnist_fashion_gate
```

## Interpretation: the FM's error decomposition is the gate's vocabulary

### Why the digit shift shows selectivity but the Fashion shift doesn't

The gate selects per-dimension of the FM error, and the FM's error dimensions reflect whatever computational patterns the FM learned to model. The gate can only be selective over distinctions that the FM's error structure encodes.

**Digit shift (0-6 → 0-9)**: Digits 7-9 live on the same visual manifold as 0-6. The FM's error decomposition — calibrated on digit computation — remains approximately valid for novel digits. The gate can meaningfully discriminate "known digit computation" from "novel digit computation" because the FM's error vocabulary covers both.

**MNIST → Fashion-MNIST**: The FM was trained to predict digit-specific computation (stroke curvature, loop counting, etc.). Fashion images produce FM errors that land in these digit-calibrated dimensions in meaningless ways. Three things break simultaneously:

1. The FM's error structure becomes incoherent for Fashion inputs — its error dimensions decompose digit-specific computation, not clothing computation.
2. The model's computation for Fashion inputs is initially unstructured (random projection through digit-trained weights), so the FM error is "random minus random."
3. The gate's learned policy ("which dimensions of digit-computation error are worth compressing") has no valid target for clothing computation.

Even on the stationary combined task (WS_LG_full), the FM learns a shared error space that doesn't decompose along domain boundaries. Each FM error dimension carries signal about both MNIST and Fashion-MNIST, so the gate has no domain-selective lever. The selectivity converges to zero because there are no FM error dimensions that are "MNIST-specific" or "Fashion-specific."

### The gate is a task-difficulty-sensitive compression controller

Across all experiments, the gate's primary signal is "is compression beneficial for the current task?" — a global confidence signal, not input-selective filtering:

| Task | Gate trajectory | Final selectivity |
|---|---|---|
| MNIST only (stationary) | Opens monotonically (0.31 → 0.77) | Known ≈ Novel (diff +0.02) |
| MNIST digits 0-6 → 0-9 | Opens (0.35 → 0.62) | Novel > Known (diff +0.13) |
| MNIST + Fashion (stationary) | Opens then closes (0.30 → 0.48 → 0.36) | MNIST ≈ Fashion (diff −0.00) |
| MNIST → MNIST + Fashion | Opens then closes (0.26 → 0.61 → 0.43) | Fashion slightly > MNIST (diff +0.06) |

On easy tasks (MNIST), compression is always beneficial and the gate opens aggressively. On hard tasks (MNIST + Fashion), the bilevel optimization discovers that over-compression hurts and the gate retreats. The input-selectivity (known vs novel diff) is real but small compared to the overall open/close decision.

### Computational non-stationarity vs data non-stationarity

The cerebellar system's meta-learning operates over the model's own computational structure as decomposed by the FM — not over data domains. The non-stationarity that makes meta-learning non-trivial isn't "new data from a different domain" but rather "new computational patterns the FM hasn't seen."

FM reinitialization already provides computational non-stationarity: each fresh FM decomposes the model's computation differently, creating genuinely new error dimensions. The gate policy's stability across FM reinitializations (post-wake ≈ post-repoint in the original 4-cycle experiment) is already meta-learning — a compression policy that generalizes across FM perspectives. It's just not the input-selective meta-learning we tested for here.

The prediction that the gate should close for known inputs required an implicit assumption: that the FM's error dimensions would align with the known-vs-novel distinction. On an easy within-manifold shift (digit 0-6 → 0-9), this approximately holds. On a hard cross-manifold shift (MNIST → Fashion), it doesn't.

### Connection to biological meta-learning

Human experience streams are continuous, not sudden teleportation between unrelated domains. The cerebellum's forward model tracks cortical dynamics as they evolve, maintaining a coherent error decomposition at all times. Starting from combined data (so the FM learns domain-relevant error dimensions from the beginning) would be the better analog — and WS_LG_full's convergence to zero selectivity suggests that even then, the gate optimizes globally rather than per-domain.

The selective gate-closing prediction may require a regime where different inputs demand genuinely different **computational strategies** (not just different features), AND the FM's error structure happens to decompose along those computational strategy boundaries. This is a much more specific condition than "non-stationary data," and may be the right target for future experiments.

## Files

| File | Purpose |
|---|---|
| `mnist_ood_gate.py` | Digit shift experiment (0-6 → 0-9) |
| `mnist_fashion_gate.py` | MNIST → Fashion-MNIST experiment (20-class) |

## Modal volume

Results saved to `language-reduction-data` volume:

```
a2a_forward/mnist_ood_gate/
└── vit_4L_4H_128D/post_block0_to_post_block3/
    ├── wslg_shift_model.pt, wslg_shift_fm.pt, wslg_shift_lgate.pt
    ├── wslg_full_model.pt, wslg_full_fm.pt, wslg_full_lgate.pt
    ├── ol_shift_model.pt
    └── results.json

a2a_forward/mnist_fashion_gate/
└── vit_4L_4H_128D/post_block0_to_post_block3/
    ├── wslg_shift_model.pt, wslg_shift_fm.pt, wslg_shift_lgate.pt
    ├── wslg_full_model.pt, wslg_full_fm.pt, wslg_full_lgate.pt
    ├── ol_shift_model.pt
    └── results.json
```
