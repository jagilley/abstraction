# MNIST Local Prediction-Error Learning

**Code**: `mnist_local_loss.py` (training), `mnist_local_loss_probes.py` (representation probes)
**Date**: 2026-06-15
**Idea doc**: [ideas/local_prediction_error_learning.md](../../ideas/local_prediction_error_learning.md)
**Prior**: [MNIST experiment](MNIST_README.md), [Baseline battery](BASELINE_BATTERY_README.md)

## Goal

Test the simplest version of the local prediction-error learning idea: use the forward model's prediction errors as an auxiliary loss for the main model's intermediate layers. The local loss pushes the main model toward FM-predictable (compressible) computation — "be predictable" pressure — providing dense supervision at all positions, not just [CLS].

The local loss: `L_main = L_classification + λ · MSE(sg(FM(post_block0)), post_block3)`, where sg = stop-gradient through the FM. The FM provides a frozen target; gradient flows through post_block3 into the main model's blocks 0–3.

## Architecture

Same as the prior MNIST experiment:
- **Main model**: 4-layer, 4-head, 128-dim ViT with 4×4 patches (50 positions), 0.80M params
- **Forward model**: 1-layer transformer, 1 head, 32-dim (bidirectional), 83K params
- **Prediction**: post_block0 → post_block3 (3-layer gap)
- **Injection** (CL conditions): after block 1

## Experiment design

Four conditions with identical seed (42), learning rate (3e-4), initial weights, and batch order:

| Condition | Injection | Local loss (λ=1.0) | What it tests |
|---|---|---|---|
| OL | no | no | Baseline |
| CL | yes | no | Current system (replication) |
| LL | no | yes | Does the local loss alone help? |
| CL_LL | yes | yes | Does continuous internalization reduce dependency? |

**Why λ=1.0**: The MSE on activations is 1–2 orders of magnitude smaller than the classification loss, so λ=1.0 gives the local loss roughly 1–10% of the total gradient signal. No sweep was run for this first experiment.

5000 steps, batch size 128, eval every 100 steps. Followed by self-knowledge probes (40 batches, 300 probe steps) and perturbation robustness test (4 epsilon levels, 20 batches).

## Results (2026-06-15)

### Learning speed: LL learns faster (confirmed)

| Step | OL | CL | LL | CL_LL |
|---|---|---|---|---|
| 500 | 0.889 | 0.875 | **0.914** | 0.873 |
| 1000 | 0.930 | 0.934 | 0.930 | **0.938** |
| 2500 | 0.953 | 0.959 | **0.972** | **0.972** |
| 4999 | 0.958 | 0.961 | **0.972** | 0.969 |

LL reaches +2.5pp accuracy over OL at step 500 and maintains the lead throughout. Final val_loss is 31% lower (0.088 vs 0.127). The classification loss gives gradient only to the [CLS] token; the local loss gives gradient to all 50 positions × 128 dims — a 6400-dimensional supervision signal per example vs effectively 128-d from [CLS] alone.

### Final metrics

| Condition | val_loss | val_acc | fwd_cos |
|---|---|---|---|
| OL | 0.127 | 0.958 | 0.973 |
| CL | 0.118 | 0.961 | 0.903 |
| LL | **0.088** | **0.972** | **0.997** |
| CL_LL | 0.129 | 0.969 | 0.985 |

LL achieves the best accuracy and loss, with near-perfect FM cosine (0.997). The "be predictable" pressure made the model's computation almost entirely FM-compressible, and this was *beneficial* for the classification task. The MNIST wake-sleep results predicted this: making computation more FM-predictable simultaneously improved task performance.

CL_LL has higher val_loss than LL (0.129 vs 0.088) despite similar accuracy (0.969 vs 0.972). The injection introduces a conflicting pressure: blocks 2–3 must use the injection (which the FM didn't predict from) while also matching the FM's prediction (which doesn't account for the injection). This conflict hurts calibration (loss) more than discrimination (accuracy).

CL_LL is the first closed-loop condition to maintain high FM cosine (0.985 vs 0.903 for CL). The local loss stabilizes co-training dynamics by actively pushing the model toward FM-predictable computation, preventing the moving-target problem that degrades FM quality in standard CL.

### Dependency: CL_LL reduces dependency by 63% (confirmed)

| Condition | Dependency | Injection benefit | Gate norm |
|---|---|---|---|
| CL | +0.022 | −0.031 | 1.49 |
| CL_LL | **+0.008** | −0.006 | 1.66 |

Dependency = val_loss_no_injection − OL_val_loss. CL_LL has 63% less dependency than CL. The continuous "be predictable" pressure from the local loss counteracts the offloading dynamic: the model can't become reliant on the injection because the local loss forces continuous internalization.

The injection benefit is also 5× smaller in CL_LL (−0.006 vs −0.031) — the model needs the injection less because it has already internalized most of what the FM provides. The gate norm is slightly higher (1.66 vs 1.49), meaning the system still learns to use the injection, but the dependency is much smaller.

### Robustness: LL is fragile, not robust (refuted)

| Condition | Δloss (ε=1.0) | Ratio vs OL |
|---|---|---|
| OL | +0.012 | 1.00 |
| CL | +0.003 | **0.24** |
| LL | +0.158 | **13.4** |
| CL_LL | +0.045 | 3.84 |

LL is 13× more sensitive to perturbations than OL — the local loss made the model precise but brittle. CL_LL is intermediate (3.8×), with the injection partially compensating.

**This dissociates regularity from robustness.** The local loss and the injection produce opposite effects on perturbation sensitivity:

- **Regularity** (from local loss): compresses routine computation. Helps learning speed and generalization. Does NOT produce robustness — the tight block0→block3 mapping amplifies perturbations at the injection point (after block 1).
- **Perturbation experience** (from injection): the model practices absorbing structured additive signals during training. This generalizes to random perturbations. Does NOT help learning speed.

This revises the interpretation from the [Jacobian analysis](JACOBIAN_ANALYSIS_README.md), which attributed CL robustness to "less functional load on blocks 2–3 from division of labor." LL achieves the same division of labor (the FM handles predictable computation) without any robustness gain. The robustness specifically requires injection experience — training with structured noise at the perturbation point.

### Self-knowledge probes: LL has none, CL_LL has the most

Self-knowledge probes: R² for predicting the FM's residual vector from model activations (no injection).

| Layer | OL | CL | LL | CL_LL |
|---|---|---|---|---|
| post_embed | 0.25 | 0.30 | 0.26 | **0.63** |
| post_block0 | 0.11 | 0.63 | −0.03 | **0.76** |
| post_block1 | 0.18 | 0.68 | −0.10 | **0.77** |
| post_block2 | 0.27 | 0.70 | −0.07 | **0.78** |
| post_block3 | 0.38 | 0.73 | −0.05 | **0.79** |

LL has zero self-knowledge (R² ≈ 0 or negative). The local loss made the FM nearly perfect (cos = 0.997), leaving a near-zero residual with no structure to predict.

CL_LL has the highest self-knowledge of any condition — higher than CL at every layer. The injection introduces irreducible unpredictability (the FM predicts from pre-injection activations), creating a meaningful residual. The local loss then helps the model encode this residual by providing direct gradient supervision about the FM's error structure.

## Representation probes: object-level vs meta-knowledge (2026-06-15)

**Code**: `mnist_local_loss_probes.py`

Does CL_LL internalize the FM's *content* (object-level: what the FM predicts) or the FM's *error structure* (meta-knowledge: where the FM is wrong)?

Four probe types at each layer (linear probes, no injection, native activations):

| Probe | Target | Measures |
|---|---|---|
| prediction | FM(post_block0) | Object-level: what the FM would predict |
| residual | post_block3 − FM(post_block0) | Composite: FM error (correlated with prediction) |
| ortho_residual | residual ⊥ prediction | Pure meta: FM error orthogonal to prediction direction |
| target | post_block3 | Ceiling control |

### Target statistics

| Condition | pred_var | res_var | ortho_var | pred–tgt cos |
|---|---|---|---|---|
| OL | 1.24 | 0.063 | 0.058 | 0.974 |
| CL | 6.37 | 1.229 | 0.677 | 0.903 |
| LL | 0.42 | **0.002** | **0.002** | 0.997 |
| CL_LL | 0.51 | 0.020 | 0.013 | 0.985 |

LL's residual variance is 600× smaller than CL's, confirming the FM is nearly perfect. CL_LL's residual variance is 10× larger than LL's — the injection creates a meaningful residual.

### The key comparison: CL_LL vs CL

**Prediction probe (object-level: "what does the FM predict?") — R²:**

| Layer | OL | CL | LL | CL_LL | CL_LL − CL |
|---|---|---|---|---|---|
| post_embed | 0.36 | 0.19 | 0.51 | 0.49 | +0.30 |
| post_block0 | 0.89 | 0.86 | 0.97 | 0.96 | **+0.10** |
| post_block1 | 0.95 | 0.93 | 0.98 | 0.97 | +0.03 |
| post_block3 | 0.96 | 0.94 | 0.98 | 0.98 | +0.04 |

**Orthogonal residual probe (pure meta: "where is the FM wrong, ⊥ prediction direction?") — R²:**

| Layer | OL | CL | LL | CL_LL | CL_LL − CL |
|---|---|---|---|---|---|
| post_embed | 0.06 | 0.22 | 0.26 | 0.68 | +0.45 |
| post_block0 | 0.10 | 0.42 | −0.07 | 0.73 | **+0.31** |
| post_block1 | 0.18 | 0.60 | −0.17 | 0.73 | +0.14 |
| post_block3 | 0.36 | 0.74 | −0.07 | 0.76 | +0.03 |

### Result: predominantly meta-knowledge, 3:1 at early layers

At post_block0, the CL_LL advantage over CL:
- Prediction (object-level): Δ R² = **+0.10**
- Ortho residual (pure meta): Δ R² = **+0.31**

The meta-knowledge gain is **3× the object-level gain**. At post_block1, the ratio is 4:1 (+0.14 vs +0.03). CL_LL predominantly encodes *where the FM is wrong* — which directions of computation will be surprising — not *what the FM predicts*.

The meta-knowledge advantage is concentrated at early layers (Δ = +0.31 at block0, +0.03 at block3). CL already has strong late-layer meta-knowledge from the injection alone; the local loss specifically adds early-layer meta-knowledge by providing direct gradient supervision about the FM's error structure through backpropagation.

The object-level gain (+0.10) is likely a byproduct: encoding where the FM is wrong implicitly encodes something about what it predicts, because the two signals share a common reference frame.

### LL encodes object-level knowledge but zero meta-knowledge

LL shows the complementary pattern: prediction probe R² = 0.97 at block0 (highest of all conditions) but ortho_residual R² = −0.07 (negative). The local loss without injection produces a model that perfectly encodes what the FM predicts (it was trained to match it) but has zero information about where the FM is wrong (because the FM is essentially never wrong — cos = 0.997). Self-knowledge requires a meaningful residual to have knowledge ABOUT, which requires the injection to introduce irreducible unpredictability.

## Key takeaways

### 1. The local loss and injection serve complementary functions

| Property | Local loss | Injection | Both (CL_LL) |
|---|---|---|---|
| Learning speed | +1.4pp accuracy | No effect | +1.1pp |
| Dependency | N/A (no injection) | +0.022 | +0.008 (63% less) |
| Robustness | 13× worse | 4× better | 3.8× worse |
| Self-knowledge (R²) | ≈ 0 | 0.63–0.73 | 0.76–0.79 (best) |
| FM cosine | 0.997 (near-perfect) | 0.903 (degraded) | 0.985 (high + stable) |
| Knowledge type | Object-level only | Meta-knowledge | Meta-knowledge (3:1) |

The local loss provides learning-time efficiency (dense supervision). The injection provides runtime self-monitoring (robustness, meta-knowledge). Together, they produce the strongest self-knowledge with reduced dependency, but the injection's robustness is partially counteracted by the local loss's precision.

### 2. Regularity ≠ robustness

The prior interpretation (from the Jacobian analysis) was that CL robustness comes from "division of labor" — the FM handles predictable computation, leaving a flatter loss landscape. LL achieves the same division of labor without any robustness gain. **Robustness requires perturbation experience** (training with structured additive signals), not computational regularity.

### 3. Self-knowledge requires a meaningful residual

LL has zero self-knowledge despite the highest task performance. The FM's prediction is so accurate that the residual is near-zero — there is nothing to have self-knowledge ABOUT. Self-knowledge is only informative when the FM is imperfect enough to produce structured prediction errors, which requires the injection to introduce computation the FM can't fully predict.

### 4. The local loss provides a direct channel for meta-knowledge at early layers

CL's meta-knowledge at early layers arises indirectly (NTP backprop through the injection path). The local loss provides a direct channel: the gradient of `MSE(FM_pred, post_block3)` at early layers is proportional to `∂post_block3/∂early_act · (post_block3 − FM_pred)`, which has the direction of "what deviation from the FM prediction was caused by your computation" — the meta-knowledge signal itself. This explains the 3:1 ratio: the local loss gradient literally carries the meta-knowledge.

## Reproduction

```bash
cd experiments/

# Main experiment (4 conditions, ~20 min)
modal run --detach a2a_forward/mnist_local_loss.py::a2a_mnist_local_loss

# Representation probes (loads checkpoints from above, ~10 min)
modal run --detach a2a_forward/mnist_local_loss_probes.py::a2a_mnist_ll_probes
```

## Next steps

1. **λ sweep**: Test λ ∈ {0.01, 0.1, 1.0, 10.0} to find the robustness crossover. At some λ, the precision pressure should be small enough that the injection's robustness dominates. The optimal λ for CL_LL balances learning speed and robustness.
2. **Language domain**: Run the same 4-condition experiment on the GPT language model. Language has a full-rank residual (200/256 effective rank), so the local loss dynamics may differ from MNIST's low-rank (18/128) case. The learning speed prediction is less clear in language where each token already provides a dense NTP signal.
3. **Precision weighting**: Replace the raw MSE local loss with a precision-weighted version (Experiment 2 in the idea doc). Weight each direction by the FM's expected error variance — this should selectively compress routine computation while leaving genuinely complex computation alone, potentially avoiding the brittleness of the uniform MSE.
4. **OOD adaptation**: The [MNIST adaptation experiment](MNIST_ADAPTATION_README.md) showed a three-way dissociation between distillation, adaptation speed, and forgetting. Test where LL and CL_LL conditions fall on all three measures — the learning speed advantage may translate to faster OOD adaptation.

## Modal volume

Results saved to `language-reduction-data` volume:

```
/data/a2a_forward/mnist_local_loss/
└── vit_4L_4H_128D/post_block0_to_post_block3/lambda_1.0/
    ├── OL_model.pt, OL_fwd.pt
    ├── CL_model.pt, CL_fwd.pt, CL_gate.pt
    ├── LL_model.pt, LL_fwd.pt
    ├── CL_LL_model.pt, CL_LL_fwd.pt, CL_LL_gate.pt
    ├── results.json
    └── probe_results.json
```
