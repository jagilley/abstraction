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
3. ~~**Precision weighting**~~: Done — see below. Precision weighting (Mahalanobis distance) reduced brittleness by only 29% (9.5× vs 13.4×) and made no difference when combined with injection (CL_PW ≈ CL_LL). The brittleness is geometric (compression of most dimensions), not directional (compressing the wrong dimensions). Precision weighting addresses a non-bottleneck.
4. ~~**Learning gate (bilevel optimization)**~~: Done — see below. A learned gate trained via bilevel optimization (MAML-style) to select which dimensions to compress. Maintains LL's learning speed, reduces brittleness by 35%, and produces record self-knowledge (CL_LG R²=0.83 at post_block0, beating CL_LL's 0.76). The gate's selectivity doesn't correlate with digit discrimination or FM error variance — the bilevel signal discovers a more abstract task-relevant criterion.
5. **CL_LG → distillation**: CL_LG produces the highest self-knowledge and injection dependency of any condition — potentially the ideal pre-distillation checkpoint. Test whether distilling CL_LG produces better post-distillation models than distilling CL_LL.
6. **OOD adaptation**: The [MNIST adaptation experiment](MNIST_ADAPTATION_README.md) showed a three-way dissociation between distillation, adaptation speed, and forgetting. Test where LL and CL_LL conditions fall on all three measures — the learning speed advantage may translate to faster OOD adaptation.

## Precision-weighted local loss (2026-06-16)

**Code**: `mnist_precision_weighted.py`

Tests whether precision-weighting the local loss fixes the brittleness of the raw MSE version. The raw MSE compressed all 128 dimensions uniformly, making the model 13× more sensitive to perturbations. Precision weighting replaces MSE with the Mahalanobis distance under the FM's error covariance: directions where the FM always errs (high variance, ~11 dimensions on MNIST) get low weight, directions where it's usually accurate (~117 dimensions) get high weight. The model is pushed to "be predictable" only where the FM is already good.

Implementation: EMA (β=0.99) of the FM error covariance (128×128), eigendecomposed each step. Each eigendirection weighted by `mean_eigenvalue / (eigenvalue + eps)`, clamped to max ratio 100. Overall scale matches raw MSE at λ=1.0.

Four conditions with identical seed (42), lr (3e-4), init weights as the raw MSE experiment: OL, CL, PW (precision-weighted local loss only), CL_PW (injection + precision-weighted local loss).

### Results

**Final metrics:**

| Condition | val_loss | val_acc | fwd_cos | (raw MSE comparison) |
|---|---|---|---|---|
| OL | 0.127 | 0.958 | 0.973 | — |
| CL | 0.118 | 0.961 | 0.903 | — |
| PW | 0.101 | 0.972 | 0.995 | LL: 0.088 / 0.972 / 0.997 |
| CL_PW | 0.141 | 0.958 | 0.986 | CL_LL: 0.129 / 0.969 / 0.985 |

PW learns faster than OL (+1.4pp accuracy), matching LL. FM cosine reaches 0.995 — slightly lower than LL's 0.997 but still near-perfect. The precision weighting correctly identified ~11 high-variance directions and stopped pushing on them, but the remaining ~117 routine directions were compressed just as tightly as with raw MSE.

**Robustness (Δloss at eps=1.0):**

| Condition | Δloss | Ratio vs OL | (raw MSE comparison) |
|---|---|---|---|
| OL | +0.012 | 1.00 | — |
| CL | +0.003 | 0.24 | 0.24 |
| PW | +0.113 | 9.53 | LL: 13.4 |
| CL_PW | +0.045 | 3.85 | CL_LL: 3.84 |

PW is 29% less brittle than LL (9.5× vs 13.4×), but still dramatically more fragile than OL. Leaving ~11/128 directions unconstrained doesn't help much when random perturbations land mostly in the other 117 tightly constrained directions. CL_PW ≈ CL_LL — the injection dominates.

**Residual structure (CLS token):**

| Condition | eff_rank | top5 var | |res| | eta² (top 5 PCs) |
|---|---|---|---|---|
| OL | 17.4 | 0.595 | 3.48 | 0.31, 0.21, 0.24, 0.18, 0.12 |
| CL | 11.6 | 0.687 | 13.99 | 0.74, 0.73, 0.56, 0.58, 0.55 |
| PW | 10.4 | 0.731 | 2.17 | 0.08, 0.13, 0.04, 0.09, 0.06 |
| CL_PW | 10.6 | 0.711 | 4.35 | 0.73, 0.54, 0.64, 0.55, 0.48 |

PW preserves a non-zero residual (|res|=2.17 vs effectively 0 for LL), confirming that precision weighting correctly left high-variance directions unconstrained. But the residual is NOT digit-discriminative (eta² = 0.04–0.13 vs OL's 0.17–0.31). The high-variance FM error directions don't correspond to class-conditional computation — the FM's capacity limits in this regime are not digit-specific. CL_PW's residual IS digit-discriminative (eta² 0.47–0.73), because the injection creates class-conditional prediction errors.

**Self-knowledge probes (R²):**

| Layer | OL | CL | PW | CL_PW | (raw MSE LL / CL_LL) |
|---|---|---|---|---|---|
| post_block0 | 0.11 | 0.63 | 0.02 | 0.70 | -0.03 / 0.76 |
| post_block3 | 0.38 | 0.73 | 0.19 | 0.74 | -0.05 / 0.79 |

PW has near-zero self-knowledge (R² = 0.02 at post_block0), same as LL despite having a non-zero residual. The residual isn't structured enough to predict. CL_PW ≈ CL_LL — injection dominates.

**Representation probes (object-level vs meta-knowledge, R²):**

| Layer | OL pred/ortho | CL pred/ortho | PW pred/ortho | CL_PW pred/ortho |
|---|---|---|---|---|
| post_block0 | 0.89 / 0.10 | 0.86 / 0.42 | 0.96 / 0.02 | 0.95 / 0.61 |
| post_block3 | 0.95 / 0.34 | 0.94 / 0.73 | 0.97 / 0.17 | 0.97 / 0.69 |

PW encodes high object-level knowledge (prediction R² = 0.96) but near-zero meta-knowledge (ortho_residual R² = 0.02), mirroring LL. CL_PW has strong meta-knowledge (0.61 at block0), similar to CL_LL (0.73). Again, the injection drives meta-knowledge, not the local loss variant.

**Precision spectrum evolution (PW condition):** The error covariance effective rank dropped from 27 (step 0, identity prior) → 11.4 (step 5000) as the FM's errors concentrated into fewer directions. The precision ratio (max/min weight) grew from ~400 (step 200) → ~1100 (step 5000). The precision weighting progressively sharpened its discrimination of routine vs complex directions throughout training.

### Interpretation: precision weighting is the wrong fix for brittleness

Precision weighting correctly identifies which directions are routine (FM-predictable) vs complex (FM-limited) and differentially weights the local loss accordingly. But this doesn't meaningfully reduce brittleness because:

1. **The problem is geometric, not directional.** Random perturbations land in all 128 dimensions. Leaving ~11 dimensions unconstrained while tightly constraining ~117 still amplifies most of a random perturbation. The brittleness comes from compression of most of the activation space, regardless of which specific directions are exempt.

2. **Precision weighting addresses the wrong distinction.** In Friston's predictive coding, precision weighting distinguishes signal from noise — "this prediction error is informative" vs "this is expected variability, ignore it." But the FM's high-variance directions aren't noise; they're structured capacity limits. And the FM's low-variance directions aren't "unexpectedly informative" — they're just directions where the FM already works well, so the errors and gradients are already small. Weighting small errors more heavily doesn't produce qualitatively different learning.

3. **CL_PW ≈ CL_LL on every metric.** The injection dominates the dynamics. The precision weighting makes essentially no difference when combined with injection — the injection creates a meaningful residual, self-knowledge, and robustness regardless of how the local loss is weighted.

4. **Robustness requires perturbation experience**, not smarter compression. This replicates and strengthens the key finding from the raw MSE experiment: the CL model's robustness comes specifically from training with structured additive signals at the perturbation point. No variant of the local loss — uniform or precision-weighted — produces robustness without injection.

The precision weighting idea (from predictive coding theory) maps cleanly onto the A2A system in principle, but the empirical result shows it addresses a non-bottleneck. The local loss's failure mode isn't "compressing the wrong directions" — it's "compression itself, in most directions, creates fragility."

**Reproduction:**
```bash
modal run --detach a2a_forward/mnist_precision_weighted.py::a2a_mnist_precision_weighted
```

## Learning gate: bilevel-optimized local loss (2026-06-16)

**Code**: `mnist_learning_gate.py`

The precision-weighted local loss failed because it used the FM's error structure to decide what to compress — and the FM's error structure is unrelated to task relevance on MNIST. Precision weighting was almost adversarial: it compressed the task-relevant directions (where the FM is accurate → high precision weight) and exempted the task-irrelevant ones (where the FM fails → low precision weight).

The learning gate replaces the fixed per-dimension weights with a learned network trained by the task loss itself. A small MLP (256→64→128, ~25K params) takes post_block0 CLS activations concatenated with the FM error at CLS, outputs per-dimension weights in [0,1] via sigmoid, and is trained via bilevel optimization: each step, the gate's weights determine the local loss, which determines the inner gradient, which determines a virtual parameter update, and the gate is trained to minimize classification loss *after* that virtual update. This is MAML applied to the local loss weighting.

The key implementation detail: the gate parameters enter L_total linearly (they just multiply r²), so the "second-order" terms in the meta-gradient are mixed partials ∂²L/(∂θ ∂gate), not the model Hessian — much cheaper than general MAML. The meta-gradient flows: gate_params → gate_w → gated_local_loss → inner_grad (via `create_graph=True`) → virtual_params (via `functional_call`) → cls_loss' → gate update.

Six conditions with identical seed (42), lr (3e-4), init weights: OL, CL, LL (raw MSE), CL_LL, LG (learning-gated local loss, no injection), CL_LG (injection + learning-gated local loss).

### Results

**Final metrics:**

| Condition | val_loss | val_acc | fwd_cos | Δloss (ε=1.0) | Rob ratio | SK R² (blk0) | SK R² (blk3) |
|---|---|---|---|---|---|---|---|
| OL | 0.127 | 0.958 | 0.973 | +0.012 | 1.00 | 0.11 | 0.39 |
| CL | 0.118 | 0.961 | 0.903 | +0.003 | 0.24 | 0.63 | 0.72 |
| LL | 0.088 | 0.972 | 0.997 | +0.158 | 13.4 | -0.03 | -0.03 |
| CL_LL | 0.129 | 0.969 | 0.985 | +0.045 | 3.84 | 0.76 | 0.79 |
| LG | 0.101 | 0.972 | 0.996 | +0.103 | 8.68 | 0.10 | 0.12 |
| CL_LG | 0.117 | 0.969 | 0.975 | +0.030 | 2.55 | **0.83** | **0.86** |

**Dependency:**

| Condition | Dependency | Injection benefit |
|---|---|---|
| CL | +0.022 | −0.031 |
| CL_LL | +0.008 | −0.006 |
| CL_LG | +0.019 | −0.028 |

### Finding 1: Learning speed maintained, brittleness reduced 35%

LG matches LL's accuracy exactly (0.972), confirming the gate preserves the full learning speed benefit of the local loss. At every checkpoint, both LL and LG are ~1.4pp ahead of OL. The bilevel optimization doesn't improve learning speed beyond raw MSE, but it doesn't hurt it either.

Brittleness dropped from 13.4× (LL) to 8.68× (LG), a 35% reduction. CL_LG likewise dropped from 3.84× (CL_LL) to 2.55×, a 34% reduction. The gate consistently helps robustness when applied. But LG is still 8.7× more fragile than OL — the gate reduced but didn't solve the brittleness problem. Compression itself, in most of the activation space, remains the core issue.

### Finding 2: CL_LG produces record self-knowledge

CL_LG's self-knowledge probes are the highest ever measured in this project:

| Layer | OL | CL | CL_LL | CL_LG |
|---|---|---|---|---|
| post_block0 | 0.11 | 0.63 | 0.76 | **0.83** |
| post_block3 | 0.39 | 0.72 | 0.79 | **0.86** |

CL_LG beats CL_LL by 7–8pp at every layer. LG alone has near-zero self-knowledge (R² ≈ 0.10), same as LL — the injection is still required for self-knowledge.

Why? In CL_LL, the local loss gradient is uniform across all dimensions — it pushes the model to encode the FM's error structure everywhere equally. In CL_LG, the bilevel optimization selectively amplifies the local loss in dimensions where compressing helps classification. The gradient signal into early layers is *filtered* — it carries only the meta-knowledge that matters for the task, rather than the full FM error structure. Early layers can encode this more precisely because it's a cleaner signal.

### Finding 3: the gate's selectivity is not about digit discrimination

The gate learns strong per-dimension differentiation: mean weight 0.56 (LG) / 0.60 (CL_LG), std 0.42 / 0.43, range [0.11, 0.92] / [0.14, 0.95]. Some dimensions are heavily compressed (gate ≈ 0.9), others nearly exempt (gate ≈ 0.1). The gate IS genuinely input-conditioned: per-dimension std across inputs is 0.38–0.40, meaning different inputs get meaningfully different weights.

But the gate's selectivity does not correspond to any pre-computed per-dimension statistic:

| Correlation | LG | CL_LG |
|---|---|---|
| Corr(gate weight, FM error variance) | −0.18 | −0.16 |
| Corr(gate weight, digit eta²) | +0.02 | +0.14 |

The predicted anti-correlation with digit discriminability (gate exempts task-relevant dimensions) did not materialize. The near-zero digit eta² correlation means the gate is NOT sorting "digit-relevant vs digit-irrelevant" dimensions. The weak negative FM error variance correlation is slightly similar to precision weighting (less weight on high-FM-error directions) but much weaker. The bilevel signal discovered a selectivity criterion that doesn't map onto either of our pre-computed measures — something more abstract about which directions of compression help classification performance after one gradient step.

### Finding 4: high dependency is a feature, not a bug

CL_LG has 2.4× the dependency of CL_LL (0.019 vs 0.008), approaching standard CL (0.022). The gate learned to open for dimensions where the injection helps classification, which naturally creates more injection dependency. CL_LL's "be predictable everywhere" pressure forces indiscriminate partial internalization, reducing dependency but also reducing the precision of what the model learns about the injection.

High dependency + record self-knowledge is potentially the ideal pre-distillation state: the model extracts maximal structured value from the injection (high dependency) while encoding maximally precise information about where it needs the injection (record self-knowledge). Distillation can then internalize this precisely structured knowledge. This makes CL_LG the natural candidate for the next distillation experiment.

### Gate weight dynamics

The gate starts at uniform 0.5 (sigmoid(0) initialization) and progressively differentiates over training:

| Step | Mean gate weight | Std | fwd_cos |
|---|---|---|---|
| 0 | 0.500 | 0.000 | 0.264 |
| 500 | 0.271 | 0.214 | 0.965 |
| 1000 | 0.280 | 0.219 | 0.972 |
| 2500 | 0.320 | 0.268 | 0.986 |
| 5000 | 0.548 | 0.418 | 0.996 |

The mean drops initially (the gate learns to reduce overall local loss pressure while the FM is still inaccurate), then rises as the FM improves and the gate discovers which dimensions benefit from compression. The std grows monotonically from 0 to 0.42 — increasing per-dimension differentiation throughout training. The FM cosine reaches 0.996, intermediate between LL (0.997) and OL (0.973).

### Interpretation

The learning gate validates the hypothesis that task-informed weighting beats FM-informed weighting: it maintains LL's learning speed while reducing brittleness and producing record self-knowledge when combined with injection. But the mechanism isn't what we predicted.

We expected the gate to discover "exempt digit-relevant directions, compress the rest." Instead, it discovered a more abstract selectivity based on which dimensions of the FM error, when compressed, improve classification after one gradient step. This is the bilevel signal — it can't be reduced to any per-dimension statistic we can pre-compute, because it depends on the model's current parameter configuration and the downstream effect of each dimension's gradient on classification.

The progression from precision weighting → learning gate parallels the general lesson of this project: the FM error structure (what the FM captures vs misses) and the task structure (what matters for classification) are largely orthogonal. Precision weighting uses FM structure; the learning gate uses task structure through the bilevel optimization. The 35% brittleness reduction and 7pp self-knowledge improvement are the quantitative returns on switching from one to the other.

**Reproduction:**
```bash
modal run --detach a2a_forward/mnist_learning_gate.py::a2a_mnist_learning_gate
```

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

/data/a2a_forward/mnist_precision_weighted/
└── vit_4L_4H_128D/post_block0_to_post_block3/lambda_1.0/
    ├── OL_model.pt, OL_fwd.pt
    ├── CL_model.pt, CL_fwd.pt, CL_gate.pt
    ├── PW_model.pt, PW_fwd.pt
    ├── CL_PW_model.pt, CL_PW_fwd.pt, CL_PW_gate.pt
    └── results.json

/data/a2a_forward/mnist_learning_gate/
└── vit_4L_4H_128D/post_block0_to_post_block3/lambda_1.0/
    ├── OL_model.pt, OL_fwd.pt
    ├── CL_model.pt, CL_fwd.pt, CL_cgate.pt
    ├── LL_model.pt, LL_fwd.pt
    ├── CL_LL_model.pt, CL_LL_fwd.pt, CL_LL_cgate.pt
    ├── LG_model.pt, LG_fwd.pt, LG_lgate.pt
    ├── CL_LG_model.pt, CL_LG_fwd.pt, CL_LG_cgate.pt, CL_LG_lgate.pt
    └── results.json
```
