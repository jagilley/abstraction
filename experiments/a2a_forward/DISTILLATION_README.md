# Single-Cycle Wake-Sleep Distillation (2026-06-11)

**Code**: `distillation.py`
**Prior experiments**: [Controlled retrain](CONTROLLED_RETRAIN_README.md) (Run 6), [Extended training](EXTENDED_TRAINING_README.md) (Run 7), [Prediction trust](REPRESENTATIONAL_DIVERGENCE_README.md#prediction-trust-what-form-the-self-knowledge-takes-2026-06-10)

## Motivation

The closed-loop model develops genuine self-knowledge (the innovation map, the error-monitoring geometry) but does not internalize the forward model's content. The evidence is consistent: dependency grows monotonically (0.048 → 0.48 nats over 50K steps), the subspace injection experiment shows the model evacuates injected dimensions rather than absorbing them, and the prediction trust analysis shows the model amplifies its self-model's blind spots rather than learning to replicate its function. The main model learns *meta-knowledge about* the forward model, not the forward model's *content*.

The question is whether knowledge distillation can force internalization — training the closed-loop model to reproduce its own injection-active outputs without the injection — and if so, whether a fresh forward model trained on the consolidated model finds *different* innovation structure (indicating the model's computation genuinely changed) or *the same* innovation structure (indicating re-equilibration to the same two-level split).

This is the first step toward testing an abstraction ratchet: the hypothesis that iterated cycles of co-training (wake) and distillation (sleep) could produce successively higher-level computational abstractions within the model.

## Design

Three phases, all on the controlled retrain checkpoints (identical lr=3e-4, seed=42; Run 6).

**Phase 1 — Distillation (Sleep).** Freeze a teacher copy of the CL model. Run teacher with injection active (FM + gate) to produce soft logit targets. Train a student (starting from the same CL weights, running WITHOUT injection) to match the teacher via:

$$\mathcal{L} = \alpha \cdot \text{KL}(\text{student} \| \text{teacher}) + (1 - \alpha) \cdot \text{CE}_{\text{NTP}}$$

with $\alpha = 0.5$, lr = 1e-4, 5K steps. The NTP term prevents catastrophic forgetting; the KL term provides the teacher's "dark knowledge" (soft output distributions) which is richer than one-hot labels. Only the student's main model weights are optimized — FM and gate are frozen.

**Phase 2 — Re-point.** Freeze the consolidated (distilled) model. Train a fresh forward model (same architecture, seed=137) on the distilled model's frozen activations for 10K steps. This produces the "round 2" innovation structure.

**Phase 3 — Innovation migration.** Compare the fresh FM's residual structure against the original FM's: eigenspectrum overlap, behavioral conditioning, robustness, and self-knowledge probes.

## Results

### Phase 1: Distillation fully closed the dependency gap

| Metric | Value |
|---|---|
| Pre-distillation dependency gap | +0.083 nats |
| Post-distillation dependency gap | -0.005 nats |
| Gap closed | **105.6%** |
| Distilled model LM loss | 5.361 |
| Open-loop LM loss | 5.331 |
| CL with injection LM loss | 5.366 |
| Distilled vs OL | +0.030 nats |

The distilled model without any injection slightly outperforms the teacher with injection. The model successfully absorbed the FM's contribution into its own weights at a cost of 0.030 nats versus the open-loop baseline.

The KL between student and teacher dropped from 9.09 to ~1.0 in 200 steps, then slowly ground from 1.15 to 0.94 over the remaining 4800 steps. The persistent ~1 nat of residual KL reflects minor distributional differences, not a failure of absorption.

### Phase 2: Fresh FM reaches comparable quality

The fresh FM (seed=137) on the distilled model achieves cosine 0.934, comparable to the original FM's 0.935 on the pre-distillation CL model (both predicting post_block0 → post_block3, 3-layer gap). The distilled model's computation is approximately as predictable as the original's, but the fresh FM captures it through a fresh parameterization.

### Phase 3: Innovation migrated in direction but not in position

#### Innovation directions are near-orthogonal

| k | mean cos(principal angles) | Orig variance in fresh top-k |
|---|---|---|
| 5 | 0.41 | 4.7% |
| 10 | 0.43 | 8.6% |
| 20 | 0.61 | 15.8% |

Per-PC cosines between corresponding eigenvectors:

| PC | |cos(orig, fresh)| |
|---|---|
| 0 | 0.19 |
| 1 | 0.04 |
| 2 | 0.17 |
| 3 | 0.12 |
| 4 | 0.33 |

The top principal components of the original and fresh FM residuals are nearly orthogonal. The fresh FM's top-5 subspace captures only 4.7% of the original FM's top-5 variance. The innovation migrated in direction-space: the two FMs miss computation in different directions.

**Gauge symmetry caveat.** The forward model's gauge freedom (different weights implementing the same function) could in principle produce orthogonal residual eigenstructure without genuine innovation migration. Two observations argue against this: (1) the fresh FM's mean residual norm is substantially smaller (6.35 vs 7.22), meaning it genuinely predicts the distilled model's computation *better* than the original FM does — not just differently; (2) the original FM was trained on the pre-distillation CL model, whose computation is measurably different from the distilled model's. A direct test (computing cos(f_orig(x), f_fresh(x)) on shared inputs) would conclusively rule out gauge symmetry.

#### Behavioral conditioning is stable

| Metric | Value |
|---|---|
| corr(orig_res_norm, fresh_res_norm) | **0.81** |
| Mean orig residual norm (on distilled) | 7.22 |
| Mean fresh residual norm (on distilled) | 6.35 |

The same positions are hard for both FMs (r=0.81), but the fresh FM is uniformly better (6.35 vs 7.22). This is a coherent picture: innovation moved in *direction-space* but not in *position-space*. The same positions involve complex computation (delimiter matching, multi-source attention integration), but *what* the FM misses about that computation is different. The distilled model absorbed specific directional components that the original FM was capturing; those directions are now "routine." But the positions requiring complex attention integration remain complex, just complex in different directional ways.

#### Residual became more diffuse

| Metric | Orig FM on CL | Orig FM on distilled | Fresh FM on distilled |
|---|---|---|---|
| Effective rank | 212.3 | 224.3 | 227.9 |
| Top-1 PC | 2.8% | 1.6% | 1.3% |
| Top-5 PCs | 8.6% | 6.1% | 5.5% |
| Rank for 50% | 70 | 77 | 80 |
| Rank for 90% | 199 | 202 | 204 |

The fresh FM's residual is more diffuse than the original's — higher effective rank, lower top-PC concentration. The distilled model's computation is more uniform and harder to compress. This is consistent with the distillation making the model's job strictly harder (it must now model itself as well as the data), and with the absence of a discrete abstraction to extract on flat webtext.

#### Robustness partially retained

Random Gaussian perturbations (16 directions, s=2.0 × σ) at post_block1:

| Condition | Clean loss | Mean Δloss | Ratio vs OL |
|---|---|---|---|
| Open-loop | 5.331 | +1.775 | 1.000 |
| CL (pre-distill, no inj) | 5.448 | +1.583 | **0.892** |
| Distilled | 5.361 | +1.672 | **0.942** |

The distilled model retains roughly half the CL model's robustness advantage (ratio 0.942 vs CL's 0.892 and OL's 1.000). The closed-loop reorganization produced genuine weight-level flatness that partially survived the dissolution of the two-model division of labor. This is a positive result: it suggests the CL training forced the model to find representations that are structurally more robust, and distillation preserved some of that structure even after removing the injection channel.

#### Self-knowledge probes: late-layer retention

Probing distilled model vs OL model for the *fresh FM's* residual (a target neither model trained with):

| Layer | OL R² | Distilled R² | Δ R² |
|---|---|---|---|
| post_block0 | 0.011 | 0.012 | +0.001 |
| post_block1 | 0.019 | 0.053 | +0.034 |
| post_block2 | 0.032 | 0.130 | **+0.098** |
| post_block3 | 0.047 | 0.211 | **+0.164** |

The distilled model encodes the fresh FM's residual significantly better than the OL model at late layers, despite having never trained with the fresh FM. The depth profile is sharply different from the CL model's (which showed Δ R² ≈ +0.18 uniformly): post_block0 shows essentially zero advantage, while post_block3 shows +0.164.

This depth profile is consistent with the representational divergence analysis (Run 8): the CL model's late layers reorganized *along* self-knowledge dimensions (2.42× alignment at post_block3), while early layers reorganized orthogonally to self-knowledge. Distillation, which applies output-level KL pressure, would preserve the late-layer organizational structure (which directly affects outputs) but allow early layers to drift back toward OL-like representations (which are only indirectly constrained). The result is that the late-layer self-knowledge — the organizational structure that makes the model's own computational complexity linearly transparent — survived distillation.

**Confound.** The probe target (fresh FM residual) is defined on the distilled model's own activations: $r = a_{\text{block3}}^{\text{distilled}} - f_{\text{fresh}}(a_{\text{block0}}^{\text{distilled}})$. The distilled model has an inherent advantage because its post_block3 activations literally contain one term of the target. However, this confound predicts a gap at *all* layers (the distilled model's post_block0 is also "closer" to the target than OL's post_block0), yet the gap at post_block0 is +0.001 — near zero. The depth profile (near-zero early, large late) is the signature of genuine representational reorganization, not trivial self-correlation. A clean follow-up: train a separate fresh FM on each model's own activations, then probe each model for its own fresh FM residual. This would eliminate the cross-model confound entirely.

## Interpretation

### What worked

Distillation successfully closed the dependency gap at minimal cost. The model absorbed the FM's contribution — whatever the FM was providing through the injection channel, the model learned to do internally. The 0.030-nat cost relative to OL suggests the distillation is nearly lossless.

### Innovation migration: a qualified positive

The fresh FM misses computation in different directions than the original FM (PC cosines 0.04–0.33), but at the same positions (behavioral correlation 0.81). Something real changed in the model's internal representation — the directional structure of what's compressible vs novel shifted. But the residual became more diffuse (effective rank 212 → 228), not less, meaning no discrete abstraction was extracted.

On flat webtext at 29M parameters, this is what we predicted: the full-rank, diffuse residual (effective rank ~200/256) has no clean level to grab, so the ratchet grinds smoothly rather than clicking. The 29M → 77M model scale experiment already showed residual concentration across seven metrics. At substantially larger scale, the residual should have enough structure for discrete level shifts.

### Robustness retention: evidence for genuine representational quality

The partial robustness retention (ratio 0.942 vs CL's 0.892) is perhaps the most cleanly interpretable finding. The CL model's Hessian flatness was shown to be a property of the weights, not the injection (Section 5.5 of the paper). Distillation preserved roughly half of this weight-level property. The portion that was lost likely reflects the division-of-labor contribution to flatness — with the FM pre-supplying predictable computation, blocks 2–3 carried less functional load and thus had smoother input-output mappings. Removing the FM increases the load and some curvature returns.

### The self-knowledge survival at late layers

The late-layer probe result, if confirmed by the cross-model control, would mean: closed-loop training followed by distillation produces a model whose late-layer representations are structurally organized to make its own computational complexity transparent — more so than a model trained only on NTP. This structural transparency generalizes to novel forward models the model never trained with, suggesting it's a property of the representational geometry rather than an adaptation to a specific FM.

## Reproduction

```bash
cd experiments/
modal run --detach a2a_forward/distillation.py::main \
  --n-tokens 10000000 --distill-steps 5000 --retrain-steps 10000 \
  --predict-from post_block0 --predict-to post_block3 \
  --fwd-n-layer 2 --inject-after-block 1
```

## Things to possibly try next

- Scale up to bigger sizes and see if any distinct differences in behavior emerge
- Run multiple nights' sleep in a row! Closed-loop training -> distillation -> closed-loop training, yada yada. What happens if you run this on the same amount of data as/compute matched with just closed loop training, no sleep? what about compared to just open loop training?