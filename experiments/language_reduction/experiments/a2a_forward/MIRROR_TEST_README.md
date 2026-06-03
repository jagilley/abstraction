# Mirror Test for Neural Self-Knowledge (2026-06-02)

**Code**: `mirror_test.py`
**Prior experiment**: [REPRESENTATIONAL_DIVERGENCE_README.md](REPRESENTATIONAL_DIVERGENCE_README.md) (divergence PCs defining the self-knowledge subspace), [CONTROLLED_RETRAIN_README.md](CONTROLLED_RETRAIN_README.md) (Run 6, 1% model), [EXTENDED_TRAINING_README.md](EXTENDED_TRAINING_README.md) (Run 7b, 10% model)

## Motivation

The controlled retrain (Run 6) showed the closed-loop model encodes self-knowledge (Δ R² ~+0.18). The directional steering (Run 8) showed it partially *uses* that self-knowledge (focused-attention cluster selectivity = 2.41). But neither test has the specific logical structure of asking: **does the model use self-representation to detect and respond to changes in its own computation?**

The [mirror test](https://en.wikipedia.org/wiki/Mirror_test) (Gallup, 1970) answers exactly this for animals. An animal is shown a representation of itself (mirror), a mark is applied that can only be detected via the mirror, and self-recognition is demonstrated by the animal acting on itself based on what the mirror reveals — reaching for the mark on its own face, not on the reflection. The test works because the behavioral response is *direction-specific*: the animal reaches for the mark, not at the mirror.

## Design

Three model conditions:

| Condition | Model | Injection | Interpretation |
|---|---|---|---|
| CL+M | Closed-loop | Active (mirror present) | Self-knowledge + real-time mirror |
| CL-M | Closed-loop | Removed at test time | Self-knowledge in weights only |
| OL | Open-loop | Never present | No self-knowledge |

**The mark**: Random perturbation vectors added to the residual stream at the injection point (post_block1). 16 random directions + 3 post_block1 divergence PCs. Perturbation magnitude scaled by b1 activation std × s, with s ∈ {0.5, 1.0, 2.0, 3.0}.

**The mirror**: The forward model's prediction, injected via the trained CerebellarGate. The perturbation does not affect the forward model's prediction (which depends only on post_block0), so the discrepancy between prediction and perturbed actual state changes — the "mirror" reveals the "mark."

**"Reaching for the mark"**: The model's downstream response at post_block3 (perturbed − baseline activations) is projected onto the **self-knowledge subspace** — the top-k divergence PCs between the closed-loop and open-loop models at post_block3 (which capture 2.42× more forward-model residual variance than random directions, per the representational divergence analysis).

The SK fraction (||response projected onto SK subspace||² / ||response||²) measures how much of the perturbation response is channeled through self-knowledge dimensions. Random baseline = k/256.

## Results

### Run 1: 1% forward model (660K params, 10K steps)

Checkpoints from the controlled retrain (Run 6). SK subspace: top-10 divergence PCs at post_block3 (10.7% of divergence variance, effective rank 226.5).

**SK fraction of perturbation response (random directions, s=2.0):**

| Condition | SK fraction | Ratio vs random (0.039) | Δ vs OL |
|---|---|---|---|
| CL+M | 0.0620 | 1.59× | +0.0106 (+20.5%) |
| CL-M | 0.0630 | 1.61× | +0.0116 (+22.5%) |
| OL | 0.0514 | 1.32× | — |

**Loss degradation from perturbation (random directions, s=2.0):**

| Condition | Baseline loss | Δloss | Response norm |
|---|---|---|---|
| CL+M | 5.361 | +0.011 | 2.40 |
| CL-M | 5.444 | +0.010 | 2.41 |
| OL | 5.335 | **+0.023** | **2.60** |

**B1 divergence PC perturbations (s=2.0):**

| Direction | CL+M | CL-M | OL |
|---|---|---|---|
| b1_div_pc0 | 0.229 | 0.226 | 0.171 |
| b1_div_pc1 | 0.286 | 0.284 | 0.174 |

### Run 2: 10% forward model (3.2M params, 15K steps)

Checkpoints from extended training Run 7b, which had 2× stronger self-knowledge probes (Δ R² = +0.34 to +0.47 vs +0.16 to +0.20).

**SK fraction (random directions, s=2.0):**

| Condition | SK fraction | Ratio vs random | Δ vs OL |
|---|---|---|---|
| CL+M | 0.0653 | 1.67× | +0.0126 (+24.0%) |
| CL-M | 0.0671 | 1.72× | +0.0145 (+27.5%) |
| OL | 0.0526 | 1.35× | — |

**Loss degradation (random directions, s=2.0):**

| Condition | Baseline loss | Δloss |
|---|---|---|
| CL+M | 5.636 | +0.015 |
| CL-M | 5.843 | +0.012 |
| OL | 5.625 | **+0.035** |

## Key findings

### 1. The mirror test passes: closed-loop models channel perturbation responses through the self-knowledge subspace

Both CL models show ~20-28% higher SK fraction than the OL model across both forward model capacities. The effect is consistent across 16 random perturbation directions, multiple perturbation strengths, and two different checkpoint pairs.

The b1 divergence PC perturbations (which are partially aligned with the SK subspace by construction) show a much larger gap: CL models at 0.23-0.29 vs OL at 0.17 (1% model). This is the "directional specificity" signature — perturbations that are more coupled to the self-knowledge structure produce proportionally larger CL-vs-OL differences.

### 2. Self-knowledge is in the weights, not the real-time mirror

CL-M ≥ CL+M in both runs. Removing the injection at test time does not reduce the SK-channeling — it slightly increases it. The self-knowledge was needed during *training* (to develop the self-knowledge representations via backprop), but at inference, it's fully internalized in the weights.

This parallels the biological mirror test: an animal that has developed self-recognition doesn't need to be looking in a mirror to know where its face is. The mirror was the training signal; the self-model persists without it.

### 3. The robustness gap: closed-loop models absorb perturbations better

The OL model takes 2-3× more loss degradation from identical perturbations:

| | 1% fwd | 10% fwd |
|---|---|---|
| OL Δloss / CL+M Δloss | 2.1× | 2.3× |

The OL model also shows larger response norms (the perturbation propagates further through its computation). The CL model's response is both *smaller* and *more structured* — it absorbs perturbations with less damage, and what response it does produce is channeled through self-knowledge dimensions rather than being isotropic.

This is a new finding that may be worth following up on. The self-knowledge representations may act as a kind of structural regularizer that constrains how perturbations propagate — the model's downstream computation is organized around a self-referential structure that resists arbitrary disruption.

### 4. Effect scales with self-knowledge strength

The 10% model (2× stronger probes) shows a larger mirror test effect:
- CL-OL gap: +27.5% relative (10%) vs +22.5% (1%)
- OL/CL loss degradation ratio: 2.3× (10%) vs 2.1× (1%)

The scaling is modest relative to the 2× probe improvement, suggesting the SK-fraction metric may be a coarse measure of the underlying self-knowledge, or that additional self-knowledge beyond a threshold contributes diminishing returns to perturbation channeling.

### Geometry control: activation geometry vs self-referential response

**Code**: `mirror_test_geometry_control.py`

The SK subspace (divergence PCs at post_block3) is defined as directions of maximum difference between the CL and OL models. The CL model's blocks 2-3 may produce more variance along these directions in general, not only in response to perturbations. To test this, we computed the SK fraction of each model's **general activation variance** at post_block3 (no perturbation) and compared it against the perturbation-response SK fractions.

**General activation SK fraction (no perturbation):**

| Condition | 1% fwd | 10% fwd |
|---|---|---|
| CL+M | 0.0784 (2.01×) | 0.0784 (2.01×) |
| CL-M | 0.1133 (2.90×) | 0.1236 (3.16×) |
| OL | 0.0703 (1.80×) | 0.0755 (1.93×) |

All models' general activations are MORE SK-aligned than their perturbation responses (which were 1.3-1.7×). The absolute SK fractions don't support a self-referential interpretation — normal activations are already more SK-enriched than perturbed ones.

**But the CL/OL *relative* enrichment tells a different story:**

| Comparison | General CL/OL | Perturb CL/OL | Perturbation more enriched? |
|---|---|---|---|
| CL+M/OL (1%) | 1.12× | 1.21× | Yes (+0.09) |
| CL+M/OL (10%) | 1.04× | 1.24× | Yes (+0.20) |
| CL-M/OL (1%) | 1.61× | 1.23× | No (−0.39) |
| CL-M/OL (10%) | 1.64× | 1.28× | No (−0.36) |

**CL+M** (with injection): the perturbation response is *more* relatively enriched than general geometry predicts. The CL+M model's perturbation handling goes slightly beyond what its activation geometry explains. Consistent across both forward model capacities, and stronger for the 10% model.

**CL-M** (without injection): the perturbation response is *less* relatively enriched than general geometry. The CL-M model's general activations are massively SK-enriched (the SK subspace is defined by CL-M vs OL divergence), and the perturbation response doesn't inherit all of that. The CL-M mirror test result is largely explained by geometry.

**Interpretation**: The geometry control partially deflates the SK-fraction finding but doesn't fully explain it for the CL+M condition. The CL+M model — the one with both self-knowledge AND the real-time mirror signal — shows a perturbation-specific enrichment that exceeds its general geometry. The CL-M model's result is mostly geometric. The robustness gap (2-3× less loss degradation) remains the cleanest finding that geometry cannot explain.

## Reproduction

```bash
# 1% forward model (controlled retrain checkpoints)
modal run --detach language_reduction/modal_app.py --stage a2a-mirror-test \
  --n-tokens 10000000 --predict-from post_block0 --predict-to post_block3 \
  --fwd-n-layer 2 --inject-after-block 1

# 10% forward model (extended training checkpoints)
modal run --detach language_reduction/modal_app.py --stage a2a-mirror-test \
  --n-tokens 10000000 --n-steps 14999 \
  --predict-from post_block0 --predict-to post_block3 \
  --fwd-n-layer 3 --fwd-d-head 128 --fwd-n-head 4 --fwd-mlp-mult 4 \
  --inject-after-block 1

# Geometry control (both checkpoints)
modal run --detach language_reduction/modal_app.py --stage a2a-mirror-geometry \
  --n-tokens 10000000 --predict-from post_block0 --predict-to post_block3 \
  --fwd-n-layer 2 --inject-after-block 1

modal run --detach language_reduction/modal_app.py --stage a2a-mirror-geometry \
  --n-tokens 10000000 --n-steps 14999 \
  --predict-from post_block0 --predict-to post_block3 \
  --fwd-n-layer 3 --fwd-d-head 128 --fwd-n-head 4 --fwd-mlp-mult 4 \
  --inject-after-block 1
```
