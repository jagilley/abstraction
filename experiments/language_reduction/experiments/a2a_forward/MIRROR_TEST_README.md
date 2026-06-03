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

## Mirror test v2: compensatory response and perturbation discrimination (2026-06-03)

**Code**: `mirror_test_v2.py`

The original mirror test (above) defines a self-knowledge subspace via divergence PCs between the CL and OL models, then tests whether perturbation responses are channeled through that subspace. This has a circularity concern: the subspace is defined by the CL-OL difference and then tested for CL-OL differential engagement. The geometry control partially addressed this but showed the CL-M result is largely explained by general activation geometry.

Mirror test v2 avoids subspace cherry-picking entirely with two tests that depend on no externally-defined directions.

### Test 1 — Compensatory response ("reaching for the mark")

Applies random perturbations at post_block1 and measures the downstream response R at post_block3. Two metrics: response norm ||R|| (does the model dampen the perturbation?) and cos(R, δ) (does the response oppose or preserve the perturbation direction?).

| Condition | \|\|R\|\| (s=2) | \|\|R\|\|/OL | cos(R, δ) | Δloss |
|---|---|---|---|---|
| CL+M | 2.421 | 0.919 | +0.788 | +0.011 |
| CL-M | 2.438 | 0.925 | +0.778 | +0.010 |
| OL | 2.635 | 1.000 | +0.715 | +0.023 |

**The dampening is robust**: CL models absorb ~8% of perturbation magnitude that the OL model doesn't, consistent across all perturbation strengths (ratios 0.918–0.920 for CL+M). Loss degradation is 2× lower for CL models, replicating the robustness gap without any subspace cherry-picking.

**The CL model does not "reach for the mark"**: cos(R, δ) is *higher* for CL models (+0.788) than OL (+0.715). The response is more aligned with the perturbation direction, not less. The model produces a *smaller, more organized* response — direction-preserving but magnitude-dampened. The OL model's response is more chaotic: the perturbation scatters across many directions (lower cosine), causing more total damage.

**Interpretation**: The self-knowledge representations provide structural scaffolding that constrains perturbation propagation into coherent, low-damage pathways. This is passive absorption, not active correction — more like "the model's skin is tougher" than "the model reaches for the mark." The Gallup mirror test analogy (deliberate, targeted self-correction) may not be the right frame; what we see is structural regularization emerging from self-knowledge training.

### Test 2 — Perturbation discrimination at the logit level (null result)

Applies 8 different perturbation directions and measures whether the model's output distribution shift (Δlogits) distinguishes which perturbation was applied (off-diagonal cosine of mean logit shift vectors).

| Condition | Off-diag cosine | \|\|shift\|\| | KL div |
|---|---|---|---|
| CL+M | 0.007 | 54.1 | 0.023 |
| CL-M | 0.010 | 52.3 | 0.022 |
| OL | 0.001 | 74.2 | 0.049 |

Off-diagonal cosine is near zero for all conditions. Random perturbation directions in 256-D space are nearly orthogonal by construction, so the model's approximately linear response produces naturally uncorrelated logit shifts regardless of self-knowledge. The metric doesn't distinguish conditions.

The magnitude comparison replicates the robustness gap: OL logit shifts are 37% larger and KL divergence is 2× higher.

**Future work**: This test would become informative with semantically structured perturbation directions (e.g., residual directions grouped by behavioral category — delimiter tracking vs focused attention) rather than random directions, and at larger model scale where the output channel can express richer concept-specific effects. See Vogel (2025), "Small Models Can Introspect, Too" for the approach at 32B scale with concept-specific steering vectors.

### Test 3 — Position-level compensation structure

Compensation varies by sequence position:

| Position | CL+M/OL | CL-M/OL |
|---|---|---|
| 0 | 0.965 | 1.003 |
| 15 | 0.929 | 0.940 |
| 63 | 0.916 | 0.921 |
| 126 | 0.914 | 0.919 |

At position 0 (no context), CL-M provides no dampening (ratio 1.003) but CL+M does (0.965) — the real-time mirror helps where internal context is minimal. At later positions, both converge to ~0.92 as the weight-based self-knowledge dominates. This is consistent with the behavioral residual finding that self-knowledge effects are position-dependent.

### Reproduction

```bash
modal run --detach language_reduction/modal_app.py --stage a2a-mirror-test-v2 \
  --n-tokens 10000000 --predict-from post_block0 --predict-to post_block3 \
  --fwd-n-layer 2 --inject-after-block 1
```

## Mirror test v3: topic-level perturbation discrimination (2026-06-03)

**Code**: `mirror_test_v3.py`
**Inspired by**: Vogel (2025), "Small Models Can Introspect, Too"

### Motivation

Mirror test v2's perturbation discrimination (Test 2) was a null result: random perturbation directions in 256-D are nearly orthogonal by construction, producing trivially uncorrelated logit shifts regardless of self-knowledge. The off-diagonal cosine metric was floor-effected — it couldn't distinguish "perfect discrimination" from "no discrimination" for orthogonal perturbations.

Vogel (2025) showed that concept-specific steering vectors (e.g. "cat", "bread") produce concept-specific logit shifts in Qwen2.5-Coder-32B — the model can identify *what* was injected, not just *that* something was. The key insight: semantically coherent perturbation directions produce semantically identifiable output effects.

However, Vogel's test relies on instruction-tuning, elaborate prompting, and ~20 downstream layers — none of which our 4-layer, 28.9M-param model has. We adapt the core idea (structured perturbation directions → concept-specific output measurement) to our setting, testing not whether the model can "introspect" in the verbal-report sense, but whether the CL model's representational reorganization produces more topic-specific perturbation responses at the logit level.

### Design

**Phase 1 — Topic directions**: Classify eval sequences by topic via keyword matching in decoded text (math, biology, history — language and geography had insufficient data). Compute contrastive post_block1 directions per topic using the OL model:

```
d_topic = normalize(mean(post_block1 | topic) - mean(post_block1))
```

Topic direction pairwise cosine (near-orthogonal, mean off-diagonal: -0.077):

| | math | biology | history |
|---|---|---|---|
| math | 1.000 | -0.082 | -0.149 |
| biology | -0.082 | 1.000 | -0.001 |
| history | -0.149 | -0.001 | 1.000 |

**Phase 2 — Perturbation discrimination**: Perturb at post_block1 along each topic direction (s=2.0) and measure the mean logit boost at each topic's vocabulary tokens (82–97 token IDs per topic, 424 total unique across topics). This produces a perturbation × topic logit matrix. Diagonal enrichment (each perturbation preferentially boosts its own topic's tokens) measures topic-specific discrimination at the output level.

**Controls**: 5 random perturbation directions (same metric — should show no diagonal structure).

### Results (1% forward model, controlled retrain checkpoints)

**Logit boost matrices** (rows = perturbation direction, columns = measured topic):

CL+M:

| | math | biology | history |
|---|---|---|---|
| math | **+0.430** | -0.112 | -0.281 |
| biology | +0.131 | **+0.699** | -0.107 |
| history | -0.134 | -0.177 | **+0.645** |

CL-M:

| | math | biology | history |
|---|---|---|---|
| math | **+0.403** | -0.093 | -0.252 |
| biology | +0.131 | **+0.668** | -0.084 |
| history | -0.123 | -0.164 | **+0.604** |

OL:

| | math | biology | history |
|---|---|---|---|
| math | **+0.665** | -0.194 | -0.370 |
| biology | +0.226 | **+0.997** | -0.203 |
| history | -0.244 | -0.323 | **+0.900** |

All three conditions show clear diagonal structure. Perturbing in the "biology" direction boosts biology tokens while suppressing math and history tokens, etc. Random perturbation directions produce near-zero topic variance (0.005–0.008), confirming the topic directions capture real structure. This fixes v2's null result — the problem was random orthogonal directions, not absence of discrimination.

**Summary metrics:**

| Condition | Diag mean | Off-diag mean | Enrichment (Δ) | Diag / \|off-diag\| | Mean Δloss |
|---|---|---|---|---|---|
| CL+M | +0.591 | -0.113 | +0.705 | 5.22 | +0.051 |
| CL-M | +0.558 | -0.097 | +0.656 | 5.73 | +0.047 |
| OL | +0.854 | -0.185 | +1.039 | 4.62 | +0.112 |

### Interpretation

**Raw magnitude: OL > CL.** The OL model shows higher absolute diagonal enrichment (+1.04 vs +0.70) because it responds ~1.5× more to all perturbations (the robustness gap: OL Δloss = 0.112 vs CL+M = 0.051, ratio 2.21×). This magnitude difference is the same robustness gap seen in v2 Test 1 and is not specific to topic discrimination.

**Proportional specificity: CL > OL.** The diag/|off-diag| ratio — how much on-target signal per unit of off-target leakage — is 13–24% higher for CL models (5.22–5.73 vs 4.62). The CL model dampens perturbation responses non-uniformly: it retains 69.2% of OL's on-target (diagonal) signal but only 61.1% of OL's off-target (off-diagonal) leakage. Cross-topic noise is suppressed 8pp more than topic-specific signal. This is not uniform compression — it is selective preservation of topic-relevant information.

**The effect is in the weights, not the mirror.** CL-M shows the highest proportional specificity (5.73), consistent with v2's finding that CL-M ≥ CL+M for self-knowledge metrics. The representational organization was needed during training but is fully internalized.

### What this does and does not show

**This is not introspection.** Vogel's test requires an instruction-tuned model with ~20 downstream layers and elaborate prompting to elicit verbal reports about injected concepts. Our 4-layer vanilla model cannot do that, and both the CL and OL models show strong diagonal logit structure — any model with topic-specific representations will produce topic-specific Jacobians. The diagonal structure itself is a linear-approximation property, not evidence of self-awareness.

**This is evidence for more organized internal representations.** The CL model's perturbation response is smaller (robustness gap) and proportionally more topic-specific (higher diag/|off-diag| ratio). Perturbations are channeled into coherent, low-damage, topic-preserving pathways rather than scattering across all dimensions. This is consistent with the representational reorganization documented by the linear probes (Run 6: R²=0.42 vs 0.26), the robustness gap (v2 Test 1: 2–3× less loss degradation), and the directional steering selectivity (Run 8: focused-attention cluster at 2.41×).

The proportional specificity result adds one new piece to the picture: the CL model's organized representations don't just preserve forward-model-error structure — they also preserve topic-level semantic structure more effectively under perturbation. This is a downstream consequence of the same representational reorganization, not a separate capability.

### Limitations

1. **Only 3 of 5 topics survived** — "language" (2 sequences) and "geography" (8 sequences) had insufficient data in the 40-batch eval window. FineWeb-Edu is heavily weighted toward math/science/history.
2. **Topic directions are nearly orthogonal** (mean off-diag cosine: -0.077). This partially recapitulates the v2 problem — with more overlapping directions, the discrimination test would be harder and more informative.
3. **No error bars** — the 13–24% proportional specificity advantage is consistent across topics but could be noise with only 3 topics. A larger-scale replication with more topics and explicit bootstrap confidence intervals would strengthen the finding.
4. **Small topic samples** for direction computation (10–38 sequences per topic). The topic directions may be noisy estimates of the true topic centroids.

### Reproduction

```bash
modal run --detach language_reduction/modal_app.py --stage a2a-mirror-test-v3 \
  --n-tokens 10000000 --predict-from post_block0 --predict-to post_block3 \
  --fwd-n-layer 2 --inject-after-block 1
```
