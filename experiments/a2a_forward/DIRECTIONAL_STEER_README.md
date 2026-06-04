# Directional Causal Steering of Self-Knowledge (2026-05-30)

**Prior experiment**: [CAUSAL_PROBES_README.md](CAUSAL_PROBES_README.md)
**Controlled retrain**: [CONTROLLED_RETRAIN_README.md](CONTROLLED_RETRAIN_README.md)
**Idea doc**: [ideas/activation_to_activation_forward.md](../../../../ideas/activation_to_activation_forward.md)

## Why this exists

The causal probes (CAUSAL_PROBES_README.md) established that:
- The closed-loop model encodes directional self-knowledge (vector probe gap +0.18, 6x the scalar gap +0.03)
- Residual direction clusters organize injection help 5x more than norm octiles (eta^2=0.0017 vs 0.0003)
- The scalar steering test (Test 2) found no novelty-gated reliance along the residual-norm direction

The scalar test asked the wrong question. It looked for a volume knob (does the model gate reliance by *how much* the forward model will err?) when the self-knowledge is directional (the model encodes *what kind* of error the forward model will make). This experiment tests whether the directional structure is causally used.

## Design

1. **Cluster residual types**: K-means (k=8) on unit-normalized residual vectors from the no-injection pass, producing 8 cluster centroids representing different types of forward model error.

2. **Multivariate probe**: Ridge regression (lambda=0.1) from post_block1 to the full 256-d residual vector. Full vector probe R^2 = 0.229 (consistent with controlled retrain's 0.262). Per-centroid projection R^2 ranges 0.37-0.72 — the probe captures real information about which error type each position will have.

3. **Derive steering directions**: For each centroid c_j, the steering direction in post_block1 space is v_j = normalize(W @ c_j) — the direction whose perturbation maximally moves the *predicted* residual toward centroid j. This is the gradient of the predicted projection onto c_j with respect to the input.

4. **Steering sweep**: For each direction v_j and s in {-3, -2, -1, 0, 1, 2, 3}, steer post_block1 by s * sigma_j * v_j. The steer is applied identically to both the injection and no-injection passes (via cerebellar_fn), so its direct logit effect cancels. Measure per-token help = loss_no_inj - loss_inj, broken down by cluster membership (fixed from the unsteered s=0 pass).

5. **Effect matrix**: M[j, c] = slope of help for cluster-c tokens vs s when steering along direction j. If the model uses directional self-knowledge, M should be approximately diagonal — steering along direction j primarily changes help for tokens in cluster j.

**Controls**: Scalar novelty (residual norm) direction (same as Test 2), 3 random directions.

## Results

### The steering produces real effects

Cluster direction slopes are ~4x larger than random direction slopes (mean absolute effect ~0.004 vs random std ~0.001). The perturbations are not noise — they change how much the injection helps.

### The effect matrix is partially diagonal

| direction | clu0 | clu1 | clu2 | clu3 | clu4 | clu5 | clu6 | clu7 |
|---|---|---|---|---|---|---|---|---|
| cluster_0 | **+.0016** | +.0031 | +.0034 | +.0048 | +.0008 | +.0035 | +.0020 | +.0015 |
| cluster_1 | +.0047 | **+.0049** | +.0077 | +.0065 | +.0080 | +.0017 | +.0074 | +.0071 |
| cluster_2 | +.0048 | +.0054 | **+.0065** | +.0031 | +.0012 | +.0005 | +.0025 | +.0009 |
| cluster_3 | -.0029 | +.0004 | -.0051 | **-.0070** | -.0073 | +.0028 | -.0038 | -.0039 |
| cluster_4 | -.0042 | +.0002 | -.0024 | -.0003 | **-.0056** | +.0011 | -.0031 | -.0018 |
| cluster_5 | -.0021 | +.0026 | -.0038 | -.0044 | -.0077 | **+.0063** | -.0029 | -.0019 |
| cluster_6 | -.0061 | -.0025 | -.0042 | +.0001 | -.0050 | -.0018 | **+.0006** | -.0034 |
| cluster_7 | -.0048 | -.0012 | -.0032 | +.0035 | -.0032 | -.0022 | -.0004 | **-.0031** |

Bold = diagonal element. Slopes in units of delta(help in nats) per sigma of steering.

| Metric | Value |
|---|---|
| Mean \|diagonal\| | 0.00444 |
| Mean \|off-diagonal\| | 0.00333 |
| \|diag\|/\|off\| ratio | **1.33** |
| Diagonal enrichment | **0.160** (uniform: 0.125) |
| Signed mean diagonal | +0.0005 |
| Signed mean off-diagonal | -0.0001 |

### Four clusters show clear selectivity

Per-direction selectivity = |M[j,j]| / mean(|M[j,:]|). Values >1 mean the diagonal element is larger than the row average.

| Cluster | Selectivity | Behavioral character |
|---|---|---|
| cluster_4 | **2.41** | Focused attention (frac max_attn>0.5 = 0.21, vs ~0.01 others) |
| cluster_2 | **2.09** | Moderate entropy, large cluster (n=20K) |
| cluster_3 | **1.70** | Moderate entropy, largest cluster (n=24K) |
| cluster_5 | **1.58** | Lowest residual norm (7.91), moderate entropy |
| cluster_7 | 1.14 | High entropy, largest cluster (n=28K) |
| cluster_1 | 0.81 | Highest entropy (2.96), highest residual norm |
| cluster_0 | 0.60 | Low entropy, moderate cluster |
| cluster_6 | 0.21 | Diagonal near zero — no directional selectivity |

**Cluster 4 is the standout.** It is the focused-attention cluster: 21% of its tokens have max attention weight >0.5, compared to ~1% for all other clusters. It also has the highest baseline injection help (+0.126, nearly 2x the average of +0.085). This is the same cluster that Test 3-enriched identified as the locus of highest injection help. Steering along its direction produces the most cluster-selective effect.

This makes computational sense: focused attention is where the forward model's prediction is most useful and most reliable (the forward model's single compressed head can match single-source retrieval well). It's the error type most worth discriminating, so the model has learned to discriminate it.

### Controls behave as expected

| Direction | Slope std across clusters |
|---|---|
| Novelty (scalar) | 0.00332 |
| Random 0 | 0.00062 |
| Random 1 | 0.00125 |
| Random 2 | 0.00106 |

Random directions produce near-zero slopes (~0.001 std), confirming the cluster effects are real. The novelty direction produces higher variance (0.003) but with no systematic diagonal structure — it affects all clusters roughly uniformly, consistent with it being a magnitude signal rather than a directional one.

### Caveat: steering directions overlap

The W@c direction cross-cosine matrix has a mean off-diagonal of 0.41. Some pairs are highly correlated (clusters 4-7: 0.75; clusters 3-4: 0.70). This means steering along direction 4 partially steers along directions 3, 5, 6, 7. A perfectly diagonal causal effect would still produce a non-diagonal effect matrix with this much direction overlap. The observed diagonal enrichment of 1.33 is likely an **underestimate** of the true directional selectivity.

A follow-up with orthogonalized directions (Gram-Schmidt on the W@c vectors) would control for this, at the cost of losing the interpretability of each direction as a specific residual type.

## Interpretation

The result is **Outcome B with partial A for specific clusters** — intermediate between "no directional specificity" and "clean diagonal."

**What the model does:** It partially uses its directional self-knowledge to modulate how it processes the injection. The effect is strongest for the focused-attention error type (cluster 4, selectivity 2.41), which is the error type where the forward model's prediction is most useful. Four of eight clusters show selectivity >1.5.

**What the model doesn't do:** It doesn't cleanly discriminate between all error types. Three clusters show selectivity <1. The overall matrix has substantial off-diagonal structure. The model has learned coarse directional discrimination — "this is a focused-attention position" vs "this is not" — rather than fine-grained discrimination across all error types.

**Why this is expected for this architecture:** The 4-layer non-looped GPT has only 2 transformer layers (blocks 2-3) downstream of the injection point to act on self-knowledge. These layers must simultaneously (a) process the injection, (b) read the directional self-knowledge from the residual stream, and (c) modulate (a) based on (b). With only 2 layers and ~2.4M parameters per layer, fine-grained directional gating is a lot to ask. The model appears to have learned the highest-value discrimination (focused attention) and left the rest coarse.

The looped transformer would give the model more recurrence to act on its self-knowledge — each loop iteration provides another opportunity to read the directional signal and adjust. This is the natural next architectural step.

## Relationship to prior results

- **Test 2 (scalar steering)**: Found no novelty-gated reliance along the scalar norm direction. This test confirms that the scalar direction is indeed not the mechanism — cluster effects are 4x larger than random, while the scalar direction produces undifferentiated effects.
- **Test 3-enriched (structural help analysis)**: Found that residual direction organizes help 5x more than magnitude (eta^2=0.0017 vs 0.0003). This test provides causal evidence that the directional organization is not just correlational — steering along direction-specific axes produces direction-specific effects.
- **Controlled retrain**: Confirmed the vector probe gap (+0.18) is 6x the scalar gap (+0.03). This test shows the model *uses* at least some of that directional information, not just encodes it.

## Files

| File | Purpose |
|---|---|
| `directional_steer.py` | Directional causal steering: multivariate probe, k-means clustering, per-direction sweep |
| `DIRECTIONAL_STEER_README.md` | This file |

Results JSON: `/data/a2a_forward/analysis/directional_steer_results.json` on the `language-reduction-data` volume.

## Reproduction

```bash
cd experiments/
modal run a2a_forward/directional_steer.py \
  --n-tokens 10000000 --predict-from post_block0 --predict-to post_block3 \
  --fwd-n-layer 2 --inject-after-block 1
```
