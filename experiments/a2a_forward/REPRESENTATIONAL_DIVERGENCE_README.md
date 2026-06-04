# Representational Divergence Analysis: Open-Loop vs Closed-Loop (2026-06-02)

**Code**: `representational_divergence.py`
**Prior experiment**: [CONTROLLED_RETRAIN_README.md](CONTROLLED_RETRAIN_README.md) (Run 6 — controlled retrain establishing the self-knowledge probe gap)

## Motivation

The controlled retrain (Run 6) showed that the closed-loop model encodes the forward model's residual dramatically better than the open-loop model (Δ R² ~ +0.18 at every layer). But the R² gap was roughly uniform across layers, which seemed to rule out layer-specific structure. This left an open question: is the reorganization the same everywhere, or do different layers change for different reasons?

Linear probes measuring a single target (residual R²) can't distinguish between "this layer reorganized *for* self-knowledge" and "this layer reorganized for other reasons and self-knowledge came along for the ride." We need to look at the full representational geometry.

## Method

Load the controlled retrain checkpoints (identical lr=3e-4, seed=42, differing only in whether the cerebellar loop is closed). Run both models on the same 327,680 eval positions. Three analysis steps:

1. **CKA** between open-loop and closed-loop activations at each layer — where do the representations diverge?
2. **PCA on the activation difference** (closed - open) at each layer — is the divergence concentrated or diffuse? What directions?
3. **Alignment between divergence directions and self-knowledge directions** — did the model reorganize *along* the dimensions that encode what the forward model misses?

Step 3 is the key test. If the divergence directions carry self-knowledge (ratio > 1), the model literally reorganized its representations along self-knowledge-relevant dimensions. If ratio ~ 1, the self-knowledge is incidental to whatever else changed.

## Results

### Step 1: CKA — monotonic divergence through the network

| Layer | CKA |
|---|---|
| post_embed | 0.972 |
| post_block0 | 0.882 |
| post_block1 | 0.819 (injection point) |
| post_block2 | 0.752 |
| post_block3 | 0.727 |

The representations diverge progressively from input to output. Even post_block0 (pre-injection, changes only via backprop) is substantially different (CKA 0.88). By post_block3, CKA is 0.73 — the two models encode substantially different geometric structure by the final layer.

### Step 2: PCA — early-layer change is concentrated, late-layer change is diffuse

| Layer | Eff rank | Top-10 frac | Relative diff norm | Rank for 50% |
|---|---|---|---|---|
| post_embed | 225.3 | 0.106 | 0.35 | 79 |
| post_block0 | **164.6** | **0.167** | 0.75 | 47 |
| post_block1 | 189.5 | 0.155 | 0.88 | 54 |
| post_block2 | 218.6 | 0.116 | 0.89 | 72 |
| post_block3 | 226.6 | 0.106 | 0.88 | 79 |

Post_block0 has the **most concentrated** divergence: lowest effective rank (164.6), highest top-10 fraction (16.7%), and only 47 PCs needed for 50% of the variance. The difference at this layer is also massive — 75% the magnitude of the activations themselves.

Later layers have more diffuse divergence (eff_rank 219-227), approaching the full-rank character of the forward model residual itself.

### Step 3: The key result — alignment grows through the network

**Residual variance captured by top divergence PCs** (ratio vs random baseline):

| Layer | Top-5 ratio | Top-10 ratio | Top-20 ratio |
|---|---|---|---|
| post_block0 | 1.10x | 1.07x | 1.07x |
| post_block1 | 1.36x | 1.32x | 1.23x |
| post_block2 | **1.87x** | **1.69x** | **1.55x** |
| post_block3 | **2.42x** | **2.19x** | **1.96x** |

At post_block3, the top-5 divergence directions capture **2.42x more** forward-model residual variance than random directions. At post_block0, the ratio is barely above 1.

**Extra self-knowledge overlap** (overlap between the *extra* probe weight directions — closed minus open probe SVD — and the divergence PCs):

| Layer | Rank-5 ratio | Rank-10 ratio | Rank-20 ratio |
|---|---|---|---|
| post_block0 | 0.73x | **0.49x** | **0.48x** |
| post_block1 | 1.65x | 1.12x | 0.84x |
| post_block2 | 1.76x | 1.35x | 1.20x |
| post_block3 | 1.65x | 1.28x | 1.28x |

At post_block0, the extra-SK directions are **below random** (0.49x at rank-10). The directions where the early layer changed most are *orthogonal* to the directions that carry its self-knowledge. At later layers (post_block1 onward), the overlap is above random.

**Projection probe confirmation** (R² predicting residual from acts projected onto diff PCs vs random directions):

| Layer | R² diff PCs | R² random | Δ | (at rank 50) |
|---|---|---|---|---|
| post_block0 | 0.154 | 0.128 | +0.026 | |
| post_block1 | 0.173 | 0.133 | +0.040 | |
| post_block2 | 0.195 | 0.150 | +0.045 | |
| post_block3 | 0.232 | 0.166 | **+0.067** | |

Consistent with the variance ratio results: diff PCs are increasingly better than random for residual prediction as you go deeper.

### Residual probe R² (replication of Run 6)

| Layer | Open R² | Closed R² | Δ R² |
|---|---|---|---|
| post_block0 | 0.024 | 0.210 | +0.187 |
| post_block1 | 0.061 | 0.263 | +0.202 |
| post_block2 | 0.157 | 0.344 | +0.187 |
| post_block3 | 0.259 | 0.422 | +0.163 |

Replicates the Run 6 finding: Δ R² is roughly uniform across layers (~+0.18). But this analysis reveals that the *nature* of the representational change underlying that uniform Δ R² is qualitatively different at early vs late layers.

## Interpretation

Two distinct phenomena drive the closed-loop model's representational reorganization:

**1. Early layers (post_block0): Large, concentrated, non-self-knowledge change.** The biggest, most structured representational change happens at the first transformer layer. The divergence is concentrated (eff_rank 164.6), large (75% of activation norm), but NOT aligned with self-knowledge directions (extra-SK overlap 0.49x, below random). This layer reorganized extensively via backprop to produce better inputs for the downstream injection-processing layers. The self-knowledge it encodes (Δ R² = +0.19) lives in *quieter* directions, orthogonal to the dominant change.

**2. Late layers (post_block2, post_block3): Diffuse, self-knowledge-aligned change.** At later layers, the divergence is more diffuse (eff_rank 219-227) but strongly aligned with self-knowledge (2.4x ratio at post_block3). The representations reorganized primarily along directions that encode what the forward model misses. These layers directly process the injected prediction, so their reorganization is driven by the need to evaluate and act on it.

This resolves an apparent contradiction from Run 6. The Δ R² was uniform across layers, which seemed to suggest the same reorganization happened everywhere. In fact, different layers changed for different reasons:
- Block 0 changed *a lot* but for general-purpose reasons; self-knowledge is a side-effect in low-variance directions
- Blocks 2-3 changed less concentratedly, but the change *is* the self-knowledge

The controlled retrain's linear probes couldn't distinguish these because they only measured a single target (residual R²). The representational divergence analysis reveals the *geometry* of the change — not just whether self-knowledge is present, but whether the dominant representational change *is* the self-knowledge.

### Biological parallel

This maps onto the distinction between cortical reorganization driven by cerebellar input vs. metacognitive encoding:
- Early processing stages (analogous to sensory/association cortex) change their representations to better serve the cerebellar prediction loop, but this change is primarily about input formatting, not self-awareness
- Later processing stages (analogous to prefrontal/executive areas) reorganize specifically around the error structure — their dominant change is metacognitive

## Reproduction

```bash
cd experiments/
modal run --detach a2a_forward/representational_divergence.py \
  --n-tokens 10000000 --predict-from post_block0 --predict-to post_block3 \
  --fwd-n-layer 2 --inject-after-block 1
```
