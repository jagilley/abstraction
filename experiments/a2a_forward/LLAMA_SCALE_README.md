# A2A Forward Model on Llama 3.2 1B

**Parent experiment**: [README.md](README.md)

## Goal

Validate the A2A forward model approach on a real pre-trained language model (Llama 3.2 1B, 1.24B params, 16 layers, d=2048) rather than our co-trained toy GPT (28.9M params, 4 layers, d=256). This is the scale-up experiment: does the forward model's capacity bottleneck produce the same residual structure at 5x the dimensionality and 43x the parameters?

This also provides a direct comparison with GLP (Luo et al., 2026), which trains a 3.4B-param *unconditional* diffusion model on Llama 8B activations. Our approach trains a ~12M-param *conditional* forward model — testing the idea doc's claim that H(activations | input, current_state) << H(activations), making conditional prediction intrinsically cheaper.

## Stage 1: Activation caching (2026-05-30)

**Code**: `llama_cache_acts.py`

Cached 100M tokens of Llama 3.2 1B activations from FineWeb-Edu. For each token position, we store the residual-stream activations at two adjacent layers (post_block_7 and post_block_8 — the middle layers, matching GLP's choice of the middlemost layer).

### Setup

- **Model**: `unsloth/Llama-3.2-1B` (Unsloth mirror of the gated Meta model)
- **Data**: FineWeb-Edu `sample-10BT` subset, streamed
- **Layers**: post_block_7 → post_block_8 (1-layer gap, analogous to the toy model's post_block_0 → post_block_1)
- **Packing**: Documents tokenized and concatenated into fixed 2048-token chunks (no padding waste)
- **Storage format**: float16 npz shards, 1M tokens per shard (~8 GB each)

### Results

| Metric | Value |
|---|---|
| Tokens cached | 100,007,936 |
| Shards | 101 |
| Storage | ~819 GB |
| Throughput | 17,882 tok/s |
| Inference time | 93 min |
| GPU | L40S ($1.95/h) |
| Total cost | ~$3.00 |

### Modal volume

Saved to `language-reduction-data` volume:

```
/data/a2a_llama/
└── acts_L7_L8/
    ├── meta.json              # Run metadata (model, layers, token count, etc.)
    ├── shard_0000.npz         # Each shard contains:
    ├── shard_0001.npz         #   source: float16 (1M, 2048) - post_block_7
    ├── ...                    #   target: float16 (1M, 2048) - post_block_8
    └── shard_0100.npz         #   token_ids: int32 (1M,) - for analysis
```

Storage fits within Modal's 1 TiB/month free volume tier.

### CLI

```bash
cd experiments/

# Cache activations (default: 100M tokens, layers 7→8)
modal run a2a_forward/llama_cache_acts.py --n-tokens 100000000

# Different layers or smaller test run
modal run a2a_forward/llama_cache_acts.py --n-tokens 10000000 \
  --source-layer 4 --target-layer 12
```

### Loading cached activations

```python
import numpy as np

shard = np.load("/data/a2a_llama/acts_L7_L8/shard_0000.npz")
source = shard["source"]      # (1000000, 2048) float16 — post_block_7
target = shard["target"]      # (1000000, 2048) float16 — post_block_8
token_ids = shard["token_ids"] # (1000000,) int32
```

## Stage 2: Forward model training

**Code**: `llama_train_fwd.py`

Train a transformer forward model on the cached activations. The model is ~1.4% of Llama's parameters (17.8M), creating a capacity bottleneck analogous to the toy model's 1% config.

### Architecture

Same `TransformerForwardModel` as the toy model, scaled to d=2048:
- 1 layer, 1 attention head, d_head=64, MLP expansion 2x (hidden=4096)
- Pre-norm (LayerNorm before attention and MLP), residual connections
- Causal attention with full 2048-token context (matching Llama's cached sequence length)
- 17,318,080 params (1.40% of Llama's 1.24B)

The attention bottleneck is tighter than the toy model: d_head=64 compresses Llama's 32-head, 2048-dim attention mechanism by 32x, vs the toy model's 4x compression (d_head=64 from 4 heads at d=64 each). This is intentional — the forward model needs to be small enough that its residual captures computational novelty.

### Training

- **Data**: Streams cached activation shards from the volume (one shard at a time to fit in CPU memory, data kept as float16 on CPU and converted to float32 per batch on GPU)
- **Split**: Last 2 shards held out for validation (~1M tokens, 491 sequences), remaining 99 shards for training (~99M tokens)
- **Shard rotation**: Train on each shard for 100 steps (random sampling with replacement), then load next shard. Full run cycles through ~99 shards ~3.3 times.
- **Optimizer**: AdamW, lr=1e-3, weight_decay=0.01
- **Loss**: MSE between predicted and actual post_block_8 activations
- **Batch**: 16 sequences x 2048 tokens = 32K tokens per step
- **Steps**: 10K (total 328M tokens seen, ~3.3 epochs through training data)
- **GPU**: L4, 64 GB CPU memory

### Results (2026-06-01)

| Metric | Value |
|---|---|
| Peak cosine sim (step 3600) | **0.884** |
| Final cosine sim (step 10000) | **0.806** |
| Final MSE | 0.005 |
| Final residual norm | 3.31 |
| Effective rank | 1824.9 / 2048 (89%) |
| Top-1 PC variance | 0.7% |
| Rank for 90% variance | 1284 / 2048 |
| Training time | 3.4 hours |

**Training trajectory**: The model improved steadily from cosine 0.268 (random) to 0.884 over the first 3600 steps. At step 3800, an lr instability caused a sharp regression to cosine 0.760 (MSE doubled from 0.003 to 0.007). The model partially recovered to 0.806 over the remaining 6200 steps but never regained its pre-spike performance. The instability is diagnosed as lr=1e-3 being too aggressive for d=2048 — the same lr worked at d=256 in the toy model, but gradient magnitudes scale with dimensionality.

**Comparison with toy model (1-layer gap):**

| | Toy (d=256, 1.1%) | Llama (d=2048, 1.4%) |
|---|---|---|
| Cosine sim | 0.972 | 0.884 (pre-spike) / 0.806 (final) |
| MSE | 0.003 | 0.003 (pre-spike) / 0.005 (final) |
| Residual norm | 0.85 | 2.56 (pre-spike) / 3.31 (final) |
| Effective rank / max | 199.8 / 256 (78%) | 1824.9 / 2048 (89%) |
| Top-1 PC variance | 2.4% | 0.7% |

The lower cosine compared to the toy model is expected: Llama's layer 8 has 32 attention heads (2048 total attention dimensions) vs the toy model's 4 heads (256 total attention dimensions). The forward model's single 64-dim head is a 32x compression of the attention mechanism, vs only 4x for the toy model. A larger forward model (10% capacity, per the scaling sweep plan) should close this gap.

### Key findings

1. **The capacity bottleneck theory validates at scale.** A 17M-param forward model captures 80–88% of the directional content of a 1.24B-param model's layer computation, trained purely on frozen activations without access to the main model's weights or gradients.

2. **The residual is high-rank and diffuse — even more so than at toy scale.** Effective rank is 89% of max (vs 78% for the toy model). Top-1 PC explains only 0.7% of variance (vs 2.4%). The forward model is slightly worse everywhere, not missing specific sub-circuits. This is consistent with the theory: language computation is distributed across all dimensions, and the capacity bottleneck binds uniformly.

3. **Conditional prediction is dramatically cheaper than unconditional.** GLP trains a 3.4B-param unconditional diffusion model on Llama 8B activations. Our conditional forward model achieves 0.88 cosine with 17M params on Llama 1B — 200x fewer parameters. This validates the idea doc's claim that H(activations | input, current_state) << H(activations). The cerebellum doesn't need to model the unconditional distribution of cortical states; it just needs to predict the next state given the current state.

4. **lr=1e-3 is too high at d=2048.** The toy model trained stably at lr=1e-3 with d=256, but the same lr caused an instability at step 3800 that cost ~8pp of cosine. Next run should use lr=3e-4.

### CLI

```bash
cd experiments/

# Default: 1L 1H d_head=64 mlp_mult=2, layers 7->8
modal run --detach a2a_forward/llama_train_fwd.py --n-steps 10000

# Custom config
modal run --detach a2a_forward/llama_train_fwd.py --n-steps 10000 \
  --fwd-n-layer 1 --fwd-n-head 1 --fwd-d-head 128 --fwd-mlp-mult 2
```

Note: the CLI default for `--fwd-d-head` is 64 (inherited from the toy model CLI). The `llama_train_fwd.py` function default is 128, but the CLI overrides it. Pass `--fwd-d-head 128` explicitly if you want the wider attention bottleneck.

### Modal volume

```
/data/a2a_llama/
├── acts_L7_L8/                        # Stage 1: cached activations
│   ├── meta.json
│   ├── shard_0000.npz ... shard_0100.npz
└── fwd_1L_1H_64d_mlp2/L7_to_L8/      # Stage 2: trained forward model
    ├── fwd_model.pt
    └── results.json
```

## Stage 3: Per-head decomposition (2026-06-08)

**Code**: `llama_head_decomposition.py`

Decompose layer 8's computation into its 32 attention head contributions + MLP contribution, then measure how each component relates to the forward model's prediction residual. The forward model's limited capacity forces it to prioritize certain components over others, and the capture profile reveals which aspects of the layer's computation are most predictable from the preceding layer state.

### Method

Llama 3.2 1B layer 8 uses GQA (32 query heads, 8 KV heads, d_head=64). The layer's output decomposes exactly as:

```
output = input + sum(head_i for i in range(32)) + mlp_out
```

where each `head_i = per_head_attn_output_i @ o_proj.weight[:, i*64:(i+1)*64].T` (since `o_proj` is linear and bias-free). We verify this decomposition has near-zero reconstruction error (0.007).

For each of the 33 components (32 heads + MLP), we compute:
- **Mean norm**: how large is this component's contribution
- **cos(component, residual)**: directional alignment with what the forward model missed
- **cos(component, fwd_delta)**: directional alignment with what the forward model predicted
- **corr(‖component‖, ‖residual‖)**: when this component is large, is the forward model's error large?
- **Residual variance fraction** (heads only): fraction of residual variance in each head's output subspace

Evaluated on 80 batches (655K tokens) from the validation shards, using live Llama inference with hooks on `self_attn.o_proj` (to capture pre-projection attention outputs) and `mlp` (to capture the MLP contribution).

### Results

**The MLP dominates the forward model's residual, not attention.**

Despite attention contributing 84% of layer 8's output norm, the MLP accounts for the vast majority of the forward model's prediction error:

| | Norm (share of Δ) | cos(residual) | corr(‖comp‖, ‖r‖) |
|---|---|---|---|
| 32 attention heads (total) | 18.0 (84%) | 0.019 avg | 0.07 avg |
| MLP | 3.4 (16%) | **0.477** | **0.675** |

The MLP has 25x the directional alignment with the residual that the average attention head has (0.477 vs 0.019). The correlation between MLP magnitude and residual magnitude (0.675) means positions where the MLP does the most work are precisely where the forward model struggles most.

**Attention heads are remarkably uniform.** cos(component, residual) ranges from -0.011 to +0.050 across all 32 heads. No single head is dramatically harder to capture than others. Residual variance fraction is nearly uniform across heads (3.6%–4.6%, vs null expectation of 3.1% for a 64-dim subspace in 2048-dim space). The forward model is slightly worse at every head, not missing specific heads.

**By KV group:**

| Group | Norm | cos(res) | cos(fwd) | corr |
|---|---|---|---|---|
| KV 0 | 0.554 | 0.021 | 0.031 | 0.122 |
| KV 1 | 0.481 | 0.019 | 0.025 | 0.112 |
| KV 2 | 0.636 | 0.027 | **0.169** | -0.022 |
| KV 3 | 0.701 | 0.022 | 0.044 | 0.039 |
| KV 4 | 0.431 | 0.010 | 0.056 | 0.042 |
| KV 5 | 0.612 | 0.013 | 0.035 | 0.046 |
| KV 6 | 0.460 | 0.032 | 0.077 | 0.030 |
| KV 7 | 0.635 | 0.011 | 0.048 | 0.101 |
| **MLP** | **3.404** | **0.477** | **0.570** | **0.675** |

KV group 2 has the highest cos(fwd_delta) among attention groups (0.169) — the forward model's single head allocates the most capacity to matching this group's attention patterns. KV groups 0, 1, 7 have the highest norm-residual correlation (~0.10–0.12), meaning their contribution magnitudes weakly predict the forward model's error. But all attention effects are dwarfed by the MLP.

### Key findings

1. **Attention routing is compressible; the MLP is the bottleneck.** The forward model's single 128-dim head captures the aggregate effect of 32 × 64-dim heads at a 16x compression ratio. The MLP, despite being only 4x compressed (hidden 4096 vs Llama's 8192 SwiGLU), dominates the prediction error. Attention is inherently easier to approximate because the forward model can learn a "principal attention pattern" through the gauge freedom in Q/K space. The MLP's point-wise nonlinear transformation through a high-dimensional space is fundamentally harder to compress.

2. **The empirical weight decomposition is real and interpretable.** The forward model has genuinely learned that attention is the "predictable" part of a transformer layer (capturable from the preceding state) while the MLP is the "novel" part (requiring capacity the forward model doesn't have). This is a concrete statement about the nature of computation in Llama's middle layers, obtained purely by training a small model to predict activations.

3. **Within attention, the "slightly worse everywhere" pattern holds.** No specific head or KV group is dramatically harder to capture. The attention residual is uniformly distributed across all 32 heads' subspaces. The forward model misses a little of every head's computation rather than missing specific attention circuits. This is the per-head analog of the uniform degradation across behavioral categories observed in causal substitution.

### CLI

```bash
cd experiments/

# Default: analyze d_head=128 forward model on layers 7->8
modal run --detach a2a_forward/llama_head_decomposition.py

# Analyze d_head=64 checkpoint
modal run --detach a2a_forward/llama_head_decomposition.py --fwd-d-head 64
```

## Multi-head ablation: is attention compression the bottleneck? (2026-06-15)

The paper's forward model compresses Llama's 32 attention heads into a single 128-dim head (32:1 compression), vs 4:1 in the toy model. We tested whether this tighter attention compression explains the lower KL recovery at Llama scale (68% vs 94% toy) by training a 4-head forward model (4H, d_head=64, 18.9M params / 1.53% of Llama) — loosening attention compression to 8:1 while keeping nearly the same parameter count (+6%).

| | 1H d128 (paper) | 4H d64 |
|---|---|---|
| Params | 17.8M (1.4%) | 18.9M (1.53%) |
| Cosine | 0.922 | 0.923 |
| MSE | 0.002 | 0.002 |
| KL recovery | 68% | 68.3% |
| Effective rank / max | 89% | 94% |

Cosine and KL recovery are identical. 4x more attention patterns didn't help because attention was already well-captured — consistent with the per-head decomposition (attention cos with residual = 0.019, MLP cos = 0.477). The gap with the toy model is entirely the MLP bottleneck, not attention compression. To close it, you'd need a larger MLP or an activation function matching Llama's SwiGLU.

**Reproduction**: `modal run --detach a2a_forward/llama_train_fwd.py::a2a_train_llama_fwd --fwd-n-head 4 --fwd-d-head 64 --fwd-mlp-mult 2 --fwd-lr 1e-4`

### Modal volume

```
/data/a2a_llama/
├── acts_L7_L8/                        # Stage 1
├── fwd_1L_1H_128d_mlp2/L7_to_L8/     # Stage 2 (paper config)
├── fwd_1L_1H_64d_mlp2/L7_to_L8/      # Stage 2 (original config)
├── fwd_1L_4H_64d_mlp2/L7_to_L8/      # Multi-head ablation
└── analysis/                          # Stage 3
    ├── head_decomposition.json
    └── head_decomposition.png
```

### Files

| File | Purpose |
|---|---|
| `llama_cache_acts.py` | Stage 1: cache Llama activations to volume |
| `llama_train_fwd.py` | Stage 2: train forward model on cached activations |
| `llama_head_decomposition.py` | Stage 3: per-head decomposition of forward model capture profile |
| `LLAMA_SCALE_README.md` | This file |
