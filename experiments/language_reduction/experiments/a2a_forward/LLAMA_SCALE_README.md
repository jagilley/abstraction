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
modal run language_reduction/modal_app.py --stage a2a-cache-llama --n-tokens 100000000

# Different layers or smaller test run
modal run language_reduction/modal_app.py --stage a2a-cache-llama --n-tokens 10000000 \
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

*Not yet implemented.* Will train transformer forward models of various sizes (0.5%–10% of Llama's params) on the cached activations, measuring cosine similarity, residual structure, and behavioral conditioning effects — replicating the toy model analyses at scale.

### Files

| File | Purpose |
|---|---|
| `llama_cache_acts.py` | Stage 1: cache Llama activations to volume |
| `LLAMA_SCALE_README.md` | This file |
