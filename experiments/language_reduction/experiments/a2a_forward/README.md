# A2A Forward Model (Cerebellum-Style Activation Prediction)

**Idea doc**: [ideas/activation_to_activation_forward.md](../../../../ideas/activation_to_activation_forward.md)
**Validated component**: fer/experiments/zipfian_grokking/cnb_self_regulation/README.md[^private]

## Goal

Co-train a small "cerebellum" forward model alongside a GPT language model. The forward model predicts the main model's activations at a later layer from an earlier layer. The prediction residual (actual minus predicted) is the novelty signal — what the main model computed that the forward model's capacity couldn't anticipate.

This is the first step toward the full architecture described in the idea doc. No feedback loop yet — the two models train independently.

## Architecture

**Main model**: 4-layer, 4-head, 256-dim GPT-2 (28.9M params). Trained on τ=0.0 FineWeb-Edu (unchanged corpus, 10M tokens).

**Forward model**: 1-layer transformer with compressed attention (330K params, ~1% of main model).
- 1 causal attention head, 64-dim key/query/value (compressed from 256-dim input)
- MLP: 256 → 512 → 256
- Pre-norm (LayerNorm before attention and MLP)
- Residual connections

The architecture mirrors the cerebellar circuit:
- Compressed attention projections = pontine relay (dimensionality reduction of cortical state)
- MLP expansion = granule cell combinatorial coding
- Linear readout = Purkinje cell output

**Prediction**: post_block0 → post_block1 (single transformer layer step prediction).

**Training**: Both models use AdamW. Main model lr=3e-4, forward model lr=1e-3. Forward model targets are detached — no gradient flows from forward model loss into the main model. The forward model loss is MSE between predicted and actual post-block1 activations.

## Results (2026-05-25)

### Run 1: Per-position MLP forward model (negative result)

A per-position MLP (256 → 512 → 256, 263K params) predicting post_embed → post_block1.

| Metric | Value |
|---|---|
| Final cosine sim | 0.788 |
| Final MSE | 0.021 |
| Residual norm | 2.28 |
| Residual-LM corr | +0.02 (none) |

**Why it failed**: A per-position forward model is structurally blind to cross-position effects. Post-embedding activations at position t contain only `wte(token_t) + wpe(t)` — they know nothing about other positions. The target (post-block1) contains information mixed in by two rounds of attention. The residual was dominated by "what attention contributed" rather than "what the forward model's capacity couldn't represent." The 0.788 cosine ceiling reflects structural blindness, not a capacity bottleneck.

The cosine similarity peaked at 0.895 (step 400) then steadily declined as the main model learned increasingly attention-dependent representations.

### Run 2: Transformer forward model (positive result)

A 1-layer transformer (1 head, 64-dim, 330K params) predicting post_block0 → post_block1.

| Metric | Value |
|---|---|
| Final cosine sim | 0.972 |
| Final MSE | 0.003 |
| Residual norm | 0.85 |
| Residual-LM corr | -0.05 (slight negative) |

The transformer forward model captures 97% of the computation in direction, with 7x lower MSE and 2.7x lower residual norm than the MLP. Cosine declined only 1.2pp over training (0.984 → 0.972) vs 10.7pp for the MLP.

The slight negative residual-LM correlation suggests that positions where the forward model struggles most tend to have *lower* LM loss — the main model's most complex computation happens where it's confidently aggregating context.

Late-training MSE uptick (0.002 → 0.003 from step 5K to 10K) shows the capacity bottleneck is binding — the main model develops computation the forward model can't fully track.

### Key lesson: capacity bottleneck vs structural blindness

The per-position MLP's residual captures "attention exists" — a trivially predictable structural fact. The transformer forward model's residual captures "computation too complex for this capacity" — genuine novelty. The information bottleneck must be the forward model's *capacity*, not its *structural inability to see the input*.

## Files

| File | Purpose |
|---|---|
| `forward_model.py` | `ForwardModel` (per-position MLP) and `TransformerForwardModel` (1-layer transformer) |
| `stages.py` | Modal stage `a2a_train` — co-training loop with eval and analysis |
| `README.md` | This file |

## Modal volume

Results saved to `language-reduction-data` volume:

```
/data/a2a_forward/
├── P_10000000/              # MLP run (post_embed → post_block1)
│   ├── model.pt
│   ├── fwd_model.pt
│   └── results.json
└── transformer/P_10000000/  # Transformer run (post_block0 → post_block1)
    ├── model.pt
    ├── fwd_model.pt
    └── results.json
```

## CLI

```bash
cd experiments/
# Default: transformer forward model, post_block0 → post_block1
modal run language_reduction/modal_app.py --stage a2a-train --n-tokens 10000000 --n-steps 10000

# MLP forward model (for comparison)
modal run language_reduction/modal_app.py --stage a2a-train --n-tokens 10000000 --n-steps 10000 \
  --fwd-type mlp --predict-from post_embed --predict-to post_block1
```

Note: `--fwd-type`, `--predict-from`, `--predict-to`, `--fwd-d-head`, `--fwd-n-head`, `--fwd-mlp-mult` are not yet wired as CLI args in `modal_app.py` — they're passed as kwargs to `a2a_train.remote()`. To use non-default values, either add them to the CLI parser or call the function directly.

## What the model.py change does

`model.py` was modified to add `return_intermediates=True` to `GPT.forward()`. When set, it returns a third value: a dict mapping `"post_embed"`, `"post_block0"`, ..., `"post_blockN"` to their activation tensors. This is backward-compatible — existing callers that don't pass the flag get the same `(logits, loss)` tuple.

## Next steps

1. **Wider layer gaps**: predict post_block0 → post_block3 (3 layers) or post_embed → post_block3 (full model). More computation to approximate = harder capacity bottleneck = richer residual.
2. **Feedback loop**: feed the forward model's predictions back into the main model's residual stream. The idea doc describes injecting at each recurrence step (for looped transformers) or at intermediate layers.
3. **Residual analysis**: characterize what the residual captures — does it correlate with token frequency, sequence position patterns, or specific attention head behaviors? The current per-position correlation with LM loss is near zero, but there may be structure in the residual's *direction* rather than its norm.
4. **Self-regulation**: freeze the forward model at a checkpoint and use the residual as a regularization signal (as validated in grokking). Test whether this prevents overfitting or distributional drift.
5. **Scaling**: try on larger models (more layers, wider) where the capacity gap between main and forward model is more pronounced.

[^private]: Not mirrored: this link points to a document in the private lab repo (the roadmap, the queue, an unrun spec, reading notes, or a conversation). See the top-level README for what is held back and why.
