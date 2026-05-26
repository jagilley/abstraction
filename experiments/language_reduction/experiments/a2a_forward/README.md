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

## Structure analysis (2026-05-25)

**Code**: `analyze.py`

After training, we characterized what the forward model actually learned: is it a learned weight decomposition (like SVD in grokking), or something else?

### The headline: functional equivalence through different parameters

The forward model achieves **near-perfect attention pattern cosine similarity** with block1's heads while having **zero weight cosine similarity** with them. It found a completely different parameterization that produces the same function.

| Block1 head | Attention cosine | KL divergence | Q weight cosine | K weight cosine | V weight cosine |
|---|---|---|---|---|---|
| Head 0 | **0.989** | 0.039 | -0.022 | -0.056 | -0.005 |
| Head 1 | **0.998** | 0.008 | +0.011 | -0.008 | -0.007 |
| Head 2 | **0.995** | 0.007 | -0.031 | -0.041 | -0.011 |
| Head 3 | **0.916** | 0.280 | +0.021 | +0.014 | +0.009 |

The single forward model head replicates the attention patterns of heads 0–2 at >0.98 cosine, while all QKV weight cosines are indistinguishable from zero. Head 3 is the outlier (0.916 cosine, 37x higher KL divergence).

Q subspace overlaps are moderate (0.58–0.67), confirming the forward model's projections live in a partially overlapping but rotated subspace relative to each block1 head.

### Why zero weight cosine is expected

Attention has a gauge symmetry: applying the same rotation R to both Q and K projections preserves the attention pattern, since (QR)(KR)^T = QK^T. Similarly, rotations in V are absorbed by the output projection. The forward model landed in a **rotated version of the same functional basin** — orthogonal in parameter space, identical in function space.

This is directly analogous to the cerebellar circuit: the cerebellum builds its own weights (learned via climbing fiber supervised learning) that predict cortical dynamics without copying cortical parameters. The A2A model demonstrates this is achievable — a 330K-param model with different architecture can approximate a much larger model's layer computation with 0.972 cosine fidelity through entirely different weights.

### The residual is full-rank and diffuse

| Threshold | Rank (of 256) |
|---|---|
| 50% variance | 62 |
| 75% variance | 128 |
| 90% variance | 189 |
| 95% variance | 217 |
| 99% variance | 247 |
| Effective rank (entropy) | **199.8** |

Top-1 PC explains only 2.4%, top-5 explain 9.3%, top-10 explain 15.8%. The "missed computation" is spread uniformly across all dimensions — the forward model is slightly worse everywhere, not completely missing specific sub-circuits.

This is the **opposite** of the grokking case (where SVD captured a low-rank Fourier solution). In language, the capacity bottleneck binds uniformly. The forward model hasn't learned "what mechanisms block1 uses" in a decomposition sense. It's learned to *be* a miniature block1 — same function, different weights, slightly lower fidelity everywhere.

![Structure analysis](structure_analysis.png)
*Row 1: Residual PCA (cumulative variance, SV spectrum, top PCs by position). Row 2: CKA alignment, attention pattern similarity, QKV weight comparison. Row 3: Residual conditioned on token frequency, position, and LM loss quartile.*

### CKA confirms geometric equivalence

| Comparison | CKA |
|---|---|
| Post-attention (fwd vs block1) | **0.980** |
| Final output (fwd vs block1) | **0.982** |
| Input → fwd output | 0.752 |
| Input → block1 output | 0.740 |

The forward model and block1 organize information nearly identically in activation space (CKA > 0.98). Both transform the input by similar amounts (CKA to input ~0.74–0.75), confirming the forward model applies a transformation of comparable magnitude, not a shallow approximation.

### Residual correlates with token frequency, not prediction difficulty

| Token frequency | Mean residual norm | Count |
|---|---|---|
| <1e-6 | 0.932 | 3,227 |
| 1e-6–1e-5 | 0.923 | 42,424 |
| 1e-5–1e-4 | 0.904 | 98,959 |
| 1e-4–1e-3 | 0.878 | 97,033 |
| 1e-3–1e-2 | 0.799 | 68,586 |
| >1e-2 | 0.763 | 99,371 |

| LM loss quartile | Mean residual norm |
|---|---|
| Q1 (low loss) | 0.863 |
| Q2 | 0.851 |
| Q3 | 0.837 |
| Q4 (high loss) | 0.844 |

The residual has a clear monotonic gradient by token frequency (rare tokens: 0.93, common: 0.76) but is essentially flat across LM loss quartiles (0.84–0.86). The forward model's error reflects **training exposure** (it learned common tokens' computations better because it saw them more), not computational complexity.

### Why the forward model can use different weights

Two factors explain why the forward model finds a novel parameterization:

1. **Separation of concerns**: Block1's weights must encode both knowledge about language (what patterns exist) and a computational strategy (how to transform representations). The forward model's input (post_block0) already encodes the language knowledge. The forward model only needs to learn the *transformation*, not the data. Less to encode → more parametric freedom → a different, potentially more efficient parameterization.

2. **Rotation symmetry**: Attention's gauge symmetry (QR · (KR)^T = Q · K^T) means many weight configurations implement the same function. Different training objectives (MSE on activations vs end-to-end LM loss) navigate different parts of the loss landscape but can converge on functionally equivalent solutions related by rotation.

### Implication for self-regulation

In grokking, SVD residuals detected collapse because the Fourier solution is low-rank — structural drift shows up as a change in the low-rank approximation. In language, the residual is full-rank and diffuse, so a simple MSE penalty would regularize all dimensions equally. This might still work for preventing general drift (as the grokking rank-128 ablation showed — even full-rank references prevent collapse when the checkpoint is clean), but it wouldn't selectively target specific mechanisms.

The head 3 gap (0.916 vs >0.98) is the most natural place to look for mechanism-specific structure. Whatever head 3 does that the forward model can't replicate with a single compressed head may represent the genuinely "novel" computation a cerebellar-style monitor would be most informative about.

**Reproduction**: `modal run language_reduction/modal_app.py --stage a2a-analyze --n-tokens 10000000`

### Run 3: 2-layer transformer forward model, 3-layer gap (2026-05-25)

A 2-layer transformer (1 head/layer, 64-dim, 660K params) predicting post_block0 → post_block3 (3 main model layers).

| Metric | 1L fwd, 1-layer gap | 2L fwd, 3-layer gap |
|---|---|---|
| Forward model params | 330K | 660K |
| Final cosine sim | 0.972 | 0.935 |
| Final MSE | 0.003 | 0.015 |
| Residual norm | 0.85 | 1.93 |
| Residual-LM corr | -0.05 | -0.03 |

Despite doubling the forward model's parameters, cosine dropped from 0.972 to 0.935 — the 3-layer gap is genuinely harder to approximate. MSE bottomed at 0.013 (step ~4400) then climbed back to 0.015 by step 10K as the main model continued learning computation the forward model couldn't track. The main model also overfit during training (train LM loss 4.5 vs val 5.2), and the forward model's rising MSE tracked this divergence.

Residual-LM correlation remains near zero (-0.03), consistent with the 1-layer result: forward model errors reflect capacity limits, not prediction difficulty.

**Reproduction**: `modal run language_reduction/modal_app.py --stage a2a-train --n-tokens 10000000 --n-steps 10000 --fwd-n-layer 2 --predict-from post_block0 --predict-to post_block3`

## Run 4: Closed-loop cerebellar training (2026-05-25)

**Full writeup**: [CLOSED_LOOP_README.md](CLOSED_LOOP_README.md)

Closed the cerebellar loop: the 2-layer forward model's predictions (post_block0 → post_block3) are fed back into the main model's residual stream via a learned gated projection (`CerebellarGate`, 65.8K params, zero-initialized). Injected after block 1. Forward model trains on MSE only (no LM gradient); main model + gate train on LM loss.

**Key results**:

1. **The injection helps LM loss**: Δ = -0.05 to -0.08 nats consistently (1.6% relative, ~9% perplexity reduction). Negative at every eval step across 10K training steps.

2. **The gate opens wide**: projection weight norm grew linearly 0.08 → 3.03 throughout training, never saturating. The model never stopped finding useful structure in the prediction.

3. **Forward model quality degrades**: cosine 0.935 (open-loop) → 0.897 (closed-loop). The injection changes the model's computation, creating a moving target the forward model can't fully track.

4. **The model develops novelty awareness**: Linear probes predicting the forward model's residual (actual - predicted) from post_block3 show R²=0.44 for the closed-loop model vs R²=0.28 for the open-loop baseline — a 59% improvement. The co-trained model's later layers encode substantially more information about what was surprising in its own computation.

**Reproduction**: `modal run language_reduction/modal_app.py --stage a2a-loop-train --n-tokens 10000000 --n-steps 10000 --lr 3e-4 --predict-from post_block0 --predict-to post_block3 --fwd-n-layer 2 --inject-after-block 1`

## Next steps

1. **Controlled comparison**: Retrain open-loop and closed-loop with identical lr/seed to eliminate the training quality confound from the Run 4 probe comparison.
2. **Looped transformer**: The natural architecture for cerebellar injection — inject at each recurrence step, get adaptive compute for free. The current non-looped GPT only gets one injection point.
3. **Sleep-style consolidation**: Periodically pause main model training and give the forward model extra gradient steps to catch up on the co-evolving computation.
4. **Scaling**: Larger models where the capacity gap is more pronounced and the prediction signal's value may increase.
5. **Self-regulation**: Freeze the forward model at a checkpoint and use the residual as a regularization signal (as validated in grokking). Test whether this prevents overfitting or distributional drift.
6. **Head 3 investigation**: What does head 3 attend to that the forward model can't capture? (From Run 2 structure analysis.)

[^private]: Not mirrored: this link points to a document in the private lab repo (the roadmap, the queue, an unrun spec, reading notes, or a conversation). See the top-level README for what is held back and why.
