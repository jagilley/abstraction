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
| `analyze.py` | Structure analysis: PCA, CKA, attention pattern comparison, weight-space comparison |
| `causal_substitution.py` | Causal substitution: replace block1 with forward model, measure per-behavior degradation |
| `behavioral_residual.py` | Behavior-conditioned residual analysis: attention, syntactic, difficulty, context integration categories |
| `README.md` | This file |

**Checkpoint compatibility note**: The `transformer/P_10000000` forward model checkpoint was saved with the original flat `TransformerForwardModel` API (top-level `ln1`, `q_proj`, etc.). The code was later refactored to use `ForwardBlock`/`blocks` for multi-layer support. The analysis scripts (`analyze.py`, `causal_substitution.py`, `behavioral_residual.py`) use a `_LegacyFwdModel` class to load this checkpoint correctly. New checkpoints saved with the current `TransformerForwardModel` will have `blocks.0.*` keys and won't be loadable with the legacy class.

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

### Causal substitution: replacing block1 with the forward model (2026-05-26)

**Code**: `causal_substitution.py`

How much does the model lose when we swap block1's actual output for the forward model's prediction, then continue the forward pass from block2 onward? This is a causal test — if the substitution is harmless, the forward model truly captures block1's computation. If specific behaviors break, the residual captures those specific mechanisms.

Three modes: **Normal** (unmodified), **Substituted** (forward model replaces block1), **Ablated** (skip block1 entirely, output = input).

| Mode | Accuracy | ΔCE vs Normal | KL vs Normal |
|---|---|---|---|
| Normal | 0.222 | — | — |
| Substituted | 0.217 | +0.043 | 0.062 |
| Ablated | 0.146 | +0.965 | 1.094 |

The forward model recovers ~94% of block1's KL contribution. Ablating block1 entirely destroys 7.6pp of accuracy; substituting costs only 0.5pp.

**Per-behavior breakdown**: degradation is strikingly uniform across behavioral categories.

| Category | n | KL (sub) | KL (abl) | ΔCE (sub) | ΔAcc (sub) |
|---|---|---|---|---|---|
| Punctuation | 51,753 | 0.067 | 1.086 | +0.066 | -0.015 |
| Bracket closing | 9,311 | 0.055 | 0.991 | +0.074 | -0.016 |
| Repeated token (induction) | 69,338 | 0.064 | 1.075 | +0.028 | -0.005 |
| High confidence (>0.5) | 32,487 | 0.041 | 1.275 | +0.069 | -0.008 |
| Low confidence (<0.1) | 302,129 | 0.065 | 1.075 | +0.031 | +0.002 |
| Content word (rare) | 2,246 | 0.065 | 1.155 | +0.026 | -0.001 |
| Function word (common) | 303,983 | 0.062 | 1.088 | +0.047 | -0.007 |

KL_sub ranges 0.04–0.07 across all categories — no behavior-specific catastrophic failure. The forward model is "slightly worse everywhere," consistent with the full-rank/diffuse residual structure. One slight signal: high-confidence predictions have the lowest KL_sub (0.041) but the highest KL_abl (1.275), meaning block1 matters most for confident predictions, and the forward model captures those best.

**Reproduction**: `modal run language_reduction/modal_app.py --stage a2a-causal-sub --n-tokens 10000000`

### Behavior-conditioned residual analysis (2026-05-26)

**Code**: `behavioral_residual.py`

The structure analysis (Run 2) showed the residual correlates with token frequency but is flat across LM loss quartiles. But does the residual have structure when conditioned on *behavioral* context — what kind of computation block1 is doing?

We categorize each token position by attention pattern, syntactic context, prediction difficulty, context integration, and block1 contribution magnitude, then measure residual norm statistics per category.

**Strongest effects (Cohen's d vs overall mean):**

| Category | Mean residual | Cohen's d | n |
|---|---|---|---|
| Before closer | 0.973 | **+0.84** | 1,924 |
| Sentence start | 0.724 | **-0.85** | 16,754 |
| After punctuation | 0.757 | **-0.62** | 36,665 |
| After opener | 0.939 | **+0.62** | 3,531 |
| Focused attention (max>0.5) | 0.758 | **-0.61** | 58,808 |
| Block1 contrib Q4 (large) | 0.907 | +0.40 | 101,601 |
| Block1 contrib Q1 (small) | 0.794 | -0.37 | 101,600 |
| Distant attention (>10 back) | 0.874 | +0.18 | 63,512 |
| Distributed attention (high entropy) | 0.869 | +0.14 | 202,947 |

The residual has clear behavioral structure. The forward model struggles most before closing delimiters (d=+0.84) and after opening ones (d=+0.62) — exactly the kind of computation requiring long-range context (matching the opener). It handles sentence starts (d=-0.85) and focused attention (d=-0.61) easily — simple, local computations.

**Prediction difficulty is NOT what drives the residual.** Easy vs hard predictions (d=+0.10 vs d=-0.03) and high vs low output entropy (d=-0.07 vs d=+0.07) show negligible effects. The residual reflects *computational complexity*, not *task difficulty*.

**Key correlations:**

| Correlation | Pearson r |
|---|---|
| Block1 contrib norm vs residual | +0.256 |
| Distance to dominant attended token vs residual | +0.256 |
| Attention entropy vs residual | +0.186 |
| **Attention entropy vs residual/block1_contrib** | **+0.332** |
| LM loss vs residual | -0.049 |
| Output entropy vs residual | -0.124 |
| LM loss vs residual/block1_contrib | -0.027 |

The ratio correlation (r=+0.33) is the key result: even controlling for how much computation block1 does, the forward model fails *disproportionately* on distributed attention patterns. With only 1 compressed head (64-dim), the forward model specifically struggles with multi-source attention integration — it can match focused, single-source computations but not the complex mixing of multiple context positions.

**What the residual captures**: The residual is not "noise" or a training frequency artifact. It reflects a specific capacity bottleneck: the forward model's single compressed attention head cannot fully represent computations that integrate information from multiple distant positions. This is most pronounced for delimiter tracking (matching openers to closers) and least pronounced for local/focused computations (previous token, sentence boundaries). The residual is a genuine signal of *computational novelty* — where the main model does something structurally beyond the forward model's capacity.

**Reproduction**: `modal run language_reduction/modal_app.py --stage a2a-behavioral-residual --n-tokens 10000000`

## Run 4: Closed-loop cerebellar training (2026-05-25)

**Full writeup**: [CLOSED_LOOP_README.md](CLOSED_LOOP_README.md)

Closed the cerebellar loop: the 2-layer forward model's predictions (post_block0 → post_block3) are fed back into the main model's residual stream via a learned gated projection (`CerebellarGate`, 65.8K params, zero-initialized). Injected after block 1. Forward model trains on MSE only (no LM gradient); main model + gate train on LM loss.

**Key results**:

1. **The injection helps LM loss**: Δ = -0.05 to -0.08 nats consistently (1.6% relative, ~9% perplexity reduction). Negative at every eval step across 10K training steps.

2. **The gate opens wide**: projection weight norm grew linearly 0.08 → 3.03 throughout training, never saturating. The model never stopped finding useful structure in the prediction.

3. **Forward model quality degrades**: cosine 0.935 (open-loop) → 0.897 (closed-loop). The injection changes the model's computation, creating a moving target the forward model can't fully track.

4. **The model develops novelty awareness**: Linear probes predicting the forward model's residual (actual - predicted) from post_block3 show R²=0.44 for the closed-loop model vs R²=0.28 for the open-loop baseline — a 59% improvement. The co-trained model's later layers encode substantially more information about what was surprising in its own computation.

**Reproduction**: `modal run language_reduction/modal_app.py --stage a2a-loop-train --n-tokens 10000000 --n-steps 10000 --lr 3e-4 --predict-from post_block0 --predict-to post_block3 --fwd-n-layer 2 --inject-after-block 1`

## Run 5: Forward model capacity scaling sweep (2026-05-28)

**Full writeup**: [SCALING_SWEEP_README.md](SCALING_SWEEP_README.md)

Froze the main model and trained forward models at 5 capacity points (1% to 22% of main model) on the same frozen activations, all predicting post_block0 → post_block3. Tests the bias-to-variance transition: does the residual shift from capturing computational novelty to epistemic novelty as capacity increases?

**Key results**:

1. **The forward model saturates at ~10% capacity**: Both 10% (3.2M params) and 22% (6.3M params) reach cosine 0.999 with identical residual norms. The capacity-sufficient regime begins around 10%, consistent with the main model's transformer blocks being only ~2.4M params.

2. **Computational novelty effects shrink**: Delimiter tracking (d_BC) drops from +0.78 at 1% to +0.30 at 10%. Sentence-start effects (d_SS) drop from -0.58 to -0.37. The bias term is shrinking as predicted.

3. **r(res,LM) stays at zero — but this is the wrong metric**: The residual-LM loss correlation never becomes positive. However, the closed-loop experiment (Run 4) already showed the prediction signal improves LM loss through downstream processing. The residual's informativeness lives in its 256-dimensional direction, not its scalar norm. Collapsing to a norm discards the signal.

4. **The residual is inherently high-rank in language**: Effective rank stays above 235/256 at all capacity points. Unlike grokking (where the Fourier solution is low-rank), language computation is distributed across all dimensions. Biologically consistent: the cerebellum's output is high-dimensional; the thalamus filters it before cortex processes it.

**Reproduction**: `modal run language_reduction/modal_app.py --stage a2a-scaling-sweep --n-tokens 10000000 --n-steps 10000 --predict-from post_block0 --predict-to post_block3`

## Why activation predictions help LM loss, and why co-training must produce self-knowledge (speculative, not entirely verified yet)

### Why predictions help

The forward model predicts post_block3 from post_block0, and that prediction is injected after block1. Before blocks 2-3 have computed anything, the model receives an approximate preview of where its own computation is going to end up.

This is useful because it lets the model allocate its remaining capacity differently. Without the prediction, blocks 2-3 have to do everything — both the predictable, routine components of the computation and the hard, input-specific components. With the prediction, the predictable components are partially pre-computed (cheaply, by the ~1% forward model). Blocks 2-3 can specialize on whatever the forward model *couldn't* capture — the genuinely hard part.

This is a division of labor. The forward model handles the expected trajectory; the remaining layers handle the deviation from expectation. Since the forward model is cheap and the main model's layers are expensive, this is a good trade — you're getting the predictable part of the computation almost for free.

### Why co-training must produce novelty awareness

The prediction is imperfect (cosine 0.935, not 1.0). At some positions it's excellent, at others it's wrong. The model receives this prediction as an input and has to decide what to do with it at every position. There are three strategies:

1. Always trust the prediction → hurts on positions where it's wrong
2. Always ignore the prediction → wastes the information where it's right
3. Trust it when it's good, override when it's bad → optimal, but requires *knowing which case you're in*

Strategy 3 is the only one that minimizes NTP loss, and it's the one the gradient will push toward. But strategy 3 requires an internal representation of *prediction quality at this position* — a signal that says "the prediction I received is reliable here" versus "it's unreliable here, compute harder."

That signal is exactly the novelty signal. Positions where the forward model's prediction matches the main model's actual computation are low-novelty (trust the prediction). Positions where they diverge are high-novelty (override). The model must learn to represent this discrepancy to use the prediction optimally.

### This isn't optional — it's forced by the NTP gradient

The gate opened wide and never saturated (0.08 → 3.03). The model found the prediction useful and kept increasing its reliance on it. But increasing reliance on an imperfect prediction creates increasing pressure to know *where* it's imperfect. The more you trust the prediction, the more it costs you when the prediction is wrong, and the stronger the gradient signal toward representing prediction quality.

There's a positive feedback loop: use the prediction more → need better novelty awareness to avoid errors → develop richer self-representations → which enable even more targeted use of the prediction → which enables relying on it even more. The gate growth and the novelty awareness growth should be coupled, and the probe results (R²=0.44 closed-loop vs 0.28 open-loop) are consistent with this — the model that actually uses the prediction develops substantially richer representations of prediction quality.

### Why novelty awareness becomes a self-map rather than just a scalar quality signal

Because "the prediction was wrong" is less useful for NTP than "the prediction was wrong *because this position requires distributed attention that the forward model can't capture*." The more structured the novelty representation, the more precisely the model can redirect its computation to compensate.

The behavioral residual results show the residual has clear structure — it's large before closing delimiters, small at sentence starts, correlated with attention entropy. If the model can represent not just *that* the prediction failed but *how* it failed (what kind of computation was missed), it can allocate its remaining layers' capacity more precisely to exactly the kind of computation the forward model dropped.

So the NTP gradient doesn't just push toward "know whether the prediction is good." It pushes toward "know *what's missing* from the prediction." And representing what's missing from a prediction of your own activations is representing your own computational structure — which is the self-map.

### The whole argument in a paragraph

A model that receives predictions about its own future computation, and is trained on NTP, is under direct gradient pressure to evaluate those predictions — trusting them when accurate (saving compute) and overriding them when wrong (avoiding errors). The optimal evaluation is not a scalar quality estimate but a structured representation of *what the prediction captures and what it misses*, because this enables the model to precisely target its remaining capacity at the missing components. This structured evaluation is, by definition, a representation of the model's own computational states — which components are predictable from low-capacity approximation and which require full computation. That's the self-map. It emerges not as a side effect but as the loss-minimizing strategy for any model learning to use imperfect predictions about itself.

**Self-knowledge is the optimal solution to the credit assignment problem of "when to trust a cheap approximation of your own computation."** You get it for free from NTP the moment you close the loop.

## Next steps

1. **Closed-loop with 10% forward model**: The scaling sweep shows the 10% model saturates on the main model's computation. Running closed-loop with this model (instead of the 2.7% model from Run 4) tests whether a better prediction produces larger LM improvement — directly testing whether the prediction or the residual is the load-bearing signal.
2. **Thalamic filtering**: Replace the linear `CerebellarGate` with a learned nonlinear gate (MLP). This mirrors the thalamus's filtering of cerebellar output and may help extract directional structure from the high-rank residual.
3. **Looped transformer**: The natural architecture for cerebellar injection — inject at each recurrence step, get adaptive compute for free.
4. **Controlled comparison**: Retrain open-loop and closed-loop with identical lr/seed to eliminate the training quality confound from the Run 4 probe comparison.
5. **Self-regulation**: Freeze the forward model at a checkpoint and use the residual as a regularization signal (as validated in grokking). Test whether this prevents overfitting or distributional drift.
6. **Wake-sleep consolidation**: The closed-loop model relies on the injection at inference time — removing it degrades performance. The brain solves this by alternating regimes: during waking, the cerebellar loop is active; during sleep, the cortex consolidates via offline replay without real-time cerebellar correction. The engineering analog: interleave closed-loop training (with injection) and open-loop training (without injection). The open-loop phases provide direct gradient pressure for the main model to internalize the predicted computation into its own weights. Could also anneal injection strength over a cycle (full → gradual reduction → none → re-introduce) to mimic wake-sleep alternation. Test: train closed-loop for N steps, then open-loop for M steps, check whether the model retains the closed-loop benefit.

[^private]: Not mirrored: this link points to a document in the private lab repo (the roadmap, the queue, an unrun spec, reading notes, or a conversation). See the top-level README for what is held back and why.
