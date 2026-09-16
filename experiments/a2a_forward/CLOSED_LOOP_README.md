# Closed-Loop Cerebellar Training (2026-05-25)

**Idea doc**: [ideas/activation_to_activation_forward.md](../../../../ideas/activation_to_activation_forward.md)
**Open-loop experiments**: [README.md](README.md)
**Neuroscience conversations**: conversations/cerebellum/[^private]

## Goal

Close the cerebellar loop: feed the forward model's predictions back into the main model's residual stream during training. Test whether the model (a) finds the prediction signal useful for language modeling, and (b) develops self-referential representations — internal knowledge of its own computational states.

This is the first experiment testing the feedback loop described in the idea doc. Prior experiments (README.md runs 1-3) established that a small forward model can predict the main model's activations with high fidelity (0.935 cosine for the 3-layer gap). This experiment asks: what happens when you wire that prediction back in?

## Architecture

**Main model**: 4-layer, 4-head, 256-dim GPT-2 (28.9M params). Unchanged from prior experiments except for a new `cerebellar_fn` parameter in `GPT.forward()` that accepts an optional injection into the residual stream.

**Forward model**: 2-layer transformer (1 head/layer, 64-dim, 660K params). Predicts post_block3 from post_block0 — a compressed guess at the model's full computation from its first layer's output.

**CerebellarGate**: Learned linear projection (256→256, 65.8K params) that transforms the forward model's prediction into a useful signal for the residual stream. Zero-initialized so the injection starts at exactly zero. The projection magnitude grows from zero as gradient descent discovers useful structure.

**Injection point**: After block 1, before block 2 (middle of the model). Analogous to where cerebellar returns arrive in cortex (middle layers, same laminar address as feedforward sensory input). Blocks 2-3 process the augmented stream through their own weights.

**Why the prediction, not the residual**: The cerebellum sends its predicted cortical state back to cortex. The cortex's own processing implicitly compares this against its actual state. Injecting the prediction is richer — the model can compute the residual itself (it has access to both its own state and the prediction), plus any other function of the pair. Injecting the residual pre-computes one specific comparison and discards the prediction.

## Training

**Separate objectives (biologically faithful)**:
- Forward model: MSE loss on post_block3 activations only. Learns from its own prediction errors (climbing fiber analog). No gradient from LM loss — the forward model's predictions are detached before entering the gate.
- Main model + gate: LM cross-entropy loss. The gate's projection parameters receive LM gradient through the injection, so the model learns how much and what kind of prediction signal to incorporate.
- Main model lr: 3e-4, forward model lr: 1e-3, both AdamW with weight decay 0.01.

**Single-pass implementation**: The `cerebellar_fn` closure runs the forward model on post_block0 (detached), caches the prediction for MSE loss, then returns `gate(fwd_pred.detach())` as the injection. One forward pass computes both losses.

**On-policy targets**: The forward model's MSE target is post_block3 from the closed-loop computation (with injection active). The forward model learns to predict what the model computes given the injection, not what it would compute without it. Initially (gate ≈ 0) the target matches the open-loop case; as the gate opens, co-adaptation happens gradually.

## Results

### The injection helps language modeling

| Step | lm_loop | lm_base | Δ |
|---|---|---|---|
| 0 | 10.619 | 10.619 | -0.001 |
| 200 | 6.774 | 6.804 | -0.030 |
| 600 | 6.188 | 6.333 | **-0.145** |
| 2000 | 5.481 | 5.548 | -0.067 |
| 5000 | 5.195 | 5.239 | -0.044 |
| 8000 | 5.236 | 5.307 | -0.071 |
| 9999 | 5.469 | 5.556 | **-0.087** |

Δ is negative at every single eval step — the injection consistently reduces val LM loss. Final Δ = -0.087 nats on a base of ~5.5 (1.6% relative improvement, ~9% perplexity reduction). The peak benefit (-0.145) occurs early when the forward model is most accurate, settles to -0.04 to -0.08 for the remainder of training, and trends upward again late.

The magnitude is modest but the consistency is the signal: across 50 eval points over 10K steps, the injection never hurts.

### The gate opens wide and never saturates

- gate_norm (projection weight norm): 0.08 → 3.03 (38× increase, linear growth, never plateaued)
- injection_norm: peaked at 5.4 (step 600), dropped to 3.4 (step 4000), rose again to 3.9 (step 10000)

The model never stopped finding more useful structure to extract from the prediction. The injection norm's U-shape — heavy reliance early, partial self-sufficiency mid-training, renewed reliance late — suggests different phases of how the model uses the signal.

### Forward model quality slowly degrades

| Metric | Open-loop (Run 3) | Closed-loop |
|---|---|---|
| Final cosine sim | 0.935 | 0.897 |
| fwd_pred vs post_block3 | 0.935 | 0.927 (with inj) / 0.897 (no inj) |

The closed-loop forward model achieves lower cosine similarity than the open-loop one. The injection changes what the model computes, and the forward model can't fully track the more complex co-adapted computation. This is expected and biologically consistent — the cerebellum is always slightly behind the cortex after learning.

### Self-map probes: the closed-loop model develops novelty awareness

**Probe design**: Train linear probes to predict various targets from post_block3. Compare the closed-loop model against the open-loop baseline (Run 3).

**post_block3 → residual (actual - predicted): the key result**

| Model | R² | Cosine |
|---|---|---|
| Open-loop | 0.276 | 0.527 |
| **Closed-loop (with inj)** | **0.406** | **0.630** |
| **Closed-loop (no inj)** | **0.438** | **0.656** |

The closed-loop model's later layers encode **59% more variance** in the prediction residual than the open-loop model's (R² 0.44 vs 0.28). The residual represents "what the forward model's capacity couldn't represent" — genuinely novel computation. The closed-loop model has developed representations that are more informative about what was surprising in its own processing.

The without-injection condition (0.438) is slightly better than the with-injection condition (0.406), suggesting this self-knowledge has been partially internalized into the model's weights rather than merely carried by the live injection signal.

**post_block3 → post_block0 (early-layer self-knowledge)**

| Model | R² | Cosine |
|---|---|---|
| Open-loop | 0.757 | 0.870 |
| Closed-loop (with inj) | 0.778 | 0.878 |
| Closed-loop (no inj) | 0.774 | 0.875 |

Small but consistent +0.02 R² improvement. The closed-loop model's later layers encode marginally more information about early layers than the open-loop model's.

**post_block3 → fwd_pred (confounded)**

| Model | R² | Cosine |
|---|---|---|
| Open-loop | 0.848 | 0.919 |
| Closed-loop (with inj) | 0.760 | 0.879 |
| Closed-loop (no inj) | 0.751 | 0.872 |

The open-loop model is better here, but this is confounded: fwd_pred is a prediction of post_block3, so the probe partly recovers the correlation between them. The open-loop forward model is more accurate (cosine 0.935 vs 0.897), making this probe easier. This metric doesn't cleanly test the self-map hypothesis.

### Caveats

- **lr difference in this run**: This run used open-loop lr=1e-4, closed-loop lr=3e-4. A controlled retrain ([CONTROLLED_RETRAIN_README.md](CONTROLLED_RETRAIN_README.md)) with identical lr/seed confirmed that the probe results (R²=0.42 vs 0.26) and injection benefit (Δ ≈ -0.08) are robust and not lr artifacts.
- **Small scale**: 28.9M params on 10M tokens. The model is capacity-bottlenecked; the prediction signal's value may change at scale.

## Interpretation

### What the model uses the injection for

The prediction of post_block3 from post_block0 is a compressed summary of the model's full computation, arriving at the middle of the residual stream. Blocks 2-3 receive both their normal input (post_block1) and a preview of their own output. This preview can:
- Pre-activate the right attractor basin (helping blocks 2-3 converge to better representations)
- Provide a reference for implicit comparison (blocks 2-3 can extract the difference between their input and the prediction)

The consistent LM improvement suggests the preview is genuinely useful — not just noise that the model tolerates.

### The self-knowledge result

The closed-loop model's layers encode more information about its own prediction residuals — a specific form of self-knowledge: "I know what I computed that a compressed model of me wouldn't have expected." A controlled retrain ([CONTROLLED_RETRAIN_README.md](CONTROLLED_RETRAIN_README.md)) confirmed this is real (not an lr artifact) and showed the self-knowledge is:

- **Distributed across all layers via backprop** — present even at post_block0, before the injection point. The injection changes the loss landscape, and gradients flow backward through all layers, causing the entire model to reorganize to complement the prediction.
- **Directional, not magnitude-based** — the vector probe gap (Δ R²=+0.18) is 6× the scalar gap (Δ R²=+0.03). The model encodes *what kind of computation was missed*, not *how much*.

This emerged from co-training alone, with no explicit objective encouraging self-referential representations. The biological parallel is the cortex developing representations that track cerebellar prediction errors — the pre-reflective "something feels off" signal.

### Why the forward model degrades

The closed-loop creates a moving target: the injection changes what blocks 2-3 compute, which changes post_block3, which changes the forward model's target. The forward model must track a co-evolving system rather than a fixed one. A biological analog: after an insight shifts cortical processing, familiar things briefly feel strange because the cerebellum hasn't updated its model of the cortex yet.

## Files

| File | Purpose |
|---|---|
| `forward_model.py` | `CerebellarGate` (learned gated projection for injection) |
| `stages.py` | `a2a_loop_train` — closed-loop co-training with eval |
| `analyze.py` | `a2a_loop_analyze` — post-hoc self-map probes (works for both open-loop and closed-loop) |
| `CLOSED_LOOP_README.md` | This file |

## Reproduction

```bash
cd experiments/

# Closed-loop training
modal run a2a_forward/stages.py::loop_train \
  --n-tokens 10000000 --n-steps 10000 --lr 3e-4 \
  --predict-from post_block0 --predict-to post_block3 \
  --fwd-n-layer 2 --inject-after-block 1

# Post-hoc analysis (closed-loop)
modal run a2a_forward/analyze.py::loop_analyze \
  --n-tokens 10000000 --predict-from post_block0 --predict-to post_block3 \
  --fwd-n-layer 2 --inject-after-block 1

# Post-hoc analysis (open-loop baseline)
modal run a2a_forward/analyze.py::loop_analyze \
  --n-tokens 10000000 --predict-from post_block0 --predict-to post_block3 \
  --fwd-n-layer 2 --inject-after-block 1 --open-loop
```

## Modal volume

Results saved to `language-reduction-data` volume:

```
/data/a2a_forward/
├── loop_L2/                           # Closed-loop run
│   └── post_block0_to_post_block3/
│       └── inject1/P_10000000/
│           ├── model.pt
│           ├── fwd_model.pt
│           ├── gate.pt
│           ├── results.json
│           └── loop_analysis.json
└── transformer_L2/                    # Open-loop baseline (Run 3)
    └── post_block0_to_post_block3/
        └── P_10000000/
            ├── model.pt
            ├── fwd_model.pt
            ├── results.json
            └── openloop_analysis.json
```

## Next steps

1. ~~**Controlled comparison**~~: Done — see [CONTROLLED_RETRAIN_README.md](CONTROLLED_RETRAIN_README.md). Self-knowledge confirmed, not an lr artifact.
2. **Wake-sleep consolidation**: The model becomes dependent on the injection (0.11 nats worse without it). Interleave closed-loop and open-loop training to force internalization.
3. **Directional causal test**: The self-knowledge is directional, not magnitude-based. Steer/patch along residual *direction* clusters to test whether directional self-knowledge is functionally used.
4. **Looped transformer**: The natural architecture for this — inject at each recurrence step, get adaptive compute for free.
5. **Scaling**: Larger models where the computation is more complex and the capacity gap between main and forward model is more pronounced.

[^private]: Not mirrored: this link points to a document in the private lab repo (the roadmap, the queue, an unrun spec, reading notes, or a conversation). See the top-level README for what is held back and why.
