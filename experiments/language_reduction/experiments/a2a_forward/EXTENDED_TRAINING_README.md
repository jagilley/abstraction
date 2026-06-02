# Extended Co-Training: 50K Steps (2026-06-01)

**Prior experiment**: [CONTROLLED_RETRAIN_README.md](CONTROLLED_RETRAIN_README.md)
**Idea doc**: [ideas/activation_to_activation_forward.md](../../../../ideas/activation_to_activation_forward.md)

## Why this exists

The controlled retrain (Run 6) trained for 10K steps (~9 epochs on 10M tokens). At that point the injection benefit was -0.08 nats, the dependency was 0.11 nats, and the self-knowledge probe gap was Δ R²=+0.18. But all of these were single-timepoint measurements. This experiment trains for 50K steps (~45 epochs) to observe long-run dynamics:

1. Does the injection benefit plateau, grow, or decay?
2. Does co-adaptation converge to an equilibrium or continue evolving?
3. Does the self-knowledge probe gap continue to grow?

A secondary question: does the closed-loop model overfit differently from the open-loop model? The co-adaptive dynamic means the same data produces different learning pressure each epoch (because the forward model's prediction evolves), which could in principle delay overfitting. However, validation loss on a static dataset is a limited instrument for detecting differences in representational quality (see the "validation loss and continual learning" discussion — the same representational capacity compressing the same fixed distribution should converge to similar val loss regardless of internal organization).

## Design

Same as controlled retrain: identical lr (3e-4), seed (42), initial weights, training data order. The only difference is whether the cerebellar loop is closed. Extended to 50K steps with intermediate checkpoints every 10K steps.

| Parameter | Value |
|---|---|
| Main model | 4L, 4H, 256D GPT (28.9M params) |
| Forward model | 2L transformer, 1H, 64D (660K params) |
| Forward model gap | post_block0 → post_block3 |
| Injection point | After block 1 |
| lr (both models) | 3e-4 |
| fwd_lr | 1e-3 |
| Seed | 42 |
| Steps | 50,000 (~45 epochs) |
| Tokens | 10M (FineWeb-Edu, τ=0.0) |

## Results

### The injection benefit grows monotonically — 6× over training

| Step | Injection benefit (Δ nats) | Fwd cosine | Dependency (CL_no_inj − OL) | Gate norm |
|---|---|---|---|---|
| 5,000 | -0.048 | 0.929 | +0.072 | 2.33 |
| 10,000 | -0.082 | 0.900 | +0.109 | 3.01 |
| 15,000 | -0.121 | 0.875 | +0.139 | 3.24 |
| 20,000 | -0.156 | 0.857 | +0.189 | 3.25 |
| 25,000 | -0.211 | 0.844 | +0.235 | 3.20 |
| 30,000 | -0.242 | 0.838 | +0.285 | 3.15 |
| 35,000 | -0.308 | 0.832 | +0.335 | 3.11 |
| 40,000 | -0.363 | 0.826 | +0.393 | 3.07 |
| 45,000 | -0.413 | 0.822 | +0.445 | 3.04 |
| 50,000 | -0.481 | 0.815 | +0.517 | 3.02 |

The injection benefit at 50K (-0.48 nats) is 6× the benefit at 10K (-0.08), with no sign of saturation. The forward model's prediction gets steadily worse (cosine 0.93 → 0.82) while the benefit grows — the model extracts increasing value from a less accurate prediction.

### No differential overfitting

| Step | Open-loop gap (train−val) | Closed-loop gap (train−val) |
|---|---|---|
| 5,000 | -0.727 | -0.727 |
| 10,000 | -1.524 | -1.525 |
| 20,000 | -2.698 | -2.700 |
| 30,000 | -3.611 | -3.623 |
| 40,000 | -4.253 | -4.253 |
| 50,000 | -4.864 | -4.867 |

Both models overfit at the same rate. The closed-loop architecture does not protect against overfitting on a static dataset. Final val loss is essentially identical (open: 7.36, closed with injection: 7.40). This is expected: both models have the same capacity compressing the same fixed distribution, and validation loss measures compression quality, not representational structure.

### Self-knowledge probes at 50K

**Vector probes (Δ R² = closed − open):**

| Layer | 10K (controlled retrain) | 50K (this run) |
|---|---|---|
| post_block0 | +0.185 | +0.094 |
| post_block1 | +0.200 | +0.098 |
| post_block2 | +0.185 | +0.057 |
| post_block3 | +0.161 | -0.005 |

The directional self-knowledge persists at early layers (post_block0: +0.094) but narrows compared to 10K. At post_block3 the gap disappears. This may reflect the overfit regime: both models' late-layer representations become dominated by memorized patterns that wash out the directional distinction.

**Scalar probes flip sign:**

| Layer | 10K Δ (residual_norm) | 50K Δ (residual_norm) |
|---|---|---|
| post_block0 | +0.026 | -0.384 |
| post_block3 | +0.035 | -0.318 |

The open-loop model at 50K encodes the forward model's residual norm much better than the closed-loop model — the opposite of the 10K result. The most likely explanation: residual norm correlates with token frequency (rare tokens have higher residual norm, per the behavioral residual analysis), and the overfit open-loop model has memorized token-level statistics that happen to correlate with this.

### Forward model quality

| Model | Cosine | MSE |
|---|---|---|
| Open-loop | 0.863 | 0.862 |
| Closed-loop | 0.817 | 2.302 |

Both forward models are worse than at 10K (open: 0.915 → 0.863, closed: 0.900 → 0.817). The main models' computation has grown more complex (more overfit, more input-specific patterns) and harder for a 660K-param forward model to approximate.

## Interpretation

### The injection is a structured reference frame, not a literal preview

At 10K steps, the injection looks like a shortcut — the forward model provides a decent preview of post_block3, and the main model uses it to converge faster. At 50K steps, this interpretation doesn't hold. The forward model's prediction is poor (cosine 0.82), yet the model extracts 6× more value from it than it did when the prediction was much better (cosine 0.93 at 10K).

What the model appears to be using is the prediction's *structure* — its consistent, learned approximation of the main model's computation — as a reference frame for organizing its own representations. The value isn't in the prediction being right; it's in it being a stable, structured signal against which the model can orient. This reframes the injection from "division of labor" (the forward model handles routine computation) to something more like "an external coordinate system that the model continuously discovers new uses for."

### Deepening co-specialization is the equilibrium

The injection benefit, dependency, and forward model cosine all evolve monotonically without saturation over 50K steps. This is not convergence to a fixed point — the co-adaptation keeps deepening. The model keeps finding new ways to use the prediction, offloading more computation to the forward model, and the forward model keeps falling further behind. This dynamic shows no sign of stabilizing.

### Validation loss is the wrong instrument here

The identical overfitting trajectories don't mean the two models are equivalent. They mean both models are equally good at compressing a fixed 10M-token distribution, which is a separate question from whether their representations are organized differently. The growing injection benefit (0.48 nats, visible only when you compare with-injection vs without-injection) is invisible to standard val loss but reflects a large and growing structural difference between the two models.

## Files

| File | Purpose |
|---|---|
| `extended_training.py` | Extended co-training: 50K steps with intermediate checkpoints |
| `EXTENDED_TRAINING_README.md` | This file |

## Reproduction

```bash
cd experiments/
modal run --detach language_reduction/modal_app.py --stage a2a-extended-training \
  --n-tokens 10000000 --n-steps 50000 \
  --predict-from post_block0 --predict-to post_block3 --fwd-n-layer 2
```

## Modal volume

Results saved to `language-reduction-data` volume:

```
/data/a2a_forward/extended/
└── post_block0_to_post_block3/
    └── inject1/P_10000000/
        ├── results.json
        ├── open_loop/
        │   ├── history.json
        │   └── step_*/          # Checkpoints at 10K, 20K, 30K, 40K, 50K
        │       ├── model.pt
        │       └── fwd_model.pt
        └── closed_loop/
            ├── history.json
            └── step_*/
                ├── model.pt
                ├── fwd_model.pt
                └── gate.pt
```
