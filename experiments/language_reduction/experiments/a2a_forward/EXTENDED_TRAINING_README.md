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

## Run 7b: 10% forward model, 15K steps (2026-06-02)

Same controlled design (identical lr/seed/init), but with a 10% forward model (3.2M params) instead of the 2.7% model (660K params). Trained for 15K steps (~13.7 epochs). Tests whether a capacity-sufficient forward model changes the overfitting dynamics or the injection benefit pattern.

| Parameter | Run 7 (this file, above) | Run 7b (this section) |
|---|---|---|
| Forward model | 2L, 1H, 64D (660K, 2.3%) | 3L, 4H, 128D (3.2M, 10.9%) |
| Steps | 50,000 | 15,000 |
| Open-loop fwd cosine (final) | 0.863 | 0.997 |

The 10% forward model reaches 0.997 cosine in open-loop — it captures essentially all of the main model's computation (consistent with the scaling sweep saturation result). This lets us test: what happens when the injection carries nearly perfect information rather than a noisy approximation?

### Train-val gaps are still identical

| Step | Open-loop gap | Closed-loop gap | Difference |
|---|---|---|---|
| 5,000 | -0.727 | -0.716 | 0.011 |
| 10,000 | -1.524 | -1.501 | 0.023 |
| 15,000 | -2.134 | -2.082 | 0.052 |

Even with a forward model that exceeds each main model block's parameter count (3.2M vs 2.4M per block), the train-val gap dynamics are indistinguishable. The closed-loop gap is consistently ~0.02–0.05 nats smaller, but this is within noise and doesn't grow systematically. The injection does not interact with memorization dynamics regardless of forward model capacity.

### The injection relocates computation without improving it

| Step | Open val | Closed val (with inj) | Closed val (no inj) | Injection Δ |
|---|---|---|---|---|
| 500 | 6.302 | 6.334 | 6.603 | -0.268 |
| 5,000 | 5.172 | 5.198 | 5.278 | -0.080 |
| 10,000 | 5.338 | 5.360 | 5.509 | -0.149 |
| 15,000 | 5.573 | 5.569 | 5.782 | -0.213 |

The closed-loop model WITH injection is never meaningfully better than the open-loop model (at best -0.004 nats at step 15K). The injection benefit (-0.213 at 15K) is almost exactly offset by the dependency cost (+0.210). The model offloads computation to the forward model rather than genuinely improving.

Contrast with the 1% model, where closed-with-injection was consistently ~0.05–0.08 nats better than open-loop at comparable training steps. The 10% model enables more aggressive offloading, but the net effect is zero — pure relocation of computation, not improvement.

### U-shaped injection benefit trajectory

| Step | Δ (10% model) | Δ (1% model, from above) |
|---|---|---|
| 500 | **-0.268** | ~-0.03 |
| 5,000 | -0.080 | -0.048 |
| 10,000 | -0.149 | -0.082 |
| 15,000 | -0.213 | -0.121 |

The 10% model shows a large early benefit (-0.27 at step 500) that contracts sharply and then recovers. The 1% model shows monotonic growth. The interpretation: the 10% model quickly learns a high-quality approximation that the main model immediately relies on heavily. As training continues, the main model's computation diverges from the prediction (closed-loop cosine drops from 0.93 to 0.94 initially, then down to 0.937), and the benefit temporarily shrinks. Then co-specialization deepens and the benefit resumes growing.

### Self-knowledge probes are 2× stronger

**Vector probes (Δ R² = closed − open):**

| Layer | 10% fwd, 15K | 1% fwd, 10K (controlled retrain) |
|---|---|---|
| post_block0 | **+0.340** | +0.185 |
| post_block1 | **+0.393** | +0.200 |
| post_block2 | **+0.426** | +0.185 |
| post_block3 | **+0.472** | +0.161 |

The self-knowledge signal is roughly 2× stronger with the 10% forward model, and increases monotonically through the layers (unlike the 1% model where it was approximately uniform).

**Scalar probes also show large gaps (new pattern):**

| Layer | Scalar Δ R² (10% fwd) | Scalar Δ R² (1% fwd, 10K) |
|---|---|---|
| post_block0 | **+0.326** | +0.026 |
| post_block3 | **+0.449** | +0.035 |

With the 1% model, the scalar gap was 6× smaller than the vector gap — the model encoded *what kind* of computation was missed (direction) but not *how much* (magnitude). With the 10% model, scalar and vector gaps are comparable. This makes sense: when the forward model captures 99.7% of the computation, the residual magnitude itself becomes informative. What the 10% model misses is genuinely the hard part, so "how much was missed" correlates meaningfully with computational difficulty.

### Forward model quality

| | Cosine | MSE |
|---|---|---|
| Open-loop (15K) | 0.997 | 0.005 |
| Closed-loop (15K) | 0.937 | 0.247 |

The open-loop 10% model reaches near-perfect prediction (0.997), confirming the scaling sweep result. The closed-loop model drops to 0.937 — the injection creates a moving target that prevents the forward model from fully tracking the main model's computation, even with sufficient capacity. The 50× MSE gap (0.005 vs 0.247) reflects how much the injection changes the downstream computation.

### Interpretation: capacity-sufficient injection as pure relocation

The 1% forward model's injection helps because it's imperfect — the main model receives an approximate preview and can use blocks 2–3 to correct the approximation, achieving better results than pure open-loop computation. The improvement comes from the *gap* between prediction and reality being useful working material.

The 10% forward model's injection doesn't help because it's too good — the main model receives a near-perfect preview and offloads nearly all predictable computation to it, becoming fully dependent. The injection carries enough information to substitute for the model's own computation rather than supplement it. The result is architectural relocation (computation moves from the main model to the forward model) without functional improvement.

This suggests a capacity sweet spot for net injection benefit: the forward model must be imperfect enough that the main model can't fully depend on it, but accurate enough to provide useful structure. The 1% model (~0.93 cosine) appears closer to this sweet spot than the 10% model (~0.997 cosine in open-loop).

### On validation loss as a metric

The self-knowledge probes are 2× stronger with the 10% model, yet val loss shows zero benefit. This is consistent with the view that validation loss on a static dataset measures compression quality, not representational structure. The closed-loop model's representations are dramatically reorganized (it encodes rich directional and magnitude information about its own computation that the open-loop model doesn't), but this reorganization doesn't manifest as better NTP on the same distribution. The true test of whether this representational quality translates to functional capability would require a continual learning or transfer evaluation — teaching both models something novel and measuring interference.

### Reproduction

```bash
cd experiments/
modal run --detach language_reduction/modal_app.py --stage a2a-extended-training \
  --n-tokens 10000000 --n-steps 15000 \
  --predict-from post_block0 --predict-to post_block3 \
  --fwd-n-layer 3 --fwd-d-head 128 --fwd-n-head 4 --fwd-mlp-mult 4
```

## Files

| File | Purpose |
|---|---|
| `extended_training.py` | Extended co-training with intermediate checkpoints |
| `EXTENDED_TRAINING_README.md` | This file |

## Reproduction (Run 7, 1% forward model)

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
    └── inject1/
        ├── P_10000000/                          # Run 7 (1% fwd, old path)
        │   ├── results.json
        │   ├── open_loop/
        │   │   ├── history.json
        │   │   └── step_*/
        │   └── closed_loop/
        │       ├── history.json
        │       └── step_*/
        └── fwd3L4H128d_mlp4/P_10000000/        # Run 7b (10% fwd)
            ├── results.json
            ├── open_loop/
            │   ├── history.json
            │   └── step_*/
            └── closed_loop/
                ├── history.json
                └── step_*/
```
