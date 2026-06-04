# Controlled Retrain: Eliminating the lr/seed Confound (2026-05-29)

**Prior experiment**: [CLOSED_LOOP_README.md](CLOSED_LOOP_README.md)
**Causal probes**: [CAUSAL_PROBES_README.md](CAUSAL_PROBES_README.md)
**Idea doc**: [ideas/activation_to_activation_forward.md](../../../../ideas/activation_to_activation_forward.md)

## Why this exists

The original closed-loop experiment (Run 4) compared open-loop (lr=1e-4) and closed-loop (lr=3e-4) models. The self-map probe result (R²=0.44 closed vs 0.28 open for predicting the residual from post_block3) was confounded by this learning rate difference. The causal probes (CAUSAL_PROBES_README.md) noted this confound but couldn't resolve it.

This experiment trains both open-loop and closed-loop models with **identical lr (3e-4), identical seed (42), identical initial weights, and identical training data in the same order**. The only difference is whether the cerebellar loop is closed.

## Design

Both models start from the same randomly initialized weights (seed=42). All 10K training batch indices and all eval batch indices are pre-generated with separate `torch.Generator` instances, so both runs see identical data regardless of internal RNG divergence. Open-loop is trained first, then closed-loop from the same initial weights.

| Parameter | Value |
|---|---|
| Main model | 4L, 4H, 256D GPT (28.9M params) |
| Forward model | 2L transformer, 1H, 64D (660K params) |
| Forward model gap | post_block0 → post_block3 |
| Injection point | After block 1 |
| lr (both models) | 3e-4 |
| fwd_lr | 1e-3 |
| Seed | 42 |
| Steps | 10,000 |
| Tokens | 10M (FineWeb-Edu, τ=0.0) |

After training, layerwise probes compare the two models' ability to encode the forward model's prediction residual.

## Results

### The injection helps LM loss (replicates Run 4)

| Step | Closed (with inj) | Closed (no inj) | Δ |
|---|---|---|---|
| 200 | 6.770 | 6.837 | -0.067 |
| 2000 | 5.485 | 5.560 | -0.075 |
| 5000 | 5.232 | 5.280 | -0.048 |
| 8000 | 5.254 | 5.326 | -0.073 |
| 9999 | 5.422 | 5.506 | **-0.084** |

Negative at every eval point across 10K steps. The gate opened 0.08 → 3.01 (same pattern as Run 4). Both findings replicate cleanly with the lr confound removed.

### The model becomes dependent on the injection

| Model | Final val LM loss |
|---|---|
| Open-loop | **5.398** |
| Closed-loop (with injection) | 5.422 |
| Closed-loop (no injection) | 5.506 |

The closed-loop model *with* the injection is slightly worse than the open-loop model (+0.024 nats). *Without* the injection, it's significantly worse (+0.108). The model learned to use the prediction as a crutch rather than internalizing the computation. This validates the "wake-sleep consolidation" concern from the idea doc — the model needs periodic open-loop training phases to consolidate what it learned from the prediction into its own weights.

### Self-knowledge probes: the R² gap is real, not an lr artifact

**Full 256-d residual vector probes:**

| Layer | Open R² | Closed R² | Δ R² | Open cos | Closed cos |
|---|---|---|---|---|---|
| post_block0 | 0.024 | 0.209 | **+0.185** | 0.160 | 0.447 |
| post_block1 | 0.062 | 0.262 | **+0.200** | 0.250 | 0.501 |
| post_block2 | 0.158 | 0.343 | **+0.185** | 0.396 | 0.577 |
| post_block3 | 0.260 | 0.421 | **+0.161** | 0.509 | 0.643 |

The R²=0.42 vs 0.26 at post_block3 closely replicates the original finding (0.44 vs 0.28). The lr confound was not the driver.

**Scalar probes (residual norm and control):**

| Target | Layer | Open R² | Closed R² | Δ |
|---|---|---|---|---|
| residual_norm | post_block0 | 0.364 | 0.390 | +0.026 |
| residual_norm | post_block1 | 0.412 | 0.435 | +0.023 |
| residual_norm | post_block2 | 0.424 | 0.442 | +0.019 |
| residual_norm | post_block3 | 0.387 | 0.422 | +0.035 |
| block1_contrib | post_block0 | 0.567 | 0.618 | +0.051 |
| block1_contrib | post_block1 | 0.683 | 0.735 | +0.052 |
| block1_contrib | post_block2 | 0.618 | 0.683 | +0.065 |
| block1_contrib | post_block3 | 0.547 | 0.593 | +0.046 |

Scalar residual_norm gaps are modest (+0.02-0.04). Block1_contrib (control) shows a larger scalar gap (+0.05-0.07). But the vector probe gap (+0.16-0.20) dwarfs both, meaning the closed-loop model specifically encodes the residual's *directional structure* much better, not just scalar quantities generally.

**Forward model quality (no injection):**

| Model | Cosine | MSE |
|---|---|---|
| Open-loop | 0.915 | 0.124 |
| Closed-loop | 0.900 | 0.263 |

The closed-loop forward model is worse (cosine 0.90 vs 0.92), consistent with Run 4 — it's tracking a co-evolving target.

## Interpretation

### Self-knowledge is real and distributed via backprop

The probe gap is **uniform across all layers**, including post_block0 (+0.185), which is computed before the injection enters the residual stream. This rules out the hypothesis that self-knowledge is built by downstream layers directly processing the injection signal (the "localization" mechanism). Instead, self-knowledge is distributed through the model via backpropagation: the injection changes the loss landscape, and gradients flowing backward through all layers cause every layer to reorganize its representations to complement the forward model's prediction.

The early-layer result is the cleanest evidence. Post_block0 is the *input* to the forward model. The residual is `post_block3 - fwd_model(post_block0)`. A probe from post_block0 to the residual asks: "can the early layer predict what the later computation will produce that the forward model misses?" In the open-loop model: R²=0.024 (nothing). In the closed-loop model: R²=0.209 (9× increase). The early layer has reorganized to make the forward model's blind spots linearly accessible. This is self-knowledge: the model's early representations encode information about where its own later computation will surprise a compressed model of itself.

### What the self-knowledge is not

Two specific mechanisms for self-knowledge have been tested and refuted:

1. **Magnitude gating**: The model does not use the scalar residual norm to gate reliance on the injection. The steering test (CAUSAL_PROBES_README.md, Test 2) showed zero novelty-specific effect. The scalar residual_norm probe gap is modest (+0.03) compared to the vector gap (+0.18).

2. **Post-injection localization**: The self-knowledge is not built by downstream layers processing the injection. It's present before the injection point. The mechanism is gradient-mediated, not signal-processing-mediated.

### What the self-knowledge appears to be

The self-knowledge is encoded in the residual's **direction**, not its magnitude. Evidence:

- The vector probe gap (Δ R²=+0.18) is 6× the scalar gap (Δ R²=+0.03)
- The enriched injection-help analysis (CAUSAL_PROBES_README.md, Test 3-enriched) showed residual direction organizes injection help 5× more than magnitude (η²=0.0017 vs 0.0003)
- The injection helps most at focused-attention positions, organized by attention shape — the model uses the prediction differently depending on what *kind* of computation is happening, not how *much* error there is

The model appears to encode *what kind of computation the forward model will miss* (directional) rather than *how much it will miss* (scalar). This is consistent with the behavioral residual analysis (before_closer d=+0.84, sentence_start d=-0.85): the residual has specific, named structure corresponding to different computational mechanisms.

### The consolidation problem

The model becomes dependent on the injection rather than internalizing the prediction. Without injection, the closed-loop model is 0.11 nats worse than the open-loop baseline. The model learned to *complement* the forward model's prediction (offloading predictable computation) rather than to *incorporate* it into its own weights.

This is the "wake-sleep" problem described in the idea doc. The cortex relies on real-time cerebellar correction during waking and consolidates via offline replay during sleep. The engineering analog: interleave closed-loop training (with injection) and open-loop training (without injection), forcing the model to internalize what it was relying on the prediction for.

## Files

| File | Purpose |
|---|---|
| `controlled_retrain.py` | Training + probes stage: trains both open-loop and closed-loop from identical init, runs layerwise probes |
| `CONTROLLED_RETRAIN_README.md` | This file |

## Reproduction

```bash
cd experiments/
modal run a2a_forward/controlled_retrain.py \
  --n-tokens 10000000 --n-steps 10000 \
  --predict-from post_block0 --predict-to post_block3 --fwd-n-layer 2
```

## Modal volume

Results saved to `language-reduction-data` volume:

```
/data/a2a_forward/controlled/
└── post_block0_to_post_block3/
    └── inject1/P_10000000/
        ├── results.json
        ├── open_loop/
        │   ├── model.pt
        │   └── fwd_model.pt
        └── closed_loop/
            ├── model.pt
            ├── fwd_model.pt
            └── gate.pt
```
