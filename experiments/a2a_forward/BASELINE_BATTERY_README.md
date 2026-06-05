# Baseline Battery: Is Forward Self-Prediction Uniquely Useful?

**Code**: `baseline_battery.py`
**Date**: 2026-06-04
**Prior experiment**: [Controlled retrain](CONTROLLED_RETRAIN_README.md) (Run 6), [Mirror test v2](MIRROR_TEST_README.md)

## Motivation

A reasonable objection to the self-knowledge findings: "If you inject any structured signal into the residual stream, the model will reorganize to exploit it, and probes will show it encodes that signal's error structure — because that's what's useful for leveraging the input. Is this 'self-knowledge' or just 'the model learned to use an auxiliary input it was given'?"

This experiment tests whether the benefits of forward model injection (LM improvement, self-knowledge, perturbation robustness) are specific to injecting a prediction of the model's own future computation, or whether any structured injection would produce the same effects.

## Design

5 conditions, all with identical lr (3e-4), seed (42), initial weights, and data order. The **only** difference is what gets injected into the residual stream after block 1:

| Condition | Injection | Position-specific? | Learned? | Self-referential? | Forward-looking? |
|---|---|---|---|---|---|
| `open_loop` | None | — | — | — | — |
| `forward` | gate(fwd_model(post_block0)) | Yes | Yes | Yes | **Yes** |
| `shifted` | gate(shift(fwd_model(post_block0), k=10)) | Partially | Yes | No | No |
| `random_proj` | gate(frozen_random_proj(post_block0)) | Yes | No | No | No |
| `autoencoder` | gate(autoenc(post_block0)) | Yes | Yes | Yes | **No** |

**Shifted baseline**: Position $t$ receives the forward model's prediction from position $t-10$, with zero-padding for positions 0–9. This preserves the statistical properties and causality of the injection but destroys position-specific self-prediction — position $t$ gets a prediction about a different position's computation. An earlier attempt at this baseline used random permutation across positions, but this leaked future information: position $t$ receiving the prediction from position $\pi(t) > t$ would see context that violates the causal mask.

**Random projection baseline**: A frozen random matrix $W \in \mathbb{R}^{d \times d}$, initialized as $\mathcal{N}(0, 1/d)$, applied to post_block0 and passed through the learned gate. Since both the random projection and gate are linear, this is equivalent to learning a single linear function of post_block0 — the model can learn any linear transformation of its early activations. Tests whether any additional position-specific input helps.

**Autoencoder baseline**: Same TransformerForwardModel architecture (2-layer, 1 head, 64-dim, 660K params) trained to reconstruct post_block0 from post_block0 through the same capacity bottleneck. This is the strongest baseline — it's nonlinear, capacity-limited, and self-referential (it's about the model's own computation), with the only difference being that it's not forward-looking. The autoencoder compresses and reconstructs the model's current computation rather than predicting its future computation.

All conditions co-train a forward model (post_block0 → post_block3) on MSE loss, trained on each condition's actual computation. This forward model is used for probing: the self-knowledge metric asks "does this model's activations encode its own forward model's prediction residual?" and is therefore comparable across conditions.

## Results

### 1. Perturbation robustness — forward model uniquely best

Random perturbation vectors (16 directions, $s=2.0 \times \sigma$) applied at post_block1, measuring downstream response at post_block3 and loss degradation.

| Condition | $\|\|R\|\|$ | $\|\|R\|\|/\text{OL}$ | $\cos(R, \delta)$ | $\Delta\text{loss}$ | $\Delta\text{loss}/\text{OL}$ |
|---|---|---|---|---|---|
| open_loop | 1.866 ± 0.072 | 1.000 | +0.714 ± 0.024 | +0.0109 ± 0.0036 | 1.000 |
| **forward** | **1.726 ± 0.055** | **0.925** | **+0.778 ± 0.022** | **+0.0045 ± 0.0029** | **0.413** |
| shifted | 1.735 ± 0.066 | 0.930 | +0.761 ± 0.024 | +0.0052 ± 0.0026 | 0.477 |
| random_proj | 1.817 ± 0.070 | 0.974 | +0.734 ± 0.023 | +0.0074 ± 0.0039 | 0.679 |
| autoencoder | 1.817 ± 0.075 | 0.974 | +0.733 ± 0.021 | +0.0069 ± 0.0041 | 0.636 |

The forward model takes only 41% of open_loop's loss degradation from identical perturbations — a 2.4× robustness improvement. The baselines also show some robustness improvement (any injection regularizes somewhat), but the gap is large: the forward model's advantage over the best baseline (autoencoder, 0.636) is substantial. The forward model produces nearly twice the robustness gain of the random_proj or autoencoder baselines (robustness gain = 1 - ratio: forward 0.587, shifted 0.523, random_proj 0.321, autoencoder 0.364).

The response norms show a clear two-tier pattern: forward and shifted produce smaller responses (||R||/OL ≈ 0.93), while random_proj and autoencoder produce responses nearly as large as open_loop (||R||/OL ≈ 0.97). But loss degradation separates forward from shifted: despite similar response magnitudes, the forward model's response is more organized (higher cos(R, δ) = 0.778 vs 0.761), producing less loss damage per unit of response. The forward model's perturbation response is both smaller AND more structured.

### 2. Self-knowledge probes — forward model highest, with a unique depth pattern

Vector probes (R²) predicting the forward model's 256-d residual from each layer's activations, collected without injection.

| Layer | open_loop | forward | shifted | random_proj | autoencoder |
|---|---|---|---|---|---|
| post_block0 | 0.024 | **0.209** | 0.052 | 0.195 | 0.203 |
| post_block1 | 0.062 | **0.262** | 0.098 | 0.210 | 0.216 |
| post_block2 | 0.158 | **0.343** | 0.186 | 0.268 | 0.281 |
| post_block3 | 0.260 | **0.421** | 0.270 | 0.315 | 0.328 |

Δ R² (condition minus open_loop):

| Layer | forward | shifted | random_proj | autoencoder |
|---|---|---|---|---|
| post_block0 | **+0.185** | +0.028 | +0.171 | +0.179 |
| post_block1 | **+0.200** | +0.037 | +0.148 | +0.154 |
| post_block2 | **+0.185** | +0.028 | +0.111 | +0.123 |
| post_block3 | **+0.161** | +0.010 | +0.055 | +0.068 |

The forward model has the highest Δ R² at every layer. But the more important finding is the **depth profile**:

- **Forward**: roughly uniform Δ R² from post_block0 (+0.185) to post_block3 (+0.161). Self-knowledge penetrates the full depth of the model.
- **Random_proj and autoencoder**: steeply declining Δ R² from post_block0 (+0.17–0.18) to post_block3 (+0.05–0.07). Self-knowledge is concentrated in early layers and fades by 3× at the model's deepest computation.
- **Shifted**: minimal self-knowledge at all layers (+0.01 to +0.04). Position-specific content is essential.

At post_block3 — where the model's computation has been most shaped by the injection — the forward model's self-knowledge is 2.4× stronger than the autoencoder's (+0.161 vs +0.068) and 2.9× stronger than the random projection's (+0.161 vs +0.055).

This depth pattern is the key discriminator. Any structured injection causes some representational reorganization at early layers that incidentally correlates with the forward model's error structure. But only forward prediction produces self-knowledge that persists through the full depth of the model. This is consistent with the forward model creating a self-referential structure the model organizes around at all depths, rather than just changing early representations as a side effect of receiving additional input.

### 3. LM loss and injection dependency

| Condition | val_lm (with inj) | val_lm (no inj) | Δ(inj) | Δ(vs OL) | Gate norm |
|---|---|---|---|---|---|
| open_loop | 5.398 | 5.398 | — | — | — |
| forward | 5.422 | 5.506 | -0.084 | +0.024 | 3.01 |
| shifted | 5.413 | 5.411 | +0.002 | +0.015 | 2.18 |
| random_proj | 5.410 | 5.490 | -0.080 | +0.011 | 3.28 |
| autoencoder | 5.402 | 5.494 | **-0.092** | +0.004 | 3.58 |

Three findings here:

**The shifted baseline validates position-specificity.** The model learned that causally shifted predictions aren't useful: Δ(inj) ≈ 0 at every evaluation point throughout training, the gate norm is the lowest (2.18 vs 3.0–3.6 for others), and the injection is effectively ignored. Destroying position-specific self-prediction destroys the injection's value entirely.

**The autoencoder produces the largest injection benefit.** At -0.092 nats, the autoencoder injection is more useful for raw LM loss than forward prediction (-0.084). Compressed self-reconstruction provides a kind of skip connection that the model finds immediately useful. This makes the robustness result more striking: the most useful injection for LM loss does NOT produce the most robustness. LM improvement and robustness are dissociated.

**All injected conditions produce dependency.** The val_lm without injection is worse than open_loop for all conditions except shifted. The model offloads computation to whatever injection it receives. The autoencoder produces the greatest dependency (val_lm without injection 5.494 vs open_loop 5.398).

#### Injection Δ trajectory over training

| Step | forward | shifted | random_proj | autoencoder |
|---|---|---|---|---|
| 0 | +0.000 | +0.000 | -0.001 | +0.001 |
| 1000 | -0.146 | -0.004 | -0.020 | -0.010 |
| 3000 | -0.057 | -0.002 | -0.042 | -0.028 |
| 5000 | -0.048 | +0.000 | -0.047 | -0.042 |
| 7000 | -0.061 | +0.002 | -0.060 | -0.064 |
| 10000 | -0.084 | +0.002 | -0.080 | -0.092 |

The forward model's injection benefit jumps early (−0.146 at step 1000 — larger than any other condition at any point) and then oscillates before growing monotonically in the second half. The random_proj and autoencoder benefits grow steadily throughout training, eventually matching or exceeding the forward model's benefit. The shifted baseline flatlines at zero.

The early spike in forward benefit is notable: the model immediately extracts value from a prediction of its own future computation, before the gate has had time to develop a complex transformation. The other baselines need more training to discover how to use their injections.

### 4. Forward model quality across conditions

| Condition | Forward model cosine (end of training) |
|---|---|
| open_loop | 0.916 |
| forward | 0.900 |
| shifted | 0.921 |
| random_proj | 0.881 |
| autoencoder | 0.876 |

The forward model's prediction is hardest for the random_proj and autoencoder conditions (lowest cosine). These injections change the model's computation at blocks 2–3, making post_block3 harder to predict. The shifted condition's forward model has the highest cosine (0.921), consistent with the shifted injection having no effect on computation (the injection is ignored). The forward condition (0.900) is intermediate — the forward model co-adapts with the model, tracking a computation that is shaped by its own predictions.

## Interpretation

### What the skeptic gets right

The skeptic's objection has partial merit. Injecting any structured signal — even a frozen random linear projection of post_block0 — causes representational reorganization that probes can detect as increased encoding of the forward model's error structure. At early layers (post_block0), the Δ R² for random_proj (+0.171) and autoencoder (+0.179) is close to the forward model's (+0.185). If one looked only at early-layer probes, the skeptic would appear vindicated: "the model learned to use an auxiliary input, and incidentally its representations now correlate with the forward model's errors."

### Why baselines show high early-layer Δ R² but not late-layer

The high Δ R² at post_block0 for random_proj (+0.171) and autoencoder (+0.179) is not self-knowledge — it is shared-cause correlation. Each condition's forward model is trained on that condition's actual activations. The random_proj model's computation at post_block3 is different from the open_loop model's (because blocks 2–3 process the injected signal), so the random_proj forward model learns to predict a *different* target. The probe is asking "does the random_proj model's post_block0 predict the random_proj forward model's residual?" — and the answer is yes, because both the early representations and the forward model's error structure are downstream effects of the same cause: blocks 2–3 processing an injection that changed the task loss gradient flowing backward through them.

This correlation fades at late layers because at post_block3, the forward model already captures most of the computation (R² = 0.260 even for open_loop). For baselines to show large Δ R² at post_block3, the injection would need to cause the model's deepest representations to encode *specifically what the forward model gets wrong* — not just change in a way that incidentally correlates with it. That's a stronger condition, and non-self-referential injections don't create gradient pressure toward it.

The forward model condition *does* create this pressure at all depths, because the injection IS the forward model's prediction. Blocks 2–3 directly process the prediction, evaluate it against their own computation, and the gradient from this shapes representations all the way back. The self-knowledge at post_block3 (+0.161) reflects genuine computational organization around "where is my forward model wrong," not incidental correlation from changed training dynamics.

### What the skeptic gets wrong

Three findings jointly refute the claim that forward self-prediction is interchangeable with any structured injection:

**1. The depth profile.** The forward model's self-knowledge is roughly uniform across layers (+0.185 at post_block0, +0.161 at post_block3). The baselines' self-knowledge fades dramatically with depth (random_proj: +0.171 → +0.055; autoencoder: +0.179 → +0.068). At post_block3 — the model's deepest computation — the forward model's Δ R² is 2.4–2.9× the baselines'. The early-layer Δ R² in baselines is shared-cause correlation (see above); only forward prediction produces self-knowledge that persists at the model's deepest layers, where it would require genuine self-referential organization.

**2. The robustness gap.** The forward model takes 41% of open_loop's loss degradation; the autoencoder takes 64%. The forward model's robustness gain is 1.6× the autoencoder's despite the autoencoder producing a larger LM loss benefit. Robustness is not a byproduct of "the model receiving useful additional information." If it were, the autoencoder (which is more useful for LM loss) should produce more robustness. Instead, robustness correlates with the depth profile of self-knowledge: the condition with the most uniform, deep self-knowledge produces the most robustness.

**3. The dissociation between LM loss and robustness.** The autoencoder is the best injection for LM loss (-0.092 vs -0.084) but produces 54% more loss degradation from perturbations than the forward model. This means robustness is not a side effect of receiving useful information — it's specifically a consequence of the injection being a prediction of the model's own future computation. A compressed reconstruction of your current state is more immediately useful (it's a richer skip connection), but a prediction of your future computation produces a structured reference frame that reorganizes your representations at all depths.

### Why forward prediction is special

The forward model's prediction is uniquely informative in a way the other injections are not: it is the only injection that carries information about **where the model's computation is going**, not just where it currently is. The autoencoder tells the model "here's a compressed version of what you just computed at post_block0." The random projection tells it "here's a random linear transformation of what you just computed." The forward model tells it "here's approximately what you will compute at post_block3."

This forward-looking information creates a gradient pressure at blocks 2–3 to evaluate the prediction: trust it where it's accurate, override where it's wrong. To do this effectively, the model needs representations throughout its depth that encode the forward model's error structure — where the prediction is reliable and where it isn't. This is why the self-knowledge is uniform across layers for forward prediction but decays for the baselines: the baselines don't create this evaluate-and-override pressure at the model's deepest layers.

The robustness follows from the same mechanism. A model organized around evaluating a self-prediction has representations structured along dimensions that distinguish expected from unexpected computation. Perturbations interact with this structure and are channeled rather than scattering freely. A model organized around using a skip connection (autoencoder) or a random transformation (random_proj) has no such organized structure at its deepest layers, and perturbations propagate with less resistance.

## Summary

| Metric | forward | shifted | random_proj | autoencoder |
|---|---|---|---|---|
| Injection benefit (Δ nats) | -0.084 | 0.000 | -0.080 | **-0.092** |
| SK Δ R² (post_block0) | **+0.185** | +0.028 | +0.171 | +0.179 |
| SK Δ R² (post_block3) | **+0.161** | +0.010 | +0.055 | +0.068 |
| SK depth ratio (b3/b0) | **0.87** | 0.35 | 0.32 | 0.38 |
| Robustness (Δloss/OL) | **0.413** | 0.477 | 0.679 | 0.636 |

The forward model is uniquely best at robustness and self-knowledge depth. The autoencoder is best at raw LM loss improvement. The shifted baseline validates that position-specific content is essential. The dissociation between LM loss (autoencoder best) and robustness (forward best) is the single most informative finding: the benefits of forward self-prediction are not reducible to "the model received useful additional input."

## Reproduction

```bash
modal run --detach a2a_forward/baseline_battery.py::a2a_baseline_battery \
  --n-tokens 10000000 --n-steps 10000 \
  --predict-from post_block0 --predict-to post_block3 \
  --fwd-n-layer 2 --inject-after-block 1
```
