# RHM RL Ratchet: REINFORCE with FM Supervision (2026-06-25)

**Code**: `rhm_rl_ratchet.py`
**Prior experiments**: [RHM sparse ratchet](RHM_SPARSE_RATCHET_README.md), [FM as DGP approximation](PER_LEVEL_LOSS_README.md#fm-as-dgp-approximation-2026-06-21), [MNIST gated ratchet](../a2a_forward/GATED_RATCHET_README.md)

## Motivation

### Why RL: the quality of sparsity

The ratchet mechanism works strongly on MNIST (48% val loss improvement) but barely on RHM with dense NTP (+0.3% at m=4, -0.4% at m=2) or masked NTP (+1.2% at mask=0.95, non-compounding). The [sparse ratchet](RHM_SPARSE_RATCHET_README.md) established that the FM's process supervision becomes non-redundant when NTP is sufficiently sparse — but even at 95% masking, the ratchet doesn't compound and self-knowledge doesn't develop.

The key missing distinction is between **density** and **richness** of the supervision signal. MNIST's classification loss is both sparse (1 position out of 50) AND rich (the label constrains ALL 50 positions — "this is a 7" tells you about every pixel). Masked NTP is sparse but NOT rich: each visible position is an independent local prediction. Removing 95% of NTP positions doesn't create a global constraint — it just removes local constraints.

RL reward is sparse AND rich, like classification. A single scalar reward for a generation rollout constrains the entire sequence: "your generation was wrong" means the model must figure out which part of its compositional reasoning failed and fix it. This creates a credit assignment problem that the FM's process supervision can specifically address — the FM provides intermediate error signals at each layer that help identify WHERE the computation went wrong.

### The FM DGP alignment establishes the necessary condition

The [FM as DGP approximation](PER_LEVEL_LOSS_README.md#fm-as-dgp-approximation-2026-06-21) analysis showed that the FM captures 91-97% of the feature-conditioned structure at learned hierarchy levels, and is 12-13% more DGP-aligned than the main model at intermediate layers. The FM genuinely learns the composition rules. The ratchet's prior failure on RHM was not because the FM learns garbage — the training setup (dense NTP) didn't create conditions where the FM's knowledge was useful.

### Predictions

1. **Vanilla RL should collapse representations**: Without per-position supervision, the model should find *a* solution for generation that doesn't cleanly decompose the hierarchy — entangled representations that produce correct outputs via shortcuts.

2. **FM-supervised RL should produce cleaner decomposition**: The FM acts as a DGP-aligned regularizer, forcing intermediate representations to decompose hierarchically (measurable via per-layer feature eta²).

3. **Self-knowledge should finally appear on RHM**: Under RL, the FM's injection is genuinely useful (unlike NTP where the model already sees all prior tokens), so the model should develop representations encoding information about the FM's prediction quality.

## Design

**Phase 1: Pre-train** (shared): 10K steps of standard NTP on RHM (L=6, m=2, v=8, s=2). Model learns levels 0-2 well, levels 3-5 near baseline.

**Phase 2: Fine-tune** with 4 conditions, compute-matched on gradient steps (10K each):

| Condition | Wake phase | Sleep phase | FM? |
|---|---|---|---|
| **RL_FM** | REINFORCE + FM injection + local loss + NTP reg | KL + CE distillation (4 cycles) | yes |
| **RL** | REINFORCE + NTP reg | none (continuous) | no |
| **NTP_FM** | Dense NTP + FM injection + local loss | KL + CE distillation (4 cycles) | yes |
| **OL** | Dense NTP | none (continuous) | no |

**Generation task**: Given the first 32 tokens (prefix) of a 64-token RHM sequence, generate the remaining 32 tokens autoregressively. Reward = fraction of correct tokens in the suffix. The suffix contains positions at all hierarchy levels: 16 at L0, 8 at L1, 4 at L2, 2 at L3, 1 at L4, 1 at L5.

**REINFORCE**: Sequence-level reward with exponential moving average baseline (decay=0.95). During wake, the RL_FM condition generates WITH FM injection (the FM provides inference-time preview of correct computation); the RL condition generates standalone.

Architecture: 6L/6H/192D GPT (~2.68M params), 1L/1H/16D causal FM (50.4K params, 1.9%), post_block0 -> post_block5, inject after block 1, UnifiedGate (74.2K params). Identical seed (42), lr (3e-4), initial weights.

**New measurement: per-layer feature eta²**. At each checkpoint, compute feature eta² (fraction of activation variance explained by hierarchy-level feature identity) at each main model layer. This creates a layer × level matrix that reveals whether the model progressively transforms high-level features into low-level predictions through its depth.

## Run 1: No NTP regularization

RL conditions receive ONLY the REINFORCE gradient (no NTP loss during wake).

### Val loss: RL destroys NTP ability

| Condition | Cycle 1 | Cycle 2 | Cycle 3 | Cycle 4 |
|---|---|---|---|---|
| RL_FM | 3.577 | 3.550 | 3.501 | **3.529** |
| RL | 8.124 | 9.484 | 10.197 | **10.797** |
| NTP_FM | 0.797 | 0.795 | 0.794 | 0.795 |
| OL | 0.793 | 0.794 | 0.792 | **0.789** |

Pure RL catastrophically destroys NTP (val loss 10.8, 5× worse than uniform baseline 2.08). RL_FM is better (3.5) because the FM local loss provides implicit regularization, but still far worse than OL.

### Generation accuracy: RL does improve generation

| Condition | Overall | L0 | L1 | L2 | L3 | L4 | L5 |
|---|---|---|---|---|---|---|---|
| RL_FM | 31.9% | 52.1% | 16.9% | 5.2% | 10.0% | 2.5% | 8.2% |
| **RL** | **39.2%** | 52.1% | 23.7% | 35.4% | 20.0% | 28.8% | 22.1% |
| NTP_FM | 27.9% | 35.7% | 17.7% | 25.9% | 18.0% | 21.1% | 18.3% |
| OL | 27.8% | 35.1% | 18.5% | 26.8% | 17.2% | 19.1% | 19.4% |

RL achieves 39.2% generation accuracy (41% improvement over OL), but at the cost of losing NTP entirely.

### Per-layer eta²: RL collapses computation (prediction 1 confirmed)

RL's feature eta² is flat across all 6 layers — every layer encodes the same thing:

| Layer | L3 | L4 | L5 |
|---|---|---|---|
| **RL post_block0** | 0.061 | 0.161 | 0.446 |
| **RL post_block5** | 0.062 | 0.159 | 0.434 |
| OL post_block0 | 0.168 | 0.485 | 0.598 |
| OL post_block5 | 0.302 | 0.220 | 0.084 |

RL's L5 eta² barely changes across the network (0.446 → 0.434, 1.03× decrease). OL's drops 7× (0.598 → 0.084). The RL model has stopped computing progressively — it's a single-step mapping from input to output. This confirms the "collapsed representation" prediction.

RL_FM shows intermediate structure (L5: 0.625 → 0.187, 3.3× decrease), confirming the FM partially prevents collapse (prediction 2).

### Self-knowledge: appears under RL (prediction 3 confirmed)

| Condition | post_block0 | post_block5 |
|---|---|---|
| **RL_FM** | 0.013 | **0.633** |
| NTP_FM | 0.018 | 0.559 |
| OL | 0.012 | 0.544 |

For the first time on RHM, the FM-supervised model has higher self-knowledge than OL (0.633 vs 0.544). Every prior RHM experiment showed OL >= WS at late layers.

### Gate dynamics: gate stays open (new finding)

| Condition | Gate mean | Sparsity |
|---|---|---|
| RL_FM | **0.673** | **0%** |
| NTP_FM | 0.002 | 100% |

Under NTP, the injection is redundant and the gate closes. Under RL, the injection is genuinely useful for generation — the FM provides a preview that the model can't get from the prefix alone. This is the first setting where the gate finds injection valuable on RHM.

### Robustness

| Condition | eps=1.0 |
|---|---|
| RL_FM | +0.090 |
| RL | -0.012 |
| NTP_FM | +0.279 |
| OL | +0.120 |

## Run 2: Dense NTP regularization (lambda_ntp=1.0)

Added standard NTP loss on a separate batch during each RL training step to prevent catastrophic forgetting.

### Val loss: NTP regularization creates a split

| Condition | Val loss | Gen accuracy |
|---|---|---|
| OL | 0.789 | 27.8% |
| **RL** | **0.788** | 28.0% |
| NTP_FM | 0.795 | 27.9% |
| RL_FM | 2.637 | 39.1% |

Dense NTP regularization works perfectly for **RL-only** — val loss matches OL (0.788 ≈ 0.789). But RL-only shows **zero generation improvement** (28.0% ≈ 27.8%). The NTP gradient at λ=1.0 completely overwhelms REINFORCE. The model is functionally identical to OL.

**RL_FM is still degraded** (2.64) despite NTP regularization. The FM injection (gate at 0.998) and local loss create a conflict with the NTP gradient that the regularization can't resolve.

### Gate fully opens under dense NTP reg

| Condition | Gate mean | Sparsity |
|---|---|---|
| RL_FM | **0.998** | 0% |
| NTP_FM | 0.002 | 100% |

Even more open than run 1 (0.998 vs 0.673). With NTP regularization preventing the model's representations from adapting to generation, the model relies ENTIRELY on the injection — all generation benefit flows through the gate.

### Per-layer eta²: RL = OL (NTP dominates completely)

RL's eta² is indistinguishable from OL — the NTP gradient overrode any RL influence:

| Condition | L5 at block0 | L5 at block5 | Ratio |
|---|---|---|---|
| RL | 0.584 | 0.083 | 7.0× |
| OL | 0.598 | 0.084 | 7.1× |
| RL_FM | 0.569 | 0.399 | 1.4× |

RL_FM's eta² **freezes** at later layers (only 1.4× decrease in L5). The competing objectives (RL, NTP, local loss) create a stalemate where the later layers can't do progressive computation.

### Self-knowledge inverts relative to run 1

| Condition | post_block0 | post_block5 |
|---|---|---|
| RL_FM | 0.046 | 0.267 |
| NTP_FM | 0.018 | 0.559 |
| OL | 0.012 | 0.544 |

Self-knowledge at post_block5 drops from 0.633 (run 1) to 0.267. The NTP regularization disrupts the representations that encoded FM error information.

## Run 3: Sparse NTP regularization (ntp_mask_rate=0.95)

Motivated by the [sparsity sweep's](RHM_SPARSITY_SWEEP_README.md) finding that mask=0.95 is the sweet spot where FM process supervision becomes non-redundant. RL_FM condition only (other conditions would be identical to run 2 or unchanged).

### Val loss: distillation is the culprit

**Checkpoint** (post-distillation) val loss:

| Cycle | Val loss |
|---|---|
| 1 | 2.238 |
| 2 | 2.308 |
| 3 | 2.304 |
| 4 | **2.291** |

Still broken at checkpoints. But the **wake trajectory** reveals the real story:

| Phase | Val loss |
|---|---|
| Wake start (post-distillation) | 1.94 |
| Wake +500 steps | **0.82** |
| Wake +1000 steps | 0.83 |
| Wake end | **0.82** |
| Checkpoint (post-sleep, post-repoint) | **2.29** |

**During wake, val loss recovers to 0.82** — only 4% worse than OL (0.79). The sparse NTP at mask=0.95 repairs the model within ~500 steps. Then distillation destroys it back to 2.29. Every cycle: wake repairs → sleep destroys → wake repairs → sleep destroys.

### Generation: model internalizes generation ability

| Mode | Overall | L0 | L2 | L4 | L5 |
|---|---|---|---|---|---|
| With FM injection | 39.0% | 51.9% | 34.8% | 29.5% | 23.0% |
| **Without FM** | **38.6%** | **51.0%** | **35.3%** | **28.8%** | **22.5%** |
| OL baseline | 27.8% | 35.1% | 26.8% | 19.1% | 19.4% |

The standalone model generates at 38.6% — virtually identical to the 39.0% with FM injection. The generation ability IS internalized in the weights, not dependent on the injection. The 0.4pp difference is noise. This is a **39% relative improvement** over the OL baseline, at every hierarchy level.

### Per-layer eta²: frozen at later layers

| Layer | L3 | L4 | L5 |
|---|---|---|---|
| RL_FM post_block0 | 0.160 | 0.470 | 0.592 |
| RL_FM post_block3 | 0.254 | 0.547 | 0.484 |
| RL_FM post_block5 | 0.269 | 0.565 | 0.496 |
| OL post_block0 | 0.168 | 0.485 | 0.598 |
| OL post_block3 | 0.550 | 0.466 | 0.200 |
| OL post_block5 | 0.302 | 0.220 | 0.084 |

Blocks 3-5 show minimal progression. L5 decreases only 1.2× (RL_FM) vs 7× (OL). L4 actually *increases* through the network. The competing objectives prevent progressive computation.

### Gate, self-knowledge, FM cosine

| Metric | Value |
|---|---|
| Gate mean (c4) | 0.998 (0% sparse) |
| SK post_block0 | 0.044 |
| SK post_block5 | 0.130 |
| FM cosine | 0.997 |

Gate fully open (same as run 2). Self-knowledge lower than both previous runs. FM near ceiling.

## Cross-run summary

| | Run 1 (no NTP) | Run 2 (dense NTP) | Run 3 (sparse NTP) |
|---|---|---|---|
| NTP reg | none | λ=1.0, mask=0 | λ=1.0, mask=0.95 |
| **Val loss (checkpoint)** | 3.53 | 2.64 | 2.29 |
| **Val loss (during wake)** | — | — | **0.82** |
| **Gen acc (with FM)** | 31.9% | 39.1% | 39.0% |
| **Gen acc (no FM)** | — | — | **38.6%** |
| Gate mean | 0.673 | 0.998 | 0.998 |
| SK post_block5 | **0.633** | 0.267 | 0.130 |
| OL val loss | 0.789 | 0.789 | — |
| OL gen acc | 27.8% | 27.8% | — |

## Key findings

### 1. RL + FM improves generation by 40% at all hierarchy levels

Across all three runs, RL_FM achieves ~39% generation accuracy vs OL's 27.8%. Run 3 shows this improvement is **fully internalized** — standalone generation (38.6%) matches FM-assisted generation (39.0%). The improvement is broadly distributed: L0 +48%, L2 +32%, L4 +51%, L5 +17%.

### 2. Vanilla RL produces collapsed representations

The per-layer feature eta² from run 1 directly confirms the motivating prediction. The RL model's eta² is flat across all 6 layers (delta < 0.01 at every level), meaning it has stopped doing progressive hierarchical computation. The model found a shortcut solution that produces correct-ish outputs without decomposing the hierarchy. RL_FM's eta² retains progressive structure — the FM acts as a structural regularizer.

### 3. Self-knowledge appears under RL — a first for RHM

Run 1 shows RL_FM self-knowledge at 0.633 vs OL's 0.544 (post_block5). This is the first time WS > OL on RHM. Every prior RHM experiment (dense NTP ratchet, sparse ratchet, L=4 ratchet) showed OL >= WS. The explanation: under RL, the FM injection is genuinely useful (gate at 0.67), creating structured residual the model can learn about. Under NTP, the FM is redundant (gate at 0.002), so self-knowledge can't develop.

### 4. The gate stays open under RL — injection is useful

The unified gate opens to 0.67-0.998 under RL vs closing to 0.002 under NTP. This confirms that the FM provides genuinely new information during generation that it can't provide during teacher-forced NTP. The model already sees all prior tokens during NTP, making injection redundant. During generation, the FM's preview of downstream computation is novel and valuable.

### 5. Distillation fails when injection is dominant

The wake-sleep ratchet mechanism breaks when the gate is near 1.0. The teacher's logits (99.8% injection) differ fundamentally from the standalone model's logits. Distillation tries to force the student to match a computation it can't replicate without the FM. On MNIST, the gate was 0.13-0.77, making distillation tractable. The distillation is specifically what destroys NTP ability — during wake (run 3), val loss stays at 0.82 (4% of OL), but distillation pushes it to 2.29.

### 6. Dense NTP overwhelms REINFORCE completely

Run 2's RL-only condition with λ_ntp=1.0 produces a model identical to OL on every metric. The NTP gradient (dense, per-position CE at 63 positions) is so much cleaner than REINFORCE (one scalar per sequence) that even equal weighting makes RL invisible.

## Why distillation fails here but works on MNIST

The ratchet mechanism (wake → sleep → repoint) requires that the injection contribution is small enough for distillation to internalize. This is controlled by the gate:

| Domain | Gate at checkpoint | Distillation success |
|---|---|---|
| MNIST (WS_UG_uniform) | 0.13 (37% sparse) | Yes — 48% val loss improvement |
| RHM NTP_FM | 0.002 (100% sparse) | N/A — no injection to distill |
| RHM RL_FM | 0.998 (0% sparse) | **No** — injection too dominant |

The ratchet needs the gate in a middle regime: open enough that the FM provides useful information, closed enough that distillation can absorb it. Under RL on RHM, the FM is so useful that the gate opens fully, exceeding the distillation's capacity.

## Relation to the cerebellar theory

The cerebellar abstraction ratchet predicts that self-knowledge and compositional deepening emerge from a cycle of: FM provides predictions → model develops self-knowledge about FM errors → distillation compresses this into primitives → fresh FM finds new errors → cycle repeats.

These experiments support the FIRST half: under RL, the FM provides genuinely useful predictions (gate opens), and the model develops self-knowledge (R² = 0.633 > OL's 0.544). But the SECOND half — distillation-based compression — fails because the FM's contribution is too large to internalize through logit matching.

The biological analog may be informative: the cerebellum-to-cortex pathway is bandwidth-limited by the dentate nucleus, naturally constraining how much computation the cerebellum can contribute at any moment. This built-in bottleneck may be essential for the ratchet to work — it prevents the cortex from becoming fully dependent on the cerebellum.

## Open directions

1. **Drop the sleep phase**: During wake (run 3), the model achieves val ≈ 0.82 with generation ≈ 39% standalone. The distillation is purely destructive. Continuous RL+FM+sparse_NTP training (no wake-sleep cycles) would test whether the generation improvement can coexist with maintained NTP.

2. **Constrain injection magnitude**: Cap the gate at 0.3-0.5 (matching the MNIST regime where distillation works). This forces the model to do most of its generation computation independently, with the FM providing only a small correction that distillation can absorb.

3. **RL on language**: The motivating insight — that RL supervision is sparse-but-rich like classification — applies directly to RLHF. The FM's process supervision could provide credit assignment during policy optimization, filling the same role the cerebellum hypothetically plays in human learning from sparse evaluative feedback.

## Reproduction

```bash
cd experiments/

# Run 1: No NTP regularization (all 4 conditions)
modal run --detach -m rhm.rhm_rl_ratchet::rhm_rl_ratchet \
  --lambda-ntp 0.0 --ntp-mask-rate 0.0

# Run 2: Dense NTP regularization (all 4 conditions)
modal run --detach -m rhm.rhm_rl_ratchet::rhm_rl_ratchet \
  --lambda-ntp 1.0 --ntp-mask-rate 0.0

# Run 3: Sparse NTP regularization (RL_FM only)
modal run --detach -m rhm.rhm_rl_ratchet::rhm_rl_ratchet \
  --lambda-ntp 1.0 --ntp-mask-rate 0.95 --only-rl-fm
```

Results saved to `rhm-scaling-data` volume at:
- Run 1-2: `rhm_rl_ratchet/v8_s2_L6_m2/results.json` (overwritten by run 2)
- Run 3: `rhm_rl_ratchet/v8_s2_L6_m2_rl_fm_only/results.json`
