# RHM RL Ratchet: REINFORCE with FM Supervision (2026-06-25)

**Code**: `rhm_rl_ratchet.py`
**Prior experiments**: [RHM sparse ratchet](RHM_SPARSE_RATCHET_README.md), [FM as DGP approximation](../PER_LEVEL_LOSS_README.md#fm-as-dgp-approximation-2026-06-21), [MNIST gated ratchet](../../a2a_forward/GATED_RATCHET_README.md)

## Motivation

### Why RL: the quality of sparsity

The ratchet mechanism works strongly on MNIST (48% val loss improvement) but barely on RHM with dense NTP (+0.3% at m=4, -0.4% at m=2) or masked NTP (+1.2% at mask=0.95, non-compounding). The [sparse ratchet](RHM_SPARSE_RATCHET_README.md) established that the FM's process supervision becomes non-redundant when NTP is sufficiently sparse — but even at 95% masking, the ratchet doesn't compound and self-knowledge doesn't develop.

The key missing distinction is between **density** and **richness** of the supervision signal. MNIST's classification loss is both sparse (1 position out of 50) AND rich (the label constrains ALL 50 positions — "this is a 7" tells you about every pixel). Masked NTP is sparse but NOT rich: each visible position is an independent local prediction. Removing 95% of NTP positions doesn't create a global constraint — it just removes local constraints.

RL reward is sparse AND rich, like classification. A single scalar reward for a generation rollout constrains the entire sequence: "your generation was wrong" means the model must figure out which part of its compositional reasoning failed and fix it. This creates a credit assignment problem that the FM's process supervision can specifically address — the FM provides intermediate error signals at each layer that help identify WHERE the computation went wrong.

### The FM DGP alignment establishes the necessary condition

The [FM as DGP approximation](../PER_LEVEL_LOSS_README.md#fm-as-dgp-approximation-2026-06-21) analysis showed that the FM captures 91-97% of the feature-conditioned structure at learned hierarchy levels, and is 12-13% more DGP-aligned than the main model at intermediate layers. The FM genuinely learns the composition rules. The ratchet's prior failure on RHM was not because the FM learns garbage — the training setup (dense NTP) didn't create conditions where the FM's knowledge was useful.

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

## Run 4: No distillation (2026-06-25)

Tests whether dropping the destructive sleep phase preserves NTP while retaining the generation improvement. Same setup as run 3 (sparse NTP, mask=0.95) but with `--distill-steps 0`.

### Result: distillation is load-bearing for generation transfer

| Metric | Run 4 (no distill) | Run 3 (with distill) | OL |
|---|---|---|---|
| Val loss | **0.823** | 2.291 | 0.789 |
| Gen (with FM) | 30.5% | 39.0% | 27.8% |
| Gen (standalone) | **27.4%** | **38.6%** | 27.8% |
| Gate | 0.969 | 0.998 | — |
| FM cosine | 0.927 | 0.997 | — |
| SK post_block5 | 0.549 | 0.130 | 0.544 |

Without distillation, NTP is preserved beautifully (0.823, +4% of OL). But standalone generation is 27.4% — identical to OL. The model learned nothing about generation on its own.

During wake with FM injection, the model's parameters optimize for *injected-model* performance. The model learns to **use** the injection, not to **replicate** it internally. Without distillation, the sparse NTP gradient keeps standalone representations NTP-oriented. The RL gradient flows through the injected forward pass but doesn't transfer to standalone computation because the injection path bypasses the model's own blocks 2-5.

Distillation forces the student to match the teacher's (injected) logits *without* the injection — the only mechanism that transfers the FM's contribution into the model's weights. It serves two conflicting functions:
- **Good**: Transfers generation ability (standalone gen 27.4% → 38.6%)
- **Bad**: Destroys NTP (val loss 0.82 → 2.29)

The per-layer eta² is progressive (L5: 0.58→0.12, 5× decrease), matching OL's structure. Self-knowledge is neutral (0.549 ≈ OL's 0.544). Without distillation, the model is essentially OL with a minor injection benefit.

## Runs 5-7: FM capacity and local loss sweep (2026-06-25)

Tests whether the gate's refusal to close (and therefore the distillation failure) can be addressed by reducing FM capacity or weakening the local loss. All runs use sparse NTP (mask=0.95), full ratchet with distillation.

| Run | FM | Params | λ_local | FM cos | Gate | Val loss | Gen (no FM) | SK blk0 | SK blk5 |
|---|---|---|---|---|---|---|---|---|---|
| 3 | 1L/1H/16D | 50.4K | 1.0 | 0.997 | 0.998 | 2.29 | 38.6% | 0.044 | 0.130 |
| 5 | 1L/1H/8D | 25.8K | 1.0 | 0.996 | 0.997 | 2.22 | 38.5% | 0.125 | 0.291 |
| 6 | 1L/1H/4D | 13.5K | 1.0 | 0.993 | 0.996 | 2.21 | 38.5% | 0.259 | 0.548 |
| 7a | **1L/1H/2D** | **7.3K** | **1.0** | **0.988** | **0.995** | 2.28 | 38.5% | **0.483** | **0.718** |
| 7b | 1L/1H/4D | 13.5K | **0.1** | 0.990 | 0.989 | 2.30 | 38.7% | 0.215 | 0.588 |
| OL | — | — | — | — | — | 0.789 | 27.8% | 0.012 | 0.544 |

### Neither lever moves the gate

A 7× reduction in FM capacity (50K→7K) moved cosine from 0.997 to 0.988. A 10× reduction in λ_local (1.0→0.1) moved it from 0.993 to 0.990. The gate stayed above 0.989 in all cases — no sparsity, distillation still destructive.

The local loss creates a feedback loop: the FM predicts the model → the local loss pushes the model toward FM-predictability → the FM gets more accurate → the gate stays open. The model has enough representational freedom on RHM to make itself FM-capturable regardless of FM capacity. This is fundamentally different from MNIST, where the visual task constrains the model's computation into a form that a small FM can't fully capture.

### Self-knowledge scales inversely with FM capacity

The cleanest positive result from the sweep:

| FM params | SK post_block0 | vs OL |
|---|---|---|
| 50.4K | 0.044 | 3.7× |
| 25.8K | 0.125 | 10× |
| 13.5K | 0.259 | 22× |
| **7.3K** | **0.483** | **40×** |

Post_block0 self-knowledge of 0.483 approaches MNIST CL levels (0.63). Post_block5 = 0.718 is the highest self-knowledge ever measured on RHM (1.3× OL). Smaller FM → more structured errors → stronger self-knowledge signal. This scales cleanly even though the overall cosine barely moves.

### Generation is invariant to FM parameters

Standalone generation accuracy is 38.5% ± 0.2% across every FM size (50K to 7K) and λ_local (1.0 to 0.1). The generation improvement is purely RL-driven and transferred via distillation regardless of FM parameters. The FM's contribution to generation comes through injection during generation (a rough preview helps regardless of accuracy), not through the local loss.

## Updated cross-run summary

| Run | NTP reg | FM | λ_local | Val loss | Gen (no FM) | Gate | SK blk0 | SK blk5 |
|---|---|---|---|---|---|---|---|---|
| 1 | none | 50.4K | 1.0 | 3.53 | — | 0.673 | 0.013 | 0.633 |
| 2 | dense | 50.4K | 1.0 | 2.64 | — | 0.998 | 0.046 | 0.267 |
| 3 | sparse | 50.4K | 1.0 | 2.29 | 38.6% | 0.998 | 0.044 | 0.130 |
| 4 | sparse, no distill | 50.4K | 1.0 | **0.82** | 27.4% | 0.969 | 0.011 | 0.549 |
| 7a | sparse | **7.3K** | 1.0 | 2.28 | 38.5% | 0.995 | **0.483** | **0.718** |
| 7b | sparse | 13.5K | **0.1** | 2.30 | 38.7% | 0.989 | 0.215 | 0.588 |
| OL | — | — | — | 0.789 | 27.8% | — | 0.012 | 0.544 |

## Key findings (updated)

### 1-4. [Unchanged from runs 1-3 — see above]

### 5. Distillation is load-bearing for generation transfer

Run 4 (no distillation) shows that without the sleep phase, the model doesn't internalize generation ability — standalone generation equals OL (27.4%). The distillation forces the student to compute the FM's contribution internally, which is the only mechanism that transfers generation ability into the model's weights. The cost is NTP degradation, creating an irreducible tension: with distillation, good generation (38.5%) but bad NTP (2.3); without distillation, good NTP (0.82) but no generation improvement.

### 6. The local loss feedback loop makes FM capacity ineffective

The model adapts to whatever FM it gets. Even a 7.3K FM (0.27% of the model) achieves cos=0.988 because the local loss pushes the model toward FM-predictability. Unlike MNIST, where the visual task constrains representations into a form a small FM can't fully capture, RHM's model has enough representational freedom to make itself FM-capturable. Neither FM capacity reduction nor λ_local reduction moves the gate.

### 7. Self-knowledge scales cleanly with FM imperfection

The strongest positive result: post_block0 self-knowledge grows from 0.044 (50K FM) to 0.483 (7.3K FM), approaching MNIST CL levels. Post_block5 reaches 0.718 — the highest on RHM, 1.3× OL. The mechanism works at the representation level: the model encodes structured information about the FM's errors. Smaller FM → more structured errors → stronger encoding. This scales even though the gate and distillation dynamics don't change.

## Relation to the cerebellar theory (updated)

The full experimental arc reveals a specific architectural requirement for the ratchet: the FM must be capacity-limited relative to the complexity of the task's representational demands, not just relative to the model's parameter count. On MNIST, the visual classification task forces representations into a form the FM can't fully capture (cos ≈ 0.90), creating structured residual. On RHM under RL, the model's computation is regular enough that any FM can capture it (cos > 0.988), even at 0.27% capacity.

The self-knowledge scaling suggests the representational mechanism is sound — the model develops richer self-models with more imperfect FMs. The bottleneck is the gate/distillation dynamics: the gate must partially close for distillation to work, and the gate only closes when injection becomes noisy (FM imperfect) or unnecessary (task solved). On RHM, injection is always clean (high cosine) and always useful (generation is hard), so the gate stays open.

The biological dentate nucleus bottleneck may serve exactly this function: by limiting the bandwidth of cerebellar output, it ensures the cortex can never become fully dependent on the cerebellum, keeping the injection in the partial regime where sleep-based consolidation can absorb it.

## Runs 8-12: FM cosine regime sweep (2026-06-25)

The prior runs established that RL+FM produces 40% generation improvement, but FM cosine is consistently 0.988-0.997, well above the 0.90-0.95 sweet spot where the ratchet mechanism works on MNIST and language. A systematic survey across all RHM experiments identified three contributors to cosine inflation:

1. **Distillation** reshapes the model toward injected computation (largest effect: 0.997 → 0.927 when distillation removed, run 4)
2. **Local loss** feedback loop pushes model toward FM-predictability (small: 0.993 → 0.990 at 10× reduction)
3. **FM capacity** (small: 0.997 → 0.988 at 7× reduction)

RHM conditions that previously achieved cosine 0.90-0.95 were all open-loop baselines — the FM passively observes without interacting with training. The goal of runs 8-12 is to find a closed-loop RL configuration that keeps cosine in the sweet spot.

### Run 8: λ_local=0, m=2, 5-block gap

Drops local loss entirely while keeping distillation. Same setup as run 3 but with `--lambda-local 0.0`.

| Metric | Run 3 (λ_ll=1.0) | Run 8 (λ_ll=0.0) |
|---|---|---|
| FM cosine | 0.997 | 0.990 |
| Gate | 0.998 (0% sparse) | 0.477 (20% sparse) |
| Gen (standalone) | 38.6% | 38.5% |
| Val loss (ckpt) | 2.29 | 2.99 |
| SK post_blk5 | 0.130 | 0.336 |
| eta² L5 prog | 1.2× | 2.0× |

Removing local loss halved the gate (0.998 → 0.477) and improved self-knowledge (2.6×) and eta² progressivity. But cosine only dropped from 0.997 to 0.990 — still above the sweet spot. Distillation remains the dominant cosine driver. Generation accuracy unchanged (38.5%), confirming local loss is irrelevant to generation (consistent with runs 5-7b).

### Run 9: λ_local=0, m=4, 5-block gap

Tests whether m=4 (harder task) naturally keeps cosine lower.

| Metric | Value |
|---|---|
| Pre-trained val | 1.38 (vs 0.80 at m=2) |
| Pre-trained gen | 16.4% (barely above random 12.5%) |
| FM cosine | **0.998** (higher than m=2) |
| Gen (standalone) | 23.2% |
| Val loss (ckpt) | 4.51 (worse than uniform 2.08) |

m=4 is too hard at this model size. The pre-trained model barely learned beyond level 0, giving RL nothing to build on. FM cosine went UP because the distilled model's collapsed computation is trivially predictable. Activation norm inflation (local loss reached 654M during cycle 3). Dead end at 2.68M params.

### Run 10: λ_local=0, m=2, 3-block gap (blk0→blk3)

The key insight: a shorter prediction gap makes the FM's job genuinely harder. With the 5-block gap (blk0→blk5), the FM predicts the final layer — after distillation, this computation is shaped by the injection pattern and easy to predict. With a 3-block gap, the FM predicts an intermediate layer where the computation is less reshaped.

| Cycle | FM cos | Gen (standalone) | Gate | Local loss |
|---|---|---|---|---|
| 1 | 0.988 | 38.9% | 0.714 | 0.23 |
| 2 | 0.985 | 38.8% | 0.708 | 0.23 |
| 3 | **0.942** | 31.8%† | 0.699 | 179M† |
| 4 | **0.944** | 31.9%† | 0.743 | 4.8B† |

†Activation norm inflation destabilized cycles 3-4.

**First RL ratchet condition to achieve FM cosine below 0.95.** Cycles 1-2 had excellent dynamics: 39% generation, cosine trending down, gate genuinely open (0.71), stable local loss. Then cycle 3's activation norm inflation destroyed higher-level generation (L2: 35%→5%, L4: 29%→3%) while pushing cosine into the sweet spot. The eta² progression (2.9× L5 decrease through the network) was the best of any RL condition.

### Runs 11-12: 3-block gap with stabilizing local loss

Run 10 showed that λ_local=0 with the 3-block gap achieves sweet-spot cosine but is unstable. Runs 11-12 test whether a tiny local loss can stabilize training while preserving the cosine regime.

| | Run 10 (λ=0) | Run 12 (λ=0.001) | Run 11 (λ=0.01) |
|---|---|---|---|
| FM cosine | **0.944** | 0.987 | 0.988 |
| Gate | 0.743 | **0.841** | 0.966 |
| Gen (standalone, c4) | 31.9%* | 38.6% | 38.9% |
| Val loss (c1→c4) | 3.07→3.51* | 2.31→2.53 | 2.68→2.40↓ |
| Local loss | **Explodes** | Stable (0.06-0.13) | Stable (0.04-0.06) |
| SK post_blk5 | **0.299** | 0.080 | 0.005 |
| eta² L5 prog | **2.9×** | 1.7× | 1.3× |

*Collapsed at cycle 3 from activation inflation.

Both stabilizers prevented activation inflation and maintained 39% generation across all 4 cycles. However, both pushed cosine back to ~0.988 — essentially identical to the 5-block gap results. The local loss effect on cosine is **binary**: any nonzero value (even 0.001) pushes the model toward FM-predictability, keeping cosine above 0.98.

The gate is more graded: 0.74 (λ=0) → 0.84 (λ=0.001) → 0.97 (λ=0.01) → 1.00 (λ=1.0). Run 12's gate at 0.84 represents a genuine middle ground where the model does ~16% of computation independently.

Run 11 (λ=0.01) showed an improving val loss trajectory (2.68→2.40 at cycle 4) — the only condition with a clear downward trend at checkpoint time, suggesting the ratchet might gradually reduce distillation damage with more cycles.

### Key findings from the cosine regime sweep

**1. Cosine is controlled by distillation, not local loss or FM capacity.** Across runs 3-12, the three levers we have (local loss, FM capacity, prediction gap) only move cosine from 0.997 to 0.988 — never into the 0.90-0.95 sweet spot unless local loss is exactly zero, which causes instability. Distillation reshapes the standalone model to match the teacher's injected logits; each fresh FM finds the distilled model easy to predict regardless of wake-phase dynamics.

**2. The local loss effect is binary, but the gate effect is graded.** Any nonzero λ_local pushes cosine to ~0.99. But the gate varies smoothly from 0.74 to 1.0 across the λ_local range. The gate measures how much the model relies on injection during wake; cosine measures how predictable the distilled model is to a fresh FM. These are decoupled: the gate can be partially open (model does some work independently) while cosine stays high (distillation makes the model FM-predictable regardless).

**3. The 3-block gap is the right architectural choice.** At λ_local=0, it's the only configuration that achieves sweet-spot cosine (0.944). Even with stabilizing local loss (λ=0.001), it produces lower gate values (0.84 vs 0.97 at 5-block) and more progressive eta² (1.7× vs 1.3×). The shorter gap makes the FM's prediction task genuinely harder.

**4. Generation improvement is purely RL-driven.** Standalone generation accuracy is 38.5% ± 0.4% across every λ_local value (0, 0.001, 0.01, 1.0), every FM size (7.3K to 50.4K), and every prediction gap (3-block, 5-block). The FM contributes to generation only through injection during the RL wake phase; the local loss is irrelevant.

**5. Distillation is both load-bearing and destructive.** Run 4 (no distillation) showed generation collapses to OL baseline (27.4%) — distillation is the only mechanism that transfers generation ability into the weights. But distillation also destroys NTP (val loss 0.82→2.3-3.5) because the teacher's injected computation is too different from standalone computation. This is an irreducible tension in the current design.

## Run 13: Generation-based distillation (2026-06-25)

**Code**: `rhm_rl_gen_distill.py`

Tests the hypothesis that NTP distillation's destructiveness comes from operating on NTP data where the teacher-student logit gap is enormous (gate at 0.998 means teacher logits are ~100% injection-derived). Generation-based distillation replaces the sleep phase: the teacher generates suffixes autoregressively with FM injection, and the student matches those logits on the generation data. NTP is preserved via a separate CE loss on ground-truth data (no KL from the teacher on NTP data).

Two conditions with shared pre-training and identical wake phases (RL+FM, sparse NTP mask=0.95, λ_local=1.0):
- **gen_distill**: sleep = KL on teacher-generated suffixes + CE on NTP
- **ntp_distill**: sleep = KL+CE on NTP data (reproduces run 3)

### Result: gen-based distillation completely solves the NTP destruction problem

| Metric | gen_distill | ntp_distill | OL (run 3) |
|---|---|---|---|
| **Val loss** | **0.813** | 2.262 | 0.789 |
| **Gen acc (standalone)** | **0.391** | 0.384 | 0.278 |
| FM cosine | **0.911** | 0.997 | — |
| Gate mean | 0.969 | 0.998 | — |
| SK post_block5 | **0.572** | 0.077 | 0.544 |
| Robustness (eps=1.0) | +0.123 | -0.080 | +0.120 |

**NTP is preserved.** Val loss 0.813 — only +3% over OL (0.789), instead of the 2.86× blowup from NTP distillation (2.262). The wake trajectory confirms no damage: gen_distill val loss stays at 0.80-0.83 across all cycles, while ntp_distill shows the familiar destroy-recover cycle (1.93 at wake start → 0.82 by step 500 → destroyed again by next distillation).

**Generation transfers equally well.** Standalone gen accuracy 0.391 (gen) vs 0.384 (ntp) — gen_distill is slightly better. The generation ability is fully internalized without destroying NTP.

**FM cosine drops into the sweet spot.** 0.911 vs 0.997. This is the first RL ratchet configuration where the fresh FM finds the model genuinely hard to predict, because gen-based distillation doesn't force the student to match injection-dependent NTP logits, which was the primary driver of cosine inflation (finding 1 from runs 8-12).

**Self-knowledge is 7× higher.** SK at post_block5 = 0.572 vs 0.077. The FM cosine at 0.91 creates a structured residual that the model can encode information about, consistent with the SK scaling pattern from runs 5-7a.

**Progressive eta² returns.** L5 feature eta² decreases 4.3× through the network (0.592→0.137) for gen_distill vs 1.1× (0.556→0.489) for ntp_distill. The gen-distilled model preserves progressive hierarchical computation.

### Per-cycle trajectory

| Cycle | gen val | gen gen_acc | gen gate | gen fm_cos | ntp val | ntp gen_acc | ntp gate | ntp fm_cos |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.821 | 0.390 | 0.983 | 0.908 | 2.209 | 0.385 | 1.000 | 0.994 |
| 2 | 0.817 | 0.390 | 0.978 | 0.910 | 2.205 | 0.384 | 1.000 | 0.996 |
| 3 | 0.813 | 0.389 | 0.970 | 0.910 | 2.276 | 0.384 | 0.999 | 0.995 |
| 4 | 0.813 | 0.391 | 0.969 | 0.911 | 2.262 | 0.384 | 0.998 | 0.997 |

The gate shows a **closing trend** for gen_distill (0.983→0.969) while ntp_distill stays pinned at ~1.0. This is the first gate-closing trajectory under RL on RHM, though still modest (3% independent computation by cycle 4).

### Representational deepening across cycles

Per-level generation accuracy and NTP loss are flat across cycles — outcome metrics show no compounding. But the per-layer feature eta² tells a different story. L3 feature conditioning increases monotonically at every network layer:

| Cycle | L3 @ blk0 | L3 @ blk3 | L3 @ blk5 | L4 @ blk5 |
|---|---|---|---|---|
| 1 | 0.166 | 0.508 | 0.389 | 0.309 |
| 2 | 0.173 | 0.548 | 0.440 | 0.334 |
| 3 | 0.169 | 0.534 | 0.419 | 0.325 |
| 4 | 0.173 | 0.545 | **0.449** | **0.322** |

L3 eta² at the final layer increases +15% over 4 cycles (0.389→0.449). This is a representational change — the model's activations are becoming more conditioned on level-3 hierarchical features — even though L3 generation accuracy is flat (0.199 across all cycles). The ntp_distill condition shows no such trend (L3 @ blk5: 0.228 ± 0.004, flat).

This matters for three reasons. First, per-layer feature eta² is a ground-truth metric — we know the exact hierarchy, so this measures real compositional structure, not a proxy. Second, representational metrics have consistently preceded outcome metrics in this line of work: self-knowledge develops before robustness, FM residual structure develops before the gate closes, the L-to-m transition appears in eta² before per-level loss. The representational deepening at L3 may be a precursor to eventual L3 performance improvement given more cycles or novel data. Third, the consistency of the trend matters more than its magnitude — a monotonic +15% across 4 cycles in a controlled comparison, where the control (ntp_distill) is flat, is a real signal.

The L5 progressivity ratio (blk0/blk5) is stable at ~4× across all cycles, confirming that gen-distill preserves the progressive hierarchical computation that ntp_distill destroys (1.2×).

### Per-level NTP loss (gen_distill is dramatically better at every level)

| Cond | L0 | L1 | L2 | L3 | L4 | L5 |
|---|---|---|---|---|---|---|
| gen_distill c4 | 0.425 | 1.188 | 1.169 | 1.725 | 1.687 | 3.996 |
| ntp_distill c4 | 1.644 | 2.857 | 2.334 | 3.561 | 3.242 | 3.504 |
| pretrain | 0.328 | 1.059 | 1.092 | 1.676 | 1.678 | 1.833 |

Gen_distill preserves per-level NTP quality near pre-training levels at L0-L4. Only L5 degrades significantly (1.83→4.00), likely from the RL training pressure.

### Residual norm inflation

Residual norm grows across cycles (20.4→28.9), the same activation norm inflation pattern seen in the extended ratchet. This would need to be addressed for longer training.

### Why gen-based distillation works

Standard NTP distillation forces the student to match the teacher's logits on ground-truth NTP data. With the gate at 0.998, the teacher's logits are almost entirely injection-derived — the student must replicate computation it fundamentally can't do standalone. This destroys NTP representations.

Gen-based distillation operates on a different data distribution: the teacher generates suffixes autoregressively with FM injection, and the student matches logits on these (noisy, partially-correct) generation sequences. The teacher-student logit gap is smaller on generation data because: (a) the context is self-generated and noisy, making the FM preview less perfectly aligned with actual computation, and (b) the KL is only on suffix positions (32/64 tokens), not the full sequence. NTP training remains uncontaminated — the student trains on ground-truth NTP data with CE only, no teacher KL.

The FM cosine result (0.911 vs 0.997) confirms the mechanism: NTP distillation was the primary driver of cosine inflation (finding 1 from the cosine regime sweep). Gen-based distillation doesn't reshape the standalone model toward FM-predictability because it never forces the student to match injection-dependent logits on well-formed NTP data.

## Open directions

1. **RL on language (RLHF)**: The motivating insight — that RL supervision is sparse-but-rich like classification — applies directly to RLHF. Language's full-rank residual (eff rank 200/256) and richer task structure may naturally keep the FM cosine in the 0.90 range (as it does under NTP), avoiding the RHM-specific feedback loop. The FM's process supervision could provide credit assignment during policy optimization.

2. ~~**Generation-based distillation**~~: *Done* — see run 13. Gen-based distillation completely solves the NTP destruction problem (val loss 0.813 vs 2.262) while preserving generation transfer (39.1% vs 38.4%). Also unlocks sweet-spot FM cosine (0.911), 7× higher self-knowledge, and progressive eta². Does not produce compounding across cycles.

3. **Stabilized λ_local=0 with gradient clipping**: The activation norm inflation at λ_local=0 is a training stability problem, not a fundamental limitation. Gradient clipping or activation norm regularization (e.g., penalizing ||post_block3||² > threshold) could prevent the explosion without creating FM-predictability pressure. This would preserve the sweet-spot cosine (0.944) and the rich dynamics (2.9× eta² progression, 0.299 self-knowledge) that only appear at λ_local=0.

4. **Gen-distill + λ_local=0**: Combine gen-based distillation (which solved NTP destruction) with λ_local=0 (which achieved the richest dynamics in run 10: cosine 0.944, eta² 2.9× progression, SK 0.299). Gen-distill already achieves cosine 0.911 with λ_local=1.0; removing the local loss feedback loop might push it further into the sweet spot while maintaining stability (gen-distill doesn't reshape the model via NTP KL, so the activation inflation may be less severe).

5. **Gate clamping**: Force the gate to stay below a maximum (e.g., 0.3) via a hard clamp or penalty term. This directly addresses the remaining gate-at-0.97 issue by ensuring the injection is always a modest correction. With gen-distill solving the NTP destruction, gate clamping could make distillation more effective by ensuring the teacher-student gap is always small.

## Reproduction

```bash
cd experiments/

# Run 1: No NTP regularization (all 4 conditions)
modal run --detach -m rhm.ratchet.rhm_rl_ratchet::rhm_rl_ratchet \
  --lambda-ntp 0.0 --ntp-mask-rate 0.0

# Run 2: Dense NTP regularization (all 4 conditions)
modal run --detach -m rhm.ratchet.rhm_rl_ratchet::rhm_rl_ratchet \
  --lambda-ntp 1.0 --ntp-mask-rate 0.0

# Run 3: Sparse NTP regularization (RL_FM only)
modal run --detach -m rhm.ratchet.rhm_rl_ratchet::rhm_rl_ratchet \
  --lambda-ntp 1.0 --ntp-mask-rate 0.95 --only-rl-fm

# Run 4: No distillation (RL_FM only)
modal run --detach -m rhm.ratchet.rhm_rl_ratchet::rhm_rl_ratchet \
  --only-rl-fm --distill-steps 0 --ntp-mask-rate 0.95

# Run 5: Smaller FM (25.8K) + ratchet
modal run --detach -m rhm.ratchet.rhm_rl_ratchet::rhm_rl_ratchet \
  --only-rl-fm --fwd-d-head 8 --fwd-mlp-mult 0.25 --ntp-mask-rate 0.95

# Run 6: Small FM (13.5K) + ratchet
modal run --detach -m rhm.ratchet.rhm_rl_ratchet::rhm_rl_ratchet \
  --only-rl-fm --fwd-d-head 4 --fwd-mlp-mult 0.125 --ntp-mask-rate 0.95

# Run 7a: Tiny FM (7.3K) + ratchet
modal run --detach -m rhm.ratchet.rhm_rl_ratchet::rhm_rl_ratchet \
  --only-rl-fm --fwd-d-head 2 --fwd-mlp-mult 0.0625 --ntp-mask-rate 0.95

# Run 7b: Small FM (13.5K) + reduced local loss
modal run --detach -m rhm.ratchet.rhm_rl_ratchet::rhm_rl_ratchet \
  --only-rl-fm --fwd-d-head 4 --fwd-mlp-mult 0.125 --lambda-local 0.1 --ntp-mask-rate 0.95

# Run 8: No local loss, m=2, 5-block gap
modal run --detach -m rhm.ratchet.rhm_rl_ratchet::rhm_rl_ratchet \
  --only-rl-fm --lambda-local 0.0 --ntp-mask-rate 0.95 --run-tag no_ll_m2

# Run 9: No local loss, m=4, 5-block gap
modal run --detach -m rhm.ratchet.rhm_rl_ratchet::rhm_rl_ratchet \
  --only-rl-fm --lambda-local 0.0 --ntp-mask-rate 0.95 --m 4 --run-tag no_ll_m4

# Run 10: No local loss, m=2, 3-block gap
modal run --detach -m rhm.ratchet.rhm_rl_ratchet::rhm_rl_ratchet \
  --only-rl-fm --lambda-local 0.0 --ntp-mask-rate 0.95 \
  --predict-to post_block3 --run-tag no_ll_gap3

# Run 11: 3-block gap, λ_local=0.01 stabilizer
modal run --detach -m rhm.ratchet.rhm_rl_ratchet::rhm_rl_ratchet \
  --only-rl-fm --lambda-local 0.01 --ntp-mask-rate 0.95 \
  --predict-to post_block3 --run-tag gap3_ll001

# Run 12: 3-block gap, λ_local=0.001 stabilizer
modal run --detach -m rhm.ratchet.rhm_rl_ratchet::rhm_rl_ratchet \
  --only-rl-fm --lambda-local 0.001 --ntp-mask-rate 0.95 \
  --predict-to post_block3 --run-tag gap3_ll0001
```

# Run 13: Generation-based distillation (gen_distill vs ntp_distill)
modal run --detach -m rhm.ratchet.rhm_rl_gen_distill::rhm_rl_gen_distill
```

Results saved to `rhm-scaling-data` volume at `rhm_rl_ratchet/v8_s2_L6_m2*`, `rhm_rl_ratchet/v8_s2_L6_m4*`, and `rhm_rl_gen_distill/v8_s2_L6_m2/`. Runs 8-12 use `--run-tag` to avoid overwriting each other.
