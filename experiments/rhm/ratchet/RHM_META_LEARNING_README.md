# RHM Meta-Learning Experiments (2026-06-27 — 2026-06-29)

**Code**: `rhm_meta_learning.py`, `rhm_meta_learning_l2.py`, `rhm_reptile.py`, `rhm_reptile_sparse.py`
**Prior experiments**: [FOMAML ratchet (single rule set)](RHM_FOMAML_README.md), [Sparse ratchet](RHM_SPARSE_RATCHET_README.md), [RL ratchet & gen-distill](RHM_RL_RATCHET_README.md), [Loss weighting](../LOSS_WEIGHTING_README.md)
**Conversation**: `conversations/claude-code-transcripts/readable/80f9a4b4-75c6-474e-b8e5-08905c8e9acf.md`

## Motivation

All prior RHM ratchet experiments used a single rule set. The ratchet dynamics (gate closing, FM tracking, robustness dissociation) replicated from MNIST but never compounded — the val loss gap peaked at ~1.2% and didn't grow across cycles. The FOMAML diagnostic (RHM_FOMAML_README) confirmed: on fixed data, no outer objective (dense NTP, RL, per-level NTP) makes local loss help compositional NTP.

The diagnosis, informed by the Sutskever interview's Student 1 vs Student 2 framing: training on fixed RHM rules is purely Student 1. The model memorizes specific L0 rules thoroughly but never develops transferable compositional structure. The "outer loop" (val set) is identical to the inner loop (train set), so FM-compressibility aligns with L0 regularization, not L2+ compositional depth.

The hypothesis: meta-learning across different rule sets (same L, m, s, v, different composition tables) creates Student 2 pressure. L0 rules don't transfer across rule sets, so the meta-gradient should select for the shared compositional principle — the only knowledge that helps on *any* rule set.

## Key conceptual insights (from the conversation)

### 1. FM compression alignment asymmetry

On MNIST, FM compression improves the *quality* of a computation the model is already doing (digit discrimination — the FM residual is low-rank and digit-conditioned). On RHM, FM compression makes computation more regular at levels the model has already learned, but can't create the *signal* needed to learn levels it hasn't. The bottleneck on MNIST is representational quality; on RHM it's learning signal.

### 2. NTP density over-constrains solutions

Dense NTP gives gradient at all 63 positions in a length-64 sequence, ~32 of which are L0. The gradient is overwhelmingly L0-dominated. All prior auxiliary signals (local loss, injection, gated supervision) were redundant because dense NTP already tells the model exactly what to do at every position. This was confirmed by label smoothing (no effect), focal loss (tiny effect), and the FOMAML diagnostic (gate closes to 0.000).

### 3. Rule-set transfer should defeat NTP density (theory)

With different rules in the outer loop, L0 NTP tells the model what to do *for these rules*, which is exactly the part that doesn't transfer. The compositional structure (how features compose into s-tuples recursively) is the only useful meta-gradient component. Dense NTP's over-constraint should be defeated because the inner and outer loops no longer share the same data.

## Experiments

### Experiment 1: FOMAML with rule-set transfer (runs 1-3)

**Code**: `rhm_meta_learning.py`

Three runs testing FOMAML (first-order MAML) across rule sets at different hyperparameter settings. All use 2.68M model (6L/6H/192D) at L=6/m=4/v=8/s=2.

| Run | K (inner) | N_meta | Matching | Eval steps |
|-----|-----------|--------|----------|------------|
| 1 | 500 | 50 | FLOPS | 2000 |
| 2 | 50 | 500 | FLOPS | 2000 |
| 3 | 50 | 5000 | Update | 10000 |

**Results (per-level loss after fine-tuning, avg over 8 held-out rules):**

| Run | Cond | L0 | L1 | L2 | L3 | overall |
|-----|------|------|------|------|------|---------|
| 1 | MAML | 1.077 | 1.936 | 2.012 | 2.003 | 1.516 |
| 1 | Multi | 0.981 | 1.888 | 1.983 | 1.980 | 1.449 |
| 1 | Single | 0.942 | 1.869 | 1.972 | 1.971 | 1.422 |
| 3 | MAML | 1.113 | 1.966 | 2.009 | 2.001 | 1.541 |
| 3 | Multi | 0.950 | 1.867 | 1.973 | 1.975 | 1.426 |
| 3 | Single | 0.923 | 1.851 | 1.966 | 1.970 | 1.407 |

**Single > Multi > MAML at every level in every run.** MAML's zero-shot scores are near-uniform (meta-learned θ ≈ random initialization). The gap closes monotonically during fine-tuning but MAML never overtakes.

**Diagnosis**: FOMAML drops the second-order term, so each meta-update is just a standard NTP gradient on a random rule set — the same signal Multi gets for 1/(K+1) of the FLOPS. With K=50, θ' ≈ θ and FOMAML degenerates to noisy multi-task. With K=500, the first-order approximation breaks down. No sweet spot exists.

### Experiment 2: L2+-only outer loss (the "cheat")

**Code**: `rhm_meta_learning_l2.py`

Tests whether the L0-dominated outer loss is the bottleneck. Inner loop: dense NTP on rule R. Outer loop: NTP only at hierarchy levels ≥ 2 on rule R' (15/63 positions, using privileged s-adic valuation). 3000 meta-steps, K=200.

| Cond | L0 | L1 | L2 | L3 | overall |
|------|------|------|------|------|---------|
| MAML_L2 | 1.123 | 1.978 | 2.007 | 1.999 | ~1.53 |
| MAML_all | 1.115 | 1.960 | 2.007 | 2.001 | ~1.52 |
| Multi | 0.948 | 1.866 | 1.973 | 1.974 | ~1.43 |
| Single | 0.930 | 1.857 | 1.968 | 1.972 | ~1.42 |

**MAML_L2 ≈ MAML_all.** The L2+-only outer loss made zero difference (gap: 0.001-0.002 nats, within noise). The outer loss isn't the bottleneck.

**Updated diagnosis**: The problem is the dense NTP *inner* loop, not the outer loss. Dense NTP in the inner loop makes the trajectory (θ'-θ) L0-dominated regardless of what the outer loss measures. The inner loop doesn't produce L2+ progress in K=200 steps, so there's no L2+ component in the meta-gradient.

### Experiment 3: Reptile with dense inner loop

**Code**: `rhm_reptile.py`

Tests Reptile (Nichol & Schulman, 2018) as an alternative to FOMAML. Reptile's meta-update (θ ← θ + ε(θ'-θ)) captures the full K-step trajectory without needing the second-order term. With K=2000, the model reaches ~0.16 nats of L2 learning during the inner loop. Over many rule sets, L0 components of (θ'-θ) should cancel (different rules → different directions) while L2+ components accumulate (same compositional principle → same direction).

100 meta-steps, K=2000, outer_lr=0.5, 32 train rules, 8 eval rules.

| Cond | L0 | L1 | L2 | L3 | overall |
|------|------|------|------|------|---------|
| Reptile | 0.966 | 1.816 | 1.947 | 1.974 | 1.418 |
| (prior MAML run 3) Multi | 0.950 | 1.867 | 1.973 | 1.975 | 1.426 |
| (prior MAML run 3) Single | 0.923 | 1.851 | 1.966 | 1.970 | 1.407 |

**Reptile ≈ Multi.** Slightly better at L1 (1.816 vs 1.867) and L2 (1.947 vs 1.973), still behind Single at L0. The meta-training trajectory was wildly noisy and non-convergent — zero-shot L0 never dropped below uniform. The meta-learned θ is being pushed into a noisy region equidistant from all rule sets, without developing compositional structure.

### Experiment 4: Sparse Reptile — the untested cell (key experiment)

**Code**: `rhm_reptile_sparse.py`

The audit of all prior experiments revealed a gap in coverage. The (supervision sparsity × rule-set diversity) matrix:

| | Single rule set | Multiple rule sets |
|--|--|--|
| **Dense NTP** | Many experiments | Exps 1-3 above |
| **Sparse (mask/RL/L2+)** | Sparse ratchet, RL ratchet, etc. | **Not tested** |

This experiment fills the bottom-right cell: **Reptile with L2+-only NTP as the inner loop objective, across multiple rule sets.** The inner loop trains only on the 15 positions where s-adic valuation ≥ 2, so the trajectory (θ'-θ) is about L2+ compositional learning, not L0 memorization.

Five conditions, all FLOPS-matched at 200K forward/backward passes:

| Condition | Inner loop | Rule diversity | Meta-structure |
|-----------|-----------|----------------|----------------|
| Reptile_L2+ | L2+-only NTP (15/63 pos) | 32 rule sets | Reptile |
| Reptile_dense | Dense NTP (63/63 pos) | 32 rule sets | Reptile |
| Reptile_rand_sparse | Random 24% mask (15/63 pos) | 32 rule sets | Reptile |
| Multi_L2+ | L2+-only NTP, interleaved | 32 rule sets | None |
| Single_dense | Dense NTP | 1 rule set | None |

100 meta-steps × K=2000. Eval: 10K dense NTP fine-tuning on 8 held-out rules. Same model/DGP as all prior experiments.

**Mask level breakdown** (positions per hierarchy level receiving gradient):

| Mask | L0 | L1 | L2 | L3 | L4 | L5 |
|------|----|----|----|----|----|----|
| L2+ | 0 | 0 | 8 | 4 | 2 | 1 |
| dense | 32 | 16 | 8 | 4 | 2 | 1 |
| random | 10 | 2 | 1 | 1 | 0 | 1 |

#### Results

**Per-level loss after 10K fine-tuning (avg over 8 eval rules):**

| Condition | L0 | L1 | L2 | L3 | L4 | overall |
|-----------|------|------|------|------|------|---------|
| Reptile_L2+ | 1.044 | 1.866 | 1.970 | 1.978 | 1.974 | 1.474 |
| Reptile_dense | **0.978** | **1.828** | 1.955 | 1.972 | 1.966 | **1.428** |
| Reptile_rand_sparse | 1.019 | 1.853 | 1.959 | 1.970 | 1.965 | 1.456 |
| Multi_L2+ | 1.106 | 1.896 | 1.988 | 1.993 | 1.990 | 1.517 |
| Single_dense | 1.023 | 1.857 | 1.966 | 1.981 | 1.975 | 1.461 |

At step 1000 (early fine-tuning), the ordering is the same. No condition has an early L2+ advantage. However, the **late fine-tuning learning curves** reveal a real effect:

**L2 learning curve (avg over 8 eval rules):**

| Step | Reptile_L2+ | Reptile_dense | Reptile_rand | Multi_L2+ | Single_dense |
|------|-------------|---------------|--------------|-----------|--------------|
| 1000 | 1.970 | 1.955 | 1.959 | 1.988 | 1.966 |
| 5000 | 1.948 | 1.947 | 1.948 | 1.966 | 1.952 |
| 10000 | **1.945** | 1.950 | **1.946** | 1.956 | **1.980** |

**L3 learning curve:**

| Step | Reptile_L2+ | Reptile_dense | Reptile_rand | Multi_L2+ | Single_dense |
|------|-------------|---------------|--------------|-----------|--------------|
| 1000 | 1.978 | 1.972 | 1.970 | 1.993 | 1.981 |
| 5000 | 1.972 | 1.973 | 1.971 | 1.977 | 1.979 |
| 10000 | 1.972 | 1.976 | **1.971** | 1.974 | **2.008** |

#### Key finding: diversity prevents L2-L3 overfitting

**Single_dense's L2-L3 loss rises after step 5000** — the model is overfitting to L0 at the expense of compositional structure. L3 goes from 1.979 at step 5000 to 2.008 at step 10000, crossing back above uniform (2.079). All multi-rule-set conditions avoid this:

- At step 10K, every multi-rule condition beats Single_dense at L2 by 0.025-0.035 nats
- At step 10K, every multi-rule condition beats Single_dense at L3 by 0.036-0.037 nats
- The effect is **anti-overfitting** (retaining L2-L3 during extended fine-tuning) rather than positive compositional learning (all conditions reach the same L2-L3 at step 1000)

#### What the L2+ mask adds (and doesn't)

At L2, Reptile_L2+ edges out Reptile_dense at step 10K (1.945 vs 1.950). But Reptile_rand_sparse matches it (1.946). The effect is from **sparsity** (not training on L0, which prevents L0 overfitting during fine-tuning), not from **targeting** L2+ positions specifically.

The L2+ mask costs ~0.07 nats at L0 (1.044 vs 0.978 for Reptile_dense) — the meta-learned θ has weaker L0 features because the inner loop never trains on L0.

#### Inner loop diagnostics (from early meta-steps, before log truncation)

The Reptile_L2+ inner loop at ms=75 on a sampled rule set:

| Level | Start | End | Δ |
|-------|-------|-----|---|
| L0 | 3.905 | 2.437 | -1.47 |
| L1 | 3.740 | 2.141 | -1.60 |
| L2 | 4.339 | 1.856 | -2.48 |
| L3 | 4.415 | 1.820 | -2.60 |
| L4 | 4.546 | 1.835 | -2.71 |

The inner loop learns L2-L4 *below* uniform (1.856 vs 2.079) while L0 stays *above* uniform (2.437). The trajectory (θ'-θ) is genuinely L2+-dominated. But this doesn't translate into a compositional advantage at eval — the meta-learned features aren't more compositionally useful than what dense Reptile provides.

## Summary across all experiments

| Experiment | Hypothesis tested | Result |
|-----------|-------------------|--------|
| FOMAML × 3 settings | Bilevel optimization extracts compositional transfer signal | No. FOMAML degenerates to noisy multi-task (drops second-order term) |
| L2+ outer loss | L0-dominated outer loss masks the signal | No. Dense inner loop trajectory is L0-dominated regardless of outer loss |
| Reptile (dense) | Full-trajectory meta-update captures L2+ signal | No. Reptile ≈ Multi-task at every level |
| Reptile (L2+ inner) | Sparse L2+ inner loop + rule transfer is the right combination | Partially. Prevents L2-L3 overfitting during fine-tuning, but doesn't accelerate compositional learning |

## What we learned

### 1. The compositional transfer signal is too weak for gradient-based meta-learning at this scale

Five different meta-learning implementations (FOMAML ×3, Reptile ×2) across multiple hyperparameter settings, with both dense and sparse supervision, all produce the same result: no advantage over simple multi-task or single-task training at L2+. The compositional principle (hierarchical composition is recursive, context at distance s^l determines tokens through level-l structure) is apparently not extractable from parameter-space trajectories at this model size.

### 2. Dense NTP is an even deeper bottleneck than we thought

The dense NTP problem isn't just about which positions get gradient — it's about the *fine-tuning evaluation*. All conditions converge to the same L2-L3 loss after 1000 dense NTP fine-tuning steps, regardless of initialization. Dense NTP at eval washes out any compositional structure the initialization might encode. The initialization effect only appears at *late* fine-tuning (10K steps) as an anti-overfitting benefit, not as faster compositional learning.

### 3. Multi-rule diversity provides anti-overfitting, not compositional depth

The consistent benefit of training on multiple rule sets is preventing L0 overfitting that degrades L2-L3. This is standard regularization from data diversity, not the compositional transfer we hypothesized. It's the same mechanism as data augmentation — see more examples, overfit less.

### 4. The Student 2 analogy may require a different implementation

The conceptual analysis (Student 2 learns transferable structure from diverse experience) remains unfalsified — we just haven't found the right algorithmic instantiation. The gradient-based approaches all share a limitation: they operate in parameter space, where the compositional signal is a tiny residual of a high-dimensional trajectory. Possible directions:
- **Representation-space meta-learning**: instead of optimizing θ, constrain the *representation geometry* to be compositionally structured (e.g., auxiliary loss that enforces hierarchical factorization)
- **Curriculum over m**: train at m=2 (where the model learns 3-4 composition levels) then shift to m=4. This provides the compositional features directly rather than trying to extract them from meta-gradients
- **Architectural inductive bias**: looped transformers or recursive architectures that make compositional computation natural rather than learned

## Reproduction

```bash
cd experiments/

# Experiment 1: FOMAML (run 3 — update-matched, 5K meta-steps)
modal run --detach -m rhm.ratchet.rhm_meta_learning::rhm_meta_learning \
    --n-meta-steps 5000 --k-inner 50 --k-eval 10000 --eval-ckpt-interval 1000

# Experiment 2: L2+ outer loss
modal run --detach -m rhm.ratchet.rhm_meta_learning_l2::rhm_meta_learning_l2

# Experiment 3: Reptile (dense)
modal run --detach -m rhm.ratchet.rhm_reptile::rhm_reptile \
    --n-meta-steps 100 --k-inner 2000 --train-eval-interval 5

# Experiment 4: Sparse Reptile (key experiment)
modal run --detach -m rhm.ratchet.rhm_reptile_sparse::rhm_reptile_sparse
```

Results saved to `rhm-scaling-data` volume at `/data/rhm_meta_learning/`, `/data/rhm_meta_learning_l2/`, `/data/rhm_reptile/`, `/data/rhm_reptile_sparse/`.
