# RHM Sparsity Sweep: Process Supervision as a Function of NTP Sparsity (2026-06-24)

**Code**: `rhm_sparsity_sweep.py`
**Prior experiments**: [RHM ratchet](RHM_RATCHET_README.md), [MNIST local loss](../../a2a_forward/MNIST_LOCAL_LOSS_README.md), [Language ratchet](../../a2a_forward/LANGUAGE_RATCHET_README.md)

## Goal

Test the supervision density hypothesis: does FM process supervision (local loss) help more when NTP task supervision is sparser? This directly tests the proposed explanation for why the A2A ratchet produced 48% improvement on MNIST (sparse CLS-only supervision) but only 0.3–1.6% on RHM/language (dense per-position NTP).

## Background

The A2A forward model co-trains a small "cerebellum" FM alongside the main model. The FM predicts future activations (post_block5 from post_block0). When used as a local loss — `λ · MSE(post_block5, sg(FM(post_block0)))` — it provides *process supervision*: gradient telling the model how to compute (make intermediate computation FM-predictable), orthogonal to the NTP's *task supervision* (what to output at each position).

On MNIST, where NTP only supervises the CLS position, this process supervision provides 50× more gradient dimensions than the task loss. On RHM/language, where NTP supervises all 64 positions, the process supervision is marginal. This experiment varies supervision density as a controlled independent variable to test whether it's the causal factor.

## Design

Ten conditions: {OL, LL} × {mask_rate = 0.0, 0.5, 0.75, 0.9, 0.95}.

At each training step, a random fraction `mask_rate` of positions have their NTP loss zeroed. The FM local loss applies at ALL positions regardless of mask. Evaluation always uses full unmasked NTP for fair comparison.

- **OL**: masked NTP only
- **LL**: masked NTP + `λ · MSE(post_block5, sg(FM(post_block0)))`

No injection, no gate, no distillation, no ratchet cycles. The FM co-trains in the LL condition (gradient does not flow from FM loss to main model). A fresh FM is trained post-hoc for the OL condition.

Architecture: 6L/6H/192D GPT (~2.68M params), 1L/1H/16D causal FM (50.4K params, 1.9% of main model), post_block0 → post_block5 (5-block gap, 83% of model depth). DGP: L=6, m=2, v=8, s=2 (seq_len=64). 15,000 training steps, λ=1.0. Identical initial weights, batch indices, and position masks between OL and LL at each mask rate. m=2 chosen because the model learns 3–4 hierarchy levels at this setting, providing compositional headroom.

## Results

### Val loss: monotonic crossover from LL-hurts to LL-helps

| mask | OL | LL | gap | gap% |
|---|---|---|---|---|
| 0.00 | 0.7904 | 0.7974 | -0.007 | -0.9% |
| 0.50 | 0.7956 | 0.8017 | -0.006 | -0.8% |
| 0.75 | 0.8054 | 0.8097 | -0.004 | -0.5% |
| 0.90 | 0.8401 | 0.8376 | **+0.003** | **+0.3%** |
| 0.95 | 0.8823 | 0.8681 | **+0.014** | **+1.6%** |

(Gap = OL − LL. Positive = LL wins. All eval is unmasked.)

The gap transitions monotonically from LL-hurts (−0.9%) to LL-helps (+1.6%). Crossover between mask=0.75 and mask=0.90. At mask=0.95, the FM's process supervision recovers 1.6% of val loss that the sparse NTP couldn't provide — matching the language ratchet's improvement magnitude, but achieved with a single training pass and no ratchet machinery.

### Per-level loss: process supervision helps compositional levels when they're starved for gradient

Per-level loss delta (LL − OL; negative = LL learned more at that level):

| mask | ΔL0 | ΔL1 | ΔL2 | ΔL3 | ΔL4 | ΔL5 |
|---|---|---|---|---|---|---|
| 0.00 | +0.006 | +0.005 | +0.004 | +0.006 | +0.009 | +0.004 |
| 0.50 | +0.005 | +0.005 | +0.010 | +0.017 | +0.021 | +0.000 |
| 0.75 | +0.001 | +0.009 | +0.005 | +0.006 | +0.021 | +0.001 |
| 0.90 | -0.001 | **-0.011** | +0.010 | +0.002 | +0.007 | -0.000 |
| 0.95 | **-0.013** | **-0.025** | **-0.022** | -0.004 | +0.001 | **-0.020** |

At mask=0.0, LL is uniformly slightly worse at every level — the process supervision adds optimization noise when NTP already constrains all levels. At mask=0.95, LL is better at every level except L4 (tied). The largest improvements are at L1 (−0.025) and L2 (−0.022), which are the compositional levels the model is actively learning (L0 is near-saturated at ~0.33; L3–L5 are near the uniform baseline of ln(8)=2.08).

This resolves a question from the ratchet experiments: the ratchet showed no per-level compositional improvement at mask=0.0 (the RHM m=2 ratchet), leading to the conclusion that the ratchet can't drive compositional depth. The sparsity sweep shows this was because dense NTP already provides sufficient per-level gradient — the process supervision had nothing to add. When NTP is sparse enough that compositional levels are under-constrained, the FM's process supervision fills the gap and improves those levels specifically.

The mechanism: with dense supervision, there's enough gradient at each position to learn each hierarchy level independently. With sparse supervision, the model lacks direct gradient at most positions, leaving intermediate computation under-constrained — many possible internal representations are consistent with the few supervised positions. The FM's process supervision selects among those for the one whose intermediate computation is most regular/compressible, and on a hierarchical DGP, regular computation aligns with compositional structure.

### OL robustness improves dramatically with masking

| mask | OL eps=0.5 | OL eps=1.0 | OL eps=2.0 |
|---|---|---|---|
| 0.00 | +0.017 | +0.189 | +0.612 |
| 0.50 | +0.007 | +0.093 | +0.399 |
| 0.75 | +0.003 | +0.031 | +0.195 |
| 0.90 | +0.003 | +0.020 | +0.118 |
| 0.95 | +0.001 | +0.005 | +0.025 |

Pure open-loop robustness improves 38× (eps=1.0) from mask=0.0 to mask=0.95. No self-referential architecture, no local loss — just sparser NTP. The loss landscape becomes dramatically flatter when fewer positions are supervised, because perturbations to intermediate dimensions that don't affect supervised outputs have no effect on loss.

This reveals a cost of dense supervision: dense NTP creates a loss landscape where the model is sensitive along all intermediate dimensions (every dimension contributes to predicting every position). Sparse supervision produces models that use a lower-dimensional subspace of their intermediate representations — less sample-efficient, but inherently more robust to perturbations.

### Local-loss-induced brittleness at all mask rates

| mask | OL eps=1.0 | LL eps=1.0 | ratio |
|---|---|---|---|
| 0.00 | +0.019 | +0.663 | 35× |
| 0.50 | +0.009 | +0.603 | 65× |
| 0.75 | +0.003 | +0.468 | 151× |
| 0.90 | +0.002 | +0.213 | 109× |
| 0.95 | +0.001 | +0.136 | 28× |

LL is 28–151× more brittle than OL across all mask rates. The "be predictable" pressure concentrates computation, making the model sensitive to perturbations in the dimensions it uses. This replicates the cross-domain finding from MNIST and language: process supervision helps val loss but hurts robustness.

The absolute brittleness decreases with mask rate for both conditions, reflecting the underlying loss landscape flattening from sparse supervision.

### FM cosine and residual structure

| Condition | cos | norm | rank% | top1% |
|---|---|---|---|---|
| OL mask=0.00 | 0.905 | 28.3 | 54.4 | 26.3 |
| LL mask=0.00 | 0.957 | 1.0 | 82.4 | 14.9 |
| OL mask=0.50 | 0.909 | 29.9 | 53.8 | 23.6 |
| LL mask=0.50 | 0.962 | 1.1 | 81.2 | 16.8 |
| OL mask=0.75 | 0.907 | 32.9 | 53.4 | 20.5 |
| LL mask=0.75 | 0.968 | 1.2 | 79.5 | 17.5 |
| OL mask=0.90 | 0.919 | 36.9 | 47.7 | 30.3 |
| LL mask=0.90 | 0.986 | 1.4 | 75.0 | 17.4 |
| OL mask=0.95 | 0.936 | 40.0 | 42.6 | 29.5 |
| LL mask=0.95 | 0.992 | 1.5 | 67.9 | 18.2 |

LL compresses residual norms ~25× (the "be predictable" pressure is effective at all mask rates). Both OL and LL FM cosine increase with mask rate — sparser supervision produces simpler, more predictable computation in both conditions.

OL effective rank decreases with masking (54% → 43%), while LL stays higher (82% → 68%). The OL model's computation becomes more concentrated (fewer dimensions doing the work) under sparse supervision, while LL maintains higher-rank computation because the FM encourages regularity across all dimensions.

## Key findings

### 1. Supervision density is the causal variable

The monotonic crossover from LL-hurts to LL-helps, with all other variables controlled (same architecture, same FM, same data, same initial weights, same batch order, same position masks), confirms that supervision density determines whether process supervision helps. Dense NTP makes the FM's intermediate supervision redundant; sparse NTP leaves room for the FM to provide useful information.

### 2. Process supervision helps compositional levels under sparse supervision

At mask=0.95, LL improves over OL at L0 (−0.013), L1 (−0.025), L2 (−0.022), and L5 (−0.020). The improvement is strongest at the compositional levels the model is actively learning (L1, L2). This shows that process supervision can drive compositional depth when the task supervision is sparse enough — the ratchet's null result on compositional depth was specific to the dense-NTP regime.

### 3. Dense supervision has a robustness cost

Pure OL models become 38× more robust at mask=0.95 vs mask=0.0. This is a property of the loss landscape geometry: dense supervision constrains all intermediate dimensions, making the model sensitive to perturbations everywhere. Sparse supervision leaves most dimensions unconstrained.

### 4. Local-loss-induced brittleness is independent of supervision density

LL is 28–151× more brittle than OL at every mask rate. The "be predictable" pressure makes the model sensitive to perturbations in the dimensions it uses for computation, regardless of how many positions receive NTP gradient.

## Implications

### For the A2A ratchet

The ratchet's weak results on language and RHM were not because the architecture is wrong or the DGP is unfavorable — they were because dense NTP already provides all the supervision the model needs. The ratchet is most useful in regimes where task supervision is sparse: classification tasks (MNIST), RL fine-tuning (sparse reward), few-shot adaptation, or any setting where direct supervision doesn't reach intermediate representations.

### For the per-level compositional result

The m=2 ratchet's failure to improve compositional depth was specifically a dense-NTP phenomenon. Under sparse supervision, the FM's process supervision fills gradient gaps at compositional levels. This suggests the ratchet could drive compositional deepening in tasks where some hierarchy levels lack direct gradient — e.g., tasks where only the final output is supervised but the solution requires multi-step reasoning.

### For the robustness of dense NTP

The 38× OL robustness improvement from mask=0.0 to mask=0.95 suggests that dense, position-level NTP training creates unnecessarily fragile loss landscapes. Standard NTP forces every intermediate dimension to participate in every prediction, producing models that are maximally sensitive to perturbations. Sparse or partial supervision could produce more robust models at a moderate sample-efficiency cost.

## Reproduction

```bash
cd experiments/

# Full sweep (5 mask rates × 2 conditions, ~4-5 hours on L4)
modal run --detach -m rhm.ratchet.rhm_sparsity_sweep::rhm_sparsity_sweep

# Custom mask rates
modal run --detach -m rhm.ratchet.rhm_sparsity_sweep::rhm_sparsity_sweep \
  --mask-rates "0.0,0.8,0.9,0.95,0.98"

# Different m (e.g., m=4 where model learns fewer levels)
modal run --detach -m rhm.ratchet.rhm_sparsity_sweep::rhm_sparsity_sweep --m 4
```

Results saved to `rhm-scaling-data` volume at `/data/rhm_sparsity_sweep/v8_s2_L6_m2/results.json`.
