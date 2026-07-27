# RHM Residual Rank Experiment (2026-06-20)

> ⚠️ **Superseded as the instrument of record (2026-07-27).** The entropy effective rank used throughout this file reads the
> residual's *shape* and is blind to its *magnitude* — which is why Exp. 3 below sees a 17x residual-norm range leave rank flat at
> 90-96%. That is now explained rather than contradicted: the residual follows `res_var(i) ~ act_var(i)^beta` across the model's
> principal directions, and the FM-capacity sweep moves the *level* while leaving the *exponent* fixed. Exp. 3's conclusion ("the
> residual's shape is a property of the main model's computation") **stands and is strengthened** — beta is invariant to capacity
> to +/-0.01. But do not use effective rank for new measurements; use beta and `R_res_participation` from
> [`residual_decomposition/README.md`](residual_decomposition/README.md). Note also that every gap here is 1 block, where the FM
> saturates and the leftover is numerical noise.

**Code**: `rhm_residual_rank.py`
**Prior experiment**: See `SWEEP_README.md` for RHM scaling exponent results; see `../a2a_forward/README.md` for the A2A forward model background.

## Goal

Test whether the effective rank of a forward self-model's (FSM) residual reflects the complexity of the data-generating process (DGP). Cross-domain observations from the A2A forward model experiments suggested a relationship (grokking ~15/128, MNIST ~18/128, language ~200/256), but those comparisons are confounded by everything else changing between domains. The RHM gives clean control over DGP complexity (L, m) while holding model architecture, vocabulary, and training procedure fixed.

## Design

### Experiment 1: DGP sweep (L x m)

Co-train a 4L/4H/128D GPT-2 (~0.8M params) and a 1L/1H/32d TransformerForwardModel (~83K params, 10.4% of main) on RHM data with v=8, s=2. The FSM predicts post_block0 -> post_block1 (single-block gap). 9 settings: L in {4, 6, 8} x m in {2, 4, 8}, all on 5M tokens.

Note: seq_len = s^L varies across L values (16, 64, 256), which changes training steps (20K, 5.5K, 2K via auto-scaling) and attention pattern complexity. Within-L comparisons (varying m only) are clean; across-L comparisons are confounded.

### Experiment 2: FM capacity sweep

Same setup on two well-learned settings (L=4/m=2, L=6/m=2), with FM capacity at 10%, 25%, 50%, and 100% of the main model. The 25% FM (198K params) exactly matches a single main model block's parameter count. Tests whether the observed rank reflects DGP complexity or FM approximation characteristics.

| Label | FM architecture | FM params | Capacity ratio |
|-------|----------------|-----------|----------------|
| 10%   | 1L/1H/32d/mlp2 | 83K       | 10.4%          |
| 25%   | 1L/4H/32d/mlp4 | 198K      | 24.9%          |
| 50%   | 2L/4H/32d/mlp4 | 397K      | 49.7%          |
| 100%  | 4L/4H/32d/mlp4 | 793K      | 99.5%          |

## Results

### DGP sweep

| Setting | L | m | seq_len | eff_rank | rank% | top1_var | cosine | val_loss |
|---------|---|---|---------|----------|-------|----------|--------|----------|
| v8_s2_L4_m2 | 4 | 2 | 16  | 98.0  | 76.5% | 0.076 | 0.992 | 0.937 |
| v8_s2_L4_m4 | 4 | 4 | 16  | 80.4  | 62.8% | 0.127 | 0.988 | 1.449 |
| v8_s2_L4_m8 | 4 | 8 | 16  | 70.1  | 54.8% | 0.271 | 0.997 | 1.836 |
| v8_s2_L6_m2 | 6 | 2 | 64  | 103.1 | 80.6% | 0.083 | 0.988 | 0.840 |
| v8_s2_L6_m4 | 6 | 4 | 64  | 98.6  | 77.0% | 0.110 | 0.981 | 1.414 |
| v8_s2_L6_m8 | 6 | 8 | 64  | 60.1  | 47.0% | 0.527 | 0.988 | 1.792 |
| v8_s2_L8_m2 | 8 | 2 | 256 | 93.4  | 73.0% | 0.180 | 0.958 | 0.824 |
| v8_s2_L8_m4 | 8 | 4 | 256 | 85.0  | 66.4% | 0.175 | 0.966 | 1.462 |
| v8_s2_L8_m8 | 8 | 8 | 256 | 67.0  | 52.3% | 0.633 | 0.990 | 1.827 |

Within each L, rank monotonically decreases with m. Higher m means higher per-level entropy, which the model learns less well (val_loss near raw entropy at m=8). The most concentrated residuals (L=6/m=8: top-1 PC = 52.7%, L=8/m=8: 63.3%) come from settings where the model has barely learned the DGP.

The cosine similarity column is instructive: the FSM achieves high directional agreement (0.958-0.997) in ALL settings, including those where the main model hasn't learned anything. The poorly-learned model's computation is simple, so there isn't much to disagree about.

### FM capacity sweep

| Setting | FM cap | FM params | eff_rank | rank% | cosine | res_norm | val_loss |
|---------|--------|-----------|----------|-------|--------|----------|----------|
| L4_m2   | 10%    | 83K       | 98.4     | 76.9% | 0.991  | 1.256    | 0.937    |
| L4_m2   | 25%    | 198K      | 99.6     | 77.8% | 1.000  | 0.088    | 0.937    |
| L4_m2   | 50%    | 397K      | 110.1    | 86.1% | 1.000  | 0.147    | 0.938    |
| L4_m2   | 100%   | 793K      | 110.6    | 86.4% | 1.000  | 0.131    | 0.938    |
| L6_m2   | 10%    | 83K       | 103.6    | 81.0% | 0.986  | 0.547    | 0.830    |
| L6_m2   | 25%    | 198K      | 113.7    | 88.8% | 0.998  | 0.193    | 0.838    |
| L6_m2   | 50%    | 397K      | 120.1    | 93.8% | 0.998  | 0.209    | 0.832    |
| L6_m2   | 100%   | 793K      | 120.4    | 94.0% | 0.998  | 0.224    | 0.834    |

Rank monotonically increases with FM capacity. It never decreases. The 25% FM (which exactly matches one block's parameter count) achieves cosine=1.000 on L=4 and reduces the residual norm 14x. But the rank barely moves (98.4 -> 99.6). At 50-100%, rank goes up further (to 110-120) as the remaining residual becomes more uniformly distributed.

The 100% FM actually has higher residual norm than the 25% FM in both settings, likely due to optimization difficulty with the overparameterized model. The rank metric is insensitive to this.

## Interpretation

### Effective rank does not measure DGP complexity here

The capacity sweep is the key result. If effective rank reflected DGP complexity, it should be invariant to FM capacity (or decrease with capacity as the FM captures more of the computation). Instead, it monotonically increases. This means rank is dominated by the FM's approximation characteristics, not the DGP's structure.

### What effective rank actually measures

The rank tracks how uniformly the FM's approximation error is distributed across dimensions. At low FM capacity, the FM makes large, structured errors concentrated in a few directions (the specific computations its limited architecture can't capture). At high FM capacity, those structured errors are resolved, leaving small, broadly-distributed errors from irreducible architectural mismatch. Removing concentrated errors and leaving diffuse ones increases rank mechanically.

### The architectural mismatch confound

The FSM in all experiments uses 1 attention head (or at most 4, in the larger configurations) to predict a block with 4 attention heads. Even at 100% parameter capacity, the FM's architecture is fundamentally different from the target block. The residual norm plateaus at ~0.09-0.22 rather than reaching zero because the FM cannot exactly replicate the block's computation.

This creates a specific pattern:
- **As FM capacity grows**: the structured, high-variance errors (from capacity shortage) are removed first, because they are the easiest to learn away. What remains is the irreducible mismatch — many small errors distributed broadly, because the architectural gap doesn't favor any particular direction.
- **Result**: norm goes down, rank goes up — the opposite of what "residual rank reflects DGP complexity" would predict.

If the FM could reach actual zero residual (e.g., using an architecture-matched FM identical to the target block), we would expect rank to decrease on the final approach, because the last errors to be resolved would be in specific, structured directions. We did not test this.

### What DID vary meaningfully with DGP complexity

**Residual norm** tracks DGP complexity (at fixed FM capacity). Well-learned settings (m=2) have lower residual norms than poorly-learned settings (m=8). This is the scalar version of "how much computation the FM couldn't predict."

**Cosine similarity** also tracks the model's learning quality, with an important inversion: poorly-learned models have HIGHER cosine (0.990-0.997) than well-learned ones (0.958-0.992). Simple computation is easy to predict directionally; there isn't much to disagree about.

### Residual norm is scalar — what would a multidimensional version look like?

Residual norm collapses 128 dimensions into one number. A richer metric would decompose the residual magnitude along DGP-relevant directions (e.g., per-hierarchical-level, per-position-in-subtree, per-rule). This is the analog of the behavioral residual analysis from the language A2A experiments, where residual norms conditioned on sentence starts, delimiters, and attention patterns revealed what the FM specifically couldn't capture. We did not implement this for the RHM.

### Experiment 3: Architecture-matched FM sweep

The capacity sweep (Experiment 2) showed rank monotonically increasing with FM capacity, but those FMs used different numbers of attention heads than the main model block (1H FM predicting a 4H block). This architectural mismatch creates structured errors concentrated in specific directions — removing those errors while leaving broadly-distributed mismatch errors mechanically increases rank.

This experiment eliminates the head-count confound. All FMs use 4 attention heads (matching the main model block), with d_head and mlp_mult scaling from 25% to 100% of the block's dimensions. At 100%, the FM is an exact architectural copy of one main model block (198K params). The main model is trained first with a fixed seed (42), then frozen — the FM trains on fixed activations, giving it the best chance to converge.

| Setting | FM cap | FM params | blk% | eff_rank | rank% | cosine | res_norm | tgt_norm | rel_res |
|---------|--------|-----------|------|----------|-------|--------|----------|----------|---------|
| L4_m2   | 25%    | 50K       | 25%  | 115.8    | 90.5% | 0.998  | 0.574    | 11.20    | 5.1%    |
| L4_m2   | 50%    | 100K      | 50%  | 120.0    | 93.7% | 1.000  | 0.209    | 11.19    | 1.9%    |
| L4_m2   | 75%    | 149K      | 75%  | 119.6    | 93.4% | 1.000  | 0.078    | 11.19    | 0.7%    |
| L4_m2   | 100%   | 198K      | 100% | 120.3    | 94.0% | 1.000  | 0.033    | 11.20    | 0.3%    |
| L6_m2   | 25%    | 50K       | 25%  | 117.5    | 91.8% | 0.998  | 0.242    | 3.63     | 6.7%    |
| L6_m2   | 50%    | 100K      | 50%  | 121.0    | 94.5% | 0.999  | 0.151    | 3.63     | 4.2%    |
| L6_m2   | 75%    | 149K      | 75%  | 121.6    | 95.0% | 0.999  | 0.125    | 3.63     | 3.5%    |
| L6_m2   | 100%   | 198K      | 100% | 122.4    | 95.6% | 1.000  | 0.107    | 3.63     | 3.0%    |

The 100% FM on L=4/m=2 achieved 0.3% relative residual — as close to zero as we can get. The 100% FM on L=6/m=2 reached 3.0%; the longer sequences (64 vs 16 tokens) make the block's function harder to replicate even with a perfect architectural match.

Rank remains in a narrow band (90–96%) across a 4× capacity range and a 17× residual norm range. The top-1 PC variance is nearly identical across all conditions (0.024–0.032). Even at near-zero residual, rank does not decrease.

For comparison, the Experiment 2 sweep (mismatched head count) had rank 77–86% at the same capacity points. Matching the architecture pushes rank up to 90%+ immediately; further capacity scaling barely moves it.

## What we learned

### Residual rank structure is a property of the main model's computation

The architecture-matched sweep (Experiment 3) is the key result. Holding the main model and DGP fixed while varying FM capacity from 25% to 100% of the block — driving residual norm from 0.57 down to 0.03 — left the rank structure essentially unchanged (90–96% throughout). The residual's shape is invariant to FM capacity; only its magnitude changes.

This means the rank reflects how the main model distributes its computation across dimensions, not something about the FM or the FM-main interaction. That's a real property of the main model, not an artifact.

### Rank does not measure DGP complexity in this setup (but we haven't cleanly tested whether it could)

The DGP sweep (Experiment 1) showed rank monotonically decreasing with m (synonymic multiplicity) at fixed L. But higher m also means the model learned less (val_loss near raw entropy at m=8). The rank variation tracks learning quality: a model that hasn't learned the DGP does simple, low-dimensional processing; a model that has learned it uses all 128 dimensions richly.

What we have NOT tested is whether rank varies across DGPs of different complexity when the main model has learned them equally well. This is hard to set up — more complex DGPs are harder to learn with a fixed model, so learning quality and DGP complexity are entangled. A clean test would require finding settings where the model reaches similar val_loss on DGPs of different inherent complexity (e.g., by scaling model size or training duration to equalize learning quality).

### Architectural mismatch dominates rank in the capacity sweep

Experiment 2's rank increase (77% → 86% → 94%) was primarily driven by head-count mismatch. The 1H FM creates structured errors concentrated in specific directions (the attention patterns a single head can't represent). Removing those errors leaves broadly-distributed mismatch noise, which mechanically increases rank. Experiment 3, with matched architecture, starts at 90%+ and barely moves.

### Residual norm tracks DGP complexity at fixed FM capacity

Within Experiment 1 (fixed FM), residual norm correlates with how well the model learned: well-learned settings (m=2) have lower residual norms. Cosine similarity shows an instructive inversion: poorly-learned models have HIGHER cosine (0.990–0.997) because their simple computation is easy to predict directionally.

Residual norm is a scalar, though — it collapses 128 dimensions into one number. A richer metric would decompose the residual along DGP-relevant directions (per-hierarchical-level, per-position-in-subtree, per-rule), analogous to the behavioral residual analysis in the language A2A experiments.

## Open questions

1. **DGP complexity at matched learning quality**: Can we find RHM settings where the model learns equally well but the DGPs differ in genuine complexity? This would test whether rank varies with DGP complexity independent of learning quality. Possible approaches: scale model size to equalize val_loss across different (L, m) settings; or find (L, m) pairs that happen to reach similar val_loss despite different DGP structure.

2. **Wider prediction gap**: With post_block0 → post_block3 (3-block gap), even a capacity-matched FM can't replicate the computation. The irreducible residual might carry genuine DGP structure that survives capacity scaling. This is the design used in the language A2A experiments.

3. **DGP-conditioned residual decomposition**: Does the residual encode information about the RHM's hierarchical structure (which level, which rule, which subtree)? This would bypass the rank metric entirely and directly test whether the FSM's errors are interpretable in terms of the DGP.

## Reproduction

```bash
cd experiments/

# DGP sweep (9 settings, ~30 min each on T4, parallel)
modal run --detach rhm/rhm_residual_rank.py::rhm_residual_rank_sweep

# FM capacity sweep (8 jobs: 2 settings x 4 capacities, mismatched architecture)
modal run --detach rhm/rhm_residual_rank.py::capacity_sweep

# Architecture-matched FM sweep (2 main models + 8 FM jobs)
modal run --detach rhm/rhm_residual_rank.py::architecture_matched_sweep

# Single setting (useful for testing)
modal run rhm/rhm_residual_rank.py::train_and_analyze \
  --depth 4 --m 2 --n-tokens 5000000
```

## Files

| File | Purpose |
|------|---------|
| `rhm_residual_rank.py` | Experiment script: `train_and_analyze`, `rhm_residual_rank_sweep`, `capacity_sweep`, `architecture_matched_sweep` |
| `model.py` | GPT-2 with `return_intermediates` support (backward-compatible addition) |
| `RESIDUAL_RANK_README.md` | This file |
