# RHM Regime Transition Experiment (2026-06-21)

**Code**: `rhm_regime_transition.py`
**Prior experiments**: [Residual rank](RESIDUAL_RANK_README.md), [A2A forward model](../a2a_forward/README.md)

## Goal

Test whether the FM residual transitions from diffuse (L-regime) to rule-conditioned (m-regime) as the main model learns the DGP. The hypothesis: on well-learned data, the FM's errors should concentrate along rule-discrimination directions (which of m alternatives was used at each hierarchical level), analogous to how MNIST residuals concentrate along digit-discriminative directions. On poorly-learned data, the FM's errors should stay diffuse (like language).

This was motivated by the observation that meta-learning behavior (gate selectivity, OOD adaptation) emerged on MNIST (low-rank, structured residual) but not on language (high-rank, diffuse residual). The RHM gives ground-truth rule/feature identity at each hierarchical level, enabling a direct test of whether the residual becomes structured as the model learns.

## Design

Three settings at fixed L=4 (seq_len=16), varying m:

| Setting | m | Role | Expected |
|---------|---|------|----------|
| v8_s2_L4_m2 | 2 | Easy (MNIST-like) | Full transition: eta² rises |
| v8_s2_L4_m4 | 4 | Medium | Partial transition |
| v8_s2_L4_m8 | 8 | Hard (language-like) | No transition: eta² stays low |

Model: 4L/4H/128D GPT (~0.8M params). FM: TransformerForwardModel. Identical seed (42), lr (3e-4), 5M tokens, 20K steps.

At 11 log-spaced checkpoints (including step 0), the main model is frozen and a fresh FM is trained on its activations (5K steps, seed 137). The FM residual is then analyzed for:
- Standard metrics (cosine, residual norm, effective rank)
- Hierarchy-conditioned eta²: fraction of residual variance explained by rule identity or feature identity at each of 4 hierarchical levels, using ground-truth ancestry traces from the RHM generation process
- Both aggregate (all positions) and last-position (maximum causal context) variants

Two runs: (1) 1-block gap with 1L/1H/32d FM, (2) 3-block gap with 2L/1H/32d FM.

## Results

### Val loss trajectories (as expected)

| Setting | Step 0 | Step 20K |
|---------|--------|----------|
| m=2 | 2.15 | 0.94 |
| m=4 | 2.09 | 1.46 |
| m=8 | 2.11 | 1.84 |

### FM cosine: too high for a meaningful residual

| Domain | Gap | FM | Cosine | 1−cos |
|--------|-----|----|--------|-------|
| RHM L=4/m=2 | 1-block | 1L/1H/32d | 0.994 | 0.6% |
| RHM L=4/m=2 | 3-block | 2L/1H/32d | 0.994 | 0.6% |
| RHM L=4/m=8 | 3-block | 2L/1H/32d | 0.999 | 0.1% |
| Language 29M | 3-block | 2L/1H/64d | 0.935 | 6.5% |
| MNIST ViT | 3-block | 1L/1H/32d | 0.903 | 9.7% |

The FM captures 99.4–99.9% of the computation directionally. The structured-residual experiments from the A2A work (self-knowledge probes, gate selectivity, ratchet) live in the 3–10% cosine gap. RHM at L=4 is an order of magnitude below that.

Widening the gap from 1-block to 3-block did not change the cosine. The computation at seq_len=16 is simple enough that a 2L FM predicts 3 blocks as easily as 1.

### The residual is likely architectural mismatch, not computational gap

The [architecture-matched FM sweep](RESIDUAL_RANK_README.md) showed that eliminating the head-count mismatch (1H FM vs 4H main model) and using a 100%-capacity FM drives the relative residual to 0.3% on L=4/m=2. The ~0.6% cosine gap we observe here is close to that floor, suggesting the residual is dominated by the 1H-vs-4H attention mismatch rather than computation the FM genuinely couldn't capture.

### Hierarchy eta²: low, no transition

Rule eta² (last position) stays below 2% across all settings, steps, and levels. No rising trajectory. Feature eta² shows a gradient across settings (m=2: 5–9%, m=8: <2% at final step) but no clear within-setting transition over training.

Feature eta² (last position) at step 20K, 3-block gap:

| Level | m=2 | m=4 | m=8 |
|-------|-----|-----|-----|
| L0 (root) | 0.055 | 0.002 | 0.001 |
| L1 | 0.075 | 0.005 | 0.001 |
| L2 | 0.074 | 0.016 | 0.002 |
| L3 (leaf-adjacent) | 0.081 | 0.029 | 0.015 |

The across-setting gradient is real (well-learned models have ~10x higher feature eta²) but the absolute values are small, and they don't clearly rise during training within any setting.

### Effective rank

| Setting | Step 0 | Step 20K (1-block) | Step 20K (3-block) |
|---------|--------|--------------------|--------------------|
| m=2 | ~118 | 100 (78%) | 96 (75%) |
| m=4 | ~119 | 81 (63%) | 83 (65%) |
| m=8 | ~119 | 61 (48%) | 75 (59%) |

Rank drops as the model learns (the computation becomes more structured), and is lower for higher m. Consistent with the prior residual rank experiment. The 3-block gap gives slightly different values but the same qualitative pattern.

## What we learned

### The L→m transition prediction was not testable at this scale

The hypothesis — that the FM residual transitions from diffuse to rule-conditioned as the model learns — could not be tested because the FM captures 99%+ of the computation at seq_len=16. There is no meaningful residual for rule-conditioning to live in. The prediction may still be correct, but this experimental setup cannot distinguish it from the null.

### The cosine gap determines whether the residual carries structure

Across domains, the structured-residual phenomena (self-knowledge, gate selectivity, meta-learning) emerge when the FM cosine is 0.90–0.97 (3–10% gap). At 0.99+ (RHM L=4), the residual is architectural noise. The hierarchy eta² measurement correctly reports "no structure" because there is none to find.

This suggests a necessary condition for the A2A meta-learning machinery: the FM must genuinely struggle with the computation. On MNIST (50 patches, 128-dim, 4 heads), the FM struggles because the attention patterns are complex enough that a 1H FM can't capture them. On language (128+ tokens), even more so. On RHM at 16 tokens, there isn't enough computational complexity for the FM to fail at.

### The m-gradient in feature eta² is real but uninformative

The well-learned model (m=2) has ~10x higher feature eta² than the poorly-learned model (m=8). This is consistent with the hypothesis in direction but the effect is too small (max 9%) to draw conclusions from. It could reflect genuine rule-dependent structure in the residual, or it could be a second-order effect of the embedding structure interacting with the architectural mismatch.

## What would need to change

To actually test the L→m transition, the FM cosine needs to be in the 0.90–0.97 range. Possible approaches:

1. **Longer sequences** (L=6/seq_len=64 or L=8/seq_len=256): more tokens → more complex attention patterns → harder for the FM to predict. But this changes multiple variables simultaneously and the model may not learn well at higher m.

2. **Much smaller FM**: force a genuine capacity gap even on simple computation. Risk: the FM might be too small to capture anything meaningful, making the residual "everything" rather than "what's hard."

3. **Different architecture**: a per-position MLP forward model (no attention) would miss all cross-position computation. The original A2A experiments showed this creates a residual dominated by "attention exists" rather than computational novelty — a structural blindness confound.

The deeper question is whether the RHM at any tractable scale produces the kind of computation that creates a useful capacity gap for the FM. The answer may be no: the RHM's hierarchical structure is regular enough that a small transformer can always capture it, unlike the irregular, input-dependent computation in MNIST classification or language modeling.

## Reproduction

```bash
cd experiments/

# 1-block gap (Run 1)
modal run --detach rhm/rhm_regime_transition.py::rhm_regime_transition

# 3-block gap (Run 2)
modal run --detach rhm/rhm_regime_transition.py::rhm_regime_transition \
  --predict-from post_block0 --predict-to post_block3 --fwd-n-layer 2
```

## Modal volume

Results saved to `rhm-scaling-data` volume at `/data/rhm_regime_transition/results.json` and per-setting checkpoints at `/data/v8_s2_L4_m{m}/regime_transition/`.

---

## Cosine sweep: finding the sweet spot (2026-06-21)

**Code**: `rhm_cosine_sweep.py`

The L=4 experiments above failed because the FM captured 99%+ of computation at seq_len=16 — there was no meaningful residual to structure. To find a regime where the FM genuinely struggles (cosine 0.90–0.97, like MNIST and language), we swept L∈{5,6} (seq_len=32,64), m∈{2,4,8}, and two FM sizes: "matched" (2L/1H/16d/mlp1, 84K params, 14% of gap ≈ MNIST ratio) and "current" (2L/1H/32d/mlp2, 166K params, 28% of gap ≈ language ratio). All with 3-block gap (post_block0 → post_block3) on 4L/4H/128D models.

### FM capacity analysis

The FM capacity ratio (FM params / gap params) matters for setting the cosine floor:

| Domain | FM | FM/gap ratio | Cosine |
|---|---|---|---|
| RHM L=4 (prior) | 2L/1H/32d/mlp2 | 28% | 0.994 |
| Language 29M | 2L/1H/64d/mlp2 | 28% | 0.935 |
| MNIST | 1L/1H/32d/mlp2 | 14% | 0.903 |

The current RHM FM has the same capacity ratio as language but 0.994 cosine — confirming the problem was computational simplicity at seq_len=16, not over-allocation.

### Results

| Setting | seq | m | FM | cos | 1-cos | val_loss | rank% | top1% |
|---|---|---|---|---|---|---|---|---|
| L5/m2 | 32 | 2 | matched | 0.972 | 2.8% | 0.86 | 74% | 17% |
| L5/m2 | 32 | 2 | current | 0.981 | 1.9% | 0.86 | 76% | 16% |
| L5/m4 | 32 | 4 | matched | 0.983 | 1.7% | 1.42 | 66% | 46% |
| L5/m8 | 32 | 8 | matched | 0.998 | 0.2% | 1.78 | 66% | 42% |
| **L6/m2** | **64** | **2** | **matched** | **0.963** | **3.7%** | **0.85** | **77%** | **15%** |
| L6/m2 | 64 | 2 | current | 0.975 | 2.5% | 0.85 | 78% | 15% |
| L6/m4 | 64 | 4 | matched | 0.987 | 1.3% | 1.43 | 72% | 36% |
| L6/m8 | 64 | 8 | matched | 0.999 | 0.1% | 1.80 | 68% | 46% |

### Key findings

1. **Higher m makes the FM's job EASIER, not harder.** At m=8, val_loss ≈ 1.8 (near chance = ln(8) = 2.08). The model has barely learned anything beyond bigram statistics, so the computation is trivially predictable. The FM measures the complexity of what the model has learned, not the complexity of the DGP.

2. **L=6/m=2 with matched FM is the best candidate** (cos=0.963, 1-cos=3.7%), squarely in the MNIST–language range. The model learns well (val_loss 0.85) and the longer sequences create attention patterns the FM can't fully capture.

3. **The residual at m=2 is diffuse** (top1=15%, rank 77%) — language-like, not MNIST-like. This is the L-regime: even well-learned, the m=2 binary discrimination is too simple to concentrate the residual.

4. **FM capacity matters more at low m.** At L6/m=2, matched vs current shifts cosine from 0.963 → 0.975. At m=8, capacity is irrelevant (0.999 regardless).

### Implication: the L→m transition requires a model that can learn moderate m

The settings that produce a meaningful cosine gap (low m, high L) are exactly the settings where the m-regime signature is weakest (m=2 = binary discrimination, not rich enough for concentrated structure). The settings with rich m-regime structure (high m) are the ones where the model barely learns. Resolving this tension requires scaling up the main model so it can learn higher-m DGPs.

### Reproduction

```bash
modal run --detach rhm/rhm_cosine_sweep.py::rhm_cosine_sweep
```

---

## Regime trajectory: the L→m transition at scale (2026-06-21)

**Code**: `rhm_regime_trajectory.py`

Two experiments tracking the FM residual structure over the full training trajectory, with hierarchy-conditioned eta² at ~11 checkpoints:

1. **m=3, current model** (4L/4H/128D, 0.8M): intermediate m, 5M tokens
2. **m=4, scaled model** (6L/6H/192D, ~2.7M): bigger model to learn harder DGP, 20M tokens

Both use L=6 (seq_len=64), 3-block gap, and matched FM (~14% of gap capacity).

### Experiment 1: m=3, current model — too easy

| Step | Val Loss | Cosine | 1-cos | fL5* | fL4* | fL3* |
|------|----------|--------|-------|------|------|------|
| 0 | 2.097 | 0.994 | 0.6% | .133 | .066 | .011 |
| 600 | 1.329 | 0.997 | 0.3% | .026 | .012 | .002 |
| 2000 | 1.199 | 0.994 | 0.6% | .062 | .029 | .010 |
| 5400 | 1.175 | 0.990 | 1.0% | .065 | .031 | .010 |

Cosine gap maxes at 1%. The 0.8M model learns m=3 well enough that the FM still captures 99%+ of computation. No meaningful eta² trajectory.

### Experiment 2: m=4, scaled model — the L→m transition

| Step | Val Loss | Cosine | 1-cos | Rank% | Top1% | fL4* | fL3* | fL2* | rL3* |
|------|----------|--------|-------|-------|-------|------|------|------|------|
| 0 | 2.141 | 0.997 | 0.3% | 94.1 | 1.7 | .024 | .005 | .002 | .005 |
| 200 | 1.583 | 1.000 | 0.02% | 89.4 | 12.2 | .006 | .002 | .001 | .001 |
| 2000 | 1.463 | 0.994 | 0.6% | 72.2 | 32.3 | .014 | .003 | .001 | .002 |
| 4000 | 1.412 | 0.981 | 1.9% | 71.4 | **33.3** | .018 | .007 | .003 | .002 |
| 7000 | 1.384 | 0.969 | 3.1% | 69.9 | 27.1 | .030 | .014 | .005 | .008 |
| 10000 | 1.376 | 0.958 | 4.2% | 69.9 | 25.7 | .049 | .015 | .005 | .009 |
| 14000 | 1.371 | 0.952 | 4.8% | 69.6 | 22.7 | .059 | .022 | .008 | .010 |
| 17000 | 1.366 | 0.943 | 5.7% | 71.3 | 14.3 | .068 | .034 | .010 | .024 |
| 20000 | 1.363 | 0.921 | 7.9% | 70.7 | 11.9 | **.074** | **.050** | **.015** | **.023** |

Three signatures of the L→m transition:

**1. Feature eta² rises monotonically over training.** fL4* (leaf-adjacent level) goes 0.006 → 0.074 (12×). fL3* goes 0.002 → 0.050 (25×). The FM's errors become increasingly conditioned on hierarchical feature identity. The signal propagates UP the hierarchy: fL4* rises first (local composition learned first), then fL3* catches up (abstract composition learned later).

**2. Top1 PC has a non-monotonic profile.** Rises to 33% at step 4000 (one dominant FM error mode), then DROPS to 12% by step 20000 (multiple specific error modes). Top1% going down while eta² goes up is the key regime signature: the residual diversifies from "one big thing the FM misses" (L-regime: generic capacity gap) into "several specific things" (m-regime: multiple rule/feature discriminations).

**3. The cosine trajectory enters the sweet spot.** By step 7000, cos=0.969 (language range). By step 20000, cos=0.921 (MNIST range). The eta² rise tracks the cosine drop — the transition happens when the FM has a meaningful gap (1-cos > 2%).

### Interpretation

The 2.7M model learns enough of the m=4 hierarchical structure that its computation becomes genuinely complex — complex enough that a 14%-capacity FM can't fully approximate it. As training progresses, the FM's errors shift from "I couldn't follow the computation through enough layers" (diffuse, L-regime) to "I couldn't tell which of 4 alternatives was used at levels 2–4" (structured, m-regime). This is the computational analog of an expert developing structured surprise: most inputs are unsurprising, and deviations are surprising in specific, nameable ways.

The absolute eta² values are modest (max 7.4% vs MNIST's 23% for digit identity) because the model hasn't fully learned m=4 yet (val_loss 1.36 is still well above the m=2 floor of 0.85). With more capacity or training, the transition would likely deepen.

The m=3 experiment confirms that scaling up the model was the right move: the 0.8M model can learn m=3 easily enough that the FM never struggles, so no transition occurs. The transition requires a model that has learned COMPLEX computation — simple enough for the FM to be mostly right, but complex enough that its errors are structured.

### Reproduction

```bash
cd experiments/

# m=3, current model
modal run --detach rhm/rhm_regime_trajectory.py::run_m3_current

# m=4, scaled model
modal run --detach rhm/rhm_regime_trajectory.py::run_m4_scaled
```

Results saved to `rhm-scaling-data` volume at `/data/rhm_regime_trajectory/v8_s2_L6_m3_4L4H128D.json` and `/data/rhm_regime_trajectory/v8_s2_L6_m4_6L6H192D.json`.
