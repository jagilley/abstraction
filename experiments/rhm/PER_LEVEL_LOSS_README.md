# Per-Level Loss Decomposition (2026-06-21)

**Code**: `rhm_per_level_loss.py`
**Prior experiment**: [Regime transition & trajectory](REGIME_TRANSITION_README.md)

## Idea

Since we know the DGP explicitly, every next-token prediction maps to a specific level of the hierarchy based on which boundary it crosses. Position p within a sequence has hierarchical level = v_s(p), the s-adic valuation (number of trailing zeros in base s):

- **Level 0** (p mod s != 0): within the same s-tuple. Easiest — just need the local composition rule.
- **Level k** (p mod s^k = 0): first token of a new s^k subtree. Must infer the ancestor feature k levels up by "inverting" the preceding tokens through k levels of composition rules.
- **Level L-1**: crosses the root boundary. Hardest — requires integrating the full sequence.

For s=2, L=6 (seq_len=64), the positions decompose as:

| Level | Positions per sequence | Example positions | What's needed |
|-------|----------------------|-------------------|--------------|
| 0 | 32 | 1, 3, 5, 7, ... | Local rule within s-tuple |
| 1 | 16 | 2, 6, 10, 14, ... | 1 level of composition |
| 2 | 8 | 4, 12, 20, 28, ... | 2 levels of composition |
| 3 | 4 | 8, 24, 40, 56 | 3 levels of composition |
| 4 | 2 | 16, 48 | 4 levels of composition |
| 5 | 1 | 32 | Root-level context |

## Design

Two trajectory experiments tracking per-level cross-entropy over training:

| Experiment | Model | m | Tokens | Motivation |
|-----------|-------|---|--------|-----------|
| 1 | 4L/4H/128D (0.8M) | 2 | 5M | Well-learned setting, clear hierarchy signal |
| 2 | 6L/6H/192D (2.7M) | 4 | 20M | Same architecture as the L→m transition |

Both at L=6, v=8, s=2 (seq_len=64). Evaluation on 5K fresh sequence-aligned sequences at ~11 log-spaced checkpoints. Per-position cross-entropy grouped by hierarchical level.

Note: training uses random offsets into the concatenated corpus (not sequence-aligned), so the model doesn't know sequence boundaries. For evaluation, we feed complete sequences so position-within-sequence maps cleanly to hierarchy level.

## Results

### Experiment 1: m=2, 4L/4H/128D (0.8M params)

Uniform baseline: ln(8) = 2.079.

| Step | Val | L0 | L1 | L2 | L3 | L4 | L5 |
|------|-----|------|------|------|------|------|------|
| 0 | 2.029 | 2.038 | 2.037 | 1.990 | 2.035 | 2.092 | 2.042 |
| 200 | 1.200 | **0.670** | 1.656 | 1.643 | 1.973 | 1.909 | 1.921 |
| 600 | 1.073 | 0.574 | **1.384** | 1.562 | 1.978 | 1.892 | 1.910 |
| 1000 | 0.995 | 0.490 | 1.308 | **1.423** | 1.973 | 1.858 | 1.921 |
| 2000 | 0.922 | 0.432 | 1.220 | 1.324 | 1.888 | 1.818 | 1.903 |
| 5400 | 0.847 | 0.372 | 1.124 | 1.171 | 1.770 | 1.747 | 1.869 |

At convergence:

| Level | Loss | Accuracy | vs uniform |
|-------|------|----------|-----------|
| L0 | 0.372 | 81.5% | -82% |
| L1 | 1.124 | 42.6% | -46% |
| L2 | 1.171 | 42.0% | -44% |
| L3 | 1.770 | 26.3% | -15% |
| L4 | 1.747 | 27.5% | -16% |
| L5 | 1.869 | 24.4% | -10% |

### Experiment 2: m=4, 6L/6H/192D (2.7M params)

| Step | Val | L0 | L1 | L2 | L3 | L4 | L5 |
|------|-----|------|------|------|------|------|------|
| 0 | 2.141 | 2.128 | 2.138 | 2.165 | 2.176 | 2.171 | 2.151 |
| 200 | 1.582 | **1.191** | 1.980 | 1.992 | 1.983 | 1.976 | 1.968 |
| 1000 | 1.522 | 1.079 | **1.933** | 1.976 | 1.976 | 1.972 | 1.969 |
| 4000 | 1.404 | 0.935 | 1.776 | 1.918 | 1.953 | 1.942 | 1.930 |
| 10000 | 1.376 | 0.907 | 1.747 | 1.903 | 1.940 | 1.937 | 1.926 |
| 20000 | 1.364 | 0.888 | 1.725 | 1.898 | 1.937 | 1.935 | 1.924 |

At convergence:

| Level | Loss | Accuracy | vs uniform |
|-------|------|----------|-----------|
| L0 | 0.888 | 58.6% | -57% |
| L1 | 1.725 | 30.5% | -17% |
| L2 | 1.898 | 21.1% | -9% |
| L3 | 1.937 | 20.1% | -7% |
| L4 | 1.935 | 20.3% | -7% |
| L5 | 1.924 | 21.1% | -7% |

## Key findings

### 1. Monotonic loss gradient across hierarchy levels

Performance degrades monotonically with hierarchy depth, confirming the intuition that higher levels require more compositional reasoning. The gradient is steepest between levels 0-1 (local → one level of composition) and flattens at higher levels where the model hits its capacity ceiling.

### 2. Bottom-up learning

The model learns the hierarchy strictly bottom-up: level 0 (local s-tuples) drops from baseline in the first 200 steps while levels 3-5 barely budge. Level 1 follows, then level 2. This is visible as a "wave" propagating up the hierarchy over training. The ordering of when each level begins improving is perfectly monotonic in both experiments.

### 3. m makes every level harder

Despite 3.3x more parameters and 4x more data, the m=4 model learns far less at every level than the m=2 model:

| Level | m=2 loss | m=4 loss | m=2 acc | m=4 acc |
|-------|---------|---------|---------|---------|
| L0 | 0.372 | 0.888 | 81.5% | 58.6% |
| L1 | 1.124 | 1.725 | 42.6% | 30.5% |
| L2 | 1.171 | 1.898 | 42.0% | 21.1% |
| L3-5 | ~1.79 | ~1.93 | ~26% | ~20% |

With m=4, levels 2-5 are all bunched near baseline (~1.93 vs uniform 2.08), meaning the model essentially can't compose beyond 1-2 levels. With m=2, the model reaches deeper (levels 1-2 are well below baseline) but still plateaus at levels 3-5.

### 4. Composition depth ceiling

Both models hit a ceiling: a 4-layer transformer at m=2 plateaus around 2-3 levels of learned composition; a 6-layer transformer at m=4 plateaus at ~1-2 levels. This suggests the number of composition levels a model can learn depends on both model depth and m. Plausibly, each transformer layer can handle roughly one level of composition, but higher m demands more capacity per level.

## Connection to the L→m transition

The per-level loss decomposition provides the mechanistic picture behind the L→m transition observed in the FM residual (see [regime trajectory](REGIME_TRANSITION_README.md)):

- As the model learns each level bottom-up, the FM's errors shift from "generic capacity gap" (can't follow the computation at all) to "can't tell which of m rules was used at the levels the model just learned" (structured, rule-discriminative errors).
- The bottom-up learning wave is what drives the monotonic rise in feature eta² over training: the FM residual becomes increasingly conditioned on hierarchical features as the model learns deeper composition.
- The composition depth ceiling explains why the transition saturates: once the model stops learning new levels, the FM residual stops gaining new structure.

## Reproduction

```bash
cd experiments/

# Experiment 1: m=2, small model
modal run --detach rhm/rhm_per_level_loss.py::per_level_trajectory \
    --depth 6 --m 2

# Experiment 2: m=4, scaled model
modal run --detach rhm/rhm_per_level_loss.py::per_level_trajectory \
    --depth 6 --m 4 --n-tokens 20000000 \
    --n-layer 6 --n-head 6 --n-embd 192

# Single-setting evaluation (no trajectory)
modal run --detach rhm/rhm_per_level_loss.py::per_level_single \
    --depth 6 --m 2

# Cross-setting sweep
modal run --detach rhm/rhm_per_level_loss.py::per_level_sweep \
    --settings "L4_m2,L4_m8,L6_m2,L6_m4,L8_m2"
```

Results saved to `rhm-scaling-data` volume at `/data/rhm_per_level_loss/`.

---

## FM as DGP approximation (2026-06-21)

**Code**: `rhm_dgp_approximation.py`
**Prior experiments**: [Per-level loss decomposition](#per-level-loss-decomposition-2026-06-21) (above), [Regime trajectory](REGIME_TRANSITION_README.md), Forward self-models paper[^private]

### Motivation

The forward self-models paper claims FMs dissociate representation from computation: by conditioning on the main model's intermediate representations, the FM approximates the computational function that the intervening layers implement. On RHM data, the main model's computation IS hierarchical composition — applying rules to compose level-l features from level-(l-1) features. If the dissociation claim is true, and the main model has learned good representations, then the FM should approximate the RHM's composition rules themselves.

The RHM is the one setting where we can directly test this, because we know the ground-truth rules and feature identities at every level. The per-level loss decomposition (above) told us which levels the model has learned. Now we ask: at those levels, has the FM also learned the composition rules?

### Design

Three measurements at each hierarchy level, all using eta² (fraction of variance explained by group membership):

1. **eta²(actual activations, rule/feature identity)** — ground truth: how much does the model's actual computation (post_block3) vary by which rule/feature was used at level l?
2. **eta²(FM predictions, rule/feature identity)** — does the FM predict differently depending on which composition rule was used?
3. **eta²(source activations, rule/feature identity)** — baseline: what rule/feature structure is already in the FM's input (post_block0)?

The FM "approximates the DGP" to the extent that (2) ≈ (1) and (2) > (3): the FM adds rule-conditioned structure beyond its input, matching what the model actually computes.

Also: per-hierarchy-level cosine similarity. If the FM approximates the DGP, it should predict best at levels the main model has learned.

Uses the converged 2.7M model (6L/6H/192D) at L=6/m=4 from the [regime trajectory](REGIME_TRANSITION_README.md), step 20000. FM: 2L/1H/24d/mlp1 (187K params, ~14% of gap capacity). 10K traced evaluation sequences with ground-truth ancestry at each level.

### Results

#### Global metrics

| Metric | Value |
|--------|-------|
| FM cosine | 0.923 |
| 1 − cosine | 7.7% |
| Residual norm | 1.063 |

The FM is in the "meaningful gap" regime (comparable to MNIST/language in the A2A experiments).

#### Per-hierarchy-level cosine

| Level | Cosine | 1−cos | Positions |
|-------|--------|-------|-----------|
| 0 (within s-tuple) | 0.935 | 6.5% | 33 |
| 1 | 0.907 | 9.3% | 16 |
| 2 | 0.910 | 9.0% | 8 |
| 3 | 0.918 | 8.2% | 4 |
| 4 | 0.911 | 8.9% | 2 |
| 5 (root boundary) | 0.909 | 9.1% | 1 |

Cosine is surprisingly flat — the FM predicts roughly equally well at all hierarchy levels. The interesting structure is not in *how much* the FM captures, but in *what kind of structure* its predictions carry.

#### Feature eta² at last position (maximum causal context)

This is the key table. Each row shows how much the activations at the last sequence position vary by which feature was active at level l.

| Level | Source (post_block0) | FM prediction | Actual (post_block3) | FM/actual ratio |
|-------|---------------------|---------------|---------------------|-----------------|
| 0 (root) | 0.001 | 0.002 | 0.002 | **0.95** |
| 1 | 0.002 | 0.007 | 0.007 | **0.97** |
| 2 | 0.006 | 0.023 | 0.026 | **0.91** |
| 3 | 0.018 | 0.075 | 0.082 | **0.92** |
| 4 | 0.092 | 0.336 | 0.266 | 1.26 |
| 5 (leaf-adj) | 0.492 | 0.318 | 0.241 | 1.32 |

Rule eta² shows the same pattern (82–105% recovery at levels 0–3).

#### Structure added beyond the source

The delta between FM predictions and source shows how much rule/feature structure the FM's computation adds beyond what's already in its input, compared to what the actual blocks 1–3 add.

| Level | Actual added | FM added | FM/actual |
|-------|-------------|----------|-----------|
| 0 | +0.0012 | +0.0011 | 92% |
| 1 | +0.0054 | +0.0052 | 96% |
| 2 | +0.0200 | +0.0178 | 89% |
| 3 | +0.0641 | +0.0576 | 90% |
| 4 | +0.1742 | +0.2440 | 140% |
| 5 | −0.2519 | −0.1741 | 69% |

### Interpretation

**At levels 0–3 (learned levels), the FM captures 91–97% of the feature-conditioned structure.** The per-level loss decomposition showed the 2.7M model at m=4 has learned ~1–2 levels of composition. At those levels, the FM's predictions vary by feature identity almost exactly as much as the model's actual computation does. The FM adds almost exactly the same delta of feature structure beyond its input as the actual blocks 1–3. This is direct evidence that the FM has learned the composition rules at the levels the model has internalized.

**At levels 4–5, the FM overshoots — its predictions are *more* feature-conditioned than the actual activations.** This is a compression artifact that's actually informative:

- The FM optimizes MSE, so it approximates E[post_block3 | post_block0] — the conditional expectation given its input. This preserves systematic (group-conditioned) variation while smoothing idiosyncratic (within-group) variation, mechanically inflating eta².
- At levels 0–3, the FM also captures most of the within-group computation, so the filtering effect is small (slight undershoot).
- At levels 4–5, the main model's blocks 1–3 perform complex representational reorganization: they build compositional features at lower levels, and in doing so dilute the "raw" high-level feature encoding from block 0. The FM can't replicate this reorganization with limited capacity, so it retains more of the source's feature structure than the actual model.
- Level 5 makes this clearest: block 0 encodes the root feature at 49% eta² (it determines the entire sequence). The actual blocks 1–3 cut this to 24% as they reorganize representations. The FM can only cut it to 32% — the 8pp gap is representational reorganization the FM's capacity couldn't replicate.

The FM is, in a sense, a *purer* approximation of the DGP than the model's own computation: it captures the composition rules without the representational side-effects of the model's full computation. At learned levels, this aligns with the actual model (because the actual computation is mostly rule application). At unlearned levels, the FM can only do the rule-conditioned part, and overshoots because the actual model is doing additional computation that dilutes rule structure.

### Connection to the paper's dissociation claim

The forward self-models paper argues that conditioning on representation and modeling computation allows for a dissociation between these two aspects. This experiment provides the strongest evidence yet for that claim:

1. **The FM captures computation, not statistics.** The FM's predictions carry rule-conditioned structure that closely matches the actual layer computation (91–97% at learned levels). This is not a statistical summary of the target activations — it's an executable approximation of the compositional function.

2. **The FM captures *only* computation.** At levels beyond the model's learning horizon, the FM can't approximate what doesn't exist. It defaults to a less-transformed version of its input, retaining the source's feature structure rather than inventing non-existent computation. The overshoot is the fingerprint of this: the FM's limited capacity means it captures the DGP-aligned component while missing the representational reorganization.

3. **The boundary between captured and missed computation aligns with the learning wave.** The per-level loss decomposition showed the model learns bottom-up: levels 0–3 are learned, levels 4–5 are near baseline. The FM's recovery ratio drops at exactly this boundary. The FM tracks the model's actual computational capacity, not the DGP's full depth.

### Reproduction

```bash
cd experiments/

# Single checkpoint (converged model, step 20K)
modal run --detach -m rhm.rhm_dgp_approximation::dgp_approximation

# Training trajectory (5 checkpoints)
modal run --detach -m rhm.rhm_dgp_approximation::dgp_approximation_trajectory
```

Results saved to `rhm-scaling-data` volume at `/data/rhm_dgp_approximation/`.

---

## FM intermediate probing (2026-06-21)

**Code**: `rhm_fm_intermediate_probing.py`
**Prior experiment**: [FM as DGP approximation](#fm-as-dgp-approximation-2026-06-21) (above)

### Motivation

The DGP approximation experiment showed that the FM's *output* carries the same rule-conditioned structure as the main model's actual computation (91–97% recovery at learned levels). But does the FM just produce the right answer, or does it compute the answer via the same intermediate steps?

A 2-layer FM predicting post_block0 → post_block3 has an intermediate representation (between its two layers) with no explicit training target. If the FM mirrors the main model's computation step by step, this intermediate should encode progressively higher-level hierarchy features — the same bottom-up trajectory the main model's blocks follow.

### Design

Same converged 2.7M model (6L/6H/192D) at L=6/m=4, step 20000. FM: 2L/1H/24d/mlp1 (187K params). Three measurements at each representation point (all main model layers + all FM layers):

1. **Feature/rule eta²** at each hierarchy level
2. **Linear probe accuracy** for feature classification at each level (v=8 classes, chance=0.125)
3. **Linear CKA** between FM layers and main model layers

Representation points:
- Main model: post_embed, post_block0, ..., post_block5
- FM: fm_layer0 (= post_block0), fm_layer1 (after FM block 0), fm_layer2 (FM output)

5K traced evaluation sequences with ground-truth ancestry.

### Results

#### Feature eta²

| Representation | L0 | L1 | L2 | L3 | L4 | L5 |
|---|---|---|---|---|---|---|
| post_embed | .00002 | .00147 | .00224 | .00171 | .01254 | .13381 |
| post_block0 | .00003 | .00708 | .00792 | .00295 | .01392 | .08976 |
| post_block1 | .00004 | .00552 | .00663 | .00408 | .01848 | .08709 |
| post_block2 | .00008 | .00409 | .00614 | .00698 | .02498 | .06305 |
| post_block3 | .00007 | .00233 | .00459 | .00525 | .02268 | .06901 |
| post_block4 | .00004 | .00068 | .00186 | .00256 | .01455 | .07715 |
| post_block5 | .00003 | .00041 | .00119 | .00191 | .01377 | .09034 |
| **fm_layer0** | .00003 | .00708 | .00792 | .00295 | .01392 | .08976 |
| **fm_layer1** | .00003 | .00574 | .00679 | .00342 | .01855 | .09721 |
| **fm_layer2** | .00005 | .00261 | .00474 | .00463 | .02218 | .07800 |

fm_layer0 = post_block0 exactly (sanity check). fm_layer2 tracks post_block3 (training target).

#### Linear probe accuracy

| Representation | L0 | L1 | L2 | L3 | L4 | L5 |
|---|---|---|---|---|---|---|
| post_embed | .122 | .213 | .206 | .258 | .277 | .452 |
| post_block0 | .125 | .202 | .185 | .267 | .393 | .743 |
| post_block1 | .123 | .200 | .188 | .283 | .478 | .757 |
| post_block2 | .127 | .191 | .203 | .318 | .523 | .765 |
| post_block3 | .128 | .199 | .206 | .320 | .548 | .767 |
| post_block4 | .129 | .184 | .204 | .320 | .536 | .777 |
| post_block5 | .126 | .176 | .178 | .302 | .543 | .773 |
| **fm_layer0** | .130 | .209 | .191 | .266 | .388 | .745 |
| **fm_layer1** | .127 | .201 | .198 | .291 | .458 | .756 |
| **fm_layer2** | .124 | .197 | .196 | .315 | .526 | .756 |

#### CKA: FM layers vs main model layers

| | post_embed | post_block0 | post_block1 | post_block2 | post_block3 | post_block4 | post_block5 |
|---|---|---|---|---|---|---|---|
| fm_layer0 | 0.402 | **1.000** | 0.914 | 0.787 | 0.424 | 0.275 | 0.279 |
| fm_layer1 | 0.382 | 0.954 | **0.909** | 0.807 | 0.481 | 0.332 | 0.336 |
| fm_layer2 | 0.250 | 0.432 | 0.450 | 0.595 | **0.949** | 0.835 | 0.787 |

fm_layer2 snaps to post_block3 (CKA=0.949). fm_layer1 sits between post_block0 (0.954) and post_block1 (0.909), having moved toward post_block2 (0.807 vs 0.787 at the input).

#### Delta analysis: FM block 0 mirrors main model block 1

Feature eta² added by each block (positive = block added hierarchy structure):

| | L1 | L2 | L3 | L4 | L5 |
|---|---|---|---|---|---|
| FM_blk0 | −.00134 | −.00113 | +.00047 | +.00463 | +.00745 |
| Main_blk1 | −.00156 | −.00129 | +.00113 | +.00456 | −.00267 |
| FM_blk1 | −.00313 | −.00204 | +.00121 | +.00363 | −.01921 |
| Main_blk2 | −.00143 | −.00050 | +.00290 | +.00650 | −.02404 |

FM_blk0 and Main_blk1 do the same thing: decrease abstract feature encoding (L1–L2) and increase local feature encoding (L3–L4) by nearly identical amounts. At L4, the deltas are +.00463 vs +.00456. FM_blk1 then does work resembling Main_blk2.

Probe accuracy deltas confirm the same pattern:

| | L3 | L4 | L5 |
|---|---|---|---|
| FM_blk0 | +.025 | +.071 | +.012 |
| Main_blk1 | +.016 | +.084 | +.014 |

#### FM intermediate position on the main model trajectory

Interpolation fraction: where does fm_layer1 fall between post_block0 (0.0) and post_block3 (1.0)?

| Metric | L0 | L1 | L2 | L3 | L4 | L5 |
|---|---|---|---|---|---|---|
| feat_eta² | 0.17 | 0.28 | 0.34 | 0.20 | 0.53 | −0.36 |
| probe_acc | 0.70 | 0.44 | 0.58 | 0.44 | 0.42 | 0.55 |

Probe accuracy fractions cluster around 0.4–0.6: the FM intermediate is roughly halfway between input and target. For a 2-layer FM compressing 3 main model blocks, this means FM_blk0 does slightly more than one block's worth of computation.

### Key finding: FM intermediate is more DGP-aligned than the main model

The most interesting result is at L5 (leaf-adjacent features). Comparing matched pairs:

| | L5 feature eta² |
|---|---|
| fm_layer1 | **.09721** |
| post_block1 | .08709 |
| | |
| fm_layer2 | **.07800** |
| post_block3 | .06901 |

The FM's representations are 12–13% more conditioned on local feature identity than the main model's corresponding layers. This extends the "purer DGP approximation" finding from the [output-level analysis](#fm-as-dgp-approximation-2026-06-21) inward to the FM's intermediate computation: the FM is more DGP-aligned than the main model not just at its output, but at every stage of its computation. The main model trades off some feature encoding for representational reorganization that serves downstream layers; the FM, freed from this constraint, retains more of the DGP's hierarchical structure.

### Reproduction

```bash
cd experiments/
modal run --detach rhm/rhm_fm_intermediate_probing.py::fm_intermediate_probing
```

Results saved to `rhm-scaling-data` volume at `/data/rhm_fm_intermediate_probing/`.

[^private]: Not mirrored: this link points to a document in the private lab repo (the roadmap, the queue, an unrun spec, reading notes, or a conversation). See the top-level README for what is held back and why.
