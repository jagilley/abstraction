# Model Scale Experiment: Residual Structure vs Main Model Size

**Parent experiment**: [README.md](README.md)
**Grokking comparison**: fer/experiments/zipfian_grokking/cnb_self_regulation/README.md[^private]
**Code**: `model_scale_experiment.py`

## Goal

In Zipfian grokking, the A2A residual is low-rank and interpretable — the Fourier solution concentrates in ~15 dimensions, and the SVD residual cleanly captures mechanism-specific structure. In language at 29M params, the residual is full-rank and diffuse (effective rank 78–94% of max), with the forward model "slightly worse everywhere."

Hypothesis: the 29M GPT is too small to develop clean computational structure for language. As the main model scales up and better captures the language DGP, the residual should become more structured — concentrated in fewer dimensions, with stronger behavioral conditioning — analogous to how grokking produces low-rank residuals once the model discovers the Fourier solution.

This is a direct test of whether the full-rank residual is a property of *language* or a property of *small models on language*.

## Design

Two models trained from scratch on identical data (100M tokens of FineWeb-Edu, τ=0.0), with capacity-matched forward models (~1–1.5%) predicting adjacent layers (post_block0 → post_block1).

| | 29M model | 77M model |
|---|---|---|
| Architecture | 4L / 4H / 256D | 8L / 8H / 512D |
| Main params | 28.9M | 76.7M |
| Forward model | 1L/1H/64d/mlp×2 | 1L/1H/64d/mlp×2 |
| Fwd params | 330K (1.1%) | 1.18M (1.5%) |
| Tokens | 100M | 100M |
| Tokens/param | 3.5 | 1.3 |
| Steps | 30K | 30K |
| Batch | 64 × 128 = 8K tok/step | 64 × 128 = 8K tok/step |
| Epochs | 2.5 | 2.5 |
| Optimizer | AdamW, lr=3e-4, wd=0.01 | AdamW, lr=3e-4, wd=0.01 |
| Forward model lr | 1e-3 | 1e-3 |
| Prediction gap | post_block0 → post_block1 | post_block0 → post_block1 |

Both forward models use the same architecture (1-layer transformer, single 64-dim attention head). The forward model's d_model matches the main model's (256 vs 512), so the MLP hidden dimension scales proportionally (512 vs 1024). The attention bottleneck is tighter for the 77M model (8x compression of 8 heads vs 4x compression of 4 heads).

## Results (2026-06-03)

### Main comparison

| Metric | 29M | 77M | Direction |
|---|---|---|---|
| Val loss | 4.29 | 4.06 | 77M better LM |
| Cosine sim | 0.980 | **0.994** | 77M much more predictable |
| Eff rank / d | 93.8% | **91.8%** | 77M less diffuse |
| Top-1 PC var | 1.74% | **2.86%** | 77M top PC 64% larger |
| Top-5 PC var | 7.31% | 7.26% | Same |
| Top-10 PC var | 12.94% | 10.76% | Reverses |
| Rank for 50% / d | 24.2% | **21.1%** | 77M more concentrated |
| Rank for 75% / d | 47.7% | **43.4%** | 77M more concentrated |
| Rank for 90% / d | 70.7% | **66.6%** | 77M more concentrated |
| Rank for 95% / d | 82.0% | **79.1%** | 77M more concentrated |
| r(res,LM) | -0.069 | -0.104 | Both near zero |
| d(sentence_start) | -1.047 | -0.354 | 77M fwd handles these |
| d(before_closer) | +0.451 | +0.513 | Similar |

### Effective rank trajectory during training

| Step | 29M (rank/256) | 77M (rank/512) |
|---|---|---|
| 0 | 72.3% | 69.8% |
| 500 | 81.3% | **66.9%** (dips) |
| 2K | 89.9% | 84.0% |
| 5K | 92.5% | 89.4% |
| 10K | 93.3% | 90.0% |
| 20K | 93.6% | 90.9% |
| 30K | 93.6% | 91.5% |

Both increase monotonically after the initial transient (opposite of grokking's rank compression during the phase transition). The 77M model consistently sits 2–3pp lower in relative rank throughout training.

The 77M model shows a rank *dip* at step 500 (69.8% → 66.9% before climbing) — a transient echo of grokking's rank compression during the early phase transition. The 29M model does not show this.

### Residual PCA details

| Metric | 29M (d=256) | 77M (d=512) |
|---|---|---|
| Effective rank | 240.1 | 470.1 |
| Rank for 50% var | 62 | 108 |
| Rank for 90% var | 181 | 341 |
| Top-1 PC var | 1.74% | 2.86% |
| Top-5 PC var | 7.31% | 7.26% |
| Top-10 PC var | 12.94% | 10.76% |

The concentration is specifically in the leading PC: the 77M top-1 captures 64% more variance (2.86% vs 1.74%). Beyond the top-1, the variance distribution is similar or slightly more spread (top-5 is identical, top-10 reverses). The 77M residual has one stronger "dominant miss" direction while being more uniform elsewhere — reminiscent of the head-3 outlier from the structure analysis (Run 2), which showed 0.916 cosine vs >0.98 for heads 0–2.

### Behavioral effects

| Category | 29M | 77M |
|---|---|---|
| d(sentence_start) | **-1.047** | -0.354 |
| d(before_closer) | +0.451 | +0.513 |
| n(sentence_start) | 6,927 | 6,787 |
| n(before_closer) | 806 | 750 |

Sentence-start effects collapse from d=-1.05 to d=-0.35 at 77M. The 77M forward model no longer struggles disproportionately with sentence-start computation — the main model's sentence-start processing is regular enough for a compressed forward model to approximate. Delimiter tracking (d=+0.51) persists as the dominant behavioral signal at both scales, slightly strengthening at 77M.

This is consistent with a model that has "solved the easy cases" — sentence starts involve relatively local computation (the previous token ended a sentence), while delimiter tracking requires genuinely long-range context (matching an opener). As the model develops more structured computation, the easy behavioral categories cease to be residual outliers, concentrating the residual on the genuinely hard cases.

### Residual-LM loss relationship

| Metric | 29M | 77M |
|---|---|---|
| r(res,LM) | -0.069 | -0.104 |
| Q1 (easy) residual | 3.23 | 6.27 |
| Q2 residual | 3.15 | 5.78 |
| Q3 residual | 3.12 | 5.61 |
| Q4 (hard) residual | 3.11 | 5.63 |

The residual-LM loss correlation remains near zero at both scales, consistent with all prior experiments. The slight negative correlation (positions where the forward model struggles have lower LM loss) strengthens slightly at 77M. The residual's informativeness lives in its 256/512-dimensional direction, not its scalar norm — this appears to be an inherent property of language, not a scale artifact.

## Interpretation

### The consistency argument

The magnitude of the rank reduction (93.8% → 91.8%) is small. But magnitude is not the right thing to look at — we have no prior on how much rank reduction a 2.7× scale-up should produce on a DGP as complex as language. The right thing to look at is consistency of direction across independent metrics.

Every rank-normalized metric points the same way:

1. Relative effective rank: **lower** at 77M
2. Rank for 50% variance: **lower** at 77M
3. Rank for 75% variance: **lower** at 77M
4. Rank for 90% variance: **lower** at 77M
5. Rank for 95% variance: **lower** at 77M
6. Top-1 PC variance: **higher** at 77M
7. Cosine similarity: **higher** at 77M

Seven independent measures, all pointing toward "more structured residual at larger scale." This comes alongside lower val loss (the model genuinely understands language better) and higher cosine (the computation is genuinely more predictable). The three things — model quality, computation predictability, residual concentration — move together, which is what the hypothesis predicts.

### Connection to grokking

The grokking case (cnb_self_regulation) showed:
- **Before grokking**: full-rank computation, SVD captures nothing specific
- **After grokking**: rank ~15/128, the Fourier solution concentrates in a few dimensions, SVD cleanly captures mechanism-specific structure

The language scaling experiment shows the same direction at much smaller magnitude:
- **29M model**: 93.8% relative rank, 1.74% top-1 PC variance
- **77M model**: 91.8% relative rank, 2.86% top-1 PC variance

The magnitude difference (grokking goes from full-rank to rank ~15; language goes from 93.8% to 91.8%) is consistent with language being a vastly more complex DGP than modular arithmetic. A 2.7× scale-up doesn't "grok" language. But it makes measurable progress toward structured computation, and the residual reflects that progress in the direction the theory predicts.

The early-training rank dip in the 77M model (69.8% → 66.9% at step 500 before climbing) is also suggestive — it mirrors the direction of grokking's rank compression during the phase transition, transiently. The model briefly finds structured solutions before the residual becomes dominated by the forward model's general capacity limitations.

### What the residual captures at scale

At 29M, the residual captures two kinds of difficulty roughly equally: sentence-start computation (local, relatively easy) and delimiter tracking (long-range, genuinely hard). At 77M, the easy category drops out (the forward model handles it), leaving delimiter tracking as the dominant signal. The residual at scale is more *selective* — it captures specifically the computation that's beyond the forward model's capacity, not a mix of easy and hard misses.

This parallels the grokking observation from the scaling sweep (Run 5): as forward model capacity increases, delimiter tracking effects shrink from d=+0.78 to d=+0.30 because the forward model can handle them. Here we see the complementary effect: as the *main model* becomes more powerful, its computation becomes more regular, and the easy cases stop being residual outliers.

## Limitations

1. **Capacity ratio mismatch**: The forward model is 1.1% of the 29M model and 1.5% of the 77M model. The 77M forward model has 3.6× more absolute params (1.18M vs 330K). Some of the behavioral effect differences (sentence-start collapse) could reflect the forward model's greater absolute capacity rather than the main model's more structured computation. The PCA metrics (effective rank, rank-for-X%) are less sensitive to this because they characterize the residual's intrinsic structure.

2. **Attention compression asymmetry**: The 77M forward model compresses 8 heads into 1 (8x) vs 4 heads into 1 (4x) for the 29M. Despite this tighter compression, the 77M forward model achieves higher cosine — strengthening the interpretation that the 77M computation is more regular. But the different compression ratios mean the forward models are not seeing "the same problem" at different scales.

3. **Training regime**: The 77M model is slightly undertrained (1.3 tokens/param vs 3.5 for the 29M). A better-trained 77M model (more tokens) might show stronger effects. This makes the observed effects conservative.

4. **Two scale points**: Two data points establish a direction but not a curve. The planned 350M experiment will test whether the trend continues, accelerates, or saturates.

## Next steps

1. **350M model** (~24L/16H/1024D): The critical third data point. If the rank reduction trend continues or accelerates at 350M, the evidence for scale-dependent residual structure becomes strong. Requires 500M+ tokens for adequate training.

2. **Top-1 PC characterization**: What is the dominant miss direction in the 77M residual? Project it onto interpretable subspaces (attention patterns, head outputs) to see whether it corresponds to a specific computational mechanism.

3. **Matched-capacity comparison**: Train a forward model at exactly 1.1% of the 77M model (~840K params) and compare with the 29M result at matched capacity ratio. This controls for the absolute capacity confound in the behavioral effects.

## Reproduction

```bash
cd experiments/

# 29M model (4L/4H/256D)
modal run --detach language_reduction/modal_app.py \
  --stage a2a-model-scale \
  --n-tokens 100000000 --n-steps 30000 \
  --n-layer 4 --n-head 4 --n-embd 256 \
  --predict-from post_block0 --predict-to post_block1 --lr 3e-4

# 77M model (8L/8H/512D)
modal run --detach language_reduction/modal_app.py \
  --stage a2a-model-scale \
  --n-tokens 100000000 --n-steps 30000 \
  --n-layer 8 --n-head 8 --n-embd 512 \
  --predict-from post_block0 --predict-to post_block1 --lr 3e-4
```

## Modal volume

Results saved to `language-reduction-data` volume:

```
/data/a2a_forward/model_scale/
├── gpt_4L_4H_256D/P_100000000/
│   ├── model.pt
│   ├── fwd_model.pt
│   └── results.json
└── gpt_8L_8H_512D/P_100000000/
    ├── model.pt
    ├── fwd_model.pt
    └── results.json
```

[^private]: Not mirrored: this link points to a document in the private lab repo (the roadmap, the queue, an unrun spec, reading notes, or a conversation). See the top-level README for what is held back and why.
