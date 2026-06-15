# Layer Gap Sweep: Prediction Quality vs Number of Intervening Layers

**Parent experiment**: [README.md](README.md)
**Code**: `layer_gap_sweep.py`

## Goal

All main experiments predict a single transformer layer. How does prediction quality degrade when the forward model approximates the joint computation of multiple layers at once?

## Design

Loads the frozen 77M model (8L/8H/512D) from the model scale experiment and trains identical forward models at three prediction gaps, all starting from post_block0.

| | Value |
|---|---|
| Main model | 77M (8L/8H/512D), frozen |
| Main model training | 100M tokens FineWeb-Edu, 30K steps |
| Forward model | 1L/1H/64d/mlp×2 (1.18M params, 1.5%) |
| Forward model training | 2M tokens, 10K steps on cached activations |
| Optimizer | AdamW, lr=1e-3, wd=0.01 |
| Gaps tested | 1 layer, 2 layers, 3 layers |

The forward model architecture is identical across all three gaps. The only variable is the prediction target.

## Results (2026-06-14)

### Summary

| Gap | Cosine | MSE | KL (sub) | KL (abl) | Recovery |
|---|---|---|---|---|---|
| 1 layer | **0.995** | 0.071 | 0.024 | 0.258 | **91%** |
| 2 layers | 0.977 | 0.352 | 0.092 | 0.542 | 83% |
| 3 layers | 0.950 | 0.887 | 0.213 | 0.894 | 76% |

### Residual structure

| Gap | Eff. rank (of 512) | Top-1 PC | Top-5 PC | d(sentence start) | d(before closer) | r(res,LM) |
|---|---|---|---|---|---|---|
| 1 layer | 470.2 (91.8%) | 2.6% | 7.1% | -0.27 | +0.43 | -0.16 |
| 2 layers | 484.9 (94.7%) | 1.4% | 5.5% | -0.66 | +0.67 | -0.19 |
| 3 layers | 486.0 (94.9%) | 1.0% | 4.5% | -1.06 | +0.67 | -0.18 |

### Observations

1. **Graceful degradation.** Cosine drops smoothly from 0.995 to 0.950 and causal recovery from 91% to 76%. Even across a 3-layer gap, the forward model captures most of the computation.

2. **Sentence-start effects sharpen with depth.** The d(sentence start) effect grows from -0.27 to -1.06 as the gap increases. Local computations become relatively easier for the forward model as more layers intervene, presumably because the multi-layer computation builds up increasingly complex non-local structure while local patterns remain simple.

3. **Delimiter tracking stays hard.** d(before closer) is roughly constant across gaps (+0.43 to +0.67), consistent with long-range context matching being a fixed difficulty regardless of how many layers intervene.

4. **Residual becomes more diffuse with larger gaps.** Effective rank ratio increases from 91.8% to 94.9% and top-1 PC drops from 2.6% to 1.0%. More layers of missed computation spread the error across more dimensions. This is the opposite direction from the model scale result (bigger model → slightly more concentrated residual).

5. **Validates frozen-activation training.** The 1-layer gap result (cosine 0.995) closely matches the co-trained result from the model scale experiment (cosine 0.994), confirming that training on frozen activations produces comparable quality.

## Reproduction

```bash
cd experiments/
modal run --detach a2a_forward/layer_gap_sweep.py::a2a_layer_gap_sweep
```

## Modal volume

Results saved to `language-reduction-data` volume:

```
/data/a2a_forward/layer_gap_sweep/
├── summary.json
├── 1_layer/
│   ├── fwd_model.pt
│   └── results.json
├── 2_layer/
│   ├── fwd_model.pt
│   └── results.json
└── 3_layer/
    ├── fwd_model.pt
    └── results.json
```
