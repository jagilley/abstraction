# Residual decomposition — language and vision domains

**Parent:** [`a2a_forward/README.md`](../README.md) · **Files:** [`FILES.md`](FILES.md)
**Primary writeup (read this first):** [`rhm/residual_decomposition/README.md`](../../rhm/residual_decomposition/README.md)

This node holds the language and vision ports of the residual-decomposition measurement. The finding, the instrument, and the cross-domain synthesis live in the primary writeup; this file covers only what is specific to these two domains.

Both scripts import `decomposition.py` from the RHM node rather than copying it, so all three domains are scored by literally the same code — which is the point of a confirmatory run, and why `rhm` is added to these images' local sources.

## The finding, in one line

`res_var(i) ∝ act_var(i)^β` across the main model's principal directions, R² = 0.95–0.99, with β invariant to a 4× FM-capacity range while capacity moves the frontier's level. Confirmed here in language and vision.

## Language (`language_decomposition_audit.py`)

The canonical **4L/4H/256D GPT (~28.9M params)** from [`LANGUAGE_RATCHET_README`](../LANGUAGE_RATCHET_README.md) on 10M FineWeb-Edu tokens. Chosen so the *domain* is what changes: the RHM main model is a 4L/4H/128D GPT, so both are 4-layer GPTs with the same block structure and the gaps are the same strings.

| gap | β (± sd over 4× capacity) | naive `R_res` | `R_res_participation` | frontier mass |
|---|---|---|---|---|
| b0→b1 | 0.107 ± 0.016 | 244 → 176 | 210–250 | 0.0001–0.038 |
| b0→b3 | **0.358 ± 0.003** | 245 | ~223 | **0.0995** |

Two results specific to language:

- **Naive `R_res` cannot tell noise from structure here at all.** 244.3 at the 1-block gap and 244.7 at the 3-block gap — despite relative residual 0.19 vs 0.41 and cosine 0.98 vs 0.91. On RHM it at least fell 120 → 85 between gaps. This is the sharpest available demonstration that the metric is a noise-floor readout, and it **retires the "language residual is full-rank ~200/256" figure** the belief doc cited: that number was measured in the saturated regime.
- **Language's frontier is genuinely large and high-dimensional** — 10% of activation variance unexplained even with a block-sized FM, spread over ~223 of 256 directions. The original intuition that language has a rich residual was right; rank just wasn't what showed it.

Language has the lowest β of the three domains (0.358 vs ~0.61), and the [width sweep](#width-sweep-width_sweeppy) shows this is a real domain effect, not a `d_model` effect. Its proximate correlate is `R_act/d` = 88.3% — computation smeared across nearly every direction.

## MNIST (`mnist_decomposition_audit.py`)

A **4L/4H/128D ViT** — the same dimensions as the RHM main model, making RHM vs MNIST a domain contrast (synthetic hierarchy vs natural images) at matched architecture. FM capacity configs are consequently identical to RHM's.

| gap | β (± sd over 4× capacity) | alignment | `R_res_participation` | frontier mass |
|---|---|---|---|---|
| b0→b1 | 0.139 ± 0.007 | +0.157 | ~103 | 0.0001 |
| b0→b3 | **0.610 ± 0.007** | **+0.658** | ~20 | 0.0231 |

**Load-bearing gotcha: the ViT attends bidirectionally, the default FM is causal.** [`vit.py`](../vit.py)'s `SelfAttention` applies no mask, but `TransformerForwardModel` defaults to `causal=True`. A causal FM therefore *cannot* reproduce the block regardless of capacity — the same class of confound as Exp. 2's head-count mismatch. The signature is diagnostic: it plateaued at relative residual 0.059 (where RHM reaches 0.003 and language 0.008 at matched relative capacity), never saturated, and so manufactured a spurious law at the narrow gap (β = 0.661, R² = 0.968) where every other domain shows noise. `fwd_causal` now defaults to `False`. **Match the FM's attention mode to the block it predicts.**

The decomposition is scored three ways — all positions, patches only, CLS only — rather than assuming the flattening is harmless, since only CLS receives classification gradient. It turned out not to matter for β (CLS 0.616 ± 0.008 vs patches 0.610 ± 0.007), though CLS does differ in other quantities.

## Width sweep (`width_sweep.py`)

Runs both domains at `n_embd ∈ {64, 128, 256, 512}` with depth, head count, FM capacity (exactly 100% of one block) and gap held fixed. Mounts both Modal volumes — the RHM volume at `/rhm_data`, since both default to `/data`.

β **rises** with width in both domains (RHM 0.523→0.673, MNIST 0.537→0.718), falsifying the hypothesis that language's low β was a `d_model` effect. At matched d=256, RHM/MNIST give 0.640/0.704 against language's 0.358. Frontier mass shows no trend across an 8× width range, confirming the level is set by the capacity **ratio** rather than absolute capacity.

See the primary writeup §5–6 for the concentration correlation and the matched-window control.

## Reproduction

```bash
cd experiments/
# one-time, if {DATA_DIR}/tokens is not populated:
modal run --detach -m language_reduction.experiments.pipeline.stages::tokenize \
    --n-tokens 12000000 --shard-size 4000000

modal run --detach -m a2a_forward.residual_decomposition.language_decomposition_audit::language_audit
modal run --detach -m a2a_forward.residual_decomposition.mnist_decomposition_audit::mnist_audit
modal run --detach -m a2a_forward.residual_decomposition.width_sweep::width_sweep
```

Each has a `::smoke` entrypoint that runs the full path at tiny step counts.

Volume layout (`language-reduction-data`): `/data/a2a_forward/{language,mnist}_residual_decomposition/{gap}/cap{N}pct/`, plus `residual_decomposition_width_sweep/summary.json`.
