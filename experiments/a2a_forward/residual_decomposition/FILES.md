# FILES — a2a_forward/residual_decomposition

Parent: [`README.md`](README.md) · [`a2a_forward/FILES.md`](../FILES.md)
Primary writeup: [`rhm/residual_decomposition/README.md`](../../rhm/residual_decomposition/README.md)

## Code files

| File | Purpose |
|---|---|
| `language_decomposition_audit.py` | Confirmatory run on real language: the canonical 4L/4H/256D GPT (~28.9M params) on 10M FineWeb-Edu tokens, arch-matched FM at 25/50/75/100% of one block, both prediction gaps. Holds architecture as close to the RHM main model as possible so the domain is what changes. Finds β = 0.358 ± 0.003 at the 3-block gap and 0.107 ± 0.016 at the 1-block gap, and shows naive `R_res` reading 244.3 vs 244.7 across a noise/structure regime change — retiring the "language residual is full-rank ~200/256" figure as a noise floor. |
| `mnist_decomposition_audit.py` | Vision domain: 4L/4H/128D ViT, same dimensions as the RHM main model, so RHM vs MNIST is a domain contrast at matched architecture. β = 0.610 ± 0.007 (3-block) / 0.139 ± 0.007 (1-block). Scores three position views (all / patches / CLS) since only CLS gets classification gradient. **`fwd_causal` defaults to `False`** — the ViT block attends bidirectionally and a causal FM cannot reproduce it at any capacity; see the README's gotcha. |
| `width_sweep.py` | Runs both RHM and MNIST at `n_embd ∈ {64,128,256,512}` with depth, head count, FM capacity (100% of a block) and gap fixed, to separate "β tracks model width" from "β tracks domain". Falsifies the width hypothesis (β *rises* with width in both) and confirms frontier mass is scale-invariant when the FM's capacity ratio is held fixed. Mounts both Modal volumes, RHM's at `/rhm_data`. |
| `__init__.py` | Package marker. |

The metric library itself is **not** here — it lives at [`rhm/residual_decomposition/decomposition.py`](../../rhm/residual_decomposition/decomposition.py) and is imported, so all three domains are scored by the same code.

## Children

None — leaf node.

## Related, elsewhere

| Path | Relationship |
|---|---|
| [`../../rhm/residual_decomposition/`](../../rhm/residual_decomposition/README.md) | Primary writeup, the metric library, the RHM audit, and the wake-sleep 2×2. |
| [`../LANGUAGE_RATCHET_README.md`](../LANGUAGE_RATCHET_README.md) | Source of the canonical 4L/4H/256D language model config reused here. |
| [`../MODEL_SCALE_README.md`](../MODEL_SCALE_README.md) | 29M vs 77M scale comparison whose 93.8% → 91.8% relative-rank result is reinterpreted by the width sweep (it was measured at the saturated 1-block gap). |
| [`../../language_reduction/tokenize_data.py`](../../language_reduction/tokenize_data.py) | Restored here — `experiments/pipeline/stages.py::tokenize` imported it but the module was missing, leaving the FineWeb-Edu corpus stage unrunnable. |
