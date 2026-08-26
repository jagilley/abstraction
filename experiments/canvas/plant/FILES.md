# plant — files

**Up**: [`README.md`](README.md) (the writeup) · [`../FILES.md`](../FILES.md) (canvas) · **Design record**: [SPEC.md](SPEC.md)

## Code files

| file | purpose |
|---|---|
| [`corpus.py`](corpus.py) | **local only.** Renders 36 styles × 64 seeds × 2 crops = 4608 swatches at 256×256 with **seed-disjoint** splits (a swatch-level split leaks, because two crops of one seed overlap in pixels), writes `manifest.json` + per-style `style.json`, and tars the result for `modal volume put`. |
| [`codebook.py`](codebook.py) | The quantizer. `to_patches`/`from_patches` (the patch view, a bijection), GPU Lloyd `kmeans` with dead-centroid respawn, `assign`, `encode` (images → 16×16 code grid + per-swatch patch MSE), `psnr`. Decoder is a lookup, so the reported floor is the alphabet's and nothing else's. |
| [`model.py`](model.py) | The any-order masked token model (`MaskedGrid`), shared by the plant and both graders. Mask sampler (`rect_mask` for the ladder, `train_mask` for the 75%-rect / 25%-scatter training mix), `train` with checkpoint callbacks, and `ladder_nll` — per-(style, mask-size) held-out NLL, the depth-ladder readout. |
| [`sampler.py`](sampler.py) | Any-order (MaskGIT) decoding of a hole with a beam of `width` hypotheses over `steps` confidence-ordered reveals. Returns the completion **and its cost in forward passes** (`steps × width`) — the cost axis of the cost-to-depth curve. |
| [`grader.py`](grader.py) | The grade of record, ported from `rhm/practice/critic/`. `score` (chain-rule mean per-token NLL over random reveal orders; `n_steps=1` is the cheap independent form), `tau_from`/`apply_tau` (the oracle-free q-quantile, bucketed per style × mask-size), `manufacture` (the seven candidate classes, none of which reads a program), `marginals`, `roc`, `pair_stats`. `score_tokens` is `score` un-averaged — the same NLLs per **token**, for the tail re-analysis in `tiles_twin/tailgrade.py`[^private]; `nanmean` of it reproduces `score` exactly. |
| [`recur.py`](recur.py) | Recurrence structure of the code grid: `T[2]` = non-overlapping 2×2 code blocks, `T[3]` = 2×2 blocks *of those* with sub-support entries mapped to OOV — `macros.py`'s ratchet constraint applied to the corpus before any loop runs. Concentration readouts (distinct-per-occurrence, entropy, coverage, mass at support). |
| [`plant.py`](plant.py) | The Modal module and the node itself. `unpack` (tar → volume), `quantize_ladder` (the floor vs K), `run` (plant + graders + all five readouts), `selfcheck` (the eight gates). Runnable end-to-end on CPU via `CANVAS_DATA` / `CANVAS_DEV` and `get_raw_f()`. |
| [`analyze.py`](analyze.py) | **local only.** Fetches `run.json` / `quant_stats.json` / `strip.npz` off the volume and reduces them to the five figures under `figures/`. |
| [`twosided.py`](twosided.py) | **local only.** The two-sided typical-set re-analysis of a run's panel (from `panel_dump.npz`): where each candidate class sits in its bucket's genuine-exemplar distribution, and pass rates under a lower *and* upper quantile cut. |
| [`launch_detached.py`](launch_detached.py) | Session-isolated `modal run --detach` launcher (critic's, retargeted) — a harness stop signal must not reach the client and swallow the post-cancellation `volume.commit()`. |

## Children

| folder | summary |
|---|---|
| `tiles_twin/`[^private] | The aligned/misaligned tiles twin on `tiles.py` (one variable: crop offset), its oracle columns, and the grader re-analyses (`tailgrade.py`, `adjacency.py`). Written up in [`README.md`](README.md); design record in its `SPEC.md`, files in its `FILES.md`. |

## Artifacts

| path | what |
|---|---|
| `results/run_<tag>.json` | everything the run measured: `wave`, `recurrence`, `cost_to_depth`, `grader`, `tau`, `roc`, `pair`, `selected_grader` |
| `results/quant_<qtag>.json` | the quantizer floor per style per K (MSE, PSNR, NMSE, codes used, code perplexity) |
| `results/strip_<tag>.npz` | one held-out swatch per style completed at four mask sizes by each sampler, as codes |
| `results/panel_dump_<tag>.npz` | `plant.panel_dump`'s replay of the panel and calibration slices against saved weights, per-item NLLs at 8 and 1 forwards (for `twosided.py`) |
| `figures/quant_floor.png` | the floor per style vs K, and the hardest styles to quantize |
| `figures/wave.png` | the mask-size ladder over training, raw and normalized, plus half-drop step vs mask area |
| `figures/recurrence.png` | `T[2]`/`T[3]` concentration per style, coloured by the part-y / texture-y control |
| `figures/cost_to_depth.png` | completion quality vs mask area at three sampler costs |
| `figures/grader_calibration.png` | the corpus-size flip, the ROC over q, and pass rates by candidate class |
| `figures/completions.png` | the visual strip — truth, hole, and each sampler's completion |
| `figures/two_sided_<tag>.png` | the two-sided sweep and the class percentiles |

## Modal

App `canvas`, volume `canvas-data` (both from [`../shared.py`](../shared.py)), profile `chromatic`, one L4.

```
/data/corpora/plant0.tar        the uploaded corpus (502 MB)
/data/corpora/plant0/           unpacked: <style>/s<seed>_c<crop>.png + style.json + manifest.json
/data/plant/<qtag>/             index.json, quant_stats.json, quant_K<K>.npz
/data/plant/<tag>/              run.json, strip.npz, verdicts.npz, plant.pt, gA*.pt, gB.pt
```

[^private]: Not mirrored: this link points to a document in the private lab repo (the roadmap, the queue, an unrun spec, reading notes, or a conversation). See the top-level README for what is held back and why.
