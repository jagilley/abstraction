# tiles_twin — files

**Up**: [`../README.md`](../README.md) (the writeup) · [`../FILES.md`](../FILES.md) · **Design record**: [SPEC.md](SPEC.md)

| file | purpose |
|---|---|
| [`corpus.py`](corpus.py) | **local only.** Builds both conditions from one set of renders: 36 schools (4 tilesets × 9) × 64 tilings × 2 crops, cropped aligned (multiple of 32 px) and misaligned (`offset mod 16 ∈ [4,12]`). Writes the swatches, the per-swatch oracle (tile ids, offsets), the per-school tile atlas and edge table, and the `seam`/`offstyle` damage panel with its known validity. |
| [`twin.py`](twin.py) | The Modal oracle module: `unpack_tiles`, and `oracle_run` — tile-recovery purities, grader calibration against known validity, and validation of the plant's own decoded fills. Also `panel_tokens` — the same panel and calibration slices replayed against the **saved** weights, dumping every per-**token** NLL (8-forward and 1-forward) plus the per-item oracle validity of the plant's fills, for [`tailgrade.py`](tailgrade.py). Imports `canvas.plant.plant` for its constants; adds no core readout. |
| [`describe_schools.py`](describe_schools.py) | **local only.** `tiles.describe()` for all 36 schools — coverage, single-tile repairability, greedy dead-end rate per level. The DGP's own depth margins, which cost-to-depth is read against. |
| [`analyze.py`](analyze.py) | **local only.** Fetches both conditions and the oracle, and reduces them to the twin figures. |
| [`adjacency.py`](adjacency.py) | **local only.** The adjacency-support grader (SPEC.md's last section): a possible-set test on the code grid with **zero forward passes**. Builds the set of oriented adjacent code pairs from genuine exemplars (`gA_train`, `gA_train+gB_train`; per-style / pooled / per-tileset\*; min-count 1/2/4) and passes a fill iff every pair it touches — inside the hole and across the hole boundary — is in support. Reports pass rates, soft-statistic AUCs, the plant's fills split by oracle validity, where the unsupported pairs sit, and the support-coverage / false-reject floor. Reads `results/panel_tokens_tw_*.npz` + `results/{index,quant_K512}_twq_*`. |
| [`tailgrade.py`](tailgrade.py) | **local only.** The tail re-analysis of the grade of record (SPEC.md's last section). Re-scores the known-validity panel with nine statistics of the per-token NLL — `mean` (the grade of record), `max`/`top2`/`top4`/`top8`, `frac_hi`, the demand-normalised `max_dev`/`top4_dev`, and `ring_gap` (boundary minus interior) — under an **identical** oracle-free threshold rule, and reports pass rates, the ROC of that rule, the plant's fills split by oracle validity, and where the highest-surprise tokens sit. Reads `results/panel_tokens_tw_*.npz`. |

The five core readouts are produced by [`../plant.py`](../plant.py) unchanged
(`quantize_ladder`, `run`) — that is the point of the twin, and no fork of it exists here.

## Artifacts

| path | what |
|---|---|
| `canvas/corpora/data/tiles_{aligned,misaligned}/` | the corpora (gitignored, 96 MB tar each) |
| `results/describe.json` | the DGP's depth margins per school |
| `results/run_tw_{aligned,misaligned}.json` | the five core readouts per condition |
| `results/quant_twq_{aligned,misaligned}.json` | the quantizer floor per condition |
| `results/oracle_tw_{aligned,misaligned}.json` | tile recovery, known-validity calibration, fill validity |
| `results/panel_tokens_tw_{aligned,misaligned}.npz` | `twin.panel_tokens`' replay: per-**token** NLLs of the calibration slice and of every panel class (ragged), the panel's geometry and known validity, the **per-item** oracle validity of the plant's fills, and the panel's **code grids** (`dcodes`, `fills`) that `adjacency.py` grades without a reader |
| `results/tailgrade.json` | the reduced tail numbers: pass rates, AUCs, fill validity splits, mechanism |
| `results/{index,quant_K512}_twq_{aligned,misaligned}.*` | the split index and the K = 512 codes, fetched off the volume so the support test runs entirely offline |
| `results/adjacency.json` | the adjacency-support numbers: support coverage, pass rates, AUCs, fill validity splits, boundary localisation |
| `figures/twin_*.png` | the twin figures |
| `figures/adjacency_{pass,truth}.png` | the support grader: pass rate by variant × class × level in both conditions, and its separation / truth readouts |
| `figures/tail_{pass,roc,truth}_tw_*.png` | the tail re-analysis: pass rate by statistic, the threshold sweep with the q = 0.90 rule marked, and the plant's fills split by oracle validity beside the boundary-vs-interior readout |

## Modal

App `canvas`, volume `canvas-data`, profile `chromatic`, one L4.

```
/data/corpora/tiles_{aligned,misaligned}/     unpacked corpora (swatches + dmg/ + oracle.npz)
/data/plant/twq_{aligned,misaligned}/         codebooks, codes, quantizer stats
/data/plant/tw_{aligned,misaligned}/          run.json, oracle.json, panel_tokens.npz, weights, strip.npz
```
