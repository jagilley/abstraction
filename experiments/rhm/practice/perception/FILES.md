# Files — practice/perception

**Up**: [README.md](README.md) · **Parent**: [../FILES.md](../FILES.md)

Imports `conductor/conductor.py` (the arm, the miner, `finetune_generator`), `macros.parse_features`,
`conductor/floors.py`'s replay, `logit_reading/frontier/common.py` (the detector and the induced level field)
and `logit_reading/{calibration,train_trajectory,flat_oracle}.py`, all unchanged.

## Code files

| file | purpose |
|---|---|
| `common.py` | Shared constants (the practice grammar's `v, s, depth, m`, levels, support), the Modal image with the `ignore` that skips `results/` trees, `.log` files, non-`.py` and every sibling package this node does not import, and the volume paths. |
| `record.py` | `record_run` (L4): `conductor`'s `outer_yield` arm at `cd_s0`'s literals with a pure recorder wrapping `macros.parse_features` and `finetune_generator` — the solved beam tips, the mined pieces and the level-1 parse per cycle, copied from tensors already in flight. `check_fidelity` (CPU): gates F-1 (13 series bit-identical to the banked run), F-2 (loop actions equal), F-3 (recorded mine rows equal `n_mined` every cycle). |
| `reader.py` | `train_reader` (L4): `train_trajectory`'s recipe with the window as a parameter (128); `build_stream` makes the `corpus` stream from the grammar and the `own` stream from the recorded solved beam tips. |
| `evaluate.py` | `evaluate` (L4): builds the fixed window set (each mined piece at slot 1 of a four-piece stream of the learner's productions, uniform offset), the reader's entropy profile, the recovered grid via `frontier/common.py::nested_phase` (`meanprof`), `at_support` on the true / recovered / shuffled grid keyed by leaf tokens (`at_support_series`) or, with `--stream-from mine`, by the recorded level-1 features shifted to the recovered grid (`feature_at_support_series`, gate P-1), and the per-level grid decomposition (`grid_quality`); also the corpus-venue period accuracies. |
| `replay.py` | `replay`: `conductor`'s yield rule on a series, yoked to the realised era timeline; `compare`: action-by-action divergence; `in_series_floors`: the null-ABBA dead zone re-measured on a series. |
| `sweep.py` | `sweep` (CPU coordinator): the recorded arm and its gates, then both readers, then both token-keyed panels, across containers in two waves; `smoke`. |
| `analyze.py` | Local reduction: `--fetch` pulls the panels and the recorded run's results into `results/raw/`; writes T0–T10 in [`results/tables.md`](results/tables.md) and, with `--figs`, `figs/{f1_gauge,f2_reader_climb,f3_grid_quality,f4_replay}.png`. |

## Artefacts

| path | what |
|---|---|
| `results/tables.md` | T0 (the series and the reproduction header), T1 (what was run), T2 (gates), and T3–T10 per panel: reader loss and altitude, period accuracies on both venues, the recovered grid per level, the endo read's positions, the `at_support` series and correlations, the per-era reads, the replay, the re-measured dead zones and the action-by-action divergence. |
| `results/raw/*.json` | The fetched inputs: the recorded run's setup and meta, the two readers' training logs, the four panels, the replays. |
| volume `/data/rhm_practice_perception/pc0/` | The readers' checkpoints, the panels, the recorded productions. |
| volume `/data/rhm_practice_conductor/perception_pc0/outer_yield/` | The recorded arm, bit-identical to `cd_s0/outer_yield`. |
