# `on_policy/` — file index

Up: [README.md](README.md) · node: [`../README.md`](../README.md) · memo: [`../COLLECTION_REALISM.md`](../COLLECTION_REALISM.md)

## Code files

| file | purpose |
|---|---|
| `verify_backcompat.py` | The backwards-compatibility gate. Four checks: the teleport path is **bit-identical** to the pre-flag implementation (and leaves the RNG at the same position); the plant/knob dict is unmutated; `Body`'s per-episode `MjData` matches the established `set_state`-multiplexed rollout idiom; on-policy transitions genuinely chain within an episode while teleport ones do not. **Run first, and after any edit to `arm_env.collect_pool` or `embodied.py`.** |
| `coverage_probe.py` | **E0** — prices the collection flag. Runs the B0–B3 behaviour ladder at matched environment-step budgets over a sample-size sweep, with every experimenter instrument held fixed, and reports task-probe error, broad-probe error, command-state correlation, coverage, velocity band, excursion counts, and control. Characterization, not a cut — the analogue of `arm_substrate` P0–P6. |
| `readapt_both_ways.py` | **E1** — Cut 4c-arm re-adaptation under three collection modes (`teleport` / `teleport_matched` / `on_policy`) on a milestone grid refined below 400, to test whether the step-like recovery is a teleport artifact. `readapt_both_ways_agg.py` aggregates across seeds from committed `results.json`. |
| `readapt_local.py` | **E2** — the LOCAL-drift version E1 predicted should stretch the on-policy step: a spatially-gated curl (`arm_env` `curl_field["center"]`), with the FM-error split IN-region vs OUT-region (the readout control saturation hides). `readapt_local_agg.py` aggregates. |
| `train.sh` | `verify` / `smoke` / `full` / `readapt` / `local`, logging to `logs/`. |

**E3 lives in its own sub-experiment** — [`directed_on_policy/`](directed_on_policy/README.md) ([FILES](directed_on_policy/FILES.md)): the retracted `ballistic/directed` S2 directed-collection loop (inner FM-readapt + outer `lprog×visits` where-to-collect + ballistic control), on the ON-POLICY arm, with the learning-progress survey itself metered.

> **Launch gotcha (learned the hard way, twice).** Do **not** launch the 3 seeds from one shell with backgrounded `modal run --detach ... &` + `wait` (the `full`/`readapt`/`local` targets do this). Detached mode only guarantees the *last* triggered function survives the parent, and the sibling clients evict each other — E2's first real run lost all three before their `volume.commit()`. **Launch each seed as its own independent background task** (a separate `modal run --detach --tag <t> --seed <n>` per shell). Same fragility is documented in [`../ballistic/directed/README.md`](../ballistic/directed/README.md) §Gotchas.

## Shared machinery this node introduced (lives at the `mjc` node, since both plants use it)

| file | purpose |
|---|---|
| [`../embodied.py`](../embodied.py) | `Body` (metered, per-episode-`MjData`, **no `set_state`**), `collect_on_policy`, the behaviour rungs (`OUBehaviour`, `ReachBehaviour`), goal samplers, and the per-mode diagnostics (`cmd_state_corr`, `coverage_stats`, `pool_diagnostics`). |
| [`../arm_env.py`](../arm_env.py) | `collect_pool` gained `collection_mode="teleport" \| "on_policy"`. Default path byte-identical (gated by `verify_backcompat.py`). E3 added `curl_fields` (list of gated curls) + `noise_fields` (list of spatially-gated aleatoric noise) so one workspace can host the directed-collection 2×2 of local drift/noise regions; absent ⇒ byte-identical (also gated). |
| [`../pusher_env.py`](../pusher_env.py) | New `collect_pool` mirroring the arm's, same flag. The pre-existing `collect_transitions` (cut #1/#2's scripted OU rollout) is untouched — it is the ancestor of this whole idea. |

## Modal volume layout

```
/data/on_policy_coverage/<tag>/results.json      # E0
/data/readapt_both_ways/<tag>/results.json       # E1
/data/readapt_local/<tag>/results.json           # E2
/data/directed_on_policy/<tag>/results.json      # E3
```

## Sub-experiments

| dir | summary |
|---|---|
| [`directed_on_policy/`](directed_on_policy/README.md) ([FILES](directed_on_policy/FILES.md)) | **E3 / the prize** — directed collection on the on-policy arm; the retracted S2 where-to-collect claim reproduced once the learning-progress survey is itself metered (`value` beats `lprog-only` 3/3 seeds; `error-only` collapses on the noisy-TV trap; 1.84× not 22×). |

## Auxiliary READMEs

*(none yet)*
