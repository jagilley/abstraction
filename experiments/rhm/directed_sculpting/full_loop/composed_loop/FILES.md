# composed_loop — File index

Complete file-by-file reference for this sub-experiment. Summarized in [README.md](README.md).

The apparatus this composes lives in sibling nodes and is imported, not copied:
[`../endo_expansion/endo_graders.py`](../endo_expansion/endo_graders.py) (the disagreement instrument
and `beam_belief_vs_truth`), [`../level_moves/level_moves.py`](../level_moves/level_moves.py) (the move
set, hierarchical damage, `level_value_probe`),
[`../level_moves/level_ladder/level_ladder.py`](../level_moves/level_ladder/level_ladder.py) (the span
FM, `_imagine_plan_moves`, the damage schedule), [`../channel_env.py`](../channel_env.py) (the
environment).

## Code files

| File | Purpose |
|---|---|
| `composed_graders.py` | [`../endo_expansion/endo_graders.py`](../endo_expansion/endo_graders.py) generalised from **blocks** to the level-indexed **move set**: the exact-DP teacher over moves, the endogenous MC-rollout teacher (with the optional `allowed` mask that implements §4's level floor), the wireheading readout, the five-way move-scorer, and FM diagnostics. Carries the one genuinely new object — **`belief_update_moves`**, whose plan term stacks over moves through the span FM so a teacher can write back *commit to this level-ℓ feature* rather than only *regenerate this block*; at `max_level=1` it reduces to `channel_env.belief_update` exactly. Also `draw_moves_block_matched`, the dense-term move draw whose **channel marginal is identical to uniform-over-blocks** — the confound that would otherwise make "level moves help" partly "the level arm saw more tree" |
| `composed_loop.py` | The run. Five base arms (`frozen` / `dense` / `evaluative` / `endo_rollout` / `endo_random`) plus four level-floor arms (`endo_floor` / `endo_floor_static` / `endo_floor_anti` / `endo_treefloor`) under hierarchical damage that deepens across the run. `--max-level` selects the action space (1 = the published block-only one, 4 = level-indexed); `--value-action-space {own,flat,mixed}` selects which behaviour policy generates the **value's training states**, the §5 lever. Carries `selfcheck` (CG1–CG3, CPU), `value_probe` (the static matched-span prerequisite, no loop), and `planner_mean_level` — the level of moves the **beam actually committed to**, as distinct from the teacher's target level |
| `aggregate.py` | Reads the runs and prints, **in this order**: the collapse check (does the outer signal rank moves like the inner one, against the homogeneous floor), the content control (`endo_random` must not beat `frozen`), the 2×2 with its interaction and floor-corrected recovery, the PI-instruction table with its `endo_treefloor` comparator and the H1/H2 diagnostic, teacher fidelity, and whether the teacher's target level follows the damage schedule. `--pattern` is required |

## Results

| Path | What |
|---|---|
| `figures/composed_loop_l{1,4}{a,b}_s1/results.json` | **The 2×2.** `a` = `frozen,dense,endo_random`, `b` = `evaluative,endo_rollout`, at both action spaces. Interaction −0.264 at damage depth 3; the endogenous judge's level gain (+0.189) equals `frozen`'s (+0.186) |
| `figures/composed_loop_fl_{a,b}_s1/results.json` | **The level-floor arms.** `a` = `endo_floor,endo_floor_static`, `b` = `endo_floor_anti,endo_treefloor`. `endo_floor` clears the no-loop floor at all three damage depths; **`endo_floor_anti` is as good or better**, which refutes the curriculum reading the arms were built for |
| `figures/composed_loop_mt_{a,b}_s1/results.json` | **The matched teacher budget** (`rolls=3`, `ground_states=3072` — [`../endo_expansion/`](../endo_expansion/README.md)'s published values). Accuracy rises (2.65× → 3.16× chance) while target level moves 1.72 → 1.78 against an oracle at 2.50. **These two runs predate `planner_mean_level` and report it as absent** |
| `figures/composed_loop_vas_{flat,mix}_{a,b}_s1/results.json` | **The value's exploration distribution.** Only `--value-action-space` differs from the `l4` runs. On `frozen` — where the controller is bit-identical across all three settings — ballistic @dmg3 is 0.316 (`own`) / 0.213 (`flat`) / **0.479** (`mixed`) |
| `figures/composed_value_probe_vp_l4_s1/results.json` | The matched-span prerequisite on a value trained on **level-indexed** actions: `tree|L3` premium 0.537 / 0.530 / 0.520, rank corr decaying +0.288 → +0.075 |
| `figures/composed_value_probe_vp_l1_clean_s1/results.json` | The same on a value trained on **flat** actions: premium 0.595 / 0.634 / 0.610 (published anchor 0.618), rank corr +0.53 at every damage depth — but *confidently wrong at the root* (0.565–0.596 where the DP says 0.22) |
