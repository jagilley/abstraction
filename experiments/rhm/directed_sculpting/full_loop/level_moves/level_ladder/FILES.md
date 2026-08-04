# Files — `level_ladder/`

The allocation half of [`full_loop/`](../../README.md) open item 2: a budget spent over
(channel, level) cells, under a damage depth that deepens across rounds. **Finding: the level
index is inert as an allocation axis (+0.00016 ± 0.00226) and large as an action axis (3.51×
ballistic).** See [README.md](README.md).

**Up**: [`../README.md`](../README.md) (level_moves) · **Node**: [`../../../../README.md`](../../../../README.md) (rhm)

## Code files

| File | Purpose |
|---|---|
| `level_ladder.py` | Everything: the span FM, the cell index, the move-indexed taps/planner, the cell allocator, the damage schedule, and three Modal entrypoints (`gate`, `drive_check`, `level_ladder`) plus the local `selfcheck` |
| `aggregate.py` | Cross-seed tables — the prize, the climb slope, and the deepening-vs-static discriminator. `--pattern` is **required** (results are keyed by seed; two sweeps mirrored into `figures/` would silently overwrite) |

## Key objects inside `level_ladder.py`

| Object | What it is |
|---|---|
| `build_span_fm` → `SpanLatentFM` | `rhm_sculpt_latent._build_block_fm` with the action conditioning generalised from one block to a move's span, plus a level embedding that is **exactly zero at level 1** (`- weight[1]`). That subtraction is what makes gate C4 an identity rather than an approximation |
| `build_cells` | The (channel, level) allocation index. At `max_level=1` there is one cell per channel, so the cell index *is* the channel index and this ladder reduces to the published one |
| `_draw_moves` | Two-stage sampling: draw a **cell** from `w`, then a node inside it uniformly. Drawing moves directly in proportion to `w` would give cells with more nodes a bigger share of the same weight, so `tree\|L1` (8 nodes) and `tree\|L4` (1 node) would not be comparable columns |
| `forecast_visits_cells` | `channel_env.forecast_visits` over moves; returns per-cell visits **and** per-cell \|ΔV\|. This is what the span FM was needed for — the published tap indexes `chan_blk` and cannot reach a level-ℓ node |
| `open_loop_beam_moves` / `_imagine_plan_moves` | Ballistic control over the level-indexed action space. Load-bearing: a block-only planner cannot *execute* a level-ℓ commitment, so under hierarchical damage its ceiling would be set by the damage rather than by the FM |
| `cell_dstar_gain` | Exact per-cell Δ`d*`. Doubles as `oracle_level`'s drive and as gate G-L. Computed **once per damage depth**, not per round — `d*` is a function of the rule support alone and the drift is support-fixed (`verify_backcompat` B5) |
| `allocate_cells` | The six arms. `channel_only` is the load-bearing control (same tap, aggregated to channel, uniform within it — so it *cannot* climb by construction) |
| `parse_schedule` | `"1,2,3"` deepens the error across the run; `"2"` holds it static and is the necessity discriminator |
| `make_damage_mixture` | The value's training distribution. Trained once on a mixture over the schedule's depths and frozen, so every round's damage is in-distribution and the arms differ in the allocation rule rather than in how sighted their grader is |
| `random_node` | The span-matched *uninformed* partner for any move: rewrite the same span with uniformly random **legal** level-1 features. On-grammar on purpose — an off-grammar twin would make the premium a grammaticality readout, the hazard `level_moves` §10 measured at 4.4% |
| `premium_cells` / `drive_candidates` | The six endogenous per-cell drives, side by side against the exact oracle. Built because the two published taps both turned out span-dominated once given a level index (see G-E below) |
| `selfcheck` | **C4** — the span FM is bit-identical to `BlockLatentFM` on the flat move set, tested with a *perturbed* level embedding so the reduction is exact rather than an artifact of the zero init. Also asserts the FM is not blind to the level index above level 1 |

## Gates, in the order they must pass

| Gate | What it asserts | Where |
|---|---|---|
| **C4** | span FM ≡ published block FM at level 1, max \|diff\| **0.0** | `selfcheck` / `selfcheck_remote` — **passing** |
| **G-D** | hierarchical damage is 100% on-grammar with `d*` > 0 at every depth | `level_moves.certify_damage`, asserted in-run — **passing** |
| **G-L** | *the world* makes depth necessary: the exact DP's best move level rises with the damage depth | `gate` — **passing at `n_corrupt=2`** (argmax 1/1/3/3; oracle's implied mean level 2.15 → 2.42 → 2.68). **Fails at the published `n_corrupt=3`**, which damages the whole tree and swamps the structure |
| **G-E** | *the learner* can see it: some endogenous per-cell drive tracks G-L's ordering | `drive_check` — **partial**. Three drives (`abs_dv_mean`, `dv_best`, `dv_best_span`) track with 3/3-consistent positive slopes but at only **7–15% of the oracle's rate**, and the shortfall localises to `tree\|L4`, where the value is **wrong-signed** (−0.9 against the oracle's +0.5 → +1.6). `prem_lazy` appeared to pass on one seed and did not replicate |

## Auxiliary READMEs

*(none — the writeup is [README.md](README.md))*

