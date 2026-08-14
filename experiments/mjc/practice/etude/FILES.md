# FILES — practice/etude

**Up**: [README.md](README.md) (etude) · [../README.md](../README.md) (practice) · [../../README.md](../../README.md) (mjc)

## Code files

| file | purpose |
|---|---|
| `etude.py` | The runner. Builds the étude world (a fixed K-segment piece of via-points through the puck-free corridor world, with localized command-rotation regions on the hard segments), the segment bookkeeping (per-segment tabular EWMA benchmark `b_k`, boundary error `e_k` from an at-tempo run-through, `δ_k = b_k − e_k`, the δ-silence detector), the boundary probes (a tempo ladder over `replan_every`, priced in traversal time under a declared feedback delay `d_fb`), and the compile op (`ballistic_bc` behaviour-cloned from the agent's own recent executed traces of a segment, after which routing switches to the compiled unit). Modal entrypoint `etude`; one container per arm via `.map()`. Forks machinery from `mjc/ballistic/ballistic_transmission.py` (CEM planner, BC motor program) and `mjc/bridge_assembly/bridge_assembly.py` (probe/checkpoint idioms); modifies neither, so every prior result stays reproducible. |
| `analyze_etude.py` | Local post-processing (no compute). Instrument checks (stale vs ceiling reference ladders per segment), the metering trace, the counterfactual δ-silence panel over a (c, W) grid, the priced grade (cumulative practice traversal time × waypoint error at performance tempo, time-to-criterion) and the final tempo ladder. Writes `figures/<tag>/fig1..fig3`. |
| `launch_detached.py` | setsid-isolated `modal run --detach` launcher (the `plasticity_gain` cancellation gotcha fix). Logs to `results/launch_<tag>.log`. |
| `profile_cost.py` | Cost profiler for the substrate: measures where a cycle's wall-clock goes (MuJoCo stepping vs CEM noise/H2D vs FM forwards) at etude.py's exact shapes. Used to size `n_cand`/`n_score`/`n_eval` before the main runs. |
| `etude_scratch.py` | Superseded early draft of `etude.py`, kept for the record; not imported by anything. |

## Children

None.

## Runs on disk

| tag | what it is |
|---|---|
| `smoke` | `--quick` smoke (2 arms, 6 cycles, tiny nets) |
| `cal_s0` | calibration 1: `--arms never --n-cycles 12`, `--pretrain-mode clean`, no replay, no motor noise. Diagnosed catastrophic forgetting (`fm_clean` 0.0064 → 0.0674) |
| `cal_s1` | calibration 2: `--n-cycles 16` with `pretrain_mode=exclude`, `replay_frac=0.5`, `explore_sigma=0.25`. Forgetting fixed (`fm_clean` flat at 0.011–0.013); showed the broad `fm_region` probe is the wrong criterion |
| `cal_s2` | calibration 3: `--n-cycles 30 --n-grad 10`, corridor-slice probe + window-mean silence test. Measured the descent (88% of the stale→ceiling gap closed by cycle 2) and a monotone ballistic-only late drift on the drilled segment |
| `eg_s0` | **the E-gate run**: 5 arms, seed 0, `--n-cycles 56 --probe-every 7 --n-grad 5 --n-rt 64 --sched-early 3 --sched-late 42`. Truncated at c42 (common) / c49 |
| `disc_c15` | compile-hit discriminator, part 1: seeded 15-cycle replay of `never` (bit-identical to `delta_gate`'s stream through its c15 certificate), then the drilled unit compiled three ways — BC on executed commands / on pre-noise commands / on CEM plans |
| `disc2_c15` | discriminator part 2, pure-evaluation probes at the same replay state: one fixed CEM plan, mean command sequence, per-trace own-s0 and cross-s0 replay, BC at 4x width, and the hand-over boundary spread |
| `e5_s1`, `e5_s2` | seed replication of `e5_s0`, identical configuration and bit-identical runner — the three tags pool |
| `e5_s0` | **E-5**: consolidating run at c78 so every commit gets a completed anchor window plus ≥10 post-commit cycles; `seam_drill_fix` gives straddle traces a separate buffer so they add to the selection pool rather than displacing it. Arms `never` / `seq_seam` / `seam_drill_fix`. Seed 0 |
| `e4_s0` | **E-4**: sequential phrase assembly (commit in piece order) with seam-matched selection — candidates scored on hand-over states from the current performance configuration. Arms `never` / `seq_seam` / `seq_practice` (attribution control) / `seam_drill` (straddle practice windows). Seed 0 |
| `e3b_s0` | **E-3b** (complete, c45): selection by expected performance (cross-s0 repeatability, priced into `t_cum`), `b` anchored to the committed unit's own post-commit level, plan-commit from a noise-free hand-over estimate. 5 arms, seed 0 |
| `e3_s0` | **E-3**: the compile op replaced by SELECTION + verbatim commitment, with certificate-anchored benchmarks and the overspeed drill. 5 arms, seed 0 |

## Data layout

- Modal volume `mujoco-control-data`: `/data/practice_etude/<tag>/<arm>/results.json` (+ `done.txt`).
- Local mirror written by the entrypoint: `results/<tag>/<arm>.json`.
- Figures: `figures/<tag>/`.
