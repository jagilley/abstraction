# Ballistic control arc — File & figure index

Full file-by-file reference for the ballistic-controller arc (Cuts 4a/4b/4c). Summarized in
[README.md](README.md); this is the lookup material. Code lives **in this folder** (per
[STRUCTURE.md](../../../STRUCTURE.md)); the only shared dependencies are
[`../pusher_env.py`](../pusher_env.py) and [`../shared.py`](../shared.py), which stay at the `mjc`
node because every experiment there imports them.

## Code files

| File | Purpose |
|---|---|
| `ballistic_control.py` | **Cut 4a** — the b-drive version. `run_ballistic_control(cfg)`: grade IDENTICAL FMs (one collection run per fixed `b`) at a SWEEP of commitment horizons `replan_every` (reactive→ballistic), planning horizon held at the full episode. Fixed-b arms give the control-over-b landscape at each horizon; `online_front`/`online_ctrl` self-tune `b` (front = FM-error teacher, control = ballistic-goal-dist teacher). **Result**: a confounded-drive negative — control is sighted once feasible but transmission doesn't grow monotonically with horizon, and the diagnostic (`frontier_err` lowest at `b=0`, patch-visitation inverted) shows the corridor-geometry drive is confounded. `--replan-sweep`, `--teacher-replan`, `--drift-cycles`, `--patch-amp`. |
| `ballistic_control_figure.py` | Local: aggregates the 4a landscape seeds → `figures/ballistic_control_contrast/` (`fig_landscape` control-over-b per horizon; `fig_transmission` spread vs commitment horizon; `fig_selftune` displaced-init b trajectories). |
| `ballistic_transmission.py` | **Cut 4b** — the clean controlled test. `run_ballistic_transmission(cfg)`: a CONTROLLED FM-quality axis via **damping-staleness** (train FM at `d_train`, test at `d_test`; FM-err ∝ `|d_train−d_test|`), graded by three controllers on identical reaches — `reactive` (re-plan every step), `ballistic_cem` (open-loop CEM), `ballistic_bc` (behavior-cloned feedforward motor program `π(s,g)→` full sequence). **Result**: ballistic transmits FM quality ~3–3.3× more than reactive (slopes reactive +0.35, cem +1.07, bc +1.16; 3 seeds). Reports transmission slope d(control)/d(FM-err) per controller. `--damp-trains`, `--d-test`, `--controllers`, `--corridor-r`, `--plan-h`. |
| `ballistic_transmission_figure.py` | Local: aggregates the 4b seeds → `figures/ballistic_transmission_contrast/` (`fig_transmission` control vs FM-error per controller with slopes; `fig_slopes` transmission-slope bars, ×-reactive). |
| `ballistic_readapt.py` | **Cut 4c** — the end-to-end loop. `run_ballistic_readapt(cfg)`: a Type-2 damping drift `d0→d1` leaves the FM stale; **online reward-free** FM re-adaptation at `d1` (self-supervised, growing buffer, snapshot per milestone) is graded under reactive vs ballistic. **Result**: ballistic control recovers to the matched-FM ceiling (gain +0.187, 4.3× reactive's +0.043; 3 seeds), FM error tracks the ballistic recovery. Reference ceiling = a fresh `d1` FM. `--d0`, `--d1`, `--milestones`, `--finetune-steps`, `--controllers`. |
| `ballistic_readapt_figure.py` | Local: aggregates the 4c seeds → `figures/ballistic_readapt_contrast/` (`fig_recovery` control + FM-error vs re-adaptation transitions; `fig_gain` recovery gain per controller, ×-reactive). |

## Children

| Folder | Summary |
|---|---|
| [`directed/README.md`](directed/README.md) | **Directed collection (S0/S1/S2)** — does the value drive choose *where* to collect, and does it pay? Builds the afferent link Cut 4 named as missing, on a new locally-perturbed substrate (`PusherEnv.rot_regions`, spatially-localized command→motion rotations: local *and* open-loop compensable). Directed collection buys **re-adaptation speed, not a better final model**, ~2.9× over uniform and ballistic-specific; reducibility-awareness is load-bearing (error-chasing is the worst directed policy); ensemble **disagreement cannot detect a drift at all**. Its own file index: [`directed/FILES.md`](directed/FILES.md). |

## Figures (`figures/<tag>/`, mirrored from the volume)

| Directory | Content |
|---|---|
| `ballistic_control_contrast/` | 4a: `fig_landscape`, `fig_transmission` (spread vs commitment horizon), `fig_selftune`. Per-run `ballistic_control_{land48_s*,selftune_s*,char*}/`. |
| `ballistic_transmission_contrast/` | 4b: **`fig_transmission`** (control vs FM-error, the headline slopes), `fig_slopes`. Per-run `ballistic_transmission_{trans_s*,chartr*}/` (each also has in-run `fig1_transmission`, `fig2_gap`). |
| `ballistic_readapt_contrast/` | 4c: **`fig_recovery`** (ballistic recovers, reactive flat; FM-error tracks), `fig_gain`. Per-run `ballistic_readapt_{readapt_s*}/` (each also has in-run `fig1_recovery`, `fig2_gain`). |

## Modal volume layout (`mujoco-control-data`)

```
/data/ballistic_control/<tag>/        results.json, fig1..fig4 .png   (4a)
/data/ballistic_transmission/<tag>/   results.json, fig1_transmission.png, fig2_gap.png   (4b)
/data/ballistic_readapt/<tag>/        results.json, fig1_recovery.png, fig2_gain.png   (4c)
```
