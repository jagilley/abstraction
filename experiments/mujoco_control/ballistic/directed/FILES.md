# Directed collection — File & figure index

Full file-by-file reference for the directed-collection cut (S0/S1/S2). Summarized in
[README.md](README.md); this is the lookup material. Code lives **in this folder** (per
[STRUCTURE.md](../../../../STRUCTURE.md)); the only shared dependency is
[`../../pusher_env.py`](../../pusher_env.py) and [`../../shared.py`](../../shared.py), which stay
at the `mujoco_control` node because every experiment there imports them.

## Code files

| File | Purpose |
|---|---|
| `directed_separability.py` | **S0 — the go/no-go.** `run_directed_separability(cfg)`: three preconditions in one run. **(A) compensability** — grade a matched FM in the rotated vs clean world (want ≈0 cost, so the localized command rotation is open-loop correctable, unlike 4b's incompensable force jet); **(B) control-relevance** — train one FM per single-region ablation (correct everywhere except it believes region *j* is unrotated) and grade, so "which region is stale" becomes a measurable axis; **(C) locality/transfer** — from an everywhere-stale FM, spend a fixed budget in region *j* only and read the K×K error-reduction matrix, plus a directed-vs-uniform preview at equal budget. **Result**: all pass, 3 seeds (compensability cost ≈ −0.03, ablation spread 11× larger for ballistic, locality 14–31×, directed advantage 0.45–0.49 ballistic vs 0.03–0.04 reactive). `--regions`, `--region-sigma`, `--budget`, `--replay-n`, `--controllers`. |
| `directed_ladder.py` | **S1 — the one-shot allocation ladder.** `run_directed_ladder(cfg)`: the 2×2 partition (on/off the reach × reducible/aleatoric), one scarce budget, hand-specified allocation arms. Computes four per-region signals — ensemble **disagreement** (internal, no env), **visitation** (ballistic plan rolled through the FM itself, no env), **raw error** and **learning progress** (both from a free monitoring survey; LP = fit half, measure held-out error removed). Arms: `uniform-box`, `oracle`, `random`, `visitation-only`, `error-only`, `disagreement-only`, `lprog-only`, `error × visits`, `lprog × visits [VALUE]`. **Result**: the VALUE arm recovers the oracle allocation (400/0/0/0) from reward-free signals, gap-closed 0.92–1.18 across 3 seeds, every ablation failing in its predicted direction; **disagreement is flat (0.0014–0.0025) and cannot detect the drift**. `--regions` (`cx,cy,phi_pre,phi_post,noise,name`), `--budget`, `--monitor-n`, `--lp-steps`, `--ens-n`. |
| `directed_loop.py` | **S2 — the online loop against a moving target.** `run_directed_loop(cfg)`: T rounds of survey → allocate → collect → fine-tune → grade, with one reducible region re-drifted every `drift_every` rounds along `drift_cycle` (default `0,0,1`, so 2/3 of drift events land on the reach — an off-reach drift is behaviorally inert and cannot discriminate any policy). Per-region ring buffers, invalidated on that region's drift; out-of-region replay. Policies: `uniform`, `oracle` (privileged reducible-error × visitation), `value`, `value-floor` (no-op → uniform box), `value-maint` (no-op → ∝ visitation), `lprog-only`, `visits-only`, `error-only`. **Result**: concentration ≫ uniform (2.3–3.9×) and reducibility-awareness load-bearing (`error-only` worst directed, 1.5–1.9×) are robust; the conjunction never beats `lprog-only` in a loop. `--rounds`, `--drift-every`, `--drift-cycle`, `--budget`, `--buf-cap`, `--value-floor`, `--err-floor`, `--reactive-every`, `--policies`. |
| `directed_loop_figure.py` | Local cross-seed aggregation for S2 → `figures/directed_loop_agg_<out>/`. Reports **adaptation speed** (`excess damage` = integrated ballistic distance above a shared absolute floor after each on-reach drift; `rounds-to-recover`, censored and retired) beside the **asymptotic** mean-over-rounds metric that nulls, plus tracking. Encodes two metric corrections in its own docstring: on-reach-only events, and an absolute shared recovery threshold (normalizing to each policy's own epoch minimum rewards a uniformly-mediocre policy). `--tags`, `--out`, `--tol`. |
| `directed_loop_from_logs.py` | Recovers `directed_loop.py` results from the printed **run logs** when `results.json` was never committed (loopD: all three clients killed at ~91%, and the volume commits only at the end of the function). Parses the drift schedule + per-round ballistic distances, optionally splices a completed arm from another run (`--splice-from`/`--splice-policy`), and `--verify-splice` checks the overlap empirically (observed max |diff| = 5e-5 = print rounding). This is the provenance of every loopD number in the README. |

## Env support (in the parent, shared)

| Symbol | Purpose |
|---|---|
| `PusherEnv` `rot_regions` ([../../pusher_env.py](../../pusher_env.py)) | K independent, spatially-localized **command→motion rotations**: `qfrc = gear · (Σ_j w_j(pos)·(R(φ_j) − I)) @ ctrl`, each region a Gaussian gate optionally also carrying a `noise` amplitude (the aleatoric decoy). Local *and* open-loop compensable — the design unlock this cut turns on. Runs last in `step()` and accumulates iff another writer touched the pusher DOFs that substep, so noise and rotation coexist. Off by default (`dgp` has no `rot_regions`) → all prior cuts byte-identical (verified: 0.0 max difference over 200 random transitions with φ=0). |

## Figures (`figures/<tag>/`)

| Directory | Content |
|---|---|
| `directed_separability_sep_s{0,1,2}/` | S0: `fig1_compensability` (clean/rotated matched/rotated stale, per controller), `fig2_ablation` (per-region ablation cost), **`fig3_transfer`** (K×K locality matrix), `fig4_allocation_preview` (directed vs uniform at equal budget). |
| `directed_ladder_lad_s{0,1,2}/` | S1: **`fig1_ladder`** (control per arm, both controllers, stale reference line), `fig2_signals_alloc` (the four per-region signals + a where-each-arm-spent heatmap), `fig3_gap_closed`. |
| `directed_loop_{loop,loopB,loopC}_s{0,1,2}/` | S2 per-run: `fig1_rounds` (ballistic distance per round with drift markers), `fig2_alloc_trace` (allocation heatmap per policy over rounds), `fig3_region_err` (per-region FM error over rounds). loopD has no figure dir — killed before commit. |
| `directed_loop_agg_{loopA,loopB,loopC}/` | S2 aggregates: **`fig1_speed_vs_asymptote`** (the discriminating metric beside the nulling one), `fig2_tracking_vs_speed` (does choosing better pay off?), `fig3_rounds` (seed-averaged traces). `summary.json` alongside. |
| `*_smoke/` | Smoke-test outputs from the `--quick` configurations; kept for debugging, not results. |

## Run logs (`logs/`)

`directed_{separability,ladder,loop}_<tag>.log` — stdout of every detached run. **Load-bearing for loopD**, whose `results.json` does not exist; `directed_loop_from_logs.py` parses these.

## Modal volume layout (`mujoco-control-data`)

```
/data/directed_separability/<tag>/   results.json, fig1..fig4 .png            (S0)
/data/directed_ladder/<tag>/         results.json, fig1..fig3 .png            (S1)
/data/directed_loop/<tag>/           results.json, fig1..fig3 .png            (S2; absent for loopD)
```
