# Curiosity on control — File & figure index

Full file-by-file reference for the curiosity-on-control cut. Summarized in
[README.md](README.md); this is the lookup material. Code lives **in this folder** (per
[STRUCTURE.md](../../../STRUCTURE.md)); the shared dependencies
[`../pusher_env.py`](../pusher_env.py) and [`../shared.py`](../shared.py) stay at the `mjc`
node because every experiment there imports them.

## Code files

| File | Purpose |
|---|---|
| `curiosity_control.py` | **Curiosity on control — the afferent explore drive + grounding (interface next-piece #1).** `run_curiosity_control(cfg)`: on Cut #3's puck-free reaching, an intrinsic drive directs **where to collect** as a scarce reducible **`field_patch` drifts** (RPF ensemble `build_member` + recency FIFO; teleport-region collection over a G×G grid; CEM-MPC control readout + unbiased frontier-tracking error). Drives (all share FM/init/update; only collection differs): `surprise`, `disagree`, **`reducible`** (error×disagreement), `lp` (`−d‖e‖/dt`), `taskonly` (on-policy), `random`, and **`grounded@b`** (`b·reducible + (1−b)·exploit`, exploit = start→goal path density = the `p`-tap). Findings: the drive **tracks the moving frontier** (non-stationary effect, ties when static); **value-relevance gates usefulness** (tracks off-path but only pays on-path); **pure curiosity chases the noisy TV** (disagreement fails on low-dim aleatoric noise); **grounding (two additive drives) cures it** with an intermediate-balance optimum. Knobs: `--drift-mode morph|none`, `--value-rel on|off`, `--noise`, `--arms` (incl. `grounded@<b>`), `--frontier-task`, `--balance`. Entrypoint `curiosity_control(...)`. See [README.md](README.md). |

## Env support (in the parent, shared)

| Symbol | Purpose |
|---|---|
| `PusherEnv` `field_patch` / `noise_patch` ([../pusher_env.py](../pusher_env.py)) | The **drifting frontier** (`field_patch` — a Gaussian-gated multi-mode *reducible* force) and the **noisy-TV decoy** (`noise_patch` — a Gaussian-gated stochastic/aleatoric force), both applied to the pusher via `_apply_pusher_perturb`. Additive and off by default, so the field-off path is byte-identical and Cuts #1–4e stay reproducible. |

## Figures (`figures/<tag>/`)

Every tag directory holds the same three panels plus `results.json`:
`fig1_curves` (control + frontier-tracking vs round, per arm) ·
**`fig2_occupancy`** (collection heatmaps with the frontier drift path + noise-patch marker —
the clearest view of tracking vs noise-seduction) ·
`fig3_orderparams` (final control / frontier-err / where-they-collect bars).

| Directory | Content |
|---|---|
| `curiosity_control_ft_drift_s{0,1,2}/` | **Finding 1** — the `frontier_task` (undiluted) drift regime, 3 seeds: the drive tracks the moving frontier and sustains control. |
| `curiosity_control_ft_stat_s{0,1,2}/` | **Finding 1's stationary null** (`--drift-mode none`), 3 seeds: everyone masters a static frontier → the `reducible − random` gap collapses to ≈0. |
| `curiosity_control_ground_noise_s{0,1,2}/` | **Finding 4** — the `grounded@b` balance sweep with the aleatoric `noise_patch` **on**, 3 seeds: pure explore is seduced (noise-frac 0.10–0.20) while `b≈0.3–0.5` is immune (0.00). |
| `curiosity_control_ground_clean_s{0,1,2}/` | **Finding 4's noise-off arm** of the same sweep, 3 seeds — gives the 2×2 boundary (grounding helps *most* under noise). |
| `curiosity_control_drift_{v1,s1,s2}/` | Earlier **diluted-task** drift runs (before `--frontier-task`); the drift effect is present but diluted by goals that don't straddle the frontier. |
| `curiosity_control_stat_{v1,s1,s2}/` | The matching diluted-task stationary nulls. |
| `curiosity_control_offpath_{v1,s1,s2}/` | **Finding 2** — `--value-rel off`: the frontier drifts *off* the goal corridor. The drive still tracks it best, but the control gap collapses to ≈0 (true-but-useless). |
| `curiosity_control_noise_v1/` | **Finding 3** — the first noisy-TV run: `reducible`/`disagree` chase the aleatoric patch and get derailed from the frontier. |
| `curiosity_control_smoke/` | Smoke-test output from the `--quick` configuration; kept for debugging, not results. |

## Modal volume layout (`mujoco-control-data`)

```
/data/curiosity_control/<tag>/   results.json, fig1_curves.png, fig2_occupancy.png, fig3_orderparams.png
```

Mirrored locally to `figures/curiosity_control_<tag>/` when the client survives the run;
otherwise pull from the volume (runs exceed the ~2-min client window — use `--detach`).
