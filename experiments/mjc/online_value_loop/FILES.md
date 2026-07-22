# Online value loop — File & figure index

Full file-by-file reference for the fully-online two-timescale value loop (afferent Cut A +
efferent Cut B). Summarized in [README.md](README.md); this is the lookup material. Code lives
**in this folder** (per [STRUCTURE.md](../../../STRUCTURE.md)); the shared dependencies
[`../pusher_env.py`](../pusher_env.py) and [`../shared.py`](../shared.py) stay at the `mjc`
node because every experiment there imports them.

> Not to be confused with [`../drift_value_loop/online_value_loop.py`](../drift_value_loop/online_value_loop.py)
> — the drift-value-loop arc's Cut 3, a *different* script that reuses `meta_curiosity_loop.py`'s
> machinery under a selectable FM-error teacher. See [../drift_value_loop/README.md](../drift_value_loop/README.md).

## Code files

| File | Purpose |
|---|---|
| `meta_curiosity_loop.py` | **The afferent online two-timescale loop (self-tune *where to collect*) — Cut A.** `run_meta_curiosity_loop(cfg)`: on [`../curiosity_control/curiosity_control.py`](../curiosity_control/curiosity_control.py)'s substrate, a slow reward-driven outer loop (advantage-normalized REINFORCE on the explore/exploit balance `b=σ(θ)`, updated every `outer_m` rounds; reward = −CEM-MPC goal-dist, no wireheading) self-tunes the drive set-point in ONE continuous run vs fixed-`b`/random references. **Control-negative**: the drive's tracking gradient is real (pure-explore tracks a drifting needle 3× worse under noise — the noisy-TV pathology) but CEM-MPC control is *robust* to it → flat control landscape → the loop wanders by seed. Knobs: `--noise`, `--noise-onset`, `--outer-m/-alpha/-sigma`, `--task-geom straddle|corridor`, `--patch-amp/--replan-every` (landscape characterization — no sensitive window). Entrypoint `meta_curiosity_loop(...)`. See [README.md](README.md). |
| `meta_value_online.py` | **The efferent online two-timescale loop (self-tune *what the FM models*) — Cut B.** `run_meta_value_online(cfg)`: #4e's substrate (context-FM `f(s,u,z)` + puck/pusher force-field + CEM-MPC, from [`../meta_adapt/meta_value_learn.py`](../meta_adapt/meta_value_learn.py)), but ONE live FM trains continuously while a slow reward loop tunes its scalar puck-weight `w_puck` — `--outer-mode single` (REINFORCE; needs warmup-to-convergence to kill the trend confound, which *also* washes the signal out) or `fork` (antithetic ES with common random numbers, pre-convergence). **Finding**: the value-shaping benefit is a slow, long-horizon, commit-dependent *transient* (gap shrinks/reverses with training: +0.016→+0.004 / +0.040→−0.025; 150-step fork probes too short to see it) → fast-online discovery is **obstructed** (why #4e went offline). The re-allocation IS real in one live FM (`w_puck` 0.5→0.26, pusher-vel R² 0.69→0.90). `--field-pusher-amp` (>0 = capacity competition). Entrypoint `meta_value_online(...)`. See [README.md](README.md). |

## Figures (`figures/<tag>/`)

Cut A tags carry `fig1_learned_b` (the learned set-point trajectory + onset marker),
`fig2_control` (control + frontier-tracking per arm), `fig3_occupancy`, `fig4_orderparams`.
Cut B tags carry `fig1_learned_w` (learned `w_puck` trajectory vs fixed references),
**`fig2_control_realloc`** (control + the pusher-vel↑/puck-vel↓ re-allocation in one live FM),
`fig3_orderparams`. Each directory also holds `results.json`.

| Directory | Content |
|---|---|
| `meta_curiosity_loop_char_{clean,noise}/` | **Cut A landscape characterization, strong needle** (`--patch-amp 7 --replan-every 12`, seed 0, noise off/on): everyone controls *badly* (~0.40 ≈ the random floor) and the landscape over `b` is flat — half of the "no sensitive window" result. |
| `meta_curiosity_loop_ball_{clean,noise}/` | **Cut A at the ballistic edge** (`--patch-amp 5 --replan-every 10`, seed 0, noise off/on) — the most commitment-heavy setting tried here; control ~0.27 and still flat, with `b0.0 ≈ random ≈ best`. (The commitment axis is pursued properly in [../ballistic/README.md](../ballistic/README.md).) |
| `meta_curiosity_loop_smoke/` | Smoke-test output from `--quick`; this is where the mechanism *did* fire (`b` 0.5→0.22 on a mid-run noise onset). Kept for debugging. |
| `meta_value_online_comp_s0/` | **Cut B, marginal capacity competition** (hidden 64, amp 2.0, `single` mode): the loop drives `w_puck` 0.5→0.26, cost 0.305→0.205, pusher-vel R² 0.69→0.90 — the re-allocation, live. |
| `meta_value_online_comp_strong_s0/` | **Cut B, stronger competition** (hidden 40, `--field-pusher-amp 3.0`, `single` mode): the setting whose drop-vs-keep gap **reverses** with training (+0.040 at warmup → −0.025 at convergence) — the washout, and the un-chased "modeling the irrelevant subspace regularizes" thread. |
| `meta_value_online_comp_fork_s0/` | **Cut B, `--outer-mode fork`** (antithetic ES, common random numbers, pre-convergence): the fork `diff` signals are tiny/sign-inconsistent (\|diff\| ~0.002–0.018 vs a true gap of 0.043) and the loop drives `w_puck` the **wrong way** — the long-horizon obstruction, exposed. |
| `meta_value_online_easy_s{0,1,2}/` | **Cut B's no-competition null** (`--field-pusher-amp 0.0`), 3 seeds: the landscape is flat and the loop is correctly *indifferent* (`w_puck` wanders 0.49–0.82). |
| `meta_value_online_smoke/` | Smoke-test output from `--quick`; the run that skipped warmup and chased raw training progress (the trend-confound trap, kept as the record of it). |

Cut A's 2×2 discovery matrix (`disc_clean_s*`, `disc_noise_s*`, `onset_s*`) has **no local figure
directory** — those clients disconnected before mirroring. The runs completed remotely; pull them
from `/data/meta_curiosity_loop/<tag>/` on the volume (the README's numbers come from there and
from the run logs).

## Modal volume layout (`mujoco-control-data`)

```
/data/meta_curiosity_loop/<tag>/   results.json, fig1_learned_b.png, fig2_control.png,
                                   fig3_occupancy.png, fig4_orderparams.png        (Cut A)
/data/meta_value_online/<tag>/     results.json, fig1_learned_w.png,
                                   fig2_control_realloc.png, fig3_orderparams.png  (Cut B)
```

Use `--detach`, and launch runs as **independent** background processes — parallel `&` plus
`--detach` can lose all-but-the-last on a client disconnect.
