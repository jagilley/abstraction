# The planar arm substrate — File & figure index

Full file-by-file reference for the arm substrate. Summarized in [README.md](README.md); this is the
lookup material. The characterization script lives **in this folder** (per
[STRUCTURE.md](../../../STRUCTURE.md)); the DGP [`../arm_env.py`](../arm_env.py) and the Modal
plumbing [`../shared.py`](../shared.py) stay at the `mjc` node as **shared machinery** — `arm_env.py`
is a substrate any cut in `mjc/` can be ported onto, exactly like [`../pusher_env.py`](../pusher_env.py).

## Code files

| File | Purpose |
|---|---|
| `arm_probe.py` | **Arm substrate characterization** (see [README.md](README.md)). `run_arm_probe(cfg)`: P0 FK/axis sanity · P1 capacity frontier · P2 state-locality of staleness · P3 ballistic feasibility vs do-nothing/random floors · P4 composition horizon · P5 reactive-vs-ballistic transmission slopes. `--axis payload\|curl`. FM error is graded on the **task distribution** (transitions visited by matched-FM reaches), with a broad probe kept alongside as the contrast — the fix that took transmission from 0.81x to 24x. `run_arm_capacity_sweep(cfg)`: where capacity binds, over (n_links x v_explore x frame_skip); readout = width needed for R²>=0.99 (pusher ref: 8). `run_arm_tool_probe(cfg)`: the passive tool — tool-blind vs full FM coupling + control cost of dropping the tool vs `tool_mass`. Entrypoints `arm_probe`, `arm_capacity_sweep`, `arm_tool_probe`. |
| `__init__.py` | Empty package marker — makes `mjc.arm_substrate` importable and gets the folder shipped by the Modal image's `.add_local_python_source("mjc")`. |

## Cuts ported onto this substrate

| Node | Content |
|---|---|
| [`../ballistic/arm/`](../ballistic/arm/) | **Cut 4c-arm** (`arm_readapt.py`, `arm_readapt_figure.py`, `train.sh`) — the first real cut on this substrate, run at P5's config byte-identical. Endogenous reward-free FM re-adaptation under a Shadmehr curl-field drift `b0=0 → b1=6`: **5.24× ± 0.18** ballistic-over-reactive recovery gain (3 seeds), ballistic reaching the matched-FM ceiling, plus the **mirror-signed aftereffect** (+0.175 naive-in-field vs −0.168 adapted-in-field-free, ratio 0.96). Validates P3/P4/P5 predictively and forces two corrections to them — see [README.md](README.md) §P7. |

## Env support (in the parent, shared)

| Symbol | Purpose |
|---|---|
| `ArmEnv` ([../arm_env.py](../arm_env.py)) | **Arm substrate DGP** (shared machinery). Planar N-link torque-driven arm: `DEFAULT_DGP`, `build_xml(dgp)` (MJCF from a knob dict — gravity-free planar chain, hinge axes +z, contacts off by default), `ArmEnv` (matches `PusherEnv`'s API: `get_state`/`reset`/`set_state`/`step`, plus `tip_pos`/`goal_pos`/`nonfinite`/`max_absq`), `fk(q, L, upto=k)` (analytic FK, verified against MuJoCo `site_xpos` to 2e-16), `collect_pool` (teleport collection). Perturbations, all additive & off by default: **`payload_mass`** (inertial, config-local), **`curl_field`** (velocity-dependent end-effector field via exact `mj_jacSite` Jacobian = the Shadmehr force-field protocol; the best-measured quality axis + what aftereffects are defined on), `torque_rot` (`push_rot` analog), `joint_noise` (aleatoric), and **`n_passive`/`tool_mass`/`tool_damping`/`goal_site`** (a passive tool = capacity competition from MORPHOLOGY; `goal_site` flips value-relevance with the physics byte-identical). `tool_damping` auto-scales to the explicit-integration stability limit `dt < 2I/c` — a light passive link otherwise NaNs silently. |
| `app`, `volume`, `DATA_DIR`, `NumpyEncoder` ([../shared.py](../shared.py)) | Modal app/image/volume plumbing shared by every script under `mjc/`. |

## Figures (`figures/<tag>/`)

| Directory | Content |
|---|---|
| `arm_probe_{n3_curl,n3_curl_strong,n3_gentle,n3_payload}/` | `run_arm_probe` on the 3-link arm: `fig1_capacity` (capacity frontier, arity-2 vs arity-1), `fig2_locality` (excess stale-error vs the axis's predicted state variable), `fig3_composition` (FM rollout tip-error vs horizon), **`fig4_transmission`** (control vs FM-error per controller — the headline slopes). `n3_curl_strong` is the headline run (strong planner, curl axis, 48 reaches) and carries only `fig2`/`fig4` — capacity and composition were disabled with `--no-do-capacity --no-do-composition`. |
| `arm_probe_{n5_v1,n5_gentle,n5_payload}/` | Same four figures on the 5-link arm — where capacity binds intrinsically (P1) and the composition horizon bottoms out at 14 steps (P4). |
| `arm_capacity_sweep_{cap_v1,cap_v2}/` | `fig_capacity_regimes.png` — R² vs hidden width per (n_links, v_explore, frame_skip). The dedicated, monotone capacity instrument; flat-and-high = the pusher's problem. `cap_v2` is the one quoted in the README. |
| `arm_tool_probe_tool_v1/` | `fig_tool_coupling.png` — **coupling and control cost of dropping the tool vs `tool_mass`** (P6): Cut #4d's capacity-competition boundary condition spanned by one physical knob, in kg. |
| `*_smoke/` | Smoke-test outputs from the `--quick` configurations (`arm_probe_smoke/`, `arm_tool_probe_smoke/`); kept for debugging, not results. |

Every directory also carries `results.json` — the full numeric record the README's tables are read off.

## Modal volume layout (`mujoco-control-data`)

```
/data/arm_probe/<tag>/            results.json, fig1..fig4 .png            (run_arm_probe)
/data/arm_capacity_sweep/<tag>/   results.json, fig_capacity_regimes.png   (run_arm_capacity_sweep)
/data/arm_tool_probe/<tag>/       results.json, fig_tool_coupling.png      (run_arm_tool_probe)
```
