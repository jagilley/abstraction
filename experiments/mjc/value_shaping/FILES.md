# Value-shaping — File & figure index

Full file-by-file reference for the value-shaping cut (the value→FM interface, learning-layer,
stationary). Summarized in [README.md](README.md); this is the lookup material. Code lives **in
this folder** (per [STRUCTURE.md](../../../STRUCTURE.md) — code lives at the lowest node that
shares it); the shared dependencies [`../pusher_env.py`](../pusher_env.py) and
[`../shared.py`](../shared.py) stay at the `mjc` node because every experiment there imports them.

## Code files

| File | Purpose |
|---|---|
| `value_shaping.py` | **Stage 1 (value→FM interface, learning-layer).** `run_value_shaping(cfg)`: collect with a strong seek-the-puck policy under a nonlinear `puck_field`; capacity sweep of unshaped (`λ_puck=1`) vs value-shaped (`λ_puck=0`) weighted-Huber FMs + a `λ`-frontier; per-dim R² broken out by regime (free-flight / puck-slide / contact-impact) → the re-allocation readouts + precondition checks (engagement, puck reducibility, capacity). Entrypoint `value_shaping(...)`. See [README.md](README.md). |

## Env support (in the parent, shared)

| Symbol | Purpose |
|---|---|
| `PusherEnv` `puck_field` ([`../pusher_env.py`](../pusher_env.py)) | A deterministic nonlinear **multi-mode force field** applied via `qfrc_applied` at runtime (`_field_force`/`_apply_fields`; on the puck and optionally the pusher via `pusher_amp`): the capacity-hungry-but-reducible **value-irrelevant** factor this cut needs. `#modes` is the capacity-hunger lever, max-`k` the aliasing lever. **Additive — XML byte-identical when `puck_field` is absent, so Cuts #1–3 stay reproducible.** |

## Figures (`figures/<tag>/`)

| Directory | Content |
|---|---|
| `valshape_full_v1/` | The headline run (70K transitions, single seed): `fig1_capacity_frontier` (pusher-vel + puck-slide R² vs capacity, unshaped vs shaped), **`fig2_lambda_frontier`** (the re-allocation tradeoff as `λ_puck`→0), `fig3_perdim` (per-group fidelity, unshaped vs shaped). `results.json` alongside. |
| `valshape_smoke/` | Smoke-test output from the `--quick` configuration (21K transitions); kept for debugging — it is also where the fragile "teeth" (capacity efficiency under data scarcity) show up, so it is *evidence*, not just a smoke. |

## Modal volume layout (`mujoco-control-data`)

```
/data/value_shaping/<tag>/    results.json, fig1_capacity_frontier.png,
                              fig2_lambda_frontier.png, fig3_perdim.png
```
