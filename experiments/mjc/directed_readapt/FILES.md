# Directed reward-free re-adaptation — File & figure index

Full file-by-file reference for the directed-re-adaptation cut (the two instructive negatives).
Summarized in [README.md](README.md); this is the lookup material. Code lives **in this folder**
(per [STRUCTURE.md](../../../STRUCTURE.md) — code lives at the lowest node that shares it); the
shared dependencies [`../pusher_env.py`](../pusher_env.py) and [`../shared.py`](../shared.py) stay
at the `mjc` node because every experiment there imports them.

## Code files

| File | Purpose |
|---|---|
| `directed_readapt.py` | **Directed reward-free re-adaptation (two negatives).** `run_directed_readapt(cfg)`: on Cut #3's substrate, compare disagreement-directed vs undirected reward-free collection on re-adaptation efficiency (transitions-to-recover). `shift_mode="global"` (Cut #3 drag collapse → null: no scarcity) vs `"patch"` (a localized force-jet via [`../pusher_env.py`](../pusher_env.py)'s `patch` DGP key → scarce; adds per-region in/out FM R² + patch visitation). The disagreement drive **under-visits** the patch — blind to a confident-prior shift. Self-contained (duplicates `dynamics_shift` helpers). Entrypoint `directed_readapt(...)`. See [README.md](README.md). |

## Env support (in the parent, shared)

| Symbol | Purpose |
|---|---|
| `PusherEnv` `field_patch` / `patch` DGP key ([`../pusher_env.py`](../pusher_env.py)) | A Gaussian-gated, spatially-localized force **jet** on the pusher applied via `qfrc_applied` (`_apply_pusher_perturb`) — the localized-shift substrate that manufactures scarcity (only in-patch transitions are informative). Additive and off by default, so the field-off path is byte-identical and Cuts #1–3 stay reproducible. |

## Figures (`figures/<tag>/`)

| Directory | Content |
|---|---|
| `directed_readapt_full_v1/` | **Exp 1 — global shift (the null)**, single seed: `fig1_recovery_curve` (planning dist vs #reward-free transitions, directed vs undirected), `fig2_transitions_to_recover` (headline bar), `fig3_r2_recovery` (global FM fidelity). No `fig4` — the in-patch diagnostic only exists in `patch` mode. |
| `directed_readapt_patch_v1_s{0,1,2}/` | **Exp 2 — localized force-jet patch (the drive negative)**, 3 seeds: `fig1`–`fig3` as above plus **`fig4_inpatch_recovery`** (in-patch R² recovery + the patch-visitation bar — the diagnostic that exposes the disagreement-blindness). |
| `directed_readapt_smoke/` | Smoke-test output from the `--quick` configuration; kept for debugging, not a result. (Its apparent 25-vs-50 directed edge was tiny-N noise — see [README.md](README.md) Exp 1.) |

## Modal volume layout (`mujoco-control-data`)

```
/data/directed_readapt/<tag>/   results.json, fig1_recovery_curve.png,
                                fig2_transitions_to_recover.png, fig3_r2_recovery.png,
                                fig4_inpatch_recovery.png (patch mode only)
```
