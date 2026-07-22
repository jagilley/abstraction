# Meta-adaptation arc (Cuts #4–#4e) — File & figure index

Full file-by-file reference for the meta-adaptation arc. Summarized in [README.md](README.md);
this is the lookup material. Code lives **in this folder** (per
[STRUCTURE.md](../../../STRUCTURE.md), code lives at the lowest node that shares it); the shared
dependencies [`../pusher_env.py`](../pusher_env.py) and [`../shared.py`](../shared.py) stay at the
`mjc` node because every experiment there imports them.

## Code files

| File | Purpose |
|---|---|
| `meta_adapt.py` | **Cut #4 (floor) — meta-learning a fast-adapting FM initialization.** `run_meta_adapt(cfg)`: task family = `joint_damping ∈ [0.05,2.0]` log-spaced (native DGP knob, no env change), disjoint interior (interpolation) test dampings. FOUR initializations — **META** (first-order Reptile, learned-for-adaptability), **MULTITASK** (pooled on the union of train buffers; the load-bearing control), **SINGLE-TASK** prior (one reference damping = Cut #3 d0), **SCRATCH** (random) — each adapted to held-out tasks by the *identical* inner loop (only the init differs). Readouts: held-out **velocity-dim Δs R²** vs #adaptation-transitions (damping acts on velocity; all-dim R² saturates), + CEM-MPC planning (Cut #3 readout), + per-test-task oracle ceiling (large disjoint pool). Deliverable = the **meta−multitask gap** as an order parameter (FLOOR expectation ≈0 = RHM-collapse anchor on control). Self-contained (duplicates Cut #3 helpers). Entrypoint `meta_adapt(...)`. |
| `meta_context.py` | **Cut #4b — the dissectible outer memory.** `run_meta_context(cfg)`: replaces `meta_adapt.py`'s Reptile weight-init with an **explicit context latent** `z` conditioning `f(s,u,z)` — a DeepSets encoder infers `z` from N recent `(s,u,Δs)` transitions (PEARL/amortized system-ID; few-shot adaptation = a forward pass, no gradient). Arms: **context** (`z=E(N)`) vs **z0** (severed, family-average) vs plain-`f(s,u)` oracle. Headline = the **`z`→task-param linear probe** (system-ID). Finding: `z` decodes the param in *both* regimes (floor R²≈0.99, conflict R²≈1.0) but the adaptation gap opens *only* under conflict — **system-ID ⊥ adaptation-benefit**. Same family machinery as `meta_adapt.py` (`--family damping|actuator`). Entrypoint `meta_context(...)`. See [README.md](README.md) Cut #4b. |
| `meta_active.py` | **Cut #4c — value-directed identification (VoI in the loop; a robust negative).** `run_meta_active(cfg)`: with the `z`-conditioned FM (Cut #4b), does a value-of-information drive identify φ in fewer transitions than passive? Four collection arms — **passive** (OU), **magnitude** (max‖u‖ control), **myopic VoI** (max codebook `{z_k}` disagreement), **navigating VoI** (`info_cem`/`infompc_collect` — an info-MPC planning to *reach* high-disagreement states). `z` inferred by optimization; sensitive metric = **in-patch R²** + φ-decode error; **patch visitation** = the mechanism check. Finding: scarcity gates active>passive (global rotation = null; `rot_patch` = scarce), but VoI does **not** beat magnitude even though the info-MPC verifiably navigates (visitation 0.28 vs 0.09; 4 seeds). `--family`/`--conflict`/`--rot-patch`. Entrypoint `meta_active(...)`. See [README.md](README.md) Cut #4c. |
| `meta_active_seeds_figure.py` | Local post-processing: aggregates the 4 seed runs (`act_patch4{,_s1,_s2,_s3}`) into `figures/meta_active_seeds/fig_seeds.png` — per-arm in-patch R² + φ-error (mean±sem over seeds) and the patch-visitation ladder. Resolves the single-seed ranking (which was seed-noise). |
| `meta_value_shaping.py` | **Cut #4d — value-shaping × meta-conditioning (disc-4 on MuJoCo).** `run_meta_value_shaping(cfg)`: fuses `value_shaping` (per-dim re-allocation) × `meta_context` (context latent + `push_rot` conflict) on the puck+field substrate. A **2×2** — {veridical = match all 8 Δs dims, value-shaped = match pusher dims `[0,1,4,5]` = the goal-reaching value's support, drop the puck} × {context `z=E(N)`, pooled `z=0`} — one-step weighted-Huber; per-dim R² + z→φ probe (Stage 1) and an optional (`--plan`) CEM-MPC goal-reaching readout (Stage 2). Findings: puck-drop is **value-caused** (not tuned); fusion (pooling collapses for both objectives); veridicality⊥usefulness; value-shaping is control-**neutral** on an easy pusher but control-**superior under capacity competition** (`--field-pusher-amp` on → +0.036±0.012 planning gap at h=64, 4 seeds). Entrypoint `meta_value_shaping(...)`. See [README.md](README.md) Cut #4d. |
| `meta_value_shaping_seeds_figure.py` | Local post-processing: aggregates the 4 capacity-competition seeds (`plan_pfield_v1`,`_s1/_s2/_s3`) into `figures/meta_value_shaping_seeds/fig_seeds.png` — planning goal-dist + the shaped−veridical **order parameter** (mean±sem + per-seed scatter) + the FM-side pusher-vel capacity frontier. The multi-seed headline (resolves the single-seed h=64 win as robust). |
| `meta_value_learn.py` | **Cut #4e — closing the reward loop (disc-4: value-shaping *caused* by reward).** `run_meta_value_learn(cfg)`: 4d's substrate (φ-conflict `z`-conditioned FM + puck/pusher force field + CEM-MPC, fixed h=64), but the per-dim loss weight `w` is chosen by an **outer loop whose only signal is control performance** (−median CEM-MPC goal dist), not hand-set from V's support. **A1** = reward LANDSCAPE over a scalar puck-weight (both regimes); **A2** = free `w∈ℝ⁸` optimized by a derivative-free CEM outer loop, `w=8·softmax(θ)` (pure allocation). Findings (3 seeds, capacity-gated): reward **prefers** the puck-drop (A1) and the loop's converged allocation **rediscovers V's support** (A2, corr→support +0.39±0.04) — only under capacity competition. Two gotchas: normalize `w` (fixed budget) or the loop games loss-scale not allocation (wireheading demo, `outer_normalize=False`/`outer_pfield` tag); read the CEM converged mean, not the selection-biased single-best. `--field-pusher-amp 2.0` = capacity competition, `--outer` runs A2. Entrypoint `meta_value_learn(...)`. See [README.md](README.md) Cut #4e. |
| `meta_value_learn_seeds_figure.py` | Local post-processing: aggregates the 3 competition + 3 easy `meta_value_learn` seeds into `figures/meta_value_learn_seeds/fig_seeds.png` — the two order parameters (A1 reward-prefers-puck-drop; A2 converged-allocation corr→support) as mean±sem + per-seed scatter, competition vs easy. Reads the **converged CEM distribution** for A2 (not the selection-biased single-best). The multi-seed headline for Cut #4e. |
| `meta_adapt_sweep_figure.py` | Local (non-Modal) post-processing: stitches the `meta_adapt` floor (`full_v2`) + actuator-conflict runs (`actuator_p16/p2/pi`) `results.json` mirrors into the **gap-vs-conflict** order-parameter figure (`figures/meta_adapt_sweep/fig_gap_vs_conflict.png`). |
| `__init__.py` | Package marker (empty). |

## Figures (`figures/<tag>/`, mirrored from the volume)

### Cut #4 — Reptile meta-init (`meta_adapt.py`)

Per run: `fig1_adaptation_curves` (held-out R² vs N, 4 inits — the MAML crossover) · `fig2_transitions_to_adapt` · **`fig3_meta_vs_multitask_gap`** (the order parameter) · `fig4_planning`.

| Directory | Content |
|---|---|
| `meta_adapt_full_v2/` | **The FLOOR** — 1-D `joint_damping` family, interior held-out dampings; meta−multitask gap ≈ 0 (the RHM collapse anchored on control). The reported floor run. |
| `meta_adapt_full_v1/` | Earlier damping-floor run, superseded by `full_v2`; kept for provenance. |
| `meta_adapt_actuator_p16/`, `_p2/`, `_pi/` | The **conflict sweep** — `push_rot` family at Φ = π/6, π/2, π. The gap opens monotonically (+0.006 → +0.156 → +0.369). |
| `meta_adapt_sweep/` | **`fig_gap_vs_conflict`** — the floor + three conflict runs stitched into the gap-vs-Φ order parameter (from `meta_adapt_sweep_figure.py`). |
| `meta_adapt_smoke/` | `--quick` smoke output; debugging, not a result. |

### Cut #4b — context latent (`meta_context.py`)

Per run: `fig1_context_curve` · `fig2_context_gap` · **`fig3_z_probe`** (z→param system-ID scatter).

| Directory | Content |
|---|---|
| `meta_context_ctx_p2/` | Conflict Φ=π/2 — context recovers to near-oracle in 5 transitions; `z`→φ probe R²=0.998. |
| `meta_context_ctx_floor/` | The damping **floor control** — probe R²=0.987 but context−z0 gap ≈ 0. The system-ID ⊥ adaptation-benefit dissociation. |
| `meta_context_smoke/` | `--quick` smoke output. |

### Cut #4c — value-directed identification (`meta_active.py`)

Per run: `fig1_identification_curve` (in-patch R², 4 arms) · `fig2_phi_error`.

| Directory | Content |
|---|---|
| `meta_active_act_global/` | The **regime gate** — global rotation is info-rich (passive IDs φ in ~1 transition) → all arms null. |
| `meta_active_act_patch/` | First `rot_patch` (localized-rotation) run creating scarcity; 3-arm, single seed. |
| `meta_active_act_patch4{,_s1,_s2,_s3}/` | The 4-arm × 4-seed scarcity runs (seed 0 = `act_patch4`) — the multi-seed negative. |
| `meta_active_seeds/` | **`fig_seeds`** — cross-seed aggregate: per-arm ID quality + the patch-visitation ladder (the mechanism works, identification is a wash). The headline. |
| `meta_active_smoke/` | `--quick` smoke output. |

### Cut #4d — value-shaping × meta-conditioning (`meta_value_shaping.py`)

Per run: `fig1_capacity_frontier` (puck-drop + value-relevant frontier) · `fig2_perdim_bars` · **`fig3_fusion_2x2`** (pooling collapses for both objectives) · `fig4_z_probe` (system-ID) · **`fig5_planning`** (goal-dist vs capacity + veridicality-vs-plannability scatter).

| Directory | Content |
|---|---|
| `meta_value_shaping_full_v1/` | **Stage 1** — FM-side only, easy pusher: puck-drop is value-caused; capacity benefit muted; fusion clean. |
| `meta_value_shaping_plan_v1/` | **Stage 2, easy pusher — the matched NULL.** Planning is control-neutral (veridical ≈ value-shaped at every capacity). |
| `meta_value_shaping_plan_pfield_v1/`, `_s1/`, `_s2/`, `_s3/` | The **capacity-competition test** (`--field-pusher-amp 2.0`, hard pusher), 4 seeds (`plan_pfield_v1` = seed 0) — the positive. |
| `meta_value_shaping_seeds/` | **`fig_seeds`** — 4-seed aggregate: planning order parameter (+0.036±0.012 at h=64) + FM-side frontier. The headline. |
| `meta_value_shaping_smoke/` | `--quick --plan` smoke output. |

### Cut #4e — closing the reward loop (`meta_value_learn.py`)

Per run: `fig1_reward_landscape` (A1) · `fig2_outer_loop` (A2 weight discovery).

| Directory | Content |
|---|---|
| `meta_value_learn_outer_pfield/` | The **unnormalized** outer loop — improves control with corr→support ≈ 0. Retained as the **scale-confound / wireheading demo**, not a result. |
| `meta_value_learn_outer_pfield_norm{,_s1,_s2}/` | Capacity competition (`--field-pusher-amp 2.0`), fixed-budget `8·softmax` weight, 3 seeds — both order parameters fire (A1 +0.035±0.007, A2 +0.391±0.044). |
| `meta_value_learn_outer_easy_norm{,_s1,_s2}/` | The matched **easy-pusher null**, 3 seeds — the loop wanders (no control gradient to chase). |
| `meta_value_learn_land_easy/` | A1 reward-landscape scan on the easy regime (puck-weight sweep without the outer loop). |
| `meta_value_learn_seeds/` | **`fig_seeds`** — 3+3-seed aggregate of both order parameters, competition vs easy. The headline for #4e. |
| `meta_value_learn_smoke/` | `--quick --outer` smoke output. |

## Run logs (`logs/`)

`meta_value_shaping_<tag>.log` — stdout of the six detached Cut-#4d runs (`full_v1`, `plan_v1`,
`plan_pfield_v1`, `plan_pfield_s1/_s2/_s3`). Kept because those runs exceed the ~2-min client
window and were launched with `--detach`; the logs are the client-side record of a run whose only
other trace is the volume commit.

## Modal volume layout (`mujoco-control-data`)

```
/data/meta_adapt/<tag>/           results.json, fig1..fig4 .png            (Cut #4)
/data/meta_context/<tag>/         results.json, fig1..fig3 .png            (Cut #4b)
/data/meta_active/<tag>/          results.json, fig1..fig2 .png            (Cut #4c)
/data/meta_value_shaping/<tag>/   results.json, fig1..fig5 .png            (Cut #4d)
/data/meta_value_learn/<tag>/     results.json, fig1..fig2 .png            (Cut #4e)
```

Local mirrors land in `figures/<script>_<tag>/`. The cross-seed aggregates
(`figures/meta_adapt_sweep/`, `meta_active_seeds/`, `meta_value_shaping_seeds/`,
`meta_value_learn_seeds/`) are produced **locally** by the `*_figure.py` scripts and have no
volume counterpart.
