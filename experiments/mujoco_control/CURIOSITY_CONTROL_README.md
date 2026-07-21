# Curiosity on control: an intrinsic drive on a drifting task, and grounding it against the noisy TV

**Idea doc**: [ideas/two_timescale_value_loop.md](../../ideas/two_timescale_value_loop.md) — the **explore/exploit interface** ("two afferent taps + one efferent gain"); this cut lands **next-piece #1** ("the `e`-tap and the `p`-tap are one system") on control.
**Parents / lineage**: [a2a_forward/reaching/CURIOSITY_DRIVE_README.md](../a2a_forward/reaching/CURIOSITY_DRIVE_README.md) (the drive *atom* on active vision — Phases 1/2/2b) → [a2a_forward/reaching/CURIOSITY_CONTROL_README.md](../a2a_forward/reaching/CURIOSITY_CONTROL_README.md) (drive-selection **Step 0** + the reaching-ViT substrate dead-end, which pointed the build here — *this is the MuJoCo realization it flagged; it does not subsume that doc*). Substrate = **Cut #3** [`dynamics_shift.py`](README.md#cut-3--operator-intervention-reward-free-re-adaptation-after-a-dynamics-shift) (reward-free re-adaptation; the FM that provably learns *and re-learns*). Single-shift precursors it extends: [DIRECTED_READAPT](DIRECTED_READAPT_README.md) (disagreement under-visits a confident-prior patch) and [META_ADAPT §#4c](META_ADAPT_README.md#cut-4c--value-in-the-loop-value-directed-identification-a-robust-negative-meta_activepy) (VoI ≈ max-‖u‖ on low-dim ID). The *efferent* half of the interface (value-**shaping** the FM) is [META_ADAPT §#4d/#4e](META_ADAPT_README.md#cut-4d--value-shaping--meta-conditioning-two-separable-capacity-levers-and-when-re-allocation-buys-control-meta_value_shapingpy); this cut is the *afferent* half (value **directing where to look**).
**Code**: `curiosity_control.py` · **Status**: done; mechanism metrics seed-robust (2–3 seeds/regime), control payoff modest & seed-variable. Single family. **Date**: 2026-07-20.

---

## The high-level story

The two-timescale doc splits the value system into an **explore tap** (`e`: "where am I still failing to predict, *learnably*?" → go learn there) and an **exploit tap** (`p`: "of what I can foresee, which is best?" → act toward it). Cuts #4d/#4e landed the *efferent* gain (a value objective re-allocating FM capacity). This cut is the *afferent* explore drive **on a control task**, and it does two things:

1. **An intrinsic, learning-progress-style reward directing collection on a *drifting* task earns its keep specifically on the moving frontier.** A control agent whose forward model is continually re-fit must decide *where to collect* as a scarce, reducible dynamics frontier drifts. An intrinsic reducible-surprise drive tracks the moving frontier and sustains control, where undirected/reactive collection lags — and this advantage is a **non-stationarity phenomenon**: it collapses to a tie on a *static* frontier (everyone masters it — the plateau, on control). *(This is the "value compounds only on a moving frontier" prediction. We show the drive **re-engages and sustains competence under drift and ties when static**; we did not isolate strict **compounding** = re-adaptation *accelerating* over successive shifts — see caveats.)*

2. **Pure curiosity falls for the noisy TV; grounding it with the exploit/value signal — as two additive drives — cures it.** An ungrounded intrinsic drive chases irreducible aleatoric noise (novelty-seeking off the rails). Adding a value-relevance term (the `p`-tap) as a second additive drive — *not* a hard gate — breaks the tie between the value-relevant reducible frontier and value-irrelevant noise, immunizes the drive against the noisy TV, and is **optimal at an intermediate explore/exploit balance**. This is the explore-exploit framing of meta-learning made behavioral, and it matches the human phenomenology (novelty-seeking needs task-grounding).

## Apparatus (`curiosity_control.py`)

- **Substrate**: Cut #3's puck-free momentum reaching (`frame_skip=12`, gear 10, damping 2, arena 1.8 — the known-working controller). State `[px,py,pvx,pvy]`, command `u∈[-1,1]²`.
- **The moving frontier**: a localized, drifting **`field_patch`** (a smooth multi-mode force = *reducible* but capacity-hungry; added to `pusher_env.py`, additive/off-by-default → Cuts #1–4e byte-identical). Its Gaussian `sigma` makes it **scarce** (the Phase-2b precondition); `drift_mode=morph` sweeps its center back-and-forth (gradual — the Phase-2 lesson that derivative drives degrade gracefully under gradual drift); `drift_mode=none` pins it = the **stationary null**. An optional aleatoric **`noise_patch`** (Gaussian-gated random force) is the **noisy-TV** decoy.
- **Inner loop**: an **RPF ensemble** (K random-prior forward models; Osband 2018 — a frozen random prior per member so they disagree off-data, the intended fix for `directed_readapt`'s confident-prior blindness) continually re-fit on a **recency FIFO buffer**. The ensemble mean plans.
- **Collection = teleport-region allocation** (isolates the drive from navigation — the confound that muddied #4c). Each round the drive scores a G×G grid of workspace cells, softmax-samples one, and collects a short in-cell rollout. **All arms share the FM/init/update; only the collection differs** (the `directed_readapt` discipline).
- **The drives**: `surprise` (raw ‖e‖), `disagree` (ensemble variance), **`reducible`** = error×disagreement ("reducible surprise"), `lp` (the `−d‖e‖/dt` derivative), `taskonly` (on-policy toward goals), `random`; and **`grounded@b`** = `b·reducible + (1−b)·exploit` where **exploit** = value-relevance (density of the control task's start→goal paths through each cell).
- **Readouts**: CEM-MPC control competence; the unbiased **frontier-tracking error** (is the FM current *where the frontier now is*); occupancy (did the drive find/track the moving patch, and does it get seduced by the noise). `frontier_task` (default) makes goals **straddle** the current frontier so the crossing *is* the task — the frontier dominates control (undiluting the payoff), while the drive's problem stays scarce.

## Finding 1 — the drive tracks the moving frontier; the benefit is a non-stationarity effect

Order parameter = **`reducible − random` control gap** and the **frontier-tracking gap** (both teleport-based → unconfounded), across regimes:

| regime | `reducible − random` control gap (per seed) | frontier-tracking (reducible vs random err) |
|---|---|---|
| **stationary** (fixed patch) | −0.0001, −0.0006, −0.0005 → **≈ 0** | both ~0.001 (all master it) |
| **drift** (moving frontier, undiluted) | +0.016, +0.019, +0.000 → **+ (2/3 clear)** | reducible **0.022–0.025** vs random **0.034–0.049** |

The **tracking advantage is robust every seed** (reducible tracks the drifting frontier ~0.012–0.024 better than random) and is **specifically non-stationary** — it vanishes on a static frontier (the plateau anchor, on control). `taskonly` (reactive, on-policy) is the *worst* tracker under drift (lags the drift), and `lp` (the derivative) tracks weakly (the Phase-2 blindness, reconfirmed on control). The **control** payoff is positive in expectation but **seed-variable** — it tracks the *tracking* gap, which shrinks when the undirected baseline happens to cover the frontier well (seed 2). `reducible ≈ surprise` throughout (see Finding 3).

## Finding 2 — value-relevance dissociation (tracks regardless; only pays on-path)

Drifting the frontier **off** the goal corridor (`--value-rel off`): the drive **still tracks it best** (frontier-err 0.021/0.034 vs random 0.063/0.097) but the **control gap collapses to ≈ 0** (−0.0008, 0.000) — concentrating on a value-*irrelevant* frontier is **true-but-useless**. (Bonus: `surprise`/`lp` are slightly *worse* off-path — they waste budget on the useless frontier.) This is the #4b **system-ID ⊥ value** dissociation, now for the *explore* drive: tracking is always available; whether it helps control is set by value-relevance.

## Finding 3 — pure curiosity's noisy TV (the honest negative)

With an aleatoric `noise_patch` present, the **reducibility filter fails on this substrate**: `reducible`/`disagree` **chase** the noise (noise-frac **0.10–0.20**, ~7× uniform) and, worse, are **completely derailed from the frontier** (patch-frac **0.00 on all 3 noise seeds**), tracking wrecked (0.023–0.042). The mechanism: with scarce, low-dim, bootstrapped noise data the ensemble members fit *different* noise realizations and **disagree** rather than converging to the mean — the opposite of active vision (Phase 2b), where high-dim random pixels drive the members *to* the mean and the ensemble rejects noise by construction. So **`reducible ≈ surprise`** here (the disagreement factor adds nothing without noise and *back*fires with it): a smarter *filter* is not the fix.

## Finding 4 — grounding resolves it: explore + exploit as two additive drives

The fix is **task-grounding**, not a better filter. Add the exploit/value-relevance term as a **second additive drive** (`grounded@b = b·explore + (1−b)·exploit`) and sweep the balance `b`, noise **on** vs **off** (3 / 3 seeds):

| | pure explore (`b=1`) | **grounded (`b≈0.3–0.5`)** |
|---|---|---|
| noise-frac (noise on) | 0.10–0.20 (seduced) | **0.00** (immune) |
| finds frontier (patch-frac, noise on) | **0.00** (derailed) | 0.30–0.40 |
| frontier-err, noise **off → on** | 0.025 → **0.042** (noise damages it) | 0.012 → **0.010** (unchanged) |
| control vs `grounded@0.5` | — | better every seed (noise-on +0.012/+0.007/~0; noise-off +0.007/+0.022/+0.011) |

Three things, all robust on the (stark) mechanism metrics: (1) **grounding immunizes** — the value term zeroes out the off-corridor noise, so the drive *cannot* be seduced by it (tracking unchanged noise on↔off); (2) noise **specifically damages the ungrounded drive** — it's the 2×2 boundary (grounding helps *most* under noise), the explore-drive analog of #4d's capacity-competition boundary; (3) the optimum is at an **intermediate balance** (~0.3–0.5) — both with and without noise — so **both terms pull weight** (explore sharpens frontier-tracking, exploit grounds), a mild inverted-U. Every directed/grounded drive beats `random`.

## What this establishes / caveats

- **Establishes**: (1) an intrinsic reducible-surprise drive directs control-relevant collection to a **moving** frontier and sustains control there, and the benefit is **specifically non-stationary** (ties on a static frontier); (2) **value-relevance gates usefulness** (tracks regardless, pays only on-path); (3) pure curiosity has a **noisy-TV pathology on low-dim control** (disagreement chases aleatoric noise — a substrate-dependent inversion of the active-vision result); (4) **grounding with the exploit/value signal, as two additive drives, cures it**, immunizing against the noisy TV with an intermediate optimum — the `e`-tap + `p`-tap wired as one system (interface next-piece #1).
- **Caveats / does-not-show**:
  - **Control effect is modest** (all directed arms in a narrow band; the *mechanism* metrics — noise-frac, patch-frac, frontier-err — carry the story far more cleanly than control does) and the drift control payoff is **seed-variable**.
  - **Strict compounding not isolated.** We show *engages-on-moving / ties-on-static*, not re-adaptation *accelerating* over successive shifts (the meta-layer signature). That remains open.
  - **`taskonly` is confounded** by on-policy coverage (worse even stationary) — not a clean "extrinsic-is-enough" falsifier; the clean contrast is `reducible` vs `random`.
  - **`frontier_task` makes the task point at the frontier**, so pure-`exploit` is a strong baseline and the explore term's *distinctive* value (finding a frontier the task does **not** point at — the anticipation/dark-room geometry) is **not isolated** here.
  - **Self-tuning deferred.** The balance is swept, not learned. The balance sweep is the cheap stand-in for a reward-driven outer loop over `b` (a `meta_value_learn`-style A2 — the natural next cut).
  - Single family (drifting multi-mode field); noisy-TV test is one design.

## Reproduce

```bash
cd experiments/
modal run mujoco_control/curiosity_control.py::curiosity_control --quick                                  # smoke
# Finding 1 (drift, undiluted) + its stationary null (3 seeds each):
for s in 0 1 2; do
  modal run --detach mujoco_control/curiosity_control.py::curiosity_control --tag ft_drift_s$s --seed $s --n-eval 36
  modal run --detach mujoco_control/curiosity_control.py::curiosity_control --tag ft_stat_s$s  --seed $s --n-eval 36 --drift-mode none
done
# Finding 2 (value-relevance null): --value-rel off   (diluted-task runs drift_v1/offpath_v1 also show it)
# Finding 4 (grounding balance sweep, noise on vs off; 3 seeds each):
for s in 0 1 2; do
  modal run --detach mujoco_control/curiosity_control.py::curiosity_control --tag ground_noise_s$s \
      --arms "reducible,grounded@0.7,grounded@0.5,grounded@0.3,grounded@0.0,random" --noise --n-eval 36 --seed $s
  modal run --detach mujoco_control/curiosity_control.py::curiosity_control --tag ground_clean_s$s \
      --arms "reducible,grounded@0.7,grounded@0.5,grounded@0.3,grounded@0.0,random" --n-eval 36 --seed $s
done
```
Use `--detach` (runs exceed the ~2-min client window; the remote commits `results.json` + figures to the volume regardless). Results at `/data/curiosity_control/<tag>/` mirrored to `figures/curiosity_control_<tag>/`. Key knobs: `--drift-mode morph|none`, `--value-rel on|off`, `--noise`, `--arms` (incl. `grounded@<b>`), `--balance`, `--frontier-task`.

## Figures
Per run (`figures/curiosity_control_<tag>/`): `fig1_curves` (control + frontier-tracking vs round, per arm) · **`fig2_occupancy`** (collection heatmaps with the frontier drift path + noise-patch marker — the clearest view of tracking vs noise-seduction) · `fig3_orderparams` (final control / frontier-err / where-they-collect bars).

## Next
- **Self-tune the balance**: a reward-driven outer loop over `b` (à la #4e's A2) — does the system *discover* the grounded optimum?
- **The anticipation geometry**: a frontier that will *become* value-relevant but isn't yet (fixed goals, drift through them), where the explore term is *strictly* necessary (pure-exploit can't find it) — isolates the `e`-tap's distinctive value and would give a true inverted-U.
- **Strict compounding**: successive related shifts; does per-shift re-adaptation *accelerate* (the meta-layer signature)?
- Back-translate into [ideas/two_timescale_value_loop.md](../../ideas/two_timescale_value_loop.md) (interface next-piece #1) and the belief tree.
