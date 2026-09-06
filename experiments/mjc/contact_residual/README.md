# Cut #1 — Contact-residual structure: the residual is an *event* detector, not a state detector

**Up**: [../README.md](../README.md) (mjc) · **Idea doc**: [../../../ideas/physical_control_substrate.md](../../../ideas/physical_control_substrate.md)
**Cousins**: the a2a residual story ([REACHING_INTERNAL](../../a2a_forward/reaching/REACHING_INTERNAL_README.md)), [RHM sculpting](../../rhm/RHM_SCULPTING_README.md)
**Code**: `contact_residual.py` (env: [`../pusher_env.py`](../pusher_env.py)) · File index: [FILES.md](FILES.md)
**Status**: done. **Date**: 2026-07-16.
**Builds on this**: [`arity_torque/`](../arity_torque/README.md) (Cut #2 evaluates arity on *free-flight* transitions precisely because contact is this cut's regime)

---

**Claim** (ported from the a2a residual story to physics): an arity-2 forward model `f(s,u) → Δs` of a contact-rich system has a residual (actual − predicted) that **concentrates at contact events**, because free flight is easy/near-linear and contact is stiff/near-discontinuous. No policy, no RL: a scripted OU + seek-the-puck behavior policy generates `(s,u,s')`; MuJoCo hands us the contact labels.

**Apparatus**: 125K transitions (500 eps × 250 steps); train/test split by episode (no leakage). A small MLP FM `10→256×3→8` predicts normalized `Δs`, trained with **Huber loss** (δ=1.0). Regime label = **any contact** (`ncon>0`, incl. walls) vs genuine free flight.

## Result (17.8% contact in test)

| axis | metric | finding |
|---|---|---|
| **concentration** | residual-norm ratio contact/free | **8.1×** (AUC **0.86** = P(contact resid > free resid); Cohen's d 0.72) |
| **scale-free control** | FM cosine(Δŝ, Δs) | **0.988 free vs 0.819 contact** — the FM genuinely models free-flight *direction*; contact is directionally worse, so the effect is **not** just "bigger Δs" |
| **dose-response** | Spearman(peak contact force, residual) | **ρ=0.75** within contact (over 6 orders of magnitude of force) |
| **structure** | per-dim residual | residual lives almost entirely in the **velocity dims** (positions barely move at contact) — the physics analog of MNIST's digit-discriminative residual |

## The sharpened claim: the residual is an *event* detector, not a *state* detector (`fig5`)

An onset-aligned event-triggered average (573 impacts) shows the residual **spikes at the free→contact impact** (peak 2.14, **5.5×** the pre-onset approach) and **decays back toward baseline within ~4 steps** (**6.8× peak/sustained**) — *even though* `P(in contact)` stays ~40–50% out to +20 steps. So being in contact does not keep the residual high; the residual marks the **regime transition**. Confirmed by the **separation-aligned** average: at contact *release* the residual just steps down with **no spike** — the asymmetry isolates the mechanism (a stiff, near-discontinuous velocity *impulse* at impact; release is gentle).

**Why this is expected** (three co-located mechanisms, all peaking at the boundary): (1) `(s,u)→Δs` is **near-discontinuous** at the contact boundary and a smooth MLP cannot represent a step; (2) **sub-timestep collision-timing aliasing** — whether the collision lands at substep 1 vs 5 within the 0.01 s interval is below the sampling resolution but changes the integrated Δv; (3) **stiff impact forces** (huge local `∂Δs/∂s`). Sustained pushing (bodies moving together, quasi-static) and free flight are both smooth regime *interiors* the FM fits well — the spike is the switching surface.

## Load-bearing methodological decisions

- **Huber loss, not MSE** (`huber_delta`, strict MSE generalization). Under plain MSE, raising the contact fraction *collapses* the contrast: contact's heavy-tailed Δs dominates the loss and starves free-flight learning (free cosine drops 0.99→0.93 at 22% contact). Huber caps the outlier gradient so the FM keeps modeling the **predictable** dynamics and contact becomes the residual — exactly the "FM predicts the expected trajectory; residual = surprise" framing. (At 10% contact, plain MSE is already clean — the tension only bites as contact fraction rises.)
- **Regime label = any contact, not puck-only.** Labeling only pusher↔puck contacts mislabels pusher/puck↔wall contacts as "free" and pollutes the baseline (free R² collapsed until this was fixed). `puck_force` is kept separately for the dose-response.
- **The scale-free control is cosine, not R² or ‖r‖/‖Δs‖.** Both magnitude-normalizers are ill-conditioned here: per-regime R² goes *negative* on free flight (free Δs sits below the FM's global error floor), and ‖r‖/‖Δs‖ blows up as ‖Δs‖→0. Cosine is scale-invariant by construction and cleanly shows contact predictions are directionally worse. (R² is still logged, with this caveat, in `results.json`.)

## Caveats

- **eff-rank does NOT support "contact residual is lower-rank."** Contact eff-rank (3.60) is *higher* than free (2.52) — the free residual is near-degenerate (rank ~1, a small systematic error), while contact spreads across the 4 velocity dims. The honest structure story is the per-dim figure (velocity-localized), not a rank comparison. (We'd already retired "residual rank ∝ DGP complexity" as too many-variabled anyway.)

## Figures (`figures/full_v1/`)

`fig1_trajectory` (residual spikes at contact onset over one episode) · `fig2_distributions` (residual norm + cosine, contact vs free) · `fig3_perdim` (velocity-dim localization) · `fig4_doseresponse` (residual vs contact force, log-x binned) · **`fig5_onset_eta`** (the onset/separation event-triggered averages — the cleanest single figure).

## Reproduce

```bash
cd experiments/
# smoke test (~1 min on Modal; builds the mujoco+torch image on first run)
modal run mjc/contact_residual/contact_residual.py::contact_residual --quick
# full run (~90 s): 125K transitions, 60 epochs
modal run mjc/contact_residual/contact_residual.py::contact_residual --tag full_v1
```

Knobs (`--seek-gain`, `--frame-skip`, `--huber-delta`, `--hidden`, `--epochs`, …) are auto-exposed as CLI flags. The Modal function commits `results.json` + figures to `/data/contact_residual/<tag>/` on the `mujoco-control-data` volume; the local entrypoint mirrors them to `figures/<tag>/`.

## Mechanism follow-up (open)

A **frame-skip sweep**: the onset residual spike should *grow* with coarser control intervals (aliasing) while free-flight residual stays flat — separating the aliasing cause (#2) from the stiffness cause (#3).
