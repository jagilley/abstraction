# Value-shaping as capacity re-allocation — the value→FM interface (learning-layer, stationary)

**Idea doc**: [ideas/two_timescale_value_loop.md](../../ideas/two_timescale_value_loop.md) (discriminator 4 = "does a value/outer loop *cause* the FM to become value-shaped"; the "efferent gain: value-shaping = capacity re-allocation" section)
**Spec it came from**: [../a2a_forward/reaching/CURIOSITY_CONTROL_README.md](../a2a_forward/reaching/CURIOSITY_CONTROL_README.md) (the MuJoCo redirection — bare-state disc-4 on the pusher, puck = value-irrelevant distractor)
**Cousin / the meta sequel**: [Cut #3](README.md#cut-3--operator-intervention-reward-free-re-adaptation-after-a-dynamics-shift) `dynamics_shift.py` — reward-free re-adaptation, the substrate for the non-stationary (meta) experiments this one is a control for.
**Status**: single seed, **stationary**. Robust core (re-allocation) + a fragile bonus (the "teeth"). **Date**: 2026-07-18.

---

## Where this sits (read this first — it's the whole point)

The program's goal is the **interplay between meta-learning and FM-based value supervision**. The idea doc splits that into two layers:

- **Learning layer** — *how a value signal shapes the FM*. Achievable **stationary**, no outer loop.
- **Meta layer** — the slow reward-driven **outer loop**; earns its keep only under **non-stationarity** (a stationary outer loop provably collapses to multitask — [RHM_META_LEARNING](../rhm/ratchet/RHM_META_LEARNING_README.md)).

**This experiment is the learning layer**: it characterizes the value→FM interface (the "efferent gain" — value re-prioritizing which state directions the FM spends capacity on) **in isolation, on a stationary task**. It is *not* the meta result and does not try to be. It is the **control** that makes the meta claim sharp: value-shaping is available stationary, so anything the outer loop buys *beyond* it (faster re-adaptation, compounding) is the meta contribution. Those live in the non-stationary sequel on `dynamics_shift.py`.

## The claim

An FM has limited capacity. A value signal (here: "reach a goal with the **pusher**") should make it **re-allocate** that capacity toward value-relevant dynamics and away from value-irrelevant ones — a worse literal simulator, aligned with what you want. On the pusher, the **puck** is a causally-coupled but **value-irrelevant** distractor: its own motion never enters the goal. So a value-shaped FM should drop the puck's own-motion prediction while keeping the pusher's.

**Value-relevant** = pusher dims `[0,1,4,5]` (incl. how a contact deflects the pusher — that *is* value-relevant). **Value-irrelevant** = the **puck's own** post-contact state `[2,3,6,7]`. The FM still *reads* puck state (input) but a shaped FM need not *predict* the puck's next velocity (output) — re-allocation over prediction **targets**.

Arms (identical data + FM class; only the per-dim loss weight `λ_puck` differs): **unshaped** (`λ=1`, Huber on all 8 dims) vs **value-shaped** (`λ=0`, pusher-only). The causal ablation is built in: `λ=1` *is* the unshaped arm.

## The substrate — and why it took work (the reusable lesson)

The naïve design (a heavy, frequently-hit puck) **failed**, and the failures are the useful part:

1. **Energy ≠ prediction cost.** A high-energy puck whose free-flight is a *linear* damped particle is **cheap to predict at any capacity** — so dropping a reducible distractor frees nothing, and there is no re-allocation to see.
2. **The capacity-hungry part of the physics (contact impulse) is *irreducible*** (Cut #1's sub-timestep aliasing) — it can't be "dropped" because no capacity was ever spent fitting it.
3. **Fix: a deterministic nonlinear *multi-mode force field*** on the puck (`pusher_env.py`, applied via `qfrc_applied` at runtime — **additive, XML byte-identical when off, so Cuts #1–3 stay reproducible**). Multi-mode **decouples** the two properties a single mode conflates through its wavenumber: *smooth* (bounded frequency ⇒ fine-step-reducible, a big FM fits it to high R²) yet *complex* (a sum of modes ⇒ a small FM cannot — genuine capacity hunger). `#modes` is the capacity-hunger lever; max-`k` is the aliasing lever.
4. **Both bodies must be capacity-hungry.** The pusher's dynamics are **command-dominated** (the actuator does most of the work), so with the field on the puck only, the easy pusher never competes for capacity. Putting a *distinct-phase* field on the pusher too makes the value-relevant task capacity-hungry, so freed capacity can actually buy pusher fidelity.

## Results (full run, `full_v1`, 70K transitions, single seed)

**Robust — the re-allocation is real:**
- The unshaped FM spends genuine, **capacity-hungry** effort on the value-irrelevant puck field: puck-slide R² climbs `0.45 (h8) → 0.74 (h32)` — it needs h≥16–32 to fit, so it is **not free**. It is **reducible** (ceiling 0.74) against a clean **irreducible foil**: the contact impulse sits at R² ≈ 0 at every capacity (the noisy-TV control, physically present).
- Value-shaping drops it: puck-slide → negative (unconstrained) while the pusher is preserved (~0.95).
- **λ-frontier** (h=16), monotone: as `λ_puck` 1→0, puck-slide `0.71 → −1.99` and pusher `0.93 → 0.95`. The re-allocation tradeoff.

**Fragile — the "teeth" (capacity efficiency) wash out with data:**
- Under data scarcity (smoke, 21K) dropping the puck *bought* pusher fidelity at the binding capacity (h=8: unshaped 0.835 → shaped **0.937**). With abundant data (70K) it vanished (0.933 → 0.934).
- Diagnosis: the pusher is command-dominated, so the field is only a weak perturbation on it; with enough data even a small FM fits the pusher while also modeling the puck. The value-relevant task never truly *starves*, so freed capacity has nothing to buy. **The teeth are a data-scarcity effect, not a robust capacity-pressure one — reported as such, not as a headline.**

## What this establishes / caveats

- **Establishes** (learning layer, stationary): the value→FM interface exists and re-allocates capacity away from a costly, reducible, value-irrelevant factor — on real continuous physics, with a physically-present irreducible foil. This is the *mechanism*, characterized in isolation.
- **Does not establish**: any *meta* result. It is single-loop and stationary by construction. The "value **causes** the shaping (not hand-λ)" upgrade (a planner-co-trained arm) and the capacity-efficiency teeth were **deprioritized** in favor of the non-stationary meta pivot, because value-shaping is learning-layer and the goal lives one rung up.
- **Caveats**: single seed; the teeth are fragile (above); `λ=0` hand-zeros the puck loss (the caused-by-value arm that would remove this tautology is not built here).

## Reproduce

```bash
cd experiments/
modal run mujoco_control/value_shaping.py::value_shaping --quick            # smoke (~1-2 min)
modal run mujoco_control/value_shaping.py::value_shaping --tag full_v1       # full (~8-10 min, CPU)
```

Knobs are auto-exposed CLI flags (`--field-amp`, `--field-pusher-amp`, `--field-central`, `--frame-skip`, `--caps` via code, `--seek-gain`, …). Results + 3 figures commit to the `mujoco-control-data` volume under `/data/value_shaping/<tag>/` and mirror to `figures/valshape_<tag>/`.

## Figures (`figures/valshape_full_v1/`)
`fig1_capacity_frontier` (pusher-vel + puck-slide R² vs capacity, unshaped vs shaped) · `fig2_lambda_frontier` (re-allocation tradeoff as λ_puck→0) · `fig3_perdim` (per-group fidelity, unshaped vs shaped).
