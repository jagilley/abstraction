# The decomposition over training: the arc survives, the exponent does not certify (2026-08-02)

**Parent:** [`residual_decomposition/README.md`](../README.md) · **Grandparent:** [`rhm/README.md`](../../README.md)
**Code:** `rhm_decomposition_trajectory.py` (Modal, the re-cut), `analyze_trajectory.py` (CPU, post-hoc)
**Re-measures:** [`RHM_COMPLEXODYNAMICS_README.md`](../../RHM_COMPLEXODYNAMICS_README.md) (the rise-and-fall arc)
**Instrument under test:** [`decomposition.py`](../decomposition.py), β from [`analyze_summaries.py`](../analyze_summaries.py)
**Belief touched:** [`dimensionality_expansion.md`](../../../../beliefs/dimensionality_expansion.md)

## One-line arc

Re-measuring the complexodynamics training trajectory with the instrument that replaced residual rank: the **rise-then-fall-under-pressure vs rise-then-arrest-under-control dissociation reproduces in 7/7 measurements** across two prediction gaps and four FM capacities, and two new directional facts hold in every cell (**β rises monotonically**, **`R_res_participation` falls monotonically**) — but **β itself fails its own capacity-invariance check at every checkpoint where it is measuring anything**, so the account can rest on trajectory directions and arm contrasts and not on any absolute number from this instrument.

## Why this exists

This is the third axis for this sub-experiment. [`rhm_decomposition_audit.py`](../rhm_decomposition_audit.py) measures the decomposition statically (setting × capacity × gap); [`rhm_decomposition_ratchet.py`](../rhm_decomposition_ratchet.py) measures it over wake-sleep cycles; this measures it over the **base model's own training checkpoints**.

That axis mattered because the trajectory claims were orphaned. [`RHM_COMPLEXODYNAMICS_README`](../../RHM_COMPLEXODYNAMICS_README.md)'s rise-and-fall was measured with FM-free activation effective rank, naive residual rank, and top1-PC — and the 2026-07-27 audit ([parent README](../README.md) §3) then showed the single-number rank instrument conflates the frontier's shape with its size and inflates toward `d_model` wherever the FM saturates. So the load-bearing trajectory numbers came from an instrument this sub-experiment had already retired, and nobody had run the replacement along a trajectory.

## Design

**The paired-measurement control.** The saved trajectory parts only ever stored summary scalars, and β needs the raw `(A, P)` variance spectra — so this cannot re-aggregate, it refits. Fresh FMs are trained on the **saved base checkpoints** (same substrate, same arms, same FM training distribution as `rhm_ensemble_trajectory.ensemble_ckpt`: 8k steps, AdamW lr 1e-3 wd 0.01, clip 1.0, no schedule), and the retired *and* trusted metrics are computed on the **identical `(A, P)`** at every checkpoint. Any disagreement in trajectory shape is therefore attributable to the instrument and not to run-to-run variation — the control the original trajectory work could not have had. Per-level `feature_eta2_last` is computed on the same residual, so each part is a strict superset of the old ensemble/legibility parts.

**Substrate.** m2-distinct (`v16_s2_L6_m2_distinct`, 8L/8H/256D, rule_seed=0), the complexodynamics substrate. Arms are its two arms: `fmreg:1.0` (structured compression pressure toward FM-legibility, 0→300k) and `wd:0.1` (the λ=0 control, 0→125k). `capB` is `REG_FM` verbatim, so `capB` rows stay comparable to the existing parts.

**The 2 × 2 × 4.** Two gaps × two arms × four FM capacities, 8 checkpoints each — 112 fresh-FM fits.

| axis | levels | why |
|---|---|---|
| gap | `post_embed→post_block6` (7 blocks), `post_block3→post_block6` (3 blocks) | the FM's capacity *relative to the computation it must model* — `capB` is 100% of one block either way, so it is ~14% of the wide gap and ~33% of the narrow one, and ~33% is where the audit established invariance |
| capacity (fixed depth `nl=2`) | `capB` dh16/mm2.0 (791K, 100% of a block), `capM` dh8/mm1.0, `capQ2` dh4/mm0.5 — a ~4× range | the audit's own axis: it swept 25–100% of a block at fixed depth. **Only these three may be pooled into an invariance spread.** |
| depth (separate leg) | `capQ` dh8/mm1.0 at `nl=1` vs `capM` at `nl=2` | see "What we got wrong" — retained as its own labelled axis, never pooled with the capacity spread |

**Two trajectory-specific guards.** (1) *Saturation.* At step 0 the base model computes almost nothing and a fresh FM reads cosine ~0.996, so the trajectory **starts inside** the noise regime the audit identified — honest there (nothing to model is *why* it saturates), but early and late checkpoints are not comparable without the saturation columns in view, and the FM config is held fixed across each trajectory for the same reason. (2) *Capacity invariance, pointwise.* Running the full capacity range at **every** checkpoint turns the audit's central validity claim into a per-checkpoint trustworthiness check. The analyzer computes each arc twice — over all points and over trusted points only — because a rise-and-fall present only in the untrustworthy regime is an artifact, not a finding.

The trust gate is `capacity-invariant (spread ≤ 0.01)` **and** `unsaturated`, where unsaturated requires both `relative_residual ≥ 0.02` and cosine inside the repo's established `[0.90, 0.99]` band. The two saturation criteria disagree at step 0 (relRes 0.089 looks fine at cosine 0.996), and the cosine band is what the rest of the repo gates on.

## Result 1 — the arc reproduces, and the arm dissociation is total

`frontier_mass` (fraction of activation variance unexplained), init → peak@step → final:

| gap | cap | `fmreg:1.0` (pressure) | `wd:0.1` (control) |
|---|---|---|---|
| wide E→b6 | capB | 0.016 → **0.059 @4k** → 0.045 (**fall**) | 0.016 → 0.074 (monotone-up) |
| wide | capM | 0.030 → 0.090 @4k → 0.081 (arrest) | 0.030 → 0.120 (monotone-up) |
| wide | capQ2 | 0.050 → 0.164 @160k → 0.161 (plateau) | 0.050 → 0.195 (monotone-up) |
| wide | capQ *(depth leg)* | 0.042 → 0.168 @4k → 0.165 (plateau) | 0.042 → 0.249 (monotone-up) |
| narrow b3→b6 | capB | 0.003 → **0.015 @4k** → 0.008 (**fall**) | 0.003 → 0.012 (monotone-up) |
| narrow | capM | 0.006 → 0.026 @4k → 0.015 (**fall**) | 0.006 → 0.022 (monotone-up) |
| narrow | capQ2 | 0.010 → 0.050 @12k → 0.037 (**fall**) | 0.010 → 0.041 (monotone-up) |

**The pressure arm peaks and declines in 7/7; the control arm is monotone-up and never falls in 7/7.** The peak sits at step 4k–12k throughout, matching the published activation-rank peak at 4k. Pressure also ends with a smaller frontier than control in 6/6 matched pairs. The complexodynamics claim that *the descent requires an annealer — SGD alone arrests, a glass rather than an equilibrated liquid* was one measurement on a retired scalar; it now holds across two gaps and four observer bounds on the replacement instrument.

Two qualifications. The **magnitude** of the descent scales with the observer: at the narrow gap capB −50%, capM −40%, capQ2 −25%; at the wide gap the low-capacity legs flatten to a plateau entirely. A less capable FM registers less of the fall, which is coherent with a bound-relative reading (the descent is the absorption of scaffolding, and an observer that never resolved the scaffolding cannot register its removal) but is not established by this data — it is one plausible account of a capacity trend. And absolute levels are **not** comparable to the published table: the published `act_rank` used a different convention from this module's all-token entropy `R_act`, so only shapes should be read across.

## Result 2 — two directions that hold in every cell

Robust across both arms, both gaps, and all four capacities (14 arm × gap × capacity cells: 4 capacities at the wide gap, 3 at the narrow, both arms):

- **β rises monotonically over training.** 0.076 → 0.264 (narrow, capB), 0.241 → 0.476 (wide, capB), and the same direction in every other cell. β is the exponent of `res_var(i) ∝ act_var(i)^β` across the model's principal directions: low β is a residual near-isotropic and unrelated to what the model computes; high β is a residual that is a faithful graded shadow of the whole computation.
- **`R_res_participation` falls monotonically.** 248–250 → 148–182 (narrow), 182–202 → 73–127 (wide). The frontier's dimensionality contracts by 25–60% — where naive `R_res` on the identical `(A, P)` moves only ~17%, reproducing along a trajectory the shape/level split the audit found statically.

Read together, and alongside the cosine falling monotonically the whole way (0.996 → 0.973 wide, capB): the FM's error starts **noise-shaped and spread across nearly every direction** and ends **computation-shaped and contracted onto a much smaller set**, while the host becomes steadily *harder* to predict. A randomly-initialized network is the most FM-predictable state the host ever occupies. This is the clearest arity-1 readout of "mostly noise → structured predictor" the repo has, and it is a direction the rise-and-fall metrics do not show.

Per-level `feature_eta2_last` rises alongside (wide capB, d4: 0.018 → 0.456; d6: 0.016 → 0.242), consistent with [`RHM_FRONTIER_AND_LEGIBILITY_README`](../../RHM_FRONTIER_AND_LEGIBILITY_README.md)'s traveling front.

## Result 3 — β does not certify itself here (the negative)

β capacity spread over `capB/capM/capQ2` at fixed depth, against the audit's ±0.01:

| step | wide E→b6 | narrow b3→b6 (fmreg) | FM state, narrow |
|---|---|---|---|
| 0 | 0.092 ✗ | **0.009** ✓ | cos 0.999, naive `R_res` 96% of d, β≈0.07 — **saturated** |
| 500 | 0.094 ✗ | 0.010 ✗ | cos 0.999 — **saturated** |
| 4000 | 0.174 ✗ | 0.065 ✗ | working |
| 12000–300000 | 0.127–0.169 ✗ | 0.113–0.136 ✗ | working |

**Wherever β is capacity-invariant it is measuring nothing, and wherever it is measuring something it is capacity-dependent.** The only in-tolerance points are the two saturated ones, and they read exactly the audit's own noise-floor signature (β ≈ 0.07, naive `R_res` at 96% of `d_model`). Gap width does not separate the two failures, it trades them: at the wide gap `capB` is properly in band (cosine 0.969–0.990 for steps 500+) but the spread is 0.09–0.17; at the narrow gap the spread falls to 0.009–0.062 but `capB` is out of band at all 8 checkpoints. **0/16 checkpoints across both gaps pass both trust gates.**

The spread is systematic, not noise: smaller FM → higher β, monotonically, at every checkpoint in every arm and gap.

This does not contradict the audit's measurement, which found β = 0.610 ± 0.007 (MNIST) and 0.607 ± 0.009 (RHM L6_m4) over a 4× capacity range at a 3-block gap on d=128 models. It bounds it. The audit swept capacity at fixed gap and never swept gap at fixed capacity, so the regime where invariance holds was never mapped — and this substrate (d=256, 8 layers, `nl=2` FM, both gap choices) falls outside it. What that means for the belief's unconditional "a property of *the model's computation*" phrasing is recorded in [`dimensionality_expansion.md`](../../../../beliefs/dimensionality_expansion.md).

Consequence for the instrument: `R_res_participation` and `frontier_mass` carry the whole readout here, and neither has a capacity-invariance story of its own — the audit states the *level* is capacity-dependent by design. So this node establishes robust **directions** and **contrasts**, and no trustworthy **scalar**.

## What we got wrong (recorded, not tidied)

Two diagnoses of the invariance failure, both wrong, each disconfirmed by the run designed to test it:

1. **"It's a depth confound."** The first pass used `capQ` (`nl=1`) as the low-capacity leg against `capB` (`nl=2`), varying FM depth alongside width — not the audit's axis. Re-running at fixed `nl=2` over a ~4× range barely moved the spread (0.092–0.174). The confound was real and worth fixing; it was not the cause. `capQ` is retained as an explicit depth leg, where β reads +0.15 to +0.18 above `nl=2` at matched width — so depth moves β too, but it is not what broke the tolerance.
2. **"It's gap width."** The narrow gap was predicted to restore invariance by raising the FM's share of the gap's computation from ~14% to ~33%. It moved the spread down without restoring it, and pushed the FM into saturation instead — Result 3.

## Reproduction

```bash
cd experiments
# smoke (1 ckpt x 1 capacity, tiny FM; parts suffixed _smoke)
modal run --detach -m rhm.residual_decomposition.trajectory.rhm_decomposition_trajectory::smoke
# wide gap (the trajectory gap of record), both arms, all four capacities
modal run --detach -m rhm.residual_decomposition.trajectory.rhm_decomposition_trajectory::decomposition_trajectory
# narrow gap
modal run --detach -m rhm.residual_decomposition.trajectory.rhm_decomposition_trajectory::decomposition_trajectory \
    --arms "fmreg:1.0,wd:0.1" --caps "capB,capM,capQ2" --src post_block3 --tgt post_block6
# post-hoc (CPU): arcs, trust gates, invariance; --compare adds the retired metrics side by side
modal volume get rhm-scaling-data /rhm_decomposition_trajectory/summary.json .
python3 -m rhm.residual_decomposition.trajectory.analyze_trajectory --summary summary.json --cap capB --compare
```

`max_dop` (default 8) caps in-flight jobs; the workspace has a concurrent-GPU cap and spawning the whole grid at once exceeds it. Jobs launch in waves and each wave drains before the next spawns. Every job commits its own part before returning, so an interrupted run loses at most the in-flight wave and relaunching skips everything already on the volume.

## Data & schemas

- **Parts** (volume `rhm-scaling-data`): `/rhm_decomposition_trajectory/parts/{arm}_step{step}_{cap}[_{gaptag}][_{tag}].json`. The default gap emits **no** gap tag (so parts computed before the module was gap-parametrized keep their names); the narrow gap emits `_b3tob6`. Each part carries `beta`, `beta_r2`, the full `decomposition` dict (`basic` / `naive` / `geometry` / `repaired` — `geometry` holds the two variance spectra β is refit from), `eta2_residual`, and the retired `res_effective_rank_last_pct` / `res_top1_pc_last`.
- **Summaries**: `/rhm_decomposition_trajectory/summary.json` (wide), `summary_b3tob6.json` (narrow). A summary contains only the arms it was invoked with — re-invoke with both arms to re-aggregate from existing parts (0 jobs spawn).
- **Inputs**: the m2-distinct checkpoints, corpus and rules were migrated from the `jagilley` workspace volume (created 2026-06-20, holds the pre-July-08 results) to `chromatic` (created 2026-07-10, which had no `v16_s2_L6_m2_distinct` key). Additive only. Compute runs on `chromatic`.

## Caveats

- **One substrate, one seed** (m2-distinct, rule_seed=0). Every arm shares the same base training runs, so the 7/7 dissociation is 7 measurements of two training trajectories by different observers — not 7 independent trainings. It establishes observer-robustness, not seed-robustness.
- **The published-vs-measured comparison is shape-only** (differing `act_rank` conventions).
- **The capacity trend in descent magnitude is a trend, not a mechanism.** The bound-relative reading is offered as an interpretation, not a result.
- **`fmreg:3.0`, the arm with the largest published descent (63.5 → 42.9), was not re-cut** — only `fmreg:1.0`. Adding it is one command and would test whether descent magnitude tracks λ under the trusted instrument.

## Next steps

1. **A dose-response on λ** (`fmreg:0.03 / 1.0 / 3.0` at fixed capacity and gap): does descent magnitude on `frontier_mass` track annealing pressure? The published activation-rank arc says it should.
2. **Map where β's invariance actually holds.** The audit's ~33%-of-gap regime and this node's failures bracket it; a capacity × gap grid at fixed substrate would find the boundary and give β a stated scope instead of a counterexample.
3. **Port the directional findings to language.** β-rises / participation-falls are the two capacity-robust results here, and the complexodynamics next-steps list already wants a language-checkpoint test. Neither requires a trustworthy absolute scalar, which is what makes them portable.
