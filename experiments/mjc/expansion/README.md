# Cut #5 — dimensionality expansion under drift: the flagship, run, and closed

**Up**: [../README.md](../README.md) (mjc) · **Files**: [FILES.md](FILES.md)
**Belief under test**: [`beliefs/dimensionality_expansion.md`](../../../beliefs/dimensionality_expansion.md) (§Scope, 2026-07-27)
**Instrument of record**: [`rhm/residual_decomposition/`](../../rhm/residual_decomposition/README.md) — β, `R_res_participation`, frontier mass
**Idea docs**: [`ideas/physical_control_substrate.md`](../../../ideas/physical_control_substrate.md) cut #5 · [`ideas/heterogeneous_graders.md`](../../../ideas/heterogeneous_graders.md) §4b
**Status**: three pieces, 3 seeds each, complete. Single task family (planar n=5 arm, curl drift). **Date**: 2026-07-27.

---

## One-liner

Drifting a plant's physics — continuously, perpetually, in one parameter or in four localized
regions at once — **does not open the forward model's representational frontier**. The effect is
**±0.08 directions** against a **+0.72 ± 0.42** calibration signal from a genuine support increase,
i.e. **~10× below a known-real effect** measured on the same plant with the same instrument and the
same frozen probe. Drift moves the *target function*; it does not grow the *space of representable
functions*. That distinction was an argument in the belief doc; it is now a measurement with a unit
attached.

## Why this needed three pieces instead of one

The finding was never the hard part. **Rank-shaped instruments had failed three times in this
repo**, and the re-measurement traced all three to one cause — a *saturated* forward model, where
the FM drives relative residual to ~0, β collapses to 0.11–0.17 and rank inflates to 94% of
`d_model` in every domain tested. A null read off a fourth such instrument would have been worth
nothing. So the cut is built as **calibrate → measure → calibrate the other end**:

| | question | answer |
|---|---|---|
| **Piece 1** ([`saturation_gate.py`](saturation_gate.py)) | is the instrument in the dead zone here, and does it respond to a **wrong model**? | not saturated; separates a stale FM by **2.7×** |
| **Piece 2 / 2b** ([`support_fixed.py`](support_fixed.py)) | does support-**fixed** drift open the frontier? | **no** — ±0.08 directions |
| **Piece 3** ([`support_growing.py`](support_growing.py)) | does the instrument respond to **added directions** at all? | yes — **+0.72 ± 0.42**, 3/3 seeds |

Piece 3 is what makes Piece 2 mean something. Without it a flat line is ambiguous between *the
plant does not expand* and *the readout is blind*; with it, the null has a scale.

---

## Piece 1 — the saturation gate

Sweeps the FM's k-step-displacement prediction gap (k = 1…20) on the planar arm and measures the
frontier instrument at every cell. Four axes: **k** (gap width — which on this substrate *is* the
reactive↔ballistic commitment axis), **n_links** (3 = capacity slack, 5 = capacity binds,
`arm_substrate` P1), **FM variant** (`matched` / `stale` = known-wrong positive control / `small` =
β capacity-invariance check), **probe** (`task` = matched-FM reach transitions / `broad` = teleport
pool).

**The arm FM does not saturate.** The historical failure mode does not apply here:

| | rel. residual | naive `R_res` |
|---|---|---|
| saturated regime (RHM / MNIST / language) | 0.003–0.008 | 94% of `d_model` |
| arm, k=1, matched, task probe | **0.180** (n=3) / **0.107** (n=5) | 53% / 67% |

**The positive control passes and is monotone in k.** stale/matched frontier ratio, 3 seeds:

| k | 1 | 4 | 8 | 14 | 20 |
|---|---|---|---|---|---|
| n=3 | 1.42 | 1.77 | 3.32 | **4.08** | 4.41 |
| n=5 | 1.51 | 1.72 | 2.28 | **2.73** | 2.83 |

3/3 seeds at every k (n=5). A known-wrong model is barely distinguishable one step out and
unmistakable over a commitment horizon — **the ballistic/reactive asymmetry appearing in a
representation readout rather than a control one.**

**Three instrument findings that shaped everything downstream:**

1. **The instrument is degenerate at BOTH ends.** The known failure is saturation (frontier → 0).
   Rolling a model out adds an upper one the rhm domains never hit: once the open-loop rollout
   diverges, frontier mass → 1, every ρ → 0, and *"the model has nothing to say"* is exactly as
   unreadable as *"the model said everything."* The readable window is
   **0.02 < frontier mass < 0.90**, and every later piece is sited inside it.
2. **β is unusable here.** Capacity-invariance is the gate that catches a circular basis, and β
   fails it by 6–59×: |Δβ| over a 4× FM range is 0.06–0.59 (n=3) / 0.06–0.27 (n=5) against rhm's
   **±0.01**; fit R² is 0.26–0.51 against 0.95–0.99. It is a log-log slope over 2n = 6 or 10
   directions (rhm's own guard requires 8). **Dropped from every later piece** — computed and
   stored, never read. This is the measured version of the argument that β needs a learned latent.
3. **`R_res_participation` is the independent readout; frontier mass is largely magnitude.**
   Correlation with relative residual: **+0.002** (n=3), +0.486 (n=5); frontier mass instead
   correlates **+0.908** with relative residual at n=5. Concrete dissociation, 3/3 seeds at n=3:
   k=1 and k=8 have near-identical relative residual (0.21/0.16, 0.19/0.21, 0.14/0.17) while
   frontier mass falls 2–5× and `R_res_participation` rises 1.3–2.2×. **Same error, spread over
   more directions** — the shape/level split doing real work.

**n=5 is the substrate.** Its sweep is monotone in all three quantities (rel. residual 0.107→0.42,
frontier 0.046→0.214, `R_res_part` 3.94→6.81/10); n=3 is non-monotone and flipped one seed's
verdict. Consistent with capacity: h=256 against a capacity requirement of 32 is enormous slack.

*Also checked and clean*: mean-blindness (`repaired_triple` mean-centers, so a constant-offset
residual is invisible) is ≤ 0.16 of residual energy throughout — a smoke-stage worry that did not
survive a properly-trained FM.

## Piece 2 / 2b — the support-fixed null

Eight rounds of perpetual OU drift on curl gain(s), the FM re-fitting each round, graded on a
**frozen probe**: a fixed set of (initial state, 14-step command sequence) pairs re-executed in
each round's plant, so the same experiment runs every round and only the world and the model
differ (the RHM 2×2's fixed-held-out-probe rule). `R_act` is reported every round as the check.

**The 2×2 (2b, k=8, `R_res_participation`, late−early, per seed):**

| contrast | effect |
|---|---|
| **drift**, one-parameter geometry (`global` − `static`) | **−0.073 ± 0.266** |
| **drift**, four-region geometry (`regions` − `regions_static`) | **+0.073 ± 0.384** |
| **geometry**, drift off (`regions_static` − `static`) | **−0.012 ± 0.606** |
| *reference: Piece 3's DOF-accretion calibration* | *+0.724 ± 0.419* |

All within ±0.08 of zero, ~10× below the calibrated real effect. **The null holds in both
adaptation protocols** (`fresh` at k=8: +0.286 ± 0.549), which is what makes it robust to the
buffer artifact below.

**Supporting checks.** The drift genuinely happened (mean gain excursion 1.3–2.7; `static` exactly
0). `R_act` flat to ±0.13 (2) and ~0 (2b). The `rail` yardstick — a field-free FM carried through
every round, Piece 1's stale model — separates **2.37×** in `static`, matching Piece 1's 2.7×
reference. The `frozen` FM's error tracks the drift excursion at **corr +0.92 to +0.98** while the
adapting model is decoupled at +0.25: the adaptation loop demonstrably works and the instrument is
demonstrably awake.

**Two corrections this piece forced, both caught by controls rather than by inspection:**

- **`regions` was confounded and 2b fixes it.** Four gated curl regions is a *harder operator*
  outright (live relative residual 0.38–0.51 vs 0.27–0.34 elsewhere), not merely a
  differently-drifting one, so differencing it against the one-parameter `static` control priced
  geometry as drift. `regions_static` completes the 2×2 — and the geometry contrast is itself null
  (−0.012), so the original apparent effect *was* the confound.
- **k=14 was the wrong horizon.** Piece 3 shows the dimensionality signal is weakest there
  (+0.27 ± 0.74 for an intervention as blunt as three extra DOF) and resolves at k=8
  (+0.72 ± 0.42). Re-aggregating the *same stored rows* at k=8 moved `global`−`static` from
  +0.679 ± 0.998 to +0.286 ± 0.549 at no compute cost.

> ### ⚠️ Known contamination: the frontier-mass column in 2b
>
> `--buffer-mode accumulate` was added to reduce round-wise churn. It is **wrong under drift**: the
> buffer accumulates transitions from incompatible worlds (same `(s,u)`, different `s′` depending
> on which round's gain was active), so the model fits a contradiction. The embedded positive
> control caught it immediately — in `global`, `frozen`/`live` = **0.49×** (0/3 seeds), i.e. the
> *adapting* model is worse than the never-updated one. `static` is unaffected (2.37×) precisely
> because nothing drifts there.
>
> **Consequence**: 2b's frontier-mass reading for `global` (+0.090 ± 0.088, 3/3) is the live model
> degrading, **not** a frontier opening — and it flips sign between buffer modes (−0.025 in
> `fresh`), which is the tell. **The primary `R_res_participation` result is unaffected** and holds
> in both protocols. Neither buffer mode is right: `fresh` runs 256 epochs on 2000 samples per
> round (churn), `accumulate` mixes worlds. The principled fix is a **sliding window**
> (`--buffer-cap 4000`, i.e. the last two rounds); `buffer_cap` exists but was set to 16000, where
> it never binds. **Unrun.**

## Piece 3 — DOF accretion, the dimensionality-axis calibration

**A control, not a flagship** — and reframed as such mid-stream. Unlocking a joint gives a
*higher-dimensional* function, not a *deeper* one: there is no level to climb where learning the
upper level improves the lower. So its outcome space was "the frontier grows because we made the
problem bigger" (near-tautological) or "it doesn't" (instrument failure) — a weak flagship, but
exactly the right calibration, because for a calibration you *want* a case whose answer you know.

**The tautology it has to avoid, and how.** Locked joints barely move, so the *plant's*
displacement covariance mechanically gains directions when they unlock. The probe is therefore
drawn from and executed on the **fully-unlocked arm, once, for every round of every condition** —
the target array is literally identical throughout and `R_act` is constant **by construction**
(measured spread **0.0e+00**, asserted in code). The learner only ever *trains* on the body it
currently has; it is always *graded* against the full-DOF world.

Three conditions, one protocol, one shared random init: `locked` (3 distal joints pinned
throughout), `accretion` (3→2→1→0 over rounds), `full` (0 pinned). Locking is an `equality/joint`
constraint, **not** a shorter chain, so the state stays 2n-dim and the command n-dim and one FM
architecture spans every body.

**Result (k=8, late rounds, 3 seeds):**

| readout | `locked` (2 DOF) | `full` (5 DOF) | diff |
|---|---|---|---|
| **`R_res_participation`** | 5.657 ± 0.220 | 6.381 ± 0.357 | **+0.724 ± 0.419**, 3/3 |
| frontier mass | 0.895 ± 0.082 | 0.070 ± 0.013 | −0.825 ± 0.079, 3/3 |
| relative residual | 1.201 ± 0.155 | 0.180 ± 0.008 | −1.021 ± 0.162, 3/3 |

The separation exceeds its seed spread and agrees in sign across every seed, and the `accretion`
arm reproduces it **within one continuously-trained model** (5.70 at 2 free DOF → 6.43 at 5, i.e.
+0.73, matching the between-condition +0.72). **The calibration passes.**

**Direction of the effect, and why it is the opposite of RHM.** `R_res_participation` is the
participation ratio of the frontier spectrum `{(1−ρ_i)λ_i}`. A model that explains *nothing* leaves
the frontier equal to the activation spectrum, landing at `R_act` (5.88 here — `locked` reads 5.87,
almost exactly). A model that absorbs the loud directions and leaves the quiet ones (the β<1
photocopier picture) *flattens* what remains, pushing the count **above** `R_act`. So "better
model → higher count" is expected here, and is opposite to RHM, where a good model reads 7 against
`R_act` ≈ 73 because the frontier there ends up far more *concentrated* than the activations.
**The count is a shape descriptor whose interpretation is domain-dependent, not a quality metric.**

**The weakest link, stated plainly.** Endpoints separate; the path between them does not step
cleanly. Spearman against free-DOF count across the staircase: frontier mass **−0.80** and relative
residual **−0.93** (3/3 consistent) versus `R_res_participation` at **+0.36**, with one seed
negative. On a 10-dimensional motor state the dimensionality readout works but is noisy — the same
conclusion β reached in Piece 1 by an independent route.

**Only k = 4, 6, 8 have both conditions inside the readable window**: at k=1 the full-DOF model
saturates (frontier 0.007) and at k≥10 the impaired one diverges (0.919). The window is narrower
here than in Pieces 1–2 because this piece deliberately spans a much wider model-quality range.

---

## What this settles, and what it does not

**Settles.** *Drift ≠ expansion*, on a fixed-DOF plant, measured against a calibrated unit and
de-confounded across drift geometry. The `beliefs/dimensionality_expansion.md` §Scope prediction —
a fixed-DOF plant under support-fixed drift holds the frontier flat, and that is *health* — is
confirmed. Read through [`heterogeneous_graders.md`](../../../ideas/heterogeneous_graders.md) §4b
this is an existence proof that **continuous, non-stationary, genuinely-difficult prediction
pressure produces no expansion** — which is what "expansion needs an evaluative grader, because
opening a direction makes prediction worse before it makes it better" predicts.

**Does not settle.** The belief's *core* claim — that novelty **does** grow representable
directions — remains **untested everywhere**, including RHM, whose novelty arm swapped rule sets
wholesale and so measured catastrophic forgetting rather than novelty. A motor plant was never
going to test it: an arm is a fixed map from commands to state, with no hierarchy to climb. The
arm result supports that scoping from the other direction, and the next test belongs in a
hierarchical domain — [`rhm/residual_decomposition/`](../../rhm/residual_decomposition/README.md)
next step #3 (shared structure across cycles; same rules, deeper level; partial rule swap).

## Caveats

- **Single task family**, one drift family (Shadmehr curl), 3 seeds, n=5 only for Pieces 2–3.
- **The prediction target is the simulator's own generalized coordinates** (`qpos/qvel`) — privileged
  DGP access no embodied learner has. Predicting a learned latent is both the honest version and
  what β needs; **unrun**, and the leading candidate is FK-derived observations (link-endpoint
  Cartesian positions) through a learned encoder, which needs no renderer.
- **The dimensionality readout is the noisy one** at d_state = 10 (Piece 3, Spearman +0.36).
- **36–41% of probe states sit outside the FM's teleport training box** (max|q| 3.3–4.1 rad against
  `ArmEnv.wrapped()`'s 3.0 limit). Inherited from the substrate — reaches leave the collection box,
  which is `arm_substrate` P5's known-untried `v_explore` fix. Identical across conditions, so it
  adds noise rather than bias.
- **2b's frontier-mass column is protocol-contaminated** (see the box above). Unfixed.
- **Collection is teleport, deliberately**, against the node's default-for-new-cuts convention —
  the README's collection box carves this case out itself, since on-policy couples data quality to
  model quality and the dependent variable here *is* the FM's frontier as a function of the
  intervention. *Where* you collect is not the question in this cut.

## Reproduce

```bash
cd experiments/            # MODAL_PROFILE=chromatic
modal run mjc/on_policy/verify_backcompat.py::verify          # env gate (arm_env gained `locked_joints`)

bash mjc/expansion/train.sh          # Piece 1  -> tags gate_s0..2
bash mjc/expansion/train_piece2.sh   # Piece 2  -> tags sf_s0..2   (fresh buffer, k=14 summary)
bash mjc/expansion/train_piece2b.sh  # Piece 2b -> tags sf2_s0..2  (2x2, k=8, accumulate)
bash mjc/expansion/train_piece3.sh   # Piece 3  -> tags sg_s0..2

python3 mjc/expansion/saturation_gate_agg.py   --tags gate_s0 gate_s1 gate_s2
python3 mjc/expansion/support_fixed_agg.py     --tags sf2_s0 sf2_s1 sf2_s2
python3 mjc/expansion/support_fixed_agg.py     --tags sf_s0 sf_s1 sf_s2 --k 8   # the free re-read
python3 mjc/expansion/support_growing_agg.py   --tags sg_s0 sg_s1 sg_s2
```

Volume: `/data/expansion_gate/<tag>/`, `/data/expansion_support_fixed/<tag>/`,
`/data/expansion_support_growing/<tag>/`. Local mirrors in `figures/`, logs in `logs/`.

## Gotchas (each cost a run or nearly did)

1. **Probe windows must start MID-MOTION.** The curl field is `F = b·[[0,−1],[1,0]]·v_tip` — exactly
   zero at rest. A probe whose windows begin at a standstill measures the drift from the one state
   where the drifted parameter has no effect. Sample windows at random offsets from chained reaches.
2. **Motors must be removed (gear=0) from locked joints.** MuJoCo equality constraints are soft; a
   gear-8 motor driving a pinned joint drifted it **0.16 rad** within one control step, i.e. the
   "locked" body was not locked. With gear=0 plus a stiffer `solref` the deviation is 1.2e-03.
3. **Choose the lock set from measured variance share, not from anatomy.** The obvious pick — the
   three distal joints — carries **91.3%** of the probe's angular displacement variance, so the
   impaired model sat pinned at the upper degeneracy. j2–j4 (48%) leaves the dominant mover free.
   `--probe-only` prints the per-joint shares for exactly this decision.
4. **Never seed an RNG from `hash(str)` or from a list position.** Python salts string hashes per
   process; a list-position seed changes a condition's drift trajectory as soon as another
   condition is added ahead of it. Seed from a canonical index.
5. **`k_primary` must be in `k_list`**, or every summary lookup raises `StopIteration` from inside a
   Modal coroutine and surfaces as an opaque `RuntimeError: coroutine raised StopIteration`.
   Asserted in the entrypoint now.
