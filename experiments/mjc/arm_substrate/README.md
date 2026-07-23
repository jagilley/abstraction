# The planar arm substrate: a second task family, and capacity competition that comes from the body

**Up**: [../README.md](../README.md) (mjc) · **Idea doc**: [../../../ideas/two_timescale_value_loop.md](../../../ideas/two_timescale_value_loop.md) (the efferent bridge + the value→FM capacity gain) · [../../../ideas/physical_control_substrate.md](../../../ideas/physical_control_substrate.md) §"Animals are biological robots" (morphology as an inductive-bias knob).
**Parents / lineage**: this is *substrate machinery*, not a cut. It exists to retire the honest caveat on [../ballistic/README.md](../ballistic/README.md) — **"single-family (damping drift, corridor reach)"** — by supplying a second, nonlinear family for the ballistic/transmission arc, and to replace the pusher's *manufactured* capacity competition ([../value_shaping/README.md](../value_shaping/README.md), [meta_adapt §#4d/#4e](../meta_adapt/README.md)) with competition that is intrinsic to the plant. It inherits, and independently re-derives, two rules from the pusher arc: the value-relevant-teacher correction ([../drift_value_loop/README.md](../drift_value_loop/README.md) Cut 3) and the smooth-not-localized quality axis ([../ballistic/README.md](../ballistic/README.md) Cut 4b).
**Code**: [`../arm_env.py`](../arm_env.py) (the DGP — stays at the `mjc` node as shared machinery, alongside `pusher_env.py`/`shared.py`) · `arm_probe.py` (in this folder; characterization: P0–P6 + `run_arm_capacity_sweep`). File index: [FILES.md](FILES.md).
**First cut ported**: [`../ballistic/arm/`](../ballistic/arm/) — Cut 4c-arm, endogenous reward-free re-adaptation under a curl-field drift, 3 seeds. Its findings validate (and correct) the characterization below: see **[P7](#p7--the-substrate-under-a-live-cut-what-the-first-port-validated-2026-07-23)**.
**Status**: substrate characterized and **verified against every precondition the ballistic arc requires**; P0–P6 single-seed, P7 3-seed. Physics realism only — the acquisition model is unchanged from the pusher ([`../COLLECTION_REALISM.md`](../COLLECTION_REALISM.md)). **Date**: 2026-07-22, extended 2026-07-23 (P7).

---

## Why this exists

The pusher is a 2-DOF force-actuated point mass whose free-flight dynamics are near-**linear**. A 232-parameter arity-2 forward model already reaches R²=0.999 on it ([Cut #2](../arity_torque/README.md)), so **capacity never binds by itself**. Every interesting property therefore had to be bolted on through the `qfrc_applied` perturbation layer — `puck_field`, `patch`, `push_rot`, `field_patch`, `noise_patch`, `rot_regions`. Six hand-designed mechanisms carrying the scientific load, each with its own tuning story (`value_shaping`'s field had to be *multi-mode* to be capacity-hungry at all, then needed a second `pusher_amp` copy so the value-*relevant* side competed too).

An arm supplies from the physics what those layers were faking. `M(q)q̈ + C(q,q̇)q̇ + Dq̇ = τ`: configuration-dependent inertia, Coriolis terms quadratic in velocity, a task space (Cartesian tip) distinct from the actuation space (joint torque), and — with a passive distal segment — capacity competition whose strength is a mass in kilograms. It is also the canonical motor-adaptation paradigm, which buys readouts the pusher structurally cannot produce (aftereffects; directional generalization).

**Discipline unchanged**: controllable-dynamics DGP, knob sweeps with baselines, no RL-to-SOTA. Plain MuJoCo CPU physics + PyTorch, no new dependencies — [`../arm_env.py`](../arm_env.py) is a ~40-line MJCF built from a knob dict, exactly like [`../pusher_env.py`](../pusher_env.py).

## The env (`../arm_env.py`)

Gravity-free planar chain (hinge axes +z) — a horizontal arm on a table, which is both the pusher's convention and the geometry of the human reaching experiments (arm supported against gravity). **Contacts off by default**: Cut #1 established contact is the stiff near-discontinuous regime and Cut 4b established such regimes are open-loop-**incompensable**, so self-collision would only saturate ballistic control.

- **State** `s` (2n-dim) = `[qpos(n), qvel(n)]` — joint angles (rad) and rates. **Command** `u` (n_act-dim) = motor torque, `u∈[-1,1]`, scaled by `gear`. **Goal** = a Cartesian position of a named site, *not* part of the state.
- **Joints are unlimited** by design (a joint limit is a stiff constraint = another discontinuity). The operating region is bounded by the *sampling distribution* instead, which is under our control because collection is teleport-based. `wrapped()`/`max_absq()` are the diagnostic that this actually held (it did: max |q| = 2.3–2.9 rad under every controlled rollout).
- **Analytic FK is a fixed known nonlinearity**; the *dynamics* is what is learned and what drifts. `fk(q, L, upto=k)` returns the distal end of link `k`. Verified against MuJoCo's own `site_xpos` to **2e-16** — if this disagreed, the CEM cost would be scoring a different arm than the simulator executes.

**Drift / perturbation knobs** (all additive and off by default, the [`../pusher_env.py`](../pusher_env.py) contract):

| knob | what it does | why this shape |
|---|---|---|
| `payload_mass` | point mass at the tip | changes `M(q)` **non-uniformly** over configuration space (inertia contribution ∝ tip radius `r(q)²`) — local yet smooth |
| `curl_field` | `F_tip = b·[[0,−1],[1,0]]·v_tip`, applied as `Jᵀ F` via the exact `mj_jacSite` Jacobian | the **Shadmehr & Mussa-Ivaldi** force-field protocol. Smooth, global, one-signed, open-loop **compensable**; perturbs the hand *perpendicular to its motion*, so a wrong internal model produces a lateral deviation that **accumulates** open-loop and is corrected step-by-step by feedback — precisely the ballistic/reactive asymmetry. The axis aftereffects are defined on. |
| `torque_rot` | rotates the command→torque map | the arm's `push_rot` analog (input-coupled conflict, Cut #4's requirement) |
| `joint_noise` | per-substep random torque | aleatoric / noisy-TV decoy |
| `n_passive`, `tool_mass`, `tool_damping` | the last k joints carry **no motor** — a floppy tool / hanging load | capacity competition from **morphology** (see P6) |
| `goal_site` | `tip` (chain end) vs `hand` (end of the last actuated link) | **flips value-relevance with the physics byte-identical** (see P6) |

Perturbations accumulate into a freshly-zeroed `qfrc_applied` each substep, avoiding the single-writer clobber bug that bit `_apply_actuator_rot`/`_apply_fields` in [`../pusher_env.py`](../pusher_env.py) (Cut #4d gotcha (b)).

---

## What was verified (`arm_probe.py`)

### P1 — capacity binds intrinsically, but **only once the chain is long enough**

The dedicated instrument is `run_arm_capacity_sweep` (broad collection distribution; monotone and well-behaved). Readout = the **capacity requirement**, the smallest hidden width reaching velocity-dim R² ≥ 0.99. **Pusher reference = 8** (the smallest width Cut #2 tried already gave 0.998).

| regime | linear-R² | capacity req |
|---|---|---|
| n=2, v=1.5 | +0.83 | **8** — pusher-like |
| n=2, v=8 | +0.31 | 16 |
| n=3, v=8 | +0.51 | 32 |
| n=3, v=14 | +0.35 | 128 |
| n=5, v=8 | +0.60 | 256 |
| n=5, v=14, fs=30 | +0.52 | **never** (0.917 at h=256) |

**The load-bearing negative, and the mechanism.** At the natural first choice — a **2-link** arm — capacity does **not** bind: 164 parameters already give R²=0.982. The reason is structural and says exactly where to look: for a 2-link arm `M(q)` depends on the **elbow angle alone**, so the entire configuration-dependence of the dynamics is a smooth function of *one variable*, which a tiny MLP handles.

This sharpens Cut #4d's lesson ("energy ≠ prediction cost") into a cleaner statement: **"nonlinear" and "capacity-hungry" are separate axes.** Linear-R² falls from +0.83 to +0.31 — the map is genuinely far from linear — while the capacity requirement stays at 8–16. What costs an MLP capacity is *many-mode / high-dimensional* structure, which is why `value_shaping` needed a multi-mode field. **Chain length is the knob**: `M(q)` for an n-link chain depends on n−1 angles.

*Caveat*: the in-run P1 (measured on the task distribution, ~1300–2700 on-task transitions) is **non-monotonic** and goes negative at small widths (n=3: −0.92, +0.90, +0.84, +0.79, +0.97, +0.98). Trust the dedicated sweep, not the in-run span.

### P2 — staleness is state-local in the way the physics predicts (curl), and washes out on-task (payload)

Excess stale-FM error, binned by the variable each axis physically implicates:

- **curl** (binned by tip speed — the field's force is ∝ ‖v_tip‖): monotone, excess **0.139 → 0.602** across the bins, slope **+0.29**. Robust and physically predicted.
- **payload** (binned by tip radius — inertia contribution ∝ `r(q)²`): holds on the **broad** distribution (1.95× at n=2, 2.26× at n=5, slope +4.65) but **washes out on the task distribution** (1.20× at n=3, 0.83× at n=5). Reaches simply do not sample enough range of `r(q)`.

This is the intrinsic version of the local-but-compensable drift Cut #5 had to synthesize with `rot_regions` — here it is physics, no scaffolding. **Report the slope, not the endpoint ratio**: a high/low ratio of 39.6× appeared in one run purely because the lowest-speed bin's excess was 0.014 (small-denominator artifact).

### P3 — the reach is feasible, and **P5's earlier nulls were the planner, not the substrate**

Cut 4b's precondition #2 is that the matched controller actually reaches. It does — but only with an adequately-sized planner, which turned out to be the single biggest confound in this characterization:

| n=3, curl axis, matched FM | k_shoot=256, iters=4 | k_shoot=1024, iters=8 |
|---|---|---|
| reactive | 0.0876 | **0.0279** |
| ballistic | 0.2021 | **0.0929** |
| ballistic reaches (vs do-nothing 0.40) | 48% | **77%** |
| transmission ratio (P5) | 1.73× | **4.63×** |

Open-loop CEM here optimizes a 42-dimensional action sequence (H=14 × 3 joints); the pusher used `k_shoot=256, cem_iters=4` for a 28-dim sequence under near-linear dynamics. **Under-optimized, ballistic's error is dominated by planning noise rather than FM error, which flattens the transmission slope toward zero.** Three earlier configurations returned 1.48× / 0.63× / 1.73× on this budget and looked like substrate failures.

### P4 — the composition horizon, measured

How far a one-step FM composes before its rolled-out **tip** trajectory diverges (threshold 0.05 m). Pusher reference: **~6–8 steps** (`dynamics_shift.replan_ablation`; the a2a REACHING_LOOKAHEAD limit reappearing on physics).

| n=2 | n=3 | n=5 |
|---|---|---|
| ≥30 (0.012 m at h=28) | 20–23 | **14** |

An open-loop plan longer than this runs past where the FM is trustworthy — Cut #3 saw even the *oracle* degrade there — so **H must be set below the composition horizon**, which is why an early n=5 run at H=18 handicapped ballistic before any staleness was applied.

### P5 — ballistic transmits FM quality more than reactive, on a second, nonlinear family

n=3, curl staleness axis `b_train ∈ {6.0, 4.5, 3.0, 1.5, 0.0}` at `b_test=6.0` (one-signed), strong planner, 48 reaches:

| b_train (staleness →) | 6.0 | 4.5 | 3.0 | 1.5 | 0.0 |
|---|---|---|---|---|---|
| **reactive** | 0.0268 | 0.0202 | 0.0336 | 0.0457 | 0.0644 |
| **ballistic_cem** | 0.0883 | 0.1665 | 0.1577 | 0.2139 | 0.3139 |

**Absolute damage across the controlled axis: ballistic +0.226 vs reactive +0.038 — a 6.0× dissociation** (monotone in staleness). Slope against `fm_err`: reactive +0.101, ballistic +0.466 = **4.63×** (pusher: +0.35 / +1.07 = 3.0×).

*Caveat on the slope.* `fm_err(task)` came out **non-monotonic** (0.516, 0.494, 0.479, 0.577, 0.860) while `fm_err(broad)` is cleanly monotone (0.163 → 5.532). These reaches run to 15 rad/s while collection samples at `v_explore`=8, so the task probe set is dominated by a high-velocity tail where *every* FM extrapolates badly and the curl-specific signal is second-order. **Prefer the damage-across-staleness reading (6.0×); the 4.63× slope rests on a noisy x-axis.** Fix: widen `v_explore` to cover the operating tail, or score excess error (stale − matched) rather than raw.

### P6 — capacity competition from the body, spanned by one physical knob

The passive tool: the last `n_passive` joints carry no motor. Two properties make this the principled replacement for a designed force field.

**(a) `tool_mass` continuously spans Cut #4d's boundary condition.** Readouts: *coupling* = the arm-velocity R² gap between a full FM and a **tool-blind** FM `f(s_arm,u)→Δs_arm` (what ignoring the tool costs on the part the task cares about — an operational definition of value-irrelevant-and-decoupled needing no hand-set mask); *control cost* = CEM over the blind FM vs the full FM, both executed in the true env with a `hand` goal.

| tool_mass | coupling | reactive cost of dropping | ballistic cost | capacity req full / blind |
|---|---|---|---|---|
| 0.02 kg | +0.0003 | −0.006 | −0.016 | 32 / 32 |
| 0.10 kg | +0.008 | +0.004 | −0.005 | 32 / 32 |
| 0.30 kg | +0.040 | **+0.034** | +0.006 | 256 / never |
| 0.60 kg | +0.066 | **+0.041** | +0.039 | 256 / never |
| 1.00 kg | +0.089 | **+0.046** | +0.026 | 256 / never |

Coupling rises monotonically; the capacity requirement jumps 32 → 256 between 0.1 and 0.3 kg and the blind FM stops reaching the target entirely. **The transition sits at ~3–10% of arm mass**, and the effect size at 0.3 kg (+0.034) lands on top of Cut #4e's tuned **+0.036 ± 0.012** — the same boundary condition, now a mass in kilograms rather than a field amplitude.

**(b) `goal_site` flips value-relevance with the physics byte-identical.** `hand` (upstream of the tool) makes the tool's state value-irrelevant; `tip` makes it value-critical. Same XML, same dynamics, same trajectories — **the intervention is on the value alone, with the transition operator fixed.** The pusher could never run this: making the puck value-relevant meant changing the task, which also changed its difficulty. This is Cut #3's "intervene on the operator, hold the task fixed" run in the opposite direction. Legitimacy check: analytic `FK(upto=n_act)` matches MuJoCo's `hand` site to **4e-16**, so the hand's kinematics genuinely do not depend on the tool joints and "drop the tool from the FM" is a well-defined operation.

*Caveat*: control degrades overall with tool mass (reactive 0.041 → 0.192 against a 0.372 do-nothing floor), so at 1.0 kg ballistic is entering saturation — the same compression that produced the earlier false negatives. **0.3–0.6 kg is the usable competition regime.**

### P7 — the substrate under a live cut: what the first port validated (2026-07-23)

P0–P6 are characterization: the substrate grading *itself*. [`../ballistic/arm/`](../ballistic/arm/)
is the first real cut run on it (Cut 4c-arm — endogenous, reward-free FM re-adaptation under a
curl-field drift `b0=0 → b1=6`, 3 seeds), and it is the first evidence that the characterization
was *predictive* rather than merely self-consistent. It was run at P5's config **byte-identical**
(`joint_damping=0.5, gear=8.0, frame_skip=10, vel_pen=0.5, v0_std=0.0, cem_elite=32`, n=3 curl
design point), so the only difference from P5 is where FM quality comes from.

**Confirmed, and the substrate earns its keep:**

| what P0–P6 predicted | what the live cut found |
|---|---|
| P5: ballistic transmits FM quality far more than reactive (exogenous axis, 6.0× damage, 1 seed) | **endogenous** version reproduces at **5.24× ± 0.18** recovery gain, 3 seeds (pusher Cut 4c: 4.3×) |
| P3: the reach is feasible ballistically with a strong planner | ballistic recovers to **0.0999** against a matched-FM ceiling of **0.0993** — the ceiling is reachable, not asymptotic |
| P4/P3: H=14 below the composition horizon, `k_shoot=1024`/`cem_iters=8` sized to the 42-dim sequence | no planner-induced flattening; the smoke at the pusher-tuned 256/4 gave 1.54×, the sized run 5.24× — **P3's warning reproduced exactly** |
| curl is the better transmission axis (perpendicular, accumulates open-loop) | confirmed, and it is what makes the aftereffect below possible |

**The aftereffect — measured, and it retires the substrate's biggest outstanding caveat.** Grading
the re-adapted FM back in the field-free world gives a **mirror-signed** lateral deviation:

| ballistic, signed lateral (+ = the direction the curl field pushes the hand) | 3-seed mean |
|---|---|
| naive FM, evaluated **in** the field | **+0.175** |
| adapted FM, evaluated in the **field-free** world | **−0.168** |

Ratio 0.96, signs opposed. The naive model is pushed by a field it does not represent; the adapted
model pre-compensates a field that is no longer there and misses by nearly the same amount the
other way. This is the canonical Shadmehr signature that adaptation was a **model update**, not
impedance/co-contraction — and it is **ballistic-specific** (cost of being adapted-in-the-wrong-world:
+0.177 ballistic vs +0.021 reactive, 8.6×; 12.7× on the lateral term), because feedback corrects
it away within a step. **The pusher structurally could not produce this**: it has no perpendicular,
motion-dependent axis for a mirror-image error to be defined on. This is the clearest case so far
of the arm buying a readout rather than just a second data point.

**Two P-level corrections the port forces:**

- **P5's noisy task-probe x-axis is reduced but not solved.** P5 prescribed two fixes for its
  non-monotonic `fm_err(task)`: widen `v_explore`, or score **excess** error against a matched FM.
  The port took the second (it does not perturb the substrate). Excess is much better behaved but
  still non-monotonic across the ladder (0.595, 0.124, 0.177, 0.135, 0.059, 0.060) and does **not**
  reach zero where control is already at ceiling. **Standing rule: report damage/gain on this
  substrate, never a slope against `fm_err`.** If a future cut genuinely needs a clean FM-error
  axis, widen `v_explore` to cover the reach's high-velocity tail — that fix is still untried.
- **Re-adaptation is step-like here too.** 96% of the ballistic recovery is complete by the first
  milestone (400 transitions), flat thereafter. The pusher's "recovery is fast/step-like" caveat
  was expected to stretch into a graded trajectory on a harder plant and **did not**. A curl field
  is apparently as easy to re-learn as a damping change. Milestones below 400 (50/100/200/300) are
  needed to resolve the curve; a genuinely compositional drift may be needed to stretch it.

**New gotchas from the port** (both cost a re-run): (iv) the `eval_geometry` reach-band rejection
keeps only **~25%** of joint-space candidates, so any tuple count (`bc_tuples`, eval batches) must
be sized to the *kept* count, not the drawn count — a BC arm cloned from 78 kept tuples visibly
under-fits; (v) `modal run` intermittently dies client-side with *"Could not connect to the Modal
server"* **before the function is created**, and exits 0 through a pipe — it hit 2 of 3 seeds on
the first pass. Loop on a completion marker in the log, not on exit status
([`../ballistic/arm/train.sh`](../ballistic/arm/train.sh) does this).

---

## Three transferable methodological findings

1. **Grade FM error on the TASK distribution, not the collection distribution.** A first pass scored FM quality on the broad collection pool (wide configuration box, isotropic random velocities) while scoring control on *reaches* — two different distributions. `fm_err` moved 6× while control wandered non-monotonically, because most of that error lived in states the reach never visits. Fixing the probe set to transitions actually visited by matched-FM reaches took a smoke from **0.81× → 24×**. This independently re-derives [drift_value_loop](../drift_value_loop/README.md) Cut 3's correction for the meta-loop's *teacher*: the signal that predicts behaviour is the **value-relevant** FM prediction error, not the global one. The arm inherits the rule rather than escaping it. (`arm_probe.py` keeps a broad probe alongside purely as the contrast that documents this.)
2. **Size the open-loop planner to the action-sequence dimension.** See P3. A CEM budget tuned on the pusher silently converts the ballistic transmission effect into a null on a higher-DOF plant.
3. **A light passive link violates the explicit-integration stability limit and NaNs silently.** A rod of mass `m`, length `L` has `I = mL²/3`; explicit integration with viscous damping `c` needs `dt < 2I/c`. At `tool_mass=0.02` kg that limit is ~0.001 s — **below** the 0.002 s timestep — so inheriting the arm's damping diverges. MuJoCo only *warns*; the NaN then propagates into transitions, FM training, and control numbers that still look finite. It produced a control distance of 0.79 (worse than doing nothing), which reads as a plausible bad result rather than a broken run. Now: `tool_damping` auto-scales to a safe fraction of the limit, and `step()` counts non-finite states (`ArmEnv.nonfinite()`).

## Corrections to the substrate's own pitch

Recorded because they were wrong in the proposal and are load-bearing for anyone designing on top of this:

- **Capacity does not bind on a 2-link arm.** It binds at n≥5. See P1.
- **Kinematic redundancy does *not* give a droppable subspace.** Tip position depends on *all* joint angles, so the FM cannot drop any state dimension; redundancy makes certain *motions* value-irrelevant, not certain *state dims*. The passive tool is what actually delivers a droppable subsystem.
- **Arity-1 is no longer at the floor** (R² 0.40–0.68 vs the pusher's ~0.00). A long chain has genuine autonomous dynamics — Coriolis and inter-link coupling are predictable from state alone — so the command is not nearly the whole story. **Cut #2's arity demonstration should stay on the pusher**, where the actuated dims are command-dominated and the floor is maximally clean.
- **Capacity-binding and clean transmission pull against each other.** Capacity binds precisely when the FM cannot fit the map, so the *matched* FM starts degraded and the staleness axis begins from a worse baseline (the pusher never had this problem — its FM was essentially perfect). Resolution is the arc's existing one: run transmission cuts at high capacity and treat capacity as a separately swept knob, as #4d/#4e did with `fm_hidden`.

## The recommended design point

**n=5, `n_passive=2`, `tool_mass` 0.3–0.6, `curl_field` staleness axis, `goal_site=hand`, H ≤ composition horizon, `k_shoot` ≥ 1024 / `cem_iters` ≥ 8, FM error graded on the task distribution.** This is the only setting found that satisfies every precondition at once: capacity binds, the reach is feasible ballistically, the quality axis is smooth and one-signed, staleness is state-local, and control sits well off the floor.

### What this substrate does *not* fix — read before porting a directed-collection cut

The arm replaces the pusher's **scaffolding** (six bolted-on `qfrc_applied` perturbation layers → physics; K=4 hand-drawn Gaussian gates → continuous locality, P2's stale error ∝ tip speed). It does **not** change the **acquisition model**, and says so by contract: [`../arm_env.py:57`](../arm_env.py) — *"collection is teleport-based (`set_state`)"* — which is what makes the operating region controllable via the sampling distribution (§The env) and is load-bearing for P1/P2/P6.

That is fine for every *exogenous-axis* cut (transmission, re-adaptation, capacity), where the experimenter sets FM quality and collection is just instrumentation. It is **not** fine for any cut where *where to collect* is the dependent variable. [`../ballistic/directed/`](../ballistic/directed/README.md)'s audit found its loop result was unmeasurable partly because monitoring was free: 2,240 uncharged teleported transitions per round decided where to spend a budget of 100, a **22× subsidy** that demotes the relevance term from *where should I even look* to a tiebreaker on repair effort. Porting that cut here inherits the problem unchanged.

> **Rule for the arm's directed cut: it needs on-policy collection first.** Data as a byproduct of behaviour — you get the transitions your body passed through, and practising elsewhere costs an excursion in task time. The recommended implementation is a `collection_mode: teleport | on_policy` flag on `collect_pool` rather than a second substrate, so this plant keeps its characterization and finished cuts stay byte-identical; see [`../COLLECTION_REALISM.md`](../COLLECTION_REALISM.md) §3, which also states the real cost (it couples model quality to data quality, so dissociations measured that way are compound rather than clean). *Stopgap*: charging the monitoring survey against the collection budget removes the 22× subsidy as a config change, without changing how transitions are obtained.

## Caveats

- **Single seed throughout, in this file.** Every P0–P6 number is one seed; the mechanism readouts (coupling monotonicity, capacity requirement, composition horizon) were stable across configuration changes. P7's port numbers are 3-seed and are the only multi-seed evidence the substrate has.
- **Physics realism ≠ experience realism.** The arm fixes the plant and inherits the acquisition model unchanged — see §"What this substrate does *not* fix" and the node-level memo [`../COLLECTION_REALISM.md`](../COLLECTION_REALISM.md). Everything verified here is verified under teleported, free, uniform-access collection.
- ~~**No cut has been ported.**~~ The first port is [`../ballistic/arm/`](../ballistic/arm/) (Cut 4c-arm — endogenous reward-free re-adaptation under a curl-field drift, multi-seed, plus the aftereffect readout), running now. This section is characterization only; the payload axis is retained but the curl axis is better on every measured count.
- ~~**Aftereffects** ... built for but unmeasured.~~ **Aftereffects are now measured** (P7, 3 seeds, mirror ratio 0.96) — this was the substrate's largest outstanding justification and it is discharged. **Directional generalization remains unmeasured**: whether adaptation acquired at one reach direction transfers to others, which is the second readout the arm was built for and the one that would say whether the FM learned the *field* or a set of direction-specific corrections.
- **Aftereffect caveat**: measured at a single drift magnitude (`b=6`), with the ladder's endpoints only, and it is confounded with "the FM is simply wrong for `b=0`" in the same way any aftereffect is. The *sign* is what carries the argument, not the magnitude — an unsigned miss would be equally consistent with a degraded model.
- **The 3-link/5-link `q_center` postures were chosen by hand** to sit comfortably inside the workspace; no sensitivity analysis.

## Reproduce

```bash
cd experiments/            # MODAL_PROFILE=chromatic
# substrate characterization (P0-P5) at the recommended axis
modal run mjc/arm_substrate/arm_probe.py::arm_probe --quick               # smoke
modal run --detach mjc/arm_substrate/arm_probe.py::arm_probe --tag n3_curl_strong \
  --n-links 3 --link-lengths "0.4,0.4,0.3" --link-masses "1.0,1.0,0.6" --q-center "0.4,0.8,0.6" \
  --v-explore 8.0 --axis curl --a-test 6.0 --a-trains "6.0,4.5,3.0,1.5,0.0" \
  --q-range 0.9 --reach-amp 1.2 --reach-lo 0.25 --reach-hi 0.50 --plan-h 14 \
  --pool-n 14000 --fm-steps 6000 --n-eval 48 --k-shoot 1024 --cem-iters 8 \
  --no-do-capacity --no-do-composition

# where capacity binds (the dedicated, monotone instrument)
modal run --detach mjc/arm_substrate/arm_probe.py::arm_capacity_sweep --tag cap_v2 \
  --n-links-list "3,5" --v-explore-list "8.0,14.0" --frame-skip-list "10,30" \
  --link-lengths "0.4,0.4,0.3,0.25,0.2" --link-masses "1.0,1.0,0.6,0.4,0.3" \
  --q-center "0.4,0.8,0.6,0.4,0.3"

# P6 -- the passive tool spanning the capacity-competition boundary
modal run --detach mjc/arm_substrate/arm_probe.py::arm_tool_probe --tag tool_v1 \
  --tool-masses "0.02,0.1,0.3,0.6,1.0" --n-eval 48 --k-shoot 512 --cem-iters 5 \
  --cap-hidden "32,256"
```

**Gotchas**: (i) boolean flags use Modal's `--no-<flag>` form (`--no-do-capacity`), not `--do-capacity false`; (ii) launch heavy detached runs **one at a time** — a chained tool sweep died server-side mid-run with no OOM or warning, the eviction hazard [../ballistic/README.md](../ballistic/README.md) already documents; (iii) `zsh` does not word-split unquoted variables, so a shared-flags variable reaches Modal as a single argument.

## Figures

- `figures/arm_probe_<tag>/` — `fig1_capacity` (capacity frontier, arity-2 vs arity-1) · `fig2_locality` (excess stale-error vs the axis's predicted state variable) · `fig3_composition` (FM rollout tip-error vs horizon) · **`fig4_transmission`** (control vs FM-error per controller — the headline slopes).
- `figures/arm_capacity_sweep_<tag>/fig_capacity_regimes.png` — R² vs width per (n_links, v_explore, frame_skip); flat-and-high = the pusher's problem.
- `figures/arm_tool_probe_<tag>/fig_tool_coupling.png` — **coupling and control cost of dropping the tool vs `tool_mass`** — Cut #4d's boundary condition spanned by one physical knob.
