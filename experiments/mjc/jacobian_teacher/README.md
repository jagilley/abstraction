# jacobian_teacher — the forward model in *backward* mode

**Goal.** When a sensory endpoint error is pulled back through a learned forward model's **action
Jacobian** and used as the teaching signal on a committed motor program, does it teach faster,
with less variance, and with better generalization than reward alone — and does the advantage
track the **direction** accuracy of that Jacobian rather than the model's forecast accuracy?

**Status**: Phases A/B/C run, 2026-09-03. **Parent**: [`../README.md`](../README.md) (mjc).
**Spec**: [`SPEC.md`](SPEC.md). **Files**: [`FILES.md`](FILES.md).
**Reading**: Garibbo, Filipe, Aitchison & Costa 2026[^private],
*Unifying error and reward action learning: a cerebello-basal ganglia theory* — the action-gradient
framework, eqs. 1–2. **Substrate**: forks [`../ballistic/arm/arm_readapt.py`](../ballistic/arm/arm_readapt.py)
(Cut 4c-arm), gated in Phase A.

---

## The headline

**A forward model can be locally near-perfect and still teach in the wrong direction.** Across a
capacity ladder the **one-step** action Jacobian is flat and essentially exact (cosine 0.989 →
0.9996) while the **composed** 14-step one — the object the teacher actually uses — runs −0.15 →
0.98, and it is the composed reading that orders the learning outcome. A reading taken on what the
model was trained to predict calls all five of these equally good teachers; two of them teach
backwards.

Secondary, and all replicated over three seeds:

- The error-based teacher is **~1.5× more data-efficient than reward**, and that advantage is
  **flat** across the whole reward-free re-adaptation ladder — because weighted Jacobian *sign*
  agreement never leaves 0.90–1.00 there. Garibbo's sign-sufficiency claim, confirmed by the
  absence of an effect.
- **The optimal β reverses with Jacobian quality** (eq. 2's central claim, measured).
- **Sign-flip dysmetria** reproduces Garibbo Fig. 3g–j's structure on a 2×3 sensitivity matrix:
  one of six flips nearly free, five costly, all-flipped a blend of both deficit types.
- **Generalization to untrained reach directions goes the way Garibbo Fig. 2e predicts** —
  reward wins inside the trained band, the error gradient wins outside it, by up to 2.1×.

---

## The setup, and what is new in it

Cut 4c-arm supplies everything but the derivative: the n=3 planar arm, the Shadmehr curl-field
drift `b0=0 → b1=6`, a one-step forward model `f(s,u) → Δs`, a CEM planner, the reward-free
re-adaptation ladder, the task-distribution probe, and the behavior-cloned motor program
`π(s, goal) → H×n_act` that is the object every teacher here trains. Three uses of that FM already
exist in this directory — **forward** (roll it, pick a plan: [`../ballistic/`](../ballistic/README.md)),
and as an **error** or a **magnitude** (forecast minus outcome; the arity gap
`‖FM₂(s,u) − FM₁(s)‖` in [`../agency_gate/`](../agency_gate/README.md) and
[`../plasticity_gain/`](../plasticity_gain/README.md)). **Backward** — differentiating it with
respect to the command and using the result to assign credit — had not been built.

Garibbo's derivation is exactly that: `∂e/∂φ = (∂e/∂y)(∂y/∂u)(∂u/∂φ)`. Sensor supplies the first
factor, policy the third, the cerebellum the middle one. Their eq. 1 is reward-based learning as
an action gradient, `δ·(u − μ)/σ²`; their eq. 2 mixes the two with a weight β.

**The teachers.** All differ in one line, and all return the same object — an action-space vector
of shape `(B, H, n_act)` — consumed by the same update `(g.detach() * μ_φ).sum()`. So `mixed@β` is
literally eq. 2, and no arm has a private mechanism that could be doing the work.

| arm | `g` = |
|---|---|
| `rbl_cont` / `rbl_bin` | `−δ (u − μ)/σ²`, δ = r − v, continuous (−distance) or binary (in the reward zone) |
| `ebl_sensory` | `(y − y*)ᵀ ∂ŷ/∂u` — the **realized** endpoint error, in action coordinates |
| `ebl_imagined` | `(ŷ − y*)ᵀ ∂ŷ/∂u` — the **model's own predicted** error, same Jacobian |
| `mixed@β` | `β·g_ebl + (1−β)·s·g_rbl`, `s` scale-matching the two |
| `mixed_raw@β` | the same with `s = 1` — eq. 2's literal sum |
| `ebl_committee`, `ebl_committee_gated`, `mixed_agree` | Phase C, below |

Every arm starts from an **identical copy of one behavior-cloned motor program**, trains on the
**same reaches in the same order** from the same pool, with the **same exploration-noise draws**,
for the **same 20 000 executed reaches**. The determinism this buys is audited rather than
asserted: `rbl` re-run at a different rung of the ladder returns bit-identical numbers (0.102632
vs 0.102632; 0.177547 vs 0.177547), which it must, since it never touches the FM.

**The instrument.** `plant_endpoint_jac_fd` obtains the true `∂y/∂u` from the simulator by central
finite differences, using `set_state` as the experimenter's ruler (MuJoCo is memoryless, so a
teleport-and-perturb query is exact). This is a privileged measurement — an agent has no such
thing — which is what Phase C is about.

---

## Phase A — the gates

| gate | reading |
|---|---|
| 1 · fork fidelity | 38/38 entrypoint defaults re-derived from `arm_readapt.py`'s source with `ast` and identical; fixed-command trajectory hash `d2ea2114e5076d52`; `fk_torch` ↔ `fk` 1.0e-07 (float32) |
| 2a · kinematic oracle | analytic ∂fk/∂q vs MuJoCo's own `mj_jacSite`: **4.4e-16** |
| 2b · dynamic oracle | six-point central-difference sweep; cosine vs the eps=1e-3 reference ≥ 0.99994 from 1e-4 to 1e-2, falling only at 3e-2 (0.999781). A finite difference has no error bar of its own; the plateau is the evidence |
| 2c · FM direction | matched ceiling endpoint cos **0.9863**, vjp_cos 0.9853, sign_w 0.999, scale 1.102 · stale pre-drift **0.8249** / 0.8536 / 0.945 / **0.576** |
| 3 · reward arm | `rbl` 0.2066 → 0.0176 over 20 000 executed reaches |
| 4 · gradient vs sampling planning | `ebl_imagined` 0.0562 vs `ballistic_bc` 0.0578 = **0.97×** |

Two of these needed the design changed rather than merely recorded.

**Gate 3 measured the scale mismatch.** ‖g_ebl‖/‖g_rbl‖ ≈ **1/76** on a shared batch. Eq. 2's raw
sum therefore makes β = 0.5 about 99 % reward-based, so β is not an interpolating axis unless the
two gradients are made commensurate. Both forms are run throughout — `mixed@β` scale-matched,
`mixed_raw@β` literal — and they behave differently everywhere below.

**Gate 4 was mis-posed as first written.** `ebl_imagined` is *amortised* into one π(s, goal);
`ballistic_cem` re-optimises per episode, and against a *different* objective (a running tip cost
plus a terminal velocity penalty, where `ebl_imagined` descends the terminal endpoint error alone).
The matched incumbent is `ballistic_bc` — the other amortised use of the same FM — and against it
the gate passes at 0.97×. Against the per-episode terminal-only CEM (`ballistic_cem_ep`, added for
this) it is 2.55×, and that gap is the price of amortisation, not of the gradient.

The stale model's Jacobian **scale is 0.576** — it underestimates its own leverage roughly twofold
while still getting 94.5 % of signs right. Garibbo's requirement form says that should still teach,
only slower, which Phase B is the test of.

---

## Phase B — the ladder

Three seeds. Plateau-mean train-band endpoint error; the shared initial motor program sits at
0.1868 ± 0.0175.

### The FM-quality axis, read two ways

| rung | `fm_err` | 1-step cos | endpoint cos | sign_w | `ebl_sensory` | `rbl_cont` | `ballistic_cem` |
|---|---|---|---|---|---|---|---|
| m=0 (stale) | 0.8665 | 0.9990 | 0.8204 | 0.951 | 0.0995 | 0.0184 | 0.1796 |
| m=100 | 0.9046 | 0.9965 | 0.7851 | 0.898 | 0.0420 | 0.0184 | 0.0706 |
| m=400 | 0.3106 | 0.9991 | 0.8030 | 0.939 | 0.0371 | 0.0184 | 0.0655 |
| m=1000 | 0.3589 | 0.9994 | 0.8765 | 0.959 | 0.0363 | 0.0184 | 0.0732 |
| m=2500 | 0.3863 | 0.9995 | 0.9335 | 0.991 | 0.0542 | 0.0184 | 0.0749 |
| m=6000 | 0.3246 | 0.9996 | 0.9515 | 0.997 | 0.0678 | 0.0184 | 0.0532 |
| m=14000 | 0.3684 | 0.9996 | 0.9495 | 0.998 | 0.1149 | 0.0184 | 0.0564 |
| matched ceiling | 0.2716 | 0.9996 | 0.9770 | 0.995 | 0.0207 | 0.0184 | 0.0655 |

Both readings show an early **transient** in which a few hundred drift transitions leave the model
briefly worse. Past it, the endpoint-Jacobian cosine is monotone in the re-adaptation budget in
every seed while task-probe forecast error keeps wandering. That is worth recording on *this*
substrate specifically: [`../arm_substrate/`](../arm_substrate/README.md) P5/P7 found `fm_err(task)`
non-monotonic here and set the standing rule **"report damage/gain, never a slope against
`fm_err`."** The direction reading does not have the defect that rule was written for. (It is a
more expensive reading — 2·H·n_act simulator rollouts per start state — and it is monotone on one
axis of one substrate, so this is a lead, not a replacement.)

### Data efficiency

Speedup over `rbl_cont`: its executed reaches to a **fixed** endpoint error divided by this arm's,
computed within each seed and then averaged. (Within-seed because BC quality is seed-dependent —
the shared initial policy is 0.2066 / 0.1439 / 0.2100 — so each seed has a different distance to
travel.) `reaches_to_90` cannot answer this question: an arm that plateaus high hits 90 % of its own
smaller gain early and scores well.

| threshold | m=0 | m=100 | m=400 | m=1000 | m=2500 | m=6000 | m=14000 | ceiling |
|---|---|---|---|---|---|---|---|---|
| 0.10 m | 1.71 | 1.59 | 1.76 | 1.25 | 1.28 | 1.64 | 1.16 | 1.69 |
| 0.05 m | 1.60 | 1.29 | 1.42 | 1.38 | 1.47 | 1.49 | 1.60 | 1.66 |

**~1.5×, and flat.** The endpoint-Jacobian cosine spans 0.79 → 0.98 across those columns and the
speedup does not move; Spearman of the outcome against every FM reading is ≈ 0 (`vs jacE cos`
0.000, `vs fm_err` 0.524, neither meaningful at this spread). The reason is in the `sign_w` column:
weighted sign agreement is 0.90–1.00 at every rung, and Garibbo state the requirement precisely —
"successful EBL requires only the correct sign of ∂y/∂a, whereas its magnitude merely scales the
rate of learning." **This ladder never makes the teacher bad enough to matter**, which is why the
capacity ladder below exists.

`mixed_raw@β` — eq. 2's literal sum — runs **1.02–1.47**, barely distinguishable from pure reward,
exactly as the 1/76 norm ratio predicts. Scale-matched `mixed@β` runs 1.25–1.71.

### The control that separates the Jacobian from the forecast

`ebl_imagined` uses the **same Jacobian** and swaps only the error vector. Its speedup is 0.11–1.84
and it frequently never reaches the tighter thresholds; its Spearman against the FM readings is
non-zero where `ebl_sensory`'s is zero. Same middle factor, opposite sensitivity to model quality:
the Jacobian is doing work the forecast is not. Its `imag`/`fcast` columns show the mechanism —
the model's own predicted error falls monotonically (0.1586 → 0.0325) while its forecast error
rises (0.0523 → 0.0724).

### Generalization (matched ceiling; train band 0° ± 30°)

| teacher | 0° | 30° | 60° | 90° | 150° | 180° |
|---|---|---|---|---|---|---|
| `rbl_cont` | **0.0127** | **0.0202** | 0.0759 | 0.1774 | 0.4773 | 0.4317 |
| `rbl_bin` | 0.0159 | 0.0229 | 0.0475 | 0.1064 | 0.3847 | 0.3770 |
| `ebl_sensory` | 0.0241 | 0.0383 | **0.0495** | **0.1048** | **0.3358** | **0.2072** |
| `mixed_raw@0.75` | 0.0112 | 0.0193 | 0.0695 | 0.1739 | 0.4698 | 0.4135 |

Reward wins inside the trained band; the error gradient wins outside it, by **1.7× at 90°** and
**2.1× at 180°**. This is Garibbo Fig. 2e ("the dopamine-driven RBL condition yields poorer
generalisation to novel targets than the cerebellar-driven EBL condition"), and it is the readout
where the two teachers differ most. `mixed_raw` tracks reward and `mixed@0.75` tracks the error
gradient, which is the norm-ratio story showing up behaviorally.

### The aftereffect

The initial motor program deviates **+0.0815** in the field (the direction the curl pushes) and
−0.0135 in the field-free world. After adaptation every arm's field-free lateral deviation is
strongly negative (−0.12 to −0.24) — the **mirror-signed** aftereffect, the canonical behavioural
evidence that what changed is a model rather than impedance. `ebl_sensory` at the stale rung shows
the largest (−0.2418).

### Dysmetria — flipping one component of the believed ∂y_k/∂u_j

The FM is untouched; a fixed `(2, n_act)` sign mask is applied inside the vector-Jacobian product.
Three seeds, matched ceiling, against a healthy 0.0241:

| mask | dist | Δ | radial | lateral | grad_snr |
|---|---|---|---|---|---|
| `flip_yy_u0` | 0.0351 | **+0.0110** | +0.0238 | −0.0219 | 0.579 |
| `flip_yx_u1` | 0.2291 | +0.2050 | −0.0467 | −0.2162 | 0.976 |
| `flip_yx_u2` | 0.2672 | +0.2431 | −0.0978 | −0.2373 | 0.980 |
| `flip_yx_u0` | 0.3553 | +0.3312 | −0.0197 | −0.3443 | 0.980 |
| `flip_yy_u1` | 0.3561 | +0.3320 | −0.0200 | −0.3531 | 0.980 |
| `flip_yy_u2` | 0.4515 | +0.4274 | −0.0813 | −0.4218 | 0.980 |
| `flip_all` | 0.9402 | **+0.9161** | **−0.7652** | **−0.5599** | 0.996 |

Exactly one of six single-component flips is nearly free while the other five cost 0.21–0.43, and
flipping all six costs 0.92 with a blend of *both* deficit types. That is the structure of Garibbo
Fig. 3g–j — three of their four flips map to distinct clinical signs and the fourth has no effect —
generalized from their 2×2 to this plant's 2×3. Our deficits are dominated by the lateral
(displacement) term rather than splitting cleanly into hyper-/hypometria, which the banded reach
geometry and the perpendicular curl would both push toward.

A secondary confirmation falls out of `grad_snr`: every flipped variant reads 0.98 while healthy
reads 0.507 and the inert flip reads 0.579. A *consistently wrong* teacher has a very coherent
gradient — it is driving the policy somewhere definite, just the wrong somewhere — and the one flip
that does not change the teacher does not change that reading either.

---

## The capacity ladder — where the direction reading bites

`fm_hidden ∈ {8, 16, 32, 64, 256}`, matched data, the same shared naive policy cloned from a
full-capacity pre-drift FM in both ladder modes. Three seeds.

| hidden | `fm_err` | **1-step cos** | **endpoint cos** | sign_w | vjp_cos | speedup @0.05 m |
|---|---|---|---|---|---|---|
| 8 | 2.0750 | **0.9888** | **−0.1546** | 0.461 | −0.3644 | 0.20 (1/3 seeds reached it) |
| 16 | 0.9589 | 0.9958 | 0.1632 | 0.668 | 0.0878 | 0.51 (2/3) |
| 32 | 0.5878 | 0.9979 | 0.3642 | 0.767 | 0.3329 | 0.81 |
| 64 | 0.4031 | 0.9992 | 0.6964 | 0.886 | 0.6648 | 1.85 |
| 256 | 0.2716 | 0.9996 | 0.9764 | 0.993 | 0.9712 | 1.62 |

**The one-step Jacobian is near-perfect at every capacity while the composed one goes negative.**
An 8-unit model gets the immediate action dependence essentially right and still produces, after
fourteen compositions, a teaching direction *anti-correlated* with the plant's — sign agreement
0.461, worse than a coin flip. Each step's Jacobian is a matrix and the composed one is their
product, so a small systematic per-step error compounds multiplicatively rather than adding.

The composed reading is the one that orders the outcome — monotone across the ladder, crossing
from worse-than-reward to better-than-reward at sign agreement ≈ 0.8. The one-step reading, which
is what the model was *trained* on and the natural thing to measure, calls all five equally good.

**The optimal β reverses with Jacobian quality**, three seeds:

| | β=0.25 | β=0.50 | β=0.75 |
|---|---|---|---|
| h=8 (sign_w 0.461) | **0.39** | 0.31 | 0.24 |
| h=256 (sign_w 0.993) | 1.55 | 1.66 | **1.67** |

With a bad Jacobian the weight belongs on the reward gradient; with a good one, on the error
gradient. That is eq. 2's claim that β should follow the quality of the sensitivity derivatives,
and Garibbo Fig. 3e–f's specific prediction that dysmetria from inaccurate cerebellar predictions
"may be mitigated by deliberately engaging reward-based learning" — read off a plant rather than
asserted. At the tighter 0.03 m threshold the two high-β arms at h=8 never arrive at all.

---

## Phase C — a committee instead of an oracle

Everything above scores direction against a finite-difference plant Jacobian, obtained through
`set_state`. That is an experimenter's ruler; an agent has no such thing. K=5 forward models in
[`../curiosity_control/`](../curiosity_control/README.md)'s random-prior idiom (trainable net plus
a frozen prior, per-member bootstrap) give the same reading from inside: **cross-member agreement
about the sign of each Jacobian entry**, reward-free and oracle-free.

| rung | dir_agree | sign-correct when **unanimous** | when **split** | lift | mean-Jac cos | oracle vjp_cos |
|---|---|---|---|---|---|---|
| m=0 | 0.9027 | 0.905 | 0.747 | +0.158 | 0.8833 | 0.9104 |
| m=400 | 0.8384 | 0.979 | 0.864 | +0.115 | 0.9471 | 0.9496 |
| m=2500 | 0.9369 | 0.989 | 0.711 | +0.279 | 0.9512 | 0.9652 |
| m=14000 | 0.9491 | 0.994 | 0.659 | +0.336 | 0.9595 | 0.9673 |
| ceiling | 0.9482 | 0.990 | 0.742 | +0.249 | 0.9912 | 0.9942 |

Where the committee is unanimous the entry carries the plant's sign 0.905–0.994 of the time; where
members split, 0.659–0.864. The lift needs no ordering assumption and is the whole claim. Agreement
also orders the rungs roughly the way the oracle does: Spearman(dir_agree, vjp_cos) = **+0.80**,
weighted **+0.90**, and all three committee EBL arms order at ρ = −0.80 against agreement and
−1.00 against the oracle's own `vjp_cos`.

**How to pool a committee.** The endpoint map composes the one-step model H times, so the Jacobian
of the mean model is *not* the mean of the members' Jacobians, and the two teach differently:

| rung | `ebl_committee` (mean VJP) | `ebl_committee_gated` | `ebl_sensory` (through the mean model) | `mixed_agree` |
|---|---|---|---|---|
| m=0 | 0.0338 | 0.0432 | 0.0498 | **0.0316** |
| m=400 | 0.0284 | 0.0286 | 0.0328 | **0.0202** |
| ceiling | 0.0222 | 0.0224 | 0.0225 | **0.0212** |

Averaging the members' *teaching vectors* beats planning through the mean model, by most where the
model is worst (1.5× at the stale rung), and `mixed_agree` — β read off the committee's own
per-component agreement instead of swept — is best at every rung. That is
[`../committee_head/`](../committee_head/README.md)'s question asked of **credit** rather than
of allocation.

---

## Two readings that should not be collapsed

Izawa & Shadmehr report higher trial-to-trial variability under reward alone, and Garibbo reproduce
it from the RBL action gradient's higher variance. Here the two halves of that go opposite ways:

- **Gradient noise** (`grad_snr`, the cosine between two half-batch mean action gradients, read in
  an early window): reward **0.15**, error-based **0.99**. The reward gradient is close to pure
  noise per batch, which is the mechanism Garibbo name.
- **Plateau variability**: reward **0.0018**, error-based 0.003–0.018. Reward is the *steadier* arm
  at plateau, because it settles while the EBL arms oscillate under a fixed Adam step on a gradient
  that shrinks with the error.

**Scope line.** σ is fixed and shared across every arm in this port, so executed trial-to-trial
variability is identical by construction and the Izawa–Shadmehr readout proper is not available
here. A learned σ_φ(h) is the one piece of Garibbo's policy this port does not implement.

`grad_snr` is read early on purpose: at convergence the batch-mean gradient is near zero and the
half-batch cosine swings freely (`ebl_sensory` runs +0.996, +0.999, +0.998 early and then −0.842,
+0.983, −0.932 at plateau). For the same reason the headline level is the **plateau mean** and not
the final readout, with the late instability reported separately as `divergence`.

---

## Reproduction

```bash
cd experiments/
# Phase A — the gates (run first; it fails the run on a fork mismatch)
modal run --detach mjc/jacobian_teacher/phase_a.py::phase_a --tag pa_s0 --seed 0

# Phase B — the re-adaptation ladder
for s in 0 1 2; do
  modal run --detach mjc/jacobian_teacher/jacobian_teacher.py::jacobian_teacher \
      --tag jt_s$s --seed $s
done
# the capacity ladder: a second run on a second axis, not a cross-product
for s in 0 1 2; do
  modal run --detach mjc/jacobian_teacher/jacobian_teacher.py::jacobian_teacher \
      --tag jt_cap_s$s --seed $s --capacity-ladder "8,16,32,64,256"
done

# Phase C — the committee
modal run --detach mjc/jacobian_teacher/committee.py::committee --tag jc_s0 --seed 0

python3 mjc/jacobian_teacher/jacobian_teacher_figure.py --tags jt_s0 jt_s1 jt_s2 --out agg_transition
python3 mjc/jacobian_teacher/jacobian_teacher_figure.py --tags jt_cap_s0 jt_cap_s1 jt_cap_s2 --out agg_capacity
python3 mjc/jacobian_teacher/committee_figure.py --tags jc_s0
```

Results mirror to `/data/jacobian_teacher/<tag>/` on `mujoco-control-data` and to
`figures/<tag>/`. `--quick` gives a smoke on every entrypoint; smoke settings use the pusher-tuned
`k_shoot=256, cem_iters=4` that [`../arm_substrate/`](../arm_substrate/README.md) P3 shows flattens
this substrate's slope, and they inverted both gate 2c and gate 4 relative to the full runs — smoke
numbers here are diagnostics that the code runs, never results.

## Scope

Single arm family, contacts off, one drift, one reach-band geometry, terminal feedback only.
Garibbo's derivation is for endpoint reaches with terminal feedback and the fork's reach is the
same shape, so this is the friendly case. Because both the error and the reward signal are terminal
on the same committed program, [`../two_clocks/`](../two_clocks/README.md)'s separate-window law
does not bite — there is no delay mismatch between two channels of different temporal support —
and the two-channel form was deliberately not built.

## Open

- **The direction horizon.** Sweep H against capacity to locate where composed direction accuracy
  fails, as [`../arm_substrate/`](../arm_substrate/README.md) P4 located the composition horizon
  (n=3: 20–23 steps). Forecast horizon and direction horizon need not be the same horizon, and only
  the first has been measured here.
- **One axis or two?** Does `endpoint_jac`'s direction accuracy predict the *ballistic transmission
  slope* — i.e. do the planner's degradation and the teacher's share an axis?
- **A learned σ**, which would recover the Izawa–Shadmehr variability readout proper.
- **Phase C at more seeds**, and the committee under the aleatoric noise `curiosity_control`
  Finding 3 shows derails member disagreement on this substrate.

[^private]: Not mirrored: this link points to a document in the private lab repo (the roadmap, the queue, an unrun spec, reading notes, or a conversation). See the top-level README for what is held back and why.
