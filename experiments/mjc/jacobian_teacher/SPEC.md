# SPEC — jacobian_teacher: the forward model in backward mode

**The question in one sentence**: when a sensory endpoint error is pulled back through the forward
model's action Jacobian and used as the teaching signal on a committed motor program, does that
teach faster, with less variance, and with better generalization than reward alone — and does the
advantage track the *direction* accuracy of the Jacobian rather than the FM's forecast accuracy?

**Status**: spec, 2026-09-03. Nothing run. **Parent**: [`../README.md`](../README.md) (mjc).
**Reading**: Garibbo, Filipe, Aitchison & Costa 2026[^private]
— the action-gradient framework, eqs. 1–2: reward-based and error-based updates as gradients on
one policy, the cerebellum supplying the sensitivity derivatives `dy/da`, a weight β mixing the
two. Also Baladron et al. 2023[^private] for the contrasting
dual-policy account and the aiming-error-not-task-error result.
**Attribution**: the prompt is Jasper's (2026-09-03) — that the record has used the FM's action
dependence as a *magnitude* (the agency gate) and as a *simulator* (CEM) but never its *direction*
as a teacher, and that this matters given the repo's finding that directions are reliable where
scalars are not. The substrate choice and the arm ladder came out of the exchange.

## What the record already has, and the one thing it does not

Three uses of the same arity-2 FM exist. Forward: roll commands through it and pick the best
rollout ([`../ballistic/`](../ballistic/README.md), CEM over the H×n_act sequence). Error: compare
its forecast to what happened ([`../plasticity_gain/`](../plasticity_gain/README.md),
[`../agency_gate/`](../agency_gate/README.md) — the arity gap `‖FM₂(s,u) − FM₁(s)‖` is the *norm* of
the action dependence). Backward, differentiating it with respect to the command and using the
result to assign credit to a policy, has never been built. Garibbo's cerebellum is exactly that:
`∂e/∂φ = (∂e/∂y)(∂y/∂u)(∂u/∂φ)`, sensor supplies the first factor, policy the third, the FM the
middle one. Reward-based learning gets the same direction by correlating outcome with exploration
noise, which is why [`../dynamics_shift/`](../dynamics_shift/README.md) measured its model-free arm
needing ~240× more reward-labelled data to reach the ceiling the model-based arm reached from ~50
reward-free transitions.

Four findings make the direction the interesting object here:

- **Directional, not scalar.** The repo's recurring pattern: an additive scalar in the stream is
  inert while directions carry the content ([`EMOTION_INJECTION`](../../a2a_forward/EMOTION_INJECTION_README.md),
  [`REPRESENTATIONAL_DIVERGENCE`](../../a2a_forward/REPRESENTATIONAL_DIVERGENCE_README.md), the
  conditional-revision line). The Jacobian is a direction in action space; a reward is a scalar.
- **Value-shaped beats veridical.** [`REACHING_LOOKAHEAD`](../../a2a_forward/reaching/REACHING_LOOKAHEAD_README.md)
  and [`../meta_adapt/`](../meta_adapt/README.md) #4d/#4e. Garibbo state the requirement form:
  error-based learning needs only the correct *sign* of `dy/da`, magnitude scales the rate.
- **Stale models bite ballistically.** A wrong model's signature grows with commitment horizon
  ([`../ballistic/`](../ballistic/README.md) 4b; [`../expansion/`](../expansion/README.md) 1.4× → 2.7–4.1×
  at the ballistic horizon). Their dysmetria result — flip one Jacobian sign, get hypermetria or
  hypometria — is the same signature read through a teacher rather than a planner.
- **Contact is where a first-order signal misleads.** [`../contact_residual/`](../contact_residual/README.md)
  concentrates the FM's error at the stiff map; the arm has contacts off by default, which is the
  right place to start.

## The design

**Substrate: [`../ballistic/arm/arm_readapt.py`](../ballistic/arm/arm_readapt.py), forked verbatim.**
The n=3 curl design point, the Shadmehr drift `b0=0 → b1=6`, `H=14`, one-step FM `f(s,u) → Δs`,
`fk_torch` (already differentiable), the CEM planner, the task-distribution probe, the matched-FM
ceiling trained first, the aftereffect readout, and the behavior-cloned motor program
`π(s, g) → H×n_act` (tanh) — that policy network is the object every teacher below trains. Gate the
fork with bit-identity on a fixed command sequence. The reward-based arm's precedent is
[`../dynamics_shift/dynamics_shift.py`](../dynamics_shift/dynamics_shift.py) §5, the minimal
REINFORCE on a committing motor program (Gaussian exploration, EMA baseline, reward = −final
distance); port it, do not improve it — only the slope matters.

**Three teachers on one policy, same executed-reach budget, same drift.**

1. **`rbl`** — REINFORCE as above. Garibbo's eq. 1: `δ·(u − μ)/σ²` with `δ = r − v`. Two reward
   variants, continuous (−distance) and binary (in the reward zone), since Izawa & Shadmehr's
   phenomena are measured under the binary one.
2. **`ebl_sensory`** — the new object. Execute the program on the plant, read the realized
   endpoint `y`, form `e = y − y*`. Roll the *frozen* FM from the true start state along the
   *executed* program to get `ŷ(u)`, and take the vector-Jacobian product `eᵀ · ∂ŷ/∂u` by autograd.
   That is the sensory error in action coordinates. Backpropagate it through `μ_φ` and step φ.
   The FM never sees the sensory error; it only supplies the coordinate transform.
3. **`ebl_imagined`** — the control that separates the Jacobian from the forecast: the same
   gradient but with the *model-predicted* error `ŷ − y*` in place of the realized one. This is
   gradient-based planning amortized into the policy; it uses no sensory feedback at all. If
   `ebl_sensory` and `ebl_imagined` come apart on a stale FM, the Jacobian is doing work the
   forecast is not.

Plus **`mixed@β`** — β·`ebl_sensory` + (1−β)·`rbl` summed on the same parameters, β swept
(Garibbo's eq. 2). Both signals are terminal here, so [`../two_clocks/`](../two_clocks/README.md)'s
separate-window law does not bite; say so in the README rather than building the two-channel
form. And the incumbents from the fork, `ballistic_cem` and `ballistic_bc`, which are zeroth-order
uses of the identical FM, plus `reactive` as the blind-grader control.

**The axis: FM quality, read two ways.** Take the fork's ladder — the stale pre-drift FM, the
re-adaptation milestones (400 … 14000 reward-free transitions), the matched ceiling — and add a
capacity ladder (`fm_hidden`). For every FM report *both*: the forecast error on the task probe
(the record's scalar), and the **Jacobian direction accuracy** against the plant's true action
Jacobian, obtained by central finite differences on the simulator from the same start states
(`set_state` is the experimenter's ruler here, as in every exogenous-axis cut). Report cosine and
per-component sign agreement. Then train each teacher through each FM and ask which of the two
readings the learning outcome follows.

**The dysmetria control.** Flip the sign of one component of the FM's Jacobian (a fixed
sign-mask on the vector-Jacobian product, the FM otherwise untouched) and train `ebl_sensory`.
Garibbo Fig. 3g–j predict a specific directed mis-reach per component; the fork's signed lateral
readout is built for exactly this. If a sign-flipped teacher mis-teaches in the predicted
direction, the mechanism is doing what it claims; if it does not, the gradient is not reaching
the policy the way the derivation says.

**Committee (Phase C).** K FMs in the [`../curiosity_control/`](../curiosity_control/README.md)
random-prior idiom. Teacher variants: mean vector-Jacobian product; agreement-gated, where each
action-space component is scaled by the members' sign agreement on it. Readout: does direction
agreement across members predict, per FM on the ladder, when `ebl_sensory` helps — the same
question [`../committee_head/SPEC.md`](../committee_head/SPEC.md) asks of allocation, asked here
of credit. β read off that agreement rather than swept is the trained-head seat in this cut.

## Phase A — gates before any treatment

1. **Fork fidelity**, as above.
2. **The Jacobian oracle is trustworthy.** Finite-difference plant Jacobian against `mj_jacSite`
   for the kinematic half (`fk` is already verified to 2e-16), and step-size sweep for the dynamic
   half. Then the **matched-ceiling FM's Jacobian against the oracle**. If a forecast-accurate FM
   does not have a direction-accurate Jacobian, that is the first finding of this node and it
   changes what Phase B can mean; record it before anything else.
3. **The reward arm reproduces** `dynamics_shift`'s slope on this substrate.
4. **`ebl_imagined` at the matched FM ≈ `ballistic_cem`.** Gradient planning and sampling
   planning should agree when the model is right. If they do not, the composed rollout or the
   sizing is wrong (the fork's `k_shoot`/`cem_iters` warnings apply).

## Phase B — the ladder

Teachers × FM ladder × the dysmetria control, single seed first, then seeds per
[`../../CLAUDE.md`](../../CLAUDE.md). Readouts, all from the fork or one line from it:

- adaptation curve — endpoint error against executed reaches, and reaches-to-ceiling
- trial-to-trial variability at plateau (Izawa & Shadmehr: higher under reward alone)
- generalization to held-out goal directions (train on one band of `eval_geometry`, test on the rest)
- the aftereffect in the field-free world, signed against the curl
- for every cell, the two FM readings it was trained through, so the outcome can be plotted
  against forecast error and against Jacobian direction accuracy side by side

## Why the arm and not RHM

The Jacobian is a continuous-plant object, and the arm supplies its exact oracle by finite
differences the way RHM supplies belief revision by belief propagation. RHM's actions are discrete
edits; the analog needs a differentiable relaxation of the belief update, which is a second
question. Port it there only if the arm result gives a reason to.

## Things to keep honest

Garibbo's derivation is for endpoint reaches with terminal feedback; the fork's reach is the same
shape, so this is the friendly case. The policy is a behavior-cloned motor program, which
[`../ballistic/`](../ballistic/README.md) found transmits FM quality the most; that is a feature
for the question and a scope on the answer. Single family, contacts off, one drift. Per repo norms
no outcome is interpreted here.

## Housekeeping

Node at `experiments/mjc/jacobian_teacher/`; README + `FILES.md` at writeup, row in
[`../FILES.md`](../FILES.md). Results under `/data/jacobian_teacher/<tag>/` on
`mujoco-control-data`. FM training and probes may use the fork's teleport pools (exogenous-axis
cut); the *policy's* learning data are executed reaches by construction. Launch each seed as its
own client. Invoke `/run-experiment-on-modal`[^private]
before running; subagents follow `/subagent-instructions`[^private]
and do not touch git state.

[^private]: Not mirrored: this link points to a document in the private lab repo (the roadmap, the queue, an unrun spec, reading notes, or a conversation). See the top-level README for what is held back and why.
