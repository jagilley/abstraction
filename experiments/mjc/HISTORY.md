# History of `mjc` — a big-picture reading

Scope: [`experiments/mjc/`](.), generated fresh from the READMEs, `FILES.md`, and git history
(~33 commits, 2026-07-16 → 2026-07-31). Not a changelog — an attempt to trace the turns in thinking.

## Why this node exists

`mjc` was founded as a third controllable-dynamics substrate, one rung of realism above RHM/MNIST
and below language — "a DGP whose knobs we set and sweep, not a robotics benchmark to RL a policy
to SOTA on" ([README.md](README.md)). The founding discipline — knob sweeps with a vanilla
baseline, dissociations not leaderboard numbers, single-seed-first — is inherited whole from
`rhm`/`a2a_forward` and never relaxed; almost every "surprising positive" below is later revealed
to be a confound, and the node's real skill turns out to be catching that quickly.

## Foundations: three cuts establish the substrate's own idiom (2026-07-16)

[Cut #1 (contact residual)](contact_residual/README.md), [Cut #2 (arity)](arity_torque/README.md)
and [Cut #3 (operator intervention)](dynamics_shift/README.md) port the a2a/RHM arity and residual
results onto real contact physics, but Cut #3 is the pivotal one: MuJoCo lets you **intervene on
the transition operator while holding the task fixed** — a move language/RHM cannot make. A
model-based agent re-adapts to a shifted operator from ~50 reward-free transitions while a
model-free policy needs ~240× more reward-labeled data, but only once the controller **commits**
open-loop (`replan_every` ablation) — per-step replanning lets feedback substitute for the model
and the effect vanishes. This single knob's biological reading — *`replan_every` ≈ how ballistic a
movement is* — is filed away almost as an aside, then becomes the causal spine of the whole second
half of the node's history.

## The value↔FM interface arc, attempted bottom-up, then abandoned for phenomenon-first (07-18)

[`value_shaping/`](value_shaping/README.md) isolates the *learning-layer* claim (a value signal
re-allocates a capacity-limited FM's effort) as a stationary control, deliberately built to be inert
about meta-learning. [`directed_readapt/`](directed_readapt/README.md) then asks whether a
value-directed *collection* drive speeds re-adaptation and returns two clean negatives: a global
shift has no scarcity to exploit, and a localized shift creates scarcity but ensemble disagreement
is **blind to a confident-prior shift** (members agree on the stale answer, so disagreement is
lowest exactly where the model is most wrong). This is the node's first documented strategic pivot:
verifying mechanistic atoms one at a time keeps either killing the phenomenon or leaving an
already-good baseline, so the decision (dated in the README itself) is to go **phenomenon-first** —
build the actual compounding-adaptation loop and back-translate the mechanism from what works,
rather than pre-verify every ingredient.

[`meta_adapt/`](meta_adapt/README.md) (Cuts #4→#4e) is that phenomenon, run end to end on one
substrate: a stationary damping family **collapses** (meta = multitask, the RHM anchor reproduced
on control), while an input-coupled actuator-rotation conflict **opens the gap monotonically** —
the same order parameter crossing a transition. An explicit context latent `z` then makes the outer
memory legible (system-ID R²≈1) and reveals **system-ID ⊥ adaptation-benefit** — knowing the task
is not the same as it mattering. A value-of-information collection drive is a **robust,
scarcity-gated negative** (over-engineering for 1-D identification). Finally, #4d/#4e closes the
causal loop the idea doc names: a reward-*driven* outer loop, with no hint about which dims matter,
**rediscovers** the value-relevant re-allocation on its own — but only under genuine capacity
competition, with a wireheading gotcha (unnormalized weights let the loop win by shrinking loss
scale rather than re-allocating).

[`curiosity_control/`](curiosity_control/README.md) lands the mirror-image *afferent* result: an
intrinsic reducible-surprise drive tracks a moving frontier and helps specifically because the
frontier moves (ties on a static one) — but it also **inverts a prior finding**: on this low-dim
control task, disagreement chases aleatoric noise (a noisy-TV pathology), the opposite of the
active-vision result it descends from. The fix isn't a smarter filter, it's grounding — explore and
exploit as two additive drives with an interior optimum. The recurring lesson crystallizing here:
which drive/instrument wins is set by the substrate, not fixed a priori.

## The teacher problem: two failed "fully online" attempts become a positive result (07-21)

[`online_value_loop/`](online_value_loop/README.md) tries to close both taps (explore-balance,
capacity-allocation) as one continuously-running system and is **obstructed on both**, for two
different structural reasons — CEM-MPC replanning is robust to the drive's tracking effect, and the
allocation benefit is a slow, commit-dependent transient invisible to a fast local probe. Rather
than a failure, this is read as a **positive confirmation** that value is a slow/committed,
offline-estimable quantity, and that its payoff is *adaptation speed*, never converged competence —
a reframe that quietly resolves the earlier "washout" worry (under perpetual drift you never
converge, so the fast-sprint regime is permanent).

[`drift_value_loop/`](drift_value_loop/README.md) runs the compounding experiment this named and
finds the compounding is real but **relocates its source**: it comes from the fast *memory*
(a context latent that consolidates an invariant core), not the slow value-carving, which is
capacity-gated and control-robust even under drift. The deeper diagnosis: **control is a
near-blind grader of the value** — a replanning controller reaches goals about as well with a
stale model as a fresh one. Grading the meta-loop by the FM's own prediction error (the literal
cerebellum→VTA "prediction-error messenger") instead of control reward gives the explore/exploit
balance a clean interior optimum. This "the missing piece was the *teacher*, not the value
*structure*" line becomes load-bearing for everything after it.

## Ballistic control: the blind-grader finding becomes a fact about control mode (07-22)

[`ballistic/`](ballistic/README.md) reframes the drift-value-loop obstruction as a property of
**replanning**, not of value: replanning is blind because it re-grounds every step. A **ballistic**
(feedforward, committed) controller — the biologically realistic regime, since sensorimotor delay
rules out replanning faster than ~50-150ms — makes the FM ~3× more behaviorally load-bearing, and
reward-free re-adaptation restores ballistic competence ~4.3× more than reactive after a drift. The
node's thesis statement crystallizes here: the cerebellar forward model is simultaneously the
**reward-free-maintainable** asset and the **feedforward-critical** asset — the same object.
[`ballistic/directed/`](ballistic/directed/README.md) then tries to complete the arc (does the value
drive choose *where* to collect, in a loop?) and instead **retracts its own inner-loop positive**
after an audit: the loop's grader was the one this tree had already shown is blind, the metric was
62%-contaminated by post-recovery plateau, one seed flipped the aggregate, and — the deepest cause —
every policy got a **22× measurement subsidy** of free teleported probes to decide where to spend a
metered budget. "Where should I collect" is unanswerable when looking is free and global. This
audit (explicitly: re-read stored JSON, no re-run) is a second strategic turn: publish, then verify
your own published finding survives its own logic, and retract in place rather than delete.

## The collection-realism turn (07-24)

That retraction produces a standing memo, [`COLLECTION_REALISM.md`](on_policy/COLLECTION_REALISM.md),
and [`on_policy/`](on_policy/README.md) builds the fix: an embodied `Body` that meters every
`env.step` and **cannot expose `set_state`**, so teleporting is unrepresentable rather than
discouraged. Pricing it (E0-E2) delivers a genuine surprise: most of on-policy's apparent advantage
was **not embodiment** but a mistuned teleport sampling knob (an oracle teleporter told where to
look closes the same gap). What survives as truly structural is command-state entanglement from
continuity itself — quietly retroactive on Cut #2's "i.i.d. commands" premise, now flagged as a
teleport-only luxury. The clean summary: **embodiment is nearly free when drift structure is
global, and costs ~2.5× when it's local** (and naive teleport fails outright there) — which is also
exactly the regime where "where to collect" becomes askable again.
[`on_policy/directed_on_policy/`](on_policy/directed_on_policy/README.md) (E3) re-attempts the
retracted directed-collection cut with looking itself metered (1.84×, not 22×) and the claim
**resolves positive**: relevance and reducibility both pay, matching the privileged oracle. On-policy
collection becomes the standing convention for new cuts ([README.md](README.md) collection box).

## A second task family, built to retire "single-family" (07-22/23)

[`arm_substrate/`](arm_substrate/README.md) replaces the pusher's manufactured, bolted-on capacity
competition (six hand-tuned `qfrc_applied` perturbation layers) with competition intrinsic to a
real chain plant — a passive tool spans #4d's capacity-competition boundary as one physical knob
measured in kilograms, and `goal_site` flips value-relevance with the transition operator held
byte-identical, a purer version of Cut #3's "intervene on the operator" move. Porting Cut 4c onto
it (`ballistic/arm/`) reproduces the ballistic dissociation (5.24× vs the pusher's 4.3×) and buys a
readout the pusher structurally cannot produce: a **mirror-signed aftereffect** — the canonical
Shadmehr signature that adaptation is a model update, not stiffening.

## Expansion: the flagship question, closed against its own motivating premise (07-27)

[`expansion/`](expansion/README.md) — billed at the node's founding as its biggest open question —
finally runs, and needs three calibration pieces (not one) because rank-shaped instruments had
already failed three times elsewhere in the repo. The answer: drifting a plant's physics does
**not** open the FM's representational frontier (±0.08 directions against a +0.72±0.42 calibration
from a genuine support increase — ~10× smaller). The node explicitly **retires its own founding
motivation** — "drifting dynamics is a novelty generator" — while keeping drift as essential for
realism; a fixed-DOF motor plant has no hierarchy to expand into, and the belief's real test moves
to `rhm`. This is the clearest instance of the node correcting a premise it had held since its
first README, in writing, rather than quietly dropping it.

## Metered repair: cross-pollinating instruments with `rhm` (07-29/31)

[`on_policy/metered_repair/`](on_policy/metered_repair/README.md), the most recent work, explicitly
ports two `rhm/directed_sculpting` findings onto the arm. One transfers exactly (E3's
learning-progress tap is the same fixed-budget estimator RHM found inverted, for the same
data-starvation reason), but the repair it motivates **does not cash out** in control here. The
node instead finds a genuinely new result neither substrate alone could produce: with a
**visited-but-irreducible** cell placed (a geometry neither RHM's nor E3's original ladder had),
relevance-alone becomes the *worst* arm, and only the reducibility×relevance conjunction rescues
it — closing a mirror-image gap in both substrates' prior designs. E5 finds a clean, mechanistic
null: a fixed-DOF plant with spatially-disjoint drift gives an MLP no route to shared-parameter
inference, so a "does metered data migrate repair upward" question literally cannot be posed here
— extending expansion's "no hierarchy to expand into" conclusion to the necessity question. A late,
honest loose end (§4d in that README) is left explicitly unresolved: no process variable (leak,
share, allocation variance) predicts final error, and a bursty-vs-steady confound is flagged as the
next thing to rule out before trusting the ladder further.

## Standing epistemic patterns worth naming

- **Retraction-in-place, not deletion.** At least three findings are published, then formally
  re-scoped after an internal audit (disagreement-drive ranking, S2's directed-collection null, the
  `R_act/R_comp/R_res` expansion readout) — always by re-reading stored data, and always left in the
  document with the correction layered on top rather than erased.
- **The recurring confound is "free, disembodied looking."** Corridor-geometry drive confounds,
  the 22× S2 subsidy, and the teleport-mistuning in E0 are three faces of one thing this node
  chases across half its history before naming it as a standing memo.
- **Composition horizon (~6-8 steps) reappears** across Cut #3, the arm (14-23 steps, scaling with
  chain length), and expansion's readable window — a substrate-general property, not a pusher
  artifact.
- **Instruments are built to fail loudly.** `verify_backcompat.py`, the expansion calibration
  triple, and metered_repair's "gate = strict prefix of the real run" are the same idea maturing:
  certify the instrument on data whose answer you already know before trusting it on data you don't.
