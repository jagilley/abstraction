# History of `mjc/` — the physical-control substrate

*Auto-generated big-picture retrospective. Overwrite freely; see [FILES.md](FILES.md) and per-node
READMEs for the ground truth this compresses.*

## Why this node exists

Proposed 2026-07-16 in [`ideas/physical_control_substrate.md`](../../ideas/physical_control_substrate.md)
as a third DGP rung between RHM (discrete, known latents) and language (continuous, opaque): MuJoCo
used the RHM way — **knobs you set and sweep**, never RL-to-SOTA. The one fork that was declared load-bearing
at the outset (§"The one fork that decides whether this is worth it") held for the node's entire life:
scripted/MPC/BC policies only, every deliverable a slope or dissociation against a vanilla baseline,
never a leaderboard number. That discipline is why the arc below reads as one continuous argument
rather than a pile of unrelated demos — every cut inherited the same variable-control instinct RHM
trained into this program.

## Phase 1 — foundations (2026-07-16–18): the substrate proves itself

Three cheap, single-afternoon cuts ported the program's existing claims onto real contact physics and,
more importantly, established the machinery everything later reused. [`contact_residual/`](contact_residual/README.md)
(Cut #1) found the FM residual is an **event** detector, not a state detector — spiking at the
free→contact transition and decaying even while contact persists — and surfaced the first of several
recurring gotchas: Huber loss, not MSE, or a heavy contact tail starves free-flight learning.
[`arity_torque/`](arity_torque/README.md) (Cut #2) reproduced the arity-2≻arity-1 result on a genuine
actuator under i.i.d. commands, the cleanest version of that claim anywhere in the repo. [`dynamics_shift/`](dynamics_shift/README.md)
(Cut #3) is the node's spine: MuJoCo's unique move — **intervene on the transition operator, hold the
task fixed** — showed a model-based agent re-adapts to a shifted operator from reward-free interaction
while a model-free policy needs ~240× more reward-labeled data, *gated by re-groundability*
(`replan_every`). That last knob's biological reading — "how ballistic is the movement" — is the seed
every later arc grows from.

## Phase 2 — the value↔FM interface, mechanism-first (07-18–20)

[`value_shaping/`](value_shaping/README.md) established the **learning-layer** control in isolation
(stationary): value re-allocates a capacity-limited FM away from a reducible-but-irrelevant factor.
[`directed_readapt/`](directed_readapt/README.md) then ran two directed-collection negatives — a global
shift has no scarcity to exploit; a localized shift has scarcity but **disagreement is blind to a
confident-prior shift** (it flags *uncertainty*, not *confident-wrongness*) — and used both to make an
explicit, dated pivot (2026-07-18): stop verifying isolated mechanistic atoms one at a time and instead
chase the actual meta phenomenon (compounding across a drift sequence), backfilling mechanism from
whatever works. This pivot is the clearest single inflection point in the node's history — nearly
everything after it is phenomenon-first.

[`meta_adapt/`](meta_adapt/README.md) (Cuts #4→#4e) is the arc that pivot bought. The meta−multitask
gap **collapses to zero** on a 1-D damping family (the RHM anchor reproduced on control) and **opens
monotonically** under an actuator-rotation conflict (textbook MAML crossover, ~8× at Φ=π/2) — establishing
conflict, not exposure, as what meta-learning needs. #4b made the outer memory dissectible (a context
latent `z` that decodes the task parameter at R²≈1) and showed **system-ID ⊥ adaptation-benefit** — a
dissociation that recurs, reworded, for the rest of the node. #4c found value-directed identification is
a **robust negative**: under low-dimensional system-ID, disagreement-driven active collection never
beats a trivial max-‖u‖ heuristic, because the most informative command is also the largest one. #4d/#4e
closed the idea doc's discriminator 4 end-to-end: a reward-*driven* outer loop (no hint about which
dims matter) rediscovers the value's support, but **only under capacity competition** — control-neutral
otherwise. Two reusable methodological finds surfaced here: an unnormalized per-dim weight lets an outer
loop win by gaming loss *scale* rather than allocation (fixed by a fixed-budget softmax — a concrete
"on-manifold veto" against wireheading), and a CEM optimizer's *converged distribution* is the unbiased
readout, not its cherry-picked single best.

[`curiosity_control/`](curiosity_control/README.md) built the **afferent** (explore) half to #4d/#4e's
efferent half: a reducible-surprise drive tracks a moving frontier and the benefit is specifically
non-stationary (ties on a static one) — but pure curiosity **falls for the noisy TV** on low-dim control
(disagreement chases aleatoric noise, inverting the active-vision result), and grounding it with the
exploit/value signal as a second additive drive — not a gate — cures it, with an interior optimum. The
`e`-tap/`p`-tap-as-one-system piece the idea doc asked for landed here.

## Phase 3 — the fully-online loop, and the productive obstruction (07-21)

[`online_value_loop/`](online_value_loop/README.md) tried to build the idea doc's keystone — reward
continuously shaping a *live* FM — on both levers, and both were **obstructed, for two different
reasons**: CEM-MPC control turns out to be robust to the explore drive's tracking effect (no gradient),
and the efferent value-shaping benefit is a slow, commit-dependent, pre-convergence transient invisible
to a fast local probe. Rather than a dead end, this became the node's central reframe: **value is a
slow/committed/offline-estimable quantity — the two-timescale split is forced, not chosen** — and its
payoff is *adaptation speed*, never converged competence. Everything downstream reads as a consequence
of taking that reframe seriously.

[`drift_value_loop/`](drift_value_loop/README.md) ran the experiment the reframe implied — compounding
across a drift sequence — and **reassigned the source**: compounding is real (a context-latent memory
climbs 0.31→0.97 few-shot R² across drifts; pooled memory is a *liability* under conflict), but it comes
from the fast learning-layer memory, not the slow value-carving, which is capacity-gated and control-robust.
The deeper diagnosis: **control is a near-blind grader** of FM quality (a replanning controller reaches
goals about as well stale as fresh), so the meta-loop's apparent inertness was never a value-structure
problem — it was a *teacher* problem. Re-grading by the value-relevant **FM prediction error** (the literal
cerebellum→VTA messenger) gives the explore/exploit balance a clean interior optimum a control-graded loop
could never see. This "grade by FM error, not control" correction is the node's single most-reused lesson
— it recurs verbatim in `ballistic/directed`'s S2 audit and in `on_policy/directed_on_policy`'s E3.

## Phase 4 — the efferent bridge, and the collection-realism reckoning (07-22–24)

[`ballistic/`](ballistic/README.md) resolved "control is a blind grader" as a fact about the *controller*,
not the value: a **ballistic** (feedforward, committed) controller — the biologically necessary regime,
since you cannot replan faster than sensorimotor delay — makes the FM ~3× more behaviorally load-bearing
than reactive, and online reward-free re-adaptation restores ballistic competence ~4.3× more than reactive
after a drift. The lead reading crystallized here and stuck: the cerebellar FM is simultaneously the
**reward-free-maintainable** asset and the **feedforward-critical** one — the same object.

Its child, [`ballistic/directed/`](ballistic/directed/README.md), tried to close the last open piece —
does directed collection pay in a loop — and instead produced the node's most important **retraction**.
An initial null (the relevance term never beats reducibility-chasing alone) was audited and reversed: the
grader was the one already shown blind, the metric was 62% contaminated by post-recovery plateau, one
seed flipped the mean, and — the structural finding — every policy spent **2,240 free teleported
transitions/round to decide where to spend a budget of 100**, a **22× measurement subsidy** that
demotes "where should I look" to a tiebreaker. The audit produced the standing node-level memo
[`COLLECTION_REALISM.md`](on_policy/COLLECTION_REALISM.md) (2026-07-23): every FM in this node had been
trained on teleported, omniscient, free, cheap-to-measure data, and that unrealism was silently load-bearing
well outside the place it was originally chosen for. It sorted every prior claim into three tiers — safe
dissociations (A), inflated absolute sample-counts (B), and outright unaskable allocation questions (C) —
and recommended `collection_mode: teleport | on_policy` as a flag, not a second substrate, with the honest
cost stated up front: on-policy collection **couples model quality to data quality**.

[`on_policy/`](on_policy/README.md) built and priced that flag (E0–E2, 3 seeds each). The most humbling
result came first: most of on-policy's *apparent* advantage (E0) was not embodiment at all but a
**mistuned teleport sampling knob** — an oracle teleporter told where to look closed the same gap. What
*is* structurally embodied is command-state entanglement (continuity itself, not goal-directedness).
Under the arc's existing **global** drift (E1), embodiment turned out nearly free — Cut 4c-arm's
headline reproduced under every collection mode — refuting the memo's own hypothesis that on-policy
would stretch a step-like recovery into a curve. Only once the drift was made **local** (E2) did
embodiment become genuinely load-bearing (2.5× more transitions; broad teleport never repairs the
region), via a mechanism the memo had only guessed at: not "coverage grows" but **bootstrap data
quality** — the data gathered while the model is wrong is itself wrong-distributed. That local-drift
regime is what finally made [`on_policy/directed_on_policy/`](on_policy/directed_on_policy/README.md)
(E3) askable, and there the retracted relevance claim **reproduced positive**, 3/3 seeds, once both
collecting and monitoring were embodied and metered (ratio 1.84×, not 22×).

In parallel, [`arm_substrate/`](arm_substrate/README.md) retired the arc's other standing caveat —
single-family — by replacing the pusher's six *manufactured* perturbation layers with capacity
competition intrinsic to a real plant (`M(q)q̈+C(q,q̇)q̇+Dq̇=τ`), independently re-deriving both the
value-relevant-teacher correction and the smooth-not-localized quality-axis rule, and adding readouts
the pusher structurally cannot produce (a mirror-signed Shadmehr aftereffect).

## Phase 5 — the flagship revisited, and closed (07-27)

[`expansion/`](expansion/README.md) (Cut #5) finally ran the idea doc's original ambitious extension —
does drifting dynamics act as a novelty generator, expanding the FM's representable directions — after
sitting unrun for months. Because rank-shaped instruments had already failed three times in this repo
(this node's own Cut #1 eff-rank reversal; language's residual-rank/noise confusion; RESIDUAL_RANK's flat
17× range), the cut was built as **calibrate → measure → calibrate**: Piece 1 showed the instrument is
not saturated here and separates a known-wrong FM cleanly; Piece 3 calibrated a genuine positive (DOF
accretion, +0.72±0.42, 3/3 seeds); Piece 2 then measured drift against that calibrated unit and found
**±0.08 directions — ~10× below a known-real effect, de-confounded across drift geometry**. Drift moves
the target function; it does not grow the space of representable functions. This directly falsified the
"drifting dynamics is a novelty generator" motivation in the founding idea doc (now marked retired there)
and confirmed [`beliefs/dimensionality_expansion.md`](../../beliefs/dimensionality_expansion.md)'s scope
claim that a fixed-DOF plant has no hierarchy to expand into — a negative result that closes a chapter
rather than opening one, with the belief's actual core (does novelty grow representable directions in a
domain that *does* have hierarchy) handed off to [`rhm/residual_decomposition/`](../rhm/residual_decomposition/README.md).

## What the node's priors did over its life

- **From "verify mechanistic atoms" to "chase the phenomenon and backfill mechanism"** — the explicit
  2026-07-18 pivot in [`directed_readapt/`](directed_readapt/README.md), which shaped everything after it.
- **From "a null is a null" to "audit the grader before trusting a null"** — the S2 retraction taught the
  node to distrust control as an instrument and to check measurement subsidies before believing a negative
  result; this generalized into the standing [`COLLECTION_REALISM.md`](on_policy/COLLECTION_REALISM.md) memo.
- **From teleportation as an unexamined default to a priced, tiered, flagged choice** — the single largest
  infrastructure shift in the node's history, still not fully paid off (Tier-B sample counts throughout
  the earlier cuts remain upper bounds, stated as such rather than corrected).
- **From "rank measures structure" to "rank measures saturation"** — the recurring rank-shaped-instrument
  failure (Cut #1, `expansion/`, and the RHM/language work it cross-references) converged on a shared
  diagnosis (a saturated FM's noise floor) and a shared replacement (β / `R_res_participation`, though β
  itself proved unusable below ~high state dimension).
- **From "drift is a free novelty generator" to "drift moves the target, expansion grows the space"** —
  the idea doc's own most ambitious claim, falsified by the node it was written to justify.

## Open threads at the head of the arc

Per [`README.md`](README.md) §Next steps and [`on_policy/README.md`](on_policy/README.md) §Next steps:
the live thread is re-attempting `ballistic/directed`'s S2 loop question on the **local-drift on-policy**
substrate now that allocation is genuinely zero-sum and metered, plus testing whether ensemble
disagreement — shown structurally blind to *stale* data in the pusher's S1 — recovers its job once
on-policy collection creates genuinely *missing* (never-visited) territory. Cut #3 hardening
(multi-seed REINFORCE, a second shift type) and back-translating the interface findings ("value is
slow/committed"; "the teacher, not the structure"; "the control mode gates transmission") into the belief
tree remain explicitly unclosed.
