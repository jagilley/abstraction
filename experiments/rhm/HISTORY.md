# History of `experiments/rhm`

*A "big history" pass over the Random Hierarchy Model work: not a changelog, but the
shape of what we've come to believe and why it changed. See linked READMEs for detail.*

## Why RHM exists

RHM was adopted to replace confounded natural-language scaling experiments with a
controlled DGP whose axes — depth `L`, synonymic multiplicity `m`, branching `s`,
vocabulary `v` — can be manipulated independently ([`README.md`](README.md),
[`CLAUDE.md`](CLAUDE.md)). The founding fact, established early and never overturned:
**`m` dominates scaling-exponent degradation over `L` by roughly 3:1**
([`SWEEP_README.md`](SWEEP_README.md)) — synonymic multiplicity, not depth, is the
bottleneck. Nearly every later thread traces back to some version of this claim.

## Instrument-building, and the project's first self-correction

The earliest complexity probe — FM-residual effective rank — showed within its own
first writeup that rank tracks FM *approximation artifacts* (head-count mismatch) as
much as DGP structure ([`RESIDUAL_RANK_README.md`](RESIDUAL_RANK_README.md)). In
parallel, [`REGIME_TRANSITION_README.md`](REGIME_TRANSITION_README.md) records a
failed→fixed arc: the first search for an "L→m transition" in FM-residual structure
failed because the FM captured too much of the model (cosine ≥0.99, no real gap); a
follow-up swept FM capacity to find the regime where the transition is visible
(cosine ≈0.96), and a scaled run there became **the default RHM hyperparameters for
all subsequent experiments** — a small fix with outsized downstream leverage.

## What actually gates compositional depth

A cluster of objective-modification experiments asked whether the composition-depth
ceiling is a training-objective artifact rather than a data property.
[`LABEL_SMOOTHING_README.md`](LABEL_SMOOTHING_README.md) rejected NTP's sharpening
bias as the cause (smoothing hurt every level equally).
[`LOSS_WEIGHTING_README.md`](LOSS_WEIGHTING_README.md) found position/confidence
weighting doesn't extend depth either, but does improve FM legibility — useful for
making a model legible to its own self-model, not for solving depth.
[`PER_LEVEL_EXPANSION_README.md`](PER_LEVEL_EXPANSION_README.md) then falsified an
earlier informal reading of [`PER_LEVEL_LOSS_README.md`](PER_LEVEL_LOSS_README.md)
("one level of hierarchy per layer"): a >200× capacity sweep at fixed `m` barely moved
the ceiling, while sweeping `m` alone at generous fixed capacity walked it from level 2
to level 0. **Capacity and synonymic multiplicity were cleanly separated, and `m`
won.**

## Occupancy, then learnability: a pivot within a pivot

[`RHM_DEEP_COMPOSITION_README.md`](RHM_DEEP_COMPOSITION_README.md) proposed an
"occupancy law": deep structure is recoverable (reachable by backprop) and
representable (an oracle head reaches it) but not learnable by any local
self-supervised objective alone.
[`RHM_FRONTIER_AND_LEGIBILITY_README.md`](RHM_FRONTIER_AND_LEGIBILITY_README.md)
sharpened this by sweeping `v` at fixed `m`: occupancy sets the recoverability
*ceiling*, but **`m` alone gates learnability** — necessary, not sufficient. This
unlocked a `v16/m2` regime where plain NTP reaches the ceiling exactly, the platform
for [`RHM_FM_REGULARIZER_README.md`](RHM_FM_REGULARIZER_README.md) (structured
FM-predictability pressure beats weight decay's compression floor while *improving*
legibility) and [`RHM_COMPLEXODYNAMICS_README.md`](RHM_COMPLEXODYNAMICS_README.md)
(training trajectories read via Aaronson's First Law: a transient FM-idiosyncratic
scaffold rises and falls, leaving a persistent, DGP-aligned floor).

## Self-knowledge needs a latent target, not a token target

[`RHM_LATENT_LOOP_README.md`](RHM_LATENT_LOOP_README.md) is the largest single arc and
self-corrects internally: an initial "RHM never shows self-knowledge" result was
revised to "mis-measured" (positive against a co-trained FM, invisible only to fresh
FMs); decomposing self-knowledge into meta- vs. object-level components (ported from
`experiments/a2a_forward`) showed token targets give FM-*specific* meta-knowledge
while latent targets give FM-*general* meta-knowledge; and a distillation ablation
found that on RHM's compact grammar, sleep/distillation transfers **nothing** — the
no-KL control reproduces every result, unlike MNIST or language. A parallel strand,
[`SLEEP_CHUNKING_RHM_README.md`](SLEEP_CHUNKING_RHM_README.md), found a generic
capacity bottleneck does the *opposite* of chunking unless the objective targets
external roles, and via failed recursive-chunking attempts arrived at a `vm^{ℓ+2}`
(token) vs. `vm³` (latent) dilution law explaining plateaus across the ratchet,
per-level-loss, and meta-learning threads — and seeding the latent-target
intervention above.

## From passive inference to genuine control

[`ACTIVE_RHM_README.md`](ACTIVE_RHM_README.md) established that naive active-query RHM
is "inference in disguise" — a mean-Δ forward model is useless for epistemic planning,
and a plain controller already saturates value-of-information, leaving no headroom for
internalization to matter. This motivated a pivot to *editing* as a genuine act≠plan
task ([`RHM_EDIT_CONTROL_README.md`](RHM_EDIT_CONTROL_README.md)): planning in belief
space works, but exposed belief-faithfulness-off-manifold as a second obstacle, fixed
with generator-defined moves plus a veto mechanism. A combinatorial precheck
([`sculpting_control_task.md`](sculpting_control_task.md)) then certified — before any
learning — that RHM sculpting has a genuine, depth-scaling coordination prize,
validating [`RHM_SCULPTING_README.md`](RHM_SCULPTING_README.md)'s learned arc, whose
own conclusions evolve mid-document: latent planning is first judged a merely
"efficient surrogate" on a clean channel, then found *superior* once lossy, then
superior even when clean once the loop can reshape the belief itself — grounding, not
predictability pressure, is the active ingredient.
[`RHM_SCULPT_CONTINUAL_README.md`](RHM_SCULPT_CONTINUAL_README.md) tested whether this
compounds under iteration and found only a **weak ratchet**: value iteration
dominates, one-shot internalization sets a large ceiling, repeated re-internalization
adds little — echoing the "compounds while the frontier climbs, then exhausts" pattern
seen elsewhere.

## The ratchet/meta-learning thread: an extended negative result

A separate thread ([`ratchet/README.md`](ratchet/README.md) and its children) asked
whether the MNIST "gated ratchet" — wake-sleep self-model compounding that grows a
val-loss gap over cycles — reproduces on RHM. Every MNIST *dynamic* reproduced but the
*magnitude* never did — the gap plateaus at 1-2% and never compounds. In sequence, the
thread ruled out capacity ([`ratchet/RHM_RATCHET_M2_README.md`](ratchet/RHM_RATCHET_M2_README.md)),
supervision density ([`ratchet/RHM_SPARSITY_SWEEP_README.md`](ratchet/RHM_SPARSITY_SWEEP_README.md)),
richer RL-based distillation ([`ratchet/RHM_RL_RATCHET_README.md`](ratchet/RHM_RL_RATCHET_README.md)),
the first-order MAML approximation, identical to the inner loop
([`ratchet/RHM_FOMAML_README.md`](ratchet/RHM_FOMAML_README.md)), and rule-set
diversity, which adds anti-overfitting rather than depth
([`ratchet/RHM_META_LEARNING_README.md`](ratchet/RHM_META_LEARNING_README.md)). A late
40-cycle extension debunked its own motivating trend as statistical noise
([`ratchet/RHM_RL_GEN_DISTILL_EXTENDED_README.md`](ratchet/RHM_RL_GEN_DISTILL_EXTENDED_README.md)).
Converging diagnosis: compounding needs a *moving frontier* that a stationary rule set
cannot supply once exhausted — a belief that directly seeded `directed_sculpting`
below.

## Residual rank dies properly, and two smaller siblings

[`residual_decomposition/README.md`](residual_decomposition/README.md) formally
supersedes [`RESIDUAL_RANK_README.md`](RESIDUAL_RANK_README.md) (now banner-marked
superseded): rank conflated residual *shape* and *magnitude*, which the new instrument
(power-law exponent β, plus a participation-weighted rank) separates. The old positive
finding survives, reindexed onto β; the repo-wide belief `R_act ≈ R_comp + R_res`
([`beliefs/dimensionality_expansion.md`](../../beliefs/dimensionality_expansion.md))
does not survive, nor does a matching claim in
[`experiments/mjc/contact_residual/README.md`](../mjc/contact_residual/README.md).
Two smaller threads share this arc's DNA:
[`specialization/README.md`](specialization/README.md) found breadth-vs-depth
concentration inert-to-harmful for the depth stall — reframing "broad beats narrow" as
the *enabling* condition for shared-abstraction generalization, not a deficiency.
[`confabulation/README.md`](confabulation/README.md) built on this arc's
capacity-guard methodology and found a capacity-robust first-person advantage for
reporting on FM-residual state (genuine introspection) but none for behavior
(confabulation) — access-limited, not resource-limited.

## Directed sculpting: porting mjc's E3 loop, mostly negative

[`directed_sculpting/README.md`](directed_sculpting/README.md) ports `experiments/mjc`'s
outer allocation loop onto RHM to test the "moving frontier" hypothesis above. The
naive port hit a wall immediately (uniform rule usage gives no relevance signal to
allocate over), fixed with structurally irrelevant "distractor channels." The
throughline is **instrument traps caught before a false positive**: raw cross-entropy
proved exactly blind to drift by algebraic identity, a zero-drift control revealed a
spurious "repair" gap from ordinary training, and an "own-world grading" bug had
manufactured ~65% of an apparent drift advantage. The reward-free relevance signal
transfers cleanly, but both headline predictions — climbing and representational
expansion under drift — came back negative in
[`directed_sculpting/full_loop/README.md`](directed_sculpting/full_loop/README.md). A
late reframe ("invariance ≠ necessity": free repair prices nothing) recovered a narrow
positive by starving samples-per-event instead of varying drift magnitude. The
`partial_hetero` child
([`directed_sculpting/full_loop/partial_hetero/README.md`](directed_sculpting/full_loop/partial_hetero/README.md))
built a sharing-depth knob predicting a monotone payoff curve; none appeared, traced to
the forward-model's unit of work being exactly the *unshared* level at every
intermediate depth — possibly intrinsic to surface learners, not a fixable bug.

## Minting: from "mirror vs. verifier" to "support vs. density"

[`minting/README.md`](minting/README.md) tests whether self-generated data, filtered
by a non-mirror verifier, can recover some of what broad training data buys over
narrow — motivated by
[`ideas/breadth_as_grader_heterogeneity.md`](../../ideas/breadth_as_grader_heterogeneity.md).
The thread pivoted twice: an early "mirror arm degenerates" finding was retracted once
overfitting was ruled out as the real cause, and the mirror/non-mirror axis was
reframed as separating on grader *legality vs. location*. A dose confound then
explained away an apparent "verification doesn't help" result — at high gradient
concentration even real held-out data degrades equally. At corrected dose,
verifier-accepted data beats random and mirror acceptance (3 seeds) but only reaches
break-even with no minting, far short of real data's gain. Final reading: the
rule-table verifier is a **support oracle** ("is this in-distribution"), not a
**density oracle** ("is this drawn with correct probability") — it prevents autophagy
but structurally cannot supply the density information NTP needs, so support
expansion alone buys no capability. The unbuilt next step is a density-carrying rule.

## Current state of belief

`m`, not `L`, bottlenecks compositional generalization and gates *learnability*
specifically ([`SWEEP_README.md`](SWEEP_README.md),
[`RHM_FRONTIER_AND_LEGIBILITY_README.md`](RHM_FRONTIER_AND_LEGIBILITY_README.md)).
Self-knowledge needs a latent (not token) target and doesn't transfer through
distillation the way it does on MNIST/language
([`RHM_LATENT_LOOP_README.md`](RHM_LATENT_LOOP_README.md)). Self-model compounding
requires a moving frontier that a stationary rule set exhausts
([`ratchet/README.md`](ratchet/README.md),
[`directed_sculpting/full_loop/README.md`](directed_sculpting/full_loop/README.md)),
though sculpting/editing shows grounding beats passive predictability pressure even
if iterating it compounds only weakly
([`RHM_SCULPTING_README.md`](RHM_SCULPTING_README.md),
[`RHM_SCULPT_CONTINUAL_README.md`](RHM_SCULPT_CONTINUAL_README.md)). Instrument
hygiene is the recurring hazard: rank-as-complexity, raw cross-entropy under drift,
and mirror-vs-verifier all first produced misleading signals later traced to
measurement artifacts, not the phenomenon of interest.

## Open threads

Whether a self-supervised latent target reproduces the oracle self-knowledge result
([`RHM_LATENT_LOOP_README.md`](RHM_LATENT_LOOP_README.md)); whether sculpting's drift
reopens the weak ratchet ([`directed_sculpting/README.md`](directed_sculpting/README.md));
and a density-carrying acceptance rule for minting ([`minting/README.md`](minting/README.md)).
