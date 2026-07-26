# Continual Learning as Dimensionality Expansion

**Date:** 2026-07-03 · *extended 2026-07-25 with §"What grades an expansion?"*
**Status:** Working synthesis — speculative core, partial supporting evidence from a2a_forward
**Related:**
- [Self-prediction and self-knowledge](./trees/self_prediction_and_self_knowledge.md)
- [Cerebellum and cognitive architecture](./trees/cerebellum_and_cognitive_architecture.md)
- [Metacognitive novelty learning](./metacognitive_novelty_learning.md)
- [Heterogeneous graders](../ideas/heterogeneous_graders.md) — supplies the grader this file's expansion drive left unspecified
- [A2A forward model](../experiments/a2a_forward/README.md)

## The belief

The objective of continual learning is not to minimize error on any task. It is to **grow the number of directions the base weights can coherently represent** (`R_act`), so that any individual task occupies a progressively smaller, more sharply characterized subspace of a larger whole. Novelty is sought because it is the only unbounded source of new representable directions. A persistent, nonzero self-model residual is the *signature of health*, not a failure to converge — the goal was never a residual of zero.

This reframes the recurring villain of the a2a_forward ratchet work — the absorbing state / narrow FM-capacity sweet spot on a fixed dataset — as the predicted consequence of running a compressor with no expansion drive.

## The decomposition (how base-weight expansion relates to residual rank)

Fix a layer, run the data distribution, and split each activation `A` by what the self-model (forward model) compresses:

```
A  =  FM(earlier_layer)  +  Residual
      └── compressed ──┘     └─ frontier ─┘
```

Three effective ranks, over the distribution:

- **R_act** = rank(`A`) — total directions the base weights actually use. *This is "dimensionality expansion in the main model."*
- **R_comp** = rank(`FM(·)`) — directions the self-model has internalized as routine.
- **R_res** = rank(`Residual`) — directions not yet compressed. The frontier.

Subadditively, **R_act ≈ R_comp + R_res**: the self-model draws a moving partition through the total dimensionality. R_res is therefore not a separate quantity from base-weight dimensionality — it is *the slice of R_act the self-model hasn't absorbed yet*.

## Two flows across the partition

- **Compression drains R_res → R_comp** (R_act unchanged). Wake-sleep distillation folds residual directions into routine. On a *fixed* dataset R_act is capped, so draining runs to the wall: R_res → 0. This is the observed WS collapse and absorbing state.
- **Novelty raises R_act by injecting fresh directions into R_res from the top.** New data forces representation of new computational states; those land in the frontier before they are compressed. Novelty refills faster than compression drains.

Healthy continual learning keeps both flowing: **R_act ↑, R_comp ↑, R_res persistently > 0.**

## Two routes to expansion — one bounded, one not

- **Internal (self-legibility reorganization):** the gated local loss spreads existing computation across more, more-orthogonal directions without new data. This raises R_act *on fixed data* — but it is **capped**, because you can only re-express fixed content so far.
- **External (novelty):** new data adds genuinely new content. **Unbounded.** This is the channel the human intuition is about.

The a2a data already shows the internal channel saturating: WS_LG's residual rank grows (27.8 → 30.2) yet by 16 cycles it hits activation-norm inflation instead of compounding — the internal route ran out. Prediction: a novelty-fed ratchet should not hit that wall.

## The health meter is a triple, not the residual alone

R_res alone is ambiguous — it is high both for a *weak self-model* (can't compress even routine) and for a *rich frontier* (routine absorbed, much genuinely new). The trajectory of all three disambiguates:

| Regime | R_act | R_comp | R_res |
|---|---|---|---|
| Absorbing (fixed-data WS) | flat | ↑ then flat | → 0 |
| Weak FM | flat | low | high, static |
| **Healthy continual** | **↑** | **↑** | **> 0, refilled** |

All three are directly measurable: `rank(A)`, `rank(FM output)`, `rank(residual)`.

## Supporting evidence (from a2a_forward)

- Naive wake-sleep on fixed data collapses residual rank (23.9 → 13.5); the gated ratchet grows it (27.8 → 30.2) but inflates activation norms by 16 cycles — [GATED_RATCHET_README](../experiments/a2a_forward/GATED_RATCHET_README.md).
- The FM-capacity sweet spot is narrow on a fixed dataset, and the authors independently conclude "a continual learning setting would sidestep this by providing novel data that sustains compression pressure" — [README §extended ratchet](../experiments/a2a_forward/README.md).
- The gate becomes input-selective (opens on novel, closes on routine) only under distribution shift; on a stationary set it opens uniformly — [OOD_GATE_README](../experiments/a2a_forward/OOD_GATE_README.md).
- Closed-loop self-knowledge is *computational and distribution-invariant, not epistemic* — the residual encodes what kind of computation is un-compressed, consistent with it being a frontier map — [OOD_ROBUSTNESS_README](../experiments/a2a_forward/OOD_ROBUSTNESS_README.md).

## What grades an expansion? (2026-07-25)

This file specifies the two flows and never says **what decides which novelty** — "novelty raises `R_act`" leaves the selector unnamed. The [heterogeneous-graders frame](../ideas/heterogeneous_graders.md) supplies it, and the answer is that compression and expansion are gradeable by *different kinds of signal*, necessarily.

**The activation-energy argument.** Copernicus threw away degrees of freedom and got *worse* predictions than Ptolemy; on a description-length metric at the moment of the rotation, epicycles win ([contra-long-horizon-benchmarks](../ideas/contra-long-horizon-benchmarks.md) §I–II). So compression is the **endpoint, not the process** — you often must expand first, tolerate a worse fit, and cross a barrier to reach the basis in which the compression is available at all. A learner that monotonically descends a compression objective cannot cross that barrier by construction, which is the same wall this file already measures from the inside (`R_res → 0` on fixed data).

| | graded by | availability | payoff |
|---|---|---|---|
| **compression** (`R_res → R_comp`) | prediction error | dense, every step, free | immediate |
| **expansion** (`R_act ↑`) | *cannot be prediction error* | sparse, slow | **deferred, initially negative** |

Opening a direction makes prediction worse before it makes it better, so no dense predictive signal can drive it; its grader must tolerate deferred and initially-negative payoff. That is what "evaluative" means, and the underlying impossibility is that **no single signal is both dense and evaluative** — dense requires being free (self-supervised on what happened), evaluative requires referencing outcomes you would rather not sample ([cerebellum tree](trees/cerebellum_and_cognitive_architecture.md#no-single-learning-signal-can-be-both-dense-and-evaluative--so-the-two-teacher-structure-is-derived-not-designed)). **Compression/expansion is that dichotomy viewed from the representation side rather than the signal side.**

**This says what the selector is: learning progress is the expansion grader.** LP is ~0 when mastered (dark room), ~0 when irreducible (noisy TV), and peaks at moderate-and-*falling* error — a band-pass shape recorded in [two_timescale_value_loop](../ideas/two_timescale_value_loop.md) as an empirical property of a curiosity drive and validated in [curiosity Phase 1](../experiments/a2a_forward/reaching/CURIOSITY_DRIVE_README.md). Read through this frame the shape is a **specification**, not a curiosity: it is what a grader of "is this direction worth opening?" has to look like. The experiment proposed below (residual-guided data selection) is therefore not one option among many — it is the expansion grader, wired.

**Two consequences worth recording.**

1. **The LLM reading.** Weight decay, width, depth, data mix, when to stop — every one is an expansion decision graded on deferred payoff. LLM training does not *lack* an expansion loop; **it has one, implemented in humans**, running on a wall-clock of weeks. That is the mechanism behind LLMs expanding only slowly, coarsely, and exogenously: the expansion grader was outsourced to the researcher. It also predicts that an intrinsic expansion grader is worth more than a better compressor.
2. **The abandoned flagship.** *Dimensionality expansion under drifting dynamics* was [`physical_control_substrate.md`](../ideas/physical_control_substrate.md) cut #5, billed in its own text as *"the single biggest open question of the whole program,"* and never run — the drift machinery built to make it free was consumed by the value-loop program instead ([`mjc/HISTORY.md`](../experiments/mjc/HISTORY.md) §Planned and never run). Two independent routes have now arrived back at it. **Verified unrun 2026-07-25**: no rank or spectral measurement exists anywhere in `mjc/` except cut #1's, and nothing was started-and-abandoned in git history.

   **The instrument caution this file needs to carry.** Cut #1 ran a participation ratio on the 8-dim *state residual* of one FM on a *stationary* pusher, got a negative (contact eff-rank 3.60 > free 2.52), and retired *"residual rank ∝ DGP complexity"* ([contact_residual](../experiments/mjc/contact_residual/README.md)); a2a independently found `rank ⊥ noise` — language's residual is full-rank 200/256 yet the FM captures it to cosine 0.97. So **rank has failed as an instrument here twice**, and this file's `R_act`/`R_comp`/`R_res` triple inherits that prior. What distinguishes the surviving version: those were *absolute, static, single-condition* claims about what a rank magnitude means; the triple is used *differentially against a matched control* (does `R_act` move under drift where fixed dynamics exhausts?), which is the form that held throughout the mjc tree when magnitudes did not. Two design constraints follow from the same source — the drift must have **growing support** (a one-parameter walk demands re-fitting, not new directions) and capacity must actually **bind** (2-link arms leave it slack; n ≥ 5). Specified cut: [`mjc/README.md`](../experiments/mjc/README.md) §Next steps #3.

   Note also the `R_res` type ambiguity already raised in [RHM_LATENT_LOOP](../experiments/rhm/RHM_LATENT_LOOP_README.md) — `R_res` needs a type label, since it is high both for a weak self-model and for a rich frontier (the §"health meter is a triple" argument above is the intended disambiguation, and it is the reason all three must be reported together rather than `R_res` alone).

## The experiment this predicts (self-model curiosity)

Standard curiosity rewards *world*-model prediction error. The cerebellar residual is prediction error about the model's *own computation* — a curiosity signal computed for free. Make it active: **let the FM residual select what to train on next** (high residual = frontier = where new representable dimensions live).

- **Primary readout:** the (R_act, R_comp, R_res) triple across cycles.
- **Prediction:** residual-guided data selection produces the healthy signature (R_act ↑, R_res refilled, gate stays selective) and avoids the fixed-data inflation wall; uniform/random data selection does not.
- **Contrast condition:** dedicated-practice ratchet on a fixed target vs. the same compute spent on residual-selected novelty — tests whether "lateral" beats "direct" as measured by target-task residual becoming lower-rank and more directional.

## Caveat

R_act is an *activation* rank, hence data-dependent, not a pure weight property — it mixes "base-weight capacity" with "how much this data excites it." A weight-native measure (Jacobian rank, or count of active dictionary features) would isolate base capacity if the activation-rank story gets confounded.
