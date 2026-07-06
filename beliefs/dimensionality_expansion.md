# Continual Learning as Dimensionality Expansion

**Date:** 2026-07-03
**Status:** Working synthesis — speculative core, partial supporting evidence from a2a_forward
**Related:**
- [Self-prediction and self-knowledge](./trees/self_prediction_and_self_knowledge.md)
- [Cerebellum and cognitive architecture](./trees/cerebellum_and_cognitive_architecture.md)
- [Metacognitive novelty learning](./metacognitive_novelty_learning.md)
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

## The experiment this predicts (self-model curiosity)

Standard curiosity rewards *world*-model prediction error. The cerebellar residual is prediction error about the model's *own computation* — a curiosity signal computed for free. Make it active: **let the FM residual select what to train on next** (high residual = frontier = where new representable dimensions live).

- **Primary readout:** the (R_act, R_comp, R_res) triple across cycles.
- **Prediction:** residual-guided data selection produces the healthy signature (R_act ↑, R_res refilled, gate stays selective) and avoids the fixed-data inflation wall; uniform/random data selection does not.
- **Contrast condition:** dedicated-practice ratchet on a fixed target vs. the same compute spent on residual-selected novelty — tests whether "lateral" beats "direct" as measured by target-task residual becoming lower-rank and more directional.

## Caveat

R_act is an *activation* rank, hence data-dependent, not a pure weight property — it mixes "base-weight capacity" with "how much this data excites it." A weight-native measure (Jacobian rank, or count of active dictionary features) would isolate base capacity if the activation-rank story gets confounded.
