# Design — the endogenous teaching signal

**Up**: [../README.md](../README.md) · **Date**: 2026-08-04 · **Status**: designed, unrun

## The question

In next-token prediction the text is simultaneously the *stimulus*, the *target*, and the
*teaching signal*. In a brain the word is only the stimulus; the teaching signal is generated
internally (cerebellar prediction error about the cortex's own trajectory). The proposed gap is
therefore not "shallow vs deep consumption of the text" but **exogenous vs endogenous teaching
signal**.

This cut asks the smallest version of that question that can be measured:

> **Holding the exogenous teaching signal fixed, does an endogenously-generated one change what
> the model learns?**

Operationally: token surprisal (`nll`) is the exogenous signal; the forward model's residual on
the model's own activations (`rres`) is the endogenous one. Both are per-position scalars
available every step. We reweight the *same* NTP loss by each and compare.

## Why RHM

Only RHM gives ground-truth latents at every hierarchy level, so the readout can be **which
levels moved**, not merely "did loss go down". That matters because the pre-registered
prediction below is about the *depth profile* of learning, not its magnitude. Regime is
`v16 s2 L6 m4` (occupancy 0.25), model `8L/8H/256D`, FM `1L/8H/16d` predicting
`post_block0 → post_block6` — identical to [`RHM_LATENT_LOOP`](../RHM_LATENT_LOOP_README.md), so
its reference lines transfer (BP root 0.80; token-NTP frontier d1 0.98, d3 0.88, d4 0.51,
root 0.08; greedy floor d3 0.751, d4 0.708).

## Design

One base model, plain NTP, with an **open-loop** co-trained FM (no injection — this cut is about
whether the residual can *teach*, not about the injection channel). Then `arm_steps` of continued
training from the shared base checkpoint, arms differing **only in how the per-position NTP loss
is weighted**:

| arm | weight ordered by | what it is |
|---|---|---|
| `uniform` | — (w ≡ 1) | the floor |
| `nll` | per-position NTP loss | the **exogenous** teacher — "learn more from surprising text" |
| `res` | relative FM residual | the **endogenous** teacher — "learn more from surprising experience" |
| `res_orth` | residual after regressing out `nll` (per batch) | **the sharp arm** — the part of your experience the text's surprisal does not explain |
| `res_shuffled` | `res` weights, permuted | the control that kills "any non-uniform weighting" |

**All arms use rank-normalised weights** (`w = 2·rank/(N−1)`, mean 1). Every arm therefore has a
*bit-identical weight multiset*; only the assignment to positions differs. This removes weight
scale, weight variance, and effective-learning-rate as confounds by construction, and makes
`res_shuffled` an exact distributional match to `res`.

The FM is co-trained on-policy in every arm with an unweighted MSE, so it stays a fair frontier
map. `fwd_cos` is logged (the lag/collapse diagnostic).

## Gate 0 — the cheap check, run before any arm

If `rres` is a deterministic restatement of `nll`, the whole question is empty. On the *training*
distribution (flat concatenated windows, which is what the weights actually see) we measure

`R²( rres ~ nll )`  pooled, and within-position.

> **R² > 0.9 would mean** the endogenous signal carries nothing the exogenous one does not, and the
> honest finding is "your experience of the text is a function of the text's surprisal."
> **Proceed if R² < 0.5**, i.e. there is a real `res_orth` component to weight by. Between 0.5 and
> 0.9 we proceed but report the arms as underpowered.

Gate 0 is also run on *aligned* sequences, where hierarchy level is defined per position, to
report which levels the residual mass sits at (the frontier-map claim).

## Prediction (registered before running)

Prior from four independent replications that endogenous self-prediction caps or inverts
([`MNIST_LOCAL_LOSS`](../../a2a_forward/MNIST_LOCAL_LOSS_README.md) `LL`;
[`RHM_LATENT_LOOP`](../RHM_LATENT_LOOP_README.md) λ_local; `data2vec` and `fm_cotrain` in
[`RHM_SCULPTING`](../RHM_SCULPTING_README.md)): **we expect no total-learning win.** The
prediction written here is about composition, not magnitude:

1. **`res` ≈ `res_shuffled` ≈ `uniform` on val NTP loss.** Endogenous weighting does not buy loss.
2. **`nll` concentrates learning at shallow levels** (d1–d2), because on RHM the deep levels are
   `vm^(ℓ+2)`-diluted and contribute almost nothing to token loss — the exogenous signal cannot
   see them.
3. **`res` shifts the depth profile upward relative to `nll` at matched compute** (d3–d4 gain,
   d1–d2 flat or down), because the residual is a map of the model's frontier, not of the
   corpus's surface. **This is the load-bearing prediction.**
4. **`res_orth` shows the effect in (3) at least as strongly as `res`**, since it is the part of
   the endogenous signal orthogonal to the exogenous one.
5. `res_shuffled` shows neither (2) nor (3).

**Falsified if** `res` and `res_shuffled` have indistinguishable depth profiles — that would mean
the residual carries no more information than its own marginal distribution, and "learn from your
own surprise" is reweighting noise.

**Null that is still informative**: if every arm is indistinguishable from `uniform`, the finding
is that a per-position endogenous signal cannot steer learning through loss weighting at all —
which localises the endogenous-teacher claim to the *forward path* (cancellation) rather than the
learning rule, and is a direct measurement against
[`residual_gated_gradients`](../../a2a_forward/ideas/residual_gated_gradients.md), an idea filed
2026-05-25 and never run.

## Known confounds and how they are handled

| confound | handling |
|---|---|
| weight scale / effective lr | rank-normalised weights, identical multiset across arms |
| "any non-uniform weighting helps" | `res_shuffled` |
| residual ∝ activation norm | weight uses **relative** residual (`‖r‖/‖h‖`); raw-norm variant logged |
| position ↔ level coupling | training uses flat windows (no fixed position↔level map); Gate 0 reported pooled **and** within-position |
| FM chases the model, residual collapses | `fwd_cos` and mean `rres` logged per arm over training |
| noisy-TV (residual = FM idiosyncrasy, not DGP gap) | not addressed in cut 1; the fresh-FM-ensemble-invariant variant is the follow-up if `res` separates from `res_shuffled` |
| seed | single seed for cut 1 by repo convention; seeds added only if the effect is real and plausibly seed-sensitive |

## Readouts

- **Per-level ancestor recovery d1…d6** (linear + MLP probe, best over blocks, last position) — the depth profile. Primary.
- Val NTP loss (unweighted, held out).
- Participation ratio of `post_block6`; mean relative residual; `fwd_cos`.
- Gate-0 statistics: `R²(rres~nll)`, per-level residual mass.

## What this does not test

- The injection/cancellation channel (the forward-path version of the same claim). Separate cut.
- Whether an endogenous signal can move the frontier *from scratch* — this starts from a base
  model already at its NTP frontier, and asks whether the endogenous signal redirects further
  training.
- Anything about ICL or an offline/consolidation phase.
