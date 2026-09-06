# Endogenous teacher: is the model's own surprise separable from the text's, and can it teach?

**Up**: [../README.md](../README.md) · **Files**: [FILES.md](FILES.md) · **Child**: [cancellation/README.md](cancellation/README.md)
**Design doc**: [DESIGN.md](DESIGN.md) · **Idea**: [ideas/the_forecast_needs_a_lead.md](../../../ideas/the_forecast_needs_a_lead.md)
**Date**: 2026-08-04/05 · **Status**: one positive measurement, one null.

## Goal

In next-token prediction the text is simultaneously the *stimulus*, the *target*, and the *teaching
signal*. In a brain the word is only the stimulus; the teaching signal is generated internally. This
cut asks the smallest measurable version of that distinction:

> Holding the exogenous teaching signal fixed, does an endogenously-generated one change what the
> model learns?

Token surprisal (`nll`) is the exogenous signal; the forward model's residual on the model's own
activations (`rres`) is the endogenous one. Both are per-position scalars available every step.

Regime is `v16 s2 L6 m4`, `8L/8H/256D`, FM `1L/8H/16d` predicting `post_block0 → post_block6`
open-loop — identical to [`RHM_LATENT_LOOP`](../RHM_LATENT_LOOP_README.md), whose reference lines
therefore transfer. Base at 12k steps reproduces them (d1 0.979, d3 0.836, root 0.088 against the
reference's d1 0.98, d3 0.88, root 0.08).

## Gate 0 — the headline. The two signals are largely independent, and anti-localised

Run before any intervention, on the training distribution (flat concatenated windows):

| | value |
|---|---|
| `R²(rres ~ nll)` pooled | **0.113** |
| `R²(rres ~ nll)` within-position | 0.112 |
| residual variance orthogonal to `nll` | **88.7%** |
| `corr(rres, nll)` | **−0.336** |

On aligned sequences, where each position's hierarchy level is known, they are **anti-localised**:

| position completes | n | mean `nll` | mean `rres` |
|---|---|---|---|
| level 1 (near root) | 1 | **2.609** | **0.269** |
| level 2 | 2 | 2.611 | 0.259 |
| level 3 | 4 | 2.616 | 0.236 |
| level 4 | 8 | 2.523 | 0.319 |
| level 5 | 16 | 2.077 | 0.444 |
| level 6 (leaf-adjacent) | 32 | **0.802** | **0.436** |

Token surprisal is maximal at deep subtree boundaries — where a new large constituent begins and the
next token genuinely is not determined — and internal residual is *minimal* exactly there, peaking
instead where the text is nearly free.

**Reading**: the residual tracks **computational load**, not epistemic difficulty. Where the model has
little to go on its computation is flat and easily compressed by a small FM; where it composes
confidently the FM cannot keep up. That the residual is computational rather than epistemic was
already established repeatedly on language ([`OOD_ROBUSTNESS`](../../a2a_forward/OOD_ROBUSTNESS_README.md));
what is new here is seeing it resolved against a **known DGP hierarchy**, and the specific finding that
the two signals are *anti*-correlated rather than merely distinct. Consistent with the depth FM already
conditioning on the token, so its residual should be nearly clean of exogenous surprise.

## The intervention — scalar residual weighting of the NTP loss. Null

Five arms, `arm_steps` from a shared base, differing **only in how the per-position NTP loss is
weighted**. All arms use rank-normalised weights (`w = 2·rank/(N−1)`, mean 1), so every arm has a
**bit-identical weight multiset** and only the assignment to positions differs — weight scale,
variance and effective learning rate are matched by construction, and `res_shuffled` is an exact
distributional match to `res`.

Δ from the shared base (base: val 1.5698, d4 0.379, PR 5.02):

| arm | weight ordered by | Δval | Δd3 | **Δd4** | ΔPR |
|---|---|---|---|---|---|
| `uniform` | — | −0.0183 | +0.054 | **+0.174** | +0.11 |
| `nll` | token loss (exogenous) | −0.0000 | +0.049 | +0.155 | −0.07 |
| `res` | relative FM residual (endogenous) | −0.0155 | +0.053 | +0.160 | **+1.88** |
| `res_shuffled` | `res` weights, permuted | −0.0147 | +0.050 | +0.161 | +0.36 |
| `res_orth` | residual ⊥ `nll` | −0.0081 | +0.042 | +0.152 | −1.05 |

**`res` ≈ `res_shuffled` on every learning readout** — the pre-registered falsification condition
("the residual carries no more information than its own marginal distribution") fired. Not an artefact
of a vanished residual: relative residual *rose* 0.40 → 0.44 and `fwd_cos` fell 0.908 → 0.888 in every
arm, uniformly, so the FM stayed a live frontier map throughout.

**Scope**: this is a null on **scalar loss weighting**, not on endogenous signals generally. Two prior
results predict it, and should have been read first — [`EMOTION_INJECTION`](../../a2a_forward/EMOTION_INJECTION_README.md)
(*"'this will be hard' doesn't tell the model **what** to compute differently — only **how much**"*;
cerebellar gate 3.0 vs emotion gate 0.15) and the strong-confidence belief node *"self-knowledge is
encoded directionally rather than as scalar magnitude"* (vector probe Δ R² +0.18 vs scalar +0.03;
causal steering along the residual-*norm* direction has zero novelty-specific effect). Collapsing a
directional object to its norm and using it as a multiplier tests the part already known to carry
nothing. This is nonetheless the first direct measurement against
[`residual_gated_gradients`](../../a2a_forward/ideas/residual_gated_gradients.md), an idea filed
2026-05-25 and never run.

### Two side observations

- **Ordering by token surprisal is the worst assignment of all.** Among arms with identical weight
  multisets, `nll` erases the val improvement entirely (−0.0000 vs `uniform`'s −0.0183) and has the
  smallest d4 gain. On RHM the high-`nll` positions are the *irreducible* ones (subtree boundaries),
  so "learn more from surprising text" upweights aleatoric noise — the noisy-TV pathology arriving
  through the exogenous channel.
- **`res` expanded participation ratio (5.02 → 6.90) where `res_shuffled` did not (+0.36).** The only
  res-vs-shuffled separation anywhere in the cut, and it is on the dimensionality axis rather than the
  depth axis. **Not trusted**: on a rank-shaped instrument this repo has
  found weak three times, and non-monotone across arms (`res_orth`, a blend of `res` and `nll`, is the
  *most* contractive at −1.05 rather than intermediate). Recorded, not interpreted. Seed replication
  was launched and cancelled to save compute.

### A design flaw worth recording

`res_orth` is **not interpretable as built**. Orthogonalising `rres` against `nll` was intended to
isolate the endogenous part, but `corr(rres, nll)` is *negative*, so `rres − β·nll` with β<0 **adds
token surprisal back in**. Its val (−0.0081) and PR (−1.05) accordingly sit on the `nll` side. Any
re-run should use partial ranks and handle the sign explicitly.

## Child: cancellation (the forward-path version)

[`cancellation/`](cancellation/README.md) — route the residual through the forward path instead of
collapsing it to a loss multiplier: `x = x − gate·p` at the inject block, deviation propagates,
forecast restored at the FM's target block. **The mechanism replicates the looped-ViT result closely**
(`cos(Δ,inj)` +0.293 summation → **+0.088** cancellation at matched injection norm; the ViT read +0.89
→ +0.095) but the payoffs do not: the fresh-FM swap costs +0.0024 vs +0.0015 against a dependency of
0.80 nats, so downstream depends on *a forecast* rather than on *this forecaster* in both wirings.

## Why neither cut tested the original question

Every arm of both cuts trained on `loss = NTP`, so the exogenous/endogenous variable was pinned at
"fully exogenous" throughout: cut 1 varied the *weighting* around it, cut 2 the *wiring*. And both used
the depth-wise FM, whose predictor and target have **identical information sets** — so its residual can
only ever mean "I lacked capacity," never "I was wrong." The argument that this makes depth-wise
self-prediction computational *by construction*, and the temporal/lead alternative it implies, are
developed in [`ideas/the_forecast_needs_a_lead.md`](../../../ideas/the_forecast_needs_a_lead.md).

## Reproduction

```bash
cd experiments
# gate 0 only (cheap check)
modal run -m rhm.endogenous_teacher.endogenous_teacher::endogenous_teacher \
    --arms "" --base-steps 2000 --tag gate0
# the full cut (~1h on an L4)
modal run --detach -m rhm.endogenous_teacher.endogenous_teacher::endogenous_teacher \
    --base-steps 12000 --arm-steps 8000 --tag cut1
```

Results land on the `rhm-scaling-data` volume at
`/data/v16_s2_L6_m4_distinct/endogenous_teacher/results_<tag>_seed<seed>.json`.
