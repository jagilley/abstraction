# SPEC — does the simplicity collapse survive an exogenous conditioning gap?

**Parent**: [`../README.md`](../README.md) · **Idea**: [`ideas/revision_not_surprisal.md`](../../../../ideas/revision_not_surprisal.md) §8
**Status**: unrun. Pre-registered 2026-08-08.

## The question

Every local-loss result in this repo has the same shape: add `λ‖sg(FM(h_src)) − h_tgt‖²` to NTP and
the model becomes *legible* without becoming *self-knowing*.

| result | where |
|---|---|
| FM cos **0.997**, SK R² **−0.03**; **13.4×** brittleness | [`MNIST_LOCAL_LOSS`](../../../a2a_forward/MNIST_LOCAL_LOSS_README.md) |
| the ratchet gate **opens** 0.31 → 0.77 (*"inner = outer, so more compression is always beneficial"*) | [`GATED_RATCHET`](../../../a2a_forward/GATED_RATCHET_README.md) |
| *"self-knowledge is not required for functional simplification"* — FM-free act rank 54.3 → 42.9% | [`RHM_FM_REGULARIZER`](../../RHM_FM_REGULARIZER_README.md) |
| λ_local ↑ ⇒ residual 13.4 → 0.8, fresh meta **+0.284 → −2.087**, ΔSK −0.06 → −1.54 | [`RHM_LATENT_LOOP`](../../RHM_LATENT_LOOP_README.md) F4 |

§8 argues these are one fact with a *proof* behind it, not four empirical nulls. `h_src` and `h_tgt`
are both deterministic functions of the same input through the same weights, so the only way to lower
the term is to move `h_tgt` into the image of a low-capacity FM. **With an endogenous target and
identical information sets, "minimize prediction error" is definitionally "be simpler."**

Swap the target one position instead of six blocks and the proof stops applying — `h6[t+1]` depends on
`x_{t+1}`, which a causal FM over `h6[≤t]` cannot hold. [Gate 0](../README.md#gate-0) verified the
resulting residual has a real aleatoric component on this substrate (`corr(res, nll)` −0.337 → **+0.652**,
with a protocol-matched `depth_frozen` control at −0.328 ruling out the training protocol).

> **With the proof gone, does the collapse happen anyway?**

This is a yes/no about a mechanism. It does **not** depend on how much revision the residual carries —
[Gate B](../README.md#gate-b) measured that separately and it is small (`r_temporal` 0.53–0.61 AUC under
full matching, chance 0.5). Do not conflate the two; a good outcome here is not evidence the residual
is a good revision detector, and vice versa.

## Why this and not the rest of the parent's queue

Gate C (capacity invariance of `Δ^rev`) is the careful version of a claim about the channel Gate B
already showed is weak — deprioritized. The precision/teaching line is a measured null with a general
mechanism behind it ([`sculpt_slip`](../sculpt_slip/README.md): an aleatoric filter can only pay when
the learner is variance-limited, and ours was bias-dominated across a 60× budget range). §8 is the one
unrun claim that can falsify the idea doc's central mechanism rather than tune an allocator.
The parent [SPEC](../SPEC.md) pre-registered this exit: *"if the goal is to test the reframe rather
than to build the allocator, it may be the better first cut."*

## Design

**Fork [`rhm_fm_regularizer.py`](../../rhm_fm_regularizer.py).** It is already the LL-alone,
open-loop, gradient-into-the-main-model-only version on RHM, with λ warmup, a λ sweep, and — importantly —
the **leakage-proof FM-free activation-rank** readout, plus WD-floor anchors that make each λ a drop-in
delta over a λ=0 baseline. The §8 change is the target, and only the target.

```
depth   (incumbent):  FM( post_embed[t] )  →  post_block6[t]      # 0% aleatoric, proof applies
temporal (this cut):  FM( post_block6[≤t] ) →  post_block6[t+1]    # gap = exactly one token
```

Both arms at a matched λ sweep is the control; the parent's `temporal_direct` arm showed the
update-vs-absolute parametrisation is not load-bearing, so either form is fine.

**Readouts — all of these already exist**, in `rhm_fm_regularizer.py` and `rhm_latent_loop.py`
(`_sk_probes`, `_meta_object_probes`, `_ensemble_agreement`, `_anticollapse_diag`): FM cos, fresh-FM SK
R², meta/object probes, fresh-FM ensemble agreement, FM-free activation effective rank + top1%,
per-level BP knowledge gate, per-level η². The knowledge gate is the denominator — rank and η² only
mean anything at matched `d6`.

### The replacement degeneracy, and the instrument RHM gives you for free

§8 names what should replace the simplicity collapse: **input-invariance** — lower the term by making
`h6[t+1]` insensitive to `x_{t+1}`. This is data2vec's known mode and
[`rhm_sculpt_data2vec.py`](../../rhm_sculpt_data2vec.py) has anti-collapse diagnostics for it.

Two things worth thinking about before you build:

1. **NTP partially resists total invariance** — the model still has to predict `x_{t+2}` from `h6[t+1]`.
   So the live version is *partial*: discarding the parts of `x_{t+1}` that don't matter downstream.
2. **On RHM, synonym identity is exactly such a part.** Which token realizes a latent is irrelevant to
   everything downstream except through the latent. So one available outcome is that the temporal local
   loss discards synonym identity and keeps structure — which would be the **aleatoric null showing up
   as a representational consequence rather than as a weighting**. That is a different result from
   either branch of the pre-registered kill and it would be the most interesting one.

This suggests an instrument the parent already has the machinery for: **swap the token at `t+1` among
its synonyms (the oracle labels them exactly, `B ≡ 0`) and measure how far `h6[t+1]` moves; separately
swap it for a structure-changing token and do the same.** Invariance-collapse is both falling.
The aleatoric null is the first falling and the second not. Consider building this; consider a better
one.

A bare null is uninterpretable between "collapsed the old way" and "collapsed the new way", so
instrument for both from the start rather than adding an anti-collapse arm reactively.

## What each outcome would teach

| | reading |
|---|---|
| temporal reproduces the depth signature (FM cos → ~1, SK → 0, comparable rank collapse) | **the conditioning gap is not the operative variable.** Idea doc §1 is decorative for training purposes and simplicity collapse is a property of any auxiliary be-predictable term. This is the strongest single result available here — it falsifies the reframe, not just the instrument. |
| temporal collapses toward input-invariance instead | the gap is operative and §8's predicted trade is confirmed. Then the question is whether anti-collapse machinery buys anything or just moves the degeneracy again. |
| synonym-invariance rises while structure-sensitivity holds | the aleatoric null as a representational fact. Not predicted by §8 as stated; would need the idea doc rewritten around it. |
| neither collapse, SK positive | the first non-degenerate local loss in this program. Check it hard before believing it — `RHM_LATENT_LOOP` F4's fresh-vs-co-trained split is where the previous version of this claim died. |

**Standing prior against**, from the idea doc: four independent replications say endogenous targets cap,
and `full_loop` §5 says grounding is the pivot. None was conditional-revision, so the cap does not
transfer directly — but if the honest answer turns out to be "it only works against a grounded readout,"
say so rather than discovering it later.

## Substrate — your call, here is the tradeoff

- **m2** (`v16 s2 L6 m2` distinct-rule, `rule_seed=0 seq_seed=1`, 8L/8H/256D, wd=0.1) — what
  `RHM_FM_REGULARIZER` ran on. Its WD-floor anchors and λ=0 baselines are drop-in, so you need no
  separate control run, and the model fully learns the hierarchy (d6 0.94) so belief depth is not
  binding. Best controls; recommended for a signature comparison, since the signature was measured here.
- **m4** (`/data/v16_s2_L6_m4_distinct/conditional_revision/`) — the parent's cached base, oracle and
  belief probes. But root recovery is 0.088, so deep readouts sit near floor.

The synonym-swap instrument works on both (m2 has 2 synonymous rules per feature, m4 has 4).

## What this deliberately does not test

- **Whether the temporal local loss helps anything.** LL's learning-speed win (+2.5pp at step 500, 31%
  lower final val) is real, orthogonal, and unexplained by the degeneracy reading (§8). Not this cut.
- Injection, distillation, or any closed loop. LL-alone, per §8's kill.
- Whether the residual is a good revision signal. Gate B answered that; it is not the question here.

## Conventions

`/run-experiment-on-modal`[^private] before running
anything. Smoke first (attached, minutes), then `--detach` and wait for the completion notification —
do not poll logs. Single seed; add seeds only if the effect is real and plausibly seed-sensitive.
`/writeup`[^private] for the README, with `FILES.md`, and discuss
results before writing it.

[^private]: Not mirrored: this link points to a document in the private lab repo (the roadmap, the queue, an unrun spec, reading notes, or a conversation). See the top-level README for what is held back and why.
