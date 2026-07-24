# The Confabulation Test (language)

**Status**: done, single seed. **Positive on the core dissociation, null on loop-necessity** —
and the loop-necessity null *disagrees with the RHM sibling*, which is unresolved (see
[Cross-substrate disagreement](#cross-substrate-disagreement-with-rhm)).

**Design doc / pre-registration**: [ideas/confabulation_test.md](../../../ideas/confabulation_test.md)
**Sibling instantiation**: [experiments/rhm/rhm_confabulation.py](../../rhm/rhm_confabulation.py) (RHM)
**Substrate**: [a2a_forward/stages.py](../stages.py)`::a2a_loop_train` — forks that wake recipe
(same GPT, `TransformerForwardModel`, `CerebellarGate`, lrs, FineWeb-Edu tokens) in the Run-6
controlled-retrain shape, and adds the report battery on top.
**Parent**: [../README.md](../README.md) — Run 6 (self-knowledge is directional, distributed via
backprop) and [OOD_ROBUSTNESS_README.md](../OOD_ROBUSTNESS_README.md) (self-knowledge is
*computational, not epistemic*) are the load-bearing priors.
**Files**: [FILES.md](FILES.md)

## The question

Every LLM-introspection claim faces the Nisbett & Wilson objection: a self-report may be
produced by a *theory* of oneself rather than *access* to oneself. A forward self-model gives
that objection an operational form:

```
a_j  =  FM(a_i)  +  r
        ^^^^^^^     ^
        everything a       the part no
        self-theory        model of M
        could produce      anticipated
```

`FM` is the best compressible predictor of M learned from M's own activity alone, so
**confabulation lives in the range of the self-model and whatever cannot be confabulated lives
in its complement.**

A signal `S` is introspective for M iff (1) `S` is a function of M's internal state, (2) `S` is
not *cheaply* recoverable from M's input–output map, (3) M's behaviour is causally sensitive to
`S`. **(2) is the discriminator**: calibration and entropy are functions of the I/O map, hence
third-person accessible, hence self-*inference*; `r` is a function of the *implementation*, and
two models with identical I/O behaviour have different `r`.

## Setup

**Arms** — `cl` (injection via `CerebellarGate`) vs `ol` (no injection), Run-6
controlled-retrain shape: shared init, seed, lr and data order; only the loop differs.

**Config** — 4L/4H/256D GPT (28.9M) on 10M FineWeb-Edu tokens, `T=128`, 10K steps.
`post_block0 → post_block2`, injection after block 1, report emitted from `post_block3`.

**Report targets** — `IMPL` (K=8 cluster of the residual *direction*), `IMPL_COS` (the residual
direction itself, scored by test cosine), against three controls: `BEHAV` (is M's next-token
argmax correct), `ENT` (M's output-entropy quartile), `WORLD` (5-way syntactic category, from
tokens alone).

**Instrument sweep** — every IMPL number is computed at four fresh-FM capacities (4.9% → 42.0%
of the two predicted blocks) and gated on `ens_cos` and residual structure. See
[the junk-residual trap](#the-junk-residual-trap-and-why-it-is-disarmed).

## Headline

**The predicted dissociation holds, at every capacity, in both arms.** The two implementation
targets show a large first-person advantage over the best third-party observer; all three
control targets show zero or *negative* advantage.

Control targets at the default instrument (CL arm; OL is within 0.006 everywhere):

| target | determined by | self | best `O_io` | **advantage** | confab margin |
|---|---|---|---|---|---|
| `IMPL` | implementation | 0.511 | 0.239 | **+0.272** | +0.142 |
| `IMPL_COS` | implementation | 0.590 | 0.272 | **+0.318** | +0.236 |
| `BEHAV` | I/O map | 0.782 | 0.850 | −0.067 | −0.004 |
| `ENT` | I/O map | 0.674 | 0.961 | **−0.287** | +0.039 |
| `WORLD` | input | 0.985 | 0.994 | −0.009 | −0.000 |

`ENT` is the sharpest control: a third party predicts M's own output entropy at 0.961 while M
reports it at 0.674. An I/O-map fact is *better* known from outside than M can read it off its
own residual stream — exactly what criterion (2) says should happen, and the reason
calibration-style evidence cannot support an introspection claim.

**But closing the loop changes nothing.** CL and OL are indistinguishable on every measure.

## Result 1 — the introspective advantage, swept over instrument capacity

CL arm (OL in the next table):

| instrument | % pred blocks | `ens_cos` | `IMPL` self / `O_io` / **adv** | `IMPL_COS` self / `O_io` / **adv** | `O_act` ceiling | steer ratio |
|---|---|---|---|---|---|---|
| h4m0.25 | 4.9 | **0.904** | 0.599 / 0.225 / **+0.374** | 0.693 / 0.348 / **+0.345** | 0.681 | 2.07× |
| h16m0.5 | 10.6 | 0.854 | 0.511 / 0.239 / +0.272 | 0.590 / 0.272 / +0.318 | 0.578 | 1.73× |
| h32m1 | 21.1 | 0.818 | 0.416 / 0.177 / +0.239 | 0.496 / 0.210 / +0.286 | 0.451 | 1.43× |
| h64m2 | 42.0 | 0.777 | 0.378 / 0.167 / +0.211 | 0.416 / 0.163 / +0.252 | 0.319 | 1.58× |

Three things to read off this:

- **The advantage is largest exactly where the residual is most trustworthy.** It rises
  monotonically as the instrument shrinks, and so does `ens_cos`. This is the *opposite* of the
  junk-residual signature.
- **The `O_act` ceiling behaves as designed on the continuous target** — it nearly matches the
  self-report (0.681 vs 0.693 at the smallest instrument) while I/O-only observers sit at 0.348.
  The information genuinely is in the activations and recoverable by an observer *with
  activation access*; it is the I/O-only observer that cannot get it. On categorical `IMPL` the
  ceiling does *not* reach self (0.291 vs 0.599) — recovering a direction is evidently easier
  than hitting a k-means cell.
- **The observer is not starved.** Halving its data barely moves it (`O_io` 0.272 → 0.202 on
  `IMPL_COS`, 0.239 → 0.203 on `IMPL`), so the shortfall is about access, not budget. Observer
  capacity does not rescue it either: `O_io` at 4L256D is no better than at 2L128D.

## Result 2 — the loop is not necessary (the null)

| instrument | `IMPL` adv CL / OL / **Δ** | `IMPL_COS` adv CL / OL / **Δ** | confab margin Δ (`IMPL_COS`) | steer CL / OL |
|---|---|---|---|---|
| h4m0.25 | +0.374 / +0.370 / **+0.004** | +0.345 / +0.343 / **+0.002** | −0.016 | 2.07 / 1.92 |
| h16m0.5 | +0.272 / +0.273 / **−0.001** | +0.318 / +0.318 / **−0.001** | −0.013 | 1.73 / 1.40 |
| h32m1 | +0.239 / +0.239 / **+0.000** | +0.286 / +0.289 / **−0.003** | −0.012 | 1.43 / 1.39 |
| h64m2 | +0.211 / +0.210 / **+0.002** | +0.252 / +0.256 / **−0.003** | −0.012 | 1.58 / 1.47 |

The advantage delta is ≤0.004 everywhere. The confabulator margin is, if anything, *slightly
larger* in the open loop. Only steering leans CL, and by little.

This is the pre-registered falsifier for the strong claim, and the design doc says what to do
with it: *"If OL shows the same advantage, the loop is not necessary for introspection and the
finding is about the FM decomposition alone — still interesting, but a weaker claim, and we
should say so."* So: **the residual is a channel carrying implementation-facts not cheaply
recoverable from the I/O map, and on language that is a property of the forward-model
decomposition itself, not of training the model to use it.**

### The most likely confound, and it is ours

**The CL arm here is a weak loop, and we made it weak.** Closing the loop is *net negative* in
this config:

| | val loss |
|---|---|
| OL standalone | **5.320** |
| CL standalone | 5.446 |
| CL with injection | 5.352 |

The injection helps CL by 0.094 nats against its own standalone baseline (the familiar
dependency signature — the gate opened monotonically 0.08 → 3.10), but CL *with* the injection
is still 0.031 nats **worse** than OL. There is no net benefit to closing the loop at all.

Two design choices drove this, and both trace to the report-site topology:

1. **The gap is 2 blocks, not 3.** We moved `predict_to` to `post_block2` so a real block would
   sit between the FM target and the report site (with 4 layers, the canonical `post_block3`
   gap puts the report *at* the target and degenerates Tests 2/3 into a linear readout). A
   shorter gap is easier to predict.
2. **The co-trained FM is at saturation.** At 660K it is **42.0%** of the two predicted blocks,
   and §3.2 of the paper puts FM saturation at ~44% of the predicted layers. An FM that can
   already predict the gap leaves little residual for the main model to learn to complement —
   the Run 7b regime, where a capacity-sufficient FM yields pure computation relocation and zero
   net benefit.

So the honest statement is not "the loop does not produce reportable self-knowledge on
language" but "**a loop that produces no net benefit also produces no reportable-self-knowledge
gap**". Whether a *load-bearing* language loop would is untested. See
[Next steps](#next-steps).

## Result 3 — channel ablation (Test 2)

Clean and consistent at every capacity. `IMPL_COS`, CL arm:

| instrument | full | `shuffle_r` | `shuffle_p` | `zero_r` | `zero_p` |
|---|---|---|---|---|---|
| h4m0.25 | 0.693 | **0.113** | 0.613 | 0.175 | 0.672 |
| h16m0.5 | 0.590 | **0.099** | 0.512 | 0.134 | 0.620 |
| h32m1 | 0.496 | **0.082** | 0.428 | 0.100 | 0.566 |
| h64m2 | 0.416 | **0.076** | 0.353 | 0.086 | 0.512 |

Destroying the residual collapses the report to near zero; destroying the self-theory component
leaves most of it intact — and `zero_p` (residual *only*) is the **best** variant everywhere.
The report rides on `r`, not on `FM(a_i)`.

## Result 4 — matched-KL steering (Test 3)

Positive but modest: the residual/prediction IMPL-flip ratio is **1.43–2.07×** (CL). The
matched-behaviour control works — BEHAV report-flip is 0.024 (residual) vs 0.020 (prediction),
essentially equal across families, so the asymmetry is not a behavioural artifact. The KL solve
lands on target to 4 decimals (0.0100 in every cell).

This is the weakest of the four positives. The untrained smoke run suggested 3–4.6×; on the
trained model it is roughly half that.

## The junk-residual trap, and why it is disarmed

An over-capacity FM saturates and `r` degenerates into architectural-mismatch noise — which
would *fake* this experiment's headline, since M can report noise (the head sits downstream of
`a_j`) while no third party can predict noise from tokens and logits. Two discriminators, and
both clear it:

- **`ens_cos` runs 0.777–0.911** and moves *inversely* with instrument capacity, while the
  advantage moves the same way. Junk would give a large advantage where `ens_cos` is *low*. We
  observe the opposite ordering at every point.
- **The residual reproduces the published behavioural signature.** Per-category Cohen's *d* on
  |r| at the default instrument, against the paper's table:

  | category | measured | published |
  |---|---|---|
  | sentence_start | −0.84 | −0.85 |
  | before_closer | +0.82 | +0.84 |
  | after_opener | +0.71 | +0.62 |
  | after_punct | −0.08 | −0.62 |

  The `after_punct` gap is a taxonomy difference, not a discrepancy:
  `behavioral_residual.classify_tokens_syntactic` uses *non-exclusive* masks, so
  `sentence_start` is a subset of `after_punct` (a period sets both). Our labels are
  priority-ordered and exclusive, so once the sentence-enders — which carry the published effect
  — are split into their own class, only commas/colons remain, and those are near-neutral.

### A methodological gotcha worth keeping

η² on the raw residual **vector** is the wrong yardstick and reads ~5e-4 here, which initially
looked like the junk signature. It is a *location* statistic (do the group-mean vectors differ),
while the published effect is a *scale* one (magnitude varies by category). A category sharing
the grand-mean direction but uniformly larger barely moves the group mean. `eta2_norm` — the
right statistic — is 0.022–0.033, ~60× larger. Both are now reported so the discrepancy stays
visible.

**A substantive corollary**: `eta2_dir` ≈ 0.0005. The conditioning is almost purely a magnitude
effect, so the residual's *direction* — which is what `IMPL`/`IMPL_COS` report — is **not**
recoverable from syntactic structure. That is favourable for criterion (2): a token-only
observer has no syntactic shortcut to the target, consistent with `O_input` sitting at
0.103–0.246 on `IMPL_COS`.

## Cross-substrate disagreement with RHM

**This is the main unresolved item.** The RHM sibling found the loop *does* matter; language
finds it does not. Same battery, same instrument sweep, same statistics.

| measure | RHM CL (`ntp_aux_cl`) | RHM OL (`ntp_aux`) | language CL | language OL |
|---|---|---|---|---|
| `IMPL` advantage | +0.091 … +0.109 | +0.058 … +0.065 | +0.211 … +0.374 | +0.210 … +0.370 |
| **CL − OL advantage** | **+0.026 … +0.051** | — | **−0.001 … +0.004** | — |
| confab margin | **+0.328 … +0.359** | +0.083 … +0.174 | +0.109 … +0.202 | +0.115 … +0.237 |
| steering ratio | 2.09 … 2.20 | 1.33 … 1.68 | 1.43 … 2.07 | 1.39 … 1.92 |
| residual η² (DGP / syntactic) | 0.032 … 0.036 | 0.008 … 0.016 | — | — |

On RHM the loop separates the arms on *every* secondary measure, most dramatically on the
confabulator margin (roughly 2–4×). On language it separates them on none. Note also that
language's *absolute* advantage is 3–4× RHM's — the language report is far more legible — so
this is not a weaker effect that the loop could have rescued.

Candidate explanations, in the order we would test them:

1. **The language loop is not load-bearing** (see [above](#the-most-likely-confound-and-it-is-ours))
   — CL is net *worse* than OL here, so there was no pressure to develop complementary
   self-knowledge. RHM's `ntp_aux_cl` was chosen precisely because it is the one RHM condition
   with generalizable self-knowledge. **This is the leading hypothesis and it is cheap to test.**
2. **Substrate**: RHM's residual is a deep DGP/inference-depth gap no 1-layer FM can compute;
   language's is a diffuse full-rank capacity gap. The loop may only add reportable structure
   when the residual has structure the FM is *architecturally* barred from reaching.
3. **The aux target**: RHM ran on a latent-supervised arm (`ntp_aux*`), language on pure NTP.
   The aux head may itself be what makes the loop's contribution reportable.

## What this does *not* establish

- The report head is **trained**. The claim is about the report's *causal grounding*, not its
  spontaneity. This is not spontaneous introspection.
- Use ≠ report ≠ awareness. Nothing here touches phenomenal consciousness.
- The introspection, if that is the word, belongs to the **composite** (M + FM), not to M alone.
  That matches the biology — cerebellar forward models are a separate structure — but it should
  be said rather than elided.
- **Single seed.** Given Result 2 is a null carrying real interpretive weight, it deserves a
  second seed before being leaned on.

## Next steps

1. **Make the language loop load-bearing, then re-run Result 2.** The cheapest decisive test:
   a 6-layer main model with `post_block0 → post_block3` and the report from `post_block5` —
   restores the canonical 3-layer gap *and* keeps two real blocks between target and report —
   plus a co-trained FM well below saturation (~1% of the model, not 42% of the predicted
   blocks). If CL then beats OL on net val loss and the advantage delta *stays* at zero, the
   null is real and the RHM disagreement is substrate, not strength.
2. **Second seed on both arms**, for the null specifically.
3. **Resolve the RHM disagreement** by porting language's pure-NTP arm to RHM (or RHM's aux arm
   to language) — isolates explanation 3 from explanation 2.
4. **Harden steering** if it is to carry weight: more PCs, several target-KL levels to check the
   ratio is not KL-specific.
5. **The token channel.** Everything here uses an aux head; the design doc's variant (b) routes
   the report through M's own LM head over a reserved report vocabulary, which is the version
   that is actually a *report* rather than a probe.

## Reproduction

```bash
cd experiments/
modal profile activate jagilley      # token shards live on this workspace, not chromatic

# headline: both arms, full battery (~4 h on an L4; wake phase is checkpointed)
modal run --detach a2a_forward/confabulation/confabulation.py::confabulation_test \
    --conditions "cl,ol" --tag main

# just the two junk-residual discriminators, from saved wake checkpoints (~10 min)
modal run a2a_forward/confabulation/confabulation.py::residual_diagnostics \
    --conditions "cl,ol" --tag diag

# wiring check (minutes; numbers meaningless)
modal run a2a_forward/confabulation/confabulation.py::confabulation_test --smoke
```

Results: `/data/a2a_forward/confabulation/main_results.json` and
`diag_residual_diagnostics.json` on the `language-reduction-data` volume; wake checkpoints under
`wake_ckpt/`.

**Gotchas** (details in [FILES.md](FILES.md)): `predict_to` must leave a real block before the
report site or Tests 2/3 degenerate — a runtime assert catches the related case where an
injection lands between `predict_to` and `report_block`, since `intermediates[post_block{i}]` is
recorded *before* the injection is added. `O_io` gets top-k ids/probs **plus four
full-distribution scalars**; without them it cannot compute M's entropy and the `ENT` control
shows a fake advantage. Steering must run in sequence-chunks — a full-sequence log-prob tensor
is 6.6 GB at 50257 vocab.
