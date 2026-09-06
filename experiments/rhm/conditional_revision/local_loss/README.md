# The temporal local loss: does an exogenous conditioning gap change what a local loss does?

**Up**: [../README.md](../README.md) · **Files**: [FILES.md](FILES.md) · **Design doc**: [SPEC.md](SPEC.md)
**Idea**: [`ideas/revision_not_surprisal.md`](../../../../ideas/revision_not_surprisal.md) §8
**Incumbent being compared against**: [`../../RHM_FM_REGULARIZER_README.md`](../../RHM_FM_REGULARIZER_README.md)
**Date**: 2026-08-08 · **Status**: run. One regime (m2), one axis (self-knowledge).
The epistemic-content readouts this cut was ultimately *about* were **not** computed — see
[What this does not establish](#what-this-does-not-establish).
**Followed up 2026-08-09**: [`../aleatoric_fraction/`](../aleatoric_fraction/README.md) sizes the prize
this cut was pushing on (headroom 0.0078 against a ceiling of 0.045), and
[`../synonym_retention/`](../synonym_retention/README.md) runs a scale-free content-axis version of the
arm comparison (a null, not gauge-confounded) and **re-scopes Finding 4's `syn/str`**.

## What we tried

Every local loss in this repo — `λ‖sg(FM(h_src)) − h_tgt‖²` added to NTP — has made the model more
legible to its forward model without making it more self-knowing. The idea doc's §8 proposed that this
is forced rather than empirical: `h_src` and `h_tgt` are both deterministic functions of the same input
through the same weights, so the only way to lower the term is to move `h_tgt` into the image of a
low-capacity FM. On that reading, "minimize prediction error" is definitionally "be simpler," and the
fix is to give the target a conditioning gap the predictor cannot close.

```
depth    (incumbent):  FM( post_embed[t]   ) → post_block6[t]      # same input; gap = none
temporal (this cut) :  FM( post_block6[≤t] ) → post_block6[t+1]    # gap = exactly one token
```

[Gate 0](../README.md#gate-0--the-axis-change-does-something-and-in-the-predicted-direction) had already
shown the temporal residual has a real aleatoric component on this program's substrate
(`corr(res, nll)` −0.337 → +0.652, with a protocol-matched control at −0.328). The question here is
narrower: **used as a training signal, does the temporal target produce a different degeneracy than the
depth target?**

## Design

A pure target swap on [`../../rhm_fm_regularizer.py`](../../rhm_fm_regularizer.py), which is already the
LL-alone, open-loop, gradient-into-the-main-model-only version of this objective on RHM.

- **Substrate**: m2 (`v16 s2 L6 m2` distinct-rule, 8L/8H/256D, wd=0.1) — chosen because the incumbent
  depth λ-sweep and the λ=0 anchor live there, so the depth arm needs no retraining and keeps its
  original protocol. This choice has a cost; see limitations.
- **Arms**: depth (incumbent checkpoints, read-only) and temporal, λ ∈ {0.1, 1.0, 3.0}, λ warmup
  2000 steps at 0 then a 3000-step ramp.
- **Matched batches.** The FM config is unchanged, so the torch RNG stream matches and the temporal arms
  see the same batches step-for-step as the incumbent. Confirmed by identical val/regMSE during warmup.
- **Read at two points**: the matched step (80k, where checkpoints exist for both arms and the λ=0
  anchor) *and* matched achieved compression (FM-free activation rank), because matched λ is not matched
  pressure — the temporal residual's irreducible floor supplies gradient the depth term loses as its
  error goes to zero.
- **Both forward models on every checkpoint** (fresh, seed 911 ≠ training seed), so "does the temporal
  loss buy depth legibility, and vice versa" is a 2×2 rather than two separate scales.

**Instrument check**: the reimplementation reproduces the incumbent exactly — λ=1.0@300k **45.7%**
(published 45.7), λ=3.0@300k **42.9%** (42.9), λ=0@125k **54.1%** (54.3).

## Finding 1 — most of what the term does is a gauge

The depth arm's local-loss term falls **19×** across the λ range while scale-invariant predictability
improves **1.9×**; the temporal arm reads 19× and **1.6×**. In both arms ~90% of the term's dynamic
range is the *target shrinking*, not the forecast improving. Per-dim activation variance falls ~300× in
the depth arm at ~10 points of rank movement, and the two are separable (λ=0.1: 12× variance drop, zero
rank movement).

This appears to be a gauge rather than a pathology: scale is discarded by every downstream reader of
`post_block6` — `block7.ln_1`, `ln_f` before `lm_head`, the knowledge probe (which standardizes), and
η² (a variance fraction) — which is consistent with `d6` holding at 0.93–0.95 throughout. `‖FM − tgt‖²`
is the one quantity in the system that is not scale-free, and it falls quadratically in scale.

**`RHM_FM_REGULARIZER`'s published headline is unaffected**: that claim rests on FM-free activation
*rank*, which is scale-invariant and moves independently. Nothing there is revised. What this adds is
that **the raw local-loss term is a poor readout of anything**, since most of its movement is a gauge.

**The collapse is prompt, not terminal.** It happens as λ ramps on, while the task is still being
learned — not after saturation:

| step | val | tgtVar (λ=3.0) |
|---|---|---|
| 2000 (λ still 0) | 0.9011 | 1.2011 |
| 4000 (ramp on) | 0.8619 | **0.0341** |
| 8000 | 0.8393 | 0.0157 |
| 28000 | 0.8261 | 0.0111 |
| 80000 | 0.8235 | 0.0076 |

~35× in the ~2000 steps of the ramp while val moves 0.90 → 0.86, then a further 2× over the next 72k
steps. So this is not an artifact of continued training on a saturated corpus. Whether a curriculum
would change it is untested, but the mechanism does not appear to be "ran out of things to learn."

## Finding 2 — the temporal target compresses more, not less

At matched λ and a quarter of the steps, the temporal arm reaches lower FM-free activation rank than
the incumbent: **39.4%** at λ=3.0/80k against the depth arm's 42.9% at λ=3.0/300k; temporal λ=1.0 at 80k
(42.6%) matches depth λ=3.0 at 300k. Knowledge is preserved throughout (d6 0.93–0.95).

A candidate explanation consistent with the design: the temporal residual has an irreducible floor, so
the term never reaches zero and keeps supplying gradient, where the depth term's pressure decays as its
error does. This was not isolated — matched-λ pressure differences are exactly the confound the
matched-compression view exists to route around, and it does not distinguish *why* the pressure differs.

## Finding 3 — self-knowledge, read with both forward models

Matched-compression view (sorted by FM-free activation rank). `T*` = read with a fresh temporal FM,
`D*` = with a fresh depth FM. `meta` is the orthogonalized-residual probe — does M represent where its
own forward model will err. Negative R² = worse than predicting the mean.

| arm | actRk% | d6 | syn/str | `Tmeta` | `Dmeta` |
|---|---|---|---|---|---|
| `lam0_base` | 53.4 | 0.93 | 0.635 | **+0.551** | **+0.121** |
| `depth_lam0p1@150k` | 53.2 | 0.95 | 0.660 | +0.463 | −0.235 |
| `depth_lam0p1` | 52.7 | 0.94 | 0.611 | +0.469 | −0.273 |
| `depth_lam3p0` | 48.1 | 0.93 | 0.586 | +0.174 | −1.518 |
| `depth_lam1p0` | 47.9 | 0.94 | 0.595 | +0.311 | −1.111 |
| `temporal_lam0p1` | 46.2 | 0.94 | 0.578 | −0.114 | −0.163 |
| `depth_lam1p0@300k` | 45.7 | 0.95 | 0.623 | +0.307 | −0.682 |
| `depth_lam3p0@300k` | 42.9 | 0.94 | 0.596 | +0.271 | −0.929 |
| `temporal_lam1p0` | 42.6 | 0.95 | 0.552 | **−0.693** | −0.426 |
| `temporal_lam3p0` | 39.4 | 0.93 | 0.567 | **−2.630** | −0.917 |

Each loss reduces generalizable self-knowledge most on **the axis it optimizes**: the depth loss takes
`Dmeta` +0.121 → −0.929, the temporal loss takes `Tmeta` +0.551 → −2.630. At matched compression
(42.9 vs 42.6) the temporal arm reads −0.693 on the temporal FM where the depth arm reads +0.271.
Cross-axis the temporal arm is *milder* (−0.426 vs −0.929 on `Dmeta`), which is what one would expect
of an axis that was not optimized.

This is the shape `RHM_LATENT_LOOP` recorded for token-CL — a small, FM-idiosyncratic residual that a
fresh FM's error anti-predicts. **On this axis and this substrate, the exogenous conditioning gap did
not produce a different outcome from the endogenous one; it produced the same one, further along.**

## Finding 4 — no evidence of selective synonym discarding

The sharpest thing the conditioning gap could have done is make the representation drop *which* synonym
realized a latent (irreducible on RHM) while keeping the latent (reducible). Three views, with the depth
arm as a measured null:

- **Decodability**: the per-level rule probe reads **1.000 in every arm at every λ and step**. No
  condition breaks synonym decodability. This instrument is saturated and can only register catastrophic
  movement.
- **Allocation** (`rule/feat` η² at d1, graded and probe-free): base 0.0059; depth 0.0092 / 0.0069 /
  0.0046; temporal 0.0035 / 0.0047 / 0.0036. Temporal sits below base, but non-monotonically in λ and
  within the range depth spans.
- **Causal** (matched counterfactual twins): `syn/str` 0.635 base → 0.552 (temporal λ=1.0) vs 0.596
  (depth λ=3.0@300k) at matched compression. The direction is as predicted but the effect is small,
  non-monotone in λ, and accompanied by *both* twins' displacement falling (synRel 0.128 → 0.089,
  strRel 0.201 → 0.161) — i.e. general desensitization rather than selective discard.

Deep feature η² rises over base in both arms (d6 0.2425 → 0.3046 temporal λ=3.0, 0.2961 depth
λ=1.0@300k), so neither arm is degenerately collapsing; this reproduces the incumbent's
"functional compression, not collapse" and does not distinguish the arms.

> **Re-scoped 2026-08-09 by [`../synonym_retention/`](../synonym_retention/README.md).** `syn/str = 0.635`
> **reproduces** (0.638 on `lam0_base` at the same cell) and is **distance-robust** (0.62–0.73 on m4 across
> read distances `w = 0…16`), so it is not an artifact of the read position. What it cannot support is the
> *inference*: `str` is `leaf` **plus** a rule bump — strictly a larger perturbation of the same span — and
> both arms hold the constituent's own feature fixed, so the ratio compares two nested perturbation
> magnitudes and neither arm isolates the latent. The cell is also flagged `ntp_defined: False` (the read
> position is the last token, so NTP constrains that state not at all). Finding 4's stated conclusion — no
> evidence of selective synonym discarding *attributable to the loss* — is **unaffected**, and is
> corroborated: on the distance axis the discarding is real and large (rule retention +0.858 at `w=0` →
> chance by `w=16`, ~3× faster than feature identity) and is **already present at λ=0**, with the arm
> comparison at matched compression reading +0.392 depth vs +0.393 temporal.

## What this does not establish

This is the part worth reading carefully. The cut answers a narrower question than the one that
motivated it.

- **It scores the wrong axis for the question we care about.** `meta`/SK is the *local-loss arc's*
  scoreboard — the original claim was that a local prediction-error signal would build self-knowledge,
  and the prior nulls were SK nulls. It measures legibility of the model to its own FM. It says nothing
  about whether the temporal residual carries **epistemically charged content**. The readouts that would
  — Gate B's reducible/irreducible AUC and Gate A's partial `R²` computed on these *trained*
  checkpoints — were not run. Until they are, this cut does not bear on the aleatoric-null claim.
- **m2 is knowledge-saturated** (d6 0.93–0.95, val flat from ~16k) and every `conditional_revision`
  measurement was on **m4**, where the model's frontier exists (root recovery 0.088). The substrate was
  chosen for its anchors, not for the question. Transfer is untested. `RHM_LATENT_LOOP` did obtain its
  sharpest target-axis SK result at m2 (token ΔSK −1.54), so the axis is sensitive there — but that is
  an argument for the comparison being fair, not for it generalizing.
- **Matched λ is not matched pressure**, and the matched-compression view routes around that without
  explaining it. The two arms differ in more than the conditioning gap.
- **No curriculum.** One fixed corpus throughout. Whether any of these dynamics survive a setting where
  new compressible-but-unlearned structure keeps arriving is not addressed here.
- **One regime, one model size**, throughout.
- The gauge reading of Finding 1 is a mechanism *consistent with* the data, not one isolated by an
  intervention. A scale-invariant form of the objective would test it directly and was not run.

The narrow statement the data supports: **on the self-knowledge axis, at m2, the temporal target
reproduced the depth target's signature rather than escaping it, and did so more strongly at matched
compression.** Whether that is because the conditioning gap is inoperative, because the gauge exit
dominates both arms, or because m2 has no frontier for an epistemic signal to be about, this cut does
not separate.

## Reproduction

```bash
cd experiments

# smoke (attached, ~6 min)
modal run -m rhm.conditional_revision.local_loss.temporal_local_loss::smoke

# temporal arms (3 λ × 80k steps, ~70 min on L4; the depth arm is NOT retrained)
modal run --detach -m rhm.conditional_revision.local_loss.temporal_local_loss::tll_sweep --arm temporal

# FM-free rank trajectory over the incumbent depth checkpoints (read-only, no FM trained)
modal run --detach -m rhm.conditional_revision.local_loss.temporal_local_loss::rank_traj --arm depth

# the analysis (both FMs on every checkpoint; ~25 min, 11 containers)
modal run --detach -m rhm.conditional_revision.local_loss.temporal_local_loss::analyze_ll \
    --out-tag ll1 --extra "depth_lam0p1@150000,depth_lam1p0@300000,depth_lam3p0@300000"
```

Results land on the `rhm-scaling-data` volume (**`chromatic` workspace**) under
`/data/rhm_temporal_local_loss/parts_ll1/`. Temporal checkpoints are at
`/data/v16_s2_L6_m2_distinct/tll_*_lam*/`. The incumbent depth checkpoints originate on **`jagilley`**
and were copied to `chromatic`; `_fmreg_dir` and `_sweep_dir` are read-only here and the `tll_` prefix
exists so the incumbent can never be clobbered.

## Gotchas worth not rediscovering

- **The raw local-loss term is ~90% gauge.** Do not read `regMSE` as legibility. Use a scale-free ratio.
- **Matched λ is not matched pressure** when the two targets have different irreducible floors. Compare
  at matched achieved compression.
- **80k is not the depth arm's endpoint.** At 80k the depth collapse is only ~46–62% complete and sits
  at a local maximum of a ±3-point-noisy trajectory; a matched-step read alone understates it. The
  incumbent's 150k/300k endpoints are read alongside for this reason.
- **The rule probe saturates at 1.000** and cannot show graded movement. Use the η² allocation readout
  for anything short of catastrophic change.
- **Counterfactual twins need matched perturbation size**, or a sensitivity difference is just a
  perturbation-size difference. Both twins are legal DGP productions with overlapping Hamming
  distributions (3.56 vs 3.75) and bit-identical prefixes. **Matching is not enough when the arms are
  nested** — `str` = `leaf` + a bump on the same span, so `syn/str` is a ratio of two magnitudes rather
  than a selectivity readout; see the re-scoping note in Finding 4.
- **A retention readout needs a read position outside the perturbed span.** `_swap_sensitivity` reads the
  last position of the sequence while perturbing a span that contains it, so NTP places no constraint on
  that state. [`../synonym_retention/`](../synonym_retention/README.md) supplies the distance axis.
- **The temporal arm can raise cosine by smoothing** (`h6[t+1] ≈ h6[t]`) with no conditioning effect at
  all, because its source is its target block. Read predictability jointly with `delta_over_tgt`.
