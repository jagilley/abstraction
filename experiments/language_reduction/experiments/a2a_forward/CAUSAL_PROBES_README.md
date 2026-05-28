# Causal Probes of the Closed-Loop "Self-Map" (2026-05-28)

**Idea doc**: [ideas/activation_to_activation_forward.md](../../../../ideas/activation_to_activation_forward.md)
**Prior experiment**: [CLOSED_LOOP_README.md](CLOSED_LOOP_README.md)

## Why this exists

`CLOSED_LOOP_README.md` reported that linear probes predicting the forward
model's residual (actual − predicted) from `post_block3` reach **R²=0.44 for the
closed-loop model vs 0.28 for the open-loop baseline**, and interpreted this as
the model developing "novelty awareness" — a *self-map* of what was surprising
in its own computation, and the basis for a "cheap interpretability" story.

That result is a **decoding** result: a probe *we* train can extract more
residual information from the closed-loop model. The leap to "the model encodes
and *uses* knowledge of its own reliability" is a separate claim. This file
records causal follow-ups that test that leap. **Headline: the self-map's
magnitude/gating form is not supported; its directional form is untested, not
refuted; and the R²=0.44 finding itself is real but confounded — see below.**

## Setup

All tests use the existing checkpoints (no retraining): closed-loop
`loop_L2/post_block0_to_post_block3/inject1/P_10000000` and open-loop baseline
`transformer_L2/post_block0_to_post_block3/P_10000000`. Main model: 4-layer,
4-head, 256-dim GPT (28.9M). Forward model: 2-layer, 1-head, 64-dim (660K).
Injection: `gate(fwd_model(post_block0))` added after block 1. "Novelty" /
residual = `post_block3 − fwd_pred` on the no-injection pass (post_block0 is
identical with/without injection, so the prediction is unchanged).

## Test 1 — Probe (does the structure exist, and where?)

`novelty_probe.py`. Linear probes from each layer to (a) named syntactic
categories (logistic, AUC) and (b) the residual norm (linear, R²), open vs
closed on identical inputs.

| Target | block1 | block2 | block3 |
|---|---|---|---|
| residual_norm R² — open | 0.390 | 0.393 | 0.379 |
| residual_norm R² — closed | 0.430 | 0.440 | 0.424 |
| block1_contrib R² — Δ (closed−open, control) | −0.061 | −0.101 | −0.053 |

Syntactic categories (before_closer, sentence_start, …) decode at **~0.97–0.99
AUC in both models** — no closed-loop advantage (ceiling, and these are syntactic
context, not "novelty").

**Read.** The closed-loop model decodes its residual norm ~+0.04 R² better — same
*direction* as the R²=0.44 finding, smaller magnitude (this is the scalar norm,
not the full vector). But: (1) the gap is **flat across layers, present already
at post_block1** — which is computed *before* the injection enters, so it cannot
have been built by "using the returned prediction"; this looks like a global
training difference, not a loop-built downstream self-map. (2) The comparison is
**confounded**: open lr=1e-4 vs closed lr=3e-4, and the two forward models differ
in quality (cosine 0.935 vs 0.897), so the closed-loop residual is intrinsically
larger and easier to decode. The control direction (block1_contrib) goes the
*other* way, so it is not a uniform "better probe target" effect — but the
confounds are not removed.

## Test 2 — Steering (is the structure *used* to gate the prediction?)

`novelty_steer.py`. Found the post_block1 direction that predicts residual norm
(probe R²=0.43), steered the stream by ±s·σ along it, and measured reliance on
the injection, `R(s) = KL(p_with_inj ‖ p_no_inj)` (steering applied to both
passes, so its direct logit effect differences out).

| | value |
|---|---|
| corr(reliance, residual_norm) at s=0 | **−0.015** (≈0) |
| slope dR/ds — novelty direction | +0.0015 |
| slope dR/ds — block1_contrib (control) | +0.0016 (**identical**) |
| slope dR/ds — random (control) | −0.0004 |

**Read.** No novelty-gated reliance: at rest, reliance is uncorrelated with
novelty; under steering the effect is tiny, wrong-signed, and **identical to a
non-novelty control direction** (not novelty-specific). Reliance is ~uniform
(0.090–0.099 across ±3σ). **Caveat:** this used a single *decoding* direction
(R²=0.43); a decoding direction need not be the causal one, and a scalar/1-D
probe is blind to directional structure. So this is evidence against
*magnitude*-gating, not a clean refutation of all self-use.

## Test 3 — Δloss-by-novelty (where does the injection actually help?)

`injection_help.py`. Per token, `help = loss_no_inj − loss_inj` (positive =
injection helps), vs novelty. The "division of labor" story predicts help should
be *largest at low novelty* (prediction accurate → trust it).

| | value |
|---|---|
| mean help | **+0.086 nats** (confirms the closed-loop benefit) |
| corr(help, residual_norm) | +0.021 (≈0, *not* negative) |
| help: Q1 (low nov) → Q4 (high nov) | +0.080 → +0.083 → +0.083 → +0.099 |

**Baseline-loss control** (`injection_help.py`): high-novelty positions have
*lower* baseline loss (Q4 5.32 vs Q1 5.69), so this is not a headroom artifact;
loss-decile-stratified corr(help, novelty) = +0.022 (≈ raw), within-bin Q4−Q1 =
+0.031 (≥ raw). **Read.** Help does *not* peak at low novelty — division of labor
as stated is not what's happening.

## Test 3-enriched — help by residual *structure*, not magnitude

`injection_help_structural.py`. The residual norm is a lossy summary; this
conditions help on the residual's *direction* and the *kind* of computation.

| organizing variable | η² of help | help spread across groups |
|---|---|---|
| residual-direction clusters (k=8) | **0.0017** | **0.089** |
| residual-norm octiles (8) | 0.0003 | 0.028 |

| attention shape | mean help |
|---|---|
| focused (max attn > 0.5) | **+0.153** (≈2×) |
| distributed (high entropy) | +0.081 |
| distant (mass >10 back) | +0.080 |

| attention distance to dominant token | mean help |
|---|---|
| 0–1 (local) | +0.090 |
| >10 (long-range) | +0.077 |

**Read.** Residual **direction** organizes the help ~5× more than **magnitude** —
collapsing to the norm discards most of the signal. The organizing variable is
**attention shape**: the injection helps most at **focused / sharp-retrieval**
positions and *least* at distributed/long-range ones (help *decreases* with
attention distance, even within norm-deciles: long-range − local = −0.012). The
one high-help k-means cluster is the focused, local, low-entropy one — and it
also has the *largest* residual norm, so "forward model most wrong (in
magnitude)" and "injection most useful" coincide and are both organized by
focused-attention structure, not error size.

**Caveats.** Absolute η² is tiny — per-token Δloss is dominated by noise; effects
live in robust group means. "Focused attention (max>0.5)" is coarse and may
include attention-sink/delimiter positions (not disambiguated). `before_closer`
is a striking outlier (+0.358 help, low baseline loss 3.07) but tiny (n=1,629),
so it is idiosyncratic, not representative.

## Considered and rejected

An intermediate reading of Test 3 framed the injection as a **"long-range
computational scaffold"** (helps most at hard, long-range positions). The
enriched structural analysis **overturned this**: help *decreases* with attention
distance and concentrates at focused/local positions. Recorded here so it is not
rediscovered and re-believed.

## Status of the self-map claim (R²=0.44 vs 0.28)

- The number is **real**; we partially reproduced its direction (Test 1). We did
  **not** show it is erroneous.
- Its self-map *interpretation* is **confounded** (lr; differing forward-model
  quality) and the probe is mechanically weak (it predicts `actual − pred` *from*
  `post_block3 = actual`, so part of the R² is structural and the cross-model gap
  is sensitive to forward-model quality).
- The **magnitude/gating** form of the self-map is **not supported** (Tests 2, 3).
- The **directional** form — exactly what the full-vector R²=0.44 detects, and
  where the enriched Test 3 says the signal lives — is **untested, not refuted**.
  Our causal tests used the scalar norm or a single direction and were blind to it.

### What would settle it
1. **Controlled retrain** — identical lr / seed / forward model, open vs closed.
2. **Clean early-layer probe** — predict the residual from `post_block0/1` (which
   do *not* contain the answer): does the model *anticipate* its own surprise
   before computing it? That cannot be mechanical.
3. **Directional causal test** — steer/patch along the residual-*vector*
   structure (not the scalar norm) and test whether it changes prediction use.

## Files

| File | Purpose |
|---|---|
| `novelty_probe.py` | Test 1 — layerwise probes, named categories + residual norm, open vs closed |
| `novelty_steer.py` | Test 2 — causal steering of the novelty direction; reliance R(s) |
| `injection_help.py` | Test 3 — Δloss-by-novelty + baseline-loss control |
| `injection_help_structural.py` | Test 3-enriched — help by residual direction / attention structure |

Results JSON: `/data/a2a_forward/analysis/{novelty_probe,novelty_steer,injection_help,injection_help_structural}_results.json` on the `language-reduction-data` volume.

## Reproduction

```bash
cd experiments/
COMMON="--n-tokens 10000000 --predict-from post_block0 --predict-to post_block3 --fwd-n-layer 2 --inject-after-block 1"
modal run language_reduction/modal_app.py --stage a2a-novelty-probe $COMMON
modal run language_reduction/modal_app.py --stage a2a-novelty-steer $COMMON
modal run language_reduction/modal_app.py --stage a2a-injection-help $COMMON
modal run language_reduction/modal_app.py --stage a2a-injection-help-structural $COMMON
```

## Overall caveats

28.9M-param main model, 10M tokens, single checkpoint, single (non-looped)
injection point. The self-map / metacognition hypothesis's natural home is the
*looped/iterated* transformer (see idea doc); this one-shot setting is a weak
test of it. Read the negatives as "not present in this instantiation," not as a
refutation of the idea.
