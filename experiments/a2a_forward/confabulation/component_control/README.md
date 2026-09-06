# The component control — is the advantage about the residual, or about having the state?

**Status**: done, language substrate only. **Positive**: the I/O observer reaches
~0.87 of the achievable headroom on the theory-visible component and ~0.84 on the whole state,
but only ~0.19–0.26 on the residual. The forward-model decomposition is load-bearing for
Finding 1, not only for Finding 3.

**Parent**: [../README.md](../README.md) — the confabulation battery this extends
**Design origin**: raised in conversation while auditing
papers/forward_self_models_paper2.md[^private]; not in
the original [design doc](../../../../ideas/confabulation_test.md)
**Code**: [../confabulation.py](../confabulation.py) behind `--component-control` (off by
default). No new script — the control is the same battery with two extra report targets, and
duplicating the Observer/ladder machinery would have made the contrast unfair.

## The question

The parent battery shows that no affordable third party predicts M's forward-model residual as
well as M reports it. But the report head reads `post_block3`, downstream of `a_j`, and the
observers read only tokens and outputs. So the gap could be an advantage of **having the state
at all** rather than an advantage of having the *un-theorized part* of the state. If so the
decomposition would be a framing device for Finding 1 — still load-bearing for the matched-KL
steering of Finding 3, which contrasts the two subspaces directly, but not for the observer
ladder.

The parent's ablation columns (`shuffle_r` collapses the report, `zero_p` does not) cannot
settle this: the target *is* the residual, so needing `r` to predict `r` is near-circular. What
settles it is varying which function of `a_j` is reported while holding everything else fixed.

## What was added

Two report targets built with **identical machinery** to `IMPL` — same spherical k-means, same
`impl_k=8`, same head, same observer ladder, same train/test split, same seeds — differing only
in which component of `a_j` is reported:

| target | reported quantity | shortest third-person route |
|---|---|---|
| `IMPL` | direction of `r` | cross the whole predicted span |
| `PRED` | direction of `FM(a_i)` | tokens → `a_i` is one block, then re-fit a small FM |
| `AJ` | direction of `a_j` | cross the whole predicted span |

plus `_COS` continuous variants of each, scored by test cosine.

### Why raw advantage is the wrong statistic here

`self` saturates on `PRED`/`AJ` (0.91–0.96) and does not on `IMPL` (0.51–0.69) — the report site
sits downstream of `a_j` and the residual stream is additive, so recovering `a_j`'s cluster is
nearly free from inside. Raw advantage (`self − best O_io`) therefore confounds *how private the
fact is* with *how well the self can report it at all*. So the run also reports

```
frac = (best O_io − chance) / (ceiling − chance),   ceiling = max(self, O_act)
```

the fraction of the achievable-above-chance score an I/O-only observer reaches. Lower = more of
the fact has to be paid for by reconstruction. `O_act` (an observer handed `a_i`) is the natural
denominator because it controls for target difficulty on the same target, but on categorical
`IMPL` it does not reach the self-report, so the ceiling falls back to `self` there.

## Results

CL arm, both instrument capacities. `chance` is the majority-class fraction on test.

| instrument | target | self | best `O_io` | raw adv | `O_act` | chance | **frac** |
|---|---|---|---|---|---|---|---|
| h16m0.5 (10.6%) | `IMPL` | 0.511 | 0.239 | +0.272 | 0.267 | 0.143 | **0.260** |
| | `PRED` | 0.943 | 0.847 | +0.096 | 0.937 | 0.187 | 0.873 |
| | `AJ` | 0.958 | 0.833 | +0.124 | 0.900 | 0.177 | 0.841 |
| | `IMPL_COS` | 0.590 | 0.272 | +0.318 | 0.578 | 0 | **0.462** |
| | `PRED_COS` | 0.907 | 0.696 | +0.211 | 0.961 | 0 | 0.725 |
| | `AJ_COS` | 0.956 | 0.659 | +0.297 | 0.922 | 0 | 0.690 |
| h4m0.25 (4.9%) | `IMPL` | 0.599 | 0.225 | +0.374 | 0.291 | 0.138 | **0.189** |
| | `PRED` | 0.942 | 0.847 | +0.095 | 0.947 | 0.194 | 0.867 |
| | `AJ` | 0.958 | 0.821 | +0.137 | 0.900 | 0.177 | 0.825 |
| | `IMPL_COS` | 0.693 | 0.348 | +0.345 | 0.681 | 0 | **0.503** |
| | `PRED_COS` | 0.905 | 0.708 | +0.198 | 0.966 | 0 | 0.733 |
| | `AJ_COS` | 0.956 | 0.659 | +0.297 | 0.922 | 0 | 0.690 |

**The residual is the expensive component.** On the categorical targets the separation is
3.2–4.6× and holds at both capacities. It also holds on the *raw* advantage, which is the
conservative reading: `IMPL`'s raw advantage is the largest of the three despite its self-report
being the lowest by ~0.35–0.45.

### Two supporting observations

**The token-only ladder climbs on `PRED` and not on `IMPL`.** At h16m0.5, `O_input` across
1L/64D → 2L/128D → 4L/256D:

| target | 1L/64D | 2L/128D | 4L/256D | chance |
|---|---|---|---|---|
| `IMPL` | 0.209 | 0.222 | 0.199 | 0.143 |
| `PRED` | 0.770 | 0.787 | 0.783 | 0.187 |
| `AJ` | 0.732 | 0.750 | 0.741 | 0.177 |

A tokens-only observer nearly solves the theory-visible component at the *smallest* capacity on
the ladder, and stays near the floor on the residual at every capacity. The parent README has to
borrow this "the ladder can climb when the fact is reachable" control from the `WORLD` target;
here it exists inside the implementation family, holding the report machinery and the target
construction fixed.

**`O_act` overshoots the self-report on the theory component.** 0.961 vs self 0.907 on
`PRED_COS` and 0.947 vs 0.942 on `PRED`, against 0.578 vs 0.590 on `IMPL_COS`. An observer handed
`a_i` matches or beats M on the part a theory can supply, and only there.

## Where this is weaker than it looks

**On the continuous targets the separation depends on the normalization.** Raw advantages are
`IMPL_COS` +0.318 against `AJ_COS` +0.297 — essentially tied. `frac` separates them (0.462 vs
0.690) only because `AJ_COS`'s ceiling is 0.956 against `IMPL_COS`'s 0.590. So on `_COS`, "the
residual is harder from outside than the whole state" is a claim about headroom-normalized
difficulty, not about raw advantage. (`IMPL_COS` vs `PRED_COS` does separate on the raw number:
+0.318 vs +0.211.) The categorical rows carry no such dependency — +0.272 vs +0.124 raw, 0.260 vs
0.841 normalized, same conclusion either way — so the claim rests on those, with `_COS`
corroborating under normalization.

**`AJ` tracks `PRED`, not `IMPL`.** Expected rather than awkward: at forward-model cosine 0.915
the direction of `a_j` is mostly the direction of `FM(a_i)`, so "the whole state" is mostly
theory-visible. But it does narrow the defensible sentence. The result is not "internal state is
private" — most of the state is comparatively cheap from outside. It is specifically the
un-theorized part that is expensive.

**Cluster separation differs across targets.** The residual's k-means separation is 0.174 against
`PRED`'s 0.279 and `AJ`'s 0.257, so `IMPL` is a weaker categorical target in absolute terms.
This cuts *against* the headline rather than for it — a weaker target should be harder for the
self-report too — but it is a difference between the arms and should be said.

## Harness validation

The run reloads the published battery's wake checkpoint and re-runs `IMPL`/`IMPL_COS` in the same
job, so the contrast is within-run. Every shared quantity reproduces the parent README:

| quantity | this run | parent README |
|---|---|---|
| wake val standalone / injected | 5.4458 / 5.3518 | 5.446 / 5.352 |
| `ens_cos` h16m0.5 / h4m0.25 | 0.854 / 0.904 | 0.854 / 0.904 |
| `IMPL` advantage h16m0.5 / h4m0.25 | +0.272 / +0.374 | +0.272 / +0.374 |
| `IMPL_COS` advantage h16m0.5 / h4m0.25 | +0.318 / +0.345 | +0.318 / +0.345 |
| `eta2_norm` h16m0.5 | 0.0221 | 0.022–0.033 |

A determinism check falls out for free: `AJ_COS` is identical to three decimals across the two
capacity rows (0.956 / 0.659 / 0.922), as it must be — `a_j`, the report input and the observer
inputs are all instrument-independent.

## Limitations

- **Single substrate, CL arm only.** The parent's CL/OL null makes the arm choice
  low-stakes here, but it is untested.
- **Language only.** The RHM sibling has a 6-block predicted span against language's 2, i.e. a
  much longer reconstruction route, and the parent battery's cross-substrate agreement is load-
  bearing elsewhere in this program. Not run.
- **Two capacities, not the full four-point sweep.** h32m1 and h64m2 were skipped for cost.
- The fixed targets (`BEHAV`/`ENT`/`WORLD`) and matched-KL steering were skipped via
  `--skip-fixed-targets --skip-steering`; they do not depend on the instrument and are unchanged
  from the parent run.

## Reproduction

```bash
cd experiments/
modal profile activate jagilley      # token shards live on this workspace, not chromatic

# ~70 min on an L4; reloads the published run's wake checkpoint rather than retraining M
modal run --detach a2a_forward/confabulation/confabulation.py::confabulation_test \
    --conditions cl --inst-caps-str "16:0.5,4:0.25" \
    --component-control --skip-fixed-targets --skip-steering --tag component_control

# wiring check (minutes; numbers meaningless -- ens_cos ~0.6, the junk regime)
modal run a2a_forward/confabulation/confabulation.py::confabulation_test --smoke \
    --component-control --skip-fixed-targets --skip-steering --conditions cl --tag ccsmoke
```

Results: `/data/a2a_forward/confabulation/component_control_results.json`.

**Gotcha**: `--skip-steering` alone is not enough to skip steering — steering needs the `BEHAV`
head as its matched-behaviour control, so it is additionally gated on `--skip-fixed-targets`.
`frac` is printed in the summary but not stored in the JSON (the summary runs after the dump);
it is recomputable from `self`, `best O_io`, `O_act` and the saved baselines.

[^private]: Not mirrored: this link points to a document in the private lab repo (the roadmap, the queue, an unrun spec, reading notes, or a conversation). See the top-level README for what is held back and why.
