# Temporal FM epistemics — does the residual carry revision, and does the FM concentrate it?

**Design doc**: [ideas/temporal_confabulation_test.md](../../../../../ideas/temporal_confabulation_test.md), §*The next thing to run: temporal FM epistemics, on this harness* — pre-registered there, from the parent's own results, before this code was written
**Up**: [../README.md](../README.md) — the temporal confabulation test, whose harness this keeps and whose report channel it drops
**Lineage**: [`../../../conditional_revision/`](../../../conditional_revision/README.md) — the Gate-0 temporal FM idiom, the exact BP oracle, Gate B's atom-aware matching, [SPEC.md](../../../conditional_revision/SPEC.md)'s never-run **Gate C**, and [`aleatoric_fraction/`](../../../conditional_revision/aleatoric_fraction/README.md)'s exact A/E split
**Code**: [`temporal_epistemics.py`](temporal_epistemics.py) · **Files**: [FILES.md](FILES.md)
**Status**: both arms complete (`ntp_aux` and `cr_base`), full instrument × readout sweep.
CL deliberately unrun.

## One-line arc

The pre-registered crux was whether subtracting the temporal FM's forecast **concentrates** oracle
belief revision — higher decode from `r` than from `Δ` at small probe capacity. It does not. `Δ`
decodes `B` at least as well as `r` at **every** readout capacity, every abstraction level, every
instrument capacity, and on **both** substrates; the subtraction is mildly lossy. Because the
experiment also carries the **exact ε₁ = 0 innovation** as a source, this is not a statement about our
forecaster being weak: on `cr_base`, where the learned forecast reaches `cos = 0.989` with the exact
conditional mean, `r` still loses to `Δ`, and the ideal innovation only *ties* it.

> **On this substrate the temporal FM looks epistemically inert as a signal-former.** The revision
> content is in the raw state update; forming a residual against a forecast does not make it more
> accessible. That leaves *timing* — materialising `p⁻` at the right moment for in-loop consumption —
> as the FM's remaining distinct contribution, which is a claim about control topology rather than
> about signal composition, and is not tested here.

## Why this exists

The parent settled the **privacy** question, and the way it settled it is what makes this experiment
the right next one. Shifting the FM's conditioning gap from six blocks to one token gives the residual
a genuinely charged content, and the composite can report it — but the *privileged* part is not the
charged part: self-report and a capacity-matched observer track the oracle's belief revision `B_t`
**equally well** (partial `R²` 0.007 each), so their difference carries 0.0000 of it. Privilege
attaches to the manner of updating; what public evidence taught you is public.

That inverts the value of publicity for the question this program actually cares about:

> **Does the temporal FM's signal carry genuine epistemic content, and does the FM do any epistemic
> *work* in producing it?**

No privileged access is needed to measure a public channel — and this harness reads the temporal
channel at 0.899 against a validated 0.905 ceiling, where [`rule_family`](../../../conditional_revision/rule_family/README.md)'s
Gate 1 belief probe died at 11–17% of its ceiling. So: **repoint the battery from privacy to
composition.** Drop the report channel, the observer ladder, steering, the confabulator and CL. Keep
the frozen `M`, the report positions, the instrument sweep, the junk-residual guards and the oracle.

## The object

```
h6[t+1]  =  h6[t]  +  p⁻  +  r          p⁻ = FM_T(h6[≤t]),   r = Δ − p⁻
                       ^ forecast, available   ^ what the token added
                         BEFORE x_{t+1} arrives
```

A **source × target decode matrix at matched readout capacity**.

### Sources

| source | object | swept with the instrument? |
|---|---|---|
| `r` | the residual `Δ − FM_T(h≤t)` | yes |
| `delta` | the raw update `h6[t+1] − h6[t]` | — |
| `h_next` | the raw state `h6[t+1]` | — |
| `p_minus` | the forecast alone | yes |
| `rev_pair` | the explicit two-forecast revision `p⁺ − p⁻` | yes |
| `r_mart`, `p_mart` | the **exact** innovation `h6[t+1] − E_{a∼p_M}[h6[t+1] \| x_{t+1}=a]` and its forecast | — |
| `pos` | one-hot position — **the floor** | — |
| `r_shuffled` | `r` permuted within position — the content-destroying null | — |
| `\|r\|`, `\|delta\|`, `\|rev_pair\|`, `\|r_mart\|` | the scalar norms — **the standing entropy negative control** | — |

`rev_pair` is [SPEC.md](../../../conditional_revision/SPEC.md) Gate C's matched pair, implemented as
one class (`SlotFM`) so the match is structural: identical architecture, capacity, init seed,
optimiser, step budget, and — because the two are stepped inside one loop on one batch — identical
data in identical order. **The only difference in the entire procedure is what sits in the token
slot**: `FM₂` gets `x_{t+1}`, `FM₁` a learned `MASK`. That is the `arity_torque` idiom.

`r_mart` is **beyond the pre-registration**, and it is what makes the crux readable. It is the
residual a forecaster with *zero compression error* would leave, computed by `v`-way counterfactual
substitution rather than learned, so `r = r_mart + (p⁻_mart − p⁻_learned)` is an identity and the
`ε₁` term of idea doc §6 is **measured rather than bounded**. It supplies the ceiling the
concentration question otherwise lacks.

### Targets

| target | object |
|---|---|
| `B_pxs[D]` | **the primary.** Graded oracle belief revision, **position × exact surprisal** partialled out nonparametrically (`B` minus its mean within an (atom-aware surprisal bin × position) stratum, means from train rows only) |
| `B_bp[D]` | the surprisal-only residualisation, kept as the secondary |
| `AE_pxs[D]`, `AE_raw` | [`aleatoric_fraction`](../../../conditional_revision/aleatoric_fraction/README.md)'s exact per-position `A/(A+E)` of the state update, by the law of total variance over the `v` arriving tokens with BP weights — residualised the same way, plus one level left raw so the absolute scale stays visible |
| `bp_sur`, `H_tot` | exact surprisal and exact predictive entropy, raw — what the magnitude sources are supposed to load on |
| syn/dis | Gate B's constructed contrast, scored on the `B` probes' own output under position × exact-surprisal strata |

**Both controls are mandatory, for different reasons.** Surprisal, because it is the incumbent
explanation of everything on this axis. Position, because on the RHM constituent boundaries sit at
fixed offsets, so a target residualised on surprisal *alone* is still largely a **calendar** — and the
decode matrix would then rank sources by how well they encode position, which is not the question.
This is not hypothetical: in the wiring smoke, one-hot `pos` out-decoded *every* content source on a
surprisal-only `B`. Under position × surprisal the floor goes to ~0 by construction. The gap between
`B_bp` and `B_pxs` is kept as its own readout — how much of a source's apparent revision decode is the
calendar.

Training the probe on the residualised target rather than on raw `B` is load-bearing for the same
family of reasons: `R²(B ~ bp)` is 0.07–0.22 on this substrate, so a *small* probe trained on raw `B`
spends its capacity on the surprisal part — and the whole crux is a statement about small probes.

### The new sweep axis: readout capacity

linear (exact ridge, λ chosen on a held-in split) / MLP-16 / MLP-64 / MLP-256, identical for every
source, every source z-scored on train rows. Without standardisation the sources' differing norms
(`‖h_next‖ ≫ ‖Δ‖ ≈ ‖r‖`) masquerade as capacity; without a closed-form solution at the linear rung,
optimiser noise at the small end is exactly the confound that would fake a concentration curve.

## The pre-registered crux: `r` vs `delta`

The FM's epistemic claim *as an organ* is **concentration**: subtracting the forecast should cancel
the predictable carried-state part and make `B`-content more accessible from `r` than from `Δ` — a
higher decode at **small** probe capacity, converging as the probe grows.

| outcome | reading |
|---|---|
| `r` dominates `Δ` at small capacity **and** is FM-capacity-invariant | the temporal FM is an **SNR device** for revision; downstream uses should be fed the residual |
| `r ≈ Δ` everywhere | the FM is **epistemically inert as a signal-former**; its remaining defensible role — materialising `p⁻` at the right *time* for in-loop consumption — is a claim about control topology, to be tested in sparse-feedback settings and not by more measurement |

The prior from the parent run: a report head trained for cluster identity, not for `B`, already
carried `B` at ~0.007 partial `R²`; a `B`-trained head bounds that from below. The language sibling's
`r_dir ≈ h_after_dir` says the FM merely inherited there. Both ends of the live range are
decision-relevant — they determine what signal the sparse-feedback ("density-dial") experiments
should consume.

## Gate C, folded in and run for the first time

Pre-registered in [SPEC.md](../../../conditional_revision/SPEC.md) on 2026-08-07 and never run.

1. **Capacity invariance / the ε-control.** The `B_pxs` decode from `r` must be flat across the
   four-point instrument sweep. *"If it moves with capacity we are measuring `ε₂ − ε₁`, not
   revision."* Here it is both checked directly and read off exactly, because `r_mart` is the
   `ε₁ = 0` arm.
2. **The martingale calibration.** Idea doc §3: beliefs are a martingale, so the Bayes-optimal `p⁻`
   is the identity in belief space. The exact activation-space analogue is
   `p⁻_mart = E_{a∼p_M}[h6[t+1] | a]`, under which `E[r_mart | x≤t] = 0` **exactly** when `x_{t+1}` is
   drawn from `p_M`. So the drift statistic `‖mean_i r‖ / √(E‖r‖²/n)` reads 1.0 on self-sampled
   continuations *by construction* — which validates the estimator — while on corpus text it reads
   the model's calibration error projected onto the state-update map. The learned `r` gets the same
   2×2 with the cells reversed: corpus ≈ 1 by training, self-sampled informative.
3. **The compression term, reported not hidden.** `cos(p⁻_learned, p⁻_mart)` and the relative `ε₁`,
   per instrument capacity.

## Guards, all inherited and mandatory

The junk-residual trap (`ens_cos` + hierarchy η² at every instrument capacity), matched heads,
permute-inputs-not-labels (`r_shuffled`), the position floor (`pos` — on the RHM everything is
position-coupled, and a source that does not beat one-hot position has decoded the calendar), the
substitution-vs-plain-forward identity check on the counterfactual pass, and the oracle's own
`E[B_D] = H_tot − H_irr` self-check. **Axis-specific reading, unchanged from the parent**: a *high*
`ens_cos` is expected here and is not by itself evidence of a computational gap, because the aleatoric
component is input-determined by construction — every FM misses `x_{t+1}` identically.

## Harness validation — this is a controlled fork

| | reference | here |
|---|---|---|
| `ntp_aux` val | parent's published **1.5447** | **1.5447** |
| `cr_base` val | parent's published **1.5675** | **1.5675** |
| oracle `R²(B ~ exact surprisal)` d4/d3/d2 | `conditional_revision` 0.071 / 0.115 / 0.230 | **0.072 / 0.112 / 0.223** |
| oracle mean `B` / frac `B ≡ 0` at d2 | 0.361 / 0.383 | 0.367 / 0.375 |
| oracle identity self-check | — | **PASSED**, rel. err 3.7e-03 – 6.3e-03 |
| substitution vs plain forward | `aleatoric_fraction` 2.1e-04 | 5.2e-05 (`ntp_aux`) / 1.7e-04 (`cr_base`) |

The frozen `M` is literally the parent's — same checkpoint path, same 20K-step wake recipe — so this
is a continuation of that experiment rather than a neighbouring one.

**The regime is the same non-junk one the parent certified**: temporal `ens_cos` 0.940–0.946
(`ntp_aux`) and 0.933–0.937 (`cr_base`), forecast cosine 0.77–0.88, hierarchy η² concentrated at d1
(0.048–0.070) exactly as the parent found. Reading unchanged from the parent: a *high* `ens_cos` is
expected on this axis and rules out FM-idiosyncratic noise without ruling in a computational gap.

## Results — the decode matrix

`B_pxs` (oracle belief revision, position × exact surprisal partialled out), held-out R², default
instrument `h16m1`. Every column is the same probe recipe on a standardised source.

**`ntp_aux`, d2** (the level with the strongest signal):

| source | linear | MLP-16 | MLP-64 | MLP-256 |
|---|---|---|---|---|
| `r_mart` — exact, ε₁ = 0 | 0.107 | 0.435 | **0.475** | **0.485** |
| `delta` | 0.138 | 0.393 | 0.449 | 0.446 |
| `rev_pair` = p⁺ − p⁻ | 0.125 | 0.355 | 0.394 | 0.413 |
| `h_next` | 0.138 | 0.354 | 0.382 | 0.373 |
| **`r`** | **0.091** | **0.344** | **0.362** | **0.336** |
| `p_minus` | 0.099 | 0.180 | 0.160 | 0.119 |
| `pos` — floor | −0.000 | −0.000 | −0.000 | −0.000 |
| `r_shuffled` — null | −0.000 | −0.049 | −0.186 | −0.453 |
| `\|r\|` / `\|delta\|` / `\|r_mart\|` | 0.006 / 0.000 / 0.007 | — | 0.008 / 0.000 / 0.015 | — |

**`cr_base`, d2** — the frozen base Gates A/B, `aleatoric_fraction` and `tracking` all ran on:

| source | linear | MLP-16 | MLP-64 | MLP-256 |
|---|---|---|---|---|
| `r_mart` — exact, ε₁ = 0 | 0.111 | 0.379 | **0.413** | **0.430** |
| `delta` | 0.130 | 0.352 | 0.411 | 0.424 |
| `h_next` | 0.132 | 0.345 | 0.387 | 0.395 |
| **`r`** | 0.082 | 0.315 | **0.366** | **0.385** |
| `rev_pair` | 0.107 | 0.318 | 0.358 | 0.375 |
| `p_minus` | 0.105 | 0.162 | 0.170 | 0.168 |

The ordering is the same under every scoring we ran: partial `R²` given exact surprisal (`delta`
0.457 vs `r` 0.379 at MLP-64, d2) and Gate B's syn/dis contrast under position × exact-surprisal
strata (`delta` **0.943** > `rev_pair` 0.922 > `h_next` 0.922 > `r` **0.915**). The level gradient is
steep and consistent — at MLP-64, `delta` reads 0.449 / 0.277 / 0.115 at d2 / d3 / d4.

**The crux, stated carefully.** The pre-registered concentration prediction is not observed; the
observed direction is the opposite of it, though the margin (`Δ` − `r` ≈ 0.04–0.11 in R²) is modest
next to the absolute decode level. Two features make it hard to attribute to instrument weakness:

- **It survives the ε₁ = 0 limit.** `r_mart` is the residual a forecaster with *zero* compression
  error would leave, computed by counterfactual substitution rather than learned. It beats `Δ` by
  ~0.03 on `ntp_aux` and ties it on `cr_base` — so in the ideal limit concentration is somewhere
  between nil and small, and it does not reproduce across substrates.
- **It survives a nearly-ideal learned forecaster.** On `cr_base` the learned `p⁻` reaches
  `cos = 0.966 → 0.989` with `p⁻_mart` and rel ε₁ falls to **0.131**, and `r` still loses to `Δ`.

## Gate C, run for the first time

Pre-registered in [SPEC.md](../../../conditional_revision/SPEC.md) on 2026-08-07 and never run until now.

**C.1 — capacity invariance (the ε-control).** *"If it moves with capacity we are measuring ε₂ − ε₁,
not revision."* `B_pxs` at d2 from `r`, across a 24× instrument range:

| instrument | 67K (1.1% of M) | 264K (4.2%) | 790K (12.5%) | 1578K (24.9%) |
|---|---|---|---|---|
| `ntp_aux`, linear | 0.099 | 0.091 | 0.092 | 0.090 |
| `ntp_aux`, MLP-64 | 0.354 | 0.362 | 0.381 | 0.386 |
| `cr_base`, linear | 0.099 | 0.082 | 0.081 | 0.086 |
| `cr_base`, MLP-64 | 0.364 | 0.366 | 0.369 | 0.377 |

Flat at the linear rung on both arms (±0.009 over 24×). The mild rise at MLP-64 on `ntp_aux`
(+0.032) is monotone in `cos(r, r_mart)` (0.741 → 0.833) and is much smaller on `cr_base` (+0.013,
`cos` 0.783 → 0.837), whose ε₁ range is half as wide. So the drift tracks the compression term
shrinking toward the exact innovation rather than indicating an unstable estimate. **This also
supplies a candidate account for the parent's explicitly-unexplained residue** — its temporal
advantage rose with instrument capacity (+0.014 → +0.027) while the depth arm's stayed flat.

**C.2 — the martingale calibration.** Drift ratio `‖mean_i r‖ / √(E‖r‖²/n)`; **1.0 is the null**.

| residual | self-sampled | corpus |
|---|---|---|
| exact `r_mart` | **0.99** | **1.09** |
| learned `r`, `ntp_aux` (67K → 1578K) | 3.37 → 2.51 | 3.96 → 2.74 |
| learned `r`, `cr_base` (67K → 1578K) | 1.68 → 1.81 | 1.99 → 2.01 |

The exact null lands at 0.99 by construction, which validates the estimator rather than testing
anything. Against it, corpus reads **1.09** — a small but measurable systematic mismatch between the
model's predictive distribution and the world's, projected onto the state-update map. The learned
residual sits at 2.0–4.0 on *both* distributions and improves monotonically with FM capacity on
`ntp_aux`, so most of its apparent "the world is systematically surprising me" is ε₁ bias, not world.

**C.3 — the compression term, reported not hidden.**

| | `cos(p⁻_learned, p⁻_mart)` | rel ε₁ | `‖r‖/‖r_mart‖` | `cos(r, r_mart)` |
|---|---|---|---|---|
| `ntp_aux`, 67K → 1578K | 0.870 → 0.956 | 0.475 → 0.272 | 1.575 → 1.285 | 0.741 → 0.833 |
| `cr_base`, 67K → 1578K | 0.966 → 0.989 | 0.228 → 0.131 | 1.239 → 1.102 | 0.783 → 0.837 |

The plain-NTP base's next-state map is markedly easier to forecast than the aux-latent model's.

## The controls, which are results in their own right

- **The position floor was necessary and is now quantified.** `pos` reads exactly **0.000** on
  `B_pxs` (by construction) but **+0.132 / +0.172 / +0.170** at d4/d3/d2 on the surprisal-only
  `B_bp` — so 13–17% of surprisal-residualised `B` on this substrate is *pure calendar*. The wiring
  smoke, run with surprisal-only residualisation, had one-hot position out-decoding every content
  source. Both arms give byte-identical `pos` numbers, as they must: model-independent source against
  model-independent target.
- **The magnitude negative control behaves exactly as the design doc pre-registered** — `|r|`,
  `|Δ|`, `|rev_pair|`, `|r_mart|` all read 0.000–0.015 on `B_pxs` while `|r_mart|` loads exact
  surprisal at R² 0.52. Worth recording because the *parent* falsified this prediction (`TEMP-MAG`
  showed its largest advantage). The two are consistent if the parent's contradiction came from its
  report-head readout rather than from magnitude carrying revision; that is a plausible
  reconciliation, not a demonstrated one — nothing here re-runs the parent's readout.
- **The shuffled null is clean at the linear rung** (−0.000) and goes increasingly negative with
  probe capacity (−0.453 at MLP-256 on `ntp_aux`), which is the expected overfitting signature of a
  content-free input, not a finding.
- **Headroom.** After position × surprisal residualisation, 0.270 / 0.375 / 0.520 of `Var(B)` remains
  at d2 / d3 / d4 (0.591 / 0.757 / 0.840 under surprisal alone), from 1778 strata at ~50 train rows
  each. So no null here is a "nothing left to decode" artifact — and note there is *more* headroom at
  d4, where every source decodes worst.

## An observation on the density/sparsity framing

The parent recorded the program-level lesson that introspection should pay where feedback is **sparse
and evaluative**. **This experiment does not test that** — both arms sit at the dense-public end by
construction (NTP on the same corpus, frozen models, measurement only), and no density dial was
varied. What it does deliver for that program is the decision it was run to make: *what signal should
the sparse-feedback experiments consume?* On this evidence, `Δ` or `h[t+1]`, not `r` — with the
caveat that a gate reading `‖r‖` would be reading ε₁ bias at 2–4× the exact innovation's drift.

The closest thing here to a density axis is abstraction depth, and it is worth recording as
suggestive-at-most:

| level | epistemic fraction `1 − A/(A+E)` | best `B_pxs` (any source) | headroom |
|---|---|---|---|
| d2 | 0.273 | 0.485 | 0.270 |
| d3 | 0.151 | 0.304 | 0.375 |
| d4 | 0.084 | 0.135 | 0.520 |

Where a token tells you least about a latent, the model's state also carries least about how that
latent moved — and the headroom column says this is not a ceiling artifact. Two reasons to hold that
loosely: sparse *evidence about a latent* is a different axis from sparse *evaluative feedback*, and
this regime is documented as learning only 1–2 compositional levels, so "d4 structure is weakly
represented" is at least as good an explanation.

What this run does hand the density program is an **instrument**: the drift ratio is dimensionless,
exactly nulled at 1.0, needs no probe and no trained oracle-side readout, costs one counterfactual
pass, and transfers to any causal model. In the dense-public regime it reads **1.09**.

*(A/E here is the mean of per-position `A/(A+E)` ratios, which is a different estimator from
[`aleatoric_fraction`](../../../conditional_revision/aleatoric_fraction/README.md)'s pooled
`ΣA/ΣTotal` — 0.727 vs its published 0.829 at d2 on the same frozen base. Both are valid; they are
not directly comparable, and the per-position form is the one a per-position decode target requires.)*

## What this establishes, and what it does not

**Establishes** (one regime, two substrates, 4 readout × 4 instrument capacities):

- `Δ` decodes oracle belief revision at least as well as `r` everywhere we looked; the pre-registered
  concentration effect is not present, and the residual is mildly lossy relative to the raw update.
- This is not attributable to forecaster weakness: it holds at rel ε₁ = 0.131 and against an exact
  ε₁ = 0 innovation that itself only ties `Δ` on one of the two substrates.
- Gate C.1 passes — the `B`-decode from `r` is flat in instrument capacity at the linear rung on both
  arms, so this is a measurement of revision rather than of `ε₂ − ε₁`.
- The exact martingale null is constructible and lands at 0.99; corpus drift is 1.09.
- 13–17% of surprisal-residualised `B` is position-determined on this substrate.

**Does not establish:**

- Anything about **timing**, which is the FM's remaining distinct claim and is a control-topology
  question — not measurable on a frozen model with trained probes.
- Anything about **use**. Every readout is a trained probe.
- Anything about the **closed loop**. CL is unrun, on the parent's reasoning that loop closure changes
  how well `M` encodes its residual rather than what the residual is composed of. That reasoning is
  weaker here than it was for the privacy question and should not be treated as settled.
- That `rev_pair` is or is not the better object. It beats `r` on `ntp_aux` (0.413 vs 0.336 at
  MLP-256) and does not on `cr_base` (0.375 vs 0.385). Unresolved.
- Seed robustness, transfer off this regime, or anything at scale — the doc's own scope note that the
  update map's learnability may not survive frontier scale applies unchanged.
- That the magnitude-vs-direction question is closed. See the reconciliation caveat above.

## What this will not establish, stated up front

- **Privacy** — settled by the parent, and deliberately not re-measured.
- **Use.** Every readout here is a trained probe. This is a claim about what is *in* the signal and
  about whether subtracting the forecast helps get it out; it is not a claim that anything consumes
  it. The consumability arm (gating on the wire) remains the separate, later experiment.
- **The closed loop.** CL is dropped, on the parent's reasoning: loop closure changes how well `M`
  encodes its residual, not what the residual is composed of.
- Seed robustness, or transfer off this regime.

## Reproduction

```bash
cd experiments/          # NOT the repo root -- see gotchas

# wiring smoke (minutes; numbers meaningless by construction)
modal run -m rhm.confabulation.temporal.epistemics.temporal_epistemics::temporal_epistemics --smoke

# the pre-registered primary (resumes the frozen M shared with the parent and the depth battery)
modal run --detach -m rhm.confabulation.temporal.epistemics.temporal_epistemics::temporal_epistemics \
    --conditions "ntp_aux" --tag ol

# the Gates A/B substrate, where the charge was originally measured
modal run --detach -m rhm.confabulation.temporal.epistemics.temporal_epistemics::temporal_epistemics \
    --conditions "cr_base" --tag crbase
```

Both arms are complete; re-running either reuses the cached oracle and the shared wake checkpoint,
so a repeat run trains only the instruments. Results land on the `rhm-scaling-data` volume
(**`chromatic` workspace**) under
`/data/rhm_confabulation/v16_s2_L6_m4_distinct/temporal/epistemics/`: `{ol,crbase}_results.json` plus
the cached `oracle_posteriors_*.npz`. The oracle itself is read from — and written to — the parent's
directory, so a regime the parent has already run skips the BP pass entirely. Wake checkpoints are
shared with the parent and the depth battery at `.../ckpt/`.

## Gotchas worth not rediscovering

- **`model(x)` returns two values; `model(x, return_intermediates=True)` returns three.** Sampling
  loops that unpack three from the former fail only after the oracle and counterfactual passes are
  already done.
- **The temporal residual does not exist at the position it is indexed by** (parent's gotcha, still
  binding): `r[t]` needs `h6[t+1]`, so every column here is indexed to `p = t+1 ∈ [1, T-2]`.
- **Standardise every source before comparing decodes.** Otherwise "concentration" is a conditioning
  artifact.
- **Do not train the small probes on raw `B`.** Surprisal eats the capacity and the concentration
  curve becomes a measurement of `R²(B ~ bp)`.
- **Positions with a single legal arriving token have zero total variance** and must drop out of the
  A/E target, numerator and denominator together — the same convention `aleatoric_fraction` uses
  (~18% of cells there).
- **A high `ens_cos` means less on this axis than on the depth axis.** The aleatoric component is
  input-determined, so independent FMs agree on it by construction.
- **Launch Modal from `experiments/`, not the repo root** — `modal run -m rhm...` from the root fails
  with `ModuleNotFoundError: No module named 'rhm'`.
