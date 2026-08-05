# Ballistic depth — what actually extends a learned operator's composition horizon

**Up**: [../README.md](../README.md) (one_layer_deeper) · **Files**: [FILES.md](FILES.md)
**Substrate**: repeated modular squaring, `y = x^(2^T) mod N`, from [tilde-research/one-layer-deeper](https://github.com/tilde-research/one-layer-deeper)
**Status**: complete at `N=893`; scale cut in progress. 5 arms + 3 ablations + a 4-point weight
sweep + an instrument calibration, 3 seeds each. **Date**: 2026-08-02.
**Extended 2026-08-03** by §10–§11 and two child nodes. At `N=9853` (11.6× the state set) the
cycle term at §7's optimum **collapses** (closure 0.998, ID 0.001 — a documented blind spot in the
closure instrument), repairable by schedule but not restoring the advantage; group structure in
the representation falls to the permutation null
([`rule_structure/`](rule_structure/README.md)); and the horizon numbers at that scale are
**not converged** — 2× the budget moves T=7 from 0.176 to 0.996 with ID saturated throughout
([`horizon_convergence/`](horizon_convergence/README.md)), so `N=9853` horizons are lower bounds.
`N=893` *is* converged, which leaves §1–§9 intact.
**Extended 2026-08-02** by a micro-cut (§6–§9): a closure-vs-horizon regression over every run,
a 7-point cycle-weight sweep that **moves the headline horizon from 35 to 51**, a cold-start probe
showing the operator is sound at every depth tested, and a test-time re-projection cut in which
**the grounded model reaches 1.000 at T=60 with no retraining at all**. §1's T≈35 is a *w=1.0*
number, not a ceiling — see §7.

---

## One-liner

On a task that is *purely* ballistic — the model commits at step 0 and never observes an
intermediate state — a tied recurrent operator trained on terminal cross-entropy alone has a
composition horizon of **T ≈ 13** (trained on T ≤ 6). Adding one **label-free** constraint —
*the state you rolled into must be a state your own encoder could have produced* — moves that
horizon to **T ≈ 35**, a 2.8× extension, with perfect exact-match out to T = 20. The mechanism
is **manifold closure**, not error suppression: the base model's rolled state is nearly
orthogonal to the encoder's representation of the same residue (cos **0.19**) from step 1 while
decoding it perfectly, and the constraint takes that to **0.99**. The cut's *predicted*
mechanism — that the rollout fails because the operator amplifies its own error, and that
discretising the state is therefore mandatory — is **falsified five independent ways**, and the
hard-quantisation arm that was supposed to demonstrate it failed to learn the task at all
(twice) while demonstrably achieving the error correction it was built for.

## Why this substrate

The task is `x_0 = x`, `x_t = x_{t-1}^2 mod N`, predict `x_T`, with `N`'s factorisation never
supplied. Two properties make it an unusually clean forward-model instrument:

1. **It is maximally ballistic.** Every controller in [`mjc/`](../../mjc/README.md) needed
   commitment installed as a knob (`replan_every`), because reactive control re-grounds to the
   true state each step and so is a *near-blind grader* of forward-model quality — 4b measures
   the transmission slope at **+0.35** reactive vs **+1.07** ballistic. Here there is no reactive
   mode available: exact-match at held-out depth is a fully sighted grader by construction.
2. **The ground truth is exact and free**, so latent veridicality is measurable at every rollout
   step against a known `x_t`, and the per-step error amplification of the *true* dynamics is
   known analytically (see below). Our own substrates always had to manufacture the FM-quality
   axis and grade it with proxies.

**The DGP knob that matters.** `N = 893 = 19 × 47`. Chosen so that (a) the first depth at which
`x^(2^T)` repeats in `T` is **67** — far above anything evaluated, so periodicity is never an
alternative to serial rollout; (b) squaring is a **bijection on the 207-element quadratic-residue
subgroup** for all `t ≥ 1`, so perfect attractor structure is *available in principle* — which
matters, because if the success mode were unreachable in budget the design could not distinguish
"the intervention doesn't help" from "the task is impossible"; and (c) 828 units keeps the
one-step map learnable in minutes. The generator asserts `depth_first_repeat > max_depth` at
startup and cross-checks every trajectory column against the independent trapdoor label path
`pow(x, 2^T mod φ(N), N)`.

**The analytic prediction this was built to test.** Writing `x = g^a`, squaring is `a ↦ 2a` — the
doubling map, Lyapunov exponent `ln 2`, exactly one bit of state precision destroyed per step. In
the raw residue coordinate it is worse still (local derivative `2x`, i.e. O(N)). So *for a
faithful scalar encoding* there is no contracting coordinate, and a continuous rollout must lose
one bit per step. The cut's hypothesis was that depth extrapolation therefore requires the
rollout to be error-**correcting** (re-attracted to a codebook each step), not merely accurate.

## Design

One encoder, one operator, one decoder, shared across arms; the arms differ in one variable each.

- **Encoder** — 2-layer bidirectional transformer, d=256, over the upstream prompt encoding
  (`BOS N <digits> X <digits> ANS`); reads out at the `ANS` position to give `h_0`.
- **Operator** — a single tied residual block `h ← h + MLP(LN(h))`, applied exactly `T` times.
- **Decoder** — `LN → Linear(d, 3×10)`, fixed-width decimal digits.
- **Controls on the architecture.** `T` is given to recurrent arms *only* as the loop count and
  is **absent from their prompt**, so folding depth into the initial state is not available and
  the rollout is the only route to depth. The non-recurrent control gets `T` in the prompt and
  6 **untied** blocks — the maximally favourable version of the depth-indexed solution.

**Held-out axis.** Train `T ∈ 1..6`, evaluate `T ∈ 1..60`. Base values `x` are ~90% seen in
training *on purpose*: x-generalisation is a separate question and leaving it free would confound
the depth readout. A 10% held-out-x slice is reported alongside, so both axes stay visible.

**The two interventions.**

| factor | what it does |
|---|---|
| **quantize** | straight-through VQ after every operator application — snap `h` to the nearest of K learned codes. The *hard* implementation of re-attraction. |
| **cycle** (label-free) | decode `h_t` → soft digits → re-encode through the same encoder → `h̃_t`; require `h̃_t ≈ h_t`. Forces the intermediate state into the encoder's range. **No labels.** |
| **re-entry** (reuses the terminal label) | roll `h̃_t` forward the remaining `T−t` steps and apply the *same* terminal CE. Trains the operator on cleanly-encoded states it would not otherwise visit. No new label information enters. |

Both consistency terms are gated behind a 30% warmup — see the gotcha below.

**Instruments.** `exact@T`; `verid@t` (the *terminal* decoder applied zero-shot to `h_t`, never
trained at intermediate `t`); `onmanifold@t` = cos(`h_t` rolled, `Enc(true x_t)`) — the direct
analogue of `_rollout_fidelity` in [`rhm/rhm_sculpt_twofm.py`](../../rhm/FILES.md); and
`amplify@t` = ‖δh_t‖/‖δh_0‖ under a perturbation sweep, measured along **both** a random
direction and an on-manifold one. **No arm ever trains on an intermediate residue** — the
trajectory is instrumentation only.

---

## 1. The main result — the horizon moves 13 → 35

![depth extrapolation to T=60](figures/deep60/fig_ballistic_depth.png)

Exact match, 3 seeds, trained on `T ≤ 6`:

| arm | T≤6 | T=7 | T=10 | T=13 | T=20 | T=30 | mean T>6 (to 20) |
|---|---|---|---|---|---|---|---|
| feedforward (untied stack) | 1.000 | 0.009 | 0.00 | 0.02 | 0.043 | — | 0.015 ±0.001 |
| base (recurrent, terminal CE only) | 1.000 | 1.000 | 0.92 | 0.53 | 0.026 | 0.004 | 0.503 ±0.059 |
| **+ cycle** | 1.000 | 1.000 | **1.00** | **1.00** | **0.983** | **0.708** | **0.997 ±0.004** |
| + quantize | 0.148 | 0.030 | 0.00 | 0.03 | 0.005 | — | 0.018 ±0.000 |
| + quantize + cycle + re-entry | 0.165 | 0.026 | 0.01 | 0.03 | 0.006 | — | 0.016 ±0.002 |

*Provenance: the `T ≤ 20` columns come from the T=20-range runs (`cut1` for every arm except
`+ cycle`, which is `cyc_only` — `cut1`'s consistency arm ran both terms); the `T=30` column comes
from the T=60-range runs (`deep60` / `w_re0`). Cross-run values differ by ~1–2 depths from GPU
nondeterminism (see caveats), e.g. `+ cycle` at T=20 reads 0.983 in `cyc_only` and 1.00 in
`deep60`.*

Composition horizon (first depth below 50% exact), evaluated to T=60:

| | seeds | median |
|---|---|---|
| base | 15, 13, 13 | **13** |
| + cycle | 35, 38, 37 *(re-run: 34, 35, 35)* | **≈35** |

**The footprint/operator split is about as clean as it can be shown.** The untied stack is
perfect at every trained depth and *at chance one step past it* — 1.000 → 0.009. It learned six
depth-indexed maps; there was never a step. The tied operator gets two free steps (perfect to
T=8) and then decays. This is
[`operators_not_footprints`](../../../beliefs/trees/operators_not_footprints.md) with a
scoreboard: the distribution-bound solution and the distribution-invariant one score identically
in-distribution and separate by ~100× one step outside it.

**The base horizon of 13 sits alongside our other substrates** — pusher 6–8, RHM ~6
([`full_loop`](../../rhm/directed_sculpting/full_loop/README.md) E5: open-loop success humped at
`c3` 0.092 against `c1` 0.045 and `c6` 0.043), arm 14–23. That four unrelated domains land in the
same band is suggestive rather than established, but this is the first of them where the per-step
amplification of the true dynamics is known analytically.

## 2. The mechanism is manifold closure, not error suppression

cos(`h_t` rolled, `Enc(true x_t)`):

| arm | t=1 | t=6 | t=10 | t=20 | t=30 | t=60 |
|---|---|---|---|---|---|---|
| base | **0.187** | 0.120 | 0.102 | 0.013 | −0.002 | −0.00 |
| + cycle | **0.997** | 0.997 | 0.994 | 0.974 | 0.869 | 0.37 |

**The base model's rolled state is nearly orthogonal to the encoder's representation of the same
residue — from step 1 — while decoding to that residue perfectly.** It is not iterating a closed
operator on the state set; it found a private trajectory whose points each happen to be
decodable. `F` works on the states `F` itself generates, for as far as training pushed it, and
past that it is off the edge of its own map. The cycle term pins `h_t ≈ Enc(x_t)`, every
intermediate state becomes a legitimate cold start, and the rollout keeps going. Closure decays
0.99 → 0.37 across T=6..60 and accuracy tracks it the whole way, so this is not a
near-the-boundary artifact.

This reproduces [`RHM_SCULPTING`](../../rhm/RHM_SCULPTING_README.md) Stage 5b's **rollability ≠
depth** dissociation on an unrelated substrate: base has perfect *decodability* at every trained
depth and near-zero *re-enterability*; the grounded term buys re-enterability without touching
decodability.

**An unpredicted second axis.** Held-out-x exact match at trained depths: base **0.001**, cycle
**0.323**, and cycle's held-out-x curve is nearly flat in depth. The base model memorised the
one-step map with essentially zero transfer. Either consistency term alone buys the ~0.32, and
they do not add — so the two terms buy *different* things (§4).

## 3. The predicted mechanism is falsified — five ways, and the instrument was calibrated

![re-entry sweep and amplification calibration](figures/sweep/fig_sweep.png)

Per-step amplification (geometric factor from t=8, ε=0.01):

| condition | random direction | on-manifold direction | ratio |
|---|---|---|---|
| base (terminal CE only) | 1.082 ±0.008 | 1.117 ±0.005 | 1.03× |
| cycle only | 1.055 ±0.007 | 1.089 ±0.009 | 1.03× |
| cycle + re-entry (w=1) | 1.080 ±0.028 | 1.089 ±0.030 | 1.01× |
| *analytic doubling-map bound* | — | **2.000** | — |

1. **Base and cycle amplify near-identically** (1.082 vs 1.055 random; 1.117 vs 1.089
   on-manifold) while their horizons differ 2.8×.
2. **re-entry-only has the *lowest* amplification of any arm** (1.028 random) and is the
   second-worst performer. If anything the correlation runs backwards.
3. **quant-only achieves per-step amplification of 0.889 — genuinely contracting, exactly the
   error correction it was built for — and scores 0.048.** Error correction, achieved and useless.
4. **The magnitude is off by five orders.** Over 20 steps base amplifies **9.9×** on-manifold
   where the analytic bound says 2²⁰ ≈ 10⁶. A network embedding 828 states in 256 dimensions
   squashes almost every direction for free; "there is no contracting coordinate" is true of the
   true dynamics in a scalar coordinate and false of a learned high-dimensional embedding.
5. **The accounting doesn't close.** At t=20, base carries 9.9× accumulated error at 1.7% exact;
   cycle carries 6.0× at 99.3%. A 1.65× difference in error against a 58× difference in accuracy.
   And the curves *cross* — by t=40 cycle amplifies **more** (45.7 vs 28.9) while still scoring
   far higher.

**The instrument was calibrated before the null was read**, per
[`mjc/expansion/`](../../mjc/expansion/README.md)'s calibrate → measure → calibrate discipline.
The worry was that a random direction in d=256 is almost entirely off-manifold and therefore
cheap to squash. Measured along the direction of an actually-confusable state (toward
`Enc(x')` for a different residue `x'`), amplification is **3% higher per step** and the arm
ordering is unchanged. The instrument was not the problem; the hypothesis was.

**What survives.** Amplification *is* above 1 in every continuous arm, so the qualitative claim
(the rollout expands rather than contracts) holds. It is simply not the binding constraint, and
it is ~20× gentler than the analytic bound.

## 4. The two consistency terms are different mechanisms, and they do not compose

3 seeds, cycle weight fixed at 1.0, evaluated to T=60:

| re-entry weight | horizon (T @ 50%) | T=20 | T=30 | closure @ t=30 |
|---|---|---|---|---|
| *base (no cycle)* | 13.0 ±0.0 | 0.017 ±0.003 | 0.004 ±0.003 | −0.002 |
| **0.0** | 34.7 ±0.6 | 0.993 ±0.002 | 0.708 ±0.019 | 0.869 |
| **0.1** | 38.0 ±2.6 | 0.995 ±0.005 | 0.805 ±0.055 | 0.891 |
| **0.3** | 31.0 ±1.0 | 0.978 ±0.005 | 0.541 ±0.050 | 0.771 |
| **1.0** | 22.3 ±3.1 | 0.668 ±0.237 | 0.071 ±0.069 | 0.527 |

**The label-free term is doing the work, and the label-reusing term is dose-dependently harmful
above w ≈ 0.1** — monotone in both horizon and closure. At w=0.1 it is within run-to-run noise of
w=0 (see the reproducibility gotcha below); at 0.3 and 1.0 it clearly costs. Closure falls
0.891 → 0.771 → 0.527 in lockstep with the horizon, so it degrades the same variable §2
identifies as the mechanism.

Ablated separately at w=1.0 (T=20 eval): **re-entry-only reaches 0.797 OOD against base's 0.503
while leaving closure at 0.233** (base 0.187). So it is a real but *separate* and weaker
mechanism — the covariate-shift/DAgger fix, training the operator on cleanly-encoded states —
and it buys the same 0.32 held-out-x that cycle does, without buying re-enterability.

A plausible reading of the interference, **not tested here**: re-entry applies a
label-referencing gradient at a state the model itself decoded, so when that decode is wrong it
teaches the operator to map a *wrong* state to the right answer — direct pressure against the
operator being a clean function of its input, which is what cycle installs.

**This inverts the cut's framing.** The pair was designed as the dense/evaluative structure of
[`heterogeneous_graders.md`](../../../ideas/heterogeneous_graders.md) §4, with complementarity
predicted. What the sweep shows is the purely structural, label-free constraint carrying
essentially all of it, and densifying the *evaluative* signal over counterfactual states costing
rather than adding. Whether that generalises past this substrate is open; §4's impossibility
argument is about signals available to a learner, and nothing here tests it directly.

## 5. Hard quantisation failed twice, and the way it failed is informative

| codebook K | ID (T≤6) | OOD | codes occupied @ t=20 | per-step amp |
|---|---|---|---|---|
| 4096 | 0.148 ±0.016 | 0.018 | 612 | 0.749 |
| 256 | 0.038–0.048 | 0.009 | 224 | 0.889 |
| *(true reachable set)* | | | **207** | |

The first attempt over-provisioned — 612 codes for 207 states means each residue was smeared
across ~3 codes and "snap to nearest" flipped between them instead of correcting. Shrinking to
K=256 fixed exactly that (**224 codes occupied against a true 207**) and the arm got *worse*. So
codebook size was not the binding problem; hard, non-differentiable snapping inside a recurrent
rollout is not learnable in this setup. **This is a negative on the implementation, not on the
idea** — and the informative part is that the mechanism worked while the arm did not: quantise
achieves genuine per-step contraction (0.889 < 1) and 4.8% accuracy.

The reading we take from it: **the soft constraint achieves the re-attraction that discretisation
was supposed to provide, and stays learnable, which the hard version does not.**

---

# Micro-cut (2026-08-02): is closure a dial, and is the operator actually broken?

§2 reads the mechanism off a two-arm contrast, and §1 reports a horizon at a cycle weight that
was never swept. Three follow-ups, run on **L4** (§1–§5 are A10G — see caveats):

## 6. Closure predicts the horizon across 72 runs — ordinally, not on a calibrated scale

![closure vs horizon](figures/closure_horizon/fig_closure_horizon.png)

`closure_horizon.py` is pure re-analysis: every `results/<tag>/results_seed<N>.json` already
carries `on_manifold_cos` and `exact_seen_x` at every depth, so closure-vs-outcome is a scatter
over **72 recurrent runs across 24 configs** with no GPU at all. (The `feedforward` arm is
excluded throughout — non-recurrent, so it has no rolled state to measure.)

**The pooled correlation is not the evidence, and should not be quoted as it.** Before the cycle
sweep, closure was **bimodal**: every run sat at ≈0.10 or ≈0.99 with *zero* runs between 0.14 and
0.95. A line across that gap gives R²=0.93 essentially for free — it is §2's two dots with 48
replicates, not an independent measurement. The figure marks the empty region rather than
reporting the fit through it.

What survives, over the full set including the sweep that filled the gap:

| readout | n | R² | Spearman rho |
|---|---|---|---|
| closure@10 → exact@20 | 72 | 0.80 | **+0.948** |
| closure@20 → horizon | 45 | 0.73 | **+0.935** |
| closure@**6** → horizon *(measured at a trained depth)* | 45 | 0.48 | **+0.901** |
| *control:* ID accuracy → exact@20 | 72 | 0.21 | +0.632 |

Two things worth separating. **Closure is not a proxy for "did this arm learn the task"**: all 60
ID-competent runs sit at in-distribution accuracy of exactly 1.000, and closure still orders their
OOD spread at rho +0.93. And **the ordering is readable in-distribution** — closure at t=6, the
last trained depth, ranks configurations by their eventual horizon at rho +0.90, which is what
would make it usable as an early-warning instrument rather than a post-hoc description.

**But it does not transfer on a calibrated scale.** Fit the line on the re-entry sweep alone and
it mispredicts the cycle sweep badly — at closure 0.93 it predicts exact@20 = 0.857 against an
actual 0.391 (bias −0.47); at 0.81, 0.735 against 0.145 (−0.59). In the other direction
`re_only` sits **+0.26 above** it: closure 0.135, barely above base's 0.10, yet exact@20 0.306
against base's 0.027 and horizon 18–19 against 13. So the same closure value means different
things depending on which knob produced it, and there is at least one route to extra depth that
does not go through closure at all — which corroborates §4's *separate mechanism* reading from a
completely different direction. **Closure orders configurations; it does not price them.**

## 7. Sweeping the cycle weight — the horizon was never at its ceiling, and the knob turns over

The cycle weight was held at 1.0 for the entire original cut. Sweeping it with re-entry pinned at
0 (3 seeds each, eval to T=60, no seed censored):

| cycle w | closure@10 | closure@20 | ID | exact@20 | **horizon** |
|---|---|---|---|---|---|
| 0.01 | 0.494 | 0.119 | 1.000 | 0.021 | 13.7 ±0.5 |
| 0.05 | 0.810 | 0.390 | 1.000 | 0.145 | 16.3 ±0.9 |
| 0.2 | 0.930 | 0.617 | 1.000 | 0.391 | 19.7 ±2.1 |
| 1.0 *(§1's point)* | 0.992 | 0.972 | 1.000 | 0.992 | 36.0 ±2.9 |
| 3.0 | 0.999 | 0.994 | 1.000 | 1.000 | 46.3 ±2.6 |
| **10.0** | 0.999 | **0.997** | 1.000 | 1.000 | **51.0 ±3.6** |
| 30.0 | 0.998 | 0.981 | 1.000 | 0.931 | 28.0 ±0.8 |

**The headline extension is 13 → 51, a 3.9× horizon, not the 2.8× reported in §1** — which was an
arbitrary weight, not a ceiling. At w=10 exact-match is 1.000 at T=20 and 0.977 at T=30.

**This is also what de-confounds §6.** Inside the closed cluster every *earlier* run had cycle
weight pinned at 1.0, so closure there was generated entirely by the re-entry weight and the two
were statistically indistinguishable as predictors (R² 0.847 vs 0.844 for the horizon). This
family moves closure with re-entry held fixed. And the turnover at w=30 does the rest of the work:
across the sweep the **weight** stops predicting the horizon (R²=0.02) while **closure@20 keeps
predicting it** (R²=0.69, rho +0.90), because at w=30 the knob went up while closure *and* horizon
came down together. Closure tracks the outcome through a reversal that the knob does not — which
is the strongest evidence here that it is a mediator rather than an incidental correlate.

The predicted failure mode for an over-weighted cycle term — *collapse to identity* — is **not**
what w=30 shows: in-distribution accuracy is still exactly 1.000 at every weight tested, including
30. Whatever the over-weighted term costs, it costs depth specifically, not the task.

## 8. The operator is sound at every depth; the whole failure is that it cannot reach its own inputs

![cold start](figures/coldstart/fig_coldstart.png)

`on_manifold_cos` is a *cosine* — a geometric proxy. The behavioural version: encode the **true**
intermediate residue `x_t` cold, as if it were a fresh problem, roll the remaining `T − t` steps,
score against `x_T`. `t = 0` is the ordinary ballistic rollout, so it re-derives `exact_seen_x[T]`
as a built-in self-check.

Base, T=60, 3 seeds:

| re-entry point t | 0 | 20 | 35 | 40 | 45 | 50 | 55 |
|---|---|---|---|---|---|---|---|
| exact match on `x₆₀` | **0.000** | 0.005 | 0.005 | 0.022 | 0.237 | 0.785 | **0.873** |

**The base operator computes correctly at depth 55.** Hand it `x₅₅` and it rolls the last five
steps at 0.873, while its own rollout to the same target scores 0.000. The same holds at every
depth probed — restart at t=25 for T=30 gives 0.871, at t=35 for T=40 gives 0.870. The operator
does not degrade with depth *at all*; what fails is the state, not the map. And this is true of
the cycle arm too: at T=60 it scores 0.016 from step 0 and **1.000** restarted at t=45. Closure
does not make the operator better — it makes the rollout land where the operator already works.

The sharpest number is the `t = T` column, where the model encodes and decodes with **zero**
rollout steps: **base 0.008, cycle 1.000.** Base cannot decode its own encoder's output — its
decoder is defined on the *rolled* manifold, not the encoder's. That is §2's cos 0.19 vs 0.99
cashed out behaviourally, and the consequence is total rather than partial. The
private-trajectory reading holds, and the tunnel has an entrance ramp.

**How long is the ramp? Exactly one step.** The `t` grid here jumps 15 → 20, so this cut cannot
see between 0 and 5 rollout steps; an earlier draft of this section guessed "about three
applications" from that gap, which was interpolation, not measurement. §9's oracle re-projection
at period k=1 decodes at **0.850** after a single application from an encoder state, so one step
is enough. Recorded because it is the kind of inference a sparse grid invites.

The equivalence check makes it quantitative: re-entering at `t` and rolling `T − t` should score
what an ordinary depth-`(T−t)` problem scores, because that is what it is. Cycle matches
`exact@(T−t)` to within **0.000–0.019**. Base runs **0.02–0.11 below** it, which is about what the
~10% of quadratic residues that were never training bases predicts, given base's 0.001
held-out-x.

**Gotcha carried by this probe.** `coldstart_heldout_x` is **not** a clean generalisation readout
and should not be quoted as one. The residue `x_t` of a held-out base is usually itself a base the
model trained on, because squaring maps into the 207-element QR subgroup — measured at **82–93%
overlap for every t ≥ 1** and recorded per run as `heldout_base_leak`. Only the seen-x numbers
above carry the claim.

## 9. Closure can be installed at test time — and what training-time closure actually buys is a *rate*

![test-time re-projection](figures/reprojection/fig_reprojection.png)

§8 says the operator is sound at every depth and the failure is that the rollout leaves its input
domain. That makes an intervention available that costs no training at all: every `k` steps,
snap the state back onto the encoder manifold using the model's **own** decode,
`h ← Enc(decode(h))`. An oracle variant snapping to `Enc(x_t)` with the true residue gives the
ceiling; the gap between them is the price of decode error. 3 seeds, from the `coldstart`
checkpoints, no retraining anywhere.

| arm | setting | T=10 | T=20 | T=30 | T=40 | T=60 |
|---|---|---|---|---|---|---|
| base | no re-projection | 0.919 | 0.028 | 0.004 | 0.004 | **0.000** |
| base | re-project k=8, own decode | 0.867 | 0.751 | 0.654 | 0.569 | **0.377** |
| base | re-project k=8, oracle | 0.868 | 0.868 | 0.873 | 0.868 | 0.871 |
| **+ cycle** | no re-projection | 1.000 | 0.987 | 0.750 | 0.330 | **0.016** |
| **+ cycle** | re-project, own decode *(any k ≤ 10)* | 1.000 | 1.000 | 1.000 | 1.000 | **1.000** |

**Closure is causal, and it is installable at deployment time.** The ungrounded base model,
untouched since training, goes from 0.000 to 0.377 at T=60 and from 0.028 to 0.751 at T=20 —
its horizon moves from **13 to roughly 45** (it crosses 50% between T=40 and T=60; the depth grid
here is coarse). The grounded arm goes to **1.000 at every depth out to T=60**, which is past
where this substrate can still measure it — `N=893` repeats in depth at T=67.

**Re-projection turns one rollout into a chain of restarts, and the chain is multiplicative.**
Base's oracle plateau sits at **0.871** at every period from k=1 to k=8 — the same number §8's
cold-start probe plateaus at, which is what it should be, since an oracle snap *is* a cold start.
Modelling accuracy as `p ** ⌈T/k⌉` with `p = 0.871`:

| | T=20 | T=40 | T=60 |
|---|---|---|---|
| k=8 predicted | 0.661 | 0.502 | 0.332 |
| k=8 actual | 0.751 | 0.569 | 0.377 |
| k=5 predicted | 0.576 | 0.332 | 0.191 |
| k=5 actual | 0.658 | 0.359 | 0.210 |

It tracks within 0.03–0.09 and slightly under-predicts throughout, and it **breaks where it should**
— at k=10 it over-predicts (0.576 vs 0.463 at T=40) because a 10-step segment is approaching the
base horizon of 13 and drift inside the segment stops being negligible.

**So what the training-time constraint buys is the per-restart rate, not the operator.** Cycle's
per-restart reliability is 1.000, and `1.000 ** n = 1.000` for any `n` — flat in depth. Base is
stuck at `0.871 ** n`, which decays exponentially no matter which `k` you pick. Combined with §8
(both operators compute correctly at every depth tested), the reading is that the grounded
constraint never made the *map* better; it made the model able to re-enter its own predictions
without loss.

**The period has a two-sided optimum for the ungrounded arm and none for the grounded one.**
Base peaks at k=8: smaller `k` means more restarts and so more multiplicative loss, larger `k`
means segments that outrun the composition horizon (k=15 > 13 collapses to 0.005). The oracle is
flat in `k` up to 8 because an injected state is always correct, so only the final segment
matters. Cycle is flat because its `p` is 1.

**The snap has to be complete.** At α=0.5 base collapses — 0.041 against α=1.0's 0.658 at T=20,
k=5. A half-snap leaves the state *between* the rolled manifold and the encoder manifold, which is
worse than committing to either.

**Caveat on the 0.871 ceiling.** It is not a pure re-entry-noise number. Roughly 10% of the
207 quadratic residues were never training bases, and base's held-out-x is 0.001, so a chunk of
that ceiling is base's failure to generalise in `x` rather than a cost of re-entering. The chain
model's `p` therefore mixes two things, and the cleanest place to separate them is a run where
the base pool is not ~90% seen.

**Every number in §9 is on *seen* base values.** The probe evaluates over `train_idx` only, so
the grounded arm's 1.000 at T=60 says the *depth* axis is saturated at this modulus with these
bases — not that the model learned squaring. §2's held-out-x for the same arm is **0.32**. The
depth axis and the `x` axis are separate, and only the first one is finished here.

---

# Scale (2026-08-03): the same cut at `N = 9853`

§9 put the grounded arm at 1.000 at T=60 against a first depth-repeat of 67, so `N=893` could no
longer measure its own best configuration. `N = 9853 = 59 × 167` gives 9628 units, **2407**
reachable states (11.6× the 207) and a first depth-repeat of **1149**. Two tags, because 2407
states is well past what the operator's width was chosen for: the **narrow** one changes only
`N`, the **wide** one takes the operator from `d_ff` 1024 to 8192. 3 seeds each, 60k steps
(7.5× §1's budget), cycle weight at §7's optimum w=10, eval to T=120.

## 10. At scale the cycle term collapses; the horizon numbers are **not converged**

| | operator | params | steps | ID (T≤6) | horizon (per seed) | closure@1 |
|---|---|---|---|---|---|---|
| `base` `N=893` *(§1)* | 1024 | 2.1M | 8k | 1.000 | 15, 13, 13 | 0.187 |
| `+cycle` w=10 `N=893` *(§7)* | 1024 | 2.1M | 8k | 1.000 | **51.0 ±3.6** | 0.999 |
| `base` `N=9853` narrow | 1024 | 2.1M | 60k | 0.958 | 7, 7, 7 | 0.099 |
| `base` `N=9853` **wide** | 8192 | 5.8M | 60k | **1.000** | 7, 7, 7 | 0.130 |
| `base` `N=9853` wide, **2× budget** | 8192 | 5.8M | **120k** | 1.000 | **≥10, climbing** | — |
| `+cycle` w=10 `N=9853` narrow | 1024 | 2.1M | 60k | **0.024** | 1, 1, 1 | 0.976 |
| `+cycle` w=10 `N=9853` wide | 8192 | 5.8M | 60k | **0.001** | 1, 1, 1 | 0.998 |

**Read the horizon column at `N=9853` as a lower bound, not a measurement.** An earlier version of
this section reported the base horizon "halving, 13 → 7, independent of capacity". The capacity
control is real — 8× width moves ID 0.958 → 1.000 and leaves the horizon at 7 — but
capacity-independence with saturated ID is *not* evidence of convergence, and it isn't:
doubling the budget takes exact-match at T=7 from **0.176 to 0.996** and the horizon to **10**,
still climbing, with ID already 1.000 in both runs. `N=893` *is* converged (2.5× the budget leaves
it at 13). See [`horizon_convergence/`](horizon_convergence/README.md) — **no `N=893` vs `N=9853`
horizon comparison should be quoted until both sides are converged**, and everything below about
the cycle term at scale inherits the same caveat, since those arms all ran at 60k.

**The cycle term at its `N=893` optimum fails to learn the task at all**, and widening makes it
*worse* (ID 0.024 → 0.001), so this is not capacity. It is §5's "error correction, achieved and
useless" with the cycle term now in the role of the failed intervention, and gotcha 1 at a new
scale: the term unlocks at 30% warmup, calibrated where CE saturated in ~2k steps; at `N=9853` CE
has not converged by step 18k, so the self-generated target teaches while the model is still bad
and the degenerate solution satisfies it perfectly.

**Repairing the schedule confirms the diagnosis and does not restore the advantage** — at 60k
steps, single seed, wide operator:

| cycle w | warmup | ID | T=7 | horizon | closure@1 | closure@20 |
|---|---|---|---|---|---|---|
| — *(base)* | — | 1.000 | 0.176 | 7 | 0.137 | −0.047 |
| 1 | 0.3 | 0.311 | 0.001 | 1 | 0.814 | 0.355 |
| 3 | 0.3 | 0.003 | 0.001 | 1 | 0.991 | 0.161 |
| **3** | **0.6** | **0.996** | 0.102 | 7 | 0.898 | −0.158 |
| **10** | **0.6** | **0.994** | 0.117 | 7 | 0.892 | 0.182 |
| 30 | 0.6 | 0.000 | 0.000 | 1 | 0.998 | 0.998 |

It is the **schedule, not the weight**: w=3 reaches ID 0.996 at warmup 0.6 and 0.003 at 0.3, and
no weight rescues it on the old schedule. With the schedule fixed, no weight extends the horizon
past base's 7, and raising the weight moves closure@20 monotonically (−0.158 → 0.182 → 0.998)
while the horizon does not follow (7 → 7 → 1) — a decoupling that would matter if the baseline
were converged. **It is not**, so *whether the cycle advantage survives scale remains open*, and
the run launched to settle it was killed after its `base` arm.

**Closure has a blind spot and it fired here.** The collapsed arm reads closure **0.998** with
in-distribution accuracy **0.001**: [`rule_structure/`](rule_structure/README.md) §6 confirms the
mechanism — every embedding is the same point, and `on_manifold_cos` is trivially maximised by
collapse. Every closure claim in §2/§6/§7 is measured where ID = 1.000 and stands, and §6 already
restricted its correlation to ID-competent runs, but the rule should now be explicit: **closure
is only interpretable conditional on in-distribution competence.**

## 11. Does the rule get learned at scale? No — see [`rule_structure/`](rule_structure/README.md)

A larger state set is also the capacity-economics test for
[`variable_modulus/`](../variable_modulus/README.md) §3's finding that nothing learns modular
squaring: if lookup were merely *cheaper* than the algorithm, an 11.6× bigger table should shift
the balance. Measured in the CRT coordinate where squaring is exactly the doubling map, group
structure **fell to the permutation null** (translation R² 0.040 → 0.007 against a 0.006 null) on
the ID-perfect wide model. It also shows the cycle term's held-out-`x` gain (0.001 → 0.32) is
real but **not** group-based, and kills a proposed `Enc(x²) ≈ M·Enc(x)` training term. Full
writeup, positive controls and caveats in the child node.

---

## Honest caveats

- **Single modulus, single scale.** `N=893`, 828 units, d=256. Whether the cycle advantage
  survives a larger state space is the natural second cut and is **not** tested here.
- **The interference reading is one substrate at one cycle weight.** The re-entry weight was
  swept; the *cycle* weight was not — §7 now sweeps it and finds §1's w=1.0 was well short of the
  optimum, so the §4 interference numbers are also at an unoptimised cycle weight and the
  re-entry×cycle interaction is untested.
- **§6–§8 ran on L4, §1–§5 on A10G.** The cycle sweep and the cold-start probe are each
  internally consistent (all L4), but any comparison *across* those groups carries a hardware
  change on top of the run-to-run nondeterminism below.
- **Closure is established as an ordinal predictor and a mediator candidate, not a proven
  cause.** §7 breaks the collinearity with the re-entry weight and survives the w=30 reversal,
  which is real evidence, but every manipulation of closure here is still *via* some loss weight.
- **Run-to-run nondeterminism is ~±2 on the horizon at fixed seed** (A10G, non-deterministic
  kernels): the identical `cycle-only` config gave 35/38/37 in `deep60` and 34/35/35 in `w_re0`.
  Differences below ~3 depths should not be read.
- **x-generalisation is deliberately near-constant** (~90% of bases seen). The 0.001 → 0.32
  effect is real but measured on a 10% slice at one setting.
- **The composition-horizon comparison to `mjc`/RHM is cross-substrate and informal.** The
  numbers were not measured with a common instrument.
- **Nothing here tests the actual benchmark.** No submission was built; the architecture uses
  fixed-width zero-padded fields, which the real evaluator does not.

## Gotchas

1. **A self-generated target must not teach before its own fidelity exceeds where the system
   sits.** The consistency arms initially would not train *at all* — CE pinned at chance, the
   re-entry term at chance for the whole run — because both terms were live from step 0. Gating
   them behind a 30% warmup fixed it. This is
   [`endo_expansion`](../../rhm/directed_sculpting/full_loop/endo_expansion/README.md)'s
   *"a grader is a ceiling, approached from whichever side you start"* showing up as an
   optimisation failure rather than a performance ceiling.
2. **Seed the VQ codebook from real operator outputs.** A `randn` codebook sits at an arbitrary
   scale relative to the state and starves the assignment.
3. **Keep `T` out of the recurrent arms' prompt.** If the encoder can see `T`, depth can be
   folded into `h_0` and the rollout stops being the only route to depth.
4. **More gradient steps on the same data hurt.** The iso-compute control (base at 2.5× steps)
   is *worse* — OOD 0.503 → 0.431, T=20 0.026 → 0.011 — which both kills the compute confound on
   the cycle result and reproduces
   [`metering_sweep`](../../rhm/directed_sculpting/full_loop/metering_sweep/README.md)'s
   within-round overfitting (+0.117 tree error at 16× steps) on a new substrate.
5. **`--eval-max-depth` must be ≥ the deepest training depth.** The trajectory table is built to
   the eval depth, so a smaller value indexes off the end mid-training (`IndexError: index 5 is
   out of bounds`) rather than failing at startup.
6. **Check the depth-periodicity margin before choosing `N`, and quote `tail + period`, not the
   tail.** `x^(2^T)` is eventually periodic in `T`, so a badly-chosen modulus lets a model pass
   "depth extrapolation" by discovering a cycle. An earlier version of this line reported the
   upstream Easy margins as "T=4 (N=323) and T=2 (N=899)" — those are the *tails*. The first
   depth at which the map actually repeats is `tail + period`: **10** for N=323 (tail 4, period 6)
   and **14** for N=899 (tail 2, period 12), brute-force verified over all units. `N=893` is 67.
   The generator asserts against `tail + period`; only the prose was wrong.

## Reproduce

```bash
cd experiments/
# main 2x2 + controls (3 seeds)
for s in 0 1 2; do
  MODAL_PROFILE=chromatic modal run --detach one_layer_deeper/ballistic_depth/ballistic_depth.py::ballistic_depth \
    --tag cut1 --arms "base,quant,consist,quant_consist,feedforward" \
    --steps 8000 --eval-max-depth 20 --eval-cap 828 --seed $s
done

# the horizon probe (to T=60; first depth-repeat is 67)
for s in 0 1 2; do
  MODAL_PROFILE=chromatic modal run --detach one_layer_deeper/ballistic_depth/ballistic_depth.py::ballistic_depth \
    --tag deep60 --arms "base,consist" --steps 8000 --eval-max-depth 60 --eval-cap 828 \
    --consist-reentry 0.0 --seed $s
done

# ablations: cycle-only / re-entry-only / iso-compute
for s in 0 1 2; do
  modal run --detach ...::ballistic_depth --tag cyc_only    --arms consist --consist-reentry 0.0 --seed $s
  modal run --detach ...::ballistic_depth --tag re_only     --arms consist --consist-cycle   0.0 --seed $s
  modal run --detach ...::ballistic_depth --tag iso_compute --arms base    --steps 20000      --seed $s
done

# re-entry weight sweep + on-manifold instrument calibration
for s in 0 1 2; do
  for w in 0.0 0.1 0.3 1.0; do
    modal run --detach ...::ballistic_depth --tag w_re$w --arms consist \
      --steps 8000 --eval-max-depth 60 --eval-cap 828 --consist-reentry $w --seed $s
  done
done

# repaired quantiser
modal run --detach ...::ballistic_depth --tag quant_fix --arms "quant,quant_consist" \
  --codebook-size 256 --consist-reentry 0.0 --seed $s

# --- micro-cut (§6-§8) ---
# cycle weight sweep, re-entry pinned at 0 (§7). w=10 is the optimum found.
for s in 0 1 2; do
  for w in 0.01 0.05 0.2 3.0 10.0 30.0; do
    modal run --detach ...::ballistic_depth --tag w_cyc$w --arms consist \
      --steps 8000 --eval-max-depth 60 --eval-cap 828 \
      --consist-cycle $w --consist-reentry 0.0 --seed $s --save-ckpt
  done
done

# cold-start re-enterability probe (§8) — the probe rides along with any run,
# so this tag is just `w_re0`'s config re-run with the new instrument + checkpoints.
for s in 0 1 2; do
  modal run --detach ...::ballistic_depth --tag coldstart --arms "base,consist" \
    --steps 8000 --eval-max-depth 60 --eval-cap 828 --consist-reentry 0.0 \
    --seed $s --save-ckpt
done

# fetch + analyse
MODAL_PROFILE=chromatic modal volume get one-layer-deeper-data \
  /ballistic_depth/<tag>/results_seed<N>.json one_layer_deeper/ballistic_depth/results/<tag>/ --force
python3 one_layer_deeper/ballistic_depth/analyze.py --tag deep60
python3 one_layer_deeper/ballistic_depth/sweep_figure.py
python3 one_layer_deeper/ballistic_depth/closure_horizon.py   # §6 + §7, no GPU
python3 one_layer_deeper/ballistic_depth/coldstart_figure.py  # §8

# --- scale N (running, 2026-08-03) ---
# `--p/--q` and `--d-op-ff` all default to the old behaviour (19/47, operator width =
# encoder width), so every command above reproduces unchanged.
# N = 9853: 9628 units, 2407 reachable states, first depth-repeat 1149.
# Two tags: the narrow one is the controlled continuation (only N changes); the wide one
# hedges the capacity confound, since 2407 states is 11.6x the 207 the width was set for.
for s in 0 1 2; do
  modal run --detach ...::ballistic_depth --tag scale9853 --arms "base,consist" \
    --p 59 --q 167 --steps 60000 --eval-max-depth 120 --eval-cap 4096 \
    --consist-cycle 10.0 --consist-reentry 0.0 --seed $s --save-ckpt
  modal run --detach ...::ballistic_depth --tag scale9853_ff8192 --arms "base,consist" \
    --p 59 --q 167 --d-op-ff 8192 --steps 60000 --eval-max-depth 120 --eval-cap 4096 \
    --consist-cycle 10.0 --consist-reentry 0.0 --seed $s --save-ckpt
done

# test-time re-projection (§9) — loads checkpoints, trains nothing
for s in 0 1 2; do
  modal run --detach one_layer_deeper/ballistic_depth/reprojection.py::reprojection \
    --tag coldstart --arms "base,consist" --seed $s --eval-cap 828
done
python3 one_layer_deeper/ballistic_depth/reprojection_figure.py
```

`--save-ckpt` writes `/ballistic_depth/<tag>/ckpt/<arm>_seed<N>.pt`. §1–§5 saved none, which is
why §8 needed a retrain rather than a re-eval; anything run from here on is re-probeable for free.

Modal volume `one-layer-deeper-data`, results at `/ballistic_depth/<tag>/results_seed<N>.json`.

## Next steps

1. **Scale `N` — now forced, not just advisable.** *Running as tag `scale9853` (2026-08-03).*
   §9 puts the grounded arm at 1.000 at T=60, and `N=893` repeats in depth at T=67, so this
   substrate can no longer measure its own best configuration. `N = 9853 = 59 × 167` (9628 units,
   2407 reachable states, first depth-repeat 1149) buys ~19× the depth headroom; `base` vs
   `+cycle` at §7's optimum w=10, eval to T=120. It also separates the two things mixed into §9's
   `p = 0.871`: the base pool is still 90% of units, but those cover a 4× larger reachable set,
   so held-out-`x` failure and re-entry noise stop moving together.
2. **Held-out modulus.** Everything here is fixed-`N`, so the operator never had to be
   *conditioned on the rule*. Sampling `N` turns this into the arity-2 question — a rule-blind
   operator can only predict the `N`-averaged next state — and is the cut that connects to
   [`RHM_SCULPTING`](../../rhm/RHM_SCULPTING_README.md)'s length-gen finalizer.
3. ~~**Sweep the cycle weight**~~ — done, §7. Optimum at w≈10, horizon 51.
4. ~~**Does closure predict the horizon across arms?**~~ — done, §6. Ordinally yes (rho +0.94
   over 72 runs), on a calibrated scale no.
5. **Soft attractors between the two extremes** — EMA-VQ, annealed temperature, or quantise-at-eval
   only — to find out whether §5 is really about differentiability or about something else.
6. **Re-run the §4 interference sweep at the corrected cycle weight.** The dense/evaluative
   inversion was measured at cycle w=1.0, which §7 shows is well short of the optimum; whether
   re-entry is still harmful at w=10 is untested, and the interaction is the interesting object.
7. ~~**Test-time re-projection**~~ — done, §9.
8. **Densify the cold-start `t` grid between 0 and 5 steps.** §8's grid jumps 15 → 20 and §9 had
   to supply the 1-step point. The shape of the decoder's entrance ramp is currently two points.
