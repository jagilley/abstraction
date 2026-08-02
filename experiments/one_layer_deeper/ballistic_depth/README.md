# Ballistic depth — what actually extends a learned operator's composition horizon

**Up**: [../README.md](../README.md) (one_layer_deeper) · **Files**: [FILES.md](FILES.md)
**Substrate**: repeated modular squaring, `y = x^(2^T) mod N`, from [tilde-research/one-layer-deeper](https://github.com/tilde-research/one-layer-deeper)
**Status**: complete — 5 arms + 3 ablations + a 4-point weight sweep + an instrument calibration, 3 seeds each. **Date**: 2026-08-02.

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

## Honest caveats

- **Single modulus, single scale.** `N=893`, 828 units, d=256. Whether the cycle advantage
  survives a larger state space is the natural second cut and is **not** tested here.
- **The interference reading is one substrate at one cycle weight.** The re-entry weight was
  swept; the *cycle* weight was not.
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
5. **Check the depth-periodicity margin before choosing `N`.** `x^(2^T)` is eventually periodic
   in `T`; on the upstream Easy tiers the first repeat is at T=4 (N=323) and T=2 (N=899), so a
   badly-chosen modulus lets a model pass "depth extrapolation" by discovering a cycle. The
   generator asserts against this.

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

# fetch + analyse
MODAL_PROFILE=chromatic modal volume get one-layer-deeper-data \
  /ballistic_depth/<tag>/results_seed<N>.json one_layer_deeper/ballistic_depth/results/<tag>/ --force
python3 one_layer_deeper/ballistic_depth/analyze.py --tag deep60
python3 one_layer_deeper/ballistic_depth/sweep_figure.py
```

Modal volume `one-layer-deeper-data`, results at `/ballistic_depth/<tag>/results_seed<N>.json`.

## Next steps

1. **Scale `N`.** The single-modulus caveat is the biggest one. `N = 9853 = 59 × 167` (2407
   reachable states, first depth-repeat 1149) is already characterised in `squaring_mod.py` and
   is the obvious next rung.
2. **Held-out modulus.** Everything here is fixed-`N`, so the operator never had to be
   *conditioned on the rule*. Sampling `N` turns this into the arity-2 question — a rule-blind
   operator can only predict the `N`-averaged next state — and is the cut that connects to
   [`RHM_SCULPTING`](../../rhm/RHM_SCULPTING_README.md)'s length-gen finalizer.
3. **Sweep the cycle weight**, which was held at 1.0 throughout.
4. **Does closure predict the horizon across arms?** With 12+ configurations already run, closure
   at a fixed `t` versus horizon is a cheap regression, and would turn §2's mechanism claim from
   a two-arm contrast into a slope.
5. **Soft attractors between the two extremes** — EMA-VQ, annealed temperature, or quantise-at-eval
   only — to find out whether §5 is really about differentiability or about something else.
