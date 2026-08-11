# Teacher SNR: the outer loop's obstruction is the estimator, not the teacher signal

**Up**: [../README.md](../README.md) (drift_value_loop) · **Idea doc**: [../../../../ideas/performance_error_is_the_bridge.md](../../../../ideas/performance_error_is_the_bridge.md)
**Direct parent**: [../README.md](../README.md) **Cut 3** — the corrected teacher, which found the value-relevant FM prediction error gives a clean interior optimum at `b=0.5` where control is flat, and reported its self-tuning as *"directional-but-noisy."* This node measures that noise.
**Code**: `teacher_snr_analysis.py` (post-processing only, no compute); the runs use [`../online_value_loop.py`](../online_value_loop.py) with the new `--dense-teacher` flag.
**Status**: done. A (analysis of existing runs, 3 seeds); B (3 seeds); C (**1 seed**). Single task family (corridor pusher). **Date**: 2026-08-11.

## One-liner

Cut 3's b-landscape reproduces, but the online loop that is supposed to climb it **cannot see it from a
single epoch**: the bowl (0.00397) is shallower than the per-epoch nuisance (0.00497), giving **SNR 0.80**,
and across 42 gradient samples the perturbation explains **0.0%** of the advantage variance. A dense
teacher that averages the M+1 per-round FM-error samples the drive already computes **does not help**
(sd ratio 1.059) — and the reason is measurable: **74% of the spread is between-epoch state variation that
within-epoch averaging structurally cannot touch.** Removing world motion (Type-1) collapses the absolute
noise ~8× and lifts landscape SNR to 1.40, but leaves the noise *structure* unchanged (B/W 2.86 → 3.24) and
the loop still does not climb, because the local gradient at the operating point is ~0.6× the noise. On
this substrate and budget the obstruction is the **search procedure**, not the teacher's form.

## Why we ran this

[`../README.md`](../README.md) Cut 3 swapped the outer-loop teacher from downstream control (a near-blind
grader) to the value-relevant forward-model prediction error, and found a clean interior optimum. Its
remaining defect was noisy self-tuning, attributed to "shallow-bowl REINFORCE." The idea doc
[performance_error_is_the_bridge](../../../../ideas/performance_error_is_the_bridge.md) proposed that the
missing ingredient was a **baseline-subtracted** teacher (the songbird's "flexible performance benchmark",
Gadagkar et al. 2016) and predicted it would clean up the self-tuning. Experiment A was meant to be that
retrofit. It is not: **the EWMA baseline and an advantage normalizer were already implemented**
(`online_value_loop.py`, the `base` / `run_sq` block), so `adv = R − base` is already exactly the proposed
`benchmark − error`. A became a diagnosis instead.

## A — SNR of the sparse (Cut-3) teacher

**Method.** The fixed-`b` arms hold the balance constant, so *all* their epoch-to-epoch variation in
`corridor_err` is nuisance. That gives the noise floor directly; the b-sweep gives the signal. Post-transient
(2nd half of epochs), 3 seeds, from the existing `teacher_s{0,1,2}` and `selftune15_s{0,1,2}` runs.

| quantity | value |
|---|---|
| bowl depth (5-pt, 2nd-half means) | 0.00397, argmin **b=0.5** (reproduces Cut 3) |
| per-epoch nuisance sd at fixed b | 0.00497 |
| **SNR** | **0.80** |
| corr(eps, adv), n=42 | **+0.008** (r² = 0.0001) |
| samples for 2σ certainty on the gradient sign | ~3300 (the loop gets 14) |
| final b, init 0.15, target 0.5 | 0.313 / 0.081 / 0.604 |

The landscape is shallower than a single epoch's measurement noise. Across three seeds the self-tuning
trajectory is consistent with a random walk — one seed moves away from the optimum, and the pooled
perturbation–advantage correlation is indistinguishable from zero. Cut 3's landscape claim stands (the
fixed-b sweep resolves it by averaging ~8 epochs per arm); its self-tuning characterization was optimistic.

**A paired contrast does not rescue it.** Epoch-wise correlation between b-arms within a seed is ρ=+0.595,
which looked like substantial shared nuisance to cancel via common random numbers. But
`sd_paired = sd·√(2(1−ρ))` = 0.90·sd — pairing halves the shared component and doubles the independent one,
so at ρ≈0.6 they nearly offset: **paired SNR 0.89 vs unpaired 0.80.**

## B — the dense teacher (drift, 3 seeds)

**Method.** `score_error` is already called *every round* inside `grounded_scores` to pick the collection
cell — the same per-cell FM error the teacher uses, computed M=6× more often than the outer loop consumes
it. `--dense-teacher` accumulates its corridor mean and grades on the epoch average of M+1 samples instead
of the single epoch-end probe. `corridor_err_dense` and the raw per-round samples are logged
**unconditionally**, so any run measures its own within-epoch structure. Flag defaults off; all prior Cut-3
results reproduce.

*Design note.* `arng` is constructed identically per arm from `seed+100` and the flag consumes no RNG, so
the perturbation sequence is **bit-identical** to the corresponding `selftune15_s{N}` run (verified). The
sparse/dense comparison is exactly paired: same eps, different teacher.

| | b0.3 | b0.5 | b0.7 |
|---|---|---|---|
| seed0 | 1.182 | 1.249 | 0.993 |
| seed1 | 1.070 | 0.952 | 0.967 |
| seed2 | 1.228 | 1.035 | 0.853 |

**Mean sd ratio 1.059** (dense/sparse) — no reduction, in 9/9 cells. Pooled corr(eps, adv) = +0.133 vs the
sparse +0.008; at n=42 the SE is ~0.154, so this is **within one SE of zero** and not distinguishable from
the sparse value. Final b: 0.190 / 0.071 / 0.362 (dense) vs 0.313 / 0.081 / 0.604 (sparse); mean |error to
target| 0.292 vs 0.237. Dense moves *less far* in the same per-seed direction.

**The variance decomposition is the transferable result.** From the raw per-round samples:

| component | drift (3 seeds × 3 arms) |
|---|---|
| within-epoch sd (averageable) | 0.00278 |
| between-epoch sd (not averageable) | 0.00470 |
| **B/W variance ratio** | **2.86** |
| ceiling on dense at M=∞, `√(B/(B+W))` | **0.861** |

Within-epoch averaging can only remove the within component, so the *best case* for this idea was a 14% sd
reduction — never the 0.378 that a naive √(M+1) predicts. The two-component model predicts 0.885 at M=6;
we observe 1.059. The gap is plausibly staleness: dense samples are taken *before* each round's FM update,
the epoch-end probe *after*.

## C — Type-1 (stationary DGP, 1 seed)

**Method.** Identical config, one knob: `--drift-mode none` (the code labels this regime `stationary`). The
scarce needle stays at a fixed location, so "success" — an FM that predicts the corridor — is a stable
target. Under drift there is no fixed target, and `benchmark − error` conflates *I improved* with *the world
moved*. Pre-registered readings: B/W → ~1 would mean world motion was the state noise; B/W ≈ 2.9 would mean
it is the FM's own learning; a flat landscape would mean Type-1 exhausts.

| run | regime | sd ratio | within sd | between sd | B/W | corr(eps,adv) |
|---|---|---|---|---|---|---|
| drift s0 | drift | 1.141 | 0.00280 | 0.00507 | 3.28 | −0.003 |
| drift s1 | drift | 0.996 | 0.00286 | 0.00482 | 2.84 | +0.360 |
| drift s2 | drift | 1.039 | 0.00269 | 0.00417 | 2.41 | +0.412 |
| **static s0** | stationary | 0.804 | **0.00035** | **0.00062** | **3.24** | −0.044 |

**The noise structure is unchanged (B/W 2.86 → 3.24), but the absolute noise collapses ~8×.** On the matched
3-point landscape:

| | spread | per-epoch sd | SNR | argmin |
|---|---|---|---|---|
| drift | 0.00145 | 0.00546 | 0.27 | b0.5 |
| static | 0.00100 | 0.00071 | **1.40** | b0.5 |

The landscape survives rather than exhausting at this horizon (96 rounds), and its minimum stays at b=0.5.
But the loop still does not climb: corr(eps, adv) = −0.044, trajectory 0.15 → 0.16. A local check explains
why — at the operating point b=0.15, `sigmoid'·σ` gives perturbations of ±0.089, and the landscape slope
over the 0.3→0.5 segment is ~0.005/unit-b, so the per-perturbation signal is 0.00045 against a noise sd of
0.00071: **local SNR ≈ 0.62.** Global SNR of 1.40 is a statement about the bowl's *extremes*; the loop
starts on a shallow edge and never sees it.

## What we take from this

- **The dense/sample-rate idea is a general negative with a measured ceiling.** Averaging within an epoch
  cannot reduce state variation, and that variation is intrinsic to a forward model that is still learning
  — it did not change when world motion was removed. This is not specific to drift.
- **The baseline term was already standard practice.** `adv = R − base` is the REINFORCE baseline; the
  songbird's "flexible performance benchmark" maps onto something the code already did.
- **Cut 3's diagnosis was half right.** The teacher swap fixed *what to measure* — control is a blind
  grader, FM-error is not, and that reproduces cleanly. The remaining gap to a working online loop is
  *how to search*, which no reformulation of the teacher signal addresses. A score-function estimator on a
  **single scalar parameter** with expensive noisy evaluations is a poor fit; the fixed-b arms find the
  optimum trivially because they hold b constant and average.
- **Suggestive, not established**: that a performance-error teacher needs sampling that is dense *relative
  to how fast competence changes*. Gadagkar's birds get ~50k renditions across a slowly-improving skill;
  this loop takes 6 samples per epoch with an FM update every round. We did not test this — it would need a
  substrate that decouples measurement rate from learning rate.

## Scope and caveats

Single task family (corridor pusher, `grid_n=6`, 96 rounds, 16 outer epochs). **C is one seed** — the B/W
ratio there rests on 3 arms within that seed, and the absolute-noise collapse is large enough to be robust
to seed, but the landscape spread (0.00100) and the local-SNR estimate are single-seed and should be treated
as directional. The local-gradient calculation uses the 0.3→0.5 segment as a proxy for the slope near
b=0.15; the landscape below b=0.3 was never measured, so that number is an estimate. Nothing here tests the
architecture claims in the idea doc (agency gate, arity gap, playback control) — only the afferent teacher's
statistics on this substrate.

## Reproduction

```bash
cd experiments/

# B — dense teacher under drift, 3 seeds
for s in 0 1 2; do
  modal run --detach mjc/drift_value_loop/online_value_loop.py::online_value_loop \
      --tag dense_s$s --seed $s --b-init 0.15 --task-geom corridor --noise \
      --dense-teacher --arms "online_front,b0.3,b0.5,b0.7"
done

# C — Type-1 (stationary), 1 seed
modal run --detach mjc/drift_value_loop/online_value_loop.py::online_value_loop \
    --tag type1_s0 --seed 0 --b-init 0.15 --task-geom corridor --noise \
    --dense-teacher --drift-mode none --arms "online_front,b0.3,b0.5,b0.7"

# A + B + C — all numbers in this README (post-processing only, no compute)
python3 mjc/drift_value_loop/teacher_snr/teacher_snr_analysis.py
```

A needs no runs: it reads the existing `teacher_s{0,1,2}` / `selftune15_s{0,1,2}` results from
[`../figures/`](../figures). Modal volume layout is unchanged (`/data/online_value_loop/<tag>/`).
