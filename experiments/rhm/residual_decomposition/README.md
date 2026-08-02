# Residual decomposition: what the FM residual's rank was actually measuring

**Parent:** [`rhm/README.md`](../README.md) · **Files:** [`FILES.md`](FILES.md)
**Companion code (language + vision domains):** [`a2a_forward/residual_decomposition/`](../../a2a_forward/residual_decomposition/README.md)
**Supersedes as the instrument of record:** [`rhm/RESIDUAL_RANK_README.md`](../RESIDUAL_RANK_README.md)
**Belief this tests:** [`dimensionality_expansion.md`](../../../beliefs/dimensionality_expansion.md)

## The headline

Across three domains, the residual is **not** a set of leftover directions. It is a graded shadow of the whole computation, and it obeys a power law:

> ### **res_var(i) ∝ act_var(i)^β**
>
> across the main model's principal directions `i`, with **R² = 0.95–0.99**.

β is measured to **±0.01 across a 4× FM-capacity range**, while that same capacity sweep moves the frontier's *level* by 1.8×. **The shape and the level are separate, independently-moving quantities — and the old single number `R_res` tracked neither.**

The exponent is what matters. Since β < 1, the leftover shrinks more slowly than the signal, so the fraction the FM fails to capture goes as `act_var^(β−1)`: loud directions are absorbed nearly perfectly, quiet ones barely at all. A photocopier reproducing bold lines and losing faint pencil marks. Naive residual rank was counting the faint pencil marks.

## What this explains that was previously unexplained

Four standing puzzles in this repo are the same fact:

| Observation | Where | Explanation |
|---|---|---|
| Residual rank flat (90–96%) across a **17× residual-norm range** | [`RESIDUAL_RANK_README`](../RESIDUAL_RANK_README.md) Exp. 3 | Entropy rank reads *shape*, not *size*. β and the level are orthogonal; the sweep moved the level only. |
| "Language residual is full-rank, ~200/256, yet cosine 0.97" | `a2a` language, cited in the belief doc | Measured at a **1-block gap**, where the FM saturates and the leftover is float noise. Noise is isotropic ⇒ near-full rank. It was a noise floor. |
| Contact eff-rank 3.60 **>** free 2.52 — "residual rank ∝ DGP complexity" retired | [`mjc/contact_residual`](../../mjc/contact_residual/README.md) cut #1 | Rank rises when structured error is removed and diffuse error remains. The negative was the instrument, not the hypothesis. |
| `rank ⊥ noise` | a2a | Same: rank is blind to magnitude by construction. |

**Rank had failed as an instrument three times.** It failed for one reason, and that reason is now measured.

## The instrument that replaces it

Two numbers, not one:

- **β — the shape.** How absorption falls off across the model's directions. A property of *the model's computation*: invariant to FM capacity to ±0.01.
- **`R_res_participation` — the frontier's dimensionality.** *How many of the model's own working directions still carry meaningful unexplained computation.*

`R_res_participation` is the participation ratio of the frontier spectrum `{(1−ρ_i)·λ_i}`, where λ_i is the model's variance in direction i and ρ_i the fraction the FM absorbed there. It counts **in the model's own basis, weighted by how much computation the model actually does in each direction**. Naive `R_res` instead SVDs the raw residual and counts any direction with variance in it — weighting a direction carrying 0.01% of the computation the same as one carrying 30%.

That is the entire difference, and it is enormous:

| | naive `R_res` | `R_res_participation` |
|---|---|---|
| RHM (L6_m4, 3-block gap) | 84 | **7** |
| MNIST (3-block gap) | 75 | **20** |
| language (3-block gap) | 245 | **223** |

Alongside them, **frontier mass** (fraction of activation variance unexplained) gives size: RHM 1.4–4.8%, MNIST 2.3%, language **10.0%**. Language's frontier really is large and high-dimensional — the original intuition was right, it just wasn't what rank was reporting.

## Results

### 1. The claimed decomposition is false as arithmetic

The belief asserted `A = FM(earlier) + Residual` partitions activation space, with `R_act ≈ R_comp + R_res` and `R_res` "the slice of `R_act` the self-model hasn't absorbed yet". `R_res` had been measured throughout the repo; **`R_act` and `R_comp` never had.**

Measured (RHM, 16 conditions; ratio should be ~1.00):

| setting | gap | 25% | 50% | 75% | 100% |
|---|---|---|---|---|---|
| L4_m2 | b0→b1 | 2.58 | 2.65 | 2.64 | 2.66 |
| L4_m2 | b0→b3 | 2.82 | 2.93 | 2.98 | 2.98 |
| L6_m4 | b0→b1 | 2.40 | 2.46 | 2.47 | 2.48 |
| L6_m4 | b0→b3 | 2.73 | 2.78 | 2.80 | 2.81 |

Three failures, each confirmed in all three domains:

1. **`R_comp` is a redundant readout of `R_act`.** 72.7 vs 72.7 (RHM), 207.0 vs 207.0 (language), 47.8 vs 47.8 (MNIST). A good self-model reproduces the activations it predicts, so `rank(FM output) → rank(A)` by construction. It cannot be an independent "absorbed directions" counter, which removes the conservation the flow claims need.
2. **`R_res` > `R_act` everywhere** (120.4 vs 72.7). A "slice of" cannot exceed the thing it slices.
3. **`R_res` is blind to magnitude.** L4_m2 b0→b1: relative residual 0.0509 → 0.0030 (17×) while `R_res` goes 116.0 → **120.4** — it *rises*.

The subadditivity argument was the error: `rank(X+Y) ≤ rank(X)+rank(Y)` is a theorem about **hard** rank, which entropy effective rank does not obey. Hard rank is degenerate here anyway (N ≫ D makes all three exactly D).

### 2. The geometric intuition survives — in the regime that matters

`alignment_index` measures residual variance inside the model's top-k directions against the chance baseline k/D (0 = chance, 1 = perfectly nested):

| gap | RHM | MNIST | language |
|---|---|---|---|
| 1 block | +0.07 … +0.17 | +0.16 | +0.03 … +0.11 |
| **3 blocks** | **+0.61 … +0.74** | **+0.66** | +0.08 |

At the wide gap on RHM, residual variance in the top-16 directions is **0.66–0.80 against a chance baseline of 0.125 — 5–6× chance.** The residual really does live in the model's own directions. At the narrow gap it does not, because there is nothing there but noise.

### 3. Saturation, not gap width, creates the noise regime

The initial reading was "narrow gap = noise". MNIST corrected it. Where the FM **saturates** — drives relative residual to ~0 — β collapses and rank inflates, in all three domains:

| domain | rel. residual @100% | cosine | naive `R_res` | frontier mass | β |
|---|---|---|---|---|---|
| RHM L4_m2 | 0.0030 | 1.0000 | 120.4 / 128 | 0.0000 | 0.13 |
| MNIST | 0.0056 | 1.0000 | 120.9 / 128 | 0.0001 | 0.14 |
| language | 0.0079 | 1.0000 | 176 / 256 | 0.0001 | 0.09 |

β_narrow = **0.11–0.17** across three domains. Gap width is only a proxy; saturation is the variable. **A near-full residual rank is the signature of an FM too good for the gap it was given, not of a rich frontier.**

*The confound that proved this:* the first MNIST run used a causal FM against the ViT's **bidirectional** block ([`vit.py`](../../a2a_forward/vit.py) applies no mask). Structurally unable to saturate, it plateaued at 0.059 and manufactured a spurious law at the narrow gap (β = 0.661, R² = 0.968) — the same class of confound as Exp. 2's head-count mismatch. Fixed by matching the FM's attention mode.

### 4. β is capacity-invariant; the level is not

The core of the shape/level split, at the 3-block gap:

| domain | β (± sd over 4× capacity) | frontier mass, 25% → 100% |
|---|---|---|
| MNIST (d=128) | **0.610 ± 0.007** | 0.0336 → 0.0231 |
| RHM L6_m4 (d=128) | **0.607 ± 0.009** | — |
| RHM L4_m2 (d=128) | 0.555 ± 0.019 | 0.0299 → 0.0144 |
| language (d=256) | **0.358 ± 0.003** | 0.1758 → 0.0995 |

Capacity moves tail absorption from 0.03 to 0.35 and leaves the exponent alone.

### 5. β tracks how concentrated the computation is — not width, not domain

MNIST 0.610 and RHM 0.607 at d=128 looked like β might be a width property. A width sweep in both domains **falsified that** — β *rises* with width, the opposite of what would explain language:

| domain | d=64 | d=128 | d=256 | d=512 |
|---|---|---|---|---|
| RHM | 0.523 | 0.611 | 0.640 | 0.673 |
| MNIST | 0.537 | 0.614 | 0.704 | 0.718 |
| language | — | — | **0.358** | — |

At matched d=256, RHM and MNIST give 0.640 / 0.704 against language's 0.358. **Language's low β is a genuine domain effect.**

What unifies all 9 points is activation concentration: **corr(R_act/d, β) = −0.877** (−0.956 pooled). Language has `R_act/d` = 88.3% — computation smeared over nearly every direction — and the lowest β. RHM/MNIST at d=256 sit at ~27% and ~0.64–0.70.

**Control.** β is a log-log slope and the domains span different numbers of decades, so a refit over a matched top-2-decade window checks for a range artifact. **Language is robust** (0.362 full-range vs 0.360 matched, over 255 directions — its whole spectrum fits the window). The *within-domain width trend* is **not** robust: RHM/MNIST have only 9–38 directions in their top 2 decades and MNIST's trend reverses there. Hold the cross-domain claim firmly, the width trend loosely.

### 6. Frontier mass is scale-invariant

Holding the FM at exactly 100% of a block across an 8× width range (63× block params), frontier mass shows no trend — MNIST 0.0274 / 0.0233 / 0.0135 / 0.0206, RHM 0.0204 / 0.0332 / 0.0282 / 0.0267. **The level is set by the FM's capacity *ratio* to a block, not its absolute size.**

This corrects a natural intuition about scale. Relative naive `R_res` *does* fall with model size (RHM 75.3% → 46.3%, MNIST 69.8% → 40.5%) — far more than [`MODEL_SCALE_README`](../../a2a_forward/MODEL_SCALE_README.md)'s 93.8% → 91.8%, which was measured at the narrow gap. But it is not the residual becoming lower-rank: against the model's *own* active dimensionality, `R_res/R_act` **rises** (RHM 1.82 → 2.18, MNIST 1.30 → 1.89). The denominator moved. Relative residual rank fell and meant the opposite of what it looked like.

### 7. The claimed flows do not happen (RHM wake-sleep, 2×2)

`{OL, wake-sleep} × {fixed DGP, new rules each cycle}`, 6 cycles, every cycle measured on a **fixed held-out probe** identical across conditions (so `R_act` moves only if the model's computation moved):

| condition | ΔR_act | ΔR_res | Δfrontier mass |
|---|---|---|---|
| OL_FIXED | +51.9 | −4.1 | +0.096 |
| WS_FIXED | +52.7 | −5.9 | +0.118 |

- **Wake-sleep ≈ open-loop on every metric.** The compression loop has no measurable effect on the triple.
- **`R_act` rises in all arms including OL** — its growth is ordinary continued training, not compression or novelty.
- **`R_res` does not drain toward 0** (flat at 135–143 over 6 cycles). The MNIST absorbing collapse 23.9 → 13.5 does not reproduce.

**The novelty arm is confounded and answers nothing.** New rule sets at matched (v,s,L,m) share no structure with the old, so the model fully relearns and fully forgets: val loss on cycle-0 rules 1.41 → 4.51 while val on current rules stays ~1.44. That is task switching, not novelty; the fixed-probe frontier explosion (0.60) is measuring destroyed computation. Also caveat §7's frontier growth: the FM gets a fixed 1500-step repoint budget while the model's computation grows richer, so it is partly chasing a moving target.

## Children

### [`trajectory/`](trajectory/README.md) — the decomposition over training (2026-08-02)

The third axis: the audit measures the decomposition statically, the ratchet over wake-sleep cycles, and `trajectory/` over the **base model's own training checkpoints** — the axis every complexodynamics claim rests on and the one this instrument had never been run on. It re-cuts [`RHM_COMPLEXODYNAMICS_README`](../RHM_COMPLEXODYNAMICS_README.md)'s rise-and-fall (measured with the rank instrument §1 retired) by refitting fresh FMs on the saved checkpoints and computing the retired *and* trusted metrics on the identical `(A, P)`, over 2 gaps × 2 arms × 4 capacities.

**The arc survives; the exponent does not certify itself.** On `frontier_mass`, the pressure arm peaks and declines in **7/7** measurements and the control arm is monotone-up in **7/7** — so "the descent requires an annealer" now holds across two gaps and four observer bounds. Two directions hold in every cell: **β rises** and **`R_res_participation` falls** monotonically over training, i.e. the FM's error goes from noise-shaped and spread over nearly every direction to computation-shaped and contracted, while the host gets steadily *harder* to predict.

But **β fails §4's capacity-invariance check at every checkpoint where it is measuring anything** (spread 0.065–0.174 vs ±0.01; the only in-tolerance points are saturated ones reading §3's noise floor). Gap width trades the two failures rather than separating them, and 0/16 checkpoints across both gaps pass both trust gates. §4's invariance was established sweeping capacity at fixed gap and never sweeping gap at fixed capacity — this substrate falls outside the regime it mapped. Consequence: on a trajectory this instrument yields robust **directions** and **contrasts**, and no trustworthy **scalar**.

## What survives of the belief

The core — continual learning as growing representable directions, a persistent frontier as the signature of health — is **untouched**; none of it was tested here. What died is the arithmetic operationalization: the partition, the three-column health meter, and the flows across it. The intuition that the residual maps the model's own structure is **confirmed at 5–6× chance** — it does so by *grading* the model's directions, not by *partitioning* them.

## Reproduction

```bash
cd experiments/
# RHM: static audit (2 settings x 4 capacities x 2 gaps)
modal run --detach -m rhm.residual_decomposition.rhm_decomposition_audit::decomposition_audit
# RHM: wake-sleep 2x2
modal run --detach -m rhm.residual_decomposition.rhm_decomposition_ratchet::decomposition_ratchet
# RHM: over training checkpoints -- see trajectory/README.md for the gap/capacity variants
modal run --detach -m rhm.residual_decomposition.trajectory.rhm_decomposition_trajectory::decomposition_trajectory
# language + MNIST + width sweep: see a2a_forward/residual_decomposition/README.md

# metric unit tests (local, no GPU)
python3 -m rhm.residual_decomposition.test_decomposition
```

Volume layout: `/data/{setting}/residual_decomposition/{gap}/cap{N}pct/`, `/data/residual_decomposition_audit/summary.json`, `/data/residual_decomposition_ratchet/{condition}/`.

## Next steps

1. **Re-measure the retired negatives with the new instrument.** [`mjc` cut #1](../../mjc/contact_residual/README.md) retired "residual rank ∝ DGP complexity" on a rank negative; that negative is now explained. β and `R_res_participation` on the same data would settle whether the hypothesis was ever wrong.
2. **What makes language's β low?** `R_act/d` = 88% is the proximate correlate, but concentration and β may both be downstream of underfitting. A well-fit language model (or a deliberately underfit RHM at matched `R_act/d`) separates them.
3. **A non-destructive novelty intervention** — shared structure across cycles (same rules, deeper level; partial rule swap) so the model accumulates rather than thrashes. The belief's central mechanism is still untested.
