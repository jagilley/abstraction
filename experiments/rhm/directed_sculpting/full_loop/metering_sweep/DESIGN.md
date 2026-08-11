# Design — the metering sweep

**Written before any run.** Committed ahead of the launch so the direction, the mechanism, the
discriminators and the thresholds are on record. Nothing below is a result.

**Date**: 2026-08-01. **Claim under test**:
[`ideas/meta_learning_under_metered_data.md`](../../../../../ideas/meta_learning_under_metered_data.md)
§Predictions, first bullet:

> **Falsified if** the uniform→oracle gap does *not* shrink toward zero as the monitor:collect
> ratio goes to zero.

---

## 1. The accounting, read off the ladder

[`../ladder.py`](../ladder.py) charges the [`Meter`](../../../rhm_repair_cost.py) at four sites
per round, all inside `_run_policy`:

| site | charge | at published config |
|---|---|---|
| `counterfactual_lp` (`channel_env.py:720`) | `n_channels × mon_n` | 5 × 1280 = 6400 |
| `measure_floors` (`channel_env.py:763`) | `n_channels × floor_n × floor_draws` | 5 × 512 × 3 = 7680 |
| forecast tap (`ladder.py:440`) | `forecast_n` | 512 |
| **monitor total `M`** | | **14592** |
| `charge_collect` (`ladder.py:470`) | `collect_budget` | 8192 |

`ratio = M / collect_budget = 14592 / 8192 = 1.78125` — exactly the number every ladder run in
the repo reports. The one older run (`ladder_l4_v1`, `mon_n=3000`) gives
`(5·3000 + 7680 + 512)/8192 = 1.8936`. Both numbers are now derived, not quoted.

## 2. Mechanism — stated crisply, before running (design note (a))

**(i) The meter is inert.** `Meter` is constructed as `Meter()` with `budget=None`, so `_check`
never fires; `meter.ratio` appears only in the results JSON. **Nothing in the loop reads it.** A
bare charge multiplier on monitoring would therefore change the reported ratio and *nothing else*
— every arm's trajectory would be bit-identical. A sweep of that multiplier alone is guaranteed
to return an exactly flat curve **for a reason that has nothing to do with metering economics**,
and reading such a curve as a falsification would be an instrument defect pointing at a negative.

> **A price is only a price if paying it costs you something else.** For monitoring's price to be
> real it must *displace collection*.

**(ii) `uniform` and `oracle` are both tap-blind.** `allocate` gives `uniform` a constant
`ones(5)` drive and `oracle` a constant one-hot on the tree. Neither reads `errors`, `lprog`,
`visits` or `reducible`. So the uniform→oracle prize **cannot** depend on how much monitoring
buys, or on how good the taps are — only on how much collection the monitoring bill leaves
behind. This is what makes the price/information separation of design note (b) achievable
*exactly* here rather than approximately.

**(iii) Therefore the operative variable is the collection budget `B`.** With monitoring
quantity `M` held fixed and a price `λ` per monitored sequence, at a fixed total per-round data
spend `T = λM + B`:

```
B(λ) = T − λM          ratio r = λM / B          B(r) = T / (1 + r)
```

Information is constant (same `mon_n`, `floor_n`, `floor_draws`, `forecast_n` ⇒ bit-identical
taps); only the price moves; and the price buys itself out of the collection budget. That is
design note (b)'s headline sweep, implemented so that it is not a no-op.

**(iv) What shape the prize should have.** Let `E(d)` be tree FM error as a function of tree
transitions per round. `oracle` gives the tree `0.96·B`, `uniform` gives it `0.20·B` — a fixed
**4.8× data ratio at every point of the sweep**. The prize is `P(B) = E(0.20B) − E(0.96B)`.

- Wherever `E` is **log-linear** in data (`E ≈ a − b·log d`), `P = b·log(4.8)` — **constant in
  `B`**. A flat prize is the *default* expectation across any log-linear stretch, not a
  falsification.
- `P → 0` as `B → ∞` because `E` bottoms out at the aleatoric floor (`env_check` E2: tree floor
  **0.189**, measured). **This is the doc's predicted limb.**
- `P → 0` as `B → 0` because `E` tops out at the untrained/warm-start level and *both* arms sit
  there. **This limb is trivial and is not evidence for the metering claim.**

So `P(B)` is predicted to be **humped**, and the interesting empirical question is *where the
published 1.78× point sits on that hump* and *whether either shoulder is reachable at all*.

**(v) The high end is not the mjc mechanism.** `mjc/ballistic/directed`'s 22× was a *measurement
subsidy*: 2240 free teleported probes to steer a budget of 100, which removes the relevance
term's job. That mechanism acts on **tap-driven** arms. It cannot act on the uniform→oracle
prize, because neither arm has a tap. If the prize collapses at `r = 22` here, the cause is
starvation, not subsidy — and §6 below says how to tell.

## 3. Predictions, written before the run

| # | prediction | direction |
|---|---|---|
| **P1** | uniform→oracle prize falls as `r → 0` (`B → T`) | the doc's claim; **falsifier if flat** |
| **P2** | prize falls as `r → 22` (`B → T/23`) | starvation limb; predicted, but *uninformative about metering* |
| **P3** | prize is **flat** wherever `E` is log-linear, i.e. across the middle of the sweep | the null both limbs must beat |
| **P4** | absolute `oracle` and `uniform` errors **fall together** on the P1 limb (saturation) and **rise together** on the P2 limb (starvation) | the discriminator |
| **P5** | in **E2** (monitoring *quantity* swept at fixed `B`), the prize does **not** move at all | control on this whole design — if it moves, §2(ii) is wrong |
| **P6** | tap recoveries (`visits_only` 84%, `reducible_only` 18% of the prize) **can** move under E2 and should not move under E1 | separates price from information a second way |

**The strongest single test** is the **matched-ratio pair**: two runs at nearly the same `r` and
very different `B`. If `r` is the operative variable they agree; if `B` is, they do not.

| pair | `r` | `B` | `λ` |
|---|---|---|---|
| E1-A `r=0.25` | 0.250 | 18227 | 0.312 |
| E1-B `B=65536` | 0.223 | 65536 | 1.000 |
| E2 `M×0.25` | 0.445 | 8192 | 1.000 |
| E1-B `B=32768` | 0.445 | 32768 | 1.000 |

## 4. Design

**E1-A — price at fixed total spend (the headline).** `T = 22784` (the published total:
14592 + 8192), `M = 14592` fixed, sweep `r`; `B = round(T/(1+r))`, `λ = r·B/M`.

| `r` | 0.0625 | 0.25 | 0.5 | **1.78125** | 4 | 8 | 22 |
|---|---|---|---|---|---|---|---|
| `B` | 21444 | 18227 | 15189 | **8192** | 4557 | 2532 | 991 |
| `λ` | 0.0918 | 0.3123 | 0.5205 | **1.0000** | 1.2492 | 1.3882 | 1.4941 |

`r = 1.78125` gives `λ = 1.0` and `B = 8192` **exactly** — the published anchor, reproduced from
inside the new parameterisation rather than asserted.

**E1-B — abundance (secondary, separately labelled).** `λ = 1`, `M` fixed, `B ∈ {32768, 65536}`
(⇒ `r = 0.445, 0.223`). Total spend is *not* held fixed; this is the "data becomes free" limb the
fixed-`T` design cannot reach, and it supplies both matched-ratio pairs.

**E2 — information at fixed price and fixed budget (the amount-sweep, separately labelled).**
`B = 8192`, `λ = 1`, scale monitoring quantity ×0.25 and ×4:

| scale | `mon_n` | `floor_n` | `forecast_n` | `M` | `r` |
|---|---|---|---|---|---|
| ×0.25 | 320 | 128 | 128 | 3648 | 0.445 |
| ×1 | 1280 | 512 | 512 | 14592 | 1.781 |
| ×4 | 5120 | 2048 | 2048 | 58368 | 7.125 |

**Arms** (design note (f), plus one): `uniform`, `oracle`, `visits_only`, `reducible_only`,
`value_red`, **`oracle_dup`**. 3 seeds, paired within seed, everywhere.

**Change one thing (design note (c)).** Every point in E1-A/E1-B differs from every other *only*
in `(collect_budget, mon_price)`; every point in E2 differs only in monitoring quantity. Geometry,
drift calibration, rounds, arms, seeds, `n_eval` and the drive are identical throughout, and the
anchor sits inside each sweep rather than being imported from a differently-configured run.

## 4b. The saturation certificate — added before any result was read

**(iv) above says a flat prize is the default expectation, and that is a problem for the
falsifier, not for the apparatus.** `partial_hetero` G3 measured **−0.05 to −0.08 per doubling**
of task data in *this* readout: good news (the readout is demonstrably alive to the data-quantity
knob) and bad news (that is a log-linear response, and `oracle`/`uniform` is a fixed **4.8× data
ratio at every point**, so a log-linear `E` makes `P = b·log 4.8` mechanically constant). A flat
curve would then mean *"the saturation shoulder is out of range"* — **not** *"the metering claim
is wrong"*.

So the sweep must certify which limb it is on, and it already contains the data to do it. Tree
error should be a function of **tree transitions per round**, `d = share_tree × B`, and the arms
overlap heavily in `d` across budgets (`oracle` at `B=4557` delivers 4375; `uniform` at
`B=21444` delivers 4289; `oracle` at `B=15189` delivers 14581 against `uniform` at `B=65536`'s
13107). Pooling all 54 (arm, budget) cells gives `E(d)` over a **~66× span** (198 → 62915), and
since `P(B) = E(0.20B) − E(0.96B)` exactly, `E` *determines the whole prize curve*.

| `E(d)` over the reachable span | what a flat prize then means |
|---|---|
| still log-linear at the abundant end | **the null is scoped**: saturation unreachable, falsifier not actually exercised |
| bending toward a floor (`env_check` E2: **0.189**) | **the falsifier genuinely fires** |
| cells at matched `d` **disagree across arms** | tree error is not a function of tree data at all — `metered_repair` §4d's timing warning, and the prize cannot be read as data economics |

`aggregate.py::data_curve` prints this **before** the prize, for that reason. The third row is
the self-falsifying one and is why the collapse is checked rather than assumed.

## 5. Calibration gates — all before the curve is read

- **C0 — back-compat.** `rhm.verify_backcompat.verify()` passes; `mon_price=1.0` takes the
  original `int(n)` charge path, so every prior repro command is bit-identical.
- **C1 — the meter moves as intended.** For every run, realised `meter_ratio` equals the target
  `r` to <0.5%, and `mon_n/floor_n/floor_draws/forecast_n` are *identical* across all of E1
  (certifying information is constant while price moves).
- **C2 — the anchor reproduces.** E1-A at `r = 1.78125` reproduces `ladder_fix_s*`'s
  uniform→oracle prize of **0.0629** within the pointwise noise floor.
- **C3 — pointwise noise floor.** `oracle` vs `oracle_dup` at **every** point, following
  `partial_hetero` §2 (which measured **−0.0033 ± 0.0039**). A curve is "flat" only against the
  floor measured *at that point*; the floor is expected to widen at small `B`.
- **C4 — liveness at both ends.** Before reading the prize: (a) the 6-arm spread at each point
  must exceed that point's noise floor, and (b) `oracle`'s tree error must fall from round 1 to
  round 12. If either fails at an end, the readout is dead there and no prize value from that
  point is reportable.
- **C5 — world intact.** Ground-truth best Δ`d*` = **+0.000** off-tree and ≈ **+1.7** on-tree at
  every point; `warm_fm_check` within the published range.

## 6. How each outcome will be read

| observation | reading |
|---|---|
| prize falls at low `r` **and** both absolute errors fall (P4) | **the claim is supported** — saturation kills allocation |
| prize falls at low `r` but absolute errors do **not** fall | not saturation; something else, and not reportable as support |
| prize falls at high `r` with both errors **rising** | starvation; **explicitly not** support for the metering claim |
| prize flat across the whole sweep, above the pointwise floor | **the falsifier fires** — the metering claim is wrong in its current form on this apparatus, *scoped* by whether the saturation shoulder was reachable at all |
| all arms inside the floor at an end | readout dead there; report as dead (C4), not as a prize of zero |
| matched-ratio pairs disagree | **`r` is not the operative variable**; budget is — a correction to how the claim is phrased, independent of which way the curve goes |

**A clean null is a reportable outcome.** Across PRs #15/#16 this node produced seven instrument
defects, every one biased toward the hoped-for positive; the corresponding hazard here is a
*flat* curve manufactured by an inert knob, which §2(i) is written to make impossible to publish
by accident.

## 7. Standing caveat carried into the writeup

[`mjc/on_policy/metered_repair/README.md`](../../../../mjc/on_policy/metered_repair/README.md)
§4d finds that on the mjc ladder A-error is **not** a monotone function of target budget share and
**no process column predicts the outcome column** — raising the possibility that ladder outcomes
track allocation *timing* rather than allocation quality. That does not block this experiment
(uniform and oracle both allocate with zero timing variance — `tree_share_sd` = 0.000 for both),
but it bounds how much any ladder-based prize can be trusted, and it is a live alternative
explanation for the *tap* arms' behaviour at every point of this sweep.
