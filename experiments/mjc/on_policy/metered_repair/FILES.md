# `metered_repair/` — file index

Up: [README.md](README.md) · parent node: [`../README.md`](../README.md) (on_policy) · mjc node: [`../../README.md`](../../README.md)
Direct parent experiment: [`../directed_on_policy/`](../directed_on_policy/README.md) (E3, PR #4)
Ported claims: [`rhm/directed_sculpting/full_loop/`](../../../rhm/directed_sculpting/full_loop/README.md) (PRs #15/#16)
Idea docs: [`ideas/meta_learning_under_metered_data.md`](../../../../ideas/meta_learning_under_metered_data.md) · [`ideas/breadth_as_grader_heterogeneity.md`](../../../../ideas/breadth_as_grader_heterogeneity.md)

## Code files

| file | purpose |
|---|---|
| `floor_tap.py` | **E4** ([writeup §4](README.md#4-e4--the-ladder-10-arms--26-rounds--3-seeds)) — E3's ladder with the reducibility tap **repaired** and the **visited-but-irreducible cell placed**. Copy-and-modify of `../directed_on_policy/directed_on_policy.py` (E3 stays byte-reproducible). Four changes: (A) `matched_pairs_floor` measures the aleatoric floor by k-NN matching **on the FM's residual**, from the agent's own metered in-region batch — the honest substitute for RHM's exact re-execution, which `Body` forbids. Running on the residual rather than raw Δs is load-bearing: measured bias 0.36–0.46 vs **1.00–1.15**, the latter being larger than the noise regions' true floor; (B) the **stock** tap `red = (err − floor)/err` with EMA smoothing, and both taps computed every round for every arm so only the allocation rule differs; (C) a **widened eval sector** (`pref_width`, `eval_spread`) so an on-reach noise region C exists — E3 could not place one because its eval cone was narrow, not because embodiment forbids it; (D) E3's drift rate and budget **kept**, with K 6→4 (the canonical A/B/C/D 2×2) and `mon_n` 40→60 as the explicit trade. Two later arms (`reddelta-only`, `value-reddelta`) add a **rate** tap built from the same estimator, testing rate-vs-level. Carries two gates (`--certify-only`): geometry places C, and the floor estimator separates noise from noise-free regions. |
| `floor_tap_agg.py` | Cross-seed aggregator for E4. **Merges tags by trailing `_s<N>`**, so the rate-tap arms run separately and combine with the finished ladder; errors rather than overwriting if a policy appears twice for one seed. Paired per-seed contrasts (`value-red` − `value` = the repair; `value-red` − `visits-only` / − `reducible-only` = whether the conjunction needs both terms), the leak into irreducible regions, and both taps' reducible-vs-irreducible separation. |
| `necessity.py` | **E5** ([writeup §5](README.md#5-e5--necessity-3-arms--4-budgets--16-rounds--3-seeds)) — the port of RHM's necessity sweep. A drift **generator** with two levels, `b_j(t) = c_j·β(t) + ε_j(t)`: one shared latent β (deep — explains damage everywhere at once) plus independent per-region walks ε (surface). Per-event damage matched across arms **analytically** (`s_shared = scale·√f`, `s_local = scale·√(1−f)` ⇒ equal increment variance for every mixing fraction `f`) and **measured** anyway. Sweeps samples-per-event with fixed allocation on region A, and reads **two instruments**: repair at A (surface) and at an off-reach region A′ that receives zero collection (deep). Includes a `pure_shared` calibration arm that establishes whether the transfer readout has dynamic range at all. |
| `necessity_agg.py` | Cross-seed aggregator for E5. Reports the three gates (matched damage, A′ genuinely uncollected, instrument dynamic range) before the headline slope of transfer fraction against log₂(samples per event), per seed and across seeds. |

## What each script's gates are, and why they run first

Both cuts stand on an instrument that could fail silently, so both ship a gate that is a strict
**prefix** of the real run rather than a separate re-implementation that could drift from it:

```bash
modal run mjc/on_policy/metered_repair/floor_tap.py::floor_tap --quick --certify-only --tag cert
modal run mjc/on_policy/metered_repair/necessity.py::necessity --quick --calibrate-only --tag ncal
```

E4's gate: (i) region C is placed and genuinely visited, A stays visited, off-reach regions stay
unvisited; (ii) the matched-pairs floor comes out **small on the curl regions** — whose true noise
amplitude is 0 by construction, so that number *is* the estimator's curvature bias — **large on the
noise regions**, and never above the matched-FM ceiling (a matched FM cannot beat the floor).

E5's gate: (i) A′ is off-reach and A is on-reach; (ii) measured per-event damage agrees across arms
inside `damage_tol`. RHM's `calibrate_sigma` bug — matching accumulated displacement rather than the
event, which left one level at 3.49 nats against a 0.60 target — is the reason this is measured rather
than trusted.

## Modal volume layout

```
/data/metered_repair/certify/<tag>/results.json        # E4's gates only
/data/metered_repair/floor_tap/<tag>/results.json      # E4
/data/metered_repair/necessity_cal/<tag>/results.json  # E5's gates only
/data/metered_repair/necessity/<tag>/results.json      # E5
```

Results are mirrored locally to `figures/floor_tap_<tag>/` and `figures/necessity_<tag>/`, which is
what the two aggregators read.

## Shared machinery (unchanged, and deliberately so)

Both scripts import [`../../embodied.py`](../../embodied.py) (`Body`, no `set_state`; every
`env.step` charged) and [`../../arm_env.py`](../../arm_env.py) (`curl_fields` / `noise_fields`
multi-region local drift). **Neither file is modified by this node**, so
[`../verify_backcompat.py`](../verify_backcompat.py) and every prior cut are untouched.
