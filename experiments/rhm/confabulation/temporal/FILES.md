# FILES — `confabulation/temporal/`

Complete file-by-file index for the temporal-confabulation sub-experiment. Standing context and all
results live in [README.md](README.md); the parent index is [../FILES.md](../FILES.md).

## Code files

| File | Purpose |
|---|---|
| `temporal_confabulation.py` | The whole sub-experiment, one Modal entrypoint. **`temporal_confabulation_test`** — forks the depth battery's report machinery with the decomposition swapped to the temporal axis (`h6[t+1] = h6[t] + FM_T(h6[≤t]) + r_temp`, Gate-0 update parametrisation, frozen M) and runs **both** arms on one frozen model at one set of report positions (`p ∈ [1, T-2]`), so the inert-vs-charged comparison is at matched machinery. Conditions: `ntp_aux` / `ntp_aux_cl` (the battery's wake recipe, resuming from and writing to the battery's own ckpt path) and `cr_base` (loads `conditional_revision`'s cached plain-NTP base, trains nothing). Targets `TEMP-IMPL` / `TEMP-MAG` / `IMPL` / `DEPTH-MAG` plus the standing `BEHAV` / `ENT` / `WORLD`. Five tests — (1) capacity-swept observer ladder `O_input`/`O_io`/`O_act`/**`O_h6`** (a genuine ceiling on this axis only) with the half-data budget control, (2) channel ablation, (3) matched-KL steering against **two** self-theory spans (forecast state, and the scale-matched forecast update), (4) the confabulator, (5) **the charge validation** — per-position advantage regressed on the exact BP oracle's `B_t` with exact surprisal partialled out, plus Gate B's synonym-vs-disambiguating family contrast under position × exact-surprisal strata with a within-position permutation null. Every IMPL number is swept over instrument capacity and gated on `ens_cos` + hierarchy η². The model-independent oracle is computed once and cached to the volume as an `.npz`. |
| `README.md` | This sub-experiment's writeup |
| `FILES.md` | This file |

Helpers worth knowing by name. Imported, deliberately not reimplemented, so "matched machinery" is
literal: `_kmeans_fit`/`_kmeans_assign`/`_balance`/`_eta2_by_level`/`_ensemble_cos`/`_make_head`/
`_fit_head`/`_head_acc` from [`../rhm_confabulation.py`](../rhm_confabulation.py);
`_strata`/`_stratified_auc`/`_auc`/`_partial_r2`/`_partial_r2_rank`/`_r2` from
[`../../conditional_revision/gates_ab.py`](../../conditional_revision/gates_ab.py) (so the charge
validation uses the same atom-aware estimator that produced Gate B's 0.690); and
[`../../conditional_revision/oracle.py`](../../conditional_revision/oracle.py) for `B_t`.

Defined here: `_eta2_at_positions` (the reference η² estimator with the position map exposed, because
`r_temp` materialises at `t+1` — asserted in the smoke to reduce to `_eta2_by_level` at
`positions = arange(T)`); `_strat_meandiff` + `_perm_within` + `_meandiff_z` (stratified mean
difference of the per-position advantage with a within-stratum permutation null, `bincount`-vectorised
so 200 permutations per cell are affordable); `_cross` (exact pair encoding of two stratum labellings —
arithmetic like `a * K + b` silently aliases, a documented `gates_ab` gotcha).

## Children

| Folder | Summary |
|---|---|
| [`epistemics/`](epistemics/README.md) ([FILES](epistemics/FILES.md)) | The same harness repointed from **privacy** to **composition** — report channel, observer ladder, steering, confabulator and CL dropped; frozen M, report positions, instrument sweep, guards and oracle kept. A source × target decode matrix at matched *readout* capacity. **The pre-registered concentration effect is absent**: the raw update `Δ` decodes oracle belief revision at least as well as the residual `r` everywhere (d2/MLP-64: 0.449 vs 0.362 on `ntp_aux`, 0.411 vs 0.366 on `cr_base`), and an **exact ε₁ = 0 innovation** built by counterfactual substitution only ties `Δ` on `cr_base` — so this is not forecaster weakness, and the temporal FM looks epistemically inert as a signal-former, leaving timing as its remaining claim. Runs `conditional_revision`'s never-run **Gate C** end to end: capacity invariance passes, the martingale calibration gets an exact null (0.99 self-sampled / 1.09 corpus), and `ε₁` is measured rather than bounded. |
