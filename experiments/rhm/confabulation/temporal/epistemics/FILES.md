# FILES — `confabulation/temporal/epistemics/`

Complete file-by-file index for the temporal-FM-epistemics sub-experiment. Standing context
and all results live in [README.md](README.md); the parent index is [../FILES.md](../FILES.md).

## Code files

| File | Purpose |
|---|---|
| `temporal_epistemics.py` | The whole sub-experiment, one Modal entrypoint. **`temporal_epistemics`** — the parent battery repointed from **privacy** to **composition**: the report channel, the observer ladder, steering, the confabulator and the CL arm are dropped; the frozen `M`, the report positions `p ∈ [1, T-2]`, the four-point instrument sweep, the junk-residual guards and the exact BP oracle are kept unchanged. Measures a **source × target decode matrix at matched readout capacity**. Sources: `r = Δ − FM_T(h≤t)` · `delta` · `h_next` · `p_minus` · `rev_pair = p⁺ − p⁻` (SPEC.md Gate C's matched FM₁/FM₂ pair, `SlotFM`) · `r_mart` and `p_mart` (the **exact** ε₁ = 0 innovation, by `v`-way counterfactual substitution — beyond the pre-registration) · `pos` (one-hot position, the floor) · `r_shuffled` (position-preserving null) · the scalar norms `|r|`/`|delta|`/`|rev_pair|`/`|r_mart|` (the standing entropy negative control). Targets: `B_pxs[D]` (graded oracle belief revision residualised **nonparametrically** on position × exact surprisal — the primary; stratum means fit on train rows only) · `B_bp[D]` (surprisal only, the secondary, so the gap between the two families reads off how much of a decode is calendar) · `AE_pxs[D]`/`AE_raw` (`aleatoric_fraction`'s exact per-position A/(A+E) of the state update) · `bp_sur` · `H_tot` · Gate B's syn/dis contrast scored on the `B` probes' own output under position × exact-surprisal strata. Readout capacity is the new sweep axis: linear (exact ridge, λ on a held-in split) / MLP-16 / MLP-64 / MLP-256. Also runs **Gate C** end to end for the first time — capacity invariance across the instrument sweep, the martingale calibration (corpus vs self-sampled continuations), and the compression term `‖p⁻_learned − p⁻_mart‖` reported rather than bounded. Conditions: `ntp_aux` (resumes the frozen `M` shared with the parent and the depth battery) and `cr_base`. |
| `README.md` | This sub-experiment's writeup |
| `FILES.md` | This file |

Helpers worth knowing by name. Imported, deliberately not reimplemented, so "same harness"
is literal: `_generate_with_traces`/`_ensemble_cos` from
[`../../rhm_confabulation.py`](../../rhm_confabulation.py); `_eta2_at_positions`/`_cross`
from [`../temporal_confabulation.py`](../temporal_confabulation.py) (the residual
materialises at `t+1`, so the η² diagnostic must be scored against the ancestors of
`1..T-1`); `_strata`/`_stratified_auc`/`_partial_r2`/`_r2` from
[`../../../conditional_revision/gates_ab.py`](../../../conditional_revision/gates_ab.py);
and [`../../../conditional_revision/oracle.py`](../../../conditional_revision/oracle.py)
for `B_t`, `H_tot`, exact surprisal and — via the default-off `return_leaf_posteriors`
kwarg — the two next-token posteriors that weight the A/E split.

Defined here:

- `SlotFM` — the Gate C forecaster. `TransformerForwardModel` plus a per-position token
  slot added into its residual stream; `FM₂` gets `x_{t+1}` in the slot at `t`, `FM₁` a
  learned `MASK` id. Causality is preserved for the forecast at `t` (the slot at `t' ≤ t`
  holds `x_{t'+1}`), so `FM₂` sees exactly `h≤t` plus the arriving token.
- `counterfactual_pass` — `h6[p]` for all `v` arriving tokens at every position, by
  prefix-cropped substitution. Supplies three things at once: the A/E target, the exact
  martingale forecast `E_{a∼p_M}[h6[t+1] | a]`, and the drift readout. Carries
  `aleatoric_fraction`'s substitution-vs-plain-forward identity check as a hard assert.
- `drift` — `‖mean_i r‖ / sqrt(E‖r‖²/n)`, pooled over positions. Reads **1.0** under the
  martingale null and above 1 under systematic drift; dimensionless, so the corpus and
  self-sampled cells and the exact and learned residuals all sit on one scale.
- `_stratum_residualise` — `y` minus `E[y | stratum]`, means from train rows only, with
  strata unseen in train falling back to the global train mean so no test row is
  residualised by its own label. Used with **position × exact-surprisal** strata for the
  primary targets and surprisal-only for the secondary. Both controls are load-bearing:
  `R²(B ~ bp)` is 0.07–0.22, so a *small* probe trained on raw `B` spends its capacity on
  the surprisal part; and on the RHM constituent boundaries sit at fixed offsets, so
  without the position term the decode matrix ranks sources by how well they encode the
  calendar (observed in the wiring smoke — one-hot `pos` beat every content source).
- `_standardise` / `_ridge` / `_r2_np` — matched-capacity readout machinery. Per-dimension
  z-scoring from train rows (without it the sources' differing norms masquerade as capacity)
  and a closed-form ridge at the linear rung (optimiser noise at the small end is exactly
  the confound that would fake a concentration curve).

## Children

None — this is a leaf.
