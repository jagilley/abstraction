# FILES — `confabulation/`

Complete file-by-file index for the confabulation-test sub-experiment. Standing context and all results live in [README.md](README.md); the parent index is [../FILES.md](../FILES.md).

## Code files

| File | Purpose |
|---|---|
| `rhm_confabulation.py` | The whole sub-experiment, two Modal entrypoints. **`confabulation_test`** — forks `rhm_latent_loop`'s `ntp_aux{,_cl}` wake recipe verbatim (same DGP/model/FM/gate/init/losses; reproduces its val loss to 3 dp), then runs the report battery: report heads on `post_block{n-1}` for IMPL (spherical-k-means cluster of the FM-residual *direction*) / BEHAV / ENT / WORLD, over a sequence-level split. Four tests — (1) capacity-swept third-person **observer ladder** `O_input`/`O_io`/`O_act` plus a half-data budget control, (2) **channel ablation** (`shuffle_r`/`shuffle_p`/`zero_r`/`zero_p` substituted into `a_j`, final block re-run with attention intact), (3) **matched-KL steering** of residual span vs FM-prediction span with BEHAV-flip as the built-in control, (4) a **confabulator** head trained *and* evaluated with theory-only access. Every IMPL number is swept over instrument-FM capacity (`d_head` × `mlp_mult`, 1.4%→33.5% of the predicted blocks) and gated on `ens_cos` + hierarchy-η². Resumes from a saved wake checkpoint keyed on `(cond, n_steps, seed)`. **`eta2_recheck`** — reloads the saved checkpoints (no retraining) and reports residual hierarchy η² under *both* estimators, importing `_compute_hierarchy_eta2` from `rhm_latent_loop` so the comparison to the reference is exact. |
| `README.md` | This sub-experiment's writeup |
| `FILES.md` | This file |

Helpers inside `rhm_confabulation.py` worth knowing by name: `_kmeans_fit`/`_kmeans_assign` (IMPL labels, fit on train rows only so the label definition never sees test), `_eta2_by_level` (pooled-over-positions η² — **not** comparable to the reference last-token estimator, see README), `_ensemble_cos` (fresh-FM residual invariance = the noise-vs-gap discriminator), `Observer` (causal stack with an optional continuous side-channel: M's logits for `O_io`, `post_block0` for `O_act`), `run_ladder`, `report_row`.

## Children

| Folder | Summary |
|---|---|
| [`temporal/`](temporal/README.md) ([FILES](temporal/FILES.md)) | The same battery with the FM's conditioning gap shifted from six blocks to one token, both arms on one frozen M at matched report positions, plus the exact BP oracle. Privileged access survives the axis change (+0.014 to +0.027) but its partial `R²` against oracle belief revision at matched surprisal is 0.0000 — self-report and observer track `B_t` equally, so the charge is real and public. Supplies this battery's first validated ceiling (`O_h6`). |
