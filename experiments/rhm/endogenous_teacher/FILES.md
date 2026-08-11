# FILES — endogenous_teacher

**Up**: [README.md](README.md) · [../FILES.md](../FILES.md)

## Code files

| file | purpose |
|---|---|
| `endogenous_teacher.py` | The whole cut: trains a shared base (plain NTP + open-loop FM), runs Gate 0 (`R²(rres ~ nll)`, the cheap check that the residual is not just surprisal), then the five loss-weighting arms (`uniform`, `nll`, `res`, `res_orth`, `res_shuffled`) from that shared base. One Modal L4 function. |

## Docs

| file | summary |
|---|---|
| `DESIGN.md` | The question (exogenous vs endogenous teaching signal), the rank-normalised weighting design, Gate 0's threshold, and the depth-profile prediction — all written down before the run. |
| `README.md` | Results writeup. *(pending — results not yet discussed)* |

## Key design points a future agent should not re-derive

- **Rank-normalised weights** (`_rank_weights`): every arm gets a bit-identical weight
  multiset (`w = 2·rank/(N−1)`, mean 1); only the assignment to positions differs. Weight
  scale, variance and effective learning rate are matched by construction, and
  `res_shuffled` is an exact distributional match to `res`.
- **Level naming follows the repo convention** (`rhm_latent_loop.py:174`): `_generate_with_traces`
  returns `level_features[0]` = **root**, `[L]` = leaves, and levels are reported as `d{L-ell}`,
  so **`d1` is shallowest and `d6` is the root**. Getting this backwards inverts the entire
  depth-profile readout, which is the primary result.
- **Training uses flat concatenated windows**, not aligned sequences — the phase-diversity fix
  from `RHM_LATENT_LOOP` (every position NTP-supervised). It also removes the position↔level
  coupling from the per-position weighting, which would otherwise confound Gate 0.
- **The aligned level attribution excludes the last position** (its NTP target does not exist)
  and is chunked (full intermediates for 2048×64×256×9 would be ~1.2 GB).
- **The FM is open-loop** — it watches, it does not inject. This cut is about whether the
  residual can *teach*, not about the injection channel.
- Regime matches `RHM_LATENT_LOOP` exactly (`v16 s2 L6 m4`, `8L/8H/256D`, FM `1L/8H/16d`,
  `post_block0 → post_block6`) so its reference lines transfer.
