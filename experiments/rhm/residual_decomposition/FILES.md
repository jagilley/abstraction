# FILES — residual_decomposition

Parent: [`README.md`](README.md) · [`rhm/FILES.md`](../FILES.md)

## Code files

| File | Purpose |
|---|---|
| `decomposition.py` | **The metric library — shared by all three domains, so RHM / language / MNIST are scored by literally the same code.** Three views of the same (A, FM-prediction) pair: `naive_triple` (the triple exactly as `dimensionality_expansion.md` defines it, plus the subadditivity ratio the belief says should be ~1); `residual_geometry` (does the residual occupy A's own directions? `alignment_index` against the k/D chance baseline, containment curve, principal angles, per-PC absorption spectrum ρ_i with a spectrum-matched null for the absorption step); `repaired_triple` (partitions A's variance along A's own axes — exact partition `R_comp_v + R_res_v = R_act`, plus **`R_res_participation`**, the frontier's dimensionality and the headline replacement for naive residual rank). `_pr` keys retained as deprecated aliases. Pure numpy, no torch/Modal. |
| `test_decomposition.py` | Validates `decomposition.py` against synthetic residuals of **known** geometry — isotropic / nested in A's top PCs / confined to A's tail. 9 assertions. Caught two real problems before any GPU was spent: ρ was unbounded in A's low-variance tail directions (producing a nonsense step of 140.74), and a raw absorption step is not evidence of preferential absorption because A's own spectral decay manufactures one for free. Run: `python3 -m rhm.residual_decomposition.test_decomposition` |
| `rhm_decomposition_audit.py` | **Parts A+B on RHM.** 2 settings (L4/m2, L6/m4) × 4 arch-matched FM capacities (25–100% of one block) × 2 prediction gaps (`post_block0→post_block1`, `→post_block3`). Follows `RESIDUAL_RANK_README` Exp. 3 so numbers stay comparable, and adds the 3-block gap Exp. 3 never ran — the regime where the FM genuinely cannot replicate the computation and a law exists. Deliberately does *not* reuse `rhm_residual_rank.train_matched_fm`, whose save path has no gap field and would overwrite Exp. 3's results. |
| `rhm_decomposition_ratchet.py` | **Part C: do the claimed flows happen?** 2×2 factorial `{OL, wake-sleep} × {fixed DGP, new rules per cycle}` over 6 cycles, so compression and novelty are separable. Every cycle measured twice — on a **fixed held-out probe** identical across all cycles and conditions (canonical; `R_act` can only move if the model's computation moved) and on the cycle's own data — addressing the belief's own caveat that `R_act` is data-dependent. |
| `analyze_summaries.py` | Post-hoc analysis, CPU-only. `shadow_law` fits `res_var ~ act_var^β`; `beta_matched_window` refits over a common top-2-decade window to test whether a cross-domain β difference is a fitting-range artifact (returns `n_dirs` so an estimate built on too few directions is visible); `analyze_width_sweep` tabulates β vs width with the concentration correlation; `analyze_ratchet` reports the forgetting check. Reads any domain's `summary.json`. |

## Children

None — this is a leaf node. The language and vision ports of the same measurement live under [`a2a_forward/residual_decomposition/`](../../a2a_forward/residual_decomposition/README.md) (they need that project's data and model classes) and import `decomposition.py` from here.

## Related, elsewhere

| Path | Relationship |
|---|---|
| [`../RESIDUAL_RANK_README.md`](../RESIDUAL_RANK_README.md) | The experiment this supersedes as instrument of record. Its Exp. 3 finding — rank flat across a 17× residual-norm range — is explained here rather than contradicted. |
| [`../../a2a_forward/residual_decomposition/`](../../a2a_forward/residual_decomposition/README.md) | Language (4L/4H/256D GPT on FineWeb-Edu), MNIST (4L/4H/128D ViT), and the cross-domain width sweep. |
| [`../../../beliefs/dimensionality_expansion.md`](../../../beliefs/dimensionality_expansion.md) | The belief under test; §"The decomposition, corrected" summarizes this node. |
| [`../../mjc/contact_residual/README.md`](../../mjc/contact_residual/README.md) | Cut #1's rank negative (contact eff-rank 3.60 > free 2.52), now explained by the same mechanism. |
