# Files — sculpt_slip

**Up**: [README.md](README.md) · **Parent**: [../FILES.md](../FILES.md)

## Code files

| file | purpose |
|---|---|
| `slip_gate.py` | The whole cut, three Modal entrypoints sharing one cached substrate (controller / generator / value trained once under the deterministic world, then frozen — so only the FM's training target ever varies). `_regen_flagged` mirrors Stage 3d's slippery actuator but returns **which** acted blocks slipped, which is the exact aleatoric label. `_train_block_fm_slip` is Design 2 (realised slippery targets → the FM becomes a mean-predictor and the slip lands in its residual). `_train_fm_arm` runs any Step-2 arm on an identical data stream. `_precision_operator` builds the low-rank `Π` from accumulated residual covariance. Entrypoints: `slip_gate1` (Step 1, is the slip directionally identifiable — matched on ‖r‖ with atom-aware strata borrowed from `../gates_ab.py`), `slip_gate2` (Step 2, six arm families incl. a gating ceiling and a geometry control), `slip_budget` (closeout, the three bounding arms across a data-budget grid). |
| `__init__.py` | package marker |

## Auxiliary docs

| file | contents |
|---|---|
| `README.md` | The writeup — why this substrate, Step 1 (positive: direction 0.70–0.75 matched vs 0.50 for the norm, with a mechanism-revealing trend in `q`), Step 2 (null: the geometry control matches the treatment; the prize was only ~0.03 and `top1` exactly 0), the closeout budget sweep, why a learning-progress estimator was **not** built (six prior negatives, tabulated), scope, reproduction, gotchas. |
