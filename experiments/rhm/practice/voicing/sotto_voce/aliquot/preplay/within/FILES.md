# within — machinery record

**Up**: [`README.md`](README.md) (the pointer) · [`../FILES.md`](../FILES.md) (preplay).

## Code files

| file | purpose |
|---|---|
| `within.py` | the Modal CPU scoring pass (`score_arm`, `score`): rebuilds the plant and the critic from `vo_heads.pt`, writes the critic's logit, the per-slot readouts `pw` / `pf` / `pr` and the projection-form readouts `gw` / `gf` per row to `within_scores.npz`; gates Z, A, C, B, H, R, P; `falsify` (10/10 required, P1 reported) |
| `reduce_within.py` | the numpy reducer: groups by `(slot key, obs bytes)`, the pair-weighted within-context AUC beside the pooled AUC on the same rows, the group-structure facts, the bank's diet facts; writes `figures/within_reduction.txt` and `.json` |

## Docs and figures

| file | purpose |
|---|---|
| `NOTES.md` | decisions, the gate table with what each was shown to fail on, five defects beside corrections, cost and reproduction |
| `figures/within_reduction.txt` | the table of record, tag `wi2`: [A] group structure, [A2] what a group is not, [B] within-context beside pooled per level and split, [C] the weighting, [D] the fitted controls, [E] the projection's own fitting diet |
| `figures/<tag>/<arm>/within_gates.json` | the per-arm gate records |
| `results/` | launch logs |

## Runs

`wi0` (no `pf`), `wi1` (with `pf`), `wi2` (with `pr`, `gw`, `gf`; of record). Eight CPU containers each, 272–577 s per
arm, no GPU. Volume `rhm-scaling-data:/rhm_practice_within/<tag>/<tag>/<arm>/`. Inputs: the six soundboard arms'
dumps and overtone's two (`../FILES.md`).
