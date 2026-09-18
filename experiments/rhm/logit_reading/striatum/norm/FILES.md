# Files — logit_reading/striatum/norm

**Up**: [README.md](README.md) · **Parent**: [../FILES.md](../FILES.md)

Imports `striatum/task.py`'s `train_actor` / `actor_apply` / `ridge_solve`, `junction/diets.py`'s `cons_depth`,
`junction/analyze.py`'s `Rows`, `striatum/analyze.py`'s `match_sign_balanced` / `_auc` / `tbl` and
`phasic.py`'s `balance_tokens`, all unchanged.

## Code files

| file | purpose |
|---|---|
| `diets.py` | `cell_key(etype, j)`; `strat_diets` (within-cell terciles of a per-window score — mean outcome or mean realised damage — with equal counts per `(etype, j)` cell, so the cell histogram is bit-identical across the three diets); `mix_diets` (fixed window count, varied quiet share, edited windows nested random subsets); `anchor_diets` (`full` and `natural_sized`). |
| `task.py` | The node. `norm_ckpt` (L4): one pair of forward passes per cell, one frozen actor, one clean-only critic, a critic per diet from masked rows of the shared Gram; the norm, response and outcome surprise on the fixed rows at both anchors and offsets 0 / 4 / 8; the same-prefix twins built with `phasic.py`'s recipe at a 2.0 selection caliper with both surprisals saved, the critics applied to the twin states; writes `stepNNNNNN_norm_<tag>.{json,npz}`. `norm_sweep` (CPU coordinator) fans the six cells out. Own Modal app and image with the `results/` and `.log` `ignore`. |
| `analyze.py` | Local reduction: the reproduction gate, the diets' worlds and the cell-histogram check, the norm's level and its regression on `full`, the response contrasts with guards, the ordering matrix, the outcome surprise, the twin population, the `dR ~ V_pre` slopes (all pairs, test split, token-balanced), bins, `k*` strata and residualised slopes, matched persistence, and scope by `k*`; `figs/norm_{calib,twins,scope}_*.png`. Reads only the fetched `.json`/`.npz`. |

## Children

| folder | what |
|---|---|
| [`../../orbitofrontal/adaptation/`](../../orbitofrontal/adaptation/FILES.md) (moved 2026-09-17 to the `orbitofrontal/` super-node) | The **within-subject** version of this node's question (2026-09-17). A diet stops being one accumulation over its mask and becomes a row *order* over it: a critic is adapted to the low tercile and then switched to the middle one, another comes from the high tercile to the same middle one, and both are read on the same fixed rows at the same number of windows after the switch, under five memories (two flat sliding windows, two forgetting factors, and a cumulative fit that cannot adapt). No `README.md` yet — the facts are in `adaptation/results/tables.md`; interpretation waits on a discussion. |
| [`../../orbitofrontal/projection/`](../../orbitofrontal/projection/FILES.md) (moved 2026-09-17 to the `orbitofrontal/` super-node) | **Which object the projection reads** (2026-09-17): the model's *belief* (its output distribution, public) or its *representation* (the residual stream). Two parts on this node's machinery verbatim — one linear map from the twins' post-token **forecast difference** to the critic's revision difference `dR`, read on held-out pairs against the state difference as the exact ceiling; and the critic **refit on the model's output distribution** at the query position in place of the residual stream, with `norm/`'s headline objects read on it beside the state critic. No `README.md` yet — the facts are in `projection/results/tables.md`; interpretation waits on a discussion. |

## Artefacts

| path | what |
|---|---|
| `results/tables.md` | Every number, facts only, for the six cells at both anchors, with the reproduction gate at the top of each venue. |
| volume `.../logit_reading/traj_a1_s42/step{000000,008000,064000}_norm_{a1,swap65k}.{json,npz}` | Per-cell config, diet compositions, critic fits, the per-episode arrays and the twin arrays the reduction regroups. |
