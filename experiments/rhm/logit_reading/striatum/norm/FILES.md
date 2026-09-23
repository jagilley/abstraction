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
| `discrim.py` | Local CPU reduction, added 2026-09-19: the held-out **AUC** of the critic's pre-event level `V_pre`, of the post-event level `V` and of the revision `R` against the realised outcome (`oe`), the realised damage (junction's `flip`) and the grammar's own consequence label (`cons`), per diet, per level, at steps 0 / 8k / 64k, with the **trained-minus-random difference on identical rows** and the exact shuffled-label floor. Reads only the fetched `.npz` (which holds the `split == 2` test rows), refits nothing, touches no GPU; gated by recomputing `tables.md`'s matched-row `AUC(-R, flip)` / `AUC(-V, flip)` columns with `junction/analyze.py`'s matcher. |
| `analyze.py` | Local reduction: the reproduction gate, the diets' worlds and the cell-histogram check, the norm's level and its regression on `full`, the response contrasts with guards, the ordering matrix, the outcome surprise, the twin population, the `dR ~ V_pre` slopes (all pairs, test split, token-balanced), bins, `k*` strata and residualised slopes, matched persistence, and scope by `k*`; `figs/norm_{calib,twins,scope}_*.png`. Reads only the fetched `.json`/`.npz`. |

## Children

| folder | what |
|---|---|
| [`../../orbitofrontal/adaptation/`](../../orbitofrontal/adaptation/FILES.md) (moved 2026-09-17 to the `orbitofrontal/` super-node) | The **within-subject** version of this node's question (2026-09-17). A diet stops being one accumulation over its mask and becomes a row *order* over it: a critic is adapted to the low tercile and then switched to the middle one, another comes from the high tercile to the same middle one, and both are read on the same fixed rows at the same number of windows after the switch, under five memories (two flat sliding windows, two forgetting factors, and a cumulative fit that cannot adapt). No `README.md` yet — the facts are in `adaptation/results/tables.md`; interpretation waits on a discussion. |
| [`../../orbitofrontal/projection/`](../../orbitofrontal/projection/FILES.md) (moved 2026-09-17 to the `orbitofrontal/` super-node) | **Which object the projection reads** (2026-09-17): the model's *belief* (its output distribution, public) or its *representation* (the residual stream). Two parts on this node's machinery verbatim — one linear map from the twins' post-token **forecast difference** to the critic's revision difference `dR`, read on held-out pairs against the state difference as the exact ceiling; and the critic **refit on the model's output distribution** at the query position in place of the residual stream, with `norm/`'s headline objects read on it beside the state critic. No `README.md` yet — the facts are in `projection/results/tables.md`; interpretation waits on a discussion. |
| [`precision/`](precision/FILES.md) | **The value reader against the world model's pre-event uncertainty** (2026-09-22). Every value-side contrast in the arc is taken at matched surprisal; none has asked whether the reader depends on `H_pre = H(q_(t0-1))`, the entropy of the trunk's own forecast at the position before the anchor, which `task.py` records on every fixed row and no banked table had used. The norm, the response (inside `junction`'s own matched sets, and on the 2-D `surprisal x entropy` surface that is the direct analogue of the music literature's pleasure surface), the outcome surprise and the same-prefix twins, all against that axis, three trajectory seeds, both venues, both anchors, against the random-init trunk on identical rows. CPU reduction on the banked cells plus one L4 forward pass per cell (`hpre.py`) for the twins' own column, which the bank does not carry. Then (same day) the mediators of that dependence, the glitch world on the axis, the reader's form (the entropy, the belief, placebos, a precision-weighted form, an MLP, layer norm), the legality law on a 65k venue with rare edits, and the three banked level-level claims re-read on the richer readers (`precision/rereads/`). Written up in [`precision/README.md`](precision/README.md). |

## Artefacts

| path | what |
|---|---|
| `results/tables.md` | Every number, facts only, for the six cells at both anchors, with the reproduction gate at the top of each venue. |
| `results/tables_discrim_20260919.md` | `discrim.py`'s output: the **discrimination** table the banked files do not carry — held-out AUC of `V_pre` / `V` / `R` against the realised outcome, the realised damage and the consequence label, per diet and level, at steps 0 / 8k / 64k on identical rows, with the trained-minus-random difference, on `traj_a1_s4{2,3,4}` × `a1`, `swap65k` × anchors `fd`, `tv`. Section 0 puts the three seeds side by side; nothing is averaged across seeds. Facts only. |
| volume `.../logit_reading/traj_a1_s42/step{000000,008000,064000}_norm_{a1,swap65k}.{json,npz}` | Per-cell config, diet compositions, critic fits, the per-episode arrays and the twin arrays the reduction regroups. |
