# Files — logit_reading/striatum/junction

**Up**: [README.md](README.md) · **Parent**: [../FILES.md](../FILES.md)

Imports `striatum/task.py`'s primitives (the actor, the streamed Gram, the closed-form ridge, the matcher)
unchanged; `striatum/task.py` itself is untouched.

## Code files

| file | purpose |
|---|---|
| `diets.py` | The diet definitions and their construction. `cons_depth(etype, j)` (the deterministic consequence depth); `window_diets` (the `c`, `x` and `r` families over `(etype, j)` cells, plus `full` and `natural_sized`, as boolean masks over training windows with the `none` share fixed); `damage_diets` (the `r` family, cut on realised damage); `row_diets` (Q2's row selections: `gate_top`, `gate_bot`, `rand`, `randpos`, `oracle`, at each fraction of the training rows). |
| `task.py` | The node. `junction_ckpt` (L4): one pair of forward passes per cell, one frozen actor, one clean-only critic, then a critic per diet from masked rows of the shared Gram; the fixed split; the matched contrasts, the positive control, the oracle legality and consequence probes, the banked head and the realised horizon excess on the identical rows; writes `stepNNNNNN_junction_<tag>.{json,npz}`. `junction_sweep` (CPU coordinator) fans the cells out. Own Modal app and image with the `results/` and `.log` `ignore`. |
| `analyze.py` | Local reduction: the headline and positive-control tables, the matched / unmatched / surprisal-matched contrasts, the damage and consequence tables, the Q2 containment headline and ladders, per cell and per anchor; `figs/junction_*.png` and `figs/gate_*.png`. Reads only the fetched `.json`/`.npz`. |

## Artefacts

| path | what |
|---|---|
| `results/tables.md` | Every number, facts only, for the three cells at both anchors, with the reproduction gate at the top. |
| volume `.../logit_reading/traj_a1_s42/step{064000,008000}_junction_{a1,swap65k}.{json,npz}` | Per-cell config, diets' composition, critic fits, and the per-episode arrays the reduction regroups. |
| volume `.../traj_a1_s42/step008000_striatum_a1_hexcess.npz` | The realised horizon excess column at 8k, built here with `striatum/addendum.py`'s recipe. |
