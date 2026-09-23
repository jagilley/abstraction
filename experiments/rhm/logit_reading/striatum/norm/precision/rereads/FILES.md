# Files — logit_reading/striatum/norm/precision/rereads

**Parent**: [../FILES.md](../FILES.md) (precision) · **Tables**: [calibration/results/tables_20260922.md](calibration/results/tables_20260922.md) ·
clock/results/tables_20260922.md[^omitted] · [public/results/tables_20260922.md](public/results/tables_20260922.md)

No `README.md`: these are facts only until discussed.

**What this folder is.** `precision/`'s expressivity rounds (`../results/tables_express_20260922.md`, `tables_express_ln_20260922.md`)
found that the arc's instrument, the closed-form ridge on the raw residual stream, and readers with more of the belief (the sixteen
log-probabilities appended, `fullQ`; a 128-unit MLP on the state, `fullM`) disagree on sign-level claims (the response's dependence on
the pre-event entropy and its surprise-by-entropy interaction at ℓ=1) and on how much damage information the reader contains, while
agreeing on the norm's level along the entropy axis and on the legality law. The three banked **level-level** claims of the value side
had been made with the ridge only. Each folder here re-reads one of them on the richer readers, on identical rows, gated against the
banked cells, three trajectory seeds where the banked claim had them (two order seeds for the clock), seeds side by side and never
averaged. The banked write-ups are `striatum/norm/README.md` §1 (calibration), `orbitofrontal/README.md` §1 and
`orbitofrontal/adaptation/` (the clock), and `orbitofrontal/README.md` §4 and `orbitofrontal/projection/` (the public–private split).

## calibration/ — the norm's calibration (2026-09-22)

| file | purpose |
|---|---|
| `calibration.py` | Modal, L4, 5.5–8 min per cell. Rebuilds `norm/task.py`'s trunk, split, frozen actor, outcome labels and the eleven `a1` window diets exactly (via `express.py`'s gated copy; diets gated against the banked `diet_stats`). Per diet it fits: arm 0 = the ridge in `norm/task.py`'s own order (the gate); `Q` (+ log q) and `P` (+ precision) as sub-blocks of the Gram augmented with `express._aug_cols`, the appended columns standardised on that diet's own rows; `L` (`ln_f(s)` in place of the state); and `M`, a 128-unit MLP, on every diet at levels 1–4, a = 0. Same lambda ladder and shared held-out selection. Writes `Vpre_<diet><arm>_l<l>_a{0,4,8}` plus `V_` / `R_` at a = 0 on the `split == 2` rows of both anchors. Gates: actor, diets, arm-0 val `R2` for every diet, rows and labels, arm-0 `V/Vpre/R/Ro/Vpreo` for every diet, and the `full` diet's Q/P/L against `express.py` / `express_ln` (asserted) and M (recorded). Output `<stem>_norm_a1_calib.{json,npz}`. `calib_sweep` is the CPU coordinator, one container per checkpoint. |
| `reduce_calibration.py` | CPU, ~45 s. Views each reader's columns in place of the diets' own and runs `norm/analyze.py`'s `sec_norm_level` / `sec_norm_shape` / `sec_delta` **unchanged**. Per-family orderings and the entropy slopes (`precision/reduce.py`'s `_partial` / `Cell`) are checked against the banked `sec_ordering` / `sec_level_family` on the ridge. Discrimination uses junction's matched rows and `discrim.py`'s `label_of`. Sections: 0 integrity, gates and reproduction of the README numbers; 1 fit; 2 ordering and level; 3 shift or rescaling; 4 outcome surprise at a = 0/4/8; 5 within-cell and pooled damage AUC at step 0 vs 64k; 6 per-diet entropy slope. |

| artefact | what |
|---|---|
| `calibration/results/tables_20260922.md` | Facts only: 5 readers × 11 diets × 3 seeds × 3 checkpoints × 2 anchors, seeds side by side. |
| `calibration/figs/calib_{level,slope,disc,entropy}_<anchor>.png` | Mean `V_pre` per diet against E[outcome] per reader and level; the `out` diets' slope on the reader's own `full` at step 0 and 64k; within-cell AUC(−V_pre, damage) per reader across checkpoints; the per-diet entropy slope against E[outcome] with the realised outcome's dotted. Gitignored. |
| volume `.../traj_a1_s4{2,3,4}/step{000000,008000,064000}_norm_a1_calib.{json,npz}` | The per-diet refits, with the gates in the json. `step064000_norm_a1_calib_smoke.*` on s42 is the smoke's leftover. |

## clock/ — the norm's clock (2026-09-22)

| file | purpose |
|---|---|
| `clock.py` | Modal, L4, 126–310 s per cell. `orbitofrontal/adaptation/task.py::adapt_ckpt` forked with one addition: a feature-map selector `--fmaps` (arm 0 the state on the banked code path = the gate; `Q` state + log q; `L` `ln_f(state)` in place; `H` state + H(q)). Appended columns are standardised by `express.py::_standardiser` on each fit's own Gram (fgt: weighted, `n_rows = A[-1,-1]`); the static critics, the five fitters' block Grams, the per-(fitter, target) penalty selection and `d_next` are carried per arm. Arm 0 is compared member by member against the banked `.npz` / `.json` in-container and not re-saved. Writes `<stem>_adapt_<tag>_clock.json`, `_clock{Q,L,H}.npz` (banked key names) and `_clockX.npz` (log q / H(q) at every saved anchor and twin state, `ln_f` weights). `clock_sweep` is the CPU coordinator, one container per cell. |
| `reduce_clock.py` | CPU, ~4 min, 4.2 GB. Every reader through `adaptation/analyze.py`'s own `Cell` / `dev_from_mech` / `norm_gap` / `_mech` / `_cross` / `resp_gap` / `resp_readouts` / `dnext_z` / `TwinsX` (feature-map arms as `Cell` subclasses; readouts vectorised over checkpoints behind a bounded LRU). Integrity, gates, feature-assembly checks, reproduction of banked §9 counts, then per venue: static fits, the from-low minus from-high gap with the mechanical null, deviation-sign tables printing every cell, half-times against the null, the cumulative fit, the lockstep counted under three explicit definitions, `d_next` with controls, the response's history (all rows, matched rows, raw units), damage AUC, twins; a per-checkpoint closure appendix. |

| artefact | what |
|---|---|
| `clock/results/tables_20260922.md` | Facts only; four readers × two order seeds side by side, never averaged. Notes that `orbitofrontal/README.md` §1's 200-window entry reads −0.004 in the banked files. |
| `clock/figs/clock_{gap,dev,dnext,resp,twins}_<venue>.png`, `clock_norm_<reader>_<venue>.png` | Gap against the null, deviation scatter across orderings, `d_next` z, response gap, twins, per-reader norm course. Gitignored. |
| volume `.../traj_a1_s42/step064000_adapt_{a1,swap65k}[_ord2]_clock{.json,Q.npz,L.npz,H.npz,X.npz}`, `..._adapt_a1_smoke_clock*` | The refits. |

## public/ — the public–private split (2026-09-22)

| file | purpose |
|---|---|
| `public.py` | Modal, L4, 130–440 s per cell. Forks `express.py` by import (trunk, split, actors, `_aug_cols`, `_standardiser`, `_fit_arm`, lambda ladder unchanged) and `norm/`'s diets (same seeds). Refits arm 0, `fullQ` and `fullP` and saves their coefficients (`beta__*`). Writes each per-row value split exactly into a state part (intercept and the appended block's training mean) and an appended part at both anchors (critics `fullQst` / `fullQbl` / `fullPst` / `fullPbl`). Fits the log q ridge per diet (`lq_<diet>`, projection's `logq` block), a 128-unit MLP on log q per diet (`mq_<diet>`, `fullM`'s recipe and seeds) and the MLP on [state, log q] (`fullQM`); retrains `fullM` as a recipe guard. Computes the readout span of each state block with its cosine to the ridge. On `norm/`'s banked twin pairs: every reader's V at offsets 0 / −1, the forecasts log q at t_v+0..4 and dX at t_v (for pairs with |ds| ≤ 0.35). Gates against the banked json and the `_express` npz are asserted. `public_sweep` is the CPU coordinator (`skip_done`); `basis_sweep` (CPU) writes `<stem>_readout_basis.npz`. |
| `reduce_public.py` | CPU, ~10 min for 3 seeds. G: integrity, cell gates, matched-set identity, exact-split identity, log q ridge / span / basis / priced map against projection's s42 cells, multi- against single-target fit. 1: amplitude, variance shares, damage AUC, `H_pre` slope with and without `V_pre`, z(s)·z(H), E4 per part, for `fullQ` and `fullP`. 2: fit, `a1` calibration (per-diet `V_pre`, family ρ, shift-versus-rescaling) and the response per reader, with projection's rand16 / pca16 (s42). 4: twin pairs, token-balanced `dR` per reader, the three-way split of `dR`, and the priced map on every reader (multi-target equivalent of projection's `ridge_cv`). 3: span. |
| `results/fetch.sh` | Sequential fetch of the public cells, the basis files and (s42) projection's cells (parallel `modal volume get` dropped files). |
| `results/wait_app.sh` | Waits on the remote app's state, not on file counts: Modal volumes commit in the background, so a cell's json can be visible before its last file is written. |

| artefact | what |
|---|---|
| `public/results/tables_20260922.md` | Every number, facts only, 3 seeds × 2 venues × 2 anchors × 3 checkpoints, gates first. |
| `public/figs/pub_{decomp_<venue>_<anchor>,priced_<venue>,span,calib_a1}.png` | The part split (slopes and AUC), priced `R2` per reader, span by step, per-diet norm. Gitignored. |
| volume `.../traj_a1_s4{2,3,4}/step{000000,008000,064000}_norm_{a1,swap65k}_public.{json,npz}`, `_public_twins.npz` | Cells of record. The npz member `basis__Vb` is NOT the basis; use `_readout_basis.npz`. |
| volume `.../traj_a1_s4{2,3,4}/step{000000,008000,064000}_readout_basis.npz` | `Vb` (15 × 256) per checkpoint. |

## Reproduction

All from `experiments/` with `MODAL_PROFILE=chromatic`; the fetch loops use the full destination filename (`modal volume get` refuses a
directory that exists). If a sibling's launch log is streaming in the tree, stage `rhm/**/*.py` to the scratchpad and run `modal` from there.

```bash
# calibration
modal run --detach -m rhm.logit_reading.striatum.norm.precision.rereads.calibration.calibration::calib_sweep
D=/v16_s2_L6_m4_distinct/logit_reading
for s in 42 43 44; do for st in 000000 008000 064000; do
  for f in norm_a1.json norm_a1.npz norm_a1_express.npz norm_a1_express_ln.npz norm_a1_calib.json norm_a1_calib.npz; do
    modal volume get rhm-scaling-data $D/traj_a1_s$s/step${st}_$f <dir>/traj_a1_s$s/step${st}_$f
done; done; done
python -m rhm.logit_reading.striatum.norm.precision.rereads.calibration.reduce_calibration \
    <dir>/traj_a1_s42 <dir>/traj_a1_s43 <dir>/traj_a1_s44 \
    --out rhm/logit_reading/striatum/norm/precision/rereads/calibration/results \
    --figs rhm/logit_reading/striatum/norm/precision/rereads/calibration/figs --date 20260922

# clock (smoke = the banked adaptation smoke cell, bit-identical on arm 0)
modal run -m rhm.logit_reading.striatum.norm.precision.rereads.clock.clock::clock_ckpt \
    --ckpt /data/v16_s2_L6_m4_distinct/logit_reading/traj_a1_s42/step064000.pt --stim-tag a1 \
    --max-windows 4000 --n-clean 1024 --actor-steps 300 --max-twins 1500 --block-win 10 --tag smoke
modal run --detach -m rhm.logit_reading.striatum.norm.precision.rereads.clock.clock::clock_sweep
for t in a1 swap65k a1_ord2 swap65k_ord2; do
  for f in .json .npz _clock.json _clockQ.npz _clockL.npz _clockH.npz _clockX.npz; do
    modal volume get rhm-scaling-data $D/traj_a1_s42/step064000_adapt_$t$f <dir>/traj_a1_s42/step064000_adapt_$t$f
done; done
python -m rhm.logit_reading.striatum.norm.precision.rereads.clock.reduce_clock <dir>/traj_a1_s42 --date 20260922

# public (smoke = the s42 64k a1 cell of record)
modal run -m rhm.logit_reading.striatum.norm.precision.rereads.public.public::public_ckpt \
    --ckpt /data/v16_s2_L6_m4_distinct/logit_reading/traj_a1_s42/step064000.pt --stim-tag a1
modal run --detach -m rhm.logit_reading.striatum.norm.precision.rereads.public.public::public_sweep
modal run -m rhm.logit_reading.striatum.norm.precision.rereads.public.public::basis_sweep
bash rhm/logit_reading/striatum/norm/precision/rereads/public/results/fetch.sh <pubmirror>
python -m rhm.logit_reading.striatum.norm.precision.rereads.public.reduce_public \
    --bank <mirror> --pub <pubmirror> --seeds 42,43,44 \
    --out rhm/logit_reading/striatum/norm/precision/rereads/public/results \
    --figs rhm/logit_reading/striatum/norm/precision/rereads/public/figs --date 20260922
```

`<mirror>` holds the banked `norm` and `_express` cells; `<pubmirror>` the public cells.

## Gotchas worth not rediscovering

- **The per-block split of `fullQ` is exact but collinear.** The state part and the belief part have 3–5× the variance of their sum
  and correlate at −0.8 to −0.9, so a part-level slope or AUC is weakly identified on its own (the step-0 split flips sign across
  seeds). Read part-level numbers beside their sum, never alone.
- **The lockstep sentence has no code behind it.** `orbitofrontal/README.md` §1's "the shift and the rescaling close in lockstep,
  within 0.05" is not a reduction in `adaptation/analyze.py`; under three explicit definitions it holds in 12–22 of 80 cells per seed
  for every reader, the ridge included. Near a level of 0.9 a change in slope and a change in intercept are hard to separate.
- **The reduction's per-checkpoint cache must stay bounded**; unbounded, the clock reduction ran out of memory at 12 GB on `swap65k`.
- **`swap65k_ord2`'s arm 0 reproduces to 1e-9 relative, not bit for bit** (GPU float64 summation order); the other three clock cells are
  bit-identical.
- **Wait on the app's state, not on file counts.** Modal volumes commit in the background: a cell's json can be visible while its last
  npz is still being written, and a coordinator that restarts re-runs with `skip_done` and idles.
- **Parallel `modal volume get` dropped files**; fetch sequentially.

[^omitted]: Omitted from the mirror for size (a raw results dump over the per-file cap). Available on request.
