# Files — logit_reading/orbitofrontal/adaptation

**Up**: [../README.md](../README.md) (orbitofrontal, the writeup) · **Donor**: [../../striatum/norm/README.md](../../striatum/norm/README.md) (norm; its [FILES.md](../../striatum/norm/FILES.md)) ·
**Tables**: results/tables.md[^omitted]

The facts are in `results/tables.md` and the figures; the interpretation is the super-node's
([`../README.md`](../README.md) §1 and §5), and [`README.md`](README.md) here is a pointer.

## What this node is

`norm/` is **between-subjects**: three critics raised in three worlds that differ only in expected outcome, read
on identical held-out rows. This node is **within-subject**, which is Xiang, Lohrenz & Montague's own design
(`reading/emotion_in_cortex.md`[^private] §1): one critic is adapted to
the low tercile and then switched to the middle one, another comes from the high tercile to the same middle one,
and the two are read on the same fixed rows at the same number of windows after the switch. The machinery change
is small and exact: a diet stops being one accumulation over a boolean mask over `(training window, Gram row)`
and becomes a row **order** over that mask, with the critic refit along the sequence.

Imports `striatum/task.py`'s `train_actor` / `actor_apply` / `ridge_solve` (which is `coeruleus/readout.py`'s
solve), `junction/diets.py`'s `cons_depth`, `norm/diets.py`'s `strat_diets` / `anchor_diets` / `cell_key`,
`striatum/analyze.py`'s `_auc` / `fmt` / `tbl`, `junction/analyze.py`'s `Rows` and `phasic.py`'s
`balance_tokens`, all unchanged. The split seed, the actor recipe, the diet seeds and the per-episode column
names are `norm/task.py`'s verbatim, so `Rows` reads this node's npz without modification and the static critics
refit here reproduce `norm/`'s banked ones.

## Code files

| file | purpose |
|---|---|
| `task.py` | The node. `adapt_ckpt` (L4): `norm/`'s two forward passes, frozen actor, shared row cache and tercile diets; then (a) the **static** reference critics (`norm/`'s banked closed-form fits, refit here as the reproduction gate and as the endpoints an online critic should approach), (b) the **online** critics — eight arms × five fitters × K checkpoints, each fitter a closed-form ridge on a different memory of the sequence — and (c) the fixed-row anchor states and the same-prefix twin states, saved raw so any critic can be applied to them in the reduction. Writes `stepNNNNNN_adapt_<tag>.{json,npz}`. `adapt_sweep` (CPU coordinator) fans the cells out, one container each. Own Modal app and image with the `results/` and `.log` `ignore`. |
| `analyze.py` | Local reduction: the reproduction gate against `norm/README.md` §1 and the banked actor column; the arms' worlds and the cell-histogram check; the norm's time course per fitter and level; the gap against the `mid2mid` control and the fraction of it closed against the fitter's own memory; the shift-versus-rescaling regression on the banked `full` critic at every checkpoint; the outcome surprise in three currencies (fixed violation rows, fixed validation rows overall and per world, and the rows arriving next); the matched response at the event with every guard; and the token-balanced twin contrast along the sequence. Figures under `figs/`. Reads only the fetched `.json` / `.npz`. |

## The design, in one place

**Arms** (a switch arm is phase A then phase B, each exactly one phase length of windows):

| arm | phase A | phase B | what it is |
|---|---|---|---|
| `out_lo2mid` | low-outcome tercile | middle tercile | Xiang's low-adapted group |
| `out_hi2mid` | high-outcome tercile | middle tercile | Xiang's high-adapted group |
| `out_mid2mid` | middle tercile | middle tercile (independent permutation) | the control whose world does not change |
| `out_lo_mix` / `out_hi_mix` | — | — | the same windows as the treatment arm, interleaved at random: the sequence removed |
| `dmg_lo2mid` / `dmg_hi2mid` / `dmg_mid2mid` | realised-damage terciles | | the second instance of the same contrast |

Within a family the three tercile masks pin the `(etype, j)` cell histogram bit-identically (`norm/diets.py`), so
the two phases of a switch arm differ only in the outcome. The trim that makes a phase a whole number of blocks
is allocated per `(etype, j)` cell, identically for every diet, so the histogram survives it; the json prints the
check.

**Fitters** (all exact, all closed form, all through `ridge_solve`):

| fitter | what it holds |
|---|---|
| `win200`, `win800` | a flat sliding window of the last N windows of experience; N is the memory |
| `fgt200`, `fgt800` | exponentially weighted least squares, per-row forgetting factor `exp(-1/(M·R))`, effective memory M windows — what recursive least squares with a forgetting factor converges on at every step, in closed form instead of by rank-one updates |
| `cum` | every row seen so far, no forgetting: the reference that cannot adapt |

All three come from block Grams. The sequence is chopped into blocks of `block_win` windows; each block
contributes `X_b'X_b` (for `win` / `cum`) and `(X_b·w)'X_b` with `w_i = lam^(G-1-i)` (for `fgt`). A flat window
is then a difference of two cumulative snapshots and a forgetting fit is `lam^G · A(t-G) + W_b`; both are exact
at block resolution, and the pass is matmuls rather than a Python loop over rows. The ridge penalty is selected
once per (fitter, target) on the fixed validation rows from the `out_mid2mid` control at the end of phase A and
then held fixed for every arm and every checkpoint, so no contrast is a penalty artefact.

## Order seeds

Two, run as separate cells: `order_seed = 909` (the cells with no tag) and `order_seed = 1313`
(the `_ord2` cells). The seed sets the order of the windows within each phase **and** which windows the
per-cell trim keeps, so the two differ in the sequence and in its membership; the checkpoint, venue, split,
cell histograms, held-out rows, fitters and ridge penalties are identical, and both are read on the same fixed
rows. `analyze.py` picks up the second seed automatically (`--ord2`, default `ord2`), puts its columns beside
the first's in the norm-course, lag, gap-closing and response tables, draws it dashed in the figures, and writes
`### 9. Across order seeds`, which states each of three claims — the deviation of the norm's re-calibration from
the fitter's own memory, the response carrying the running history, and the world-change excursion in `d` on the
rows arriving next — as a sign that either agrees across the seeds or does not, with the `mid2mid` and
interleaved controls tabulated beside the last of them.

## Artefacts

| path | what |
|---|---|
| `results/tables.md` | Every number, facts only, for both venues, with the reproduction gate at the top of each. |
| `results/wait_app.sh` | Waits on the remote app rather than the local launch log. |
| `figs/adapt_{norm,fitters,shape,delta,event}_<tag>_<step>.png` | The time courses (regenerated by `analyze.py`; PNGs are gitignored under `experiments/`). |
| volume `.../logit_reading/traj_a1_s42/step064000_adapt_{a1,swap65k}.{json,npz}` | Per-cell config, diet and arm compositions, the static critics' fits, the online betas per (arm, fitter, checkpoint) with their per-checkpoint diagnostics, and the fixed-row and twin **states** the reduction applies them to. |

## Reproduction

```bash
cd experiments            # MODAL_PROFILE=chromatic
# two cells (a1 and swap65k at the clean trajectory's 64k checkpoint), one L4 container
# each, fanned out by a CPU coordinator
modal run --detach -m rhm.logit_reading.orbitofrontal.adaptation.task::adapt_sweep --block-win 25
# tables and figures (local; needs the .json and .npz pulled off the volume)
python -m rhm.logit_reading.orbitofrontal.adaptation.analyze <dir>/traj_a1_s42 \
    --tags a1,swap65k --out rhm/logit_reading/orbitofrontal/adaptation/results \
    --figs rhm/logit_reading/orbitofrontal/adaptation/figs
```

## Gotchas worth not rediscovering

- **The cell histogram does not survive a naive trim.** Cutting each diet's window list to a multiple of the
  block size at random breaks the bit-identical `(etype, j)` match that is the whole point of the tercile diets.
  Allocate the trim per cell, the same allocation for every diet.
- **A sliding window's lag is partly its own memory.** `win<N>` has replaced exactly `min(o/N, 1)` of its rows
  `o` windows after the switch, so the time to re-calibrate cannot be read as a property of the norm on its own;
  the tables print the mechanical fraction beside the measured one, and `fgt` (no hard cutoff) and `cum` (no
  forgetting) are there for the comparison.
- **`d` on the fixed held-out rows is not zero in steady state.** Those rows come from the whole pool, not from
  the phase the critic is in. The per-world split of the validation residual (`d_val_lo/mid/hi`) is the reading
  that goes to zero when the critic is calibrated to that world; `d_next`, on the rows actually arriving, is the
  running one but is noisy at one block.
- **An npz member is decompressed on every access.** The reduction applies several hundred betas to the same
  state arrays; cache them once or the reduction takes minutes instead of seconds.
- **Separate cells with `;`, not `,`.** The coordinator's per-cell anchor list is itself comma separated.
- **A sibling's streaming launch log kills the image build, and the `ignore` does not save you.** The build
  refuses with "`<path>` was modified during build process" for a file that `_ignore` already excludes, so
  retrying while the sibling's run is live fails every time (3/3 here). The fix that works: copy the `.py`
  files of `rhm/` into the scratchpad and run `modal run` from there, so nothing outside your own tree can
  change under the build.
- **`modal volume get` can truncate or corrupt a large `.npz` silently.** One 62 MB fetch came back short and
  another came back full size with a bad CRC on a single member. `np.load(f)` and `len(z.files)` both pass on
  the second — they only read the central directory — so verify by touching every member before reducing.

[^private]: Not mirrored: this link points to a document in the private lab repo (the roadmap, the queue, an unrun spec, reading notes, or a conversation). See the top-level README for what is held back and why.

[^omitted]: Omitted from the mirror for size (a raw results dump over the per-file cap). Available on request.
