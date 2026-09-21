# Files — logit_reading/orbitofrontal/projection

**Up**: [../README.md](../README.md) (orbitofrontal, the writeup) · **Donor**: [../../striatum/norm/README.md](../../striatum/norm/README.md) (norm; its [FILES.md](../../striatum/norm/FILES.md)) ·
**Tables**: [results/tables.md](results/tables.md)

The facts are in `results/tables.md` and the figures; the interpretation is the super-node's
([`../README.md`](../README.md) §4), and [`README.md`](README.md) here is a pointer.

## What this node is

`norm/` established that the outcome-trained critic's pre-event level is a learned expectation over
outcomes that calibrates to the world the critic was fed, and the reading settled in discussion is
that the norm is the world model read through a projection whose weights the goal's history sets.
This node asks **which object that projection reads**: the model's *belief* (its output distribution,
which is public) or its *representation* (the residual stream, which is not). Two parts, both on
`norm/`'s machinery, split seed, diets and twin construction verbatim:

1. **The priced twin.** On the same-prefix pairs, one linear map from the **forecast difference**
   (violator's post-token forecast minus its legal twin's, at `t_v + 1` and four offsets beyond,
   each teacher-forced on that window's own continuation and causal at every offset) to the critic's
   revision difference `dR`, fitted on training pairs and read on held-out pairs; the same map from
   the **state difference** beside it as the ceiling. Both worlds (`norm/` §5's edit and glitch).
2. **The public critic.** The critic refit with the model's output distribution at the query position
   as the feature block in place of the residual stream — same targets, same diets, same held-out
   rows, same ridge — and `norm/`'s headline objects read on it beside the state critic.

Imports `striatum/task.py`'s `train_actor` / `actor_apply` / `ridge_solve`, `junction/diets.py`'s
`cons_depth`, `norm/diets.py`'s `strat_diets` / `mix_diets` / `anchor_diets` / `cell_key`,
`striatum/analyze.py`'s `_auc` / `fmt` / `tbl`, `junction/analyze.py`'s `Rows` and `phasic.py`'s
`balance_tokens`, all unchanged. The split seed, the actor recipe, the diet seeds and the anchor
recipe are `norm/task.py`'s verbatim, so the `state` block reproduces `norm/`'s banked critics.

## Feature blocks

Every block is a set of features at one position, and every one is causal — nothing below ever reads
a future position's forecast.

| block | dim | what |
|---|---|---|
| `state` | 256 | the frozen trunk's `post_block7` residual stream: `norm/`'s critic of record |
| `logq` | 16 | `log q(· \| w_≤p)`, the model's published belief about the next token — **the public critic** |
| `logq_rich` | 33 | `[log q, q, H(q)]`: the same belief, reparameterised (the guard against reading the gap as a link-function artefact) |
| `logq_tok` | 32 | `[log q, onehot(x_p)]`: the tokens-plus-logits observer at one position |
| `pub_hist` | 96 | `[log q at p…p−3, onehot(x_p), onehot(x_{p−1})]`: the same observer over a short public window |
| `rand16` | 16 | a fixed random 16-dim linear projection of the state — the dimension-matched control for `logq` |
| `pca16` | 16 | the top-16 principal directions of the state on `full`'s own rows — the best 16-dim linear compression |

`rand16` and `pca16` are derived exactly from the state Gram (`A_P = P A Pᵀ`, `C_P = P C`), so they
cost no second pass and no second forward.

**The rollout block was priced and declined.** A rollout-averaged forecast over the horizon would
need, per Gram row, `S` sampled continuations of `H` steps, each step a forward pass over the window:
at `full`'s 432,520 rows with `S = 4`, `H = 8` that is ~1.4 × 10⁷ window-forwards per cell against
the ~3 × 10⁵ the whole cell currently runs, i.e. three to four orders of magnitude more GPU time than
the rest of the node put together. `pub_hist` is the cheap substitute actually run: it widens the
public observer along the *past*, which costs nothing, rather than along the future.

## The readout span

`logits = W_U diag(γ)(x − mean(x))/σ(x)`, and the softmax kills the constant direction in logit
space, so the belief exposes **exactly** the 15-dimensional projection `M x` of the state, up to a
per-position positive scale. `readout_span` in each cell's json is `‖P_M β‖/‖β‖` for every state
critic column: the fraction of the critic's own direction the output layer can express at all. A
random direction gives `sqrt(15/256) = 0.242`. The basis is saved in the twins npz as `basis__Vb`,
which is what lets the reduction split `dR` exactly into its readable and unreadable parts.

## Code files

| file | purpose |
|---|---|
| `task.py` | The node. `proj_ckpt` (L4): `norm/`'s two forward passes, frozen actor, shared row cache and diets; a Gram and a closed-form ridge per (feature block, diet); the norm, response and outcome surprise on the fixed rows for every block; the readout span; and the same-prefix twins in both worlds with, per pair, every block's `V` along the trace, the raw post-token forecasts at offsets 0–4 and the state difference at `t_v`. Writes `stepNNNNNN_proj_<tag>.json`, `_rows_{tv,fd}.npz` and `_twins.npz`. `proj_sweep` (CPU coordinator) fans the cells out, one container each. Own Modal app and image with the `results/` and `.log` `ignore`. |
| `analyze.py` | Local reduction: the reproduction gate against `norm/README.md` §1 and the banked actor column; the per-block held-out fits; the readout span; the norm per world per block with its Spearman rank order; the shift-versus-rescaling regression; the matched response; the outcome surprise; the twin `dR` and matched persistence per block in both worlds; the priced-twin maps with their ceiling, null and cross-world transfer; and the exact split of `dR` into what the logits express and what they do not. Figures under `figs/`. Reads only the fetched `.json` / `.npz`. |

## Artefacts

| path | what |
|---|---|
| `results/tables.md` | Every number, facts only, for the four cells, with the reproduction gate at the top of each. |
| `figs/proj_{fit,calib,twins,priced}_<tag>_<step>.png` | Regenerated by `analyze.py`; PNGs are gitignored under `experiments/`. |
| volume `.../logit_reading/traj_a1_s42/step{000000,008000,064000}_proj_{a1,swap65k}.{json,_rows_tv.npz,_rows_fd.npz,_twins.npz}` | Per-cell config, diet compositions, per-block critic fits, the readout span, the per-episode readouts at both anchors, and the twin arrays the reduction regroups. |

## Reproduction

```bash
cd experiments            # MODAL_PROFILE=chromatic
# six cells (steps 0 / 8k / 64k x a1 / swap65k), one L4 container each, fanned out by a CPU
# coordinator (~11 GPU-min; 73-173 s per cell, peak RSS 5.9-7.8 GB against a 12 GB request)
D=/data/v16_s2_L6_m4_distinct/logit_reading
modal run --detach -m rhm.logit_reading.orbitofrontal.projection.task::proj_sweep \
    --cells "$D/traj_a1_s42/step064000.pt:a1:1:0,$D/traj_a1_s42/step064000.pt:swap65k:0:1,\
$D/traj_a1_s42/step008000.pt:a1:1:0,$D/traj_a1_s42/step008000.pt:swap65k:0:1"
modal run --detach -m rhm.logit_reading.orbitofrontal.projection.task::proj_sweep \
    --cells "$D/traj_a1_s42/step000000.pt:a1:1:0" --max-twins 8000
# the step-0 swap65k cell needs the smaller twin cap (see the gotcha below)
modal run -m rhm.logit_reading.orbitofrontal.projection.task::proj_ckpt \
    --ckpt $D/traj_a1_s42/step000000.pt --stim-tag swap65k --no-do-window-diets \
    --do-glitch --max-twins 8000
# tables and figures (local; needs the .json and .npz pulled off the volume)
python -m rhm.logit_reading.orbitofrontal.projection.analyze <dir>/traj_a1_s42 \
    --tags a1,swap65k --out rhm/logit_reading/orbitofrontal/projection/results \
    --figs rhm/logit_reading/orbitofrontal/projection/figs
```

## Gotchas worth not rediscovering

- **The twin pair population is block-independent and must stay that way.** Pairs are selected by the
  *model's* surprisal, before any critic exists, so every block is read on the identical pairs and
  `dR` is comparable across blocks by construction. Selecting per block would make the comparison
  meaningless.
- **`V_pre` cancels in `dR` for every block, not just the state one.** The prefix is identical within
  a pair, so the forecast at `t_v − 1` is identical too (checked: max |Δ log q| ~1e−05), and a public
  critic's `V_pre` is a function of that forecast alone. This is what makes `dR` a clean paired
  contrast on the public side as well.
- **`dR = β · dX` holds exactly**, because the critic is linear and `V_pre` cancels. So the state
  difference is not a "strong baseline" — it is the critic itself, and its map's `R²` is 1 by
  construction. The informative number is what the *forecast* difference recovers against it.
- **The state critic's direction is not a random direction in readout terms.** The random floor for
  `‖P_M β‖/‖β‖` is 0.242 and the measured value is far below it, so "the belief cannot express the
  critic" is a stronger statement than "the belief is only 15 of 256 dimensions".
- **Save the anchors one file per anchor.** The per-episode readouts for seven blocks x twelve diets
  come to ~2000 arrays; one npz per anchor keeps each fetched file well under the size at which
  `modal volume get` has silently truncated or corrupted npz files. Every member is touched after the
  fetch regardless.
- **`save_caliper` is 0.35, not the wide selection caliper.** Pairs are still *selected* at 2.0 (so
  the surprisal match is the nearest legal token, not a filtered one), but only those within 0.35 are
  written, which keeps `_twins.npz` at tens of MB and still leaves the reduction free to impose the
  parent's 0.30.
- **At random init the save caliper stops filtering, so cap `max_twins` there.** The twin caliper
  is on the *model's* surprisal difference, and an untrained model's surprisals are nearly uniform:
  at step 0 all 45,000 eligible violations pass the 2.0 selection caliper and 35,000 pass the 0.35
  save caliper, against 5,270 at 64k, which makes `_twins.npz` ~400 MB and past the size at which
  `modal volume get` has silently truncated. `--max-twins 8000` at step 0 gives 6,241 saved pairs —
  comparable to the trained cells and ample for a floor. The pair population is re-selected per
  checkpoint by the model's own surprisal in any case (`phasic.py`'s own caveat), so the cap adds
  no new incomparability.
- **A sibling's streaming launch log kills the image build, and the `ignore` does not save you.**
  Stage `rhm/`'s `.py` files into the scratchpad and run `modal run` from there.
