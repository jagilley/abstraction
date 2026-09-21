# striatum — file reference

**Up**: [README.md](README.md) · [../FILES.md](../FILES.md)

## Code files

| file | purpose |
|---|---|
| `parse.py` | The generating parse of the Part 2 stimuli, per level, for the original **and** the edited stream. Replays `stimuli.make_edits`'s RNG with a latent-recording `realise`, asserts the reconstruction is bit-identical to the cached `stimuli_<tag>.npz` on eight arrays, and caches `y_orig` / `y_edit` (level-ℓ feature of the constituent containing each window index) as `parse_<tag>.npz`. Also `clean_answers`, the same map for `altitude.units.windows_with_parse` output. |
| `task.py` | The node. The clean-trained actor (a v-way head per query level on the frozen trunk), the outcome-trained critic `V[l, d]` (streamed Gram, closed-form ridge over every block and every ridge penalty, MLP beside it, shuffled-outcome head beside every one), the value revision at the edit onset and at `t_v`, the consequence and realised-damage labels, the matched contrasts, the oracle probes and the structural references. Has its own Modal app/image so the mount ignores sibling nodes' `results/` trees. |
| `analyze.py` | Every table in `results/tables.md` and the figure in `figs/`, from the JSONs pulled off the volume. Also holds the local re-matching (position in the strata, sign-balanced surprisal pairs) and the dated addendum (conditional AUCs inside quantile bins of a second score, and out-of-fold two-column combinations). |
| `addendum.py` | One extra column: the model's realised horizon excess surprise `sum_{u=t}^{t+8}(NLL_u - H(q_u))` on the stimulus windows, written beside the main artefacts. `task.py` stores only the single-position excess at the anchor, which is a different object from the one the banked `coeruleus/` head predicts. |

## Children

| folder | summary |
|---|---|
| [`junction/`](junction/README.md) | The value reader's diet (2026-09-17): eleven training diets spanning a violation–cost association of −1 to +1 on fixed held-out rows, the width-free positive control and the oracle legality probe as ceiling; the surprise-gated row ladder against random, position-matched random, bottom-gated and oracle-damage arms with the containment of the banked excess head as the criterion. [`FILES.md`](junction/FILES.md). |
| [`norm/`](norm/README.md) | The value side's own norm (2026-09-17): diets that move the base rate of outcome or damage with the `(etype, j)` cell histogram pinned bit-identically; the critic's pre-event level and its regression on the banked critic's across worlds; the response and the outcome surprise on fixed rows; the same-prefix twins in outcome currency with the regression-to-the-mean confound cancelled and a token-balance guard; scope by `k*`. [`FILES.md`](norm/FILES.md). |
| [`../orbitofrontal/regime/`](../orbitofrontal/regime/FILES.md) (moved 2026-09-17 to the `orbitofrontal/` super-node) | A world where a violation predicts cost (2026-09-17): the grammar's stream with a two-state hidden corruption regime (mean dwell 288 clean / 32 noisy, ε = 0.002 / 0.12, marginal 0.0138) against an i.i.d. control at the same marginal rate; one next-token trunk per world; the exact running regime filter as the reference the critic's revision is read against; natural and legal twins at the corrupted token. Tables only — no README, the interpretation waits on a discussion. [`FILES.md`](../orbitofrontal/regime/FILES.md). |

## Artefacts on the volume

Under `/data/v16_s2_L6_m4_distinct/logit_reading/` (`rhm-scaling-data`, **`chromatic`** workspace):

| path | what |
|---|---|
| `parse_{a1,swap65k}.npz` | `y_orig`, `y_edit`: `(6, n, 65)` level-ℓ answers for every window index. Model-independent, built once. |
| `traj_*/stepNNNNNN_striatum_<tag>.json` | Config, the actor's clean accuracy per level and block, the critic's held-out R², the value trace, and every reduced table. |
| `traj_*/stepNNNNNN_striatum_<tag>.npz` | Per-episode arrays for the held-out rows (revisions, consequence, damage, probes, references), so `analyze.py` can regroup without re-running. |

## Reproduction

```bash
cd experiments
# the generating parse of both stimulus sets (CPU, ~2 min each; asserts bit-identity
# against the cached stimuli before writing anything)
modal run -m rhm.logit_reading.striatum.parse::build_parse --tag a1
modal run -m rhm.logit_reading.striatum.parse::build_parse --tag swap65k \
    --n 65536 --seed 2027 --swap-only

# the node: every (checkpoint, venue) cell in its own container, <= 4 at a time
D=/data/v16_s2_L6_m4_distinct/logit_reading
modal run --detach -m rhm.logit_reading.striatum.task::striatum_sweep \
    --ckpts "$D/traj_a1_s42/step000000.pt,$D/traj_a1_s42/step008000.pt,$D/traj_a1_s42/step024000.pt,$D/traj_a1_s42/step064000.pt,$D/traj_eps01_s42/step064000.pt" \
    --stim-tags a1,swap65k --n-clean 6144

# the addendum column: the model's REALISED horizon excess on the stimulus windows,
# the object the banked coeruleus head was trained to predict (~2 min each, non-destructive)
modal run -m rhm.logit_reading.striatum.addendum::hexcess_ckpt \
    --ckpt $D/traj_a1_s42/step064000.pt --stim-tag swap65k
modal run -m rhm.logit_reading.striatum.addendum::hexcess_ckpt \
    --ckpt $D/traj_a1_s42/step064000.pt --stim-tag a1

# tables and figures (local; needs the .json, the .npz and, for the addendum, the
# _hexcess.npz pulled off the volume)
modal volume get rhm-scaling-data $D/traj_a1_s42 <dir>/traj_a1_s42
python -m rhm.logit_reading.striatum.analyze <dir>/traj_a1_s42 --tags a1,swap65k \
    --out rhm/logit_reading/striatum/results --figs rhm/logit_reading/striatum/figs
```

`results/wait_app.sh <app-id>` waits on the **remote** app: `modal run --detach`'s client
returns when it disconnects, which is not the end of the run, so the local launch log is
not the authority.

Measured: 142 s per a1 cell (peak RSS 6.9 GB), 208 s per swap65k cell (8.6 GB), ~30 L4
GPU-minutes for all ten cells. The launch requested 24 GB; 12 GB is enough.

## Gotchas worth not rediscovering

- **`modal run --detach` returns before the run does.** Its client exits on disconnect, so
  a waiter on the local launch log fires early. Wait on `modal app list` or on the expected
  artefacts (`results/wait_app.sh`).
- **A sibling node writing its launch log breaks the image build.** `add_local_python_source`
  raises `ExecutionError: ... was modified during build process` even for a file it will
  ignore. This node declares its own app and image with an `ignore` that skips `results/`
  trees and `.log` files.
- **Outcome arrays are `uint8`; `o_orig - o_edit` underflows to 255.** Cast before
  differencing.
- **The matcher's strata have to contain the anchor position.** With strata on `(k*, j)`
  alone, the consequential rows at the `t_v` anchor are systematically earlier in the
  window (position guard 0.06-0.34) because consequence there means the violation was
  detected while still inside the edited span. Adding a position bucket pins it, and
  sign-balancing inside `|Δs|` bands pins the surprisal guard (the parent's gotcha, which
  a caliper alone does not fix: it left 0.375 at the onset anchor).
- **`excess` in `task.py`'s record is the single-position excess at the anchor**, not the
  horizon sum the banked `coeruleus/` head predicts. They behave differently on the damage
  label (0.47-0.50 vs 0.55-0.61 matched at `t_v`), so do not read one for the other.
- **At the edit onset the consequence contrast is not identifiable once position is
  controlled**, by construction: the query's constituent is inside the edit span iff
  `0 <= t_q - e < 2^j`, so `(j, t_q - e)` determines the label. That cell survives only at
  the `t_v` anchor, where the detection delay decouples `t_v` from `e`.
