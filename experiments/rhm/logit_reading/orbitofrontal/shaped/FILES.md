# shaped — file index

**Up**: [`../README.md`](../README.md) (orbitofrontal, the writeup) · **Donor**: [`../../striatum/README.md`](../../striatum/README.md) (striatum) · **Tables**: [`results/tables.md`](results/tables.md)

## Code files

| file | purpose |
|---|---|
| [`shape.py`](shape.py) | The fine-tune. `shape_ckpt` continues training `traj_a1_s42/step064000.pt` with the actor's own objective — 16-way level-ℓ query heads (ℓ = 1…6) on `post_block7`, trained jointly with the trunk on fresh clean windows and parses — in three arms (`task`, `task_ntp`, `ntp`) at matched steps and bit-identical data; `shape_sweep` is the CPU coordinator that runs one arm per container. Checkpoints are written in `train_trajectory`'s format so `calibration.load_trajectory_ckpt` and every downstream cell read them unchanged; the query heads are saved beside the checkpoint as `heads_stepNNNNNN.pt` (never `step*_heads.pt`, which `calibration_sweep`'s glob would try to load as a trunk). Also `probe_ckpt` / `probe_sweep`: Part 2a's per-level violation detection (the parent's `detect_auc_stratified`) plus a clean held-out next-token CE, per checkpoint. |
| [`analyze.py`](analyze.py) | The cross-arm reduction → `results/tables.md` and `figs/`. Joins the banked artefacts (striatum, junction, calibration/altitude, probe, shape log) across the four trunks, and adds the two reductions the banked analyzers cannot do on a shaped trunk: `addendum_hx` (striatum's containment addendum with the model's **realised** horizon excess in the banked `coeruleus/` head's place — that head is fitted to the frozen trunk's states and does not transfer) and `legality_matched` (junction's matched legality contrast computed from a *striatum* per-episode npz, so every arm has it, with striatum's oracle legality probe on identical rows as the ceiling). |

### Shell / helper files in `results/`

| file | purpose |
|---|---|
| `fetch.sh` | pull just this round's artefacts off the volume into a local mirror |
| `verify.py` | touch every member of every fetched `.npz` (`modal volume get` has silently truncated one, and once returned one with a bad CRC on a single member that `np.load` did not catch) |
| `reduce.sh` | the whole local reduction: verify, the banked per-arm analyzers, then the cross-arm tables and figures |
| `wait_artifacts.sh` | polls the volume for all 27 expected artefacts; `modal run --detach` returns before the run does, so this waits on the artefacts, not on a launch log |

## What is run on each trunk, and by whom

Everything else is the banked machinery, run unchanged on the shaped checkpoints:

| readout | entry point | venues |
|---|---|---|
| the value-side battery (actor, critic, revisions, oracle probes, across-level) | [`../../striatum/task.py`](../../striatum/task.py) `striatum_sweep` | `a1`, `swap65k` |
| the realised horizon excess column | [`../../striatum/addendum.py`](../../striatum/addendum.py) `hexcess_sweep` (added here; `hexcess_ckpt` unchanged) | `a1`, `swap65k` |
| the diet dimension (window diets only) | [`../../striatum/junction/task.py`](../../striatum/junction/task.py) `junction_sweep` | `a1` |
| Part 1's arrays | [`../../calibration.py`](../../calibration.py) `calibration_sweep` | — |
| Part 1's altitude (`κ*`, `α*`, the identity) | [`../../altitude/identity.py`](../../altitude/identity.py) `identity_sweep` | — |
| Part 2a detection + clean CE | [`shape.py`](shape.py) `probe_sweep` | `a1` |

Per-arm tables in the banked format are written by running the banked analyzers on each
arm's directory; `results/tables.md` carries the cross-arm join.

## Artefacts on the volume

`rhm-scaling-data` (**`chromatic`** workspace), under `/data/v16_s2_L6_m4_distinct/logit_reading/`:

- `shape_{task,task_ntp,ntp}_s42/step{001000,003000}.pt` — the shaped trunks
- `shape_*/heads_step{001000,003000}.pt` — the shaping-time query heads
- `shape_*/shape_log.json` — per-arm shaping curve (query accuracy per level, held-out clean NTP CE)
- `shape_*/step003000_striatum_{a1,swap65k}.{json,npz}`, `..._hexcess.npz`, `step003000_junction_a1.{json,npz}`
- `shape_*/step*_calibration.{json,npz}`, `shape_*/altitude_identity.json`, `shape_*/step*_probe_a1.json`

## One backwards-compatible edit outside this folder

[`../../striatum/analyze.py`](../../striatum/analyze.py) `addendum()` now skips its banked-head columns when no
`coeruleus/` head sits beside the checkpoint (shaped trunks have none). With a head
present the table is byte-identical to before.
