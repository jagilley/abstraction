# abstain — files

**Up**: [`../README.md`](../README.md) (orbitofrontal, the writeup) · **Donor**: [`../../striatum/README.md`](../../striatum/README.md) (striatum; [`junction/`](../../striatum/junction/README.md),
[`norm/`](../../striatum/norm/README.md)) · **Tables**: [`results/tables.md`](results/tables.md)

The interpretation is the super-node's ([`../README.md`](../README.md) §3) and [`README.md`](README.md) here is a
pointer. `results/tables.md` and `results/tables_stream.md` carry every number, their reproduction headers and the definitions of every
object below.

Two venues. The **anchor venue** (CPU only, no new compute) reads the choice at five query offsets after
the edit onset and after the first impossible token, on the banked `striatum/` cells. The **stream
venue** (one L4 cell per stimulus set) re-runs the same frozen trunk, frozen actor and critic recipe and
writes the critic's level and the realised outcome at *every* position of held-out windows, so the same
choice can be read on the learner's whole life rather than only around an event.

## What this node is

`striatum/` fixed the actor and gave it no choice; this node lifts that one decision. At a query the
actor either **answers** — returning the realised outcome, 1 if the frozen actor named the level-`l`
constituent's feature correctly and 0 otherwise — or **abstains**, returning a fixed `r_abs` swept over a
grid. Every value reading in the arc (the critic's pre-event level `V_pre`, its level `V` at the event,
the revision `R`, the outcome surprise `δ`) becomes a candidate **gate** on that decision, scored by the
share of an oracle's prize it captures, against a constant floor and beside the model's own
surprise-shaped readouts, `basalis`'s binned out-of-sample `R²` screen and three pure-noise nulls.
Every gate is read under four conditionings — nothing; anchor-position strata (the parent's matcher
gotcha in decision currency, and load-bearing: the bare position out-gates the critic marginally);
position × the model's surprisal; and position × the critic's level `V` — the last two being the two
halves of `striatum`'s addendum in decision currency. Under each, the floor is the per-stratum
constant, so a share is always what the gate adds to what the decider already had.

## Code files

| file | purpose |
|---|---|
| [`analyze.py`](analyze.py) | The whole anchor (CPU) venue. Reads the banked `step*_striatum_{a1,swap65k}.npz` cells, builds every gate's scalar per (venue, anchor, level, offset), fits coeruleus's quantile step policy on half the banked test rows and reads it out on the other half — marginally, inside anchor-position strata, and inside position × model-surprisal strata — and writes `results/tables.md` and the figures. Also carries `bin_r2` (basalis A5's screen), `subgroup_returns` (where a gate's value lives: violation vs quiet, by `k*`), and `sec_repro`, which re-runs `striatum/analyze.py`'s `rematch` on the same npz as the reproduction gate. |
| [`task.py`](task.py) | The stream venue's Modal cell (`abstain_stream`, one L4 per stimulus set, fanned out by the CPU coordinator `abstain_sweep`). Same trunk, same frozen actor, same split and the same streamed-Gram critic recipe as `striatum/task.py`, but it writes `V[l,d](s_t)`, the clean-only and shuffled-outcome critics' levels, the realised outcome, the model's surprisal and entropy and the banked coeruleus head at **every** position of three arms: `edit` (the event is inside), `orig` (the unedited counterpart, position for position) and `quiet` (fresh clean windows). Prints the actor and critic reproduction gates. Its docstring records why the log Bayes factor cannot be computed per position. |
| [`analyze_stream.py`](analyze_stream.py) | The stream venue's reduction. Same gates, policy, floors, conditionings and screens as `analyze.py`, with the fit / read split taken **by window**, and the realised return decomposed by arm and by distance since the violation. |
| `__init__.py` | package marker |

## Results and figures

| path | contents |
|---|---|
| [`results/tables_stream.md`](results/tables_stream.md) | The stream venue: the reproduction gate, the decision on the whole stream under all four conditionings, the decomposition by arm and by distance since the violation, the screen, the offset sweep. |
| `results/wait_app.sh` | Waits on the Modal app itself rather than on a local launch log, which `modal run --detach` abandons. |
| [`results/tables.md`](results/tables.md) | The anchor venue. Every number: the reproduction gate against `striatum/README.md` §2 with all four guards; per (venue, anchor, checkpoint) the decision at the bite point under each conditioning, the AUC-beside-binned-`R²` screen, the `r_abs` sweep, the offset sweep, the climb across checkpoints, and the subgroup decomposition. |
| `figs/abstain_prize.png` | Share of the oracle prize per gate and level, marginal and inside position strata, plus the detection-vs-gating scatter (AUC on "the actor will be wrong" against binned out-of-sample `R²` on the stake). |
| `figs/abstain_conditioning.png` | Every gate's share under the four conditionings at `l = 2`, offsets 0 and 4 — striatum's addendum, and the position confound, in one picture. |
| `figs/abstain_where.png` | Where the value lives: share by offset since the anchor, the violation-vs-quiet decomposition on the mixed `a1` venue, and the decomposition by `k*`. |
| `figs/abstain_climb.png` | The share across the clean trajectory's four checkpoints, inside position strata, at levels 1–3. |
| `figs/abstream_{a1,swap65k}.png` | The stream venue: every gate's share under the four conditionings, where the value lives along the distance since the violation, and the offset sweep. |

PNGs are gitignored under `experiments/`; regenerate with the command in `results/tables.md`'s
reproduction header.

## Inputs

Nothing new is computed on Modal. The node reads the banked `striatum/` artefacts on the
`rhm-scaling-data` volume (**`chromatic`** workspace) under
`/data/v16_s2_L6_m4_distinct/logit_reading/traj_a1_s42/`:
`step{000000,008000,024000,064000}_striatum_{a1,swap65k}.npz`, and the stream venue's own
`step064000_abstream_{a1,swap65k}.{npz,json}` (61 MB each) written by `task.py`. Fetch them with
`modal volume get` and
**touch every member after the fetch** — `modal volume get` has silently truncated a large npz and once
returned one with a bad CRC that `np.load` did not catch.
