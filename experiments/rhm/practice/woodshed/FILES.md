# FILES — `woodshed` (Track F1: the trust-formation-rate instrument, and the rehearsal arm)

Machinery record for the node. **No writeup lives here yet** — the curves and the run are to be
discussed with Jasper before anything is interpreted, per repo norms. This file records what
was built, what it measures, and what it cannot resolve.

**Up**: parent arc [`../README.md`](../README.md) · **asked in**
`ROADMAP.md`[^private] §4.3 (Track F, first shape **F1**) and queued in
`QUEUE.md`[^private] ("F1 — the trust-formation-rate instrument, offline first")
and in [`../census/README.md`](../census/README.md)'s interpretation (d) and Next steps.

**Direct donors** (untouched): [`../assay/`](../assay/FILES.md) (`assay.py` is the fork base;
`as_s0`/`as_s1` are the arrival cross this round extends) · [`../census/`](../census/FILES.md)
(the `given` gift, `cs_s0`) · [`../conductor/`](../conductor/FILES.md),
[`../maestro/`](../maestro/FILES.md), [`../crescendo/`](../crescendo/FILES.md),
[`../audiation/`](../audiation/FILES.md) (the logs part 1 reduces) ·
[`../intonation/`](../intonation/FILES.md), [`../caesura/`](../caesura/FILES.md),
[`../spiral/`](../spiral/FILES.md) (read as a bonus comparison, not as the core).

The node has two parts. **Part 1 is CPU-only and reads bytes that already exist**; part 2 is a
new GPU round built on top of what part 1 measured.

## Code files

| file | purpose |
|---|---|
| `reduce_trust.py` | **Part 1.** The offline instrument. Reads every A-lineage node's already-fetched `figures/<tag>/<arm>/results.json` and produces post-arrival growth curves for π's per-level proposal mass — mass vs cycles-since-arrival, per level, per arm, per tag — plus the derived rate statistics, the arrival/credit/new-rung contrasts, and the exposure-vs-clock regressions. No Modal, no GPU, no network. |
| `woodshed.py` | **Part 2.** Forks `../assay/assay.py` verbatim and adds exactly one thing, marked `# [woodshed]`: targeted **rehearsal** of a received vocabulary, with a one-bit exposure-vs-credit control. |
| `analyze_woodshed.py` | Part 2's reduction: the gates that license the tag, the rehearsal ledger, part 1's trust instrument re-run on the new tag (imported from `reduce_trust.py`, not re-implemented), and `assay`'s money readout in the same form so the numbers are comparable to `as_s0`'s. |
| `launch_detached.py` | `../assay/launch_detached.py` retargeted (session-isolated `modal run --detach`). |

## Part 1 — what the readout is

`log["probe"][k]["pi"]["mean_mass"]`, written by
[`../native/prop/prop_net.py`](../native/prop/prop_net.py)`::decompose_probe` (line 242):

```python
logits = prop(z, roots).masked_fill(~avail_mask[None, :], -float("inf"))
p      = torch.softmax(logits, dim=-1)
"mean_mass": [float(x) for x in p.mean(0).cpu()]
```

So it is a **normalised distribution over action slots**, averaged over the probe batch — not
counts, not an EMA. Slots are level-major (`prop_net.slot_layout`): at depth 6, s = 2,
L1 0–31, L2 32–47, L3 48–55, L4 56–59 (56 slots at `max_macro_level=3`, 60 at 4). Per-level
mass is the slice sum; per-level argmax share is the same slice of `argmax_hist` over `n`.

Two structural consequences the reduction reports rather than hides:

1. **Unavailable slots are masked to −inf and read exactly 0.0.** A pre-arrival level's 0.000
   is a construction, not a measurement.
2. **A level's mass is spread over only the (level, node) slots that hold a table entry**, and
   extension/recert opens more slots mid-run. So the reduction carries a **uniform reference**
   — that level's open slots over all open slots — and reports `lift = mass / uniform`, which
   is what makes arms with different table coverage comparable, and makes "π is indifferent to
   this level" a well-defined point (lift = 1) rather than a guess.

**Arrival** is the `commit` event for the level (earned arms) or cycle 1 (`given_*`, where
`log["vocab"][0][l]` is already non-null). `tau = probe cycle − arrival`.

### Statistics it computes, per (arm, level)

| stat | what it is |
|---|---|
| `m0`, `mend` | mass at the first post-arrival probe and at the last probe |
| `u`, `lift` | the uniform reference and `mass / uniform` (per probe) |
| `lift0`, `liftmin @tau` | lift on arrival, and the trough — π's de-funding of the newly opened level |
| `cross1` | first tau **at or after the trough** at which lift reaches 1.0 — the censoring-safe *time to indifference* |
| `tau@.10`, `tau@.20` | first tau at an **absolute** mass of 0.10 / 0.20 — the complement to `cross1`, whose threshold needs less absolute mass higher up (uniform is 0.286 at L2, 0.143 at L3, 0.067 at L4) |
| `d16`, `d32` | mass gained per 10 cycles over tau ∈ [0,16] and [0,32], from the interpolated curve — the window-matched rate |
| `t1/2` | cycles to half the arm's own final mass |
| `T`, `d0`, `r2sat` | saturating fit `m = M(1 − e^{−tau/T})`; `d0 = M/T` is the initial rate. `T` is starred when it falls outside the window or the fit is poor — read `d0`/`d32` there |
| `rho`, `up`, `r2lin` | monotonicity and linearity of the post-arrival series |
| `nav`, `amax`, `win` | open slots, argmax share, and the observation window |

### Sections of `figures/reduction.txt`

`(0)` inventory and arrival cycles · `(1)` the per-(arm, level) rate table, plus the **replay
groups** (bit-identical series across nodes — the arc's fidelity discipline showing up in this
instrument, and what the aggregates de-duplicate by) · `(2)` window-matched mass and lift at
fixed tau · `(2b)` the same on the **calendar** clock · `(2c)` the shape: trough and crossing ·
`(3)` per-level ordering, within-arm paired comparisons, and the by-level aggregate ·
`(4)` the named contrasts (arrival, credit, the new rung, pacing) · `(5)` exposure vs clock, in
both levels and increments form · `(6)` the instrument caveats.

Figures: `f1_growth_by_level.png` (mass and lift vs tau, one column per level),
`f1_contrasts.png` (the four named cells), `f1_rate_by_level.png` (crossing time, t½ and
end-state lift by level, gift vs earned).

### Instrument caveats (also printed as §6 of the reduction)

- **Cadence.** `probe_every = 8` plus the first and last cycle of each era. Any time constant
  under ~8 cycles is at the sampling limit. There is no per-cycle π mass series in
  `results.json`; the only per-cycle π artifact in the arc is `audiation`'s
  `snapshots/probe_trace.npz` (**raw, unmasked logits**), which lives on the volume and is not
  in the fetched mirror.
- **m0 is not zero and is not comparable across arms.** A level's slots re-enter the softmax
  carrying untrained logits, i.e. near the uniform share. The number to read is the rise above
  the uniform reference, not the raw mass.
- **Right-censoring is severe and uneven** — L4 arrivals leave 21–79 cycles of window, L2
  arrivals 80–183. `M` is extrapolation wherever the window is short relative to `T`; the
  matched-tau columns exist because of this.
- **Mass is a simplex.** L1's 32 primitive slots always compete; a rise at L3 is arithmetically
  a fall somewhere else.
- **The two clocks are different questions.** A gifted arm's tau = 0 is cycle 1, when the trunk
  is untrained; an earned arm's is cycle 18–129, when it is not. Early-tau *rates* across the
  two families are not like for like; §2b's calendar view is the other half.
- **Single seed everywhere.** The stream-displaced twins (`ma_s1`, `cr3_s1`, `as_s1`) bound
  draw-luck only. `census` finding 7's floors are error-rate floors and do not transfer to
  mass; **no floor for mass has been measured**, and the displaced pairs are the only handle.
- The exposure column exists only where `entry` recording was on; `census`'s gifted arms have
  no `entry` and are absent from §5.
- **Tag-name collision, handled:** `maestro/figures/ma_s0` (`/data/rhm_practice_maestro/ma_s0`)
  and `intonation/figures/ma_s0` (`/data/rhm_practice_intonation/ma_s0`) are different runs.
  The manifest keys on `(node, tag)`, and §0 prints both under their node.

## Part 2 — the one addition to `assay.py`

Every insertion is marked `# [woodshed]`.

| addition | where |
|---|---|
| `reh_level` / `reh_mode` / `reh_n` / `reh_every` cfg keys | `_d6_cfg`; `reh_level=None` is OFF and is the default on every donor arm |
| the rehearsal block `(c'')` | `run_arm`, between the ordinary cycle's `prop_pairs` append and `prop_train`, so this cycle's update already sees the pairs |
| `rehrng` / `reh_erng` | `run_arm`, beside the port's own `prng`/`erng` — rehearsal draws from its **own** streams, so the arm-shared position is untouched and the twin discipline still licenses every donor comparison in the tag |
| `log["reh"]` | one record per cycle (`None` when rehearsal did not fire) |
| arms `exact_reh`, `exact_exp` + their `TWIN` entries | `ARMS` / `TWIN` |
| `woodshed_run` | the entrypoint (`assay_run` renamed); `WOODSHED_ARMS = "anchor,exact,given_c1,exact_reh,exact_exp"` |

### What rehearsal is

Extra practice at the **received level's own damage cells**, drawn across **all** of that
level's nodes (varied demand, rather than the era's single node), run through the same beam at
the same width, and harvested by the same `prop_pairs` path the ordinary cycle uses. No new
learning rule, no new signal, no oracle.

It reaches π's buffer and **nothing else**, which is what keeps the intervention on the trust
half alone:

- the **miner** never sees a rehearsal solve → the table does not grow, so coverage is not
  confounded with trust;
- the **value buffer** never sees one → the critic is unchanged;
- the **generator** never sees one → the plant is unchanged;
- **no extra gradient steps** are taken — the rehearsal arms run the same `prop_steps` on the
  same `prop_batch` as every other arm. What changes is the composition of the buffer.

### The exposure-vs-credit control (`reh_mode`)

| mode | π imitates |
|---|---|
| `credit` | the rehearsal trajectories that **solved** |
| `expose` | the **same number** of trajectories, drawn at random from the same episodes' survivors |

Identical episodes, identical states, identical pair count, identical gradient steps; the
success filter is the only difference. The permutation is drawn in **both** modes, so the two
arms consume identical RNG. `prop_pairs`' own docstring names this filter "the selection step
that makes this self-imitation rather than averaging over everything the agent did" — the
control is that sentence, made into an arm.

Rehearsal **groundings are recorded, never added to `counts`**: it is extra practice compute,
reported so the price is on the record, and it does not enter the arm's declared per-solve
budget (which prices a beam *width*, not a cycle count).

### Arms

| arm | table | rehearsal | job |
|---|---|---|---|
| `anchor` | own mined | — | the floor, and a fidelity carrier |
| `exact` | the full true table at each commit cycle | — | late arrival, perfect content — the clock as it runs |
| `given_c1` | the full true table from cycle 1 | — | the ceiling: the clock already paid |
| `exact_reh` | `exact`'s | L3, **credit** | the treatment |
| `exact_exp` | `exact`'s | L3, **expose** | the exposure-vs-credit control |

Arm order is the donor's budget-risk convention: fidelity carriers first, then the treatment,
then the control (most cuttable — the treatment's own sign is readable without it).

### Gates

| gate | what it asserts |
|---|---|
| **knobs-off / cross-tag replay** (`analyze_woodshed.py` §0a) | `anchor`, `exact`, `given_c1` set no `reh_level`, so they must reproduce `as_s0`'s arm of the same name bit-for-bit over all 116 cycles on 13 series. This *is* the fork-fidelity gate; nothing else in the tag is readable without it. |
| **in-tag twin gate** (§0b) | `exact_reh`/`exact_exp` must be bit-identical to `exact` up to the cycle rehearsal first fires, and the first divergence must be that cycle. |
| **one-bit pair** (§0c) | `exact_reh` and `exact_exp` must be pair-count matched on every rehearsal cycle (`n_pairs_used` equal, `n_pairs_all` equal). |
| **donor gates, inherited** | the entry-recorder check, the G-Y soundness check and the RNG-sandbox assertion around the pre-setup selfchecks all still run (the last one cost the donor 0.6 GPU-h to learn). |

## Runs on disk

| tag | what | cost |
|---|---|---|
| `wd_smoke` | 3 arms at `--quick`; verified launch + plumbing only — no commit at quick scale, so `log["reh"]` is correctly all-`None` | 0.13 GPU-h |
| `wd_smoke2` | 2 `given_c1` arms with `reh_level=3` forced, so rehearsal fires from c1; verified the block body, both modes, and the pair-count match | 0.12 GPU-h |
| `wd_s0` | the round: `anchor, exact, given_c1, exact_reh, exact_exp`, 116 cycles each, `as_s0`'s ladder, seed 0 | 2.04 GPU-h |
| `wd_s1` | the corrected-dose re-run: `exact_reh, exact, exact_exp, given_c1`, same ladder and seed. `anchor` dropped as an unchanged replay (`wd_s0` proved all three no-rehearsal arms bit-identical to `as_s0`) and imported into the bracket from `as_s0` after asserting the era references match | 1.61 GPU-h |

`wd_s0` gates: cross-tag replay **PASS** on all three no-rehearsal arms (max|Δ| = 0.000e+00
over 13 series × 116 cycles vs `as_s0`); in-tag twin gate **PASS** (first rehearsal cycle 71,
c1–c70 max|Δ| = 0.000e+00, first divergence from `exact` exactly c71); first-cycle pair-count
match **PASS** (184 vs 184).

## Two apparatus lessons from `wd_s0`, both on the record

**1. The rehearsal demand was delivered at ~2% of its design dose, and the cause is
`corrupt_hier`'s node semantics.** `corrupt_hier(..., level, nodes, ...)` damages **every**
node in the list it is handed (`k = len(nodes)`, all `k` replaced in every instance);
`era_ctx` hands it a single-element list. The first implementation of "varied demand across
the level's nodes" handed it *all* `s**(depth−l)` nodes, so every rehearsal instance carried 8
**simultaneous** L3 corruptions — requiring ≥8 macro applications against a budget of exactly
8. Measured consequence: rehearsal solve rate **0.003** against the main path's **0.29**, and
1728 (credit) / 1232 (expose) credited pairs total over 46 cycles, against ~85,000 main-path
pairs into the same 60,000-cap buffer. The offline sizing that licensed the launch used era-3
success (0.327) as the proxy and was wrong for exactly this reason: era 3 damages one node,
the rehearsal ctx damaged eight. **Fixed** in `woodshed.py` — the batch is now split across
nodes, one node per sub-batch, at the era's own difficulty. `wd_s0` therefore measures a
near-null dose, not rehearsal.

**2. The one-bit pair gate was mis-specified.** Pair-count matching between `exact_reh` and
`exact_exp` is only *assertable* on the **first** rehearsal cycle, when the two arms are still
the same agent. From the next cycle they are different agents — which is the treatment — so
their rehearsal solve rates, and hence the matched pair count, legitimately diverge (18/45
later cycles differ in `wd_s0`). The original gate asserted equality across all 46 cycles,
which tests that the treatment had no effect: the opposite of what the gate is for. Also
recorded: `n_pairs_all` is structurally constant (`train_on="tips"` keeps every surviving
trajectory), so matching it is vacuous and is no longer reported as a gate.

`wd_s1` gates: cross-tag replay **PASS** on `exact` and `given_c1` (0.000e+00); twin gate
**PASS** (first rehearsal c71, c1–c70 identical, first divergence exactly c71); first-cycle
pair-count match **PASS** (1264 vs 1264). Dose corrected: rehearsal solve rate **0.151/0.156**
(credit/expose) against `wd_s0`'s 0.003 and the main path's ~0.29, delivering **56,840/58,832**
credited pairs against `wd_s0`'s 1728/1232 — a ~33x increase in supervision volume.

Two properties of the treatment worth carrying, both measured rather than assumed. **The
rehearsal supervision is not stationary**: `f_macro`/`f_level` start at 0.685/0.552 on the
first rehearsal cycle and settle to ~0.35-0.45/~0.15-0.28, the transition coinciding with the
beam width recovering from 5 to 18 — the earliest rehearsal cycles are the most
macro-directed. **Arm order is verified irrelevant**: `exact` ran 4th in `wd_s0` and 2nd in
`wd_s1` and produced identical logs, which is the per-arm stream discipline (`TWIN`/`STREAM`)
working as designed.

## The in-tag noise handle

Rehearsal targets L3 only, so the spread of the three `exact` arms at **L2** is what
trajectory divergence alone produces in this instrument. It is not a measured floor (single
seed, n = 3) but it is the only same-tag reference the round contains, and
`analyze_woodshed.py` prints it beside the L3 spread at the end of section (2). **No floor for
π mass has been measured anywhere in the arc** — this remains the instrument's largest gap.

## Volume layout

Part 2's runs: `rhm-scaling-data:/data/rhm_practice_woodshed/<tag>/{setup.json, summary.json,
done.txt, <arm>/results.json}`. Fetched copies under `figures/<tag>/`.

Part 1 reads, read-only, the already-fetched mirrors under
`../{conductor,maestro,crescendo,audiation,census,assay,intonation,caesura,spiral}/figures/`.
The corresponding volume paths are `/data/rhm_practice_<node>/<tag>/`.

## Reproduce

```bash
cd experiments/            # MODAL_PROFILE=chromatic

# Part 1 — CPU, no Modal. Reads the fetched mirrors already in the tree.
python3 rhm/practice/woodshed/reduce_trust.py --figures

# Part 2 — the smoke first (rehearsal forced to L2 so it fires inside a 15-cycle quick run)
python3 rhm/practice/woodshed/launch_detached.py --fn woodshed_run --tag wd_smoke \
    --arms "anchor,exact_reh:reh_level=2,exact_exp:reh_level=2" --quick

# Part 2 — the round, at as_s0's exact configuration and ladder
python3 rhm/practice/woodshed/launch_detached.py --fn woodshed_run --tag wd_s0 \
    --eras "1:25:48,2:12:40,3:6:12,4:3:9,5:1:7" \
    --arms "anchor,exact,given_c1,exact_reh,exact_exp" \
    --budget 8 --g-budget 482 --n-corrupt 1 --mine-cap 8 --gy-level 4 --entry-rec \
    --span-min-hold 128 --collect-task-matched --tm-episodes 8192 \
    --recert-every 5 --n-aud 192 --probe-every 8 --n-rt 384 --n-score 256 \
    --sil-cv 0.15 --lp-min-drop 0.10 --reh-n 64 --reh-every 1 --seed 0
python3 rhm/practice/woodshed/analyze_woodshed.py --tag wd_s0 --fetch --figures
```

Note `--reh-level` and `--reh-mode` are deliberately **not** CLI flags: a run-level
`reh_level` would silently switch rehearsal on for the fidelity carriers too. They are
declared per arm in `ARMS`, and can still be moved for one arm at a time with the donor's
`arm:key=value` override syntax.

[^private]: Not mirrored: this link points to a document in the private lab repo (the roadmap, the queue, an unrun spec, reading notes, or a conversation). See the top-level README for what is held back and why.
