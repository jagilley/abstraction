# Files — `metering_sweep/`

See [README.md](README.md) for the result and [PREREGISTRATION.md](PREREGISTRATION.md) for the
design as it was fixed before any run.

## Code files

| File | Purpose |
|---|---|
| [`sweep_plan.py`](sweep_plan.py) | The grid, **derived rather than hand-typed**. Computes `M` (monitor sequences/round) from `mon_n`/`floor_n`/`floor_draws`/`forecast_n`, then for each target monitor:collect ratio `r` emits `(collect_budget, mon_price)` under the fixed-total-spend identity `B = T/(1+r)`, `price = rB/M`. `--commands` prints every launch line. |
| [`run_sweep.py`](run_sweep.py) | Spawns all 33 `ladder` runs in parallel from one detached orchestrator (`rhm/CLAUDE.md`'s `dgp_sweep` pattern) rather than 33 CLI clients, which this node's gotchas warn evict each other. Grid imported from `sweep_plan.py` so launcher and documented table cannot drift. `--which e1a\|e1b\|e2` runs one limb. |
| [`aggregate.py`](aggregate.py) | Reads the sweep. Prints **calibration gates first** (realised-vs-target ratio, information constancy, pointwise `oracle_dup − oracle` noise floor, liveness, world intact), then `E(d)` and the interference/data decomposition, then the curve and the matched-ratio discriminators. |

## Two readout bugs the gate ordering caught, both in `aggregate.py`

Recorded because they are the same class of defect PRs #15/#16 produced seven of — except these
pointed at a *negative*, which is the direction this node had not previously had to watch.

1. **The curvature verdict tested only `|curvature|`** and reported "the saturation shoulder is in
   range" for any bend. Sign matters: `E(d)` **steepens** with data, which is moving *away* from
   saturation. Uncorrected it would have licensed reading the rising prize as a real falsification
   of an asymptotic claim the sweep never reached. Fixed to branch on sign and to print the
   doublings-to-floor extrapolation.
2. **C4's second clause was not a liveness test.** Pre-registered as *"`oracle`'s error must fall
   r1→r12, else the readout is dead"*, it struck 7 of 11 points. On a continuous-drift maintenance
   task a rising error means repair is losing to drift, not that the instrument is blind — the arms
   still separated by 0.06–0.09 at those points, 8–43× the floor. Reclassified as a **regime**
   diagnostic (drift-limited vs data-limited). The headline slope is reported both ways and is
   numerically identical, which is why the reclassification is safe to make.

A third analysis bug, found by round 2: `data_curve` pooled cells across `fm_epochs`, so once
E4 introduced runs at 2 and 32 epochs the "learning curve" mixed three different compute levels.
It is now restricted to the published (drift 0.30, 8 epochs) configuration and relabelled as the
**fixed-epoch diagonal** rather than a learning curve — which is also the correction that
retracts round 1's headline.

A fourth, non-analysis bug is worth carrying as a Modal gotcha: `metering_sweep/` initially had no
`__init__.py`, so `add_local_python_source("rhm")` never shipped `sweep_plan.py` and the
orchestrator died on import. **A Modal app whose container fails on import appears in
`modal app list` as `ephemeral (detached)` with 0 tasks**, which reads as "queued, be patient"
rather than "dead". `modal app logs <id>` is what distinguishes them.

## Code touched outside this folder (both additive, both back-compat)

| File | Change |
|---|---|
| [`../ladder.py`](../ladder.py) | `PricedMeter(Meter)` + `mon_price` / `meter_budget` kwargs. At `mon_price=1.0` the charge takes the original `int(n)` path, so every prior repro command is bit-identical. The price is **report-only inside the loop** — see [PREREGISTRATION.md](PREREGISTRATION.md) §2(i) — and bites only through the `collect_budget` it displaces. |
| [`../channel_env.py`](../channel_env.py) | `allocate` gains an `oracle_dup` branch: a bit-for-bit duplicate of `oracle`, run as a second arm so `oracle_dup − oracle` measures the instrument noise floor **at each operating point**. Deliberately **not** added to `POLICIES`, so no default run's arm set changes. |

## Round-2 additions to `../ladder.py` (both additive, both default-off)

| Change | What |
|---|---|
| `TAP_READS` / `tap_bill` | Which monitoring each policy's *drive* actually reads. `uniform`/`oracle` read nothing; `visits_only` needs only the forecast (512 seq/round); the reducibility taps need the monitor set **and** the repeat-execution floors (14080). Unit-tested against every policy the ladder can run. |
| `charge_own_monitoring` / `total_budget` | The **honest ladder**: one shared total split by each arm's own bill, so a smart allocator pays for its own smartness. Default off and bit-identical. |

## Data

Results JSON on the `rhm-scaling-data` volume under `directed_sculpting/ladder_met_*`, mirrored
to [`figures/`](figures/), with the full aggregator output saved as
[`figures/aggregate_output.txt`](figures/aggregate_output.txt).

| Tags | Limb |
|---|---|
| `met_r*` | E1-A — price at fixed total spend |
| `met_ab*`, `met_big262144` | E1-B / E1-C — abundance limb, to 262144 |
| `met_info*` | E2 — the information control |
| `met_kl*` | E3 — the damage sweep and its deep corner |
| `met_fc*`, `met_fd*` | E4 — data vs compute at matched product |
| `met_own*` | E5 — the honest ladder |
| `met_ic*` | E6/E7 — the iso-compute curves |
| `met_ep{1,2,4}*` | E8 — the published ten-arm ladder re-run at 1/2/4 epochs |

## Auxiliary READMEs

| File | Summary |
|---|---|
| [`PREREGISTRATION_ROUND2.md`](PREREGISTRATION_ROUND2.md) | Round 2's design, fixed before any round-2 run: the damage-sweep premise (refuted by its own 2×2), the data-vs-compute cut (which overturned round 1), and the honest ladder. Predictions P7–P10 with their kill criteria. P8's premise was wrong and P10 was right by ~40×. |
| [`PREREGISTRATION.md`](PREREGISTRATION.md) | The design, the derived mechanism, six numbered predictions, the five calibration gates and the outcome-reading table — all committed before the first run. Its §2 is the load-bearing part: the meter is *inert* in the published ladder, so a bare price multiplier is a provable no-op and a sweep of it alone would return a flat curve for reasons unrelated to metering. |
