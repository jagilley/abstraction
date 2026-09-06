# FILES — `ostinato` (E′ round 3: the benchmark-estimability sweep)

Machinery record for the node. **No results are interpreted here** — the reduction's numbers go
to the orchestrator and are discussed with Jasper before any writeup (repo norm). There is
deliberately no `README.md` for the same reason.

**Up**: parent arc [`../README.md`](../README.md) · joint record of the round this follows up,
[`../two_deltas/README.md`](../two_deltas/README.md) (finding 6 and its unmeasured candidate
mechanism) · `../../../../ROADMAP.md`[^private] §7.1.3 (Track E′) ·
`../../../../QUEUE.md`[^private] ("the benchmark-estimability sweep").
**Idea doc**: [`performance_error_is_the_bridge`](../../../../ideas/performance_error_is_the_bridge.md)
§13(c) — the benchmark timescale's **interior optimum**, "above sampling jitter, below
competence drift", §12(c)'s estimability condition made quantitative; §13(b) (δ-over-raw
hygiene, the `rawx` comparator's reason); §14(b) (the gain's regime dependence).

**Direct donor** (untouched): [`../intonation/`](../intonation/FILES.md) — `intonation.py` is
forked here; every addition is marked `# [ostinato]`. Through it:
[`../tacet/`](../tacet/FILES.md) → [`../crescendo/`](../crescendo/FILES.md) (A3: the
thermostat, the L4 frontier, the yoke, the measured floors) →
[`../maestro/policy.py`](../maestro/FILES.md) (**imported**, not forked) and
[`../native/span/span_net.py`](../native/span/FILES.md) (**imported** — `SpanHead`,
`SpanExecutor`, `parity`; gates S-1…S-6 are that module's).

## The question, and the one thing that varies

`ma_s0` (`intonation` round A) found the gain trio **reversed** against abundance: the
un-benchmarked raw-`e` gain posted the tag's only above-floor value cells while the benchmarked
δ gain — abundance's winner — dropped to the worst L2 parity. The candidate mechanism,
discussed and **not measured**, is §13(c)'s estimability condition: `b(s)` is meaningful only
under repetition per context, and scarcity is the removal of repetition. **This node measures
it.**

### What the banked record actually says scarcity removed

Sized offline from `in_s0`'s and `ma_s0`'s own per-cycle `log["perf"]` before anything was
built (the discipline `ma_s0` itself used):

| per era-5 cycle | `in_s0` (abundance) | `ma_s0` (scarcity) |
|---|---|---|
| macro **calls per slot** | 14.85 | 14.26 |
| **rows per call** | 159.4 | **75.4** |
| active slots | 28 | 28 |
| era-4 rows per call | 134.8 | 63.4 |

So scarcity left the number of times a context **recurs** essentially unchanged — `b(s)` got
the same number of EWMA updates in both regimes — and halved the number of **performances
behind each update**. The estimability quantity on this substrate is therefore the effective
sample count standing behind `b(s)` at read time, `(rows per update) × (1/α updates)`, and
`ma_s0`'s own thinning was **ρ_eff = 75.4 / 159.4 = 0.47**.

### The knob: `perf_bench_frac` (ρ)

**The benchmark integrates a uniform random ρ-subsample of each execution's rows.** Every row
still receives a δ and still carries credit; only `b(s)`'s **own evidence** is thinned.

| held fixed by construction | how |
|---|---|
| **total volume** | same instances, same beam, same executions, same rows trained on, same ledger, same grounding. `n_pr`/`pr_width` are abundance's, unchanged, in every arm |
| **difficulty** | the grammar, damage ladder, era schedule, action set, caps and every floor are the donor's |
| **competence drift** | the EWMA updates at the **same wall-clock instants** at the **same α**, so the benchmark's *staleness* is identical across rungs and only its *sampling jitter* moves. §13(c) names both halves of the interior optimum; this isolates one |
| **the diet** | no gate arm in the tag; the plasticity gain is the only consumer |
| **ρ = 1.0 ≡ the donor** | `_bench_mean` returns `np.mean(e)` by the same numpy reduction and **draws no RNG at all** (verified offline: the ρ = 1 benchmark series is bit-invariant to the meter seed). This is what gate G-F certifies |

**Why the knob sits on the benchmark's evidence rather than on the world's supply.** A
world-side repetition knob (fewer performances per cycle) is exactly what `ma_s0` already did,
and it moves volume with repetition — the confound this run exists to break. It would also
cost **three arms per rung** (the raw and uniform comparators would each have to be re-run in
every supply regime). Thinning the benchmark's evidence makes `os_rawx` and `os_log`
**rung-invariant** — neither reads `b(s)` — so a rung costs **one** arm and one raw line serves
the whole ladder. That is what makes a four-rung sweep fit the budget at all.

**The ladder, sized on `in_s0`'s own numbers.** era-5 abundance: 159.4 rows/call, sd(`e`) ≈
0.13, mean |b − e| ≈ 0.012, α = 0.05 so an EWMA read averages (2−α)/α ≈ 39 updates.
SE(b) ≈ sd(e)/√(39·ρ·rows) ≈ 0.00165/√ρ. Offline simulation at the measured scale confirms it
(b's sd over the last 200 updates: **0.00227 / 0.00467 / 0.02856** at ρ = 1 / 0.20 / 0.008,
against √-law predictions 0.0023 / 0.0051 / 0.0254). So

| rung | ρ | benchmarked rows per update (era 5, predicted) | signal / SE(b) |
|---|---|---|---|
| `os_r100` | 1.000 | 159 | ≈ 5.3 |
| `os_r020` | 0.200 | 32 | ≈ 2.6 |
| `os_r004` | 0.040 | 6.4 | ≈ 1.0 |
| `os_r001` | 0.008 | 1 (the floor binds) | ≈ 0.4 |

which brackets SNR = 1 in the interior. `ma_s0`'s own ρ_eff = 0.47 sits between the top two
rungs and is reported as a **landmark on the measured curve**, not paid for as an arm. The
bottom rung is the "one performance per update" floor: `_bench_mean` always lets at least one
row through, because a benchmark update that integrates nothing is a *frozen* benchmark, which
would be a different treatment.

## Arms

Abundance (`in_s0`'s configuration verbatim). Every arm is a **clock yoke** of
`os_log`, so all six are lifetime-, era-boundary- and commit-cycle-identical and differ in
exactly one knob. All six twin onto `enum_live`'s stream (`anchor_long`'s), as the donor's do.

| arm | ρ | consumes | role |
|---|---|---|---|
| `os_log` | — | nothing | the instrument arm, the **uniform-plasticity control**, and the **clock source** every other arm replays |
| `os_rawx` | — | raw `e` gain, `exp(+e/τ_w)` | §13(b)'s hygiene control in `ma_s0`'s **fixed** form (same transform, same cap, same normaliser as the δ arms; no benchmark, no agency gate). **Rung-invariant** — it never reads `b(s)` |
| `os_r100` | 1.000 | δ gain | the ladder's top rung, and `intonation`'s `perf_gain` **exactly** (ρ = 1 draws no RNG) |
| `os_r020` | 0.200 | δ gain | |
| `os_r004` | 0.040 | δ gain | |
| `os_r001` | 0.008 | δ gain | the one-row floor: the benchmark sees a single performance per update |
| `perf_given_rho` | 0.040 | δ gain | **preflight only** — a loop arm cannot commit on a forty-step substrate, so no slot is ever minted and the ρ path would go untested. Holds the true tables, so slots mint at c1. `given`'s stream |

**`in_s0`'s `perf_raw` is never used and never quoted** (realised mean weight 0.40×, budget
confounded, superseded by `rawx` — `two_deltas` caveats). The gate arms (`perf_hi`/`perf_mean`/
`delta_hi`) are **not carried**: both regimes already measured the gate seat losing, the
`perf_mean` coverage fix is in the lineage and inherited, and the sweep's question is about the
plasticity seat.

## Design calls, and the reasoning

**Abundance, not scarcity, is the regime the ladder runs in.** The reversal's endpoints are
already on disk; what is not on disk is whether removing repetition **alone, at abundance's
volume**, moves the benchmarked-vs-raw ordering. Running at abundance makes ρ = 1 the in-tag
reproduction of the abundance reference and every lower rung a controlled removal of
repetition with nothing else touched.

**The subsample is drawn on the meter's own stream** (`intonation` pricing note 3: the meter
draws no RNG on the shared stream), keyed by `perf_bench_seed` and the arm seed. At ρ ≥ 1 the
draw is **skipped entirely**, which is what makes the identity rung bit-identical rather than
equal-in-expectation.

**The subsample serves both the first-sight init and the EWMA update**, so a rung is uniformly
"the benchmark sees ρ of this context's performances" rather than a mixture of two policies.

**`perf_alpha` is left at the donor's 0.05 in every arm.** α is the *other* half of §13(c)
(staleness), reduction §P5 already recomputes δ across an α ladder offline for free, and moving
both halves at once would make neither readable. The sweep moves sample count with the horizon
pinned.

**Four rungs + two rung-invariant comparators = six arms**, against the queue's ~2 GPU-h sizing.
An interior crossover needs at least three rungs plus an endpoint to bracket it, and the two
comparators are the ordering's other half; the knob choice already halved the cost by making
them rung-invariant. The overrun is recorded rather than engineered around.

## Code files

| file | purpose |
|---|---|
| `ostinato.py` | The substrate. Forks `../intonation/intonation.py`; every addition marked `# [ostinato]`. New: `PerfMeter.bench_frac`/`brng`/`_bench_mean` (the rung), the `n_upd`/`n_bench` accumulators (the rung's own measurement), `perf_bench_frac`/`perf_bench_seed` in `_cfg`, the six arms + `perf_given_rho` + their `TWIN` entries, `bench_frac` on the per-cycle meter row and in `perf_mode`, gate **O**, and G-F retargeted to `intonation.py` **on the metered path**. Entrypoints: `preflight`, `fidelity_smoke`, `ostinato_run`. |
| `analyze_ostinato.py` | The reduction. `intonation`'s (§0–§8, §A/§C/§E, §G/§G2/§H, §P1–§P6 — all of which already enumerate gain arms from their configs rather than by name, so four rungs report through them unchanged), plus §R1 (`rung_ladder`: declared vs **realised** ρ, rows/update, benchmarked rows/update, effective n, SE(b), the arm's own \|b−e\|, and signal/SE — per run and per era) and §R2 (`rung_orderings`: the benchmarked-vs-raw and benchmarked-vs-uniform deltas at every rung in recovered fraction and in multiples of the measured floors, the 2×2 occupancy per rung, and trust formation per rung). |
| `launch_detached.py` | Session-isolated detached launcher (`intonation`'s, retargeted). |

`policy.py`, `floors.json`, `phase0_l4.py` and `span_net.py` are **deliberately absent** — the
rule, the dead zones, the offline `Miner.build` replay and the executor primitive are reached by
import, as in the donor.

## What the fork adds (each `# [ostinato]`-marked)

| addition | where | why |
|---|---|---|
| `bench_frac`, `brng`, `_bench_mean` | `PerfMeter` | **the one treatment**: how much of each execution `b(s)` integrates. ρ ≥ 1 draws no RNG and returns `np.mean(e)` — the donor |
| `em` used for init **and** update | `PerfMeter.score` | one policy per rung, not two |
| `n_upd`, `n_bench` | `PerfMeter._cell` | the rung as a **measured** variable: realised ρ = `n_bench / n`, realised repetition = `n_bench / n_upd` |
| `perf_bench_frac`, `perf_bench_seed` | `_cfg`, `ostinato_run` | default 1.0 / a fixed key, so an unnamed config is `intonation.py` |
| the six arms + `perf_given_rho` + `TWIN` | `ARMS`, `TWIN` | all on the anchor's stream |
| `bench_frac` on the meter row and in `perf_mode` | `run_arm` | the rung on the arm's own record |
| gate **O** | `preflight` | below |
| G-F retargeted **and strengthened** | `fidelity_smoke` | below |
| §R1, §R2 | `analyze_ostinato.py` | the sweep's own readouts |

## Gates

| gate | what it asserts | where |
|---|---|---|
| P-1…P-12, L-1…L-8 | A1's/A2's full offline policy suite against the **imported** module | `maestro.policy.policy_gate()` |
| C-1…C-5, T-1…T-5, I-0…I-5 | A3's, E3′'s and `intonation`'s gates, unchanged, re-run in this fork. I-2 and I-4 extended to the sweep's arms — **every rung is a gain arm, so the budget normalisation has to hold at every rung** or the ladder compares plasticity budgets instead of benchmarks (`in_s0`'s `perf_raw` lesson, applied to the sweep) | `preflight` |
| **O-1** | **the rung binds**: each rung arm carries its declared ρ on its own record, and the **realised** thinning (`n_bench / n`) sits between the one-row-per-update floor and ρ. The independent variable is measured, not declared | `preflight` |
| **O-2** | **the rung is inert at ρ = 1**: the identity rung thins exactly zero rows (`n_bench == n`). The bit-identity half is G-F's | `preflight` |
| **O-3** | **the rungs differ**: two arms at different ρ do not end with the same benchmark state. If they did, the ladder would be one arm run four times | `preflight` |
| **G-F** | with `perf_bench_frac = 1.0`, this fork replays **`intonation.py`** at 0.000e+00 against a 0.000e+00 donor self-replay control. Run on the **metered** arms (`perf_given` = meter alone, `perf_given_g` = meter + δ gain consuming `b(s)`) rather than the donor's un-metered pair, because the knob lives on that path and only that path; `span_min_hold`/`span_tau` are dropped so slots actually open, and `n_fired > 0` is **asserted** so the gate cannot pass vacuously | `fidelity_smoke` |
| twin | every arm vs the anchor's stream up to its own first action | `analyze_ostinato.py` §0 |
| N-1, N-2, F, B-1, panel-vs-G-Y | A1's/A3's, unchanged | as donors |

## Volume layout

`rhm-scaling-data:/data/rhm_practice_ostinato/<tag>/{setup.json,summary.json,<arm>/results.json,done.txt}`.
Fetched copies, figures and `reduction.txt` under `figures/<tag>/`.

## Runs

| tag | what | outcome |
|---|---|---|
| `_preflight` | interface + gates C, T, I and **O**, toy sizes, 8 arms (`perf_given_rho` added for the reason `perf_given` exists in the donor) | **ALL PASS**. O-1 on `perf_given_rho`: 27 937 rows, 1 028 updates, 1 645 benchmarked → realised ρ 0.0589 at declared 0.04 (the one-row floor binding at 27.2 rows/update, exactly as designed). The four `os_*` loop arms minted no slot at preflight scale, as expected. `results/preflight.log` |
| `os_gf` | G-F: in-process fork-vs-`intonation.py` replay on the **metered** arms at `max_macro_level=4` | **PASS — 0.000e+00** on both arms (`perf_given` = meter alone, `perf_given_g` = meter + δ gain) against a 0.000e+00 donor self-replay control, commits equal, and **98 070 head rows fired** so the gate is not vacuous. `results/gf.log` |
| `os_s0` | the main run: 6 arms (2 rung-invariant comparators + 4 rungs), A3's ladder/caps/floors, abundance | clean, **12 023 s = 3.34 GPU-h** (+ ~0.7 lost to one preemption), 14.3–14.5 s/cycle, all six arms **131** cycles, commits at c49/c72/**c104** and advances at 58/88/114/122/131 in every arm — `in_s0`'s clock exactly. All gates PASS; in-tag twin gate 0.000e+00 for all five yoked arms over c1–c48. Record `figures/os_s0_reduction.txt` (§0–§8, §A/§C/§E, §G/§G2/§H, §P1–§P6, **§R1–§R2**) |

**A free full-scale cross-tag identity, stronger than the smoke G-F.** Because ρ = 1 draws no
RNG, the identity rung is not merely *like* the donor's abundance reference — it *is* it. Over
all 13 logged series and all 131 cycles, against `in_s0`:

| pair | max\|Δ\| | commits |
|---|---|---|
| `os_s0/os_log` vs `in_s0/perf_log` | **0.000e+00** | equal |
| `os_s0/os_r100` vs `in_s0/perf_gain` | **0.000e+00** | equal |

i.e. the fork reproduces `intonation`'s two abundance arms bit-for-bit on a different day in a
different container, so every rung below ρ = 1 is measured against a reference that is
literally the donor's.

### Instrument note: preemption restarts the arm sweep from arm 1

`os_s0`'s first attempt was **preempted by Modal** after finishing arm 1 (`os_log`) and starting
arm 2 (`launch_os_s0.log:370`, *"Container terminated due to preemption. Your Function will be
restarted with the same input."*). The function restarted **from the shared setup and arm 1,
cycle 1**: the runner's per-cycle `volume.commit()` checkpoints artifacts but there is **no
per-arm resume guard**, so a partially-finished arm sweep is redone in full. The preempted
attempt banked nothing and cost ~1 arm + 1 setup (~0.7 GPU-h).

**Worth building into any future fork of this launcher**: a per-arm resume guard — on start,
skip any arm whose `<arm>/results.json` already carries `complete: true` on the volume, and
rebuild `measured_plans` from those banked results so the yoke handoff still works (arm order
is load-bearing and asserted, so the guard has to restore the clock source's plan, not just
skip the arm). That is a change to `ostinato_run`'s arm loop, not to `run_arm`, so it would not
touch any gated code path.

Contingency standing for this node (coordinator, 2026-08-31): on a **second** preemption,
relaunch with `os_r020` dropped — rungs 1.0 / 0.04 / 0.008 still bracket the SNR ≈ 1
crossing — rather than paying the full six-arm ladder a third time.

## Reproduce

```bash
cd experiments/            # MODAL_PROFILE=chromatic

# offline, no GPU: the policy suite (P-1..P-12 + L-1..L-8)
PYTHONPATH=. python3 rhm/practice/maestro/policy.py

modal run rhm/practice/ostinato/ostinato.py::preflight \
    --arms "perf_given,perf_given_rho,os_log,os_rawx,os_r100,os_r020,os_r004,os_r001"
modal run rhm/practice/ostinato/ostinato.py::fidelity_smoke --tag os_gf

# the main run. Every flag but --arms is in_s0's (hence tc_s0's, hence cr3_s0's), verbatim;
# n_pr / pr_width are left at their ABUNDANCE defaults (64 / 16) on purpose — the rung, not
# the supply, is the variable. The floors are the defaults (A1's measured values) and are
# asserted, so no --tol-* literals are passed.
python3 rhm/practice/ostinato/launch_detached.py --fn ostinato_run --tag os_s0 \
    --eras "1:25:60,2:12:50,3:6:70,4:3:12,5:1:9" --era-caps "60,50,70,12,9" \
    --arms "os_log,os_rawx,os_r100,os_r020,os_r004,os_r001" \
    --max-macro-level 4 --endo-price 267 --gate-frac 0.5 \
    --span-tau-fire 0.50 --perf-alpha 0.05 --perf-tau-w 0.20 \
    --budget 8 --g-budget 482 --n-corrupt 1 --mine-cap 8 --gy-level 4 --entry-rec \
    --span-min-hold 128 --collect-task-matched --tm-episodes 8192 \
    --recert-every 5 --n-aud 192 --probe-every 8 --n-rt 384 --n-score 256 \
    --sil-cv 0.15 --lp-min-drop 0.10 --seed 0

python3 rhm/practice/ostinato/analyze_ostinato.py --tag os_s0 --gf-tag os_gf --fetch --figures
```

## Instrument notes inherited from the donor (they bound what this node can say)

- The four **L4 corridors are trivial** (`e` ≡ 0.0000 exactly), so all live performance error is
  L2/L3 and the meter cannot bite where the corridor is degenerate. `b(s)` at L4 is a constant
  zero and the rung cannot move it.
- The **agency gate's calibration is degenerate** (`g₀ = 0`) in most arms — the ACT/PLAYBACK
  exclusion is structural, not gate-carried.
- The ladder's arms are **clock-yoked**, so pacing is pinned to `os_log`'s optimum; reduction
  §H replays A1's thermostat on each arm's own logged L4 at-support series and reports when
  each arm's *own* gauge would have fired.
- Ranks, signs and multiples of measured floors are the claims.

[^private]: Not mirrored: this link points to a document in the private lab repo (the roadmap, the queue, an unrun spec, reading notes, or a conversation). See the top-level README for what is held back and why.
