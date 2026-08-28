# FILES — `conductor` (A1: the outer loop drives the crank)

Machinery record for the node. **No results are interpreted here** — the reduction's numbers go
to the orchestrator and are discussed before any writeup (repo norm).

**Up**: parent arc [`../README.md`](../README.md) · `../../../../ROADMAP.md`[^private] §4.1
**Direct donors** (untouched): [`../assay/`](../assay/FILES.md) (`assay.py` forked; `as_s0` is
the cross-tag replay reference and, with `cs_s0`/`as_s1`, the source of the measured floors) ·
[`../census/`](../census/README.md) (findings 4, 6, 7 — the yoke mechanic and the stream floor) ·
[`../spiral/`](../spiral/README.md) (the three clocks, the battery) ·
[`../teacher_slot/decision/policy.py`](../teacher_slot/decision/policy.py) and
`../teacher_slot/endo_yield/SPEC.md`[^private] (the outer loop being
ported, the null-ABBA floor method, and the shadow-panel pattern).

## Code files

| file | purpose |
|---|---|
| `policy.py` | The outer loop, **outside the Modal app** so every rule is auditable and gate-able with no GPU and no substrate. `QuietPolicy` (the thermostat), `SchedulePolicy` (the fidelity carrier), `YokePolicy` (clock replay), `ReadLedger` (ear's pricing convention), `summarise_trace`, `null_abba`, and `policy_gate()` — 12 offline checks, P-1…P-12. |
| `floors.py` | The **measured dead zones**, derived offline from logs already on disk (`as_s0`, `cs_s0`, `as_s1`) by the null-ABBA method, plus the span sweep that chose the horizon and a Phase-0 replay of the rule over the donor's own gauge series. Writes `floors.json`. No GPU, no Modal. |
| `floors.json` | The derivation's output: floors, provenance, span sweep, phase-0 replay. `floor_gate()` asserts the literals in `conductor.py` still match it. |
| `conductor.py` | The substrate. Forks `../assay/assay.py` verbatim; five additions, each marked `# [conductor]`. Entrypoints: `preflight`, `fidelity_smoke`, `conductor_run`. |
| `analyze_conductor.py` | The reduction. `--floors` prints only the in-tag null-ABBA floors (how the endo dead zone is measured on the smoke tag); the full run prints gates, action traces, gauge-vs-yoke, the three clocks, π, the ledger, the shadow panel, and the battery. |
| `launch_detached.py` | Session-isolated detached launcher (`handle/`'s, retargeted). |

## What the fork adds (each `# [conductor]`-marked)

| addition | where | why |
|---|---|---|
| **the loop owns the crank's actions** | `run_arm` blocks (d2), (g) `commit=="loop"`, (g4); the era `for` → `while` with a cap | `commit/hold` per level and `era advance` pass from the certificate and the ladder to a rule reading one scalar. The era *sequence* is the world and does not move. |
| **the observation panel** | `run_arm`, `obs_miners` / `obs_hist` | measured reason, not a preference: `gauge_hist[3]` is **identically 0 through all 48 cycles of era 1** in `as_s0/anchor`, because the committable miners are gated to `era_level+1`. The one-level-up gauge an era-1 loop needs does not exist in the donor. The panel only observes; its level-4 entry is asserted equal to the donor's G-Y miner every cycle (`obs_gy_agree`). |
| **the endo read** | `parent_span_blocks`, `cell_span_blocks`, `endo_read`, `endo_gate`, `endo_bench` | `endo_yield`'s label-free one-level-up loss, ported to a masked-block plant. |
| **the shadow panel + the ledger** | `run_arm` (d2), `policy.ReadLedger` | every gauge in every arm, uncharged; the ledger charges only the policy's input stream. |
| **the yoked arms** | `conductor_run`'s arm loop, `policy.YokePolicy` | `census`'s mechanic, widened to both actions. |

## The rule: what it is, and what it is **not**

Stated here because the estimator is the round's real design decision.

The donor (`PairedPolicy`) contrasts two **reversible** conditions lived in adjacent intervals in
an ABBA block, so any improvement rate linear in time cancels exactly. **This round's actions are
absorbing** — a commit freezes a table into the ratchet, an era advance moves the damage cell
down and there is no way back — so there is no C-K-K-C over "committed / not committed".

Two ports were available. **(a) trial-commit** — install the candidate table for the C quarters
and remove it after — keeps the donor's contrast verbatim and was rejected on a substrate fact:
during a C quarter the arm mines the next level over a committed lower table and the plant trains
on trajectories that used the macro, and neither can be taken back when the table is removed, so
the "reversible" block leaves a permanent residue of exactly the treatment it is measuring.

**(b) is what is implemented.** The donor's block exists to cancel the ambient improvement trend,
which is a nuisance when the question is "which of two conditions is better". For an absorbing
action the question is "has the currency this level mints stopped moving" — **the ambient trend
is the signal**, and there is no second condition to cancel it against. So the estimator becomes
`D = mean(u)` over the block's quarters (the paired-interval improvement rate), `V` is its
recency-weighted average, and the rule is **hold while `V > v_tol`, act when `V <= v_tol`**.

Everything else in the donor ports unchanged: one scalar read per arm in error convention (so
arms differ in nothing but the read); a scale-free sign decision against a **measured** dead zone;
the 5-read block geometry; full determinism (no policy draws one RNG value); the priced ledger;
and "positive and then flat" — the loop may not act on a gauge that has never cleared its own
floor, which is the certificate's `lp_min_drop` re-expressed in floor units, adding no constant.

**The donor's contrast survives as the instrument's own noise meter.** `N = mean(u over C) −
mean(u over K)` is computed at every block, in every arm, and never driven; on a fixed-condition
series its true value is exactly zero, so its live spread is the in-tag floor. The conversion is
exact: for iid quarter improvements, `Var(N) = σ²` and `Var(D) = σ²/W`, so `v_tol = sd(N)/2` at
W=4 (`policy_gate` P-1 checks the cancellation, P-2 the variance relation).

**Said plainly: this is not an ABBA paired trial.** It is a thermostat on a paired-interval slope
with a measured dead zone, and its nearest ancestor in this repo is the census's G-A rule, which
finding 4 measured to be *indistinguishable from a clock*. That is exactly why every gauge-driven
arm carries a yoked-clock control and why the reduction reports gauge-vs-yoke divergence first.

## The reads (all in ERROR convention, lower is better)

| read | what it is | level | price |
|---|---|---|---|
| `ledger` | the arm's own metering error on the era's own cell | the active level | **free** — the learner's own experience, already computed this cycle |
| `yield` | `−(distinct level-(active+1) tuples at support)` from the observation panel | `active+1`, **clamped to 4** above that (level 5 has no instrument at any affordable budget) — logged per cycle as `read_level` | **free** — mined by a numpy counter from trajectories the agent produced anyway |
| `endo` | the plant's own masked-infill NLL over the span **one level above the era's damage cell**, whole span masked and whole span scored | `era.level+1` | **priced** at its measured cost (`endo_bench`), folded into `counts["ground"]` → `t_cum` |

Also logged, never driven: `endo_cell` (the same read on the cell's *own* span — the
level-matched within-level control) and `yield_active`.

**Why masking the whole parent span is the one-level-up read.** With the parent span entirely
unobserved, recovering it requires the level-(ℓ+1) latent inferred from *outside* the span; masking
only the cell requires the level-ℓ latent. Which span that is comes from `(era.level, era.node, s,
depth)` — the grammar's *shape*, which the agent has, and the era's own cell, which it lives in —
and the target is `bottom_map`'s level-1 features, the plant's own training objective in every
arm. Gate N-1 asserts this by parsing the read's code (docstrings stripped) and checking that none
of `truth / rules / inverse_maps / committed / canon / true_mask / grade` appears in what executes.

**Scope note, stated not hidden**: at era 5 the cell is `L5n1` and its parent is the root, so the
endo read there is the unconditional block marginal. Logged as `parent_is_root` in the gate.

## The measured floors (`floors.py`; span 1, W 4; 574 pooled windows per series)

Sources: `as_s0{anchor,complete,junk_dose,strip}` + `cs_s0{spiral_route,census_extend}` +
`as_s1{exact_j}`. Windows containing an era boundary or a commit cycle are dropped (a regime
change is not noise). `mean(N)` came out at −0.003 / −0.003 / +0.021 against spreads of 0.059 /
0.474 / 0.732 — the null construction checking itself.

| series | sd(N) | **v_tol = sd(N)/2** | mean(D) | D/floor | frac(D > tol) |
|---|---|---|---|---|---|
| `ledger` | 0.0588 | **0.029421** | +0.0104 | 0.35 | 0.10 |
| `yield` @ L3 | 0.4736 | **0.236800** | +0.3071 | 1.30 | 0.46 |
| `yield` @ L4 | 0.7322 | **0.366111** | +0.5470 | 1.49 | 0.56 |
| `endo` | — | measured in-tag on the smoke tag's schedule arm | — | — | — |

**The span choice is measured too.** A longer horizon raises every gauge's signal-over-floor
(yield L4: 1.49 → 2.22 → 2.74 at spans 1/2/3) but needs a `4·span+1` window, and the consumption
eras are 12/9/7 cycles. **span 1 is the only horizon with decision points in every era**; the
sweep is in `floors.json`.

**`endo` has no offline series** — it needs forward passes through a live plant — so it is
measured on `cd_smoke`'s schedule arm (fixed-condition by construction, `endo_yield`'s `no_wall`
pattern) and passed in as `--tol-endo`. The reduction re-derives every floor in-tag on the main
run's own `anchor` beside the offline ones that governed it.

## Phase 0 — the rule replayed over the donor's own series, before any GPU

`floors.replay_rule` (armed = the gauge cleared its own floor; quiet = it then fell back inside
it). This does not predict what the loop will do — an action changes everything downstream — but
it answers the two questions that size the round.

| series | era 1 (48c) | era 2 (40c) | era 3 (12c) | era 4 (9c) | era 5 (7c) |
|---|---|---|---|---|---|
| `ledger` | armed c5, quiet c10 | armed c75, quiet c76 | never armed | never armed | never armed |
| `yield` L3 | — (dead in the donor, see the panel) | armed c53, never quiet | armed c93, quiet c95 | armed c105, quiet c108 | armed c114, never quiet |
| `yield` L4 | armed c21, quiet c24 | armed c53, never quiet | armed c93, never quiet | armed c105, never quiet | armed c114, never quiet |

**Scope note**: `as_s0` carries no observation panel, so its level-3 at-support series is
identically zero through era 1 — this replay is silent about era 1 for the yield read, which is
the gap the panel exists to close; the L4 row is shown beside it as the proxy.

## Arms (single seed, seed family = the donors')

All routing-only (`prop_k=4`, no span port), all with the recert, all on the anchor's torch
stream — so **every loop arm is bit-identical to the anchor until its own first action**, which
is the in-tag twin gate.

| arm | driver | read | role |
|---|---|---|---|
| `anchor` | certificate-else-boundary, fixed ladder | — | the scheduled crank; the cross-tag replay carrier |
| `outer_yield` | the loop | `yield` (free) | the treatment |
| `yoked_yield` | clock | — | `outer_yield`'s realised cycles, replayed |
| `outer_endo` | the loop | `endo` (priced) | the label-free treatment |
| `yoked_endo` | clock | — | `outer_endo`'s realised cycles, replayed |
| `outer_ledger` | the loop | `ledger` (free) | the within-level reader |

Arm **order is load-bearing**: the anchor first (nothing is readable without the carrier); each
gauge arm immediately before its yoke (the yoke's cycles are the gauge arm's *measured* ones,
passed in at runtime); `outer_ledger` last as the most cuttable.

**The loop replaces the certificate; it does not conjoin with it.** §4.1 reads the outer loop as
taking the certificate's seat, and conjunction is what `census_gate` did — finding 4 is that the
conjunction contributed nothing beyond a cycle number. The shadow certificate keeps running
read-only in every arm, so cycles-to-cert stays observable beside what the loop actually did. A
loop commit is **not** provisional: the loop chose it, so it pays the pre-commit audition.

**On what the yokes can and cannot show.** A read here has no channel into the trajectory (fixed
batches, `no_grad`, no RNG), so bit-identity between a gauge arm and its yoke is *structurally
expected* rather than surprising. What the yoke buys is therefore three things, all real: it is a
live check that the policy is non-invasive (a divergence would mean it is not); it makes "the
criterion contributed nothing beyond a cycle number" checkable in this round rather than assumed;
and the priced-time difference is what the read cost for a trajectory a clock reproduces exactly.

## Caps

The era sequence is fixed; what the loop owns is when to leave. `CONDUCTOR_CAPS = 60,50,15,12,9`
(= 146) is 1.25× the scheduled ladder 48/40/12/9/7 (= 116): room to hold a quarter longer than
the schedule in every era, while six arms stay inside ~2.6 GPU-h at `cs_s0`'s measured 10.4
s/cycle. **A refusing arm rides its caps to the end and terminates — that is a readout, not a
hang.** An arm that advances early simply costs less. A loop commit the empty-table guard cancels
counts as an action for the loop's clock, so the rule must re-arm before licensing another.

## Gates

| gate | what | where | status |
|---|---|---|---|
| P-1…P-12 | the donor's policy-gate suite re-asked: trend cancellation (7.1e-15), the `sd(N)/sd(D)=2` variance relation the dead zone is derived through, mirror symmetry under relabelling, dead zone on a null series, positive-then-flat, reset on action, determinism + RNG neutrality, read key is a pure label, ledger arithmetic, no defaulted floor, yoke replays by clock, span is the horizon axis | `policy.policy_gate()` — offline, no GPU | **ALL PASS** |
| N-1 | the endo read is label-free **by construction**: positions depend only on `(era.level, era.node, s, depth)`, the parent contains the cell and is exactly `s`× as wide in all five eras, and no oracle name appears in the read's executed code | `conductor.endo_gate()` | **PASS** |
| N-2 | the endo read's price, measured on the same GPU rather than assumed | `conductor.endo_bench()` | run at setup, in `setup.json` |
| F | the floors that govern a run are present, positive, and equal to `floors.json`'s derivation | `conductor.floor_gate()` | **PASS** |
| G-F smoke | with the loop off **and the shadow panel ON**, this fork replays `assay.py` in-process; the panel moves nothing (`endo_yield`'s transparency gate T folded in) | `conductor.py::fidelity_smoke` | see Runs |
| G-F full | `cd_s0/anchor` vs `as_s0/anchor` over all 116 cycles, cross-tag | `analyze_conductor.py` §0 | see Runs |
| twin | every loop arm vs the anchor up to its **own first action** | `analyze_conductor.py` §0 | see Runs |
| panel | the observation panel's level-4 entry equals the donor's G-Y miner on every cycle | in-run `obs_gy_agree` | see Runs |
| preflight | every call signature and every new branch — a chosen commit, a chosen advance, a capped advance, a yoked replay of each — at toy sizes before a paid setup. Its tolerances are 1e-9 dust and **no number from it is a measurement** | `conductor.py::preflight` | see Runs |

## Runs

| tag | what | outcome |
|---|---|---|
| `_preflight` | interface + every loop branch, toy sizes | PASS (yoke replay `actions_equal`, `max_abs_delta` 0.0; loop arms inert before first action; obs≡G-Y) |
| `cd_gf` | G-F, in-process vs `assay.py`, panel ON | PASS — 0.000e+00, control 0.000e+00, commits equal |
| `cd_smoke` | `conductor_run --quick`, six arms | mechanics verified; **cannot calibrate gauges** (nothing minable at `--quick`; its endo floor was plant-thrash — surfaced the phantom-commit reporting bug, fixed) |
| `cd_ef` | the anchor alone at **full config** (superseding the smoke as floor source) | the null-ABBA floors that governed `cd_s0`; cross-tag replay of `as_s0/anchor` 0.000e+00 over 116 cycles, commits identical |
| `cd_s0` | **the main run**, six arms, measured floors | clean; findings in [README.md](README.md) |

## Reproduce

```bash
cd experiments/            # MODAL_PROFILE=chromatic

# offline, no GPU: the floors, the span sweep, the phase-0 replay, the policy suite
PYTHONPATH=. python3 rhm/practice/conductor/floors.py
PYTHONPATH=. python3 rhm/practice/conductor/policy.py

modal run rhm/practice/conductor/conductor.py::preflight
modal run rhm/practice/conductor/conductor.py::fidelity_smoke --tag cd_gf
python3 rhm/practice/conductor/launch_detached.py --fn conductor_run --tag cd_smoke --quick \
    --n-probe-clean 512 --gen-steps 20 --era-cycles 14
python3 rhm/practice/conductor/analyze_conductor.py --tag cd_smoke --fetch --floors   # -> tol_endo

python3 rhm/practice/conductor/launch_detached.py --fn conductor_run --tag cd_s0 \
    --eras "1:25:48,2:12:40,3:6:12,4:3:9,5:1:7" --era-caps "60,50,15,12,9" \
    --arms "anchor,outer_yield,yoked_yield,outer_endo,yoked_endo,outer_ledger" \
    --tol-endo <measured on cd_smoke> --endo-price <measured by endo_bench> \
    --budget 8 --g-budget 482 --n-corrupt 1 --mine-cap 8 --gy-level 4 --entry-rec \
    --span-min-hold 128 --collect-task-matched --tm-episodes 8192 \
    --recert-every 5 --n-aud 192 --probe-every 8 --n-rt 384 --n-score 256 \
    --sil-cv 0.15 --lp-min-drop 0.10 --seed 0

python3 rhm/practice/conductor/analyze_conductor.py --tag cd_s0 --fetch --figures
```

[^private]: Not mirrored: this link points to a document in the private lab repo (the roadmap, the queue, an unrun spec, reading notes, or a conversation). See the top-level README for what is held back and why.
