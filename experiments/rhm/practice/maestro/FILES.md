# FILES — `maestro` (A2: does *learning* the rule buy anything the thermostat did not?)

Machinery record for the node. **No results are interpreted here** — the reduction's numbers go
to the orchestrator and are discussed before any writeup (repo norm).

**Up**: parent arc [`../README.md`](../README.md) · `../../../../ROADMAP.md`[^private] §4.1 (second shape, A2)
**Direct donor** (untouched): [`../conductor/`](../conductor/FILES.md) — A1. `conductor.py` and
`conductor/policy.py` are forked here; `cd_s0` is the offline corpus, the floor source, and the
cross-tag replay reference; `cd_ef` is A1's floor calibration.
**Through it**: [`../assay/`](../assay/FILES.md) (the substrate) ·
[`../teacher_slot/decision/policy.py`](../teacher_slot/decision/policy.py) (the outer loop, and
the round-1 postmortem this round has to answer) · [`../census/`](../census/README.md) (the yoke
mechanic, findings 4 and 7) · [`../spiral/`](../spiral/README.md) (the three clocks, the battery)
· [`../../directed_sculpting/full_loop/composed_loop/`](../../directed_sculpting/full_loop/composed_loop/README.md)
(the negative control's lineage: the one learned endogenous judge ever built, rewarded
within-level, had no upward direction).

## The question, and the one thing that varies

A1 measured that a **hand-written** thermostat reading the learner's own one-level-up currency
drives the crank as well as the schedule inside the earnable range and better beyond it. A2 asks
the roadmap's next question: **does *learning* the rule buy anything the thermostat did not?**

Same action set, same substrate, same ladder, same caps, same reads, same measured floors. The
rule becomes a small learned policy over (gauge readings, level state) whose reward is
**next-level yield**, graded against two things at once:

| arm | the rule | role |
|---|---|---|
| `anchor` | certificate-else-boundary, fixed ladder | the scheduled crank; the in-tag carrier and a cross-tag replay gate |
| `outer_yield` | A1's **hand-written thermostat** on `yield` | **the comparator**, and the round's strongest fidelity gate |
| `learned_yield` | the learned class, reward = **next-level yield** | the treatment |
| `learned_task` | the learned class, reward = **within-level** | the negative control (`composed_loop`'s question, in the crank's action space) |
| `yoked_learned` | `learned_yield`'s realised cycles, by clock | the non-invasiveness check on the NEW machinery |

`learned_yield` and `learned_task` share the policy class, the inputs, the fitting procedure, the
floors, the priced read and the absence of exploration. **They differ in the reward's type and in
nothing else** — `fit_gate()` asserts that on the literals before the run starts.

## Code files

| file | purpose |
|---|---|
| `policy.py` | The outer loop, **outside the Modal app** so every rule is auditable with no GPU. A1's `SchedulePolicy` / `YokePolicy` / `QuietPolicy` / `ReadLedger` / `summarise_trace` / `null_abba` byte-for-byte, plus **`LearnedPolicy`** and gates **L-1…L-8**. `policy_gate()` now runs 20 offline checks (P-1…P-12 + L-1…L-8). |
| `fit.py` | **The offline phase.** Builds the corpus from A1's logged arms, fits both twins, measures each mixture's dead zone by the donor's own null method, replays both counterfactually over every logged arm, and judges whether the live contrast is non-vacuous. Writes `fit.json`. No GPU, no Modal, no substrate. |
| `fit.json` | The fit's output: mixtures, thetas and their provenance, per-bucket diagnostics, gap sensitivity, the counterfactual replays, and the vacuity verdict. `fit_gate()` asserts the literals in `maestro.py` still match it. |
| `floors.json` | A1's measured dead zones, copied unchanged. They are reused here for two jobs at once: as dead zones for the thermostat arm, and as the **normalising units** of the learned mixture. |
| `maestro.py` | The substrate. Forks `../conductor/conductor.py`; every addition marked `# [maestro]`. Entrypoints: `preflight`, `fidelity_smoke`, `maestro_run`. |
| `analyze_maestro.py` | The reduction. A1's, plus §1b — the fitted mixtures, the offline fit diagnostics, the learned-vs-thermostat action diff, and what the policy read at each of its own actions. |
| `launch_detached.py` | Session-isolated detached launcher (A1's, retargeted). |

## What the fork adds (each `# [maestro]`-marked)

| addition | where | why |
|---|---|---|
| **`LearnedPolicy`** | `policy.py` | the rule as a learned gauge mixture indexed by level state; see below |
| **the fitted policies as literals** | `maestro.py` `FITTED`, `PO_FIT_UNITS` | the policy that governed a run belongs in that run's own `setup.json`, not only in a script — `floors.py`'s pattern, applied to the object A2 adds |
| **`fit_gate()`** | `maestro.py` | `FITTED` must equal `fit.json`; both thetas positive; the twins must share gauges, buckets and geometry and differ only in their numbers; the run's floors must be the units the mixture was fitted in |
| **level state in the panel** | `run_arm` (d2) — `will_commit`, `n_committed`, `active_level` | the learned policy's table index, evaluated *before* block (g), which is what keeps the offline fit and the live policy in the same state |
| **three arms + their stream keys** | `ARMS`, `TWIN` | `learned_yield`, `learned_task`, `yoked_learned`, all on the anchor's stream |
| **the fit reaches the arm** | `run_arm` loop build | a learned arm is handed its fitted policy by reward; the two learned arms reach that line with identical specs but for `reward` |
| **the bucket at every action** | the two `loop_actions.append` sites | so the reduction can print what the policy weighted and read at each action |
| **G-F retargeted** | `fidelity_smoke` | this fork's direct donor is `conductor.py`, not `assay.py` |

## The learned rule: what it is, and what it is **not**

Stated here because, as in A1, the estimator is the round's real design decision.

**The form.** A1's thermostat is `hold while V > v_tol` for one hand-picked read's
recency-weighted paired-interval slope against that read's measured dead zone. Write the slope in
**floor units**, `V_k / tol_k`, and A1's rule is exactly `hold while V_yield/tol_yield > 1`. The
learned generalisation is a **mixture over the floor-normalised slopes**:

```
Vhat(t) = sum_k a[s(t)]_k * V_k(t) / tol_k(t)        the decision statistic, in floor units
theta[s]                                              its own MEASURED dead zone
rule:  HOLD while Vhat > theta ;  ACT when Vhat <= theta
```

`a[s]` is a unit-norm weight vector over `(ledger, yield, endo_excess)` per level-state bucket
`s`, and `theta[s]` is measured on a fixed-condition reference series by **the donor's own
null-ABBA method carried through the mixture** (`Nhat = sum_k a_k N_k / tol_k` has true value
zero under any improvement rate linear in time, so `theta = sd(Nhat)/2` at W=4, which is A1's
`v_tol = sd(N)/2` exactly).

**Three properties this form buys, each load-bearing.**

1. **It contains the thermostat exactly.** `a = e_yield` gives `Vhat = V_yield/tol_yield` and
   `theta = 1`, i.e. `V_yield <= tol_yield` — A1's rule, decision for decision. Gate **L-1**
   asserts this against the donor class on a live series (0 disagreements over 401 decisions with
   21 firings); gate **L-2** asserts the floor construction reduces to `null_abba`'s own
   `sd(N)/2`; gate **F-1** asserts the whole floor machinery reproduces A1's *measured in-tag
   numbers* on A1's own reference arm (82/82 windows; `sd(N)` to 0.007%). So "does learning the
   rule buy anything" is asked **inside one hypothesis class**, with the hand-written rule as a
   named point in it.
2. **It is scale-free**, so a fit's shrinkage cannot make the arm inert: `Vhat` and `theta` are
   both linear in `a`, so only its direction is the policy (gate **L-3**).
3. **It reads level state.** `a` and `theta` are a table over `will_commit = (active <= maxl and
   committed[active] is None)` — whether a firing would **install a table** or **move the world**.
   A1's thermostat runs one rule for both actions and cannot tell them apart. A bucket change is
   treated as a regime change and resets the latch, for the donor's own stated reason (gate
   **L-8**); live this only ever coincides with an era start or the policy's own commit, so it
   moves no trajectory — it is what makes the *offline replay* over another arm's series honest.

**Said plainly: this is not a bandit and there is no exploration.** It is A1's thermostat with a
fitted gauge mixture and a measured dead zone, frozen before the run starts.

### Two forms were tried and the first was rejected offline

Recorded because the failure is a property of the form, not of the corpus.

**(a) A fitted VALUE thresholded at its own noise** — `Vhat = b + w.x` over gauge slopes *and*
level-state features, ridge with centring and an intercept, `theta` from the null. Rejected: ridge
shrinkage pulls the prediction to its mean (0.87 tuples/cycle) which never comes near any noise
floor, so `learned_yield` would never have acted; and dropping the intercept to fix that leaves a
statistic whose **zero point is arbitrary**, so the two twins' statistics sat at unrelated
locations and were not being compared on the same axis at all. Measured, before any GPU.

**(b) What is implemented**: ridge **through the origin** on floor-normalised features with the
target in the *same* floor units. Then `c . V` is an estimate of the forward reward rate itself,
its zero **is** the reward's zero, A1's semantics survive intact, and the degenerate point stays
exact (regressing the forward yield rate on the trailing yield rate through the origin returns a
coefficient of 1 — the thermostat).

## The fit (`fit.py`), and what its n supports

**Why offline.** Both actions are absorbing (A1's `FILES.md` records why a reversible
trial-commit is unsound), so a run contains one or two commit events and five era exits and
**within-run exploration of commit timing is close to impossible**. The fit runs on the corpus of
*previous turns of the crank* and the policy is frozen — meta-learning across turns rather than
within one.

**The corpus.** The four **distinct** trajectories in `cd_s0`: `anchor`, `outer_yield`,
`outer_endo`, `outer_ledger` (the two yoked arms are bit-identical replays and are excluded rather
than double-counted; `cd_ef/anchor` replays `cd_s0/anchor` at 0.000e+00 and is the floor
*reference*, not extra data). Between them: L2 commits at c10/16/18/49, L3 at c69/92-or-never,
era-1 exits at c18/48/56/60.

**What that n does and does not support**, stated in the module docstring and repeated in the
reduction:

- a **policy-gradient** fit would be estimating a return differential from **n = 6 commit
  events**, which this corpus cannot support and no reversible trial would grow;
- what the corpus supports densely is the **value** — every arm logs its gauge readings and its
  next-level yield every cycle, giving **257 windows** (165 in the commit bucket, 92 in the
  advance bucket) across four trajectories.

So the learned object is the value, and the action rule is greedy with respect to it against a
**measured** dead zone — the same device A1 used, and one that needs no counterfactual returns.
**Nothing here estimates the value of acting.** That is the round's honest limit.

**The regression**, identical for both twins and differing only in the target:

```
target(t) = ( R(t+GAP+H) - R(t+GAP) ) / H  / tol      the forward reward RATE, in floor units
  learned_yield : R = at_support one level up          NEXT-LEVEL YIELD   / tol_yield
  learned_task  : R = -(the arm's own metering error)  WITHIN-LEVEL      / tol_ledger
a[s] = normalise( ridge_through_origin( target ~ [V_ledger, V_yield, V_endo_excess] | s ) )
```

- **`GAP = 1` is not a tuned knob**: it is the minimum gap for which the trailing window and the
  forward window share no data point. Without it the reading at `t` enters the features and the
  target with opposite signs and one point's noise manufactures a correlation on its own — the
  round-1 postmortem's hazard class (an instrument measuring itself), designed out before the GPU
  rather than diagnosed after it. `fit.json`'s `gap_sensitivity` records the fits at GAP 0/1/2.
- **`H = W*span = 4`** is the donor's block geometry, so the target is the same kind of object the
  features are.
- **lambda by leave-one-ARM-out** over a fixed grid — the only cross-validation four trajectories
  admit. Folds are whole trajectories, never shuffled cycles: windows inside one arm are serially
  dependent and a random split would leak a neighbour of every held-out point into training.
- **windows dropped**: any window containing an era boundary or a commit cycle, across the
  trailing window, the gap and the forward window alike (A1's `floors.py` convention).
- **two window sets, on purpose**: the fit needs a clean trailing *and* forward window; the floor
  needs only a clean trailing one, because `Nhat` is a property of the trailing block alone.

## The round-1 postmortem, answered

`teacher_slot/decision/policy.py::PairedPolicy` records that round 1 **measured its own
instrument**: the learning curve's first-interval improvement was 92–97× ambient, the decaying
baseline needed ~13 decision points to come within 2× of it, and since an untried action was only
ever reached by a scheduled trial the untried action was *always* sampled second and systematically
favoured. Three arms reading three different quantities committed at the identical step.

A2 answers this **structurally, not with burn-in**: there is no within-run learner. The policy is
fitted on a fixed offline corpus and frozen before the run starts, so there is **no learning curve
whose slope can swamp an action differential, no baseline to decay, and no sampling order to
bias** — the three mechanisms that produced round 1's signature. Nothing in `LearnedPolicy.step`
consumes an RNG draw or updates a parameter (gate L-6). The one place the hazard *could* have
re-entered — the offline regression sharing a data point between its features and its target — is
the `GAP` above, caught before the GPU.

## The reads and their prices

Unchanged from A1, except that a learned arm's input stream is **all three gauges at once**, so it
pays the priced one. `ledger` and `yield` are free (the learner's own experience and a numpy
counter over trajectories it produced anyway); `endo_excess` is a forward pass the arm does not
otherwise make and is priced at A1's measured **267 g/read**. **Both twins consume the identical
stream and therefore pay the identical price**, so the ledger column cannot separate them either.

## The floors

**Reused, not re-measured** — the brief's rule (reuse `cd_ef`'s measured floors wherever the same
statistic is thresholded), and here they carry a second job: they are the mixture's **normalising
units**, so they must be the very numbers `fit.py` fitted against or a learned weight means
something else. `maestro_run` therefore gives `tol_endo` a default (A1's measured value) where A1
deliberately had none, and asserts all four against `PO_FIT_UNITS`.

The **one new statistic** the learned policy thresholds is `Vhat`, and it gets its **own measured
floor** by the null method, offline, on the fixed-condition reference arm — per bucket.

## Gates

| gate | what | where | status |
|---|---|---|---|
| P-1…P-12 | A1's suite, re-asked of the forked module | `policy.policy_gate()` — offline | **ALL PASS** |
| **L-1** | **the learned class contains the thermostat exactly** — `a = e_yield`, theta = 1 agrees with the donor `QuietPolicy` decision for decision | `policy.policy_gate()` | **PASS** (0 disagreements / 401 decisions, 21 firings) |
| **L-2** | the floor construction reduces to the donor's `null_abba` `sd(N)/2` | `policy.policy_gate()` | **PASS** (rel diff 0.0) |
| L-3 | scale-freeness in the mixture | `policy.policy_gate()` | **PASS** |
| L-4 | an unvisited table cell is inert | `policy.policy_gate()` | **PASS** |
| L-5 | reset on action reaches every sub-estimator | `policy.policy_gate()` | **PASS** |
| L-6 | determinism + RNG neutrality of the learned rule | `policy.policy_gate()` | **PASS** |
| L-7 | no defaulted fit, no defaulted/non-positive theta | `policy.policy_gate()` | **PASS** |
| L-8 | a bucket change resets the latch | `policy.policy_gate()` | **PASS** |
| **F-1** | **the floor machinery reproduces A1's own in-tag numbers** on A1's own reference arm, window count included | `fit.gate_f1()` — offline | **PASS** (82/82 windows; `sd(N)` within 0.007%) |
| vacuity | the two fitted twins must not fire identically on every logged arm, and at least one must ever act — otherwise halt instead of launching | `fit.main()` verdict, re-asserted by `fit_gate` | **PASS** (non-vacuous) |
| fit | `FITTED` equals `fit.json`; twins share class; run floors are the fit's units | `maestro.fit_gate()` | **PASS** |
| N-1, N-2, F | A1's endo label-freeness, endo price, floor gate | `maestro.py` | as A1 |
| G-F smoke | with learning off and the panel ON, this fork replays **`conductor.py`** in process | `maestro.py::fidelity_smoke` | see Runs |
| **cross-tag ×3** | `ma_s0/anchor` vs `cd_s0/anchor`; **`ma_s0/outer_yield` vs `cd_s0/outer_yield`**; `ma_s0/anchor` vs `as_s0/anchor` | `analyze_maestro.py` §0 | see Runs |
| twin | every loop arm vs the anchor up to its own first action | `analyze_maestro.py` §0 | see Runs |
| yoke | `yoked_learned` vs `learned_yield` over every trajectory series | `analyze_maestro.py` §2 | see Runs |
| preflight | every call signature and every new branch — a learned commit, a learned advance, a capped advance, the bucket switch, the ledger's three-gauge charge, and the yoke handoff **from a learned arm** — at toy sizes before a paid setup. Its tolerances are dust and **no number from it is a measurement** | `maestro.py::preflight` | see Runs |

**The `outer_yield` cross-tag replay is the round's strongest fidelity statement.** Nothing
upstream of A1's thermostat changed, so it must reproduce `cd_s0/outer_yield` bit for bit over all
139 cycles with the learned class present in the file and the level-state keys present in the
panel. If it does not, the comparator has moved and no learned-vs-thermostat difference in this tag
means anything.

## Caps and arm order

Caps are A1's: `60,50,15,12,9` (= 146), 1.25× the scheduled ladder. A refusing arm rides its caps
to termination — a readout, not a hang.

Arm **order is load-bearing**: `anchor` first (the carrier), `outer_yield` second (the comparator,
and the arm the whole tag's fidelity rests on), then the two learned twins, then `yoked_learned`
last — it replays `learned_yield`'s *measured* actions, so it must run after it, and it is the
most cuttable arm if the budget presses.

## Runs

| tag | what | outcome |
|---|---|---|
| `_preflight` | interface + every learned branch, toy sizes, 16 arms | PASS |
| `ma_smoke` | `maestro_run --quick`, five arms | mechanics only; **cannot calibrate anything** |
| `ma_s0` | **the main run**, five arms, A1's caps and floors | clean; findings in [README.md](README.md), full record `figures/ma_s0/reduction.txt` |
| `ma_s1` | stream-displaced twins of `outer_yield` and `learned_yield` (`--ref-tag ma_s0`, 1024-draw burn, refs asserted identical in-job and at merge) | the displacement floors and the rank cells — §9 of `reduction.txt` |

Volume `rhm-scaling-data:/data/rhm_practice_maestro/<tag>/`; fetched copies and figures under
`figures/<tag>/`.

## Reproduce

```bash
cd experiments/            # MODAL_PROFILE=chromatic

# offline, no GPU: the policy suite (P-1..P-12 + L-1..L-8)
PYTHONPATH=. python3 rhm/practice/maestro/policy.py
# offline, no GPU: the corpus, both fits, the measured thetas, gate F-1, the counterfactual
# replays, and the vacuity verdict -> fit.json
PYTHONPATH=. python3 rhm/practice/maestro/fit.py

modal run rhm/practice/maestro/maestro.py::preflight
modal run rhm/practice/maestro/maestro.py::fidelity_smoke --tag ma_gf

python3 rhm/practice/maestro/launch_detached.py --fn maestro_run --tag ma_smoke --quick \
    --quick-cycles 12 --quick-probe-clean 512 --quick-gen-steps 20 \
    --arms "anchor,outer_yield,learned_yield,learned_task,yoked_learned" \
    --endo-price 267 --seed 0

python3 rhm/practice/maestro/launch_detached.py --fn maestro_run --tag ma_s0 \
    --eras "1:25:48,2:12:40,3:6:12,4:3:9,5:1:7" --era-caps "60,50,15,12,9" \
    --arms "anchor,outer_yield,learned_yield,learned_task,yoked_learned" \
    --endo-price 267 \
    --budget 8 --g-budget 482 --n-corrupt 1 --mine-cap 8 --gy-level 4 --entry-rec \
    --span-min-hold 128 --collect-task-matched --tm-episodes 8192 \
    --recert-every 5 --n-aud 192 --probe-every 8 --n-rt 384 --n-score 256 \
    --sil-cv 0.15 --lp-min-drop 0.10 --seed 0

python3 rhm/practice/maestro/analyze_maestro.py --tag ma_s0 --fetch --figures
```

The floors are the defaults (A1's measured values) and are asserted, so no `--tol-*` literals are
passed: passing a different one would silently change what the learned weights mean, which is why
`fit_gate` refuses it.

[^private]: Not mirrored: this link points to a document in the private lab repo (the roadmap, the queue, an unrun spec, reading notes, or a conversation). See the top-level README for what is held back and why.
