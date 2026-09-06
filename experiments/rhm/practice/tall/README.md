# tall — the depth-6 port: a diagnosed apparatus negative, and what survives it

**Up**: [../README.md](../README.md) (rhm/practice) · [../../README.md](../../README.md) (rhm)
**Idea doc**: [practice_manufactures_its_own_credit](../../../../ideas/practice_manufactures_its_own_credit.md)
(§17 points here)
**Parents**: [`../recital/`](../recital/README.md) (whose ladder confound this round was built
to break) · [`../ear/`](../ear/README.md), [`../ratchet/`](../ratchet/README.md) (substrate,
imported unmodified)
**Status**: closed 2026-08-16 (Jasper's call after the densification gate). One gates run, one
feasibility probe, one era-sizing calibration, one voided main run, one densification test.
File index: [FILES.md](FILES.md) — which carries the full measured record
(inadmissibility tables, gate results, calibrations, the re-run recipe); this README is the
narrative and the verdicts.

## One-liner

Every level-3 statement in the depth-4 arc is confounded: level 3 there is simultaneously the
deepest *earnable* level and one below the grammar's *ceiling*, so "endogenous signals fail at
the frontier" and "endogenous signals fail at the top of the ladder" cannot be told apart. At
depth 6 they come apart — the damage ladder runs to level 5 while the earnable range stops at 3.
The port established its setting by measurement (**m=4, the repo's LM default, is inadmissible
for sculpting** — synonymy flattens the depth ladder to nothing and walls off every level above
3), passed every structural gate at depth 6/m=2 with the steepest ladder the arc has had
(gradient 3.31×), and then **failed in the loop: the value head is signal-starved at 64 tokens**
(stale-rollout success 0.076 vs depth-4's 0.296), so no arm sculpted and the main run is voided.
A ~0.5 GPU-h densification test then showed the fix direction is real but insufficient
(collection-target count is the whole lever; the loop resumes descending at ~¼–⅛ of depth-4's
rate), and surfaced a pricing inversion that would confound any re-run: **at this scale,
committing a vocabulary costs more beam width than the macros are worth.** Two findings are
regime-independent and stand; the discriminator itself remains open.

## What stands (regime-independent)

1. **Endogenous pacers cannot traverse a ladder longer than the earnable range.** In the voided
   run, `pace_cert` advanced twice and then sat in era 3 for 66 cycles (75.8% of its budget);
   `pace_vocab` for 40. Era 3 earns level 4 > `max_macro_level`, so the certificate and the
   admission rate are not noisy there — they are **undefined**, and no advancement signal
   exists. Structural, independent of whether the loop learns, and invisible at depth 4, where
   the ladder ended exactly at the earnable range. The arc's teacher thread gains a third,
   sharpest clause: beyond the earnable range, self-pacing is impossible *by construction*, not
   merely miscalibrated.
2. **The mined table cannot churn — [`../recital/`](../recital/README.md)'s "same table or
   different?" mechanism question is retired a priori.** The entry-identity instrument recorded
   zero removals everywhere, and must: miner counts only increase, a support threshold on a
   monotone count is monotone, and the ratchet filter can only drop an entry if the lower table
   shrinks, which it cannot. Late-vs-early tables always differ by additions only, and the
   recert counterfactual prices the additions at zero (0 swaps here; 0 in every depth-4 world).

Also standing, as measured facts about the setting rather than the loop: the m-inadmissibility
result (the 2×2 localises the ladder collapse to m, not depth — FILES.md §"The finding that
changes the spec"), the doubly-exponential earnability wall (levels 2–3 earnable at m=2; level 4
needs ~384 cycles to cover, level 5 ~98k), the exact reader at 64 tokens (`read_acc` 1.000 — the
binding constraint on macro level is table size and mining coverage, **not** perception, the
opposite of the depth-4 diagnosis), and the true-macro inertness at levels 4–5 (`macro_true`
error 0.958 / 0.992 — the refs document the grammar's ceiling before any loop runs there).

## What is voided, and why

`tl_s0` (6 arms, matched T = 5.6e7, five-era ladder): competence recovered only **3.5–16.5%**
of the stale→floor range in era 1 against depth-4's 50–80%, arm spread 0.0255 < the ±0.034
stream noise — so the schedule-shape and pacer-quality questions are unanswerable from it.
Root cause is isolated: controller (0.820), reader (1.000) and plant are all healthy; the
**stale value buffer's terminal success is 0.076** vs depth-4's 0.296, so the value model that
steers every beam trained on a ~4× sparser positive signal and cannot rank states (width ladder:
e at width 1/4/16 = 0.80/0.78/0.81 — not search-limited). The calibration run had already shown
the warning (era-1 e *rose* over 35 cycles) and it was not flagged; **"competence descends
toward the floor in a realistic budget" now belongs in the arc's permanent feasibility gate
set**, and its absence is the whole cost of this round.

**Confounded, not claimed:** the level-3 certificate fired non-provisionally at interior level 3
in every arm given ≥28 cycles of era 2 (vs 35–40-and-usually-never at depth 4) — the
discriminator pointing at *boundary artifact* — but a detector going quiet inside a loop that is
not learning is not evidence it went quiet because learning finished. The `cal0`-based prior
recorded at launch is retracted; **boundary-artifact vs frontier-effect remains open.**

## The densification gate (`dens0`, ~0.5 GPU-h)

| n_corrupt / collect budget | terminal success |
|---|---|
| 3/8 (the `tl_s0` config, in-run control) | 0.070 (reproduces 0.076) |
| **1/8** | **0.133** |
| 1/12 · 2/12 · 1/16 | 0.110 · 0.068 · 0.109 |
| *depth-4 reference* | *0.296* |

**Fewer collection targets is the entire lever; more rollout attempts is not** (retracting half
the pre-registered diagnosis: with `explore_eps=0.3`, extra steps break correct configurations
as often as they fix broken ones). The descent gate on the densified head: recovered fraction
0.262 (vs 0.035–0.165 voided, 0.50–0.80 depth 4) — but **0.233 of it was present at cycle 1**,
supplied by the better value head; the loop itself descends at −0.004/cycle, ~4–8× slower than
depth 4. Densification fixed the starting policy and only partly restored learning.

**The finding that dominates any re-run: committing costs more than it buys under this
pricing.** The gate run produced the best table the tall setting has seen (16 entries, recall
0.786, non-provisional) — and the cycle after commit, error jumped +0.180 (a 4.6× outlier
against cycle noise), because the action set grew 32→48 and `fit_width` dropped the beam from
width 2 to 1 at `g_budget=482`. At depth 4 the identical transition was net-positive: macros
there cover a 4× larger fraction of the sequence, so the macro was worth more than the width. At
64 tokens the sign flips. Recorded as an open observation, conditional on the pricing model:
**a committed unit pays its action-space rent only when it covers a meaningful fraction of the
problem** — and as a hard prerequisite for any re-run (`g_budget ≥ 722` so width survives
commit), without which the earning question is confounded by a width penalty.

## Verdict and the re-run recipe

Closed without a main-run result, by decision (the port is not one fix away but at least three:
value-signal density still <½ of depth-4's, learning 4–8× slow, width economics inverted — the
signature of a substrate redesign, not a patch). Everything a future attempt needs is recorded
in [FILES.md](FILES.md): depth 6/m=2 (never m≥3), `n_corrupt=1` collection (plus task-matched
collection damage as the next untested lever), the descent gate as a hard precondition,
`g_budget ≥ 722`, probe-cost re-pricing (probes were 47% of `tl_s0`'s runtime), and the
approved 6-arm design with the `sched_earnable` shape — the one arm only this setting can ask
for, since only here is the ladder longer than the earnable range.

## Runs on disk

| tag | what |
|---|---|
| `feas0` | feasibility probe: setup, refs, reader/plant exactness, 6 cycles (after two interface-drift crashes; `preflight` now exists because of them) |
| `cal0` | era-sizing calibration: 35 cycles era 1 + 35 era 2, all three detectors logged |
| `tl_s0` | the main run — **voided** by the value-starvation regime failure |
| `dens0` | the value-densification sweep + the descent gate |

Volume: `/data/rhm_practice_tall/<tag>/`; fetched copies in `figures/<tag>/`; the reduction
report in `results/tl_s0_report.txt`.

## Reproduce

```bash
cd experiments/            # MODAL_PROFILE=chromatic
modal run rhm/practice/tall/tall.py::gates_remote
python3 rhm/practice/tall/launch_detached.py --fn feasibility --tag feas0 \
    --cycles 6 --max-macro-level 3 --budget 8 --g-budget 482 \
    --eras "1:25,2:12,3:6,4:3,5:1" --arm practice_climb
python3 rhm/practice/tall/analyze_tall.py --tag tl_s0 --fetch --figures
```

## Caveats

- One rule draw, and a loop that never reached its operating regime — every
  number from `tl_s0` except the two structural findings is a statement about a broken regime.
- The chunk-rent observation is conditional on `fit_width`/`g_budget` pricing; a different
  budget-allocation rule could change the sign again.
- The m-inadmissibility verdict is about the *sculpting* substrate; the repo's L=6/m=4 default
  remains correct for the autoregressive-scaling experiments it was calibrated for.
