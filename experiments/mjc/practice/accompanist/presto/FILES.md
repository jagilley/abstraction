# presto — file index and design record

**Up**: [`../README.md`](../../README.md) (practice) · [`../../README.md`](../../../README.md) (mjc)
**Donor (forked verbatim)**: [`../offbook/`](../../offbook/FILES.md) — `world.py` and `nets.py` are
copy-forks, `offbook/` is untouched and every `g0/O1/O2/O3/d0` result stays byte-reproducible.
Gate **G-F** asserts the fork reproduces the donor traversal bit-for-bit on the DONOR piece and
DONOR plant rather than trusting that the copy was faithful; gate **G-F2** asserts the one additive
flag this node adds (`obs_predict`) is a strict no-op at its default.
**Constraints carried from the mjc side**: [`../../ballistic/README.md`](../../../ballistic/README.md)
cut 4b (the quality axis must be smooth/global; the reach must be feasible) ·
[`../../arm_substrate/README.md`](../../../arm_substrate/README.md) P3/P4 (size the planner to the
action-sequence dimension; the composition horizon, and H below it) ·
[`../../dynamics_shift/README.md`](../../../dynamics_shift/README.md) (feedback substitutes for the
model under per-step re-grounding; the FM is load-bearing to the degree you must commit) ·
[`../legato/`](../../legato/README.md) F2 (calibrate to preserve the axis, not to minimise an arm's
error), F3 (no controller-level seam law — **on a slow piece**), F4/F5 ·
[`../offbook/`](../../offbook/README.md) Round 4 (`d0`, the delay gate this node re-runs on a new
piece).
**Idea doc**: [`practice_manufactures_its_own_credit`](../../../../../ideas/practice_manufactures_its_own_credit.md)
§2(b), §3, §6 — fast movements outrun feedback (~50–150 ms), are ballistic within, and are
evaluated at the boundary; the conditions do not occur naturally and must be manufactured.

**No findings README yet — numbers get discussed before any writeup** (repo policy). There is no
`SPEC.md`: the orchestrator's prompt is the record of what was asked, and this file is the record
of what was decided and why.

## The question

`../offbook/`'s Round 4 (`delay_gate.py`, run `d0`) taxed the reflex loop with an observation delay
and confirmed the mechanism exactly — degradation from Δ = 0 to 16 ordered by feedback consumption:
reactive (61 fb/piece) **63.1×**, live-per-segment (4 fb) 2.69×, chain (2 fb) 1.69×, segment tape
(4 fb) 1.28×. And the gate still failed. The playable region ended at Δ = 2 (40 ms) and reactive won
all of it; the ordering inverted only at Δ = 16 by universal collapse, which the pre-fixed guard
refuses. Δ\* = None. The standing reading (Jasper's, adopted) is that depth on that piece is
**geometric, not economic** — an environment property, because the piece was inherited verbatim from
`legato/`, whose question was a comparison *among committed arms*, and a 400 ms segment on a 20 ms
control loop is a slow movement.

> Does a piece designed against the reflex delay — momentum-dominated, smooth, with segments
> shorter than a human proprioceptive loop — open a regime where a stored motor unit is the best
> **playable** option?

## The mapping from offbook (what changes, what does not)

| piece | offbook | presto | why |
|---|---|---|---|
| `world.py` | the substrate | **byte-identical fork** + one additive flag | fork discipline; G-F/G-F2 assert both halves |
| the figure | legato's closed 4-leg loop, legs 0.42–0.45 m | a closed, slightly irregular **pentagon**, legs ≈ 1.176 R | the piece is the variable |
| `h_seg` | `20,20,20` (400 ms/segment) | `6,6,6,6,6` (**120 ms**/segment) | a human proprioceptive loop is ~100 ms = 5 control steps at `dt_ctrl = 0.02`; closed-loop correction *within* a segment must be physically impossible, not forbidden (offbook Round 4 rejected a feedback cap as necessity-by-fiat) |
| phrase | 60 steps, 3 segments | **30 steps, 5 segments** | the chain must sit past the plant's composition horizon while each segment sits under it (legato F4). offbook's plant measured ~21; **G-H re-measures it here**, because a faster, lower-damped plant need not have the same one |
| difficulty | a localised curl patch (`curl_b = 14`) mid-phrase | **`curl_b = 0`**; turn rate at low `joint_damping` | `ballistic/` 4b: a localised needle is open-loop-**incompensable** — even a perfect FM cannot counteract a strong local kick feedforward, which is *why* biology uses feedback there — so it saturates every ballistic arm at "fail". Momentum/braking is the smooth, open-loop-compensable axis, and it is the only one presto uses |
| `joint_damping` | 0.5 | swept {0.5, 0.15, 0.05}, design point read off G-T | momentum-dominated: braking must be *anticipated*. The distal links are damping-dominated at 0.5 (wrist τ = I/c ≈ 0.036 s ≈ 1.8 control steps) and momentum-dominated at 0.05 |
| the delayed observation | naive: act on where you were | **both**: naive (continuity) and **efference copy** (`obs_predict`) | see the decisions below — a delay a nervous system can bridge with its own forward model is not a delay that forces anything to be stored |
| stored content | 72 tapes/cell harvested at `sigma_practice = 0.15`, slots by k-means over CONTENT, representative = lowest realised error, selected by FM audition | the **legato recipe**: harvested at `sigma_perf`, cross-state **plant** audition against a greedy seam-matched score set, `select_library` (keyed) and `select_fixed` (one tape) | offbook's `d0` stored strategies never cleared the anchor at any Δ (0.268 / 0.270 against `ref_stale` 0.146). The recipe is the sibling node's, so the two are comparable |
| the criterion | Δ\* = smallest Δ with `e_chain ≤ e_live_seg ≤ e_reactive` s.t. `min ≤ ref_stale` | **unchanged**, plus a task-anchored second component in the guard | see the playability guard below |

## Decisions taken, with reasons (the things a reader could disagree with)

1. **The figure is a small tight pentagon, not a scaled-down grand tour.** At `dt_ctrl = 0.02 s` a
   6-step segment is 120 ms, and holding the donor's tempo (0.021 m/step) makes a leg 0.13 m — so
   the figure necessarily lives in a ~0.3 m patch of workspace rather than sweeping r = 0.45 → 0.90
   the way legato's loop did. What is bought: the **turn rate** goes from ~90° every 400 ms to
   ~70° every 120 ms, ~5× — and turn rate at speed is exactly the anticipatory-braking difficulty
   the design wants, smooth and global. What is paid: `M(q)` varies less along the figure than it
   did on the donor. Vertices still span radius 0.46–0.76 of a 1.10 m arm (donor: 0.47–0.90), so
   the variation is reduced, not removed. Stated rather than hidden.
2. **The vertex angles are irregular** (70/76/66/76/72° apart), so the five segments differ in leg
   length (spread ≈ 1.16×) and turn. A regular polygon at constant speed is a translation wearing a
   piece's name, and a keyed library over identical segments is trivially degenerate.
3. **The approach is the donor's, unchanged** (~0.25–0.33 m over `H_app = 14`, i.e. the donor's
   0.300 m / 14). The mastered lead-in is not part of the comparison and must not be a second
   variable; its plan is computed once per cell from the **pre-practice** model at a fixed
   `vel_pen_mid = 0` and shared by every arm, which keeps the phrase-launch distribution a property
   of the world rather than of the arm.
4. **`curl_b = 0` — the patch is off, not moved.** Consequence recorded: with the gate identically
   zero, `exclude_region` keeps 100% of the pool, so the pre-practice model is not artificially
   blinded anywhere and `ref_stale` is a *smaller* number than offbook's. That makes the playability
   guard tighter, which is why a second, task-anchored component was added to it (below).
5. **The delayed incumbent gets an efference copy** (`obs_predict`), *and* the naive operator is
   kept. offbook's `obs()` hands a delayed consumer the raw stale state, which is a strawman: a
   nervous system with a reflex delay does not act on where it *was*, it acts on where its own
   forward model says it now *is*, given the commands it has already issued. With the flag on,
   `s_hat(t) = FM-rollout(s(t−Δ), raw commands issued in [t−Δ, t))`, so the delay costs the agent
   exactly the part it could **not** have predicted — the motor noise the efference copy does not
   contain — plus the model's own drift over Δ steps. That is the physically correct statement of
   what a reflex delay costs and it is a much harder incumbent to beat. Measured in the first smoke:
   at Δ = 4 the naive controller ran 0.347 and the predicting one 0.128. **Both are reported at
   every Δ**; the naive column is the continuity read against offbook `d0`, the predicting column is
   the honest one. *(Rejected alternative: leaving the naive operator alone for continuity. A result
   that only beats a strawman is not a result.)*
   The flag is **additive and off by default**, and G-F2 asserts it is a strict no-op at Δ = 0 —
   offbook's own idiom, so `d0` and every prior offbook run stay reproducible.
6. **The score set is built greedily, under the configuration that will deploy it.** legato's
   `score_set` is explicit ("launch states produced by the CURRENT performance configuration at
   performance tempo, truncated at the span's start") and the first presto smoke shows exactly what
   happens when it is not: auditioned against seam states a *reactive* traversal produced, the
   deployed tape arm's per-segment error ran **0.095 → 0.430** across the phrase while its own
   audition said 0.042–0.091. Every tape a library commits moves the next seam off the reactive
   manifold, so the library was being selected for a distribution it would never see. Segment k's
   score set is therefore harvested from a traversal that plays the already-committed tapes for
   0..k−1 and stops there (`stop_seg = k`) — which runs no planner and costs almost nothing. The
   keyed and the fixed libraries each get **their own** greedy prefix: sharing one audition matrix
   graded the fixed arm on a distribution it was never selected for and returned 0.606, worse than
   doing nothing.
7. **The live unit's hand-over is calibrated per cell, before anything else is read** (`CAL`).
   The same smoke found the donor's inherited cost shaping is wrong on this piece: a `plan_launch`
   unit committed to a 6-step span with `w_waypoint = 16` on its last step and no terminal velocity
   penalty hits its waypoint at any speed it likes (peak tip speed **17.9 m/s** against reactive's
   7.5) and hands the next segment an unplayable state. That is legato l1's measured disease ("hit
   waypoints while ignoring HANDOFF VELOCITY") on a piece with three times the seam rate. Two
   repairs are swept, neither assumed: `lookahead_gamma` (the live unit plans one segment past its
   commitment with the look-ahead cost discounted by γ — legato l2's op, which measured **γ = 0 best
   on the donor piece** and recorded it as F3, "*there is no controller-level seam law*"; presto's
   seams arrive every 120 ms, which is exactly why it is re-asked) and `vel_pen_mid`. Smoke:
   γ = 0 → 0.282 (vmax 17.3), γ = 0.5 → 0.179 (14.9), γ = 1.0 → 0.163 (14.1).
   **Selection rules, written down before the grid is read** (legato F2: a shared calibration knob
   may not be chosen by one arm's minimum error, because a knob that minimises error can silently
   minimise *measurability* — l0 chose `vel_pen_mid` by reactive's minimum and left 3% of the
   outcome attributable to model quality):
   - `gamma*` = argmin of the practised live-per-segment error. γ is a knob of the **live unit
     alone**, so tuning it *strengthens the incumbent side* of this node's comparison — the
     conservative direction.
   - `vel_pen_mid*` = argmax of the **usable range** (stale − practised live-per-segment error, an
     arm-neutral reference), subject to a competence guard (practised error ≤ half the do-nothing
     floor). Reactive's number is reported at every value because the knob is shared. If the guard
     admits nothing the fallback is the whole grid, and the report says so.
   - `look*` = argmin of reactive error at Δ = 0 — the vanilla baseline, so the incumbent is not
     handicapped by a lookahead sized for 20-step segments. The full look × Δ grid is reported in
     G-P because a *delayed* controller prefers a longer one; Phase B gives reactive its per-Δ best.
   - `w_waypoint` is **not** swept; it stays at the donor's 16.
8. **The playability guard gains a task-anchored second component, fixed before the run.**
   offbook's anchor is `ref_stale` — ballistic-per-segment under the pre-practice forward model,
   undelayed — and it is carried unchanged. But offbook's `ref_stale` was large (0.146) partly
   *because* the curl patch was excluded from the pre-practice model's training, and presto has no
   patch, so the same formula can produce a much smaller number and veto by construction — which is
   the exact failure offbook's own guard revision was written to avoid. The presto guard is
   therefore `ref_play = max(ref_stale, ½ × mean drilled leg)`: the second term says the waypoints
   are still resolved, i.e. the figure is recognisable. Both components are reported separately at
   every cell so either can be read off the record. Neither depends on any compared arm, and both
   are computed before any strategy's number is known.
9. **The candidate pool is nested too, not only the score set** (Phase B; recorded 2026-08-26 on
   the sibling `d1` round's evidence). Decision 6 fixed the *score set*; the sibling found the other
   half of legato's protocol is equally load-bearing and equally easy to drop. `d1` ported legato's
   plant audition onto offbook's piece with a **reactive-harvested** candidate pool and reproduced
   legato's seam-0 content exactly (chosen 0.041 vs 0.043) while seams 1–2 came out **4× worse at
   the per-state-oracle level** — i.e. the pool contained nothing that works from the committed
   configuration's launch states at all, which no amount of better selection can repair. legato's
   own commit events show why: at its seam-1/2 commits, **142 of 144** candidates had been recorded
   from practice in the already-committed configuration. So presto's Phase B build walks the piece
   in order and draws *both* the candidates and the score set from the configuration that will
   deploy the unit, at every seam; the chain pool is grown from the fully segment-committed
   configuration (legato F5) with a reactive-sourced pool kept beside it as `chain_react`, the
   continuity control on the content recipe. **`chosen` and `per_state_oracle` are reported per
   cell** so the recipe's fidelity is on the record for the new piece. Also **harvest late**: all
   harvesting happens after the full warm-up, since `d1` measured tape vintage as its single largest
   content effect.
   *Known limitation of Phase A `t0`, which was already in flight when this landed*: its G-L content
   column uses the greedy score set but a **reactive** candidate pool, so it understates achievable
   content at seams 1–4. `t0`'s design decision rests on `react` / `live_seg` / `ref_stale` /
   G-H / G-S, none of which are affected.
10. **The forward model trains *through* the commits and is frozen only for the sweep** (revised
   2026-08-26 on the sibling `d2` round's evidence; this reverses an earlier presto decision, and
   the reversal is the record). `d2` added the nested candidates to offbook's piece and recovered
   legato's content to within **0.87–1.16×** of legato's per-state oracle at every seam and the
   chain cell (against `d1`'s 4×), with its keyed nested chain reaching **0.1455** against that
   node's **0.1460** anchor — the first stored strategy to reach playability there, flat across the
   whole delay sweep (1.09× over 0→320 ms) and strictly the best of the three from 80 ms on. Its
   residual against legato is most plausibly that `d2` **froze** the model through the nested
   practice, to keep a round-4 control bit-identical, while legato's model kept training through its
   commits. presto's first draft made the same freeze, for a different reason — so that every
   strategy in the sweep is compared under ONE model and the delay axis cannot be confounded with
   which arm the model was last fitted to. **Both aims are satisfiable at once**, which is why the
   decision flipped: the model trains through the between-commit practice cycles (legato's
   protocol, so the content is legato-quality) and is frozen **at the end of the build**, before
   the chain pool is grown and before the sweep (so the sweep still has one model). presto has no
   cross-round control to preserve, so nothing is lost. `ref_stale` stays defined under the
   **pre-practice** model `fm0` and is measured before the build, so the playability anchor cannot
   move with any of this.
   **The model's own competence is on the record at every commit** (`e_react`, and `e_ball_seg`
   beside it, on the held-out evaluation geometries at Δ = 0). This is legato CAL-D's curve, and it
   is the quantity that says whether content committed at seam *k* was frozen against a model that
   was still improving — legato's own measured failure mode, where `L1` committed at c25–c39
   against a ballistic clock that had not plateaued by c72.
   One departure from legato is kept and stated: **practice is noisy, harvest is not.** Practice
   runs at `sigma_practice` and trains the model; candidates are harvested separately at
   `sigma_perf`. legato drew its candidates from the practice trace pool itself; offbook's `d0` did
   the same and its stored strategies never cleared the anchor at any Δ.
11. **The criterion is reported per Δ with its margins, not only its verdict.** `d2` missed the
   guard by **3.6%** on offbook's piece; a pass/fail column alone reports that as an
   indistinguishable "no". Every row carries the two ordering margins (`e_live_seg − e_chain`,
   `e_reactive − e_live_seg`), the guard margin (`ref_play −` the best of the three) in metres and
   as a fraction of the guard, and the chain's own distance to the guard. The verdict rule itself
   is unchanged and stays pre-fixed.
12. **Single seed 0 on every cell**, as everywhere in this arc. The claims are ranks, signs and
   bit-identity assertions, not point estimates.

## Code files

| file | purpose |
|---|---|
| `piece.py` | The piece as a pure-python generator (`pentagon` / `waypoints` / `legs` / `mean_leg`) and its constants — a module that imports only `math`, because a Modal *local* entrypoint runs on a machine with neither numpy nor mujoco, and importing a sibling node's runner registers its entrypoints on the shared app. Holds the R and damping ladders and, once G-T is reduced, the design point. |
| `world.py` | The verbatim fork of `offbook/world.py` (two import lines re-pointed) plus **one additive flag**: `obs_predict`, efference copy through the reflex delay, off by default and asserted no-op by G-F2. |
| `nets.py` | The verbatim fork of `offbook/nets.py`. Only `SlotLayout` is used here; the two ports are dormant and carried so a Phase B port starts from donor code. |
| `tempo.py` | **Phase A** — G-F, G-F2, CAL (γ / `vel_pen_mid` / `look`), G-T (the R × damping tempo ladder), G-H (composition horizon), G-S (seam information), G-L (content calibration), G-P (planner ladder). One Modal container. |
| `analyze_tempo.py` | The Phase A report and the design record the delay sweep's knobs are read from. |
| `delay_gate.py` | **Phase B** — the Δ sweep at the chosen design point, four strategies (+ content controls), under offbook's pre-fixed criterion with Δ = 0 as the vanilla baseline. |
| `analyze_delay.py` | Reduces the Δ sweep: the sweep table, degradation factors, the criterion's verdict, crossovers, and figures. |

## Modal volume layout

```
/data/practice_presto/<tag>/tempo/results.json        # Phase A
/data/practice_presto/<tag>/delay_gate/results.json   # Phase B
```

Fetched copies and reports under `results/<tag>/`.

## Reproduce

```bash
cd experiments/            # MODAL_PROFILE=chromatic
# smokes (attached and short by design; every gate runs before every launch)
modal run mjc/practice/accompanist/presto/tempo.py::tempo --quick --tag tsmoke

# Phase A -- the design decision
modal run --detach mjc/practice/accompanist/presto/tempo.py::tempo --spawn --tag t0 --seed 0
python3 mjc/practice/accompanist/presto/analyze_tempo.py --tag t0 --fetch

# Phase B -- the delay sweep (knobs read off the Phase A report)
modal run --detach mjc/practice/accompanist/presto/delay_gate.py::delay_gate --spawn --tag p0 --seed 0 [...]
python3 mjc/practice/accompanist/presto/analyze_delay.py --tag p0 --fetch
```

## Gotchas (inherited, and re-verified here)

- **Launch only via the `--spawn` path.** `modal run --detach <file>::<local_entrypoint>` does NOT
  keep a run alive: a local entrypoint runs on the CLIENT, and a blocking `.map()`/`.remote()` dies
  with the launching process, cancelling the inputs still in flight and leaving the completed ones.
  It cost offbook two runs. `--spawn` returns in seconds and the detached app carries the work.
  The non-spawn path is kept for `--quick` smokes, which are attached and short by design.
- **Never import a sibling node's runner.** `mjc/shared.py` exposes ONE Modal `app` for the whole
  package, so importing another node's runner registers its `@app.local_entrypoint()`s and
  `modal run` dies with `InvalidError: Duplicate local entrypoint name`. presto's entrypoints are
  `tempo` / `run_tempo` and `delay_gate` / `run_delay_gate`, and the DONOR piece constants that gate
  G-F needs are held **by value** in `tempo.py` rather than imported from `../offbook/piece.py`.
  (Importing `offbook.world` inside the *remote* function body is fine and is what G-F does.)
- **A Modal *local* entrypoint runs locally**, on a machine with neither numpy nor mujoco. Anything
  reachable from a runner's module scope must import none of them — hence `piece.py`'s `math`-only
  contract.
- **Launcher logs live outside the mounted package** (`experiments/.launch_logs/`): `shared.py`
  mounts `mjc` with `add_local_python_source("mjc")` and Modal hashes the directory, so a live log
  under `mjc/` makes every *other* concurrent `modal run` die with `ExecutionError: <path> was
  modified during build process`, naming the log rather than the broken run.
- **Tag namespaces differ by more than case** (legato's macOS collision): Phase A `t*`, Phase B
  `p*`, smokes `tsmoke*` / `psmoke*`.

## Phase A record (`t0`, seed 0, 2026-08-26)

*Launched 2026-08-26 23:08 UTC; `modal run --detach … --spawn --tag t0 --seed 0`, app
`ap-diJ9yUBbr4luBvdwFGF2Ux`, function `fc-01M105AHKC2SNDRB4KFRFV75XR`. Ladder R ∈ {0.09, 0.13, 0.17}
× `joint_damping` ∈ {0.5, 0.15, 0.05}; 15 warm reactive cycles per cell at batch 16; 24 held-out
evaluation geometries, 48 score geometries, 96 candidate tapes, `n_lib = 8`.*

Complete, 54 min wall. **G-F 0.000e+00 · G-F2 0.000e+00** — the fork reproduces offbook on the
donor piece and plant bit-for-bit, and the efference-copy flag is a strict no-op at Δ = 0.

**G-T — the tempo ladder** (mean piece error, 24 held-out geometries, `sigma_perf`; `rct` = naive
delayed observation, `prd` = the same controller with efference copy):

| damping | R | m/s | hold | ref_stale | half_leg | rct/prd Δ2 | rct/prd Δ4 | rct/prd Δ6 | live_seg Δ0 | live_chain Δ0 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.5 | 0.09 | 0.88 | 0.2075 | 0.0216 | 0.0529 | 0.1778 / 0.0198 | 0.2802 / **0.0316** | 0.3868 / 0.0482 | 0.0182 | 0.0532 |
| 0.5 | 0.13 | 1.27 | 0.1940 | 0.0330 | 0.0764 | 0.0964 / 0.0220 | 0.2361 / **0.0246** | 0.3574 / 0.0281 | 0.0235 | 0.0583 |
| **0.5** | **0.17** | **1.66** | 0.2156 | 0.0477 | 0.0999 | 0.1206 / 0.0279 | 0.2394 / **0.0346** | 0.3188 / 0.0424 | 0.0324 | 0.0922 |
| 0.15 | 0.09 | 0.88 | 0.2729 | 0.0365 | 0.0529 | 0.1556 / 0.0169 | 0.3821 / **0.0272** | 0.4468 / 0.0429 | 0.0261 | 0.0820 |
| 0.15 | 0.13 | 1.27 | 0.2544 | 0.0558 | 0.0764 | 0.1872 / 0.0201 | 0.4166 / **0.0274** | 0.5042 / 0.0375 | 0.0295 | 0.0868 |
| 0.15 | 0.17 | 1.66 | 0.2752 | 0.0785 | 0.0999 | 0.1823 / 0.0261 | 0.4014 / **0.0422** | 0.4693 / 0.0600 | 0.0409 | 0.1377 |
| 0.05 | 0.09 | 0.88 | 0.3887 | 0.0863 | 0.0529 | 0.3339 / 0.0249 | 0.4293 / **0.0427** | 0.4418 / 0.0633 | 0.0375 | 0.2048 |
| 0.05 | 0.13 | 1.27 | 0.4333 | 0.1097 | 0.0764 | 0.2392 / 0.0293 | 0.4644 / **0.0432** | 0.4921 / 0.0592 | 0.0421 | 0.1748 |
| 0.05 | 0.17 | 1.66 | 0.3796 | 0.1047 | 0.0999 | 0.3463 / 0.0373 | 0.5074 / **0.0617** | 0.5449 / 0.0914 | 0.0577 | 0.2072 |

Reactive at Δ = 0 runs 0.0100–0.0261 in every cell; fb/traversal: reactive 31, live_seg 6,
live_chain 2.

**The ladder's headline is a structural fact about the intervention, not about the piece.** The
naive delay is devastating and monotone in every cell (Δ = 4 costs reactive **10–25×**), and the
efference copy removes essentially all of it: `prd` at Δ = 4 lands at **0.0246–0.0617**, i.e. within
1.5–3× of the *undelayed* reactive number, and at Δ = 6 (120 ms — a whole segment of delay) it is
still 0.028–0.091. **`prd` at Δ = 4 is below the playability anchor in all nine cells.** The reason
is mechanical: bridging a Δ-step delay is a Δ-step forward-model rollout, and G-H measures the
composition horizon at 8–31 steps, so Δ ∈ {4, 6} sits comfortably inside it. The residual cost of
the delay is only the part the efference copy cannot contain — the motor noise — and at
`sigma_perf = 0.06` over 4–6 steps that is small. So **the same forward model that would be needed
to run stored content well is the model that makes the delay bridgeable**, and taxing feedback with
delay cannot separate them.

**The momentum axis did not do what the design expected, and the numbers say why.** Lowering
`joint_damping` 0.5 → 0.05 makes *everything* worse without breaking the predicting incumbent: the
one-step FM error on the pool rises 0.091 → 0.209, the practised composition horizon collapses
**23 → 8 steps** (R = 0.09), the do-nothing floor rises 0.21 → 0.39, and the best stored content
degrades 0.102 → 0.220 — while `prd` at Δ = 4 moves only 0.0316 → 0.0427. Low damping degrades the
*model* far more than it degrades *delayed control*. This is a negative on decision 1's momentum
hypothesis, recorded rather than worked around.

**G-H — the composition horizon, re-measured** (practised / stale, first step median tip divergence
crosses 0.05 m; segment = 6 steps, phrase = 30):

| cell | 0.5/0.09 | 0.5/0.13 | 0.5/0.17 | 0.15/0.09 | 0.15/0.13 | 0.15/0.17 | 0.05/0.09 | 0.05/0.13 | 0.05/0.17 |
|---|---|---|---|---|---|---|---|---|---|
| practised | 23 | **31** | **21** | 22 | 18 | 15 | 8 | 12 | 14 |

The design requirement — segment under the horizon, phrase over it — holds in every cell **except
`0.5/0.13`, where the practised horizon (31) exceeds the 30-step phrase**, so the chain would not be
past the model's reach there. That single number disqualifies the otherwise-attractive middle cell.

**G-S — seam information** (posture spread ÷ matched repeat-noise, 48 performers = 8 postures × 6
repeats): **0.38–1.54×, and below 1.0 in 44 of 45 seam-cells.** The seams do *not* carry
across-performer posture structure above their own repeat noise — the donor's seam 0 measured 5.17×.
A keyed library therefore has little systematic signal to address with, which is visible directly in
G-L: the keyed chain's chosen-vs-oracle gap runs **3.0–5.3×**, the largest in the table.

**CAL — the per-cell cost shaping** (selection rules declared in decision 7, applied before anything
else was read). γ mattered enormously and `vel_pen_mid` did not, once γ was set:

| cell | γ grid (live-per-seg error) | γ\* | `vel_pen_mid` (seg / react / stale) | v\* | look grid | look\* |
|---|---|---|---|---|---|---|
| 0.5/0.17 | 0.0: 0.0989 · 0.5: 0.0311 · **1.0: 0.0310** | **1.0** | 0.0: .0313/.0203/.0489 · 0.1: .0320/.0205/.0486 · 0.3: .0359/.0253/.0531 | **0.0** | 6: .0587 · **12: .0190** · 30: .0511 | **12** |
| 0.5/0.13 | 0.0: 0.0704 · **0.5: 0.0229** · 1.0: 0.0234 | 0.5 | 0.0: .0233/.0137/.0281 · 0.3: .0271/.0190/.0322 | 0.3 | 6: .0326 · **12: .0187** · 30: .0359 | 12 |
| 0.15/0.13 | 0.0: 0.1433 · 0.5: 0.0290 · **1.0: 0.0288** | 1.0 | 0.0: .0307/.0147/.0542 | 0.0 | 6: .0246 · **12: .0150** · 30: .0286 | 12 |

**γ is worth 2.3–5.0× on the live-per-segment arm** (0.099 → 0.031 at 0.5/0.17; 0.143 → 0.029 at
0.15/0.13). legato F3 measured γ = 0 best and concluded "there is no controller-level seam law" — on
a piece whose seams arrive every 400 ms. At 120 ms seams the law is there and it is large. This is
the clearest single reason presto's live arms are competitive where offbook's `live_seg` was not
(0.198 against a 0.146 anchor).

**G-P — the planner ladder** (`arm_substrate` P3). Flat at the top of the grid everywhere, so no arm
is planner-starved: at 0.5/0.17 the segment span runs 256: 0.0369 · 1024: 0.0316 · 4096: 0.0312 and
the phrase span 256: 0.0893 · 1024: 0.0927 · 4096: 0.0929 (non-monotone within 4%). A *delayed*
reactive controller prefers a longer lookahead (0.5/0.09, Δ = 4 naive: look 6 → 0.2840, 12 → 0.2482,
30 → 0.1923), which is why Phase B gives reactive its per-Δ best rather than one inherited number.

**G-L — content, at the Phase A (pessimistic) recipe.** The best stored strategy at Δ = 0 clears the
anchor in exactly one cell (0.5/0.17: chain 0.0928 vs 0.0999) and misses narrowly at 0.5/0.13
(0.0829 vs 0.0764). But the per-state **oracle** for the chain cell is 0.033–0.110 with a gain of
3.0–5.3× — *the content exists and the addressing does not*, which is G-S's finding arriving as a
number. Segment-level oracles degrade with seam index (seg0 0.026–0.194 → seg4 0.063–0.411), the
distribution-shift signature decision 9 fixes for Phase B.

### The design point, and why

**`joint_damping = 0.5`, `R = 0.17`** (mean drilled leg 0.200 m, nominal tip speed **1.66 m/s**,
γ\* = 1.0, `vel_pen_mid`\* = 0.0, `look` grid {6, 12, 30} with the per-Δ best taken, CAL-P
`1:1024:8, 5:1024:12`), with `n_score` 48 → **72** and `n_lib` 8 → **12** (holding states-per-cell at
6 while doubling the addressing resolution — licensed by the 3.0–5.3× chosen-vs-oracle gap, which is
a resolution limit and not an information limit).

Four reasons, all from the table above and none of them a preference:
1. **The composition horizon is 21** — between the 6-step segment and the 30-step phrase, which is
   the design's structural requirement (legato F4). `0.5/0.13` fails it outright at 31.
2. **It is the only cell where measured content already clears the playability anchor at Δ = 0**
   (chain 0.0928 ≤ 0.0999) *under the pessimistic Phase A recipe*, so Phase B's nested build starts
   from the one cell that does not need the recipe to rescue it.
3. **It is the fastest playable piece** (1.66 m/s, largest legs), so the task-anchored half of the
   guard is at its most generous and the turn rate — the difficulty this node manufactures — is at
   its highest.
4. **Lower damping is contra-indicated by measurement**, not by taste: it collapses the horizon
   (23 → 8) and the model (one-step pool error 0.091 → 0.209) while leaving the predicting incumbent
   essentially untouched, so it buys difficulty in the one currency the design cannot use.

**What Phase A already implies about the gate, stated before the run so it cannot be read back in
afterwards.** `prd` at Δ ∈ {4, 6} beats every stored strategy in every cell, and `live_seg` at
Δ = 0 (0.0324 at the design point) beats every stored strategy everywhere — so under the PRIMARY
(predicting) incumbent the ordering `e_chain ≤ e_live_seg` is a long way from holding. Under the
NAIVE incumbent it is much closer: at 0.5/0.13, Δ = 4 already gives chain 0.1299 ≤ live_seg 0.1338 ≤
reactive 0.2361 — ordered, and failing only the guard (0.1299 against 0.0764). Whether Phase B's
nested content closes that guard gap is the run's question, and the two operators are reported side
by side precisely so the answer distinguishes "the delay does not force memory" from "offbook's
delay operator was a strawman".

### Smoke record (what the gate caught before the real launch, and why each is on the record)

| smoke | what it caught |
|---|---|
| `tsmoke` / `tsmoke2` | G-F and G-F2 both **0.000e+00**. The efference-copy flag recovers most of the naive delay penalty (Δ = 4: 0.347 → 0.128), which is why the strong incumbent exists at all. |
| `tsmoke3` | The live unit's hand-over is broken on this piece (per-segment error 0.09 → 0.27, peak tip speed 17.9 m/s) — legato l1's disease at 3× the seam rate. Motivated CAL. |
| `tsmoke3` | The library's score set was on the wrong distribution (deployed 0.095 → 0.430 against an audition saying 0.042–0.091). Motivated the greedy score set — legato's own protocol, which the first draft had silently dropped. |
| `tsmoke4` | The greedy fix helped the keyed arm (0.303 → 0.268) and **broke** the fixed arm (0.328 → 0.606), because it was being graded on the keyed prefix's distribution. Motivated the two-prefix build. |
| `tsmoke5` / `tsmoke6` | CAL runs first and selects; the k_shoot ladder is flat at the segment span (0.171 / 0.170 / 0.182 — not planner-starved, `arm_substrate` P3); a *delayed* reactive controller prefers a longer lookahead (Δ = 4: look 6 → 0.520, 12 → 0.427, 30 → 0.354), which is why Phase B gives it its per-Δ best. |

## Phase B record (`p0`, seed 0, 2026-08-27) — THE GATE FAILS UNDER BOTH OPERATORS: Δ\* = None

Launched `--r 0.17 --joint-damping 0.5 --lookahead-gamma 1.0 --vel-pen-mid 0.0 --react-look 12
--look-grid 6,12,30 --calp 1:1024:8,5:1024:12 --delays 0,1,2,3,4,5,6,8 --n-warm 20 --n-between 5
--n-eval 24 --n-score 72 --n-harvest 4 --n-lib 12`; app `ap-YQvXUyKCS16uRWycvbn7US`.
Complete, **14 min** wall. Report `results/p0/report.txt`, figures `results/p0/figs/`.

Anchors: `ref_stale` **0.0477**, `half_leg` **0.0999** → **`ref_play` = 0.0999** (the task-anchored
term binds, exactly as decision 8 anticipated: with no curl patch the pre-practice model is not
blinded anywhere and `ref_stale` collapses). Do-nothing floor **0.2156**. fb/traversal: reactive 31,
live_seg 6, seg_tape 6, seg_fixed 6, chain 2, chain_fixed 2, chain_react 2, live_chain 2.

### The sweep

**PREDICTING incumbent (efference copy through the delay) — the PRIMARY read**

| Δ | ms | reactive | live_seg | live_chain | seg_tape | seg_fixed | chain | chain_fixed | chain_react |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | **0.0184** | 0.0310 | 0.1053 | 0.0949 | 0.1194 | 0.0996 | 0.1420 | 0.0960 |
| 2 | 40 | **0.0257** | 0.0356 | 0.1097 | 0.0875 | 0.1194 | 0.0964 | 0.1420 | 0.0940 |
| 4 | 80 | **0.0291** | 0.0439 | 0.1121 | 0.0958 | 0.1194 | 0.0934 | 0.1420 | 0.0985 |
| 6 | 120 | **0.0347** | 0.0469 | 0.1126 | 0.0964 | 0.1194 | 0.0934 | 0.1420 | 0.0944 |
| 8 | 160 | **0.0418** | 0.0526 | 0.1130 | 0.0958 | 0.1194 | 0.0960 | 0.1420 | 0.0947 |

degradation 0→160 ms: reactive **2.28×** · live_seg 1.70× · live_chain 1.07× · seg_tape 1.01× ·
chain 0.96× · seg_fixed / chain_fixed 1.00×.

**NAIVE incumbent (offbook `d0`'s operator: act on where you were)**

| Δ | ms | reactive | live_seg | live_chain | seg_tape | seg_fixed | chain | chain_fixed | chain_react |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | **0.0184** | 0.0310 | 0.1053 | 0.0949 | 0.1194 | 0.0996 | 0.1420 | 0.0960 |
| 2 | 40 | 0.1166 | 0.0827 | 0.1026 | 0.1269 | 0.1194 | **0.0979** | 0.1420 | 0.1068 |
| 4 | 80 | 0.1894 | 0.1562 | 0.0956 | 0.1287 | 0.1194 | **0.1063** | 0.1420 | 0.1296 |
| 6 | 120 | 0.2831 | 0.2974 | 0.1063 | 0.2334 | 0.1194 | 0.1201 | 0.1420 | 0.1198 |
| 8 | 160 | 0.3506 | 0.3090 | 0.1739 | 0.3153 | **0.1194** | 0.1291 | 0.1420 | 0.1210 |

degradation 0→160 ms: reactive **19.10×** · live_seg 9.96× · seg_tape 3.32× · live_chain 1.65× ·
chain 1.30× · seg_fixed / chain_fixed 1.00×.

**offbook `d0`'s core mechanism replicates on a new piece and a new plant, and survives the honest
operator.** Delay sensitivity is ordered by feedback consumption in *both* columns — naive
19.1 / 9.96 / 3.32 / 1.30 / 1.00 and predicting 2.28 / 1.70 / 1.01 / 0.96 / 1.00, monotone in
fb/traversal each time. What the efference copy changes is the *magnitude*, not the ordering: it
divides the incumbent's degradation by **8.4×** (19.10 → 2.28).

### The pre-fixed criterion, with margins

`Δ*` = smallest Δ with `e_chain ≤ e_live_seg ≤ e_reactive`, subject to `min(three) ≤ ref_play`.

| Δ | ms | | chain | live_seg | reactive | ordered (ch−live / live−rct) | playable (guard margin) | passes |
|---|---|---|---|---|---|---|---|---|
| 0 | 0 | pred/naive | 0.0996 | 0.0310 | 0.0184 | **no** (−0.069 / −0.013) | yes (+81.6%) | no |
| 2 | 40 | predict | 0.0964 | 0.0356 | 0.0257 | no (−0.061 / −0.010) | yes (+74.3%) | no |
| 4 | 80 | predict | 0.0934 | 0.0439 | 0.0291 | no (−0.050 / −0.015) | yes (+70.9%) | no |
| 8 | 160 | predict | 0.0960 | 0.0526 | 0.0418 | no (−0.043 / −0.011) | yes (+58.1%) | no |
| 2 | 40 | naive | 0.0979 | 0.0827 | 0.1166 | no (−0.015 / **+0.034**) | yes (+17.2%) | no |
| **3** | **60** | **naive** | **0.1008** | **0.1071** | **0.1513** | **YES (+0.006 / +0.044)** | **no (−0.9%, −0.0009 m)** | **no** |
| 4 | 80 | naive | 0.1063 | 0.1562 | 0.1894 | **YES** (+0.050 / +0.033) | no (−6.4%) | no |
| 5 | 100 | naive | 0.1194 | 0.2273 | 0.2692 | **YES** (+0.108 / +0.042) | no (−19.6%) | no |
| 8 | 160 | naive | 0.1291 | 0.3090 | 0.3506 | **YES** (+0.180 / +0.042) | no (−29.3%) | no |

**Δ\* = None under both operators — and they fail for opposite reasons, which is the round's
result.** Under the PRIMARY (predicting) incumbent the ordering never comes close at any delay: the
`chain − live_seg` margin is −0.069 at Δ = 0 and still −0.043 at 160 ms, because the piece stays
comfortably playable throughout (reactive 0.0184 → 0.0418) and reactive wins every row. Under the
NAIVE incumbent **the ordering does hold, from Δ = 3 (60 ms) onward** — which offbook `d0` never
achieved at any playable delay on its own piece, where the ordering appeared only at 320 ms by
universal collapse — and the gate is lost on playability instead, **by 0.9% (0.0009 m) at 60 ms**,
then 6.4% / 19.6% / 29.3%. The margins column is what makes that visible; a pass/fail column would
have reported the 60 ms row as an indistinguishable "no".

**Crossovers.** Under the naive operator, **`chain < reactive` first lands at Δ = 2 (40 ms) and is
INSIDE playability** (0.0979 vs 0.1166) — a phrase-span stored unit beating reactive MPC inside the
playable region, which `d0` never produced (its only inside-playability crossover was
`seg_tape < live_seg`, one level down). `chain < live_seg` follows at Δ = 3 (60 ms), just outside.
Under the predicting operator no stored strategy ever beats either live strategy; the one crossover
that does land inside playability is **`chain < seg_tape` at Δ = 3** (0.0960 vs 0.0967) — the depth
ordering appearing *among stored units*, phrase-span over segment-span.

### Content fidelity (the nested build)

| cell | chosen | best single tape | per-state oracle | gain | distinct |
|---|---|---|---|---|---|
| key/seg0 | 0.0450 | 0.0566 | 0.0295 | 1.92× | 10/12 |
| key/seg1 | 0.0355 | 0.1013 | 0.0161 | 6.27× | 12/12 |
| key/seg2 | 0.0690 | 0.1309 | 0.0460 | 2.85× | 9/12 |
| key/seg3 | 0.1042 | 0.1605 | 0.0747 | 2.15× | 10/12 |
| key/seg4 | 0.1149 | 0.1571 | 0.0670 | 2.35× | 12/12 |
| chain | 0.0816 | 0.1296 | 0.0517 | 2.51× | 10/12 |
| chain_react | 0.0785 | 0.1185 | **0.0377** | 3.15× | 9/12 |
| chain_fixed | 0.1296 | 0.1296 | 0.0517 | 2.51× | 1/1 |

96 candidates × 72 score states per cell.

**The nested build reproduces the sibling `d1`→`d2` signature exactly on presto's piece.** Against
Phase A's reactive-pool recipe at the *same* cell, the per-state oracle is unchanged at seam 0
(0.026 → 0.030, 0.9×) and **2.1–2.9× better at seams 1–4** (0.046/0.104/0.157/0.142 →
0.016/0.046/0.075/0.067). That is `d1`'s diagnosis confirmed independently: the seam-0 pool was
never the problem, and the later seams' pools were.

**legato F5 does not pay here, contra expectation.** The chain pool grown from the fully
segment-committed configuration (`chain`, the declared primary) is *worse* than the reactive-sourced
control at both the oracle (0.0517 vs **0.0377**) and deployed (0.0996 vs 0.0960 at Δ = 0) level.
Recorded as measured; the arms are one declared-primary and one declared-control, so no selection
was made after the fact.

**The model's competence clock is flat** (`e_react` / `e_ball_seg` on the held-out geometries at
Δ = 0, after each block of between-commit practice): c25 0.0157/0.0259 · c30 0.0155/0.0252 · c35
0.0180/0.0295 · c40 0.0138/0.0270 · c45 0.0157/0.0306 · c50 (chain) 0.0167/0.0309. So no content
was frozen against a model that was still improving — legato CAL-D's measured failure mode is
absent here, and the `d2` residual it was meant to address does not apply to this run.

### Sanity checks

- **Δ = 0 rows are identical under both operators** (0.0184 / 0.0310 / 0.0996 …) — G-F2's no-op
  assertion holding in-run, not only in the gate.
- **`seg_fixed` (0.1194) and `chain_fixed` (0.1420) are constant to four decimals at every Δ and
  under both operators.** A unit that reads no state cannot be touched by an observation delay, and
  that is the check that the delay enters the traversal through `obs()` and nowhere else.
- **By-segment medians**: every stored strategy's error grows monotonically down the phrase
  (chain at Δ = 0: 0.053 / 0.058 / 0.083 / 0.122 / 0.128) while reactive's stays flat
  (0.045 / 0.002 / 0.043 / 0.004 / 0.001). The nested build reduces the compounding that
  decision 6 diagnosed but does not remove it.
- **Non-monotonicity worth flagging**: naive `seg_tape` at Δ = 1 (0.1405) is worse than at Δ = 2
  (0.1269), and the naive incumbent's chosen lookahead flips 12 → 30 → 6 → 30 across the sweep.
  Both are inside the run-to-run scatter of 24 evaluation geometries at a single seed.
- **Ledger**: agent-side 35,200 env steps / 18,800 feedback events / 16,800 plans /
  1.272e9 FM rollout-steps / `t_priced` 2,584 s — i.e. the practice and the nested build.
  Instrument-side 1,083,600 steps / 160,104 fb / 62,184 plans: the whole sweep, the anchors and
  every plant audition are experimenter-side and the agent is charged for none of them.

### Standing (factual; no interpretation written — repo policy)

The gate fails under the pre-fixed rule at every Δ, under both operators, and the round stops there:
Δ is not extended and the criterion is not re-fitted (legato F2). The two failures are of different
kinds — one an ordering that never forms, one an ordering that forms at 60 ms and misses playability
by 0.9% — and the difference between them is entirely the incumbent's efference copy.
**`p0` is the last run of this design** (2026-08-27): the line is being redirected to the forward
model's role in the incumbent and in the audition, which is the axis both phases kept returning.
