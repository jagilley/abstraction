# legato — file index

**Up**: [`../README.md`](../README.md) (practice) · [`../../README.md`](../../README.md) (mjc)
**Parents**: [`../fingering/`](../fingering/FILES.md) (round 1 — the substrate this forks, and the
result this round starts from) · [`../etude/`](../etude/README.md) §E-4/§E-5 (sequential phrase
assembly, seam-matched score sets, and the fusion-vacuity finding this round overturns) ·
[`../../arm_substrate/README.md`](../../arm_substrate/README.md) §P3 (planner sizing) / §P4
(composition horizon) · [`../../on_policy/COLLECTION_REALISM.md`](../../on_policy/COLLECTION_REALISM.md)
(tier C: on-policy collection mandatory, monitors charged).

No findings README yet — results get discussed before any writeup.

## The question

Round 1 settled the op taxonomy on **one** segment (H=20, at the plant's composition horizon) and
live content won everything: one CEM plan from the current forward model at the observed launch
state beat every frozen op **1.6× on error at 2.3× less priced time**, while frozen content rotted
by diet-narrowing. Round 2 asks where that stops being true.

> **Frozen, measured content re-enters exactly where the committed span exceeds the model's
> composition horizon.**

A phrase is 60 steps — ~3× past where a one-step FM's rolled-out tip trajectory is trustworthy — so
there is no live plan of the whole phrase to be had. A chain of measured, *executed* renditions has
no such bound, because the body produced it, not the model; it is limited by execution variance.

## Code files

| file | purpose |
|---|---|
| `world.py` | The substrate: the closed 4-leg loop, the plant + hard patch, the on-policy diet, the metered `Body` traversal under a **routing** (a partition of the drilled segments into contiguous committed groups), CEM-MPC with a **per-step goal schedule**, the compile ops at any span, the audition, and the priced `Ledger`. |
| `gates.py` | **Round 2a** — gates G5/G6/G7 plus CAL-C (cost shaping), CAL-P (planner sizing *per span*) and CAL-D (mastery clocks). One Modal container per world. |
| `legato.py` | **Round 2b** — the arm comparison, one Modal container per arm. |
| `analyze_gates.py` | Gate report + the calibration record the main run's knobs are read from. |
| `analyze_legato.py` | Reduces a round-2b run: headline, per-segment breakdown, commit record, the fusion comparison, itemised priced time, and the `d_fb × d_plan × d_delib` economics surface. |
| `launch_detached.py` | Session-isolated detached launcher (`--fn gates\|legato`). |

**Downstream, additive (2026-08-20).** `traverse` gained four pure readouts for
[`../span/`](../span/FILES.md): `acts_app` / `acts_app_raw` (the approach leg's issued and raw
commands, so consecutive laps of the closed loop concatenate into one contiguous executed command
sequence — G5a's 60-step probe was about to hit its own ceiling against a model reaching 58) and
`tips` / `tips_app` (the tip after every control step, by analytic FK, so an FM rollout can be
scored against the trajectory the body *flew* rather than against a `true_tips` replay that
re-enters through `set_state` from a float32 state). No RNG draw moves and nothing upstream reads
them, so every `l0` / `l1` / `l2` / `L1` result stays byte-reproducible.

## The piece — a closed 4-leg loop

`W0 = fk(q_center) = (0.1968, 0.7785)` is the start tip **and** the last waypoint, so leg D closes
the loop.

```
q0 --A approach, H=14, 0.300 m, mastered/frozen planner--> W1 (0.0359, 0.5248)
   --B drill 0,  H=20, 0.428 m--> W2 (0.3820, 0.2739)
   --C drill 1,  H=20, 0.453 m, THE PATCH--> W3 (0.6029, 0.6696)
   --D drill 2,  H=20, 0.420 m--> W0
```

The phrase = legs **B + C + D**, launched at W1, span **60 steps**.

| property | value | why |
|---|---|---|
| tempo | 0.0215 / 0.0214 / 0.0227 / 0.0210 m per step | uniform, and matched to round 1's drilled segment (0.44 m / 20 steps = 0.0220), so a legato segment **is** a fingering segment in difficulty class |
| turns | 86.4° / 96.8° / 104.2° (cos 0.062 / −0.118 / −0.245) | a closed quadrilateral's exterior angles sum to 360°, so its turns average **exactly** 90° — round 1's `cos ∈ [0.10, 0.75]` window is structurally unavailable to a loop, and the admissible band here is `[−0.25, 0.45]`. Every seam is a genuine direction change, which is what makes crossing one blind a real bet. |
| radius | every path point in [0.446, 0.901] of a 1.10 m arm; `x ≥ 0.036` | off the singular full-extension shell, in front of the base — while sweeping r from 0.45 to 0.90, so `M(q)` changes a lot along the loop |
| patch | Gaussian curl, σ = 0.08, on the **midpoint of leg C** — the MIDDLE of the phrase | gate weight **0.018** at every waypoint and ≤ 0.023 at the closest approach of every other leg (round 1: 0.023; étude: 0.044). Putting it mid-phrase means the boundary a phrase commitment gives up — W2, immediately before the hard passage — is exactly the one carrying the most information. The sharpest available form of the bet. |
| `curl_b` | 14.0 | round 1 swept {6, 14, 26} + an obstacle world; b14 was the design point on every gate. The knob **this** round sweeps is span, inside G5. |

Geometry found by local search over pure-numpy FK subject to the constraints above; verified against
MuJoCo's own `site_xpos` to 2.2e-16 in C0.

## What is new against `fingering/world.py`

`world.py` is a **copy-fork**, not a subclass — the piece, the traversal, the controller's goal
handling and the compile ops all change shape, and round 1 must stay byte-reproducible. Only the
pure helpers (`planar_jacobian`, `null_direction`, `start_postures`, `kmeans`) are imported from it.

1. **Per-step goal schedule** in `plan_fn`, so one CEM plan can span several segments and still play
   the piece's shape instead of cutting the corner to the last waypoint. With a constant goal the
   arithmetic is identical to round 1.
2. **Routing** — `traverse` executes a partition of the drilled segments into contiguous committed
   groups. Feedback events are charged **1 for the approach launch, then 1 per committed group
   launch, or one per replan inside a reactive group**. The routing *is* the price.
3. **Receding-horizon reactive control over the piece** (fixed lookahead, per-step goals, crossing
   seams), which reduces to round 1's reactive unit on a single segment.
4. **`Ledger.delib`** — CEM rollout-steps (`k_shoot × cem_iters × horizon`), counted alongside plan
   events and never charged. A phrase plan is *one event* but three segments' worth of *work*, and
   pricing them the same would hand the phrase arm a subsidy it did not earn.

## The arms (round 2b) — granularity × content

| arm | routing | content | fb / piece | plans / piece |
|---|---|---|---|---|
| `never` | live | live | 61 | 60 |
| `seg_plan_launch` | committed per segment | live | 4 | 3 |
| `seg_frozen` | committed per segment | frozen, keyed at each seam | 4 | 0 |
| `phrase_plan_launch` | committed per phrase | live (one H=60 plan) | 2 | 1 |
| `phrase_frozen` | committed per phrase | frozen chain, keyed **only** at the phrase launch | 2 | 0 |
| `phrase_chain_fixed` | committed per phrase | one chain, state-independent | 2 | 0 |
| `seg_frozen_proj` / `phrase_frozen_proj` | as above | supervised 1-D key | — | 0 |

Counts verified in smoke. `seg_frozen` is the control that makes the fusion claim attributable:
`phrase_frozen` differs from it by exactly two things — two fewer feedback events, and no mid-flight
re-keying.

**Why fusion is not vacuous here, where the étude proved it was.** The étude's committed units were
*state-independent* command sequences, so two adjacent units concatenated to a bit-identical
trajectory and removing the internal re-grounding was a no-op on the physics (accuracy cost exactly
zero, time win exactly one `d_fb`). Round 1 established that boundaries here carry information
(hand-over R² = 0.82), so these units are state-*conditioned*. A fused unit cannot re-key mid-flight;
it must choose its whole chain at the phrase launch. **G6 measures the information that gives up
before the arms are ever run.**

The étude's other collapse — "the phrase pool contains nothing but the concatenation" — also fails
here: because the segment units are state-conditioned, different traversals fire different key
combinations, so the pool of whole-phrase renditions holds genuinely distinct *measured* chains.

## Standing findings from gate data alone (these survive whatever round 2b becomes)

### F1 — Launch-time keying of a chain is near-inert; the information at internal seams cannot be substituted

Measured on `l0`, held-out (build the key on even score columns, grade on odd), so it is not a
selection artifact:

| what is keyed, and where | held-out gain, `l0` | held-out gain, `l1` |
|---|---|---|
| segment 0, keyed at its own seam | 1.13× | 1.01× |
| segment 1, keyed at its own seam | 1.89× | **3.58×** |
| segment 2, keyed at its own seam | 2.39× | 1.90× |
| **the whole 3-segment chain, keyed once at the phrase launch** | **1.08×** | **1.03×** |

**Qualified by F5**: this is the *audition-estimated* gain; the closed-loop gain was 1.34x, because a phrase pool harvested from segment-committed traversals is far more diverse than one harvested from reactive warmup. Reproduced across two cost regimes (`l0` and its repair `l1`, which changed the
controller's objective enough to move every other arm-level number), which is what makes this the
node's most durable result so far.

The per-state oracle at phrase level is 1.89× (`l0`) / 2.63× (`l1`), so the information *exists* — a
launch-time key just cannot reach it. The mechanism is measured directly: state spread along a fixed
chain grows **0.106 → 0.216 → 0.352** (`l0`) / **0.098 → 0.140 → 0.204** (`l1`) across the three
seams, against a phrase-launch spread of 0.0229 in both — a 9–15× dispersion by segment 2.
Consistently, R² of the chain's realised per-segment error on segment 2 is **0.492 from the phrase
launch vs 0.736 from its own seam** in `l0`, and **0.079 vs 0.535** in `l1` — fusion loses 0.24 and
0.46 respectively.

**So conditioning information is *local in time*, and commitment spends it.** This is the
quantitative form of what the étude could only assert: fusing is not free once units are
state-conditioned, and what it costs is exactly the conditioning it can no longer perform.

### F2 (methodology) — Calibrate to preserve the measurement axis, not to minimise an arm's error

`l0`'s CAL-C chose `vel_pen_mid` by *minimum reactive piece error*. It found 0.0 (error 0.126 vs
0.171 / 0.175) and thereby selected a configuration in which only **3%** of the reactive error was
attributable to forward-model quality (round 1's comparable figure: 65%), with the phrase-level
usable range outright **negative**. The knob optimised the number while destroying the axis the
entire round is a measurement of.

The rule this node now follows: **a shared calibration knob is chosen by the fraction of the outcome
that remains attributable to the variable under study**, on an arm-neutral reference, subject to a
competence guard so a uselessly bad controller cannot win by making the fraction large. Generalises
to any node with a "usable range" precondition — a calibration that maximises performance can
silently minimise *measurability*, and those are different objectives.

### F3 (negative) — there is no controller-level seam law: planning past your commitment does not help

The natural conjecture from F1 was that the seam law should reappear one level down, in the
controller: *a segment's objective ought to include the state it hands over, not just its own
notes.* `l2` tested it directly with a look-ahead weight γ — a live unit COMMITS to its span but
PLANS one segment further, with the look-ahead portion's cost discounted by γ (γ=0 is no lookahead,
γ=1 weights the next segment as heavily as this one). Swept against terminal `vel_pen`, on the
neutral criterion:

| γ | segment arm | phrase arm | reactive | neutral mean |
|---|---|---|---|---|
| **0.0** | **0.1003** | 0.1981 | 0.0074 | **0.1020** |
| 0.5 | 0.1217 | 0.1981 | 0.0074 | 0.1091 |
| 1.0 | 0.1244 | 0.1981 | 0.0074 | 0.1100 |

**γ = 0 wins at both `vel_pen` values, and every γ > 0 is monotonically worse.** The conjecture is
not supported. The mechanism that makes sense of it: a committed unit issues only the steps it
committed to, so weight spent on a segment it will *not* control buys a compromise it never gets to
cash — you pay the accuracy cost on your own span and hand the plan away before the benefit
arrives. Re-grounding at the seam supersedes anticipating it.

This is worth keeping precisely because F1 makes the opposite intuition so attractive. Seam
information is real and local (F1), but the way to use it is to **re-ground at the seam**, not to
anticipate it from upstream — which is, on reflection, the same thing commitment is buying out of.

### F4 — the crossover: live content wins inside the composition horizon, measured chains win beyond it

*Single seed (`L1`).* A seed pair (`L1_s1`/`L1_s2`) was launched 2026-08-20 and **cancelled ~15 min
in by Jasper (GPU budget)** — apps stopped, no partial results consumed; the launch config is in the
Gotchas section's reproduce line if a future hardening pass wants it. Per `recital`'s methodology
export, ranks are the claim; the within-run triangulation is that signs and ordering are identical
at the two horizons (c75 and c90) and the fusion control is double-sided. Reported at the common
horizon c75.

| span | live content | frozen content | winner |
|---|---|---|---|
| **segment**, H=20 — inside the horizon (stale FM composes to ~21–23 steps) | `seg_plan_launch` **0.0850** | `seg_frozen` 0.1065 | **live, 1.25×** |
| **phrase**, H=60 — ~3× past it | `phrase_plan_launch` 0.1874 | `phrase_frozen` **0.1026** | **frozen, 1.83×** |

Round 1's result (live content dominates every frozen op) reproduces *inside* the horizon and
**inverts** outside it. The node's own fusion control makes the inversion attributable, because each
pair differs by exactly two things — two feedback events, and mid-flight re-keying:

- frozen content, seg → phrase: **−0.0039 error (fusion is free)**, 2 fb/piece saved
- live content, seg → phrase: **+0.1023 error (2.2× worse)**

So fusing is free for a chain of *measured, executed* renditions and catastrophic for a *live plan*.
The mechanism is visible per segment: `phrase_plan_launch` reaches **0.4169** on segment 2 — error
compounding to the end of a 60-step model rollout — against `phrase_frozen`'s 0.1007. A measured
chain was produced by the body, not the model, so it carries no composition error; it is bounded by
execution variance instead. Independently, G5a (planner-free) puts the stale model's rollout
divergence at 0.031 → 0.389 → 0.728 m at 20/40/60 steps.

In steady state per traversal (74 steps, `dt=0.02`) `phrase_frozen` **dominates `seg_frozen` on both
axes** — equal-or-better error at 1.1× (`d_fb`=0.1) to 1.8× (`d_fb`=3) less time. Cumulative priced
time hides this, because it is dominated by the pre-commit phase every arm shares and by a phrase
audition that replays 60 steps × n_cand × n_score.

**Caveats.** `never` still dominates on error (0.0107) — round 1's rarity law, reproduced at phrase
level; no price vector rescues any committed arm, so the comparison against `never` is
error-dominated. And every committed arm froze content mid-descent (ballistic clock c72 vs commits
c25–55), which biases `never`-vs-committed but **not** the within-committed cells above, since all
committed arms share the schedule.

### F5 — the lower level's library funds the upper level's addressable variation

*Single seed (`L1`); the seed pair was cancelled (see F4's note).* F1 measured launch-time keying of a whole chain at
**1.03×** held-out and called it near-inert. In the closed loop it bought **1.34×**
(`phrase_frozen` 0.1026 vs `phrase_chain_fixed` 0.1374) — F1's audition estimate understated it by
roughly an order of magnitude in effect size.

The discrepancy is not noise, and it names a real dependency. F1's estimate came from a chain pool
harvested from *reactive warmup* traversals. `L1`'s phrase pool is harvested from a
**segment-committed** configuration, where state-conditioned segment units fire *different key
combinations* on different traversals — so the pool holds genuinely distinct measured chains rather
than one chain plus motor noise. That is exactly the property the étude lacked when it proved fusion
vacuous ("the phrase pool contains nothing but the concatenation").

**So a phrase-level library is only as addressable as the segment-level libraries underneath it
make it.** The lower level's state-conditioning manufactures the variation the upper level's key has
something to select among — which is a concrete argument for bottom-heavy assembly beyond the
scheduling one, and a warning that auditing a hierarchical op against a flat pool will understate it.

## The metering convention (what "the grade" is, and what follows from it)

A run-through traverses **one lap**: approach → drilled 0 → drilled 1 → drilled 2, ending at `W0`.
The grade `e_piece` is the **mean of the three waypoint arrival errors**, each measured at that
segment's own last step. The grade never references velocity.

So a single-lap run-through *does* have a genuine end, and whether the controller should be asked to
**stop** there is a separate choice from how it is graded — which is why the terminal `vel_pen` is
calibrated (`l2`) rather than assumed. Round 1 inherited "stop at the end" because its piece
genuinely ended; on a closed loop, treating the lap as a fragment of a continuing traversal is
equally coherent and makes all three segments symmetric (otherwise segment 2 is the only one whose
planning window reaches a terminal state, which confounds the span comparison, since a phrase span
always contains segment 2). The chosen convention is recorded in `cal_cost.chosen`.

## Calibrations (each set by measurement, all recorded in the run config)

| cal | what it sets | why it exists |
|---|---|---|
| **CAL-C** | `w_waypoint`, `vel_pen_mid`, `commit_lookahead` | Round 1 penalised terminal joint velocity unconditionally — right there, because its drilled segment *ended* the piece. On a loop, braking at every seam plays it staccato and would handicap segment-wise planning for a reason unrelated to commitment. Smoke measured reactive piece error **0.126 at `vel_pen_mid=0` vs 0.287 at 0.5** — a 2.3× effect that would have decided the round by accident. Calibrated on the **reactive** arm, which is arm-neutral, so a shared knob cannot favour either committed arm. `commit_lookahead` lets a live segment-wise unit *plan* past its span while *committing* only to it, so `seg_plan_launch` is not myopic at exactly the seams the round is about. |
| **CAL-P** | planner size **per span** | `arm_substrate` P3 bites harder here than in round 1: a segment plan optimises 60 action dimensions, a phrase plan **180**. Each span gets its own grid reaching *upward* from round 1's 1024:8, and the reported verdict is whether error is **still falling at the top of the grid** — if it is, that span is *starved* and any degradation G5b reports there is not attributable to composition. Elite count is held at a fixed **fraction** (1/32, reproducing round 1's 1024/32) so a wider search does not become a greedier one. |
| **CAL-D (not validated in `L1`)** | commit cycles — **the schedule `L1` actually ran was too early.** The gate's 20-cycle clocks were lower bounds (c17–c20); read off `never`'s own 75-cycle ladder the ballistic clock is **c72 and still not plateaued**, against commits fired at c25/32/39/55. Every committed arm therefore froze content at ~c25–39 competence while the model kept improving — round 1's original inversion, repeated. It biases `never`-vs-committed, but **not** the within-committed comparisons, which all share the schedule. |
| **CAL-D** | commit cycles | Round 1b's rule verbatim: trailing-3 mean of the held-out **ballistic** curve, plateau over the last 5 probes, first sustained entry within 1 sd. Measured per segment and for the phrase. Reading the *reactive* curve instead said "mastered" 39 cycles too early in round 1 — open-loop competence is the later of the two clocks. |

## The gates (round 2a)

| gate | asks | round-1 analogue |
|---|---|---|
| **G5** | Does live plan-at-launch degrade **structurally** with committed span? (a) FM composition divergence with **no planner in it** — the mechanism; (b) live error vs **re-grounding count** (3 / 2 / 1 groups over the same segments to the same endpoint), each span at its CAL-P best. | G4 |
| **G6** | Does the phrase launch carry usable information, and **how much does fusing give up**? R² of a chain's realised per-segment error on the phrase launch vs on its own seam. Replayed **with motor noise** — noiseless replay makes every downstream state a deterministic function of the launch and the measure would read ~1.0 by construction. | G1 |
| **G7** | Usable range + noise floor at phrase granularity, per control mode; plus the mastery clocks. | G3 |

## Modal volume layout

```
/data/practice_legato/<tag>/<world>/results.json     # round 2a (one dir per world)
/data/practice_legato/<tag>/<arm>/results.json       # round 2b (one dir per arm)
```

## Reproduce

```bash
cd experiments/            # MODAL_PROFILE=chromatic
# smokes (non-detached, always before a detached launch)
modal run mjc/practice/legato/gates.py::gates   --quick --tag lsmoke
modal run mjc/practice/legato/legato.py::legato --quick --tag Lsmoke

# round 2a -- the gates. l0 is the first pass (G5/G6 pass, G7 fails, and it is the run that
# exposed the grader/controller mismatch); l1 is the repair, with `w_waypoint` in CAL-C.
python3 mjc/practice/legato/launch_detached.py --fn gates --tag l0 --seed 0 --n-warm 36
python3 mjc/practice/legato/launch_detached.py --fn gates --tag l1 --seed 0 --n-warm 20
python3 mjc/practice/legato/analyze_gates.py --tag l1 --fetch

# round 2b -- the arms (commit cycles and calp read off the l0 report)
python3 mjc/practice/legato/launch_detached.py --fn legato --tag L1 --seed 0 [...]
python3 mjc/practice/legato/analyze_legato.py --tag L1 --fetch
```

## Gotchas

- **Launcher logs must live outside the mounted package.** `shared.py` mounts `mjc` with
  `add_local_python_source("mjc")` and Modal hashes the whole directory, so a detached launcher
  appending to a log *under* `mjc/` makes every *other* concurrent `modal run` die with
  `ExecutionError: <path> was modified during build process`. The failure names the log file, not
  the run that is actually broken. This node's launcher writes to `experiments/.launch_logs/`.
- **A library whose cells all pick the same candidate is a `fixed` unit wearing a library's name.**
  Round 1 hit this (2 of 4 duplicate cells at round 0). `n_distinct == 1` now logs a loud `[WARN]`
  at commit time and the reducer flags it, because a silent collapse would make the fusion
  comparison vacuous without anyone noticing.
- **`never` at 90 reactive cycles exceeds the 8 h Modal function timeout.** `L1`'s `never` was
  killed by `timeout=28800` at c75 of 90 (client shows `RemoteError`; the volume has
  `results.json` but no `done.txt`). A reactive cycle costs ~60 CEM plan calls for practice plus
  ~60 for metering, and the ladder adds a full reactive probe every 3 cycles — so `never` is ~3×
  the wall-clock of any committed arm and scales with `n_cycles`. Budget it: raise
  `timeout` to 43200 for `n_cycles ≥ 90`, or cap `n_cycles` at ~60 for `never`. The partial was
  fully usable (probe series intact, monotone `t_cum`, no NaN), and the round was reduced at the
  **common horizon** per the étude precedent (`eg_s0`, lost at c42) — `analyze_legato.py` now does
  this automatically and flags it.
- **Tags differing only in case collide locally on macOS.** The gate runs use `l0/l1/l2` and the
  main runs use `L1`; on the Modal volume (Linux, case-sensitive) these are distinct, but
  `results/l1` and `results/L1` are the **same directory** on a default macOS filesystem, so
  `--fetch` merges a gate's world output into a main run's arm directory. `analyze_legato.load()`
  now ignores any `results.json` without an `arm` key, which makes the collision harmless — but
  prefer tag namespaces that differ by more than case (e.g. `g0/g1` for gates, `m1/m2` for main
  runs) in future nodes.
- **Size the CEM to the action-sequence dimension** (P3), and *per span* here — see CAL-P.
- **The grader and the controller must score the same thing.** `l0` graded arrival at each waypoint
  *at its seam step* while every planner minimised *mean distance over a window that spans the
  seam* — so a window crossing a seam was rewarded for leaving waypoint k early to get a head start
  on k+1. It cost `never`, the reference arm, a factor of 6 (reactive segment-0 arrival 0.194
  against 0.032 for a committed unit whose window ends at the waypoint, same launch states, same
  model), flattened the FM-attributable fraction to 3%, inverted ballistic-vs-reactive, and made
  `commit_lookahead=20` measure *worse* than 0. The tell was reactive per-segment error
  **[0.194, 0.165, 0.010]** — fine only on the last segment, the one with no next waypoint to be
  pulled toward. Fixed in `l1` by `World.waypoint_mask` / `w_waypoint`, applied through `plan_fn` to
  **every** planner (reactive, segment, phrase, approach) so no arm gets a private objective;
  `w_waypoint = 0` reproduces `l0` bit-for-bit. Committed units are *not* bit-unchanged by the
  repair (their group-boundary step gains the weight), so CAL-P cells are re-measured under it.
