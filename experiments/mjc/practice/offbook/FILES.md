# offbook — file index and design record

**Up**: [`../README.md`](../README.md) (practice) · [`../../README.md`](../../README.md) (mjc)
**Donor (forked verbatim)**: [`../legato/`](../legato/FILES.md) — `world.py` is a copy-fork,
`legato/` is untouched and every `l0/l1/l2/L1` result stays byte-reproducible. Gate **G-F** asserts
the fork still reproduces the donor traversal bit-for-bit rather than trusting that.
**The ports' code shape**: [`../../../rhm/practice/native/`](../../../rhm/practice/native/README.md)
(`prop_net.py`, `span_net.py`, `SPEC.md`'s gate discipline) ·
[`../../../rhm/practice/spiral/SPEC.md`](../../../rhm/practice/spiral/SPEC.md) (the two-phase
discipline this node follows).
**Constraints from the mjc side**: [`../fingering/`](../fingering/README.md) (the op taxonomy, G1,
pricing incl. the plans counter) · [`../legato/`](../legato/README.md) (F1–F5; the calibrated `l2`
cost cell) · [`../span/`](../span/README.md) (F2/F4 — why a live plan cannot be the consolidated
object) · [`../README.md`](../README.md) (the standing apparatus cautions).

**No findings README yet — numbers get discussed before any writeup** (repo policy). There is no
`SPEC.md` for this node by design: the orchestrator's prompt is the record of what was asked, and
this file is the record of what was decided and why.

## The question

Every committed unit on this substrate has been an **external object**: a keyed tape
(`fingering/`) or a live CEM plan at launch (`legato/`). The RHM side ran the port back and found
that routing consolidates into the planner, the corridor into the executor, and the table survives
as the *address book* — needed for growth, not performance. This node asks the same question where
the units are motor and the model has a **composition horizon**.

> Does a committed motor vocabulary consolidate into the learner — proposed and run as one unit,
> its members no longer enumerated — while the library table stays outside as the mining substrate?

## The mapping (the design, and why each piece is what it is)

**The seam is the decision point; the library is the action set.** At each re-grounding point the
agent chooses among {library units at each level} ∪ {the live plan}.

| piece | what it is here | why |
|---|---|---|
| **enumeration** | **seam-time audition** — from the *realized* seam state, roll every candidate's stored commands open-loop under the current forward model and score its waypoints; launch the argmin. Priced per materialisation (`aud`) and per FM rollout-step (`delib`). | The op this node introduces. Independently licensed by legato **G6** (a seam is handled *at* the seam, or by content *selected on realised seams*) and never run: `fingering/`'s audition only ever fired at **commit** time, in the **plant**, on **recorded** score states, and legato's consumption-time selection was a frozen k-means key. |
| **the action set** | ns = 1 (a segment tape) and ns = n_seg − k (a chain to the end of the piece: 40–60 steps) | legato **F4**: a unit must exceed the plant's composition horizon (~21) or chunks do not pay at all. The chain level does; the segment level deliberately does not, so the crossover is inside the action set rather than assumed. |
| **what may be consolidated** | measured, **executed** command tapes (post-noise `acts`), with the seam state they were launched from and the error they actually achieved | `span/` **F2/F4**: a live plan is a model's promise off-distribution and practice makes the promise *worse*. Nothing modelled or planned enters the library. |
| **Port 1 (routing)** | π(unit-slot \| seam state, seam index) → top-k slots auditioned; trained **online** by self-imitation on the agent's own successful traversals; ε-exploration in practice only | `native/prop`'s measured closed-loop lock-in, sharper here: a slot never auditioned also never gets a seam-time score, so nothing else can rescue it. Read is the seam **posture** — `fingering/` G1's measured object (tip sd 0.031, posture spread 0.29 rad, per-state oracle 4.0×). |
| **Port 2 (corridor)** | span head (slot, seam posture, **FM trunk read**) → the unit's measured **commands**, behind a per-slot parity gate re-checked every cycle | Identity is the *input*, content the *target* — `teacher_slot/handle/`'s measured negative says identity-as-target is poison. On the FM trunk **deliberately**: the interference question is the treatment, and the plant guard is what has to say so. |
| **the address book** | the library table, outside; slots fit **once** and frozen; chain spellings over segment slots | legato **F5** is the measured seed (the segment library manufactures the phrase pool's addressable variation), which is why chain cells are grown *later*, from already-routed traversals. |
| **the rent** | a declared per-decision deliberation budget D (FM rollout-steps) spent first on audition, remainder buys the live plan's CEM width via `fit_cem` | `ratchet`'s `fit_width(budget, G)` in the arm's own currency. `tall/`'s widening-action-set rent, which routing is claimed to dissolve. `delib_budget ≤ 0` turns the conversion off and reports the rent as a cost instead. |

## Decisions taken, with reasons (the things a reader could disagree with)

1. **The primitive action is `plan_launch` over the current segment, not full reactive MPC.**
   A reactive controller re-grounds every step, so it has no command sequence until it has already
   been run and cannot be auditioned against a stored tape at all. `plan_launch` is
   `fingering/`'s measured champion inside the horizon and produces a comparable object. Full
   reactive MPC stays where it belongs — as the `never` reference arm.
2. **Pre-commit, every arm plays the donor reactive path** (`legato.World.traverse` with a reactive
   routing). This is legato's `sched_late` property: every arm runs a bit-identical stream up to
   its first commit, which is what makes a single seed readable. `traverse_route` engages only once
   the library is non-empty.
3. **Slots are fit once by k-means over CONTENT and frozen.** A growing library would otherwise
   re-partition every commit and π's logits, the span head's conditioning and the trust series
   would all be measuring a moving target. This is `native/prop_net`'s "(level, node), not a
   position in `ms`" on a substrate where the action set is a pile of tapes.
4. **No commit-time plant audition.** legato ran `n_cand × n_score` plant replays per commit; at
   K = 288 per cell that would be ~10⁶ charged environment steps per commit and would dominate the
   ledger. Instead every entry carries the error it **actually achieved when it was executed** —
   free, already recorded, and epistemically better (it is a measurement, not a replay). The
   winner's curse (étude E-3b) is therefore confined to `key_frozen`'s within-slot representative
   and to the poison construction; every arm whose claims this node carries selects by seam-time
   audition, which scores on states the tape never saw.
5. **Per-performer decisions.** Two performers arriving at different postures may commit to
   different *levels*, so each carries its own next-decision step while the body still steps all B
   in lockstep. Anything else would make the level choice a batch-level artifact.
6. **Port 2's targets are captured on the metering traversal, for EVERY selected slot** — not only
   the launched one. `native/span`'s convention is that every macro call feeds its slot's buffer;
   here the reason is sharper, because a live plan can win most decisions at a loose budget and a
   head that only learned from launched calls would starve exactly when the corridor most needs
   consolidating. The extra within-slot audition that refreshes the targets is charged like any
   other audition.
7. **Parity is measured as execution reproduction under the same model**, on held-out seam states
   split by a deterministic code of the state (so a recurring posture cannot straddle the split —
   `native/span`'s measured trap). Continuous fractions are logged beside the verdict so a
   different τ can be read off the record.
8. **`key_frozen` is the taxonomy's continuity arm, not a strict one-variable neighbour of
   `audit_all`.** A keyed frozen library has no way to key a live plan, so it has no primitive
   option — which is exactly what legato's `seg_frozen`/`phrase_frozen` were. The strict
   one-variable chain that carries this node's claims is
   `audit_all → audit_prop_k → route_native → fid`. Said out loud rather than papered over.

## The arm ladder

| arm | decision rule | the step it isolates |
|---|---|---|
| `never` | reactive MPC throughout — the **donor code path verbatim** | the reference; legato's `never` |
| `key_frozen` | nearest key centroid → that slot's representative tape | legato's frozen-key op in this harness (continuity, not a one-variable step) |
| `audit_all` | seam-time audition over the whole action set, O(K) | the enumeration reference |
| `audit_prop_k` | **Port 1**: π ranks slots, only top-k auditioned, O(k) | routing |
| `route_native` | **Port 1 + Port 2**: an open slot contributes one head emission instead of all its members | + corridor |
| `fid` | both ports wired and **shut** | bit-identical to `audit_all` for the whole run |
| `audit_prop_kN` | π live, k = every legal slot | the in-run form of gate G-P; bit-identical to `audit_all` |

**End-of-run battery** on the final trained state: `no_prim` (primitives ablated — can it play on
chunks alone?), `no_table` (the address book deleted; π + the span head must serve; current-level
error *and* phrase-level minability, counted table-free from the agent's own chosen traversals),
`restored`. `fid`'s untrained heads under the same forced-open policy are the negative control.
**The poison twin**: one plausible-but-bad address — genuine measured renditions of the right
segment harvested from pre-competence cycles — pinned to a reserved slot. Enumeration must
materialise it at every seam; does π starve it? (`native/` finding 6.)

## Code files

| file | purpose |
|---|---|
| `piece.py` | The piece's two constants, and nothing else — a module that imports **nothing**, because a Modal *local* entrypoint runs on a machine with no numpy and importing `legato/gates.py` for them registers legato's entrypoints on the shared app (see Gotchas). |
| `world.py` | The verbatim fork of `legato/world.py` plus: `Ledger.aud`; `World.fm_trunk_t` / `rollout_score` / `aud_checkpoints` / `audition_fm` (seam-time audition) / `fit_cem` (the budget conversion) / `train_online_span` / `traverse_route` (the dynamic per-performer traversal); `Cell` + `Library` (the address book); `RoutePolicy` (the decision rule, one mode per arm). |
| `nets.py` | The two ports: `SlotLayout`, `isolated_rng`, `build_prop` / `select_slots` / `explore_slots` / `PropTrainer` / `prop_probe` (Port 1 + the trust and can't-decompose readouts), `build_span` / `SpanBuffer` (Port 2). |
| `gates.py` | **Phase A** — G-F (fork fidelity), G-R (rng discipline), G-P (π fidelity at k=N), G-K (the rent gate), G-B (does a chunk ever pay?), G-S (pool spread), G-C (audition calibration), G-D (cycle cost). |
| `offbook.py` | **Phase B** — the arm comparison, one Modal container per arm, plus the end-of-run battery. |
| `analyze_gates.py` | The gate report and the calibration record the main run's knobs are read from. |
| `analyze_offbook.py` | Reduces a main run (headline, routing mix, trust formation, can't-decompose, parity, plant guard, consumption-phase ledger, battery, poison) + `::figs`, a **remote** figure job. |
| `launch_detached.py` | Session-isolated detached launcher (`--fn gates\|offbook`). |

## Modal volume layout

```
/data/practice_offbook/<tag>/gates/results.json     # Phase A
/data/practice_offbook/<tag>/<arm>/results.json     # Phase B (one dir per arm)
/data/practice_offbook/<tag>/figs/*.png             # figures, written remotely
```

## Reproduce

```bash
cd experiments/            # MODAL_PROFILE=chromatic
# smokes (non-detached, always before a detached launch)
modal run mjc/practice/offbook/gates.py::offbook_gates --quick --tag gsmoke
modal run mjc/practice/offbook/offbook.py::offbook     --quick --tag Osmoke

# Phase A
python3 mjc/practice/offbook/launch_detached.py --fn gates --tag g0 --seed 0
python3 mjc/practice/offbook/analyze_gates.py --tag g0 --fetch

# Phase B  (knobs read off the Phase A report — see "Phase A record" below)
python3 mjc/practice/offbook/launch_detached.py --fn offbook --tag O1 --seed 0 [...]
python3 mjc/practice/offbook/analyze_offbook.py --tag O1 --fetch
modal run mjc/practice/offbook/analyze_offbook.py::figs --tag O1
```

## Gotchas

- **Never import `legato/gates.py` (or any sibling runner) from this node.** `mjc/shared.py`
  exposes ONE Modal `app` for the whole package, so importing another node's runner registers its
  `@app.local_entrypoint()`s and `modal run` dies with `InvalidError: Duplicate local entrypoint
  name: gates`. It also silently overrides same-named functions
  (`Warning: function name 'run_gates' collision!`). This node's gate entrypoint is therefore
  `offbook_gates` / `run_offbook_gates`, and the piece constants live in `piece.py`.
- **A Modal *local* entrypoint runs locally**, on a machine that in these sessions has neither
  numpy, torch nor mujoco. Anything reachable from module scope of a runner must import none of
  them (the `arm_env.py` contract). `analyze_offbook.py` inserts `experiments/` on `sys.path` so
  the same file works under both `python3 …` and `modal run`.
- **Figures are drawn remotely** (`analyze_offbook.py::figs`) for the same reason.
- **Launcher logs live outside the mounted package** (`experiments/.launch_logs/`): `shared.py`
  mounts `mjc` with `add_local_python_source("mjc")` and Modal hashes the whole directory, so a
  live log under `mjc/` makes every *other* concurrent `modal run` die with
  `ExecutionError: <path> was modified during build process`, naming the log rather than the run
  that is actually broken.
- **`never` and the wall clock.** legato's `never` was killed by `timeout=28800` at c75 of 90 — a
  reactive cycle costs ~60 CEM plan calls for practice plus ~60 for metering and the ladder adds a
  full reactive probe. This node runs `timeout=43200` **and** caps reactive arms separately
  (`--n-cycles-react`), sized off gate G-D. The reducer reports everything at the **common
  horizon** and flags truncation (the étude `eg_s0` / legato precedent).
- **Tag namespaces should differ by more than case** (legato's macOS collision). Gates use `g*`,
  main runs `O*`; the reducer additionally ignores any `results.json` without an `arm` key.
- **`delib` and `aud` are counted, never charged.** `d_delib` stays 0 and the price surface is
  swept post hoc, as in `fingering/f1c` and `legato/`. Adam is invariant to global loss rescaling
  (`priced_plasticity/`), so `span_lam` acts only through the *relative* weight of the two loss
  terms and must never be read as a plasticity budget.

## Phase A record (`g0`, seed 0, 2026-08-25 — full report at `results/g0/gate_report.txt`)

Warm-up 20 reactive cycles (batch 16, 15.7 s/cycle); library grown to **K = 288 per cell**
(`1:0`, `3:0`, `1:1`, `2:1`, `1:2`), `n_slot = 8`.

**The three assertions pass exactly.** G-F fork fidelity max\|Δ\| = **0.000e+00** over
`e_piece/e_seg/acts/acts_raw/tips/launch/final`, with `fb` and priced time equal. G-R: minting
either head leaves the shared torch stream untouched and π's output layer is exactly zero.
G-P: π at k = every legal slot reproduces full seam audition **bit-for-bit** on commands, level,
slot, source, score and candidate count.

**The measured O(K) → O(k) cut** (one seam, per performer-decision):

| k | materialisations | audition rollout-steps | candidates/performer |
|---|---|---|---|
| ALL | 4616 | 184,480 | 577.0 |
| 8 | 1945 | 70,620 | 243.1 |
| 4 | 1092 | 48,560 | 136.5 |
| **2** | **531** | **23,180** | **66.4** |
| 1 | 286 | 11,480 | 35.8 |

Routing at k = 2 is **8.7× fewer materialisations / 8.0× fewer rollout-steps**.

**G-S** (posture spread ÷ matched repeat-noise): seam 0 **5.17×** (reproducing `fingering/` G1's
5.65×), seam 1 2.27×, seam 2 **1.12×** — at the closing seam the arm is moving fast (joint speed
7.93 vs 2.30 at seam 0) and arrival spread is at its own noise floor. Tip ratios 1.07–1.66×
throughout: the information is **postural**, invisible in tip space, exactly as round 1 found.

**G-K(a) — the decisive negative. The planner ladder is not monotone and is unusable as a
currency**: 32:0.0452 · **64:0.0249** · 128:0.0349 · 256:0.0369 · 512:0.0480 · 1024:0.0448 ·
2048:0.0485. Past k = 64 more search is neutral-to-harmful — `arm_substrate` P3's sign-inversion
regime and `span/`'s exploitation gap (16× search buys a lower *believed* error; believed vs true
r = 0.01), both already on the record. **Consequence: the budget conversion has nothing to buy.**
The rent table is 0.0000 at every D where both policies afford a rung, and **−0.0100 at D = 40,960**
(enumeration's *narrower* planner is better). The rent binds as a **cost** and not as an outcome, on
this plant, for a measured reason. Phase B should therefore run with `delib_budget = 0`.

**G-K(b) — a bigger library buys rent, not competence.** ns=1: 1:0.0699 · 4:0.0658 · **16:0.0472** ·
64:0.0466 · 144:0.0475 · 288:0.0482 — flat past K ≈ 16. ns=3: 1:0.4157 · 4:0.3145 · 16:0.3097 ·
**64:0.2084** · 144:0.2325 · 288:0.2903 — *worse* past K = 64. With chain-level audition poorly
calibrated (G-C), a larger pool mainly gives the model more chances to mis-rank, so growth is
actively harmful there. K = 288 is kept because it is what makes enumeration expensive; nothing
about it is a competence claim.

**G-C — the `aud_horizon` calibration, and a constraint on the op itself.** ns=1: ρ = **0.924**,
regret 0.0313 m, per-state oracle **2.23×**, 6/8 slots visited. ns=3: ρ = **0.337**, regret 0.1378,
oracle 2.71×, 6/7 slots. Horizon truncation makes the chain level **worse** (regret 0.1378 →
0.1809), so **`aud_horizon = 0` (full span) is the calibrated setting** — the opposite of what
truncation was introduced to fix, and it is on the record. Optimism is 2.5×/1.9× (the model
under-predicts realised error even on matched checkpoints), which is a *level* bias and does not
touch ranking. The distinctness check passes at both levels: the library is not a `fixed` unit
wearing a library's name.

**G-B — underpowered as run; do not read its ordering.** `chain_only` **0.1445 at fb 2** beats
`seg_only` **0.1552 at fb 4** (1.07× better error at half the feedback events — legato F4's
crossover inside this node's action set, and content-only so it is constant in D, as it must be).
But `prim_only` swings **0.0763–0.1829 (2.4×) across D at fixed content**, tracking the ladder's own
non-monotonicity, and that scatter is larger than the seg-vs-chain gap. One batch of 16 cannot
resolve the ordering. Phase B's ladder (40 probes × 48 held-out geometries) is the properly powered
version of this question, and `frac_prim` / `frac_chain` are logged at every probe.

**G-D — Phase B sizing.** Reactive practice cycle 15.7 s at batch 16; a committed traversal 1.74 s
(`audit`) / 1.55 s (`prop`) at batch 16. Scaling to batch 24 with metering at 48 and a ladder probe
every 3 cycles, a committed arm is **~30–40 s/cycle** (120 cycles ≈ 1.5 h) and `never` is dominated
by the reactive `e_react` reference inside the ladder — legato measured its own `never` at ~384
s/cycle in the same cell, so **cap reactive arms at ~90 cycles** and reduce at the common horizon.
`timeout = 43200` is already set.

### What Phase B should be run with, and why

| knob | value | reason |
|---|---|---|
| `delib_budget` | **0** (conversion off) | G-K(a): the planner ladder is non-monotone, so freed budget buys nothing. The rent is reported as a cost, which *is* measured. |
| `aud_horizon` | **0** | G-C: truncation raises chain-level regret 0.1378 → 0.1809. |
| `lib_add` / commits | 24 × 3 → **K = 72/cell** | at the measured plateau — see below. |
| `n_poison` | 8 | scaled with K (11% of a cell, as 24 was of 288). |
| `n_slot` | 8 | G-C: 6/8 and 6/7 slots visited by the per-state oracle — the partition is live at both levels. |
| `k_prop` | 2 | k = 1 is pure argmax-following, which `native/` finding 1 measured as never reaching enumeration's success. |
| `eps_act` / `explore_eps` / `force_window` | 0.10 / 0.15 / 3 | forced by G-C — see below. |
| `n_cycles` | 120, `n_cycles_react` 90 | G-D; reduce at the common horizon. |

### Why K sits at the plateau, and why that does not weaken the routing claim

G-K(b) measured competence **flat past K ≈ 16** (ns=1) and **degrading past K = 64** (ns=3). A
library far beyond that would manufacture the routing arm's advantage: the rent reported would be
rent no competent agent needed to pay. So `lib_add = 24 × 3 commits → K = 72/cell`, at the plateau
for both levels.

What this costs the claim: **nothing, because the cut is a property of the slot partition, not of
K.** At K = 288 the ladder was 577 → 66.4 candidates/performer (8.7×); at K = 72 it is
≈ 152 → ≈ 19 (≈ 8.0×). `k_prop = 2` of `n_slot = 8` selects a quarter of the slots at each of two
levels either way; K sets the *absolute* price and the partition sets the *ratio*. The reducer
prints the rent rescaled to K ∈ {16, 64, 288} (exactly linear here, since every cell is grown by
the same `lib_add`), so both the honest number and the counterfactual are on the record.

What it costs elsewhere, stated: ~9 members per slot instead of ~36, so Port 2's within-slot cut is
9 → 1 rather than 36 → 1, and the chain-spelling pool the battery's minability counts over is
thinner. If minability comes back degenerate, that is the first thing to suspect.

### Why exploration had to be added at the ACTION, not only at the audition set

G-C is the binding constraint on this node's op: seam-time audition ranks segment tapes well
(ρ = 0.924) and chains barely at all (**ρ = 0.337**) — `span/` F2 arriving as a constraint on the
*operator* rather than on the content — and in the gate the audition committed a chain **0.00 of
the time at every budget**. Widening what gets *scored* therefore cannot get a chain executed: the
ranker deciding among the widened set is exactly the thing that is blind there. Three mechanisms
are live, and all three were needed:

1. **`explore_eps = 0.15`** — one uniformly-drawn not-proposed legal slot enters the *audition set*
   in practice. Covers chain slots (they are ~half the legal set at seams 0 and 1). This is
   `native/prop`'s measured lock-in fix, and on its own it is not sufficient here.
2. **`force_window = 3`** — for three cycles after a cell is grown, **every** slot of that cell is
   forced into the audition set regardless of π's (untrained) logit. `native/prop_net`'s `forced`
   idiom at commit; with chain commits at c35/38/41 this forces chain audition continuously
   c35–c44.
3. **`eps_act = 0.10`** — ε-greedy on the **launched action**: practice sometimes plays a
   uniformly-drawn slot (uniform over *slots*, so a populous slot cannot crowd out a sparse one)
   from its own audition set. The **body** then grades the result. This is the only channel by
   which a level the model-graded op cannot rank can enter π's targets at all.

And the credit is execution-graded by construction: `PropTrainer`'s targets are filtered by
`e_piece` — the mean of the three waypoint arrivals the **body actually reached** — never by the
audition score. Selection before regression, with the selection done by the plant. If the run's
headline turns out to be *trust formed from realised use where the model-graded op is structurally
blind*, this is the apparatus that makes it visible; if instead chains never earn trust despite all
three channels, that is a much stronger negative than a run in which they were never executed.

ε-greedy applies uniformly to every committed mode (`audit`, `prop`, `native`), so it is not a
second variable between `audit_all` and `audit_prop_k`; and because the draw is one per performer
per decision on a dedicated stream, the twins (`fid`, `audit_prop_kN`) stay bit-identical to
`audit_all`, whose audition sets are the same.

### The battery and the poison twin

Both are end-of-run probes on the final trained state, not arms, so they are not in the six-arm
string: `no_prim` / `no_table` / `restored` run for every arm with a library, and the poison twin is
created at the first segment commit whenever `n_poison > 0`. **If the wall clock forces a cut, the
poison twin is the more cuttable of the two** (`--n-poison 0`): the table-ablation battery is the
node's headline prediction measured in both directions, while the poison contrast is a replication
of `native/` finding 6 on a new substrate.


## Phase B record (`O1`, seed 0, 2026-08-25 → 08-26)

Launched `--n-cycles 120 --n-cycles-react 90 --delib-budget 0 --aud-horizon 0 --n-slot 8
--k-prop 2 --lib-add 24 --n-poison 8 --eps-act 0.10 --force-window 3`, arms
`never,key_frozen,audit_all,audit_prop_k,route_native,fid`.

**The run ended by cancellation — cause DIAGNOSED 2026-08-26, see the Gotcha below.** It was not
external and not a manual stop: the launch invoked a Modal **local entrypoint** under `--detach`,
which does not protect the run, and the app died when the launching client process was reaped. The
signature is exact: at 02:12:01–02:12:04 the two inputs still in flight were cancelled while the
four already-completed ones were untouched, which is precisely what happens when a blocking
`.map()` on the client dies. State on the volume:

| arm | cycles | complete | battery |
|---|---|---|---|
| `key_frozen`, `audit_all`, `audit_prop_k`, `route_native` | **120/120** | yes | ran (`base`/`no_prim`/`no_table`/`restored`) |
| `never` | 57/90 | no | none — and none is *defined*: `never` has no library by construction |
| `fid` | 69/120 | no | none (truncated before the end-of-run block) |

**How the truncation is handled** (legato's precedent, and the étude's `eg_s0` before it): every
claim involving a partial arm is read at the **common horizon c57** and flagged in the table
header; every within-committed claim is read at **c120**, where all four completed arms share a
commit schedule and are unaffected by another arm dying. `fid`'s twin assertion is stated as
holding over the **120 probe values it reached (through c69)**, not over the full run. The reducer
was patched to tolerate `complete=False` partials rather than anything being relaunched; no new GPU
was spent.

Reduced report: `results/O1/report.txt`; figures `results/O1/figs/fig1…fig4.png`.


## Round 2 — the currency round (`O2`, seed 0, launched 2026-08-26 05:32 UTC)

**The question.** In O1 the chain level never earned any trust: pi's per-level mass stayed
<= 0.010 and ended at 0.000-0.001, and the routed arms executed a chain 0.000 of the time at every
probe -- despite all three exposure channels (eps 0.15, force_window 3 at each chain commit,
eps_act 0.10) firing. The leading candidate explanation is a **currency mismatch**: chains do not
pay in meters, they pay in FEEDBACK EVENTS (legato's steady-state table; `key_frozen` ran 2.7
fb/piece against 4.0 here), while pi's self-imitation filter reads raw `e_piece`. Round 2 asks
whether trust in the chain level forms when pi is graded in the currency chunks actually pay in.
Held loosely: **the trust series is the readout either way**, and "no, even then" is a real result
about this substrate, not a failed round.

**The treatment, and why this form.** `prop_priced` = `audit_prop_k` with one change, an
IMPORTANCE WEIGHT on pi's self-imitation buffer. Two rejected forms are on the record because the
rejection is the design:

* a **quantile filter** over `fb` (keep the cheaper half of the competent set) was the first form
  and fails silently: cheap traversals are rare, so the quantile sits on the mass at `fb = 4` and
  selects nothing -- the dose would have been zero without announcing it.
* a strict **`fb == min`** filter has the opposite failure: it drops ~11 of 12 competent traversals
  per cycle and starves pi of the segment-level signal it *did* learn in O1.
* **replication** does neither. Stage 1 is O1's competence filter verbatim (same bar, same targets,
  so the treatment is a strict addition rather than a replacement), and a traversal that saved
  feedback events counts `pi_fb_rep` times. It cannot starve a signal and cannot be silently zero.

**No exchange rate is invented.** Priced time is `steps*dt + fb*d_fb`; every performer plays all 74
control steps, so the only term that varies across performers is `fb` -- and ranking by priced time
is ranking by `fb` for *any* positive `d_fb`. The declared price therefore never enters the
arithmetic, which removes the one knob that could have manufactured the result. "Cheap" is
structural too: `fb < 1 + n_seg`, i.e. the performer committed at least one unit spanning more than
a segment. Dose `pi_fb_rep = 8` (strong end, per the canary instruction); `pi_fb_rep = 1` is the
control and is bit-identical.

**Exposure is unchanged from O1** (`eps_act 0.10`, `force_window 3`, `explore_eps 0.15`), so what
pi is graded on is the only variable.

**The control is cross-tag and exact by construction**, not by replay: the round-2 machinery is
ADDITIVE flags on `offbook.py` whose defaults reproduce O1's code path, so O1's `audit_prop_k` *is*
the control. Asserted, not assumed -- two gates in `--quick`:

| gate | result |
|---|---|
| `audit_prop_k` under the O2 runner vs under the pre-patch code (`Osmoke3`) | max\|Δ\| = **0.000e+00** over 122 values |
| `prop_priced` at `pi_fb_rep = 1` vs `audit_prop_k` | max\|Δ\| = **0.000e+00** over 122 values |

**Battery extension.** A `no_table_no_plan` cell is added: the address book deleted *and* the live
planner removed. O1's `no_table` cost the routed arms only +0.005, but with Port 2 shut, deleting
the table just falls back to the live plan (`frac_prim` -> 1.00) -- so that number measured the
PLANNER's competence, not the table's dispensability. The new cell is the unconfounded read.

**Retro on O1 is not possible and was not respent.** O1 persisted no model states (no `torch.save`
in the runner; the volume holds only `results.json` and `done.txt`), so the same cell cannot be run
on O1's arms without re-training them. Recorded rather than paid for.

**Winner's-curse instrument.** O1 persisted only the mix summaries, so chosen-vs-realised audition
optimism was not recoverable and the mechanism had to be inferred from its consequence (below).
`decision_mix` now logs `aud_pred` / `aud_real` / `aud_optimism` per probe, so the direct test is
free from here on.

**Port 2 is deliberately deferred, as scoping and not as a negative.** In O1 twenty slots acquired
held-out sets, none cleared tau = 0.9 (end fractions 0.63/0.55/0.55/0.40/0.33), and the head served
0 of 5,482 unit calls -- so the corridor port is *untested*, not refuted. Round 2 changes one
variable (pi's currency) and adding a second would make neither readable; the span head is left out
of the arm entirely so the round stays cheap.

### Free result from O1's logs (no GPU) — the winner's curse, at consequence level

Paired probes, same seed, same commit schedule, bit-identical until the first commit:

| era | `audit_all` cand/dec | mean e_perf gap (`audit_all` − `audit_prop_k`) |
|---|---|---|
| segment library only, c21–34 | ~47 | **−0.0095** (enumeration slightly *better*) |
| + chain library, c41–120 | ~124 | **+0.0240** (enumeration worse) |

corr(argmax width, gap) = **+0.496** over 34 probes; enumeration is worse in **31/34**. The sign
*flips* as the argmax widens. That is the winner's-curse signature -- at a narrow argmax extra
candidates buy choice; at a wide argmax over a scorer G-C measured as noisy they buy optimism -- and
it is independent support for reading routing's error win (0.0460 vs 0.0576 at c120) as **selection
hygiene rather than better content**. Caveat: the two arms also differ in composition
(`audit_all` executes chains 2.1% of decisions, the routed arms 0.0%) and the era split coincides
with the chain commits, so era and width are partially confounded; the within-era correlation and
the 31/34 sign consistency are the claim, not the point estimates.


## Gotcha (diagnosed 2026-08-26, after it cost two runs)

**`modal run --detach <file>::<local_entrypoint>` does NOT keep a run alive.** This node's launcher
pointed `ENTRY` at `offbook.py::offbook`, which is an `@app.local_entrypoint()`. Two independent
sources said so before either run died, and both were in front of me:

* Modal printed it in **both** launch logs: *"Note that running a local entrypoint in detached mode
  only keeps the last triggered Modal function alive after the parent process has been killed or
  disconnected."*
* `/run-experiment-on-modal` says it in the repo's own words: *"running e.g. `modal run --detach
  a2a_forward/permutation_test.py::main` as a local entrypoint will also result in premature
  cancellations, so don't do this."*

**Mechanism.** A local entrypoint runs on the CLIENT. `offbook()` blocked in
`run_offbook.map(...)` for the entire run, so the run lived exactly as long as the launching process
did; when that process was reaped, the map's **outstanding** inputs were cancelled and its
**completed** ones survived. `start_new_session=True` in `launch_detached.py` delays this (it
survived several halts) but does not prevent it.

**The signature is diagnostic, and retro-explains both deaths.** `O1`: four arms already complete
and untouched, the two still in flight (`never` c57, `fid` c69) cancelled together in a 3-second
window. `O2`: one arm, cancelled 10 minutes in, ~1 minute after the launching turn ended. The gate
run `g0` survived only because that turn blocked in-process until `done.txt` appeared, so its
entrypoint was never reaped mid-flight — luck, not design.

**The fix**: `offbook.py::offbook --spawn`. The entrypoint calls `run_offbook.spawn(...)` and
**returns in seconds**, so there is no long-lived client to reap and the detached app carries the
work. Verified on the `O2` relaunch: client exited 16 s after launch, app `ap-Hs3lUPfrRPr9sVhEgVDwCc`
live with 1 task and nothing of this session's still running.

**Carry-over caveat, from Modal's own wording**: detach keeps *"the last triggered Modal function"*
alive, so a multi-arm spawn loop from one entrypoint is **not** known to be safe. A single-arm canary
is. For multi-arm rounds, launch one process per arm or move the fan-out server-side; do not trust
one spawn loop with six arms. The non-spawn path is left in place for `--quick` smokes, which are
attached and short by design.


### Round 2 result (`O2`, complete, 120/120 cycles; control = O1's `audit_prop_k`, same code path)

**The trust series moved, then extinguished.** pi's chain-level proposal mass against the
error-graded control at matched cycles:

| window | O2 `prop_priced` (rep 8) | O1 `audit_prop_k` |
|---|---|---|
| c21–34, before any chain cell exists | 0.0000 | 0.0000 |
| c35–44, forced-exposure window open | max 0.0173 / mean 0.0095 | max 0.0018 / mean 0.0009 |
| **c45–60, forcing closed** | **max 0.0295 / mean 0.0126** | max 0.0061 / mean 0.0038 |
| c61–120, late | max 0.0075 / mean **0.0017** | max 0.0062 / mean **0.0017** |

Peak 0.0295 at c45, ~6x the control at the same cycles — then monotone decay to the control's own
level by c61 and indistinguishable from it for the remaining 60 cycles. **`frac_chain` executed in
performance is 0.000 at every probe in both arms**, start to finish. Trust rose and was
extinguished; it never converted into a single executed chain at performance tempo.

**Why, as far as O2 can say.** The pi target-stream composition (addendum): over 100 routed cycles,
**3 chain-containing traversals total ever entered pi's imitation targets** — 0 before the chain
commits, 0.30/cycle during the forced window (chain weight share 0.084), and **0 for all 76 cycles
after forcing closed**. The mean `fb` of the kept set is 3.9975 against a maximum of 4.000: the
stream is, to three decimals, entirely all-segment traversals. **So the x8 weight had almost nothing
to weight.** The rise coincides exactly with the only window in which chain plays reached the
stream, and the decay begins when that window closes.

**What O2 cannot settle, and the instrument now added.** O2 logged only chain plays that had
*already* passed the stage-1 competence band, so "stream empty" cannot by itself separate
band-gating (chains played, band rejected them) from exposure collapse (chains stopped being
played). `pi_cheap_all` / `pi_cheap_e` / `pi_band_e` are added for round 3 — pure readout, no RNG
draw moves, every bit-identity gate still holds.

**The arithmetic points at exposure, not the band** (from the configured knobs, an argument and not
a measurement): `explore_eps = 0.15` adds one uniformly-drawn non-proposed slot to the *audition
set*, and `eps_act = 0.10` then picks uniformly among the ~4 slots in that set. The two epsilons
**multiply**: P(a chain is launched) ~ 0.15 x (8/15) x 0.10 x (1/4) ~ 0.002 per decision, i.e.
~0.05 chain plays per cycle and ~1.4 expected in the stream across the 60 late cycles — consistent
with the observed 0. `eps_act` draws from the audition set, which is itself pi-gated, so
**exploration sits downstream of the very lock-in it was added to break.** That is `native/prop`'s
closed loop reappearing one level out, and it is the design target for round 3, not the dose.

**Treatment safety (both axes, never error alone).** At c120: O2 e_perf 0.0556 / fb 4.00 / t_priced
24443 against the control's 0.0460 / 4.00 / 24443. Late-window (c75–120) means 0.0497 vs 0.0484,
with O2 worse in 9/16 probes — inside the run-to-run scatter. **No feedback events were saved**
(fb/piece identical at 4.00), which is the direct corollary of the stream finding: the currency
could not be paid because the behaviour it prices never occurred. Plant guard flat (0.1309 ->
0.0581 vs the control's 0.0591). Poison starved in both (pi mass 1.4e-4 / 2.3e-4, launch 0.0000).

**The winner's-curse instrument now exists directly.** Realised ÷ predicted at the launched
candidate runs **4.3–7.3x, mean 4.97x over c61–120** at 22.5 candidates/decision — seam-time
audition is optimistic by a factor of five about the unit it picks. O1 could only infer this from
its consequence (corr(argmax width, enumeration-minus-routing gap) = +0.496, worse in 31/34 probes).

**Battery, including the new unconfounded cell.** `base` 0.0605 · `no_prim` 0.1018 · `no_table`
0.0617 · **`no_table_no_plan`: no legal action at all** — the arm has no execution path once both
the address book and the live planner are removed, which is the unconfounded read O1's +0.005
`no_table` could not give (there, deletion silently fell back to the planner, `frac_prim` -> 1.00).
Minability 12–24, no collapse. Port 2 remained out of this arm by design.

Reduced report and figures: `results/O2/` (`fig1`–`fig4`).


## Round 3 — the rehearsal round (`O3`, seed 0, launched 2026-08-26 15:53 UTC)

**The O2 finding this builds on.** Chain trust tracked exposure and only exposure: it rose to 6x the
control exactly while the forced window was open, decayed to the control's level within ~15 cycles
of its closing, and **3 chain-containing traversals in total entered pi's targets across 100 routed
cycles**. The exposure path was a PRODUCT of two epsilons -- `explore_eps` widens the audition set,
`eps_act` draws from that set, and the set is itself pi-gated -- so the correction sat *downstream*
of the lock-in it was added to break (~0.002 chain launches per decision).

**The treatment.** `prop_rehearse` = O2's `prop_priced` (credit unchanged, `pi_fb_rep = 8`) with the
launch-level draw **moved upstream of pi's gate**: a REHEARSAL draw uniform over every legal slot,
launched regardless of what pi proposes, taken before candidate assembly so the drawn slot is
actually materialised and priced. `census/`'s queued "targeted rehearsal of a received vocabulary",
on a plant.

**Rate: moved, not added.** `p_rehearse = 0.10` with `eps_act = 0.0`, so the exploration *budget* is
identical to O2's 10% of practice decisions and **only its placement changes** -- a true
one-variable step rather than "more exploration". `explore_eps = 0.15` is untouched (it feeds pi's
proposal machinery, not the launch). Practice only; performance probes stay pure exploit policy.

**Expected counts, from the arithmetic** (stated in advance, per the spiral convention): at seam 0
the legal set is ~17 slots (8 segment + 8 chain + prim), so a rehearsal draw lands on a chain slot
with p ~ 8/17 ~ 0.47, giving P(chain launched) ~ 0.10 x 0.47 ~ **0.047 per seam-0 decision** against
O2's ~0.002 -- roughly **50x**. At 24 practice traversals per cycle that is ~1.1 chain plays/cycle at
seam 0 alone, and over the commit-to-horizon window (chain cells commit c35-41; horizon c120,
~80 cycles) **~90-180 chain plays**, against O2's 3 for the whole run. Comfortably past the "order
tens" target while staying a minority of decisions.

**What the one arm settles.** (a) Whether chain trust HOLDS under sustained exposure at full credit
-- the direct converse of O2's rise-and-decay. (b) Band-gating vs body-refusal, via the pre-band
instruments added after O2 (`pi_cheap_all` = chain plays before the competence band, `pi_cheap_e` =
what they scored, `pi_band_e` = where the band sat). (c) The first real sample of **chains' realised
e_piece and fb on the body** -- O1 and O2 produced 3 plays between them, so "do chains pay in any
currency at realised quality" has never actually been measurable. `reh_n` logs the draws;
`aud_optimism` continues.

**Gate.** `prop_priced` under the round-3 runner reproduces the O2 code path at matched flags:
max\|Δ\| = **0.000e+00** over 122 values -- the rehearsal path is a strict no-op at
`p_rehearse = 0`, so O1 and O2 stay reproducible and the cross-tag controls (O1 `audit_prop_k`,
O2 `prop_priced`) remain exact. One smoke was discarded first for changing two knobs at once
(`--eps-act 0.0` against O2's default); recorded because the failed gate was the thing that caught it.

**Staging.** This one arm only. If trust holds, the decisive follow-up is rehearsal WITHOUT the
priced credit (`pi_fb_rep = 1`) to separate exposure from currency -- that decision goes back to the
orchestrator and was not launched here. If trust still extinguishes despite rehearsal, credit and
stream entry, that is body-refusal and no second arm is needed.


### Round 3 result (`O3`, complete, 120/120; controls = O2 `prop_priced`, O1 `audit_prop_k`)

**(a) Chain trust HOLDS, and it is the converse of O2's rise-and-decay.** pi's chain mass by window
(mean, three arms at matched cycles):

| window | O3 `prop_rehearse` | O2 `prop_priced` | O1 `audit_prop_k` |
|---|---|---|---|
| c35–44, chain cells commit | 0.0093 | 0.0095 | 0.0009 |
| c45–60 | 0.0193 | 0.0126 | 0.0038 |
| c61–90 | **0.0217** | 0.0024 | 0.0019 |
| c91–120 | **0.0230** | 0.0009 | 0.0014 |

O2 peaked at 0.0295 and decayed to the control's own level within ~15 cycles of forcing closing. O3
does not decay: it **rises monotonically across windows** and ends at **25x O2's late mass**, with
`argmax_chain` nonzero in **23 of 40 probes**. Moving exploration upstream of pi's gate is what the
trust series was tracking all along -- O2's "currency" effect was exposure in disguise.

**(b) The adjudication, and it is BODY-REFUSAL — not band-gating.** The pre-band instruments give a
clean verdict:

* **203 chain plays on the body** across 100 routed cycles (O2's post-band total was 3; O2 never
  logged the pre-band count, which is exactly why it could not settle this).
* Of those, **22 passed the competence band = 10.8%**.
* Mean realised `e_piece` of a chain play: **0.2889**, against a band (batch median) of **0.1044** --
  chains execute **2.77x worse** than the median traversal, stably across every window
  (c45–60 2.42x, c61–90 3.13x, c91–120 2.85x).

So the band is not mis-specified and it is not gating chains out on a technicality: **chains are
genuinely played badly on this plant, and the band is correctly reporting it.** The round-4 target
the alternative branch would have named (band-free or action-relative banding) is **not** indicated.

**(c) Chains' realised quality at scale — the number O1 and O2 could never produce** (3 plays
between them). At 203 plays: chain `e_piece` 0.2889 against the arm's own performance error of
0.0745 and `never`'s 0.0085. For reference this sits near `key_frozen`'s 0.42, the other arm that
committed chains at scale. **Chains do not pay in either currency at realised quality**: they are
~4x worse in meters, and the feedback saving is never collected because they are never selected in
performance.

**(d) Zero chains executed in performance, in all three arms, at every probe.** `frac_chain` = 0.000
throughout (O3 c75–120: 0/16 probes). Trust of 0.023 is real proposal mass and still an order of
magnitude below what would win an argmax.

**(e) The arithmetic held, and slightly over-delivered.** Predicted ~1.1 chain plays/cycle at seam 0;
realised **2.03/cycle** (seam 1 contributes, as expected), 7.21 rehearsal draws per practice
traversal. 203 plays against a predicted 90–180 -- the exposure treatment did what it was designed
to do, so (a) and (b) rest on a mechanism that verifiably fired.

**(f) Audition optimism** 5.52x at 22.0 cand/dec (O2: 4.97x at 22.5) -- unchanged, as expected: the
scorer is the same.

**(g) Safety and the poison curve.** Plant guard flat (0.1309 -> 0.0588 vs O2 0.0581, O1 0.0591).
Both axes at c120: e_perf 0.0745 / fb 4.00 against O2's 0.0556 / 4.00 and O1's 0.0460 / 4.00; late
window 0.0641 vs 0.0497 / 0.0484 -- **rehearsal costs ~30% on error and saves nothing**, which is
the direct consequence of (b) and (d): 10% of practice decisions are spent playing units that
execute 2.8x worse, and none of them is ever selected at performance tempo.

**The poison twin under uniform rehearsal — the quarantine-under-forced-exposure readout.** A
uniform draw over legal slots forces poison plays too, and the series is a repeated bump-and-decay:
mean **0.00234** over c45–120 with a max of **0.00608**, against O2's 0.00042 and O1's 0.00061
(~5x elevated), with visible peaks at c36 (0.0062), c63 (0.0033), c84 (0.0055), c96 (0.0037), each
decaying within a few probes. **Launch fraction in performance stays 0.0000 throughout.** So forced
exposure repeatedly re-funds a bad address and self-imitation repeatedly starves it back down --
`native/` finding 6's quarantine holding under an adversarially generous exposure regime rather than
merely under an enumeration it can ignore.

**Battery.** `base` 0.0794 · `no_prim` 0.1005 · `no_table` 0.0565 · **`no_table_no_plan`: no legal
action at all** (reproducing O2) · `restored` 0.0811. Minability 16–18, no collapse.

**Standing.** The line's question -- does the chain level consolidate -- now has a mechanism-level
answer on this substrate: trust is exposure-limited and *can* be made to hold, but the units
themselves execute ~2.8x worse than segment play, so nothing downstream will select them. Deferred
and unaffected: Port 2 (never cleared parity in O1, untested rather than refuted).


## Round 4 — making depth bind (`organist`; Phase A gate `d0`, launched 2026-08-26 20:32 UTC)

**The standing fact this attacks.** Across O1/O2/O3 the reactive and live-replan strategies won on
error every round, and a chain was never selected at performance tempo once (`frac_chain` = 0.000 at
every probe of every arm). Depth exists GEOMETRICALLY on this piece — a chain is 60 control steps
against a ~21-step composition horizon — but it has been cheap to flatten, because re-grounding
costs exactly one feedback event and nothing else. O3 settled that this is not a teacher problem:
rehearsal at full credit put **203 chain plays on the body**, they executed **2.77x worse** than the
median traversal, and the competence band correctly refused them. If chunks are to pay here, the
WORLD has to make feedback expensive.

**The intervention.** `obs_delay` — an observation delay on the reflex loop. Every AGENT-SIDE
feedback consumer sees the state from Δ control steps ago (the reactive controller's per-step
observation, CEM's initial state at a seam replan, the library key at a launch, pi's own read), and
motor noise makes the stale estimate diverge from the truth. Biologically honest: reflex delay
against movement time is why ballistic chunks exist in animals. `dt_ctrl = 0.02 s`, so Δ = 2/4/8/16
is **40/80/160/320 ms** against a 400 ms drilled segment and a 1.2 s phrase — human proprioceptive
loops run ~100 ms and visual ~200–250 ms, so the middle of the sweep is where a person lives.

**One variable, by construction.** Experimenter-side instruments stay TRUE-state: `e_seg`,
`e_piece`, `tips`, `launches`, the plant guard, the competence band, the metering ledger — and the
forward model's TRAINING TRANSITIONS. The delay is a decision/actuation constraint, not a
learning-data treatment. It enters through a single `obs()` read in each traversal.

**The alternative considered and set aside**: a hard cap on feedback events per traversal. Rejected
because adoption under necessity-by-fiat weakens the very claim the arc is chasing — a chunk adopted
because nothing else is legal says nothing about whether it was worth adopting. Recorded rather than
silently not done.

**Gate.** `prop_rehearse` at `obs_delay = 0` reproduces the O3 code path: max\|Δ\| = **0.000e+00**
over 155 values. The first attempt **FAILED at 6.1e+04** and caught a real bug: one of the four
`hist.append` insertions silently no-op'd because the donor line carries an inline comment my
match pattern lacked, so during the approach leg `obs()` returned the RESET state and the first
reactive replan of every traversal planned from the wrong posture. `str.replace` fails silently;
the identity gate is what caught it. Recorded because the near-miss is the point of having the gate.

### The neutral criterion, pre-fixed before the run

    Δ* = the SMALLEST Δ at which  e_chain <= e_live_seg <= e_reactive  on mean piece error,
         subject to  min(e_chain, e_live_seg, e_reactive) <= ref_stale.

`ref_stale` = ballistic-per-segment piece error under the STALE (pre-practice) forward model at
Δ = 0 — this node's own arm-neutral usable-range anchor, the same quantity every offbook run reports
in `setup`. **If no Δ satisfies both, the round STOPS at the gate** and that is a substrate finding;
Δ is not to be extended or re-tuned until the ordering inverts (legato F2).

**The guard was revised once, before the real run, on smoke evidence, blind to the trust question.**
The first draft anchored playability at `2x the best of the three at Δ = 0`. The smoke showed that is
DEGENERATE: at Δ = 0 the incumbent is reactive MPC, ~0.009 on this piece in a real run, so the
ceiling lands at ~0.017 — below anything a chunk has ever scored here (O3's realised chain plays:
0.289). A guard anchored to the undelayed champion cannot fire once feedback is taxed at all, so it
would have vetoed every Δ by construction and discriminated nothing. `ref_stale` is task-anchored
and arm-neutral, which is what F2 actually prescribes. Both ceilings are reported in the output so
the original can be read off the record.

**The mechanism prediction** (stated in advance): chain error should be roughly FLAT in Δ — a
committed chain is open-loop, so a stale observation costs it only its single launch key — while
reactive degrades steeply and live-per-segment intermediately. The smoke already shows that shape
(reactive 0.0585 -> 0.529 across Δ = 0..16, **9x**; live_seg 0.129 -> 0.395, **3x**; chain
0.249 -> 0.236, **flat**), and in the smoke the ordering does invert by Δ = 16 but fails playability
— i.e. inversion by universal collapse, which is exactly what the guard exists to refuse. Whether
the real FM and library put a crossover *inside* the playable region is the gate's whole question.

**Phase B, if and only if the gate licenses one** (orchestrator's call, not launched here): one arm,
`prop_rehearse` at Δ*, with **`pi_fb_rep = 1`** — plain error-graded credit. Deliberate: if depth
binds, the O1 mechanism that refused chains at Δ = 0 should fund them with no currency machinery at
all, and the attribution is then to the world rather than to the teacher.


### Round 4 Phase A result (`d0`, complete) — THE GATE FAILS: Δ* = None, the round stops here

Warm 20 cycles, library K = 72/cell, `n_slot` 8. Playability anchor **`ref_stale` = 0.1460**
(ballistic-per-segment under the stale FM, undelayed). Original draft ceiling would have been
**0.0228** — reported beside it, and the real run confirms the revision was necessary: that ceiling
would have vetoed even Δ = 2, where the piece is still comfortably played.

**Mean piece error by strategy and delay** (fb/piece: reactive 61, live_seg 4, seg_tape 4, chain 2):

| Δ | ms | reactive | live_seg | seg_tape | chain |
|---|---|---|---|---|---|
| 0 | 0 | **0.0114** | 0.1978 | 0.2682 | 0.2703 |
| 2 | 40 | **0.1110** | 0.2201 | 0.2036 | 0.2945 |
| 4 | 80 | 0.2231 | 0.2447 | **0.2081** | 0.3860 |
| 8 | 160 | 0.5294 | 0.3340 | **0.2358** | 0.4110 |
| 16 | 320 | 0.7190 | 0.5311 | **0.3437** | 0.4569 |

**The mechanism prediction is confirmed, and sharply.** Degradation factor from Δ = 0 to Δ = 16:
**reactive 63.1x · live_seg 2.69x · chain 1.69x · seg_tape 1.28x.** Delay sensitivity is exactly
ordered by how much feedback a strategy consumes — 61 fb/piece degrades 63x, 4 fb/piece degrades
~1.3–2.7x, and the stored strategies barely move because they are open-loop and a stale observation
costs them only their launch key. Reactive's collapse is worst on the patch segment (by-segment
median 1.004 at Δ = 16). The intervention does exactly what it was designed to do.

**And the gate still fails, because every crossover lands outside playability.**

| Δ | ordered (chain ≤ live ≤ react) | playable (min ≤ 0.1460) | passes |
|---|---|---|---|
| 0 | no | yes | no |
| 2 | no | yes | no |
| 4 | no | no | no |
| 8 | no | no | no |
| 16 | **yes** | no | no |

**Δ\* = None.** The ordering does inline at Δ = 16, exactly as the smoke foreshadowed — by
universal collapse, which is what the guard exists to refuse.

**The negative is robust to every weaker reading of "the ordering inverts."** The playable region is
Δ ∈ {0, 2} and **reactive wins both** (0.0114, 0.1110). First crossover delays: `chain < reactive`
at Δ = 8 (not playable); `seg_tape < reactive` at Δ = 4 (not playable); `any stored < both live` at
Δ = 4 (not playable). The single crossover that *does* occur inside playability is
`seg_tape < live_seg` at Δ = 2 — frozen measured segment content overtaking a live segment plan —
but reactive MPC still beats both there, so nothing about chunk adoption follows.

**The wrinkle the pre-fixed criterion did not ask about, reported rather than re-fitted.** From
Δ = 4 onward the best strategy at every delay is **`seg_tape` — per-segment frozen measured
content — not the chain.** The chain is not even the most delay-robust option (1.69x vs seg_tape's
1.28x): its single launch key is stale *and* seam-time audition is badly calibrated at chain span
(gate G-C: ρ = 0.337), so delay compounds a scoring problem it already had. The criterion asked
about the chain level specifically and the answer there is no; the data additionally say that if
anything is going to pay under delay on this plant it is the *segment* tape, one level down. That
is a hypothesis for a future round, not a result of this one, and the round stops at the gate as
pre-committed — Δ is not extended and the criterion is not re-fitted.

**Standing conclusion for the line.** Depth on this piece is geometric but not economic. Making the
reflex loop expensive degrades feedback-hungry control exactly as predicted (63x), and still does
not create a niche for the 60-step chunk inside the region where the piece can be played at all.
Combined with O3 (chains execute 2.77x worse than the median traversal at 203 plays) the picture is
consistent: on this plant the chain level is refused on its own merits, and no amount of taxing its
competitors rescues it while the task remains performable.
