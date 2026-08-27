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
| `delay_gate.py` | **Round 4 (`organist`/`d0`)** — the observation-delay gate: Δ × {reactive, live_seg, seg_tape, chain} under the pre-fixed criterion and the `ref_stale` playability guard. **Round 5 (`d1`)** — the content ladder bolted on behind `--arms`: harvest pools (`fresh`/`perf`/`raw`/F5-routed), `build_uniform` (the donor's draw, refactored into a function), `build_auditioned` (legato's `compile_unit` ported onto this node's `Cell`/`RoutePolicy`), and the per-arm sweep. Every round-5 flag defaults to the round-4 behaviour and the ladder's own `d0` arm asserts bit-identity against the round-4 sweep in-run. |
| `analyze_gates.py` | The gate report and the calibration record the main run's knobs are read from. |
| `analyze_delay.py` | Reduces a delay-gate run (`d0`) and its content ladder (`d1`): both ceilings, the sanity block (in-run identity + the library-independence of `reactive`/`live_seg` across arms), strategy × Δ per arm, the pre-fixed verdict per arm, per-segment medians, and the ladder read against `ref_stale` and legato's own frozen-unit numbers. `--figs` draws locally into `results/<tag>/`. |
| `analyze_offbook.py` | Reduces a main run (headline, routing mix, trust formation, can't-decompose, parity, plant guard, consumption-phase ledger, battery, poison) + `::figs`, a **remote** figure job. |
| `launch_detached.py` | Session-isolated detached launcher (`--fn gates\|offbook`). |

## Modal volume layout

```
/data/practice_offbook/<tag>/gates/results.json     # Phase A
/data/practice_offbook/<tag>/<arm>/results.json     # Phase B (one dir per arm)
/data/practice_offbook/<tag>/figs/*.png             # figures, written remotely
/data/practice_offbook/<tag>/delay_gate/results.json  # rounds 4 & 5 (one file; arms nested inside)
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


## Round 5 — the content ladder (`d1`, seed 0, launched 2026-08-26 22:40 UTC)

**What this attacks, and what it does NOT.** Round 4 returned Δ\* = None with every stored strategy
sitting **above the playability anchor already at Δ = 0** (`seg_tape` 0.268, `chain` 0.270 against
`ref_stale` 0.146). The round read that as an environment property. But on the *same* piece and the
*same* plant, [`../legato/`](../legato/README.md)'s plant-auditioned, launch-keyed frozen units run
**0.1026** (phrase) and **0.1065** (segment) — *under* d0's own playability bar before any delay is
applied. So d0's stored content is ≈2.5× worse than content this substrate is known to produce, and
the gate's question was asked of a library that was never built the way legato builds one. This
round re-runs the **identical gate** — same piece, same plant, same FM, same eval geometries, same
Δ sweep, same four strategies, same pre-fixed criterion, same `ref_stale` anchor — and varies **only
how the library is built and read**. It does not re-tune Δ, does not change the criterion, and does
not touch the round-4 numbers.

**The four setup choices, turned into rungs.** (`delay_gate.py` module docstring carries the same
list with line references.)

| # | the choice d0 made | what legato does | rung(s) |
|---|---|---|---|
| 1 | tapes are the post-noise issued commands (`acts`) of 20 **reactive** warm-up traversals at `sigma_practice = 0.15` (2.5× the performance noise), recorded *while the FM was still training* — replayed open-loop such a tape carries one noise realisation **plus** the closed-loop corrections made for it; and the chain cells come from that same reactive pool, which `world.Library`'s own docstring names as the configuration legato F1 measured at 1.03× and F5 corrected to 1.34× | harvest at performance conditions, and cut the chain pool from **segment-committed** play (F5: the lower level's library funds the upper level's addressable variation) | `fresh` (post-training pool, same σ — the vintage control) · `perf` (σ_perf, **matched noise seed**) · `raw` (**the same traversals'** pre-noise commands) · `f5` (chain cells from keyed segment play) |
| 2 | `lib_k = 72` tapes/cell drawn **uniformly**, never cross-validated (justified by étude E-3b's winner's curse) | pay for a **plant audition**: `n_cand` candidates replayed open-loop on `n_score` recorded launch states, `select_library` commits one argmin per cell of the launch distribution — not the winner's curse, because it scores on states the tape never saw | `sel` (audition) · `small` (uniform, **size-matched** to `sel`, so the audition is not credited with the effect of merely holding fewer entries) |
| 3 | seam selection is **FM audition**, which Phase A G-C measured as calibrated at segment span (ρ = 0.924) and blind at chain span (ρ = 0.337) | — | `key_d0`, `f5_key`, `all4` |
| 4 | launches are **unkeyed** — played from whatever posture the body reached | **key**: nearest launch-state centroid picks the rendition (`RoutePolicy`'s existing `key` mode, unmodified) | as above |

Defects 3 and 4 are carried by the same knob here (the seam selector), and the ladder separates
*keying uncurated content* (`key_d0`, `f5_key`) from *keying curated content* (`all4`).

**The arms** (10; `--arms "d0,fresh,perf,raw,f5,small,sel,all4,f5_key,key_d0"`):

| arm | library harvest | selection | seam | isolates |
|---|---|---|---|---|
| `d0` | round 4's warm pool | uniform 72 | `audit` | **the in-run control** |
| `fresh` | post-training reactive pool @ σ_practice | uniform 72 | `audit` | pool vintage / FM staleness |
| `perf` | post-training reactive pool @ σ_perf | uniform 72 | `audit` | + harvest noise level |
| `raw` | **the same** σ_perf traversals, pre-noise commands | uniform 72 | `audit` | + the baked-in noise realisation |
| `f5` | ns=1 from `raw`; ns>1 from segment-committed play | uniform 72 | `audit` | + chain addressability (legato F5) |
| `small` | as `f5` | uniform `n_pick` | `audit` | size control for `sel` |
| `sel` | as `f5` | **plant audition**, `n_pick`/cell | `audit` | + cross-state selection |
| `all4` | as `f5` | plant audition | **`key`** | **all four fixed** |
| `f5_key` | as `f5` | uniform 72 | `key` | the key on uncurated F5 content |
| `key_d0` | round 4's warm pool | uniform 72 | `key` | the key alone, on d0's own content |

**Knobs**: `n_harvest = 20` (× `batch` 16 = 320 rows/cell, matching d0's pool size exactly),
`n_cand = 64`, `n_score = 96`, `n_pick = 4` — legato's own `n_cand`/`n_score`/`n_lib`. Score
geometries are held out (`geom(n_score, seed+5100)`, distinct from the eval set at `seed+5200`).

**Three controls that make the ladder readable.**

1. **`reactive` and `live_seg` are library-independent by construction** — reactive never reads the
   library, and `live_seg` restricts the action set to the live plan (`pol.levels = set()`), so
   `Library.populated` returns nothing. Every arm is therefore scored against the *same* two, which
   is asserted post hoc at 0.000e+00 across all arms × Δ in `analyze_delay.py`'s sanity block. Only
   the two stored strategies are re-run per arm.
2. **`perf` vs `raw` is a perfectly matched pair**: the identical traversals, the identical launch
   states, the identical realised errors, read two ways (`acts` vs `acts_raw`). The only thing that
   differs is whether the stored commands contain the noise realisation they were recorded under.
3. **Additive flags, donor defaults, in-run identity gate.** `--arms ""` skips the whole ladder;
   the `--quick` default path reproduces `results/dsmoke/delay_gate.json`'s sweep at **0.000e+00
   over 72 values** (verified before launch as `d1s0`, and note this also re-verifies that round
   4's own late `ref_stale` insertion was RNG-neutral); and the ladder's `d0` arm re-runs the
   stored strategies through the ladder's code and asserts bit-identity against the round-4 sweep
   **in-run**, raising if it ever differs (`d1s1` smoke: 0.000e+00).

**Pricing.** The plant audition is charged to the agent exactly as legato charges it
(`World.audition(..., who="agent")`), and so are the score-set traversals (the body has to play up
to a seam to find out where it arrives) and every harvest traversal. Per-library `build_cost` and
per-pool `harvest_cost` deltas are recorded on the ledger and printed by the analyzer.

**One thing that had to be added, and why (found by the smoke, not by inspection).** `Cell._assign`
hands slot assignment to k-means over *content*, and k-means++ cannot run when the drawn tapes hold
fewer **distinct** sequences than there are slots — the seeding step samples from an all-zero
distance distribution and numpy raises `probabilities do not sum to 1`. Round 4's pool is reactive
play, where every rendition is unique, so the degeneracy was unreachable. Round 5's `f5`/`small`
pools are cut from **keyed** play, where many performers replay the same stored rendition bit for
bit, so it *is* reachable there — and the ladder smoke hit it immediately. `degenerate_slots()` in
`delay_gate.py` returns an explicit assignment in exactly that case (identical tapes share a slot)
and `None` — the donor's k-means path, untouched — otherwise, so nothing that previously worked
changes. The distinct-tape count it computes is logged per cell either way, because it *is* legato
F5's quantity: how much addressable variation the lower level manufactures. The fix lives in
`delay_gate.py` and not in `world.py`, so gate G-F's bit-for-bit fork fidelity is unaffected.

**Reproduce.**

```bash
cd experiments/            # MODAL_PROFILE=chromatic
# smokes (both, always, before the detached launch)
modal run mjc/practice/offbook/delay_gate.py::delay_gate --quick --tag d1s0
modal run mjc/practice/offbook/delay_gate.py::delay_gate --quick --tag d1s1 \
    --arms "d0,fresh,perf,raw,f5,small,sel,all4,f5_key,key_d0"
# the run (--spawn only; see the launch-path Gotcha above)
modal run --detach mjc/practice/offbook/delay_gate.py::delay_gate --spawn --tag d1 --seed 0 \
    --arms "d0,fresh,perf,raw,f5,small,sel,all4,f5_key,key_d0"
python3 mjc/practice/offbook/analyze_delay.py --tag d1 --fetch --figs
```

### Round 5 result (`d1`, complete, 10 arms, ~26 min) — Δ\* = None on **every** arm

**Both sanity gates passed.** The ladder's `d0` arm is bit-identical to the round-4 sweep
(**0.000e+00**, in-run assertion), and `reactive`/`live_seg` are identical across all 10 arms
(**0.000e+00 over 100 values**) — they are library-independent by construction, so every arm is
scored against the same two. `raw` and `f5` share their ns=1 cells by construction (all pools hold
320 rows, so the shared `lrng` stream lands on the same draw) and their `seg_tape` rows agree at
0.000e+00 across all five Δ, which makes `raw → f5` a clean single-variable contrast **on the
chain**.

**Ceilings, unchanged and reported per arm.** `ref_stale` = **0.1460**; the original draft ceiling
(2× the best at Δ=0) = **0.0228**. legato on this piece: `phrase_frozen` 0.1026, `seg_frozen`
0.1065, `seg_plan_launch` 0.0850, `never` 0.0107.

**Mean piece error at Δ = 0, best stored per arm** (the quantity the round was built to move —
round-4 baseline 0.2682, target 0.1460):

| arm | chain | seg_tape | best stored | gap closed to `ref_stale` | best stored, any Δ |
|---|---|---|---|---|---|
| `d0` | 0.2703 | 0.2682 | 0.2682 | 0.0% | 0.2036 |
| `fresh` | 0.3521 | **0.1831** | **0.1831** | **69.6%** | **0.1805** |
| `perf` | 0.2645 | 0.2606 | 0.2606 | 6.2% | 0.2169 |
| `raw` | 0.3030 | 0.2223 | 0.2223 | 37.5% | 0.2223 |
| `f5` | 0.2936 | 0.2223 | 0.2223 | 37.5% | 0.2223 |
| `small` | 0.2610 | 0.2480 | 0.2480 | 16.5% | 0.2452 |
| `sel` | 0.3557 | 0.2828 | 0.2828 | −11.9% | 0.2119 |
| `all4` | **0.2360** | 0.2306 | 0.2306 | 30.7% | 0.1998 |
| `f5_key` | 0.3287 | 0.2979 | 0.2979 | −24.3% | 0.2543 |
| `key_d0` | 0.2669 | 0.3265 | 0.2669 | 1.1% | 0.2373 |

**Δ\* = None on every one of the ten arms**, under the pre-fixed criterion and either ceiling. The
playable region is Δ ∈ {0, 2} in every arm (it is set by `reactive`, which is shared), reactive wins
both in every arm, and no arm's stored content ever reaches `ref_stale` = 0.146 at any Δ — the best
number anywhere in the round is `fresh`'s `seg_tape` 0.1805 at Δ = 2, still 1.24× above the anchor
and 1.69× above legato's `seg_frozen`. Ordering rows: `chain ≤ live_seg ≤ reactive` first holds at
Δ = 16 for six arms and at Δ = 8 for `raw`, `f5`, `all4`, `key_d0` — all outside playability, i.e.
by universal collapse, which is what the guard exists to refuse.

**The single-variable contrasts at Δ = 0** (each row changes exactly one thing):

| step | what changes | seg_tape | chain |
|---|---|---|---|
| `d0` → `fresh` | pool vintage: harvested after FM training instead of during | 0.2682 → **0.1831** | 0.2703 → 0.3521 |
| `fresh` → `perf` | σ_practice → σ_perf, matched noise seed | 0.1831 → 0.2606 | 0.3521 → 0.2645 |
| `perf` → `raw` | the SAME traversals, post-noise → pre-noise commands | 0.2606 → 0.2223 | 0.2645 → 0.3030 |
| `raw` → `f5` | chain pool: reactive → segment-committed play (F5) | 0.2223 → 0.2223 (identical by construction) | 0.3030 → 0.2936 |
| `f5` → `small` | library size 72 → 4, both uniform | 0.2223 → 0.2480 | 0.2936 → 0.2610 |
| `small` → `sel` | uniform → **plant audition**, size matched | 0.2480 → 0.2828 | 0.2610 → 0.3557 |
| `sel` → `all4` | FM-audit read → **keyed** read, same library | 0.2828 → **0.2306** | 0.3557 → **0.2360** |
| `f5` → `f5_key` | FM-audit → keyed read, same library | 0.2223 → 0.2979 | 0.2936 → 0.3287 |
| `d0` → `key_d0` | FM-audit → keyed read, same library | 0.2682 → 0.3265 | 0.2703 → 0.2669 |

Two things in that table are worth stating flatly because they were not what the round was set up
to expect. **(i)** The largest single move on `seg_tape` comes from the *vintage control* — the pool
harvested after the forward model finished training — not from any of the four named defects, and
it costs the chain (0.2703 → 0.3521). **(ii)** The two named fixes that were expected to carry the
effect *individually make things worse in the read they were paired with*: the plant audition at
matched size (`small` → `sel`) is worse under an FM-audit read on both levels, and keying alone
(`d0` → `key_d0`, `f5` → `f5_key`) is worse on `seg_tape` in both cases. They only pay **together**:
`sel` → `all4` (identical library, keyed instead of auditioned at the seam) is the round's largest
single improvement, 1.23× on `seg_tape` and **1.51× on the chain**.

**The audition is well calibrated when its output is read the way it was selected, and not
otherwise.** The plant audition's own held-out predictions (noiseless open-loop replay on recorded
launch states, `n_cand` 64 × `n_score` 96, 4 committed per cell):

| cell | chosen | best_fixed | per-state oracle | distinct picks |
|---|---|---|---|---|
| 1:0 | 0.0414 | 0.0465 | 0.0131 (3.54×) | 4/4 |
| 1:1 | 0.2171 | 0.2234 | 0.1995 (1.12×) | 2/4 |
| 1:2 | 0.3023 | 0.3408 | 0.2280 (1.49×) | 3/4 |
| 3:0 (chain) | 0.2199 | 0.2499 | 0.1273 (1.96×) | 2/4 |
| 2:1 | 0.4625 | 0.4625 | 0.4429 (1.04×) | 1/4 |

Realised against predicted, same library: the **keyed** read (`all4`) runs 1.23× on `seg_tape` and
**1.07×** on the chain — legato-like calibration; the **FM-audit** read (`sel`) runs 1.51× and
**1.62×**. This is Phase A G-C (ρ = 0.337 at chain span) showing up at consequence level: reading a
state-conditioned library with a rollout score breaks the state↔rendition pairing the audition
validated, and the chain level is where it breaks worst.

**legato F5's mechanism fires but buys nothing here.** Keyed segment play does manufacture
addressable variation — the chain pool cut from it holds **18 distinct sequences in 72 draws** at
cell 3:0 and 16/72 at cell 2:1, against 72/72 for round 4's reactive pool — and the resulting
contrast (`raw` → `f5`) is 0.3030 → 0.2936 on the chain, ~3%, with the closing-leg median unmoved
(0.6430 → 0.6427).

**Per-segment medians: the closing leg is where everything lives, and content does not move it.**
Round 4's chain profile at Δ=0 was [0.0451, 0.1252, 0.3130]. Across the ten arms the chain's seg-2
median at Δ=0 spans 0.2769 (`key_d0`) to 1.0633 (`sel`), while seg-0 never leaves 0.039–0.109. The
best-behaved closing leg in the round belongs to `all4` (chain [0.0713, 0.2323, 0.3007], and it is
the only arm whose closing leg stays under 0.37 at every Δ: 0.3007 / 0.3200 / 0.3632 / 0.2808 /
0.3410). `sel` under the FM-audit read is the worst (1.0633, flat across Δ = 0–8 because the
audition selects the same chain at every delay and an open-loop chain's trajectory is
delay-independent once the tape is fixed).

**Ledger — what the fix cost.** Total agent priced time 30,865 s. The plant audition plus its
score-set traversals cost **22,986 s (74% of the run), 992,832 environment steps and 31,296 feedback
events** for one library, against 2,426 s for round 4's entire 20-cycle warm-up. The `fresh` and
`perf` harvests cost 2,426 s each, the keyed route-pool harvest 602 s (planner-free). Uniform
library builds are free by construction (they only read pools already paid for). So the
best-scoring content in the round (`fresh`) is also the cheapest to build, and the most expensive
arm (`sel`, 74% of the ledger) is the second-worst on `seg_tape`.

**Caveats.** Single seed; the eval is 24 shared geometries; the spread across all ten content arms
at Δ = 0 is 0.183–0.327 (≈1.8×) and there is no in-run replicate to size the uniform-draw sampling
noise against, so the small steps (≤0.03) in the contrast table should be read as ranks at best.
Mean and median disagree in sign for some arms (`d0` `seg_tape` 0.2682/0.1912 vs `sel`
0.2828/0.3421), and the criterion is stated on the mean, as in round 4. The auditioned library is
committed at `n_pick` = 4 (legato's `n_lib`), so `sel`/`all4` hold 4 entries per cell against 72 for
the uniform arms — `small` is the size control for that, and it sits between them. `sel`'s ns=1
cells are audition-selected while the chain pool it draws from was generated by *uniform* ns=1 keyed
play, which keeps the selection axis a single variable but is one step short of legato's strict
left-to-right nesting for the chain level.

**Artifacts.** `results/d1/d1_report.txt` (full tables), `results/d1/d1_delay_by_arm.png` (error vs
Δ, one panel per arm, log scale, with `ref_stale` and legato's `seg_frozen` drawn),
`results/d1/d1_ladder_delay0.png` (the rung bar chart against all three anchors),
`results/d1/d1_by_segment.png` (per-segment medians at Δ = 0, all arms).
Raw: `results/d1/d1/delay_gate/results.json`. Smokes: `results/d1s0` (donor identity),
`results/d1s1` (ladder path).


## Round 6 — legato's nesting on the gate (`d2`, seed 0, launched 2026-08-26 23:22 UTC)

**The diagnosis this implements** (orchestrator's, from `legato/results/l1/l1/*/results.json`'s
`events`, and now reproduced automatically by `analyze_delay.py`'s content-fidelity block against
round 5's own record):

| cell | legato `L1` chosen / per-state oracle | `d1` `sel` chosen / oracle | ratio on the oracle |
|---|---|---|---|
| 1:0 | 0.0426 / 0.0137 (c25) | 0.0414 / 0.0131 | **0.96×** (match) |
| 1:1 | 0.1271 / 0.0481 (c32) | 0.2171 / 0.1995 | **4.14×** |
| 1:2 | 0.1993 / 0.0554 (c39) | 0.3023 / 0.2280 | **4.12×** |
| 3:0 | 0.1005 / 0.0586 (c55) | 0.2199 / 0.1273 | **2.17×** |

`per_state_oracle` is the best **any** candidate achieves per launch state, so a 4× gap there is a
property of the **pool**, before selection: round 5's candidate pools contained no tape that works
from the committed configuration's launch states. legato's own events say why — at its seam-1/2
commits `n_reactive_traces` = **2 of 6** traces, i.e. 4 of 6 were practice recorded *in the
already-committed configuration* (commits c25/32/39, `trace_window` 6, `batch` 24 → `n_pool` 144,
`interleave_period` 3). A candidate for seam k has to start from the frozen predecessor's arrival
distribution — the same distribution the score set is drawn from. `d1` harvested every segment pool
from purely reactive traversals, so candidate and score-set launch distributions were mismatched at
every seam past the first. (Round 5's own `fresh` result — the *vintage* control being the largest
single move on `seg_tape` — points the same way: a tape is only as good as the configuration it was
recorded in.)

**`build_nested()`** ports legato's sequential assembly whole:

```
stage k = 0..NS-1 :  practice `n_nest` cycles in the CURRENT configuration at sigma_practice
                     (committed seams played by their keyed `lib` unit, uncommitted seams reactive,
                      one cycle in `nest_interleave` fully reactive — legato's diet-rent interleave),
                     keeping a `nest_window`-deep trace buffer;
                     score at seam k from that same configuration, on held-out geometries;
                     commit cell (1, k) by plant audition + `select_library`.
stage NS          :  practice `n_nest` more cycles in the FULLY segment-committed configuration and
                     commit the chain cells from THOSE traces — legato F5 done properly, where
                     round 5's `f5` only approximated it with uniform keyed play.
```

Practice and score-set traversals go through `World.traverse` with legato's **own** `lib` unit
(`World.select_library`'s return value is exactly what `unit_commands`' `"lib"` branch consumes), so
the configuration the pool is recorded in is legato's machinery bit for bit; the same renditions are
also written into this node's `Cell`s so the gate can read them through `RoutePolicy` in either
mode. Knobs are legato `L1`'s own: `n_nest` 7 (its inter-commit gap), `nest_window` 6, `nest_batch`
24 (6 × 24 = its recorded `n_pool` of 144), `nest_interleave` 3, `n_cand` 64 / `n_cand_phrase` 48 /
`n_score` 96 / `n_pick` 4.

**Arms** (7; `--arms "d0,sel,all4,nest_u,nest_u_key,nest,nest_key"`):

| arm | library | seam | isolates |
|---|---|---|---|
| `d0` | round 4's warm pool, uniform 72 | `audit` | the in-run control |
| `sel` | round 5's plant audition on **reactive** pools | `audit` | the pool contrast's other end |
| `all4` | same library as `sel` | `key` | round 5's best configuration |
| `nest_u` | legato's **nested** per-cell pools, uniform `n_pick` | `audit` | the POOL alone |
| `nest_u_key` | nested pools, uniform `n_pick` | `key` | pool + keyed read |
| `nest` | nested pools + plant audition | `audit` | pool + selection |
| `nest_key` | nested pools + plant audition | `key` | **the legato configuration, end to end** |

`nest_u` reads the *same* per-cell pools `nest` committed from (snapshotted at each of its commits),
so `nest_u → nest` is selection given legato's pool, and `sel → nest` is legato's pool given
legato's selection.

**One deliberate deviation, and it is the main caveat.** The forward model stays **frozen** at its
post-warm state throughout the nesting; legato adapted its FM over 90 cycles. Freezing is what keeps
`reactive`, `live_seg` and `ref_stale` bit-identical to round 4 — without it the in-run control and
the whole cross-round comparison dissolve. So `d2` reproduces legato's **launch-distribution**
nesting but not legato's **model quality**, and any residual gap has to be read against that.

**A statistics correction that changes the target** (found while wiring the fidelity check, and now
enforced in `analyze_delay.py`). legato's headline table reports `e_perf`, a **median** over its 48
eval geometries; this gate's criterion is stated on the **mean** over 24. Read from `L1`'s own
ladder:

| legato arm | median @c75 | **mean @c75** | median @c90 | **mean @c90** | by-seg @c90 (medians) |
|---|---|---|---|---|---|
| `seg_frozen` | 0.0940 | **0.1311** | 0.1206 | **0.1495** | [0.0550, 0.1108, 0.1861] |
| `phrase_frozen` | 0.1011 | **0.1173** | 0.1117 | **0.1213** | [0.0964, 0.1198, 0.1103] |
| `never` | 0.0098 | 0.0099 | — | — | [0.0102, 0.0187, 0.0013] |

So the mean-for-mean target is **0.117–0.150**, not the 0.1026/0.1065 the round-5 record compared
against — that comparison was mean-vs-median and overstated the gap by ~25%. `ref_stale` = 0.1460
is itself a mean, so legato's `phrase_frozen` clears the playability anchor on the mean and its
`seg_frozen` straddles it. Round 5's best stored number (0.1831) is 1.25× above `ref_stale` and
1.40× above legato's `seg_frozen` mean, not 1.7×.

**Three identity gates, all passed before launch.**

1. default path (`--arms ""`, `--quick`) vs the recorded donor smoke `results/dsmoke/delay_gate.json`
   — **0.000e+00 over 72 values** (`d2s0`);
2. the ladder's `d0` arm vs the in-run round-4 sweep — **0.000e+00** (in-run assertion, raises
   otherwise);
3. **cross-round**: `d2`'s `sel` and `all4` arms vs `d1`'s — **0.000e+00 over 72 values** (`d2s1`
   vs `d1s1`), which is what licenses reading `d1`'s ten-arm ladder and `d2`'s nesting arms on one
   axis.

**Reproduce.**

```bash
cd experiments/            # MODAL_PROFILE=chromatic
modal run mjc/practice/offbook/delay_gate.py::delay_gate --quick --tag d2s0
modal run mjc/practice/offbook/delay_gate.py::delay_gate --quick --tag d2s1 \
    --arms "d0,sel,all4,nest_u,nest_u_key,nest,nest_key"
modal run --detach mjc/practice/offbook/delay_gate.py::delay_gate --spawn --tag d2 --seed 0 \
    --arms "d0,sel,all4,nest_u,nest_u_key,nest,nest_key"
python3 mjc/practice/offbook/analyze_delay.py --tag d2 --fetch --figs
```

### Round 6 result (`d2`, complete, 7 arms, ~33 min) — the content port SUCCEEDS; Δ\* = None on every arm

**All three identity gates held in the real run.** Ladder `d0` arm vs the round-4 sweep
**0.000e+00**; `reactive`/`live_seg` identical across all 7 arms **0.000e+00 over 70 values**; and
`d2`'s `sel`/`all4` arms reproduce `d1`'s bit for bit (**0.000e+00 over 72 values**, asserted at
smoke scale before launch), so rounds 5 and 6 read on one axis.

#### 1. Content fidelity: the nesting reproduces legato, and the pool defect is gone

`build_nested()` reproduces legato's pool **geometry** exactly — `n_pool` 144 from 6 traces at every
commit, with **6/6 reactive at seam 0** and **2/6 reactive** at seams 1, 2 and the chain, which is
what `L1`'s own events record — and its **content**:

| cell | legato chosen / per-state oracle | `d1` `sel` (reactive pools) | `d2` `nest` (nested pools) | oracle ratio d1 → d2 |
|---|---|---|---|---|
| 1:0 | 0.0426 / 0.0137 | 0.0414 / 0.0131 (0.96×) | 0.0390 / 0.0139 (**1.01×**) | 0.96× → 1.01× |
| 1:1 | 0.1271 / 0.0481 | 0.2171 / 0.1995 (4.14×) | 0.1253 / 0.0419 (**0.87×**) | **4.14× → 0.87×** |
| 1:2 | 0.1993 / 0.0554 | 0.3023 / 0.2280 (4.12×) | 0.2302 / 0.0502 (**0.91×**) | **4.12× → 0.91×** |
| 3:0 | 0.1005 / 0.0586 | 0.2199 / 0.1273 (2.17×) | 0.1239 / 0.0679 (**1.16×**) | 2.17× → 1.16× |
| 2:1 | (no legato analogue) | 0.4625 / 0.4429 | 0.1914 / 0.1064 | — |

The diagnosis is confirmed and the fix works: the per-state oracle — the best **any** candidate
achieves per launch state, i.e. the pool before selection — was 4.1× legato's at seams 1–2 with
reactive pools and is **0.87–0.91×** legato's with nested pools. `chosen_score` lands at
0.91×/0.99×/1.15×/1.23× of legato's. Cell 2:1 improves 4.2× on the oracle for the same reason.

**Undelayed by-segment medians against legato's plateau** (medians, legato c90):

| source | seg0 | seg1 | seg2 |
|---|---|---|---|
| legato `seg_frozen` | 0.0550 | 0.1108 | 0.1861 |
| legato `phrase_frozen` | 0.0964 | 0.1198 | 0.1103 |
| `nest` `seg_tape` | 0.0611 | 0.1668 | **0.1464** |
| `nest_key` `seg_tape` | 0.0574 | 0.1624 | 0.2052 |
| `nest` `chain` | 0.0460 | 0.1831 | 0.2545 |
| `nest_key` `chain` | 0.0628 | **0.1146** | 0.2047 |
| (`d0` `chain`, round 4) | 0.0451 | 0.1252 | 0.3130 |

Piece error, mean-for-mean: `nest_key` **0.1455** (chain) / **0.1491** (`seg_tape`) against legato's
`phrase_frozen` **0.1173** and `seg_frozen` **0.1311** — **1.24×** and **1.14×**. Medians: `nest_key`
0.1392 / 0.1488 against legato's 0.1011 / 0.0940. The residual is consistent with the one deliberate
deviation (frozen FM vs legato's 90 adapted cycles) but is not attributed here, because nothing in
this round varies the model.

#### 2. The gate: Δ\* = None on all seven arms, and the reason has changed

| arm | chain @Δ0 | seg_tape @Δ0 | best stored | gap closed to `ref_stale` | best stored, any Δ | Δ\* |
|---|---|---|---|---|---|---|
| `d0` | 0.2703 | 0.2682 | 0.2682 | 0.0% | 0.2036 | None |
| `sel` | 0.3557 | 0.2828 | 0.2828 | −11.9% | 0.2119 | None |
| `all4` | 0.2360 | 0.2306 | 0.2306 | 30.7% | 0.1998 | None |
| `nest_u` | 0.2010 | 0.2395 | 0.2010 | 55.0% | 0.1874 | None |
| `nest_u_key` | 0.2199 | 0.2506 | 0.2199 | 39.5% | 0.1727 | None |
| `nest` | 0.1864 | **0.1613** | 0.1613 | 87.4% | 0.1613 | None |
| `nest_key` | **0.1455** | 0.1491 | **0.1455** | **100.4%** | **0.1455** | None |

**`nest_key` is the first stored strategy in this node to reach the playability anchor**: its chain
runs 0.1455 against `ref_stale` = 0.1460 at Δ = 0. Round 4's stored content was 1.84× the anchor.

The verdict rows, and exactly where the criterion breaks:

| Δ | ms | chain | live_seg | reactive | ordered? | min | vs `ref_stale` |
|---|---|---|---|---|---|---|---|
| 0 | 0 | 0.1455 | 0.1978 | **0.0114** | no (reactive wins) | 0.0114 | playable |
| 2 | 40 | 0.1563 | 0.2201 | **0.1110** | no (reactive wins) | 0.1110 | playable |
| 4 | 80 | **0.1595** | 0.2447 | 0.2231 | **no — the MIDDLE term fails** (`live_seg` > `reactive`) | 0.1595 | miss by 9.3% |
| 8 | 160 | **0.1513** | 0.3340 | 0.5294 | **yes** | 0.1513 | **miss by 3.6%** |
| 16 | 320 | **0.1580** | 0.5311 | 0.7190 | **yes** | 0.1580 | miss by 8.3% |

Two facts worth stating precisely because the pre-fixed rule does not ask about them. **(i)** From
Δ = 4 onward the chain is strictly the **best of the three** (0.1595 < 0.2231 < 0.2447); the full
ordering fails at Δ = 4 only because `live_seg` overtakes `reactive`, i.e. the middle term, not the
chain. **(ii)** At Δ = 8 the ordering holds and the round misses the playability guard by
**0.1513 vs 0.1460 — 3.6%**. Both are reported, neither is acted on: Δ is not extended, the
criterion is not re-fitted, and the ceiling is not moved. Δ\* = None stands.

**Delay robustness.** `nest_key`'s chain is the flattest strategy measured anywhere in this node:
**1.09×** across 0 → 320 ms (0.1455 / 0.1563 / 0.1595 / 0.1513 / 0.1580), with a by-segment profile
that barely moves ([0.063, 0.115, 0.205] at Δ = 0 → [0.068, 0.173, 0.210] at Δ = 16) against
reactive's 63.1×. `nest` (audit read) 0.97×, `nest_u_key` 0.79×, `nest_u` 0.93×.

#### 3. Attribution: the pool carries it, selection and the key add on top, and they interact

Single-variable contrasts at Δ = 0 (mean piece error):

| step | what changes | seg_tape | chain |
|---|---|---|---|
| `sel` → `nest` | POOL: reactive → legato nesting (audition read both) | 0.2828 → 0.1613 (1.75×) | 0.3557 → 0.1864 (**1.91×**) |
| `all4` → `nest_key` | POOL: reactive → legato nesting (keyed read both) | 0.2306 → 0.1491 (1.55×) | 0.2360 → 0.1455 (**1.62×**) |
| `nest_u` → `nest` | SELECTION: uniform → plant audition, same nested pool | 0.2395 → 0.1613 (1.48×) | 0.2010 → 0.1864 (1.08×) |
| `nest_u_key` → `nest_key` | SELECTION, keyed read | 0.2506 → 0.1491 (1.68×) | 0.2199 → 0.1455 (1.51×) |
| `nest` → `nest_key` | READ: FM audition → legato key, same library | 0.1613 → 0.1491 (1.08×) | 0.1864 → 0.1455 (1.28×) |
| `nest_u` → `nest_u_key` | READ, **uniform** library | 0.2395 → 0.2506 (0.96×) | 0.2010 → 0.2199 (0.91×) |
| `d0` → `nest_key` | round 4 → all of legato | 0.2682 → 0.1491 (**1.80×**) | 0.2703 → 0.1455 (**1.86×**) |

The pool is the largest single factor (1.6–1.9× on the chain), which is what round 5 was missing.
Round 5's interaction reproduces exactly: **the key pays on audition-selected content and costs on
uniform content** (`nest` → `nest_key` 1.28× on the chain; `nest_u` → `nest_u_key` 0.91×). Selection
carries most of the `seg_tape` effect (1.48–1.68×) and little of the chain's under an audit read
(1.08×) but a lot under a keyed read (1.51×).

**Audition calibration** (realised ÷ its own held-out prediction): `nest_key` **1.13× / 1.17×**
(seg / chain) — legato-like; `nest` (audit read) 1.23× / 1.50×; `d1`'s `sel` was 1.51× / 1.62×.
Reading a state-conditioned library with a state key rather than a rollout score keeps the pairing
the audition validated, at both spans.

#### 4. Ledger

Total agent priced time **51,984 s**, 1,957,440 environment steps, 128,352 feedback events. The two
plant-auditioned builds dominate: `nest` **23,545 s (45%)** — 893,568 steps, 56,736 fb, 26,880 plans
(the nesting practice is the only priced *planning* here) — and `sel` **22,986 s (44%)**. Harvests:
`perf` 2,426 s, `route` 602 s. Uniform builds (`nest_u`) are free by construction: they re-read the
per-cell pools `nest` already paid for.

#### 5. Caveats

Single seed; 24 shared eval geometries; the criterion is on the mean and mean/median disagree in
places (`nest` chain 0.1864/0.1527, `nest_key` chain 0.1455/0.1392 — both *lower* on the median).
**The forward model is frozen** at its post-warm state through the nesting while legato adapted its
over 90 cycles; that is what keeps `reactive`/`live_seg`/`ref_stale` bit-identical to round 4, and
it is the most likely residual against legato's 1.14–1.24×, but this round varies nothing about the
model so the attribution is not made here. `nest_u` reads the pools `nest`'s *auditioned* commits
generated, so the selection contrast is one-variable in the draw but not in the configuration that
produced the pool. `nest_key` at Δ = 8 misses the pre-fixed guard by 3.6%; no ceiling, sweep or
criterion was changed in response.

**Artifacts.** `results/d2/d2_report.txt`, `results/d2/d2_delay_by_arm.png`,
`results/d2/d2_ladder_delay0.png`, `results/d2/d2_by_segment.png`; raw at
`results/d2/d2/delay_gate/results.json`. Smokes: `results/d2s0` (donor identity),
`results/d2s1` (nesting path + cross-round identity vs `d1s1`).


## Round 7 — the model adapts too (`d3`), and the strong incumbent (`d3b`), seed 0, 2026-08-27

Two runs, launched together, attacking the two things that were still soft after `d2`.

### `d3` — legato's plasticity, not just legato's nesting

`d2` reproduced legato's launch-distribution nesting with the **forward model frozen** at its
post-warm state; legato's adapted through its practice (39 cycles of online training by its seam-2
commit, 55 by the phrase commit). That is not cosmetic: a tape harvested off an **uncommitted** seam
is a reactive rendition whose quality is bounded by the model that produced it, and `live_seg` is a
model bet outright. `--nest-adapt` turns legato's own `train_online` call on between commits — plain
uniform-lr steps on a window of practice transitions plus the replay fraction, its own batch stream,
**training data true-state** per round 4's construction — then freezes the model for the sweep.

**Two consequences, handled rather than papered over.**

1. **The in-run identity of `reactive`/`live_seg` against round 4 dissolves by construction.** They
   are library-independent but *model*-dependent. `base_pair()` recomputes them **once** under the
   adapted model (same rng seeds, same geometries, same code path) and every arm is scored against
   those, so the **cross-arm** identity assertion still holds; the round-4 delta is *reported*
   instead of asserted. The replacement in-run control is the content ladder re-stated under the one
   adapted model: the `warm` library (d0's own content) read by `audit` (`d0`) **and** by `key`
   (`key_d0`), plus `sel`/`all4`, which are cheap because their libraries are built *before* the
   nesting and are therefore bit-identical to `d2`'s.
2. **A `key` read never touches the forward model** — `_decide_key` is a nearest-centroid lookup in
   state space. So `key_d0` and `all4` must stay bit-identical to their round-5/6 twins even with
   the model moving underneath, and that is the check that the model did not leak where it should
   not. Verified at smoke scale: `key_d0` vs `d1` **0.000e+00 over 36 values**, `all4` vs `d2`
   **0.000e+00 over 36 values**, while `sel` (same library, *audit* read) moves by 1.59e-01 — the
   pure model effect on the read.

**Anchors.** `ref_stale` is *defined* under the stale pre-practice FM at Δ = 0; it does not move and
the criterion keeps using it untouched. The same ballistic-per-segment anchor is additionally
reported under the **warm** (pre-nesting) and **adapted** (post-nesting) models as two flagged
numbers, so the pair brackets what the adapted cycles bought.

**FM competence on the record.** `react_probe()` runs a held-out reactive traversal
(`who="instrument"`, free) at every nesting stage boundary — `start`, `pre_seg0/1/2`, `pre_chain` —
recording `e_react` median, mean and by-segment, so the model behind each commit sits next to
legato's (its `e_react` ran ~0.010 at c25–39).

**Arms** (8): `d0,key_d0,sel,all4,nest_u,nest_u_key,nest,nest_key` with `--nest-adapt`. Cycle
budget: 20 warm + 7×3 = 41 by the seam-2 commit, 48 by the chain commit, against legato's 39 / 55.

### `d3b` — the honest incumbent: efference copy through the delay

The naive delayed controller in `delay_gate.py` plans from the state Δ steps ago and does not
correct for it. That is a **strawman**: a nervous system with a reflex delay does not act on where it
*was*, it acts on where its forward model says it now *is*, given the commands it has already
issued — the cerebellar forward model is exactly the efference-copy predictor. The sibling `presto`
node added this (`World.obs_predict`) and measured it removing essentially all of the naive delay
penalty on their piece, which it should: **bridging a Δ-step delay *is* a Δ-step FM rollout**, and
Δ = 4–8 sits well inside this plant's ~21-step composition horizon.

**Ported, not re-derived.** `../accompanist/presto/world.py`'s implementation is copied verbatim into
`offbook/world.py`: an `issued` efference-copy buffer alongside `hist`, and

```
obs()  ->  s_hat(t) = s(t-D) ; for u in issued[t-D:t]: s_hat += fm_delta(fm, s_hat, u)
```

in both `traverse` and `traverse_route`, off by default. The offbook copy is now byte-equivalent to
presto's apart from the header and two package-rename import lines.

**Every strategy is re-swept under it, not just the incumbent** — scoring a predictor-equipped
`reactive` against predictor-less stored content would be the mirror strawman. `--dual-obs` runs the
whole ladder twice, storing `sweep`/`verdict` (naive, the round-4 continuity read) and
`sweep_predict`/`verdict_predict` (efference copy) side by side. `ref_stale` is untouched in both,
and the analyzer prints Δ\* under both incumbents with the per-Δ **playability margin**
(min of the three ÷ `ref_stale`).

`d3b` is otherwise **`d2`'s exact configuration** (frozen FM, `d2`'s seven arms), so its naive half
is a bit-identity control against `d2` and the predictive half is the only new variable. This is the
run that says whether `d2`'s "the chain is strictly the best of the three from 80 ms on" survives a
Δ-step FM bet.

### Identity gates, all passed before launch

| gate | result |
|---|---|
| default path (`--arms ""`, `--quick`) vs `results/dsmoke/delay_gate.json`, **after** the `world.py` port | **0.000e+00 over 72 values** (`d3bs0`) |
| Phase-A **G-F fork fidelity** re-run after the `world.py` port | **0.000e+00 over 7 arrays**, pass (`gsmoke2`); G-R and G-P also pass |
| `--arms …` with `nest_adapt` OFF vs `d2`'s smoke — round 6 reproduced | **0.000e+00 over 252 values** (`d3s1` vs `d2s1`) |
| `key_d0` vs `d1`'s — `key` reads are model-independent | **0.000e+00 over 36 values** |
| the same two `key`-read arms under `--nest-adapt` | **0.000e+00** (`key_d0` vs `d1`, `all4` vs `d2`) |
| `obs_predict` is a **no-op at Δ = 0** (in-run assertion, raises otherwise) | **0.000e+00 over 90 values** (`d3bs1`) |
| ladder `d0` arm vs the in-run round-4 sweep (frozen-FM runs only; reported not asserted under `--nest-adapt`) | 0.000e+00 |

### Reproduce

```bash
cd experiments/            # MODAL_PROFILE=chromatic
modal run mjc/practice/offbook/delay_gate.py::delay_gate --quick --tag d3bs0
modal run mjc/practice/offbook/gates.py::offbook_gates --quick --tag gsmoke2     # G-F after the port
modal run mjc/practice/offbook/delay_gate.py::delay_gate --quick --tag d3s1 \
    --arms "d0,key_d0,sel,all4,nest_u,nest_u_key,nest,nest_key"
modal run mjc/practice/offbook/delay_gate.py::delay_gate --quick --tag d3s2 --nest-adapt \
    --arms "d0,key_d0,sel,all4,nest_u,nest_u_key,nest,nest_key"
modal run mjc/practice/offbook/delay_gate.py::delay_gate --quick --tag d3bs1 --dual-obs \
    --arms "d0,sel,all4,nest_u,nest_u_key,nest,nest_key"
# the runs (--spawn only)
modal run --detach mjc/practice/offbook/delay_gate.py::delay_gate --spawn --tag d3 --seed 0 \
    --nest-adapt --arms "d0,key_d0,sel,all4,nest_u,nest_u_key,nest,nest_key"
modal run --detach mjc/practice/offbook/delay_gate.py::delay_gate --spawn --tag d3b --seed 0 \
    --dual-obs --arms "d0,sel,all4,nest_u,nest_u_key,nest,nest_key"
python3 mjc/practice/offbook/analyze_delay.py --tag d3  --fetch --figs --vs d2
python3 mjc/practice/offbook/analyze_delay.py --tag d3b --fetch --figs --vs d2
```

**Note on ordering.** `d3` was launched *before* the `world.py` `obs_predict` port, so it ran the
pre-7b code — i.e. the naive operator, which is exactly its intended round-4-continuity read. The
port is byte-neutral at its default (two independent gates above), so `d3` stays reproducible
against the current tree.

### Round 7 result (`d3`, complete, 8 arms, ~36 min) — **Δ\* = 8 (160 ms). THE GATE PASSES.**

**Sanity, with the two references that had to change.** Cross-arm identity of the
library-independent strategies holds at **0.000e+00 over 80 values** against the pair recomputed
under the adapted model. The `d0` arm vs round 4 is **4.754e-01** — expected nonzero and reported,
not asserted: the FM moved and `d0` is an `audit` read. The adapted pair vs round 4's differs by
**1.269e-01**, which *is* the model effect. And the two model-independence checks pass in the real
run: **`key_d0` vs `d1` = 0.000e+00 over 60 values**, **`all4` vs `d2` = +0.0000 at every Δ** — a
`key` lookup never touches the forward model, and nothing leaked.

**The forward model through the nesting** (held-out `e_react`, free instrument; legato's ran ~0.010
at its c25–39 commits):

| stage | nest cycle | e_react med | e_react mean | by-segment medians |
|---|---|---|---|---|
| `start` | 0 | **0.0098** | 0.0122 | 0.0088 0.0103 0.0036 |
| `pre_seg0` | 7 | 0.0102 | 0.0134 | 0.0113 0.0140 0.0030 |
| `pre_seg1` | 14 | 0.0160 | 0.0189 | 0.0079 0.0385 0.0017 |
| `pre_seg2` | 21 | 0.0068 | 0.0072 | 0.0066 0.0107 0.0031 |
| `pre_chain` | 28 | **0.0063** | 0.0070 | 0.0074 0.0084 0.0021 |

The warm-20 model was *already* at legato's competence (0.0098 vs ~0.010) before a single nested
cycle; adaptation took it to 0.0063 with one dip at `pre_seg1`. Undelayed `reactive` improves
0.0114 → **0.0065** and `live_seg` 0.1978 → **0.1573**.

**Content fidelity: `nest` now matches or beats legato on every cell.**

| cell | legato chosen / oracle | `d3` `nest` chosen / oracle | ratios |
|---|---|---|---|
| 1:0 | 0.0426 / 0.0137 | 0.0403 / 0.0134 | 0.95× / **0.98×** |
| 1:1 | 0.1271 / 0.0481 | 0.1224 / 0.0342 | 0.96× / **0.71×** |
| 1:2 | 0.1993 / 0.0554 | 0.1648 / 0.0463 | 0.83× / **0.84×** |
| 3:0 | 0.1005 / 0.0586 | 0.1148 / 0.0603 | 1.14× / **1.03×** |

Pool geometry stays legato's exactly (144 rows from 6 traces; 6/6 reactive at seam 0, 2/6 after).

**The gate, arm `nest`** (`ref_stale` = 0.1460, untouched):

| Δ | ms | chain | live_seg | reactive | ordered? | min ÷ `ref_stale` | verdict |
|---|---|---|---|---|---|---|---|
| 0 | 0 | 0.1255 | 0.1573 | **0.0065** | no | 0.045× | playable |
| 2 | 40 | 0.1244 | 0.1823 | **0.0781** | no | 0.535× | playable |
| 4 | 80 | **0.1205** | 0.2121 | 0.1971 | no (middle term: `live_seg` > `reactive`) | 0.825× | playable |
| 8 | 160 | **0.1416** | 0.2818 | 0.4025 | **yes** | **0.970×** | **PASS** |
| 16 | 320 | 0.2001 | 0.5045 | 0.7132 | yes | 1.371× | not playable |

**Δ\* = 8 — 160 ms, inside the human range, and the round's pre-fixed criterion is satisfied for
the first time in this node.** Nothing was re-fitted: same Δ sweep, same ordering, same `ref_stale`
computed under the stale pre-practice FM. The margin is 3.0% inside the guard.

**Δ\* by arm:** `nest` **8**; every other arm None. `nest_key` is ordered at Δ = 8 but misses
playability at 1.042× (0.1521 vs 0.1460) — **under the adapted model the `audit` read overtakes the
`key` read**, reversing rounds 5–6, where keying was worth 1.28× on the chain. The FM audition is
now accurate enough at chain span to out-pick the launch key.

**The ladder at Δ = 0** (mean; medians in brackets where notable):

| arm | chain | seg_tape | best stored | gap closed to `ref_stale` |
|---|---|---|---|---|
| `d0` | 0.2908 | 0.1568 | 0.1568 | 91.1% |
| `key_d0` | 0.2669 | 0.3265 | 0.2669 | 1.1% |
| `sel` | 0.2770 | 0.2319 | 0.2319 | 29.7% |
| `all4` | 0.2360 | 0.2306 | 0.2306 | 30.7% |
| `nest_u` | 0.1504 | 0.2389 | 0.1504 | 96.4% |
| `nest_u_key` | 0.2077 | 0.2465 | 0.2077 | 49.5% |
| `nest` | **0.1255** [0.0960] | **0.1116** [0.0961] | **0.1116** | **128.1%** |
| `nest_key` | 0.1297 [0.0971] | 0.1355 [0.1178] | 0.1297 | 113.3% |

`nest`'s stored content is now **below legato's own means** (`seg_frozen` 0.1311, `phrase_frozen`
0.1173) on both spans, and below the playability anchor by 28%.

**`d2` → `d3` per-arm deltas** (the model's own effect, everything else held): `all4` **+0.0000 at
every Δ** (key read, library built pre-nesting — the control); `d0` `seg_tape` −0.1114 at Δ = 0
(round 4's *uncurated* library read by a better model is 1.71× better); `nest` −0.0497 / −0.0608
(seg/chain) at Δ = 0 and −0.0887 / −0.0510 at Δ = 8; `sel` −0.0509 / −0.0787.

**An anchor curiosity, reported and not interpreted.** The same ballistic-per-segment quantity is
0.1460 under the **stale** FM (the guard), **0.2298** under the warm FM, and **0.1814** under the
adapted FM — i.e. per-segment ballistic planning is *worse* with a better model here. legato's
README flags a related asymmetry (stale vs ceiling models degrading for different reasons). The
criterion uses only the stale-FM number, which is unchanged from round 4, so nothing in the verdict
depends on this.

**Ledger.** Agent priced time **51,984 s** — identical to `d2`'s, because online FM training costs
GPU time but no priced environment steps or feedback events. `nest` build 23,545 s (45%, 26,880
plans — the nesting practice is the only priced planning), `sel` build 22,986 s (44%), harvests
`perf` 2,426 s / `route` 602 s.

**Caveats.** Single seed; 24 shared eval geometries; the criterion is on the mean and `nest`'s
medians are lower throughout (chain 0.0960 at Δ = 0, 0.1161 at Δ = 8), so the PASS is not a
median-vs-mean artifact in the favourable direction. The Δ = 8 pass sits 3.0% inside the guard, and
the Δ = 4 row fails only on the middle term (`live_seg` > `reactive`), not on the chain. `nest_u`
reads the pools `nest`'s auditioned commits generated. The **incumbent here is the naive delayed
controller**; whether the pass survives an efference-copy incumbent is exactly what `d3b` asks.

**Artifacts.** `results/d3/d3_report.txt`, `results/d3/d3_delay_by_arm.png`,
`results/d3/d3_ladder_delay0.png`, `results/d3/d3_by_segment.png`; raw at
`results/d3/d3/delay_gate/results.json`.

### Round 7b result (`d3b`, complete, 7 arms × 2 operators, ~40 min) — **Δ\* = None under BOTH incumbents**

**Four controls, all clean.** The ladder `d0` arm vs the round-4 sweep **0.000e+00**; cross-arm
identity of the library-independent strategies **0.000e+00 over 70 values**; `obs_predict` is a
**no-op at Δ = 0** — **0.000e+00 over 90 values**, asserted in-run; and the whole naive half is
bit-identical to `d2` — **+0.0000 across all 7 arms × 5 Δ × 2 strategies (140 values)**. So the
efference-copy half is the only new variable in the run.

**The strong incumbent is very strong.** Efference copy costs the agent only the part of the last Δ
steps it could not have predicted (the motor noise) plus the model's own drift, and on this plant
Δ = 2–8 sits well inside the ~21-step composition horizon:

| Δ | ms | reactive naive → efference | live_seg naive → efference |
|---|---|---|---|
| 0 | 0 | 0.0114 → 0.0114 (no-op) | 0.1978 → 0.1978 (no-op) |
| 2 | 40 | 0.1110 → **0.0191** (5.8×) | 0.2201 → 0.2113 (1.04×) |
| 4 | 80 | 0.2231 → **0.0210** (10.6×) | 0.2447 → 0.1837 (1.33×) |
| 8 | 160 | 0.5294 → **0.0828** (6.4×) | 0.3340 → 0.1618 (2.06×) |
| 16 | 320 | 0.7190 → **0.3559** (2.0×) | 0.5311 → 0.3750 (1.42×) |

Reactive's delay degradation drops from **63.1× to 31.2×**, and at 80 ms a predictor-equipped
reactive controller is **0.0210** — 1.8× *better* than the undelayed ballistic anchor. Most of
round 4's headline delay tax was the naive operator, not the delay.

**Δ\* = None on every arm under both operators.** Per-Δ playability margins (min of the three ÷
`ref_stale`, naive / efference; `<1.00` = inside playability):

| arm | Δ=0 | Δ=2 | Δ=4 | Δ=8 | Δ=16 | naive Δ\* | efference Δ\* |
|---|---|---|---|---|---|---|---|
| `d0` | 0.08/0.08 | 0.76/0.13 | 1.53/0.14 | 2.29/0.57 | 3.13/1.97 | None | None |
| `sel` | 0.08/0.08 | 0.76/0.13 | 1.53/0.14 | 2.29/0.57 | 2.16/2.44 | None | None |
| `all4` | 0.08/0.08 | 0.76/0.13 | 1.53/0.14 | 1.58/0.57 | 1.79/1.58 | None | None |
| `nest_u` | 0.08/0.08 | 0.76/0.13 | 1.53/0.14 | 1.66/0.57 | 1.28/1.77 | None | None |
| `nest_u_key` | 0.08/0.08 | 0.76/0.13 | 1.22/0.14 | 1.35/0.57 | 1.18/1.61 | None | None |
| `nest` | 0.08/0.08 | 0.76/0.13 | 1.34/0.14 | 1.32/0.57 | 1.24/1.30 | None | None |
| `nest_key` | 0.08/0.08 | 0.76/0.13 | 1.09/0.14 | 1.04/0.57 | 1.08/**0.98** | None | None |

Under efference copy the whole Δ ∈ {0, 2, 4, 8} region becomes *comfortably* playable (margins
0.08–0.57) — and reactive owns all of it.

**`d2`'s "the chain is strictly the best of the three from 80 ms on" does NOT survive.** The first
Δ at which the chain beats both live strategies:

| arm | naive | efference copy |
|---|---|---|
| `nest`, `nest_key`, `nest_u_key` | **4** (80 ms) | **16** (320 ms) |
| `all4`, `nest_u` | 8 | **16** |
| `d0`, `sel` | 16 | 16 |

Under the honest incumbent the chain is best only at 320 ms, on every arm.

**The one thing that does hold up.** `nest_key`'s chain under efference copy is **flat and under the
anchor at every delay**: 0.1455 / 0.1425 / 0.1425 / 0.1425 / 0.1425 (0.98–1.00× `ref_stale`;
degradation **0.98×** — marginally *better* at 320 ms than at 0). At Δ = 16 it is strictly the best
of the three *and* playable (min ÷ anchor **0.976**), but the pre-fixed criterion still fails on the
**middle term**: `live_seg` 0.3750 > `reactive` 0.3559. That is the **third** appearance of this
pattern (`d2` at Δ = 4, `d3` at Δ = 4, `d3b` at Δ = 16) — the ordering the criterion asks for is
broken by the live *segment* plan overtaking the reactive controller, not by the chain.

**Stored content is mostly slightly WORSE under the predictor** — the efference copy changes which
tape the seam selector picks, and not always for the better. Best stored error at any Δ, naive →
efference: `d0` 0.2036 → 0.2365, `sel` 0.2119 → 0.2821, `all4` 0.1998 → 0.2291, `nest_u`
0.1874 → 0.2010, `nest_u_key` 0.1727 → 0.2110, `nest` 0.1613 → 0.1613, `nest_key`
0.1455 → **0.1425**. Only the fully-legato arm improves.

**The comparison this round does NOT make.** `d3b` runs `d2`'s configuration (FM frozen) under both
operators; `d3`'s Δ\* = 8 pass was under the *adapted* FM with the *naive* incumbent. The
adapted-FM × efference-copy cell was not run, so whether `d3`'s pass survives the strong incumbent
is **untested**. What is measured: under the frozen FM, efference copy improves reactive 6.4× at
Δ = 8 (0.5294 → 0.0828), while `d3`'s passing chain scored 0.1416 there.

**Ledger.** Agent priced time **51,984 s** — identical to `d2` and `d3`, since the second sweep is
experimenter instrumentation (`who="instrument"`); instrument steps rise 163,392 → 314,352.

**Caveats.** Single seed; 24 shared eval geometries; the criterion is on the mean. The predictor
uses the *same* forward model the agent plans with, so its quality and the planner's are not
separable here — the natural next question, and the one Jasper is redirecting to.

**Artifacts.** `results/d3b/d3b_report.txt`, `results/d3b/d3b_delay_by_arm.png`,
`results/d3b/d3b_ladder_delay0.png`, `results/d3b/d3b_by_segment.png`; raw at
`results/d3b/d3b/delay_gate/results.json`. Smokes: `results/d3bs0` (donor identity after the
`world.py` port), `results/d3bs1` (dual-operator path), `results/gsmoke2` (G-F re-run).
