# fourwall — File Index

Complete file-by-file reference for this node. **Findings**: [README.md](README.md) (the four-tag
writeup, 2026-08-17). This file documents **machinery and calibration only**.

**Idea doc**:
[`recurrence_manufactures_confounds`](../../../../ideas/recurrence_manufactures_confounds.md)
(§2 the confound debt, §3 the scaffold's value, §4 the rotation typed, §5 the merge op, §8 the
instrument list). **Spec sibling**: `../merge/SPEC.md`[^private] — this node builds a
narrowed slice of it (see *Scope, and what was deliberately left out* below).
**Machinery donors**: [`../ratchet/`](../ratchet/README.md) (substrate, macros, beam, pricing,
online plant and selector), [`../setlist/`](../setlist/FILES.md) (`DecayMiner` via
`../transpose/drift.py`, the demand primitives, the calibration discipline),
[`../ear/`](../ear/FILES.md) (shared setup, `write_results`),
[`../crystallize/units.py`](../crystallize/FILES.md) (level moves, hierarchical damage,
exact-DP oracle).

## What the round installs

`transpose` moved **what is true**. `setlist` moved **what is asked**. Both moved the content a
library is graded on. This round holds content *and* demand fixed and moves the **index**.

Every task instance carries a surface token `w`. In phase 1 `w = f`, where `f` is the true
level-2 feature the damage destroyed at the keyed node — a perfect, free-to-read correlate of a
latent the agent would otherwise have to infer from the root and the intact siblings. A
**rotation** event then permutes the map (`w = (f + q) mod v`) without deleting `w`, so a raw
`w`-key becomes actively misleading rather than merely absent. Phase 2 rotates cyclically every
`rot_period` cycles.

**The admissibility gate is discharged by construction, not by measurement.** `setlist` had to
prove that demand moved while difficulty did not (six gates, a GPU calibration run). Here the
metering instances, their roots, their damage, their latent keys and the grammar are *the same
arrays* at every cycle of every arm; the rotation is a relabelling of an observable that no
grading path reads. There is nothing for a hardening confound to enter through.

**Contamination is impossible by construction, deliberately.** A committed unit is a table of
level-1 feature tuples; there is no slot in it for `w`. `w` reaches the agent through exactly
two doors — the library's **key** (the keyed arms) and the **selector's input**
(`free_selector`) — so any `wall_track` vs `wall_merge` difference is *pure index compression*.
Arms whose programs read `w` are a deliberate non-goal.

## Code files

| File | Purpose |
|---|---|
| `rekey.py` | **`fw_s1`.** The re-key op: earning the index's *basis* rather than only pruning it. `service_matrix` is the exact model-free record of which committed program repairs which instance. `greedy_cover` is the closure of "the same unit serves them" — a minimal covering set of programs, each instance labelled by the one that covers it; that labelling **is** the earned partition, and nothing in it can see the latent. `Router` makes an earned class *addressable*: a linear read of the agent's own controller state onto those self-generated labels, deliberately the same shape as `fourwall.key_probe` so the gap against Gate 0's 0.927 oracle probe is legible. `adjusted_rand` is an oracle readout only |
| `wall.py` | The wall primitives, kept out of the app so they are auditable and gate-testable with no substrate. `wall_map` / `rotation_index` / `is_rotation_cycle` are the schedule. `sample_pool_latent` / `latents_at` / `context_instances_wall` are `ratchet.context_instances` with the generative latents carried out alongside (`sample_derivations_weighted(..., return_trace=True)`), so the key is **exact** rather than parsed back through the last-writer-wins inverse map. `feature_leaves` / `entry_leaves` / `write_span` / `entry_success` / `entry_profile` / `consistent_features` are the exact, **model-free** readouts the grammar's own possible-set DP makes available — the forced-transfer instrument of the idea doc's §5. `Library` is key → index class → committed table, carrying a per-cell `DecayMiner`, a committed table, an observation count and a ring of the cell's own recent consumption; `merge` is the only method that touches the partition. `dedup_union` is the merged cell's content. `transfer_matrix` / `merge_candidates` are the alias evidence |
| `fourwall.py` | The Modal app. `build_world` draws every graded instance once. `wall_refs` is the references **and Gate 0**. `build_wall_head` / `build_wall_value` give `free_selector` a selector that can read `w`, zero-initialised so it is bit-identically the plain head as a function at cycle 1; the `WallValue` wrapper reconstructs the wall vector from `beam_moves`' own `repeat_interleave`, which is what lets `ratchet.beam_moves` be used **unmodified**. `value_steps_w` is `ratchet.value_steps` with the wall threaded. `beam_keyed` runs one beam per distinct action set and reassembles in order; `build_action` places a class's macro at the damaged node **only**. `run_arm` is the loop. `policy_err` is the priced grader, `merge_audition` the priced second evidence route, `rotation_profile` the exact per-entry forced-transfer readout at each turn. Entrypoints: `fourwall` (main), `gate0`, `selfcheck{,_remote}` |
| `analyze_fourwall.py` | Reduction. `--fetch` pulls from the volume; the report runs Gate 0, phase-1 priced time-to-competence, the rotation response, phase-2 error and spend, **library key cardinality over time** (the MDL readout), recert, merge (with phase 1 as its negative control), the wall-permutation probe, the first-rotation forced-transfer profile, and the instruments. `--figures` writes fig1–fig4 |
| `launch_detached.py` | `setlist`'s session-isolated launcher (`start_new_session=True`), retargeted |

## Arms

The library key is the **only** difference. Everything else — substrate, damage cell, plant,
mining, commit rule, recert, pricing — is identical across arms.

| arm | library key | index | selector reads `w` | merge | re-key | retire | round |
|---|---|---|---|---|---|---|---|
| `given_key` | the true level-2 latent `f`, oracle | one cell per feature | no | no | no | — | all |
| `wall_track` | `w` | one cell per wall | no | no | no | — | `fw_s0` |
| `wall_merge` | `w` | one cell per wall, **coarsenable** | no | **yes** | no | — | s0–s2 |
| `unkeyed` | — | one cell | no | — | no | — | `fw_s0` |
| `free_selector` | — | one cell | **yes** | — | no | — | `fw_s0` |
| `wall_rekey` | `w`, then **earned** | wall cells + earned cells | no | no | **yes** | — | s1, s2 |
| `scratch_rekey` | — , then **earned** | one cell + earned cells | no | no | **yes** | — | s1, s2 |
| `rekey_mothball` | `w`, then **earned** | wall cells retired **unmaintained but servable** | no | no | **yes** | mothball | `fw_s3` |
| `rekey_delete` | `w`, then **earned** | retired wall cells **removed** (keys → no unit) | no | no | **yes** | delete | `fw_s3` |

`given_key` is the upper anchor and is **rotation-invariant by construction**: its key is the
latent, which the wall's rotation cannot touch. In phase 1 `w` is a bijection of `f`, so
`given_key` and `wall_track` induce *literally the same partition* — the two arms are the same
experiment until the first rotation, which is what makes the rotation a clean within-design
manipulation rather than a between-arm comparison.

**`unkeyed` is not one of the four; it is the confound control.** `free_selector` differs from
the keyed arms in two ways at once (no index, *and* a selector that reads `w`), so `unkeyed`
holds the second fixed to separate them: `unkeyed` vs `given_key` prices the index,
`free_selector` vs `unkeyed` prices the wall as a selector feature.

## `fw_s1`: the re-key op

`fw_s0` gave the library one index op — **merge** — which can only *delete* distinctions. It
collapsed a misleading 8-cell wall index to 2 and extinguished the rotation response, but it
could never beat having no index at all, because deletion cannot *build* a class. `fw_s0`'s
Gate 0 named the missing regime exactly: the true key is linearly decodable from the agent's own
state at 0.927 while the consuming DP recovers it at 0.518 — the information is there and the
credit machinery is not wired to route on it. Re-key is that wiring, earned.

**Functional addressing.** Two contexts are in the same class iff the same committed unit serves
them. Merge is the pairwise version of that test; re-key is its closure. Four pieces, in loop
order:

1. **`service_matrix`** — `S[p, i]` = "does program `p` repair instance `i`", decided by the
   grammar's own possible-set DP. Exact, model-free, and **priced**: it is the agent asking its
   own feedback channel, once per (program, instance).
2. **`greedy_cover`** — a minimal covering set of programs over a pooled bank of recent
   practice instances; each instance is labelled by the program that covers it. That labelling
   *is* the earned partition, and its size is the earned index's cardinality. `rekey_min_gain`
   stops a stray instance from spawning a class.
3. **`Router`** — a linear read of `controller.state(x)` (+ root) onto those labels, so an
   earned class becomes **addressable** from the observation. The labels are self-generated;
   the latent is never used.
4. **The migration gate** (`rekey_gate`, corrected in `fw_s2`) — both routings are run through
   the **real beam on the same held-out instances** and compared on the **policy error each
   produces**, not on any-entry coverage (see the machinery note above for why `fw_s1`'s
   coverage gate never opened). Routing is then per-instance: an earned class takes over where
   the router's confidence clears `rekey_conf`, otherwise the incumbent keeps it — which is
   what makes the migration a *curve* rather than a switch.
5. **The decision rule** (`fw_s2`) — `ear`'s shape, ported to the index. Two ways in, both
   agent-internal, no persistence on either: **(A) provisional**, on a collapse in the arm's
   *own metered competence* (`e` exceeding the median of the last `collapse_win` cycles by
   `collapse_delta`) — adopt at once, unproven; **(B) certified**, when the corrected gate says
   the earned basis is simply better by `rekey_margin`. Provisional adoption is made safe by
   **`revert_net`**: each active earned class is graded in consumption on its own recent
   instances against what the incumbent would have done with them, and a class reading worse
   across `revert_persist` consecutive checks stops being routed to. A reverted cell **keeps its
   content and keeps mining** — only the routing decision is withdrawn.

**Why the trigger is competence and not the rotation boundary.** Reading the boundary directly
would be simpler code and an **oracle**: the rotation is unannounced in `fw_s0`/`fw_s1` (the
arc's invariant since `setlist` — "no arm reads a demand signal"), and letting the re-key arms
alone see it would hand them a signal no other arm has. What an arm may read is its own
competence, and a collapse in that is what a rotation feels like from inside.

**Stale-cell re-mining is deliberately NOT suppressed.** The incumbent wall cells keep tracking
while the earned basis climbs, so the tracking organ and the migration organ race each other on
the same priced clock. That competition is a measured quantity of this round, not a nuisance.

**Why the gate has the right shape without being told the schedule.** In phase 1 a wall-keyed
incumbent is near-perfect, so `wall_service ≈ 1` and the gate stays shut — the scaffold is left
alone. After a rotation the incumbent collapses and the gate opens. The tear-down is paid for by
**evidence**, not scheduled. `smoke1` showed exactly this: `wall_rekey` adopted at c12 with
incumbent service already down to 0.229 against the earned basis's 0.573.

**Cells are keyed by program identity** (`Library.sync_earned`), not by position, so a class
whose covering program survives a re-key keeps the content it has been accumulating and only
genuinely new classes start empty. That is what makes it a migration rather than a rebuild.
Earned cells are ordinary cells otherwise: they mine, they commit, they recert.

**Undefined at t = 0, by construction.** With no committed units the cover is empty and there
are no classes — the basis cannot precede the vocabulary. That is the ratchet's shape one level
up, and it is the regime `fw_s0` could not enter.

**Forks not taken (cheap version built first, as before):** the cover is greedy, not optimal;
the router is linear on the existing state, not a trained encoder; migration is a per-instance
confidence threshold over a single global gate, not per-class adoption; the cover is recomputed
each check rather than updated incrementally. `wall_rekey` carries re-key **without** merge, so
the two index ops stay separated rather than compounding.

## `fw_s3`: the retire op — what to do with the scaffold once you have migrated off it

`fw_s2` left `wall_rekey` paying **16.16% of priced time recerting wall cells that route ~7% of
instances**, and holding 8 wall + 6 earned cells for a `stored/served` of 3.12 (against
`given_key`'s 1.29). The scaffold had been migrated off but not put down. `fw_s3` adds one op
with two settings, so the bill separates into two questions:

| arm | content | maintenance | isolates |
|---|---|---|---|
| `wall_rekey` (fw_s2) | kept | kept | — (the reference) |
| **`rekey_mothball`** | **kept**, still servable | **stopped** | the **rent** of maintenance |
| **`rekey_delete`** | **removed** (keys route to class −1, i.e. no unit at all) | stopped | the value of the retained **fallback content** |

Both stop maintenance, so `mothball` vs `delete` isolates the content and either vs `fw_s2`'s
`wall_rekey` isolates the rent. `delete` is the direct test of the idea doc §5's
**routing-not-pruning** prediction — that the wall-index should persist as a fallback
re-grounding beacon once it is no longer the primary address.

**The trigger is evidence-driven, never scheduled**, in the spirit of the other gates: a wall
cell is retired only once the index has migrated (`rekey_on`) **and** that cell's own **routing
share** has stayed below `retire_share` across a `retire_win`-cycle window. Routing share is
agent-internal, and nothing in the trigger reads the rotation.

**Instrumented as first-class**: `retire` events (cycle, class, mode, share history, entries
surrendered); `n_wall_maintained` / `n_retired` in the library state; a per-cycle **spend
composition** (`log["spend"]`, one entry per maintenance organ plus the residual) so the freed
budget can be followed rather than assumed; and the **fallback population** (`log["fallback"]`)
— the fraction of instances the router will not take, with their error reported separately from
the migrated ones, since those are exactly the instances a retired cell would otherwise have
served.

**The cross-tag freeze check FAILED, so `wall_rekey` is re-run in-tag.** `fw_s3`'s arms are
`wall_rekey`'s exact configuration plus the retire op, so cycles before adoption ought to be
identical to `fw_s2`'s `wall_rekey`. Priced time and the whole library trajectory *are*
identical (c1 `t` 27686, `cls=8+0e cm=2 ent=4/3 mined=24` in both), but the error series is not
(c1 0.6380 vs 0.6432; c2 0.5703 vs 0.4375). **Cause: `torch`'s global RNG.**
`ratchet.finetune_generator` draws `n_mask` with `torch.randint(...)` from the global generator,
which is seeded once per *run* and never reset per *arm* — so arm #1 of `fw_s3` inherits a
different stream position than arm #3 of `fw_s2`. This is the arc's standing stream-position
effect (`recital` measured it at **±0.034**) and it is why the arc reports rank orderings rather
than absolute fractions. Every prior round has it; nothing is wrong with the runs. It simply
means cross-tag *level* comparisons at this resolution are not licensed, so `fw_s3` carries
`wall_rekey` as its own in-tag reference.

**The `c121` identity return is the built-in discriminator.** At `q = 8` the wall means what it
meant in phase 1, so a *mothballed* cell's content can come back into fashion and a *deleted*
one cannot. It is the one event in the schedule that separates the two ops on the revival axis,
and it costs nothing extra to read.

## Machinery note: coverage is not the objective, and an evaluator that scores it will lie

**Exported from `fw_s1`; reusable anywhere in this arc.** `fw_s1`'s migration gate asked
*"does **any** entry of the routed cell's table repair this instance?"* — a **coverage**
question. The arms are graded on **held-out policy error under current consumption**. Those two
came apart badly:

| `wall_rekey`, fw_s1 | value |
|---|---|
| incumbent index's coverage (`wall_service`) | **0.895** |
| incumbent index's actual phase-2 policy error | **0.338** |
| entries per wall cell, c50 → c80 → c130 | 1.75 → 4.25 → **5.25** |
| stored entries per distinct program served | **6.00** |

A stale cell that keeps re-mining accumulates entries until it holds roughly one program per
feature. It then answers the coverage question well *by enumeration* while remaining a bad
index — the beam still has to choose among its entries, which is exactly where the error is. So
the gate preferred a bloated incumbent that the policy did not, and the earned basis (cover 4–5,
router accuracy 0.95, ARI 0.53) was never adopted in 130 cycles.

This is the étude's **seam law in its third coordinate** (`ear`: an audition must match the
consumption distribution) applied to the *index* rather than to an entry, and `ratchet`'s
finding (**the vocabulary's value is concentration, not coverage**) restated for the addressing
layer. The rule it yields:

> **Grade an index op in the currency the arm is graded in.** Any-entry coverage is monotone in
> table size, so it rewards precisely the enumeration that having an index is supposed to avoid.
> `fw_s2`'s `rekey_gate` runs the real beam on held-out instances under both routings and
> compares the error each produces.

A second, independent lesson from the same run: **certify-then-adopt with hysteresis vetoed its
one correct firing.** `wall_rekey` cleared the gate by **+0.137 at c55**, four cycles after the
first rotation — exactly where the theory says the incumbent should be worst — and
`rekey_persist = 2` required a second consecutive pass that never came, because the incumbent's
re-mining had restored its coverage by c60. `fw_s2` replaces this with `ear`'s shape:
**provisional adoption plus a revert net**, no persistence on the way in.

## Design decisions, and the forks not taken

- **One damage cell, not the depth ladder.** `era = "2:3"` throughout. The ladder is `ratchet`'s
  instrument for the *vocabulary* question; this round's variable is the index, and a single
  fixed cell is what makes "the same held-out instances at every cycle" possible. Level-3
  damage survives as an unpriced **deep probe** (`deep_era = "3:1"`), never trained on.
- **The macro is placed at the damaged node only**, not at every node of its level.
  `ratchet.macro_moves` instantiates everywhere because its vocabulary is position-independent;
  a keyed library's entry is *what to write at the node the key is about*, so placing it
  elsewhere would be keying on one node and acting on another. `wall_refs` reports `ratchet`'s
  all-nodes convention beside the node-only one so levels stay comparable to the published
  ladder.
- **Demand is static and is the DGP's own (`demand_sigma = 0`).** Measured before the run: the
  marginal over the level-2 feature at the damaged node is *already* skewed by the grammar —
  H = 1.56 of a possible 2.08 nats, one feature at 37% and one at 0% — so manufactured
  recurrence arrives with the substrate. Layering `setlist`'s OU concentration on top starves
  cells instead of sharpening the question (at σ = 1.0 one feature takes 53% and three take 0%,
  leaving the index almost nothing to be wrong about). The knob is kept and swept in the
  calibration table below. **One lever: the wall.**
- **Rotation is a cyclic shift** (`w = (f + q) mod v`, `q` incrementing by `rot_step`), which is
  the line dance's own structure and gives a free instrument: after `v` rotations the wall means
  what it meant at the start again, so "can a stale cell come back into fashion" is answered
  without a second design.
- **Merge fires on two routes, and the cheap one is a screen for the priced one.** Route (a),
  `transfer_matrix`, is exact and model-free — the grammar's DP decides whether class *a*'s
  programs repair class *b*'s recent demand — and costs only graded configurations. Route (b),
  `merge_audition`, is the real policy comparing `e_keep` (each cell's table on its own demand)
  against `e_merge` (the deduplicated union on both). **The fork**: `merge_audit = False` runs
  the cheap version alone. Default is both, because the exact screen is nearly free and the
  audition is the route the idea doc's §5 names as the one the dance protocol actually runs.
- **Merge takes the deduplicated union, never an average.** `crystallize`'s law (averaging valid
  renditions destroys them, 3.6–5.0×) is about blending one *unit* with another; a vocabulary is
  a *set* of units, and two cells holding the same program contribute it once. That
  deduplication is where index compression shows up in storage, which is why
  `n_entries_total` and `n_entries_distinct` are both logged.
- **Merge carries hysteresis** (`merge_persist = 2`). Merge is irreversible and coarsening a
  *right* index cannot be undone, so a pair must survive two consecutive checks. This is what
  keeps a bootstrap accident — two cells briefly holding the same junk before either has earned
  its content — from collapsing the index for good. A dry run without it fired 5 merges where
  the hysteresis version fired 3.
- **Merge is *not* disabled in phase 1, deliberately.** Phase 1 is merge's negative control:
  the wall is a perfect key there and a correctly-behaving merge op should find nothing to do.
  The phase-1 merge count is a readout, not a nuisance.
- **`free_selector`'s wall embedding is zero-initialised** and its head is otherwise a deep copy
  of the shared `value0`, so at cycle 1 it is the plain selector *as a function*. Any binding to
  `w` is therefore acquired online, which is what makes the arm a measurement of **spontaneous**
  binding rather than of a gift.

## Gates (`selfcheck`, no GPU, no substrate)

| Gate | What it asserts |
|---|---|
| **W-1** | the key is the true latent: writing the canonical derivation of `f` at the keyed node repairs its own instance, always |
| **W-2** | the wall map is a bijection, and every rotation is a **derangement** (no wall keeps its meaning) |
| **W-3** | the schedule: phase 1 stationary at q = 0, one rotation per period thereafter |
| **W-4** | merge coarsens the index and **deduplicates** content — it neither invents nor destroys programs, and the merged cell holds the union |
| **W-5** | an unkeyed library routes every key to one class |
| **W-6** | `consistent_features` is exact: the true feature is consistent on **every** instance |
| **W-7** | a merged-away class leaves the wall id set, so every id the screen enumerates is still addressable. *A regression gate*: `smoke1` found this the hard way — `wall_ids` was the constructor's set and `merge` never pruned it, so the second screen after a merge raised `KeyError` |
| **W-8** | re-key earns cells by **program identity**: a surviving cover program keeps its cell, a dropped one is retired, and neither touches the wall partition |
| **W-9** | `greedy_cover` is the closure of "same unit serves them", and is **empty with no units** — the basis cannot precede the vocabulary |

Measured at (v=8, s=2, depth=4, m=2, rule_seed=0), all passing: W-1 **1.000**, W-6 true-always
**1.000**, W-4 index 8 → 7 with distinct-program count unchanged.

## Calibration record

Every knob set by a measurement, in the order the measurements forced.

| run | measured | change it forced |
|---|---|---|
| offline (CPU, `wall.py` only) | the **key's demand distribution and information content** at four σ. At σ = 0 the grammar already delivers H = 1.563 of 2.079 nats with mass `[.294 .059 .068 .372 .022 .034 .151 .000]`; `mean_consistent` = 1.85 of 8. At σ = 1.0 one feature takes 53% and **three take zero**, and `mean_consistent` falls to 1.48 | `demand_sigma` = **0.0**. OU concentration starves cells instead of sharpening the question, and the substrate supplies the recurrence for free. One lever |
| `g0` (GPU, full training, 1024 held-out instances) | **Gate 0**, below | `with_noop` = **False** (NOOP changes nothing and is slightly worse for base: 0.5801 vs 0.5596); the design is admissible; one caveat recorded below |

### Gate 0 (`g0`, 324 s)

**Substrate fidelity first.** `e_pol_base` = **0.5596** against `ratchet`'s published base w2 =
0.576 at L2n3, and `e_pol_true_allnodes` = **0.2988** against its `+L2 w1` = **0.293** — the
published ladder reproduced to 0.006 on a different draw of the metering set. The substrate is
the arc's.

| quantity | value | reading |
|---|---|---|
| `mean_consistent` | **1.855 of 8** | a *random* level-2 feature repairs only 23.2% of instances; the key carries ≈ 2.1 bits |
| `frac_uniquely_determined` | 0.216 | the observation pins the key outright on a fifth of instances |
| **linear probe on the agent's own state** | **0.927** (majority 0.374) | see the caveat |
| unkeyed true-table DP recovers `f` | **0.586** | the mechanism that actually consumes the key gets it right 59% of the time |
| ONE ACTION, `e` | 0.3408 → **0.0000** | a perfectly keyed table one-shots this cell *exactly* |
| FULL POLICY, `e` | base 0.5596 · true-unkeyed 0.3203 · **true-KEYED 0.1895** | **a right index is worth +0.1309** (1.69× error reduction), against the arc's ±0.034 stream-position noise floor |
| `floor_base` (exact DP) | 0.4102 | the keyed table beats the base action space's *privileged oracle* |

**The gate's verdict, and its caveat.** The wall is **not** vacuous on the axis that decides the
design: supplying the key is worth 0.131 policy error, ~3.9× the noise floor, and the keyed
one-action ceiling is exactly 0. But the gate also returned something the design did not
anticipate: **the key is linearly decodable from the agent's existing state at 0.927**, so on
this substrate the true key is *already representable* — which is not the regime the idea doc's
§3 scaffold argument describes ("the addresses you can key credit on early are the ones you
already have… a loan against representational machinery that hasn't been built"). Two
qualifications cut the other way and are why the round proceeds:

1. The probe is trained on **16k supervised labels of the latent that the agent never receives**.
   Its only feedback channel is terminal success. So 0.927 measures linear decodability *given
   labels*, not what the agent can extract from its own reward.
2. The mechanism that actually consumes the key recovers it at **0.586**, and the resulting
   policy sits at 0.3203 against the oracle key's 0.1895. The information is in the state and is
   **not being routed into action selection**.

So what this substrate instantiates is the weaker, still-real claim — *a free key that the
credit machinery is not wired to use* — rather than the strong one — *a key that is not yet
representable at all*. **Every index instrument (rotation response, track vs merge, key
cardinality, forced transfer) is untouched by this**; it bears only on how the phase-1
scaffold half is read. Flagged for discussion rather than resolved here.

**One asymmetry introduced deliberately.** `free_selector`'s wall embedding starts at zero and
has `n_grad × max_cycles` ≈ 520 online steps to reach a scale comparable to a root embedding
trained for 12k. At the shared online lr (3e-5) it structurally cannot, so a null on that arm
would measure the optimisation budget rather than the binding. It therefore gets its own param
group at `value_lr_online × wall_lr_mult` (20× → 6e-4). The asymmetry favours the arm detecting
binding, which is the conservative direction for a **null competitor**. If the probe still reads
zero, the lr caveat travels with the number.

## Scope, and what was deliberately left out

Narrowed from `../merge/SPEC.md`[^private] to one lever and one question. **Not
built** (all named as follow-ups, not as oversights): drift-rate sweeps; paced-vs-sudden
rotation (sudden only); contamination arms whose programs read `w`; the loudness sweep over
which frozen features bind; the unmetered dense learner under free i.i.d. variation (that arm is
the SPEC's own addition and needs a different harness). Single seed, one rule draw, one
(v, s, L, m) setting, one damage cell.

## Runs on disk

| tag | what it is |
|---|---|
| `g0` | Gate 0 at full training, 1024 held-out instances: the exact consistent-set size, the linear key probe, and what a right index buys on the DGP's own tables, with and without a NOOP. 324 s |
| `smoke0` | attached `--quick` smoke, 5 arms, 14 cycles, phase1 6 / period 3. 94 s |
| `fw_s0` | the main run: 5 arms, seed 0, 130 cycles, phase 1 = c1–50, rotations at c51/61/…/121 (q returns to identity at c121), `demand_sigma` 0, `mine_decay` 0.95, `cell_min_obs` 8, `merge_persist` 2, **no screen floor** (`screen_min_n` 0, `screen_min_mass` 0). 1013 s |
| `smoke1` | attached `--quick` smoke of the `fw_s1` arms. Found gate W-7. 69 s |
| `fw_s1` | the re-key round: 4 arms, **identical substrate, rotation schedule and instances to `fw_s0`**, plus the re-key op and the carried screen floor (`screen_min_n` 32, `screen_min_mass` 0.02). **The migration gate never opened** — see the machinery note above. 4 arms × 130 cycles |
| `smoke2` | attached `--quick` smoke of the corrected round; both re-key arms adopt. 112 s |
| `fw_s2` | the corrected round: same 4 arms, same substrate/schedule/instances/floors as `fw_s1`, with two changes — the migration gate graded in **consumption policy error** instead of any-entry coverage, and **provisional adoption + a revert net** in place of certify-then-adopt with hysteresis |
| `smoke3` | attached `--quick` smoke of the retire arms; both modes fire, `delete`'s route-to-nothing path exercised, and the **provisional/collapse adoption path fired for the first time anywhere** (earned momentarily worse, so certified could not) |
| `fw_s3` | the retire round: 3 arms (`wall_rekey` re-run **in-tag** as reference after the cross-tag freeze check failed — see the stream-position note above), same substrate/schedule/instances as `fw_s2`, plus `--retire-every 5 --retire-share 0.02 --retire-win 10`. Both retire arms drop all 8 wall cells by c15 |

**`fw_s1` → `fw_s2` diff.** Exactly two knobs' worth of behaviour: `rekey_gate` (new evaluator)
and the adoption/revert rule. `rekey_tol`/`rekey_persist` are retired in favour of
`rekey_margin`, `n_gate`, `collapse_win`, `collapse_delta`, `revert_every`, `revert_margin`,
`revert_persist`, `n_revert`. Everything else — arms, seeds, instances, rotation schedule,
screen floors, mining, recert, merge — is frozen, so `given_key` and `wall_merge` are directly
comparable across `fw_s1` and `fw_s2`.

**`fw_s1` vs `fw_s0` comparability.** Everything the two rounds share is bit-identical by
construction — same seeds, same grammar, same demand, same metering instances, same rotation
schedule. The one difference that touches a *shared* arm is the carried screen floor, so
`fw_s1`'s `wall_merge` is `fw_s0`'s `wall_merge` **plus the floor**; `given_key` is unaffected
(it never merges). To reproduce `fw_s0`'s exact `wall_merge`, pass `--screen-min-n 0
--screen-min-mass 0.0`.

**Checkpoint files.** Mid-run state goes to `<arm>/checkpoint.json`; `<arm>/results.json` is
written **exactly once, at completion**. `fw_s0` wrote both under `results.json`, which made an
external monitor read an arm as finished at its first 5-cycle checkpoint.

**Machinery note from `fw_s0`, for whoever runs this next.** The priced merge audition
(`merge_audit`) **contributed no decisions**: 6 merges fired, **0 were refused**, and all the
filtering was done by the free exact screen plus the hysteresis. Its four policy evaluations per
candidate were spent and never changed an outcome, so `merge_audit=False` is the cheaper default
until a regime is found where the two routes disagree. One merge did pass the audition while
being *worse* by 0.015 (c15: `e_keep` 0.266 vs `e_merge` 0.281), i.e. inside
`merge_audit_margin=0.05` — the margin is doing the admitting, not the evidence.

Modal volume (`rhm-scaling-data`): `/data/rhm_practice_fourwall/<tag>/<arm>/results.json`, with
`setup.json` beside them (config, refs, key mass, and the full rotation schedule). `gate0`
writes `<tag>/gate0.json`. Fetched copies live in `figures/<tag>/`.

## Reproduce

```bash
cd experiments/            # MODAL_PROFILE=chromatic

modal run rhm/practice/fourwall/fourwall.py::selfcheck_remote
python3 rhm/practice/fourwall/launch_detached.py --fn gate0 --tag g0 --n-rt 1024

# fw_s0 -- the deletion-only round (note the two zeros: it predates the screen floor)
python3 rhm/practice/fourwall/launch_detached.py --fn fourwall --tag fw_s0 --seed 0 \
    --arms "given_key,wall_track,wall_merge,unkeyed,free_selector" \
    --era "2:3" --deep-era "3:1" --phase1 50 --rot-period 10 --max-cycles 130 \
    --demand-sigma 0.0 --n-pr 64 --n-rt 384 --n-deep 192 --mine-cap 24 --cell-min-obs 8 \
    --recert-every 5 --n-recert 32 --merge-every 5 --n-screen 32 --merge-persist 2 \
    --screen-min-n 0 --screen-min-mass 0.0 --wall-lr-mult 20 --probe-every 5

# fw_s1 -- the re-key round (same substrate, same instances, plus the carried floor)
python3 rhm/practice/fourwall/launch_detached.py --fn fourwall --tag fw_s1 --seed 0 \
    --arms "given_key,wall_merge,wall_rekey,scratch_rekey" \
    --era "2:3" --deep-era "3:1" --phase1 50 --rot-period 10 --max-cycles 130 \
    --demand-sigma 0.0 --n-pr 64 --n-rt 384 --n-deep 192 --mine-cap 24 --cell-min-obs 8 \
    --recert-every 5 --n-recert 32 --merge-every 5 --n-screen 32 --merge-persist 2 \
    --screen-min-n 32 --screen-min-mass 0.02 \
    --rekey-every 5 --rekey-bank 512 --rekey-min-bank 128 --rekey-min-gain 2 \
    --rekey-steps 400 --rekey-conf 0.5 --rekey-tol 0.02 --rekey-persist 2 \
    --wall-lr-mult 20 --probe-every 5

python3 rhm/practice/fourwall/analyze_fourwall.py --tag fw_s0 --fetch --figures
python3 rhm/practice/fourwall/analyze_fourwall.py --tag fw_s1 --fetch --figures
```

## Figures

| Path | What |
|---|---|
| `figures/<tag>/fig1_competence.png` | Competence per cycle on the fixed held-out set, rotations and the phase boundary marked, with the true-keyed and true-unkeyed oracle levels as horizontal references; beside it the unpriced L3n1 deep probe |
| `figures/<tag>/fig2_index.png` | **The MDL readout**: index classes, stored entries, and distinct stored programs, per arm, over time |
| `figures/<tag>/fig3_transfer.png` | The exact forced-transfer profile at the first rotation: per index class, what its committed table repairs of the demand it *was* serving against the demand it is *handed* |
| `figures/<tag>/fig4_probe_spend.png` | The wall-permutation probe (spontaneous binding) and the priced maintenance layer as a share of priced time |
| `figures/<tag>/fig5_rekey.png` | **`fw_s1`–`fw_s3`.** The index-migration curve (fraction routed by an earned class), the earned basis's ARI against the true `f`-partition, and the migration gate itself (`fw_s1`: coverage-based; `fw_s2`+: consumption policy error under both routings) |

## Seeds

`--seed` sets damage draws, the metering set, mining and probe RNG; `--rule-seed` /
`--train-seed` set the DGP draw and the substrate, defaulted to `ratchet`'s (`0`/`0`/`1`) so the
substrate is the arc's. Single seed by default; absolute error levels are not comparable across
rule draws, so orderings and signs are the reported quantities.

## Inherited, not copied

`../ratchet/` (substrate, `macros.py`, `beam_moves`, `build_ms`, `fit_width`, pricing,
`finetune_generator`, `value_steps`, `plant_probe`, `audition_macro`, `parse_eras`/`era_ctx`),
`../ear/` (`_shared_plus`, `write_results`), `../transpose/drift.py` (`DecayMiner`),
`../setlist/demand.py` (the OU demand primitives, used only when `demand_sigma > 0`),
`../crystallize/units.py` (`build_move_set`, `corrupt_hier`, `grade`, `oracle_rollout`),
and `rhm/rhm_drift.py` (`sample_derivations_weighted`, for its `return_trace`). Nothing in
those files is modified.

[^private]: Not mirrored: this link points to a document in the private lab repo (the roadmap, the queue, an unrun spec, reading notes, or a conversation). See the top-level README for what is held back and why.
