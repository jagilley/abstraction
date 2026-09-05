# solo — file index and design record

**Up**: [`../README.md`](../README.md) (mjc/practice) · **Contract**: [`SPEC.md`](SPEC.md)
**Substrate and machinery donor**: [`../acappella/`](../acappella/FILES.md) — world, piece, reflex
law, nested library, plant audition, the naive delay operator. Donor untouched; `world.py` here is
a verbatim fork plus additions, and gate **S-F1** asserts bit-identity with `acappella/b1`.
**Port donor**: [`../offbook/nets.py`](../offbook/nets.py) (`SlotLayout`, `build_prop`,
`select_slots`, `explore_slots`, `PropTrainer`, `prop_probe`, `build_span`, `SpanBuffer`) —
imported, not forked (it is a pure-numpy module and registers no Modal app; `acappella/world.py`
already imports `SlotLayout` from it).
**Trunk donor**: [`../../ballistic/ballistic_readapt.py`](../../ballistic/ballistic_readapt.py)
(`ballistic_bc`) and [`../../jacobian_teacher/core.py`](../../jacobian_teacher/core.py)
(`clone_policy`) — the behaviour-cloning pattern, with the teacher swapped from a CEM planner
under a forward model to the reflex law's own closed-loop traversals.

**Standing constraint (Jasper, 2026-08-27, inherited verbatim from `acappella/`)**: **no forward
model anywhere in this node, not even as an extra control or reference arm.** Nothing in this
folder imports, trains or evaluates an `f(s,u)`. The behaviour-cloned trunk predicts no state; it
reproduces what the reflex law would *command* from the read it has. If the design ever seems to
demand a model, that is a finding to halt and report, not a gap to patch.

**Writeup**: [`README.md`](README.md) (2026-09-05, after discussion with Jasper). This file is the
factual record and states what each readout measures; the README carries the interpretation.

## Code files

| file | one line |
|---|---|
| `piece.py` | The piece and world as pure-python constants — `acappella/piece.py` forked (no numpy/mujoco/torch, so a Modal *local* entrypoint can import it). Every value is re-asserted against `etude/etude.py`'s own source by gate S-F0. |
| `world.py` | `acappella/world.py` forked verbatim (rollout pool, ledger, `World.traverse`, `reflex_cmd`, `plant_cem`, the deciders, the two-level `Library`, `select_tapes`) **plus** the four things solo adds: `TrunkNet`/`clone_trunk` (the behaviour-cloned reflex), `RolloutPool.run_cl`/`World.rollout_trunk` (a closed-loop grounding), `MemberLibrary`/`build_member_cell` (slots with members), and `SoloDecider` (the seam decision for every arm and every battery row). |
| `solo.py` | The runner: gates S-F0 / S-F1 / S-T0 / S-T1 / S-S / S-P, the donor library, the per-Δ reflex re-fit, the trunk, the treatment library, the eight-arm sweep with online training, the parity gate, the address-book battery and the δ_perf instruments. Modal fn `run_solo`, entrypoint `solo_run` (with `--spawn`). |
| `analyze_solo.py` | Reducer for a run → `results/<tag>/solo_report.txt` (pure `json` + `math`), **plus** the figure job: Modal fn `make_solo_figures`, entrypoint `solo_figs`, and a `--figures` flag that dispatches it and pulls the PNGs into `figures/<tag>/`. Figures are drawn **remotely** because the analysis machine in these sessions has neither numpy nor matplotlib (`offbook/analyze_offbook.py::figs`'s idiom). |

## Modal volume layout

`mujoco-control-data`: `/data/practice_solo/<tag>/solo.json` (+ `done.txt`, `figs/*.png`).
Fetched copies and reports under `results/<tag>/`; figures under `figures/<tag>/`.

## Figures

Regenerated from `results/<tag>/solo.json` alone by `analyze_solo.py --tag <tag> --figures`. Axes
are in the piece's own units: piece error is the mean over segments of the median boundary error at
each waypoint, in arena units, on a piece whose segment leg is 0.8. No panel title states a
reading.

| figure | what it plots |
|---|---|
| `figures/<tag>/fig1_piece_error_over_cycles.png` | Left: piece error over practice cycles at Δ = 8 for the four loop arms (`audit_prop_k`, `route_native`, `audit_prop_kN`, `fid`), against five horizontal references — the reflex through the trunk (0.1869), the PD law it clones (0.1697), `key_seg` (0.1312), `audit_all` (0.1142) and the donor `lib_seg` from `acappella/b1` (0.0838). `audit_prop_kN` and `fid` lie exactly on the `audit_all` reference, which is gate S-P drawn. Right: the same for `prop_k_d0` at Δ = 0, against the Δ = 0 PD law (0.0045), `key_seg` (0.0648) and donor `lib_seg` (0.0392). |
| `figures/<tag>/fig2_trust_by_level.png` | Left: π's proposal mass on the segment slots (solid) and on the primitive (dashed) over cycles, for `audit_prop_k`, `route_native` and `prop_k_d0`, with `fid`'s untrained-π uniform lines (0.800 / 0.100) as the control. Right: π's mass on the poison twin on a log axis (floored at 1e-5 for display), same three arms, with `fid`'s uniform 0.100 as the control. |
| `figures/<tag>/fig3_address_book_battery.png` | The battery as grouped bars per arm — `base` / `no_prim` / `no_table` / `no_table_no_prim` — on a log axis, including `fid`'s untrained-head row as the negative control, with the reflex through the trunk (0.1869) and `audit_all` (0.1142) as horizontal references. |

## Reproduce

```bash
cd experiments/                      # MODAL_PROFILE=chromatic
modal run mjc/practice/solo/solo.py::solo_run --quick --tag ssmoke
modal run --detach mjc/practice/solo/solo.py::solo_run --spawn --tag s0 --seed 0
python3 mjc/practice/solo/analyze_solo.py --tag s0 --fetch
python3 mjc/practice/solo/analyze_solo.py --tag s0 --figures     # regenerates from solo.json
```

## The arms

| arm | Δ | decision rule | the step it isolates |
|---|---|---|---|
| `reflex` | 8 | the reflex law **through the trunk**, closed loop, 136 fb | the reference |
| `key_seg` | 8 | frozen posture key → nearest slot, then nearest member; 0 groundings | committed content, unrouted |
| `audit_all` | 8 | audition every member of every legal slot **+ the primitive** | the enumeration reference |
| `audit_prop_k` | 8 | **Port 1**: π's top-k slots auditioned (+ forced primitive) | routing |
| `route_native` | 8 | **Ports 1 + 2**: an open slot contributes one head emission instead of all its members | + corridor |
| `fid` | 8 | both ports wired and **shut** (heads minted, never trained, never consulted) | must be identical to `audit_all` every cycle |
| `audit_prop_kN` | 8 | π **trained and live** at k = every legal slot | must be identical to `audit_all` every cycle |
| `prop_k_d0` | 0 | `audit_prop_k` at Δ = 0 | the adoption control: does the learner *keep* playing by feel where feel pays? |

**End-of-run battery** on every arm with heads: `base` / `no_prim` / `no_table` /
`no_table_no_prim` / `restored`. `fid`'s **untrained** heads are the negative control.

## Gates

| gate | what it asserts |
|---|---|
| **S-F0** | 13 donor constants read out of `etude/etude.py` with `ast` (never by import). |
| **S-F1** | Cross-tag exact control against `acappella/b1`: the donor library build (4 segment + 3 chain cells) and the **Δ = 0 and Δ = 8** rows for `key_seg` / `key_chain` / `lib_seg` / `lib_all` / `aud_chain` / reflex-at-the-re-fit-gains reproduce b1 at max\|Δ\| = 0, and the per-Δ gain re-fit lands on b1's own (kp, kd). **Declared inapplicable** (never silently "passed") under any config that is not b1's — a smoke has a different library, so the control is not defined there. |
| **S-T0** | The numpy trunk forward the pool workers use is the torch trunk forward (< 1e-4). |
| **S-T1** | **The trunk gate.** The clone reproduces the reflex law's piece error within a pre-fixed band in every cell of {trunk at Δ = 0, trunk at Δ = 8} × {clean, rotated} × {evaluated at Δ = 0, Δ = 8}. **Hard stop on failure** on a real run; explicitly non-binding under `--quick`, where the clone is deliberately undertrained. |
| **S-S** | Seam information under the **delayed** read: per-state oracle gain over the member library, distinct argmins, and what the frozen key actually achieves against it. |
| **S-P** | `fid` ≡ `audit_all` and π-at-k=N ≡ `audit_all`, bit-for-bit, at **every metered cycle**. |
| **S-T2** | *(round 2)* The re-cloned trunk reproduces the reference run's trunk at max\|Δ\| = 0 on every S-T1 cell and on the clone MSE. The probe re-clones rather than reloading (no weights are persisted), so this asserts the re-clone landed where `s0`'s did. **NOT DEFINED**, never silently passed, if the reference run is absent or any key in `S0_KEYS` differs. |
| **S-M** | *(round 2)* The arm that did **not** change — `audit_prop_k` on the `median` filter — reproduces the reference run cycle for cycle, on piece error **and** on π's per-level mass, at max\|Δ\| = 0. This is what makes any difference in a band arm attributable to the filter and nothing else. Same NOT-DEFINED discipline. `S0_KEYS` deliberately excludes `deltas`, `arms`, `n_cycles` and `tag`: arms are independent of one another (every head is minted through `isolated_rng`/`build_span`, every trainer carries its own numpy stream, every per-cycle stream is seeded by the cycle index), so running fewer arms for fewer cycles cannot move an arm both runs share. |

## Decisions taken, with reasons (the things a reader could disagree with)

Each row is a choice the SPEC left open or did not force, with what settled it.

| decision | why |
|---|---|
| **The trunk is cloned from the reflex law at the per-Δ RE-FIT gains** (Δ = 8 → kp 5, kd 4; Δ = 0 → kp 10, kd 2), one trunk per Δ in the ladder, each trained on reads collected at *both* Δ. | One trunk can only clone one law, and the law that matters is the incumbent's — acappella's B1 gave the reflex every advantage by re-fitting its gains per Δ, and a trunk cloned from a handicapped law would make the primitive a strawman. Training each trunk on both Δ's read distribution is what lets gate S-T1 bind at both. |
| **The trunk is FROZEN**: the span head's loss is detached from it, and it never adapts through practice. | Three reasons. (i) `native/`'s encoder was frozen and this node is the port of native. (ii) The trunk *is* the incumbent whose piece error every arm and every battery row is measured against; letting a regression loss move it would confound all of them, and `offbook/`'s `regress` measured exactly that damage. (iii) With the trunk frozen the plant guard becomes a **control** rather than a treatment readout — the `reflex` arm's error is constant by construction — which converts an uncontrolled variable into an assertion. The interference question Port 2 was built to ask in `offbook/` is therefore **deferred, and its absence is stated** rather than silently dropped. |
| **The trunk's output layer is LINEAR, not tanh.** | The teacher's command is already clipped to [-1,1] and `traverse` clips again, so a squashing output would add approximation error exactly where the PD law saturates — which is most of the first half of a segment. |
| **The trunk reads (delayed observation, seam one-hot, phase (h+1)/H) and is never given the schedule's target.** | The target is the thing the clone has to learn; handing it over would make the gate vacuous. |
| **The primitive is auditioned CLOSED-LOOP on a resettable plant copy, with the same observation delay applied INSIDE the trial**, and is always forced into the audition set when legal. | A grounding here is a real trial you perform, and the primitive is a closed-loop controller, so its trial is a closed-loop rollout — no model is involved. The internal delay is not cosmetic: without it the audition would run the primitive under a capability the body does not have and would systematically over-rate it at Δ > 0, biasing `frac_prim` — this node's plainest adoption readout — in the primitive's favour. Forcing it in is `offbook/`'s rule that an untrained logit may never rank the primitive out. **Approximation on the record**: the trial starts from the Δ-old read with its history clamped, so the first Δ steps inside the trial read the start state. |
| **`key_seg` has NO primitive option.** | `offbook/` decision 8: a frozen key has no way to key a closed-loop controller. Said out loud rather than papered over; it is the taxonomy's continuity arm, not a one-variable neighbour of `audit_all`. |
| **The treatment library has slots with MEMBERS** (`n_slot = 8` slots × `m_member = 4`, plus a poison slot), fit once by k-means over CONTENT and frozen — a second library, built from the *same* harvest pools as the donor build. | With acappella's one-tape-per-slot library the span head's target would be constant per slot, so Port 2 could only ever reproduce a stored tape and the within-slot question the SPEC names ("does the conditioned head beat verbatim playback on off-key seams") would have no content. With members, the audition's within-slot argmin is a real state-conditioned choice and the head replaces exactly that (m groundings → 1). k-means over content and frozen is `offbook/` decision 3, for its reason: a re-partitioning library would make π's logits, the head's conditioning and the trust series all measure a moving target. The donor build runs first and untouched, so S-F1's bit-identity is unaffected; the treatment build draws its own candidates from its own rng stream. |
| **Members are drawn uniformly and ranked by their HELD-OUT audition score**, not by their own realised error. | `select_tapes`' own convention, verbatim: ranking by own outcome favours the sequence most finely tuned to its own start state, i.e. the least transferable (étude E-3b). |
| **The poison twin is the `n_poison` WORST-scoring candidates of the same cell**, pinned to the reserved slot id. | `offbook/` harvested its poison from pre-competence cycles; there is no pre-competence phase here (the reflex law is fixed and competent from the first cycle), so the bad-rendition tail is the honest analogue — a genuine measured rendition of the right segment that came out badly, exactly as plausible an address. |
| **Chains are built in the donor library (that is what makes S-F1 exact) and used by NO treatment arm.** | acappella finding 5: on this piece the chain arms decide once at seam 0 from rest and are *exactly* delay-invariant, so no chain number here would be evidence about depth. The SPEC's instruction is to leave the level populated and spend nothing on it. |
| **Parity is `mean over held-out states of 1[e_head ≤ e_verbatim]`, both executed on the plant, open at τ = 0.75**, with the continuous fraction and the mean gap logged per slot per check. | The comparison is the head's single emission against **the audition's own chosen member** at that state — the executor the head is replacing — so there is no arbitrary slack term to choose: the criterion is a pure ordering. τ: `offbook/`'s 0.9 never cleared and was never swept; 0.75 is still a majority-plus criterion on a continuous-command reproduction task, and because the full distribution is on the record the verdict at any other τ is recoverable. The split is `nets._split_code`'s deterministic code of the state (offbook decision 7), never a coin flip. |
| **The head's parity check is charged; the tape side is not.** | The tape side is a deterministic function of (slot, state) and is the reference, i.e. an instrument. The head side is a real self-check the agent performs, and it is charged to its own ledger and reported separately from performance-time cost. |
| **`no_table` = π's argmax names a slot and the head emits its content (or names the primitive and the trunk plays); 0 groundings, 1 fb per seam.** | The purest form of "the address book is gone and π + the head must serve", and the form `native/`'s own no-table row measured. Auditioning head emissions would still have been legal (a grounding needs no table), but it would blend the ablation with the enumeration op. |
| **Non-learning arms (`reflex`, `key_seg`, `audit_all`) are metered ONCE.** | Nothing in them adapts and the world, library and trunk are frozen, so their metering row is constant by construction. `fid` and `audit_prop_kN` still run the full cycle loop, because their whole job is the per-cycle identity assertion. |
| **`fid` mints both heads and never trains or consults them; `audit_prop_kN` trains π and keeps it live at k = N.** | `fid` is then the pure stream-fidelity twin (minting must not perturb the shared stream — `isolated_rng`/`build_span`'s reason for existing) *and* the battery's untrained-head negative control, which is what the SPEC asks it to be. `audit_prop_kN` with a *trained* π is the non-vacuous in-run form of G-P: the no-op at k = N holds against real logits, not only against zero ones. |
| **Every arm draws from the same per-cycle rng streams** (a shared stream for the traversal, dedicated streams for ε-at-the-action and audition-set exploration), seeded identically and never by arm name. | This is what makes the twins exact: arms differ only by their decision rule. `explore_slots` consumes nothing when the selection already covers the legal set, and `select_slots` is an exact no-op at k = n_legal by construction (a stable descending sort truncated at k and re-sorted). |
| **π's targets are graded by the BODY** (per-performer piece error ≤ the batch median on the practice traversal), never by the audition score. | `offbook/`'s licensing condition: selection before regression, with the selection done by the plant. |
| **Exploration is at the ACTION as well as at the audition set** (`eps_act` 0.10, `explore_eps` 0.15, `force_window` 3). | `offbook/` O3's measured lesson: widening what gets *scored* cannot get an option executed if the ranker deciding among the widened set is the thing that is blind; ε-greedy on the launched action is the only channel by which an option the audition ranks badly can enter π's targets at all. |
| **A Δ = 0 control cell is run, but only one arm deep** (`prop_k_d0`), not the whole ladder. | acappella b1 already gives every Δ = 0 committed row, and S-F1 re-derives them here at 0.000e+00; what Δ = 0 adds that nothing else can is the **converse** of the adoption readout — at Δ = 0 the primitive is the *better* option, so `frac_prim` should go the other way. One arm buys that; a full second ladder would double the run for rows already on the record. |
| **The `given` library is NOT built this round.** | An earned-vs-given contrast needs a differently *sourced* library, and the only model-free source on this substrate is the priced plant search that acappella's A-I retired as an incumbent (best cell 0.0607 at 15011 s priced against the reflex's 0.0045 at 16.9 s). Recorded, deliberately not taken. |
| **The can't-decompose readout is NOT reported at segment span.** | At segment span a unit's "spelling" would be the primitive executing the same segment, and there is **one** primitive slot per seam shared by every segment slot there — so π's mass on a unit's own spelling collapses to `mass_prim`, which is already the adoption readout. There is no distinct per-slot spelling to put mass on. Saying so is what the SPEC asks for if the readout has no form here. |
| **The treatment library keeps the best `n_slot × m_member` candidates BEFORE clustering** (the construction audition's own keep), then partitions those by content and repairs any short slot from the unused kept candidates, best score first. | Clustering the whole draw admits into every cell content the construction audition had already rejected, and the `ssize2` probe measured what that costs: `audit_all` over the all-candidate partition reached **0.1196** at Δ = 8 against the donor library's own `lib_seg` **0.0838** on the same pool, with slot sizes as ragged as [4,4,1,1,1,4,2,4,4]. The criterion names only the treatment library's agreement with the donor construction it is meant to be a re-partition of — it is arm-neutral, applied identically to every arm, and was fixed before any arm-vs-arm number existed (`legato/` F2: calibrate to preserve the axis, never to make an arm win). Both constructions are on the record. |
| **`torch.set_num_threads(1)` and single-threaded BLAS, set before torch initialises and before the pool forks.** | Measured, not assumed: every net here is a tiny MLP and the parallelism is the 16-process MuJoCo pool, so torch intra-op threading is pure contention. The identical smoke went **344 s → 79 s** and per-cycle cost fell ~10× (11 s → 0.7–1.3 s). |
| **The trunk is cloned with 20 000 Adam steps on ~130 k samples with a step LR decay (×0.25 at 50% and 80%), hidden 256.** | Set *before* the gate was ever run, so it is a cloning budget rather than a gate tweak. The gate's band is fixed in `cfg` and is not a function of the budget. |
| **Round 2: π's imitation filter is the ONE variable** — `--pi-filter {median,band}` with `--pi-band`. `median` (the default) is s0's `ep <= np.median(ep)`; `band` is `ep <= <fixed error>`. | s0's filter makes π imitate the better *half* of each cycle's own traversals. `offbook/`'s "competence band" turns out to be the same construction (`np.nanmedian(pr["e_piece"])`, `offbook.py:485`), so an **absolute** band is new to this lineage. The reading it tests, stated in advance: under delay the body's grade is only weakly a function of the slot chosen, so a relative filter admits posture luck as competence and π imitates it. The `median` path is asserted byte-identical to s0 by gate S-M, so the band arms differ by the filter alone. |
| **The two bands are 0.1066 and 0.0838, both pre-fixed from published, arm-neutral references on this exact piece at Δ = 8.** | **0.1066** = `ref_play`, étude's `never` at performance tempo (3-seed mean), which `acappella/` already uses blind as its playability guard and which was published before any node in this arc existed. **0.0838** = the donor 8-tape `lib_seg` at Δ = 8 from `acappella/b1`, re-derived at 0.000e+00 by gate S-F1 *inside* `s0` — a competent-**execution** reference on the same piece at the same delay. Neither is chosen by an outcome, and neither is swept. |
| **Band pass fraction and π target-row count are logged every cycle.** | Practice traversals run under motor noise and exploration, so an absolute band can admit little or nothing — **starvation is the failure mode to watch**. If a band admits nothing for several cycles that is a finding to report, not a knob to tune. (The `--quick` smoke already showed the signature: both band arms admitted 0 rows for the first three cycles against the smoke's deliberately poor library.) |
| **δ_perf's benchmark `b(s)` is a per-slot EMA (α = 0.2) of the execution error, and the gate is the body's grade.** | The `bridge` line's form, in the only currency available without a model. It is an instrument this round and is consumed by nothing. |

## Gotchas (inherited, and re-paid here)

- **Never import a sibling runner.** `mjc/shared.py` exposes ONE Modal `app` for the whole
  package. Gate S-F0 reads `etude/etude.py`'s constants with **`ast`**, never by import.
  `offbook/nets.py` *is* importable — it is a pure-numpy module with no `@app` decorators.
- **A Modal *local* entrypoint runs on the client**, which in these sessions has neither numpy,
  torch nor mujoco. `piece.py` is pure python for this reason; `world.py` and `offbook/nets.py`
  are imported only inside the Modal function body; `analyze_solo.py` uses `json` + `math` only.
  **Cost paid here**: the first smoke died locally with `ModuleNotFoundError: numpy` because the
  local entrypoint imported `SlotLayout` to precompute a slot→seam map. It is now computed
  remotely.
- **Launch runners only via `--spawn`.** `modal run --detach <file>::<entrypoint>` does not
  protect the run: a local entrypoint blocking in `.remote()`/`.map()` dies with the client
  (`offbook/FILES.md`; it cost two runs).
- **Entrypoint names must be unique across the package** — `solo_run` / `run_solo`.
- **Rollouts run in a `ProcessPoolExecutor`, never threads** (`acappella/profile_cost.py`: 8
  threads are 3× *slower* than 1, because `_apply_rot_regions` runs per physics substep in Python
  and holds the GIL). The closed-loop worker therefore runs the trunk's forward pass in **numpy**,
  not torch — gate S-T0 asserts the two agree.
- **A local name shadowing a long-lived one across a 400-line function body.** `s1q` crashed with
  `AttributeError: 'numpy.ndarray' object has no attribute 'get'`: the reference run's JSON was
  bound to `ref` before the arm loop, and the δ_perf block 250 lines later rebound `ref` to a
  member's stored trajectory, so gate S-M's `ref.get("arms", {})` met an ndarray. The three uses
  are now `refd` (the reference record), `ref_traj` (the intention reference) and `ref_e` (the twin
  gate's baseline). **What saved the run** was the per-arm `save()`: everything measured was already
  on the volume, so the fix cost a reduction rather than a re-run. Cheap general lesson —
  cross-tag state that is read *after* the arm loops should be named so that nothing inside them
  can shadow it.
- **CPU only.** `cpu=16.0`, no CUDA. π and the span head are small MLPs and the trunk is smaller;
  everything expensive is MuJoCo.


## Gate record (measured, `ssize` / `ssize2` / `ssmoke`, seed 0)

**S-F0**: 13/13 donor constants, 0 mismatched.

**S-F1 — the fork IS `acappella/b1`'s code path.** `max|delta| = 0.000e+00`, `applicable = True`,
`gains_match = True`, on all fourteen checks: the donor library build (4 segment + 3 chain cells)
and the **Δ = 0 and Δ = 8** rows for `key_seg` / `key_chain` / `lib_seg` / `lib_all` / `aud_chain`
and the reflex at the per-Δ re-fit gains. The per-Δ gain re-fit independently rediscovered b1's own
(kp, kd) = (10, 2) at Δ = 0 and (5, 4) at Δ = 8 on the widened grid, both interior to it.

**S-T0**: the numpy trunk forward the pool workers use equals the torch forward at 7.1e-06.

**S-T1 — THE TRUNK GATE PASSES, all eight cells**, against the pre-fixed band
`|e_clone − e_law| ≤ max(0.25·e_law, 0.010)`. Clone command MSE 3.90e-04 (Δ = 8 trunk) and
4.12e-04 (Δ = 0 trunk) on 130 560 samples each.

| trunk | world | eval Δ | e_law | e_clone | diff | band | |
|---|---|---|---|---|---|---|---|
| Δ=8 | rot | 0 | 0.0362 | 0.0358 | −0.0003 | 0.0100 | PASS |
| Δ=8 | rot | 8 | 0.1697 | 0.1869 | +0.0172 | 0.0424 | PASS |
| Δ=8 | clean | 0 | 0.0337 | 0.0334 | −0.0003 | 0.0100 | PASS |
| Δ=8 | clean | 8 | 0.1239 | 0.1219 | −0.0021 | 0.0310 | PASS |
| Δ=0 | rot | 0 | 0.0045 | 0.0044 | −0.0001 | 0.0100 | PASS |
| Δ=0 | rot | 8 | 0.4117 | 0.4142 | +0.0025 | 0.1029 | PASS |
| Δ=0 | clean | 0 | 0.0030 | 0.0030 | −0.0000 | 0.0100 | PASS |
| Δ=0 | clean | 8 | 0.4301 | 0.4343 | +0.0042 | 0.1075 | PASS |

The Δ = 8 trunk's `rot @ Δ=8` cell (+0.0172) is the largest deviation and is the one that matters:
the trunk is the primitive at the operating point, and it plays 0.1869 where the law it clones
plays 0.1697. Reported, inside the band, not tuned.

**S-P**: `fid` ≡ `audit_all` and π-at-k=N ≡ `audit_all` at `max|delta| = 0.000e+00` at every
metered cycle, in every smoke and every probe that ran them.

## Smoke record (what each one caught before the launch)

| tag | config | what it caught |
|---|---|---|
| `ssmoke` (1st) | `--quick`, all 8 arms, 344 s | (i) a Modal *local* entrypoint importing `offbook/nets.py` for a slot→seam map — died on the client with `ModuleNotFoundError: numpy`; the map is now built remotely. (ii) δ_perf counted a slot's **first** launch as "silent" (no benchmark exists yet); first launches are now excluded from both δ and the silence fraction. (iii) A design gap: the span head was trained only in `route_native`, so `no_table` was meaningless for the Port-1-only arms. It is now trained in every learning arm and **consulted** only by `route_native`, which is the variable that arm isolates. |
| `ssize` | full config, 3 cycles, soft trunk gate | Crashed on an instrumentation bug — `row["t_parity"]` read `t_pa` above its assignment, and `row["t_train"]` had drifted below the parity block where it would have silently absorbed parity's cost. All six `t_*` timers re-ordered. **But it delivered S-F1 = 0.000e+00 and S-T1's eight cells at the full cloning budget**, which is why the gate record above exists. |
| `ssmoke` (2nd) | `--quick`, 3 arms, 79 s | Confirmed the timer fix and measured the `set_num_threads(1)` win (344 s → 79 s on the identical config). |
| `ssize2` | full config, 6 arms, 3 cycles, soft gate, 382 s | The sizing table below, and the ragged/degraded member library that motivated the keep-before-cluster change. |
| `ssmoke` (3rd) | `--quick`, 2 arms, 63 s | Confirmed the new library construction: slot sizes exactly even. |

## Sizing (measured, 16 CPUs)

Fixed costs (`ssize`, full trunk budget): donor library **58 s**, per-Δ reflex re-fit **34 s**,
two trunk clones **370 s**, treatment library **3 s**; plus the S-F1 donor rows and S-S.

Per metered cycle (`ssize2`, full config): `fid` **3.7 s** · `audit_prop_kN` **6.6 s** ·
`audit_prop_k` **4.4 s** · `route_native` **4.4 s** · `prop_k_d0` **4.4 s** = **23.5 s/cycle**
across the five arms that run the loop; `reflex` / `key_seg` / `audit_all` are metered once.
At `n_cycles = 120` (offbook's) that is ≈ 2820 s of arm loop on ≈ 580 s of fixed cost, **≈ 57 min**.

## Runs on disk

| tag | what |
|---|---|
| `ssmoke` | `--quick` smokes of the full chain (every gate exercised end to end before each launch). |
| `ssize` | Sizing probe at full config, 3 cycles, **soft trunk gate**; crashed in the arm loop on the timer bug. Its S-F1 and S-T1 numbers are the gate record above. |
| `ssize2` | Sizing probe at full config, 6 arms × 3 cycles, **soft trunk gate**, `--bc-steps 2000` (S-T1 was already validated at the full budget, so the probe did not re-pay it). Complete, 382 s. |
| `s1q` | **(round 2) COMPLETE IN SUBSTANCE, `complete=False` on disk** — every arm, cycle row and battery was committed by the per-arm `save()`, then the runner crashed in the post-arm block (see the Gotcha below) before writing `S_M` / `S_P` / `flags` / `done.txt`. `S_P` is undefined here anyway (no `fid` / `audit_prop_kN` arms); `S_M` is recovered post hoc by `analyze_solo.py::recover_sm` from the two saved records, with the runner's arithmetic verbatim, and is labelled as recovered wherever it is printed. **Not relaunched**: nothing measured was lost. π-filter probe: Δ = 8 only, `--n-cycles 30`, `--ref-tag s0`; arms `reflex` / `key_seg` / `audit_all` (metered once), `audit_prop_k` (the **median twin**, for gate S-M), and `audit_prop_k` / `route_native` at each of the two bands. Battery kept; Δ = 0 and the poison-only extras skipped. |
| `s0` | **The run**: seed 0, 8 arms, `--n-cycles 120`, Δ = 8 with the Δ = 0 adoption control, binding trunk gate. Complete, 3336 s wall on 16 CPUs. Report at `results/s0/solo_report.txt`. |


## `s0` record (seed 0, complete, 3336 s) — numbers only

Interpretation is not written here; it gets discussed with Jasper first. Each block says what the
readout measures and nothing more.

**Every gate passes, all binding.** S-F0 13/13. **S-F1 `max|delta| = 0.000e+00` on all fourteen
checks, `applicable = True`, `gains_match = True`.** S-T0 7.1e-06. **S-T1 8/8 cells inside the
pre-fixed band, `binding = True`.** **S-P: `fid` ≡ `audit_all` and π-at-k=N ≡ `audit_all` at
`0.000e+00` over all 120 metering cycles.**

**The treatment library**: 4 seams × (8 slots + 1 poison) × 4 members, every slot exactly full;
64 candidates scored on 16 held-out states per seam, best 32 kept then partitioned; 7–11 members
per seam placed by the short-slot repair. Poison members score 0.13–1.08 against cell bests of
0.043–0.096.

**S-S, seam information under the delayed read** (per-state oracle over the member library, the
gain being best-single-fixed-slot ÷ per-state argmin): Δ = 8 seams 0–3 = **2.37× / 1.75× / 1.52× /
2.40×**, with 8/9, 5/9, 5/9, 5/9 distinct argmins; the frozen key achieves 0.0383 / 0.0866 /
0.0447 / 0.1235 against oracles of 0.0169 / 0.0262 / 0.0292 / 0.0197. At Δ = 0 the same gains are
2.37× / 1.75× / 1.92× / 2.83×.

**The arms at the last metered cycle** (performance tempo; `fb` = feedback events per traversal):

| arm | Δ | e_piece | fb | ground | priced s | fb-only s | frac_prim | n_aud | n_prop |
|---|---|---|---|---|---|---|---|---|---|
| `reflex` (through the trunk) | 8 | 0.1869 | 136.0 | 0 | 16.9 | 16.9 | — | — | — |
| `key_seg` | 8 | 0.1312 | 4.0 | 0 | 3.7 | 3.7 | 0.00 | 0 | — |
| `audit_all` | 8 | **0.1142** | 8.2 | 148.0 | 139.7 | 4.1 | 0.03 | 37.0 | 9 |
| `fid` (twin) | 8 | 0.1142 | 8.2 | 148.0 | 139.7 | 4.1 | 0.03 | 37.0 | 9 |
| `audit_prop_kN` (twin) | 8 | 0.1142 | 8.2 | 148.0 | 139.7 | 4.1 | 0.03 | 37.0 | 9 |
| `audit_prop_k` | 8 | 0.1212 | 4.0 | 36.0 | 36.6 | 3.7 | 0.00 | 9.0 | 2 |
| `route_native` | 8 | 0.1309 | 4.0 | 26.8 | 28.2 | 3.7 | 0.00 | 6.7 | 2 |
| `prop_k_d0` | 0 | 0.0043 | 138.9 | 36.0 | 50.1 | 17.2 | **0.99** | 9.0 | 2 |

The battery runs one training round further on than these rows (on the final trained state); its
`base` for `audit_prop_k` is 0.1219, for `route_native` 0.1174.

**Adoption (`frac_prim`, the primitive's share of launched decisions at performance tempo).**
Δ = 8: `audit_all` 0.03, `audit_prop_k` 0.00, `route_native` 0.00 (0.05 at cycle 0 for both).
Δ = 0: `prop_k_d0` **0.99**, flat across all 120 cycles.

**Trust (π's proposal mass, cycle 0 → 119).** `audit_prop_k`: seg 0.729 → 0.950, prim 0.239 →
0.025, **poison 0.032 → 0.025**, top1 0.394 → 0.718, entropy 1.477 → 0.743, argmax-prim 0.250 →
0.000. `route_native`: seg 0.729 → 0.977, prim 0.239 → 0.020, **poison 0.032 → 0.003**, top1
0.394 → 0.725, entropy 1.477 → 0.726. `audit_prop_kN`: seg 0.725 → 0.934, prim 0.242 → 0.036,
poison 0.033 → 0.030. `prop_k_d0`: **prim 0.996 → 0.997, seg 0.003, poison 0.000**, top1 0.997,
entropy 0.018, argmax-prim 1.000 at every probe. `fid`'s untrained π is exactly uniform over the
10 legal ids at every probe (mass_seg 0.800, prim 0.100, poison 0.100, entropy 2.303 = ln 10) —
the negative control.

**Poison, behaviourally** (launches of a poison member as a fraction of all tape launches over the
run): `audit_all` 4/124 = 3.23%, `audit_prop_k` 430/15323 = 2.81%, **`route_native` 88/11639 =
0.76%**, `prop_k_d0` 0/120.

**Parity, execution reproduction on the plant** (τ = 0.75, held-out states split by state code;
the head's single emission against the audition's own chosen member, both executed):
`route_native` **7/23** slots open at cycle 119 (ids 0, 18, 19, 21, 22, 24, 37), the series running
5/8 → 2/18 → … → 7/23; `audit_prop_k` 3/22 (instrument only, never consulted); `audit_prop_kN`
8/36; `prop_k_d0` 3/6. Continuous fractions and mean gaps are per slot in the report, so any other
τ is recoverable. Parity costs 934–1078 groundings per check (charged to its own ledger, separate
from performance-time cost). Span-head loss 0.0583 → 0.0026–0.0031; π loss 0.738 → 0.432/0.501.

**The address-book battery** (e_piece; `fid`'s untrained heads are the negative control):

| arm | base | no_prim | no_table | no_table_no_prim | restored ≡ base |
|---|---|---|---|---|---|
| `fid` (untrained heads) | 0.1142 | 0.1142 | **0.7176** | 0.7176 | 0.000e+00 |
| `audit_prop_kN` | 0.1142 | 0.1142 | 0.1026 | 0.1061 | 0.000e+00 |
| `audit_prop_k` | 0.1219 | 0.1224 | 0.1146 | 0.1146 | 0.000e+00 |
| `route_native` | 0.1174 | 0.1174 | 0.1142 | 0.1164 | 0.000e+00 |
| `prop_k_d0` (Δ = 0) | 0.0043 | 0.0516 | 0.0044 | 0.1103 | 0.000e+00 |

`no_table` costs 0 groundings and 4–9.3 fb per traversal in every row.

**δ_perf instruments** (reference = the committed tape's own stored trajectory; consumed by
nothing this round). Mean per-launch execution error against the tape's own trajectory, and mean
δ-silence at the stated threshold (|δ| ≤ 0.05·e, first launch of a slot excluded):
`audit_all` e_step 0.0842 / silence 0.601 · `audit_prop_k` 0.0934 / 0.443 · `route_native`
0.1002 / 0.533 · `prop_k_d0` 0.0265 / 1.000. Per-slot series in the report.

**Wall clock** (16 CPUs): 3336 s total; two trunk clones 377 s; donor library 58 s; per-Δ reflex
re-fit 34 s; treatment library 4 s. Last-cycle per-arm: `audit_prop_k` 4.92 s
(practice 1.24 / meter 2.00 / train 0.38 / parity 1.30), `route_native` 4.82 s.

### Flags carried out of `s0` (not smoothed)

1. **The treatment library is a different object from the donor's and reaches a different place at
   Δ = 8**: `audit_all` over 36 members + the primitive reaches 0.1142 where the donor `lib_seg`
   over 8 tapes reaches 0.0838 on the same pool (S-F1 re-derives that 0.0838 at 0.000e+00 in this
   very run). More candidates auditioned from a Δ-old read is not the same object as fewer, better
   ones — `acappella/` flag 6's mechanism (the audition is the more delay-fragile selector) applies
   at the level of the candidate count. Every within-node comparison is on the member library.
2. **`route_native`'s last-cycle row (0.1309) is above its own battery `base` (0.1174)** because
   the parity set moves between cycles (7/23 open at c119, 9/22 at c95, 3/22 at c105). The arm's
   cycle series is non-monotone in a way the other arms' are not.
3. **The Δ = 0 control arm collapses onto one slot**: `prop_k_d0` launches a tape from exactly one
   slot over the whole run (120 launches, 1 slot) and puts 0.997 of π's mass on the primitive. Its
   `no_table` (0.0044) is therefore a statement about the trunk, not about the corridor.
4. **`n_hold` is uneven across slots** (4 to 64) because a slot only fills its held-out buffer when
   π proposes it; parity for a rarely-proposed slot is read off as few as 4 states (`parity_min`).
5. **The poison slots at seams 1–3 never fill a hold buffer under `route_native`** (they are absent
   from its parity table), so their parity is undefined rather than failed.
6. **The reflex reference here is the trunk (0.1869), not the PD law (0.1697)** — the +0.0172 of
   S-T1's binding cell. Every arm's margin against "the reflex" is against the clone.


## `s1q` record (round 2, π's imitation filter, seed 0, 30 cycles, 1035 s)

Numbers only. The one variable is π's imitation filter; everything else is `s0`'s.

**Gates.** S-F0 13/13 · **S-F1 `0.000e+00`** on all fourteen checks, applicable · S-T0 4.7e-06 ·
**S-T1 4/4 cells** inside the band, binding (the Δ = 8 trunk only; Δ = 0 was not cloned this round)
· **S-T2 `0.000e+00`** — the re-cloned trunk is bit-for-bit `s0`'s, on every S-T1 cell *and* on the
clone MSE · **S-M `0.000e+00`** on piece error over all 30 cycles **and** on π's per-level mass over
all 6 probes (recovered post hoc; see the run row above). S-P is not defined here.

**The arms at the last metered cycle** (Δ = 8; `reflex` / `key_seg` / `audit_all` metered once and
identical to `s0`'s):

| arm | filter | e_piece | fb | ground | frac_prim | n_aud |
|---|---|---|---|---|---|---|
| `reflex` (through the trunk) | — | 0.1869 | 136.0 | 0 | — | — |
| `key_seg` | — | 0.1312 | 4.0 | 0 | 0.00 | 0 |
| `audit_all` | — | 0.1142 | 8.2 | 148.0 | 0.03 | 37.0 |
| `audit_prop_k` (median twin) | median | 0.1221 | 4.0 | 36.0 | 0.00 | 9.0 |
| `audit_prop_k_ref` | band 0.1066 | 0.1138 | 4.0 | 36.0 | 0.00 | 9.0 |
| `route_native_ref` | band 0.1066 | 0.1176 | 7.2 | 27.0 | 0.02 | 6.8 |
| `audit_prop_k_lib` | band 0.0838 | 0.0955 | 10.4 | 36.0 | 0.05 | 9.0 |
| `route_native_lib` | band 0.0838 | 0.0961 | 10.4 | 27.0 | 0.05 | 6.8 |

**What each filter admitted** (`pass` = fraction of the 16-performer practice batch at or under the
filter; `rows` = π target rows added):

| arm | filter | mean pass | total target rows | cycles admitting NOTHING |
|---|---|---|---|---|
| `audit_prop_k` | median | 0.500 | 960 | **0/30** |
| `audit_prop_k_ref` | band 0.1066 | 0.065 | 124 | **8/30** |
| `route_native_ref` | band 0.1066 | 0.048 | 92 | **15/30** |
| `audit_prop_k_lib` | band 0.0838 | 0.013 | 24 | **24/30** |
| `route_native_lib` | band 0.0838 | 0.015 | 28 | **23/30** |

Per-cycle pass fractions are 0.00 / 0.06 / 0.12 throughout for every band arm — i.e. 0, 1 or 2 of
16 practice traversals.

**The early hump.** `audit_prop_k` (median) reproduces `s0` exactly: 0.0955 at c0 → 0.1498 by c3,
still 0.1474 at c18, 0.1221 at c29. `audit_prop_k_ref` goes 0.0955 → 0.1140 by c3 and stays in
0.1103–0.1140 for the rest of the run. `audit_prop_k_lib` sits at **exactly 0.0955** — the c0 value
— at every one of the 30 cycles. `route_native_ref` 0.1117–0.1176; `route_native_lib` 0.0912–0.1091,
final 0.0961, `frac_prim` 0.00–0.14.

**Trust** (c0 → c29). `audit_prop_k` (median): seg 0.729 → 0.959, prim 0.239 → 0.024, poison
0.032 → 0.017, entropy 1.477 → 0.903. `audit_prop_k_ref`: seg 0.996 → 0.923, prim 0.002 → 0.077,
**poison 0.002 → 0.000**, entropy 0.574 → 0.476. `route_native_ref`: seg 0.996 → 0.962, poison
0.002 → 0.000. `audit_prop_k_lib`: the untrained uniform (seg 0.800 / prim 0.100 / poison 0.100,
entropy 2.303) is held through the first probes — π has had no targets — then seg 0.947, prim
0.053, poison 0.000, entropy 0.259. `route_native_lib` likewise, reaching top1 0.988 by its second
probe.

**Battery** (e_piece; `restored ≡ base` at 0.000e+00 in every arm):

| arm | base | no_prim | no_table | no_table_no_prim |
|---|---|---|---|---|
| `audit_prop_k` (median) | 0.1209 | 0.1209 | 0.0969 | 0.0969 |
| `audit_prop_k_ref` | 0.1138 | 0.1138 | 0.1074 | 0.1091 |
| `route_native_ref` | 0.1271 | 0.1271 | 0.1132 | 0.1132 |
| `audit_prop_k_lib` | 0.0955 | 0.0919 | 0.0747 | 0.0747 |
| `route_native_lib` | 0.1041 | 0.0999 | 0.1019 | 0.1019 |

### Flags carried out of `s1q` (not smoothed)

1. **Both bands starve, the tighter one almost completely.** At 0.0838, 24 of 30 cycles admit
   nothing and the whole run supplies 24 target rows against the median filter's 960 — a 40×
   difference in π's training signal. Any comparison between a band arm and the median arm is
   therefore confounded with the amount of imitation, not only its selectivity.
2. **`audit_prop_k_lib`'s piece error is exactly its cycle-0 value at every cycle**, consistent with
   π having been moved too little to change any launch. Its `no_table` (0.0747) is read off a head
   trained from 24 rows.
3. **The band is applied to a practice traversal run under motor noise and exploration**
   (`explore_sigma` 0.25, `eps_act` 0.10, `explore_eps` 0.15), so it is not the same quantity as the
   performance-tempo piece error the bands were taken from. This was known before the probe and is
   why the pass fraction is logged; it is the first thing to change if the probe is repeated.
4. **The two `_ref` arms differ in how often they starve** (8/30 vs 15/30) despite sharing a band,
   because their practice traversals diverge once Port 2 opens slots.
5. **The probe is 30 cycles**; `s0`'s median arm was still descending at c29 (0.1221) toward its
   c119 value (0.1212), so no band-vs-median comparison here is a statement about the settled value.
