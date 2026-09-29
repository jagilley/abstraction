# DESIGN — `rubato`: the loop, resumable

**Brief**: [`SPEC.md`](SPEC.md) (the orchestrator's, verbatim) · **Machinery record**: [`FILES.md`](FILES.md) ·
**Donor**: [`../sostenuto/sostenuto.py`](../sostenuto/sostenuto.py) at its `st2_s1`/`st2_s2` head ·
**Why**: [`../sostenuto/DESIGN.md`](../sostenuto/DESIGN.md) §9.8, `../sostenuto/CONVERSATION.md`[^private] §6,
and the standing rule in [`experiments/CLAUDE.md`](../../../../../../CLAUDE.md) ("Long loops must be resumable").

---

## §0 What this node adds, in one paragraph

An arm of this loop is ~201 cycles and ~2.3 GPU-hours, and until now it had no resume path
(`checkpoint_every` writes the results log only), so every retroactive change to a gate that takes
effect at cycle 60 has been a full re-run. `rubato.py` is `sostenuto.py` plus a saved state: at every
era boundary and at a fixed cadence the whole state the loop carries across cycles is written to the
run's volume directory as one object graph with a hash tree beside it, and an arm can be started
from any saved state in a fresh container and continued. With every knob off the file is
`sostenuto.py` (G-F). Saved at c, restored in a fresh container and continued to 2c, the arm is the
uninterrupted one in every logged quantity, every table and every weight (R-1). The ungated
admission arm `st_gn_yk` was re-run with saved states on at both seeds, bit-identical to its banked
mirrors, so the next round's admit-then-grade op starts from those states rather than from cycle one.
Phase 2 (§10, results §4.5–§4.8) built that op on the saved states, admit-then-grade, as two arms per
seed: the read grade at no bill and the world grade, billed. Each is `st_gn_yk` by hash until its
first revocation. Round 3 (§11) re-ran the read arm at seed 0 with one knob, a per-span diet, after the
seed-0 records located its wrong revocations at spans the readout had never been fit on.

---

## §1 Decisions

### §1.1 One object graph, rebound in one statement

`run_arm` keeps its state in ~75 local variables whose objects reference one another: the plant is
inside the optimizer (`gopt`), the projection bank (`vo_om.trunk`) and the shaper; the span head, the
critic and the shaping head are parameter groups of `gopt`; the slots are inside the executor and
each slot is bound to a macro move of `ms`; the quotient is inside every miner; the critic and the
shaper are inside the objective handle `vo_obj`, whose two lambdas close over `vo_rec` and `quot`.
Saving organ by organ (a state dict each) and re-wiring them by hand would be a second copy of this
wiring, and a place for the two to drift. So every state name is pickled in ONE call — pickle's memo
keeps every shared reference — and the restore rebinds every name to the restored graph in ONE
tuple assignment. The closures `run_arm` defines are rebuilt at setup as always and close over the
names' cells, so they see the restored objects the moment the names are rebound.

The alternative I rejected: frame hacks (`PyFrame_LocalsToFast`) to rebind locals dynamically. The
image is Python 3.11, where that works, but the explicit statement is readable, and gate R-0 checks
it against the list.

### §1.2 What is state, what is a constant, what is a temporary — and gate R-0

Every name `run_arm` binds before its main loop is in exactly one of five lists
(`_RB_STATE`, `_RB_CONST`, `_RB_SETUP_TEMP`, `_RB_OWN`, `_RB_ARGS`) or is a nested def; every name
first bound inside the loop is a per-cycle temporary. Gate **R-0** is static (it reads the source's
AST, no GPU) and asserts: (a) the restore statement rebinds exactly `_RB_STATE`, in order; (b) no name
bound at setup is unlisted; (c) the lists are disjoint; (d) every state name is actually assigned;
(e) no name first bound in the loop is read before it is bound in source order (a temporary that
leaks across cycles would be state nobody saves); (f) after the save, the cycle body only decides
whether to break. Four perturbations, each RED: a stream dropped from the state list, an unnamed
object added at setup, a loop temporary carried across cycles, a save moved above the cycle's last
write. R-0 is what makes the list a checked object rather than a transcription, and it is what a
phase-2 edit that adds an organ trips first.

**The state** (`_RB_STATE`, 75 names): the plant, `gopt`, the value head and `vopt`; the span head,
its slots, slot and gate events and its stream; the meter, the perf gain's running normaliser and
records; the executor (its buffers, its credit column, its own streams); the recorder (per-slot
record buffers, probe buffers, the governed set, its CUDA `torch.Generator` and three numpy streams);
the critic; the projection bank (every row, its fitted readout, its streams); the push counters and
the token-class cache; the quotient; the merge events; the objective handle; the committed flags,
every committable miner, the G-Y miner, the gauge history, the observation panel and its histories;
the shaper (its head, its stream, its state); `_sb_ctx` (the yield currency's target, the label
counters, the world-model plan, the era dumps' records); the operative move set `ms`; the port, its
optimizer, streams and buffer, `move_age`, the filter flag; the value buffer; `t_cum`; `events`;
`log`; the question port's stream and ledger; the demand cache; the shadow certificates; the dsil
detector and its events; both loop policies (the yoke is by CLOCK — `YokePolicy` holds the plan and
answers by cycle number — so the yoke plan's position IS `cyc`); the read ledger (the bill);
`loop_actions`; the panel history; and the clock: `cyc`, `era_i`, `era`, `active`, `cert`,
`era_start`, `ec`, `era_end`, `c_in_era`, and `era_last` (whether the saved cycle closed its era,
which decides where the restored run re-enters). Plus, beside the locals: the module's three recorder
globals (`_ENTRY_REC`, `_SLOT_REC`, `_EXP_REC`, the last of which carries the oracle-read count and
the token-mask cache across cycles) and the four global streams (torch CPU, every CUDA device,
numpy's legacy global, Python's `random`).

**The admission set and everything a gate op needs** are in that list by construction: `_adm` holds
the admitted and rejected key sets per level, the `walked` flags, every counter, the confusion, and
`passes` — every pass's per-level row with its per-candidate `steps` (both decisions, the world's
error, the read's difference, the entry histogram); `obs_miners` is the observation panel; `vo_om`
is the readout bank with its rows and fitted weights.

### §1.3 The world is rebuilt and ASSERTED, not saved

The constants (`_RB_CONST`: the config, the spec, the rules, the substrate `shared`, the metering and
shadow pools, ...) are recomputed at setup exactly as the saved run computed them and are not in the
file. Anything in the graph that refers to one of them is written as a TOKEN (`rb_registry`: the
arguments, every constant, every value of `shared` two levels deep, every element of the ladder,
every closure, the objective handle's lambdas) and resolved against the fresh container's own object.
Two checks keep this honest at every restore: the config must equal the saved run's except for the
checkpoint's own `rb_*` knobs and whatever `rb_allow_cfg` names, and the WORLD (`_RB_WORLD`: the
substrate, the ladder, the refs, the rules, the pools) is content-hashed at every save and must hash
the same in the fresh container. A substrate that trained differently in the second container
refuses the resume instead of silently forking it. (It never has: the fingerprint matched on every
resume in this node, which is itself the evidence that `_spiral_shared` is bit-deterministic across
containers.)

### §1.4 The file and the digest

`c{NNN}.pt` is this node's own container, not `torch.save`: a pickle whose `persistent_id` writes
tokens, local classes (every `nn.Module` built inside a factory — the plant's `BlockInfiller`, the
heads, the critic, the shaping head, the executor's `PerfExecutor` — as its qualname, resolved against
the fresh container's own instances) and CUDA `torch.Generator`s (device and state bytes), and writes
tensor storages ONCE each, keyed exactly as `torch.save` keys them, so views and parameters stay
shared. Integer storages (the token buffers: int64 over an alphabet of eight) are zlib-compressed,
lossless; float storages are raw. `c{NNN}.json` beside it holds the metadata (the config, the spec,
the world fingerprint, the digest) and the hash TREE of the whole state: floats by their bits,
tensors by dtype, shape, device type and bytes, sets order-free, dicts in insertion order, objects
through `__getstate__`, optimizers through theirs, tokens as tokens, three levels deep so a
difference can be localised. The loop's two wall-clock fields (`t_s` on the walk's pass records and
on the world-model diagnostics) are left out of the digest (§5.2). The rule is VERSIONED and each
save records its version (`digest_v`), because a refinement of the rule must not make a state saved
under the old one unreadable (§5.4).

Gate **R-2** runs inside every restore: the restored graph is re-hashed against the fresh
container's registry and asserted equal to the saved tree, name by name. A round trip that lost
anything fails at the resume, not sixty cycles later.

### §1.5 Where the save sits, and where a restored run re-enters

The save is the cycle's last statement before the break decisions: after every log row, the
`checkpoint_every` results write and the era-boundary head dumps. A restored run rebinds the state
before the main loop and then either re-enters the saved era's inner loop without running the
era preamble (`_rb_mid`, when the saved cycle did not close its era) or runs the next era's preamble
as the uninterrupted run would (when it did — the preamble re-arms both loop policies, so skipping it
there would be wrong). The global streams are set LAST, after everything else in the restore.

The loop head's `while era_i + 1 < len(eras)` became
`while not _rb_stop and (_rb_mid or era_i + 1 < len(eras))` and the era preamble moved under an
`else:`; with the knobs off both flags are False and this is the donor's control flow, which G-F
checks.

### §1.6 The generators, and one correction to the brief

The brief lists "the module-level `torch.Generator`s seeded at lines around 2795, 5102, 6362, 6792".
Read at those lines they are not module-level and carry nothing across cycles: 2795 is
`train_reader`'s batch generator (setup only, seeded from its argument on every call), 5102 and 6362
are the E-1 / E-7 recorder checks' fixed generators (pre-setup gates, seeded 11 on every call), and
6792 is `build_critic`'s initialisation generator (seeded from its argument, used once at mint).
The streams that DO carry state across cycles are: the global torch CPU and CUDA streams (the plant's
masking draws on both every cycle), the per-arm numpy streams `rng`, `grng`, `ggrng`, `span_rng`,
`prng`, `erng`, `qrng`, and the organs' own streams — the executor's `rng` and `rrng`, the
recorder's CUDA `gen` and its `crng`, `prng`, `srng`, the bank's `rng`/`mrng`/`frng`, the shaper's
`rng`. All of them are in the graph (§1.2); none of them needed special handling beyond the CUDA
generator's persistent id. Neither Python's `random` nor numpy's legacy global stream is drawn by the
loop; both are saved anyway.

### §1.7 What is deliberately NOT saved

`.grad` on parameters: every trainer in the file calls `zero_grad(set_to_none=True)` before its
backward, so gradients are dead at a cycle boundary. The per-cycle temporaries (R-0 (e) is the check
that they are temporaries). The substrate and pools (§1.3). The saved states are also not copied by
`sweep`'s merge into the merged tag (they stay where their container wrote them, and every saved
path in the arm file points there).

### §1.8 The fork contract for the next round

A gate op that changes the arm's behaviour from cycle c is run as a NEW arm restored from
`st_gn_yk`'s saved state at c, with its new knob named in `rb_allow_cfg`. The restore allows exactly
that: the config may differ on the named keys; the arm spec must be the saved arm's outside its
`cfg`; the world must hash the same. Two mechanisms make a fork's own state work: (1) SCHEMA
EVOLUTION — a state name, a dict key of a state dict, or an attribute of a state object that the
fork's setup creates and the saved state lacks starts at its fresh value, and every such addition is
recorded in the arm file (`rb.resumed_from.schema_added`); (2) the `_adm` OVERLAY — `_adm` carries
the arm's own gate config (`gate`, `r2`, `delta`, ...) beside its state, so on restore those keys take
the fork's fresh values (recorded in `rb.resumed_from.overlay`) while the decided keys, the counters
and every per-trial record stay the saved run's. The restored `log` and `events` keep the source
arm's label on records made before the fork, which is their provenance.

---

## §2 The knobs

| knob | level | default | meaning |
|---|---|---|---|
| `rb_ck_every` | run | `0` | save at every cycle divisible by this (the arms of record: 10) |
| `rb_ck_era` | run | `False` | save at every era boundary (the arms of record: on) |
| `rb_stop_cycle` | run | `0` | stop the arm after this cycle, saving there; the arm-level battery, dumps and results then run on that state |
| `rb_resume` | run | `""` | a saved state (relative to `rhm_practice_rubato/`, or absolute) to restore before the loop |
| `rb_allow_cfg` | run | `""` | comma-separated config keys a fork may change against the saved run's |
| `rb_falsify` | run | `""` | gate R-1's red runs only: `rng:<stream>` or `local:<name>` left unrestored (and out of R-2) |

| `rb_ref_ck` | run | `""` | [phase 2] a reference arm's saved-state directory for gate G-I (§10.7) |
| `rb_auto_resume` | run | `False` | resume a restarted arm from its own latest complete save (§5.5); `rb_resume` may name `{arm}` |
| `adm_grade` | arm | absent | [phase 2] `read` / `world` (§10) |
| `adm_grade_obs`, `adm_grade_margin`, `adm_grade_cons`, `adm_grade_falsify` | run | 160, 0.0, 4, `""` | [phase 2] the window, the dial, consumers fired per key, the preflight's red runs |
| `adm_grade_diet` | arm | `0` (512 on `rb_grd_yk`) | [r3] the read grade's per-span diet (§11) |

All are in `voicing_run`'s signature (the arm-level ones also in the arm's spec) and pass through
`sweep`'s `--flags-json`. FILES.md has the same table with each knob's full meaning.

---

## §3 The gates

| gate | where | claim | falsified by |
|---|---|---|---|
| **G-F** | `fidelity_smoke` | with every `[rubato]` knob off this file replays `sostenuto.py` bit for bit, against a donor self-replay control | — (the lineage's gate; its falsifications are the donors') |
| **R-0** | `rb_static_gate`, local or `rb_gates` | §1.2 (a)–(f) | four perturbations, 4/4 RED |
| **R-2** | inside every restore | the restored graph re-hashes to the saved tree, name by name; the config matches outside the allowed keys; the world fingerprint matches | — (asserted on every resume) |
| **R-1** | `rb_gate_r1` | saved at c, restored in a FRESH container (`single_use_containers`), continued to the stop: the whole state's hash tree and the whole arm file equal the uninterrupted run's | a generator deliberately unrestored (torch CPU; `grng`) |
| **bank** | `reduce_rubato.py --bank-check` | the re-run `st_gn_yk` with saved states ON is its banked mirror, bit for bit, outside the `rb_*` knobs, the `rb` record and wall-clock fields | — |
| **G-D, G-O** | `rb_grade_gates`, CPU | [phase 2] the grade's rule and the re-offer rule on designed cases | 5 perturbations, 5/5 RED |
| **G-I, G-W, G-S** | in-run | [phase 2] the graded arm IS the reference until its first revocation; the world grade's fire IS the donor's audition; a revoked key is served nowhere (§10.7) | `touch`, `node`, `serve`, 3/3 RED (§10.8) |
| **G-N** | `rb_grade_gates`, CPU | [r3] the per-span diet (§11) | 3 perturbations, 3/3 RED |

---

## §4 Results

### §4.1 Gate R-1 (`rb_r1`, `figures/rb_r1_r1.txt`)

The preflight substrate, twin `st_pf_gn`, every call in its own container. A ran uninterrupted to
c24, saving every cycle. B1 restored A's c12 (an era boundary) in a fresh container and continued
to c24; B2 did the same from A's c14 (mid-era). F1 and F2 are B1 with the torch CPU stream, and
then `grng`, deliberately left unrestored.

| run | state at c24 vs A | arm file vs A | verdict |
|---|---|---|---|
| B1 (c12, era boundary) | identical: every table, weight, optimizer moment, buffer, stream | identical | PASS |
| B2 (c14, mid-era) | identical except `log/vo_om` | one float: `vo_om[19].w_norm` 4.9126548857360515 vs …052 (one ulp) | the host floor, §4.2 |
| F1 (`rng:torch_cpu` unrestored) | 40 names differ | diverges at c13 (plant weights, admission records, `vloss`) | RED |
| F2 (`local:grng` unrestored) | 40 names differ | diverges at c13 | RED |
| A2 (a second uninterrupted A) | Python's `random` (never seeded, never drawn by the loop; saved and restored anyway) and the `rb_*` knobs in the recorder's config copy (since left out of the digest) | identical in every logged quantity | the floor |

The donor `sostenuto.run_arm` on A2's substrate equals the fork on all 13 fidelity series, commits
equal. Saves at preflight: 21 MB (c1) to 152 MB (c24), 7–63 s each (the preflight's entry log is
what grows).

### §4.2 The cross-container floor is a host type

`rb_host_survey` over 12 fresh L4 containers: 9 landed on AVX-512 hosts, where numpy 1.26
dispatches `log`/`exp` to SVML (`np.log` one ulp off libm on the probe value,
`-0x1.3a68f009b1576p+2` vs `…1577p+2`), and 3 on hosts without AVX-512. B2 logged the libm value;
A, A2 and B1 the SVML one. The value that differed is a degenerate readout fit whose only weight is
`np.log(q / (1 - q))`, which reached one logged diagnostic and no decision. With
`NPY_DISABLE_CPU_FEATURES` set to the AVX-512 features (`rb_host_probe_pinned`), all 12 hosts give
identical scalar and vector `log`/`exp` bits (`results/host_survey_pinned.log`). The flag is NOT
set: it would change numerics against the banked arms, which ran on whichever host type they drew.
Every `rubato` run records its host (`rb_host_info`). So, precisely: bit-exact across containers of
the same host type; across the two types numpy's transcendentals differ by one ulp. This is why
the phase-2 preflight runs all its arms in one container (§10.8).

### §4.3 The seed-0 reference re-run (`rb_s1`, `figures/rb_s1_bank_check.txt`)

`st_gn_yk`, seed 0, a save every 10 cycles and at every era boundary, yoked to `vo_s3:voi3_dp`:
complete, 201 cycles, 6625 s (1.84 GPU-h), peak RSS 7062 MiB. Against the banked
`sostenuto/figures/st_s1/st_gn_yk`: all 13 fidelity series and every other log series equal over
all 201 cycles (max |diff| 0.0); the entry record, every event, every table, the battery, the bill,
the admission set's decisions and passes equal. The re-run carries 14 keys the bank does not, all
`sostenuto` round 2's bookkeeping (`adm_delta`, `adm_demand_*` in the config; `r2`, `delta`, `thr`,
`floor`, `n_silent`, `nxt_src`, `exports`, `strong_own` … in `_adm`), which round 2 added after
`st_s1` was banked on the round-1 binary. No shared key differs.

Saved: 22 states (every 10th cycle to c200, plus the era boundaries c192 and c201), 19 MB (c10)
to 89 MB (c201), 1.24 GB in all. Storages grow 150 to 594 MB raw and pack to 19 to 66 MB (lossless
zlib on the int64 token buffers); the pickled records grow to 23 MB. A save takes a median 10 s
(max 16 s); all 22 together 199 s, 3% of the arm.

### §4.4 The seed-2 reference re-run (`rb_s2`, `figures/rb_s2_bank_check.txt`)

`st_gn_yk`, seed 2, the same knobs, yoked to `vo_s3d2:voi3_dp`: preempted once at c60 and
restarted by Modal from c1 (§5.5); the run of record is the restarted one, complete, 201 cycles,
8393 s (2.33 GPU-h, 38.3 s a cycle), peak RSS 7033 MiB. Against the banked
`sostenuto/figures/st_s2/st_gn_yk`: all 13 fidelity series and every other log series equal over
all 201 cycles (max |diff| 0.0), the entry record equal, every shared key equal. Keys only in the
re-run: the same 14 of `sostenuto` round 2, and, because `rb_s2` ran on the phase-2 binary with
the grade OFF, the four run-level grade knobs in its config (`adm_grade_obs` 160,
`adm_grade_margin` 0.0, `adm_grade_cons` 4, `adm_grade_falsify` ''); the arm's own `adm_grade` is
absent and `_adm` carries no grade key. So it is also a full-scale check that phase 2 is inert
with the grade off. Saved: 22 states at the same cycles as `rb_s1`, 19.0 to 88.6 MB, 1.24 GB in
all; a save a median 13.4 s (max 22.3 s), 287 s in all. Its states are digest v2 with no recorded
version (§5.4).

### §4.5 Phase 2, seed 0 (`rb_g1`, `figures/rb_g1_reduction.txt` [L], `figures/rb_grade_seedtable.txt`)

Both arms complete, 201 cycles, restored from `rb_s1`'s c20, three container segments (§5.5);
clock-yoked, so the commits (L2 c48, L3 c100, L4 c151, L5 c186) and the advances are the ungated
arm's on both.

The identity checks:

| check | `rb_gw_yk` (world) | `rb_gr_yk` (read) |
|---|---|---|
| R-2 at the restore from `rb_s1` c20 | PASS | PASS |
| R-2 at the resume from its own c130 | PASS (digest = the c130 save's) | PASS (digest = the c130 save's) |
| first revocation | c30 | c55 |
| G-I by hash, saves before it | none (the first save is c30) | equal at c30, c40, c50 |
| G-I off the log: first cycle any series differs from `st_gn_yk` | c30 (`quot`), = the first revocation | c55 (`quot`), = the first revocation |
| G-I off the records (events, merge events, label dropped) before it | 19 = 19, 14 = 14, none differ | 38 = 38, 32 = 32, none differ |
| G-W (first grade-changing fire vs the donor's audition) | c25, L3, e 1.0 -> 0.640625 = audition, 0 instances differ | the same fire, the same result |
| G-S (a revoked key served) | never red | never red |

The grade (margin 0.0, window 160, 4 consumers):

| | `rb_gw_yk` | `rb_gr_yk` |
|---|---|---|
| key-passes graded / kept | 683 / 669 | 316 / 291 |
| revocations (no consumer, worthless) | 14 (8, 6) | 25 (1, 24) |
| re-offers | 6 | 14 |
| revoked at the end, L2 / L3 / L4 | 3 / 4 / 1 | 11 / 0 / 0 |
| distinct keys ever revoked, per level | L2 3, L3 5, L4 4 | L2 14, L3 1 |
| fires / world queries billed to the grade | 2153 / 413,376 | 679 / 0 |
| this decision vs the other currency's on the same fires (KK/KR/RK/RR) | L2 366/60/5/0, L3 178/31/0/1, L4 25/0/0/0 | L2 116/4/21/3, L3 54/0/0/0, L4 9/0/0/0 |
| gain quantiles 10/25/50/75/90 %, world | +0.16 +0.23 +0.32 +0.45 +0.64 (n 666) | +0.20 +0.27 +0.40 +0.47 +0.80 (n 271) |
| gain quantiles, read | -0.01 +0.01 +0.12 +0.20 +0.32 | 0.00 0.00 +0.09 +0.22 +0.42 |
| of `st_gn_yk`'s walked L3 keys, a half held revoked at the end | 4/19 | 18/19 |
| of its L4 keys | 5/9 | 0/9 |
| L3 rows admitted/live, L5 rows, at the last pass | 62/65, 80/80 | 9/9, 1/1 (`st_gn_yk`: 71/71, 88/88) |

The read arm's 24 "worthless" revocations are at L2 (c55–c197): on every one the read's level on
the fired states was BELOW the unfired pool's (read gain -0.0007 to -0.10), while the world's
success on the same fires rose on 21 of them (+0.04 to +0.40; 0.0 on three).

Mean task error per era (seedtable (a)(ii)), against the ungated arm `st_gn_yk`:

| | era1 | era2 | era3 | era4 | era5 |
|---|---|---|---|---|---|
| `rb_gw_yk` − `st_gn_yk` | -0.0036 | -0.0019 | +0.0347 | +0.1762 | +0.0532 |
| `rb_gr_yk` − `st_gn_yk` | -0.0009 | -0.0093 | +0.0770 | +0.1903 | +0.2390 |

Cost: 0.38 GPU-h refused launch (§5.4), ~2.0 GPU-h segment 2 (c21–c137/140), ~0.4 GPU-h the
preemption's restart (stopped), 2.62 GPU-h segment 3 (c131–c201: 1.15 world, 1.47 read) — ~5.4
GPU-h against the ~4.2 stated. Saved: every 10th cycle c30–c200 and c192, c201, per arm, in
`rhm_practice_rubato/rb_g1__<arm>/<arm>/ck/`.


### §4.6 Seed 0's grade, re-read off its own records (`figures/rb_g1_reduction.txt` [M]; zero GPU)

Four readings the coordinator asked for, all off the arm files, the saved states read on CPU
(`rb_peek_state` for the readout's row buffer by span, `figures/rb_g1_diet.json`;
`rb_consumer_survey --which revoked` for every revoked key at every later save,
`figures/rb_g1_revoked_survey.json`) and the setup's true L2 table (analysis only).

**M-1 The dial.** Keep iff the best consumer's gain exceeds the margin, re-applied to every recorded
graded key-pass. The other currency's margin-0 decision on the same fires is the split. This is
counterfactual on the recorded passes only: a key the arm revoked was not graded again until
re-offered. Margin 0 reproduces each arm's own record exactly.

| read arm, margin | KK | KR | RK | RR | key-passes revoked | keys | L3 keys of `st_gn_yk` with neither half revoked |
|---|---|---|---|---|---|---|---|
| -0.10 | 199 | 7 | 1 | 0 | 2 | 2 | 19/19 |
| -0.075 | 194 | 7 | 6 | 0 | 7 | 6 | 11/19 |
| -0.05 | 192 | 7 | 8 | 0 | 9 | 6 | 11/19 |
| -0.02 | 186 | 4 | 14 | 3 | 18 | 11 | 6/19 |
| 0 (the arm) | 179 | 4 | 21 | 3 | 25 | 15 | 1/19 |
| +0.05 | 165 | 1 | 35 | 6 | 42 | 18 | 0/19 |

Every one of the read arm's 24 worthless gains lies in [-0.10, -0.0007]. On the world arm, every
margin at or below -0.005 revokes only its 8 no-consumer keys (19/19 L3 keys whole). Its 6
worthless gains are exactly 0.000: firing the consumers of junk L2 keys changed nothing.

**M-2 Persistence.** Runs of consecutive non-positive priced passes ending at each worthless
revocation, read arm: 17 of length 1, 6 of length 2, 1 of length 3 (the key had been graded 0 to
32 times before). Under "revoke only after k consecutive non-positive passes":

| k | read arm revocations still firing at their pass | world arm |
|---|---|---|
| 2 | 7 of 24 | 2 of 6 |
| 3 | 1 of 24 | 1 of 6 |
| 4 | 0 of 24 | 0 of 6 |

Of the read arm's 24, 17 are of TRUE L2 keys and 7 of junk (`[3,7]`, `[4,1]`, `[5,7]`, `[7,5]`,
`[7,4]`). The world arm's 6 are all junk.

**M-3 The diet at the seam.** The readout's row buffer held no span-4 (L3) row at any save through
c100. The first span-4 rows are at the c110 save (804 on the read arm), and 7,641 of 30,000 at c201.

| | before any span-4 row | after |
|---|---|---|
| read arm, L2 priced key-passes | 82 | 62 |
| … revoked worthless | 20 (0.244) | 4 (0.065) |
| … of which the world would keep | 18 | 3 |

By era the read arm's 24 fall 3 / 17 / 3 / 0 / 1, and the world arm's 6 fall 2 / 2 / 2 / 0 / 0.

**M-4 Re-offers.** The read arm made 14 re-offers, and 12 of them released a revocation the world
would have kept. The 21 wrong revocations (worthless where the world kept) fall on 12 keys. Twelve
were followed by a re-offer; 9 were not.

Every read-arm key still revoked at the end was in the live build at every later save. On the
world arm the same holds, except for two L3 keys whose L2 half `[7,4]` it had revoked; those were
unbuildable afterwards. Its consumer set at
every later save was exactly the set at its revocation: in the survey, no consumer at support
appeared that it did not already have. On the record, then, a key that did not come back had no
new consumer. Its level above had stopped growing around it, so the re-offer rule had no new
evidence to act on.

**M-5 The bill.** World arm:
- 35 grade passes from c25 and 2,153 fires: 61.5 a pass and 12.2 a cycle over c25–c201.
- 3.15 fires per graded key-pass (the unfired base pool included).
- 192 world queries a fire (one pool of `n_aud`), 413,376 in all, or 2,335 a cycle.

For scale, `st_gw_yk`'s walk gate billed 13,632 over its whole run. The read arm fired 679 times
(3.8 a cycle) at no bill.


### §4.7 Phase 2, seed 2 (`rb_g2`, `figures/rb_g2_reduction.txt` [L] [M], `figures/rb_grade_seedtable.txt`)

Both arms complete, 201 cycles, one container each. They were restored from `rb_s2`'s c20: the
digest version was resolved by trial to v2 (§5.4), and R-2 PASSed on both. Clock-yoked to
`vo_s3d2:voi3_dp`. Cost 2.17 (read) + 2.19 (world) GPU-h.

| check | `rb_gw_yk` (world) | `rb_gr_yk` (read) |
|---|---|---|
| first revocation | c45 | c65 |
| G-I by hash with `rb_s2`, saves before it | equal at c30, c40 | equal at c30, c40, c50, c60 |
| G-I off the log: first cycle any series differs from `st_gn_yk` | c45, = the first revocation | c65, = the first revocation |
| G-I off the records before it | 33 = 33, 28 = 28 | 47 = 47, 40 = 40 |
| G-W | c30, L3, e 1.0 -> 0.6875 = audition, 0 instances differ | the same |
| G-S | never red | never red |

| | `rb_gw_yk` | `rb_gr_yk` |
|---|---|---|
| key-passes graded / kept | 670 / 655 | 563 / 542 |
| revocations (no consumer, worthless) | 15 (7, 8) | 21 (6, 15) |
| worthless the other currency would have kept | 6 of 8 (the read) | 10 of 15 (the world) |
| re-offers | 7 | 10 |
| decision vs the other currency, KK/KR/RK/RR | L2 399/17/5/2, L3 165/29/1/0, L4 35/0/0/0 | L2 281/20/6/5, L3 122/14/4/0, L4 12/0/0/0 |
| revoked at the end L2/L3/L4 | 3/3/1 | 6/1/1 |
| `st_gn_yk`'s L3 / L4 keys with a half held revoked at the end | 7/24, 1/22 | 10/24, 1/22 |
| M-1 at margin 0: L3 / L4 keys with neither half ever revoked | 17/24, 21/22 | 8/24, 16/22 |
| fires; world queries billed to the grade | 2,058; 395,136 (2,232 a cycle) | 1,669; 0 |
| L3 rows adm/live, L5, at the last pass | 65/65, 112/112 | 75/77, 48/48 (`st_gn_yk` 85/85, 144/144) |
| task error − `st_gn_yk`, era 1..5 | -0.001 -0.033 -0.002 +0.017 +0.008 | +0.000 +0.026 +0.010 +0.029 +0.092 |

**The seed-2 read arm's 15 worthless revocations.**
- 11 are at L2: c65 (2), c75 (3), c80, c135, c175 and c197 (3). 4 are at L3: c120 (2), c125 and c197.
- The world kept 10: six L2 revocations and all four L3 revocations (+0.09 to +0.40). The six L2 are five of true keys and one of the junk key `[5,7]`, gains +0.18 to +0.29. The grammar is the same on both seeds (`rule_seed` 0).
- The five the world agreed with are junk L2 keys (`[4,1]` three times, `[1,4]`, `[7,4]`), world gain 0.000.

**Seed 2's diet.** Span-4 rows appear at the c80 save (536 on the read arm), against c110 on seed 0.
Span-8 rows appear at the c160 save (104), and were at 761 by c201.
- Priced at a pass whose later bracketing save held fewer than 512 rows at the consumers' span: 5 of the 15, 4 of them world-kept.
- Priced with no row there at all: 5, 4 world-kept. These are the two c65 L2 revocations and the three L3 revocations at c120/c125, whose span-8 buffer was empty.
- On seed 0, the same counts were 20 of 24 below 512 and 20 with no row (18 world-kept each).

**Persistence (M-2).** Of the 15, 1 would still fire under 2 consecutive non-positive passes and 0
under 3. The world arm's 8: 4 under k=2, 2 under k=3, 0 under k=4.

**Re-offers (M-4).** 6 of the read arm's 10 re-offers released a revocation the world would have
kept. 4 wrong revocations were never re-offered: `[5,7]` c65, `[0,7]` and `[1,1]` at c197, and
`[3,0,0,7]` at c197.

### §4.8 The no-consumer revocations against the ungated arm, both seeds ([M] M-6)

For every revocation for want of a consumer, M-6 records three things. First, which of
`st_gn_yk`'s walked keys one level up have the revoked key as a half. Second, the revoked key's own
consumers in `st_gn_yk` at every save of the ungated re-run (`rb_consumer_survey` over `rb_s1`'s
and `rb_s2`'s saves c30–c201, `figures/rb_s1_consumer_survey.json`,
`figures/rb_s2_consumer_survey.json`). Third, whether this arm re-offered the key later.

| arm | no-consumer revocations | consumed in `st_gn_yk` by the revocation | first consumed in `st_gn_yk` after it | never consumed in `st_gn_yk` | not a key `st_gn_yk` admitted |
|---|---|---|---|---|---|
| seed 0 `rb_gw_yk` | 8 | 1 | 2 | 1 | 4 |
| seed 0 `rb_gr_yk` | 1 | 1 | 0 | 0 | 0 |
| seed 0 `rb_grd_yk` | 1 | 0 | 0 | 0 | 1 |
| seed 2 `rb_gw_yk` | 7 | 1 | 1 | 1 | 4 |
| seed 2 `rb_gr_yk` | 6 | 0 | 0 | 4 | 2 |

**Seed 0, world arm.** One of its 8 no-consumer revocations is a half of `st_gn_yk`'s walked
keys: the L3 key `[1,1,1,1]`, revoked at c110 at age 185 observations.
- It is a half of 5 of `st_gn_yk`'s 9 walked L4 keys. In `st_gn_yk` it already had a consumer at the c80 save.
- The world arm never re-offered it.
- At the end, those 5 of 9 L4 keys have that half revoked for want of a consumer. `st_gn_yk`'s 4 L3 keys held out at the end are held out by `worthless` revocations of junk L2 keys.
- Two of the other seven were first consumed in `st_gn_yk` only after the revocation. `[2,5,1,6]` at the c160 save (the world arm re-offered it at c165). `[2,5,2,5,3,0,0,7]` at the c180 save, by when its age in `st_gn_yk` was 479 observations, past the 160 window (re-offered here at c190).
- Four are keys `st_gn_yk` never admitted at any surveyed save, and `[7,4,2,6]` was never consumed in `st_gn_yk`.
- The read arm revoked the same `[1,1,1,1]` at c100 for want of a consumer and re-offered it at c120.
- The world arm's deep-era task error against `st_gn_yk` is +0.176 in era 4 and +0.053 in era 5.

**Seed 2, world arm.** None of its 7 no-consumer revocations is a half of any key `st_gn_yk`
walked. `st_gn_yk`'s L3/L4 keys held out at the end (7/24, 1/22) are held out by `worthless`
revocations. Its deep-era error against `st_gn_yk` is +0.017 in era 4 and +0.008 in era 5.

---

## §5 Defects and facts, in order, each beside its correction

**§5.1 The first R-1 vehicle hit a lineage gate that is not an identity at preflight's tolerance.**
The first machinery smoke saved at c1 without incident and died at c2 on the donor's own in-loop
A-4 ("the read gate handed the world's success IS the world gate"). Not the save: `preflight` runs
`extend_tol = 1.0` (so that `census_extend` fires at toy sizes), under which the world gate admits
everything and the read-as-world rule does not, so A-4 holds only on a pass where no candidate moves
the success. It held on `sostenuto`'s `st_pf1` because that substrate solved nothing; my first R-1
substrate also skipped `preflight`'s pre-setup `entry_recorder_check`, which draws on the global
streams, so it was a different substrate on which a candidate did move the success. Corrected twice
over: R-1 now builds `preflight`'s substrate call for call (so the vehicle is the lineage's own), and
the admission twin runs at the arm of record's `extend_tol = 0.0` (at which A-4 is the identity it
is on the paid arms). The twin admits everything either way (`admit="none"`).

**§5.2 The digest compared wall clock.** The second smoke restored c3 in a fresh container (R-2
PASS, the world identical) and continued to c5; the two runs' c5 digests differed on exactly
`_adm/passes[3..4]` and `log/sb[3..4]`, the two cycles the restored run executed. Every differing
leaf was a `t_s`: the walk's per-pass seconds and the world-model diagnostics' seconds, the only two
`time.time()` readings the loop stores (found by reading every `time.time()` in `run_arm`). The whole
arm file, `t_s` dropped, was identical. Corrected: the digest leaves `t_s` out (`_RB_CLOCK_KEYS`),
and says why.

**§5.3 The preflight's entry record is uncapped.** At preflight the arm file grew ~60 MB a cycle (the
true L5 table's 262,144-row probe vector logged every cycle), which made a save every cycle cost more
each cycle and would have put ~1.4 GB arm files in front of the R-1 comparison. R-1 runs the twin
at the paid arms' `entry_rec_cap = 4096`, which caps a log, not a behaviour.

**§5.4 A refinement of the digest rule made every state saved before it unreadable.** Phase 2's
first preflight (§10.8, defect 1) corrected the digest to leave out the arm's label on records and
the grade's knobs. `rb_s1`'s states had been saved under the OLD rule, and the digest is also what
gate R-2 re-hashes at a restore, so the first seed-0 graded launch (`rb_g1`,
`ap-hrbeGNVpNJffzeXx5LKaFf`) was refused by R-2 on both arms, on exactly `env/events` and
`env/merge_events` (every record "children equal, node differs": the `arm` field, dropped from the
re-hash, hashed in the save). The gate did its job; the state was intact. Cost: both containers'
setup and restore, ~0.38 GPU-h, nothing run. Corrected by VERSIONING the rule: every save records
the version it hashed with (`digest_v`; a save without one is v1 = `t_s` and `rb_*`; v2 adds `arm`
and `adm_grade*`), R-2 re-hashes under the SAVED version, and G-I re-hashes this run's state under
its reference's version. Against a v1 reference the label-carrying record lists (`events`,
`merge_events`) cannot be compared by hash, so G-I leaves those two out there (`labels=True`) and
the reducer compares them record by record, label dropped. Checked before paying again, on CPU
against `rb_s1`'s saved c20 (`rb_peek_state`): re-hashed under v1, `events`, `merge_events`, `log`
and `_adm` all reproduce their saved hashes; under v2, `events` and `merge_events` do not — the
refusal, reproduced.

The same check on `rb_s2`'s c20 found the other case: `rb_s2` was launched on the binary that
already carried the v2 rule but did not yet record a version, so its states are v2 with no
`digest_v` (v2 reproduces `events`/`merge_events`, v1 does not). So a save that recorded no
version is resolved by trial at the restore: v1, then v2, and the one that reproduces the saved
tree name for name is the save's (R-2 is exact under it; the resolved version and whether it was
recorded are in `rb.resumed_from`). G-I, against a reference that recorded none, uses the version
the restore resolved when the reference is the run the arm was restored from (it is, on both
seeds), and v1 otherwise.

**§5.5 Preemption restarts an input from its start, and the saved states were not used for it.**
Two preemptions, both answered by Modal restarting the input with the same arguments. (1) `rb_s2`'s
arm container at c60: the arm re-ran from c1 (its saves c10–c60 were rewritten; ~0.6 GPU-h). (2)
`rb_g1`'s `sweep` coordinator at ~00:44: the restarted coordinator cancelled both graded arms
(at c140 and c137) and re-spawned them from the c20 restore. The re-spawned pair reproduced the
cancelled pair's saved digests exactly (c30 on both arms, c40 on `rb_gw_yk`: an incidental
cross-container identity check), and was stopped at ~c45, because each arm had its own complete
c130 save. ~0.4 GPU-h paid for nothing. Corrected by three
knobs-off additions: `rb_resume` may name `{arm}` (one sweep, a per-arm path); `rb_auto_resume`
makes an arm that starts with complete saves of its own later than `rb_resume`'s (the `.pt` in
place and its sidecar readable) resume from the latest of them, so a restarted input continues
instead of re-running; and every save's metadata carries the saving run's record (`rb_hist`:
where it was restored from, its saves' G-I results, its host, and the run it itself continued),
so a resumed arm's file keeps its whole chain of container segments. `rb_auto_resume` is off by
default, because under a reused tag it would resume a stale directory. Every run with a save
knob on now records its host (`rb.host`, `rb_host_info`); before this only R-1's runs did.

`rb_g1` is therefore three segments: `rb_s1`'s c1–c20 (restored), the graded arms c21–c130 in
their first containers, and c131–c201 resumed from their own c130 saves (`RUN_rb_g1_resume.sh`).
The first graded segment's c130 saves predate `rb_hist`, so its record is taken from its log
(`results/launch_rb_g1_seg1.log`): G-I by hash equal to `rb_s1` at c30, c40 and c50 on
`rb_gr_yk` (first revocation c55); `rb_gw_yk` revoked first at c30, before its first save, so
its identity with `st_gn_yk` over c21–c29 is the reducer's, off the log and the records.

---

## §6 Reproduction

```bash
cd experiments/            # MODAL_PROFILE=chromatic
B=rhm/practice/voicing/sotto_voce/aliquot/rubato
python3 -c "from rhm.practice.voicing.sotto_voce.aliquot.rubato import rubato as R; R.rb_static_gate(); R.rb_static_falsify()"   # R-0
python3 $B/launch_detached.py --fn fidelity_smoke --tag rb_gf1          # G-F against sostenuto.py
sh $B/results/RUN_rb_r1.sh                                              # R-1 (+ its falsifications)
sh $B/results/RUN_rb_s1.sh                                              # the seed-0 reference, saved states on
sh $B/results/RUN_rb_s2.sh                                              # the seed-2 reference
python3 $B/fetch_compact.py --tag rb_s1 --fetch --replace
python3 $B/reduce_rubato.py --bank-check --tag rb_s1 --bank-tag st_s1
python3 $B/reduce_rubato.py --bank-check --tag rb_s2 --bank-tag st_s2
python3 $B/reduce_rubato.py --r1 --tag rb_r1
# phase 2
python3 -c "from rhm.practice.voicing.sotto_voce.aliquot.rubato import rubato as R; R.rb_grade_gates()"   # G-D, G-O
modal run $B/rubato.py::rb_pf_grade --tag rb_pfg2                       # the grade preflight (§10.8)
sh $B/results/RUN_rb_g1.sh                                              # seed-0 graded arms, from rb_s1's c20
python3 $B/fetch_compact.py --tag rb_g1 --fetch --replace
sh $B/results/RUN_rb_g1_resume.sh                                       # (after the preemption, §5.5)
python3 $B/reduce_rubato.py --grade --tag rb_g1 --sos-tag st_s1 --sos-tag st2_s1 --bank-tag sb_s1   # reads results/launch_rb_g1_seg1.log
sh $B/results/RUN_rb_g2.sh                                              # seed-2 graded arms, from rb_s2's c20
sh $B/results/RUN_rb_g1d.sh                                             # round 3: the diet-gated read arm, seed 0, from rb_s1's c50
python3 $B/fetch_compact.py --tag rb_g2 --fetch --replace
python3 $B/fetch_compact.py --tag rb_g1d --fetch --replace
# [M]'s CPU reads of the saved states (outputs copied to figures/ as the *_diet.json, *_revoked_survey.json, *_consumer_survey.json)
modal run $B/rubato.py::rb_peek_state --paths "rb_g1__rb_gr_yk/rb_gr_yk/ck/c050.pt,..." --expr "<vo_om rows by span, DESIGN §4.6>"
modal run $B/rubato.py::rb_consumer_survey --ck-dir rb_g1__rb_gr_yk/rb_gr_yk/ck --which revoked --out-name revoked_survey.json
modal run $B/rubato.py::rb_consumer_survey --ck-dir rb_s1__st_gn_yk/st_gn_yk/ck --cycles 30,40,...,201 --out-name consumer_survey_full.json
python3 $B/reduce_rubato.py --grade --tag rb_g2 --sos-tag st_s2 --sos-tag st2_s2 --bank-tag sb_s2
python3 $B/reduce_rubato.py --grade --tag rb_g1d --sos-tag st_s1 --sos-tag st2_s1 --bank-tag sb_s1   # beside rb_g1's arms
python3 $B/reduce_rubato.py --seedtable                                 # the two seeds beside sostenuto's five
```

---

## §10 Phase 2: admit-then-grade

The brief is `SPEC.md`'s phase-2 section. The op, in one sentence: every at-support key is served
provisionally (the `none` walk, unchanged), and at the walk's own cadence a key that has been
served long enough for the level above to use it is **graded by what uses it one level up**,
revoked on positive evidence that nothing there uses it or that what uses it is worthless, and
re-offered when new evidence arrives. Every addition is marked `# [rubato]`; with `adm_grade`
absent the arm is `st_gn_yk` (and G-F / G-I say so).

### §10.1 Admission: the `none` walk, untouched, cap and all

Provisional admission IS `st_gn_yk`'s walk: every not-yet-decided at-support key offered in the
miner's `(-count, key)` order and admitted. The walk is not touched, the offer cap of eight
included: it never bound on any sostenuto arm (key arrival was the budget), and keeping it is what
makes the arm `st_gn_yk` exactly until its first revocation. The only change to the walk is that
a REVOKED key counts as decided while it is revoked, so the walk does not re-admit it five cycles
later; re-offering is the grade's own act (§10.5).

### §10.2 Consumption: consumers at support one level up, in the panel

A level-l key's **consumers** are the level-(l+1) keys the OBSERVATION PANEL holds at support
(`mine_support` 3) that have one of the key's level-l classes as a half (`rb_consumers`; the
classes are `quot.id_of` of the key's rows, sostenuto r2's `sos_classes`). The panel, not the
committable miner, for sostenuto §9.1's reason: the committable level-(l+1) miner is era-gated and
empty exactly while level l's keys are being admitted; the panel counts the same reader parse of
the same solved configurations at every level in every era, and it is what the merge op already
reads for its one-level-up licence. This is "the next level can build on it" in the build's own
currency: `ClassMiner.build` emits only at-support keys whose halves are served.

The executor's own writes of a key in solved instances are NOT a separate criterion. At an
unadopted level they are zero by construction (there is no macro to write with), so reading them
would be the demand-blindness failure again at every frontier level; at an adopted level a key the
executor writes into solved configurations is exactly what the panel counts one level up (the
panel observes the level-(l+1) span containing the repaired cell, whose halves are what was
written), so it is already in the consumer count.

**The window**, in the panel's own currency: a key is graded only after the level-(l+1) panel has
made `adm_grade_obs` observations since the key's admission (the admitting pass's `obs_nobs`,
recorded per key). A level whose panel holds nothing at support is SILENT (no key there is
graded): the frontier's emptiness is a fact about the level, not the key (sostenuto §9.3 Fact 2).
The window's value is set from the ungated arm's own record (§10.6).

L5 keys are served and never graded: the only level above L5 is the panel's whole-string L6, which
has no committable level and holds nothing at support before c190 on the record.

### §10.3 The grade: counts decide non-use; the value of the fired consumers prices use

Both in sequence, which is the reasoning the brief left to this node:

1. **No consumer** (the level above non-silent, the window elapsed): REVOKE. That is positive
   evidence of non-use in the miner's own currency, and it needs no organ. It is the `demand`
   gate's criterion moved to AFTER admission with a window, which is the difference that matters:
   `demand` refused keys whose consumers could not exist yet, because the level above can only
   form keys over halves that are served.
2. **Consumers exist**: each consumer (the `adm_grade_cons` = 4 best-supported) is FIRED —
   pp1's single-entry fire: a level-(l+1) table of that consumer's rows alone, built over what is
   SERVED at level l, forced as the macro at the level-(l+1) node containing the era's cell, on a
   fresh pool of that level-(l+1) cell (`n_aud` 192, its own seed family, disjoint from the
   practice, gate and test pools). The same pool is also graded UNFIRED. A consumer's gain is the
   value on its fired configurations minus the value on the unfired pool, where the value is the
   shaped projection's level (`read`, the arm of record's readout at the level-(l+1) span, the
   same `VoProjBank.predict` sostenuto A-2 pinned to `preplay.pj_predict`) or the world's success
   (`world`). The key's grade is the **best** consumer's gain: a table's audition is a max, not a
   sum (preplay), and a key is as useful as its best use. KEEP iff it exceeds `adm_grade_margin`
   (0.0 = strictly positive), else REVOKE.
   Silent (keep) where there is nothing to read: no consumer buildable over what is served
   ("unpriced"), and on `read` a readout that is unfit or degenerate ("no_reader": a constant level
   would otherwise make every gain 0 and revoke everything that has a consumer).

This puts the read where it has been shown to work: on fired states (pp1: the level's mean over
single-entry fires prices a candidate in the world's order, Spearman 0.17–0.53, frozen and twin at
chance), never on base-versus-base-plus-one, where a candidate moves 1–5 of 192 instances.

**Both values on every graded fire, whatever the arm consumes**, and what the OTHER currency would
have decided on the same fires (`keep_other`), so the per-key confusion against the world exists on
the read arm at zero bill. Only `world` is billed: every graded fire, the unfired pool's included,
at the walk's rate.

### §10.4 Revocation, and what each state serves

| a key's state | served by `operative` |
|---|---|
| at support, level not yet walked (sostenuto's pending) | yes — the donor's ungated line |
| admitted, inside its window (provisional) | yes |
| admitted and graded KEPT, or silent | yes (re-graded every pass: the evidence moves) |
| REVOKED | no — out of the admitted set, so `sos_filter` drops its rows |
| re-offered | back to the walk; served again from the next pass's admission |

A revocation is applied after the whole pass has been read (every level's decisions are taken on
the same state), then the move set is rebuilt. A merge re-keys the revoked set with the admitted
and rejected sets; a revoked key merged into an admitted one is admitted (counted).

### §10.5 Re-offering

A revoked key goes back to the walk when the level above holds, at support, a consumer it did NOT
hold when the key was revoked (`rb_reoffer_new`): the consumers present at revocation were weighed,
a new one is new evidence. Its window restarts at re-admission. While a key is revoked its
consumers can still form: at an unadopted level the panel is table-free, and at an adopted one the
base moves still write its spellings.

### §10.6 The window and the restore cycle, from the record

Read off `rb_s1`'s own saved states (`rb_consumer_survey` at c10–c120; the ungated arm, the
no-revocation floor) before anything was paid, every admitted key's age in the level-(l+1)
panel's observations at the moment it first had a consumer at support:

- **L2** (17 keys by c120): every one is consumed eventually. Nine have a consumer within 40
  observations of admission, and the slowest two, (2,6) and (4,1), took **160**. The panel observes
  ~8 configurations a cycle at every level, so 160 is ~20 cycles.
- **L3** (keys from c65): consumers form from c70, once the L4 panel holds keys (10 at c50, 78 at
  c120). Two keys had no consumer 136 observations after admission.
- **L4**: the L5 panel holds nothing at support before ~c180 on this arm, so L4 is SILENT
  (§10.2) for all but the last era.

**The window is 160 observations** (`adm_grade_obs`): the smallest window at which no key that the
ungated record shows being consumed would be revoked for want of a consumer. At 40, keys (1,1),
(2,6) and (4,1) would have been revoked at L2 before their consumers formed, which is the frontier
failure `demand` made, on a timer.

**The restore cycle is c20.** Keys admitted at the c5 pass (L3 panel `n_obs` 35 then) reach an
age of 160 at `n_obs` 195, i.e. the c25 pass is the first cycle at which a grade can fire. The
latest saved state before it is c20, which `rb_s1` saved on its cadence. That buys 20 of 201 cycles:
not a large saving, because the level above holds keys at support from c5. The saving the saved
states buy is elsewhere: the next change to this op starts from the saved states of THIS arm.
Before the readout's first fit (c50 on the record) the read arm's grade is silent on consumed keys
(`no_reader`) and revokes only for want of a consumer; the world arm grades from c25.

**Seed 2, from `rb_s2`'s own record** (`rb_consumer_survey` over its c10–c120 saves, and the
panel's observation counts in its arm file): L2 keys are consumed as on seed 0 (8 of 10 at c10,
13 of 13 by c40, 15 of 15 at c120); L3 keys from c70 (6 of 6), 13 of 16 at c120; L4 is silent
through c120 (nothing at support in the L5 panel). Keys admitted at the c5 pass (L3 panel
`n_obs` 39) are 158 observations old at the c25 pass and 198 at c30, so **c30 is the first cycle
a grade can fire on seed 2, and the restore cycle is again c20**, the latest save before it.

### §10.7 The in-loop identity checks (the offline `gated_walk` cannot reproduce a temporal op)

| gate | where | claim | falsified by |
|---|---|---|---|
| **G-D** | `rb_grade_gates`, CPU | the rule on 11 designed cases | mean for max; `>=` for `>`; an unfit reader read as a verdict; no consumer kept — 4/4 RED |
| **G-O** | `rb_grade_gates`, CPU | re-offer iff a NEW consumer | any consumer re-offers — RED |
| **R-2** | every restore | the restored `st_gn_yk` state hashes to the saved one | (phase 1) |
| **G-I** | every save, in-run | until its first revocation the graded arm IS the reference arm, by hash, name by name, at every cycle both saved — everything but the grade's own `_adm` keys, Python's unseeded stream, and on `world` the bill and what carries it. The block snapshots and puts back everything its reads touch (`operative`'s counters, every miner's `last_build`, the readout's `stat`) | `adm_grade_falsify=touch`: one draw from `grng` at the first pass, no revocation |
| **G-W** | in-run, once | the world grade's fire IS the donor's `audition_macro` on the same table, node and pool (`e` equal exactly), on the first fire that repairs anything | `node`: the audition at the sibling node |
| **G-S** | in-run, after every change | a revoked key is served nowhere (`operative`'s groups ∩ revoked = ∅) | `serve`: revocation that leaves the key admitted |

### §10.8 The preflight (`rb_pf_grade`, tag `rb_pfg2`), and the two defects it found first

Gate R-1's vehicle (`preflight`'s substrate, the plan of `rb_r1_P`, the admission twins with the
panel SEEDED so the level above holds keys at support), window 40, stop c12, a save every 2
cycles, and all six arms IN ONE CONTAINER, so the identity checks are not confounded by the
host floor (§4.2):

| run | what | result |
|---|---|---|
| N | `rb_pf_gn`, the no-grade floor | the reference for G-I |
| GR | read grade from c1 | G-I exact at 6 of 6 saves; 69 keys graded, all kept |
| GW | world grade from c1 | G-I exact at 6 of 6 saves (bill left out); **G-W PASS**: the grade's first grade-changing fire (c4, L3, e 1.0 -> 0.9375) equals the donor's `audition_macro` and, instance by instance, `MC.apply_any` + `grade` (0 instances differ) |
| GRs | read grade RESTORED from N's c2 (the fork path: a no-grade state into a graded arm, `rb_allow_cfg`, schema evolution adding the grade's `_adm` keys) | R-2 PASS; G-I exact at 5 of 5 saves; at the stop equal to GR in every value. The one hash difference is the ORDER of `_adm.gr_obs0[2]` (GR recorded window starts key by key as they were admitted, GRs all at its first pass), which nothing reads |
| GM | world grade, margin 0.5 | revocations run: first at c4, 10 by c12; G-S clean after every change; G-I exact at the save before |
| F | GM with `touch,node,serve` | **G-I RED** (c2, before any revocation), **G-W RED** (sibling node), **G-S RED** (a revoked key still served) — 3/3 |

At margin 0 on the seeded panel nothing is revoked, and that is the rule working: the consumers
there are TRUE keys, and their fires repair.

**Defect 1 (the first preflight, `rb_pfg1`).** G-I was red on every graded arm from c2 without a
revocation. Two causes, both bookkeeping. Every event and merge record carries the arm's LABEL, and
a graded arm writes its own. The recorder keeps a COPY of the config, which carries the grade's own
knobs. Corrected in the digest (`_rb_skip_key`: the `arm` field and the `adm_grade*` knobs, beside
`t_s` and `rb_*`; digest rule v2, which then needed versioning, §5.4). The one other difference in that run was the host floor: the restored arm landed
on the only non-AVX-512 host of the five, and its readout's degenerate refit at c12 differed in
`loss` by an ulp. That is why the preflight now runs in one container.

**Defect 2 (same run).** G-W's falsification tied: at 16 instances a write at the sibling node
repaired 1 of 16 as well (possible-set grading is not local). Corrected: G-W now compares the
per-instance success vectors, and it runs on the first fire that CHANGES the grade.

---

## §11 The read grade with a per-span diet (`rb_grd_yk`, tag `rb_g1d`)

**Why.** Seed 0's re-reading (§4.6, M-3) located the read arm's wrong revocations. Twenty of its 24
worthless revocations, 18 of them on fires where the world's success rose, were priced while the
readout's buffer held no row at all at the span the fired states were read at. An L2 key is
graded on its L3 consumers' fires, which the readout reads at span 4. Its first span-4 row came
between c100 and c110. The revocation rate on L2 priced key-passes was 0.244 before any span-4 row
and 0.065 after. The readout was being asked to price states it had never been fit on at that
span, which is pp1's off-diet degradation. The grade's order (admit, then revoke on evidence) is
not what failed.

**The change, and nothing else.** The read arm's `no_reader` silence already keeps a key when the
readout is unfit. The diet extends it per span. A level-l key's consumers, at span s^l, are priced
only once the readout's FITTED buffer holds at least N rows at that span. Until then the key is
kept, silently (`why = "no_diet"`). Want of a consumer still revokes, since it needs no organ.
Above the diet the rule is `rb_gr_yk`'s: margin 0, window 160, four consumers, decided per pass,
the same re-offer rule. The fires still happen below the diet, so the read's and the world's value
on every fire stay on the record. Persistence and the margin are NOT knobs of this arm. They are
re-applied post hoc off its record (M-1, M-2), so the diet's effect is read alone.

**The count.** The count is the buffer's rows at the consumers' span among the rows the last refit
drew its fit and validation from (the non-hold rows, `h >= hold`). The refit runs in the same cycle
before the grade block, with no push between, so this count is exactly what the readout had been
fit on. Every pass logs the buffer's rows by span, all of them and the fit-available ones
(`gr_passes[*].span_rows`, `span_rows_fit`), plus each level's count and whether it was open
(`levels[*].span_fit`, `diet_ok`), so the choice of N can be re-read.

**N = 512.** 512 is the readout's own `vo_om_min`: the number of rows below which it fits nothing
and reads nothing (`no_reader`). The diet asks the same of each span. It is one number the lineage
already uses for "enough to read at all", not one tuned to this record.

On `rb_gr_yk`'s buffer, span 4 held 0 rows through the c100 save and 804 of 18,672 at c110
(about 720 of them fit-available). So the L2 keys' consumers open between c100 and c110. That is
about 7% of the fit's 8,192-row subsample once the buffer passes it. Span 8 (the L3 keys'
consumers) held 430 rows at c201 on `rb_gr_yk`, so under 512 L3 keys may never be priced on this
arm. L3 keys then leave service only for want of a consumer. On `rb_gr_yk`'s record, none of its
64 L3 graded key-passes was a worthless revocation.

**Gate G-N** (`rb_grade_gates`, CPU): 7 designed cases. Silent below the diet on a read, whatever
the gain. Want of a consumer still revokes. `unpriced` and `no_reader` keep their precedence. The
world needs no diet. Above the diet the verdict is G-D's. Falsified 3/3 RED: the diet ignored; the
diet silencing want of a consumer; the diet silencing the world. R-0, G-D and G-O stay green, with
all their falsifications RED.

**The restore** is `rb_s1`'s c50, the latest save before `rb_gr_yk`'s first revocation (c55).
`rb_gr_yk` was `rb_s1` by hash at c30, c40 and c50 (G-I, §4.5), so the restore point is common to
the arm and the arm it corrects. The grade's window starts are back-filled from each key's
admission pass (as for any restore of a no-grade state, §10.8's GRs), so the restore changes no
key's age. G-I runs against `rb_s1`'s saves until this arm's first revocation. The saved state
carries everything the diet reads (the readout's `spn` and `h` columns are state). One arm, read
only; `rb_auto_resume` on. Cost is about 2 GPU-h.

### §11.1 Result, seed 0 (`rb_g1d`, `figures/rb_g1d_reduction.txt` [L] [M] beside `rb_gr_yk` and `rb_gw_yk`; seedtable (v))

Complete: 201 cycles, 1.61 GPU-h, one container.

**Identity.**
- R-2 PASS at the restore from `rb_s1` c50.
- G-I by hash equal to `rb_s1` at c60, c70, c80, c90 and c100.
- First revocation c105. Off the log, the first series to differ from `st_gn_yk` does so at c105.
- Every event and merge record before c105 is equal: 110 = 110 and 86 = 86.
- G-W: c55, L3, e 1.0 -> 0.6615, equal to the donor's audition.
- G-S never red.

**The diet on the record.**
- 235 key-passes were silent for the diet.
- L2 first priced at c105. The span-4 fit rows were 0 at the c100 pass and 593 at c105.
- L3 first priced at c190, once span 8 reached 512 fit rows.
- L4 keys were never priced.

| | `rb_gr_yk` | `rb_grd_yk` (diet 512) | `rb_gw_yk` |
|---|---|---|---|
| first revocation | c55 | c105 | c30 |
| revocations (no consumer, worthless) | 25 (1, 24) | 13 (1, 12) | 14 (8, 6) |
| worthless the world would have kept | 21 of 24 | 10 of 12 | — |
| re-offers | 14 | 5 | 6 |
| decision vs world on the same fires, KK/KR/RK/RR | L2 116/4/21/3, L3 54/0/0/0 | L2 272/11/6/2, L3 88/0/4/0 | — |
| revoked at the end L2/L3/L4 | 11/0/0 | 7/1/0 | 3/4/1 |
| `st_gn_yk`'s L3 keys with a half held revoked at the end | 18/19 | 11/19 | 4/19 |
| M-1 at margin 0: L3 keys with neither half ever revoked | 1/19 | 5/19 | 15/19 |
| task error − `st_gn_yk`, era 1..5 | -0.001 -0.009 +0.077 +0.190 +0.239 | +0.000 +0.002 +0.029 +0.081 +0.227 | -0.004 -0.002 +0.035 +0.176 +0.053 |
| L3 rows admitted/live, L5, at the last pass | 9/9, 1/1 | 6/24, 32/32 | 62/65, 80/80 |

**How the 12 worthless revocations fall.**
- Five came at c105, the first priced pass: `[2,5]`, `[5,7]` and `[7,5]` (read -0.0013 each, world +0.28), `[4,6]` (-0.030, world +0.17) and `[4,1]` (-0.101, world 0.000, junk).
- `[1,1]` at c110 (read -0.0073, world +0.50) and `[7,4]` at c115 (junk, world 0).
- `[0,7]` at c180, after 15 priced passes all kept (read +0.003 to +0.27) and 10 silent for the diet (read -0.021, world +0.05).
- Four L3 keys at c190/c197, at the first priced L3 passes (world +0.07 to +0.46).
- Every one is a run of length 1: under 2 consecutive non-positive passes, none would fire at its pass (M-2).
- The L2 revocation rate with span-4 rows was 0.051 (8 of 158), against `rb_gr_yk`'s 0.065 on the same side of the seam and 0.244 before it (§4.6 M-3).

**Re-offers.**
- Of the 7 wrong revocations never re-offered, 6 are L2 keys.
  - 3 of them (`[2,5]`, `[5,7]`, `[7,5]`, one merged class) had a consumer they had not had at revocation, first seen at the c200/c201 saves, after the last grade pass (c197). Their re-offer was never checked.
  - The other 3 (`[4,6]`, `[1,1]`, `[0,7]`) kept exactly their revocation-time consumer set at every later save.
- `[2,6,2,6]` gained 16 new consumers by c200, also after its revocation at the last pass.

[^private]: Not mirrored: this link points to a document in the private lab repo (the roadmap, the queue, an unrun spec, reading notes, or a conversation). See the top-level README for what is held back and why.
