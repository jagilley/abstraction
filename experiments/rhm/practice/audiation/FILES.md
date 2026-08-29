# FILES — `audiation` (E1: the instrument, the dataset, and the decode matrix)

Machinery record for the node. **No results are interpreted here**, and there is still no
`README.md`: phase 1 produced the dataset (`audiation.py`, `analyze_audiation.py`), phase 2 the
offline `rev_pair` / `SlotFM` port and the decode matrix over it (`fit.py`), and the writeup
happens after both, with Jasper. Phase 2's numbers are in
[`figures/au_s0/fit/reduction.txt`](figures/au_s0/fit/reduction.txt) and are deliberately not
restated or interpreted anywhere in this file.

**Up**: parent arc [`../README.md`](../README.md) ·
`../../../../ROADMAP.md`[^private] §4.2, first shape **E1**
**Direct donor** (untouched): [`../conductor/`](../conductor/FILES.md) — `conductor.py` is
forked verbatim; `cd_s0/anchor` is the cross-tag replay reference, in-flight and offline.
**Recipe phase 2 ports** (into `fit.py`):
[`../../confabulation/temporal/epistemics/`](../../confabulation/temporal/epistemics/FILES.md)
— `SlotFM` (FM₂ gets the arriving item in a slot, FM₁ a learned `MASK`), the matched-capacity
readout ladder, `_stratum_residualise`, the `ens_cos` / junk-η² / scalar-norm / shuffled-null
guards. The observer ladder's *rung structure* (capacity-swept third-party predictors holding
increasing public information, plus a half-data budget control) comes from
[`../../confabulation/temporal/temporal_confabulation.py`](../../confabulation/temporal/temporal_confabulation.py)
(`Observer`, `train_observer`, `run_ladder`); its sequence *shape* does not port, because the
practice datum is a configuration, not a prefix.
**Design exchange**: `conversations/955af184-f31a-4ada-9361-716c7662f284.md` lines 1670–1745 —
where the E1 hypothesis was revised: **E1 tests agency, not privacy.** In teacher-forced NTP
the arity gap `g = ‖FM₂(s,u) − FM₁(s)‖` is identically zero (the world fills the second slot);
in practice the slot is the agent's own move and `g > 0`. The observer ladder is the mandatory
control, not the point.

## Code files

| file | purpose |
|---|---|
| `audiation.py` | The substrate. Forks `../conductor/conductor.py` **verbatim**; the additions are all one class of thing — instrumentation that consumes no RNG and changes no behaviour — and each is marked `# [audiation]`. Entrypoints: `audi_preflight`, `audiation_run`. |
| `analyze_audiation.py` | The **data report**, not a reduction of a result: the cross-tag replay gate, the in-run gates read back, dataset counts, four join-integrity properties, and a few descriptive statistics. Re-implements the two heads in numpy (no GPU, no torch) and validates that re-implementation against the run's own logged readouts before quoting it. |
| `launch_detached.py` | Session-isolated detached launcher (the donor's, retargeted). |
| `fit.py` | **Phase 2 — the offline fitter and the decode matrix.** No GPU, no Modal, no substrate: reads the phase-1 bytes plus each cycle's two bracketing head snapshots and re-derives every readout through `analyze_audiation.head_forward` (the recompute phase 1 certified). Builds the per-datum choice-state table; trains the matched `SlotFM` pair (FM₂ gets the slot of the agent's own move, FM₁ a learned MASK) on the **update**; runs the source × target decode matrix at matched readout capacity, the provenance (agency) conditioning, the practice-shaped observer ladder, the guards, a within-cycle ceiling and a strict-timing forward split. MLPs, AdamW, ridge and η² are all numpy; `_gradcheck()` against central differences is a hard assert at the top of `main`. Writes `figures/<tag>/fit/`. **`--ladder`** runs a second, disjoint mode — the conditioning-completion ladder of §7 — and writes `ladder.txt` / `ladder.json` / `ladder.png` instead; the default path is untouched by it. **`--e1b`** runs a third, disjoint mode — the E1b per-datum refit of §8, which reads the `revisions/` tables rather than the phase-2 table and writes `figures/<tag>/fit/reduction.txt` + `e1b.json`; `--e1b-no-cycle-row` skips its whole-cycle comparability row. Both extra modes branch before the phase-2 table is built, so the default path is byte-stable under them. |

## What the fork adds (each `# [audiation]`-marked)

| addition | where | why |
|---|---|---|
| **the per-decision recorder** | `DecRec`, `cand_src`, `flatten_cycle`, `DecWriter`; `rec=` threaded through `beam_moves` / `beam_moves_prop` / `plan`; attached at **one** call site, `run_arm` block (a) | everything E1 needs is computed and discarded inside the practice beam: `z_tip`, π's logits, the candidate value scores, the topk, the kept moves, tip lineage. `rec=None` is the donor expression-for-expression. |
| **per-cycle head snapshots** | `HeadSnapper`, `state_np` / `load_state_np`; called at the **top** of every cycle | the controller is frozen, so the only revisable readouts are `value` and π. Snapshotting before anything in the cycle runs is what makes `heads_{c+1} − heads_c` the revision cycle *c*'s grades produced. |
| **the fixed snapshot probe + the recompute assert** | `HeadSnapper.snap`, `write_trace` | 64 states of the era's own metering set, encoded once: their live fp32 `v`/π every cycle (a free readout trajectory on a fixed state set), and every cycle the same two readouts recomputed from the serialised snapshot and asserted **bit-identical**. |
| **the in-flight cross-tag replay gate** | `replay_delta`, `replay_ref_log`; asserted every `checkpoint_every` in `run_arm` and once at the end | the donor tag is on the same volume under `rhm_practice_conductor/`, so "the instrument moved nothing" can be a **hard assert every 5 cycles** instead of a post-hoc reduction. A divergence costs two minutes, not a GPU-hour. |
| **`schema.md`, written into the outdir** | `SCHEMA_MD` | the dataset carries its own description, including the exact cycle-phase ordering that gives `heads_c` its meaning. |

## The dataset (per arm, on the volume at `/data/rhm_practice_audiation/<tag>/<arm>/`)

`schema.md` in the tag's outdir is authoritative; this is the map.

| path | what |
|---|---|
| `results.json` | the donor's, unchanged — the replay gate's carrier |
| `audiation.json` | shard + snapshot manifests, every recompute check, the replay gate, and the **per-cycle context**: `ms_slots` (position in `ms` → slot id, which changes at a commit), `avail_slots`, `routed`, `k_eff`, `forced`, committed table sizes |
| `decisions/dec_cXXXX_cYYYY.npz` | two flat tables — **TIPS** (`t_x` int8, `t_z` fp16, `t_pi` fp16 in slot space, lineage, terminal `t_succ`/`t_dres`/`t_vfin`) and **CANDIDATES** (`c_parent`, `c_mv`, `c_score` fp32, `c_child`, `c_src`) |
| `snapshots/heads_cXXXX.npz` | `value` + π as fp32 numpy, **every** cycle, plus a terminal snapshot at `c_last+1` |
| `snapshots/plant_cXXXX.npz` | the plant, every `plant_snap_every` (8) cycles — materialisation counterfactuals only; no readout needs it |
| `snapshots/probe_trace.npz` | the fixed probe's live `v`/π, fp32, per cycle, plus the probe configs per era |

**Design calls, stated so they can be disagreed with.**

1. **All surviving tip-steps, not the chosen trajectory.** The unchosen-but-scored candidates
   are the deliberation state the design conversation is about, and the byte math holds: ~112
   compressed bytes per tip-step covers the tip row *and* its share of the candidate table.
2. **π is stored in SLOT space, not `ms` order.** A position in `ms` stops meaning the same
   thing at a commit; a slot is the head's own output vocabulary and is stable across the run.
   Slots off the live action set hold `NaN`.
3. **`z` is fp16, `c_score` and the probe readouts are fp32.** `z` is recoverable exactly from
   `t_x` with a controller rebuild, and it is logged only to spare phase 2 a GPU; the readouts
   phase 2 *differences* are recomputed offline from fp32 snapshots at the same `z`, so the
   fp16 quantisation is common-mode and cancels to first order. Measured residual of the
   offline numpy recompute against the logged π: **1.2e-3 on a logit scale of 3.4**.
4. **Only the practice beam is logged.** The metering beam, the recert's grading beams, the
   auditions, the probes and the ablation battery are instruments; only the practice beam
   carries the agent's own exploration and only it feeds the updates.
5. **The plant is snapshotted every 8 cycles, not every cycle.** No readout needs it; it is
   there so a materialisation counterfactual stays possible.
6. **One arm (`anchor`).** The round's output is a dataset, and the arm to log is the one whose
   trajectory is already certified identical to `cd_s0/anchor` and `as_s0/anchor` — so the log
   is a log *of a known run*. The loop arms would cost 5× GPU and 5× bytes for a variable phase
   2 has no question about yet.

## Cycle-phase semantics (the thing phase 2 must not get wrong)

The controller is **frozen**. Three things learn, all inside one cycle body:

```
cycle c
  SNAPSHOT heads_c            value + pi as they stand BEFORE anything in cycle c runs
  (a) practice beam           plans with (value_c, pi_c, plant_c); RECORDED here and only here
  (b) mining                  no weights move
  (c) plant update, then value update
  (c') pi update
  (d) metering beam           post-update heads; NOT recorded
  (d2..g) panel, recert, commit, era advance   -- change `ms`, never a weight
cycle c+1
  SNAPSHOT heads_{c+1}
```

so `revision(s) = f(heads_{c+1}, s) − f(heads_c, s)` for `f ∈ {value, π}` and any logged `s`.
**The update is per-cycle and batched** — `value_steps` samples from a buffer of every
trajectory state (cap `buf_cap`) plus replay, `prop_train` from the last `prop_buf_cap` solved
pairs across cycles — so the per-datum variation E1 is about is variation **across states**
under one shared update, not "what this datum taught the learner".

## Phase 2 design calls (`fit.py`), stated so they can be disagreed with

The module docstring is authoritative and longer; this is the map. Nothing here is a result.

1. **The datum is a CHOICE STATE** — one surviving tip at a step `t < budget` (6,528 per routed
   cycle, 733,568 over cycles 5–116). A beam takes a *set* of moves at a state, so "the agent's
   own move `u`" is defined as the **argmax-`c_score` child**: the move the agent's own value
   head ranked first there. Whenever exactly one child survives the topk — the modal case — that
   child *is* the argmax, because the topk is global and monotone in the score. `u_kept`,
   `n_kept` and `u_on_solved` are carried per row so the definition's slack is visible.
   Rejected: one row per kept edge (duplicates the target across a parent's survivors for no
   gain) and the child's state (makes `u` retrospective, and the efference copy is prospective).
2. **The state the FM consumes is `(z, root)`** — exactly what both heads read and nothing else.
   The observer twin gets the same information's public form, `(one-hot x, root)`, and has to
   rebuild an encoder; the self side gets the frozen controller's encoding for free, which is a
   reconstruction cost *and* partly an architecture gap, and the reduction says so.
3. **The FM predicts the UPDATE**, `Δ(s) = [Δv, Δπ(56 slots)]`, recomputed offline from the two
   bracketing fp32 snapshots **at the same logged fp16 `z`** so the storage quantisation is
   common-mode. Slots the action set had not yet contained are exactly 0 in both terms.
4. **One deviation from the donor**: the FM's target is per-dimension z-scored on FM-train rows
   and the forecast is un-scaled before any residual is formed. Raw MSE weights the 57 output
   dimensions by raw variance (the measured per-dimension σ spans 0.084–0.49), which would make
   the FM under-fit the value component and hand `r` the value error by construction.
5. **Two nested splits.** FM by **cycle** (primary: every 4th cycle held out, so both sides
   cover all five eras; secondary: the last quarter of the run, the strict-timing form) — `Δ` is
   one shared per-cycle update, and holding out whole cycles is the only arrangement under which
   the forecast could have been on a wire before the grades arrived. Probes and observers by
   **instance** inside the held-out cycles — a cycle's 64 instances are independent freshly
   corrupted problems. Strata are **cycle × step**, which the instance-level probe split is what
   makes possible.
6. **Cycles 1–4 are excluded**: the beam enumerated, π was never read, there is no explore
   injection, so the agency contrast does not exist there.
7. **Targets.** `verr = ȳ − σ(v_c)` where `ȳ` is the fraction of the tip's descendant terminal
   survivors that solved and `n_desc` their count — `(ȳ, n_desc)` is a sufficient statistic for
   the tip's contribution to `value_steps`, which pushes every surviving trajectory's states
   with that trajectory's `succ`. `dprop` is the change in π's log-softmax (over the cycle's live
   slots) at the move the agent chose. Both raw and cycle × step residualised; `dv` and
   `dprop_max` ride along as wiring/diagnostic cells.
8. **Sources**: `r = Δ − FM₂(s,u)`, `delta`, `gap = FM₂ − FM₁`, `r1 = Δ − FM₁`, `p2`, the scalar
   norms `|r|`/`|delta|`/`g`, a cycle × step-preserving `r_shuffled`, and four controls the
   crux needs — `delta_u` (the raw update with the efference copy stapled on and no forecast
   anywhere, because `r` carries `u` inside it and `dprop` is a `u`-indexed coordinate),
   `r_cyc` (the residual against a **calendar-only** forecaster), `u_only`, and `calendar`.
   Two reference rows: `z_state` and `readout0` (the learner's own pre-update readout).
9. **Ladders.** Readout capacity: exact ridge, then MLP-16/64/256. Instrument capacity: FM
   hidden 64/256/1024, i.e. 8%/33%/130% of the two heads' 172K parameters, so the sweep crosses
   saturation. Observer rungs: `x + root`, `+ action`, `+ grade`, capacity-swept, with a
   half-data budget control at the top rung.
10. **Guards**: `_gradcheck()` against central differences (a hard assert — this is why the file
    is allowed to have no autograd), `ens_cos` over independent-seed FM₂ residuals, a
    practice-shaped junk-residual η² of every source against cycle / step / era, the scalar-norm
    negative controls, the shuffled null, and a **within-cycle ceiling** (an FM trained on other
    instances of the same cycles it is scored on — not a forecast, the bound the honest split
    has to be read against).

## Gates

| gate | what | where | status |
|---|---|---|---|
| B (beam) | `audiation.beam_moves` / `beam_moves_prop` reproduce **`conductor.py`'s** bit-for-bit, recorder **off and on**, on a real mixed action set, from a snapshotted RNG state; and the recorder recorded something, with the explore class actually tagged | `audiation.beam_gate()`, run in the pre-setup RNG sandbox | see Runs |
| W (writer) | the shard round trip: terminal join positional and complete, lineage closes, every kept candidate names a real tip, π's live-slot count equals `\|ms\|`, `x` in range, `z` finite | `audiation.audi_preflight` | see Runs |
| S (snapshot) | readouts recomputed from the serialised snapshot equal the live ones **exactly**, every cycle, on 64 fixed states; every `snap_check_every` cycles the round trip goes through the written `.npz` | in-run hard assert, `HeadSnapper.snap` | see Runs |
| R (replay, in flight) | this tag's `anchor` vs `rhm_practice_conductor/cd_s0/anchor` over the 13 `GF_SERIES`, **every checkpoint**, hard assert | in-run, `run_arm` | see Runs |
| R (replay, offline) | the same comparison over every cycle, plus `as_s0/anchor` through it; config-guarded so a `--quick` tag reports NOT COMPARABLE rather than a meaningless FAIL | `analyze_audiation.py` §0 | see Runs |
| the donor's own | `entry_recorder_check`, `gy_soundness`, `policy_gate`, `endo_gate`, RNG-neutrality of the whole pre-setup block | inherited unchanged | see Runs |

## Runs

| tag | what | outcome |
|---|---|---|
| `_audi_preflight` | gate B + gate W at toy sizes, no paid setup | **PASS** — beams 0.000e+00 against the donor with the recorder off *and* on; writer round trip clean; explore class tagged |
| `au_smoke` | `audiation_run --quick --arms anchor`, mechanics + schema | **PASS** — 15 cycles, 43,008 tip-steps / 332,688 candidates / 4.8 MB in 3 shards, 16 head + 3 plant snapshots; gate S exact on all 16 checks; join integrity clean; offline numpy recompute matches logged π to 1.2e-3 (fp16-`z` scale); π moves at every cycle boundary on available slots and is **exactly 0.0** on the 24 slots the action set never contained |
| `au_s0` | **the main run** — the anchor alone at full config, seed 0 | **PASS on every gate.** 116 cycles, 2001 s (0.56 GPU-h, L4). Cross-tag replay `0.000e+00` against **both** `cd_s0/anchor` and `as_s0/anchor` over all 116 cycles, and in flight at every checkpoint. 881,280 tip-steps / 4,937,600 candidates / 111.5 MB in 24 shards; 117 head + 15 plant snapshots, 95.2 MB. Gate S exact on all 117 checks. Join integrity clean. Logging cost **+7.8%** wall (11.36 vs 10.53 s/cycle). Full report: `figures/au_s0/reduction.txt` |
| `au_s0` **phase 2** | `fit.py --tag au_s0 --arm anchor`, seed 0 | **Ran clean, 3,508 s on 4 CPU cores, no GPU.** 733,568 choice-state rows over cycles 5–116; 84 FM-train / 28 held-out cycles; 137,616 probe-train / 45,872 probe-test rows. `_gradcheck` PASS; `ens_cos` 0.956 over 3 independent-seed FM₂ residuals; the FM capacity sweep spans 8%/33%/130% of the two heads' 172K parameters. Outputs: `figures/au_s0/fit/{reduction.txt, fit.json, decode_matrix.png, capacity_ladder.png, agency.png, observer_ladder.png, guards.png}`. **The numbers are in the reduction and are not summarised here — reading them is the writeup's job, with Jasper.** |
| `au_s0` **phase 2, ladder** | `fit.py --tag au_s0 --arm anchor --ladder`, seed 0 | **Ran clean, 492 s on 4 CPU cores, no GPU.** The conditioning-completion ladder: does the update's per-cycle common mode reduce to public, prospective *batch content*, or is it optimizer/composition noise? Five conditioning sets in the same FM₂ class, same cycle-held-out split, same target scaling, same training recipe. Two hard checks passed: `train_fm2` reproduces `train_pair`'s FM₂ **bit-for-bit** (so rung 1 is genuinely reused, not refitted), and the two reconstructed training buffers match the arm's own independently logged `log['prop']['n_pairs']` and live buffer size **exactly (max deviation 0 over all 116 cycles)**. Outputs: `figures/au_s0/fit/{ladder.txt, ladder.json, ladder.png}`. **Numbers in `ladder.txt`; not summarised or interpreted here.** |

#### Two bugs found and fixed during phase 2, recorded because both were invisible

1. **Held-in splits must be by GROUP, not by row** (`_group_split`). A beam's 16 tips at one step
   of one instance descend from one corrupted start and differ by a few moves, so within an
   instance the rows are near-duplicates with near-identical targets. A positional 90/10 of the
   probe-train rows therefore leaves a copy of nearly every held-in row in the fit set:
   validation MSE fell monotonically (0.65 → 0.36) while out-of-sample R² on rows from *other*
   instances collapsed (−0.08 → −0.56). Early stopping on that slice stops at the worst iterate,
   and ridge's λ selection picks too little regularisation. Symptom: every MLP rung reporting
   large negative R² while the exact ridge rung on the same source was fine — i.e. memorisation
   presenting as a capacity fact, in the exact place the matched-capacity ladder is supposed to
   be read. The train/test split was already by instance; every held-in split inside it now is
   too, for both the MLP rungs and the ridge rung.
2. **The half-data budget control took the first half of the row index**, which is the first
   ~14 cycles rather than half the data — a different experiment, not a budget. It now draws
   half the *instances* across all the held-out cycles.

### `au_s0` at a glance (numbers only — the interpretation is phase 2's, with Jasper)

- **Action set**: `\|ms\|` 32 → 48 → 56; the L2 commit fires at c18 and the L3 at c69, and the
  action set the *practice beam* sees grows one cycle later (c19, c70) because the per-cycle
  context is recorded in block (a), before the commit lands later in the same cycle. Phase 2
  must align on `ms_slots`, not on the commit event's cycle.
- **Candidate provenance**: 59.4% pi-proposed, 18.8% enumerated (the four `prop_warmup`
  cycles), 15.0% **explore-injected**, 7.0% forced. On a steady routed cycle the beam scores
  32,640 candidates per cycle at `kk = 5` (4 proposed + 1 explore = 20% explore); the
  enumerated and forced-all cycles (c1–6, c19–20, c70–71) run up to 231,424.
- **Revision magnitudes**, computed offline exactly as phase 2 will (two consecutive head
  snapshots, the logged `z`, live slots only), pooled over 10 evenly spaced routed cycles at
  4,096 states each: `|Δπ|` median **0.126**, p95 0.467, max 3.29; `|Δv|` median **0.058**,
  p95 0.206. Zero at **no** state — π's trunk moves every cycle, as it must.
- **Two things worth a second look before phase 2 leans on them.** (i) c71 — the first cycle
  after the L3 commit — has a `|Δπ|` p95 of 1.25 against ~0.40 everywhere else, and c27 a max
  of 1.84 against ~1.0; the revision distribution is not stationary across the run and is
  visibly larger just after a commit and early in era 1. (ii) `|Δv|` decays roughly 3× from
  era 1 (0.19 at c5) to the late eras (~0.04), so a per-datum signal pooled across the whole
  run pools across a changing scale.
- **Storage precision, measured not assumed.** The offline numpy recompute sits 1.8e-3 from
  the logged π on a logit scale of 4.70; perturbing the logged `z` by half a fp16 ulp per
  element moves π by 5.5e-4. Same scale — the residual is `t_z`'s storage precision, not a
  recompute error, and it is common-mode in any difference taken at a fixed `z`.

## Reproduce

```bash
cd experiments/            # MODAL_PROFILE=chromatic

# gates B + W, toy sizes, no paid setup
modal run rhm/practice/audiation/audiation.py::audi_preflight

# mechanics + schema at quick scale (the replay ref is off: --quick is a different substrate)
python3 rhm/practice/audiation/launch_detached.py --fn audiation_run --tag au_smoke --quick \
    --arms anchor --replay-ref "" --snap-check-every 5
python3 rhm/practice/audiation/analyze_audiation.py --tag au_smoke --fetch

# the main run: the anchor alone at cd_s0's exact literals, instrument on
python3 rhm/practice/audiation/launch_detached.py --fn audiation_run --tag au_s0 \
    --eras "1:25:48,2:12:40,3:6:12,4:3:9,5:1:7" --arms anchor \
    --budget 8 --g-budget 482 --n-corrupt 1 --mine-cap 8 --gy-level 4 --entry-rec \
    --span-min-hold 128 --collect-task-matched --tm-episodes 8192 \
    --recert-every 5 --n-aud 192 --probe-every 8 --n-rt 384 --n-score 256 \
    --sil-cv 0.15 --lp-min-drop 0.10 --seed 0 \
    --replay-ref rhm_practice_conductor/cd_s0

python3 rhm/practice/audiation/analyze_audiation.py --tag au_s0 --fetch --write
```

The offline replay gate needs the donor's reference fetched locally at
`../conductor/figures/cd_s0/anchor/results.json` (and, optionally,
`../assay/figures/as_s0/anchor/results.json`); both are already on disk in this repo.

Phase 2 needs no Modal, no GPU and no torch — only the fetched `figures/au_s0/anchor/` bytes:

```bash
cd experiments/
python3 rhm/practice/audiation/fit.py --tag au_s0 --arm anchor       # ~1 h on 4 CPU cores
python3 rhm/practice/audiation/fit.py --tag au_s0 --smoke            # ~2 min wiring pass
python3 rhm/practice/audiation/fit.py --tag au_s0 --ladder           # ~8 min; needs fit.json
```

It caches the per-datum table at `figures/<tag>/fit/table_v2_r0.npz` (~690 MB, gitignored);
`--no-cache` rebuilds it (~12 min) and the `v2` in the name is the schema key — bump it when
`build_table`'s columns change, or a stale cache is loaded silently.

## Children

None. Phase 2 (the `SlotFM` port and the decode matrix) landed **in this node** as `fit.py`
rather than as a child directory: it adds no substrate and no run, it reads this node's own
dataset, and splitting it out would have put the recipe and the bytes it is a recipe for in
different places. Its outputs live beside the dataset's, under `figures/<tag>/fit/`.

---

# E1b — the per-datum-credit arm

Appended by the E1b implementer. Everything above describes phase 1 (`au_s0`, the pooled
anchor's dataset) and phase 2 (`fit.py`); this section describes the arm added on top of them
and nothing else in the node changes because of it.

**Asked in**: the coordinator's E1b brief, authorized by Jasper, after phase 2 found the
anchor's update dominated by the calendar (η²(Δ~cycle) = 0.438, |r|/|Δ| > 1 at every capacity,
within-cycle ceiling FM at cos 0.141). The diagnosis under discussion: the inner loop's credit
is **batch-pooled** — one shared update per cycle, sampled from buffers — so "the learner's own
next state given its own action" barely exists as a function of (state, action), because the
update's cause sits outside any prospective conditioning set. E1b makes credit per-datum, which
is the condition under which the FM residual becomes the vector form of the δ performance
error. It also gives the practice arc its first pooled-vs-per-datum credit contrast, so the
arm's own clocks are a real secondary readout.

## The knob, in one paragraph

A **datum** is one graded trajectory: its `budget+1` lineage-resolved states, the `budget`
moves it took (in slot space — the arity-2 second argument), and its terminal grade. The
`anchor` pools these: `value_steps` samples `n_grad × value_batch` states out of a 100k-state
buffer plus a replay pool, and `prop_train` samples `prop_steps × prop_batch` pairs out of a
60k-pair history. `perdatum` spends them **one at a time, in arrival order** — per datum, one
value step on its own states with its own grade as the label, and one π step on its own
(state, slot) pairs if it solved. Same content, different granularity of credit. `schema.md`
in the tag's outdir carries the full statement.

## Design calls (every one of them mine; disagree with them here)

1. **Datum = one graded trajectory**, not one state and not one instance. It is exactly the
   unit the donor pools: `push` writes every trajectory state labelled by that tip's `succ`,
   and `prop_pairs` writes every pair of a *solved* trajectory. So the treatment changes the
   granularity and nothing about *what* is credited.
2. **`pd_n = 128` data per cycle**, uniformly subsampled without replacement from the cycle's
   1024 graded tips, processed in beam (= arrival) order. Every update in the arm is therefore
   attributable **and logged** — there is no unlogged data quietly moving the model. 128 × 9 =
   1152 state-updates per cycle against the anchor's 1024 sampled states, so data volume lands
   close by construction rather than by matching.
3. **What is matched: the per-cycle gradient budget, as `n_steps × lr`.** Under Adam the
   per-step parameter displacement is ≈ lr almost regardless of batch size or gradient scale,
   so `Σ lr` is the honest "how far did the learner move this cycle" invariant. Value is matched
   **exactly**: `4 × 3.00e-5` → `132 × 9.091e-7 = 1.200e-4`. **What is not matched, and cannot
   be: batch composition** — that is the treatment. **What is not matched, and is reported:**
   samples seen, and π's step count, because π steps only on data that *solved*, so its
   per-cycle budget follows the solved count (`pd_prop_nominal = 32` is the count the lr is set
   against: `24 × 3.00e-4` → `36 × 2.00e-4`).
4. **Replay is kept, but SEPARATED rather than mixed.** Mixing replay into each per-datum batch
   would make every per-datum revision partly caused by replay samples — destroying the
   attribution the arm exists to create. Dropping it risks drift (value's guard is the clean
   replay pool; π's is its own history across commits). So it runs as a separate pooled
   **maintenance pass** after the per-datum steps, on history only: value `pd_maint_v = 4`
   steps at `replay_frac = 1.0` (the clean pool, nothing else), π `pd_maint_p = 4` steps on
   `pbuf` **as it stood before this cycle's pairs were appended** (the append is moved after
   the maintenance call for this arm). No credit for the cycle's own grades is ever pooled,
   and every logged revision has exactly one cause.
5. **The plant stays pooled and unchanged.** It feeds no readout, so leaving `finetune_generator`
   alone isolates the knob — the coordinator's recommendation, and I agree with it.
6. **Revisions are logged, not snapshotted.** ~15k update events × a state dict each is
   gigabytes. Instead the readout **difference** around each event is recorded at three sets:
   the datum's **own** states, the era's **fixed probe** (`pd_probe_n = 32`), and a
   **reference** set of `pd_ref_n = 32` same-cycle tip states this pass does *not* step on —
   the spillover term.
7. **The differences are formed on device in fp32 and stored in fp32.** The per-datum lr is
   ~1e-6, so a revision is small; fp16's ulp at a logit of 5 is 4e-3. Storing endpoints in fp16
   and differencing offline would have been pure round-off. Only the baseline is fp16, where it
   is context rather than signal. (This is the one place E1b could have been silently junk.)
8. **Reported with RMS over states × slots, not max.** The three sets hold different numbers of
   states, and a max over more states is mechanically larger — which is exactly the artifact
   that made the first smoke read a spillover ratio > 1.
9. **`anchor` runs beside `perdatum` in the same tag.** The setup is built once and shared, so
   the in-tag anchor costs only its own cycles, and it is simultaneously the pooled comparator
   and the **inverse gate**. Its decision log is switched off (`--log-arms perdatum`): it would
   be a byte-for-byte duplicate of `au_s0`'s, which the replay gate proves rather than assumes.

## The inverse gate

`per_datum` is `False` in the base config; every line E1b adds is behind it, and `pdrng` (its
only source of randomness — which data, which reference states) is drawn *only* inside
`per_datum_pass`. So with the flag off the arm's torch and numpy streams are the donor's, and
the `anchor` arm of an E1b tag must replay `cd_s0/anchor` **bit-identically**. Asserted in
flight every checkpoint and again offline.

## Code and record delta

| where | what |
|---|---|
| `audiation.py` `RevRec` / `flatten_rev_cycle` / `RevWriter` | the revision log and its shards |
| `audiation.py` `per_datum_pass` | the treatment itself; draws only from `pdrng` |
| `audiation.py` `run_arm` blocks (c-pd), (c), (c') | the treatment wired in, the maintenance pass, and the moved `pbuf` append — all behind `per_datum` |
| `audiation.py` `ARMS["perdatum"]`, `TWIN`, `AUDIATION_ARMS_E1B` | the arm |
| `audiation.py` `SCHEMA_MD` | the E1b section of the outdir's `schema.md` |
| `analyze_audiation.py` §4, §5 | the per-datum data report and the two arms' clocks side by side |
| `<tag>/<arm>/revisions/rev_cXXXX_cYYYY.npz` | one row per update event (schema in `schema.md`) |
| `<tag>/<arm>/perdatum.json` | per-cycle step counts, losses, live lrs, and the sampled/reference indices |

`fit.py` is untouched.

## Runs (E1b)

| tag | what | outcome |
|---|---|---|
| `au_smoke_pd` | `--quick --arms anchor,perdatum --log-arms perdatum`, mechanics + schema | **PASS.** 1,920 events over 15 cycles (9.7 MB); revision↔decision join 1920/1920 with the grade agreeing on all; finiteness clean; **exact-zero invariant PASS both ways** (Δπ exactly 0 on 100% of the 741 events where no π step fired, > 0 on 100% of the 1,179 where it did, Δv > 0 on 100% of all events); spillover RMS ratio 0.71 (probe) / 0.80 (reference); Δv 9.7× larger on solved than unsolved data. **Inverse gate at quick scale**: `au_smoke_pd/anchor` vs `au_smoke/anchor` = `0.0` on all 13 series. |
| `au_s1` | **the main E1b run** — `anchor` + `perdatum` at full config, seed 0 | **PASS on every gate.** 3320 s (0.92 GPU-h, L4) for both arms. **Inverse gate `0.000e+00`** against `cd_s0/anchor` *and* `au_s0/anchor` over all 116 cycles (in flight every checkpoint and again offline) — every line E1b adds is inert when disabled. 14,848 update events over 116 cycles, 50.1 MB in 24 shards; revision↔decision join 14848/14848 with the grade agreeing on all; two-sided exact-zero invariant PASS; decomposition at the shared probe states closes to a 4.8% median residual (= the maintenance pass, by construction). The `perdatum` arm also carries the full phase-1 dataset (881,280 tip-steps, 117 head snapshots). Full report: `figures/au_s1/reduction.txt` |
| `au_s1` **the refit** | `fit.py --tag au_s1 --arm perdatum --e1b`, seed 0 | **Ran clean, ~27 min on 4 CPU cores, no GPU.** The analyst's side of E1b, in `fit.py`'s third mode. Q1 a conditioning ladder ((s,a) → +grade → +full own lineage → +own pre-update readout) in the same FM₂ class and the same cycle-held-out discipline as phase 2, on three target blocks (`dv1`, `own57`, `own57@pi`) because `e_did_pi == e_succ` **exactly** — so on the joint target `+grade` is handed "Δπ is identically zero here" on 79% of rows, which is wiring, not skill. Q2 the analytic-δ identity for the value component, measured rather than assumed. Q3 incremental decodes over a baseline holding *every* free scalar (per-state δ, its mean, the grade, **and log‖Δᵢ‖**) — the earlier ratio target was dropped because `own_rms` sat on both sides of it. Q4 provenance conditioning, the observer twin, and the nulls. Plus a whole-cycle comparability row: the phase-2 FM class on this arm's own boundary snapshots. Outputs: `figures/au_s1/fit/{reduction.txt, e1b.json, provenance.npz}`. **Numbers in the reduction; not summarised or interpreted here.** |

### `au_s1` at a glance (numbers only — the interpretation is the analyst's, with Jasper)

- **Cost of the treatment**: 12.1 s/cycle for `perdatum` vs 10.7 s/cycle for the in-tag
  `anchor` (+13%). Revision log 50 MB, ~3.4 KB/event compressed.
- **The matched budget held.** Value exactly: `4 × 3.00e-5` → `132 × 9.091e-7 = 1.200e-4` every
  cycle. π's realised step count came out at a **median 26/cycle** (range 2–55), i.e. **0.74×**
  the anchor's Σlr — below the `pd_prop_nominal = 32` the lr was set against, because 20.6% of
  the sampled data solved. The nothing-solved fallback **never fired** (0/116 cycles).
- **The update is still far from local.** Spillover (RMS at other states / at the datum's own
  states, medians): Δv **0.43** at the fixed probe and **0.50** at the same-cycle reference set;
  Δπ **0.60** / **0.61** over π-stepping events. Per-datum credit moved locality only partway.
- **Δv is *larger* on data that did NOT solve** (1.744e-3 vs 1.323e-3 RMS, 0.76×) — the
  opposite direction from the quick smoke, where the nothing-solved fallback dominated.
- **By provenance** (explore-injected moves on the datum's own lineage; all 14,848 data
  resolved through the decision tables): Δv falls monotonically with the number of explore
  moves, 2.101e-3 at 0 explore moves → 1.332e-3 at 7, while Δv spillover *rises* monotonically
  0.471 → 0.601. Solve rate peaks at 2 explore moves (24.4%) and falls off either side.
- **Decomposition check**: the 128 per-datum revisions of a cycle plus the maintenance pass sum
  exactly to the cycle-boundary snapshot difference at the shared probe states; the maintenance
  share is a **4.8% median** (p95 10.2%) of the boundary revision, so the revision tables see
  ~95% of what a cycle does to the readouts.
- **The arms' own clocks** (secondary readout, single seed): `perdatum` commits L2 one cycle
  later (c19 vs c18) and L3 **seven cycles earlier** (c62 vs c69); ends with a larger L2 table
  (12 vs 11) and a **smaller L3 table (8 vs 12)**; fewer next-level observations (L3/L4 at
  support 51/55 vs 62/68); mean solved/cycle 213.6 vs 236.2; era-mean `e` 0.503/0.639/0.719/
  0.905/0.976 vs the anchor's 0.513/0.620/0.671/0.850/0.949 — i.e. ahead in era 1, behind in
  eras 2–5. π's proposal mass ends **less** consolidated: macro mass 0.374 vs 0.484, L2 mass
  0.133 vs 0.250, entropy 2.816 vs 2.610.

[^private]: Not mirrored: this link points to a document in the private lab repo (the roadmap, the queue, an unrun spec, reading notes, or a conversation). See the top-level README for what is held back and why.
