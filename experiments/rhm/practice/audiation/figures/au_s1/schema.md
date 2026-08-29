# `audiation` record schema — E1 phase 1

Written by the run, into the tag's own outdir, so the dataset carries its own description.
Nothing here is interpreted; this is what the bytes mean.

## What learns, and when — the ordering that gives `heads_c` its meaning

The CONTROLLER IS FROZEN in this substrate (`build_shared` trains it once; `run_arm` only
deep-copies the plant and the value off the shared setup). Three things learn, all of them
inside one cycle's body and nowhere else:

    cycle c
      SNAPSHOT  <- heads_c: value + pi as they stand BEFORE anything in cycle c runs
      (a) practice beam   plans with (value_c, pi_c, plant_c); the per-decision tables are
                          recorded here and ONLY here; tips graded -> succ, dres; every
                          trajectory state pushed to the value buffer `buf`
      (b) mining          the vocabulary is mined from what was solved; no weights move
      (c) plant update    finetune_generator(...)  <- plant_c   -> plant_{c+1}
          value update    value_steps(...)         <- value_c   -> value_{c+1}
      (c') pi update      prop_train(...)          <- pi_c      -> pi_{c+1}
      (d) metering beam   PERFORMANCE conditions, post-update heads, NOT recorded
      (d2) shadow panel / the loop's decision point
      (e..g) recert, commit, era advance  -- these change the ACTION SET `ms`, never a weight
    cycle c+1
      SNAPSHOT  <- heads_{c+1}

So for any state `s` logged in cycle c:

    revision(s) = f(heads_{c+1}, s) - f(heads_c, s),   f in {value, pi}

is exactly the revision cycle c's grades produced at `s`. Two facts to hold onto:

  * THE UPDATE IS PER-CYCLE AND BATCHED. `value_steps` samples `n_grad` batches from a buffer
    holding EVERY trajectory state (cap `buf_cap`) plus replay; `prop_train` samples
    `prop_steps` batches from `pbuf`, which is the last `prop_buf_cap` (state, root, slot)
    pairs from SOLVED trajectories across cycles. So the revision at `s` is not "what cycle
    c's grade at `s` taught the learner" — it is the effect at `s` of one batched update whose
    data is dominated by, but not limited to, cycle c. The per-datum variation E1 is about is
    variation ACROSS STATES under one shared update.
  * THE FINAL SNAPSHOT is indexed `c_last + 1`, so `heads_{c+1} - heads_c` exists for every
    logged cycle including the last.

## `decisions/dec_cXXXX_cYYYY.npz` — two flat tables

Row order inside a step is BATCH-MAJOR, TIP-MINOR, which is the order `tips.reshape(B*W, T)`
uses and therefore the order `grade` returns `succ`/`dres` in: the terminal join is positional.

TIPS — one row per surviving beam entry at each step `0..budget`.

| key | dtype | meaning |
|---|---|---|
| `t_cycle` | int16 | the cycle |
| `t_inst` | int16 | instance index in the cycle's `n_pr` fresh instances |
| `t_step` | int8 | beam step; `0` is the corrupt start, `budget` the graded tip |
| `t_tip` | int16 | index within the beam at that step |
| `t_parent` | int16 | tip index at `step-1` this entry descends from; `-1` at step 0 |
| `t_mv` | int16 | index into `ms` of the move that produced it; `-1` at step 0 |
| `t_root` | int8 | the root target `r*` |
| `t_x` | int8 (n, T) | the configuration |
| `t_z` | float16 (n, D) | `controller.state(x)` — the FROZEN encoder state the heads read |
| `t_pi` | float16 (n, n_slots) | pi's logits at that tip, in SLOT space; NaN off the live action set, and all-NaN on warmup cycles where the beam enumerated and no proposal was read |
| `t_vfin` | float32 | the beam's terminal re-score; NaN except at `t_step == budget` |
| `t_succ` | int8 | exact possible-set membership; `-1` except at `t_step == budget` |
| `t_dres` | float32 | exact residual edits to a valid derivation; NaN except at the tip |

CANDIDATES — one row per SCORED CHILD at each step `0..budget-1`. This is the deliberation
state: the moves the beam materialised and valued and then did not keep.

| key | dtype | meaning |
|---|---|---|
| `c_cycle`, `c_inst`, `c_step` | int16/int16/int8 | join to TIPS |
| `c_parent` | int16 | the tip index at `c_step` this child was expanded from |
| `c_mv` | int16 | index into `ms` |
| `c_score` | float32 | `value(controller.state(child), r*)` — the number the topk ranked |
| `c_child` | int16 | the tip index it became at `c_step + 1`, or `-1` if the topk dropped it |
| `c_src` | int8 | provenance: 0 enumerated (no proposal), 1 pi's top-k, 2 forced (move too new for the head), 4 explore-injected |

`c_src == 4` is where the agency lives: `prop_explore=1` draws one move per tip uniformly from
what pi did NOT propose, on a private numpy stream, so the chosen action is not a deterministic
function of the state. That is the `g > 0` condition teacher-forced NTP structurally lacks.

## `snapshots/`

  `heads_cXXXX.npz`   fp32 numpy, keys `value.<param>` and `prop.<param>`. Rebuild the module
                      and `load_state_dict`; both heads read `(z, root)` and nothing else, so
                      any readout at any logged state is recomputable with no GPU and no
                      controller.
  `plant_cXXXX.npz`   the plant, every `plant_snap_every` cycles. Present for materialisation
                      counterfactuals; NO readout needs it.
  `probe_trace.npz`   `cycle`, `era`, `v` (n_cycles, n_probe), `pi` (n_cycles, n_probe,
                      n_slots), all fp32, plus `probe_x_e{era}` / `probe_r_e{era}`. The probe
                      states are the era's OWN metering set, so they change at an era boundary:
                      a cycle-to-cycle difference is only meaningful WITHIN an era.

## `audiation.json`

The shard manifest, the snapshot manifest, every recompute check, and the PER-CYCLE CONTEXT
(`cycles`): `ms_slots` (position in `ms` -> slot id — this is what makes `t_mv`/`c_mv`
resolvable, and it changes at a commit), `ms_level`/`ms_node`, `avail_slots` (the mask pi's
softmax was taken under), `routed`, `filter_on`, `k_eff`, `forced`, and the committed table
sizes. `slot_offsets` gives the (level -> first slot) layout.

## E1b — the `perdatum` arm (present only in a tag that runs it)

### The knob

`anchor` pools credit: one `value_steps` call samples `n_grad x value_batch` states from a
100k-state buffer plus a replay pool, and one `prop_train` call samples `prop_steps x
prop_batch` pairs from a 60k-pair history. One shared update per cycle, whose cause is a batch.

`perdatum` spends the cycle's grades ONE TRAJECTORY AT A TIME, in arrival order. A DATUM is one
graded trajectory: its `budget+1` lineage-resolved states, the `budget` moves it took (in slot
space), and its terminal grade. Per datum:

    readout BEFORE at (own states, fixed probe, same-cycle reference states)
    one value step on the datum's own states,  label = its own succ
    one pi step on the datum's own (state, slot) pairs, IF it solved
    readout AFTER at the same three sets

That is the same content the donor pools (`push` writes every trajectory state with the tip's
`succ`; `prop_pairs` writes every pair of a solved trajectory) at a different GRANULARITY.

### What is matched, and what is not — stated, not implied

  MATCHED  everything that generates trajectories: the crank, the era ladder, the beam, the
           mining, the commit policy, the recert, the caps, the seed, the RNG streams, and the
           arm's torch stream (it twins onto the anchor's).
  MATCHED  the per-cycle GRADIENT BUDGET, as `n_steps x lr`. Under Adam the per-step parameter
           displacement is ~lr almost regardless of batch size or gradient scale, so `sum(lr)`
           is the honest "how far did the learner move this cycle" invariant. Value:
           `n_grad x value_lr_online` -> `(pd_n + pd_maint_v) x pd_value_lr`, exactly.
  NOT MATCHED, and cannot be: BATCH COMPOSITION. That is the treatment.
  NOT MATCHED, reported: SAMPLES SEEN per cycle, and pi's step count — pi steps only on data
           that SOLVED, so its per-cycle budget follows the solved count. `pd_prop_nominal` is
           the count the lr is set against; the realised count is in `perdatum.json`.
  UNTOUCHED the PLANT. It is still `finetune_generator`, pooled, unchanged. It feeds no
           readout, so leaving it pooled isolates the knob.

### Replay: kept, but SEPARATED rather than mixed

Mixing replay into each per-datum batch would have made every per-datum revision partly caused
by replay samples — destroying the attribution the arm exists to create. Dropping replay
entirely risks drift (value's anti-drift guard is the clean replay pool; pi's is its own
history across commits). So replay is kept as a SEPARATE pooled MAINTENANCE pass, run after the
per-datum steps, on HISTORY ONLY:

  * value: `pd_maint_v` steps at `replay_frac=1.0` — the fixed clean replay pool, nothing else.
  * pi:    `pd_maint_p` steps on `pbuf` as it stood BEFORE this cycle's pairs were appended
           (the append is moved to after the maintenance call for this arm).

No credit for the cycle's own grades is ever pooled, and every logged revision has exactly one
cause. The maintenance pass is not logged as a revision event; its losses are in
`perdatum.json` per cycle, so its size relative to the per-datum steps is checkable.

### `revisions/rev_cXXXX_cYYYY.npz` — one row per update event

`pd_n` events per cycle. Shapes are dense: `n_own = budget+1`, `n_prb = pd_probe_n`,
`n_ref = pd_ref_n`.

| key | dtype | meaning |
|---|---|---|
| `e_cycle`, `e_order` | int16 | the cycle, and the datum's position in arrival order |
| `e_inst`, `e_tip` | int16 | which trajectory (joins to the decision tables' `t_inst`/`t_tip`) |
| `e_succ`, `e_dres`, `e_root` | int8/f32/int8 | the datum's own grade and target |
| `e_did_pi`, `e_n_pairs`, `e_fallback` | int8 | whether the pi step fired, on how many pairs, and whether the cycle hit the nothing-solved fallback |
| `e_vloss`, `e_ploss` | float32 | the losses at that step |
| `e_own_x`, `e_own_z`, `e_own_a` | int8 / f16 / int16 | the datum's states, their frozen encoder states, and THE ACTIONS IT TOOK in slot space — the arity-2 second argument |
| `e_own_v0`, `e_own_pi0` | float16 | the readout BEFORE the step, at the own states (context) |
| `e_own_dv`, `e_own_dpi` | **float32** | the revision at the datum's own states |
| `e_prb_dv`, `e_prb_dpi` | **float32** | the revision at the era's fixed probe |
| `e_ref_dv`, `e_ref_dpi` | **float32** | the revision at same-cycle states this pass did NOT step on — the SPILLOVER term |

**Why the differences are stored, and in fp32.** The per-datum lr is ~1e-6, so a revision is
small; fp16's ulp at a logit of 5 is 4e-3, which is far larger. Storing the two endpoints in
fp16 and differencing them offline would be pure round-off. The difference is therefore formed
on device in fp32 and stored in fp32; only the baseline is fp16, where it is context.

### `perdatum.json`

Per cycle: `n_data`, the realised `n_value_steps` / `n_pi_steps`, `n_ref` / `n_prb`, the
per-datum and maintenance losses separately, the live learning rates, and `sel` / `ref` — the
flat tip indices this cycle stepped on and used as reference. Plus the run-level matched-budget
record (both arms' lrs and step counts).

### The inverse gate

`per_datum` is False in the base config; every line E1b adds is behind it, and `pdrng` is drawn
only inside `per_datum_pass`. So with the flag off the arm's torch and numpy streams are the
donor's, and the `anchor` arm of an E1b tag must replay `cd_s0/anchor` BIT-IDENTICALLY over
every cycle — which is asserted in flight every checkpoint and again offline. That is the proof
that all new code is inert when disabled.

## What is NOT logged

  * the METERING beam (block d), the recert's grading beams, the auditions, the probes and the
    end-of-run ablation battery. Only the practice beam carries the agent's own exploration and
    only it feeds the updates; the rest are instruments.
  * the world. No oracle, no rules, no true tables enter these tables — `t_succ` and `t_dres`
    are the two oracle readouts, and they are the grades the arm already pays for and learns
    from, not side information.
