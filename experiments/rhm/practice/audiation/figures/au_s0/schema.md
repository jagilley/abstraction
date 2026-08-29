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

## What is NOT logged

  * the METERING beam (block d), the recert's grading beams, the auditions, the probes and the
    end-of-run ablation battery. Only the practice beam carries the agent's own exploration and
    only it feeds the updates; the rest are instruments.
  * the world. No oracle, no rules, no true tables enter these tables — `t_succ` and `t_dres`
    are the two oracle readouts, and they are the grades the arm already pays for and learns
    from, not side information.
