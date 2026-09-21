# DESIGN — `soundboard`: the outcome error in the plant's own weights, in the loop

Decisions, and every withdrawn diagnosis kept beside its correction. The brief is
[`SPEC.md`](SPEC.md); the machinery index is [`FILES.md`](FILES.md); the interpretation, discussed 2026-09-20, is in the aliquot writeup ([`README.md`](README.md) at the aliquot level); the facts live in the reduction under `figures/` and here.

**Up**: [`../DESIGN.md`](../DESIGN.md) (`aliquot`) · [`../../DESIGN.md`](../../DESIGN.md)
(`sotto_voce`) · **Sibling**: [`../duplex/DESIGN.md`](../duplex/DESIGN.md) (the same question
offline, on banked dumps).

Donor: [`../aliquot.py`](../aliquot.py) at its `al_s1`/`al_s2` head, untouched. Every addition in
`soundboard.py` is marked `# [soundboard]`, every knob defaults off, and with every knob off the
file is `aliquot.py` and therefore `sotto_voce.py` and `enharmonic.py` (gate G-F).

---

## §0 The name

A piano's **soundboard** is the body the bridge is glued to and everything else is mounted on: the
strings move it, and what you hear is the board. `aliquot` is named for strings that are never
struck; `duplex` for the short string lengths on the bridge that hold them; both are things
*mounted on* the board. This node is the one that changes **the board** — the plant's own weights —
while it is playing, so it is named for the board. `duplex` §0 claimed "the body" for the offline
form of the question; this is the same body, in the loop, and the two names are the two halves of
one instrument on purpose.

---

## §1 What is new, and what an in-loop version can answer that the offline one cannot

`aliquot` measured, in the loop, that a ridge-logistic readout of the plant's pooled hiddens over
the substituted configuration reaches held-out AUC 0.7591 / 0.7282 on the live trunk (seeds 0 / 2)
against 0.7158 / 0.7403 on a never-trained trunk of the same architecture, while the mirror — a
net trained end to end on outcomes — reads 0.99 on the same rows. The reading offered in
discussion: a plant trained only on masked infilling represents what *prediction* needs, and a
linear map of that is not a map of the verdict, so the outcome error has to reach the trunk's
**weights** rather than only the readout's.

`duplex` tested that offline, on `overtone`'s banked dumps, and found the direction: 400–1600 steps
of an outcome head beside the infill loss raised the duplicate-free filed read from 0.63 to
0.73 / 0.78 and seed 0's probe read from 0.80 to 0.83, at +0.008 nats of held-out infill loss
against a matched infill-only control; outcome-alone destroyed the plant inside 100 steps and
bought nothing. Two things about that round are the reason this one exists, and `duplex` §1 states
both before its numbers:

1. **Every `duplex` arm read rows an UNSHAPED plant wrote.** A loop whose plant is shaped writes
   differently from cycle 1, so the probe's job is a different job on different rows, and nothing
   offline is a prediction of the shaped loop's AUC.
2. **Nothing offline says anything about the CHOOSER.** The quantity `sotto_voce` and `aliquot`
   care about is repair accuracy under a grader, and a grader's AUC on banked rows is upstream of
   that by a whole loop.

What this node adds, and what only an in-loop version has: the plant is also the **executor**. The
composed chooser's prior is the DP's max-sum over the book scored by `core.feature_head(pooled)`,
and `pooled` is exactly what the shaping moves. So the outcome-only arm is not merely a cost
measurement — it is the question of what a practice loop *does* when its own chooser's prior is
computed through a plant that has left the masked-infill family. `duplex` could not ask that.

---

## §2 What is NOT changed

The readout in the grader's seat is `aliquot`'s, to the line: `VoProjBank`, the same features
(`overtone`'s `post_slot` through the `mask_block` idiom), the same per-goal design matrix, the
same IRLS solve, the same ridge grid selected on the same train-internal split, the same recompute
of the buffered rows through the live trunk at every refit, the same filed probability. The diet
(src 0 alone), the schedule, the dedup, the probe channel's substitution and draw, the
instrument-only world verdict, the zero bill, the composed chooser and the clock yoke are all the
donor's. **The only difference from the banked `al_pj_yk` is that the plant has heard outcomes.**

That is the control-variables statement of the round and it is what makes gate S-4r (§5) the gate
the round rests on: the shaped arms and `al_pj_yk` share a stream key, a seed and a yoke source, so
they are bit-identical until the first shaping step fires.

---

## §3 The shaping, decided

### §3.1 The head: `duplex`'s form, with the root added

```
[ pooled.mean(1) (96) ; pooled[span].mean (96) ; root_emb(r) (32) ; blk_emb(blk0) (32) ]
    -> Linear(256, 256) -> GELU -> Linear(256, 1)
```

`duplex::build_out_head`'s MLP at the same hidden width (256) and the same embedding width (32),
over the same two pools the linear probe reads, with **one** change and one thing kept:

- **The root is in.** `duplex` fit its probe per slot on rows whose root it never saw, so its head
  had a slot embedding and nothing else. In the loop the world's verdict is "the root's possible
  set contains r\*" — a *match* between the parse and the goal, which `aliquot` §2.2 argues at
  length is why the readout in the grader's seat is per root (`vo_pj_root="inter"`). A head that
  cannot see the root is being fitted to "how solvable is this configuration averaged over goals",
  which is a different target, and shaping the trunk for it would be shaping it for a question the
  reader does not ask. The slot embedding is kept beside it as the in-loop analogue of `duplex`'s
  (`blk0`, the slot's first block, which is what a bank row carries).
- **No candidate embedding**, kept from `duplex` §3.1: the candidate is already rendered into the
  input, so its content can reach the head only through the trunk, which is the object under test.

Minted with `build_outcome`'s discipline (save the shared RNG state, construct, restore,
re-initialise every parameter from a dedicated generator at `normal(0, 0.02)` with biases zeroed),
so the head's existence costs the shared per-arm stream nothing and gate G-F stays closed.

**It reads the state the projection reads.** The same masked input — one block masked *outside* the
span, `vo_pj_mask`'s idiom — for two reasons at once: the plant has never seen a fully unmasked
input (`ratchet/macros.py::parse_features`; `calr_s0` measured 0.63 block accuracy on a visible
block read unmasked), and the shaping must write into the very state the grader is about to read
rather than into a neighbouring one. `duplex` read unmasked because *its* licence was reproducing
`overtone`'s table cell for cell; this node has no such licence and takes the in-distribution read.
Gate **S-2** asserts that `sb_shape_input` plus the head's pooling reproduce
`VoProjBank._features`' 192-wide vector **to the bit**, because it is a second implementation of
the donor's arithmetic (the first lives inside a `no_grad` the shaping cannot inherit) and a second
implementation is a defect waiting to happen.

### §3.2 The yield currency, defined before the paid arms

For a configuration `x` and the run's current target level `L = min(max_macro_level, era_level+1)`
— the mining step's own `_tgt`, so during era k the currency is the level-(k+1) chunks the learner
is forming:

```
label(x) = 1[x solved] * (1/n_spans) * #{ disjoint level-L spans of x whose KEY is at support
                                          in the learner's own level-L miner }
```

where the parse is `MC.parse_features(shared["reader"], x)` — the mining step's own call, so the
features are the features the miner counted — and the key is the **miner's own** key
(`sb_miner_keys`). Four decisions, each with its reason:

- **The miner's own keying, not the raw tuple.** With a quotient in play (every arm here:
  `merge_mode="ledger"`) the committable miners are `quotient.ClassMiner`s whose `counts` are keyed
  by the pair of the halves' **class ids** at level `L-1`. A label computed against the raw level-1
  tuple would miss on every key and read identically zero — a silent nil, which is the shape of
  defect this lineage keeps finding (`voicing` Q3's #8). Gate **S-5**'s perturbation is exactly
  that, and it goes red.
- **The whole configuration, not the era's cell.** A key is node-independent, so every level-L span
  of the configuration can be read against the counts. Reading only the era's own cell would have
  made the label nearly constant: `voicing` Q0 measured the executor writing **one** token class on
  up to 99.6% of its frontier calls, so that cell's key is almost always the same key and almost
  always at support, and a near-constant target gives the trunk no gradient at all.
- **At support, not "did it advance a key".** The credit form (`run_arm`'s own `_credit`: the
  observation moved a key that was not yet banked) is the other candidate the brief offers and is
  **rejected**: it is 1 only until a key banks, so its base rate collapses over the run and the
  head's target would be dominated by the run's clock rather than by the configuration.
- **Zero on an unsolved piece.** Only solved configurations are mined (`mine_from="chosen"` over
  `ps > 0.5`), so an unsolved piece contributes nothing to the next level's build whatever it is
  made of. The currency therefore **dominates** the solve verdict — zero wherever the verdict is
  zero, graded above it — which is what makes the two shaped arms a contrast in the *target* and
  not in its support. It is also why this arm is not simply "the verdict with extra steps": on the
  solved rows the label is the fraction of the piece that is made of banked bricks, and nothing
  about the solve verdict distinguishes those.

Computed **at push time**, in `push_experience`, against the counts the miner holds *then* —
before this cycle's own observations are added a few lines later in the loop. That is the only
causally clean reading of the brief's "at observe time": a label recomputed later would be a
function of counts the future filled in. The label is banked as a column (`yq`), so the column is
the record and the reduction never recomputes it.

The target level **moves with the era**. That is honest — the run's own currency moves — and it is
logged per cycle (`tgt_level`) so the head's target is never inferred. It is also the one thing
about this arm that has no offline analogue: `duplex` had no era.

**The per-probe instrument**, three columns, consumed by nothing (gate M-1's discipline, the same
as the world's verdict beside it):

| column | parse | counted against |
|---|---|---|
| `yqe` | the learner's own reader | the learner's own miner — the currency the head was trained on |
| `yqt` | the learner's own reader | the **true** level-L table |
| `yqo` | `MC.exact_features` (the oracle) | the **true** level-L table |

The brief asks for two; three costs nothing and separates the reader's parse error (`yqt` vs `yqo`)
from the miner's incompleteness (`yqe` vs `yqt`), which the two-column form conflates. Every
projection arm carries them, not the currency arm alone, because the columns are how the three arms
are compared in **one** currency.

### §3.3 The losses, the weights, the rate and the step budget

- **Weight 1, added in the plant's own optimizer step.** `vo_critic_terms`' arrangement one organ
  over: the critic's BCE is "added to the plant's loss in the SAME optimizer step — the arc's
  convention, so the treatment buys no extra steps", and `span_lam = 1.0`. `duplex` §3.3 chose the
  same and did not tune it. The head's parameters join `gopt` through `add_param_group`, exactly as
  the critic's do.
- **The rate is `gen_lr = 1e-4`**, the rate at which the loop already continues this plant, so the
  step counts are readable against the loop's own budget and nothing was tuned.
- **The step budget is the run's own.** `gen_steps = 20` per cycle over ~200 cycles ≈ 4,000 shaping
  steps — which is 2.5× `duplex`'s largest arm (`@1600`). Stated because it matters: `duplex`'s
  `both@1600` was its arm of record and this run's plant gets more than twice that, spread over the
  whole run instead of applied to a finished one.
- **The batch is 256**, the run's own `batch_size` and the infill term's own. `duplex` drew 1,920
  outcome rows per step (32 per slot × 30 slots × 2 doors) because its buffers were per-slot; the
  bank here is a flat sample of the learner's *sentences*, so there are no slots to spread over and
  the natural match is the infill batch. A departure from `duplex`, stated.
- **The warm-up is 64 rows** (`vo_sh_min`), far below the readout's `vo_om_min = 512`. A
  1,737-column ridge needs hundreds of rows; a 256-wide MLP does not, and the plant should hear
  outcomes from as early as there is an outcome to hear. It also makes the liveness gate reachable
  at preflight scale, where the bank holds a few hundred rows over a handful of cycles.
- **The head is not clipped**, matching the critic, which is also unclipped in the donor. The plant
  is clipped at `_train_generator`'s 1.0 on every arm alike, so the clip is never an arm difference.

### §3.4 What "the infill loss off" means in a loop, and what it cannot mean

`duplex`'s `out@N` had one term to remove. The loop's plant step has four: the masked-infill
cross-entropy, the corridor head's self-imitation at `span_lam`, the head's record target on a
governed slot, and the critic's BCE. **Only the first is removed**, and the reason is the
control: the corridor head's term and the critic's term are the *substrate's* and not this round's,
so removing them too would make the arm two knobs from its twin instead of one.

The term is **detached** rather than deleted, which keeps `gloss` meaning the plant's own infill
loss on the batch it was handed — an instrument this arm needs more than any other — while
contributing no gradient to any parameter, and leaves `core.feature_head` frozen exactly as it
would be if the term were absent (nothing else in the step touches it). A step in which the loss
ends up with no graph at all — the outcome-only arm's first cycles, before the bank fills — is
**counted** (`n_nograd`) rather than asserted, so a dead step is readable in the arm file.

**One consequence of the outcome-only arm, found at paid scale and worth stating.** Its plant is
trained by the outcome term alone, and the term does not fire until the bank holds
`vo_sh_min = 64` rows — which needs *macro* writes, which need a committed level. On the anchor's
clock that is **c48** at seed 0 and **c59** at seed 2. So `sb_so_yk`'s plant is **frozen through
its whole first era** while every other arm's trains on infill, and its `n_nograd` counter (a step
whose loss ended with no graph at all) runs to roughly `48 x gen_steps`. That is the honest reading
of "the infill loss off during the run" and not a defect — but it means the arm is not "the same
plant minus one term" in era 1, it is "no plant training in era 1", and any era-1 number from it
should be read as such. The clock yoke holds the ladder fixed regardless, which is what makes the
later eras comparable at all.

**The emission head is never reached by the outcome error, on any arm.** `SN.trunk` returns
`pooled` *before* `core.feature_head`, so the outcome gradient flows into the token and position
embeddings, the encoder and the final norm and into nothing else of the plant. This is `voicing`
Q1 made structural rather than incidental: a calibration loss *on* the emission distribution drove
the head's mass to the base rate, corridor parity collapsed to 0.20–0.33 against a 0.50 firing
gate, and the arms never tested their question. Gate **S-1b** asserts no `feature_head` parameter
accumulates a gradient from the outcome term.

### §3.5 The three arms

| arm | `vo_sh_target` | `vo_sh_infill` | what it is |
|---|---|---|---|
| `sb_sv_yk` | `solve` | on | the world's verdict in the trunk's weights, beside the plant's own training |
| `sb_yd_yk` | `yield` | on | the abstraction grader's own currency in the trunk's weights |
| `sb_so_yk` | `solve` | **off** | `duplex`'s `out@N` in the loop, where the DP scores through the shaped plant |

All three are `al_pj_yk` in every other respect, clock-yoked to the seed's banked anchor
(`vo_s3:voi3_dp` at seed 0, `vo_s3d2:voi3_dp` at seed 2) — the same yoke `sotto_voce`'s three
mirror arms, `voicing` Q3b's floor and ceiling, and `aliquot`'s two projection arms all ride, so
every comparison in the reduction is at a matched clock. Banked and not re-run: the floor
(`voi3b_comp_yk`), the ceiling (`voi3b_comp_pr_yk`), the mirror / committee / hybrid (`so_mg_yk`,
`so_cg_yk`, `so_hy_yk`), and `aliquot`'s `al_pj_yk` / `al_rt_yk`.

---

## §4 The instruments

Per cycle, on every arm that carries the series (the three shaped arms and, at preflight, the
unshaped `al_pf_pj` twin):

- **The plant's own fingerprint** (sum of |parameter|, float64). Logged every cycle and not only at
  a refit, which matters: `VoProjBank` records `trunk_sig` only when the readout refits, and that
  needs 512 bank rows — so it does not exist in the first cycles or at preflight scale at all,
  where the liveness gate has to read it.
- **The outcome head's own within-batch BCE, AUC and base rate**, and the plant's own infill loss
  on the arm that has switched it off.
- **The world-model diagnostics** (`sb_wm_diag`, `duplex::wm_diag` moved inside the loop): the
  held-out masked-infill cross-entropy and fill accuracy on **fresh** corpus windows at a **fixed**
  window/mask plan drawn once per arm (so a movement between cycles is the plant and not the draw),
  split by how many blocks are masked; and the block head's level-1 parse accuracy against
  `exact_features`, read with one block masked (`parse_features`' idiom) and unmasked. The corpus is
  `_sample_pool(rules, 8192, s, 777002)` — `duplex`'s own eval seed, distinct from the run's
  `train_seed = 1` and from its probe-clean stream — and `plant_holdout = 0` in these configs, so
  the regenerated corpus is the run's own distribution.
- At **checkpoints** (`vo_wm_dp_every = 50` cycles, plus once at end of arm): the **nested DP
  parse** at L2–L5 over the **true** tables, span masked, recovered level-1 features against the
  exact ones. `duplex` §4.3's replacement for `nested_phase`, for the reason recorded there:
  `nested_phase` estimates a period-`s^k` boundary from a next-token reader's per-column entropy
  profile, and the practice plant is a masked **block** infiller whose block grid is aligned to the
  code grid by construction, so there is no phase to recover. This is the round's altitude
  instrument. It runs at checkpoints and not every cycle because the top rung's true table has
  262,144 rows.

The parse and DP cells are **oracle** instruments — they compare against `exact_features` and the
true tables — which the brief licenses. They enter no loss, no buffer and no bill.

Also per cycle, all of them the donor's and unchanged: the executor's misfire and corridor parity
(section [B]/[M] of the reduction), which matter more here than anywhere upstream because the DP
scores through the shaped generator; the projection's held-out AUC on experience, its drift between
refits, and the unmasked variant beside it; and the bill.

**New on the readout, both from `duplex`'s findings:**

- **The level, beside the ranking.** `duplex`'s filed gain was mostly *context* rather than
  candidate — the pre-write read rose almost as much as the post-write one — which says shaping
  writes "how solvable is this state" into the global pool. That is a statement about the level, so
  the level is measured on the bank's held-out rows: `hold_pbar` (the mean filed probability)
  against `hold_base`, the Brier score, and a 10-bin calibration error, beside `hold_auc`.
- **The cross-cycle duplicate share.** `duplex` §2.6 established that `aliquot`'s bank dedups the
  sibling-slot duplicate away and is open to the *other* channel: the hold key is a **coin flip**
  (`_append`) against a **per-cycle** dedup, so the same `(configuration, root)` recurring in a
  later cycle is appended again with a fresh draw and can straddle the split carrying the same
  verdict — and it could not size it, because `aliquot`'s arm files do not carry the bank's buffer.
  The draw is **kept as it is**, for comparability with the banked arms; what is new is
  `dup_share` (the fraction of held-out rows whose `(configuration, root)` is also a training row),
  `dup_same_y` (of those, the share whose twin carries the same verdict), computed every tenth
  refit and at end of arm.

**The dumps**, at the end of every arm (`vo_dump`): `vo_rows.npz` + `vo_rows_meta.json` (the
per-slot filed and probe buffers with the DP's own score of the candidate through the final core,
the held-out code, and the world's verdict on the probe rows — `voicing::ov_dump_rows`' column
naming, so a reader written for `overtone`'s dumps can be pointed here), `vo_bank.npz` (the bank's
whole buffer: `x, r, y, yq, blk, spn, tid, h, u0, src`), and `vo_heads.pt` (the critic, the plant's
core, the shaping head, the readout's fitted `w/mu/sd/lam`). `aliquot` banked none of these and its
own child had to read `overtone`'s older dumps instead of its parent's, which is why every `duplex`
arm reads rows an unshaped plant wrote.

**The trajectory id** (`tid`) is banked in the recorder and carried into the bank, so a
leave-one-trajectory-out split is constructible offline. `duplex` §2.5 closes with the sentence
this implements: *"the dump carries no trajectory or instance id, so a leave-one-trajectory-out
split is not constructible from the banked data; it would need the recorder to bank the id."* A
trajectory is `(cycle, beam row)` and the id is `cycle * 10**6 + row`, asserted against the beam's
own size.

---

## §5 The gate table

Everything `aliquot` gated still holds and is re-run: G-F, V-1, V-4A/B, V-4c/V-4d, V-5, V-6/V-6b,
Y-1, M-1 … M-5 and their run-level forms, P-1 … P-5 and P-2r/P-3r/P-4r, on every projection arm
including the three shaped ones. New, each shown to fail on a deliberate perturbation before being
reported ([`gates/falsify.py`](gates/falsify.py), **100/100**):

| gate | claim | the perturbation it goes red on |
|---|---|---|
| **S-1** | the outcome error **reaches the trunk**: one term, one backward, one step, and the plant's fingerprint moves | the pooled state is **detached** (the head still learns; the trunk does not) · the term is **zeroed** before it reaches the plant's loss |
| **S-1b** | it never reaches the **emission head**: no `feature_head` parameter accumulates a gradient from the outcome term | (asserted as a construction; `SN.trunk` returns `pooled` before `feature_head`) |
| **S-2** | the shaping reads the state the **projection** reads, to the bit | the mask is dropped (an unmasked read the plant has never seen and the grader never takes) · the **span** pool is dropped, so the head reads the global pool alone |
| **S-3** | it is **sandboxed**: no shared torch stream moves and no bank row, hold draw or bootstrap draw changes | the shaping draws from the shared stream |
| **S-4** | the **target** is the one the arm named | a "yield" shaper falls back to the verdict where the label is missing |
| **S-5** | the yield currency is the **miner's own** keying, and zero on an unsolved piece | the label is keyed by the raw level-1 tuple (identically zero on a quotiented miner) · the label ignores the solve verdict |
| **S-6 / S-6b** | … and through the **real** `merge.LearnedQuotient` and `quotient.ClassMiner`, whose class id is a representative *tuple* and not an integer: 1.0 on a configuration made of an at-support span, 0.0 on an unobserved one, 0.5 on one that is half each — and the same **under a merge**, which is what makes the halves' level index load-bearing | the raw-tuple keying again, on the real objects · the halves keyed one level up, which a merge-free fixture cannot see |
| **S-1r / S-2r / S-3r** | run-level, off the arm file: the term fired, the plant's fingerprint moved, the target matches the config, the yield currency is not constant, the diagnostics exist and the DP ran | a shaped arm that never stepped · a shaped arm whose plant never moved · a constant currency · a yield arm with no labels · no DP · an arm naming `vo_wm` with no diagnostics at all |
| **S-4r** | **the twin form, and the gate the round rests on**: the shaped arm's plant differs from the **unshaped twin's**, cycle by cycle, with its own non-vacuity half (the shaped arm must have stepped) | the two plants are identical · the shaped arm never stepped |

`vo_preflight_gates` runs S-1r … S-3r on the three shaped twins and on `al_pf_pj`, then S-4r and
`vo_gate_v6b` on each (shaped, unshaped) pair — so both "the plant differs" and "the run differs"
are asserted, and the first is the stronger statement about *where* the difference entered.

The **GPU-side gate block** (`proj_gates_gpu`) now runs S-1 … S-5 on an L4 beside P-1 … P-5, for
the reason `aliquot` §7.3 records: its second preflight defect was a host tensor matmul'd against a
device tensor and was invisible to every CPU gate. The shaping builds tensors
(`sb_shape_input`'s mask positions, the span weights) and hands them to a device forward, so it
gets a gate on the device too.

---

## §6 Cost, and the shape of the run

**The arms fan out.** `aliquot` ran its two arms one after another inside one container, so a
two-arm tag cost twice the wall clock of its longest arm for the same GPU-hours, and its own
DESIGN §6 names the fix. Every arm here is clock-yoked to a **banked** anchor, so no arm waits on
another: `sweep` is a CPU coordinator that `.starmap`s `voicing_run` over the arms, one container
each, and merges the results into one tag directory. The detach rule is intact because the fan-out
happens inside a Modal function and never from a local entrypoint.

**What the fan-out costs**, stated rather than hidden: each container pays the shared setup
(`_spiral_shared` trains the controller, the plant, the reader and the stale value head once per
container), so three containers pay it three times — roughly 0.3 GPU-h against roughly 2.5 h of
wall clock saved per seed.

**What the shaping costs**, per cycle: `gen_steps = 20` extra trunk forwards and backwards over 256
rows at 64 tokens, which is small beside the plant's own 20 steps over 256 windows. The world-model
diagnostics add 8 batches of 256 windows through `block_logits`, one parse read over 2,048
configurations, and — at checkpoints only — the nested DP. The yield arm additionally pays one
reader forward per cycle over the pushed rows and, where `vo_yq_probe` is on, one reader forward and
one exact parse per governed slot per cycle plus Python-level key counting; era 1 is its worst case
(target level 2, 16 spans per configuration).

**Measured, preflight (`sb_pf1`, five twins at `budget=2`, ~27 cycles each).** ~2 h of wall clock
in one container for all five, which is the shape the paid tags do not have. Peak GPU memory was
never a constraint; the arm files are ~1.18 GB each and the bulk is `log["miner"]` at preflight's
`preflight_seed_miner=True` (the miners are seeded with half the true table), inflated ~8× by
`write_results`' `indent=2` — a preflight-only artifact, since the paid tags pass
`entry_rec_cap=4096` and `aliquot`'s paid arm files came out at 205–345 MB.

**Measured, paid.** `sb_s1` and `sb_s2`, three arms each, one container per arm: **2.25 h per arm**
(8096–8142 s), so ~13.5 GPU-h for the six, plus two CPU coordinators and ~3 GPU-h of gates, G-F and
the preflight — about **17 GPU-h** for the round. The two tags ran concurrently, so the wall clock
was 2.25 h rather than 13.5. Against `aliquot`'s 1.4–1.8 h per projection arm the shaping costs
**+25 to +60%**, and the accounting is: the shaping adds 3,060 / 2,840 trunk forward-and-backward
passes over 256 rows (against the plant's own ~4,020 infill steps, so up to +75% on the
plant-training line), and the world-model diagnostics run **every cycle** — 8 batches of 256 windows
through `block_logits` plus two parse reads over 2,048 configurations, 201 times. The brief's ~1.3
GPU-h per arm was `sotto_voce`'s figure and was already stale at `aliquot`; the honest number for a
shaped arm on this substrate is 2.25 h.

**Where a later round can take it back**, stated because it is cheap and this round did not:
`vo_wm_dp_every` already keeps the DP to five checkpoints, but the infill/parse block has no such
knob and runs at cycle granularity for a series that moves smoothly over ~50 cycles. Every fifth
cycle would cost a fifth and lose nothing readable. `memory=12288` was never approached.

---

## §7 Defects, in order, each beside its correction

**§7.1 Gate S-3r asserted its content only where the content existed.** The first version wrote
`if wm: assert ...` — so an arm that logged **no** world-model diagnostic at all passed, and the
round's whole cost side could have been silently absent. The falsification harness reported the
gate **BLIND** on exactly that perturbation. This is the inertness-shaped blindness `voicing` V-6
exists for (a treatment that is not wired is the most inert thing there is) reappearing in a gate
written the same week, which is the argument for running the harness before reporting a gate rather
than after. Correction: an arm whose config names `vo_wm` must carry the series; one that does not
(the unshaped `al_pf_pj` twin, which predates the instrument) is not asked to. Harness **95/95**.

**§7.2 A graph-less term crashed the gate instead of tripping it.** One of the harness's
disconnections zeroes the shaping term before it reaches the plant's loss (`t.detach() * 0.0`), and
`sb_shape_check` called `term.backward()` unconditionally — so the perturbation raised a
`RuntimeError` out of autograd instead of the gate reporting "the trunk did not move". A gate that
crashes is not a gate. Correction: `has_graph` is recorded and the backward is guarded, so the
perturbation is *seen* rather than fatal. The same guard is in the runner
(`finetune_generator_span`), where a step with no graph is the outcome-only arm's first cycles and
is counted (`n_nograd`).

**§7.3 The donor's own falsification perturbations broke on the two new buffer columns, and
correctly.** `_append` and `push_experience` gained `yq`/`tid` and `label_fn`; three harness
monkeypatches reproduced the donor's signatures verbatim and raised `TypeError` at the first M-2
perturbation. Nothing about the gates had changed, which is what made it obvious. Corrected by
forwarding the new keywords in the perturbations — and recorded because the *reason* the harness
caught it is that it exercises the real call path rather than a re-derivation of it.

**§7.4 Gate P-4r's strong form was conditioned on the wrong slice, and the preflight found it.**
`aliquot`'s P-4r asserts, where the readout's **held-out** slice carries both classes, that
`n_refit > n_degenerate` — i.e. that the readout fit the state at least once rather than only the
base rate. `sb_pf_so` tripped it: `hold_base = 0.0179` (one solved row in 56) with **all 21 refits**
degenerate. Read against the code that is not a defect and not a claim about the wiring. The
degenerate branch is entered *iff* the **training** slice is single-class, so what the gate caught
is a coin-flip split on a substrate that solves almost nothing putting the run's one solved
experience row on the held-out side — the state `aliquot` §7.2 itself describes three times
("at PREFLIGHT sizes the substrate … solves nothing, so every experience row's verdict is 0").
`aliquot` never hit it because both of its preflight arms had `hold_base = 0.0` exactly;
`sb_pf_so`, whose plant has no infill term and writes differently, solved one more instance.

Correction, and it makes the gate *right* rather than weaker. `VoProjBank` now records
`fit_base`, the **training** slice's own unclamped base rate, on every refit — `base_rate_fit` is
clamped to `1/(n+2)` and so can never say "all one class" exactly. P-4r's strong form is asserted
on that: **an arm whose training slice ever carried a contrast must have fit the state**. Where it
never did, the fact is recorded with the reason and not asserted. The harness shows both halves:
the defect (a contrast in the training slice and every refit still degenerate) trips, and the
preflight-scale fact does not. The strong form still binds on the paid tag, where ~20% of filed
writes solve (`voicing` Q1's 0.18–0.23).

**§7.5 Gate S-2r's "the yield currency is not constant" was false at preflight by construction, and
for the same reason.** The currency is zero on an unsolved piece by design (§3.2), so on a
substrate that solves nothing it is identically zero — and `sb_pf_yd`'s was: `label_mean = 0.0`,
`label_pos = 0` over 276 labelled rows. That made `sb_pf_sv` and `sb_pf_yd` **bit-identical**
(same plant fingerprint series to the last digit, same BCE, same diagnostics), because the two
heads saw the same all-zero target. Correction, `aliquot`'s own treatment of
V-6b(proj_vs_rand) verbatim in shape: the arm now counts how many of its labelled rows solved
(`n_label_solved`), the assertion fires only where that is non-zero, and the fact is recorded with
its reason where it is not. A `V-6b(verdict_vs_yield)` pair check is added to the preflight on the
same terms — measured always, asserted only where the two targets can differ. **The contrast
between the two targets is therefore a claim on the paid tag and not on the preflight**, and the
mechanism check that the two targets are read from different columns at all is gate S-4, which
asserts it on a synthetic bank where they disagree.

**§7.6 An observation, not a defect: the nested DP parse does not discriminate at preflight scale.**
`dp_parse` came out identical to six decimals at every checkpoint and across all three shaped arms
(L2 0.2559, L3 0.2500, L4 0.2881, L5 0.2827) while `infill_ce` and `parse_mask1` both moved. The
reading: with a 40-step plant the max-sum DP's argmax over the true table is dominated by the table,
so the recovered features do not move with the plant. `duplex` measured the instrument reading 0.39
on the banked (12,000-step) trunk and 0.15 on a never-trained one, so it *does* discriminate between
plants that differ enough; these three do not, and the preflight's value (0.256) sits between
`duplex`'s two. Recorded here so that a flat `dp_parse` on the **paid** tag would be read as a
finding about the shaped plant rather than as this same artifact.

*(further entries appended as the round produces them)*

---

## §8 Reproduction

```bash
cd experiments/            # MODAL_PROFILE=chromatic
modal run rhm/practice/voicing/sotto_voce/aliquot/soundboard/soundboard.py::voicing_gates    # 27 CPU gates
modal run rhm/practice/voicing/sotto_voce/aliquot/soundboard/soundboard.py::falsify_remote   # 95/95
modal run rhm/practice/voicing/sotto_voce/aliquot/soundboard/soundboard.py::proj_gates_gpu   # P-1..P-5 and S-1..S-5 on an L4
python3 rhm/practice/voicing/sotto_voce/aliquot/soundboard/launch_detached.py --fn fidelity_smoke --tag sb_gf1
python3 rhm/practice/voicing/sotto_voce/aliquot/soundboard/launch_detached.py --fn preflight --outdir-tag sb_pf1 \
    --arms "voi3b_pf_src,al_pf_pj,sb_pf_sv,sb_pf_yd,sb_pf_so"
python3 rhm/practice/voicing/sotto_voce/aliquot/soundboard/launch_detached.py --fn sweep --tag sb_s1 --seed 0
python3 rhm/practice/voicing/sotto_voce/aliquot/soundboard/launch_detached.py --fn sweep --tag sb_s2 --seed 2
modal run rhm/practice/voicing/sotto_voce/aliquot/soundboard/soundboard.py::preflight_gates --outdir-tag sb_s1
python3 rhm/practice/voicing/sotto_voce/aliquot/soundboard/fetch_compact.py --tag sb_s1 --fetch --replace
python3 rhm/practice/voicing/sotto_voce/aliquot/soundboard/analyze_soundboard.py --tag sb_s1 --yoke-src vo_s3:voi3_dp \
    --bank vo_s3:voi3_dp,vo_s3b:voi3b_comp_yk,vo_s3b:voi3b_comp_pr_yk,so_s1:so_mg_yk,so_s1:so_cg_yk,so_s1:so_hy_yk,al_s1:al_pj_yk,al_s1:al_rt_yk
python3 rhm/practice/voicing/sotto_voce/aliquot/soundboard/mk_seedtable.py
```

Artifacts on `rhm-scaling-data:/rhm_practice_soundboard/<tag>/`; compact local mirrors and
reductions under `figures/` (`sb_seedtable.txt` is the two-seed table of record).
