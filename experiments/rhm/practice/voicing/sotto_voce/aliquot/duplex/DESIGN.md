# DESIGN — `duplex`: the outcome error in the trunk's weights, read offline

Decisions, and every withdrawn diagnosis kept beside its correction. The brief is
[`SPEC.md`](SPEC.md); the machinery index is [`FILES.md`](FILES.md); the interpretation, discussed 2026-09-20, is in the aliquot writeup ([`README.md`](README.md) at the aliquot level); the facts live in the reduction under `figures/` and here.

**Up**: [`../DESIGN.md`](../DESIGN.md) (`aliquot`) · [`../../DESIGN.md`](../../DESIGN.md)
(`sotto_voce`) · **Protocol donor**:
[`../../../overtone/analyze_dump.py`](../../../overtone/analyze_dump.py) `::post_write_probe`,
imported in spirit and reproduced cell for cell as gate G1, not edited.

---

## §0 The name

A piano's **duplex scale** is the aliquot system: the short lengths of string on either side of
the speaking length, set by the bridge's own bearing points so that they ring in sympathy. It is
the part of the instrument's *body* that the aliquot strings are mounted on — so a child of
[`aliquot`](../DESIGN.md) that asks whether the reading has to change the body rather than the
reader is named for the body. The second reading is the arm of record: a duplex is two things
running at once, which is what "outcome BCE beside the masked-infill loss" is.

---

## §1 What is being asked, and what an offline read cannot answer

`aliquot` measured, in the loop, that a ridge-logistic readout of the plant's pooled hiddens over
the substituted configuration reaches held-out AUC 0.7591 / 0.7282 on the live trunk (seeds 0 /
2) against 0.7158 / 0.7403 on a never-trained trunk of the same architecture — i.e. **the trained
representation's margin over the random one is +0.043 at one seed and −0.012 at the other**,
while the mirror, a net trained end to end on outcomes, reads 0.99 on the same rows
(`figures/al_seedtable.txt` (e)). The reading offered in discussion: a plant trained only on
masked infilling represents what *prediction* needs, and a linear map of that is not a map of
what the *outcome* needs; the outcome error therefore has to reach the trunk's weights.

`logit_reading/orbitofrontal/shaped/` is the same intervention one arc over, and it separates
into three arms — task only, task plus next-token, next-token only — precisely because "the
trunk moved" and "the trunk moved toward the outcome" are different facts. This node is that
three-arm shape on the practice plant, run **offline on banked checkpoints** rather than in the
loop.

**What that buys.** The whole thing is one L4 for minutes instead of three yoked arms at ~1.3
GPU-h each, and every arm reads the *identical rows*, which no in-loop comparison can offer
(each in-loop arm generates its own experience).

**What it cannot answer, stated before the numbers.** The rows in `vo_rows.npz` were produced by
a loop whose plant was **never** shaped. So:

1. Every arm here reads a distribution of contexts and writes that the *frozen* plant's
   behaviour created. A loop whose plant is shaped would write differently from cycle 1, and the
   probe's job would be a different job on different rows. Nothing here is a prediction of the
   shaped loop's AUC.
2. Nothing here says anything about the **chooser**: the quantity `sotto_voce` and `aliquot`
   care about is repair accuracy under a grader, and a grader's AUC on banked rows is upstream of
   that by a whole loop.
3. The banked critic was trained against the frozen trunk. Its column is a fixed reference, and
   where it is recomputed through a *shaped* trunk (reported, because it is free) it is a
   mismatched pair, which is a fact about readers fed by a moving learner and not a critic
   ablation.
4. The world-model diagnostics are the cost side of the ledger, measured on fresh corpus
   windows. They say what the shaping cost the plant's *predictive* competence. They do not say
   what that cost would do to the loop, because in the loop the plant is also the executor.

---

## §2 The protocol, decided

### §2.1 The read of record is `overtone`'s, unmasked — because the gate is

Features are `SN.trunk`'s pooled per-block hiddens over the post-write configuration (the write
rendered into the masked context at the slot), pooled two ways and concatenated: the mean over
all blocks and the mean over the written span. That is `overtone::post_write_probe`'s `post_slot`,
192 wide at `state_dim = 96`, and it is read **unmasked**, exactly as overtone read it.

The alternative was `aliquot`'s read (§2.1 there): mask the first block outside the span, because
the plant is trained with `mask_min = 1` and has never seen a fully unmasked input
(`ratchet/macros.py::parse_features`), and `aliquot` measured that masking mattered in the loop
(holdAUC 0.7591 masked against 0.7470 unmasked on the live trunk; 0.7158 against 0.6411 on the
random one). **The unmasked read was chosen as the read of record anyway**, for one reason: the
node's gate is that overtone's table comes back cell for cell, and a different read is not that
table. The masked read is computed on the identical rows and reported as its own column
(`postm`), so the size of the choice is a number here too rather than an argument.

One thing to hold in view when comparing `postm` to `aliquot`'s masked column: the two are not
the same estimator. `aliquot` fits a per-root design matrix (`onehot(root) ⊗ features`, 1737
columns) with a ridge selected per refit on a train-internal split, on **experience rows only**,
against a live trunk. This node fits overtone's 193-column probe at a fixed ridge of 1e-2 on both
doors against a static trunk. A sign disagreement between the two masked columns is therefore not
a contradiction, and neither is quoted as the other's replication.

### §2.2 One estimator, gated against the donor's

`overtone::_fit_probe` calls `voicing::ov_irls`, a numpy Newton solve. The bulk here is ~2,600
fits (11 trunks × 60 buffers × 4 reads, per seed), so the estimator was moved into torch float64
(`duplex.py::irls_t`) — same Newton step, same `clamp(-30, 30)`, same `+1e-6` on the working
weights, same `ridge * I` on **every** column including the intercept (the donor penalises the
intercept and that is kept rather than improved), same `1e-9` early stop.

**Gate G0** fits one real buffer with both and reports `max |Δw|` and `max |Δscore|`. Measured on
the smoke: `max |Δw| = 2.6e−11`, `max |Δscore| = 2.9e−6`, AUC identical to nine decimals. Nothing
in this node is a second estimator by intent.

### §2.3 Gate G1, the reproduction, is the node's licence

Two of the eight-to-eleven trunks are objects overtone already read: the banked core, and a
never-trained core of the same architecture. The random one is minted under **overtone's own**
`torch.manual_seed(20260915)`, after the trained core is loaded, in overtone's order — not
`aliquot`'s `20260918`, which is a different draw and would only look like the same control.
`analyze_duplex.py` parses `overtone/figures/ov_s0b_dump.txt` and `ov_s2_dump.txt` and compares,
slot by slot and diet by diet:

| this node | overtone's column |
|---|---|
| `frozen` / `pre` | `pre_slot` |
| `frozen` / `pmean` | `post_mean` |
| `frozen` / `post` | `post_slot` |
| `frozen` / `mlp` | `mlp` |
| `frozen` / `dp` | `dp` |
| `rand` / `post` | `rand_slot` |

On the smoke's four slots every one of those cells agreed to the printed three decimals
(e.g. `filed 2:0` 0.585 / 0.586 / 0.619 / 0.625 / 0.495 and `rand_slot` 0.676, all matched). The
full-scale gate is section (G1) of the reduction. **If G1 is not clean, nothing below it is a
re-read of overtone's protocol and the shaped rows are not comparable to its floor.**

### §2.4 The held-out split, and why there is no leak

`VoRecorder.hold_code` is `(((obs + 1) · powers).sum % 1000003 · 48271) % 100` — a function of
the **context alone**. So:

- within a buffer, a context recurring across cycles cannot straddle the split (the property the
  critic's own audit relies on);
- **across** the two doors at one slot, the same context gets the same code, so the filed
  buffer's training rows and the probe buffer's held-out rows cannot share a context either;
- the shaping trains on training rows only (`code >= 100·vo_critic_hold = 10`), so no held-out
  context reaches the trunk's weights, the outcome head's weights, or the probe's fit.

What the shaped arms *do* have is a representation fitted to the training rows' verdicts. That is
the object under test, not a leak, and the in-loop version has exactly the same property.

### §2.5 The exact-duplicate channel, and why there are two tags

**Found by reading the output, then measured.** For a **filed** row the post-write configuration
*is* the trajectory's final configuration, while `hold_code` hashes the **pre-write** context —
which differs from slot to slot because a different span is masked. So one trajectory that wrote
at several slots contributes several rows carrying the **same post-write input** and the **same
verdict**, and the audit's split can put them on opposite sides. Anything that reads the
post-write input therefore sees some held-out inputs verbatim, with their labels.

Measured on the banked rows (the run's own instrument, `overlap` in the JSON, and independently in
numpy on the local dumps first):

| diet | seed 0 | seed 2 |
|---|---|---|
| **filed** held-out rows whose exact post-write configuration is also a training row | **5994 / 22937 = 26.1%** | **5542 / 22474 = 24.7%** |
| … of which the training twin carries the same verdict | 5736 (96%) | 5269 (95%) |
| **probe** | 53 / 19774 = **0.27%** | 127 / 19558 = **0.65%** |

Probe rows are almost clean because a probe row's post-write configuration carries a *substituted*
class graded for itself, so two probe rows of one trajectory are different objects with different
labels.

Three things follow and all three are stated rather than fixed silently:

1. **It is inherited, not introduced.** `vo_critic_terms` splits by exactly this `hold_code`, so
   the in-loop critic's filed audit AUC has this channel too. `du0`'s numbers are therefore the
   ones directly comparable to it, and that is why `du0` is the tag of record for the
   **reproduction**. (§2.6 corrects what was first claimed about `aliquot`'s `hold_auc`.)
2. **It cannot inflate `frozen` or `rand`**, which never see a verdict at all. It can inflate any
   arm whose *representation* was fitted to verdicts — i.e. exactly the arms whose gain is the
   claim — and it can inflate the linear probe's own filed fit on every trunk alike.
3. **So the node runs twice.** `du0` is the protocol as overtone and the loop have it. `du1` is
   bit-identical except for one filter: every training row whose post-write configuration appears
   among the held-out rows of **any** buffer is dropped from the shaping diet **and** from the
   probe's own fit. It drops 2.7% / 2.0% of the training rows, so nothing is lost to power, and
   `du1`'s filed column is the one with no exact train/test duplicate in it. `du1` is therefore
   **not** a reproduction of overtone and its G1 section is expected to differ; the size of that
   difference is itself the measurement of what the channel was worth.

What `du1` does **not** close: a held-out *probe* row can still be a near-twin of a filed training
row (the same trajectory with one class substituted). Those are different configurations with
different labels, and the exact-hash filter does not touch them. The dump carries no trajectory or
instance id (`obs, write, y, code, dp, unif` and nothing else), so a leave-one-trajectory-out split
is not constructible from the banked data; it would need the recorder to bank the id.

---

### §2.6 A withdrawn claim about `aliquot`, and what the code actually implies

**Withdrawn.** §2.5 first said that `aliquot`'s `hold_auc` "inherits the duplicate channel",
meaning the sibling-slot mechanism measured above. **Read against the code, that is wrong**, and
the two organs inherit *different* channels — which is the more interesting fact and is why the
sentence is corrected rather than deleted.

`aliquot`'s readout trains on `VoOutcomeBank`'s src-0 buffer, filled by `push_experience`
([`../aliquot.py`](../aliquot.py) L6057). Its rows are keyed on the trajectory's **final
configuration** and deduplicated within the push by `(configuration, root)`:

```python
keyed = np.concatenate([x.numpy(), r.numpy()[:, None]], axis=1)
vv    = keyed.view(np.dtype((np.void, ...))).ravel()
first = np.sort(np.unique(vv, return_index=True)[1])
x, r, y = x[first], r[first], y[first]
```

So **one tip contributes one row however many macro slots it wrote** — the buffer is a sample of
the learner's *sentences*, not of its write calls. The sibling-slot mechanism that produces this
node's 26% cannot arise there at all. That half of the claim does not survive.

**But the split there is weaker in the other direction, and this is live.** `_append`
(L6035–6052) draws the hold key as a **coin flip**, not as a function of the configuration:

```python
u = torch.from_numpy(self.rng.random((n, self.K)).astype("float32"))
h = torch.from_numpy(self.rng.random(n).astype("float32"))
```

and the held-out slice is `self.h < self.hold` with `vo_om_hold = 0.1` (L181, L321 of
`VoProjBank`). The dedup is **per push**, i.e. per cycle. So the same `(final configuration,
root)` recurring in a **later** cycle is appended again with a **fresh independent draw** and can
land on the other side of the split — carrying the same configuration *and* the same verdict.
`vo_critic_terms`' own docstring names the precondition and treats it as a given: *"The metering
set is fixed per era, so the same masked context recurs across cycles; a coin flip would put
copies of one context on both sides of the critic's split and its AUC would be reading its own
training data"* — which is exactly why the **critic** hashes the context instead. `VoOutcomeBank`
did not take that step, and `voicing` Q0's finding that the executor writes one or two rows of a
128–192 row book on nearly every frontier call (top-1 share 0.93–0.996) makes the recurrence of a
final configuration expected rather than hypothetical. For a configuration appearing `k` times the
straddle probability is `1 − 0.9^k − 0.1^k`: 0.18 at `k = 2`, 0.41 at `k = 5`, 0.65 at `k = 10`.

**The size is not measurable from the banked artefacts.** `aliquot`'s arm files do not carry the
bank's buffer, so the realised `k` is unknown, and no number is put on it here. What the code
supports is the direction and the mechanism, and nothing more.

So, corrected: **the critic's split (this node's, and `voicing`'s) is a function of the context
and is therefore immune to cross-cycle recurrence but open to the sibling-slot duplicate;
`aliquot`'s bank dedups the sibling-slot duplicate away and is open to cross-cycle recurrence.**
The pairing is the exact opposite in the two organs, and the original sentence conflated them.

---

---

## §3 The shaping, decided

### §3.1 The outcome head: an MLP over the probe's own features

The brief allowed either an MLP over the pooled features the linear probe reads, or the critic's
own form. **The former was chosen**, because then the shaping gradient flows through exactly the
representation the probe is about to read and through nothing else:

```
[ pooled(obs+write).mean ; pooled(obs+write)[span].mean ; slot_emb ]  ->  Linear(224,256) -> GELU -> Linear(256,1)
```

Two properties worth stating:

- **No candidate embedding.** The critic's own form adds `cand_state(cand)`, a content embedding
  of the written candidate, *beside* the context. Here the candidate is already rendered into the
  input, so its content can reach the head only through the trunk — which is the object under
  test. A candidate embedding would let the head fit the verdict without the trunk ever having to
  carry "this candidate here".
- **The slot embedding cannot inflate the probe.** The probe is fit **per slot**, so slot
  identity is constant within every fit it appears in and contributes no AUC. The embedding is
  there only so one head can serve 30 slots, which is the critic's own arrangement
  (`Critic.slot`).

The head is minted with `build_critic`'s own discipline (save the global RNG state, construct,
restore, then re-initialise every parameter from a dedicated generator at `normal(0, 0.02)` with
biases zeroed), so its existence costs the shared stream nothing.

### §3.2 The diet is the critic's

`vo_critic_terms` trains the critic on the **union** of the filed buffer and the probe buffer per
slot, 64 rows drawn per slot per step from the rows with `code >= 10`. The shaping draws
`SHAPE_SLOT_BATCH = 32` rows from each of the two buffers at each of the 30 slots, i.e. 64 rows
per slot per step split evenly between the two doors. The two buffers are the same size (8192
rows each), so this is within sampling noise of the loop's own weighting, and it is stated rather
than glossed. One forward of the concatenated 1,920-row batch per step; the loss is the mean over
rows, which equals the mean over per-slot means because the counts are equal — `span_train_terms`'
averaging.

### §3.3 The losses, the weights, the rate and the ladder

- **Weights 1 and 1.** `vo_critic_terms`' docstring: the critic's term is "added to the plant's
  loss in the SAME optimizer step — the arc's convention, so the treatment buys no extra steps",
  and `span_lam = 1.0`. So `loss = outcome_BCE + infill_CE` with no relative weight, and the two
  single-loss arms are that expression with a term removed.
- **The infill loss is `_train_generator`'s, verbatim in form**: fresh corpus windows, `n_mask`
  drawn uniformly in `[1, n_blocks]`, cross-entropy on the masked blocks' true level-1 features
  only. `SHAPE_INFILL_BATCH = 256` is the run's own `batch_size`. `clip_grad_norm_(core, 1.0)` is
  the donor's clip and is applied on **every** arm alike, including outcome-only, so the clip is
  not an arm difference.
- **One learning rate: 1e-4**, which is the run's own `gen_lr` — the rate at which the loop
  continues this plant. Choosing the loop's own rate means the step counts are readable against
  the loop's own budget and nothing was tuned.
- **Three step counts: 100 / 400 / 1600.** The paid arm continues its plant for `gen_steps = 20`
  per cycle over ~200 cycles ≈ 4,000 steps, so these are 2.5% / 10% / 40% of the paid arm's step
  budget at the paid arm's rate. Three rather than two because the smoke showed **20** steps of
  outcome-only already moved the level-1 parse accuracy by −0.12, so the interesting range starts
  low. `AdamW(weight_decay=1e-4)`, `_train_generator`'s optimiser.
- **Fresh corpus.** `_sample_pool(rules, 32768, s, 777001)` for the shaping stream and
  `_sample_pool(rules, 8192, s, 777002)` for the diagnostics, both distinct from the run's own
  `train_seed = 1` and from its probe-clean stream (`seed + 31337`). `plant_holdout = 0` in both
  banked configs, so no level-2 tuple is withheld and the regenerated corpus is the run's own
  distribution.

### §3.4 The streams are matched, which is what makes the three arms a contrast

- The outcome row draw is one `np.random.default_rng(10000 + steps)`, keyed on the step count and
  **not on the mode**, so `both@N` and `out@N` see the same rows in the same order.
- The infill window and mask draw is one `torch.Generator(device).manual_seed(20000 + steps)`,
  likewise keyed on the step count alone, so `both@N` and `infill@N` see the same windows and the
  same masks.

A difference between two arms at the same `N` is therefore the loss and not the data. (The smoke
confirms it: `both@20` and `infill@20` log the same `infill_ce` at step 1 to four decimals, and
diverge only as the trunk does.)

### §3.5 What is *not* changed

Nothing in the plant's architecture, nothing in the critic, nothing in the probe, nothing in the
rows, nothing in the grammar. The frozen arm is the banked trunk with `load_state_dict` and no
optimiser at all.

---

## §4 The world-model diagnostics

`shaped`'s cost side was the trunk's clean next-token cross-entropy plus the **altitude**
instrument (`κ*`, `KL(p_L‖q)`), which asks whether the trunk still sits in the coarse-Bayesian-
observer family. The practice plant is not a next-token model, so both halves need a decided
analogue.

### §4.1 Held-out masked-infill loss — the direct analogue of "clean CE"

Cross-entropy and fill accuracy on the masked blocks of **fresh** corpus windows, at a **fixed**
window/mask plan (`WM_SEED = 909`, 16 batches of 512, `n_mask` drawn once and reused) that is
**identical on every trunk**, so a difference is the trunk and not the draw. Reported overall and
split by how many blocks are masked (1 / 2–4 / 5–8 / 9–16 / 17–32), because a shaping step could
in principle trade the easy end of that curve for the hard end.

### §4.2 The block head's level-1 parse accuracy — and why its ceiling is not 1

`ratchet/macros.py::parse_features` is the agent's own parse of a configuration into level-1
features: `block_logits(x).argmax(-1)`, with one block masked elsewhere. Compared against
`exact_features` (the exact bottom inverse map) this is an **oracle instrument**, which the brief
licenses. Two cells are reported: `mask1` (mask block 0, score blocks 1…31 — the idiom, and the
read every mining organ in the arc uses) and `nomask` (score all 32).

The number to expect is **not** near 1.0 and that is not a defect: `_train_generator` computes
its loss on masked blocks only, so the head is never trained to report a block it can see.
`calr_s0` measured 0.63 on a visible block; the smoke reads 0.649 / 0.656 on the frozen trunk,
which is that number. What the instrument is for is the *difference between trunks*, and the
random trunk reads 0.193 / 0.189, so it discriminates.

### §4.3 `nested_phase` is **not** the practice-side altitude instrument — the decision, with
### its reason

The brief offered `logit_reading/frontier/common.py::nested_phase` in its template-free
(`meanprof`) mode "if it is cheap". It is not cheap and, more to the point, it is not the same
object here. `nested_phase` estimates, level by level, which column of a flat token stream
carries the grammar's period-`s^k` boundary, from a **next-token reader's per-column entropy
profile** (`Xd`). `practice/perception/` had to train two such readers (`train_trajectory`'s
recipe, 8L/8H/256D, 16–32k steps) to run it, because the practice plant does not produce that
profile: it is a masked **block** infiller whose block grid is aligned to the code grid by
construction. There is no phase to recover, and shaping the trunk cannot move a phase estimate
that the architecture never had to make.

**What replaces it**, and is the genuine analogue of "how deep does the trunk's grasp of the
nested structure go": the **nested DP parse**. `ratchet/macros.py::macro_features` masks a level-ℓ
span and runs the executor's own max-sum DP over the composition table to recover the span's
level-1 features. Run with the **TRUE** tables (`MC.true_tables`), it asks how deep the plant's
generative competence survives — a level-2 span is 2 masked blocks of 32, a level-5 span is 16 of
32 — with the table's contribution held exactly fixed, so the only thing that varies across
trunks is the trunk. Levels 2…5, two nodes each, reported per rung. The top rung's table has
262,144 rows, so its batch is sized from the entry count and the loop is guarded.

---

## §5 The transfer split

One readout over the read of record (`post`), **pooled across slots** rather than per slot, fit on
the training rows of the L2/L3 slots and scored on

- the **held-out rows of L2/L3** — the within-level fit, which is the number the transfer cell has
  to be read against;
- the **held-out rows of L4/L5** — the transfer cell itself;

beside the same readout fit on **L4/L5's own** training rows and scored on L4/L5's held-out rows,
which is the ceiling the transfer cell is reaching for. Seed 0's ladder gives L4 and L5 slots;
seed 2's stops at L4, so its "L45" group is L4 alone.

Three decisions:

- **The two fits use the same number of rows.** `XFER_CAP = 20000` per group, and then both fits
  are truncated to `min(n_L23, n_L45)` after a fixed permutation, so "the L2/L3 fit is better"
  cannot be a sample-size statement. The permutation matters: the rows arrive buffer by buffer,
  so truncating without it would take the low-node slots only.
- **No root term and no slot term.** The point of the cell is whether *one* direction found at
  the levels every seed reaches carries to the frontier. A per-slot or per-root readout would be
  a different question (and is `aliquot`'s).
- **The two doors are split.** A filed row's candidate set is the beam's own narrow one and a
  probe row's is the babbler's; `overtone` finding 2 is that almost nothing candidate-specific is
  readable on filed rows at all. Pooling them would read a candidate-set difference as transfer.

### §5a What `du0`/`du1`'s (d) does and does not test — and `du2`

In `du0` and `du1` the shaping diet is **every** slot, so by the time the L2/L3-fit readout is
scored at L4/L5 the trunk has already heard L4/L5 verdicts. Section (d) there is therefore a test
of **the readout's direction**: given a representation shaped on everything, does one linear cut
found at L2/L3 carry to L4/L5? That is a real question and it is the one the brief asked for, but
it is not the question as Jasper posed it in discussion, which is about the **representation**.

`du2` is that question. One knob, `shape_levels = "2,3"`: the outcome loss hears the training rows
of the L2/L3 slots **only** — the levels every seed reaches — and nothing else changes. The
probe's fit, the held-out rows, the infill stream, the diagnostics and the dedup filter are all
`du1`'s. Section (d) then reads: **does an outcome heard at the levels every seed reaches teach a
direction that carries to the frontier the seeds barely touch?** Arms: `both@400`, `both@1600`
(the arm of record), plus `frozen`, `rand` and `infill@400/1600` kept in the table as reference.

Two properties of the construction worth stating. The outcome row draw is a numpy generator and
the infill window/mask draw a torch generator, both keyed on the step count alone, so restricting
the slot list changes the numpy draws only: **`du2`'s `infill@N` trunks are bit-identical to
`du1`'s and `du0`'s**, which the reduction checks as a cross-tag fingerprint gate, and `du2`'s
`both@N` sees the same corpus windows and masks as both of theirs. And the L4/L5 *held-out* rows
were never in any shaping diet under any tag, so `du2` removes the L4/L5 **training** rows from
the trunk's diet and nothing else.

One denominator to keep in view, found in the output rather than anticipated: at seed 2 the
`probe/d` (disagreement-drawn) held-out rows solve so rarely — base rate 0.017 — that **only 6 of
the 28 slots have both classes present and therefore an AUC at all**, which is why overtone's own
table printed `(6 slots)` there. Every `probe/d` median in the reduction is a median over those
six. The reduction's `n` column now reports defined AUCs rather than slots present, for this
reason (§7.3).

---

## §6 The arms, the cost and the shape of the run

Eleven trunks per seed: `frozen`, `rand`, and `{both, out, infill} × {100, 400, 1600}`. Two seeds
across two containers from a CPU coordinator (`sweep` `.starmap`s `probe_seed`), which adds no
GPU-hours — the seeds were always two passes — and keeps the detach rule intact. One L4 each.

Smoke (`dusm`, seed 0, 8 buffers, 20 steps): 23 s, peak RSS 5.3 GB (torch's CUDA context is most
of it), peak GPU 1.2 GB. The full run's memory request is sized from that at 12 GB rather than the
lineage's inherited 32 GB. Realised: `du0` 766 s / 732 s per seed, peak RSS 5.5 GB, peak GPU
2.0 GB; the shaping is 12% of it (the nine arms cost 1.0–193 s each) and the eleven probe-plus-
diagnostic passes are the rest. Realised: `du1` 780 s / 743 s (same 11 trunks, filter on), `du2` 365 s / 379 s (6 trunks). Total
for the node, smokes included, ≈1.2 GPU-h across six L4 containers — against the ~4 GPU-h the paid
three-arm in-loop version would have cost for one of the three questions.

---

## §7 Defects and withdrawn diagnoses

*(kept beside their corrections, in the order they were found)*

1. **The node was first written as a sibling of `aliquot` rather than a child of it.** The brief
   says "a child folder of `sotto_voce/aliquot/`"; `sotto_voce/duplex/` was created and then moved
   to `sotto_voce/aliquot/duplex/`, with `analyze_duplex.py`'s relative path to
   `overtone/figures/` corrected from two `dirname`s to three. Caught by re-reading the brief
   before writing the docs, not by anything failing.
2. **The masked read was going to be the read of record**, on the strength of `aliquot` §2.1's
   finding that masking mattered. Withdrawn: the node's licence is that overtone's table comes
   back cell for cell, and overtone read unmasked. The masked read is reported beside it as
   `postm` (§2.1), and on the smoke it is *worse* than the unmasked read on the frozen trunk
   (filed 0.552 against 0.619, probe 0.752 against 0.795) — the opposite sign from `aliquot`'s
   in-loop column, which §2.1 records is not a contradiction because the two estimators differ in
   more than the mask.
3. **The reduction's `n` column counted buffers, not buffers with a defined AUC.** `collect()`
   appended `None` values and then `med()` filtered them, so every median was right and the
   denominator beside it was wrong. Caught by comparing against the `(N slots)` counts overtone
   itself printed: the banked table says 29 filed slots at seed 0 and **6** `probe/d` slots at
   seed 2 where the first version of the reducer said 30 and 28. Corrected to count defined AUCs.
   This mattered: the `probe/d` column is a median over six slots, and a denominator of 28 would
   have hidden that.
4. **The exact-duplicate channel (§2.5) was not in the design.** It was found by asking, after the
   first table was in hand, what the held-out split actually guarantees — `hold_code` hashes the
   pre-write context, and the shaping reads the post-write configuration, and those are not the
   same key. Measured at **26.1% / 24.7%** of held-out filed rows and 0.27% / 0.65% of probe rows,
   and closed in `du1` by an exact-hash filter on both the shaping diet and the probe's fit. The
   first reading of the filed column — "the outcome loss in the weights buys +0.10 to +0.15 AUC on
   filed rows" — is **withdrawn pending `du1`**, and `du1`'s filed column is what replaces it. The
   probe column was never at risk and stands as read.
5. **Gate G0 broke when the filter was switched on, and correctly.** The numpy reference still fit
   `~hold` while the torch fit used the filtered mask, so the gate read `ΔAUC 2.7e−3` and
   `max |Δscore| 0.43` on the dedup smoke. Nothing about the estimator had changed, which is what
   made it obvious; corrected to fit both on `b0["tr"]`, and the gate went back to `Δ = 0.0` at
   full scale on both seeds. A gate that goes red when only the row set moved is the gate working.
6. **A claim about `aliquot` that did not survive a read of the code, kept beside its
   correction in §2.6.** §2.5's first draft said `aliquot`'s `hold_auc` inherits *this* duplicate
   channel. `push_experience` deduplicates by `(configuration, root)` within the push, so the
   sibling-slot mechanism cannot arise there; what `aliquot` has instead is a **coin-flip** hold
   key (`_append`: `h = self.rng.random(n)`) against a per-cycle dedup, so the same final
   configuration recurring in a later cycle straddles the split with a fresh draw. Different
   mechanism, same direction, size not measurable from the banked artefacts. Caught by the
   orchestrator asking for the claim to be checked against the code before it stood as a fact.
7. **One located G1 cell.** Of the 180 cells compared against the banked overtone dumps, 179 agree
   to within 0.0046 and one — seed 2, `rand` / `post` / `filed:4:3`, n_hold 842 — differs by
   **0.0398** (0.6958 here against 0.656 banked). The bulk of the spread is float32 feature
   arithmetic on a GPU against overtone's on a CPU, amplified through a 193-column IRLS on a
   held-out set of hundreds; the one large cell is presumably an ill-conditioned fit on the random
   trunk's feature map at that slot tipping across a few ranks. It is bounded two ways and neither
   is an argument: it does not touch the `frozen` trunk (every one of whose cells is ≤ 0.0046), and
   **gate G1b** — this node's `frozen` and `rand` medians against the `MEDIAN over slots` lines
   overtone printed itself — agrees on **all 36 cells to within 0.0015**.

---

## §8 Reproduction

```bash
cd experiments/            # MODAL_PROFILE=chromatic
modal run rhm/practice/voicing/sotto_voce/aliquot/duplex/duplex.py::sweep --out-tag dusm --smoke 1 --seeds s0
modal run --detach rhm/practice/voicing/sotto_voce/aliquot/duplex/duplex.py::sweep --out-tag du0
modal run --detach rhm/practice/voicing/sotto_voce/aliquot/duplex/duplex.py::sweep --out-tag du1 --dedup 1
modal run --detach rhm/practice/voicing/sotto_voce/aliquot/duplex/duplex.py::sweep --out-tag du2 --dedup 1 \
    --shape-levels 2,3 --modes both,infill --steps 400,1600
python3 rhm/practice/voicing/sotto_voce/aliquot/duplex/analyze_duplex.py --tag du0 --fetch
python3 rhm/practice/voicing/sotto_voce/aliquot/duplex/analyze_duplex.py --tag du1 --fetch
python3 rhm/practice/voicing/sotto_voce/aliquot/duplex/analyze_duplex.py --tag du2 --fetch
```

Artifacts on `rhm-scaling-data:/rhm_practice_duplex/<tag>/{s0,s2}.{json,log}`; the reductions of
record are [`figures/du0_reduction.txt`](figures/du0_reduction.txt) (the protocol as overtone and
the loop have it, and the reproduction gate) and
[`figures/du1_reduction.txt`](figures/du1_reduction.txt) (the same with §2.5's exact-duplicate
filter on, which is the filed column to read) and
[`figures/du2_reduction.txt`](figures/du2_reduction.txt) (§5a: the outcome heard at L2/L3 only,
which is the transfer question as it was posed).
