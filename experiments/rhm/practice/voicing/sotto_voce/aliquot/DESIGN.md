# DESIGN — `aliquot`: the verdict read off the world model's own state

Decisions, and every withdrawn diagnosis kept beside its correction. The brief is
[`SPEC.md`](SPEC.md); the machinery index is [`FILES.md`](FILES.md); the interpretation, discussed 2026-09-20, is in the aliquot writeup ([`README.md`](README.md) at the aliquot level); the facts live in the reduction under `figures/` and here.

**Child**: [`duplex/`](duplex/DESIGN.md) — §1's finding taken offline: whether the outcome error has
to reach the trunk's *weights* rather than only the readout's, read on the banked `overtone` dumps
before the paid in-loop version.

Donor: [`../sotto_voce.py`](../sotto_voce.py) at its `so_s1`/`so_s2` head, untouched. Every
addition in `aliquot.py` is marked `# [aliquot]`, every knob defaults off, and with every knob
off the file is `sotto_voce.py` and therefore `enharmonic.py` (gate G-F).

---

## §0 The name

`aliquot` strings are the strings in a piano that are never struck: they resonate because the
struck ones do, and they are what gives the instrument its bloom. The node's object is a reading
taken off the instrument's own body rather than from a second voice — which is the difference
between this node and its parent, where the grader was a separate net with a representation of
its own. It also pairs with its sibling [`../../overtone/`](../../overtone/README.md), whose
offline probe is this node's going-in prior: aliquot strings reinforce overtones.

---

## §1 What is being put in the grader's seat, and why it is a different object

`sotto_voce`'s **mirror** is a tree net from `(rendered configuration, root)` to P(solved),
trained by BCE on the learner's own experienced configurations and the world's verdicts on them.
Its representation is learned from the learner's experience alone, so its altitude is set by what
the learner chose to write.

The object `ideas/calibration_and_violation_are_one_object.md` §12.4 names is the verdict read off
the **world model's own state** through a **linear readout**. Two things differ, and both are the
question:

1. **Where the representation comes from.** The trunk is the practice loop's generator: trained
   on the corpus by masked infilling and then continued through the run. Its features are not a
   function of the learner's write choices, so a region the learner never wrote at can still be
   represented — or not, and then the projection is blind in a *different* place than the mirror.
2. **The shape of the map.** A linear readout can only price what is already on the wire. That is
   the constraint §12.4 makes the argument from, and it is why a candidate has to be *fired* (the
   world model put into the state the candidate implies) to have a price at all. The probe channel
   already fires the candidate — the substitution into the final configuration *is* the firing —
   so this node isolates the *pricing* half and nothing else.

`overtone`'s offline table is the prior going in (medians over slots, uniform probe rows):
pre-write 0.57, random trunk 0.74, trained trunk 0.80, MLP critic 0.85. Two things about it are
worth keeping in view: the trained trunk's margin over the **random** trunk was small
(0.796 vs 0.735 at seed 0; 0.739 vs 0.717 on seed 2's uniform rows), and the probe there was fit
**on probe rows**. This node fits on experience only, which is strictly harder.

---

## §2 The readout, decided

### §2.1 What it reads — `overtone`'s `post_slot`, through the mask idiom

Features are `SN.trunk`'s pooled per-block hiddens over the **substituted final configuration**
(the same rendered object the mirror grades and the world grades), pooled two ways and
concatenated: the mean over **all** blocks and the mean over the **written span**. That is
`overtone::post_write_probe`'s `post_slot`, which beat `post_mean` on both of its seeds
(0.796 vs 0.697 at seed 0, 0.882 vs 0.766 at seed 2). `state_dim = 96`, so the feature vector is
192 wide.

**The mask.** The plant is trained by masked infilling with `mask_min = 1` and has *never* seen a
fully unmasked input; `calr_s0` measured 0.63 block accuracy on a *visible* block when read
unmasked, which is why `ratchet/macros.py::parse_features` masks one block elsewhere before
reading. So the read of record here masks **the first block outside the span**: in distribution,
with every block of the span still visible. `overtone`'s offline probe read the configuration
unmasked, so the two are not the same read, and the unmasked variant is refit and scored on the
same held-out rows every tenth refit (`hold_auc_nomask`) so the size of that choice is a number
rather than an argument.

**The span pointer, and its one honest wrinkle.** A probe row's span is the substituted slot's.
An experience row's span is the slot the write happened at — and `push_experience` deduplicates
by `(configuration, root)` within the cycle, so a tip that wrote at several slots contributes one
row carrying whichever slot came first in the dict. The mirror is unaffected (it never reads the
slot). For the projection this means the span pointer on an experience row is *a* slot the learner
wrote at, not *the* slot; `hold_auc_mean` (mean-pooling only, no span pool) is logged beside the
arm's own AUC every refit so the reduction can say whether the span pool was carrying anything.

### §2.2 The root — a readout **per goal**, not a per-goal bias

The trunk is not root-conditioned (`SN.trunk` asserts it), so the pooled state carries the
configuration and not the goal. The world's verdict is "the root's possible set contains r\*",
which is a *match* between the parse and the root: an additive root term cannot express it, while
the mirror's own `top * e` product can. Being unfair to the projection here would answer a
different question, so the design matrix is

```
[ std(features) , onehot(root) , onehot(root) ⊗ std(features) , 1 ]        (192 + 8 + 1536 + 1)
```

which is exactly **one linear readout of the state per root**, with a shared component the ridge
shrinks the per-root deviations toward. That is also the value arc's own object: the reader in
`logit_reading/orbitofrontal/` is fit per goal. The additive-root variant is refit at the same
ridge every refit and reported as `hold_auc_add`, so "was the interaction load-bearing" is
measured.

The intercept column is **unpenalised**; every other column is penalised. Without that, a large
ridge would drag the overall level toward P = 0.5, which at this probe base rate is a large bias.

### §2.3 How it is fit — ridge logistic, solved not stepped

IRLS (`vo_pj_irls`), which is `ov_irls`'s estimator — same Newton step, same clip, same early
stop — moved into torch so it can run on the GPU every cycle. Two reasons for solving rather than
stepping:

- `logit_reading/coeruleus/readout.py` made the same choice for the same reason: *"an SGD-fit MLP
  is not a fair test of whether the information is there"*. A closed-form fit removes the
  optimizer from the question.
- The features move every cycle (§2.4). A stepped readout chasing a moving representation
  confounds "the information is not there" with "the optimizer has not caught up". A solve is the
  exact minimiser of the current objective on the current features, every cycle.

The Gram is accumulated in float32 and the Newton solve is done in float64: with the ridge the
system is well conditioned, and a float64 Gram on an L4 is ~30× slower for no measurable
difference. Cost, measured in the smoke: see §6.

**The ridge.** A grid `1, 32, 1024` on standardised features, selected per refit on a
**train-internal** validation split — the rows with `u[:,0] >= vo_om_boot`, a draw the buffer
already holds, so no new RNG and no new bookkeeping. This matters for a reason worth stating: the
bank's held-out slice (`h < vo_om_hold`) is the *instrument*, and if the ridge were selected on it
then `hold_auc` would be a selection target and its drift would be uninterpretable. Below 8
validation rows the selection would be noise dressed as a choice, so the middle of the grid is
taken and `lam_fixed` records that per cycle. At run sizes (`vo_om_min = 512`) the slice is in the
hundreds and that branch never fires; at preflight sizes it always does, which is one more reason
no preflight number is a measurement.

### §2.4 Recompute, not cache — and the gate that makes it a fact

The trunk is the **live** generator the loop trains every cycle, so the features move under the
readout. The buffered rows' features are therefore **recomputed through the current trunk at
every refit** and never cached. Three reasons:

1. It is the object under test. §12.4's claim is about a linear readout *of the world model's
   state*; a cached feature is a readout of a stale world model, which is a different and weaker
   claim.
2. `practice/tuning/` and `orbitofrontal/adaptation/` are the record on readers fed by a moving
   learner, and a stale feature is the failure mode they describe.
3. It is cheap here (§6): the trunk is a 2-layer, 96-wide encoder over 64 tokens.

A caching bug of exactly this kind is invisible to every *inertness* gate in the arc — a cached
feature is more inert than a live one — which is `sotto_voce` DESIGN's defect-#8 shape one organ
over. So **gate P-3** is a liveness gate: with the readout's weights held fixed, perturbing the
live trunk must change the live arm's probability on the same row. The falsification harness runs
the caching perturbation and the gate goes red on it.

The trunk's own fingerprint (sum of |parameter|, in float64) is logged at **every refit**, so
"the representation moved over the run" is read off the arm file (`P-3r`) and never assumed — and
on the random twin it is the same fact in reverse (`P-2r`: constant, and equal to its mint).

### §2.5 What is filed

The readout's **probability**, exactly as the mirror files its own, for `sotto_voce` DESIGN §3d's
reason: the critic's loss takes a soft target, and at the probe base rate (0.02–0.09) a 0.5
threshold throws away almost everything the grader knows. The thresholded verdict is the
**readout** and is what [P]/[R] report against the world. The hard-verdict variant is untested
here too.

### §2.6 What is *not* changed

The diet (src 0 alone), the schedule (`train()` then `refresh()`, every cycle, before the probes
run), the dedup, the probe channel's substitution and draw, the instrument-only world verdict, the
zero bill, the composed chooser, and the clock yoke. `vo_om_mode = "proj"` dispatches in
`vo_run_probes` exactly where `"model"` does (`if mode in ("model", "proj")`), so the *only*
difference between `al_pj_yk` and the banked `so_mg_yk` is what the probability came out of. The
agreement rule is vacuous on a projection arm by construction (K = 1, spread ≡ 0, thr = ∞,
everything filed), and gate M-4r asserts that rather than leaving it implied.

---

## §3 The arms, and the twin that is the point

| arm | `vo_om_mode` | `vo_pj_trunk` | what it is |
|---|---|---|---|
| `al_pj_yk` | `proj` | `live` | the readout over the generator the loop trains every cycle |
| `al_rt_yk` | `proj` | `rand` | the same readout over a never-trained generator of the same architecture at `vo_pj_rand_seed = 20260918` |

`al_rt_yk` is `overtone`'s **control two, in the loop**, and it is the arm that makes the result
interpretable: the readout's link function, its diet, its schedule and its design matrix are
identical, so a gap between the two arms is the **trained representation** and nothing else. It is
minted by the architecture's own default init under `manual_seed(seed)` inside the file's RNG
sandbox — which is precisely what `overtone`'s offline `rand_slot` control did, so the two floors
are the same object read in two places.

The parameters are deliberately **not** hand-zeroed the way `build_outcome`'s are: a transformer
carries LayerNorm weight vectors, and zeroing those makes the trunk output identically zero. That
would have produced a "random trunk" floor of 0.5 and a meaningless contrast.

**A withdrawn diagnosis, kept.** Gate P-2's first version took the twin's fingerprint *after*
`rt.train()` had already run and compared the end state to that. The falsification harness then
reported the gate **BLIND** on the one perturbation it exists to catch — a twin whose trunk is
trained inside `train()` — because the perturbation fired before the baseline was taken. The
correction is one line: the baseline is the twin's **mint** (`rt.sig0`, taken in `__init__`), so
any step anywhere is visible. This is the round's first defect and it was caught by the harness
and not by a claim resting on the gate, which is the harness's whole purpose.

---

## §4 The gate table

Everything `sotto_voce` gated still holds and is re-run: G-F, V-1, V-4A/B, V-4c/V-4d, V-5, V-6/V-6b,
Y-1, M-1, M-1b, M-2, M-2b, M-3, M-4, M-5, and the run-level M-1r/M-2r/M-3r/M-4r. `"proj"` joins the
mirror-dispatch gate's mode table, so M-1/M-3/M-4 are asserted on the new dispatch as a
construction, and `vo_gate_mirror_run` accepts it with `want_bill = 0` and an added assertion that
a projection arm files and "agrees on" the whole draw.

New, each shown to fail on a deliberate perturbation before being reported
([`gates/falsify.py`](gates/falsify.py), **72/72**):

| gate | claim | the perturbation it goes red on |
|---|---|---|
| **P-1** | the readout's logit is **affine** in the pooled state (checked on a convex combination of two rows' features at a fixed root, which no implementation detail can fake); the plant's parameters are unchanged **to the bit** by a fit and by a query; no trunk parameter accumulates a gradient; no shared RNG stream moves across a fit, a refresh and a query | a square term smuggled into the link · the fit steps the trunk · the fit draws from the shared torch stream |
| **P-2** | the random twin is a **different object** and is **frozen** against its mint | the "random" twin is secretly the live trunk · the twin's trunk is trained |
| **P-3** | the features **follow the trunk**: perturbing the live trunk moves the live arm's probability on the same row and leaves the twin's unchanged | the features are **cached** |
| **P-4** | the readout is **live**: it refits, the weights move between refits on different buffers, the filed probability is not constant | the readout is fit once and never refit (disconnection) |
| **P-2r / P-3r / P-4r** | run-level, off the arm file: the trunk's fingerprint **moves** on a `live` arm and is **constant** on the `rand` twin; the refit count is non-zero; the ridge came out of the stated grid; the held-out instrument exists; no row the readout read was missing its slot | a `rand` arm whose fingerprint moved · a `live` arm whose fingerprint never moved · `n_refit = 0` · a ridge off the grid · no AUC recorded |

`vo_preflight_gates` additionally runs the two projection twins through Y-1 and both V-6b forms:
the projection-graded diet must **change the run** against the world-graded twin, and the two
projection arms must differ from **each other** — because if they did not, the trunk's
representation would be doing nothing at all and the round would have no contrast to report.

**A defensive choice, stated.** An experience row whose slot has no bound move would have no span
to pool over and no safe block to mask. The first version asserted on it. That assertion would
have killed a 1.3 GPU-h arm at cycle 40 over a diagnostic, so it is now a **counter**
(`n_noslot`): such a row is pooled over the whole configuration bar its last block, the count is
logged, and `P-4r` asserts it is zero **post hoc**. The span is clamped below `n_blocks` for the
same reason (`n_spanclamp`). A counted fact where an assertion is a dead arm.

---

## §5 What the reduction reports

[`analyze_aliquot.py`](analyze_aliquot.py) keeps every `sotto_voce` section working and changes
three things:

- **[P] and [R]** now print the **banked** mirror arms beside this tag's own (`so_s1:so_mg_yk`
  and its committee/hybrid siblings, resolved through the four-tree loader), so the projection's
  per-level column and the mirror's sit in one table instead of two files. Nothing banked is
  re-run.
- **[T]** additionally asserts P-2r/P-3r/P-4r on every projection arm, by calling the substrate's
  own gate rather than a second implementation of it.
- **[PJ]** is new: the refit count, the skipped-refit count, the ridge chosen per refit (with the
  `lam_fixed` count and the validation slice's size), the held-out AUC on experience **first vs
  last** and the mean and max of its **drift between refits**, the three secondary variants
  (`add`, `mean`, `nomask`) at the same ridge on the same held-out rows, and the trunk's
  fingerprint series per arm with its range.

The chooser's repair accuracy ([Z]/[S]), the critic's counterfactual AUC against the world ([J],
`auc_world`) and the bill ([O]) are the donor's sections, unchanged, and the floor, ceiling and
mirror columns come from `--bank`.

[`mk_seedtable.py`](mk_seedtable.py) is the cross-seed table, `sotto_voce`'s
`figures/so_seedtable.txt` one node on, written to `figures/al_seedtable.txt`. Two things in it
are deliberate and are stated in its own output rather than buried here:

- **(b)'s aggregate is this script's recipe** — the median over slots of the median over cycles —
  applied to every arm alike. `sotto_voce`'s seedtable reports 0.667 / 0.617 for the seed-0 mirror
  where this recipe gives 0.675 / 0.641, and its exact aggregation could not be recovered from the
  arm files. So the table says out loud that only WITHIN-table comparisons are valid there.
- **(g), the grader's own ranking of the world's verdict per level, is not comparable to the
  banked mirror's row.** `sotto_voce`'s README already records why: its per-probe sample filled
  front-to-back and holds cycles 50-54 only, the reservoir having shipped for later tags — which
  these are. The mirror's row is therefore a reading of a five-cycle-old mirror over ~4 cycles of
  probes and the projection's is a uniform sample of the whole run. The row is printed with the
  caveat rather than dropped; the mirror-vs-projection comparison runs through (c)'s 2x2, which is
  whole-run on every arm, and through (b)'s downstream critic AUC.

The second of those is the round's one real instrument gap, and it is inherited rather than
introduced: the number that would most directly answer "does the projection rank the world's
verdict better or worse than the mirror, at matched level" needs the mirror re-run with the
reservoir, which is a banked arm this node was told not to re-run.

---

## §6 Cost, and where it goes

**Measured.** `al_s1` 12761 s and `al_s2` 10261 s of container time for two arms each — 3.55 h
and 2.85 h, so 1.8 h and 1.4 h per arm against `sotto_voce`'s ~1.3 h. The projection therefore
costs roughly 20-40% more wall-clock per arm than the mirror, which is the price of recomputing
the buffered rows' features through the live trunk at every refit plus five IRLS solves per cycle
at 1737 columns. 151 refits at seed 0 and 141 at seed 2, none skipped. Round total including the
gates, the G-F smoke and three preflights: ~9 GPU-h against the brief's ~10.

The two tags ran concurrently in separate containers, so the wall clock was the longer of the two
rather than their sum; the arms within a tag ran one after another, which is where a further
halving is available to a later round (a CPU coordinator that `.starmap`s the arms, per the
subagent skill's note).

`memory=12288` was inherited from `sotto_voce`'s measured peak and neither tag hit it. The peak-RSS
line each arm prints was lost from the local logs when the session's container restarted mid-run
and killed the `modal run --detach` clients; the remote apps were unaffected, which is why the
waiter for these tags watches `modal app list` and not the local log
(`results/wait_app.sh`, copied from `practice/embouchure`).

The projection adds, per cycle: one trunk forward over the fit rows (≤ `vo_pj_fit_cap = 8192`),
one over the held-out rows (≤ 2048), three IRLS solves for the ridge grid plus two for the
secondary variants, and — every tenth refit — one more trunk forward and one more solve for the
unmasked diagnostic. The trunk is 2 layers at width 96 over 64 tokens, so the forwards are the
cheap half; the design matrix is 1737 columns, so the solve is a 1737² Gram and a 1737³ Cholesky
per iteration, which is why it is on the GPU.

`vo_pj_fit_cap` bounds the whole thing: it is a **uniform subsample of the trainable rows** drawn
from the bank's **own separate** numpy stream (`frng`), so the fit's subsampling cannot move the
buffer's bootstrap or hold draws by a single value. That is the same discipline `build_outcome`
applies to a committee member's minting.

---

## §7 Defects, in order, each beside its correction

**§7.1 Gate P-2 measured the twin's fingerprint after its first fit.** Reported BLIND by the
falsification harness on the "the twin's trunk is TRAINED" perturbation. Corrected to compare
against the twin's mint (`sig0`, taken in `__init__`). §3.

**§7.2 A constant-label buffer produced no fit, so the whole probe channel went dark.** The first
version of `VoProjBank.train` returned `None` when every label in the training slice was the same
class — which looks defensive and is in fact a silent disconnection. At **preflight** sizes the
substrate is trained for a few dozen steps and solves nothing, so every experience row's verdict
is 0; the readout was therefore never fit, `ready()` stayed False, `_om_ready` was False on every
cycle, every probe was skipped, and nothing was filed. The mirror does not have this shape because
BCE on an all-zero target is a perfectly well-posed step, so the donor's preflight was reachable
and mine was not.

**Caught by gate M-1r on `al_pf_pj`**: `M-1r VACUOUS: no probe was ever drawn` with
`om_push_exp = 333`, `n_probe = 0`, `hold_base = 0.0`. A gate written for the mirror's
instrument-only claim caught a projection-only defect, which is the argument for keeping the
donor's gates running on a new arm rather than writing fresh ones.

Correction, three parts:

1. **The degenerate fit.** With a constant label there is no contrast for any slope and IRLS on a
   separable constant target runs the intercept to ±∞, so the readout is fit **intercept-only at
   the clamped base rate** (`q = clamp(mean(y), 1/(n+2), 1−1/(n+2))`). That is the
   maximum-likelihood readout of the state under that data, it keeps the grader answering, and it
   is replaced by a real fit the first cycle a label varies. `n_degenerate` counts these and
   `base_rate_fit` records the value, so the reduction shows how many refits were degenerate; at
   run scale it should be zero or only the earliest.
2. **P-4r distinguishes an undefined AUC from a missing one.** A single-class held-out slice has
   no AUC by the rank identity. So where no AUC was recorded the gate asserts the slice *was*
   single-class throughout; and where the slice carries both classes it asserts
   `n_refit > n_degenerate` — the readout must have fit the state and not just the base rate.
   That is the strong form, and the paid tag is where it binds.
3. **V-6b(proj_vs_rand) is measured always and asserted only where it can be.** With every verdict
   at 0 both arms' readouts are the same intercept-only constant, so the two arms are
   bit-identical — a fact about the preflight substrate, not about the wiring. The gate therefore
   records `both_arms_fit_more_than_the_base_rate` plus the realised `max|Δe|` and asserts the
   difference only when both arms fit more than the base rate. The live/random contrast is a claim
   on the **paid** tag, read in [PJ], and not at preflight scale.

The perturbation P-5 must fail on is the defect verbatim (a constant-label buffer gets no fit);
the harness reports it red, at **73/73**.

**§7.3 The degenerate fit's weight vector was built on the host, and a CPU-only gate could not
see it.** `vo_pj_irls` returns its solution on the design matrix's device, so every *solved*
readout was already on the GPU; the intercept-only vector of §7.2 is *constructed* rather than
solved, and its first version was a plain `torch.zeros`. `_score` moved `Z` to the device and left
`w` where it was, so on a GPU the matmul raised
`Expected all tensors to be on the same device`. The whole P-block had passed 26/26 and the
harness 73/73 — on a CPU container, where the bug does not exist.

It took `al_pf2` down at `al_pf_pj` c8, inside `vo_run_probes -> predict`, which also *confirms*
§7.2's fix: the degenerate readout had become ready and was being asked for a verdict on a probe,
which is exactly the path that was dark before.

Correction, two parts. `_score` now moves **both** operands to the bank's device and the degenerate
vector is constructed on it. And, because the class of defect is "a gate that cannot see the device
the arm runs on", `vo_proj_check` takes a **`device` parameter** and a new GPU entrypoint
`proj_gates_gpu` runs the identical P-1 -- P-5 block on an L4: two minutes there against
forty-five for a preflight. It passes on `cuda` with the degenerate readout answering at a flat
p = 0.0065 and `ready = True`.

The general lesson, recorded because it is not specific to this node: every gate in this lineage
that constructs tensors runs on CPU for speed, and a fork that adds a device-resident organ needs
at least one gate on the device. The arc had no such gate before this round.

*(further entries appended as the round produces them)*
