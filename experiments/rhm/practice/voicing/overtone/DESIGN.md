# overtone — DESIGN

**Up**: [`../README.md`](../README.md) (the `voicing` node) · **Machinery record**:
[`FILES.md`](FILES.md) · **Gates**: `voicing.py::ov_gates_cpu` and
[`gates/falsify_ov.py`](gates/falsify_ov.py) · **Run of record**:
[`results/RUN_ov_s0.sh`](results/RUN_ov_s0.sh).

Decisions and withdrawals, kept beside their corrections in this file rather than by editing
the decision that was wrong. Nothing here pre-registers an outcome; per the repo's norms the
readings are made after the data.

## §1. What this round is a child of, and what it is asking

`voicing` Q3b measured a **judge**: an MLP over the executor trunk's pooled hiddens plus a
candidate tuple's level-1 features, gradient confined, trained by BCE on the grader's verdict
over the learner's own filed writes, and composed with the trunk's own max-sum likelihood
(`z(dp/span) + w·z(critic)`). At seed 0, yoked to the anchor's ladder, that chooser is above the
anchor at 6 of 6 L4/L5 cells and ≈19% relative up at the levels every seed reaches.

Two readings put a question under that result.

- `reading/emotion_in_cortex.md`: the biological value readout sits **outside** cortex, reads a
  broadcast copy of it, and is trained by outcomes. That is the arrangement Q3b built, and it
  is the reason the critic is a separate organ rather than a loss on the emission head (Q1).
- `reading/Epistemic State Representations in Large Language Models.pdf` (Steenwyk, preprint):
  a next-token base model already carries a **linearly readable** correctness direction. A
  supervised linear probe on the residual stream reaches AUC 0.86–0.92 for answer correctness;
  cheap contrastive vectors reach 0.80–0.87; both carry information **beyond the model's own
  logprob**; instruction tuning erases some of it.

Read together they ask one thing of the judge: **how much of its job is already in the main
model, readable by a probe, and how much needs the verdict diet?** Three shapes, all on the
built machinery, all at seed 0, all yoked to the same banked anchor.

| | the question | the form |
|---|---|---|
| **S1** | how much is in the READOUT'S SHAPE? | a linear twin of the critic, and a one-direction twin with no slot embedding, trained on the critic's own rows and audited on the critic's own held-out split |
| **S2** | how much needs the VERDICT AT ALL? | the DP's own score of the candidate and the trunk's confidence at the slot's blocks, ranked against the same verdict on the same rows; then an out-of-fold logistic combination, so the critic's beyond-prior increment is a number |
| **S3** | can the probe budget be SPENT BETTER? | the same `vo_probe_n` aimed where the prior and the critic disagree, at the same bill |

## §2. The three shapes, as built

### S1 — the shadow readouts

`Critic` grew two parameters, both defaulting to the donor's value:

- `hidden_mult = 0` replaces `Sequential(Linear(dim,h), GELU, Linear(h,1))` with
  `Linear(dim, 1)`. Every other value builds the MLP the donor built (gate R-7 asserts both
  halves, and `vo_gates_cpu` still reads 18/18 with `V-4b` at 0.000e+00).
- `ctx_mode = "mean"` replaces the critic's context state — `ctx(pooled.mean) + slot(id) +
  Σ_j in_proj_j(pooled[blk0+j])` — with `ctx(pooled.mean)` alone. Combined with
  `hidden_mult = 0` that is **one direction over the pooled mean plus the candidate's
  content**, which is the Steenwyk shape as closely as this substrate admits.

Two shadows are declared by `ov_shadow`: `lin` (linear readout, the critic's own context) and
`dir` (linear readout, the mean context, no slot). Both are **trained and scored and never
consulted**.

**Why they ride the critic's batch rather than drawing their own.** The brief allowed a `crng`-
style stream per shadow. Taking the critic's own `pick` instead is strictly better on both
counts that matter: the shadows then consume **no draw from any stream at all**, so the treated
arm's bit-identity with the banked cell is a property of the construction rather than of a
bookkeeping discipline; and they see *exactly* the rows the critic sees, so the comparison is
within-row and the readout's shape is the only difference. Their gradient never touches the
trunk (`pooled.detach()`, unconditionally, whatever `vo_critic_trunk` says) and they step
**their own** `Adam`, never `gopt`. Gate R-1 is the claim; gate R-2 is its liveness dual.

### S2 — the zero-verdict scores

Four scores per held-out row, all read off tensors the audit already computes, all oriented so
that higher means "more likely to solve":

| column | what |
|---|---|
| `dp` | the DP's per-entry score of **the candidate**, divided by the span (the file's own convention for this quantity — `vo_compose`'s first line) |
| `dpz` | the same, z-scored over the candidate set: **exactly** the prior term the composed chooser sums, in the chooser's own units |
| `conf` | the negative mean entropy of the trunk's next-feature distribution over the slot's blocks — the model's confidence *where it is about to write*, candidate-blind |
| `marg` | the mean top1−top2 logit margin over the same blocks |

Then the Steenwyk move: an **out-of-fold** logistic combination of (`dpz`, critic logit), fit by
IRLS with a small ridge, folded by the audit's own bijective hold code (not a coin flip — the
reason `hold_code` exists), scoring every row from a fit that never saw it. `prior_oof` is the
same estimator on the prior alone, and the reduction prints
`incr = comb − max(prior_oof, crit)` — the critic's **beyond-prior increment**. Gate R-4 shows
that an in-fold fit inflates and the out-of-fold one does not.

**The asymmetry is §26's and is kept.** On FILED rows the written candidate is usually the DP's
own argmax, so `dpz` reads the prior's confidence in **its own choice**; on PROBE rows it reads
the prior on a **substituted** class. Different questions; the audit reports them apart and
never pools them, exactly as it already does for the critic's AUC.

**A denominator that has to be said.** The buffers outlive the table: a row filed before a
commit or a merge can spell a class the *current* operative table no longer holds, and the DP
has no score for a candidate that is not on the table. Those rows are excluded from the `dp`,
`dpz` and combination columns and counted (`n_ontable`), and the critic's own AUC is re-read on
that same sub-population (`crit`) so the comparison and the increment share one denominator.

### S3 — the probe budget spent by disagreement

`ideas/heterogeneous_graders.md` §9 at the smallest scale it can be tested. §26 substitutes a
class drawn **uniformly** over the classes the book holds at the slot, excluding the one
written. S3 substitutes the on-table row, of a class other than the one written, that maximises
`|z(dp/span) − z(critic)|` at that context — the two terms the composed chooser sums.

Three decisions:

1. **The context draw is untouched.** The same `sel` rows are probed; only the candidate moves.
   The brief offered "contexts where the two argmaxes differ" as an alternative; that form moves
   two things at once (which contexts, and which candidate), and the candidate form keeps the
   confound to the one axis the idea names.
2. **A fixed share stays uniform** (`ov_probe_unif_frac`, 0.25). The confound the brief names is
   real: the probe AUC would otherwise be read on a harder, selected candidate set and would not
   be comparable to `voi3b_comp_pr_yk`'s. The rows carry a draw tag (`VoRecorder.pmask`), the
   audit reports `unif_auc` and `dis_auc` apart, and reduction [J3] states in the header that
   the `dis` column is **not** comparable to the bank.
3. **The bill is identical per governed slot per cycle** — `take` gradings either way, one per
   context. Gate R-5 asserts billed-equals-rows under the new rule, and V-4B's residual-0 claim
   is unchanged.

   **Correction, from `ov_pf1` (kept beside the decision rather than replacing it):** the
   *total* bill over a run is **not** identical, and saying "the bill is identical" without the
   qualifier was wrong. Gate R-6 read `n_probe` 211 against the uniform twin's 116 over 27
   preflight cycles. The rule spends the same budget at every governed slot at every cycle; but
   a live treatment diverges from its twin, so *which* slots are governed and how many filed
   contexts each holds diverge too, and the running total follows. The invariant that holds and
   is gated is per-slot-per-cycle; the share of the cycle ledger is the quantity to read in
   reduction [O], as it was in Q3b.

**The one unpriced cost, said out loud rather than buried.** The allocation reads one extra
trunk forward over `take` rows per governed slot per cycle, to get `pooled` and the block
logits. It is **not billed**, for the reason the audit's larger forward is not billed: the meter
prices *groundings*, and this costs none. It is smaller than the audit's own per-cycle forward
(`take` ≤ 64 rows against the audit's ≤ 1024 per slot).

## §3. The dump

Nothing in this lineage has ever banked the critic's rows, which is why every question about its
diet has cost a fresh 1.3 GPU-h arm. `ov_dump` writes, per arm at end of run, `vo_rows.npz`
(+ `vo_rows_meta.json`): per slot and per buffer, `obs` and `write` as int16 (asserted lossless,
not assumed), the verdict, the row's hold code, the draw tag for probe rows, and the **DP score
of that candidate recomputed through the final core** — recomputed rather than stored at file
time because the trunk moves under the buffer, which is the audit's own convention. Gate R-8
asserts the round-trip and the DP column. It is unpriced, RNG-sandboxed, wrapped so a dump
failure cannot cost a paid arm, and it costs a few MB.

## §4. The arms, and what each is against

All three are `voi3b_comp_pr_yk` — the Q3b cell whose seed-0 numbers are banked — with one thing
changed, yoked to `vo_s3:voi3_dp` (the anchor, banked twice over and bit-identical to
`en_s9:endo_ledger_open_ung5_ra`).

| arm | the one change | what it is read against |
|---|---|---|
| `ovt_comp_pr_sh` | carries the S1 shadows, the S2 scores and the dump | **banked** `vo_s3b:voi3b_comp_pr_yk` — and must be **bit-identical** to it (gate R-1, full scale) |
| `ovt_comp_pr_dis` | + `ov_probe_dis` | `ovt_comp_pr_sh`, the bank, and the anchor |
| `ovt_comp_pr_lin` | + `ov_critic_hidden = 0` (a linear readout in the **chooser's seat**) | the same three |

`dp` is not re-run: the anchor is banked twice and the yoke reads its plan off the volume.
`voi3b_comp_pr_yk` is not re-run either — it is `ovt_comp_pr_sh`'s own identity target, which is
the point of building the shadows to be inert.

**Why S1 gets a governing arm at all.** The brief left it to judgement at 1.3 GPU-h. S1's
shadow half answers "could a linear readout have *predicted* as well"; it cannot answer "would a
linear readout have *chosen* as well", because a composed chooser's argmax over a candidate set
is not a monotone function of AUC. The third leg of Jasper's question is about the seat, so the
seat gets an arm.

**Withdrawn**: a second Steenwyk-shaped *governing* arm (`ctx_mode="mean"` in the seat). The
`dir` shadow already says whether that readout can rank; putting it in the seat would be a
fourth arm for a question the third arm's result will frame better.

### S1b — the correction: S1 as run tests ADDITIVITY, not linearity

**Kept beside S1 rather than replacing it, because S1 as built is still the right measurement
of the thing it measures — it just measures a narrower thing than "linear".** `lin` and `dir`
are `Linear(dim, 1)` over `u_ctx + e_cand`. A linear map of a sum is the sum of the maps, so
within one context the candidate ranking is `w · e_cand` plus a per-row offset: **the same
candidate order in every context**. Such a readout can hold a respectable *across-row* AUC — it
reads context difficulty and a candidate's base rate — while carrying no "this candidate
**here**" at all. That is exactly the shape of `ovt_comp_pr_lin`'s result: `moved` 0.0001 at L5
and 0.031 at L4, against an AUC within 0.03 of the MLP's on filed rows.

Steenwyk's probe is linear on the state **after** the answer is in the context, and the trunk
mixes context and candidate before any linearity is imposed. The RHM analogue is therefore a
linear probe on the trunk's pooled hiddens of the **post-write configuration** — the write
rendered into the masked context, the same rendered object the probe channel grades. With
`vo_heads.pt` banking the trunk and the dump holding (obs, write, verdict), this is CPU-only
and is `analyze_dump.py --post`, section **[S1b]**. Protocol: fit on the rows the critic
*trained* on, score on the rows the critic was *audited* on, filed and probe apart, beside the
MLP's and the prior's AUC on the identical rows.

**Two controls, and the second was not planned.** `pre_slot` is the same features with the
write *not* in the state, so whatever it reaches is context difficulty. `rand_slot` is
`post_slot`'s features through a **randomly initialised trunk of the same architecture** — and
it exists because the section's dry run was done against a synthetic `vo_heads.pt` whose core
was random, and the probe still reached **0.754** on probe rows. A random transformer is a
nonlinear feature map of the token sequence, so a linear probe on its post-write hiddens reads
nothing the trained trunk learned. That number is the floor, and `post_slot` is read against
it, not against 0.5. Accidents of this kind are the reason a dry run is worth its wall clock.

**One capacity caveat, since three of the four probes are compared directly.** `pre_slot`,
`rand_slot` and `post_slot` are all 2·dim = 192 features, so `post_slot` against `rand_slot` is
matched — same architecture, same feature count, same rows, only trained weights against random
ones. `post_mean` is 96 and is not matched to them; it is there to say whether the slot's own
blocks carry anything the sequence mean does not.

## §5. The gate table, and what each was shown to fail on

The lineage's rule is inherited verbatim: **a gate is not reported until it has been shown to
fail on a deliberate perturbation of what it protects**, and a knob whose purpose is to change
behaviour needs a liveness gate whose perturbation is **disconnection** (`../DESIGN.md` §30).
Perturbations live in [`gates/falsify_ov.py`](gates/falsify_ov.py).

See [`FILES.md`](FILES.md) for the table with denominators. Two notes belong here:

**One perturbation is NOT caught, and is reported rather than quietly dropped.** R-4's folds are
taken from `hold_code`; replacing them with a coin flip leaves the gate green, because on the
gate's own synthetic rows a coin flip still splits them. The bijective split is inherited from
`hold_code` and gated there (`vo_critic_terms`/`vo_critic_audit`, Q2), not here. Stated, not
claimed.

**R-2's containment half had to be rewritten after its first perturbation came back green.** The
first attempt added a shadow-only penalty to the critic's term; the gate stayed green and was
right to — the perturbation did not reproduce the hazard, since the shadow's graph is
structurally disjoint from the critic's. The gate now also reads, *before* the caller's own
backward, how many critic and trunk parameters are carrying a gradient after the shadow step
(must be 0), and the perturbation is an **aliased** head whose optimizer reaches the critic.
The correction is kept here beside the decision that was wrong.

**R-5's first perturbation was also a no-op, for a reason worth keeping.** Negating `ov_z` does
not perturb the rule: `|(−a) − (−b)| = |a − b|`. That is why the pick is a named function
(`ov_pick_disagree`) rather than three inline lines — so that "picks agreement instead of
disagreement" is reachable at all.

## §6. Compute shape, and the round's one departure from the lineage's launch habit

`vo_s3b` ran four arms **sequentially in one container**: one 606 s setup plus 4 × ~70 min, so
5.1 GPU-h took 4.9 h of wall clock. The arms in this round are stream-independent by
construction (`run_arm` reseeds per arm from `TWIN`/`STREAM`, which is why cross-tag bit
identity is possible at all), and the substrate is deterministic from `train_seed`, so they can
run in **three containers at once**: ~4.05 GPU-h instead of 3.7 (three setups instead of one)
for ~1.4 h of wall clock instead of ~3.7.

They share one tag directory on the volume. Three files there are written by all three jobs —
`setup.json`, `summary.json`, `done.txt` — and are last-writer-wins. **Nothing the reduction
reads is among them**: `analyze_voicing.py` reads per-arm `results.json` only, and each job owns
its own arm directory. The waiter is therefore on the launch logs / `modal app list` and not on
`done.txt`.

Each job also passes `--ref-tag vo_s3b`, which asserts in-run that this round trains the same
substrate as the banked tag (`refs`, the stale value buffer, `read_acc`). That is free, it fails
fast if it fails at all, and it is the **precondition for gate R-1** — a cross-tag bit-identity
cannot hold if the substrate moved.

`memory=32768` is the donor's inherited request on a runner whose models are tiny. The runner
now prints peak RSS at the end of a tag so the next launch can be sized on it; this round's
request is left unchanged so that nothing but the round's own knobs differs from `vo_s3b`.

## §6b. What seed 2 returned (2026-09-15), and what it does and does not replicate

One arm, `ovt_comp_pr_dis` at `--seed 2`, yoked to banked `vo_s3d2:voi3_dp`. Gate Y-1 exact:
commits [59, 77, 157], advances [60, 110, 180, 192, 201], **0 cancelled**. Seed 2's ladder
reaches L4 and not L5, so there is no L5 cell to read and the L4/L5 pooled column IS the L4
column. The uniform twin at this seed is banked `vo_s3e:voi3b_comp_pr_yk`.

**Replicates.** The S3 arm over the ANCHOR at the levels every seed reaches: **+38.8%** relative
pooled repair accuracy at L2/L3 (n = 197,044 against the anchor's 195,077), against **+31.7%** at
seed 0. Over the uniform twin, **+5.7%** against seed 0's +2.5% — same sign, same small size.
Consumption-era error against the anchor: era-3 −0.064, era-4 −0.071, era-5 −0.144, all outside
the lineage's 0.08 matched-clock floor at era 5 and inside it at eras 3 and 4.

**Does not replicate: the S3 advantage over the UNIFORM TWIN at the frontier.** At seed 0 the
disagreement arm was −0.179 (era 4) and −0.195 (era 5) below its twin, both 2.2–2.4× the floor.
At seed 2 the same three gaps are **+0.013 / −0.005 / −0.021**, every one of them inside the
floor. Pooled L4 repair accuracy is +3.7% over the twin here against +12.8% over it at seed 0,
and per cell the arm is above the twin at 2 of 4 where at seed 0 it was above at 4 of 6. This is
`voicing`'s own frontier caveat happening again, at the same magnitude it warned about: the
era-5 figure moves 3–4× between seeds.

**Does not replicate: the prior's sign on the selected rows.** From the banked dumps, the prior's
AUC against the verdict on the rows the rule drew reads **0.404 at seed 0 and 0.765 at seed 2**
(uniform-drawn rows 0.683 and 0.717, which do replicate). `|z(dp) − z(critic)|` is symmetric and
selects large disagreement in EITHER direction, so which organ is the extreme one on the selected
rows — and therefore the sign of the prior's AUC there — is not a property of the rule. The
seed-0 reading that "the rule selects rows the prior gets wrong" is withdrawn as a general claim
and kept here beside its correction; at seed 0 it is what the number said, and at seed 2 it is
not. What survives is that the selected substitutions are much less likely to solve: probe solve
rate 0.0168 here and 0.0437 at seed 0, against filed rates of 0.313 and 0.322.

**S1 and S2 replicate closely**, on the same arm at both seeds (medians over reads; `ov_s0`'s
`prior`/`comb` are the pooled form and are excluded, the single-feature columns are not
affected):

| | dpz | conf | crit | `lin` | `dir` |
|---|---|---|---|---|---|
| FILED seed 0 / seed 2 | 0.575 / 0.558 | 0.592 / 0.585 | 0.685 / 0.661 | 0.632 / 0.628 | 0.616 / 0.632 |
| PROBE seed 0 / seed 2 | 0.746 / 0.704 | 0.500 / 0.540 | 0.876 / 0.932 | 0.786 / 0.853 | 0.776 / 0.844 |

Every filed column is within 0.024 across the seeds. The ordering on probe rows is identical at
both (MLP > lin ≈ dir > prior > confidence), the MLP-minus-linear gap is 0.053 → 0.033 on filed
rows and 0.090 → 0.079 on probe rows, and the trunk's candidate-blind confidence is at or near
chance on probe rows at both (0.500, 0.540). The corrected combination at seed 2 reads FILED
0.633 against a critic of 0.661 and a prior of 0.543, and PROBE 0.942 against 0.932 and 0.717 —
the same shape as seed 0's: no gain over the critic on filed rows, a small gain on probe rows.

## §7. What this round cannot say, stated before the data

- One seed. The lineage's own caveat stands: matched-clock error gaps between near-identical
  arms reach 0.08 here, and the one magnitude `voicing` states as a magnitude is the era-3 gap.
- The shadows are readouts over **this** trunk, which is a small span-net corridor head's
  substrate, not a language model's residual stream. Nothing here transfers a number from the
  Steenwyk preprint; what transfers is the *shape* of the question.
- `w = 1` and the z-score composition are Q3b's and are not swept; `ov_critic_hidden` changes
  the readout inside that fixed composition.
- The probe verdict still answers §26's question — "would this write **alone** have been right,
  holding the rest of the trajectory fixed" — not "would the beam have solved it".

## §8. Defects found this round

1. **`vo_probe_offstream_check` and two `falsify.py` perturbations unpacked the probe part as a
   3-tuple.** The draw tag made it a 4-tuple. Caught by `vo_gates_cpu` (V-4d raised) within
   minutes of the change, before anything was launched. Fixed by unpacking positionally and
   keeping whatever tail a part carries, so a §26 part and an S3 part both read.
2. **Gate R-5's first reference compared ROW INDICES.** The operative table can hold the same
   level-1 tuple at more than one row, and the DP's per-entry score is a function of the tuple
   alone, so duplicate rows tie and `argmax` breaks the tie by index. The gate read 6/12 and was
   right to. Fixed by comparing the **candidate**, which is VO-6's own correction one organ over.
3. **Gate R-5's reference then read the contexts in the wrong order.** `vo_run_probes` draws its
   contexts without replacement from its own stream and does not return the selection. Fixed by
   replaying that stream in the gate — which, as a side effect, asserts the draw is the one
   V-4d's construction assumes.
4. **R-7 crashed instead of failing** when its perturbation made the readout a `Sequential`. A
   gate that raises is a red gate, but it should say so; the linear reference is now guarded and
   the harness reports a raise as a red gate explicitly.
5. **The out-of-fold combination POOLED its folds' scores into one ranking**, and that is the
   one defect this round shipped into a paid tag. Each fold's out-of-fold score is that fold's
   own affine map of the features; the folds' intercepts differ whenever their base rates
   differ, which they do here, because the held-out set is small and the same contexts recur
   across cycles. The pooled ranking then ranks fold membership as much as it ranks the verdict.
   **Measured** on synthetic rows with a known single-feature AUC of 0.698: pooled reads 0.482
   where per-fold reads 0.720.

   *What it did and did not touch.* It affects `comb_auc`, `prior_oof_auc` and the `incr`
   column derived from them, in `ov_s0` only. It does not touch a single other number in the
   round: `dpz`, `dp`, `conf`, `marg` and `crit` are raw scores with no fold structure, and
   [J1], [J3], [Z], [I], [O] and gate R-1 never go near the estimator.

   *How it was caught.* Not by a gate — by reading the output. `prior_oof_auc` came back at
   0.414 on rows whose raw `dpz` AUC was 0.585, and a monotone re-fit of one feature cannot
   lose 0.17 of AUC. The disagreement was checked against a synthetic construction before
   anything was concluded from it. **R-4 did not catch it because R-4's own rows have equal
   base rates by construction**, which is the sharper lesson: the gate tested the estimator on
   a population the run does not have. R-4 now carries the unequal-base-rate half, and goes red
   if `ov_oof_auc` ever reads the pooled figure again.

   *The repair.* `ov_oof_auc` takes the AUC **per fold** and averages by pair count, returning
   the pooled figure beside it so `ov_s0`'s numbers stay interpretable as what they were. One
   arm (`ovt_comp_pr_sh`) was re-run as tag **`ov_s0b`** to recover the increment; nothing else
   about the arm moved, and the re-run must be bit-identical to `ov_s0:ovt_comp_pr_sh` (the
   estimator lives in an unpriced instrument that consumes no draw).

   *Confirmed.* `ov_s0b:ovt_comp_pr_sh` came back bit-identical to `ov_s0:ovt_comp_pr_sh` and
   to banked `vo_s3b:voi3b_comp_pr_yk` at 0.000e+00 on all eleven behaviour series including
   `t_cum`, commits equal — the estimator is where it was claimed to be. Its [J2] is the S2
   column of record; `ov_s0`'s is kept and marked, and the audit now stores the pooled figure
   beside the per-fold one so the size of the defect is readable in both tags (pooled 0.558 /
   0.799 against per-fold 0.620 / 0.832 on filed / probe rows, pooled over 3,410 and 3,267
   reads).

   *The standing cost it removed.* The re-run was needed only because nothing in this lineage
   had ever saved the critic, so redoing an audit meant paying for the arm again. `ov_dump`
   now also banks `vo_heads.pt` — the critic, both shadow readouts and the trunk they read
   (~a few MB) — beside `vo_rows.npz`. With the heads and the rows banked, the next audit
   change is a CPU job over the dump. Gate R-8's second half reloads the heads and asserts the
   critic's scores come back identical.
