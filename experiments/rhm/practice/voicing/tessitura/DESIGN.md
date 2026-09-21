# tessitura — DESIGN

Decisions, and every withdrawn reading kept beside its correction. **Up**:
[`../README.md`](../README.md) (voicing) · **Machinery record**: [`FILES.md`](FILES.md) ·
**Reductions**: [`results/`](results/) · **Figures**: `figures/*.png` (gitignored under
`experiments/`; regenerate with [`plot_tessitura.py`](plot_tessitura.py)).

No README. Interpretation waits on a discussion with Jasper; the facts are in the reductions and
in this file.

---

## §0 What this node is

`logit_reading/striatum/norm/` asked three things of an outcome-trained critic on a **frozen
trunk**, a fixed world, dense free feedback and no agency:

1. does the critic's pre-event level calibrate to the world it was fed, and as a shift or a
   rescaling;
2. at the event, does it read the violation's cost or its structure;
3. on same-prefix twins, how does it price the illegal token against its legal twin.

`voicing`'s judge is the same kind of object under the opposite conditions: an outcome-trained
readout on the **executor trunk's** pooled hiddens, trained online by the grader's verdict on the
learner's own filed writes, on a live learner minting its own vocabulary under priced feedback.
This node asks norm's three questions of it.

Three things differ and are kept in view rather than mapped away:

- **The judge is Q-shaped.** `critic(context, candidate) → P(solve)`, not a state value. A norm
  has to be *derived*, and which derivation is the norm is a choice (§2).
- **The event is the write**, not a violation arriving in a stream. The pre-write state is `obs`
  with the slot's span masked; there is no "anticipatory response" to separate from the outcome
  surprise, because the write and the outcome are one act.
- **The judge is trained online through an era ladder** whose damage depth rises 1 → 5. That is a
  *within-subject* shift in expected cost — Xiang's actual design — which norm's between-critic
  diets could not give.

---

## §1 What the banked record can and cannot answer (decided after the zero-cost pass)

**The norm is not in the log.** `vo_critic_audit` logs, per slot per cycle, `n`, `auc`,
`base_rate`, `probe_n`, `probe_auc`, `probe_base_rate` (plus `overtone`'s `free` and `sh_auc`).
Every one of those is either the **world** or a **rank statistic**. The judge's *level* appears
nowhere: `vo_critic_terms` computes a per-slot `cstat` carrying the batch's base rate and the
call site discards it (`voicing.py` ~line 2248, `cterm, cstat = vo_critic_terms(...)`, and
`cstat` is never read again — grep returns one hit).

So the zero-cost pass ([`results/logged.txt`](results/logged.txt), 24 arms with a critic across
7 tags, 3 seeds, 5 diets and 4 graders) is the **world side only**, and that is its value: it
says what outcome distribution each judge was fed, per era and per slot, which is what a norm
would have to calibrate *to*.

**The norm IS recoverable statically**, from the two arms that banked `vo_heads.pt` beside
`vo_rows.npz` — `ov_s0b:ovt_comp_pr_sh` and `ov_s2:ovt_comp_pr_dis`. `ov_s0`'s three arms banked
rows but **not** heads (the heads were added for `ov_s0b`), so the brief's "five arms carry
`vo_rows.npz` and `vo_heads.pt`" is three arms short on the heads. Two limits, repeated at every
table in [`results/rows.txt`](results/rows.txt): the rows are the ring-buffer tail (~60 filed and
~128 probe cycles, era 1 gone) and the heads are the final ones. It is a static read of a reader
that was trained online.

**Decision.** Run the zero-cost and CPU passes, then the re-run, because the within-subject time
course is not in the bank in any form and it is the question the frozen trunk could not ask.

---

## §2 The norm's derivation, chosen and reported as three

A state value from a Q needs a rule. All three are reported and none is privileged in the
numbers; the one the *run itself* consults is `V_w`.

| symbol | what it is | why it is a candidate |
|---|---|---|
| `V_w` | `σ(critic(ctx, the candidate actually written))` | the judge's expectation at the write it will make — the only one the governed chooser ever evaluates at the write |
| `V_max` | `max` over the slot's candidate set | the slot's value under a judge-optimal write |
| `V_mean` / `V_tab` | `mean` over that set | the slot's value under a blind write |

In the CPU pass the "candidate set" is the slot's **realised** set (every distinct tuple that
appears as a filed write or a probe substitution), because the dump does not carry `move["flat"]`.
In the re-run it is the operative table itself, and the max over table rows *is* the max over
classes (a class's value is the max over its spellings, `vo_class_max`); the mean over table rows
is **not** the mean over classes and is labelled `V_tab` for that reason.

On filed rows `V_w` is the judge's expectation of what it chose; on probe rows it is its
expectation of a substitution drawn off-policy. Different objects; never pooled.

---

## §3 The twins are stronger here than in norm, and the reason is in the construction

`voicing`'s probe channel takes a filed context, substitutes a **different** on-table class at
that slot in the trajectory's final configuration (gate V-4d asserts "different", exactly and not
on average), and pays the grader for one verdict. So every probe row is a filed row with one
class swapped at an identical context — a same-prefix twin **by construction**, and unlike norm's
twins the *outcome* is matched too: both verdicts are on the same trajectory's final
configuration with one slot's content swapped. norm's caveat ("after `t_v` the twins share the
edited continuation, so only offset 0 is a clean event reading") has no analogue here.

**The join is on the context, and a context is a group.** A masked context recurs across beam rows
and across cycles, so the reduction groups filed rows by their `obs` bytes and pairs each probe
row against the group mean. `split` — the share of contexts whose filed verdicts disagree with
each other — is printed per slot (0.09–0.18) as the pairing's own noise.

**The dependence on the pre-write expectation needs a context-level `V_pre`.** `dQ = logit(c) −
logit(w)` contains `−logit(w)` by construction, so regressing `dQ` on `V_w` is norm §3's
regression-to-the-mean with nothing cancelling it — here it cannot be cancelled by pairing,
because the twin is a *candidate* and not a second row. `V_mean` (the judge's mean over the
candidate set at that context) is candidate-blind and is the primary regressor; the `V_w`
regression is printed beside it and labelled the confounded comparator.

---

## §4 Gate T-4 — the structural label is NOT recoverable from the bank. WITHDRAWN and kept.

**What was tried.** Q2 needs, per row, the `rep` instrument's label: does the written class hold
a feature in the repair set at that slot (`fourwall/wall.py::consistent_features`)? Two facts,
and the first is good news:

- `consistent_features` **overwrites** the slot's token span `[node·s^level, (node+1)·s^level)`
  before it parses, and that span is exactly what `VoRecorder.assemble` masks to −1. So the mask
  is irrelevant and `obs` gives the identical label to the unmasked context. Gate **TS-4** asserts
  this as an identity on 48 rows and shows it breaking when any *other* span is masked.
- the **root** is missing from the dump and nothing in the dump determines it.

[`structure.py`](structure.py) reconstructs the root: a re-implementation of
`rhm_sculpt_precheck.nearest_derivation_cost` with the slot's bottom blocks zeroed after the
bottom layer (`free_span_cost`), which answers "how many token edits outside the slot would this
string need to derive root r, if I may write any legal unit at the slot". `root_hat` is the
argmin.

**It fails, and the failure is the finding.** Against the run's own `rep` rates over the last 30
cycles, the reconstruction is **biased upward: +0.064 mean signed residual, median |residual|
0.092, positive on 25 of 30 slots**, and ambiguous (two roots tie) on **41–80%** of rows. The
mechanism is identified: the free-span argmin prefers precisely the roots under which *some*
write at this slot repairs, because those roots reach cost 0 there, so a true root under which
nothing repairs loses to one under which something does — the reconstruction manufactures repair
sets. A deliberately wrong root (+1 mod v) lands at median |residual| 0.236, so the comparison
discriminates and the residual above is the reconstruction's own.

**Consequence.** No reading in this node rests on the reconstruction. Q2's structural half is
answerable only with the label taken at the write, where the true root is in hand — which is what
`ts_struct` adds. The reconstruction and its failure are kept in
[`results/structure.txt`](results/structure.txt) so the next agent does not rebuild it.

---

## §5 The re-run's three additions, and why each is where it is

All three are **run-level instruments**, default off, `# [tessitura]`-marked in `voicing.py`. None
can reach a decision: no gradient, no choice, the RNG sandboxed, the oracle calls counted in
`_EXP_REC["reads"]` as `rep`'s are and never in `counts["ground"]`. The arm itself is
`ovt_comp_pr_sh` **unchanged**, which is what makes gate TS-1 (bit-identity against banked
`vo_s3b:voi3b_comp_pr_yk`) free.

- **`ts_norm`** — inside `vo_critic_audit`'s `_score`, on the rows the base rate is already read
  from: `mean_p`, `mean_pmax`, `mean_ptab`. One `(B, R)` MLP over a context state the audit has
  already computed; no extra trunk forward.
- **`ts_panel N`** — `ts_panel_audit`, called beside the audit. One panel per `(slot, era)`,
  frozen the first cycle in that era at which the slot's buffer holds `N` rows (the **last** `N`,
  so the panel is that era's own contexts), scored by the **live** critic every cycle after. The
  panel's base rate is a constant, so `mean_p` drifting is the reader.
- **`ts_struct N`** — `ts_struct_audit`, called after the probes. `N` evenly spaced rows per slot
  per cycle from *this cycle's* filed rows and probe rows; the structural label under the true
  root; the judge's logit and the DP prior on the identical rows; and the AUCs of each scorer
  against each label, **marginally and with the other label held fixed**. The joint 2×2 is emitted
  too, so the two labels' own association is visible and never inferred from the AUCs.

**Two design notes that cost something to rediscover.**

- The probe channel's rows had no root. `vo_run_probes` now appends the root as a **fifth** tuple
  element **only when `ts_struct > 0`**; `push_probe` reads `p[0..3]` by index and already
  tolerates a longer tuple (its `len(p) > 3` guard, added by `overtone` for the draw tag). Gate
  **TS-6** asserts the element is absent with the knob off and that the probe buffer the critic is
  fed is bit-identical either way.
- `vo_rep_n` was **left at 64**. Raising it was the obvious way to get more structural rows, but
  `rep`'s grader calls are counted in `exp_reads`, and `ts_struct` needs its own sample anyway
  (the probe half has no `rep` at all). A separate knob keeps the banked `rep` series comparable
  across tags.

---

## §6 Gates

Offline, `voicing.py::ts_gates_cpu` (5/5), falsified by
[`gates/falsify_ts.py`](gates/falsify_ts.py) (11 of 12 perturbations turn their gate red; the
twelfth is blind **by design** and named as such). The arc's rule is inherited: a gate is not
reported until it has been shown to fail on a deliberate perturbation of what it protects.

| gate | claim | shown to fail on |
|---|---|---|
| **TS-2** | the judge's level is added *beside* the audit's own fields and displaces none of them; `mean_p` IS the mean sigmoid of the critic's logit on the audited rows and `mean_pmax` its max over the operative table; the read consumes no draw | the level reported as a mean logit · the level overwriting `auc` · the audit consuming a draw |
| **TS-3** | the panel freezes once per (slot, era) and never again; its base rate is a constant while the buffer under it moves; a later era adds a second panel and keeps the first; the reading moves when and only when the critic does | re-freezing on every call · scoring with a critic frozen at first sight (disconnection) |
| **TS-4** | the structural label is invariant to the span mask, because its own call overwrites exactly the span `assemble` masked; masking any *other* span breaks it | a label that reads the mask instead of overwriting it · (the root permuted is **blind by design** here — the root is T-4's subject, §4) |
| **TS-5** | the structural audit produces the repair-set membership and the verdict on the identical rows, counts its grammar reads in `_EXP_REC["reads"]` as `rep` does, and consumes no draw | reads not counted · the structural label replaced by the verdict · the audit consuming a draw |
| **TS-6** | the probe tuples gain a root element only when `ts_struct` is on, and the probe buffer is bit-identical either way | the root appended unconditionally · `push_probe` reading the fifth element into the verdict |

In the CPU pass, [`reduce_rows.py`](reduce_rows.py) carries its own:

| gate | claim | note |
|---|---|---|
| **T-2** | the dumped buffer IS the buffer the run's last audit read: `n` and `base_rate` come back exact on every buffer | 59/57 buffers, max &#124;Δn&#124; = 0, max &#124;Δbase&#124; = 1.5e-08 (the log stores a float32 mean). Falsified by dropping one held-out row. |
| **T-3** | the reloaded heads are the run's critic, and **cannot** be an identity on the scores | `vo_critic_audit` runs at `voicing.py:9729` and `finetune_generator_span` at `:10010`, so the banked heads are one optimizer step ahead of the last logged audit. Reported, not asserted to zero: &#124;ΔAUC&#124; median 0.018 filed / 0.009 probe. Falsified relatively — a re-drawn critic is 5.1× worse and sits at chance. |
| **T-5** | the dedupe (one trunk forward per distinct context) equals the naive per-row path | max &#124;Δlogit&#124; = 0.000e+00 |
| **T-4** | the structural label is recoverable offline | **FAILS** — §4 |

Not offline, and both now read:

| gate | claim | result |
|---|---|---|
| **G-F** | with every `ts_*` knob off, this file's `run_arm` reproduces the donor's | **PASS — 0.000e+00** on both arms, donor self-replay control also 0.000e+00 (`results/gf_and_preflight.txt`) |
| **preflight `ts_pf1`** | the callsite wiring resolves at the depth-6 grammar with the three instruments ON at dust budgets | **PASS** — `voi3b_pf_src` + `ovt_pf_comp_pr`, 27 cycles each, 508 filed writes with `verify_bad = 0`, the critic governed, 116 probes billed and filed |
| **TS-1** | the paid arm is bit-identical to banked `vo_s3b:voi3b_comp_pr_yk` on every behaviour series including `t_cum` | **PASS — 0.000e+00** on all eleven series, commits equal (`results/ts_s0_reduction.txt` section [R]) |
| **TS-1b** | *(unplanned, and stronger than TS-1)* the paid arm's banked `vo_rows.npz` and `vo_heads.pt` are **byte-identical** to `ov_s0b`'s | **PASS** — same md5 on both files. Every buffer row, every DP score and every parameter of the trunk and the critic came back bit for bit, which is an inertness claim the series comparison cannot make |
| **liveness** | the instruments must have *done* something | `exp_reads` 5,618,042 · 128 panels frozen · 6,897 structural records · 153 cycles carrying a level |

---

## §7 Readings recorded, with their limits attached

Everything here is a fact from a reduction; the interpretation is Jasper's to make.

**The world (zero-cost, 24 arms).** The era ladder is a real within-subject shift in expected
cost and it does **not** have one sign: at seeds 0 and 2 the filed solve rate *rises* across eras
(0.105 → 0.341 at `ov_s0b`), at seed 1 it *falls* (0.200 → 0.091), and one seed-0 arm
(`vo_s3:voi3_comp`) falls too. The learner improving and the damage deepening move it in opposite
directions and the banked record cannot separate them — which is a limit on reading any
era-to-era change as "the world got harder". The probe (counterfactual) rate is 3–20× lower than
the filed rate in every arm and every era.

**The diet's own lag.** The audit's `base_rate` is a trailing estimate: an EMA fit gives the filed
buffer an effective window of **20–50 cycles** at every full-length arm (rmse 0.011–0.032 against
a signal that moves by 0.2). Any measurement of the judge's lag has to be read against this floor.
The probe column's fit (τ 1–8) is **not** interpretable: the probe rate is nearly flat, so the fit
is ill-conditioned, and for the `sotto_voce` arms the probe buffer's "base rate" is the *mirror's
mean probability*, a different object from a verdict rate.

**Ranking quality over time.** The filed AUC is remarkably flat across eras in every arm
(0.60–0.73); the probe AUC *rises* strongly in every probe-fed arm (0.589 → 0.863 at `ov_s0b`,
0.645 → 0.903 at the disagreement arm) while the free prior's AUC on the same probe rows *falls*
(0.804 → 0.715, 0.769 → 0.596). The judge's increment over the prior on counterfactual rows grows
over the run.

**Q1, statically (CPU, two arms).** The norm calibrates to the world **in level, not only in
rank**, per slot: regressing the judge's mean P(solve) on the world's base rate across 30 slots
gives slope **+1.225 / +1.373** (seed 0 / seed 2) on filed rows with R² 0.86 / 0.75 and pooled
bias −0.023 / +0.012, and slope **+0.961 / +1.098** on probe rows with R² 0.93 / 0.86 and bias
+0.002 / +0.006. The reliability curve is near-diagonal over ten deciles at both seeds and both
buffers. In norm's vocabulary this is a *rescaling* with a slope slightly above one on the harder
(filed) rows and essentially unit on the counterfactual rows — but the comparison is **across
slots within one critic**, not across critics on identical rows, and that is a different
manipulation from norm's.

**Q1, across arms.** Each arm's critic scored on the other's rows through its own trunk (the
grammar is identical at every seed; a candidate is content, not an index). In 3 of 4 cells the
arm's own critic is closer to the world's base rate on its own rows than the foreign critic is
(seed-0 filed rows: base 0.336, own 0.308, other 0.286; seed-2 filed rows: base 0.335, own 0.345,
other 0.252). The two readers disagree row by row (R² of one on the other 0.16–0.40, slopes
0.37–0.56). **The caveat is large**: each critic carries its own trunk, so this is two whole
readers and not two readouts of one state, and the two arms differ in seed *and* in probe draw at
once. It is one contrast, not a family.

**Q2, cost side only.** On filed rows the judge reads the verdict at AUC 0.732 / 0.746 against the
prior's 0.578 / 0.587; conditioned on the prior the judge holds 0.720 / 0.730, while the prior
conditioned on the judge falls to **0.502 / 0.507 — chance**. On probe rows: judge 0.883 / 0.955,
prior 0.705 / 0.601, judge-given-prior 0.851 / 0.943, prior-given-judge 0.575 / 0.732. That is
striatum §5's containment with the roles reversed — there the *critic's level* subsumed the
surprise readout; here the outcome-trained judge subsumes the surface prior on the learner's own
writes and only partly on counterfactuals. **The structural half is absent** (§4).

**Q3, the twins.** Pooled over 123k (seed 0) and 117k (seed 2) context-matched pairs: the world
says a substituted class solves 0.069 / 0.016 against the written class's 0.324 / 0.324
(`dY` = −0.256 / −0.308), and the judge prices the substitution **−3.02 / −8.01 logits** below the
written one, which in probability is **−0.224 / −0.304** — within 0.03 and 0.004 of the world's
own contrast. The judge gets the **sign** of the world's twin contrast right on **0.93 / 0.98** of
the pairs where the world moved. This is the opposite sign to norm §3, where the critic was *more*
optimistic after the illegal token than after its legal twin; here the twin is a wrong write and
really is worse news, and the judge says so at close to the right size.

**Q3, the dependence on the pre-write expectation.** After the mechanical confound is removed
(§3), `dQ ~ V_mean` has slope −0.415 (r −0.010) at seed 0 and −2.205 (r −0.041) at seed 2 —
essentially nil at both. The confounded comparator reads −3.10 (r −0.24) and −2.48 (r −0.13).

**One seed difference, with a mechanism.** At matched prior (quintiles of the prior's own
difference) seed 0's judge shrinks its contrast as the prior's shrinks (−3.93 → −2.25, tracking
the world's −0.332 → −0.188), while seed 2's *grows* it (−5.93 → −9.18) against a flat world
(−0.320 → −0.282). Seed 2 is the **disagreement-drawn** arm: its probe candidates were selected to
maximise |z(dp) − z(critic)|, so its small-|Δprior| rows are by construction the rows where the
critic is the extreme organ. This is a selection property of `overtone` S3's draw, not a property
of the judge, and the two seeds are not comparable on this panel.

---

---

## §8 The re-run (`ts_s0`, 1.50 GPU-h, seed 0, 201 cycles, peak RSS 7.0 GB against a 32 GB request)

The arm is `RUN_ov_s0b.sh`'s verbatim plus `--ts-norm --ts-panel 128 --ts-struct 64`. Ladder
realised at commits [48, 100, 151, 186] and advances [60, 110, 180, 192, 201] — the anchor's,
exactly. Reduction: [`results/ts_s0_rerun.txt`](results/ts_s0_rerun.txt); the bit-identity is
[`results/ts_s0_reduction.txt`](results/ts_s0_reduction.txt) section [R].

### The norm's time course — the judge undershoots a rising world and converges by era 3

`V_w − base`, n-weighted over slots inside each era, on the rows the base rate is read from:

| era (damage depth) | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| world (filed base rate) | 0.137 | 0.180 | 0.240 | 0.311 | 0.319 |
| the norm `V_w` | 0.105 | 0.155 | 0.239 | 0.318 | 0.322 |
| **`V_w` − base** | **−0.032** | **−0.025** | **−0.001** | **+0.007** | **+0.002** |
| `V_max` over the table | 0.123 | 0.186 | 0.272 | 0.362 | 0.371 |
| filed AUC | 0.700 | 0.648 | 0.637 | 0.632 | 0.640 |

On probe rows the sign is the other way and the size is a third of it: `V_w − base` = +0.005,
+0.011, +0.013, +0.012, +0.012, with the probe AUC climbing 0.590 → 0.863. So the judge runs a
small persistent *over*-estimate on counterfactual rows and a shrinking *under*-estimate on its
own writes.

### The lag is the diet's, not the reader's

An EMA of each world series fitted to the judge's level, with the whole rmse curve printed rather
than only its argmin:

| fitted against | τ=1 | 2 | 3 | 5 | 8 | 12 | 20 | 30 | **50** | 80 | 120 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| the cycle's own solve rate | .1038 | .0927 | .0884 | .0828 | .0766 | .0698 | .0586 | .0482 | **.0390** | .0428 | .0536 |
| the ring buffer's trailing rate | .0381 | **.0379** | .0382 | .0394 | .0415 | .0443 | .0496 | .0549 | .0621 | .0677 | .0710 |

The judge's level is a ~**50-cycle** average of the world and a ~**2-cycle** average of its own
buffer; `corr(level, trailing base) = 0.896`, mean signed bias −0.0102. The ring buffer's own
window, measured independently in [`results/logged.txt`](results/logged.txt) [B], is 20–50 cycles.
**So essentially all of the judge's lag behind the world is the diet's and the reader adds one to
two cycles.** Both curves are well-shaped (rmse changes by 2.7× and 1.9× across them), which is
why the argmin is reported as a measurement here and not only as a number.

### The fixed panels — identical rows, and the level follows the era it is read in

Mean `V_w` over the panels frozen in each era, by the era they are read in. The `base` column is
those panels' own (constant) verdict rate:

| frozen in | panels | base | read in e1 | e2 | e3 | e4 | e5 |
|---|---|---|---|---|---|---|---|
| 1 | 16 | 0.102 | 0.143 | 0.144 | 0.205 | **0.245** | 0.229 |
| 2 | 24 | 0.148 | — | 0.149 | 0.200 | **0.244** | 0.235 |
| 3 | 28 | 0.317 | — | — | 0.250 | 0.305 | 0.295 |
| 4 | 30 | 0.339 | — | — | — | 0.376 | 0.364 |
| 5 | 30 | 0.396 | — | — | — | — | 0.371 |

**On rows that never changed, the judge's level rises by +0.10 from era 1 to era 4** — the norm
re-calibrating, with the world held out of it by construction. It is not full convergence: read
in era 4, era-1 panels sit at 0.245 and era-4 panels at 0.376, so the judge still tells the rows
apart. The *ranking* on those same frozen rows moves differently and much less: era-1 panels read
0.728 / 0.772 / 0.726 / 0.635 / 0.621 across eras 1–5. **The level drifts with the world; the
ranking on old rows decays slowly.**

Two caveats attached at the table. **The trunk moves under the panel**, so what drifts is the whole
reader (critic + trunk) and no reduction can split them on a live learner — this is exactly what
norm's frozen trunk could hold still. And a panel frozen at the *first* cycle of an era is built
from a buffer still dominated by the previous era's rows, so the `base` column is a mixture and
the freeze-time calibration (`V_w` vs `base` in the diagonal cells) is **not** a clean reading;
the drift across a row of the table is, because the rows never change.

### Cost versus structure — and it runs the OPPOSITE way to striatum §2

The structural label is `rep`'s, taken per row under the true root, on 64 filed and 64 probe rows
per slot per cycle (≈210k filed and ≈208k probe rows over the run). The two labels are strongly
associated on this substrate: `P(y | struct)` runs 0.53 → 0.82 by era against `P(y | ¬struct)`
0.04 → 0.14.

| filed, by era | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| `crit ~ y` (cost, marginal) | 0.596 | 0.577 | 0.594 | 0.642 | 0.617 |
| `crit ~ struct` (structure, marginal) | 0.598 | 0.572 | **0.631** | **0.708** | **0.678** |
| **`crit ~ y` at matched structure** | 0.553 | 0.558 | **0.525** | **0.549** | **0.521** |
| **`crit ~ struct` at matched cost** | 0.585 | 0.534 | **0.608** | **0.702** | **0.663** |
| `dp ~ y` / `dp ~ struct` | .657/.673 | .586/.598 | .568/.594 | .573/.604 | .569/.617 |

Probe rows, same columns: `crit ~ y` 0.554 → 0.831, `crit ~ struct` 0.557 → 0.857, cost at matched
structure 0.578 → 0.769, structure at matched cost 0.557 → **0.849**.

> **At matched structure the judge is near chance on the cost (0.52–0.56 from era 2 on); at
> matched cost it reads the structure at 0.53–0.70 and rising with the era. striatum's
> outcome-trained readout found the damage and not the structure; this one finds the structure at
> least as well as the cost, and what survives conditioning is the structure.**

**The interpretive fork is a design difference and is stated rather than resolved.** striatum built
its consequence label to be independent of `k*` and of surprisal *by construction*, so cost and
structure were dissociable there. Here they are not: whether the written class repairs the
instance at that slot is close to the proximal cause of whether the trajectory solves, and an
outcome-trained readout that learns the outcome's cause will look like it reads structure.
Whether that makes this a reversal of striatum's finding or a different experiment is Jasper's
call, and the numbers needed for it — the 2×2, `P(y|struct)`, `P(y|¬struct)` and the repair-set
size — are all in the reduction.

By slot level, pooled over the run (filed): structure-at-matched-cost is 0.587 / 0.639 / 0.612 at
L2 / L3 / L4 and **flips to 0.598 against cost's 0.629 at L5** — on 1,424 rows, so it is a cell
count and not a claim.

### §8c The second seed (`ts_s2`, 1.89 GPU-h): one headline survives as stated, one as a weaker claim, one not at all

`ts_s2` is `ts_s0`'s arm at seed 2, yoked to `vo_s3d2`, same three instrument flags, so the seed
is the only thing that moves. Gate **TS-1: 0.000e+00** on all eleven behaviour series against
banked `vo_s3e:voi3b_comp_pr_yk`, commits equal. Ladder realised at [59, 77, 157] L[2, 3, 4] —
seed 2's own, and it does **not** reach L5. Peak RSS **6790 MB** against the new 16384 request.
Full table: [`results/seed_signs.txt`](results/seed_signs.txt).

**An exclusion criterion, mechanical and stated before the tables.** A critic governs only once a
slot holds `vo_critic_min = 256` filed rows, so an early enough era can contain little but the
readout's **initialisation** — and `build_critic` zeroes every bias and draws weights at
N(0, 0.02), so an untrained critic emits a ~0 logit and reads 0.500 at *every* candidate. A cell
is flagged `init` iff |V_w − 0.5| < 0.01 **and** V_max − V_tab < 0.005 (the judge cannot tell its
candidates apart). The separator is not marginal: seed 2's era 1 has that gap at **0.000292 on
671 rows** while its era 2 has **0.106531 on 309,334**. Exactly one cell in either seed is
flagged. Every pattern below is printed **both ways** and nothing is dropped.

**Headline 1 — WITHDRAWN as stated.** `V_w − base` per era: seed 0 `−0.032, −0.025, −0.001,
+0.007, +0.002` (`---++`), seed 2 `+0.388*, +0.002, −0.011, −0.033, −0.027` (`*+---`). With the
init cell excluded the sign patterns are `---++` and `+---`: they **do not agree**. So *"the norm
undershoots a rising world and converges by era 3"* is a **seed-0 trajectory and not a claim**.

  What replicates is the same object in its state-like form, and it replicates tightly: **the
  maximum |V_w − base| over the non-init eras is 0.032 at seed 0 and 0.033 at seed 2 — both
  calibrated to within 0.04 of the world they are being fed, at every era.** The judge sits on
  its world; which side of it, and in what order, is a trajectory.

**The lag decomposition — REPLICATES, and this was the rate-like claim most at risk.** Seed 2,
the whole rmse curve: against the cycle's own solve rate `.1048 → .0420 (τ = 50) → .0484`;
against the ring buffer's trailing rate `.0419 → .0405 (τ = 5) → .0875`. Seed 0 read τ = 50 and
τ = 2. **At both seeds the judge's level is a ~50-cycle average of the world and a 2–5-cycle
average of its own buffer**, so the decomposition — the lag is the diet's and the reader adds a
handful of cycles — holds at both. `corr(level, trailing base)` 0.896 and 0.848.

**Headline 2 — REPLICATES in sign once the init cell is excluded.** Panel drift (last minus first
non-init reading era): seed 0 `+0.086` (era-1 panels) and `+0.085` (era-2 panels); seed 2
`+0.125` and `+0.059`. Over every era seed 2 reads `mixed`, and the whole of that is its single
flagged cell (0.5001 in era 1). Over the non-init eras both seeds are `+`. At seed 2 every panel
set rises: era-3 panels +0.028, era-4 +0.001, era-5 read once. **The magnitudes move by 2× between
the seeds and are not a claim; the sign is.**

**Headline 3 — REPLICATES, and it is the strongest of the three.** (structure | cost) − (cost |
structure) on filed rows, eras 3–5: seed 0 `+0.083, +0.153, +0.143`; seed 2 `+0.092, +0.169,
+0.181`. Same sign, and the magnitudes agree to about a hundredth. The underlying columns agree
too — `crit ~ struct | y` runs 0.573 / 0.633 / 0.676 / 0.702 across eras 2–5 at seed 2 against
0.534 / 0.608 / 0.702 / 0.663 at seed 0, while `crit ~ y | struct` sits at 0.540 / 0.541 / 0.507 /
0.521 against 0.558 / 0.525 / 0.549 / 0.521. Over all five eras the patterns differ (`+-+++` vs
`-++++`), entirely in era 1–2 where the critic is warming up. On **probe** rows the agreement is
weaker and is *not* claimed: eras 3–5 read `+++` at seed 0 and `-++` at seed 2.

> **Net: the two state-like claims read on identical rows — the norm's calibration, and the judge
> reading structure rather than cost at matched everything — hold at both seeds. The one rate-like
> claim about a direction (undershoot-then-converge) does not, and is withdrawn. The other
> rate-like claim, the lag decomposition, does. A second seed was the only way to tell these
> apart, and it separated them.**

## §9 The diets — norm §1's between-subjects design on a fixed trunk (`refit.py`)

`reduce_rows.py` [R] compared two banked critics and had to carry a caveat it could not remove:
each carries its **own trunk**, so it is two whole readers and not two readouts of one state, and
the two arms differ in seed *and* in probe draw at once. This section removes that caveat the way
norm did — one trunk, held fixed; one set of held-out rows, identical for every diet; and the
critic re-fitted from scratch under diets that differ in what outcome distribution they were fed.

**The protocol.** The trunk is `ov_s0b`'s banked core, constant across every diet. The critic is
rebuilt by `build_critic` from the **same init seed** for every diet, trained with the same
optimizer, the same learning rate, the same batch (64 rows per slot per step) and the same number
of steps. Every refit is read on `code < 100·vo_critic_hold` — the run's own bijective hold code,
42,722 rows — which **no diet trains on** (gate T-8, leak = 0). The random-init trunk is carried
beside the trained one, because that is the control that made norm §1 a finding rather than an
observation: on a random feature map the diets could only *shift* the norm, while on the trained
trunk they also *rescaled* it.

**The feature cache and gate T-6.** `Critic.ctx_state` reads `pooled.mean(1)` and the slot's own
`span` block vectors and nothing else, so those two are cached once per *distinct* context and
the trunk never runs again. Against the real forward through the banked critic that is an
identity (max |Δ| = 0.000e+00 over six buffers × 1024 rows), and it breaks as required when one
cached block vector is perturbed by 1e-3.

### §9a WITHDRAWN: the context-tercile construction, and its correction

norm's `out_lo/mid/hi` are within-cell terciles of the *window's mean outcome*, a continuous
quantity, and all three of its diets kept a workable base rate (0.596 / 0.686 / 0.770). The
direct transcription here — group the training rows by context, sort the contexts by their mean
verdict, cut three groups of equal row count — produced

| diet | E[outcome] | E[y] filed | E[y] probe |
|---|---|---|---|
| `out_lo` | **0.0000** | 0.0000 | 0.0001 |
| `out_mid` | 0.0509 | 0.0911 | 0.0032 |
| `out_hi` | 0.5888 | 0.9227 | 0.1928 |

`out_lo` is not a low-outcome world; it is a world in which nothing ever works, and a critic
cannot be identified on it at all. The cause is a property of this substrate rather than a bug:
a context's rows are beam siblings that nearly always share one verdict — the twin reduction
measures their disagreement rate at 0.09–0.18 — so a sort on the context mean is a near-binary
sort and the bottom third is simply every all-zero context. **This is the difference between
norm's event stream, where a window carries a continuous mean outcome, and a practice beam, where
a context carries one verdict.**

**The correction, which is what `refit.py` runs**: three target base rates on a multiplicative
ladder, `0.6× / 1.0× / 1.4×` around each (slot, source) cell's *own* natural rate, realised by
drawing that cell's positives and negatives in the required proportion. Slot counts, level counts
and the filed:probe ratio stay pinned exactly (gate T-7), every diet stays trainable, and the only
thing that moves is E[outcome]:

| diet | rows | filed | probe | E[outcome] |
|---|---|---|---|---|
| `out_lo` | 272,698 | 147,954 | 124,744 | **0.1275** |
| `out_mid` | 272,698 | 147,954 | 124,744 | **0.2125** |
| `out_hi` | 272,698 | 147,954 | 124,744 | **0.2976** |

beside the **source** family, which is the axis `voicing` Q3 built and needs no construction at
all — a filed write solves 0.329 of the time and a substituted class 0.065, so `filed` / `both` /
`probe` at equal total rows per slot is a three-world ladder the substrate hands over for free.

Selecting on the critic's own teacher **is** the manipulation, as norm says of its own: "as
Xiang's experimenters chose which offers each group saw."

### §9b The result: norm §1's rank order reproduces exactly, and its control does not

Full table in [`results/refit_ov_s0b.txt`](results/refit_ov_s0b.txt). 600 steps per refit, batch
64 per slot per step, init seed 7 for every diet, read on the same 42,711 held-out rows.

**The norm calibrates to the diet, in level, on identical rows.** `norm − E[outcome]`, trained
trunk: `full` −0.007, `filed` −0.014, `probe` +0.028, `both` −0.001, `out_lo` −0.000,
`out_mid` +0.009, `out_hi` −0.016. Across a five-fold range of E[outcome] (0.065 → 0.329) the
judge's mean predicted P(solve) on rows it has never seen **is** the world it was fed, to ±0.03.

**Spearman ρ against the diet's own E[outcome] is +1.000 in every family on both trunks** — the
rank-one result norm reported at every level, checkpoint and anchor, reproduced here with the
trunk genuinely held fixed. This is what `reduce_rows.py` [R] could not say.

**Every diet carried signal**: held-out AUC 0.595–0.652 (trained trunk) against per-diet
shuffled-outcome floors of 0.501–0.532.

**Shift versus rescaling — and the direction is opposite to norm's.** Each diet's per-row norm
regressed on `full`'s across the identical rows:

| trained trunk | slope | intercept | R² | sd ratio | | random trunk | slope | R² | sd ratio |
|---|---|---|---|---|---|---|---|---|---|
| `out_lo` (E 0.128) | 0.552 | +0.013 | 0.778 | 0.626 | | `out_lo` | 0.557 | 0.938 | 0.575 |
| `out_mid` (E 0.213) | 0.993 | +0.017 | 0.869 | 1.065 | | `out_mid` | 0.871 | 0.947 | 0.895 |
| `out_hi` (E 0.298) | 1.188 | +0.037 | 0.759 | 1.363 | | `out_hi` | 1.153 | 0.840 | 1.258 |

The diets rescale, not merely shift — slopes 0.55 → 1.19 and sd ratios 0.63 → 1.36 — but **a
low-outcome world here COMPRESSES the judge's dynamic range and a high-outcome world WIDENS it**,
which is the reverse of norm §1 ("a low-outcome world gives the critic a wider dynamic range on
the same states, a high-outcome world a compressed one", its slopes running 1.16 for `out_lo` to
0.82 for `out_hi`). There is a mechanism available and it is a difference in the readout, not in
the finding: norm's critic is a **ridge**, linear and unbounded, while this judge emits a
**logit through a sigmoid**, so a low base rate parks its predictions where the sigmoid is flat
and the spread in probability shrinks by construction. Which of the two is "the" calibration is
not something this node can decide.

**And norm's own control does NOT reproduce.** norm found that a random-init trunk's norm can
only *shift* while the trained trunk's also *rescales*. Here **both rescale, by nearly the same
amount** (`out_lo/mid/hi` slopes 0.557 / 0.871 / 1.153 on the random trunk against 0.552 / 0.993
/ 1.188 on the trained one). What the trained representation buys is elsewhere: **discrimination**
(AUC 0.618–0.652 against the random trunk's 0.572–0.581) and a *less* affine diet-to-diet map
(R² 0.76–0.87 against 0.84–0.95). Read together with the paragraph above: with a sigmoid readout
the scale follows the base rate on any feature map, so the rescaling is the link function and not
the representation; what the representation supplies is where the rows sit, not how wide the
scale opens.

**The source family is the sharper manipulation and it is free.** `filed` (E 0.329) against
`probe` (E 0.065) at equal rows per slot: slopes 0.396 and 0.400 on `full`, but with intercepts
+0.233 and +0.011 and R² of **0.143** and 0.499 — two readers that disagree strongly about which
rows are good, not two rescalings of one reader. And the per-kind columns say why in one line:
the `filed`-only refit reads **0.3127 on held-out filed rows and 0.3173 on held-out probe rows**
— it cannot tell a counterfactual from a real write at all — while `full` reads 0.2970 and 0.1006,
separating them by 0.20. That is `voicing` Q3's "a critic trained only on what the beam chose to
write ranks counterfactuals at chance" restated on the **level** instead of the ranking, which is
a form the arc has not had before.

**Non-vacuity**: the seven diets span 0.221 (trained) and 0.234 (random) in norm on the identical
rows, so the diet axis is live — the dual of `voicing`'s gate V-6, one round over.

## §10 What is deliberately not here

- **No README.** Interpretation waits on a discussion.
- **No third seed.** Two paid arms, seeds 0 and 2 (1.50 + 1.89 GPU-h), the arm of
  `overtone/results/RUN_ov_s0b.sh`. §8c says which of §8's headlines survived the second seed and
  which is withdrawn. Seeds 1 and 3 have anchors banked but neither reaches L4, so a third seed
  would add eras 1–3 and nothing at the frontier.
- **No claim from the era-to-era movement of the world**, in either direction, until the learner's
  own improvement is separable from the damage's deepening. The fixed panel is the instrument that
  separates them for the *reader*; nothing separates them for the *world*.
- **No anticipatory response.** norm's `R = V(s_t) − V(s_{t−1})` has no analogue here: the write
  and the outcome are one act, so this node has a norm and an outcome contrast and no third term.
