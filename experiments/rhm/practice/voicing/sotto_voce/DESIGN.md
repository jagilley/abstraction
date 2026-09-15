# sotto voce — DESIGN

Decisions, and every withdrawn diagnosis kept beside its correction. **Up**:
[`../DESIGN.md`](../DESIGN.md) (`voicing`, §0–§38). **Spec**: [`SPEC.md`](SPEC.md) (the
orchestrator's brief verbatim). **Machinery record**: [`FILES.md`](FILES.md).

No README: per the repo's norms the interpretation waits on a discussion, so the facts live in
the reduction (`figures/`) and here.

---

## §1. The question, and why it is askable here and almost nowhere else

`voicing` Q3 produced one clean negative and one clean positive. The negative: **a critic fed
only what the learner chose to write ranks counterfactual rows at chance** — held-out AUC 0.491
on probe rows against 0.65 on filed rows, 612–895 rows per slot over 16 slots. The learner writes
one class per slot (Q0 finding 3: at L5, one token class on 100% of calls out of six the book
holds), so its experience carries no information about the untaken class. The positive:
**paying the world for a substituted verdict fixes it** — 0.491 → 0.798 like-for-like, at
0.44–0.60% of priced time across seeds.

There are exactly three ways to get a label for a road not taken:

1. **vary the behaviour** — and `voicing` Q1 measured the price: a 9% deviation in the L2 writes
   cost the L3 commit window, because on a depth ladder every adopted level feeds the one being
   earned;
2. **pay the world** — the probe channel, cheap on RHM and a real trial anywhere else;
3. **use a model of outcomes trained on experience and applied to sentences the learner never
   wrote** — the forward model's seat in the practice arc, and this node's subject.

RHM is where (3) is measurable, and the reason is worth stating because it is the node's whole
methodological point: **the world's verdict is exact and cheap, so it can be computed for every
model-graded probe in the background as an instrument that no arm consumes.** The mirror can be
caught being wrong, row by row, at a cost of nothing. Elsewhere you would have to choose between
using the world's verdict and measuring against it.

Three pieces of the record set the expectation and are not re-derived here.
`experiments/mjc/committee_head/README.md`: disagreement is the best reward-free reader of "is
this learnable" (0.716), beating both the incumbent counterfactual survey and a *privileged*
reader given a matched-FM ceiling. `experiments/mjc/ballistic/directed/README.md`: disagreement
finds where you **lack data**, not where data went stale — and a counterfactual is by
construction where you lack data. And the hazard, `experiments/mjc/practice/span/README.md` F3: a
single model's belief about its own error off the practised corridor was **uncorrelated** with
its true error (corr +0.009, p=0.96, 40 cells), and more search bought a better-scoring plan
under the model while the truth got slightly worse. `ideas/heterogeneous_graders.md` §8 names the
shape: an outcome model trained on the learner's own experience is **the same grader in a
mirror** — same corpus, same failure geometry, blind in the same places — so the question is not
"does it work" but *how far the mirror extends before it goes blind, and whether anything visible
from inside marks the edge.*

---

## §2. What is held fixed, and what moves

Everything about the substrate, the chooser and the probe channel is `voicing`'s at its `vo_s3b`
head. All three arms are the **composed chooser** (`vo_govern_mode="composed"`, `vo_w=1.0`) with
the probe channel ON (`vo_probe=True`, `vo_probe_n=64`) and clock-yoked to the banked anchor, so
every level-resolved readout is read at the same slots at the same cycles (gate Y-1). The floor
and the ceiling are **banked and not re-run**: `vo_s3b:voi3b_comp_yk` / `voi3b_comp_pr_yk` at
seed 0 and their seed-2 twins in `vo_s3e`. The one knob that moves is `vo_om_mode`.

| arm | `vo_om_mode` | who answers a probe | what is billed |
|---|---|---|---|
| (banked floor) | — | nothing; no probe channel | 0 |
| (banked ceiling) | `"world"` | the world's grader | every probe |
| `so_mg_yk` | `"model"` | one outcome model | **0** |
| `so_cg_yk` | `"committee"` | K=5 models, and the row is filed **only where they agree** | **0** |
| `so_hy_yk` | `"hybrid"` | as committee; **where they disagree the world is paid** | the disagreeing fraction |

On all three the world's verdict is computed for **every** probe and kept as an instrument
(gate M-1).

---

## §3. The outcome model — five decisions, each with its reason

**(a) It is a separate object, not a readout of the surface model.** The composed chooser
already uses the surface model as its prior (`z(dp/span) + w·z(critic)`), so a critic trained on
the prior's own opinion would be learning the prior. The outcome model therefore never sees the
generator's logits, the DP's score, or the critic. It reads a **rendered configuration and its
root** — exactly what the world's grader reads — and is trained by BCE against the world's
verdict on configurations the learner actually produced. It has its own parameters, its own
optimizer (deliberately **not** `gopt`: a treatment that buys the plant no extra steps must also
not borrow the plant's optimizer), its own numpy streams per member, and every forward and
backward is wrapped in the file's `_rng_snapshot` / `_rng_restore` sandbox, which is the critic's
own idiom and is why G-F and V-4 stay closed.

**(b) Its shape is a tree, and nothing more.** The world's verdict is a bottom-up parse
(`possible_sets`): `s` children combine into a parent under per-level rules, `depth` times, and
success is "the root's possible set contains r*". The model is given the tree — merge `s`
adjacent states with one weight set per level, `log_s(length) = 6` levels, then combine the top
with a root embedding — and **no rule table, no possible-set semantics and no privileged
features**. It has to learn the rules from the sentences it has seen. The architecture is the
animal's cortex, not the answer. A multiplicative interaction in the merge (`cat([l, r, l*r])`)
was tried offline and was *slower and no better* (held-out AUC 0.988 against 0.994 at 1500
steps), so the plain form is what ships.

**(c) The training corpus is the filed rows' `fin` and `y`, deduplicated by (configuration,
root).** `fin` is the trajectory's final configuration and `y` its tip's verdict, so the pair is
exactly (a sentence the learner produced, what the world said about it) — asserted, not assumed:
gate **M-2b** re-grades a sample of the training rows every strict cycle and requires 0
mismatches. The dedup is because one tip contributes one row however many macro slots it wrote,
so the buffer is a sample of the learner's **sentences** and not of its write calls.
*Deliberately not used*, though available for free: the metering beam's graded tips, the
practice beam's initial configurations, and the value buffer's intermediate states. The last is
the important exclusion — its label is "did this state lead to success under the policy", not
"is this state solved", and mixing the two would make the mirror a value head.

**(d) The filed target is the mirror's PROBABILITY, not a verdict thresholded at 0.5.** This is
the one place the implementation departs from the most literal reading of the brief, and the
reason is a number. The critic's loss is `binary_cross_entropy_with_logits`, which takes a soft
target. The world measures the probe base rate at 0.02–0.13 (a substituted class solves 3.5–4×
less often than the written one), and at that base rate a 0.5 threshold discards almost
everything the mirror knows: in the offline study a constant "not solved" scores **higher
accuracy** (0.913) than the thresholded mirror (0.853) while the mirror's *ranking* is far above
chance (AUC 0.747). Filing `p` hands the critic the mirror's belief and its uncertainty together;
filing `1[p>0.5]` adds threshold noise on top of model bias and is dominated. The thresholded
verdict is kept as the **readout** — it is what [P]/[R] report against the world — so nothing is
lost from the measurement. **The hard-verdict variant is untested and is a live alternative.**
On the hybrid this means agreeing rows carry a probability and disagreeing rows carry the world's
hard 0/1; that asymmetry is real and is stated rather than hidden, since the world's label
genuinely is certain and the mirror's genuinely is not.

**(e) The committee differs by init, by bootstrap and by draw.** K=5 as in `committee_head`.
Each member is minted from its own generator seed (`build_critic`'s discipline, one organ
across), trains on its own Bernoulli(`vo_om_boot`=0.8) subset of the non-held-out rows — a draw
made once per row at insertion and stored, so membership is stable as the buffer rolls — and
draws its batches from its own numpy stream. Gate **M-5**'s second perturbation is a committee
that is K copies of one member, and it must read spread identically zero.

---

## §4. The agreement threshold, sized offline

The rule: a probe is **agreed** iff the committee's spread on it is at most the `q`-th quantile
of the committee's spread on **held-out filed rows** — "the committee is at least as concordant
about this counterfactual as it is about the most concordant `q` of the experience it has
actually seen". The threshold is recomputed every cycle from the current held-out split, so it
tracks the mirror rather than being frozen at a number measured before the run.

`q` is the setting, and it was sized offline. The brief asks for the sizing to be done on banked
filed rows from `vo_s3b`; **those rows do not exist** — `voicing` logs `vo_buf` / `vo_probe_buf`
as sizes and never persisted `fin`, which is precisely the field this node adds. So
[`q0_sotto.py`](q0_sotto.py) builds a **proxy** corpus from the same world with the same
corruption operator and the same substitution, says so at the top of its own output, and reports:

| q | threshold | filed fraction | accuracy (filed) | accuracy (dropped) | AUC (filed) |
|---|---|---|---|---|---|
| 0.25 | 0.0121 | 0.200 | **0.998** | 0.817 | 0.999 |
| 0.50 | 0.0454 | 0.473 | 0.924 | 0.790 | 0.875 |
| 0.60 | 0.0568 | 0.600 | 0.907 | 0.773 | 0.833 |
| 0.75 | 0.0849 | 0.784 | 0.883 | 0.746 | 0.790 |
| 0.90 | 0.1291 | 0.923 | 0.861 | 0.758 | 0.761 |
| 1.00 | 0.4361 | 1.000 | 0.853 | — | 0.747 |

**q = 0.50 is the setting of record.** q = 0.75 drops only ~22% of the rows, which would make the
committee arm barely distinguishable from the model-graded arm; q = 0.25 files a fifth of the
rows, which starves the critic of the volume the 0.798 ceiling was bought with. q = 0.50 is where
the contrast is readable on both axes at once. The offline table also says what the hybrid will
cost: roughly half the ceiling's bill, i.e. ~0.25% of priced time.

**The proxy's gap, stated so it is not discovered at reduction time.** Its writes are drawn
uniformly over the *true* table; the learner's are drawn from *its book* and are near-degenerate
at the frontier. So the proxy's support is **wider** than the learner's, which makes it optimistic
on the counterfactual half. Expect the run's filed fractions to be *lower* than 0.473 and its
counterfactual accuracies *worse* than 0.853.

---

## §5. What the offline study already establishes, before any GPU

From `figures/so_q0_reduction.txt` (proxy corpus, 30,000 experience rows, 30,000
counterfactuals, K=5, 1500 steps/member):

- the proxy's counterfactual/experience solve-rate ratio is **3.40×**, against the 3.5–4×
  `voicing` measured in run — so the corpus is in the right regime;
- one outcome model: held-out **experience** AUC **0.913** (acc 0.848, base 0.300);
  **counterfactual** AUC **0.738** (acc 0.841, base 0.087);
- the committee: experience AUC **0.935**, counterfactual AUC **0.747**;
- **disagreement as a detector of the mirror's own error on counterfactuals: AUC 0.690** — and
  0.759 on held-out experience. `committee_head` read 0.716 for disagreement as a reducibility
  reader in a motor domain; this is the same statistic, one domain over, within 0.03;
- the spread ratio counterfactual/experience is **0.97** — i.e. in the proxy the committee is
  *not* measurably more spread off-support, even though its spread does track its error. That is
  a mild surprise and is flagged here in advance of the run's own number;
- by level, the counterfactual AUC falls monotonically with depth: L2 0.826, L3 0.800, L4 0.800,
  L5 0.750 — the mirror is weakest exactly at the rungs the round is about.

**None of this is a measurement of the run.** It is the de-risking the brief asked for: the
architecture learns the grader on-support, degrades off-support without collapsing, and the
agreement rule has a real gradient to trade against. The run's numbers come from `log["vo_om"]`
and `vo_mirror`.

---

## §6. Gates, and the rule that a gate must be shown to fail

The arc's binding rule is inherited: **a gate is not reported until it has been shown to FAIL on
a deliberate perturbation of the thing it claims to protect**, and Q3's addition — an inertness
gate needs a liveness twin whose perturbation is *disconnection* — is inherited with it. The full
table is in [`FILES.md`](FILES.md); what follows is why each new gate has the shape it has.

**M-1 is two gates, because it makes two claims.** "The world's verdict exists for every probe
and is not what gets filed" is a property of the dispatch and is checked on a construction: a
stub mirror that says *not solved* on every row against a stub world that says *solved* on every
row, so the two disagree on **every** probe and "which one was filed" is a count, not a rate.
"The world's verdict enters no loss" is a different claim and cannot be checked by reading which
buffer the loss names — so **M-1b measures it**: one critic, one slot, a `pbuf` of model verdicts
and a `wbuf` whose verdicts contradict them row for row, one optimizer step each way, and the
two parameter sets must be equal to the bit. Its perturbation is to point the critic's loss at
`wbuf`, and it reads 2.0e-02 against the clean 0.000e+00.

**M-2 needs a back door as well as a door.** `push_world` asserts the mode, so a model-graded
probe verdict cannot enter through the function that exists to admit world verdicts — and the
falsification opens that door and watches the model arm admit four rows. But a row appended
*around* the door would leave the counter at zero, so the gate also asserts that the number of
rows the buffer **carries** as world-graded equals the number `push_world` **admitted**; the
second perturbation smuggles a row in through `_append` and the identity breaks. **M-2b** is the
other half and is about meaning rather than plumbing: an experience row must be (a configuration
the learner produced, *the world's own verdict on that configuration*), which is re-graded both
in the CPU gate and in the run itself on its strict cycles.

**M-3 is an identity on a count, not a tolerance.** The bill is `counts["ground"] +=` the
world-graded probe count and nothing else, and the run-level form **M-3r** re-asserts
`probe_ground == 0` on the two unbilled arms and `== n_probe − n_agree` on the hybrid, with
`probe_priced == probe_ground · d_fb` beside it.

**M-4's perturbation is the spec's own**: flip the threshold. Wide open, the committee arm files
everything and is the model-graded arm wearing a different name; shut, it files nothing and the
hybrid sends every probe to the world. Both must fail, and both do.

**M-5 is the liveness gate**, and its first perturbation is defect #8's shape one organ over: a
mirror that is built, fed, consulted and never trained. Its second is the committee collapsed to
K copies of one member. It runs on a synthetic, learnable target and is a statement about the
machinery, never about RHM.

**Every existing gate still holds and is re-run**, G-F included and mandatory — the probe path,
the recorder and the critic audit are all shared paths this node touched.

Counts: `vo_gates_cpu` **24/24** (18 inherited + 6 new), `gates/falsify.py` **56/56** (36
inherited + 20 new).

---

## §7. Withdrawn diagnoses, kept beside their corrections

1. **"Fit the outcome models offline on banked filed rows from `vo_s3b`."** Withdrawn on
   inspection: the banked arm files carry `vo_buf` and `vo_probe_buf` as *sizes* and never
   persisted `fin`. **Correction**: `q0_sotto.py` fits on a proxy corpus built from the same
   world with the same operators, and its output says so in its first paragraph; the sizing
   decision (§4) is made on that, with its optimism named.

2. **"File the mirror's verdict."** Withdrawn after the offline base rates: at a probe base rate
   near 0.09 a constant "not solved" beats the thresholded mirror on accuracy while the mirror
   ranks far above chance, so a 0.5 threshold would have made the arm fail for a reason that is
   not the question. **Correction**: file the probability, keep the thresholded verdict as the
   readout (§3d). The hard-verdict variant is untested.

3. **"Use a multiplicative merge so the net can represent the possible-set recursion."** The
   bilinear form is what the world actually computes, so the expectation was that `cat([l, r,
   l*r])` would learn faster. Measured offline on 20k rows: plain 0.994 AUC in 26 s, multiplicative
   0.988 in 61 s. **Correction**: the plain form ships; the inductive bias was not the binding
   constraint at this data volume.

4. **"M-5 can assert the mirror learns a parity."** Parity of two tokens plus the root was the
   first synthetic target and the net read AUC 0.497 on it at every setting tried. Parity is hard
   for this architecture and for most; asserting it would have made a liveness gate fail for a
   reason unrelated to liveness. **Correction**: the target is a one-merge match
   (`(x0 + x1) mod 4 == r`), which the net learns, and the gate says out loud that it is a
   statement about the machinery.

5. **"The run-level bill identity can read the last `vo_bill` line."** Withdrawn on the
   preflight: `probe_ground_c` is reset at the top of every cycle, so `vo_bill[-1]` is the LAST
   CYCLE's probe count and not the run's. The gate passed vacuously on the two unbilled arms
   (0 == 0) and tripped on the hybrid — the one arm where the two differ — after all nine twins
   had run. **Correction**: M-3r sums the ledger lines, and the same trip produced a second,
   more useful correction: **the whole file-based gate block was factored out of `preflight`
   into `vo_preflight_gates(outdir)`**, with a CPU-only entrypoint `preflight_gates` beside it,
   so a gate that trips can be re-evaluated against the saved artifacts in seconds instead of
   costing a re-run of the twins to recover the results of the gates printed after it. A gate
   block whose first failure throws away every other gate's result is a gate block that
   discourages being gated.

6. **"On a mirror arm the inherited `assert _pg == _pn` still holds."** It does not: "every
   probe drawn is a probe billed" is the WORLD-graded channel's identity, and a model-graded arm
   must violate it. Caught by the gate itself (`so_pf_mg: 202 probes graded but 0 billed`).
   **Correction**: the assertion is mode-aware — `voicing`'s verbatim on a `"world"` arm, M-3's
   on a mirror arm — with `mode`, `n_world`, `n_filed`, `n_agree`, `agree_share` and
   `filed_share` added to the per-arm gate record.

7. **"No back door = the buffer carries exactly the world-graded rows `push_world` admitted."**
   Holds only until the cap binds. `vo_om_cap = 30000` and the hybrid pushed 118,757 rows
   through it, so at the end the buffer CARRIED 19,240 world-graded rows against 73,823
   admitted — the window's composition, not a smuggled row (30,000 × 73,823/118,757 = 18,649,
   within 3% of what was observed). The gate passed at preflight only because the preflight's
   cap never bound. **Correction**: a cumulative `n_append` counter at `_append`, the single
   choke point every row passes through, so the claim is cap-independent; the window's
   composition is reported and bounded rather than asserted as an identity. `so_s1` predates
   the counter and is checked in the bounded form.

8. **"A front-to-back cap on the per-probe instrument sample is a sample of the run."** It is
   not: at `vo_probe_n = 64` over ~30 governed slots the 4,000-row cap fills in five cycles, so
   `so_s1`'s row-level sample is entirely from **cycles 50–54** — a five-cycle-old mirror whose
   held-out AUC was still ~0.52–0.69, reported as if it were the run's. This is why reduction
   [Q]'s between-probe detector AUC reads ≈0.51 there and should not be read as "disagreement
   does not find the mirror's error". **Correction**: reservoir sampling on the recorder's own
   readout stream (separate from `prng`, so the sample cannot move the probe's draws), plus a
   **between-cycle panel** in [Q] computed from the per-cycle mirror counters, which cover
   every probe of every cycle and need no sample at all. `so_s1` and `so_s2` predate the
   reservoir; both print their sample's window.

9. **The dedup in `push_experience` first used a structured-dtype view** (`[("f", int64)] * n`),
   which numpy rejects for repeated field names. **Correction**: a `np.void` row view, which is
   the standard idiom and is what ships.

---

## §8. Cost, and where the time goes

The mirror's own compute is negligible against the plant: 16 steps × 5 members × batch 256
through an ~85k-parameter net per cycle, plus one prediction pass over the cycle's probes and
one over ≤2048 held-out rows. What is **not** negligible and is paid on every mirror arm is the
world's instrument grade on every probe — the same `grade` call the world-graded ceiling pays,
except unbilled. That is the price of being able to catch the mirror being wrong, and it is a
wall-clock cost only, not a cost on the meter.

`vo_om_cap = 30000` holds roughly the last 30 cycles of experience at this substrate's ~1,024
tips per cycle. The buffer is a rolling window by construction, which means the mirror is a model
of **recent** experience; the per-cycle trajectory in reduction [P] is what says whether that
matters.

---

## §9. What `so_s1` returned (seed 0) — facts only

Three arms, 201 cycles each, **Y-1 exact on all three with 0 cancelled** (commits
[48, 100, 151, 186], advances [60, 110, 180, 192, 201]), so L4 and L5 cells exist and every
level-resolved number below is read at the anchor's slots at the anchor's cycles. Gate table in
[`FILES.md`](FILES.md); full reduction `figures/so_s1_reduction.txt`. Peak RSS 7,045 MiB over
the tag, which is what sized `memory=12288` for `so_s2`.

**(a) The chooser, at the replicated cell (L2/L3 pooled repair accuracy, n ≈ 195k per arm).**

| arm | rep | vs the anchor | vs the floor |
|---|---|---|---|
| **ceiling** `vo_s3b:voi3b_comp_pr_yk` (world-graded probes, banked) | 0.2736 | +28.5% | +9.6% |
| **`so_mg_yk`** (model-graded) | **0.2602** | +22.2% | **+4.2%** |
| **floor** `vo_s3b:voi3b_comp_yk` (filed diet, banked) | 0.2496 | +17.2% | — |
| `so_cg_yk` (committee-graded) | 0.2466 | +15.8% | −1.2% |
| `so_hy_yk` (hybrid) | 0.2420 | +13.6% | −3.1% |
| anchor `vo_s3:voi3_dp` (banked) | 0.2130 | — | −14.7% |

The single model-graded mirror recovers **44%** of the floor→ceiling gap. The two committee
arms do not: both sit slightly **below** the filed floor.

**(b) The critic's held-out AUC on the probe diet, scored against the WORLD** — the number the
brief asks for, and against `voicing`'s 0.491 for a critic fed only filed writes:

| arm | vs the FILED verdict | **vs the WORLD** | mirror's labels correct on filed rows | slots |
|---|---|---|---|---|
| ceiling (world-graded) | 0.773 | 0.773 (same column) | — | 29 |
| `so_hy_yk` | 0.712 | **0.712** | 0.982 | 30 |
| `so_mg_yk` | 0.667 | **0.617** | 0.932 | 30 |
| `so_cg_yk` | **0.772** | **0.592** | 0.973 | 15 |

`so_cg_yk` is the whole node in one row: the critic learns to rank **the mirror's opinion**
0.772 — as well as the ceiling ranks the world — while ranking the world at 0.592. The student
faithfully learned the teacher; the teacher is what is wrong. This is
`ideas/heterogeneous_graders.md` §8's criterion measured rather than argued.

**(c) The mirror itself** (held-out experience vs probes, against the world):

| arm | K | held-out EXPERIENCE acc / AUC / base | on PROBES acc / base | n probes |
|---|---|---|---|---|
| `so_mg_yk` | 1 | 0.978 / **0.996** / 0.307 | 0.926 / 0.066 | 204,096 |
| `so_cg_yk` | 5 | 0.977 / **0.996** / 0.273 | 0.926 / 0.068 | 206,806 |
| `so_hy_yk` | 5 | 0.949 / 0.987 / 0.302 | 0.947 / 0.062 | 206,861 |

The mirror learns the world's grader **almost perfectly on its own support** (AUC 0.996, far
above Q0's proxy estimate of 0.935) and its accuracy off support looks high only because the
base rate is 0.06: on `so_mg_yk`'s 204k probes it calls 15,071 positives of which **8,410 are
false**, and misses 6,771 of 15,432 true ones. Precision ≈ 0.44, recall ≈ 0.43.

**(d) The blind region is the frontier, and it is monotone in level.** Mirror accuracy on probes
(`so_mg_yk`, in-support rows `s1` beside out-of-support `s0`):

| level | n | world solve rate | mirror acc | committee agree share (`so_cg_yk`) |
|---|---|---|---|---|
| L2 | 141,034 | 0.052 | 0.931 | 0.492 |
| L3 | 49,631 | 0.076 | 0.925 | 0.525 |
| L4 | 12,242 | 0.148 | 0.889 | 0.327 |
| L5 | 1,189 | 0.184 | 0.786 | **0.174** |

Two things ride together here. The mirror is worst exactly where the round is about, and
**committee agreement is anti-correlated with level**: the agreement rule drops 83% of L5 probes
and only 51% of L2 probes, so filtering on agreement *systematically starves the frontier*. That
is a candidate mechanism for (a)'s committee result and it is a posteriori, not predicted.

**(e) The hybrid's bill, and what it bought.** 73,823 world-graded probes of 206,861 drawn
(**35.7%** world-query rate, against Q0's proxy prediction of ~53%), for **0.199%** of priced
time mean / 0.390% max, against the ceiling's 0.557% / 0.831%. So the hybrid bought **92% of the
ceiling's counterfactual AUC (0.712 of 0.773) for 36% of its bill** — and nonetheless chose
*worse* than the filed floor. A better critic did not become a better chooser on this arm.

**(f) A mechanism for the hybrid, visible in the counters.** Its outcome models train on
experience *plus* world-graded probes, and by the end the buffer is **62% probe rows**
(73,823 world pushes against 44,934 experience pushes, in a 30,000-row window). Probe rows carry
a base rate of 0.062 against experience's 0.302, so the hybrid's mirror is pulled toward "not
solved": it predicts positives at **0.025** where the world's rate is 0.062, against 0.074 for
the model-graded arm. Training the outcome model on what the bill bought changed its base rate.

**(g) Disagreement as a detector.** Between probes, on `so_s1`'s row-level sample, AUC ≈ 0.51 —
but that sample is cycles 50–54 only (§7.8), so it is a reading of a five-cycle-old mirror and
is not the run's. Between cycles, over the whole run and every probe: **ρ(spread, mirror error)
= +0.817** and AUC 0.928 for `so_cg_yk`, +0.496 / 0.697 for `so_hy_yk`, undefined for
`so_mg_yk` (K = 1, spread identically zero). So the committee's spread tracks the mirror's error
**across the mirror's maturation** strongly, while whether it identifies *which individual
counterfactual* is wrong is not yet measured at a denominator worth reporting.

**(h) Era gaps at matched clocks**, treated minus anchor: era-3 −0.046 / −0.024 / −0.018 and
era-5 −0.270 / −0.277 / −0.126 for mg / cg / hy, against the banked floor's −0.048 / −0.223 and
the ceiling's −0.081 / −0.070. `voicing` records that matched-clock error gaps between
near-identical arms reach 0.08 in this lineage, so the era-5 column is a single-trajectory
number and the era-3 column is the one that reproduced across seeds there.

---

## §10. What `so_s2` returned (seed 2), and the two seeds side by side

Three arms, 201 cycles each, **Y-1 exact on all three with 0 cancelled** — commits
[59, 77, 157], advances [60, 110, 180, 192, 201]. **Seed 2's anchor commits L2, L3 and L4 and
never L5**, exactly as `voicing` Q3d recorded, so this tag has **L4 cells only** and its frontier
panel is not the same table as seed 0's. Peak RSS 6,959 MiB at `memory=12288`. `so_s2` is the
first tag carrying the cumulative `n_append` counters, and M-2r's **strict, cap-independent**
identity holds on it: `n_append_world == n_push_world == 86,530` and
`n_append_exp == n_push_exp == 41,543`. Full table: `figures/so_seedtable.txt`; reduction
`figures/so_s2_reduction.txt`. **The seeds are not averaged anywhere.**

### What replicates

1. **The ceiling is above every mirror arm, on both seeds** (+9.6% / +10.2% over the filed
   floor at L2/L3 pooled). Paying the world is still the best diet.
2. **The model-graded arm beats the filed floor on both seeds** — +4.2% and +5.5%, i.e. **44%
   and 54% of the floor→ceiling gap** recovered by a model trained on experience alone, at
   **zero** bill.
3. **Committee-graded < model-graded on both seeds** (−1.2 vs +4.2; +2.8 vs +5.5). Refusing to
   file where the committee disagrees is worse than filing everything the model says.
4. **The critic's probe AUC against the world is far above chance on every mirror arm and
   ordered ceiling > hybrid > {model, committee} on both seeds.** Against `voicing`'s 0.491 for
   a filed-only critic: ceiling 0.773/0.750, hybrid 0.712/0.701, model 0.617/0.645, committee
   0.592/0.648.
5. **The mirror-grader signature**, and the node's sharpest result. The committee arm's critic
   ranks **the mirror's opinion** at 0.772 / 0.748 — as well as the ceiling ranks the world
   (0.773 / 0.750) — while ranking the **world** at 0.592 / 0.648. The student learned the
   teacher faithfully; the teacher is what is wrong. `ideas/heterogeneous_graders.md` §8
   measured rather than argued, twice.
6. **The blind region is the frontier and is monotone in level.** Mirror accuracy on probes
   (model-graded arm): L2 0.931/0.932, L3 0.924/0.929, L4 0.890/0.832, L5 0.786 (seed 0 only).
7. **Committee agreement is anti-correlated with level**: 0.493/0.532 at L2, 0.526/0.588 at L3,
   0.327/0.450 at L4, 0.174 at L5 (seed 0). The agreement rule deletes the frontier
   preferentially — which is the mechanism on offer for (3), and it is *a posteriori*.
8. **The hybrid's efficiency.** World-query fraction 0.357 / 0.420, bill 0.199% / 0.249% of
   priced time against the ceiling's 0.557% / 0.596% — so ~36% / ~42% of the ceiling's bill
   buys ~92% / ~93% of its counterfactual AUC.
9. **The hybrid's base-rate collapse.** Its outcome buffer ends 62.2% / 67.6% world-graded probe
   rows, and probe rows carry a base rate near 0.06 against experience's ~0.30, so its mirror
   calls positives at 0.0128 / 0.0060 where the world's L2 rate is 0.0512 / 0.0514.

### What does not replicate — stated plainly, not averaged

**The hybrid's chooser.** At seed 0 it is the **worst** of the three (−3.1%, *below* the filed
floor); at seed 2 it is the **best** (+7.7%, above both other mirror arms). Nothing about the
hybrid's machinery moved between the two tags.

**The frontier cells.** Seed 0 has six (L4 ×4, L5 ×2) and seed 2 has four (L4 ×4), and the
ordering of the five arms moves between them. `voicing` Q3e already found its own frontier
advantage did not replicate between exactly these two seeds (+16.1% / −1.7%), so nothing at
L4/L5 here is offered as a claim.

**Era-5.** −0.270 / −0.277 / −0.126 (mg / cg / hy) at seed 0 against −0.224 / −0.236 / −0.278 at
seed 2, with the floor at −0.223 / −0.062 — and seed 2 has no L5 *commit*, so its era 5 is not
the same object. Single-trajectory at each seed. The era-**3** column behaves as `voicing`
recorded: the ceiling's gap is −0.081 / −0.077 across seeds.

### §10.1 A withdrawal: the seed-0 hybrid mechanism

**Withdrawn.** §9(f) offered the base-rate collapse as the mechanism for the hybrid's seed-0
chooser deficit. Seed 2 refutes it: the collapse is **worse** there (0.0060 against a world rate
of 0.0514, and the hybrid's mirror is also worse on its own held-out experience — accuracy 0.888
and AUC 0.959 against 0.946–0.980 and 0.986–0.998 on the other arms) and the hybrid's chooser is
nonetheless the **best** of the three. The collapse is real and replicates; it is not the
explanation of the deficit. **Correction**: the seed-0 hybrid deficit stands as a single-seed
fact with no mechanism, and the general shape it sat inside — *a better judge of counterfactuals
did not become a better chooser* — is itself what does not hold up, since at seed 2 the arm with
the best critic also has the best chooser. What survives across both seeds is only the weaker
statement that **the critic's AUC against the world and the chooser's repair accuracy are not
monotonically related**: the committee arm has the worst world-AUC at seed 0 and nearly the
hybrid's at seed 2, while being the weakest chooser of the three on both.
