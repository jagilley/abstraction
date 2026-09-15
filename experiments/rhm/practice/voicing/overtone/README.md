# overtone — the readout round: how much of the judge is already in the main model, and what a linear readout, the free prior, and a re-aimed probe budget each buy

**Up**: [`../README.md`](../README.md) (voicing) · **Design record**: [`DESIGN.md`](DESIGN.md)
§1–§8 (the three shapes as built, the additivity correction, the gate table with what each was
shown to fail on, what seed 2 did and did not replicate, five defects) · **Machinery record**:
[`FILES.md`](FILES.md) (every knob, gate, run and app id) · **Code**: every substrate hunk is
`# [overtone]`-marked in [`../voicing.py`](../voicing.py) with its knob default off; the reducer
sections [J1] [J2] [J3] [Z2] [R] live in [`../analyze_voicing.py`](../analyze_voicing.py);
[`analyze_dump.py`](analyze_dump.py) reads the banked rows offline; [`gates/falsify_ov.py`](gates/falsify_ov.py)
is the round's falsification harness.
**Runs**: 2026-09-14 → 15, `ov_pf1` (preflight), `ov_gf1`/`ov_gf2` (fidelity), `ov_s0` (three
arms, seed 0), `ov_s0b` (the corrected-estimator re-run), `ov_s2` (one arm, seed 2); ≈10 GPU-h.
Every treated arm is clock-yoked to a banked anchor (gate Y-1 exact, 0 cancelled, both seeds);
the shadow-carrying arm is bit-identical to the banked `vo_s3b:voi3b_comp_pr_yk` on eleven
behaviour series including the bill (gate R-1 at full scale, three times). **Ranks, signs and
orderings are the claims; the one class of magnitude stated as such is a column that agrees
across two seeds.**
**Attribution**: the question — whether an internal correctness direction in the main model can
carry the judge's job, read through `reading/emotion_in_cortex.md`[^private]
and the Steenwyk preprint on epistemic-state directions — is Jasper's (2026-09-14), as is the
reading that the composed chooser is the biological readout picture. The three shapes, the
additivity correction to S1, and the seed decision are the orchestrator's. The build, the gate
table, the random-init-trunk control, the per-fold repair of the combination and the seed-2
withdrawal are the implementer's.

## One-liner

`voicing` made the chooser — the step from a chosen chunk to the spelling written for it — a
judge trained on the learner's own graded attempts and composed with the executor's own
likelihood, the *prior*. This round asks how much of that judge was already in the main model.
**At two seeds: the main model's confidence direction reads how hard the situation is and is at
chance on which write is right; its per-candidate likelihood carries real choice information;
and an outcome-trained readout on top of it adds 0.10–0.22 AUC on counterfactual rows, which is
where all the candidate-specific signal lives — on the learner's own writes every readout sits
in one narrow band.** An additive readout (linear in context plus candidate) cannot rank
candidates within a context at all: in the chooser's seat it never moves the frontier write and
is worse than the free chooser at L2/L3. A linear probe on the *completed* state gets partway,
0.02–0.06 above a random-init trunk's feature map, and the nonlinear judge the remaining
0.05–0.07. Spending the counterfactual budget where the prior and the judge disagree beats the
anchor at both seeds and does **not** beat uniform spending: the seed-0 frontier gain did not
replicate, and the mechanism offered for it was withdrawn because the criterion is symmetric.

## The question

The roadmap's third pillar reserved a seat for a forward self-model. A calibration pass on
2026-09-14 found that on RHM the seat had been taken by a *value readout on the main model*:
`voicing`'s critic reads the executor trunk's pooled hiddens plus a candidate's content, is
trained on the grader's verdict with its gradient confined, and is composed with the trunk's own
likelihood. That is the emotion-in-cortex picture — a readout outside the representation, reading
a broadcast copy of it, trained by outcomes — and Steenwyk's preprint reports that base language
models carry a linearly readable correctness direction that exceeds their own logprob. Jasper's
hypothesis: perhaps the judge's job is largely *in* the main model already, readable by a probe,
and the verdict diet is buying less than it seems. Three shapes, all on `voicing`'s built
machinery, all read on the identical held-out rows as the critic:

- **S1 — the readout's shape.** Linear twins of the MLP critic on the same diet, as shadows and
  in the chooser's seat.
- **S2 — the zero-verdict scores.** The trunk's free per-row scores — its per-candidate
  likelihood (the prior the chooser already sums) and its entropy at the slot — ranked against
  the verdict with no labels, and an out-of-fold combination to put a number on the critic's
  increment beyond the prior.
- **S3 — the probe budget by disagreement.** `voicing` §26 spends its counterfactual probes
  uniformly. Spend the same per-slot-per-cycle budget where the prior and the critic disagree
  most (`ideas/heterogeneous_graders.md` §9 at the smallest scale it can be tested), keeping a
  uniform quarter so the audit stays comparable.

Vocabulary, since it accumulates: a *filed write* is what the learner actually wrote at a slot
during priced practice, joined to the verdict on the trajectory it created; a *probe* is a
different on-table class substituted into a filed context and graded for one verdict, off-stream;
*AUC* is the chance a scorer ranks a solved row above an unsolved one; the *anchor* is the
banked composed arm of `en_s9` at each seed; the *uniform twin* is the banked composed-with-probes
arm at each seed (`vo_s3b:voi3b_comp_pr_yk`, `vo_s3e:voi3b_comp_pr_yk`).

## What was built

Knob-gated additions to `voicing.py`, every one off by default and gated inert: two shadow
readouts trained on the critic's own batches from their own optimizer and RNG stream (`lin`,
`Linear(dim,1)` over the critic's own state; `dir`, the same over pooled-mean plus candidate
content, no slot); the free scores per held-out row and a per-fold out-of-fold logistic
combination; `ov_critic_hidden=0`, the linear readout *governing*; `ov_probe_dis`, the
disagreement draw (the on-table candidate of another class maximising |z(dp) − z(critic)| at a
filed context, 3/4 of the budget, 1/4 uniform); an end-of-run dump of the recorder's rows and the
heads (`vo_rows.npz`, `vo_heads.pt`, on the volume); and `analyze_dump.py`, which recomputes the
prior over the whole buffer and, with `--post`, fits a linear probe on the trunk's hiddens of the
**post-write** configuration with three controls on the identical rows. Gates R-1 (the shadows
are inert with governance on), R-6 (the re-aimed draw changes the run), R-2…R-8, `falsify_ov.py`
26/26, under `voicing`'s rule that a gate is not reported until shown to fail.

## S1 — the readout's shape

**Shadows on the critic's own held-out rows** (medians over ≈3,300 reads per diet, 30 slots;
seed 0 / seed 2):

| rows | MLP critic | `lin` | `dir` |
|---|---|---|---|
| filed writes | 0.685 / 0.661 | 0.632 / 0.628 | 0.616 / 0.632 |
| probe substitutions | 0.876 / 0.932 | 0.786 / 0.853 | 0.776 / 0.844 |

Every filed column agrees within 0.024 across the seeds and the probe-row order is the same at
both. (The seed-2 probe column is read on a three-quarters disagreement-drawn set; the uniform
quarter is split out in [J3].)

**The linear readout in the seat** (`ovt_comp_pr_lin`, seed 0, yoked): `moved` — the share of
calls where the composed argmax left the prior's row — is 0.031 at L4 and **0.0001 at L5**
against the MLP's 0.300 / 0.063; pooled repair accuracy **−8.8%** relative to the anchor at
L2/L3 and +0.4% at L4/L5, where the MLP reads +28.4% / +3.9%. The reason is structural, not a
tuning fact: a linear map over `u_ctx + e_cand` ranks candidates by `w·e_cand` plus a per-row
offset, so its candidate order is the same in every context. It can read situation difficulty
and a candidate's base rate; it cannot say "this candidate *here*". S1 as run therefore tests
additivity, not linearity ([`DESIGN.md`](DESIGN.md) §S1b).

**The post-write linear probe**, the honest analogue of a residual-stream correctness probe: a
linear map on the trunk's pooled hiddens with the write rendered into the context, fitted on the
critic's training rows, scored on its held-out rows. Uniform-drawn probe rows, medians over
slots, seed 0 / seed 2 (filed rows in [`figures/`](figures/)):

| readout | seed 0 | seed 2 |
|---|---|---|
| context only, write absent | 0.566 | 0.534 |
| linear on a **random-init** trunk, same architecture | 0.735 | 0.717 |
| linear on the trained trunk | 0.796 | 0.739 |
| the MLP critic, recomputed from the banked heads | 0.850 | 0.807 |
| the free prior | 0.686 | 0.703 |

The random-init control was not in the design: the dry run against a synthetic heads file with
a random core reached 0.754, because a random transformer is already a nonlinear feature map
of the rendered string and the grader is exact on that string. Read against chance the trained
trunk's probe would have looked linearly complete; read against its floor it is partly so.

## S2 — the zero-verdict scores

Seed 0 (`ov_s0b`, the corrected per-fold combination; medians over reads): `dpz` is the prior
term the composed chooser sums, `conf` the trunk's negative entropy at the slot's blocks, `crit`
the MLP, `comb` the out-of-fold logistic on (`dpz`, `crit`).

| rows | `dpz` | `conf` | `crit` | prior (oof) | `comb` |
|---|---|---|---|---|---|
| filed writes | 0.577 | 0.584 | **0.646** | 0.556 | 0.620 |
| probe substitutions | 0.717 | 0.498 | **0.805** | 0.708 | **0.832** |

The critic beyond the free prior: **+0.090** filed / **+0.097** probe at seed 0, **+0.118 /
+0.215** at seed 2; the combination beyond the prior +0.124 / +0.225 on probe rows and no gain
over the critic on filed rows at either seed. The trunk's candidate-blind confidence is at
chance on probe rows at both seeds (0.498, 0.540) and 0.58 on filed rows. Recomputed offline
over the whole buffer, the prior's AUC is 0.530 on 230k filed rows and 0.688 on 194k uniform
probe rows (seed 0); the uniform-row figure replicates at seed 2 (0.717).

## S3 — the probe budget spent by disagreement

One arm per seed, yoked to the seed's anchor, three quarters of the probes drawn by the rule at
the uniform twin's bill share (0.0056 / 0.0059 mean of the cycle ledger).

**Pooled repair accuracy against the repair set, n-weighted** (n ≈ 197k at L2/L3; 14.1k / 10.8k
at L4/L5; seed 2's ladder stops at L4):

| | seed 0 vs anchor | seed 0 vs uniform twin | seed 2 vs anchor | seed 2 vs uniform twin |
|---|---|---|---|---|
| L2/L3 | **+31.7%** | +2.5% | **+38.8%** | +5.7% |
| L4(/L5) | +17.2% | +12.8% | +2.0% | +3.7% |

**Consumption-era error at matched clocks** (the lineage's floor between near-identical arms is
0.08):

| arm | seed 0: e3 / e4 / e5 | seed 2: e3 / e4 / e5 |
|---|---|---|
| disagreement probes | 0.537 / **0.465** / **0.508** | 0.561 / 0.551 / 0.687 |
| uniform twin | 0.532 / 0.644 / 0.703 | 0.548 / 0.556 / 0.708 |
| anchor | 0.613 / 0.671 / 0.773 | 0.625 / 0.622 / 0.831 |

Against the anchor the arm is below by 1.8–3.3× the floor at era 5 at both seeds. Against the
uniform twin, seed 0's era-4/5 gaps of −0.179 / −0.195 (2.2–2.4× the floor) become +0.013 /
−0.005 / −0.021 at seed 2, every one inside the floor; per cell the arm is above the twin at 4 of
6 frontier cells at seed 0 and 2 of 4 at seed 2.

**What the rule selected.** From the banked dumps, the prior's AUC against the verdict on the
uniform-drawn rows is 0.683 / 0.717 (seed 0 / seed 2) and on the disagreement-drawn rows
**0.404 / 0.765**. The seed-0 reading offered at the time — "the rule finds the rows the prior is
wrong about" — is withdrawn as a general claim and kept beside its correction in
[`DESIGN.md`](DESIGN.md) §6b: |z(dp) − z(critic)| is symmetric and selects large disagreement in
either direction, so which organ is the extreme one on the selected rows is not a property of
the rule. What survives at both seeds is that the selected substitutions solve far less often
(0.044 / 0.017 against filed rates of 0.32 / 0.31), and that the critic's AUC on them is very
high (0.899 / 0.961).

## The update (discussed with Jasper 2026-09-15)

1. **The main model's confidence direction reads situations, not choices.** It predicts which
   contexts are hard (0.58 on filed rows) and is at chance on which substitution is right, at two
   seeds. Choice-level information lives in the model's per-candidate likelihood and, beyond it,
   in a readout trained on outcomes. The hypothesis that a probed internal correctness direction
   could carry the judge splits along exactly that line.
2. **On the learner's own writes there is almost nothing candidate-specific to read.** Every
   readout — the prior, the confidence, the linear twins, the MLP — sits in one band on filed
   rows, and the critic's whole increment is on counterfactual rows. This is `voicing`'s
   concentration point measured from the probe side: the filed diet is a diet of context
   difficulty, which is why the off-stream channel was worth its half a percent.
3. **Additivity is the wrong shape for a judge; linearity on the completed state is half the
   right one.** An additive readout cannot rank candidates within a context and, in the seat, is
   worse than the free chooser. A linear probe on the post-write state carries "this candidate
   here" and closes about half of the gap between a random feature map and the trained MLP; the
   rest needs the readout to mix context and candidate non-additively. On this two-layer trunk
   that is a fact about the trunk as much as about the readout.
4. **Disagreement-directed probing is not established over uniform probing.** Over the anchor it
   is solid at both seeds, but that is the composed chooser with probes, `voicing`'s result. Over
   the uniform twin the L2/L3 gain is positive twice and small, and the frontier gain is a seed-0
   number that seed 2 took back along with the mechanism offered for it. The honest next form is
   a *signed* rule (queued).
5. **The biological reading survives, with one arrow missing.** The composed chooser is the
   emotion-in-cortex readout picture in miniature: the prior is the expectation of what fits,
   the critic a readout outside the representation trained by outcomes, acting as a correction
   where the expectation is wrong. The situation/choice split of finding 1 has the shape of the
   doc's insula-versus-orbitofrontal split, and its "cortex untangles so the readout can be
   simple" is half true here. The arrow the stack does not have is the readout reshaping cortex;
   `voicing` Q1 is the reason it must be a plasticity gate rather than a shared loss.
6. **For the calibration this round served**: nothing forward-model-shaped was needed anywhere
   in the judge. The seat the roadmap reserved for privileged introspection is occupied, on RHM,
   by a value readout on the main model composed with the main model's own prior.

## What this does not show

The frontier claims in S3 are read at one seed each way and disagree; the composition weight
`w = 1` and the z-score sum are `voicing` Q3b's and were not swept, so `ov_critic_hidden`
changes the readout inside a fixed composition. The probe verdict answers "would this write
alone have been right, holding the rest of the trajectory fixed", not "would the beam have
solved it". The disagreement-drawn probe rows are a selected candidate set; their AUCs are
printed apart and are not comparable to uniform ones. The trunk is a two-layer block model, so
what transfers from the Steenwyk preprint is the shape of the question and not its numbers. The
post-write probe is fitted offline on ≈7k rows per slot and has an easier job than a critic
trained online. One seed-0 oddity — the context-only probe beating the MLP on filed rows — did
not repeat and is not claimed. Five implementer defects were found; one reached a paid tag (the
out-of-fold combination pooled its folds' scores with unequal intercepts), was caught by reading
the output rather than by a gate, and cost one re-run; every withdrawal sits beside its
correction in [`DESIGN.md`](DESIGN.md) §8.

## Reproduction

```bash
cd experiments/            # MODAL_PROFILE=chromatic; the system interpreter behind `modal` needs numpy
PYTHONPATH=. python3 -c "from rhm.practice.voicing import voicing as V; V.vo_gates_cpu(); V.ov_gates_cpu()"
PYTHONPATH=. python3 rhm/practice/voicing/gates/falsify.py                     # 36/36
PYTHONPATH=. python3 rhm/practice/voicing/overtone/gates/falsify_ov.py         # 26/26
bash rhm/practice/voicing/overtone/results/RUN_ov_s0.sh                        # three arms, seed 0
bash rhm/practice/voicing/overtone/results/RUN_ov_s2.sh                        # the S3 arm, seed 2
python3 rhm/practice/voicing/fetch_compact.py --tag ov_s0b --fetch --replace
python3 rhm/practice/voicing/analyze_voicing.py --tag ov_s0 --yoke-src vo_s3:voi3_dp \
    --bank vo_s3:voi3_dp,vo_s3b:voi3b_comp_pr_yk,vo_s3b:voi3b_comp_yk \
    --out rhm/practice/voicing/overtone/figures/ov_s0_reduction.txt
python3 rhm/practice/voicing/overtone/analyze_dump.py --tag ov_s0b --post       # [S1b], CPU
```

Artifacts on `rhm-scaling-data:/rhm_practice_voicing/<tag>/`; the banked heads and rows
(`vo_heads.pt`, `vo_rows.npz`) live there and are gitignored. Reductions and dumps under
[`figures/`](figures/); every flag, gate and app id in [`FILES.md`](FILES.md).

## Next steps (queued in `QUEUE.md`[^private], not started)

A signed allocation rule for the probe budget · a third seed on the uniform-twin comparison at
L2/L3 if that magnitude becomes load-bearing · the post-write linear probe as an in-loop shadow ·
the plasticity-gate arrow, a design step.

## Files

| file | purpose |
|---|---|
| `analyze_dump.py` | the offline reducer over the banked rows: the prior's AUC over the whole buffer, the draw split, and with `--post` the post-write linear probe with its three controls ([S1b]) |
| `gates/falsify_ov.py` | the round's falsification harness — every new gate shown to fail on a deliberate perturbation (26/26) |
| `results/RUN_*.sh` | the commands of record: `ov_pf1`, `ov_s0`, `ov_s0b`, `ov_s2` |
| `figures/` | `ov_s0`, `ov_s0b`, `ov_s2` reductions and dump reductions |
| `DESIGN.md`, `FILES.md` | the decisions and withdrawals; the machinery record |

[^private]: Not mirrored: this link points to a document in the private lab repo (the roadmap, the queue, an unrun spec, reading notes, or a conversation). See the top-level README for what is held back and why.
