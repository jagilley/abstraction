# sotto voce — who grades the babble: an outcome model trained on the learner's own experience stands in for the world on the counterfactual probes, and a committee's disagreement says where it cannot

**Up**: [`../README.md`](../README.md) (voicing) · **Spec**: [`SPEC.md`](SPEC.md) (the
orchestrator's brief verbatim) · **Machinery record**: [`FILES.md`](FILES.md) (every gate, run,
flag and diagnostic, with the run tables) · decisions and every withdrawn diagnosis in
[`DESIGN.md`](DESIGN.md) §1–§10 · **Conversation**: `CONVERSATION.md`[^private] (the
calibration discussion that led here, and the two-seed read).
**Machinery donors** (untouched, forked or imported): [`../`](../README.md) (`sotto_voce.py`
forks `voicing.py` at its `vo_s3b`/`vo_s3e` head; every addition `# [sotto]`-marked, every knob
default off; G-F 0.000e+00 against `enharmonic.py` with every knob off) ·
[`../../../../mjc/committee_head/`](../../../../mjc/committee_head/README.md) (the committee as
a reader, and K) · [`../../fourwall/`](../../fourwall/README.md) (`entry_profile`, the
substitution the probe channel reuses).
**Runs**: 2026-09-14 → 15, two result tags at seeds 0 and 2 plus a G-F replay and a nine-twin
preflight (≈10 GPU-h), every arm clock-yoked to its seed's banked anchor with the replay exact
and nothing cancelled. **Ranks, signs, located mechanisms and per-cell counts are the claims;
the seeds are never averaged.**
**Attribution**: the calibration question that opened the conversation — whether the outer
loop, the forward model and the dynamics framing are strictly necessary, or whether an internal
correctness direction in a well-trained model could carry the climb — the steelman that
hippocampal replay might supply imagined states without a forward model, the demand for a
learning-theoretic rather than a diet argument, the recollection that committees had found where
a forward model is wrong before, and the go are Jasper's (2026-09-14). The reading of
`reading/emotion_in_cortex.md`[^private] against the record,
the identifiability argument (three routes to a label for the road not taken, and no fourth),
the seating of the forward model as the source of the judge's off-support labels, the arm
design, the instrument-only world verdict as the control-variables trick, and the seed choice
are the orchestrator's. The fork, the outcome model and its committee, the gate table and its
falsification, the `vo_preflight_gates` refactor, the reductions, the mechanism offered for the
committee's loss, and every withdrawal beside its correction are the implementer's.

## One-liner

`voicing` found that a judge fed only what the learner chose to write ranks the roads not taken
at chance, and that paying the world for a verdict on off-stream counterfactuals fixes it. This
node asks whether the world has to be paid. **An outcome model trained only on the learner's own
graded sentences, applied to sentences it never wrote, supplies verdicts that recover about half
the value of paying the world, on both seeds, at zero bill** — and it is blind exactly where the
type law says it should be: near-perfect on its own support (held-out AUC 0.996), mediocre one
step off it (precision and recall ≈0.44 at a base rate of 0.06), and degrading monotonically with
level, so the deeper the rung the blinder the mirror. A committee's disagreement **locates** that
blind region between levels and is **self-defeating as a filter**, because on a depth ladder
where you lack data is the frontier; used instead to decide which probes to send to the world,
it buys over 90% of the world-graded ceiling's counterfactual ranking for about 40% of its bill.
The sharpest single number is the mirror-grader criterion of
[`ideas/heterogeneous_graders.md`](../../../../../ideas/heterogeneous_graders.md) §8 measured
twice: a critic taught by the mirror ranks the mirror's opinion as well as the ceiling ranks the
world, and ranks the world far below that.

## The question

The critic in `voicing` is a small judge that reads a situation and a candidate spelling and
predicts the grader's verdict. Its diet is the problem: the learner writes one class per slot, so
its experience carries no information about the untaken class, and a judge cannot be identified
off the policy's support. There are three routes to a label for the road not taken and no
fourth: vary the behaviour (`voicing` Q1: 9% write deviation at L2 costs the L3 commit window),
pay the world (`voicing`'s probe channel: 0.5% of priced time on RHM, a real trial on a plant),
or assume a model that ties unseen pairs to seen ones. The third is the forward model's seat in
the practice arc, stated as a learning-theory claim rather than a diet one, and it is the seat
the neuroscience note's *compilation problem* names from the other side: a value readout has to
be trained on situations for which no real outcome ever arrived, and the candidate mechanism is
offline simulation supplying surrogate outcomes.

RHM is where to measure it, because the world's verdict is exact and cheap enough to compute on
every model-graded probe **as an instrument that no arm consumes**. So for every probe the record
holds both what the mirror said and what the world would have said, and the blind region can be
located rather than inferred. On RHM the imagined state is exact (the substitution into the final
configuration is the probe's own machinery), so this node isolates the *grading* half of
imagination; on a plant both halves would be modelled.

Vocabulary: the **mirror** is the outcome model, a tree net from a rendered configuration and
its root to P(solved), trained by BCE against the world's verdict on the configurations the
learner actually produced, a separate object with its own parameters and streams. The **floor**
is `voicing`'s composed chooser on the filed diet; the **ceiling** is the same chooser with
world-graded probes; both are banked and not re-run. **rep** is the chooser's class accuracy
against the repair set on held-out rows, the honest measure of a choice. A **probe** is a paid
question to the world about an action the learner did not take; here the mirror answers it
instead, or a committee of mirrors does.

## What was built

[`sotto_voce.py`](sotto_voce.py) forks `voicing.py`. Three arms, all the composed chooser with
the probe channel on, every one clock-yoked to its seed's anchor (gate Y-1: exact, 0 cancelled,
on all six paid arms):

- **model-graded** (`so_mg_yk`): the same substitutions, the verdict filed into the critic's
  buffer from one mirror; the world not consulted for filing.
- **committee-graded** (`so_cg_yk`): a K=5 committee of mirrors differing by seed and bootstrap;
  a probe is filed only where the members agree, the threshold the median of the committee's
  spread on held-out experience (DESIGN §4).
- **hybrid** (`so_hy_yk`): as the committee, but where it disagrees the world is paid for the
  verdict and that verdict also trains the mirrors. The bill is the world-graded fraction only.

The mirror files its *probability* rather than a 0.5-thresholded verdict (DESIGN §3d: at the
probe base rate a constant "not solved" out-scores the thresholded mirror on accuracy while its
ranking is far above chance; the hard-verdict variant is untested). The world's verdict is
computed for every probe on every arm and enters no loss, no buffer the critic reads, and no bill
(gate M-1). The mirrors train on experience only, plus the world-graded disagreeing probes on the
hybrid (M-2). Bill exactness (M-3) and the committee's filing rule (M-4) are gated. Twenty-four CPU
gates and a falsification harness at 56/56, under the rule that a gate is not reported until it
has been shown to fail on a deliberate perturbation of what it protects. The whole file-based
preflight gate block is factored into `vo_preflight_gates(outdir)` with a CPU entrypoint, so a
tripped gate is re-evaluated against saved artifacts in seconds rather than by re-running the
twins. [`analyze_sotto.py`](analyze_sotto.py) keeps `voicing`'s sections working on the banked
tags and adds [P] the mirror against the world, [Q] disagreement as a detector, [R] the blind
region located per (level, in-support), and [T] the mirror's gate table re-asserted post hoc.

## Q0 — the mirror sized offline (CPU)

On a proxy corpus from the same world (the banked `voicing` arms never persisted the final
configurations; DESIGN §7.1): the counterfactual-to-experience solve ratio is 3.4× against
`voicing`'s measured 3.5–4×; a K=5 committee reads held-out experience at AUC 0.935 and
counterfactuals at 0.747; its disagreement detects its own error at AUC 0.690, beside
`committee_head`'s 0.716 on the motor substrate; and the agreement rule at the median quantile
files 47% of probes at accuracy 0.92. The median is the setting of record.
[`figures/so_q0_reduction.txt`](figures/so_q0_reduction.txt).

## The two seeds (`so_s1`, `so_s2`)

Seed 0's anchor commits L2/L3/L4/L5 at c48/100/151/186; seed 2's commits L2/L3/L4 at
c59/77/157 and never L5, as `voicing` Q3d recorded, so the seeds' frontier panels are different
tables and are printed apart. Full record:
[`figures/so_seedtable.txt`](figures/so_seedtable.txt).

**(a) The chooser** — pooled L2/L3 repair accuracy (≈195k held-out rows per arm), the cell
`voicing` replicated on every seed that produced those rungs:

| arm | seed 0 | vs floor | seed 2 | vs floor |
|---|---|---|---|---|
| ceiling, world-graded (banked) | 0.2736 | +9.6% | 0.2745 | +10.2% |
| hybrid | 0.2420 | −3.1% | 0.2683 | +7.7% |
| model-graded | 0.2602 | +4.2% | 0.2626 | +5.5% |
| committee-graded | 0.2466 | −1.2% | 0.2560 | +2.8% |
| floor, filed diet (banked) | 0.2496 | — | 0.2490 | — |
| anchor (banked) | 0.2130 | −14.7% | 0.2090 | −16.1% |

**(b) The critic** — held-out AUC ranking counterfactuals against the *world's* verdict, where
`voicing` measured 0.491 for a filed-only critic:

| arm | seed 0 vs filed | seed 0 vs world | seed 2 vs filed | seed 2 vs world |
|---|---|---|---|---|
| ceiling | 0.773 | 0.773 | 0.750 | 0.750 |
| hybrid | 0.712 | 0.712 | 0.703 | 0.701 |
| model-graded | 0.667 | 0.617 | 0.679 | 0.645 |
| committee-graded | 0.772 | 0.592 | 0.748 | 0.648 |

**(c) The mirror against the world per level** (model-graded arm), and the committee's
agreement share per level (committee arm):

| level | s0 n | s0 world rate | s0 mirror acc | s0 agree | s2 n | s2 world rate | s2 mirror acc | s2 agree |
|---|---|---|---|---|---|---|---|---|
| L2 | 141k | 0.054 | 0.931 | 0.49 | 134k | 0.046 | 0.932 | 0.53 |
| L3 | 50k | 0.076 | 0.924 | 0.53 | 62k | 0.061 | 0.929 | 0.59 |
| L4 | 12k | 0.149 | 0.890 | 0.33 | 11k | 0.157 | 0.832 | 0.45 |
| L5 | 1.2k | 0.183 | 0.786 | 0.17 | — | — | — | — |

On held-out experience the mirror reads AUC 0.996 on every arm. On the probes its accuracy
reads 0.93 at a base rate of 0.06, i.e. precision ≈0.44 and recall ≈0.43 for the single mirror
(seed 0: 15,071 positives called, 8,410 false; 6,771 of 15,432 true ones missed).

**(d) The hybrid's bill and its mirror.** World-query fraction 0.357 / 0.420; bill 0.199% /
0.249% of priced time against the ceiling's 0.557% / 0.596%. Its outcome buffer ends 62% / 68%
world-graded probe rows and its mirror calls positives at 0.013 / 0.006 where the world's L2
rate is 0.051.

**(e) Era gaps** at matched clocks, treated minus anchor: the ceiling's era-3 gap is −0.081 /
−0.077, `voicing`'s own observation that the era-3 gap reproduces; era 5 is single-trajectory
at each seed and seed 2 has no L5 commit, so it is not the same object.

## The update

1. **The forward model's grading seat is occupied, and its limit is the type law.** A mirror
   trained on experience alone beats the filed floor on both seeds (+4.2% / +5.5%, 44% / 54% of
   the floor→ceiling gap) with zero world queries on the probes, and every mirror diet lifts the
   critic's counterfactual ranking far off chance. The mirror is near-perfect on its own support
   and degrades monotonically with level off it: dense at the levels the learner has experience
   of, sparse one rung up, which is the arc's type law showing up inside the value organ's diet.
2. **The mirror-grader criterion, measured.** The committee arm's critic ranks the mirror's
   opinion at 0.772 / 0.748, as well as the ceiling ranks the world (0.773 / 0.750), and ranks
   the world at 0.592 / 0.648. The student learned the teacher faithfully; the teacher is what is
   wrong. A judge trained on the learner's own corpus is blind where the learner is blind, as a
   number rather than an argument.
3. **Disagreement is a locator, not a gate.** The committee's agreement is anti-correlated with
   level on both seeds (0.49–0.53 at L2 down to 0.17 at L5), so it correctly says where the
   mirror is blind, and "file only where the committee agrees" deletes the frontier
   preferentially, which is the mechanism on offer, *a posteriori*, for committee-graded losing
   to model-graded on both seeds. `committee_head`'s finding that disagreement marks where you
   lack data holds here; on a depth ladder where you lack data is the frontier.
4. **The right use of disagreement is to decide what to pay for.** The hybrid world-grades a
   third to two fifths of the probes and buys 92–93% of the ceiling's counterfactual AUC for
   36–42% of its bill, on both seeds. Its mirror's positive rate collapses under the probe rows
   it admits, on both seeds; whether that collapse matters is open (§10.1).
5. **The critic's counterfactual ranking and the chooser's accuracy are not monotonically
   related.** The hybrid has the best mirror-arm critic on both seeds and is the worst chooser
   of the three at seed 0 and the best at seed 2 with nothing in its machinery moved; the
   committee arm is the weakest chooser on both seeds with nearly the hybrid's world-AUC at
   seed 2. The seed-0 base-rate mechanism for the hybrid's deficit is withdrawn (DESIGN §10.1).
6. **A gate that holds at preflight scale can be false at run scale.** Three gate defects this
   round, all in the new gates and none in the substrate, all that shape: a per-cycle ledger
   line read as a cumulative count; "every probe drawn is billed", which is the world-graded
   channel's identity; a no-back-door check against a rolling buffer's contents. Each was caught
   by the gate and never by a claim resting on it, and each sits beside its correction in
   DESIGN §7. The reusable output is the CPU-re-runnable gate block.

## What this does not show

The hybrid's chooser and the frontier cells do not replicate between the two seeds and are not
offered as claims; `voicing` Q3e found its own frontier result did not replicate between exactly
these two draws. Era 5 is single-trajectory at each seed. The per-probe disagreement-as-detector
readout is untested on a mature mirror: the row-level instrument sample filled front-to-back and
holds cycles 50–54 only on both tags (reservoir sampling ships for later tags), and the
between-cycle correlation of spread with mirror error (+0.82 on the committee arm) is confounded
with the level mix moving over the run, so only the between-level location in (c) is
established. The mirror's probability rather than a thresholded verdict was filed; the
hard-verdict variant, a cap on the probe share of the hybrid's buffer, and the composition rule
were not swept. The model-graded win is second-order against a floor whose own advantage over
the anchor is +15–16%, measured on two draws. On RHM the imagined state is exact by
construction; the state-prediction half of imagination is untested here.

## Reproduction

```bash
cd experiments/            # MODAL_PROFILE=chromatic
PYTHONPATH=. python3 rhm/practice/voicing/sotto_voce/q0_sotto.py                                # Q0, CPU
PYTHONPATH=. python3 -c "from rhm.practice.voicing.sotto_voce import sotto_voce as S; S.vo_gates_cpu()"  # 24 gates
PYTHONPATH=. python3 rhm/practice/voicing/sotto_voce/gates/falsify.py                            # 56/56
python3 rhm/practice/voicing/sotto_voce/launch_detached.py --fn fidelity_smoke --tag so_gf1     # G-F
bash rhm/practice/voicing/sotto_voce/results/RUN_so_s1.sh                                       # seed 0
bash rhm/practice/voicing/sotto_voce/results/RUN_so_s2.sh                                       # seed 2
modal run rhm/practice/voicing/sotto_voce/sotto_voce.py::preflight_gates --outdir-tag so_s1     # gates on saved artifacts, CPU
python3 rhm/practice/voicing/sotto_voce/fetch_compact.py --tag so_s1 --fetch --replace
python3 rhm/practice/voicing/sotto_voce/analyze_sotto.py --tag so_s1 --yoke-src vo_s3:voi3_dp \
    --bank vo_s3:voi3_dp,vo_s3b:voi3b_comp_yk,vo_s3b:voi3b_comp_pr_yk
```

Volume `rhm-scaling-data:/rhm_practice_sotto/<tag>/`; compact mirrors and reductions under
`figures/` (`so_seedtable.txt` is the two-seed table of record). Every flag per tag, every gate
and every app id is in [`FILES.md`](FILES.md).

## Next steps (queued in `QUEUE.md`[^private], not started)

The per-probe detector on a mature mirror, with the reservoir sample · spread as a weight on the
filed row rather than an admit-or-refuse gate · a cap on the probe share of the hybrid's outcome
buffer · the hard-verdict filed target · the Bayesian composition rule `voicing` already queued,
now with the seed-swapping hybrid as a second reason · the plant port, both halves of imagination
modelled, once `tempo`'s seam-diversity prerequisite is met.

## Files

| file | purpose |
|---|---|
| `sotto_voce.py` | the substrate: `voicing.py` forked; the outcome model and its committee, the two doors, the mode-aware probe channel with the instrument-only world verdict, the four M-gates and their run-level forms, `vo_preflight_gates` + `preflight_gates`, the peak-RSS readout |
| `q0_sotto.py` | Q0 (CPU): the mirror sized on a proxy corpus |
| `analyze_sotto.py` | the reducer: `voicing`'s sections kept, [J] gains `auc_world`, new [P] [Q] [R] [T] |
| `gates/falsify.py` | the falsification harness, 56/56 |
| `launch_detached.py`, `fetch_compact.py` | the launcher and the compact mirror |
| `results/RUN_*.sh` | the commands of record |
| `SPEC.md`, `DESIGN.md`, `FILES.md`, `CONVERSATION.md` | the brief; the decisions and withdrawals; the machinery record; the conversation |

[^private]: Not mirrored: this link points to a document in the private lab repo (the roadmap, the queue, an unrun spec, reading notes, or a conversation). See the top-level README for what is held back and why.
