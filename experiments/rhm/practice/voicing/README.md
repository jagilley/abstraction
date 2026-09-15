# voicing — the chooser at the class: a judge trained on the learner's own attempts, composed with the free forward model, fed by babbling that stays out of the book

**Up**: [`../README.md`](../README.md) (practice arc) · **Spec**: [`SPEC.md`](SPEC.md) (the
orchestrator's prompt verbatim, with the dated revisions Q1 → Q3d appended) · **Machinery
record**: [`FILES.md`](FILES.md) (every gate, run, flag and diagnostic, with the run tables) ·
decisions and every withdrawn diagnosis in [`DESIGN.md`](DESIGN.md) §0–§38 · **Conversation**:
`CONVERSATION.md`[^private].
**Machinery donors** (untouched, forked or imported): [`../enharmonic/`](../enharmonic/README.md)
(`voicing.py` forks `enharmonic.py` at its `en_s9` head; `quotient.py` and `merge.py` imported;
every addition `# [voicing]`-marked, every knob default off; the G-F replay against the donor at
0.000e+00 nine times across the round) · [`../embouchure/`](../embouchure/README.md) (the write
record, the record key, the two ears) · [`../native/span/span_net.py`](../native/span/span_net.py)
(the corridor head) · [`../fourwall/`](../fourwall/README.md) (`entry_profile`, the substitution
the probe channel reuses).
**Runs**: 2026-09-12 → 14, eight result tags at seeds 0–3 (≈30 GPU-h of paid runs, ≈8 GPU-h of
preflights and fidelity replays), every fork gated bit-identical against its donor with the knobs
off, every yoked arm's replay exact. **Ranks, signs, located mechanisms, per-cell counts and
multiples of in-tag floors are the claims; one magnitude is stated as a magnitude because it
reproduces to the third decimal across seeds.**
**Attribution**: the go for the lane, the reading of the merge licence as a prior whose veto is
priced (a higher-order weight decay), and the reading that a separately trained judge on top of
the forward model is how biology does it — combined, not substituted — are Jasper's (2026-09-12
→ 13); the class lane was named in `embouchure`'s spec. The lane's design, the actor/critic
split after Q1, babbling off-stream as a priced channel that never enters the repertoire, the
clock yoke as the round's control, and the seed decisions are the orchestrator's. The fork, the
write record captured at the write, the gate table, the falsification rule (a gate is not
reported until it has been shown to fail), the liveness gate V-6 after the one defect the
inertness gates could not see, and every per-seed mechanism are the implementer's.

## One-liner

On every prior RHM practice node the step from a chosen chunk to the spelling written for it —
the *chooser* — was a free procedure: a max-sum search over the book scored by the frozen surface
model, never trained on what the learner chose and whether it worked. This node makes the chooser
an organ trained on the learner's own graded attempts, filed by what it meant. **A judge that
reads the context and a candidate's content, trained by the grader's verdict on filed writes, and
combined with the surface model's prior rather than replacing it, raises the executor's class
accuracy against the repair set by ≈19% relative at the levels every seed reaches (≈31% when the
judge is also fed priced counterfactual probes) and lowers consumption-era error, on three seeds,
with the era-3 gap reproducing to the third decimal.** Replacing the prior with the judge chooses
worse. Babbling has to live off-stream: any variation in the writes the next level is mined from
costs that level's commit window, and a judge fed only what the beam chose to write ranks
counterfactuals at chance, where priced substitutions into solved sentences buy discrimination
of 0.80 for half a percent of priced time. **What the node cannot claim is a frontier effect**:
the composed chooser's advantage at L4/L5 was large on the one seed whose ladder reached L5 and
absent on the only other seed that reached L4 — and the anchor's ladder itself reaches L5 at one
seed in four, which makes the pacer's top-rung windows the lineage's critical path.

## Children

### [`sotto_voce/`](sotto_voce/README.md) — who grades the babble: the mirror, the committee and the hybrid (2026-09-14→15)

**Goal**: ask whether the world has to be paid for the probe channel's verdicts. An outcome model
trained only on the learner's own graded sentences (the *mirror*) grades the off-stream
counterfactuals instead, alone, as a K=5 committee filing where it agrees, and as a hybrid that
pays the world where the committee disagrees; every model-graded probe is also graded by the
world as an instrument no arm consumes. Two seeds, every arm yoked to its seed's anchor.

**Finding**: **the mirror recovers about half the value of paying the world, on both seeds, at
zero bill** (+4.2% / +5.5% over the filed floor at L2/L3 against the ceiling's +9.6% / +10.2%),
and it is blind exactly where the type law says: held-out AUC 0.996 on its own support,
precision and recall ≈0.44 one step off it, degrading monotonically with level. The
mirror-grader criterion measured twice: a critic taught by the mirror ranks the mirror's opinion
at 0.77 / 0.75 and the world at 0.59 / 0.65. The committee's disagreement locates the blind
region between levels and is self-defeating as a filter (agreement falls from ≈0.5 at L2 to 0.17
at L5, so the gate deletes the frontier); used to decide what to pay for, it buys 92–93% of the
ceiling's counterfactual ranking for 36–42% of its bill. The hybrid's chooser and the frontier
cells do not replicate between seeds and are not claims. Full record:
[`sotto_voce/README.md`](sotto_voce/README.md).

## The question

`enharmonic` reached L5 on the learner's own evidence and got no value from it: a richer,
all-legal book performed worse than a poorer one because the executor's expansion choice is
`macro_features`' argmax over the frozen generator's logits, a forward model of the surface used
as a scorer. `embouchure` found, one organ down, that the learner learns the intent-to-word rule
from its own graded attempts faster than from the corpus, iff it files those attempts by the
record of what it meant (the efference copy) and not by what its ear parsed. Read together, the
chooser is a third organ beside address and trust — the inverse model of production — and every
biological inverse model is trained on the animal's own attempts. The question: **can the executor
learn which class to write at a slot from its own graded attempts, keyed by the record, and does
that turn the quotient's arrival at L5 into value?**

Vocabulary, since it accumulates: a *slot* is a (level, node) position the planner π has chosen to
fill with a chunk; the *book* is the committed table of spellings for that level; the *DP* is the
free chooser (the surface model's max-sum over the book); the *corridor head* is the small trained
head that takes over a slot once it agrees with the DP often enough (its agreement is *parity*, and
below the firing gate the DP takes the slot back); the *record* is what the executor wrote at a
call, captured at the write; the *verdict* is the grader's per-instance judgement, the meter's own
feedback; the *repair set* at a node is the set of classes that would have repaired the instance
there, the honest measure of a choice; the *anchor* is `en_s9`'s composed arm, re-run in every tag
and bit-identical to the banked run.

## What was built

[`voicing.py`](voicing.py) forks `enharmonic.py`. A write record captured at the macro call on
both the fired and the closed path, with the verdict joined per write through the beam's own
parent gathers (gate V-1: exact on 1.59 M filed writes). On-table sampling at a per-block
temperature and ε-greedy on a chosen level. A *critic*: a value head reading the pooled context
and a candidate tuple's level-1 features (content, never a label), emitting P(solve), trained by
BCE against the verdict, its gradient confined to its own parameters, governing a slot once it
holds enough filed rows. Two choice rules on a governed slot — *replace* (the critic's max over
the on-table classes) and *composed* (the DP's per-block score and the critic's logit each
z-scored over the candidate set, summed at weight `w`, argmax) — with the corridor head imitating
the record on governed slots so parity keeps its meaning. A priced *probe channel*: for a filed
context, substitute a different on-table class at that slot in the trajectory's final
configuration, pay the grader for one verdict, file (context, candidate, verdict) into the
critic's buffer and nowhere else. Clock *yokes* that replay a source arm's realised commit and
advance cycles. Eighteen CPU gates and a falsification harness ([`gates/falsify.py`](gates/falsify.py),
36/36): a gate is not reported until it has been shown to fail on a deliberate perturbation of
what it protects, and every inertness gate has a liveness twin whose perturbation is
disconnection. [`analyze_voicing.py`](analyze_voicing.py) reduces with sections [A]–[Z]; [Y]
(the replay matches) is read first on every yoked tag. [`q0_voicing.py`](q0_voicing.py) is the
offline sizing.

## Q0 — the chooser does not choose (CPU)

On the banked `en_s9` run, rebuilt exactly from the logged builds (gate VQ-1): the executor is the
corridor head from era 3 on (share of macro rows 0.96–1.00, a copy of the DP within a 5–7%
misfire); at the L5 commit it writes **one token class on 100% of its calls** out of the six the
book holds, top-1 share 0.996, and 3 of 13 keys at L4 — while its class accuracy against the
repair set at 5n1 is 0.58 and two rows of the book would cover seven of the eight features. Volume
is not the constraint (148–366 captured rows per slot per cycle); variance is. On this draw the
record and the ear's parse agree by construction on the token class — the one reachable bottom-map
collision is between two features that render alike — so the record-vs-recall axis of
`embouchure` Q1 is nil here, and the generator's mislabel with it.

## Q1 — two failures with exact causes (`vo_s1`, 4.04 GPU-h)

| arm | write | head's objective | commits | corridor | misfire |
|---|---|---|---|---|---|
| `dp` | argmax | the donor's self-imitation of the DP | L2, L3, L4, **L5** (= banked `en_s9` at 0.000e+00) | 30 / 30 | 0.068 |
| `own_record` | argmax | BCE of the head's own class likelihood against the verdict | L2 | 2 / 0 | 0.538 |
| `xp` | sampled, T = 0.25 | the donor's | L2 | 16 / 16 | 0.042 |
| `own_record_xp` | sampled | BCE | L2, L3@93 | 2 / 0 | 0.557 |

1. **A calibration loss on the emission head destroys the corridor.** At a filed solve rate of
   0.18–0.23 the loss drives the head's mass on its written class to the base rate and sends the
   rest off-table; parity 0.20–0.33 against a 0.50 firing gate, so the DP wrote every macro call
   and the arms never tested the question. A normalised distribution over what to write cannot
   also hold per-class solve likelihoods: the chooser needs what π already has, a proposal organ
   and a separate judge.
2. **Any early perturbation of the writes costs the commit window above.** Every era advance in
   every arm fired on its cap, and the commit owner reads the era's own level, so L3's window was
   era 2 and closed at c110; the anchor commits at c100. Deviating on 9% of L2 writes missed it,
   and so did the argmax calibration arm. A realised deviation of 0.091 at T = 0.25 against the
   preflight's 0.52–0.67: a sharper plant deviates far less at fixed temperature.

## Q2 — the critic keeps the corridor; frontier-only exploration is exploration in the earned stream (`vo_s2`, 4.04 GPU-h)

| arm | commits | corridor | raw misfire | excess misfire |
|---|---|---|---|---|
| `dp` | L2, L3, L4, **L5** | 30 / 30 | 0.068 | 0.068 |
| `critic` (replace) | L2, **L3@81, L4@111** | 28 / 24 | 0.619 | **0.018** |
| `xp_f` (ε = 0.3 on the highest adopted level) | L2 | 16 / 16 | 0.106 | 0.008 |
| `critic_xp` | L2 | 16 / 16 | 0.784 | n/a |

- **The actor/critic split works as machinery.** With the verdict on a separate head the corridor
  stays open on 28 slots and its misfire, once the critic's own re-decisions are removed, is below
  the anchor's. The critic predicts the verdict above chance from (context, candidate): held-out
  AUC 0.68 / 0.70 / 0.63 at L2 / L3 / L4 on 730–830 rows per slot at base rates 0.13–0.24. It
  governs 28 slots and moves the write off the DP's row on 0.35–0.78 of calls; the L4 cells write
  up to 3.9 distinct tuples where the anchor writes 1–2.
- **As the sole decider it chooses worse.** By the repair measure at 4n0–4n3 in era 4: 0.24 / 0.20
  / 0.20 / 0.36 against the anchor's 0.26 / 0.41 / 0.48 / 0.60. The surface model is a strong prior
  on this world and a modest judge that overrides it on half its calls throws the prior away. Its
  faster climb (L3 nineteen cycles and L4 forty cycles early) came with poorer books and an era
  advanced by quiet, and no L5.
- **ε on the highest adopted level is ε on the stream the next level is earned from.** While L3 is
  being earned the frontier is L2, so both ε arms stayed at L2 for their whole lives and ε never
  acted where the question lives. ε itself was clean: 226,653 fires at a realised 0.302 against
  0.300, 0 off-frontier cycles. On a depth ladder there is no adopted level that is not feeding
  the one above it.

## Q3 — composed on its own clock; the diet unwired (`vo_s3`, 4.6 GPU-h)

The composed chooser on the filed diet, self-paced, committed L2 only: its L3 yield read rose
earlier than the anchor's but flattened near 0.43 and never armed the latch before the c110 cap —
a third different L3-window outcome from a third L2 write distribution. The probe channel was
built, priced and never consumed: a keyword not passed at one call site kept 198,382 and 115,017
graded probes out of the critic's loss, and both probe arms are bit-identical to their no-probe
twins on every behaviour series with the bill differing by exactly the probe count. Every
inertness gate passed, because a diet that is not wired is the most inert thing there is; gate
**V-6** (a diet must move the parameters, shown to fail on the defect itself) now holds the line.
The defect left one clean number: **a critic trained only on what the beam chose to write ranks
counterfactual probe rows at chance** (held-out AUC ≈0.49 on probe rows against ≈0.65 on filed
rows, 612–895 rows per slot over 16 slots).

## Q3b — the ladder held fixed (`vo_s3b`, 5.1 GPU-h, seed 0)

Every treated arm replays the anchor's commits and advances (gate Y-1: exact, 0 cancelled, on all
four), so for the first time every arm has L4 and L5 slots at the anchor's cycles. Class accuracy
against the repair set on the priced beam, held-out (468–3,200 rows per cell):

| cell | anchor | **`comp_yk`** | `comp_pr_yk` | `rep_yk` | `rep_pr_yk` |
|---|---|---|---|---|---|
| 4n0 | 0.269 | **0.284** | 0.278 | 0.281 | 0.259 |
| 4n1 | 0.393 | **0.451** | 0.392 | 0.233 | 0.374 |
| 4n2 | 0.368 | **0.489** | 0.307 | 0.302 | 0.404 |
| 4n3 | 0.632 | **0.683** | 0.727 | 0.655 | 0.626 |
| 5n0 | 0.356 | **0.526** | 0.505 | 0.245 | 0.366 |
| 5n1 | 0.691 | **0.830** | 0.702 | 0.689 | 0.702 |

The composed chooser on the filed diet is above the anchor at 6 of 6 cells; it keeps the prior and
overrides it on 0.03–0.13 of L4 calls (0.46 at 5n1, where it gains most), where replace overrides
on 0.60–0.97 and is worse; corridor misfire in eras ≥ 3 is 0.28 / 0.25 on the composed arms
against 0.59 / 0.82 on replace. Era-5 error at matched clocks: `comp_yk` 0.550 against the
anchor's 0.773, `comp_pr_yk` 0.703, `rep_pr_yk` 0.611, `rep_yk` 0.824. **The probe diet works and
is cheap**: the critic's counterfactual AUC goes 0.491 → 0.798 like-for-like (0.84 / 0.72 / 0.74 at
L3 / L4 / L5), a substituted class solves 3.5–4× less often than the written one, and the bill is
0.0055 of priced time on average, 0.0083 at most. The 2×2 interacted on this seed (probes helped
replace and hurt composed at the frontier), which the seeds below undo. The composed arm ends with
fewer L5 keys at support than the anchor (8 against 17): a better chooser narrows the stream the
next level is mined from.

## Q3c–e — seeds 1, 2, 3: what replicates, and the ladder (`vo_s3c`, `vo_s3d2/3`, `vo_s3e`)

The anchor's own ladder, self-paced, at four seeds (the world, `rule_seed 0`, never moves):

| seed | rungs adopted | advances |
|---|---|---|
| 0 | L2@48, L3@100, L4@151, **L5@186** | 5 on cap |
| 1 | L2@47 and nothing else; 108 cycles | era 2 left at 17/50 and era 3 at 10/70 **by quiet** |
| 2 | L2@59, L3@77, **L4@157** (152 entries) | 5 on cap |
| 3 | L2@52, L3@66 (**4** entries) | 5 on cap |

**`en_s9`'s first endogenous L5 commit is a one-in-four event at this configuration.** Only seed 1
advanced early: delta-silence, which owns the advance, declared era 2 finished while the L3
audition accuracy the commit licence rides on was still swinging by 0.23. Seeds 2 and 3 ran every
era to its cap and still did not commit L5 inside the 12-cycle era 4 — seed 2 with a larger and
more easily licensed L4 book than seed 0's. The yoke is exact at every seed (0 cancelled).

The composed chooser, yoked to each seed's anchor, treated minus anchor:

| | L2/L3 pooled repair accuracy | L4/L5 pooled | era-3 error gap | era-5 error gap |
|---|---|---|---|---|
| seed 0 · `comp_yk` | **+18.5%** (n ≈ 185k) | +16.1% (n ≈ 13.9k) | **−0.048** | −0.223 |
| seed 0 · `comp_pr_yk` | **+31.2%** | +3.9% | −0.081 | −0.070 |
| seed 1 · `comp_yk` | **+24.5%** (n ≈ 48k) | — no frontier | −0.052 | — |
| seed 1 · `comp_pr_yk` | **+27%** | — | | — |
| seed 2 · `comp_yk` | **+19.3%** (n ≈ 197k) | **−1.7%** (n ≈ 10.8k) | **−0.048** | −0.062 |
| seed 2 · `comp_pr_yk` | **+31.5%** | −1.6% | −0.077 | −0.123 |

At the levels every seed reaches the advantage replicates with sizes close enough to be
startling, and the era-3 gap is the same number twice. At the frontier the measurement exists on
two seeds and agrees with itself on neither: seed 0's 6 of 6 cells is a seed-0 fact, the era-5
magnitude moves 3–4× between seeds with the two arms swapping order, and "probes hurt the composed
chooser" reverses at L2/L3 on every seed and in era-5 error at seed 2. The probe bill is
0.44–0.60% of priced time at every seed.

## The update

1. **The chooser can be trained as a judge from the learner's own graded attempts, and it pays as
   a correction to the free forward model, not as a replacement.** The surface model's ranking is
   right more often than a judge of AUC ≈0.65 is; combined at the choice, the judge overrides it on
   a minority of calls and raises class accuracy by ≈19% relative at every level every seed
   reaches, lowering consumption-era error with a gap that reproduces to the third decimal. The
   forward model supplies the prior and the judge the correction — the biological arrangement,
   now measured. The judge's seat is the FM seat list's first *inverse* seat.
2. **The choice organ and the value organ must be separate objects.** A normalised emission
   distribution calibrated to a base rate empties itself; the corridor closes and the treatment
   never runs. This is the same actor/critic split π already has, one organ over.
3. **Babbling must live off-stream, and it is cheap there.** On a depth ladder every adopted level
   feeds the one being earned, so variation in the mined writes — even 9% at L2 — costs the next
   commit window. Priced substitutions into solved sentences that never enter the repertoire buy
   counterfactual discrimination of 0.80, from chance, for half a percent of priced time. A judge
   fed only what the learner chose to write cannot rank what it did not try.
4. **The frontier is untested, and the reason is the ladder.** The anchor reaches L5 at one seed
   in four, and the top-rung windows are 12 and 9 cycles; on the seeds that get there, whether
   the L5 read quiets in time is close to a coin flip. Any claim about value *at* the newest rung
   waits on those windows, which are the `enharmonic` queue's item (iv) with a seed table behind
   it, and on the pacer's advance owner, whose early-quiet at seed 1 is the non-scale-free dead
   zone `inflection` found.
5. **The write record is exact and free on this fork**, captured at the write (1.59 M writes, 0
   bad); on the token class the record and the ear's parse coincide at this draw, so the
   efference-copy contrast of `embouchure` needs a homophonous world to bite at the class.

## What this does not show

No claim about L4/L5 survives the seeds: the two frontier measurements disagree, and the L5 cells
exist at one seed. The era-5 magnitudes are single-trajectory numbers at each seed. The composed
chooser's weight `w = 1` and z-score composition were set once and not swept; the z-scoring
discards the calibration the probe diet buys, which is a candidate mechanism for the seed-0
interaction and untested. The record-vs-recall axis and `ear_record` are nil on this draw by
construction and were withdrawn, not tested. The probe grades a substitution into a trajectory's
final configuration, "would this write alone have been right," not "would the beam have solved
it." Matched-clock error gaps between near-identical arms reach 0.08 in this lineage; the
structural readouts are the claims, and the one error magnitude stated as such is the era-3 gap.
Eight implementer defects were found this round, all caught by gates or by the data before any
claim rested on them, and one gate was found vacuous and repaired; the withdrawals are kept beside
their corrections in `DESIGN.md`.

## Reproduction

```bash
cd experiments/            # MODAL_PROFILE=chromatic; the system interpreter behind `modal` needs numpy
PYTHONPATH=. python3 rhm/practice/voicing/q0_voicing.py                                  # Q0, CPU
PYTHONPATH=. python3 -c "from rhm.practice.voicing import voicing as V; V.vo_gates_cpu()"  # 18 gates
PYTHONPATH=. python3 rhm/practice/voicing/gates/falsify.py                                # 36/36
python3 rhm/practice/voicing/launch_detached.py --fn fidelity_smoke --tag vo_gf9         # G-F
# the tags, each with its full flag line in FILES.md and results/RUN_<tag>.sh, e.g. the yoked 2x2:
bash rhm/practice/voicing/results/RUN_vo_s3b.sh
python3 rhm/practice/voicing/fetch_compact.py --tag vo_s3b --fetch --replace
python3 rhm/practice/voicing/analyze_voicing.py --tag vo_s3b --yoke-src vo_s3:voi3_dp \
    --bank en_s9:endo_ledger_open_ung5_ra,vo_s3:voi3_dp,vo_s2:voi2_critic
```

Volume `rhm-scaling-data:/rhm_practice_voicing/<tag>/`; compact mirrors and reductions under
`figures/` (`vo_s3d_seedtable.txt` is the four-seed table). Every flag per tag, every gate and
every app id is in [`FILES.md`](FILES.md).

## Next steps (queued in `QUEUE.md`[^private], not started)

The top-rung windows — a longer era 4 and 5 on the anchor's ladder, with the seed table as the
reason · the composition rule — a Bayesian combination (log prior + log P(solve)) in place of the
z-score sum · more anchor seeds that reach L5, then the composed chooser yoked to them · the
chooser on a homophonous draw (`embouchure`'s lexicon), where the record-vs-recall axis is not
nil at the class · the advance owner's dead zone, made scale-free.

## Files

| file | purpose |
|---|---|
| `voicing.py` | the substrate: `enharmonic.py` forked; the write record, the sampler and ε, the critic, the two choice rules, the probe channel, the yoke, the instruments, `vo_gates_cpu`; entrypoints `preflight`, `fidelity_smoke`, `enharmonic_run` |
| `analyze_voicing.py` | the reducer: [A]–[I] inherited, [J]–[O] the critic / ε / governance / corridor / composed choice / babbling, [S] per-seed ladders, [Y] the replay match, [Z] the frontier panel |
| `q0_voicing.py` | Q0 (CPU), gate VQ-1 |
| `gates/falsify.py` | the falsification harness — every gate shown to fail before it is reported |
| `launch_detached.py`, `fetch_compact.py` | the launcher and the compact mirror (`--replace`) |
| `results/RUN_*.sh` | the commands of record |
| `SPEC.md`, `DESIGN.md`, `FILES.md` | the prompt with its revisions; the decisions and withdrawals; the machinery record |

[^private]: Not mirrored: this link points to a document in the private lab repo (the roadmap, the queue, an unrun spec, reading notes, or a conversation). See the top-level README for what is held back and why.
