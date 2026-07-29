# Corpus breadth was buying grader heterogeneity

**Status**: conceptual synthesis from a 2026-07-29 discussion. **No new experiment, and no new measurement.**
The reframe (§4), the mirror-grader constraint (§5), the retraction (§6), and the minting cut (§7) are new and
unbuilt. §6 withdraws a claim made earlier in the same discussion; it is recorded here because the
misreading it corrects is invited by a table in a sibling doc.
**Date**: 2026-07-29
**Prompt**: *"I wonder if another partial reason for the classic scaling-era finding was that traditional LLMs
have no intrinsic expansion drive, only compression. And therefore, you would do well to start them from as
broad a basis of training data as possible… But if the models have an intrinsic expansion drive… the models
can in principle explore a narrower semantic band or bands, and trust themselves to self-calibrate to not mode
collapse."*
**Builds on**: [meta_learning_under_metered_data.md](meta_learning_under_metered_data.md) (§6's "all of it"
reading, which this doc adds a *second* mechanism to, not replaces), [heterogeneous_graders.md](heterogeneous_graders.md)
(§4's impossibility and §8's mirror-grader criterion — the load-bearing constraint here),
[dimensionality_expansion.md](../beliefs/dimensionality_expansion.md) (the internal/external cap §2 scopes),
[contra-long-horizon-benchmarks.md](contra-long-horizon-benchmarks.md) (text as a corpus of rotations).
**Key experiments**: none new. Anchors re-read from
[`rhm/specialization/`](../experiments/rhm/specialization/README.md) (Exp 1's FULL/SUBTREE control; the
NTP-floor / oracle-aux ceiling pair) and [`rhm/directed_sculpting/full_loop/`](../experiments/rhm/directed_sculpting/full_loop/README.md)
(the ladder, the satiety result).
**Attribution**: the hypothesis (§1), the self-curated-curriculum-from-minted-data move that defeats the
support objection (§2), the reading that minted data's dubious quality is what puts the evaluative side on the
critical path (§3), and the insistence that prior instantiations were not drive-shaped (§6) are Jasper's. The
breadth-as-grader-heterogeneity reframe (§4), the mirror-grader constraint (§5), and the cut in §7 came out of
the exchange.

---

## One-liner

**A semantically distant corpus region is heterogeneous grading material.** Physics is sighted where fantasy
football is blind, so breadth was a cheap, fossilized way to buy the second grader that
[heterogeneous_graders](heterogeneous_graders.md) §4 says no single signal can be. That dissolves the question
"can an intrinsic expansion drive substitute for corpus breadth?" into a better one — **what must the drive buy
instead, and is that thing cheaper inside a narrow band?** The answer the frame forces is *a grader sighted
where the learner is blind*, which verification supplies in narrow bands where breadth cannot, and which
self-evaluation cannot supply at any scale (§8's mirror). The named test is **minting with a non-mirror
verifier**, and RHM is unusually good for it because its rule table is a free verifier that is not made of the
model.

---

## 1. The hypothesis, and why it is not §6

[meta_learning_under_metered_data.md](meta_learning_under_metered_data.md) §6 already concludes "train on all
of it" was right. The hypothesis here reaches the same conclusion by a different mechanism, and the two must
not be merged, because they disagree about what narrowing *costs*:

| | why breadth is right | what narrowing costs |
|---|---|---|
| **§6 — geometric** | language is a shared-substrate world (specialization's single-ruleset geometry, not full_loop's independent-grammar one) | **nothing.** Relevance filtering is *inert* under shared substrate |
| **this doc — mechanistic** | breadth is the only expansion channel a pure compressor has | **the expansion source.** An intrinsic drive would pay it back |

§6 says narrowing is free; the hypothesis says narrowing is expensive. Both predict the same historical
outcome, which is exactly why the distinction has to be drawn deliberately rather than discovered later.

One correction to the hypothesis as stated. *"Traditional LLMs have no intrinsic expansion drive"* is right
with the emphasis on **intrinsic**, but [heterogeneous_graders](heterogeneous_graders.md) §4b is sharper: LLM
training does not *lack* an expansion loop, **it has one implemented in humans**, running on a wall-clock of
weeks — weight decay, width, depth, data mix, when to stop. So corpus breadth is one exogenous expansion
channel among at least two; the other is data-mixture ablation, which §6 identifies as the relevance tap in
concrete form (*"train N models on N mixtures, read downstream evals, pick"*). This matters because it gives
the hypothesis a second and cleaner target: an intrinsic drive should let you skip the **ablation loop**, which
is metered, before it lets you skip the **corpus**, which is not.

## 2. The internal/external cap does not bind, because the drive mints

[`dimensionality_expansion`](../beliefs/dimensionality_expansion.md) splits expansion into *internal*
(self-legibility reorganization — **capped**, *"you can only re-express fixed content so far"*) and *external*
novelty (**unbounded**). The obvious objection to the hypothesis is that an intrinsic drive is the capped
channel: it allocates over content that is present and cannot manufacture content that is absent, so narrowing
the corpus trades the unbounded channel for the bounded one.

**That objection assumes a fixed corpus, and the hypothesis does not.** The vision is that expansion puts the
learner in a state of **self-curating an external curriculum, potentially from newly-minted data** — synthetic
constructions, self-generated problems, actual experiments. Under minting, support is not a constraint on the
learner; it is an *output* of the learner. The cap in the belief file was drawn over the no-new-data case and
does not extend to this one.

This also retires an intermediate distinction that looked load-bearing during the discussion — *narrow
trajectory vs. narrow support*, the observation that a human specialist narrows their trajectory while
retaining a broad reachable library. It is a real distinction, but it is not the operative one, because a
minting learner grows its own support and so is narrow in neither sense in the limit.

**Belief-file consequence.** The internal/external dichotomy needs a third cell, and it is the interesting one:

| route | new content? | bounded? |
|---|---|---|
| internal (re-expression) | no | **capped** |
| external, received (a broad corpus) | yes | unbounded, but exogenous and pre-paid |
| **external, minted (self-curated curriculum)** | **yes** | **unbounded — and the quality is unwarranted** |

## 3. Where the meter actually sits: verified tokens, not tokens

The third cell's distinguishing property is that its data arrives **without warrant**. Received wisdom comes
pre-certified by the systems that produced it — [heterogeneous_graders](heterogeneous_graders.md) §3's reading
of pretraining as *"distillation from the output channel of an already-finished mind"* is exactly a statement
that the corpus is pre-paid. Minted data has no such backing, and its defining problem is that it is *cheap in
volume and dubious in quality*.

So the binding cost moves from **acquisition** to **verification**. This is a sharper localization of the meter
than [metered-data](meta_learning_under_metered_data.md) §3 gives: that doc's meter is on samples, and the
correct statement for a minting learner is that **the scarce resource was never tokens, it is verified tokens**.
Its own §3 already lists the right instance without naming it as this — *"expert attention is a third (each
hour costs an hour)"*. Doing research is the canonical case: minting a new concisely-expressible idea in this
line of work costs many experiments, and the experiments, not the ideas, are the metered thing.

That closes a loop the metered-data doc leaves open. Evaluative meta-learning is described there as the optimal
move under metered data; here the meter and the evaluative loop turn out to be *the same object seen twice* —
what is metered is verification, and verification is what an evaluative grader does.

## 4. The reframe: breadth was buying grader heterogeneity

The hypothesis's own framing contains the clue and does not cash it in — *"they can draw connections which may
be surprisingly **distant** semantically."* Read *distance* through [heterogeneous_graders](heterogeneous_graders.md) §4:

> **No single learning signal can be both dense and evaluative.** […] the two-teacher structure is **derived,
> not designed**.

and through §5's claim that the payoff of multiple organs is not capacity or division of labour but **the
detectability of grader blindness**, which single-objective learning cannot have at any scale.

**A semantically distant corpus region is heterogeneous grading material.** Physics and fantasy football are
sighted in different places. A corpus wide enough to contain both contains, fossilized, many perspectives that
are blind in different places — and the wider it is, the less correlated their blind spots. On this reading the
scaling era's breadth was not primarily an *expansion source* (more content) but a **cheap proxy for grader
heterogeneity** (more independent vantage points), purchased in a single homogeneous channel.

This is a better account of the original intuition than "more content" is, for one reason: it explains why
*distance specifically* is what pays. Under a content reading, twice as much physics should be worth roughly
what physics-plus-fantasy-football is worth. Under the heterogeneity reading it is not, and the asymmetry is
the whole phenomenon. It also composes cleanly with §6's geometric reading rather than competing with it —
shared substrate is what makes distant regions *commensurable* enough to grade each other at all, and distance
is what makes the grading informative. **Geometry supplies the common currency; distance supplies the
disagreement.**

**And it re-poses the hypothesis in its strongest form.** The question is not "does the drive substitute for
breadth." It is:

> **What does the drive need instead of breadth, and is that thing cheaper inside a narrow band?**

Answer: a grader sighted where the learner is blind. Breadth is *one* way to buy that. **Verification is
another, and it is available in narrow bands where breadth is not** — which is, non-coincidentally, §8's
prediction about where RL post-training visibly works (verifiable-reward domains ≫ LLM-as-judge). That is the
version of the hypothesis this doc endorses, and it is an argument, not a measurement.

## 5. The constraint: the mirror-grader criterion is what "self-calibrate" runs into

The hypothesis's closing clause — *"trust themselves to self-calibrate to not mode collapse"* — is where the
frame bites hardest, and the objection is from the frame's own sharpest statement
([heterogeneous_graders](heterogeneous_graders.md) §8):

> An LLM-as-judge is not a second grader. **It is the same grader in a mirror** — same corpus, same failure
> geometry, blind in the same places. A genuine second grader must be **sighted where the first is blind**.

Narrow the band, mint inside it, and evaluate the mintings with anything derived from that band, and you have
one grader in a mirror. That is not a hypothetical: it is model autophagy, and it is the concrete mechanism of
the mode collapse the hypothesis is trying to rule out. **So "self-calibrate" is coherent iff the acceptance
signal is non-mirror.** Self-consistency, own-likelihood, and a judge fine-tuned on the same band all fail this
by construction; a verifier, an experiment, or contact with the world all pass it.

**Two failure modes bracket the proposal, at opposite ends, and both are already measured or named here.**

- **Broad corpus + intrinsic drive → dilettantism.** [`full_loop`](../experiments/rhm/directed_sculpting/full_loop/README.md)'s
  satiety arm walks into `structA` — reducible 0.845, relevance 0.164, *"genuine, learnable, richly-structured,
  useless"* — and is worse overall by **+0.0691 ± 0.0107**, larger than the entire uniform→oracle prize of
  0.063. metered-data §7: *"a stop-signal without a direction produces sideways motion."* Note this is the
  **opposite sign** from the hypothesis's stated worry: the measured failure of an untethered drive is
  over-broadening, not collapse.
- **Narrow corpus + intrinsic drive, no minting → stall.** LP is ~0 when mastered and ~0 when irreducible.
  Master a narrow band and the drive goes quiet — the dark room, plus §2's internal cap. Not collapse; stall.

**Minting is what escapes the second, and the verifier is what escapes the first.** A minting learner cannot
stall, because it makes more content — but it can drift, and drift-under-self-evaluation is the first failure
wearing a new coat. This is why §5's constraint is the load-bearing one and not a caveat.

## 6. Retraction: cut-2a is not evidence about intrinsic drives, and §5 of the metered-data doc invites reading it as such

**Withdrawn claim** (made earlier in the 2026-07-29 discussion, before this doc): that
[specialization](../experiments/rhm/specialization/README.md) cut-2a shows corpus breadth and an intrinsic
expansion drive are **complements rather than substitutes** — on the grounds that breadth is worth ≈0 without a
depth-recruiting mechanism (Exp 1: every Δ ≤ 0.013) and worth 0.140 at d4 with one (`aux_all` 0.581 vs `aux_A`
0.440 on A's own domain, m=8).

**Why it does not hold.** `aux_all`/`aux_A` is a **precomputed external oracle label** — the ground-truth
ancestor CE. It has no selection step, no outer loop, and no evaluation of what to learn next; metered-data's
own "What this does not establish" classifies this family as *"external and precomputed."* It is a *teacher*,
not a drive. Whatever it measures about breadth cannot transfer to an intrinsic-drive setting without an
argument that the two are interchangeable, and there is no such argument — indeed §4b's whole point is that the
grader's *type* is the operative variable.

**Two further reasons it was the wrong instrument even setting type aside**, both already flagged in the source
README: cut-2a is **supervision-breadth on a corpus that stays broad in every arm** (*"every condition trains
flat NTP on all roots"*), not corpus-breadth; and it carries an unresolved **count confound** (`aux_A` draws
~50k A-sequences vs `aux_all`'s 200k), whose equal-count control is specced and unrun.

**The conflation to fix upstream.** [metered-data](meta_learning_under_metered_data.md) §5's geometry table
lists, under one *"inert"* verdict, both Exp 1's corpus-breadth restriction (Δ ≤ 0.013, genuinely inert) and
cut-2a's supervision concentration (0.581 vs 0.440 — **not** inert; −0.140). Those are two different
interventions on two different axes with opposite verdicts, and the second is not a breadth result at all. A
pointer to this section has been added there.

**What survives, and it is worth keeping.** Exp 1 is a clean matched-token corpus-breadth control, and the
specialization line supplies a **calibrated ladder** — NTP floor root **0.08**, oracle-aux ceiling **0.80**,
FULL vs SUBTREE at matched tokens with root at chance in both arms. Those are the right anchors to place §7's
arms against. Note also that Exp 1's inertness is a **null measured at the floor** (both arms stall at ~d3.5,
root at chance even on S), so it does not falsify the hypothesis either — nothing expanded in either arm, so it
cannot speak to whether breadth supplies expansion.

**Generalized lesson, which is the part to carry.** This repo's standing instrument caution is *"check
independence and magnitude-sensitivity before designing around a rank-shaped instrument"*
([dimensionality_expansion](../beliefs/dimensionality_expansion.md)). The analogue for the grader frame:
**before reading a prior result as evidence about a grader, check that the thing that graded it is the kind of
grader the claim is about.** An external precomputed label and an endogenous drive are different rows of §4b's
table, and the table's entire content is that rows differ.

## 7. The cut: minting against a non-mirror verifier

metered-data's central gap is *"no run is both endogenous and expanding"* — `visits` is endogenous but tested
only for allocation; the DP `k*` teacher expands but is external. That cell is still the right one, but **the
version specced there is over full_loop's fixed five channels, which makes it allocation over existing data.**
Nothing is minted, so it cannot test §2–§3. The hypothesis needs a **generative** step, and the acceptance
criterion is where all the content is.

RHM is unusually well-suited for one reason: **the rule table is a free non-mirror verifier.** A minted
sequence either parses bottom-up under the true rules or it does not. That is "does the code run" — no oracle
depth label, no DP teacher, and nothing derived from the model.

**Design** — single controlled variable per contrast, against the standing anchors (floor root 0.08, ceiling 0.80):

- **Seed breadth**: narrow (Exp 1's `SUBTREE`, roots `{0,1,2,3}`, already built) vs broad (`FULL`), matched
  tokens so the existing anchors transfer.
- **Mint**: the learner generates candidate sequences; accepted ones enter the training pool.
- **Acceptance grader** — *the load-bearing arm pair*: **rule-verifier** (non-mirror) vs **own-likelihood**
  (mirror). §5 predicts the mirror arm degenerates, and the degeneration is directly measurable as
  rule-violation rate in the accepted pool. This is mode collapse with a number attached.
- **Static control**: no minting, both seeds — this is Exp 1, already run, so it is free.
- **Readout**: per-level ancestor recovery `d1…d6` on a **frozen held-out probe** drawn from the full tree
  (not the seed), so the readout cannot move just because the pool moved.

**The question it answers**: can a narrow-seeded self-minting learner reach depth it was never shown, and how
much of the broad-seed advantage does verification buy back?

**The confound to instrument from the start.** Mint-and-accept preferentially accepts what the model already
generates well, so the pool drifts toward the easy part of the band *regardless of grader type* — which would
mimic the mirror arm's predicted failure and mimic away the verifier arm's predicted success. The frozen probe
handles the readout side; the pool side needs an explicit **coverage/diversity statistic reported per round**
(e.g. distinct level-ℓ features exercised, and accepted-pool entropy against the seed's). Without it the
headline contrast is uninterpretable in both directions.

## Predictions / falsification

- **Falsified if** the own-likelihood (mirror) minting arm does as well as the rule-verifier arm. That would
  mean the acceptance signal's *type* is not what matters, which is §8's mirror-grader criterion failing on the
  cleanest substrate available for it — and it would restore "trust themselves to self-calibrate" in its
  original, unqualified form.
- **Falsified if** the verifier arm from a narrow seed does not exceed the narrow static control. That is
  minting failing to be an expansion channel at all, and it would reinstate §2's internal cap over the minting
  case.
- **Predicts** narrow-seed + verifier lands strictly between narrow-static and broad-static — i.e. verification
  buys back *some but not all* of breadth. A full recovery would be the strong form of the hypothesis; the
  frame in §4 predicts partial, because a single verifier is one vantage point where a broad corpus is many.
- **Predicts**, from §4, that the payoff of adding a *second, differently-typed* verifier exceeds that of
  doubling the first one's budget. This is [heterogeneous_graders](heterogeneous_graders.md) §10's
  redundancy-vs-heterogeneity falsifier, restated on the minting substrate.
- **Predicts** that an intrinsic expansion drive displaces **data-mixture ablation** (metered, on a human's
  calendar) before it displaces **the corpus** (unmetered), per §1. Cheaper to check against public training
  practice than to run.

## What this does not establish

- **Nothing here is measured.** Every number is re-read from an existing node, and the four claims that matter
  — breadth-as-heterogeneity (§4), minting-defeats-the-cap (§2), verification-as-the-meter (§3), and the
  mirror constraint applying to minting (§5) — are **arguments**. §5's is the best-supported, because it is a
  direct application of a criterion this repo already committed to; §4's is the least, because "distance ⇒
  decorrelated blind spots" is asserted and never quantified.
- **The LLM claims are structural arguments about training setups**, not measurements on any model we have run.
  This is the same caveat metered-data §6 carries, and it applies with more force here because §4 makes a claim
  about *why* a historical training practice worked.
- **The RHM verifier is stronger than any verifier a real learner has.** It is the exact generative process,
  which is privileged access of the same kind the repo criticizes elsewhere (predicting `qpos/qvel`). It makes
  the *mirror-vs-non-mirror* contrast clean and it makes the *absolute* recovery numbers optimistic. A
  learned-verifier arm is the honest follow-on and is not in §7.
- **§4 does not distinguish its own mechanism from §6's on any existing data**, and may not be separable in
  language at all — shared substrate and semantic distance co-vary in every real corpus. §7 does not test §4;
  it tests §5. Separating §4 from §6 needs cut-3 geometry (tunable sharing depth), and is not specced.
- **No claim that a narrow band is *preferable*.** The strongest defensible reading is that verification makes
  a narrow band *viable*, which is a claim about feasibility, not about optimality.

## Open questions

- Is "distance ⇒ decorrelated blind spots" measurable in a corpus, or only assertable? If it is measurable, it
  is the same quantity §6's open question asks for (*"a measurable proxy for sharing depth in a real corpus"*)
  approached from the grader side, and the two should agree.
- Does the mirror-grader criterion admit **degrees**? §8 states it as a dichotomy, but a judge trained on a
  *partially* overlapping corpus is presumably partially sighted. If the failure is graded rather than binary,
  the minting design wants a *sweep* over verifier-seed overlap rather than the two-arm contrast in §7.
- Which of the three routes in §2's table does human learning actually use for expansion after formal
  schooling ends? The hypothesis's own analogy (we teach kids every subject early) is about the *received*
  route, and the interesting claim is that adults switch to the *minted* one — but expert practice looks like
  it leans hard on a small number of trusted external verifiers rather than on self-curation.
- Does minting change the **shape** of the meter? metered-data's open question asks whether a hard budget, a
  per-sample cost, and a risk of catastrophic loss behave differently. Verification cost looks like the second
  kind, but a *wrong* acceptance is absorbing (it poisons the pool), which makes it partly the third.
- If breadth buys heterogeneity, what is the right **unit** — number of distinct vantage points, or their
  pairwise blind-spot correlation? §10's "error covariance" open question is the same question, and answering
  it once would serve both docs.

## Context pointers for a future agent

1. [heterogeneous_graders.md](heterogeneous_graders.md) §4, §4b, §5, §8 — the impossibility, the
   compression/expansion projection, grader blindness, and the mirror criterion. **§8 is the load-bearing one
   for this doc**; §4b is what §6 here says was misapplied.
2. [meta_learning_under_metered_data.md](meta_learning_under_metered_data.md) §5, §6, §7 — the geometry table
   (**read the §6-retraction note now attached to §5 before using its specialization row**), the "all of it"
   argument this doc adds a second mechanism to, and the satiety/dilettantism result.
3. [`rhm/specialization/README.md`](../experiments/rhm/specialization/README.md) — Exp 1 (the corpus-breadth
   control and the anchors §7 reuses) and cut-2a (the retracted evidence). Read Exp 1 and cut-2a as
   **different axes**; conflating them is the error §6 documents.
4. [dimensionality_expansion.md](../beliefs/dimensionality_expansion.md) §"Two routes to expansion" — the
   internal/external cap, and the third cell §2 adds to it.
5. [`rhm/directed_sculpting/full_loop/README.md`](../experiments/rhm/directed_sculpting/full_loop/README.md)
   §3 — satiety, for the dilettantism half of §5's bracket.
