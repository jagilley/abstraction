# History: `experiments/rhm`

RHM (Random Hierarchy Model) was adopted as a controlled synthetic-language
substrate because natural language wouldn't let us separate the
data-generating process (DGP) from channel interventions
([`README.md`](README.md)). Everything below is the story of what that
control bought us, and how often the instruments built to read it out
turned out to be measuring something else.

## Act I — the knob wasn't the one we thought

Depth `L` was assumed to be the primary DGP lever. The first scaling sweep
overturned that: synonymic multiplicity `m` dominates depth by roughly 3:1
([`SWEEP_README.md`](SWEEP_README.md)). Nearly everything downstream is the
slow unpacking of what `m` actually is — and, later, that it governs
*learnability*, not just *recoverability* (Act III).

## Act II — the instrument crisis (rank → β → gauge)

The program's first tool for reading DGP complexity off a forward-model (FM)
was residual rank. It failed repeatedly, each failure first misread as a
scale problem before being correctly diagnosed as an instrument problem.
[`RESIDUAL_RANK_README.md`](RESIDUAL_RANK_README.md): rank rises with FM
capacity — it reflects "how the main model distributes its computation," not
DGP structure — and is superseded by `beta`.
[`RHM_FRONTIER_AND_LEGIBILITY_README.md`](RHM_FRONTIER_AND_LEGIBILITY_README.md)
retracts an earlier overclaim ("m2's residual is NOT MNIST-like low-rank")
and rules rank head-contaminated while η² is head-robust.
[`residual_decomposition/trajectory/README.md`](residual_decomposition/trajectory/README.md)
performs a full autopsy: naive rank (84) collapses to a participation-ratio
read (7), and even the replacement `β` "does not certify itself either" —
0/16 checkpoints pass both trust gates. Net: directions survive, no
trustworthy scalar. Years later,
[`conditional_revision/local_loss/README.md`](conditional_revision/local_loss/README.md)
finds the same disease in new guise: a loss term fell 19× while
scale-invariant predictability moved only 1.9× — "the raw local-loss term is
a poor readout of anything, since most of its movement is a gauge."

The recurring lesson, stated most starkly in
[`REGIME_TRANSITION_README.md`](REGIME_TRANSITION_README.md): a metric can
only read out a phenomenon that's *there* to be measured — "the FM must
genuinely struggle" is a precondition, not a detail.

## Act III — the composition ceiling was information, not optimization

[`RHM_DEEP_COMPOSITION_README.md`](RHM_DEEP_COMPOSITION_README.md) is the
largest single reframe in the corpus: "we have been asking models to recover
deep composition in a regime where deep composition is barely recoverable in
principle." Meta-learning's failures across the ratchet arc (Act VI) are
retroactively re-diagnosed as suffering the same information ceiling as
single-task next-token prediction (NTP).
[`RHM_FRONTIER_AND_LEGIBILITY_README.md`](RHM_FRONTIER_AND_LEGIBILITY_README.md)
then partially corrects *that*: recoverability alone doesn't determine
outcome — matched-recoverability configs (v16m2 vs v32m4) learn at 0.93 vs
0.07 — so `m` carries a learnability effect on top of its recoverability
effect. Organizing claim that emerges: the FM loop is "an amplifier of
structure the base objective already extracted, not a source."

Earlier attempts to move the ceiling by *allocation* rather than objective
all failed cleanly ([`specialization/README.md`](specialization/README.md)):
breadth restriction is inert, deep-skew reweighting hurts. The stall itself
became the phenomenon — "what recruits level ℓ+1 once ℓ is consolidated,"
still open.

## Act IV — complexity ≠ norm, self-knowledge isn't required for it

Weight decay compresses norm 3.2× without touching functional rank —
"knowledge-grok without circuit-grok." Then
[`RHM_FM_REGULARIZER_README.md`](RHM_FM_REGULARIZER_README.md) beats even
that with an open-loop term: "self-knowledge is not load-bearing for
functional simplification." And
[`RHM_COMPLEXODYNAMICS_README.md`](RHM_COMPLEXODYNAMICS_README.md) reframes
a long-standing negative — a deep-η² "arrest" — as the floor, not a
transient, needing "an annealer... a glass, not an equilibrated liquid."

## Act V — controllability and its bottleneck

[`ACTIVE_RHM_README.md`](ACTIVE_RHM_README.md) inverts the arc's reaching
prior: mean-Δ FMs are a structural null on epistemic queries, and even
after fixing that, internalization has no headroom because active-query is
"an inference task in disguise" (act ≈ plan) — "the reaching positive was
the exception, not the rule."
[`RHM_EDIT_CONTROL_README.md`](RHM_EDIT_CONTROL_README.md) finds a second,
unanticipated axis: "the bottleneck is belief faithfulness, not planning or
arity" — planning nails action choice while 99% of sequences are
off-grammar. [`RHM_SCULPTING_README.md`](RHM_SCULPTING_README.md) supplies
the mechanism behind an earlier struggle: Stage-2 corrupt-repair was a task
problem, not a method problem — latents only beat tokens once tokens stop
being a sufficient statistic.

## Act VI — the ratchet: dynamics without magnitude

A multi-month line asked whether the MNIST wake-sleep self-improvement loop
(+48% over 4 cycles) compounds on RHM, where ground truth is known.
[`ratchet/RHM_RATCHET_README.md`](ratchet/RHM_RATCHET_README.md) finds
dynamics replicate but magnitude doesn't (+0.3%), and
[`ratchet/RHM_RATCHET_M2_README.md`](ratchet/RHM_RATCHET_M2_README.md) shows
open loop winning at every cycle. The sharpest reframe is
[`ratchet/RHM_SPARSITY_SWEEP_README.md`](ratchet/RHM_SPARSITY_SWEEP_README.md):
the null wasn't architectural — "dense NTP already provides all the
supervision the model needs" — and local loss only helps once supervision
is sparse (95% masking), giving real single-cycle deepening with no
cross-cycle compounding
([`ratchet/RHM_SPARSE_RATCHET_README.md`](ratchet/RHM_SPARSE_RATCHET_README.md)).
[`ratchet/RHM_RL_RATCHET_README.md`](ratchet/RHM_RL_RATCHET_README.md) draws
a further distinction — masked NTP is sparse but not *rich*; RL reward
constrains the whole sequence, and self-knowledge finally beats open-loop.
The arc concludes at
[`ratchet/RHM_META_LEARNING_README.md`](ratchet/RHM_META_LEARNING_README.md):
five gradient-based meta-learners collapse to multitask learning ("anti-
overfitting, not compositional depth"), and a stationary RHM cannot supply
the moving frontier compounding requires.

## Act VII — the target was wrong, not the input

[`SLEEP_CHUNKING_RHM_README.md`](SLEEP_CHUNKING_RHM_README.md) contains the
arc's other large pivot: "we had the input right and the target wrong —
every run before Exp 6 predicted tokens" instead of the intended structure.
[`RHM_LATENT_LOOP_README.md`](RHM_LATENT_LOOP_README.md) resolves a
standing blind spot: self-knowledge nulls elsewhere weren't a transfer
failure, they were mis-measured (fresh vs. co-trained FMs) — "RHM is the
controlled setting that exposes the precondition language and MNIST hide."
Distillation is correspondingly re-scoped: "on this domain, distillation
reduces to injection-off fine-tuning."

## Act VIII — the endogenous judge, minting, and the composed loop

[`minting/README.md`](minting/README.md) tests self-minted training data
under ten acceptance rules and lands on "a support oracle is not a density
oracle" — verified data helps, unverified hurts, real data still wins.
[`endogenous_teacher/README.md`](endogenous_teacher/README.md) then tests
whether an endogenously-generated residual can itself serve as a teaching
signal: Gate 0 is positive but anti-localized ("tracks computational load,
not epistemic difficulty"), and the intervention is a clean null. Closed
out, with the explicit successor being a *temporal/lead* FM rather than a
depth-wise one — a depth-wise FM's predictor and target share an
information set, so its residual "can only ever mean 'I lacked capacity,'
never 'I was wrong.'" That successor becomes Act IX; the 2026-08-05
"composed loop" commit joins these three threads. In parallel,
[`directed_sculpting/full_loop/README.md`](directed_sculpting/full_loop/README.md)
finds a recurring instrument identity (`KL(w‖uniform) = log m − H(w)`) makes
raw cross-entropy provably blind to drift — "a null by construction that
reads as a finding" — and, once fixed, that expansion is "a property of the
grader's type, not of non-stationarity," converging on: "levels are an
action-space axis, not an allocation-space one."

## Act IX — the temporal turn: belief revision, and the charge that's public

The most recently and heavily worked thread.
[`conditional_revision/SPEC.md`](conditional_revision/SPEC.md) opens from a
phenomenological claim — "felt surprise is how much a word changes your
model, not how improbable it was" — and finds, after gating, that the
result survives only in *belief* coordinates, not surprisal-explained
variance: a "directional-not-scalar" pattern the repo has now hit four
times. The binding constraint then turns out to be belief *depth*, not the
conditioning gap that was the original independent variable
([`conditional_revision/README.md`](conditional_revision/README.md)).

Two sub-threads were killed cleanly, worth citing because the kills
themselves are the finding:
[`sculpt_slip/README.md`](conditional_revision/sculpt_slip/README.md) ("the
FM is bias-dominated, not variance-dominated, at every budget" — an
aleatoric filter only pays off when the learner is variance-limited) and
[`aleatoric_fraction/README.md`](conditional_revision/aleatoric_fraction/README.md)
(the model sits on a measured NTP-protected floor with no headroom).
[`synonym_retention/README.md`](conditional_revision/synonym_retention/README.md)
reassigns credit for an apparent selective-forgetting effect to plain NTP
with no local loss at all, re-scoping `local_loss`'s headline ratio as
comparing nested perturbation magnitudes, not isolating selectivity — "the
conclusion survives; the inference does not." And
[`rule_family/NOTES.md`](conditional_revision/rule_family/NOTES.md) reframes
the substrate itself: under one fixed rule set, reducibility is static, so
the line's headline constraint "may be regime-induced rather than
intrinsic." Drawing contexts from multiple rule sets produces the repo's
first positive in-context-learning reading on RHM (0.645 of an oracle
ceiling).

Crossing back into confabulation,
[`confabulation/temporal/README.md`](confabulation/temporal/README.md)
moves the conditioning gap from six blocks to a single token and finds
privileged access survives — but decomposes it into the arc's cleanest
dissociation: "the temporal residual is charged, and the charge is public."
Self-report and a capacity-matched external observer track the revision
equally well; privilege is not privacy. A follow-on `epistemics` cut then
finds the temporal FM "epistemically inert as a signal-former," leaving
timing as its only remaining distinct claim — the open question at the
frontier of this directory as of writing.

## Where this leaves us

The through-line is methodological: almost every "finding" here was, on a
longer timescale, revealed to be partly an artifact of the instrument used
to see it (rank, KL-to-uniform, pooled η², raw local-loss magnitude,
single-position probes). The survivors — `m` over `L` as the real
complexity knob, the information-ceiling reading of the composition stall,
richness over sparsity as what makes self-supervision bite, revision as
directional not scalar — all came from noticing an instrument's blind spot,
not a first-pass positive. Live edge: `rule_family/precision/SPEC.md`
(designed, not run) and the timing-vs-privacy question left open by
[`confabulation/temporal`](confabulation/temporal/README.md).
