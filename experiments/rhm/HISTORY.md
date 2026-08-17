# HISTORY — experiments/rhm

*A "Big History"-style reading of this node, from its READMEs rather than its commit log (only ~30 commits
touch this directory — most of the record lives in the documents, not the diffs). Auto-generatable;
overwrite freely. See [README.md](README.md) for status, [FILES.md](FILES.md) for the file index.*

## Why this substrate exists

RHM is a prosthesis for a control the team couldn't get on real language: prior natural-language reduction
work found no clean way to change the data-generating process (vocabulary reduction changed β, a channel
property, not the DGP-tied γ; spectral denoising imprinted its own structure onto what it measured). The
Random Hierarchy Model (Cagnetta & Wyart) gives fully controllable hierarchical depth with known ground
truth, cleanly separating DGP from channel interventions ([README.md](README.md) Motivation). The first
result reproduced this promise immediately: **m** (synonymic multiplicity) dominates **L** (depth) by
roughly 3:1 as the scaling-exponent bottleneck ([SWEEP_README.md](SWEEP_README.md)) — entropy, not depth,
structures everything downstream.

## The residual-rank instrument: built, trusted, and eventually retired

An early belief — residual rank reflects DGP complexity, inherited from language/MNIST/grokking work —
got a full RHM test and came back tangled: rank tracked FM capacity and architecture mismatch as much as
the DGP ([RESIDUAL_RANK_README.md](RESIDUAL_RANK_README.md)). The **L-to-m transition**
([REGIME_TRANSITION_README.md](REGIME_TRANSITION_README.md)) and the bottom-up learning wave
([PER_LEVEL_LOSS_README.md](PER_LEVEL_LOSS_README.md)) established the composition depth ceiling
(~1-2 levels at the working hyperparameters) as a real, reproducible phenomenon. Loss-shaping experiments
then asked whether that ceiling was an *objective*-induced bias. It is not: label smoothing hurts every
level equally ([LABEL_SMOOTHING_README.md](LABEL_SMOOTHING_README.md)); focal loss and confidence
thresholding preserve compositional learning while incidentally buying FM legibility "for free," motivating
focal loss to prime later self-knowledge work
([LOSS_WEIGHTING_README.md](LOSS_WEIGHTING_README.md)). [RHM_DEEP_COMPOSITION_README.md](RHM_DEEP_COMPOSITION_README.md)
then retroactively reinterpreted part of the team's own history: exact belief propagation showed the
composition ceiling is partly an **information ceiling**, not purely an optimization failure — a
tripartite recoverable / representable / not-self-supervised-learnable statement replaced the earlier
capacity-only story. [RHM_FRONTIER_AND_LEGIBILITY_README.md](RHM_FRONTIER_AND_LEGIBILITY_README.md) found
the actual learnability gate is **m**, not occupancy (a v16m2 vs v32m4 kill shot at identical occupancy,
opposite root recovery) — and caught its own false negative along the way: an earlier "knowledge grok
without circuit grok" reading turned out to rest on a weight-decay setting that was *effectively zero
pressure*, discovered only by checking the weight norm directly. The rank instrument itself was finally
retired on 2026-07-27: [`residual_decomposition/README.md`](residual_decomposition/README.md) showed
across RHM, language, and MNIST that the old single `R_res` number was reading a **saturated FM's noise
floor**, not DGP complexity, and replaced it with two decoupled quantities — β (shape) and
`R_res_participation` (level) — falsifying a repo-wide belief doc
([`beliefs/dimensionality_expansion.md`](../../beliefs/dimensionality_expansion.md)) in the process.

## The ratchet arc: a negative result that redirected the whole program

Parallel to the above, [`ratchet/README.md`](ratchet/README.md) asked whether MNIST's self-knowledge-driven
"gated ratchet" (34%→48% compounding over 4 cycles) also compounds on RHM. Across wake-sleep, sparse
supervision, RL/gen-distillation, and FOMAML variants (2026-06-23 → 06-29), every MNIST *dynamic*
reproduced — gate closing, FM tracking, robustness dissociation, crossover timing — but the *magnitude*
never did (gap ≤1%). The arc ruled out capacity (an m=2 run with genuine headroom still failed),
m-sharpening overhead (thresholding changed gate openness but not the gap), and the distillation
bottleneck (generation-based distillation fixed cosine to 0.91 and still didn't compound) before FOMAML's
per-level outer objective settled mechanism cleanly. Verdict: **on stationary data, no outer objective
makes FM-predictability pressure improve compositional NTP — compounding requires a moving frontier, which
stationary RHM cannot supply.** This is the arc's biggest pivot: the explicit reason the program moved
toward active/sculpting/latent-loop lines that manufacture non-stationarity.

## FM-as-regularizer and complexodynamics: self-knowledge is not load-bearing

[RHM_FM_REGULARIZER_README.md](RHM_FM_REGULARIZER_README.md) (2026-07-01) answered the frontier/legibility
arc's question directly: a co-trained, open-loop FM-predictability pressure beats weight decay's
functional-complexity floor by ~10 points at preserved knowledge — "L2-norm complexity ≠ functional
complexity," made empirical, and **self-knowledge (a closed loop) is not load-bearing for functional
simplification — the simplest open-loop pressure suffices.**
[RHM_COMPLEXODYNAMICS_README.md](RHM_COMPLEXODYNAMICS_README.md) (2026-07-07) re-read those saved
trajectories against Aaronson's First Law of Complexodynamics, adding two amendments: the descent needs an
**annealer** (SGD alone arrests mid-descent, "a glass rather than an equilibrated liquid"), and the floor is
not zero but the **DGP's own sophistication relative to the observer's bound**. Its header flags that its
own instrument was later retired by `residual_decomposition/`, and the arc re-cut on the replacement.

## The substrate arc: sleep-chunking → latent loop → sculpting

A separate thread (2026-07-03 → 07-15) built the machinery that everything from directed_sculpting onward
depends on. [SLEEP_CHUNKING_RHM_README.md](SLEEP_CHUNKING_RHM_README.md) found chunking is an **objective**
phenomenon, not a capacity one, and — after a self-retraction that every earlier run had accidentally
tested "compressed NTP" rather than genuine latent self-prediction — that only a **latent target**, not a
token target, lets reorganization cross the model's own inference frontier.
[RHM_LATENT_LOOP_README.md](RHM_LATENT_LOOP_README.md) (2026-07-04→08) is the conceptual keystone: crossing
target-type × loop-closure finally produced the first generalizable RHM self-knowledge (+0.22, comparable
to language's +0.18) — but only under a latent target; token-conditional-loop *inverts* it. This doc
minted the **frontier-moving-vs-capped** distinction (grounded targets move the frontier, endogenous/
self-referential targets cap it) and the **two novelties** result — sample-novelty refills breadth, only
*abstraction*-novelty moves depth — which becomes the explicit motivating question for
[`specialization/`](specialization/README.md). [RHM_EDIT_CONTROL_README.md](RHM_EDIT_CONTROL_README.md)
turned RHM into a control task and found a second axis orthogonal to planning quality: belief-space plans
can be nearly optimal (0.71–0.98) while true world-success stays near zero, until a generator-proposes/
cerebellar-veto fix repairs it. [RHM_SCULPTING_README.md](RHM_SCULPTING_README.md) is the substrate doc
proper: after an early failure diagnosed as a task problem rather than a method problem, its central
finding is that **grounding is the pivot** — an endogenous "be-predictable" planner caps, a grounded
planner both moves *and expands* the belief. [RHM_SCULPT_CONTINUAL_README.md](RHM_SCULPT_CONTINUAL_README.md)
closed this phase with a "weak ratchet" — the plannability frontier exhausts in a single pass on a fixed
task — naming the explicit next step: non-stationary abstraction-novelty, the seam into `specialization/`
and `directed_sculpting/`.

## Specialization: a null that got reframed as a positive

[`specialization/README.md`](specialization/README.md) (2026-07-16/18) tested whether the depth frontier
can be moved by *allocation* rather than a direct target: breadth restriction is inert, level-reweighting
is actively harmful, and only a direct oracle label recruits depth. Concentrating even a direct target onto
a subtree still failed to specialize — broad training recovers a subtree's own deep structure *better*
than narrow training on it (**Student-B > Student-A**), because RHM's single shared ruleset gives every
subtree the same generative machinery below the root. A 2026-07-18 reframe inverted the reading: this
"failure" is the *enabling condition* for sample-efficient generalization, not a deficiency — sharpening
the future target into what recruits level ℓ+1 once ℓ is consolidated, with context-sensitive (rather than
context-free) rules floated as the favored way to give RHM genuine, non-shared domains.

## Directed sculpting: an instrument built, broken, and repaired — repeatedly

[`directed_sculpting/`](directed_sculpting/README.md) (2026-07-28 onward) ported mjc's E3 inner/outer loop
to RHM to delete geometry cost — mjc's own arc had been repeatedly derailed by hand-tuned environment bugs.
Both founding bets broke immediately: RHM's uniform rule usage makes relevance × visits collapse to visits
alone by construction, and raw CE is *exactly* blind to support-fixed rule drift. This "build an instrument
→ discover it's confounded → repair → retest" pattern recurs at nearly every step: the gap-to-reference
metric was itself confounded by ordinary continued training
(fixed with a difference-in-differences); the reducibility tap came back *exactly inverted* because a
fixed-budget counterfactual measures marginal, data-starved return; and the arc's largest single correction
is logical rather than empirical — the homeostatic argument's "therefore a learner will climb" doesn't
follow from its own premises, since nothing had priced the *integral* of repair cost until
samples-per-drift-event was starved directly. [`full_loop/README.md`](directed_sculpting/full_loop/README.md)'s
reward-free relevance signal is the arc's positive: it separates real tokens from distractors 13–18× and
recovers 76% of a privileged oracle's allocation — but expansion turns out to be a property of the
**grader's type**, not of non-stationarity itself. A late correction in
[`level_moves/README.md`](directed_sculpting/full_loop/level_moves/README.md) found an apparent depth
preference was actually **span**, repaired into "the value prefers compositional reach." This substrate
feeds [`conditional_revision/`](conditional_revision/README.md) (which treats full_loop's replicated
"endogenous targets cap" as a standing prior) and [`practice/`](practice/README.md) (which forks
level_moves' action space and hierarchical damage wholesale).

## Minting: a support oracle is not a density oracle

[`minting/README.md`](minting/README.md) (2026-07-30) tested self-generated-and-filtered training data: a
learner mints candidate sequences and folds accepted ones back in. Acceptance type matters — the true rule
table beats no filter and beats own-likelihood — but the entire range runs from harmful to break-even, not
worse-to-better, while real held-out data at matched volume clears the same bar by 10×. The mechanism: a
rule-table verifier certifies **support** (is this sequence legal?), not **density** (how likely should it
be?) — and next-token training is density matching, so a support oracle can delete out-of-support mass but
never supply the density information actually missing. A retraction: an earlier single-seed "mirror
geometry collapses to replay" finding turned out to be ordinary overfitting, not self-mirroring per se.

## The introspection thread: directional-not-scalar, six times over

A signal established in `a2a_forward`'s `EMOTION_INJECTION_README.md` — a forecast's value is in its
*content*, not its scalar *evaluation* ("this will be hard" doesn't say what to do differently) — recurs
across this entire late-summer thread as both warning and confirmation.
[`confabulation/README.md`](confabulation/README.md) (2026-07-22) found the first-person report has a real
advantage over a capacity-matched observer, but only on the target not cheaply recoverable from the I/O
map, and retracted an intermediate reading that its pooled effect-size estimator had manufactured a
contrast that didn't exist. [`endogenous_teacher/README.md`](endogenous_teacher/README.md) (2026-08-04/05)
found the model's residual is real and anti-correlated with token surprisal — it tracks computational load,
not epistemic difficulty — but a *scalar* weighting intervention was a clean null, diagnosed as structural:
a depth-conditioned FM's predictor and target share an identical information set, so its residual can only
mean "I lacked capacity," never "I was wrong." [`conditional_revision/README.md`](conditional_revision/README.md)
(2026-08-07/08), built explicitly as "the measurement `endogenous_teacher` needed before its
interventions," moves the FM's conditioning gap from depth to time so belief revision becomes exactly
computable via belief propagation. Its own dramatic sign flip (correlation with surprisal −0.34 → +0.65) is
then partly deflated by its own follow-up gate: most of the flip is surprisal, and the binding constraint is
**belief depth**, not the conditioning gap; `sculpt_slip/` further kills an aleatoric-filtering line because
the FM is bias-dominated, not variance-dominated, at every budget tested. The thread's most recent turn,
`confabulation/temporal/` (2026-08-11) and its child `confabulation/temporal/epistemics/`, is the current
standing belief: privileged self-access survives the depth-to-time axis change, but it is **epistemically
empty** — "the temporal residual is charged, and the charge is public," correlating with belief revision no
better than a matched external observer, with the FM adding no value as a *signal-former*, only as a
*timer*. The parent [README.md](README.md) has sections for confabulation, endogenous_teacher, and
conditional_revision but none yet for this newest turn — not yet crystallized upward.

## Practice: the most recent arc, and the current frontier belief

[`practice/README.md`](practice/README.md) (2026-08-14→16) ports the étude's compile op (from
[`mjc/practice/etude/`](../mjc/practice/etude/README.md), which had closed with a structural embarrassment —
its committed units were provably state-independent) onto RHM sculpting, which finally supplies a substrate
where commitment carries state. Six rounds run in tight sequence, each correcting the last:
[`crystallize/`](practice/crystallize/README.md) confirms state-conditioned commitment but falsifies the
certificate as vacuous (committing at cycle 1 matches every gated arm at 26× less cost) and reframes:
δ-silence gates compilation only where practice moves the *executor*. [`practice/ratchet/`](practice/ratchet/README.md)
gives it something to move — a mined, level-indexed action vocabulary — and discovers premature commitment
*forecloses the next level's representation*, not just its accuracy. [`ear/`](practice/ear/README.md)
climbs the evaluator and still gets refused at level 3, the sharpest form of "measurement and decision come
apart." [`recital/`](practice/recital/README.md) removes the era clock and finds no internal pacing signal
prices time at the ladder's bottom — a fixed bottom-heavy schedule wins everywhere.
[`tall/`](practice/tall/README.md) is voided by instrument starvation but leaves two regime-independent
findings standing. The current frontier, [`typed_gaps/`](practice/typed_gaps/README.md) (2026-08-16),
crosses truth-drift against demand-drift into a clean double dissociation, landing on the arc's belief:
**a committed chunk stores demand-concentration, not truth** — maintenance is demand-tracking and
evaluative, not entropic — and the conditioning gap is a **type system**: one organ per currency of change
(dense learning ↔ truth, repair/metering ↔ interface, evaluative re-selection ↔ demand, the teacher ↔ level).

## Threads that recur across the whole node

Two patterns cut across nearly every arc here. **Instrument distrust is a first-class research activity**:
the residual-rank story, the directed-sculpting loop, the specialization null, and conditional_revision's
tracking-appendix audit all follow the same shape — build a readout, discover it's confounded by something
structural (a saturated noise floor, a marginal-return artifact, a shared grammar, a two-error
cancellation), repair it, and only then trust the result. **Negative results are treated as redirections,
not dead ends**: the ratchet arc's stationary-data null is the explicit reason later arcs manufacture
non-stationarity; sculpt_continual's "exhausts in one pass" seams into specialization and directed_sculpting;
specialization's "no domain-specific depth" reframes as evidence for shared-structure transfer, not failure.
The introspection thread's converging belief — privileged access is real but carries no epistemic content
beyond what an external observer can already read off — and the practice arc's typed-gap belief are, as of
this writing, the two live frontiers this node is working from.
