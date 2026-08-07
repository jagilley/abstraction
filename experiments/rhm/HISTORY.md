# History of `rhm` — a big-picture reading

Scope: [`experiments/rhm/`](.), generated fresh from the READMEs, `FILES.md`, and git history
(~85 commits touching this node, 2026-06-20 → 2026-08-05). Not a changelog — an attempt to trace
the turns in thinking.

## Why this node exists

`rhm` was founded as a clean-room substitute for natural-language forward-self-model experiments,
where vocab reduction was entangled with the statistics being measured and no lever cleanly
changed the data-generating process (DGP) ([README.md](README.md)). The Random Hierarchy Model
(Cagnetta & Wyart) offered the opposite: a controllable hierarchy depth `L`, per-level synonymic
multiplicity `m`, and exact ground truth at every position — so DGP interventions and channel
interventions could finally be told apart.

## Foundations: scaling exponents and a residual-rank instrument that wasn't what it seemed (06-20/21)

The opening sweep found `m` dominates the scaling exponent over `L` by roughly 3:1 — the
bottleneck is per-level ambiguity, not raw depth ([SWEEP_README.md](SWEEP_README.md)). A
forward-model (FM) residual-rank instrument was then built to test whether DGP complexity shows up
in the FM's leftover error — and immediately produced a trap that wouldn't be understood for over
a month: rank tracked *learning quality* and *FM capacity*, not DGP complexity, with a 17×
norm range under matched rank as the tell ([RESIDUAL_RANK_README.md](RESIDUAL_RANK_README.md)).
The regime-transition and per-level-loss work salvaged the idea by making it *dynamic*: composition
is learned bottom-up, and as the model masters shallow levels the FM's errors become progressively
conditioned on hierarchical feature identity (the "L-to-m transition") — this became the default
model-size regime for everything after ([REGIME_TRANSITION_README.md](REGIME_TRANSITION_README.md),
[PER_LEVEL_LOSS_README.md](PER_LEVEL_LOSS_README.md)).

## Objective-shape probes: the ceiling is capacity, not the loss's sharpening bias (06-22/23)

Label smoothing, focal loss, confidence thresholding, and FM-surprise-weighted NTP all tested
whether cross-entropy's sharpening incentive *causes* the composition-depth ceiling. It doesn't —
smoothing hurts uniformly, and the ceiling holds regardless of smoothing intensity
([LABEL_SMOOTHING_README.md](LABEL_SMOOTHING_README.md)). But gradient-reallocation methods
(focal loss, thresholding, FM-surprise weighting) reveal a *separable* payoff: they leave
compositional learning intact while making the model's computation more predictable to a
compressed self-model, up to FM cosine 0.995 ([LOSS_WEIGHTING_README.md](LOSS_WEIGHTING_README.md)).
This is the first appearance of **legibility** as an axis independent from raw task performance —
a distinction the rest of the node's history keeps re-deriving.

## The ratchet arc: an elimination tournament that ends on a moving-frontier requirement (06-23–06-29)

An MNIST "gated ratchet" (self-model-driven compounding across wake/sleep cycles, 34%→48% over 4
cycles) is ported to RHM and immediately reproduces every *dynamic* — gate closing, FM tracking,
crossover timing — but never the *magnitude* (~1% gap, no compounding). Rather than stopping there,
the arc runs a disciplined tournament ruling out one explanation at a time: excess capacity
([RHM_L4_RATCHET_README.md](ratchet/RHM_L4_RATCHET_README.md)), supervision density and task
richness via RL ([RHM_SPARSE_RATCHET_README.md](ratchet/RHM_SPARSE_RATCHET_README.md),
[RHM_RL_RATCHET_README.md](ratchet/RHM_RL_RATCHET_README.md)), the distillation mechanism
([RHM_RL_GEN_DISTILL_EXTENDED_README.md](ratchet/RHM_RL_GEN_DISTILL_EXTENDED_README.md)), and
finally the meta-learning objective itself via true bilevel FOMAML and Reptile across varied rule
sets ([RHM_FOMAML_README.md](ratchet/RHM_FOMAML_README.md),
[RHM_META_LEARNING_README.md](ratchet/RHM_META_LEARNING_README.md)). The crystallized belief:
**compounding requires a moving frontier** — on stationary data, no outer objective makes
FM-predictability pressure improve compositional NTP, because the local loss is either redundant
(dense NTP already covers it) or uninformative for levels not yet learned
([ratchet/README.md](ratchet/README.md)). This single negative result reorients the rest of the
node's history toward non-stationary and active tasks.

## The frontier/legibility vocabulary hardens, and self-knowledge earns its first real payoff (06-29–07-01)

Composition depth turns out to be capped by synonymy/occupancy, not capacity — a >200× parameter
range hits the same ceiling ([PER_LEVEL_EXPANSION_README.md](PER_LEVEL_EXPANSION_README.md)), and
deep structure is representable and even linearly recoverable but not *learnable* by any local
self-supervised objective — only a privileged oracle-latent target reaches it
([RHM_DEEP_COMPOSITION_README.md](RHM_DEEP_COMPOSITION_README.md)). This sharpens into the
"frontier" (how deep training actually reaches) and "legibility" (whether the FM residual is
structured exactly there) vocabulary: `m` gates the learnable frontier, and the residual is an
**amplifier of the frontier, not a source of new information about it**
([RHM_FRONTIER_AND_LEGIBILITY_README.md](RHM_FRONTIER_AND_LEGIBILITY_README.md)). FM-as-regularizer
then answers the ratchet arc's implicit question directly: structured FM-predictability pressure
compresses functional complexity below weight decay's floor, but an **open-loop** pressure term
suffices — self-knowledge (a closed loop) is not load-bearing for functional simplification
([RHM_FM_REGULARIZER_README.md](RHM_FM_REGULARIZER_README.md)).

## Closing the loop actually moves the frontier — but only with the right target (07-04–07-08)

[RHM_LATENT_LOOP_README.md](RHM_LATENT_LOOP_README.md) is the node's first genuine positive on
self-knowledge: closing the self-model loop produces generalizable improvement, but *only* under a
latent (own-lifted-DGP) target, never a token target at any depth — self-knowledge here is
*meta*-knowledge (a map of where the frontier's errors are), not object-level transfer.
[RHM_COMPLEXODYNAMICS_README.md](RHM_COMPLEXODYNAMICS_README.md) reads the saved trajectories
against Aaronson's "First Law of Complexodynamics" and finds the rise-then-fall of sophistication
is real but needs two amendments: the descent requires an annealer (SGD alone freezes mid-descent),
and it doesn't fall to zero but to the DGP's own sophistication — a hump that is trajectory-owned
scaffolding versus a floor that is problem-owned complexity, the same decomposition the latent loop
needed to explain its own success.

## From autonomous to controlled: the active-inference pivot (07-10–07-15)

[ACTIVE_RHM_README.md](ACTIVE_RHM_README.md) names the distinction the whole later program turns
on: RHM as usually run is an **autonomous** task (predict a given sequence); the sibling reaching
work in `a2a_forward` is **controlled** (choose actions). Porting the controlled apparatus back
shows the *arity* impossibility ports cleanly (mean-Δ forward models can't represent
query-conditional structure) but *usability* does not — planning needs a value-of-information head
predicting the decision variable itself, not the mean transition, and internalizing the forward
model turns out to have **no headroom** here because active-query RHM is inference in disguise
(`act ≈ plan`). That "no headroom" result becomes the reason to go looking for a task where acting
and planning genuinely differ. [RHM_EDIT_CONTROL_README.md](RHM_EDIT_CONTROL_README.md) finds one —
editing is plannable in belief space, reversing the query null — but surfaces a *second* obstacle:
the belief is gameable off-manifold (~99% off-grammar edits despite high confidence), fixed only by
constraining moves to a learned generator plus a cerebellar self-consistency veto (0.006→0.65 true
control). [`sculpting_control_task.md`](sculpting_control_task.md) then proves the resulting task
has a genuine, depth-scaling lookahead prize that myopic reflexes cannot capture, licensing the full
sculpting arc: [RHM_SCULPTING_README.md](RHM_SCULPTING_README.md) shows latent-space planning is
merely a cheaper surrogate on a clean channel (~92% of token performance, ~8× lower cost) but
genuinely *beats* token-space once the channel is lossy (partial observability, stochastic
dynamics) or once the belief is grounded by internalizing the plan loop itself.

## Does planning compound? The same "moving frontier" law, rediscovered (07-13)

[RHM_SCULPT_CONTINUAL_README.md](RHM_SCULPT_CONTINUAL_README.md) iterates the sculpting loop to
ask whether planning is a ratchet on its own substrate. It is a **weak** one: value-iteration is
the large compounding lever (+0.25), one-shot belief internalization is a large ceiling-setter
(+0.15), but continuous re-internalization adds only +0.02 because the representational frontier
exhausts in a single pass. This independently reconfirms, on a completely different mechanism, the
ratchet arc's June conclusion two months earlier — compounding needs a moving target, and a fixed
task's frontier is captured once, not incrementally.

## A reflexive turn: auditing the node's own instruments (07-22–08-02)

[residual_decomposition/README.md](residual_decomposition/README.md) goes back to the very first
instrument built in this node and asks what residual rank was actually measuring. The answer —
`res_var(i) ∝ act_var(i)^β` (R²=0.95–0.99), with a capacity-invariant *shape* (β) decoupled from a
capacity-dependent *level* (frontier mass) — retroactively explains three separate historical
"rank negatives" (this node's own Exp. 3, a language-model residual, and an `mjc` contact-dynamics
result) as instances of a saturated FM's noise floor, and falsifies the `R_act ≈ R_comp + R_res`
partition the residual-rank framing had implicitly assumed. The
[trajectory child](residual_decomposition/trajectory/README.md) reruns this over training
checkpoints and finds the *directions* replicate 7/7 but the absolute β scalar does not hold outside
its validated regime — the instrument earns trust for contrasts, not for standalone numbers.
[confabulation/README.md](confabulation/README.md) runs the same reflexive question on self-reports:
a report grounded in the FM residual shows a genuine first-person advantage over any capacity-matched
third party (survives a 24× capacity sweep), while reports about behavior or world-state show none —
but the pre-registered guess that closing the loop is *necessary* for this advantage is falsified;
it only amplifies an already-present dissociation.

## Self-knowledge is real but doesn't leverage: specialization, minting, endogenous teaching (07-16–08-05)

Three independent attempts to convert self-access into a learning advantage each come up empty, for
structurally similar reasons. [specialization/README.md](specialization/README.md) finds
breadth-restricted training is inert and level-reweighting is actively harmful; even concentrating a
*direct* deep target onto one subtree hurts that subtree relative to broad training
(Student-B > Student-A) — RHM's single shared ruleset has no domain-specific deep structure to
specialize into, the node's first clean RHM-vs-language disanalogy, later reframed as evidence that
shared abstraction is what buys sample efficiency rather than a specialization failure.
[minting/README.md](minting/README.md) builds on that reframe: filtering self-minted sequences by
the true rule table expands the training distribution's support but only reaches break-even with a
no-minting control — a rule table is a "support oracle," not a "density oracle," and NTP is density
matching, so it can delete junk but cannot manufacture the missing statistical information real data
would supply. [endogenous_teacher/README.md](endogenous_teacher/README.md) finds the model's FM
residual and token surprisal are 89% orthogonal and *anti*-localized (residual tracks computational
load, surprisal tracks epistemic difficulty) — but weighting the loss by residual magnitude is a
clean null, consistent with a belief already crystallizing elsewhere in the repo that self-knowledge
is *directional*, not *scalar*. Its [cancellation child](endogenous_teacher/cancellation/README.md)
reroutes the residual through the forward path instead and replicates the mechanism but not its
downstream payoff — computation depends on having *a* forecast, not *this* forecaster.

## Directed sculpting: porting `mjc`'s control loop, and instrument repair as a discipline (07-28–08-05)

[directed_sculpting/README.md](directed_sculpting/README.md) ports `mjc`'s E3 inner/outer control
loop onto RHM and opens by discovering the naive port is vacuous by construction — RHM's rule usage
is uniform, so there is nothing for an attention/relevance signal to allocate over — requiring new
primitives (distractor channels, support-fixed drift, a repair-cost instrument) before the loop
could test anything. What follows is the node's most concentrated run of self-correction: raw
cross-entropy turns out to be *exactly* blind to the drift being measured; a reducibility instrument
was initially inverted; a participation-ratio "frontier" metric proved anti-informative; and an
apparent 5× level preference in [`level_moves/`](directed_sculpting/full_loop/level_moves/README.md)
was traced to a branching-factor confound before a matched-span control revealed a genuine, but far
more modest, graded abstraction preference. Nearly every "interesting" positive across this arc was
first produced by an instrument carrying a hidden bias toward that positive, and each repair moved
the result toward null or toward a narrower, more defensible claim — the same lesson `mjc`'s history
names independently. What survived repeated repair: a reward-free relevance signal ports cleanly and
strongly (13–18× separation from distractors, 76% of an oracle's ceiling,
[full_loop/README.md](directed_sculpting/full_loop/README.md)), expansion is a property of a
grader's *type* rather than of non-stationarity, and hierarchy levels belong in the *action space*
rather than the *allocation space*
([level_ladder/README.md](directed_sculpting/full_loop/level_moves/level_ladder/README.md)).

## Standing epistemic patterns worth naming

- **An instrument's first exciting result is provisional until re-examined.** Residual rank (06-20),
  the reducibility tap and PR-as-frontier metric in directed_sculpting, and the level-preference
  confound were all published, trusted, built on for weeks, and only later found to be measuring
  something narrower or different than believed — never by discarding the original data, always by
  re-deriving what it actually meant.
- **"Compounding requires a moving frontier" recurs across unrelated substrates.** The ratchet arc's
  June conclusion (stationary RHM can't compound) is independently rediscovered in July by the
  sculpting-continual result (planning's own frontier exhausts in one internalization pass) — the
  same law, found twice, by two different methods.
- **Self-knowledge is directional, not scalar, and doesn't manufacture missing information.**
  Confabulation and residual_decomposition find genuine, structured self-access; endogenous_teacher,
  minting, and specialization each show that access cannot be turned into leverage in its naive
  (scalar-weighting, self-generated-data, targeted-training) form — the self-signal reports *how
  much* was computed, not *what* is still missing.
- **A passive/autonomous framing quietly caps what self-models can show.** The active-RHM,
  edit-control, and sculpting arcs exist because `act ≈ plan` on the original query task left no
  headroom for internalization to matter — the node had to construct a genuine control task before
  the reaching arc's positive result had anywhere to port to.
