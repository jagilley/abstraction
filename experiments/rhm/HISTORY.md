# HISTORY: experiments/rhm

*A big-picture pass over ~9 weeks (2026-06-20 → 2026-08-23) of work using the Random Hierarchy
Model as a controlled substrate. This is a synthesis of shifts in thinking, not a changelog —
see [`README.md`](README.md) and [`FILES.md`](FILES.md) for the full experiment-by-experiment
record every claim below is drawn from.*

## Why RHM at all

The line starts as a patch for a specific failure: natural-language experiments
([`experiments/language_reduction/STATUS.md`](../language_reduction/STATUS.md)) couldn't cleanly
separate a DGP intervention from a channel intervention — vocab reduction and spectral denoising
kept being entangled with the statistics being measured. RHM was adopted because it makes that
separation for free: known ground truth, controllable depth `L`, controllable per-level ambiguity
`m`, and an exact hierarchy you can decompose loss and residuals against ([`README.md`](README.md)
§"Goal"). That founding motive — *buy exactness so a confound can't hide* — is the thread that
holds together everything that follows, including the eventual return to exact computation (BP
oracles) once correlational instruments started failing to settle questions on their own.

## The instrument had to be rebuilt twice before it could be trusted

The earliest work ([SWEEP_README.md](SWEEP_README.md), [RESIDUAL_RANK_README.md](RESIDUAL_RANK_README.md))
read forward-model residual **rank** as a proxy for how much unmodeled computation a model was
doing. That proxy quietly broke: architecture-matched sweeps showed residual norm could drop 17×
while rank barely moved ([README.md](README.md) §"FM residual rank experiments" Exp. 3) — rank read
the residual's *shape*, blind to its *magnitude*, and a saturated FM's noise floor could mimic real
structure in all three domains RHM/language/MNIST at once
([residual_decomposition/README.md](residual_decomposition/README.md)). The replacement — `res_var
∝ act_var^β` plus `R_res_participation` — falsified a standing belief doc
([`beliefs/dimensionality_expansion.md`](../../beliefs/dimensionality_expansion.md)'s `R_act ≈
R_comp + R_res` partition) rather than just refining it. The same demotion happened to **weight
norm**: a 150k-step WD sweep compressed weight norm 3.2× at zero knowledge cost while effective
rank barely moved (≤13%) — "simple DGP ⇒ simple circuit" died as a readable claim, and η² became
the trusted instrument (`RHM_FRONTIER_AND_LEGIBILITY_README.md` Exp. 3, `RHM_FM_REGULARIZER_README.md`).
By the `conditional_revision`/`endogenous_teacher` era (Aug), the team stopped trusting scalar
readouts almost on principle: **the same finding recurs at least four separate times** that an
effect is *directional, not scalar* — a residual-norm event carries much less signal than the same
event read in belief coordinates (`conditional_revision/README.md`), self-knowledge from
corollary-discharge only shows up as a directional match not a magnitude one
(`endogenous_teacher/cancellation/README.md`), and both predicted a scalar-weighting intervention
would fail before it was even run (`endogenous_teacher/README.md`, citing
[`a2a_forward/EMOTION_INJECTION_README.md`](../a2a_forward/EMOTION_INJECTION_README.md)).

## "The information is there" stopped being a satisfying answer early

[`RHM_DEEP_COMPOSITION_README.md`](RHM_DEEP_COMPOSITION_README.md) (2026-06-29) is the arc's most
consequential trichotomy, and it recurs, restated, for the rest of the project's life:
**recoverable** (an exact BP ceiling exists), **representable** (a vanilla transformer given an
oracle auxiliary loss reaches that ceiling), and **learnable** (no local self-supervised objective
— NTP, MLM, masked-span, contrastive — reaches it at m≥4, because the *supervised* signal for deep
structure is diluted by `m^(L-d)` while a *local* sibling-context signal is depth-flat). This
crystallized into a belief doc
([`beliefs/trees/rhm_compositional_learnability.md`](../../beliefs/trees/rhm_compositional_learnability.md))
holding that **m, not occupancy, is the learnability control** — a correction of an earlier
"occupancy law" reading of the original scaling sweep (m dominates scaling ~3:1 was previously read
as an information-ceiling effect; it is partly a learnability effect on top of that). The same
recoverable/representable/learnable move reappears reframed as **act ≈ plan** in
[`ACTIVE_RHM_README.md`](ACTIVE_RHM_README.md) (a belief can be observation-complete without being
plannable *unless* value-of-information, not the mean update, is the training target), as
**belief-depth vs the conditioning gap** in [`conditional_revision/README.md`](conditional_revision/README.md)
(revision is separable from surprisal only where the model *has* a belief), and most recently and
most generally in [`rl_dimensionality/README.md`](rl_dimensionality/README.md): reward optimization
never creates structure the pretrained basis lacks, at any of 3 RL mechanisms × 5 dimensionalities.
Nine weeks of work keep re-deriving one law under different names: **a signal being present in the
data is not the same fact as an objective being able to reach it, and no amount of optimization
pressure substitutes for the objective actually targeting the right thing.**

## The ratchet's negative reframed the whole second half of the project

The wake-sleep/RL/FOMAML "ratchet" arc (2026-06-23→06-29, [`ratchet/README.md`](ratchet/README.md))
asked whether an MNIST-observed compounding effect (a gated ratchet improving 34%→48% over 4
cycles) also compounds on RHM. It doesn't, anywhere the team looked — capacity, task structure, the
distillation bottleneck, and FM-cosine regime were each ruled out in turn, and the FOMAML
diagnostic delivered the structural verdict: on **stationary** data, no outer objective makes
FM-predictability pressure improve compositional NTP, because the outer loop collapses into the
inner one. The retrospective the arc wrote about itself is sharper than the headline: **supervision
density**, not architecture or DGP, is the variable that actually explains MNIST vs RHM (sparse
single-position supervision genuinely benefits from FM compression; dense NTP already tells the
model what to do everywhere) — a monotonic crossover confirmed directly by sweeping mask density
(`ratchet/RHM_SPARSITY_SWEEP_README.md`). But the more load-bearing conclusion for what came next
is that **compounding requires a moving frontier**, which stationary RHM structurally cannot
supply. That single sentence is the hinge the rest of the project pivots on: the active-query,
sculpting, directed-sculpting, and practice arcs are all, in one way or another, RHM instrumented
to *have* a moving frontier (drift, distractor channels, an editable target, an earnable action
vocabulary) — see [`ratchet/README.md`](ratchet/README.md)'s own "open threads" for the residue that
never got picked up (an unstabilized λ_local=0 regime, and representational deepening without
behavioral crossover — the "self-model-driven grokking" idea still sitting unclaimed).

## Autonomous vs controlled — the distinction that reorganized the last third of the arc

[`ACTIVE_RHM_README.md`](ACTIVE_RHM_README.md) (2026-07-10/11) found that RHM's natural mode —
passively predicting, or choosing which masked block to reveal — is an **autonomous / epistemic**
problem where acting and planning are the same computation (a value-of-information head plans
exactly by doing what a competent actor already does). This explains, in hindsight, why the
sibling `a2a_forward` reaching arc's internalization result (forecasting reorganizes a policy) never
had anything to port to on RHM: internalization only has purchase when acting and forecasting are
genuinely different computations a model-free learner can shortcut between — true of motor control,
false of query selection. To even *see* the reaching-style phenomenon, RHM had to become a real
control task. The sculpting arc (`RHM_SCULPTING_README.md`, `RHM_EDIT_CONTROL_README.md`, Stages
1–6, 2026-07-11 onward) built that by switching from revealing to *editing* toward a target — and
immediately found a second, orthogonal obstacle nobody predicted: a belief that is a perfect parser
on-manifold is trivially gameable off it (P(reward)→1 while true success stayed at 0.006), fixed
only by constraining actions to a learned generator plus a self-consistency veto. A Stage-2 planning
collapse was then diagnosed, expensively, as **the task's fault, not the method's** — the corrupted
target had no lookahead requirement at all — which produced a standing lesson the arc names
explicitly: build (and verify, via a perfect-simulator DP pre-check,
[`sculpting_control_task.md`](sculpting_control_task.md)) a task that actually requires coordination
*before* trusting a null about the planner. Once that was fixed, the arc's answer to "does latent
planning beat token planning" kept moving and never simplified to one story: on a clean channel
latent planning is a cheaper surrogate, not a better one (92% of token-beam quality at ~8× lower
cost); it only *wins* once the token channel stops being a sufficient statistic (partial
observability, stochastic dynamics); and once the belief itself is reshaped by a grounded
(not endogenous) planning loop, the win reappears even on the clean channel. Planning itself turned
out to be at best a **weak ratchet** on its own substrate — value-iteration compounds, belief
internalization is a one-shot ceiling-setter, and re-internalizing every round adds almost nothing
([`RHM_SCULPT_CONTINUAL_README.md`](RHM_SCULPT_CONTINUAL_README.md)) — a smaller-scale replay of the
ratchet arc's own stationary-frontier lesson, this time inside a task built specifically to have a
frontier.

## Directed sculpting: injecting the very things RHM lacked by construction

[`directed_sculpting/README.md`](directed_sculpting/README.md) (2026-07-28) found the naive reuse of
the sculpting substrate for a2a_forward's mjc-style "relevance" experiment was a confounded null by
construction — RHM rule usage is uniform, so nothing like `visits` or non-stationarity exists without
deliberately building it in (distractor channels for relevance, support-fixed rule drift for
non-stationarity; a repair-cost instrument with two documented traps — raw CE is *exactly* blind to
this drift, and the naive fix is confounded by ordinary continued training). Once wired up, the
reward-free relevance signal ported cleanly (13–18× separation of real from distractor tokens), but
belief **expansion** turned out to be a property of the grader's *type*, not of non-stationarity per
se — obtained by repeatedly *repairing its own instrument* until a positive that looked like a
drift-compensation artifact survived a harder control (matched samples-per-event, not just magnitude).

## The passive/autonomous side kept asking who — or what — can teach

In parallel with the control-task line, a second thread stayed on the passive/autonomous side and
asked what a model's own internal signals are good for. [`endogenous_teacher/README.md`](endogenous_teacher/README.md)
(2026-08-04/05) showed the FM residual and token surprisal are mostly orthogonal and *anti-localized*
across the hierarchy (residual peaks where text is nearly free) — the model's own surprise really is
a distinct signal from the text's — but a scalar reweighting of NTP loss by that signal is a clean
null, scoped explicitly to *scalar* interventions by the directional-not-scalar pattern above.
[`conditional_revision/README.md`](conditional_revision/README.md) (2026-08-07/08) then supplied the
measurement `endogenous_teacher` needed but didn't have: an **exact**, BP-computable split of a
token's surprisal into reducible (belief revision) and irreducible (synonym noise) parts. Belief
revision beats surprisal, but only at the levels (d2/d3) where the model actually holds a belief —
above that it's near-total surprisal, below that it's untestable. [`question_model/README.md`](question_model/README.md)
(2026-08-14ish) extends the identity by a third, exactly-computable term (question-news — what a
drifting demand distribution is currently asking) and finds NTP only builds a question model above
an evidence-density threshold. The throughline across this whole sub-thread, stated most crisply in
[`practice/typed_gaps/README.md`](practice/typed_gaps/README.md) (2026-08-16): **the conditioning gap
is typed** — drifting *what is true* and drifting *what is asked* are handled by different organs
(dense learning vs. evaluative re-selection), and neither substitutes for the other.

## The practice arc: importing a method rather than a result

[`practice/README.md`](practice/README.md) (2026-08-14 onward) is a direct transplant of the
"étude" compile-op from [`mjc/practice/etude/`](../mjc/practice/etude/README.md) — itself evidence
that the monorepo's practice concept has become portable machinery, not a one-off MuJoCo result.
State-conditioned commitment beats state-independent commitment 1.8–3.0×
([`practice/crystallize/README.md`](practice/crystallize/README.md)), but the certificate that gates
commitment turns out to be **vacuous on a frozen plant** — with only the selector learning,
committing immediately matches every gated arm at 26× less priced time. Round 2
([`practice/ratchet/README.md`](practice/ratchet/README.md) — explicitly *not* the same arc as the
top-level meta-learning [`ratchet/`](ratchet/README.md)) gives practice something to move (its own
mined action vocabulary) and finds earning it recovers 68–98% of a given vocabulary's value, with a
striking asymmetry: committing one level early is *worse than never committing*, because the mined
tables are nested and a bad compile forecloses the next level's representation, not just its
accuracy. Rounds 3–6 chase what internal signal, if any, can price *when* to stop practicing at a
given level, and find — across three independently-seeded worlds — that **no internal pacing signal
does**; only a fixed, bottom-heavy external schedule is robust
([`practice/recital/README.md`](practice/recital/README.md)). The newest, still-unindexed extension,
[`practice/teacher_slot/README.md`](practice/teacher_slot/README.md) (2026-08-20), decomposes "every
ladder tops out at a teacher" (the arc's own prior conclusion) one step further: a thermostat-grade
rule suffices to hold the right decision *if it is plumbed to a gauge outside the within-level
ledger* — and, strikingly, a **verbal LLM reasoner supplies that gauge from priors alone**, even
pre-rotation with no numeric evidence favoring it, because (the team's stated reading) text is
"post-ratchet": a corpus produced by minds that already learned when to stop clinging to a crutch.

## Self-report, and where the arc currently stands

[`confabulation/README.md`](confabulation/README.md) (2026-07-22) asks a question orthogonal to all
of the above: does a self-report track the residual (implementation) or a recoverable theory of it?
A report head shows a first-person advantage specifically on the FM-residual target and nowhere
else (behavior is a dead null; entropy/world-state reports are actually *worse* than a third-party
observer) — and, in a pre-registered expectation that failed, **closing the report loop is not
necessary** for the effect, only amplifying. Most recently, [`rl_dimensionality/README.md`](rl_dimensionality/README.md)
(2026-08-23) returns to the project's oldest question — what does reward optimization actually buy —
with the full apparatus built in between (exact bounds vs. attainment, verifier-type as an
independent axis) and lands on the cleanest statement yet of the recurring law above: pass rates can
rise while the underlying rules stay unlearned, visible only because RHM's ground truth makes the
gap measurable at all.

## Loose ends worth flagging

Two substantial, load-bearing writeups — [`RHM_DEEP_COMPOSITION_README.md`](RHM_DEEP_COMPOSITION_README.md)
(the recoverable/representable/learnable trichotomy nearly everything above depends on) and
[`SLEEP_CHUNKING_RHM_README.md`](SLEEP_CHUNKING_RHM_README.md) (chunking is an objective phenomenon —
external-role prediction, not reconstruction — a plausible substrate for the abstraction-ratchet
idea revisited later in `practice/ratchet/`) — are not linked from [`FILES.md`](FILES.md) or the
main [`README.md`](README.md) results timeline despite being cited by name from other auxiliary
READMEs. Worth reconnecting into the index next time either is touched.
