# History of `experiments/rhm`

*A big-picture account of the thinking, not the day-to-day. See linked files for numbers and commands.*

## Why this experiment exists

RHM was adopted as a controlled substrate after natural-language work
([`language_reduction/STATUS.md`](../language_reduction/STATUS.md)) hit a wall: vocab reduction
turned out to be a channel intervention (β not γ), and every attempt to change language's
*data-generating process* was entangled with the statistics used to measure it. The Random Hierarchy
Model (Cagnetta & Wyart) offered what language couldn't: a generative process with a knob for depth
(`L`), per-level ambiguity (`m`), and a known ground truth any decomposition could be checked exactly
against. The founding bet was that exact ground truth would let the team tell apart questions
language conflates — see [`README.md`](README.md) for the per-experiment index and
[`FILES.md`](FILES.md) for the file-level map.

## Phase 1 — scaling laws, and the first instrument crisis (through 2026-06-21)

The opening sweep ([`SWEEP_README.md`](SWEEP_README.md)) found `m` (synonymic multiplicity) dominates
the scaling exponent over `L` (depth) roughly 3:1 — per-level entropy, not raw depth, is the scaling
bottleneck. The team then tried to read a forward model's (FM) residual rank as a proxy for "how much
DGP complexity the model has internalized" ([`RESIDUAL_RANK_README.md`](RESIDUAL_RANK_README.md)) and
immediately hit the pattern
that would recur for the rest of the project's life: **the obvious scalar instrument was measuring
something else.** Rank tracked learning quality, not DGP complexity, and also tracked head-count
mismatch between model and FM — an architectural artifact unrelated to computation. This first
crisis produced the project's standing habit: build a metric, adversarially ask what confound could
produce the same number, then build the matched-architecture control.
[`REGIME_TRANSITION_README.md`](REGIME_TRANSITION_README.md) and
[`PER_LEVEL_LOSS_README.md`](PER_LEVEL_LOSS_README.md) then established the mechanistic picture that
holds for the rest of the project: learning is bottom-up (level 0 first), and depth composition
saturates a small model within 1–2 levels regardless of NTP objective shape — label smoothing, focal
loss, and confidence thresholding ([`LABEL_SMOOTHING_README.md`](LABEL_SMOOTHING_README.md),
[`LOSS_WEIGHTING_README.md`](LOSS_WEIGHTING_README.md)) can reallocate *which* levels get gradient but
cannot lift the composition ceiling itself — the ceiling is a capacity/signal constraint, not an
objective-induced sharpness bias.

## Phase 2 — what's learnable is not what's recoverable (2026-06-30)

[`RHM_FRONTIER_AND_LEGIBILITY_README.md`](RHM_FRONTIER_AND_LEGIBILITY_README.md) sharpened this into
a trichotomy that organizes everything downstream: a feature can be **recoverable** (occupancy sets
an information ceiling), **representable** (capacity + the right signal reaches it — oracle-aux
proves this), or **learnable by plain NTP** (which turns out to be gated by `m` alone, independent of
occupancy — two DGPs with identical recoverability ceilings can have opposite learned depth). On the
one substrate where NTP does reach the root (m=2), the FM residual becomes hierarchy-legible exactly
where and when the base model is learning each level — the first direct evidence for **"the
self-model amplifies structure the base objective already extracted; it does not manufacture
structure."** This amplifier-not-source hypothesis explained, retroactively, why every earlier
closed-loop attempt (all run at m≥4) had seen a diffuse, illegible residual — there was nothing yet
to amplify.

## Phase 3 — the ratchet's negative, and the concept of a moving frontier (2026-06-23→29)

The MNIST wake-sleep ratchet compounds (34%→48% over 4 cycles); the whole point of porting it to RHM
([`ratchet/README.md`](ratchet/README.md)) was to ask why, with exact ground truth to watch. The arc
reproduced every *dynamic* of the MNIST ratchet (gate closing, FM tracking, the robustness
dissociation) but never the *magnitude* — and methodically ruled out capacity, task structure (RL),
the distillation bottleneck, and FM-cosine regime as the cause, catching and correcting its own
instrument confounds along the way (an oversized FM had been silently saturating cosine in Run 1; a
REINFORCE-based outer loop's stuck gate was diagnosed as meta-gradient variance, not evidence, and
re-run deterministically to confirm). The verdict that survived: **on stationary data, no outer
objective makes FM-predictability pressure improve compositional NTP, because the outer loop ends up
identical to the inner loop.** Compounding needs a *moving* frontier — MNIST's root-level supervision
is an implicit curriculum (strokes → parts → classes) that keeps refilling the residual from above;
stationary RHM has no such refill and every ratchet variant exhausts after ~1 cycle. This
"moving-frontier" idea — first named here as a negative explanation — becomes the organizing concept
for two later positive threads: the latent-loop line (make the *target* move up the tree) and
directed sculpting (manufacture drift in the *data* itself).

## Phase 4 — self-knowledge finally appears, but only under a latent target (2026-07-04→08)

[`RHM_LATENT_LOOP_README.md`](RHM_LATENT_LOOP_README.md) is the hinge document for everything that
follows. It found that closing the cerebellar loop produces genuine, generalizable self-knowledge on
RHM for the first time — but *only* when the model is trained toward its own lifted latent structure,
never toward tokens at any depth or loop strength. A pre-registered prediction ("a deep residual
should be sufficient") was explicitly tested and **falsified** by an m=2 control: token-NTP there
already reaches the root and has a deep-structured residual, yet self-knowledge still fails to
generalize. A follow-on decomposition (meta vs. object-level, ported from the MNIST local-loss
probes) found the whole effect lives in a *meta* channel ("where my self-model errs"), and a
distillation ablation found that on RHM's compact, rule-derivable grammar **sleep/distillation
transfers nothing** — RHM's shared rule table makes 5% of positions sufficient to recover the whole
grammar, so distillation is solving an already-solved problem. This "residual = frontier map" framing
becomes the lens for reinterpreting three historical "high-rank residual" negatives —
[`residual_decomposition/README.md`](residual_decomposition/README.md) (2026-07-27) shows all three
were reading a *saturated FM's noise floor*, and replaces naive rank with two properly-typed numbers
(β and `R_res_participation`), falsifying the `R_act ≈ R_comp + R_res` partition in
[`beliefs/dimensionality_expansion.md`](../../beliefs/dimensionality_expansion.md).

## Phase 5 — self-knowledge is separable from function; complexity rises then hits a floor, not zero

[`RHM_FM_REGULARIZER_README.md`](RHM_FM_REGULARIZER_README.md) (2026-07-01) showed structured
FM-predictability pressure beats weight decay's functional-complexity floor at preserved knowledge:
"self-knowledge is *not* load-bearing for functional simplification."
[`RHM_COMPLEXODYNAMICS_README.md`](RHM_COMPLEXODYNAMICS_README.md) (2026-07-07) then read the saved
trajectories from that arc against Aaronson's "First Law of Complexodynamics" and amended it twice: the
rise-and-fall needs an annealer (SGD alone freezes mid-descent, a glass not a liquid), and the descent
lands on the **DGP's own sophistication relative to the bound**, not zero. [`confabulation/README.md`](confabulation/README.md)
(2026-07-22) then asked the question from the *report* side: a self-report about the residual shows a
first-person advantage no capacity-matched observer can match, a report about behavior is a dead null —
and, a second falsified pre-registration, the closed loop turns out **not necessary** for this
dissociation: closing the loop changes how well the model *knows* itself, not what there is to know.

## Phase 6 — autonomous vs. controlled, and manufacturing a real control task

[`ACTIVE_RHM_README.md`](ACTIVE_RHM_README.md) asked whether the a2a reaching-control machinery
(query-conditioned forward models, internalization) ports back onto RHM's native, autonomous
querying task. It doesn't: value-of-information (not mean-Δ) is the right forecast target for
epistemic actions, and internalization has **no headroom** on RHM's active-query task because acting
*already is* planning there (act ≈ plan) — no model-free shortcut splits the forecast from the
forecaster the way reaching's policy splits from its dynamics model. This motivated deliberately
turning RHM into a genuine control task with a real action/dynamics split —
[`RHM_EDIT_CONTROL_README.md`](RHM_EDIT_CONTROL_README.md) and
[`RHM_SCULPTING_README.md`](RHM_SCULPTING_README.md) (Stages 1–6) — where belief-space editing *is*
plannable, latent planning beats token planning once the observation channel is lossy, and
[`RHM_SCULPT_CONTINUAL_README.md`](RHM_SCULPT_CONTINUAL_README.md) finally gets a **weak ratchet**:
value-iteration is the big compounding lever, belief internalization a one-shot ceiling-setter, and
continuous re-internalization adds almost nothing once the frontier saturates in a single pass.

[`directed_sculpting/README.md`](directed_sculpting/README.md) then supplied the missing ingredient
for porting MJC's full control-loop machinery: RHM's rule usage is *uniform by construction*, so the
naive port's `value = learning-progress × visitation` collapses to `lprog` — a null by construction.
Three primitives fixed this (distractor channels for structural irrelevance, support-fixed rule drift
for non-stationarity without task-switching, and a repair-cost instrument built because raw
cross-entropy is *exactly* blind to this drift), each catching a real confound (an 11.9× per-level mis-scaling; a "static" control that was secretly a handicap). The full loop
([`directed_sculpting/full_loop/README.md`](directed_sculpting/full_loop/README.md)) found a
reward-free relevance signal is a smoking gun, and — the central pivot — that **expansion is a
property of the grader's type, not of non-stationarity itself**: an evaluative grader expands the
belief where an endogenous dense grader caps below the no-loop floor, in both static and drifting
worlds. A "climbing null" was re-scoped through repeated instrument repair (each repair removing a
bias favoring the positive reading) until starving samples-per-event at fixed drift produced the
arc's first real depth-vs-drift positive. This grader-type-matters finding and Phase 3's
moving-frontier idea converge directly into the practice arc below.

## Phase 7 — typing a token's "news" exactly (2026-08-04→17)

A parallel thread used RHM's exact Bayesian tractability to pull apart categories that are
inseparable in natural language. [`endogenous_teacher/README.md`](endogenous_teacher/README.md)
measured that 89% of the FM residual's variance is orthogonal to token surprisal and the two are
*anti-localized* across the hierarchy — but reweighting NTP by that residual is a clean null, scoped
to *scalar* weighting (self-knowledge is directional, not scalar, per two prior
[a2a](../a2a_forward/EMOTION_INJECTION_README.md) results). Its
[`cancellation/`](endogenous_teacher/cancellation/README.md) child found the corollary-discharge
mechanism replicates across substrates but its downstream payoffs don't — what matters is *a*
forecast existing, not *this* forecaster's content. [`conditional_revision/README.md`](conditional_revision/README.md)
then built the missing instrument itself: an exact belief-propagation oracle splitting surprisal into
*reducible* (taught new structure) and *irreducible* (a synonym of known structure) — the model's
belief revision tracks the reducible component only where it has a belief at all, capped by belief
depth, not by the measurement gap. [`question_model/README.md`](question_model/README.md) (2026-08-17)
closed the line with a third exact term for *what the world is currently asking* (a drifting demand
distribution): NTP tracks it only above an evidence-density threshold, and a "demand-keyed
self-knowledge" readout is a vacuous null on an unmetered generalist — the premise, not the organ, is
absent. In parallel, [`specialization/README.md`](specialization/README.md) found the first real
RHM-vs-language disanalogy: restricting training to a subtree doesn't buy depth (inert), and
concentrating even a *direct* deep target on a subtree actively hurts it — broad beats narrow, because
RHM's shared rule set has no domain-specific deep structure to specialize into — motivating
heterogeneous-DGP variants and the self-training work in [`minting/README.md`](minting/README.md).

## Phase 8 — the practice arc: instruments correcting themselves in real time (2026-08-14→21, ongoing)

The newest and now-dominant thread ([`practice/README.md`](practice/README.md)) ports "practice" — a
control loop over the *conditions and units* of learning, not the model itself — from
[`mjc/practice/etude/`](../mjc/practice/etude/README.md), whose conclusion had been blocked by a
degenerate substrate (state-independent committed units, so post-commit drift was exactly zero), which
sculpting removes. What stands out across its now eleven-plus rounds is not any single finding but the
rate of self-correction: the compile certificate is demoted three consecutive times, each round
finding a different reason it has nothing genuine to certify
([`crystallize/README.md`](practice/crystallize/README.md),
[`ear/README.md`](practice/ear/README.md)); [`recital/README.md`](practice/recital/README.md) finds no
internal pacing signal prices the ladder, and a fixed bottom-heavy schedule beats every adaptive one;
[`tall/README.md`](practice/tall/README.md) is voided outright by measurement mid-arc (m=4 declared
inadmissible for sculpting); and [`merge/`](practice/merge/README.md) /
[`fourwall/`](practice/fourwall/README.md) each catch and publish their own prior measurement errors
(a grid-inflated penalty, a self-grading artifact). The clearest positive result, the crossed pair
[`transpose/`](practice/transpose/FILES.md) × [`setlist/`](practice/setlist/FILES.md) jointly written
up in [`typed_gaps/README.md`](practice/typed_gaps/README.md), delivers a double dissociation:
drifting *what is true* degrades the learner and leaves verification silent; drifting *what is asked*
leaves the learner flat and fires verification decisively. **Committed chunks store
demand-concentration, not truth; maintaining a skill is demand-tracking, not entropic** — the arc's
"conditioning gap" becomes an explicit type system (one organ per currency of change), echoing
directed sculpting's grader-type lesson (Phase 6). The most recent rounds
([`fourwall/lm/`](practice/fourwall/lm/README.md), [`teacher_slot/README.md`](practice/teacher_slot/README.md),
[`native/README.md`](practice/native/README.md)) push further — what an endogenous reader can
self-correct unaided, what a verbal (LLM) reasoner adds over a numeric rule on identical evidence, and
whether earned vocabulary can consolidate into the agent's own planner/executor — live threads, not
settled ones, as of this writing.

## The throughline

Read end to end, the arc is less a sequence of findings than a sequence of **instrument
corrections**, each dissolving an apparent property of the model into a property of the measurement:
residual rank was a noise floor, not complexity; a deep residual was not sufficient for
self-knowledge without the right target; weight-norm compression was not circuit compression; token
surprisal was not belief revision; committed practice chunks were not beliefs. The recurring
constructive move surviving this scrutiny is the **moving frontier** — first named as why the ratchet
fails on stationary RHM (Phase 3), then engineered via a latent training target (Phase 4), then via
manufactured DGP drift (Phase 6), and now the organizing variable of the practice arc's
demand-vs-truth typing (Phase 8). The project's discipline has visibly tightened: later arcs
pre-register falsifiable predictions and treat their failure as informative (the m2 control,
confabulation's loop-not-necessary, the endogenous-teacher scalar null), and documented, real-time
self-auditing of one's own metric is now the default mode rather than an occasional correction.
