# HISTORY: experiments/mjc

*A big-picture pass over ~5 weeks (2026-07-16 → 2026-08-20) of work using MuJoCo as a controllable-dynamics
substrate one rung of realism above RHM and below language. This is a synthesis of shifts in thinking, not a
changelog — see [`README.md`](README.md) and [`FILES.md`](FILES.md) for the full node-by-node record every
claim below is drawn from.*

## Why MuJoCo, and the discipline that kept it from becoming a robotics project

The substrate was proposed in [`ideas/physical_control_substrate.md`](../../ideas/physical_control_substrate.md)
(2026-07-16) to test the program's control-oriented claims — arity, composition, latent planning,
operators-not-footprints robustness — on genuinely continuous dynamics with real contact, while keeping the
variable-control discipline that makes RHM's toy results trustworthy. The founding fork, restated at the top
of [`README.md`](README.md) ever since, is **controllable-dynamics DGP, not RL-to-SOTA**: physics parameters
are knobs to sweep, policies are obtained the cheap way (scripted, MPC, short imitation), and every experiment
reports a slope or dissociation, never a single benchmark number. This discipline is why the arc could run for
five weeks without ever tuning PPO, and it is the reason later corrections (below) read as *measurement*
fixes rather than chasing a leaderboard.

## The three foundational cuts set the substrate's vocabulary

Cut #1 ([`contact_residual/`](contact_residual/README.md)) found that a forward model's residual concentrates
at contact — not just in magnitude (8.1×) but directionally (cosine 0.99 free vs 0.82 contact) — and, sharpened
by an onset-aligned average, is an *event* detector (spikes at the impact transition, decays while contact
persists) rather than a state detector. Cut #2 ([`arity_torque/`](arity_torque/README.md)) killed a specific
confound — commands correlated with state under the default collection scheme — and showed that under i.i.d.
commands, arity beats resolution outright: the smallest command-aware model beats the largest command-blind
one. Cut #3 ([`dynamics_shift/`](dynamics_shift/README.md)) is the node most later work builds on: intervening
on the transition operator alone (not the task) lets a model-based agent recover from a dynamics shift in
~50 reward-free transitions, against a committing model-free baseline needing ~240× more reward-labeled data —
gated by re-groundability, peaking at the FM's ~6–8-step composition horizon. That gating parameter,
`replan_every`, read biologically as "how ballistic a movement is," seeded the entire ballistic line four
arcs later.

## The value↔FM interface arc: four attempts before the obstruction became informative

[`value_shaping/`](value_shaping/README.md) established the stationary control — a value signal re-allocates a
capacity-limited FM away from a reducible-but-irrelevant force field — before anything moved. The first
non-stationary attempt, [`directed_readapt/`](directed_readapt/README.md), produced two informative negatives:
a global shift gives no scarcity to exploit, and a localized shift has scarcity but the disagreement drive is
**blind to a confident-prior region** — it doesn't know where it's wrong. That negative motivated a pivot to
phenomenon-first work: [`meta_adapt/`](meta_adapt/README.md)'s Cut #4→#4e arc ran five variations in sequence
and each corrected the last. #4 found a 1-D damping shift collapses to no meta-benefit (the RHM anchor) while
an actuator-rotation *conflict* opens the gap monotonically — the first sign that what matters is not
non-stationarity per se but a specific geometric property of the shift. #4b showed a context latent decodes the
task parameter essentially perfectly, and — the first real surprise — **system-identification accuracy is
orthogonal to adaptation benefit**: knowing the parameter and using it well are different things. #4c is a
second robust negative in the same shape as `directed_readapt`'s: value-directed identification demonstrably
concentrates on the informative region but still doesn't beat a trivial heuristic. #4e closed the loop and
surfaced a wireheading gotcha (normalize the weight budget, or the outer loop games loss-scale instead of
allocation).

[`curiosity_control/`](curiosity_control/README.md) supplied the missing *afferent* half — a reducible-surprise
drive that tracks a moving frontier — and, in doing so, reproduced a known failure mode from a different part
of the program: pure curiosity on low-dimensional control **chases aleatoric noise** (a noisy-TV pathology,
read as a substrate inversion of the a2a active-vision result), fixed by grounding explore against exploit as
two additive drives. [`online_value_loop/`](online_value_loop/README.md) then tried to build the keystone the
original idea doc named — reward continuously shaping a *live* FM — and got obstructed on both the afferent and
efferent lever. The obstruction itself was the finding: value is a slow, committed, offline-estimable
quantity, so the two-timescale split is *forced by the phenomenon*, not a design choice, and its benefit is
adaptation speed, not converged competence. [`drift_value_loop/`](drift_value_loop/README.md) took that
diagnosis three cuts deeper under perpetual drift and landed on the arc's cleanest correction: what looked like
a value-shaping problem across its first two cuts was a **grader** problem — grading the meta-loop by control
outcome fails, but grading it by the value-relevant FM prediction-error (the literal cerebellum→VTA
messenger) gives the explore/exploit balance a clean interior optimum. *The missing piece was the teacher, not
the value structure* is the sentence [`README.md`](README.md) uses to close that cut, and it is the hinge the
next two arcs (below) both build directly on.

[`ballistic/`](ballistic/README.md) resolved a standing puzzle from earlier in this arc — why the FM→behavior
bridge kept failing to transmit under CEM-MPC — as a fact about *control mode*, not value: replanning is a
blind grader because it re-grounds past a stale model on every step. A feedforward, committed controller makes
the FM ~3× more behaviorally load-bearing, and reward-free re-adaptation restores ~4.3× more ballistic
competence than reactive after a drift. Its child, [`ballistic/directed/`](ballistic/directed/README.md), then
closed the directed-collection ladder that `directed_readapt` and #4c had each failed to close, with a
reward-free `learning-progress × visitation` signal recovering the oracle allocation — but the audit of that
result is what produced the next arc.

## Realism was a standing debt, and paying it retracted a headline motivation

[`arm_substrate/`](arm_substrate/README.md) (2026-07-22) moved the *plant* to a second task family and, in its
own caveats, named what it deliberately left untouched: "physics realism ≠ experience realism… inherits the
acquisition model unchanged." That acquisition model — every FM in the node trained on **teleported**
(`set_state`-based, discontinuous, omnisciently covering, free) transitions — was audited in
[`on_policy/COLLECTION_REALISM.md`](on_policy/COLLECTION_REALISM.md), which tiered every prior claim: safe
dissociations, inflated sample counts, and — the sharpest tier — questions about *where experience gets spent*
that were simply unaskable on teleported data. [`on_policy/`](on_policy/README.md) built the fix as a flag
(`collection_mode`, backed by a metered `Body` with no `set_state`) and pried apart what embodiment actually
costs. The result inverted the memo's own hypotheses twice: E0 found on-policy's apparent advantage was mostly
a **mistuned teleport knob**, not embodiment; E1 found the prediction that on-policy would stretch recovery was
refuted in the opposite direction — only the mistuned default *looked* gradual; E2 finally found the real
effect, but the mechanism was bootstrap data-quality, not the guessed "coverage grows." **Net: embodiment is
free when a drift is global-structure-learnable, and expensive only when it's local** — which is also, for the
first time, a regime where "where should I practice?" is a real, askable question rather than a rhetorical one.

[`expansion/`](expansion/README.md) (2026-07-27) then ran Cut #5, the substrate memo's own flagship claim —
that drifting dynamics is a "novelty generator" that should grow a forward model's representable directions —
and retired it. Built as calibrate→measure→calibrate after three prior instrument failures, it found drift
moves the target function by ±0.08 directions against a genuine-support-increase calibration of +0.72 ± 0.42,
roughly 10× smaller, and de-confounded the result across drift geometry. `README.md`'s own next-steps section
states the update plainly: "the memo's original motivation was wrong and should not be re-imported." What
survived was narrower and more useful than what was retired: the instrument itself (degenerate at both a
saturation floor and a diverged-rollout ceiling), a confirmation that drift is still essential for *realism*
(the entire `drift_value_loop`/`on_policy` line depends on it), and an explicit deferral of the belief's actual
core claim — does novelty grow representable directions? — to a domain with real hierarchical structure to
expand into, which a fixed-DOF motor plant does not have.

## A second biological hypothesis, tested end to end, with two corrections along the way

[`ideas/performance_error_is_the_bridge.md`](../../ideas/performance_error_is_the_bridge.md) (2026-08-11)
proposed a specific form for the FM↔value bridge — a benchmark-subtracted, agency-gated prediction error δ,
consumed as a multiplicative plasticity gain — superseding an earlier draft in the same session that had
claimed the opposite sign (that cancellation *hides* self-caused content from value, rather than agency
*enabling* credit for it). Before running anything new, [`drift_value_loop/teacher_snr/`](drift_value_loop/teacher_snr/README.md)
killed the doc's cheapest proposed experiment: its benchmark term "was already implemented" as an ordinary
REINFORCE baseline, and the claim that only a variance fix was needed did not survive contact with the data —
the obstruction was in the search procedure, not the teacher's form. [`agency_gate/`](agency_gate/README.md)
then ran the literal biological control (Gadagkar's ACT vs PLAYBACK): δ fires only when self-produced, the
ungated `b−e` fires under both, and the gate is load-bearing — but leaks, because σ(0)=0.5 gives it a floor,
forcing a first correction (the gate must be *centered*). [`plasticity_gain/`](plasticity_gain/README.md)
showed the gain consumption works on ballistic re-adaptation speed (not ceiling), and forced a second
correction of the same shape: a scalar-EWMA benchmark is mechanistically pathological and must be
context-conditional, `b(s)`, or it fixates on noise worse than the raw error it was meant to improve on.
[`two_clocks/`](two_clocks/README.md) then upheld the doc's riskiest structural bet in a stronger form than
proposed: two channels (precision and value-relevance) beat one at matched capacity in every seed, and sharing
them is *destructive interference*, not redundancy. [`bridge_assembly/`](bridge_assembly/README.md) assembled
the corrected δ live and produced the arc's real surprise: the hygiene advantage reached behavior through the
**reactive**, not the ballistic, controller — inverting the naive expectation from the ballistic arc above —
yielding a transferable law that which controller transmits an FM difference depends on where that
difference's mass lives (localized → ballistic, diffuse → reactive), alongside the honest negative that
uniform plasticity still won outright in this short-recovery regime.

## Practice, chunking, and a detour that had to happen

The practice line asks what a control loop wrapped *around* ordinary learning — practicing, not just learning —
buys, using a musical vocabulary (étude, fingering, legato, span). [`practice/etude/`](practice/etude/README.md)
(2026-08-12→14) found compilation is selection-and-commitment, not distillation-by-regression — averaging
valid command sequences destroys them — but hit a self-diagnosed dead end: post-commit drift measured exactly
0.0000, because the committed units were state-independent, making "hierarchy" vacuous. That forced a detour
into [`experiments/rhm/practice/README.md`](../rhm/practice/README.md), where the precondition could actually
be tested and confirmed: state-conditioned commitment beats state-independent 1.8–3.0×. Ported back onto MuJoCo,
[`practice/fingering/`](practice/fingering/README.md) (2026-08-19→20) reproduced the RHM law and then inverted
the étude's own head-to-head: on a substrate where boundaries carry real (postural) information, *live* content
under committed routing beats every frozen alternative — "commit the routing, not the content."
[`practice/legato/`](practice/legato/README.md) found the actual crossover the fingering result implied: live
plans win inside the model's ~21-step composition horizon, frozen measured chains win beyond it, because a
measured chain carries no composition error. [`practice/span/`](practice/span/README.md), the arc's current
edge, checked this against a literature result (Iwane et al. 2026) and found a dissociation: practice roughly
doubles the model's composition horizon through a flat task metric, but a live plan's *reach* does not follow
it — practice sharpens the model on the corridor it was practiced on and degrades it off-corridor. The
standing read, in [`README.md`](README.md)'s own words, is that chunks extend committed execution **past** the
model's reach rather than beating planning **inside** it.

## The throughline

Read end to end, the arc's biggest shift is not any single finding but a recurring correction to *where the
team looked for the effect*. Three times independently — `directed_readapt`/#4c's directed-collection
negatives, `on_policy`'s teleport-knob reversals, and `expansion`'s retired novelty-generator motivation — a
plausible non-stationarity story turned out to be a measurement artifact or a category error, and the honest
fix was narrower and more mechanistic than the original hypothesis. The performance-error bridge arc shows the
same pattern at a finer grain (two centering/conditioning corrections before the signal was even usable), and
the ballistic/`bridge_assembly` and practice/`legato` results both land on the same shape of law — *which
channel or controller carries an effect depends on the spatial or temporal structure of the effect itself*,
not on a scalar magnitude — suggesting that "match the reader to the structure of the signal" is this arc's
closest thing to a crystallizing belief, still informal and not yet written up in [`beliefs/`](../../beliefs/).
