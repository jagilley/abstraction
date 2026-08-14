# HISTORY: experiments/mjc

A big-history pass over the MuJoCo controllable-dynamics substrate, read from its READMEs
rather than commits (git history here is squashed). Full index: [FILES.md](FILES.md) ·
[README.md](README.md).

## 1. Why this substrate exists, and the discipline that stuck

`mjc` was built as a third DGP rung between RHM and language — one step more physically
real, but explicitly *not* a robotics benchmark. The founding discipline, stated in
[README.md](README.md) and never relaxed across ~20 nodes, is: sweep DGP knobs, report a
slope or a dissociation, never a leaderboard number — which is what let every subsequent
negative result be read as informative rather than as failure, a posture this document
keeps bumping into.

## 2. The three foundational cuts: learning to read residuals and instruments

[`contact_residual/`](contact_residual/README.md) (Cut #1) set out to show an FM's error
concentrates at contact, and it did — but the *sharpened* finding, forced by an
onset-aligned control, was that the residual marks the **regime transition**, not the
contact state. A wrong prior was named and retired in the same doc
(`residual rank ∝ DGP complexity`), and a labeling bug that silently poisoned the
free-flight baseline is documented rather than erased — already the node's norm is
forming: report the instrument's limits alongside the result.

[`arity_torque/`](arity_torque/README.md) (Cut #2) killed the RHM "received-wisdom"
confound (i.i.d. commands) before claiming arity beats capacity: "The command slot is not
a capacity problem."

[`dynamics_shift/`](dynamics_shift/README.md) (Cut #3) is the pivot node. An early
per-step-replanning run came back a near-null — feedback substitutes for the model — and
rather than reporting that as the answer, the team diagnosed *why*: reward-free MB
recovery only shows up when the controller commits long enough for the model to be
load-bearing. That composition-horizon finding ("~6-8 steps") reproduces a number from
language/reaching work on raw physics, and is named explicitly as "the seed of the whole
ballistic arc" — this doc forecasting the shape of six months of downstream work.

## 3. The interface arc opens, and a program-level pivot gets dated

[`value_shaping/`](value_shaping/README.md) is deliberately built as the *stationary*
control for everything that follows — value re-allocates FM capacity away from an
irrelevant force field — and is honest that its "teeth" (capacity efficiency) are a
fragile, data-scarcity effect, not a robust one. It also states a wrong prior plainly:
"Energy ≠ prediction cost."

[`directed_readapt/`](directed_readapt/README.md) is where the arc's central mechanistic
finding first appears and where the *methodology itself* changes. Disagreement-directed
collection failed twice — null under global drift, actively blind under local drift, because
an ensemble warm-started from the old world model agrees with itself exactly where it is
wrong: "disagreement flags where the model is uncertain, not where it is confidently wrong."
That finding recurs through [`curiosity_control/`](curiosity_control/README.md) and
[`ballistic/directed/`](ballistic/directed/README.md) below. But the doc's larger move is a
declared pivot away from verifying every mechanistic atom in isolation — which kept either
killing the phenomenon or leaving an already-strong baseline — toward **phenomenon-first
work, backfilled with mechanism**. Everything from `meta_adapt/` onward is written that way.

## 4. Cut #4 → #4e: a single arc that keeps promoting its own null results

[`meta_adapt/`](meta_adapt/README.md) is the longest single-node arc and reads as a chain
of nulls each telling the next experiment what to ask. #4's floor (1-D damping) collapses
to zero gap — the RHM anchor reproducing on physics — and that null is credited with
motivating the next design: "The floor told us what's missing: task conflict." #4b
dissociates two things that had looked identical: a context latent decodes the task at
R² ≈ 1.0 while buying zero adaptation benefit — "Knowing the task is true but useless."
#4c is a defended negative: VoI collection works mechanically (3× patch visitation) yet
still loses to a trivial heuristic, because on a low-dim rotation task the most
informative command is simply the largest one — "On low-dim system-ID, VoI is
over-engineering," a boundary condition rather than a defeat. #4d keeps a "load-bearing
false start" (a vacuous position-only loss) in the doc rather than deleting it, and finds
an FM-gain/control-gain mismatch (biggest one-step gain at h=16, biggest control gain at
h=64) echoing Cut #3's composition-horizon logic. #4e closes the loop and produces a
lesson beyond this substrate: an unconstrained reward loop games loss *scale* rather than
allocation (a wireheading instance), fixed by a normalized budget.

## 5. Curiosity and the online loop: value gets pushed onto a slow timescale, on purpose

[`curiosity_control/`](curiosity_control/README.md) tried the intended fix for
`directed_readapt`'s blindness — a reducibility-filtering ensemble — and it inverted: "the
**reducibility filter fails on this substrate**," a substrate inversion of an earlier
active-vision result. The fix that worked was not a smarter filter but grounding: an
explore drive and an exploit drive run as two additive taps with an interior balance
optimum. [`online_value_loop/`](online_value_loop/README.md) then let a single online
reward loop try to *discover* the right setting of both taps, and both attempts were
obstructed — CEM-MPC control proved robust to the drive's tracking gap, and the
value-shaping benefit needed more convergence time than any online probe could grant. The
doc reframes this double failure as the result: "the idea doc *asserts* value belongs on
the slow timescale; these experiments show why it is forced there." The belief that
survives into every later node: value's benefit is *adaptation speed*, not competence.

## 6. Chasing the blind grader: drift_value_loop and ballistic

[`drift_value_loop/`](drift_value_loop/README.md) found the compounding effect predicted
upstream, but reassigned its source from value-carving (a clean, capacity-gated null) to
fast memory. Diagnosing *why* the value carve stayed null produced this arc's sharpest
instrument critique: "control is a near-blind grader" — a replanning controller reaches
goals about as well on a stale model as a fresh one. Swapping the teacher signal to
value-relevant FM prediction error (read as the cerebellum→VTA messenger) opened a clean
interior optimum where control had been flat: "the missing piece was the teacher, not the
value structure." [`teacher_snr/`](drift_value_loop/teacher_snr/README.md) then refuted the
idea doc's proposed fix for that teacher *before it ran* — the fix was already implemented
— and located the real obstruction in an estimator variance that within-epoch averaging
structurally cannot touch.

[`ballistic/`](ballistic/README.md) relocates the "blind grader" one more time, from teacher
to controller: a feedforward, committed controller makes the FM ~3× more behaviorally
load-bearing than a reactive one, and reward-free re-adaptation restores ballistic
competence ~4.3× more. The reading is explicitly evolutionary — the FM is "the
reward-free-maintainable asset *and* the feedforward-critical asset — the same object,"
motivated by sensorimotor delay making pure reactivity biologically impossible. Its child,
[`ballistic/directed/`](ballistic/directed/README.md), is the arc's clearest self-audit: an
S2 result claiming the value drive picks *where* to look was retracted after discovering a
22× measurement subsidy in its own monitoring loop — "reading a program-level belief
downgrade off the instrument you have already published as blind" is named as the
transferable lesson, more durable than the retracted finding. That retraction motivates the
on-policy work in §7.

## 7. Auditing the substrate itself: the teleportation correction

[`on_policy/COLLECTION_REALISM.md`](on_policy/COLLECTION_REALISM.md), written directly out
of the `ballistic/directed` audit, turns skepticism on the whole prior corpus: every FM to
that point trained on teleported, omniscient, free transitions. Its epistemic move is
triage, not blanket retraction — a three-tier trust system (dissociations stay **safe**;
absolute sample counts are **inflated** upper bounds; anything grading *allocation of
experience* was **unaskable**). [`on_policy/README.md`](on_policy/README.md) then found,
twice, that an apparent on-policy advantage was a mistuned teleport sampling knob:
"Teleport was mistuned, and that masqueraded as a scientific effect twice." What survived
as genuinely embodied was command-state entanglement, plus a sharp regime split —
embodiment is nearly free under global drift, expensive under local drift, because "the
data you gather while your model is wrong is itself wrong-distributed."
[`on_policy/directed_on_policy/`](on_policy/directed_on_policy/README.md) reran the
retracted §6 claim under metered monitoring and it reproduced — vindicating the audit, not
the original result. [`metered_repair/`](on_policy/metered_repair/README.md) keeps the
self-correction going: even the repaired tap's own allocation behavior doesn't fully cash
out, left as a standing, named embarrassment rather than smoothed over.

## 8. Cut #5: retracting the substrate's own founding pitch

[`expansion/`](expansion/README.md) was billed at the program's outset as its single
biggest open question and sat unrun for months because "rank-shaped instruments had failed
three times in this repo." What made it askable was a calibrate → measure → calibrate
design, justified because "a null read off a fourth such instrument would have been worth
nothing." The result is a clean refutation of the substrate's own motivating claim: physics
drift does not open a forward model's representable directions (±0.08, against a genuine
+0.72 ± 0.42 calibration). This produced a direct retraction, dated and quoted in the idea
doc itself (`ideas/physical_control_substrate.md:63`, "⚠️ Retired 2026-07-27"): drift moves
the target function; only a genuine support increase grows the space of representable
functions. Drift stays essential for realism, only its standing with respect to novelty is
withdrawn — and the belief's core claim is reassigned to a hierarchical domain
([`rhm/residual_decomposition/`](../rhm/residual_decomposition/README.md)), untested anywhere.

## 9. A second plant, and its own list of corrected priors

[`arm_substrate/`](arm_substrate/README.md) replaced the pusher's six hand-tuned
`qfrc_applied` mechanisms with capacity competition intrinsic to the body, and is candid
about where its own pitch was wrong: capacity does not bind on a 2-link arm, and kinematic
redundancy does not hand you a droppable subspace as the memo assumed. Its best result — a
mirror-signed aftereffect the pusher structurally cannot produce — is framed as the standard
this program now holds new substrates to: "the arm buying a readout rather than just a
second data point." It states its own limits too: "Physics realism ≠ experience realism" —
it still inherits teleported acquisition from §7's audit.

## 10. Four rounds testing a single biological signal

[`agency_gate/`](agency_gate/README.md), [`plasticity_gain/`](plasticity_gain/README.md),
[`two_clocks/`](two_clocks/README.md), and [`bridge_assembly/`](bridge_assembly/README.md)
(all dated 2026-08-11) test and correct successive stages of
`ideas/performance_error_is_the_bridge.md`'s δ signal, literalizing specific neuroscience
protocols (Gadagkar's playback control, Kim/Parvin/Ivry's plasticity gain, Suvrathan's
delay-matched credit). `agency_gate` finds a centering bug in the idea doc's own formula
(σ(0)=0.5 leaks half the signal under passive playback) and shows the gate is graded, not
binary. `plasticity_gain` shows δ buys re-adaptation speed, not ceiling, and that the idea
doc's literal scalar-EWMA benchmark is "mechanistically pathological" — it fixates on noise
*worse* than no benchmark at all. `two_clocks` overshoots its own prediction: sharing a
credit-assignment window is "destructive interference," worse than either channel alone.
`bridge_assembly` closes the round with two reversals — a transmission inversion (localized
FM error reaches behavior via the *ballistic* path, diffuse error via the *reactive* path,
opposite the arc's working assumption) and an anti-climax reported rather than buried: at
matched budget, plain uniform plasticity beats both error-modulated forms here.

## 11. What actually changed, across the whole arc

Three throughlines run underneath the eighteen individual nodes:

- **The team learned not to trust its own instruments before calibrating them.** Rank-shaped
  readouts failed three times before `expansion/` got one right; a teleport sampling knob
  produced a phantom effect twice before `on_policy/` diagnosed it; a 22× measurement subsidy
  produced a phantom finding in `ballistic/directed/` before retraction. The lesson, stated
  in `COLLECTION_REALISM.md`, is to sort claims by which parts of the measurement apparatus
  they depend on before trusting the number.
- **"Who's the blind grader" is a single question relocated five times** — from the value
  structure (`drift_value_loop`) to the teacher signal (`teacher_snr`) to the controller
  (`ballistic`) to the collection substrate (`on_policy`) — each relocation an actual
  finding, not a retreat.
- **Value's payoff crystallized as adaptation speed, not competence**, and that became
  load-bearing rather than disappointing. First seen as an obstruction in
  `online_value_loop/` ("none of it raises the ceiling on a fixed task — all of it
  accelerates adaptation"), it recurs as the organizing frame for `drift_value_loop/`'s
  interior optimum and `bridge_assembly/`'s regime-dependent δ result — a null the team
  explains away that becomes the arc's headline claim.
