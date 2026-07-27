# HISTORY — the `mjc` node

*A "Big History" of the MuJoCo control substrate: what we set out to do, what actually happened, and
where the thinking turned. Scope: `experiments/mjc/` only (formerly `experiments/mujoco_control/`).
Sources: git history (`c10b503` … `aef54dc`), every node README, and the idea docs that drove the
work. Auto-generatable — overwrite freely.*

**Span**: 2026-07-16 → 2026-07-27. Twelve days. Twelve child nodes plus one grandchild
(`on_policy/directed_on_policy/`). **Current state**: [README.md](README.md) · [FILES.md](FILES.md)

---

## Timeline at a glance

| Date | Commit | What landed | The shift it marks |
|---|---|---|---|
| 07-16 | `c2946bb`, `c10b503` | Idea memo + Cuts #1/#2/#3 (`contact_residual`, `arity_torque`, `dynamics_shift`) | Substrate born; **flagship reframed before it shipped** |
| 07-17 | `df6f065`, `20ad281` | Repo-wide reorg; the `two_timescale_value_loop` memo written | **A second program takes the substrate over** |
| 07-18 | *(in `080bb8a`)* | `value_shaping`, `directed_readapt` | Two negatives → **the phenomenon-first pivot** |
| 07-19 | `080bb8a` | `meta_adapt` #4 → #4e | The interface arc; **disc-4 closed** (reward *causes* the shaping) |
| 07-20 | *(in `a1fcd86`)* | `curiosity_control` | The afferent half; the noisy-TV inversion; grounding |
| 07-21 | `a1fcd86` | `online_value_loop`, `drift_value_loop` | **Obstruction as confirmation**; then *"the teacher, not the structure"* |
| 07-22 | `94eed10` | `ballistic`, `ballistic/directed`, `arm_substrate` | **"The control mode, not the value"**; a second task family |
| 07-22 | `6c3ab11`, `797b622` | `mujoco_control/` → `mjc/`, flat → 11 nested nodes | The writeup structure catches up with the science |
| 07-23→24 | `0d017e9` … `13028b4` | `on_policy/directed_on_policy/` (E3) | **The retracted S2 claim reproduces** once looking, not just collecting, is embodied |
| 07-25→26 | `4f503cc` | `ideas/heterogeneous_graders.md` written | The week's results reread as one claim: **grader disagreement is the fire** |
| 07-26 | `40a6826`, `396a902` | Cut #5's rank-triple readout retracted | **Rank fails a third time** — caught pre-compute this time |
| 07-27 | `aef54dc` | Corrected readout lands (in `rhm/`); cut #5 gated on gap-width | The open cut gets a precondition instead of a spec |

Effect sizes over the arc are their own story: **8.1×, 0.999-vs-0, 240×** (days 1) → **+0.156, +0.036 ±
0.012, ≈0, ≈0** (days 3–6, the interface descent) → **3.0×, 4.3×, 6.0×, 2.9×** (day 7, after the
reframe). The arc *descends into diminishing returns and then climbs back out* — and the climb is not
a new mechanism, it is a corrected measurement.

---

## Part I — The founding bet, and the reframing that happened the same day

The substrate was proposed in [`ideas/physical_control_substrate.md`](../../ideas/physical_control_substrate.md),
dated **2026-07-16**, status *"no experiment run yet."* By the end of that same day three cuts had
landed. That compression is the first thing a historian should notice: the memo was not a plan to be
executed later, it was a bet placed and immediately called.

The memo's core discipline — **controllable-dynamics DGP, not RL-to-SOTA** — is the one thing that
never changed across the entire week. Every experiment in the tree is still a knob sweep with a
vanilla baseline, delivering a slope or a dissociation, never a leaderboard number. It is worth
saying plainly that this constraint was *productive*, not merely defensive: it is why the arc could
run eight nodes deep in a week on CPU physics, and why the negatives were publishable to ourselves
instead of embarrassing.

Two founding choices were revised almost immediately:

1. **MJX/JAX → plain MuJoCo CPU + PyTorch.** The memo's practicalities section assumed MJX for
   GPU-parallel thousands-of-envs. The actual work never needed it — contact *fidelity* mattered
   more than throughput, and the data volumes were tiny. MJX was deferred to "the sim2sim robustness
   sweep," which never ran. The bet that the expensive thing would be *simulation* was wrong; the
   expensive thing was always *thinking of the right comparison*.

2. **The flagship was rewritten before it shipped.** The memo's headline was a **sim2sim degradation
   slope** — train on friction ∈ [a,b], test on [c,d], show the forward-model agent's slope is
   flatter (operators-not-footprints on physics). Cut #3 abandoned this *on theoretical grounds
   during execution*: a shifted operator makes the stale FM itself wrong, so the framing was shaky,
   and per-step replanning made it a near-null anyway. What replaced it is the **factorization
   claim** — a dynamics shift corrupts only the world-model factor, so a model-based agent re-adapts
   from ~50 **reward-free** transitions while a protocol-matched model-free agent is flat and needs
   ~240× more *reward-labeled* data.

That substitution is the single most consequential event in the node's history. The abandoned framing
was about *robustness* (a static property); the replacement is about *re-adaptation* (a dynamic one).
Every node built after 07-16 is downstream of re-adaptation. Cut #3's `replan_every` ablation — the
benefit grows with open-loop commitment and turns over at the FM's ~6–8-step composition horizon —
also planted, in a single throwaway "biological reading" paragraph, the seed that would become the
entire final day's work six days later.

**What Cuts #1–#3 established, and what they cost.** Cut #1 (the residual is an *event* detector, not
a state detector) and Cut #2 (arity beats resolution: 232 params command-aware beat 70,152 params
command-blind) are the cleanest results in the tree, and they are the *only* two that are clean
because the substrate was doing the work rather than being manipulated into cooperating. Both
carried a methodological gotcha that survived the whole week: **Huber, not MSE** (contact's heavy
tail starves free-flight learning), and **cosine, not R²** as the scale-free control (both magnitude
normalizers were ill-conditioned). The habit of naming the metric that *doesn't* work, and why, starts
here.

## Part II — The takeover: a second program claims the substrate

On **07-17** a different memo was written: [`ideas/two_timescale_value_loop.md`](../../ideas/two_timescale_value_loop.md)
— *"the value function is the scalar forward model the a2a program never built."* Its thesis is a fast
supervised forward model (cerebellum) plus a slow reward-driven value loop (basal ganglia), with the
FM↔value interface as the contribution: **two afferent taps** (explore off the residual `e`, exploit
off the forecast `p`) and **one efferent gain** (value-shaping = capacity re-allocation).

Everything in `mjc/` after Cut #3 is that program's laboratory. The substrate memo's own next cuts —
**partial-obs latent planning** (#4) and **drifting dynamics → dimensionality expansion** (#5, billed
as *"the single biggest open question of the whole program"*) — were never run. Neither was the
musculoskeletal/morphology thread, except in the minimal form that arrived on the last day.

This is the biggest narrative in the file and it is almost invisible in the commit log, because the
main README's numbered "Next steps" list kept items 4 and 5 alive as text for six days after they had
been de facto abandoned. A reader tracking only stated intentions would miss that the substrate had
changed owners.

## Part III — The mechanism-atom cul-de-sac, and the phenomenon-first pivot (07-18)

Two nodes landed on 07-18 and both are, in different senses, failures:

- **`value_shaping/`** — the learning-layer control. It *worked*, but only after the naïve design
  failed three times, and the failures are more instructive than the result: energy ≠ prediction
  cost; the capacity-hungry part of the physics (contact impulse) is *irreducible* so it can't be
  dropped; a single-mode field conflates smoothness with complexity; and both bodies must be
  capacity-hungry or the easy one never competes. The robust core (re-allocation) survived; the
  "teeth" (capacity efficiency) turned out to be a data-scarcity artifact and were reported as such.
- **`directed_readapt/`** — two clean negatives. A global shift has **no scarcity** (nothing for a
  drive to exploit); a localized shift has scarcity but **ensemble disagreement is blind to a
  confident-prior shift** — all members warm-start from the same prior, agree on the wrong answer,
  and the drive under-visits the patch *below uniform*.

The composed lesson — *a drive needs scarcity **and** a signal that flags confident-wrong regions* —
was correct and would be re-derived twice more. But the operational conclusion recorded on 07-18 is
the turn: **stop verifying isolated mechanistic atoms a priori; go phenomenon-first.** Isolating a
mechanism kept either killing the phenomenon or leaving the baseline already good enough.

Two guardrails were attached so this wouldn't become RL-to-SOTA: the deliverable stays a
*dissociation* against an ablation, and the architecture stays *factored and dissectible*. Both held.

## Part IV — The interface arc: efferent, then afferent (07-19 → 07-20)

`meta_adapt/` is the densest node in the tree — five cuts (#4 → #4e) in two days — and it reads as a
single argument:

- **#4**: meta-learning **collapses at a smooth floor** (gap ≈ 0, reproducing the RHM anchor on
  control) and the gap **opens monotonically with input-coupled conflict** (~8× at Φ=π/2). Collapse
  and separation are the same order parameter crossing a transition. This is the phenomenon the
  pivot went looking for.
- **#4b**: replace the opaque Reptile weights with an explicit context latent `z` — which *decodes
  the task parameter* (R²≈1.0). And the dissociation that mattered: **system-ID ⊥ adaptation
  benefit.** The encoder always identifies the task; whether that helps is set by conflict.
- **#4c**: value-of-information directed identification is a **robust, scarcity-gated negative** —
  the navigating info-MPC provably navigates (3× patch visitation) and still doesn't beat max-‖u‖.
  For a 1-D rotation, the most informative command *is* the largest one. VoI is over-engineering on
  low-dim system-ID.
- **#4d/#4e**: the efferent gain, landed. Value-shaping and meta-conditioning are **separable
  capacity levers**; the re-allocation is **value-caused**, veridicality ⊥ usefulness, and it buys
  control **only under capacity competition** (+0.036 ± 0.012 at h=64). Then #4e closed
  discriminator 4 outright: an outer loop whose *only* signal is control performance rediscovers the
  value's support with no hint. Two gotchas that generalize far beyond this node — **normalize the
  weight budget or the loop games loss-scale rather than allocation** (a concrete wireheading
  instance, and the idea doc's "on-manifold veto" made literal), and **read the optimizer's
  converged distribution, not its selection-biased single best.**

`curiosity_control/` (07-20) is the afferent complement, and it contains the week's most interesting
*inversion*. The a2a active-vision line had established ensemble disagreement as the noisy-TV filter
(high-dim random pixels drive members toward the mean, so the ensemble rejects noise by
construction). On low-dimensional control the opposite happens: with scarce bootstrapped noise data,
members fit *different* noise realizations and **disagree**, so the drive *chases* the noise. The
same instrument, opposite sign, for a structural reason. The fix was not a better filter but
**grounding** — explore and exploit as two additive drives, with an intermediate optimum. That is the
idea doc's "two taps are one system" claim, landed.

By the end of this stretch the pattern of the whole arc is visible: **every positive result arrives
with a boundary condition attached, and the boundary condition is usually the more transferable
finding.**

## Part V — The obstruction that became the thesis (07-21)

`online_value_loop/` set out to build the keystone the idea doc names verbatim: reward continuously
shaping a *live* forward model, one loop, both taps. It was built both ways and **obstructed on
both**, for two different reasons:

- **Afferent**: the drive's tracking effect is real (3× under noise) but **CEM-MPC control is robust
  to it** — the reward landscape over the balance `b` is flat, so the loop has no gradient and
  wanders by seed.
- **Efferent**: the value-shaping benefit is a **slow, commit-dependent, pre-convergence transient**.
  It washes out at convergence, and a probe short enough to be "online" cannot see a re-allocation
  that takes thousands of steps to manifest. A probe long enough *is* a full retrain — which is
  exactly why #4e had to go offline.

The interpretive move here is the intellectual high point of the week. Instead of filing two
negatives, the node reread them as a **positive, mechanistic confirmation of the framework's own
premise**: the idea doc *asserts* value belongs on the slow timescale; these experiments show **why
it is forced there**. The two-timescale split is not a design choice — it is imposed by the structure
of value.

Two reframes followed, both of which govern everything after:

1. **The value's benefit is adaptation *speed*, not converged competence.** Nothing in the arc ever
   raised a ceiling; everything accelerated adaptation. And this *rescues* the washout finding —
   washout only bites if you converge, and under perpetual drift you never do. Every new context is a
   fresh pre-convergence sprint.
2. **The fast system is cortex + cerebellum, with the FM as the bridge.** Task-doing and
   self-calibration are fast; invariant-selection is slow; the FM's prediction error is the messenger
   in both directions — which is precisely why it kept refusing to sit cleanly on either timescale in
   the experiments.

## Part VI — "The missing piece was the teacher, not the value structure" (07-21)

`drift_value_loop/` ran the experiment its parent named, three cuts deep, and reassigned credit twice:

- **Compounding is real — and it comes from the *memory*, not the value-carving.** A task-conditioned
  context latent climbs from 0.31 to 0.97 few-shot R² across a drift sequence while pooled monolithic
  memory *anti-compounds* (0.66 → 0.21) and is worse than memoryless scratch. The learning layer does
  the heavy lifting.
- **Value-relevance carving is capacity-gated and control-robust** — a clean, informative null.
- **And then the diagnosis.** The reason the meta layer looked inert is that **control is a near-blind
  grader**: a replanning controller reaches goals about as well with a stale model as a fresh one, so
  a better model barely registers. Grade the loop by the **value-relevant FM prediction error**
  instead — the literal cerebellum→VTA messenger — and the explore/exploit balance has a **clean
  interior optimum** (b = 0.5) exactly where control is *dead flat* (0.1% spread).

*You cannot self-tune a value by a metric that cannot see what the value does.* This is the sentence
the whole preceding week was walking toward, and it retroactively explains the parent's wandering
loop, #4c's wash, and half the earlier nulls.

Note also what stayed honest: the self-tuning that follows from the corrected teacher is
**directional, not crisp** — the bowl is shallow and REINFORCE is noisy — and this was recorded as
optimization engineering deferred, not quietly upgraded to a result.

## Part VII — "It was the control mode" (07-22)

The final day inverted the other half of the blind-grader diagnosis. If replanning is blind *because
it re-grounds*, then the fix is not a better value — it is a controller that **cannot** re-ground.

`ballistic/` made "how ballistic the movement is" the causal knob, and the effects came back large:

- **4a** — a confounded-drive negative, kept as a diagnostic: in the corridor geometry the intended
  explore/exploit roles **invert** (exploit already covers the needle; explore chases off-corridor
  noise). Conclusion recorded as *control variables* — drop the drive, manufacture the quality axis
  directly.
- **4b** — with a smooth, one-signed damping-staleness axis, **ballistic control transmits FM quality
  ~3–3.3× more than reactive**, and the genuinely feedforward behavior-cloned motor program transmits
  the *most*.
- **4c** — end-to-end: after a drift, **online reward-free FM re-adaptation restores ballistic
  competence ~4.3× more than reactive**, reaching the matched-FM ceiling.

The reading offered — and it is the most *synthetic* claim the node makes — is the **reward-free /
ballistic synergy**: the forward model is the one motor asset you can keep calibrated without reward
(every transition is a dense self-supervised label; in the wild, sampling rewards is dangerous) *and*
exactly the asset feedforward control depends on. The same object. Evolution gets fast committed
movement and cheap safe maintenance in one.

Its child `ballistic/directed/` then landed the directed-collection ladder that 07-18's negative could
not, and produced the week's most transferable mechanism finding, on a *structural* rather than
empirical argument: **ensemble disagreement cannot detect a drift at all.** Every committee member
trained pre-drift agrees, and they are all wrong together. Disagreement finds where you **lack data**,
not where your data went **stale** — and the idea doc's own load-bearing Type-2 non-stationarity is
precisely the stale case. That is a falsification aimed back at the program's own instrument. What
worked instead was a **counterfactual fit probe** (fit on a small sample, check whether held-out error
drops), which also sidesteps the derivative's re-opening blindness.

The node also downgraded a claim it had spent days supporting: the two-taps structure is supported
**one-shot** and unsupported **in a loop**, because budget spent somewhere irrelevant is a one-time
cost you amortize rather than a permanent loss. Reported as a null inside a positive, with real
diagnostic work (a multiplicative signal with no lower bound normalizes floating-point noise into
confident nonsense; gating fixes it) done specifically so that *the null would be a real result
rather than a broken implementation*.

**That downgrade was retracted the same day, and the retraction is the more interesting event.** An
audit of the stored per-round records (`directed_loop_audit.py` — read-only, no re-run) found the
loop's verdict rested on a *control* grader, which Part VI of this very history had established is
**near-blind to FM quality**; on a metric 62% composed of post-recovery plateau, i.e. the asymptotic
readout the same node had just discredited; and on an aggregate flipped by a single seed. Re-scored by
the corrected teacher — value-relevant FM error — the policy ranking reproduces the predicted ladder
exactly with the privileged oracle second, whereas the control ranking puts that oracle *fourth*,
behind two unprivileged heuristics. The conjunction beats reducibility-chasing on the sighted grader
in 6/6 seeds. Underneath all of it sat a **22× measurement subsidy**: 2,240 free teleported probe
transitions per round, everywhere in the world, deciding where to spend 100 — which removes the
relevance term's entire job, since it exists to say *where to even look*. Status corrected to
**untested in a loop**; the idea doc, which had not yet absorbed the downgrade, needed no change.

The generalizable failure is sharper than the finding it retracted, and it belongs in Part IX's
cross-cutting list: *the arc's own headline discovery — that most nulls were instrument failures —
did not stop the very next node from reading a program-level belief update off the instrument it had
just published as blind.* Knowing which grader is blind is not the same as remembering to stop using
it.

## Part VIII — Buying back the caveat: the arm (07-22)

`arm_substrate/` is not a cut. It exists because every ballistic-arc claim carried a "single-family"
caveat, and because of a realization about the pusher itself: **six hand-designed perturbation
mechanisms** (`puck_field`, `patch`, `push_rot`, `field_patch`, `noise_patch`, `rot_regions`) had been
bolted onto a plant whose free-flight dynamics are near-linear, each with its own tuning story. The
substrate had been carrying the scientific load by scaffolding rather than by physics.

The arm characterization is a list of corrections to its own proposal, which is the most honest form
this document records:

- **Capacity does not bind on a 2-link arm** — it binds at n ≥ 5. `M(q)` for a 2-link chain depends on
  the elbow angle alone, so the whole configuration-dependence is a smooth function of one variable.
  This sharpened #4d's "energy ≠ prediction cost" into **"nonlinear" and "capacity-hungry" are
  separate axes.**
- **Kinematic redundancy does not give a droppable subspace** — a *passive tool* does, and it puts
  #4d's capacity-competition boundary on one knob measured **in kilograms**, landing at +0.034 right
  on top of #4e's tuned +0.036 ± 0.012.
- **`goal_site` flips value-relevance with the physics byte-identical** — Cut #3's "intervene on the
  operator, hold the task fixed" run in the opposite direction, which the pusher structurally could
  never do.
- **Arity-1 is no longer at the floor** — a long chain has genuine autonomous dynamics, so Cut #2's
  arity demonstration should *stay on the pusher*. A rare instance of deliberately declining to
  generalize a result to a better substrate because the older one is cleaner.

And three methodological findings that read as the week's accumulated scar tissue: grade FM error on
the **task** distribution not the collection distribution (independently re-deriving the
drift-value-loop teacher correction); **size the open-loop CEM to the action-sequence dimension** (a
pusher-tuned budget silently converts the transmission effect into a null); and a light passive link
**NaNs silently** past the explicit-integration stability limit, producing a plausible-looking bad
number rather than a crash.

## Part IX — The structure catching up (07-22)

The last two commits are pure reorganization, and they encode a real fact. The main README grew
175 → 190 lines over six days while sibling `TOPIC_README.md` files accumulated flat beside it. On
07-21 the first *folder-style* child appeared (`drift_value_loop/README.md`). On 07-22 the whole tree
was converted: 11 nested nodes, each with its own `README.md` + `FILES.md`, and the hub README
**shrank to 85 lines** — from a monolith that told you everything to an index that tells you where.

Two details worth preserving: the directory is `mjc/`, not `mujoco/`, because a local package named
`mujoco` would shadow the real pip package on Modal's `sys.path`. And the Modal volume and app kept
their original names deliberately, with every result path set *in-script*, so the entire
reorganization moved zero results and broke zero reproductions. The "keep all prior results
reproducible" rule was tested at scale here and held.

## Part X — The retraction reproduces: looking has to be embodied too (07-23 → 07-24)

`ballistic/directed/`'s S2 audit (Part VII) had retracted the where-to-collect relevance claim as
**unmeasurable**, not false: every round both policies got ~2,240 free teleported probe/monitor
transitions to decide where to spend a budget of 100, a 22× measurement subsidy that removed the
question's entire job. [`on_policy/directed_on_policy/`](on_policy/directed_on_policy/README.md) (E3)
is the direct re-attempt, and its existence is itself a small methodological event: the retraction did
not read as a dead end, it read as *a spec for what would make the question askable* — embodied
looking, not just embodied collecting.

The fix took eleven small commits across a single day (`0d017e9` → `9b48b83`) before the ladder ran
clean, and the commit log itself is a compressed lesson in **why region geometry is not free to
assume**. Two corrections recur: on/off-reach has to be defined by actual FK'd path visitation, not a
corridor-distance or angle-from-start proxy (a proxy once placed a "distractor" on top of the start
posture, which every reach passes through by construction); and regions must sit in a radius band
where the arm's dynamics are tame, or a folded/inward placement's ~10× larger free-flight FM error
swamps the signal being measured. Both are instances of the week's oldest villain — the grader, or
here the *stimulus*, silently confounding the manipulation — just moved one level upstream, into the
world-building step that precedes any grader at all.

**The result, once the geometry held:** with monitoring metered at 1.84× instead of S2's 22×, both
halves of the retracted ladder reproduce over 3 seeds. `value = lprog × visits` beats `lprog-only`
(relevance pays once looking costs something) and beats `error-only` (reducibility-awareness pays;
`error-only` burns 47% of its budget on the noisy-TV trap and collapses toward uniform), and matches
the privileged oracle (0.350 vs 0.372). Control is again near-blind (~0.08, saturated) — Part VI's
"grade by the value-relevant FM error, not control" is now confirmed a third time, on a third
substrate, without anyone having to re-derive it.

The more interesting negative result is structural rather than statistical: **the on-reach-noise decoy
that separated `value` from `visits-only` in the teleport world cannot be built on-policy at all.**
Visitation concentrates almost entirely on the reach target (1.0 there, ~0 elsewhere) — you only go
where you reach — so there is no "visited but irrelevant" territory for a distractor to occupy. The
relevance lever survives, but it now lives entirely in the off-reach direction. This is the first place
in the arc where embodiment doesn't just cost more transitions, it **removes an experimental condition
that the teleport substrate had access to for free.**

## Part XI — Naming the fire, and the third rank failure caught in time (07-25 → 07-27)

[`ideas/heterogeneous_graders.md`](../../ideas/heterogeneous_graders.md) is not a new experiment; it is
the week's first attempt to answer, in one frame, a question a conversation asked directly: in what
sense is motor learning *harder* than distribution modeling, "or else animals would not have needed
anything except a cortex." The document's discipline is unusual for this tree — its evidence is
entirely **already-measured** results re-read, and it says so up front — which makes it a different
kind of artifact than everything before it: a synthesis pass over the arc's own instrument problem,
rather than another cut.

Its central move is to reject target-complexity as the axis (a 2-link arm's command-aware FM needs 232
params where command-blind needs 70,152 and still loses; language's residual is full-rank and still
learnable) and relocate "harder" onto **epistemic conditions**: commitment under delay, metered
measurement, endogenous data, Type-2 non-stationarity, and — the one this history has been building
toward since Part VI — **two teachers of different type**, because no single signal can be both dense
(free, every step, self-supervised) and evaluative (referencing outcomes you'd rather not sample). The
doc's sharpest reframe of this arc's own six days: **most of the nulls were instrument failures, and
every one was caught by playing two differently-typed graders against each other** — control vs
value-relevant FM error, ensemble disagreement vs a counterfactual fit probe. That is not incidental to
the science; the doc's claim is that *this is what heterogeneous grading is for*, and that a **single
homogeneous ensemble (same objective, different seeds) cannot ever produce this**, because a
structurally blind grader is blind for every member alike. The proposed next cut (§9, unbuilt) is to
make cross-grader disagreement itself the allocation signal — the natural sequel to E3, which already
computes both graders per round and currently reconciles them by hand in the writeup.

The same window closes a loose thread that had been live since the addendum written at commit
`40a6826`: cut #5 (dimensionality expansion under drift, billed in the founding memo as "the single
biggest open question of the whole program") had a readout spec — the `(R_act, R_comp, R_res)`
rank-decomposition triple — that was **retracted before it ever ran** (`40a6826`, `396a902`) when
in-flight re-measurement on the a2a/RHM substrate showed the triple's partition doesn't hold:
`R_comp` is not independent of `R_act`, so a matched-control defense doesn't rescue it. That makes
**three** independent times this tree has found rank a weak instrument for this kind of question
(`contact_residual`'s retired "rank ∝ complexity," a2a's `rank ⊥ noise`, and this), and the third is
qualitatively different from the first two: it is the first one caught *before* the compute was spent,
because a second, differently-typed measurement was actually run rather than reasoned about — the
grader-heterogeneity discipline the same week had just named, working on itself in real time.

On 07-27 the corrected readout landed downstream, at
[`rhm/residual_decomposition/README.md`](../rhm/residual_decomposition/README.md) — **β** (the
`res_var ∝ act_var^β` exponent, capacity-invariant) and **`R_res_participation`**, in place of the
withdrawn triple — and cut #5 in `mjc/` picked up a precondition it didn't have before: a **gap-width
gate**. Re-measurement found the residual geometry is regime-dependent on whether the FM *saturates*
(drives relative residual to ~0); where it does, both β and rank collapse to a noise-floor reading
regardless of the underlying DGP. An arm FM doing one-step `(s,u) → s′` is the narrowest possible gap,
so the cut may have nothing to find on this substrate until it is run at the ballistic horizon instead
of k=1 — which reframes "gap width" as the same axis `ballistic/` has been about all along
(reactive control re-grounds every step; ballistic control commits for a horizon). Cut #5 is therefore
not abandoned and not yet a spec either: it is now **one cheap calibration check away from being
launchable**, which is a more honest place for it to sit than the six days it spent as an unqualified
"biggest open question."

---

## Cross-cutting narratives

**1. The grader was the recurring villain.** Read as one line, the week is a sequence of measurement
corrections escalating in depth: the *metric* is wrong (R² → cosine, Cut #1) → the *estimator* is
wrong (single-best → converged distribution, #4e) → the *probe distribution* is wrong (broad → task,
arm) → the *summary statistic* is wrong (mean-over-rounds → excess damage; `rounds-to-recover`
retired) → the **teacher** is wrong (control → value-relevant FM error) → the **control mode** is
wrong (replanning re-grounds past the thing you're measuring). Each level up explained more of the
preceding nulls. The deepest finding in the node is not about value at all; it is that most of the
arc's null results were instrument failures of increasing subtlety.

*Addendum, and its own counter-example (07-25 → 07-27, told in full in Part XI):* cut #5's planned
readout — the `(R_act, R_comp, R_res)` triple — was retracted *while the cut was still being specified*,
not after it ran, because a second, differently-typed measurement was actually taken rather than
reasoned about. Rank has now failed as an instrument here three times
(`contact_residual`, a2a's `rank ⊥ noise`, and this), but this one is the exception that proves the
rule directly above: the discipline that "did not stop the very next node" in Part VII *did* work here
— the difference is that someone ran the second grader instead of trusting the first one's own
diagnosis of its blind spots.

**2. Negatives were the engine, not the exhaust.** `directed_readapt` (two), #4c, both halves of
`online_value_loop`, drift Cut 2, ballistic 4a, directed S2's internal null. Every one names the next
experiment in its own text, and the chain of "Builds on this" pointers across the tree is almost
entirely a chain of negatives. The one place this discipline is most visible: `online_value_loop`,
where a double negative was reread as a positive confirmation of the framework — a move that is
either the best or the most dangerous thing in the file, and which the node itself flagged as *"a
prediction, not yet a result."*

**3. Boundary conditions outlived headlines.** Value-shaping buys control **iff** capacity binds.
Curiosity pays **iff** the frontier is value-relevant *and* moving. Meta beats pooling **iff** the
family is input-coupled-conflicted. Directed collection pays **iff** allocation is zero-sum. FM
quality reaches behavior **iff** control is committed. In every case the conditional is more portable
than the effect.

**4. Biology as compass, repeatedly vindicated.** The `replan_every` biological reading in Cut #3 was
a paragraph of speculation on day one; on day seven it was the causal knob carrying the largest
effects in the tree. The cerebellum→VTA pathway went from a citation in an idea doc to the literal
specification of the corrected teacher. The Shadmehr force-field protocol became the arm's best
quality axis. This is a real methodological result about *this* research process, not a decoration.

**5. Evidentiary standards rose as effects shrank.** Days 1–2 are single-seed by explicit convention
(effects of 8× and R²-0.999-vs-0 don't need seeds). By day 4 the effects were +0.036 and 3–4 seeds
were mandatory, with the standing note that *mechanism metrics are trustworthy and control is the
noisy readout*. The convention didn't change; the regime did, and the work noticed.

**6. What the substrate refused.** Two things the pusher structurally could not deliver, both of which
cost days before being named: capacity competition that isn't manufactured (which drove the arm), and
a value-relevance intervention that holds the physics fixed (which `goal_site` finally provided). The
tell was visible early — six additive perturbation knobs on a near-linear plant — but it took the
"single-family" caveat piling up across three nodes to force the reckoning.

---

## Priors, then and now

| Question | 07-16 prior | 07-27 position |
|---|---|---|
| What is MuJoCo *for* here? | Testing operators-not-footprints robustness on physics (degradation slope) | Testing the FM↔value interface; robustness slope abandoned as theoretically shaky |
| What does a dynamics shift buy? | A static robustness comparison | A **re-adaptation sample-efficiency dissociation** (~50 reward-free vs ~240× reward-labeled) |
| Where does value help? | Not yet asked | **Adaptation speed, never converged competence**; and only under capacity competition |
| Why two timescales? | Asserted by the idea doc | **Forced** — fast-online discovery of value is structurally obstructed on both taps |
| What compounds? | The value-carving (meta layer) | The **memory architecture** (learning layer); the carving is capacity-gated and control-robust |
| How to grade a value loop? | Task reward | **Value-relevant FM prediction error** — control is a near-blind grader; now confirmed a third time on a third substrate (E3) |
| Is ensemble disagreement the reducibility instrument? | Yes (a2a Phase 2b) | **No for Type-2 drift** — blind to stale data by construction; use a counterfactual fit probe |
| Do the `e`-tap and `p`-tap form one system? | Predicted | **Reproduces once looking is embodied too** (E3, on a third substrate) — one-shot; in a loop with genuinely metered monitoring it now measures as positive, not just untested |
| Does directed collection pay once measurement is metered? | Retracted as unmeasurable (22× subsidy, `ballistic/directed` S2) | **Reproduces at 1.84× metering** (E3) — the retraction was about the instrument, not the claim |
| Does a better FM reach behavior? | Assumed | **Only under feedforward commitment** (~3×; ~4.3× for re-adaptation; 6.0× on the arm) |
| Is "nonlinear" enough for capacity pressure? | Implicitly yes | **No** — nonlinear and capacity-hungry are separate axes; chain length is the knob |
| Why does the substrate need *two* graders at all? | Unasked | **Derived, not designed** — no signal can be both dense (free, every step) and evaluative (references outcomes); [`heterogeneous_graders.md`](../../ideas/heterogeneous_graders.md) names this the arc's own operating principle |
| Is rank a trustworthy readout for representational structure here? | Untested | **No, three times over** — but the third failure was caught before any compute ran, by actually running a second, differently-typed check instead of reasoning about it |

## Planned and never run

Recorded because their absence is informative, not because they failed:

- **Partial-observability / latent planning** (substrate memo cut #4) — the Stage-3c/3d analog on real
  dynamics. Still listed in Next steps; never started.
- **Dimensionality expansion under drifting dynamics** (memo cut #5, billed as the program's biggest
  open question) — no longer simply unrun. As of 07-27 it has a design (support-growing drift, chain
  length ≥ 5) and a corrected readout (β, `R_res_participation`) but is **gated behind a cheap
  gap-width calibration** that has not yet been run — see Part XI. The nearest it has come to actually
  starting.
- **Heterogeneous-vs-homogeneous grader disagreement as an allocation signal** (`heterogeneous_graders.md`
  §9) — proposed 07-25/26, not on any Next-steps list yet, cheap on the existing E3 substrate (both
  graders are already computed per round).
- **MJX, sim2sim at scale, musculoskeletal bodies, morphological computation** — the arm's passive
  tool is the only piece of the morphology thread that exists.
- **Cut #3 hardening** (multi-seed the REINFORCE slope; a mass-shift as a second qualitatively
  different operator) — proposed 07-16, still open 07-27, restated verbatim in every README version.
- **Cut #1's frame-skip sweep** (separating the aliasing cause from the stiffness cause) — the one
  loose mechanistic thread on the cleanest result in the tree.
- **Porting a cut onto `arm_substrate/`'s passive-tool capacity axis** beyond what 4c already ports —
  the structural debt every day-7 ballistic claim still carries as a "single-family" caveat outside 4c.
- **Back-translation into the belief tree** — flagged as owed by six separate nodes now (E3 and the
  heterogeneous-graders synthesis add to the debt rather than paying it down). The interface findings
  (*value is slow/committed*; *the teacher, not the structure*; *the control mode gates transmission*;
  *heterogeneous grading is why the nulls were catchable at all*) are belief-shaped and not yet
  crystallized. This is the largest outstanding debt.

## Where it stands

The live thread as of 07-27 is the **gap-width calibration** gating cut #5 (Part XI): a small, cheap
check of whether an arm FM's one-step residual geometry differs from its ballistic-horizon geometry,
run before any full expansion-under-drift ladder. If the calibration finds nothing at either gap
width, cut #5 closes on this substrate rather than staying open by default. In parallel, the
`heterogeneous_graders.md` §9 cut — using cross-grader disagreement itself, rather than one grader
alone, as the allocation signal — is unbuilt but cheap, since E3 already computes both instruments per
round. The largest structural debt is unchanged from 07-22: [`arm_substrate/`](arm_substrate/README.md)
carries only the 4c port, so the ballistic transmission/re-adaptation/directed-collection results all
still lean on the pusher for their strongest form.

Read as a whole, the node is now a twelve-day arc in which a substrate proposed for one program was
taken over by another, kept its methodological discipline entirely intact through the transfer,
descended through five increasingly subtle measurement failures into a set of near-nulls, climbed back
out by discovering the nulls were about *how we were looking* — first the teacher, then the control
mode — and then, on reflection, watched its own retracted claims **reproduce** once the same fix
(embody the measurement, not just the collection) was applied one substrate later. The final turn is
the arc naming its own method: the reason nulls kept turning into findings is that no result here ever
depended on a single grader, and the one time a readout *did* implicitly assume its components were
independent (cut #5's rank triple), the failure was caught before it cost anything — because a second
grader was run, not because anyone remembered to be careful.
