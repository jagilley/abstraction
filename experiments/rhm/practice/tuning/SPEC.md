# SPEC — tuning: does a two-basis discrepancy type an out-of-distribution event, does the type pick the op, and does a forward self-model grade the op?

**Status**: spec, unbuilt (2026-08-29; FM factor added the same day). Written for an orchestrating
agent who will delegate each gate to a subagent (see
`/write-spec-or-prompt`[^private] and
`/subagent-instructions`[^private]).
**Up**: [`../README.md`](../README.md) (practice arc).
**Substrate donors**: [`../fourwall/lm/`](../fourwall/lm/README.md) — inherit `wall.py` / `wall_lm.py`
and the one-worker-per-arm bit-identity design verbatim, untouched (its results must stay
byte-reproducible); [`../teacher_slot/decision/`](../teacher_slot/FILES.md) — the per-checkpoint
reversible condition choice, the ABBA paired-trial rule, priced reads, and
`endo_yield/`[^private]'s **uncharged shadow panel** (gauges computed but not acted
on, with a transparency gate that the panel moves nothing);
[`../../conditional_revision/`](../../conditional_revision/README.md) Gate 0 — the `1L/8H/16d` activation
FM idiom, for the FM factor.
**Parents**: [`operators_are_arity_two`](../../../../ideas/operators_are_arity_two.md) §1–§2, §7 ·
[`revision_not_surprisal`](../../../../ideas/revision_not_surprisal.md) §4 (the aleatoric null) ·
[`performance_error_is_the_bridge`](../../../../ideas/performance_error_is_the_bridge.md) (the FM's error
against its own running benchmark as the meter) ·
[`two_timescale_value_loop`](../../../../ideas/two_timescale_value_loop.md) (`−d‖e‖/dt` on the reducible
residual) · [`heterogeneous_graders`](../../../../ideas/heterogeneous_graders.md) §7–§8 (a footprint
model has one basis to grade in; grader-type diversity) ·
[`recurrence_manufactures_confounds`](../../../../ideas/recurrence_manufactures_confounds.md) §5 ·
[`beliefs/trees/operators_not_footprints.md`](../../../../beliefs/trees/operators_not_footprints.md).
**Attribution**: the claim this tests is Jasper's (conversation 2026-08-29): vanilla transformers
judge a novel point along whatever representational axes the training data made cheapest, whereas a
two-system learner with a self-model can judge it along the axes of its own cognition. The sharpening
to "both bases held at once, and the *discrepancy* between them is the judgment," the mapping onto
`fourwall/lm`'s key/parse split, and the RASP-L reading (the key is the shortcut program; the parse is
the operator that survives the shift) came out of the exchange. The first draft cashed "second basis"
as two readers differing in conditioning set and left the self-model out; Jasper asked why, and the FM
re-entered as a **co-equal independent variable** rather than an add-on — the reasoning is in §*Two
factors* below.

## Why this experiment

The repo has measured the two-basis quantity three ways and used it zero times:

- Revision is separable from surprisal at the levels the model represents
  ([`conditional_revision`](../../conditional_revision/README.md) Gate B, AUC 0.690 at d2 with surprisal
  and position pinned) — open-loop by design.
- The content lives in the raw state update and is public
  ([`confabulation/temporal/`](../../confabulation/temporal/README.md),
  [`epistemics/`](../../confabulation/temporal/epistemics/README.md)); on that line the FM's remaining
  defensible role was left as *timing/topology*, untested.
- Every gate we have handed a learner was surprisal-shaped or within-level, and each drove
  assimilation or refused: `nll`-ordering was the worst assignment in
  [`endogenous_teacher`](../../endogenous_teacher/README.md); the bare NTP reader in
  [`fourwall/lm`](../fourwall/lm/README.md) tracks every rotation (re-maps in 50–250 steps) and never
  merges; [`teacher_slot`](../teacher_slot/README.md)'s `outer_task` loop, reading its own loss, refuses
  the merge every time it is offered.

`teacher_slot` then showed that a loop plumbed to a *counterfactual* gauge — the model's own NLL with
the key neutralised — merges at step 1500, **before any rotation**, and holds. That reframes the
question: given a continuously-read counterfactual currency, the merge decision does not need an
event at all. What it leaves untested is the claim above in its sharp form — **when an OOD event
arrives, can the learner tell *what kind* it is, does the kind select the op, and can the learner
grade the op it took?** Three kinds of news raise surprisal identically on the same tokens and call
for three different responses:

| event | what actually changed | correct op | wrong op costs |
|---|---|---|---|
| **rotation** (`wall` schedule) | the key's meaning; the grammar is untouched | **merge** (drop the key basis, route through the parse) — or track, at the price of staying hollow | tracking forever: pathway 0.27 vs 0.84 |
| **noise burst** (new) | nothing learnable; irreducible corruption on the indexed span | **skip** — the aleatoric null as an *action* | learning from it: a transient in the wrong direction; merging on it: the 0.152 nats/token index thrown away |
| **grammar drift** (`transpose`-style, extension) | a rule cell; the key is still valid | **track** | merging: index lost for nothing; skipping: the change never learned |

A single-reader surprisal gauge cannot separate these on the indexed span. Two different kinds of
instrument might:

- a gauge that holds **two bases of the input** — the key reader and the parse reader — and reads
  their disagreement *at the event instant*: the events disrupt different bases (a rotation moves
  only the key basis; a burst moves both; a drift moves both);
- a **forward self-model** of the reader, whose residual reads whether the *reader's own computation*
  is moving over the *following window*, and whether that movement is reducible. Drift and burst
  both move both input bases at the instant; they differ in what happens next.

These are different in kind — input-conditioning vs self-modeling; instant vs window; same type as
the reader vs a different type — and that is why they are two factors rather than one gauge with a
sub-option. The experiment crosses them.

One trap to design against from the start: rotations are span-local and a naive noise burst would be
global, so span-locality alone would type them. **Noise bursts must be confined to the indexed span
(leaves 0–15) and magnitude-matched to a rotation's NLL spike**, so that a single reader's
indexed-span surprisal is the same for both. Only then does either instrument have to earn the
distinction.

## Reading list (ordered; first three load-bearing)

1. [`../fourwall/lm/README.md`](../fourwall/lm/README.md) + [`FILES.md`](../fourwall/lm/FILES.md) — the
   world, the instrument glossary, the bit-identity design, the floors (0.0012 seed / 0.0039 placebo
   on indexed-span NLL), the `wall_fast` checkpoint-grid correction.
2. [`../teacher_slot/README.md`](../teacher_slot/README.md) findings 1–3 and the `decision/` + `endo_yield/`
   machinery — the reversible per-checkpoint condition choice, the ABBA paired trial (cancels a linear
   trend exactly), dead zones at measured floors, priced reads, and the shadow panel.
3. [`OOD_ROBUSTNESS`](../../../a2a_forward/OOD_ROBUSTNESS_README.md) and
   [`confabulation/temporal/epistemics/`](../../confabulation/temporal/epistemics/README.md) — what the FM
   has been measured to do (distribution-invariant robustness; models the transformation, not the
   manifold) and not do (concentrate revision). The FM factor here is built on the first and is the
   test the second left standing.
4. [`revision_not_surprisal`](../../../../ideas/revision_not_surprisal.md) §4 and
   [`conditional_revision/README.md`](../../conditional_revision/README.md) Gate B — why surprisal
   cannot be the gate and what "aleatoric null" means on RHM.
5. [`performance_error_is_the_bridge`](../../../../ideas/performance_error_is_the_bridge.md) §1 — the
   benchmark-subtracted FM error as the meter; [`two_timescale_value_loop`](../../../../ideas/two_timescale_value_loop.md)
   — why the criterion is `−d‖e‖/dt` on the reducible residual and never raw magnitude.
6. [`operators_are_arity_two`](../../../../ideas/operators_are_arity_two.md) §7 and
   [`heterogeneous_graders`](../../../../ideas/heterogeneous_graders.md) §7 — why a second basis has to
   be disjoint by construction, and the mirror-grader problem the input-basis factor walks into.
7. [`../transpose/FILES.md`](../transpose/FILES.md) — the hard rule-cell resampling machinery and its
   admissibility result (level-2 drift degrades the executor; level-3 drift is admissible) for Gate 2.

## Two factors

### Factor T — the typing gauge, read at the event instant

Two readers of the same batch, differing only in the input condition (the `arity_torque` idiom —
identical data, only the input differs):

```
p_key   = reader( x_<t , w )        the key basis   (the shortcut; what the data made cheapest)
p_parse = reader( x_<t , ∅ )        the parse basis (the operator; what survives rotation)

D_pair  = KL( p_key ‖ p_parse )  on the indexed span, per checkpoint window     [two models]
D_self  = the same, with p_parse read counterfactually from the keyed model itself   [one model, two conditions — teacher_slot's pathway instrument]
nll     = the keyed reader's own indexed-span NLL                                   [the I/O-public twin]
```

`D_pair` needs a second reader trained in lockstep (the `no_wall` arm, which already exists as a
bit-identical donor). `D_self` is what `teacher_slot` already reads at 6 step-equivalents per read.
`nll` is the observer-simulated twin that [`temporal_confabulation_test`](../../../../ideas/temporal_confabulation_test.md)
says any "signal X helps" claim must beat. Whether `D_self` suffices or `D_pair` is needed is itself a
question: `fwlm0` found the keyed model *suppresses* its own parse pathway (d4 0.27), so the one-model
counterfactual may be reading a hollow basis. **Both readers are of the same type** — this factor is
disjoint by conditioning set, not by decomposition, which is the standing mirror worry: two NTP
readers may go blind together on exactly the events (drift) where the design needs them to disagree.

### Factor M — the meter, read over the window after the op

A forward self-model of the keyed reader — depth FM `h0[≤t] → h6[t]` on the indexed span, `1L/8H/16d`,
the `conditional_revision` Gate-0 idiom — trained online alongside the reader, with its residual
`e_t = ‖h6 − FM(h0)‖` read per window against its own running benchmark `b_t = EWMA(e)`:

```
δ_t = b_t − e_t          the performance-error object (positive = the reader is more predictable to its own model than lately)
```

What this reads, by the `OOD_ROBUSTNESS` result, is **whether the reader's computation is moving**,
largely independent of whether the *input* moved — the FM compresses the weight-defined
transformation, so its advantage was distribution-invariant across corpora while the manifold model's
collapsed. Walked through the events, with the op taken:

| event → op | input | reader's weights | Factor T (instant) | Factor M (following window) |
|---|---|---|---|---|
| rotation → track | in-distribution | re-mapping key→z | `D_pair` spikes | `e` rises while the re-map runs, decays as the FM catches up |
| rotation → merge | in-distribution | parse pathway recruiting | spikes | rises then falls — reducible self-change |
| burst → skip | off-distribution | unchanged | ~flat | ~flat, if the FM is as input-invariant here as on natural text |
| burst → continue | off-distribution | pulled toward noise | ~flat | rises with no reducible trend — the noisy-TV signature |
| drift → track | in-distribution | learning the new cell | both readers move | rises then falls |

So the FM is not the event-typer; it is **the grader of the op taken** — did the response produce
self-change, and was it reducible? — which is exactly where `performance_error_is_the_bridge` and the
two-timescale loop place it. It is also of a *different type* from both readers, which is the
principled answer to the mirror worry rather than a third NTP head. And because it reads reducibility
from the learner's own trajectory, it is a candidate replacement for the ABBA paired trial (which buys
the same information by spending a `C K K C` block of real training on every ambiguous event).

Two FM variants, both logged from Gate 0 on: **`e_online`** (FM learns alongside the reader; its
residual is self-change *minus its own catch-up*, i.e. the reducibility read) and **`e_frozen`** (FM
frozen at the last pre-event checkpoint; reads raw self-change with no lag). Their difference is the
FM's lag, which is part of the signal, not noise — keep both. FM reads are priced like every read in
`teacher_slot` (one FM forward pass per window; bench it).

**Regime caveat, stated up front.** Every FM measurement in `conditional_revision` and the
confabulation line was against a *frozen* `M`. Here the FM sits over a reader whose weights are moving
under it, and the "burst → skip is ~flat" row is a prediction from `OOD_ROBUSTNESS` (natural text, a
1–3% FM) rather than a measurement on a corrupted RHM span. Gate 0 exists to measure both before
anything is built on them.

## Design

**World**: `fourwall/lm` exactly (v16/s2/L6/m4, rule_seed 0, key = level-2 node 0, 8L/8H/256D,
20k × 64, one worker per arm, zero global RNG). Fidelity gate: a fixed-policy arm must reproduce
`fwlm0`/`fwlm1`/`tsdB` twins at max|Δ| = 0.0 on the five instruments, as `teacher_slot` did. The FM
must be attached without consuming global RNG or touching the reader's forward pass (read activations
under `no_grad`; own optimizer), so an arm with an uncharged FM stays bit-identical to its donor —
that is the FM's transparency gate.

**Events** (Gate 1): the `wall` rotation schedule (identity to 8000, +4 mod 16 every 2000, identity
return at 14000) **interleaved with span-local noise bursts** at steps the rotations don't occupy —
e.g. 250-step bursts at 5000, 11000, 17000 — each replacing indexed-span tokens with uniform draws
at a rate tuned (Gate 0) so the keyed reader's indexed-span NLL spike matches a rotation's. Uniform
corruption is suggested because its excess over exact Bayes is irreducible by construction, which is
what makes "skip" the correct op; if the implementer finds a cleaner irreducible construction (e.g.
resampling from the exact predictive marginal ignoring the latent), that is fine — state the choice
and its Gate 0 calibration. Bursts are in the *consumed* stream, so a learner that doesn't skip
trains on them.

**Actions**, per 125-step checkpoint window, reversible as in `teacher_slot`: `continue`, `merge`
(collapse `w` to the neutral filler from here), `skip` (learning rate 0 on the window; or a
precision down-weight — `continue` at reduced lr — if the implementer prefers a graded form; say
which).

**Policies** — Factor T picks the op at the instant; Factor M grades it over the next windows and may
revert or switch. Keep every rule thermostat-grade with dead zones at measured floors, in
`teacher_slot`'s spirit; no learned policy here.

| arm | T (instant) | M (window) | what it isolates |
|---|---|---|---|
| `pair` | `D_pair` → typed op map | none | typing alone, two readers |
| `self` | `D_self` → typed op map | none | typing alone, one model's counterfactual read |
| `sur_merge`, `sur_skip` | `nll` → one fixed policy for every spike | none | the surprisal twin can't type; run each policy it could pick (`sur_track` = donor `wall`) |
| `pair_fm`, `self_fm` | as above | FM `δ` grades the op: non-reducible self-change → revert to `skip`; reducible → hold | does the meter catch typing errors |
| `sur_fm` | `nll` spike → `continue` for one window | FM reads reducibility → non-reducible → `skip` for the burst; reducible → `merge` | **the FM factor alone**: can the meter substitute for the second basis, with no counterfactual read at all |
| `pair_abba` | `D_pair` | ABBA paired trial in place of the FM | the trial-based meter, priced, for the FM to beat |
| `sched` | — | — | `merge_8000` re-run in-tag with the burst schedule |
| `rand` | — | — | fires ops at the treatments' realised rate on random windows |

References that need no re-run: `wall`, `no_wall`, `true_wall`, `merge_8000` (pre-burst checkpoints
are bit-identical by construction — verify at the gate). Note the `sur_fm` mapping "reducible → merge"
will mis-merge on a reducible drift; that is deliberate and is what Gate 2 measures.

**Grading** in three currencies, kept separate as the arc always does:

1. the next-level currency: token-pathway d4 (`rand`/`none` probe, ceiling 0.921) and its
   probe-free twin; d5 trajectory if cheap;
2. the within-level ledger: indexed-span NLL lifetime integral vs `no_wall` at matched steps;
3. **per-event cost of the op taken**, in nats against the exact oracle — the new quantity. Each
   event has a correct op; the cost of the op actually taken (post-event NLL excess over the arm
   that took the correct op, integrated to recovery) is the price of a typing error. False-fire
   rate on bursts, missed-merge rate on rotations, and — for the M arms — revert latency and
   wrong-revert rate are the discrete summaries. Read prices (FM passes, counterfactual passes,
   ABBA blocks) are deducted from the shared budget as in `teacher_slot`.

## Gates

**Gate 0 — the shadow panel (2 workers, ~2–3 GPU-h).** Train `wall` and `no_wall` in lockstep on the
burst-augmented stream with `D_pair`, `D_self`, both NLLs, and **both FM residuals** (`e_online`,
`e_frozen`, with `b` and `δ`) logged at every 125-step checkpoint, uncharged, plus the same gauges on
*shadow* burst batches evaluated but not consumed (so the irreducibility calibration is model-free).
Transparency gates: the panel moves nothing (`--shadow` vs `--no-shadow` trajectory max|Δ| = 0.0),
and the attached FM moves nothing. This is `conditional_revision` Gate B restated on this substrate,
for both factors at once:

- *T*: do `D_pair` / `D_self` separate rotation from burst at matched indexed-span surprisal? If
  `D_self` already separates, Gate 1 can drop the second reader; if not, that is a finding about what
  a single model's counterfactual self-read can see, and Gate 1 runs both.
- *M*: does `e` stay near its benchmark on a shadow burst (input-invariance holds on this world) and
  rise on a consumed one and after a rotation (self-change is read)? Since `wall` tracks every
  rotation, the shadow panel already contains the "rotation → track" and "burst → continue" rows of
  the table above for free; "burst → skip" needs one skip-scheduled arm, which can be the third worker
  if budget allows.

Calibrate the burst rate here. Report both factors' separations side by side; the point of Gate 0 is
that either factor can be dropped or reshaped for ~3 GPU-h before the crossed design is built.

**Gate 1 — the crossed loop (~9 arms, ~1.5–2 GPU-h each, DoP ≤ 4 per wave).** Wave 1: the T-only
arms (`pair`, `self`, `sur_merge`, `sur_skip`) plus `sched`/`rand`. Wave 2: the M arms (`pair_fm`,
`self_fm`, `sur_fm`, `pair_abba`), with wave 1's typing results in hand so the op maps are informed.
If Gate 0 has already settled `D_self` vs `D_pair`, drop the redundant cells. Single seed; `fourwall/lm`'s
dead-pair floors carry over and should be re-measured in-tag under the burst schedule (a burst may
change the placebo floor).

**Gate 2 — the third event type (extension, after Gate 1 reports).** Add `transpose`-style level-3
rule-cell drift at steps the other events don't occupy. Drift and burst both move both input bases at
the instant, so Factor T alone cannot separate them; what separates them is reducibility over the
following window, which is Factor M's whole job. This is where `pair_fm` vs `pair_abba` becomes the
head-to-head: FM-read reducibility against trial-based reducibility, at their respective prices.

**Gate 3 — endogenise (after Gate 1).** Fold the second reader into the learner: one trunk, a
key-dropout head, `D` computed inside and acting on the model's own input stream. The live risk is
the mirror problem — a shared trunk may not keep the second basis alive, since the keyed model
suppresses the parse pathway. The FM is the natural comparison here: it already lives *outside* the
trunk by construction, so if the shared-trunk `D` loses the separation Gate 0 found for `D_pair` while
`δ` keeps grading, the architecture requirement has been located to the input-basis factor
specifically. Neither outcome should be assumed.

## What the contrasts isolate

- `pair` / `self` vs `sur_merge` / `sur_skip`: whether typing buys anything over the best single
  policy a surprisal gauge can run. Note `sur_merge` merges at the first spike and then has no key
  left to lose on later bursts — its cost shows up on the *first* burst and as the lost index, not
  repeatedly; the per-event ledger has to be read with that in mind.
- `pair` vs `self`: whether the input-basis discrepancy has to come from a separate system.
- `X` vs `X_fm`: whether a meter of a different type catches what the typing gauge gets wrong, at
  what price, with what latency.
- `sur_fm` vs `pair`: whether self-modeling over time can *substitute* for a second input basis at
  the instant — the FM factor standing alone. If it can, the two-system story's second basis is the
  self-model, not a second reader; if it can't, the FM's role is confined to grading.
- `pair_fm` vs `pair_abba`: the FM against the trial-based meter it is meant to replace.
- `pair` vs `sched` and vs `teacher_slot`'s `outer_path`: whether event-typing adds anything over an
  always-on counterfactual currency. It may not, on the merge half — `outer_path` merges pre-rotation
  and is then immune to rotations. The typing claim is then carried by the burst (and drift) cells.
- `rand`: whether the ops themselves, fired blind at the same rate, explain any gain.

Per repo norms nothing here pre-registers what an outcome means. Readings that are *not* exclusive
and should all be kept in view: the typing gauge may work and still lose the within-level ledger (the
arc's standing dissociation); `D_self` may separate the events while the keyed model's parse basis is
too hollow to route through after a merge (separation and usability are different properties); the
FM may read self-change cleanly and still be too slow to revert an op before the damage is done
(latency is a result, not a defect); and the FM may fail input-invariance on corrupted spans, which
would be a real scope limit on `OOD_ROBUSTNESS` worth knowing on its own.

## Run order, cost, delegation

Gate 0 first; it can close or reshape either factor for ~3 GPU-h. Then Gate 1 in two waves at DoP ≤ 4.
Gates 2–3 only after discussing Gate 1 with Jasper. Delegate the whole build loop per gate to one Opus
subagent (script, smoke on Modal attached, launch detached via a session-isolated launcher as the
donors do, reduction and figures); keep the wait and the interpretation. Expect ~2 halts per gate.
Subagents do not touch git state. Donor code is imported, never edited; fork what must change into
this folder. Results land on `rhm-scaling-data` (`chromatic`) under `/data/rhm_practice_tuning/<tag>/`;
fetched copies under `figures/<tag>/`. README lands after the run and after discussing results.

## Caveats to carry from the start

- Single seed on every treatment; the dead pair supplies the floor. Ranks, signs, trajectories and
  floor-multiples are the claims, as throughout the arc.
- The burst construction is the node's real design cost. If the burst rate needed to match a
  rotation's spike is large enough to change what the parse reader learns, the "both readers move
  together" signature is confounded; Gate 0's shadow batches (evaluated, not consumed) exist to
  separate the gauges' response from the learner's.
- The FM over a learning reader is a new regime for this repo's FM line (every prior measurement was
  against a frozen `M`); the online FM's lag is part of its signal, which is why `e_frozen` is logged
  beside it. The FM's own learning rate is a free parameter that should be swept once in Gate 0.
- Weak-form scaffold regime only (the true key is representable and cheaply extractable, Gate 0's
  0.921), one key level and node, one loudness — inherited from the donor.
- `merge` is irreversible in effect even when reversible in mechanism (`fwlm1` finding 5: the
  abandoned circuitry evaporates within ~250 steps), so a false merge on a burst is a permanent
  loss of the index; price it as such. The M arms' `revert` can therefore only ever save a wrongly
  *skipped* rotation or a wrongly *continued* burst, not a wrongly merged one.

## Addendum — Gate 1 reshaped on Gate 0's panel (2026-08-30, agreed with Jasper)

Gate 0 (`tn0`; reduction under `figures/tn0/`, ledger entry in `ROADMAP_PROGRESS.md`) and a
parallel-session note on Garcia-Garcia et al., *"Granule cells reorient cortical manifolds to
separate contexts but preserve their geometry"* (bioRxiv 2026.03.03.709240;
`reading/2026.03.03.709240v1.full.pdf`[^private])
reshaped the crossed loop before its build. The measured
basis, computed retroactively from `tn0`'s own panel (68 bracketed quiet mature checkpoints,
matched indexed-span surprisal via the ladder):

- **The KL magnitudes were the wrong contraction of the panel.** Movement under the event, per
  view, types the pair: Δself (the key-neutralized read's own NLL movement) is exactly 0 under a
  rotation at every checkpoint vs +0.596 median (min +0.269) under a matched burst; Δparse
  likewise 0 vs +0.558. The one-model counterfactual **suffices in decomposed form**; as KL
  magnitudes (`D_pair` 1.1–1.3×, `D_self` anti-separating) it did not. The rotation-side zero is
  by construction — any key-free view is blind to a key permutation — so the claim is that
  holding such a view makes rotation-vs-world-news typing trivial at the instant, not that the
  instrument did nontrivial work on this pair.
- Combined with out-of-span NLL (a rotation leaks +0.073 off the span; a burst ±0.0001; a
  level-3 drift plausibly leaks), the instant triple **T\* = (Δkey, Δself, Δout)** may type all
  three events from one model: Δself ≈ 0 → rotation; Δself > 0, Δout ≈ 0 → burst; both > 0 →
  drift. The sharpest open question Gate 1 now measures: whether a **hollow** key-free basis
  (mirror control ~60% of `D_pair`'s clean level; d4 0.27) still *registers* drift — structural
  news one level up — where a burst is surface corruption any reader sees. The drift row of T\*
  is a plausibility to test, not a prediction.
- **Factor M types by sign**, not by reducibility-trend: rotation (reorientation) raises
  `e_online` +25–39% and recovers; a consumed burst (degradation) *lowers* it −18…−30%; skip is
  flat; `e_frozen` orders the events oppositely (+68…+187% on bursts), so the online-vs-frozen
  difference is itself part of the signal. M-arm op maps read sign first, magnitude second.
  FM lr 3e-3 (Gate 0's sweep), not the donor 1e-3.

Changes in force for Gate 1:

1. **Drift moves into wave 1** (from Gate 2): `transpose`-style level-3 rule-cell drift at steps
   the other events don't occupy, with a drift ladder so the matched-surprisal discipline covers
   the third event. Drift calibration is the implementer's design cost, as the burst's was.
2. **Arm map v2.** Wave 1: `self` = T\* → typed op map (the headline arm, one model);
   `pair` = the same op map from the two-reader movement read (whether a separate system adds
   anything — now a control, not the headline); `sur_merge`/`sur_skip` unchanged; in-tag
   baselines re-run under the full event schedule (track-everything, `no_wall`-equivalent,
   `sched`, `rand`, the dead pair for floors) — `tn0`'s arms do not share the new stream.
   Wave 2, informed by wave 1: `self_fm` (FM sign grades the op, reverts what it can),
   `sur_fm` (the FM factor alone, sign mapping), `self_abba` (the priced trial meter for the FM
   to beat), `self_rekey` (below). DoP ≤ 4 per wave.
3. **The re-key op joins the action set** as a supplied op: on a rotation-typed event, re-key
   instead of merge — preferably the priced candidate search over the rotation family (re-key
   onto the map minimizing one window's NLL; a milder oracle leak than supplying the answer;
   implementer's call, stated either way), with an admissibility control priced like
   `merge_dead`. It zeroes the recurring rotation debt, isolating merge's case to the
   withheld-pathway currency — fourwall's mothball/re-key, supplied at last. The identity
   return at 14000 gives its saved-map payoff a free read.
4. **Checkpoint the readers** (`torch.save` at panel checkpoints; both readers + FMs; the
   implementer bounds volume cost — stride/precision their call) so hidden-state geometry reads
   (RSA / Procrustes angle — the reorient-vs-degrade instrument the paper motivates) never
   again need a replay. The Gate 0b geometry replay is *held* unless the drift/hollow-basis
   cells say shape is load-bearing.
5. **The movement decomposition folds into `analyze_tuning.py`** and the standing panel;
   `tn0`'s reduction re-emitted with it (retroactive, CPU).

Everything else above stands, including the grading currencies, the transparency gates, and the
caveat list. Per repo norms nothing here pre-registers what an outcome means.

[^private]: Not mirrored: this link points to a document in the private lab repo (the roadmap, the queue, an unrun spec, reading notes, or a conversation). See the top-level README for what is held back and why.
