# Practice manufactures its own credit: allocation, metering, and re-chunking — a three-component definition of practice

**Status**: Proposal / conceptual synthesis (2026-08-12). Nothing new run. Every load-bearing component cites an
existing result in this repo; what is new is the *assembly* and one named operation (re-chunking / compilation) that
no node has built.
**Date**: 2026-08-12
**Prompt**: a conversation with Jasper (2026-08-11/12) reading the bridge results against classical-piano practice
phenomenology. The piano anchors (Rubinstein, the overspeed drill, the teachers' dictum) are first-person reports,
used here as anchors for the *shape* of the theory, not as evidence.
**Builds on**: [performance_error_is_the_bridge.md](performance_error_is_the_bridge.md) (δ and the matched-filter
synthesis, §14), [two_timescale_value_loop.md](two_timescale_value_loop.md) (learning/meta layer split; the
allocation machinery), [cerebellar_abstraction_ratchet.md](cerebellar_abstraction_ratchet.md) (the ratchet; sleep as
the distillation window)
**Experiment anchors**: [`agency_gate`](../experiments/mjc/agency_gate/README.md) ·
[`plasticity_gain`](../experiments/mjc/plasticity_gain/README.md) ·
[`bridge_assembly`](../experiments/mjc/bridge_assembly/README.md) ·
[`two_clocks`](../experiments/mjc/two_clocks/README.md) ·
[`benchmark_vs_cost`](../experiments/mjc/curiosity_control/benchmark_vs_cost/README.md) ·
[CURIOSITY_DRIVE](../experiments/a2a_forward/reaching/CURIOSITY_DRIVE_README.md) ·
[DISTILLATION](../experiments/a2a_forward/DISTILLATION_README.md) ·
[GATED_RATCHET](../experiments/a2a_forward/GATED_RATCHET_README.md) ·
[`one_layer_deeper`](../experiments/one_layer_deeper/README.md)

## One-liner

Practice is not a learning rule. It is the closed-loop behavioral regime that **manufactures the conditions under
which a performance signal can exist and be consumed**: (1) **allocation** makes the benchmark estimable by making
contexts recur, (2) **metering** (δ) assigns benchmark-relative, agency-gated credit at the current unit level, and
(3) **re-chunking** coarsens the units when credit at the current level goes silent — re-instantiating (1) and (2)
one level up. The "lift" between levels requires nothing exotic: it is an ordinary distillation step (**compilation**
of one's own reactive trace into a committed ballistic unit), gated by δ-silence.

## 1. The definition

| component | acts on | signal | machinery already in repo | status |
|---|---|---|---|---|
| **Allocation** (policy) | the *conditions* of learning: make a context recur at a rate fast relative to competence drift, against a stable target — the estimability condition for `b(s)` | ensemble disagreement — **not** δ ([`benchmark_vs_cost`](../experiments/mjc/curiosity_control/benchmark_vs_cost/README.md): `b−e` is a gain signal, actively repulsive at re-opening as an allocation score) | [CURIOSITY_DRIVE](../experiments/a2a_forward/reaching/CURIOSITY_DRIVE_README.md) Phase 2b | exists; never wired to the bridge |
| **Metering** (credit) | the *model*: δ = (b(s)−e)·σ((g−g₀)/θ) as a per-sample gain on plasticity — funds repair, protects mastered content, withdraws from unimprovable noise | δ | [`agency_gate`](../experiments/mjc/agency_gate/README.md), [`plasticity_gain`](../experiments/mjc/plasticity_gain/README.md), [`bridge_assembly`](../experiments/mjc/bridge_assembly/README.md) | built and dissected, 3 seeds |
| **Re-chunking** (representation) | the *variables*: compile a mastered reactive sequence into a single committed ballistic unit; the chunk becomes the new "syllable" and (1)+(2) re-instantiate at its level | δ-silence (sustained `b−e ≈ 0`, low variance) as the trigger | the distillation op ([DISTILLATION](../experiments/a2a_forward/DISTILLATION_README.md)) + the ballistic controller line ([`ballistic`](../experiments/mjc/ballistic/README.md)) | never built — the new component |

The learning rule is only component (2). What makes practice *practice* rather than mere learning is that (1) and
(3) act on the **conditions of learning** instead of on the model: allocation manufactures the recurrence the
benchmark needs; re-chunking manufactures the units that credit can address. Jasper's phrasing, sharpened: practice
is a **change of variables in what the credit system can address**. When the units coarsen, everything
re-instantiates at the new level — the chunk is the thing that recurs, the thing with its own context-conditional
benchmark, the thing executed ballistically (agency = the efference copy of the *chunk* command) and evaluated at
its boundary.

This slots directly into §14 of [performance_error_is_the_bridge](performance_error_is_the_bridge.md): *performance*
is what a matched reader extracts from the FM's error stream; *practice* is the behavior that creates and maintains
the matching conditions. And it extends the round's division of labor by one term: **disagreement allocates, δ
meters, reward masks — and distillation compiles.**

The phenomenology of deliberateness falls out of the layer split
([two_timescale_value_loop](two_timescale_value_loop.md) (c)): only component (1) is deliberate (meta-layer
allocation policy — choosing what to drill); (2) and (3) are learning-layer substrate reorganization. That is why
there is nothing reasoning-shaped about going from sightreading a piece to performing it from memory: you experience
selecting the passage, and the improvement arrives non-deliberately.

## 2. Hierarchy is forced, not chosen — two derivations

The claim that practice is hierarchical (16th-notes → quarter-notes → measures; motor chunking) does not need to be
assumed. Two results we already have each force it independently.

**(a) Benchmark convergence silences credit at the mastered level.** Once `b(s)` converges at the primitive level,
`b − e ≈ 0` on every keystroke — you do not get better at pressing individual keys; that level is at ceiling and δ
is silent there. The remaining reducible error lives in the *seams* — transitions, coordination between primitives —
and seams are not primitive-level contexts, so primitive-level credit structurally cannot see them. The only way to
make seam-error addressable is to define units at the seam scale. The frontier climbs because the performance signal
at the mastered level has nothing left to say. (Whether errors and pauses empirically concentrate at chunk
boundaries is a motor-chunking literature question — flagged in §10, recalled as yes, unverified.)

**(b) Delay variance forces chunk boundaries.** An eligibility window is a delay hypothesis that can match a fixed
delay, not a stochastic one ([performance_error_is_the_bridge](performance_error_is_the_bridge.md) §7;
[`two_clocks`](../experiments/mjc/two_clocks/README.md): the credit-fidelity peak sits at τ = d and moves with d).
Along a composed open-loop chain, error-return timing gets noisier with depth, so per-primitive credit becomes
illegible as sequences lengthen. Chunk boundaries are periodic **re-grounding points** — the places where a matched
filter can be rebuilt (the reactive controller's surface-average re-grounding,
[`bridge_assembly`](../experiments/mjc/bridge_assembly/README.md), applied at the boundary instant). Ballistic
within, evaluated at the boundary.

The mature state of this process is the bird: a crystallized song **is** a chunked system, and Gadagkar's
per-syllable, temporally-aligned benchmark **is** chunk-level credit. The biology was showing the end state of
practice all along.

## 3. The lift is two unexotic processes, with δ-silence gating the second

- **Content formation** — δ-metered reactive execution drives `b − e → 0` at level k. This is the bridge machinery
  as built.
- **The committability certificate** — `b − e → 0` is not just "now you are good"; it is the condition under which
  open-loop execution becomes *viable at all*. A ballistic controller is an FM bet — a deep, narrow line-integral
  along a committed corridor ([`bridge_assembly`](../experiments/mjc/bridge_assembly/README.md) §Q1;
  [`ballistic`](../experiments/mjc/ballistic/README.md) 4b) — and the bet only pays when corridor error is low,
  because within the chunk there is no feedback to catch you. The handoff is not a decision that follows mastery;
  it is an option that mastery unlocks.
- **Compilation** — distilling one's own repeated, low-variance, correct traversals into a single committed
  open-loop unit: self-imitation of one's own reactive trace. The operation itself is demonstrated in this repo:
  [DISTILLATION](../experiments/a2a_forward/DISTILLATION_README.md) absorbed the FM's contribution into the model's
  own weights at a cost of +0.030 nats (dependency gap closed 105.6%) — absorption is cheap and works. Kim, Parvin
  & Ivry's attenuation-on-success supplies the natural preamble: success freezing plasticity is exactly what you
  want immediately before compiling, so you do not keep editing the thing you are about to commit.

After compilation, the chunk is the new syllable and the credit machinery re-instantiates one level up: chunk-level
contexts, chunk-level `b(s)`, chunk-level agency (the arity gap of the chunk command), evaluation at the boundary.
Neither process is exotic and both already exist as code; what has never existed is the **gate between them**
(δ-silence triggering consolidation) and the **re-instantiation after** (credit rebuilt at the compiled level).

## 3½. Where the chunk lives before consolidation: the FM is the holding structure

The question this section answers (raised by Jasper, 2026-08-12): does compilation deposit the quarter-note
instruction directly in cortex, alongside the 16th-note instructions — or does the new chunk need a
hippocampus-analog to hold it in a holding pattern until sleep integrates it?

Neither, exactly: **the holding structure exists, and it is the cerebellar side — the FM itself.**
[DISTILLATION](../experiments/a2a_forward/DISTILLATION_README.md) already has the full shape of the mechanism:
during wake, the composite system (model + FM injection) outperforms the model alone — the FM is *carrying* content
the cortex does not yet own, live in the loop; sleep trains the injection-free student to match the injection-active
teacher; the dependency gap closes (105.6%, at +0.030 nats vs open-loop). Mapped onto §3: in the interim state the
new chunk exists in cortex only as a thin command token, and the corridor it unpacks into lives in fast cerebellar
weights — which is just what ballistic execution already is, an FM bet along a committed corridor. Compilation
happens *in the structure that executes the chunk*; no third structure is required. DISTILLATION Phase 3 even shows
the step after: post-consolidation, a fresh FM finds *different* innovation structure — the cerebellum re-pointing
at the reorganized cortex, which is §1's re-instantiation-one-level-up seen from the FM's side.

**Why no episodic buffer is needed.** The complementary-learning-systems argument for the hippocampus (McClelland,
McNaughton & O'Reilly) is that fast cortical learning of *arbitrary one-shot* content catastrophically interferes,
so a fast binder plus interleaved replay into the slow learner is required. Chunk content is the opposite regime by
construction: massively recurrent, low-variance, and pre-regularized by having driven `b−e → 0`. The recurrence
that makes `b(s)` estimable is the same recurrence that makes direct slow consolidation safe — **repetition is the
interleaving; practice manufactures its own consolidation safety.** The two fast learners then split cleanly by
data regime: the hippocampus is the fast store for content that *cannot recur* (episodes, n=1, arbitrary binding);
the cerebellum is the fast store for content that *recurs by construction* (procedures). The classic amnesia
dissociations are the empirical backing: hippocampal patients acquire motor skills at normal rates across sessions
while having no episodic memory of having practiced (§11 — recalled as well-established, to be cite-checked).

**Routing, not pruning.** The primitive level is not deleted after compilation, and should not be: what is scarce
is the serial closed-loop channel (attention), not storage. Automatization changes the *default routing* — the
chunk becomes the policy's unit and the primitive level stops being consulted per step — while the primitives
persist as the fallback (an expert can always drop back into slow practice, i.e. deliberately re-enter the reactive
regime). Choking under pressure is the tell that they persist: explicit monitoring mid-performance re-engages
per-primitive control exactly where per-primitive credit and correction are illegible (§5's tempo threshold), and
execution breaks — nothing pruned-away could be re-engaged. On the DISTILLATION evidence, what sleep prunes is
**dependency** (reliance on the FM's carry — the literally measured quantity), not representation.

**The honest residual: the target register.** One object in the loop genuinely has the one-shot, must-hold
profile — the *target*. The benchmark requires a stable target held across the whole practice window, and
ecologically that target is often acquired in a single exposure: the songbird memorizes the tutor song fast, then
practices against the stored template for months; a pianist reads the score once and holds what-it-should-sound-like
for weeks. In every experiment in this repo, that role is played by the task specification — *the experimenter is
the hippocampus*. So the information-processing necessity reduces to a small, stable target register; nothing about
the loop requires that register to be a large episodic memory system. This *sharpens* rather than threatens the
repo's standing premise that the hippocampus is an artifact of biological necessity more than
information-processing necessity: on this account it is load-bearing for target *acquisition* (one-shot capture in
a world where the tutor does not repeat himself), inert for the loop itself.

Where the uncertainty genuinely lives: **reasoning-chunks.** The replay/schema-formation literature is heavily
relational; cognitive material recurs far more thinly than motor material; and first-person phenomenology includes
rest-time replay episodes that rehearse recently-practiced material unprompted (Jasper's report, presumably
hippocampal). Two readings stay open: hippocampal involvement may be real and load-bearing for chunks whose
recurrence is thin — with §6's drill/schooling technology being precisely the machinery for moving content *out of*
the episodic regime *into* the practice regime — or rest-time replay may be a **scarce-data adaptation** (flagged
by Jasper): a rehearsal amplifier biological learners need because their trials are metered, which machine learners
with abundant recurrence largely do not. That connects it to
[meta_learning_under_metered_data](meta_learning_under_metered_data.md)'s sample-price condition, and it is a
question for the literature pass, not a claim.

## 4. Why the wake/sleep ratchet didn't ratchet on its own — a retrodiction

Flagged by Jasper: the wake/sleep distillation work predates all of this vocabulary, and the missing pieces are now
nameable. Stated carefully, because the prior record is more nuanced than "no ratchet":

- [GATED_RATCHET](../experiments/a2a_forward/GATED_RATCHET_README.md) **did** compound (WS_LG val-loss gap over
  open-loop widening −34% → −48% across 4 cycles, monotone) — but its own interpretation is **implicit
  regularization**: the model becoming progressively more legible to a compressed self-model *within a fixed
  distribution*. Its gate **opened** rather than closed (0.31 → 0.77), and the README's own diagnosis is that with
  inner and outer distributions identical there was *nothing to protect* — no mastered content under threat, no
  novelty to discriminate.
- [DISTILLATION](../experiments/a2a_forward/DISTILLATION_README.md) on language found the residual got more
  *diffuse* after consolidation and named the reason: "the absence of a discrete abstraction to extract on flat
  webtext."

Under §1's vocabulary, three pieces were missing, and they are the three components: **(i)** consolidation was
*scheduled*, not gated by any committability certificate — δ-silence did not exist as a concept, so sleep compiled
whatever was there, mastered or not; **(ii)** the substrates (flat webtext, i.i.d. MNIST) had no recurring practiced
contexts against stable targets — nothing chunk-shaped recurs, so no `b(s)` could converge *on a unit*, so there was
nothing committable to find; **(iii)** nothing re-instantiated credit at the new level after consolidation — the
sleep loop was consolidative, not evaluative ([two_timescale_value_loop](two_timescale_value_loop.md)'s own phrase).
This is a retrodiction, not a demonstrated diagnosis — §8 Experiment 4 is its test. Consistent with this repo's
norms: the prior result was a data point generated under different priors, not a verdict on the mechanism.

## 5. The two-knob map of practice technique

Practice technique is a setting of two knobs:

- **Tempo** selects the *controller regime* — the legibility knob. Below the threshold where the error-return delay
  fits inside the inter-primitive interval, execution is reactive and per-primitive credit is collectible; above it,
  execution is ballistic whether you like it or not.
- **Strictness** selects *which dimensions collect credit* — which constraints are graded, and therefore which
  errors trigger δ-driven plasticity or reactive intervention.

The named techniques are corners of this space:

| corner | regime | credit | what forms | phenomenological name |
|---|---|---|---|---|
| slow + strict | reactive; per-primitive credit legible | full δ, all dims | chunk **content**, cleanly | Rubinstein's "perfectly, at slower tempi" |
| overspeed + lax | ballistic **forced** — the reactive fallback is physically unavailable | deliberately **suspended** (lax grading keeps δ silent on dims that would bake corrections into a still-fragile unit) | **commitment** — the controller switch | the "load the chunk" drills (Jasper's report: above performance tempo, deliberately not grading rhythm evenness) |
| at-tempo + strict | ballistic, evaluated at boundaries | chunk-level δ | level-k+1 credit | run-throughs, performance |
| at/above tempo + erroneous content, repeated | ballistic — consolidation-eligible | demanded but illegible | **compiled error** | sloppy practice — the poisoned region |

Almost every path through this space is valid — the piano-teacher consensus that there is no wrong way to practice
except sloppily — because each corner trains a different component: content (slow+strict), commitment
(overspeed+lax), or the next level's credit (at-tempo+strict). The one destructive region is repeated ballistic
execution of wrong content: the execution is consolidation-eligible and the content is wrong, so **the chunk learns
the mistakes**. Sloppy practice is not "insufficient care"; it is *compilation of uncorrected error*. And the
below-tempo → at-tempo shift is the most universal technique because it is the canonical trajectory through the
map: form content in the credit-legible regime, then hand off. (Sports appears to have converged on the same
overspeed trick — sprint towing, over/under-weighted throwing programs — flagged for the literature pass.)

## 6. Rarity: the regime-dependence result at the scale of a life

[`bridge_assembly`](../experiments/mjc/bridge_assembly/README.md)'s honest headline — ungated uniform plasticity
wins short/easy recovery outright, δ pays only where recovery transients are long/hard — is plausibly why dedicated
practice is *rare* in human life. Most everyday skill acquisition lives in the short/easy regime, where the flow of
experience suffices and manufactured recurrence would be pure overhead. Practice pays exactly on the long/hard
frontier — and there the estimability conditions do not occur naturally, so they must be **manufactured**: études,
scales, drills, and problem sets are cultural technology for producing recurring contexts against stable targets
(the metronome is literally a target-stabilization device). Same exaptation shape as
[two_timescale_value_loop](two_timescale_value_loop.md) (b)'s schooling-as-scaffolding. Students drilling proof
techniques on problem sets are practicing in exactly the formal sense of §1 — a problem class is a manufactured
recurring context — and the reason it does not feel like piano practice is that reasoning-chunk consolidation is
phenomenologically silent: you notice the drill, never the re-chunking.

## 7. Offline consolidation and the replay filter (the speculative edge)

Two nested claims, in decreasing order of confidence, and deliberately kept short — this is the closest thing to
speculation in the current line:

1. **Structural**: offline rehearsal has no world in the loop, so only committed open-loop units can be *executed*
   offline at all — reactive-level content has nothing to react to. If offline replay contributes to compilation,
   the content it can rehearse is the committed kind.
2. **Timescale** (flagged by Jasper): replay is reported at compressed/enhanced timescales relative to waking. At
   compressed speed, closed-loop timing cannot be honored even in principle, so replay speed would act as a natural
   *filter selecting compiled chunks* — the reactive level drops out. This meshes with
   [cerebellar_abstraction_ratchet](cerebellar_abstraction_ratchet.md) §4 (sleep as the distillation window; spindles
   through VL, the cerebellar output relay), and with the standard pianist report that a passage which would not gel
   at night works the next morning.

Worth mentioning, not worth leaning on: none of §1–§6 depends on it, and the replay-timescale literature (SWS vs
REM compression factors, sleep-dependent motor consolidation) needs an actual pass before this paragraph is more
than a pointer.

## 8. Experiments

Ordered by cost. Substrates: the mjc ballistic line ([`plasticity_gain`](../experiments/mjc/plasticity_gain/README.md)
/ [`bridge_assembly`](../experiments/mjc/bridge_assembly/README.md) world) and
[`one_layer_deeper`](../experiments/one_layer_deeper/README.md) (purely ballistic, exact ground truth at every
rollout step — the composition-depth instrument).

### Experiment 1 — the practice loop (components 1+2 wired together)

Disagreement-driven allocation chooses *which* context to revisit; δ meters plasticity *within* the visited context;
trials are closed-loop (fixing [`two_clocks`](../experiments/mjc/two_clocks/README.md)'s scripted-trial caveat).
Graded on **within-level compounding across a drift sequence** — does the combination hold/recover competence across
successive re-openings where neither component alone does? This is the direct test of §1's claim that allocation and
metering are separable components of one loop.

### Experiment 2 — credit level vs composition depth

Compare primitive-level δ against boundary-level (chunk) δ as composed sequence length grows.
**Prediction** (from §2(b)): per-primitive credit degrades with depth as error-return delay variance accumulates;
boundary-level credit survives. `one_layer_deeper` is the clean instrument — exact per-step ground truth lets
within-chunk vs seam error be measured directly rather than proxied.

### Experiment 3 — the overspeed manipulation (compilation isolated)

Force open-loop execution at horizons beyond the reactive controller's re-planning rate while suspending δ on
off-corridor dims; compare ballistic-unit formation against continued reactive practice at matched samples.
**Prediction**: forced commitment accelerates the reactive→ballistic handoff *on already-mastered corridors*
(where the committability certificate holds) and compiles error on unmastered ones — §5's poisoned region,
manufactured deliberately as the control.

### Experiment 4 — δ-silence-gated consolidation (the ratchet retrofit)

Re-run a wake/sleep cycle where consolidation is *triggered by sustained δ ≈ 0 on a practiced context* rather than
by schedule, and where credit re-instantiates at the compiled level afterward (chunk-level benchmark on chunk-level
contexts). This is §4's retrodiction made testable: does mastery at level k now open level k+1 in a way scheduled
distillation did not? Worth building only after 1 and 3 land — it needs both the loop and the compilation op.

### Sequencing

1 and 3 are independent and can run in parallel; 2 sharpens 3's grading; 4 composes 1+3.

## 9. Risks

Held loosely — places we would expect to learn something, not conditions we commit to treat as falsifiers.

- **Compilation may need more than self-imitation.** A chunk-level FM might not form from reactive traces alone
  (e.g. the trace distribution is too narrow to support open-loop robustness). That would refine §3's "unexotic"
  claim, not the definition.
- **δ-silence may be the wrong gate.** Variance, not mean, might carry the committability signal — or the gate
  might need meta-layer input. The certificate concept survives either way; its estimator changes.
- **The poisoned region may not reproduce in silico.** Sloppy practice as compilation-of-error is a retrodiction
  from phenomenology; Experiment 3's control cell tests it.
- **The seam-concentration recollection may be wrong.** §2(a) leans on remembered motor-chunking findings; the
  literature pass could come back different, which would weaken derivation (a) but not (b).
- **§4's retrodiction may fail** — δ-gated consolidation might stall exactly as scheduled consolidation did, which
  would mean the missing ingredient in the ratchet was something else entirely. That would be the most informative
  negative in the set.

## 10. Honest status

Nothing here has been run. §1–§3 are an assembly of results that each individually have support in this repo; the
assembly does not — the same epistemic shape [performance_error_is_the_bridge](performance_error_is_the_bridge.md)
§11 had before its empirical rounds. The strongest ground is §2's two derivations, each of which leans directly on
a measured result (benchmark convergence and context-conditionality from the bridge nodes; delay-matching from
`two_clocks`). The weakest is §7, which is flagged as speculation and load-bearing for nothing. The piano
phenomenology (Rubinstein, the overspeed drill, the teachers' dictum, the morning effect) shaped the theory and is
recorded as anchor, not evidence.

## 11. Literature pass (recalled, not verified — titles need checking before citation)

- **Motor chunking**: whether errors/pauses concentrate at chunk boundaries and whether chunk structure reorganizes
  with expertise — Sakai et al. (chunk patterns in sequence learning); Wymbs, Bassett & Grafton (chunk
  concatenation vs segmentation). Highest priority: §2(a) leans on this.
- **Gadagkar et al. 2016 re-read** (already in `reading/cerebellum/`[^private]): does syllable
  crystallization look like boundary formation? Also songbird staging generally (subsong → plastic song →
  crystallized — Tchernichovski et al.).
- **Deliberate practice**: Ericsson, Krampe & Tesch-Römer 1993 — the manufactured-conditions reading of §6 against
  their original framing.
- **Sleep-dependent motor consolidation**: Walker et al. 2002 (overnight finger-tapping gains) and the subsequent
  replication debate — verify current status before leaning on it.
- **Replay timescales**: Lee & Wilson (SWS time-compression); Louie & Wilson 2001 (REM replay timescale — check
  whether "enhanced" is the right reading per §7.2); Nádasdy et al.
- **Overspeed training**: sprint towing / over-under weighted throwing programs — the sports analog of §5's
  forced-commitment corner.
- **Amnesia dissociations** (§3½'s empirical backing): Milner / Corkin on H.M. — mirror tracing and other motor
  skills acquired at normal rates with no episodic memory of the practice sessions. Recalled as well-established;
  cite-check before leaning on specifics.
- **Complementary learning systems**: McClelland, McNaughton & O'Reilly 1995 — the interference argument §3½
  inverts for the massively-recurrent regime.
- **Explicit monitoring / choking**: Beilock & Carr (and successors) — attention to automatized components degrades
  skilled execution; §3½'s routing-not-pruning tell.
- **Schema formation**: Tse et al. 2007 (rapid cortical consolidation when new material fits a prior schema) — the
  relational counterpart relevant to §3½'s reasoning-chunk uncertainty; also awake rest replay (Foster & Wilson)
  for the rest-time rehearsal phenomenology.

---

## 12. What we ran (2026-08-12 → 08-14): the étude arc

**Source of truth**: [`experiments/mjc/practice/etude/README.md`](../experiments/mjc/practice/etude/README.md)
— the joint writeup of the whole arc (E-gate → compile-op discriminators → E-3/E-3b → E-4 → E-5 +
3-seed replication). **This section is a pointer, not a summary of record** — read that node for
findings, tables, retractions and reproduction.

Where the doc's claims landed, in that writeup's terms: §1's re-chunking trigger (δ-silence as the
committability certificate) **works**; §3's compile op was corrected in kind — "self-imitation of
one's own reactive trace" is right only read *singular*: compilation is **selection + commitment**
of one realisation, ranked by expected performance under the consumption distribution, and
distillation-by-regression (averaging renditions) is what destroys committed units. §5's overspeed
corner **inverted** as candidate generation (closed-loop traces are the better candidates), and
sloppy practice's catastrophe was regression's property — the weak form (compiled error persists)
stands. §2's hierarchy derivations acquire a stated precondition: **boundaries must carry
information** for re-chunking to be a real operation. New, not anticipated here: the
practice→performance **seam-state shift** (audition vs consumption distribution), assembly's
benchmark-reset cost, and chain compounding. The headline: sequential assembly with seam-matched
selection dominates never-compiling on both axes, 3/3 seeds.

[^private]: Not mirrored: this link points to a document in the private lab repo (the roadmap, the queue, an unrun spec, reading notes, or a conversation). See the top-level README for what is held back and why.
