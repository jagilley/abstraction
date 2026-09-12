# SPEC — embouchure: the chooser trained on its own attempts

**Status**: spec, 2026-09-10, written as the orchestrator's prompt to the implementer (the `tempo`
pattern: this file is that prompt verbatim). Nothing built, nothing run. Leaf lane only; the class
lane and the composed fork are named at the bottom and are not this round's job.
**Parent**: [`../inflection/`](../inflection/README.md) (`embouchure.py` forks `inflection.py`).
**Sibling it answers to**: [`../enharmonic/`](../enharmonic/README.md) finding 5 (a richer book is
capped by the chooser). **Name**: the embouchure is how the same note is shaped per register, and it
is learnable only by playing and hearing yourself.
**Attribution**: the chooser as the organ both PRs point at, the reading of the max-sum DP as
inverse-by-search over a free forward model, and the biological framing (the inverse map is trained
on the animal's own attempts: feedback-error learning, DIVA's babbling, subsong) came out of the
2026-09-10 discussion with Jasper after reading PRs #101 and #102; the decision to run the leaf
lane first, the arm table and the name are the orchestrator's; the §8 argument this spec argues
with is the inflection builder's. Record: `conversation_2026-09-10.md` beside this file (written
at close-out).

---

You are the implementer for a new practice node. Read `/subagent-instructions` before you wait on
anything and `/run-experiment-on-modal` before you touch Modal. You never touch git. You halt twice
per experiment: once with the launch handle, once with the reduced results and figures. You reduce;
I interpret. No README: facts go to `FILES.md`, decisions to `DESIGN.md`, reductions under
`figures/`. `QUEUE.md` and `ROADMAP_PROGRESS.md` are mine.

## The question

On every RHM practice node the executor's realization step, which synonym to write for a feature,
was either a constant (`canon`) or, in `inflection`, a head fitted from the blocks the *world*
wrote. Every biological inverse model is trained instead on the animal's *own* attempts and their
consequences. When the renderer learns only from what the learner itself produced under the meter,
does the spelling step stay live long enough to be an execution-currency signal at the pacer's
timescale, does it still transfer to unpractised registers, and what does the source of the
training signal cost against its volume?

## Background, distilled

`inflection` (Q0–Q3, 24 GPU-h) made the bottom synonym a function of a register and made the
grader report spelling beside meaning. Fitted renderers spell at ≈0.02 written error against
0.42–0.48 for `canon` and `leaf`; a scalar-register head transfers to unpractised registers and a
one-hot one does not; a per-feature switch point is identifiable only once an interior register is
practised; on a matched clock correct spelling is worth ≈0.08 in meaning through the learner's own
planner, not the grader. Then Q3: with the fitted renderer the E′ pacer seat was *inert by
mechanism*, because the head learned the rule from the world's abundant surface far faster than the
ladder advances, so `e_spell ≡ e_feat` on every live cycle; with the unlearnable `leaf` tape the
spelling-charged series made δ-silence fire on scale, not on convergence (A1's quiet statistic is a
raw slope against a fixed absolute dead zone). `enharmonic` reached the same organ from above: the
quotient buys L5 arrival, but an all-legal 9-class book performs worse than a 3-class one at
matched clocks because the executor's expansion choice is `macro_features`' argmax over the frozen
generator's block logits, a free forward model of the surface used as a scorer, never trained on
the learner's own successes.

The inflection builder chose, in `DESIGN.md` §8, to fit the renderer from rule-spelled blocks
observed in solved instances and *not* from verdicts on its own writes, on three grounds: per-block
verdicts are a label the world does not volunteer; `rubato`'s body model was fitted on the body's
response, not a grader's opinion; restriction to solved configurations is what makes it practice.
The builder named `fit_rule_verdict` as the natural ceiling and logged the rows for it
(`log["spell_rows"]`, 512 per cycle, never consumed; `DESIGN.md` Q1.6).

My view, for you to weigh rather than adopt: the line was drawn in the right place for the
*per-block* verdict and the wrong place for the observable. The body's response in `rubato` was
the response to the learner's *own* command. Its RHM twin is not the world's blocks; it is what
happens to the learner's own written block: the grader's second number on the instance it wrote
(the meter's own feedback, already returned per row by `grade_spelled`), and what its own reader
reads back. The world's surface is the *template*, which the bird also learns by listening; the
map from intent to command is learned by singing. On this substrate the template and the map may
coincide, because writing a synonym *is* producing the sound with no articulator in between. If so,
that is a result about the substrate and you should say it with numbers rather than force an arm.

## What to build

`experiments/rhm/practice/embouchure/embouchure.py` forks `../inflection/inflection.py`. Every
addition `# [embouchure]`-marked, every knob default off, and the fidelity gate replays
`inflection.py` at 0.000e+00 with the knobs off (their G-F idiom; re-run after every change to a
shared path). Do not read the donor whole: navigate by `../inflection/FILES.md`'s touch-point map,
the `# [inflection]` marks, `DESIGN.md` §3 (the Renderer duck), §9 and Q1.5 (what `fit_rule` is
and how it is fitted: one linear layer, own generator, `Khat` cached, `blk_render` ledger) and Q3.2
(`perf_e_spell`, the mirror seat, the shadow meter). `analyze_embouchure.py` forks
`analyze_inflection.py`.

Renderer modes, beside the existing `canon`, `given`, `leaf`, `fit` (call the existing fit
`surface` in the reduction; it is the perceptual route):

- **`own_scalar`** — the production route with the meter's own feedback. Training rows are the
  learner's own written blocks in the instances it attempted, labelled only by that instance's
  per-row spelling error from `grade_spelled` (a bag-level label over the blocks it wrote there).
  You choose the loss; the simplest principled bag-level objective is fine, and say what it is.
  No per-block verdict enters.
- **`own_verdict`** — the builder's `fit_rule_verdict` ceiling: own written blocks with the
  per-block verdict. The strictly stronger channel; fit it offline first (below).
- **`surface_matched`** — the existing harvest subsampled per cycle to the number of labelled
  blocks `own_scalar` receives that cycle. This is the control that separates source from volume,
  and the node is not science without it.
- **A read-back check**, offline before it is an arm: on the learner's own written blocks, does
  the frozen reader ever return a feature other than the one intended (on the collision-free draw
  every synonym may parse correctly, in which case a misspelling has no sensory consequence
  through the reader and the arm is dead by construction; that is a finding, log it and do not
  build the arm).

Keep the `blk_render` ledger; decide whether renderer fits are priced and log the decision either
way. The head stays one linear layer over (one-hot feature, scalar register) so the sizing lane's
ceilings still apply.

## Sequence

**Q0, offline, CPU, before any GPU.** From the banked `if_q1c_yk` and `if_q3_*` logs (fetched
under `../inflection/figures/<tag>/`; volume `rhm-scaling-data:/data/rhm_practice_inflection/`):
(i) fit `own_verdict` from `spell_rows` cycle by cycle and report held-out accuracy and θ̂ against
cycles of own writes, beside `surface`'s trajectory from the same tags; (ii) the volume ratio, own
written labelled blocks per cycle against harvested surface blocks per cycle, by register; (iii)
the read-back check; (iv) the identifiability ceiling at own-write volume (the sizing lane's
`sizing/SIZING.md` method). If (iv) says the rule is not identifiable from own writes at any
budget this ladder reaches, stop and report.

**Q1, the source of the signal on one clock.** `if_q1c_yk`'s design exactly (collision-free
world `rule_seed 6`, `E_R8`, practised {0,3,7}, `--setup-render rule`, the 140-cycle ladder,
schedule-paced; the flag line is in `../inflection/FILES.md` §Q1c and the README's Reproduce):
`canon`, `given_rule`, `leaf`, `surface`, `surface_matched`, `own_scalar`, and `own_verdict` if Q0
says it differs from `surface`. Readouts: written spelling error over the ladder (the time to
solve, not only the endpoint); transfer at the held-out registers and θ̂ by cycle (§1I/§2I);
meaning on the matched clock (§Q1c's two-cluster read); the F2 per-slot record with the slot's
execution reliability now earned by production. Smoke first. ~3 GPU-h on an L4. Halt with the
handle, then with the reduction.

**Q2, the pacer seat, after I have read Q1.** `if_q3_e`'s two-stage design under the mirror seat
(yield commits, δ-silence advances) with `perf_e_spell` on/off pairs for the production-route arm,
floors re-derived on that arm's own spelled series as Q3 did. Add a **scale-free quiet statistic**
as a knob, default off and bit-identical when off (a slope normalised by the series' own scale, or
a floor derived per series; you propose it and say why), and run the pairs both ways. This is the
queue's item 2 for `inflection` and it belongs on this fork. ~3 GPU-h.

## Norms

Smoke on Modal first at every step; L4; `modal run --detach` with the explicit function; the
session-isolated `launch_detached.py` idiom from the donor. Single seed; ranks, signs, located
mechanisms and multiples of measured floors are the claims. Volume, difficulty mix, priced budget
and lifetime held fixed across arms; only the renderer's training signal moves. No outcome is
interpreted in advance, here or by you. Where the substrate says the question is ill-posed, the
numbers that say so are the deliverable.

**Not yours**: the class lane (the corridor head on `enharmonic.py`'s open book trained on the
learner's own solved classes, with a read-back parse through the class-keyed table) — another agent
is on the composed arm in that lineage and `enharmonic.py` must not be touched; and the composed
fork. Both wait on what this lane says.

---

## Q2, revised (2026-09-11, after Q0): the homophonous lexicon replaces the pacer seat

**What Q0 found.** On every banked row the correct synonym for the learner's own written block
is the same table entry the world writes at that feature and register: template and map are one
function. The bottom map is injective on the collision-free draw and the reader inverts it
exactly and for free, so the world's command at every block is public and own attempts carry no
information the surface does not. The read-back check is dead by construction: not one of 359
misspelled own-write blocks parsed to a different feature. The asymmetry that makes production
learning its own thing (`operators_are_arity_two` §4: my command is observed, theirs is latent)
is zero here. No body is needed to restore it, only a many-to-one map, and Jasper's reading puts
that map in the lexicon, not at an emitter: the operative thing is matching word to intent.
"Tip of the tongue" (an intent with no retrievable word) and "let me take that back, I
mis-phrased that" (a word that reads back as a different meaning) are the two failure modes of
that map, and neither exists on an injective lexicon.

**The world.** The bottom map on RHM already is a lexicon: a feature is a meaning, its synonyms
are words, a leaf tuple is a word-form. Take the collision-free draw (`rule_seed 6`) and replace
*only* `bottom` with a constructed map carrying a controlled number of shared forms (homophones:
two (feature, synonym) pairs on one leaf tuple), every level above untouched, so mining dynamics
stay `tutti`'s and only the lexicon changes. The rule `K[f, ρ]` is the same ordered-register
family. A homophone should be placed so the world's block is genuinely ambiguous under the
rule: `bottom[f1, K[f1, ρ]] == bottom[f2, K[f2, ρ']]` for some registers, so that the
perceptual route sees one form with two intents while the learner's own writes are unambiguous
to itself. `rule_seed 0` has two such cells of fourteen and inflection Q1 saw the fingerprint
there (the fitted heads learned the rule modulo the reader's confusions; 7 of 15 deviations at a
reader-unreadable cell).

**Q2.0, offline, CPU, first.** How much homophony the world tolerates: for each collision
count, whether every instance stays parseable to the root (`possible_sets` non-empty, `d*`
unchanged), how the reader's last-writer-wins inverse map resolves each homophone, and what the
frozen neural reader does at those cells at production `reader_steps` (its confusion matrix by
register). Pick the densest lexicon that keeps the world solvable and the reader above its
`if_smoke` baseline away from the homophones; report the ladder of counts you rejected and why.
If no count above two survives, say so and stop.

**Q2.1, the three routes on one clock.** `if_q1c_yk`'s design on the homophonous lexicon:
`canon`, `surface` (others' words with inferred meanings, ambiguous at every homophone),
`own_scalar` (own words graded by the meter, as built in Q1), and **`own_readback`** (own words
read back through the frozen reader and compared to the learner's own intent, the retraction
route: no grader in the loop; the label is whether the read-back feature is the intended one).
`given_rule` as the ceiling if cheap. Readouts: each route's recovered `K` at the homophonous
cells against the non-homophonous ones, by cycle (the perceptual route's mode-averaging is the
fingerprint to look for, and its absence is a finding too); the mis-phrasing rate, own blocks
whose read-back is not the intent, over the ladder and per route; the tip-of-the-tongue rate as
demanded-but-unheld, which the at-support instruments already carry; written spelling error and
the matched-clock meaning read as before. ~3 GPU-h on an L4. The pacer seat (the original Q2) is
deferred, and whether it is worth anything is decided by Q1's time course.

**Corrections and decisions, 2026-09-11 (after the smoke and Q2.0).** (i) The sentence above,
"the read-back check is dead by construction: not one of 359 misspelled own-write blocks parsed
to a different feature", is withdrawn: RB-1 with the production reader flips one block to its
feature's other synonym and the reader returns a different feature 0.5625 of the time. The
banked evidence was a different condition (coherent misspelling across a whole answer). So the
retraction route is live on the injective lexicon and runs as `own_readback` beside Q1
(`em_q1b`). (ii) Q2.0 sized the ladder: every H from 0 to 8 stays a world; `d*` falls with H,
so rungs are not difficulty-matched and the clean contrast is within a rung. (iii) Q2.0 also
showed that reconstructing the learner's own intent *from the form* collapses with H in lockstep
with the observer's ambiguity, so the asymmetry cannot come from the lexicon alone; it comes
from the learner carrying its own command forward from the write. Decision: carry it. The
executor computes the intended features from the learner's own table before it renders them
(`macro_features` returns `feats`; the renderer maps them to synonyms), so the (feature,
synonym) it wrote is its own state, free at write time: the efference copy as
`operators_are_arity_two` §4 defines it ("you emitted it, it is free"). The oracle would be the
*world's* intent at a shared form, which the learner never receives. Under homophony the
retraction label is therefore "did my own reader give back the feature I meant", with the
intent recalled from the write record, never from the form.

**Q2.2 (2026-09-11, after the Q2.1 read; Jasper's go): the listener's teacher.** Q2.1 found the
loser owner of each shared form all but absent from the learner's own intents at the shared
registers, and every arm's mined table ~78% wrong in feature space at token precision 1.0. The
learner mines its vocabulary through its reader, and that reader's training target is the
last-writer lookup, so a distinction the lookup discards is never learned as a word. Q2.2 is the
one-bit substrate change that tests whether the collapse belongs to the listener's teacher: the
same H=3 lexicon, the same five arms, `--own-intent record`, with **reader B** (the same
architecture, corpus, steps and seed, trained on the level-1 features the world derived) as the
learner's ear everywhere the ear is used (the executor's DP, the harvest, mining, the read-back).
The grader is untouched. This is a modified substrate and the tag's record must say so. Readouts:
the per-cell own-write counts at the loser cells against Q2.1's; the mined table's precision in
feature space against token space; recovered K at the shared cells per route; the mis-read rate at
the shared cells, where a context-resolving ear can for the first time mishear a lone off-register
word *and* resolve a homophone, so the retraction route may have a signal it did not have; and
whether any route departs from the rule at a shared cell to be heard. ~3.5 GPU-h.
