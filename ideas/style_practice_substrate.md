# A style-practice substrate: scarce image corpora, a data-defined action alphabet, and taste as the missing organ

**Status**: Idea / research-direction memo (2026-08-21). **Updated 2026-08-25**: the first node ran
([`canvas/plant`](../experiments/canvas/plant/README.md)); §11 records what it did to §3a, §4, §5, §6, §8 and §9 —
the tile grammar is the substrate, aligned, with a support test for truth and typicality for taste. **Updated
2026-08-22** after the two RHM de-risking twins reported (`critic/`[^private],
PR #61; `teacher_slot/endo_yield/`[^private], PR #62;
their READMEs pending discussion): §1's justification, §5's instruments, §6's self-taste, and §8's
node 1 carry the changes. **Auxiliary to
[practice_manufactures_its_own_credit.md](practice_manufactures_its_own_credit.md)** (the theory the
practice arc hangs off) and the substrate sibling of
[physical_control_substrate.md](physical_control_substrate.md) (which made the same move for
MuJoCo). Options are deliberately left open where the discussion left them open; each is marked.
**Date**: 2026-08-21
**Builds on**: [practice_manufactures_its_own_credit](practice_manufactures_its_own_credit.md) §1
(the three components), §3½ (chunk identity vs corridor), §16 (the level-(k+1) law), §18
(distribution is forced; the meter; pretraining as the degenerate case), §20 (the port back) ·
[recurrence_manufactures_confounds](recurrence_manufactures_confounds.md) §5 (the merge op; "a
belief is a chunk in the limit of maximally varied demand") ·
[cerebellar_abstraction_ratchet](cerebellar_abstraction_ratchet.md) §3 (A becomes input to B) ·
[meta_learning_under_metered_data](meta_learning_under_metered_data.md) (the sample-price condition) ·
[physical_control_substrate](physical_control_substrate.md) (the one fork: controllable DGP, not
benchmark; a third substrate, not a migration).
**Relates to (experiments)**: [`rhm/practice/native/`](../experiments/rhm/practice/native/README.md)
(the port back: routing and corridor consolidate, the table stays as the address book) ·
[`rhm/practice/ratchet/`](../experiments/rhm/practice/ratchet/README.md) and
[`macros.py`](../experiments/rhm/practice/ratchet/macros.py) (what a vocabulary is in code) ·
[`typed_gaps/`](../experiments/rhm/practice/typed_gaps/README.md), [`setlist/`](../experiments/rhm/practice/setlist/FILES.md),
[`merge/`](../experiments/rhm/practice/merge/README.md), [`fourwall/`](../experiments/rhm/practice/fourwall/README.md),
[`teacher_slot/`](../experiments/rhm/practice/teacher_slot/README.md), [`reread/lm/`](../experiments/rhm/practice/reread/lm/README.md) ·
[`mjc/practice/fingering/`](../experiments/mjc/practice/fingering/README.md),
[`legato/`](../experiments/mjc/practice/legato/README.md), [`span/`](../experiments/mjc/practice/span/README.md) ·
`rhm/practice/critic/`[^private] (a learned grade of record, calibrated
against the exact one) · `teacher_slot/endo_yield/`[^private]
(the label-free one-level-up read) ·
`fer/`[^private] (FER/UFR metrics in `LAB_NOTES.md`, an optional readout; Picbreeder genomes in `picbreeder_genomes/`).
**Reading**: `reading/quanta.pdf`[^private] (Michaud, Liu, Girit & Tegmark 2023),
`reading/quanta2.pdf`[^private] (Michaud 2026, the retrospective),
`reading/Lluminate.pdf`[^private] (Simon; LLM-evolutionary creative exploration under
CLIP-space novelty). Not yet in
`reading/` and worth pulling: DreamCoder (Ellis et al. 2021), Bayesian Program Learning on Omniglot
(Lake, Salakhutdinov & Tenenbaum 2015), SPIRAL (Ganin et al. 2018), Learning to Paint (Huang et al.
2019), "On the creation of narrow AI: hierarchy and nonlocality of neural network skills" (Michaud
et al. 2025), "Physics of Skill Learning" (Liu et al.), and the FER paper (Kumar, Clune, Lehman &
Stanley 2025; summarized in `fer/README_previous.md`[^private]).
**Prompt and attribution**: a conversation with Jasper (2026-08-21) reading the last twelve practice
PRs against the quanta model. The substrate move itself, style as the meter, dropping the oracle,
producing an image in pixel space as a control task, the Picbreeder preference model, and
sequencing the forward model for later are Jasper's. Discrete spatial tokens as the RHM-shaped
instantiation, style as demand-concentration over the valid set, the preference model as a learned
teacher gauge, and self-taste as next-level yield came out of the exchange.

## One-liner

Port the practice arc to a third substrate where the latent structure is rich, the exemplars are
scarce, and nothing in the learner is pretrained on language: a small corpus of images in a style,
quantized into a grid of discrete codes so that the data defines the action alphabet, with
inpainting under a token budget as the piece. The RHM-side machinery (sculpting, `ratchet`,
`native/`) ports with the grammar replaced by a learned token model, the meter supplied by the
scarcity of the style itself, and no DGP oracle. The design question that did not port cleanly is
the grader: a style has a crisp notion of *wrong* and a multimodal notion of *right*. `critic/` has
since run the loop on RHM with a learned grade of record and the exact grade logged only, and
handed this substrate a calibration protocol; what has still never happened is running it
unchecked, which is what images are for. Taste, a grader with preferences inside the valid set, is
the one organ neither RHM nor MuJoCo had; its candidates are kept open in §5.

## 1. Why a third substrate, and why now

[`native/`](../experiments/rhm/practice/native/README.md) closed gap 6 of the arc: an earned
vocabulary consolidated into the learner's own planner (routing) and executor (corridor), with the
table kept as the address book the next level is mined over, the can't-decompose signature
present, and deleting the table costing the current level nothing and the next level everything
it can structurally cost. That is the ratchet as the program originally meant the word. The two
substrates it was built on have done what they are for. RHM supplied an exact DP oracle, a known
true vocabulary, and an exactly known aleatoric floor; MuJoCo supplied a body that is its own
ground truth and the composition-horizon axis. What remains on them is specific and still queued
(the live re-earning spiral in `tall/`'s depth-6 world, the rate claim that era k+1 certifies
faster given native era k). Those should run in parallel, not be skipped; they are the cheapest
way to learn whether the crank has a wall.

What we want from a third substrate is latent structure that the first regime does not hand over.
The two-regime reading of the arc (the conversation's compressed form): some representations are
given to you by gradient descent on existing data, and some you have to mine by acting and seeing
what works, because their value is denominated one level above the loss that forms the substrate
and the substrate's operator destroys them.

Tokens-on-inpainting (§3a) is RHM sculpting with one substitution, so the two substrates are more
directly comparable than the first draft of this memo assumed. That is the arc's twin pattern
(mjc ↔ rhm: fork the donor verbatim, one variable, bit-for-bit gate on the rest), and it makes the
port smaller rather than redundant: node 1 reads against `ratchet`/`critic` on the same ledger and
the same arm names. The justification therefore rests on what images add that RHM structurally
cannot (settled 2026-08-22):

1. **The alphabet's provenance.** RHM's `T[1]` is the DGP's own features. On images `T[1]` is a
   codebook the gradient built for reconstruction. "Does mining recover parts over an alphabet the
   first regime built" is the two-regimes question in its sharpest form, and RHM cannot ask it
   because its alphabet was never learned.
2. **No oracle in earnest.** `critic/` showed the grader protocol survives without ground truth
   *where we could check*. Images are where it has to survive unchecked: a driving grader, a
   reporting grader, distance to held-out truth, the self-manufactured-damage rule, and nothing
   behind them. RHM can only rehearse that condition.
3. **Real demand.** RHM's style is OU drift on a derivation distribution. An alphabet, a sprite set,
   an artist's line is a demand-concentration produced by human ratchets, so taste has content
   there; on RHM a preference inside the valid set is a lookup table (§6).
4. **External comparators and legibility.** DreamCoder and BPL name specific disagreements (§7),
   and images are the domain where the degenerate case, the diffusion model, exists in the wild, so
   "practice loop vs pretraining" is a contrast outsiders can see.

The honest risk of "more comparable" is "less new," and the four items above are what carry the
difference.

Desiderata, as Jasper set them: rich latents that pretraining on the raw data does not unlock;
limited inbuilt cognition, so language modeling is off the table; a compact data-generating process
so compute stays small. The discipline is the one
[physical_control_substrate](physical_control_substrate.md) set for MuJoCo: a controllable
substrate used the way we use RHM, not a benchmark; a third substrate, not a migration; and port
one node at a time, forking the closest donor verbatim with a bit-for-bit gate on the untreated
arms.

## 2. The filter: what the loop has been measured to need from a substrate

Each row is a precondition some arc node died on or leaned on, with what a style substrate
supplies. Three rows carry revisions from the 2026-08-21 discussion.

| precondition | where it was measured | what style supplies |
|---|---|---|
| **A meter.** Priced feedback, and depth unaffordable to the primitive action space. Without it pretraining is the correct degenerate solution (§18 ground 3; `ratchet`: the exact-DP oracle and a width-16 beam at 5.9× budget lose to one L2 macro). | `ratchet`, `crystallize`, §18 | **Both halves, by construction** (revision). Few exemplars is the metered-*data* regime of `reread/` (renewable in competence, vocabulary-gated extraction, a corpus-size wall). Creating and judging a new in-style work is expensive, which is the metered-*feedback* regime the loop prices as `steps·dt + fb·d_fb`. The budget has to bind and be counted; the Adam-invariance lesson from `priced_plasticity/` says a claimed budget that rests on a normalizer does not. |
| **A discrete, compositional action space over the inner learner's interface**, so chunks nest (`T[l]` over `T[l−1]` entries) and are addressable. | `macros.py` header; `native/` | Strokes, or the codebook of a quantized image (§3). |
| **Boundaries that carry information.** State-independent units gave post-commit drift 0.0000 and made fusion vacuous. | `etude/` finding 7; `fingering/` G1 | Pen state between strokes; the filled neighborhood around a masked region. |
| **Consumption distinct from audition**, with demand drift and truth drift both expressible. | `ear/`, `typed_gaps/` | Audition on a fixed panel; consumption = the drifting mix of styles and masks asked for. Demand drift = rotate that mix (`setlist`). Truth drift = a new style whose statistics the model has not seen (`transpose`). |
| **An oracle.** | every node | **Split three ways** (revision). (i) Agent-consumed oracles: none, and the arc is essentially there already (`macros.py` logs agreement with the exact map "as an oracle readout, never consumed"; `teacher_slot` replaced the last exogenous justification with a reasoner). (ii) An experimenter-known DGP, which gave earned-vs-given and foreclosure diagnosis against the true table: not required. The replacement is the method the arc already practices: ranks, signs, in-tag twin pairs for floors, bit-for-bit gates. What is lost is the ceiling, not the method. (iii) Grader calibration: this is the one that bites, and §5 is about it. **Measured 2026-08-22** (`critic/`): a learned grade of record drives the `ratchet` loop at +0.008/+0.001 over its oracle-graded twin at the calibrated corpus size, for 1.6–3.0% of priced time, and the calibration rule that picks that corpus size is oracle-free. |
| **A forward model with a composition horizon.** | `fingering`, `legato`, `span` | **Not required for re-ingestion** (revision). `native/` needs a planner that proposes over an action set, an executor that materializes content, a selection stream, and a table. The FM was load-bearing for the *content* question on the arm (live plan-at-launch vs frozen measured chains) and for `span`'s horizon-as-trajectory. A substrate with no FM sits, by construction, in the regime legato found beyond the horizon: everything committed is measured content and chunks are the object. The horizon questions return when the FM does (§8). |
| **Rich latents, limited inbuilt cognition.** | (Jasper's desiderata) | Images in a style; the learner trained from scratch on the corpus; language only in the teacher slot. |

## 3. Two instantiations (option kept open)

Both are images; they differ in who defines the action alphabet. They are the mjc/rhm twin pattern
again, and the recommendation is about which node to run first, not about which to build.

### 3a. Discrete spatial tokens (RHM donor)

Let the data define the brush.

1. **Quantize.** 32×32 images cut into 4×4 patches, a codebook of K entries learned by
   reconstruction. The simplest form is k-means on patches with lookup as the decoder; a VQ-VAE is
   the neural form. Each image becomes an 8×8 grid of code indices. *Open*: K (64 vs 256 sets the
   granularity at which tuples become parts); k-means vs VQ-VAE; for pixel art, skip the codebook
   and use raw palette indices as `T[1]` (1024 actions per image instead of 64, and the most literal
   reading of "deliberately painting in pixel space").
2. **Generate.** A small masked token model over the grid, any-order (MaskGIT-style): given some
   cells filled and the rest masked, predict the masked codes. A move is "fill these cells with
   these codes." *Open*: raster-order autoregression is simpler but imposes an arbitrary sequence
   and makes the seam a prefix instead of a neighborhood; any-order keeps the seam spatial and
   matches sculpting's span-based moves.
3. **Decode** the finished grid to pixels for grading or display.

The mapping onto the arc is RHM sculpting with the grammar replaced by a learned token model over
real images:

- `T[1]` is the codebook, which is `macros.py`'s "`T[1]` = the v level-1 features (native: the
  generator predicts these)." `T[2]` is recurring code-tuples (a 2×1 or 2×2 block that co-occurs
  in successful completions): edges, textures. `T[3]` is tuples of `T[2]` entries: parts, motifs.
  Nested over entries, so foreclosure is structural exactly as in `ratchet`.
- A macro stamps a whole tuple as one move. The chunk becomes the new syllable.
- The codebook is a first-regime object: reconstruction-optimal patch summaries, possibly entangled
  in the FER sense. Whether mining recovers part structure as tuples over an alphabet that was not
  built for it is a readout, and arguably the point.

### 3b. Strokes (mjc donor)

A turtle or brush renderer as the body; commands are strokes; the seam is pen state (position,
heading, pen up/down), which is the arm's hand-over posture in miniature. The piece is reconstruct
a target under a stroke budget. `legato/world.py`'s object (piece, plant, diet, `Body`, CEM, compile
ops, `Ledger`) maps nearly one-to-one onto target, renderer, target distribution, canvas,
stroke planner, compile ops, `Ledger`. Stroke data exists in Omniglot (pen trajectories per
exemplar), Quick, Draw!, and TU-Berlin sketches. SPIRAL and Learning to Paint are the ML analog of a
painter deliberately painting in pixel space. A deterministic renderer may have no regime beyond
the horizon; a stochastic brush supplies one the way legato's motor noise did.

### The two side by side

| | strokes | spatial tokens |
|---|---|---|
| action alphabet | designed (stroke parameters) | learned from the data, or raw palette |
| seam | pen state | neighborhood |
| needs | a renderer, stroke data | any images |
| arc donor | `mjc/fingering`–`legato` | `rhm/ratchet`–`native` |
| grader without an oracle | reconstruction of a target; a critic | inpainting against held-out truth; a critic; the VQ floor |
| painter analog | strong | weak |
| FER readout | direct (fit CPPNs to the agent's units) | indirect (fit CPPNs to decoded macros) |
| named comparator | BPL, SPIRAL | DreamCoder, in spirit |

**Recommendation**: tokens first, because the node we most want to see on a new substrate is
re-ingestion and tokens port `native/` with the least change. Strokes are where the compile-op
taxonomy and the seam law (`fingering`) port most directly, and where the Liszt→Beethoven claim
(a stroke primitive transfers across alphabets) is most legible. Run strokes as the sibling when a
question needs them.

## 4. The piece, the ladder, the meter (tokens)

- **The piece is inpainting.** Mask a region of a held-out image, complete it in-style under a
  token budget. This is sculpting's "repair damaged configurations by level-wise moves," with a
  grader that needs no DGP (§5).
- **The depth ladder is mask size.** Era 1: small holes. Era 3: half the image. This is
  `ratchet`'s damage-by-depth schedule and supplies the regime where depth is unaffordable at the
  primitive level: a 64-cell completion at beam width w is w^64-shaped; a 16-cell part as one move
  collapses it.
- **The meter.** Each emitted token costs `dt`; each grounding is a forward pass; each
  materialization a decode; each critic call `d_fb`. Declared grounding budget picks the beam width
  exactly as G = 58 did in `ratchet`.
  (A prior on `d_fb` from RHM, added 2026-08-21: the endogenous evaluative teacher in
  [`endo_expansion/`](../experiments/rhm/directed_sculpting/full_loop/endo_expansion/README.md)
  §6 paid 1218 materialisations per label, ~150× the privileged DP teacher's 8, while a
  reported-currency teacher at 14 per label performed comparably on every paid readout.)
- **The seam** is the filled neighborhood around the hole. A state-conditioned library keyed by it
  is `crystallize`'s library keyed by the observed target.
- **Consumption vs audition.** Audition: completion quality on a fixed panel. Consumption: the
  drifting mix of styles and masks actually asked for. Demand drift rotates the mix; truth drift
  introduces a style the token model has not seen.
- **A floor without an oracle.** The quantizer's own reconstruction error bounds what anything
  downstream can achieve. Strokes do not give this.
- **`native/` ports directly.** π over {codes ∪ macros} beside the token logits (Port 1); a span
  head emitting the macro's codes in one pass conditioned on (level, node), with the parity gate as
  "the stamp reproduces the tuple" (Port 2); the end-of-run table-ablation battery; the
  can't-decompose readout as π's mass on a macro's code spelling; next-level minability as |T3|
  candidates at support.
- **Mining** is `macros.py`'s rule: tuples come only from completions the agent succeeded on,
  parsed by the agent's own token model, so nothing above its own competence enters the vocabulary.
- **Corpora** (*open*): Omniglot images (50 alphabets, ~20 exemplars per character; an alphabet is
  a style; scarcity by construction); a sprite sheet or pixel-art set (discrete already, palette as
  alphabet); a single font; a small set of one artist's line drawings. The choice sets what
  "style" means and how large the parts are relative to the grid.

## 5. The grader is the load-bearing design question

Everything else in the loop has now ported twice. The grader had never been run without ground
truth behind it until `critic/`[^private] did so on RHM
(2026-08-21→22) with the exact grade logged only; it has still never been run unchecked, and the
arc's most-repeated blocker was the grader (mis-levelled, noisier with depth; `native/`
interpretation (e)).

**Precision on "never without ground truth"** (added 2026-08-21, after reading the last twelve
practice PRs against this memo). True at the *terminal* layer: every practice node's grade of
record is [`crystallize/units.py:grade`](../experiments/rhm/practice/crystallize/units.py) —
possible-set success and exact residual `d*` from `rules` — consumed four ways on the sculpting
substrate (the value head's regression target, `ratchet.py:360`; the mining gate, since only
solved repairs enter the table; the audition, `ratchet.py:573`; the reported error). One layer
down, a non-oracle cell already exists: [`ear/grader.py`](../experiments/rhm/practice/ear/grader.py)'s
`(mfg, policy)` audition — the agent damages its own clean derivations with its own level-(k−1)
table, "nothing in `manufacture_damage` touches `rules`" — and it carries a calibration record
against the withheld oracle (`calm_s0`/`cald*`: optimistic 1.3–2.1× at level 3, self-seeding
bias −0.07/−0.16, agreement collapsing into noise at level 3). That record is the template for
calibrating a grader the agent holds against an oracle only the experimenter holds. And pre-arc,
[`endo_expansion/`](../experiments/rhm/directed_sculpting/full_loop/endo_expansion/README.md)
and its parallel sibling [`endogenous_expansion/`](../experiments/rhm/directed_sculpting/full_loop/endogenous_expansion/README.md)
(PR #32) ran an endogenous evaluative *teacher* — no DP, no rule table, no channel label, only
paid terminal success — and it expands: 19–34% of the DP teacher's lift, 3/3 seeds; "a grader
is a ceiling, approached from whichever side you start"; the residual gap attributable to target
fidelity, not endogeneity (two external DP teachers degraded to matched fidelity lose to it). Its
terminal labels were still oracle-computed — on RHM the environment's reward always is, and that
is the one thing that changes on images.

**Shape.** A style has a crisp notion of *wrong* and a multimodal notion of *right*. That is
already the shape of RHM sculpting's grader: terminal possible-set success asks whether the
configuration is consistent with *some* valid derivation, never whether it is the one truth. So
the inpainting grader should be possible-set-shaped: on-style, by a style model trained on
*held-out* exemplars (so it is not the agent's own model grading the agent), with distance to the
held-out truth as a secondary readout rather than the score. Held-out pixel truth alone
over-penalizes valid alternatives; the étude's E-3b lesson (score under the consumption
distribution, not against one rendition) is the same point. And averaging valid completions goes
off-style, which `crystallize` measured exactly (the modal span off-grammar while every
contributing realization was on-grammar), so selection-not-averaging reappears on day one.

**How we would know it lied.** Instruments, none of them an oracle, as revised after `critic/`
scored the first draft's four exactly on RHM (2026-08-22; `cals_s0`, `cr_s0`, `calc_s0`):

- **Self-manufactured damage, the one that works.** Corrupt a held-out exemplar without touching
  any rule (`ear.grader.manufacture_damage`), require pass ≤ 0.10 on it and ≥ 0.90 on clean
  exemplars, and take the smallest corpus that qualifies. On RHM the flip (8192 → 32768 exemplars)
  is where agreement with the exact grade jumps 0.85 → 0.97, optimism crosses 1, and a beam that
  optimises the critic directly stops lying (claims 0.96 against true 0.49 at 8192; 0.543 against
  0.545 at 32768). This is the calibration rule, and it ports verbatim.
- **Distance to held-out truth**, kept as an *independent* secondary readout rather than a lie
  detector: r(score, −d_truth) is 0.01–0.04 for the good grader and the bad one alike, against
  0.83–0.85 for −d*. The orthogonality is the grader being right about valid alternatives (which
  sit ~12 tokens from the one truth), so it tells you the score is not a copy of d_truth and
  nothing else.
- **A second reader on a disjoint split: blind to the failure that matters.** The scarce critic
  passes 21–41% of damaged configurations and costs the loop +0.096/+0.058, while its rank
  agreement with the exact grade is 0.72–0.83 at pairwise cost 0.000–0.047, inside the homogeneous
  band.
- **`own − world`** (`merge/` round 2), now a number: a reader refitted on the arm's own graded
  successes passes clean exemplars 0.5% and valid alternatives 0%, while passing the agent's own
  fixes at up to 67%. That is the quantified warning for "internalize the grader."
- **Paired A/B probes by a verbal teacher**: not run; still the heterogeneous instrument of last
  resort, and the one the reasoner in `teacher_slot` built for itself.

The caveat that predicted the third result (added 2026-08-21, confirmed by `critic/`): two style
models on disjoint splits are
*homogeneous* disagreement — same objective, different data — and
[`heterogeneous_graders`](heterogeneous_graders.md) §9 (with `mjc/ballistic/directed` S1 as the
measured negative) says homogeneous members are blind together wherever the blindness is
structural; `endo_expansion` §7.6 put numbers on it (same-type graders cost each other 0.036 of
their own range, different-type 0.48–0.69). So the cross-grader floor catches sampling noise, not
a shared failure mode such as the agent gaming a feature both readers learned; the held-out-truth
distance and the teacher A/B probe are the heterogeneous instruments and should carry the weight.
`endo_expansion`'s five-scorer pairwise disagreement table is the reusable form.

**The teacher slot.** A VLM in the verbal port, as `teacher_slot` ran a text-trained reasoner:
gauge choice, merge/retire decisions, and A/B calibration, with the learner itself untouched by
pretraining. This is where inbuilt cognition is allowed to enter, and only here.

**The taste gauge: options, held loosely** (Picbreeder demoted from primary candidate
2026-08-22). Three are on the table. (a) The endogenous next-level yield of §6, now label-free
and nearly free (`endo_yield/`), which needs no external data at all. (b) CLIP-space novelty as
Lluminate uses it: a *coverage* gauge on outputs, which should be read at the level of the
vocabulary (new tuples at support) rather than the output, since a novel image that composes into
nothing raises the first and not the second, and `fourwall` has coverage-graded gates locking
scaffolds in forever. (c) A learned preference model from Picbreeder, kept here as the record of
why it was considered: Picbreeder users did not
grade finished quality; they chose which child to branch from. That is a stepping-stone judgment,
a value denominated in what the image might lead to, which is the arc's level-(k+1) law and the
gauge `teacher_slot` found the teacher's scarce part to be. A preference model trained on branch
choices (pairwise, from sibling branch counts in the lineage data in `fer/picbreeder_genomes/`) is
therefore a learned instance of the one-level-up gauge, not a quality critic, and should be
treated as one. Two caveats on the record: it is a first-regime object and gameable, and the
arc's usage pattern (sparse priced calls to select among the agent's own successes, never a
gradient target) is less exposed to that than reward optimization but not immune; and transfer
from CPPN images to tokens or strokes is an unknown, so it should meet the loop first on a
CPPN-native sibling where it is in distribution (§8).

## 6. Style, taste, schools

The one conceptual result of the discussion worth writing down on its own.

**A style is a demand-concentration over the grammar's valid set.** The grammar says what is valid
(truth); a style says which valid compositions get used, how often, in which contexts (demand).
That is what `typed_gaps` measured a committed chunk to store: demand-concentration, not truth. So
an agent's mined table under one derivation distribution *is* a style, and the measured fact that a
demand-concentrated table beats the full true table in audition reads as: the stylist beats the
encyclopedist, in-style. `setlist` (OU drift on the derivation distribution, nothing becomes
false) is style drift, already run. A school is a shared demand-concentration transmitted between
agents, and `teacher_slot`'s verbal channel and `handle/` are the two transmission ports tried so
far.

**What RHM lacks is taste, not style.** RHM's only graders are possible-set success and priced
time. A grader with preferences *within* the valid set is the organ neither substrate had. Add it
and schools follow; let it move and movements follow. (A grammar extension, `transpose`'s
truth-drift with a rule added rather than resampled, is the other route to an avant-garde.) The
style substrate is where taste is natural, because the critic in §5 is one.
The structural reason, on the record (added 2026-08-21): RHM's valid set is enumerable — 19
reachable level-2 entries in [`merge/`](../experiments/rhm/practice/merge/README.md), every venue
demanding 9–15 of them — and its true demand is computable in closed form
([`setlist/demand.py:demand_table`](../experiments/rhm/practice/setlist/demand.py), the
concentration ceiling), so a preference *within* the valid set there is a lookup table. The one
axis that is not demand frequency is next-level yield, which is self-taste below; and on RHM the
oracle can be withheld but never lacked, so every RHM grader experiment is a calibration study
by construction — the right remaining use of the substrate for this question, not a discovery
venue.

**Self-taste has a measured seed.** `teacher_slot` A½ read the learner's *own next-level mining
yield* as the gauge (a 13× read premium, 9.6% of budget) and it funded the merge before the first
rotation. "Interesting" = "raises next-level minability" is Picbreeder's stepping-stone criterion,
the level-(k+1) law, and the only endogenous one-level-up gauge the arc has found, in one line. The
standing caveat is `merge/` round 2: an agent whose practice distribution is its own
competence-amplified footprint collapses coverage at β = 2, with holes 30–70× starved. Self-taste
narrows unless demand is varied from outside. The long-run shape is an external gauge that seeds,
the agent's next-level yield that augments, and a teacher that keeps varying the demand.

**Closed label-free** (2026-08-22, `endo_yield/`[^private],
`ey0`). The learner's own NLL at the positions that close the level one above the
keyed one (a fact of the grammar's shape, no labels) drives the same merge at 1500, pre-rotation,
with a condition sequence identical to the labelled probe's: at 1.5% of budget under the neutral
condition, and free under the arm's own consumed condition. The 2×2 (keyed span vs next level) ×
(own condition vs neutral) says the **level** is the discriminating coordinate, not the condition:
the keyed-span read under the consumed condition refuses forever, the one-level-up read under the
same condition commits at zero price. So no neutral condition has to be constructed on images. Two
cautions carried: the hold is thin (the driven read cleared its dead zone on 1/10 trials, 1.3× the
corrected in-tag floor; the offline scoping overstated range/floor ~73×), and the shadow
sibling-span read has 8× the range/floor, so it is round 2's instrument. The form that ports to
tokens is `at_support` next-level yield, which node 1 logs for free.

## 7. Named external comparators

The arc has never had one. This substrate has three, and the differences are the arc's content.

- **DreamCoder** (Ellis et al. 2021). Wake / abstraction-sleep / dream-sleep is solve / mine /
  consolidate; LOGO graphics is one of its domains. No meter, no seam law, no demand drift, no
  address-book ablation. Its abstraction grader is MDL compression, a *within-level* signal, and the
  arc found within-level graders lose to consumption-grading at every frontier (`ear`, `recital`,
  `typed_gaps`). That is a testable disagreement about what licenses a library entry.
- **Bayesian Program Learning on Omniglot** (Lake et al. 2015). Concepts as programs over a learned
  library of sub-stroke primitives, learned on 30 background alphabets and transferring to 20
  held-out ones. The primitives-transfer result is the Liszt→Beethoven claim measured on a real
  human corpus.
- **SPIRAL, Learning to Paint** (Ganin et al. 2018; Huang et al. 2019). Stroke agents on a pixel
  canvas with a real or neural renderer. The action interface for §3b, graded adversarially or by
  reconstruction, without a vocabulary, a meter, or a teacher.

(`fer/`'s FER/UFR metrics remain available as an optional representation readout on a mined
library, §3's table; they are not part of the justification.)

The quanta papers sit behind all of this as the first-regime account: the codebook and the token
model are Michaud's quanta, and the mined table is the second kind. The conversation's reading of
that relationship is queued as a §21 for the parent doc, not written here.

## 8. Proposed sequence of nodes (held loosely)

1. **`ratchet` on tokens, with `critic/`'s protocol inherited** (amended 2026-08-22). Earn a
   level-indexed code-tuple vocabulary over an inpainting depth ladder under a priced budget. The
   *driving* grade of record is a reader trained on held-out exemplars, conditioned on the surround
   and the request, thresholded at the oracle-free q = 0.90 typicality quantile, corpus size picked
   by the self-manufactured-damage rule, swapped in at value target, mining gate and audition, and
   priced per call. Beside it, a *reporting* grader on a disjoint split and distance to held-out
   truth: images have no exact grade to report with, which is the one thing `critic/` did not have
   to solve. `at_support` next-level yield logged from day one as the endogenous pacer. Claims are
   ranks against `never_base` and the cost-to-depth curve, with in-tag twins for floors. No `given`
   arm exists without a DGP; the ceiling is not on offer.
2. **`native/` on tokens.** Both ports, the table-ablation battery, can't-decompose, next-level
   minability, on the vocabulary from node 1. The first re-ingestion result off RHM.
3. **`setlist` on tokens, then taste.** Style drift as demand drift; then the §5 critic as a grader
   with preferences inside the valid set, and one of §5's taste-gauge options in the teacher slot.
4. **Later.** The forward model comes back in two forms Jasper named: introspective planning (the
   masked model dreams a completion before committing; the horizon is how many imagined cells stay
   on-style under the held-out style model, and `span`'s horizon-vs-metric readout ports), and
   exogenous consumption of other images (`reread/lm`'s line: what a style learner extracts from
   its few exemplars as its vocabulary climbs, and the ~10×-per-half-level corpus wall). A
   CPPN-native sibling only if the Picbreeder gauge is pursued. Strokes when a question needs the
   seam law or the painter analog.

**RHM twins, run in parallel as de-risking (kicked off 2026-08-21; both reported 2026-08-22 as
`critic/`[^private] and
`endo_yield/`[^private], READMEs pending
discussion; results folded into §5 and §6 above).** Two
grader experiments RHM can still do and images cannot check: (i) the **label-free A½** queued in
[`teacher_slot/`](../experiments/rhm/practice/teacher_slot/README.md)'s Next steps — the learner's
own next-level mining yield as the gauge, read without ground-truth level-(k+1) labels (on the
sculpting substrate `native/` already logs it as `at_support`, keyed by the flat tuple and gated by
no table; `recital` finding 9 located the seed-stable predictor of what time at the bottom buys in
exactly that place); (ii) a **learned grader of record** — a reader trained on held-out
derivations from a demand D (`reread/lm` / `fourwall/lm`'s reader-beside-BP-oracle pattern) swapped
in for `units.grade` as the decision grader of an existing node, oracle logged only, so each of
§5's original four instruments is scored exactly once (done; §5 now carries the revised list). Neither gates node 1; both are meant to hand the
image substrate a calibrated instrument it cannot build for itself.

*Open*: where it lives. A new top-level substrate directory under `experiments/` with a `practice/`
child mirroring `rhm/practice/` and `mjc/practice/` (name to pick: `canvas/`, `sketch/`, or
`style/`). Fork `rhm/practice/ratchet/ratchet.py` and `macros.py` verbatim with the fork notice at
the top, per the `handle/` and `native/` convention; do not modify the donors.

**Instrument checks for the first smoke** (what the apparatus has to be able to do, not what the
result has to be): the meter binds under the optimizer actually used; the driving grader rejects
self-manufactured damage at ≤ 0.10 while passing clean exemplars at ≥ 0.90 (`critic/`'s rule), with
its pass rate on the token model's own argmax completion logged; the quantizer floor is measured and reported
beside every number; the codebook supports tuple structure at all (whether mined `T[2]` are
textures or parts is a readout, recorded either way).

## 9. Risks

Held loosely, per repo norms: places we would expect to learn something, not conditions we commit
to treat as falsifiers.

- **The alphabet may not factor.** A reconstruction-optimal codebook may be entangled enough that
  recurring tuples are textures at every level and never parts. That would say something about
  the first regime's alphabet rather than about the loop, and the raw-palette and stroke variants
  are the controls.
- **The grader may be gamed, or may be self-graded without our noticing.** §5's revised
  instruments are the defense. The precedent is `critic/`'s scarce reader: 21–41% of damaged
  configurations passed and the loop paid +0.096/+0.058, while its rank agreement with the exact
  grade stayed inside the homogeneous band. On images the rule that caught it has to be trusted
  without the exact grade beside it.
- **No ceiling.** Ranks-only grading means we cannot say what fraction of the gettable we got. RHM
  stays the twin for any op whose ceiling we want.
- **The seam may be weak.** A neighborhood carries information, but less of it than a posture or a
  pen state; `fingering`'s G1 gate (spread relative to noise, a per-state oracle ratio) should be
  re-run in its token form before commitment nodes are trusted.
- **Corpus choice confounds style with scale.** Omniglot parts are strokes; pixel-art parts are
  sprites; an artist's parts are larger than the grid. The first corpus should be chosen so that
  plausible parts span a few cells, not one and not the whole canvas.

## 10. Honest status

Nothing has been run on images. Every component named here has support somewhere in the repo: the
vocabulary machinery and both re-ingestion ports on RHM, the compile-op taxonomy and the seam law
on the arm, the verbal teacher in `teacher_slot`, and since 2026-08-22 a learned grade of record
driving the loop (`critic/`) and a label-free one-level-up gauge (`endo_yield/`), both
with READMEs pending. The assembly on images does not, and the grader in particular has only
ever operated with the exact grade logged beside it, never without.
That is the same epistemic position [practice_manufactures_its_own_credit](practice_manufactures_its_own_credit.md)
§10 recorded for the arc before its first round, and the same sequencing applies: one node,
forked verbatim, gated bit-for-bit, numbers brought back for discussion before any README.

## 11. What we ran (2026-08-25): the substrate description — [`canvas/plant`](../experiments/canvas/plant/README.md)

**Status**: the first node on canvas, discussed with Jasper before this section
was written. Not a practice loop. What it did to the sections above, in one place:

- **§3a / §9 risk 1 ("the alphabet may not factor") — measured, with a mechanism.** On the GLSL
  library a 16-px k-means codebook did not factor: 2×2 code-block recurrence tracked per-style
  code entropy (Spearman −0.84), the part-y/texture-y control was null, the depth ladder had no
  gradient past the marginals, and search bought nothing. The cause is *phase*: random crop
  offsets spell the same motif differently every time, so only translation-invariant styles
  recur. The aligned/misaligned tiles twin (one variable, the crop offset, on `tiles.py`) shows
  that when the code grid nests in the DGP's lattice the miner recovers the tile catalogue as
  its level-2 vocabulary (purity 0.93–0.98, T[3] nested-representable 0.996) and the ladder is
  graded; misaligned reproduces the GLSL null. **The alphabet has to nest, and a learned one
  can.** §1's justification 1 is answered on the aligned tiles substrate.
- **§5 (the grader) — the typicality reader is the taste organ, not the truth gauge.** Run
  unchecked as designed, `critic/`'s mean-NLL grade of record was gamed by the marginal mode on
  the GLSL library (the plant's own bland fills pass above real exemplars, worse as the reader
  improves), and against *known* validity on aligned tiles it passes invalid-but-in-style
  `seam` damage more than valid-but-atypical `offstyle`. No tail statistic fixes it: validity is
  a *relation* between tokens, typicality a *property* of tokens, and on an aligned alphabet the
  violation falls between two individually legal codes. RHM's grade was never a likelihood —
  possible-set success is a support test — and `critic/` could not have seen this because the
  possible set is flat at uniform demand. The learned analogue on canvas is an
  **adjacency-support test** (every adjacent code pair in the fill must have occurred in genuine
  exemplars): a conjunction, zero forward passes, AUC .96–.99 seam-vs-offstyle, rejects 96.6 %
  of the plant's oracle-invalid fills at the clean ceiling. Its scope condition is
  pairwise-local validity; its binding limit is coverage. §5's instrument list should be read
  with this split: support for truth, typicality for demand, `own − world` and the teacher A/B
  probe unchanged.
- **§6 (style, taste, schools) — the two organs now exist and dissociate.** The panel measures
  `typed_gaps/`'s one-organ-per-currency on the grader side: support passes rare-but-legal,
  typicality passes in-style-but-broken. Taste is real on canvas, as a grader with preferences
  inside the valid set; schools are `tiles.py`'s demand-concentrations, nine per tileset.
- **§4 (corpora, *open*) — settled: the tile grammar is the substrate.** `tiles.py` was kept as
  the calibration twin and turned out to be the substrate: nesting alphabet, graded ladder,
  depth unaffordable to single-tile repair (0.076 / 0.019 at L2 / L3), truth and taste gauges,
  style drift and truth drift, a withheld oracle, and — unlike the memo's §8 node 1 — a `given`
  arm from the tile catalogue. The GLSL library (`lib0`) is the taste venue for later.
- **§8 node 1 — re-specified.** `ratchet` on aligned tiles, support as the grade of record for
  the value target / mining gate / audition, typicality and `at_support` logged from day one,
  ranks against `never_base` and `given`. Queued; design choices in the node's next steps.

[^private]: Not mirrored: this link points to a document in the private lab repo (the roadmap, the queue, an unrun spec, reading notes, or a conversation). See the top-level README for what is held back and why.
