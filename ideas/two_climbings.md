# Two climbings: what practice climbs, what NTP climbs, and the two senses of "genuine"

**Status**: Conceptual synthesis (2026-08-25). Nothing new run. Every number is re-read from an
existing node; what is new is the decomposition of "climbing" into two objects driven by two
engines, the pinned-target / lifting-target reframe of the comparison, the split of "genuine"
into two senses the repo's data already separate, and a mechanistic reading of the
endogenous-discovery intuition. **Amended the same day against PR #69**
([`spiral`](../experiments/rhm/practice/spiral/README.md) ·
[`census`+`assay`](../experiments/rhm/practice/census/README.md)): the rate half is now measured
(three clocks disagree — §3(3), §9), and the un-giftable part of practice progress turns out to
be the *gradient-trained* part, not the discrete op (§3(2), §5, §9). **Auxiliary to
[practice_manufactures_its_own_credit.md](practice_manufactures_its_own_credit.md)** (§18) and
[meta_learning_under_metered_data.md](meta_learning_under_metered_data.md) (the 2026-08-17
bracket); to be folded into one of them as the evidence consolidates.
**Date**: 2026-08-25
**Prompt**: a question from Jasper (2026-08-23), reading the practice arc against the NTP-on-RHM
line: what does "climbing" mean in practice, is it the same thing NTP does when it learns RHM
level-by-level, and is there a sense in which practice-based progress is more *genuine* than
progress that arrives as a byproduct of NTP — given that humans treat what they discover
endogenously as more worthy of insight than what they are handed by decree.
**Attribution**: the question, the "genuine" framing, and the human-intuition observation are
Jasper's. The three-way decomposition of what climbs, the pinned/lifting-target reframe, the two
senses of "genuine", and the mechanistic port of the intuition came out of the exchange.
**Anchors**: [`rhm/practice/ratchet/`](../experiments/rhm/practice/ratchet/README.md) ·
[`rhm/practice/native/`](../experiments/rhm/practice/native/README.md) ·
[`rhm/practice/spiral/`](../experiments/rhm/practice/spiral/README.md) ·
[`rhm/practice/census/`+`assay/`](../experiments/rhm/practice/census/README.md) ·
[`rhm/practice/fourwall/lm/`](../experiments/rhm/practice/fourwall/lm/README.md) ·
[`rhm/practice/reread/lm/`](../experiments/rhm/practice/reread/lm/README.md) ·
[`rhm/practice/merge/`](../experiments/rhm/practice/merge/README.md) ·
[`rhm/practice/teacher_slot/`](../experiments/rhm/practice/teacher_slot/README.md) ·
neural_networks_learn_bottom_up[^private] ·
[`RHM_LATENT_LOOP`](../experiments/rhm/RHM_LATENT_LOOP_README.md) §5 ·
learn-from-your-own-latents notes[^private] ·
[`mjc/practice/etude/`](../experiments/mjc/practice/etude/README.md) ·
[`mjc/practice/legato/`](../experiments/mjc/practice/legato/README.md)

## One-liner

The practice ratchet and the NTP learning wave are **the same constraint projected onto two
different objects and driven by two different engines**. They look alike because both trace the
RHM tree bottom-up, and they *must* look alike, because both are forced by one fact — you cannot
identify level ℓ without level ℓ−1. But they dissociate in both directions in our own data: the
ratchet climbs the *action space* with a reader that never moves, and the NTP twin climbs the
*reader* with no action space at all. "Genuine" then splits: practice progress is **not** more
genuine in the representational sense, but it **is** in a precise measurement sense — it is
denominated in a currency (next-level representability) that NTP's own grader structurally cannot
read, and the NTP twins show that within-level loss certifies progress that is hollow in exactly
that currency. PR #69 then adds the piece that reorganises the rest: a full climb is an
**address** (discrete op, giftable) plus **trust** in the address (gradient self-imitation over
one's own use of it, on its own slow clock, only earnable) — and the deep-era value rides mostly
on the second. The un-giftable part of practice progress is its NTP-like part, run in
coordinates the op lifted.

## 1. What actually climbs

The repo has three distinct climbings, and the question only resolves once they are told apart.

| | what climbs | engine | pace | where measured |
|---|---|---|---|---|
| **NTP learning wave** | the *representation* — what the reader can parse | gradient locking onto the loudest signal; dilution `vm^{ℓ+2}` per level | endogenous, decelerating exponentially; ceiling set by `m`, not capacity | learning-wave essay[^private]; [`reread/lm`](../experiments/rhm/practice/reread/lm/README.md) finding 1 |
| **Practice ratchet** | the *action space* — what credit can address in one move: `T[ℓ] ⊆ T[ℓ−1]×T[ℓ−1]`, mined from the agent's own solved trajectories | selection + verbatim commitment under a meter (depth is *unaffordable* to primitives, not merely harder) | **exogenous** — the era clock; [`recital`](../experiments/rhm/practice/recital/README.md) found no internal signal that prices bottom-time | [`ratchet`](../experiments/rhm/practice/ratchet/README.md) findings 1–3, 7 |
| **Homeostatic migration** | where a dense learner's *gain* lands (deep vs surface) | starving samples per event | — | [`full_loop`](../experiments/rhm/directed_sculpting/full_loop/README.md) §6, 3/3 seeds, with an open compute confound (`metered_climb/SPEC`[^private], unbuilt) |

The load-bearing fact is `ratchet` finding 10: **the plant stayed inert** — parse 0.62–0.65,
infill 0.69–0.71, flat across 90 cycles in all five arms; *"the entire descent is
vocabulary-carried."* On the substrate where practice-climbing is best measured, the learner's
representation did not climb at all. It earned a better toolkit while remaining the same reader.
Conversely `reread/lm` climbs the reader ladder (tokens-to-50%-ceiling monotone d1 < d2 < d3 < d4
in every arm) with no action space whatsoever. The two climbings are orthogonal axes that happen to
be indexed by the same tree.

## 2. Why they look the same

**One shared constraint: you cannot skip a step.** In NTP it is an identifiability fact — a level-ℓ
feature is only visible through the arrangement of level-(ℓ−1) features, so until the parents are
readable the grandparents have nothing to correlate against. In the ratchet it is a
representability fact built into the data structure — `T3 ⊆ T2 × T2`, so `practice_early`'s
single bogus L2 entry produced zero L3 candidates at any cycle (finding 7). Same constraint, two
instantiations, and it is why both produce a strictly monotone level ordering.

*Scoped by PR #69.* The nesting is structural and runs both ways — completing L2 in the assay
made all 48 missing L3 entries buildable, and a Phase-0 replay found the L2 freeze caps
buildable L3 at frozen-recall². But its **cost** under a routing planner is small: `census`
finding 4 has gate-later dead (the coverage gauge bit-identical to a yoked clock hold, at ~zero
price) and commit timing cheap in both directions, because a bad level is de-funded rather than
paid for — *"the ratchet's timing law reads as an enumeration-era fact."* Foreclosure of
representation stands; foreclosure of performance was enumeration's.

## 3. Three differences that carry the distinction

**(1) Target height — pinned vs lifting.** NTP's target is a surface token, pinned at level 0
forever; dilution is fate, and the climb is a *byproduct* of the diluted signal reaching upward,
which is why it decelerates and stalls where `m` says
([belief tree](../beliefs/trees/rhm_compositional_learnability.md)). The ratchet's effective
target is the next level's minability: each committed level becomes the substrate the next is
mined over ([`native`](../experiments/rhm/practice/native/README.md) finding 7 — chosen
trajectories built of L2 calls are better mining substrate, |T3| +3.4). That is the **same move
the latent-target objective makes** — KFW's `vm³` per phase, independent of depth;
[`RHM_LATENT_LOOP`](../experiments/rhm/RHM_LATENT_LOOP_README.md) §5's *"a change of coordinates
on the supervision, moving the target up the tree in lockstep with the frontier… a native
ratchet."* So the closer kinship is **practice ≈ latent-target learning done in the action space**,
and NTP is the degenerate case where the target never lifts. The comparison is less
practice-vs-NTP than pinned-target-vs-lifting-target.

**(2) The op — continuous vs discrete.** NTP's climb is continuous descent. The ratchet's climb
has a discrete op — select a rendition and commit it verbatim — that §18 argues and
[`fourwall/lm`](../experiments/rhm/practice/fourwall/lm/README.md) measures gradients cannot
supply: *"Gradients natively supply track; merge, re-key, and mothball are absent… genuinely
missing organs, measured on an actual NTP transformer."* Averaging valid renditions destroys them
([étude](../experiments/mjc/practice/etude/README.md) 2.1×,
[crystallize](../experiments/rhm/practice/crystallize/README.md) 3.6–5.0×), and regression to the
mean is what descent does. The op acts on the *conditions* of learning (the action space, the
index), not the weights.

*Amended by the assay.* The op's **output** — the address — is exactly the giftable part.
[`assay`](../experiments/rhm/practice/census/README.md) finding 6: the full true table arriving
at commit time (`exact`) recovers only a minority of the deep-era gap to the identical table
held from cycle 1 (`given_c1`); the residual certifies at 3.24× the measured stream floor and is
carried by trust — π's L3 mass 0.580 / argmax 0.831 from birth vs ≤0.276 / 0.331 for *every*
late arrival, perfect content included. π is trained by online self-imitation on the beam's own
chosen trajectories — a gradient process. So the decomposition is: **address creation is
discrete and can be handed over; trust formation is continuous, runs on its own slow clock, and
was not shortcut by gift anywhere in the unit.** NTP has only the second process; enumeration
practice (`ratchet`, plant inert) had only the first; `native`/`spiral` have both, and the
deep-era value rides mostly on the second. The two climbings are not rivals: the practice
learner's climb *contains* a gradient climb, in coordinates the op set up.

**(3) Pacing — the tree's growth paid in two currencies.** NTP's climb is endogenous but
diluting. The ratchet's climb is non-diluting but exogenously paced: no internal signal places
the era boundaries (`recital`), and beyond the earnable range the pacers' signals are
*undefined, not noisy* ([`tall`](../experiments/rhm/practice/tall/README.md)). Neither learner
climbs "by itself" all the way — one runs out of signal, the other runs out of clock.

The rate half, unclaimed until [`spiral`](../experiments/rhm/practice/spiral/README.md), now has
a measurement, and the answer depends on the clock (finding 4): on the **certificate clock** the
crank *holds, mildly accelerating* (net cycles-to-cert 18→16, 17→15, 17→19 across arms, for a
level 4× larger); on the **coverage clock** it *decays hard* (committed recall 0.57–0.79 at L2 →
0.054–0.089 at L3, in every earning arm); on the **value clock** it *holds inside the earnable
range* (earned ≥ given-native, eras 1–3, at ~half enumeration's priced time). Spiral's
interpretation (a) is the tightest form of the "same constraint, two projections" claim: the
space of possible units roughly squares per level while the experience stream is fixed, so each
turn earns a shrinking *fraction* of a widening floor at roughly constant cycles per turn —
*"units broaden, coverage narrows — the same hierarchy fact seen from the token side and the
agent side."* **NTP pays the tree's `m^ℓ` growth as dilution** (samples per level, exponential;
coverage of a reached level high); **practice pays it as widening** (cycles per level constant;
coverage of a reached level collapsing to a demand-concentrated sliver). Same combinatorics, two
currencies — and which one reads as "decay" depends on whether the grader is a distribution-wide
probe or present demand, which is §4(b) again.

A fourth difference arrives as a measurement rather than an argument, and it runs *opposite*
between the two: the NTP reader's lower levels stay fully legible as it climbs (d1/d2 saturate
first and stay), whereas the consolidated practice learner **loses cheap access to its own
spelling** — `native` finding 3, π's mass on a proposed chunk's own primitive decomposition falls
to ~0.12 while chunk mass rises to 0.80–0.85. The can't-decompose signature is a qualitative
divergence between the two climbings, not a resonance.

## 4. Two senses of "genuine"

The word does two jobs, and the repo's data split them.

**(a) Genuine = endogenously discovered (provenance).** Partially. The vocabulary is earned
(68–98% of `given`, at 0.85× priced time), selection-filtered, and carries a *consumption prior*
— a mined 8-entry table beats a random 8 of the true 14 by ~0.20 (`ratchet` finding 8), and a
stale mined table beats the full current true table in audition
([`transpose`](../experiments/rhm/practice/typed_gaps/README.md)). But the ladder, the pace, and
the reader are all handed over. The practice learner climbs a ladder someone else built, at a
pace someone else set, reading with perception someone else supplied. §18's own line concedes
it: *the factory cannot manufacture the price of its own next level, which is why every ladder
tops out at a teacher.*

**(b) Genuine = not hollow.** This is the sharp, defensible sense, and it is the 2026-08-17
bracket's reframe (i): *"correct" was graded by a within-level signal.* Three measurements:

- `fourwall/lm` finding 5: keyed arms sit at Bayes-level task error with the token-derived
  inference pathway at d4 = 0.27–0.30 against controls at 0.83–0.84 — *including `true_wall`,
  which never experienced a rotation.* Hollow at Bayes error, invisible in the only currency the
  loss reads.
- [`merge`](../experiments/rhm/practice/merge/README.md) round 1: task error identical under
  rotation; next-level representability |T3| = 12 vs 67.
- [`teacher_slot`](../experiments/rhm/practice/teacher_slot/README.md) `outer_task`: reading its
  own experienced loss, the outer loop refuses the merge all five times, posts nearly the arc's
  best within-level lifetime ledger (1.4690), and ends threefold hollow — *"the ledger doesn't
  fail to fund the crossing, it votes it down optimally."*

So NTP-style progress *can* be hollow in the next-level currency while perfect in its own, and
the practice learner's progress is by construction graded in the next-level currency — mining
yield, the arc's most seed-stable fact (`recital`). **This is a claim about measurement, not
virtue.** The monolith is not worse at what it is graded on; its grader cannot see the thing that
is missing.

*Recursed by the spiral, and kept honest by the assay.* The hollow-grader result is not a
property of NTP; it is a property of any within-X gauge, and X moves up a rung per round.
`spiral` finding 6: the practice learner's own certificate is **satisfiable by concentration** —
it certified level 3 "finished" on <10% of the level in every earning arm, with the priced
audition, the unpriced oracle audition, and the true-macro ceiling agreeing to 0.02–0.04, so
this is not mis-measurement. `census` then found its extension audition satisfied by
harmlessness-under-present-demand (bit-flat on 8 of 12 admitting events). Its interpretation
(b): *"every audition-shaped gauge prices what current demand exercises, and coverage-for-the-
future is precisely what it cannot price."* Introspection-is-level-bounded, one rung up. What
the practice learner has is not a non-hollow grader but the *next* grader — and there is always
a currency above it.

The twist that keeps this from being a tidy story: the coverage-hollowness the certificate could
not see turned out **not to be the deficiency**. Buying 2.5× the L3 table closed ≤0.08 of the
eras-4–5 bracket (`census` finding 2); the gap was arrival (`assay` finding 6). So the currency
a grader is blind to is not always the one that matters — `spiral`'s own reading of its finding
5 ("the measured value of coverage beyond demand") was overturned by its child. §4(b)'s claim
generalises; its application has to be re-measured each time.

## 5. The human intuition, ported as a mechanism

The intuition — what you discover yourself is more worthy of insight than what you are handed —
does not need to port as a bias. The arc gives it a mechanism: **endogenous discovery comes with
three properties for free that decree has to buy separately.**

1. **Concentration.** What you mined is indexed to what you actually demanded — the consumption
   prior above. The exhaustive true vocabulary is *possible*; the mined one is *probable*, and in
   audition the probable beats the possible.
2. **Representability in your own terms.** Mined entries are built over your own lower-level
   vocabulary, so they are guaranteed addressable by your own credit system. The measured
   negative is [`handle/`](../experiments/rhm/practice/teacher_slot/README.md): chunk identity
   piped into the executor as a dense supervised target moved the plant *negatively*,
   dose-ordered — being told the answer in someone else's coordinates interferes. The fix was to
   *route* to it (`native` Port 1), not to teach it.
3. **Quarantine.** `native` finding 6: a proposal head trained only on solved trajectories
   starves a bogus address to ~0.03 mass (6× fewer calls), whereas enumeration is forced to
   materialise it at every tip — *"its action set is its exposure."* What you found by selection
   is self-filtering; what you were handed is not. *Refined by `census` finding 3*: the
   quarantine has **level granularity** — π's action space is one slot per (level, node), so it
   vets levels, not entries, and a precision-degrading addition de-funds the whole level, true
   entries included (the arm holding the *most* L3 entries ended with the *lowest* L3 proposal
   mass). The one real vocabulary hazard is dilution of a level's reliability, not individual
   bad addresses — and the executor DP filters those anyway (a 6.2%-true table yields 71–76%
   true executions, `assay` finding 5).
4. **Trust — the dominant one, and only earnable.** The assay dissociates the intuition from
   provenance. `given_c1` and `exact` are *both* gifts by decree, holding the identical perfect
   table; they differ only in when it arrived, and the early arrival wins most of the deep-era
   gap. Content is giftable and buys a real minority (`complete` − `anchor` +0.23/+0.31, 3.56×
   floor); truth matters little (the DP filters); **use is not giftable.** Endogenous discovery
   bundles use with content by necessity — `mine_from = chosen` means you cannot mine a chunk
   you have not already executed in a solved trajectory — while decree can deliver content
   without it. So "what you discovered yourself is more valuable" is, on this data, tracking
   **time-in-use**, and provenance is its proxy.

None of this is mystical, and it should not overclaim: `given` is the best arm on raw error in
most cells, and `ear`/`recital` beat it via timing and schedule rather than via earning per se.
The earned advantage is on **growth, concentration, cost — and trust** — the address-book half
of `native` finding 5 (deleting the table costs +0.000–0.008 e at the current level and
collapses the next one; reproduced on a live-earned vocabulary in `spiral` finding 8) plus the
arrival half of `assay` finding 6 — not on current-level content. The natural next instrument,
queued in `census` and not run: the trust-formation rate itself (π's per-level mass growth
after arrival) and whether targeted rehearsal of a *received* vocabulary moves it.

## 6. What cuts against, kept on the record

- **The rate half of compounding is measured, not resolved.** Three clocks disagree (§3(3)):
  "each earned level makes the next cheaper" holds on the certificate clock, decays on the
  coverage clock, and holds on the value clock only inside the earnable range. The L2
  cert-cycle deltas between arms (1–2 cycles) are inside plausible stream noise, and every L3
  cert-cycle arm ordering at depth 6 sits under an 11–14-cycle stream floor (`assay` finding
  7) — the cross-level and cross-clock *structure* is the claim, not any rate number.
- **No head-to-head exists.** `given` is an oracle-vocabulary practice learner, not an NTP
  learner. The `*/lm/` twins ask a different question — which practice ops a gradient reader
  supplies natively — and answer it (track yes; merge/re-key/mothball no).
- **In `ratchet` the reader is handed over.** *"Everything above level 1 is earned. That line is a
  design decision, and a different line would give a different earned-vs-given fraction."*
  Still true of every node through the assay: the plant guard is flat by design wherever
  routing lives.
- **Hygiene: the depth-6 stream floor grows with era depth** (displacing RNG position alone
  moves eras-3–5 recovered fractions by up to 0.087 in the `exact` family and 0.083/0.148/0.344
  in the `given` family). Fractions, shallow-era orderings, and cert-cycle orderings are
  demoted; ranks and signs in the deep eras are the currency. Most numbers in this doc are
  depth-4 ranks and signs; none of the argument rests on a fraction.
- **A sighting, not a finding** (`census` finding 8, uninterpreted there): `given_c1` executes
  macros 2.4× more than any commit-time arm and observes the *fewest* distinct L4-shaped tuples
  (279–318 vs 482–542), stable under displacement. Trust concentrates trajectories — the
  `merge/` round-2 self-play narrowing showing up inside the practice learner. If it holds, it
  is the practice learner's own hollow-in-the-next-level signature, and it cuts against reading
  trust as pure upside.

## 7. Across domains

- **RHM** is the one substrate where both climbings are measured against the same tree — hence
  the dissociations in §1.
- **mjc**: the level is not a grammar rung but committed span against the plant's composition
  horizon (~21 steps), and the hierarchy is *manufactured* by the learner at re-grounding points
  rather than given by a DGP. [`legato`](../experiments/mjc/practice/legato/README.md)'s crossover
  says chunks pay only *beyond* the model's reach (1.83×) and lose inside it (1.25×); étude
  finding 2 says the regression/BC analogue of NTP does not produce the unit at all. In a motor
  domain "climbing" means extending committed execution past the horizon, and nothing
  NTP-shaped does it.
- **Language / data domains**: the learning wave is the empirical fact (bottom-up, ceiling set by
  data ambiguity). The transferable practice result is `reread/lm` — a corpus has a depth
  capacity, extraction is level-ordered, re-reading is free extraction up to a wall at ~10×
  distinct corpus per half level. *Free data ≠ free extraction*: a token's value is indexed by
  the reader's current vocabulary. §18's "correct degenerate solution when the meter is off"
  stands, amended by the bracket to *"mistakes free data for free extraction."*

## 8. Open

No run pits a practice learner against an NTP learner on the same RHM at the same budget with
both reader and action space free to climb. `native/lm/` (tokenizer expansion as the
native-primitive op, queued in `native`'s next steps) is the natural vehicle, and
[adaptive_core](adaptive_core_and_hierarchy_climb.md) §6's two-instrument discipline — read
repair cost *and* the depth probe together — is the right shape for the readout. After PR #69
the instrument that most directly bears on §5 is the trust-formation rate (π's per-level mass
growth after arrival), queued in `census`. Per repo norms no interpretation is pre-registered
here.

## 9. PR #69 in one place: three clocks, and use is only earnable

The direct test §8 named ran the same week
([`spiral`](../experiments/rhm/practice/spiral/README.md) at depth 6, then the
[`census` → forensics → `assay`](../experiments/rhm/practice/census/README.md) unit). What it
did to each section above:

| doc claim | before | after PR #69 |
|---|---|---|
| §2 shared constraint | nesting forecloses the next level; committing early is worse than never | nesting stands, both ways (L2 completion makes all L3 buildable); its *cost* is enumeration's — under routing, commit timing is cheap in both directions and a bad level is de-funded, not paid for |
| §3(1) lifting target | argued from one turn of the crank | measured live across two turns: the L3 observation stream *grows*, routed arms are the better mining substrate, the address book survives on a live-earned vocabulary |
| §3(2) the op | the discrete op is what gradients can't supply | still true — but the op's *output* is the giftable part; what cannot be gifted is trust, which is gradient self-imitation over one's own use. A climb = address (discrete, giftable) + trust (continuous, only earnable) |
| §3(3) pacing | rate half unclaimed | three clocks: certificate holds/mildly accelerates, coverage decays hard, value holds inside the earnable range. NTP pays the tree's growth as dilution; practice pays it as widening |
| §4(b) not hollow | NTP's grader can't read the next-level currency | any audition-shaped gauge can't read beyond present demand — the practice certificate is *satisfiable by concentration*, the extension gauge by harmlessness. And the currency a gauge is blind to (coverage) was *not* the deficiency (arrival was) |
| §5 the intuition | three free properties of endogenous discovery | a fourth dominates: **trust, only earnable**. Provenance is a proxy for time-in-use; quarantine is level-granular |

The compressed form, from the unit's own synthesis: *the value of a vocabulary is mostly not in
the table — it is in the policy that grew up with the table; use is only earnable.* For this
doc's question that reads: **the part of practice progress that cannot be handed over is its
NTP-like part — a gradient process — run in coordinates the discrete op lifted.** The two
climbings are not rivals; one contains the other.

[^private]: Not mirrored: this link points to a document in the private lab repo (the roadmap, the queue, an unrun spec, reading notes, or a conversation). See the top-level README for what is held back and why.
