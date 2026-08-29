# practice — the practice arc on RHM

**Up**: [../README.md](../README.md) (rhm)
**Idea doc**: [practice_manufactures_its_own_credit](../../../ideas/practice_manufactures_its_own_credit.md)
**Sibling arc**: [`mjc/practice/`](../../mjc/practice/README.md) — where the arc started, on MuJoCo
control; [`mjc/practice/etude/`](../../mjc/practice/etude/README.md) is the direct parent of
everything here.
**Files**: [FILES.md](FILES.md)

## Scope

Practice as a control loop wrapped around an ordinary learning rule — acting on the *conditions and
units* of learning rather than on the model — ported onto the RHM sculpting substrate. The move here
is deliberate: the étude closed with a precondition it could not test on MuJoCo (*hierarchy is
meaningful only over boundaries that carry information*; its committed units were state-independent
command sequences, so post-commit drift was exactly 0.0000 and fusion was provably vacuous).
Sculpting removes that degeneracy — a committed unit is a move program executed by a generator that
reads the observed configuration — and adds instruments no other substrate has: an exact DP oracle,
a known true rule vocabulary, and an exactly-known aleatoric floor.

## Children

### [`crystallize/`](crystallize/README.md) — certificate-gated compilation under priced feedback (2026-08-14)

**Goal**: instantiate the étude's compile op — δ-silence certificate, selection-not-averaging,
scoring under the consumption distribution — on a production/control task with priced feedback, and
grade it against never-compiling on success × priced time.

**Finding**: the étude's open question is answered — **state-conditioned commitment (a library keyed
by the observed target) beats state-independent commitment by 1.8–3.0× in 6/6 commit states**, and
averaging valid realisations destroys them by 3.6–5.0×, with RHM giving the mechanism exactly (the
modal token span went off-grammar in half its blocks while every contributing realisation was fully
on-grammar). But the certificate turns out to have nothing to certify: with the plant frozen and only
the selector learning, **committing at cycle 1 matches every gated arm at 26× less priced time**, and
a shadow-compile instrument shows the committable content of practice traces is flat from cycle 1
while the closed-loop policy improves by 0.12–0.14. The interpretation we settled on (argued, not
measured): **δ-silence gates compilation only where practice moves the executor** — the étude's
practice trained the forward model its ballistic units bet on, this one trains the judge; practice on
a frozen plant makes you better at improvising, and improvisation is what does not compile. A
precheck closes the loop: let the plant learn and the committed-unit ceiling moves at 31–66× the
metering noise floor, which is what the next round is built on.

### [`ratchet/`](ratchet/README.md) — earning a level-indexed vocabulary, one depth era at a time (2026-08-14)

**Goal**: give practice something it can actually move — its own **action space** — by mining a
level-indexed macro vocabulary from the agent's own successful repairs, over a depth-laddered damage
schedule, and grade it against the DGP's true vocabulary handed over for free.

**Finding**: **earning the vocabulary recovers 68–98% of what being given it buys**, at 0.85× the
priced time, and it flattens the cost-to-depth curve 4.1× against base moves (error growth +0.072
across three depth eras vs `never_base`'s +0.293) in a regime where depth is *unaffordable* rather
than merely harder — the base action space's privileged exact-DP oracle and its width-16 beam at 5.9×
the declared budget both lose to one level-2 macro at width 1. Commitment timing matters
catastrophically: **committing one cycle into an era is worse than never committing at all**
(earned-vs-given fraction −0.372), because a compiled level-2 error forecloses level 3's
*representation* — the learned tables are nested (`T3 ⊆ T2 × T2`), so a frozen bad vocabulary makes the
next level unrepresentable rather than merely worse. The unit-LP certificate fires within one cycle of
its offline prediction at level 2, and **correctly refuses at level 3**, where the macro is auditioned
on shallower damage than it was built for and the audition *understates* it by 1.75× — the étude's
seam law with its sign flipped, and the refusal costs the era. The plant stayed inert throughout
(a manufactured frontier was measured and failed: the agent must be able to *fund* crossing it), so
the whole descent is vocabulary-carried, which is round 1's scope condition satisfied by the action
space rather than by the executor.

### [`ear/`](ear/README.md) — climbing the evaluation layer (2026-08-15)

**Goal**: build the grader the ratchet's level-3 refusal said was missing — audition contexts and
evaluators that match the consumption distribution — and grade certificate-driven commitment
against provisional commitment at the era boundary.

**Finding**: climbing the grader fixes the **measurement** (a consumption-matched in-policy
audition is calibrated to 0.97, completing the seam law's third coordinate) and not the
**decision**: every certificate arm still refused level 3, because an arm already holding level-2
vocabulary leaves a new unit almost no learning-progress headroom to demonstrate. What rescued
era 3 was **committing provisionally at the boundary and letting consumption grade** —
`practice_prov`/`practice_climb` are the arc's first arms to beat `given` — and the recert safety
net fired 0 times in 24: boundary commitment is safe because mining is selection-filtered
upstream. Evaluation itself costs 5–29% of priced time.

### [`recital/`](recital/README.md) — self-paced eras (2026-08-15)

**Goal**: remove the era clock — each arm holds one priced budget and decides when to advance the
depth ladder; advancement policy is the only variable. Can any internal signal place the
boundaries?

**Finding**: **none of three internal signals prices what time at the bottom buys** — the
certificate advances earliest and finishes bottom-half in all three independent worlds
(commit-readiness ≠ level-exhaustion), task-progress starves at the bottom in 2 of 3 (degenerates
to never advancing), and the vocabulary-growth pacer behaves exactly as predicted and is still
early (saturation ≠ sufficiency). A fixed bottom-heavy schedule (50% of budget in era 1) is
**rank 1 of 8 in every world** and beats the DGP's own tables outright in both independent
worlds; corr(first boundary, mean error) = −0.94/−0.69/−0.79. The mechanism check rules out "late
buys the L2 table" (the post-saturation increment prices at zero) and locates the seed-stable
predictor in the **next level's** table — a quantity no within-level signal reads. Methodology
export: recovery *fractions* at one RNG stream position carry ±0.15; rank orderings are the
currency.

### [`tall/`](tall/README.md) — the depth-6 port attempt (2026-08-15→16, closed)

**Goal**: break the arc's standing confound — at depth 4, level 3 is both the deepest earnable
level and one below the grammar's ceiling, so frontier effects and boundary artifacts cannot be
told apart. At depth 6 the ladder (5 levels) outruns the earnable range (2–3).

**Finding**: the setting was established by measurement (m=4 is **inadmissible** for sculpting —
synonymy flattens the depth ladder 1.06× vs m=2's 3.25× and walls off every level above 3), the
structural gates all passed, and the main run was **voided**: the value head is signal-starved at
64 tokens (stale-rollout success 0.076 vs depth-4's 0.296) and no arm sculpted. A densification
gate showed the fix direction is real but insufficient, and surfaced a pricing inversion
(committing grows the action set and costs beam width more than the macros are worth at this
scale). Two regime-independent findings stand: **endogenous pacers cannot traverse a ladder
longer than the earnable range** (their signals are undefined there, not noisy), and **the mined
table cannot churn** (recital's same-table-or-different question retired a priori). The
boundary-vs-frontier discriminator remains open; the full re-run recipe is recorded.

### [`transpose/`](transpose/FILES.md) — truth-news: within-run grammar drift (2026-08-16)

**Goal**: install the news gap every prior round lacked — hard rule-cell resampling mid-run, so
committed content can become wrong after commitment — and re-run the commit-policy comparison.

**Finding**: level-2 drift is **inadmissible** (it degrades the executor, not the vocabulary —
arm separation collapses to noise while the DP floor itself moves); at admissible level-3 drift
the static ordering survives, the post-commit flat baseline breaks by **world-hardening rather
than invalidation** ((held − true) ≤ 0.05, sign-inconsistent; a 36%-invalidated frozen table
tracks the current truth), recert stays near-silent (1/62), and a mined *stale* table **beats
the full current true table** in audition — concentration dominates currency. Findings record:
[`typed_gaps/`](typed_gaps/README.md); machinery: [transpose/FILES.md](transpose/FILES.md).

### [`setlist/`](setlist/FILES.md) — demand-news: the consumption distribution drifts (2026-08-16)

**Goal**: the complement — grammar fixed, OU drift on the derivation distribution, so *what is
asked* moves while nothing becomes false (precision-vs-truth pinned at 1.000 by gate).

**Finding**: the verification channel wakes under its native currency — **9/62 recert swaps,
every one coverage-improving**, and ablating recert costs **Δ0.271 deep error (≈8× noise)** on a
clean isolation; the oracle bracket prices concentration (tracking 1.00–1.24 vs frozen 0.35–0.75
earned-vs-given); the earned, recert-maintained vocabulary finishes rank 1, matching the tracking
oracle at 7.9% grader cost; forgetting-by-decay costs coverage while audit-and-reselect works.
Findings record: [`typed_gaps/`](typed_gaps/README.md); machinery:
[setlist/FILES.md](setlist/FILES.md).

### [`typed_gaps/`](typed_gaps/README.md) — the crossed pair, jointly written up (2026-08-16)

**Goal**: test §18's conditioning-gap caveat — no round had ever combined the compile op with a
live news channel — by running the two candidate currencies of news as a crossed pair.

**Finding**: a **double dissociation** — truth-drift degrades the dense learner (plant parse
0.62→0.50) and leaves recert silent (1/62); demand-drift leaves the plant flat (0.59→0.60) and
fires recert (9/62). Chunks are not beliefs: they store demand-concentration, so maintenance of
committed skill is **demand-tracking, not truth-tracking**, evaluative rather than entropic.
Pre-commit certification stays demoted (third consecutive frontier refusal); post-commit
verification is rehabilitated **currency-specifically**. The conditioning gap matures into a
type system — one organ per currency of change (dense learning ↔ truth; repair/metering ↔
interface; evaluative re-selection ↔ demand; the teacher ↔ level). Single seed; ranks are the
claim.

### [`reread/`](reread/README.md) — is a fixed archive renewable as the vocabulary climbs? (2026-08-17)

**Goal**: test the metered-data caveat's reframe iv (a token's extractable news is indexed by
the reader's current vocabulary) by holding an archive byte-frozen across 9 passes and reading
mining yield, competence, and a paired frozen-vs-fresh novelty probe against vocabulary stage.

**Finding**: on sculpting, the archive is **renewable in competence, not vocabulary** — yield
decays to zero with no re-arm at commits while metered error keeps falling (gains 0.19–0.50
where eight dense epochs buy ≤0.09) — a negative scoped by the instrument (the exogenous reader
rewrites what it reads, so vocabulary-gated perception is inexpressible there). The child
[`reread/lm/`](reread/lm/README.md) rebuilds the question with an **endogenous reader** (NTP vs
exact BP oracles) and the claim reproduces: extraction is strictly level-ordered in every arm,
and a 6.4×-re-read corpus is **fully renewable at zero token penalty** (two independent
instruments) — bounded by a corpus-size wall below which re-reading is *capped*, not slowed
(**~10× distinct corpus per half level of achievable depth**). Side finding with legs: archive
depth-mix is a curriculum — a deep-only archive mines a worse-than-random L2 that structurally
forecloses L3 ([`recital`](recital/README.md)'s bottom-heavy law on the data side). Single seed.

### [`merge/`](merge/README.md) — the merge op, enumeration vs merge, and self-play's holes (2026-08-17)

**Goal**: build the index-coarsening op ([recurrence_manufactures_confounds](../../../ideas/recurrence_manufactures_confounds.md)
§5/§8) and test §18's "correct degenerate solution" clause: is invariance-by-enumeration the
same object as invariance-by-merge, and is free data free coverage?

**Finding**: two dissociations. **Round 1**: rotation and merge are substitutes for the quotient
on *task error* — but not on the next level: at every rotation rate the enumerated index's
operative cell supports |T3| = 12 vs the merged 67 against its own union of 68 — **the content
is collectively there; the index withholds it** (§16's currency claim measured on the index
axis). The merge op refuses genuinely distinguishable cells; **paced rotation is what renders
them indistinguishable** — manufactured decorrelation as the licensing condition for merging,
measured as mechanism (18→1 collapse, 8× less storage). The affordance-matched unmetered
monolith ties in-distribution and keeps full minability — it never fractured, having never been
forced to key. **Round 2**: an unmetered learner whose practice distribution is its own
competence-amplified footprint (β=2) is rank 1 by its own grading (beats the DGP's table) and
second-worst on coverage, with holes 30–70× starved, positive tail excess, and 18% less of the
next level representable; a yoked placement-matched, strictly-narrower exogenous control
reproduces only 22% of the loss — **the damage is the competence coupling's collapse dynamics,
not narrowness** — and the onset is a cliff between β=1.5 and β=2. Self-grading inflation
(`own−world`) turned out to be a narrowness artifact (correction on the record). Single seed;
ranks and signs are the claims.

### [`fourwall/`](fourwall/README.md) — the spurious index: scaffold, debt, merge, re-key, retire (2026-08-17)

**Goal**: first contact for [recurrence_manufactures_confounds](../../../ideas/recurrence_manufactures_confounds.md)
— manufacture a spurious-but-free library key (a "wall" token perfectly correlated with the true
level-2 latent under narrow demand), rotate the correlation away unannounced, and measure the
scaffold's value, the confound debt, and what "whittling toward the compressed form" actually
consists of.

**Finding**: the scaffold delivers **oracle-index value for free** while the correlation holds
(≈`given_key` within noise, +0.10–0.13 over no index) and the debt is real (+0.19–0.41 per
rotation); "whittling" then decomposes into **three ops, of which the idea doc named one** —
*merge* deletes distinctions but can only recover the no-index baseline (deletion cannot build),
*re-key* moves the library onto a basis earned by functional addressing ("same unit serves them")
and closes a third of the terminal gap, becoming rotation-proof, and *retire* recovers the
maintenance rent on the abandoned scaffold (16→10% of priced time, error improving), with
mothball-vs-delete a priced bet on demand recurrence — the c121 identity return revives mothballed
content and cannot revive deleted content. A good scaffold **raises the bar for its own
replacement** (rational stickiness under a consumption-graded gate; a coverage-graded gate locks
it in forever — the seam law's evaluator coordinate applied to the index), the earned quotient is
**capped by the unit vocabulary** (ARI ~0.5–0.6; the address ladder is funded by the action
ladder), no spontaneous binding formed (available ≠ taken), and tear-down preceded stress in every
round that had the op. Single seed; ranks are the claim; the node also measured stream-position
sensitivity exceeding the ±0.034 floor in adoption timing, which is the seed pair's named job.

**Update (2026-08-18, [`fourwall/lm/`](fourwall/lm/README.md))**: the endogenous twin — the same
shape asked of a real NTP reader against exact BP oracles (`reread/lm`'s pattern), then the missing
op supplied exogenously and priced. The binding null **inverts** (available *is* taken — instantly,
to the full exact bracket); the reader tracks and **never quotients** at any rotation rate, holds
no mothball (the identity return costs full price), and ends with its token-derived inference
pathway threefold suppressed, invisibly to task error; the supplied merge op (an input-stream
collapse — conditions, not weights) repairs exactly that failure at ~zero price, while the
within-level lifetime ledger *rewards* never merging — **the op and its justification must both
come from outside the loop**. (One correction on the record: round 1's `wall_fast` penalty was ~8×
a checkpoint-grid artifact; corrected +0.019, aligning the endogenous reader with `merge/`'s
substitutes-on-task-error.)

### [`teacher_slot/`](teacher_slot/README.md) — the teacher slot decomposed: gauge-choice, the learner's own currency, and the verbal channel (2026-08-20)

**Goal**: split `fourwall/lm`'s ending — the merge op *and its justification* must come from
outside the loop — into testable parts: will a minimal outer loop fund the crossing given the
right currency (Rung A); can the currency be the learner's own next-level yield (A½); can a
verbal reasoner make the call pre-rotation, when nothing in-world marks the key as spurious
(B1); and was the arc's inert plant just a missing backprop handle (`handle/`).

**Finding**: the teacher's scarce part is not judgment but **which gauge is consulted**. A
thermostat-grade paired-trial rule reading any one-level-up currency — the pathway readout or
the learner's own d5 next-level yield (at a measured 13× read premium, 9.6% of budget) — merges
at 1500–2000, *before the first rotation*, with zero bail-outs, and an unrotated control merges
identically; the same rule reading the learner's own experienced loss **refuses the merge all
five times it is offered**, posting nearly the arc's best within-level lifetime ledger (1.4690,
second only to `true_wall`) while ending threefold hollow — the ledger doesn't fail to fund the
crossing, it votes it down optimally. A blinded text-trained reasoner (pinned Opus, neutral
vocabulary, 0/3 removals on a dead-token control, 0/2 on a late join) then removes the key 6/6
sessions before any disruption exists — including 3/3 on the task-only information set the
numeric refuser had — on stated grounds about what the steering readout should *mean* ("I would
rather pay a visible loss increase now than carry an invisible dependency that I am the sole
reason still works"), spontaneously constructing a span-A-vs-span-B contrast instrument the
numeric loop lacked: the teacher slot is a gauge-choice port, and post-ratchet text carries the
priors that fill it. Sideline: giving the ratchet's earned macros real vocabulary slots makes
the inert plant move *negatively*, dose-ordered by engagement — a gradient path buys
interference, not the hoped-for plant learning. Two corrections queued for the fwlm ledger
(the `wall` lifetime integral was grid-inflated; never-merge's advantage is larger than
published). Single seed; one world; A-m and B2 queued.

### [`native/`](native/README.md) — the port back: consolidating the earned vocabulary into planner and executor (2026-08-20→21)

**Goal**: close gap 6 of the arc — every earned-vocabulary node kept the vocabulary outside the
learner, as a table consumed by an exogenous beam. Can it be consolidated into the agent's own
planner (a proposal head routing to chunks) and executor (a span head running the corridor in one
pass), while the table remains the address book the next level is mined over — and was `handle/`'s
interference a fact about consolidation or a type error?

**Finding**: **all three clauses land, and one prediction inverts.** Routing beats enumeration
outright at budgets matched to 0.3% (e 0.084/0.105/0.156 vs 0.194/0.228/0.346 — the freed
groundings buy width; 0.32× cost to match), and the control is decisive: the same head over
primitives buys 2.2–2.5× less — **the chunk is the object**. The corridor consolidates into the
executor with Δparse at/below floor where `handle/`'s identity port ran 3–9× floor dose-ordered —
the §3½ identity/corridor split is the variable (L2 reaches parity; L3 mostly does not). After
consolidation, **deleting the table costs +0.000–0.008 e at the current level** (untrained-head
control +0.40–0.51) while built T3 collapses structurally and the mining observation stream
survives — the address book is needed for growth, not performance, in both directions. The
can't-decompose signature appears (π's mass on a proposed chunk's own spelling falls to ~0.12 —
the arc's first expertise-psychology readout), and native L2 routing *raises* L3 minability
(|T3| +3.4, committed recall 0.571 vs 0.500). The inversion: a natively-routed *bad* table is
**better** than a frozen one in every era — π, trained on selected trajectories, starves the bogus
address to ~0.03 mass (6× fewer calls) where enumeration is forced to keep paying for it —
selection-before-regression is also a quarantine. Single seed; ranks, signs and floor-multiples
are the claims.

### [`spiral/`](spiral/README.md) — the live re-earning spiral on the depth-6 substrate (2026-08-22→23)

**Goal**: run the loop the arc had never run — earn level k live, consolidate it into the
learner's own planner and executor, earn level k+1 natively over the routed policy — in the
depth-6 world (`tall/`'s recorded recipe), where the damage ladder outruns the earnable range;
read whether the crank accelerates, holds, or decays per turn, and which wall bites first.

**Finding**: **the crank holds, narrows, and the certificate cannot see the narrowing.** Routing
rehabilitates the substrate `tall/` closed (descent gate passes at 0.50–0.59 recovered fraction;
enum reproduces the broken reference in-run) and dissolves its pricing inversion live (enum's
commit penalty permanent, routing's a 2-cycle transient; the g722 insurance arm buys nothing).
The spiral turns twice on its own certificates, at roughly constant cycles per turn — but
committed coverage falls 0.57–0.79 (L2) to **0.05–0.09** (L3) in every earning arm while the
certificate fires anyway: the audition-shaped gauge is **satisfiable by concentration**, the
grader wall in a sharper form than "noisy at depth". The two predicted walls did not bite (the
observation stream *grows*; L3 corridors reach parity 0.99+ where depth-4's never did). Inside
the earnable range the earned sliver matches or beats the true-table native ceiling at ~half
enumeration's priced time (earned ≥ given, natively); beyond it the complete vocabulary pulls
away by +0.3–0.4 recovered fraction — the measured value of coverage past the demand it was
earned on. The address book survives live (table deletion ~free, corridor deletion not); the
both-ports arm underperforms routing-alone in consumption (native's unresolved cell, now a
two-node pattern). Single seed; ranks, signs, floor-multiples are the claims.

### [`census/`](census/README.md) + [`assay/`](assay/README.md) — amount × truth × arrival, crossed: arrival wins (2026-08-23→25)

**Goal**: the spiral's certificate certifies a level on a <10%-coverage sliver while the complete
vocabulary wins the deep eras — so is the deep gap a coverage gap? Census: can an endogenous
gauge buy coverage, and does bought coverage pay? Forensics: when it didn't, why not. Assay:
oracle surgery on the committed table, isolating which ingredient of `given`'s advantage —
amount, truth, or arrival — the deep eras actually pay for, certified against a measured
stream-position floor (single super-writeup: [`census/README.md`](census/README.md)).

**Finding**: **arrival dominates — the value of a vocabulary is mostly not in the table but in
the policy that grew up with it.** Coverage is buyable (extension: 2.5× the L3 table) and does
not close the deep gap; the failure is no artifact, and its mechanism is structural — **π vets
levels, not entries**, so impure additions de-fund the whole level while the executor DP already
filters junk at execution (a 6.2%-true table yields 71–76% true executions). The decisive cell:
the full true table arriving at commit time recovers only a minority of the gap to the same
table held from cycle 1 (era-4 residual **3.24×** the measured floor; deep-era rank order
invariant under stream displacement), and the carrier is trust — π's L3 mass 0.58/argmax 0.83
in the from-birth arm vs ≤0.28/0.33 in every late-arrival arm, robust under displacement.
Gate-later is dead (criterion epiphenomenal, hold free); the gauge blindness recursed (an
audition under present demand cannot price future coverage); and the unit measured the depth-6
stream floor itself, demoting fractions, shallow-era orderings, and all L3 cert-cycle arm
readings across three rounds. Use is only earnable: content transfers, trust does not. Single
seed; the stream twins bound stream noise, not seed noise.

### [`conductor/`](conductor/README.md) — A1: the composition; the outer loop drives the crank (2026-08-27→28)

**Goal**: compose the arc's first two pillars in one node (ROADMAP §4.1, shape A1) — the
`teacher_slot` outer-loop rule, charged for what it reads with dead zones at measured floors,
sitting over the routing-only depth-6 practice learner and owning commit/hold per level and era
advance; the certificate demoted to a read-only instrument; arms differing in nothing but what
the rule reads; a yoked-clock control per gauge arm.

**Finding**: **the one-level-up reader is the one that can drive.** The free at_support gauge
holds each commit ~20–30 cycles past the certificate and stretches the earning eras, buying
bigger tables at unchanged precision and the most next-level minability of any arm — matching
the scheduled anchor inside the earnable range (era-3 Δ exactly 0.000) and beating it where
demand outruns what was earned (era-5 +0.251 recovered fraction, **2.9× the earning-family
stream floor**). The within-level reader re-derives the refusal in a sharper form: it commits
and exits early on its own quieting slope, never arms again, and ends worst on value with the
fewest next-level observations — a reader that starves the level above it. The raw one-level-up
NLL never arms at all (an absorbing action has no second condition to cancel plant drift; its
excess form licenses one commit and saturates), and both yoked clocks are **bit-identical** to
their gauge arms — census finding 4 generalized to the full action set: the criterion's whole
contribution is the cycle numbers it writes, which is exactly the part only a gauge can write
endogenously. Single seed; ranks, signs, floor-multiples are the claims; arms are not
lifetime-matched (pacing is what the loop owns and pays for).

### [`maestro/`](maestro/README.md) — A2: learn the rule (2026-08-28)

**Goal**: replace A1's hand-written thermostat with a small learned policy over (gauge
readings, level state) rewarded by next-level yield — fitted offline on the logged corpus of
prior crank turns, since the actions are absorbing and within-run trial-and-error is
structurally impossible — against A1's rule in-tag (bit-for-bit replica) and against a
within-level-rewarded twin of the identical class as the negative control.

**Finding**: **the reward's type makes the judge; learning adds a timing refinement
gauge-choice already paid for.** From identical inputs, class, and fitting, the two rewards
produce near-orthogonal policies: the yield-rewarded mixture predicts forward next-level yield
11–13% better than the raw gauge out-of-arm and paces the crank at least as well as the
thermostat at ~8% less priced time by committing *earlier with thinner tables* (the
census/assay arrival mechanism, produced by an endogenous choice — its early L2 commit bought
2.2× the L4-shaped observations by era-1 exit); the within-level twin commits before its own
certificate and **never takes L3 while its own input reads the yield gauge at 2+ floor units**
— the crossing's value is not invisible to a within-level objective but *unvalued* by it.
Stream-displaced twins then police the node's own cells: each gauge-driven arm's era-5 win
over the schedule survives displacement above the floor (2.3–3.8×), the era-5
learned-vs-thermostat ordering flips sign and is demoted, and era 4 is the cell where
learning's edge holds on both draws. Single seed; the twins bound stream position, not seed;
6 logged commit events mean the learned object is the value of states, not of acting.

### [`crescendo/`](crescendo/README.md) — A3: the signature (2026-08-28→29)

**Goal**: the roadmap's one distinctive prediction (§1.4) — the earnable range extends with
the turn of the crank — tested by opening L4 to the loop-paced learner on the same depth-6
world, against a ceiling control that is a clock yoke of the treatment carrying exactly one
extra bit (`commit_max_level=3`): lifetime-identical, bit-identical to the forbidden commit,
on both stream draws.

**Finding**: **the value clock holds past the range the previous turn certified, on both
draws.** The thermostat crank commits L4 on its own signals (c129 certified / c121 on the
displaced draw — both inside Phase 0's offline-predicted window and book size), and against
the one-bit ceiling the value clock reads **+0.388/+0.427 at era 4** (clearing every floor,
12.9× the node's own measured displacement floor) and +0.519/+0.929 at era 5 (sign-stable,
1.27× its in-node floor) — a 4–5-entry sliver of an 816-tuple level buying a third of the
deep-era gap, with π's L4 mass rising monotonically post-commit on both draws while every
L4-forbidden arm sits at exactly 0.000. The r² wall above is real and arithmetic (frozen
recall² predicted buildable L4 to ±2) and extension is the measured lever, putting Track F3's
index ops on the critical path of any further climb. One rung, single seed, exogenous demand
ladder — §1.3's "semi-autonomous" stands.

## Reproduce

```bash
cd experiments/            # MODAL_PROFILE=chromatic
modal run rhm/practice/crystallize/crystallize.py::selfcheck_remote
python3 rhm/practice/crystallize/launch_detached.py --fn crystallize --tag cg_s0 \
    --arms "never,sched_early,sched_late,delta_gate,gate_single" \
    --n-cycles 60 --n-grad 4 --value-lr-online 3e-5 --n-rt 384 --n-score 512 --n-cand 32 \
    --sil-c 0.06 --sil-cv 0.10 --sil-win 5 --sil-hold 2 --sched-early 1 --sched-late 30 \
    --probe-every 4 --shadow-compile
python3 rhm/practice/crystallize/analyze_crystallize.py --tag cg_s0 --fetch --figures

python3 rhm/practice/ratchet/launch_detached.py --fn ratchet --tag rr_s0 --seed 0 \
    --arms "never_base,given,practice_gated,practice_early,practice_late" \
    --eras "1:6,2:3,3:1" --era-cycles 30 --budget 4 --g-budget 58 --mine-cap 8 \
    --sil-c 0.06 --sil-cv 0.15 --lp-min-drop 0.10 --late-offset 3 --probe-every 4
python3 rhm/practice/ratchet/analyze_ratchet.py --tag rr_s0 --fetch --figures
```

Full commands, calibrations and volume layout: [`crystallize/README.md`](crystallize/README.md),
[`ratchet/README.md`](ratchet/README.md), [`ear/README.md`](ear/README.md),
[`recital/README.md`](recital/README.md) (+ its `FILES.md` for the fidelity replay and the
seed-triple convention), [`tall/README.md`](tall/README.md),
[`transpose/FILES.md`](transpose/FILES.md), [`setlist/FILES.md`](setlist/FILES.md)
(joint findings: [`typed_gaps/README.md`](typed_gaps/README.md)).
