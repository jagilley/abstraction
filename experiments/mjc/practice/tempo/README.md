# tempo — levels as execution span on the plant: the delay ladder, the tempo ladder, and the factored program

**Up**: [`../README.md`](../README.md) (practice) · [`../../README.md`](../../README.md) (mjc)
**Children** (each carries a `SPEC.md` — the orchestrator's prompt verbatim — and a `FILES.md` with
every decision, gate, smoke, run record and retraction; no child README by policy):
[`prestissimo/`](prestissimo/FILES.md) (the four-rung ladder on a fast piece under delay) ·
[`accelerando/`](accelerando/FILES.md) (a fast body, tempo as the era knob, the force-level program,
the crank) · [`rubato/`](rubato/FILES.md) (the kinematic program through a body model: ceiling,
learned cerebellum, online correction). **Files**: [`FILES.md`](FILES.md).
**Conversation**: `conversation_2026-09-05.md`[^private] — Jasper's prompts
verbatim; the design turns were his.
**Roadmap**: `ROADMAP.md`[^private] §4.5 Track M, §2.2 (the two forward models),
§7.1.3 (the FM's surviving seats). Progress entries dated 2026-09-05 → 06 in
`ROADMAP_PROGRESS.md`[^private]; open items in `QUEUE.md`[^private].
**Runs**: 2026-09-05 → 06, three nodes, 23 tags, all seed 0, all on 16 CPUs (one L4 for the two
learned heads), no run over an hour. Every treatment gate-checked bit-for-bit against its donor
(`acappella/b1` → `prestissimo/a1g` → `accelerando/t1` → `rubato/k1`, each reproduced at
0.000e+00 through the next node's code). **Ranks, signs, wedges and
bit-identity twins are the claims.**
**Attribution**: the mapping this whole node tests — *an RHM level is an automatic execution
span*, and being able to execute a long motor chunk without checking in is the level jump — is
Jasper's, as is the note that error at fast tempi is partly a byproduct of chunking and a listener's
clock does not speed up with the notes; the reading that a chunk is a *program* rather than a
recording came out of the exchange; the factoring hypothesis — the main model's chunk is kinematic
and a cerebellum-style body model handles the timing — is Jasper's, and it is what `rubato/` tests.
The substrate design (a fast body, a fixed human delay, tempo as the era ladder, the listener's-clock
grade) and the smoke-first process rule came out of the exchange. One orchestrating conversation;
three implementer agents, one per child, whose own pre-registered checks produced every retraction
below.

## One-liner

On a body fast enough for the question (τ = 30 ms) at a fixed human delay (120 ms), **tempo alone
opens the motor level ladder** — a stored whole-figure unit beats a per-tempo re-fit reflex by 10×
at 48 ms notes on one feedback event against sixteen, and the segment rung has a niche of its own
for the first time in the arc — and **a chunk stored as a path transfers three octaves of tempo
through an executor that knows the body**, where a chunk stored as forces transfers to no tempo at
any step: in band at levels 2–4 at 2×, levels 3–4 at 4×, level 4 at 8×, beating the reflex 0.30× /
0.35× / 0.55× on one read. A body model **learned from the learner's own slow practice at two
tempi** keeps 7 of the perfect model's 9 in-band cells, and **correction against its forecast**
turns the 5% command-accuracy bar every open-loop executor was bounded by into a 20% one, at one
launch read per span at the source tempo. What no executor does: reach the verbatim tape's accuracy
at the deep rung, or play the figure at 48 ms notes inside the band — and what no judge did: arm.

## The question

`solo/` (2026-09-05) had re-internalized a committed motor vocabulary model-free at the segment
span and set the multi-level ladder aside as possibly ill-constructed on the plant. Jasper's
mapping supplies the construction: level ℓ is a unit executed open-loop over 2^(ℓ−1) segments
from one feedback event, nested as `T[ℓ] ⊆ T[ℓ−1]×T[ℓ−1]` exactly as on RHM. Against that mapping
the motor row was missing six things, in order of how much this node is about them: a nesting op
(chains were whole-tail tapes, not pairs of committed slots); an era ladder (RHM's damage depth
forces level k+1 at era k+1; nothing forced span); a piece whose first seam is not from rest; a
currency (the grounding economy does not run on a plant, so what a level buys had to be re-sourced
as playability under feedback starvation); a discrete form for can't-decompose above the
continuous primitive; and a certifier for timing. Three nodes closed the first four and measured
the fifth; the sixth remains the plant's.

The organizing frame that emerged, and that every result below is read against, is **three
timescales**: the body's velocity time constant τ (mass over damping), the note length T, and the
feedback delay Δ. A stored unit can execute a note blind only if τ < T; feel fails only if T < Δ. On
the donor pusher τ = 0.5 s > Δ = 0.19 s, so the band τ < T < Δ is empty at every tempo — which is why
`prestissimo/` had to inflate Δ to reach the regime, and why `accelerando/` changed the body.

## Findings

Numbered; each names the node, tag and `FILES.md` decision it rests on. Retractions are stated
where they happened, with the check that produced them.

### 1. On the slow body under delay, the ladder is real at the deep rungs and the segment rung never pays on its own (`prestissimo/`)

The four-rung ladder on an eight-segment octagon at 120 ms notes (`a0`) read as a null: every
committed arm worse than the reflex at every delay, everything above the do-nothing floor past 192
ms. Two apparatus facts accounted for most of it — the incumbent's gain grid was clipped at its floor
(decision 13: extending it below 2.0 halved the reflex's error at 384 ms and improved the committed
arms 6.5×, because the shared lead-in used the same gains), and the lead-in was played by the
*delayed* reflex, so every unit launched from a state the reflex had already lost (decisions
17–20: the fixed lead-in at the Δ = 0 read **and** gains hands over an identical state at every Δ).
With both fixed, on the 240 ms-segment piece (`dH10g`, `app=fixed`):

| Δ (ms) | reflex, 80 fb | L1 | L2 | L3 | L4, 1 fb |
|---|---|---|---|---|---|
| 0 | in | in | in | in | in |
| 48 | in | **out** | in | in | in |
| 96 | in | out | **out** | in | in |
| 144 | in | out | out | **out** | in |
| 288 | **out** | out | out | in | in |
| 384 | out | out | out | out | in |

Rungs leave the band in strict order of feedback consumption; the whole-figure unit is in band at
every delay and beats the re-fit reflex 2.2× / 1.8× / 4.2× / 4.5× at 144–384 ms; delay flatness over
the ladder is 22× for the reflex against 1.7× for the deep rungs (`acappella` finding 4's law on a
piece with a lead-in, so finding 5's from-rest artifact is gone). **The reflex outlasts rungs 1–3 in
every cell** (era₁ null everywhere): demand jumps from feel to the half- or whole-figure unit, and
the rungs that pay are those whose span exceeds τ. On the 120 ms piece the deep unit is 1.04×-flat
across the whole delay ladder and never inside the band — a 120 ms tape on a 0.5 s body carries its
hand-over error instead of damping it (flag 3; seam information 1.00–1.03× at every level on the
donor plant, 1.25× at damping 10). A tested mechanism that was *not* it: the reflex is not a tape —
its open-loop replay is 13.6× worse at Δ = 0 and the pre-fixed ≤ 1.10 reading fired nowhere.

### 2. On a fast body at a fixed human delay, tempo is the era knob, and the segment rung has its own niche (`accelerando/t1`)

Mass 1.0 → 0.06 (τ = 30.00 ms measured, terminal speed unchanged), Δ fixed at 5 steps = 120 ms,
the gain grid and practice noise scaled to the body, tempi H ∈ {32, 16, 8, 4, 2} = 768 → 48 ms
notes, two grades (per-waypoint, and a fixed 384 ms listener's-clock window):

| note length | 768 ms | 384 | 192 | 96 | 48 |
|---|---|---|---|---|---|
| reflex / best recording | 0.21× | 1.18× | 2.32× | 2.96× | **10.02×** |
| feedback events, reflex vs L4 | 256 vs 1 | 128 vs 1 | 64 vs 1 | 32 vs 1 | 16 vs 1 |
| L4 pass at ¼ leg | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

The crossing is at 384 ms and the advantage is monotone in tempo. **era₁ exists**: at 48 ms the
incumbent leaves the band (0.0690 > 0.0611) while every recorded level is inside (L1 0.0399 … L4
0.0074) — null in every `prestissimo` cell. The delay instrument, run at every tempo, says tempo is
doing the work: at 192 ms notes the incumbent never leaves the band over the whole delay ladder, and
at 96 ms it leaves at 288 ms with every level, L1 included, still inside. **A recording does not
survive a tempo change**: the slowest tape resampled to each faster tempo is 1.4–36× worse than a
tape cut there and in band nowhere; the s² inertial correction is 2–2.6× worse still (the body is
drag-dominated at slow tempi and inertial at fast ones, so neither pure scaling is right). The
listener's-clock grade smooths every arm and narrows the gap slightly (9.2× vs 10.0× at 48 ms)
rather than widening it — the chunk advantage is robust under both grades, not manufactured by the
lenient one. Seam information is 1.98× at 96 ms and 1.00 at the other four tempi, with the six
harvested renditions per cell near-identical — a practice-pool diversity fact (σ ported down 4×),
not a seam property.

### 3. A force-level program is an excellent executor exactly where it practiced and does not transfer one metronome step (`accelerando/p1`, `fsmoke_a`, `fsmoke_a2`, `fsmoke_b`)

A per-step emitter `u(φ, z, slot, tempo)` trained by self-imitation on the body's own executed
traversals (`p1`) read 60–120× worse than the tape it cloned even at practiced tempi. **Two
retractions**, both produced by the node's own instruments: capture at Δ = 0 under harvest gains vs
deployment at Δ = 5 (worth 1.9×, `fsmoke_a`), and — the one that mattered — the head was trained on
the *true* seam state and deployed on the Δ-stale *observed* one, 2.8 legs apart at 48 ms notes
(worth 72×, decision 35, `fsmoke_a2`). With the head trained on the posture it can actually see, the
program practiced at every tempo is in band at every level at 48 ms and **below the tape it was
cloned from at the deep rung** (L4 0.0056 vs tape 0.0069 vs reflex 0.0690, 12.3×). What stands from
`p1`: **no transfer**. Asked one octave past its practiced set, 0/42 · 0/24 · 0/12 · 0/6 slots pass
parity and held-out command error jumps five orders of magnitude; on the metronome ladder (steps of
1.14× → 1.25×, the head refit at every step on its own traversals, `fsmoke_b`) the largest open
fraction anywhere is 5/48 and L3–L4 open nothing — a cliff at the practiced/unpractised boundary, as
deep at 1.14× as at 1.25×, while the time-resampled tape carries two fine rungs with zero training.
Held-out error *worsens* with more practiced tempi (0.04 after one, 0.42 after four): the tempo
input acts as an index and the head shares nothing across tempi. Two smaller facts: **P-E**, the
exchange rate — perturbing the library's own best tape by 5% of command spends the whole band at
the deep rung open-loop while the segment rung barely moves (0.0087 → 0.0614 at ε = 0.05, L4, 384 ms)
— so the level axis is the open-loop command-sensitivity axis; and at a practiced tempo the head is
best at the deepest rung and worst at the shallowest (L4 0.0063 vs L1 0.0858 at 192 ms), the
opposite of P-E's ordering, because a level-4 arm makes one decision from one stale read and a
level-1 arm eight.

### 4. The crank on the plant: the machinery works and the judge does not arm (`accelerando/c1`)

`conductor`'s thermostat over per-tempo recording tables, five arms, 80 cycles. Tables formed at
every tempo; the level-3 unit at 48 ms beat the re-fit reflex 3.2× at 3 feedback events against 16,
inside the band; can't-decompose at L2 is positive at both tempi asked (+0.0059, +0.0147 with 3–4
fewer reads). But the within-level gauge's null-ABBA dead zone is **exactly 0** (sd(N) = sd(D) = 0
over 48 windows): on a deterministic plant with a static per-tempo library and no executor
plasticity, the arm's own error cannot move except when the loop acts, so `conductor`'s refuser
cannot be reproduced here — only its impossibility shown. And `at_support` rises monotonically for
80 cycles and never quiets inside its dead zone (2 → 39 → 63 at 768 ms), so all 12 of the yield
arm's actions were cap-forced: a cap schedule with a different period, not a driven loop. With a
table per tempo and the program never executed, a tempo advance is a change of piece rather than a
level crossing; this run measures within-tempo pacing, not a tempo ratchet (stated at the top of
every report the reducer writes).

### 5. A path stored slow, played through an executor that knows the body, transfers three octaves (`rubato/k1`)

The factoring: the unit is the **realized path** of its tape at its harvest tempo (position and
velocity on phase φ ∈ [0,1], re-measured — decision 3), nested as before; at a target tempo the
clock is outside the program (v = x′(φ)/T′, a = x″(φ)/T′²); the exact inverse dynamics
`u = (m·a + c·v)/gear` is an experimenter-side oracle, the way RHM uses exact DP. The first ceiling
smoke (`ksmoke1`) read a null — the carried path 5–18× worse than the same path at its own tempo and
no better than the naive resample — and **that column is retracted**: phase resampling 32 → 2
samples per note is a 16× decimation, and the resampler did it by point sampling with no
anti-aliasing (decision 22, caught by the check written to exclude it). The anti-alias filter's width
is `H_src/H_dst`, so the correct resampler *is* a tempo-dependent low-pass — the invariant is the
path up to the resolution the tempo permits, arrived at from the apparatus side. On the corrected
arm (`kzsa`, `aud`, five tempi, `app=fixed`, all 9 gates, P-T1 125 checks at 0.000e+00):

| step | note | L1 | L2 | L3 | L4 | best kinematic vs reflex | fb |
|---|---|---|---|---|---|---|---|
| 1× | 768 ms | 0.0408\* | 0.0399\* | 0.0326\* | 0.0213\* | — | — |
| 2× | 384 | 0.0836 | **0.0426\*** | **0.0426\*** | **0.0446\*** | **0.30×** | 1 vs 128 |
| 4× | 192 | 0.0868 | 0.0838 | **0.0529\*** | **0.0550\*** | **0.35×** | 1 vs 64 |
| 8× | 96 | 0.2298 | 0.1230 | 0.0886 | **0.0555\*** | **0.55×** | 1 vs 32 |
| 16× | 48 | 0.3498 | 0.2033 | 0.1210 | 0.0660 | 1.05× | 1 vs 16 |

(\* = inside the 0.0611 band.) The in-band set is a **wedge that closes from the shallow end**: L1
never transfers, L4 survives to 8× and misses 16× by 8%. Saturation is 0.000 everywhere; both grades
agree everywhere; the launch state is not the cause (own-launch vs actual within 8% at every tempo);
the shape distance between the carried path and the fast tempo's own is flat at 0.41–0.50 of the
band across all four unpractised tempi while the executed error varies 7×. Two things the run added:
the **frozen key on the source tempo's own postures beats the plant audition** at three of four
unpractised tempi (auditioning at a new tempo from a stale read is fragile, as `acappella` found;
the slow-tempo key is a stable index); and **no executor, the exact one at its own tempo included,
opens a level-4 parity slot at any tempo** — forces reconstructed from a measured path lose ~4×
against the verbatim tape at the deep rung (L4 0.0213 vs 0.0065 at 768 ms), in band but not at
parity. P-E's exchange rate a third time, bounding any executor.

### 6. A body model learned from the learner's own slow practice keeps most of the ceiling, once it has seen the body move at two speeds (`rubato/klsmoke1–3`)

The linear inverse `u = c_a·a + c_v·v` fitted as a general 2×4 map with bias (isotropy not given),
and a generic MLP, each fitted on executed (v, a, u) triples from practice traversals at the slow
tempi only:

| fitted on | drag `c_v` recovered (true 0.200) | in-band cells kept, of the oracle's 9 |
|---|---|---|
| 768 ms only | 0.104, 0.087 — 52% / 43% | linear **0**, MLP 6 |
| 768 + 384 ms | 0.169, 0.160 — 84% / 80% | linear **7**, MLP 4 |
| 192 ms only | 0.199, 0.196 — 99.6% / 98% | linear 6, MLP 6 |

**The linear family's transfer tracks its recovered drag coefficient and nothing else.** At the
slowest tempo the drag term (≈ 0.032) sits below the practice noise (σ = 0.0628) and one of its two
numbers is invisible; add one more slow tempo, doubling the velocity range practiced, and it keeps
more of the ceiling than the MLP reaches from any fit set. `klsmoke1`'s "inductive bias" reading is
withdrawn in the record (decision 29): it was a statement about where the fit was taken. **Velocity
coverage is the axis** — the fraction of the 48 ms query outside the fit's range falls 0.40 → 0.33 →
0.27 across the three fit sets and the held-out error there moves with it, while acceleration reach
never exceeds 1.16× (practice noise on a 167 m/s² body already injects ~10 m/s² at every tempo, so
slow practice covers acceleration and not speed — the design's premise, contradicted by the coverage
instrument built in case it was). The fitted model's extra terms are **the piece's rotation region
absorbed into the executor**: refitted on the same renditions replayed with the rotation off, it
recovers the body's constants to ten figures with zero off-diagonal and zero bias (decision 34,
1.84e-02 / 3.58e-03 against 9.9e-11 / 3.7e-11).

### 7. Correction against the learned forecast turns the accuracy bar into a read budget (`rubato/kc1`, `kc2`)

The corrected executor: feedforward from the {768, 384} ms linear inverse, a forward model derived
**algebraically from the same fitted inverse** (one body model in two directions, no new parameters
— decision 31) dead-reckoning between reads from the Δ-stale read through the executor's own issued
commands, and a PD term on the forecast against the path. The first gain rule (continuous-time
critical damping tied to the note) clamped `kd` to zero wherever T_note > 4πτ and hit grid edges at
four of eight cells; the re-posed rule (decision 36) places both poles of the *exactly linear*
discrete error system at a chosen decay ρ in closed form, so the search is one-dimensional and
body-only (1 edge of 10). With the read ladder in **reads per span** (decision 37):

| note | L1 | L2 | L3 | L4 | ε at which the band is spent, L4 |
|---|---|---|---|---|---|
| 768 ms | **1** | **1** | **1** | **1** | 0.085 (launch only) → 0.11 (16 reads); open-loop 0.12 |
| 384 | 1 | 1 | 1 | 2 | already out at 1 read; **> 0.20** at 4 and 16 |
| 192 | 2 | 4 | 4 | 4 | already out at 1; **> 0.20** at 4 and 16 |
| 48 | x | x | x | x | out at every read count and ε |

(Fewest reads per span in band, exact forward model; the fitted one tracks it at all but one cell.)
**One launch read suffices at every level at the source tempo** — decision 18's once-at-launch end
of the fork — and at three of four levels one octave up; the requirement rises to 2–4 reads at two
octaves; at four octaves no read count on the ladder puts any level in band. Where the arm is in
band, **correction absorbs a 20% command error** where open-loop spent the band at 5%. At the source
tempo the corrected L4 unit at **one** feedback event (0.0239) matches the reflex handed the same
forward model through the delay at 256 (0.0252); at equal feedback the corrected unit beats that
reflex at every tempo and loses to the model-free reflex at every tempo — its value is not accuracy
at full feedback but staying in band on far fewer reads while tolerating an imprecise feedforward.
One prediction written down and falsified: a per-step dead-reckoning forecast with an exact model
should make the best ρ independent of the read interval; it is not, at three of four tempi (768 ms:
0.8 vs 0.6; 192 ms: 0.0 vs 0.8), and this tag does not locate why.

## Interpretation (discussed with Jasper across the conversation — argued, not measured)

- **The mapping holds on the plant.** Deeper is more automatic, automatic is what pays where
  feedback is too slow, and rungs fail in the order of their feedback consumption under both knobs
  that starve it. Delay and tempo are two knobs on one ratio, Δ/T; delay holds the library and the
  body's execution fixed while turning it (which is what a ratchet over tapes needs), and tempo is
  the phenomenon's own knob (a pianist's Δ never changes). On a body with τ > Δ no tempo opens the
  regime, which is why the slow-body results had to inflate Δ, and why the body was changed. The
  scaffolding result — rungs 1–2 paying only in rungs 3–4's currency — is a property of τ against the
  span, not of levels: on the fast body the segment rung has its own niche.
- **What a level buys is playability where the lower level cannot play, plus reads.** Never lower
  error at full feedback: the model-free reflex wins every equal-feedback comparison in every node,
  as `never` won raw error in `fingering` and `legato`. The currency is the meter's own.
- **A chunk is a program at the address and a recording in its content.** The force-level head is
  the best executor in the arc exactly where it practiced and transfers nowhere, because forces carry
  the body's timing inside them. The tempo-invariant object is the path up to the resolution the
  tempo permits; converting it to forces is inverse dynamics; doing that online against a forecast is
  what makes it robust and cheap in reads. That is the factoring Jasper proposed, measured with its
  ceiling (three octaves), its learned version (7 of 9 cells from two slow tempi) and its bound (no
  executor reaches the verbatim tape at the deep rung). The bound is the arc's commit-verbatim law
  seen from the other side: a function approximator is an average, and deep content wants exact
  content. The program's job is addressing and tempo, not regenerating the bytes.
- **The cerebellum's two jobs, seated on the learner's side of the meter** (ROADMAP §2.2, §7.1.3):
  a body model learnable from one's own slow practice, provided the practice spans enough speed for
  the drag term to be identifiable; and correction against a forecast, which trades reads for
  robustness on a measured curve. Neither is available to the incumbent as a planner — that is the
  `accompanist` confound, and `reflex_ec` is carried as a reported row so the incumbent stays honest.
- **The listener's clock did not bite here.** Both grades agree in every cell; the generous band was
  not the binding constraint on any piece. The mechanism it was meant to detect — corner-cut but in
  place — never appeared. The place it may still matter is the fastest tempo, where nothing is in
  band for anyone; whether that is the band's absolute width at a barely-playable tempo or a limit of
  the corrected unit is an open item.
- **Two things this node does not have.** Seams: on fast pieces the practice pool is near one path
  per cell at most tempi, so routing and trust had nothing to select on, and the crank ran over a
  degenerate address book. And a judge: the within-level gauge is constant by construction on a
  deterministic plant with no executor plasticity, and the next-level count never plateaus, so the
  motor crank needs either an adapting executor or a rate-shaped gauge before Track A's rules can be
  tested here.

## Apparatus lessons (every one caught by a pre-registered check, none by the treatment)

- A gain grid must be wide enough to hold the incumbent's per-condition optimum; a clipped floor
  reads as a delay result. On a fast piece the reflex wants `kp` well below 2.0.
- The lead-in must be played at the Δ = 0 read **and** the Δ = 0 gains, identical at every rung of
  a sweep; a delayed or re-fit lead-in re-introduces the incumbent's collapse as every unit's
  hand-over.
- A read at Δ ≥ k₀·H is the reset state (aliasing); the lead-in must be longer than the largest Δ.
- A head must be trained on the posture it will be handed at deployment — the observed, stale one —
  not the true state; the oracle leak was worth 72×.
- A 16× phase decimation needs anti-aliasing, and the correct filter width is the tempo ratio.
- A body model must be fitted where the body moves: at 768 ms notes the drag term is below the
  practice noise and the fit recovers half of it.
- A continuous-time gain rule on a body with dt/τ = 0.8 is the wrong object; the discrete error
  system is exactly linear and its pole placement is closed-form.
- A shape distance needs a denominator that does not collapse with the library's diversity; metres
  against the band, not a ratio to the within-library spread.

## Caveats

- The claims are ranks, signs, wedges across tempi and levels, and
  bit-identity twins, with every treatment gate-checked against its donor.
- The band is ½ of the mean leg at every tempo; pass fractions at ⅓ and ¼ are reported everywhere
  and do not change any ordering, but the fastest tempo is where the absolute width bites and it is
  unresolved.
- `rubato` lifts the no-forward-model rule for the executor only; the incumbent of record stays
  model-free. `reflex_ec0` (the undelayed reflex) beats every corrected row at every tempo, as it
  must.
- The Phase 2 and Phase 3 runs of record in `rubato/` were not run; their smokes and controls are
  the record, and the queue holds both with the reason they may already be the finding.
- Two things are on the record as unlocated: the ρ*-vs-read-interval dependence (finding 7), and
  the fitted-beats-exact cells at the deep rung, of which the rotation absorption is confirmed
  present but not confirmed sufficient.

## Runs on disk

| node | tag | what |
|---|---|---|
| `prestissimo` | `a0`, `a0s` | Phase A on the 120 ms octagon at two tempi; the first (retracted-in-part) null |
| | `a1`, `d5`, `d10`, `dH10` | the diagnostic round: extended gain grid, damping and segment-length cells |
| | `a1g`, `dH10g` | the fixed lead-in (`app=fixed`); `dH10g` is the ladder of record on the slow body |
| `accelerando` | `t1` | Phase 1 of record: the fast body, the tempo ladder, recordings and resamples |
| | `p1`, `fsmoke_a`, `fsmoke_a2`, `fsmoke_b` | the force-level program: the run, two retractions, the metronome ladder |
| | `c1` | Phase 3: the crank over per-tempo recordings |
| `rubato` | `ksmoke1`, `ksmoke2`, `kdsmoke1` | the ceiling smoke, parity, and the diagnostic that found the resampler |
| | `k1` | **Phase 1 of record**: the kinematic unit through the exact executor, five tempi |
| | `klsmoke1`, `klsmoke2`, `klsmoke3` | the learned cerebellum: fit on {32}, {32,16}, {8} |
| | `kc1`, `kc2` | correction: the first gain rule, then the discrete rule on the reads-per-span ladder |

Volume `mujoco-control-data`: `/data/practice_prestissimo/<tag>/`, `/data/practice_accelerando/<tag>/`,
`/data/practice_rubato/<tag>/`; fetched copies, reports and figures under each node's `results/<tag>/`
and `figures/<tag>/`.

## Reproduce

```bash
cd experiments/                                  # MODAL_PROFILE=chromatic
# prestissimo — the ladder of record on the slow body
modal run --detach mjc/practice/tempo/prestissimo/ladder.py::prestissimo_ladder --spawn --tag dH10g --h-seg 10
python3 mjc/practice/tempo/prestissimo/analyze_ladder.py --tag dH10g --fetch --figures
# accelerando — Phase 1 of record
modal run --detach mjc/practice/tempo/accelerando/tempo.py::accelerando_tempo --spawn --tag t1
python3 mjc/practice/tempo/accelerando/analyze_tempo.py --tag t1 --fetch --figures
# rubato — Phase 1 of record, then the learned cerebellum and correction smokes
modal run --detach mjc/practice/tempo/rubato/kin.py::rubato_kin --spawn --tag k1 --arm-set record --diagnostics
python3 mjc/practice/tempo/rubato/analyze_kin.py --tag k1 --fetch --figures
```

Each child's `FILES.md` carries the exact flags for every tag, including the smokes.

## Next steps (queued in `QUEUE.md`[^private], not started)

- `rubato/` Phase 2 and Phase 3 runs of record (five tempi; fit on {768, 384} ms; the
  reads-per-span ladder under the discrete rule) — held: the smokes and controls may already be the
  finding.
- The two unlocated items: ρ* against the read interval, and the fastest tempo under a band re-posed
  on the listener's clock.
- Seam information on a fast piece (practice-pool diversity, or a rotation region sized to the fast
  body) — the prerequisite for routing and trust to have anything to select on.
- The judge on the plant: an adapting executor (`solo`'s queued variant) or a rate-shaped
  next-level gauge, before Track A's outer loop can be tested here.
- `/update-beliefs` for the unit: levels as execution span; a chunk is a program at the address and a
  recording in its content; the cerebellum's two jobs on the learner's side of the meter; the
  three-timescale condition τ < T < Δ.

[^private]: Not mirrored: this link points to a document in the private lab repo (the roadmap, the queue, an unrun spec, reading notes, or a conversation). See the top-level README for what is held back and why.
