# orbitofrontal — the norm is the world model read through a projection whose weights the goal's history sets: a running estimate, public in its level and private in its response, what it reads, and what consumes it

**Up**: [../README.md](../README.md) (logit_reading) · **Children**: [`adaptation/`](adaptation/README.md)
(the norm as a running estimate, within-subject) · [`projection/`](projection/README.md) (belief or
representation) · [`shaped/`](shaped/README.md) (a trunk fine-tuned on the goal) · [`regime/`](regime/README.md)
(a world where a violation predicts future cost) · [`abstain/`](abstain/README.md) (the consumer) · **Sibling in
the practice arc**: [`rhm/practice/voicing/tessitura/`](../../practice/voicing/tessitura/README.md) (the same
questions asked of the live learner's own judge; it lives with its machinery and is written up here) ·
**Re-reads of banked nodes**: [`striatum/results/NOTES_2026-09-17_rereads.md`](../striatum/results/NOTES_2026-09-17_rereads.md)
(the MLP and width-stratified across-level contrast, a second and third trajectory seed on the norm round, a
variance head) · **Files**: [FILES.md](FILES.md) · **Donors**: [`../striatum/`](../striatum/README.md),
[`../striatum/norm/`](../striatum/norm/README.md), [`../striatum/junction/`](../striatum/junction/README.md),
[`../coeruleus/readout.py`](../coeruleus/readout.py), [`../basalis/`](../basalis/README.md),
[`../phasic.py`](../phasic.py), all unchanged.
**Date**: 2026-09-17 · **Status**: six rounds run and reduced in one session, five Opus implementers; three
trajectory seeds behind the norm re-reads, two order seeds behind `adaptation/`, two learner seeds behind
`tessitura/`, six cells across three checkpoints behind `projection/`; ≈ 12 L4 GPU-hours in all.
**Prompt**: the discussion after #121[^private] and its
`CONVERSATION.md`[^private]. The value-side rounds had put the emotion story's "learned norm" in the
critic's own expectation over outcomes ([`norm/`](../striatum/norm/README.md)), and Jasper asked two things: how
the readings could be strengthened in a way that increases their generality, and, once the norm had been named,
whether it could be anything other than world-model-based — "in order to be angry that an outcome was worse than
you expected, you have to have a world model of expectation in the first place." Six priorities were agreed and
run: the same questions on the practice learner's own judge; the within-subject form of Xiang's design; a goal
whose downstream cost a violation could predict, with the trunk shaped by it; a consumer for the value reading;
cheap re-reads and seeds; and, held for discussion, a language-model port. A seventh, the quick test of the
reframing, was added mid-session.
**Attribution**: the six priorities, the reframing of the norm as the world model read through a history-set
projection, the design of the regime world and the abstain choice, and the belief-versus-representation test are
the orchestrator's, agreed in discussion; the clarification that "value-relevant" means a goal exists and events
cost something, the push that an expectation cannot be other than world-model-based, and the call for a quick
confirming test are Jasper's; the block-Gram fitters and the mechanical null, the position stratification and
the pure-noise gates, the readout-span measurement and the exact split of the revision, the regime filter as one
reader for both worlds, the `init`-cell exclusion and the sign tables, and the identical-rows-versus-trajectory
rule are the implementers' (one Opus session per node). The tiering of the norm round by what replicates is the
re-reads implementer's and was adopted as the rule for reading the rest.

## One-liner

> An outcome-trained reader on a learned representation has an expectation over how well things will go. It is
> the world model's state read through weights the goal's outcome history sets: input from the world model,
> weights from the history. That expectation is a running average of recent outcomes with no memory beyond its
> window, on a frozen trunk and on a live learner alike. It is what a consumer uses, and it is worth the most away
> from the events the arc had been measuring it at. It reads whatever in the state predicts cost, so it ignores a
> rule violation where the violation says nothing about the future and reads the hidden state a violation reveals
> where it does, priced by how much the reader's belief moved. Its level is public, recoverable from the model's
> own output distribution; its revision at an event is private, living in directions the output layer does not
> read, into which the trunk moved them while learning the grammar. Across three independently trained models,
> what is read as a difference between conditions on identical rows reproduces to a few decimals, and what is read
> as a rate or a direction along one trajectory mostly does not.

## Goal

Six questions, each on its own child, all on the value reader the arc built in [`striatum/`](../striatum/README.md):
a closed-form ridge on a frozen next-token trunk's residual stream, trained on a fixed actor's outcomes at a
structural query, reading three objects — its pre-event level (the **norm**), its revision at an event (the
**response**), and the outcome minus the norm (the **outcome surprise**).

1. **Is the norm a running estimate, and on what clock?** [`norm/`](../striatum/norm/README.md) was
   between-subjects; Xiang, Lohrenz and Montague's design is within-subject. One critic across a switch of
   worlds, and the live learner's judge across its own era ladder ([`adaptation/`](adaptation/README.md),
   [`tessitura/`](../../practice/voicing/tessitura/README.md)).
2. **What does the reader read, once the trunk cares about the goal?** The arm striatum deferred: the trunk
   fine-tuned on the actor's task ([`shaped/`](shaped/README.md)).
3. **What does it read when a violation has something to predict?** A world with a persistent hidden regime,
   so an illegal token is evidence about future cost ([`regime/`](regime/README.md)).
4. **What consumes it?** An abstain-or-commit choice at a price, gated by each object in turn, on and off the
   events ([`abstain/`](abstain/README.md)).
5. **Belief or representation?** Whether the projection reads the model's published output distribution or
   directions of the state the output does not expose ([`projection/`](projection/README.md)).
6. **What holds under seeds?** The norm round on three trajectory seeds, its across-level contrast on the MLP
   critic and inside width strata, a variance head ([re-reads](../striatum/results/NOTES_2026-09-17_rereads.md)).

Same substrate as the arc throughout (`v16 s2 L6 m4`, `8L/8H/256D`, the `a1` and `swap65k` stimuli); the
practice-arc round uses `voicing`'s substrate and judge unchanged.

## Instruments, in one paragraph each

- **`adaptation/`.** A diet stops being one accumulation over its mask and becomes a row order over it: rows from
  one tercile world, then rows from another, with the critic refit along the sequence by five fitters (sliding
  windows of 200 and 800 rows, exponentially weighted least squares at two memories, and a cumulative fit that
  cannot forget), read at 29–37 checkpoints on norm's fixed held-out rows and twins. Phases `out_lo→mid`,
  `out_hi→mid`, `out_mid→mid` (the control whose world does not change), the same for the `dmg` family, and the
  same rows interleaved at random. Two order seeds.
- **`tessitura/`.** The voicing judge's own level was never logged, so a yoked re-run of the banked arm with three
  default-off knobs: the judge's mean predicted solve rate per slot per cycle, a fixed per-(slot, era) panel of
  rows scored by the live critic every cycle, and the structural label (does the written class repair at that
  slot) per row. Bit-identical to the banked arm with the knobs off, and byte-identical in its row dump and heads.
  Plus a CPU refit of the critic under outcome diets on the fixed final trunk, norm §1's design exactly. Two
  learner seeds.
- **`shaped/`.** From the 64k checkpoint, 3000 steps of continued training with the actor's own objective jointly
  with the trunk, in three arms on bit-identical data: task only, task plus next-token, and next-token only as
  the control that moves the trunk without moving it toward value. Then every striatum and junction readout on
  each arm, and Part 1's altitude and Part 2a's detection so the cost of shaping is on the record.
- **`regime/`.** A two-state Markov corruption regime along the stream (mean dwell 288 clean, 32 noisy; rates
  0.002 and 0.12), one next-token trajectory trained on it and one on an i.i.d. control at the same marginal rate;
  the exact running regime filter, the same reader applied to both worlds, as the reference; striatum's actor and
  critic on each; natural twins (the corruption undone) and surprisal-matched legal twins.
- **`abstain/`.** At a query the actor answers, returning the outcome, or abstains, returning a fixed price.
  Each gate's policy is coeruleus's quantile step function fitted on half the rows and read on the other half,
  against a per-stratum constant floor and the realised outcome as the ceiling, under four conditionings
  (marginal, inside position strata, inside position and surprisal, inside position and value). Two venues: the
  banked anchors at five offsets after an event, and the whole stream at every position of held-out edited,
  original and quiet windows.
- **`projection/`.** The critic refit on the model's output distribution at the query (sixteen numbers) in place
  of the state, with dimension-matched controls (a random sixteen-dimensional slice of the state; its top sixteen
  principal components; a short public window of past logits and tokens). And on the twins, one linear map from
  the model's forecast difference after the illegal token against its legal twin to the critic's revision
  difference. The critic's direction measured against the output layer's row space, at steps 0, 8k and 64k.

## 1. The norm is a running estimate, and its clock is its memory's

**Across a switch of worlds the expectation re-calibrates at exactly the rate its memory turns over.** On
identical fixed rows with an identical outcome, the outcome surprise of the critic that came from the low world
and the one that came from the high world differ at the switch and converge; `swap65k`, the 200-row window,
ℓ = 3, gap in `δ` by windows after the switch:

| windows after switch | 0 | 25 | 50 | 100 | 150 | 200 |
|---|---|---|---|---|---|---|
| from-low minus from-high | 0.215 | 0.186 | 0.161 | 0.113 | 0.054 | 0.004 |

The control's noise band is sd 0.010 on `V_pre`. Against the mechanical null — the fraction of its rows the
fitter has already replaced — the measured closure deviates by a few hundredths whose sign agrees across two
order seeds in 29 of 52 cells on `a1` and 26 of 52 on `swap65k`, which is chance. The cumulative fit has closed
almost exactly half the gap by the end of the second phase in both seeds. The same rows interleaved at random are
flat for the whole run. The shift and the rescaling of the norm close in lockstep: the slope and the intercept of
each online critic's norm on the banked `full` critic's cross the same fraction of the way at every checkpoint
within 0.05.

**The anticipatory response carries the history too, and re-calibrates on the norm's clock.** Norm §2's
between-subjects reading was that the response at an identical event does not order with the world; its own
within-family scaled response was already monotone, and the sign reversed between families. Within one critic the
pre-switch gap in `R / sd` against the control has the same sign and the same size in every readable cell at both
order seeds (29 of 29 on `a1` at r = +0.94; 26 of 27 on `swap65k` at r = +0.89) and falls to half within the
phase in every cell, median 250–300 windows. The damage reading of the response stays flat and arm-independent at
0.54–0.56 throughout, as norm reported.

**The outcome surprise is a world-change alarm in outcome currency.** Read on the rows arriving next, strictly
out of sample, `δ` sits at zero before the switch, jumps to about five standard errors at it in the direction of
the change, and returns inside the band by 175–300 windows; the sign agrees across order seeds in every one of
the 38 and 31 cells that clear two standard errors in both, at r = +0.92 and +0.99. It is nil in the interleaved
arms and in the world-does-not-change control (a handful of control cells reach 2–3 standard errors on `a1`, so
the separation rests on amplitude rather than on the controls never crossing). The cumulative fit never returns:
at ℓ = 3 it is still outside the band at the end of the run in both venues and both seeds. This is the structural
side's excess-over-stated-uncertainty ([coeruleus Q2](../coeruleus/README.md)) restated on the value side: zero-mean
on the adapted world by construction, nonzero exactly where the world moved, and quiet again once the norm has
moved with it.

**On the live learner the clock decomposes, and the reader adds almost nothing.** The voicing judge's level,
regressed against the world's own solve rate, lags by about 50 cycles; against its own ring buffer's trailing
rate it lags by 2–5; the buffer's window, measured independently, is 20–50 cycles. Both seeds. The judge sits on
its world to within 0.04 at every era where it has begun governing (0.032 and 0.033 at the two seeds). The
direction from which it approaches — undershooting a rising world at seed 0 — did not replicate and is withdrawn
in [`tessitura/DESIGN.md`](../../practice/voicing/tessitura/DESIGN.md) §8c. On rows frozen in era 1 and scored by
the live critic every cycle after, the level drifts upward with the world by about a tenth while the ranking on
those same rows erodes slowly, at both seeds.

> **The norm is a moving window over experience whose time constant is the window's, on a frozen trunk and on a
> live learner, with no reproducible bias in either direction. The response carries the history and moves with
> it. The outcome surprise fires when the world changes and goes quiet once the norm has caught up, so a system
> watching only the surprise could not tell it had been re-calibrated; the history lives in the level, silently.**

> **Re-read 2026-09-22 by [`striatum/norm/precision/rereads/clock/`](../striatum/norm/precision/rereads/clock/README.md), on readers with more
> of the belief than the ridge** (`+log q`, `ln_f(state)`, `+H(q)`; both order seeds). Every reading above holds on every reader: the gap tracks the
> mechanical null, the half-times sit at the null's, the deviation's sign agreement across orderings is at chance, the excursion fires, returns and
> stays out of the controls identically; the response carries the history on the same clock and is about 1.8× larger and readable in nearly twice as
> many cells on the belief-appended reader, with the same cell pattern (r ≈ 0.9). Two corrections: the sentence "the shift and the rescaling of the
> norm close in lockstep … within 0.05" has no reduction behind it in `adaptation/analyze.py`, and under three explicit definitions it holds in
> 12–22 of 80 cells per seed on every reader, the ridge included (near a level of 0.9 a change in slope and a change in intercept are hard to
> separate); and the headline row's 200-window entry reads **−0.004** in the banked files, not +0.004.

## 2. What the value reader reads

### 2a. Shaping the trunk on the goal sharpens the cost reading and leaves legality alone

Refit actor on clean held-out windows (chance 0.0625), the trunk's clean next-token cross-entropy, and Part 1's
altitude:

| arm | ℓ=1 | ℓ=2 | ℓ=3 | ℓ=4 | ℓ=5 | ℓ=6 | clean CE | `κ*` | `KL(p_L‖q)` |
|---|---|---|---|---|---|---|---|---|---|
| frozen 64k | 0.904 | 0.815 | 0.705 | 0.524 | 0.325 | 0.141 | 1.206 | 4.90 | 0.017 |
| task only | 0.892 | 0.796 | 0.706 | 0.590 | 0.430 | 0.222 | **1.843** | 4.60 | **0.654** |
| task + next-token | 0.899 | 0.805 | 0.716 | **0.601** | **0.442** | **0.223** | 1.215 | 4.90 | 0.026 |
| next-token only | 0.902 | 0.812 | 0.695 | 0.512 | 0.319 | 0.138 | 1.211 | **5.40** | 0.022 |

Task-plus-next-token buys the whole shaping gain at the deep levels for +0.008 nats of predictive cost; task-only
pays +0.61 nats and, by the altitude instrument, stops being a coarse observer at all (the fitted family's share of
its residual goes from 0.81 to 0.99 while `κ*` barely moves) rather than becoming a coarser one; plain continued
next-token training climbs the altitude further. On the value side, matched as striatum matches (`swap65k` `t_v`,
every guard at 0.500):

| readout, ℓ=1…4 | frozen | task | task + ntp | ntp |
|---|---|---|---|---|
| damage `V` | .694/.691/.656/.609 | .704/.699/.677/.643 | .699/.700/.661/.651 | .732/.720/.643/.605 |
| oracle consequence probe | .581/.565/.574/.593 | .592/.588/.577/**.634** | .589/.583/.600/**.645** | .561/.542/.535/.577 |
| legality at matched consequence, `R`, ℓ=1–3 | .510/.452/.524 | .520/.490/.528 | .517/.462/.513 | .512/.445/.541 |
| junction's eleven diets, span | ≤ 0.022 | ≤ 0.016 | ≤ 0.016 | ≤ 0.016 |
| across-level win rate, `t_v` | 0.424 | 0.403 | 0.398 | 0.418 |

Shaping widens the state's consequence headroom and improves the critic's fit, and the critic converts every bit
of it into a sharper reading of how much was lost; its legality reading at matched consequence stays at chance on
every arm and across every diet, the oracle legality headroom does not rise, and the across-level contrast stays
a null. The containment addendum survives too: conditioned on the value level, the realised horizon excess reads
nothing at ℓ = 4 on every arm. So the null of [junction](../striatum/junction/README.md) is not an artefact of a
purely next-token-shaped state read through a linear cut. But the goal built here never needed legality to
predict its outcomes, which is the world the next child builds.

### 2b. Where a violation predicts future cost, the reader reads the hidden state, priced by its belief

The bursty world's trunk carries the exact regime posterior: a ridge probe on `post_block7` reads the filter's
forward belief at R² 0.851 (true regime AUC 0.851; embedding floor 0.499; the same probe on the i.i.d. world
0.486–0.523). The two trunks are indistinguishable on the clean actor task and on critic fit.

| readout, `swap`-free world of corrupted events, ℓ=2 | burst | i.i.d. control |
|---|---|---|
| corr(`R`, filter's revision `db`) at `a` = 0 / 2 / 8 | −0.602 / −0.482 / −0.404 | −0.140 / −0.078 / −0.140 |
| … partial on the model's own surprisal | −0.242 / −0.160 / −0.134 | −0.076 / −0.081 / −0.076 |
| AUC(−`V`, true regime) at corrupted events / at quiet events | 0.703 / **0.556** | — |
| AUC(−`V`, ≥ 1 corruption in the next 12), full horizon | 0.606 | 0.486 |
| natural-twin `dV` at offset 0 / 8, ℓ=2 | −0.122 / **−0.031** | −0.068 / **−0.007** |

The critic's level reads the hidden state even at quiet positions with no corruption in the preceding eight
tokens; its revision at a corrupted token tracks the exact filter's revision, and appears between step 0 and
8k; and the persistence of the twin gap is the dissociation, 4.7× at offset 8 between the worlds. The "weighted
by expectation" clause holds in the currency where it can be true. Holding the token's own evidence fixed (log
likelihood ratio 4.05–4.07) and binning by the pre-event belief that the world is noisy:

| pre-event belief `b_pre` | 0.030 | 0.381 | 0.829 |
|---|---|---|---|
| filter's allowed revision `db` | 0.588 | 0.537 | 0.136 |
| critic's `R`, ℓ=2, `a`=0 / 2 / 8 | −0.268 / −0.155 / −0.089 | −0.221 / −0.114 / −0.047 | −0.182 / −0.090 / −0.027 |

Monotone, and the shrinkage grows with the horizon (32% → 42% → 70%); the i.i.d. world has almost no range in
`b_pre` and no ordering. Around a regime switch both of norm's objects appear on a world that changes on its own
clock: clean→noisy, `V` 0.809 → 0.746 and the outcome surprise −0.024 → −0.075 by offset 8; noisy→clean, `V`
0.701 → 0.800 and the surprise −0.052 → +0.058.

**The legality contrast moves hugely, and for a reason that is not the norm.** Illegal against legal corrupted
token at matched immediate damage and matched surprisal, AUC(−`R`): burst 0.815 / 0.828 / 0.799 / 0.708 for
ℓ = 1…4 against striatum's 0.45–0.52 — but the i.i.d. control reads 0.753 / 0.782 / 0.688 / 0.569, and on those
rows the illegal corruption is the cheaper one at every horizon (natural-twin damage at ℓ = 2, illegal / legal:
0.303 / 0.393 at `a` = 0, 0.162 / 0.315 at `a` = 2). The actor's realised accuracy after an illegal token is
0.550 against 0.320 after a legal-but-wrong one; the critic is well calibrated after the illegal token (`V` 0.591
against 0.550 realised) and grossly over-optimistic after the legal one (0.739 against 0.320), identically in
the control. An off-grammar token is recognisable as noise and gets discounted; a legal-but-wrong token is a
plausible member of another constituent, the model re-parses around it, and the damage is real and invisible.
What the reader picks up from legality is visibility. One structural limit: in this world the model's own
surprisal and the regime evidence are nearly one variable (corr 0.85), so "regime or surprise" cannot be fully
separated here, though the partial correlations still separate the worlds.

### 2c. On the live learner the reader reads structure where structure is the cause of cost

With the structural label logged per row, the voicing judge reads whether the written class repairs at its slot
more strongly than whether the trajectory solved. Filed rows, eras 1→5, seed 0: `crit ~ struct` 0.598 / 0.572 /
0.631 / 0.708 / 0.678 against `crit ~ y` 0.596 / 0.577 / 0.594 / 0.642 / 0.617; holding the other label fixed,
structure at matched cost 0.585 / 0.534 / 0.608 / 0.702 / 0.663 and cost at matched structure 0.553 / 0.558 /
0.525 / 0.549 / 0.521. The difference of the two conditionals agrees in sign at both seeds in eras 3–5 (+0.083 /
+0.153 / +0.143 and +0.092 / +0.169 / +0.181). This is the opposite ordering from striatum §2, and the reason is
the design rather than the reader: striatum built its consequence label independent of level and surprisal by
construction, so structure and cost were dissociable; here a repairing class is close to the proximal cause of a
solve (`P(y | struct)` 0.53 → 0.82 by era against `P(y | ¬struct)` 0.04 → 0.14), so an outcome-trained reader
that learns the outcome's cause reads structure. Read with 2a and 2b: the reader learns whatever in the state
best predicts outcome.

> **An outcome-trained reader is a projection of the world model onto the goal. Where a violation carries no
> information about outcomes beyond its immediate damage it correctly ignores legality, and shaping the trunk on
> that goal does not change that. Where a violation is evidence about a hidden state that predicts cost, the
> reader carries the state, its revision is the Bayesian revision, and the revision is scaled by what it already
> believed. Where structure is the cause of cost, it reads structure. It prices what its world model can see, and
> inherits what it cannot.**

## 3. What consumes it: the expectation gates, away from the events

Share of the oracle's prize captured by each gate, inside position strata, `swap65k` 64k, `a` = 0, standard
error about 0.005, at ℓ = 1…4; the anchor venue's numbers at `t_v` beside for the two that matter:

| gate | whole stream | at `t_v` (anchor venue) |
|---|---|---|
| **`V`, the critic's level** | **0.595 / 0.506 / 0.375 / 0.283** | 0.121 / 0.074 / 0.109 / 0.136 |
| `V_pre`, the norm | 0.490 / 0.462 / 0.349 / 0.265 | −0.005 / 0.004 / 0.039 / 0.105 |
| clean-only critic's `V` | 0.579 / 0.509 / 0.374 / 0.281 | — |
| `δ` from the previous query | 0.226 / 0.338 / 0.409 / 0.410 | (at `a` = 4) 0.235 / 0.237 / 0.255 / 0.235 |
| the previous outcome itself | 0.007 / 0.331 / 0.446 / 0.448 | (at `a` = 4) 0.019 / 0.054 / 0.265 / 0.282 |
| the model's surprisal | 0.402 / 0.240 / 0.161 / 0.109 | **0.282** / 0.125 / 0.056 / 0.084 |
| the model's entropy | 0.450 / 0.290 / 0.155 / 0.091 | — |
| `R`, the response | 0.369 / 0.145 / 0.064 / 0.052 | 0.092 / 0.054 / 0.022 / −0.032 |
| banked excess head | 0.055 / 0.031 / 0.022 / 0.055 | 0.110 / 0.095 / 0.049 / 0.023 |
| shuffled `V`, `R`; position; pure noise | ≤ 0.04; ≤ 0.008 | ≤ 0.05 |

**The value level is the sufficient state statistic for the choice.** Conditioned on the value level, the model's
surprisal keeps 0.008 of the prize at ℓ = 2 and every other state readout goes to zero; conditioned on surprisal,
the level keeps 0.354 of its 0.506. Striatum's containment, which the anchor venue had backwards (surprisal kept
0.24 under the value conditioning there), holds cleanly on the stream: that was the anchor population, not the
fact. What survives conditioning on the level is only the actor's own previous outcome, which is not a state
reading, and its norm-subtracted form earns its keep exactly where the norm varies (0.226 against 0.007 at ℓ = 1,
a tie at ℓ = 3–4). The anchor venue had understated the value reading about fivefold, because standing at a
violation is where surprisal is most informative and the level least distinguishing.

**And the level is worth the least at the events it was supposed to be about.** One policy fitted on the whole
stream, share of each subgroup's own prize at ℓ = 2, by offset since the violating token:

| subgroup | quiet windows | before | `t` = 0 | 1 | 2 | 3–4 | 5–8 | 9–16 | > 16 |
|---|---|---|---|---|---|---|---|---|---|
| `V` | 0.557 | 0.443 | **−1.628** | −0.842 | −0.189 | 0.159 | 0.462 | 0.546 | 0.552 |
| surprisal | 0.260 | 0.213 | −0.072 | −0.364 | −0.163 | −0.002 | 0.225 | 0.258 | 0.235 |
| `R` | 0.189 | 0.076 | −0.892 | −0.916 | −0.448 | −0.071 | 0.081 | 0.137 | 0.140 |

The level does not fall nearly enough at a violation to earn the abstention the accuracy drop (0.795 → 0.494)
warrants, and does worse than a constant for three tokens; it recovers by three or four and delivers its full
value from about five on, on quiet windows as much as anywhere. Exposure to damage buys the gate essentially
nothing (the clean-only critic matches the trained one), though [striatum](../striatum/README.md) showed it buys
the detection-side subsumption entirely. The response is the weak object in both venues. Two methodological
facts the venue forced: the bare anchor position, an integer with no state in it, captures a third of the prize
marginally, so every table is read inside position strata with the floor moved to the per-stratum constant and
three pure-noise gates bounding the binning tax; and basalis's screen ports — the binned out-of-sample R² of a
gate's own step function predicts its realised share better than its AUC does (0.938 against 0.830 marginal over
all cells), with the value level the concrete instance of a good gate with a poor AUC.

> **Given something to decide, the object that does the work is the expectation. The level is the sufficient
> statistic of the state for the choice, the norm is the bulk of it and the response a residue, and nothing else
> in the state adds to it. It acts almost entirely away from the events, and against the learner right at them.
> Learning uses the error; acting uses the level.**

## 4. Belief or representation: the norm is public, the response is private

The output layer exposes a rank-15 projection of the 256-dimensional state (the unembedding after layer norm,
with softmax discarding one direction), so a critic refit on the model's own output distribution at the query is
the strict "public" reader. Held-out fit of `V[ℓ, 0]`, `a1` 64k, and what each of norm's objects does on it:

| block | dim | fit ℓ=1 / 2 / 3 / 4 | share of the state critic |
|---|---|---|---|
| state (the critic of record) | 256 | 0.219 / 0.249 / 0.189 / 0.135 | — |
| **output distribution** | 16 | 0.143 / 0.098 / 0.035 / 0.009 | 65% / 39% / 18% / 7% |
| a random 16-dim slice of the state | 16 | 0.105 / 0.057 / 0.038 / 0.024 | |
| the state's top 16 principal components | 16 | 0.133 / 0.101 / 0.042 / 0.017 | |
| four positions of logits plus two tokens | 96 | 0.212 / 0.196 / 0.107 / 0.048 | 97% / 79% / 57% / 36% |

**The level is public.** On the sixteen-number reader, Spearman ρ of the norm against the diet's expected outcome
is +1.000 inside every family at every level, exactly as on the state; the shift-versus-rescaling regression
reproduces (slopes 1.126 → 0.844 against the state's 1.161 → 0.816, with a higher R², so the public map is closer
to exactly affine); the outcome surprise keeps its rank order. **The response is not.** Its amplitude on the
public reader is a third of the state's (ℓ = 3, `full`: −0.024 against −0.071); its damage reading is lost
(0.531 against 0.556, below the random slice's 0.564); at ℓ = 3–4 the output distribution fits value worse than
a random sixteen-dimensional slice; the token-balanced twin optimism keeps its sign at ℓ = 1–3 at 40–60% of the
size; and in the glitch world the public reader keeps nothing of the event signal (every cell under one standard
error where the state reads 5.1σ at ℓ = 2).

**The value direction is anti-aligned with what the output reads, and the trunk learned to put it there.** The
overlap of the critic's direction with the output layer's row space, against a chance line of 0.242 for a random
direction, mean over every critic column:

| step | `a1` (96 columns) | `swap65k` (16) | / chance |
|---|---|---|---|
| 0 | 0.2403 | 0.2389 | 0.99 |
| 8 000 | 0.0435 | 0.0428 | 0.18 |
| 64 000 | 0.0414 | 0.0436 | 0.17 |

At random initialisation the critic's direction is, to within a percent, a random direction in readout terms;
by 8k it is at a fifth of chance and stays there, for every diet and level, the clean-only critic included, on
the same schedule over which norm's twin signal switches on. Splitting the revision exactly into its readable and
unreadable parts, the mean event effect is carried by the readable part with the unreadable part pulling back, and
the per-pair revision is a near-cancellation dominated by the part the logits cannot express.

**The priced twin.** One linear map from the model's forecast difference after the illegal token against its
legal twin to the critic's revision difference, held-out R²: 0.537 / 0.335 / 0.199 / 0.090 at 64k for ℓ = 1…4
(the state difference is the ceiling at 1.000 by construction); knowing only which token arrived recovers 0.524
of the 0.537 at ℓ = 1, so the belief-specific claim rests on ℓ = 2–3. At random initialisation the same map reads
0.604 / 0.599 / 0.586 / 0.478: the belief's grip on the revision is strongest before training and is
progressively lost where the value signal is deepest. And one map fitted on the edit world does not transfer
through the persistence dissociation: the published beliefs about the two worlds' continuations diverge by
comparable amounts, and the edit-world pricing predicts a held value advantage in the glitch world about ten
times what the critic shows (offset 1, ℓ = 2: predicted 0.054, actual 0.006). Whatever tells the critic which
world it is in is not in the logits.

> **The expectation is the priced belief and is readable from the outside; the revision at a violation reads
> directions of the representation the output does not speak, and the model moved the value-relevant directions
> out of the output's subspace while it learned the grammar. The private part is not a residue left over after
> the public part; it is where the learning went. This is paper 2[^private]'s
> re-derivation boundary landing on the value side, with the boundary running between the norm and the response,
> and the sixth instance of the arc's directional-not-scalar pattern with the geometry explicit.**

> **Re-read 2026-09-22 by [`striatum/norm/precision/rereads/public/`](../striatum/norm/precision/rereads/public/README.md), three seeds.** The
> level is public for a linear and a nonlinear public reader alike (an MLP on the sixteen log-probabilities reproduces the norm's calibration and
> rescaling). **"The response is private" is a statement about linear public readers**: the belief-only MLP gives more response amplitude, a better
> damage reading at ℓ=1–2 (0.67 against the state ridge's 0.61) and a stronger twin optimism than the linear state critic, while the state still
> wins at ℓ=3–4 and nonlinear state readers stay above it at every level; the honest phrasing is that the level is linearly public, the shallow-level
> response is public to a nonlinear read of the belief, and the deep-level response needs the state. The ridge's overlap with the output row space
> reproduces on three seeds (0.04–0.05); a critic handed the belief moves its state block to 0.09–0.10, still under half of chance, at cosine
> 0.73–0.88 to the ridge's direction, so part of the anti-alignment above is the linear reader compensating for what it cannot read from the belief.
> Two single-seed rows do not replicate on `s43` / `s44`: the `a1` ℓ=3 "below a random slice" damage comparison (state 0.556 / 0.520 / 0.481
> against the public ridge's 0.531 / 0.471 / 0.481) and the sign of "the readable part carries the mean event effect".

## 5. What held under seeds, and a rule for reading the rest

Three independently trained trajectories (`a1_s42`, `a1_s43`, `a1_s44`; actor accuracies matched to the third
decimal) sort the norm round into two tiers. **Reproduces to three decimals on all three**: the norm's rank
correlation with the world inside every family at every level; the shift-at-init-to-rescaling-when-trained
regression (`out_lo` slopes 1.161 / 1.176 / 1.164, `out_hi` 0.816 / 0.787 / 0.792); the outcome surprise's
ordering and magnitudes; the token-balanced twin optimism at ℓ = 1–3 at 64k (every seed past 2σ at every level);
the persistence dissociation between the edit and glitch worlds (edit world +0.066 / +0.066 / +0.041 at offset 1,
glitch world −0.003 / −0.003 / −0.005). **Does not**: the twin row at ℓ = 4 has no stable sign at either
checkpoint (`−++` at 8k, `+−−` at 64k) and is retired; the glitch world's event-level optimism holds at 8k on all
three seeds at every level, on four times the pairs, and fails at 64k, where the banked seed is positive alone
against two seeds of the opposite sign (ℓ = 2: +0.0192 at 5.1σ against −0.0068 and −0.0102 at −2.3σ). The
addendum's event-level claim stands on the 8k cells.

The re-reads also settled two banked readings. Striatum's across-level contrast, "a null, slightly backwards",
is a scale artefact of ranking wide and deep edits' large revisions against narrow ones: inside `(j, k*)` strata
the linear critic's win rate goes 0.424 → 0.482 and the MLP's 0.462 → 0.502 on `swap65k`, the noise-trained
checkpoint's 0.384 → 0.518; the honest claim is a flat null. A mean-and-variance critic on a continuous target
fits above its degeneracy floor and carries none of Xiang's U-shaped insula row; the variance prediction error is
the squared surprise. And the fixed-trunk refit on the voicing judge reproduced norm §1's rank order exactly while
showing its random-trunk control does not carry over: with a sigmoid readout both trunks rescale by the same
amount (0.557 / 0.871 / 1.153 random against 0.552 / 0.993 / 1.188 trained), because the link function sets the
scale on any feature map; what the trained representation buys is discrimination (AUC 0.618–0.652 against
0.572–0.581). On a sigmoid the level is cheap and the ranking is what the representation earns.

The rule that fell out, adopted for every table above: **what is read as a difference between conditions on
identical inputs replicated; what is read as a rate or a direction along one trajectory mostly did not.** The
adaptation node's deviation from the mechanical null dissolved and its two identical-rows contrasts held at
r ≈ 0.9; tessitura's approach direction was withdrawn and its identical-rows conditionals held to a hundredth;
the one rate-like claim that survived, the lag decomposition, is a claim about the reader having no time constant
of its own rather than about the run it took.

## What this establishes, and what it does not

**Establishes** (one grammar; a frozen linear critic on three trajectory seeds and, for the shaping, regime and
projection rounds, one; the practice learner's own judge on two seeds; every reproduction gate against the banked
nodes exact):

- The value reader's expectation is a running average of recent outcomes that re-calibrates at the rate its
  memory turns over, with no reproducible deviation in either direction, on a frozen trunk across a switch of
  worlds and on a live learner across its own era ladder, where the reader adds 2–5 cycles to a 50-cycle lag that
  is its diet's. The response carries the history on the same clock; the outcome surprise fires at a world change
  and returns; a reader that cannot forget never does.
- Shaping the trunk on the goal widens the state's consequence headroom and sharpens the critic's cost reading,
  keeps the predictive nearly intact when the next-token loss stays on, and leaves the legality reading at
  chance across every diet; task-only shaping leaves the observer family.
- In a world where a violation is evidence about a persistent hidden state that predicts cost, the trunk carries
  the exact regime posterior, the critic's level reads it at quiet positions, its revision at a violation tracks
  the exact filter's revision and shrinks with the prior belief, and the twin gap persists; none of this appears
  in an i.i.d. control at the same corruption rate. The legality contrast that appears in both worlds is
  visibility: the reader is calibrated after the corruption it can see and over-optimistic after the one it cannot.
- On the live learner's judge, structure is read above cost where structure is the proximal cause of cost, at two
  seeds; the two labels are not independent there, and the ordering is a property of that world.
- Given an abstain-or-commit choice, the critic's level captures about half of the oracle's prize on the whole
  stream, beats every surprise-shaped reading, is the sufficient state statistic for the choice, and is worth
  the least at the violating token and the most from five tokens on and on quiet windows; the response gates
  almost nothing.
- The norm and everything that depends on it are recoverable from the model's output distribution alone; the
  response, its damage reading, its deep-level fit and its glitch-world signal are not, and the critic's direction
  sits at a fifth of the chance overlap with the output layer's row space from 8k on, having started at chance.
- Three trajectory seeds: the expectation-level findings of the norm round reproduce to three decimals; the
  ℓ = 4 twin row and the 64k glitch-world event reading do not; the 8k glitch-world reading does.

**Does not establish:**

- Which of the level and the outcome surprise is "the feeling" in a functional sense beyond this one consumer:
  the level gates action, the error trains the level, and a consumer richer than a priced abstention was not
  built. The private part of the revision is real and directionally rich and nothing here consumes it.
- Whether the regime effects survive a world where the model's surprisal and the regime evidence are not nearly
  one variable, or a shaped trunk on that world (the shaping diet would have to be the world's own windows).
- The specific overlap numbers of §4: this trunk has a sixteen-token vocabulary and an unusually tight output
  bottleneck. The below-chance floor and its appearance between step 0 and 8k are the claims.
- The voicing round's structure-over-cost ordering as a reversal of striatum's: the labels are not dissociable
  there, and a design that dissociates them on the practice learner has not been built.
- Anything off `v16 s2 L6 m4` and the voicing substrate; the language-model port is held for discussion.

## Corrections to banked nodes, made from this round

Each is a dated note in the node it corrects, with the numbers in the re-read tables.

- [`striatum/README.md`](../striatum/README.md) §5: the across-level tilt is a width artefact; a flat null.
- [`striatum/norm/README.md`](../striatum/norm/README.md) §1: the level's calibration is a property of the readout
  on any feature map; the ranking is what the representation buys (held as the epistemically uncertain reading
  until a dissociating design exists). §3: the ℓ = 4 twin row retired. §5: the event-level equality of the two
  worlds stands on the 8k cells; the 64k cells reported as not replicating.
- [`ideas/calibration_and_violation_are_one_object.md`](../../../../ideas/calibration_and_violation_are_one_object.md)
  §11: "projection" where §5 has "organ"; the one-liner's placement of the norm.

## Reproduction

```bash
cd experiments            # MODAL_PROFILE=chromatic
D=/data/v16_s2_L6_m4_distinct/logit_reading
# adaptation: two L4 cells, ~80 s and ~165 s; add --order-seed 1313 --tag ord2 for the second order seed
modal run --detach -m rhm.logit_reading.orbitofrontal.adaptation.task::adapt_sweep --block-win 25
python -m rhm.logit_reading.orbitofrontal.adaptation.analyze <dir>/traj_a1_s42 --tags a1,swap65k
# projection: six cells (steps 0 / 8k / 64k x a1 / swap65k), 73-173 s each; the step-0 swap65k cell needs --max-twins 8000
modal run --detach -m rhm.logit_reading.orbitofrontal.projection.task::proj_sweep \
    --cells "$D/traj_a1_s42/step064000.pt:a1:1:0,$D/traj_a1_s42/step064000.pt:swap65k:0:1"
python -m rhm.logit_reading.orbitofrontal.projection.analyze <dir>/traj_a1_s42 --tags a1,swap65k
# shaped: three fine-tune arms (~2 min each), then the banked striatum / junction / calibration batteries on each
modal run --detach -m rhm.logit_reading.orbitofrontal.shaped.shape::shape_sweep
modal run --detach -m rhm.logit_reading.orbitofrontal.shaped.shape::probe_sweep
bash rhm/logit_reading/orbitofrontal/shaped/results/fetch.sh <mirror> && bash rhm/logit_reading/orbitofrontal/shaped/results/reduce.sh <mirror>
# regime: two worlds, two trajectories (~40 min each), four readout cells; ~2.3 GPU-h
modal run --detach -m rhm.logit_reading.orbitofrontal.regime.task::regime_all \
    --worlds burst,iid --n 65536 --steps 64000 --read-steps 0,8000,24000,64000
bash rhm/logit_reading/orbitofrontal/regime/results/fetch.sh <mirror> && bash rhm/logit_reading/orbitofrontal/regime/results/reduce.sh <mirror>
# abstain: the anchor venue is CPU on the banked striatum cells; the stream venue is two L4 cells
python -m rhm.logit_reading.orbitofrontal.abstain.analyze <dir>/traj_a1_s42 --tags a1,swap65k
modal run --detach -m rhm.logit_reading.orbitofrontal.abstain.task::abstain_sweep
python -m rhm.logit_reading.orbitofrontal.abstain.analyze_stream <dir>/traj_a1_s42 --tags a1,swap65k
# tessitura, the re-reads and the extra trajectory seeds: see their own FILES.md and the NOTES file
```

Every child's `FILES.md` carries the exact command of record, the app ids, the measured wall clock and peak RSS,
and the volume paths (`rhm-scaling-data`, **`chromatic`** workspace, under `$D/` and, for tessitura,
`/rhm_practice_voicing/ts_s{0,2}/`).

## Gotchas worth not rediscovering

- **Read a rate claim against its own null and a second ordering; read a state claim on identical rows.** Anything
  expressed as "fraction of a gap closed per window" inherits the fitter's arithmetic; the block-Gram fitters
  print the mechanical fraction beside every measured one for this reason. Sign tables across seeds that print
  every seed's sign regardless of significance catch the case a significance filter hides (one seed at +5σ,
  another at −1.8σ, printed as agreement).
- **Stratify the decision by position before reading any gate.** An integer with no state in it took a third of
  the prize; the unstratified table reads the opposite conclusion. Carry pure-noise gates to bound the binning tax.
- **The anchor population is not the stream.** Every ordering of the consumer round flipped between the two
  venues; read a consumer on the learner's whole life.
- **Legality in a corrupted stream is a label on the prefix.** After the first violation in a window the exact
  predictive is undefined, so every later token reads as illegal; label legality only where the within-window
  prefix is still legal, and apply the same filter to both worlds.
- **A sigmoid readout calibrates its mean on any feature map.** Norm's random-trunk control (shift only) is a
  property of an unbounded ridge; on a logit-through-sigmoid judge the random trunk rescales too. Read a
  calibration claim beside a discrimination claim.
- **Flag a critic still at its initialisation.** A judge that has not begun governing reads 0.500 at every
  candidate; declare the exclusion mechanically before the tables and print every pattern both ways.
- **At random initialisation a surprisal caliper stops filtering.** All eligible violations pass, the twin file
  grows past the size at which `modal volume get` has silently truncated; cap the twins. And touch every member
  of every fetched `.npz`: one fetch returned a full-size file with a bad CRC on one member that `np.load`
  did not catch.
- **A sibling's actively streaming launch log inside the tree breaks the Modal image build even with the
  `ignore`.** Stage the package's `.py` files into the scratchpad and run `modal run` from there. `modal volume
  ls` returns unsorted; sort before concluding anything about a run's progress.
- **Basalis's log Bayes factor does not port to a per-position venue** (16 branch passes per position); the
  regime filter is its running form and needs the same reader in both worlds.

[^private]: Not mirrored: this link points to a document in the private lab repo (the roadmap, the queue, an unrun spec, reading notes, or a conversation). See the top-level README for what is held back and why.
