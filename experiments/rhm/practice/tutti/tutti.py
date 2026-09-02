"""tutti (the unification node) — BOTH CURRENCIES, AND THE SELECTOR, IN ONE LOOP.

FORKED from `../caesura/caesura.py` (E' round 3), which is untouched; every addition here is
marked `# [tutti]`. IMPORTED, not forked: `../maestro/policy.py` (A1's rule and its measured
dead zones), `../antiphon/questions.py` (the question port's selectors and its quota — so
gates Q-1..Q-10 apply here unchanged), and, through the donor, `../native/span/span_net.py`.
Through the donor: `../intonation/` (delta_perf, the fallible executor), `../tacet/`,
`../crescendo/` (A3 — the yoke idiom, the L4 frontier, the floors), `../maestro/`,
`../conductor/`.

THE QUESTION. The loop has THREE actions — COMMIT level l, ADVANCE the era, SELECT the
questions — and two currencies can license them:

  outcome currency    one-level-up `at_support` yield, A1's thermostat (`crescendo`, A3).
  execution currency  delta-SILENCE, `dsil` = mean b(s) over OPEN slots, read through A1's
                      `QuietPolicy` unchanged (`caesura`). Live only because `intonation`
                      un-gated the span head below parity, so the executor is fallible.

`two_deltas` interpretation (b) asks whether the within-level type law is CURRENCY-scoped
rather than LEVEL-scoped. `caesura` finding 5 is the only positive evidence for it: single
seed, un-yoked, and governed by a floor 1.43x TIGHTER than its own in-tag noise. `antiphon`
adds a third lever whose deep-era value was ~ fully INTERACTION with the pacer. So:

  (1) do the two currencies DIVIDE LABOR or INTERFERE across the loop's actions --
      (a) delta-silence owns both, (b) yield owns both, (c) the SPLIT: delta-silence paces
      commit, yield owns era advance (the QUEUE's stated shape, unmeasured)?
  (2) does `caesura` finding 5 survive RE-INSTANTIATION at a new fork -- a different file,
      a corrected dead zone, and a world in which the selector is live?
  (3) does the selector's value depend on WHO PACES -- the interaction, one level up from
      `antiphon`'s?

WHAT THIS FORK ADDS — two things, and no new rule.

  1. THE QUESTION PORT, grafted from `antiphon.py`'s `# [antiphon]` block. Each cycle the
     world offers a MENU of K = 2048 candidate repair episodes drawn from the era's own
     damage cell, and the arm's SELECTOR picks the n_pr = 64 it practises on. The damage cell
     at level l is exactly one half of the level-(l+1) span that gets mined, so the selector
     controls half of every mined key by construction. `exo` = the menu's head in the donor's
     own RNG order (i.e. NO selection, and a bit-identical replay of the donor); `endo` =
     half-key novelty vs the arm's own at-support set x a posed-vs-landed delivery ledger --
     the only selector that can be an outer-loop ACTION. The selection compute (a reader
     forward and a value forward over all 2048) runs in EVERY arm and is used only by the
     arms whose rule reads it (`ostinato`'s rung-invariant-comparator discipline).

  2. `loop_commit` — A SECOND POLICY OBJECT that owns the COMMIT while the primary owns the
     ADVANCE. This is not a new rule: `QuietPolicy` is already a pure object with its own
     buffer, latch and clock, and the donor already builds a second one (`dsil_det`, the
     veto's read-only detector) in every arm. The split is that detector promoted from
     read-only to licensing. Pinned, and each could have gone the other way:

       - PRECEDENCE is the donor's, unchanged: a quiet commit-latch COMMITS the active level
         if there is one to commit; a quiet advance-latch ADVANCES otherwise. If both are
         quiet on one cycle, COMMIT wins and the advance is not taken that cycle.
       - BOTH policies reset on EVERY action and at every era start, not only on their own.
         The donor's stated reason for resetting is REGIME CHANGE, and a commit installs a
         table (a regime change for the yield gauge) while an advance changes the damage cell
         (a regime change for the executor delta-silence hears). Resetting own-action-only
         would make the split differ from (a)/(b) in TWO things rather than one. The
         own-action-only rule is a COUNTERFACTUAL readout in the reduction, never an arm.
       - THE BOOTSTRAP carries over verbatim: delta-silence cannot license the first crossing
         (slots are minted BY a commit — gate D), so where the commit owner reads `dsil` and
         the gauge is absent the commit falls back to the arc's default rule (`delta_prov`)
         and is logged `kind: "bootstrap"`. In the split this produces a configuration that
         has never run: bootstrapped commits ALONGSIDE yield-driven era advances in era 1.
       - A cycle whose read is undefined is not a decision point, PER POLICY.
       - The yoke generalises for free: `YokePolicy` replays COMMIT and ADVANCE cycles
         separately from a plan, so `loop_commit` is ignored under `kind == "yoke"`.
       - With `loop_commit` absent, `loop_c is None` and every line is the donor's (gate X-2).

THE FLOOR. `tol_dsil` is re-derived in-tag by A1's own null-ABBA statistic on `ca_s0`'s logged
`dsil` series (a pure function of `log["panel"]`, so it costs zero GPU): POOLED 0.0046 over the
donor's four arms, against the 0.00321442 the donor's run was actually governed by. The
consequence is stated rather than hidden: `tu_d_exo` is then NOT a bit-identical replay of
`ca_s0/dsil_read` — offline replay of A1's rule at both floors puts the first divergence in the
quiet verdict at c43 — so that arm is a BOUNDED cross-tag gate over c1..c42 and a genuine
re-instantiation above it, while `tu_y_exo` (yield-paced, and therefore floor-independent)
carries the round's full-scale replay gate.

L5 IS CLOSED IN THIS TAG AND CANNOT BE OPENED IN IT. `max_macro_level` also sets the proposal
head's slot layout, the ranges of the committed/miners dicts, the panel's committable half and
the shadow audition's matched-size random control, which draws from the arm's own shared RNG
stream — so an maxl=5 arm diverges from `ca_s0` in the shared stream from the first cycle
`miners[5]` is non-empty, and BOTH cross-tag replays die. See `DESIGN.md` §8 for the shape of
the sibling tag.

  --- the donor's own header follows, unedited ---

intonation (E' round 2) — THE delta_perf NODE: a live, FALLIBLE executor, so that
PERFORMANCE ERROR (execution-vs-intention) exists on this substrate for the first time.

FORKED from `../tacet/tacet.py` (E3'), which is untouched; every addition here is marked
`# [intonation]`. Through it: `../crescendo/` (A3 — the thermostat, the L4 frontier, the yoke,
the measured floors), `../maestro/policy.py` (IMPORTED, not forked — this round adds no
outer-loop rule), and `../native/span/span_net.py` (IMPORTED — the executor primitive, the
parity gate and gates S-1..S-6 are that module's, unmodified).

THE DIAGNOSIS THIS IS BUILT FROM. Two quantities got one name. The bridge line
(`ideas/performance_error_is_the_bridge.md`) defines delta as a PERFORMANCE error —
`e = ||realization - intention||`, benchmarked against a CONTEXT-CONDITIONAL running average of
itself `b(s)`, and gated on agency: `delta_perf = (b(s) - e) * sigmoid((g - g0)/theta)`. It
reads execution quality INDEPENDENT OF TASK SUCCESS. The E-track nodes on this stack
(`audiation`, `tacet`) instead used `delta = grade - v(s)`, an OUTCOME surprise.
`../../../mjc/two_clocks/` measured that credit factorizes into precision (which experiences —
delta_perf's content) x value (toward what end — the grade's content) and that sharing one
channel is DESTRUCTIVE interference; gating mining on outcome-delta is the shared-channel case.

WHY RHM HAS NEVER HAD delta_perf. The A-track stack is routing-only: a macro is executed by
`macros.apply_any`, an exact max-sum DP over `T[l-1]` plus per-block infill. Intention and
realization COINCIDE, so `e == 0` identically and the quantity is zero by construction.
`../native/span/` built the live executor (`SpanHead`, conditioned on the macro slot, emitting
the span's level-1 features) and gated it SHUT behind a held-out exact-match parity gate at
tau = 0.95 — i.e. it is allowed to fire only where it is already ~never wrong.

WHAT THIS NODE CHANGES — one thing. The head fires BELOW parity (`span_tau_fire`, default
0.50, with `span_min_hold` untouched), so realization is FALLIBLE and `e` is a live quantity:

    intention   what `apply_any` would write for this slot command on this observation.
                A FREE, EXACT reference: the head's own firing path already computes the
                trunk's block logits, and the DP is a pure function of them (`dp_features`).
                No learned forward model is needed anywhere.
    realization what the head actually emitted.
    e           1 - (blocks matching) / span, per executed row.  [graded Hamming; the
                exact-match BIT is `e == 0` and is logged beside it]
    b(s)        per-macro-SLOT EWMA of e — the slot is the syllable, i.e. Gadagkar's
                per-context benchmark. NEVER a global scalar (S13(b): the scalar form
                fixates worse than raw error).
    g           agency = (the head's emission BECAME the realization) x (the arity gap:
                the fraction of the span where the realization differs from the
                SLOT-FREE per-block argmax, i.e. what the macro command explains).
                Graded, per S13(a). g == 0 identically on base moves and on DP-executed
                macros: those are this substrate's PLAYBACK condition.
    delta_perf  (b(s) - e) * sigmoid((g - g0)/theta), the gate CENTERED (sigma(0) = 0.5
                leaks half the channel — S13(a)'s correction), g0/theta calibrated
                self-supervised from the arm's own active/passive g medians.

THE READOUT THAT HAS NEVER EXISTED HERE: the 2x2, executed-as-intended x solved, per
trajectory. Outcome-delta gives the well-motivated failure and the ill-motivated success
maximal credit with opposite signs, and mining keys on the outcome column — so the current
rule mines the lucky success and discards the honest failure. Every arm in this tag tabulates
that 2x2: counts, what was mined from each cell, what pi mass formed, and the era table.

THE TWO CONSUMPTIONS (each against its own control, and never sharing a channel with the
grade — `two_clocks`' separate-channels law is a design constraint here):

  (i)  delta_perf as a PER-SAMPLE GAIN on the span head's own plasticity (the bridge's
       efferent form, `w = exp(-delta/tau_w)`, budget-matched by running-mean normalisation —
       `plasticity_gain`'s canonical form) against UNIFORM (ungated) head training and against
       a RAW-`e` gain (S13(b)'s hygiene comparison).
  (ii) delta_perf as a GATE on what gets mined and what trains pi, against grade-only
       selection (the current rule) and against the OUTCOME-delta gate `tacet` ran.

ARMS (single seed; every arm is a clock YOKE of `perf_log`, so all six are lifetime-,
era-boundary- and commit-cycle-identical and differ in exactly one knob — `tacet`'s reasoning,
unchanged; the pacing half is bought back offline by the thermostat replay in the reduction):

    perf_log     span head firing below parity; delta_perf METERED AND CONSUMED BY NOTHING.
                 The instrument arm, the uniform-plasticity control for (i), the grade-only
                 baseline for (ii), and the clock source every other arm replays.
    perf_gain    (i) delta_perf gain on the head's per-sample plasticity.
    perf_raw     (i) raw-`e` gain, no benchmark, no agency gate — the hygiene control.
    perf_gate    (ii) delta_perf gate on mining and on pi, at `tacet`'s matched volume.
    outcome_gate (ii) `tacet`'s `gate_delta_hi` rule (delta = grade - v) at the same volume,
                 on this same live-executor substrate — the matched-volume CONTENT control.

PRICING, SAID OUT LOUD. (a) The intention reference is an INSTRUMENT, not part of the agent's
execution: it consults the committed table on the head's firing path, so `native/` finding 5's
table-ablation claim is NOT made in a delta-metering arm, and `blk_ref` is tallied separately
and never folded into the ledger. (b) Head misfires are NOT priced into `t` — the ledger is the
arc's cross-tag comparable instrument and making it arm-dependent would confound "the gate
changed what was learned" with "the gate changed what things cost"; instead `n_misfire` and the
counterfactual `t_misfire = n_misfire * c_mat` are printed BESIDE the ledger, never in it.
(c) delta metering draws no RNG on the shared stream and costs no grounding.

FIDELITY. With every `# [intonation]` knob off, this file replays `tacet.py` in process at
0.000e+00 (gate G-F), and `tacet.py` in turn replays `crescendo.py`. The span head, its parity
gate and its executor are `span_net`'s, imported: gates S-1..S-6 are re-run here unchanged
(`span_selfcheck` in `../native/span/span.py` is the record; gate I-0 asserts the imports are
that module's). A `PerfExecutor` with `perf_meter=False` and `span_tau_fire=None` IS
`SN.SpanExecutor`.

DONOR DOCSTRING FOLLOWS.

tacet (E3') — THE delta-GATE: does choosing what NOT to learn from change what is learned?

FORKED from `../crescendo/crescendo.py` (A3), which is untouched; every addition here is
marked `# [tacet]`. The substrate, the ladder, the reads, the shadow panel, the yoke mechanic,
the measured floors, `commit_max_level` and every gate are A1's, A2's and A3's. `policy.py` is
NOT forked — this round adds no outer-loop rule; it IMPORTS A2's module, which carries A1's
thermostat byte-for-byte. `crescendo.py` is imported only by the G-F smoke, as the donor.

THE QUESTION (ROADMAP Sec 7.1.3, first shape E3'; QUEUE "Track E'"). The current learning rule
is GRADE-ONLY SELECTION: mining consumes the agent's own chosen answers on the instances it
solved, and pi is supervised on every surviving trajectory that solved (`train_on="solved"`).
Nothing else about the datum is read. `audiation` (E1/E1b) measured that the per-datum signal
an in-loop gate should consume is alive at DECISION TIME and needs no forward model: delta =
grade - v(s), the residual of the value head's own outcome forecast (pre-update readout decodes
the value-error target at 0.293), and the DELIBERATION STATE, the unchosen candidates' own
value scores (0.133), against <= 0.05 for every update-derived source. So this round gates on
those two scalars and on nothing else. There is no forward model anywhere in this file.

WHAT CHANGES, AND IT IS ONE FUNCTION.

  1. THE GATE (`gate_select`). Per cycle the practice beam returns B = `n_pr` instances x
     W = `pr_width` surviving tips. Two consumption sites are gated, at the unit each one
     actually consumes:
       * MINING consumes `out["x"]` — the beam's own chosen answer per instance — restricted
         to the instances it SOLVED, then randomly subsampled to `mine_cap`. The gate replaces
         the random subsample with a delta-ranked one AT THE SAME CAP, so mining volume is
         EXACTLY unchanged and only the content moves. (Measured on `cr3_s0/outer_yield_m4`:
         ~16 instances solve per cycle against a cap of 8, and the cap binds on 96% of cycles,
         so the mining channel is a 50%-selective, exactly-volume-matched content choice on
         almost every cycle of the run. Stated because it is a design fact and not an
         omission: on that channel the RANDOM gate is the current rule re-drawn — same rate,
         different draw — so `gate_random` is the honest floor for "a gate that selects at
         this rate on no signal", in both channels, rather than a copy of the baseline.)
       * pi SUPERVISION consumes every surviving tip that solved, all `budget` steps of it.
         The gate keeps `gate_frac` of those tips, ranked by the same scalar. Volume is matched
         ACROSS gate arms by construction (the same K every cycle), and `gate_random` is the
         volume control against the ungated baseline.
     The VALUE BUFFER is deliberately NOT gated. It is the source of the very forecast delta is
     the residual of; gating its diet would make the gate self-referential and would confound
     "the gate changed what was learned" with "the gate changed the gauge".

  2. THE SCALARS ARE ALREADY ON THE WIRE, and cost nothing. `beam_moves`/`beam_moves_prop`
     already compute `final` — the value head's score of every surviving tip — to pick the
     answer. This fork returns it (`out["tip_val"]`, `out["best"]`); no extra forward pass, no
     extra grounding charged, no RNG touched. From it:
         delta_tip[i,w] = succ[i,w] - sigmoid(final[i,w])       (per trajectory)
         delta_ans[i]   = ps[i]    - sigmoid(final[i,best_i])   (per instance, the mined unit)
         margin[i]      = sigmoid(top1) - sigmoid(top2) over instance i's W tip scores
                          — the deliberation state, as a scalar: how decided the final
                            selection was, i.e. how much the unchosen candidates' own value
                            scores disagreed with the chosen one.
     Both are materialised BEFORE the grade is applied, which is the surviving "timing" claim
     of ROADMAP Sec 4.2 in the scalar form Sec 7.1.2 says the value head already satisfies.

  3. NO GRADIENT-BUDGET MATCHING IS NEEDED, and that is a property of the substrate, not a
     concession. `prop_train` takes a FIXED `prop_steps` gradient steps per cycle sampling from
     a fixed-capacity replay buffer, so a gate changes the buffer's DIET, never the number of
     updates; mining takes no gradients at all; the value head is ungated. `audiation`'s
     per-datum arm had to match Sigma-lr to its anchor; here there is nothing to match.

  4. THE ARMS ARE PACING-MATCHED BY THE YOKE. The outer loop's actions are absorbing, so an arm
     whose gate changes what pi learns would also drift in WHEN it commits and advances, and
     the deep-era comparison would conflate the gate's content with the gate's pacing. Every
     gate arm is therefore a CLOCK YOKE of the ungated baseline (`crescendo`'s `ceiling_m3`
     mechanic): it replays `outer_yield_m4`'s realised commit and advance cycles, so it is
     lifetime-identical, era-boundary-identical and commit-cycle-identical, and differs from
     the baseline in exactly one knob — what the gate let it learn from. The pacing question is
     then answered OFFLINE at no extra arm: A1's thermostat is replayed on each arm's own
     logged L4 at-support series (`phase0_l4.py`'s machinery, reused by the reduction), which
     says when each arm WOULD have committed had it been driving.

  5. THE ARMS. Six, all routing-only, all on the anchor's stream, all at `max_macro_level=4`,
     all sharing `crescendo`'s Phase-0-sized ladder and caps:
       outer_yield_m4   A3's treatment, VERBATIM (gate off)      the baseline == GRADE-ONLY
                                                                 SELECTION, the clock source,
                                                                 and the full-life cross-tag
                                                                 replay gate against cr3_s0
       gate_delta_hi    yoke; keep the HIGHEST-delta successes   "learn from the surprising"
       gate_delta_lo    yoke; keep the LOWEST-delta successes    "learn from the expected"
       gate_delib       yoke; keep the LOWEST-margin instances   the deliberation-state gate:
                                                                 "learn from the contested"
       gate_random      yoke; keep a random K at matched size    THE VOLUME CONTROL
       gate_all         yoke; `mine_cap=0` + `prop_train_on=     LEARN FROM EVERYTHING
                        "tips"`                                  (see below)
     Both delta directions are run because ROADMAP Sec 1.2 claim 3 / Sec 4.3's trust-vs-habit
     framing makes either sign interesting and an arm pair is cheaper than an argument.

  6. WHAT "LEARN FROM EVERYTHING" MEANS HERE, stated because it is a design call. Mining is
     only DEFINED over solved configurations — "a solved config is a valid r* derivation, so
     the target is well defined" (the substrate's own comment on `finetune_generator`); mining
     an unsolved answer would feed the miner a parse of something that is not a derivation, so
     "everything" there would not be learning from everything, it would be learning from
     garbage. So `gate_all` takes the widest well-defined diet on each channel: mining drops
     the `mine_cap` subsample entirely (every solved answer, not 8 of them), and pi is
     supervised on EVERY surviving tip regardless of grade (`prop_train_on="tips"`, an existing
     donor knob — the beam's value head selected those tips, so they are the widest set the
     donor's own machinery admits). Both are donor knobs; `gate_all` adds no new code.

  7. FORK DISCIPLINE. With `gate_mode=None` (the default everywhere) every addition is inert:
     the two new `out` keys are unread, `prop_pairs` takes its donor path, and the mining
     subsample draws the identical `rng.permutation` from the identical stream. `fidelity_smoke`
     asserts the in-process replay of `crescendo.py` at 0.000e+00; `preflight` asserts a pure
     yoke with the gate off is bit-identical to the baseline; and the reduction asserts
     `outer_yield_m4` replays `cr3_s0/outer_yield_m4` over its WHOLE life.

DONOR DOCSTRING FOLLOWS.

crescendo (A3) — THE SIGNATURE: DOES THE EARNABLE RANGE EXTEND WITH THE TURN OF THE CRANK?

FORKED from `../maestro/maestro.py` (A2), which is untouched; every addition here is marked
`# [crescendo]`. The substrate, the ladder, the reads, the shadow panel, the yoke mechanic,
the measured floors and every gate are A1's and A2's. `policy.py` is NOT forked — this round
adds no rule, so it IMPORTS A2's module, which carries A1's thermostat byte-for-byte. That
import is the structural form of the claim "the rule did not change; the rung did".

WHAT CHANGES, AND IT IS ONE CONSTANT. Every run in this arc has held `max_macro_level=3` and
treated level 4 as unearnable. A3 opens L4 to the crank. `active = era.level + 1`, so with
`max_macro_level=4` level 4 is mined from era 3 (the mining gate `min(maxl, era.level + 1)`)
and committable in era 3 — exactly the deal L2 got in era 1 and L3 got in era 2. The ladder
already outruns the earnable range on this substrate, so `tall`'s re-run recipe needs no
deeper world: it needs PERMISSION and TIME AT THE FRONTIER.

  1. `commit_max_level` — THE ONE NEW CFG KEY, and the round's real control. `max_macro_level`
     caps what may be MINED, OBSERVED, SLOTTED and AUDITED; `commit_max_level` caps what may be
     COMMITTED. Splitting them is what makes the ceiling control exact: `ceiling_m3` runs at
     `max_macro_level=4` — identical proposal head, identical miners, identical observation
     panel, identical per-cycle shadow auditions, identical RNG stream — and differs from the
     treatment IN ONE BIT, whether the L4 commit may install a table. Running the control at
     `max_macro_level=3` instead would ALSO change the head's slot layout, the `committed` dict,
     the audition loop's range and (through the matched-size random control's `rng.permutation`)
     the shared RNG stream from era 3 on. Defaulting `commit_max_level` to `max_macro_level`
     makes this file identical to its donor when the key is not passed.

  2. THE ARMS. Four, all routing-only, all on the anchor's stream, all at `max_macro_level=4`:
       anchor_long      certificate-else-boundary; era lengths == the caps   the carrier and
                        the BUDGET-MATCHED comparator (see LIFETIME below)
       outer_yield_m4   A1's thermostat, L4 committable                      the treatment
       ceiling_m3       clock replay of `outer_yield_m4`'s realised actions,
                        `commit_max_level=3`                                 THE CEILING CONTROL
       outer_yield_m4x  the treatment + `census_extend`'s post-commit
                        extension                                            the r**2-wall bypass

  3. LIFETIME, MATCHED BY CONSTRUCTION — A1's and A2's standing caveat, closed. In both donors
     the loop arms outran the anchor (139 vs 116 cycles), so every deep-era comparison carried
     a pacing-vs-time confound. Here the anchor's SCHEDULED era lengths are set equal to the
     CAPS, so the comparator runs the maximum lifetime any loop arm is allowed and a loop arm
     can only be SHORTER. Any loop advantage therefore cannot have been bought with time. And
     `ceiling_m3`, being a clock yoke of the treatment, has the treatment's era boundaries
     cycle-for-cycle: the pair the signature is read on is lifetime-identical by construction.

  4. THE GAUGE AT THE FRONTIER, CHOSEN BY MEASUREMENT. One level up from L4 is L5, and Phase 0
     measured the L5-shaped read DEGENERATE on this substrate: `obs_hist[5]` reaches a maximum
     of 1-2 distinct tuples at support over an ENTIRE run and is first non-zero at c89-124
     (level 5 has 205,824 distinct true tuples against ~8 observations/cycle). A thermostat's
     paired-interval slope on a series whose whole dynamic range is {0,1} is zero almost
     everywhere. So the L4 commit is paced on THE L4 STREAM'S OWN QUIETING — `read_level`
     clamps to `gy_level`=4 exactly as it already does, which at era 3 makes the driven read
     the level being earned rather than one above it. Stated, not hidden: `read_level` is
     logged per cycle, the L5 and L6 series are logged uncharged in every arm's panel, and the
     reduction prints both. The nearest ancestor of an at-support read at the committing level
     is `census`'s G-A admission-rate gauge, NOT `outer_ledger`'s within-level task error --
     the reader A1 finding 2 measured to refuse.

DONOR DOCSTRING FOLLOWS.

maestro (A2) — DOES *LEARNING* THE RULE BUY ANYTHING THE THERMOSTAT DID NOT?

FORKED from `../conductor/conductor.py` (A1), which is untouched; every addition here is
marked `# [maestro]`. The substrate, the ladder, the caps, the reads, the shadow panel, the
yoke mechanic and every gate are A1's. What A2 changes is WHO WRITES THE RULE: A1's
hand-written thermostat is replicated in-tag as the comparator, and beside it run two arms
of one small LEARNED class (`policy.LearnedPolicy`, fitted offline by `fit.py`) that differ
from each other in ONE THING ONLY — the reward the fit was run against:

    learned_yield   reward = NEXT-LEVEL YIELD   (at_support one level up)      the treatment
    learned_task    reward = WITHIN-LEVEL       (the arm's own metering error) the control

Same policy class, same inputs, same fitting procedure, same measured-floor discipline, no
exploration in either. `composed_loop`'s question, re-asked with the action set that matters.

The fit is OFFLINE and the policy is FROZEN before the run starts — see `fit.py` for why
(both actions are absorbing, so within-run exploration of commit timing is close to
impossible) and for what the corpus's n does and does not support.

DONOR DOCSTRING FOLLOWS.

conductor — A1, the composition: teacher_slot's gauge-reading outer loop DRIVING the
consolidated practice learner's crank, instead of the schedule and the certificate.

FORK NOTICE. This file forks `../assay/assay.py` VERBATIM and adds five things, each marked
`# [conductor]` at its insertion point. `assay/`, `census/`, `spiral/` and everything upstream
are NOT modified. With the loop off (`commit="delta_prov"`, the `anchor` arm) this file IS
`assay.py`: the era loop's rewrite from a `for` over a fixed cycle count to a `while` with a
cap is exactly equivalent when nothing ever advances early, the reads consume no RNG and no
gradient, and the in-tag `anchor` doubles as the cross-tag replay gate against `as_s0/anchor`.

  1. THE OUTER LOOP OWNS THE CRANK'S ACTIONS. `../conductor/policy.py` (out of the Modal app,
     so every rule is auditable with no GPU) supplies a thermostat-grade rule over ONE scalar
     read: hold while the read still moves, act when it quiets inside a dead zone that was
     MEASURED (`floors.py`, the null-ABBA method) and never defaulted. Its two actions are the
     two the crank has this round — COMMIT the active level, and ADVANCE the era — and it takes
     them in place of the certificate and the ladder's fixed cycle counts. The era SEQUENCE (the
     damage ladder and its cells) is the world and does not move; only the timing is the loop's.
     Read `policy.py`'s docstring for why the donor's ABBA paired trial does not port to an
     absorbing action, what was kept, and what the estimator is and is not.

  2. THE OBSERVATION PANEL. `census`'s G-Y miner is an unpriced, read-only level-4 miner that
     observes in EVERY era. This file generalises it to every level 3..depth, for one measured
     reason: `gauge_hist[3]` is identically ZERO through the whole of era 1 (verified on
     `as_s0/anchor`: 0 at every cycle c1..c48), because the committable miners are deliberately
     gated to `era_level + 1` so era 1 cannot hand era 2 a finished vocabulary. The gauge an
     era-1 loop needs — level-3 minability while level 2 is being earned — therefore does not
     exist in the donor at all and has to be instrumented. The panel NEVER commits, never gates,
     never reaches `committed`, and never touches the era-gated `miners[...]` the tables are
     built from; its level-4 entry is asserted equal to the donor's G-Y miner in-run.

  3. THE ENDO READ — endo_yield's label-free one-level-up loss, ported. RHM's tree is balanced,
     so which span closes one level above the era's damage cell is a fact about the grammar's
     SHAPE (depth, s, and the era's own cell, which the agent lives in) and not about its
     content. `endo_read` masks that parent span entirely and scores the plant's own masked-
     infill NLL on it — the learner's own training objective, on a fixed clean batch, with no
     table, no oracle and no truth mask anywhere in the computation. Masking the WHOLE parent
     span is what makes it one level up: recovering it requires the level-(l+1) latent inferred
     from outside the span, where masking only the cell requires the level-l latent. Gate N-1
     asserts the label-freeness by construction; `endo_bench` prices it.

  4. THE SHADOW PANEL AND THE LEDGER (`endo_yield`'s pattern, `ear`'s convention). Every
     candidate gauge is logged in every arm at every decision point, uncharged, so a
     counterfactual decision trace exists for every gauge in every arm. The ledger charges only
     what the POLICY's input stream contains: the arm's own error and its own mined counts are
     free (the learner's own experience), and the endo read — a forward pass the arm does not
     otherwise make — is priced at its measured cost. Pricing enters `counts["ground"]` and so
     `t_cum`, which nothing in the cycle loop reads: the ledger is a readout, never a governor.

  5. THE YOKED ARMS (`census`'s mechanic, widened to both actions). `conductor_run` runs each
     gauge arm first and passes its REALISED action cycles — commits and era advances, including
     the advances forced by a cap — into its yoke, which replays them by clock and by nothing
     else. `census` finding 4 (`census_gate` == `yoked_delay` bit-identical over 116 cycles) is
     why these are mandatory rather than optional.

--- the donor's own header follows, unmodified ---

assay — which ingredient of the complete vocabulary is the currency?

FORK NOTICE. This file forks `../census/census.py` VERBATIM and adds four things, each marked
`# [assay]` at its insertion point. `census/`, `spiral/` and everything upstream are NOT
modified.

  1. ORACLE SURGERY AT COMMIT. At each level's own commit cycle an arm commits a CONSTRUCTED
     table instead of its mined one, crossing amount x truth x arrival:
       strip      own mined, junk removed          (truth up, amount down)
       complete   own mined U all missing true     (amount up, truth up)
       exact      the full true table              (complete content, commit-time arrival)
       junk_dose  own mined U K realistic junk     (amount up, truth down); K matched to
                  `complete`'s addition count, which at L2 is exact because the two arms are
                  still twins there, and at L3 is per-arm because the L2 surgery has already
                  separated them — both realized counts are logged.
     Everything else is the anchor's: policy, pricing, schedule, streams. The surgery runs
     BEFORE the commit body, so the priced pre-commit audition grades what is actually
     committed and an emptied table cancels the commit through the donor's existing guard.
     The mined table's own grade is recorded beside the surgical one, and the shadow
     certificate still records what the certificate WOULD have done, so the counterfactual
     survives the treatment rather than being overwritten by it.

  2. PER-EXECUTION ENTRY IDENTITY — the instrument the census forensics found structurally
     uncomputable. `macros.macro_features` ends at `best = cur.argmax(-1)`: the table entry the
     max-sum DP selects for each row of each macro execution. That value is discarded by the
     donor. `_install_entry_recorder` swaps in a byte-identical copy of the donor's function
     that additionally accumulates a per-entry histogram on-device, so the choice is logged
     without a host sync, without touching `counts` (unpriced), and without consuming any RNG.
     `entry_recorder_check()` asserts the copy reproduces the donor's `(feats, pos)` bit for
     bit, and `preflight` asserts a treated arm stays bit-identical to its twin up to the first
     surgery WITH the instrument on.

  3. THE REALISTIC JUNK POOL. `junk_dose`'s junk is not uniform random tuples: it is drawn from
     tuples actually mined-but-false in `census/cs_s0` and `spiral/sp_s0`, read off those runs'
     `log["miner"][...]["keys_at_support"]` on the volume at setup and keyed against
     `MC.true_tables`. The pool and the draw are logged verbatim in `setup.json`.

  4. The surgery tables themselves are logged verbatim (entries, flat tuples, truth split) on
     every surgery event, so the treatment is auditable after the fact rather than trusted.

--- the donor's own header follows, unmodified ---

census — can an endogenous gauge buy the coverage the certificate cannot see?

FORK NOTICE. This file forks `../spiral/spiral.py` VERBATIM and adds four things, each marked
`# [census]` at its insertion point. `spiral/` and everything upstream of it are NOT modified.

  1. THE G-Y INSTRUMENT — an unpriced, read-only level-4 miner in EVERY arm, decoupled from
     `max_macro_level` (which caps what may be COMMITTED, not what may be OBSERVED). Phase 0
     established this stream is not in `sp_s0`'s logs at all and cannot be reconstructed from
     them: no level-4 miner is ever built, and per-key counts carry no adjacency. Level 4 stays
     unearnable (1,024 entries, ~384 cycles to cover) so this is a growth-DIRECTION readout and
     never a commit candidate. `gy_soundness()` checks it against hand-enumerated counts.

  2. COMMIT-THEN-EXTEND (`census_extend`) — the recert loop widened to every COMMITTED level in
     every era, and given the power to ADD post-commit mined entries to a frozen table instead
     of only offering whole-table swaps (which never fired in `sp_s0`: 0 swaps in 26 recerts).
     Entries enter by SELECTION, not append: each candidate is auditioned on fresh held-out
     instances of the era's own cell, priced, and admitted only if it does not raise the
     table's audition error. Phase 0's reason for the widening: the inherited loop fires only
     for `rc_level = era["level"]`, so it runs in eras 2-3 and **not at all in eras 4-5** —
     which are exactly where this round's payoff is measured.

  3. THE L2 GATE (`census_gate`) — a CONJUNCTION rule at level 2: commit when the certificate
     has fired AND the admission-rate gauge is quiet (G-A on the at-support series, W=12,
     theta=0, Phase 0 R5). A conjunction, not a replacement, so the op is strictly "hold longer
     than the certificate", `yoked_delay` is exactly interpretable as delay-alone, and no
     setting can accidentally ACCELERATE the commit. The gate is at L2 and not L3 because
     Phase 0 measured that a frozen L2 of recall r caps buildable L3 coverage at about r**2
     (0.327 in the routed arms), while L2 coverage on offer is +0.357 recall.

  4. `yoked_delay` — commits at the cycle `census_gate` actually committed, by clock rather
     than by signal. The cycle is not hard-coded: `census_run` runs `census_gate` first and
     passes its measured L2 commit cycle in, recording which case occurred (gate fired, or
     gate fell back to the era boundary).

Everything else is `spiral.py`'s, unchanged — which is what `fidelity_smoke` and the in-tag
`spiral_route` anchor assert.

--- the donor's own header follows, unmodified ---

spiral — the LIVE RE-EARNING SPIRAL on the depth-6 substrate.

FORK NOTICE. This file forks `../native/full/full.py` VERBATIM and then adds three things,
each marked `# [spiral]` at its insertion point. `native/`, `tall/`, `ear/`, `recital/` and
`ratchet/` are NOT modified; nothing is copied out of them except where a comment says so.

  1. THE DEPTH-6 TRANSPLANT (`tall/tall.py`'s lessons, not its code). `_spiral_shared` is
     `tall`'s `_tall_shared` idiom applied to THIS file's own `build_shared`: it populates
     `probe_clean` — the key every round's ENTRYPOINT populates and no builder does, which
     cost `tall` a ~510 s setup to discover — and then ASSERTS the complete key set that
     `run_arm` / `measure_refs` / `plant_probe` read, so interface drift fails in seconds
     instead of after a paid setup. `preflight` calls every one of those at toy sizes for the
     same reason. The grammar itself is a cfg substitution (`depth=6, m=2`), so with the
     transplant OFF this file is depth-4 `full.py` — which is what `fidelity_d4` gates.

  2. THE COMMIT POLICY, transplanted from `ear/ear.py` (and carried by `recital/recital.py`):
     CERTIFY-ELSE-PROVISIONAL-AT-BOUNDARY (`commit="delta_prov"`) plus the LIVE RECERT.
     `full.py`'s run loop is ratchet-lineage with a fixed in-tag commit schedule
     (`delta` / `early` / `late`), so the boundary-provisional policy is genuinely absent here
     and had to be written in, not switched on. A provisional commit deliberately skips the
     priced pre-commit audition — you pay to grade the unit where it is CONSUMED instead — and
     the recert grades the frozen unit on the next era's cell against what the live table
     would score. Both branches are additive: an arm whose `commit` is `delta`/`early`/`late`
     and whose spec has no `recert` key takes exactly `full.py`'s path, byte for byte.

  3. THE SPIRAL ARMS and the Phase-A instruments: the descent gate WITH ROUTING LIVE
     (`n_corrupt=1`, `tall/dens0`'s measured lever, folded into `build_shared` rather than
     bolted on afterwards), the commit-cycle PRICING READOUT (does `dens0`'s +0.180
     commit penalty survive when width refits under effective branching k instead of under
     |action set|?), and `tall`'s per-cycle entry-identity miner instrument.

WHY THE ROUND EXISTS: earn level k live from the beam's own chosen trajectories, consolidate
it at commit (pi's action space grows; span slots open behind the per-macro parity gate), then
earn level k+1 natively over the routed policy — and ask whether the crank accelerates, holds,
or decays per turn. See `SPEC.md` in this folder.

Run from experiments/:
  modal run rhm/practice/spiral/spiral.py::gates_remote          # T-1..T-4 + P/C at depth 6
  modal run rhm/practice/spiral/spiral.py::preflight             # interface, ~2 min
  modal run rhm/practice/spiral/spiral.py::fidelity_d4           # G-F: the fork did not drift
  python3 rhm/practice/spiral/launch_detached.py --fn phase_a --tag pa0 ...

--- the donor's own header follows, unmodified ---

full — PHASE 2 of `native`: the two ports COMPOSED, plus the ablation battery.

FORK NOTICE, and the choice behind it. This file forks `../prop/prop.py` (Port 1), which is
itself a verbatim fork of `../../ratchet/ratchet.py`, and grafts Port 2 onto it by IMPORTING
`../span/span_net.py` — nothing is copied from `span/`, and `ratchet/`, `teacher_slot/`,
`prop/` and `span/` are all left untouched. The alternative was to re-fork `ratchet.py` and
import both nets; forking `prop.py` was chosen because Port 1's insertions are the larger and
more delicate of the two (a rewritten beam with a cached-encoder-state contract and a fidelity
gate that has already been verified bit-for-bit in-tag), while Port 2's are a clean seam — one
executor object threaded through the beam's materialisation call. Re-deriving Port 1 by hand
would have put the already-gated half at risk to save nothing.

WHAT IS NEW relative to `prop.py`:

 * The executor seam. `beam_moves` and `beam_moves_prop` both materialise through an executor
   object (`span_net.PlainExecutor` / `SpanExecutor`) instead of calling `macros.apply_any`
   directly. A `PlainExecutor` call IS `apply_any` (span/'s gate S-3), so an untreated arm is
   op-for-op ratchet and Port 1's beam is op-for-op `prop.py`'s.
 * `finetune_generator_span` — the plant loss plus the span head's self-imitation term in the
   same optimizer steps, with a `lam == 0` short-circuit that restores the original graph
   exactly (this is what makes the composed fidelity arm a real gate).
 * The NATIVE arms: proposal head (k = 4, Port 1's measured sweet spot) AND span head on the
   same vocabulary — `given_native`, `practice_late_native`.
 * The POISON PAIR: `practice_early` (a frozen 1-entry bogus level-2 table, exogenously
   enumerated) against `practice_early_prop_k4` (the same bad table, natively ROUTED).
 * `ablation_battery` — an end-of-run probe battery on the final trained state, per arm:
   (a) full system, (b) table ablated with macro calls forced through the span head,
   (b') table and span reach both removed, (c) primitives ablated. Plus, under each, the
   next-level mining yield split into its STRUCTURAL component (entries `Miner.build` can
   assemble over the surviving lower table) and its INFORMATIVE one (distinct level-1 tuples
   observed at support in the chosen trajectories, which no lower table gates).

The fork's fidelity is not assumed: the untreated arms are re-run IN THIS TAG as the twins,
and `given_fid` (k = n_moves, span loss off, parity gate nailed shut) is carried as an in-tag
assertion that the composed loop reproduces enumeration bit-for-bit for 90 cycles.

WHAT IS NEW (everything else is ratchet's, unchanged):

 * `prop_net.ProposalHead` — pi(move | z, r*) beside the value, on the SAME encoder read
   (`controller.state(x)` plus the root target), over a fixed (level, node) slot vocabulary.
 * `beam_moves_prop` — at plan time only the top-k proposed moves per tip are materialised and
   scored, instead of all `n_moves`. Everything else about the beam is `beam_moves` verbatim.
   The kept children's encoder states are CACHED and reused as the next step's proposal read,
   because the value already paid for them when it scored those children.
 * `prop_train` — self-imitation on the beam's own chosen trajectories: (state, r*) -> the move
   the surviving-and-SOLVED tip actually took, masked cross-entropy over the live action set.
   Selection happens before the regression, which is the whole reason this is not "gradient
   descent averaging over everything the agent did".
 * Arms `*_prop_k{1,2,4,N}` over three vocabularies (base / true / mined), and the readouts the
   spec commits Port 1 to: the can't-decompose signature, the effective-branching ledger, and
   the next-level currency under native vs exogenous routing.

WHAT IS PRICED, SAID OUT LOUD. A grounding in this substrate is "score one state" — an encoder
pass plus a value read (`beam_moves` charges one per materialised child). The proposal head
reads the SAME z. For every tip below the root that z was already computed and charged when the
value scored that state as a child, so the port caches it and the ledger charges nothing; the
head-only forward on a cached vector is counted separately as `n_prop` and priced at `c_prop`
(default 0.0, with a sensitivity column at c_mat = 0.05 in the reduction). The ROOT state is
the one honest exception — it has never been anybody's child — so the port pays one full
grounding per instance per solve for it, and `fit_width_k` sees that cost when it picks the
width. The freed groundings are REINVESTED, not banked: each arm runs the widest beam that fits
the same declared G = 58 under ITS effective branching factor, which is exactly ratchet's
matched-pricing idiom with n_moves -> k. The banked reading (same width, fewer groundings) is
recoverable from the width ladder logged at every probe.

Run from experiments/:
  modal run rhm/practice/native/prop/prop.py::prop_run --quick --tag smoke0
  python3 rhm/practice/native/prop/launch_detached.py --fn prop_run --tag np_s0 ...

--- ratchet's own header follows ---

Round 2 of the practice arc on RHM. Round 1 (`../crystallize/`) found the compile certificate
VACUOUS on a frozen plant: practice trained the judge (the value) while the generator that
executes a committed unit never moved, so the committable content was flat from cycle 1 and
committing at cycle 1 was optimal at 26x less priced time. Its precheck (`calp_s0`) validated
the fix — let the plant learn on the agent's own successful repairs and the committed-unit
ceiling moves at 31-66x the metering noise floor.

WHAT THIS ROUND COMPOSES.

 1. A LEARNING PLANT, carried by every arm. The generator fine-tunes online on configurations
    the agent actually solved (replay 0.5 against the clean setup pool), and the value adapts
    online against it — round 1's precheck confound (its wide beam degraded because the value
    was trained against the old generator) fixed by construction.

 2. A DEPTH-LADDERED DAMAGE SCHEDULE, the `level_ladder` idiom: the damage cell moves one level
    deeper per era, through NESTED cells (L1 n6 subset L2 n3 subset L3 n1), so era k+1's error
    is literally era k's error one level up. At a declared per-solve grounding budget, a level-3
    error is four blocks wide and a base-move agent must spend its whole move budget on it with
    no coordination left over — climbing is necessary, not merely available.

 3. AN EARNED VOCABULARY (`macros.py`). The committed unit is not a whole-solve program (round
    1's unit) but a MACRO ACTION: the same max-sum operator the true level move uses, over a
    table MINED from the agent's own solved configurations and parsed by the agent's own
    generator. Committing it puts a new primitive in the practice action space — the beam
    proposes it, the value scores it, the plant's fine-tuning sees trajectories that use it.
    The level-l table is defined over level-(l-1) ENTRIES, so an incomplete committed level-2
    vocabulary is a hard cap on level 3: the poisoned-region test with a mechanism.

 4. A UNIT-LP CERTIFICATE. Round 1's central lesson, in Jasper's constructive form: certify on
    the LEARNING PROGRESS OF THE COMMITTABLE CONTENT — the shadow-audition trajectory of the
    candidate macro, positive and then flat — never on task-performance level or task-LP, which
    conflate selector and plant improvement. "Positive and then flat" is enforced: silence alone
    cannot certify unless the audition has first descended by `lp_min_drop`.

ARMS differ ONLY in vocabulary-acquisition policy:
    never_base     base level-1 moves forever; must buy depth with search at the declared budget
    given          the DGP's own level-2/3 tables from cycle 1 (the `level_moves` ceiling)
    practice_gated earns them, gated by the unit-LP certificate
    practice_early commits one cycle into each era (minimal coverage) -- the poison test
    practice_late  commits at each era's end (the rent-paying control)

Run from experiments/:
  modal run rhm/practice/native/full/full.py::full_selfcheck_remote
  modal run rhm/practice/native/full/full.py::full_run --quick --tag smoke0
  python3 rhm/practice/native/full/launch_detached.py --fn full_run --tag nf_s0 ...

(ratchet's own run lines: `rhm/practice/ratchet/ratchet.py::ratchet`.)
"""

import collections     # [tutti] the port's key/half-key histograms
import copy
import inspect          # [conductor] gate N-1 reads the endo read's own source
import json
import os
import time

import modal
import numpy as np

from rhm.rhm_data import build_inverse_maps, generate_rules_distinct
from rhm.shared import DATA_DIR, NumpyEncoder, image, volume
from rhm.rhm_sculpt_precheck import nearest_derivation_cost
from rhm.rhm_edit_control import _train_edit_controller
from rhm.rhm_generative_planner import _build_generator, _train_generator
from rhm.rhm_latent_planner import _build_value_head
from rhm.rhm_sculpt_planner import _corrupt, _encode_chunked, _sample_pool
from rhm.rhm_sculpt_latent import _build_rich_controller
from rhm.practice.crystallize.units import (
    apply_move, build_move_set, corrupt_hier, grade, on_grammar_rate, oracle_rollout)
from rhm.practice.ratchet import macros as MC
# [tutti] the QUESTION PORT's selectors and its difficulty quota, IMPORTED from
# `antiphon/questions.py` and not forked -- exactly as `maestro/policy.py` is imported
# and not forked. The selector rule did not change; the loop did. Gates Q-1..Q-10 in
# `questions.py::question_gate` therefore apply here unchanged and are re-run for this node.
from rhm.practice.antiphon import questions as QS
from rhm.practice.native.prop import prop_net as PN
from rhm.practice.native.span import span_net as SN
# [conductor] the outer loop lives outside the Modal app, so every rule is auditable and
# gate-able with no GPU and no substrate (the `teacher_slot/decision` convention).
# [maestro] the FORKED loop: A1's classes byte-for-byte plus `LearnedPolicy`.
from rhm.practice.maestro import policy as PO


# [tutti] a new app and a new remote root; the donors' tags are never written to.
app = modal.App("rhm-practice-tutti", image=image)          # [tutti]
REMOTE = "rhm_practice_tutti"                                # [tutti]

# [spiral] the depth-6 substrate, exactly as `tall/` measured it admissible. m=2 is NOT a
# choice: m>=3 flattens the depth ladder (gradient 1.06-1.30x against m=2's 3.25x) and walls
# off every macro level above 3. The nested ladder's node indices are `25 // 2**(l-1)`.
SPIRAL_ERAS = "1:25,2:12,3:6,4:3,5:1"
DEPTH6 = dict(depth=6, m=2, v=8, s=2)
# The keys `run_arm` / `measure_refs` / `plant_probe` read out of the shared dict. Asserted by
# `_spiral_shared` so drift fails in seconds rather than after a ~510 s setup (`tall`'s lesson).
SHARED_KEYS = ("base_ms", "bottom_map", "canon", "controller", "generator0", "hold",
               "inverse_maps", "leaves_pool", "length", "n_blocks", "probe_clean", "read_acc",
               "reader", "replay", "roots_pool", "rules", "rules_t", "truth", "value0")

# [maestro] THE FITTED POLICIES, from `fit.py` — derived offline, before any GPU, on the
# four DISTINCT trajectories in `cd_s0` (257 windows; the two yoked arms excluded as
# bit-identical replays). Copied here as literals rather than read from `fit.json` so the
# policy that governed a run is in the run's own `setup.json`, and CHECKED against the
# file by `fit_gate()` so the two cannot drift apart — `../conductor/floors.py`'s pattern.
#
#   mixes[bucket]  a unit-norm mixture over (ledger, yield, endo_excess) paired-interval
#                  slopes IN FLOOR UNITS; bucket = will_commit, i.e. whether a firing
#                  would INSTALL A TABLE (1) or MOVE THE WORLD (0).
#   thetas[bucket] that mixture's own MEASURED dead zone, `sd(Nhat)/2` by the donor's
#                  null-ABBA method on the fixed-condition reference arm.
#
# A1's thermostat is the point mixes = (0, 1, 0) with theta = 1 in this same class, which
# is what `policy.policy_gate()` L-1 asserts and what makes the comparison one-class.
# [maestro] the NORMALISING UNITS the mixture was fitted in. A learned weight only means
# something relative to the floor its gauge was divided by, so the run asserts these are the
# floors it is actually governed by (`fit_gate`'s companion assertion).
PO_FIT_UNITS = {"ledger": 0.02944712566990095,
                "endo_excess": 0.01091575129919287,
                "yield_by_level": {3: 0.46127129019246205, 4: 0.5166900731510206}}

FITTED = {
    "yield": {
        "reward": "yield",
        "fit_id": "maestro-fit-yield",
        "gauges": [
            "ledger",
            "yield",
            "endo_excess"
        ],
        "buckets": [
            "1",
            "0"
        ],
        "mixes": {
            "1": [
                0.21060533596693154,
                0.944006380431357,
                0.25396327720192946
            ],
            "0": [
                0.04946387529238211,
                0.8505980361855522,
                0.5234847713910517
            ]
        },
        "thetas": {
            "1": 1.208500542604683,
            "0": 1.0979866716790605
        }
    },
    "task": {
        "reward": "task",
        "fit_id": "maestro-fit-task",
        "gauges": [
            "ledger",
            "yield",
            "endo_excess"
        ],
        "buckets": [
            "1",
            "0"
        ],
        "mixes": {
            "1": [
                -0.7805776968216638,
                0.21455449152515405,
                0.5870816207231914
            ],
            "0": [
                -0.4221446205589885,
                0.1366584639685819,
                0.8961687249390349
            ]
        },
        "thetas": {
            "1": 1.4253060718406116,
            "0": 0.7083064560849142
        }
    }
}


# [conductor] THE MEASURED DEAD ZONES, from `floors.py` — derived offline, before any GPU, by
# the null-ABBA method on fixed-condition series in `as_s0` (4 arms), `cs_s0` (2 arms) and
# `as_s1` (1 arm): 574 pooled windows per series, span 1, W 4, era boundaries and commit cycles
# dropped. `mean(N)` came out at -0.003 / -0.003 / +0.021 against spreads of 0.059 / 0.474 /
# 0.732, which is the null construction checking itself. Copied here as literals rather than
# read from `floors.json` so the value that governed a run is in the run's own `setup.json`,
# and CHECKED against the file by `floor_gate()` so the two cannot drift apart.
#
#   ledger    0.0294 error/cycle    the arm's own metering error on the era's own cell
#   yield L3  0.2368 tuples/cycle   distinct level-3 tuples at support
#   yield L4  0.3661 tuples/cycle   distinct level-4 tuples at support (the G-Y instrument)
#   endo      no offline series exists — it needs forward passes through a live plant. Measured
#             on the smoke tag's SCHEDULE arm (fixed-condition by construction) and passed in on
#             the command line; the reduction re-derives it in-tag on this run's own `anchor`,
#             exactly as `endo_yield` re-derived `xspan`'s on `no_wall`.
MEASURED_FLOORS = {
    # GOVERNING: re-derived at FULL CONFIG on `cd_ef/anchor` — a 116-cycle schedule arm, which
    # is a fixed-condition series by construction and carries the ACTUAL instruments this round
    # drives (the observation panel included). The offline derivation below is the independent
    # pre-GPU cross-check, and on the one series both can measure it agrees to 0.09%:
    # ledger 0.029447 in-tag vs 0.029421 offline.
    #
    # The yield floors MOVED (L3 0.2368 -> 0.4613) and had to: the offline L3 series came from
    # `as_s0`, where level-3 at-support is identically zero through all 48 cycles of era 1
    # because the committable miners are era-gated. The observation panel makes that series live
    # (0 -> -41 over era 1), so it moves faster and its floor is larger. A floor must be measured
    # on the series the rule actually thresholds.
    "ledger": 0.02944712566990095,
    "yield_by_level": {3: 0.46127129019246205, 4: 0.5166900731510206},
    "endo": 0.01091575129919287,   # the EXCESS read; see the panel's `endo_excess` note
    "provenance": ("cd_ef/anchor, full config, 116 cycles, null-ABBA span 1 W 4, era "
                   "boundaries and commit cycles dropped; cross-checked against floors.py's "
                   "offline derivation on as_s0+cs_s0+as_s1 (574 windows/series)"),
}


# --------------------------------------------------------------------------- #
# the depth ladder
# --------------------------------------------------------------------------- #

def parse_eras(spec):
    """"1:6,2:3,3:1" -> [{level, node, name}] — the nested damage cells, one per era.

    [spiral] An OPTIONAL third field gives that era's own cycle count:
    `"1:25:28,2:12:44,3:6:12,4:3:9,5:1:7"`. The donor's loop runs every era for the same
    `cfg["era_cycles"]`, which at depth 6 is wrong in both directions at once — era 2 has to
    carry L3 mining coverage (~24 cycles at `mine_cap=8` by `tall`'s coverage law) while eras
    3–5 are pure consumption and need only a stable cost-to-depth read. Two fields keep the
    donor's behaviour exactly (no `cycles` key -> `cfg["era_cycles"]`), so the depth-4
    fidelity gate is untouched."""
    out = []
    for part in spec.split(","):
        bits = part.split(":")
        lv, node = int(bits[0]), int(bits[1])
        era = {"level": lv, "node": node, "name": f"L{lv}n{node}"}
        if len(bits) > 2 and bits[2]:
            era["cycles"] = int(bits[2])
        out.append(era)
    return out


def era_ctx(era):
    return {"name": era["name"], "level": era["level"], "nodes": [era["node"]]}


def era_l2_nodes(era, s):
    """The level-2 nodes an era's damage cell spans — where a withheld level-2 tuple has to
    be produced for that era's instance to be repairable."""
    span = s ** (era["level"] - 1)
    b0 = era["node"] * span
    return sorted({b // s for b in range(b0, b0 + max(span, 1))} if span >= s
                  else {b0 // s})


def needs_excluded(clean_np, era, hold, inverse_bottom, v, s):
    """Which instances REQUIRE a withheld level-2 tuple: the clean derivation the damage was
    applied to uses one at the damage cell. The coordinator's constraint-1 gate is read on
    this split — a plant frontier is only a frontier if bootstrap success on it is
    low-but-nonzero; at ~0 the design collapses into exploration-gating."""
    if not hold:
        return np.zeros(clean_np.shape[0], bool)
    tup = level2_tuples(clean_np, inverse_bottom, v, s)
    nodes = era_l2_nodes(era, s)
    bad = np.zeros(clean_np.shape[0], bool)
    for t in hold:
        for nd in nodes:
            bad |= (tup[:, nd, :] == np.array(t)).all(-1)
    return bad


def context_instances(rules, ctx, n, s, depth, v, m, seed, require_broken=True,
                      with_clean=False):
    """n fresh instances of a damage cell (crystallize's, verbatim): a clean derivation of a
    random r*, the cell hierarchically damaged, rejection-sampled on d* > 0."""
    length = s ** depth
    roots = np.zeros(n, np.int64)
    x = np.zeros((n, length), np.int64)
    clean = np.zeros((n, length), np.int64)
    rng = np.random.default_rng(seed + 90_001)
    need = np.arange(n)
    for attempt in range(64):
        if len(need) == 0:
            break
        r, lv = _sample_pool(rules, len(need), s, seed + 7919 * attempt)
        xd = corrupt_hier(lv, rules, depth, v, m, s, ctx["level"], ctx["nodes"], rng)
        ok = (nearest_derivation_cost(rules, xd, r, s) > 0) if require_broken \
            else np.ones(len(need), bool)
        roots[need[ok]] = r[ok]
        x[need[ok]] = xd[ok]
        clean[need[ok]] = lv[ok]
        need = need[~ok]
    if len(need):
        raise RuntimeError(f"context {ctx['name']}: {len(need)} instances never broke")
    return (roots, x, clean) if with_clean else (roots, x)


# --------------------------------------------------------------------------- #
# [tutti] THE QUESTION PORT — the menu, and the arm's selection from it
# --------------------------------------------------------------------------- #

def pose_questions(mode, *, rules, era, cfg, s, depth, v, m, cyc, head, shared, value_net,
                   controller, device, miners, operative, maxl, qledger, n_pr, k_menu):
    """Build the cycle's MENU and let the arm's selector pick its `n_pr` questions.

    `head` is the donor's own draw, already made, at the donor's own RNG position — so the
    menu's first `n_pr` entries ARE the donor's questions and `mode="exo"` returns them
    unchanged, in order. The tail is drawn on a disjoint seed through `context_instances`'s own
    `default_rng`, which touches no global stream. Returns the selected `(roots, x, clean)`, the
    per-instance DESIGNED key and half-key (logged for every arm — the dose instrument), and the
    cycle's question row.

    Nothing here is priced or gated; the selection is an ACTION, and the beam that follows is
    identical in size, budget and difficulty mix whatever this returns.
    """
    import torch
    r_h, x_h, c_h = head
    n_tail = max(0, int(k_menu) - int(n_pr))
    if n_tail:
        r_t, x_t, c_t = context_instances(
            rules, era_ctx(era), n_tail, s, depth, v, m,
            seed=int(cfg["seed"]) + 500_000_000 + 10_000 * int(cyc), with_clean=True)
        r_all = np.concatenate([r_h, r_t])
        x_all = np.concatenate([x_h, x_t])
        c_all = np.concatenate([c_h, c_t])
    else:
        r_all, x_all, c_all = r_h, x_h, c_h

    # the difficulty alphabet, and THE QUOTA — the donor's own realised d* mix, which every
    # pinned arm must reproduce bin-for-bin. `nearest_derivation_cost` is the same exact DP the
    # donor already runs for `require_broken`; it is world machinery, not an agent read.
    d = nearest_derivation_cost(rules, x_all, r_all, s)
    quota = QS.quota_of(d, n_pr)
    geom = QS.target_geometry(era["level"], era["node"], maxl, s)
    tgt = geom["level"]
    support = int(cfg["mine_support"])

    # --- the DESIGNED key of every candidate: the EXACT level-1 features of its CLEAN
    #     derivation over the mined span. Logged for every arm (the dose instrument); handed to
    #     a selector only in oracle mode.
    designed = QS.keys_at(MC.exact_features(c_all, shared["inverse_maps"][-1], v, s), geom)

    # --- the selection compute, taken by EVERY arm and used only by the arms whose rule reads
    #     it. Both are no-grad forwards on dropout-free nets: no RNG, no gradient.
    x_dev = torch.from_numpy(x_all).to(device)
    with torch.no_grad():
        pf = MC.parse_features(shared["reader"], x_dev, s=s).cpu().numpy()
        value_net.eval()
        controller.eval()
        vs = torch.sigmoid(value_net(_encode_chunked(controller, x_dev),
                                     torch.from_numpy(r_all).to(device))).reshape(-1)
    vs = vs.detach().cpu().numpy()
    halves = QS.half_keys(pf, geom)

    counts = dict(miners[tgt].counts) if tgt in miners else {}
    banked = {k for k, c in counts.items() if c >= support}
    oracle = agent = None
    if mode == "bisect":
        lower = operative(tgt - 1)
        oracle = {"keys": [tuple(int(z) for z in r) for r in designed],
                  "truth_flat": {tuple(int(z) for z in r)
                                 for r in shared["truth"][tgt]["flat"]},
                  "counts": counts, "support": support,
                  "lower_flat": {tuple(int(z) for z in r) for r in lower["flat"]}}
    elif mode != "exo":
        off = geom["clean_offsets"]
        cov = (collections.Counter(tuple(int(k[o]) for o in off) for k in banked)
               if off else collections.Counter())
        agent = {"halves": [tuple(int(z) for z in h) for h in halves],
                 "covered": dict(cov), "value": vs, "ledger": qledger}

    sel = QS.select(mode, n_pr, d=d, quota=quota, oracle=oracle, agent=agent)
    qok, got = QS.check_quota(d, sel, quota)
    truth_flat = {tuple(int(z) for z in r) for r in shared["truth"][tgt]["flat"]}
    dsel = [tuple(int(z) for z in r) for r in designed[sel]]
    qrow = {
        "cycle": int(cyc), "mode": mode, "k_menu": int(len(d)), "n_pr": int(n_pr),
        "target_level": int(tgt), "target_node": int(geom["node"]),
        "has_clean": bool(geom["has_clean"]),
        "quota": {str(k): int(x) for k, x in quota.items()},
        "d_hist": {str(k): int(x) for k, x in got.items()},
        "quota_ok": bool(qok),
        "d_mean": float(np.mean(np.asarray(d)[sel])),
        "d_mean_menu": float(np.mean(np.asarray(d))),
        # what the QUESTIONS aimed at, by the oracle's own accounting — computed for every arm,
        # so `endo`/`novel`/`comp` are scored on the same axis they never get to see.
        "n_distinct_designed": int(len({k for k in dsel})),
        "n_designed_true": int(sum(1 for k in dsel if k in truth_flat)),
        "n_designed_banked": int(sum(1 for k in dsel if k in banked)),
        # the BREADTH statistic the selector actually optimises: how many DISTINCT true keys
        # that are not yet at support this cycle's 64 questions aim at. `mine_cap` keeps only
        # 8 of the answers, so breadth is what converts into coverage, not instance count.
        "n_distinct_needy": int(len({k for k in dsel
                                     if k in truth_flat and k not in banked})),
        "n_distinct_halves": int(len({tuple(int(z) for z in h) for h in halves[sel]})),
        "at_support_before": int(len(banked)),
    }
    return (r_all[sel], x_all[sel], c_all[sel], designed[sel], halves[sel], qrow)



# --------------------------------------------------------------------------- #
# pricing: the declared per-solve grounding budget picks the beam width
# --------------------------------------------------------------------------- #

def beam_ground(n_moves, budget, w):
    total, width = 0, 1
    for _ in range(budget):
        total += width * n_moves
        width = min(w, width * n_moves)
    return total + width


def fit_width(n_moves, budget, g_budget):
    """The widest beam that fits the declared budget — round 1's performance idiom, now
    arm-dependent because arms carry different numbers of moves and each materialisation is
    a grounding. This is what MATCHED PRICING means here: a level move costs the same
    groundings whoever holds it."""
    best = 1
    for w in range(1, 129):
        if beam_ground(n_moves, budget, w) <= g_budget:
            best = w
        else:
            break
    return best


# --------------------------------------------------------------------------- #
# [intonation] THE FALLIBLE EXECUTOR AND delta_perf
# --------------------------------------------------------------------------- #
#
# `SN.SpanExecutor` is imported, not copied; `PerfExecutor` SUBCLASSES it, so `native/span/` is
# untouched and gates S-1..S-6 apply to the parent unchanged. Two things are added and nothing
# is removed:
#
#   (1) THE FIRING THRESHOLD IS LOWERED, NOT REMOVED. `span_tau_fire` (0.50) replaces
#       `span_tau` (0.95) in the OPEN decision only; `span_min_hold` is untouched, the parity
#       series is still measured and logged at full resolution, and the reduction can read off
#       what the tau = 0.95 gate would have done. The point is a FALLIBLE executor, not a
#       vandalised one: `native/` finding 4 measured L3 heads ending at exact-match parity
#       0.76-0.94, i.e. wrong on 6-24% of calls — real error, and the majority of executions
#       still correct — while `native/` finding 5's UNTRAINED-head control (parity ~0) pays
#       +0.40-0.51 e, which is what vandalism looks like. 0.50 sits between them.
#
#   (2) THE METER. On every row the head actually executes, all four bridge quantities are
#       computed from tensors the firing path already has:
#
#         intention   tgt  = SN.dp_features(logits, move, s)   -- what `apply_any` writes.
#                     `trunk` returns (pooled, logits) and the donor discards `logits`; the DP
#                     is a pure function of them, so the EXACT reference is free. This is why
#                     no learned forward model appears anywhere in this node.
#         realization got  = head.emit(...)                    -- what the head wrote.
#         e           1 - mean(got == tgt) over the span's blocks (graded Hamming; the
#                     alphabet is unordered categorical, so Hamming is the metric. The
#                     exact-match BIT the parity gate uses is `e == 0`, logged beside it).
#         g           a_exec * a_gap, both in [0, 1]:
#                       a_exec  1 iff the head's emission became the realization (an efference
#                               copy exists for THIS output). 0 on base moves and on
#                               DP-executed macros -- this substrate's PLAYBACK condition,
#                               identical in every other respect.
#                       a_gap   mean(got != argmax(logits over the span)) -- the arity gap:
#                               how much the SLOT COMMAND explains over the slot-free
#                               per-block prediction. Zero when the macro call bought nothing
#                               the base infill would not have written, which is the "an
#                               efference copy of doing nothing grants no agency" case
#                               (S13(a): agency came out GRADED, not binary).
#         delta       (b(slot) - e) * sigmoid((g - g0)/theta), the gate CENTERED.
#
#       `b(slot)` is a per-macro-slot EWMA of e at rate `perf_alpha` -- Gadagkar's
#       per-syllable benchmark, and NOT the global scalar (S13(b) measured the scalar form
#       fixating worse than raw error). It is read BEFORE it is updated, so delta is never a
#       residual against a benchmark that has already seen this row.
#
# Everything here is unpriced instrument: no grounding is charged, and the executor's own
# dedicated numpy stream is drawn from exactly once per macro call, as in the donor, so the
# shared per-arm torch stream is untouched and the twin gate stays licensed.


def perf_gate_sigma(g, g0, theta):
    """The CENTERED agency gate. sigma(0) = 0.5, so the literal sigma(g/theta) leaks half the
    channel when g = 0 (S13(a)'s correction, measured on the playback control)."""
    return 1.0 / (1.0 + np.exp(-np.clip((g - g0) / max(theta, 1e-6), -30.0, 30.0)))


class PerfMeter:
    """Per-slot benchmark b(s), the gate's calibration, and the per-cycle accumulators.

    Lives beside the executor rather than in it so that a yoked arm's meter state is a plain
    picklable object the log can carry, and so the reduction can recompute delta offline at a
    different `perf_alpha` from the per-cycle sums (S13(c): the benchmark timescale has an
    INTERIOR optimum, above sampling jitter and below competence drift, and this substrate has
    never measured where that is -- so the series that lets it be read off is logged rather
    than the choice being defended)."""

    def __init__(self, alpha=0.05, g0=None, theta=None, calib_min=2048):
        self.alpha = float(alpha)
        self.bench = {}                 # slot_key -> b(s)
        self.g0, self.theta = g0, theta
        self.calib_min = int(calib_min)
        self.cal_active, self.cal_passive = [], []
        self.calibrated = g0 is not None and theta is not None
        self.cal_event = None
        self.reset_cycle()

    def reset_cycle(self):
        self.acc = {}                   # slot_key -> dict of sums
        self.rows = []                  # bounded per-row sample, for the record
        self.cell = {}                  # 2x2 counters, filled by the runner
        self.n_call = 0

    def _cell(self, key):
        if key not in self.acc:
            self.acc[key] = {"n": 0, "n_fire": 0, "sum_e": 0.0, "sum_e2": 0.0, "sum_g": 0.0,
                             "sum_d": 0.0, "sum_bme": 0.0, "sum_b": 0.0, "n_exact": 0,
                             "sum_w": 0.0}
        return self.acc[key]

    def calibrate(self, force=False):
        """g0 = midpoint of the ACTIVE and PASSIVE g medians, theta = their gap / 8 --
        `agency_gate`'s protocol verbatim, self-supervised, no test-set tuning. On this
        substrate the passive class is exactly g == 0 (a base move, or a macro the DP
        executed), so the estimate reduces to g0 = median(g_active)/2."""
        if self.calibrated and not force:
            return False
        if len(self.cal_active) < self.calib_min:
            return False
        m_a = float(np.median(np.asarray(self.cal_active, dtype=np.float64)))
        m_p = (float(np.median(np.asarray(self.cal_passive, dtype=np.float64)))
               if self.cal_passive else 0.0)
        gap = abs(m_a - m_p)
        if gap <= 1e-9:                 # degenerate: the head never leaves the base prediction
            self.g0, self.theta = 0.5 * m_a, max(m_a / 8.0, 1e-3)
        else:
            self.g0, self.theta = 0.5 * (m_a + m_p), gap / 8.0
        self.calibrated = True
        self.cal_event = {"m_active": m_a, "m_passive": m_p, "g0": self.g0,
                          "theta": self.theta, "n_active": len(self.cal_active),
                          "n_passive": len(self.cal_passive), "degenerate": gap <= 1e-9}
        self.cal_active, self.cal_passive = [], []
        return True

    def score(self, key, e, g, fired):
        """e, g: 1-D float arrays over the rows of ONE execution. Returns (delta, b_used).

        Before calibration the gate is held at 1.0 and delta is (b - e) ungated, which is
        logged as such -- the warmup is short (one calibration draw) and pretending to a gate
        that has not been calibrated would be the leak the centering exists to close."""
        b0 = self.bench.get(key)
        if b0 is None:
            b0 = float(np.mean(e))      # a slot's first sight sets its own benchmark: delta ~ 0
            self.bench[key] = b0
        gate = (perf_gate_sigma(g, self.g0, self.theta) if self.calibrated
                else np.ones_like(g))
        delta = (b0 - e) * gate
        self.bench[key] = (1.0 - self.alpha) * b0 + self.alpha * float(np.mean(e))
        if not self.calibrated:
            (self.cal_active if fired else self.cal_passive).extend(
                [float(q) for q in g[:512]])
        c = self._cell(key)
        c["n"] += int(e.shape[0])
        c["n_fire"] += int(e.shape[0]) if fired else 0
        c["sum_e"] += float(e.sum()); c["sum_e2"] += float((e * e).sum())
        c["sum_g"] += float(g.sum()); c["sum_d"] += float(delta.sum())
        c["sum_bme"] += float(((b0 - e)).sum()); c["sum_b"] += b0 * int(e.shape[0])
        c["n_exact"] += int((e <= 1e-12).sum())
        self.n_call += 1
        return delta, b0

    def sample_rows(self, key, e, g, delta, b0, fired, cap, rng):
        """A bounded random subsample of the raw per-row quantities, so the reduction can
        recompute delta at any benchmark timescale and any gate calibration off the record."""
        if len(self.rows) >= cap:
            return
        n = int(e.shape[0])
        take = min(4, n, cap - len(self.rows))
        idx = rng.integers(0, n, size=take) if n > take else np.arange(take)
        for i in idx:
            self.rows.append([key, int(fired), round(float(e[i]), 5), round(float(g[i]), 5),
                              round(float(delta[i]), 5), round(float(b0), 5)])


def _perf_executor(base_cls):
    """Built lazily so the module imports without torch (the donor's own idiom for the head)."""
    import torch

    class PerfExecutor(base_cls):
        """`SN.SpanExecutor` plus the meter. With `meter=None` and `tau_fire=None` this class
        IS its parent: `apply` takes the parent's branch on every call (gate I-1)."""

        kind = "perf"

        def __init__(self, *a, meter=None, row_cap=192, row_seed=0, **kw):
            super().__init__(*a, **kw)
            self.meter = meter
            self.row_cap = int(row_cap)
            self.rrng = np.random.default_rng(int(row_seed))
            self.last = None            # per-row (delta, e, exact) of the most recent apply
            self.cred = {}              # slot_key -> per-row credit, in lockstep with `buf`
            self.counts.update(self._extra())

        @staticmethod
        def _extra():
            # `blk_ref` is what the INTENTION REFERENCE consumed and is never folded into `t`;
            # `n_misfire` is the counterfactual price of a botched span, tallied beside the
            # ledger rather than in it (see the module header, PRICING).
            return {"blk_ref": 0, "n_misfire": 0, "n_fired": 0}

        def reset(self):
            super().reset()
            self.counts.update(self._extra())

        # -- the credit-carrying buffer ------------------------------------------------ #
        def _store(self, obs, key, cred=None):
            """The parent's `_store`, with a per-row credit column kept in lockstep. The rng
            draw, the bijective held-out code and the caps are the parent's exactly."""
            n = obs.shape[0]
            take = min(self.per_call, n)
            idx = torch.from_numpy(self.rng.permutation(n)[:take]).to(obs.device)
            rows = obs.index_select(0, idx).detach().cpu()
            code = ((rows + 1) * self.powers).sum(1) % 1000003
            is_hold = (code * 48271) % 100 < int(round(self.hold_frac * 100))
            if cred is None:
                cv = torch.full((take,), float("nan"))
            else:
                cv = torch.as_tensor(np.asarray(cred, dtype=np.float32))[idx.cpu()]
            for tgt, cap, sel, cbuf in ((self.hold, self.hold_cap, is_hold, None),
                                        (self.buf, self.cap, ~is_hold, self.cred)):
                part = rows[sel]
                if part.shape[0] == 0:
                    continue
                tgt[key] = part if key not in tgt else torch.cat([tgt[key], part])[-cap:]
                if cbuf is not None:
                    cpart = cv[sel]
                    cbuf[key] = cpart if key not in cbuf else \
                        torch.cat([cbuf[key], cpart])[-cap:]

        # -- execution ----------------------------------------------------------------- #
        def _fire_only(self, x, obs, pos, move, info, canon, n0, chunk):
            """The head executes, nothing is metered — the parent's own firing path, with the
            capture kept (a probe's contexts are still contexts the head will be tested on, and
            excluding them would change the held-out split the parity gate reads)."""
            if self.capture:
                self._store(obs, SN.slot_key(move["level"], move["node"]), None)
            span = int(move["span"])
            outs = []
            with torch.no_grad():
                for i in range(0, n0, chunk):
                    ob = obs[i:i + chunk]
                    pooled, _ = SN.trunk(self.core, ob)
                    sid = torch.full((ob.shape[0],), info["id"], dtype=torch.long,
                                     device=x.device)
                    outs.append(self.head.emit(pooled, move["blk0"], span, sid))
            got = torch.cat(outs) if len(outs) > 1 else outs[0]
            self.counts["mat_head"] += n0
            self.counts["blk_head"] += n0
            self.last = None
            new = x.clone()
            new.scatter_(1, pos, canon[got].reshape(n0, -1))
            return new

        def apply(self, generator, x, move, rules_t, canon, depth, v, m, s, chunk=16384):
            if self.meter is None:
                return super().apply(generator, x, move, rules_t, canon, depth, v, m, s,
                                     chunk=chunk)
            n0 = x.shape[0]
            if move.get("kind") != "macro":
                self.last = None
                return super(SN.SpanExecutor, self).apply(generator, x, move, rules_t, canon,
                                                          depth, v, m, s)
            key = SN.slot_key(move["level"], move["node"])
            info = self.slots.get(key)
            pos = SN.span_positions(move, n0, s, x.device)
            obs = x.clone().scatter_(1, pos, torch.full_like(pos, -1))
            fired = bool(self.fire and info is not None and info.get("open"))
            # THE BENCHMARK IS OVER THE AGENT'S OWN PERFORMANCES, not over its instruments.
            # `_ENTRY_REC["phase"]` already splits the priced practice + metering beams
            # ("beam") from every unpriced probe and audition ("probe"), and the probes run on
            # a different instance distribution — a context-conditional b(s) fed from them
            # would be benchmarking the bird against someone else's song. The head still FIRES
            # in probes (the arm's behaviour must not depend on who is watching); only the
            # meter is off there.
            metering = _ENTRY_REC.get("phase") == "beam"
            if fired and not metering:
                return self._fire_only(x, obs, pos, move, info, canon, n0, chunk)
            if not fired:
                # PLAYBACK: the DP realizes the move, so nothing the head emitted became real.
                # No efference copy for the head's output -> g == 0 -> the centered gate closes
                # by mechanism rather than by a hand-written exclusion. The head's would-be
                # error on these contexts is not left unmeasured: the parity read computes it
                # for every held-out row of every slot, open or closed, once per cycle.
                if self.capture and info is not None:
                    self._store(obs, key, None)
                self.last = None
                return super(SN.SpanExecutor, self).apply(generator, x, move, rules_t, canon,
                                                          depth, v, m, s)
            span, b0_ = int(move["span"]), move["blk0"]
            got_l, tgt_l, base_l = [], [], []
            with torch.no_grad():
                for i in range(0, n0, chunk):
                    ob = obs[i:i + chunk]
                    pooled, logits = SN.trunk(self.core, ob)
                    sid = torch.full((ob.shape[0],), info["id"], dtype=torch.long,
                                     device=x.device)
                    got_l.append(self.head.emit(pooled, b0_, span, sid))     # realization
                    tgt_l.append(SN.dp_features(logits, move, s))            # intention (free)
                    base_l.append(logits[:, b0_:b0_ + span, :].argmax(-1))   # arity-1 read
            got = torch.cat(got_l) if len(got_l) > 1 else got_l[0]
            tgt = torch.cat(tgt_l) if len(tgt_l) > 1 else tgt_l[0]
            base = torch.cat(base_l) if len(base_l) > 1 else base_l[0]
            match = (got == tgt)
            e = (1.0 - match.float().mean(1)).cpu().numpy().astype(np.float64)
            g = (got != base).float().mean(1).cpu().numpy().astype(np.float64)
            delta, b0 = self.meter.score(key, e, g, True)
            self.meter.sample_rows(key, e, g, delta, b0, True, self.row_cap, self.rrng)
            if self.capture and info is not None:
                self._store(obs, key, delta)
            self.counts["mat_head"] += n0
            self.counts["blk_head"] += n0
            self.counts["blk_ref"] += n0 * span      # what the INSTRUMENT consumed, never `t`
            self.counts["n_fired"] += n0
            n_bad = int((~match.all(1)).sum())
            self.counts["n_misfire"] += n_bad
            self.last = {"delta": delta, "e": e, "exact": match.all(1).cpu().numpy()}
            new = x.clone()
            new.scatter_(1, pos, canon[got].reshape(n0, -1))
            return new

    return PerfExecutor


_PERF_EXEC = None


def build_perf_executor(*a, **kw):
    global _PERF_EXEC
    if _PERF_EXEC is None:
        _PERF_EXEC = _perf_executor(SN.SpanExecutor)
    return _PERF_EXEC(*a, **kw)


def perf_span_train_terms(core, head, ex, slots, s, batch, rng, device, *, mode,
                          tau_w, w_raw_cap, w_clip, wnorm):
    """(i) THE EFFERENT RETURN: delta_perf as a PER-SAMPLE GAIN on the head's own plasticity.

    `SN.span_train_terms`' loss exactly -- same slots, same rng draw, same targets recomputed
    from the current executor, same cross-entropy -- with per-row weights:

        delta : w_raw = exp(-delta / tau_w)     `plasticity_gain`'s canonical direction, from
                                                Kim, Parvin & Ivry 2019: worse than benchmark
                                                (delta < 0) -> BOOST, at benchmark f(0) = 1 ->
                                                nominal, better than benchmark -> ATTENUATE.
                                                The gain is protection, not acceleration.
        raw   : w_raw = e                       "any error-modulated lr", no benchmark, no
                                                agency gate -- S13(b)'s hygiene control.

    `w_hat = clip(w_raw / EWMA(w_raw), 0, w_clip)` is the matched-average-budget normalisation
    (META_ADAPT #4e: fix the budget or the rule games scale rather than allocation) with the
    saturating cap (Kim's effect is categorical). A stored row carries credit only if the head
    EXECUTED it in the beam phase; rows captured on a closed slot, or in a probe, or before the
    slot first opened carry none and take w_raw = 1 (nominal). Their share is logged as
    `cred_frac` in every metered arm, because it bounds how much of the diet the gain can
    actually reach.

    `mode="uniform"` never reaches this function: the runner calls `SN.span_train_terms`
    itself, so the ungated arm is bit-identical to a plain span arm rather than merely equal
    in expectation."""
    import torch
    import torch.nn.functional as F
    terms, accs, wstat = [], {}, {"n": 0, "n_cred": 0, "sum_w": 0.0, "sum_d": 0.0}
    for key, info in slots.items():
        rows = ex.buf.get(key)
        if info.get("move") is None or rows is None or rows.shape[0] < 8:
            continue
        idx = torch.from_numpy(rng.integers(0, rows.shape[0], size=min(batch, rows.shape[0])))
        obs = rows[idx].to(device)
        move = info["move"]
        pooled, logits = SN.trunk(core, obs)
        with torch.no_grad():
            tgt = SN.dp_features(logits.detach(), move, s)
        sid = torch.full((obs.shape[0],), info["id"], dtype=torch.long, device=device)
        lg, _ = head(pooled, move["blk0"], move["span"], sid, teacher=tgt)
        ce = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), tgt.reshape(-1),
                             reduction="none").view(tgt.shape).mean(1)
        cred = getattr(ex, "cred", {}).get(key)
        d = (cred[idx].numpy().astype(np.float64)
             if (cred is not None and cred.shape[0] == rows.shape[0])
             else np.full(idx.shape[0], np.nan))
        have = ~np.isnan(d)
        if mode == "rawx":
            # [intonation/A] THE HYGIENE CONTROL, FIXED. `in_s0`'s `raw` arm used
            # `plasticity_gain`'s literal `w_raw = e`, which is right for a continuous MuJoCo
            # residual and wrong here: 94% of executions are EXACT, so `EWMA(e) ~ 0.03`, the 6%
            # non-exact rows saturate `w_clip`, and the realised mean weight came out 0.4015
            # against the delta arm's 0.9997 — the control trained its head at ~40% of nominal
            # plasticity and the comparison was confounded by budget rather than by hygiene.
            # `rawx` uses the SAME exponential transform, the same cap and the same normaliser
            # as the delta arm and differs from it in exactly the two things S13(b) is about:
            # no benchmark subtraction and no agency gate. exp(+e/tau_w) is exp(-delta/tau_w)
            # with delta replaced by -e.
            with torch.no_grad():
                err = (1.0 - (lg.argmax(-1) == tgt).float().mean(1)).cpu().numpy()
            w_raw = np.exp(np.clip(err.astype(np.float64) / tau_w, -30.0, np.log(w_raw_cap)))
        elif mode == "raw":
            # the raw-error gain is evaluated with the arm's CURRENT head, exactly as
            # `plasticity_gain`'s `raw_err` arm evaluates it with the arm's current FM: it
            # needs no stored credit and therefore no benchmark and no agency gate, which is
            # precisely the comparison S13(b) asks for.
            with torch.no_grad():
                err = (1.0 - (lg.argmax(-1) == tgt).float().mean(1)).cpu().numpy()
            w_raw = err.astype(np.float64)
        else:
            w_raw = np.where(have, np.exp(np.clip(-np.nan_to_num(d) / tau_w, -30.0,
                                                  np.log(w_raw_cap))), 1.0)
        wnorm[0] = (1 - wnorm[1]) * wnorm[0] + wnorm[1] * float(np.mean(w_raw))
        w_hat = np.clip(w_raw / max(wnorm[0], 1e-8), 0.0, w_clip).astype(np.float32)
        wt = torch.as_tensor(w_hat, device=device)
        terms.append((ce * wt).mean())
        accs[key] = float((lg.argmax(-1) == tgt).all(-1).float().mean())
        wstat["n"] += int(w_hat.shape[0]); wstat["n_cred"] += int(have.sum())
        wstat["sum_w"] += float(w_hat.sum())
        wstat["sum_d"] += float(np.nan_to_num(d).sum())
    if not terms:
        return None, accs, wstat
    return sum(terms) / len(terms), accs, wstat


def expand_selected_perf(generator, flat, sel, ms, ex, rules_t, canon, depth, v, m, s):
    """`PN.expand_selected`, plus the per-(row, col) execution credit the beam has to carry.

    Line-for-line the donor's loop -- same move order, same `hit.nonzero`, same `apply` call,
    same assembly -- so the child tensor and `n_mat` are bit-identical to `PN.expand_selected`.
    What is added is reading `ex.last` after each call, which the executor set for exactly the
    rows it was just handed. Only used when the meter is on."""
    import torch
    n, kk = sel.shape
    out = torch.zeros(n, kk, flat.shape[1], dtype=flat.dtype, device=flat.device)
    cred = torch.zeros(n, kk, dtype=torch.float32, device=flat.device)
    nbad = torch.zeros(n, kk, dtype=torch.float32, device=flat.device)
    nexe = torch.zeros(n, kk, dtype=torch.float32, device=flat.device)
    n_mat = 0
    for j in range(len(ms)):
        hit = sel == j
        if not bool(hit.any()):
            continue
        rows, cols = hit.nonzero(as_tuple=True)
        child = ex.apply(generator, flat[rows], ms[j], rules_t, canon, depth, v, m, s)
        out[rows, cols] = child
        last = getattr(ex, "last", None)
        if last is not None:
            cred[rows, cols] = torch.as_tensor(last["delta"], dtype=torch.float32,
                                               device=flat.device)
            nbad[rows, cols] = torch.as_tensor(~last["exact"], dtype=torch.float32,
                                               device=flat.device)
            nexe[rows, cols] = 1.0
        n_mat += int(rows.numel())
    return out, n_mat, cred, nbad, nexe


# --------------------------------------------------------------------------- #
# the beam, over a MIXED action set (base level moves + earned macros)
# --------------------------------------------------------------------------- #

def beam_moves(controller, generator, value, x0, roots, ms, rules_t, canon, depth, v, m, s,
               *, budget, beam_width, device, collect=False, ex=None):
    """ratchet's beam, with ONE change (span/'s): every materialisation goes through an executor
    object. `PlainExecutor.apply` IS `macros.apply_any` (span/'s gate S-3), so this is op-for-op
    ratchet for every untreated arm."""
    import torch
    if ex is None:
        ex = SN.PlainExecutor()
    with torch.no_grad():
        controller.eval(); generator.eval(); value.eval()
        x0 = x0.to(device); roots = roots.to(device)
        batch, length = x0.shape
        n_moves = len(ms)
        beams = x0[:, None, :]
        seqs = torch.zeros(batch, 1, 0, dtype=torch.long, device=device)
        width = 1
        counts = {"mat": 0, "ground": 0}
        hist_x = [beams.clone()] if collect else None
        hist_par = [] if collect else None
        # [intonation] the per-TIP execution ledger the 2x2 is read off. `track` is on only
        # when the meter is (`ex.last` exists), so an untreated beam is the donor's.
        track = (getattr(ex, "meter", None) is not None
                 and _ENTRY_REC.get("phase") == "beam")
        acc = ({q: torch.zeros(batch, 1, device=device) for q in ("d", "bad", "exe")}
               if track else None)
        for _ in range(budget):
            flat = beams.reshape(batch * width, length)
            children, kcred, kbad, kexe = [], [], [], []
            for k in range(n_moves):
                children.append(ex.apply(generator, flat, ms[k], rules_t, canon,
                                         depth, v, m, s))
                if track:
                    lz = getattr(ex, "last", None)
                    z = torch.zeros(flat.shape[0], device=device)
                    kcred.append(z if lz is None else torch.as_tensor(
                        lz["delta"], dtype=torch.float32, device=device))
                    kbad.append(z if lz is None else torch.as_tensor(
                        ~lz["exact"], dtype=torch.float32, device=device))
                    kexe.append(z if lz is None else torch.ones_like(z))
            cand = torch.stack(children, dim=1).reshape(batch, width * n_moves, length)
            counts["mat"] += batch * width * n_moves
            tgt = roots.repeat_interleave(width * n_moves)
            scores = value(_encode_chunked(controller, cand.reshape(-1, length)), tgt)
            scores = scores.reshape(batch, width * n_moves)
            counts["ground"] += batch * width * n_moves
            keep = min(beam_width, width * n_moves)
            _, top = scores.topk(keep, dim=1)
            parent = torch.div(top, n_moves, rounding_mode="floor")
            mv = top % n_moves
            if track:
                for nm, src in (("d", kcred), ("bad", kbad), ("exe", kexe)):
                    inc = torch.stack(src, dim=1).reshape(batch, width * n_moves)
                    acc[nm] = acc[nm].gather(1, parent) + inc.gather(1, top)
            beams = cand.gather(1, top[:, :, None].expand(-1, -1, length))
            prev = seqs.gather(1, parent[:, :, None].expand(-1, -1, seqs.shape[2])) \
                if seqs.shape[2] else seqs.new_zeros(batch, keep, 0)
            seqs = torch.cat([prev, mv[:, :, None]], dim=2)
            width = keep
            if collect:
                hist_par.append(parent)
                hist_x.append(beams.clone())
        tgt = roots.repeat_interleave(width)
        final = value(_encode_chunked(controller, beams.reshape(-1, length)), tgt).reshape(batch, width)
        counts["ground"] += batch * width
        best = final.argmax(dim=1)
        rows = torch.arange(batch, device=device)
        out = {"x": beams[rows, best], "seq": seqs[rows, best], "counts": counts,
               "tips_x": beams, "tips_seq": seqs,
               # [tacet] THE SCALARS THE GATE READS, already computed above to pick the answer:
               # the value head's score of every surviving tip, and which tip won. Returning
               # them costs nothing — no forward pass, no grounding charged, no RNG drawn — and
               # they are unread unless `gate_mode` is set. `final` is a LOGIT (the value head
               # is trained under `binary_cross_entropy_with_logits`), so the gate takes its
               # sigmoid; `best` indexes `tip_val` at the answer `out["x"]` actually is.
               "tip_val": final, "best": best}
        if track:
            # [intonation] per surviving tip: the summed delta_perf of the head executions on
            # its own trajectory, how many of them there were, and how many were NOT exact.
            # `tip_bad == 0 and tip_exe > 0` is "executed as intended" — the 2x2's row axis.
            out["tip_dperf"] = acc["d"]
            out["tip_bad"] = acc["bad"]
            out["tip_exe"] = acc["exe"]
        if collect:
            idx = torch.arange(width, device=device)[None, :].expand(batch, -1).contiguous()
            traj = [None] * (budget + 1)
            for t in range(budget, -1, -1):
                traj[t] = hist_x[t].gather(1, idx[:, :, None].expand(-1, -1, length))
                if t > 0:
                    idx = hist_par[t - 1].gather(1, idx)
            out["traj"] = traj
    return out


# --------------------------------------------------------------------------- #
# THE PORT: the same beam, but only the top-k PROPOSED moves per tip are expanded
# --------------------------------------------------------------------------- #

def beam_moves_prop(controller, generator, value, x0, roots, ms, rules_t, canon, depth, v, m, s,
                    *, budget, beam_width, device, collect=False,
                    prop, slots, k, forced=(), n_slots, explore=0, erng=None, ex=None):
    """`beam_moves` with the enumeration replaced by a proposal.

    Structurally line-for-line the original: same children ordering (tip-major, move-minor),
    same `topk` over value scores, same parent/seq bookkeeping, same final re-score. The only
    changes are (i) which children exist at all, and (ii) that the kept children's encoder
    states are carried forward instead of being recomputed for the proposal read.

    `slots` maps position-in-`ms` -> slot id, so the head's output vocabulary is stable across
    an action set that grows at a commit. `forced` is a list of positions in `ms` that are
    expanded unconditionally (a move too new for the head to have a training example on).

    FIDELITY. At k >= len(ms) with `forced` empty, `select_moves` returns `arange(n_moves)` for
    every tip, `expand_selected`'s row gather is the identity, and the assembled child tensor,
    the scores, the topk and the sequences are the enumerated ones bit-for-bit. The only
    difference is the extra root encode, which is charged and touches nothing else."""
    import torch
    if ex is None:
        ex = SN.PlainExecutor()
    with torch.no_grad():
        controller.eval(); generator.eval(); value.eval(); prop.eval()
        x0 = x0.to(device); roots = roots.to(device)
        batch, length = x0.shape
        n_moves = len(ms)
        slot_t = torch.as_tensor(slots, dtype=torch.long, device=device)      # (n_moves,)
        beams = x0[:, None, :]
        seqs = torch.zeros(batch, 1, 0, dtype=torch.long, device=device)
        width = 1
        counts = {"mat": 0, "ground": 0, "prop": 0}
        hist_x = [beams.clone()] if collect else None
        hist_par = [] if collect else None

        # the ROOT read: the one encoder pass the port genuinely adds, charged in full.
        z_tip = _encode_chunked(controller, x0)                               # (batch, D)
        counts["ground"] += batch

        # [intonation] the per-TIP execution ledger; see `beam_moves`.
        track = (getattr(ex, "meter", None) is not None
                 and _ENTRY_REC.get("phase") == "beam")
        acc = ({q: torch.zeros(batch, 1, device=device) for q in ("d", "bad", "exe")}
               if track else None)
        for _ in range(budget):
            flat = beams.reshape(batch * width, length)
            zf = z_tip.reshape(batch * width, -1)
            rf = roots.repeat_interleave(width)
            plog = prop(zf, rf)[:, slot_t]                                    # -> `ms` order
            counts["prop"] += batch * width
            sel = PN.select_moves(plog, k, forced=list(forced))               # (N, k)
            if explore and sel.shape[1] < n_moves:
                # NB: not named `ex` — that is the executor parameter now.
                exp_idx = PN.explore_moves(sel, n_moves, explore, erng, device)
                sel = torch.sort(torch.cat([sel, exp_idx], dim=1), dim=1).values
            kk = sel.shape[1]
            if track:
                child, n_mat, kcred, kbad, kexe = expand_selected_perf(
                    generator, flat, sel, ms, ex, rules_t, canon, depth, v, m, s)
            else:
                child, n_mat = PN.expand_selected(generator, flat, sel, ms, ex.apply,
                                                  rules_t, canon, depth, v, m, s)
            cand = child.reshape(batch, width * kk, length)
            counts["mat"] += n_mat
            tgt = roots.repeat_interleave(width * kk)
            z_all = _encode_chunked(controller, cand.reshape(-1, length))
            scores = value(z_all, tgt).reshape(batch, width * kk)
            counts["ground"] += batch * width * kk
            keep = min(beam_width, width * kk)
            _, top = scores.topk(keep, dim=1)
            parent = torch.div(top, kk, rounding_mode="floor")
            col = top % kk
            mv = sel.reshape(batch, width, kk).gather(
                1, parent[:, :, None].expand(-1, -1, kk)).gather(2, col[:, :, None]).squeeze(2)
            if track:
                for nm, src in (("d", kcred), ("bad", kbad), ("exe", kexe)):
                    acc[nm] = acc[nm].gather(1, parent) + \
                        src.reshape(batch, width * kk).gather(1, top)
            beams = cand.gather(1, top[:, :, None].expand(-1, -1, length))
            z_tip = z_all.reshape(batch, width * kk, -1).gather(
                1, top[:, :, None].expand(-1, -1, z_all.shape[-1]))
            prev = seqs.gather(1, parent[:, :, None].expand(-1, -1, seqs.shape[2])) \
                if seqs.shape[2] else seqs.new_zeros(batch, keep, 0)
            seqs = torch.cat([prev, mv[:, :, None]], dim=2)
            width = keep
            if collect:
                hist_par.append(parent)
                hist_x.append(beams.clone())
        tgt = roots.repeat_interleave(width)
        final = value(_encode_chunked(controller, beams.reshape(-1, length)), tgt).reshape(batch, width)
        counts["ground"] += batch * width
        best = final.argmax(dim=1)
        rows = torch.arange(batch, device=device)
        out = {"x": beams[rows, best], "seq": seqs[rows, best], "counts": counts,
               "tips_x": beams, "tips_seq": seqs,
               # [tacet] the same two keys the enumerated beam now returns; see there.
               "tip_val": final, "best": best}
        if track:                                        # [intonation] see `beam_moves`
            out["tip_dperf"] = acc["d"]
            out["tip_bad"] = acc["bad"]
            out["tip_exe"] = acc["exe"]
        if collect:
            idx = torch.arange(width, device=device)[None, :].expand(batch, -1).contiguous()
            traj = [None] * (budget + 1)
            for t in range(budget, -1, -1):
                traj[t] = hist_x[t].gather(1, idx[:, :, None].expand(-1, -1, length))
                if t > 0:
                    idx = hist_par[t - 1].gather(1, idx)
            out["traj"] = traj
    return out


def plan(controller, generator, value, x0, roots, ms, rules_t, canon, depth, v, m, s,
         *, budget, beam_width, device, collect=False, port=None, ex=None):
    """One entry point for every place the agent plans, so Port 1 is either everywhere the beam
    runs or nowhere; `ex` is Port 2's executor, threaded the same way. `port=None` with a plain
    executor is ratchet's `beam_moves`, byte-for-byte."""
    if port is None:
        return beam_moves(controller, generator, value, x0, roots, ms, rules_t, canon, depth,
                          v, m, s, budget=budget, beam_width=beam_width, device=device,
                          collect=collect, ex=ex)
    return beam_moves_prop(controller, generator, value, x0, roots, ms, rules_t, canon, depth,
                           v, m, s, budget=budget, beam_width=beam_width, device=device,
                           collect=collect, ex=ex, **port)


# --------------------------------------------------------------------------- #
# online learning: the value (selector) and the generator (plant)
# --------------------------------------------------------------------------- #

def value_steps(value, opt, controller, buf, replay, *, n_steps, batch, replay_frac, device, rng):
    import torch
    import torch.nn.functional as F
    if buf["x"].shape[0] == 0:
        return 0.0
    value.train()
    n_rep = int(round(batch * replay_frac)) if replay["x"].shape[0] else 0
    n_new = batch - n_rep
    last = 0.0
    for _ in range(n_steps):
        px, pr, py = [], [], []
        if n_new:
            i = torch.from_numpy(rng.integers(0, buf["x"].shape[0], size=n_new))
            px.append(buf["x"][i]); pr.append(buf["r"][i]); py.append(buf["y"][i])
        if n_rep:
            i = torch.from_numpy(rng.integers(0, replay["x"].shape[0], size=n_rep))
            px.append(replay["x"][i]); pr.append(replay["r"][i]); py.append(replay["y"][i])
        x = torch.cat(px).to(device); r = torch.cat(pr).to(device); y = torch.cat(py).to(device)
        with torch.no_grad():
            z = controller.state(x)
        loss = F.binary_cross_entropy_with_logits(value(z, r), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(value.parameters(), 1.0)
        opt.step()
        last = float(loss.item())
    value.eval()
    return last


def finetune_generator(generator, opt, new_leaves, replay_leaves, bottom_map, *, v, s, n_blocks,
                       n_steps, batch, replay_frac, device, rng):
    """THE PLANT LEARNS (calp_s0's recipe, verbatim): masked infilling on configurations the
    agent SOLVED — a solved config is a valid r* derivation, so the target is well defined —
    with replay `replay_frac` against the clean setup pool as the anti-drift guard."""
    import torch
    import torch.nn.functional as F
    powers = v ** torch.arange(s, device=device)
    generator.train()
    last = 0.0
    n_rep = int(round(batch * replay_frac))
    n_new = batch - n_rep
    for _ in range(n_steps):
        parts = []
        if n_new and new_leaves.shape[0]:
            parts.append(new_leaves[torch.from_numpy(
                rng.integers(0, new_leaves.shape[0], size=n_new))])
        if n_rep or not parts:
            k = n_rep if parts else batch
            parts.append(replay_leaves[torch.from_numpy(
                rng.integers(0, replay_leaves.shape[0], size=k))])
        leaves = torch.cat(parts).to(device)
        b = leaves.shape[0]
        feats = bottom_map[(leaves.view(b, n_blocks, s) * powers).sum(-1)]
        keep = (feats >= 0).all(1)
        if keep.sum() < 8:
            continue
        leaves, feats = leaves[keep], feats[keep]
        b = leaves.shape[0]
        n_mask = int(torch.randint(1, n_blocks + 1, ()).item())
        order = torch.rand(b, n_blocks, device=device).argsort(dim=1)
        mb = order[:, :n_mask]
        pos = (mb[:, :, None] * s + torch.arange(s, device=device)).reshape(b, -1)
        obs = leaves.clone().scatter_(1, pos, torch.full_like(pos, -1))
        logits = generator.block_logits(obs)
        mask = torch.zeros(b, n_blocks, dtype=torch.bool, device=device)
        mask.scatter_(1, mb, True)
        loss = F.cross_entropy(logits[mask], feats[mask])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
        opt.step()
        last = float(loss.item())
    generator.eval()
    return last


def finetune_generator_span(generator, head, ex, slots, opt, new_leaves, replay_leaves,
                            bottom_map, *, v, s, n_blocks, n_steps, batch, replay_frac, device,
                            rng, span_rng, span_batch, lam=1.0,
                            perf_gain=None):
    """THE PLANT LEARNS, WITH A CORRIDOR HEAD (`../span/span.py`'s, verbatim apart from the
    `lam == 0` short-circuit). `finetune_generator` op-for-op — same batches, same number of
    optimizer steps, same masking draws, same level-1 loss — plus the span head's self-imitation
    cross-entropy at weight `lam`, averaged over minted slots, in the SAME steps.

    `lam == 0` skips the term entirely rather than adding `0 * term`. Adding a zeroed term would
    be numerically a no-op but not GRAPH-identical, and the composed fidelity arm (`given_fid`)
    is an assertion about graph identity, not about tolerances."""
    # [intonation] `perf_gain` (a dict, or None) switches the span term from `SN`'s uniform
    # form to the per-sample weighted one. None is the donor EXACTLY — the donor's own function
    # is called, not an equal-in-expectation re-derivation — which is what makes the ungated
    # arm a real control rather than a numerically-close one.
    import torch
    import torch.nn.functional as F
    powers = v ** torch.arange(s, device=device)
    generator.train()
    if head is not None:
        head.train()
    last, slast, sacc = 0.0, None, {}
    wstat = None
    n_rep = int(round(batch * replay_frac))
    n_new = batch - n_rep
    for _ in range(n_steps):
        parts = []
        if n_new and new_leaves.shape[0]:
            parts.append(new_leaves[torch.from_numpy(
                rng.integers(0, new_leaves.shape[0], size=n_new))])
        if n_rep or not parts:
            k = n_rep if parts else batch
            parts.append(replay_leaves[torch.from_numpy(
                rng.integers(0, replay_leaves.shape[0], size=k))])
        leaves = torch.cat(parts).to(device)
        b = leaves.shape[0]
        feats = bottom_map[(leaves.view(b, n_blocks, s) * powers).sum(-1)]
        keep = (feats >= 0).all(1)
        if keep.sum() < 8:
            continue
        leaves, feats = leaves[keep], feats[keep]
        b = leaves.shape[0]
        n_mask = int(torch.randint(1, n_blocks + 1, ()).item())
        order = torch.rand(b, n_blocks, device=device).argsort(dim=1)
        mb = order[:, :n_mask]
        pos = (mb[:, :, None] * s + torch.arange(s, device=device)).reshape(b, -1)
        obs = leaves.clone().scatter_(1, pos, torch.full_like(pos, -1))
        logits = generator.block_logits(obs)
        mask = torch.zeros(b, n_blocks, dtype=torch.bool, device=device)
        mask.scatter_(1, mb, True)
        loss = F.cross_entropy(logits[mask], feats[mask])
        if head is not None and slots and lam:
            if perf_gain is None:
                sterm, sacc = SN.span_train_terms(generator, head, ex, slots, s, span_batch,
                                                  span_rng, device)
            else:
                sterm, sacc, wstat = perf_span_train_terms(
                    generator, head, ex, slots, s, span_batch, span_rng, device,
                    mode=perf_gain["mode"], tau_w=perf_gain["tau_w"],
                    w_raw_cap=perf_gain["w_raw_cap"], w_clip=perf_gain["w_clip"],
                    wnorm=perf_gain["wnorm"])
            if sterm is not None:
                loss = loss + lam * sterm
                slast = float(sterm.item())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
        if head is not None and lam:
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()
        last = float(loss.item())
    generator.eval()
    if head is not None:
        head.eval()
    if perf_gain is not None:
        perf_gain["last_wstat"] = wstat          # [intonation] logged, never consumed
    return last, slast, sacc


def push(buf, x, r, y, cap):
    import torch
    buf["x"] = torch.cat([buf["x"], x])[-cap:]
    buf["r"] = torch.cat([buf["r"], r])[-cap:]
    buf["y"] = torch.cat([buf["y"], y])[-cap:]


# --------------------------------------------------------------------------- #
# THE PORT'S LEARNING: self-imitation of the beam's own chosen trajectories
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# [tacet] THE GATE — what the learner consumes but chooses not to learn from
# --------------------------------------------------------------------------- #

GATE_MODES = ("delta_hi", "delta_lo", "delib", "random",
              # [intonation] the two delta_PERF orders. `perf_hi` keeps the trajectories whose
              # HEAD EXECUTIONS went best relative to their own recent selves — "learn from
              # what you played well", which is orthogonal to whether the task was solved and
              # is the whole point of the node. `perf_lo` is its mirror, carried so the
              # direction is a measured fact rather than an assumption; neither is run in the
              # main tag beyond `perf_hi` (arm budget).
              "perf_hi", "perf_lo",
              # [intonation/A] THE COVERAGE FIX. `in_s0` measured Sigma-delta_perf EXACTLY ZERO
              # on 57.0% of the gate's candidates — trajectories the head never executed on —
              # so more than half the ranking fell to index order and the gate was, on those
              # rows, `gate_random` with extra steps. `perf_mean` makes the no-execution policy
              # DELIBERATE and two-key: (1) a trajectory the head executed on is always
              # preferred to one it did not (no performance evidence is not the same as average
              # performance), and (2) within the executed set the key is delta_perf PER
              # EXECUTION, not summed — a trial's quality is its average rendition, and the sum
              # conflates how well it was played with how much of it was played.
              "perf_mean")


def gate_features(out, succ, ps, n_pr, width):
    """The two decision-time scalars, per trajectory and per instance, from what the beam
    already computed. No forward pass, no RNG, no grounding charged.

    `audiation` finding 3 located the per-datum grade-news at decision time and named exactly
    two carriers: the pre-update value readout (which decodes the value-error target at 0.293)
    and the deliberation state — the unchosen candidates' own value scores (0.133). Both are
    materialised BEFORE the grade arrives. This function is the whole of what the gate reads.

        v_tip    sigmoid of the value head's score of each surviving tip           (B, W)
        d_tip    grade - v, per trajectory: `succ` is that tip's own terminal
                 possible-set success, so for a solved tip d = 1 - v is how
                 SURPRISING the success was                                        (B, W)
        v_ans    v at the tip the beam actually answered with                       (B,)
        d_ans    ps - v_ans: the same residual at the MINED unit                    (B,)
        margin   sigmoid(v at the chosen tip) - mean over the UNCHOSEN tips of
                 sigmoid(v) — the deliberation state as one scalar. Small margin
                 == the unchosen candidates scored nearly as well as the chosen
                 one, i.e. the selection was contested                              (B,)
        m2       sigmoid(top1) - sigmoid(top2), the textbook decision margin,
                 LOGGED but not gated on. It is the more familiar statistic and
                 the more degenerate one: the beam's tips are configurations, and
                 two distinct move sequences can reach the same configuration, so
                 top1 == top2 exactly whenever the runner-up is a duplicate of the
                 answer, and the gate would then be ranking on ties. The mean form
                 uses all W-1 unchosen scores — which is literally what `audiation`
                 finding 3 decoded — and is zero only if every tip scores alike     (B,)

    Returned as numpy so everything downstream (ranking, logging, the reduction) is host-side
    and deterministic.
    """
    import torch
    val = out["tip_val"].detach()                       # (B, W) LOGITS
    B, W = val.shape
    assert B == n_pr and W == width, f"gate: unexpected tip block {val.shape}"
    p = torch.sigmoid(val)
    best = out["best"].detach()
    v_ans = p.gather(1, best[:, None]).squeeze(1)
    if W > 1:
        rest = (p.sum(1) - v_ans) / (W - 1)             # mean over the UNCHOSEN tips
        margin = v_ans - rest
        top = p.topk(2, dim=1).values
        m2 = top[:, 0] - top[:, 1]
    else:
        margin = torch.zeros_like(v_ans)
        m2 = torch.zeros_like(v_ans)
    v_tip = p.cpu().numpy().astype(np.float64)
    y_tip = np.asarray(succ, dtype=np.float64).reshape(B, W)
    fe = {"v_tip": v_tip, "d_tip": y_tip - v_tip, "y_tip": y_tip,
          "v_ans": v_ans.cpu().numpy().astype(np.float64),
          "d_ans": np.asarray(ps, dtype=np.float64) - v_ans.cpu().numpy().astype(np.float64),
          "margin": margin.cpu().numpy().astype(np.float64),
          "m2": m2.cpu().numpy().astype(np.float64)}
    # [intonation] THE EXECUTION COLUMN, per tip and at the answer. Free on the wire in exactly
    # the sense the value scores are: the beam accumulated it while materialising, no forward
    # pass and no grounding. `p_tip` is the trajectory's summed delta_perf; `bad_tip` how many
    # of its head executions were NOT exact; `exe_tip` how many there were. A tip the head
    # never executed on has p = 0 and exe = 0 — no performance evidence, which ranks neutral
    # rather than good or bad, and is counted separately in the 2x2 so it is never silently
    # folded into "as intended".
    if "tip_dperf" in out:
        for src, dst in (("tip_dperf", "p_tip"), ("tip_bad", "bad_tip"), ("tip_exe", "exe_tip")):
            fe[dst] = out[src].detach().cpu().numpy().astype(np.float64).reshape(B, W)
        bi = best.cpu().numpy()
        rows_ = np.arange(B)
        fe["p_ans"] = fe["p_tip"][rows_, bi]
        fe["bad_ans"] = fe["bad_tip"][rows_, bi]
        fe["exe_ans"] = fe["exe_tip"][rows_, bi]
    return fe


def gate_order(mode, cand, score, margin, grng, perf=None, nexe=None):
    """The candidate indices, ordered best-first under `mode`.

    `cand` is a 1-D index array into the candidate set (mining: solved instances; pi: solved
    tips). `score` is delta at that unit, `margin` the deliberation state broadcast to it.
    Ties are broken by index so the ordering is a deterministic function of the run, never of
    dict order or of a sort's internal state — which matters because `delib` ties by
    construction (all W tips of an instance share their instance's margin).

    `random` draws from `grng`, the gate's OWN numpy stream, so the shared per-arm RNG the twin
    gate rests on is never perturbed by the gate's own bookkeeping."""
    if mode == "delta_hi":
        key = (-score[cand], cand)
    elif mode == "delta_lo":
        key = (score[cand], cand)
    elif mode == "delib":
        key = (margin[cand], cand)                  # ascending margin == most contested first
    elif mode == "random":
        return cand[grng.permutation(cand.shape[0])]
    # [intonation] the delta_PERF orders. `perf` is the summed execution credit at the same
    # unit; ties (a unit the head never executed on, so perf == 0) fall to index order, and
    # the reduction prints how many there were, on `tacet`'s own `m2 == 0` convention.
    elif mode == "perf_hi":
        key = (-perf[cand], cand)
    elif mode == "perf_lo":
        key = (perf[cand], cand)
    elif mode == "perf_mean":
        ex_ = np.asarray(nexe, dtype=np.float64)[cand] if nexe is not None \
            else np.zeros(cand.shape[0])
        mu = np.where(ex_ > 0, perf[cand] / np.maximum(ex_, 1.0), 0.0)
        key = (-(ex_ > 0).astype(np.float64), -mu, cand)
    else:
        raise ValueError(f"unknown gate_mode {mode!r}")
    return cand[np.lexsort(key[::-1])]


def prop_pairs(out, roots, succ, slots, budget, train_on="solved", keep=None):
    """The beam's own chosen trajectories, as (state, root-index, slot) supervision.

    `beam_moves(collect=True)` already hands these back: `traj[t]` is the per-step state of
    every SURVIVING trajectory (ancestry resolved back from the tips) and `tips_seq[:, :, t]`
    is the move that trajectory took at step t. `succ` is those tips' terminal possible-set
    success, so `train_on="solved"` regresses only onto trajectories that WORKED — the
    selection step that makes this self-imitation rather than averaging over everything the
    agent did.

    Move indices are converted to SLOTS here, at collection time, because a position in `ms`
    stops meaning the same thing the moment the action set grows."""
    import torch
    traj, seq = out["traj"], out["tips_seq"]
    B, W = seq.shape[0], seq.shape[1]
    length = traj[0].shape[2]
    slot_t = torch.as_tensor(slots, dtype=torch.long, device=seq.device)
    sel = torch.as_tensor(succ > 0.5, device=seq.device).reshape(B, W) \
        if train_on == "solved" else torch.ones(B, W, dtype=torch.bool, device=seq.device)
    # [tacet] THE GATE, on the pi channel. `keep` is a (B, W) boolean over trajectories; the
    # grade filter above still runs first, so the gate can only ever REFUSE data the donor's
    # rule already admitted. `keep=None` is the donor exactly.
    if keep is not None:
        sel = sel & torch.as_tensor(np.asarray(keep), device=seq.device).reshape(B, W)
    keep = sel
    if not bool(keep.any()):
        return None
    rr = roots.to(seq.device)[:, None].expand(B, W)
    xs, as_, rs = [], [], []
    for t in range(budget):
        xs.append(traj[t][keep])                       # (n_keep, T)
        as_.append(slot_t[seq[:, :, t][keep]])         # (n_keep,)
        rs.append(rr[keep])
    return {"x": torch.cat(xs).cpu(), "a": torch.cat(as_).cpu(),
            "r": torch.cat(rs).cpu(), "n_traj": int(keep.sum())}


def prop_train(prop, opt, controller, pbuf, avail_mask, *, n_steps, batch, device, rng):
    """Masked cross-entropy of pi(. | z, r*) against the move the selected trajectory took.

    The mask is the CURRENT action set: slots that do not exist are not competitors and are
    never targets. Buffered pairs from before a commit are replayed under the current mask —
    the mask only ever gains options and the stored target stays legal."""
    import torch
    import torch.nn.functional as F
    n = pbuf["x"].shape[0]
    if n == 0:
        return {"loss": None, "acc": None, "n": 0}
    prop.train()
    last, acc = 0.0, 0.0
    neg = torch.finfo(torch.float32).min / 4
    for _ in range(n_steps):
        i = torch.from_numpy(rng.integers(0, n, size=min(batch, n)))
        x = pbuf["x"][i].to(device)
        r = pbuf["r"][i].to(device)
        a = pbuf["a"][i].to(device)
        with torch.no_grad():
            z = controller.state(x)
        logits = prop(z, r).masked_fill(~avail_mask[None, :], neg)
        loss = F.cross_entropy(logits, a)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(prop.parameters(), 1.0)
        opt.step()
        last = float(loss.item())
        acc = float((logits.argmax(-1) == a).float().mean())
    prop.eval()
    return {"loss": last, "acc": acc, "n": int(n)}


# --------------------------------------------------------------------------- #
# setup: instruments trained ONCE, forked per arm
# --------------------------------------------------------------------------- #

def behavior_step(controller, generator, x, roots, ms, rules_t, canon, depth, v, m, s,
                  eps, device, rng):
    import torch
    with torch.no_grad():
        batch = x.shape[0]
        props, scores = [], []
        for mv in ms:
            p = apply_move(generator, x, mv, rules_t, canon, depth, v, m, s)
            lp = controller.root_logits(controller.state(p)).log_softmax(-1)
            scores.append(lp.gather(1, roots[:, None]).squeeze(1))
            props.append(p)
        scores = torch.stack(scores, dim=1)
        props = torch.stack(props, dim=1)
        chosen = scores.argmax(dim=1)
        explore = torch.from_numpy(rng.random(batch) < eps).to(device)
        rnd = torch.from_numpy(rng.integers(0, len(ms), size=batch)).to(device)
        chosen = torch.where(explore, rnd, chosen)
        return props.gather(1, chosen[:, None, None].expand(-1, 1, x.shape[1])).squeeze(1)


def collect_value_buffer(controller, generator, rules, rules_t, canon, ms,
                         roots_pool, leaves_pool, cfg, device):
    import torch
    v, s, depth, m = cfg["v"], cfg["s"], cfg["depth"], cfg["m"]
    n_blocks = leaves_pool.shape[1] // s
    rng = np.random.default_rng(cfg["train_seed"] + 4321)
    xs, rs, ys = [], [], []
    done = 0
    while done < cfg["value_episodes"]:
        b = min(cfg["value_batch_collect"], cfg["value_episodes"] - done)
        done += b
        idx = rng.integers(0, leaves_pool.shape[0], size=b)
        roots_np = roots_pool[idx]
        c = int(rng.integers(1, cfg["n_corrupt"] + 1))
        start = _corrupt(leaves_pool[idx], n_blocks, c, v, s, rng)
        x = torch.from_numpy(start).to(device)
        roots = torch.from_numpy(roots_np).to(device)
        traj = [x.clone()]
        for _ in range(cfg["budget"]):
            x = behavior_step(controller, generator, x, roots, ms, rules_t, canon,
                              depth, v, m, s, cfg["explore_eps"], device, rng)
            traj.append(x.clone())
        succ, _ = grade(x.cpu().numpy(), roots_np, rules, s)
        y = torch.from_numpy(succ.astype(np.float32))
        for st in traj:
            xs.append(st.cpu()); rs.append(roots.cpu()); ys.append(y)
    return {"x": torch.cat(xs), "r": torch.cat(rs), "y": torch.cat(ys)}


def train_reader(reader, leaves, bottom_map, *, v, s, n_blocks, n_steps, batch, lr, device,
                 seed=0):
    """THE READ. `_train_generator` computes its loss only on MASKED blocks, so the block head
    at a VISIBLE position is never supervised — `calp2_s0` measured the consequence: 0.63
    accuracy reading a fully visible block, *worse* than the 0.70 it gets infilling a hidden
    one, and masking elsewhere to fix the distribution shift did not help (0.599). That caps an
    earned level-3 vocabulary's precision at ~0.3, which is what `calr_s0` saw.

    So the agent gets a reader: the same architecture, the same corpus, and the SAME
    supervision channel the generator already trains against (`bottom_map` on clean
    configurations), but trained to read rather than to fill. This is bottom-level perceptual
    competence — the inverse of the `canon` rendering the agent already performs — and it is
    deliberately the only thing handed over: every composition above level 1 still has to be
    earned."""
    import torch
    import torch.nn.functional as F
    opt = torch.optim.AdamW(reader.parameters(), lr=lr, weight_decay=1e-4)
    powers = v ** torch.arange(s, device=device)
    g = torch.Generator().manual_seed(seed)
    reader.train()
    last = (0.0, 0.0)
    for step in range(n_steps):
        idx = torch.randint(0, leaves.shape[0], (batch,), generator=g)
        x = leaves[idx].to(device)
        feats = bottom_map[(x.view(batch, n_blocks, s) * powers).sum(-1)]
        ok = feats >= 0
        logits = reader.block_logits(x)
        loss = F.cross_entropy(logits[ok], feats[ok])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(reader.parameters(), 1.0)
        opt.step()
        if (step + 1) % max(1, n_steps // 4) == 0:
            acc = float((logits[ok].argmax(-1) == feats[ok]).float().mean())
            last = (float(loss.item()), acc)
            print(f"  reader       step {step + 1:5d}/{n_steps}: loss={last[0]:.4f} "
                  f"read_acc={acc:.3f}", flush=True)
    reader.eval()
    for p in reader.parameters():
        p.requires_grad_(False)
    return last


def holdout_set(rules, depth, s, v, m, n_hold, seed):
    """The plant's frontier: `n_hold` of the grammar's level-2 tuples, withheld from the
    setup corpus. Level 2 is chosen because every era's damage requires producing a correct
    level-2 tuple (era 1 at one block of it, era 2 at the whole node, era 3 at both nodes of a
    level-3 span), so the plant's hole and the vocabulary's hole are the SAME hole — which is
    what keeps the compound descent from decoupling."""
    truth2 = MC.true_tables(rules, depth, s, v, m, 2)[2]
    flats = sorted({tuple(int(x) for x in r) for r in truth2["flat"]})
    rng = np.random.default_rng(seed)
    pick = rng.permutation(len(flats))[:n_hold]
    return {flats[i] for i in pick}


def level2_tuples(leaves_np, inverse_bottom, v, s):
    """Exact level-1 features per block -> the (N, n_l2, s) level-2 tuples."""
    feats = MC.exact_features(leaves_np, inverse_bottom, v, s)
    return feats.reshape(feats.shape[0], -1, s)


def filter_pool(leaves_np, roots_np, inverse_bottom, v, s, hold):
    """Reject configurations that use a held-out level-2 tuple anywhere. The setup corpus is
    filtered; the EVALUATION instances are drawn from the unfiltered DGP, exactly the
    train-on-generic / evaluate-on-the-real-thing structure round 1 used to make its value
    arrive stale."""
    if not hold:
        return roots_np, leaves_np, 1.0
    tup = level2_tuples(leaves_np, inverse_bottom, v, s)
    bad = np.zeros(leaves_np.shape[0], bool)
    for t in hold:
        bad |= (tup == np.array(t)).all(-1).any(-1)
    keep = ~bad
    return roots_np[keep], leaves_np[keep], float(keep.mean())


def build_shared(cfg, device):
    """Controller / generator / reader / stale value, trained once. The BASE action space
    (level-1 moves only) is what the setup value's behaviour policy explores, so every arm
    starts from the same stale selector whatever vocabulary it is later given."""
    import torch
    v, s, depth, m = cfg["v"], cfg["s"], cfg["depth"], cfg["m"]
    length = s ** depth
    n_blocks = length // s

    rules = generate_rules_distinct(v, s, depth, m, seed=cfg["rule_seed"])
    inverse_maps = build_inverse_maps(rules)
    bottom_map = torch.from_numpy(inverse_maps[-1]).to(device)
    canon = torch.from_numpy(np.ascontiguousarray(rules[depth - 1][:, 0, :])).to(device)
    rules_t = [torch.from_numpy(np.ascontiguousarray(r)).to(device) for r in rules]
    base_ms = build_move_set(depth, s, max_level=1)

    hold = holdout_set(rules, depth, s, v, m, cfg["plant_holdout"], cfg["holdout_seed"])
    want = cfg["n_train_episodes"]
    over = 1.0
    roots_np, leaves_np = _sample_pool(rules, want, s, cfg["train_seed"])
    if hold:
        # oversample so the FILTERED corpus is still `n_train_episodes` long -- otherwise the
        # holdout arm would also be a less-data arm and the two would be confounded
        _, _, frac = filter_pool(leaves_np, roots_np, inverse_maps[-1], v, s, hold)
        over = max(1.05, 1.0 / max(frac, 0.05)) * 1.15
        roots_np, leaves_np = _sample_pool(rules, int(want * over), s, cfg["train_seed"])
        roots_np, leaves_np, frac = filter_pool(leaves_np, roots_np, inverse_maps[-1],
                                                v, s, hold)
        roots_np, leaves_np = roots_np[:want], leaves_np[:want]
        print(f"  plant holdout: {len(hold)} level-2 tuples withheld; corpus keep-rate "
              f"{frac:.3f}, corpus size {leaves_np.shape[0]}")
    leaves = torch.from_numpy(leaves_np)
    roots = torch.from_numpy(roots_np)

    controller = _build_rich_controller()(v, length, s, cfg["state_dim"], n_head=4, n_layer=2).to(device)
    _train_edit_controller(controller, leaves, roots, batch_size=cfg["batch_size"],
                           n_blocks=n_blocks, block_size=s, n_steps=cfg["controller_steps"],
                           lr=3e-4, device=device, p_full=0.5)
    generator = _build_generator()(v, length, s, cfg["state_dim"], n_head=4, n_layer=2,
                                   root_conditioned=False).to(device)
    _train_generator(generator, leaves, roots, bottom_map, batch_size=cfg["batch_size"],
                     n_blocks=n_blocks, v=v, block_size=s, mask_min=1, mask_max=n_blocks,
                     n_steps=cfg["generator_steps"], lr=3e-4, device=device)
    reader = _build_generator()(v, length, s, cfg["state_dim"], n_head=4, n_layer=2,
                                root_conditioned=False).to(device)
    read_loss, read_acc = train_reader(reader, leaves, bottom_map, v=v, s=s, n_blocks=n_blocks,
                                       n_steps=cfg["reader_steps"], batch=cfg["batch_size"],
                                       lr=3e-4, device=device, seed=cfg["train_seed"] + 5)
    controller.eval(); generator.eval()
    for p in controller.parameters():
        p.requires_grad_(False)

    print(f"Collecting the generic (stale) value buffer: {cfg['value_episodes']} rollouts")
    vb = collect_value_buffer(controller, generator, rules, rules_t, canon, base_ms,
                              roots_np, leaves_np, cfg, device)
    value = _build_value_head()(cfg["state_dim"], v).to(device)
    opt = torch.optim.AdamW(value.parameters(), lr=cfg["value_lr"], weight_decay=1e-4)
    rng = np.random.default_rng(cfg["train_seed"] + 31)
    empty = {"x": vb["x"][:0], "r": vb["r"][:0], "y": vb["y"][:0]}
    print(f"  buffer {vb['x'].shape[0]} states, terminal success {float(vb['y'].mean()):.3f}")
    value_steps(value, opt, controller, vb, empty, n_steps=cfg["value_steps"],
                batch=512, replay_frac=0.0, device=device, rng=rng)

    truth = MC.true_tables(rules, depth, s, v, m, cfg["max_macro_level"])
    return {"rules": rules, "rules_t": rules_t, "canon": canon, "inverse_maps": inverse_maps,
            "bottom_map": bottom_map, "base_ms": base_ms, "controller": controller,
            "generator0": generator, "reader": reader, "read_acc": read_acc,
            "hold": hold, "value0": value, "replay": vb, "leaves_pool": leaves_np,
            "roots_pool": roots_np, "truth": truth, "n_blocks": n_blocks, "length": length}


# --------------------------------------------------------------------------- #
# [spiral] the depth-6 transplant: one complete shared dict, asserted
# --------------------------------------------------------------------------- #

def _spiral_shared(cfg, device, eras=None):
    """`tall/_tall_shared`'s idiom over THIS file's `build_shared`.

    `probe_clean` is populated by every round's ENTRYPOINT and by no builder, so importing the
    loop without the entrypoint silently omits it and `plant_probe` is the first thing to
    notice — after the setup has already been paid. `tall` lost two ~510 s setups to exactly
    this class of drift. Constructed here once, so every entrypoint in this file gets an
    identical and complete dict, and the key set is asserted rather than hoped for.

    NOTE ON `n_corrupt`, which is the whole densification lever `tall/dens0` measured: it is a
    cfg key read by `collect_value_buffer` INSIDE `build_shared`, so setting `n_corrupt=1` in
    the cfg densifies the stale value buffer at construction. `dens0` had to retrain a value
    head afterwards because it was sweeping variants; there is nothing to sweep any more, so
    the densified buffer is simply the buffer this file builds."""
    import torch
    shared = build_shared(cfg, device)
    shared["probe_clean"] = torch.from_numpy(
        _sample_pool(shared["rules"], cfg["n_probe_clean"], cfg["s"], cfg["seed"] + 31337)[1])
    shared["stale_random_blocks"] = float(shared["replay"]["y"].mean())

    # [spiral] TASK-MATCHED COLLECTION DAMAGE, adopted for every arm uniformly.
    #
    # `collect_value_buffer` corrupts `n_corrupt` RANDOM blocks anywhere in the sequence, while
    # every graded instance the agent ever meets is corrupted by `corrupt_hier` at one era's
    # cell. At 64 tokens those are very different distributions, and the value head's job is to
    # rank states drawn from the second having been fitted on the first. `tall` named this the
    # one untested densification lever; Phase A measured the buffer's terminal success at
    # 0.1541 task-matched against 0.1269 random-block (1.21x), at 41 s.
    #
    # Matched to the FIRST era's cell, not to a mixture of all five: the mixture would fold in
    # the L4/L5 cells, where nothing succeeds, and LOWER the positive rate — the opposite of
    # densification. Era 1's cell is the distribution the run starts in; the buffer is meant to
    # be a stale prior that the value then adapts away from online, which is the design.
    #
    # Uniform across arms by construction (it is built once, before any arm runs), so in-tag
    # ranks — which are this round's claims — are unaffected by the change of distribution.
    if cfg.get("collect_task_matched") and eras:
        t0 = time.time()
        vb = _collect_task_matched(shared, cfg, eras[0], device,
                                   n_episodes=cfg.get("tm_episodes") or cfg["value_episodes"])
        rate = float(vb["y"].mean())
        print(f"  task-matched buffer at {eras[0]['name']}: {vb['x'].shape[0]} states, "
              f"terminal success {rate:.4f} (random-block "
              f"{shared['stale_random_blocks']:.4f}); retraining the stale value head "
              f"({time.time() - t0:.0f}s to collect)", flush=True)
        value = _build_value_head()(cfg["state_dim"], cfg["v"]).to(device)
        opt = torch.optim.AdamW(value.parameters(), lr=cfg["value_lr"], weight_decay=1e-4)
        empty = {"x": vb["x"][:0], "r": vb["r"][:0], "y": vb["y"][:0]}
        value_steps(value, opt, shared["controller"], vb, empty,
                    n_steps=cfg["value_steps"], batch=512, replay_frac=0.0, device=device,
                    rng=np.random.default_rng(cfg["train_seed"] + 606_061))
        shared["value0"] = value
        shared["replay"] = vb
        shared["stale_task_matched"] = rate

    missing = [k for k in SHARED_KEYS if k not in shared]
    assert not missing, f"shared dict is missing {missing} — interface drift"
    return shared


def _install_identity_miner(support):
    """`tall`'s instrument, re-derived here (not imported: `tall` installs it on the shared
    `MC.Miner` class and this file must not depend on `tall`'s import side effects).

    `recital`'s mechanism check could see the mined table's SIZE and recall but never its
    MEMBERSHIP, so "did the table change or only grow?" was unanswerable. `Miner.state()` is
    already called once per cycle per level and logged verbatim, so extending it reaches
    exactly the missing data without touching the arm loop. Idempotent."""
    if getattr(MC.Miner, "_spiral_identity", False):
        return
    orig = MC.Miner.state

    def state(self):
        out = orig(self)
        out["keys_at_support"] = sorted(
            [int(x) for x in k] for k, c in self.counts.items() if c >= support)
        return out

    MC.Miner.state = state
    MC.Miner._spiral_identity = True


# --------------------------------------------------------------------------- #
# the arm loop
# --------------------------------------------------------------------------- #

ARMS = {
    # --- ratchet's five, verbatim: the untreated references / twins ------------------------
    "never_base":     {"vocab": "base",   "commit": None,    "prop_k": None, "span": False},
    "given":          {"vocab": "true",   "commit": None,    "prop_k": None, "span": False},
    "practice_gated": {"vocab": "earned", "commit": "delta", "prop_k": None, "span": False},
    "practice_early": {"vocab": "earned", "commit": "early", "prop_k": None, "span": False},
    "practice_late":  {"vocab": "earned", "commit": "late",  "prop_k": None, "span": False},
    # --- Port 1 alone (phase 1's arms, kept so a composed arm can be read against both
    #     single-port results in the same tag if wanted) ---------------------------------
    "given_prop_k4":         {"vocab": "true",   "commit": None,    "prop_k": 4, "span": False},
    "practice_late_prop_k4": {"vocab": "earned", "commit": "late",  "prop_k": 4, "span": False},
    # --- THE POISON TWIN. `practice_early` freezes a 1-entry bogus level-2 table at c1;
    #     ratchet measured that this forecloses level 3's REPRESENTATION. The routed version
    #     asks the SPEC's question: does a natively-routed bad L2 foreclose worse, because the
    #     policy now calls the address instead of merely having it available? Routing only —
    #     the corridor port is not the variable here.
    "practice_early_prop_k4": {"vocab": "earned", "commit": "early", "prop_k": 4, "span": False},
    # --- Port 2 alone -----------------------------------------------------------------------
    "span_true":  {"vocab": "true",   "commit": None,   "prop_k": None, "span": True},
    "span_mined": {"vocab": "earned", "commit": "late", "prop_k": None, "span": True},
    # --- THE FULL NATIVE PRIMITIVE: proposed as one call (Port 1) and run as one pass
    #     (Port 2). k = 4 is Port 1's measured sweet spot — exactly budget-matched to `given`'s
    #     enumeration at G = 58 (57 groundings per solve either way).
    "given_native":         {"vocab": "true",   "commit": None,   "prop_k": 4, "span": True},
    "practice_late_native": {"vocab": "earned", "commit": "late", "prop_k": 4, "span": True},
    # --- THE COMPOSED FIDELITY ARM. Both ports present and wired, both nailed shut: the
    #     proposal filter at k = n_moves (an exact no-op) and the span loss off with the parity
    #     gate unreachable. Must reproduce `given` bit-for-bit for 90 cycles, which is what
    #     licenses reading the composed file's other arms at all.
    "given_fid": {"vocab": "true", "commit": None, "prop_k": -1, "span": True,
                  "cfg": {"span_lam": 0.0, "span_tau": 2.0}},
    # ---------------------------------------------------------------------------------------
    # [spiral] THE SPIRAL ARMS. Everything above is the donor's and is untouched, so a donor
    # arm run out of this file is byte-for-byte the donor (that is `fidelity_d4`'s assertion).
    # These carry the transplanted commit policy — certify-else-provisional-at-boundary, with
    # the live recert — which is the arc's approved policy for this substrate and is held
    # FIXED across the spiral arms so vocabulary-acquisition timing is not a free variable.
    # ---------------------------------------------------------------------------------------
    # the non-native crank: live earning, no ports. `tall`'s original question, in the regime
    # it needed.
    "enum_live":    {"vocab": "earned", "commit": "delta_prov", "prop_k": None, "span": False,
                     "recert": True},
    # which port carries the effect: routing only.
    "spiral_route": {"vocab": "earned", "commit": "delta_prov", "prop_k": 4, "span": False,
                     "recert": True},
    # THE TREATMENT: earn -> consolidate into planner AND executor -> earn the next level
    # natively over the routed policy.
    "spiral":       {"vocab": "earned", "commit": "delta_prov", "prop_k": 4, "span": True,
                     "recert": True},
    # the native ceiling: nothing has to be earned, both ports live. (An alias of the donor's
    # `given_native`, kept because the SPEC's arm table names it separately.)
    "given_sp":     {"vocab": "true",   "commit": None,         "prop_k": 4, "span": True},
    # THE UNCONFOUNDED ENUM CRANK. `enum_live` at G=482 takes a PERMANENT width collapse at the
    # L2 commit (n 32->48; holding width 2 needs G >= 722), measured in Phase A as +0.172 in
    # `e` that never recovers. That handicaps its L3 EARNING trajectory, which would inflate
    # any native-vs-enum rate comparison in the native arms' favour for an apparatus reason.
    # This arm pays the un-handicapped price so the rate claim has a clean enum reference;
    # priced-TIME claims stay inside the G=482 set, where the ledger is matched.
    # ---------------------------------------------------------------------------------------
    # [census] THE CENSUS ARMS. Routing-only throughout (`spiral_route`'s configuration): the
    # both-ports consumption gap is unresolved and letting it into this round would confound
    # the coverage readout with the span question.
    # ---------------------------------------------------------------------------------------
    # the ROUTING-ONLY ceiling: true tables, proposal head, no corridor. `given_native`'s
    # both-ports bracket stays available cross-tag in `sp_s0`.
    "given_route":   {"vocab": "true",   "commit": None,         "prop_k": 4, "span": False},
    # the zero-timing-price op: commit at the certificate, then EXTEND the frozen table.
    "census_extend": {"vocab": "earned", "commit": "delta_prov", "prop_k": 4, "span": False,
                      "recert": True, "extend": True},
    # the gate-later op, at L2, as a CONJUNCTION with the certificate.
    "census_gate":   {"vocab": "earned", "commit": "delta_prov", "prop_k": 4, "span": False,
                      "recert": True, "cfg": {"gate_level": 2}},
    # the timing control: same commit cycle, no signal.
    "yoked_delay":   {"vocab": "earned", "commit": "yoked",      "prop_k": 4, "span": False,
                      "recert": True, "cfg": {"gate_level": 2}},
    # ---------------------------------------------------------------------------------------
    # [assay] THE SURGERY ARMS. All six are routing-only (`spiral_route`'s configuration) and
    # all but `given_c1` are the anchor exactly, differing only in what the commit installs.
    # ---------------------------------------------------------------------------------------
    "anchor":    {"vocab": "earned", "commit": "delta_prov", "prop_k": 4, "span": False,
                  "recert": True},
    "strip":     {"vocab": "earned", "commit": "delta_prov", "prop_k": 4, "span": False,
                  "recert": True, "cfg": {"surgery": "strip"}},
    "complete":  {"vocab": "earned", "commit": "delta_prov", "prop_k": 4, "span": False,
                  "recert": True, "cfg": {"surgery": "complete"}},
    "exact":     {"vocab": "earned", "commit": "delta_prov", "prop_k": 4, "span": False,
                  "recert": True, "cfg": {"surgery": "exact"}},
    "junk_dose": {"vocab": "earned", "commit": "delta_prov", "prop_k": 4, "span": False,
                  "recert": True, "cfg": {"surgery": "junk_dose"}},
    # the c1-arrival ceiling, in-tag: a `given_route` replica, so it carries `given`'s stream
    # exactly as `given_route` did in `cs_s0` — which keeps the anchor -> given_c1 bracket the
    # same shape of comparison as `cs_s0`'s spiral_route -> given_route bracket.
    "given_c1":  {"vocab": "true", "commit": None, "prop_k": 4, "span": False},
    # [assay] STREAM-DISPLACED TWINS. Identical to their originals in every config bit; the
    # only difference is a controlled burn of the per-arm RNG stream. They bound the
    # arrival-vs-stream confound the `exact` / `given_c1` residual has to be read against.
    "given_c1_j": {"vocab": "true", "commit": None, "prop_k": 4, "span": False,
                   "cfg": {"stream_burn": 1024}},
    "exact_j":    {"vocab": "earned", "commit": "delta_prov", "prop_k": 4, "span": False,
                   "recert": True, "cfg": {"surgery": "exact", "stream_burn": 1024}},
    # ---------------------------------------------------------------------------------------
    # [conductor] THE OUTER-LOOP ARMS. All six are routing-only (`spiral_route`'s
    # configuration), all carry the recert, all twin onto the SAME stream as the anchor, and all
    # differ from the anchor in exactly one thing: WHO DRIVES THE CRANK. The three driven arms
    # differ from each other in exactly one thing: WHAT THE RULE READS.
    #
    # `commit="loop"` means the outer loop owns the commit, not the certificate and not the era
    # boundary. The shadow certificate keeps running read-only in every arm, so cycles-to-cert
    # stays observable beside what the loop actually did — which is the comparison the round is
    # for. A loop commit is NOT provisional: the loop chose it, so it pays the pre-commit
    # audition, exactly as a certified commit does.
    # ---------------------------------------------------------------------------------------
    # the WITHIN-LEVEL reader: its own error on the era's own cell. `teacher_slot`'s `outer_task`
    # analogue — the arm whose currency is the one the crossing's value is NOT denominated in.
    "outer_ledger": {"vocab": "earned", "commit": "loop", "prop_k": 4, "span": False,
                     "recert": True, "loop": {"kind": "quiet", "read": "ledger"}},
    # ONE LEVEL UP, free: distinct level-(active+1) tuples at support, mined from the agent's own
    # chosen trajectories by the observation panel.
    "outer_yield": {"vocab": "earned", "commit": "loop", "prop_k": 4, "span": False,
                    "recert": True, "loop": {"kind": "quiet", "read": "yield"}},
    # ONE LEVEL UP, label-free and priced: the plant's own masked-infill NLL over the span one
    # level above the era's damage cell (`endo_yield`'s read, ported).
    "outer_endo": {"vocab": "earned", "commit": "loop", "prop_k": 4, "span": False,
                   "recert": True, "loop": {"kind": "quiet", "read": "endo", "priced": True}},
    # THE MANDATORY TIMING CONTROLS (census finding 4): same actions, same cycles, no signal.
    "yoked_yield": {"vocab": "earned", "commit": "loop", "prop_k": 4, "span": False,
                    "recert": True, "loop": {"kind": "yoke", "of": "outer_yield"}},
    "yoked_endo": {"vocab": "earned", "commit": "loop", "prop_k": 4, "span": False,
                   "recert": True, "loop": {"kind": "yoke", "of": "outer_endo"}},
    # ---------------------------------------------------------------------------------------
    # [maestro] THE LEARNED ARMS. Identical to `outer_yield` in every respect except that the
    # rule is `policy.LearnedPolicy` rather than `QuietPolicy`, and identical TO EACH OTHER in
    # every respect except the reward the offline fit was run against. Both consume the same
    # three-gauge input stream and both therefore pay the same priced endo read, so the ledger
    # column cannot separate them either — the reward's TYPE is the only difference.
    # ---------------------------------------------------------------------------------------
    "learned_yield": {"vocab": "earned", "commit": "loop", "prop_k": 4, "span": False,
                      "recert": True,
                      "loop": {"kind": "learned", "reward": "yield", "priced": True}},
    "learned_task": {"vocab": "earned", "commit": "loop", "prop_k": 4, "span": False,
                     "recert": True,
                     "loop": {"kind": "learned", "reward": "task", "priced": True}},
    # the mandatory timing control on the NEW machinery: the learned policy is new code inside
    # the cycle loop, so A1's structural argument that a read is inert has to be re-verified
    # live rather than inherited (census finding 4).
    "yoked_learned": {"vocab": "earned", "commit": "loop", "prop_k": 4, "span": False,
                      "recert": True, "loop": {"kind": "yoke", "of": "learned_yield"}},
    # ---------------------------------------------------------------------------------------
    # [maestro] THE STREAM-DISPLACED TWINS (`assay`'s mechanic, which came along in the fork
    # chain unchanged). Each is its original in EVERY config bit, with the per-arm RNG stream
    # displaced by a controlled 1024-draw burn on both the CPU and the CUDA generator. So
    # |original - displaced| is what stream POSITION alone moves, measured rather than assumed.
    #
    # Why these two and not the whole tag: the learned-vs-thermostat era-5 delta (+0.082) sits
    # just under the measured earning-family stream floor (0.087), which is exactly the cell
    # census finding 7 says is not readable from one draw. The twins turn that pairwise
    # ordering into a question with an answer — does the rank survive displacement — and hand
    # back a second draw of each arm's anchor-delta at the same time.
    #
    # They take their ORIGINAL's stream key in `TWIN` below, so the base seed is identical and
    # the burn is the only thing that moves.
    # ---------------------------------------------------------------------------------------
    "outer_yield_j": {"vocab": "earned", "commit": "loop", "prop_k": 4, "span": False,
                      "recert": True, "loop": {"kind": "quiet", "read": "yield"},
                      "cfg": {"stream_burn": 1024}},
    "learned_yield_j": {"vocab": "earned", "commit": "loop", "prop_k": 4, "span": False,
                        "recert": True,
                        "loop": {"kind": "learned", "reward": "yield", "priced": True},
                        "cfg": {"stream_burn": 1024}},
    "enum_live_g722": {"vocab": "earned", "commit": "delta_prov", "prop_k": None,
                       "span": False, "recert": True,
                       # G=722 is what holds width 2 at n=48; `width_cap=2` stops the same
                       # budget ALSO buying width 3 at n=32 (measured), which would make the
                       # control differ from `enum_live` through era 1 as well and confound the
                       # cycles-to-certification comparison the arm exists to clean up.
                       "cfg": {"g_budget": 722, "width_cap_ref_g": 482}},
    # ---------------------------------------------------------------------------------------
    # [crescendo] THE A3 ARMS. All four are routing-only, all carry the recert, all twin onto
    # the anchor's stream, and all run at `max_macro_level=4` — so they share the proposal
    # head's slot layout, the level-4 miner, the observation panel and every RNG draw. They
    # differ in exactly the three things the round is about: WHO PACES (schedule vs loop),
    # WHETHER L4 MAY BE COMMITTED, and WHETHER THE FROZEN TABLE MAY BE EXTENDED.
    # ---------------------------------------------------------------------------------------
    # the scheduled crank at the new rung, and the round's LIFETIME CEILING: `crescendo_run`
    # sets its scheduled era lengths equal to the caps, so it runs the longest life any arm may
    # have and a loop arm can only be shorter. A1's and A2's lifetime confound, closed by
    # construction rather than caveated.
    "anchor_long":     {"vocab": "earned", "commit": "delta_prov", "prop_k": 4, "span": False,
                        "recert": True},
    # THE TREATMENT: A1's thermostat, verbatim, with L4 committable.
    "outer_yield_m4":  {"vocab": "earned", "commit": "loop", "prop_k": 4, "span": False,
                        "recert": True, "loop": {"kind": "quiet", "read": "yield"}},
    # THE CEILING CONTROL, and the pair the signature is read on. A clock replay of the
    # treatment's realised actions with the L4 commit FORBIDDEN, so it holds the treatment's
    # era boundaries cycle-for-cycle and its L2/L3 commits cycle-for-cycle, and differs in one
    # bit. It is also this round's non-invasiveness check: it must be BIT-IDENTICAL to the
    # treatment up to the treatment's own L4 commit cycle, which the reduction asserts.
    # A gauge-driven `commit_max_level=3` twin would have the same pacing FREEDOM but not the
    # same realised pacing (the quiet latch that licenses the forbidden commit would license an
    # era advance instead, block (g4)), which is precisely the lifetime confound this arm
    # exists to avoid; it is the round's named deferred fifth arm.
    "ceiling_m3":      {"vocab": "earned", "commit": "loop", "prop_k": 4, "span": False,
                        "recert": True,
                        "loop": {"kind": "yoke", "of": "outer_yield_m4"},
                        "cfg": {"commit_max_level": 3}},
    # THE r**2-WALL BYPASS: the treatment plus `census_extend`'s post-commit extension, which
    # the recert loop runs at every committed level in every era. Phase 0 measured the wall
    # binds and that extension is the lever that lifts it — buildable L4 over the FROZEN L3 vs
    # over the LIVE L3 is 15 vs 25 (`cd_s0/outer_yield`), 13 vs 26 (`ma_s0/learned_yield`),
    # 8 vs 13 (`anchor`) — so extension roughly doubles what an L4 commit can hold.
    "outer_yield_m4x": {"vocab": "earned", "commit": "loop", "prop_k": 4, "span": False,
                        "recert": True, "extend": True,
                        "loop": {"kind": "quiet", "read": "yield"}},
    # THE FREE-PACING CEILING CONTROL. Gauge-driven like the treatment, with L4 forbidden — so
    # it has the treatment's pacing FREEDOM but not its realised pacing: the quiet latch that
    # licenses the forbidden commit licenses an era advance instead (block (g4)), so this arm
    # leaves era 3 where the treatment committed. That is a lifetime difference, which is why
    # `ceiling_m3` and not this arm is the pair the signature is read on. What THIS arm adds is
    # the other half: what the rung's existence did to the loop's own PACING.
    "outer_yield_m3":  {"vocab": "earned", "commit": "loop", "prop_k": 4, "span": False,
                        "recert": True, "loop": {"kind": "quiet", "read": "yield"},
                        "cfg": {"commit_max_level": 3}},
    # ---------------------------------------------------------------------------------------
    # [crescendo] THE STREAM-DISPLACED TWINS OF THE SIGNATURE PAIR (`as_s1`/`ma_s1`'s mechanic).
    # Identical to their originals in every config bit; the only difference is a controlled
    # 1024-draw burn of the per-arm CPU and CUDA streams. They take their ORIGINAL's stream key
    # (below), so the base seed is identical and the burn is the only thing that moves.
    #
    # THE PAIR IS REBUILT ON THE DISPLACED DRAW, which is the one design point that matters
    # here: on draw B the treatment's own action cycles move, so a `ceiling_m3_j` yoked to the
    # ORIGINAL treatment would be neither lifetime-identical to `outer_yield_m4_j` nor
    # one-bit-differing from it — it would be a third thing. `ceiling_m3_j` therefore yokes to
    # `outer_yield_m4_j` and carries the same burn, so the draw-B pair reproduces the draw-A
    # construction exactly: same stream, same era boundaries, same L2/L3 commit cycles, and one
    # bit of difference at L4. Arm ORDER is load-bearing — the treatment must run first.
    "outer_yield_m4_j": {"vocab": "earned", "commit": "loop", "prop_k": 4, "span": False,
                         "recert": True, "loop": {"kind": "quiet", "read": "yield"},
                         "cfg": {"stream_burn": 1024}},
    "ceiling_m3_j":     {"vocab": "earned", "commit": "loop", "prop_k": 4, "span": False,
                         "recert": True,
                         "loop": {"kind": "yoke", "of": "outer_yield_m4_j"},
                         "cfg": {"commit_max_level": 3, "stream_burn": 1024}},
    # ---------------------------------------------------------------------------------------
    # [tacet] THE GATE ARMS (E3'). The baseline is A3's treatment `outer_yield_m4` VERBATIM —
    # not a renamed copy — so this tag's ungated arm is the same object `cr3_s0` measured and
    # the full-life cross-tag replay is a real gate rather than a re-definition.
    #
    # EVERY gate arm is a CLOCK YOKE of that baseline (`ceiling_m3`'s mechanic, one level of
    # abstraction up: there it isolated one bit of the COMMIT RULE, here it isolates one knob
    # of the LEARNING RULE). The reason is the one A3 wrote down: the outer loop's actions are
    # absorbing, so an arm that learns from a different diet would also commit and advance at
    # different cycles, and the era-4/5 read would conflate the gate's content with the gate's
    # pacing. Yoked, the arms are lifetime-identical, era-boundary-identical and
    # commit-cycle-identical, and differ in exactly one knob. What the yoke costs is the
    # pacing half of the question, and that is bought back offline for free: A1's thermostat
    # replayed on each arm's OWN logged L4 at-support series says when it would have committed
    # had it been driving (`phase0_l4.py`'s machinery, in the reduction).
    #
    # All five carry `gate_frac` from the run config; only `gate_mode` differs between the
    # four gates, and `gate_all` carries no gate at all — it relaxes the donor's own two knobs.
    # ---------------------------------------------------------------------------------------
    "gate_delta_hi": {"vocab": "earned", "commit": "loop", "prop_k": 4, "span": False,
                      "recert": True, "loop": {"kind": "yoke", "of": "outer_yield_m4"},
                      "cfg": {"gate_mode": "delta_hi"}},
    "gate_delta_lo": {"vocab": "earned", "commit": "loop", "prop_k": 4, "span": False,
                      "recert": True, "loop": {"kind": "yoke", "of": "outer_yield_m4"},
                      "cfg": {"gate_mode": "delta_lo"}},
    "gate_delib":    {"vocab": "earned", "commit": "loop", "prop_k": 4, "span": False,
                      "recert": True, "loop": {"kind": "yoke", "of": "outer_yield_m4"},
                      "cfg": {"gate_mode": "delib"}},
    "gate_random":   {"vocab": "earned", "commit": "loop", "prop_k": 4, "span": False,
                      "recert": True, "loop": {"kind": "yoke", "of": "outer_yield_m4"},
                      "cfg": {"gate_mode": "random"}},
    # LEARN FROM EVERYTHING, in the widest form each channel admits (see the module docstring,
    # item 6): mining keeps every solved answer instead of `mine_cap` of them, and pi is
    # supervised on every surviving tip instead of only the solved ones. Both are DONOR knobs.
    "gate_all":      {"vocab": "earned", "commit": "loop", "prop_k": 4, "span": False,
                      "recert": True, "loop": {"kind": "yoke", "of": "outer_yield_m4"},
                      "cfg": {"mine_cap": 0, "prop_train_on": "tips"}},
    # THE INERTNESS ARM, for `preflight` only: a pure yoke with no gate and no relaxation, so
    # it must be BIT-IDENTICAL to `outer_yield_m4`. It is what licenses reading every arm above
    # as "the baseline plus one knob" rather than "the baseline plus a yoke plus one knob".
    "gate_off_y":    {"vocab": "earned", "commit": "loop", "prop_k": 4, "span": False,
                      "recert": True, "loop": {"kind": "yoke", "of": "outer_yield_m4"}},
    # ---------------------------------------------------------------------------------------
    # [intonation] THE delta_perf ARMS. Every one is A3's treatment `outer_yield_m4` — same
    # thermostat, same reads, same ladder — PLUS the span head, firing below parity, PLUS the
    # meter. `perf_log` is the clock source and consumes nothing; the other four are clock
    # yokes of it carrying exactly one knob each, so they are lifetime-, era-boundary- and
    # commit-cycle-identical (`tacet`'s reasoning, which is `crescendo`'s one level up).
    #
    # `perf_log` is deliberately NOT `outer_yield_m4` itself: turning the executor on changes
    # the trajectory, so this tag's baseline is its own object and the cross-tag replay against
    # `tc_s0`/`cr3_s0` is a bounded gate (the arms are bit-identical until the span loss's
    # first optimizer step after the L2 commit) rather than a full-life one.
    # ---------------------------------------------------------------------------------------
    "perf_log":     {"vocab": "earned", "commit": "loop", "prop_k": 4, "span": True,
                     "recert": True,
                     "loop": {"kind": "quiet", "read": "yield"},
                     "cfg": {"span_tau_fire": 0.50, "perf_meter": True}},
    "perf_gain":    {"vocab": "earned", "commit": "loop", "prop_k": 4, "span": True,
                     "recert": True, "loop": {"kind": "yoke", "of": "perf_log"},
                     "cfg": {"span_tau_fire": 0.50, "perf_meter": True,
                             "perf_gain": "delta"}},
    "perf_raw":     {"vocab": "earned", "commit": "loop", "prop_k": 4, "span": True,
                     "recert": True, "loop": {"kind": "yoke", "of": "perf_log"},
                     "cfg": {"span_tau_fire": 0.50, "perf_meter": True,
                             "perf_gain": "raw"}},
    "perf_gate":    {"vocab": "earned", "commit": "loop", "prop_k": 4, "span": True,
                     "recert": True, "loop": {"kind": "yoke", "of": "perf_log"},
                     "cfg": {"span_tau_fire": 0.50, "perf_meter": True,
                             "gate_mode": "perf_hi"}},
    "outcome_gate": {"vocab": "earned", "commit": "loop", "prop_k": 4, "span": True,
                     "recert": True, "loop": {"kind": "yoke", "of": "perf_log"},
                     "cfg": {"span_tau_fire": 0.50, "perf_meter": True,
                             "gate_mode": "delta_hi"}},
    # PREFLIGHT ONLY. `perf_off_y` is the yoke's inertness check with the meter off and the
    # firing threshold at the donor's tau; `perf_fid` is the span head wired and the meter ON
    # but firing at tau = 0.95 — i.e. the donor's own executor with an observer attached, which
    # is how gate I-2 asserts that metering alone changes no behaviour.
    # ---------------------------------------------------------------------------------------
    # [caesura] delta-SILENCE AS A COMMIT INPUT. Four arms, one seed, `intonation`'s executor
    # and meter in all of them (so `dsil` exists in every arm's panel and every counterfactual
    # is computable offline everywhere), differing ONLY in what paces the commit.
    #
    #   dsil_yield   A1's thermostat on `yield` — `intonation`'s `perf_log` verbatim. The
    #                comparator, the clock reference, and the arm whose logged `dsil` series
    #                gives the ungated counterfactual.
    #   dsil_read    the SAME thermostat, same span/W/burn/alpha, reading `dsil` INSTEAD of
    #                `yield`. delta-silence CHOOSES the crossing. This is the arm that asks
    #                whether a within-level execution signal can license a crossing at all —
    #                `teacher_slot` measured the within-level LEDGER refusing exactly that, and
    #                b(s) is a different within-level quantity, so the refusal is a question
    #                and not a prediction.
    #   dsil_and     the yield thermostat chooses, delta-silence may only DELAY (`dsil_veto`).
    #                The arm that separates the metering signal's seat: TIMING, not choice.
    #   dsil_sched   the schedule (`delta_prov`, certify-else-boundary), so the two treatments
    #                are read against a pacer that reads nothing at all as well as against A1's.
    #
    # NOT yoked, and that is the round's one deliberate departure from `intonation`: the
    # question here IS the pacing, so pinning the arms to a common clock would delete the
    # variable. The cost is the one A1/A2 carried and named — the arms are not
    # lifetime-matched, so deep-era deltas conflate pacing with practice time — and it is
    # bought back the way A3 bought it back: `dsil_sched`'s scheduled era lengths are the caps,
    # so it is the LIFETIME CEILING and no loop arm can outrun it.
    "dsil_yield":   {"vocab": "earned", "commit": "loop", "prop_k": 4, "span": True,
                     "recert": True,
                     "loop": {"kind": "quiet", "read": "yield"},
                     "cfg": {"span_tau_fire": 0.50, "perf_meter": True}},
    "dsil_read":    {"vocab": "earned", "commit": "loop", "prop_k": 4, "span": True,
                     "recert": True,
                     "loop": {"kind": "quiet", "read": "dsil"},
                     "cfg": {"span_tau_fire": 0.50, "perf_meter": True,
                             "dsil_bootstrap": True}},
    "dsil_and":     {"vocab": "earned", "commit": "loop", "prop_k": 4, "span": True,
                     "recert": True,
                     "loop": {"kind": "quiet", "read": "yield"},
                     "cfg": {"span_tau_fire": 0.50, "perf_meter": True, "dsil_veto": True}},
    "dsil_sched":   {"vocab": "earned", "commit": "delta_prov", "prop_k": 4, "span": True,
                     "recert": True,
                     "cfg": {"span_tau_fire": 0.50, "perf_meter": True}},
    # PREFLIGHT ONLY, and load-bearing there for the same reason `perf_given` was in
    # `intonation`: a loop arm cannot arm on a forty-step substrate (the donor's own gate-C
    # note — "the loop arms ride their caps here"), so neither the gauge's DRIVING path nor the
    # veto would ever execute. `dsil_pf_gauge` holds the true tables, which mints every slot at
    # c1 and makes `dsil` live from c2, so `QuietPolicy("dsil")` is stepped on real reads.
    # `dsil_pf_veto` commits on the SCHEDULE with seeded miners, so commits actually fire and
    # the veto is consulted — before the first commit the gauge does not exist (the `absent`
    # branch, which must be inert) and after it does (the `defer`/`pass` branches).
    "dsil_pf_boot": {"vocab": "earned", "commit": "loop", "prop_k": 4, "span": True,
                     "recert": True, "loop": {"kind": "quiet", "read": "dsil"},
                     "cfg": {"span_tau_fire": 0.50, "perf_meter": True,
                             "dsil_bootstrap": True}},
    "dsil_pf_gauge": {"vocab": "true", "commit": None, "prop_k": 4, "span": True,
                      "loop": {"kind": "quiet", "read": "dsil"},
                      "cfg": {"span_tau_fire": 0.50, "perf_meter": True}},
    "dsil_pf_veto":  {"vocab": "earned", "commit": "late", "prop_k": 4, "span": True,
                      "recert": True,
                      "cfg": {"span_tau_fire": 0.50, "perf_meter": True, "dsil_veto": True}},
    # ---------------------------------------------------------------------------------------
    # [tutti] THE UNIFICATION ARMS. Eight paid, one seed. `intonation`'s executor and meter in
    # every one (so `dsil` exists in every panel and every counterfactual is computable
    # everywhere), and the QUESTION PORT ON in every one (so the selection compute — the reader
    # forward and the value forward over all K menu candidates — is taken by every arm and used
    # only by the arms whose selector reads it; `ostinato`'s rung-invariant-comparator
    # discipline). `question_mode="exo"` is the menu's HEAD in the donor's own RNG order, i.e.
    # NO selection, and a bit-identical replay of the donor. `question_mode=None` is used only
    # inside G-F.
    #
    # The two axes:
    #
    #   COMMIT / ADVANCE assignment      SELECTOR
    #     yield  / yield   (b)             exo   (no selection; the replay gate + baseline row)
    #     dsil   / dsil    (a)             endo  (half-key novelty x delivery ledger — the only
    #     dsil   / yield   (c) THE SPLIT           selector that can be an outer-loop ACTION)
    #     yield  / dsil    the MIRROR
    #
    # The MIRROR exists because "any two-gauge mix beats one" is a live alternative reading of a
    # split win under the simultaneous-reset rule: it gives each signal the OTHER's seat, so a
    # split win that is about the ASSIGNMENT separates from one that is about having two latches.
    #
    # NOT yoked on the pacing axis, and that is the donor's deliberate departure carried
    # forward: the question here IS the pacing, so pinning the pacers to a common clock would
    # delete the variable. The lifetime confound is bought back A3's way — the donor's
    # `ca_s0/dsil_sched` (201 cycles at the caps) is the lifetime ceiling no loop arm can
    # outrun, borrowed cross-tag and licensed by `tu_y_exo`'s own 0.000e+00 replay.
    # The SELECTOR axis IS yoked: `tu_s_yk` replays `tu_s_endo`'s realised commit AND advance
    # cycles with the selector off — lifetime-identical, one bit differing (`crescendo`'s
    # `ceiling_m3` idiom), which separates selection CONTENT from the clock selection induces.
    "tu_y_exo":     {"vocab": "earned", "commit": "loop", "prop_k": 4, "span": True,
                     "recert": True,
                     "loop": {"kind": "quiet", "read": "yield"},
                     "cfg": {"span_tau_fire": 0.50, "perf_meter": True,
                             "question_mode": "exo"}},
    "tu_d_exo":     {"vocab": "earned", "commit": "loop", "prop_k": 4, "span": True,
                     "recert": True,
                     "loop": {"kind": "quiet", "read": "dsil"},
                     "cfg": {"span_tau_fire": 0.50, "perf_meter": True,
                             "dsil_bootstrap": True, "question_mode": "exo"}},
    # (c) THE SPLIT — the QUEUE's stated shape. `loop` owns the ADVANCE, `loop_commit` owns the
    #     COMMIT. delta-silence gets the compile-trigger seat
    #     (`practice_manufactures_its_own_credit` S1 component (3) designed it for) and the
    #     one-level-up yield gauge keeps the crossing-of-eras it has always owned.
    "tu_s_exo":     {"vocab": "earned", "commit": "loop", "prop_k": 4, "span": True,
                     "recert": True,
                     "loop": {"kind": "quiet", "read": "yield"},
                     "loop_commit": {"kind": "quiet", "read": "dsil"},
                     "cfg": {"span_tau_fire": 0.50, "perf_meter": True,
                             "dsil_bootstrap": True, "question_mode": "exo"}},
    "tu_s_endo":    {"vocab": "earned", "commit": "loop", "prop_k": 4, "span": True,
                     "recert": True,
                     "loop": {"kind": "quiet", "read": "yield"},
                     "loop_commit": {"kind": "quiet", "read": "dsil"},
                     "cfg": {"span_tau_fire": 0.50, "perf_meter": True,
                             "dsil_bootstrap": True, "question_mode": "endo"}},
    # THE ONE-BIT CLOCK YOKE of the composed arm. Runs immediately after it and replays its
    # MEASURED actions (commits and advances, including cap-forced ones) by clock and by
    # nothing else, with the selector OFF. Unlike `ceiling_m3` the pair is NOT bit-identical up
    # to a divergence cycle — the selector acts at c1 — so the twin-window gate is replaced by
    # X-5's exact-replay gate.
    "tu_s_yk":      {"vocab": "earned", "commit": "loop", "prop_k": 4, "span": True,
                     "recert": True, "loop": {"kind": "yoke", "of": "tu_s_endo"},
                     "cfg": {"span_tau_fire": 0.50, "perf_meter": True,
                             "question_mode": "exo"}},
    "tu_d_endo":    {"vocab": "earned", "commit": "loop", "prop_k": 4, "span": True,
                     "recert": True,
                     "loop": {"kind": "quiet", "read": "dsil"},
                     "cfg": {"span_tau_fire": 0.50, "perf_meter": True,
                             "dsil_bootstrap": True, "question_mode": "endo"}},
    "tu_y_endo":    {"vocab": "earned", "commit": "loop", "prop_k": 4, "span": True,
                     "recert": True,
                     "loop": {"kind": "quiet", "read": "yield"},
                     "cfg": {"span_tau_fire": 0.50, "perf_meter": True,
                             "question_mode": "endo"}},
    # THE MIRROR: each signal in the other's seat. yield licenses the COMMIT, delta-silence the
    # ADVANCE. delta-silence still cannot speak before a slot is open, so the ADVANCE owner
    # needs the same fallback the commit owner needs — `dsil_bootstrap` covers both sites.
    "tu_m_exo":     {"vocab": "earned", "commit": "loop", "prop_k": 4, "span": True,
                     "recert": True,
                     "loop": {"kind": "quiet", "read": "dsil"},
                     "loop_commit": {"kind": "quiet", "read": "yield"},
                     "cfg": {"span_tau_fire": 0.50, "perf_meter": True,
                             "dsil_bootstrap": True, "question_mode": "exo"}},
    # PREFLIGHT ONLY, for the reason `dsil_pf_*` and `perf_given` exist: a loop arm cannot arm
    # on a forty-step substrate, so neither the split's driving path nor the port's selectors
    # would ever execute at preflight sizes.
    #   tu_pf_split  true tables -> every slot minted at c1 -> `dsil` live from c2, so BOTH
    #                policies are stepped on real reads and gate X-1/X-4 bite.
    #   tu_pf_boot   the split with EARNED tables -> exercises the split's bootstrap branch
    #                (bootstrapped commits alongside yield-driven advances), gate X-3.
    #   tu_pf_q      the endogenous selector with true tables, so the port's agent bundle,
    #                the quota and the delivery ledger all execute.
    #   tu_pf_yk     a yoke of `tu_pf_boot`, so X-5's two-policy replay is exercised.
    #   tu_pf_off    `loop_commit` absent and the port off -> gate X-2's inertness twin.
    "tu_pf_split":  {"vocab": "true", "commit": "loop", "prop_k": 4, "span": True,
                     "loop": {"kind": "quiet", "read": "yield"},
                     "loop_commit": {"kind": "quiet", "read": "dsil"},
                     "cfg": {"span_tau_fire": 0.50, "perf_meter": True,
                             "dsil_bootstrap": True, "question_mode": "exo"}},
    "tu_pf_boot":   {"vocab": "earned", "commit": "loop", "prop_k": 4, "span": True,
                     "recert": True,
                     "loop": {"kind": "quiet", "read": "yield"},
                     "loop_commit": {"kind": "quiet", "read": "dsil"},
                     "cfg": {"span_tau_fire": 0.50, "perf_meter": True,
                             "dsil_bootstrap": True, "question_mode": "exo"}},
    "tu_pf_q":      {"vocab": "true", "commit": "loop", "prop_k": 4, "span": True,
                     "loop": {"kind": "quiet", "read": "yield"},
                     "loop_commit": {"kind": "quiet", "read": "dsil"},
                     "cfg": {"span_tau_fire": 0.50, "perf_meter": True,
                             "dsil_bootstrap": True, "question_mode": "endo"}},
    "tu_pf_yk":     {"vocab": "earned", "commit": "loop", "prop_k": 4, "span": True,
                     "recert": True, "loop": {"kind": "yoke", "of": "tu_pf_boot"},
                     "cfg": {"span_tau_fire": 0.50, "perf_meter": True,
                             "question_mode": "exo"}},
    "tu_pf_off":    {"vocab": "earned", "commit": "loop", "prop_k": 4, "span": True,
                     "recert": True,
                     "loop": {"kind": "quiet", "read": "yield"},
                     "cfg": {"span_tau_fire": 0.50, "perf_meter": True}},
    # ---------------------------------------------------------------------------------------
    # [intonation/A] THE STRONGLY-METERED REGIME. Same arms, same knobs, one regime: the run
    # config buys FEWER GRADED PERFORMANCES PER CYCLE (`n_pr` 64 -> 24, `pr_width` 16 -> 8), so
    # useful experience is scarce relative to what the learner needs. `in_s0` measured the
    # abundance it is being contrasted with: 14.9 solved instances/cycle against a mining cap of
    # 8 (the cap bound on 95.4% of cycles) and a pi replay buffer that filled at cycle 26 of 131
    # and stayed full. Nothing about the WORLD moves — same grammar, same damage ladder, same
    # action set, same budget per plan, same floors — only how many performances the meter can
    # afford to grade. Two arms carry the round's own fixes (`perf_mean`, `rawx`); the other two
    # are `in_s0`'s arms verbatim so the regime is the only thing that changed.
    "mperf_log":    {"vocab": "earned", "commit": "loop", "prop_k": 4, "span": True,
                     "recert": True,
                     "loop": {"kind": "quiet", "read": "yield"},
                     "cfg": {"span_tau_fire": 0.50, "perf_meter": True}},
    "mperf_gain":   {"vocab": "earned", "commit": "loop", "prop_k": 4, "span": True,
                     "recert": True, "loop": {"kind": "yoke", "of": "mperf_log"},
                     "cfg": {"span_tau_fire": 0.50, "perf_meter": True,
                             "perf_gain": "delta"}},
    "mperf_gate":   {"vocab": "earned", "commit": "loop", "prop_k": 4, "span": True,
                     "recert": True, "loop": {"kind": "yoke", "of": "mperf_log"},
                     "cfg": {"span_tau_fire": 0.50, "perf_meter": True,
                             "gate_mode": "perf_mean"}},
    "mout_gate":    {"vocab": "earned", "commit": "loop", "prop_k": 4, "span": True,
                     "recert": True, "loop": {"kind": "yoke", "of": "mperf_log"},
                     "cfg": {"span_tau_fire": 0.50, "perf_meter": True,
                             "gate_mode": "delta_hi"}},
    "mperf_rawx":   {"vocab": "earned", "commit": "loop", "prop_k": 4, "span": True,
                     "recert": True, "loop": {"kind": "yoke", "of": "mperf_log"},
                     "cfg": {"span_tau_fire": 0.50, "perf_meter": True,
                             "perf_gain": "rawx"}},
    # PREFLIGHT ONLY, and load-bearing there: a loop arm cannot commit on a forty-step
    # substrate (the thermostat correctly never arms — see gate C's note), so no slot is ever
    # MINTED and the whole firing path would go untested. `perf_given` holds the true tables,
    # which mints every slot at c1, so the fallible executor, the meter, the 2x2 and both
    # consumers all execute before a paid setup. `given`'s stream, so it perturbs nothing.
    "perf_given":   {"vocab": "true", "commit": None, "prop_k": 4, "span": True,
                     "cfg": {"span_tau_fire": 0.50, "perf_meter": True}},
    "perf_given_g": {"vocab": "true", "commit": None, "prop_k": 4, "span": True,
                     "cfg": {"span_tau_fire": 0.50, "perf_meter": True,
                             "perf_gain": "delta"}},
    "perf_off_y":   {"vocab": "earned", "commit": "loop", "prop_k": 4, "span": True,
                     "recert": True, "loop": {"kind": "yoke", "of": "perf_log"},
                     "cfg": {"span_tau_fire": None}},
    "perf_fid":     {"vocab": "earned", "commit": "loop", "prop_k": 4, "span": True,
                     "recert": True, "loop": {"kind": "yoke", "of": "perf_log"},
                     "cfg": {"span_tau_fire": None, "perf_meter": True}},
}
ARMS["fid"] = dict(ARMS["given_fid"])          # the SPEC's name for the in-tag assertion

# A treated arm and its untreated twin must share a torch RNG stream, or the contrast is
# confounded by stream position (`handle/`'s discipline). Streams are keyed by the TWIN.
TWIN = {
    "given_prop_k4": "given", "span_true": "given", "given_native": "given",
    "given_fid": "given",
    "practice_late_prop_k4": "practice_late", "span_mined": "practice_late",
    "practice_late_native": "practice_late",
    "practice_early_prop_k4": "practice_early",
    # [spiral] the treated spiral arms twin onto `enum_live`, which is their untreated
    # baseline: same vocabulary policy, same commit policy, no ports. `given_sp`/`fid` twin
    # onto `given` exactly as `given_native`/`given_fid` do.
    "spiral_route": "enum_live", "spiral": "enum_live",
    "given_sp": "given", "fid": "given",
    # `enum_live_g722` twins onto `enum_live`: they differ in `g_budget` alone, and at n=32
    # both buy width 2, so they are bit-identical until the L2 commit and diverge EXACTLY at
    # the width transition the arm exists to isolate.
    "enum_live_g722": "enum_live",
    # [census] every treated census arm twins onto the SAME stream as `spiral_route` (the in-tag
    # anchor), so each differs from the anchor by exactly one thing and by nothing else.
    "census_extend": "enum_live", "census_gate": "enum_live", "yoked_delay": "enum_live",
    "given_route": "given",
    # [assay] every surgery arm twins onto `spiral_route`'s stream, so each differs from the
    # anchor by the surgery and by nothing else.
    "anchor": "enum_live", "strip": "enum_live", "complete": "enum_live",
    "exact": "enum_live", "junk_dose": "enum_live",
    "given_c1": "given",
    # the displaced twins take their ORIGINAL's stream key, so the base seed is identical and
    # the burn is the only thing that moves.
    "given_c1_j": "given", "exact_j": "enum_live",
    # [conductor] every loop-driven and yoked arm twins onto the anchor's stream, so each is
    # BIT-IDENTICAL TO THE ANCHOR UNTIL ITS FIRST ACTION — an in-tag gate the reduction checks,
    # and the strongest available statement that the reads themselves are non-invasive.
    "outer_ledger": "enum_live", "outer_yield": "enum_live", "outer_endo": "enum_live",
    "yoked_yield": "enum_live", "yoked_endo": "enum_live",
    # [maestro] the learned arms take the SAME stream as the anchor and as `outer_yield`, so
    # every arm in the tag is bit-identical to the anchor until its own first action.
    "learned_yield": "enum_live", "learned_task": "enum_live", "yoked_learned": "enum_live",
    # [maestro] the displaced twins take their ORIGINAL's stream key, so the base seed is
    # identical and the 1024-draw burn is the only thing that moves.
    "outer_yield_j": "enum_live", "learned_yield_j": "enum_live",
    # [crescendo] the A3 arms take the anchor's stream too, so every arm in the tag is
    # bit-identical to `anchor_long` until its own first action — the in-tag twin gate.
    "anchor_long": "enum_live", "outer_yield_m4": "enum_live",
    "ceiling_m3": "enum_live", "outer_yield_m4x": "enum_live",
    "outer_yield_m3": "enum_live",
    # [crescendo] the displaced twins take their ORIGINAL's stream key.
    "outer_yield_m4_j": "enum_live", "ceiling_m3_j": "enum_live",
    # [tacet] every gate arm takes the anchor's stream, exactly as the baseline does, so each
    # differs from `outer_yield_m4` by its gate knob and by nothing else.
    "gate_delta_hi": "enum_live", "gate_delta_lo": "enum_live", "gate_delib": "enum_live",
    "gate_random": "enum_live", "gate_all": "enum_live", "gate_off_y": "enum_live",
    # [intonation] every delta_perf arm takes the anchor's stream too, so each is bit-identical
    # to `anchor_long`/`outer_yield_m4` until its own port switches on (the span head is minted
    # off its OWN generator — gate S-1 — so its mere existence costs the shared stream nothing).
    "perf_log": "enum_live", "perf_gain": "enum_live", "perf_raw": "enum_live",
    "perf_gate": "enum_live", "outcome_gate": "enum_live",
    "perf_off_y": "enum_live", "perf_fid": "enum_live",
    "perf_given": "given", "perf_given_g": "given",
    "dsil_yield": "enum_live", "dsil_read": "enum_live", "dsil_and": "enum_live",
    "dsil_sched": "enum_live", "dsil_pf_gauge": "given", "dsil_pf_veto": "enum_live",
    "dsil_pf_boot": "enum_live",
    # [tutti] every unification arm takes the anchor's stream, so each is bit-identical to
    # its family's carrier until its own first action (the in-tag twin gate). The endo arms
    # diverge at c1 BY CONSTRUCTION — the selector acts on cycle 1 — which is why their
    # lifetime control is the yoke and not a twin window.
    "tu_y_exo": "enum_live", "tu_d_exo": "enum_live", "tu_s_exo": "enum_live",
    "tu_s_endo": "enum_live", "tu_s_yk": "enum_live", "tu_d_endo": "enum_live",
    "tu_y_endo": "enum_live", "tu_m_exo": "enum_live",
    "tu_pf_split": "given", "tu_pf_boot": "enum_live", "tu_pf_q": "given",
    "tu_pf_yk": "enum_live", "tu_pf_off": "enum_live",
    "mperf_log": "enum_live", "mperf_gain": "enum_live", "mperf_gate": "enum_live",
    "mout_gate": "enum_live", "mperf_rawx": "enum_live",
}
STREAM = {"never_base": 0, "given": 1, "practice_gated": 2, "practice_early": 3,
          "practice_late": 4,
          # [spiral] a NEW stream index, so the spiral arms neither perturb nor inherit the
          # donor arms' stream position.
          "enum_live": 5}


def parse_arms(spec):
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        base, *rest = part.split(":")
        ov = {}
        for kv in rest:
            k, val = kv.split("=")
            try:
                ov[k] = int(val)
            except ValueError:
                try:
                    ov[k] = float(val)
                except ValueError:
                    ov[k] = val
        label = base + ("" if not ov else "_" + "_".join(f"{k}{x}" for k, x in ov.items()))
        out.append((label, base, ov))
    return out


def macro_moves(level, tables, s, depth, device):
    """Instantiate the level-`level` macro at EVERY node of that level, so a committed
    vocabulary is a position-independent rule set (which is what the DGP's own tables are)
    and the committed action space matches `given`'s move-for-move."""
    n_nodes = s ** (depth - level)
    return [MC.to_device(MC.make_macro(level, j, s, tables[level]), device)
            for j in range(n_nodes)]


def build_ms(base_ms, committed, s, depth, device):
    ms = list(base_ms)
    for level in sorted(committed):
        if committed[level] is None:
            continue
        ms += macro_moves(level, {level: committed[level]}, s, depth, device)
    return ms


# =========================================================================== #
# [assay] INSTRUMENT: per-execution entry identity
# =========================================================================== #

# On-device accumulator. `on` gates recording; `phase` tags which part of the cycle the
# executions came from, so beam executions are separable from instrument ones (auditions,
# probes, the ablation battery). Never touches `counts` and never draws from any RNG.
_ENTRY_REC = {"on": False, "phase": "beam", "acc": {}}


def _rec_reset():
    _ENTRY_REC["acc"] = {}


def _rec_take(device=None):
    """Pull the accumulator to host once, and clear it. One sync per cycle, not per call."""
    out = {}
    for (phase, lvl), t in _ENTRY_REC["acc"].items():
        out.setdefault(phase, {})[str(lvl)] = [int(x) for x in t.cpu().numpy()]
    _rec_reset()
    return out


def _rng_snapshot():
    """[assay] Capture every global RNG stream the arm loop's substrate depends on."""
    import torch
    st = {"torch": torch.get_rng_state(), "numpy": np.random.get_state()}
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def _rng_restore(st):
    import torch
    torch.set_rng_state(st["torch"])
    np.random.set_state(st["numpy"])
    if "cuda" in st and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st["cuda"])


def macro_features_rec(generator, x, move, s, v):
    """A byte-identical copy of `macros.macro_features`, plus the one line the donor throws
    away: which table entry the max-sum DP actually chose, per row.

    Copied rather than wrapped because the donor returns `(feats, pos)` and discards `best`;
    wrapping would mean recomputing the whole DP to recover it. `entry_recorder_check()`
    asserts this copy reproduces the donor's outputs exactly, and the G-F gate covers it
    end-to-end. The recording is a `bincount` into a device tensor — no host sync, no RNG, no
    effect on the returned values.
    """
    import torch
    blk0, span = move["blk0"], move["span"]
    table = move["table"]
    batch = x.shape[0]
    blocks = torch.arange(blk0, blk0 + span, device=x.device)
    pos = (blocks[:, None] * s + torch.arange(s, device=x.device)[None, :]).reshape(-1)
    pos = pos[None, :].expand(batch, -1).contiguous()

    obs = x.clone().scatter_(1, pos, torch.full_like(pos, -1))
    logits = generator.block_logits(obs)
    cur = logits[:, blk0:blk0 + span, :]
    if move["level"] == 1:
        return cur.argmax(-1), pos

    chain = move["chain"]
    for child in chain:
        n_par = cur.shape[1] // s
        kids = cur.view(batch, n_par, s, cur.shape[-1])
        total = None
        for i in range(s):
            picked = kids[:, :, i, :].index_select(2, child[:, i].contiguous())
            total = picked if total is None else total + picked
        cur = total
    best = cur.argmax(-1)
    flat = move["flat"]
    # ---- [assay] the instrument, and nothing else -------------------------------------- #
    if _ENTRY_REC["on"]:
        n = int(flat.shape[0])
        key = (_ENTRY_REC["phase"], int(move["level"]))
        acc = _ENTRY_REC["acc"].get(key)
        if acc is None or acc.shape[0] != n:
            acc = torch.zeros(n, dtype=torch.long, device=x.device)
            _ENTRY_REC["acc"][key] = acc
        acc.index_add_(0, best.reshape(-1),
                       torch.ones(best.numel(), dtype=torch.long, device=x.device))
    return flat[best.reshape(-1)].view(batch, span), pos


def _install_entry_recorder():
    """Swap the recording copy in for the donor's. `macros.apply_any` resolves
    `macro_features` as a module global at call time, so this reaches every macro execution
    everywhere without touching `macros.py`. Idempotent."""
    if getattr(MC, "_assay_entry_rec", False):
        return
    MC._assay_macro_features_orig = MC.macro_features
    MC.macro_features = macro_features_rec
    MC._assay_entry_rec = True


def entry_recorder_check(v=8, s=2, depth=6, m=2, max_level=3, n=48, rule_seed=0):
    """Assert the recording copy reproduces the donor's `macro_features` BIT FOR BIT, on every
    macro of a real action set, with the recorder both off and on — the precondition for the
    fidelity gates meaning anything once the instrument is installed."""
    import torch
    from rhm.rhm_data import generate_rules_distinct
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0); np.random.seed(0)
    length = s ** depth
    rules = generate_rules_distinct(v, s, depth, m, seed=rule_seed)
    gen = _build_generator()(v, length, s, 96, n_head=4, n_layer=2,
                             root_conditioned=False).to(device)
    gen.eval()
    roots_np, leaves_np = _sample_pool(rules, n, s, 3)
    x = torch.from_numpy(_corrupt(leaves_np, length // s, 3, v, s,
                                  np.random.default_rng(5))).to(device)
    truth = MC.true_tables(rules, depth, s, v, m, max_level)
    ms = build_ms(build_move_set(depth, s, max_level=1),
                  {ell: truth[ell] for ell in range(2, max_level + 1)}, s, depth, device)
    orig = getattr(MC, "_assay_macro_features_orig", MC.macro_features)
    out = {"n_macros": 0, "max_abs_feats": 0.0, "max_abs_pos": 0.0, "n_exec_recorded": 0}
    for rec_on in (False, True):
        _ENTRY_REC["on"] = rec_on
        _rec_reset()
        for mv in ms:
            if mv.get("kind") != "macro":
                continue
            with torch.no_grad():
                fa, pa = orig(gen, x, mv, s, v)
                fb, pb = macro_features_rec(gen, x, mv, s, v)
            out["n_macros"] += 1
            out["max_abs_feats"] = max(out["max_abs_feats"],
                                       float((fa.long() - fb.long()).abs().max()))
            out["max_abs_pos"] = max(out["max_abs_pos"],
                                     float((pa.long() - pb.long()).abs().max()))
        if rec_on:
            taken = _rec_take()
            out["n_exec_recorded"] = int(sum(sum(h) for lv in taken.values()
                                             for h in lv.values()))
            out["recorded_levels"] = sorted({k for lv in taken.values() for k in lv})
    _ENTRY_REC["on"] = False
    _rec_reset()
    assert out["max_abs_feats"] == 0.0 and out["max_abs_pos"] == 0.0, \
        f"the recording copy of macro_features is not bit-identical: {out}"
    assert out["n_exec_recorded"] > 0, "the entry recorder recorded nothing"
    # each execution must attribute exactly one entry per row, per macro call
    want = out["n_macros"] // 2 * n
    assert out["n_exec_recorded"] == want, \
        f"recorded {out['n_exec_recorded']} selections, expected {want}"
    out["verdict"] = "PASS"
    return out


def gy_soundness(s=2, depth=6, level=4, eras=None, v=8):
    """[census] THE G-Y INSTRUMENT'S SOUNDNESS CHECK — a known-composition toy, hand-enumerated.

    The spiral README promised the next-level-yield readout CONDITIONALLY on the instrument
    being mechanically sound, so the check is part of the record rather than an assumption.
    Three things are asserted, all at sizes small enough to count by hand:

      1. SPAN. A level-`level` miner keys on `s**(level-1)` level-1 features — 8 blocks at
         level 4, s=2. Wrong span means the instrument is silently counting a different object.
      2. COUNTS. Feed a hand-built array with a known multiset of rows and check `n_distinct`,
         `n_obs` and `n_at_support` at every threshold against the hand count. This is what
         catches a keying bug (e.g. keying on positions instead of values, or collapsing rows).
      3. NODE INDEXING. For every era of the real ladder, the level-`level` node the instrument
         reads must be in range (`s**(depth-level)` nodes exist) and its column slice must lie
         inside a real configuration's block count.
    """
    span = s ** (level - 1)
    out = {"level": level, "span_blocks": span, "expected_span": span}
    mn = MC.Miner(level, s)
    assert mn.span == span, f"G-Y span is {mn.span}, expected {span}"

    # 2. hand-enumerated counts: A x3, B x2, C x1 -> 3 distinct, 6 observations
    A = list(range(span))                       # 0,1,2,...,span-1
    B = [0] * span
    C = [1] + [0] * (span - 1)                  # differs from B in one position only
    rows = np.array([A, B, A, C, B, A], dtype=np.int64)
    mn.observe(rows)
    st = mn.state()
    hand = {"n_obs": 6, "n_distinct": 3,
            "n_at_support": {"1": 3, "2": 2, "3": 1, "5": 0, "10": 0}}
    out["hand"] = hand
    out["measured"] = {"n_obs": st["n_obs"], "n_distinct": st["n_distinct"],
                       "n_at_support": st["n_at_support"]}
    assert st["n_obs"] == hand["n_obs"], f"G-Y n_obs {st['n_obs']} != {hand['n_obs']}"
    assert st["n_distinct"] == hand["n_distinct"], (
        f"G-Y n_distinct {st['n_distinct']} != {hand['n_distinct']} — B and C differ in exactly "
        f"one position, so a keying bug that collapsed them would show here")
    for t, want in hand["n_at_support"].items():
        got = st["n_at_support"][t]
        assert got == want, f"G-Y n_at_support[{t}] = {got}, hand count {want}"

    # the identity instrument, if installed, must report the same key set
    if "keys_at_support" in st:
        keys3 = [tuple(k) for k in st["keys_at_support"]]
        assert keys3 == [tuple(A)], f"G-Y keys_at_support at support 3 = {keys3}, expected [A]"
        out["keys_at_support_3"] = keys3

    # 3. node indexing over the REAL ladder
    n_nodes = s ** (depth - level)
    n_blocks = s ** depth // s
    nodes = []
    for era in (eras or parse_eras(CENSUS_LADDER)):
        node = (era["node"] * s ** (era["level"] - 1)) // span
        ok = (0 <= node < n_nodes) and ((node + 1) * span <= n_blocks)
        nodes.append({"era": era["name"], "level": era["level"], "node": era["node"],
                      f"L{level}_node": int(node), "in_range": bool(ok)})
        assert ok, (f"G-Y node {node} out of range for {era['name']} "
                    f"({n_nodes} nodes at level {level}, {n_blocks} blocks)")
    out["nodes_per_era"] = nodes
    out["n_nodes_at_level"] = n_nodes
    # 4. the instrument is DECOUPLED from what may be committed
    out["decoupled_from_max_macro_level"] = bool(level > 3)
    out["verdict"] = "PASS"
    return out


# =========================================================================== #
# [conductor] THE ENDO READ — endo_yield's label-free one-level-up loss, ported
# =========================================================================== #

def parent_span_blocks(era, s, depth):
    """The BLOCK range of the span ONE LEVEL ABOVE this era's damage cell.

    RHM's tree is balanced, so this is pure arithmetic on the grammar's SHAPE (`s`, `depth`)
    and the era's own cell — both of which the agent has, because the cell is the world it is
    living in. No table, no rule set, no truth mask enters. This is
    `endo_yield/SPEC.md` §3's `pos_top_level` argument, re-expressed for a masked-block plant:
    the cell at (level l, node n) covers leaves [n*s^l, (n+1)*s^l); its parent is
    (l+1, n//s) and covers twice as many, and a block is `s` leaves."""
    pl, pn = era["level"] + 1, era["node"] // s
    lo, hi = pn * s ** pl, (pn + 1) * s ** pl
    length = s ** depth
    lo, hi = max(0, min(lo, length)), max(0, min(hi, length))
    return int(lo // s), int(hi // s)


def cell_span_blocks(era, s, depth):
    """The block range of the era's OWN damage cell — the level-matched within-level control,
    logged in the shadow panel and never driven."""
    lo, hi = era["node"] * s ** era["level"], (era["node"] + 1) * s ** era["level"]
    length = s ** depth
    lo, hi = max(0, min(lo, length)), max(0, min(hi, length))
    return int(lo // s), int(hi // s)


def endo_read(generator, x, bottom_map, blocks, *, v, s, n_blocks):
    """THE PLANT'S OWN MASKED-INFILL NLL over a block range: the learner's own loss, on its own
    data, at positions fixed by tree arithmetic.

    The WHOLE range is masked and the WHOLE range is scored. That is what makes it a one-level-
    up read rather than a within-level one: with the parent span entirely unobserved, recovering
    it requires the level-(l+1) latent inferred from OUTSIDE the span, where masking only the
    cell requires the level-l latent. The target is `bottom_map`'s level-1 features — the plant's
    own training objective, optimised by every arm in every cycle — so nothing here is an oracle
    the loop would not otherwise have.

    Deterministic and non-invasive by construction: a fixed batch, no sampling, `no_grad`, and
    the net left in whatever mode the cycle put it in. It consumes no draw of any RNG, which is
    what lets the shadow panel run in EVERY arm without voiding the fidelity gate."""
    import torch
    import torch.nn.functional as F
    b0, b1 = blocks
    if b1 <= b0:
        return None
    b = x.shape[0]
    powers = v ** torch.arange(s, device=x.device)
    feats = bottom_map[(x.view(b, n_blocks, s) * powers).sum(-1)]
    obs = x.clone()
    obs[:, b0 * s:b1 * s] = -1
    with torch.no_grad():
        logits = generator.block_logits(obs)
    lg = logits[:, b0:b1, :].reshape(-1, logits.shape[-1])
    tg = feats[:, b0:b1].reshape(-1)
    ok = tg >= 0
    if int(ok.sum()) == 0:
        return None
    with torch.no_grad():
        return float(F.cross_entropy(lg[ok], tg[ok]).item())


def endo_gate(s=2, depth=6, eras=None):
    """N-1: THE READ IS LABEL-FREE BY CONSTRUCTION, checked the way `endo_yield` checked its
    own — by asserting what the read's POSITIONS depend on, and that they land where the tree
    says they land.

    `parent_span_blocks` is a closed form in (era level, era node, s, depth) and touches no
    rules array, no `truth`, no `inverse_maps` and no committed table. The check enumerates the
    ladder's cells and asserts (i) the parent strictly contains the cell, (ii) it is exactly `s`
    times as wide, (iii) it is in range, and (iv) it is the cell's parent under integer node
    arithmetic."""
    eras = eras or parse_eras(CENSUS_LADDER)
    rows = []
    for i, era in enumerate(eras):
        cb = cell_span_blocks(era, s, depth)
        pb = parent_span_blocks(era, s, depth)
        wide = (pb[1] - pb[0]) == s * (cb[1] - cb[0]) or (pb[1] - pb[0]) == s ** depth // s
        assert pb[0] <= cb[0] and cb[1] <= pb[1], f"era {i+1}: parent does not contain the cell"
        assert 0 <= pb[0] < pb[1] <= s ** depth // s, f"era {i+1}: parent out of range"
        assert wide, f"era {i+1}: parent is not s x the cell ({pb} vs {cb})"
        rows.append({"era": i + 1, "cell": era["name"], "cell_blocks": list(cb),
                     "parent_level": era["level"] + 1, "parent_node": era["node"] // s,
                     "parent_blocks": list(pb),
                     "parent_is_root": bool(pb == (0, s ** depth // s))})
    # The check is on the read's CODE, not its prose: parse the function, drop its docstring,
    # and assert that no oracle name appears in what actually executes. `endo_read` is checked
    # the same way — it may touch `bottom_map` (the plant's own training target, which every arm
    # optimises every cycle) and nothing else that carries grammar content.
    def _body(fn):
        import ast
        tree = ast.parse(inspect.getsource(fn))
        fdef = tree.body[0]
        if (fdef.body and isinstance(fdef.body[0], ast.Expr)
                and isinstance(fdef.body[0].value, ast.Constant)
                and isinstance(fdef.body[0].value.value, str)):
            fdef.body = fdef.body[1:]
        return ast.unparse(tree)

    banned = ("truth", "rules", "inverse_maps", "committed", "canon", "true_mask", "grade")
    checked = {}
    for fn in (parent_span_blocks, cell_span_blocks, endo_read):
        body = _body(fn)
        hits = [b for b in banned if b in body]
        assert not hits, f"{fn.__name__} touches {hits} — the read is not label-free"
        checked[fn.__name__] = sorted(set(body.split()) & {"bottom_map", "s", "depth"})
    return {"verdict": "PASS", "depends_on": ["era.level", "era.node", "s", "depth"],
            "code_checked": sorted(checked), "banned_absent": list(banned), "rows": rows}


def endo_bench(shared, cfg, device, n, reps=3):
    """N-2: WHAT THE READ COSTS, measured on the same GPU rather than assumed — the donor's
    `slot_lm.bench` convention. A GROUNDING in this substrate is `score one state` (an encoder
    pass plus a value read, which is what `beam_moves` charges one of per materialised child),
    so the endo read's price is quoted in grounding-equivalents: the wall time of one
    `block_logits` forward over `n` sequences, divided by the per-state cost of the encode the
    beam pays for."""
    import torch
    x = shared["probe_clean"][:n].to(device)
    gen = shared["generator0"]
    ctl = shared["controller"]

    def _t(fn):
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(reps):
            fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return (time.time() - t0) / reps

    obs = x.clone()
    obs[:, :2] = -1
    t_endo = _t(lambda: torch.no_grad()(gen.block_logits)(obs))
    t_ground = _t(lambda: torch.no_grad()(_encode_chunked)(ctl, x))
    per_state = t_ground / max(n, 1)
    price = max(1, int(round(t_endo / per_state))) if per_state > 0 else n
    return {"n": int(n), "t_endo_s": t_endo, "t_ground_batch_s": t_ground,
            "t_per_grounding_s": per_state, "price_g": int(price),
            "note": "grounding-equivalents per endo read, measured on this GPU"}


def floor_gate(cfg):
    """The dead zones that governed this run, checked against `floors.json` so the literals in
    `MEASURED_FLOORS` and the derivation on disk cannot drift apart, and asserted non-default:
    a driven read with no measured floor is an error, not a silent zero."""
    out = {"used": {k: cfg[k] for k in ("tol_ledger", "tol_yield_l3", "tol_yield_l4",
                                        "tol_endo")},
           "provenance": MEASURED_FLOORS["provenance"]}
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "floors.json")
    if os.path.isfile(p):
        f = json.load(open(p))["floors"]
        same = (abs(f["ledger"] - cfg["tol_ledger"]) < 1e-12
                and abs(f["yield_by_level"]["3"] - cfg["tol_yield_l3"]) < 1e-12
                and abs(f["yield_by_level"]["4"] - cfg["tol_yield_l4"]) < 1e-12)
        out["matches_floors_json"] = bool(same)
        assert same, f"MEASURED_FLOORS has drifted from floors.json: {f} vs {out['used']}"
    for k in ("tol_ledger", "tol_yield_l3", "tol_yield_l4", "tol_endo"):
        assert cfg[k] and cfg[k] > 0, f"{k} is not a measured floor ({cfg[k]!r})"
    out["verdict"] = "PASS"
    return out


def fit_gate(cfg):
    """[maestro] THE FITTED POLICIES THAT GOVERN THIS RUN, checked against `fit.json` so the
    literals in `FITTED` and the offline derivation on disk cannot drift apart, and asserted
    well-formed: an unfitted learned rule, or one whose dead zone is not positive, is an error
    and not a silent zero. `floor_gate`'s pattern, applied to the object A2 adds.

    It also re-asserts the round's structural claim on the actual literals: the two twins must
    share the gauge set, the bucket set and the geometry, and differ ONLY in their fitted
    numbers. If that ever fails, the arms differ in more than the reward's type and the
    contrast is not the one the round is for."""
    fitted = cfg.get("fitted") or {}
    out = {"rewards": sorted(fitted), "mixes": {}, "thetas": {}}
    assert set(fitted) == {"yield", "task"}, f"expected both twins, got {sorted(fitted)}"
    for rw, f in fitted.items():
        assert f.get("mixes") and f.get("thetas"), f"{rw}: no fitted policy"
        for b, th in f["thetas"].items():
            assert th and th > 0, f"{rw}[{b}]: theta {th!r} is not a measured dead zone"
        out["mixes"][rw] = f["mixes"]
        out["thetas"][rw] = f["thetas"]
    a, b = fitted["yield"], fitted["task"]
    assert a["gauges"] == b["gauges"], "the twins read different gauges"
    assert sorted(a["mixes"]) == sorted(b["mixes"]), "the twins have different bucket sets"
    out["twins_share_class"] = True
    out["differ_only_in_numbers"] = bool(a["mixes"] != b["mixes"])
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fit.json")
    if os.path.isfile(p):
        d = json.load(open(p))
        same = True
        for rw in ("yield", "task"):
            g = d["fits"][rw]
            for bk in g["mixes"]:
                same &= all(abs(x - y) < 1e-12
                            for x, y in zip(g["mixes"][bk], fitted[rw]["mixes"][bk]))
                same &= abs(g["thetas"][bk] - fitted[rw]["thetas"][bk]) < 1e-12
        out["matches_fit_json"] = bool(same)
        out["fit_verdict"] = d.get("verdict")
        out["gate_f1"] = d.get("gate_f1", {}).get("ALL")
        assert same, "FITTED has drifted from fit.json — re-run fit.py"
        assert not d["verdict"]["vacuous"], (
            "fit.py judged the live contrast vacuous; halt and report rather than run")
    out["verdict"] = "PASS"
    return out


# =========================================================================== #
# [assay] THE REALISTIC JUNK POOL, and the surgery itself
# =========================================================================== #

CS_TAGS = [("rhm_practice_census", "cs_s0"), ("rhm_practice_spiral", "sp_s0")]


def build_junk_pool(truth, max_level, tags=None, data_dir=None):
    """Tuples the arc's own loops actually mined and that are FALSE — the kind of junk
    selection produces, not uniform random tuples.

    Read off `log["miner"][...]["keys_at_support"]` (the entry-identity instrument installed
    since `tall/`) in every arm of `census/cs_s0` and `spiral/sp_s0` on the volume, unioned,
    then keyed against `MC.true_tables`. Sorted for determinism; the pool and its provenance
    are logged verbatim in `setup.json`.
    """
    data_dir = data_dir or DATA_DIR
    pool, seen, prov = {ell: [] for ell in range(2, max_level + 1)}, set(), []
    for remote, tag in (tags or CS_TAGS):
        root = os.path.join(data_dir, remote, tag)
        if not os.path.isdir(root):
            prov.append({"tag": tag, "found": False})
            continue
        arms, n_keys = [], 0
        for arm in sorted(os.listdir(root)):
            fp = os.path.join(root, arm, "results.json")
            if not os.path.isfile(fp):
                continue
            res = json.load(open(fp))
            arms.append(arm)
            for cyc in res["log"]["miner"]:
                for ell in range(2, max_level + 1):
                    for k in (cyc.get(str(ell)) or {}).get("keys_at_support", []) or []:
                        t = tuple(int(x) for x in k)
                        if (ell, t) in seen:
                            continue
                        seen.add((ell, t))
                        n_keys += 1
                        pool[ell].append(t)
        prov.append({"tag": tag, "found": True, "arms": arms, "new_keys": n_keys})
    out, split = {}, {}
    for ell in range(2, max_level + 1):
        true_set = {tuple(int(x) for x in r) for r in truth[ell]["flat"]}
        false_keys = sorted(t for t in pool[ell] if t not in true_set)
        out[ell] = false_keys
        split[str(ell)] = {"mined_distinct": len(pool[ell]), "false": len(false_keys),
                           "true": len(pool[ell]) - len(false_keys)}
    return out, {"provenance": prov, "split": split}


def rows_from_flats(flats, lower, s, span):
    """Child-index rows over `lower` for each flat level-1 tuple; unbuildable ones dropped.

    Same idiom as `census.extend_candidates` and `macros.Miner.build`: an entry exists only if
    every one of its `s` halves is already an entry of the lower table (the ratchet)."""
    half = span // s
    lut = {tuple(int(x) for x in r): i for i, r in enumerate(lower["flat"])}
    rows, dropped = [], []
    for k in flats:
        k = tuple(int(x) for x in k)
        kids = [lut.get(k[i * half:(i + 1) * half]) for i in range(s)]
        if any(j is None for j in kids):
            dropped.append(list(k))
        else:
            rows.append(kids)
    return rows, dropped


def apply_surgery(mode, level, tbl, lower, truth_l, junk_pool, s, truth_full=None):
    """Replace the mined table `tbl` with the arm's constructed one. Returns (table, record).

    `lower` is the arm's OWN committed lower table, so every constructed entry is buildable
    over what that arm actually holds — the r-squared ratchet is respected, not bypassed. The
    only exception is `exact`, which installs the DGP's own table verbatim (whose lower is the
    DGP's own lower, and which that arm committed at the level below one era earlier), exactly
    as `vocab == "true"` does.
    """
    span = s ** (level - 1)
    own_flat = [tuple(int(x) for x in r) for r in tbl["flat"]]
    true_set = {tuple(int(x) for x in r) for r in truth_l["flat"]}
    own_child = [list(map(int, r)) for r in tbl["child"]]
    rec = {"mode": mode, "level": level, "n_own": len(own_child),
           "n_own_true": sum(1 for k in own_flat if k in true_set),
           "n_added": 0, "n_added_true": 0, "n_removed": 0, "n_unbuildable": 0}
    if mode in (None, "none"):
        return tbl, rec
    if mode == "exact":
        out = truth_full[level] if truth_full else truth_l
        rec.update({"n_after": int(out["child"].shape[0]), "n_removed": len(own_child),
                    "n_added": int(out["child"].shape[0]),
                    "n_added_true": int(out["child"].shape[0])})
        return out, rec
    if mode == "strip":
        keep = [c for c, k in zip(own_child, own_flat) if k in true_set]
        rec["n_removed"] = len(own_child) - len(keep)
        out = MC.make_table(level, np.asarray(keep, np.int64).reshape(-1, s), lower, s)
    elif mode in ("complete", "junk_dose"):
        missing = sorted(k for k in true_set if k not in own_flat)
        add_rows, dropped = rows_from_flats(missing, lower, s, span)
        rec["n_missing_true"] = len(missing)
        rec["K"] = len(add_rows)                 # `complete`'s addition count == the dose
        rec["n_unbuildable"] = len(dropped)
        if mode == "junk_dose":
            have = set(own_flat)
            cand = [k for k in (junk_pool.get(level) or []) if k not in have]
            add_rows, dropped2 = rows_from_flats(cand, lower, s, span)
            add_rows = add_rows[:rec["K"]]       # matched dose, deterministic order
            rec["n_junk_available_buildable"] = len(cand) - len(dropped2)
            rec["n_unbuildable"] = len(dropped2)
            rec["dose_shortfall"] = max(0, rec["K"] - len(add_rows))
        rec["n_added"] = len(add_rows)
        out = MC.make_table(level, np.asarray(own_child + add_rows,
                                              np.int64).reshape(-1, s), lower, s)
    else:
        raise ValueError(f"unknown surgery mode {mode}")
    out_flat = [tuple(int(x) for x in r) for r in out["flat"]]
    rec["n_after"] = len(out_flat)
    rec["n_added_true"] = sum(1 for k in out_flat if k in true_set) - rec["n_own_true"]
    rec["flat_after"] = [list(k) for k in out_flat]
    rec["true_mask_after"] = [int(k in true_set) for k in out_flat]
    rec.update({f"tab_{k}": val for k, val in MC.grade_table(out, truth_l).items()})
    return out, rec


def extend_candidates(miner, lower, support, frozen):
    """[census] The rows `Miner.build` would produce that `frozen` does not already hold, as
    child indices into `lower` — recomputed against the CURRENT lower table so the indices are
    correct by construction rather than inherited from a stale build.

    Ordered by (-count, key): the most-supported candidate is offered first, and the order is
    deterministic, so a greedy admission is reproducible.
    """
    half = miner.span // miner.s
    lut = {tuple(int(x) for x in row): i for i, row in enumerate(lower["flat"])}
    have = {tuple(int(x) for x in r) for r in frozen["flat"]}
    out = []
    for k, c in sorted(miner.counts.items(), key=lambda kv: (-kv[1], kv[0])):
        if c < support or k in have:
            continue
        kids = [lut.get(k[i * half:(i + 1) * half]) for i in range(miner.s)]
        if any(j is None for j in kids):
            continue
        out.append({"key": k, "child": kids, "count": int(c)})
    return out


def refresh_upper(committed, level, s):
    """[census] After level `level` is EXTENDED, rebuild level `level+1` over the new lower
    table so its newly available halves become usable.

    Safe because extension only ever APPENDS: existing rows keep their indices and their
    flattened level-1 expansion, so `level+1`'s stored child indices stay valid and its `flat`
    recomputes identically for every row it already had.
    """
    up = level + 1
    if committed.get(up) is None:
        return
    committed[up] = MC.make_table(up, committed[up]["child"], committed[level], s)


def _cert_summary(shadow_cert):
    """[spiral] the rate readout, reduced: for each earnable level, the cycle its shadow
    certificate first fired, the cycle its candidate series became non-empty (`c0` — the
    certificate cannot fire before the vocabulary exists, so cycles-to-cert is only meaningful
    net of it), and the trajectory the silence test was run on."""
    out = {}
    for ell, c in shadow_cert.items():
        out[str(ell)] = {"fired": c["fired"], "fired_era": c.get("fired_era"),
                         "fired_c_in_era": c.get("fired_c_in_era"),
                         "fired_at_active": c.get("fired_active"),
                         "first_candidate_cycle": c["c0"],
                         "cycles_to_cert": (None if (c["fired"] is None or c["c0"] is None)
                                            else int(c["fired"] - c["c0"] + 1)),
                         "ref": c["ref"], "emin": c["emin"], "run_at_end": int(c["run"]),
                         "hist": c["hist"]}
    return out


def audition_macro(generator, x0, roots_np, move, rules_t, canon, depth, v, m, s, rules):
    """The macro's own audition: ONE action on held-out instances of the current era, graded
    by terminal possible-set success. This is the committable content — the quantity the
    unit-LP certificate watches."""
    xf = MC.apply_any(generator, x0, move, rules_t, canon, depth, v, m, s)
    succ, dres = grade(xf.cpu().numpy(), roots_np, rules, s)
    return {"e": 1.0 - float(succ.mean()), "dres": float(dres.mean())}


def run_arm(label, base, overrides, shared, cfg, eras, refs, outdir, device):
    import torch
    arm = label
    spec = ARMS[base]
    cfg = {**cfg, **spec.get("cfg", {}), **overrides}
    v, s, depth, m = cfg["v"], cfg["s"], cfg["depth"], cfg["m"]
    rules, rules_t, canon = shared["rules"], shared["rules_t"], shared["canon"]
    base_ms = shared["base_ms"]
    controller = shared["controller"]
    maxl = cfg["max_macro_level"]
    # [crescendo] what may be COMMITTED, as against what may be mined/observed/slotted/audited.
    # `None` -> `maxl`, which is the donor exactly. The ceiling control sets it to 3 while
    # keeping `maxl` at 4, so it is the treatment in every other respect — same head layout,
    # same miners, same panel, same audition loop, same shared RNG draws.
    commit_maxl = int(cfg.get("commit_max_level") or maxl)
    assert commit_maxl <= maxl, (
        f"commit_max_level {commit_maxl} exceeds max_macro_level {maxl}: a level that is "
        f"never mined cannot be committed")

    # [spiral] An optional cap on the beam width the declared budget buys, expressed as a
    # REFERENCE BUDGET rather than as a number.
    #
    # It exists for exactly one arm and the reason is a measurement: `enum_live_g722` is meant
    # to be `enum_live` WITHOUT the permanent width collapse at the L2 commit (n 32->48 drops
    # width 2 -> 1 at G=482), and nothing else. But G=722 also buys width **3** at n=32, so
    # raising the budget alone would hand the arm a wider beam through the whole of era 1 too,
    # and the cycles-to-L2-certification comparison the arm exists to de-confound would itself
    # be confounded, in the control's favour.
    #
    # A literal cap would be wrong: `width_cap=2` is a constant of `budget=8` and silently
    # becomes a *handicap* at any other rollout budget (at budget 2 the matched arm runs width
    # 13, and a cap of 2 would make the control four times narrower than the arm it controls —
    # caught by `preflight`, which runs at budget 2). So the cap is stated as the RULE it is:
    # this arm may never run a wider beam than the matched-budget arm runs over the BASE action
    # set; it may only decline to collapse when the action set grows.
    _cap = (fit_width(len(base_ms), cfg["budget"], cfg["width_cap_ref_g"])
            if cfg.get("width_cap_ref_g") else None)

    def cap_w(w):
        return int(w) if not _cap else int(min(int(w), int(_cap)))

    # ---- PER-ARM torch STREAM. ratchet let the global stream run across arms, so an arm's
    #      trajectory depended on the arms before it. Keyed by the TWIN (see `TWIN`/`STREAM`),
    #      so a treated arm and its untreated baseline draw the identical sequence and can only
    #      diverge through the port itself. -------------------------------------------------
    stream_key = TWIN.get(base, base)
    torch.manual_seed(cfg["train_seed"] * 1000 + 7 + STREAM[stream_key])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg["train_seed"] * 1000 + 7 + STREAM[stream_key])
    # ---- [assay] THE STREAM-DISPLACEMENT BURN ------------------------------------------ #
    # A controlled draw-and-discard immediately after the per-arm seeding and before anything
    # else touches the arm: same seed, same config, same everything — a different position in
    # the same stream. `|original - displaced|` on the deep-era readouts is then a direct
    # measurement of what stream position ALONE moves at depth 6.
    #
    # BOTH generators are burned because both are consumed, every cycle, by the plant's
    # masking in `finetune_generator`: `torch.randint(1, n_blocks + 1, ())` draws on the CPU
    # generator and `torch.rand(b, n_blocks, device=device)` on the CUDA one, `gen_steps`
    # times per cycle. Burning only one would leave the other aligned and understate the
    # displacement. (Dropout is NOT a consumer here — every net in this substrate is built
    # with `dropout=0.0` — so the burn point was chosen by reading what actually draws rather
    # than by assuming training does.)
    burn = int(cfg.get("stream_burn") or 0)
    burn_rec = {"draws": burn, "cpu": False, "cuda": False,
                "point": "after per-arm torch/cuda seeding, before the generator deepcopy"}
    if burn:
        _ = torch.randint(0, 2 ** 31 - 1, (burn,))
        burn_rec["cpu"] = True
        if torch.cuda.is_available():
            _ = torch.rand(burn, device=device)
            burn_rec["cuda"] = True
        print(f"[burn]   arm={arm} displaced the per-arm stream by {burn} draws "
              f"(cpu={burn_rec['cpu']}, cuda={burn_rec['cuda']})", flush=True)

    # ---- per-arm plant + selector (the plant is a substrate property this round: every arm
    #      carries it, and the value co-adapts against it) ----------------------------------
    generator = copy.deepcopy(shared["generator0"]).to(device)
    for p in generator.parameters():
        p.requires_grad_(True)
    gopt = torch.optim.AdamW(generator.parameters(), lr=cfg["gen_lr"], weight_decay=1e-4)
    value = copy.deepcopy(shared["value0"]).to(device)
    for p in value.parameters():
        p.requires_grad_(True)
    vopt = torch.optim.AdamW(value.parameters(),
                             lr=cfg["value_lr_online"] or cfg["value_lr"], weight_decay=1e-4)
    rng = np.random.default_rng(cfg["seed"] + 77)
    grng = np.random.default_rng(cfg["seed"] + 991)
    # [tacet] THE GATE'S OWN numpy STREAM, on the idiom the port's explore RNG already uses
    # ("its RNG is its own, so the shared stream is undisturbed and the twin gate stays
    # licensed"). Only `gate_mode="random"` ever draws from it, so the two delta arms and the
    # deliberation arm are deterministic functions of the run.
    ggrng = np.random.default_rng(cfg["seed"] + 3_141_593)

    # ---- PORT 2: the span head, its slots, and the executor (`../span/span.py`'s block,
    #      verbatim). The head is minted off its OWN generator, so its existence costs the
    #      shared stream nothing; its slots start CLOSED, so it cannot fire before parity.
    use_span = bool(spec.get("span"))
    head, slots = None, {}
    slot_events, gate_events = [], []
    span_rng = np.random.default_rng(cfg["seed"] + 20_250_820)     # its OWN stream
    # ---- [intonation] THE METER and the two consumers. All three default OFF, so with them
    #      unset this block is `tacet.py` exactly.
    #        perf_meter  the executor becomes fallible-and-measured (needs `span`).
    #        span_tau_fire  the firing threshold, BELOW `span_tau`; the parity record stays at
    #                     `span_tau` so the reduction can read off what the 0.95 gate did.
    #        perf_gain   (i) the per-sample plasticity consumer: "delta" | "raw" | None.
    #        gate_mode="perf_hi"  (ii) the selection consumer.
    pmeter = (PerfMeter(alpha=cfg["perf_alpha"], g0=cfg.get("perf_g0"),
                        theta=cfg.get("perf_theta"), calib_min=cfg["perf_calib_min"])
              if (use_span and cfg.get("perf_meter")) else None)
    perf_gain = None
    if pmeter is not None and cfg.get("perf_gain"):
        assert cfg["perf_gain"] in ("delta", "raw", "rawx"), cfg["perf_gain"]
        perf_gain = {"mode": cfg["perf_gain"], "tau_w": cfg["perf_tau_w"],
                     "w_raw_cap": cfg["perf_w_raw_cap"], "w_clip": cfg["perf_w_clip"],
                     # [w_norm, ewma_alpha] — the matched-average-budget normaliser, carried
                     # across cycles so the budget is matched over the run, not per cycle.
                     "wnorm": [1.0, cfg["perf_w_ewma"]], "last_wstat": None}
    perf_log_rows, perf_cells, calib_events = [], [], []
    if use_span:
        head = SN.build_head(SN.slot_count(s, depth, maxl), v, cfg["state_dim"],
                             s ** (maxl - 1), cfg["seed"] * 100 + 13, device,
                             hidden_mult=cfg["span_hidden_mult"])
        for p in head.parameters():
            p.requires_grad_(True)
        gopt.add_param_group({"params": list(head.parameters()),
                              "lr": cfg["span_lr"] or cfg["gen_lr"], "weight_decay": 1e-4})
        if pmeter is None:
            ex = SN.SpanExecutor(generator, head, slots, cap=cfg["span_buf_cap"],
                                 hold_cap=cfg["span_hold_cap"], hold_frac=cfg["span_hold_frac"],
                                 per_call=cfg["span_capture"], seed=cfg["seed"] + 5_150_101,
                                 v=v, length=shared["length"])
        else:
            # [intonation] the SAME constructor arguments, the same dedicated seed, the same
            # buffers — `PerfExecutor` differs from its parent only by carrying the meter.
            ex = build_perf_executor(generator, head, slots, cap=cfg["span_buf_cap"],
                                     hold_cap=cfg["span_hold_cap"],
                                     hold_frac=cfg["span_hold_frac"],
                                     per_call=cfg["span_capture"],
                                     seed=cfg["seed"] + 5_150_101,
                                     v=v, length=shared["length"], meter=pmeter,
                                     row_cap=cfg["perf_row_cap"],
                                     row_seed=cfg["seed"] + 6_260_101)
    else:
        ex = SN.PlainExecutor()

    def mint(level, table):
        if not use_span:
            return
        n_nodes = s ** (depth - level)
        for j in range(n_nodes):
            key = SN.slot_key(level, j)
            slots[key] = {"id": SN.slot_index(level, j, s, depth, maxl), "open": False,
                          "move": None, "level": int(level), "node": int(j)}
        slot_events.append({"level": int(level), "n_nodes": int(n_nodes),
                            "n_entries": int(table["child"].shape[0]),
                            "span": int(s ** (level - 1))})
        print(f"[slot]   arm={arm} level={level} nodes={n_nodes} "
              f"entries={table['child'].shape[0]} span={s ** (level - 1)}", flush=True)

    def bind_slots(ms_now):
        """Point every slot at its CURRENT macro move object (the table changes on commit)."""
        for mv in ms_now:
            if mv.get("kind") != "macro":
                continue
            key = SN.slot_key(mv["level"], mv["node"])
            if key in slots:
                slots[key]["move"] = mv

    # ---- vocabulary state ---------------------------------------------------------------
    committed = {ell: None for ell in range(2, maxl + 1)}
    miners = {ell: MC.Miner(ell, s) for ell in range(2, maxl + 1)}
    # [census] G-Y: the next-level-yield instrument. UNPRICED and READ-ONLY — it never enters
    # `operative`, never gates anything, never reaches `committed`, and its level is set by
    # `gy_level` rather than by `max_macro_level`, because that constant caps what may be
    # COMMITTED and this miner only OBSERVES. Phase 0 established the stream is absent from
    # `sp_s0` entirely and unreconstructible from its logs, so it has to be instrumented live.
    gy_level = cfg.get("gy_level")
    gy_miner = MC.Miner(int(gy_level), s) if gy_level else None
    gy_nodes = s ** (depth - int(gy_level)) if gy_level else 0
    # [census] the gauge the L2 gate reads: the at-support series per level, per cycle. Kept
    # here rather than read back out of `log["miner"]`, which is only appended at cycle end.
    gauge_hist = {ell: [] for ell in range(2, maxl + 1)}
    # [conductor] THE OBSERVATION PANEL. `census`'s G-Y miner generalised to every level above
    # the base, for a measured reason: the committable miners are gated to `era_level + 1`
    # (mining level 3 in era 1 would hand era 2 a finished vocabulary), so `gauge_hist[3]` is
    # identically ZERO through the whole of era 1 — verified on `as_s0/anchor`, 0 at every one of
    # c1..c48. The one-level-up gauge an era-1 loop needs therefore does not exist in the donor
    # and has to be instrumented. These miners OBSERVE ONLY: they never enter `operative`, never
    # gate anything, never reach `committed`, and never touch the era-gated `miners[...]` the
    # committed tables are built from. Pure numpy counting on features the cycle already parsed,
    # so they draw no RNG and cost no GPU — which is what lets them run in every arm without
    # voiding the replay gate. The level-4 entry is asserted equal to the donor's G-Y miner
    # in-run (`obs_gy_agree`), so the generalisation is checked against the instrument it
    # generalises rather than trusted.
    obs_levels = list(range(3, depth + 1)) if cfg.get("obs_panel") else []
    obs_miners = {ell: MC.Miner(ell, s) for ell in obs_levels}
    obs_hist = {ell: [] for ell in obs_levels}
    obs_gy_agree = []
    # [spiral] PREFLIGHT-ONLY instrument, and it exists because the first preflight could not
    # reach the transplanted commit path at all: a substrate trained for forty steps solves
    # nothing, so the miner never observes, `Miner.build` returns an empty table and the
    # `do_commit` guard turns every commit off — leaving the one code path the preflight is
    # supposed to be checking untested. Seeding the miner with the DGP's own level-l tuples
    # (the same `observe(concatenate([flat] * support))` idiom `selfcheck`'s C-R uses) makes the
    # table non-empty so the commit, the provisional branch and the recert all execute.
    # Guarded by a cfg key that no real config carries, and scientifically meaningless by
    # construction — it hands over the answer, which is exactly why it is preflight-only.
    if cfg.get("preflight_seed_miner"):
        # [census] the seed is PARTIAL, and the rest arrives later. A fully-seeded miner
        # produces a frozen table that already holds everything, so `extend_candidates` returns
        # nothing and the extension op — the thing the preflight most needs to exercise — never
        # runs. Seeding a fraction now and topping up after the commit reproduces the only
        # situation extension exists for: entries that show up AFTER the freeze.
        for ell in range(2, maxl + 1):
            flat = shared["truth"][ell]["flat"]
            k_ = max(1, int(len(flat) * float(cfg.get("preflight_seed_frac", 1.0))))
            miners[ell].observe(np.concatenate([flat[:k_]] * (cfg["mine_support"] + 1)))
    if spec["vocab"] == "true":
        for ell in range(2, maxl + 1):
            committed[ell] = shared["truth"][ell]
            mint(ell, committed[ell])
    ms = build_ms(base_ms, committed, s, depth, device)
    bind_slots(ms)
    p_width = cfg["pr_width"]

    # ---- THE PORT ------------------------------------------------------------------------
    #      The head is built inside `isolated_rng`, so it consumes none of the shared stream:
    #      a treated arm is bit-identical to its twin until the filter first switches on.
    prop_k_spec = spec.get("prop_k")
    if overrides.get("prop_k") is not None:
        prop_k_spec = int(overrides["prop_k"])
    offsets, n_slots = PN.slot_layout(depth, s, maxl)
    prop = popt = None
    prng = np.random.default_rng(cfg["seed"] + 4207)
    erng = np.random.default_rng(cfg["seed"] + 8821)
    pbuf = {"x": torch.zeros(0, shared["length"], dtype=torch.long),
            "r": torch.zeros(0, dtype=torch.long), "a": torch.zeros(0, dtype=torch.long)}
    move_age = {}                       # slot -> cycles this move has been in the action set
    state = {"filter_on": False}
    if prop_k_spec is not None:
        prop = PN.build_head(cfg["state_dim"], v, n_slots,
                             cfg["seed"] * 7919 + 13, device)
        popt = torch.optim.AdamW(prop.parameters(), lr=cfg["prop_lr"], weight_decay=1e-4)
        for slot in PN.move_slots(ms, offsets):
            move_age[slot] = 0

    def port_spec():
        """The `plan` port for the CURRENT action set, and its effective branching factor.
        Returns `(None, n_moves)` while the filter is off — during the warmup the head trains
        on the beam's trajectories but never gates, so the arm is still its twin exactly."""
        if prop is None or not state["filter_on"]:
            return None, len(ms)
        slots = PN.move_slots(ms, offsets)
        k = len(ms) if prop_k_spec < 0 else int(prop_k_spec)
        forced = tuple(i for i, sl in enumerate(slots)
                       if move_age.get(sl, 0) < cfg["prop_new_cycles"])
        k_eff = min(min(k, len(ms)), len(ms) - len(forced)) + len(forced)
        return ({"prop": prop, "slots": slots, "k": k, "forced": forced,
                 "n_slots": n_slots}, k_eff)

    def avail_mask():
        import torch as _t
        mask = _t.zeros(n_slots, dtype=_t.bool, device=device)
        for sl in PN.move_slots(ms, offsets):
            mask[sl] = True
        return mask

    def operative(ell):
        """The table macros at level `ell` would be built over: the committed one if this arm
        has committed, else the live mined candidate. This is where the ratchet bites — once
        level l-1 is frozen, level l can only ever be built over what was frozen."""
        if ell == 1:
            return MC.base_table(v)
        if committed[ell] is not None:
            return committed[ell]
        return miners[ell].build(operative(ell - 1), cfg["mine_support"])

    def port_for(mset):
        """[spiral] the `plan` port and beam width for an ARBITRARY action set. `port_spec`
        closes over the live `ms`; the recert has to grade a counterfactual one."""
        if prop is None or not state["filter_on"]:
            return None, cap_w(fit_width(len(mset), cfg["budget"], cfg["g_budget"]))
        sl = PN.move_slots(mset, offsets)
        k = len(mset) if prop_k_spec < 0 else int(prop_k_spec)
        forced = tuple(i for i, sl_ in enumerate(sl)
                       if move_age.get(sl_, 0) < cfg["prop_new_cycles"])
        k_eff = min(min(k, len(mset)), len(mset) - len(forced)) + len(forced)
        return ({"prop": prop, "slots": sl, "k": k, "forced": forced, "n_slots": n_slots},
                cap_w(PN.fit_width_k(k_eff, cfg["budget"], cfg["g_budget"])))

    def pol(x, r_np, mset):
        """[spiral] `ear/grader.policy_e`'s measurement — the agent's OWN performance policy on
        `x` with `mset` as the action set, at the declared per-solve grounding budget, scored by
        terminal success — re-expressed through THIS file's `plan` so a routed arm is graded
        through its own port. `ear`'s version enumerates, which at k=4 would price the recert
        as though the port were not there.

        Capture is suspended for the duration: the recert is a GRADING step, and letting its
        beams feed the span head's self-imitation buffer would give a treated arm strictly more
        corridor training data on recert cycles than its twin — a confound, not a treatment."""
        r_np = np.asarray(r_np)
        prt, w = port_for(mset)
        was_cap = getattr(ex, "capture", False)
        ex.capture = False
        b = plan(controller, generator, value, x, torch.from_numpy(r_np), mset,
                 rules_t, canon, depth, v, m, s, budget=cfg["budget"], beam_width=w,
                 device=device, port=prt, ex=ex)
        ex.capture = was_cap
        succ, dres = grade(b["x"].cpu().numpy(), r_np, rules, s)
        return {"e": 1.0 - float(succ.mean()), "dres": float(dres.mean()), "w": int(w),
                "counts": b["counts"], "x": b["x"]}

    buf = {"x": torch.zeros(0, shared["length"], dtype=torch.long),
           "r": torch.zeros(0, dtype=torch.long), "y": torch.zeros(0)}
    t_cum = 0.0
    events = []
    log = {"cycle": [], "era": [], "level": [], "t_cum": [], "e": [], "succ": [], "dres": [],
           "n_moves": [], "width": [], "g_per_solve": [], "e_practice": [], "vloss": [],
           # [tacet] the gate's per-cycle record, written in EVERY arm (see `gate_log`).
           "gate": [],
           # [intonation] the meter's per-cycle record: per-slot e/b/g/delta sums, the 2x2
           # cells, the gate calibration and the unpriced misfire counterfactual.
           "perf": [],
           "gloss": [], "n_solved": [], "n_mined": [], "miner": [], "aud": [], "cert": [],
           "probe": [], "vocab": [], "prop": [], "span": [], "blocks": [],
           "m_per_solve": [], "committed_grade": [], "gy": [], "entry": [],
           # [conductor] the decision trace and the shadow panel, per cycle, in every arm.
           "loop": [], "panel": [],
           # [tutti] the question port's per-cycle record: what was on the menu, what was
           # selected, what difficulty mix came out, and what the questions actually DELIVERED
           # against what they designed. Written in every arm, including the `exo` arms (where
           # it is the donor's own draw, described) — so the dose is measured on the same axis
           # everywhere, and on a FALLIBLE executor for the first time.
           "q": []}

    # ---- [tutti] THE QUESTION PORT'S OWN STATE -------------------------------------------
    # `question_mode=None` -> the port is off and every line below is inert, which is the
    # configuration `fidelity_smoke` gates at 0.000e+00 against `caesura.py`.
    qmode = cfg.get("question_mode") or None
    k_menu = int(cfg.get("question_k") or 0) or int(cfg["n_pr"])
    # its OWN numpy stream (`woodshed`'s `rehrng` discipline) — reserved so a future rule that
    # needs randomness has somewhere to draw it that is not the shared position.
    qrng = np.random.default_rng(cfg["seed"] + 909_090)
    qledger = QS.DeliveryLedger() if qmode in ("endo",) else None
    q_designed = q_halves = None
    if qmode:
        print(f"[q]      arm={arm} question_mode={qmode} K={k_menu} n_pr={cfg['n_pr']} "
              f"ledger={'on' if qledger is not None else 'off'}", flush=True)

    # ---- fixed held-out sets, per era ----------------------------------------------------
    meter, shadow = {}, {}
    for i, era in enumerate(eras):
        r_np, x_np = context_instances(rules, era_ctx(era), cfg["n_rt"], s, depth, v, m,
                                       seed=cfg["seed"] + 5000 + 17 * i)
        meter[i] = (r_np, torch.from_numpy(x_np))
        r2, x2 = context_instances(rules, era_ctx(era), cfg["n_score"], s, depth, v, m,
                                   seed=cfg["seed"] + 6100 + 23 * i)
        shadow[i] = (r2, torch.from_numpy(x2).to(device))

    # [spiral] the shadow certificate's state, PER LEVEL and per RUN (not per era): a level's
    # certificate can fire in an era later than the one that earns it, and the rate readout
    # wants the cycle it fired, not the era it fired in.
    shadow_cert = {ell: {"b": None, "ref": None, "emin": None, "hist": [], "dhist": [],
                         "run": 0, "fired": None, "c0": None}
                   for ell in range(2, maxl + 1)}

    # ---- [conductor] THE OUTER LOOP ------------------------------------------------------ #
    # Built here, once per arm, from the arm's own spec. A schedule arm gets `SchedulePolicy`,
    # which never acts — under it the era loop below is the donor's `for` loop exactly, and the
    # commit rule is the donor's `delta_prov`. The dead zones are the MEASURED ones and are
    # passed in rather than defaulted; `QuietPolicy` raises if any is missing.
    loop_spec = dict(spec.get("loop") or {})
    if loop_spec.get("read") == "endo":
        loop_spec["read"] = cfg.get("endo_read_key") or "endo"
    # [maestro] a learned arm is handed its FITTED policy by reward. The two learned arms reach
    # this line with identical specs but for `reward`, so the fitted object is the only thing
    # that differs between them anywhere in the run.
    if loop_spec.get("kind") == "learned":
        rw = loop_spec.get("reward")
        fitobj = (cfg.get("fitted") or {}).get(rw)
        if not fitobj:
            raise ValueError(f"no fitted policy for reward {rw!r} — see fit.py")
        loop_spec["fit"] = fitobj
    loop_floors = {"ledger": cfg["tol_ledger"], "endo": cfg["tol_endo"],
                   "endo_excess": cfg["tol_endo"],
                   # [caesura] delta-silence's own MEASURED dead zone. `QuietPolicy` refuses a
                   # defaulted floor by construction, which is the property that forces this to
                   # be measured rather than chosen: it is derived by the same null-ABBA
                   # procedure that measured `tol_endo`, on this round's Phase-A probe, and
                   # passed in explicitly (reduction §6 re-derives it in-tag).
                   "dsil": cfg["tol_dsil"],
                   "yield_by_level": {3: cfg["tol_yield_l3"], 4: cfg["tol_yield_l4"]}}
    yoke_plan = cfg.get("yoke_plan") or []
    if isinstance(yoke_plan, str):
        yoke_plan = json.loads(yoke_plan)
    # [caesura] THE VETO'S DETECTOR. `dsil_and` lets delta-silence gate commit TIMING without
    # letting it choose the crossing: A1's yield thermostat still decides WHICH rung and WHEN it
    # is ready, and a commit it licenses is DEFERRED on any cycle the executor is still moving.
    # The detector is A1's own `QuietPolicy` on the `dsil` read, stepped read-only — this round
    # adds no rule, it adds a gauge and one conjunction.
    dsil_det = (PO.build_policy({"kind": "quiet", "read": "dsil",
                                 "span": cfg["loop_span"], "W": cfg["loop_W"],
                                 "burn": cfg["loop_burn"], "alpha": cfg["loop_alpha"]},
                                floors=loop_floors)
                if cfg.get("dsil_veto") else None)
    dsil_events = []
    loop = PO.build_policy({**loop_spec,
                            "span": cfg["loop_span"], "W": cfg["loop_W"],
                            "burn": cfg["loop_burn"], "alpha": cfg["loop_alpha"]},
                           floors=loop_floors, yoke_plan=yoke_plan)
    driven = loop.kind == "quiet"
    # [tutti] THE SPLIT: A SECOND POLICY OBJECT THAT OWNS THE COMMIT.
    #
    # `loop` owns the ADVANCE, and owns the COMMIT too unless `loop_commit` is present. This
    # adds no RULE — `QuietPolicy` is A1's, imported, and the donor already builds a second
    # instance of it (`dsil_det`, the veto's read-only detector) in every arm. The split is
    # that detector promoted from read-only to LICENSING.
    #
    # Both objects are built from the same span/W/burn/alpha and the same measured floors, so
    # the ONLY thing that differs between a split arm and a pure one is WHICH SERIES licenses
    # WHICH ACTION. Under `kind == "yoke"` the plan already distinguishes COMMIT cycles from
    # ADVANCE cycles, so a yoke of a split arm needs no second object and this is skipped —
    # which is what makes the one-bit clock yoke work for a two-policy source.
    loop_c_spec = dict(spec.get("loop_commit") or {})
    loop_c = None
    if loop_c_spec and loop.kind != "yoke":
        loop_c = PO.build_policy({**loop_c_spec,
                                  "span": cfg["loop_span"], "W": cfg["loop_W"],
                                  "burn": cfg["loop_burn"], "alpha": cfg["loop_alpha"]},
                                 floors=loop_floors)
        assert loop_c.read_key != getattr(loop, "read_key", None), (
            f"{arm}: the split's two policies must read DIFFERENT series "
            f"(both read {loop_c.read_key!r}) — otherwise it is not a split")
        print(f"[split]  arm={arm} COMMIT reads {loop_c.read_key} "
              f"(tol {loop_c.tol_for(None) if loop_c.v_tol_default is not None else 'per-level'})"
              f" | ADVANCE reads {loop.read_key}", flush=True)

    def _acted_all(kind, cyc_, why=""):
        """[tutti] BOTH policies re-arm on EVERY action and at every era start, not only on
        their own. The donor's stated reason for resetting is REGIME CHANGE: a commit installs
        a table (a regime change for the yield gauge) and an advance changes the damage cell (a
        regime change for the executor delta-silence hears). Resetting own-action-only would
        make a split arm differ from a pure one in TWO things — which gauge licenses, and how
        each clock re-arms — instead of one. The own-action-only rule is a COUNTERFACTUAL
        replayed offline in the reduction from each policy's own logged trace, never an arm."""
        loop.acted(kind, cyc_, why=why)
        if loop_c is not None:
            loop_c.acted(kind, cyc_, why=why)

    # [conductor] BOTH endo keys carry the price. The driven key is `endo_read_key`, which is
    # `endo_excess` by default, and a price map keyed only on `endo` silently falls through to
    # 0 — which is what happened in `cd_s0`, where `outer_endo` recorded `spend 0g` for 146
    # priced reads. Harmless to the science by construction (a charge reaches only
    # `counts["ground"]` -> `t_cum`, which nothing in the cycle loop reads, so no trajectory
    # moved) and exactly reconstructible after the fact (`charge` records `n_reads` before it
    # prices), but the ledger column is the round's answer to "what did the read cost", so it
    # is fixed here and reconstructed in the reduction for that tag.
    _ep = int(cfg.get("endo_price") or 0)
    ledger = PO.ReadLedger({"endo": _ep, "endo_excess": _ep})
    loop_actions, panel_hist = [], []
    caps = list(cfg.get("era_caps") or ())
    total_cap = int(cfg.get("total_cap") or (sum(caps) if caps else 0))
    # The whole clean probe batch the endo read is taken on, fixed for the run: no sampling, no
    # RNG, so the series' only cycle-to-cycle movement is the plant's own.
    endo_x = shared["probe_clean"][:int(cfg["n_endo"])].to(device)
    print(f"[loop]   arm={arm} policy={loop.kind}"
          + (f" read={loop.read_key} span={cfg['loop_span']} W={cfg['loop_W']} "
             f"burn={cfg['loop_burn']} priced={loop.priced}" if driven else "")
          # [maestro] a learned arm prints the policy it was HANDED, so the run's own log says
          # which mixture and which dead zone governed it.
          + (f" reward={loop.reward} mixes={loop.mixes} thetas={loop.thetas} "
             f"gauges={loop.gauges} priced={loop.priced}"
             if loop.kind == "learned" else "")
          + (f" plan={len(yoke_plan)} actions" if loop.kind == 'yoke' else ""), flush=True)

    cyc = 0
    era_i = -1
    while era_i + 1 < len(eras):
        era_i += 1
        era = eras[era_i]
        active = era["level"] + 1                     # the macro level being EARNED this era
        cert = {"b": None, "ref": None, "emin": None, "hist": [], "dhist": [], "run": 0,
                "fired": None}
        era_start = cyc + 1
        # [conductor] under the outer loop the era's length is a CAP, not a schedule: `ec` is
        # the most cycles this era may take, and the loop may leave sooner. With the loop off
        # this is the ladder's own count and the `while` below is the donor's `for` exactly.
        if loop.kind == "schedule":
            ec = int(era.get("cycles") or cfg["era_cycles"])
        else:
            ec = int(caps[era_i]) if era_i < len(caps) else \
                int(era.get("cycles") or cfg["era_cycles"])
        era_end = cyc + ec
        _acted_all("era_start", era_start, why=f"era{era_i + 1}")   # [tutti]
        print(f"\n----- arm={arm} ERA {era_i + 1}: damage {era['name']} "
              f"(earning level {active}) c{era_start}..{era_end}"
              f"{' (cap)' if loop.kind != 'schedule' else ''} -----", flush=True)

        c_in_era = 0
        while True:
            c_in_era += 1
            cyc += 1
            # [assay] the entry-identity instrument: cleared each cycle, tagged `beam` for the
            # priced practice + metering beams and `probe` for everything unpriced, so beam
            # executions are separable from instrument ones in the reduction.
            _rec_reset()
            _ENTRY_REC["on"] = bool(cfg.get("entry_rec", True))
            _ENTRY_REC["phase"] = "beam"
            if (cfg.get("preflight_seed_miner")
                    and cyc == int(cfg.get("preflight_seed_topup_cycle", 0))):
                for ell in range(2, maxl + 1):
                    flat = shared["truth"][ell]["flat"]
                    miners[ell].observe(np.concatenate([flat] * (cfg["mine_support"] + 1)))
            counts = {"mat": 0, "ground": 0, "prop": 0}
            ex.reset()
            # THE FILTER SWITCHES ON after `prop_warmup` cycles of the run. Until then the head
            # trains on the beam's trajectories but never gates, so (i) the arm is bit-identical
            # to its twin through the warmup and (ii) the filter never gates from a random init.
            if prop is not None and cyc > cfg["prop_warmup"]:
                state["filter_on"] = True
            port, k_eff = port_spec()
            # PRACTICE EXPLORES, PERFORMANCE DOES NOT — the substrate's own convention
            # (`collect_value_buffer` runs its behaviour policy at explore_eps = 0.3). Without
            # it the filtered beam is a closed loop: a move the head does not propose never
            # appears in a solved trajectory and so never becomes a training target.
            port_pr = port
            if port is not None and cfg["prop_explore"]:
                port_pr = dict(port, explore=int(cfg["prop_explore"]), erng=erng)

            # --- (a) practice: wide closed-loop beam on fresh instances of this era's cell
            # [tutti] THE QUESTION PORT'S ONE CALL SITE. The draw below is the donor's,
            # byte-for-byte, at the donor's own RNG position — `with_clean=True` changes the
            # return value and nothing else (`context_instances` computes `clean` either way).
            # With `question_mode=None` these are the donor's two lines and nothing more runs.
            r_np, x_np, cl_np = context_instances(
                rules, era_ctx(era), cfg["n_pr"], s, depth, v, m,
                seed=cfg["seed"] + 100_000 + 1000 * cyc, with_clean=True)
            qrow, q_designed, q_halves = None, None, None
            if qmode:
                # RNG SANDBOX (belt, braces, third belt): the port draws on its own numpy
                # streams and takes only no-grad forwards on dropout-free nets, so it consumes
                # nothing — and the snapshot/restore makes that true by construction rather
                # than by inspection. The round's load-bearing gate is `tu_y_exo` replaying
                # `ca_s0/dsil_yield` bit-for-bit at full scale; this is what protects it.
                # (The nets are left in `.eval()` on the way out, which is safe here because
                # every trainer in this file calls `.train()` at entry — asserted by the
                # 0.000e+00 replay rather than by inspection.)
                _q_state = _rng_snapshot()
                (r_np, x_np, cl_np, q_designed, q_halves, qrow) = pose_questions(
                    qmode, rules=rules, era=era, cfg=cfg, s=s, depth=depth, v=v, m=m, cyc=cyc,
                    head=(r_np, x_np, cl_np), shared=shared, value_net=value,
                    controller=controller, device=device, miners=miners, operative=operative,
                    maxl=maxl, qledger=qledger, n_pr=cfg["n_pr"], k_menu=k_menu)
                _rng_restore(_q_state)
                if not qrow["quota_ok"]:
                    print(f"[q!]     arm={arm} c{cyc} QUOTA DEVIATION mode={qmode} "
                          f"d*={qrow['d_mean']:.3f} vs menu {qrow['d_mean_menu']:.3f}",
                          flush=True)
            out = plan(controller, generator, value, torch.from_numpy(x_np),
                       torch.from_numpy(r_np), ms, rules_t, canon, depth, v, m, s,
                       budget=cfg["budget"], beam_width=p_width, device=device,
                       collect=True, port=port_pr, ex=ex)
            for key in counts:
                counts[key] += out["counts"].get(key, 0)
            tips = out["tips_x"]
            B, W, T = tips.shape
            tips_flat = tips.reshape(B * W, T)
            succ, _ = grade(tips_flat.cpu().numpy(), np.repeat(r_np, W), rules, s)
            xs = torch.cat([t.reshape(B * W, T).cpu() for t in out["traj"]])
            rs = torch.from_numpy(np.tile(np.repeat(r_np, W), len(out["traj"])))
            ys = torch.from_numpy(np.tile(succ.astype(np.float32), len(out["traj"])))
            push(buf, xs, rs, ys, cfg["buf_cap"])
            solved = tips_flat[torch.from_numpy(succ > 0.5).to(device)]
            ps, _ = grade(out["x"].cpu().numpy(), r_np, rules, s)
            e_practice = 1.0 - float(ps.mean())

            # --- (a2) [tacet] THE GATE. Read the two decision-time scalars off what the beam
            #          already computed, then decide, per consumption channel, which of the
            #          data the GRADE already admitted this arm will actually learn from.
            #
            #          Order matters and is stated: the grade filter is upstream of the gate in
            #          both channels, so the gate is always a RESTRICTION of the current rule,
            #          never a relaxation. `gate_all` is the one arm that relaxes it, and it
            #          does so through the donor's own knobs (`mine_cap=0`, `prop_train_on`),
            #          not through the gate.
            #
            #          The VALUE BUFFER is above this line on purpose: `push` has already run.
            #          delta is the residual of the value head's forecast, so gating the value
            #          head's own diet would make the gate self-referential.
            gmode = cfg.get("gate_mode")
            gfe = (gate_features(out, succ, ps, B, W)
                   if (gmode or cfg.get("gate_log")) else None)
            keep_tip, mine_ord, grec = None, None, None
            # [intonation] THE 2x2 — executed-as-intended x solved — and delta_perf as one more
            # ordering scalar. `p_tip` is the trajectory's summed delta_perf; the row axis is
            # `bad == 0`, i.e. every head execution on this trajectory exactly matched its
            # intention. A trajectory the head never executed on is its OWN cell (`exe == 0`)
            # and is never folded into "as intended": nothing was performed, so there is no
            # performance to grade. This block runs in EVERY metered arm, consuming nothing
            # unless `gate_mode` is a perf mode.
            p_tip = p_flat = None
            if gfe is not None and "p_tip" in gfe:
                p_tip, p_flat = gfe["p_tip"], gfe["p_tip"].reshape(-1)
                bad_f, exe_f = gfe["bad_tip"].reshape(-1), gfe["exe_tip"].reshape(-1)
                sol_f = (succ > 0.5)
                intent = (bad_f == 0) & (exe_f > 0)
                none_x = (exe_f == 0)
                cells = {
                    # the four cells, plus the no-execution row kept apart
                    "int_solved": int((intent & sol_f).sum()),
                    "int_failed": int((intent & ~sol_f).sum()),
                    "bad_solved": int((~intent & ~none_x & sol_f).sum()),   # LUCKY SUCCESS
                    "bad_failed": int((~intent & ~none_x & ~sol_f).sum()),
                    "non_solved": int((none_x & sol_f).sum()),
                    "non_failed": int((none_x & ~sol_f).sum()),
                    "n_tip": int(sol_f.shape[0]),
                    "n_exec_tip": int((exe_f > 0).sum()),
                    "sum_exe": float(exe_f.sum()), "sum_bad": float(bad_f.sum()),
                    "d_int": round(float(p_flat[intent].mean()), 5) if intent.any() else None,
                    "d_bad": round(float(p_flat[~intent & ~none_x].mean()), 5)
                             if (~intent & ~none_x).any() else None,
                }
                # and the same 2x2 at the MINED unit (the beam's own answer per instance),
                # which is the unit the vocabulary is actually built from.
                ai = (gfe["bad_ans"] == 0) & (gfe["exe_ans"] > 0)
                an = (gfe["exe_ans"] == 0)
                sa = (ps > 0.5)
                cells.update({"ans_int_solved": int((ai & sa).sum()),
                              "ans_int_failed": int((ai & ~sa).sum()),
                              "ans_bad_solved": int((~ai & ~an & sa).sum()),
                              "ans_bad_failed": int((~ai & ~an & ~sa).sum()),
                              "ans_non_solved": int((an & sa).sum()),
                              "ans_non_failed": int((an & ~sa).sum())})
                cells["c"] = cyc
                perf_cells.append(cells)
            if gfe is not None:
                d_tip, mg = gfe["d_tip"].reshape(-1), np.repeat(gfe["margin"], W)
                sol_tip = np.flatnonzero(succ > 0.5)
                n_keep = (max(1, int(round(float(cfg["gate_frac"]) * sol_tip.shape[0])))
                          if sol_tip.shape[0] else 0)
                grec = {"c": cyc, "mode": gmode, "n_sol_tip": int(sol_tip.shape[0]),
                        "n_sol_inst": int((ps > 0.5).sum()), "n_keep_tip": 0,
                        # the gate's own inputs, per instance, in EVERY arm — 3 x n_pr floats
                        # a cycle, which is what makes the counterfactual gate computable
                        # offline for the ungated baseline too.
                        "d_ans": [round(float(x), 5) for x in gfe["d_ans"]],
                        "margin": [round(float(x), 5) for x in gfe["margin"]],
                        # the textbook top1-top2 margin, logged beside the gated one so the
                        # reduction can say how degenerate it is (see `gate_features`).
                        "m2": [round(float(x), 5) for x in gfe["m2"]],
                        "ps": [int(x) for x in (ps > 0.5)]}
                if p_flat is not None:                     # [intonation]
                    grec["p_ans"] = [round(float(x), 5) for x in gfe["p_ans"]]
                    grec["bad_ans"] = [int(x) for x in gfe["bad_ans"]]
                    grec["exe_ans"] = [int(x) for x in gfe["exe_ans"]]
                if gmode and sol_tip.shape[0]:
                    order = gate_order(gmode, sol_tip, d_tip, mg, ggrng, perf=p_flat,
                                       nexe=(gfe["exe_tip"].reshape(-1)
                                             if "exe_tip" in gfe else None))
                    kept = order[:n_keep]
                    keep_tip = np.zeros(B * W, dtype=bool)
                    keep_tip[kept] = True
                    grec["n_keep_tip"] = int(n_keep)
                    grec["d_kept"] = round(float(np.mean(d_tip[kept])), 5)
                    grec["d_drop"] = (round(float(np.mean(d_tip[order[n_keep:]])), 5)
                                      if order.shape[0] > n_keep else None)
                    grec["m_kept"] = round(float(np.mean(mg[kept])), 5)
                    if p_flat is not None:                 # [intonation]
                        grec["p_kept"] = round(float(np.mean(p_flat[kept])), 5)
                        grec["p_drop"] = (round(float(np.mean(p_flat[order[n_keep:]])), 5)
                                          if order.shape[0] > n_keep else None)
                        # how many of the kept tips were EXECUTED AS INTENDED, and how many
                        # of the ranking's decisions were made on ties (perf == 0, i.e. no
                        # head execution on that trajectory) — `tacet`'s `m2 == 0` convention.
                        grec["int_kept"] = int(((gfe["bad_tip"].reshape(-1)[kept] == 0)
                                                & (gfe["exe_tip"].reshape(-1)[kept] > 0)).sum())
                        grec["p_tie"] = int((p_flat[sol_tip] == 0).sum())
                        # [intonation/A] the tie count on the key THIS mode ranks by: for
                        # `perf_mean` the no-execution rows are ordered last on purpose and are
                        # not ties, so the honest number is ties WITHIN the executed set.
                        _ex = gfe["exe_tip"].reshape(-1)[sol_tip] > 0
                        grec["n_exec_cand"] = int(_ex.sum())
                        _mu = np.where(_ex, p_flat[sol_tip] / np.maximum(
                            gfe["exe_tip"].reshape(-1)[sol_tip], 1.0), np.nan)
                        grec["p_tie_exec"] = int((_mu[_ex] == 0).sum()) if _ex.any() else 0
                if gmode and cfg.get("gate_mine"):
                    # the MINING order: the same scalar at the mined unit (the beam's own
                    # answer per instance), over the solved instances only.
                    sol_inst = np.flatnonzero(ps > 0.5)
                    if sol_inst.shape[0]:
                        mine_ord = gate_order(gmode, np.arange(sol_inst.shape[0]),
                                              gfe["d_ans"][sol_inst], gfe["margin"][sol_inst],
                                              ggrng,
                                              perf=(gfe["p_ans"][sol_inst]      # [intonation]
                                                    if "p_ans" in gfe else None),
                                              nexe=(gfe["exe_ans"][sol_inst]
                                                    if "exe_ans" in gfe else None))

            # --- (b) THE VOCABULARY IS MINED from what the agent actually solved, parsed by
            #         the agent's OWN generator. Every level's span containing this era's
            #         damage cell is observed; whether an observation becomes an ENTRY is
            #         decided at build time by the operative lower table (the ratchet).
            #         `mine_from="chosen"` records only the beam's OWN final answer per
            #         instance — the agent's performances, not the leaves of its search tree.
            #         With 16 tips per instance the tree saturates a 14-entry table in one
            #         cycle, which would make the certificate vacuous for a bookkeeping reason
            #         rather than a substrate one (round 1's `cald_s0` trap, one level up).
            mine_src = (out["x"][torch.from_numpy(ps > 0.5).to(device)]
                        if cfg["mine_from"] == "chosen" else solved)
            if cfg["mine_cap"] and mine_src.shape[0] > cfg["mine_cap"]:
                # [tacet] THE DRAW IS UNCONDITIONAL, exactly as in the donor, and only the
                # ORDERING it is read through changes. Keeping the `rng.permutation` call on
                # the same branch with the same argument is what makes the shared per-arm
                # stream identical between a gate arm and the baseline, so the first
                # divergence between them is the gate's CONTENT and never its bookkeeping.
                perm = rng.permutation(mine_src.shape[0])
                take = (perm if (mine_ord is None or cfg["mine_from"] != "chosen")
                        else mine_ord)[:cfg["mine_cap"]]
                sel = torch.from_numpy(np.ascontiguousarray(take))
                mine_src = mine_src[sel.to(device)]
                mine_take = np.asarray(take)               # [intonation]
            else:
                mine_take = np.arange(mine_src.shape[0])   # [intonation] the cap did not bind
            # [tutti] THE MINED ROWS' INDICES BACK INTO THE POSED BATCH, so the designed key of
            # a question can be joined to the key its answer actually delivered. `antiphon`
            # carried its own parallel index here; the donor already computes exactly that
            # object (`intonation`'s `mine_take` over `np.flatnonzero(ps > 0.5)`), so this
            # REUSES it rather than adding a second one. Pure bookkeeping: it reads
            # `rng.permutation`'s result, it does not move it. `mine_ord` is None in every arm
            # of this tag (no diet gate is built — that seat is closed by measurement), so
            # `mine_take` is the donor's own permutation.
            _mine_idx = None
            if qmode and cfg["mine_from"] == "chosen":
                _si = np.flatnonzero(ps > 0.5)
                _mine_idx = (_si[mine_take] if mine_take.shape[0] <= _si.shape[0] else _si)
            # [intonation] WHAT GOT MINED, BY 2x2 CELL — in every arm, gated or not, so the
            # ungated baseline's own diet is on the record in the same units. `mine_take`
            # indexes the solved instances (`mine_from="chosen"`), which is the mined unit.
            if grec is not None and gfe is not None and "p_ans" in gfe \
                    and cfg["mine_from"] == "chosen":
                si = np.flatnonzero(ps > 0.5)
                ki = si[mine_take] if mine_take.shape[0] <= si.shape[0] else si
                grec["mine_n"] = int(ki.shape[0])
                grec["mine_int"] = int(((gfe["bad_ans"][ki] == 0)
                                        & (gfe["exe_ans"][ki] > 0)).sum())
                grec["mine_bad"] = int((gfe["bad_ans"][ki] > 0).sum())
                grec["mine_non"] = int((gfe["exe_ans"][ki] == 0).sum())
                grec["mine_p"] = round(float(gfe["p_ans"][ki].mean()), 5) if ki.size else None
            #         Levels are mined only up to `era_level + 1`: during era k the agent forms
            #         level-(k+1) chunks out of what it is currently producing. Mining level 3
            #         during era 1 would hand era 2 a finished vocabulary before era 2 begins,
            #         which would erase both the poison test and the unit-LP curve it needs.
            if mine_src.shape[0]:
                # read with the READER, not the generator: `calp2_s0` measured that the
                # generator's block head is unsupervised at visible positions (0.63), which
                # capped the earned level-3 vocabulary's precision at ~0.3.
                pf = MC.parse_features(shared["reader"], mine_src, s=s).cpu().numpy()
                # [tutti] the counts the target miner held BEFORE this cycle's observations,
                # so "did this question move a key that was not already banked" is answerable.
                _tgt = min(maxl, era["level"] + 1)
                _cnt_before = dict(miners[_tgt].counts) if qmode and _tgt in miners else None
                for ell in range(2, min(maxl, era["level"] + 1) + 1):
                    span = s ** (ell - 1)
                    node = (era["node"] * s ** (era["level"] - 1)) // span
                    miners[ell].observe(pf[:, node * span:(node + 1) * span])
                # [tutti] DELIVERED DOSE vs DESIGN DOSE, and the delivery ledger's credit.
                # `wd_s0`'s lesson as an instrument: the question-posing machinery is measured
                # against its own design every cycle, in every arm, rather than assumed. NEW
                # HERE: the repair is executed by a FALLIBLE span head, so this is the first
                # time the dose is read on an executor that can miss — if it falls below
                # `an_s0`'s 0.45 at era 1 the knob's grip is narrower on this substrate, and
                # that is a number rather than a caveat.
                if qmode and qrow is not None and _mine_idx is not None \
                        and _cnt_before is not None:
                    _sp = s ** (_tgt - 1)
                    _nd = (era["node"] * s ** (era["level"] - 1)) // _sp
                    _delivered = pf[:, _nd * _sp:(_nd + 1) * _sp]
                    _des = q_designed[_mine_idx]
                    _n = min(len(_mine_idx), _delivered.shape[0])
                    _hit = [bool(np.array_equal(_delivered[j], _des[j])) for j in range(_n)]
                    _support = int(cfg["mine_support"])
                    _truth_t = {tuple(int(z) for z in r)
                                for r in shared["truth"][_tgt]["flat"]}
                    _dk = [tuple(int(z) for z in r) for r in _delivered[:_n]]
                    # credit = the observation advanced a key that was NOT already banked
                    _credit = [1 if _cnt_before.get(k, 0) < _support else 0 for k in _dk]
                    if qledger is not None and q_halves is not None and q_halves.shape[1]:
                        qledger.update(
                            [tuple(int(z) for z in h) for h in q_halves[_mine_idx][:_n]],
                            _credit)
                    qrow.update({
                        "n_mined": int(_n), "n_solved": int(len(np.flatnonzero(ps > 0.5))),
                        "dose_hit": float(np.mean(_hit)) if _hit else None,
                        "delivered_true": (float(np.mean([k in _truth_t for k in _dk]))
                                           if _dk else None),
                        "designed_true_mined": (float(np.mean(
                            [tuple(int(z) for z in r) in _truth_t for r in _des[:_n]]))
                            if _n else None),
                        "credit_rate": float(np.mean(_credit)) if _credit else None,
                        "at_support_after": int(sum(
                            1 for c in miners[_tgt].counts.values() if c >= _support)),
                        "ledger": (qledger.state() if qledger is not None else None)})
                # [census] G-Y observes in EVERY era, not only where `era_level + 1` reaches
                # its level: the readout is the growth DIRECTION of the next level's candidate
                # stream across the whole ladder, so a partial trajectory would not be one.
                if gy_miner is not None:
                    span_y = s ** (gy_miner.level - 1)
                    node_y = (era["node"] * s ** (era["level"] - 1)) // span_y
                    if 0 <= node_y < gy_nodes and (node_y + 1) * span_y <= pf.shape[1]:
                        gy_miner.observe(pf[:, node_y * span_y:(node_y + 1) * span_y])
                # [conductor] the observation panel, on the SAME features and with the SAME
                # span/node arithmetic G-Y uses — so the panel's level-4 entry is the G-Y miner
                # recomputed, which is what `obs_gy_agree` asserts below.
                for ell, mnr in obs_miners.items():
                    span_o = s ** (ell - 1)
                    node_o = (era["node"] * s ** (era["level"] - 1)) // span_o
                    if 0 <= node_o < s ** (depth - ell) and (node_o + 1) * span_o <= pf.shape[1]:
                        mnr.observe(pf[:, node_o * span_o:(node_o + 1) * span_o])
            for ell in range(2, maxl + 1):
                gauge_hist[ell].append(
                    int(miners[ell].state()["n_at_support"][str(cfg["mine_support"])]))
            for ell, mnr in obs_miners.items():
                obs_hist[ell].append(
                    int(mnr.state()["n_at_support"][str(cfg["mine_support"])]))
            if gy_miner is not None and int(gy_miner.level) in obs_miners:
                obs_gy_agree.append(
                    obs_hist[int(gy_miner.level)][-1]
                    == int(gy_miner.state()["n_at_support"][str(cfg["mine_support"])]))

            # --- (c) the plant learns, then the selector ---------------------------------
            sloss, sacc = None, {}
            if use_span:
                gloss, sloss, sacc = finetune_generator_span(
                    generator, head, ex, slots, gopt, solved.cpu(), shared["replay"]["x"],
                    shared["bottom_map"], v=v, s=s, n_blocks=shared["n_blocks"],
                    n_steps=cfg["gen_steps"], batch=cfg["batch_size"],
                    replay_frac=cfg["replay_frac"], device=device, rng=grng,
                    span_rng=span_rng, span_batch=cfg["span_batch"], lam=cfg["span_lam"],
                    perf_gain=perf_gain)      # [intonation] None == the donor's own function
            else:
                gloss = finetune_generator(generator, gopt, solved.cpu(), shared["replay"]["x"],
                                           shared["bottom_map"], v=v, s=s,
                                           n_blocks=shared["n_blocks"], n_steps=cfg["gen_steps"],
                                           batch=cfg["batch_size"], replay_frac=cfg["replay_frac"],
                                           device=device, rng=grng)

            # --- (c2) THE PARITY GATE (span/'s, verbatim). Held-out exact-match against what
            #         the DP would write with the CURRENT trunk, per macro, re-checked every
            #         cycle rather than latched, so a slot the moving plant drifts away from
            #         closes again. Instrument cost only; never priced.
            par = {}
            if use_span and slots:
                par = SN.parity(generator, head, ex, slots, s, v,
                                min_hold=cfg["span_min_hold"])
                # [intonation] THE ONE CHANGE: the FIRING threshold. `span_tau_fire` (None ==
                # the donor's `span_tau`) is what decides whether the head executes; the
                # PARITY RECORD is unchanged and `open_tau` is logged beside `open` on every
                # event, so the tau = 0.95 counterfactual is readable straight off the log.
                # Lowering it is the whole treatment: at 0.95 the head is allowed to fire only
                # where it is already ~never wrong, so `e` is ~0 and delta_perf is degenerate.
                tau_fire = cfg.get("span_tau_fire")
                tau_fire = cfg["span_tau"] if tau_fire is None else float(tau_fire)
                for key, cell in par.items():
                    was = slots[key]["open"]
                    now = cell["exact"] is not None and cell["exact"] >= tau_fire
                    at_tau = cell["exact"] is not None and cell["exact"] >= cfg["span_tau"]
                    slots[key]["open"] = now
                    if now != was:
                        gate_events.append({"cycle": cyc, "slot": key, "open": bool(now),
                                            "open_tau": bool(at_tau),
                                            "exact": cell["exact"], "n": cell["n"]})
                        print(f"[gate]   arm={arm} c{cyc} slot {key} "
                              f"{'OPEN' if now else 'CLOSE'} exact={cell['exact']} "
                              f"n={cell['n']} tau_fire={tau_fire}", flush=True)
            vloss = value_steps(value, vopt, controller, buf, shared["replay"],
                                n_steps=cfg["n_grad"], batch=cfg["value_batch"],
                                replay_frac=cfg["replay_frac"], device=device, rng=rng)

            # --- (c') THE PORT LEARNS: self-imitation of the beam's own chosen trajectories,
            #          restricted to the survivors that actually SOLVED. Its data comes from
            #          the same beam it gates, so the imitation is on-policy from the cycle the
            #          filter switches on; its RNG is its own, so the shared stream is
            #          undisturbed and the twin gate stays licensed.
            pinfo, fallback = {"k": None}, False
            if prop is not None:
                pairs = prop_pairs(out, torch.from_numpy(r_np), succ,
                                   PN.move_slots(ms, offsets), cfg["budget"],
                                   train_on=cfg["prop_train_on"],
                                   keep=keep_tip)          # [tacet] the gate; None == donor
                if pairs is None and cfg["prop_train_on"] == "solved":
                    # nothing solved this cycle: a policy that cannot generate a success cannot
                    # be improved by imitating an empty set, so fall back to the beam's own
                    # survivors (which the value did select) rather than skipping the update
                    # and letting a bad filter freeze itself in.
                    # [tacet] the fallback is UNGATED, deliberately: it fires only when the
                    # grade filter itself came back empty, so there is nothing for the gate to
                    # be selective about, and gating a rescue path would turn "no successes
                    # this cycle" into "no pi update this cycle" — a different intervention.
                    pairs = prop_pairs(out, torch.from_numpy(r_np), succ,
                                       PN.move_slots(ms, offsets), cfg["budget"],
                                       train_on="tips")
                    fallback = True
                if pairs is not None:
                    n_cap = cfg["prop_buf_cap"]
                    pbuf["x"] = torch.cat([pbuf["x"], pairs["x"]])[-n_cap:]
                    pbuf["r"] = torch.cat([pbuf["r"], pairs["r"]])[-n_cap:]
                    pbuf["a"] = torch.cat([pbuf["a"], pairs["a"]])[-n_cap:]
                pinfo = prop_train(prop, popt, controller, pbuf, avail_mask(),
                                   n_steps=cfg["prop_steps"], batch=cfg["prop_batch"],
                                   device=device, rng=prng)
                pinfo.update({"k": (len(ms) if prop_k_spec < 0 else int(prop_k_spec)),
                              "k_eff": int(k_eff), "filter_on": bool(state["filter_on"]),
                              "n_pairs": (0 if pairs is None else int(pairs["x"].shape[0])),
                              "fallback": bool(fallback)})

            # --- (d) metering under PERFORMANCE conditions (the declared grounding budget) -
            #         The width is refit to the SAME declared budget under this arm's own
            #         effective branching factor: ratchet's matched-pricing idiom with
            #         n_moves -> k. The groundings the port frees are reinvested in width, not
            #         banked; the banked reading is the width ladder logged at every probe.
            mr_np, mx = meter[era_i]
            port, k_eff = port_spec()
            width = cap_w(fit_width(len(ms), cfg["budget"], cfg["g_budget"]) if port is None
                          else PN.fit_width_k(k_eff, cfg["budget"], cfg["g_budget"]))
            b = plan(controller, generator, value, mx, torch.from_numpy(mr_np), ms,
                     rules_t, canon, depth, v, m, s, budget=cfg["budget"],
                     beam_width=width, device=device, port=port, ex=ex)
            for key in counts:
                counts[key] += b["counts"].get(key, 0)
            # snapshot the block-level tally NOW: the probes below are unpriced instruments
            # and must not enter the counterfactual ledger any more than the real one.
            blocks_cycle = dict(ex.counts)
            _ENTRY_REC["phase"] = "probe"          # [assay] everything after (d) is unpriced
            msucc, mdres = grade(b["x"].cpu().numpy(), mr_np, rules, s)
            e = 1.0 - float(msucc.mean())
            gps = b["counts"]["ground"] / mx.shape[0]
            meter_cost = {"g": gps, "mat": b["counts"]["mat"] / mx.shape[0],
                          "prop": b["counts"].get("prop", 0) / mx.shape[0],
                          "w": int(width), "k_eff": int(k_eff)}

            # --- (d2) [conductor] THE SHADOW PANEL AND THE DECISION POINT -------------------
            #     Every candidate gauge, in EVERY arm, at every decision point, uncharged
            #     (`endo_yield`'s pattern): one run therefore carries a counterfactual decision
            #     trace for each gauge in each arm, replayable offline through the same rule.
            #     The LEDGER then charges only what THIS arm's policy actually read — ear's
            #     convention, stated in `policy.ReadLedger`.
            #
            #     Placed here, after the metering beam, because that is where the arm's own
            #     error for this cycle exists; every gauge is therefore read at the same instant
            #     of the same cycle, and the three arms differ in which of them is consumed.
            #
            #     READ LEVEL. "One level up" from the level being earned is `active + 1`. Above
            #     level 4 nothing is instrumented (level 5 has 2^20 candidate tuples and no
            #     miner), so in the consumption eras the read CLAMPS to level 4 — the deepest
            #     available next-level gauge. Stated rather than hidden: in eras 3-5 the yield
            #     read is not literally `active + 1`, and `read_level` is logged per cycle so
            #     the reduction can separate the eras where it was.
            read_level = min(active + 1, max(obs_miners) if obs_miners else active + 1)
            read_level = min(read_level, int(cfg.get("gy_level") or 4))
            y_next = obs_hist.get(read_level, [None])[-1] if obs_hist.get(read_level) else None
            y_active = obs_hist.get(active, [None])[-1] if obs_hist.get(active) else None
            e_par = endo_read(generator, endo_x, shared["bottom_map"],
                              parent_span_blocks(era, s, depth),
                              v=v, s=s, n_blocks=shared["n_blocks"])
            e_cell = endo_read(generator, endo_x, shared["bottom_map"],
                               cell_span_blocks(era, s, depth),
                               v=v, s=s, n_blocks=shared["n_blocks"])
            # [caesura] the gauge. `pmeter.bench` is `intonation`'s per-macro-slot benchmark,
            # already maintained; nothing new is computed and nothing new is priced. Averaged
            # over the OPEN slots only, because a closed slot's b(s) describes an executor the
            # agent is not running. `dsil_all` (every minted slot) is logged beside it so the
            # other definition is readable off the record.
            _open_keys = [k for k, sl in slots.items() if sl.get("open")]
            _bench = (pmeter.bench if pmeter is not None else {})
            _bo = [float(_bench[k]) for k in _open_keys if k in _bench]
            _ba = [float(x) for x in _bench.values()]
            _dsil_now = (float(np.mean(_bo)) if _bo else None)
            _dsil_all = (float(np.mean(_ba)) if _ba else None)
            _dsil_n, _dsil_open = len(_ba), len(_bo)
            panel = {"cycle": cyc, "era": era_i + 1, "active": int(active),
                     "dsil_all": _dsil_all,
                     "read_level": int(read_level),
                     # ERROR CONVENTION throughout (lower is better), so every gauge shares one
                     # sign convention and `policy.py` is indifferent to which it is handed.
                     "ledger": float(e), "ledger_level": int(active),
                     "yield": (None if y_next is None else -float(y_next)),
                     "yield_level": int(read_level),
                     "yield_active": (None if y_active is None else -float(y_active)),
                     # [caesura] delta-SILENCE, in the panel's error convention already (b(s)
                     # IS an error: the per-slot mean of the head's own execution error against
                     # what `apply_any` would have written). One scalar: the mean of b(s) over
                     # the slots that are OPEN this cycle, i.e. over the executor the agent is
                     # actually running. `None` before any slot opens — the gauge does not
                     # exist yet, which is a fact about the signal's TYPE and is why `dsil`
                     # cannot pace the L2 commit. Logged in EVERY arm, driven in two.
                     "dsil": _dsil_now, "dsil_level": int(active),
                     "dsil_n": int(_dsil_n), "dsil_open": int(_dsil_open),
                     "endo": e_par, "endo_level": int(era["level"] + 1),
                     "endo_cell": e_cell,
                     # [conductor] THE EXCESS FORM, and why it exists. `endo_yield`'s read was
                     # an EXCESS (`excess_by_level_rand`), and its SPEC says so explicitly: the
                     # additive constant "cancels exactly in the paired contrast". That
                     # cancellation is a property of the DONOR'S TWO-CONDITION block, and this
                     # round's action is absorbing, so there is no second condition and nothing
                     # cancels — a raw NLL read therefore inherits the plant's own global drift.
                     # Measured on `cd_ef/anchor`: the raw parent-span NLL DEGRADES over the run
                     # (mean D -0.0015 against a floor of 0.0108, and the sign gets worse at
                     # every longer horizon: -0.14 -> -0.34 -> -0.53 -> -0.63 -> -0.81 at spans
                     # 1/2/3/4/6, with frac(D>tol) falling to 0.00). So the cancellation has to
                     # be built into the READ instead of into the block: parent minus cell is
                     # one level up RELATIVE to within level, the plant's global drift is common
                     # to both terms and cancels, and it costs nothing extra (both terms come
                     # from forward passes the panel already makes). Same tree arithmetic, same
                     # label-free property, same price. BOTH are logged and either can be
                     # driven, so the counterfactual trace for the other always exists.
                     "endo_excess": (None if (e_par is None or e_cell is None)
                                     else e_par - e_cell),
                     # [maestro] LEVEL STATE, read at the decision instant. `will_commit` is
                     # the learned policy's table index and is exactly block (g)'s own commit
                     # guard, evaluated HERE — before (g) runs — so a commit landing on this
                     # cycle is not yet in `committed`. `fit.py` reconstructs it the same way
                     # (`cc < cycle`), which is what keeps the offline fit and the live policy
                     # in the same state. Logged in EVERY arm, uncharged, like the panel.
                     # [crescendo] tracks block (g)'s guard, which now reads `commit_maxl`, so
                     # the level-state index stays exactly "would a firing install a table".
                     "will_commit": int(active <= commit_maxl
                                        and committed.get(active) is None),
                     "commit_max_level": int(commit_maxl),
                     "n_committed": int(sum(1 for _q in range(2, maxl + 1)
                                            if committed.get(_q) is not None)),
                     "active_level": int(active),
                     "at_support": {str(k): (vv[-1] if vv else None)
                                    for k, vv in obs_hist.items()},
                     "gauge_hist": {str(k): (vv[-1] if vv else None)
                                    for k, vv in gauge_hist.items()}}
            panel_hist.append(panel)
            # the ledger charges the POLICY's input stream and nothing else. A priced read is
            # folded into `counts["ground"]`, so it surfaces in `t_cum` — a readout that nothing
            # in this loop consumes, which is why pricing cannot change what any arm does.
            #
            # A gauge that does not exist yet is not a decision point: the loop reads nothing,
            # is charged nothing, and its clock does not advance. (In practice every driven read
            # is defined from cycle 1 here; the guard is so a missing gauge fails loudly in the
            # trace instead of silently as a zero.)
            read_val = panel.get(loop.read_key) if driven else 0.0
            if driven and read_val is None:
                linfo = {"rule": "quiet", "cycle": cyc, "read": loop.read_key, "e": None,
                         "quiet": False, "skipped": "read undefined this cycle"}
            else:
                charged = ledger.charge(loop.needs_at(cyc))
                if charged:
                    counts["ground"] += int(charged)
                linfo = loop.step(cyc, panel)
            # [caesura] THE DETECTOR, read-only: same cycle, same panel, A1's rule unchanged.
            # Stepped OUTSIDE the driven/undriven split, because a veto arm may be paced by
            # `yield` or by the schedule and the conjunction has to be evaluated either way —
            # and stepped only when the gauge EXISTS, on the donor's own idiom two lines up: a
            # cycle whose read is undefined is not a decision point, so handing the thermostat
            # a None would not merely crash it (it does), it would invent a decision.
            if dsil_det is not None:
                if panel.get("dsil") is None:
                    linfo = {**linfo, "dsil_quiet": False, "dsil_V": None,
                             "dsil_read": None, "dsil_skipped": "gauge absent"}
                else:
                    _dinfo = dsil_det.step(cyc, panel)
                    linfo = {**linfo, "dsil_quiet": bool(_dinfo.get("quiet")),
                             "dsil_V": _dinfo.get("V"), "dsil_read": panel.get("dsil")}
            # [tutti] THE COMMIT POLICY, stepped on the same cycle and the same panel with
            # A1's rule unchanged. The donor's guard applies PER POLICY: a cycle whose read is
            # undefined is not a decision point for the policy that reads it, so `loop_c` is
            # stepped only where its own gauge exists. `cq` is the commit licence the block (g)
            # dispatch reads; `None` means "this policy could not speak this cycle", which is
            # what the bootstrap branch is for.
            cq, cinfo = None, None
            if loop_c is not None:
                _cread = panel.get(loop_c.read_key)
                if _cread is None:
                    linfo = {**linfo, "c_quiet": None, "c_V": None, "c_read": None,
                             "c_key": loop_c.read_key, "c_skipped": "read undefined this cycle"}
                else:
                    charged_c = ledger.charge(loop_c.needs_at(cyc))
                    if charged_c:
                        counts["ground"] += int(charged_c)
                    cinfo = loop_c.step(cyc, panel)
                    cq = bool(cinfo.get("quiet"))
                    linfo = {**linfo, "c_quiet": cq, "c_V": cinfo.get("V"),
                             "c_read": _cread, "c_key": loop_c.read_key,
                             "c_v_tol": cinfo.get("v_tol"), "c_v_mult": cinfo.get("v_mult")}
            linfo["era"] = era_i + 1
            linfo["c_in_era"] = c_in_era

            # --- (e) THE SHADOW AUDITION: what the candidate macro would score, every cycle.
            #         `cand` is the earned table; `true` is the DGP's own (an oracle readout,
            #         unpriced); `held` is the frozen committed one (the counterfactual recert
            #         readout — what freezing costs, measured but never acted on).
            sr_np, sx = shadow[era_i]
            aud = {}
            for ell in range(2, maxl + 1):
                span = s ** (ell - 1)
                node = (era["node"] * s ** (era["level"] - 1)) // span
                cell = {}
                # the LIVE candidate, built over whatever this arm's level-(l-1) vocabulary
                # actually is: the frozen one if it committed (the ratchet), the live mined one
                # otherwise, the DGP's own for `given`. Computed for EVERY arm so the
                # compounding question ("does level 3 descend faster given level 2?") is a
                # four-way comparison rather than a within-arm anecdote.
                tbl = miners[ell].build(operative(ell - 1), cfg["mine_support"])
                cell["n_entries"] = int(tbl["child"].shape[0])
                if cell["n_entries"]:
                    mv = MC.to_device(MC.make_macro(ell, node, s, tbl), device)
                    cell["cand"] = audition_macro(generator, sx, sr_np, mv, rules_t, canon,
                                                  depth, v, m, s, rules)["e"]
                    cell.update({f"tab_{k}": val for k, val in
                                 MC.grade_table(tbl, shared["truth"][ell]).items()})
                    if ell == active and committed.get(ell) is None:
                        counts["ground"] += sx.shape[0]      # priced: the agent's own check
                        counts["mat"] += sx.shape[0]
                else:
                    cell["cand"] = None
                mvt = MC.to_device(MC.make_macro(ell, node, s, shared["truth"][ell]), device)
                cell["true"] = audition_macro(generator, sx, sr_np, mvt, rules_t, canon,
                                              depth, v, m, s, rules)["e"]
                # MATCHED-SIZE RANDOM CONTROL (oracle instrument, unpriced). `cal_ladder`
                # measured that a k-entry table's audition has huge entry-IDENTITY variance at
                # small k (L3 k=4 spans 0.10-0.97 over uniform draws). If mining recovered a
                # random subset the earned vocabulary would be a lottery; if it beats the
                # matched-size random control, practice mines the USEFUL part of the
                # vocabulary first, which is a claim about practice rather than about size.
                if cell["n_entries"]:
                    full = shared["truth"][ell]
                    n_all = full["child"].shape[0]
                    k = min(cell["n_entries"], n_all)
                    es = []
                    for _ in range(3):
                        keep = np.sort(rng.permutation(n_all)[:k])
                        sub = MC.make_table(ell, full["child"][keep], full["lower"], s)
                        mvr = MC.to_device(MC.make_macro(ell, node, s, sub), device)
                        es.append(audition_macro(generator, sx, sr_np, mvr, rules_t, canon,
                                                 depth, v, m, s, rules)["e"])
                    cell["rand_k"] = float(np.mean(es))
                    cell["rand_k_sd"] = float(np.std(es))
                if committed[ell] is not None and spec["vocab"] == "earned":
                    # the counterfactual recert: what the frozen unit scores now against what
                    # the live vocabulary would score. Measured, never acted on -- committed
                    # macros stay frozen, which is what makes the poison test a real test.
                    mvc = MC.to_device(MC.make_macro(ell, node, s, committed[ell]), device)
                    cell["held"] = audition_macro(generator, sx, sr_np, mvc, rules_t, canon,
                                                  depth, v, m, s, rules)["e"]
                    live = miners[ell].build(MC.base_table(v) if ell == 2
                                             else miners[ell - 1].build(MC.base_table(v),
                                                                        cfg["mine_support"]),
                                             cfg["mine_support"])
                    if live["child"].shape[0]:
                        mvl = MC.to_device(MC.make_macro(ell, node, s, live), device)
                        cell["live"] = audition_macro(generator, sx, sr_np, mvl, rules_t, canon,
                                                      depth, v, m, s, rules)["e"]
                        cell["live_entries"] = int(live["child"].shape[0])
                aud[str(ell)] = cell

            # --- (f) THE UNIT-LP CERTIFICATE: silence on the audition trajectory of the
            #         committable content, with an explicit "it descended first" precondition.
            A = aud.get(str(active), {}).get("cand")   # None until the vocabulary is non-empty
            fire_cert = False
            if A is not None and committed.get(active) is None:
                if cert["ref"] is None:
                    cert["ref"] = A; cert["b"] = A; cert["emin"] = A
                cert["emin"] = min(cert["emin"], A)
                d = cert["b"] - A
                scale = max(cert["ref"] - cert["emin"], cert["b"], 1e-6)
                cert["hist"].append(A); cert["dhist"].append(d)
                we, wd = cert["hist"][-cfg["sil_W"]:], cert["dhist"][-cfg["sil_W"]:]
                quiet = (len(we) >= cfg["sil_W"]
                         and abs(float(np.mean(wd))) < cfg["sil_c"] * scale
                         and float(np.std(we)) < cfg["sil_cv"] * scale)
                cert["run"] = cert["run"] + 1 if quiet else 0
                cert["b"] = cert["b"] + cfg["alpha"] * (A - cert["b"])
                dropped = (cert["ref"] - cert["emin"]) >= cfg["lp_min_drop"]
                fire_cert = (cert["run"] >= cfg["sil_hold"] and dropped
                             and c_in_era >= cfg["sil_min_cycle"])

            # --- (f2) [spiral] THE SHADOW CERTIFICATE, per level, every cycle, in EVERY arm.
            #          The donor's certificate above stops being evaluated the moment a level
            #          commits — which is fine when the certificate IS the commit rule, and
            #          useless under `delta_prov`, where a level can commit AT THE BOUNDARY
            #          without the certificate ever having fired. Cycles-to-certification is
            #          this round's primary rate readout, so it has to be observable
            #          independently of what actually caused the commit, in enum arms and
            #          native arms alike.
            #
            #          Run on the LIVE candidate audition for every earnable level (the same
            #          `aud[l]["cand"]` series the donor's certificate reads at the active
            #          level), with the donor's silence test verbatim. Read-only: nothing here
            #          feeds `do_commit`, so the commit path is bit-identical to the donor's.
            for ell in range(2, maxl + 1):
                Ash = aud.get(str(ell), {}).get("cand")
                if Ash is None:
                    continue
                sc_ = shadow_cert[ell]
                if sc_["ref"] is None:
                    sc_["ref"] = Ash; sc_["b"] = Ash; sc_["emin"] = Ash
                    sc_["c0"] = cyc
                sc_["emin"] = min(sc_["emin"], Ash)
                d_ = sc_["b"] - Ash
                scale_ = max(sc_["ref"] - sc_["emin"], sc_["b"], 1e-6)
                sc_["hist"].append(Ash); sc_["dhist"].append(d_)
                we_, wd_ = sc_["hist"][-cfg["sil_W"]:], sc_["dhist"][-cfg["sil_W"]:]
                quiet_ = (len(we_) >= cfg["sil_W"]
                          and abs(float(np.mean(wd_))) < cfg["sil_c"] * scale_
                          and float(np.std(we_)) < cfg["sil_cv"] * scale_)
                sc_["run"] = sc_["run"] + 1 if quiet_ else 0
                sc_["b"] = sc_["b"] + cfg["alpha"] * (Ash - sc_["b"])
                dropped_ = (sc_["ref"] - sc_["emin"]) >= cfg["lp_min_drop"]
                if (sc_["fired"] is None and sc_["run"] >= cfg["sil_hold"] and dropped_
                        and c_in_era >= cfg["sil_min_cycle"]):
                    sc_["fired"] = cyc
                    sc_["fired_era"] = era_i + 1
                    sc_["fired_c_in_era"] = c_in_era
                    sc_["fired_active"] = bool(ell == active)
                    print(f"[cert]   arm={arm} SHADOW certificate L{ell} FIRED c{cyc} "
                          f"(era {era_i + 1} c_in_era {c_in_era}, active level {active}, "
                          f"committed={committed.get(ell) is not None})", flush=True)

            # --- (g) the commit rule ------------------------------------------------------
            do_commit, prov = False, False
            # [crescendo] `commit_maxl`, not `maxl`. The ONE bit that separates the treatment
            # from its ceiling control: whether a firing at the active level may install a
            # table. Everything upstream of this line — the gauge, the miner, the panel, the
            # audition, the RNG stream — is identical in both arms.
            if spec["commit"] and active <= commit_maxl and committed.get(active) is None:
                # [spiral] `at_boundary` and the two policies below are `ear/ear.py`'s, which
                # `recital` then carried: the arc's rescue for a certificate that goes quiet
                # for a substrate reason rather than a learning one. `delta_prov` commits on
                # the certificate if it fires and AT THE ERA BOUNDARY otherwise, marking the
                # boundary case PROVISIONAL — which skips the priced pre-commit audition,
                # because the point of a provisional commit is to pay for grading where the
                # unit is CONSUMED (the live recert, block (g2)) rather than before it has
                # been. This is the policy the SPEC fixes across every spiral arm.
                at_boundary = c_in_era >= ec - cfg["prov_offset"]
                # [census] THE GAUGE, read at the gate level only. G-A on the at-support
                # series: quiet iff the level admitted no more than `gate_theta` new distinct
                # tuples at support over the trailing `gate_W` cycles. Phase 0 R5's constants
                # (W=12, theta=0) are the only grid corner that fired after the certificate in
                # all four `sp_s0` earning arms; that firing cycle was an extrapolation, which
                # is exactly why the rule below is a CONJUNCTION and not a replacement.
                gate_lv = cfg.get("gate_level")
                gate_quiet = None
                if gate_lv and active == int(gate_lv):
                    h = gauge_hist[active]
                    W_, th_ = int(cfg["gate_W"]), int(cfg["gate_theta"])
                    gate_quiet = bool(len(h) > W_ and (h[-1] - h[-1 - W_]) <= th_)
                if spec["commit"] == "delta":
                    do_commit = fire_cert
                elif spec["commit"] == "early":
                    do_commit = c_in_era >= cfg["early_offset"]
                elif spec["commit"] == "late":
                    do_commit = c_in_era >= ec - cfg["late_offset"]
                elif spec["commit"] == "prov":
                    do_commit, prov = at_boundary, True
                elif spec["commit"] == "delta_prov":
                    # [census] at the gate level the licence is the CONJUNCTION; everywhere
                    # else this is exactly the donor's rule, unchanged.
                    lic = bool(fire_cert and (gate_quiet if gate_quiet is not None else True))
                    do_commit = bool(lic or at_boundary)
                    prov = not lic
                elif (cfg.get("dsil_bootstrap") and spec["commit"] == "loop"
                      # [tutti] the bootstrap belongs to whichever policy OWNS THE COMMIT: the
                      # second object when the split is on, the primary otherwise. `_c_owner`
                      # is `loop_c or loop`, and the branch is entered exactly when the commit
                      # owner reads `dsil` and the gauge does not exist this cycle. With
                      # `loop_commit` absent this is the donor's condition, character for
                      # character.
                      and getattr((loop_c or loop), "read_key", None) == "dsil"
                      and panel.get("dsil") is None):
                    # [caesura] THE BOOTSTRAP, and why the arm cannot exist without it.
                    #
                    # delta-silence is computed over the OPEN SLOTS, and slots are minted BY a
                    # commit. So before the first commit the gauge does not exist, the donor's
                    # own "read undefined this cycle" guard correctly skips the thermostat, the
                    # loop never acts, no slot is ever minted — and the arm DEADLOCKS. That is
                    # not a scale artifact: it is the type structure of the signal, and gate
                    # D-1 shows it directly (`dsil_read` at preflight: 0 cycles with dsil, 0
                    # open slots, 0 commits, at every scale).
                    #
                    # So the honest statement of what this arm can be asked is: delta-silence
                    # CANNOT license the first crossing — it can only license crossings above
                    # the level whose executor it can hear. While the gauge is absent the arm
                    # falls back to the arc's own default rule (`delta_prov`, certify-else-
                    # boundary), and every commit taken that way is LOGGED AS A BOOTSTRAP so
                    # the reduction never counts it as a delta-silence firing.
                    lic = bool(fire_cert and (gate_quiet if gate_quiet is not None else True))
                    do_commit = bool(lic or at_boundary)
                    prov = not lic
                    if do_commit:
                        dsil_events.append({"cycle": cyc, "era": era_i + 1,
                                            "level": int(active), "kind": "bootstrap",
                                            "dsil": None, "dsil_open": 0,
                                            "why": ("cert" if lic else "boundary")})
                        print(f"[dsil]   arm={arm} c{cyc} BOOTSTRAP L{active} commit "
                              f"({'cert' if lic else 'boundary'}) — the gauge does not exist "
                              f"until a slot is open", flush=True)
                elif spec["commit"] == "loop":
                    # [conductor] THE OUTER LOOP OWNS THE COMMIT. Not a conjunction with the
                    # certificate and not a fallback to the era boundary: the loop drives, the
                    # anchor is the comparator arm, and the shadow certificate keeps running
                    # read-only beside both so cycles-to-cert stays observable. (§4.1 reads the
                    # loop as REPLACING the certificate; conjoining them is what `census_gate`
                    # did, and finding 4 is that the conjunction contributed nothing beyond a
                    # cycle number.) A loop commit is NOT provisional — the loop chose it, so it
                    # pays the pre-commit audition exactly as a certified commit does.
                    #
                    # [tutti] THE SPLIT'S ONE LINE. With `loop_commit` absent this reads
                    # `loop.quiet` — the donor exactly. With it present the COMMIT licence is
                    # the second policy's latch and the ADVANCE licence (block (g4)) stays the
                    # primary's, which is the whole of the division-of-labor treatment. A yoked
                    # arm ignores both and replays the plan's COMMIT cycles, which is what lets
                    # a two-policy arm be clock-yoked at all.
                    if loop.kind == "yoke":
                        do_commit = bool(loop.commit_now(cyc))
                    elif loop_c is not None:
                        do_commit = bool(cq)          # None (gauge absent) -> False
                    else:
                        do_commit = bool(loop.quiet)
                    prov = False
                elif spec["commit"] == "yoked":
                    # [census] the timing control: at the gate level commit at the cycle
                    # `census_gate` actually committed, by clock and not by signal; at every
                    # other level fall back to the shared policy so the arm differs from the
                    # anchor by exactly one thing.
                    yk = cfg.get("yoke_cycle")
                    if gate_lv and active == int(gate_lv) and yk:
                        do_commit = bool(cyc >= int(yk))
                        prov = bool(cfg.get("yoke_provisional", False))
                    else:
                        do_commit = bool(fire_cert or at_boundary)
                        prov = not fire_cert
            surg_rec = None
            # [caesura] THE VETO. delta-silence gates commit TIMING and nothing else: the yield
            # thermostat still chose the rung and the cycle, and a commit it licensed is
            # DEFERRED on any cycle the executor is still moving. Deliberately NOT counted as a
            # loop action — the rule did not spend its firing, it was held — which is the
            # difference between "delta-silence delayed the commit" (this arm) and
            # "delta-silence replaced the gauge" (`dsil_read`). Before any slot is open the
            # gauge does not exist, and the veto is INERT by construction rather than blocking:
            # a signal that cannot speak must not be read as saying no. Every deferral is
            # logged with the read that caused it.
            if (dsil_det is not None and do_commit
                    and panel.get("dsil") is not None and not linfo.get("dsil_quiet")):
                dsil_events.append({"cycle": cyc, "era": era_i + 1, "level": int(active),
                                    "kind": "defer", "dsil": panel.get("dsil"),
                                    "dsil_open": panel.get("dsil_open"),
                                    "V": linfo.get("dsil_V")})
                print(f"[dsil]   arm={arm} c{cyc} DEFER L{active} commit — executor still "
                      f"moving (dsil={panel.get('dsil')}, V={linfo.get('dsil_V')})", flush=True)
                do_commit = False
            elif dsil_det is not None and do_commit:
                dsil_events.append({"cycle": cyc, "era": era_i + 1, "level": int(active),
                                    "kind": ("pass" if panel.get("dsil") is not None
                                             else "absent"),
                                    "dsil": panel.get("dsil"),
                                    "dsil_open": panel.get("dsil_open"),
                                    "V": linfo.get("dsil_V")})
            loop_tried_commit = bool(do_commit and spec["commit"] == "loop")
            if do_commit:
                tbl = miners[active].build(operative(active - 1), cfg["mine_support"])
                if tbl["child"].shape[0] == 0:
                    do_commit = False
            # [conductor] a loop commit the empty-table guard cancelled still COUNTS AS AN
            # ACTION for the loop's clock: the rule licensed one, the substrate refused it, and
            # the rule must re-arm before licensing another. Without this the loop would fire
            # again on the very next cycle from the same quiet reading.
            if loop_tried_commit and not do_commit:
                _acted_all(PO.COMMIT, cyc, why="empty_table")       # [tutti]
                loop_actions.append({"cycle": cyc, "era": era_i + 1, "kind": PO.COMMIT,
                                     "level": int(active), "why": "quiet",
                                     "cancelled": "empty_table",
                                     "V": linfo.get("V"), "v_tol": linfo.get("v_tol"),
                                     "v_mult": linfo.get("v_mult"),
                                     "bucket": linfo.get("bucket"), "a": linfo.get("a"),
                                     "per_gauge": linfo.get("per_gauge")})
                print(f"[loop]   arm={arm} c{cyc} L{active} COMMIT licensed but the mined "
                      f"table is empty — cancelled, loop re-armed", flush=True)
            # ---- [assay] ORACLE SURGERY ------------------------------------------------- #
            # Placed HERE, before the commit body, for two reasons. First, the donor's
            # empty-table cancellation already lives at this point, so a surgery that empties a
            # table (`strip`'s L3 mined table is only a few true entries, and an empty table has
            # no entry axis for the DP's argmax) cancels the commit through the existing path
            # instead of installing an unservable macro. Second, the priced pre-commit audition
            # in the body below then grades WHAT IS ACTUALLY COMMITTED, which is what that
            # audition is for, and keeps its meaning identical across arms.
            # The shadow certificate is untouched, so what the certificate would have done with
            # the mined table is still recorded beside what the surgery installed.
            if do_commit and cfg.get("surgery"):
                lower_x = MC.base_table(v) if active == 2 else committed[active - 1]
                mined_grade = MC.grade_table(tbl, shared["truth"][active])
                tbl, surg_rec = apply_surgery(
                    cfg["surgery"], active, tbl, lower_x, shared["truth"][active],
                    shared.get("junk_pool") or {}, s, truth_full=shared["truth"])
                surg_rec.update({"arm": arm, "cycle": cyc, "era": era_i + 1,
                                 "mined_recall": mined_grade["recall"],
                                 "mined_precision": mined_grade["precision"]})
                if int(tbl["child"].shape[0]) == 0:
                    surg_rec["cancelled_empty"] = True
                    do_commit = False
                    print(f"[surgery] arm={arm} c{cyc} L{active} mode={cfg['surgery']} "
                          f"produced an EMPTY table — commit cancelled", flush=True)
                else:
                    print(f"[surgery] arm={arm} c{cyc} L{active} mode={cfg['surgery']} "
                          f"own={surg_rec['n_own']}({surg_rec['n_own_true']} true) -> "
                          f"{surg_rec['n_after']} added={surg_rec['n_added']}"
                          f"({surg_rec['n_added_true']} true) "
                          f"removed={surg_rec['n_removed']} K={surg_rec.get('K')} "
                          f"unbuildable={surg_rec['n_unbuildable']} "
                          f"recall={surg_rec.get('tab_recall')} "
                          f"prec={surg_rec.get('tab_precision')}", flush=True)
                events.append({"kind": "surgery", **surg_rec})
            if do_commit:
                # a fresh held-out audition, PRICED — the agent pays to verify what it commits.
                # [spiral] a PROVISIONAL commit deliberately skips it (ear's rescue): the unit
                # is graded where it is consumed, by the live recert, instead of before it has
                # been consumed at all.
                held = None
                if not prov:
                    cr_np, cx_np = context_instances(rules, era_ctx(era), cfg["n_score"], s,
                                                     depth, v, m,
                                                     seed=cfg["seed"] + 700_000 + 1000 * cyc)
                    cx = torch.from_numpy(cx_np).to(device)
                    span = s ** (active - 1)
                    node = (era["node"] * s ** (era["level"] - 1)) // span
                    mv = MC.to_device(MC.make_macro(active, node, s, tbl), device)
                    held = audition_macro(generator, cx, cr_np, mv, rules_t, canon, depth,
                                          v, m, s, rules)
                    counts["ground"] += cx.shape[0]; counts["mat"] += cx.shape[0]
                # [spiral] THE AUDITION CALIBRATION INSTRUMENT (the grader wall). Under
                # `delta_prov` a boundary commit deliberately buys no audition, so `audition`
                # is None exactly where the calibration question is most interesting. This is
                # the same one-action held-out audition of the committed table, computed on the
                # era's fixed shadow set, ORACLE and UNPRICED — it is never read by the loop,
                # only logged, so it cannot change what the agent does. It is what makes
                # "what did the grader think this unit was worth, and what was it worth?"
                # answerable at both levels for every commit, priced or provisional.
                span_o = s ** (active - 1)
                node_o = (era["node"] * s ** (era["level"] - 1)) // span_o
                mv_o = MC.to_device(MC.make_macro(active, node_o, s, tbl), device)
                aud_oracle = audition_macro(generator, shadow[era_i][1], shadow[era_i][0],
                                            mv_o, rules_t, canon, depth, v, m, s, rules)["e"]
                w_before, ms_len_before = int(width), len(ms)
                committed[active] = tbl
                ms = build_ms(base_ms, committed, s, depth, device)
                mint(active, tbl)              # the corridor becomes the head's responsibility
                bind_slots(ms)
                if prop is not None:
                    # the head has never had a training example on the new slots. Their output
                    # rows are still at their zero init, so they enter the ranking neutrally
                    # (above whatever the head learned to reject, below what it learned to
                    # accept) — and for `prop_new_cycles` cycles they are expanded regardless,
                    # priced at their true cost, so an untrained logit cannot lock them out.
                    for slot in PN.move_slots(ms, offsets):
                        move_age.setdefault(slot, -1)
                cert["fired"] = cyc
                # [spiral] THE PRICING READOUT AT EVERY COMMIT, both levels: what the declared
                # budget buys before and after the action set grows, and the `prop_new_cycles`
                # forced-expansion window's own transient (k_eff jumps by the number of new
                # nodes, so the routed width collapses for exactly that many cycles). Phase A
                # measured this at L2 — 18 -> 3 -> 18 under routing against a PERMANENT 2 -> 1
                # under enumeration — and it has to be readable at L3 too.
                port_after, k_after = port_spec()
                w_after = cap_w(fit_width(len(ms), cfg["budget"], cfg["g_budget"])
                                if port_after is None
                                else PN.fit_width_k(k_after, cfg["budget"], cfg["g_budget"]))
                ev = dict(kind="commit", arm=arm, era=era_i + 1, level=active, cycle=cyc,
                          c_in_era=c_in_era, t_cum=t_cum + priced(counts, cfg),
                          provisional=bool(prov),
                          audition=(None if held is None else held["e"]),
                          aud_oracle=aud_oracle, shadow=A, e_task=e,
                          cert_fired=shadow_cert[active]["fired"],
                          surgery=cfg.get("surgery"),
                          surgery_n_after=(surg_rec or {}).get("n_after"),
                          surgery_recall=(surg_rec or {}).get("tab_recall"),
                          surgery_precision=(surg_rec or {}).get("tab_precision"),
                          gate_level=cfg.get("gate_level"), gate_quiet=gate_quiet,
                          yoke_cycle=cfg.get("yoke_cycle"),
                          gauge_at_sup=(gauge_hist[active][-1] if gauge_hist[active] else None),
                          n_entries=int(tbl["child"].shape[0]),
                          n_moves_before=int(ms_len_before),
                          n_moves_after=len(ms), sil_run=int(cert["run"]),
                          g_budget=cfg["g_budget"],
                          width_before=w_before, width_after=int(w_after),
                          k_eff_before=int(k_eff), k_eff_after=int(k_after),
                          g_per_solve_before=float(gps),
                          # [conductor] what the loop saw at the instant it acted, on the event
                          # itself, so a commit is auditable against its own decision statistic.
                          driver=("loop" if spec["commit"] == "loop" else spec["commit"]),
                          loop_kind=loop.kind, loop_read=loop.read_key,
                          loop_V=linfo.get("V"), loop_v_tol=linfo.get("v_tol"),
                          loop_v_mult=linfo.get("v_mult"), loop_D=linfo.get("D"),
                          loop_read_level=linfo.get("read_level"),
                          # [tutti] WHO LICENSED THIS COMMIT. On a split arm the commit
                          # owner is the second policy, so the donor's `loop_*` columns above
                          # describe the ADVANCE owner and would silently mis-attribute the
                          # event. These are the commit owner's own read at the instant it
                          # fired; `None` on every non-split arm.
                          commit_owner=(loop_c.read_key if loop_c is not None
                                        else getattr(loop, "read_key", None)),
                          commit_V=linfo.get("c_V"), commit_read=linfo.get("c_read"),
                          commit_v_tol=linfo.get("c_v_tol"),
                          commit_v_mult=linfo.get("c_v_mult"),
                          split=bool(loop_c is not None),
                          **{f"tab_{k}": val for k, val in
                             MC.grade_table(tbl, shared["truth"][active]).items()})
                events.append(ev)
                if spec["commit"] == "loop":
                    _acted_all(PO.COMMIT, cyc)                      # [tutti]
                    loop_actions.append(
                        {"cycle": cyc, "era": era_i + 1, "kind": PO.COMMIT,
                         "level": int(active),
                         "why": ("clock" if loop.kind == "yoke" else "quiet"),
                         "V": linfo.get("V"), "v_tol": linfo.get("v_tol"),
                         "v_mult": linfo.get("v_mult"), "c_in_era": c_in_era,
                         # [tutti] the commit owner's own statistic (None off the split)
                         "owner": (loop_c.read_key if loop_c is not None else None),
                         "owner_V": linfo.get("c_V"),
                         "owner_v_tol": linfo.get("c_v_tol"),
                         "owner_v_mult": linfo.get("c_v_mult"),
                         # [maestro] what the LEARNED policy read and weighted at this instant
                         "bucket": linfo.get("bucket"), "a": linfo.get("a"),
                         "per_gauge": linfo.get("per_gauge")})
                print(f"[commit] arm={arm} era{era_i+1} level={active} c{cyc} "
                      f"{'PROVISIONAL ' if prov else ''}entries={ev['n_entries']} "
                      f"recall={ev['tab_recall']:.3f} "
                      f"prec={ev['tab_precision']} audition={ev['audition']} "
                      f"aud_oracle={aud_oracle:.4f} shadow={A} cert={ev['cert_fired']} "
                      f"moves={ms_len_before}->{len(ms)} "
                      f"k_eff={ev['k_eff_before']}->{ev['k_eff_after']} "
                      f"width={w_before}->{ev['width_after']} G={cfg['g_budget']}", flush=True)

            # --- (g2) [spiral] THE LIVE RECERT (`ear/ear.py`'s, transplanted). Grade the
            #          FROZEN committed unit where it is CONSUMED — this era's damage cell is
            #          the consumption distribution for the macro level committed one era ago —
            #          against what the LIVE mined table would score there. Priced, because it
            #          is the agent's own feedback and not an oracle readout; measured but
            #          never acted on, so the ratchet still bites and the poison test is still
            #          a test. Only spiral arms carry it (`spec["recert"]`), so a donor arm's
            #          path through this block is a single falsy lookup.
            if (spec.get("recert") and era["level"] >= 2
                    and committed.get(era["level"]) is not None
                    and c_in_era % cfg["recert_every"] == 0):
                rc_level = era["level"]
                live = miners[rc_level].build(operative(rc_level - 1), cfg["mine_support"])
                xr = shadow[era_i][1][:cfg["n_aud"]]
                rr_ = shadow[era_i][0][:cfg["n_aud"]]
                pf_ = pol(xr, rr_, ms)
                counts["ground"] += pf_["counts"]["ground"]
                counts["mat"] += pf_["counts"]["mat"]
                rec = {"kind": "recert", "arm": arm, "era": era_i + 1, "level": rc_level,
                       "cycle": cyc, "c_in_era": c_in_era, "e_frozen": pf_["e"],
                       "n_frozen": int(committed[rc_level]["child"].shape[0]),
                       "n_live": int(live["child"].shape[0]), "swapped": False}
                if live["child"].shape[0]:
                    ms_live = build_ms(base_ms, {**committed, rc_level: live}, s, depth, device)
                    pl_ = pol(xr, rr_, ms_live)
                    counts["ground"] += pl_["counts"]["ground"]
                    counts["mat"] += pl_["counts"]["mat"]
                    rec["e_live"] = pl_["e"]
                    # an identical `e` from two different tables is only informative if the
                    # answers differ, so the discordance is logged rather than assumed.
                    # NOTE for whoever reads a 0 here on a span arm: once a slot is OPEN the
                    # head serves that macro call and the table is not consulted at execution
                    # time at all, so swapping the table CANNOT change the answer. That is a
                    # property of the corridor, not a bug — and it is the cleanest available
                    # statement of what "the table survives as the address book" costs.
                    rec["n_diff"] = int((pf_["x"] != pl_["x"]).any(1).sum().item())
                    rec.update({f"live_{k}": val for k, val in
                                MC.grade_table(live, shared["truth"][rc_level]).items()})
                events.append(rec)
                print(f"[recert] arm={arm} era{era_i+1} level={rc_level} c{cyc} "
                      f"e_frozen={rec['e_frozen']:.4f} e_live={rec.get('e_live')} "
                      f"n_frozen={rec['n_frozen']} n_live={rec['n_live']} "
                      f"n_diff={rec.get('n_diff')}", flush=True)

            # --- (g3) [census] COMMIT-THEN-EXTEND ------------------------------------------
            #     The recert loop widened to EVERY COMMITTED LEVEL in every era, and given the
            #     power to ADD entries rather than only to offer a whole-table swap. Scoped to
            #     arms carrying `extend`, so the anchor and the gate arms take the donor's path
            #     untouched and each differs from the anchor by exactly one thing.
            #
            #     SELECTION, NOT APPEND (the SPEC's requirement): each candidate is auditioned
            #     on fresh held-out instances of this era's own cell, through the same
            #     `audition_macro` a commit pays for, and admitted only if it does not raise
            #     the table's audition error. Priced exactly as the commit audition is.
            #
            #     Extension does NOT grow the action set — the macro nodes are unchanged, only
            #     the table behind them — so unlike a commit it carries no width transient and
            #     no `prop_new_cycles` forced window. That is the "~zero timing price" claim,
            #     and the pricing readout is logged either way so it is checked, not assumed.
            if spec.get("extend") and c_in_era % cfg["recert_every"] == 0:
                for lv_x in range(2, maxl + 1):
                    if committed.get(lv_x) is None:
                        continue
                    lower_x = MC.base_table(v) if lv_x == 2 else committed[lv_x - 1]
                    cands = extend_candidates(miners[lv_x], lower_x, cfg["mine_support"],
                                              committed[lv_x])
                    span_x = s ** (lv_x - 1)
                    node_x = (era["node"] * s ** (era["level"] - 1)) // span_x
                    ev_x = {"kind": "extend", "arm": arm, "era": era_i + 1, "level": lv_x,
                            "cycle": cyc, "c_in_era": c_in_era,
                            "n_frozen": int(committed[lv_x]["child"].shape[0]),
                            "n_candidates": len(cands), "n_admitted": 0, "admitted": [],
                            "rejected": 0, "n_auditions": 0}
                    if not cands or not (0 <= node_x < s ** (depth - lv_x)):
                        events.append(ev_x)
                        continue
                    xr_np, xx_np = context_instances(
                        rules, era_ctx(era), cfg["n_aud"], s, depth, v, m,
                        seed=cfg["seed"] + 900_000 + 1000 * cyc + lv_x)
                    xx = torch.from_numpy(xx_np).to(device)

                    def _aud(child_rows, _lv=lv_x, _lo=lower_x, _nd=node_x, _xx=xx,
                             _xr=xr_np):
                        tb = MC.make_table(_lv, np.asarray(child_rows, np.int64), _lo, s)
                        mv_ = MC.to_device(MC.make_macro(_lv, _nd, s, tb), device)
                        return tb, audition_macro(generator, _xx, _xr, mv_, rules_t, canon,
                                                  depth, v, m, s, rules)["e"]

                    base_child = [list(map(int, r)) for r in committed[lv_x]["child"]]
                    _, best_e = _aud(base_child)
                    ev_x["n_auditions"] += 1
                    ev_x["e_before"] = float(best_e)
                    kept = list(base_child)
                    for cand in cands[:int(cfg["extend_cap"])]:
                        trial = kept + [list(map(int, cand["child"]))]
                        _, e_x = _aud(trial)
                        ev_x["n_auditions"] += 1
                        if e_x <= best_e + cfg["extend_tol"]:
                            kept, best_e = trial, e_x
                            ev_x["n_admitted"] += 1
                            ev_x["admitted"].append({"key": list(cand["key"]),
                                                     "count": cand["count"],
                                                     "e_after": float(e_x)})
                        else:
                            ev_x["rejected"] += 1
                    # the agent PAYS for every audition it ran, at the commit audition's rate
                    counts["ground"] += ev_x["n_auditions"] * xx.shape[0]
                    counts["mat"] += ev_x["n_auditions"] * xx.shape[0]
                    ev_x["e_after"] = float(best_e)
                    if ev_x["n_admitted"]:
                        committed[lv_x] = MC.make_table(lv_x, np.asarray(kept, np.int64),
                                                        lower_x, s)
                        refresh_upper(committed, lv_x, s)
                        ms = build_ms(base_ms, committed, s, depth, device)
                        bind_slots(ms)
                        ev_x["n_after"] = int(committed[lv_x]["child"].shape[0])
                        ev_x.update({f"tab_{k}": val for k, val in
                                     MC.grade_table(committed[lv_x],
                                                    shared["truth"][lv_x]).items()})
                    events.append(ev_x)
                    if ev_x["n_candidates"]:
                        print(f"[extend] arm={arm} c{cyc} L{lv_x} cand={ev_x['n_candidates']} "
                              f"admitted={ev_x['n_admitted']} rej={ev_x['rejected']} "
                              f"n {ev_x['n_frozen']}->{ev_x.get('n_after', ev_x['n_frozen'])} "
                              f"e {ev_x['e_before']:.4f}->{ev_x['e_after']:.4f} "
                              f"recall={ev_x.get('tab_recall')} "
                              f"prec={ev_x.get('tab_precision')}", flush=True)

            # --- (g4) [conductor] THE OUTER LOOP'S ERA-ADVANCE DECISION --------------------
            #     The second of the round's two actions, and the one that has no reversible
            #     shadow at all: the era sequence is the world, and there is no way back up the
            #     damage ladder. Precedence is fixed and stated — a quiet reading COMMITS the
            #     active level if there is one to commit, and ADVANCES otherwise; committing
            #     re-arms the loop, so one quiet reading can never fire both.
            #
            #     `era_last` is computed HERE, before the probe block, because the donor's probe
            #     fires on the era's final cycle and under the loop that cycle is not known in
            #     advance. With the loop off `era_last == (c_in_era >= ec) == (c_in_era == ec)`,
            #     which is the donor's trigger exactly.
            #     [tutti] THE SPLIT NEEDS NO CODE HERE, AND THAT IS THE POINT. `loop` IS the
            #     advance owner by definition, so the donor's three lines below are already
            #     the split's advance rule. PRECEDENCE is preserved by the donor's own
            #     mechanism rather than by a new branch: a commit in block (g) calls
            #     `_acted_all(COMMIT, ...)`, which resets BOTH policies, so `loop.quiet` is
            #     False by the time this block reads it — one quiet reading can never fire
            #     both, exactly as in `conductor`. On the MIRROR arm (`loop` reads `dsil`) the
            #     gauge does not exist before a slot opens, so the advance falls through to
            #     `hit_cap` below — and since the ladder is set EQUAL to the caps, the cap is
            #     the arc's own advance bootstrap and lands on the schedule arm's own cycle.
            adv, adv_why = False, None
            if spec["commit"] == "loop" or loop.kind == "yoke":
                if loop.kind == "yoke":
                    adv, adv_why = bool(loop.advance_now(cyc)), "clock"
                elif loop.quiet:
                    adv, adv_why = True, "quiet"
            hit_cap = bool(c_in_era >= ec)
            if hit_cap and not adv and loop.kind != "schedule":
                adv, adv_why = True, "cap"
            era_last = bool(adv or hit_cap)
            if adv and loop.kind != "schedule":
                _acted_all(PO.ADVANCE, cyc, why=adv_why)            # [tutti]
                loop_actions.append(
                    {"cycle": cyc, "era": era_i + 1, "kind": PO.ADVANCE, "level": None,
                     "why": adv_why, "c_in_era": c_in_era,
                     "V": linfo.get("V"), "v_tol": linfo.get("v_tol"),
                     "v_mult": linfo.get("v_mult"),
                     # [maestro] the learned policy's own record at the action
                     "bucket": linfo.get("bucket"), "a": linfo.get("a"),
                     "per_gauge": linfo.get("per_gauge")})
                events.append({"kind": "advance", "arm": arm, "era": era_i + 1, "cycle": cyc,
                               "c_in_era": c_in_era, "why": adv_why, "cap": int(ec),
                               "driver": loop.kind, "read": loop.read_key,
                               "read_level": linfo.get("read_level"),
                               "V": linfo.get("V"), "v_tol": linfo.get("v_tol"),
                               "v_mult": linfo.get("v_mult"),
                               "committed": {str(q): (None if committed[q] is None
                                                      else int(committed[q]["child"].shape[0]))
                                             for q in range(2, maxl + 1)}})
                print(f"[advance] arm={arm} c{cyc} leaving era {era_i + 1} at c_in_era "
                      f"{c_in_era}/{ec} why={adv_why} V={linfo.get('V')} "
                      f"tol={linfo.get('v_tol')}", flush=True)

            # --- (h) probes: the width ladder + all eras' metering sets + the plant guard.
            #         Instruments, never priced.
            probe = None
            if c_in_era == 1 or cyc % cfg["probe_every"] == 0 or era_last:
                # the action set may have grown in (g) this very cycle, so the port is rebuilt
                # here rather than reused: a stale `slots` list would silently make the probe
                # blind to the moves just committed.
                pport, pk = port_spec()
                probe = {"cycle": cyc, "era": era_i + 1, "ladder": {}, "all_eras": {},
                         "k_eff": int(pk), "n_moves": len(ms)}
                for w in cfg["probe_widths"]:
                    bb = plan(controller, generator, value, mx, torch.from_numpy(mr_np),
                              ms, rules_t, canon, depth, v, m, s, budget=cfg["budget"],
                              beam_width=w, device=device, port=pport, ex=ex)
                    sc, _ = grade(bb["x"].cpu().numpy(), mr_np, rules, s)
                    probe["ladder"][str(w)] = {
                        "e": 1.0 - float(sc.mean()),
                        "g": bb["counts"]["ground"] / mx.shape[0],
                        "mat": bb["counts"]["mat"] / mx.shape[0],
                        "prop": bb["counts"].get("prop", 0) / mx.shape[0]}
                for j in range(len(eras)):
                    jr, jx = meter[j]
                    wj = cap_w(fit_width(len(ms), cfg["budget"], cfg["g_budget"])
                               if pport is None
                               else PN.fit_width_k(pk, cfg["budget"], cfg["g_budget"]))
                    bb = plan(controller, generator, value, jx, torch.from_numpy(jr), ms,
                              rules_t, canon, depth, v, m, s, budget=cfg["budget"],
                              beam_width=wj, device=device, port=pport, ex=ex)
                    sc, _ = grade(bb["x"].cpu().numpy(), jr, rules, s)
                    probe["all_eras"][str(j)] = 1.0 - float(sc.mean())
                probe["plant"] = plant_probe(generator, shared, cfg, device)
                # THE SPAN-HEAD-BYPASSED TWIN of the competence readout (span/'s): the same
                # metering beam with the head switched off, so what the port's FIRING costs or
                # buys is separable from what its gradient did to the trunk.
                if use_span and any(sl["open"] for sl in slots.values()):
                    wn = cap_w(fit_width(len(ms), cfg["budget"], cfg["g_budget"])
                               if pport is None
                               else PN.fit_width_k(pk, cfg["budget"], cfg["g_budget"]))
                    ex.fire = False; ex.capture = False
                    bb = plan(controller, generator, value, mx, torch.from_numpy(mr_np), ms,
                              rules_t, canon, depth, v, m, s, budget=cfg["budget"],
                              beam_width=wn, device=device, port=pport, ex=ex)
                    ex.fire = True; ex.capture = True
                    sc, _ = grade(bb["x"].cpu().numpy(), mr_np, rules, s)
                    probe["plant"]["e_nospan"] = 1.0 - float(sc.mean())
                # THE CAN'T-DECOMPOSE SIGNATURE, on the era's own metering roots: where pi puts
                # its mass, and — on the states where it proposes a MACRO — how much mass it
                # keeps on that macro's own primitive decomposition. An expert cannot decompose
                # its chunks. Unpriced instrument.
                if prop is not None:
                    with torch.no_grad():
                        zp = _encode_chunked(controller, mx.to(device))
                    rp = torch.from_numpy(mr_np).to(device)
                    am = avail_mask()
                    probe["pi"] = PN.decompose_probe(prop, zp, rp, ms, offsets, s, maxl, am)
                    bm = PN.base_mass_probe(prop, zp, rp, offsets, am)
                    probe["pi"].update({"top1": bm["top1"], "entropy": bm["entropy"]})
                log["probe"].append(probe)

            t_cum += priced(counts, cfg)
            log["cycle"].append(cyc); log["era"].append(era_i + 1); log["level"].append(era["level"])
            log["t_cum"].append(t_cum); log["e"].append(e); log["succ"].append(float(msucc.mean()))
            log["dres"].append(float(mdres.mean())); log["n_moves"].append(len(ms))
            log["width"].append(width); log["g_per_solve"].append(gps)
            log["e_practice"].append(e_practice); log["vloss"].append(vloss)
            log["gloss"].append(gloss); log["n_solved"].append(int(solved.shape[0]))
            log["n_mined"].append(int(mine_src.shape[0]))
            # [tacet] the gate's own record. `n_pairs`/`pbuf` are appended here rather than in
            # (c') so the diet the gate actually produced is beside its inputs in one row.
            if grec is not None:
                grec["n_pairs"] = int(pinfo.get("n_pairs") or 0)
                grec["pbuf_n"] = int(pinfo.get("n") or 0)
                grec["fallback"] = bool(pinfo.get("fallback"))
                log["gate"].append(grec)
            log["miner"].append({str(ell): miners[ell].state() for ell in range(2, maxl + 1)})
            log["aud"].append(aud)
            log["cert"].append({"A": A, "b": cert["b"], "run": int(cert["run"]),
                                "ref": cert["ref"], "emin": cert["emin"]})
            # [spiral] the COMMITTED table's own recall/precision per cycle, beside its size.
            # `aud[l]["tab_*"]` grades the LIVE candidate; once a level is frozen the two are
            # different tables and the round's crank-health readout wants the frozen one — it
            # is what the policy is actually calling. `grade_table` is a set comparison against
            # the DGP's table: an oracle readout, unpriced, never consumed.
            log["vocab"].append({str(ell): (None if committed[ell] is None
                                            else int(committed[ell]["child"].shape[0]))
                                 for ell in range(2, maxl + 1)})
            # [assay] the entry-identity readout for this cycle, with the truth mask of each
            # committed table alongside so the offline keying is a lookup, not a re-derivation.
            _ENTRY_REC["on"] = False
            ent = _rec_take()
            tmask = {}
            for ell in range(2, maxl + 1):
                if committed.get(ell) is None:
                    continue
                tset = {tuple(int(x) for x in r) for r in shared["truth"][ell]["flat"]}
                tmask[str(ell)] = [int(tuple(int(x) for x in r) in tset)
                                   for r in committed[ell]["flat"]]
            log["entry"].append({"hist": ent, "true_mask": tmask})
            log["gy"].append(gy_miner.state() if gy_miner is not None else None)
            log["committed_grade"].append(
                {str(ell): (None if committed[ell] is None
                            else MC.grade_table(committed[ell], shared["truth"][ell]))
                 for ell in range(2, maxl + 1)})
            pinfo["n_prop"] = int(counts.get("prop", 0))
            pinfo["n_ground"] = int(counts["ground"]); pinfo["n_mat"] = int(counts["mat"])
            pinfo["meter"] = meter_cost          # the PRICED beam's own cost per instance
            log["prop"].append(pinfo)
            log["m_per_solve"].append(b["counts"]["mat"] / mx.shape[0])
            log["blocks"].append(blocks_cycle)
            tb, hb = ex.sizes() if use_span else ({}, {})
            log["span"].append({
                "parity": {k: cell["exact"] for k, cell in par.items()},
                "parity_block": {k: cell["block"] for k, cell in par.items()},
                "n_hold": {k: cell["n"] for k, cell in par.items()},
                "open": {k: bool(sl["open"]) for k, sl in slots.items()},
                "n_open": int(sum(1 for sl in slots.values() if sl["open"])),
                "buf": tb, "hold": hb, "sloss": sloss, "sacc": sacc})
            # ---- [intonation] THE METER'S OWN ROW. Per-slot sums (from which the reduction
            #      recomputes delta at ANY benchmark timescale — S13(c)'s interior optimum,
            #      unmeasured on this substrate, so the series is logged rather than the choice
            #      defended), the 2x2 cells, the gate's calibration, and the counterfactual
            #      misfire price that is deliberately NOT in the ledger.
            if pmeter is not None:
                mrow = {"c": cyc,
                        "slot": {k: {q: (round(x, 6) if isinstance(x, float) else x)
                                     for q, x in cell.items()}
                                 for k, cell in pmeter.acc.items()},
                        "bench": {k: round(float(x), 6) for k, x in pmeter.bench.items()},
                        "g0": pmeter.g0, "theta": pmeter.theta,
                        "calibrated": bool(pmeter.calibrated),
                        "n_call": int(pmeter.n_call),
                        "rows": list(pmeter.rows),
                        "cells": perf_cells[-1] if perf_cells else None,
                        "n_fired": int(blocks_cycle.get("n_fired", 0)),
                        "n_misfire": int(blocks_cycle.get("n_misfire", 0)),
                        "blk_ref": int(blocks_cycle.get("blk_ref", 0)),
                        # never folded into `t`; see the module header, PRICING (b)
                        "t_misfire": round(blocks_cycle.get("n_misfire", 0) * cfg["c_mat"], 4),
                        "wstat": (perf_gain or {}).get("last_wstat")}
                log["perf"].append(mrow)
                if pmeter.calibrate():
                    calib_events.append({"cycle": cyc, **pmeter.cal_event})
                    print(f"[calib]  arm={arm} c{cyc} agency gate g0={pmeter.g0:.5f} "
                          f"theta={pmeter.theta:.5f} "
                          f"(active median {pmeter.cal_event['m_active']:.5f}, "
                          f"passive {pmeter.cal_event['m_passive']:.5f}, "
                          f"n={pmeter.cal_event['n_active']})", flush=True)
                if mrow["n_fired"]:
                    print(f"[perf]   arm={arm} c{cyc} fired={mrow['n_fired']} "
                          f"misfire={mrow['n_misfire']} "
                          f"({mrow['n_misfire'] / max(mrow['n_fired'], 1):.3f}) "
                          f"cells={ {k: v for k, v in (mrow['cells'] or {}).items() if k.startswith(('int_', 'bad_', 'non_'))} }",
                          flush=True)
                pmeter.reset_cycle()
            for slot in list(move_age):
                move_age[slot] += 1
            a_act = aud.get(str(active), {})
            print(f"[c{cyc:3d}] arm={arm:20s} era{era_i+1} t={t_cum:10.0f} e={e:.4f} "
                  f"w={width} nm={len(ms)} k={pinfo.get('k')}/{pinfo.get('k_eff')} "
                  f"solved={solved.shape[0]:4d} "
                  f"A{active}={a_act.get('cand')} true={a_act.get('true')} "
                  f"ent={a_act.get('n_entries')} rec={a_act.get('tab_recall')} "
                  f"pacc={pinfo.get('acc')} sil={cert['run']}", flush=True)
            # [conductor] the loop's own per-cycle record: what it read, what the statistic was,
            # whether it was armed, whether it was quiet. This IS the decision trace the
            # reduction summarises, and it is written for every arm including the schedule one
            # (where it is empty by construction).
            log["loop"].append(linfo)
            log["panel"].append(panel)
            # [tutti] the question row, in every arm (None when the port is off). The VOLUME
            # control (`antiphon`'s, and `wd_s0`'s lesson's second half) is asserted here rather
            # than absorbed: a selector that starves the solve rate mines fewer than `mine_cap`
            # and that is a CONTROL FAILURE, reported as a number, not a shrug. On the `exo`
            # arms it fires as the substrate's own warm-up null, which is what makes it
            # interpretable when it fires on a treated arm.
            if qrow is not None:
                _nm = qrow.get("n_mined")
                if _nm is not None and cfg["mine_cap"] and _nm < int(cfg["mine_cap"]):
                    print(f"[q!]     arm={arm} c{cyc} VOLUME SHORTFALL: mined {_nm} < "
                          f"{cfg['mine_cap']} (solved {qrow.get('n_solved')})", flush=True)
            log["q"].append(qrow)
            if cyc % cfg["checkpoint_every"] == 0:
                write_results(outdir, arm, cfg, eras, refs, log, events, complete=False,
                              extra={"prop_k": prop_k_spec, "span_mode": use_span,
                                     "twin": TWIN.get(base), "slot_events": slot_events,
                                     "gate_events": gate_events,
                                     "shadow_cert": _cert_summary(shadow_cert),
                                     "stream_burn": burn_rec,
                                     "loop_actions": loop_actions,
                                     "ledger": ledger.state()})
            if era_last:
                break
            if total_cap and cyc >= total_cap:
                print(f"[loop]   arm={arm} TOTAL CAP {total_cap} reached inside era "
                      f"{era_i + 1}", flush=True)
                break
        if total_cap and cyc >= total_cap:
            break

    # ------------------------------------------------------------------ #
    # THE ABLATION BATTERY — on the FINAL trained state, after the 90 cycles.
    # ------------------------------------------------------------------ #
    # Run as an end-of-run probe rather than a mid-run switch, so the trajectory every other
    # readout is measured on is never corrupted by the ablation. Unpriced (it is an instrument),
    # but each condition's own per-solve cost is recorded so the conditions are comparable.
    #
    #   a_full   the system as it ran
    #   b_table  THE TABLE IS DELETED. The macro stays in the action set — the policy can still
    #            call the address — but nothing may consult the table at execution time, so
    #            every macro call is served by the span head. POLICY, ON THE RECORD: under this
    #            condition the head serves ALL macro calls regardless of the parity gate, so a
    #            parity shortfall shows up as error rather than being hidden behind a fallback
    #            to the very DP path the condition is supposed to have removed. Arms with no
    #            span head have no execution path at all and are reported as `b_span`.
    #   b_span   table AND span reach removed: pure enumeration over base moves.
    #   c_prims  the primitives are deleted instead; macros only.
    #
    # Next-level currency is split into the two components the prediction confounds:
    #   built        entries `Miner.build` can assemble over the surviving lower table. Under
    #                `b_table` this is 0 BY CONSTRUCTION (T3 subset T2 x T2), so it is the
    #                STRUCTURAL half and carries no information on its own.
    #   at_support   distinct level-1 tuples observed at least `mine_support` times in the
    #                chosen trajectories, keyed by the flat tuple and gated by no table at all.
    #                This is the INFORMATIVE half: does the observation stream of chunk-shaped
    #                spans survive when the routing is native?
    def battery():
        macro_ms = [mv for mv in ms if mv.get("kind") == "macro"]
        base_only = [mv for mv in ms if mv.get("kind") != "macro"]
        conds = {"a_full": (ms, "normal"), "b_table": (ms, "force_head"),
                 "b_span": (base_only, "plain"), "c_prims": (macro_ms, "normal")}
        saved = {k: sl["open"] for k, sl in slots.items()}
        if use_span:
            ex.capture = False
        out = {}
        for cname, (ms_c, mode) in conds.items():
            if not ms_c:
                out[cname] = None
                continue
            if mode == "force_head" and not use_span:
                out[cname] = {"skipped": "no span head — no execution path; see b_span"}
                continue
            if use_span:
                for k_ in slots:
                    slots[k_]["open"] = (mode == "force_head") or (
                        mode == "normal" and saved[k_])
                ex.fire = mode != "plain"
            cell = {"n_moves": len(ms_c), "eras": {}}
            for j, era_j in enumerate(eras):
                jr, jx = meter[j]
                slots_c = PN.move_slots(ms_c, offsets) if prop is not None else None
                if prop is not None:
                    k_c = len(ms_c) if prop_k_spec < 0 else min(int(prop_k_spec), len(ms_c))
                    port_c = {"prop": prop, "slots": slots_c, "k": k_c, "forced": (),
                              "n_slots": n_slots}
                    w_c = cap_w(PN.fit_width_k(k_c, cfg["budget"], cfg["g_budget"]))
                else:
                    port_c, w_c = None, cap_w(fit_width(len(ms_c), cfg["budget"],
                                                       cfg["g_budget"]))
                ex.reset()
                bb = plan(controller, generator, value, jx, torch.from_numpy(jr), ms_c,
                          rules_t, canon, depth, v, m, s, budget=cfg["budget"],
                          beam_width=w_c, device=device, port=port_c, ex=ex)
                sc, _ = grade(bb["x"].cpu().numpy(), jr, rules, s)
                n = jx.shape[0]
                row = {"e": 1.0 - float(sc.mean()), "w": int(w_c),
                       "g_solve": bb["counts"]["ground"] / n,
                       "mat_solve": bb["counts"]["mat"] / n,
                       "blocks": {kk: vv / n for kk, vv in ex.counts.items()}}
                # next-level currency, mined from THIS condition's own chosen trajectories
                keep = bb["x"][torch.from_numpy(sc > 0.5).to(device)]
                mn = {ell: MC.Miner(ell, s) for ell in range(2, maxl + 1)}
                if keep.shape[0]:
                    pf = MC.parse_features(shared["reader"], keep, s=s).cpu().numpy()
                    for ell in range(2, maxl + 1):
                        span_l = s ** (ell - 1)
                        node_l = (era_j["node"] * s ** (era_j["level"] - 1)) // span_l
                        mn[ell].observe(pf[:, node_l * span_l:(node_l + 1) * span_l])
                row["n_solved"] = int(keep.shape[0])
                row["mine"] = {}
                for ell in range(2, maxl + 1):
                    st = mn[ell].state()
                    lower = MC.base_table(v) if ell == 2 else operative(ell - 1)
                    built = mn[ell].build(lower, cfg["mine_support"])["child"].shape[0]
                    row["mine"][str(ell)] = {
                        "at_support": int(st["n_at_support"][str(cfg["mine_support"])]),
                        "n_distinct": int(st["n_distinct"]), "n_obs": int(st["n_obs"]),
                        "built": int(built)}
                cell["eras"][str(j)] = row
                print(f"[ablate] arm={arm} {cname:8s} era{j + 1} nm={len(ms_c)} w={w_c} "
                      f"e={row['e']:.4f} g={row['g_solve']:.1f} solved={row['n_solved']} "
                      f"T2@sup={row['mine']['2']['at_support']} "
                      f"T3@sup={row['mine'].get('3', {}).get('at_support')} "
                      f"T3built={row['mine'].get('3', {}).get('built')}", flush=True)
            out[cname] = cell
        if use_span:
            for k_ in slots:
                slots[k_]["open"] = saved[k_]
            ex.fire = True
            ex.capture = True
        out["_parity_at_end"] = (log["span"][-1]["parity"] if log["span"] else {})
        out["_open_at_end"] = saved
        return out

    print(f"\n----- arm={arm} ABLATION BATTERY (final trained state) -----", flush=True)
    ablation = battery()

    # [conductor] the loop's own summary, computed by the same pure reducer the analyzer and the
    # offline gate use, plus the panel's cross-check that the generalised observation miner
    # reproduces the donor's G-Y instrument at level 4 on every cycle.
    loop_summary = PO.summarise_trace(log["loop"], loop_actions, len(log["cycle"]))
    loop_summary.update({"policy": loop.kind, "read": loop.read_key,
                         "span": cfg["loop_span"], "W": cfg["loop_W"],
                         "burn": cfg["loop_burn"], "alpha": cfg["loop_alpha"],
                         "priced": bool(getattr(loop, "priced", False))})
    # [maestro] a learned arm carries its own fitted policy into its results.json, so a single
    # arm file is self-describing: which reward fitted it, what it weighted, what its dead zone
    # was, and how often each level-state cell was the one in force.
    if loop.kind == "learned":
        bk = [d.get("bucket") for d in log["loop"] if d.get("bucket") is not None]
        loop_summary.update({
            "reward": loop.reward, "fit_id": loop.fit_id, "gauges": list(loop.gauges),
            "mixes": loop.mixes, "thetas": loop.thetas,
            "bucket_cycles": {b: int(bk.count(b)) for b in sorted(set(bk))}})
    obs_extra = {"loop": loop_summary, "loop_actions": loop_actions,
                 "ledger": ledger.state(),
                 "obs_hist": {str(k): q for k, q in obs_hist.items()},
                 "obs_gy_agree": bool(all(obs_gy_agree)) if obs_gy_agree else None,
                 "obs_gy_agree_n": len(obs_gy_agree),
                 "era_caps": list(caps), "total_cap": int(total_cap)}
    print(f"[loop]   arm={arm} {loop.kind}"
          + (f" read={loop.read_key}" if loop.read_key else "")
          + f": {loop_summary['n_commits']} commits at "
            f"{loop_summary['commit_cycles']}, {loop_summary['n_advances']} advances at "
            f"{loop_summary['advance_cycles']} "
            f"({loop_summary['n_chosen']} chosen / {loop_summary['n_capped']} capped); "
            f"spend {ledger.state()['spend_g']}g; "
            f"obs==G-Y on {len(obs_gy_agree)} cycles: "
            f"{bool(all(obs_gy_agree)) if obs_gy_agree else None}", flush=True)
    write_results(outdir, arm, cfg, eras, refs, log, events, complete=True,
                  extra={"prop_k": prop_k_spec, "span_mode": use_span,
                         "twin": TWIN.get(base), "slot_events": slot_events,
                         "gate_events": gate_events, "ablation": ablation,
                         "shadow_cert": _cert_summary(shadow_cert),
                         "stream_burn": burn_rec,
                         "gy_final": (gy_miner.state() if gy_miner is not None else None),
                         "gauge_hist": {str(k): q for k, q in gauge_hist.items()},
                         # [intonation] the meter's own summary objects
                         "perf_mode": (None if pmeter is None else
                                       {"tau_fire": cfg.get("span_tau_fire"),
                                        "alpha": cfg["perf_alpha"],
                                        "gain": cfg.get("perf_gain"),
                                        "gate_mode": cfg.get("gate_mode")}),
                         "calib_events": calib_events,
                         "perf_cells": perf_cells,
                         "dsil_events": dsil_events,          # [caesura]
                         # [tutti] the port's own record and the split's own record, in the
                         # arm's results.json, so the reduction never has to infer either.
                         "q_mode": qmode, "q_k": (k_menu if qmode else None),
                         "q_ledger": (qledger.state() if qledger is not None else None),
                         "split": ({"commit_read": loop_c.read_key,
                                    "advance_read": getattr(loop, "read_key", None),
                                    "reset": "any-action"}
                                   if loop_c is not None else None),
                         **obs_extra})
    return {"log": log, "events": events, "ablation": ablation, "stream_burn": burn_rec,
            "calib_events": calib_events, "perf_cells": perf_cells,     # [intonation]
            "dsil_events": dsil_events,                                 # [caesura]
            "q_mode": qmode, "q_k": (k_menu if qmode else None),        # [tutti]
            "q_ledger": (qledger.state() if qledger is not None else None),
            "split": ({"commit_read": loop_c.read_key,
                       "advance_read": getattr(loop, "read_key", None),
                       "reset": "any-action"} if loop_c is not None else None),
            "shadow_cert": _cert_summary(shadow_cert),
            "gy_final": (gy_miner.state() if gy_miner is not None else None),
            "gauge_hist": {str(k): q for k, q in gauge_hist.items()}, **obs_extra}


def plant_probe(generator, shared, cfg, device):
    """The self-imitation-collapse guard: the plant's fidelity to the grammar, measured on
    CLEAN held-out configurations it never trains on directly.
      infill_acc -- masked-block level-1 feature accuracy (the generator's actual job)
      parse_acc  -- unmasked-block feature accuracy, i.e. how right the agent's own parse is,
                    which is what mining reads. Both are oracle readouts, never consumed."""
    import torch
    v, s = cfg["v"], cfg["s"]
    n_blocks = shared["n_blocks"]
    x = shared["probe_clean"].to(device)
    b = x.shape[0]
    powers = v ** torch.arange(s, device=device)
    feats = shared["bottom_map"][(x.view(b, n_blocks, s) * powers).sum(-1)]
    pick = torch.from_numpy(
        np.random.default_rng(0).integers(0, n_blocks, size=b)).to(device)
    pos = pick[:, None] * s + torch.arange(s, device=device)[None, :]
    with torch.no_grad():
        parse = generator.block_logits(x).argmax(-1)
        obs = x.clone().scatter_(1, pos, torch.full_like(pos, -1))
        infill = generator.block_logits(obs).argmax(-1)
    tgt = feats.gather(1, pick[:, None]).squeeze(1)
    pred = infill.gather(1, pick[:, None]).squeeze(1)
    ok = feats >= 0
    ok1 = tgt >= 0
    with torch.no_grad():
        rd = shared["reader"].block_logits(x).argmax(-1)
    return {"parse_acc": float((parse[ok] == feats[ok]).float().mean()),
            "infill_acc": float((pred[ok1] == tgt[ok1]).float().mean()),
            "read_acc": float((rd[ok] == feats[ok]).float().mean())}


def priced(counts, cfg):
    """ratchet's ledger, plus the proposal reads at `c_prop`. `c_prop` defaults to 0.0: every
    proposal read below the root is a head-only forward on an encoder state the value already
    paid a grounding for, and the one read that is NOT (the root) is charged a full grounding
    inside the beam. The raw count is logged either way so the reduction can re-price it."""
    return (counts["ground"] * cfg["d_fb"] + counts["mat"] * cfg["c_mat"]
            + counts.get("prop", 0) * cfg.get("c_prop", 0.0))


def write_results(outdir, arm, cfg, eras, refs, log, events, complete, extra=None):
    d = os.path.join(outdir, arm)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "results.json"), "w") as fh:
        json.dump({"arm": arm, "config": cfg, "eras": eras, "refs": refs,
                   "log": log, "events": events, "complete": complete, **(extra or {})},
                  fh, indent=2, cls=NumpyEncoder)
    volume.commit()


# --------------------------------------------------------------------------- #
# references, measured at setup ON THE METERING SETS THEMSELVES
# --------------------------------------------------------------------------- #

def measure_refs(shared, cfg, eras, device):
    """`stale` is the detector's scale anchor and round 1's bug: it measured the reference on
    a 512-instance reference draw and the metering on a different 384-instance draw, so the
    two disagreed by more than the descent and the scale collapsed to a much looser fallback.
    Fixed here — every reference below is measured on the ERA'S OWN METERING SET.

    Also measured per era: the exact-DP floor over the base move set, the base-move and
    true-vocabulary performance beams at the declared budget, and the one-action ceilings of
    the true level-l macro (the number `given` is buying and `practice_*` is chasing)."""
    import torch
    rules, rules_t, canon = shared["rules"], shared["rules_t"], shared["canon"]
    v, s, depth, m = cfg["v"], cfg["s"], cfg["depth"], cfg["m"]
    base_ms = shared["base_ms"]
    true_ms = build_ms(base_ms, {ell: shared["truth"][ell]
                                 for ell in range(2, cfg["max_macro_level"] + 1)},
                       s, depth, device)
    out = {"stale": [], "stale_true": [], "floor": [], "d0": [], "on_grammar": [],
           "width_base": [], "width_true": [], "g_base": [], "g_true": [], "macro_true": []}
    for i, era in enumerate(eras):
        r_np, x_np = context_instances(rules, era_ctx(era), cfg["n_rt"], s, depth, v, m,
                                       seed=cfg["seed"] + 5000 + 17 * i)
        x0 = torch.from_numpy(x_np)
        out["d0"].append(float(nearest_derivation_cost(rules, x_np, r_np, s).mean()))
        out["on_grammar"].append(on_grammar_rate(x_np, shared["inverse_maps"][-1], v, s))
        for tag, mset in (("base", base_ms), ("true", true_ms)):
            w = fit_width(len(mset), cfg["budget"], cfg["g_budget"])
            b = beam_moves(shared["controller"], shared["generator0"], shared["value0"], x0,
                           torch.from_numpy(r_np), mset, rules_t, canon, depth, v, m, s,
                           budget=cfg["budget"], beam_width=w, device=device)
            sc, _ = grade(b["x"].cpu().numpy(), r_np, rules, s)
            out["stale" if tag == "base" else "stale_true"].append(1.0 - float(sc.mean()))
            out[f"width_{tag}"].append(w)
            out[f"g_{tag}"].append(b["counts"]["ground"] / x0.shape[0])
        xo, _ = oracle_rollout(shared["generator0"], x0.to(device), r_np, base_ms, rules,
                               rules_t, canon, depth, v, m, s, cfg["budget"])
        sc, _ = grade(xo.cpu().numpy(), r_np, rules, s)
        out["floor"].append(1.0 - float(sc.mean()))
        cell = {}
        for ell in range(2, cfg["max_macro_level"] + 1):
            span = s ** (ell - 1)
            node = (era["node"] * s ** (era["level"] - 1)) // span
            mv = MC.to_device(MC.make_macro(ell, node, s, shared["truth"][ell]), device)
            cell[str(ell)] = audition_macro(shared["generator0"], x0.to(device), r_np, mv,
                                            rules_t, canon, depth, v, m, s, rules)["e"]
        out["macro_true"].append(cell)
    return out


# --------------------------------------------------------------------------- #
# entrypoints
# --------------------------------------------------------------------------- #

def _cfg(**kw):
    cfg = dict(
        v=8, s=2, depth=4, m=2, rule_seed=0, train_seed=1, seed=0, state_dim=96,
        n_train_episodes=100_000, controller_steps=12_000, generator_steps=12_000,
        reader_steps=6_000, plant_holdout=0, holdout_seed=11,
        value_steps=12_000, value_episodes=40_000, value_batch_collect=1024, batch_size=256,
        value_lr=3e-4, value_lr_online=3e-5, n_corrupt=3, explore_eps=0.3,
        budget=4, pr_width=16, g_budget=58, max_macro_level=3,
        n_pr=64, n_rt=384, n_score=512, n_probe_clean=512,
        era_cycles=24, n_grad=4, value_batch=256, replay_frac=0.5, buf_cap=100_000,
        gen_lr=1e-4, gen_steps=20, mine_support=3, mine_from="chosen", mine_cap=0,
        alpha=0.2, sil_c=0.06, sil_cv=0.10, sil_W=5, sil_hold=2, sil_min_cycle=6,
        lp_min_drop=0.05, early_offset=1, late_offset=2,
        d_fb=1.0, c_mat=0.05, checkpoint_every=5, probe_widths=(1, 2, 4, 8, 16), probe_every=4,
        # --- the port ---------------------------------------------------------------------
        prop_lr=3e-4, prop_steps=24, prop_batch=256, prop_buf_cap=60_000,
        prop_warmup=4, prop_new_cycles=2, prop_explore=1, prop_train_on="solved", c_prop=0.0,
        # --- the port 2 knobs (`../span/`'s, unchanged) ------------------------------------
        span_lam=1.0, span_lr=1e-3, span_batch=64, span_buf_cap=8192, span_hold_cap=2048,
        span_hold_frac=0.1, span_capture=24, span_min_hold=256, span_tau=0.95,
        span_hidden_mult=4,
        # --- [spiral] the transplanted commit policy's own knobs (`ear`'s names and defaults).
        #     `prov_offset=0` == the boundary is the era's LAST cycle. `n_aud` is the recert's
        #     grading set size; `recert_every` its cadence within an era.
        prov_offset=0, recert_every=5, n_aud=192,
        # --- [census] this round's own knobs ------------------------------------------------
        # G-Y: the unpriced next-level observation instrument. `gy_level=4` is deliberately
        # ABOVE `max_macro_level=3` — the instrument observes, it never commits.
        gy_level=4,
        # the L2 gate's gauge (Phase 0 R5: the only grid corner that fired after the
        # certificate in all four sp_s0 earning arms).
        gate_level=None, gate_W=12, gate_theta=0,
        # commit-then-extend. `extend_cap` bounds the auditions one recert may buy;
        # `extend_tol` is the admission slack on the audition error (0.0 = must not hurt).
        extend_cap=8, extend_tol=0.0,
        yoke_cycle=None, yoke_provisional=False,
        # --- [tacet] THE GATE'S OWN KNOBS. All four default to the donor's behaviour, so a
        #     config that does not name them produces `crescendo.py` bit-for-bit.
        #     NAME CAUTION, because this substrate now carries three unrelated "gates":
        #       `gate_level`/`gate_W`/`gate_theta`  — census's L2 ADMISSION gate on the commit
        #       `span_tau` + `gate_events`          — span's PARITY gate on a corridor slot
        #       `gate_mode`/`gate_frac`/`gate_mine` — THIS round's LEARNING gate, below
        #     Nothing below reads or writes any key belonging to the other two.
        #       gate_mode   None (off) | delta_hi | delta_lo | delib | random
        #       gate_frac   the fraction of SOLVED TIPS the pi channel keeps. 0.5 is the
        #                   chosen selectivity: comparable to what the mining channel's
        #                   `mine_cap=8` already applies to ~13 solved instances/cycle (0.6),
        #                   round, and gentle enough to leave the replay buffer a real diet.
        #       gate_mine   whether the gate also reorders the mining subsample. On by
        #                   default when a mode is set; the cap is UNCHANGED, so mining volume
        #                   is matched exactly and only its content moves.
        #       gate_log    the per-cycle record of the gate's own inputs. Pure reads of
        #                   tensors the beam already produced; on in EVERY arm, including the
        #                   ungated baseline, so the counterfactual "what would this gate have
        #                   kept here" is computable offline for every arm.
        gate_mode=None, gate_frac=0.5, gate_mine=True, gate_log=True,
        # --- [intonation] THE delta_perf KNOBS. Every one of them defaults OFF/None, so a
        #     config that does not name them produces `tacet.py` bit-for-bit (gate G-F).
        #       span_tau_fire  the FIRING threshold, below `span_tau`. None == the donor
        #                      (fire only at parity). 0.50 is the treatment: `native/`
        #                      finding 4 measured L3 heads ending at 0.76-0.94 exact-match
        #                      parity — fallible and mostly right — while finding 5's
        #                      untrained-head control (parity ~0) costs +0.40-0.51 e. The
        #                      point is a fallible executor, not a vandalised one.
        #       perf_meter     compute e / b(s) / g / delta_perf per execution. Needs `span`.
        #       perf_alpha     the benchmark's EWMA rate, per macro CALL. S13(c) says the
        #                      timescale has an interior optimum (above sampling jitter,
        #                      below competence drift) and nobody has measured where on this
        #                      substrate — so 0.05 is a starting point sized on the smoke's
        #                      measured calls/slot/cycle, and the per-cycle per-slot SUMS are
        #                      logged so the reduction can recompute delta at any alpha.
        #       perf_g0/theta  the centered gate's parameters. None == calibrated in-run from
        #                      the arm's own active/passive g medians (`agency_gate`'s
        #                      protocol: g0 = midpoint, theta = gap/8), self-supervised.
        #       perf_gain      (i) the plasticity consumer: None | "delta" | "raw".
        #       perf_tau_w     f(delta) = exp(-delta/tau_w). CALIBRATED, not guessed, against
        #                      the effect the direction comes from: Kim, Parvin & Ivry 2019
        #                      measure task success ATTENUATING implicit adaptation by ~35%.
        #                      `in_smoke` measured the head's own scale here — misfire rate
        #                      0.04-0.20, so b(s) sits near 0.1 and a correct row earns
        #                      delta ~ +0.1 — and at tau_w = 0.20 that row's weight is
        #                      exp(-0.5) = 0.61, i.e. a 39% attenuation. A missed row
        #                      (delta ~ -0.4) saturates the cap either way, which is the
        #                      categorical half of Kim's result.
        #       perf_w_raw_cap the pre-normalisation cap (Kim's effect is categorical).
        #       perf_w_clip    the post-normalisation clip.
        #       perf_w_ewma    the budget normaliser's own rate.
        #       perf_row_cap   how many raw per-row (e, g, delta, b) samples to keep a cycle.
        span_tau_fire=None, perf_meter=False, perf_alpha=0.05,
        perf_g0=None, perf_theta=None, perf_calib_min=2048,
        perf_gain=None, perf_tau_w=0.20, perf_w_raw_cap=4.0, perf_w_clip=4.0,
        perf_w_ewma=0.05, perf_row_cap=192,
        # --- [caesura] delta-silence as a commit input. Both default OFF, so a config that
        #     does not name them produces `intonation.py` bit-for-bit (gate G-F).
        #       dsil_veto  the CONJUNCTION arm: A1's yield thermostat chooses, delta-silence
        #                  may only delay. Never an action — a deferred commit re-fires as
        #                  soon as the executor quiets.
        #       tol_dsil   delta-silence's MEASURED dead zone. `QuietPolicy` refuses a default,
        #                  which is what forces this to be measured; the sentinel below is a
        #                  placeholder that only the never-driven arms may carry.
        dsil_veto=False, dsil_bootstrap=False, tol_dsil=None,
        # --- [tutti] the question port's two keys, defaulted OFF: with `question_mode=None`
        #     `run_arm` takes the donor's own draw and NOTHING in the port runs, which is the
        #     configuration `fidelity_smoke` gates at 0.000e+00 against `caesura.py`.
        #       question_mode  None | "exo" | "endo" | "bisect" | "novel" | "comp" |
        #                      "comp_free"  (the last four are `questions.py`'s and exist for
        #                      the offline gate suite Q-1..Q-10 and for gate Q-11; only `exo`
        #                      and `endo` are RUN in this tag — the oracle is not an action the
        #                      learner can take, and the two comfort poles are `antiphon`'s
        #                      measured negatives).
        #       question_k     the MENU size. `antiphon` sized K = 2048 offline as the value
        #                      that matches reach to the per-era observation budget at both L3
        #                      and L4; the era caps here are the same, so the sizing carries
        #                      and `phase0_tutti.py` re-checks it rather than assuming it.
        question_mode=None, question_k=2048,
    )
    cfg.update(kw)
    return cfg


def _d6_cfg(**kw):
    """[spiral] `_cfg` with the depth-6 substrate and the two hard prerequisites `tall` paid
    for, so no entrypoint can forget them:

      * `n_corrupt=1` — `dens0` measured this is the WHOLE densification lever (terminal
        success 0.133 against the `tl_s0` config's 0.070; collection budget is not a lever and
        slightly hurts). It is read inside `build_shared`, so the stale value buffer every arm
        starts from is densified by construction.
      * `budget=8, g_budget=482` — `beam_ground(32, 8, 2) = 482`, i.e. exactly width 2 over the
        base action set, and exactly `dens0`'s declared budget so the enum descent rate this
        file measures is comparable to the −0.004/cycle it measured rather than to a
        projection. Under ENUMERATION this budget cannot hold width 2 past the L2 commit
        (n=48 needs >= 722) — which is precisely the pricing inversion routing is claimed to
        dissolve, so raising G here would confound the test that Phase A exists to run.
      * probes re-priced — they were ~47% of `tl_s0`'s runtime (13 beams per firing at ~68 s).
    """
    cfg = _cfg(**DEPTH6)
    # [crescendo] `commit_max_level` caps what may be COMMITTED, independently of
    # `max_macro_level`, which caps what may be MINED / OBSERVED / SLOTTED / AUDITED. `None`
    # means "equal to max_macro_level", i.e. the donor's behaviour exactly — so this file is
    # `maestro.py` unless an arm sets the key.
    cfg.update(commit_max_level=None)
    cfg.update(n_corrupt=1, budget=8, g_budget=482, max_macro_level=3,
               probe_widths=(1, 2, 4), probe_every=8,
               # span_min_hold 256 -> 128. MEASURED reason: in Phase A the composed arm
               # accumulated 184-218 held-out observations per slot in the 6 cycles it had
               # after the L2 commit (~30-35/cycle), so at 256 the parity gate was never even
               # EVALUATED (`parity` returned None on every slot, 0 open in all 22 cycles) and
               # the corridor wall — one of the three walls this round exists to autopsy — was
               # untestable. 128 puts a first parity read ~4 cycles after a commit. The quality
               # guard is unchanged: tau = 0.95 exact-match, re-checked every cycle, not
               # latched, so a slot the moving plant drifts away from closes again.
               span_min_hold=128,
               collect_task_matched=True, tm_episodes=8192,
               # the certificate's own constants are `ear`'s, not the donor's, because the
               # commit policy transplanted here is `ear`'s and these are the values it was
               # calibrated with (`full._cfg` carries 0.10 / 0.05 / no mine cap). Cycles-to-
               # certification is this round's primary rate readout, so the detector has to be
               # the one the arc actually tuned. `mine_cap=8` is also the cap `tall`'s coverage
               # law is stated at (L2 ~6 cycles, L3 ~24, L4 ~384).
               sil_cv=0.15, lp_min_drop=0.10, mine_cap=8,
               # [assay] `surgery` is per-arm (set from each arm's own `cfg`); `entry_rec`
               # switches the per-execution entry-identity instrument, on by default.
               surgery=None, entry_rec=True,
               # [conductor] the outer loop's own knobs.
               #   era_caps       per-era cycle caps for loop-driven and yoked arms; schedule
               #                  arms keep the ladder's own cycle counts untouched.
               #   loop_span/W    the block geometry. span=1 is `floors.py`'s measured choice:
               #                  a longer horizon raises every gauge's signal-over-floor
               #                  (L4 1.49 -> 2.74 at span 3) but needs a 4*span+1 window, and
               #                  the consumption eras are 12/9/7 cycles long, so span 1 is the
               #                  only horizon with decision points in EVERY era.
               #   loop_burn      decision points held after every action and at every era
               #                  start. The regime just changed; it has not been observed yet.
               #   tol_*          THE MEASURED DEAD ZONES (`MEASURED_FLOORS`). Never defaulted:
               #                  `floor_gate` asserts each is present, positive, and equal to
               #                  what `floors.json` derived.
               #   n_endo         sequences per endo read, from the fixed clean probe pool.
               #   endo_price     grounding-equivalents charged per PRICED endo read, measured
               #                  by `endo_bench` on the same GPU and passed in explicitly.
               era_caps=tuple(int(x) for x in CONDUCTOR_CAPS.split(",")),
               total_cap=0, loop_span=1, loop_W=4, loop_burn=4, loop_alpha=0.5,
               tol_ledger=MEASURED_FLOORS["ledger"],
               tol_yield_l3=MEASURED_FLOORS["yield_by_level"][3],
               tol_yield_l4=MEASURED_FLOORS["yield_by_level"][4],
               tol_endo=None, n_endo=256, endo_price=0, yoke_plan=None,
               endo_read_key="endo_excess", obs_panel=True)
    # [census] Phase 0 R6's ladder: the gate is at L2, so era 1 is what must be long enough to
    # contain its firing (latest all-arm L2 firing in the replay was c40, against era 1's old
    # c32 boundary). The cycles come out of era 2, which under R1 no longer hosts a gate, so
    # the total per arm is unchanged at 116 and `sp_s0`'s measured cost carries over.
    cfg.update(eras_default=CENSUS_LADDER)
    cfg.update(kw)
    return cfg


# --------------------------------------------------------------------------- #
# [spiral] PHASE B — the main run
# --------------------------------------------------------------------------- #

# The sized ladder. Levels/nodes are `tall`'s nested cells; the third field is that era's own
# cycle count, from OFFLINE DETECTOR REPLAY on Phase A's own logged audition series plus the
# arc's ~50% margin convention — not from round numbers, and deliberately not from `cal0`'s
# clock-paced figures, which the replay shows are optimistic at this depth.
#
#   THE REPLAY, and why it changed the sizing. Phase A's `sil run` was 0 at every cycle, which
#   looks like a dead detector and is not: its arms committed at c16 under `commit="late"`,
#   and the donor's certificate stops being evaluated the moment a level commits. Re-running
#   the silence test offline over the full 22-cycle audition series (`log["cert"]["A"]`, the
#   same series the shadow certificate reads) at this round's constants
#   (sil_c 0.06 / sil_cv 0.15 / lp_min_drop 0.10):
#
#       L2 certificate fires at c19 (enum arm) and c21 (routed arm).
#
#   `cal0` predicted c12 for L2, so depth-6 certification is ~1.6-1.75x later than the
#   clock-paced calibration suggested. That ratio is the sizing input.
#
#   era 1 (earns L2), 32 cycles — max replayed firing c21, x1.5 margin = 32. 28 would have
#     truncated the L2 certificate, which is exactly the 30-cycle-truncation mistake
#     `ratchet`/`ear` made and `tall`'s design doc warns against repeating. At 32 the L2 commit
#     should be CERTIFIED rather than provisional-at-boundary, and there are ~11 cycles left in
#     the era for the corridor to start accumulating parity data before era 2 begins.
#   era 2 (earns L3), 56 cycles — the binding constraint of the run. `cal0` saw L3 at c22;
#     scaled by the measured 1.7x and given the same 1.5x margin, 53-58. The mining-coverage
#     floor (~24 cycles at mine_cap=8) sits comfortably inside it, and the remaining headroom
#     is spent here on purpose: this is where BOTH the L3 crank and the corridor wall live (a
#     span slot needs ~4 cycles past commit to reach a first parity read at span_min_hold=128,
#     and a parity TRAJECTORY needs many more than one read).
#   eras 3-5 (consumption; L4+ is not earnable at any affordable budget), 12/9/7 — long enough
#     for at least two probes each at probe_every=8, which is what a stable cost-to-depth read
#     needs, and no longer: nothing can be earned there.
#
# 116 cycles/arm. At Phase A's measured per-cycle costs (8.1-13.1 s routed/enum at n=32; 23.4
# s/cycle for the 56-move `given`/`fid` shapes) plus the four extra all-eras probe beams a
# five-era ladder adds, seven arms come to ~4.2 GPU-h — inside the 6 GPU-h cap.
SPIRAL_LADDER = "1:25:32,2:12:56,3:6:12,4:3:9,5:1:7"

# [census] Phase 0 R6. Same 116 cycles, redistributed: era 1 carries the L2 gate and must be
# long enough for it to fire in-era (replayed latest all-arm L2 firing c40 vs era 1's old c32
# boundary, +20% margin => 48); era 2 gives up the difference because under R1 it no longer
# hosts a gate.
CENSUS_LADDER = "1:25:48,2:12:40,3:6:12,4:3:9,5:1:7"

# Arm order is load-bearing, not cosmetic: `census_gate` must run BEFORE `yoked_delay`,
# because the yoked arm's commit cycle is `census_gate`'s MEASURED one, passed in at runtime.
# The anchor runs first so the in-tag replay exists before anything is read against it.
CENSUS_ARMS = ("spiral_route,census_gate,yoked_delay,census_extend,given_route,given_native")

# Arm ORDER is deliberate and is a budget-risk decision, not alphabetical: the ceiling
# reference and the in-tag fidelity assertion first (nothing else is readable without them),
# then the enum crank, then the two treatments, then the g722 rate control, and `given_native`
# last because the SPEC names it the most cuttable (its content is a depth-6 replication of
# `native/`). If the run is cut short, what is lost is what was cheapest to lose.
SPIRAL_ARMS = ("given,fid,enum_live,spiral_route,spiral,enum_live_g722,given_native")


# Arm ORDER is a budget-risk decision: the anchor first (it is the in-tag fidelity carrier
# and nothing else is readable without it), then the two amount x truth cells the round exists
# to price, then the mechanism dose, then the bracket's top, and `strip` last because the SPEC
# names it most cuttable — its content is a subset of what `complete` and `junk_dose` bracket.
ASSAY_ARMS = "anchor,complete,exact,junk_dose,given_c1,strip"


# [conductor] THE CAPS. The era SEQUENCE is the world and is fixed; what the loop owns is when
# to leave each era. A cap per era (and a total) is what makes a REFUSING arm terminate — an arm
# that rides its caps to the end is a readout, not a hang. Set at 1.25x the scheduled ladder
# (48/40/12/9/7 -> 60/50/15/12/9 = 146), which gives the loop room to hold a quarter longer than
# the schedule did in every era while keeping six arms inside ~2.6 GPU-h at `cs_s0`'s measured
# 10.4 s/cycle. An arm that advances EARLY simply costs less.
CONDUCTOR_CAPS = "60,50,15,12,9"

# Arm ORDER is load-bearing, not alphabetical. The anchor runs first because it is the in-tag
# replay carrier and nothing else is readable without it; each gauge arm runs immediately before
# its yoke, because the yoke's action cycles are the gauge arm's MEASURED ones, passed in at
# runtime (`census`'s mechanic); `outer_ledger` runs last because it is the only arm whose
# content is a re-ask of a donor measurement rather than a new cell, and so the most cuttable.
CONDUCTOR_ARMS = "anchor,outer_yield,yoked_yield,outer_endo,yoked_endo,outer_ledger"

# [maestro] A2's five arms, in the order the round needs them.
#   anchor         the schedule replica — the in-tag carrier AND the cross-tag replay gate
#   outer_yield    A1's hand-written thermostat, replicated in-tag: the COMPARATOR, and the
#                  strongest fidelity gate in the round (nothing upstream of it changed, so it
#                  must replay `cd_s0/outer_yield` bit for bit)
#   learned_yield  the treatment
#   learned_task   the negative control — same class, within-level reward
#   yoked_learned  learned_yield's realised cycles by clock: the non-invasiveness check on the
#                  NEW machinery, and the most cuttable arm if the budget presses
MAESTRO_CAPS = "60,50,15,12,9"
MAESTRO_ARMS = "anchor,outer_yield,learned_yield,learned_task,yoked_learned"

# [crescendo] A3's ladder, caps and arms.
#
# THE CAPS ARE SIZED BY PHASE 0, NOT CHOSEN. Eras 1 and 2 keep A1's caps exactly (60, 50), so
# the L2 and L3 books — and therefore the r**2 wall the L4 commit runs into — are the ones the
# donors measured, and so that this tag's treatment is bit-identical to `ma_s0/outer_yield`
# through both earning eras (the round's free full-scale fidelity gate).
#
# ERA 3 GOES 15 -> 100, and that is the whole design. Three measurements set it:
#   (i)  `miners[4]` is COLD at era 3. The mining gate is `min(maxl, era.level + 1)`, so the
#        committable level-4 miner observes only from era 3 — exactly as L2's observed only
#        from era 1 and L3's only from era 2. The earning eras that produced committable books
#        got 48-60 cycles each; giving L4 fewer would not be a measurement of the frontier, it
#        would be a measurement of the cap.
#   (ii) Phase 0's era-3-restart proxy: over the ~28-36 post-era-2 cycles the donors actually
#        ran, the keys that first reach support in eras 3+ yield only 2-5 buildable L4 entries
#        over the frozen L3 (4-8 over the live one). Non-empty — so the commit can fire — but
#        far thinner than the 12-21-entry books L3 committed. The stream runs at a saturated
#        ~8 observations/cycle in every era, so era length maps linearly onto observations.
#   (iii) The gauge has to be able to QUIET inside the era, and Phase 0 measured WHEN. A1's
#        thermostat, replayed offline on each donor arm's own logged L4 at-support series from
#        era-3 start, first goes quiet at c_in_era3 = 17 / 18 / 18 / 22 / 28 across the five
#        distinct trajectories, and fires AGAIN 9-21 cycles later. So: the commit lands around
#        c_in_era3 ~18 (by which time the cold `miners[4]` has had ~144 observations, enough
#        for a non-empty buildable table, so the empty-table guard is unlikely to cancel it),
#        and the era-advance lands around c_in_era3 ~30-40. A cap of 70 is ~2x the expected
#        exit — headroom for two or three empty-table retries without the cap ever binding,
#        which is what keeps this a measurement of the loop rather than of the cap.
# Eras 4 and 5 keep A1's caps UNCHANGED (12, 9): they are where the value clock is read, and
# the displacement floors this round reuses (census finding 7; `ma_s1`) were measured at those
# era lengths. Changing them would put the readout on a scale no floor covers.
CRESCENDO_CAPS = "60,50,70,12,9"
# The SCHEDULED ladder is set equal to the caps, which is what makes `anchor_long` the lifetime
# ceiling: a schedule arm reads `era["cycles"]`, a loop arm reads `caps[era_i]`, so with the two
# equal the comparator runs the longest life any arm may have.
CRESCENDO_LADDER = "1:25:60,2:12:50,3:6:70,4:3:12,5:1:9"
# Arm ORDER is load-bearing: `anchor_long` first (the carrier and the cross-tag replay gate),
# then the treatment, then `ceiling_m3` — which replays the treatment's MEASURED actions and so
# must run after it — then the two variants, `outer_yield_m3` last as the most cuttable.
CRESCENDO_ARMS = ("anchor_long,outer_yield_m4,ceiling_m3,outer_yield_m4x,outer_yield_m3")
# The draw-B tag: the signature pair rebuilt on a displaced stream. Order load-bearing.
CRESCENDO_TWINS = "outer_yield_m4_j,ceiling_m3_j"

# [tacet] E3'. The LADDER AND CAPS ARE A3'S, UNCHANGED — this round moves the learning rule,
# not the world, so `TACET_CAPS is CRESCENDO_CAPS` and `TACET_LADDER is CRESCENDO_LADDER`, and
# the baseline arm is A3's treatment verbatim. Anything else would make the cross-tag replay
# against `cr3_s0` a comparison of two different runs rather than a fidelity gate.
TACET_CAPS = CRESCENDO_CAPS
TACET_LADDER = CRESCENDO_LADDER
# Arm ORDER is load-bearing twice over: `outer_yield_m4` must run FIRST because every other arm
# replays its MEASURED action cycles (passed in at runtime by the yoke handoff), and the
# remaining order is a budget-risk decision — the two delta directions first (the round's
# question), then the matched-volume random control WITHOUT WHICH THEY DO NOT MEAN ANYTHING,
# then the deliberation-state gate (the queue's other named scalar), and `gate_all` last as the
# most cuttable: it is the only arm nothing volume-controls (its diet differs in size and in
# composition at once), so it is the least interpretable per cycle spent, and both of its knobs
# are the donor's rather than this round's.
TACET_ARMS = ("outer_yield_m4,gate_delta_hi,gate_delta_lo,gate_random,gate_delib,gate_all")

# [intonation] THE LADDER AND CAPS ARE A3'S, UNCHANGED, for `tacet`'s reason one level further
# on: this round moves the EXECUTOR, not the world, so the ladder, the caps, the reads, the
# floors and the outer-loop rule are all inherited and the only new object is the meter.
INTON_CAPS = TACET_CAPS
INTON_LADDER = TACET_LADDER
# Arm ORDER is load-bearing twice, as in `tacet`. `perf_log` must run FIRST because every other
# arm replays its MEASURED action cycles (the yoke handoff). The rest is a budget-risk
# ordering, cuttable from the right:
#   perf_gain     (i) the proposal itself — the bridge's own efferent form.
#   perf_gate     (ii) the E3' question asked with the RIGHT delta.
#   outcome_gate  (ii)'s exactly-volume-matched CONTENT control, and `tacet`'s own arm re-run
#                 on the live-executor substrate — without it `perf_gate` means nothing.
#   perf_raw      (i)'s hygiene control (S13(b)). Last, and the most cuttable: it is the one
#                 comparison in this node that already has a 3-seed answer in another domain
#                 (`plasticity_gain`, `bridge_assembly`), so cutting it costs a cross-domain
#                 replication rather than the node's own question.
INTON_ARMS = "perf_log,perf_gain,perf_gate,outcome_gate,perf_raw"
# [intonation/A] the strongly-metered re-run. `mperf_log` first (the yoke handoff); then the
# weighting arm (the half of the hypothesis that does not reduce volume at all), then the two
# volume-matched gates, then the fixed hygiene control last as the cuttable one.
METERED_ARMS = "mperf_log,mperf_gain,mperf_gate,mout_gate,mperf_rawx"
# Phase A: the baseline arm ALONE through era 1, at the real configuration. `--quick` cannot
# size this round — its substrate is trained for 800 steps and solves 0.16 instances/cycle,
# so every regime looks scarce there. This probe measures the three things the design criterion
# names (solved instances/cycle, whether the mining cap still binds, whether pi's buffer fills)
# AND the one risk the offline sizing cannot settle: whether a 1.5x thinner observation stream
# still lets L2 commit inside era 1.
METERED_PROBE = "mperf_log"
# [caesura] the round's arms. `dsil_sched` runs FIRST: its scheduled era lengths are the caps,
# so it is the lifetime ceiling and the carrier, and nothing is readable without it. Then the
# comparator, then the two treatments, the conjunction last as the most cuttable (it is the
# weaker claim: a veto can only ever delay what the yield gauge already chose).
CAESURA_ARMS = "dsil_sched,dsil_yield,dsil_read,dsil_and"
# [tutti] THE UNIFICATION ARMS, in run order, and the order IS the design.
#   tu_y_exo    (b) yield owns both, no selection. FIRST: it is the in-tag fidelity carrier and
#               the round's load-bearing full-scale cross-tag replay of `ca_s0/dsil_yield`
#               (yield-paced, so `tol_dsil` cannot touch it), and it is the (yield, exo) cell.
#   tu_d_exo    (a) delta-silence owns both, no selection — `caesura` finding 5 RE-INSTANTIATED
#               at the corrected floor, and a BOUNDED cross-tag gate to c42 (the offline-
#               computed first cycle at which 0.0046 and 0.00321442 differ in the verdict).
#   tu_s_exo    (c) THE SPLIT, no selection — the QUEUE's cell, the division of labor clean.
#   tu_s_endo   the composition: both currencies AND the selector in one loop.
#   tu_s_yk     its ONE-BIT CLOCK YOKE (selector off). Runs immediately after its source,
#               because the plan is that source's MEASURED actions, passed in at runtime.
#   tu_d_endo   the selector under a pure execution-currency pacer.
#   tu_y_endo   the selector under a pure outcome-currency pacer — the live-executor twin of
#               the cell `an_s2` is measuring on the exact-DP executor.
#   tu_m_exo    THE MIRROR (yield commits, delta-silence advances). LAST, i.e. the tail cut if
#               the smoke's s/cycle overruns: it is a control on the interpretation of a split
#               win ("any two-gauge mix" vs "this assignment"), not a cell of the 3x2.
TUTTI_ARMS = ("tu_y_exo,tu_d_exo,tu_s_exo,tu_s_endo,tu_s_yk,tu_d_endo,tu_y_endo,tu_m_exo")
# The in-tag pooled null-ABBA re-derivation of delta-silence's dead zone on `ca_s0`'s own logged
# `dsil` series (`phase0_tutti.py` prints it beside the donor's two numbers). NOT a default:
# it is passed explicitly on the command line, and this constant exists so the value the design
# was reviewed with is on the record in the file that uses it.
TUTTI_TOL_DSIL = 0.0046


@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=36000, memory=32768)
def tutti_run(                                             # [tutti]
    tag: str = "tu_s0",
    arms: str = TUTTI_ARMS,
    # [conductor] the loop's own knobs. `tol_endo` has NO default on purpose: a driven read with
    # no measured dead zone is an error (`floor_gate`), and the endo floor is the one that has no
    # offline series, so it is measured on the smoke tag and passed in explicitly.
    era_caps: str = INTON_CAPS, total_cap: int = 0,
    loop_span: int = 1, loop_w: int = 4, loop_burn: int = 4, loop_alpha: float = 0.5,
    tol_ledger: float = MEASURED_FLOORS["ledger"],
    tol_yield_l3: float = MEASURED_FLOORS["yield_by_level"][3],
    tol_yield_l4: float = MEASURED_FLOORS["yield_by_level"][4],
    # [maestro] tol_endo now HAS a default, and must: it is no longer only a dead zone, it is
    # the NORMALISING UNIT of the learned mixture's `endo_excess` component, so it has to be the
    # very number `fit.py` fitted against. A1 measured it at full config on `cd_ef/anchor` and
    # the same statistic is thresholded here, so the brief's rule applies — reuse the measured
    # floor rather than re-measure it on a smoke tag. `fit_gate` asserts the agreement.
    tol_endo: float = MEASURED_FLOORS["endo"],
    n_endo: int = 256, endo_price: int = 0, obs_panel: bool = True,
    endo_read: str = "endo_excess",
    # [conductor] `--quick` collapses everything, but the SMOKE TAG is also where the endo dead
    # zone is measured, and a floor measured on a 128-sequence probe batch and a 5-step plant is
    # not the floor the main run will have. These three put the endo instrument at its
    # PRODUCTION configuration inside an otherwise-quick run, and make the eras long enough for
    # the rule to form windows at all.
    quick_cycles: int = 0, quick_probe_clean: int = 0, quick_gen_steps: int = 0,
    eras: str = INTON_LADDER, seed: int = 0, era_cycles: int = 24, budget: int = 8,
    # [crescendo] the round's one constant: L4 becomes committable. `commit_max_level` stays
    # `None` at the run level (== `max_macro_level`); only `ceiling_m3` sets it, per arm.
    pr_width: int = 16, g_budget: int = 482, max_macro_level: int = 4,
    depth: int = 6, mm: int = 2, n_corrupt: int = 1,
    prov_offset: int = 0, recert_every: int = 5, n_aud: int = 192,
    probe_widths: str = "1,2,4", probe_every: int = 8,
    n_pr: int = 64, n_rt: int = 384, n_score: int = 256, n_grad: int = 4,
    gen_lr: float = 1e-4, gen_steps: int = 20, mine_support: int = 3,
    mine_from: str = "chosen", mine_cap: int = 8,
    value_lr_online: float = 3e-5, sil_c: float = 0.06, sil_cv: float = 0.15,
    sil_win: int = 5, sil_hold: int = 2, sil_min_cycle: int = 6, lp_min_drop: float = 0.10,
    early_offset: int = 1, late_offset: int = 3,
    plant_holdout: int = 0, holdout_seed: int = 11,
    d_fb: float = 1.0, c_mat: float = 0.05, c_prop: float = 0.0,
    prop_lr: float = 3e-4, prop_steps: int = 24, prop_batch: int = 256,
    prop_warmup: int = 4, prop_new_cycles: int = 2, prop_explore: int = 1,
    prop_train_on: str = "solved",
    span_lam: float = 1.0, span_lr: float = 1e-3, span_batch: int = 64,
    span_tau: float = 0.95, span_min_hold: int = 128, span_capture: int = 24,
    collect_task_matched: bool = True, tm_episodes: int = 8192,
    gy_level: int = 4, gate_win: int = 12, gate_theta: int = 0,
    extend_cap: int = 8, extend_tol: float = 0.0, entry_rec: bool = True,
    # [tacet] the gate, at run level. `gate_mode` is deliberately NOT settable here — it is a
    # per-ARM property (`ARMS[...]["cfg"]`), because an arm IS its gate; what the run owns is
    # the selectivity every gate arm shares and whether the mining channel is gated at all.
    gate_frac: float = 0.5, gate_mine: bool = True, gate_log: bool = True,
    # [intonation] the meter, at run level. `perf_meter`, `perf_gain` and `gate_mode` are NOT
    # settable here — they are per-ARM properties, because an arm IS its consumer; what the run
    # owns is the shared calibration every metered arm uses.
    # [caesura] delta-silence's measured dead zone. No default: `QuietPolicy` raises without
    # one, which is the property that keeps a floor measured rather than chosen. -1 is the
    # "not supplied" sentinel (Modal's CLI cannot pass None through a typed float) and is only
    # legal in a run whose arms never DRIVE on `dsil`.
    tol_dsil: float = -1.0,
    # [tutti] the port's MENU size at run level. `question_mode` is deliberately NOT settable
    # here — it is a per-ARM property (`ARMS[...]["cfg"]`), because an arm IS its selector;
    # what the run owns is the menu every arm draws from and pays the same forwards over.
    question_k: int = 2048,
    span_tau_fire: float = 0.50, perf_alpha: float = 0.05,
    perf_g0: float = -1.0, perf_theta: float = -1.0, perf_calib_min: int = 2048,
    perf_tau_w: float = 0.20, perf_w_raw_cap: float = 4.0, perf_w_clip: float = 4.0,
    perf_w_ewma: float = 0.05, perf_row_cap: int = 192,
    ref_tag: str = "", rule_seed: int = 0, train_seed: int = 1, quick: bool = False,
):
    """INTONATION (E′ round 2). Give the practice arc a live, FALLIBLE executor, so that
    performance error — execution vs intention, independent of task success — exists on this
    substrate for the first time; then consume it two ways, each against its own control.

    Five arms on A3's ladder, caps and world, at `max_macro_level=4`, all with `native/span`'s
    corridor head firing BELOW its parity gate. `perf_log` meters delta_perf and consumes
    nothing (the instrument arm, the uniform control, the grade-only baseline and the clock
    source); `perf_gain`/`perf_raw` consume it as a per-sample plasticity gain against a raw-e
    gain; `perf_gate`/`outcome_gate` consume it as a selection gate against the outcome-delta
    gate `tacet` ran. See the module docstring for the signal's exact form, why the intention
    reference is free and exact, what is and is not priced, and the 2x2.

    DONOR DOCSTRING FOLLOWS.

    TACET (E3′). Does gating what the learner LEARNS FROM — on delta = grade - v(s) and on
    the deliberation state, both alive at decision time and both free on the wire — change what
    the table contains, what trust forms, and whether the range extends?

    Six arms on A3's ladder, caps and world, all routing-only, all at `max_macro_level=4`. The
    baseline is A3's treatment verbatim (grade-only selection, the current rule); every other
    arm is a clock yoke of it carrying exactly one knob. See the module docstring for the gate's
    definition, why the value buffer is ungated, why no gradient-budget matching is needed, and
    what "learn from everything" is taken to mean on a channel that is only defined over solved
    derivations.

    DONOR DOCSTRING FOLLOWS.

    CRESCENDO (A3). Does the earnable range extend with the turn of the crank — does the
    value clock hold past the range the previous turn certified?

    Every run in this arc capped commitment at `max_macro_level=3` and treated L4 as
    unearnable. Here L4 is opened to the crank and given the earning time each previous rung
    got. Four arms at A1's exact configuration, all routing-only, all on the anchor's stream,
    all at `max_macro_level=4`:

      anchor_long      certificate-else-boundary; schedule == caps  the carrier, the cross-tag
                                                                    replay gate, and the
                                                                    LIFETIME CEILING
      outer_yield_m4   A1's thermostat; L4 committable              the treatment
      ceiling_m3       clock replay of the treatment's actions,
                       `commit_max_level=3`                         THE CEILING CONTROL — the
                                                                    pair the signature is read
                                                                    on, lifetime-identical by
                                                                    construction, differing in
                                                                    one bit
      outer_yield_m4x  the treatment + post-commit extension        the r**2-wall bypass

    The reads, the floors, the ladder's SEQUENCE and the rule are A1's, unchanged. What moved
    is `max_macro_level`, era 3's cap (15 -> 100, sized by Phase 0), and the anchor's schedule
    (set to the caps, which is what closes A1's and A2's lifetime confound).
    """
    import torch
    cfg = _d6_cfg(depth=depth, m=mm, n_corrupt=n_corrupt, prov_offset=prov_offset,
                  # [maestro] the fitted policies travel in the config, so they land in the
                  # run's own `setup.json` and in every arm's `results.json`.
                  fitted=copy.deepcopy(FITTED),
                  era_caps=tuple(int(x) for x in era_caps.split(",") if x.strip()),
                  total_cap=total_cap, loop_span=loop_span, loop_W=loop_w,
                  loop_burn=loop_burn, loop_alpha=loop_alpha,
                  tol_ledger=tol_ledger, tol_yield_l3=tol_yield_l3, tol_yield_l4=tol_yield_l4,
                  tol_endo=(tol_endo or None), n_endo=n_endo, endo_price=endo_price,
                  obs_panel=obs_panel, endo_read_key=endo_read,
                  recert_every=recert_every, n_aud=n_aud,
                  probe_widths=tuple(int(x) for x in probe_widths.split(",")),
                  probe_every=probe_every,
                  span_lam=span_lam, span_lr=span_lr, span_batch=span_batch,
                  span_tau=span_tau, span_min_hold=span_min_hold, span_capture=span_capture,
                  c_prop=c_prop, prop_lr=prop_lr, prop_steps=prop_steps, prop_batch=prop_batch,
                  prop_warmup=prop_warmup, prop_new_cycles=prop_new_cycles,
                  prop_explore=prop_explore, prop_train_on=prop_train_on,
                  seed=seed, era_cycles=era_cycles, budget=budget, pr_width=pr_width,
                  g_budget=g_budget, max_macro_level=max_macro_level, n_pr=n_pr, n_rt=n_rt,
                  n_score=n_score, n_grad=n_grad, gen_lr=gen_lr, gen_steps=gen_steps,
                  mine_support=mine_support, mine_from=mine_from, mine_cap=mine_cap,
                  value_lr_online=value_lr_online, sil_c=sil_c, sil_cv=sil_cv, sil_W=sil_win,
                  sil_hold=sil_hold, sil_min_cycle=sil_min_cycle, lp_min_drop=lp_min_drop,
                  early_offset=early_offset, late_offset=late_offset,
                  plant_holdout=plant_holdout, holdout_seed=holdout_seed,
                  d_fb=d_fb, c_mat=c_mat, collect_task_matched=collect_task_matched,
                  tm_episodes=tm_episodes, rule_seed=rule_seed, train_seed=train_seed,
                  gy_level=gy_level, gate_W=gate_win, gate_theta=gate_theta,
                  extend_cap=extend_cap, extend_tol=extend_tol, entry_rec=entry_rec,
                  # [tacet] the gate's run-level knobs. `gate_mode` stays absent here, so the
                  # run config is the donor's unless an ARM names a mode.
                  gate_frac=gate_frac, gate_mine=gate_mine, gate_log=gate_log,
                  # [intonation] a negative sentinel means "calibrate in-run", because Modal's
                  # CLI cannot pass None through a typed float.
                  tol_dsil=(None if tol_dsil < 0 else tol_dsil),     # [caesura]
                  question_k=question_k,                             # [tutti]
                  span_tau_fire=span_tau_fire, perf_alpha=perf_alpha,
                  perf_g0=(None if perf_g0 < 0 else perf_g0),
                  perf_theta=(None if perf_theta < 0 else perf_theta),
                  perf_calib_min=perf_calib_min, perf_tau_w=perf_tau_w,
                  perf_w_raw_cap=perf_w_raw_cap, perf_w_clip=perf_w_clip,
                  perf_w_ewma=perf_w_ewma, perf_row_cap=perf_row_cap)
    if quick:
        cfg.update(controller_steps=800, generator_steps=800, value_steps=800,
                   reader_steps=600, value_episodes=6_000, n_train_episodes=20_000,
                   era_cycles=3, n_pr=24, n_rt=96, n_score=96, n_aud=48, n_probe_clean=128,
                   checkpoint_every=2, sil_min_cycle=2, gen_steps=5, tm_episodes=512,
                   prop_warmup=1, prop_steps=12, prop_buf_cap=20_000, probe_every=2,
                   recert_every=1, span_min_hold=16, span_tau=0.0)
        if quick_cycles:
            cfg["era_cycles"] = int(quick_cycles)
        if quick_probe_clean:
            cfg["n_probe_clean"] = int(quick_probe_clean)
        if quick_gen_steps:
            cfg["gen_steps"] = int(quick_gen_steps)
    ers = parse_eras(eras)
    if quick:
        for e_ in ers:
            e_["cycles"] = cfg["era_cycles"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg["train_seed"]); np.random.seed(cfg["train_seed"])
    torch.set_float32_matmul_precision("high")
    started = time.time()
    total_cycles = sum(int(e_.get("cycles") or cfg["era_cycles"]) for e_ in ers)
    # [conductor] in `--quick` the caps have to shrink with the ladder or a loop arm would run
    # the full-scale caps against a 3-cycle era and every arm would ride them.
    if quick:
        cfg["era_caps"] = tuple(max(cfg["era_cycles"] + 4,
                                    cfg["loop_W"] * cfg["loop_span"] + 2) for _ in ers)
    cap_total = int(cfg["total_cap"] or sum(cfg["era_caps"] or ()))
    print(f"conductor tag={tag} depth={cfg['depth']} m={cfg['m']} "
          f"seqlen={cfg['s'] ** cfg['depth']} arms={arms}\n"
          f"  ladder={[(e_['name'], e_.get('cycles') or cfg['era_cycles']) for e_ in ers]} "
          f"= {total_cycles} cycles/arm scheduled; loop caps {list(cfg['era_caps'])} "
          f"= {cap_total} max\n"
          f"  loop: span={cfg['loop_span']} W={cfg['loop_W']} burn={cfg['loop_burn']} "
          f"alpha={cfg['loop_alpha']}  floors ledger={cfg['tol_ledger']} "
          f"yieldL3={cfg['tol_yield_l3']} yieldL4={cfg['tol_yield_l4']} "
          f"endo={cfg['tol_endo']}  n_endo={cfg['n_endo']} endo_price={cfg['endo_price']}g\n"
          f"  entry_rec={entry_rec}   device={device}\n"
          # [tacet] the round's own line: the gate's shared selectivity and, per arm, its mode.
          f"  GATE: frac={cfg['gate_frac']} mine={cfg['gate_mine']} log={cfg['gate_log']}  "
          f"modes=" + str({a: (ARMS[a].get("cfg") or {}).get("gate_mode")
                           for _, a, _ in parse_arms(arms)}), flush=True)

    # ---- selfchecks BEFORE the paid setup, INSIDE AN RNG SANDBOX --------------------- #
    # `entry_recorder_check` builds models and samples pools, and reseeds to 0 to do it
    # deterministically — so run bare it leaves the global torch/numpy streams somewhere else
    # than `torch.manual_seed(train_seed)` put them, and `build_shared` then trains a DIFFERENT
    # substrate from the one `cs_s0` trained. That is not a fidelity nicety: the anchor is this
    # round's full-scale G-F carrier against `cs_s0/spiral_route`, and a moved stream voids it.
    # Caught in flight on the first launch (stale buffer 0.1028 against `cs_s0`'s 0.1269, same
    # config and seed). The states are snapshotted and restored around the checks, and the
    # restore is ASSERTED rather than assumed.
    _rng0 = _rng_snapshot()
    _ref_draw = torch.randn(4).tolist()
    _rng_restore(_rng0)
    erc = entry_recorder_check(v=cfg["v"], s=cfg["s"], depth=cfg["depth"], m=cfg["m"],
                               max_level=cfg["max_macro_level"])
    print(f"[erc] entry-recorder check PASS: {json.dumps(erc, cls=NumpyEncoder)}", flush=True)
    gy = gy_soundness(s=cfg["s"], depth=cfg["depth"], level=cfg["gy_level"], eras=ers)
    print(f"[gy] soundness PASS: {json.dumps(gy, cls=NumpyEncoder)}", flush=True)
    # [conductor] the round's own pre-setup gates, in the same RNG sandbox. All three are pure
    # and cost nothing; running them here means a bad floor or a leaked label fails in seconds
    # rather than after a paid setup.
    pg = PO.policy_gate(verbose=False)
    assert pg["ALL"], f"policy gate FAILED: {pg}"
    print(f"[pol] policy gate PASS: {sum(1 for k in pg if k != 'ALL')} checks "
          f"({', '.join(k for k in pg if k != 'ALL')})", flush=True)
    eg = endo_gate(s=cfg["s"], depth=cfg["depth"], eras=ers)
    print(f"[endo] label-free-by-construction PASS: {json.dumps(eg, cls=NumpyEncoder)}",
          flush=True)
    _rng_restore(_rng0)
    _got_draw = torch.randn(4).tolist()
    _rng_restore(_rng0)
    assert _ref_draw == _got_draw, (
        f"the pre-setup selfchecks are not RNG-neutral: {_ref_draw} != {_got_draw} — "
        f"`build_shared` would train a different substrate and the cross-tag G-F would be void")
    print(f"[rng] selfchecks are RNG-neutral (next draw unchanged: "
          f"{[round(x, 6) for x in _got_draw]})", flush=True)

    _install_identity_miner(cfg["mine_support"])
    _install_entry_recorder()
    t0 = time.time()
    shared = _spiral_shared(cfg, device, ers)
    t_setup = time.time() - t0

    # ---- the realistic junk pool, from the arc's own runs on the volume --------------- #
    junk, jprov = build_junk_pool(shared["truth"], cfg["max_macro_level"])
    shared["junk_pool"] = junk
    print(f"[junk] pool {jprov['split']} provenance={jprov['provenance']}", flush=True)
    for ell, ks in junk.items():
        print(f"[junk]   L{ell}: {len(ks)} mined-but-false tuples available", flush=True)

    for ell in range(2, cfg["max_macro_level"] + 1):
        print(f"  true table L{ell}: {shared['truth'][ell]['child'].shape[0]} entries")
    print(f"[setup] {t_setup:.0f}s  read_acc={shared['read_acc']:.4f}  "
          f"stale buffer: random-block {shared.get('stale_random_blocks'):.4f} -> "
          f"task-matched {shared.get('stale_task_matched')}", flush=True)

    # ---- [conductor] N-2: WHAT THE ENDO READ COSTS, on this GPU ----------------------- #
    # Measured after the setup, on the setup's own generator, INSIDE the RNG sandbox — the bench
    # runs forward passes only, but the sandbox is asserted anyway so a future edit that adds a
    # sampling step cannot silently move the substrate.
    _rng1 = _rng_snapshot()
    bench = endo_bench(shared, cfg, device, cfg["n_endo"])
    _rng_restore(_rng1)
    print(f"[endo] bench: one read over {bench['n']} sequences = {bench['t_endo_s']*1e3:.1f} ms "
          f"= {bench['price_g']} grounding-equivalents (charging "
          f"{cfg['endo_price']}g); a grounding is {bench['t_per_grounding_s']*1e6:.1f} us",
          flush=True)

    # ---- [conductor] THE FLOORS THAT WILL GOVERN THIS RUN ---------------------------- #
    # Enforced only for the reads that are actually DRIVEN in this tag, so a fidelity or
    # floor-measuring tag can run without an endo floor it is being run to measure.
    parsed = parse_arms(arms)
    driven_reads = {(ARMS[b].get("loop") or {}).get("read") for _, b, _ in parsed}
    # [maestro] a learned arm drives ALL THREE gauges at once, so it needs every floor present:
    # each one is the normalising unit of its own component of the mixture.
    learned_arms = [lb for _, lb, _ in parsed
                    if (ARMS[lb].get("loop") or {}).get("kind") == "learned"]
    if learned_arms:
        driven_reads |= {"ledger", "yield", "endo"}
    fg = None
    if "endo" in driven_reads or "yield" in driven_reads or "ledger" in driven_reads:
        if "endo" not in driven_reads:
            cfg = {**cfg, "tol_endo": cfg["tol_endo"] or 1.0}   # not read; keep the gate honest
        fg = floor_gate(cfg)
        print(f"[floor] {json.dumps(fg, cls=NumpyEncoder)}", flush=True)
    # [maestro] the fitted policies' own gate, beside the floors' — same pattern, same place.
    fitg = None
    if learned_arms:
        fitg = fit_gate(cfg)
        assert abs(cfg["tol_endo"] - PO_FIT_UNITS["endo_excess"]) < 1e-12, (
            f"tol_endo {cfg['tol_endo']} is not the unit the mixture was fitted in "
            f"({PO_FIT_UNITS['endo_excess']}) — the learned weights would mean something else")
        assert abs(cfg["tol_ledger"] - PO_FIT_UNITS["ledger"]) < 1e-12
        assert abs(cfg["tol_yield_l3"] - PO_FIT_UNITS["yield_by_level"][3]) < 1e-12
        assert abs(cfg["tol_yield_l4"] - PO_FIT_UNITS["yield_by_level"][4]) < 1e-12
        print(f"[fit] {json.dumps(fitg, cls=NumpyEncoder)}", flush=True)

    t0 = time.time()
    refs = measure_refs(shared, cfg, ers, device)
    t_refs = time.time() - t0
    plant = plant_probe(shared["generator0"], shared, cfg, device)
    print(f"[refs] {t_refs:.0f}s {json.dumps(refs, cls=NumpyEncoder)}", flush=True)

    # [assay] SUBSTRATE IDENTITY against a reference tag. A tag whose arms will be compared
    # against another tag's arms is only comparable if the two trained the SAME substrate.
    # Asserted rather than assumed: this is exactly the failure that voided the first `as_s0`
    # attempt, caught there only because the stale-buffer number happened to be printed.
    ref_check = None
    if ref_tag:
        rp = f"{DATA_DIR}/{REMOTE}/{ref_tag}/setup.json"
        assert os.path.isfile(rp), f"ref_tag {ref_tag} has no setup.json on the volume"
        rs = json.load(open(rp))
        same_refs = all(np.allclose(np.asarray(refs[k], float),
                                    np.asarray(rs["refs"][k], float))
                        for k in refs if k != "macro_true")
        ref_check = {"ref_tag": ref_tag, "refs_identical": bool(same_refs),
                     "stale_random_blocks": [shared.get("stale_random_blocks"),
                                             rs.get("stale_random_blocks")],
                     "stale_task_matched": [shared.get("stale_task_matched"),
                                            rs.get("stale_task_matched")],
                     "read_acc": [float(shared["read_acc"]), rs.get("read_acc")]}
        assert same_refs, f"refs differ from {ref_tag}: not the same substrate"
        assert (shared.get("stale_random_blocks") == rs.get("stale_random_blocks")
                and shared.get("stale_task_matched") == rs.get("stale_task_matched")), \
            f"stale value buffer differs from {ref_tag}: not the same substrate — {ref_check}"
        print(f"[ref] substrate identical to {ref_tag}: {json.dumps(ref_check)}", flush=True)

    outdir = f"{DATA_DIR}/{REMOTE}/{tag}"
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "setup.json"), "w") as fh:
        json.dump({"config": cfg, "eras": ers, "refs": refs, "plant": plant,
                   "ref_check": ref_check,
                   "gy_soundness": gy, "entry_recorder_check": erc,
                   "junk_pool": {str(k): [list(x) for x in v] for k, v in junk.items()},
                   "junk_pool_provenance": jprov,
                   "true_tables": {str(ell): {
                       "n": int(shared["truth"][ell]["child"].shape[0]),
                       "flat": shared["truth"][ell]["flat"].tolist()}
                       for ell in range(2, cfg["max_macro_level"] + 1)},
                   "read_acc": float(shared["read_acc"]),
                   "stale_random_blocks": shared.get("stale_random_blocks"),
                   "stale_task_matched": shared.get("stale_task_matched"),
                   "t_setup_s": t_setup, "t_refs_s": t_refs,
                   # [conductor] the round's own record: the gates, the bench, and — most
                   # importantly — the DEAD ZONES THAT GOVERNED THIS RUN with their provenance,
                   # written into the run rather than left in a script.
                   "policy_gate": pg, "endo_gate": eg, "endo_bench": bench, "floor_gate": fg,
                   # [maestro] the fitted policies that governed this run, with their gate.
                   "fit_gate": fitg, "fitted": cfg["fitted"],
                   "floors": {"ledger": cfg["tol_ledger"], "yield_L3": cfg["tol_yield_l3"],
                              "yield_L4": cfg["tol_yield_l4"], "endo": cfg["tol_endo"],
                              "provenance": MEASURED_FLOORS["provenance"]},
                   "era_caps": list(cfg["era_caps"]), "cap_total": cap_total,
                   "total_cycles_per_arm": total_cycles}, fh, indent=2, cls=NumpyEncoder)
    volume.commit()

    summary = {"arms": {}, "order": [], "cycle_seconds": {}}
    # [conductor] the yoke handoff (`census_run`'s mechanic, widened to both actions): a gauge
    # arm's REALISED action cycles are collected as it finishes and passed into its yoke, which
    # replays them by clock. Not hard-coded, and asserted present — a yoke with an empty plan
    # would silently be an arm that never acts.
    measured_plans = {}
    for label, base, ov in parse_arms(arms):
        lspec = ARMS[base].get("loop") or {}
        if lspec.get("kind") == "yoke":
            src = lspec["of"]
            assert src in measured_plans, (
                f"{label} yokes onto {src}, which has not run yet — arm ORDER is load-bearing "
                f"(see CONDUCTOR_ARMS)")
            ov = {**ov, "yoke_plan": json.dumps(measured_plans[src])}
            print(f"[yoke]  {label} replays {src}'s {len(measured_plans[src])} actions "
                  f"by clock: {[(p['kind'], p['cycle']) for p in measured_plans[src]]}",
                  flush=True)
        print(f"\n===== arm {label} (base {base}, overrides "
              f"{ {k: (v if k != 'yoke_plan' else '<plan>') for k, v in ov.items()} }) =====",
              flush=True)
        t0 = time.time()
        r = run_arm(label, base, ov, shared, cfg, ers, refs, outdir, device)
        n = len(r["log"]["cycle"])
        measured_plans[label] = [{"cycle": a["cycle"], "kind": a["kind"], "level": a["level"]}
                                 for a in r.get("loop_actions", [])
                                 if not a.get("cancelled")]
        summary["order"].append(label)
        summary["cycle_seconds"][label] = (time.time() - t0) / max(n, 1)
        summary["arms"][label] = {
            "n_cycles": n, "shadow_cert": r["shadow_cert"],
            "gy_final": r["gy_final"], "gauge_hist": r["gauge_hist"],
            "surgery": ARMS[base].get("cfg", {}).get("surgery"),
            "stream_burn": r.get("stream_burn"),
            # [conductor] the round's headline per-arm readouts, in the summary so a partial run
            # is still readable without pulling every `results.json`.
            "loop": r.get("loop"), "loop_actions": r.get("loop_actions"),
            "ledger": r.get("ledger"), "obs_hist": r.get("obs_hist"),
            "obs_gy_agree": r.get("obs_gy_agree"),
            "advances": [e for e in r["events"] if e["kind"] == "advance"],
            "commits": [e for e in r["events"] if e["kind"] == "commit"],
            "surgeries": [{k: q for k, q in e.items() if k != "flat_after"}
                          for e in r["events"] if e["kind"] == "surgery"],
            "n_recerts": sum(1 for e in r["events"] if e["kind"] == "recert"),
            "e_by_era": {}}
        for j in range(len(ers)):
            idx = [i for i, q in enumerate(r["log"]["era"]) if q == j + 1]
            if idx:
                summary["arms"][label]["e_by_era"][str(j + 1)] = {
                    "first": r["log"]["e"][idx[0]], "last": r["log"]["e"][idx[-1]],
                    "min": min(r["log"]["e"][i] for i in idx),
                    "mean": float(np.mean([r["log"]["e"][i] for i in idx]))}
        cs = r["shadow_cert"]
        ent = r["log"]["entry"][-1] if r["log"].get("entry") else {}
        beam = (ent.get("hist") or {}).get("beam", {})
        lp = r.get("loop") or {}
        print(f"[arm] {label}: {n} cycles at {summary['cycle_seconds'][label]:.1f} s/cycle; "
              f"cert L2 c{cs['2'].get('fired')} L3 c{cs.get('3', {}).get('fired')}; "
              f"loop={lp.get('policy')}/{lp.get('read')} "
              f"commits@{lp.get('commit_cycles')} L{lp.get('commit_levels')} "
              f"advances@{lp.get('advance_cycles')} "
              f"chosen={lp.get('n_chosen')} capped={lp.get('n_capped')} "
              f"spend={(r.get('ledger') or {}).get('spend_g')}g; "
              f"last-cycle beam entry selections "
              f"{ {k: int(sum(v)) for k, v in beam.items()} }", flush=True)
        with open(os.path.join(outdir, "summary.json"), "w") as fh:
            json.dump(summary, fh, indent=2, cls=NumpyEncoder)
        volume.commit()

    summary["elapsed_s"] = time.time() - started
    with open(os.path.join(outdir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, cls=NumpyEncoder)
    with open(os.path.join(outdir, "done.txt"), "w") as fh:
        fh.write(f"elapsed {time.time() - started:.0f}s\n")
    volume.commit()
    print(f"\nDONE in {time.time() - started:.0f}s "
          f"({(time.time() - started) / 3600:.2f} GPU-h) -> {outdir}", flush=True)
    return {"refs": refs, "elapsed": time.time() - started, "summary": summary}


@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=14400, memory=32768)
def cal_ladder(tag: str = "call_s0", eras: str = "1:6,2:3,3:1", seed: int = 0,
               budget: int = 4, n_ref: int = 512, g_budgets: str = "33,58,86,100,108",
               widths: str = "1,2,3,4,8,16", max_macro_level: int = 3, quick: bool = False):
    """CALIBRATION 1 — the depth ladder, before any loop is built.

    Answers, per era and on identical held-out instances: how hard is level-k damage for the
    BASE move set at each beam width, what does handing over the TRUE level vocabulary buy,
    what does one true macro action buy, and where the exact-DP floor is. This is what picks
    the declared per-solve grounding budget `g_budget`: it must be a regime where depth is
    unaffordable to base moves and affordable with the vocabulary, otherwise the round's
    headline is a property of a budget chosen by hand."""
    import torch
    cfg = _cfg(seed=seed, budget=budget, n_rt=n_ref, max_macro_level=max_macro_level)
    if quick:
        cfg.update(controller_steps=800, generator_steps=800, value_steps=800,
                   value_episodes=6_000, n_train_episodes=20_000, n_rt=128)
    ers = parse_eras(eras)
    ws = [int(x) for x in widths.split(",")]
    gs = [int(x) for x in g_budgets.split(",")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg["train_seed"]); np.random.seed(cfg["train_seed"])
    torch.set_float32_matmul_precision("high")
    started = time.time()

    shared = build_shared(cfg, device)
    rules, rules_t, canon = shared["rules"], shared["rules_t"], shared["canon"]
    v, s, depth, m = cfg["v"], cfg["s"], cfg["depth"], cfg["m"]
    base_ms = shared["base_ms"]
    l2_ms = build_ms(base_ms, {2: shared["truth"][2]}, s, depth, device)
    true_ms = build_ms(base_ms, {ell: shared["truth"][ell]
                                 for ell in range(2, max_macro_level + 1)}, s, depth, device)
    sets = (("base", base_ms), ("l2", l2_ms), ("true", true_ms))
    out = {"config": cfg, "eras": ers, "cells": {},
           "n_moves": {t: len(x) for t, x in sets},
           "ground_table": {f"{t}_w{w}": beam_ground(len(mset), budget, w)
                            for t, mset in sets for w in ws}}
    for i, era in enumerate(ers):
        r_np, x_np = context_instances(rules, era_ctx(era), cfg["n_rt"], s, depth, v, m,
                                       seed=cfg["seed"] + 5000 + 17 * i)
        x0 = torch.from_numpy(x_np)
        cell = {"d0": float(nearest_derivation_cost(rules, x_np, r_np, s).mean()),
                "on_grammar": on_grammar_rate(x_np, shared["inverse_maps"][-1], v, s)}
        for tag_, mset in sets:
            cell[tag_] = {}
            for w in ws:
                b = beam_moves(shared["controller"], shared["generator0"], shared["value0"], x0,
                               torch.from_numpy(r_np), mset, rules_t, canon, depth, v, m, s,
                               budget=budget, beam_width=w, device=device)
                sc, _ = grade(b["x"].cpu().numpy(), r_np, rules, s)
                cell[tag_][str(w)] = {"e": 1.0 - float(sc.mean()),
                                      "g": b["counts"]["ground"] / x0.shape[0]}
        cell["at_budget"] = {}
        for g in gs:
            row = {}
            for tag_, mset in sets:
                w = fit_width(len(mset), budget, g)
                row[tag_] = {"w": w, "g": beam_ground(len(mset), budget, w),
                             "e": cell[tag_].get(str(w), {}).get("e")}
            cell["at_budget"][str(g)] = row
        # one macro ACTION, at full and at partial vocabulary coverage. The coverage curve is
        # what decides whether the unit-LP certificate has a resolvable descent to certify:
        # if audition is flat in coverage, the round has no signal and that is worth knowing
        # before the main run rather than after it.
        cell["macro_true"], cell["coverage"] = {}, {}
        crng = np.random.default_rng(cfg["seed"] + 4242)
        for ell in range(2, max_macro_level + 1):
            span = s ** (ell - 1)
            node = (era["node"] * s ** (era["level"] - 1)) // span
            full = shared["truth"][ell]
            mv = MC.to_device(MC.make_macro(ell, node, s, full), device)
            cell["macro_true"][str(ell)] = audition_macro(
                shared["generator0"], x0.to(device), r_np, mv, rules_t, canon, depth,
                v, m, s, rules)["e"]
            row = {}
            n_all = full["child"].shape[0]
            ks = sorted({1, 2, 4, 8, max(1, n_all // 4), max(1, n_all // 2),
                         max(1, 3 * n_all // 4), n_all})
            for k in ks:
                es = []
                for _ in range(1 if k == n_all else 5):
                    keep = np.sort(crng.permutation(n_all)[:k])
                    sub = MC.make_table(ell, full["child"][keep], full["lower"], s)
                    mvs = MC.to_device(MC.make_macro(ell, node, s, sub), device)
                    es.append(audition_macro(shared["generator0"], x0.to(device), r_np, mvs,
                                             rules_t, canon, depth, v, m, s, rules)["e"])
                row[str(k)] = {"mean": float(np.mean(es)), "sd": float(np.std(es)),
                               "min": float(np.min(es)), "max": float(np.max(es)),
                               "n_entries": k, "frac": k / n_all}
            cell["coverage"][str(ell)] = row
        xo, _ = oracle_rollout(shared["generator0"], x0.to(device), r_np, base_ms, rules,
                               rules_t, canon, depth, v, m, s, budget)
        sc, _ = grade(xo.cpu().numpy(), r_np, rules, s)
        cell["dp_floor_base"] = 1.0 - float(sc.mean())
        out["cells"][era["name"]] = cell
        print(f"\n=== era {era['name']} (d0={cell['d0']:.2f}, on_gram={cell['on_grammar']:.3f}) ===")
        for tag_, _ in sets:
            print(f"  {tag_:5s}: " + " | ".join(
                f"w{w}: e={cell[tag_][str(w)]['e']:.3f} ({cell[tag_][str(w)]['g']:.0f}g)"
                for w in ws))
        print(f"  one true macro action: " + " ".join(
            f"L{ell}={cell['macro_true'][str(ell)]:.3f}" for ell in cell["macro_true"]))
        for ell, row in cell["coverage"].items():
            print(f"  coverage L{ell} (entries: e[min,max]): " + " ".join(
                f"{k}:{c['mean']:.3f}[{c['min']:.2f},{c['max']:.2f}]"
                for k, c in sorted(row.items(), key=lambda kv: int(kv[0]))))
        print(f"  DP floor (base moves) = {cell['dp_floor_base']:.3f}")
        for g in gs:
            r = cell["at_budget"][str(g)]
            print(f"  @G={g:4d}: " + " | ".join(
                f"{t} w{r[t]['w']} ({r[t]['g']}g) e={r[t]['e']}" for t, _ in sets))

    outdir = f"{DATA_DIR}/{REMOTE}/{tag}"
    os.makedirs(outdir, exist_ok=True)
    out["elapsed"] = time.time() - started
    with open(os.path.join(outdir, "cal_ladder.json"), "w") as fh:
        json.dump(out, fh, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"\nDONE in {out['elapsed']:.0f}s -> {outdir}/cal_ladder.json")
    return out


@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=7200, memory=32768)
def cal_parse(tag: str = "calp2_s0", eras: str = "1:6,2:3,3:1", seed: int = 0,
              budget: int = 4, n_ref: int = 512, max_macro_level: int = 3,
              quick: bool = False):
    """CALIBRATION 3 — the perception bound on an earned vocabulary.

    `calr_s0` measured that the mined tables have LOW PRECISION (level-3: 10 true entries out
    of 34 mined, and `never_base` mined 100 distinct 4-tuples where the grammar has 56). The
    suspected cause is not the mining rule but the READ: the generator is trained by masked
    infilling with `mask_min=1` and had never seen a fully unmasked configuration, so parsing
    one is out of distribution.

    This isolates it. On the SAME oracle-solved configurations, the level-1 parse is taken
    three ways -- unmasked (what `calr_s0` did), with one block masked ELSEWHERE (in
    distribution, every span block still visible), and by the exact bottom inverse map (the
    privileged ceiling) -- and the resulting vocabulary is graded against the DGP's own each
    way. If the masked read recovers precision, the defect is the instrument; if it does not,
    the vocabulary is bounded by the plant and that is a substrate fact about the round."""
    import torch
    cfg = _cfg(seed=seed, budget=budget, n_rt=n_ref, max_macro_level=max_macro_level)
    if quick:
        cfg.update(controller_steps=800, generator_steps=800, value_steps=800,
                   value_episodes=6_000, n_train_episodes=20_000, n_rt=128)
    ers = parse_eras(eras)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg["train_seed"]); np.random.seed(cfg["train_seed"])
    torch.set_float32_matmul_precision("high")
    started = time.time()

    shared = build_shared(cfg, device)
    rules, rules_t, canon = shared["rules"], shared["rules_t"], shared["canon"]
    v, s, depth, m = cfg["v"], cfg["s"], cfg["depth"], cfg["m"]
    gen, n_blocks = shared["generator0"], shared["n_blocks"]
    out = {"config": cfg, "eras": ers, "parse": {}, "cells": {}}

    # ---- block-level parse accuracy on CLEAN held-out configs, three ways ---------------
    clean = torch.from_numpy(_sample_pool(rules, cfg["n_rt"], s, cfg["seed"] + 31337)[1]).to(device)
    powers = v ** torch.arange(s, device=device)
    truth_f = shared["bottom_map"][(clean.view(clean.shape[0], n_blocks, s) * powers).sum(-1)]
    ok = truth_f >= 0
    for name, mb in (("unmasked", None), ("mask_elsewhere", 0)):
        pf = MC.parse_features(gen, clean, s=s, mask_block=mb)
        sel = ok.clone()
        if mb is not None:
            sel[:, mb] = False                       # the masked block is inferred, not read
        out["parse"][name] = float((pf[sel] == truth_f[sel]).float().mean())
    out["parse"]["ambiguous_frac"] = float((~ok).float().mean())
    print(f"[parse] clean-config block accuracy: {json.dumps(out['parse'])}", flush=True)

    # ---- the vocabulary each read yields, on oracle-solved repairs ----------------------
    for i, era in enumerate(ers):
        r_np, x_np = context_instances(rules, era_ctx(era), cfg["n_rt"], s, depth, v, m,
                                       seed=cfg["seed"] + 5000 + 17 * i)
        x0 = torch.from_numpy(x_np).to(device)
        xo, _ = oracle_rollout(gen, x0, r_np, shared["base_ms"], rules, rules_t, canon,
                               depth, v, m, s, budget)
        sc, _ = grade(xo.cpu().numpy(), r_np, rules, s)
        solved = xo[torch.from_numpy(sc > 0.5).to(device)]
        cell = {"n_solved": int(solved.shape[0]), "solve_rate": float(sc.mean()), "reads": {}}
        top_span = s ** (max_macro_level - 1)
        top0 = ((era["node"] * s ** (era["level"] - 1)) // top_span) * top_span
        mb = 0 if top0 > 0 else top0 + top_span
        reads = {"unmasked": MC.parse_features(gen, solved, s=s).cpu().numpy(),
                 "mask_elsewhere": MC.parse_features(gen, solved, s=s,
                                                     mask_block=mb).cpu().numpy(),
                 "exact": MC.exact_features(solved.cpu().numpy(),
                                            shared["inverse_maps"][-1], v, s)}
        for name, pf in reads.items():
            row = {}
            chain, obs = {1: MC.base_table(v)}, {}
            for ell in range(2, max_macro_level + 1):
                sp = s ** (ell - 1)
                nd = (era["node"] * s ** (era["level"] - 1)) // sp
                mn = MC.Miner(ell, s)
                seg = pf[:, nd * sp:(nd + 1) * sp]
                mn.observe(seg[(seg >= 0).all(1)])
                chain[ell] = mn.build(chain[ell - 1], cfg["mine_support"])
                obs[ell] = len(mn.counts)
            for ell in range(2, max_macro_level + 1):
                tbl = chain[ell]
                row[str(ell)] = {"n_entries": int(tbl["child"].shape[0]),
                                 "n_distinct_obs": obs[ell],
                                 **MC.grade_table(tbl, shared["truth"][ell])}
            cell["reads"][name] = row
        out["cells"][era["name"]] = cell
        print(f"\n=== era {era['name']} (oracle solve rate {cell['solve_rate']:.3f}, "
              f"{cell['n_solved']} solved) ===")
        for name, row in cell["reads"].items():
            print(f"  {name:15s}: " + " | ".join(
                f"L{ell}: {c['n_entries']:3d} entries, prec="
                f"{(c['precision'] if c['precision'] is not None else float('nan')):.3f}, "
                f"recall={c['recall']:.3f} (true {c['n_true']})" for ell, c in row.items()))

    outdir = f"{DATA_DIR}/{REMOTE}/{tag}"
    os.makedirs(outdir, exist_ok=True)
    out["elapsed"] = time.time() - started
    with open(os.path.join(outdir, "cal_parse.json"), "w") as fh:
        json.dump(out, fh, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"\nDONE in {out['elapsed']:.0f}s -> {outdir}/cal_parse.json")
    return out


@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=14400, memory=32768)
def cal_stale(tag: str = "cals_s0", eras: str = "1:6,2:3,3:1", seed: int = 0,
              budget: int = 4, n_ref: int = 512, g_budget: int = 58,
              holdouts: str = "0,3,5", max_macro_level: int = 3, quick: bool = False):
    """CALIBRATION 4 — the plant's frontier, and the gate that keeps it one.

    `calr_s0` measured a plant that does not move: parse/infill accuracy flat to three
    decimals over 48 cycles, the true-table audition trendless, and at gen_lr 3e-4 actively
    degrading. The cause is that the fine-tuning diet is IN DISTRIBUTION -- solved
    configurations are ordinary clean derivations, and replay 0.5 is against the same corpus,
    so there is nothing to learn. Round 1 solved the same problem for the *value* by training
    it on a distribution that excluded the contexts it would be metered on; this does it for
    the *plant*.

    `plant_holdout` withholds level-2 tuples from the setup corpus while the evaluation
    instances keep coming from the unfiltered DGP. Every era's damage requires producing a
    correct level-2 tuple, so the plant's hole and the vocabulary's hole are the same hole.

    THE GATE (and it is a gate, not a knob): on instances that REQUIRE a withheld tuple,
    bootstrap solvability must be low but NONZERO. A frontier the agent can never cross is not
    a learner regime, it is exploration-gating by another name, and a ~0 reading is a
    halt-and-report rather than something to tune around."""
    import torch
    ers = parse_eras(eras)
    hs = [int(x) for x in holdouts.split(",")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    started = time.time()
    out = {"eras": ers, "g_budget": g_budget, "holdouts": {}}

    for h in hs:
        cfg = _cfg(seed=seed, budget=budget, n_rt=n_ref, g_budget=g_budget,
                   max_macro_level=max_macro_level, plant_holdout=h)
        if quick:
            cfg.update(controller_steps=800, generator_steps=800, value_steps=800,
                       reader_steps=600, value_episodes=6_000, n_train_episodes=20_000,
                       n_rt=128)
        torch.manual_seed(cfg["train_seed"]); np.random.seed(cfg["train_seed"])
        print(f"\n########## plant_holdout = {h} ##########", flush=True)
        shared = build_shared(cfg, device)
        rules, rules_t, canon = shared["rules"], shared["rules_t"], shared["canon"]
        v, s, depth, m = cfg["v"], cfg["s"], cfg["depth"], cfg["m"]
        gen, reader, hold = shared["generator0"], shared["reader"], shared["hold"]
        inv = shared["inverse_maps"][-1]
        base_ms = shared["base_ms"]
        true_ms = build_ms(base_ms, {ell: shared["truth"][ell]
                                     for ell in range(2, max_macro_level + 1)}, s, depth, device)
        sets = (("base", base_ms), ("true", true_ms))

        clean = torch.from_numpy(
            _sample_pool(rules, cfg["n_rt"], s, cfg["seed"] + 31337)[1]).to(device)
        powers = v ** torch.arange(s, device=device)
        tf = shared["bottom_map"][(clean.view(clean.shape[0], shared["n_blocks"], s)
                                   * powers).sum(-1)]
        ok = tf >= 0
        with torch.no_grad():
            rd = reader.block_logits(clean).argmax(-1)
            gp = gen.block_logits(clean).argmax(-1)
        cell0 = {"read_acc": float((rd[ok] == tf[ok]).float().mean()),
                 "generator_parse_acc": float((gp[ok] == tf[ok]).float().mean()),
                 "n_hold": len(hold), "eras": {}}
        print(f"  READ: reader={cell0['read_acc']:.4f}  "
              f"generator(unsupervised at visible positions)={cell0['generator_parse_acc']:.4f}",
              flush=True)

        for i, era in enumerate(ers):
            r_np, x_np, c_np = context_instances(rules, era_ctx(era), cfg["n_rt"], s, depth,
                                                 v, m, seed=cfg["seed"] + 5000 + 17 * i,
                                                 with_clean=True)
            exc = needs_excluded(c_np, era, hold, inv, v, s)
            x0 = torch.from_numpy(x_np)
            cell = {"frac_excluded": float(exc.mean()), "policies": {}}
            for tag_, mset in sets:
                w = fit_width(len(mset), budget, g_budget)
                b = beam_moves(shared["controller"], gen, shared["value0"], x0,
                               torch.from_numpy(r_np), mset, rules_t, canon, depth, v, m, s,
                               budget=budget, beam_width=w, device=device)
                sc, _ = grade(b["x"].cpu().numpy(), r_np, rules, s)
                cell["policies"][tag_] = {
                    "w": w, "g": b["counts"]["ground"] / x0.shape[0],
                    "e_all": 1.0 - float(sc.mean()),
                    "solve_excluded": float(sc[exc].mean()) if exc.any() else None,
                    "solve_rest": float(sc[~exc].mean()) if (~exc).any() else None}
            # the vocabulary the READER now yields, from exact-DP-solved repairs
            xo, _ = oracle_rollout(gen, x0.to(device), r_np, base_ms, rules, rules_t, canon,
                                   depth, v, m, s, budget)
            so, _ = grade(xo.cpu().numpy(), r_np, rules, s)
            solved = xo[torch.from_numpy(so > 0.5).to(device)]
            cell["oracle_solve_rate"] = float(so.mean())
            cell["vocab"] = {}
            if solved.shape[0]:
                pf = MC.parse_features(reader, solved, s=s).cpu().numpy()
                chain = {1: MC.base_table(v)}
                for ell in range(2, max_macro_level + 1):
                    sp = s ** (ell - 1)
                    nd = (era["node"] * s ** (era["level"] - 1)) // sp
                    mn = MC.Miner(ell, s)
                    seg = pf[:, nd * sp:(nd + 1) * sp]
                    mn.observe(seg[(seg >= 0).all(1)])
                    chain[ell] = mn.build(chain[ell - 1], cfg["mine_support"])
                    cell["vocab"][str(ell)] = {
                        "n_entries": int(chain[ell]["child"].shape[0]),
                        **MC.grade_table(chain[ell], shared["truth"][ell])}
            cell0["eras"][era["name"]] = cell
            pol = cell["policies"]
            print(f"  era {era['name']}: frac_excluded={cell['frac_excluded']:.3f}  "
                  f"oracle_solve={cell['oracle_solve_rate']:.3f}")
            for tag_ in ("base", "true"):
                p = pol[tag_]
                se = p["solve_excluded"]
                print(f"    {tag_:5s} w{p['w']} ({p['g']:.0f}g): e_all={p['e_all']:.3f}  "
                      f"SOLVE|excluded={'n/a' if se is None else f'{se:.3f}'}  "
                      f"solve|rest={p['solve_rest']:.3f}")
            for ell, c in cell["vocab"].items():
                print(f"    vocab L{ell}: {c['n_entries']} entries prec="
                      f"{(c['precision'] if c['precision'] is not None else float('nan')):.3f} "
                      f"recall={c['recall']:.3f}")
        out["holdouts"][str(h)] = cell0

    outdir = f"{DATA_DIR}/{REMOTE}/{tag}"
    os.makedirs(outdir, exist_ok=True)
    out["elapsed"] = time.time() - started
    with open(os.path.join(outdir, "cal_stale.json"), "w") as fh:
        json.dump(out, fh, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"\nDONE in {out['elapsed']:.0f}s -> {outdir}/cal_stale.json")
    return out


# --------------------------------------------------------------------------- #
# gates
# --------------------------------------------------------------------------- #

def selfcheck(v=8, s=2, depth=4, m=2, rule_seed=0, n=256, max_level=3,
              eras="1:6,2:3,3:1"):     # [spiral] the ladder is a parameter now (depth 6)
    """C-M — the macro operator with the DGP's OWN table is BIT-IDENTICAL to the true level
    move (`units.apply_move`) at every level. This is what makes `given` and `practice_*` the
    same machinery differing only in which tuples are in the table, and therefore what makes
    the earned-vs-given fraction a statement about vocabulary rather than about operators.
    C-R — the ratchet constraint bites: a level-3 table built over a truncated level-2 table
    cannot contain entries whose halves were dropped.
    G-D — the nested damage cells are on-grammar at every level and leave d* > 0 often enough
    that rejection sampling is cheap."""
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0); np.random.seed(0)
    length = s ** depth
    n_blocks = length // s
    rules = generate_rules_distinct(v, s, depth, m, seed=rule_seed)
    inv = build_inverse_maps(rules)
    canon = torch.from_numpy(np.ascontiguousarray(rules[depth - 1][:, 0, :])).to(device)
    rules_t = [torch.from_numpy(np.ascontiguousarray(r)).to(device) for r in rules]
    roots_np, leaves_np = _sample_pool(rules, n, s, 3)
    gen = _build_generator()(v, length, s, 96, n_head=4, n_layer=2, root_conditioned=False).to(device)
    gen.eval()
    rng = np.random.default_rng(5)
    x = torch.from_numpy(_corrupt(leaves_np, n_blocks, 3, v, s, rng)).to(device)
    truth = MC.true_tables(rules, depth, s, v, m, max_level)
    out = {"table_sizes": {str(k): int(t["child"].shape[0]) for k, t in truth.items()}}

    # C-M
    cm = {}
    for ell in range(2, max_level + 1):
        ok = True
        for j in range(s ** (depth - ell)):
            true_mv = {"level": ell, "node": j, "blk0": j * s ** (ell - 1),
                       "span": s ** (ell - 1), "name": f"L{ell}n{j}"}
            a = apply_move(gen, x, true_mv, rules_t, canon, depth, v, m, s)
            mac = MC.to_device(MC.make_macro(ell, j, s, truth[ell]), device)
            b = MC.apply_any(gen, x, mac, rules_t, canon, depth, v, m, s)
            ok &= bool(torch.equal(a, b))
        cm[f"L{ell}"] = ok
    out["CM_macro_equals_true_move"] = cm
    assert all(cm.values()), f"C-M failed: {cm}"

    # C-R: truncate T2, rebuild T3 over it, and check the drop
    mn = MC.Miner(3, s)
    flat3 = truth[3]["flat"]
    mn.observe(np.concatenate([flat3] * 3))
    full = mn.build(truth[2], support=1)
    trunc = MC.make_table(2, truth[2]["child"][:6], MC.base_table(v), s)
    part = mn.build(trunc, support=1)
    out["CR_ratchet"] = {"t3_over_full_t2": int(full["child"].shape[0]),
                         "t3_over_truncated_t2": int(part["child"].shape[0]),
                         "t2_full": int(truth[2]["child"].shape[0]), "t2_trunc": 6}
    assert part["child"].shape[0] < full["child"].shape[0], "C-R failed: truncation did not bite"

    # G-D
    gd = {}
    for era in parse_eras(eras):
        dmg = corrupt_hier(leaves_np, rules, depth, v, m, s, era["level"], [era["node"]],
                           np.random.default_rng(7))
        d = nearest_derivation_cost(rules, dmg, roots_np, s)
        gd[era["name"]] = {"on_grammar": on_grammar_rate(dmg, inv[-1], v, s),
                           "dstar": float(d.mean()), "dstar_zero": float((d == 0).mean())}
        assert gd[era["name"]]["on_grammar"] == 1.0, f"G-D failed at {era['name']}"
    out["GD_damage"] = gd

    # grounding table — what the declared budget buys each action set
    out["ground"] = {f"n{n_}_w{w}": beam_ground(n_, 4, w)
                     for n_ in (8, 12, 14) for w in (1, 2, 3, 4)}
    print(json.dumps(out, indent=2, cls=NumpyEncoder))
    return out


@app.function(image=image, timeout=1800, memory=16384)
def selfcheck_remote():
    return selfcheck()


# --------------------------------------------------------------------------- #
# gates for the PORT itself (no training; the fidelity gate's other half)
# --------------------------------------------------------------------------- #

def full_selfcheck(v=8, s=2, depth=4, m=2, rule_seed=0, n=64, max_level=3, budget=3,
                   price_budget=4, price_g=58):   # [spiral] pricing table is a parameter now
    """P-1 .. P-7. The properties the fidelity and twin gates rest on.

    P-1  `select_moves` at k = n_moves returns `arange(n_moves)` for EVERY row, including rows
         of exact ties (a zero-init head gives exactly that) — the tie-ordering hazard the spec
         names, closed by a stable sort rather than by `topk`.
    P-2  `expand_selected` at k = n_moves is BIT-IDENTICAL to the enumerated
         `stack([apply_any(flat, ms[j]) for j], 1)`.
    P-3  `beam_moves_prop` at k = n_moves reproduces `beam_moves` bit-for-bit — same final
         states, same tips, same move sequences, same materialisation and grounding counts
         apart from the one root encode the port pays for and declares.
    P-4  building the head consumes NONE of the global torch stream (the twin gate's
         precondition): the random draw after `build_head` equals the draw without it.
    P-5  the head is exactly uniform at init (zero-init output layer), so a freshly committed
         macro enters the ranking neutrally rather than at an arbitrary trained logit.
    P-6  `select_moves` at k < n_moves returns exactly the k highest-logit moves, ascending,
         and with `forced` returns k + |forced| including every forced index.
    P-7  the (level, node) slot layout is injective over the full 14-move action set, and the
         pricing table `fit_width_k` the k ladder rests on (printed, not asserted).

    And the COMPOSITION's own gates, which are what this phase adds:

    C-1  `PlainExecutor` is `macros.apply_any` on every move, and a non-firing `SpanExecutor`
         is too — so threading an executor through the beam changes nothing by itself.
    C-2  THE COMPOSED FIDELITY GATE: the proposal beam at k = n_moves with a span executor
         present but closed reproduces the enumerating beam bit-for-bit (x, seq, tips, traj,
         materialisation count), the only delta being the declared root encode.
    C-3  an OPEN span slot rewrites exactly the macro's own token positions and nothing else,
         through the composed beam's own executor.
    C-4  minting BOTH heads leaves the shared torch stream exactly where it was — the twin
         gate's precondition, now for two new RNG consumers rather than one."""
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0); np.random.seed(0)
    length = s ** depth
    n_blocks = length // s
    rules = generate_rules_distinct(v, s, depth, m, seed=rule_seed)
    canon = torch.from_numpy(np.ascontiguousarray(rules[depth - 1][:, 0, :])).to(device)
    rules_t = [torch.from_numpy(np.ascontiguousarray(r)).to(device) for r in rules]
    roots_np, leaves_np = _sample_pool(rules, n, s, 3)
    gen = _build_generator()(v, length, s, 96, n_head=4, n_layer=2,
                             root_conditioned=False).to(device)
    ctrl = _build_rich_controller()(v, length, s, 96, n_head=4, n_layer=2).to(device)
    val = _build_value_head()(96, v).to(device)
    for mod in (gen, ctrl, val):
        mod.eval()
    rng = np.random.default_rng(5)
    x = torch.from_numpy(_corrupt(leaves_np, n_blocks, 3, v, s, rng)).to(device)
    roots = torch.from_numpy(roots_np).to(device)
    truth = MC.true_tables(rules, depth, s, v, m, max_level)
    ms = build_ms(build_move_set(depth, s, max_level=1),
                  {ell: truth[ell] for ell in range(2, max_level + 1)}, s, depth, device)
    offsets, n_slots = PN.slot_layout(depth, s, max_level)
    slots = PN.move_slots(ms, offsets)
    out = {"n_moves": len(ms), "n_slots": n_slots, "slots": slots}

    # P-7 (layout)
    assert len(set(slots)) == len(slots) == len(ms), "slot layout is not injective"
    out["P7_slots_injective"] = True
    # [spiral] priced at the caller's declared budget rather than depth 4's hard-coded (4, 58),
    # and over the action-set sizes THIS grammar actually produces, so the table is readable at
    # depth 6 (32 base / 48 after L2 / 56 after L3) as well as at depth 4 (8 / 12 / 14).
    pb, pg = price_budget, price_g
    ns_ = sorted({len(build_move_set(depth, s, max_level=lv))
                  for lv in range(1, max_level + 1)})
    out["P7_price"] = {f"k{k}": {"w": PN.fit_width_k(k, pb, pg),
                                 "g": PN.beam_ground_k(k, pb, PN.fit_width_k(k, pb, pg)) + 1}
                       for k in (1, 2, 3, 4, 8, 12, 14)}
    out["P7_price_enum"] = {f"n{n_}": {"w": fit_width(n_, pb, pg),
                                       "g": beam_ground(n_, pb, fit_width(n_, pb, pg))}
                            for n_ in ns_}

    # P-1 / P-6 (selection)
    nm = len(ms)
    ties = torch.zeros(n, nm, device=device)
    rnd = torch.randn(n, nm, device=device)
    ar = torch.arange(nm, device=device)[None, :].expand(n, -1)
    out["P1_ties_k_eq_n"] = bool(torch.equal(PN.select_moves(ties, nm), ar))
    out["P1_rand_k_eq_n"] = bool(torch.equal(PN.select_moves(rnd, nm), ar))
    assert out["P1_ties_k_eq_n"] and out["P1_rand_k_eq_n"], "P-1 failed"
    sel4 = PN.select_moves(rnd, 4)
    want = torch.sort(torch.topk(rnd, 4, dim=1).indices, dim=1).values
    out["P6_topk_matches"] = bool(torch.equal(sel4, want))
    selF = PN.select_moves(rnd, 2, forced=[8, 9])
    out["P6_forced_shape"] = list(selF.shape)
    out["P6_forced_contains"] = bool(((selF == 8).any(1) & (selF == 9).any(1)).all())
    assert out["P6_topk_matches"] and out["P6_forced_contains"], "P-6 failed"
    assert selF.shape[1] == 4, "P-6: forced expansion is not k + |forced| wide"

    # P-2 (expansion)
    flat = x
    enum_children = torch.stack(
        [MC.apply_any(gen, flat, ms[j], rules_t, canon, depth, v, m, s) for j in range(nm)],
        dim=1)
    sel_all = PN.select_moves(torch.zeros(flat.shape[0], nm, device=device), nm)
    got, n_mat = PN.expand_selected(gen, flat, sel_all, ms, MC.apply_any, rules_t, canon,
                                    depth, v, m, s)
    out["P2_expand_maxabs"] = float((got.long() - enum_children.long()).abs().max())
    out["P2_n_mat"] = int(n_mat)
    assert out["P2_expand_maxabs"] == 0.0, "P-2 failed"

    # P-4 / P-5 (the head costs no shared randomness, and is uniform at init)
    torch.manual_seed(1234)
    ref = torch.randn(8)
    torch.manual_seed(1234)
    head = PN.build_head(96, v, n_slots, seed=99, device=device)
    got_draw = torch.randn(8)
    out["P4_stream_untouched"] = bool(torch.equal(ref, got_draw))
    assert out["P4_stream_untouched"], "P-4 failed: the head consumed the shared stream"
    with torch.no_grad():
        z = _encode_chunked(ctrl, x)
        lg = head(z, roots)
    out["P5_init_logit_maxabs"] = float(lg.abs().max())
    assert out["P5_init_logit_maxabs"] == 0.0, "P-5 failed"

    # P-3 (the beam)
    a = beam_moves(ctrl, gen, val, x, roots, ms, rules_t, canon, depth, v, m, s,
                   budget=budget, beam_width=4, device=device, collect=True)
    b = beam_moves_prop(ctrl, gen, val, x, roots, ms, rules_t, canon, depth, v, m, s,
                        budget=budget, beam_width=4, device=device, collect=True,
                        prop=head, slots=slots, k=nm, forced=(), n_slots=n_slots)
    out["P3_x_equal"] = bool(torch.equal(a["x"], b["x"]))
    out["P3_seq_equal"] = bool(torch.equal(a["seq"], b["seq"]))
    out["P3_tips_equal"] = bool(torch.equal(a["tips_x"], b["tips_x"]))
    out["P3_traj_equal"] = bool(all(torch.equal(p, q) for p, q in zip(a["traj"], b["traj"])))
    out["P3_counts_enum"] = a["counts"]
    out["P3_counts_prop"] = b["counts"]
    out["P3_extra_ground"] = int(b["counts"]["ground"] - a["counts"]["ground"])
    out["P3_mat_equal"] = bool(a["counts"]["mat"] == b["counts"]["mat"])
    assert all(out[k_] for k_ in ("P3_x_equal", "P3_seq_equal", "P3_tips_equal",
                                  "P3_traj_equal", "P3_mat_equal")), "P-3 failed"
    assert out["P3_extra_ground"] == x.shape[0], "P-3: the root encode is mispriced"
    # ---- C-4: BOTH heads cost the shared stream nothing --------------------------------
    torch.manual_seed(4321)
    ref2 = torch.randn(6)
    torch.manual_seed(4321)
    head1 = PN.build_head(96, v, n_slots, seed=7, device=device)
    shead = SN.build_head(SN.slot_count(s, depth, max_level), v, 96, s ** (max_level - 1),
                          seed=11, device=device)
    out["C4_stream_untouched"] = bool(torch.equal(ref2, torch.randn(6)))
    assert out["C4_stream_untouched"], "C-4 failed: a head consumed the shared stream"

    # ---- C-1: the executor seam is a no-op by itself ------------------------------------
    macros_l = [mv for mv in ms if mv.get("kind") == "macro"]
    slots_c = {SN.slot_key(mv["level"], mv["node"]):
               {"id": SN.slot_index(mv["level"], mv["node"], s, depth, max_level),
                "open": False, "move": mv, "level": mv["level"], "node": mv["node"]}
               for mv in macros_l}
    plain = SN.PlainExecutor()
    sx = SN.SpanExecutor(gen, shead, slots_c, cap=4096, hold_cap=1024, hold_frac=0.1,
                         per_call=32, seed=3, v=v, length=length)
    worst = 0.0
    for mv in ms:
        with torch.no_grad():
            a_ = MC.apply_any(gen, x, mv, rules_t, canon, depth, v, m, s)
            b_ = plain.apply(gen, x, mv, rules_t, canon, depth, v, m, s)
            c_ = sx.apply(gen, x, mv, rules_t, canon, depth, v, m, s)
        worst = max(worst, float((a_ - b_).abs().max()), float((a_ - c_).abs().max()))
    out["C1_executor_seam_maxabs"] = worst
    assert worst == 0.0, "C-1 failed: the executor seam is not a no-op"

    # ---- C-2: THE COMPOSED FIDELITY GATE ------------------------------------------------
    sx.capture = False
    e_ = beam_moves(ctrl, gen, val, x, roots, ms, rules_t, canon, depth, v, m, s,
                    budget=budget, beam_width=4, device=device, collect=True,
                    ex=SN.PlainExecutor())
    f_ = beam_moves_prop(ctrl, gen, val, x, roots, ms, rules_t, canon, depth, v, m, s,
                         budget=budget, beam_width=4, device=device, collect=True,
                         prop=head1, slots=slots, k=nm, forced=(), n_slots=n_slots, ex=sx)
    out["C2_x_equal"] = bool(torch.equal(e_["x"], f_["x"]))
    out["C2_seq_equal"] = bool(torch.equal(e_["seq"], f_["seq"]))
    out["C2_tips_equal"] = bool(torch.equal(e_["tips_x"], f_["tips_x"]))
    out["C2_traj_equal"] = bool(all(torch.equal(p_, q_) for p_, q_ in zip(e_["traj"], f_["traj"])))
    out["C2_mat_equal"] = bool(e_["counts"]["mat"] == f_["counts"]["mat"])
    out["C2_extra_ground"] = int(f_["counts"]["ground"] - e_["counts"]["ground"])
    assert all(out[k_] for k_ in ("C2_x_equal", "C2_seq_equal", "C2_tips_equal",
                                  "C2_traj_equal", "C2_mat_equal")), "C-2 failed"
    assert out["C2_extra_ground"] == x.shape[0], "C-2: the root encode is mispriced"

    # ---- C-3: an OPEN slot writes only its own span, through the composed beam ----------
    for k_ in slots_c:
        slots_c[k_]["open"] = True
    sx.fire = True
    c3 = {}
    for mv in macros_l:
        pos = SN.span_positions(mv, x.shape[0], s, device)
        with torch.no_grad():
            got = sx.apply(gen, x, mv, rules_t, canon, depth, v, m, s)
        keepm = torch.ones_like(x, dtype=torch.bool).scatter_(1, pos, False)
        c3[mv["name"]] = {"outside_maxabs": float((got - x)[keepm].abs().max()),
                          "inside_changed": float((got != x).gather(1, pos).float().mean())}
    out["C3_open_slot"] = c3
    out["C3_mat_head"] = int(sx.counts["mat_head"])
    assert all(cc["outside_maxabs"] == 0.0 for cc in c3.values()), "C-3 failed"
    assert out["C3_mat_head"] > 0, "C-3: the head never fired"

    # ---- C-5: THE PRACTICE-PATH CALL SHAPE ----------------------------------------------
    # C-2 exercises the composed beam only at k = n_moves with `forced` and `explore` empty,
    # which is the METERING path. The practice beam runs a different branch — k < n_moves,
    # exploration on — and that branch is where a local name collided with the executor
    # parameter and threw at the first materialisation of the second step. A gate that only
    # covers the metering shape is not a gate on the composition, so it is widened here.
    sx.fire = False
    sx.capture = True
    c5 = {}
    for kk_, exp_, forced_ in ((2, 1, ()), (2, 1, (8, 9)), (4, 0, ()), (1, 1, ())):
        sx.reset()
        g_ = beam_moves_prop(ctrl, gen, val, x, roots, ms, rules_t, canon, depth, v, m, s,
                             budget=budget, beam_width=4, device=device, collect=True,
                             prop=head1, slots=slots, k=kk_, forced=forced_, n_slots=n_slots,
                             explore=exp_, erng=np.random.default_rng(7), ex=sx)
        tot = sum(sx.counts[q] for q in ("mat_base", "mat_dp", "mat_head"))
        c5[f"k{kk_}_e{exp_}_f{len(forced_)}"] = {
            "x_shape": list(g_["x"].shape), "mat": int(g_["counts"]["mat"]),
            "executor_mat": int(tot), "agree": bool(tot == g_["counts"]["mat"]),
            "seq_width": int(g_["seq"].shape[0])}
        assert list(g_["x"].shape) == list(x.shape), "C-5: wrong output shape"
        assert tot == g_["counts"]["mat"], "C-5: beam and executor disagree on materialisations"
        assert len(g_["traj"]) == budget + 1, "C-5: trajectory collection broke"
    out["C5_practice_shapes"] = c5
    sx.fire = True

    print(json.dumps(out, indent=2, cls=NumpyEncoder))
    return out


@app.function(image=image, gpu="L4", timeout=1800, memory=16384)
def full_selfcheck_remote():
    return full_selfcheck()


# =========================================================================== #
# [spiral] THE GATE SET FOR THIS ROUND
# =========================================================================== #

def tall_gates(eras=SPIRAL_ERAS, max_level=3, n=256, rule_seed=0):
    """`tall/tall.py::gates`' T-1 .. T-4, re-derived here (not imported: `tall` imports
    `ear`/`recital`, and this file deliberately depends on neither).

      T-1  the ladder NESTS at all five levels and every node index is in range
           (`s**(depth-l)` nodes at level `l`)
      T-2  damage is 100% on-grammar at EVERY ladder level, with the rejection acceptance rate
           the port pays
      T-3  the oracle's repair distance is MONOTONE in ladder depth — the cost-to-depth the
           whole arc measures — reported with its gradient
      T-4  the action set and its grounding price over the 32..62 move range, at the budget
           this round declares

    Cheap and CPU-only; `tall` reported all four passing at depth 6, m=2 and this re-runs them
    because a re-run costs seconds and an assumption costs a launch."""
    v, s, depth, m = DEPTH6["v"], DEPTH6["s"], DEPTH6["depth"], DEPTH6["m"]
    rules = generate_rules_distinct(v, s, depth, m, seed=rule_seed)
    inv = build_inverse_maps(rules)[-1]
    ers = parse_eras(eras)
    out = {"eras": eras, "depth": depth, "m": m}

    t1 = []
    for i, era in enumerate(ers):
        n_nodes = s ** (depth - era["level"])
        ok_range = 0 <= era["node"] < n_nodes
        nested = True
        if i + 1 < len(ers):
            nxt = ers[i + 1]
            nested = (era["node"] * s ** (era["level"] - 1)) // s ** (nxt["level"] - 1) \
                == nxt["node"]
        t1.append({"era": era["name"], "n_nodes_at_level": n_nodes,
                   "node_in_range": bool(ok_range), "nests_into_next": bool(nested)})
        assert ok_range, f"T-1 failed: {era['name']} node out of range (max {n_nodes - 1})"
        assert nested, f"T-1 failed: {era['name']} does not nest into the next era"
    out["T1_ladder"] = t1

    roots, leaves = _sample_pool(rules, n, s, rule_seed + 31337)
    t23 = {}
    for era in ers:
        rng = np.random.default_rng(7)
        dmg = corrupt_hier(leaves.copy(), rules, depth, v, m, s, era["level"],
                           [era["node"]], rng)
        og = on_grammar_rate(dmg, inv, v, s)
        d = nearest_derivation_cost(rules, dmg, roots, s)
        brk = d > 0
        t23[era["name"]] = {"on_grammar": og, "accept_rate": float(brk.mean()),
                            "dstar_mean": float(d[brk].mean()),
                            "dstar_max": int(d[brk].max()),
                            "p_dstar_gt_budget": {str(b): float((d[brk] > b).mean())
                                                  for b in (4, 6, 8, 10)}}
        assert og == 1.0, f"T-2 failed: damage off-grammar at {era['name']}"
    out["T23_damage"] = t23
    ds = [t23[e["name"]]["dstar_mean"] for e in ers]
    out["T3_depth_gradient"] = float(max(ds) / min(ds))
    assert all(b >= a for a, b in zip(ds, ds[1:])), \
        f"T-3 failed: repair distance is not monotone in depth: {ds}"

    sizes = {f"max_level={lv}": len(build_move_set(depth, s, max_level=lv))
             for lv in range(1, depth)}
    out["T4_action_set"] = sizes
    out["T4_pricing"] = {f"n{n_}_b{b}_w{w}": beam_ground(n_, b, w)
                         for n_ in sorted(set(sizes.values()))
                         for b in (4, 8) for w in (1, 2)}
    # THE PRICING INVERSION, and what routing does to it — the round's central affordance,
    # printed rather than argued. Under enumeration the width the declared budget buys drops
    # when the action set grows at commit; under routing the width depends on k, not on |ms|.
    g = 482
    out["T4_width_enum_at_G482"] = {f"n{n_}": fit_width(n_, 8, g)
                                    for n_ in sorted(set(sizes.values()))}
    out["T4_width_route_at_G482"] = {f"k{k}": PN.fit_width_k(k, 8, g)
                                     for k in (1, 2, 4, 8, 20, 32)}
    out["T4_g_to_hold_w2_enum"] = {f"n{n_}": beam_ground(n_, 8, 2)
                                   for n_ in sorted(set(sizes.values()))}
    return out


@app.function(image=image, timeout=3600, memory=32768)
def gates_remote(eras: str = SPIRAL_ERAS, max_level: int = 3, n: int = 256):
    """CPU gates: the arc's C-M / C-R / G-D at depth 6, plus `tall`'s T-1..T-4."""
    out = {"inherited": selfcheck(v=DEPTH6["v"], s=DEPTH6["s"], depth=DEPTH6["depth"],
                                  m=DEPTH6["m"], n=n, max_level=max_level, eras=eras),
           "tall": tall_gates(eras=eras, max_level=max_level, n=n)}
    print("\n=== SPIRAL GATES (depth 6, m=2) ===")
    print(json.dumps(out["tall"], indent=2, cls=NumpyEncoder))
    return out


@app.function(image=image, gpu="L4", timeout=1800, memory=16384)
def gates_d6_remote(max_level: int = 3, n: int = 64, budget: int = 3):
    """GPU gates: `native`'s OWN fidelity gates (P-1..P-7, C-1..C-5) re-run ON THIS SUBSTRATE.

    P-3 and C-2 are the two that matter for this round's licence to read anything: the proposal
    beam at k = n_moves reproduces the enumerating beam BIT-FOR-BIT (same x, seq, tips, traj,
    materialisation count; the only delta the one declared root encode), with a span executor
    present but closed. `native` asserted them at depth 4 over a 14-move action set; here the
    action set is 56 moves at 64 tokens, which is a different code path in every batched op."""
    out = full_selfcheck(v=DEPTH6["v"], s=DEPTH6["s"], depth=DEPTH6["depth"], m=DEPTH6["m"],
                         n=n, max_level=max_level, budget=budget,
                         price_budget=8, price_g=482)
    return out


@app.function(image=image, volumes={DATA_DIR: volume}, gpu="L4", timeout=3600, memory=32768)
def preflight(cycles: int = 2, eras: str = "1:25:6,2:12:6,3:6:5,4:3:5,5:1:5",
              max_macro_level: int = 4, budget: int = 2, arms: str = ""):
    """`tall/tall.py::preflight`'s idiom: build a TINY depth-6 substrate and call every
    function `phase_a` calls, in the order it calls them, so interface drift fails in ~2
    minutes instead of after a ~510 s paid setup (which is what it cost `tall`, twice).

    The substrate here is trained for a few dozen steps and is scientifically worthless. The
    point is solely that every call SIGNATURE and every `shared[...]` access resolves, that the
    transplanted commit policy and recert actually fire, and that both ports run at 64 tokens.
    It takes a GPU because beam search over a 32-56 move action set at 64 tokens is
    minutes-per-cycle on CPU; `budget=2` keeps the beam shallow, since it is the move set and
    not the rollout depth that exercises the interface."""
    import torch
    # the default ladder here carries PER-ERA cycle counts (the 3-field form), so the
    # per-era-length path Phase B depends on is exercised rather than assumed
    cfg = _d6_cfg(max_macro_level=max_macro_level, budget=budget, g_budget=482,
                  era_cycles=cycles, probe_every=1, probe_widths=(1,),
                  seed=0, rule_seed=0, train_seed=1,
                  # deliberately tiny — an interface test, not an experiment
                  controller_steps=40, generator_steps=40, value_steps=40, reader_steps=40,
                  value_episodes=256, n_train_episodes=2000, value_batch_collect=128,
                  n_pr=8, n_rt=32, n_score=32, n_aud=16, n_probe_clean=32,
                  gen_steps=2, prop_warmup=0, prop_steps=2, prop_buf_cap=2000,
                  recert_every=1, sil_min_cycle=1, checkpoint_every=10 ** 9,
                  span_min_hold=4, span_tau=0.0, span_capture=8, mine_cap=8,
                  tm_episodes=256, preflight_seed_miner=True,
                  # [census] the gate / extend / yoke paths must all FIRE at toy sizes
                  gate_W=1, gate_theta=0, extend_cap=3, extend_tol=1.0, yoke_cycle=2,
                  # [conductor] the LOOP's paths must all fire too — a chosen commit, a chosen
                  # advance, a capped advance and a yoked replay of each. The block geometry is
                  # collapsed (W=1, burn=1) and the dead zones set to float dust so `quiet`
                  # fires as soon as a gauge stops improving, which at six-cycle eras it will.
                  # SAID OUT LOUD: preflight checks the CODE PATH and never a number — these
                  # tolerances are not measured floors and no reading from this run is one.
                  loop_W=2, loop_burn=1, loop_span=1,
                  tol_ledger=1e-9, tol_yield_l3=1e-9, tol_yield_l4=1e-9, tol_endo=1e-9,
                  # [caesura] dust, like every other floor in preflight — SAID OUT LOUD: this
                  # is a code-path check and no number here is a measurement.
                  tol_dsil=1e-9,
                  # [tutti] a TINY menu (n_pr=8 here), so the port's every branch — the
                  # tail draw, the quota, the exact-feature parse, both selector bundles, the
                  # delivery ledger and the dose join — runs at toy sizes before a paid setup.
                  # The K that matters is sized offline (`phase0_tutti.py`) and set at run
                  # level; this number is a code-path check and is not a measurement.
                  question_k=64,
                  n_endo=32, endo_price=3, era_caps=(6, 6, 5, 5, 5), total_cap=0,
                  # [maestro] the LEARNED paths must fire here too. The real mixtures are used
                  # (so the fitted object's shape, its bucket table and its per-gauge estimator
                  # wiring are all exercised) but the dead zones are collapsed to dust, exactly
                  # as the thermostat's are above, because at six-cycle eras on a substrate
                  # trained for forty steps a MEASURED floor would never be reached and the
                  # branch would go untested. SAID OUT LOUD, again: preflight checks the code
                  # path and never a number — nothing here is a fitted or measured quantity.
                  fitted={rw: {**f, "thetas": {b: 1e-9 for b in f["thetas"]}}
                          for rw, f in copy.deepcopy(FITTED).items()},
                  gy_level=4, preflight_seed_frac=0.5, preflight_seed_topup_cycle=4,
                  entry_rec=True)
    ers = parse_eras(eras)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg["train_seed"]); np.random.seed(cfg["train_seed"])
    ok = {}
    _install_identity_miner(cfg["mine_support"])
    _install_entry_recorder()                      # [assay] the instrument is ON in preflight
    ok["entry_recorder_check"] = entry_recorder_check(
        v=cfg["v"], s=cfg["s"], depth=cfg["depth"], m=cfg["m"],
        max_level=cfg["max_macro_level"])
    shared = _spiral_shared(cfg, device, ers)
    junk, jprov = build_junk_pool(shared["truth"], cfg["max_macro_level"])
    shared["junk_pool"] = junk
    ok["junk_pool"] = {**jprov["split"], "provenance": jprov["provenance"]}
    ok["_spiral_shared"] = sorted(shared.keys())
    refs = measure_refs(shared, cfg, ers, device)
    ok["measure_refs"] = bool(refs)
    ok["plant_probe"] = plant_probe(shared["generator0"], shared, cfg, device)
    ok["read_acc"] = float(shared["read_acc"])
    ok["stale_buffer_terminal_success"] = float(shared["replay"]["y"].mean())
    outdir = f"{DATA_DIR}/{REMOTE}/_preflight"
    os.makedirs(outdir, exist_ok=True)
    # every arm SHAPE this round runs: enum, routed, composed, the fidelity twin, the ceiling
    # [census] the instrument's soundness is checked BEFORE anything is paid for
    ok["gy_soundness"] = gy_soundness(s=cfg["s"], depth=cfg["depth"], level=cfg["gy_level"],
                                      eras=ers)
    # [conductor] the pure gates, before any arm runs
    ok["policy_gate"] = PO.policy_gate(verbose=False)
    assert ok["policy_gate"]["ALL"], ok["policy_gate"]
    ok["endo_gate"] = endo_gate(s=cfg["s"], depth=cfg["depth"], eras=ers)
    ok["endo_bench"] = endo_bench(shared, cfg, device, min(cfg["n_endo"], 64), reps=2)
    # checked against the DEFAULT (measured) floors, not preflight's dust — the gate's job is
    # that `MEASURED_FLOORS` and `floors.json` agree, and preflight's tolerances are neither.
    ok["floor_gate"] = floor_gate({**_d6_cfg(), "tol_endo": 1.0})
    ok["floor_gate"]["note"] = ("checked on the MEASURED floors; this preflight itself runs on "
                                "1e-9 dust so the loop's branches fire at toy sizes")
    # [maestro] the fitted policies' gate, on the REAL fitted object (not preflight's dust), so
    # a drift between `FITTED` and `fit.json` fails here rather than after a paid setup.
    ok["fit_gate"] = fit_gate({**_d6_cfg(), "fitted": copy.deepcopy(FITTED)})
    # every arm the MAIN run carries, plus the donor shapes the fidelity gates rest on. The
    # loop arms run here at toy sizes so every branch of the new commit/advance dispatch — a
    # driven commit, a driven advance, a capped advance, and a yoked replay of all three — is
    # executed before a paid setup, which is what `preflight` is for.
    _plans = {}
    # [crescendo] the A3 arms lead, and in the MAIN RUN'S ORDER, so the L4 commit branch, the
    # `commit_max_level` refusal, the L4 extension and the yoke handoff FROM an m4 arm are all
    # executed before a paid setup. `ceiling_m3` must follow `outer_yield_m4`, exactly as in
    # the real run.
    # [crescendo] the arm list is a PARAMETER, defaulting to the full sweep. Every assertion
    # below is guarded on the arm having actually run, so a targeted re-run (`--arms ...`)
    # after a fix re-checks the branches it names instead of re-paying for all of them. The
    # A3 arms lead, in the MAIN RUN'S ORDER, so the L4 commit branch, the `commit_max_level`
    # refusal, the L4 extension and the yoke handoff FROM an m4 arm are all executed before a
    # paid setup; `ceiling_m3` must follow `outer_yield_m4`, exactly as in the real run.
    # [tacet] THE GATE ARMS LEAD, in the MAIN RUN'S ORDER, with `gate_off_y` appended
    # immediately after the baseline: every gate branch (each mode's tip ranking, each mode's
    # mining reorder, the relaxed `mine_cap=0` / `prop_train_on="tips"` diet, the yoke handoff
    # FROM the gated baseline, and the gate log in an arm with no gate) is executed at toy
    # sizes before a paid setup, and the pure-yoke inertness check below has its arm.
    # [intonation] THE delta_perf ARMS LEAD, in the MAIN RUN'S ORDER, with `perf_off_y` and
    # `perf_fid` appended right after `perf_log`: every new branch — the fallible executor's
    # fired and playback paths, the meter's benchmark/gate/2x2, both plasticity-gain modes, the
    # perf ordering on both channels, the yoke handoff FROM a span arm, and a metered arm with
    # the firing threshold left at the donor's tau — runs at toy sizes before a paid setup.
    ARM_SWEEP = (# [tutti] the unification arms first: the split's two policy objects, the
                 # bootstrap under a split, the endogenous selector, the two-policy yoke and
                 # the inertness twin. A loop arm cannot arm on a forty-step substrate, which
                 # is why `tu_pf_split`/`tu_pf_q` hold the TRUE tables (every slot minted at
                 # c1, so `dsil` is live from c2 and BOTH policies are stepped on real reads).
                 "tu_pf_split", "tu_pf_boot", "tu_pf_q", "tu_pf_yk", "tu_pf_off",
                 "tu_y_exo", "tu_d_exo", "tu_s_exo", "tu_s_endo", "tu_s_yk",
                 "tu_d_endo", "tu_y_endo", "tu_m_exo",
                 "dsil_sched", "dsil_yield", "dsil_read", "dsil_and",
                 "dsil_pf_gauge", "dsil_pf_veto", "dsil_pf_boot",
                 "perf_given", "perf_given_g",
                 "mperf_log", "mperf_gain", "mperf_gate", "mout_gate", "mperf_rawx",
                 "perf_log", "perf_off_y", "perf_fid", "perf_gain", "perf_gate",
                 "outcome_gate", "perf_raw",
                 "outer_yield_m4", "gate_off_y", "gate_delta_hi", "gate_delta_lo",
                 "gate_random", "gate_delib", "gate_all",
                 "anchor_long", "ceiling_m3", "outer_yield_m4x",
                 "outer_yield_m3",
                 # `census_extend` carries the extension op on a SCHEDULE commit, which is what
                 # makes gate C-3 non-vacuous at preflight sizes (the loop cannot commit here).
                 "census_extend",
                 "anchor", "outer_yield", "learned_yield", "learned_task", "yoked_learned",
                 "yoked_yield", "outer_endo", "yoked_endo",
                 "outer_ledger", "complete", "exact", "junk_dose", "strip", "given_c1",
                 "spiral_route", "given_native")
    _sweep = tuple(x.strip() for x in arms.split(",") if x.strip()) or ARM_SWEEP
    _ran = lambda a: os.path.isfile(f"{outdir}/{a}/results.json")
    for arm in _sweep:
        _ov = {}
        _ls = ARMS[arm].get("loop") or {}
        if _ls.get("kind") == "yoke":
            _ov["yoke_plan"] = json.dumps(_plans.get(_ls["of"], []))
        # [intonation] the donor's preflight forces the parity gate OPEN (`span_tau=0.0`) so the
        # corridor path is reachable on a forty-step substrate. A treated arm's `span_tau_fire`
        # of 0.50 would then be STRICTER than the gate it is supposed to sit below, and the
        # firing path — the whole node — would go untested. Pinned to the preflight's own
        # `span_tau` here so every metered arm fires; the 0.50 the main run uses is a config
        # fact the reduction reads, not something this run can check.
        if (ARMS[arm].get("cfg") or {}).get("span_tau_fire") is not None:
            _ov["span_tau_fire"] = float(cfg["span_tau"])
        # a driven endo arm needs a floor; at preflight scale there is no measured one, so a
        # placeholder is used and SAID SO — preflight checks the code path, never a number.
        _cfg_a = ({**cfg, "tol_endo": 1e-3, "_preflight_placeholder_endo_tol": True}
                  if _ls.get("read") == "endo" and not cfg["tol_endo"] else cfg)
        _r = run_arm(arm, arm, _ov, shared, _cfg_a, ers, refs, outdir, device)
        _plans[arm] = [{"cycle": a["cycle"], "kind": a["kind"], "level": a["level"]}
                       for a in _r.get("loop_actions", []) if not a.get("cancelled")]
        res = json.load(open(f"{outdir}/{arm}/results.json"))
        ok[f"run_arm:{arm}"] = {
            "cycles": len(res["log"]["cycle"]),
            "loop": res.get("loop"), "loop_actions": res.get("loop_actions"),
            "ledger": res.get("ledger"), "obs_gy_agree": res.get("obs_gy_agree"),
            "panel_last": (res["log"]["panel"][-1] if res["log"].get("panel") else None),
            "g_budget": res["config"]["g_budget"],
            "commits": [(e["level"], e["cycle"], e.get("provisional"),
                         e.get("n_moves_before"), e.get("n_moves_after"),
                         e.get("width_before"), e.get("width_after"),
                         e.get("k_eff_before"), e.get("k_eff_after"))
                        for e in res["events"] if e["kind"] == "commit"],
            "recerts": sum(1 for e in res["events"] if e["kind"] == "recert"),
            "shadow_cert": {k: q["fired"] for k, q in res["shadow_cert"].items()},
            "surgery": [{k: q for k, q in e.items() if k not in ("flat_after",
                                                                 "true_mask_after")}
                        for e in res["events"] if e["kind"] == "surgery"],
            "entry_beam_last": {k: int(sum(v)) for k, v in
                                ((res["log"]["entry"][-1].get("hist") or {}).get("beam", {})
                                 or {}).items()},
            "gy": {"distinct": (res.get("gy_final") or {}).get("n_distinct"),
                   "n_obs": (res.get("gy_final") or {}).get("n_obs"),
                   "at_sup": ((res.get("gy_final") or {}).get("n_at_support") or {}).get("3")},
            "extends": sum(1 for e in res["events"] if e["kind"] == "extend"),
            "entries_added": sum(e.get("n_admitted", 0) for e in res["events"]
                                 if e["kind"] == "extend"),
            "committed_grade_last": res["log"]["committed_grade"][-1],
            "identity": bool(res["log"]["miner"][-1].get("2", {}).get("keys_at_support")
                             is not None)}
    # the instruments must APPEAR, not merely fail to crash. [census] retargeted: the
    # spiral-shaped assertions now read `census_extend` (this round's provisional-committing,
    # earning arm) and the span assertion reads `given_native`, the only arm here that carries
    # a corridor — the rest are routing-only per the finding-9 quarantine.
    sp = json.load(open(f"{outdir}/anchor/results.json")) if _ran("anchor") else None
    if sp is not None:
     assert any(e["kind"] == "commit" and e.get("provisional")
                for e in sp["events"]), "the provisional commit path never fired"
     assert all(c.get("2", {}).get("keys_at_support") is not None
                for c in sp["log"]["miner"]), "the entry-identity instrument missed the log"
     assert sp["log"]["committed_grade"], "the committed-table grade did not reach the log"
    # [crescendo] generalised from the donor's hardcoded {"2","3"}: the shadow certificate runs
    # at every committable level, so at `max_macro_level=4` it is per level 2, 3 AND 4. The
    # donor's literal was a maxl=3 constant, not a claim about the certificate.
     assert set(sp["shadow_cert"]) == {str(q) for q in range(2, cfg["max_macro_level"] + 1)}, \
        f"the shadow certificate is not per level: {sorted(sp['shadow_cert'])}"
    if _ran("given_native"):
        gn = json.load(open(f"{outdir}/given_native/results.json"))
        assert any(c["n_open"] > 0 for c in gn["log"]["span"]), \
            "no span slot ever opened in given_native — the corridor path is untested"
        ok["span_open_max"] = max(int(c["n_open"]) for c in gn["log"]["span"])
    ok["task_matched_buffer"] = {"random_blocks": shared.get("stale_random_blocks"),
                                 "task_matched": shared.get("stale_task_matched")}
    assert shared.get("stale_task_matched") is not None, \
        "task-matched collection did not run"

    # ---- [crescendo] GATE C: the A3 branches must have EXECUTED, not merely not crashed.
    # Everything below is a CODE-PATH check at toy sizes on a seeded miner; no number here is
    # a measurement, and the L4 tables the preflight commits are hand-fed truth.
    #
    # WHY THE CEILING CONTROL IS TESTED ON THE SCHEDULE ARM AND NOT ON THE LOOP ARMS. The
    # thermostat's "positive-then-flat" precondition (P-5) means it may not act on a gauge that
    # has never cleared its own floor, and at preflight sizes the observation panel sees almost
    # nothing solved, so the yield series is flat at zero and the loop CORRECTLY never arms.
    # (`preflight_seed_miner` seeds `miners[...]`, which is what makes a commit non-empty; it
    # deliberately does not seed the read-only panel, which would be seeding the gauge.) So the
    # loop arms ride their caps here, and testing `commit_max_level` on them would be vacuous.
    # It is tested where it is not vacuous: on the schedule arm, which commits at the era
    # boundary by `delta_prov`, against a preflight-only clone of itself carrying the ceiling
    # control's one cfg override and nothing else. That IS the mechanism, isolated.
    # [intonation] guarded: the arm list is a parameter, and running the delta_perf subset
    # alone (which is the normal way to re-check this node after a fix) leaves the donor's
    # schedule arm absent. `tacet` guarded the one donor line that assumed the full sweep for
    # exactly this reason; gate C is re-checked whenever `anchor_long` is in `--arms`.
    ok["crescendo_gate_C"] = "skipped - anchor_long not in --arms"

    def _commits(r, lv=None):
        return [e for e in r["events"]
                if e["kind"] == "commit" and (lv is None or e["level"] == lv)]

    if _ran("anchor_long"):
        _al = json.load(open(f"{outdir}/anchor_long/results.json"))

        # C-1  THE NEW RUNG IS REACHABLE: an L4 commit installs a level-4 table and grows the
        #      action set by one macro per L4 node (2**(depth-4) = 4).
        _al_l4 = _commits(_al, 4)
        assert _al_l4, "no arm committed L4 — the new rung's commit path is untested"
        _grew = _al_l4[0]["n_moves_after"] - _al_l4[0]["n_moves_before"]
        assert _grew == cfg["s"] ** (cfg["depth"] - 4), \
            f"the L4 commit did not grow the action set by one macro per L4 node: +{_grew}"

        # C-2  `commit_max_level` BINDS, and binds ONLY on the level it names. The clone runs on
        #      the same stream, the same spec and the same cfg but for the one key.
        _r3 = run_arm("anchor_long_m3", "anchor_long", {"commit_max_level": 3},
                      shared, cfg, ers, refs, outdir, device)
        _c3 = json.load(open(f"{outdir}/anchor_long_m3/results.json"))
        assert not _commits(_c3, 4), "commit_max_level=3 did not forbid the L4 commit"
        assert ([(e["level"], e["cycle"]) for e in _commits(_c3)]
                == [(e["level"], e["cycle"]) for e in _commits(_al) if e["level"] <= 3]), \
            "commit_max_level=3 moved a commit at a level it does not name"
        assert _c3["log"]["panel"][-1]["commit_max_level"] == 3, "commit_max_level did not reach"
        assert _al["log"]["panel"][-1]["commit_max_level"] == cfg["max_macro_level"]
        #      and the pair must be BIT-IDENTICAL up to the forbidden commit — the property the
        #      whole ceiling-control design rests on.
        _cut = int(_al_l4[0]["cycle"])
        _w = max(_cut - 1, 0)
        _worst = max(float(np.abs(np.asarray(_al["log"][k][:_w], float)
                                  - np.asarray(_c3["log"][k][:_w], float)).max())
                     for k in ("e", "succ", "n_moves", "width", "n_solved", "gloss")) if _w else 0.0
        assert _worst == 0.0, \
            f"the ceiling pair diverges BEFORE the forbidden L4 commit (c{_cut}): max|d|={_worst}"

        # C-3  EXTENSION reaches the new rung's frozen table. Checked on `census_extend`, which
        #      commits by the same boundary rule and carries `extend` — the loop-driven extension
        #      arm cannot commit here for the reason given above.
        _cx = json.load(open(f"{outdir}/census_extend/results.json"))
        _ext_lv = {e["level"] for e in _cx["events"] if e["kind"] == "extend"}
        assert _ext_lv, "census_extend never ran an extension"
        assert 4 in _ext_lv or cfg["max_macro_level"] < 4, \
            f"extension never reached the new rung (levels seen: {sorted(_ext_lv)})"

        # C-4  THE LIFETIME CEILING binds: no loop arm outruns the schedule arm.
        _n = lambda r: len(r["log"]["cycle"])
        _loops = {}
        for _lbl in ("outer_yield_m4", "ceiling_m3", "outer_yield_m4x", "outer_yield_m3"):
            _loops[_lbl] = json.load(open(f"{outdir}/{_lbl}/results.json"))
            assert _n(_loops[_lbl]) <= _n(_al), \
                f"{_lbl} ran {_n(_loops[_lbl])} cycles against the schedule arm's {_n(_al)} — the " \
                f"lifetime ceiling does not bind; `eras` and `era_caps` must agree"
        # C-5  neither ceiling arm holds an L4 table, whatever the loop did.
        for _lbl in ("ceiling_m3", "outer_yield_m3"):
            assert not _commits(_loops[_lbl], 4), f"{_lbl} committed L4"

        ok["crescendo_gate_C"] = {
            "C-1 L4 commit (schedule arm)": [
                (e["level"], e["cycle"], e["n_entries"], e.get("tab_recall"),
                 e.get("tab_precision"), e["n_moves_before"], e["n_moves_after"]) for e in _al_l4],
            "C-2 commit_max_level binds": {
                "anchor_long": [(e["level"], e["cycle"]) for e in _commits(_al)],
                "anchor_long_m3": [(e["level"], e["cycle"]) for e in _commits(_c3)],
                "pre_commit_window": _w, "max_abs_delta": _worst},
            "C-3 extend levels": sorted(_ext_lv),
            "C-4 cycles": {"anchor_long": _n(_al),
                           **{k: _n(v) for k, v in _loops.items()}},
            "C-5 loop_armed_at_preflight_scale": {
                k: bool(v.get("loop_actions")) for k, v in _loops.items()},
            "note": "code paths only — the preflight seeds its miners from truth and its dead "
                    "zones are 1e-9 dust, so no number here is a measurement"}

    # ---- [tacet] GATE T: the gate must have BOUND, and the yoke must be INERT without it.
    # Every check here is a code-path / invariant check at toy sizes; no number is a
    # measurement. The gate's own effect is a question for the main run, not for this one.
    _T = {}
    _base = json.load(open(f"{outdir}/outer_yield_m4/results.json")) if \
        _ran("outer_yield_m4") else None
    _gseries = ("e", "succ", "n_moves", "width", "n_solved", "n_mined", "gloss", "vloss")

    def _maxd(r1, r2, keys=_gseries, n=None):
        n = n if n is not None else min(len(r1["log"]["cycle"]), len(r2["log"]["cycle"]))
        return max(float(np.abs(np.asarray(r1["log"][k][:n], float)
                                - np.asarray(r2["log"][k][:n], float)).max())
                   for k in keys) if n else 0.0

    # T-1  THE PURE YOKE IS INERT. `gate_off_y` replays the baseline's own realised actions
    #      with no gate and no relaxation, so it must be BIT-IDENTICAL to it over its whole
    #      life. This is what licenses reading every gate arm as "the baseline plus one knob"
    #      rather than "the baseline plus a yoke plus one knob".
    if _base is not None and _ran("gate_off_y"):
        _gy = json.load(open(f"{outdir}/gate_off_y/results.json"))
        _d = _maxd(_base, _gy)
        _T["T-1 pure yoke inert"] = {"max_abs_delta": _d,
                                     "cycles": (len(_base["log"]["cycle"]),
                                                len(_gy["log"]["cycle"])),
                                     "commits_equal": ([(e["level"], e["cycle"]) for e in
                                                        _commits(_base)]
                                                       == [(e["level"], e["cycle"]) for e in
                                                           _commits(_gy)])}
        assert _d == 0.0 and _T["T-1 pure yoke inert"]["commits_equal"], \
            f"a pure yoke with the gate off is NOT the baseline: {_T['T-1 pure yoke inert']}"

    # T-2  THE GATE LOG EXISTS IN EVERY ARM, including the ungated baseline — that is what
    #      makes the counterfactual gate computable offline everywhere.
    for _lbl in ("outer_yield_m4", "gate_delta_hi", "gate_delta_lo", "gate_random",
                 "gate_delib", "gate_all"):
        if not _ran(_lbl):
            continue
        _r = json.load(open(f"{outdir}/{_lbl}/results.json"))
        _g = _r["log"].get("gate") or []
        assert len(_g) == len(_r["log"]["cycle"]), \
            f"{_lbl}: the gate log has {len(_g)} rows against {len(_r['log']['cycle'])} cycles"
        assert all(len(q["d_ans"]) == _r["config"]["n_pr"] for q in _g), \
            f"{_lbl}: the gate log's per-instance features are not one per instance"
        _T[f"T-2 gate log:{_lbl}"] = {
            "rows": len(_g),
            "mode": _g[-1]["mode"],
            "n_sol_tip": [q["n_sol_tip"] for q in _g[-3:]],
            "n_keep_tip": [q["n_keep_tip"] for q in _g[-3:]],
            "n_pairs": [q["n_pairs"] for q in _g[-3:]],
            "d_ans_range": [round(min(min(q["d_ans"]) for q in _g), 4),
                            round(max(max(q["d_ans"]) for q in _g), 4)],
            "margin_range": [round(min(min(q["margin"]) for q in _g), 4),
                             round(max(max(q["margin"]) for q in _g), 4)],
            # [tacet] how often the textbook top1-top2 margin is EXACTLY zero (a duplicate
            # runner-up), which is why the gate reads the mean-over-unchosen form instead.
            "m2_exact_zero_frac": round(
                sum(1 for q in _g for x in q["m2"] if x == 0.0)
                / max(1, sum(len(q["m2"]) for q in _g)), 4)}

    # T-3  THE GATE BINDS: on a cycle with more than one solved tip, a gate arm must keep
    #      strictly fewer of them than the baseline did, and it must keep the number
    #      `gate_frac` names — the same number in every gate arm, which is what makes the
    #      random arm a matched-volume control rather than merely a random one.
    _keeps = {}
    for _lbl in ("gate_delta_hi", "gate_delta_lo", "gate_random", "gate_delib"):
        if not _ran(_lbl):
            continue
        _g = json.load(open(f"{outdir}/{_lbl}/results.json"))["log"]["gate"]
        _keeps[_lbl] = [(q["n_sol_tip"], q["n_keep_tip"]) for q in _g]
        for _ns, _nk in _keeps[_lbl]:
            if _ns:
                assert _nk == max(1, int(round(cfg["gate_frac"] * _ns))), \
                    f"{_lbl}: kept {_nk} of {_ns} solved tips at gate_frac={cfg['gate_frac']}"
    if len(_keeps) > 1:
        _ref = list(_keeps.values())[0]
        for _lbl, _v in _keeps.items():
            assert [q[1] for q in _v[:len(_ref)]] == [q[1] for q in _ref[:len(_v)]], \
                f"{_lbl}: the gate arms are not volume-matched cycle for cycle"
    _T["T-3 gate binds"] = {k: v[-3:] for k, v in _keeps.items()}
    # the gate can only REFUSE something on a cycle that solved more than one tip; at preflight
    # sizes (24 instances, a 40-step substrate) that is not guaranteed, so the refusal check is
    # asserted only where it is defined and RECORDED either way.
    _binds = [(ns, nk) for v in _keeps.values() for ns, nk in v if ns > 1]
    _T["T-3 refusals"] = {"cycles_with_>1_solved_tip": len(_binds),
                          "refused_on": sum(1 for ns, nk in _binds if nk < ns)}
    if _binds:
        assert any(nk < ns for ns, nk in _binds), \
            "the gate never refused a solved tip on any cycle where it could have"

    # T-4  THE MINING CHANNEL IS VOLUME-MATCHED AND CONTENT-MOVED. Every gate arm mines the
    #      same NUMBER of answers as the baseline on every cycle (the cap is untouched), and
    #      at least one of them mines a DIFFERENT set (otherwise the reorder is a no-op).
    if _base is not None:
        for _lbl in ("gate_delta_hi", "gate_delta_lo", "gate_random", "gate_delib"):
            if not _ran(_lbl):
                continue
            _r = json.load(open(f"{outdir}/{_lbl}/results.json"))
            _n = min(len(_base["log"]["cycle"]), len(_r["log"]["cycle"]))
            # only the pre-divergence window is a fair test: after the arms diverge they solve
            # different instances, so the mined COUNT may legitimately differ.
            _pre = next((i for i in range(_n)
                         if _base["log"]["n_solved"][i] != _r["log"]["n_solved"][i]), _n)
            assert (_base["log"]["n_mined"][:_pre] == _r["log"]["n_mined"][:_pre]), \
                f"{_lbl}: the mining CAP moved before the arms diverged — volume is not matched"
            _T[f"T-4 mining matched:{_lbl}"] = {"pre_divergence_cycles": _pre,
                                                "n_mined": _r["log"]["n_mined"][:_pre]}

    # T-5  `gate_all` RELAXES BOTH CHANNELS: it mines without a cap (so on a cycle where the
    #      baseline hit the cap it mines strictly more) and supervises pi on unsolved tips too.
    if _base is not None and _ran("gate_all"):
        _ga = json.load(open(f"{outdir}/gate_all/results.json"))
        assert _ga["config"]["mine_cap"] == 0 and _ga["config"]["prop_train_on"] == "tips", \
            "gate_all did not receive the relaxed diet"
        _gg = _ga["log"]["gate"]
        _T["T-5 learn-from-everything"] = {
            "mine_cap": _ga["config"]["mine_cap"],
            "prop_train_on": _ga["config"]["prop_train_on"],
            "n_mined_last3": _ga["log"]["n_mined"][-3:],
            "baseline_n_mined_last3": _base["log"]["n_mined"][-3:],
            "n_pairs_last3": [q["n_pairs"] for q in _gg[-3:]],
            "baseline_n_pairs_last3": [q["n_pairs"] for q in
                                       (_base["log"]["gate"] or [{}])[-3:]]}
    ok["tacet_gate_T"] = _T

    # ---- [intonation] GATE I: the meter is real, the executor is fallible-not-vandalised, and
    # the 2x2 is NON-DEGENERATE. Code paths and invariants at toy sizes; no number here is a
    # measurement. Gate I-3 is the one the brief names explicitly — below-parity firing has to
    # actually produce executed-NOT-as-intended rows, or the whole node measures nothing.
    _I = {}
    # the 2x2 gate reads the first metered arm that MINTED slots and fired; at preflight sizes
    # that is `perf_given` (see its arm note), and in the main run it is `perf_log`.
    _pl = None
    for _c in ("perf_given", "perf_log", "perf_gain"):
        if not _ran(_c):
            continue
        _cand = json.load(open(f"{outdir}/{_c}/results.json"))
        if sum(int(q.get("n_fired", 0)) for q in (_cand["log"].get("perf") or [])):
            _pl = _cand
            _I["I-3 source arm"] = _c
            break

    # I-0  the executor primitive is `native/span/span_net.py`'s, imported and subclassed —
    #      never re-implemented. If this ever fails, gates S-1..S-6 no longer cover this file.
    _PE = _perf_executor(SN.SpanExecutor)
    _I["I-0 imported primitive"] = {
        "PerfExecutor_bases": [c.__name__ for c in _PE.__mro__[1:3]],
        "span_net_module": SN.__name__,
        "subclass_of_SpanExecutor": issubclass(_PE, SN.SpanExecutor),
        "parity_is_span_nets": SN.parity.__module__,
        "dp_features_is_span_nets": SN.dp_features.__module__}
    assert issubclass(_PE, SN.SpanExecutor) and SN.parity.__module__.endswith("span_net"), \
        _I["I-0 imported primitive"]

    # I-1  METERING ALONE MOVES NOTHING. `perf_fid` carries the meter with the firing threshold
    #      left at the donor's tau, so it is `span`-on-at-parity with an observer attached; it
    #      must be bit-identical to a `perf_off_y` yoke, which has neither. This is what
    #      separates "the head fires below parity" (the treatment) from "the meter runs" (an
    #      instrument), and it is the reason the meter is allowed to be unpriced.
    if _ran("perf_fid") and _ran("perf_off_y"):
        _pf = json.load(open(f"{outdir}/perf_fid/results.json"))
        _po = json.load(open(f"{outdir}/perf_off_y/results.json"))
        _d = _maxd(_pf, _po)
        _I["I-1 meter inert"] = {
            "max_abs_delta": _d,
            "cycles": (len(_pf["log"]["cycle"]), len(_po["log"]["cycle"])),
            "commits_equal": ([(e["level"], e["cycle"]) for e in _commits(_pf)]
                              == [(e["level"], e["cycle"]) for e in _commits(_po)])}
        assert _d == 0.0 and _I["I-1 meter inert"]["commits_equal"], \
            f"the meter is NOT inert at tau_fire == span_tau: {_I['I-1 meter inert']}"

    # I-2  THE FIRING THRESHOLD BINDS, AND ONLY IT. Every metered arm carries tau_fire 0.50;
    #      `perf_fid` carries the donor's 0.95. The gate EVENTS carry `open_tau` beside `open`,
    #      so the tau = 0.95 counterfactual is on the record in every arm.
    for _lbl in ("perf_given", "perf_given_g", "perf_log", "perf_gain", "perf_raw",
                 "perf_gate", "outcome_gate"):
        if not _ran(_lbl):
            continue
        _r = json.load(open(f"{outdir}/{_lbl}/results.json"))
        assert _r["config"]["perf_meter"], f"{_lbl} did not receive the meter"
        assert _r["config"]["span_tau_fire"] is not None, f"{_lbl} has no firing threshold"
        assert float(_r["config"]["span_tau_fire"]) <= float(_r["config"]["span_tau"]), \
            f"{_lbl}'s firing threshold sits ABOVE its parity gate — nothing fires below parity"
        _ge = _r.get("gate_events") or []
        _I[f"I-2 firing:{_lbl}"] = {
            "tau_fire": _r["config"]["span_tau_fire"], "span_tau": _r["config"]["span_tau"],
            "n_gate_events": len(_ge),
            "opened_below_tau": sum(1 for e in _ge if e.get("open")
                                    and not e.get("open_tau")),
            "n_open_last": (_r["log"]["span"][-1]["n_open"] if _r["log"]["span"] else None),
            "perf_rows": len(_r["log"].get("perf") or [])}
        assert len(_r["log"].get("perf") or []) == len(_r["log"]["cycle"]), \
            f"{_lbl}: the meter log has {len(_r['log'].get('perf') or [])} rows against " \
            f"{len(_r['log']['cycle'])} cycles"

    # I-3  THE 2x2 IS NON-DEGENERATE — the gate the brief names. Below-parity firing must
    #      produce (a) actual head executions, (b) executions that were NOT as intended, and
    #      (c) both a bad-and-solved cell (the LUCKY SUCCESS) and an as-intended cell. If the
    #      head never fires, or never misses, `e` is identically zero and delta_perf does not
    #      exist on this substrate after all.
    if _pl is not None:
        _pr = _pl["log"].get("perf") or []
        _cells = [q["cells"] for q in _pr if q.get("cells")]
        _tot = {k: int(sum(c.get(k, 0) for c in _cells))
                for k in ("int_solved", "int_failed", "bad_solved", "bad_failed",
                          "non_solved", "non_failed", "n_exec_tip")}
        _fired = int(sum(q.get("n_fired", 0) for q in _pr))
        _miss = int(sum(q.get("n_misfire", 0) for q in _pr))
        _I["I-3 2x2 non-degenerate"] = {
            **_tot, "n_fired_rows": _fired, "n_misfire_rows": _miss,
            "misfire_frac": round(_miss / max(_fired, 1), 4),
            "n_slots_with_bench": len(_pr[-1]["bench"]) if _pr else 0,
            "calibrated": bool(_pr[-1]["calibrated"]) if _pr else False,
            "g0": _pr[-1]["g0"] if _pr else None,
            "theta": _pr[-1]["theta"] if _pr else None}
        # At preflight sizes the substrate is trained for a few dozen steps and a slot may not
        # reach `span_min_hold` at all, so this is asserted only where firing HAPPENED and
        # recorded either way — the main run's smoke is where it must bite.
        _I["I-3 2x2 non-degenerate"]["both_rows"] = bool(
            (_tot["int_solved"] + _tot["int_failed"] > 0)
            and (_tot["bad_solved"] + _tot["bad_failed"] > 0))
        if _fired:
            # THE GATE THE NODE RESTS ON, in the half this run can check: below-parity firing
            # has to produce executed-NOT-as-intended rows. The other half — that the
            # as-intended row is also populated — needs a TRAINED head, which a forty-step
            # preflight substrate is not; it is checked on the smoke and on the main run's own
            # meter log (reduction P1/P2), and recorded here as `both_rows`.
            assert _miss > 0, (
                "the head fired and was NEVER wrong — `e` is identically zero, so there is no "
                "performance error on this substrate: lower `span_tau_fire`")
            assert _tot["bad_solved"] + _tot["bad_failed"] > 0, \
                "no executed-NOT-as-intended trajectory at all — the 2x2 has one row"

    # I-4  THE GAIN ARMS ACTUALLY WEIGHT. `perf_gain`/`perf_raw` must have produced a weight
    #      statistic with credit attached to a real fraction of their rows, and their mean
    #      weight must sit near 1 (the matched-average-budget normalisation working).
    for _lbl in ("perf_given_g", "perf_gain", "perf_raw"):
        if not _ran(_lbl):
            continue
        _r = json.load(open(f"{outdir}/{_lbl}/results.json"))
        _ws = [q["wstat"] for q in (_r["log"].get("perf") or []) if q.get("wstat")]
        assert _r["config"]["perf_gain"] in ("delta", "raw"), f"{_lbl}: no gain mode"
        _I[f"I-4 gain:{_lbl}"] = {
            "mode": _r["config"]["perf_gain"], "n_weighted_cycles": len(_ws),
            "mean_w": (round(sum(q["sum_w"] for q in _ws)
                             / max(sum(q["n"] for q in _ws), 1), 4) if _ws else None),
            "cred_frac": (round(sum(q["n_cred"] for q in _ws)
                                / max(sum(q["n"] for q in _ws), 1), 4) if _ws else None)}

    # I-5  THE PERF GATE BINDS AND IS VOLUME-MATCHED TO THE OUTCOME GATE. Same `gate_frac`,
    #      same kept count cycle-for-cycle: that is what makes `outcome_gate` a matched-volume
    #      CONTENT control for `perf_gate` rather than merely a different arm.
    _pk = {}
    for _lbl in ("perf_gate", "outcome_gate"):
        if not _ran(_lbl):
            continue
        _g = json.load(open(f"{outdir}/{_lbl}/results.json"))["log"]["gate"]
        _pk[_lbl] = [(q["n_sol_tip"], q["n_keep_tip"]) for q in _g]
        _I[f"I-5 keeps:{_lbl}"] = {"last3": _pk[_lbl][-3:],
                                   "p_tie_last3": [q.get("p_tie") for q in _g[-3:]],
                                   "mine_cells_last3": [(q.get("mine_int"), q.get("mine_bad"),
                                                         q.get("mine_non")) for q in _g[-3:]]}
    if len(_pk) == 2:
        _a, _b = list(_pk.values())
        _n2 = min(len(_a), len(_b))
        assert [q[1] for q in _a[:_n2]] == [q[1] for q in _b[:_n2]], \
            "perf_gate and outcome_gate are not volume-matched cycle for cycle"
        _I["I-5 volume matched"] = True
    ok["intonation_gate_I"] = _I

    # ---- [caesura] GATE D: the delta-silence gauge exists, is readable by A1's UNCHANGED
    # thermostat, is DRIVEN in exactly the arm that names it, and the veto both fires and stays
    # inert where the signal cannot speak. Code paths and invariants at toy sizes; no number is
    # a measurement.
    _D = {}
    for _lbl in ("dsil_sched", "dsil_yield", "dsil_read", "dsil_and",
                 "dsil_pf_gauge", "dsil_pf_veto", "dsil_pf_boot"):
        if not _ran(_lbl):
            continue
        _r = json.load(open(f"{outdir}/{_lbl}/results.json"))
        _pn = _r["log"]["panel"]
        _has = [q for q in _pn if q.get("dsil") is not None]
        _D[f"D-1 gauge:{_lbl}"] = {
            "panel_rows": len(_pn),
            "cycles_with_dsil": len(_has),
            "first_cycle_with_dsil": (_has[0]["cycle"] if _has else None),
            "dsil_range": ([round(min(q["dsil"] for q in _has), 6),
                            round(max(q["dsil"] for q in _has), 6)] if _has else None),
            "max_open_slots": max((q.get("dsil_open") or 0) for q in _pn),
            "read": (_r.get("loop") or {}).get("read"),
            "dsil_events": len(_r.get("dsil_events") or [])}
        # D-1  the gauge is in EVERY arm's panel (that is what makes the counterfactual
        #      computable offline everywhere), and it is None before any slot opens.
        assert len(_pn) == len(_r["log"]["cycle"]), f"{_lbl}: panel rows != cycles"
        assert all((q.get("dsil") is None) == ((q.get("dsil_open") or 0) == 0) for q in _pn), \
            f"{_lbl}: dsil is present exactly when a slot is open — it is not"
    # D-2  the READ is what the arm names, and only that arm drives on it.
    if _ran("dsil_read") and _ran("dsil_yield"):
        _rr = json.load(open(f"{outdir}/dsil_read/results.json"))
        _ry = json.load(open(f"{outdir}/dsil_yield/results.json"))
        _D["D-2 driven read"] = {"dsil_read": (_rr.get("loop") or {}).get("read"),
                                 "dsil_yield": (_ry.get("loop") or {}).get("read")}
        assert (_rr.get("loop") or {}).get("read") == "dsil", "dsil_read is not reading dsil"
        assert (_ry.get("loop") or {}).get("read") == "yield", "dsil_yield lost its read"
    # D-3  THE VETO FIRES, AND IS INERT WHERE THE SIGNAL CANNOT SPEAK. Every `absent` event is
    #      a cycle where a commit was licensed with no open slot; the veto must have let it
    #      through, or a gauge that does not exist would be blocking the L2 commit forever.
    _vet = "dsil_pf_veto" if _ran("dsil_pf_veto") else ("dsil_and" if _ran("dsil_and")
                                                        else None)
    if _vet:
        _ra = json.load(open(f"{outdir}/{_vet}/results.json"))
        _D["D-3 veto source arm"] = _vet
        _ev = _ra.get("dsil_events") or []
        _D["D-3 veto"] = {
            "n_events": len(_ev),
            "by_kind": {k: sum(1 for e in _ev if e["kind"] == k)
                        for k in ("defer", "pass", "absent")},
            "commits": [(e["level"], e["cycle"]) for e in _commits(_ra)],
            "levels_deferred": sorted({e["level"] for e in _ev if e["kind"] == "defer"})}
        assert all(e["dsil"] is None for e in _ev if e["kind"] == "absent"), \
            "an `absent` veto event carried a live gauge"
        assert not any(e["kind"] == "defer" and e["dsil"] is None for e in _ev), \
            "the veto blocked a commit on a gauge that does not exist — it is not inert"
        # the veto may only DELAY: it must not remove a level the yield arm reached, unless the
        # run ended first. Recorded, and asserted only where the comparator ran.
        if _ran("dsil_yield"):
            _ry = json.load(open(f"{outdir}/dsil_yield/results.json"))
            _D["D-3 veto"]["yield_commits"] = [(e["level"], e["cycle"]) for e in _commits(_ry)]
        assert _D["D-3 veto"]["n_events"] > 0, (
            f"{_vet}: the veto was never consulted — the conjunction path is untested")
    # D-4  `dsil_sched` reads nothing and must carry no dsil events at all.
    if _ran("dsil_sched"):
        _rs = json.load(open(f"{outdir}/dsil_sched/results.json"))
        assert not (_rs.get("dsil_events") or []), "the schedule arm ran the veto"
        _D["D-4 schedule inert"] = {"dsil_events": 0,
                                    "commits": [(e["level"], e["cycle"])
                                                for e in _commits(_rs)]}
    # D-5  THE BOOTSTRAP FIRES, AND ONLY WHILE THE GAUGE IS ABSENT. Without it a `dsil`-driven
    #      arm deadlocks by construction (slots are minted BY a commit, and the gauge is
    #      computed over slots), which is a fact about the signal's TYPE, not about scale.
    _bs = "dsil_pf_boot" if _ran("dsil_pf_boot") else ("dsil_read" if _ran("dsil_read")
                                                       else None)
    if _bs:
        _rb = json.load(open(f"{outdir}/{_bs}/results.json"))
        _ev = [e for e in (_rb.get("dsil_events") or []) if e["kind"] == "bootstrap"]
        _pn = _rb["log"]["panel"]
        _first = next((q["cycle"] for q in _pn if q.get("dsil") is not None), None)
        _D["D-5 bootstrap"] = {
            "arm": _bs, "n_bootstrap_commits": len(_ev),
            "at": [(e["level"], e["cycle"], e["why"]) for e in _ev],
            "first_cycle_with_dsil": _first,
            "commits": [(e["level"], e["cycle"]) for e in _commits(_rb)],
            "max_open_slots": max((q.get("dsil_open") or 0) for q in _pn)}
        assert _ev, (f"{_bs}: the bootstrap never fired, so the gauge never came alive and a "
                     f"dsil-driven arm deadlocks — see the arm note")
        assert _first is not None and max((q.get("dsil_open") or 0) for q in _pn) > 0, \
            f"{_bs}: no slot ever opened even after the bootstrap commit"
        assert all(_first is None or e["cycle"] < _first for e in _ev), \
            f"{_bs}: a bootstrap commit was taken while the gauge WAS available"
    ok["caesura_gate_D"] = _D

    # ---- [tutti] GATE X: THE SPLIT. Two policy objects, each licensing one action; the
    # bootstrap under the split; the donor's precedence preserved; the two-policy yoke; and
    # inertness when `loop_commit` is absent. Code paths and invariants at toy sizes; no number
    # here is a measurement.
    _X = {}

    def _acts(rj, kind):
        return [a for a in (rj.get("loop_actions") or []) if a["kind"] == kind
                and not a.get("cancelled")]

    # X-1  the two policies exist, are DISTINCT, and each reads only its own series.
    for _lbl in ("tu_pf_split", "tu_pf_boot", "tu_pf_q", "tu_s_exo", "tu_s_endo",
                 "tu_m_exo"):
        if not _ran(_lbl):
            continue
        _r = json.load(open(f"{outdir}/{_lbl}/results.json"))
        _sp = _r.get("split")
        assert _sp, f"{_lbl}: no split record — `loop_commit` did not build a second policy"
        assert _sp["commit_read"] != _sp["advance_read"], \
            f"{_lbl}: both policies read {_sp['commit_read']!r} — that is not a split"
        _want = ARMS[_lbl]["loop_commit"]["read"]
        assert _sp["commit_read"] == _want, \
            f"{_lbl}: the COMMIT owner reads {_sp['commit_read']!r}, arm says {_want!r}"
        assert _sp["advance_read"] == ARMS[_lbl]["loop"]["read"], \
            f"{_lbl}: the ADVANCE owner reads the wrong series"
        # the commit owner's own read is on the LOOP TRACE in every cycle it could speak,
        # under its own key and never under the advance owner's.
        _lg = _r["log"]["loop"]
        _spoke = [q for q in _lg if q.get("c_quiet") is not None]
        _X[f"X-1 {_lbl}"] = {**_sp, "n_cycles": len(_lg),
                             "n_cycles_commit_owner_spoke": len(_spoke),
                             "n_cycles_gauge_absent": sum(
                                 1 for q in _lg if q.get("c_skipped"))}
        assert all(q.get("c_key") == _want for q in _lg if q.get("c_key")), \
            f"{_lbl}: the commit owner's trace carries the wrong key"

    # X-2  INERTNESS. With `loop_commit` absent the arm is the donor: `tu_pf_off` must carry no
    #      split record, no `c_*` trace columns, and must be bit-identical to `dsil_yield`'s
    #      shape of decision (same driven key, same policy kind).
    if _ran("tu_pf_off"):
        _r = json.load(open(f"{outdir}/tu_pf_off/results.json"))
        assert _r.get("split") is None, "tu_pf_off built a second policy with no loop_commit"
        assert _r.get("q_mode") is None, "tu_pf_off ran the question port with no mode"
        _lg = _r["log"]["loop"]
        assert not any("c_quiet" in q or "c_key" in q for q in _lg), \
            "tu_pf_off's loop trace carries the split's columns — the branch is not inert"
        _X["X-2 inertness"] = {"split": None, "q_mode": None, "n_cycles": len(_lg),
                               "driven_read": (_lg[0].get("read") if _lg else None)}

    # X-3  THE BOOTSTRAP UNDER A SPLIT — and the configuration that has never run: bootstrapped
    #      commits ALONGSIDE yield-driven era advances, in the same era.
    if _ran("tu_pf_boot"):
        _r = json.load(open(f"{outdir}/tu_pf_boot/results.json"))
        _ev = [e for e in (_r.get("dsil_events") or []) if e["kind"] == "bootstrap"]
        _pn = _r["log"]["panel"]
        _first = next((q["cycle"] for q in _pn if q.get("dsil") is not None), None)
        _adv = _acts(_r, PO.ADVANCE)
        _X["X-3 split bootstrap"] = {
            "n_bootstrap_commits": len(_ev),
            "at": [(e["level"], e["cycle"], e["why"]) for e in _ev],
            "first_cycle_with_dsil": _first,
            "advances": [(a["cycle"], a.get("why")) for a in _adv],
            "advances_before_gauge": [a["cycle"] for a in _adv
                                      if _first is None or a["cycle"] < _first]}
        assert _ev, ("tu_pf_boot: the split's bootstrap never fired, so a dsil-COMMIT arm "
                     "deadlocks exactly as `dsil_read` would without it")
        assert all(_first is None or e["cycle"] < _first for e in _ev), \
            "tu_pf_boot: a bootstrap commit was taken while the gauge WAS available"

    # X-4  PRECEDENCE, and the simultaneous re-arm. A cycle can carry at most one loop action,
    #      and where both latches were quiet the action taken must be the COMMIT.
    for _lbl in ("tu_pf_split", "tu_pf_boot", "tu_pf_q"):
        if not _ran(_lbl):
            continue
        _r = json.load(open(f"{outdir}/{_lbl}/results.json"))
        _by, _byq = {}, {}
        for a in (_r.get("loop_actions") or []):
            _by.setdefault(int(a["cycle"]), []).append(a["kind"])
            # PRECEDENCE is a statement about QUIET READINGS: "a quiet reading commits if
            # there is something to commit and advances otherwise". A CAP-FORCED advance is
            # not a quiet reading — it is the era ending — and the donor takes one on the same
            # cycle as a boundary commit whenever the two coincide (which at preflight's
            # six-cycle eras is common). So the assertion is over the gauge-driven actions and
            # the cap co-occurrence is REPORTED beside it rather than asserted away.
            if a.get("why") not in ("cap", "clock"):
                _byq.setdefault(int(a["cycle"]), []).append(a["kind"])
        _both = {c: k for c, k in _byq.items() if len(k) > 1}
        _cap_co = {c: k for c, k in _by.items() if len(k) > 1 and c not in _both}
        _lg = {int(q["cycle"]): q for q in _r["log"]["loop"] if q.get("cycle")}
        _dual = [c for c, q in _lg.items()
                 if q.get("quiet") and q.get("c_quiet") and c in _by]
        _X[f"X-4 {_lbl}"] = {"cycles_with_two_gauge_actions": _both,
                             "cycles_commit_plus_capped_advance": _cap_co,
                             "cycles_both_latches_quiet": sorted(_dual),
                             "action_taken_there": {c: _by[c] for c in sorted(_dual)}}
        assert not _both, (f"{_lbl}: cycle(s) {sorted(_both)} carry two GAUGE-DRIVEN loop "
                           f"actions — the donor's precedence (commit-or-advance, never both) "
                           f"is broken")
        # PRECEDENCE, in the donor's own words: "a quiet reading COMMITS the active level IF
        # THERE IS ONE TO COMMIT, and ADVANCES otherwise." A both-quiet cycle that advanced is
        # a violation only where a commit was AVAILABLE — i.e. the era's active level did not
        # already carry a table. (The reduction's S5T check carries the same correction.)
        _pan = {int(q["cycle"]): q for q in _r["log"]["panel"]}
        _cev = [e for e in _r.get("events", []) if e.get("kind") == "commit"]
        for c in _dual:
            if _by[c][0] == PO.COMMIT:
                continue
            _act = int((_pan.get(c) or {}).get("active") or 0)
            _prior = any(int(e["level"]) == _act and int(e["cycle"]) < c for e in _cev)
            _X[f"X-4 {_lbl}"].setdefault("both_quiet_already_committed", {})[c] = _prior
            assert _prior, (
                f"{_lbl}: at c{c} BOTH latches were quiet, a commit WAS available at level "
                f"{_act}, and the action taken was {_by[c]} — precedence says COMMIT wins")

    # X-5  THE TWO-POLICY YOKE replays its source's realised COMMIT and ADVANCE cycles exactly,
    #      and builds no second policy of its own (a yoke reads nothing).
    if _ran("tu_pf_yk") and _ran("tu_pf_boot"):
        _ry = json.load(open(f"{outdir}/tu_pf_yk/results.json"))
        _rs = json.load(open(f"{outdir}/tu_pf_boot/results.json"))
        assert _ry.get("split") is None, "the yoke built a second policy — it reads nothing"
        _pc = lambda r, k: sorted(a["cycle"] for a in _acts(r, k))
        _X["X-5 two-policy yoke"] = {
            "source_commits": _pc(_rs, PO.COMMIT), "yoke_commits": _pc(_ry, PO.COMMIT),
            "source_advances": _pc(_rs, PO.ADVANCE), "yoke_advances": _pc(_ry, PO.ADVANCE)}
        assert _pc(_ry, PO.COMMIT) == _pc(_rs, PO.COMMIT), \
            f"the yoke did not replay its source's COMMIT cycles: {_X['X-5 two-policy yoke']}"
        assert _pc(_ry, PO.ADVANCE) == _pc(_rs, PO.ADVANCE), \
            f"the yoke did not replay its source's ADVANCE cycles: {_X['X-5 two-policy yoke']}"

    # X-6  THE PORT RAN, and the two selectors did different things. The offline suite
    #      (Q-1..Q-10) proves the rules; Q-11 proves the wiring; this proves the ARM carries it.
    for _lbl in ("tu_pf_q", "tu_pf_split", "tu_s_endo", "tu_y_exo"):
        if not _ran(_lbl):
            continue
        _r = json.load(open(f"{outdir}/{_lbl}/results.json"))
        _q = [q for q in _r["log"]["q"] if q]
        _want = ARMS[_lbl]["cfg"].get("question_mode")
        assert _r.get("q_mode") == _want, \
            f"{_lbl}: q_mode {_r.get('q_mode')!r} != arm's {_want!r}"
        assert len(_q) == len(_r["log"]["cycle"]), \
            f"{_lbl}: the question row is missing on {len(_r['log']['cycle']) - len(_q)} cycles"
        _X[f"X-6 {_lbl}"] = {
            "mode": _r.get("q_mode"), "k": _r.get("q_k"),
            "quota_ok_frac": round(sum(1 for q in _q if q["quota_ok"]) / max(len(_q), 1), 4),
            "d_mean": round(float(np.mean([q["d_mean"] for q in _q])), 4),
            "d_mean_menu": round(float(np.mean([q["d_mean_menu"] for q in _q])), 4),
            "n_distinct_needy": [q["n_distinct_needy"] for q in _q[:4]],
            "has_clean": [q["has_clean"] for q in _q[:4]],
            # the dose instrument, on a FALLIBLE executor — reported at every scale it can be
            # read, because the full-scale read wants a shape to compare to.
            "dose_hit": [q.get("dose_hit") for q in _q[:8]],
            "n_mined": [q.get("n_mined") for q in _q[:8]],
            "ledger": _r.get("q_ledger")}
        assert all(q["quota_ok"] for q in _q), \
            f"{_lbl}: the difficulty quota was missed — selection moved the difficulty"
    ok["tutti_gate_X"] = _X

    # ---- [assay] the surgery and the instrument must have EXECUTED, not merely not crashed
    # [crescendo] scoped to the DONOR'S levels (2..3). No surgery arm runs in A3's tag, the
    # arc's junk pool carries no level-4 keys (`cs_s0`/`sp_s0` ran at `max_macro_level=3`, so
    # their logs hold no level-4 miner), and a `junk_dose` dose that cannot be filled at L4 is
    # a fact about the pool, not a fault in this fork. Levels 2-3 are asserted exactly as the
    # donor asserted them; what happens at L4 is recorded and not asserted.
    _surg_levels = range(2, min(cfg["max_macro_level"], 3) + 1)
    _surg_modes = [m for m in ("complete", "exact", "junk_dose", "strip") if _ran(m)]
    for mode in _surg_modes:
        rj = json.load(open(f"{outdir}/{mode}/results.json"))
        sv = [e for e in rj["events"] if e["kind"] == "surgery"]
        assert sv, f"{mode}: no surgery event"
        assert {e["level"] for e in sv} >= set(_surg_levels), \
            f"{mode}: surgery did not fire at every level ({[e['level'] for e in sv]})"
        for e in sv:
            assert e["mode"] == mode, f"{mode}: wrong mode recorded"
        ok[f"surgery:{mode}"] = [
            {k: e.get(k) for k in ("level", "cycle", "n_own", "n_own_true", "n_added",
                                   "n_added_true", "n_removed", "K", "n_after",
                                   "tab_recall", "tab_precision", "dose_shortfall")}
            for e in sv]
    # each mode must do what its name says
    def _sv(mode, lv):
        rj = json.load(open(f"{outdir}/{mode}/results.json"))
        return next(e for e in rj["events"] if e["kind"] == "surgery" and e["level"] == lv)
    for lv in (_surg_levels if len(_surg_modes) == 4 else ()):
        st_, cp_, ex_, jd_ = (_sv(m_, lv) for m_ in ("strip", "complete", "exact",
                                                     "junk_dose"))
        assert st_["n_added"] == 0, "strip added entries"
        assert st_["tab_precision"] in (None, 1.0), \
            f"strip L{lv} left junk behind: precision {st_['tab_precision']}"
        assert cp_["n_added"] == cp_["n_added_true"], "complete added a non-true entry"
        assert ex_["n_after"] == int(shared["truth"][lv]["child"].shape[0]), \
            "exact did not install the full true table"
        assert jd_["n_added_true"] == 0, "junk_dose added a TRUE entry"
        assert jd_["K"] == cp_["K"] or lv > 2, \
            f"the dose is not matched at L{lv}: {jd_['K']} vs {cp_['K']}"
        ok[f"dose_match_L{lv}"] = {"complete_K": cp_["K"], "junk_dose_added": jd_["n_added"],
                                   "shortfall": jd_.get("dose_shortfall")}
    # `exact` and `given_c1` must hold IDENTICAL content and differ only in arrival
    if _ran("exact") and _ran("given_c1"):
        ex_end = json.load(open(f"{outdir}/exact/results.json"))["log"]["committed_grade"][-1]
        g1_end = json.load(open(f"{outdir}/given_c1/results.json"))["log"]["committed_grade"][-1]
        assert ex_end == g1_end, f"exact and given_c1 differ in content: {ex_end} vs {g1_end}"
        ok["exact_vs_given_c1_content"] = ex_end

    # THE INSTRUMENT: entry selections must be recorded, at both levels, in the beam phase
    # [crescendo] read off whichever schedule arm ran, so a targeted re-run still checks it.
    _anc = "anchor" if _ran("anchor") else ("anchor_long" if _ran("anchor_long") else None)
    # [intonation] guarded: the schedule arms are donor shapes, and a delta_perf-only
    # `--arms` subset carries neither. The instrument and twin checks below read one of
    # them, so they are skipped (and SAID SO) rather than crashing a targeted re-run.
    if _anc is None:
        ok["entry_instrument"] = "skipped - no schedule arm in --arms"
        ok["twins_with_instrument"] = "skipped - no schedule arm in --arms"
    else:
        an = json.load(open(f"{outdir}/{_anc}/results.json"))
        ent = [q for q in an["log"]["entry"] if (q.get("hist") or {}).get("beam")]
        assert ent, "the entry-identity instrument recorded nothing in the beam phase"
        lvs = sorted({k for q in ent for k in q["hist"]["beam"]})
        assert set(lvs) & {"2"}, f"no L2 entry selections recorded (levels seen: {lvs})"
        assert all("true_mask" in q for q in ent), "the truth mask did not reach the log"
        ok["entry_instrument"] = {
            "cycles_with_beam_selections": len(ent), "levels": lvs,
            "total_beam_selections": int(sum(sum(v) for q in ent
                                             for v in q["hist"]["beam"].values())),
            "phases": sorted({p for q in an["log"]["entry"] for p in (q.get("hist") or {})})}

        # THE TWIN CLAIM WITH THE INSTRUMENT ON: a treated arm must be bit-identical to the anchor
        # up to its first surgery. This is the assertion the SPEC asks for by name.
        SER_ = ["e", "succ", "dres", "t_cum", "n_moves", "width", "g_per_solve", "e_practice",
                "vloss", "gloss", "n_solved", "n_mined", "m_per_solve"]
        ok["twins_with_instrument"] = {}
        for mode in _surg_modes:
            rj = json.load(open(f"{outdir}/{mode}/results.json"))
            c1_ = min(e["cycle"] for e in rj["events"] if e["kind"] == "surgery")
            w = c1_ - 1
            d = 0.0
            for k in SER_:
                x = np.asarray(an["log"][k][:w], float)
                y = np.asarray(rj["log"][k][:w], float)
                q = min(len(x), len(y))
                d = max(d, float(np.abs(x[:q] - y[:q]).max()) if q else 0.0)
            ok["twins_with_instrument"][mode] = {"first_surgery": c1_, "window": w,
                                                 "max_abs_delta": d}
            assert d == 0.0, (f"{mode} diverges from the anchor BEFORE its first surgery "
                              f"(c1-c{w}, max|delta| = {d:.3e}) — the instrument or the surgery "
                              f"is not behaviour-neutral")
    # ---- [conductor] THE LOOP'S OWN IN-TAG ASSERTIONS ----------------------------------
    # The donor's stream-burn check is dropped here (its twins `given_c1_j`/`exact_j` are
    # `assay`'s arms and this round does not run them); what replaces it is the pair of
    # assertions this round actually rests on.
    #
    #   (i)  THE READS ARE NON-INVASIVE. Every loop arm shares the anchor's stream, so it must
    #        be BIT-IDENTICAL to the anchor up to its OWN first action. A divergence before
    #        then would mean the shadow panel, the observation panel or the endo forward is
    #        perturbing the trajectory, which would void every comparison in the round.
    #   (ii) THE YOKE REPLAYS EXACTLY. A yoked arm must reproduce its gauge arm's realised
    #        actions cycle for cycle. At preflight scale this is a mechanic check, not the
    #        finding — the finding is what `analyze_conductor.py` §2 measures at full scale.
    ok["loop_twins"] = {}
    an = (json.load(open(f"{outdir}/{_anc}/results.json")) if _anc else None)
    # [crescendo] the donor's five loop arms PLUS this round's four, each against the schedule
    # arm that shares its stream. The A3 arms are the ones that matter here: the check says the
    # level-4 miner, the widened panel and `commit_max_level` are all non-invasive before the
    # arm's own first action.
    for arm_ in ([x for x in ("outer_yield", "yoked_yield", "outer_endo", "yoked_endo",
                              "outer_ledger", "outer_yield_m4", "ceiling_m3",
                              "outer_yield_m4x", "outer_yield_m3") if _ran(x)]
                 if an is not None else []):
        rj = json.load(open(f"{outdir}/{arm_}/results.json"))
        acts = [a["cycle"] for a in (rj.get("loop_actions") or [])]
        w = (min(acts) - 1) if acts else len(rj["log"]["cycle"])
        d = 0.0
        for k in [q for q in SER_ if q != "t_cum"]:       # t_cum carries the priced ledger
            x = np.asarray(an["log"][k][:w], float)
            y = np.asarray(rj["log"][k][:w], float)
            q = min(len(x), len(y))
            d = max(d, float(np.abs(x[:q] - y[:q]).max()) if q else 0.0)
        ok["loop_twins"][arm_] = {"first_action": (min(acts) if acts else None),
                                  "window": w, "max_abs_delta": d,
                                  "spend_g": (rj.get("ledger") or {}).get("spend_g")}
        assert d == 0.0, (f"{arm_} diverges from the anchor BEFORE its first action "
                          f"(c1-c{w}, max|delta| = {d:.3e}) — a read is not inert")
    ok["yoke_replay"] = {}
    # [crescendo] `ceiling_m3` is a yoke of `outer_yield_m4` AND carries `commit_max_level=3`,
    # so it replays every ADVANCE but refuses the L4 COMMIT by design. It is therefore checked
    # by gate C-2/C-5 (below/above) rather than by the donor's exact-actions assertion, which
    # would be false for it whenever the treatment reaches the rung.
    for yk_, src_ in [(y, x) for y, x in (("yoked_yield", "outer_yield"),
                                          ("yoked_endo", "outer_endo"))
                      if _ran(y) and _ran(x)]:
        a_ = json.load(open(f"{outdir}/{src_}/results.json"))
        b_ = json.load(open(f"{outdir}/{yk_}/results.json"))
        pa = [(x["kind"], x["cycle"], x.get("level")) for x in (a_.get("loop_actions") or [])
              if not x.get("cancelled")]
        pb = [(x["kind"], x["cycle"], x.get("level")) for x in (b_.get("loop_actions") or [])]
        dd = 0.0
        n_ = min(len(a_["log"]["cycle"]), len(b_["log"]["cycle"]))
        for k in [q for q in SER_ if q != "t_cum"]:
            dd = max(dd, float(np.abs(np.asarray(a_["log"][k][:n_], float)
                                      - np.asarray(b_["log"][k][:n_], float)).max()))
        ok["yoke_replay"][yk_] = {"of": src_, "gauge_actions": pa, "yoke_actions": pb,
                                  "actions_equal": pa == pb, "n_cycles": n_,
                                  "max_abs_delta": dd,
                                  "t_cum_gauge": a_["log"]["t_cum"][-1],
                                  "t_cum_yoke": b_["log"]["t_cum"][-1]}
        assert pa == pb, f"{yk_} did not replay {src_}'s actions: {pb} != {pa}"
    ok["gy_observed"] = {a_: json.load(open(f"{outdir}/{a_}/results.json"))["gy_final"]["n_obs"]
                         for a_ in ("anchor", "outer_yield", "outer_endo", "outer_ledger",
                                    "anchor_long", "outer_yield_m4") if _ran(a_)}
    # [tacet] `if _ran(a_)` added — the donor's line was the one place in `preflight` that
    # assumed the FULL arm sweep, so any `--arms` subset omitting `outer_yield`/`outer_endo`/
    # `outer_ledger` crashed here AFTER every gate had already passed. (Caught in flight on this
    # node's first preflight, which runs a 13-arm subset.) The sibling line above was already
    # guarded; this one was not.
    ok["obs_gy_agree"] = {a_: json.load(open(f"{outdir}/{a_}/results.json"))["obs_gy_agree"]
                          for a_ in ("anchor", "outer_yield", "outer_endo", "outer_ledger",
                                     "outer_yield_m4") if _ran(a_)}
    print("\n=== PREFLIGHT OK — every call resolved at the depth-6 grammar ===")
    print(json.dumps({k: q for k, q in ok.items() if k != "_spiral_shared"},
                     indent=2, cls=NumpyEncoder))
    print(f"shared keys: {ok['_spiral_shared']}")
    return ok


@app.function(image=image, volumes={DATA_DIR: volume}, gpu="L4", timeout=5400, memory=32768)
def fidelity_d4(tag: str = "fid_d4", cycles: int = 5, arms: str = "given,practice_late",
                seed: int = 0):
    """G-F — THE DONOR-FIDELITY GATE. With the depth-6 transplant switched OFF (depth 4, m=2,
    the donor's own budget and ladder), does THIS file's `run_arm` reproduce
    `native/full/full.py`'s, bit for bit?

    Run IN PROCESS against the imported donor rather than by diffing two volume tags
    (`critic/fidelity_check.py`'s cross-tag form), for two reasons: it needs no prior donor tag
    to exist at a matched config, and it removes cross-run hardware from the comparison so a
    failure is unambiguously the fork's. The donor's `run_arm` is called with the shared dict
    THIS file built, so the substrate is literally the same object.

    A SELF-REPLAY CONTROL is carried: the donor is run twice, and its own arm-to-arm delta is
    reported beside the fork's. Without it a non-zero delta is uninterpretable — GPU
    nondeterminism and a real fork bug look identical."""
    import torch
    from rhm.practice.native.full import full as FL

    cfg = _cfg(seed=seed, era_cycles=cycles, budget=4, g_budget=58, max_macro_level=3,
               mine_from="chosen", mine_cap=8, probe_every=2, probe_widths=(1, 2, 4),
               controller_steps=800, generator_steps=800, value_steps=800, reader_steps=600,
               value_episodes=6_000, n_train_episodes=20_000, n_pr=24, n_rt=96, n_score=96,
               n_probe_clean=128, checkpoint_every=10 ** 9, gen_steps=5,
               sil_min_cycle=2, early_offset=1, late_offset=1,
               prop_warmup=1, prop_steps=12, prop_buf_cap=20_000,
               span_min_hold=32, span_tau=0.0)
    ers = parse_eras("1:6,2:3,3:1")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg["train_seed"]); np.random.seed(cfg["train_seed"])
    torch.set_float32_matmul_precision("high")
    started = time.time()
    print(f"[fidelity_d4] depth={cfg['depth']} m={cfg['m']} arms={arms} cycles={cycles}",
          flush=True)

    # the donor's own builder, so even the SUBSTRATE is not this fork's work
    shared = FL.build_shared(cfg, device)
    shared["probe_clean"] = torch.from_numpy(
        _sample_pool(shared["rules"], cfg["n_probe_clean"], cfg["s"], cfg["seed"] + 31337)[1])
    refs = FL.measure_refs(shared, cfg, ers, device)
    outdir = f"{DATA_DIR}/{REMOTE}/{tag}"
    os.makedirs(outdir, exist_ok=True)

    SERIES = ["e", "succ", "dres", "t_cum", "n_moves", "width", "g_per_solve", "e_practice",
              "vloss", "gloss", "n_solved", "n_mined", "m_per_solve"]
    rows, worst_fork, worst_ctrl = [], 0.0, 0.0
    for arm in [a.strip() for a in arms.split(",") if a.strip()]:
        a1 = FL.run_arm(arm, arm, {}, shared, cfg, ers, refs, f"{outdir}/donor1", device)
        a2 = FL.run_arm(arm, arm, {}, shared, cfg, ers, refs, f"{outdir}/donor2", device)
        b = run_arm(arm, arm, {}, shared, cfg, ers, refs, f"{outdir}/fork", device)
        row = {"arm": arm, "fork": {}, "control": {}}
        for k in SERIES:
            x, y, z = (np.asarray(a1["log"][k], float), np.asarray(b["log"][k], float),
                       np.asarray(a2["log"][k], float))
            L = min(len(x), len(y), len(z))
            row["fork"][k] = float(np.abs(x[:L] - y[:L]).max()) if L else float("nan")
            row["control"][k] = float(np.abs(x[:L] - z[:L]).max()) if L else float("nan")
            worst_fork = max(worst_fork, row["fork"][k])
            worst_ctrl = max(worst_ctrl, row["control"][k])
        ev = lambda r: [(e["era"], e["level"], e["cycle"], e["n_entries"])
                        for e in r["events"] if e["kind"] == "commit"]
        row["events_equal"] = bool(ev(a1) == ev(b))
        row["n_cycles"] = len(a1["log"]["e"])
        rows.append(row)
        print(f"[fidelity_d4] {arm}: max|fork-donor| = {max(row['fork'].values()):.3e}  "
              f"max|donor-donor| = {max(row['control'].values()):.3e}  "
              f"commit events equal: {row['events_equal']}", flush=True)

    out = {"rows": rows, "worst_fork": worst_fork, "worst_donor_self_replay": worst_ctrl,
           "events_equal": all(r["events_equal"] for r in rows),
           "seconds": time.time() - started}
    with open(f"{outdir}/gate.json", "w") as fh:
        json.dump(out, fh, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"\n[fidelity_d4] === G-F ===  max|fork - donor| = {worst_fork:.3e}   "
          f"max|donor - donor| (control) = {worst_ctrl:.3e}", flush=True)
    assert out["events_equal"], "G-F FAILED: commit events differ from the donor"
    assert worst_fork <= worst_ctrl, (
        f"G-F FAILED: the fork perturbed the donor's code path "
        f"({worst_fork:.3e} > self-replay control {worst_ctrl:.3e})")
    print("G-F PASS — with the transplant off, the fork is a replay of the donor.")
    return out




def q_interface_check(shared, cfg, ers, device):
    """[tutti] GATE Q-11 — every selector, every era, on the REAL substrate, at toy sizes.

    The offline suite (`questions.question_gate`) checks the rules on synthetic menus; this
    checks the wiring: the reader forward, the value forward, the exact-feature parse, the
    operative-lower-table lookup and the quota, for all six modes across all five era cells.
    It runs inside `fidelity_smoke`, where the substrate has already been paid for, so every
    new branch is executed before any main run — `preflight`'s discipline at no extra cost.
    """
    import torch
    v, s_, depth, m = cfg["v"], cfg["s"], cfg["depth"], cfg["m"]
    maxl = cfg["max_macro_level"]
    rules = shared["rules"]
    miners = {ell: MC.Miner(ell, s_) for ell in range(2, maxl + 1)}

    def operative(ell):
        if ell == 1:
            return MC.base_table(v)
        return miners[ell].build(operative(ell - 1), cfg["mine_support"])

    # a quarter of each true table banked, so `banked` and `lower_flat` are non-trivial and the
    # oracle's "already at support" and "buildable" branches both execute.
    for ell in range(2, maxl + 1):
        flat = shared["truth"][ell]["flat"]
        k_ = max(1, len(flat) // 4)
        miners[ell].observe(np.concatenate([flat[:k_]] * (cfg["mine_support"] + 1)))
    value_net = copy.deepcopy(shared["value0"]).to(device)
    out, k_menu = {}, max(4 * cfg["n_pr"], 128)
    for era in ers:
        head = context_instances(rules, era_ctx(era), cfg["n_pr"], s_, depth, v, m,
                                 seed=cfg["seed"] + 12_345, with_clean=True)
        for mode in ("exo", "bisect", "endo", "novel", "comp", "comp_free"):
            led = QS.DeliveryLedger() if mode == "endo" else None
            r_, x_, c_, des, hal, row = pose_questions(
                mode, rules=rules, era=era, cfg=cfg, s=s_, depth=depth, v=v, m=m, cyc=1,
                head=head, shared=shared, value_net=value_net,
                controller=shared["controller"], device=device, miners=miners,
                operative=operative, maxl=maxl, qledger=led, n_pr=cfg["n_pr"], k_menu=k_menu)
            assert r_.shape[0] == x_.shape[0] == c_.shape[0] == des.shape[0] == cfg["n_pr"], \
                f"{era['name']}/{mode}: selector returned the wrong volume"
            assert row["quota_ok"] or mode == "comp_free", \
                f"{era['name']}/{mode}: difficulty quota missed — {row['d_hist']} vs " \
                f"{row['quota']}"
            if mode == "exo":
                assert np.array_equal(r_, head[0]) and np.array_equal(x_, head[1]), \
                    "exo did not return the donor's own head unchanged"
            out[f"{era['name']}/{mode}"] = {
                "quota_ok": bool(row["quota_ok"]), "d_mean": row["d_mean"],
                "n_distinct_designed": row["n_distinct_designed"],
                "n_designed_true": row["n_designed_true"],
                "n_designed_banked": row["n_designed_banked"],
                "n_distinct_halves": row["n_distinct_halves"],
                "n_distinct_needy": row["n_distinct_needy"],
                "target_level": row["target_level"], "has_clean": row["has_clean"]}
    # The ceiling must out-target the exogenous draw on the statistic it optimises: DISTINCT
    # true keys not yet at support. Asserted only where Phase 0 says there IS headroom — at L2
    # the whole 14-key table is reachable by a uniform draw (reach 14/14 at every menu size), so
    # the cell is reported and not asserted; the assertion lives at L3 and L4, where the reach
    # curve is 40/56 and 2/816 at K=256 and the tail is what selection buys.
    for era in ers:
        b, e = out[f"{era['name']}/bisect"], out[f"{era['name']}/exo"]
        tag_ = f"{era['name']} -> L{b['target_level']}"
        print(f"[q] Q-11 {tag_}: distinct needy true keys aimed at — "
              f"exo {e['n_distinct_needy']}  bisect {b['n_distinct_needy']}", flush=True)
        if int(b["target_level"]) >= 3:
            assert b["n_distinct_needy"] >= e["n_distinct_needy"], (
                f"{tag_}: the oracle aims at no more distinct needy true keys "
                f"({b['n_distinct_needy']}) than the head does ({e['n_distinct_needy']})")
    print(f"[q] Q-11 in-substrate interface check PASS ({len(out)} cells)", flush=True)
    return out


@app.function(image=image, volumes={DATA_DIR: volume}, gpu="L4", timeout=7200, memory=32768)
def fidelity_smoke(tag: str = "gf_smoke", cycles: int = 4, seed: int = 0):
    """G-F, layer 1 — WITH THE SURGERY OFF, does this file's `run_arm` reproduce
    `census.py`'s, bit for bit, on the depth-6 substrate, WITH the entry instrument installed?

    In process against the imported donor (`critic/fidelity_check.py`'s idiom, as used for the
    spiral's own G-F): the donor's `run_arm` is called with the shared dict THIS file built, so
    the substrate is literally the same object and no cross-run hardware enters the comparison.
    A DONOR SELF-REPLAY control is carried, because without it a non-zero delta is
    uninterpretable — GPU nondeterminism and a real fork bug look identical.

    SCALE, on the record: this runs at SMOKE sizes (a few hundred training steps, `cycles`
    cycles), not at `sp_s0`'s configuration. A full-scale in-process replay would mean paying
    for three 116-cycle depth-6 arms plus a ~630 s setup — about 1.5 GPU-h to re-derive a
    number the in-tag anchor gives away free. The FULL-SCALE half of G-F is therefore the
    cross-tag one, in `analyze_census.py`: this round's in-tag `spiral_route` against
    `sp_s0`'s, over the 32 cycles the two ladders share. Era 1 is 48 cycles here against 32
    there, but the only things `ec` reaches are the loop bound, `at_boundary` (c18's certified
    commit precedes both), and the era-end probe trigger (which fires at c32 either way, and
    probes consume neither the shared torch stream nor any arm RNG). So c1-c32 is a genuine
    bit-for-bit window at the real configuration, for free.
    """
    import torch
    # [tacet] retargeted to THIS fork's direct donor, `crescendo.py` (which in its own turn
    # replays `maestro.py` at 0.000e+00, which replays `conductor.py`, which replays
    # `assay.py`, so the chain back to the substrate is intact). The gate runs with
    # `gate_mode` unset, which is the exact configuration in which this file IS its donor: the
    # two new `out` keys are unread, `prop_pairs` takes its `keep=None` branch, the mining
    # subsample draws the identical `rng.permutation`, and the only other addition is one
    # unread per-cycle log list. If this is not 0.000e+00 the fork has drifted and no gate
    # contrast means anything.
    #
    # `gate_log` is left ON for the gate, deliberately: the per-cycle feature record runs in
    # every arm of the main run, so the thing being certified inert is the file AS IT WILL RUN,
    # not a stripped version of it.
    # [intonation] retargeted one more link along the chain: THIS fork's direct donor is
    # `tacet.py` (which replays `crescendo.py` at 0.000e+00, which replays `maestro.py`, which
    # replays `conductor.py`, which replays `assay.py`). The gate runs with EVERY
    # `# [intonation]` knob off — `perf_meter` False, `span_tau_fire` None, `perf_gain` None,
    # `gate_mode` unset — which is the exact configuration in which this file IS `tacet.py`:
    # `PerfExecutor` is never constructed, `beam_moves*`'s `track` is False on a plain or plain
    # span executor, `finetune_generator_span` calls `SN.span_train_terms` itself, the parity
    # gate's `tau_fire` resolves to `span_tau`, and the only remaining addition is one unread
    # per-cycle log list. `gate_log` stays ON, as in `tacet`'s own G-F, so what is certified
    # inert is the file AS IT WILL RUN.
    # [tutti] retargeted one more link along the chain: THIS fork's direct donor is
    # `caesura.py` (which replays `intonation.py` at 0.000e+00, which replays `tacet.py`, which
    # replays `crescendo.py`, `maestro.py`, `conductor.py`, `assay.py`). The gate runs with
    # EVERY `# [tutti]` knob off — `question_mode` None and `loop_commit` absent on both arms —
    # which is the exact configuration in which this file IS `caesura.py`: `pose_questions` is
    # never called, `context_instances` is asked for `with_clean=True` and the third return is
    # discarded into `cl_np` (the donor computes `clean` either way, so no work and no RNG
    # moves), `loop_c` is None so `_acted_all` is `loop.acted` and the commit dispatch reads
    # `loop.quiet`, and the only remaining additions are two unread per-cycle log lists.
    from rhm.practice.caesura import caesura as SP

    cfg = _d6_cfg(era_cycles=cycles, seed=seed, probe_every=2, probe_widths=(1, 2),
                  controller_steps=800, generator_steps=800, value_steps=800, reader_steps=600,
                  value_episodes=6_000, n_train_episodes=20_000, n_pr=24, n_rt=96, n_score=96,
                  n_aud=48, n_probe_clean=128, checkpoint_every=10 ** 9, gen_steps=5,
                  sil_min_cycle=2, prop_warmup=1, prop_steps=12, prop_buf_cap=20_000,
                  # [tacet] the G-F runs at `max_macro_level=4`, not the donor's 3. A3 had to
                  # run its own G-F at 3 because `maestro.py` had no L4 path to compare
                  # against; `crescendo.py` does, so the gate here can exercise the exact
                  # configuration the main run uses — L4 miner live, panel widened, four-level
                  # slot layout — rather than a configuration one rung short of it.
                  max_macro_level=4,
                  tm_episodes=512, mine_cap=8)
    ers = parse_eras("1:25,2:12,3:6")
    for e_ in ers:
        e_["cycles"] = cycles
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg["train_seed"]); np.random.seed(cfg["train_seed"])
    torch.set_float32_matmul_precision("high")
    started = time.time()
    print(f"[gf] depth={cfg['depth']} m={cfg['m']} cycles/era={cycles} maxl={cfg['max_macro_level']} "
          f"— tutti vs caesura (gate_log ON; every [tutti] knob off)",
          flush=True)

    _install_identity_miner(cfg["mine_support"])
    # [assay] the instrument is INSTALLED for this gate. It patches `MC.macro_features`, which
    # the donor's own code path also calls, so both sides of the comparison run through the
    # recording copy — which is the point: the gate must show the instrument is neutral.
    _install_entry_recorder()
    shared = _spiral_shared(cfg, device, ers)
    refs = measure_refs(shared, cfg, ers, device)
    outdir = f"{DATA_DIR}/{REMOTE}/{tag}"
    os.makedirs(outdir, exist_ok=True)

    SERIES = ["e", "succ", "dres", "t_cum", "n_moves", "width", "g_per_solve", "e_practice",
              "vloss", "gloss", "n_solved", "n_mined", "m_per_solve"]
    rows, worst_fork, worst_ctrl = [], 0.0, 0.0
    for arm in ("anchor", "given_c1"):
        a1 = SP.run_arm(arm, arm, {}, shared, cfg, ers, refs, f"{outdir}/donor1", device)
        a2 = SP.run_arm(arm, arm, {}, shared, cfg, ers, refs, f"{outdir}/donor2", device)
        b = run_arm(arm, arm, {}, shared, cfg, ers, refs, f"{outdir}/fork", device)
        row = {"arm": arm, "fork": {}, "control": {}}
        for k in SERIES:
            x, y, z = (np.asarray(a1["log"][k], float), np.asarray(b["log"][k], float),
                       np.asarray(a2["log"][k], float))
            L = min(len(x), len(y), len(z))
            row["fork"][k] = float(np.abs(x[:L] - y[:L]).max()) if L else float("nan")
            row["control"][k] = float(np.abs(x[:L] - z[:L]).max()) if L else float("nan")
            worst_fork = max(worst_fork, row["fork"][k])
            worst_ctrl = max(worst_ctrl, row["control"][k])
        ev = lambda r: [(e["era"], e["level"], e["cycle"], e["n_entries"])
                        for e in r["events"] if e["kind"] == "commit"]
        row["events_equal"] = bool(ev(a1) == ev(b))
        rows.append(row)
        print(f"[gf] {arm}: max|fork-donor| = {max(row['fork'].values()):.3e}  "
              f"max|donor-donor| = {max(row['control'].values()):.3e}  "
              f"commits equal: {row['events_equal']}", flush=True)

    out = {"scale": "smoke", "rows": rows, "worst_fork": worst_fork,
           "worst_donor_self_replay": worst_ctrl,
           "events_equal": all(r["events_equal"] for r in rows),
           "seconds": time.time() - started}
    with open(f"{outdir}/gate.json", "w") as fh:
        json.dump(out, fh, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"\n[gf] === G-F (smoke scale) === max|fork - caesura| = {worst_fork:.3e}   "
          f"control = {worst_ctrl:.3e}", flush=True)
    assert out["events_equal"], "G-F FAILED: commit events differ from caesura.py"
    assert worst_fork <= worst_ctrl, (
        f"G-F FAILED: the tutti fork perturbed caesura.py's code path "
        f"({worst_fork:.3e} > self-replay control {worst_ctrl:.3e}) — the question port's "
        f"`with_clean` draw, the split's second policy object, or one of the two new log rows "
        f"is not transparent")
    print("G-F PASS — with every [tutti] knob off, this fork replays caesura.py.")

    # [tutti] GATE Q-11, on the substrate this gate already paid for. The port's selectors are
    # `questions.py`'s, imported unchanged, so Q-1..Q-10 already hold offline; this is the
    # WIRING check, and it runs the oracle and both comfort poles too — modes this tag does not
    # pay an arm for — because the endogenous judge is scored against the oracle's own aiming
    # statistic in the reduction and that statistic has to be known to exist here.
    out["q11"] = q_interface_check(shared, cfg, ers, device)
    with open(f"{outdir}/gate.json", "w") as fh:
        json.dump(out, fh, indent=2, cls=NumpyEncoder)
    volume.commit()
    return out


# =========================================================================== #
# [spiral] PHASE A — the paid depth-6 feasibility block
# =========================================================================== #

def _slope(y):
    """least-squares slope per cycle over a series (the descent rate)."""
    y = np.asarray(y, float)
    if y.size < 2:
        return float("nan")
    x = np.arange(y.size, dtype=float)
    return float(np.polyfit(x, y, 1)[0])


def _descent(log, refs, commit_cycle=None):
    """`tall/dens0`'s descent-gate readout, plus the decomposition that mattered more than the
    headline there: 0.233 of its 0.262 recovered fraction was already present AT CYCLE 1,
    supplied by the densified value head rather than by the loop. So the recovered fraction is
    reported at c1, over the first five and the last five cycles, and the PRE-COMMIT slope is
    reported separately — that is the part the loop is actually responsible for."""
    e = np.asarray(log["e"], float)
    st, fl = float(refs["stale"][0]), float(refs["floor"][0])
    rng_ = max(st - fl, 1e-9)
    pre = e[:commit_cycle] if commit_cycle else e
    return {"stale": st, "floor": fl, "n_cycles": int(e.size),
            "e_c1": float(e[0]), "e_first5": float(e[:5].mean()),
            "e_last5": float(e[-5:].mean()), "e_min": float(e.min()),
            "rec_c1": float((st - e[0]) / rng_),
            "rec_first5": float((st - e[:5].mean()) / rng_),
            "rec_last5": float((st - e[-5:].mean()) / rng_),
            "rec_best": float((st - e.min()) / rng_),
            "slope_all": _slope(e), "slope_pre_commit": _slope(pre),
            "n_pre_commit": int(len(pre)),
            "delta_cycle_mean": float(np.abs(np.diff(e)).mean()) if e.size > 1 else float("nan"),
            "delta_cycle_max": float(np.abs(np.diff(e)).max()) if e.size > 1 else float("nan")}


def _pricing_rows(log, commit_cycle, window=3):
    """THE COMMIT-CYCLE PRICING READOUT. `dens0` measured a +0.180 jump in `e` at the cycle
    AFTER the L2 commit — 4.6x its own mean cycle-to-cycle |delta| of 0.019 — because the
    action set grew 32 -> 48 and `fit_width` then dropped the beam from width 2 to width 1 at
    G=482. Under top-k routing the width refits under k, not |ms|, so the same commit should
    cost nothing in width. The two exceptions worth watching, both visible here: the
    `prop_new_cycles` window in which newly committed slots are FORCED into the expansion
    (k_eff jumps by the number of new nodes), and the cycles before the filter switches on."""
    if not commit_cycle:
        return []
    lo, hi = max(0, commit_cycle - window - 1), min(len(log["cycle"]), commit_cycle + window)
    out = []
    for i in range(lo, hi):
        pr = log["prop"][i] if log.get("prop") else {}
        out.append({"cycle": int(log["cycle"][i]), "n_moves": int(log["n_moves"][i]),
                    "k_eff": pr.get("k_eff"), "filter_on": pr.get("filter_on"),
                    "width": int(log["width"][i]),
                    "g_per_solve": float(log["g_per_solve"][i]),
                    "e": float(log["e"][i])})
    return out


@app.function(image=image, volumes={DATA_DIR: volume}, gpu="L4", timeout=21600, memory=32768)
def phase_a(tag: str = "pa0",
            era: str = "1:25", cycles: int = 22, commit_at: int = 16,
            arms: str = "practice_late,practice_late_prop_k4,practice_late_native",
            fid_arms: str = "given,fid", fid_cycles: int = 5,
            budget: int = 8, g_budget: int = 482, max_macro_level: int = 3,
            n_corrupt: int = 1, mine_cap: int = 8, probe_every: int = 8,
            # n_score 512 -> 256: the shadow audition runs 5 macro applications per level per
            # cycle on this set and it is an unpriced INSTRUMENT, so it is the cheapest place
            # to buy back cycle time. `tall`'s post-mortem (probes were ~47% of `tl_s0`'s
            # runtime) is the reason this is a default rather than an afterthought.
            probe_widths: str = "1,2,4", n_pr: int = 64, n_rt: int = 384, n_score: int = 256,
            seed: int = 0, rule_seed: int = 0, train_seed: int = 1,
            task_matched: bool = True, tm_episodes: int = 8192):
    """PHASE A: the paid depth-6 block. One setup (~510 s), then everything that needs it.

    1. SETUP + REFERENCES at depth 6, m=2, `n_corrupt=1` — the densified stale value buffer is
       built by `build_shared` itself, so its terminal success is reported beside `dens0`'s
       0.133 and depth 4's 0.296 without any retraining step.
    2. THE DESCENT GATE WITH ROUTING LIVE. Era 1 only, `cycles` cycles, three arms sharing one
       torch stream: enumeration (the in-run reproduction of `dens0`'s −0.004/cycle reference),
       routing only, and both ports. Recovered fraction, its c1 decomposition, and the
       pre-commit slope — the part the loop is responsible for.
    3. THE COMMIT-CYCLE PRICING READOUT. The commit is placed at a FIXED cycle in every arm
       (`commit="late"` with `late_offset = cycles - commit_at`) rather than left to the
       certificate, precisely so the arms' pricing transitions are at the same cycle and the
       comparison is a comparison. The transplanted `delta_prov` policy is what the MAIN run
       uses; it is exercised by `preflight` and by the `spiral` arm shape, not here, because a
       certificate that fires at different cycles in different arms would confound the one
       number this block exists to read.
    4. THE IN-TAG FIDELITY TWINS AT DEPTH 6: `fid` (both ports wired, both nailed shut) against
       `given` (plain enumeration), same stream, `fid_cycles` cycles, asserted bit-for-bit.
    5. TASK-MATCHED COLLECTION DAMAGE (`task_matched`), the one untested densification lever
       `tall` named: collect the generic value buffer with damage drawn from the ERA'S OWN cell
       (`corrupt_hier` at the era's level/node) instead of `_corrupt`'s random blocks, and
       report its terminal success against `n_corrupt=1`'s. A measurement only — nothing is
       trained on it here — so Phase B can adopt it on evidence rather than on argument."""
    import torch
    cfg = _d6_cfg(era_cycles=cycles, budget=budget, g_budget=g_budget,
                  max_macro_level=max_macro_level, n_corrupt=n_corrupt, mine_cap=mine_cap,
                  probe_every=probe_every,
                  probe_widths=tuple(int(x) for x in probe_widths.split(",")),
                  n_pr=n_pr, n_rt=n_rt, n_score=n_score,
                  late_offset=max(0, cycles - commit_at),
                  # Phase A MEASURED task-matched collection and did not adopt it (adoption is
                  # Phase B's, on the strength of that measurement), and it ran at
                  # span_min_hold=256. Pinned here so `pa0` stays reproducible from this file.
                  collect_task_matched=False, span_min_hold=256,
                  sil_cv=0.10, lp_min_drop=0.05,
                  seed=seed, rule_seed=rule_seed, train_seed=train_seed)
    ers = parse_eras(era)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg["train_seed"]); np.random.seed(cfg["train_seed"])
    torch.set_float32_matmul_precision("high")
    started = time.time()
    print(f"[phase_a] tag={tag} depth={cfg['depth']} m={cfg['m']} "
          f"seqlen={cfg['s'] ** cfg['depth']} era={era} cycles={cycles} commit_at={commit_at} "
          f"budget={budget} g_budget={g_budget} n_corrupt={n_corrupt} arms={arms}", flush=True)

    _install_identity_miner(cfg["mine_support"])
    t0 = time.time()
    shared = _spiral_shared(cfg, device, ers)
    t_setup = time.time() - t0
    stale_succ = float(shared["replay"]["y"].mean())
    print(f"[phase_a] SETUP {t_setup:.0f}s   stale buffer terminal success = {stale_succ:.4f}"
          f"   [dens0 n_corrupt=1: 0.133 | tl_s0 config: 0.070 | depth 4: 0.296]", flush=True)

    sizes = {lv: len(build_move_set(cfg["depth"], cfg["s"], max_level=lv))
             for lv in range(1, cfg["depth"])}
    pricing = {"action_set": sizes,
               "width_enum": {n_: fit_width(n_, budget, g_budget) for n_ in sizes.values()},
               "g_enum": {n_: beam_ground(n_, budget, fit_width(n_, budget, g_budget))
                          for n_ in sizes.values()},
               "g_to_hold_w2_enum": {n_: beam_ground(n_, budget, 2) for n_ in sizes.values()},
               "width_route": {k: PN.fit_width_k(k, budget, g_budget)
                               for k in (1, 2, 4, 8, 12, 20, 32)}}
    print(f"[phase_a] pricing {json.dumps(pricing, cls=NumpyEncoder)}", flush=True)

    t0 = time.time()
    refs = measure_refs(shared, cfg, ers, device)
    t_refs = time.time() - t0
    plant = plant_probe(shared["generator0"], shared, cfg, device)
    read_acc = float(shared["read_acc"])
    span = cfg["s"] ** (max_macro_level - 1)
    print(f"[phase_a] REFS {t_refs:.0f}s stale={refs['stale']} floor={refs['floor']} "
          f"d0={refs['d0']}", flush=True)
    print(f"[phase_a] read_acc={read_acc:.4f}  compounded over the level-{max_macro_level} "
          f"span of {span} blocks: {read_acc ** span:.4f}   plant={plant}", flush=True)

    outdir = f"{DATA_DIR}/{REMOTE}/{tag}"
    os.makedirs(outdir, exist_ok=True)
    with open(f"{outdir}/setup.json", "w") as fh:
        json.dump({"config": cfg, "eras": ers, "refs": refs, "plant": plant,
                   "read_acc": read_acc, "stale_buffer_terminal_success": stale_succ,
                   "pricing": pricing, "t_setup_s": t_setup, "t_refs_s": t_refs},
                  fh, indent=2, cls=NumpyEncoder)
    volume.commit()

    # ---- 2/3. the descent gate + the pricing readout ------------------------------------ #
    gate = {"descent": {}, "pricing_at_commit": {}, "cycle_seconds": {}}
    for label in [a.strip() for a in arms.split(",") if a.strip()]:
        t0 = time.time()
        r = run_arm(label, label, {}, shared, cfg, ers, refs, outdir, device)
        n = len(r["log"]["cycle"])
        gate["cycle_seconds"][label] = (time.time() - t0) / max(n, 1)
        cc = next((e["cycle"] for e in r["events"] if e["kind"] == "commit"), None)
        gate["descent"][label] = _descent(r["log"], refs, commit_cycle=cc)
        gate["descent"][label]["commit_cycle"] = cc
        gate["descent"][label]["commit"] = [
            {k: e.get(k) for k in ("cycle", "level", "n_entries", "tab_recall",
                                   "tab_precision", "provisional")}
            for e in r["events"] if e["kind"] == "commit"]
        gate["pricing_at_commit"][label] = _pricing_rows(r["log"], cc)
        d = gate["descent"][label]
        print(f"\n[phase_a] === DESCENT {label} ===  rec_c1={d['rec_c1']:.3f} "
              f"rec_last5={d['rec_last5']:.3f} rec_best={d['rec_best']:.3f} "
              f"slope_pre={d['slope_pre_commit']:+.5f}/cyc "
              f"[dens0: rec_last5 0.262, rec_c1 0.233, slope -0.004]", flush=True)
        for row in gate["pricing_at_commit"][label]:
            print(f"[phase_a]   c{row['cycle']:3d} nm={row['n_moves']:3d} "
                  f"k_eff={row['k_eff']} w={row['width']} g={row['g_per_solve']:.0f} "
                  f"e={row['e']:.4f}", flush=True)
        with open(f"{outdir}/gate.json", "w") as fh:
            json.dump(gate, fh, indent=2, cls=NumpyEncoder)
        volume.commit()

    # ---- 4. the in-tag fidelity twins at depth 6 ---------------------------------------- #
    fcfg = {**cfg, "era_cycles": fid_cycles, "checkpoint_every": 10 ** 9,
            "probe_every": 10 ** 9}
    flogs = {}
    for label in [a.strip() for a in fid_arms.split(",") if a.strip()]:
        t0 = time.time()
        flogs[label] = run_arm(label, label, {}, shared, fcfg, ers, refs,
                               f"{outdir}/fid", device)
        print(f"[phase_a] fidelity arm {label}: {time.time() - t0:.0f}s", flush=True)
    fnames = list(flogs)
    fid = {"arms": fnames, "cycles": fid_cycles}
    if len(fnames) >= 2:
        a, b = flogs[fnames[0]]["log"], flogs[fnames[1]]["log"]
        worst = 0.0
        fid["series"] = {}
        # The BEHAVIOUR series must be bit-identical. `g_per_solve` and `t_cum` must NOT: the
        # port declares one extra grounding per instance for the root encode (`native`'s C-2
        # measured exactly that), so they are reported separately and the surcharge is checked
        # to be exactly the declared one rather than asserted to zero.
        for k in ("e", "succ", "dres", "n_moves", "width", "e_practice", "vloss", "gloss",
                  "n_solved", "n_mined", "m_per_solve"):
            x, y = np.asarray(a[k], float), np.asarray(b[k], float)
            L = min(len(x), len(y))
            dd = float(np.abs(x[:L] - y[:L]).max()) if L else float("nan")
            fid["series"][k] = dd
            worst = max(worst, dd)
        fid["max_abs_delta_behaviour"] = worst
        gx, gy = np.asarray(a["g_per_solve"], float), np.asarray(b["g_per_solve"], float)
        L = min(len(gx), len(gy))
        fid["root_encode_surcharge"] = (sorted(set(np.round(gy[:L] - gx[:L], 6).tolist()))
                                        if L else [])
        fid["pass"] = bool(worst == 0.0)
        print(f"[phase_a] === IN-TAG FIDELITY at depth 6: {fnames[0]} vs {fnames[1]} "
              f"behaviour max|delta| = {worst:.3e} over {fid_cycles} cycles "
              f"-> {'PASS' if fid['pass'] else 'FAIL'}; "
              f"g/solve surcharge = {fid['root_encode_surcharge']} ===", flush=True)
    gate["fidelity_d6"] = fid

    # ---- 5. task-matched collection damage (a measurement, not a treatment) ------------- #
    if task_matched:
        t0 = time.time()
        tm = _task_matched_buffer(shared, cfg, ers[0], device, n_episodes=tm_episodes)
        tm["seconds"] = time.time() - t0
        tm["reference_random_blocks"] = stale_succ
        gate["task_matched_collection"] = tm
        print(f"[phase_a] === TASK-MATCHED COLLECTION ===  terminal success "
              f"{tm['terminal_success']:.4f} against random-block {stale_succ:.4f} "
              f"[dens0 best 0.133 | depth 4 0.296]  ({tm['seconds']:.0f}s)", flush=True)

    gate["totals"] = {"t_setup_s": t_setup, "t_refs_s": t_refs,
                      "elapsed_s": time.time() - started}
    with open(f"{outdir}/gate.json", "w") as fh:
        json.dump(gate, fh, indent=2, cls=NumpyEncoder)
    with open(f"{outdir}/done.txt", "w") as fh:
        fh.write(f"elapsed {time.time() - started:.0f}s\n")
    volume.commit()
    print(f"\n[phase_a] DONE in {time.time() - started:.0f}s -> {outdir}", flush=True)
    # asserted LAST and not at the point of measurement: a gate failure must not throw away
    # the paid work that was already on the volume, but it must still fail the run visibly.
    assert gate["fidelity_d6"].get("pass", True), (
        "in-tag fidelity FAILED at depth 6: behaviour max|delta| = "
        f"{gate['fidelity_d6'].get('max_abs_delta_behaviour')}")
    return gate


def _task_matched_buffer(shared, cfg, era, device, n_episodes=8192):
    """The measurement-only wrapper (Phase A's): collect and report the rate, keep nothing."""
    vb = _collect_task_matched(shared, cfg, era, device, n_episodes=n_episodes)
    return {"terminal_success": float(vb["y"].mean()), "n_episodes": int(n_episodes),
            "n_states": int(vb["x"].shape[0]), "era": era["name"],
            "n_corrupt_equivalent": "corrupt_hier at the era's cell"}


def _collect_task_matched(shared, cfg, era, device, n_episodes=8192):
    """`collect_value_buffer` with the corruption drawn from THE TASK'S OWN damage cell.

    Same shape and same contract as the donor's collector — `{"x","r","y"}` over every state of
    every rollout, `y` the episode's terminal success broadcast down its trajectory — so it is
    a drop-in for `shared["replay"]` and for `value_steps`. The only difference is where the
    broken configurations come from: `context_instances` at this era's cell (rejection-sampled
    on d* > 0, exactly as the graded task is) instead of `_corrupt`'s random blocks.

    Its RNG is its OWN generator, so adding this instrument moves nothing else in the run."""
    import torch
    v, s, depth, m = cfg["v"], cfg["s"], cfg["depth"], cfg["m"]
    rules, rules_t, canon = shared["rules"], shared["rules_t"], shared["canon"]
    ms = shared["base_ms"]
    rng = np.random.default_rng(cfg["train_seed"] + 606_060)
    xs, rs, ys, done = [], [], [], 0
    while done < n_episodes:
        b = min(cfg["value_batch_collect"], n_episodes - done)
        done += b
        r_np, x_np = context_instances(rules, era_ctx(era), b, s, depth, v, m,
                                       seed=int(rng.integers(0, 2 ** 31 - 1)))
        x = torch.from_numpy(x_np).to(device)
        roots = torch.from_numpy(r_np).to(device)
        traj = [x.clone()]
        for _ in range(cfg["budget"]):
            x = behavior_step(shared["controller"], shared["generator0"], x, roots, ms,
                              rules_t, canon, depth, v, m, s, cfg["explore_eps"], device, rng)
            traj.append(x.clone())
        succ, _ = grade(x.cpu().numpy(), r_np, rules, s)
        y = torch.from_numpy(succ.astype(np.float32))
        for st in traj:
            xs.append(st.cpu()); rs.append(roots.cpu()); ys.append(y)
    return {"x": torch.cat(xs), "r": torch.cat(rs), "y": torch.cat(ys)}


@app.local_entrypoint()
def main(quick: bool = True):
    tutti_run.remote(quick=quick)
