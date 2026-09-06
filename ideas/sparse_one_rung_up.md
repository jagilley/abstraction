# Every domain is sparse one rung up — the two faces of the learner's own trajectory, and what the introspective line must supply

**Status**: idea, 2026-09-02. Nothing new run. Written from Claude Code session
`session_01Ge2biQyLLGQE6NFkEeMdmf`, a re-read of the Track E/F line against Track A after the
`tutti` round ([#89](https://github.com/jagilley/research/pull/89)), with one new fact verified
against the banked reductions (§5). Every other number is re-read from an existing node.
**Conversation record**:
`conversations/tracks_ef_two_faces_2026-09-02.md`[^private]
— Jasper's prompts verbatim, responses summarized, including the two in-conversation claims the
code corrected before this doc was written (the miner reads reader-parsed outputs, not macro
calls; the output face is public).
**Attribution**: the questions are Jasper's — is there a regime where the introspective signal is
unnecessary; does pure practice make limited data bite less (the `reread` intuition); the FM is
the DP approximator on any domain we care about, so executor exactness is a control-variables
choice rather than a regime; and the "two sides of one coin" framing. The `an_m0` clock reading
in §5 is a parallel agent's, relayed by Jasper and verified here against the reductions. The
(domain, level) form of the density claim, the output-face / command-face split, and the
"the oracle becomes the FM" port came out of the exchange.
**Evidence base**: `ROADMAP.md`[^private] §2, §7.1–§7.2 ·
[`tutti/`](../experiments/rhm/practice/tutti/README.md) · `an_m0`
([`antiphon/FILES.md`](../experiments/rhm/practice/antiphon/FILES.md) §3a) ·
[`conductor/`](../experiments/rhm/practice/conductor/README.md) ·
[`crescendo/`](../experiments/rhm/practice/crescendo/README.md) ·
[`two_deltas/`](../experiments/rhm/practice/two_deltas/README.md) ·
[`reread/`](../experiments/rhm/practice/reread/README.md) +
[`reread/lm/`](../experiments/rhm/practice/reread/lm/README.md) ·
`papers/forward_self_models_paper2.md`[^private] ·
[`performance_error_is_the_bridge`](performance_error_is_the_bridge.md) §1, §4 ·
[`temporal_confabulation_test`](temporal_confabulation_test.md) (the program-level lesson) ·
the June sparsity sweep
([`RHM_SPARSITY_SWEEP_README`](../experiments/rhm/ratchet/RHM_SPARSITY_SWEEP_README.md)).
**Related**: [`practice_manufactures_its_own_credit`](practice_manufactures_its_own_credit.md)
§18 · [`two_climbings`](two_climbings.md) ·
[`meta_learning_under_metered_data`](meta_learning_under_metered_data.md) (the 08-17 bracket) ·
[`absorption_blinds_the_evaluator`](absorption_blinds_the_evaluator.md) ·
[`efference_copy_cancellation`](efference_copy_cancellation.md) (the same object on the a2a
wiring side) · [`heterogeneous_graders`](heterogeneous_graders.md) §4.

## One-liner

Dense versus sparse feedback is a property of a domain **at a level**, not of a domain: every
domain is dense at the level its supervision names and sparse one rung up, so the regime in which
a one-level-up signal looks unnecessary is exactly the regime whose grader cannot see what is
missing. Track A climbed without any introspective organ because it read the **output face** of
the learner's own trajectory — a count, over its own successes, of the next level's raw
material, parsed by an exact reader — and RHM supplied that reader, an exact executor and an
exact grader for free. The E/F line is the **command face** of the same trajectory: what the
learner meant to do, which trust, the use record and the metering signal all read. The command
face needs a forecast of each command's realization before it can be an error signal; on RHM
that forecast was exact DP, and on any domain we care about it is the forward self-model. That is
where the FM-to-value connection lives, and it is not what carried the climb.

## §1 The question

What must the E/F line supply that Track A, or the pure practice nodes, cannot — and is there a
regime where none of it is necessary? Two candidate regimes were on the table: domains with
abundant outcome feedback, where bona fide process supervision (intent being more invariant than
outcome) is not needed; and domains where data is limited, where the `reread` result suggests a
practice learner is less data-bound than a normal learner. Both turn out to be the same
question asked from two sides, and the record already answers most of it.

## §2 Density is a property of (domain, level), not of domains

The practice arc's most seed-stable fact is that the value of level-ℓ work is denominated in
level ℓ+1's currency and invisible to any within-level signal (§18; `recital`; `teacher_slot`).
Read as a statement about supervision rather than about practice, it says: a learner's feedback
is dense exactly at the level the supervision names, and sparse and evaluative one rung above
it, always. Pretraining is dense at the token and carries no signal about which invariance got
built. The measurements that make this concrete are all in one currency:

- `fourwall/lm` finding 5: keyed arms sit at Bayes-level task error with the token-derived
  inference pathway hollow at the next level, including an arm that never saw a rotation.
- `merge` round 1: task error identical under rotation while next-level representability differs
  by a factor of five.
- `teacher_slot` `outer_task`: the within-level ledger refuses the crossing *optimally* and ends
  hollow. It does not fail to fund the crossing; it votes it down.

Two consequences.

**Necessity claims are one-level-up claims.** Whether a regime needs a one-level-up signal cannot
be settled from inside that regime, because the grader that would show the deficiency is the one
the regime lacks. A regime map's "unnecessary" cells must be read as "unnecessary in the currency
that regime can read." The June sparsity sweep is a case in point: it found the FM's process
supervision redundant under dense NTP and useful only above ~0.9 masking, and that verdict was
graded in NTP loss. It shows the FM bought nothing in that currency, not that it bought nothing.

**The dense era was never dense one rung up.** §18's reading of pretraining as the correct
degenerate solution when the meter is off already noted that wherever LLM practice becomes
metered the missing components reappear with a human in the role. This is the same fact from the
supervision side: benchmarks, mixture ablations and reward models were the one-level-up gauge,
and a human carried it. Sparse-and-evaluative is not where continual learning is headed. It is
where every learner already is, one level above its supervision.

## §3 The coin: two faces of the learner's own trajectory

Practice's defining data source is `mine_from = chosen`: everything is read off the learner's own
trajectories. Those trajectories have two faces, and the arc's gauges divide cleanly between them.

**The output face — what I produced.** The miner takes the beam's own final answers, parses them
with the reader into level-1 features, and counts which spans recur across successful repairs;
the ratchet decides at build time whether a recurring span is an entry by requiring its halves to
be committed entries one level below. `at_support` for level ℓ+1 is that count. It is
one-level-up by nesting (successes at level ℓ contain adjacent level-ℓ calls, which are the raw
material of ℓ+1), label-free (the world grades solved/not, nothing else), and **public**: an
observer holding the learner's outputs and the same reader computes it identically. It owns
*what to freeze*, and it is the gauge that carried Track A.

**The command face — what I meant to do.** π's per-level proposal mass (trust), the beam's
selection mass on committed rows (the use record), the proposal-grain efference tag
(`antiphon` finding 5), and the intended spelling against its realization (δ_perf). These read
the learner's commands, not its outputs. They are **private** in paper 2's sense once the
commands are not logged: an outside observer sees the realized behavior and must reconstruct
what was meant, which costs the actor's own policy and forecast. They own *when to move on*
(the `tutti` mirror), *which questions* (the band-pass judge, whose value is trust-carried), junk
discrimination (`tr_s0`: the use record is the one endogenous guard that separates worthless
questions), and the executor's plasticity gain (`two_deltas`).

**What RHM externalized.** Not introspection, but four oracles attached to the two faces:

| face | needs | RHM supplied | generic domain |
|---|---|---|---|
| output | a reader that parses my outputs at my current vocabulary | the exact reader (`read_acc` 1.0) | my own perception, which `reread/lm` says climbs with the vocabulary |
| output | a grader | exact solved/not | sparse, noisy, but present |
| command | a log of my commands | π's proposals on disk (public by instrumentation) | free to the actor, private to everyone else |
| command | a forecast of each command's realization | exact DP: intention ≡ realization, so δ_perf ≡ 0 | **the forward self-model** |

The command face's forecast is the arity-2 self-model by definition: `bridge` §1 writes the
execution error as the value-relevant residual of a forecast of the next state given the
learner's own action, with the agency gate as the arity gap. On the routing-only A-track stack
that forecast was exact and the error identically zero, which is the substrate-level reason E1
found nothing to forecast but the learning rule. `intonation` made the executor fallible and the
error live, with the forecast still exact because the committed table is knowable. Off RHM the
forecast is not a stored row. It is the FM.

## §4 Why Track A climbed without any of the command face

Neither of the natural guesses holds. It was not a prior that next-level minability follows from
current-level utility: the within-level reader refuses the crossing and starves the level above
(`conductor` finding 2). It was not calendar timing, though it is timing in a precise sense.

- The climb's engine reads only the output face and the grader. Address is selection plus
  verbatim commit from outcome-graded successes; trust is π's self-imitation on the same
  successes. `two_climbings` §9's closing line is the answer: the un-giftable part of the climb
  is a gradient process in coordinates the discrete op lifted, and the op's input is successes.
- The pacer is a thermostat on the paired-interval slope of `at_support` with a dead zone
  measured by null-ABBA: it arms when the next-level count rises and commits when it quiets. In
  `crescendo` the L4 commit fired on the L4 stream the consolidated L3 policy was generating,
  two cycles after the shadow certificate, on both draws. Of the three candidate currencies only
  the mined yield stream cleared its floor at full config (`conductor` finding 3).
- The yoked clocks are bit-identical to the gauge arms, so the gauge's whole contribution is
  the cycle numbers it emits. But the schedule did not exist until the gauge wrote it. The
  gauge is a clock-writer, and the clock it writes is indexed by next-level candidates, not by
  the calendar. The pure schedule arms ran the experimenter's ladder from offline sizing and
  stumbled onto L4 at the era boundary rather than choosing it.

What the output face cannot see is where everything since has landed. It certifies tables that
are a quarter to a half junk, tolerable only because the exact executor and grader filter junk
downstream (`census`: a table 6% true executes 71–76% true). It reads "this level's next-level
stream has quieted," which licenses a commit, and Track A used the same quiet statistic for the
advance (`crescendo` finding 5), so it froze L3 before the policy had plateaued there; the
`tutti` mirror handed the advance to the execution currency, never froze L3, and was the only arm
above the lifetime ceiling. It goes sub-floor at L5 in flat coordinates, where the candidates are
too many to recur (`tutti/sizing`). And under the meter its clock parts from the command face's
(§5).

## §5 The meter desynchronizes the two faces (verified on disk)

The parallel agent's reading of `an_m0`, checked against `figures/an_m0/reduction.txt` and
`figures/an_s0/reduction.txt`:

| | abundance (`an_s0`) | metered (`an_m0`) |
|---|---|---|
| L3 commit cycle (six schedule-paced arms) | c69–82 | c73–85 |
| L3 observations at end of run | ~1,120 | ~590–670 |
| π rows per cycle (from `tutti` finding 5) | ~2,100 | ~300–350 |

The commit policy in these arms is certify-else-boundary, and the certificate is the unit-LP
audition descent — an output-face competence clock. That clock fired at the same calendar time
on half the observations. The trust clock runs on rows π imitates, and it was about six times
behind at that moment. So the meter did not shrink the reward per right question; it pulled apart
two clocks that abundance keeps collinear (`woodshed`: exposure and clock are collinear offline).
This is the sharpest instance yet of the meter making a second organ necessary rather than
nice: nothing Track A's loop reads is on the trust clock, and the instruments that are — the
trust-formation rate, the use record, a plateau read of the learner's own use — are all command
face. Recorded as a reading of one run; the fix it implies (a pacer whose clock is
the command face's) is the matched-row ladder's question, not its result.

## §6 Reread and limited data

The `reread` intuition — that a practice learner is less bound by limited data than a normal
learner — is right, but not for the reason first guessed, and it is bought by the F side rather
than by practice alone.

- The renewability result belongs to a **climbing reader** (`reread/lm`: a 6.4× re-read corpus
  fully renewable, capped not slowed below a corpus-size wall). On the practice side the frozen
  archive renewed competence and not vocabulary, because mining rewrites the archive by reading
  it (`reread` finding 5).
- Where practice is less data-hungry: affordability (depth unaffordable to primitives at any
  budget; earned tables at less priced time than given), and re-practice keeps raising
  competence where extra dense epochs buy nothing.
- Where it is more: in flat coordinates practice pays the tree's growth as widening, a flat
  entry says nothing about a tuple it has not seen, and arrival at L5 is doubly exponential
  (`tutti/sizing`). And it needs demand *variation* more than volume: self-demand manufactures
  starved holes (`merge` round 2).
- What makes limited data bite less for practice is the **change of key**. Under the category
  key, owning level ℓ−1 as classes lets the learner parse any level-ℓ tuple whose halves it can
  categorise (`enharmonic` fact 5). That is the reread mechanism — you cannot parse ℓ until you
  own ℓ−1 — installed in the action space. `enharmonic` Q1 is its direct test.

## §7 The regime map, corrected

The first draft of this map had a row for "executor exactness." Exact DP is a control-variables
choice on RHM, not a property of domains, so the row is re-stated as what it measures:

| coordinate | the command face is degenerate or redundant | the command face has a seat |
|---|---|---|
| intention vs realization | intention ≡ realization: no execution error exists (the routing-only stack, where E1 ran) | a fallible executor; on generic domains, every executor |
| feedback density, level-relative (§2) | dense at the graded level: every self-signal null in the line was cashed here (`temporal_confabulation_test`); the FM's local loss redundant *in NTP currency* | sparse and evaluative, which is every domain one rung up |
| recurrence per context | contexts do not recur: raw error beats the benchmarked form (`ma_s0`) | contexts recur: benchmarked δ_perf (`intonation`) |

So the live coordinate is not exactness but **estimability of the command face's forecast and
benchmark** at the available recurrence. The record carries two warnings for the port: `span`
measured the mjc plant FM's believed error against its true error at no correlation, and
`ostinato` found the benchmarked form's deep-era value negative at every subsampling rung. The
signal survives the meter; its baseline-subtracted form has not, and an estimated forecast has
not been tested in any seat.

**What the FM is for on generic domains, stated so it is not over- or under-claimed.** Needed:
the command face's forecast, without which the execution currency does not exist; the junk
filter the exact executor supplied for free; the advance and plasticity seats that currency owns.
Not needed: the output-face counter, which is the climb's pacer and ports as it stands. What
died in the E line is specific — the FM as a forecaster of its own weight update, and privilege
over content the world taught in public — and paper 2's privilege on implementation facts stands.
Paper 2 measured access and left open whether the private channel carries news the system should
act on; `two_deltas` and `tutti` answer that it does, in the advance seat and the plasticity seat
and not in the diet-gate seat. Paper 2 is the access half, the practice arc is the use half, and
`performance_error_is_the_bridge` is the bridge in the literal sense.

## §8 Open, testable

Questions, not predictions, per repo norms.

- **The observer twin on the metering signal.** Never applied, because on RHM the twin holds the
  logs. At the mjc port: can an observer reconstruct performance error from the realized
  movement alone, and does the actor's own read beat it?
- **A command-face pacer under the meter.** Does a commit clock indexed by π's rows, or by the
  trust-formation rate, re-synchronize `an_m0`'s L3 commits with the trust clock, and what does
  that do to the deep-era value the schedule-paced arms could not cash?
- **The learned intention reference.** The `acappella` lineage with an estimated forecast: do
  the advance and plasticity seats survive estimation error, and at what recurrence does the
  benchmark become estimable?
- **The output-face counter without the exact reader.** `native/lm`'s question with a sharper
  form: does `at_support` still pace the crank when the reader that parses the learner's own
  outputs is itself the learner's climbing perception?
- **`enharmonic` Q1** as the direct test of §6: does L5 arrive at the existing budget once T[4]
  is keyed by class.

## What this does not establish

Everything here except §5's table is argued from existing nodes, not measured. No
practice loop has run off RHM without the accompanist confound, so the portability of the
output-face counter is an argument from its inputs, and the necessity of the FM on generic
domains is an argument from what RHM supplied for free. The (domain, level) claim in §2 is a
reading of three measurements in one currency; the sharpest thing it predicts — that a regime's
own grader cannot certify the regime as self-sufficient — is a limit on evidence, and should be
treated as a methodological constraint rather than a result. The scope condition of ROADMAP §1.2
claim 4 stays in force: both faces assume the hierarchy lives in the experience stream, so that
the learner's own successes contain the next level's raw material.

[^private]: Not mirrored: this link points to a document in the private lab repo (the roadmap, the queue, an unrun spec, reading notes, or a conversation). See the top-level README for what is held back and why.
