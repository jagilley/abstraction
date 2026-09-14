# SPEC — voicing: the chooser at the class, trained on the learner's own attempts

**Status**: spec, 2026-09-12, written as the orchestrator's prompt to the implementer (the `tempo`
/ `embouchure` pattern: this file is that prompt verbatim). Nothing built, nothing run.
**Parent**: [`../enharmonic/`](../enharmonic/README.md) (`voicing.py` forks `enharmonic.py` at its
`en_s9` head; `quotient.py` and `merge.py` are imported, untouched).
**Siblings it answers to**: [`../embouchure/`](../embouchure/README.md) (the class lane it names;
the efference-copy key; the two ears) · `enharmonic` finding 5 and the `en_s9` addendum (the first
endogenous L5 commit, with no value visible). **Name**: figured bass gives the chord as a class;
the voicing is which notes realise it here.
**Attribution**: the class lane was named in `embouchure/SPEC.md` (2026-09-10). The readings that
arrival's value is deferred once more to the chooser, that the licence is a prior whose value is
read post hoc (Jasper's weight-decay reading), and that the chooser's ear reverts because its
in-run labels come through the inverse map rather than the write record, are from the 2026-09-12
discussion with Jasper (session `01QzhkcW5rAR1L8DnaKokqX8`, record to follow at close-out). The
arm axes, the sequence and the name are the orchestrator's.

---

You are the implementer for a new practice node. Read `/subagent-instructions` before you wait on
anything and `/run-experiment-on-modal` before you touch Modal. You never touch git. You halt twice
per experiment: once with the launch handle, once with the reduced results and figures. You reduce;
I interpret. No README: facts go to `FILES.md`, decisions to `DESIGN.md`, reductions under
`figures/`. `QUEUE.md` and `ROADMAP_PROGRESS.md` are mine.

## The question

The practice loop now reaches L5 on its own evidence (`en_s9`, cycle 186), and the book it adopted
has no visible value. `enharmonic` finding 5 says why a richer book cannot pay on this stack: after
π picks a slot, the entry written is chosen by a max-sum DP over the frozen generator's logits, a
free forward model of the surface used as a scorer, never trained on what the learner itself
chose and whether it worked. `embouchure` then found, one organ down, that a learner does learn the
intent-to-word rule from its own graded attempts, faster than from the corpus, if and only if it
files those attempts by its record of what it meant (the efference copy) and not by what its ear
parsed. This node asks the same question at the object the crank actually mints: **can the
executor learn which class to write at a slot from its own graded attempts, keyed by the record,
and does that turn the quotient's arrival into value?**

## Background, distilled

- `enharmonic` Q0 → `en_s9`: the token class is the earnable key; L5 arrives; the merge is
  within-level invisible and its value is read only post hoc as arrival at the level above; the
  adopted L5 book is 8 class-pair keys × 16 spellings at the cap, auditions worse than the oracle,
  no era-4/5 change in fifteen cycles. The expansion-choice instrument (`_EXP_REC`, `en_s5`/`en_s7`)
  shows the frontier choices beating the flat yoke's and the open book's slightly worse than the
  frozen anchor's. Read `enharmonic/README.md` in full, `figures/en_s9_reduction.txt`, and
  `results/RUN_en_s9.sh` (the command of record; its header is the mechanism).
- `embouchure` Q1: the meter's one coarse number per attempt trains the exact per-feature rule at
  c7 (surface route c19) on the record key; on the recall key the same objective plateaus, because
  the ear drops 16% of own writes and 99% of those are the errors. Q2.2: the chooser's ear (the
  generator) reverts to a last-writer student within ~8–16 cycles because `finetune_generator`'s
  in-run target is `bottom_map[code]` on the learner's own solved configurations. Read
  `embouchure/README.md` and `DESIGN.md` §2b, §3, §10, §12.
- The roadmap frame (`ROADMAP.md` §2.2, §7.1.3, §7.2): practice = address + trust; this lineage
  says there is a third organ, realisation, and that Track E′'s command port is the write record.

## What the code does today (verified; do not re-derive)

- **The chooser, closed slot**: `MC.apply_any` → `enharmonic.py::macro_features_pick` (l. ~4395):
  masks the span, one `generator.block_logits` read, the max-sum DP over `move["chain"]`/`flat`,
  `best` = the chosen entry per row, rendered through `canon`. Under the class key the table is
  the cross-product of held spellings (`quotient.py::ClassMiner.build`, capped at
  `quot_spell_cap` per class), so the DP's max over that inventory is the class choice and the
  spelling choice at once.
- **The chooser, open slot**: `native/span/span_net.py::SpanExecutor.apply` — once a slot passes
  parity the head emits the span's level-1 features from the trunk's pooled hiddens conditioned on
  the slot id; no DP, no table. Its training (`span_train_terms`) is self-imitation: the target is
  `dp_features` **recomputed from the current executor at training time** on stored masked
  contexts (`_store`). Targets are not stored. So the corridor head is the chooser on every open
  slot, and it is trained to chase the DP, not to improve on it.
- **The record exists at the moment of the write, per row, for free**: `macro_features_pick` and
  `dp_features_rec` return `best`; `SpanExecutor.apply` computes `feats`. No replay is needed
  (`embouchure` had to replay because its rendering organ was elsewhere; gate EB-8's coverage
  story does not apply here). The existing recorders (`_ENTRY_REC`, `_SLOT_REC`) keep bincounts,
  not per-row intents.
- **The verdict**: per instance, `succ > 0.5` at l. ~5856 (`solved = tips_flat[...]`), the meter's
  own feedback; `finetune_generator` already restricts the plant's training to `solved`.
- **The chooser's ear**: `finetune_generator` / `finetune_generator_span` (l. ~1832 / ~1879), level-1
  target `feats = bottom_map[...]` on solved configurations plus replay. On `rule_seed 0` the
  bottom map collides at codes 12 and 39 (`enharmonic/sizing/SIZING.md`, `inflection` finding 5),
  so a block the learner wrote as f1 is labelled f4 for its own generator.
- **The reader**: `shared["reader"]`, the frozen ear, also a `bottom_map` student; the harvest and
  the miner's observation parse go through it.

## The one change, in three moves

1. **A write record at the macro call.** For every macro call the learner makes in the priced
   beam, store beside the masked context (where `_store` already puts it) the intent: the chosen
   entry's level-1 tuple and its token class (`quot.id_of`), captured from `best`/`feats` at the
   write, plus enough of an instance handle to join the verdict when the instance is graded.
   Base moves are not macro calls and are out of scope.
2. **The corridor head trained on the learner's own attempts, keyed by the record.** Replace the
   self-imitation target with the recorded class on the learner's own calls, filtered or weighted
   by the grader's verdict. Two design choices are yours, with my recommendation:
   - *What the head emits.* Keep its output alphabet level-1 features (`span_net`'s
     identity/corridor distinction: no arbitrary label is ever a target; `handle/` is the measured
     negative of the other choice) and make the loss class-aware: any spelling in the recorded
     class is a valid target, the analogue of the grader's any-synonym reduction. A class-index
     output with a spelling pick is the alternative; I would not start there.
   - *The objective.* Start with imitation of the record on solved calls only (the plant's own
     precedent for "practice" in `finetune_generator`), with a pushed-away term on unsolved calls
     as a knob. `embouchure` found the advantage-weighted form worse than calibration on the
     recall key and never ran it on the record key; do not assume either way.
3. **The recall key as the contrast, and the read-back through the class.** The same training
   rows filed by the reader's parse of the written span, quotiented to a class — Q1's axis one
   rung up. And the class-level retraction signal: did the written span read back, through the
   reader and `quot.id_of`, as the class the record says was meant. A per-(level, node) confusion
   tally between record class and read-back class, never summed with the verdict.

A fourth knob, independent of the head and cheap: **`ear_record`** — the generator's in-run level-1
target on the blocks the learner itself wrote taken from the record instead of `bottom_map`, the
world's derived features elsewhere. This is DESIGN §12's "no truth to substitute" answered with the
one truth that exists for a learner-produced block, and it composes with any arm above.

Everything is a knob, default off, `# [voicing]`-marked; with every knob off the fork is
`enharmonic.py` at 0.000e+00.

## Sequence

**Q0, offline, CPU, before any GPU** (banked compact mirrors under `enharmonic/figures/`, the
`en_s9`/`en_s8` reductions, the slot recorder's counts): (i) volume — macro calls per slot per cycle
in solved instances against the self-imitation buffer sizes, so the own-attempt route's row count
is known before it is built; (ii) the DP's class accuracy today per (level, node, era) from the
instrument, the number the head must beat; (iii) how often record and recall would differ on this
draw (the two collisions), so the Q1 contrast is sized; (iv) the L5 book's classes against its
spellings — what execution needs (one spelling per class) against what the build needs, and
whether the cap keyed to the merged class (QUEUE item (i)) should ride along as a knob or wait;
(v) an offline ceiling if one is cheap: a head fitted to the oracle's per-node repair set
(`consistent_features`) on the audition path, the analogue of `embouchure`'s `own_verdict`. Halt
with facts in `DESIGN.md`.

**Q1**, on `en_s9`'s exact ladder and character (`endo_ledger_open_ung5_ra`, self-paced), at most
four arms in parallel: `dp` (the anchor, bit-identical to `en_s9`), `own_record`, `own_recall`,
`ear_record`. Smoke first. Halt with the launch handle, then with the reduction.

**Q2**, per what Q1 says: `own_readback` (the retraction route at the class), the composition
`own_record` + `ear_record`, clock yokes on the `dp` character for any arm whose outcome-currency
read matters (self-paced arms are not comparable on error; `enharmonic`'s matched-clock gaps
between near-identical arms reach 0.08, so the structural readouts are the claims), and the cap
knob if Q0 says it binds the head. Discussed with me before launch.

## Readouts

The expansion-choice instrument (`contains`, `contains_rep`, `success` per (level, node) per era)
extended to the priced beam phase if it is audition-only today · the head's class accuracy against
the repair set on the held-out split · L5/L6: rows, keys at support, class coverage per cycle,
commit cycles · matched-clock era-4/5 error with the in-tag floor and the caveat above · the
record-vs-read-back class confusion per (level, node) · corridor parity, misfire, slot churn ·
π's per-slot mass against use (F2, for the record; expected habit) · merge precision in both
spaces · the bill. Reduce with `analyze_enharmonic.py`'s sections plus the new ones, banking
`en_s9:endo_ledger_open_ung5_ra`, `en_s8:endo_ledger_open_ung5`, `en_s8:endo_ledger` beside.

## Gates

G-F: `fidelity_smoke` retargeted at `enharmonic.py`, the composed arm on the short config,
0.000e+00 with every knob off, re-run after every change to a shared path · V-1: the record is
exact — on the smoke, the stored intent equals what was rendered, per block, asserted (it is
captured at the write, so this is an identity, not a coverage) · V-2: the new training path with
the target switched back to `dp_features` reproduces `span_train_terms` bit for bit · V-3: the
inherited E-0 (singleton classes = flat) and E-7 (knobs absent vs False) still pass on the fork ·
`gates_cpu` in the lineage's style, asserting identities only where deterministic (DESIGN §11).

## Norms

Modal profile `chromatic`, L4, `modal run --detach …::fn` with the explicit function, the
session-isolated `launch_detached.py` idiom from the donor, `fetch_compact.py` for mirrors (raw
`modal volume get` corrupted `embouchure`'s files; `pack_tag` there is the fallback). Single seed;
ranks, signs, located mechanisms and multiples of measured floors are the claims. Volume, ladder,
caps, floors, licence and budget held fixed across arms; only the chooser's training signal and key
move. ~1.3 GPU-h per self-paced arm at this ladder. No outcome is interpreted in advance, by you or
by me: if the class choice was never the bottleneck, or the own-attempt route cannot reach the DP's
accuracy at this volume, the numbers that say so are the deliverable. Where a diagnosis of mine is
wrong, keep the withdrawal beside the correction in `DESIGN.md`.

**Not yours**: the pacer's owners and latches (three defects in one week; a redesign is queued
separately — do not add a fourth flag); the reader's teacher (`reader_target` stays `last_writer`;
this node runs on the substrate as it is); the leaf lane's follow-ups in `embouchure`; git.

---

## Q1, revised (2026-09-12, after Q0 — `DESIGN.md` §2–§7, gate VQ-1)

**What Q0 found that changes the arms.** (i) On the token class key the record and the recall of a
canonical write agree by construction on this draw: code 12 is unreachable by any write, and code
39's two writers render alike, which is the same fact that puts them in one class (§4). So
`own_recall` as specced has no structural contrast, and `ear_record`'s only reachable mislabel is
within-class, invisible to the grader. Both withdrawn for this draw, reasons kept. (ii) The thing
that writes is the head from era 3 on (share 0.96–1.00), a faithful copy of the DP within a 5–7%
misfire; the treatment is entirely the head's objective (§3). (iii) The executor writes one or two
rows on nearly every call at L4/L5 — one token class of six at the L5 commit, top-1 share 0.996 —
while `contains_rep` there is 0.58, so there is headroom the book covers (2 rows suffice for 7 of 8
features) and no variation to learn it from (§5, §6). Solved and unsolved calls carry the same
record. At L2/L3 the write genuinely varies.

**The reading, and the one addition.** The efference copy is only useful if the attempts vary. Every
inverse model this node was framed on is trained on babbling; `embouchure`'s renderer varied
because it was still learning, and here the chooser is an argmax that is constant at the frontier.
So the treatment gains its missing half: **exploration at the write**, in the priced practice beam
only, recorded exactly (V-1 holds on the sampled entry). Mechanically: on a closed slot, sample the
entry from the DP's scores at a temperature instead of the argmax; on an open slot, either sample
the head's per-block emission at that temperature or score the book's entries through the head
and sample among them (the latter keeps writes on-table, which is the question — choosing among
the classes the book holds; your call, stated in `DESIGN.md`). Size the temperature on the smoke
so top-1 share at L4/L5 falls to roughly 0.6–0.8 from 0.93–0.99; one temperature, stated. The
metering beam and every audition stay at argmax.

**Q1's arms, a 2×2, one tag, `en_s9`'s exact ladder and character:**

| arm | write | head's objective |
|---|---|---|
| `dp` | argmax | the donor's self-imitation of `dp_features` (bit-identical to banked `en_s9`; the in-tag identity check and the full-scale G-F at once) |
| `own_record` | argmax | the record-verdict calibration: reinforce the recorded class on solved calls, push away on unsolved, class-aware (any spelling in the class); unlabelled probe rows excluded from the term, kept for parity |
| `xp` | sampled | the donor's objective — the control for variation alone changing mining and the plant |
| `own_record_xp` | sampled | the calibration objective — the practice route whole: vary, file by the record, learn from the verdict |

**Instruments in every arm**, none billed to the learner: `contains` on the practice beam (free);
`contains_rep` on the beam subsampled to bound wall-clock, oracle-contained, its compute logged;
per (level, node) per cycle the distinct rows / keys / token classes written and the top-1 share
(Q0's §5 table, live); the in-situ read-back tally — the frozen reader run on the learner's own
written spans in place, quotiented through `quot.id_of`, against the record class, per (level,
node), never summed with the verdict (this is the number §4 says is unmeasured, and it decides
whether Q2's `own_readback` has anything to read); the head's class accuracy against the repair
set on the held-out split. The record on closed calls is still captured (cheap, §8), and an
off-table tuple's class is recorded as unnamed, not dropped.

**Not changed**: the ladder (the L5 slot's ten cycles are thin; a longer era 4 is a Q2 knob, not a
Q1 change, so `dp` stays the banked anchor); the cap (Q0: not what binds the head); the four-arm
ceiling; the gates G-F / V-1 / V-2 / V-3 and `gates_cpu`. Smoke first; two halts.

---

## Q2, revised (2026-09-12, after Q1 — `figures/vo_s1_reduction.txt`, `DESIGN.md` §18)

**What Q1 established.** The anchor is exact (0.000e+00 on every series against banked `en_s9`;
V-1 exact on 1.59 M filed writes) and the instruments deliver numbers no banked tag had: the
writer's class accuracy against the repair set on the priced beam at 5n1 is 0.717 / 0.677 (eras
4 / 5), the executor writes one token class on 100% of its L5 calls, and the ear's in-situ
read-back agrees with the record on 0.89–0.99 of L2 writes. Two treated cells never tested the
question, for two different mechanisms, both exact:

1. **The calibration objective destroyed parity.** `BCE(p_C, y)` on the head's *normalised*
   emission distribution drives the mass on the written class to the base rate (0.12–0.26) and
   sends the rest off-table; parity 0.20–0.33 against the 0.50 firing gate, corridor never open,
   the DP wrote everything (§18). The reading is about shape, not weight: a distribution over
   what to write cannot also hold how likely each class is to solve. A composed objective (donor
   term plus calibration) only sets a weight on that conflict, so it is not the Q2 fix. The
   chooser needs what π already has — a proposal organ and a separate critic.
2. **Exploration below the frontier costs the window at the frontier.** Every era advance in
   every arm fired on the cap; the commit owner reads the era's own level, so L3's window was
   era 2 and closed at c110. The anchor committed at c100. `voi_xp`, deviating on 9% of its L2
   writes, missed it (A3 0.37 vs 0.46 at c100) and no L4/L5 slot ever existed for it; so did the
   argmax calibration arm, with no exploration at all. The ladder's commit windows are fragile
   to any early perturbation of the mining stream, which is a standing confound of this lineage
   (the pacer's fourth defect this week, still not this node's to fix) — and it means exploration
   must not touch levels below the frontier, which is also the only place the chooser is constant.

**Q2's chooser: a critic over the book's classes.** A value head that reads the pooled context and
a candidate tuple's level-1 features (content, never a label — the corridor principle) and emits
P(solve). Trained by BCE against the verdict on filed writes keyed by the record, on the candidate
that was written; a class's value is the max over its held spellings. The executor, on a slot the
critic governs, picks the class by the critic over the on-table candidates and the spelling within
it as now (the head's likelihood or the DP's score). The corridor head's self-imitation target on
those slots becomes **the record** — what was written — so parity is against the executor's own
choice and the firing gate keeps its meaning. Without exploration the critic only ever sees the
one class the anchor writes at the frontier, which is the control, not a defect.

**Exploration, frontier-only and scale-free.** On the slots of the highest *adopted* level only,
in the priced practice beam, ε-greedy: on a fraction ε of macro calls write a uniformly drawn
on-table class (recorded exactly). ε = 0.3, no temperature to size — the realised deviation is ε
by construction and is logged. Every lower level, the metering beam and every audition stay at
argmax. When a new level is adopted, exploration moves up to it and the level below returns to
argmax.

**Arms, one tag, `en_s9`'s ladder, ≤ 4:** `dp` (anchor, bit-identical) · `critic` (the critic
chooses, no exploration) · `xp_f` (frontier ε-greedy, the donor's chooser and objective — the
control for variation alone) · `critic_xp` (the critic plus frontier ε-greedy — the practice route
whole). Gates: G-F unchanged · V-1 on the ε-draws · **V-4**: the critic arm with the critic's
governance off and its target switched back to `dp_features` is `dp` at 0.000e+00 · V-2 inherited.

**Instruments**: as Q1, plus the read-back budget allocated per (level, node) cell so L4/L5 are
sampled (the Q1 defect), the critic's held-out AUC against the verdict per slot, and the
deviation rate per slot which should read ε at the frontier and 0 elsewhere. The bill stays
unbilled and logged; ~6× the donor's `exp_reads` is accepted for this round.

**Halt at the smoke**, with the gate table and the critic's first held-out AUC, before the launch.

---

## Q3 (2026-09-13, after Q2 — `figures/vo_s2_reduction.txt`, `DESIGN.md` §23 and its correction)

**What Q2 established.** The critic works as an organ: with the verdict on a separate value head
the corridor stays open (28 slots, excess misfire 0.018 against the anchor's 0.068), the critic
predicts the verdict above chance from context and candidate content (held-out AUC 0.63–0.70 on
~800 rows per slot), and the frontier chooser stops being degenerate (up to four distinct tuples
where the anchor writes one). Two things did not work, each with an exact mechanism: (i) the
critic *replacing* the DP's score chooses worse than the DP by the repair-set measure at L4
(0.20–0.36 against 0.26–0.60), because the surface model is a strong prior on this world and a
modest judge that overrides it on half its calls throws the prior away; the faster climb was the
pacer reacting to a noisier executor, with poorer books and no L5. (ii) ε on the highest adopted
level is ε on the stream the next level is earned from; both ε arms stayed at L2 all run. On a
depth ladder every adopted level feeds the one being earned, so exploration cannot live in the
mined beam at all.

**Jasper's reading, which Q3 is built on**: a separately trained judge on top of the forward
model is how biology does it — the forward model supplies the prior and the judge the correction,
combined, not one replacing the other; and babbling happens in a context separate from
performance. Two changes follow.

**Change 1 — the two organs combined at the choice.** On a governed slot the executor scores
each on-table candidate by the DP's per-block score (the surface model's prior, the same quantity
the Q1 sampler used) and by the critic's logit, each standardised over the candidate set, summed
with the critic's weight `w = 1` (one knob, logged; the z-scoring is what keeps the two organs'
votes comparable across levels without a hand-set scale, the lesson of Q1's temperature and the
pacer's dead zone). Argmax of the sum. Report per slot how often the composed choice differs from
the DP's — the critic's actual influence is a measurement, never an assumption. The corridor head
still imitates the record on governed slots.

**Change 2 — babbling off-stream: the critic trained by priced counterfactual probes.** For each
governed slot, each cycle, take up to `n_probe` filed contexts from the practice beam (the final
configuration of the trajectory, solved or not), substitute a different on-table candidate class
at that slot (uniform over the classes the book holds there, excluding the one written), render
it, and pay the grader for one verdict on the substituted configuration. File
(context, candidate, verdict) into the critic's training set keyed by the record — and nowhere
else: never the miner, never the plant's solved pool, never π's or the value head's buffers. This
is `fourwall.entry_profile`'s substitution, which the merge already imports and which `enharmonic`
Q0 sized, pointed at the critic instead of the partition. It is priced: every probe grading is
billed to the meter as a grounding with its own bill line, so babbling costs what practice costs
and nothing it writes enters the repertoire. Size `n_probe` on the smoke so the probe bill is a
stated share of the anchor's priced time — start at 64 per governed slot per cycle and report the
share; cap the share at ~5% and say so if the cap binds.

**Arms, one tag, `en_s9`'s ladder, ≤ 4** — the 2×2 over {choice: composed vs replace} × {critic's
diet: filed writes vs filed + probes}, with Q2's `voi2_critic` as the banked {replace, filed}
cell: `dp` (anchor) · `comp` (composed, filed only) · `comp_pr` (composed, filed + probes) ·
`rep_pr` (replace, filed + probes). No ε anywhere in the mined beam.

**Gates.** G-F · V-1 · **V-4** in two forms: governance off with the critic trained on filed
writes is `dp` at 0.000e+00 as before; governance off with probes on is `dp` on every series
*except the bill lines*, which must differ by exactly the probe count times the unit cost —
assert both · **V-5**: the probe channel never touches the miner, the solved pool, π's buffer or
the value buffer — the probe-on/governance-off twin's `n_mined`, the miner's counts and the
solved pool are bit-identical to `dp`'s, asserted · V-4b extended to the composed scorer and to
the probe path, and, per the rule adopted in Q2, **no new gate is reported until it has been
shown to fail on a deliberate perturbation** · `gates_cpu`. Also carry the deferred one-line
recorder change: split `n_xp_shift` by fired vs closed path so reduction [M]'s decomposition
stops being a lower bound.

**Instruments.** As Q2, plus: the composed choice's critic-moved share per slot; the probe's own
counts and bill share per cycle; the critic's held-out AUC on probe rows and on filed rows
*separately*, with denominators and base rates; the read-back tally at L4/L5 now that the budget
is per cell; the head's class accuracy against the repair set on the priced beam, which is the
readout the round turns on.

**Sequence.** Smoke; halt with the gate table (denominators printed) and the probe bill share.
Launch only if every gate in the list is clean with non-empty denominators; otherwise halt with
the table. Two halts. ~5 GPU-h plus the probe bill.

---

## Q3b (2026-09-14, after Q3 — `DESIGN.md` §30–§31)

**What Q3 established, and what it could not.** The anchor is exact. The `{composed, filed}` cell
is valid and negative on its own clock: it committed L2 only. Its L3 yield read rose earlier than
the anchor's but flattened near 0.43 and never satisfied the commit latch before era 2's cap
closed at c110, where the anchor's read climbed to 0.52 and fired at c100. That is Q1 §19's
window read through a third different L2 write distribution: the anchor commits at c100,
Q2's replace-critic at c81, the composed arm never. The diet axis was not tested at all — the
probe rows were bought, billed and filed but a missing keyword kept them out of the critic's
loss; every inertness gate passed because with governance off a diet cannot reach the run
whether it is wired or not. Fixed; gate V-6 (a diet must move the parameters) added and shown to
fail on the actual defect. The defect left one clean number: a critic trained only on what the
beam chose to write ranks counterfactual probe rows at chance (held-out AUC ≈0.49 on probe rows
against ≈0.65 on filed rows), which is the argument for the probe diet as a measurement.

**The decision.** No treated arm except Q2's replace-critic has reached the levels this node is
about, and each reached or missed L3 through the commit latch's response to its own L2 writes.
The lineage's control for exactly this is the clock yoke (`enharmonic` Q1: the −0.40 gap was the
clock; the yokes erased it). So the re-run holds the ladder fixed: every treated arm replays the
anchor's realised commit and advance cycles (`en_s9`'s 48 / 100 / 151 / 186 and 60 / 110 / 180 /
192 / 201) through the fork's existing yoke machinery, and the chooser is read at L4 and L5 with
the same slots open at the same cycles as the anchor. The 2×2 over {choice: composed vs replace}
× {diet: filed vs filed + probes}, all yoked: `comp_yk` · `comp_pr_yk` · `rep_yk` · `rep_pr_yk`,
with `voi3_dp` / banked `en_s9` as the anchor. Same `w = 1`, `n_probe = 64`, no ε.

**Gates.** G-F · V-1 · V-4A/B · V-5 · V-6 · the yoke's own gate: the replay MATCHES the anchor on
every commit and advance (reduction [Y] in `enharmonic`'s style) · `gates_cpu`, with every new
gate shown to fail before it is reported. The falsification harness moves into the node
(`gates/`) so the evidence is part of the record. **Readouts that the round turns on**: the
writer's class accuracy against the repair set at the L4/L5 cells on the priced beam; distinct
tuples and top-1 share there; the critic's held-out AUC by diet with denominators; the composed
choice's moved share at L4/L5; era-4/5 error now at genuinely matched clocks, with the in-tag
floor. Launch on a clean table with non-empty denominators; ~5 GPU-h plus the probe bill.

---

## Q3c (2026-09-14, after Q3b — `DESIGN.md` §35): the seed pair on the load-bearing result

**What Q3b established.** With the ladder yoked to the anchor's clock, every treated arm has L4
and L5 slots at the anchor's cycles (Y-1 exact, 0 cancelled). The composed chooser on the
filed-only diet (`comp_yk`) is above the anchor at 6 of 6 frontier cells by the repair measure
and 0.223 below it in era-5 error at matched clocks, with the gain growing by era (0.05 / 0.08 /
0.22 at eras 3 / 4 / 5); it keeps the surface prior and overrides it rarely (moved 0.03–0.13 at
L4, 0.46 at 5n1 where it gains most). The replace chooser is worse than the anchor at matched
clocks, as in Q2. The probe diet buys counterfactual discrimination cheaply (probe AUC 0.49 →
0.80 at 0.55% of priced time) and the 2×2 interacts: probes help replace and hurt composed, one
draw, no mechanism offered. This is the first consumption-era value signal in the lineage with
L5 adopted, and `experiments/CLAUDE.md`'s conditions for a seed check hold: the result is
load-bearing, its magnitude rests on one trajectory, and the structural readouts triangulate
its sign but not its size.

**The run.** Seed 1: the anchor character self-paced at `--seed 1` (its own commits and
advances), then `comp_yk` and `comp_pr_yk` yoked to that anchor, identical flags otherwise.
Readouts as Q3b, [Y] first, then the frontier panel and era-4/5 error at the seed-1 anchor's
clock, and the in-tag floor. Two launches if the yoke must read a banked plan, one if the paid
run can resolve an in-tag source. ~6.5 GPU-h. No new gates beyond Y-1 on the new source.

---

## Q3d (2026-09-14, after Q3c — `DESIGN.md` §36): the anchor at seeds 2 and 3

**What Q3c established.** The yoke is seed-robust (exact, 0 cancelled). The anchor's ladder is
not: at seed 1 it commits L2 at c47 and nothing else, leaving era 2 at c77 (17 of 50 cycles) and
era 3 at c87 (10 of 70) by the advance owner going quiet, while its L3 yield read wanders 0.34–0.46
and never arms the commit latch; the run ends at c108 with no L3. So `en_s9`'s first endogenous
L5 commit, which this node and the `enharmonic` addendum lean on, is so far a seed-0 event, and the
frontier magnitude of Q3b (0.223 in era 5) is untested rather than refuted. What replicates: the
composed chooser's advantage in sign at every level both seeds reach (pooled repair accuracy at
L2/L3 +18–31% relative at seed 0, +25–27% at seed 1; 9 of 10 L2 cells) and the era-3 gap to 0.004
(−0.048 / −0.052). The probe–composition interaction of Q3b §5 reverses at L2/L3 on both seeds, so
it is a frontier-cell, one-seed observation.

**The run.** The anchor character (`voi3_dp`, self-paced, otherwise `en_s9`'s script) at
`--seed 2` and `--seed 3` in one tag, ~2.6 GPU-h. Readouts: the commit and advance record per
seed with the reason for each advance (cap or quiet), the certificate, the L3 yield read through
era 2, and the ladder reached. This is a seed check on `en_s9`'s headline as much as on Q3b's; it
is also how a seed at which the frontier exists is found. If either anchor reaches L4 or L5, the
follow-up is `comp_yk` yoked to it (one arm, ~2.6 GPU-h); if neither does, the pacer is the
lineage's critical path before any further chooser claim at the frontier, and that is a design
conversation, not a run.
