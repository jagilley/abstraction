# embouchure — the chooser trained on its own attempts: the efference copy, the listener's teacher, and the two ears

**Up**: [`../README.md`](../README.md) (practice arc) · **Spec**: [`SPEC.md`](SPEC.md) (the
orchestrator's prompt verbatim, with its dated corrections and the Q2 revisions appended) ·
**Machinery record**: [`FILES.md`](FILES.md) (every gate, run, flag and diagnostic, with the
complete run table) · decisions in [`DESIGN.md`](DESIGN.md) §1–§12 · **Conversation**:
`CONVERSATION.md`[^private].
**Machinery donors** (untouched, forked): [`../inflection/`](../inflection/README.md)
(`embouchure.py` forks `inflection.py`; every addition `# [embouchure]`-marked, every knob default
off, the G-F replay against the donor at 0.000e+00 twelve times across the round) ·
[`../inflection/sizing/`](../inflection/sizing/SIZING.md) (the identifiability ceilings) ·
[`../enharmonic/`](../enharmonic/README.md) (finding 5, the question this node answers to) ·
[`../../../../ideas/operators_are_arity_two.md`](../../../../ideas/operators_are_arity_two.md) §4–§5
(the command observed vs latent; inverse by forward search).
**Runs**: 2026-09-10 → 12, seed 0 throughout; **19.9 GPU-h measured** over eleven result tags plus
twelve G-F gates, five preflights and three crashed launches that no line records. Every cross-tag
anchor that is asserted is 0.000e+00; the one that is expected to fail (Q2.2's modified substrate)
is printed as the record of the change. **Ranks, per-cell counts, table precisions in two spaces,
recovered rules and multiples of in-tag floors are the claims.**
**Attribution**: the reading of PRs #101/#102 as ending on one organ, the max-sum DP as
inverse-by-search over a free forward model, the biological framing (the inverse map is trained on
the animal's own attempts) and the arm table are the orchestrator's; the questions that redirected
the round are Jasper's — whether a body is needed or whether this is commands vs efference copies
(2026-09-11), that the operative thing is matching word to intent rather than emitting phonemes
("tip of the tongue", "let me take that back"), and the go for the listener's-teacher pair; the
inflection builder's DESIGN §8 argument is the one this node argues with; the fork, the gates, the
bag objective, the write record and its two EB-8 corrections, the lexicon construction, the reader
probe, the drift instrument and the record corrections are the implementer's.

## One-liner

On every prior RHM practice node the step from a meaning to the word that says it was a free
lookup. This node makes it an organ, the *chooser*, and asks how that organ should be trained.
**A learner learns the intent-to-word rule from its own graded attempts, exactly and faster than
from the world's abundant surface, if and only if it files its practice by the record of what it
meant to say — the efference copy — and not by what it heard itself say**, because its own ear
fails to parse precisely its own errors (16% of own writes dropped, 99% of those the misspelled
ones). Listening to itself teaches it only to be consistent. Under homophony the learner's ears turn
out to be students of a lossy inverse map: the losing meaning of every shared form is never learned
as a word, four fifths of every arm's mined chunks are the right word for the wrong meaning, and the
exact grader cannot see it. Teach the listening ear the world's meanings and the junk goes to zero;
teach the chooser's ear too and the lost words return to the learner's own speech, though that
repair does not survive practice, because there is no truth with which to label a sentence the
learner itself produced.

## The question

Both PRs this node followed ended on the same organ. `enharmonic` found that a richer, all-legal
vocabulary performs worse than a poorer one because the executor's expansion choice is an argmax
over the frozen generator's logits, a free forward model of the surface used as a scorer, never
trained on the learner's own successes. `inflection` built a context-dependent spelling rule below
the tables and found a renderer fitted from the world's blocks learns it long before the pacer can
see it, while an unlearnable one misfires the pacer. Read together: the step from *what to say*
(π's slot; the class) to *how to say it here* (the synonym, the row) is the inverse model of
production, and the DP recovers it by search over a forward model. Every biological inverse model
(feedback-error learning, DIVA's babbling, the finch's subsong) is trained on the animal's own
attempts. So the treatment variable is the **source of the renderer's training signal**:

- **surface** — the world's rule-spelled blocks in solved instances (the perceptual route; the
  existing `fit_rule`).
- **own_scalar** — the learner's own written blocks, labelled only by the grader's per-instance
  spelling-error fraction, the meter's own feedback (a bag label over the ~9 blocks it wrote).
- **own_readback** — own written blocks, labelled by whether the learner's own frozen reader,
  run on the finished answer, returns the feature it meant (no grader; the retraction route).
- **own_verdict** — own blocks with the grader's per-block verdict (the inflection builder's named
  ceiling; fitted offline from the never-consumed `spell_rows`).

with `canon`, `given_rule` and `leaf` as before, and one further knob that Q1 forced: the
**cell key**, whether a training row is filed under the reader's parse of the learner's own write
(*recall*) or under the learner's record of what it wrote (*record*, the efference copy).

Vocabulary, since it accumulates: the *register* is the formality level 0–7 that picks which of
two synonyms is correct for a meaning; a *misspelled write* is the wrong synonym for the register,
never an ungrammatical one; the *reader* (the *listener* when pointed at the learner's own
answer) is the frozen neural ear that turns words into meanings; the *grader* is the world's exact
judge, meaning plus spelling; the *meter* is the price on its verdicts.

## What was built

[`embouchure.py`](embouchure.py) forks `inflection.py`: `fit_signal` (the source), `own_intent`
(the key), a write record taken beside `grade_spelled` before any training (gate EB-8, which caught
a replay through already-moved nets, then a one-in-forty-thousand batch-shape argmax tie, and is now
a coverage gate reporting its own exactness: 0.008–0.015%), the bag objective and its self-imitation
twin, a per-cell tally of off-rule against mis-read on the same blocks that is never summed, RB-1
(flip one block to its feature's other synonym and re-read), a constructed homophonous lexicon
(`lexicon_merge`, every level above untouched), and `reader_target ∈ {last_writer, listener,
both}` for Q2.2. [`analyze_embouchure.py`](analyze_embouchure.py) adds §1E–§11E and reduces tags
across each other with per-tag bit-identity assertions. `gates_cpu` is 81 checks (68 without torch;
the count is printed with the trap). Fetches go through `pack_tag`, since raw `modal volume get`
corrupted five of five 13 MB result files.

## Q0 — template and map are one function (CPU)

On every banked inflection row the correct synonym for the learner's own written block equals the
world's synonym at that (feature, register). So a renderer fitted from own writes lands on the
same table as one fitted from the world's blocks, **to the digit** (acc 1.000 practised / 0.875
held-out, θ̂ [2,2,5,5,5,5,5,2]), at full volume and at own-write volume (0.28 of the graded
configuration's 32 blocks). Neither channel is volume-limited: all 24 practised cells arrive at
cycle 1 either way. Writing a synonym *is* producing the sound; there is no articulator between
command and surface, and "learn from own attempts" carries no information the surface does not.
What own attempts can change is the key and the coarseness of the label. (Q0 also inferred that a
misspelling has no consequence through the reader; RB-1 with the production reader withdrew that:
a lone flipped synonym is misheard 0.56–0.65 of the time, a coherently misspelled answer is not.)

## Q1 — the efference copy is the seat of the practice route (four tags as one design)

`if_q1c_yk`'s design exactly (collision-free lexicon, practised registers {0,3,7}, setup rendered
as the world spells, 140-cycle ladder, schedule-paced), `canon_s` bit-identical across all four
tags and `own_scalar_s` bit-identical across two, so the six arms read as one design.

| arm | signal · key | acc held-out | first reached and held | lifetime written spelling error | θ̂ at c140 |
|---|---|---|---|---|---|
| `fit_rule_s` (surface) | world's blocks | 0.8750 | c19 | 0.0051 | [2,2,5,5,5,5,5,2] |
| **`own_scalar_rec_s`** | meter scalar · **record** | **0.8750** | **c7** | 0.0073 | **[2,2,5,5,5,5,5,2]** |
| `own_scalar_s` | meter scalar · recall | 0.7250 | c12 | 0.1880 | [3,4,4,3,4,3,3,3] |
| `own_readback_s` | own reader · record | 0.4500 | c25 | 0.3865 | [5,6,8,8,8,8,8,8] |
| `own_scalar_pg_s` | self-imitation · recall | 0.4250 | c133 | 0.3279 | [6,6,2,5,7,1,1,5] |
| `canon_s` | — | — | — | 0.4592 | — |

1. **The key carries the whole effect.** The same objective, rows and label, with only the cell
   key moved from the reader's parse to the write record, goes from one shared threshold it never
   leaves to the exact per-feature table, twelve cycles ahead of the surface route.
2. **The mechanism is selective deafness, not mis-hearing.** The reader never returned a different
   owner for an own write (0 of 148,305 blocks). It *fails to parse* 15.9% of them, and the dropped
   blocks are off-rule at 0.9915 against 0.0447 among the kept (22×). Filed by recall, the learner
   never gets a row for the cells it gets wrong while the grader's charge lands on the ones it got
   right; filed by record, seven blocks in 79,772 are lost. Six closed-loop simulations of that
   drop — uniform, random, the arm's feature marginal, error-correlated, coherence-conditioned, and
   the marginal crossed with both, all one register per bag — recover the exact table, so the
   dynamics by which the recall key costs 0.15 of held-out accuracy are **unexplained on the
   record** (DESIGN §10). The loss falls monotonically all run while the recovered table sits still.
3. **The retraction route learns what its listener depends on, and this listener does not depend
   on the register.** Cells with off-rule 1.000 and mis-read 0.000 over thousands of blocks: a
   coherently misspelled answer is register 0's spelling, and the reader never encoded the register.
   Only a lone off-register block in an on-register answer is misheard (RB-1 0.6455 at full scale).
4. **The objective is not the plateau.** The advantage-weighted self-imitation form, built to fix
   the calibration form, is worse on the same key. Not run on the record key, where the calibration
   form is at the ceiling.
5. **Trust is habit again**: π's per-slot mass is use count at 0.72–0.79 in three of four arms; the
   recall-keyed meter route is the exception (0.389) with π's spread across slots halved and its
   solves halved, unexplained beyond that.

## Q2.0 — the reader is a last-writer lookup, because its teacher is (CPU + 0.85 GPU-h)

A homophonous lexicon is constructed on the collision-free draw by merging forms at the bottom
map only; every H from 1 to 8 shared forms stays a solvable world, `d*` falls with H, and H=3 is
the densest rung that leaves every register some unambiguous surface. Probed at production reader
steps, the substrate's reader (**A**) is exact away from shared forms (0.9992–1.0000), at chance
where the rule admits both owners, and **a deterministic last-writer lookup at a shared form, 24 of
24 cells**, because `train_reader`'s target is `bottom_map[code]`: a function of the form alone,
so no corpus could teach it the register. A control reader (**B**), identical but trained on the
features the world derived, resolves even the rule-ambiguous cells from sentence context at
0.97–0.9998. The "rule-resolves" column at H=2,3 sits at an unpractised register and is a
generalisation test, not a disambiguation one. RB-1's earlier tension resolves: the reader is
label-faithful, context-sensitive where its labels were consistent and last-writer where they were
not.

## Q2.1 — a listener deaf to a distinction deletes it upstream of speech (3.5 GPU-h)

H=3, reader A, `--own-intent record`, five arms. The expected choice at the shared cells between
following the rule and being understood never arose: **the losing owner of each shared form all
but vanishes from the learner's own intents at the shared registers** (f0 at ρ3: 68 own writes
against f1's 2,883; f2 and f5 at ρ3: 1 and 2; pooled loser share 0.0018 in the retraction arm,
0.00015 in the meter arm). The learner mines its vocabulary through its reader, so a distinction the
reader cannot hear is never learned as a word. Every arm's mined L3 table is **precision 0.22–0.24
in feature space at precision 1.000 in token space** — the `enharmonic` junk channel, now on a knob
and route-independent — and the exact grader cannot see it, since writing the winner's form where
the loser was demanded is graded correct. The surface route's recovered rule falls from 0.931 at
unshared cells to 0.743 at shared ones (the relabelling fingerprint); the record-keyed meter route
holds 0.917 at the practised shared cells; no route recovers the one rule-resolving cell, which is
at an unpractised register.

## Q2.2 — the listener's teacher, and the two ears (6.6 GPU-h, a modified substrate)

The learner has two ears. Every listening path (the harvest, mining's observation parse, the
read-back, RB-1) goes through the reader; the executor's expansion choice is scored by the arm's own
generator, a second `bottom_map[code]` student. Two tags, one flag line, `--reader-target` the only
difference: `listener` (reader B as the ear on every reader path) and `both` (plus the generator's
setup supervision). The reader's four numbers are identical across the pair; the generator's invert.
Flagged as a modified substrate in every record; never merged with a reader-A tag.

| readout | `em_q2` (A) | listener | both |
|---|---|---|---|
| loser share of own writes, retraction arm | 0.00180 | 0.00220 | **0.01732** |
| loser share of own writes, meter arm | 0.00015 | 0.00674 | **0.01195** |
| precision(feature) at L3, surface / given / meter arms | 0.22–0.24 | **1.000** | **1.000** |
| recovered K at shared cells, surface | 0.743 | **0.914** | **0.914** |
| recovered K at shared cells, meter (record) | 0.771 | 0.800 | **0.914** |
| recovered K at shared cells, retraction | 0.600 | 0.286 | 0.457 |
| own writes misheard at shared cells (retraction / meter) | 0.001 / 0.000 | 0.381 / 0.289 | 0.268 / 0.248 |
| the chooser's ear, gen→derived at loser cells, c1 → c140 | — | 0.002 → 0.03 | **0.49–0.53 → 0.07–0.13** |

1. **The junk was entirely the listener's teacher.** Repair the listening ear alone and every
   correctly spelling route's mined table is 1.000 in feature space.
2. **The lost words return mainly when the chooser's ear is repaired too**, eight to eighty times
   more attempts at the losing meanings than under reader A, because every own-attempt route learns
   only what the chooser tries.
3. **The chooser's repair does not survive practice.** The generator's in-run targets come through
   the inverse map (for a sentence the learner produced there is no derived truth), so it reverts to
   a last-writer student within ~8–16 cycles in every arm (DESIGN §12, §11E). The words that came
   back were seeded in that window and persisted in the table and π's slots.
4. **The rule-vs-understood choice appears, and the routes split on it.** A context-resolving ear
   mishears the world's sentences at shared cells 0.015 of the time and the learner's own
   rule-following sentences 0.25–0.38 of the time. The retraction route leaves the rule to be heard
   (off-rule ≈ 1.0 at f0:ρ3 and f1:ρ3, 0.93 at f4:ρ3 and f6:ρ3) and is stuck where neither synonym
   is heard reliably (f6:ρ3, misheard 0.61 while off-rule), ending at 0.457 at the shared cells. The
   meter route follows the grader, is misheard by its own ear a quarter of the time at the same cells,
   and reaches the exact table at every practised shared cell from c5.
5. Mis-keying, dead under reader A (0.0004), is 0.12–0.14 here: the instrument built in Q1 has a
   world in which it reads.

## The update

1. **The efference copy is the seat of the practice route.** The meter's one coarse number per
   attempt suffices for the exact rule, faster than the free corpus, provided the learner keys its
   practice by what it meant and not by what it heard, because its own perception fails selectively
   on its own errors. The command port of Track E′ is the write record.
2. **Self-monitoring through one's own listener corrects toward being understood by oneself, not
   toward the rule.** On a listener indifferent to the register it teaches coherence; on one that
   resolves by context it teaches the speaker to abandon the rule where the speaker's own
   non-fluency gets it misheard. Only the grader, or a listener that cannot be fooled, teaches the
   rule. "Let me take that back" is real and fixes the wrong thing.
3. **The learner's ears are students of the inverse map, and that is what the arc's junk was.**
   `operators_not_footprints`' inversion hazard, measured in the learner's own organs: a many-to-one
   surface, a last-writer lookup as the ears' teacher, and the losing meaning never becomes a word.
   The reader stops being a free lunch and becomes the open organ; the arc has no way to teach the
   learner's ears the truth from the learner's own output.
4. **The chooser is a second ear, and it seeds what the learner can learn.** π picks the slot; the
   generator's logits pick the content; every own-attempt route learns only what was tried.
5. **On this substrate the template and the map coincide**, so the question of own attempts is a
   question of key and label coarseness, not of information, until an ear that can mishear is in
   the loop.

## What this does not show

The meaning currency is not load-bearing anywhere in this node: it inverts between the injective
and homophonous worlds (correct spellers 6.3–6.5 in-tag floors *worse* than `canon` under reader A,
11.6 floors *better* under the listener-only substrate, a ±1-floor band under both ears), the gaps
are of the size near-identical arms have shown at matched clocks in `enharmonic`, and no mechanism
is offered. The recall-key plateau's dynamics are unexplained after six simulations. The
self-imitation objective was tested only on the recall key. The write record is a post-beam replay
of the accepted moves, exact to 0.008–0.015%, not a capture inside the beam. Two of the
orchestrator's diagnoses (the objective; mis-keying) were tested and withdrawn, and are kept beside
their corrections in DESIGN.md. The homophony rung is one draw (H=3 of a greedy ladder) and the
loser features' true share of the world's demand at the shared registers is not measured, so the
recovery in Q2.2 is a ratio across tags, not an absolute.

## Reproduction

```bash
cd experiments/            # MODAL_PROFILE=chromatic; CPU steps need the venv with torch
PYTHONPATH=. python3 rhm/practice/embouchure/phase0_embouchure.py          # Q0
PYTHONPATH=. python3 rhm/practice/embouchure/phase0_q2_lexicon.py          # Q2.0 ladder
PYTHONPATH=. python3 -c "from rhm.practice.embouchure import embouchure as E; E.gates_cpu()"
python3 rhm/practice/embouchure/launch_detached.py --fn fidelity_smoke --tag em_gf12
# the tags, each with its full flag line in FILES.md §Runs; e.g. the Q1 corner and the Q2.2 pair:
python3 rhm/practice/embouchure/launch_detached.py --fn embouchure_run --tag em_q1 \
    --rule E_R8 --rule-seed 6 --practiced 0,3,7 --setup-render rule --policy schedule \
    --arms "canon_s,fit_rule_s,own_scalar_s" --eras "1:25:40,2:12:45,3:6:55" ...
bash rhm/practice/embouchure/results/launch_q22.sh em_q2b_l listener
bash rhm/practice/embouchure/results/launch_q22.sh em_q2b both
python3 rhm/practice/embouchure/analyze_embouchure.py --tag em_q1 --gf-tag em_gf8 --merge-tag em_q1b,em_q1c,em_q1d
python3 rhm/practice/embouchure/analyze_embouchure.py --tag em_q2 --gf-tag em_gf9
python3 rhm/practice/embouchure/analyze_embouchure.py --tag em_q2b --gf-tag em_gf12 --vs-tag em_q2b_l,em_q2
```

Volume `rhm-scaling-data:/rhm_practice_embouchure/<tag>/`; fetched copies and reductions under
`figures/`. Every flag per tag, every gate and every app id is in [`FILES.md`](FILES.md).

## Next steps (queued in `QUEUE.md`[^private], not started)

The ear's teacher: a reader trained on outcomes rather than forms, or on something a learner can
supply for its own productions · the class lane (the corridor head on `enharmonic`'s open book
trained on the learner's own solved classes, keyed on the write record, with a read-back parse
through the class-keyed table), once the composed arm in that lineage lands · a capture-at-write
efference copy if the record ever becomes load-bearing · the self-imitation objective on the record
key, for completeness · a denser rung or a second draw of the lexicon.

## Files

| file | purpose |
|---|---|
| `embouchure.py` | the substrate: `inflection.py` forked; `fit_signal`, `own_intent`, the write record, the bag objective and its self-imitation twin, `cell_tally`, RB-1, `lexicon_merge`, `reader_target`; entrypoints `preflight`, `fidelity_smoke`, `embouchure_run`, `q2_reader_sizing`, `q2_reader_probe`, `pack_tag`, `gates_cpu` |
| `analyze_embouchure.py` | the reduction: `analyze_inflection.py`'s sections plus §1E–§11E; `--merge-tag` (asserted bit-identity) and `--vs-tag` (measured, never merged) |
| `phase0_embouchure.py`, `phase0.json` | Q0 (CPU) |
| `phase0_q2_lexicon.py`, `phase0_q2.json` | Q2.0's ladder (CPU) |
| `launch_detached.py`, `results/launch_q22.sh`, `results/wait_app.sh` | the launcher, the Q2.2 pair's one flag line, the waiter on the Modal app state |
| `figures/` | every reduction and every fetched tag |
| `SPEC.md`, `DESIGN.md`, `FILES.md` | the prompt with its appended corrections; the decisions; the machinery record |

[^private]: Not mirrored: this link points to a document in the private lab repo (the roadmap, the queue, an unrun spec, reading notes, or a conversation). See the top-level README for what is held back and why.
