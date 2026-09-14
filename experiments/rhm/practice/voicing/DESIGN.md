# voicing — DESIGN

Decisions and the facts they rest on. **Facts** live in [`FILES.md`](FILES.md) (the machinery
record) and under [`figures/`](figures/); the prompt is [`SPEC.md`](SPEC.md). Parent:
[`../enharmonic/`](../enharmonic/README.md). Sibling: [`../embouchure/`](../embouchure/README.md).
No README until discussed.

Status, 2026-09-12: **Q0 done, offline, CPU, no GPU spent, nothing built.** The reduction of
record is [`figures/vo_q0_reduction.txt`](figures/vo_q0_reduction.txt), produced by
[`q0_voicing.py`](q0_voicing.py) from the banked compact mirrors of `en_s9` and `en_s8` under
`../enharmonic/figures/`. Q1 has not been designed past what §7 says, and nothing has been
launched.

---

## §0. Scope, and why `voicing.py` does not exist yet

The SPEC's Q0 clause says the volume question is asked "so the own-attempt route's row count is
known **before it is built**". So this session wrote no fork. `voicing.py` (the `# [voicing]`
fork of `enharmonic.py` at its `en_s9` head, `quotient.py` and `merge.py` imported untouched)
and its gates G-F / V-1 / V-2 / V-3 belong to the Q1 step, and §8 is the located hook map that
makes that step short. `enharmonic.py` has not been touched; nothing in `../enharmonic/` has
been touched.

One piece of local infrastructure was stood up and is not checked in: a `uv` venv at
`<repo>/.venv` (Python 3.11, `numpy==1.26.4`, `scipy==1.16.3`, `torch==2.7.0+cpu`,
`modal==1.5.5` — `modal` only because `rhm/rhm_sculpt_precheck.py` imports it at module scope and
`quotient.py` imports `possible_sets` from there). `.venv/` is already in the repo `.gitignore`.

---

## §1. What Q0 could and could not ask offline

Everything in this file is a pure function of the DGP at `rule_seed 0`, the banked per-cycle
logs, and the `picks` / `n_rows_in_class` fields `figured_bass` added to `ClassMiner.build`.
**Gate VQ-1** is what licenses the book-level claims: the operative book at every logged build is
rebuilt from `picks` and its row count asserted against the run's own `n_entries` — 606 builds on
`en_s9:endo_ledger_open_ung5_ra`, 628 on `en_s8:endo_ledger_open_ung5`, 591 on
`en_s8:endo_ledger`, all PASS. `alias_audit.py` could not do this (it re-ranks a capped class by
index, and the cap binds on every L4/L5 build in this tag); reading the logged `picks` closes it.

One reconstruction subtlety, recorded because it cost a gate failure: `log["quot"][i]["build"]`
is the miner's `last_build`, and **more than one call sets it per cycle** — the gauge path builds
live over live at every level, while `operative(l-1)` hands a *closed* arm the table its commit
froze. The rebuild therefore selects its lower chain by the build's own `n_lower_rows` rather
than assuming one. On an open arm the two coincide.

---

## §2. [V1] Volume — the own-attempt route is not starved above L3, and is thin at L5

**The capture that already exists.** `SpanExecutor._store` keeps at most `per_call = 24` rows of
the `n_pr x width` beam block per macro call, in both priced beams and in neither probe
(`ex.capture` is set False around the battery and the probe ladder). Measured on slots below
cap, that is **148 / 260 / 289 / 366 rows per slot per cycle at L2 / L3 / L4 / L5**, i.e. 6–15
macro calls per slot per cycle. The L5 slot's first cycle of life shows exactly
`budget x per_call x 2 beams = 384`, which identifies the two capturing sites as the practice
beam (`n_pr = 64`, per-**tip** `succ`) and the metering beam (`n_rt = 384`, graded at the
**answer** only).

**The buffer is already at cap where it matters.** `span_buf_cap = 8192`; 28 of 30 slots are at
cap by era 4. The head consumes `span_batch x gen_steps = 1280` row-draws per open slot per
cycle.

**What a verdict filter leaves.** Solved-tip rate 0.201–0.287 by era (answer rate 0.171–0.269).
So a solved-only filter on the existing capture leaves a steady-state buffer of ~1,650–2,350 rows
per slot — still 1.3–1.8x the per-cycle consumption. **Volume is not the binding constraint for
an own-attempt route at L2–L4.**

**The join ladder, which is the real cost.** The beam expands every move at every step and keeps
`width` of `width x n_moves` children, so `1 - 1/n_moves` of all writes are discarded by the value
head before any verdict exists. Per cycle, practice beam only:

| era | writes | kept writes | on solved tips | macro share | macro rows on solved tips |
|---|---|---|---|---|---|
| 1 | 298,412 | 8,414 | 2,384 | 0.028 | 67 |
| 2 | 445,340 | 8,950 | 1,802 | 0.094 | 170 |
| 3 | 523,452 | 9,070 | 2,174 | 0.239 | 521 |
| 4 | 527,175 | 8,619 | 2,132 | 0.440 | 938 |
| 5 | 571,392 | 9,216 | 2,645 | 0.462 | 1,223 |

938 macro rows on solved tips per cycle in era 4, over ~28 open slots, is **~33 rows per slot per
cycle** under the strictest join, against 289 captured today and 1,280 consumed. Accumulated in
an 8,192-row buffer that is ~1,300 rows/slot after forty cycles — workable, but era 4 is twelve
cycles long on this ladder.

**The L5 slot's whole life** (the level the node is about): minted at c187, buffer 354, held-out
30; parity unreadable until `span_min_hold = 128` held-out rows arrive; **opens at c192**, six
cycles after the L5 commit at c186, and lives ten cycles to c201. Parity at open is 0.923 / 1.000
and `sacc` (self-imitation train accuracy against the DP) is 0.77–1.00 from its first cycle. So
the head is an almost-immediate faithful copy of the DP at L5, on a book of 128 rows.

---

## §3. [V2] The DP's class accuracy today, and the two places the instrument does not look

The expansion-choice instrument per (level, node, era), `en_s9:endo_ledger_open_ung5_ra`
(`contains` = the chosen row's token class holds the clean latent; `contains_rep` = it holds some
feature that repairs the instance; `succ` = the repair graded):

| era | 2n12 c / rep / succ | 3n6 | 4n3 | 5n1 |
|---|---|---|---|---|
| 1 | 0.523 / 0.736 / 0.707 | 0.440 / 0.616 / 0.615 | 0.780 / 0.802 / 0.802 | 0.807 / 0.687 / 0.481 |
| 2 | 0.540 / 0.705 / 0.697 | 0.431 / 0.574 / 0.573 | 0.769 / 0.784 / 0.784 | 0.914 / 0.634 / 0.438 |
| 3 | 0.274 / 0.125 / 0.125 | 0.389 / 0.460 / 0.459 | 0.603 / 0.627 / 0.612 | 0.588 / 0.506 / 0.358 |
| 4 | 0.285 / 0.024 / 0.024 | 0.245 / 0.138 / 0.134 | 0.670 / 0.718 / 0.711 | 0.645 / 0.582 / 0.463 |
| 5 (2n8) | 0.219 / 0.012 / 0.012 | (3n4) 0.205 / 0.023 / 0.023 | (4n2) 0.406 / 0.106 / 0.104 | 0.707 / 0.553 / 0.443 |

`succ` tracks `contains_rep` within ~0.01–0.12 everywhere, so the repair set is the tight
reference and the clean latent is not (`enharmonic`'s own note, confirmed at every cell). The
frontier number the head has to beat is therefore **`contains_rep` 0.582 at 5n1 in era 4 and
0.553 in era 5**, and 0.718 / 0.106 at the 4-node.

**Two scopes, stated because both bite.**

1. **The instrument is audition-only.** `_exp_add` is called from `audition_macro`, which always
   materialises through `MC.apply_any` / `macro_apply_rec` — the DP. It never sees the beam, and
   it never sees the head. So the table above is the DP's accuracy on the audition pool, not the
   accuracy of the thing that writes.
2. **The thing that writes is the head, not the DP.** Head share of macro rows by era:
   **0.037 / 0.672 / 0.959 / 0.868 / 1.000**. From era 3 on, the DP is essentially not the
   executor. `enharmonic` finding 5 names the DP as the cap on a richer vocabulary; on this arm
   the DP's role by era 4 is as the head's *training target*, and the head reproduces it to within
   a misfire rate of **0.064 / 0.072 / 0.071 / 0.053** (eras 2–5, exact-span match on fired rows).
   The finding survives, but through the imitation chain, not through the execution path.

Extending the instrument to the priced beam is cheap on one of its two columns and not on the
other: the practice beam already computes the clean derivation (`context_instances(...,
with_clean=True)` at the beam's own draw, and `_latents_at` reads the demand off it with no RNG
moved), so `contains` on the beam is free; `contains_rep` needs `fourwall.consistent_features` per
beam state, which is a grader call per row and is priced. The metering beam is drawn without
`with_clean`, and the SPEC-era comment at the shadow draw records that asking for `clean` moves no
stream, so adding it there is safe.

---

## §4. [V3] Record vs recall — a correction to the SPEC's premise

The SPEC (and `enharmonic`'s own notes) say the bottom map "collides at codes 12 and 39", so a
block the learner wrote as f1 is labelled f4 for its own generator, and the Q1 contrast
`own_record` vs `own_recall` is sized by those collisions. Measured on this draw:

- The two collisions are **code 12, written by (f=1, rule 1) and (f=4, rule 1)**, and **code 39,
  written by (f=2, rule 0) and (f=7, rule 0)**.
- A macro writes `canon[f]` — **rule 0 only**. So **code 12 is unreachable by any write the
  executor makes**; only code 39 is.
- Code 39's two writers collide *on their canonical renderings*: `canon[2] == canon[7]`. So the
  written surface is literally the same string, and the token class of the record and of the
  recall are **identical by construction** — `token_class_sets` is a function of `canon[feats]`
  alone.

Propagated over the books the arm actually held (all rows of every operative table at c151, c186,
c201):

| level | rows | record tuple != recall tuple | **token class differs** | learned class differs | recall unnamed by the learner |
|---|---|---|---|---|---|
| L2 | 55 | 0.182 | **0.000** | 0.000 | 0.000 |
| L3 | 273 | 0.476 | **0.000** | 0.062 | 0.000 |
| L4 | 304 | 0.250 | **0.000** | 0.000 | 0.000 |
| L5 | 224 | 0.036 | **0.000** | 0.036 | 0.036 |

**So the Q1 contrast as specced is not sized by the collisions.** On the token class key the two
keys cannot differ structurally at all on this draw. What is left for `own_recall` to differ on is
two things, both real and neither structural:

1. **the arm's own learned quotient** — a union-find over tuples it has *seen*, so a recalled
   tuple can land in a different learned class (0.062 of L3 rows) or in no class the arm can name
   at all (0.036 of L5 rows at c151, before the L5 commit). This is `embouchure`'s "the ear drops
   what it cannot name", one rung up, and it is the only place the class key itself carries the
   contrast.
2. **the neural ear's error in situ**, which **no number in this tag measures**. `plant_probe`
   gives the generator's `parse_acc` 0.684 -> 0.651 over the run and the frozen reader's
   `read_acc` 1.000 -> 1.000, but both are measured on `probe_clean` — clean held-out
   configurations — against `bottom_map`. The reader's accuracy on a span the *learner* wrote,
   sitting inside a partly-damaged configuration, is exactly what an `own_recall` arm would
   expose and is not measured anywhere.

The withdrawal, kept beside the correction as the SPEC asks: the diagnosis "on `rule_seed 0` the
bottom map collides at codes 12 and 39, **so** a block the learner wrote as f1 is labelled f4 for
its own generator" is half right. The collision at 12 exists in the map and cannot be reached by a
write. The collision at 39 is reachable and is invisible to the class, because the two features
that collide are the two features that *render alike*, which is the same fact that puts them in
one token class.

---

## §5. [V4] The book: classes against spellings, and a chooser that is not choosing

**What the cap does.** `quot_spell_cap = 4` per class per half, so a class-pair key materialises
at most 16 rows. At the L5 commit (c186) the L4 book holds 168 rows in 16 classes with spellings
per class `[40, 24, 16, 16, 16, 16, 16, 16, 1x8]`: eight classes are over the cap and truncated to
4, eight are singletons. `n_class_capped` counts capping events over `pick` calls: **16 at L5, which is every one of the
8 built keys x 2 halves**, and 46 at L4 over at most 78 calls (39 at-support keys x 2). The cap
binds on every half of every built L5 key. The resulting L5 book is **128 rows = 8 class-pair keys x 16
spellings, spanning 6 distinct token classes, every key single-class**. At c200 it is 192 rows /
12 keys / 12 token classes, and two of the twelve keys are no longer single-class.

**What execution needs.** The table-side ceiling (§6) says the union of the L5 book's token
classes reaches 7 of the alphabet's 8 features and that **2 of its rows suffice** for that whole
union. So at L5 the
book holds 128 rows to do the work of 2, and the other 126 are spelling choices inside a key that
cannot change the class.

**What the chooser actually does with them.** The entry recorder (closed path) and the slot
recorder (fired path) are bincounts over the operative table's rows:

| cycle | level | rows in table | distinct rows written | top-1 share | keys used / held | **token classes used** |
|---|---|---|---|---|---|---|
| 48 | L2 | 13 | 13 | 0.394 | 13 / 13 | 9 |
| 100 | L3 | 61 | 28 | 0.697 | 9 / 11 | 6 |
| 151 | L4 | 136 | 8 | 0.905 | 5 / 10 | 5 |
| 186 | L4 | 168 | 4 | 0.951 | 3 / 13 | 3 |
| **186** | **L5** | **128** | **2** | **0.996** | **2 / 8** | **1** |
| 188 | L5 | 144 | 1 | 1.000 | 1 / 9 | 1 |

Aggregated per (era, level) over the fired path's own per-(level, node) vectors — mean distinct
rows written per cycle and mean top-1 share:

| era | L2 | L3 | L4 | L5 |
|---|---|---|---|---|
| 3 | 10.3 of 17.9, top1 0.400 | 9.9 of 88.2, top1 0.549 | 2.6 of 154.7, top1 0.906 | — |
| 4 | 10.5 of 18.0, top1 0.452 | 10.4 of 89.0, top1 0.594 | 3.4 of 173.0, top1 0.936 | 1.5 of 144.0, top1 0.800 |
| 5 | 10.4 of 18.3, top1 0.480 | 10.9 of 93.0, top1 0.611 | 3.6 of 186.7, top1 0.926 | 1.6 of 152.4, top1 0.930 |

At L5 the executor writes **one token class on 100% of its calls** at the commit cycle — two rows,
both in the same class, 99.6% of calls on one of them — out of the six classes the book holds. At L4 it writes three of thirteen. At L2 and L3 the write genuinely varies.

**This is the fact that most changes the design, and it is stated as a fact, not as a
prediction.** An own-attempt objective learns from the contrast between what the learner wrote on
its solved calls and what it wrote on its unsolved ones. At L4 and L5 on this arm there is almost
no such contrast to learn from: the same one or two rows are written on nearly every call of a
cycle, solved and unsolved alike. The contrast exists at L2 and L3 and thins monotonically with
level. `embouchure` DESIGN §3 recorded the same shape one organ down as the "no babbling"
property of a deterministic renderer ("an arm that stops learning when it stops being wrong"); the
form it takes here is stronger, because the write is not merely right but *constant*.

**The QUEUE's cap item (i), sized.** Raising or re-keying `quot_spell_cap` changes how many
spellings a class-pair key materialises. It does not change the number of *keys*, which is what
the class choice is over, and the executor at L5 is already using 2 of 8 keys and 1 of 6 classes.
So the cap is not visibly what binds the head; it binds the *book*. Whether the cap knob rides
along in Q1 is the orchestrator's call (§7.5), but Q0's answer to "does the cap bind the head" is
**not on this evidence**.

---

## §6. [V5] The offline ceiling — not available in the specced form, and what replaced it

The SPEC asks for "a head fitted to the oracle's per-node repair set (`consistent_features`) on
the audition path, the analogue of `embouchure`'s `own_verdict`", **if one is cheap**. It is not,
and the reason is structural rather than effortful:

- the head's input is `trunk(core, obs)` — the pooled hiddens of the arm's own generator at that
  cycle. No banked tag carries the trunk's weights; the compact mirror is `results.json` plus the
  entry recorders. There is no `x` to fit on.
- the contexts are not banked either. `SpanExecutor._store` holds them in memory and the log
  records only their count.
- `embouchure` could fit `own_verdict` offline because its head's input was a (feature, register)
  pair — not a learned representation — and its rows were on the volume in `spell_rows`.

So the analogous ceiling here is an **in-run** arm whose target is the repair set, and it belongs
to Q1/Q2, not to Q0.

What was computed instead is the **table-side ceiling**, which is cheap and is a real bound:
whatever the chooser is, it picks one row of the operative table, so it cannot do better than the
table's own coverage.

| cycle | level | rows | token classes in book | features coverable (of v=8) | rows needed for full cover |
|---|---|---|---|---|---|
| 186 | L2 | 18 | 10 | 8 | 6 |
| 186 | L3 | 89 | 9 | 7 | 5 |
| 186 | L4 | 168 | 9 | 6 | 2 |
| 186 | L5 | 128 | 6 | 7 | 2 |
| 200 | L4 | 188 | 11 | 7 | 3 |
| 200 | L5 | 192 | 12 | 7 | 2 |

The book is not the constraint at L4/L5: it reaches 6–7 of 8 features with 2–3 rows and holds
128–192.

---

## §7. What Q0 changes about the design, and what needs the orchestrator

Facts first (§2–§6), then the five places they touch the SPEC. Nothing here interprets an outcome
that has not been gathered.

1. **The chooser the SPEC names is the DP; the thing that writes is the head** (head share
   0.96–1.00 from era 3). Move 2 of the SPEC — replace the corridor head's self-imitation target —
   is therefore the whole of the treatment on this arm, and move 1's record on *closed* macro
   calls covers 13% of era-4 macro rows and 0% of era-5. Cheap to capture either way (§8), but the
   weight is on the head.
2. **The Q1 contrast `own_record` vs `own_recall` is not sized by the two collisions.** On the
   token class key the record and the recall of a canonical write agree by construction on this
   draw (§4). The contrast that remains is the learned quotient's (0–6% of rows) and the reader's
   in-situ error, which is unmeasured. Whether `own_recall` is still worth an arm — or whether the
   honest version of Q1's axis one is "record vs the *neural* read-back", with the read-back rate
   measured as its own readout — is the orchestrator's call.
3. **The own-attempt route has contrast at L2–L3 and almost none at L4–L5** (§5). This is not a
   volume problem (§2) — the rows are there — it is a variance problem: the same row is written on
   nearly every call at the frontier, so solved and unsolved calls carry the same record. Three
   handles exist and none is chosen here: (a) exploration at the write (the beam already explores
   at the *move* level via `prop_explore`, never at the write); (b) an objective over the head's
   output distribution rather than its argmax, where the contrast is not degenerate; (c) accept
   the degeneracy and read the arm at L2/L3 where the contrast is real. Decision needed before
   Q1's arm table is fixed.
4. **The expansion-choice instrument must be extended to the beam** for Q1 to read anything about
   the head at all — today no banked number says what the head wrote in class terms. `contains` on
   the practice beam is free (the clean derivation is already drawn); `contains_rep` is a priced
   grader call per row. Recommend: `contains` on the beam in every arm, `contains_rep` on the beam
   behind a knob with its own bill line.
5. **The spelling cap is not what binds the head** on this evidence (§5), so the QUEUE's item (i)
   does not need to ride along for the head's sake. It may still be wanted for the book's sake;
   that is a separate question and this node should not silently answer it.

---

## §8. The located hooks, for when the fork is written

Verified by reading; line numbers are `../enharmonic/enharmonic.py` at its `en_s9` head.

| what | where | note |
|---|---|---|
| the record on a **fired** macro call | `PerfExecutor.apply`, l. 1365–1445 (the fired+metered branch) | `got` (the head's emission, the realisation) and `tgt` (the DP's) are both already in hand before `_store` is called. The record is free. |
| the record on a **closed / playback** macro call | `PerfExecutor.apply`'s `not fired` branch, l. ~1394 | goes through `PlainExecutor.apply` -> `MC.apply_any`, which does not expose the choice. `macro_apply_rec` (l. 4421) is the byte-identical recording copy and **gate E-7 already asserts that identity**, so the switch is a one-line branch, not a new path. |
| the record on the **fire-only** (probe) path | `PerfExecutor._fire_only`, l. 1342 | `_store` is called *before* the emission; a record column needs the two lines swapped. Probe-phase rows have no verdict and should be left unlabelled rather than dropped (dropping them changes the held-out split the parity gate reads). |
| the per-row column in lockstep with the buffer | `PerfExecutor._store`, l. 1317 | `self.cred` already does exactly this for `delta_perf`, with the parent's rng draw, bijective held-out code and caps preserved. The record column is that idiom again. **`cred` is kept for `buf` only, not for `hold`** — a record used on the held-out split needs both. |
| the instance handle | `beam_moves` / `beam_moves_prop`, l. 1576 (`beam_moves`) / l. 1694 (`beam_moves_prop`) | `flat = beams.reshape(batch*width, length)`, so row `i` is instance `i // width`. Free, no bookkeeping. |
| the per-tip verdict join | `beam_moves`, the `acc` gather at l. 1628 (and l. 1748 in the port's copy) | the beam already gathers per-trajectory accumulators along `parent` for `tip_dperf` / `tip_bad` / `tip_exe`. A write at (move `k`, beam row `(b,w)`) is candidate `w*n_moves + k`; `top` says whether it was kept, and the same gather carries a row-id column to the tips. Exact, and ~O(budget) int tensors of shape (batch, width). |
| the verdict | `run_arm`, l. 5856 | `solved = tips_flat[succ > 0.5]`; `ps` is the answer's grade per instance. The metering beam (l. ~6227) grades the **answer only** — half the captured rows come from it. |
| the class map at the write | `quotient.py::Quotient.id_of` / `merge.py::LearnedQuotient.id_of` | memoised per (level, tuple); the endogenous class id is a tuple. |
| the corridor head's objective | `native/span/span_net.py::span_train_terms` and the fork's `perf_span_train_terms` (l. 1454) | targets are recomputed from the current executor at training time and are not stored. V-2 (the new path with the target switched back to `dp_features` reproduces `span_train_terms` bit for bit) is the gate that keeps this honest. |
| the generator's in-run level-1 target (`ear_record`) | `finetune_generator` l. 1855 and `finetune_generator_span` l. 1916 | `feats = bottom_map[(leaves * powers).sum(-1)]` in both. §4 says the only code a macro write can reach where this is wrong is 39, and there it is wrong in a way the token class cannot see — so `ear_record`'s effect on this draw is on the *feature* the generator is taught, not on the class. |

---

## §9. Gates

| gate | what | status |
|---|---|---|
| **VQ-1** | the operative book rebuilt from the logged `picks` has the run's own row count, at every level of every logged build | **PASS** — 606 / 628 / 591 builds on `en_s9:endo_ledger_open_ung5_ra`, `en_s8:endo_ledger_open_ung5`, `en_s8:endo_ledger` |
| G-F | `fidelity_smoke` retargeted at `enharmonic.py`, composed arm, 0.000e+00 with every knob off | Q1 — no fork yet |
| V-1 | the stored intent equals what was rendered, per block, asserted on the smoke | Q1 |
| V-2 | the new training path with the target switched back to `dp_features` reproduces `span_train_terms` bit for bit | Q1 |
| V-3 | the inherited E-0 (singleton classes = flat) and E-7 (knobs absent vs False) still pass on the fork | Q1 |

The discipline inherited from `embouchure` DESIGN §11 applies and is why VQ-1 is a row-count
assertion and not a tensor equality: **assert an identity only where the substrate is
deterministic**; the rebuild is deterministic, the row *identity* under a bound cap depends on the
run's own `self.spell` counts, which the compact log carries only through `picks` — so the row
count is asserted and the identity is inherited from `picks` being the run's own record of the
pick.

---

## §10. Corrections and withdrawals kept beside

- **The two collisions** (§4). Withdrawn: that both collisions reach the learner's own writes.
  Correction: only code 39 does, and there the token class cannot see it. Kept: that the generator
  and the reader are both students of a lossy inverse map, which is `embouchure` §12's finding and
  is untouched — what changes is what that costs *at the class* on this draw.
- **"The chooser is the DP"** (§3). Not withdrawn — it is where the target comes from — but
  narrowed: by era 3 the DP writes 4% of macro rows and the head writes the rest, at a misfire
  rate of 5–7%. `enharmonic` finding 5's mechanism reaches the executor through the imitation
  chain.
- **The offline ceiling** (§6). The SPEC's form is not available; the substitute is a table-side
  bound and is reported as a different quantity, not as the thing asked for.

---

# Q1 (2026-09-12, after the coordinator's revision)

The revision is appended to [`SPEC.md`](SPEC.md) under "Q1, revised"; that file is the record
and this section is the implementation's half of it.

## §11. The arms, and the one confound that rides with an axis

| arm | write | head's objective |
|---|---|---|
| `voi_dp` | free-run argmax | the donor's self-imitation of `dp_features` (its own function, called) |
| `voi_own_record` | free-run argmax | the record-verdict calibration |
| `voi_xp` | sampled, on-table | the donor's |
| `voi_own_record_xp` | sampled, on-table | the calibration |

All four are `endo_ledger_open_ung5_ra` — `en_s9`'s arm character for character — plus at most
two knobs, all four twin onto `enum_live`'s stream, and `vo_record` is on in all four (the
record and its instruments are unpriced readouts; the anchor has to carry them or the arms are
not comparable on them). `voi_dp` names no consumer of the record, so it is the banked arm and
doubles as the in-tag identity check and the full-scale half of G-F.

**The confound, stated because it is real and uniform.** On-table execution rides with the
*sampling* axis: the argmax arms keep the head's free run, which leaves the table on the 5–7%
of calls Q0 §3 measured as its misfire rate, while the sampled arms choose among the book's
rows by construction. Making on-table execution its own axis would have needed a fifth arm.
Because it rides with one axis in **both** objective rows, the row contrast (donor vs
calibration) is clean at each write regime, and the projection component is logged apart from
the sampling component: `n_ontable_shift` (the on-table argmax is not the head's free-run
emission) against `n_xp_shift` (the sampled entry is not the on-table argmax), per (level,
node) per cycle in `log["vo"]["xp"]`.

## §12. Exploration at the write — three decisions and a defect the smoke exposed

**(a) On-table, and not by preference.** The coordinator left the open-slot mechanism to me and
named the alternative as closer to the question. It is also the only one that works here, and
the reason is a number: an L5 span is 16 blocks, so a temperature that moves top-1 share from
0.93 to ~0.7 perturbs ~2% of blocks, and a single flipped block is essentially never another
row of the book — two rows of one class-pair key differ across a whole half. Per-block sampling
would put the write off-table on nearly every deviation, its class **unnamed**, and the
calibration term would have nothing to reinforce. So `vo_head_scores` scores every row of the
operative table through the head's own autoregressive log-likelihood and samples among them.

**(b) It is one batched MLP, not R forwards.** The autoregressive state at block j is
`span_state + Σ_{j'<j} feat(t_{j'} + v·j')`, and the second term depends only on the TABLE — one
embedding lookup plus a cumsum per call, chunked over R (`vo_chunk`) to bound the
`(B, R, span, dim)` intermediate. Gate VO-5 measures it against the naive per-row teacher-forced
loop at max|Δ| = 9.5e-7 (measured, not asserted: different summation order).

**(c) The closed slot samples the DP's own scores.** `vo_dp_scores` is `macro_features_pick`
stopped one line before its argmax; gate VO-4 asserts that argmax **is** `dp_features`. So the
closed-slot treatment is the donor's own chooser with the argmax replaced and nothing else.

**(d) THE DEFECT THE SMOKE EXPOSED, and the fix.** Both score functions SUM over the span — the
head's log-likelihood over `span` blocks, the DP's max-sum over `span` logits — so the spread
between rows grows with the span and **one temperature cannot serve a span-2 slot and a span-16
slot at once**. At a fixed T the same knob would be near-inert at L2 and near-uniform at L5, or
the reverse. The fix is to sample on the per-BLOCK score, `scores / span`, before the
temperature. This is not an invention: it is the file's own convention for exactly this
quantity — `vo_train_terms` divides by `span` for the same reason, and that division is what
makes gate V-2b equal `F.cross_entropy`. The argmax is unchanged by a positive constant, so
VO-6 still reads as an identity.

**(e) WHAT THE SMOKE COULD NOT SIZE, said plainly.** The sizing rule is "top-1 share at L4/L5
falls to roughly 0.6–0.8". No smoke can read it: **an L4 or L5 slot does not exist at any scale
short of the full ladder** (in `en_s9` they are minted at c151 and c186 and the L5 slot opens at
c192), and the preflight's per-cell write counts are 1–6, so a top-1 share there is 1.0 by
construction. What the preflight CAN read, now that the score is span-normalised, is the
deviation rate `shift / n` per level — the quantity the top-1 target is a proxy for — and
whether one temperature now gives comparable deviation across levels. The absolute rate still
does not transfer, because the preflight's plant is trained for forty steps and a sharper plant
deviates LESS at fixed T. So the temperature is **set from the smoke's per-level curve and an
argument, not sized on the target quantity**, and the target quantity is logged per (level,
node) per cycle (`log["vo"]["var"]` top-1 share, `log["vo"]["xp"]` deviation) so the reduction
reads it and a re-launch at another T is one flag.

## §13. The objective

Per filed row, with recorded write `w`, its class `C = {rows of the operative table sharing
key(w)}`, and verdict `y`:

```
p_C   = Σ_{t ∈ C} P_head(t)              the head's JOINT likelihood of the class
term  = −log(p_C) / span                 if y = 1   (reinforce)
      = −push · log(1 − p_C) / span      if y = 0   (push away)
```

averaged over the slot's rows, then over slots — `span_train_terms`' own shape.

**Why the joint and not a table-normalised share.** A share over the book leaves the head's
absolute emission unconstrained, and with the self-imitation target replaced there would be
nothing anchoring it to emitting a legal tuple at all. The joint anchors it: raising `−log p_C`
drives the head toward *emitting a spelling of C*, which is on-table by construction.

**Why `/span`.** With `|C| = 1`, `y ≡ 1` and `push = 0` the term is then `F.cross_entropy` over
the span's blocks under the same teacher forcing — the donor's loss, generalised from one target
to a set. That is gate V-2b, measured at |Δ| = 0.000e+00 (measured rather than asserted:
log-sum-exp against a direct sum is a different summation order).

**The class is the learner's own.** `key(T) = (quot.id_of(left half, ℓ−1), quot.id_of(right
half, ℓ−1))` — `ClassMiner.observe`'s key, computed from the write, equal by construction across
the rows one at-support key emitted, and reading no oracle. The token class appears only in
instruments. Gate VO-3 asserts that under an unmerged map this is the flat half-pair.

## §14. The join, and what the smoke measured

**Per-write, not per-instance.** A write is filed only if the beam KEPT it and its trajectory
survived to a graded tip; the beam's own `parent` gathers carry a write-id column exactly as
they carry `tip_dperf`. The cheap instance-level bag label was considered and rejected with a
number: `_store` subsamples 24 of ~704 parent rows and only ~1/n_moves of writes are kept, so an
instance label would put a verdict on writes that were never executed — y independent of the
write given the context, which is noise on precisely the ordering the arm exists to learn. The
per-write join delivers Q0 §2's ~128 macro rows per slot per cycle in era 4, all labelled.

**Gate V-1 held on every filed write of every arm**: `verify_bad = 0` of 1,653 / 1,277 / 1,462 /
1,040 rows (`voi_pf_dp` / `voi_pf_own` / `voi_pf_xp` / `voi_pf_own_xp`), asserting on every cycle
(`vo_verify_cycles` is infinite in preflight). G-F is **0.000e+00 on both arms with commits
equal**, donor self-replay control also 0.000e+00.

**One reading the smoke gave for free, and it is the argmax arms' own hazard.** `n_unnamed`
counts training rows whose recorded class has no member in the current table. It is **7,025 on
`voi_pf_own` and 0 on `voi_pf_own_xp`**. The mechanism is (a) above from the other side: under
the free-run argmax the head's own emission leaves the table, and an off-table write's key is a
singleton no table row shares — so the record is unnamed and the row drops out of the term.
Under sampling the write is a table row by construction. At preflight scale the head is
untrained so this is extreme; Q0 §3's 5–7% misfire rate is the estimate for the real run. It is
logged per cycle, and it means `own_record`'s labelled set is strictly thinner than
`own_record_xp`'s by the head's own off-table rate — a property of the arm, not a defect, and one
the 2×2 will read.

## §15. Withdrawn for this draw, with the reason beside it

`own_recall` and `ear_record`, on the coordinator's decision and for §4's reason: on the token
class key the record and the recall of a canonical write agree **by construction** on this draw
(code 12 is unreachable by any write; code 39's two writers render alike, which is the same fact
that puts them in one class), so `own_recall` had no structural contrast and `ear_record`'s only
reachable mislabel is within-class and invisible to the grader. Neither is deleted from the
question — both are withdrawn *for `rule_seed 0`*, and a collision-free or differently-colliding
draw would restore them.

What replaced them is an instrument in every arm: the frozen reader run on the learner's own
written span **in place**, quotiented through the arm's own class map and tallied against the
record's class per (level, node), never summed with the verdict (`log["vo"]["readback"]`:
`class_agree`, `tuple_agree`, `unnamed`, with denominators). §4 says this number is measured
nowhere in the parent tag; it decides whether Q2's `own_readback` has anything to read.

## §16. Instruments, and one collapse

All four are on in every arm, oracle-contained, counted in `_EXP_REC["reads"]` and never in
`counts["ground"]`, and all run on the **priced practice beam** — the population the executor
actually writes into, which `en_s5`'s instrument never reached (it hooks `audition_macro`, always
a DP path).

| tally | what |
|---|---|
| `var` | Q0 §5's table, live: filed writes, distinct tuples, distinct class keys, distinct token classes, top-1 share, per (level, node) per cycle |
| `contains` | the written row's token class holds the clean latent (free — the beam's own draw already computes it) |
| `rep` | it holds some feature that repairs the instance THERE, subsampled to `vo_rep_n` rows per cell per cycle |
| `readback` | §15's in-situ read-back |
| `xp` | the sampler's own per-level deviation and projection rates |

**The collapse, stated.** The SPEC asks separately for "the head's class accuracy against the
repair set on the held-out split". On an open slot the writer **is** the head, and the practice
beam's instances are freshly drawn every cycle (`seed + 100_000 + 1000·cyc`), so `rep` already
*is* that number on held-out data, for the actual writer. A second instrument on the audition
pool would have measured the DP instead. `vo_head_aud` was specced, found redundant, and removed
rather than left as a dead knob.

## §17. The temperature, set (2026-09-12)

Two preflight passes, the two sampled twins only, `--vo-explore-t 0.25` and `0.80`, read on
`xp_by_level` — `shift / n`, the share of sampled calls on which the sampled entry left the
on-table argmax. `verify_bad = 0` on all four arm-runs, so V-1 still holds under the span
normalisation.

**First, the defect of §12(d) is fixed.** Deviation is now comparable across levels instead of
scaling with the span. `voi_pf_xp`, per level (min–max over that level's nodes):

| T | L2 (span 2) | L3 (span 4) | L4 (span 8) | L5 (span 16) |
|---|---|---|---|---|
| 0.25 | 0.520–0.672 | 0.379–0.434 | 0.437–0.455 | 0.463–0.477 |
| 0.80 | 0.773–0.840 | 0.517–0.614 | 0.552–0.599 | 0.664–0.667 |

The whole spread across four levels is ~0.2 at each temperature. Without the `/span` division
these columns would have been orders of magnitude apart.

**T = 0.25 is the value of record**, on four grounds, of which only the first is a measurement:

1. At T = 0.25 the preflight's L4/L5 deviation is 0.44–0.48, i.e. a top-1 share of ≈0.52–0.56 —
   already at or just below the target band's floor **on a plant trained for forty steps**.
2. **The sharper-plant argument**, which is the reason the preflight number is a floor and not
   an estimate. The sampler competes with the gap between table rows in per-block score. A
   production plant is far sharper than a preflight one — in `en_s9` the corridor's parity is
   0.92–1.00 at L5 within six cycles of its slot opening, and the executor's top-1 share is
   0.93–0.996 at L4/L5 — so the same temperature deviates **less** in the real run. The realised
   rate therefore moves *up* the band from 0.52, not down through its floor.
3. **The two failure modes are not symmetric.** Too cold wastes the write axis: `xp` ≈ `dp` and
   the column contrast carries nothing. Too hot vandalises the executor — `intonation`'s own
   measured negative is an untrained head at +0.40–0.51 e, and `native` finding 4 put real
   fallible heads at parity 0.76–0.94, "real error, and the majority of executions still
   correct" — and it additionally destroys the paired `own_record_xp` arm's ability to say
   anything about the objective, because a vandalised executor confounds the objective row.
4. **The hot end compounds, measured.** The calibration objective flattens the head's on-table
   distribution (it drives `p_C` toward the class's own solve rate rather than toward 1), so the
   same temperature deviates *more* on the calibration arm: at T = 0.25 `voi_pf_own_xp` sits at
   0.81–0.91 at L3/L4 against `voi_pf_xp`'s 0.38–0.46, and at T = 0.80 it reaches **1.000** at
   L5 — near-uniform, which is the vandalism case. The anchor temperature has to be chosen
   against the arm that flattens, not the one that does not.

**What is still not sized, said again.** This is a temperature *set from a per-level curve and an
argument*, not one sized on the target quantity, because no smoke can read a top-1 share at
L4/L5 (§12(e)). The target quantity is logged per (level, node) per cycle — `log["vo"]["var"]`
carries the realised top-1 share and `log["vo"]["xp"]` the deviation, from the first sampled
cycle — so the reduction reads it, and a re-launch at another T is one flag on the same script.

One thing seen and not read: `ontable_shift` sits near 0.50 almost everywhere at preflight
scale, i.e. the on-table argmax differs from the head's free run on half the calls. That is a
forty-step head, not a reading; Q0 §3's 5–7% misfire rate is the estimate for the real run, and
the column is logged per cycle so the projection component stays separable from the sampling
one (§11).

## §18. What Q1 returned, against what this file predicted (2026-09-12)

Facts are in [`FILES.md`](FILES.md) and `figures/vo_s1_reduction.txt`. This section keeps only
the corrections to diagnoses made *in this file before the run*, as the SPEC's norm requires.

**Confirmed, and by a wide margin: §17's sharper-plant argument.** It said the preflight's
deviation rate was a floor and that a production plant would deviate *less* at fixed T. It does:
`voi_xp`'s realised L2 deviation is **0.091** against the preflight's 0.520–0.672 at the same
temperature — the direction was right and the magnitude was badly understated. T = 0.25 is
therefore too COLD in the real run, not too hot: the write axis on `voi_xp` moved the executor
much less than intended. The projection component `ont` is **0.004**, against ~0.50 at preflight
scale and against Q0 §3's 5–7% estimate — so §11's stated confound is, in the event, negligible.

**Confirmed: §12(d)'s span defect and its fix.** Deviation stayed comparable across levels in
the real run rather than scaling with the span.

**Confirmed, at production scale: §14's `unnamed` hazard, and its size.** 17,540 unnamed training
rows on `voi_own_record` against 428,980 filed — **4.1%**, inside Q0 §3's 5–7% misfire band, and
0 on every sampled arm. The preflight's near-total unnamed rate was a forty-step head, as stated.

**Confirmed live: Q0 §4's structural prediction.** In `voi_own_record_xp`, era 2, L3, the read-back
tally reads `class_agree = 1.0000` with `tuple_agree = 0.8533` on 450 rows — the reader returned a
*different tuple in the same class*, which is exactly the code-39 mechanism (`canon[2] == canon[7]`)
showing up in a run. Everywhere else the two columns are equal, as §4 said they must be.

**WITHDRAWN: the choice to REPLACE the self-imitation target rather than compose with it.** The
SPEC licensed "replace", I flagged the risk in §13 before launching ("the head has no signal
telling it to stay on-table"), and I launched anyway. The mechanism is now exact and quantitative
rather than a worry: `BCE(p_C, y)` at a filed solve rate of 0.175–0.225 drives the head's class
posterior to **0.12–0.26** — the base rate, which is what a calibration objective is *for* — and a
head whose posterior on its own chosen class is ~0.22 does not reproduce the DP exactly on half
its spans. Measured parity is **0.20–0.33** against the anchor's 0.87–0.94, below the 0.50 firing
gate, so the corridor never opened (max 2 slots, 0 at the end) and misfire sat at **0.54–0.56**
against the anchor's 0.068. Both calibration arms therefore ran with the DP executing every macro
call, and neither tested the question the node asks. The composed form (donor term **plus**
calibration term) is the alternative I named in my own notes and did not take; it is the obvious
Q2 candidate, and this is a correction to my design rather than a finding about the hypothesis.

**A measured defect of my own instrument, not of the substrate.** `vo_instruments` spends the
`vo_readback_n` budget in dict-iteration order, so the low-level cells consume it and **L4/L5 were
never sampled**. The number §15 says decides whether Q2's `own_readback` exists is therefore
measured at L2 (class agreement 0.890–0.992, `unnamed` 0.000) and at a trickle of L3, and not at
the frontier. The fix is to allocate the budget per cell rather than first-come; it is one line,
and it belongs to Q2 rather than to a re-run of Q1.

**Not withdrawn, and not confirmed either: the question.** `voi_xp` kept a healthy corridor (16
slots open, misfire **0.042**, *below* the anchor's 0.068) and still did not climb past its L2
commit. So variation at the write neither vandalised the executor nor moved the ladder at this
temperature — which, given the realised deviation of 0.091, is a reading about T and not yet about
exploration. Where the ladder stalled is the commit owner's latch, and the pacer is explicitly not
this node's to touch.

---

# Q2 (2026-09-12/13) — the critic, and exploration at the frontier

The revision is appended to [`SPEC.md`](SPEC.md) under "Q2, revised"; that file is the record and
this section is the implementation's half.

## §19. The two mechanisms Q1 left, and what each one forces

**(a) A distribution over what to write cannot also hold how likely each class is to solve.**
§18's withdrawal, restated as the design constraint it is. `BCE(p_C, y)` on the head's normalised
emission drove the mass on the written class to the base rate and the rest off-table. The
coordinator's reading, adopted: this is about SHAPE, not weight — the first quantity must sum to
one over the alphabet and the second must not, so a composed loss only puts a weight on the
conflict. The verdict therefore moves off the emission head entirely and onto a **critic**, which
is the organ π already has.

**(b) Exploration below the frontier costs the commit window above it.** The coordinator found the
second stall's mechanism in Q1's log: every era advance in every arm fired on the cap, and the
commit owner reads the era's own level, so L3's window was era 2 and closed at c110. The anchor
committed at c100; `voi_xp`, deviating on 9% of its L2 writes, missed it and no L4 or L5 slot ever
existed for it. The ladder's commit windows are fragile to any early perturbation of the mining
stream. So the explorer may only touch the **highest adopted level** — which is also the only
place the chooser was constant in the first place.

## §20. The critic

**What it reads**: the pooled context the head reads, plus a candidate tuple's level-1 features —
content, never a label. `span_net`'s identity/corridor principle: a candidate it has never seen is
still describable, and a table that moves under it does not invalidate it.

**What it emits**: one logit per candidate, P(solve). **A class's value is the max over the
spellings the book holds for it**, because the executor may write any of them and will write the
best one it can spell.

**How it is trained**: `BCE(critic(context, the candidate that was written), verdict)` on filed
rows keyed by the record, in the donor's same optimizer step — a treatment buys no extra steps.

**How it governs**: on a slot it has earned (`vo_critic_min` filed rows), it picks the CLASS over
the on-table candidates and the spelling within it is chosen as now — the head's own on-table
likelihood on an open slot, the DP's per-entry score on a closed one.

**Its gradient is confined** (`vo_critic_trunk`, default off) and this is a decision, not an
oversight: Q1 is the measurement of what happens when a base-rate objective reaches shared
parameters. The knob exists so the other choice is one flag and on the record.

**Governance and exploration have deliberately different scopes.** Governance acts in both PRICED
beams, the metering beam included, because that is what `e` is measured on and an arm whose critic
chose only where nobody was looking would report the donor's error. Exploration is practice-only.
Probes are phase `probe` and every audition goes through `MC.apply_any`, so neither is touched.

**The head's target on a governed slot becomes the record** — what the executor actually wrote —
so parity is against the executor's own choice and the firing gate keeps its meaning. No verdict
filter: this is imitation of the executor, not of the successful executor; the verdict lives in
the critic.

## §21. Exploration, ε-greedy and frontier-only

On the highest adopted level only, in the priced practice beam, a share ε of macro calls write a
uniformly drawn on-table class, recorded exactly. **There is no temperature to size** — Q1's §17
problem was that the score scale is not knowable before the run that produces it, and ε removes
the question: the realised deviation is ε by construction. Measured on the preflight at ε = 0.3:
**0.18–0.35 per cell**, and the shortfall from 0.30 is explained, not fudged — a uniform class
draw can land on the class the argmax would have written anyway.

## §22. What the gates caught, in order, and the one that caught nothing

Seven defects this round, all mine, none in the treatment. They are listed because the pattern
matters more than any one of them.

| # | defect | caught by | cost |
|---|---|---|---|
| 1 | a patch script whose 4th hunk failed, so its first three silently never wrote | traceback (`NameError: closs`) | 1 preflight |
| 2 | a local `_adopted` shadowing `run_arm`'s existing closure | traceback, c1 | 1 preflight |
| 3 | the critic drew its batches from `span_rng`, the head's dedicated stream | **V-4**, max\|Δ\| 9.19 | 1 preflight |
| 4 | `vo_rec.governed` did double duty, so the head's record target leaked into an arm whose target is `dp` | **V-4**, max\|Δ\| 9.00 | 1 preflight |
| 5 | the critic's term added before the donor's span term, so `gloss` was short by exactly `lam·sterm` | **V-4**, max\|Δ\| 2.0187 | 1 preflight |
| 6 | **V-4b tested nothing** — random leaves are off-grammar, `bottom_map` returns −1, the function's own `keep.sum() < 8` skipped every step, and the gate compared two untouched copies | asking whether my own gate could fail | — |
| 7 | the frontier was resolved after the beam that used it, so ε fired one level low on transition cycles (3 of 20) | the ε-per-level readout | — |

**The rule adopted from #6, and it is the one worth keeping**: *a new gate is not reported until
it has been shown to fail on a deliberate perturbation of the thing it claims to protect.*
`embouchure` DESIGN §11 says assert only where the substrate is deterministic; this is the other
half — **assert only where the assertion is reachable**. V-4b now carries its own non-vacuity
assertion (the step loop must have run, the span term must exist, the critic term must have been
computed) and was verified to fail under two perturbations, one of which reproduces defect #5 to
the digit (`d_gloss` = 2.0586, the span term itself).

**The rule adopted from #3/#4/#5**: the in-substrate identity gate is the right claim but the
wrong loop. Three remote preflights at ~25 minutes each found what a 3-second CPU
parameter-and-logged-value check finds directly. V-4 stays — it covers what V-4b cannot — but it
is no longer the first place a perturbation gets caught.

## §23. What Q2 returned, against what this file predicted (2026-09-13)

Facts in [`FILES.md`](FILES.md) and `figures/vo_s2_reduction.txt`. Only the corrections to
diagnoses made *in this file before the run* belong here.

**CONFIRMED, and it is §18's correction landing: moving the verdict off the emission head keeps
the corridor.** Q1's calibration arms ended with 0 open slots and a 0.54 misfire that no
re-decision explained. `voi2_critic` ends with 28 open slots and, once its own re-decisions are
removed, an excess misfire of **0.018** — below the anchor's 0.068. The head kept a proper
imitation target and the firing gate kept its meaning, which is exactly what §20 was for.

**CONFIRMED: the critic learns something.** Held-out AUC 0.63–0.70 per level on 730–830 rows per
slot at base rates 0.13–0.24, 84–88% of reads above chance. §20's "content, never a label" input
is sufficient to predict the verdict above chance from (context, candidate).

**CONFIRMED: the chooser is no longer degenerate.** Q0 §5's finding was that the executor writes
one class on essentially every frontier call. Under the critic the L4 cells write 1.0–3.9 distinct
tuples at top-1 0.49–0.87, against the anchor's 1.0–2.1 at 0.85–1.00.

**WITHDRAWN: that frontier-only exploration escapes Q1 §19's stall.** The SPEC's reasoning was
that the highest adopted level is the one place the chooser is constant, so perturbing it costs
nothing. The premise is true and the conclusion does not follow, and the run says why: **the
highest ADOPTED level is also the level that feeds the level being EARNED.** While L3 is being
earned the frontier is L2, so "frontier-only" ε is ε on exactly the mining stream L3's commit
window depends on — the same mechanism that stalled `voi_xp` in Q1, relocated but not removed.
Both ε arms committed L2 and nothing above, and the frontier never left L2 in either
(`c49–c175`, `c49–c117`), so ε never once acted at the level the node is about. Whatever
exploration this substrate can afford, "at the frontier" is not the same constraint as "not in
the stream something else is earning from", and it was the second that Q1 §19 actually identified.

**A LIMITATION OF THE READOUT, not of the substrate.** `n_sampled` / `n_xp_shift` count
re-decided calls on open and closed slots alike while `n_fired` counts only calls the head
executed, so the misfire decomposition in reduction section [M] is a lower bound and is printed as
`n/a` where the two cannot be reconciled (`voi2_critic_xp`). Splitting the shift by path is a
one-line recorder change; it was NOT made mid-round, because `voicing.py` is the code that
produced `vo_s2` and a local file diverged from the run it has to explain is worse than a coarse
readout.

**NOT RESOLVED HERE, and deliberately left open**: `voi2_critic` reached L3 nineteen cycles and L4
forty cycles earlier than the anchor while its class accuracy against the repair set at those
cells FELL (4n0–4n3, era 4: 0.32/0.28/0.31/0.20 against the anchor's 0.73/0.88/0.84/0.77), and it
still did not commit L5. Whether a chooser that is worse by the repair-set measure but earlier up
the ladder is a better chooser is the reading, and the reading is the orchestrator's.

**Orchestrator's correction to the paragraph above (2026-09-13).** The four numbers quoted for
"class accuracy against the repair set" at 4n0–4n3 in era 4 (0.32 / 0.28 / 0.31 / 0.20 against
0.73 / 0.88 / 0.84 / 0.77) are reduction [E]'s `contains` column — the chosen class holding the
CLEAN LATENT. The repair-set column (`rep`) at the same cells reads **0.24 / 0.20 / 0.20 / 0.36**
for `voi2_critic` against **0.26 / 0.41 / 0.48 / 0.60** for the anchor (era 3: 0.18 / 0.13 / 0.13 /
0.21 against 0.30 / 0.38 / 0.34 / 0.74). So the critic's frontier choices are worse by the honest
measure too, but by less than the paragraph says, and at 4n0 in era 4 they are level. The reading
that follows is the same in sign; the magnitude belongs to `rep`, and the `contains` gap is the
critic spreading its writes over classes the clean derivation did not carry — which the
any-legal-repair grader partly forgives. Kept beside the original per the node's norm.

---

# Q3 (2026-09-13) — the two organs at the choice, and babbling off-stream

## §24. The two mechanisms Q2 left, and what each one forces

Q2 ended on two facts, and Q3's shape is those two facts read literally.

**The critic replacing the DP throws away a strong surface prior.** `voi2_critic` reached L3
nineteen cycles and L4 forty cycles earlier than the anchor, and its class accuracy against the
repair set at those cells was *worse* — `rep` 0.20–0.36 at 4n0–4n3 in era 4 against the anchor's
0.26–0.60, worse in sign but by less than the first reading said, and level at 4n0. A chooser
that discards the DP's ranking outright is discarding a prior that is right more often than the
critic is. So the two are **combined** rather than swapped: on a governed slot the write is

    argmax_t   z( dp_t / span )  +  w · z( critic_t )

over the on-table candidates, each term z-scored over the candidate set, `w = 1`, one knob,
logged. The `/span` is the file's own convention — the same division that makes gate V-2b equal
`F.cross_entropy` and the same one Q1's span-scale defect was fixed with — and the z-scores are
what make `w` read a preference rather than a unit: without them `w = 1` would mean something
different at span 2 and span 16. Gate V-4c asserts all three properties.

**ε on the highest adopted level is ε on the stream the next level is earned from.** Both ε arms
stayed at L2 for the whole run and the frontier never left it. Exploration therefore leaves the
mined beam entirely: it becomes a **priced counterfactual probe** whose rows reach the critic and
nothing else. No ε anywhere in Q3.

## §25. The composed chooser, and what the smoke measured that the design did not expect

`vo_dp_scores_from_logits` was split out of `vo_dp_scores` so the OPEN path can score the DP from
the logits `SN.trunk` already returned; the composed chooser is therefore free on both paths (the
closed path already pays a `block_logits` read, the open path already pays the trunk).

**MEASURED AT THE SMOKE, AND IT IS NOT WHAT "TWO ORGANS" ASSUMES.** The composed choice's
reference is the DP's argmax, and the shift counter's reference is the head's own on-table argmax
(kept at Q2's so the banked `{replace, filed}` cell's series is not silently redefined). The
preflight reports `dp_vs_head = 0` — on **all 24,765** governed open-path calls, the DP's argmax
and the head's on-table argmax were **the same row**.

That is a mechanism, not a defect, and it was checked offline rather than assumed. On an
untrained head the two disagree on 128/128 contexts; after **five steps** of the corridor's own
objective (the head's target *is* `dp_features`) they agree on 126/128, while the head's free-run
emission is still off-table on 128/128. The head learns to *rank* the DP's row first long before
it learns to *emit* it.

So on an open slot there are not two priors to combine — there is one surface prior, read two
ways, plus the critic. This does not change the arm table (`composed` scores the row by
`z(dp) + w·z(critic)`; `replace` picks the class by the critic's max-over-spellings and the
spelling by the head — still different objects), and it does not change the knob. It changes what
`moved` means: it is the critic's whole contribution, not the critic's contribution net of a
disagreement between two priors. The `dp_vs_head` column stays in the reduction precisely so the
run says whether this holds at scale, where the head is trained far past five steps.

**Also measured, and said out loud as a path check:** at `w = 1` the composed choice differed from
the DP's row on **92.5%** of governed calls (22,904 / 24,765). At preflight the critic is trained
on 16-row minima and the DP's ranking is near-degenerate (2 distinct argmaxes over a 64-row table
in the offline replica), so a near-random critic with more spread wins most ties. Nothing is read
from this number; it is here so that if 0.925 recurs at scale, the reading is that `composed` at
`w = 1` is behaviourally close to "the critic decides" and its contrast with `rep_pr` is narrow —
a reading to make *after* the data, not before.

## §26. Babbling off-stream

For each governed slot each cycle, up to `vo_probe_n` filed contexts from the practice beam: take
the trajectory's **final** configuration (solved or not), substitute a different on-table class at
that slot — uniform over the classes the book holds there, excluding the one written — render,
and pay the grader for one verdict. `(context, candidate, verdict)` goes into the critic's
training set and nowhere else.

Two properties are the whole point and both are gated rather than asserted in prose:

**PRICED.** Every probe grading is a grounding on the meter at `d_fb`, inside the cycle's own
ledger, with its own line (`log["vo_bill"]`). Gate V-4B asserts the bill moved by *exactly* the
probe count × unit cost and by nothing else.

**OFF-STREAM.** Nothing it produces reaches the repertoire — not the miner, not the plant's
solved pool, not π's buffer, not the value buffer. Gate V-5 asserts that on the objects
(`n_mined`, the miners' own state, the solved pool, the committed vocabulary), bit for bit,
against the anchor.

**WHAT THE VERDICT ACTUALLY ANSWERS, stated because the reducer's comparison depends on it.** The
substitution goes into the *final* configuration and the rest of the trajectory is the beam's
own, unchanged. So the verdict answers "would this write **alone** have been right, holding
everything else fixed" — not "would the beam have solved it". The probe's solve rate is therefore
comparable to the *filed* solve rate only under that reading, and reduction [O] prints the two
side by side with that sentence attached.

**THE BILL, SIZED ON THE BANKED ANCHOR RATHER THAN ON THE PREFLIGHT.** The preflight's own share
(`n_probe = 8`, dust) is 0.39% mean / 1.14% max and is not a measurement. Projected onto
`vo_s2/voi2_critic`'s actual per-cycle ledger at the run's `n_probe = 64` and its realised
governed-slot count (28 at the end, 140 billed cycles of 188):

| n_probe | mean share | median | max | whole run |
|---|---|---|---|---|
| 16 | 0.0016 | 0.0017 | 0.0020 | 0.0011 |
| 32 | 0.0032 | 0.0035 | 0.0040 | 0.0023 |
| **64** | **0.0064** | **0.0069** | **0.0081** | **0.0046** |

**The ~5% cap does not bind**, by an order of magnitude. `vo_probe_n = 64` as specced.

The preflight's probe solve rate is **0.013** (6 of 451) against a filed solve rate of 0.048–0.069
on the same arms — i.e. a substituted class solves far less often than the written one, which is
the contrast the critic needs and the reason the probe channel is worth its bill. A path check at
preflight scale, not a measurement.

## §27. The deferred recorder change, made

Reduction [M]'s decomposition in Q2 was a lower bound: `n_sampled` / `n_xp_shift` counted every
re-decided call, open and closed alike, while `n_misfire` counts only calls the head executed, so
`shift − misfire` was a subtraction across two denominators and `excess` had to be withheld
wherever `shift > n_misfire`. The recorder now tallies the shift **by path**
(`n_shift_fired` / `n_shift_closed`), and [M] reports

    excess = (n_misfire − shift_FIRED) / n_fired

on one denominator throughout, with Q2's undecomposable form kept beside it so a banked Q2 tag
reduces to the numbers it did (verified: `vo_s2` re-reduces unchanged, and the new columns read
`n/a` there rather than a zero that would say "the treatment never moved a fired write").

## §28. Gates, and the rule that a gate must be shown to fail

Q2's sixth defect was a gate that read green on a path that never ran. The rule adopted then —
**a new gate is not reported until it has been shown to fail on a deliberate perturbation of the
thing it claims to protect** — is applied to every gate this round, and to make the in-substrate
ones falsifiable without paying for a remote preflight, V-4's two forms and V-5 were factored out
of `preflight` into `vo_gate_v4_v5(a, b, form)`, which takes two loaded arm files. A gate welded
into a remote entrypoint can only be perturbed by paying for the entrypoint.

**21 of 21 perturbations behaved as required.**

| gate | perturbations it was shown to fail on |
|---|---|
| **V-4c** (composed scorer) | `w` ignored; the critic not z-scored; the DP not z-scored; no \|C\|=1 guard |
| **V-4d** (probe path) | rows pushed into the FILED buffer; the probe not billed; the probe re-grading the class that was written; an off-table candidate |
| **V-4A** (filed critic, governance off) | one `succ` value moved by 1e-9; the commit cycle moved; the critic never audited (vacuous); the twin billing probes |
| **V-4B** (probes on, governance off) | one `e` value moved by 1e-9; `t_cum` off by 1.0 (an unbilled cost); a probe graded but not counted; the probe never firing (vacuous) |
| **V-5** (off-stream) | the miner state moved; the committed vocabulary moved; probes billed but nothing filed (vacuous) |

V-4d's "a DIFFERENT class" claim is checked **exactly, not on average**: `vo_run_probes` draws its
contexts from its own stream without replacement, so the caller cannot recover which rows were
picked — the gate therefore constructs a case in which *every* row writes the same class, and the
claim becomes a property of the returned candidates alone.

## §29. The Q3 gate table of record (`q3a` + `vo_gf8`), with denominators

| gate | claim | result |
|---|---|---|
| **G-F** | with every `[voicing]` knob off the fork replays `enharmonic.py` | **PASS** — 0.000e+00, donor self-replay control 0.000e+00, commits equal |
| **V-1** | the record renders to what the beam wrote, asserted every cycle | **PASS** — 0 bad of **7,863** filed writes over five twins (1,653 ×3 + 1,452 ×2), `unnamed = 0` |
| **V-4A** | the filed-write critic, built and trained, governance off, head target back at `dp` IS `voi3_pf_dp` | **PASS** — max\|Δ\| **0.0** over 27 cycles, commits equal, 0 probes billed, bill residual 0.0 |
| **V-4B** | the same with probes ON is `dp` on every series *including* `g_per_solve`, and `t_cum` larger by exactly probe count × `d_fb` | **PASS** — max\|Δ\| **0.0**, commits equal, **451** probes, bill residual **0.0** |
| **V-5** | the probe-on twin's `n_mined`, miner state, solved pool and vocabulary are `dp`'s bit for bit | **PASS** — all four equal; 451 rows filed into `pbuf` across 30 slots, and nowhere else |
| **V-4b** | the critic changes no plant/head parameter and no logged value | **PASS** — 0.000e+00 on all four |
| **V-4c** | composed: `w=0` IS the DP's argmax; both z-scores affine-invariant; \|C\|=1 passes; the critic actually moves something at `w=1` | **PASS** — affine residual 3.1e-06, moved 3/7 |
| **V-4d** | probe: on-table, a different class, off-stream, billed row for row | **PASS** — 16/16 rows, 16 billed, 16 filed, 0 same-class draws, filed buffer untouched |
| **VO-8** | rows filed, the composed chooser ran, the probe ran and filed, the bill equals the grading count | **PASS** — composed n = **24,765**; probe **451** / **342** graded = billed, over 20 / 19 cycles, 30 slots each |
| `gates_cpu` | every offline gate | **PASS 17/17** |

`g_per_solve` is deliberately inside V-4's strict set and not treated as a bill line: it is read
off the **metering** beam's own counts dict, not the cycle ledger, so a probe that moved it would
be a probe running inside the metering beam — exactly the failure worth catching.

**The preflight's critic AUCs (0.89–0.91 mean over 9–14 reads) are a path check and not a
measurement**, said out loud for the third round running: they sit on held-out sets of a few
dozen rows at preflight base rates. The first read worth interpreting is this tag's, and it is
reported per slot with its denominator, FILED and PROBE rows apart.

## §30. Defect #8 — the probe channel was built, priced and never consumed

`vo_s3` ran clean, 4.60 GPU-h, no traceback, every gate green. It is still wrong in one axis, and
the axis is the round's second question.

**The defect.** `vo_critic_terms` grew a `use_probe` parameter and `run_arm` built
`vo_obj["use_probe"]` correctly, but the call site inside `finetune_generator_span` never passed
the keyword. It took its default of `False`. So in both probe arms the babbler drew its contexts,
substituted a different on-table class, rendered, paid the grader, billed the meter and filed
into `pbuf` — **198,382 probes in `voi3_rep_pr`, 115,017 in `voi3_comp_pr`** — and not one of
those rows ever entered the critic's loss.

**What that looks like in the data**, and it is unambiguous:

| pair | every behaviour series | `t_cum` | residual against the probe bill |
|---|---|---|---|
| `voi3_comp` vs `voi3_comp_pr` | **0.000e+00** on all 11 | 1.150e+05 | **3.7e-09** |
| `voi3_rep_pr` vs banked `vo_s2:voi2_critic` | **0.000e+00** on all 11 | 1.984e+05 | **0.0** |

Both probe arms are their no-probe twins plus a bill, to the digit, with commits equal.

**Why every gate passed.** V-4A, V-4B and V-5 all assert that the critic and the babbler change
NOTHING **with the critic's governance off**. That is the right claim and each one earned its
keep — but with governance off the critic never chooses, so **its diet cannot reach the run
whether the wiring carries it or not**. The whole inertness half of the gate table is blind to a
disconnected diet by construction. Defect #8 passed V-4A at 0.0, V-4B at 0.0 with a bill residual
of 0.0, V-5 on all four objects, VO-8, V-4c, V-4d and 17/17 `gates_cpu`, and the smoke could not
have caught it.

**An inertness gate needs a liveness twin.** Gate **V-6** is that twin: one critic, one slot, a
filed buffer and a probe buffer whose verdicts **contradict each other** on the same contexts, and
one optimizer step each way — with `use_probe=True` the critic's parameters must move where
`use_probe=False` leaves them, and the training set must actually have grown. Asserted (CPU, one
boolean apart, contradictory diets), with its own non-vacuity half (both steps must have run, and
`n_train` must increase). It reads `d_critic = 5.31e-01`, 117 → 234 rows.

Per the round's rule, it was shown to fail on the thing it protects — **on the actual defect**:
with the keyword dropped at the call site it returns `d_critic = 0.000e+00` and 117 → 117 rows.
The falsification harness is now **24/24**.

**The general lesson, worth keeping past this node.** Every gate in this arc has been an
*inertness* gate: "the treatment, with its knob off, is the control at 0.000e+00." That shape
catches a treatment that leaks. It cannot catch a treatment that is *not wired at all*, because a
disconnected treatment is the most inert thing there is. Any knob whose whole purpose is to change
behaviour needs a gate asserting that it **does**, on the smallest substrate that can carry the
claim, and the perturbation that gate must fail on is *disconnection*.

## §31. What Q3 returned (2026-09-13/14)

Four arms, 164–201 cycles, 4.60 GPU-h. Two of the four cells are invalid as specced (the diet axis
is dead, §30). What the tag does say, said plainly:

**1. The anchor is exact.** `voi3_dp` against banked `en_s9:endo_ledger_open_ung5_ra`:
**0.000e+00 on all thirteen series**, commits equal (L2@48, L3@100, L4@151, L5@186). The composed
dispatch, the probe path, `moved_add`, the split shift counters and the bill series are all inert
on an arm that consumes none of them. That is the full-scale half of G-F for Q3's additions.

**2. The composed chooser stalls the ladder harder than the critic that replaced the DP.** The
`{composed, filed}` cell is valid and it is the round's main negative result:

| arm | cycles | commits | corridor open max/last | misfire |
|---|---|---|---|---|
| `voi3_dp` (anchor) | 201 | L2@48, L3@100, L4@151, **L5@186** | 30 / 30 | 0.068 |
| `voi3_comp` | 164 | **L2@48 only** | 16 / 16 | 0.301 |
| `voi3_rep_pr` ≡ Q2's `voi2_critic` | 188 | L2@48, L3@81, L4@111 | 28 / 24 | 0.619 |

Composing the two organs is *worse for the ladder* than letting the critic replace the DP
outright: `replace` at least reached L4 (and earlier than the anchor); `composed` never left L2 in
164 cycles. Its corridor stayed open at 16 and its misfire (0.301) sits between the anchor's and
`replace`'s — so it is not that the corridor shut. **No interpretation is offered here.** This is
one draw, one seed, and the mechanism is not in this table.

**3. The composed choice's own numbers, and a correction to §25.** At `w = 1`, over **3,820,426**
governed calls, the composed argmax differed from the DP's on **0.330** of them — not the
preflight's 0.925, which §25 flagged as a path check and which the run confirms was one.

And `dp_vs_head` is **0.084–0.160** per slot, **not 0**. §25's preflight reading — that the DP's
argmax and the head's on-table argmax are the same row — **does not hold at scale**, and the
correction is kept here rather than by editing §25. The preflight's 0 was a preflight artifact:
a five-step-trained head against a DP whose ranking was near-degenerate (2 distinct argmaxes over
a 64-row table in the offline replica). With a properly trained head the two priors genuinely
disagree on 8–16% of calls. **The column earned its place**: it was added to keep `moved` from
being read as the critic's whole contribution when it is not, and at scale that is exactly what it
is doing.

**4. The probe channel works and costs what it was sized to cost.** Priced at `d_fb`, per era:

| arm | probes | graded = billed | solved | probe solve rate | filed solve rate | bill mean / max |
|---|---|---|---|---|---|---|
| `voi3_comp_pr` | 115,017 | 115,017 | 4,813 | **0.0418** | 0.1792 | 0.0043 / 0.0046 |
| `voi3_rep_pr` | 198,382 | 198,382 | 7,618 | **0.0384** | 0.1367 | **0.0058 / 0.0077** |

Predicted before launch from `vo_s2/voi2_critic`'s own ledger: 0.0064 mean / 0.0081 max. Realised
0.0058 / 0.0077. **The ~5% cap does not bind, by nearly an order of magnitude**, and the bill grows
mildly with the ladder (e1 0.0008 → e5 0.0073) as governed slots accumulate. A substituted class
solves **3.5–4× less often** than the one that was written, which is the contrast the critic was
meant to be fed.

**5. The measurement the defect accidentally made, and it is the most useful number in the tag.**
Because the probe rows were filed but never trained on, `voi3_comp_pr`'s probe AUC is a clean
**held-out, off-distribution** test of a critic trained on filed writes alone. Per slot, last read,
16 slots, 612–895 held-out probe rows each at base rate 0.023–0.063:

- on **FILED** rows the critic is **above chance** — AUC 0.40–0.78, median ≈ 0.65, and `>0.5` on
  0.77–1.00 of reads at 14 of 16 slots;
- on **PROBE** rows it is **at or below chance** — AUC 0.365–0.624, median ≈ 0.49, and `>0.5` on
  0.00–0.02 of reads at three slots.

A critic trained only on what the beam chose to write **does not rank counterfactual candidates at
all**. That is the argument for the probe diet stated as a measurement rather than as a
motivation — and it is why defect #8 is worth the re-run rather than worth writing off.

**6. An unintended full-scale gate.** `voi3_rep_pr` is bit-identical to banked `vo_s2:voi2_critic`
on every behaviour series across two tags, with only the bill moved and a bill residual of exactly
0.0 against the cumulative probe count. That is gate V-5's off-stream claim confirmed at full
scale, on 198,382 priced probes, across a tag boundary — far stronger evidence than the
27-cycle preflight twin it was asserted on. The defect cost the diet axis; it bought this.

---

# Q3b (2026-09-14) — the diet axis re-run, with the ladder held fixed

## §32. Why the ladder becomes a control, and what the yoke is

Q3 §31 left one structural fact: **no treated arm except Q2's replace-critic has ever been read
at the levels this node is about.** The anchor committed L3 at c100, Q2's replace-critic at c81,
and the composed arm never — its L3 yield read rose *earlier* than the anchor's (0.36 against 0.32
at c70) and then flattened near 0.43 and never satisfied the commit latch before era 2's cap
closed at c110. Every one of those is the latch responding to that arm's own L2 write
distribution. That is Q1 §19's window, now seen through a third distribution, and it means the
chooser's quality at L4/L5 has never been measured — only its effect on when L3 opened.

The lineage's control for exactly this is the clock yoke: `enharmonic` Q1's −0.40 gap was the
clock, and the yokes erased it. So Q3b replays the anchor's realised commits and advances —
**48 / 100 / 151 / 186** and **60 / 110 / 180 / 192 / 201** — in every treated arm, and reads the
chooser at L4 and L5 with the same slots open at the same cycles. The 2×2 is
{composed, replace} × {filed, filed + probes}, all yoked.

**The source is `vo_s3:voi3_dp`, not `en_s9`.** It is the same arm — 0.000e+00 on all thirteen
series against `en_s9:endo_ledger_open_ung5_ra`, commits equal — and it is on *this* node's
volume, which is where `--yoke-from-tag` reads its plan. So `dp` is not re-run and the anchor is
banked twice over. Its logged plan is exactly the nine actions above.

`loop_commit` is **absent** from every yoked arm: a yoke is one policy owning both actions, which
is the donor's own shape (`flat_yk_endo_ledger_open_ung5`). `dsil_bootstrap` is kept for
cfg-similarity with the anchor and is **inert by mechanism** here — its branch requires the commit
owner's `read_key` to be `dsil`, and a yoke policy has no read key at all. `rearm_advance_only` is
**not** inert and is kept because it must be: it addresses the open inventory's re-arm hook, which
fires on the replayed advances exactly as it does on the anchor's.

## §33. Two gates the round adds, and the shape of the hole they close

**Y-1 — the replay matches.** The yoked arm's realised commit and advance cycles must equal the
source's, asserted. If they do not, every level-resolved readout in the round is comparing two
different ladders again and there is no control. `cancelled` actions are **excluded from the match
and reported separately**: a replayed commit whose build is empty is cancelled by the substrate,
not by the clock (`enharmonic` [Y] calls it "empty flat build"), and a treated arm whose book is
thinner than the anchor's at some level will have some. That is a finding about the book and it is
surfaced rather than folded into a pass.

**V-6b — the diet is live, in substrate.** This is the hole defect #8 went through, stated as a
gate. Every identity in this node's table had the same shape — *the treatment, with its knob off,
is the control at 0.000e+00* — and that shape is structurally blind to a treatment that is not
wired at all, because a disconnected treatment is maximally inert. V-6b is the dual: two arms with
the critic's governance **ON**, one boolean of diet apart, which must **not** be identical. The
*size* of the difference is measured and reported, never asserted — how much a diet moves a run is
not something a gate may pin — but that it moves it at all is exact, and it carries a non-vacuity
half (both arms must have governed; the probe arm must have filed rows).

Both were shown to fail before being reported, Y-1 on five clock mismatches and an empty plan,
V-6b on defect #8 itself and on all three vacuity cases. The falsification harness now lives in
the node at `gates/falsify.py` and stands at **36/36**.

## §34. The Q3b gate table of record (`q3b` + `vo_gf9`), with denominators

| gate | claim | result |
|---|---|---|
| **G-F** | knobs off → the fork replays `enharmonic.py` | **PASS** — 0.000e+00 both arms, control 0.000e+00, commits equal. Mandatory this round: the defect-#8 fix touched `finetune_generator_span`, a shared path. |
| **Y-1** | the replay matches the source on every commit and advance | **PASS** on all five yoked twins — commits `[6]`, advances `[6, 12, 17, 22, 27]`, **0 cancelled**, against a 6-action plan |
| **V-1** | the record renders to what the beam wrote | **PASS** — 0 bad of **3,267** filed writes over five twins, asserting every cycle, `unnamed = 0` |
| **V-4A** (yoked) | the filed-write critic, governance off, IS `voi3b_pf_dp` | **PASS** — max\|Δ\| **0.0** over 27 cycles, commits equal, 0 probes, bill residual 0.0 |
| **V-4B** (yoked) | the same with probes on, except a bill of exactly probe count × `d_fb` | **PASS** — max\|Δ\| **0.0**, commits equal, **329** probes, bill residual **0.0** |
| **V-5** (yoked) | the probe-on twin's mined count, miner state, solved pool and vocabulary are `dp`'s | **PASS** — all four equal; 329 rows into `pbuf` across 16 slots |
| **V-6b** | with governance ON the diet **changes the run** | **PASS** — max\|Δ\| **1.0** on `n_solved`, first differing at cycle 9; both arms governed; 116 probes, 116 rows filed |
| **V-6** (CPU) | the diet moves the critic's parameters | **PASS** — `d_critic` 5.31e-01, 117 → 234 training rows |
| `gates_cpu` | every offline gate | **PASS 18/18** |
| falsification | every new gate fails on a perturbation of what it protects | **36/36** |

**Two preflight-scale artifacts, stated rather than papered over.**

1. **The critic's held-out AUC has an EMPTY denominator here** (`reads = 0` on every arm) and is
   therefore *not reported* for this preflight. The cause is scale and is understood: the yoked
   twins commit once (L2@c6 — the source's dust plan), so they file ~45 rows per slot against an
   audit that needs ≥32 rows and ≥16 held out at `hold = 0.2`. The audit path's coverage comes
   from `q3a` (9–14 reads) and from `vo_s3` at full scale (813 held-out probe rows per slot,
   `probe_auc` 0.624), and `vo_critic_audit` did not change between them. In the paid run each
   slot carries thousands of rows.
2. **`moved = 1.000` and `dp_vs_head = 0`**, as in `q3a` and for the same reason (§25, §31.3): a
   forty-step head against a near-degenerate DP ranking. At full scale `vo_s3` read 0.330 and
   0.084–0.160. Nothing here is a measurement.

Neither is in the gate list, and every gate in the list is clean with a non-empty denominator.

## §35. What Q3b returned (2026-09-14)

Four yoked arms, 201 cycles each, 5.1 GPU-h. **Gate Y-1 holds exactly on all four**: commits
`[48, 100, 151, 186]`, advances `[60, 110, 180, 192, 201]`, **0 cancelled**, and every arm
realised L2/L3/L4/**L5** at the anchor's cycles, with lifetimes within 0.3% of the anchor's
ledger (51.14–51.36 M against 51.21 M). So for the first time in this node **every treated arm
has an L4 and an L5 slot**, at the same cycles as the anchor, and the chooser is read where the
node is about. Everything below is a controlled comparison; in Q3 none of it existed.

**1. With the ladder held fixed, the composed chooser is BETTER than the anchor at the frontier
and the replace chooser is not.** `rep` — the written class holds some feature that repairs the
instance there, which on an open slot is the head's own class accuracy on held-out data:

| cell | anchor `voi3_dp` | `comp_yk` | `comp_pr_yk` | `rep_yk` | `rep_pr_yk` |
|---|---|---|---|---|---|
| 4n0 | 0.269 | **0.284** | 0.278 | 0.281 | 0.259 |
| 4n1 | 0.393 | **0.451** | 0.392 | 0.233 | 0.374 |
| 4n2 | 0.368 | **0.489** | 0.307 | 0.302 | 0.404 |
| 4n3 | 0.632 | **0.683** | 0.727 | 0.655 | 0.626 |
| 5n0 | 0.356 | **0.526** | 0.505 | 0.245 | 0.366 |
| 5n1 | 0.691 | **0.830** | 0.702 | 0.689 | 0.702 |

(denominators 468–3,200 held-out rows per cell per arm). `comp_yk` is above the anchor at **6 of
6** cells and by the widest margins at L5. This **reverses the sign of Q2's reading**, which had
the critic's frontier choices *worse* than the anchor's by this measure — and Q2's arm was
self-paced, so what that round was reading was partly its own clock.

**2. The located mechanism.** `moved` — how often the governed write is not the DP's row — at
L4/L5: `comp_yk` **0.026–0.134** (0.461 at 5n1), `rep_yk` **0.598–0.969**. The composed chooser
keeps the surface prior and overrides it rarely; the replace chooser discards it almost always.
The corridor says the same: misfire (era ≥ 3) is **0.284/0.246** on the composed arms against
**0.592/0.818** on the replace arms, with the anchor at 0.068. §24's argument — that a chooser
discarding the DP's ranking discards a prior that is right more often than the critic is — is
what the frontier cells now show, measured rather than argued.

**3. Era-4/5 error, at genuinely matched clocks.** All four arms ran the anchor's 201 cycles with
the anchor's commits and advances, which is the first time this table has meant anything here:

| arm | e1 | e2 | e3 | e4 | e5 |
|---|---|---|---|---|---|
| anchor (`voi3_dp` ≡ `en_s9`) | 0.499 | 0.620 | 0.613 | 0.671 | 0.773 |
| `voi3b_comp_yk` | 0.502 | 0.624 | 0.565 | 0.588 | **0.550** |
| `voi3b_comp_pr_yk` | 0.501 | 0.614 | 0.532 | 0.644 | 0.703 |
| `voi3b_rep_pr_yk` | 0.505 | 0.650 | 0.637 | 0.568 | 0.611 |
| `voi3b_rep_yk` | 0.505 | 0.653 | 0.662 | 0.728 | 0.824 |
| `vo_s2:voi2_critic` (self-paced) | 0.505 | 0.680 | 0.692 | 0.684 | 0.779 |

`comp_yk` is **0.223 below the anchor in era 5**. `enharmonic`'s matched-clock gap between
near-identical arms reaches 0.08, so this is well outside it — but it is one seed and the caveat
in [I] stands.

**4. THE DIET WORKS, and this is what defect #8 was hiding.** The like-for-like comparison is at
L2, the same level, the same arm shape, comparable denominators — `vo_s3`'s composed probe arm
with the diet *disconnected* against `vo_s3b`'s with it *connected*:

| | filed AUC (n) | **probe AUC** (n) |
|---|---|---|
| `vo_s3:voi3_comp_pr` — diet DISCONNECTED | 0.661 (675) | **0.491** (375) |
| `vo_s3b:voi3b_comp_pr_yk` — diet CONNECTED | 0.640 (723) | **0.798** (481) |

A critic fed only what the beam chose to write ranks counterfactual substitutions **at chance**;
fed the counterfactuals, it ranks them at **0.798**, while its discrimination on the filed
distribution is essentially unchanged (0.661 → 0.640). It holds at every rung: probe AUC **0.841 /
0.720 / 0.740** at L3 / L4 / L5 for the composed arm and 0.765 / 0.766 / 0.719 for the replace
arm. Off-distribution discrimination is bought, and bought cheaply.

**5. And yet the diet does not help the arm that needed it most.** The 2×2 interacts: probes make
the **replace** chooser better (era-5 error 0.824 → 0.611; `rep` up at 4 of 6 frontier cells) and
the **composed** chooser worse (0.550 → 0.703; `rep` down at 4 of 6). Both directions are one
draw and no mechanism is offered here. The honest statement is that the probe diet buys
counterfactual discrimination in every arm, and that what the chooser then *does* with it depends
on whether it was already keeping the surface prior.

**6. The bill.** 208,487 and 207,601 probes, every one graded = billed. Share of the cycle ledger
**0.0055–0.0056 mean, 0.0082–0.0083 max**, against the pre-launch prediction of 0.0064 / 0.0081
off the banked `vo_s2` ledger — the ~5% cap misses binding by nearly an order of magnitude, and
the per-era growth is mild (e1 0.0008 → e5 0.0072). A substituted class solves 0.055–0.064 against
a filed rate of 0.229–0.296, so the counterfactual contrast the critic is fed is real and holds at
the full ladder.

## §36. What Q3c returned (2026-09-14) — the seed check, and what it could and could not check

Three arms at `--seed 1` (`rule_seed` unchanged at 0, so the world, the ladder, the caps and
every floor are the ones seed 0 ran on and only the run's own stochastic stream moves). One tag:
the anchor self-paced, then `comp_yk` and `comp_pr_yk` yoked to **its** clock, resolved in-tag by
arm order.

**1. The yoke is seed-robust. Y-1 is exact with 0 cancelled**, on both arms: commits `[47]`,
advances `[60, 77, 87, 99, 108]`, realised exactly, lifetimes 25.70 M / 25.76 M against the
source's 25.70 M. The machinery does what it claims at a seed it has never seen.

**2. THE ANCHOR'S LADDER IS NOT SEED-ROBUST, and this is the round's finding.**

| | seed 0 | seed 1 |
|---|---|---|
| commits | L2@48, L3@100, L4@151, **L5@186** | **L2@47 — and nothing else** |
| advances | 60, 110, 180, 192, 201 | 60, 77, 87, 99, 108 |
| lifetime | 201 cycles / 51.2 Mg | **108 cycles** / 25.7 Mg |
| certificate | L2, L3, L4, L5 | **L2 @ c17, L3 never** |

The run's own log says the mechanism: `1 commits at [47], 5 advances at [60,77,87,99,108]
(3 chosen / 3 capped)`. The commit owner reads `yield` and never licensed L3; delta-silence, which
owns the ADVANCE, went quiet and advanced anyway. So the arm walked eras 2 → 5 in 48 cycles
without earning a single rung above L2 and the run ended at c108. That is Q1 §19's mechanism in
its most extreme form and **at the anchor**, not at a treated arm.

The consequence is direct and must not be softened: **the L4/L5 frontier panel and the era-5 gap
of §35 cannot be checked at seed 1, because at seed 1 no arm — the anchor included — adopts L4 or
L5 at all.** [Z] reads "no L4/L5 cell was ever written" for every arm. The headline magnitude
(0.223 in era-5 error) rests on a rung that exists at one of the two seeds tried.

**3. What CAN be checked replicates, in sign and in relative size.** At the levels that exist at
both seeds, pooled class accuracy against the repair set, era ≥ 2, on held-out beam instances:

| | anchor | `comp_yk` | `comp_pr_yk` |
|---|---|---|---|
| seed 0 (n ≈ 185,000/arm) | 0.2141 | **0.2538** (+18.5% rel) | **0.2809** (+31% rel) |
| seed 1 (n ≈ 48,000/arm) | 0.1076 | **0.1340** (+24.5% rel) | **0.1370** (+27% rel) |

Per cell at seed 1 the composed arm is above the anchor at 9 of the 10 L2 cells. The absolute
level is half of seed 0's because the whole substrate is worse there — but the ordering and the
relative gain are the same at both seeds.

**4. The era-3 gap replicates almost exactly.** Era-3 error, treated minus anchor: **−0.048** at
seed 0 (0.565 vs 0.613), **−0.052** at seed 1 (0.829 vs 0.881). Era 4 and 5 at seed 1 are not a
comparison — both arms are running the same unearned ladder there (anchor 0.939 / 0.960; `comp_yk`
0.927 / 0.958), and the whole table sits 0.2–0.3 above seed 0's because nothing above L2 was ever
adopted.

**5. One nuance the seed pair sharpens.** §35 reported that probes *hurt* the composed chooser.
That was specific to the L4/L5 cells at seed 0: pooled at L2/L3, `comp_pr_yk` is above `comp_yk`
at **both** seeds (0.2809 vs 0.2538; 0.1370 vs 0.1340). So the interaction §35 flagged is not
"probes hurt composed" in general — it is something that happens at the frontier cells, on one
seed, and is now known to reverse at the lower rungs on both.

**What the seed check therefore licenses.** The SIGN of the composed chooser's advantage is
replicated: it beats the anchor by the repair measure at both seeds, at every level where a
comparison exists, and its era-3 error gap reproduces to within 0.004. The SIZE at the frontier
is not replicated and not refuted — it is untested, because the frontier does not exist at seed 1.
Any further claim about L4/L5 needs either more seeds that happen to climb, or an intervention on
why the anchor's commit latch fails to license L3 at seed 1 while its advance policy keeps
walking. That second question is now the more load-bearing of the two, and it is not this round's
to answer.

## §37. What Q3d returned (2026-09-14) — the seed check on `en_s9`'s own L5 commit

Two more draws of the anchor character alone, at `--seed 2` and `--seed 3`, two launches so the
clean probe pool moves with the seed exactly as it did at seeds 0 and 1. No code changed.

| seed | cycles | ladder reached | advances |
|---|---|---|---|
| **0** | 201 | L2@48(13) L3@100(61) L4@151(136) **L5@186(128)** | 0 quiet / 5 cap |
| **1** | 108 | L2@47(14) — and nothing else | **2 quiet** / 3 cap |
| **2** | 201 | L2@59(13) L3@77(46) **L4@157(152)** | 0 quiet / 5 cap |
| **3** | 201 | L2@52(15) L3@66(**4**) | 0 quiet / 5 cap |

**1. L5 replicates at 1 of 4 seeds. L4 at 2 of 4. L3 at 3 of 4.** The rung this node's frontier
result sits on, and that the `enharmonic` addendum leans on, is a one-in-four event at this
configuration. That is the round's finding and it is not softened by anything below.

**2. Seed 1's early advance is a real failure mode but NOT the main one.** Seeds 0, 2 and 3 all
ran `0 quiet / 5 cap` — every era to its full window, 201 cycles. Only seed 1 had delta-silence
declare quiet early (era 2 at 17/50, era 3 at 10/70). So §36's diagnosis stands as a description
of seed 1 and is withdrawn as an explanation of the L5 fragility in general: **seeds 2 and 3 got
the entire schedule, every era at its cap, and still did not commit L5.** Whatever stops L5 is
not the advance policy walking early.

**3. What does stop it is visible in the entry counts and the era budget.** Era 4 is 12 cycles
and era 5 is 9. Seed 3 committed L3 at c66 with a **4-entry** table (against 61 at seed 0 and 46
at seed 2) and never built an L4 candidate worth committing — its era-3 A4 swings 0.195–0.727 and
lands at 0.246, a gauge that never settles. Seed 2 built a healthy L4 (152 entries, committed at
c157 with the commit owner's V at 0.125 against a 0.2197 floor, `mult` 0.568 — comfortably
licensed) and then had 12 + 9 cycles to earn L5 and did not. Seed 0 committed L5 at c186 with
V = 0.125 against tol = 0.1836 (`mult` 0.680), inside era 4's window. So the difference between
seed 0 and seed 2 is not the quality of L4 — seed 2's is larger and licensed more easily — it is
whether the L5 read happened to quiet inside a 12-cycle window.

**4. What this does to §35.** Nothing about §35's internal comparison changes: the yoked 2×2 is a
controlled comparison and Q3c already showed its sign replicates where a comparison exists. What
changes is the SCOPE of the claim. "The composed chooser beats the anchor at the L4/L5 frontier"
is a statement about a substrate that occurs at 1–2 of 4 seeds, and the honest form of it is:
*on the draws where the ladder reaches the frontier, the composed chooser is better there.* The
frontier is not a reliable feature of this configuration, and a result stated as though it were
would be overclaiming.

**No re-plan is proposed here.** Whether the next move is more seeds, a longer era-4/5 budget, or
an intervention on the commit licence at the top rungs is a design decision, and this section
exists to put the spread in front of it rather than to pick.

## §38. What Q3e returned (2026-09-14) — the frontier on the second seed that reached it

Two arms yoked to the banked seed-2 anchor (`--yoke-from-tag vo_s3d2`, `--seed 2`), 201 cycles
each. **Y-1 exact on both, 0 cancelled**: commits `[59, 77, 157]`, advances
`[60, 110, 180, 192, 201]`, L2/L3/L4 realised at the anchor's cycles, lifetimes within 0.4% of the
source's. The control holds; seed 2 committed L4 and never L5, so this tests §35's **L4** cells
and cannot test its L5 cells.

**The result splits, and the split is the finding.** Pooled class accuracy against the repair set,
held-out beam instances, treated minus anchor:

| | L2/L3 | L4/L5 frontier |
|---|---|---|
| seed 0 · `comp_yk` | **+18.5%** (n ≈ 185,000) | **+16.1%** (n ≈ 13,900) |
| seed 0 · `comp_pr_yk` | **+31.2%** | +3.9% |
| seed 1 · `comp_yk` | **+24.5%** (n ≈ 48,000) | — (no frontier) |
| seed 1 · `comp_pr_yk` | **+27%** | — |
| seed 2 · `comp_yk` | **+19.3%** (n ≈ 197,000) | **−1.7%** (n ≈ 10,800) |
| seed 2 · `comp_pr_yk` | **+31.5%** | −1.6% |

**1. At L2/L3 the advantage replicates across all three seeds that produced those rungs, and the
sizes are close enough to be startling**: +18.5 / +24.5 / +19.3% for `comp_yk` and
+31.2 / +27 / +31.5% for `comp_pr_yk`, on 48,000–197,000 held-out rows per arm. That is the
node's solid result.

**2. At the L4/L5 frontier it does not replicate.** Seed 0's +16.1% becomes **−1.7%** at seed 2 —
both composed arms are level with, marginally behind, the anchor at every pooled frontier read.
Per cell, `comp_yk` is above the seed-2 anchor at 3 of 4 L4 cells but by 0.005–0.017 rather than
seed 0's 0.015–0.121, and is 0.060 below at 4n2. **§35's headline — the composed chooser beats
the anchor at 6 of 6 frontier cells — is a seed-0 fact and is now known not to generalise to the
one other draw that reached the frontier.**

**3. Era error at the seed-2 clock replicates in SIGN everywhere and in SIZE only at eras 2–3.**

| | e1 | e2 | e3 | e4 | e5 |
|---|---|---|---|---|---|
| seed-2 anchor | 0.466 | 0.599 | 0.625 | 0.622 | 0.831 |
| seed-2 `comp_yk` | 0.469 | 0.589 | **0.577** | 0.597 | **0.769** |
| seed-2 `comp_pr_yk` | 0.469 | 0.593 | **0.548** | 0.556 | **0.708** |

Era-3 gap: **−0.048** at seed 0 and **−0.048** at seed 2 for `comp_yk` (−0.081 / −0.077 for
`comp_pr_yk`) — the same number twice. Era-5 gap: −0.223 (seed 0) against −0.062 (seed 2) for
`comp_yk`, and −0.070 against −0.123 for `comp_pr_yk` — both arms beat the anchor at both seeds,
but the magnitude moves by 3–4× **and the ordering of the two arms flips**. So "composed beats the
anchor in the consumption eras" survives; "by 0.223" does not, and neither does "probes hurt the
composed chooser" (§35), which reverses here as it already did at L2/L3 in §36.

**4. The bill is stable across seeds**: 0.0060 mean / 0.0080 max (207,413 probes, every one graded = billed), against 0.0055–0.0056 /
0.0082–0.0083 at seed 0 and 0.0044 / 0.0046 at seed 1's short run.

**What the node can now claim, and what it cannot.** It CAN claim that the composed chooser —
the DP's surface prior z-scored against a verdict critic, argmax of the sum — raises the
executor's class accuracy against the repair set by ~19% relative at L2/L3 and lowers
consumption-era error, both replicated on every seed that produced the rungs to measure them on.
It CANNOT claim a frontier effect: at L4/L5 the measurement exists on two seeds and agrees with
itself on neither. The frontier claim of §35 is hereby narrowed to seed 0, and §37's scope
sentence — *on the draws where the ladder reaches the frontier, the composed chooser is better
there* — is **withdrawn**: it held on the one draw it was written from and fails on the second.
