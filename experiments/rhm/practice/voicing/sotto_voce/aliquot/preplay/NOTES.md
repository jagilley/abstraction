# preplay — NOTES

The implementer's record for this node: every decision and why, every gate and what it was
shown to fail on, every defect beside its correction. No README — the orchestrator discusses
results before any writeup.

**Node**: `experiments/rhm/practice/voicing/sotto_voce/aliquot/preplay/`
**Runs**: 2026-09-21. `pp1` (the read as a RANKING of candidate entries), six arms across six
containers, app `ap-8LpmDBVCBYSdjcLwHH7qnS`, **0.12 GPU-h**. `pp2` (the read in the
SELECTOR'S seat), six arms, app `ap-91frINqMJ1rwcOoWyxJH2u`, **0.14 GPU-h** (504 container-
seconds). `pp4` (the read as the ORDER for the loop's own incremental audition), the two
arms of record, app `ap-ggxCqmIoQ9rab2FRqtoYEr`, **0.14 GPU-h** (496 container-seconds).
`pp5` (the read as the GATE, the prior as the order), the two arms of record, app
`ap-iQYLGGBcORtkHtkHckBOrk`, **0.10 GPU-h** (348 container-seconds).
Sections 0-6 are pp1's; section 7 is pp2's; section 8 is pp4's; section 9 is pp5's.
**Volume**: `rhm-scaling-data:/rhm_practice_preplay/pp1/`. Mirrored locally under `figures/pp1/`
(the per-instance `*_inst.npz` are left on the volume only; everything the reduction reads is in
the per-arm JSON).

## 0. The question, in one line

`ideas/calibration_and_violation_are_one_object.md` §12.4: a candidate that exists only as a
table entry has no price until the world model is put into the state it implies and the
projection is read there. So: **fire candidate entries through the shaped plant's own executor,
read the projection over the fired configurations, and ask whether it ranks the candidates as
the world's audition does.**

## 1. Decisions

1. **Two candidate forms, and the single-entry one carries the round.** The *delta* form is the
   loop's own question (`census`'s commit-then-extend, `soundboard.py` 13346–13427: audition
   `base ∪ {cand}` against `base`). The *single-entry* form asks the same question with the
   candidate alone as a one-row table. Both were run; §4 records that the delta form is
   under-powered at the loop's own base sizes and why.
2. **The base is a matched-size random subset of the TRUE table at the arm's committed
   `n_entries`**, three draws, over the TRUE lower table at every level — `soundboard`'s own
   `rand_k` idiom (12855–12868) verbatim in shape. Holding the lower vocabulary at the truth
   means only level ℓ varies between base and base+candidate, which is the controlled
   comparison; it also makes "true entry" and "wrong entry" well defined as set membership in
   `MC.grade_table`'s own flat set.
3. **A level with no commit falls back to the arm's own live table size at the last cycle**
   (`log["aud"][ℓ]["n_entries"]`), recorded per cell as `base_src = live`. This is only seed 2
   at L5, which the brief named as a level the plant never trained on; it is kept and read as
   such, not dropped.
4. **The forty lines of the projection's read are re-implemented, not imported.** Importing
   `soundboard.soundboard` pulls a 1.2 MB module with a `modal.App` at its top into every paid
   container. The copy is licensed by gate F-2, which runs the real `VoProjBank` against the
   copy elementwise (§3). `context_instances` is likewise copied into `pool.py`, gated on
   **text identity** with soundboard's (F-6).
5. **The frozen and twin controls use THIS node's readout form, not `duplex`'s per-slot probe
   form.** A deviation from the brief, taken deliberately: the object under test is the in-loop
   projection (pooled features + root one-hot + root×feature interaction, 1737 columns, IRLS
   with the ridge selected on a train-internal split), and fitting the controls with `duplex`'s
   simpler per-slot design would confound *trunk* with *design*. All three refits share one
   train/validation index subsample drawn from the arm's own `vo_bank.npz`, so they differ by
   the trunk and by nothing else. The arm's **banked** `(w, mu, sd)` is read beside the refit as
   the byte-faithful in-loop object; the two agree to 0.001–0.012 on home ground, which is the
   evidence that the refit is the same estimator.
6. **The twin is minted at `vo_pj_rand_seed = 20260918`**, the in-loop seed `build_rand_trunk`
   used, not `overtone`'s offline `20260915`. Stated rather than inherited: the floor here is
   the object `aliquot`'s own random-twin arm read.
7. **Fresh pool seeds** `cfg.seed + 5_100_000 + 1000·ℓ`, disjoint from the run's own audition
   families (`+820_000+…` for the merge ledger, `+900_000+…` for commit-then-extend). Candidate
   and base draws live on a second disjoint stream (`6_200_000 + …`), the refit subsample on a
   third (`7_300_000 + …`).
8. **`n_score = 256`** is the loop's own audition size and is kept. 48 candidates per class per
   base for the delta form (3 bases), 96 per class for the single form; at L2 the true class is
   capped by the table itself (16 rows, 14 distinct flats) and the wrong class by the 8×8 pair
   space (50 off-table flats). Counts are printed in every table.
9. **The executor's own free signal** is recorded from the same DP pass that fires the move:
   `dp_top` (the winning entry's score per block of the span), `dp_lse` (top − logsumexp over
   entries, the DP's confidence in its pick) and `dp_marg` (top − runner-up). On a one-row table
   `dp_lse` and `dp_marg` are identically zero by construction, so the single form's free-signal
   column is `dp_top`; the reduction says so where the `dp_lse` column reads `--`.
10. **Six arms, not two.** The two `sb_sv_yk` arms are the arms of record. `sb_so_yk` (verdict
    with the infill term off — the run collapsed as an executor while its readout still read
    0.80) and `sb_yd_yk` (yield-shaped) were added because the whole sweep costs 0.12 GPU-h and
    the `so` arm is the lineage's own dissociation control. They are independent arms of the
    banked design spread across containers; no GPU-hours are added by the fan-out.

## 2. What was NOT built, and why

**The learner's own mined-but-uncommitted keys as a third candidate set.** The brief allowed
skipping this if the class map is not serialised enough. It is not, on two counts:

- `results.json["log"]["miner"][-1][ℓ]["keys_at_support"]` holds **class-pair keys**
  (e.g. `[[2,6],[4,3]]` at L3), and turning one into a child row needs
  `last_build["picks"]`, which maps a class key to row indices **into that arm's own operative
  lower table** — not into the true lower table this node's bases are built over. A mined lower
  row that is not a true derivation has no index in the true table at all, so the map is partial
  by construction.
- `picks` is serialised only for the **last** cycle's build, which is taken at the L5 era. The
  L2/L3/L4 candidate sets would have to be read at each level's own era, and that state is not
  in the dump.

Reconstructing it would mean re-running the miner, which is a loop, not an offline read. Left
for a node that banks the quotient's map.

## 3. The gates. Every one below has been shown to fail (`falsify`, 7/7).

Run order: `gates` → `fidelity_gate` → `falsify` → smoke → the paid sweep.

| gate | what it asserts | closed at | falsified by |
|---|---|---|---|
| **F-1** | the recording DP (`fire_rec`, which keeps the DP's scores) reproduces `MC.apply_any` **bit for bit**, so the free-signal instrument cannot move a fired state — `soundboard`'s E-7 idiom | identical at L2/L3/L4/L5 in `gates`, and again on the first audition of every level of every paid arm | a recorder taking `argmin` instead of `argmax`: 48/48 rows differ |
| **F-2** | the re-implemented `_features`/`_design`/`_score`/`predict` against **`VoProjBank` itself**, elementwise, on the arm's own banked held-out rows through its own banked core with its own banked weights; masked and unmasked | `max|Δp| = 0.000e+00`, `max|ΔF| = 0.000e+00`, AUCs equal to 1e-9, on **all six arms** | reading the same rows through the never-trained twin: `max|Δp| = 0.999` |
| **F-2b** | the re-read scored against the arm's **banked** held-out AUC, inside the readout's own per-refit drift band (2× its \|drift\| p90, floored at 0.02) | \|Δ\| = 0.0035 / 0.0012 / 0.0008 / 0.0066 / 0.0005 / 0.0017 against bands 0.047 / 0.023 / 0.041 / 0.071 / 0.027 / 0.050 | the same wrong trunk: AUC 0.491 against the banked 0.819, outside the band |
| **F-3** | a "wrong" candidate really is off the true table and a "true" one really is on it, by `MC.grade_table`'s own set | wrong-table precision 0.000 and true-table precision 1.000 at L2/L3/L4 | handing `wrong_rows` an empty true set: precision 0.281 |
| **F-4** | the pool is the level's own damage cell and is broken there | bootstrap success 0.000 and mean d\* 2.2 / 2.9 / 4.3 / 5.7 at L2/L3/L4/L5 | grading the clean derivations instead: success 1.000 |
| **F-5** | the design matrix is the one the arms ran: 2·96 + 8 + 2·96·8 + 1 = 1737 columns | 1737 | dropping the root-interaction block: 201 |
| **F-6** | the copied `context_instances` is **text-identical** to `soundboard.py`'s | identical | changing the copy's pool seed offset by one character |

Also asserted inside every paid arm, not in the table: the banked readout's config is the one
the arm ran (`vo_om_mode = proj`, `vo_pj_trunk = live`, `vo_pj_mask`, `vo_pj_root = inter`), and
the shaped and frozen cores are not the same object (fingerprints differ by 30–60).

## 4. Defects, and their corrections

**Defect 1 — the fidelity gate the brief asked for does not exist, and the first build asserted
it anyway.** The brief's F-2 was "`predict` on the held-out rows of `vo_bank.npz` must reproduce
the arm's banked held-out AUC to the third decimal". The first paid attempt tripped at
`0.822252` against the banked `0.818703`. The diagnosis is structural, not a bug in the copy:
**the loop trains the plant AFTER the readout's `refresh` inside the same cycle**
(`soundboard.py` 12227 for `vo_om.train(); vo_om.refresh()` against 12553 for the plant
continuation), so the core in `vo_heads.pt` has had one more plant update — `gen_steps = 20`
plus the shaping step — than the core the last logged `hold_auc` was measured through. No
re-implementation can close that gap. Confirmed directly: running the **real** `VoProjBank.predict`
through the banked core on the same rows gives `0.822252` as well, to the bit.

The correction was not to loosen the gate but to **replace it with an exact one**: F-2 now runs
the copy against the class itself, elementwise, and closes at `max|Δp| = 0.000e+00`; the banked
scalar survives as F-2b with a band derived from the readout's own measured per-refit drift
(median |drift| 0.004–0.017, p90 0.011–0.035 over the last 60 refits). The realised gaps are
5–20× smaller than the band on every arm. Both are reported; neither is called the other.

*This is also a fact about the banked artefacts worth carrying forward*: `vo_heads.pt`'s `core`
and its `proj` are one plant update out of step with each other, and any node that reads them
together inherits that. It is small (0.0005–0.0066 of held-out AUC here) and it is not zero.

**Defect 2 — a stray non-ASCII character** in the reducer's `[W]` header text (a CJK glyph
where "picks" belonged). Cosmetic, caught on the first reduction, corrected.

**Not a defect, but a design fact the data forced into the record — the delta form is
under-powered at the loop's own sizes.** At the committed base sizes, the executor's DP never
picks **92–99% of candidates** at L3/L4/L5 (35% at L2), so their audition price is identically
zero by construction and the world's own true-vs-wrong separation collapses to 0.49–0.62 there.
The effective n is 2–76 of 288 candidates. The `rho` entries of ±1.000 in `[A:delta]` are a
handful of untied values dressed as a correlation and carry **no claim**; the reduction now
prints `n(W≠0)`, `n(pick)` and the median number of instances a picked candidate is written on,
beside every delta row, so this cannot be read as a result.

What size would it need? Not more instances: with 5% of candidates picked on ≥1 of 256
instances, restoring a per-candidate signal needs roughly 20× the pool (`n_score ≈ 5000`) and
still resolves the price at 1/5000. The cheaper and more informative fix is a **smaller base** —
the candidate has to win the DP against fewer alternatives — which is a different question from
the loop's own, and so was not substituted for it. The single-entry form is the same question
asked at a base of size zero, and it is where this round's signal is.

## 5. Reproduction

```bash
cd experiments/            # MODAL_PROFILE=chromatic
modal run rhm/practice/voicing/sotto_voce/aliquot/preplay/preplay.py::gates           # F-1,3,4,5,6
modal run rhm/practice/voicing/sotto_voce/aliquot/preplay/preplay.py::fidelity_gate \
    --arms s0_sv,s2_sv,s0_so,s2_so,s0_yd,s2_yd                                      # F-2, exact
modal run rhm/practice/voicing/sotto_voce/aliquot/preplay/preplay.py::falsify         # 7/7
modal run rhm/practice/voicing/sotto_voce/aliquot/preplay/preplay.py::sweep \
    --out-tag pp_smoke --arms s0_sv --smoke 1
modal run --detach rhm/practice/voicing/sotto_voce/aliquot/preplay/preplay.py::sweep \
    --out-tag pp1 --arms s0_sv,s2_sv,s0_so,s2_so,s0_yd,s2_yd
python3 rhm/practice/voicing/sotto_voce/aliquot/preplay/reduce_preplay.py --tag pp1 --fetch
```

## 6. Files

| file | purpose |
|---|---|
| `preplay.py` | the Modal job: the re-implemented projection read, the recording DP, the candidate sets, the fire-and-read loop, `gates`, `fidelity_gate`, `falsify`, the CPU coordinator `sweep` |
| `pool.py` | `soundboard::context_instances`, copied verbatim, gated on text identity (F-6) |
| `reduce_preplay.py` | the reducer: `figures/preplay_reduction.txt` (the table of record) and `figures/pp1/<arm>_candidates.tsv` (the per-candidate tables) |
| `figures/preplay_reduction.txt` | the table of record: [G] gates, [H] home ground, [W] the world's audition, [A:single] / [A:delta] the candidate tables, [A3] within-class, [A2] the level, [I] the per-instance transfer, [C] the bill |
| `figures/pp1/` | the per-arm JSON and log as written on the volume, plus the per-candidate TSVs |
| `results/launch_pp1.log` | the launch of record |


---

# 7. pp2 — the read in the selector's seat

`preplay/selector.py`, `preplay/reduce_selector.py`. Table of record: `figures/select_reduction.txt`,
appended to `figures/preplay_reduction.txt` as section **[S]** (idempotently; re-running replaces
the block rather than stacking). Volume `rhm-scaling-data:/rhm_practice_preplay/pp2/`.

## 7.0 The question

pp1 established that the shaped plant's projection RANKS candidate entries roughly as the
world's audition does. pp2 puts that ranking in the seat that decides what goes into the table
and audits the table the world then gets: **does selecting by the endogenous read build a better
table than selecting at random, and how close to selecting by the world?** The
abstraction-supervision consumer, in one offline step.

## 7.1 Decisions

1. **Priced on one pool, tested on three DISJOINT ones.** The world selector must not be
   allowed to test on the pool it priced on, and every selector must meet the same sampling
   noise. Gate S-1 asserts row-by-row disjointness (pricing x test and test x test) at every
   level of every arm; realised overlap is 0 everywhere.
2. **Re-priced rather than reusing pp1's prices**, and on a LARGER pool (`n_price = 512`
   against the loop's 256). The brief allowed either. The reason for re-pricing: the selector's
   input is a ranking, and pp1 measured the within-true spread of the world's price at sd
   0.12-0.22 against a binomial standard error of 0.025 at n = 256; halving that error costs a
   few container-seconds, and "is the read a good selector" should not be answered against a
   needlessly noisy ceiling. The TEST pools are kept at the loop's own `n_score = 256`, three
   of them, so the reported spread is the loop's own and not an artificially tightened one.
3. **Seven selectors, one pool, one shared tie-break.** `world` (ceiling), `banked` (the object
   under test), `shaped_refit`, `frozen`, `twin`, `dp_top` (the executor's own free prior) and
   `random`. The tie-break permutation is drawn ONCE per level and shared, so it cannot favour
   a selector; `random` is then exactly "rank by a constant", which gate S-2 asserts.
4. **Two forms, as briefed.** From scratch (the table IS the top-k) at k = the arm's committed
   `n_entries` and at 1/2 and 1/4 of it; extension of a small base (a quarter of committed,
   because pp1 measured that at the committed sizes the DP never picks 92-99% of added entries)
   by the top-k of the remaining candidates, three base draws. References: pp1's `rand_k`
   (a matched-size random subset of the TRUE table, three draws) and the full true table.
   **`rand_k` is NOT the fair random baseline** — it draws from a pool with no wrong entries in
   it. The fair one is `random`, a uniform draw from the same candidate pool. Both are in the
   table, labelled.
5. **The read is IMPORTED from `preplay.py`, not copied again.** pp1's gate F-2 — the
   elementwise check of those exact functions against `VoProjBank` itself, max|dp| = 0.000e+00
   on all six arms — therefore covers this node by identity rather than by re-argument.
6. **The candidate pool is pp1's construction** at 128 per class (L2 is capped by the tables
   themselves at 16 true and 50 wrong). At every level and every arm `k` is well below the pool
   size, so the top-k is a real choice and not "take everything".

## 7.2 The gates. S-1..S-5, all closed, all shown to fail (`falsify2`, 6/6).

| gate | what it asserts | closed at | falsified by |
|---|---|---|---|
| **S-1** | the pricing pool and every test pool share no instance, and the test pools share none with each other | 0 shared rows at L2/L3/L4/L5 on all six arms | drawing the test pool on the pricing pool's own seed: 256 shared |
| **S-2** | a CONSTANT price reduces exactly to the shared tie-break, i.e. to the random draw; and the top-k really is the top-k | identical selections; min(top) >= max(rest) | a per-selector tie-break (selections differ); an ascending sort (the bottom-k) |
| **S-3** | the selection's own true/wrong accounting ties to `MC.grade_table`'s | `n_correct` == distinct true flats selected at L2/L3/L4 | claiming every admitted row as a true entry: 18 against 10 |
| **S-4** | selecting by the world's own price beats a random k on the pool it was priced on — the instrument is not inverted | e 0.613 against 0.742 | taking the LOWEST-priced candidates: e 1.000 against 0.742 |
| **S-5** | `grade_flat` (the truth set precomputed) is `MC.grade_table` on real tables | identical on all five fields at L2/L3/L4 | a truth set holding half the true flats |

`preplay.py`'s F-1..F-6 carry over unchanged; F-2b is re-run inside every pp2 arm and is inside
its band on all six.

## 7.3 Defects, and their corrections

**Defect 3 — the file was named `select.py`, which shadows the standard library's `select`.**
Modal copies the entrypoint to `/root/` on the container, and `asyncio` -> `socket` ->
`selectors` -> `import select` then picked up this node's module, which imports `modal`, which
was still being initialised. The symptom was an eleven-minute hang followed by
`AttributeError: partially initialized module 'modal' has no attribute 'Image'` under a banner
reading "Something with the Modal installation seems broken". Nothing was broken. Renamed to
`selector.py` (and `reduce_select.py` -> `reduce_selector.py`); the same gate block then ran to
completion in 0.5 s. **Worth carrying: no Modal entrypoint in this repo may be named after a
stdlib module.** Two diagnostic errors of my own compounded it: the first run piped its output
into `tail`, which buffers until the pipe closes, so the failure was invisible for eleven
minutes; and a later run wrote to the same log file as the still-retrying first one, so a stale
traceback appeared beside a fresh one. Both are now avoided by streaming to a per-run log.

**Defect 4 — S-2's first assertion was wrong, not the machinery.** It compared the selection
under a constant price against `tie[:k]`, the permutation's first k VALUES; `np.lexsort` returns
INDICES ordered by the tie value, i.e. `argsort(tie)[:k]`. Both are uniform subsets; only the
second is what the code produces. The gate tripped on correct machinery and the assertion was
corrected, not the selector.

**Defect 5 — the selection's accounting and `MC.grade_table`'s are not the same quantity.**
S-3 first asserted `grade_table`'s `n_correct` equals the number of true CANDIDATES selected. It
does not: `grade_table` works on SETS OF FLATS, and the true table carries synonymous rows (L2:
16 child rows, 14 distinct level-1 expansions), so nine selected true rows can be eight distinct
true flats. The gate caught it at L2 on the real tables. The corrected gate ties `n_correct` to
the distinct true flats and the row count separately, and both quantities are reported: `w` in
the tables counts CANDIDATES admitted, `precision`/`recall` count FLATS.

**Defect 6 — `MC.grade_table` rebuilds the truth side on every call, and at L5 that is 205824
tuples of 32 ints.** The pp2 smoke spent 60 of its 76 seconds inside it, on the `true_full`
reference alone. Replaced by `grade_flat`, the same set arithmetic against a truth set built
once per level (gate S-5 asserts the identity), plus a shortcut for `true_full`, whose precision
and recall are 1.0 by definition. The arm went from 76 s to 16 s at smoke sizes. **This is a
property of the donor helper, not of this node**: any caller that grades many tables against a
high-level truth table pays it.

## 7.4 Two things the reduction says about itself, not about the result

- **The capture fraction is a ratio, and its denominator collapses in some cells.** Where the
  world selector's own advantage over random is at or below the test pools' spread, the ratio is
  noise over noise; those cells print `--` and the pooled summary is also given restricted to
  cells with a gap of at least 0.10. At `s0_sv` L5 the gap is 0.000-0.007 on every k, so that
  whole cell carries no capture number.
- **`n_wrong_admitted` is the least noisy column in the round** and needs no audition at all —
  only the selector's ranking and the oracle's own set. [S4] reports it as a share of k against
  what a uniform draw from the same pool would admit, which is the cleanest statement of what
  each selector is actually doing.

## 7.5 Reproduction (pp2)

```bash
cd experiments/            # MODAL_PROFILE=chromatic
B=rhm/practice/voicing/sotto_voce/aliquot/preplay
modal run $B/selector.py::gates2      # S-1..S-5
modal run $B/selector.py::falsify2    # 6/6
modal run $B/selector.py::sweep2 --out-tag pp2_smoke --arms s0_sv --smoke 1
modal run --detach $B/selector.py::sweep2 --out-tag pp2 \
    --arms s0_sv,s2_sv,s0_so,s2_so,s0_yd,s2_yd
python3 $B/reduce_selector.py --tag pp2 --fetch
```

## 7.6 Note on the node directory

Files `own.py` and `learner_tables.py` appeared in this directory from another session while pp2
was being built. They are not mine, I have not touched them, and nothing here imports them. My
files are `preplay.py`, `pool.py`, `reduce_preplay.py`, `selector.py`, `reduce_selector.py`, and
my tags are `pp1`, `pp_smoke`, `pp2`, `pp2_smoke`.


---

# 8. pp4 — the read as the ORDER for the loop's own incremental audition

`preplay/incremental.py`, `preplay/reduce_incremental.py`. Table of record:
`figures/incremental_reduction.txt`, appended to `figures/preplay_reduction.txt` as section
**[W4]** (idempotently). Volume `rhm-scaling-data:/rhm_practice_preplay/pp4/`.

## 8.0 The question

pp2 found that a table's audition is set by which entries win the executor's argmax, so the
top-k by any individual price is the wrong consumer above L2. The loop's ACTUAL consumer is
`census_extend`: walk the candidates in an order, audition `base + {cand}` on a fresh pool of
`n_aud = 192`, admit iff `e_x <= best_e + extend_tol` with `extend_tol = 0.0`, grow the base on
admission, one audition per candidate, `extend_cap = 8` per pass. **With that gate fixed, does
walking in the read's order reach a given table quality with fewer auditions than a random
order, and how close to the world's own order?**

## 8.1 Decisions

1. **The gate is the loop's, and is asserted to be** — against a hand-computed synthetic case
   (G-1b) and against a literal transcription of `census_extend` on real auditions at cap 8, at
   the full walk, and at tol 0.01 (G-1a). Nothing about the admission rule is this node's; only
   the ORDER varies between its arms.
2. **Three disjoint pool families** (G-2, 0 shared instances at every cell): PRICING n = 512
   (pp2's family) where every candidate's price comes from; **GATE n = 192, the loop's own
   `n_aud`** (this node's family), one pool per (level, repeat) shared by every order in that
   repeat, so the orders differ by nothing but the order; TEST n = 256 x3 (pp2's family) where
   the table's error is reported and on which nothing is ever admitted.
3. **The full walk, so the loop's cap of 8 is one point on a budget axis** (8 / 16 / 32 / full)
   rather than the whole experiment. The budget points are evaluated on all three test pools;
   the CURVE between them is evaluated on test pool 0 only, because at tol 0 the gate admits
   most candidates and a three-pool curve would triple the round's cost for a series read as a
   shape. The "auditions to within 0.02" statistic is computed on that one-pool series against
   the world order's own one-pool final, so both sides carry the same pool and the same noise.
4. **The ungated top-b by the same price is printed beside every gated order**, at the same
   budget — pp2's selector on the same pool with the same base — so the gate's own contribution
   is visible rather than assumed.
5. **Setting (b) is the learner's own rows** from pp3's `learner_tables.py` replay (18 / 84 /
   204 operative rows at L2/L3/L4, of which 12 / 29 / 54 are true at seed 0), **base EMPTY** --
   build the table from nothing by incremental audition, so the walk is the whole story. The
   empty table is handled as `census`'s own empty case (`soundboard.py` 11172-11176): no move is
   applied and the pool is graded as it stands. L4's 204 rows are subsampled to 96 with the
   BUILD ORDER PRESERVED among those taken, so `build` is still the learner's emission order
   restricted.
6. **`build` is a learner-side order, not THE learner's order**, and is labelled so everywhere.
   The loop's `extend_candidates` sorts by `(-count, key)`; the per-key support COUNTS are not
   in the dump (only `n_at_support` at fixed thresholds), so the loop's actual order is not
   reconstructible offline. `build` is the operative table's own row order, which is
   `ClassMiner.build`'s emission order (keys sorted, then the cross-product within each key).
7. **A tolerance sensitivity pass at `extend_tol = 0.01`** (about two instances of 192), one
   repeat, two orders, since at tol 0 a single unlucky instance rejects a good entry.
8. **Two arms, the two of record.** The round came in at 0.14 GPU-h against a 0.3 budget; the
   four extra arms were not run because the brief asked for the arms of record and the result
   does not turn on them.

## 8.2 The gates. G-1a, G-1b, G-2, G-3, all closed, all shown to fail (`falsify4`, 4/4).

| gate | what it asserts | closed at | falsified by |
|---|---|---|---|
| **G-1b** | the walk reproduces a HAND-COMPUTED case: base 0.50, then 0.40 0.45 0.40 0.50 0.10 0.11 at tol 0 gives admissions `[0, 2, 4]`, 3 rejected, 7 auditions, final 0.10 | exactly | a base that never grows, so `best_e` stays at the base's error |
| **G-1a** | the walk is IDENTICAL to a literal transcription of `census_extend` on real auditions — admissions, rejections, audition count and final error — at cap 8, at the full walk, and at tol 0.01 | identical on all three | the admission test as `<` instead of the loop's `<=` (a tie is then rejected) |
| **G-2** | the pricing, gate and test pool families are pairwise disjoint, and the pools within a family are disjoint from each other | 0 shared instances at every level of every cell of every arm | drawing the gate pool on the test family's seed |
| **G-3** | a constant price reduces the order to the shared tie-break (the random order), and a strictly decreasing price gives the identity order | both exact | a per-order tie-break permutation |

`preplay.py`'s F-1..F-6 and `selector.py`'s S-1..S-5 carry over. The read and the fire are
IMPORTED from `preplay.py`, so its exact gate F-2 (max|dp| = 0.000e+00 against `VoProjBank`
itself) covers this node by identity. Setting (b)'s tables carry pp3's 22 reconstruction gates
(R-0..R-4), re-run and asserted in-container on every arm.

## 8.3 Defects, and their corrections

**Defect 7 — `learner_tables.reconstruct` returns the whole chain INCLUDING level 1.** Level 1
is `MC.base_table(v)` and has no diag, so keying the setting-(b) levels off `lt_tables` raised
`KeyError: 1` on the first smoke. The levels of record are the diags', and the level loop now
keys off those.

**Defect 8 — the first cost-to-quality summary pooled orders that do not run over the same
cells.** `frozen` and `twin` run only in setting (a) and `build` only in setting (b), so a
single pooled median put `frozen` at 0.104 against `banked`'s 0.201 over different cell sets.
Caught on the first reduction; the summary is now split by setting, every order in a block is
read over the same cells, and the share of cells the order ARRIVED in at all is printed beside
the median (an order that never arrives has no median to compare).

## 8.4 What the round's own instrument says about itself

- **At `tol = 0` and a full walk the order cannot matter**, because the gate admits nearly
  every candidate that does not strictly hurt: setting (a)'s full-walk finals agree to about
  0.005 across all seven orders in most cells. Any effect lives at small budgets, which is what
  the budget axis is for, and `b = 8` is the loop's own.
- **The world's price is a much weaker instrument on the learner's own rows than on constructed
  ones**: its true-vs-wrong AUC is 0.91-1.00 in setting (a) and 0.57-0.90 in setting (b). The
  learner's false rows are correlated with its own state and many of them repair instances
  anyway, so "wrong against the grammar's enumeration" and "useless to the executor" are not
  the same set. That is a fact about the substrate, and it caps every selector in (b).
- Cells where the base is already good or the outcome is insensitive (s0_sv aL3 rp1, aL4, aL5
  rp0) have every order arriving at audition 0 and carry no comparison; they are left in the
  table with their zeros rather than filtered, and the `arrived` counts say how many cells each
  median rests on.

## 8.5 Reproduction (pp4)

```bash
cd experiments/            # MODAL_PROFILE=chromatic
B=rhm/practice/voicing/sotto_voce/aliquot/preplay
modal run $B/incremental.py::gates4       # G-1a, G-1b, G-2, G-3
modal run $B/incremental.py::falsify4     # 4/4
modal run $B/incremental.py::sweep4 --out-tag pp4_smoke --arms s0_sv --smoke 1
modal run --detach $B/incremental.py::sweep4 --out-tag pp4 --arms s0_sv,s2_sv
python3 $B/reduce_incremental.py --tag pp4 --fetch
```


---

# 9. pp5 — the read as the GATE, the prior as the order

`preplay/readgate.py`, `preplay/reduce_readgate.py`. Table of record:
`figures/readgate_reduction.txt`, appended to `figures/preplay_reduction.txt` as section
**[G5]**. Volume `rhm-scaling-data:/rhm_practice_preplay/pp5/`.

## 9.0 The question

pp4 settled that the executor's own score is the right ORDER and that the value readout is not
a scheduler. The readout's seat is the other half of `census_extend`: the loop's gate is an
ORACLE READ — "did the world's error on the gate pool rise?" — and the readout is the thing
that gets to see the preplayed state and put a number on it. **Can that number stand in for the
world's verdict?** This is the fully endogenous consumer and the seat where the read replaces
the oracle.

## 9.1 Decisions

1. **One fire per trial, read by every gate.** The world's error and all three readouts'
   per-instance levels come from the SAME fired gate-pool configurations. That is what makes
   the per-candidate agreement a comparison rather than two experiments, and it costs exactly
   what the loop pays: one audition for the base, one per candidate.
2. **The world's decision is recorded at every step whatever the gate decides.** Divergent
   gates visit divergent states, so "the same candidate" has to be pinned to a trial; the
   confusion counts are therefore over the states each gate actually visited, which is the only
   sense in which they are comparable at all.
3. **The margin is measured, not chosen**: `delta` is the standard error of the BASE table's
   own level on that gate pool (`std(p_base) / sqrt(n_gate)`), computed per cell and repeat and
   reported beside every row. It is the read's own noise scale on the object it is judging.
4. **`ungated` is in the table as the floor**, and it is the row that has to be read first
   (section 9.4).
5. The order is `dp_top` throughout (pp4's answer), with the world's order and the read's own
   order as secondary rows at repeat 0 for the `world` and `read` gates, so the gate's standing
   can be checked not to depend on the order.

## 9.2 The gates. P-1a, P-1b, P-2, P-3, P-4, P-5, all closed, 5/5 falsified (`falsify5`).

| gate | what it asserts | closed at | falsified by |
|---|---|---|---|
| **P-1a** | the world gate here is IDENTICAL to pp4's `census_walk` on the same inputs | identical | (see P-1) |
| **P-1b** | and reproduces pp4's hand-computed sequence: admissions `[0, 2, 4]`, 3 rejected, 7 auditions | exactly | `best_e` pinned at the base, on that sequence: admits everything |
| **P-2** | THE PLUMBING IDENTITY — handed the world's error as its level, the read gate reproduces the world gate EXACTLY, in all three of its forms (`read`, `read_pair`, `read_m`) | identical on all three | the level as the world's error with the sign not flipped |
| **P-3** | a constant level admits everything, in every read form and both controls | 26/26 offered, six gates | a strict `>` on the level, which then admits nothing |
| **P-4** | the pricing, gate and test pool families are pairwise disjoint | 0 shared instances, every level | the gate pool on the test family's seed |
| **P-5** | the CHANGED set (`trial.entry == len(kept)`) covers every instance whose fired configuration differs, and is NON-VACUOUS | 192/192 flagged, 0 unflagged-but-differ, six candidates | the changed set taken as `entry == 0` |

## 9.3 Defects — both of the same kind: a gate that cannot fail is not a gate

**Defect 9 — P-1's falsification did not trip, twice.** Pinning `best_e` at the base produced
exactly the reference admissions on the gate block's real pool, and so did a `best_e` that
tracks every trial rather than only the admitted ones. The reason is the pool, not the rule: on
that pool the base is already the minimum and the world gate admits every candidate, so three
different update rules coincide. Fixed by pinning the rule on pp4's DESIGNED error sequence,
where the correct rule admits `[0, 2, 4]` and a pinned `best_e` admits everything. **The
general lesson for this lineage: a walk gate has to be falsified on a designed sequence, not on
whatever pool is to hand, because a pool that admits everything makes most wrong rules right.**

**Defect 10 — P-5 was vacuous.** Against the twelve-row base used elsewhere in the gate block,
the executor's DP never picked the added row on any instance (`n_differ = 0`, six candidates
over), so "no row changed unflagged" held trivially and the gate could not fail. Rebuilt on a
ONE-row base with the six candidates the DP scores highest on that pool — 192/192 instances now
flagged — and the non-vacuity is itself asserted (`sum(n_flagged) > 0`), so the gate cannot
quietly go hollow again.

## 9.4 Two things the round says about its own instruments

- **Raw agreement with the world gate is not evidence.** The world gate admits about 95% of
  the candidates it is offered, so `ungated` — admit everything — already agrees with it 95.5%
  of the time, MORE than any read gate does (`read` 93.8%). What separates them is the
  composition: `read` has TF 0.022 / FT 0.050 against `ungated`'s TF 0.045 / FT 0.000, so it
  halves the junk admitted at the price of refusing 5% of the entries the world would keep.
  The reduction says this above the agreement table, and the quality table is the decider.
- **The paired form IS the pooled form at threshold 0, necessarily.** Only the changed
  instances move, so `pooled_diff = (n_changed / n) * paired_diff` exactly: the two differ by a
  positive factor and therefore never in sign. Identical admission lists in 28 of 28 cells.
  The paired form can only separate from the pooled one under a MARGIN, where that factor
  rescales the threshold — so a paired margin, not the paired level, is the object that would
  be worth building. `read_m` as built uses the pooled scale and is therefore not that object.

## 9.5 Reproduction (pp5)

```bash
cd experiments/            # MODAL_PROFILE=chromatic
B=rhm/practice/voicing/sotto_voce/aliquot/preplay
modal run $B/readgate.py::gates5        # P-1a, P-1b, P-2, P-3, P-4, P-5
modal run $B/readgate.py::falsify5      # 5/5
modal run $B/readgate.py::sweep5 --out-tag pp5_smoke --arms s0_sv --smoke 1
modal run --detach $B/readgate.py::sweep5 --out-tag pp5 --arms s0_sv,s2_sv
python3 $B/reduce_readgate.py --tag pp5 --fetch
```
