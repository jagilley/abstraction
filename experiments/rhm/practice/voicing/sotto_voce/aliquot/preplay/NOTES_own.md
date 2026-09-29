# preplay / pp3 — the learner's OWN mined entries, priced. NOTES.

The implementer's record for this round: every decision and why, every gate and what it was
shown to fail on, every defect beside its correction. No README — the orchestrator discusses
results before any writeup.

**Node**: `experiments/rhm/practice/voicing/sotto_voce/aliquot/preplay/` (tag `pp3`; `pp1` is
`preplay.py`/`NOTES.md`, `pp2` is another builder's and is not touched here).
**Run**: 2026-09-21, tag `pp3`, six arms across six containers, two passes
(`ap-t5Qgpba2eGiBlP4f8nw8RD`, then `ap-eV7VI14wmOZaKH0CLia7Rp` with `k = 1` added):
**0.09 GPU-h total** (147 + 176 container-seconds on L4s; the budget was ~1 GPU-h). Peak RSS
4.8 GB against a 6 GB request, peak GPU 978 MB; `memory=6144` is right-sized within ~20%.
**Volume**: `rhm-scaling-data:/rhm_practice_preplay/pp3/`, mirrored to `figures/pp3/`.

## 0. The question, in one line

pp1 priced **constructed** candidate entries: true entries drawn from the grammar's own table
over the grammar's own lower table, wrong entries random pairs of **true** lower rows. The
realistic object is the learner's **own operative table** — the rows its own miner built, over
its own (partly wrong) lower vocabulary, at its own at-support class-pair keys, capped by its
own `spell_cap`. Its false rows are the ones the learner itself produced. So: **reconstruct
that table offline, price every one of its rows through the shaped plant's own executor, and
then ask whether the read can serve as the selector over it.**

## 1. Step 1 — the reconstruction, and why it is exact

The loop never banks a table as an object, but `soundboard.py` 13729 says an offline replay of
any build is exact with `build.picks` beside it. `learner_tables.py` is that replay:

* `log["miner"][-1][ℓ]["keys_at_support"]` — the at-support class-pair keys, already sorted by
  `ClassMiner.state` at `sup = 3`, which is `cfg["mine_support"]` on these arms (gate R-0).
  `state` sorts the **jsonable** keys (tuples → lists) while `build` sorts the raw ones; the map
  is elementwise and order-preserving on homogeneous keys, so the orders coincide, and R-3/R-4
  are what check it rather than the argument.
* `quotient["last_build"][ℓ]["picks"]` — class id (as `str` of the tuple) → the surviving
  lower-row indices, which **is** the build up to the cross-product. Indices are into
  `lower_flat`, the width-filtered, dedup'd lower rows, which the replay reconstructs itself.
* the recursion closes at L2 on `MC.base_table(v)` (8 rows); L3 is built over the reconstructed
  L2, L4 over the reconstructed L3.

**`quotient["last_build"]` is the OPERATIVE chain, not the cycle's own build.** `soundboard.py`
13653–13676 computes `omask` through `operative(ell)`, which calls
`miners[ell].build(operative(ell-1), …)` bottom-up, and `build` assigns `self.last_build`. The
`_lb_save`/restore around that block puts the cycle's own `last_build` back into `log["quot"]`,
but the top-level `quotient` dict is serialised after it. The evidence is arithmetic and is
gated: `n_lower_rows` reads 8 / 18 / 84 at L2 / L3 / L4 on seed 0 — exactly `len(base_table)`
and the replay's own L2 and L3 row counts — and `n_entries` reads 18 / 84 / 204, exactly the
length of `entry.json.gz[-1]["operative_mask"][ℓ]`.

**L5 is out of reach and was not reconstructed.** Its last build has `n_keys_built = 0` (the L4
book blocks all 14 of its at-support keys) while the operative L5 table still holds the 128 rows
of an *earlier* build, whose keys and picks are not in the dump. **The frozen committed tables
(13 / 59 / 88 / 64 rows at seed 0) are not reconstructible either**: only their sizes
(`events[kind=commit].n_entries`) and their truth masks (`entry.json.gz[-1]["true_mask"]`) are
banked, never their keys or their picks. Both are stated in the reduction's header.

The tables are written to `figures/<arm>_operative_tables.json` with child rows, flat level-1
spellings and the truth flag per row.

## 2. Decisions

1. **The single-entry form only.** pp1 §4 established that the delta form is under-powered at
   the loop's own base sizes (the DP never picks 92–99% of candidates at L3–L5, so their
   audition price is identically zero by construction). Every row here is priced alone, as a
   one-row table over the learner's own lower table — pp1's single form, on the learner's own
   rows. On a one-row table `dp_lse` and `dp_marg` are identically zero, so the executor's
   free-signal column is `dp_top`; the reduction says so.
2. **Every row is priced, not a sample.** 18 / 84 / 204 rows at seed 0 — the whole table. There
   is no candidate draw here at all, so the arbitrariness pp1 had to carry in its `rand_k`
   candidate sets is simply absent.
3. **The read machinery is imported from `preplay.py`, not re-copied.** `fire_rec`, `pj_fit`,
   `pj_predict`, `vo_auc`, `spearman`, `PJ_*`, `TWIN_SEED`, `ARMS`, `OVERTONE`, `FIT_SEED_BASE`.
   The F-2 exactness gate pp1 closed at `max|Δp| = 0.000e+00` therefore covers this node without
   being re-run: it is the same object.
4. **The refit subsample deliberately reuses pp1's `FIT_SEED_BASE`**, so the frozen and twin
   controls here are the *same objects* pp1 read and the two nodes' control columns are
   comparable. Confirmed by [H]: the held-out AUCs reproduce pp1's table to the fourth decimal.
5. **Pool and draw seeds are the free `8_400_000` family**, disjoint from pp1's (`5_100_000`
   pools, `6_200_000` draws) and from whatever pp2 chose: price pool `seed + 8_400_000 + 1000ℓ`,
   test pools `seed + 8_440_000 + 1000ℓ + 137d`, random selectors `seed*97 + 8_470_000 + ℓ`.
6. **`n_score = 256`**, the loop's own audition size and pp1's.
7. **Step 3's k.** `k_big` is the arm's **own frozen committed size** at that level (13 / 59 / 88
   at seed 0), capped at the operative table's row count; `k_small = round(k_big/3)`; and `k = 1`
   was added after the first pass (§5). Four disjoint test pools, five uniform-random draws at
   matched count. `world` (the row's own audition price) and `true_only` are **oracle**
   selectors and are labelled as the ceiling, not competitors.
8. **Six arms.** The two `sb_sv_yk` are the arms of record; the whole sweep is 0.04 GPU-h, so
   the other four were run too, as pp1 did.

## 3. The gates. Every one below has been shown to fail (7/7 falsification cases, 6/6 arms).

Run order: `learner_tables.py --all --falsify` (local, CPU) → `own.py::own_gates` (the same on
the **volume's** own copies) → the smoke → the paid sweep.

| gate | what it asserts | closed at | falsified by |
|---|---|---|---|
| **R-0** | `mine_support == 3`, the support `ClassMiner.state` hard-codes for `keys_at_support` | 3 on all six arms | relabelling it 2 |
| **R-1** | the replayed row count == `last_build[ℓ]["n_entries"]` | 18 / 84 / 204 (s0_sv), 19 / 99 / 176 (s2_sv), all six arms | dropping one **buildable** at-support key at L3 |
| **R-1b** | the replayed `n_keys_built`, `n_lower_rows`, `inventory_max` == the logged ones | exact on all six arms | truncating one class's picks to a single row |
| **R-1c** | no `picks` index falls outside the reconstructed lower table — the recursion's own consistency check | 0 out-of-range on all six | the same two cases, at the level above |
| **R-2** | `MC.grade_table` against the true table == the logged `tab_n_correct` / `tab_n_learned` | 12/18, 29/84, 54/204 (s0_sv) — the loop's own audit numbers | the cases above |
| **R-3** | the per-row truth vector == `entry.json.gz[-1]["operative_mask"][ℓ]`, **row for row** | exact on all six arms, **locally only** | reversing the banked mask at L3 (an ORDER-only break, which R-3s, its order-blind twin, survives — that is what makes R-3 a statement about order) |
| **R-4** | the table digest == the one R-3 was closed on | exact, in every paid container | reversing the at-support keys (row set intact, order changed), **with and without** the banked mask in hand |
| **F-1** | `fire_rec` reproduces `MC.apply_any` **bit for bit** on the first fire of every level of every arm | PASS at L2/L3/L4, six arms | pp1's own falsification (`argmin` recorder) |
| **F-2b** | the re-implemented read against the arm's **banked** hold AUC, inside the readout's own drift band | \|Δ\| = 0.0035 / 0.0012 / 0.0008 / 0.0066 / 0.0005 / 0.0017 against bands 0.047 / 0.023 / 0.041 / 0.071 / 0.027 / 0.050 — **pp1's numbers to the bit** | pp1's own (the wrong trunk, AUC 0.491 against 0.819) |

**Why R-4 exists.** `entry.json.gz` is **not on the volume** — only the local mirror carries it —
so R-3, the strongest gate, cannot run inside a paid container. R-4 carries the closure there
instead: a sha256 over the child rows, the flat spellings and the truth flag of the exact table
R-3 was closed on, asserted in every container against the table it rebuilt from the volume's
own `results.json`. It closed on all six arms, which is also the evidence that the volume's
`results.json` and the local mirror's are the same object.

## 4. Defects, and their corrections

**Defect 1 — a falsification case that was vacuous on one arm.** The first `drop_key` case
removed `keys_at_support[0]` at L3. On `s2_sv` that key's halves had no picks, so dropping it
changed no row and **no gate fired** — the case reported MISSED. The correction was not to pick
a different index but to make the case *mean* what it was for: it now drops the first key whose
halves both have picks, i.e. one that actually contributes rows, and raises if there is none.
Caught by running the falsification on all six arms rather than the two of record.

**Defect 2 — a broken replay crashed instead of failing a gate.** With one L3 key dropped, the
L4 build indexed a `lower_flat` that had become too short and raised `IndexError` before any
gate could be read. A crash is a refusal, not a diagnosis. The correction is gate **R-1c**: an
out-of-range pick index is *counted* and reported as a failed gate, which says the thing that is
actually wrong ("this replay's level-(ℓ−1) table is not the one the arm built over") instead of
a stack trace.

**Defect 3 — `reconstruct` assumed the banked mask was always in hand.** It raised `KeyError`
inside the container, where `entry.json.gz` is absent. Corrected: the per-row truth vector is
computed whenever the true tables are in hand, and the *gate* is simply absent (never silently
passed) when the mask is not — with R-4 carrying the closure instead.

**Not a defect — a hypothesis of mine that the data refuted, recorded so it is not re-formed.**
When step 3 showed every score-based selector giving the *identical* audition error at L4, I
guessed the cause was a single dominant row that every read keeps and a random draw keeps only
with probability k/n. The diagnostic block says otherwise: on `s0_sv` L4 the best single row
(#32, world price 0.852) ranks 93rd under the banked read, 196th under frozen, 201st under twin
and 203rd under `dp_top` — it is in **no** read's top-29 — and yet all of them audition at
0.1875–0.1895, while a random 29 auditions at 0.66. The agreement is therefore not about that
row. The reduction reports the ranks and the pairwise Jaccard (0.34–0.42 between the selectors
at L4 on `s0_sv`) and makes **no claim about the mechanism**.

**A reproducibility wrinkle, stated so it is not read as instability.** Adding `k = 1` changed
the order in which the uniform-random selector consumes its rng, so the `random` column moved
between the tag's two passes (e.g. `s0_sv` L2 k=13: 0.4557±0.094 → 0.3846±0.116). **Every
deterministic column is identical across the passes** — checked on the whole of [P] and on every
non-random entry of [S]. The random column is a matched-count floor, not a measured quantity of
any arm, so it is the one column that may move; the reduction says so where it is printed.

## 5. What the first pass forced into the design

The first pass put the whole 204-row operative table's audition error at 0.1895 while its own
best single row priced at 0.852 success (error 0.148) — i.e. **the table looked worse than one
of its own rows**, on two different pools. That is a claim about the executor, not about the
read, and it could not be settled across pools. So `k = 1` was added to step 3's k list and the
sweep re-run at the **same seeds**, which reproduces every earlier deterministic column
bit-identically and puts the top-1 selection on the same test pools as everything else. `k = 1`
is also the hardest selection problem a read can be asked, which is why it earns its place
beyond the check.

It settled the question: on both arms of record at L4 the **best single row alone** auditions at
0.1875 / 0.1768 against the whole operative table's 0.1895 / 0.1768 and the true-rows-only
reference's 0.1885 / 0.1768. So at that cell one row is the table's equal, and the 203 others —
150 of them false — cost the world nothing and buy it nothing. At L2 and L3 the opposite holds
and sharply: k = 1 auditions at 0.51–1.00 against 0.31–0.35 for the whole table, so there the
table's size is doing the work. Both facts are in [S]; neither was a designed contrast.

## 6. Reproduction

```bash
cd experiments/                                   # MODAL_PROFILE=chromatic
PYTHONPATH=. python3 rhm/practice/voicing/sotto_voce/aliquot/preplay/learner_tables.py \
    --all --falsify --write                       # R-0..R-4 + 7/7 falsification, local
modal run rhm/practice/voicing/sotto_voce/aliquot/preplay/own.py::own_gates
modal run rhm/practice/voicing/sotto_voce/aliquot/preplay/own.py::sweep_own \
    --out-tag pp3_smoke --arms s0_sv --smoke 1
modal run --detach rhm/practice/voicing/sotto_voce/aliquot/preplay/own.py::sweep_own \
    --out-tag pp3
python3 rhm/practice/voicing/sotto_voce/aliquot/preplay/reduce_own.py --tag pp3 --fetch
```

## 7. Files

| file | purpose |
|---|---|
| `learner_tables.py` | the offline replay of the learner's operative builds, gates R-0..R-4, the falsification harness, and the `DIGESTS` R-4 carries into the containers |
| `own.py` | the Modal job: step 1 (the reconstruction, re-gated in-container), step 2 (every operative row priced), step 3 (the read as the selector), the CPU coordinator `sweep_own`, and `own_gates` |
| `reduce_own.py` | the reducer: `figures/own_reduction.txt` and `figures/pp3/<arm>_rows.tsv` |
| `figures/own_reduction.txt` | the table of record: [G] gates, [T] the tables, [H] home ground, [P] step 2, [Q] the margin over the executor's prior, [S] step 3, [C] the bill |
| `figures/<arm>_operative_tables.json` | the reconstructed operative tables, child rows + flat spellings + truth flag |
| `figures/pp3/` | the per-arm JSON and log as written on the volume, plus the per-row TSVs |
| `results/launch_pp3.log`, `results/launch_pp3b.log` | the launches of record (the second adds `k = 1`) |

Imported from pp1 and not re-implemented: `fire_rec`, `pj_features`/`pj_design`/`pj_score`/
`pj_predict`/`pj_fit`, `vo_auc`, `spearman`, the seed and readout constants, `ARMS`, `OVERTONE`,
and `pool.py::context_instances`.

## 8. Two facts carried in from pp1, not re-litigated

* `vo_heads.pt`'s `core` and its `proj` are **one plant update out of step** (pp1 NOTES defect
  1). F-2b's band is the readout's own measured per-refit drift, not a tolerance chosen here.
* the executor's own prior already separated pp1's true from wrong **constructed** entries at
  0.54–0.68, so the question these tables have to answer is what the read adds **over** that
  prior — which is why [Q] reports the margin `sep:banked − sep:dp_top` and not just `sep`.
