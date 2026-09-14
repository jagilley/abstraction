# SIZING — commit the key, not the content (`figured_bass` Q0)

Offline sizing lane for the third `enharmonic` child. One question — **what the book would have
held at every cycle after the commit had the commit frozen only the partition and the executor's
adoption, leaving the inventory live** — asked of the banked `en_s0` / `en_s1` logs. **No GPU, no
Modal, no substrate; CPU only.** Machinery: [`phase0_open.py`](phase0_open.py). Output:
`phase0_open.json`, console reduction `phase0_open.txt`.

Facts only. Interpretation is the orchestrator's.

World: v = 8, s = 2, m = 2, depth 6, `rule_seed = 0`, `mine_support = 3`, `quot_spell_cap = 4` —
the arms' own. Arms replayed: `en_s0/{given_cat_tok, given_cat_min, flat}` and
`en_s1/{flat_yk_tok, flat_yk_min}`.

**The counterfactual's scope.** The open bit changes `enharmonic.py`'s `operative()` only. It
does not change what the miners *observe*, so every per-cycle `keys_at_support` this replay is
driven by is the log's verbatim. It *would* change what the executor executes (the DP maxes over
a bigger inventory), hence the solve rate, hence the stream. **Every "open" number below is an
all-else-equal replay on the banked observation stream, not a prediction of a run** — the same
contract `tutti/sizing`'s `buildable()` and [`../../sizing/fork_arrival.py`](../../sizing/fork_arrival.py)
operate under.

**Three regimes.** `frozen` = the run. `open4` = the L4 commit adopts the level but leaves its
inventory live; L2's commit still freezes. `openAll` = no inventory is ever frozen. **L3 is live
in all three: no arm in `en_s0`/`en_s1` ever committed L3** — the `tutti` precedent the SPEC cites
is the regime these runs were already in at L3.

---

## Gate verdicts

| gate | what | verdict |
|---|---|---|
| **F-1** | the re-implemented `ClassMiner.build` reproduces every logged `last_build` field (all eight) on every cycle carrying a genuine live build | **PASS on 7 of 8 (arm, level) cells** — 11/11, 96/96, 58/58, 91/91 (`given_cat_tok` L2–L5); 13/13, 88/88, 49/49 (`given_cat_min` L2–L4). `given_cat_min` L5: **31/51**, one field (`n_lower_classes`), one window (c48–c67, all pre-commit) |
| **F-1f** | the flat replay reproduces the commit event's `n_entries` | **PASS — 6/6** (`flat` L2 13/13; `flat_yk_tok` L2 11/11, L4 1/1; `flat_yk_min` L2 11/11, L4 3/3, L5 1/1) |
| **F-2** | is the answer pick-invariant? class coverage and L5 lookup-ability under `pick=lo` / `pick=hi` / cap-free | **`tok`: PASS** — 9 classes, 52 lookup-able pairs under all three. **`min`: FAIL** — 5/4/5 classes, 19/14/19 pairs |
| **F-3** | row IDENTITY: the committed book's `true_mask` elementwise | **PASS at L2 in every arm** (16/16 rows, 10 true; `flat` 13/13, 9 true). **FAIL at L4/L5 under a class key** — see below |

### What F-1's one failure and F-3 are about

`ClassMiner.build`'s `pick()` ranks a class's lower rows by the miner's own `spell` counts before
truncating at `spell_cap`, and **those counts are not logged in any banked tag**. Three exact
consequences, measured rather than assumed:

* **Row counts are exact under any pick** — the count is `min(|class|, cap)` either way. This is
  what F-1 confirms on 437 of 457 checked cycle-levels.
* **Under the `tok` key class coverage is exact**, because a token class id is a function of the
  class pair by the closure property `quotient.py` documents — which spellings are picked cannot
  move it. F-2 confirms it against both the reversed pick and a cap-free build.
* **Row identity is not reproduced, and under the `min` key class coverage is not either** (a
  min-label class holds rows of several token classes, so which survive the cap changes what the
  entries compose to). The committed L4 book's true-row count: `given_cat_tok` logged 16 of 64,
  replay 9 (`lo`) / 1 (`hi`); `given_cat_min` logged 17 of 144, replay 21 / 4. **So every `min`
  number in §1–§2 is given as `lo`, and F-2 brackets it.** Precision is not reported at all: the
  logged value sits outside the `lo`/`hi` bracket, so this replay cannot speak to it.

Instrumenting `ClassMiner` to log `spell` per cycle would close F-3 for good and costs nothing at
run time.

---

## 0. The clocks — where the commit sits in them

The committable miners are era-gated to `era_level + 1`, so the L4 miner starts at era 3 and the
L5 miner at era 4. That gate, not the freeze, is why an L5 key count is 0 before era 4.

| arm | key | cycles | era starts | commits | L4 book frozen for |
|---|---|---|---|---|---|
| `given_cat_tok` | tok | 176 | 1:c1 2:c36 3:c86 4:c156 5:c168 | L2@c12, **L4@c95** | **82 of 176 cycles (47%)** |
| `given_cat_min` | gen | 114 | 1:c1 2:c38 3:c44 4:c94 5:c106 | L2@c14, **L4@c87**, L5@c99 | 28 of 114 cycles (25%) |
| `flat` | flat | 93 | 1:c1 2:c61 3:c71 4:c79 5:c85 | L2@c48 (**no L4 commit**) | — |
| `flat_yk_tok` | flat | 176 | (yoked to `given_cat_tok`) | L2@c12, L4@c95 | 82 of 176 (47%) |
| `flat_yk_min` | flat | 114 | (yoked to `given_cat_min`) | L2@c14, L4@c87, L5@c99 | 28 of 114 (25%) |

`given_cat_tok`'s L4 commit lands **9 cycles after the L4 miner starts** (era 3 opens at c86) and
**61 cycles before the L5 miner starts** (era 4 opens at c156).

---

## 1. The live inventory's class coverage, per cycle after the commit

`own` = classes in the arm's own key (`tok`: of the 13 L4 token classes; `min`: of 8 min-labels).
Rows shown where something moves.

### `given_cat_tok` (L4 commit c95)

| cycle | L4 keys@sup | frozen rows / classes | open4 rows / classes | openAll rows / classes | live L3 rows / classes |
|---|---|---|---|---|---|
| 95 (commit) | 5 | 64 / **3** | 64 / 3 | 64 / 3 | 91 / 8 |
| 96 | 6 | 64 / 3 | 80 / 4 | 80 / 4 | 91 / 8 |
| 98 | 8 | 64 / 3 | 96 / 5 | 112 / 5 | 91 / 8 |
| 100 | 9 | 64 / 3 | 112 / 6 | 128 / 6 | 91 / 8 |
| 108 | 11 | 64 / 3 | 128 / 7 | 160 / 7 | 91 / 8 |
| 111 | 12 | 64 / 3 | 144 / **8** | 176 / **8** | 91 / 8 |
| 138 | 13 | 64 / 3 | 160 / 8 | 192 / 8 | 91 / 8 |
| 172 | 14 | 64 / 3 | 176 / **9** | 208 / **9** | 94 / 9 |
| 176 (end) | 15 | **64 / 3** | 176 / 9 | **224 / 9** | 95 / 9 |

**3 → 8 of 13 classes within 16 cycles of the commit; 9 by the end. The frozen book stayed at 3
for all 82 remaining cycles.** The classes are named: frozen holds `{0,3,6} {1,5,7} {3}`; live
holds `{0} {0,3,6} {0,6} {1} {1,5,7} {2} {3} {3,6} {6}`. Freezing kept two ambiguity sets and one
singleton, and locked out every other singleton the level had.

**All 15 L4 keys at support at the end are legal L4 class pairs** (of the 40 the closure
enumerates); 5 of 5 at the commit. Zero junk keys at support under the class key.

### `given_cat_min` (L4 commit c87), `pick=lo`

| cycle | L4 keys@sup | frozen rows / min-cls / tok-cls | open4 | openAll | live L3 rows / min-cls |
|---|---|---|---|---|---|
| 87 (commit) | 11 | 144 / **4** / 6 | 144 / 4 / 6 | 160 / 4 / 6 | 90 / 7 |
| 111 | 11 | 144 / 4 / 6 | 144 / 4 / 6 | 160 / 4 / 6 | 102 / 8 |
| 112 | 12 | 144 / 4 / 6 | 160 / **5** / 7 | 176 / **5** / 7 | 102 / 8 |
| 114 (end) | 12 | **144 / 4 / 6** | 160 / 5 / 7 | **176 / 5 / 7** | 102 / 8 |

**4 → 5 min-labels (6 → 7 token classes) over the arm's whole remaining lifetime, and the move
happens once, at c112 of 114.** Under `pick=hi` the live book stays at 4 / 5 and the gain is
zero (F-2).

### L3, which was never frozen

No arm committed L3, so the L4 build always looked up in a live L3 set. At the L4 commit that set
held **91 rows in 8 of the 11 L3 token classes** (`given_cat_tok`) and **90 rows in 7 classes**
(`given_cat_min`); by the end, 95 rows / 9 classes and 102 rows / 8 classes. The L4 book's 3
classes at the commit were therefore **not** a shortage of L3 inventory — the L3 side of the
ratchet was already 8 classes deep. The binding object at c95 was the L4 miner's own **5** keys at
support, which grew to 15 by c176 with nowhere to go.

---

## 2. What L5 could have built over it

`look` = legal L5 class pairs both of whose halves the L4 book holds a row of (the `buildable()`
idiom in class coordinates), of **73** in token coordinates / **25** in min coordinates.
`L5k@s` = the committable L5 miner's own keys at support 3. `obs5` = the ungated observation
panel's count (`obs_hist[5]`) — a different miner over the same stream (it observes in every era
at that era's own node, so it is not era-gated and not node-matched to the committable one).

### `given_cat_tok`

| cycle | L5k@s | obs5 | frozen look / built / rows | open4 look / built / rows | openAll look / built / rows |
|---|---|---|---|---|---|
| 95 (commit) | 0 | 24 | **6** / 0 / 0 | 6 / 0 / 0 | 6 / 0 / 0 |
| 96 | 0 | 25 | 6 / 0 / 0 | 11 / 0 / 0 | 11 / 0 / 0 |
| 100 | 0 | 26 | 6 / 0 / 0 | 24 / 0 / 0 | 24 / 0 / 0 |
| 111 | 0 | 27 | 6 / 0 / 0 | **43** / 0 / 0 | **43** / 0 / 0 |
| 156 (era 4) | 1 | 36 | 6 / 1 / 16 | 43 / 1 / 16 | 43 / 1 / 16 |
| 161 | 4 | 36 | 6 / 2 / 32 | 43 / 3 / 48 | 43 / 3 / 48 |
| 162 | 6 | 36 | 6 / 2 / 32 | 43 / 4 / 64 | 43 / 4 / 64 |
| 176 (end) | 7 | 37 | **6 / 2 / 32** | 52 / 5 / 80 | **52 / 5 / 80** |

**First cycle with a non-empty L5 build: c156 under all three regimes.** The L5 miner's era gate,
not the L4 book, is what set that cycle. What the freeze cost is the *content*: at the end
**5 of the 7 L5 keys at support name a class the frozen book does not hold**; over the live book
only 2 do. Lookup-able pairs: **6 of 73 frozen, 43 of 73 by c111, 52 of 73 by c172.**
`tok` never committed L5 in the run; the build it would have committed is 2 keys / 32 rows frozen
against 5 keys / 80 rows open.

**All 7 L5 keys at support are legal class pairs** (of 73). The ungated observation panel already
carried **24** L5 class-pair keys at support at the moment of the L4 commit and 37 by the end,
against the committable miner's 0 until c156.

### `given_cat_min` (L4 commit c87, L5 commit c99), `pick=lo`

| cycle | L5k@s | obs5 | frozen look / built / rows | open4 | openAll |
|---|---|---|---|---|---|
| 87 (commit) | 0 | 15 | 14 / 0 / 0 | 14 / 0 / 0 | 14 / 0 / 0 |
| 95 | 2 | 15 | 14 / 1 / 16 | 14 / 1 / 16 | 14 / 1 / 16 |
| 99 (**L5 commits**) | 3 | 15 | 14 / 2 / 32 | 14 / 2 / 32 | 14 / 2 / 32 |
| 104 | 4 | 15 | 14 / 3 / 48 | 14 / 3 / 48 | 14 / 3 / 48 |
| 112 | 5 | 15 | 14 / 3 / 48 | 19 / 4 / 64 | 19 / 4 / 64 |
| 114 (end) | 5 | 15 | **14 / 3 / 48** | 19 / 4 / 64 | 19 / 4 / 64 |

**First non-empty L5 build: c95 under all three regimes** — 8 cycles after the L4 commit and 4
before the L5 commit the run actually made. The L5 commit at c99 took 2 keys / 32 rows, matching
the logged `n_entries` 32. At the end 2 of 5 L5 keys at support are blocked by the frozen book, 1
by the live one. Lookup-able min-pairs: 14 of 25 frozen, 19 of 25 open.

---

## 3. The flat key, replayed the same way

A flat arm's book *is* its at-support keys filtered by the ratchet, so this replay is exact
(F-1f, F-3 at L2). `tok` = token classes the rows cover, the coordinate §1's contrast is in.

| arm | cycle | L4 keys@sup | frozen rows / tok | open4 rows / tok | openAll rows / tok | openAll L2/L3 rows | L5 rows (frozen → openAll) |
|---|---|---|---|---|---|---|---|
| `flat_yk_tok` | 95 (commit) | 1 | **1 / 1** | 1 / 1 | 1 / 1 | 16 / 30 | 0 → 0 |
| | 117 | 8 | 1 / 1 | 3 / 3 | 4 / 4 | 17 / 33 | 0 → 0 |
| | 176 (end) | 14 | **1 / 1** | 3 / 3 | **7 / 4** | 18 / 40 | **1 → 1** |
| `flat_yk_min` | 87 (commit) | 21 | **3 / 3** | 3 / 3 | 3 / 3 | 16 / 20 | 0 → 0 |
| | 114 (end) | 24 | **3 / 3** | **3 / 3** | **3 / 3** | 18 / 26 | **1 → 1** |
| `flat` (en_s0) | 93 (end) | 2 | 1 / 1 | 1 / 1 | 1 / 1 | 16 / 17 | 0 → 0 |

`en_s0/flat` never committed L4 at all, so there is nothing to open there; the yokes are the
matched-clock flat comparators.

* **`flat_yk_tok`: 1 → 7 rows and 1 → 4 token classes under `openAll`** (1 → 3 rows / 3 classes if
  only L4 opens; the rest of the gain comes from L2 staying live, 11 → 18 rows, which carries L3
  from 19 to 40).
* **`flat_yk_min`: 3 → 3 rows, 3 → 3 classes. Opening the inventory changes nothing at all** —
  its 24 L4 keys at support are blocked by the flat ratchet (their halves are not rows of the L3
  book), not by the freeze.
* **The L5 flat build is 1 row frozen and 1 row open in both yokes.** Opening the flat inventory
  moves no L5 row in either arm.

---

## 4. The executor's exposure

`macro_features`' max-sum DP runs over the operative table's *entries*; under a class key each
surviving key contributes up to `spell_cap ** s = 16` spellings, so the expansion choice *is* the
inventory. `inv max/mean` and `cap binds` are the `openAll` L4 build's own
`inventory_max` / `inventory_mean` / `n_class_capped`.

| arm | key | cycle | frozen | open4 | openAll | openAll / frozen | inv max | inv mean | cap binds | L5 rows frozen → openAll |
|---|---|---|---|---|---|---|---|---|---|---|
| `given_cat_tok` | tok | 95 | 64 | 64 | 64 | 1.00 | 16 | 16.00 | 9 | 0 → 0 |
| | | 136 | 64 | 144 | 176 | 2.75 | 16 | 16.00 | 21 | 0 → 0 |
| | | 176 | **64** | 176 | **224** | **3.50** | 16 | 16.00 | 26 | **32 → 80** |
| `given_cat_min` | gen | 87 | 144 | 144 | 160 | 1.11 | 16 | 14.55 | 17 | 0 → 0 |
| | | 114 | **144** | 160 | **176** | **1.22** | 16 | 14.67 | 22 | **48 → 64** |
| `flat_yk_tok` | flat | 176 | **1** | 3 | **7** | **7.00** | — | — | — | 1 → 1 |
| `flat_yk_min` | flat | 114 | **3** | 3 | **3** | **1.00** | — | — | — | 1 → 1 |

**The cap is what bounds this.** The same last cycle with `spell_cap` off — the full cross-product
the class key licenses:

| arm | L4 rows (cap 4) | L4 rows (cap-free) | L5 rows (cap 4) | L5 rows (cap-free) |
|---|---|---|---|---|
| `given_cat_tok` | 224 | **1,693** | 80 | **19,025** |
| `given_cat_min` | 176 | **2,512** | 64 | **591,936** |

At the L4 commit cycle the cap-free L4 cross-product would already have been 649 rows
(`given_cat_tok`) and 1,632 (`given_cat_min`), against the 64 and 144 the run committed.
`n_class_capped` rises from 9 to 26 across `given_cat_tok`'s open replay, i.e. the cap binds on
more classes as the inventory opens.

---

## Caveats

- One rule draw (`rule_seed = 0`), one seed per arm, five banked arms. Every class alphabet,
  cover and junk number inherits [`../../sizing/SIZING.md`](../../sizing/SIZING.md)'s draw
  dependence.
- **The open columns are a replay on a fixed observation stream.** A real open arm executes over
  a bigger inventory, so its solve rate, its mined stream and therefore its own
  `keys_at_support` would differ from the ones driving this replay. Nothing here bounds that
  difference in either direction.
- **`min` is bracketed, not exact** (F-2, F-3): the unlogged `spell` ranking moves its class
  coverage between 4 and 5 and its lookup-able pairs between 14 and 19.
- **No precision, recall or grading number is reported.** The replay does not reproduce row
  identity under a class key (F-3), so it cannot speak to what the open book's rows *are*, only
  to how many there are and which classes they cover.
- `obs_hist[5]` is the ungated observation panel's count. It observes at each era's own node
  (`(era.node * s**(era.level-1)) // span`), so it accumulates across nodes and is not the same
  object as the committable, era-gated, node-fixed L5 miner. It is reported as context for L5
  arrival, not as a substitute for `L5k@s`.
- F-1 checks only the cycles where `log["quot"][c]["build"][ell]` carries a genuine build
  (`n_lower_rows > 0` and the level not yet committed). 457 cycle-levels were checked and 703
  skipped; a committed level's entry is stale and the audition's `build(base_table(v))`
  counterfactual leaves a degenerate one.
- The L4 → L5 lookup-ability count treats a class as held if the book has **any** row of it; the
  DP's ability to *use* that row is a separate question this lane does not touch.

---

## Reproduction

```
cd experiments/
python3 rhm/practice/enharmonic/figured_bass/sizing/phase0_open.py
```

Reads the compact mirrors under `../../figures/{en_s0,en_s1}/` and writes `phase0_open.json`
beside the script; `phase0_open.txt` is the console reduction of record.
