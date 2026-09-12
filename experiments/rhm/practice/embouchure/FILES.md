# FILES — `embouchure` (the chooser trained on its own attempts)

Machinery record for the node, in [`../inflection/FILES.md`](../inflection/FILES.md)'s register.
**No results are interpreted here** — facts, gate numbers and the touch-point map only.
Decisions live in [`DESIGN.md`](DESIGN.md); the question, the arms and the sequence are in
[`SPEC.md`](SPEC.md). There is deliberately no `README.md` (repo norm: results are discussed
before a writeup).

**Direct donor** (untouched): [`../inflection/`](../inflection/FILES.md) — `inflection.py` is
forked here as `embouchure.py`, `analyze_inflection.py` as `analyze_embouchure.py`.
`inflection.py` is the in-process G-F reference.
**Sibling it answers to**: [`../enharmonic/`](../enharmonic/README.md) finding 5. Not touched;
another agent is on that lineage.
**Reads, does not write**: `../inflection/figures/<tag>/` (the banked Q1c/Q3 logs) and
`../inflection/sizing/SIZING.md` (the identifiability ladder's method).

## Files

| file | purpose |
|---|---|
| `SPEC.md` | the orchestrator's prompt: the question, the arms, the sequence |
| `embouchure.py` | the substrate: `inflection.py` forked; `fit_signal`, the own-write bag, the bag objective, the RB-1 gate; entrypoints `preflight`, `fidelity_smoke`, `embouchure_run`, `gates_cpu` |
| `analyze_embouchure.py` | the reduction: `analyze_inflection.py`'s sections plus §1E–§8E (own twin, time course, volume, mis-phrasing, mis-keying, recovered K at the shared cells, the in-tag `e` floor, the mined table in two spaces), the RB-1 block, and `--merge-tag` as a comma-separated list with cross-tag 0.000e+00 assertions |
| `phase0_embouchure.py`, `phase0.json` | Q0 (CPU, offline, from the banked `inflection` logs) |
| `phase0_q2_lexicon.py`, `phase0_q2.json` | Q2.0 (CPU, offline): how much homophony the world tolerates |
| `figures/q0_reduction.txt` | Q0's printed reduction |
| `figures/q2_lexicon_reduction.txt` | Q2.0's printed reduction (the CPU ladder and the reader leg's sizing) |
| `figures/em_q1_reduction.txt`, `figures/em_q1_joint_reduction.txt` | Q1: the single-tag and the four-tag joint reductions |
| `figures/em_q2_reduction.txt` | Q2.1's reduction (the homophonous lexicon at H=3) |
| `figures/<tag>/` | the fetched artifacts per tag (`setup.json`, `results.json`, `done.txt`) |
| `launch_detached.py` | the session-isolated launcher (`inflection`'s, retargeted) |
| `results/launch_<tag>.log`, `results/preflight*.log` | every launch and preflight log, one per run |
| `results/wait_app.sh` | the waiter on the Modal **app state** rather than the local log (survives a container restart) |
| `DESIGN.md` | the decisions the code and the data forced |

## Volume layout

`rhm-scaling-data:/data/rhm_practice_embouchure/<tag>/` (the app is `rhm-practice-embouchure`;
the donors' tags are never written to). Fetched copies and reductions under `figures/<tag>/`
and `figures/<tag>_reduction.txt`.

---

# Q0 — the four offline questions (2026-09-10)

CPU, no Modal, no substrate, ~4 min. Everything below is read out of
`../inflection/figures/{if_q1c_yk,if_q3_e,if_q3_e_sp,if_q3_f2}/<arm>/results.json` and
`setup.json`. Reproduce:

```bash
cd experiments/
PYTHONPATH=. python3 rhm/practice/embouchure/phase0_embouchure.py \
    > rhm/practice/embouchure/figures/q0_reduction.txt
```

The world throughout is `if_q1c_yk`'s: `E_R8`, `rule_seed 6` (collision-free),
θ = [1, 1, 6, 4, 5, 5, 5, 1], practised {0, 3, 7}, held out {1, 2, 4, 5, 6}
(alias {1, 2, 6}, novel {4, 5}), `n_pr` = 64 instances/cycle, 32 bottom blocks/instance,
140 cycles, seed 0.

## Q0.0 — where the banked channels actually live

| object | what it is | where it comes from |
|---|---|---|
| `log["render"]["n"]` | the SURFACE channel's cumulative buffer | `rbuf`, filled from `x_np[sol_i]` — every bottom block of every SOLVED observation, parsed by `shared["reader"]`, labelled by the observed synonym |
| `log["spell_rows"]` | the VERDICT channel, capped at 512/cycle | `out["x"][sol_i]` — every bottom block of the learner's own ANSWER at the solved instances, as `(feature-as-read, register, k_written, k_rule)` |

**`spell_rows` exists only where the arm has a rendering organ** (`rhead` or `spell_store`):
`canon_s`, `given_rule_s`, `m_given_rule` carry **none**, in every tag. Rows banked:

| tag | arm | cycles with rows | total rows |
|---|---|---|---|
| `if_q1c_yk` | `fit_rule_s` / `fit_index_s` / `leaf_s` | 140 / 140 / 140 | 60,928 / 60,544 / 42,882 |
| `if_q3_e` | `m_fit_rule` / `m_leaf` | 137 / 139 | 51,616 / 42,467 |
| `if_q3_e_sp` | `m_fit_rule_sp` / `m_leaf_sp` | 140 / 62 | 55,552 / 24,948 |
| `if_q3_f2` | `leaf_s` | 140 | 42,882 |

The 512 cap binds in 70/09/51 % of era-1/2/3 cycles for `fit_rule_s`, 65/02/00 % for `leaf_s`.

**Row identities, checked on all 100k+ rows**: `k_rule == K[feature, register]` in **every**
row, in every arm and tag. The registers visited are **{0, 3, 7} and nothing else** — exactly
`practiced_set`, in BOTH channels, because the priced practice draw takes
`context_instances(..., practiced=practiced)`.

## Q0.1 — `own_verdict` fitted offline from `spell_rows`

`inflection`'s own head (`build_rule_head`, one linear layer over [one-hot feature ; scalar
register]), its own optimiser (Adam, lr 3e-2), its own step count (64/cycle), batch (256),
buffer cap (200,000, keep-last) and its own report (`rule_head_report`). Label = `k_rule`.

| fit | source | n_buf at c140 | acc_practised | acc_held-out | det θ | θ̂ |
|---|---|---|---|---|---|---|
| `own_verdict` full logged rows | `if_q1c_yk/fit_rule_s` | 60,928 | **1.0000** | **0.8750** | 0.375 | [2,2,5,5,5,5,5,2] |
| `own_verdict` at own-write volume (×0.2812) | same | 17,136 | **1.0000** | **0.8750** | 0.375 | [2,2,5,5,5,5,5,2] |
| `own_verdict` misspelled rows only | same | 97 | 0.6250 | 0.5750 | 0.000 | [0,2,1,1,0,0,0,2] |
| `own_verdict` full logged rows | `if_q1c_yk/leaf_s` | 42,882 | **1.0000** | **0.8750** | 0.375 | [2,2,5,5,5,5,5,2] |
| `own_verdict` at own-write volume | same | 12,058 | **1.0000** | **0.8750** | 0.375 | [2,2,5,5,5,5,5,2] |
| `own_verdict` misspelled rows only | same | 4,912 | 0.8333 | 0.6000 | 0.000 | [2,2,1,1,4,1,1,2] |
| **`surface`, banked** (`log["render"]`) | `if_q1c_yk/fit_rule_s` | 76,319 | **1.0000** | **0.8750** | 0.375 | **[2,2,5,5,5,5,5,2]** |
| `surface`, banked, one-hot | `if_q1c_yk/fit_index_s` | 76,831 | 1.0000 | 0.7500 | 0.375 | [1,1,7,7,7,7,7,1] |

**Harness noise** (the run's fit stream shares `frng` with the spell-row subsample and cannot
be replayed, so cycles-to-convergence is only comparable WITHIN this harness). First cycle at
which `acc_heldout` reaches its endpoint and stays, four fit seeds:

| volume | seeds | first cycle at endpoint | endpoint |
|---|---|---|---|
| full logged rows | +0/+1/+2/+3 | **9 / 8 / 10 / 14** | 0.8750, θ̂ [2,2,5,5,5,5,5,2] in 4/4 |
| own-write volume (×0.2812) | +0/+1/+2/+3 | **15 / 11 / 17 / 16** | 0.8750, θ̂ [2,2,5,5,5,5,5,2] in 4/4 |

## Q0.2 — volume: own written blocks vs harvested surface blocks, per cycle

`render["n"]` grows by **exactly `n_solved × 32`** every cycle (99.3 % of cycles, the residue
being the 200k cap and integer rounding of `n_solved` from `e_practice`). Own writes are not
logged as a count; they are recovered from two logged numbers that share the graded
configuration as denominator —

```
own written blocks per instance = 32 × e_sp_practice / spell["spell_err"]
```

— the numerator being exact (the world spells every block it wrote on-rule, so only an own
write can be wrong) and the denominator being the renderer's error over the beam's whole render
stream rather than over the accepted answer alone. Pooled over the 18 (arm, era) cells where
the renderer misspells at a measurable rate:

**own-write fraction of the graded configuration's 32 blocks = 0.2812, range [0.2512, 0.3257]**
— i.e. ≈ 9 of 32 blocks per instance, stable across `canon`/`leaf`/`fit_*` and across eras.

Per cycle, `if_q1c_yk`:

| arm | era | n_solved | surface blocks/cyc | own blocks/inst | own blocks/cyc | surface ÷ own |
|---|---|---|---|---|---|---|
| `canon_s` | 1 / 2 / 3 | 24.1 / 10.7 / 9.8 | 773 / 341 / 314 | 8.04 / 8.67 / 9.24 | 514 / 555 / 591 | **1.50 / 0.62 / 0.53** |
| `leaf_s` | 1 / 2 / 3 | 27.1 / 8.7 / 8.3 | 869 / 278 / 265 | 8.52 / 9.60 / 10.42 | 545 / 615 / 667 | **1.59 / 0.45 / 0.40** |
| `fit_rule_s` | 1 | 25.9 | 827 | 10.61 | 679 | 1.22 |
| `m_leaf` (`if_q3_e`) | 1 / 2 / 3 | 40.0 / 7.7 / 8.2 | 1280 / 247 / 261 | 8.32 / 9.63 / 10.07 | 533 / 616 / 645 | 2.40 / 0.40 / 0.41 |

The surface harvest is **solve-gated** (`sol_i = flatnonzero(ps > 0.5)`) and own writes are
not: every one of the 64 attempted instances carries own writes and a per-row spelling label.
`n_solved` falls from ~26/64 in era 1 to ~9/64 in era 3, which is where the ratio inverts. At
MATCHED instance sets the ratio is a flat **0.2812** at every era.

By register, both channels: **{0: ~33 %, 3: ~34 %, 7: ~33 %}, and 0 at every held-out
register**, in every arm.

## Q0.3 — the read-back check

**Analytic half.** `observed_synonyms` marks a block `good` iff its code is one of the m tuples
of the feature the reader returned; with no bottom-tuple collision every legal code has exactly
one owner, so `good` ⟺ the reader returned that code's true owner, and a misspelling — the
OTHER synonym of the SAME feature — is a legal code owned by that same feature.

| draw | legal codes of v·m = 16 | collisions | read-back is the identity for an exact reader |
|---|---|---|---|
| `rule_seed 0` (the collision draw) | 14 | **2** — code 12 = (f1,k1)/(f4,k1); code 39 = (f2,k0)/(f7,k0) | **no** |
| **`rule_seed 6`** (every clean tag) | **16** | **0** | **yes** |

**Measured half.** `spell_rows` keeps only `good` blocks of the learner's own answer and
`render["n"]` only `good` blocks of the observation; both denominators are known
(`n_solved × 32`), so the shortfall IS the reader's block error on those configurations.

| tag / arm | rows == min(n_sol·32, 512) | read-back acc, uncapped cycles | misspelled own-write rows | P(fail \| misspelled) | P(fail \| on-rule) | enrichment |
|---|---|---|---|---|---|---|
| `if_q1c_yk/fit_rule_s` | **1.0000** (140/140) | **1.000000** | **97** | 0.063\* | −0.000 | — |
| `if_q1c_yk/fit_index_s` | **1.0000** | **1.000000** | 68 | 0.000 | 0.000 | — |
| `if_q3_e/m_fit_rule` | **1.0000** | **1.000000** | 97 | 0.000 | 0.000 | — |
| `if_q3_e_sp/m_fit_rule_sp` | **1.0000** | **1.000000** | 97 | 0.000 | 0.000 | — |
| `if_q1c_yk/leaf_s` | 0.2000 | 0.885341 | 4,912 | 0.195 | 0.103 | 1.90 |
| `if_q3_e/m_leaf` | 0.2786 | 0.877734 | 4,422 | 0.264 | 0.102 | 2.58 |
| `if_q3_e_sp/m_leaf_sp` | 0.6129 | 0.883081 | 2,393 | 0.255 | 0.099 | 2.56 |

\* the conditional split is an arithmetic decomposition of a **zero** drop rate against
`e_sp_practice` measured over all 64 instances while the rows are the solved subset; where the
drop is 0.0000 in every cycle the split has no content. The load-bearing number in those four
rows is that **the row count equals `min(n_solved·32, 512)` in every one of 557 cycles**, so
not one of the 359 misspelled own-write blocks in the four `fit_*` arms was dropped.

The reader-parsed rows' misspelling rate against the same quantity through the EXACT inverse
map (`e_sp_practice`), by era — the two agree, so misspelled blocks are not being lost:

| arm | era 1 | era 2 | era 3 |
|---|---|---|---|
| `leaf_s` reader-rows / exact-map | 0.10545 / 0.11841 | 0.10688 / 0.12424 | 0.13442 / 0.13302 |
| `fit_rule_s` reader-rows / exact-map | 0.00501 / 0.00594 | 0.00000 / 0.00000 | 0.00000 / 0.00000 |

`setup.json → plant.by_register.read_acc` = **1.000 at all eight registers** (4,096 blocks
each), practised and held out, on clean rule-spelled probes; θ ∈ [1, 6] so every feature's
BOTH synonyms are exercised across the eight columns.

## Q0.4 — identifiability at own-write volume

`SIZING.md` §2's `ladder_rowsep`, generalised from a practised COLUMN SET to the cells the
stream has actually delivered by cycle c (`ladder_observed`: a class member is consistent iff
it agrees with K on exactly the observed cells; `det` = features pinned at the held-out
register, `acc` = expected accuracy of a uniform draw over consistent rows). With `obs[f]` =
the practised set for every f it reduces to `ladder_rowsep` exactly.

Ceiling with all 24 practised cells observed, **this** subset {0,3,7} and **this** θ draw
(SIZING's §2 table averages over practised subsets and reports 0.54 / 0.81 at C = 3):

| class | det | acc |
|---|---|---|
| `additive_scalar` (monotone rows, scalar register) | **0.4750** | **0.7875** |
| `table` / one-hot | 0.0000 | 0.5000 |

**All 24 practised (feature, register) cells are observed in cycle 1, at BOTH volumes** — the
surface stream (1,024 rows in c1) and the own-write-volume stream (144 rows in c1). `det`/`acc`
are therefore at the ceiling from c1 onward in both, and neither channel is ever
volume-limited on this ladder. The SPEC's stop condition (Q0 (iv) says the rule is not
identifiable from own writes at any budget this ladder reaches) is **not** triggered.

## Q0.5 — what Q0 could not settle

- **Position.** The banked verdict channel is the whole ANSWER (≈ 9 own-written blocks + ≈ 23
  untouched world blocks per instance); which rows are own writes is not recoverable from the
  log. Every offline fit above therefore emulates own-write **volume**, not own-write
  **position**. The one certainly-own-write subset available — the misspelled rows — is
  selected by "the current renderer got this cell wrong", which conditions on the label, and
  its fit (0.5750–0.6000 held-out, det 0.000) is a property of that selection, not of own
  writes. A lower bound on own-write cell coverage from the misspelled rows alone:
  **16 of 24 cells** (`leaf_s`), **12 of 24** (`fit_rule_s`).
- **The bag-level channel (`own_scalar`).** Bag membership per instance is not banked, so the
  arm's information content cannot be simulated offline at all.
- **The neural reader under a controlled flip.** The read-back numbers above are observational
  (what the reader did on the answers each arm happened to write). The controlled version —
  take a clean rule-spelled configuration at register ρ, flip one block to the other synonym,
  re-read — needs the frozen reader and therefore `build_shared` (≈ 610 s of setup). It is
  queued as a preflight gate, not a launch.

## Gates

| gate | what it asserts | status |
|---|---|---|
| **Q0-1** | `k_rule == K[feature, register]` in every banked `spell_rows` row (100,000+ rows, 8 arm-tags) | **PASS** |
| **Q0-2** | `render["n"]` increments by exactly `n_solved × 32` | **PASS** (99.3 % of cycles; residue = the 200k cap) |
| **Q0-3** | `spell_rows` row count == `min(n_solved × 32, 512)` in every cycle of every `fit_*` arm | **PASS** (557/557 cycles) |
| **Q0-4** | `rule_seed 6`'s bottom map has 0 collisions in 16 legal codes; `rule_seed 0` has 2 in 14 | **PASS** |
| **Q0-5** | `ladder_observed` with `obs[f]` = the practised set reproduces `SIZING.md`'s `ladder_rowsep` | **PASS** (det 0.4750 / acc 0.7875 at C = 3, {0,3,7}) |

---

# Q1 — the fork, its gates, and the three arms (2026-09-11)

## The one bit

`embouchure.py` forks `inflection.py`: **+423 lines, 26 `# [embouchure]` marks**, every knob
default off. The knob is `fit_signal`:

| value | the renderer's diet |
|---|---|
| `"surface"` (default — `inflection.py` exactly) | the rule-spelled blocks the WORLD wrote, in the instances the learner SOLVED, parsed by the frozen reader, labelled by the observed synonym |
| `"own_scalar"` | the blocks the LEARNER wrote, in the instances it ATTEMPTED, labelled only by `spell["per_row"] × 32 / n_own` — one scalar per instance |
| `"own_readback"` | the same blocks, labelled per block by whether the learner's own frozen reader gives back the feature it MEANT. **No grader number anywhere.** |

**The arms are four across two tags** (the coordinator's decisions, 2026-09-10 and
2026-09-11): `em_q1` runs `canon_s`, `fit_rule_s`, `own_scalar_s`; `em_q1b` runs `canon_s`,
`own_readback_s` on a second L4, in parallel. `canon_s` is in both tags and is asserted
**bit-identical across them** on every logged series (the cross-tag anchor in
`analyze_embouchure.py`'s `--merge-tag` path) — that assertion is what makes the two tags one
design rather than two experiments.

The three original arms (the coordinator's decision, 2026-09-10): `canon_s`, `fit_rule_s`
(called **`surface`** in the reduction — the banked `if_q1c_yk` arm, config-identical, re-run)
and **`own_scalar_s`**. `own_verdict`, `surface_matched` and the read-back arm are not run;
the two links Q0 closed are §Q0.1 (own_verdict ≡ surface's exact table) and §Q0.1's seed
spread (a 3.6× volume cut moves the endpoint not at all). `given_rule_s` and `leaf_s` are read
from `../inflection/figures/if_q1c_yk/` and **not re-run**.

## What the fork adds (each `# [embouchure]`-marked)

| addition | where | why |
|---|---|---|
| `fit_signal` + `own_buf_cap` (bags) + `own_batch` (bags/step) | `_cfg` | the one knob; default `"surface"` |
| `own_write_mask` | module | which blocks the learner wrote, per instance, from the accepted move sequence's own `blk0`/`span` — geometry, no replay of the beam, last-writer-correct by construction |
| `own_bag_rows` | module | the bag (cells read back through the reader's parse and `k_of`, the SAME two instruments the surface channel uses — never the exact inverse map), the label, and gates EB-1 / EB-2 |
| `own_buf_append` | module | bags padded to 32 with a mask; keep-last cap on BAGS |
| `rule_head_fit_bag` | module | the bag objective (below) |
| `readback_flip_check` | module | gate RB-1, the controlled read-back test |
| `Renderer.last_k`, `.tally_state()`, `.tally_restore()` | module | unread in the shipped path (the bag builder no longer replays); kept because `last_k` is the only place the synonym of a write is available without the inverse map |
| `log["own"]`, `log["ans_hash"]` | `run_arm` | the production route's per-cycle record, and a crc32 of the practice answer so the in-tag twin gate is a measured cycle. Consumed by nothing; `ans_hash` is `None` at `rule=None`, so the G-F path never runs it |
| arm `own_scalar_s` + its `TWIN` row | module | `fit_rule_s` in every config bit but `fit_signal` (asserted — gate EB-0) |
| `results.json → own_gate`, `fit_signal` | `run_arm` | the whole-arm gate tally |

**The bag objective.** `p = σ(head(f, ρ))` is P(synonym 1); the predicted probability that a
given write was wrong is `p` where synonym 0 was written and `1 − p` where synonym 1 was, so
`ŷ = mean_j q_j` over the bag and the loss is `BCE(ŷ, y)`. At `y = 0` the gradient pushes every
cell of the bag toward the synonym that WAS written; at `y > 0` it pushes them away. Feedback-
error learning with a bag credit assignment. Its dead zone is stated in advance, not
discovered: a renderer right everywhere it writes has `y = 0` and therefore no gradient, and no
exploration noise is added.

## Gates

| gate | what it asserts | status |
|---|---|---|
| **G-F** | with every `[embouchure]` knob off (`rule=None`), this fork replays **`inflection.py`** in process, against a donor self-replay control | **PASS — 0.000e+00** on `anchor` and `given_c1`, control 0.000e+00, commits equal. `em_gf`, app `ap-a2GrhdbZsrQqxRaQpIt5BE` |
| **Q-11** | every selector, every era, re-run in this fork | **PASS (18 cells)**, exo→bisect **7→10 / 8→18 / 1→9** — the donor's own numbers, cell for cell |
| **EB-0** | `fit_signal` defaults to `"surface"`, and `own_scalar_s` − `fit_rule_s` = **{`fit_signal`}** — a literal one-bit contrast | **PASS** (`gates_cpu`, **69** checks at Q1; the suite ends the round at **78**) |
| **EB-1** | the learner's own-write count reproduces the grader's number: `(~ok & written).sum(1) == round(per_row × scored)`, row for row, every cycle | **PASS — 0 mismatches** over 60 cycles / 12,744 labelled cells (`em_sm`); asserted in-run, so a failure stops the arm |
| **EB-2** | `scored == n_blocks` on every row, so the label's renormalising constant really is the whole bottom row | **PASS — 0** |
| **EB-3** | the bag objective moves both ways on synthetic bags | **PASS** — p 0.4973 → **0.9855** at y=0; 0.4973 → **0.0143** at y=1 |
| **EB-4** | the coordinator's gate (a): the per-block branch driven by the GRADER's verdict instead of the reader's must recover K — with `w = 1{the synonym written is not the rule's}` the objective is exactly `BCE(σ(logit), K[f, ρ])`, i.e. `own_verdict`'s supervised fit | **PASS** — acc_practised **1.0000**, acc_held-out **0.8750**, det **0.375**, θ̂ **[2,2,5,5,5,5,5,2]**: Q0's `own_verdict` table to the digit |
| **EB-5** | with no per-block label the bag fit is `own_scalar`'s branch exactly; the readback route is an ADDED branch, not a changed one | **PASS** |
| **RB-1** | the CONTROLLED read-back flip (below) | **measured** — and it contradicts §Q0.3's inference |
| **B-1 (bind)** | `n_unbound = 0` in every cycle of every arm | **PASS** |
| GG-*, R-*, SR-*, PE-*, P-*, L-*, Q-*, C-*, T-*, I-*, D-*, X-* | the donors' suites | as donors |

## Runs — the complete record (every tag, its app, its GPU time)

Every GPU run of this node, in launch order. GPU time is the substrate's own `DONE in …`
line where the entrypoint prints one; the G-F gate runs and the reader-leg entrypoints do
not, so their cost is given as measured seconds where the log records them and as *gate
run* where it does not (each is a few minutes of L4). Profile `chromatic`, one L4 per run,
`modal run --detach`; every launch log is under `results/launch_<tag>.log`.

| tag | app | GPU | what | outcome |
|---|---|---|---|---|
| — | — | — | `gates_cpu` — the offline suite, CPU, no substrate | **ALL PASS (78 checks)**, ~3 min. **Needs torch**: the EB-3…EB-15 block sits behind `try: import torch / except ImportError`, so an interpreter without torch prints **68** and silently skips the ten head/objective/simulation checks. Run it with the venv that has torch, and read the count |
| `_preflight` | `ap-HSl6OV1L0eGQePTeZtXiS3` | gate run | interface, `--arms "canon_s,fit_rule_s,own_scalar_s" --rule E_R8 --n-ctx 8 --practiced 0,3,7 --setup-render rule` — every new code path at toy sizes | **PASS**. `own_scalar_s` ran all eras with 0 EB-1/EB-2 failures and `n_unbound = 0`. RB-1's number there is **not readable** — 40 reader steps, `read_acc` 0.816 — which is why the gate carries a `base_read_wrong` column |
| `em_gf` | `ap-a2GrhdbZsrQqxRaQpIt5BE` | gate run | **G-F** + Q-11, the fork itself | **PASS — 0.000e+00**, control 0.000e+00, commits equal |
| `em_sm` | `ap-QeWo9OIOf4f4CmobzGPrkF` | **1913 s = 0.53 GPU-h** | the `--quick` smoke, 3 arms, 20-cycle eras, `--reader-steps 6000`, `rule_seed 6`, `--setup-render rule` | clean. Machinery record below; nothing at this scale is a result |
| `em_gf2` | `ap-R6HeJeNHQ8DznxpEHbxdcp` | gate run | G-F after `cell_tally`, the `lexicon_merge` hook and `own_intent` staged | **PASS — 0.000e+00** |
| `em_sm2` | `ap-YSWG5mtkuqYKl41EqNr0kZ` | **967 s = 0.27 GPU-h** | the short readback smoke, `--quick --quick-cycles 14`, `--arms "canon_s,own_readback_s"` | clean |
| `em_gf3` | `ap-WXKndHGy63XpnzZ58GdGOQ` | gate run | G-F after the write record (`own_intent=record`) and `q2_reader_sizing` | **PASS — 0.000e+00** |
| `em_gf4` | `ap-LM6TSVzvzPggoay2sCqF2E` | gate run | G-F after the EB-8 write-time fix — **the `--gf-tag` of the Q1 reduction** | **PASS — 0.000e+00** |
| `em_q2r` | `ap-0ywOhmdXysWDlMMS0yTLHg` | 629 s + 628 s = **0.35 GPU-h** | Q2.0's reader leg, round 1: `q2_reader_sizing` at H=2 and H=4 | clean |
| `em_q2p` | `ap-GWlTNsJbmgftwMBkhUZDMW` | no total printed; **≈ 0.5 GPU-h** (three rungs at `em_q2r`'s ≈ 629 s each) | Q2.0's reader leg, round 2: H=2,3,4, reader A (the substrate's) against reader B (generation-time labels), returned feature logged | clean; gate Q2R-1 in tag |
| `em_q1` | `ap-Yh4MSHcGSUJnFeBBxUYcD5` | **7346 s = 2.04 GPU-h** | **THE Q1 LADDER** — `canon_s`, `fit_rule_s`, `own_scalar_s`; `if_q1c_yk`'s design exactly, 140 cycles | clean |
| `em_q1b` | `ap-24ZxDE7Ni2LLcPbGyryVmf` | **5283 s = 1.47 GPU-h** | **THE RETRACTION LADDER** — `canon_s`, `own_readback_s`, same design, second L4 | clean |
| `em_gf5` | `ap-RRYpj8QDunzWWPzwHAzdJS` | gate run | G-F after the objective branch (`own_scalar_pg`) | **PASS — 0.000e+00** |
| `em_q1c` | `ap-tavcIgRjQ0WRMkkfj7NNKp` | **5220 s = 1.45 GPU-h** | **THE OBJECTIVE AXIS** — `canon_s`, `own_scalar_pg_s` | clean |
| `em_gf6` | `ap-ip08YfqwnutmPIc41eNeFP` | gate run | G-F after the keying branch (`own_intent` per-arm) | **PASS — 0.000e+00** |
| `em_q1d` (1) | `ap-CyMeb1Rdv88QDfVVABRUTF` | — | **THE KEYING AXIS**, first launch | **CRASHED** at `canon_s` c20 — the shadowed `_own` (below) |
| `em_gf7` | `ap-nrSwbGr4p0MqFLJM8ZNuf4` | gate run | G-F after the `_is_own` rename and the EB-8 split | **PASS — 0.000e+00** |
| `em_q1d` (2) | `ap-com82TIrVLqnhUjz83kg44` | — | the same, relaunched | **CRASHED** at `own_scalar_s` c73 — the batch-shape argmax tie (below) |
| `em_gf8` | `ap-QToZaR74Aqq4FHTkUWBKHc` | gate run | G-F after EB-8 became a coverage gate — **the `--gf-tag` of the four-tag reduction** | **PASS — 0.000e+00** |
| `em_q1d` (3) | `ap-x07UJZGwIlIbkCMVfacPZH` | **7566 s = 2.10 GPU-h** | **THE KEYING AXIS** — `canon_s`, `own_scalar_s`, `own_scalar_rec_s` | clean |
| `em_gf9` | `ap-Lembd16aJXkybDfb0ot7aB` | gate run | G-F after EB-13/EB-14 and the Q2.1 launch machinery (`_lexicon_record`) — **the `--gf-tag` of the Q2.1 reduction** | **PASS — 0.000e+00** |
| `em_q2` (1) | — | — | **Q2.1**, first launch | **CRASHED** at `own_readback_s` c20 — EB-6 on unavailable blocks (below) |
| `em_q2` (2) | `ap-NbITHQ0asNS4xV84M1zZcr` | **12524 s = 3.48 GPU-h** | **Q2.1 — THE HOMOPHONOUS LEXICON at H=3**, five arms, 140 cycles, reader A, `--own-intent record` | clean |
| `em_gf10` | `ap-b49evzpT1NRPvVikRZPQiL` | gate run | G-F after the Q2.2 listener change (`reader_target`) | **PASS — 0.000e+00** |
| `_preflight` (Q2.2) | `ap-0EgtrzUzObTX27eLebwEuH` | gate run | five arms, reader B, the lexicon and the record key at toy sizes | **PASS**; reader B differs from `bottom_map` on 0.2816 of 64,000 corpus blocks |
| `em_gf11` | `ap-iSQ0pGRtQxNmvtjQ35RFRz` | gate run | G-F after the chooser change (`_train_generator_feats`) | **PASS — 0.000e+00** |
| `_preflight` `listener` / `both` | `ap-zZuLj0SfQqeRTmFZIR4Fnu` / `ap-L5l10xbhf1pPBkwvGi4uy4` | gate runs | the renamed knob and the new generator trainer, end to end | **PASS** both; EB-1/EB-2/EB-6 0, record mismatch 0 |
| `em_smb_l` | `ap-5ANu6oDTfHqSNUvc4x4ECz` | **2535 s = 0.70 GPU-h** | the `listener` smoke at PRODUCTION reader steps, five arms | clean; `read_acc` 0.9952 |
| `em_smb` | `ap-m5IV04dXAeeqKuo2TJw9rw` | **3422 s = 0.95 GPU-h** | the `both` smoke at production reader steps | clean; `read_acc` 0.9952 |
| `em_gf12` | `ap-93vDfqEkxgeDQ5iaVGhdnh` | gate run | G-F after RB-1's derived-truth pass — **the `--gf-tag` of the Q2.2 reduction** | **PASS — 0.000e+00** |
| **`em_q2b_l`** | `ap-uDj1FSpldx8bn1cL7gdPlt` | **11674 s = 3.24 GPU-h** | **Q2.2, THE LISTENER ONLY** — reader B as the ear on every `shared["reader"]` path | clean |
| **`em_q2b`** | `ap-3cKaCQ9E1L3N3sDkJWzPoB` | **11937 s = 3.32 GPU-h** | **Q2.2, BOTH EARS** — the same, plus the generator's setup supervision | clean |

**GPU measured for the node: 19.90 GPU-h** (11.69 through Q2.1, plus Q2.2's 0.70 + 0.95
smokes and 3.24 + 3.32 tags), on the same accounting as below.

**GPU measured through Q2.1: 11.69 GPU-h.** 11.34 from the seven runs that print a
`DONE in …` line — `em_sm` 0.53, `em_sm2` 0.27, `em_q1` 2.04, `em_q1b` 1.47, `em_q1c` 1.45,
`em_q1d` 2.10, `em_q2` 3.48 — plus `em_q2r`'s 0.35, summed from its own per-rung seconds.
Not in that figure and not measured: `em_q2p` (≈ 0.5 by the same per-rung cost), the nine
G-F gate runs, the three preflights, and the three crashed launches, which died at
`canon_s` c20, `own_scalar_s` c73 and `own_readback_s` c20 respectively and so burned real
L4 time that no line records.

**Two record corrections.** (a) An earlier version of this table gave `em_sm2`'s app as
`ap-R6HeJeNHQ8DznxpEHbxdcp`; that is **`em_gf2`**'s app. `em_sm2` ran as
`ap-YSWG5mtkuqYKl41EqNr0kZ` (`results/launch_em_sm2.log`, `DONE in 967s`). (b) `em_gf9`'s
artifact was taken at the EB-13/EB-14 commit and **re-used**, not re-run, after the EB-6
availability fix. That is sound and not a shortcut: EB-6 lives inside the own-arm branch
(`own_readback_rows`' call site, `embouchure.py` ~line 7027), which the knobs-off replay
never enters, so the fix cannot move a G-F number. It is recorded because the artifact's
timestamp is earlier than the fix's.

A tenth app, `ap-O5Q5uH6Hf7Q5G0MN4pSkt8`, appears in `modal app list`: that is `pack_tag`
fetching `em_q2` off the volume, not a run.

### `em_sm` — the machinery record (facts; the interpretation is the orchestrator's)

`read_acc` **1.000**; `setup_render_check.buffer_on_rule` per the knob.

**The twin gate** (`log["ans_hash"]`, crc32 of the practice answer, cross-arm):

| pair | first divergence | cycles identical |
|---|---|---|
| `own_scalar_s` vs `canon_s` | **c2** | 1 / 60 |
| `fit_rule_s` vs `canon_s` | **c5** | 4 / 60 |

Both fit from c1 (`fit_warmup=0`); `fit_rule_s`'s c1 table is all-canon (`θ̂` = [8]×8, i.e. it
never spells synonym 1), so its writes stay `canon`'s for four more cycles. This is the
`_sp`-pair idiom of `inflection` Q3, on a different pair, and it is what makes "`own_scalar`
renders at `canon` until its first fit" a measured cycle.

**The bag, counted rather than estimated.** Q0 could only infer the own-write count from
`e_sp_practice / spell_err` and got **0.2812 of 32 = 9.00 blocks/instance**
(range [0.2512, 0.3257]). The own arm now counts it:

| | measured (`log["own"]`) |
|---|---|
| own written blocks per instance | **8.92** |
| labelled cells per cycle | 212.4 |
| instances with 0 own writes | 0.00 |
| **read-back drop on own writes** (`written & agood` vs `written`) | **0.0104** |

**The arms at smoke scale** (60 cycles, 20-cycle eras — machinery, not results):

| arm | `fit_signal` | lifetime written spelling error | `e_sp_practice` | acc_prac | acc_held | θ̂ at end |
|---|---|---|---|---|---|---|
| `canon_s` | surface (no head) | 0.3977 | 0.1128 | — | — | — |
| `fit_rule_s` | surface | 0.0386 | 0.0096 | 1.000 | **0.875** | [2,2,5,5,5,5,5,2] |
| `own_scalar_s` | own_scalar | 0.0163 | 0.0047 | 1.000 | **0.850** | [2,2,6,6,5,6,5,2] |

### GATE RB-1 — the CONTROLLED read-back flip (`em_sm`, production reader)

Take a clean rule-spelled configuration at register ρ, flip ONE block to the other synonym of
its own feature, re-read with the frozen reader. `read_acc` 1.000, and the reader is wrong on
the UNFLIPPED block on 0.0039 of rows, so the baseline is clean.

| | pooled | ρ=0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|---|
| n | 512 | 64 | 64 | 64 | 64 | 64 | 64 | 64 | 64 |
| base read wrong (unflipped) | 0.0039 | 0.000 | 0.000 | 0.000 | 0.000 | 0.016 | 0.016 | 0.000 | 0.000 |
| **flipped block's read CHANGES** | **0.5625** | 0.719 | 0.609 | 0.766 | 0.609 | 0.703 | 0.313 | 0.344 | 0.438 |
| **flipped block's read is WRONG** | **0.5586** | 0.719 | 0.609 | 0.766 | 0.609 | 0.688 | 0.297 | 0.344 | 0.438 |
| changed, given the baseline was right | 0.5608 | 0.719 | 0.609 | 0.766 | 0.609 | 0.698 | 0.302 | 0.344 | 0.438 |
| collateral (untouched blocks) | 0.0029 | 0.000 | 0.000 | 0.000 | 0.000 | 0.009 | 0.014 | 0.000 | 0.000 |

Practised registers are {0, 3, 7} = 0.719 / 0.609 / 0.438; held out {1, 2, 4, 5, 6} = 0.609 /
0.766 / 0.703 / 0.313 / 0.344. **This contradicts §Q0.3's inference** — see
[`DESIGN.md`](DESIGN.md) §2, where the withdrawn claim and the arithmetic that produced it are
kept in full.

### `own_readback_s` — the retraction route (added 2026-09-11)

Built because gate RB-1 measured that a misspelling changes the frozen reader's read-back
**0.5625** of the time on this world, against the observational inference from Q0 that said it
never does ([`DESIGN.md`](DESIGN.md) §2 carries the retraction in full).

| piece | what it is |
|---|---|
| the cells | `own_recall_table` — the learner's RECALL of its own write: it emitted `bottom[f, k]`, so the form it left identifies (f, k) wherever the lexicon is injective. `n_tab > 1` marks a homophone, where recall is genuinely ambiguous; those cells are excluded from the fit and counted (`n_ambiguous`) rather than tie-broken |
| the label | `w = 1{read-back feature ≠ intended feature}`, from one no-grad reader forward on the completed answer |
| the loss | `own_scalar`'s exactly, with `w` per BLOCK in place of `y` per bag: same `q = p` / `1 − p`, same direction, same 64 steps, same batch in bags. `w` absent ⇒ the function is `own_scalar`'s (gate EB-5) |
| the ledger | `blk_readback` (blocks the reader passed over) and `blk_readback_used` (blocks whose verdict entered the fit), measured and **never folded into `t`**; `priced: False` is logged with them. Both own arms pay the same forward; only one uses it as a label |
| the confusion | `log["own"]["confusion_n"] / ["confusion_mis"]`, an 8×8 (feature, register) tally per cycle — the coordinator's gate (b) |

**The one asymmetry against `own_scalar`, stated rather than smoothed over**: `own_scalar` keys
its cells on the READER's parse of its own answer, `own_readback` on RECALL. It has to: a label
that asks "did the reader give back what I meant" is undefined if the cell's feature is itself
what the reader said. Where the two sources differ IS the label, so this is the signal and not
a confound; the disagreement rate is logged as `cells_reader_agrees`, and Q0 measured it at
0.0104 on this world.

## Reproduce

```bash
cd experiments/            # MODAL_PROFILE=chromatic; CPU steps need no profile
# NOTE: the local `modal` runs on /usr/bin/python3, which needs numpy installed
#       (`uv pip install --python /usr/bin/python3 --system numpy`); the scratchpad venv's
#       `modal` cannot reach the Modal server at all (gRPC).

PYTHONPATH=. python3 rhm/practice/embouchure/phase0_embouchure.py \
    > rhm/practice/embouchure/figures/q0_reduction.txt                     # Q0, ~4 min
PYTHONPATH=. python3 -c "from rhm.practice.embouchure import embouchure as E; E.gates_cpu()"
#   -> ALL PASS (78 checks), ~3 min. Use an interpreter WITH torch: without it the
#      EB-3..EB-15 block is skipped by its `except ImportError` and the count reads 68.

modal run rhm/practice/embouchure/embouchure.py::preflight \
    --arms "canon_s,fit_rule_s,own_scalar_s" --rule E_R8 --n-ctx 8 \
    --practiced "0,3,7" --transfer-n 8 --setup-render rule

PYTHONPATH=. python3 rhm/practice/embouchure/phase0_q2_lexicon.py \\
    > rhm/practice/embouchure/figures/q2_lexicon_reduction.txt                 # Q2.0, ~2 min

python3 rhm/practice/embouchure/launch_detached.py --fn fidelity_smoke --tag em_gf

# THE Q1 LADDER (single seed) — `if_q1c_yk`'s design exactly; only `--arms` moves
python3 rhm/practice/embouchure/launch_detached.py --fn embouchure_run --tag em_q1 \
    --arms "canon_s,fit_rule_s,own_scalar_s" \
    --rule E_R8 --n-ctx 8 --practiced "0,3,7" --transfer-n 64 \
    --rule-seed 6 --setup-render rule --reader-steps 6000 \
    --eras "1:25:40,2:12:45,3:6:55" --era-caps "40,45,55" \
    --max-macro-level 3 --budget 8 --g-budget 482 --question-k 2048 \
    --tol-dsil 0.0046 --endo-price 267 --gate-frac 0.5 \
    --span-tau-fire 0.50 --perf-alpha 0.05 --perf-tau-w 0.20 \
    --n-corrupt 1 --mine-cap 8 --gy-level 4 --entry-rec \
    --span-min-hold 128 --collect-task-matched --tm-episodes 8192 \
    --recert-every 5 --n-aud 192 --probe-every 8 --n-rt 384 --n-score 256 \
    --sil-cv 0.15 --lp-min-drop 0.10 --seed 0
# the smoke is the same line with --tag em_sm --quick --quick-cycles 20 --transfer-n 16
#   --eras "1:25,2:12,3:6" --era-caps "20,20,20" --question-k 512 --tm-episodes 2048
#   --n-aud 64 --n-rt 128 --n-score 128

# em_q1b is the SAME line with --tag em_q1b --arms "canon_s,own_readback_s"
# em_q1c is the SAME line with --tag em_q1c --arms "canon_s,own_scalar_pg_s"   (the OBJECTIVE axis)
# em_q1d is the SAME line with --tag em_q1d \\
#     --arms "canon_s,own_scalar_s,own_scalar_rec_s"                           (the KEYING axis)

# THE Q2.1 LADDER — the same line again, plus the lexicon and the record key
python3 rhm/practice/embouchure/launch_detached.py --fn embouchure_run --tag em_q2 \\
    --arms "canon_s,given_rule_s,fit_rule_s,own_scalar_rec_s,own_readback_s" \\
    --lexicon-merge "0:1=1:1,2:0=4:0,5:0=6:0" --own-intent record \\
    <... every other flag exactly as the Q1 ladder above ...>

# the reductions (each asserts G-F 0.000e+00 in tag and canon_s bit-identity across tags)
python3 rhm/practice/embouchure/analyze_embouchure.py --tag em_q1 --gf-tag em_gf4 \\
    --merge-tag em_q1b --fetch                          # Q1, two tags as one design
python3 rhm/practice/embouchure/analyze_embouchure.py --tag em_q1 --gf-tag em_gf8 \\
    --merge-tag em_q1b,em_q1c,em_q1d --fetch            # the four-tag joint reduction
python3 rhm/practice/embouchure/analyze_embouchure.py --tag em_q2 --gf-tag em_gf9 --fetch
#   --fetch on a ~13 MB results.json corrupts mid-stream through `modal volume get`;
#   pack_tag (tar.gz on the volume) is the working path — see the `pack_tag` note below.
```

## Open at halt

All planned runs are complete. `em_q2` (Q2.1) is the last of them; the round's next step is
discussion, not a launch. Nothing is in flight; no waiter is armed.

---

# Q2.0 — how much homophony the world tolerates (2026-09-11, CPU, offline)

`phase0_q2_lexicon.py`, ~2 min, no Modal, no substrate. The SPEC's Q2 (revised) replaces
`bottom` ALONE with a constructed lexicon carrying H shared forms — two (feature, synonym)
pairs on one leaf tuple — every level above untouched (`true_tables` reads `rules[0…L−2]`, so
every flat key at L2–L5 is the same object). World: `rule_seed 6`, `E_R8`,
θ = [1,1,6,4,5,5,5,1], practised {0,3,7}, 512-instance pools.

## Q2.0(1) — which merges are ambiguous UNDER THE RULE

A merge is legal only between DIFFERENT features (`generate_rules_distinct` guarantees a
feature's m tuples are distinct and `k_of`, the grader and the miner all rely on it): **112
legal merges** of the 16 (feature, synonym) pairs. A merge is *ambiguous under the rule* at
register ρ iff the rule admits both owners there. For the ordered-register family that count is
closed form — `min(θ1,θ2)` for two synonym-0 words, `8 − max(θ1,θ2)` for two synonym-1 words,
`max(0, θ1−θ2)` for (0,1) — and **the closed form agrees with the enumeration on all 112**.

**34 of the 112 are never ambiguous under the rule**: they change the surface and nothing else,
and are not what Q2 asks for. The richest by ambiguous *practised* registers:

| merge | θ | n_amb | practised | held out | registers |
|---|---|---|---|---|---|
| (f0,k1) = (f1,k1) | [1,1] | 7 | 2 | 5 | 1–7 |
| (f0,k1) = (f7,k1) | [1,1] | 7 | 2 | 5 | 1–7 |
| (f1,k1) = (f7,k1) | [1,1] | 7 | 2 | 5 | 1–7 |
| (f2,k0) = (f4,k0) | [6,5] | 5 | 2 | 3 | 0–4 |
| (f4,k0) = (f5,k0) | [5,5] | 5 | 2 | 3 | 0–4 |
| (f2,k0) = (f3,k0) | [6,4] | 4 | 2 | 2 | 0–3 |

No merge is ambiguous at all three practised registers: {0,3,7} spans the scale, and a
threshold pair can bracket at most two of them.

## Q2.0(2) — the ladder

Disjoint greedy order (each form owned by exactly two meanings, so H merges leave 16−H forms
and no chain of three): (f0,k1)=(f1,k1) · (f2,k0)=(f4,k0) · (f5,k0)=(f6,k0) ·
(f3,k1)=(f7,k1) · (f4,k1)=(f5,k1) · (f2,k1)=(f6,k1) · (f0,k0)=(f1,k0) · (f3,k0)=(f7,k0).
**Eight merges are available**, and the eighth exhausts the lexicon.

| H | forms | root reachable | root set | root amb | L1 amb (all ρ) | **L1 amb (practised ρ)** | on-grammar | **recallable** | d\* L1 | L2 | L3 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 16 | 1.0000 | 1.3945 | 0.3398 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.336 | 2.342 | 3.365 |
| 1 | 15 | 1.0000 | 1.3945 | 0.3398 | 0.1783 | **0.1293** | 1.0000 | **0.8772** | 1.303 | 2.258 | 3.182 |
| 2 | 14 | 1.0000 | 1.3945 | 0.3398 | 0.3069 | **0.2552** | 1.0000 | **0.7692** | 1.303 | 2.141 | 3.135 |
| 3 | 13 | 1.0000 | 1.4180 | 0.3477 | 0.5420 | 0.5086 | 1.0000 | 0.6383 | 1.264 | 1.904 | 2.762 |
| 4 | 12 | 1.0000 | 1.4180 | 0.3477 | 0.7060 | 0.6160 | 1.0000 | 0.5116 | 1.230 | 1.842 | 2.611 |
| 5 | 11 | 1.0000 | 1.4180 | 0.3477 | 0.8154 | 0.7136 | 1.0000 | 0.4000 | 1.258 | 1.857 | 2.592 |
| 6 | 10 | 1.0000 | 1.4180 | 0.3477 | 0.9000 | 0.8051 | 1.0000 | 0.3158 | 1.281 | 1.865 | 2.574 |
| 7 | 9 | 1.0000 | 1.4180 | 0.3477 | 0.9277 | 0.8824 | 1.0000 | 0.2703 | 1.312 | 1.883 | 2.576 |
| 8 | 8 | 1.0000 | 1.4180 | 0.3477 | 1.0000 | 1.0000 | 1.0000 | 0.2222 | 1.246 | 1.795 | 2.463 |

**Every rung stays a world.** `root_reachable` = 1.0000 and `on_grammar` = 1.0000 at every H,
including H=8 where every one of the eight surviving forms is shared: the true root is always
in the root possible-set and every block is a legal tuple. The root set barely moves
(1.3945 → 1.4180, root-ambiguous 0.3398 → 0.3477) — the bottom ambiguity is absorbed by the
grammar above it, which is `SIZING.md` premise (3) again from the other side.

**The confound, stated: the ladder is not difficulty-matched.** `d*` FALLS with H
(L3: 3.365 → 2.463, −27 %), because more spellings are acceptable so the nearest derivation is
nearer. A cross-rung comparison confounds homophony with repair difficulty; within a rung every
arm shares one world and the comparison is clean, which is the same discipline `inflection` Q2
recorded for its curriculum rungs.

## Q2.0(3) — last-writer-wins

What `build_inverse_maps` keeps, i.e. which meaning the arc's exact parse silently picks (and
therefore what every instrument routed through `inverse_maps[-1]` reports):

| code | owners | keeps |
|---|---|---|
| 12 | (f0,k1) (f1,k1) | f=1 |
| 33 | (f2,k0) (f4,k0) | f=4 |
| 28 | (f5,k0) (f6,k0) | f=6 |
| 19 | (f3,k1) (f7,k1) | f=7 |
| 5 | (f4,k1) (f5,k1) | f=5 |
| 40 | (f2,k1) (f6,k1) | f=6 |
| 26 | (f0,k0) (f1,k0) | f=1 |
| 41 | (f3,k0) (f7,k0) | f=7 |

In every case the higher-numbered feature wins — `build_inverse_maps` writes in feature order,
so last-writer is largest-index. Not a fact about meaning; a fact about the loop.

## Q2.0(4) — the writer's side, and the asymmetry the SPEC hoped for

The SPEC asks for a lexicon where "the perceptual route sees one form with two intents **while
the learner's own writes are unambiguous to itself**". Measured: **that asymmetry does not
exist in the lexicon.** "Both owners admissible at ρ" is one condition, and it is exactly the
condition under which neither the observer nor a writer reconstructing from (form, register)
can resolve the block. Recallable (form, register) cells fall 1.0000 → 0.2222 across the
ladder, in lockstep with the observer's ambiguity.

At H=8, of 36 live (form, register) cells 8 are recallable; by register:

| ρ | practised | recallable | ambiguous |
|---|---|---|---|
| 0 | yes | 0 | 4 |
| 1 | | 2 | 3 |
| 2 | | 2 | 3 |
| 3 | yes | 2 | 3 |
| 4 | | 0 | 4 |
| 5 | | 2 | 3 |
| 6 | | 0 | 4 |
| 7 | yes | 0 | 4 |

**The consequence for Q2.1's machinery, which is the actionable part**: the asymmetry has to
come from the learner RECORDING its own command, not from the lexicon. `own_readback` as built
for Q1 reconstructs its intent with `own_recall_table` — correct on an injective lexicon, and
on a homophonous one it would silently lose 12–78 % of its cells to `n_ambiguous`. For Q2.1 the
executor must carry the feature it wrote forward from the beam (the Renderer-side record, which
Q1 did not need), and that is a real addition rather than a flag.

## Q2.0 — what is not measured here, and why

The third leg the SPEC asks for — **what the frozen neural reader does at the homophonous cells
at production `reader_steps`** — is a measurement on a trained reader, i.e. one `build_shared`
(~610 s of L4). The coordinator's instruction is no GPU for Q2 until this ladder and the two Q1
reductions are read, so it is **not run**. The machinery is in place (`readback_flip_check`'s
idiom, plus `own_recall_table` for the ambiguous cells); the call is one line in `preflight`,
and RB-1 already shows the reader is register-sensitive enough that the answer will not be a
formality.

**No rung is rejected on the CPU evidence**: every H from 1 to 8 keeps the world solvable and
on-grammar. What picks a rung is the reader half plus how many of its own cells `own_readback`
can still recall — on the CPU numbers alone, **H = 2** (14 forms, L1 ambiguity 0.2552 at the
practised registers, recall 0.7692) is the densest rung that leaves the writer's route mostly
intact, and **H = 4** (0.6160 / 0.5116) is the densest at which it is still half intact.

---

# The efference copy, and what gate EB-8 caught (2026-09-11)

`own_intent` — where `own_readback`'s INTENT comes from. `"recall"` (default, and what `em_q1b`
ran) decodes it from the form via `own_recall_table`; `"record"` carries it from the write.

| gate | what it asserts | status |
|---|---|---|
| **EB-6** | the write record and the form decode agree block for block wherever the form identifies the meaning | **PASS** (preflight, both lexicons; and on CPU, 0 of 88 blocks disagree) |
| **EB-7** | the record's block mask equals the move geometry's (`own_write_mask`) | **PASS** |
| **EB-8** | the open-loop replay through the arm's own executor reproduces `out["x"]` | **FAILED, then PASS** — see below |
| **G-F** | re-run after the lexicon hook and `cell_tally` (`em_gf3`, app `ap-WXKndHGy63XpnzZ58GdGOQ`) | **PASS — 0.000e+00**, control 0.000e+00, commits equal |

## EB-8 failed first, and the failure is the design note

The record was first taken in block (c''), beside the rest of the renderer's harvest. It failed
at **c7** — the first cycle of era 2, the first with a macro for the span head to fire on:

```
GATE EB-8 FAILED at c7: the open-loop replay through the executor does not reproduce out['x']
```

**Why.** Block (b) trains the plant and the span head, and it runs BEFORE the harvest. Replaying
the accepted move sequence down in (c'') therefore goes through nets that have moved since the
beam produced the answer, so the features the executor re-derives are not the ones it wrote.

**The fix is the concept.** The record is now taken immediately after the practice beam, beside
`grade_spelled` and before any training — which is what an efference copy *is*: a copy of the
command at the moment of the command. Had EB-8 not been written, the arm would have trained on
a plausible-looking reconstruction of its own intent that was quietly the wrong one.

## What `record` buys, measured

On a homophonous lexicon (`0:1=1:1,2:0=4:0`), 128 written blocks, a perfect reader:

| `own_intent` | usable cells | ambiguous | EB-6 |
|---|---|---|---|
| `recall` (decode from the form) | 88 / 128 | **40** | — |
| `record` (the efference copy) | **128 / 128** | **0** | 0 disagreements of 88 checked |

The decode loses 31 % of the learner's own writes to homophony; the record loses none, and
agrees with the decode everywhere the decode is defined. On `rule_seed 6` (injective) the two
are identical, which is why `em_q1b` is unaffected by the default staying `"recall"`.

**The replay is invisible by construction, not by inspection**: `_ENTRY_REC["phase"]` is set to
`"probe"` for its duration so a fired macro takes `_fire_only` (the same `head.emit` on the same
pooled trunk output as the metered path, so the write is identical) and `meter.score` is never
called; `ex.capture` is off so the span buffer and its held-out split do not move; `ex.counts`,
`ex.last` and the renderer's whole tally are snapshotted and restored; and nothing in it draws
RNG.

---

# Q2.0, the reader leg — `em_q2r` (2026-09-11, app `ap-0ywOhmdXysWDlMMS0yTLHg`)

`q2_reader_sizing`, two rungs, 629 s + 628 s = **0.35 GPU-h**. One `build_shared` per rung on the
constructed lexicon, production `reader_steps` 6000, then a clean rule-spelled draw of 256
instances per register **kept with the level-1 features the world derived** (on a homophonous
lexicon the surface no longer identifies them, so they are captured at generation rather than
parsed back). 8,192 blocks per register. Nothing graded, no arm, no ladder.

| rung | merge | forms | shared | `read_acc` (setup probe) | **away** | homophone | **rule resolves** | rule ambiguous |
|---|---|---|---|---|---|---|---|---|
| H=2 | `0:1=1:1, 2:0=4:0` | 14 | 2 | 1.0000 | **0.9992** | 0.5688 | **0.0000** (n=703) | 0.5894 |
| H=4 | + `5:0=6:0, 3:1=7:1` | 12 | 4 | 1.0000 | **1.0000** | 0.5581 | **0.8118** (n=3,736) | 0.5357 |

**The baseline is not cost.** Away from the shared forms the reader is at 0.9992 / 1.0000, at
every register, practised and held out alike — one cell (H=2, ρ=4) at 0.9929 and the rest exact.
Neither rung damages the organ where the lexicon is untouched.

**Where the rule leaves both owners admissible the reader is at chance, as it must be.** 0.4590
–0.6503 across cells, against 0.5000 for a coin between two owners. This is the control and it
behaves: the ambiguity there is the lexicon's, not the reader's, and no reader can beat it.

**Where the rule uniquely resolves the owner, the reader is bimodal — 0.0000 or 1.0000, never
in between**, and which one is predicted cell for cell by the last-writer-wins table:

| rung | ρ | resolving code | owners | rule's unique owner | last-writer keeps | n | reader |
|---|---|---|---|---|---|---|---|
| H=2 | 5 | 33 | (f2,k0) (f4,k0) | **f2** | f4 | 703 | **0.0000** |
| H=4 | 1 | 19 | (f3,k1) (f7,k1) | **f7** | f7 | 1,011 | **1.0000** |
| H=4 | 2 | 19 | (f3,k1) (f7,k1) | **f7** | f7 | 996 | **1.0000** |
| H=4 | 3 | 19 | (f3,k1) (f7,k1) | **f7** | f7 | 1,026 | **1.0000** |
| H=4 | 5 | 33 | (f2,k0) (f4,k0) | **f2** | f4 | 703 | **0.0000** |

Four cells, four predictions, four hits: the frozen reader returns the owner the last-writer
tie-break favours and does **not** use the register to resolve a shared form, even where the
register determines it uniquely. `inflection` finding 5 listed "reading the register off the
surface" as a job the reader does not do; this is that, measured directly.

**Caveat, stated**: the reduction logs accuracy, not what the reader returned, so the
last-writer correspondence is inferred from four cells whose predictions are exact rather than
read off a confusion target. Logging the returned feature is a one-line addition to
`q2_reader_sizing` and a re-run at 0.35 GPU-h if the correspondence is to be load-bearing.

**One structural fact for the rung choice**: at H=4, register ρ=4 has `n_away = 0` — every
block at that register is a homophone, so the rung leaves one of the eight registers with no
unambiguous surface at all.

## `em_sm2` — the readback smoke's machinery record

42 cycles, 14-cycle eras, `read_acc` 1.000. Nothing at this scale is a result.

| | `own_readback_s` |
|---|---|
| EB-1 / EB-2 mismatches | **0 / 0** over 42 cycles |
| labelled cells | 9,382 (223.4 / cycle) |
| read-back ledger | 32,256 blocks read, 9,382 used as labels, `priced: false` |
| `n_ambiguous` (recall undefined) | **0** — the lexicon is injective |
| **mis-phrasing rate** (own blocks read back as another feature) | **0.02355** |
| twin gate vs `canon_s` | first divergence **c2** (1 / 42 identical) |
| recovered table at the end | acc_practised 0.667, acc_held-out 0.550, θ̂ [2,4,8,8,8,8,8,8] |

For contrast, `em_sm`'s arms at a comparable scale reached acc_practised 1.000 with
acc_held-out 0.875 (`fit_rule_s`, surface) and 0.850 (`own_scalar_s`). The readback route's
label is positive on 2.4 % of its own writes here, and `θ̂` shows it moved f0 and f1 off
`canon` and left the other six there. Whether that is the route's speed or its ceiling is
`em_q1b`'s question at 140 cycles, and is not readable from 42.

---

# Q2.0, the reader leg round 2 — `em_q2p` (2026-09-11, app `ap-GWlTNsJbmgftwMBkhUZDMW`)

`q2_reader_probe`, H ∈ {2, 3, 4}, two readers per rung, the returned feature logged. No
`build_shared` (the reader is all this needs, and the controller / generator / value / buffers
are ~80 % of that setup).

**Reader A** reproduces the substrate's own supervision — `feats = bottom_map[code]`, i.e.
`build_inverse_maps`' last-writer-wins table. **Reader B** is identical in architecture,
initialisation, corpus, steps, batch, lr and seed, trained on the level-1 features the world
**derived**. Only the label moves. Reader B is *a change to the substrate's reader* and is
labelled as one — [`DESIGN.md`](DESIGN.md) §9.

## Gate Q2R-1 — the substitution is sound, and exactly

| rung | reader | away | homophone | rule resolves | rule ambiguous |
|---|---|---|---|---|---|
| H=2 | **A** | 1.0000 | **0.5688** | **0.0000** | **0.5894** |
| | `em_q2r`'s `build_shared` reader | 0.9992 | **0.5688** | **0.0000** | **0.5894** |
| H=4 | **A** | 1.0000 | **0.5581** | **0.8118** | **0.5357** |
| | `em_q2r`'s `build_shared` reader | 1.0000 | **0.5581** | **0.8118** | **0.5357** |

Reader A reproduces the `build_shared` reader **to four decimals on every column**. **PASS**.

## The last-writer correspondence, now a target and not an inference

At H=2 the two shared forms are code 12 = (f0,k1)/(f1,k1) → last writer **f1**, and code 33 =
(f2,k0)/(f4,k0) → last writer **f4**. What reader A returned, per (intended feature, register),
over all 24 homophonous cells:

| cell | n | rule resolves? | returned |
|---|---|---|---|
| f0:ρ1…ρ7 | 612–1,113 each | no | **f1, 100 %** in all seven |
| f1:ρ1…ρ7 | 999–1,113 each | no | f1, 100 % in all seven |
| f2:ρ0…ρ4 | 695–774 each | no | **f4, 100 %** in all five |
| **f2:ρ5** | 703 | **yes** | **f4 661, f5 39, f6 3 — f2 never** |
| f4:ρ0…ρ4 | 733–825 each | no | f4, 100 % in all five |

**Reader A is a deterministic last-writer lookup at a shared form, in 24 of 24 cells.** It is
not failing to use the register; its teacher never encoded one. The single cell with any spread
is the one where the rule *does* resolve the form — and there it still never returns the right
answer. Reader B on the same cells returns the **true** feature: f0:ρ1 → f0 612/612, f2:ρ0 → f2
774/774, f4:ρ0 → f4 733/733, and so on.

## Two corrections to round 1's reading

**(i) The rule-ambiguous cells are NOT a floor. Withdrawn.** Round 1 reported them at
0.4590–0.6503 against "0.5000 for a coin between two owners" and called that a property of the
lexicon no reader could beat. Reader B reaches **0.9998 / 0.9851 / 0.9682** at H = 2 / 3 / 4 on
exactly those cells. The block's FORM is ambiguous; the SEQUENCE is not, and a reader whose
labels preserve the distinction reads it off the context. What round 1 measured was reader A's
label, not the lexicon's information.

**(ii) The three-way split is confounded with register coverage.** The reader's corpus is drawn
over the practised registers only (`_sample_pool_r(..., practiced=prac)` — {0,3,7}). At H=2 and
H=3 *every* "rule resolves" cell is at **ρ=5, an unpractised register**, so that column is a
generalisation test rather than a disambiguation test — which is why reader B scores 0.3755 /
0.5533 there while scoring ~0.99 on the ambiguous cells it was trained across. At H=4 the
resolving cells at ρ=1, 2, 3 include a practised register and **both** readers score 1.0000 on
all three. The two effects must not be read as one.

## The three rungs side by side

| rung | forms | shared | A away | A homophone | B away | B homophone | registers with no unambiguous surface |
|---|---|---|---|---|---|---|---|
| H=2 | 14 | 2 | 1.0000 | 0.5688 | 0.9987 | **0.9780** | none |
| H=3 | 13 | 3 | 1.0000 | 0.5253 | 0.9990 | **0.9765** | none |
| H=4 | 12 | 4 | 1.0000 | 0.5581 | 0.9996 | **0.9580** | **ρ=4** (`n_away` = 0) |

Reader A is at 1.0000 away from the homophones on every rung; reader B pays 0.0004–0.0013 for
carrying the harder task, concentrated at ρ=4 (0.9900 / 0.9930 / —). H=3 is the densest rung
that leaves every register some unambiguous surface.

---

# `em_q1b` — the retraction ladder (2026-09-11, app `ap-24ZxDE7Ni2LLcPbGyryVmf`)

Clean, **5283 s = 1.47 GPU-h**, 140 cycles per arm, `canon_s` + `own_readback_s`.
`if_q1c_yk`'s design exactly; only `--arms` moves. Machinery record only — the joint reduction
with `em_q1` is the reading, and it is the orchestrator's.

**Setup gates**: `read_acc` **1.000**; SR-1 `buffer_on_rule` **1.000**.

**GATE RB-1 at full scale** (n = 1,024, production reader): flipping one block to the other
synonym of its own feature changes the read **0.6455** of the time and makes it wrong 0.6387,
against a baseline error of 0.0068 on the unflipped block and 0.0096 collateral. (`em_sm2`'s
smoke-scale figure was 0.5625.)

**Production-route gates**, `own_readback_s`:

| | |
|---|---|
| EB-1 / EB-2 mismatches | **0 / 0** over 140 cycles |
| labelled cells | **79,588** |
| read-back ledger | 286,720 blocks read, 79,588 used as labels, `priced: false` |
| `n_ambiguous` | **0** (injective lexicon; `own_intent = "recall"`) |
| twin gate vs `canon_s` | first divergence **c6** (5 / 140 identical) |

**The two instruments, by era** — the reason they are never summed:

| era | off-rule (synonym ≠ the rule's) | mis-read (read-back ≠ intent) |
|---|---|---|
| 1 | **0.4796** | **0.0727** |
| 2 | **0.3917** | **0.0402** |
| 3 | **0.3958** | **0.0340** |

Lifetime written spelling error **0.3865**.

**The recovered table, by cycle**:

| cycle | n | acc practised | acc held-out | det θ | θ̂ |
|---|---|---|---|---|---|
| 1 | 64 | 0.5417 | 0.3750 | 0.000 | [8,8,8,8,8,8,8,8] |
| 10 | 640 | 0.6250 | 0.4250 | 0.000 | [6,6,8,8,8,8,8,8] |
| 40 | 2,560 | 0.6250 | 0.4500 | 0.000 | [5,6,8,8,8,8,8,8] |
| 140 | 8,960 | **0.6250** | **0.4500** | **0.000** | **[5,6,8,8,8,8,8,8]** |

θ̂ = 8 means "never spells synonym 1", i.e. `canon`. Six of eight features are still there at
c140. For the banked comparators on the same world and clock: `fit_rule_s` (surface) reached
acc_practised 1.0000 / held-out 0.8750, θ̂ [2,2,5,5,5,5,5,2], and Q0's `own_verdict` fit lands
on the same table.

---

# Q1 read as ONE DESIGN — `em_q1` + `em_q1b` (2026-09-11)

`em_q1` clean, **7346 s = 2.04 GPU-h**, app `ap-Yh4MSHcGSUJnFeBBxUYcD5`; `em_q1b` clean,
**5283 s = 1.47 GPU-h**, app `ap-24ZxDE7Ni2LLcPbGyryVmf`. Four arms, 140 cycles each, one
world, one clock. Reduction: `figures/em_q1_reduction.txt`
(`analyze_embouchure.py --tag em_q1 --gf-tag em_gf4 --merge-tag em_q1b`). Facts only.

## The cross-tag anchor — the two tags ARE one design

```
[merge] em_q1b -> em_q1: refs identical = True, stale buffer identical = True
[cross-tag] canon_s: max|em_q1 - em_q1b| = 0.000e+00 over 140 cycles
```

Every logged series of `canon_s` is bit-identical across the two L4s. Without this the
cross-tag ranks below would mean nothing; the assertion refuses the merge if it fails.

## §1E — the in-tag twin gate

| arm | first divergence from `canon_s` | cycles identical | EB-1 / EB-2 | labelled cells |
|---|---|---|---|---|
| `own_scalar_s` | **c2** | 1 / 140 | **0 / 0** | 68,892 |
| `own_readback_s` | **c6** | 5 / 140 | **0 / 0** | 79,588 |

`own_readback_s`'s read-back ledger: **286,720 blocks read, 79,588 used as labels**,
`priced: false`, **0 cells ambiguous by recall** (injective lexicon).

## §2E — the time course, the readout that decides Q2

| arm | signal | endpoint acc held-out | first reached **and held** | lifetime written spelling error | θ̂ at c140 |
|---|---|---|---|---|---|
| `fit_rule_s` (**surface**) | world's blocks, solved instances | **0.8750** | **c19** | **0.0231** | [2,2,5,5,5,5,5,2] |
| `own_scalar_s` | the meter's scalar per attempt | **0.7250** | **c12** | **0.2049** | [3,4,4,3,4,3,3,3] |
| `own_readback_s` | its own reader, per block | **0.4500** | **c25** | **0.3944** | [5,6,8,8,8,8,8,8] |
| `canon_s` | — (no organ) | — | — | **0.4592** | — |

Q0's offline `own_verdict` fit reaches 0.8750 with θ̂ [2,2,5,5,5,5,5,2] — `surface`'s exact
table. θ̂ = 8 is "never spells synonym 1", i.e. `canon`: six of `own_readback_s`'s eight features
are still there at c140. `own_scalar_s`'s θ̂ is a near-constant ≈3 across features (det θ 0.000
–0.125 all run), against the true [1,1,6,4,5,5,5,1].

## §4E — the two instruments, and where they come apart

`own_readback_s`, pooled over 79,588 own-written cells: **off-rule 0.41148, mis-read 0.04359**.
By era, mis-read 0.0727 / 0.0402 / 0.0340 against off-rule 0.4796 / 0.3917 / 0.3958.

Per (intended feature, register) — off-rule / mis-read, n in brackets, only the practised
registers carry writes:

| f | ρ=0 | ρ=3 | ρ=7 |
|---|---|---|---|
| f0 | 0.000/0.000 (3938) | **1.000/0.104** (4043) | 0.048/0.031 (3271) |
| f1 | 0.000/0.000 (2333) | **1.000/0.043** (3205) | 0.050/0.070 (2195) |
| f2 | 0.000/0.000 (2537) | 0.000/0.000 (2527) | **1.000/0.182** (2763) |
| f3 | 0.000/0.000 (341) | 0.000/0.000 (299) | **1.000/0.134** (694) |
| f4 | 0.000/0.000 (3667) | 0.000/0.000 (2843) | **1.000/0.000** (3005) |
| f5 | 0.000/0.000 (5908) | 0.000/0.001 (5140) | **1.000/0.106** (5624) |
| f6 | 0.000/0.000 (3958) | 0.000/0.000 (3670) | **1.000/0.313** (4207) |
| f7 | 0.000/0.000 (4478) | **1.000/0.000** (4913) | **1.000/0.036** (4029) |

f4:ρ7 and f7:ρ3 are the extreme cells: **the arm misspells every one of 3,005 and 4,913 blocks
and its reader returns the intended feature every time.** Summing the two instruments would
have hidden that; they are never summed.

`own_scalar_s` reports the same quantity in its own form (cells its reader-keyed parse could
not use): 0.20369 / 0.15334 / 0.14615 by era. It predates the per-cell tally.

**GATE RB-1 at full scale** (n=1,024): flipped read changed **0.6455**, wrong 0.6387, baseline
error on the unflipped block 0.0068, collateral 0.0096. It falls monotonically with register —
ρ=0 0.797, ρ=1 0.719, ρ=2 0.688, ρ=3 0.664, ρ=4 0.641, ρ=5 0.555, ρ=6 0.539.

## §3E — what each channel ate

| era | own blocks / instance (readback · scalar) | own cells / cycle (readback · scalar) | surface harvest / cycle |
|---|---|---|---|
| 1 | 7.33 · 7.34 | 468.9 · 377.0 | **827** |
| 2 | 9.03 · 9.20 | 577.7 · 500.4 | **369** |
| 3 | 9.90 · 10.41 | 633.4 · 569.0 | **484** |

Q0 estimated 9.00 own blocks per instance from `e_sp_practice / spell_err`; the counted value
is 7.33–10.41. Past era 1 both own routes eat **more** labelled blocks per cycle than the
solve-gated surface harvest, which is what Q0 predicted from `n_solved` falling.

## The meaning currency (§3I), against the banked two-cluster read

| arm | `e` (end) | `e_sp` (end) | `e_sp` written, lifetime | `blk_render` |
|---|---|---|---|---|
| `fit_rule_s` | **0.6068** | 0.0000 | 0.0231 | 7,175,200 |
| `own_readback_s` | **0.6797** | 0.0532 | 0.3944 | 7,120,696 |
| `canon_s` | **0.6979** | 0.0562 | 0.4592 | 0 |
| `own_scalar_s` | **0.7474** | 0.0186 | 0.2049 | 7,223,762 |

The banked `if_q1c_yk` on this world and clock split into correct spellers at e ≈ 0.607–0.617
and misspellers at e ≈ 0.698–0.701, a gap of 4.65 × its in-tag null-ABBA floor on `e`
(v_tol 0.0173). **`fit_rule_s` 0.6068 and `canon_s` 0.6979 reproduce both clusters.** Against
that floor: `own_readback_s` sits 4.2 floors above `fit_rule_s` and ~1 floor below `canon_s`;
**`own_scalar_s` at 0.7474 is 2.9 floors ABOVE `canon_s`**, i.e. outside both banked clusters,
while spelling less than half as badly (0.2049 against 0.4592). *(The 0.0173 floor is the donor
tag's, measured by null-ABBA on `e`; this reduction's in-tag table covers the gauges, and the
`e` floor was not re-derived here.)*

## F2 (§5I) — trust against habit, with two new arms

| arm | π~solve | π~use | π~solve\|use | π~spell-ok | mean π | mean use |
|---|---|---|---|---|---|---|
| `canon_s` | 0.136 | **0.750** | 0.140 | −0.259 | 0.0106 | 6.60 |
| `fit_rule_s` | 0.241 | **0.718** | 0.169 | nan | 0.0130 | 7.50 |
| `own_readback_s` | 0.162 | **0.786** | 0.151 | −0.270 | 0.0112 | 6.85 |
| `own_scalar_s` | 0.238 | **0.389** | **0.228** | 0.085 | 0.0100 | 7.78 |

`inflection`'s finding 6 (π's mass is use count, 0.73–0.78 in every arm) reproduces in three of
the four. `own_scalar_s` is the exception on both columns.

## Gates in tag

G-F `em_gf4` **0.000e+00** (control 0.000e+00, commits equal) · `read_acc` 1.000 · SR-1
`buffer_on_rule` 1.000 · `n_unbound` 0 in all four arms · EB-1 / EB-2 0 over 280 arm-cycles ·
cross-tag `canon_s` 0.000e+00 · all four arms 140 cycles, schedule-paced, 0 loop actions.

---

# `own_scalar_s`'s F2 exception — read from the logs only (2026-09-11, no GPU)

The question: `own_scalar_s`'s π~use is **0.389** against 0.718–0.786 in the other three arms.
The three candidates, each answered from `log["f2"]`, `log["n_solved"]` and the probe rows:

| | canon_s | fit_rule_s | own_scalar_s | own_readback_s |
|---|---|---|---|---|
| lifetime solves | 34,326 | 39,825 | **21,608** | 38,331 |
| solve rate / cycle | 0.1716 | 0.2154 | **0.0866** | 0.2035 |
| mean `solve_used` per slot-cell | 0.1339 | 0.1977 | **0.0556** | 0.1876 |
| mean `n_used` per slot-cell | 6.60 | 7.50 | **7.78** | 6.85 |
| sd(`n_used`) | 4.41 | 4.37 | 4.59 | 4.60 |
| Gini of `n_used` | 0.313 | 0.278 | **0.298** | 0.320 |
| cells with `n_used` = 0 | 2 | 0 | **1** | 3 |
| probe cycles / π-carrying cells | 19 / 408 | 19 / 408 | **20 / 424** | 19 / 408 |
| **sd(π) across slots, per probe cycle** | 0.01604 | 0.01573 | **0.00891** | 0.01356 |
| π entropy / top-1 | 2.778 / 0.235 | 2.702 / 0.232 | 2.775 / 0.222 | 2.819 / 0.228 |

**Fewer solves: yes, decisively** — 21,608 against 34,326–39,825, a solve rate of 0.0866
against 0.172–0.215, and a mean per-slot `solve_used` of 0.0556 against 0.134–0.198.

**A different distribution of use over slots: no.** `n_used` has the *highest* mean of the four
(7.78), an ordinary sd (4.59), a Gini squarely inside the others' range (0.298 against
0.278–0.320) and one unused cell against 0–3 elsewhere.

**Fewer π rows: no.** It has *more* — 20 probe cycles and 424 π-carrying cells against 19 and
408.

**What the logs do show as the proximate cause**: **π's mass is about half as spread across
slots** — sd(π) 0.00891 against 0.01356–0.01604 — while its entropy (2.775) and top-1 (0.222)
are unremarkable. A correlation across slots in which one variable has half the spread is
attenuated, and `n_used`'s spread is unchanged. So π~use falls because π is flatter over slots,
not because use moved.

**What the logs do not settle, and is therefore left open**: *why* π is flatter in this arm.
The per-slot record carries π's mass, the beam's use, solves and spelling, and nothing about
the proposal head's inputs — so whether the flattening follows from the halved solve rate
(fewer solved rollouts reaching the head), from the arm's own writes, or from something else is
not in them.

---

# The meter route's objective — three CPU probes, all reported, none asserted (2026-09-11)

`fit_signal = "own_scalar_pg"` is built beside `"own_scalar"` (which is untouched, so `em_q1`
stays reproducible): same head, same rows, same EB gates, and the arc's self-imitation form with
an advantage —

```
loss = - mean_bags  (b - y) * sum_j log P_head(k_j written)
```

`b` an EMA over bags as they arrive (`own_baseline_alpha`, default 0.05). Writes in
better-than-baseline bags are reinforced, writes in worse ones pushed away.

The CPU gate asked for was: on synthetic bags with a known per-feature rule, the self-imitation
objective recovers the per-feature thresholds where the BCE form recovers only the shared one.
**That gate FAILED as specified, and was demoted to reported rather than weakened until it
passed.** Three closed-loop simulations — the head writes `Khat[f, ρ]` at the cells a bag
touches, the bag's label is the fraction of *those* writes that are off-rule, 60 rounds × 64
steps, everything else as in the arm:

| probe | draw | bce | pg |
|---|---|---|---|
| **EB-9** | uniform features | acc_prac **1.0000**, held **0.8750**, θ̂ [2,2,5,5,5,5,5,2] | 0.9583 / 0.8000, θ̂ [1,6,6,5,6,6,6,1] |
| **EB-10** | uniform, with 0 / 10 / 17 / 25 % of the bag's cells dropped while `y` stays over all writes (the arm's own denominator mismatch — `own_scalar_s` drops 0.204 / 0.153 / 0.146 at the `agood` filter) | **1.0000 / 0.8750** at every drop up to 0.17; 1.0000 / 0.8500 at 0.25 | — |
| **EB-11** | the arm's own feature marginal, [0.141, 0.097, 0.098, **0.017**, 0.120, 0.210, 0.149, 0.169], a 12× spread measured on `em_q1b` | **1.0000 / 0.8750** | **1.0000 / 0.8750** |

**In closed loop the BCE form recovers `surface`'s exact table.** Neither the objective, nor the
bag/label denominator mismatch, nor the arm's skewed feature marginal reproduces
`own_scalar_s`'s plateau at θ̂ ≈ 3 / acc held-out 0.7250.

**What the run's own log does show**, and it is a different shape of evidence: `own_scalar_s`'s
fit loss falls monotonically — 0.530 (c1) · 0.738 (c5) · 0.601 (c20) · 0.467 (c40) · 0.415
(c80) · 0.379 (c120) · **0.368 (c140), still falling** — while `acc_practiced` is flat at
0.833–0.875 from c5 and θ̂ stays a near-constant ≈3. **The objective keeps improving and the
recovered table does not.** That decoupling is the signature a calibration target would leave;
that three closed-loop simulations of the same objective do not reproduce it is not explained,
and is on the record as unexplained rather than resolved.

`em_q1c` runs the self-imitation arm anyway: the CPU probes do not predict a difference either
way, and the data is the arbiter.

---

# The two axes — `em_q1c` (objective) and `em_q1d` (keying), 2026-09-11

`em_q1` is the corner where both are the originals: the calibration objective, keyed on the
reader's parse of the arm's own write. Each new tag moves exactly one axis and holds the other,
and each carries its own cross-tag anchor.

| tag | app | arms | axis moved |
|---|---|---|---|
| `em_q1` | `ap-Yh4MSHcGSUJnFeBBxUYcD5` | `canon_s`, `fit_rule_s`, `own_scalar_s` | — (the corner) |
| **`em_q1c`** | `ap-tavcIgRjQ0WRMkkfj7NNKp` | `canon_s`, `own_scalar_pg_s` | **the objective** — self-imitation with an advantage in place of calibration |
| **`em_q1d`** | `ap-CyMeb1Rdv88QDfVVABRUTF` | `canon_s`, `own_scalar_s`, `own_scalar_rec_s` | **the cell key** — the efference copy in place of the reader's parse |

## `own_intent` is per-arm, so `em_q1d` carries its own gate in tag

`own_scalar_s` runs at the run-level default `recall`; `own_scalar_rec_s` is the same arm with
`own_intent: "record"` and nothing else (**EB-0**: `own_scalar_rec_s − own_scalar_s =
{own_intent}`, asserted). On reduction `--merge-tag` asserts that **both** `canon_s` *and*
`own_scalar_s` replay `em_q1` at 0.000e+00, so the coordinator's gate — "with `--own-intent
recall` the arm must replay `em_q1/own_scalar_s`" — is a cross-tag assertion over every logged
series rather than a claim.

**This is also how `em_q1`'s own mis-keying rate is recovered.** It cannot be computed from
that tag (the record was not taken there), but `em_q1d/own_scalar_s` is bit-identical to it by
the assertion above and does log both keys.

## The mis-keying instrument

The write record is now taken for **every** own arm, not only the `record` ones, because the
instrument needs both keys. It stays invisible: the replay snapshots and restores the
executor's `counts`, `last` and capture flag, `_ENTRY_REC["phase"]`, and the whole renderer
tally, and draws no RNG — which is exactly what the 0.000e+00 assertion checks.

Per cycle, in both intents: `n_both_keys`, `n_miskey`, `miskey_rate`, and a per-(feature,
register) tally whose second column is the mis-**keying** rate for these arms (the per-cell
table labels itself accordingly). `analyze_embouchure.py` §5E tables it.

## Preflight (`results/preflight_key.log`, clean, all three arms)

At toy sizes `own_scalar_rec_s` keeps **930 labelled cells against `own_scalar_s`'s 749** from
the same 6,912 blocks read — the record recovers the ~20 % the `agood` filter drops. The
preflight's `n_miskey` is 0, but its reader is 40 steps at `read_acc` 0.816, so that number
carries nothing.

## Gates

| gate | what | status |
|---|---|---|
| **G-F `em_gf5`** | after the objective branch | **PASS — 0.000e+00**, control 0.000e+00, commits equal |
| **G-F `em_gf6`** | after the keying branch | **PASS — 0.000e+00**, control 0.000e+00, commits equal |
| **G-F `em_gf7`** | after the `_is_own` rename and the EB-8 split | **PASS — 0.000e+00**, control 0.000e+00, commits equal |
| **EB-0** | every own arm is a one-bit contrast against the arm it answers to; `fit_signal` defaults to `surface` and `own_intent` to `recall` | **PASS** |
| `gates_cpu` | | **ALL PASS (74 checks)** at this point in the round; the suite ends at 78 |

## `em_q1d` crashed on its first launch, and the crash was two bugs

`em_q1d` (app `ap-CyMeb1Rdv88QDfVVABRUTF`) died at **`canon_s` c20** with
`GATE EB-8 FAILED: the open-loop replay through the executor does not reproduce out['x']`.
Relaunched as `ap-com82TIrVLqnhUjz83kg44` after both were fixed. Neither bug touches any banked
tag: `em_q1`, `em_q1b` and `em_q1c` all ran with the record block's earlier guard
(`_own and _own_intent == "record"`), which never fired because `own_intent` was `recall`.

**(1) A shadowed name.** The record block's guard was `if _own and rule is not None`. The
donor's `adm_tab` construction, twenty lines above, does

```python
for _c, _own in owners.items():
```

— rebinding `_own` to a non-empty list of owner tuples, truthy for every code. The guard
therefore fired in **every ruled arm**, including `canon_s`, which has no rendering organ, no
`rhead`, and nothing to consume a write record. Renamed to `_is_own` and given an explicit
`rhead is not None`. The gate was doing its job on an arm it should never have been evaluated
on, which is the only reason this was caught before a reduction.

**(2) EB-8 was all-or-nothing over 4,096 tokens.** `torch.equal(_xr, out["x"])` fails if one
block of one instance differs, and where the span head fires the beam takes `PerfExecutor`'s
metered branch while the replay takes `_fire_only`. Those are the same `head.emit` on the same
pooled trunk output and they are expected to agree, but expecting is not measuring. EB-8 is now
split:

- **the hard half** — a block whose last writer was a BASE move goes through `apply_any` in the
  beam and in the replay alike, so **any** disagreement there is a broken replay and stops the
  run;
- **the measured half** — a MACRO block may take different branches; disagreements are counted
  (`n_record_mismatch_macro` / `n_record_macro`), the affected blocks are marked
  intent-unavailable (`rec_f = -1`), excluded from the bag and from the mis-keying denominator,
  and reported.

At preflight scale both arms report **0 mismatches on 748 and 664 macro-written blocks**, so
the two branches do agree where it has been possible to check; the full-scale number is a
readout of `em_q1d` rather than an assumption.

## …and crashed a second time, on GPU numerics, which is why EB-8 is now a coverage gate

The relaunch (`ap-com82TIrVLqnhUjz83kg44`) died at **`own_scalar_s` c73**:

```
GATE EB-8 FAILED at c73: the replay disagrees with out['x'] on 1 blocks written by
BASE moves, where both paths are `apply_any` — the write record is broken
```

**One block, after ~41,000 clean ones** (72 cycles × ~576 written blocks) — a rate of 0.0024 %.
That is not the signature of a logic error, which would fire in every cycle. The beam calls
`ex.apply` on the whole flattened beam (batch × width rows) while the replay calls it on the
handful of rows that took that move at that timestep, so the two go through different batch
shapes and different kernels, and a near-tie in `node_features`' max-sum DP argmax can flip.
GPU numerics at a batch-shape boundary.

**EB-8 is now a coverage gate.** Every disagreeing block — base or macro — is marked
intent-unavailable (`rec_f = -1`), excluded from the bag and from the mis-keying denominator,
and counted; base and macro counts are kept apart because their causes differ (a numerical tie
against a branch difference). What is asserted is that the record covers **≥ 50 %** of the
blocks the learner wrote: a genuinely broken replay loses most of the record at once and is
caught immediately, while numerics never approach it. `n_record_mismatch_base` /
`n_record_base` and `n_record_mismatch_macro` / `n_record_macro` go to `own_gate`, so how exact
the efference copy was is a reported number rather than an assumption.

Third launch: `ap-x07UJZGwIlIbkCMVfacPZH`.

**Nothing banked is affected by either crash.** `em_q1`, `em_q1b` and `em_q1c` never ran the
record block at all (their guard also required `own_intent == "record"`, and it was `recall`);
`em_q1c`'s own `[own] canon_s: n_cycles 0` line is that fact on the record.

---

# The two axes, read as one design — `em_q1` + `em_q1b` + `em_q1c` + `em_q1d` (2026-09-11)

`figures/em_q1_joint_reduction.txt`
(`analyze_embouchure.py --tag em_q1 --gf-tag em_gf8 --merge-tag em_q1b,em_q1c,em_q1d`).
Six arms, four tags, 140 cycles each, one world, one clock. Facts only.

## Every cross-tag assertion at 0.000e+00

```
[merge] em_q1b -> em_q1: refs identical = True, stale buffer identical = True
[cross-tag] canon_s:      max|em_q1 - em_q1b| = 0.000e+00 over 140 cycles
[merge] em_q1c -> em_q1: refs identical = True, stale buffer identical = True
[cross-tag] canon_s:      max|em_q1 - em_q1c| = 0.000e+00 over 140 cycles
[merge] em_q1d -> em_q1: refs identical = True, stale buffer identical = True
[cross-tag] canon_s:      max|em_q1 - em_q1d| = 0.000e+00 over 140 cycles
[cross-tag] own_scalar_s: max|em_q1 - em_q1d| = 0.000e+00 over 140 cycles
```

The last line is the gate that makes `em_q1d`'s mis-keying number `em_q1`'s number: the record
was never taken in `em_q1`, and `em_q1d/own_scalar_s` is bit-identical to it on every logged
series.

## §2E — the time course, all six arms

| arm | tag | signal · key | acc held-out | first reached **and held** | lifetime written spelling error | θ̂ at c140 |
|---|---|---|---|---|---|---|
| `fit_rule_s` (**surface**) | `em_q1` | world's blocks | **0.8750** | c19 | **0.0051** | [2,2,5,5,5,5,5,2] |
| **`own_scalar_rec_s`** | `em_q1d` | meter scalar · **record** | **0.8750** | **c7** | **0.0073** | **[2,2,5,5,5,5,5,2]** |
| `own_scalar_s` | `em_q1`/`em_q1d` | meter scalar · recall | 0.7250 | c12 | 0.1880 | [3,4,4,3,4,3,3,3] |
| `own_readback_s` | `em_q1b` | own reader, per block | 0.4500 | c25 | 0.3865 | [5,6,8,8,8,8,8,8] |
| `own_scalar_pg_s` | `em_q1c` | **self-imitation** · recall | 0.4250 | c133 | 0.3279 | [6,6,2,5,7,1,1,5] |
| `canon_s` | all | — | — | — | 0.4592 | — |

**The objective axis does not carry the effect.** `own_scalar_pg_s` — the self-imitation form
with an advantage, same key — reaches 0.4250, *below* the calibration form's 0.7250.

**The keying axis carries all of it.** `own_scalar_rec_s` — the calibration objective unchanged,
only `own_intent` moved — reaches `surface`'s exact table, θ̂ and all, at **cycle 7**, twelve
cycles earlier than `surface` itself, with a written spelling error of 0.0073 against surface's
0.0051 and `own_scalar_s`'s 0.1880.

## §5E — the mis-keying rate is ZERO, and the mechanism is elsewhere

| | `own_scalar_s` (recall) | `own_scalar_rec_s` (record) |
|---|---|---|
| blocks where both keys exist | 68,886 | 79,419 |
| **mis-keyed** (reader's parse ≠ efference copy) | **0** | **0** |
| rate | **0.00000** | **0.00000** |

**The reader never returns a different owner for the learner's own write.** Not the 15–20 % the
`agood` drop hinted at, not Q0's 1 % — zero, over 148,305 blocks.

**What the key actually changes is which blocks are USABLE, and the dropped set is 99 % errors:**

| arm | own writes | usable cells | dropped | off-rule among **kept** | off-rule among **dropped** | enrichment |
|---|---|---|---|---|---|---|
| `own_scalar_s` (recall) | 81,926 | 68,892 | **13,034 (15.91 %)** | 0.04471 | **0.99148** | **22.2×** |
| `own_scalar_rec_s` (record) | 79,772 | 79,765 | 7 (0.01 %) | 0.00868 | — | — |
| `own_readback_s` | 79,588 | 79,588 | 0 | 0.41148 | — | — |

Under `recall` the bag holds the arm's **correct** writes and excludes its **wrong** ones —
the reader fails to parse almost exactly the blocks that are misspelled — while the label `y`,
which comes from the grader, still counts every write. The bag and the label disagree about
which blocks are in play, in a way 22× correlated with the error. Under `record` that
disagreement is 7 blocks in 79,772.

## The efference copy's own exactness (gate EB-8, measured)

| arm | base blocks | base mismatch | macro blocks | macro mismatch |
|---|---|---|---|---|
| `own_scalar_s` | 34,444 | **4** | 47,482 | **2** |
| `own_scalar_rec_s` | 33,256 | **5** | 46,516 | **2** |

6 and 7 blocks of ~80,000 — **0.008 %**. The metered branch and `_fire_only` agree on macro
blocks to 2 in 47,000, and the base-move disagreements are the batch-shape numerical ties that
killed the second launch. The write record is the learner's own command to four decimal places.

## The meaning currency (§3I) and lifetime solves

| arm | `e` (end) | `e_sp` written, lifetime | lifetime solves |
|---|---|---|---|
| `own_scalar_rec_s` | **0.6354** | 0.0334 | 29,562 |
| `fit_rule_s` | 0.6068 | 0.0231 | 39,825 |
| `own_readback_s` | 0.6797 | 0.3944 | 38,331 |
| `canon_s` | 0.6979 | 0.4592 | 34,326 |
| `own_scalar_s` | 0.7474 | 0.2049 | 21,608 |
| `own_scalar_pg_s` | **0.7917** | 0.3470 | 19,365 |

**The `e` floor, re-derived IN TAG** (§7E, `em_q1/canon_s`, the donors' `null_abba` at span 1,
W 4, 116 windows over 140 cycles with the 4 era-boundary/commit cycles **skipped** rather than
deleted): **v_tol(e) = 0.0224916**, against the donor tag's 0.0173 on its own world.

| arm | `e`(end) | − `canon_s` | in floors |
|---|---|---|---|
| `fit_rule_s` | 0.6068 | −0.0911 | **−4.05** |
| `own_scalar_rec_s` | 0.6354 | −0.0625 | **−2.78** |
| `own_readback_s` | 0.6797 | −0.0182 | −0.81 |
| `canon_s` | 0.6979 | 0.0000 | 0.00 |
| `own_scalar_s` | 0.7474 | +0.0495 | **+2.20** |
| `own_scalar_pg_s` | 0.7917 | +0.0937 | **+4.17** |

**A correction to an earlier reading in this file.** The multiples first reported here
(`own_scalar_s` +2.86, `own_scalar_pg_s` +5.4, `fit_rule_s` 5.27 below `canon_s`) used the
DONOR tag's floor, 0.0173, because no in-tag `e` floor had been derived. On this tag's own
floor they are +2.20, +4.17 and −4.05. The signs and the order do not move; the multiples were
inflated ≈ 1.3×.

**And a method note, because the first attempt got it wrong by 3×.** Deleting the skipped
cycles from the series before `null_abba` splices non-adjacent cycles together and inflates
sd(N): that gives 0.0521 on the same arm, against 0.0225 when the cycles are passed as `skip=`
and the WINDOWS touching them are dropped, which is the donors' own idiom. The 0.0521 was never
reported; it is recorded here so the next derivation does not repeat it.

## Four simulations still do not reproduce the plateau

The run answers the question; the CPU probes do not explain it. All four recover `surface`'s
exact table with the calibration objective:

| probe | what it varied | bce result |
|---|---|---|
| EB-9 | uniform feature draw | acc_prac 1.0000 / held 0.8750 |
| EB-10 | random cell dropout, 0–25 % | 1.0000 / 0.8750 up to 0.17 |
| EB-11 | the arm's own 12× feature marginal | 1.0000 / 0.8750 |
| **EB-12** | **error-correlated dropout at the arm's measured rates** (P(drop\|off-rule) 0.808, P(drop\|on-rule) 0.0017) | **1.0000 / 0.8750** |

EB-12 was written specifically to close EB-10's gap and does not. So the empirical answer is
unambiguous — the keying axis is the whole effect and the objective axis is not — while the
mechanism by which the recall key costs 0.15 in held-out accuracy remains **unreproduced in
simulation**, and is on the record as such.

---

# Q2.1 — the homophonous lexicon at H=3 (`em_q2`, 2026-09-11, app `ap-NbITHQ0asNS4xV84M1zZcr`)

Clean, **12524 s = 3.48 GPU-h**, five arms, 140 cycles each. `if_q1c_yk`'s flag line with
`--lexicon-merge "0:1=1:1,2:0=4:0,5:0=6:0"` and `--own-intent record`; reader A (the
substrate's own) in every arm, reader B in none. Reduction:
`figures/em_q2_reduction.txt`. Facts only.

**Note on the run**: the local `modal run --detach` client died in a container restart and its
log ends in a proxy error; the remote app finished normally. `done.txt` (`elapsed 12524s`) and
all five arms are on the volume, and the waiter was re-armed on `modal app list` rather than
the log (`results/wait_app.sh`).

## The lexicon

3 shared forms of 13; **35 (feature, register) cells** sit on a shared form and **1** of them
the rule still resolves.

| form | owners | last writer keeps |
|---|---|---|
| 12 | (f0,k1) (f1,k1) | f=1 |
| 28 | (f5,k0) (f6,k0) | f=6 |
| 33 | (f2,k0) (f4,k0) | f=4 |

**GATE RB-1 in tag**: flipped read changed **0.5859**, wrong 0.5684, baseline error 0.0176,
collateral 0.0016 (n = 1,024).

Per register (n = 128 each): ρ0 0.500 · ρ1 0.523 · ρ2 0.414 · ρ3 0.438 · ρ4 0.508 · ρ5 0.609 ·
ρ6 0.867 · ρ7 0.828. Baseline read error is 0 at every register but ρ4/ρ5 (0.018 pooled), so
the gate is readable everywhere in this tag — unlike the preflight, where `read_acc` 0.816 made
it unreadable.

## The Q2.1 machinery record

**The five arms.** One conductor, one world, one seed; the arms differ only in where the
realization step's table comes from.

| arm | realization step | `fit_signal` | cell key |
|---|---|---|---|
| `canon_s` | constant — always the canonical synonym | — | — |
| `given_rule_s` | the world's own `K` handed over, never fitted | — | — |
| `fit_rule_s` | fitted from blocks **the world wrote** | `surface` | the block's own (feature, register) |
| `own_scalar_rec_s` | fitted from the learner's **own** writes, one scalar per bag | `own_scalar` | the **write record** (`own_intent=record`) |
| `own_readback_s` | fitted from the learner's own writes, parsed back by its own reader | `own_readback` | the reader's parse, with the record as the EB-6 control |

**The lexicon exactly as `setup.json` records it** (`figures/em_q2/setup.json → lexicon`, written
by `_lexicon_record`):

```
merge            "0:1=1:1,2:0=4:0,5:0=6:0"      # "f1:k1=f2:k2" -> bottom[f2,k2] := bottom[f1,k1]
                                                # so bottom[1,1] takes f0's spelling, etc.
n_forms          13          n_shared_forms  3
shared_forms     12 : owners [[0,1],[1,1]]  last_writer_keeps 1
                 28 : owners [[5,0],[6,0]]  last_writer_keeps 6
                 33 : owners [[2,0],[4,0]]  last_writer_keeps 4
cells            35 in all — (f0,ρ1..7) (f1,ρ1..7) | (f5,ρ0..4) (f6,ρ0..4) | (f2,ρ0..5) (f4,ρ0..4)
cells_rule_resolves   [[2,5]]
```

A cell (f, ρ) is on a shared form when `K[f, ρ]` sends f to the merged synonym at that register;
that is why the ranges follow θ = [1,1,6,4,5,5,5,1] (f0, f1 at θ=1 take k1 from ρ≥1; f5, f6 at
θ=5 take k0 below ρ=5; f2 at θ=6 takes k0 below ρ=6, f4 at θ=5 below ρ=5). **(f2, ρ5) is the one
cell where the two owners part** — f2 still writes k0 there while f4 has crossed to k1 — so the
form identifies the feature and the rule resolves it. **ρ=5 is an unpractised register**, which
makes that single cell a register-generalisation test and not a memorisation one.

**Setup in tag**: `read_acc` **1.000** · `setup_render` `rule` with `buffer_on_rule` **1.0000**
and `buffer_on_grammar` 1.0000 over 262,144 scored blocks (8,192 rows) · `entry_recorder_check`
**PASS** (48 macros, 1,152 executions recorded at levels 2–3, `max_abs_feats` 0.0,
`max_abs_pos` 0.0) · setup 707.7 s · era caps 40/45/55 = **140 cycles per arm** · rule `E_R8`,
v 8, m 2, n_ctx 8, `context_kind` register, θ [1,1,6,4,5,5,5,1], practised {0,3,7}, held out
alias {1,2,6} / novel {4,5} · `rule_seed` 6, `train_seed` 1, `seed` 0.

**The §7E floor, as derived**: reference arm `canon_s` in this tag, the donors' `null_abba` at
span 1, W 4, `v_tol = sd(N)/2`, **120 windows** over 140 cycles with the era-boundary and commit
cycles passed as `skip=` (the windows that *touch* them are dropped) rather than deleted from
the series — deleting them splices non-adjacent cycles and inflates sd(N) by ≈ 3×, which is the
error corrected earlier in this file.

**The §8E method** (`mined_precision_two_spaces`, `analyze_embouchure.py`): the arm's mined
keys are the distinct `keys_at_support` pooled over the whole run at each level (`mine_support`
3). The world's true tuples come from `setup["true_tables"][level]["flat"]`; the bottom map is
rebuilt **offline** from `setup["config"]["rule_seed"]` (6) and `setup["lexicon"]["merge"]`, so
nothing extra had to be logged in-run. A key is **true-in-feature** iff the tuple is one of
those true tuples. It is **true-in-token** iff, for **at least one** practised register
ρ ∈ {0,3,7}, the code string the key spells at ρ under the true rule equals the string some
true tuple spells at that same ρ — `render(tup, ρ) = (bot[f, K[f, ρ]] · v^arange(s)).sum()`,
per feature. The two therefore differ exactly where the lexicon (or the realization step) makes
two different feature tuples produce one surface, and `precision(token) − precision(feature)`
is the share of an arm's mined keys that are false as meanings and indistinguishable as sound.
The miner keys on the **reader's** parse, which at a shared form returns the last-writer owner
— which is how a chunk the learner produced as (f1, f2) can be banked as (f1′, f2′).

## §6E — recovered K at the shared-form cells, by route

At c140:

| arm | signal | acc SHARED | acc SHARED ∩ practised | acc rule-resolves | acc unshared | acc all |
|---|---|---|---|---|---|---|
| `own_scalar_rec_s` | meter scalar · record | **0.7714** | **0.9167** | **0.0000** | 0.8621 | 0.8125 |
| `fit_rule_s` (surface) | world's blocks | 0.7429 | 0.7500 | **0.0000** | **0.9310** | 0.8281 |
| `own_readback_s` | own reader | 0.6000 | 0.7500 | **0.0000** | 0.5172 | 0.5625 |

`surface` is the best of the three on the **unshared** cells (0.9310) and the worst gap between
unshared and shared (0.9310 → 0.7429). `own_scalar_rec_s` is the best on the **shared** cells
and on shared ∩ practised (0.9167). `own_readback_s` is the only route whose shared accuracy
**exceeds** its unshared (0.6000 against 0.5172).

**No route ever recovers the one cell where the rule resolves a shared form — 0.0000 at every
cycle, in all three.**

## §4E — the two instruments, with the shared cells marked

`own_readback_s`, pooled over 80,148 own-written cells: off-rule **0.19350**, mis-read
**0.02581**. Split by the lexicon:

| | off-rule | mis-read | n |
|---|---|---|---|
| **SHARED-form cells** | **0.06936** | **0.00116** | 43,237 |
| **unshared cells** | **0.33892** | **0.05470** | 36,911 |

Its mis-phrasing rate falls over the ladder: 0.04678 / 0.02047 / 0.02021 by era, with
`n_ambiguous` 0.10 / 0.04 / 0.02 per cycle.

`own_scalar_rec_s`: off-rule **0.00528**, mis-keyed **0.00042** (n 78,803), read-back drop
0.00003 / 0.00031 / 0.00008 by era. Its split runs the same way: SHARED-form cells off-rule
**0.00173** / mis **0.00002** (n 45,196) against unshared **0.01006** / **0.00095** (n 33,607)
— a 5.8× off-rule ratio where `own_readback_s`'s is 4.9×.

## §5E — mis-keying is non-zero for the first time

| arm | era | both keys | mis-keyed | rate |
|---|---|---|---|---|
| `own_scalar_rec_s` | 1 | 431.9 | 0.80 | 0.00185 |
| | 2 | 572.6 | 0.02 | 0.00004 |
| | 3 | 648.5 | 0.00 | 0.00000 |
| | **ALL** | **78,707** | **33** | **0.00042** |

On the injective world the rate was exactly 0 over 148,305 blocks. Here it is 33 blocks, all
but one of them in era 1, and it decays to 0 by era 3.

## §3I / §7E — the two numbers, and the floor re-derived in tag

**v_tol(e) = 0.0136302** (`em_q2/canon_s`, 120 null-ABBA windows over 140 cycles, span 1, W 4).

| arm | `e`(end) | − `canon_s` | in floors | `e_sp` written, lifetime |
|---|---|---|---|---|
| `own_readback_s` | 0.6927 | −0.0026 | −0.19 | 0.1721 |
| `canon_s` | 0.6953 | 0.0000 | 0.00 | 0.4191 |
| `fit_rule_s` | 0.7083 | +0.0130 | +0.96 | 0.0229 |
| `own_scalar_rec_s` | **0.7812** | +0.0859 | **+6.30** | 0.0208 |
| `given_rule_s` | **0.7839** | +0.0885 | **+6.50** | **0.0000** |

**`given_rule_s` — the ceiling, which never misspells — carries the worst meaning error in the
tag**, 6.5 floors above `canon_s`, with `own_scalar_rec_s` beside it at 6.30. The two arms with
the *best* spelling have the *worst* meaning; the two with the worst spelling (`canon_s` 0.4191,
`own_readback_s` 0.1721) have the best. On the injective world (`if_q1c_yk`, `em_q1`) the order
was the opposite: correct spellers clustered at e ≈ 0.61 and misspellers at e ≈ 0.70.

## §8E — the mined table: feature space against token space

A mined key is true-in-feature if it is one of the world's own tuples; true-in-token if what it
spells at a practised register is what some true tuple spells there.

| arm | L2 keys | precision(feature) | precision(token) | gap | L3 keys | precision(feature) | precision(token) | gap |
|---|---|---|---|---|---|---|---|---|
| `canon_s` | 15 | 0.4667 | 0.8000 | +0.333 | 41 | 0.2195 | 0.8780 | **+0.659** |
| `given_rule_s` | 15 | 0.6000 | **1.0000** | +0.400 | 40 | 0.2250 | **1.0000** | **+0.775** |
| `fit_rule_s` | 16 | 0.6250 | **1.0000** | +0.375 | 46 | 0.2174 | **1.0000** | **+0.783** |
| `own_readback_s` | 13 | 0.5385 | 0.9231 | +0.385 | 42 | 0.2381 | **1.0000** | +0.762 |
| `own_scalar_rec_s` | 12 | 0.6667 | **1.0000** | +0.333 | 41 | 0.2439 | **1.0000** | +0.756 |

At L3 roughly **78 % of every arm's mined keys are false in feature space and indistinguishable
in token space** — they spell exactly what some true tuple spells. Only `canon_s`, which
misspells 42 % of its writes, has token-space precision below 1.000 at either level.

## Gates in tag

G-F `em_gf9` **0.000e+00** · `read_acc` 1.000 · `n_unbound` 0 in all five arms · EB-1 / EB-2 0
over 140 cycles in both own arms · EB-6 0 disagreements · record mismatch 12 of 78,815
(0.015 %) · `n_ambiguous` 0 under `--own-intent record`, where Q2.0 projected `recall` would
lose 12–78 % of its cells.

---

# Q2.2 — the listener's teacher, and the chooser's (`em_q2b_l` + `em_q2b`, 2026-09-12)

**MODIFIED SUBSTRATE.** Both tags change a component of the learner, not of the world, and
neither may be merged with a reader-A tag in a reduction. `setup.json`'s **first key** is
`substrate_mod`, naming which ears moved; the run log and the reduction print the same banner.

| tag | app | GPU | `reader_target` | listener (`shared["reader"]`) | chooser (the generator's SETUP supervision) |
|---|---|---|---|---|---|
| `em_q2` (banked) | `ap-NbITHQ0asNS4xV84M1zZcr` | 3.48 | `last_writer` | last-writer table | last-writer table |
| **`em_q2b_l`** | `ap-uDj1FSpldx8bn1cL7gdPlt` | **11674 s = 3.24 GPU-h** | `listener` | **derived features** | last-writer table |
| **`em_q2b`** | `ap-3cKaCQ9E1L3N3sDkJWzPoB` | **11937 s = 3.32 GPU-h** | `both` | **derived features** | **derived features** |

One flag line for the pair (`results/launch_q22.sh <tag> <listener|both>`); the only difference
on disk is `--reader-target`. Otherwise `em_q2`'s line exactly: five arms, H=3 lexicon,
`--own-intent record`, `--reader-steps 6000`, 140 cycles, seed 0. Reduction:
`figures/em_q2b_reduction.txt` (`--tag em_q2b --gf-tag em_gf12 --vs-tag em_q2b_l,em_q2`).
Facts only.

## The gates

**G-F run three times this round, all 0.000e+00**: `em_gf10` (`ap-b49evzpT1NRPvVikRZPQiL`)
after the listener change, `em_gf11` (`ap-iSQ0pGRtQxNmvtjQ35RFRz`) after the generator change,
`em_gf12` (`ap-Lembd16aJXkybDfb0ot7aB`'s successor, `ap-93vDfqEkxgeDQ5iaVGhdnh`) after the RB-1
change — anchor and `given_c1`, control 0.000e+00, commits equal, Q-11 PASS (18 cells) each
time. `gates_cpu` **ALL PASS (81 checks)**, including two new ones:

- **EB-16** (the listener's teacher): the corpus is bit-identical with and without the feature
  capture; the captured features re-render through the rule back to the corpus exactly; the two
  teachers disagree ONLY at loser cells — **18 loser cells and 0.2350** of blocks on the H=3
  lexicon, **0 and 0.0000** with no merge, so reader B *is* reader A on the arc's own world.
- **EB-17** (the chooser's teacher): `_train_generator_feats` consumes the global torch RNG
  exactly as the donor's `_train_generator` does (12 draws, same order and shapes), and a
  300-step fit parses **0.3156** to the derived feature against **0.1393** to the last-writer
  one at the loser blocks.

**In tag, both tags**: `read_acc` (own teacher) **0.9949**, `plant.read_acc_derived` 0.9932
against `plant.read_acc` 0.7654 (the last-writer reference, which the ear is no longer taught);
`buffer_on_rule` 1.0000 over 262,144 blocks; EB-1 0, EB-2 0; EB-6 **0** disagreements of
52,873 (`em_q2b`) and 49,910 (`em_q2b_l`); record mismatch 16–17 of ~77,000 and 13–16 of
~73,000 (**0.02 %**); `n_ambiguous` 4 and 5 (against 0 in `em_q2`).

**GATE RB-1, in both tags, identical to the digit** — the reader is the same object in the pair:

| scored against | n | base_read_wrong | flipped_read_changed | flipped_read_wrong | collateral |
|---|---|---|---|---|---|
| last-writer table | 1024 | **0.2480** | 0.5488 | 0.4619 | 0.0101 |
| **derived feature** | 1024 | **0.0332** | 0.4141 | 0.3877 | 0.0098 |

The `base_read_wrong` gap is the measurement artifact the derived pass was added for: scored
against the table the ear is no longer taught, the baseline reads the teacher-disagreement rate
(0.2358) rather than the ear's error. `em_q2`'s own baseline was 0.0176.

## §9E — the loser cells' own-write counts, the readout the round was built for

Pooled over the six practised loser cells (f0:ρ3, f0:ρ7, f2:ρ0, f2:ρ3, f5:ρ0, f5:ρ3):

| arm | `em_q2` (A) | `em_q2b_l` (listener) | `em_q2b` (both) |
|---|---|---|---|
| `own_readback_s` | 78 / 43,159 = **0.00180** | 72 / 32,646 = **0.00220** | 613 / 34,789 = **0.01732** |
| `own_scalar_rec_s` | 7 / 45,189 = **0.00015** | 242 / 35,652 = **0.00674** | 449 / 37,113 = **0.01195** |

Per cell in `em_q2b`, `own_readback_s`: f0:ρ3 **209** (was 68), f0:ρ7 **60** (was 7), f2:ρ0 35
(was 0), f2:ρ3 **71** (was 1), f5:ρ0 109 (was 0), f5:ρ3 **129** (was 2).

## §8E — the mined table, precision in feature space against token space

precision(feature) at L3, and the token−feature gap:

| arm | `em_q2` | `em_q2b_l` | `em_q2b` |
|---|---|---|---|
| `canon_s` | 0.2195 (gap +0.659) | 0.7436 (+0.077) | 0.7632 (+0.026) |
| `fit_rule_s` | 0.2174 (+0.783) | **1.0000 (+0.000)** | **1.0000 (+0.000)** |
| `given_rule_s` | 0.2250 (+0.775) | **1.0000 (+0.000)** | **1.0000 (+0.000)** |
| `own_readback_s` | 0.2381 (+0.762) | 0.8049 (+0.073) | 0.8214 (+0.036) |
| `own_scalar_rec_s` | 0.2439 (+0.756) | **1.0000 (+0.000)** | **1.0000 (+0.000)** |

At L2 every arm but `canon_s` is at 1.0000 in both reader-B tags.

## §6E — recovered K at the shared cells, c140 (SHARED / SHARED∩prac / unshared / all)

| arm | `em_q2` | `em_q2b_l` | `em_q2b` |
|---|---|---|---|
| `fit_rule_s` | 0.7429 / 0.7500 / 0.9310 / 0.8281 | **0.9143 / 1.0000** / 0.9310 / 0.9219 | **0.9143 / 1.0000** / 0.9310 / 0.9219 |
| `own_scalar_rec_s` | 0.7714 / 0.9167 / 0.8621 / 0.8125 | 0.8000 / 0.9167 / 0.9310 / 0.8594 | **0.9143 / 1.0000** / 0.9310 / 0.9219 |
| `own_readback_s` | 0.6000 / 0.7500 / 0.5172 / 0.5625 | **0.2857 / 0.4167** / 0.5172 / 0.3906 | 0.4571 / 0.5833 / 0.5172 / 0.4844 |

`acc rule-resolves` is **1.0000 at c1** in both reader-B tags (0.0000 at c1 in `em_q2`) and
**0.0000 from c5/c10 onward** in all three.

## §4E — the two instruments at the shared cells

`own_readback_s`, pooled and split:

| tag | pooled off-rule / mis-read | SHARED cells | unshared cells |
|---|---|---|---|
| `em_q2` | 0.19350 / 0.02581 (n 80,148) | 0.06936 / 0.00116 | 0.33892 / 0.05470 |
| `em_q2b_l` | 0.45324 / 0.21788 (n 71,300) | **0.49156 / 0.38059** | 0.42074 / 0.07991 |
| `em_q2b` | 0.44045 / 0.15655 (n 76,104) | **0.45526 / 0.26840** | 0.42757 / 0.05926 |

`own_scalar_rec_s`:

| tag | pooled off-rule / mis-keyed | SHARED cells | unshared cells |
|---|---|---|---|
| `em_q2` | 0.00528 / 0.00042 | 0.00173 / 0.00002 | 0.01006 / 0.00095 |
| `em_q2b_l` | 0.00591 / 0.13905 | **0.00293 / 0.28918** | 0.00864 / 0.00125 |
| `em_q2b` | 0.00755 / 0.11945 | **0.00871 / 0.24799** | 0.00648 / 0.00091 |

Under reader A the shared cells were the CLEANER half on both instruments. Under reader B the
mis-read/mis-key rate at the shared cells is 0.25–0.38 against 0.0009–0.08 unshared, while the
off-rule rate stays flat across the split.

## §5E — mis-keying

| tag | both keys | mis-keyed | rate | by era |
|---|---|---|---|---|
| `em_q2` | 78,707 | 33 | **0.00042** | 0.00185 / 0.00004 / 0.00000 |
| `em_q2b_l` | 74,927 | 10,429 | **0.13919** | 0.23845 / 0.12030 / 0.10472 |
| `em_q2b` | 78,070 | 9,352 | **0.11979** | 0.15850 / 0.12384 / 0.09567 |

## §7E — the in-tag `e` floor and the meaning order

| tag | v_tol(e) | order, e(end), in floors from `canon_s` |
|---|---|---|
| `em_q2` | 0.0136302 | own_readback −0.19 · **canon 0.00** · fit_rule +0.96 · own_scalar_rec **+6.30** · given_rule **+6.50** |
| `em_q2b_l` | 0.0127517 | given_rule **−11.64** · own_readback **−9.19** · own_scalar_rec −4.90 · fit_rule −2.65 · **canon 0.00 (worst, e 0.8854)** |
| `em_q2b` | 0.0220758 | fit_rule −0.59 · own_scalar_rec −0.47 · **canon 0.00** · own_readback +0.24 · given_rule +0.94 |

`em_q2b_l` restores the injective world's order (the ceiling best, `canon_s` worst) and widens
it to 11.6 floors; `em_q2b` collapses the whole field into a ±1-floor band.

## §11E — the chooser's ear over the run, at the loser cells (n 3,845 per probe)

The generator's SETUP supervision is the derived features only at `both`; its IN-RUN target is
`bottom_map` in every tag. The reader's own two columns are **0.9854 / 0.0146 at every cycle of
both tags** — it is frozen and it is the same object, which is what makes the pair a contrast.

| tag | arm | gen→derived c1 → c140 | gen→last-writer c1 → c140 |
|---|---|---|---|
| `em_q2b` | `canon_s` | **0.4947 → 0.0741 (−0.4205)** | 0.2453 → 0.6200 (+0.3748) |
| `em_q2b` | `given_rule_s` | 0.5277 → 0.1274 (−0.4003) | 0.2205 → 0.4869 (+0.2663) |
| `em_q2b` | `fit_rule_s` | 0.4947 → 0.1251 (−0.3696) | 0.2453 → 0.5118 (+0.2666) |
| `em_q2b` | `own_scalar_rec_s` | 0.4947 → 0.0926 (−0.4021) | 0.2453 → 0.5381 (+0.2928) |
| `em_q2b` | `own_readback_s` | 0.4947 → 0.0837 (−0.4109) | 0.2453 → 0.5469 (+0.3017) |
| `em_q2b_l` | `canon_s` | 0.0021 → 0.0562 (+0.0541) | 0.6840 → 0.6531 (−0.0309) |
| `em_q2b_l` | the other four | 0.0010–0.0021 → 0.0086–0.0252 | 0.6840–0.7550 → 0.6343–0.6772 |

**At setup, before any cycle**, `em_q2b`'s generator was at gen→derived **0.7899** /
gen→last-writer **0.0354**; `em_q2b_l`'s at **0.0010 / 0.6861**. By c8 `em_q2b` is already at
0.1659 / 0.4819 and by c16 at 0.1300 / 0.5342.

## §10E — the cross-tag `canon_s` check, measured and not asserted

Printed as the record of the substrate change, per the coordinator's instruction:

```
em_q2b vs em_q2b_l  canon_s  e  max|d| = 3.6458e-01  first divergence c0
em_q2b vs em_q2     canon_s  e  max|d| = 3.6458e-01  first divergence c0
```

(the reduction reports `t_cum` as the worst series at 2.356e+05; `e` is the substantive one and
is given here.) The world is confirmed the same object in all three: `lexicon.merge`,
`n_forms` 13 and `n_shared_forms` 3 all SAME.
