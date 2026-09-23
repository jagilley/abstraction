# DESIGN — `sostenuto`: the value reader's level as the admission gate on the miner's live build, in the loop

**Brief**: [`SPEC.md`](SPEC.md) (the orchestrator's, verbatim) · **Machinery record**: [`FILES.md`](FILES.md) ·
**Donor**: [`../soundboard/soundboard.py`](../soundboard/soundboard.py) at its `sb_s1`/`sb_s2` head ·
**The offline rounds this reproduces**: [`../preplay/README.md`](../preplay/README.md) §5 (pp4) and §6 (pp5),
[`../preplay/readgate.py`](../preplay/readgate.py) (`decide`, `gated_walk`), [`../preplay/preplay.py`](../preplay/preplay.py)
(`pj_predict`, `fire_rec`) · **Motivation**: `ideas/calibration_and_violation_are_one_object.md` §12.4.

---

## §0 What this node adds, in one paragraph

pp5 put the shaped projection's level in the seat of the world's verdict on a try-and-keep walk over candidate
table entries, offline, on banked plants, and found it captures 0.27 of the oracle gate's advantage over no gate at
the loop's own budget of eight auditions and 0.57 at the full walk, while a frozen or never-trained readout in the
same seat is worse than no gate. Nothing ran in the loop, and — the correction the orchestrator found while
designing this round — the walk pp4/pp5 transcribed (`census_extend`) never ran on the arms of record at all. This
node builds the op those arms *could* carry, in their own currency: an **admission set** over the miner's live
support-count build, walked at the loop's own cadence, with the gate as the single knob (`world` / `read` / `none`).

---

## §1 The two design choices the brief left open

### §1.1 A rejected key is decided ONCE, not re-offered (`adm_reoffer = False`)

The offline walk decides once; `census_extend` re-offers, because it recomputes `extend_candidates` against the
frozen table every pass and a rejected key is not in the frozen table. Four reasons for deciding once:

1. **It is the object pp4/pp5 measured.** Gate **A-1** asserts that one real in-loop pass reproduces
   `readgate.gated_walk` on the same base, candidates, order and pool. `gated_walk` walks a fixed candidate list once.
   A re-offering op is a different op and the reproduction gate would be asserting something weaker.
2. **The cap drains the queue.** The order is `(-count, key)` and is stable, so a rejected high-count key sits at
   the head of the order forever. With `extend_cap = 8` and, at L4, 39 at-support keys, three persistent rejects cost
   three eighths of every pass's cap and the tail keys are never offered. Deciding once spends the cap on new
   information.
3. **It bounds the bill by the inventory rather than by the clock.** Total trials ≈ the number of distinct keys the
   run ever supports (~100–150 at seed 0), not passes × cap (~40 × 8 × 4 levels).
4. **It makes the per-candidate confusion a per-candidate statistic.** Each key contributes exactly one trial and one
   (gate decision, world decision) pair, which is the form pp5's TT/TF/FT/FF table is in.

The cost, stated: a key rejected early against a weak plant and a small base is rejected permanently, and the read is
a difference of two compressed levels, so an early mistake is baked in. **One second chance exists and is not a
special case**: a merge re-keys the miner's counts (`ClassMiner.rekey`), and the admission set is re-keyed with the
same map, so a key that merges into a new class becomes undecided again and is re-offered. Collisions (an admitted
key and a rejected key merging into one) resolve to **admitted**, and are counted (`n_rekey_collide`) — the merged
class's rows are a superset of an admitted class's rows, so the positive evidence is about rows that will be served
either way. The rejected set is logged in full, so a later round can measure what re-offering would have bought
without re-running the arms.

### §1.2 The gate touches EVERY row: the admission set starts EMPTY at every level (`adm_seed_commit = False`)

The alternative is to seed the admission set with the commit-time build and gate only the growth (`census_extend`'s
shape). Against it:

- **The junk is already in at commit time.** At seed 0 the L3 commit installs 59 rows at precision 0.31 and the L4
  commit 88 rows at precision 0.36. Seeding admits all of that unexamined and leaves the gate only the growth, which
  is where the question has the least power.
- **pp4/pp5's setting (b) is the empty base**, so the in-loop and offline numbers are in the same setting.
- **The seat is the live build.** Under `open_inventory` the commit freezes only the *flag*; the table served is
  rebuilt every cycle (157 of 201 cycles at seed 0). "Gate the growth" has no natural referent when the whole table
  is growth.

**Servability, and what the fallback is.** `operative(ell)` must always return a table with at least one row for an
adopted level, or the macro has no entry axis for `macro_features`' argmax. Three states, all counted per level:

| state of the admission set at `ell` | what `operative(ell)` serves | counter |
|---|---|---|
| **pending** — no pass has walked this level yet | the ungated live build (the donor's own line) | `n_pending_serve` |
| **in force**, non-empty | the live build filtered to admitted keys | — |
| **in force**, empty, level adopted | the committed snapshot (the donor's own fallback) | `n_empty_fallback` |
| **in force**, empty, level not adopted | the empty table — the gate's verdict stands | `n_admitted_empty` |

The *pending* state is the one new thing and it is deliberate: before the first pass at a level no decision has been
made, so there is nothing for the gate to be in force about, and the arm is the donor exactly. The first pass at a
level happens within `recert_every = 5` cycles of that level's first at-support key, so the window is at most five
cycles (L2: keys from c1, first pass c5; L3: keys from c63, first pass c65; L4: c111 → c115; L5: c106 → c110, at
seed 0 on the banked `sb_sv_yk`). Once walked, the gate is in force even if it has admitted nothing — an empty
admitted build is a *finding*, not a servability problem, and is reported.

**The one residual risk, stated before the launch.** A commit builds `tbl = miners[active].build(operative(active-1),
support)` and is **cancelled** if that table is empty. `operative(1)` is the base table and is never gated, so the L2
commit is safe; L3/L4/L5 commits are safe iff the level below has admitted at least one key by then. At the world
gate's ~95% admission rate over ≥13 keys this is not a live worry, but it is exactly the failure that would break the
yoke (Y-1 requires 0 cancelled), so `n_commit_cancelled_empty` is asserted to be 0 post hoc and printed loudly in the
run.

---

## §2 The op, exactly

New per-arm knob `admit ∈ {None, "world", "read", "none"}`; `None` (every other arm in the file) is the donor.

**Cadence and scope.** In block **(g2.5)**, immediately after the live recert and before `census_extend`'s block, on
every cycle with `c_in_era % recert_every == 0` (the loop's own cadence, 5): for every level `lv ∈ [2, maxl]` whose
miner has at-support keys, and whose level-`lv` node containing the era's cell is in range.

**Candidates.** The not-yet-decided at-support keys, in `sorted(counts.items(), key=(-count, key))` — the loop's own
count order, `extend_candidates`' order — capped at `extend_cap = 8` per pass per level. A key whose halves have no
row in the operative lower table is skipped and counted (`n_unbuildable`), which is `extend_candidates`' own
`if any(j is None ...)`.

**A candidate is a KEY, so it is a GROUP of rows.** `ClassMiner.build` emits, for one at-support key, the capped
cross product of the lower rows of its `s` half-classes — up to `quot_spell_cap ** s = 16` rows. Those rows are
interchangeable by construction (the same class pair), and the brief's object is "rows of the live build whose flat
key has been admitted", so the admission unit is the key and the trial fires `base ∪ {all of the key's rows}`. This
is the one place the in-loop walk is not literally `gated_walk`'s one-row-per-candidate walk, and it is why
`readgate.gated_walk` gains a default-off `groups` kwarg (§4).

**The trial.** `base ∪ {candidate}` is made into a level-`lv` table over `operative(lv-1)`, wrapped as a macro at
node `(era_node * s**(era_level-1)) // s**(lv-1)`, and fired on a fresh gate pool of `n_aud = 192` instances of the
era's own damage cell (`context_instances(rules, era_ctx(era), ...)`, the loop's own draw). **One fire per trial**,
read by everything: the world's error (`1 - grade().mean()`), the DP's per-instance winning entry, and the
projection's per-instance level. An empty base fires no move and grades the pool as it stands, which is `census`'s
own empty case and `readgate._fire`'s.

**The gate**, `sos_decide`, a transcription of `readgate.decide` restricted to the three values this round runs:

```
world :  admit iff e_trial <= e_cur + extend_tol            (extend_tol = 0.0)      pp4's rule
read  :  admit iff mean(p_trial) - mean(p_cur) >= -0.0      pooled, strict          pp5's `read`
none  :  admit                                              the walk's own floor
```

`p` is `VoProjBank.predict(x_fired, roots, blk0=node*span, span=span)[0]` — the arm's own live readout, the same
object in the grader's seat, read at the candidate cell. pp5's **margin** form is not run: it was worse than the
strict form at both budgets, and its **paired** form is the pooled form at threshold 0 by identity (NOTES §9.4).

**What is recorded on every trial whatever the gate decides**: the world's error and its decision `a_w`, the gate's
decision `a_g`, the read's difference, the DP's entry histogram, and whether the readout was fit at all. So the
per-candidate confusion against the world gate exists on all three arms, as in pp5.

**Billing.** `counts["ground"] += n_auditions * n_aud` **on the `world` arm only**. On `read` and `none` the world's
error is an instrument, exactly as `sb_yield_true` and the probe channel's world verdict are; `n_world_billed` is
asserted to be 0 there. The arms are clock-yoked, so a differing bill cannot move the era schedule; it moves
`g_per_solve` and the meter's instruments, and that is said out loud rather than hidden.

**Instrument at the end of each pass, unpriced on every arm**: the world's audition error of the *admitted* table on
a **test pool of 256** instances of the same cell, drawn on a disjoint seed family and never admitted on. This is
pp5's test family and it is the quality column the reduction turns on.

**After a pass that changed any admission set**: `_rebuild_ms()`, the donor's own re-materialisation of the macro
moves, so the executor sees the new operative table from the same cycle. `len(ms)`, every slot id and the priced
width are untouched by construction.

---

## §3 What is NOT changed

The plant, the shaping head, the readout, the critic, the chooser, the probe channel, the miner, the quotient, the
merge op, the commit policy, the recert, the yoke, the era ladder, the budget, the width, and every pool the donor
draws. The three arms are `sb_sv_yk` with one knob added. `census_extend`'s block (g3) is untouched and unreachable
(no arm carries `extend`).

---

## §4 One defect in the donor this round had to route around, and it is a finding

**`extend_candidates` is structurally inert under a quotient.** It slices the miner's key as a flat level-1 tuple —
`k[i*half:(i+1)*half]` with `half = span // s` — which is `MC.Miner`'s key. `ClassMiner`'s key is a tuple of `s`
*representative tuples*, so the slice is a tuple-of-tuples, `lut.get` misses on every key, and the function returns
`[]` at every level. Verified directly: at L2 with two keys at support it returns `[]`; at L3 likewise.

No banked result is wrong — the only two arms that ever carried `extend` (`census_extend`, `outer_yield_m4x`) carry
no `quotient`/`merge_mode`, so their miners are `MC.Miner` and the function is correct there. But the brief's "
`extend_candidates` already produces exactly that list" does not hold for the arms of record, which run
`merge_mode: "ledger"`. **So this node produces the list itself**, in the miner's own currency:

- `sos_key_of(flat_row, level, s, quot)` — the key `ClassMiner.observe` *would* assign this row, computed from the
  row's own flat level-1 expansion: `tuple(quot.id_of(flat[i*h:(i+1)*h], level-1) for i in range(s))`, `h = span//s`;
  with `quot is None` it is the flat tuple itself, which is `MC.Miner`'s key. Gate **A-8** asserts this against the
  real `ClassMiner.observe` on the real `LearnedQuotient`, including under a merge.
- `sos_candidates(miner, lower, support, decided, quot, s)` — the at-support keys not in `decided`, in
  `(-count, key)` order, each with the rows `miner.build` would emit under it (obtained by building once and
  grouping the rows by `sos_key_of`). Gate **A-7** asserts that the union of all groups is exactly the live build's
  row set, and that filtering the live build by "all keys admitted" returns the live build unchanged.

The admission set is stored as the **flat** key (the `s` representatives concatenated), which is a faithful encoding
(the decomposition into `s` halves of equal width is unique) and is JSON-serialisable for the arm file.

---

## §5 The four cheap things the offline rounds asked for

1. **`vo_wm_every = 5`** — the world-model diagnostics at a fifth of the cadence. *Measured on the banked
   `sb_sv_yk`: the whole block costs 12.8 s of 8096 s (mean 64 ms/cycle), so this saves ~10 s, not a fifth of the
   run.* Done because the brief asks and because it shrinks the arm file; reported with its true size so the lineage
   stops carrying "the diagnostics are expensive" as a belief.
2. **`vo_dump_era`** — `vo_heads_e{k}.pt` at every era boundary as well as `vo_heads.pt` at the end, so the next
   offline round can read candidates at each level's own era.
3. **`vo_dump_refit`** — the end-of-arm readout dumped **twice**: `proj` as it stands (pp1's one-update skew) and
   `proj_final` after one more `VoProjBank.train()` on the final core. Both are in `vo_heads.pt` with
   `proj_skew_resolved: true`, so the next round reads the one it wants instead of carrying a 0.0005–0.0066 AUC
   ambiguity.
4. **The test-pool error of the admitted table at the end of every pass** (§2), n 256, disjoint, unpriced.

---

## §6 The gates

Every gate below is shown to FAIL on a deliberate perturbation (`sostenuto_gates`'s `falsify` block) before it is
reported. Walk gates are falsified on a **designed** error sequence, not on whatever pool is to hand — pp5 §9.3's
lesson, that a pool where the base is already the minimum makes three different update rules coincide.

| gate | where | claim | falsified by |
|---|---|---|---|
| **G-F** | `fidelity_smoke` | with every `# [sostenuto]` knob off this file replays `soundboard.py` bit for bit, against a donor self-replay control | the admission op forced on |
| **A-1a** | `sostenuto_gates` | `sos_walk` reproduces pp4's HAND-COMPUTED sequence: base 0.50 then 0.40 0.45 0.40 0.50 0.10 0.11 at tol 0 gives admissions `[0,2,4]`, 3 rejected, 7 auditions, final 0.10 | `best_e` pinned at the base (then admits everything); `<` for `<=` (then a tie is rejected) |
| **A-1b** | `sostenuto_gates` | `sos_walk` == `readgate.gated_walk` on random stats, all three gates, groups of 1 and of 3 | the group appended one row at a time |
| **A-1c** | `preflight` / in-loop | ONE REAL PASS: the in-loop walk's admissions, rejections, audition count and final error are IDENTICAL to `readgate.gated_walk(gate=<the arm's>, groups=True)` re-run in-container on the same base, candidate groups, order and pool | the order re-sorted |
| **A-2** | `sostenuto_gates_gpu` / in-loop | the in-loop read over a pass's fired configurations equals `preplay.pj_predict` ELEMENTWISE (max\|Δ\| = 0) on the same core and weights | the mask off; `blk0` off by one |
| **A-3** | in-loop | with `gate="read"`, the admissions reproduce `gated_walk(gate="read", groups=True)` on the same fires | the level's sign flipped |
| **A-4** (pp5's P-2) | `sostenuto_gates` | THE PLUMBING IDENTITY — handed the world's per-instance success as its level, the read gate reproduces the world gate exactly | the sign not flipped |
| **A-5** (pp5's P-3) | `sostenuto_gates` | a constant level admits everything | a strict `>` on the level |
| **A-6** (pp5's P-4) | in-loop, every pass | the gate, test and PRACTICE pools are pairwise disjoint (0 shared instances) | the gate pool drawn on the test family's seed |
| **A-7** | `sostenuto_gates` | the admission set governs what is served: filtering by all keys returns the live build unchanged, by none returns empty, and a served row's key is always admitted | the filter keyed on the child row instead of the flat key |
| **A-8** | `sostenuto_gates` | `sos_key_of` == `ClassMiner.observe`'s keying through the REAL `LearnedQuotient`, including under a merge | the half width taken as `span` |
| **A-9** | in-loop / arm file | `read` and `none` bill ZERO world reads for their admission decisions; `world` bills exactly `n_auditions × n_aud` | the bill added on every arm |
| **A-1r … A-4r** | `sostenuto_gates_run` | run-level forms off the arm file: the walk fired; every trial carries both decisions and the confusion is non-vacuous; the served rows match the admission set at every logged pass; `n_commit_cancelled_empty == 0` | an arm that logged no pass at all passes the gate (the `soundboard` §7.1 shape) |
| **Y-1** | `preflight_gates` | the yoke replay is exact with **0 cancelled** | the donor's |
| **S-1…S-6b, M-*, P-*, V-*** | the donor's | unchanged, and run on the three new arms | the donor's |

---

## §7 The arms, and the cost

| arm | `admit` | otherwise | yoke |
|---|---|---|---|
| `st_gw_yk` | `world` | `sb_sv_yk` exactly | `vo_s3:voi3_dp` (seed 0) |
| `st_gr_yk` | `read` | `sb_sv_yk` exactly | `vo_s3:voi3_dp` |
| `st_gn_yk` | `none` | `sb_sv_yk` exactly | `vo_s3:voi3_dp` |
| `st_pf_gw` / `st_pf_gr` / `st_pf_gn` | as above | preflight twins, governance on, yoked to `voi3b_pf_src` | |

The banked `sb_sv_yk` is the **no-walk reference** and is not re-run; it is read at reduction time with `--bank`.

**Cost.** `sb_sv_yk` ran 201 cycles in 8096 s (2.25 h) at seed 0, peak RSS 7073 MiB against a 12 GiB request. The op
adds, per arm: ~100 base fires + ~150–250 candidate fires + ~100 test fires of 192/256 instances, each one
`MC.apply_any` + `grade` + one trunk forward — small beside the recert's two full beam searches every five cycles.
Budget 2.3 GPU-h per arm, three arms in three containers (the donor's `sweep` fan-out), ~7 GPU-h plus gates, G-F and
a three-twin preflight. `memory` stays at 12288 and is re-reported from the smoke.

**The read gate's own warm-up, found while designing and reported before the run.** `VoProjBank` needs
`vo_om_min = 512` filed rows before it fits; on the banked `sb_sv_yk` its first refit is at **cycle 50**, and the L2
admission walk runs from **cycle 5**. Until the readout is fit, `predict` returns 0.5 for every row, the pooled
difference is exactly 0, and `read` admits — so **L2's admission set is decided ungated on the `read` arm by
construction**, and the `read` arm is the `none` arm at L2 up to c50. This is not a defect and is not worked around:
delaying the walk until the readout is fit would make the arms gate at different cycles, which is the confound the
round exists to avoid. Trials decided with an unfit readout are counted (`n_unfit`) and the reduction reports L2
separately.

---

## §7 Defects, in order, each beside its correction

**§7.1 A falsification that could not fail, at L2.** The A-7 perturbation — "the filter keyed on
the child row instead of the flat key" — came back GREEN on the first run of the harness. The
reason is the level, not the rule: at L2 the lower table is `MC.base_table(v)`, whose flat row `i`
IS `(i,)`, so a child index and its flat key coincide and keying the filter on either gives the
same answer. Corrected by pinning the perturbation at L3 over a real L2 table, where the key is a
width-4 flat tuple and the child row is a pair of indices. Harness **8/8**. This is pp5 defect 9's
shape in a different organ: a gate has to be falsified where the two rules can disagree.

**§7.2 The preflight could close the plumbing and could not exercise a refusal, for two reasons
named before the paid run.** `preflight` sets `extend_tol = 1.0`, so the world's decision is
always admit; and at preflight sizes the readout is degenerate on all 21 refits (`fit_base = 0.0`
— the substrate solves nothing, `soundboard` §7.4), so its level is a constant and gate A-5
applies exactly. All three preflight twins were therefore bit-identical, which is a real inertness
statement and not a rejection test. Not worked around: the rule's falsification lives in A-1a/A-1b
on designed sequences, and the in-loop gate was strengthened afterwards so that A-1c re-walks
**every** pass through `readgate.gated_walk` out of the fire cache, and the counterfactual block
runs up to `SOS_REPRO_MAX` times and stops at the first pass that SEPARATES the gates. On the paid
tag that pass was found on all three arms at c10 L2, where the world gate admits 2 of 5 and the
read and none gates admit 5 of 5.

**§7.3 The reduction pooled over eras and read the cell moving up as the table decaying.** The
gate and test pools are drawn on the ERA'S OWN CELL, so a level-2 macro auditioned during era 4 is
being asked to repair an L4n3 cell — its error rises to 0.8-0.97 on every arm, including the one
that admits everything. Section [B] is now reported inside each level's own era, the paired
within-pass `admitted − live` is the comparison of record (same plant, same pool, same cycle), and
the figures shade the level's own era.

**§7.4 The reducer shadowed its own `tag` parameter, twice, and silently emptied section [H].**
`for tag, sel in (("own", ...), ("all", ...))` in [B] and `tag = f"{lv}"` in [D] rebound the
function's `tag`, so `figures/<tag>/summary.json` resolved to `figures/fit/summary.json`, the
per-container elapsed and peak RSS came back as `—`, and the wall-clock section said nothing
rather than failing. Corrected to `scope` and `ltag`. Recorded because the failure mode is the one
this lineage keeps meeting: an instrument that goes quiet reads as "no data", not as "bug".

**§7.5 The reducer shadowed `ref` as well, and it broke the figures rather than the text.** The
coverage trace's local `refused` list was first called `ref`, which is the banked reference arm;
`make_figures` then did `ref["log"]["e"]` on a list of strings. Same class as §7.4, caught on the
first seed-2 run, and it had also left a stray `figures/all/` from the earlier `tag` shadowing
(removed). The lesson the two together carry: this reducer's section blocks share one namespace,
so a section-local name must not be a word the function already owns.

**§7.6 A CUDA fault killed one seed-2 arm, and it was the arm under test.** The first seed-2 app
(`ap-ylKRwrDaYeSuEe3Pxp16Wg`) ran `st_gw_yk` (7721 s, 2.14 GPU-h) and `st_gn_yk` (8203 s, 2.28
GPU-h) to completion and lost `st_gr_yk` to `RuntimeError: CUDA error: unknown error`. Not a code
fault: the identical binary ran all three arms to completion at seed 0 and the other two arms of
the same tag on the same day. `sweep`'s own assertion surfaced it and the merge kept the two good
arms, so the relaunch cost one arm rather than three (`ap-HfAKLrIiKnT6GYEgNAl4yR`, `st_gr_yk`
alone, 8454 s, 2.35 GPU-h). A one-arm `sweep` rewrites the tag's `summary.json` with only that
arm, so `sweep --merge-only` was run afterwards to restore the three-arm merge
(`ap-wQQw73Tem9tBCl3koUInAG`, 11 s on CPU) — and it cannot recover `elapsed_s_per_container`,
which comes from the live call's return, so seed 2's container times in [H] are estimated as
`s/cycle x cycles` and marked `*`. Separately, this session's container restarted while parked,
which killed the local waiter and the `modal run --detach` client; the remote apps were
unaffected, which is the exact failure `results/wait_app.sh`'s header exists for.

---

## §8 Reproduction

```bash
cd experiments/            # MODAL_PROFILE=chromatic
B=rhm/practice/voicing/sotto_voce/aliquot/sostenuto
modal run $B/sostenuto.py::sostenuto_gates          # A-1a/b, A-4, A-5, A-7, A-8 + falsifications
modal run $B/sostenuto.py::sostenuto_gates_gpu      # A-2 + the donor's P-/S- blocks on an L4
python3 $B/launch_detached.py --fn fidelity_smoke --tag st_gf1                 # G-F
python3 $B/launch_detached.py --fn preflight --outdir-tag st_pf1 \
    --arms "voi3b_pf_src,st_pf_gw,st_pf_gr,st_pf_gn"
modal run $B/sostenuto.py::preflight_gates --outdir-tag st_pf1
python3 $B/launch_detached.py --fn sweep --out-tag st_s1 --seed 0
python3 $B/launch_detached.py --fn sweep --out-tag st_s2 --seed 2
python3 $B/fetch_compact.py --tag st_s1 --fetch --replace
python3 $B/reduce_sostenuto.py --tag st_s1 --bank-tag sb_s1 --bank-arm sb_sv_yk
python3 $B/reduce_sostenuto.py --tag st_s2 --bank-tag sb_s2 --bank-arm sb_sv_yk
python3 $B/mk_seedtable.py                       # -> figures/st_seedtable.txt
# ---- round 2 (§9) ----
modal run $B/sostenuto.py::sostenuto_gates          # + A-12..A-15, A-4y, A-5y, A-5d, 11/11 RED
python3 $B/launch_detached.py --fn fidelity_smoke --tag st2_gf1                # G-F
python3 $B/launch_detached.py --fn preflight --outdir-tag st2_pf1 \
    --arms "voi3b_pf_src,st_pf_gy,st_pf_gd"
modal run $B/sostenuto.py::preflight_gates --outdir-tag st2_pf1               # + E-Y / E-D
python3 $B/launch_detached.py --fn sweep --out-tag st2_s1 --seed 0 --arms "st_gy_yk,st_gd_yk"
python3 $B/fetch_compact.py --tag st2_s1 --fetch --replace
python3 $B/reduce_sostenuto.py --tag st2_s1 --with-tag st_s1 --bank-tag sb_s1 --bank-arm sb_sv_yk
python3 $B/mk_seedtable.py --round 2              # -> figures/st2_seedtable.txt
```

---

## §9 Round 2 (2026-09-22): two gates in the next level's currency

The brief is `SPEC.md`'s round-2 section. The op is unchanged: cadence, cap, pool, count order,
decided once, empty base and the yoke are all round 1's (§1, §2). Two new values of the one knob
`admit`, one new margin, both new objects logged on every trial of both new arms, one exported
pass per weight state, and new gates. Two arms, seed 0, yoked to `vo_s3:voi3_dp`: **`st_gy_yk`**
(`admit: "yield"`) and **`st_gd_yk`** (`admit: "demand"`). Round 1's three seed-0 arms and the
banked `sb_sv_yk` are the comparators and are not re-run. Every addition is marked
`# [sostenuto r2]`.

### §9.1 Which miner is "the next level's": the observation panel, not the committable miner

**The fact the brief did not have.** The committable miners are fed only up to `era_level + 1`
(plus L5 under `ungate_l5`; the donor's own rule, so that era k cannot hand era k+1 a finished
vocabulary). A level-l key reaches support, and is offered, in era l-1, which is exactly when the
committable level-(l+1) miner has not started counting. On round 1's `st_gn_yk` (seed 0) the
committable L3 miner holds **0** at-support keys through c60 while 13 L2 keys are offered in
era 1, and the committable L4 miner holds 0 through c110 while 14 L3 keys are offered in era 2.
Reading the committable miner, the yield share would be identically 0 (a tie, so the gate admits
everything), and the demand count would refuse every key offered in its level's first era.

**What the arms read instead.** The observation panel (`obs_miners`, on in every arm of record)
counts the same reader parse of the same solved configurations at every level in every era. It is
the learner's own: table-free counts on its own reader's parse through its own `LearnedQuotient`,
re-keyed with the committable miners on every taken merge. It is also what the lineage already
reads for its one-level-up merge licence (`obs_miners.get(ml + 1) or miners.get(ml + 1)`), for
this same reason. So both new gates read `obs_miners[l + 1]`, falling back to the committable
miner. At L5 that is the panel's L6 (whole-string) miner; the committable set has no L6.

Panel at-support key counts on round 1's `st_gn_yk`, seed 0, as L3 / L4 / L5 / L6:

| cycle | 5 | 20 | 60 | 110 | 180 | 201 |
|---|---|---|---|---|---|---|
| panel | 2/0/0/0 | 18/0/0/0 | 49/14/0/0 | 38/64/0/0 | 38/104/6/0 | 41/102/13/2 |

### §9.2 `yield`: what the object is, and what the gate does before the readout's first fit

Per fired configuration, `y = share × level`. `share` is `sb_yield_label(parse, obs_miners[l + 1],
mine_support, solved=None)`: the fraction of the configuration's level-(l+1) spans whose key is at
support, from the learner's own reader (`_sb_reader_parse`, RNG-sandboxed). `level` is round 1's
read: the arm's live projection at the candidate's own cell. `solved=None` because the "zero on an
unsolved piece" of the label is what the level stands in for, endogenously. The gate admits iff
`mean(y_trial) - mean(y_cur) >= -adm_delta`. That is pp5's pooled read rule with `y` as the level,
and gate A-12 asserts exactly that against `readgate.gated_walk`.

**Before the first fit (c < 50), the gate acts on the share alone.** `VoProjBank.predict` answers
0.5 for every row before `vo_om_min` filed rows, so `y = 0.5 × share` and the rule compares
shares. I let it act, for two reasons. The share needs no readout. And L2 in era 1 is exactly
where round 1's world gate refused the halves L3 later needed, so suppressing the gate there would
make the arm the `none` arm at L2 by my choice rather than by construction. The trials are still
marked `unfit`, and the reduction reports them apart, as in round 1.

**Where the gate is silent by construction.** If the next-level miner holds no key at support,
the share is 0 on every fire, the difference is 0, and the strict rule admits. On round 1's trace
that covers L4 keys before about c180 and L5 keys before about c190. The share is taken over all
of the configuration's next-level spans (the label's own form). Between trial and base, though,
only the span containing the candidate's node can change, so the difference is that one span's
at-support status, weighted and divided by the number of spans.

### §9.3 `demand`: the brief's literal form cannot answer its question here, and what runs instead

**Fact 1: under the panel, the literal form is the `none` arm.** Each cycle the panel observes the
level-(l+1) span containing the era's cell, and the level-l miner observes the level-l span
containing the same cell, on the same parsed rows (both use `node = (era_node · s^(era_level-1)) //
span`). The level-l span is always a half of the level-(l+1) span observed with it. So every
observation that counted toward key k at level l also counted toward a level-(l+1) key that has
k's class as a half, and the raw sum over all next-level keys is at least count_l(k), which is at
least support (3), for every offered key. That holds up to the spelling cap at L3+ and to merges.
At threshold 1, then, the literal form admits everything. It would be bit-identical to `st_gn_yk`
and would cost 2.3 GPU-h to reproduce it. **Under the committable miner**, the literal form
instead refuses every key offered in its level's first era (§9.1). Neither can separate one
candidate from another.

**What runs: the same sum over next-level keys AT SUPPORT** (`adm_demand_floor = -1`, which
resolves to `mine_support`), at threshold `adm_demand_min = 1`. The gate admits a key iff the
candidate is a half of at least one next-level key the learner holds at support. That is a key
the next level's build can realise only if this candidate is admitted: `ClassMiner.build` emits
only at-support keys, and `sos_filter` then serves only admitted halves. So this is "raises
next-level minability" in the build's own currency. It is the currency the yield share reads and
the one the coverage trace counts. And it is not tautological: count_l(k) ≥ 3 does not put any
single (k, sibling) key at 3.

**Fact 2: the top of the ladder is empty for most of the run, so the gate is SILENT there.** The
panel holds no at-support L5 key before about c180 and no L6 key before about c190. Every
candidate's at-support sum is then 0. Refusing them all would report that the next level is not
minable yet, which is a fact about the level and not about the candidate, and decide-once would
make it permanent. So when the next-level miner holds no key at the floor, the demand gate is
silent: it admits and counts the key (`n_silent`). This is the yield gate's automatic tie (§9.2)
and round 1's "no reader, no verdict", made explicit.

**Both forms are on the record for every offered key on both new arms**: `value` (at the floor),
`raw` (the literal sum), `n_keys`, `n_keys_raw`, `n_next_at_floor` and `silent`. The literal
form's verdict and any threshold can therefore be read afterwards, and Fact 1 is checked on the
paid record (raw ≥ the key's own count).

**The candidate's classes** are `quot.id_of(flat_row, l)` of its rows, which are the ids the
next-level `ClassMiner` keys a half by. A next-level key is counted once however many of its halves
match.

**"No fire"** holds for the decision: demand is read off the counts before the walk. The trial
fires still run, unpriced, exactly as on round 1's `none` arm, so the per-candidate confusion
against the world and the test-pool column exist on this arm too. The bill is zero (A-9r).

### §9.4 The margin, and what every trial now logs

`adm_delta` (default 0.0): admit iff the difference is at least `-adm_delta`, on `read`, `yield`
and `demand` (on demand the difference is `value - threshold`). It is not run as an arm. At 0.0
every round-1 decision is unchanged (`d >= -0.0`). On the two new arms every step logs
`d_yield`, `share_cur/trial`, `y_cur/trial`, `d_read`, `d_world` and the full `demand` record,
whichever gate decided.

### §9.5 The gates, round 2

Each one was run red on its perturbation before being reported (`sostenuto_gates`: round 1's 8/8
plus **11/11** new).

| gate | where | claim | falsified by |
|---|---|---|---|
| A-0 | CPU | `st_gy_yk`/`st_gd_yk` are `sb_sv_yk` plus the one `admit` key | — |
| **A-12** | CPU | the yield walk IS `readgate.gated_walk`'s read rule on `y`, at margin 0 and 0.02 (24 random cases; yield separates from read on 23) | reading the level instead of `y`; the margin's sign flipped |
| **A-4y** | CPU + in-loop | handed a constant share of 1, the yield gate reproduces the READ gate | — (identity) |
| **A-5y** | CPU + in-loop | a constant object admits everything | a strict `>` |
| **A-13** | CPU | the demand count equals a hand count on a real `LearnedQuotient`, including a merge read through the map | first half only; a key counted once per matching half; the floor dropped; the class read at the level below |
| **A-14** | CPU | the demand walk IS readgate's read rule on the encoded level (`-#undemanded keys served`); silence admits | `>` for `>=` at value == threshold; silence read as a zero demand |
| **A-5d** | CPU + in-loop | a zero threshold admits everything | — |
| **A-15** | CPU | the yield object IS `sb_yield_label(solved=None) × level` | the next-level miner taken at the candidate's level; the weight dropped |
| **A-1c** | in-loop, every pass | the arm's own walk re-derived through `readgate.gated_walk` via `sos_readgate_spec` (yield = read on `y`; demand = read on the encoded level) | round 1's |
| **A-1c sep.** | in-loop, run-level | a pass on which the ARM'S OWN gate admits a different set from another gate (`strong_own`) was found; the counterfactual carries all five gates | an arm without one fails `sos_gate_run` |
| **E-Y** | export, off-container | every fired configuration's share IS `sb_yield_label` rebuilt from the exported next-level counts and class map; `y == share × level`; the walk re-derived through readgate from the exported fires | on the same export: the miner at the candidate's level; the parse shifted one block (reported RED or BLIND per pass) |
| **E-D** | export, off-container | classes are `quot.id_of(flat, l)`; demand IS a plain double-loop hand count over the exported keys; decisions re-derived | on the same export: first half only; floor dropped (RED or BLIND) |
| A-9r, A-2r | run-level | zero world reads billed; every step carries both decisions and the demand | round 1's |

The exports are `adm_export_unfit.json.gz` (the first non-silent pass with ≥ 2 candidates while the
weight is the constant 0.5) and `adm_export_fit.json.gz` (the first such pass after the first fit).
Each holds the next-level miner's counts, the class map at that level, every fire's parse, level,
share and `y`, and every candidate's rows, classes and demand, so E-Y and E-D run in a separate
process against the donor's own functions.

### §9.6 The preflight twins seed the panel

`st_pf_gy`/`st_pf_gd` carry `preflight_seed_panel`. It seeds the observation panel the way
`preflight_seed_miner` already seeds the committable miners: half the true table, the rest at the
top-up cycle. At preflight sizes the substrate solves almost nothing, so an unseeded panel holds
no key at support, both new gates are silent on every pass, and the count, the share and the
export would never run. The seed hands over the answer, which is why it is per-arm and on those
two twins only.

### §9.7 What each arm can show, written down before the launch

- **Both arms are bit-identical to round 1's `st_gn_yk` until their first refusal.** They share
  the stream key; the fires are an argmax DP and a grade, so they draw no RNG; and the reader parse
  is RNG-sandboxed. The reduction checks this on the per-cycle series, which is a free in-run
  fidelity gate.
- `yield` can separate from `none` from c5 at L2 (the panel holds 2 at-support L3 keys at c5) and
  at L3 in era 2. It is silent at L4/L5 until the top of the ladder fills in (§9.2).
- `demand` can separate from `none` wherever the next level holds an at-support key. It cannot
  refuse an L4 key before about c180, and round 1's read arm refused L4 keys earlier than that.
- A refusal is permanent (decide-once), and both gates read the next level at decision time. A key
  refused because its next-level superkeys were not yet at support stays refused when they get
  there. The coverage trace (iii) reads that directly.

### §9.8 Cost

Two arms at about 2.4 GPU-h each on L4 (round 1: 2.33–2.45). The new instruments add one reader
parse and one `sb_yield_label` per trial fire, and one count over the next level's keys per
offered key. Round 1's whole walk was about 0.1% of an arm (§7 [H] of the reduction). `memory`
drops from 12288 to 10240, because the six round-1 containers peaked at 6966–7080 MiB. There is no
resume path, and the two arms are identical to `st_gn_yk` until their first refusal, but there is
no checkpoint to fork them from, so the only honest shortening is not re-running what is banked.

### §9.9 Defects and facts from the round-2 runs, in order

**§9.9.1 My separation gate could not be satisfied at preflight, and the reason was the search
budget, not the gate.** `st2_pf1`'s run-level gate required a pass on which the arm's own gate
admits a different set from another gate (`strong_own`). The counterfactual block is bounded at
`SOS_REPRO_MAX = 8` full runs. At preflight scale nothing separates in the first cycles: the
world admits everything at `extend_tol` 1.0, and the readout is constant. All 8 runs were spent by
c6, and the demand gate's first refusal came at c8. Corrected in two parts. On a round-2 arm the
block also runs on the first pass where the arm's own gate refuses anything; that pass separates
from `none` by construction. And the run-level gate now requires a separating pass iff the arm
refused anything. An arm that refused nothing IS the `none` arm, and that is reported, not passed
as a separation. `st2_pf2` is green: `st_pf_gd`'s separating pass was re-walked at c8 L4, and
`st_pf_gy` refused nothing at preflight scale. G-F was re-run on the corrected binary
(`st2_gf2`, 0.000e+00).

**§9.9.2 The residual risk §1.2 named happened, on the demand arm.** On `st_gd_yk` (seed 0) the
L5 commit at c186 was CANCELLED: `miners[5].build(operative(4))` was empty. The demand gate had
refused 5 of 6 L4 keys offered in era 3, each at demand 0 against a non-empty L5 panel, leaving
one admitted L4 key, and no at-support L5 key had both halves in it. The yoke therefore did NOT
replay exactly on this arm: its commits are [48, 100, 151], and L5 was never adopted. Y-1 /
A-3r is red on `st_gd_yk`, and its eras 4–5 are not at a matched clock with the other four arms.
The arm ran to completion. Every other gate on it is green: A-1c on all 26 level-passes, a
separating pass at c5 L2, A-4y/A-5y/A-5d, 0 world reads, 0 pool overlap, and E-Y/E-D on both
exports. This is not worked around; it is a result of the gate.

**§9.9.3 Fact 1 (§9.3) holds on the paid record.** On every offered key at L2–L4, on both arms
(17/17, 18/18, 9/9 and 20/20, 12/12, 9/9), the literal raw demand is at least the key's own
count. So the brief's literal form would have admitted every one of them. At L5 it holds on 2
of 4. The spelling cap bites there, because a candidate's 16 rows cover few of the 16-token
spellings its key was observed under.

**§9.9.4 The seed-2 decision.** The SPEC's condition (meaningful, and small enough to be
seed-sensitive) is judged met for `st_gy_yk` and not for `st_gd_yk`.
- Yield's coverage signature is a small count: 3 of the none arm's 19 L3 keys blocked, against
  the world gate's 7 and the read gate's 0. Its deep-era cost against the none arm (+0.09 / +0.12
  in eras 4 / 5) is the quantity whose sign flipped across seeds for round 1's read and none arms.
- Demand's effects are structural and far outside round 1's seed-to-seed spread: 17 of 19 L3 keys
  blocked, the L5 commit cancelled, and +0.28 / +0.27 against the none arm.

So `st2_s2` runs `st_gy_yk` alone, yoked to `vo_s3d2:voi3_dp`, against round 1's banked seed-2
arms and `sb_s2:sb_sv_yk`.

**§9.9.5 At seed 2 the same residual risk hit the YIELD arm, one level lower.** On `st2_s2`'s
`st_gy_yk` the L4 commit at c157 was CANCELLED. The yield gate had refused 8 of the 13 L2 keys
offered in era 1, all before the readout's first fit, so on the share alone. It refused one more
L2 key later, and 2 L3 keys. That left 4 admitted L3 keys, and `miners[4].build(operative(3))` was
empty from then on. Its commits are [59, 77]; L4 and L5 were never adopted and never walked (their
live builds were empty all run, 529 unbuildable offers). Y-1 / A-3r is red on it. Every other
gate is green: separating pass at c5 L2, A-1c on all 15 level-passes, A-4y/A-5y/A-5d, 0 world
reads, 0 pool overlap, E-Y/E-D on both exports with all four on-export perturbations RED. The
residual risk §1.2 named before round 1 therefore occurred on two of the three round-2 containers.
**The silence at the top is on the record at seed 2 only through the panel**, because no L4/L5 key
was ever offered there: the panel held **0** at-support L5 keys and **0** L6 keys at every logged
cycle through c201. Any L4 or L5 trial would therefore have had share 0 on every fire, and the
yield gate would have been silent by construction.
