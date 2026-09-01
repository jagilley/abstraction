# FILES — `antiphon/provenance` (shape P: the provenance primitive, offline pass)

Machinery record for the shape. **No writeup lives here** — the numbers are to be discussed with
Jasper before anything is interpreted, per repo norms. This file records what was built, what it
measures, what it verified, and what it cannot resolve.

**Up**: node spec [`../SPEC.md`](../SPEC.md) §2 and first shape **P** ("Log efference copies of
posed questions; tag arriving data self/other by match; ask whether provenance-gated credit
changes what forms trust").
**Donor** (untouched): [`../../woodshed/`](../../woodshed/FILES.md) — `reduce_trust.py` is
**imported**, not re-implemented, for every trust statistic in §3–§4.
**Reads, read-only**: the already-fetched mirrors under
`../../{assay,woodshed,conductor,maestro,crescendo,audiation,intonation,caesura,tacet,ostinato}/figures/<tag>/`.

**Status: CPU-only, no Modal, no GPU, no network, nothing launched.** The offline pass was run
first per `ostinato`'s standing lesson (size the premise against the banked logs before
building), and its verdict was that the sketched GPU intervention does not earn its cost in the
form the SPEC sketches it. See "What the offline pass settled" below.

## Code files

| file | purpose |
|---|---|
| `prov_tag.py` | The whole shape. Builds the efference-copy log, the index→key map for every committed macro table in the banked record, the self/other tag in four variants (holding × use, live × frozen-at-arrival), the gates that verify the match rule against independent artifacts, the two-clock comparison against `reduce_trust`'s trust statistics, and the floor/handle arithmetic. `--figures` writes the three PNGs. |

```bash
cd experiments/                                   # no MODAL_PROFILE needed
python3 rhm/practice/antiphon/provenance/prov_tag.py --figures
# optional: restrict to some tags
python3 rhm/practice/antiphon/provenance/prov_tag.py --tags as_s0,wd_s1
```

Outputs: `figures/reduction.txt` (the six sections), `figures/cross_rows.json` (the per-(tag,
arm, level) row set, for downstream use), `figures/f1_exafference_decay.png`,
`figures/f2_two_clocks.png`, `figures/f3_credit_provenance.png`.

## The primitive, as it resolves on this substrate

**The efference copy.** The agent's outgoing act is a rollout. When one *solves*, its own
generator parses the corrected configuration into level-1 features and `macros.Miner.observe`
counts them; `Miner.state()["keys_at_support"]` (the identity instrument installed since
`tall/`, `mine_support = 3`) is the sorted list of level-ℓ flat keys the agent has itself
emitted at least three times. So

```
E_l(c) = { level-l flat keys the agent's OWN solved rollouts have produced
           at support >= mine_support, by cycle c }
```

is the efference-copy log: one entry per spelling the agent produced in its own hand. It is
native (the agent's own parse, no oracle), cumulative and monotone (gate G0, 0 violations).

**The arriving datum, and the match.** The arriving data are the entries of the committed macro
table — the vocabulary the executor DP serves from. Entry *i* at level ℓ carries flat key `k_i`:

```
self(i, c)   <=>   k_i in E_l(c)        reafference — I have emitted this spelling myself
other(i, c)  <=>   k_i not in E_l(c)    exafference — an arriving datum matching no outgoing question
```

Two tags: `arr` freezes it at the arrival cycle (what was exafferent when it landed); `live`
re-evaluates every cycle, so **exafference decays** as the agent independently re-derives a
gifted spelling. The live tag is provenance of *re-derivation*, not of origin, and it is the
arity-2 reading: the datum joins the agent's causal history when the agent's own act reproduces
it. Two weightings: by **holding** (over table entries) and by **use** (weighted by
`log["entry"]["hist"][phase][l][i]`, `assay`'s per-execution entry-identity histogram — which
entry the max-sum DP actually served, per cycle).

## The index→key map, and why it exists at all

Everything is recovered from banked bytes; nothing is re-derived on GPU.

| table source | key list |
|---|---|
| `complete`, `junk_dose`, `strip` | the `surgery` event's `flat_after`, verbatim |
| `exact` (surgery returns the DGP table early, no `flat_after`) | `setup.json["true_tables"][l]["flat"]` |
| `given_c1` (a c1 gift leaves **no events at all** — detected by `log["vocab"][0][l] != None`) | same |
| earned, no surgery | **reconstructed** by replaying `Miner.build`: `sorted(E_l(c_commit))` filtered by buildability over the arm's own committed lower table |
| earned + `census`-style extension | the commit-cycle build, then the `extend` events' `admitted` keys appended in event order (extension APPENDS, so index order is commit-order-then-admission-order, not sorted order) |

The committed table is a **frozen snapshot**: the map is rebuilt only when `vocab[l]` moves, and
carried forward otherwise. Rebuilding it every cycle was the first implementation and it is
wrong — it tracks the miner past the freeze and over-counts by 1–4 entries within four cycles of
every commit.

## Gates

| gate | what it asserts | result |
|---|---|---|
| **G0** | `keys_at_support` monotone in cycle, every arm, every level — the efference log only ever grows | **0 violations** |
| **G1** | earned tables: the reconstruction from the efference-copy log reproduces the logged `vocab[l]` size **and** the logged per-cycle `true_mask` element-wise | PASS |
| **G2** | gifted tables: the logged `true_mask` is all-ones with length `= |true_tables[l]|` | PASS |
| **G3** | surgical tables: `flat_after` length `=` `true_mask` length `=` `vocab[l]`, and the truth mask recomputed from the keys equals the logged one | PASS |

**125 / 125 (arm, level) key-map cells PASS, 0 FAIL**, over 15 (node, tag) runs and
65 arms (the `maestro`/`intonation` `ma_s0` tag-name collision is keyed on `(node, tag)`). G1 is the
load-bearing one: it verifies the match rule against an artifact the rule did not produce (the
per-cycle `true_mask` written by the run loop), and it forces `self_frac == 1.000` on every
earned arm, which is the construction check. Without G1 the tag would be an assumption.

## What the offline pass settled

Numbers live in `figures/reduction.txt` §§1–4. The four load-bearing ones:

1. **The tag is anti-correlated with trust across the arrival cross.** Kendall τ_b against π's
   L3 argmax share over `as_s0`'s six arms: `self_frac_end` **−0.29**, `n_self_end` **−0.47**,
   `self_use_end` **−0.43**, `self_use_mean` **−0.43**, `conv` 0.00. `given_c1` — the arm that
   wins the trust bracket (argmax 0.831) — holds the *lowest* self fraction of any arm (0.156)
   and ran **47 cycles at exactly 0.0000 self share of L3 executions**. Provenance-gated credit
   would have withheld credit from precisely that arm.
2. **The reason is structural: the efference log is downstream of use.** An entry becomes self
   only by being served often enough that the agent's own solved rollouts spell it out three
   times, so self-tagging is a near-deterministic consequence of an entry being used — and
   competence *suppresses* re-derivation diversity. The concentration read, new here: the
   **effective number of L3 entries in use (`ess_use`, exp-entropy of the cumulative per-entry
   execution distribution) is 1.94–3.42 in every arm**, whether the arm holds 5 entries
   (`strip`) or 64 (`given_c1`, at 9.2M L3 executions). `given_c1` touches 43 of its 64 entries
   and leans on an effective 3.42.
3. **On the one contrast where an effect exists, no provenance statistic both orders the arms
   as trust does and clears its own handle.** `wd_s1`'s credit > exposure > none is reproduced
   by `self_use_end` (0.9699 / 0.9573 / 0.9234) — but that spread is **0.67×** the in-tag
   unrehearsed-L2 handle and **0.86×** the `wd_s0` null-dose handle. `n_self_end` clears at
   4.00×/8.00×/8.00× but orders credit > *none* > exposure. `self_use_mean` is inverted
   outright (none > exposure > credit). Trust itself clears every handle on the same arms
   (argmax 5.50× / 6.72× / 2.16×).
4. **The two clocks are not the same clock.** Commit-time gifts land already re-derived
   (`exact` L3: 0.905 self-by-use at τ = 8, τ to 90% = 1.2 cycles), i.e. the exafference
   transient is *below the trust instrument's 8-cycle sampling interval*; the c1 gift's is 47.7
   cycles and is a step, not a decay — it fires when the miner's L3 counts cross
   `mine_support`, not when the agent starts using the table.

## Caveats, printed as §5 of the reduction

- The efference log is **downstream of use** (see 2 above): the holding→use concentration column
  is mechanism, not effect.
- **No floor for π mass exists anywhere in the arc** (`woodshed` FILES.md), and this round adds
  that **no floor for any provenance statistic exists either**. The only handles available are
  the `as_s1` stream twins (n = 1 pair per family) and `wd_s1`'s unrehearsed L2 (n = 3); the
  `wd_s0` null-dose trio (n = 3, rehearsal solve rate 0.003) is the most honest of the three and
  it is *larger* than the effect on the one ordering-correct statistic. Every floor-multiple in
  §4 is limited by this and none of them is a measured noise floor.
- `mine_support = 3` is a free parameter of the match rule and is **not sweepable offline**: the
  banked logs carry `keys_at_support` at that one threshold only. §5 prints `n_at_support` at
  1/2/3/5/10 so the log's size-sensitivity is on the record (L3 at `as_s0/exact`: 81 / 52 / 35 /
  21 / 10) — but the membership does not exist on disk at any other threshold. A machinery
  limit, not a measured invariance.
- Single seed everywhere; `wd_s1`'s rehearsal is not compute-matched (~2× practice compute).
- `n_self` divergence in `wd_s1` appears only in the last ~10 cycles (`f3`, middle panel) —
  right-censored, like every other deep-era statistic in this lineage.
- `beam` (practice rollouts) and `probe` (metering rollouts) agree to < 0.01 everywhere; only
  `beam` is carried into §3–§4.

## The join that is missing, and what it would cost

The offline pass had to use the **miner's observation record** as the efference copy, because it
is the only per-key record on disk. That record is of *answers the agent produced*, not
*questions it posed*. The genuine efference copy of a posed question on this substrate is **π's
proposal**: at each beam node π emits an action slot (level, node) — a claim that a macro at that
slot applies here — and the executor DP answers by serving an entry. Both halves are already
computed in the run loop; only the **aggregates** are logged
(`log["probe"]["pi"]["argmax_hist"]` and `entry.hist`), never the per-execution join. With the
join, exafference becomes *executions the DP served from an entry π did not propose* — the
executor answering a question the planner did not ask — which is arity-2 at the right
granularity, is not circular through use, and lives on the cycle grid the trust clock can be
compared against.

The change is small and lives entirely in the recorder: `macro_features_rec` already holds
`best` (the served entry); it needs the proposing slot beside it, and one extra log key. A
logging-only fork of `woodshed.py` marked `# [antiphon_p]`, gated bit-identical at 0.000e+00
against `wd_s1` with the recorder off, then a re-run of `wd_s1`'s four arms ≈ **1.6 GPU-h**.
That is a *measurement* arm, not an intervention: it would establish whether provenance-gated
credit has anything to gate on before any intervention is priced. The intervention itself (a
`reh_mode = "credit_self"` third arm, pair-count matched against `credit` by subsampling) is
~0.5 GPU-h on top. **Neither was launched**; the decision is the node's to make against the
numbers above.

## Volume layout

None. This shape wrote no volume paths and consumed no Modal resources.

---

# P′ — the per-execution provenance join (authorized measurement arm)

Added 2026-09-01 after the offline verdict above was accepted. **P″ (the `credit_self`
intervention arm) is NOT authorized and is not built** — it stays specced at the end of this
file, pending discussion with Jasper.

## Why P′ and not P

The offline pass could only use the **miner's** record as the efference copy: the spellings the
agent's own solved rollouts emitted. That is a record of *answers the agent produced*, not of
*questions it posed*, and it is downstream of use. The genuine efference copy of a posed
question here is **π's proposal**: at each beam node π emits an action slot (level, node) — a
claim that a macro at that slot applies — and the executor DP answers by serving a table
**entry**. Both halves are computed in the run loop; only the aggregates are logged
(`probe.pi.argmax_hist`, `entry.hist`), never the join. P′ logs the join.

This is the **proposal grain** of the same primitive the node root's main lane exercises at the
**question grain** (a posed repair instance from its K=2048 menu, with a posed-vs-landed
delivery ledger). The main lane runs on all-earned tables, where other-provenance does not
exist; P′ runs on the trust/gift harness, where it does. Keys are plain flat keys in both, so
the two ledgers read as one primitive at two grains.

## Code files (added)

| file | purpose |
|---|---|
| `antiphon_p.py` | Forks `../../woodshed/woodshed.py` **verbatim** and adds exactly one thing, marked `# [antiphon_p]` at each of its 40 insertion points: the per-execution provenance join, logging-only. |
| `launch_detached.py` | `../../woodshed/launch_detached.py` retargeted (session-isolated `modal run --detach`, new module path, `--fn antiphon_p_run`). |
| `analyze_pp.py` | `pp_s1`'s gates (twin at the first CONSUMED cycle, first-consumed pair-count match, in-tag one-bit, substrate identity) and its reduction: dose, trust against the unrehearsed-L2 handle, and value against the `anchor`→`given_c1` bracket and the measured stream floor. |
| `analyze_ap.py` | `ap_s0`'s gates, reduction and the circularity check: the five gates below, then the source split, the source x provenance 2x2, and the one-bit contrast against its handle. Imports `prov_tag.py` for the match rule and `../../woodshed/reduce_trust.py` for the trust statistics — neither is re-implemented. |

Modal app `rhm-practice-antiphon-p`; volume path
`rhm-scaling-data:/data/rhm_practice_antiphon_p/<tag>/`; fetched copies under `figures/<tag>/`.

## The log format — `log["ap_join"][c]`

```
{"keys": {"<level>": [[f0, f1, ...], ...]},                # PLAIN FLAT KEYS, index-aligned
 "hist": {"<phase>": {"<level>": {"prop":    [count per entry index],
                                  "forced":  [...],
                                  "explore": [...],
                                  "unknown": [...]}}}}
```

`keys[l][i]` is the level-1 feature tuple of committed entry *i* at level ℓ — the same key grain
as `miner[l]["keys_at_support"]`, as `setup.json["true_tables"][l]["flat"]`, and as the main
lane's delivery ledger. It is written every cycle from `committed[ell]["flat"]`, so the offline
read is a lookup, never a re-derivation.

`phase` is the donor's (`beam` = the priced practice + metering beams, `probe` = everything
unpriced). `source` is **who put the move on the beam at the moment that entry was served**:

| source | meaning |
|---|---|
| `prop` | π's own top-k proposal — **efferent**: π asked this question |
| `forced` | the forced-expansion window (a move too new for the head to have a training example) — **exogenous**: the schedule asked it |
| `explore` | the exploration draw — **exafferent w.r.t. π**: nobody asked, the beam looked anyway |
| `unknown` | reached outside the proposal path. In the beam/probe phases this means the **enumerated** beam, i.e. `state["filter_on"]` is still False during the `prop_warmup` window — a well-defined class, not a gap. It also covers the auditions and the ablation battery. |

Recovering the source needs no permutation tracking: `select_moves` masks the forced set to
−inf before its top-k, and `explore_moves` excludes everything already selected, so the three
sets are **disjoint** and the source of a selected move is a property of its move index in that
row alone. The donor's `torch.sort(...).values` line is therefore untouched.

## The insertions

| addition | where |
|---|---|
| `AP_SRC`, `_APREC`, `_ap_reset`, `_ap_take` | beside the donor's `_ENTRY_REC` scaffolding |
| `select_moves_rec`, `explore_moves_rec` | note-taking wrappers that return the donor's own output object |
| `expand_selected_rec` | a byte-identical copy of `PN.expand_selected` plus two lines that hand the per-row source to the recorder around each `apply_any` |
| `_ap_source_of` | the (N, k) source codes, by set membership |
| the join block in `macro_features_rec` | the same `best` the donor already computes, binned by source — one extra `index_add_` |
| `_install_entry_recorder` extension | also swaps `PN.select_moves` / `PN.explore_moves` / `PN.expand_selected`; `beam_moves_prop` resolves all three as module globals at call time, so `prop_net.py` is untouched |
| `log["ap_join"]` + the per-cycle append | beside `log["entry"]`'s append |
| cfg key `ap_rec` (CLI `--ap-rec`, default on) | `_d6_cfg` and the entrypoint |
| `antiphon_p_run`, `ANTIPHON_P_ARMS` | the entrypoint; arms are `wd_s1`'s set **and order**, verbatim |

The addition draws no RNG, changes no numerics, enters no `counts`, and takes no gradient steps.

## Gates

| gate | what it asserts | result |
|---|---|---|
| **`entry_recorder_check`** (extended, in-job at setup, RNG-sandboxed) | the recording copy of `macro_features` is bit-identical to the donor's with the join recorder both **off and on**; the join totals the same executions as the entry recorder; direct (non-beam) calls all bin as `unknown` | **PASS** on `ap_smoke` and `ap_s0` (`max_abs_feats` 0.0, `max_abs_pos` 0.0, `ap_join_total` 1152 = `n_exec_recorded`, all `unknown`) |
| **P-7** (preflight) | `expand_selected_rec` reproduces the **donor callable captured at install time** — children and `n_mat` — with the recorder off and on | built; runs in `preflight` |
| **(a) cross-tag replay** (offline, the load-bearing one) | every arm run out of this file **with the join recorder ON** must reproduce `wd_s1`'s arm of the same name bit-for-bit. Running the gate with the recorder *on* is strictly stronger than a recorder-off replay and costs no extra GPU: the round is its own gate. | **PASS**, all four arms, max\|Δ\| = **0.000e+00** over 13 series × 116 cycles |
| **(b) substrate identity** (offline) | `setup.json`'s `eras`, `refs`, `plant`, `true_tables`, `junk_pool`, the stale references and `read_acc` must equal `wd_s1`'s, and no config key may differ except `ap_rec` (neither run used `--ref-tag`, so the references were recomputed and must land identically) | **PASS**, 8/8 blocks identical, config drift `none` |
| **(c) join ⊆ entry** (offline) | for every (cycle, phase, level), the join's four source vectors must sum **element-wise** to the donor's `entry.hist` vector — the join is a refinement of the donor's instrument, not a second measurement | **PASS**, 0 mismatches over **1592** cells (376/376/376/464) |
| **(d) `unknown` window** (offline) | in the beam phase, `unknown` executions may occur only where `log["prop"][c]["filter_on"]` is False | **PASS**, 0 violations; `unknown` share is 0.0000 at both levels on every arm (the 4 filter-off cycles precede the first commit) |
| **(e) logged keys == reconstruction** (offline) | the `ap_join["keys"]` written by the run must equal `prov_tag.py`'s offline reconstruction of the committed table's key list, entry for entry, every cycle | **PASS**, 0 mismatches over **670** cells — which retro-certifies the offline pass's key maps against ground truth from inside the run |

## Runs on disk

| tag | what | cost |
|---|---|---|
| `ap_smoke` | 1 arm (`given_c1`, so a table exists from c1 and macros execute) at `--quick`, join recorder on; verified the log key, the key format, the source taxonomy, and the join ⊆ entry identity | 0.06 GPU-h |
| `ap_s0` | the round: `exact_reh, exact, exact_exp, given_c1`, 116 cycles each, `wd_s1`'s exact configuration, ladder and arm order, seed 0, join recorder on. App `ap-zeWauSeh26C2eRcpV34XkO`. | **1.61 GPU-h** (5798 s) — the same wall cost as `wd_s1`, as an inert instrument should be |

`ap_smoke` read exactly as designed: cycle 1 all `unknown` (filter off during warmup), cycle 2
all `forced` (every move new to the head), cycles 5+ `prop`-dominant with a small `explore` tail
in the beam and a pure-`prop` probe (performance does not explore).

## Reproduce

```bash
cd experiments/            # MODAL_PROFILE=chromatic

# smoke — `given_c1` so the table exists from c1 and the join has something to record
python3 rhm/practice/antiphon/provenance/launch_detached.py --fn antiphon_p_run \
    --tag ap_smoke --arms "given_c1" --quick

# the round, at wd_s1's exact configuration, ladder and arm ORDER
python3 rhm/practice/antiphon/provenance/launch_detached.py --fn antiphon_p_run --tag ap_s0 \
    --eras "1:25:48,2:12:40,3:6:12,4:3:9,5:1:7" \
    --arms "exact_reh,exact,exact_exp,given_c1" \
    --budget 8 --g-budget 482 --n-corrupt 1 --mine-cap 8 --gy-level 4 --entry-rec \
    --span-min-hold 128 --collect-task-matched --tm-episodes 8192 \
    --recert-every 5 --n-aud 192 --probe-every 8 --n-rt 384 --n-score 256 \
    --sil-cv 0.15 --lp-min-drop 0.10 --reh-n 64 --reh-every 1 --seed 0
modal volume get --force rhm-scaling-data rhm_practice_antiphon_p/ap_s0 \
    rhm/practice/antiphon/provenance/figures
```

## What `ap_s0` measured

All numbers from `figures/ap_s0_reduction.txt`; figures `p1_proposal_grain.png`,
`p2_exafference_and_handle.png`. Everything below is a measured quantity, not an
interpretation — the round is to be discussed with Jasper before anything is read into it.

**Reproduce the reduction**: `python3 rhm/practice/antiphon/provenance/analyze_ap.py --figures`

### (1) Who put the move on the beam (L3, execution-weighted over the run)

| arm | executions | `prop` | `forced` | `explore` | `prop`@end |
|---|---|---|---|---|---|
| `given_c1` | 9,174,369 | **0.9542** | 0.0000 | 0.0099 | 0.9930 |
| `exact_reh` | 1,904,741 | 0.7221 | 0.2344 | 0.0435 | 0.9759 |
| `exact` | 1,095,906 | 0.6565 | 0.3046 | 0.0389 | 0.9756 |
| `exact_exp` | 1,436,487 | 0.6300 | 0.3108 | 0.0592 | 0.9594 |

`given_c1`'s `forced` column is 0.0000 by construction: its table never commits, so no move is
ever new to the head. τ from a level's arrival to π directing ≥90% of its executions at that
level: `given_c1` **4** cycles, `exact_reh` 14, `exact` 18, `exact_exp` 22.

### (2) The source × provenance 2×2 (L3)

`exaff_prop` = of the executions **π proposed**, the share served by an entry whose spelling the
agent has not itself re-derived (`prov_tag.py`'s live match rule) — exafference at the proposal
grain.

| arm | π-proposed L3 execs | of which exafferent | `exaff_prop` | `exaff_prop`@end |
|---|---|---|---|---|
| `given_c1` | 8,754,131 | 4,048,618 | **0.4625** | 0.1388 |
| `exact_reh` | 1,375,382 | 212,353 | 0.1544 | 0.0295 |
| `exact_exp` | 905,033 | 100,967 | 0.1116 | 0.0375 |
| `exact` | 719,430 | 57,100 | 0.0794 | 0.0737 |

`given_c1` sits at `exaff_prop` = **1.0000 for 44 contiguous cycles (c5–c48)** — every L3
execution π proposed was served by a spelling the agent had never produced — then steps to
0.289 / 0.206 / 0.199 / 0.168 over the next four cycles, at the cycle its L3 miner counts first
cross `mine_support`. Holding-grain and use-grain versions of the same tag for these arms are in
`figures/reduction.txt` §§1–2 and are unchanged from `wd_s1` (the replay gate).

### (3) The one-bit contrast at the proposal grain, against its only in-tag handle

L3 is rehearsed; L2 is the same three arms where rehearsal never fired (n = 3).

| statistic | `exact_reh` | `exact_exp` | `exact` | L3 spread / L2 handle | orders as trust? |
|---|---|---|---|---|---|
| `amax` (trust) | 0.5547 | 0.3750 | 0.2396 | **6.72×** | yes |
| `prop_share` | 0.7221 | 0.6300 | 0.6565 | 2.38× | mixed |
| `exaff_prop` | 0.1544 | 0.1116 | 0.0794 | **1.68×** | **yes** |
| `exaff_prop`@end | 0.0295 | 0.0375 | 0.0737 | 0.64× | inverted |
| `explore_share` | 0.0435 | 0.0592 | 0.0389 | 0.51× | mixed |
| `n_ent_prop` | 17 | 12 | 12 | 5.00× | tied pair |

`exaff_prop` is the **first provenance statistic in this shape that both orders the one-bit pair
as trust does and clears its own handle** (the best the offline pass could do was `self_use_end`,
right order at 0.67× the same handle). `n_ent_prop` — the number of distinct L3 entries π
actively proposes — moves 12 → 17 under credit while the exposure control does not move it at
all (12 = 12); the strict credit > exposure > none test scores that "mixed" because of the tie,
so it is reported here as the credit-vs-tied-pair split it is.

### (4) Circularity check — is the `exaff_prop` ordering content, or π's composition?

Raised in review: `exaff_prop` conditions on π-**proposed** executions and trust is π's mass, so
both move with π's behaviour; the ordering could be an artifact of proposal volume or of the
different speeds at which the arms ramp into proposing. CPU-only on the banked `ap_s0` join;
`analyze_ap.py` section (4); figure `p3_circularity.png`.

**Diagnosis of the dependence structure.** The run-mean is `Σ_c w_c r_c` with `w_c` the arm's own
share of proposed L3 executions in cycle c and `r_c` that cycle's exafference rate. Three routes
by which conditioning could manufacture the ordering: (i) ask-conditioning, (ii) volume (credit
makes π propose L3 1.9× more than `exact`), (iii) **ramp** — `r_c` decays after arrival and the
arms ramp at different speeds (τ to `prop_share` ≥ 0.90: reh 14, none 18, exp 22), so an
execution-weighted mean puts different weight on the high early part of the same curve. (iii)
was the leading suspect: `exaff_prop`'s run-mean and @end orderings disagree in sign.

**(a) Ask-independent reads.** Dropping the conditioning keeps the ordering:

| statistic | reh | exp | none | ×handle | order |
|---|---|---|---|---|---|
| `exaff_prop` (as reported) | 0.1544 | 0.1116 | 0.0794 | 1.68 | ok |
| `exaff_all` — all sources pooled | 0.1385 | 0.1068 | 0.0851 | 1.55 | ok |
| `exaff_forced` — **exogenous only** | 0.091795 | 0.092054 | 0.092483 | — | flat |

`exaff_forced` is the purest ask-independent cell: the forced-expansion window puts those moves
on the beam by **schedule**, not by π. Its L2 handle is degenerate (all three arms read
0.225715 **exactly** — L2's forced window fires at c18–20, before rehearsal at c71 makes the
arms differ), so no multiple is defined; the informative comparison is within L3:

> π-**asked** executions spread across the trio = **0.075027**
> schedule-**forced** executions spread = **0.000688** — **109× smaller**

Same table, same self/other tag, same cycles, same three agents: the exafference difference is
confined to the executions π chose.

**(b) Volume-matched and ramp-free reads.** Uniform subsampling to a matched execution count
leaves a cycle's exafference *rate* unbiased, so the matched-count statistic is exactly a
re-weighting of the per-cycle rates with weights **common to all arms** — computable exactly
rather than simulated. Every such weighting keeps the ordering and **raises** the multiple:

| weighting | reh | exp | none | ×handle | order |
|---|---|---|---|---|---|
| execution-weighted (as reported) | 0.1544 | 0.1116 | 0.0794 | 1.68 | ok |
| matched-count (weights = per-cycle min over the trio) | 0.1545 | 0.1123 | 0.0794 | **1.87** | ok |
| `exaff_all`, equal-cycle | 0.1573 | 0.1120 | 0.0841 | **2.75** | ok |
| `ent_other_prop` (**volume-free**: of the distinct entries π proposed, the share other-provenance) | 0.6168 | 0.5353 | 0.4894 | **3.31** | ok |
| equal-cycle weight | 0.1693 | 0.1118 | 0.0724 | **3.56** | ok |

Explicit hypergeometric subsample to the matched count (400 draws, seed 0) — the ± is the
subsample's own Monte-Carlo error, **not** an experimental floor: L3 reh 0.1545 ± 0.0003,
exp 0.1123 ± 0.0002, none 0.0794 ± 0.0000; three orders of magnitude below the gaps.

**(c) Rate/weight decomposition.** `gap = Σ_c w̄_c (r^X_c − r^Y_c)` [RATE] `+ Σ_c (w^X_c − w^Y_c) r̄_c`
[WEIGHT], the standard symmetric two-term split (the two sum to the gap exactly):

| pair (L3) | gap | rate part | weight part | rate % |
|---|---|---|---|---|
| `exact_reh` − `exact` | 0.0750 | 0.0760 | −0.0010 | **101.3%** |
| `exact_reh` − `exact_exp` | 0.0428 | 0.0434 | −0.0006 | **101.3%** |
| `exact_exp` − `exact` | 0.0322 | 0.0334 | −0.0012 | **103.8%** |

The weight (composition/ramp) term is **negative and ≈1–4% of the gap** in all three L3 pairs:
the ramp works slightly *against* the observed ordering, and the entire gap is rate — different
exafference content at matched τ. **The circularity concern is retired for this cell**; the
volume weighting was suppressing the effect, not creating it.

### A machinery fact found by this check, which demotes two columns already on the record

The entry accumulator is keyed `(phase, level)` and reallocates on table size, and the per-cycle
**audition** — which grades the live *candidate* table, not the committed one — runs under the
`probe` phase tag. So a probe row is the candidate table's whenever an audition ran. Measured on
`ap_s0`: **probe** size-mismatched cycles L2 69–78 / L3 31–32 per arm; **beam** 0/0 on every arm
at both levels. Size-matching does not *prove* a probe row is the committed table's — it only
fails to detect the mixing — so the surviving probe cycles are ambiguous rather than clean, and
no probe number is reported here.

This is a property of the **donor's** `entry.hist` too (gate (c) shows the join reproduces it
element-wise), so it is an `assay`-lineage instrument fact, not a fork bug. Consequence for what
is already written down: it **demotes the `probe@end` and `probearr` columns of
`figures/reduction.txt` section (2)**. Nothing in sections (3)–(4) of that file used the probe
phase — that reduction already stated "only `beam` is carried into (3)-(4)" — and nothing in
`ap_s0_reduction.txt` sections (1)–(3) uses it either. `analyze_ap.py`'s reducer now skips
size-mismatched cycles and counts them.

### Caveats specific to this round

- **Still single seed, still no floor.** The only handle is `wd_s1`'s unrehearsed L2 (n = 3);
  `ap_s0` is a bit-identical replay of `wd_s1`, so it adds **no** independent noise handle —
  every floor-multiple above is the same n = 3 handle the offline pass used. No floor for π
  mass, and none for any provenance statistic, exists anywhere in the arc.
- `exaff_prop` and `prop_share` are **not independent of use volume**: an arm that executes L3
  more also re-derives more of it, and `given_c1` executes L3 ~8× more than any commit-time arm.
  The run-mean and @end columns disagree in sign for the one-bit pair, which is that dependence
  showing.
- The `forced` column is a schedule artefact, not a behavioural variable: it is large for
  commit-time arms only because their L3 moves are new to the head at c71.
- `given_c1`'s 44-cycle `exaff_prop` = 1.0000 plateau ends at the `mine_support` = 3 crossing,
  so its length is a property of that threshold as much as of the agent — and the threshold is
  not sweepable from the banked logs (see the offline caveats above).
- The circularity check in (4) retires the composition route for the one-bit trio only. It says
  nothing about `given_c1`, whose proposal volume differs from the trio's by ~8×; the
  arrival-contrast numbers in (1)-(2) remain execution-weighted and uncontrolled for that.
- `ent_other_prop` (0.49-0.62) sits far above the execution-weighted `exaff_prop` (0.08-0.15)
  because execution mass concentrates on the few entries the agent has re-derived — the offline
  pass's `ess_use` ~ 3 effective entries. The two are answering different questions and should
  not be compared as levels, only within a column.

## P″ — the provenance-gated-credit intervention (authorized 2026-09-01)

Green-lit by Jasper after P′ and the circularity check. **The gate semantics, fixed before
launch, are below.** Machinery lives in the same fork (`antiphon_p.py`, `# [antiphon_p]`
markers); no new file.

### What the gate is

The gate is **one extra conjunct in `prop_pairs`' existing `keep` mask** — the same
trajectory-level selection step whose docstring already names it "the selection step that makes
this self-imitation rather than averaging over everything the agent did". No new learning rule,
no new signal, no oracle.

> **`credit_self`** — π imitates a rehearsal trajectory iff it (i) **SOLVED** and (ii) **every
> level ≥ 2 (macro) move in its chosen action sequence came from π's own top-k proposal**: none
> drawn by the exploration draw, none handed over by the forced-expansion window.
>
> **`credit_match`** — the **same number** of trajectories, drawn uniformly at random from the
> **same solving set**, provenance-blind.

Why *every* macro and not only the rehearsed level's: in the `exact` family **both** L2 and L3
are received — `exact` surgery installs the DGP's table at each level's own commit cycle — so
the received vocabulary is all of the macros, not just L3's. The looser level-restricted variant
is computed and logged every cycle (`prov.pass_rate_level`) but is **not** the gate; it bounds
how much of the strict gate's exclusion comes from the lower level.

Why level-1 primitives are exempt: they carry no table entry, are never received vocabulary, and
so have no provenance. A solving trajectory with **no macro at all** passes vacuously — it used
only vocabulary the agent has always had — and is counted separately (`prov.n_nomacro`).

This is the arity-2 law as a filter — *credit requires your own act in the datum's causal
history* — read at **the grain where the act happens** (π's proposal), which is what P′ built and
what the offline answer-level key match could not do.

### Why the matching makes the one bit the provenance filter

`prop_pairs` emits exactly `budget` pairs per kept trajectory, so **matching the trajectory count
matches the pair count exactly**. Both modes build a `(B, W)` boolean mask of the *same size* out
of the *same* solving set and hand it to the *same* `prop_pairs` call; the only difference is
whether the mask is the provenance filter or a uniform draw. Identical episodes, identical
states, identical trajectory count, identical pair count, identical gradient steps. The random
draw (`rehrng.permutation`) is made in **both** modes so RNG consumption matches, and the donor's
own `perm` draw is left exactly where it was so `credit` / `expose` stay bit-identical.

Per the `woodshed` lesson, **the pair-count match is assertable only on the first rehearsal
cycle**, when the two arms are still the same agent. From the next cycle they are different
agents and their self-counts legitimately diverge — which is the treatment, not a fault.

### The dose ledger (`log["reh"][c]["prov"]`)

Computed **unconditionally** on every mode — pure tensor ops, no RNG, no numerics — so any future
rerun of `credit` / `expose` carries it too: `n_traj`, `n_solved`, `n_self`, `n_self_level`,
`n_nomacro`, `n_macro_appl`, `n_macro_prop` / `_forced` / `_explore`, `n_atlevel_appl`,
`n_atlevel_prop`, `pass_rate`, `pass_rate_level`, `n_traj_used`. `wd_s0` delivered its treatment
at ~2% of design and nothing on stdout said so; here **the pass rate prints on every rehearsal
cycle**, so a degenerate dose is visible from the launch log in flight.

### The gate-unavailable branch

The proposal grain only exists while π is **gating** the action set. Before `prop_warmup`
expires the beam is the enumerated one, nobody proposed anything, and there is nothing to gate
on — so the cycle is **not consumed**, and is recorded (`reh.gate_unavailable`) and printed
rather than silently credited. At this round's configuration rehearsal starts at c71 and the
filter has been on since c5, so the branch is dead in the round and live only in the quick smoke
(where it fired at c1 exactly as intended).

### Arms and gates

| arm | table | rehearsal | job |
|---|---|---|---|
| `exact_reh_s` | `exact`'s | L3, **credit_self** | the treatment |
| `exact_reh_m` | `exact`'s | L3, **credit_match** | the count-matched provenance-blind control |

The unchanged arms are **not** re-run: `ap_s0`'s banked `exact` is the twin these two must
reproduce through the last pre-rehearsal cycle, and that twin gate is what licenses reading them
against `ap_s0`'s `exact_reh` / `exact_exp`. Both new arms carry `TWIN = "enum_live"`, the same
stream key as every other surgery arm.

| gate | what it asserts |
|---|---|
| **twin gate** | `exact_reh_s` / `exact_reh_m` must be bit-identical to `ap_s0`'s `exact` over 13 series through the last pre-rehearsal cycle (c70), with the first divergence exactly at the first rehearsal cycle. This also certifies the new `beam_moves_prop` source bookkeeping inert, since `ap_s0`'s `exact` ran without it. |
| **first-cycle pair-count match** | `n_pairs_used` equal on the first rehearsal cycle only (the `wd_s0` lesson) |
| **in-tag one-bit** | both arms' `prov` ledgers must show the same `n_solved` on the first rehearsal cycle (same agent), and `n_traj_used` equal in both |
| **`entry_recorder_check` / P-7** | unchanged, run in-job at setup |

### Runs

| tag | what | cost |
|---|---|---|
| `pp_smoke`, `pp_smoke2` | two failed launches, **my error**, on the record: the first patch of `beam_moves_prop`'s source bookkeeping landed in `beam_moves` instead (identical lines, first-occurrence replace), then the hard `assert` on the gate-unavailable branch killed the quick smoke at c1 | ~0.15 GPU-h lost |
| `pp_smoke3` | 2 arms (`given_c1` + `reh_level=3`) at `--quick`, both new modes; verified the log key, both mask constructions, the pair path, the ledger, and the gate-unavailable branch | 0.13 GPU-h |
| `pp_s0` | the round, **stopped after arm 1**: `exact_reh_s` complete (116 cycles, 11.0 s/cycle, banked); `exact_reh_m` cut, because a count-matched control for a 4.9%-dose treatment buys nothing. App `ap-YsmWrpGL2ftEbP9Kn7mh6t`. | **0.54 GPU-h** (setup 665 s + 1276 s) |

**Dose read from `pp_smoke3`, and its limits.** Aggregate strict pass rate ≈ 0.11 at `--quick`,
with `n_macro_forced` = 0 on every cycle — the smoke used `given_c1`, whose table exists from c1,
so no move is ever new and the forced channel never opens. The smoke therefore exercises only
one of the gate's two exclusion channels, and its solve rate (0.001–0.04) is 4–150× below the
round's (0.151), which makes solving there unusually exploration-dependent and the pass rate
correspondingly pessimistic. **The dose at the round's own scale is not known in advance**; it
prints from the first rehearsal cycle and is the first thing to read.

### What `pp_s0` measured, and why arm 2 was not run

**Run stopped by me, deliberately, after arm 1 completed and committed.** `exact_reh_s` ran all
116 cycles (11.0 s/cycle) and is banked; `exact_reh_m` was cut before it did meaningful work,
because a count-matched control for a treatment delivered at 4.9% of dose buys nothing. Total
spend **≈0.82 GPU-h** against the 1.2 cap (0.28 smokes incl. ~0.15 lost to my own error, 0.54
for setup + arm 1). This call was mine, on the in-flight dose ledger; it is the mitigation I
stated at the launch halt.

**Gates.** Twin gate **PASS** on the corrected statement: `exact_reh_s` is bit-identical to
`ap_s0`'s `exact` over 13 series through **c72** (max|Δ| = 0.000e+00) with the first divergence
at exactly **c73**. The strict form I first wrote — "first divergence == first rehearsal cycle"
— is mis-specified for this arm and read FAIL: rehearsal *fires* at c71 but *consumes nothing*
at c71–c72 (the forced window, below), so the arm legitimately stays identical to `exact` until
the first cycle rehearsal is actually consumed. Same class of gate mis-specification `woodshed`
recorded for its pair-count gate; the corrected statement is "identical until the first
CONSUMED cycle, first divergence exactly there". This also certifies the new
`beam_moves_prop` source bookkeeping inert, since `ap_s0`'s `exact` ran without it.
`entry_recorder_check` PASS in-job.

**The dose, over all 46 rehearsal cycles.**

| quantity | value |
|---|---|
| solving rehearsal trajectories | 7,319 |
| **strict gate passes** | **359 = 4.91%** — of which **135 (38%) carry no macro at all** (vacuous passes) |
| level-restricted passes (logged, not the gate) | 2,434 = **33.3%** |
| pairs delivered into π's buffer | **2,872 vs `credit`'s 58,552 → dose 0.049** |
| rehearsal cycles consuming zero credit | **19 of 46** |

For scale, `wd_s0` — the run this node's own record keeps as "a measured near-null-dose run" —
delivered ~2%. **`credit_self` at 4.9% is the same order.** The treatment was not delivered, so
the arm measures the gate's dose, not provenance-gated credit.

**The mechanism, which is the informative result and is new.** Among the macro applications
inside *solving* rehearsal trajectories:

> **prop 7,364 (31.5%) · forced 1,576 (6.8%) · explore 14,415 (61.7%)**

Nearly **two thirds of the macro work that produces a rehearsal solve comes from the
exploration draw**, and under a third from π's own top-k proposal. The forced window is a small
total share but is *total* while it is open: at c71–72 every L3 move is new to the head
(`prop_new_cycles` = 2), macro sources read 50/698/118 and 32/878/136, and the gate passes
nothing — which is why the first consumed cycle is c73.

So the strict gate is not merely underpowered: at this substrate a strict arity-2 reading says
most of `woodshed`'s rehearsal supervision is **not** π's own ask, and what little survives the
filter is 38% trajectories carrying no received vocabulary at all.

**The corrected shape, specified rather than guessed.** The level-restricted variant — only the
*rehearsed* level's macros must be π-proposed — passes **33.3%**, a workable dose, and it was
logged on every cycle of this run rather than being a hypothesis. It is the gate a follow-up
would use. Whether it should be preferred on principle is exactly the question the strict
version's failure raises (the `exact` family receives L2 by surgery too, which is why the strict
form was chosen first), and that is a question for Jasper, not a call to make here.

**Caveats.** Single seed; one arm; no control arm, so nothing about *value* or *trust* is
readable from `pp_s0` — only the dose and the source decomposition. The dose ledger is exact
(counts, not estimates). Figure: `figures/pp1_dose.png`.

## pp_s1 — the corrected-dose round (level-restricted gate)

Authorized by the coordinator after `pp_s0`'s near-null dose, on the **`wd_s0` → `wd_s1`
corrected-dose precedent**: a near-null dose is an apparatus outcome, not an answer, and the
corrected re-run belongs to the same conversation. `pp_s0`'s strict cell stays banked as the
measured stronger form.

### ⚠ OPEN PRINCIPLE QUESTION FOR JASPER — flagged, not settled

**Is the level-restricted gate the right operationalization, or a weakening?** The two readings:

- **Against it** (why `pp_s0` used the strict form): in the `exact` family **both** L2 and L3
  are received — `exact` surgery installs the DGP's table at each level's own commit cycle — so
  "the received vocabulary" is *all* the macros, and a gate that lets an exploration-sourced L2
  macro through is crediting a solve that partly rode on someone else's answer.
- **For it** (the coordinator's reasoning, recorded verbatim in intent): the question P″ was
  asked is whether *crediting answers to your own asks about the received vocabulary* forms
  different trust. Requiring every exploration step **elsewhere** in the trajectory to also be
  π-proposed conflates *provenance-of-the-received-level's-use* with *no-exploration-anywhere*,
  and `pp_s0`'s ledger shows that conflation is fatal at this stage of learning: **61.7% of
  solving macro work is exploration**, so the strict gate is a de-facto no-exploration filter.

**If Jasper judges the level-restricted form unprincipled, `pp_s1` still measures it as a
labeled variant, not a headline.** Both forms are computed and logged on every cycle of both
rounds, so either can be read as the primary without a re-run.

### The gate, stated precisely (written before launch)

`reh_gate` is a new cfg key. `"macro"` is `pp_s0`'s semantics **verbatim** and is the default,
so `pp_s0` stays reproducible; `"level"` is this round.

> **Eligible pool** — a solving rehearsal trajectory that applies **at least one** rehearsed-level
> (L3) macro. Trajectories that never touch the received level carry none of its vocabulary and
> therefore carry no provenance information either way; they are excluded from **treatment and
> control alike**, rather than passing vacuously.
>
> **`credit_selfL`** (`exact_rehL_s`) — π imitates a trajectory iff it is **eligible** and
> **every rehearsed-level macro in its chosen action sequence came from π's own top-k
> proposal** (none from the exploration draw, none from the forced-expansion window). Macros at
> other levels and level-1 primitives are unconstrained.
>
> **`credit_matchL`** (`exact_rehL_m`) — the **same number** of trajectories, drawn uniformly at
> random from the **same eligible pool**, provenance-blind.

This is the change of substance from `pp_s0`: there the vacuous class **passed** and made up
38% of the strict gate's tiny yield; here it is removed from both sides, so the one bit is
purely *were the received level's uses π's own asks*. The `pp_s0` semantics (vacuous-pass, pool
= all solves) remain available under `reh_gate="macro"` and are what that tag ran.

### Expected dose, from `pp_s0`'s logged ledger (the pair arithmetic)

`prop_pairs` emits exactly `budget` = 8 pairs per kept trajectory, so trajectory counts and pair
counts are proportional and the control matches both exactly. From `pp_s0`'s 46 rehearsal cycles
(7,319 solving trajectories, 58,552 `credit` pairs):

| quantity | measured in `pp_s0` |
|---|---|
| rehearsed-level macro applications per solving trajectory | **1.528** |
| of those applications, share π-proposed | **0.2976** |
| level-gate pass rate **including** vacuous (logged) | **0.3326** |
| vacuous class (no L3 macro at all), Poisson estimate | **0.217** |

The Poisson × independent-prop model reproduces the logged rate to 3% (predicts 0.342 vs logged
0.3326), so it is trustworthy for the decomposition: **non-vacuous level passes ≈ 0.125** of all
solves, i.e. **≈ 16.0% of the eligible pool** (0.125 / 0.783).

> **Projected delivery: ≈ 0.125 × 58,552 ≈ 7,300 pairs**, matched exactly by the control.
> For scale: `pp_s0`'s strict gate delivered **2,872** (dose 0.049); `wd_s0`'s near-null
> **1,728**; `wd_s1`'s full `credit` **56,840**.

So the corrected gate is **2.5× the strict dose and 4.2× `wd_s0`'s**, but still **~1/8 of full
credit**. That is the honest risk: the arms are count-matched, so the comparison is fair at
whatever volume lands, but if 7,300 pairs move neither arm the contrast may not clear the
handle. The dose prints every rehearsal cycle, as before.

### Gates

| gate | statement |
|---|---|
| **twin gate** (corrected form) | both arms bit-identical to `ap_s0`'s `exact` over 13 series through the **last cycle before rehearsal is CONSUMED**, with the first divergence exactly at the first consumed cycle. Not "the first cycle rehearsal fires" — `pp_s0` showed those differ (fired c71, consumed c73) whenever the gate passes nothing early. |
| **first-consumed-cycle pair-count match** | `n_pairs_used` equal on the first **consumed** cycle only — the `woodshed` lesson, in the form `pp_s0` sharpened |
| **in-tag one-bit** | both arms' ledgers show the same `n_solved` and the same `n_eligible` on the first consumed cycle (same agent), and equal `n_traj_used` |
| `entry_recorder_check` / P-7 | unchanged, in-job at setup |

### Budget flag

Setup 648 s + 2 × 116 cycles at `pp_s0`'s measured **11.0 s/cycle** = 3,200 s = **≈0.889
GPU-h**, against the stated **≤0.8** cap — an **11% overrun**, flagged before launch rather than
after. A one-armed round is scientifically void here (the count-matched control is the design),
so the pair is launched; if the cap is hard, the run can be stopped after arm 1 at ≈0.53 GPU-h
and the control run separately. No paid smoke was taken for this variant: it is a six-line delta
on machinery `pp_smoke3` and `pp_s0` already exercised end-to-end (same modes, same mask
construction, same `prop_pairs` path), and the budget is better spent on the round.

### What `pp_s1` measured

Reduction `figures/pp_s1_reduction.txt`, figure `figures/pp2_level_pair.png`, reducer
`analyze_pp.py`. Cost **0.90 GPU-h** (3,253 s) against the ≤0.8 cap — the 11% overrun flagged
before launch, realized at 12.5%.

**Gates: ALL PASS.** Twin gate in the corrected first-CONSUMED-cycle form — both arms
bit-identical to `ap_s0`'s `exact` over 13 series through **c80** (max|Δ| = 0.000e+00), first
divergence exactly **c81**, which is also the first consumed cycle (rehearsal *fires* at c71 and
passes nothing for ten cycles). First-consumed-cycle pair count **8 vs 8**; in-tag one-bit
identical there (`n_solved` 111, `n_eligible` 77, `n_traj_used` 1). Substrate identical to
`ap_s0` on eras/refs/plant/true_tables/junk_pool with no config drift outside `ap_rec`,
`reh_gate`.

#### Realized dose, and the sizing miss (apparatus records)

| arm | solved | eligible | no-L3 | passes | pairs | dose | zero-credit cycles |
|---|---|---|---|---|---|---|---|
| `exact_rehL_s` | 7,124 | 5,353 | 1,771 | 607 | **4,856** | **0.085** | 17 / 46 |
| `exact_rehL_m` | 7,409 | 5,723 | 1,686 | 897 | **7,176** | **0.121** | 16 / 46 |

**Realized 0.085 vs projected 0.125.** The pre-launch projection used a Poisson(1.528) ×
independent-prop(0.2976) model fitted to `pp_s0`'s aggregates, which had reproduced `pp_s0`'s own
logged level-gate rate to 3%. Decomposing the miss: the **eligible share** came in at 0.7514 vs
0.7830 predicted (close), while the **conditional pass rate** came in at 0.1134 vs 0.1595
predicted (misses by 29%). So the failing term is the conjunction: **within a trajectory the
sources of its rehearsed-level applications are positively correlated** — a trajectory that
explores into one L3 macro tends to explore into the others — and "every L3 application was π's
own ask" is rarer than independence predicts. Same class of error as the `wd_s0` sizing miss,
caught this time before the round rather than after it.

**A DESIGN DEFECT, named: the run-total volumes are not matched.** Each arm uses *its own*
provenance-filter yield as its per-cycle draw count — the `woodshed` `expose` idiom — and the
arms are different agents from c82. So the pair is matched **only on the first consumed cycle**,
and over the run **the control delivered 7,176 pairs against the treatment's 4,856, a 48%
volume advantage to the control**. Every comparison below is confounded by that, in the
direction of the control. The fix for any future round is to **yoke** the control's per-cycle
count to the treatment arm's logged count (the arc's own `yoked_*` idiom), so run-total volume
is matched rather than only the first cycle.

#### Trust — does not clear

| | L3 mass | L3 amax | L2 mass | L2 amax |
|---|---|---|---|---|
| `exact_rehL_s` (provenance-gated) | 0.2383 | 0.2891 | 0.2479 | 0.1589 |
| `exact_rehL_m` (count-matched blind) | 0.2818 | 0.3828 | 0.2277 | 0.0729 |
| **\|effect\|** | 0.0434 | **0.0938** | 0.0202 | 0.0860 |
| unrehearsed-L2 handle (n=3, incl. `ap_s0/exact`) | — | — | 0.0237 | 0.0859 |
| **L3 effect / L2 handle** | **1.83×** | **1.09×** | | |

The sharpest way to say it: **the two arms differ as much at L2, where rehearsal never fired
(amax 0.0860), as at L3, where it did (0.0938)** — ratio 1.09. The contrast does not clear its
own divergence handle. For scale, `ap_s0`'s credit-vs-exposure pair differs by 0.0960 mass /
0.1797 amax at L3, roughly twice this pair.

The sign is also **against** the treatment: provenance-gated credit produced *less* L3 trust
than provenance-blind credit on both statistics — and the control had 48% more supervision, which
predicts exactly that, so the sign is not interpretable either.

#### Value — one cell separable, and it goes the control's way

Recovered fraction, and closure of the `as_s0/anchor` → `ap_s0/given_c1` bracket (era references
asserted identical):

| arm | era4 | era5 | closure e4 | closure e5 |
|---|---|---|---|---|
| `exact_rehL_s` | 0.658 | 0.393 | 0.160 | 0.261 |
| `exact_rehL_m` | 0.636 | 0.557 | 0.124 | 0.485 |
| `ap_s0/exact_reh` (full credit) | 0.976 | 0.705 | 0.701 | 0.687 |
| `ap_s0/exact_exp` (exposure) | 0.733 | 0.448 | 0.289 | 0.336 |
| `ap_s0/exact` (untreated) | 0.670 | 0.410 | 0.180 | 0.284 |
| `pp_s0/exact_reh_s` (strict, dose 0.049) | 0.745 | 0.464 | 0.309 | 0.358 |

Against the measured depth-6 stream floor (0.087 on recovered fraction, `as_s1` / `census`
finding 7): **era 4 difference 0.022 — well below the floor, not separable**; **era 5 difference
0.164 — 1.9× the floor, favouring the control**, and fully consistent with the control's 48%
volume advantage.

Both `pp_s1` arms sit at or below `ap_s0`'s *exposure* control on bracket closure and far below
full credit (0.70/0.69), which is what a ~9–12% dose predicts. Note also that `pp_s0`'s strict
arm closed **more** (0.309/0.358) at a **lower** dose (0.049) than either `pp_s1` arm — a
single-arm spread larger than the treatment effect, and a direct demonstration that differences
at this dose are not separable.

#### Verdict, plainly

**The cell does not resolve.** There is no evidence here that provenance-gated credit forms
different trust from count-matched provenance-blind credit: the trust contrast fails its own
unrehearsed-level handle (1.09×), era-4 value is below the stream floor, and the one separable
cell (era 5) favours the control and is explained by an unmatched 48% volume advantage that the
design did not control. The round's real products are the apparatus records — the realized dose,
the correlation diagnosis, and the named volume-matching defect with its `yoked_*` fix.

## P″ — the original spec, superseded by the section above

Provenance-gated credit as a one-bit variant of `woodshed`'s `reh_mode`: a third mode
`credit_self` that imitates only the rehearsal trajectories that solved **and** whose macro
executions were served from self-tagged entries, pair-count matched against `credit` by
subsampling so volume and gradient steps are held. ~0.5 GPU-h on top of a `wd_s1`-shaped run.
The offline pass argues against it (the gate would remove ≤14% of L3 credit on the treated arms
and is anti-correlated with what forms trust); `ap_s0` is what would say whether the *proposal*
grain changes that. To be discussed with Jasper before anything is priced.
