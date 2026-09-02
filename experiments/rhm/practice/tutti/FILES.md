# FILES — `tutti` (the unification node: both currencies, and the selector, in one loop)

Machinery record for the node. **No results are interpreted here** — the reduction's numbers go
to the orchestrator and are discussed before any writeup (repo norm). There is deliberately no
`README.md` yet for the same reason.

**Up**: parent arc [`../README.md`](../README.md) · [`DESIGN.md`](DESIGN.md) (the reviewed
design: the fork choice, the arms, the gates, the yoke plan, the L5 section) ·
`../../../../QUEUE.md`[^private] "Critical path" → *the unification node* ·
`../../../../ROADMAP.md`[^private] §7.2.2/§7.2.3
**Direct donor** (untouched): [`../caesura/`](../caesura/FILES.md) — E′ round 3. `caesura.py` is
forked here; `ca_s0` is the cross-tag replay reference and the borrowed lifetime ceiling.
**Imported, not forked**: `maestro/policy.py` (A1's rule and dead zones — *this round adds no
rule; it adds one more POLICY OBJECT for A1's existing rule to be instantiated twice*) ·
[`../antiphon/questions.py`](../antiphon/FILES.md) (the selectors and the quota, so Q-1…Q-10
apply unchanged) · `native/span/span_net.py` (through the donor) · `crescendo/phase0_l4.py`
(through the reduction).
**Through the donor**: [`../intonation/`](../intonation/FILES.md) (δ_perf, the fallible
executor) · [`../tacet/`](../tacet/FILES.md) · [`../crescendo/`](../crescendo/FILES.md) (A3 —
the yoke idiom, the L4 frontier, the measured floors) · [`../maestro/`](../maestro/FILES.md) ·
[`../conductor/`](../conductor/FILES.md).
**Sibling lane**: `sizing/` — the L5 offline sizing, written by another lane, not touched here.

## The question, and the two things that vary

The loop has three actions — COMMIT level ℓ, ADVANCE the era, SELECT the questions — and two
currencies can license them: the **outcome** currency (one-level-up `at_support` yield, A1's
thermostat) and the **execution** currency (δ-silence, `dsil` = mean `b(s)` over open slots,
read through A1's `QuietPolicy` unchanged). Two axes:

| | commit | advance |
|---|---|---|
| **(b)** `tu_y_exo`, `tu_y_endo` | yield | yield |
| **(a)** `tu_d_exo`, `tu_d_endo` | dsil | dsil |
| **(c) the split** `tu_s_exo`, `tu_s_endo` | **dsil** | **yield** |
| **the mirror** `tu_m_exo` | yield | dsil |

crossed with the selector off (`exo` — the menu's head in the donor's own RNG order) and on
(`endo` — half-key novelty × the posed-vs-landed delivery ledger).

## What the fork adds (each `# [tutti]`-marked)

| addition | where | why |
|---|---|---|
| `pose_questions` + `QS` import | module | **the question port**, grafted from `antiphon.py`'s `# [antiphon]` block. Its ONE call site is `run_arm` block (a)'s `context_instances` draw — which `caesura` leaves byte-identical to `crescendo`, so the graft touches no donor edit |
| the dose join | `run_arm` block (b) | `antiphon` carried a parallel index here; the donor already computes it (`intonation`'s `mine_take` over `np.flatnonzero(ps > 0.5)`), so this REUSES it. `mine_ord` is None in every arm (no diet gate is built) |
| **`loop_commit`** | `run_arm`, blocks (d)/(g) | **the split**: a second `QuietPolicy` owning the COMMIT while the primary owns the ADVANCE. Not a new rule — the donor already builds a second instance (`dsil_det`, the veto's read-only detector); this promotes it to licensing |
| `_acted_all` | `run_arm` | both policies re-arm on EVERY action and at every era start, not only their own — the donor's regime-change reasoning. The own-action-only alternative is a counterfactual in reduction §5T, never an arm |
| the bootstrap generalised | block (g) | the fallback belongs to whichever policy OWNS the commit (`loop_c or loop`). With `loop_commit` absent this is the donor's condition character for character |
| `question_mode`/`question_k`, `TUTTI_ARMS`, `TUTTI_TOL_DSIL` | `_cfg`, module | all default OFF, so an unnamed config is `caesura.py` |
| `log["q"]`, `split`/`q_mode`/`q_k`/`q_ledger` in results | `run_arm` | the port's and the split's own records |
| gate **X** | `preflight` | below |
| §0T, §1T, §2T, §3T, §4T, §5T, §7T, §8T | `analyze_tutti.py` | the round's own readouts |

**Precedence needs no code.** A commit calls `_acted_all(COMMIT, …)`, which resets both
policies, so `loop.quiet` is already False when block (g4) reads it — one quiet reading can
never fire both, exactly as in `conductor`. The donor's rule is *"a quiet reading COMMITS the
active level IF THERE IS ONE TO COMMIT, and ADVANCES otherwise"*, and the second clause is
load-bearing: a both-quiet cycle that advances is correct where the active level already
carries a table. Gate X-4 and reduction §5T both check availability rather than assuming it.

**The MIRROR's advance owner reads `dsil`, which does not exist before a slot opens**, so its
era-1 advance falls through to `hit_cap` — and since `CRESCENDO_LADDER ≡ CRESCENDO_CAPS`, the
cap IS the arc's own advance bootstrap and lands on the schedule arm's own cycle.

## The floor, re-derived in-tag

`caesura` ran governed by an **offline** `tol_dsil = 0.00321442` while its own reduction
re-derived 0.00492 in-tag. `phase0_tutti.py` re-derives it with the protocol pinned (A1's
`null_abba`, span 1, W 4, `v_tol = sd(N)/√W`, commit and advance cycles skipped as regime
changes) on `ca_s0`'s own logged `dsil` series — a pure function of `log["panel"]`, so it costs
zero GPU:

| source | n blocks | sd(N) | `v_tol` |
|---|---|---|---|
| `ca_s0/dsil_sched` | 163 | 0.009391 | 0.004696 |
| `ca_s0/dsil_yield` | 43 | 0.004631 | 0.002316 |
| `ca_s0/dsil_read` | 117 | 0.010824 | 0.005412 |
| `ca_s0/dsil_and` | 31 | 0.005205 | 0.002602 |
| **pooled — this tag's governing value** | **354** | **0.009187** | **0.00459** |

**1.429× looser than the donor's governing value.** The stated consequence: `tu_d_exo` is no
longer a full-life replay of `ca_s0/dsil_read`. Phase 0 computes where they part — replaying
A1's rule on that arm's own series at both floors puts the **first divergence in the quiet
verdict at c43** — so §0T asserts that replay over **c1–c42** and the arm is a re-instantiation
above it. `tu_y_exo` is yield-paced and therefore floor-independent, and carries the full-life
gate. Reduction **§7T** then replays both floors on *every* arm's own series, which is what
separates the dead-zone change from the trajectory.

## Code files

| file | purpose |
|---|---|
| `tutti.py` | The substrate. Forks `../caesura/caesura.py`; every addition marked `# [tutti]`. New: the question port and its one call site, the dose join, `loop_commit` + `_acted_all` + the generalised bootstrap, the eight arms plus five preflight arms, `question_mode`/`question_k`, and gate **X**. Entrypoints: `preflight`, `fidelity_smoke` (G-F + gate Q-11), `tutti_run`. |
| `analyze_tutti.py` | The reduction. The donors' sections, plus **§0T** (the cross-tag replays), **§1T** (the port's controls), **§2T** (the dose, on a fallible executor), **§3T** (the division-of-labor table), **§4T** (the selector per pacer, free-paced and yoked), **§5T** (the split's trace and the reset counterfactual), **§7T** (the floor counterfactual at both dead zones), **§8T** (the priced clock). |
| `phase0_tutti.py` | **The offline phase.** The `tol_dsil` re-derivation, the bounded-gate cycle, the split sized on the donor's own series (and the deadlock check), and `antiphon`'s menu sizing re-checked on this ladder. Writes `phase0.json`. No GPU, no Modal, no substrate. |
| `phase0.json` | Phase 0's output. |
| `launch_detached.py` | Session-isolated detached launcher (the donor's, retargeted). |
| `DESIGN.md` | The reviewed design note, with the decisions as taken and §8 corrected by the sizing lane. |

`policy.py`, `floors.json`, `questions.py` and `phase0_l4.py` are **deliberately absent** — the
rule, the dead zones, the selectors and the offline `Miner.build` replay are reached by import.

## Gates

| gate | what it asserts | status |
|---|---|---|
| P-1…P-12, L-1…L-8 | A1's/A2's offline policy suite against the **imported** module | **ALL PASS** (21 checks) |
| **Q-1…Q-10** | `questions.py::question_gate`, unchanged (containment, quota, determinism, breadth-over-depth, the ledger, the span geometry) | **ALL PASS** (10/10) |
| C-1…C-5, T-1…T-5, I-0…I-5, D-1…D-5 | the donors' suites, re-run in this fork | **ALL PASS** |
| **X-1** | the two policies exist, are **distinct**, and each reads only its own series | **PASS** |
| **X-2** | **inertness** — with `loop_commit` absent the arm is the donor: no split record, no `c_*` trace columns | **PASS** |
| **X-3** | the split's bootstrap fires, only while the gauge is absent, and era-1 advances are yield-driven while commits bootstrap | **PASS** |
| **X-4** | **precedence** — no cycle carries two gauge-driven actions; a both-quiet cycle that advanced had no commit available. Cap-forced co-occurrence reported, not asserted | **PASS** |
| **X-5** | the **two-policy yoke** replays its source's realised COMMIT *and* ADVANCE cycles exactly, and builds no second policy | **PASS** |
| **X-6** | the port ran in every arm, the quota held bin-for-bin, the two selectors did different things | **PASS** |
| **G-F** | with every `# [tutti]` knob off, this fork replays **`caesura.py`** in process | **PASS — 0.000e+00** on `anchor` and `given_c1`, vs a 0.000e+00 donor self-replay control, commits equal |
| **Q-11** | every selector, every era, on the real substrate | **PASS** (18 cells); needy-true-key aiming exo→bisect **7→10 / 8→18 / 1→9** |
| **cross-tag (full life)** | **`tu_y_exo` ≡ `ca_s0/dsil_yield`**, 13 series × 131 cycles, commits and advances identical. **The round's load-bearing gate**, and free | **PASS — 0.000e+00** |
| **cross-tag (bounded)** | **`tu_d_exo` ≡ `ca_s0/dsil_read` over c1–c42** (Phase 0's floor-divergence cycle) | **PASS — 0.000e+00** |
| in-tag twin | every **exo** arm vs `tu_y_exo` to its own first action | **PASS — 0.000e+00** (`tu_d_exo`/`tu_s_exo` to c17, `tu_s_yk` to c29, `tu_m_exo` to c48). The **endo** arms differ from c1 **by construction** — the selector acts on cycle 1 — so their lifetime control is the yoke, not a twin window |
| panel vs G-Y | the observation panel reproduces the donor's G-Y miner at L4 | **PASS**, all 8 arms, every cycle |
| quota | per-cycle d\* histogram bin-for-bin | **PASS — 1.000** in all 8 arms; selected d\* deviates from the menu by −0.021…+0.004 |
| N-1, N-2, F, B-1 | A1's/A3's, unchanged | as donors |

**Name caution** (extending the donors'): this substrate carries *five* unrelated "gates"
(census's L2 admission, span's parity, tacet's learning, intonation's agency, caesura's
δ-silence veto) and now *two* policy objects. None reads a key belonging to another; X-1
asserts it.

## Volume layout

`rhm-scaling-data:/data/rhm_practice_tutti/<tag>/{setup.json,summary.json,<arm>/results.json,done.txt}`.
Fetched copies, figures and `reduction.txt` under `figures/<tag>/`; the banked reduction is
`figures/tu_s0_reduction.txt`.

## Runs

| tag | what | outcome |
|---|---|---|
| `_preflight` (×3) | interface + gates C, T, I, D and **X**, toy sizes. Pass 1 tripped X-4's own over-strong precedence assert (a cap-forced advance coinciding with a boundary commit — the donor's behaviour, not the split's); the gate now scopes to gauge-driven actions | **ALL PASS** (`results/preflight2.log`, `preflight3.log`) |
| `tu_gf` | G-F: in-process fork-vs-`caesura.py` replay at `max_macro_level=4`, **plus gate Q-11** | **PASS — 0.000e+00** (`results/gf.log`) |
| `tu_smoke` | 3 arms at `--quick`, `--question-k 2048`. The pair `tu_pf_off` / `tu_y_exo` is one bit apart (port off vs on) and measures **the port's marginal cost: +0.1 s/cycle**; `tu_s_endo` adds +0.3 (port + endo selector + the second policy step). The dose read is NOT informative at this scale (`n_mined` = 1 on every readable cycle) and moves to the main run, as `an_smoke`'s lesson says | clean, **1050 s = 0.29 GPU-h** |
| `tu_s0` | the main run: 8 arms, `ca_s0`'s ladder/caps/floors, one seed, `--tol-dsil 0.0046 --question-k 2048` | clean, **4.43 GPU-h** of arm time (13.1–14.7 s/cycle; 131/184/141/117/117/160/126/164 cycles). Record `figures/tu_s0_reduction.txt` |

## Reproduce

```bash
cd experiments/            # MODAL_PROFILE=chromatic

PYTHONPATH=. python3 rhm/practice/maestro/policy.py          # the offline policy suite
PYTHONPATH=. python3 rhm/practice/antiphon/questions.py      # Q-1..Q-10
PYTHONPATH=. python3 rhm/practice/tutti/phase0_tutti.py      # the floor, the bound, the sizing

modal run rhm/practice/tutti/tutti.py::preflight \
    --arms "tu_pf_split,tu_pf_boot,tu_pf_q,tu_pf_yk,tu_pf_off"
modal run rhm/practice/tutti/tutti.py::preflight \
    --arms "tu_y_exo,tu_d_exo,tu_s_exo,tu_s_endo,tu_s_yk,tu_d_endo,tu_y_endo,tu_m_exo"
modal run rhm/practice/tutti/tutti.py::fidelity_smoke --tag tu_gf

python3 rhm/practice/tutti/launch_detached.py --fn tutti_run --tag tu_s0 \
    --eras "1:25:60,2:12:50,3:6:70,4:3:12,5:1:9" --era-caps "60,50,70,12,9" \
    --arms "tu_y_exo,tu_d_exo,tu_s_exo,tu_s_endo,tu_s_yk,tu_d_endo,tu_y_endo,tu_m_exo" \
    --tol-dsil 0.0046 --question-k 2048 \
    --max-macro-level 4 --endo-price 267 --gate-frac 0.5 \
    --span-tau-fire 0.50 --perf-alpha 0.05 --perf-tau-w 0.20 \
    --budget 8 --g-budget 482 --n-corrupt 1 --mine-cap 8 --gy-level 4 --entry-rec \
    --span-min-hold 128 --collect-task-matched --tm-episodes 8192 \
    --recert-every 5 --n-aud 192 --probe-every 8 --n-rt 384 --n-score 256 \
    --sil-cv 0.15 --lp-min-drop 0.10 --seed 0
# (the banked command is also `results/RUN_tu_s0.sh`)

python3 rhm/practice/tutti/analyze_tutti.py --tag tu_s0 --gf-tag tu_gf --fetch --figures
```

Every flag but `--arms`, `--tol-dsil` and `--question-k` is `ca_s0`'s, verbatim — which is what
makes `tu_y_exo` reproduce `ca_s0/dsil_yield` bit for bit.

**Two defects on the record.** (i) The donor's own cross-tag gates in reduction §0 (against
`cr3_s0`/`ma_s0`) print "not fetched locally — skipped": they are the *donor's* lineage checks,
superseded here by §0T, and re-enabling them needs only `crescendo/figures/cr3_s0` and
`maestro/figures/ma_s0` on disk. (ii) `L5` is closed in this tag and cannot be opened in it —
`max_macro_level` moves the shadow audition's `rng.permutation` on the shared stream, so both
cross-tag replays would die. See [`DESIGN.md`](DESIGN.md) §8 for the sizing lane's corrected
account and the re-keying direction that replaces a `tu_l5_*` tag.

## Children

| folder | what |
|---|---|
| [`sizing/`](sizing/SIZING.md) | **The offline sizing of the second rung and the deep-era question grip** (CPU, no GPU). `phase0_l5.py`: the exact level sizes to L6, the r² law at L5, the L5 arrival budget over every banked L4 book (40 arms, B-1′ 101/101), the L5/L6 gauge floors, the world and rule-draw sweeps, the parent-feature level-size arithmetic. `phase0_grip.py`: the structural grip law `(s−1)/s`, the dose law `κ/|T_(ℓ−1)|`, grip × era × parameterization, node choice priced, the damage-schedule counterfactual, the support lever. Record: `SIZING.md`; outputs `phase0.json`, `phase0_grip.json`. Authoritative over `DESIGN.md` §8 where they disagree. |

**Writeup**: this node's [`README.md`](README.md) is the round's joint writeup (five lanes: this
node, `sizing/`, `antiphon`'s `an_s2`/`an_m0`, `antiphon/trap/`, and `enharmonic/SPEC.md`).

[^private]: Not mirrored: this link points to a document in the private lab repo (the roadmap, the queue, an unrun spec, reading notes, or a conversation). See the top-level README for what is held back and why.
