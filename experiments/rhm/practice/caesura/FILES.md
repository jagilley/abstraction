# FILES — `caesura` (E′ round 3: δ-silence as the commit decision)

Machinery record for the node. **No results are interpreted here** — the reduction's numbers go
to the orchestrator and are discussed before any writeup (repo norm). There is deliberately no
`README.md` yet for the same reason.

**Up**: parent arc [`../README.md`](../README.md) ·
`../../../../ROADMAP.md`[^private] §7.1.3 (Track E′)
**Idea doc**: [`practice_manufactures_its_own_credit`](../../../../ideas/practice_manufactures_its_own_credit.md)
§1 component (3) — **δ-silence as practice's own commit trigger**, the op this signal was
originally designed for and which the A-track thermostat has never read.
**Direct donor** (untouched): [`../intonation/`](../intonation/FILES.md) — E′ round 2, which
built the per-macro-slot performance error and its context-conditional benchmark `b(s)` and
consumed it on the execution side only. `intonation.py` is forked here.
`maestro/policy.py` is **imported, not forked** — *this round adds no outer-loop rule; it adds
one more GAUGE for A1's existing thermostat to read*, which is the structural form of the claim.
`native/span/span_net.py` is imported through the donor, unmodified.
**Through them**: `../crescendo/` (A3), `../maestro/` (A2), `../conductor/` (A1),
`../census/`, `../assay/`, `../tacet/` (E3′).
**The prior that made this askable**: [`../crystallize/`](../crystallize/README.md) and
[`../ratchet/`](../ratchet/README.md) used δ-silence gates on the executor's **own movement**;
what has never existed on RHM until `intonation` is a δ to be silent.

## The question, and the one thing that varies

A1's thermostat paces commit and hold on the `yield` stream — distinct next-level tuples at
support, a **one-level-up** count. This round asks whether the metering signal can pace the
**crossing** instead, or only its **timing**.

**The gauge** (`dsil`), in the panel's error convention (lower is better) — nothing new is
computed and nothing new is priced, because the meter already maintains it:

```
dsil = mean over the OPEN slots of b(s)
```

Read through **A1's `QuietPolicy` unchanged** (span 1, W 4, burn 4, alpha 0.5): hold while it
still moves, act when it quiets inside its own measured dead zone.

**The type structure, stated and not assumed away.** At the cycle level ℓ is committed the
level-ℓ slots **do not exist** — they are minted *by* that commit. So `dsil` only ever speaks
about levels ≤ ℓ−1: a **within-level** execution signal being asked to license a **crossing**,
which is what `teacher_slot` measured the within-level *ledger* refusing. `b(s)` is a different
within-level quantity, so the refusal is a question, not a prediction.

## The design flaw gate D caught, and how it is handled

`dsil` requires open slots; slots require a commit; a `dsil`-driven arm therefore **deadlocks by
construction** — the donor's own "read undefined this cycle" guard correctly skips the
thermostat every cycle, the loop never acts, no slot is ever minted, forever. Gate D-1 shows it
directly at every scale (`dsil_read` at preflight: 0 cycles with `dsil`, 0 open slots, 0
commits). This is not a scale artifact.

Handled deliberately rather than by dropping the arm, because the flaw **is** the type
structure: **δ-silence cannot license the first crossing; it can only license crossings above
the level whose executor it can hear.** While the gauge is absent, `dsil_read` falls back to the
arc's own default rule (`delta_prov`, certify-else-boundary) under `dsil_bootstrap`, and every
commit taken that way is logged `kind: "bootstrap"` so the reduction never counts it as a
δ-silence firing. Gate D-5 asserts the bootstrap fires, that a slot opens after it, and that no
bootstrap commit was ever taken while the gauge *was* available.

## Arms

| arm | pacer | role |
|---|---|---|
| `dsil_sched` | schedule (`delta_prov`), era lengths **=** the caps | the carrier, and the **lifetime ceiling** — no loop arm can outrun it |
| `dsil_yield` | A1's thermostat on `yield` | the comparator; `intonation`'s `perf_log` verbatim |
| `dsil_read` | A1's thermostat on **`dsil`** (+ bootstrap) | δ-silence **chooses** the crossing |
| `dsil_and` | `yield` thermostat + **δ-silence veto** | δ-silence may only **delay** — the arm that separates the signal's *seat* |
| `dsil_pf_gauge`, `dsil_pf_veto`, `dsil_pf_boot` | preflight only | gate D's coverage (see below) |

**Deliberately NOT yoked** — the one departure from `intonation`. The question here *is* the
pacing, so pinning the arms to a common clock would delete the variable. The cost is the one
A1/A2 carried and named (arms are not lifetime-matched, so deep-era deltas conflate pacing with
practice time) and it is bought back A3's way: the schedule arm's era lengths are the caps.

The **veto** is a conjunction, not a rule: A1's own `QuietPolicy` on the `dsil` read, stepped
read-only, and a commit the yield thermostat licensed is deferred on any cycle the executor is
still moving. A deferral **does not spend the rule's firing** — the commit re-fires as soon as
the executor quiets. Where the gauge does not exist the veto is **inert by construction**: a
signal that cannot speak must not be read as saying no.

## The dead zone, measured and not chosen

`QuietPolicy` raises without a floor, which is the property that keeps this measured. `dsil`'s
was derived by **A1's own null-ABBA statistic** (`policy.null_abba`, span 1, W 4,
`v_tol = sd/√W`) on the `dsil` series **reconstructed offline from `in_s0`** — `dsil` is a pure
function of `log["perf"][i]["bench"]` and `log["span"][i]["open"]`, both already logged, so the
floor cost **zero GPU**. Pooled over five arms: **`tol_dsil = 0.00321442`** (per-arm
0.00225–0.00447, within 2×). Re-derived **in-tag** on `ca_s0/dsil_sched`: **0.00492**, i.e. the
run was governed by a floor 1.53× *tighter* than in-tag noise warrants — the narrower claim is
the in-tag number and it is printed in reduction §6.

`dsil` joins `PANEL_KEYS`, which is what makes §6 measure it and §7 replay it through A1's
unchanged rule in **every** arm, so the "when would δ-silence have fired here" counterfactual
exists for the yield-paced and schedule arms too, at no extra cost. §6's blanket skip for a
series containing `None` is relaxed for `dsil` only, on its first-live-cycle window, with the
offset printed.

## Code files

| file | purpose |
|---|---|
| `caesura.py` | The substrate. Forks `../intonation/intonation.py`; every addition marked `# [caesura]`. New: the `dsil`/`dsil_all`/`dsil_n`/`dsil_open` panel entries, the read-only veto detector (A1's `QuietPolicy` on `dsil`), the veto at the commit site with its per-consultation event log, the `dsil_bootstrap` branch, `tol_dsil`/`dsil_veto`/`dsil_bootstrap` cfg knobs, the four arms plus three preflight arms, and gate **D**. Entrypoints: `preflight`, `fidelity_smoke`, `caesura_run`. |
| `analyze_caesura.py` | The reduction. The donors' sections, plus `dsil` in `PANEL_KEYS` (so §6 and §7 cover it) and §D: the gauge's trajectory, the commit cycles it paced against A1's and the schedule's, the veto's per-consultation ledger, and the era table. |
| `launch_detached.py` | Session-isolated detached launcher (the donor's, retargeted). |

## Gates

| gate | what it asserts | status |
|---|---|---|
| **D-1** | the gauge is in **every** arm's panel and is present *exactly* when a slot is open | **PASS** |
| **D-2** | the driven read is what the arm names, and only that arm drives on it | **PASS** |
| **D-3** | the veto fires **and is inert where the signal cannot speak**, and was consulted at least once | **PASS** — 6 defer / 0 pass / 1 absent at toy scale |
| **D-4** | the schedule arm carries no veto | **PASS** |
| **D-5** | the bootstrap fires, a slot opens after it, and no bootstrap commit was taken while the gauge was available | **PASS** — 1 commit (L2 c6, boundary), gauge alive c7, 16 slots |
| G-F | with every `# [caesura]` knob off, this fork replays **`intonation.py`** in process | **PASS — 0.000e+00** vs a 0.000e+00 self-replay control, commits equal, both donor arms |
| in-tag twin | every arm vs `dsil_sched` up to first action | **PASS — 0.000e+00**, all three |
| cross-tag (free, unplanned) | `dsil_yield` reproduces `in_s0/perf_log` — commits @49/72/104, advances @58/88/114/122/131, certs L2 c18 / L3 c76, identical whole-run 2×2 | **PASS** over 131 cycles at full scale |
| P-1…P-12, L-1…L-8, C, T, I | the donors' suites, re-run | ALL PASS |

Three preflight-only arms exist because a loop arm cannot arm on a forty-step substrate (the
donors' own gate-C note): `dsil_pf_gauge` (true tables → gauge live from c1, exercises the
driving path), `dsil_pf_veto` (schedule commits with seeded miners → exercises the `absent` and
`defer`/`pass` branches), `dsil_pf_boot` (exercises the bootstrap).

**Name caution** (extending the donors'): this substrate now carries *five* unrelated "gates" —
census's L2 admission gate, span's parity gate (`span_tau`, `span_tau_fire`), tacet's learning
gate (`gate_mode`), intonation's agency gate (`perf_g0`/`perf_theta`, a σ), and this round's
δ-silence **veto** (`dsil_veto`, a conjunction). None reads a key belonging to another.

## Volume layout

`rhm-scaling-data:/data/rhm_practice_caesura/<tag>/{setup.json,summary.json,<arm>/results.json,done.txt}`.
Fetched copies, figures and `reduction.txt` under `figures/<tag>/`.

## Runs

| tag | what | outcome |
|---|---|---|
| `_preflight` | interface + gates C, T, I and **D**, toy sizes | **ALL PASS** (`results/preflight3.log`); two earlier passes caught the detector's missing `None` guard and then the deadlock |
| `ca_gf` | G-F: in-process fork-vs-`intonation.py` replay at `max_macro_level=4` | **PASS — 0.000e+00** (`results/gf.log`) |
| `ca_s0` | the main run: 4 arms, A3's ladder/caps/floors, one seed, `--tol-dsil 0.00321442` | clean, **10,150 s = 2.82 GPU-h**; `dsil_sched` 201 cycles / 15.6 s-cyc, `dsil_yield` 131 / 14.8, `dsil_read` 176 / 15.9, `dsil_and` 114 / 13.8. Record `figures/ca_s0_reduction.txt` |

## Reproduce

```bash
cd experiments/            # MODAL_PROFILE=chromatic

PYTHONPATH=. python3 rhm/practice/maestro/policy.py       # the offline policy suite

modal run rhm/practice/caesura/caesura.py::preflight \
    --arms "dsil_pf_gauge,dsil_pf_veto,dsil_pf_boot,dsil_sched,dsil_yield,dsil_read,dsil_and"
modal run rhm/practice/caesura/caesura.py::fidelity_smoke --tag ca_gf

python3 rhm/practice/caesura/launch_detached.py --fn caesura_run --tag ca_s0 \
    --eras "1:25:60,2:12:50,3:6:70,4:3:12,5:1:9" --era-caps "60,50,70,12,9" \
    --arms "dsil_sched,dsil_yield,dsil_read,dsil_and" \
    --tol-dsil 0.00321442 \
    --max-macro-level 4 --endo-price 267 --gate-frac 0.5 \
    --span-tau-fire 0.50 --perf-alpha 0.05 --perf-tau-w 0.20 \
    --budget 8 --g-budget 482 --n-corrupt 1 --mine-cap 8 --gy-level 4 --entry-rec \
    --span-min-hold 128 --collect-task-matched --tm-episodes 8192 \
    --recert-every 5 --n-aud 192 --probe-every 8 --n-rt 384 --n-score 256 \
    --sil-cv 0.15 --lp-min-drop 0.10 --seed 0

python3 rhm/practice/caesura/analyze_caesura.py --tag ca_s0 --gf-tag ca_gf --fetch --figures
```

Every flag but `--arms` and `--tol-dsil` is `in_s0`'s, verbatim — which is what makes
`dsil_yield` reproduce `intonation`'s baseline bit-for-bit.

**One cosmetic defect on the record**: `fidelity_smoke`'s printed strings still say "tacet"
where the donor is now `intonation` (the import and the comparison are correct; only the label
is stale).

[^private]: Not mirrored: this link points to a document in the private lab repo (the roadmap, the queue, an unrun spec, reading notes, or a conversation). See the top-level README for what is held back and why.
