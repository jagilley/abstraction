# FILES — `tutti/sizing` (offline sizing for the second extension and the deep-era question)

Machinery record for the sizing lane. **No results are interpreted here** — the measurements go
to the orchestrator, who discusses them with Jasper (repo norm). The numbers themselves are in
[`SIZING.md`](SIZING.md).

**Up**: [`../DESIGN.md`](../DESIGN.md) (the unification node; its §8 "the L5 flag — designed, not
opened" is the consumer of this lane's result) · parent arc [`../../README.md`](../../README.md) ·
`QUEUE.md`[^private] "F3" · `ROADMAP.md`[^private] §1.4
**Donors** (imported, never edited):
[`../../crescendo/phase0_l4.py`](../../crescendo/FILES.md) — `buildable()`, `key_stream`,
`analyse_arm`'s shape and gate B-1, all extended one level here ·
[`../../antiphon/phase0_question.py`](../../antiphon/FILES.md) — the parse-ambiguity/junk-mass
measurement, the true-key marginals, the random-draw calibration, reach-vs-K and the r² payoff
curve ·[`../../antiphon/questions.py`](../../antiphon/FILES.md) — `target_geometry`, imported
directly so the grip law is the substrate's own arithmetic and not a restatement of it ·
[`../../conductor/policy.py`](../../conductor/FILES.md) — `null_abba`, `_sd`, `QuietPolicy`,
imported so the L5/L6 floors are derived by the arc's own method ·
[`../../ratchet/macros.py`](../../ratchet/FILES.md) — `Miner.build`, `true_tables`, `base_table`,
`exact_features` · `rhm/rhm_data.py`, `rhm/rhm_sculpt_precheck.py` — the DGP.

CPU only, no Modal, no GPU, no substrate. Both scripts run from `experiments/`.

## Code files

| file | purpose |
|---|---|
| `phase0_l5.py` | **The second extension, sized.** Exact level sizes L1–L6 (L6 by inclusion–exclusion, since 1.3e10 rows cannot be enumerated); gate B-1′ (the `Miner.build` replay over all 40 banked arms, 101 commit events including the L4 rung A3's donors could not reach); the buildable-L5 ceiling over each arm's frozen / live / true L4; the r² payoff curve at L5; the L5/L6 observation streams as logged and the demand-cell test on the L5 mining node; the DGP's key marginal and junk mass at every mining node L2–L6; gate G-1 (the random-draw model calibrated against the logs); the observation budget an L5 book costs; the arrival-vs-ratchet wall decomposition; `tol_yield_l5`/`tol_yield_l6` derived by null-ABBA plus A1's thermostat replayed on the L5 series; and the world-sizing arithmetic with three candidate worlds' L5 budgets measured. Writes `phase0.json`. |
| `phase0_grip.py` | **The deep-era question determinator.** The structural-grip law swept over era × `max_macro_level` (from the imported `target_geometry`); grip in blocks vs bits per level; the delivered dose measured per era per arm on `an_s0`'s own question log and the κ/\|T_(l−1)\| law it obeys; the grip × era × parameterization table over eight candidate question parameterizations; node choice priced against the per-node true-key marginals; menu reach vs K extended to L5; the damage-schedule counterfactual (narrowed deep-era cell) with its demand-coverage cost; and the support-lever sizing from the G-Y at-support histogram. Writes `phase0_grip.json`. |
| `phase0.json` | `phase0_l5.py`'s output — gate verdicts, per-arm ceilings, curves, budgets, floors, worlds. |
| `phase0_grip.json` | `phase0_grip.py`'s output — the grip grid, the dose law, the parameterization table, node/reach/counterfactual/support tables. |

## Auxiliary READMEs

| file | summary |
|---|---|
| `SIZING.md` | What was measured, the gate verdicts, the tables, and the four premise corrections. Facts, not interpretation. |

## Reproduce

```bash
cd experiments/          # CPU only; no MODAL_PROFILE needed

PYTHONPATH=. python3 rhm/practice/tutti/sizing/phase0_l5.py     # ~4 min
PYTHONPATH=. python3 rhm/practice/tutti/sizing/phase0_grip.py   # ~1 min
```

Both read only fetched mirrors already on disk under
`rhm/practice/{crescendo,antiphon,caesura,intonation,maestro,conductor}/figures/<tag>/<arm>/results.json`.
Nothing was fetched from the `rhm-scaling-data` volume for this lane; the corpus was complete
locally.

## The corpus

40 banked arms across 9 tags, every one carrying the observation panel (`obs_panel`) so the L5
and L6 series exist:

| tag | arms | `max_macro_level` | role here |
|---|---|---|---|
| `cr3_s0` | 5 | 4 | A3's main run — the L4 books the L5 ratchet is measured over |
| `cr3_s1` | 2 | 4 | A3's displaced twins |
| `an_s0` | 7 | 4 | the question port — the `log["q"]` dose ledger, and the arc's best L4 book (`q_bisect`) |
| `an_s1` | 1 | 4 | the pacer-only certification arm |
| `ca_s0` | 4 | 4 | the δ-silence pacer — the most-trusted L4 commits, and the corpus's largest L5 at-support count (`dsil_read`, 11) |
| `in_s0` | 5 | 4 | `intonation`'s live-executor arms |
| `ma_s0` (intonation) | 5 | 4 | the metered-data arms |
| `cd_s0` | 6 | 3 | A1 — kept so the L2/L3 numbers stay comparable to A3's own Phase 0 |
| `ma_s0` (maestro) | 5 | 3 | A2 — same reason |

[^private]: Not mirrored: this link points to a document in the private lab repo (the roadmap, the queue, an unrun spec, reading notes, or a conversation). See the top-level README for what is held back and why.
