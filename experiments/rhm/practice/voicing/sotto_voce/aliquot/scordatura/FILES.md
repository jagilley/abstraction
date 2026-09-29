# FILES — `scordatura`

The machinery record: every code file, every knob, every gate, every run, and the volume path of
every saved state. The decisions are in [`DESIGN.md`](DESIGN.md); the brief is [`SPEC.md`](SPEC.md).
No README yet (the brief's instruction: written after the results are discussed).

**Up**: [`../FILES.md`](../FILES.md) (`aliquot`) · **Donor**: [`../rubato/FILES.md`](../rubato/FILES.md)

## Code files

| file | purpose |
|---|---|
| `scordatura.py` | the substrate: `../rubato/rubato.py` forked at its `rb_g1d` head. Adds the junk (`mine_junk`; `sc_junk_count`, `sc_junk_draw`, the append after the solved draw on its own stream `SC_JUNK_SEED`), the shadow miners (`sc_shadow_of`; `_mj`, a new state name, R-0 over 76), the junk columns (`sc_truth_set`, `sc_key_true`, `sc_key_info`, `sc_level_stats`; on every walk step and pass, graded key, fired consumer, revocation, and per cycle in `log["mj"]`), the in-run checks J1–J4 (`mj_falsify` for their red runs), persistence (`adm_grade_persist`, `sc_persist`), the end-of-arm fire survey (`sc_fire_keys`, `sc_fire_cap`; a restore at the stop cycle now runs no cycle, so the survey can read any saved state), the designed gates G-J / G-P (`sc_gates`, also run by `rb_gates`), the preflight `sc_pf` (and a `donor` option in `rb_r1_run`'s multi role, which runs a job through `rubato.run_arm`), `fidelity_smoke` retargeted at `rubato.py` with the in-run G-J runs, and the arms `sc_gn_yk`, `sc_gw_yk`, `sc_grd_yk` and their preflight twins (built from the donors' specs by `_sc_arm`). Volume dir `rhm_practice_scordatura`, app `rhm-practice-scordatura`. Every addition `# [scordatura]`-marked |
| `reduce_scordatura.py` | local, CPU-only. `--dose --tag <pre-check tag> --seed S` → `figures/<tag>_dose.txt` ([D1] the junk mined, [D2] keys at support both ways against `st_gn_yk`, [D3] the walk and its cap, [D3b] the walk's own base-plus-one fires by class, [D4] the fire survey). `--grade --seed S` → `figures/sc_grade_s{S}.txt` ([G1] provenance and identity, [G2] task error by era against `st_gn_yk` and the junk floor, [G3] coverage and the level above blocked by a revoked half, [G4] the served table over cycles, [G5] revocations by kind and class, [G6] the two currencies on the same fires, [G7] the diet at the seam and the bank, [G8] persistence re-read, [G9] the bill, [G10] the fire surveys). `--seedtable` → `figures/sc_seedtable.txt` (both seeds side by side, never averaged). Reads rubato's mirrors (`../rubato/figures/`) for `st_gn_yk`, `rb_gw_yk`, `rb_grd_yk`, `rb_gr_yk` |
| `launch_detached.py` | rubato's session-isolated detached launcher, retargeted at `scordatura.py` |
| `fetch_compact.py` | rubato's, retargeted at `rhm_practice_scordatura`; also splits `log["quot"]` (the quotient's per-cycle accounting, ~90 MB of a graded junk arm's file, read by no reducer) into `quot.json.gz`, and leaves the `.npz`/`.pt` dumps on the volume |
| `results/wait_app.sh` | rubato's: waits until ALL listed apps leave the running states (run with `bash`) |
| `results/wait_any.sh` | waits until ANY listed app leaves them (used to fill a freed container slot) |
| `results/RUN_*.sh` | the commands of record (below) |

## The new knobs

All in `voicing_run`'s signature; each enters the config only when it is off its default, so with all
of them at default the config is rubato's.

| knob | level | default | meaning |
|---|---|---|---|
| `mine_junk` | arm (`sc_*`: 1.0) | 0 (off) | append the cycle's unsolved chosen answers after the solved draw at this multiple of the outcome-blind proportion (DESIGN §1.1–§1.2) |
| `mj_falsify` | run | `""` | J1–J4's red runs only: `rng` (the junk drawn on the shared stream), `solved` (drawn from every answer), `prepend` (put before the solved rows), `shadow` (the shadow fed every row) |
| `sc_fire_keys` | run, free at a restore | False | the end-of-arm fire survey (DESIGN §1.4) |
| `sc_fire_cap` | run, free at a restore | 48 | keys fired per class (junk-only, solved-sourced) per source and level |
| `adm_grade_persist` | arm (`sc_grd_yk`: 2) | 1 | revoke a worthless key only after this many consecutive non-positive priced passes (DESIGN §1.5) |

## Gates

| gate | where | claim |
|---|---|---|
| **G-F** | `fidelity_smoke` | every `[scordatura]` knob off, this file replays `rubato.py` bit for bit (smoke scale, donor self-replay control) |
| **G-F grade path** | `sc_pf` D0/D1 | the same on the preflight vehicle's world grade at margin 0.5, revocations firing |
| **G-J / G-P** | `sc_gates` (CPU; in `rb_gates`) | the dose, the draw, the shadow, junk-only, the stats; persistence — each on designed cases, 10 perturbations RED |
| **J1–J4** | in run, every cycle `mine_junk` is on | the shared stream untouched; every junk row unsolved; the solved rows first; each shadow fed exactly the solved rows |
| **c1 identity** | `fidelity_smoke` | the junk arm's cycle 1 IS the knob-off arm's in every series the mining does not feed |
| inherited | rubato's R-0, R-2, R-1, G-I, G-W, G-S, G-D, G-O, G-N | unchanged |

## Arms

| arm | = | restored from | role |
|---|---|---|---|
| `sc_gn_yk` | `st_gn_yk` + `mine_junk` 1.0 | — (from c0) | the ungated junk floor |
| `sc_gw_yk` | `rb_gw_yk` + `mine_junk` 1.0 | the floor's c10 (seed 0), c5 (seed 2) | the world grade, the ceiling as built |
| `sc_grd_yk` | `rb_grd_yk` + `mine_junk` 1.0 + `adm_grade_persist` 2 | the same | the read grade with its diet and persistence |
| `sc_pf_gn`, `sc_pf_gw`, `sc_pf_gwp`, `sc_pf_grd` | the rubato preflight twins + the junk | — | preflight only |

## Runs

| tag | function | what | app id |
|---|---|---|---|
| `sc_gf1` | `fidelity_smoke` | G-F: max\|fork − rubato\| = 0.000e+00 (control 0), commits equal; in-run G-J on the smoke substrate: clean green, `rng` and `shadow` RED, `solved` and `prepend` BLIND (no instance solved in the 3 cycles); c1 identity equal, the `rng` knob moves c1's value loss | `ap-km6iwcNJuJoWTWEYgA0suj` |
| `sc_pf1` | `sc_pf` | the preflight, seven runs in one container: J clean; G-I equal at c2; the arms-of-record restore path (R-2, schema evolution); persistence (23 held, every revocation at run 2); D0 = D1 on the grade path. `sc_pf.json` on the volume | `ap-LzFcJ4PySByXxPW7sByckE` |
| `sc_pf2` | `rb_r1_run` (multi) | J1–J4's red knobs on gate R-1's vehicle (which solves): 4/4 RED, each on its own check only | `ap-GgkYKszaafrfD8wkjKhxQ9` |
| `sc_p1` | `sweep` | the pre-check, seed 0: `sc_gn_yk` c1–c40, saves every 5, survey at c40; preempted after c5 and auto-resumed from its c5 (`RUN_sc_p1.sh`). `figures/sc_p1/`, `figures/sc_p1_dose.txt` | `ap-ZEt3QV4MskVCh6HEF7ThFp` |
| `sc_p2` | `sweep` | the seed-2 floor's first segment, the same (`RUN_sc_p2.sh`). `figures/sc_p2/`, `figures/sc_p2_dose.txt` | `ap-8Lq665cCq0TPjUWhvjPK5T` |
| `sc_s1` | `sweep` | the seed-0 floor c41–c201 from `sc_p1`'s c40, survey at c201; preempted after c170 and auto-resumed from its own c170 (`RUN_sc_s1.sh`). `figures/sc_s1/` | `ap-CIiBcAwMAcNqwPJuirsOf4` |
| `sc_s2` | `sweep` | the seed-2 floor c41–c201 from `sc_p2`'s c40; preempted after c170 and auto-resumed from its own c170 (`RUN_sc_s2.sh`). `figures/sc_s2/` | `ap-cKMiPXBu2v7xkaeKB4AvzE` |
| `sc_g1` | `sweep` | seed 0: `sc_gw_yk` and `sc_grd_yk` from `sc_p1`'s c10, two containers (`RUN_sc_g1.sh`). `figures/sc_g1/` | `ap-0fRzRL9i3QnsNzIcEHm4Rq` |
| `sc_g2w`, `sc_g2r` | `sweep` | seed 2: `sc_gw_yk`, `sc_grd_yk` from `sc_p2`'s c5, one container each (`RUN_sc_g2w.sh`, `RUN_sc_g2r.sh`). `figures/sc_g2w/`, `figures/sc_g2r/` | `ap-JzJg9RVEYGYeNxqT9g3Rlm`, `ap-lEOVcFPqmCtEYgw9kX5a0S` |

## The saved states on the volume

`rhm-scaling-data:/rhm_practice_scordatura/<tag>__<arm>/<arm>/ck/c{NNN}.pt` (the state) and
`c{NNN}.json` beside it (metadata, world fingerprint, digest v2 and hash tree). A sweep's merged
`<tag>/<arm>/` holds the arm file only.

| tag / arm | cycles saved | directory |
|---|---|---|
| `sc_p1` / `sc_gn_yk` (seed 0, c1–c40) | 5, 10, …, 40 | `rhm_practice_scordatura/sc_p1__sc_gn_yk/sc_gn_yk/ck/` |
| `sc_s1` / `sc_gn_yk` (seed 0, c41–c201) | 50, 60, …, 200, the era boundaries (60, 110, 180 on the cadence; 192, 201) | `rhm_practice_scordatura/sc_s1__sc_gn_yk/sc_gn_yk/ck/` |
| `sc_p2` / `sc_gn_yk` (seed 2, c1–c40) | 5, 10, …, 40 | `rhm_practice_scordatura/sc_p2__sc_gn_yk/sc_gn_yk/ck/` |
| `sc_s2` / `sc_gn_yk` (seed 2, c41–c201) | as `sc_s1` | `rhm_practice_scordatura/sc_s2__sc_gn_yk/sc_gn_yk/ck/` |
| `sc_g1` / `sc_gw_yk`, `sc_grd_yk` (seed 0) | 20, 30, …, 200, 192, 201 | `rhm_practice_scordatura/sc_g1__sc_gw_yk/sc_gw_yk/ck/`, `…/sc_g1__sc_grd_yk/sc_grd_yk/ck/` |
| `sc_g2w` / `sc_gw_yk`, `sc_g2r` / `sc_grd_yk` (seed 2) | 10, 20, …, 200, 192, 201 | `rhm_practice_scordatura/sc_g2w__sc_gw_yk/sc_gw_yk/ck/`, `…/sc_g2r__sc_grd_yk/sc_grd_yk/ck/` |

## Figures (the record, all local)

| file | what |
|---|---|
| `figures/sc_p1/`, `figures/sc_p2/`, `figures/sc_s1/`, `figures/sc_s2/`, `figures/sc_g1/`, `figures/sc_g2w/`, `figures/sc_g2r/` | compact mirrors (`results.json` without `log["entry"]` and `log["quot"]`, which sit beside it gzipped; `setup.json.gz`, `summary.json`, `sweep.json`) |
| `figures/sc_p1_dose.txt`, `figures/sc_p2_dose.txt` | the dose read, per seed |
| `figures/sc_grade_s0.txt`, `figures/sc_grade_s2.txt` | the graded arms on the junk floor, per seed |
| `figures/sc_seedtable.txt` | THE TWO-SEED TABLE |
