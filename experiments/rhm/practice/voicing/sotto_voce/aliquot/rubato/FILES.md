# FILES — `rubato`

The machinery record: every code file, every knob, every gate, every run, and the volume path of
every saved state. The decisions are in [`DESIGN.md`](DESIGN.md); the brief is [`SPEC.md`](SPEC.md).
Writeup: [`README.md`](README.md) (2026-09-24); this node's record is facts only (DESIGN §4, §11).

**Up**: [`../FILES.md`](../FILES.md) (`aliquot`) · **Donor**: [`../sostenuto/FILES.md`](../sostenuto/FILES.md)

## Code files

| file | purpose |
|---|---|
| `rubato.py` | the substrate: `../sostenuto/sostenuto.py` forked at its `st2_s1`/`st2_s2` head. Adds the saved state (`_RB_STATE` and its sibling lists, `rb_registry`, `rb_local_classes`, `_RbHasher` / `rb_digest` / `rb_tree_diff`, `_RbPickler` / `_RbUnpickler`, `rb_write` / `rb_read`, `rb_save` / `rb_restore` / `rb_take`), the restore statement and the save in `run_arm`, the six `rb_*` knobs, gate R-0 (`rb_static_gate`, `rb_static_falsify`, `rb_gates`) and gate R-1 (`rb_r1_run`, `rb_gate_r1`); lifts `preflight`'s config into `_preflight_cfg` verbatim; `fidelity_smoke` retargeted at `sostenuto.py`. Every addition `# [rubato]`-marked; every knob default off [phase 2] The admit-then-grade op in `run_arm`'s block (g2.5) (`rb_grade_decide`, `rb_reoffer_new`, `rb_consumers`, gates G-D/G-O in `rb_grade_gates`, the in-run G-I/G-W/G-S, the arms `rb_gr_yk` / `rb_gw_yk` and their preflight `rb_pf_grade`), the versioned digest (`_RB_DIGEST_V`), `rb_auto_resume` and the `{arm}` resume template, `rb_hist` in every save's metadata, and the analysis entrypoints `rb_consumer_survey` (`which=admitted|revoked`), `rb_peek_state`, `rb_diff_arms`, `rb_peek_grade`, `rb_host_survey`. [r3] The per-span diet (`adm_grade_diet`, `no_diet`, gate G-N) and the arm `rb_grd_yk` |
| `reduce_rubato.py` | local, CPU-only: `--bank-check` (a re-run arm against its banked `sostenuto` mirror, key by key and series by series, with the save record's sizes and times) -> `figures/<tag>_bank_check.txt`; `--r1` (gate R-1's record as a table) -> `figures/<tag>_r1.txt`; `--grade` [phase 2] (`sostenuto`'s own reduction with the graded arms as two more columns, then [L] the grade: revocations, re-offers, confusion of the two currencies on the same fires, gains, served keys, era errors, and G-I off the log and off the records) -> `figures/<tag>_reduction.txt`; `--seedtable` [phase 2] (`sostenuto`'s `mk_seedtable` with the graded arms as two more rows, plus (v) the grade per seed: revocations, re-offers, the level above blocked by a revoked half) -> `figures/rb_grade_seedtable.txt` |
| `launch_detached.py` | the session-isolated detached launcher, retargeted at `rubato.py` |
| `fetch_compact.py` | fetch a tag from the volume and write the compact local mirror, retargeted at `rhm_practice_rubato`. A merged tag holds no saved states (`sweep` does not copy `ck/`), so nothing large is fetched |
| `results/wait_app.sh` | `sostenuto`'s, copied: waits on `modal app list`. **Run it with `bash`**, not `sh` (it uses arrays) |
| `results/RUN_rb_r1.sh` | gate R-1 of record |
| `results/RUN_rb_s1.sh`, `results/RUN_rb_s2.sh` | the reference re-runs of record, seeds 0 and 2 |
| `results/RUN_rb_g1.sh`, `results/RUN_rb_g1_resume.sh` | [phase 2] the seed-0 graded arms of record: from `rb_s1`'s c20, then (after a preemption) from their own c130 |
| `results/RUN_rb_g2.sh` | [phase 2] the seed-2 graded arms of record, from `rb_s2`'s c20, `rb_auto_resume` on |
| `results/RUN_rb_g1d.sh` | [r3] the diet-gated read arm `rb_grd_yk`, seed 0, from `rb_s1`'s c50 |

## Docs

| file | purpose |
|---|---|
| `SPEC.md` | the orchestrator's brief, verbatim |
| `DESIGN.md` | the decisions (what is state and why, the file, the digest, where the save sits, the fork contract), the gates, the results (§4: R-1, the floor, both reference re-runs, both seeds' graded arms and their re-reading), the defects beside their corrections (§5), phase 2's op (§10) and round 3's diet (§11) |
| `FILES.md` | this file |

## The new knobs

All in `voicing_run`'s signature (so `sweep --flags-json` carries them), all default off, and
with all of them off the file is `sostenuto.py` (G-F).

| knob | default | meaning |
|---|---|---|
| `rb_ck_every` | `0` | save at every cycle divisible by this |
| `rb_ck_era` | `False` | save at every era boundary |
| `rb_stop_cycle` | `0` | stop after this cycle (and save there); the battery, dumps and arm file then run on that state |
| `rb_resume` | `""` | the saved state to restore, relative to `rhm_practice_rubato/` or absolute |
| `rb_allow_cfg` | `""` | config keys a fork may change against the saved run's (DESIGN §1.8) |
| `rb_falsify` | `""` | R-1's red runs only: `rng:<stream>` / `local:<name>` left unrestored |
| `rb_ref_ck` | `""` | [phase 2] a reference arm's saved-state directory: every save is compared with the reference's save at the same cycle by hash (gate G-I) until the arm's first revocation |
| `rb_auto_resume` | `False` | resume from the arm's OWN latest complete save in its out directory when it is later than `rb_resume`'s (a preempted input restarted from its start continues instead of re-running; DESIGN §5.5). `rb_resume` may name `{arm}` |
| `adm_grade` | **arm**, absent | [phase 2] `"read"` / `"world"`: admit-then-grade on top of `admit="none"` (DESIGN §10) |
| `adm_grade_obs` | `160` | [phase 2] the window, in the level-above panel's own observations (DESIGN §10.6) |
| `adm_grade_margin` | `0.0` | [phase 2] the dial: keep iff the best consumer's gain exceeds it |
| `adm_grade_cons` | `4` | [phase 2] consumers fired per key per pass (the best-supported) |
| `adm_grade_falsify` | `""` | [phase 2] the preflight's red runs only: `touch`, `node`, `serve` |
| `adm_grade_diet` | `0` (off); **512** on `rb_grd_yk` | [r3] the read grade's per-span diet: a key's consumers are priced only once the readout's fitted buffer holds this many rows at their span; below it the key is kept silently (`no_diet`). DESIGN §11 |

## The saved state on the volume

`rhm-scaling-data:/rhm_practice_rubato/<tag>__<arm>/<arm>/ck/c{NNN}.pt` (the state) and
`c{NNN}.json` beside it (metadata, world fingerprint, digest and hash tree). A `sweep` tag's merged
directory `<tag>/<arm>/` holds the arm file; the states stay in the per-container directory, and
`<tag>/summary.json["ck_dirs"]` and every `rb.saves[*].path` in the arm file point at them.

| tag / arm | cycles saved | total | directory |
|---|---|---|---|
| `rb_s1` / `st_gn_yk` (seed 0) | every 10th cycle 10–200, plus the era boundaries 192 and 201 (60, 110 and 180 are both) — 22 states | 1.24 GB | `rhm_practice_rubato/rb_s1__st_gn_yk/st_gn_yk/ck/` (+ `consumer_survey.json`, `consumer_survey_full.json`) |
| `rb_s2` / `st_gn_yk` (seed 2) | the same 22 cycles | 1.24 GB | `rhm_practice_rubato/rb_s2__st_gn_yk/st_gn_yk/ck/` (+ `consumer_survey.json`, `consumer_survey_full.json`) |
| `rb_g1d` / `rb_grd_yk` (seed 0, diet) | every 10th cycle c60–c200, plus c192 and c201 — 17 states | ~1.1 GB | `rhm_practice_rubato/rb_g1d__rb_grd_yk/rb_grd_yk/ck/` (+ `revoked_survey.json`) |
| `rb_g2` / `rb_gr_yk`, `rb_gw_yk` (seed 2, graded) | every 10th cycle c30–c200, plus c192 and c201 — 20 states per arm | ~1.2 GB per arm | `rhm_practice_rubato/rb_g2__rb_gr_yk/rb_gr_yk/ck/`, `rhm_practice_rubato/rb_g2__rb_gw_yk/rb_gw_yk/ck/` (+ `revoked_survey.json` each) |
| `rb_g1` / `rb_gr_yk`, `rb_gw_yk` (seed 0, graded) | every 10th cycle c30–c200, plus c192 and c201 — 20 states per arm (c30–c130 from segment 2, c140– from segment 3) | ~1.2 GB per arm | `rhm_practice_rubato/rb_g1__rb_gr_yk/rb_gr_yk/ck/`, `rhm_practice_rubato/rb_g1__rb_gw_yk/rb_gw_yk/ck/` |

Every save's digest version is in its `c{NNN}.json` (`meta.digest_v`). Absent on every `rb_s1` and
`rb_s2` state: `rb_s1`'s are v1, `rb_s2`'s are v2, and a restore resolves an absent version by trial
(DESIGN §5.4).

## Figures (the record, all local)

| file | what |
|---|---|
| `figures/rb_r1/r1_gate.json`, `figures/rb_r1_r1.txt` | gate R-1 of record (DESIGN §4.1) |
| `figures/rb_s1/`, `figures/rb_s2/` | compact mirrors of the reference re-runs (`setup.json.gz`, the arm file, the entry record, the readout's banked rows and heads) |
| `figures/rb_s1_bank_check.txt`, `figures/rb_s2_bank_check.txt` | the re-runs against the banked `st_s1` / `st_s2`: BIT-IDENTICAL on every shared quantity |
| `figures/rb_s1_consumer_survey.json`, `figures/rb_s2_consumer_survey.json` | every admitted key of the ungated re-runs at every save c30–c201: age, consumers at support (M-6) |
| `figures/rb_g1/`, `figures/rb_g2/`, `figures/rb_g1d/` | compact mirrors of the graded arms |
| `figures/rb_g1_reduction.txt`, `figures/rb_g2_reduction.txt`, `figures/rb_g1d_reduction.txt` | `sostenuto`'s [A]..[K] with the graded arms as columns, [L] the grade, [M] its record re-read (the dial, persistence, the diet, re-offers, the no-consumer revocations against `st_gn_yk`, the bill). `rb_g1d`'s carries `rb_grd_yk` beside `rb_g1`'s two |
| `figures/rb_g1_diet.json`, `figures/rb_g2_diet.json`, `figures/rb_g1d_diet.json` | the readout's row buffer by span at every save of each graded arm (`rb_peek_state`) |
| `figures/rb_g1_revoked_survey.json`, `figures/rb_g2_revoked_survey.json`, `figures/rb_g1d_revoked_survey.json` | every revoked key of each graded arm at every later save: in the live build, and its consumers (`rb_consumer_survey --which revoked`) |
| `figures/rb_grade_seedtable.txt` | THE TWO-SEED TABLE OF RECORD: `sostenuto`'s `mk_seedtable` with the graded arms as more rows, (v) the grade per seed, and every section [A]..[M] of both seeds' reductions side by side |

## Gates

| gate | where | claim |
|---|---|---|
| **G-F** | `fidelity_smoke` | with every `[rubato]` knob off this file replays `sostenuto.py` bit for bit, against a donor self-replay control |
| **R-0** | `rb_static_gate` (+ `rb_static_falsify`) | the state list is complete and checked against the source (DESIGN §1.2) |
| **R-2** | every restore | the restored graph hashes to the saved one; config and world match |
| **R-1** | `rb_gate_r1` | saved at c, restored fresh, continued: the uninterrupted arm, bit for bit; falsified by an unrestored generator |
| **bank** | `reduce_rubato.py --bank-check` | the re-run with saved states on is the banked arm, bit for bit |
| **G-D / G-O** | `rb_grade_gates`, CPU | [phase 2] the grade's rule and the re-offer rule on designed cases; 5/5 perturbations RED |
| **G-I** | every save of a graded arm | [phase 2] the arm IS the ungated arm by hash until its first revocation (and the reducer's log-based form for the paid arms) |
| **G-W** | once per graded arm, in-run | [phase 2] the grade's fire IS the donor's audition, instance by instance |
| **G-S** | after every revocation | [phase 2] a revoked key is served nowhere |
| **G-N** | `rb_grade_gates`, CPU | [r3] the diet: silent below it on a read, never on the world or on want of a consumer; 3/3 perturbations RED |

## Arms (phase 2)

| arm | grade | restored from | role |
|---|---|---|---|
| `rb_gr_yk` | `read` | `rb_s1__st_gn_yk/st_gn_yk/ck/c020.pt` (seed 0), `rb_s2__st_gn_yk/st_gn_yk/ck/c020.pt` (seed 2) | the projection's level on the fired consumers; zero world queries |
| `rb_gw_yk` | `world` | the same | the world's success on the same fires; billed |
| `rb_grd_yk` | `read` + diet 512 | `rb_s1__st_gn_yk/st_gn_yk/ck/c050.pt` (seed 0) | `rb_gr_yk` with the per-span diet (DESIGN §11) |
| `rb_pf_gn` / `rb_pf_gr` / `rb_pf_gw` | none / read / world | — | preflight twins, panel seeded |

`st_gn_yk` (banked, and re-run bit-identical as `rb_s1`) is the no-revocation floor and is not re-run.

## Runs

| tag | function | what | app id |
|---|---|---|---|
| `rb_r1_P`, `rb_t1_A` | `rb_r1_run` | machinery smoke 1: plan, then a save every cycle. **Died at c2 on the donor's in-loop A-4** at preflight's `extend_tol` 1.0 on a substrate built without `preflight`'s pre-setup check (DESIGN §5.1); the c1 save itself was fine (21.9 MB, 6.9 s) | — (attached) |
| `rb_r1_P`, `rb_t2_A`, `rb_t2_B` | `rb_r1_run` | machinery smoke 2 (preflight-exact substrate, `extend_tol` 0.0): plan (27 cycles, 418 s), a save every cycle to c5, c3 restored in a fresh container and continued to c5. R-2 PASS, world identical; the two c5 states differed ONLY in the loop's two wall-clock `t_s` fields (DESIGN §5.2) | — (attached) |
| `rb_gf1` | `fidelity_smoke` | gate G-F: every `[rubato]` knob off, the fork against `sostenuto.py` at smoke scale: max\|fork − sostenuto\| = 0 (control 0). PASS | `ap-WFQDYYKaZLFPHRrvVFatAM` |
| `rb_r1` | `rb_gate_r1` | gate R-1 of record (`figures/rb_r1_r1.txt`, `figures/rb_r1/r1_gate.json`): saved at c12 (era boundary) and c14 (mid-era), restored in fresh containers, continued to c24. B1 bit-exact; B2 one ulp in one `w_norm` (the host floor, DESIGN §4.2); both falsifications (`rng:torch_cpu`, `local:grng`) RED | `ap-1uzIk6y3MnFP99nwdIGTdP` |
| `rb_s1` | `sweep` | the seed-0 reference re-run, `st_gn_yk`, a save every 10 cycles and at every era boundary: 22 states, 1.24 GB. Bit-identical to the banked `st_s1` on every shared quantity (`figures/rb_s1_bank_check.txt`) | `ap-qCV1ighQF4HIH5jQ870yim` |
| `rb_s2` | `sweep` | the seed-2 reference re-run, the same: 22 states, 1.24 GB. Preempted at c60 and restarted by Modal from c1 (DESIGN §5.5). Bit-identical to the banked `st_s2` on every shared quantity (`figures/rb_s2_bank_check.txt`); carries the phase-2 grade knobs with the grade off | `ap-kbDyLMgdBHPzFSydcrn8JH` |
| `rb_pfg1` | `rb_pf_grade` | [phase 2] the first grade preflight: G-I red from label and config-copy bookkeeping, G-W's falsification tied (DESIGN §10.8, defects 1–2) | `ap-VF8Mbe8bOowUYx3bRPJmgV` |
| `rb_pfg2` | `rb_pf_grade` | [phase 2] the grade preflight of record, six arms in one container: every gate green, every falsification RED (DESIGN §10.8) | `ap-QPc2MsJmcef0gwn46GQ9fy` |
| `rb_g1` (1st) | `sweep` | [phase 2] seed-0 graded arms: **refused by R-2 on both arms** at the restore, the digest rule having been refined after `rb_s1` saved (DESIGN §5.4); ~0.38 GPU-h, nothing run. Log `results/launch_rb_g1_fail1.log` | `ap-hrbeGNVpNJffzeXx5LKaFf` |
| `rb_g1` (seg. 2) | `sweep` | [phase 2] seed-0 graded arms `rb_gr_yk`, `rb_gw_yk`, restored from `rb_s1`'s c20 (`results/RUN_rb_g1.sh`); R-2 PASS on both; G-I equal at c30/c40/c50 on `rb_gr_yk`. Ran to c140 / c137, when a coordinator preemption restarted the pair from c20; the restart reproduced the c30/c40 digests and was stopped at ~c45 (DESIGN §5.5). Log `results/launch_rb_g1_seg1.log` | `ap-7bCvwqMpJEGC3cpcXrXOYY` |
| `rb_g1` (seg. 3) | `sweep` | [phase 2] both arms resumed from their own c130 saves (R-2 PASS, digests = the c130 saves') to c201, complete (`results/RUN_rb_g1_resume.sh`, `rb_auto_resume` on); 1.15 + 1.47 GPU-h. Compact mirror `figures/rb_g1/`; reduction `figures/rb_g1_reduction.txt` ([L] reads segment 2 off `results/launch_rb_g1_seg1.log`); `figures/rb_grade_seedtable.txt` (DESIGN §4.5) | `ap-rFAurlv2S3xWuYbCbUDTex` |
| `rb_g2` | `sweep` | [phase 2] seed-2 graded arms `rb_gr_yk`, `rb_gw_yk`, restored from `rb_s2`'s c20 (`results/RUN_rb_g2.sh`; digest version resolved by trial to v2, R-2 PASS): complete, 2.17 + 2.19 GPU-h; G-I equal to `rb_s2` until the first revocation (read c65, world c45). Mirror `figures/rb_g2/`; `figures/rb_g2_reduction.txt`; `figures/rb_g2_diet.json`, `figures/rb_g2_revoked_survey.json` (DESIGN §4.7, §4.8) | `ap-iCnpm04UW3ZVTJLDqCGbOK` |
| `rb_g1d` | `sweep` | [r3] `rb_grd_yk`, seed 0, restored from `rb_s1`'s c50 (`results/RUN_rb_g1d.sh`): complete, 1.61 GPU-h; G-I equal c60–c100, first revocation c105. Mirror `figures/rb_g1d/`; `figures/rb_g1d_reduction.txt` (beside `rb_gr_yk`, `rb_gw_yk`); `figures/rb_g1d_diet.json`, `figures/rb_g1d_revoked_survey.json` (DESIGN §11.1) | `ap-iW08uefItTAbE1eDcOJdq8` |
