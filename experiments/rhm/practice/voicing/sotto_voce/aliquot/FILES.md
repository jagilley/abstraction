# FILES — `aliquot`

The machinery record: every code file, every knob, every gate, every arm, every run. Decisions and
withdrawn diagnoses are in [`DESIGN.md`](DESIGN.md); the brief is [`SPEC.md`](SPEC.md). The writeup is [`README.md`](README.md); the session record is `CONVERSATION.md`[^private].

**Up**: [`../FILES.md`](../FILES.md) (`sotto_voce`) · [`../../FILES.md`](../../FILES.md) (`voicing`)

## Code files

| file | purpose |
|---|---|
| `aliquot.py` | the substrate: `../sotto_voce.py` forked at its `so_s1`/`so_s2` head. Adds `vo_om_mode="proj"`, `VoProjBank`, `vo_pj_irls`, `build_rand_trunk`, the `vo_pj_*` knobs, the four P-gates and their run-level forms, the two paid arms and their two preflight twins, and a CPU `falsify_remote` entrypoint. Every addition `# [aliquot]`-marked; every knob default off |
| `mk_seedtable.py` | the cross-seed table of record -> `figures/al_seedtable.txt`: (a) the chooser, (b) the critic, (c) the grader against the world per level, (d) the blind region per (level, in-support), (e) the readout itself, (f) the bill and the yoke, (g) the grader's own ranking of the world's verdict from the per-probe reservoir |
| `analyze_aliquot.py` | the reducer: `../analyze_sotto.py` forked. [P]/[R] extended to the banked mirror arms, [T] extended with P-2r/P-3r/P-4r, new [PJ] |
| `gates/falsify.py` | the falsification harness: `../gates/falsify.py` forked, retargeted at `aliquot.py`, plus seven projection perturbations and seven run-level ones. **72/72** |
| `launch_detached.py` | the session-isolated detached launcher, retargeted at `aliquot.py` |
| `results/wait_app.sh` | copied from `practice/embouchure/results/`: waits on `modal app list` rather than on the local log, which this round needed when the session's container restarted mid-run and killed the `--detach` clients |
| `fetch_compact.py` | fetch a tag from the volume and write the compact local mirror, retargeted at `rhm_practice_aliquot` |

## Docs

| file | purpose |
|---|---|
| `SPEC.md` | the orchestrator's brief, verbatim |
| `DESIGN.md` | the decisions, the gate table, the cost, and every withdrawn diagnosis beside its correction |
| `FILES.md` | this file |
| `README.md` | the writeup, the super-node for `aliquot`, `duplex` and `soundboard` |
| `CONVERSATION.md` | the session record behind the three rounds: Jasper's prompts verbatim, the responses summarised |

## The new knobs

`vo_om_mode` gains one value; everything else is new and defaults to the donor's behaviour.

| knob | level | default | meaning |
|---|---|---|---|
| `vo_om_mode="proj"` | **arm** | `"world"` | the grader of record for a probe is a ridge-logistic readout of `SN.trunk`'s pooled hiddens over the substituted configuration. Dispatches exactly where `"model"` does: everything filed, nothing billed, the world's verdict an instrument |
| `vo_pj_trunk` | **arm** | `"live"` | `"live"` reads the generator the loop trains every cycle; `"rand"` reads a never-trained generator of the same architecture (`overtone`'s control two, in the loop) |
| `vo_pj_rand_seed` | run | `20260918` | the random twin's init seed, stated so the floor is reproducible |
| `vo_pj_mask` | run | `True` | the `mask_block` idiom: mask one block **outside** the span before reading, because the plant has never seen a fully unmasked input. The unmasked read is a periodic diagnostic, never filed |
| `vo_pj_root` | run | `"inter"` | `"inter"` = root one-hot **and** root × pooled state (a linear readout per goal); `"add"` = one-hot only, logged as the secondary column |
| `vo_pj_ridge` | run | `"1,32,1024"` | the ridge grid, selected per refit on a train-internal validation split |
| `vo_pj_iters` | run | `25` | IRLS iterations (early-stops on the step norm) |
| `vo_pj_fit_cap` | run | `8192` | rows drawn (uniformly, from the bank's own `frng`) from the trainable rows for one refit |
| `vo_pj_every` | run | `1` | refit cadence in cycles; 1 is the mirror's own schedule |

## New objects in `aliquot.py`

| object | what it is |
|---|---|
| `build_rand_trunk(v, length, s, dim, seed, device)` | a generator of the same architecture, never trained, minted inside the file's RNG sandbox under `manual_seed(seed)` — `overtone`'s `rand_slot` control, in the loop. Deliberately **not** hand-zeroed (zeroing LayerNorm weights makes the trunk output identically zero) |
| `vo_pj_irls(Z, y, lam, pen, iters)` | `ov_irls`'s estimator in torch: float32 Gram, float64 Newton solve, per-column penalty (`pen[-1] = 0` leaves the intercept free) |
| `VoProjBank(VoOutcomeBank)` | the projection in the mirror's seat. Same buffer, same two doors, same schedule, same filed probability. Owns no optimizer and no parameters but `w`; every fit and forward inside `_rng_snapshot`/`_rng_restore`. `_features` (the trunk read, masked), `_design` (the per-goal design matrix), `train` (recompute + ridge selection + solve + the two secondary fits), `predict`, `refresh` (the donor's instrument plus the secondary variants and the AUC drift), `state` |
| `vo_proj_check()` | CPU gate P-1/P-2/P-3/P-4 on a toy trunk and a synthetic parity target |
| `vo_gate_proj_run(a, arm)` | run-level gates P-2r/P-3r/P-4r, read off one arm file |
| `falsify_remote()` | a CPU Modal entrypoint that runs `gates/falsify.py` — this box has no local torch |
| `proj_gates_gpu()` | the same P-1 -- P-5 block on an **L4**. `vo_proj_check` takes a `device` parameter because defect #3 was a host tensor matmul'd against a device tensor, which a CPU-only gate cannot see (DESIGN §7.3) |

## Base changes in `aliquot.py` (all inert with the knobs off)

| change | why it is inert on a donor arm |
|---|---|
| `VoOutcomeBank.kind` (`"proj"` vs `"net"`) and `K = 1` for `"proj"` | every donor mode is `"net"`; `"model"` already forced K = 1 |
| `self.blk` / `self.spn` buffer columns, `_append(..., blk=, spn=)` | two extra `torch.cat`s and **no new RNG draw** in `_append`, in the donor's order; a mirror bank stores them and never reads them |
| `push_experience(rows, roots_np, slots=None)` | the slot map is read only to fill the two columns |
| `push_world(x, r, y, blk=None, spn=None)`, `world_rows` as a 5-tuple | hybrid-only path; the extra elements are the slot |
| `predict(..., blk0=None, span=None)` | the base class ignores them |
| `vo_run_probes`: `if mode in ("model", "proj")` | `"proj"` is unreachable unless an arm names it |
| `run_arm`: the bank factory branches on `vo_om_mode == "proj"` | the `else` branch is the donor's construction verbatim |

## Arms

| arm | yoke source | `vo_om_mode` | `vo_pj_trunk` | role |
|---|---|---|---|---|
| `al_pj_yk` | `voi3_dp` | `proj` | `live` | the projection over the live trunk |
| `al_rt_yk` | `voi3_dp` | `proj` | `rand` | the random-trunk twin |
| `al_pf_pj` | `voi3b_pf_src` | `proj` | `live` | preflight twin, governance ON |
| `al_pf_rt` | `voi3b_pf_src` | `proj` | `rand` | preflight twin, governance ON |

Banked and **not** re-run, all read at reduction time with `--bank`:

| banked arm | what it is |
|---|---|
| `vo_s3:voi3_dp` / `vo_s3d2:voi3_dp` | the seed-0 / seed-2 anchor, and the yoke source |
| `vo_s3b:voi3b_comp_yk` / `vo_s3e:voi3b_comp_yk` | the **floor**: composed chooser, filed diet |
| `vo_s3b:voi3b_comp_pr_yk` / `vo_s3e:voi3b_comp_pr_yk` | the **ceiling**: composed chooser, world-graded probes |
| `so_s1:so_mg_yk` / `so_s2:so_mg_yk` | the **mirror**: composed chooser, model-graded probes |
| `so_s1:so_cg_yk`, `so_s1:so_hy_yk` (and seed-2 twins) | the committee and the hybrid |

## Gates

| gate | where | claim |
|---|---|---|
| G-F | `fidelity_smoke` | with every knob off this file replays `enharmonic.py` bit for bit, against a donor self-replay control |
| V-1, V-4A/B, V-4c, V-4d, V-5, V-6, V-6b, Y-1 | `vo_gates_cpu`, `vo_preflight_gates` | the donor's, unchanged |
| M-1, M-1b, M-2, M-2b, M-3, M-4, M-5 | `vo_gates_cpu` | the mirror's, unchanged; `"proj"` joins M-1/M-3/M-4's mode table |
| M-1r / M-2r / M-3r / M-4r | `vo_gate_mirror_run` | the run-level forms; `"proj"` accepted with `want_bill = 0` and an added "files and agrees on the whole draw" assertion |
| **P-1** | `vo_proj_check` | the readout's logit is affine in the pooled state; the plant is untouched to the bit; no gradient on the trunk; no shared RNG stream moves |
| **P-2** | `vo_proj_check` | the random twin is a different object and is frozen **against its mint** |
| **P-3** | `vo_proj_check` | the features follow the trunk and are never cached |
| **P-4** | `vo_proj_check` | the readout is live: it refits, the weights move, the verdict is not constant |
| **P-5** | `vo_proj_check` | a constant-label buffer still gets an answer — the intercept-only fit at the base rate — instead of the grader silently going dark (DESIGN §7.2) |
| **P-2r / P-3r / P-4r** | `vo_gate_proj_run` | the run-level forms, off the arm file: fingerprint moving vs frozen, the refit count, the ridge on the grid, the instrument present, `n_noslot == 0` |
| V-6b(proj), V-6b(proj_vs_rand) | `vo_preflight_gates` | the projection-graded diet changes the run against the world-graded twin, and the two projection arms differ from each other |

`gates_cpu` **26/26** (CPU) · `proj_gates_gpu` P-1 -- P-5 on an L4 · falsification harness **73/73**.

## Runs

| tag | function | what | app id |
|---|---|---|---|
| `al_gf1` | `fidelity_smoke` | G-F, smoke scale. **PASS**: `max\|fork − enharmonic\| = 0.000e+00` on both arms, donor self-replay control `0.000e+00`, commits equal; Q-11 interface check PASS on 18 cells | `ap-mismGzzDsSRTW2ur4iFhkx` |
| `al_pf1` | `preflight` | 8 twins. **TRIPPED** gate M-1r on `al_pf_pj` — defect #2, DESIGN §7.2 (no probe was ever drawn: a constant-label buffer produced no fit). Everything before the gate block ran; kept as the record of the defect | `ap-J6Xh0UTaxBHIcjqGHSPZl8` |
| `al_pf2` | `preflight` | 4 twins after §7.2's fix. **TRIPPED** at `al_pf_pj` c8 in `predict` — defect #3, DESIGN §7.3 (the degenerate weight vector was on the host). Its crash site confirms §7.2's fix: the readout was ready and being asked for a probe verdict | *(app id not kept)* |
| `al_pf3` | `preflight` | the same 4 twins after §7.3's fix | `ap-jQJYJuLbGNgAyGavInBYZn` |
| `al_s1` | `voicing_run` | seed 0, `al_pj_yk,al_rt_yk`, yoked to `vo_s3:voi3_dp`. **COMPLETE**, 201 cycles per arm, 12761 s, every gate green | `ap-b7AkKRb15ChFr5nUiFtpIM` |
| `al_s2` | `voicing_run` | seed 2, `al_pj_yk,al_rt_yk`, yoked to `vo_s3d2:voi3_dp`. **COMPLETE**, 201 cycles per arm, 10261 s, every gate green | `ap-kvwFL5NZMT70rOTG4ZxA22` |

The local launch logs of the two **tripped** preflights (`al_pf1`, `al_pf2`) were not kept, so
their evidence is the assertion text and the traceback quoted verbatim in DESIGN §7.2 and §7.3,
plus `al_pf1`'s app id above and its artifacts still on the volume at
`rhm_practice_aliquot/_preflight_al_pf1/`. `al_pf2`'s app id was in the log that went with it.
The local logs of the runs that MATTER (`al_gf1`, `al_pf3`, `al_s1`, `al_s2`) are kept, with the
caveat that `al_s1`/`al_s2`'s end in the session container's own `ProxyConnectionError` rather than
a completion line — the restart killed the `--detach` clients while both remote apps ran on
unaffected. `modal app list` is the authority there and `results/wait_app.sh` is what watches it.

Volume `rhm-scaling-data:/rhm_practice_aliquot/<tag>/`; compact local mirrors and reductions under
`figures/` (`al_seedtable.txt` is the two-seed table of record; `al_s1_reduction.txt` and
`al_s2_reduction.txt` are the per-tag reductions).

**The gates on the paid tags, post hoc** (section [T] of each reduction, calling the substrate's
own gate functions): M-1r / M-2r / M-3r / M-4r **PASS** on both projection arms and on all three
banked mirror arms at both seeds; P-2r / P-3r / P-4r **PASS** on both projection arms at both
seeds, with `n_degenerate = 0` (so P-4r's strong form bound: the readout fit the state and not
just the base rate), `n_noslot = 0`, `n_spanclamp = 0`, `refit_skipped = 0`, and the trunk's
fingerprint moving by 204.3 / 180.5 on the live arms and constant to the bit at
16198.119362383673 on both twins. Y-1 exact with **0 cancelled** on every arm.

## Children

| node | what |
|---|---|
| [`duplex/`](duplex/FILES.md) | the offline follow-up to §1's finding: does the outcome error have to reach the TRUNK's weights? `overtone::post_write_probe`'s protocol reproduced cell for cell on the banked dumps as a gate, then re-read after shaping the trunk on an outcome head (both losses / outcome only / infill only at matched steps), with world-model diagnostics and a L2/L3 -> L4/L5 transfer split. Decisions in [`duplex/DESIGN.md`](duplex/DESIGN.md) |
| [`soundboard/`](soundboard/FILES.md) | the in-loop follow-up: the outcome head's gradient reaching the plant's trunk during the run, aliquot's projection unchanged in the grader's seat, three arms per seed (the verdict with both losses, next-level yield, the verdict with the infill term off), per-cycle world-model diagnostics, the yield instrument, the trajectory id and the dumps. Decisions in [`soundboard/DESIGN.md`](soundboard/DESIGN.md) |
| [`preplay/`](preplay/FILES.md) | the offline follow-up on banked state, nothing paid in the loop: within-context candidate discrimination on the six soundboard dumps (`within/`), and candidate table entries fired through the shaped executor and priced by the projection against the world's audition, as a selector, and on the learner's own operative tables replayed exactly from the banked keys and picks (`pp1`, `pp2`, `pp3`). Decisions in [`preplay/NOTES.md`](preplay/NOTES.md), [`preplay/NOTES_own.md`](preplay/NOTES_own.md), [`preplay/within/NOTES.md`](preplay/within/NOTES.md) |

## Reproduction

```bash
cd experiments/            # MODAL_PROFILE=chromatic
modal run rhm/practice/voicing/sotto_voce/aliquot/aliquot.py::voicing_gates    # 26 CPU gates
modal run rhm/practice/voicing/sotto_voce/aliquot/aliquot.py::falsify_remote    # 73/73
modal run rhm/practice/voicing/sotto_voce/aliquot/aliquot.py::proj_gates_gpu     # P-1..P-5 on an L4
python3 rhm/practice/voicing/sotto_voce/aliquot/launch_detached.py --fn fidelity_smoke --tag al_gf1
python3 rhm/practice/voicing/sotto_voce/aliquot/launch_detached.py --fn preflight --outdir-tag al_pf3 \
    --arms "voi3b_pf_src,voi3b_pf_comp_pr,al_pf_pj,al_pf_rt"
bash rhm/practice/voicing/sotto_voce/aliquot/results/RUN_al_s1.sh               # seed 0
bash rhm/practice/voicing/sotto_voce/aliquot/results/RUN_al_s2.sh               # seed 2
modal run rhm/practice/voicing/sotto_voce/aliquot/aliquot.py::preflight_gates --outdir-tag al_s1
python3 rhm/practice/voicing/sotto_voce/aliquot/fetch_compact.py --tag al_s1 --fetch --replace
python3 rhm/practice/voicing/sotto_voce/aliquot/analyze_aliquot.py --tag al_s1 --yoke-src vo_s3:voi3_dp \
    --bank vo_s3:voi3_dp,vo_s3b:voi3b_comp_yk,vo_s3b:voi3b_comp_pr_yk,so_s1:so_mg_yk,so_s1:so_cg_yk,so_s1:so_hy_yk
```

One environment fact of the box this node was built on, stated rather than worked around: there is
**no local torch**, so `vo_gates_cpu` and `gates/falsify.py` run as CPU Modal functions
(`voicing_gates`, `falsify_remote`) instead of as local scripts, and the system interpreter behind
the `modal` CLI needs numpy (`/usr/bin/python3 -m pip install --user numpy==1.26.4`).

[^private]: Not mirrored: this link points to a document in the private lab repo (the roadmap, the queue, an unrun spec, reading notes, or a conversation). See the top-level README for what is held back and why.
