# FILES — `soundboard`

The machinery record: every code file, every knob, every gate, every arm, every run. Decisions and
withdrawn diagnoses are in [`DESIGN.md`](DESIGN.md); the brief is [`SPEC.md`](SPEC.md). The writeup is the parent's, [`../README.md`](../README.md); [`README.md`](README.md) here is a pointer.

**Up**: [`../FILES.md`](../FILES.md) (`aliquot`) · [`../../FILES.md`](../../FILES.md)
(`sotto_voce`) · **Sibling**: [`../duplex/FILES.md`](../duplex/FILES.md)

## Code files

| file | purpose |
|---|---|
| `soundboard.py` | the substrate: `../aliquot.py` forked at its `al_s1`/`al_s2` head. Adds `VoShaper` and its head, the yield currency, the in-loop world-model diagnostics, the three dumps, the trajectory id, the readout's calibration and duplicate-share columns, the five S-gates and their four run-level forms, the three paid arms and their three preflight twins, and the `sweep` fan-out coordinator. Every addition `# [soundboard]`-marked; every knob default off |
| `analyze_soundboard.py` | the reducer: `../analyze_aliquot.py` forked. [T] extended with S-1r/S-2r/S-3r, [PJ] with the readout's LEVEL and the duplicate share, new [SB] |
| `mk_seedtable.py` | the cross-seed table of record -> `figures/sb_seedtable.txt`: `aliquot`'s seven sections plus (h) the shaping itself, (i) the world-model diagnostics, (j) the yield currency in both currencies |
| `gates/falsify.py` | the falsification harness: `../gates/falsify.py` forked, retargeted, plus ten shaping perturbations and sixteen run-level ones. **100/100** |
| `launch_detached.py` | the session-isolated detached launcher, retargeted at `soundboard.py` |
| `fetch_compact.py` | fetch a tag from the volume and write the compact local mirror, retargeted at `rhm_practice_soundboard` |
| `results/wait_app.sh` | copied from `../results/`: waits on `modal app list` rather than on the local log |

## Docs

| file | purpose |
|---|---|
| `SPEC.md` | the orchestrator's brief, verbatim |
| `DESIGN.md` | the decisions, the yield currency's definition, the gate table, the cost, and every withdrawn diagnosis beside its correction |
| `FILES.md` | this file |

## The new knobs

Everything below is new and defaults to the donor's behaviour.

| knob | level | default | meaning |
|---|---|---|---|
| `vo_sh` | **arm** | `False` | build the outcome head and add its BCE to the plant's own optimizer step at weight 1. Its gradient reaches the trunk and never the emission head |
| `vo_sh_target` | **arm** | `"solve"` | `"solve"` = the world's verdict on the learner's own experienced configurations (the mirror's src-0 diet); `"yield"` = next-level yield from the learner's own miner (DESIGN §3.2) |
| `vo_sh_infill` | **arm** | `True` | keep the plant's own masked-infill term. `False` is `duplex`'s `out@N` in the loop and changes nothing else |
| `vo_yq_probe` | **arm** | `False` | compute the per-probe yield instrument (three columns). An instrument, like the world's verdict beside it (gate M-1) |
| `vo_wm` | **arm** | `False` | the world-model diagnostics per cycle |
| `vo_dump` | **arm** | `False` | dump the rows, the bank's buffer and the heads at end of arm |
| `vo_sh_mask` | run | `True` | the `mask_block` idiom, `vo_pj_mask`'s default: the shaping reads the state the GRADER reads |
| `vo_sh_batch` | run | `256` | rows per shaping step (`batch_size`, the infill term's own) |
| `vo_sh_min` | run | `64` | bank rows before the term fires. Far below `vo_om_min = 512` (DESIGN §3.3) |
| `vo_sh_lr` | run | `None` -> `gen_lr` | the head's optimizer-group rate |
| `vo_sh_detach` | run | `False` | **the harness's perturbation only**: detach the pooled state, so the error reaches the head and not the trunk. Gate S-1 must go red on it; no arm sets it |
| `vo_wm_smoke` | arm | `False` | the diagnostics at preflight sizes |
| `vo_wm_n` / `vo_wm_batches` / `vo_wm_batch` / `vo_wm_parse_n` / `vo_wm_dp_n` | run | `8192 / 8 / 256 / 2048 / 512` | the fixed plan's sizes (`duplex::wm_diag`'s, halved for a per-cycle read) |
| `vo_wm_dp_every` | run | `50` | the nested DP parse's cadence in cycles; it also runs once at end of arm |
| `vo_wm_seed` | run | `777002` | `duplex`'s own eval-corpus seed, so the windows are fresh against `train_seed = 1` |
| `vo_dump_cap` | run | `0` | rows per (diet, slot) in `vo_rows.npz`; 0 = the whole buffer |

## New objects in `soundboard.py`

| object | what it is |
|---|---|
| `sb_shape_input(x, blk, spn, nb, s, mask, device)` | `VoProjBank._features`' input arithmetic WITHOUT the `no_grad`: the masked configuration and the span weights. Gate S-2 asserts the two paths agree to the bit |
| `build_shape_head(...)` / `_build_shape_head()` | `duplex::build_out_head`'s MLP over the two pools, with a ROOT embedding beside the slot's (DESIGN §3.1). Minted with `build_outcome`'s RNG discipline |
| `sb_miner_keys(miner, feats)` | the keys `miner.observe` WOULD count for these rows — the miner's own keying, because a quotiented miner's counts are keyed by class pairs and the raw tuple would read identically zero |
| `sb_yield_label(pf, miner, support, solved)` | the next-level yield currency (DESIGN §3.2) |
| `sb_yield_true(pf, truth, level, s, solved)` | the oracle's column of the same currency: the same spans counted against the TRUE table. An instrument only |
| `VoShaper` | the outcome head in the plant's own optimizer step. Owns the head and a numpy stream; `term()` returns the tensor `finetune_generator_span` adds, `snapshot()` the plant's per-cycle fingerprint |
| `sb_wm_plan(shared, cfg, device, smoke)` | the fixed window/mask/configuration plan, drawn once per arm |
| `sb_wm_diag(core, plan, shared, cfg, device, dp)` | `duplex::wm_diag` in the loop: held-out infill CE and accuracy by mask count, the block head's level-1 parse accuracy masked and unmasked, and (at checkpoints) the nested DP parse at L2..L5 |
| `sb_row_index(move, cand, cache)` | `voicing::ov_row_index`, copied, so the dump carries the DP's own score of each candidate |
| `sb_dump_rows` / `sb_dump_bank` / `sb_dump_heads` | the three dumps (DESIGN §4) |
| `sb_shape_check(tdim, n, device)` | CPU/GPU gates S-1, S-1b, S-2, S-3, S-4, S-5 on a toy trunk and a synthetic target |
| `sb_gate_shape_run(a, arm)` | run-level gates S-1r / S-2r / S-3r, off one arm file |
| `sb_gate_shape_twin(a_off, a_on)` | run-level gate S-4r: the shaped arm's plant against the unshaped twin's |
| `sweep(out_tag, seed, arms, ...)` | the CPU coordinator: one `voicing_run` container per arm, then the merge into one tag directory. `SB_FLAGS` holds the paid run's flag line |

## Base changes in `soundboard.py` (all inert with the knobs off)

| change | why it is inert on a donor arm |
|---|---|
| `VoOutcomeBank.yq` / `.tid` columns, `_append(..., yq=, tid=)` | two extra `torch.cat`s and **no new RNG draw**, in the donor's order; a donor bank stores `-1` and never reads them |
| `push_experience(..., label_fn=None)` | `None` on every arm but the currency arm, and then the column stays at `-1` |
| `VoRecorder.assemble` adds `"tid"` | one `torch.arange`, no RNG; unread unless a bank asks for it |
| `vo_run_probes(..., yq_fn=None)` | `None` unless the arm names `vo_yq_probe` |
| `VoProjBank.refresh` adds the level and the duplicate share | instrument-only, no gradient, no RNG; `dup_share` every tenth refit |
| `finetune_generator_span`: the shaping term and the `no_infill` branch | both keyed on `vo_obj`, which is `None` on every arm without a critic and carries neither key unless the arm names `vo_sh` |
| `log["sb"]` per cycle | `None` unless the arm is shaped or is a projection arm |
| `run_arm`'s `[soundboard]` block | every branch keyed on `cfg.get("vo_sh")` / `vo_wm` / `vo_dump` / `vo_yq_probe` |

## Arms

| arm | yoke source | `vo_sh_target` | `vo_sh_infill` | role |
|---|---|---|---|---|
| `sb_sv_yk` | `voi3_dp` | `solve` | on | the world's verdict in the trunk's weights |
| `sb_yd_yk` | `voi3_dp` | `yield` | on | the abstraction grader's own currency in the trunk's weights |
| `sb_so_yk` | `voi3_dp` | `solve` | **off** | `duplex`'s `out@N` in the loop |
| `sb_pf_sv` / `sb_pf_yd` / `sb_pf_so` | `voi3b_pf_src` | as above | as above | preflight twins, governance ON |

Banked and **not** re-run, all read at reduction time with `--bank`:

| banked arm | what it is |
|---|---|
| `vo_s3:voi3_dp` / `vo_s3d2:voi3_dp` | the seed-0 / seed-2 anchor, and the yoke SOURCE |
| `vo_s3b:voi3b_comp_yk` / `vo_s3e:voi3b_comp_yk` | the **floor**: composed chooser, filed diet |
| `vo_s3b:voi3b_comp_pr_yk` / `vo_s3e:voi3b_comp_pr_yk` | the **ceiling**: composed chooser, world-graded probes |
| `so_s1:so_mg_yk` / `so_s2:so_mg_yk` | the **mirror**; `so_cg_yk` / `so_hy_yk` the committee and the hybrid |
| `al_s1:al_pj_yk` / `al_s2:al_pj_yk` | `aliquot`'s **frozen-plant projection** — the arm the three shaped ones are one knob from |
| `al_s1:al_rt_yk` / `al_s2:al_rt_yk` | `aliquot`'s **random-trunk twin** |

`al_pf_pj` is re-run in this node as a PREFLIGHT arm only, because gate S-4r needs an unshaped twin
that carries the per-cycle fingerprint series and `aliquot`'s banked arms predate it.

## Gates

| gate | where | claim |
|---|---|---|
| G-F | `fidelity_smoke` | with every knob off this file replays `enharmonic.py` bit for bit, against a donor self-replay control |
| V-*, M-*, P-*, Y-1 | `vo_gates_cpu`, `vo_preflight_gates` | the donor's, unchanged, and run on the three shaped arms too |
| **S-1 / S-1b** | `sb_shape_check` | the outcome error reaches the TRUNK and never the emission head |
| **S-2** | `sb_shape_check` | the shaping reads the state the projection reads, to the bit |
| **S-3** | `sb_shape_check` | sandboxed: no shared stream, no bank row moves |
| **S-4** | `sb_shape_check` | the target is the one the arm named |
| **S-5** | `sb_shape_check` | the yield currency is the miner's own keying, and zero on an unsolved piece |
| **S-6 / S-6b** | `sb_shape_check` | … and through the REAL `LearnedQuotient` / `ClassMiner`, including under a merge (which is what makes the halves' level index load-bearing) |
| **S-1r / S-2r / S-3r** | `sb_gate_shape_run` | the term fired, the plant moved, the target matches the config, the currency is not constant, the diagnostics exist and the DP ran |
| **S-4r** | `sb_gate_shape_twin` | the shaped arm's plant differs from the UNSHAPED twin's — the gate the round rests on |

`gates_cpu` **27/27** (CPU) · `proj_gates_gpu` P-1 … P-5 **and** S-1 … S-6b on an L4 ·
falsification harness **100/100**.

## Runs

| tag | function | what | app id |
|---|---|---|---|
| — | `voicing_gates` | 27 CPU gates, **PASS** | `ap-uYTlXMQR9y4Wxk9lLOqlmt` |
| — | `falsify_remote` | **100/100** perturbations behaved as required | `ap-bdDG4K8IeifizeodAAwSUH` (95/95 at `ap-GdMMwih1vLnbU8rXEtC3RV` before S-6/S-6b) |
| — | `proj_gates_gpu` | P-1 … P-5 and S-1 … S-6b on an L4, **PASS** | `ap-qedZyLkai8NyCovMoes6Lq` |
| — | `preflight_gates --outdir-tag sb_pf1` | the whole file-based block re-asserted on the saved preflight artifacts after the two gate corrections. **PASS**: Y-1 exact with 0 cancelled on all four yoked twins, M-1r…M-4r and P-2r/P-3r/P-4r on all four projection arms, S-1r…S-3r on the three shaped twins and the unshaped one, **S-4r PASS on all three** (max\|Δ sig\| 8.92 / 8.92 / 8.47 against `al_pf_pj`, first differing cycle 6 / 6 / 0, 42 shaping steps each) and V-6b beside it (the RUN differs too: `dres` by 0.219, `n_solved` by 5 on the outcome-only arm) | `ap-N6ICEnCBrtw7H3s4KTYyZI` |
| `sb_gf1` | `fidelity_smoke` | G-F. **PASS**: `max\|fork − enharmonic\| = 0.000e+00` on both arms, donor self-replay control `0.000e+00`, commits equal; Q-11 interface check PASS on 18 cells | `ap-7UeoQUEkftrQkwYEB0Sucw` |
| `sb_pf1` | `preflight` | 5 twins: `voi3b_pf_src`, `al_pf_pj`, `sb_pf_sv`, `sb_pf_yd`, `sb_pf_so` | `ap-WrrBCpYhlrAMYcgcZ97k1p` |
| `sb_s1` | `sweep` | seed 0, `sb_sv_yk`/`sb_yd_yk`/`sb_so_yk`, one container each, yoked to `vo_s3:voi3_dp`. **COMPLETE**, 201 cycles per arm, 8096 s (2.25 h) per arm, 8117 s wall, 0 errors; every gate green | `ap-5Df31y7Nd2Ki6kfKl3BziH` |
| `sb_s2` | `sweep` | seed 2, the same three, yoked to `vo_s3d2:voi3_dp`. **COMPLETE**, 201 cycles per arm, 8142 s (2.26 h) per arm, 8165 s wall, 0 errors; every gate green | `ap-nhsTYd4AGyQ87OrYfUi1ap` |

**The gates on the paid tags, post hoc** (sections [T] / [PJ] / [SB] of each reduction, calling the
substrate's own gate functions — `vo_preflight_gates` is keyed on the PREFLIGHT arm names and
returns `{}` on a paid tag, which is `aliquot`'s arrangement too): M-1r / M-2r / M-3r / M-4r and
P-2r / P-3r / P-4r **PASS** on all six shaped arms and on the five banked mirror/projection arms at
both seeds, with `n_degenerate = 0`, `n_noslot = 0`, `n_spanclamp = 0`, `refit_skipped = 0`;
S-1r / S-2r / S-3r **PASS** on all six, with the shaping firing 3,060 / 2,840 times over 782,700 /
725,920 rows and the yield arm's currency non-constant (`n_label_solved` 9,346 / 9,466,
`label_mean` 0.0797 / 0.0730) so S-2r's assertion binds; Y-1 exact with **0 cancelled** on every
arm. The twin form S-4r against an *unshaped* plant is read on the preflight (the banked `aliquot`
arms predate the fingerprint series); its PAIRWISE form is read here and is the contrast the
preflight could not assert — `sb_sv_yk` and `sb_yd_yk` are bit-identical until **c48 / c59**, each
seed's L2 commit and so the first cycle a macro write exists for the bank to hold, and diverge
exactly there (max\|Δ fingerprint\| 15.4 / 28.8).

Volume `rhm-scaling-data:/rhm_practice_soundboard/<tag>/` (the per-arm containers write
`<tag>__<arm>/` and the coordinator merges them into `<tag>/`); compact local mirrors and reductions
under `figures/` — `sb_seedtable.txt` is the two-seed table of record, `sb_s1_reduction.txt` /
`sb_s2_reduction.txt` the per-tag reductions, `sb_pf1_reduction.txt` the preflight's. The local
mirror carries each arm's three dumps beside its `results.json` (`vo_rows.npz` + `_meta.json`,
`vo_bank.npz`, `vo_heads.pt` — ~6 MB an arm, and the point of them is that the next audit is a CPU
job). `setup.json` is **gzipped** in the mirror: it is 58 MB uncompressed and almost all of it is the
TRUE TABLES written out flat, the top rung's alone being 262,144 rows — a verbatim copy of the DGP
that `rule_seed = 0` reproduces exactly, so it is kept compressed rather than checked in raw.

## Reproduction

See [`DESIGN.md`](DESIGN.md) §8.
