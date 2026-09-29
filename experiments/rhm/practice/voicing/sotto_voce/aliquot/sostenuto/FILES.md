# FILES — `sostenuto`

The machinery record: every code file, every knob, every gate, every arm, every run. The
writeup is [`README.md`](README.md); the decisions and the design choices are in [`DESIGN.md`](DESIGN.md);
the briefs are [`SPEC.md`](SPEC.md); the session record is `CONVERSATION.md`[^private].

**Up**: [`../FILES.md`](../FILES.md) (`aliquot`) · **Donor**: [`../soundboard/FILES.md`](../soundboard/FILES.md) ·
**The offline rounds this reproduces in the loop**: [`../preplay/FILES.md`](../preplay/FILES.md) (pp4, pp5)

## Code files

| file | purpose |
|---|---|
| `sostenuto.py` | the substrate: `../soundboard/soundboard.py` forked at its `sb_s1`/`sb_s2` head. Adds the admission set (`sos_key_of`, `sos_flat_key`, `sos_groups`, `sos_filter`, `sos_candidates`, `sos_decide`, `sos_walk`, `sos_repro_gate`), the in-loop block (g2.5), `operative`'s filter and its three counted states, the admission set's re-key on a taken merge, the world-model diagnostics' cadence, the era-boundary and post-refresh head dumps, the A-gates and their falsifications, and the six new arms. Every addition `# [sostenuto]`-marked; every knob default off |
| `mk_seedtable.py` | the cross-seed table of record -> `figures/st_seedtable.txt`: a replication header off the arm files (the L3-L5 confusion, the deep-era task error, the coverage trace and the bill, per seed) then every section [A] to [J] of both per-tag reductions, seed 0's block beside seed 2's, never averaged |
| `reduce_sostenuto.py` | the reduction, local and CPU-only: the run-level gates, the admitted table's test-pool error against the ungated build, the junk share, the per-candidate confusion against the world gate, the task error against the banked no-walk reference, the readout's series, the fallbacks, the bill and the walk's wall clock. Writes `figures/<tag>_reduction.txt` and four panels |
| `launch_detached.py` | the session-isolated detached launcher, retargeted at `sostenuto.py` |
| `fetch_compact.py` | fetch a tag from the volume and write the compact local mirror, retargeted at `rhm_practice_sostenuto` |
| `results/wait_app.sh` | copied from `../soundboard/results/`: waits on `modal app list` rather than on the local log |

## Docs

| file | purpose |
|---|---|
| `README.md` | the writeup, both rounds |
| `SPEC.md` | the orchestrator's briefs for both rounds, verbatim |
| `CONVERSATION.md` | the session record, 2026-09-21 → 23 |
| `DESIGN.md` | the two design choices with their reasons, the op, the gate table, the arms, the cost, and the one defect in the donor this routes around |
| `FILES.md` | this file |

## The new knobs

Everything below is new and defaults to the donor's behaviour.

| knob | level | default | meaning |
|---|---|---|---|
| `admit` | **arm** | absent | `"world"` / `"read"` / `"none"`: the admission set is in force and this is its gate. Restricted to `open_inventory` at the point of resolution, for `_open_inv`'s own reason — with the inventory closed there is no live build to gate |
| `adm_reoffer` | run | `False` | `True` re-offers a rejected key on a later pass (`census_extend`'s shape). `False` is DESIGN §1.1: decide once, with a re-key by a merge as the one second chance |
| `adm_seed_commit` | run | `False` | `True` seeds the admission set with the commit-time build and gates only the growth. `False` is DESIGN §1.2: the gate touches every row |
| `adm_n_test` | run | `256` | the unpriced test pool at the end of each pass, pp5's test family. Disjoint from the gate and practice pools (gate A-6) |
| `vo_wm_every` | run | `1` | the world-model diagnostics' cadence in cycles. `1` is the donor; the arms run at `5` |
| `vo_dump_era` | run | `False` | also dump `vo_heads_e{k}.pt` at every era boundary |
| `vo_dump_refit` | run | `False` | the end-of-arm `vo_heads.pt` also carries `proj_final`: the readout solved once more against the core as it stands, so preplay's one-update skew is resolved rather than passed on |
| `admit: "yield"` / `admit: "demand"` | **arm** | — | [r2] the two gates in the next level's currency (DESIGN §9.2, §9.3) |
| `adm_delta` | run | `0.0` | [r2] the admission margin on `read`, `yield` and `demand`: admit iff the difference is at least `-adm_delta`. `0.0` is round 1's strict rule exactly. Logged, not run as an arm |
| `adm_demand_min` | run | `1.0` | [r2] the demand gate's threshold on its sum |
| `adm_demand_floor` | run | `-1` | [r2] the per-key count a next-level key must reach to enter the demand sum; `-1` resolves to `mine_support` (the build's currency, what runs); `0` is the brief's literal raw form (DESIGN §9.3 Fact 1: the `none` arm under the panel) |
| `preflight_seed_panel` | **arm**, preflight only | absent | [r2] seed the observation panel as `preflight_seed_miner` seeds the committable miners; on `st_pf_gy`/`st_pf_gd` only (DESIGN §9.6) |

The cadence (`recert_every` 5), the cap (`extend_cap` 8), the gate pool (`n_aud` 192) and the
tolerance (`extend_tol` 0.0) are **the loop's own knobs** and are not duplicated.

## New objects in `sostenuto.py`

| object | what it is |
|---|---|
| `sos_key_of(flat_row, level, s, quot)` | the key `ClassMiner.observe` WOULD assign a row, from the row's own flat level-1 expansion. `quot is None` gives `MC.Miner`'s key. Gate A-8 |
| `sos_flat_key(key)` | the admission set's serialisable identity: the `s` representatives concatenated |
| `sos_groups(tbl, level, s, quot)` | {flat key -> the row indices the build emitted under it} |
| `sos_filter(tbl, level, s, quot, admitted, lower)` | the build restricted to admitted keys. Gate A-7 |
| `sos_candidates(miner, tbl, level, s, quot, support, decided, cap)` | the not-yet-decided at-support keys in `(-count, key)` — `extend_candidates`' order — with their rows. Gate A-11. Exists because `extend_candidates` is inert under a quotient (gate A-10, DESIGN §4) |
| `sos_decide(gate, cur, tri, tol)` | `preplay/readgate.py::decide` restricted to the three values this round runs |
| `sos_walk(stats_fn, base_rows, cands, gate, tol)` | `readgate.gated_walk`'s loop over GROUPS of rows. Gates A-1a / A-1b |
| `sos_repro_gate(...)` | A-1c / A-4 / A-5 on ONE REAL PASS: `readgate.gated_walk` re-run in-container on the same base, groups, order and pool, and asserted identical; plus what the other two gates would have admitted |
| `sos_pool_keys(roots, x)` | `readgate`'s `keyset`, for the pool-disjointness gate A-6 |
| `sos_designed_stats(base_e, seq, p)` | a `stats_fn` over a DESIGNED sequence, in call order — the object pp5 §9.3 says a walk gate must be falsified on |
| `sos_gate_block` / `sos_falsify_block` | A-0, A-1a, A-1b, A-4, A-5, A-7, A-8, A-10, A-11 and their eight perturbations |
| `sos_gate_run(a, arm)` | A-1r … A-4r, A-9r off one arm file. Called from `vo_preflight_gates` and from the reducer |
| `sostenuto_gates` / `sostenuto_gates_gpu` / `sostenuto_gates_run` | the three entrypoints |
| `sos_classes(flat_rows, level, quot)` | [r2] a candidate's level-l classes, `quot.id_of(flat, l)` of its rows — the ids the next-level `ClassMiner` keys a half by |
| `sos_demand(miner, classes, floor)` | [r2] the next level's counts over keys having one of `classes` as a half, each key once; both the at-floor `value` and the literal `raw`, and `silent` when the miner holds no key at the floor. Gate A-13, E-D |
| `sos_yield_obj(pf, miner, support, level)` | [r2] `sb_yield_label(pf, next-level miner, support, solved=None)` and `share × level`. Gate A-15, E-Y |
| `sos_readgate_spec(gate, cands, delta, thr)` | [r2] every gate of this node AS a `readgate.gated_walk` call: yield = the read rule on `y`; demand = the read rule on the level `-#undemanded keys served` |
| `sos_designed_stats_y` / `sos_gate_block_r2` / `sos_falsify_block_r2` | [r2] A-12 … A-15, A-4y, A-5y, A-5d on designed objects and their eleven perturbations |
| `sos_export_gate(ex)` | [r2] E-Y and E-D off an exported pass (`adm_export_{unfit,fit}.json.gz`), in another process, against the donor's own functions, with on-export perturbations reported RED or BLIND |

## Base changes in `sostenuto.py` (all inert with the knobs off)

| change | why it is inert on a donor arm |
|---|---|
| `operative`'s admission branch | keyed on `_adm_on`, which is `cfg.get("admit")` and `_open_inv`; with `admit` absent the donor's two lines run unchanged (gate G-F) |
| block (g2.5), the walk | keyed on `_adm_on`: a single falsy lookup on every other arm |
| the admission set's re-key on a taken merge | keyed on `_adm_on` |
| the commit's empty-table cancellation counter | one extra branch inside a branch the donor already takes |
| `vo_wm_every` | `1` reproduces the donor's `if _sb_ctx["wm"] is not None` exactly |
| `sb_dump_heads(..., refit=, extra=)` | two new keyword arguments, both defaulted; the blob gains `proj_final: None` and `proj_skew_resolved: False`, which no existing consumer reads |
| the era-boundary dump | keyed on `vo_dump and vo_dump_era` |
| `obs_extra["adm"]` | `None` on every arm without the knob |
| `readgate.gated_walk(..., groups=False)` | a default-off keyword in `../preplay/readgate.py`; pp4/pp5's candidates are single rows and take the donor's path. The `groups=True` path refuses `read_pair`, whose changed-set test assumes one added row |

## Arms

| arm | `admit` | yoke source | role |
|---|---|---|---|
| `st_gw_yk` | `world` | `vo_s3:voi3_dp` | pp4's rule in the loop; the only arm billed for its auditions |
| `st_gr_yk` | `read` | `vo_s3:voi3_dp` | pp5's strict pooled read gate in the loop, at zero world queries |
| `st_gn_yk` | `none` | `vo_s3:voi3_dp` | the walk's own floor |
| `st_pf_gw` / `st_pf_gr` / `st_pf_gn` | as above | `voi3b_pf_src` | preflight twins, governance ON |
| `st_gy_yk` | `yield` | `vo_s3:voi3_dp` | [r2] the value read in the next level's currency: next-level at-support share × the projection's level, pooled, does not fall |
| `st_gd_yk` | `demand` | `vo_s3:voi3_dp` | [r2] the free comparator: the candidate is a half of ≥ 1 next-level key at support (silent where the next level holds none) |
| `st_pf_gy` / `st_pf_gd` | as above | `voi3b_pf_src` | [r2] preflight twins, panel seeded |

Banked and **not** re-run, read at reduction time:

| banked arm | what it is |
|---|---|
| `sb_s1:sb_sv_yk` | **the no-walk reference** — the arm all three are one knob from |
| `vo_s3:voi3_dp` | the seed-0 anchor, and the yoke SOURCE |

## Gates

| gate | where | claim |
|---|---|---|
| **G-F** | `fidelity_smoke` | with every `# [sostenuto]` knob off this file replays `soundboard.py` bit for bit, against a donor self-replay control |
| **A-0** | `sostenuto_gates` | the three arms are `sb_sv_yk` plus exactly one key |
| **A-1a** | `sostenuto_gates` | the walk reproduces pp4's HAND-COMPUTED sequence: `[0, 2, 4]`, 3 rejected, 7 auditions, final 0.10 |
| **A-1b** | `sostenuto_gates` | `sos_walk` == `readgate.gated_walk(groups=True)` on 72 designed cases, all three gates, groups of 1 and 3, with the gates shown to disagree on 24 of 24 |
| **A-1c / A-3** | in-loop, once per arm | ONE REAL PASS reproduces `readgate.gated_walk` on the same base, groups, order and pool |
| **A-2** | `sostenuto_gates_gpu` | the in-loop read == `preplay.pj_predict` elementwise |
| **A-4** | `sostenuto_gates`, in-loop | handed the world's success as its level, the read gate IS the world gate (pp5's P-2) |
| **A-5** | `sostenuto_gates`, in-loop | a constant level admits everything (pp5's P-3) |
| **A-6** | in-loop, every pass | the gate, test and practice pools are pairwise disjoint (pp5's P-4) |
| **A-7** | `sostenuto_gates` | the admission set governs what `operative` serves |
| **A-8** | `sostenuto_gates` | `sos_key_of` == `ClassMiner.observe`'s keying, including under a merge |
| **A-9** | in-loop / arm file | `read` and `none` bill zero world reads for their decisions |
| **A-10** | `sostenuto_gates` | `extend_candidates` is inert under a quotient and correct without one — the finding, gated |
| **A-11** | `sostenuto_gates` | the offer order is the loop's own count order, capped |
| **A-1r … A-4r, A-9r** | `sos_gate_run`, via `preflight_gates` and the reducer | the run-level forms, off the arm file, with the inertness-shaped blindness closed |
| the donor's | unchanged | G-F, V-*, M-*, P-*, S-*, Y-1, and the 100-perturbation harness |
| **A-12 … A-15, A-4y, A-5y, A-5d** | `sostenuto_gates` (+ in-loop for A-4y/A-5y/A-5d) | [r2] the yield and demand rules on designed objects; 11/11 perturbations RED (DESIGN §9.5) |
| **A-1c sep.** | in-loop / `sos_gate_run` | [r2] a pass where the arm's OWN gate admits a different set from another gate was found |
| **E-Y / E-D** | `sos_export_gate`, via `preflight_gates` and the reducer's [K5] | [r2] the share IS `sb_yield_label` and the demand IS a hand count, off an exported pass |

## Runs

| tag | function | what | app id |
|---|---|---|---|
| — | `sostenuto_gates` (r2) | round 1's block unchanged **PASS**, 8/8 RED; and A-0 (five arms), A-12 (yield separates from read on 23 of 24 designed cases), A-4y, A-5y, A-13, A-14, A-5d, A-15 **PASS**, **11/11 RED** | `ap-ehTdbhcIa7Sni2DpN3aoWi` |
| — | `sostenuto_gates` | A-0, A-1a, A-1b, A-4, A-5, A-7, A-8, A-10, A-11 **PASS**; falsification **8/8 RED**. A-1b's 72 cases separate the three gates on 24 of 24 designed sequences. One perturbation was BLIND on the first run — the A-7 filter keyed on the child row is invisible at L2, where the lower table is `base_table(v)` and a child index IS its flat key; re-pinned at L3 over a real L2 table | `ap-QiZ9eB3uX8seEnlX2CKekP` |
| — | `sostenuto_gates_gpu` | A-2 **PASS**: the in-loop read equals `preplay.pj_predict` elementwise, `max\|Δ\| = 0.000e+00`, over five (blk0, span) cells; falsification (mask off / `blk0` off by one) **4/4 RED** | `ap-BkeW6WbPGplT6LkFYzps2z` |
| `st_gf1` | `fidelity_smoke` | G-F. **PASS**: `max\|fork − soundboard\| = 0.000e+00` on both arms, donor self-replay control `0.000e+00`, commits equal; Q-11 interface check PASS on 18 cells | `ap-0ukuodseMt9sXEMuSUg3j6` |
| `st_pf1` | `preflight` | 4 twins: `voi3b_pf_src`, `st_pf_gw`, `st_pf_gr`, `st_pf_gn`. **PASS**: 27 passes and 153 candidates offered per arm; A-1c (repro against `readgate.gated_walk`), A-4 and A-5 in the loop; A-6 pool overlap 0 at every pass; **A-9 the bill — `world` 3312, `read` 0, `none` 0**; no commit cancelled, no fallback, no re-key collision. **The walk gates could not REJECT there, for two preflight-scale reasons named before the run**: `extend_tol = 1.0` at preflight, so the world's decision is always admit; and the readout is degenerate on all 21 refits (`fit_base = 0.0` — the substrate solves nothing, `soundboard` DESIGN §7.4), so its level is a constant and A-5 applies. The three arms were therefore bit-identical, which is itself the inertness statement. The rule's falsification lives in A-1a/A-1b on DESIGNED sequences, which is pp5 §9.3's prescription | `ap-xK71fyqIJr3XfML93K75PS` |
| `st_s1` | `sweep` | seed 0, `st_gw_yk`/`st_gr_yk`/`st_gn_yk`, one container each, yoked to `vo_s3:voi3_dp`. **COMPLETE**, 201 cycles per arm, 8837 / 8820 / 8388 s (2.45 / 2.45 / 2.33 GPU-h), 14987 s wall (container scheduling, not work), 0 errors; peak RSS 6984 / 7032 / 7053 MiB against the 12288 request. Every gate green: A-1c re-walked 28 / 27 / 29 passes through `readgate.gated_walk` with 0 plumbing errors, a SEPARATING pass found on all three at c10 L2, A-6 pool overlap 0, **A-9 the bill — `world` 13,632 world reads, `read` 0, `none` 0** | `ap-3pYFwA3LfeCfeIyZoccBb8` |
| `st_s2` | `sweep` | seed 2, the same three, yoked to `vo_s3d2:voi3_dp`. Launched after the seed-0 reduction on the SPEC's condition. **PARTIAL**: `st_gw_yk` (7721 s, 2.14 GPU-h) and `st_gn_yk` (8203 s, 2.28 GPU-h) complete; `st_gr_yk` lost to `RuntimeError: CUDA error: unknown error` — a container fault, not a code fault (the same binary ran all three at seed 0). `sweep`'s assertion surfaced it and the merge kept the two good arms | `ap-ylKRwrDaYeSuEe3Pxp16Wg` |
| `st_s2` | `sweep --arms st_gr_yk` | the read arm alone, relaunched. **COMPLETE**, 201 cycles, 8454 s (2.35 GPU-h), 38.54 s/cycle, peak RSS 7005 MiB, 0 errors; every gate green | `ap-HfAKLrIiKnT6GYEgNAl4yR` |
| `st2_gf1` | `fidelity_smoke` | [r2] G-F on the round-2 binary. **PASS**: `max\|fork − soundboard\| = 0.000e+00`, control `0.000e+00`, commits equal | `ap-njiN3b9nizjjRGg5cVk4vg` |
| `st2_pf1` | `preflight` | [r2] `voi3b_pf_src`, `st_pf_gy`, `st_pf_gd`. The whole path ran (walk, counts, share, exports, A-1c on every pass); E-Y/E-D **PASS** on all four exports. **ONE RED, mine**: `sos_gate_run` asserted a separating pass and the bounded search (8 blocks) was spent by c6, before the demand gate's first refusal at c8 L4. Corrected: on a round-2 arm the block also runs on the first pass where the arm's own gate refuses anything, and the run-level gate requires a separating pass iff the arm refused anything | `ap-QORzzeT6EJsJ4KQgE10R4N` |
| `st2_gf2` | `fidelity_smoke` | [r2] G-F on the corrected binary. **PASS**, `0.000e+00` / `0.000e+00` | `ap-xaBcyveaxAMayd9lCxJWaM` |
| `st2_pf2` | `preflight` | [r2] the same three. **PASS**. 27 passes per twin, A-1c re-walked all 54 level-passes, pool overlap 0, **0 world reads billed on both**. `st_pf_gd` refused 1 key (c8 L4, demand 0) and its separating pass was re-walked there; `st_pf_gy` refused nothing at preflight scale (the world admits everything at `extend_tol` 1.0 and the readout is constant), so it has none, which is reported rather than passed. 72 silent offers per twin (the unseeded L6). E-Y and E-D **PASS** on all four exports; on-export perturbations 3/4 RED, "floor dropped" BLIND (the seeded panel puts every key at count ≥ 4; A-13 carries it) | `ap-yeNSqerB5P2dT7XF3XxMKh` |
| `st2_s1` | `sweep` | [r2] seed 0, `st_gy_yk`/`st_gd_yk`, one container each, yoked to `vo_s3:voi3_dp`. **COMPLETE**, 201 cycles per arm, 7892 / 7689 s (2.19 / 2.14 GPU-h), 7963 s wall, 0 errors; peak RSS 7050 / 6927 MiB against the 10240 request. Gates: `st_gy_yk` all green (separating pass c5 L2, A-1c on 26 level-passes, 0 world reads, E-Y/E-D on both exports with all four perturbations RED); `st_gd_yk` all green EXCEPT **A-3r / Y-1: the L5 commit at c186 was cancelled** (the admitted L4 build held one key; DESIGN §9.9.2). Twin-prefix fidelity: both replay `st_gn_yk` bit for bit until their first refusal (yield refuses at c10 and diverges at c16; demand refuses at c5 and diverges at c8) | `ap-O8PARgV8zTaSsDy1XoULo7` |
| `st2_s2` | `sweep --arms st_gy_yk` | [r2] seed 2, the yield arm alone (DESIGN §9.9.4), yoked to `vo_s3d2:voi3_dp`. **COMPLETE**, 201 cycles, 7163 s (1.99 GPU-h), 32.16 s/cycle, peak RSS 6919 MiB, 0 errors. Gates green EXCEPT **A-3r / Y-1: the L4 commit at c157 was cancelled** (4 L3 keys admitted, the L4 build over them empty; DESIGN §9.9.5); L4/L5 never walked. Twin-prefix fidelity PASS (first refusal c5, diverges c8). E-Y/E-D PASS on both exports, perturbations 4/4 RED | `ap-X4DLbInI8H60M9JaMeOxAK` |
| `st_s2` | `sweep --merge-only` | the three-arm merge restored after the one-arm sweep rewrote `summary.json`. 11 s on CPU. It cannot recover `elapsed_s_per_container` (that comes from the live call's return), so seed 2's container times in [H] are estimated as `s/cycle x cycles` and marked `*` | `ap-wQQw73Tem9tBCl3koUInAG` |

Volume `rhm-scaling-data:/rhm_practice_sostenuto/<tag>/` (the per-arm containers write
`<tag>__<arm>/` and the coordinator merges them into `<tag>/`); compact local mirrors and
reductions under `figures/` — [r2] `st2_s1_reduction.txt` / `st2_s2_reduction.txt` are round 2's
reductions at seed 0 (five arms) and seed 2 (four: round 1's three read from `st_s2`, and the yield
arm), `st2_seedtable.txt` round 2's two-seed table (never averaged), and
`figures/st2_s{1,2}/q_coverage.png` the coverage trace; `st_seedtable.txt` is round 1's two-seed table of record,
`st_s1_reduction.txt` / `st_s2_reduction.txt` the per-tag reductions, and `figures/<tag>/q_*.png`
the four panels. `setup.json` is **gzipped** in the mirror (58 MB uncompressed, almost all of it
the true tables written out flat, which `rule_seed = 0` reproduces exactly). Each arm carries its
three dumps beside `results.json`, plus `vo_heads_e1..e5.pt` (this node's per-era line) and
`proj_final` inside `vo_heads.pt` (the readout re-solved against the core as it stands, so
preplay's one-update skew is resolved rather than passed on).

## Reproduction

See [`DESIGN.md`](DESIGN.md) §8.

[^private]: Not mirrored: this link points to a document in the private lab repo (the roadmap, the queue, an unrun spec, reading notes, or a conversation). See the top-level README for what is held back and why.
