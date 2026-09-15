# overtone — FILES

The machinery record for the readout round: every flag, every gate, every run. The decisions and
the withdrawals are in [`DESIGN.md`](DESIGN.md); the node this is a child of is
[`../README.md`](../README.md) and its own record is [`../FILES.md`](../FILES.md).

## Status

Round in flight (2026-09-14). Substrate edits live in `../voicing.py`, every one of them
`# [overtone]`-marked and every knob default off; with them off the file is `voicing.py` at Q3b
and the anchor replays at 0.000e+00. This folder holds the round's record, its gates' falsifier,
its run scripts and its reduction.

## Code files

| file | purpose |
|---|---|
| `gates/falsify_ov.py` | the round's falsification harness — every gate shown to fail before it is reported (26/26) |
| `results/RUN_ov_pf1.sh` | the preflight of record (gates, 7 arms) |
| `results/RUN_ov_s0.sh` | the tag of record (3 arms, seed 0, three containers, one tag) |
| `figures/` | the reduction of record and any figures |
| `DESIGN.md` | decisions, withdrawals kept beside their corrections, defects |

The round adds no new module: the substrate is `../voicing.py` and the reducer is
`../analyze_voicing.py`, both extended in place and both backwards-compatible (the banked
`vo_s3b` reduction re-reduces byte-identically apart from the `--bank` line).

### What changed in `../voicing.py` (every hunk `# [overtone]`-marked)

| object | change |
|---|---|
| `Critic` | `hidden_mult = 0` → `Linear(dim,1)` readout; `ctx_mode = "mean"` → context is `ctx(pooled.mean)` alone. Both default to the donor's behaviour. |
| `build_critic` | passes `ctx_mode` through; minting discipline unchanged |
| `ov_build_shadows` | mints the shadow readouts and **their own** `Adam` (never `gopt`) |
| `ov_z`, `ov_pick_disagree` | the standardisation `vo_compose` uses, and the S3 allocation rule as a named function so it is falsifiable |
| `ov_row_index` | candidate tuple → operative-table row, cached per table (`vo_class_groups`' idiom), −1 off-table |
| `ov_free_scores` | the four zero-verdict scores off tensors the audit already has |
| `ov_logit_oof`, `ov_irls` | the out-of-fold logistic combination and its estimator |
| `ov_oof_auc` | the out-of-fold AUC taken **per fold** and averaged by pair count (defect #5) |
| `ov_dump_heads` | banks `vo_heads.pt` — the critic, the shadow readouts and the trunk — so a later audit change is a CPU job, not a re-run |
| `ov_dump_rows` | the end-of-run `vo_rows.npz` + `_meta.json` |
| `VoRecorder` | `pmask` (the probe rows' draw tag, in lockstep with `pbuf`), `shadow`, `shadow_opt`, `dp_row_cache` |
| `VoRecorder.push_probe` | carries the draw tag; a part with no tag is uniform by definition (backwards-compatible) |
| `vo_critic_terms` | trains the shadows on the critic's own batch and steps their own optimizer |
| `vo_critic_audit` | `s=` keyword; shadow AUCs, the free scores, the combination, and the probe half's `unif`/`dis` split |
| `vo_run_probes` | `dis` / `unif_frac` / `critic` / `core` / `chunk`; returns a 4th element per part |
| `run_arm` | builds the shadows, honours `ov_critic_hidden`, writes the dump before `write_results` |
| `voicing_run` / `preflight` | the run-level flags; peak-RSS print at the end of a tag |
| `ARMS` / `TWIN` | `ovt_comp_pr_{sh,dis,lin}` and `ovt_pf_{comp_pr,noshadow,dis,lin}` |
| `ov_gate_r1` / `ov_gate_r6` / `ov_gates_cpu` | the round's gates |
| `vo_probe_offstream_check`, `gates/falsify.py` | unpack the probe part positionally (defect 1) |

## Flags and knobs

Every one is `# [overtone]`-marked and defaults off. Run-level knobs are instruments every arm
in a tag shares; per-ARM knobs change what the arm **does** and live in `ARMS[...]["cfg"]`,
which is this file's own rule (an arm IS its chooser). An arm cfg overrides the run-level value,
which is what makes gate R-1's in-run pair possible.

| knob | level | default | what |
|---|---|---|---|
| `ov_shadow` (`--ov-shadow`) | run | `""` | **[S1]** comma list of shadow readouts: `lin` = `Linear(dim,1)` over the critic's own state · `dir` = `Linear(dim,1)` over `pooled.mean` + candidate content, no slot (the Steenwyk shape). Trained on the critic's own rows, audited on the critic's own held-out split, never consulted by a chooser. |
| `ov_free` (`--ov-free`) | run | False | **[S2]** the zero-verdict scores in the audit: `dp`, `dpz`, `conf`, `marg`, plus the out-of-fold logistic combination and the prior-alone fit |
| `ov_comb_folds` (`--ov-comb-folds`) | run | 2 | **[S2]** folds for that combination, taken from the audit's bijective hold code |
| `ov_dump` (`--ov-dump`) | run | False | write `vo_rows.npz` (+ `_meta.json`) per arm at end of run |
| `ov_dump_cap` (`--ov-dump-cap`) | run | 0 | rows per slot per buffer in the dump; 0 = the whole buffer |
| `ov_probe_unif_frac` (`--ov-probe-unif-frac`) | run | 0.25 | **[S3]** share of `vo_probe_n` still drawn uniformly, so the audit keeps a candidate set comparable to the banked uniform twin's |
| `ov_probe_dis` | arm | False | **[S3]** spend `vo_probe_n` on the on-table candidate of another class maximising \|z(dp/span) − z(critic)\|, at the same bill |
| `ov_critic_hidden` | arm | −1 | the **governing** critic's `hidden_mult`; −1 = `span_hidden_mult` (the donor's MLP), 0 = a linear readout in the composed chooser's seat |

## Gates

`../FILES.md` holds the node's own table (18/18 `gates_cpu`, 36/36 falsification). This round
adds eight, all shown to fail (26/26, `gates/falsify_ov.py`). The node's tables are unmoved:
`vo_gates_cpu` still reads **18/18** and `gates/falsify.py` still reads **36/36** against the
edited file.

| gate | claim | where | shown to fail on | status |
|---|---|---|---|---|
| **R-1** | **The shadows are INERT, with governance ON**: an arm carrying the S1 shadows and the S2 scores is its shadow-free twin at 0.000e+00 on every behaviour series, commits equal. Non-vacuity: both twins governed, the shadow arm actually trained a shadow, the bare twin carries none. | `ov_gate_r1`; `preflight` (`ovt_pf_comp_pr` vs `ovt_pf_noshadow`); and at **full scale** against banked `vo_s3b:voi3b_comp_pr_yk` in reduction [R] | one `e` value moved by 1e-9 · the shadow arm never trained one (VACUOUS) · the bare twin carries shadows (VACUOUS) · neither twin governed (VACUOUS) | **PASS** (`ov_pf1`) — max\|Δ\| **0.0**, commits equal, 80 shadow steps, both shadows trained, bare twin carries none, both governed |
| **R-2** | **The shadows are LIVE** (R-1's dual, §30's rule): their parameters move, on the critic's own rows (`n_train` equal); their gradient reaches **neither the critic nor the trunk** (0 params carrying a gradient after their step); and the critic lands where it would have landed without them, at 0.000e+00. | `ov_gates_cpu` | DISCONNECTION: the shadow optimizer is never stepped · an aliased head whose optimizer reaches the critic | **PASS** |
| **R-3** | **The free scores are the file's own quantities and cost no draw**: the DP score read for a candidate IS `vo_dp_scores_from_logits`' entry at that row (identity); `conf`/`marg` are the trunk's entropy and margin at the slot's blocks (measured — a different summation order); the read is RNG-neutral. | `ov_gates_cpu` | the DP column shifted by one row · `conf` read as the mean logit · the read draws from the shared stream | **PASS** |
| **R-4** | **The combination is OUT OF FOLD, and its AUC is taken PER FOLD**: 40 noise features over 120 coin flips rank at chance out of fold and materially above it in fold, using the same estimator; a label that IS a logistic function of a feature is still recovered; and on rows whose folds carry **different base rates**, the per-fold average recovers the raw single-feature AUC where pooling the folds' scores does not. | `ov_gates_cpu` | the fit is in fold (scores the rows it trained on) · reading the POOLED figure (defect #5, added after it shipped into `ov_s0`). *Not caught*: replacing the bijective folds with a coin flip — the bijective split is inherited from `hold_code` and gated there. Stated, not claimed. | **PASS** — raw 0.671, per-fold 0.683, pooled 0.502 |
| **R-5** | **The disagreement draw stays inside §26's claims**: every candidate on-table; every candidate a DIFFERENT class (asserted exactly, on a construction where every row writes the same class); the chosen **candidate** IS the argmax of \|z(dp)−z(critic)\| off that class; exactly `round(unif_frac·take)` rows tagged uniform; billed row for row; the filed buffer untouched. | `ov_gates_cpu` | the rule picks agreement (argmin) · the uniform share dropped · DISCONNECTION: `dis` ignored · the rule may re-grade the class that was written | **PASS** |
| **R-6** | **The re-aimed budget is LIVE** (R-5's dual): with governance ON, an arm one allocation-boolean from its twin drew rows by the rule, paid the same bill, and is NOT identical to it. The size is measured, never asserted. | `ov_gate_r6`; `preflight` (`ovt_pf_dis` vs `ovt_pf_comp_pr`) | DISCONNECTION: re-aimed and nothing moved · `ov_probe_dis` on and no row drawn by the rule · the uniform twin drew disagreement rows (VACUOUS) · neither twin governed (VACUOUS) | **PASS** (`ov_pf1`) — max\|Δ\| 95.0 on `t_cum` from cycle 9, **183** rows drawn by the rule, **0** in the uniform twin, both governed |
| **R-7** | **The readout's shape**: `hidden_mult = 0` is ONE affine map of the critic's state, checked against a hand-written `W(u+e)+b`; the default still builds the donor's `Sequential`; `ctx_mode="mean"` drops the slot embedding and the per-offset reads. | `ov_gates_cpu` | `hidden_mult=0` falls through to the MLP · `ctx_mode='mean'` ignored | **PASS** |
| **R-8** | **The dump is faithful**: the arrays round-trip the buffers exactly (the int16 cast raises if lossy); the stored DP column IS the audit's score through the same core; and the banked heads reload to a critic that scores identically. | `ov_gates_cpu` | a lossy cast on `obs` · a DP column that is not the audit's · the rows reversed against their verdicts | **PASS** — reloaded critic \|Δ\| 0.000e+00 |
| G-F | with every knob off the fork still replays `enharmonic.py` at 0.000e+00. **Mandatory this round**: the edits touch `finetune_generator_span`, `vo_run_probes` and `run_arm`, all shared paths. | `fidelity_smoke` (`ov_gf1`) | (the node's own; inherited) | **PASS** (`ov_gf1`) — 0.000e+00 on both arms, donor self-replay control 0.000e+00, commits equal |
| Y-1 | the replay matches on every yoked arm, 0 cancelled | `vo_gate_yoke`; `preflight` and reduction [Y] | (the node's own; 5 perturbations) | **PASS** (`ov_pf1`) — exact on all **six** yoked twins, commits [6], advances [6,12,17,22,27], **0 cancelled** |
| V-4A/B, V-5, V-6b | the node's own, re-run because the shared paths moved; V-4B's probe-bill residual must still be exactly 0 | `vo_gate_v4_v5`, `vo_gate_v6b`; `preflight` | (the node's own) | **PASS** (`ov_pf1`) — V-4B max\|Δ\| **0.0**, 329 probes, **bill residual 0.0**; V-5 all four objects equal, 329 rows across 16 slots and nowhere else |

## Runs

| tag | what | app id | result |
|---|---|---|---|
| `ov_gf1` | G-F, `fidelity_smoke` against the donor with every knob off | `ap-7BuadT3MrKaHZkA3FuKjk3` | **PASS** 0.000e+00 |
| `ov_pf1` | preflight: 7 arms, the node's gates + R-1 + R-6 | `ap-CH2E6NpBLY9mOF4nwyCfIn` | **ALL PASS** |
| `ov_s0` | the tag of record: `ovt_comp_pr_sh`, `ovt_comp_pr_dis`, `ovt_comp_pr_lin`, seed 0, yoked to `vo_s3:voi3_dp` | `ap-AkEY7j5GabMT6fNbIIgHH7`, `ap-4Xn0sT4kSXTK3hl26Ku6gT`, `ap-ZLKwS8ShijGgyrd4nzWASc` | **DONE** 1.92 + 1.80 + 1.44 = **5.16 GPU-h**, 201 cycles each, Y-1 exact on all three (0 cancelled), **R-1 at full scale 0.000e+00 on all eleven series** |
| `ov_gf2` | G-F re-run after the defect-#5 repair touched `run_arm` and the audit | | **PASS** 0.000e+00, control 0.000e+00 |
| `ov_s0b` | the one-arm re-run with the corrected per-fold estimator, + `vo_heads.pt` | `ap-Ck8Dee3VUj3NMZWdR2AGxD` | **DONE**, 201 cycles. **Bit-identical to `ov_s0:ovt_comp_pr_sh` AND to banked `vo_s3b:voi3b_comp_pr_yk` at 0.000e+00 on all eleven series including `t_cum`**, commits equal — so the estimator lives in an instrument, as claimed. Banks `vo_heads.pt` (3.2 MB: critic + both shadows + trunk). **[J2] here is the S2 column of record.** |
| `ov_s2` | the S3 arm at **seed 2**, yoked to banked `vo_s3d2:voi3_dp`, one arm | `ap-f9Nf6UmoYl5rysfNa1CyzF` | **DONE**, 201 cycles. **Y-1 exact** — commits [59, 77, 157], advances [60, 110, 180, 192, 201], **0 cancelled**. Bill 205,967 probes, 154,471 (0.750) by the rule, share 0.0059 mean / 0.0080 max. Banks its own rows and heads. |

Peak RSS on the paid arms is **6,860–6,989 MB** against the donor's inherited `memory=32768`
request; a future launch of this runner can ask for ~10 GB. Each mirrored tag is ~140 MB
locally, of which 60 MB is `setup.json`'s true-table dump — the lineage's norm (`vo_s3b` and
`vo_s3` are 144 and 130 MB), not this round's addition. The row dumps are 3.2–3.5 MB per arm
for 420–427 k rows over 60 keys, 99.5–99.8% of them on the final operative table.

Volume `rhm-scaling-data:/rhm_practice_voicing/<tag>/`; compact mirrors under `../figures/<tag>/`
(that is where `fetch_compact.py` puts them), the reduction under `figures/` here.

## Reduction

`../analyze_voicing.py` gained four sections, all additive and all silent on a tag that does not
carry the round's knobs:

| section | what |
|---|---|
| **[J1]** | the shadow readouts against the MLP critic, per slot, **on the same held-out rows** — FILED and PROBE apart, medians over the run's reads and the last read, pooled at the end |
| **[J2]** | the zero-verdict scores: `dpz`, `dp`, `conf`, `marg`, the critic on the same sub-population, the prior-alone out-of-fold fit, the combination, and `incr = comb − max(prior, crit)` |
| **[J3]** | where the probe budget went: uniform against disagreement, with the AUC read on each and the header's warning that the `dis` column is not comparable to the bank |
| **[R]** | gate R-1 at full scale: the shadow-carrying arm against the banked Q3b cell, per series, with `t_cum` in the strict set |
| **[Z2]** | the same repair accuracy and `moved` share **pooled by level**, treated minus banked, n-weighted with the convention stated |
| **[S1b]** | (in `overtone/analyze_dump.py --post`) the **post-write linear probe** on the banked trunk, against a random-trunk floor, a pre-write control, the MLP recomputed from `vo_heads.pt` and the prior — all on identical rows |

**A denominator warning for the `--post` section at other tags**: it needs `vo_heads.pt`, which
is banked from `ov_s0b` on. `ov_s0`'s three arms have rows but no heads, so `--post` skips them
by name rather than failing.

`[J]`, `[I]` and `[Z]` now also read banked arms, so the round's arms and `vo_s3b`'s sit in one
table.

## Environment

As `../FILES.md` §Environment. On this box there was no `<repo>/.venv`; the system interpreter
was given `numpy==1.26.4` (which `modal` needs to import `voicing.py` before it ships anything)
and `torch==2.7.0+cpu` from the PyTorch CPU index, which is what the offline gate tables run on.

```
cd experiments/            # MODAL_PROFILE=chromatic
PYTHONPATH=. python3 -c "from rhm.practice.voicing import voicing as V; V.vo_gates_cpu(); V.ov_gates_cpu()"
PYTHONPATH=. python3 rhm/practice/voicing/gates/falsify.py                       # 36/36
PYTHONPATH=. python3 rhm/practice/voicing/overtone/gates/falsify_ov.py           # 26/26
```

### Two coverage holes in `ov_pf1`, said out loud

The preflight's audit reached its 32-row / 16-held-out minimum only once or twice per arm, and
its **probe** buffer never reached it at all (`probe_x_reads: 0`, `comb_reads: 0`). That is the
same preflight artifact `../DESIGN.md` §34 records for `vo_critic_audit` itself — the yoked twins
commit once, so no slot accumulates rows — and it means two branches did **not** run in
substrate: the probe half's `unif`/`dis` split and the out-of-fold combination. Both were
exercised locally on the real functions with real tensors (a 400-row synthetic buffer through
`vo_critic_terms` and `vo_critic_audit`, producing populated `probe_x` and `comb_auc`), and at
full scale each slot carries 600–900 held-out rows, far past both minima. Stated so it is on the
record rather than discovered in the reduction.
