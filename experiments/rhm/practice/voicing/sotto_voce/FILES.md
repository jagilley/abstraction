# sotto voce — FILES

The machinery record: every file, gate, run, flag and diagnostic, with the run tables. Decisions
live in [`DESIGN.md`](DESIGN.md); the orchestrator's brief is [`SPEC.md`](SPEC.md). **Up**:
[`../FILES.md`](../FILES.md) (`voicing`). **Parent node**: [`../`](../README.md) —
`sotto_voce.py` forks `voicing.py` at its `vo_s3b`/`vo_s3e` head and imports nothing from it, so
the parent is untouched.

Results are facts-only. Nothing in `../` or `../../enharmonic/` has been edited by this node.
The writeup is [`README.md`](README.md), written after the two-seed discussion with Jasper; the conversation is `CONVERSATION.md`[^private].

## Status

| | |
|---|---|
| Q0 | **done**, offline, CPU, no GPU. The mirror sized before a GPU was spent: architecture, committee, agreement quantile, and the counterfactual degradation on a proxy corpus. `figures/so_q0_reduction.txt`; decisions in `DESIGN.md` §4–§5. |
| gates | **PASS**. `vo_gates_cpu` **24/24** (18 inherited + M-1, M-1b, M-2/M-2b, M-4, M-3, M-5 across six new checks); `gates/falsify.py` **56/56** (36 inherited + 20 new perturbations). |
| G-F | see the run table (`so_gf1`). |
| preflight | see the run table (`_preflight_s1`). |
| `so_s1` | **done** 2026-09-15, seed 0, 201 cycles/arm, peak RSS 7,045 MiB. **Y-1 exact on all three arms, 0 cancelled**, so L4/L5 cells exist. Reduction `figures/so_s1_reduction.txt`; facts in `DESIGN.md` §9. |
| `so_s2` | **done** 2026-09-15, seed 2, 201 cycles/arm, peak RSS 6,959 MiB at `memory=12288`. **Y-1 exact, 0 cancelled** (commits [59,77,157] — **L4 only, no L5**). First tag carrying the cumulative `n_append` counters, so M-2r's strict cap-independent identity holds on it. Reduction `figures/so_s2_reduction.txt`; two-seed table `figures/so_seedtable.txt`; facts in `DESIGN.md` §10. |

## Code files

| file | purpose |
|---|---|
| `sotto_voce.py` | The substrate: `../voicing.py` forked at its `vo_s3b` head, every addition `# [sotto]`-marked and every knob default off (with them off, `vo_om_mode` is `"world"` and the file IS `voicing.py`; with `vo_record` also off it is `enharmonic.py`, which is what G-F asserts). Adds: `_build_outcome` / `build_outcome` (the tree net and its minting discipline); `VoOutcomeBank` (the committee, its experience buffer, its two doors, its step, its agreement threshold); `VoRecorder.om` / `.mirror` / `.probe_rows` / `.wbuf`; `vo_run_probes(outcome=, mode=, freq_share=, sample_cap=)` returning `(filed rows, BILLED count, world rows)`; `vo_critic_audit`'s `probe_auc_world` column; the `vo_om` log series and the arm-level `vo_om_state` / `vo_mirror` / `vo_probe_rows`; gates `vo_mirror_check`, `vo_mirror_loss_check`, `vo_mirror_source_check`, `vo_om_live_check`, `vo_gate_mirror_run`; `_rng_same`; `YOKE_REMOTE`; the peak-RSS readout. |
| `q0_sotto.py` | Q0, offline, CPU. Builds the world exactly (`v=8, s=2, depth=6, m=2, rule_seed=0`), a proxy experience corpus and the probe channel's own substitution, fits one outcome model and a K=5 committee, and reports [S1] the corpus, [S2] one model on experience and on counterfactuals, [S3] the committee and the agreement rule at each quantile, [S4] disagreement as a detector of the model's own error, [S5] the blind region by level and by the substituted row's frequency. Pure numpy + torch CPU; no Modal, no volume. |
| `analyze_sotto.py` | The reducer: `../analyze_voicing.py` forked, sections [A]–[I] and [J]–[O], [S], [Y], [Z] kept working on the banked `voicing` tags (the loader searches this node's `figures/`, the parent's, and `enharmonic`'s). [J] gains `auc_world` (the probe AUC scored against the WORLD's verdict rather than the filed one) and `filed==world`. New: **[P]** the mirror against the world, on held-out experience and on the probes it graded, with its per-cycle trajectory and the committee's spread and threshold; **[Q]** disagreement as a detector of the mirror's own error, overall and split by support; **[R]** the blind region — the 2×2 of mirror against world per (level, in-support) with the agree/filed fractions; **[T]** the mirror's gate table, re-asserted post hoc by calling the substrate's own `vo_gate_mirror_run`. |
| `gates/falsify.py` | The falsification harness, forked and extended. Every new gate shown to FAIL on a deliberate perturbation: the world's verdict filed into `pbuf`; the world's verdict never recorded (vacuous instrument); a model-graded probe billed; the hybrid billing its agreeing probes; the agreement threshold flipped open and flipped shut; the critic's loss pointed at `wbuf`; `push_world`'s door opened on every arm; a world-marked row appended around the door; a training row whose verdict is not the world's; a mirror never trained; a committee of K copies. **56/56.** |
| `launch_detached.py` | `voicing`'s launcher, retargeted at `sotto_voce.py`. |
| — | `sotto_voce.py::vo_preflight_gates(outdir)` + the CPU entrypoint `preflight_gates(outdir_tag)` — **every file-based preflight gate, re-runnable against SAVED artifacts.** Added because this node's own M-3r tripped after all nine twins had run and threw away every gate result printed after it. Pure functions of the arm files, seconds on CPU, remotely against the volume or locally against a fetched copy; `preflight` calls the same function in place so the two cannot drift. |
| `fetch_compact.py` | `voicing`'s, retargeted at the `rhm_practice_sotto` volume dir. |
| `results/RUN_*.sh` | The commands of record, one per launch, each carrying its own gate table and the reason for every setting. |

## Runs

| tag | what | status |
|---|---|---|
| `so_gf1` | **G-F**, `fidelity_smoke` against `enharmonic.py`, every `# [voicing]` and `# [sotto]` knob off. MANDATORY this round: the probe path, the write recorder and the critic audit are all shared paths this node touched. | see `results/launch_so_gf1.log` |
| `_preflight_s1` | The six Q3b twins plus the three mirror twins `so_pf_{mg,cg,hy}`, all yoked to `voi3b_pf_src`, all with governance ON. Runs Y-1, V-1, V-4A/B, V-5, V-6b (and V-6b's mirror form), and M-1r/M-2r/M-3r/M-4r on the mirror arms. | see `results/launch_s1.log` |
| `so_s1` | Seed 0. `so_mg_yk`, `so_cg_yk`, `so_hy_yk`, `--yoke-from-tag vo_s3`. App `ap-I95XnDRcMskYdSi8igUVHp`. | **done**, 201 cycles/arm. Y-1 exact, 0 cancelled (commits [48,100,151,186], advances [60,110,180,192,201]). M-1r/M-2r/M-3r/M-4r green on all three. `results/RUN_so_s1.sh` |
| `so_s2` | Seed 2. The same three arms, `--yoke-from-tag vo_s3d2`. App `ap-18QMDQVl69baxDsZq5h5dY`. | **done**, 201 cycles/arm. Y-1 exact, 0 cancelled (commits [59,77,157], advances [60,110,180,192,201]) — the seed-2 ladder reaches **L4 only**. M-1r/M-2r/M-3r/M-4r green on all three, M-2r in its strict form. `results/RUN_so_s2.sh` |

Banked and **not** re-run, read at reduction time with `--bank`:

| banked arm | what it is |
|---|---|
| `vo_s3:voi3_dp` | the seed-0 anchor, and the yoke source for `so_s1` (== `en_s9:endo_ledger_open_ung5_ra` at 0.000e+00) |
| `vo_s3d2:voi3_dp` | the seed-2 anchor, and the yoke source for `so_s2` |
| `vo_s3b:voi3b_comp_yk` | **the floor**, seed 0 — composed chooser on the filed diet |
| `vo_s3b:voi3b_comp_pr_yk` | **the ceiling**, seed 0 — composed chooser with world-graded probes |
| `vo_s3e:voi3b_comp_yk` / `_comp_pr_yk` | the seed-2 floor and ceiling |

## Gates

Inherited from `voicing` and re-run unchanged: **VQ-1, G-F, V-1, V-2a/b, V-3, VO-3…VO-12, V-4,
V-4b, V-4A/V-4B, V-5, V-4c, V-4d, V-6, V-6b, Y-1**. Their statements are in
[`../FILES.md`](../FILES.md); their status this round is in the run table above. New below.

| gate | statement | where | shown to fail on |
|---|---|---|---|
| **M-1** | On a model-graded arm the world's verdict EXISTS for every probe (it is in `wbuf` and in the mirror instrument) and NOT ONE filed verdict is the world's. Exact, not on average: the stub mirror says *not solved* on every row and the stub world says *solved* on every row, so the two disagree on every probe and "which was filed" is a count. | `vo_mirror_check`, `vo_gates_cpu`; run form `vo_gate_mirror_run` (**M-1r**) | the world's verdict wired into `pbuf`; the world's verdict never recorded (vacuous instrument) |
| **M-1b** | The world's verdict enters NO LOSS. **Measured**, not read off which buffer the loss names: one critic, one slot, a `pbuf` of model verdicts and a `wbuf` that CONTRADICTS them row for row, one optimizer step each way — the two parameter sets must be equal to the bit. Carries its own non-vacuity half (the step must move the critic). | `vo_mirror_loss_check`, `vo_gates_cpu` | the critic's loss pointed at `wbuf` (reads 2.0e-02 against 0.000e+00) |
| **M-2** | The outcome models train only on their stated sources — experience on every arm, plus WORLD-graded disagreeing probes on the hybrid alone. Two halves: **the door** (`push_world` raises on any arm but the hybrid) and **no back door** (the rows the buffer CARRIES as world-graded equal the rows `push_world` ADMITTED, so a row appended around the door is visible). | `vo_mirror_source_check`, `vo_gates_cpu`; run form **M-2r** | the door opened on every arm (a model-graded probe row fed in); a world-marked row appended around the door |
| **M-2b** | Every outcome-model training row is (a configuration the learner produced, THE WORLD'S OWN VERDICT on it). Re-graded, 0 mismatches — offline in the CPU gate, and in the run on its first `vo_verify_cycles` cycles. | `vo_mirror_source_check`; in-run assert in the cycle loop | one training row's verdict flipped |
| **M-3** | The bill equals EXACTLY the world-graded probe count × `d_fb`: zero on `"model"` and `"committee"`, the disagreeing count on `"hybrid"`. Run form also asserts `probe_priced == probe_ground · d_fb` on the ledger line. | `vo_mirror_check`, `vo_gate_mirror_run` (**M-3r**) | a model-graded probe billed; the hybrid billing its agreeing probes too |
| **M-4** | The committee's agreement rule is what files: on `"committee"` a probe above threshold is NOT in `pbuf`; on `"hybrid"` it is world-graded and returned for the outcome models. The stub's spread alternates about the threshold, so the effect is a count. | `vo_mirror_check`, `vo_gate_mirror_run` (**M-4r**) | the threshold flipped wide open; flipped shut; shut on the hybrid |
| **M-5** | THE MIRROR IS LIVE — the liveness twin, defect #8's shape one organ over. Its step MOVES its members, the members DIFFER from each other (init, bootstrap and draw all differ), `refresh` returns a FINITE threshold from a non-empty held-out split, and the shared torch stream never moves. On a synthetic, learnable target: a statement about the machinery, never about RHM. | `vo_om_live_check`, `vo_gates_cpu` | the mirror never trained (DISCONNECTION); the committee collapsed to K copies of one member |

## Flags and knobs

Every one is `# [sotto]`-marked. With `vo_om_mode` at its default `"world"` the probe channel is
`voicing`'s verbatim and no outcome model is ever constructed; with `vo_record` off as well the
file is `enharmonic.py` (G-F). Per-ARM knobs live in `ARMS[...]["cfg"]` because an arm IS its
grader; run-level knobs are what every arm shares.

| knob | level | default | what |
|---|---|---|---|
| `vo_om_mode` | arm | `"world"` | who answers a probe: `"world"` (`voicing`'s channel verbatim, billed row for row) · `"model"` · `"committee"` · `"hybrid"` |
| `vo_om` | arm | False | build and train the outcome model(s) at all |
| `vo_om_k` (`--vo-om-k`) | run | 5 | committee size; `"model"` forces 1. `committee_head`'s K |
| `vo_om_dim` | run | 64 | the outcome model's width |
| `vo_om_lr` | run | 1e-3 | its own optimizer group, never the plant's |
| `vo_om_steps` | run | 16 | optimizer steps per member per cycle |
| `vo_om_batch` | run | 256 | rows per step |
| `vo_om_cap` | run | 30000 | the experience buffer's cap, in rows (≈30 cycles at this substrate's tip count) |
| `vo_om_min` | run | 512 | rows a member needs before the mirror may grade anything; below it the cycle's probes are SKIPPED entirely (nothing filed, nothing billed) and `n_probe_skipped` counts the cycle |
| `vo_om_boot` | run | 0.8 | per-member bootstrap share, drawn once per row at insertion and stored |
| `vo_om_hold` | run | 0.1 | held-out share of the experience buffer; never trained on, and what the threshold and the mirror's own accuracy are read on |
| `vo_om_agree_q` | run | **0.5** | THE AGREEMENT THRESHOLD: the q-th quantile of the committee's spread on held-out filed rows. Sized offline — `DESIGN.md` §4 |
| `vo_om_freq_share` | run | 0.05 | a class is in a slot's frequently-written set at ≥ this share of the writes filed there |
| `vo_om_sample_cap` | run | 4000 | per-arm cap on the raw per-probe instrument rows kept (`vo_probe_rows`), which is what reduction [Q] reads |
| `yoke_remote` (`--yoke-remote`) | run | `rhm_practice_voicing` | where `--yoke-from-tag` resolves its source, so this node replays the PARENT's banked anchor rather than a copy of it |

## What the run logs

| object | where | what |
|---|---|---|
| `log["vo_om"]` | per cycle | the mirror's buffer size and held-out size, its loss, its held-out accuracy / AUC / base rate against the world, the committee's mean spread and the threshold that spread set, K, the mode, and the two push counters |
| `log["vo"][i]["n_probe_filed"/"n_probe_agree"/"n_probe_world"/"n_probe_skipped"]` | per cycle | who answered this cycle's probes |
| `log["vo"][i]["critic"][slot]["probe_auc_world"]` | per cycle per slot | the critic's held-out AUC on the probe diet scored against the WORLD's verdict (the number the spec asks for on a model-graded arm), with `probe_base_world` and `probe_file_vs_world` beside it |
| `vo_mirror` | arm level | cumulative, per (level, in-support): the 2×2 of the mirror's verdict against the world's, the agree and filed counts, the world's own solve rate, and the summed probability and spread |
| `vo_probe_rows` | arm level | the capped per-probe instrument sample: cycle, level, node, in-support, the mirror's p, the committee's spread, the mirror's verdict, the world's verdict, agreed, filed |
| `vo_om_state` / `vo_om_push_cum` / `vo_om_world_cum` | arm level | the bank's own counters, which gate M-2r asserts against the run's billed probe count |
| `summary["peak_rss_mib"]` | tag level | the container's peak RSS, so the next launch's `memory=` is sized on use |

## Environment

`numpy==1.26.4`, `scipy==1.16.3`, `torch==2.7.0+cpu`, `modal==1.5.5` in the session
interpreter (the Modal CLI imports `sotto_voce.py` locally before it ships, so the interpreter
behind `modal` needs numpy; `launch_detached.py` carries that note). Run from `experiments/`:

```bash
PYTHONPATH=. python3 rhm/practice/voicing/sotto_voce/q0_sotto.py                      # Q0, CPU
PYTHONPATH=. python3 -c "from rhm.practice.voicing.sotto_voce import sotto_voce as V; V.vo_gates_cpu()"
PYTHONPATH=. python3 rhm/practice/voicing/sotto_voce/gates/falsify.py                 # 56/56
python3 rhm/practice/voicing/sotto_voce/launch_detached.py --fn fidelity_smoke --tag so_gf1
python3 rhm/practice/voicing/sotto_voce/launch_detached.py --fn preflight --outdir-tag s1 \
    --arms "voi3b_pf_src,voi3b_pf_dp,voi3b_pf_v4,voi3b_pf_v4pr,voi3b_pf_comp,voi3b_pf_comp_pr,so_pf_mg,so_pf_cg,so_pf_hy"
modal run rhm/practice/voicing/sotto_voce/sotto_voce.py::preflight_gates --outdir-tag s1  # gates only, CPU
bash rhm/practice/voicing/sotto_voce/results/RUN_so_s1.sh
python3 rhm/practice/voicing/sotto_voce/fetch_compact.py --tag so_s1 --fetch --replace
PYTHONPATH=. python3 rhm/practice/voicing/sotto_voce/analyze_sotto.py --tag so_s1 \
    --yoke-src vo_s3:voi3_dp \
    --bank vo_s3:voi3_dp,vo_s3b:voi3b_comp_yk,vo_s3b:voi3b_comp_pr_yk
```

Volume `rhm-scaling-data:/rhm_practice_sotto/<tag>/`; the yoke sources are read from
`/rhm_practice_voicing/`. Compact mirrors and reductions under `figures/`.

## Figures

| file | what |
|---|---|
| `figures/so_seedtable.txt` | **the two-seed table of record** — (a) the chooser at the pooled L2/L3 cell, (b) the critic's probe AUC against the world, (c) the mirror per level and the committee's agreement share per level, (d) the hybrid's bill and its mirror's base rate, (e) the era gaps, (f) the frontier cells with each seed's own denominator. Says at each panel what replicates and what does not; the seeds are never averaged. |
| `figures/so_s2_reduction.txt` | the seed-2 reduction of record, same sections as seed 0's, with the seed-2 floor and ceiling (`vo_s3e`) banked in [Z]. |
| `figures/so_s1_reduction.txt` | the seed-0 reduction of record — sections [A]–[I], [J]–[O], [P]–[R], [T], [Y], [Z], with the banked floor and ceiling in [Z] and a pooled L2/L3 panel beside it. |
| `figures/so_q0_reduction.txt` | the Q0 reduction of record — the proxy corpus, one outcome model and the K=5 committee on experience and on counterfactuals, the agreement rule at each quantile, disagreement as an error detector, and the blind region by level. **Proxy corpus: nothing in it is a measurement of the run.** |

[^private]: Not mirrored: this link points to a document in the private lab repo (the roadmap, the queue, an unrun spec, reading notes, or a conversation). See the top-level README for what is held back and why.
