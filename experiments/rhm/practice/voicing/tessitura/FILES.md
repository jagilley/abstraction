# tessitura — FILES

The machinery record: every file, knob, gate, run and reduction. Decisions and withdrawals live
in [`DESIGN.md`](DESIGN.md). **Up**: [`../FILES.md`](../FILES.md) (voicing). **Siblings**:
[`../overtone/FILES.md`](../overtone/FILES.md) (the readout round this forks its arm from) ·
[`../sotto_voce/FILES.md`](../sotto_voce/FILES.md). **Ancestor of the question**:
[`../../../logit_reading/striatum/norm/`](../../../logit_reading/striatum/norm/README.md).

The interpretation is written up in [`rhm/logit_reading/orbitofrontal/README.md`](../../../logit_reading/orbitofrontal/README.md)
(§1, §2c, §5); [`README.md`](README.md) here is a pointer. The facts are in [`results/`](results/) and in `DESIGN.md`.

Results are facts-only. Nothing in `../overtone/` or `../sotto_voce/` has been edited by this
node; `../voicing.py` has been edited **additively**, every hunk `# [tessitura]`-marked and every
knob default off.

## Status

| | |
|---|---|
| zero-cost pass | **done**, CPU, no GPU. 24 arms with a critic across 7 tags, 3 seeds, 5 diets, 4 graders. `results/logged.txt`. Establishes that the judge's LEVEL is not in the banked record. |
| diets (`refit.py`) | **done**, CPU, no GPU. norm §1 on a fixed trunk: 7 diets, 42,711 identical held-out rows, both trunks. **Spearman ρ = +1.000 against E[outcome] in every family on both trunks**; `norm − E[outcome]` within ±0.03 across a 5× range. The diets RESCALE (slope 0.55 → 1.19) but in the direction OPPOSITE to norm's, and norm's random-trunk control does **not** reproduce — both trunks rescale alike. `results/refit_ov_s0b.txt`, `DESIGN.md` §9. |
| CPU pass | **done**, CPU, no GPU. The two arms that banked `vo_heads.pt`. `results/rows.txt`, gates T-2/T-3/T-5. The norm, the cost side of Q2, and the twins — statically, final critic, buffer tail. |
| gate T-4 | **FAILS, and that is the finding.** `results/structure.txt`. The structural label is not recoverable from the dump; the root is not banked and the reconstruction that infers it is biased upward on 25 of 30 slots. |
| offline gates | **done**. `voicing.py::ts_gates_cpu` 5/5; `gates/falsify_ts.py` 11/12 red + 1 blind by design and named. |
| G-F | **PASS, 0.000e+00** on both arms with the donor self-replay control also 0.000e+00 (`ts_gf1`, `results/gf_and_preflight.txt`). |
| preflight | **PASS** (`ts_pf1`, `voi3b_pf_src` + `ovt_pf_comp_pr`): the callsite resolved at the depth-6 grammar with the three instruments ON at dust budgets — 508 filed writes, `verify_bad = 0`, the critic governed, 116 probes billed and filed. |
| seeds | **`ts_s0` + `ts_s2` both done.** §8's three headlines restated as signs in `results/seed_signs.txt`: the norm's per-era gap direction **does not** replicate and is withdrawn (what does: max \|V_w − base\| = 0.032 / 0.033, both inside 0.04 at every non-init era); the **lag decomposition does** (τ = 50 cycles against the world, 2–5 against the buffer, both seeds); the **panel drift does** in sign once the one initialisation cell is excluded; **cost-versus-structure does**, on filed rows at eras 3–5, with magnitudes agreeing to a hundredth. `DESIGN.md` §8c. |
| re-run | **done** 2026-09-17, `ts_s0`, **1.50 GPU-h**, 201 cycles at 23.6 s/cycle, peak RSS 7.0 GB against a 32 GB request. Ladder realised at the anchor's commits [48, 100, 151, 186] and advances [60, 110, 180, 192, 201]. **Gate TS-1: 0.000e+00 on all eleven behaviour series including `t_cum`**, and **TS-1b: the banked `vo_rows.npz` and `vo_heads.pt` are byte-identical to `ov_s0b`'s**. Reductions `results/ts_s0_rerun.txt` and `results/ts_s0_reduction.txt`; readings in `DESIGN.md` §8. |

## Code files

| file | purpose |
|---|---|
| `reduce_logged.py` | the zero-cost pass over every banked `results.json` with a critic: [W] the world each arm was fed per era and per slot, [E] the era ladder as a within-subject shift with its step sizes and settling times, [B] the ring buffer's own lag (the floor any judge-lag measurement is read against), [A] ranking quality over time with the free prior beside it, [S] the per-slot worlds. Carries gate T-1 (the diet/grader columns asserted against each arm's own config). |
| `reduce_rows.py` | the CPU pass over `vo_rows.npz` + `vo_heads.pt`: [T] the gates, [N] the norm in three derivations against the world on identical rows plus the reliability curve and the shift-vs-rescaling regression, [C] cost at matched prior in striatum §5's conditional form, [X] the twins, [R] the two critics cross-scored on one another's rows. One trunk forward per *distinct* context (gate T-5). |
| `structure.py` | the structural label offline: `free_span_cost` (a min-edit DP to every root with the slot's span left free), `infer_roots`, `repair_labels`, and gate **T-4**, which fails. Run as a script it writes `results/structure.txt`. Nothing in this node reads its output as a measurement. |
| `refit.py` | **norm §1's between-subjects design on a fixed trunk**: the trunk held constant, the critic re-fitted from scratch under diets that differ in expected outcome (the source family `filed`/`probe`/`both`, and outcome terciles taken inside each slot and each source so the slot and level counts and the filed:probe ratio are pinned), each refit read on the IDENTICAL held-out rows, with the random-init trunk carried as norm's own control. Closes the two-whole-readers caveat in `reduce_rows.py` [R]. Gates T-6/T-7/T-8. |
| `reduce_rerun.py` | the re-run's reduction: [N] the norm's time course and its lag, [P] the fixed panels, [C] cost versus structure per era and per slot level, [T] the tag's gates. |
| `plot_tessitura.py` | the figures, drawn from the reduction files so the two cannot disagree: `figures/ts_world.png`, `ts_norm.png`, `ts_twins.png`. PNGs are gitignored under `experiments/`. |
| `gates/falsify_ts.py` | the round's falsification harness — every gate shown to fail on a deliberate perturbation of what it protects; the one blind case is named and explained rather than dropped. |
| `results/RUN_ts_s0.sh` | the command of record for the paid arm at seed 0. |
| `results/RUN_ts_s2.sh` | the same arm at seed 2, yoked to `vo_s3d2`, banked twin `vo_s3e:voi3b_comp_pr_yk`. Run because §8's two headlines are rate-like; `reduce_rerun.py --seeds` turns each into a sign that agrees across the seeds or does not. |

## The hunks in `../voicing.py`

Every one is `# [tessitura]`-marked and every knob defaults off. With all three off the file is
`overtone`'s and the anchor replays at 0.000e+00.

| where | what |
|---|---|
| `_VO_DEFAULTS` | `ts_norm` / `ts_panel` / `ts_struct`, with the reason they exist stated inline |
| `vo_critic_audit._score` | the judge's level on the rows the base rate is read from (`rec["ts"]`), and `"ts"` added to the probe half's `probe_x` passthrough so the probe buffer's level survives |
| `ts_panel_audit` (new) | the fixed per-(slot, era) panel: freeze once, score live, forever |
| `_ts_auc_within` (new) | AUC inside each level of a binary conditioner — striatum §2's matched read |
| `ts_struct_audit` (new) | the structural label per row under the true root, with the judge and the prior on the identical rows and each label held fixed in turn |
| `vo_run_probes` | the root appended as a **fifth** tuple element, gated on `ts_struct` (gate TS-6) |
| `run_arm`'s cycle loop | `_pr_rows = None` hoisted, and the two audits called beside `vo_critic_audit` |
| `voicing_run`, `preflight` | the three flags; ON at dust budgets in `preflight` so every branch runs before a paid setup |
| `ts_gates_cpu` (new) | the offline gate table, TS-2 … TS-6 |

## Knobs

| knob | level | default | meaning |
|---|---|---|---|
| `ts_norm` | run | `False` | the judge's mean predicted P(solve) per slot per cycle, at the row's own candidate (`mean_p`) and over the operative table (`mean_pmax`, `mean_ptab`), beside the base rate already logged |
| `ts_panel` | run | `0` | rows in a fixed panel per (slot, era), frozen once and scored by the live critic every cycle after. `0` = no panel |
| `ts_struct` | run | `0` | rows per slot per cycle on which the structural label is taken under the true root and scored against the judge and the prior. `0` = off |

Not a knob but a constant on the same path: **`VO_MEMORY_MB = 16384`** (`voicing.py` ~12771) is
`voicing_run`'s container memory request, cut from the donor's inherited 32768 on the strength of
`ts_s0`'s measured peak of **7048 MB** — the longest arm this lineage runs, carrying every
instrument of `overtone` and this node's three. Modal bills the greater of the request and the
use. 16384 halves the reservation and keeps 2.3x headroom over the only measurement, which is the
margin to leave on a shared path. The peak-RSS line now prints the constant, so the log reports
the request it actually ran under.

## Gate table

| gate | where | claim | status |
|---|---|---|---|
| **TS-2** | `ts_gates_cpu` | the level is added beside the audit's fields and displaces none; `mean_p` is the mean sigmoid of the critic's logit and `mean_pmax` its max over the operative table; no draw consumed | **PASS**, `\|Δ\|` 0.000e+00 on both, draw neutral |
| **TS-3** | `ts_gates_cpu` | the panel freezes once per (slot, era); its base rate is constant while the buffer moves; a later era adds a second and keeps the first; the reading moves iff the critic does | **PASS**, base 0.1875 three times (identity) |
| **TS-4** | `ts_gates_cpu` | the structural label is invariant to the span mask and only to that span | **PASS**, 48/48 identical, 0/48 under a wrongly masked span |
| **TS-5** | `ts_gates_cpu` | the structural audit produces the repair-set membership and the verdict on identical rows, counts its grammar reads in `_EXP_REC["reads"]`, consumes no draw | **PASS**, `base_struct` identity, reads +204 ≥ 192 |
| **TS-6** | `ts_gates_cpu` | the probe tuples gain a root only when `ts_struct` is on; the probe buffer is bit-identical either way | **PASS**, widths 4 → 5, buffer identical |
| **T-2** | `reduce_rows.py` | the dumped buffer IS the buffer the last audit read (`n` and `base_rate` exact) | **PASS**, max `\|Δn\|` = 0, max `\|Δbase\|` = 1.5e-08 |
| **T-3** | `reduce_rows.py` | the reloaded heads are the run's critic, and cannot be a score identity (the audit precedes the cycle's optimizer step) | **reported, bounded**: median `\|ΔAUC\|` 0.018 filed / 0.009 probe; a re-drawn critic is 5.1× worse and at chance |
| **T-5** | `reduce_rows.py` | the dedupe equals the naive per-row path | **PASS**, `\|Δlogit\|` = 0.000e+00 |
| **T-4** | `structure.py` | the structural label is recoverable offline | **FAILS** — biased +0.064, positive on 25/30 slots, 41–80% root-ambiguous |
| **T-1** | `reduce_logged.py` | each arm's diet/grader column agrees with its own config | **PASS**, 24 arms |
| **T-6** | `refit.py` | the cached context state IS the trunk's — `ctx_state` reads `pooled.mean(1)` and the slot's own blocks and nothing else | **PASS**, max `\|Δ\|` = 0.000e+00 over 6 buffers × 1024 rows; falsified by perturbing one cached block (3.06e-04) |
| **T-7** | `refit.py` | the diets are pinned: identical rows per slot and identical filed:probe ratio inside each family, so only E[outcome] moves | **PASS** on both families |
| **T-8** | `refit.py` | every refit is read on the identical held-out rows and no diet trains on one | **PASS**, 42,722 held-out rows, leak = 0 |
| **TS-1** | `analyze_voicing.py` [R] | each paid arm is bit-identical to its seed's banked twin on every behaviour series including `t_cum` | **PASS — 0.000e+00** at BOTH seeds (`vo_s3b:voi3b_comp_pr_yk` at seed 0, `vo_s3e:voi3b_comp_pr_yk` at seed 2), commits equal |
| **TS-1b** | md5 of the banked dumps | *(unplanned, stronger)* the paid arm's `vo_rows.npz` and `vo_heads.pt` are byte-identical to `ov_s0b`'s | **PASS** — every buffer row, DP score and parameter came back bit for bit |
| **G-F** | `fidelity_smoke` | with every knob off, this file's `run_arm` reproduces the donor's | **PASS — 0.000e+00**, control 0.000e+00 |
| liveness | `reduce_rerun.py` [T] | the instruments must have done something | `exp_reads` 5,618,042 · 128 panels · 6,897 structural records · 153 cycles with a level |

## Falsification (`gates/falsify_ts.py`)

11 of 12 perturbations turn their gate red. The twelfth — the **root permuted** under TS-4 — is
blind **by design** and is named in the output: TS-4's subject is the span mask, and the root is
gate T-4's subject, which is exactly the gate that fails. Perturbations, per gate, are listed in
[`DESIGN.md`](DESIGN.md) §6.

## Runs

| tag | what | status |
|---|---|---|
| `ts_gf1` | `fidelity_smoke` — G-F with every `ts_*` knob off | **done**, 0.000e+00 on both arms with the control also 0.000e+00, app `ap-Xm7P6g2jn05xtYxBFlWUu9` |
| `ts_pf1` | `preflight --arms voi3b_pf_src,ovt_pf_comp_pr`, the three instruments ON at dust budgets | **done**, PREFLIGHT OK, app `ap-B3LDPvTXnZh7OteTjnaLwT` |
| `ts_s0` | the paid arm: `RUN_ov_s0b.sh`'s flag line verbatim plus `--ts-norm --ts-panel 128 --ts-struct 64` | **done**, 1.50 GPU-h, 5412 s, app `ap-5NFh1Z7PTbGYRocP2zHN2v`. TS-1 and TS-1b both 0.000e+00 |
| `ts_s2` | the same arm at seed 2, `--yoke-from-tag vo_s3d2`, no `--ref-tag` | **done**, 1.89 GPU-h, 6817 s, app `ap-SBZo2uSawqn3LC0RqpEnnN`. **TS-1: 0.000e+00** on all eleven series against banked `vo_s3e:voi3b_comp_pr_yk`; ladder [59, 77, 157] L[2,3,4], no L5; peak RSS **6790 MB** against the new 16384 request |

## Reproduction

```bash
cd experiments/            # MODAL_PROFILE=chromatic
# offline, no GPU
PYTHONPATH=. python3 -c "from rhm.practice.voicing import voicing as V; V.ts_gates_cpu()"
PYTHONPATH=. python3 rhm/practice/voicing/tessitura/gates/falsify_ts.py
python3 rhm/practice/voicing/tessitura/reduce_logged.py          # results/logged.txt
python3 rhm/practice/voicing/tessitura/structure.py              # results/structure.txt (T-4)
# needs the two banked dumps pulled off the volume first:
#   modal volume get rhm-scaling-data \
#     /rhm_practice_voicing/ov_s0b/ovt_comp_pr_sh/{vo_rows.npz,vo_heads.pt} \
#     rhm/practice/voicing/figures/ov_s0b/ovt_comp_pr_sh/
#   (and the same for ov_s2/ovt_comp_pr_dis)
python3 rhm/practice/voicing/tessitura/reduce_rows.py            # results/rows.txt
python3 rhm/practice/voicing/tessitura/plot_tessitura.py         # figures/*.png
# the paid arm, and its two reductions
bash rhm/practice/voicing/tessitura/results/RUN_ts_s0.sh
python3 rhm/practice/voicing/fetch_compact.py --tag ts_s0 --fetch --replace
python3 rhm/practice/voicing/analyze_voicing.py --tag ts_s0 --yoke-src vo_s3:voi3_dp \
    --bank vo_s3b:voi3b_comp_pr_yk \
    --out rhm/practice/voicing/tessitura/results/ts_s0_reduction.txt     # gate TS-1 is [R]
python3 rhm/practice/voicing/tessitura/reduce_rerun.py --tag ts_s0       # results/ts_s0_rerun.txt
```

Artifacts on `rhm-scaling-data:/rhm_practice_voicing/<tag>/`; compact mirrors under
`../figures/<tag>/`; `vo_rows.npz` and `vo_heads.pt` are gitignored.
