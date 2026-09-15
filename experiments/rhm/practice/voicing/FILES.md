# voicing — FILES

The machinery record: every file, gate, run, flag and diagnostic, with the run table. Decisions
live in [`DESIGN.md`](DESIGN.md); the prompt is [`SPEC.md`](SPEC.md). **Up**:
[`../FILES.md`](../FILES.md) (practice). **Parent node**:
[`../enharmonic/`](../enharmonic/FILES.md) — `voicing` forks `enharmonic.py` and imports
`quotient.py` / `merge.py` untouched. **Writeup**: [`README.md`](README.md) (2026-09-14).

Results are facts-only. Nothing in `../enharmonic/` has been edited by this node.

## Status

| | |
|---|---|
| Q0 | **done** 2026-09-12, offline, CPU, no GPU. `figures/vo_q0_reduction.txt`. |
| Q1 | fork built, gates PASS, temperature sizing in flight. Design in `DESIGN.md` §11–§16. |
| Q2 | **done** 2026-09-13. `vo_s2` reduced (`figures/vo_s2_reduction.txt`); design and outcome in `DESIGN.md` §19–§23. |
| Q3 | `vo_s3` **done** 4.60 GPU-h, reduced (`figures/vo_s3_reduction.txt`). **Defect #8: the probe channel was built, priced and never consumed** — one missing keyword at the call site, so the diet axis of the 2×2 is dead. Fixed, and gate **V-6** added (the dual of V-4/V-5). The anchor and the `{composed, filed}` cell are valid; §30–§31. |
| Q3e | **done** 2026-09-14, `vo_s3e` (the two composed arms yoked to the banked **seed-2** anchor). Y-1 exact, 0 cancelled. **The L2/L3 advantage replicates across all three seeds that produced those rungs (+18.5 / +24.5 / +19.3% for `comp_yk`); the L4/L5 frontier advantage does NOT (+16.1% at seed 0, −1.7% at seed 2).** §35's frontier headline is narrowed to seed 0 and §37's scope sentence is withdrawn. `DESIGN.md` §38. |
| Q3d | **done** 2026-09-14, `vo_s3d2` / `vo_s3d3` (the anchor alone at seeds 2 and 3, two launches). **L5 replicates at 1 of 4 seeds, L4 at 2 of 4, L3 at 3 of 4.** Seeds 0/2/3 all ran 201 cycles with every era at its cap; seed 1's early-quiet advance is a distinct failure mode at 1 of 4, not the explanation of L5's fragility. `DESIGN.md` §37; four-seed table `figures/vo_s3d_seedtable.txt`. |
| Q3c | **done** 2026-09-14, `vo_s3c` (seed 1, one tag, in-tag yoke). **Y-1 exact with 0 cancelled — the yoke is seed-robust. The ANCHOR'S LADDER is not: at seed 1 it commits L2@47 and nothing else, runs 108 cycles, and never certifies L3, so no arm has an L4/L5 cell and §35's frontier magnitudes are untested at this seed.** What is checkable replicates: the composed chooser's class-accuracy advantage at L2/L3 is +18.5% relative at seed 0 and +24.5% at seed 1, and the era-3 error gap is −0.048 / −0.052. `DESIGN.md` §36. |
| Q3b | **done** 2026-09-14, `vo_s3b` 5.1 GPU-h, reduced (`figures/vo_s3b_reduction.txt`). Y-1 exact on all four arms, so the chooser is read at L4/L5 for the first time. **The composed chooser beats the anchor at 6 of 6 frontier cells and by 0.223 in era-5 error; the replace chooser does not. The probe diet lifts the critic's counterfactual AUC from chance (0.491) to 0.798 like-for-like.** Design and outcome in `DESIGN.md` §32–§35. |
| GPU-h spent | preflight `vo1`/`q2a`–`q2f`/`q3a` + G-F `vo_gf1`–`vo_gf8` + sizing `t025`/`t080` + `vo_s1` 4.04 + `vo_s2` 4.04 |

## Code files

| file | purpose |
|---|---|
| `gates/falsify.py` | **The falsification harness** — the evidence that this node's gate table is not decorative. The round's binding rule (a new gate is not reported until it has been shown to FAIL on a deliberate perturbation of the thing it protects) plus Q3's addition (an inertness gate needs a liveness twin, and the perturbation *that* one must fail on is DISCONNECTION — V-6/V-6b's first perturbation is defect #8 itself). Standalone: `python3 rhm/practice/voicing/gates/falsify.py` from `experiments/`. **36/36.** |
| `analyze_voicing.py` | The reducer. Sections [A]–[I] are `analyze_enharmonic.py`'s that still apply plus this node's; [J] the critic's held-out AUC per slot with **FILED and PROBE rows apart** (never pooled — the probe's candidate set is a different class by construction and a pooled AUC would read that as discrimination); [K] ε's own counters and the frontier; [L] governance; [M] the corridor under a moved write, now on one denominator via the fired/closed shift split, with Q2's undecomposable form kept beside it; **[N]** the composed choice (`moved` against the DP's argmax, with `dp_vs_head` beside it); **[O]** babbling's counts, solve rate against the filed rate, and bill share per cycle and per era. Backwards-compatible: the banked `vo_s2` tag re-reduces to the numbers it did. |
| `voicing.py` | The fork of `../enharmonic/enharmonic.py` at its `en_s9` head (every addition `# [voicing]`; **G-F 0.000e+00 with the knobs off**, both arms, commits equal, donor self-replay control also 0.000e+00). Adds: the write record captured at the write on both the fired and the closed branch (`VoRecorder`, the `PerfExecutor.cred` idiom); the verdict join through the beam's own `parent` gathers (`commit_step` / `assemble`, `out["vo_wpath"]`); exploration at the write (`vo_head_scores` on-table through the head, `vo_dp_scores` on the closed slot, `vo_sample` on the per-BLOCK score); the record-verdict calibration objective (`vo_train_terms`); four instruments on the priced practice beam (`vo_instruments`); `gates_cpu` (`vo_gates_cpu`); the Q1 2×2 and its four preflight twins. |
| `launch_detached.py` | `enharmonic`'s launcher, copied and retargeted. Carries this image's one environment fact: the CLI imports `voicing.py` locally before it ships, so the **system** interpreter behind `modal` needs numpy (`python3 -m pip install --user numpy==1.26.4`) — a venv copy of the Modal client cannot reach the API through this session's egress proxy. |
| `fetch_compact.py` | `enharmonic`'s, copied and retargeted at `rhm_practice_voicing`. |
| `q0_voicing.py` | The whole of Q0, offline: five sections [V1]–[V5] plus gate VQ-1, reduced from the banked compact mirrors under `../enharmonic/figures/`. Rebuilds the operative book at any logged build EXACTLY from the `picks` / `n_rows_in_class` fields (`figured_bass`'s addition), which `alias_audit.py` could not — it re-ranks a capped class by index, and the cap binds on every L4/L5 build in this tag. Pure numpy; no torch, no substrate, no GPU. |

`voicing.py`'s Q3 additions, all `# [voicing Q3]`-marked and every knob default off:
`vo_dp_scores_from_logits` (split out so the OPEN path scores the DP from the logits the trunk
already returned — the composed chooser is free on both paths); `vo_compose` (z-scored
composition, `/span` first, a guard at |C| = 1); `VoRecorder.moved_add` and the `moved` stat;
`vo_run_probes` + `push_probe` + `pbuf` (the priced counterfactual probe and its own buffer and
RNG stream); `vo_critic_terms(use_probe=…)` and `vo_critic_audit`'s diet split; the `vo_bill`
log series; the `n_shift_fired` / `n_shift_closed` split; `vo_compose_inert_check` (V-4c) and
`vo_probe_offstream_check` (V-4d); and `vo_gate_v4_v5`, factored **out of** `preflight` so V-4's
two forms and V-5 can be falsified without paying for a remote preflight.

## Children

| folder | what | record |
|---|---|---|
| [`sotto_voce/`](sotto_voce/README.md) | who grades the babble: an outcome model trained on the learner's own experience (the *mirror*) supplies the probe channel's verdicts in place of the world, alone, as a K=5 committee, and as a hybrid paying the world where the committee disagrees; the world's verdict computed on every probe as an instrument no arm consumes. Seeds 0 and 2, yoked. `sotto_voce.py` forks `voicing.py` and imports nothing from it; this node is untouched. | [`sotto_voce/FILES.md`](sotto_voce/FILES.md), [`sotto_voce/DESIGN.md`](sotto_voce/DESIGN.md), [`sotto_voce/SPEC.md`](sotto_voce/SPEC.md) |
|  [`overtone/`](overtone/README.md) | the readout round (2026-09-14 → 15): S1 linear twins of the critic as shadows and in the seat, S2 the trunk's zero-verdict scores and the out-of-fold combination, S3 the probe budget by disagreement; `ov_s0`, `ov_s0b`, `ov_s2`; the `# [overtone]` hunks in `voicing.py` and sections [J1] [J2] [J3] [Z2] [R] of `analyze_voicing.py` are its; [`overtone/FILES.md`](overtone/FILES.md) has every knob, gate and app id | [`overtone/FILES.md`](overtone/FILES.md), [`overtone/DESIGN.md`](overtone/DESIGN.md) |

## Runs

| tag | what | status |
|---|---|---|
| — | Q0 is offline. No Modal app was created, no volume was written, no `modal run` was issued. | — |
| `vo_gf1` | **G-F**, `fidelity_smoke` retargeted at `enharmonic.py`, composed short config, every `# [voicing]` knob off | **PASS** — `max|fork − donor| = 0.000e+00` on both arms, commits equal, donor self-replay control 0.000e+00 |
| `_preflight_vo1` | the four preflight twins `voi_pf_{dp,own,xp,own_xp}`, `--vo-explore-t 0.25`, V-1 asserting on every cycle | **PASS** — gate VO-8; `verify_bad = 0` of 1,653 / 1,277 / 1,462 / 1,040 filed writes |
| `_preflight_t025`, `_preflight_t080` | the two sampled twins at T = 0.25 and T = 0.80, re-gated after the span normalisation, to read `xp_by_level` | **PASS** — `verify_bad = 0` on all four arm-runs; the per-level deviation curve is in `DESIGN.md` §17 and sets **T = 0.25** |
| `vo_s1` | Q1's four-arm 2×2 on `en_s9`'s ladder, T = 0.25 | **done**, 4.04 GPU-h, 14,547 s. Reduction `figures/vo_s1_reduction.txt`. |
| `vo_gf2`…`vo_gf7` | G-F, re-run after every change to a shared path | **PASS** each time — 0.000e+00, commits equal |
| `_preflight_q2a`…`q2f` | Q2's five twins. `q2a` crashed (`NameError`), `q2b`/`q2c`/`q2d` failed **V-4** at max\|Δ\| 9.19 / 9.00 / 2.0187 — three distinct real defects, see `DESIGN.md` §22 — `q2e` **PASS** on every gate, `q2f` **PASS** confirming the two readout fixes | see §22 |
| `vo_gf8` | G-F, re-run after the Q3 changes to shared paths | **PASS** — 0.000e+00, control 0.000e+00, commits equal |
| `_preflight_q3a` | Q3's five twins `voi3_pf_{dp,v4,v4pr,comp,comp_pr}` | **PASS on every gate, first try** — G-F, V-1, V-4A, V-4B, V-5, VO-8; table in `DESIGN.md` §29 |
| `vo_gf9` | G-F, re-run after the defect-#8 fix touched `finetune_generator_span` (a shared path) | **PASS** — 0.000e+00, control 0.000e+00, commits equal |
| `_preflight_q3b` | Q3b's six twins `voi3b_pf_{src,dp,v4,v4pr,comp,comp_pr}`, all but the source clock-yoked to it | **PASS on every gate, first try** — Y-1 on all five twins, V-1 on 3,267 writes, V-4A/B, V-5, **V-6b**; table in `DESIGN.md` §34 |
| `vo_s3e` | Q3e's frontier check on the second seed that reached it — `voi3b_comp_yk` / `voi3b_comp_pr_yk` yoked to the **banked** `vo_s3d2:voi3_dp` (`--yoke-from-tag vo_s3d2`, `--seed 2`), no anchor re-run. No code change, no G-F. | **done**, 201 cycles each. **Y-1 exact, 0 cancelled** (commits [59,77,157], advances [60,110,180,192,201], L2/L3/L4 realised). Reduction `figures/vo_s3e_reduction.txt`. See `DESIGN.md` §38 |
| `vo_s3d2`, `vo_s3d3` | Q3d's seed check on the anchor's own climb — `voi3_dp` alone at `--seed 2` and `--seed 3`, **two launches** (a per-arm `seed=` override would hold `shared["probe_clean"]` fixed at the run-level seed, which seeds 0 and 1 moved). No code change, no G-F. | **done**, 201 cycles each. seed 2 → L2@c59, L3@c77, **L4@c157**; seed 3 → L2@c52, L3@c66(4 entries). Both `0 quiet / 5 cap`. See `DESIGN.md` §37 |
| `vo_s3c` | Q3c's seed pair — `voi3_dp` self-paced at `--seed 1` then `voi3b_comp_yk` / `voi3b_comp_pr_yk` yoked to it **in-tag** (no `--yoke-from-tag`; `phase_a` resolves an in-tag source by arm order). No code changed for this round; G-F not needed and not run. | **done**, 108 cycles/arm. Reduction `figures/vo_s3c_reduction.txt`. **Y-1 exact, 0 cancelled**; the seed-1 anchor commits **L2@c47 only** (3 advances chosen / 3 capped, cert L2@c17, L3 never). See `DESIGN.md` §36 |
| `vo_s3b` | Q3b's four yoked arms `voi3b_{comp,comp_pr,rep,rep_pr}_yk`, `--yoke-from-tag vo_s3`, `w = 1`, `n_probe = 64` | **done**, 5.1 GPU-h. Reduction `figures/vo_s3b_reduction.txt`. **Y-1 exact on all four** — commits [48,100,151,186], advances [60,110,180,192,201], 0 cancelled, L5 realised on every arm, lifetimes within 0.3% of the anchor's ledger. See `DESIGN.md` §35 |
| `vo_s3` | Q3's four arms `voi3_{dp,comp,comp_pr,rep_pr}` on `en_s9`'s ladder, `w = 1`, `n_probe = 64`, no ε | **done**, 4.60 GPU-h, 16,562 s. Reduction `figures/vo_s3_reduction.txt`. Anchor exact (0.000e+00 × 13 vs `en_s9`); `voi3_comp` is the valid `{composed, filed}` cell; **both probe arms are bit-identical to their no-probe twins on every behaviour series** — defect #8, `DESIGN.md` §30 |
| `vo_s2` | Q2's four-arm 2×2, ε = 0.3 | **done**, 4.04 GPU-h, 14,558 s. Reduction `figures/vo_s2_reduction.txt`. The local launch log died at a container restart (a `ProxyConnectionError` from the streaming client, not the run); the tag was fetched from the volume. |

### `vo_s2` — the run table

| arm | cycles | commits | corridor (open max / last) | misfire raw | **misfire excess** | L5@sup |
|---|---|---|---|---|---|---|
| `voi2_dp` | 201 | L2@48, L3@100, L4@151, **L5@186** | 30 / 30 | 0.068 | 0.068 | 17 |
| `voi2_critic` | 188 | L2@48, **L3@81, L4@111** | 28 / 24 | 0.619 | **0.018** | 17 |
| `voi2_xp_f` | 175 | L2@48 | 16 / 16 | 0.106 | 0.008 | 0 |
| `voi2_critic_xp` | 117 | L2@48 | 16 / 16 | 0.784 | n/a | 0 |

**The identity is exact against BOTH anchors.** `voi2_dp` vs `en_s9:endo_ledger_open_ung5_ra` and
vs `vo_s1:voi_dp`: **0.000e+00 on all thirteen series**, commits equal, `t_cum` identical
(51,208,256). The critic, the explorer and all five instruments are inert on the arm that
consumes none of them.

**The corridor survives the treatment — this is §20's design working.** `voi2_critic`'s raw
misfire of 0.619 is the treatment, not damage: on a governed slot the head imitates the RECORD, so
it is *supposed* not to match `dp_features`. With the treatment's own re-decisions removed the
**excess is 0.018**, below the anchor's 0.068, on 28 open slots. Q1's calibration arms, for
contrast, had 0 slots open and a 0.54 misfire that no re-decision explained.

**The critic is a real predictor, not a path check.** Held-out AUC by level (mean of per-slot
medians), on 730–830 held-out rows per slot at base rates 0.13–0.24:

| arm | L2 | L3 | L4 | share of reads above 0.5 |
|---|---|---|---|---|
| `voi2_critic` | 0.677 (16 slots) | 0.697 (8) | 0.634 (4) | 0.84–0.88 |
| `voi2_critic_xp` | 0.694 (15) | — | — | 0.78 |

**It genuinely chooses**: it governed 28 slots (16 at L2 from c49, 8 at L3 from c82, 4 at L4 from
c112) and moved the write off the DP's argmax on **0.35–0.78** of calls at every one.

**And the degeneracy Q0 §5 measured is broken.** Distinct tuples written per cycle / top-1 share
at the L4 cells, era 5: anchor 1.0–2.1 / 0.845–1.000; `voi2_critic` 1.0–3.9 / 0.493–0.868.

**But class accuracy against the repair set FELL at the frontier.** `rep` at 4n0–4n3, era 4:
anchor 0.729 / 0.882 / 0.838 / 0.773; `voi2_critic` 0.315 / 0.281 / 0.308 / 0.196. The mechanical
account is that the anchor writes one row nearly always and that row repairs, while the critic
spreads across classes of which many do not; what it means is not this file's to say.

**Both ε arms stalled at L2**, and the frontier never left L2 for either (`c49–c175:L2`,
`c49–c117:L2`), so frontier-only ε never acted at L4 or L5 at all. ε's own counters are clean:
226,653 fires over 127 cycles on `voi2_xp_f` at a realised rate of **0.302** against a nominal
0.300, and **0 off-frontier cycles** on both ε arms.

**Era-4/5 error, not a claim** (self-paced, different lifetimes): anchor 0.671 / 0.773;
`voi2_critic` 0.684 / 0.779; `voi2_xp_f` 0.898 / 0.935; `voi2_critic_xp` 0.955 / 0.981.

**Bill**: `exp_reads` 0.95–2.29 M; `t_cum` 47.5 M for `voi2_critic` against the anchor's 51.2 M
(188 cycles against 201).

### The Q2 gate table of record (`q2f` + `vo_gf7`), with denominators

| check | result |
|---|---|
| G-F | 0.000e+00, both arms, commits equal, donor self-replay control 0.000e+00 |
| **V-4** | `max_abs_delta = 0.0`, `commits_equal = True` |
| **V-1** | `verify_bad = 0` of **6,822** filed writes (1,653 / 1,653 / 1,416 / 1,363 / 1,195), asserting every cycle |
| **VO-8** | passed on all five twins; `unnamed = 0` on every one |
| **ε off the frontier** | **0 cycles**, on both ε arms — against **2,162** ε fires over 30 cells on `voi2_pf_xp_f`, so the assertion is not passing on an empty denominator |
| realised ε | `n_eps / sampled` = 2162 / 7150 = **0.302** against a nominal 0.300, no temperature sized |
| critic held-out AUC | `voi2_pf_v4` 0.909 (n = 14 reads), `voi2_pf_critic` 0.910 (n = 9), `voi2_pf_critic_xp` 0.698 (n = 9) — **a path check, not a measurement**: held-out sets of 16–26 rows at base rates 0.045–0.08 |

### The Q3 gate table of record (`q3a` + `vo_gf8`), with denominators

| check | result |
|---|---|
| G-F | 0.000e+00, both arms, commits equal, donor self-replay control 0.000e+00 |
| **V-1** | `verify_bad = 0` of **7,863** filed writes (1,653 ×3 + 1,452 ×2), asserting every cycle; `unnamed = 0` on all five |
| **V-4A** | filed-write critic, governance off: `max_abs_delta = 0.0` over 27 cycles, commits equal, 0 probes billed, bill residual 0.0 |
| **V-4B** | probes ON, governance off: `max_abs_delta = 0.0` on every series *including* `g_per_solve`, commits equal, **451** probes, **bill residual 0.0** |
| **V-5** | `n_mined`, miner state, solved pool and vocabulary all bit-identical to `dp`'s; 451 rows filed into `pbuf` across 30 slots and nowhere else |
| **V-4c** | composed scorer: `w = 0` IS the DP's argmax, affine residual 3.1e-06, \|C\| = 1 passes, the critic moved 3/7 at `w = 1` |
| **V-4d** | probe: 16/16 on-table, 16 billed, 16 filed, **0** same-class draws, filed buffer untouched |
| **VO-8** | composed chooser ran on **24,765** calls; probe **451** / **342** graded = billed over 20 / 19 cycles, 30 slots each |
| `gates_cpu` | **17/17 PASS** |
| falsification | **21/21** deliberate perturbations made the gate they target fail (`DESIGN.md` §28) |
| composed choice | `moved` **0.9249** (22,904 / 24,765) at `w = 1`; **`dp_vs_head = 0`** — see `DESIGN.md` §25, a mechanism, not a defect |
| probe solve rate | **0.013** (6 / 451) against a filed solve rate of 0.048–0.069 on the same arms |
| probe bill | preflight (dust, `n_probe = 8`) 0.39% mean / 1.14% max. **Sized on the banked `vo_s2/voi2_critic` ledger at the run's `n_probe = 64`: 0.64% mean, 0.81% max, 0.46% of the whole run. The ~5% cap does not bind.** |
| critic AUC | 0.89–0.91 mean over 9–14 reads — **a path check, not a measurement** |

### `vo_s3` — the run table

| arm | cycles | commits | corridor open max/last | misfire | probes graded = billed | probe bill mean/max |
|---|---|---|---|---|---|---|
| `voi3_dp` | 201 | L2@48, L3@100, L4@151, **L5@186** | 30 / 30 | 0.068 | — | — |
| `voi3_comp` | 164 | **L2@48 only** | 16 / 16 | 0.301 | — | — |
| `voi3_comp_pr` | 164 | L2@48 only | 16 / 16 | 0.301 | 115,017 | 0.0043 / 0.0046 |
| `voi3_rep_pr` | 188 | L2@48, L3@81, L4@111 | 28 / 24 | 0.619 | 198,382 | **0.0058 / 0.0077** |

**The identity check is exact.** `voi3_dp` vs banked `en_s9:endo_ledger_open_ung5_ra`:
**0.000e+00 on all thirteen series**, commits equal. Q3's additions are inert on an arm that
consumes none of them.

**Defect #8 (`DESIGN.md` §30).** `vo_critic_terms`' `use_probe` keyword was never passed at the
call site, so the probe buffer never entered the critic's loss. `voi3_comp` vs `voi3_comp_pr` and
`voi3_rep_pr` vs banked `vo_s2:voi2_critic` are **0.000e+00 on every behaviour series** with only
`t_cum` moved, at bill residuals of 3.7e-09 and **exactly 0.0**. Every inertness gate passed,
because all of them assert with governance OFF where the diet cannot reach the run. Fixed; gate
**V-6** now holds the line and was shown to fail on the defect itself.

Selected readouts (full tables in the reduction):

- **the composed choice** — `moved` **0.330** over **3,820,426** governed calls at `w = 1` (the
  preflight's 0.925 was a path check, as flagged); `dp_vs_head` **0.084–0.160**, so §25's
  preflight reading that the two priors coincide **does not hold at scale** — see §31.3.
- **the probe channel** — a substituted class solves **3.5–4×** less often than the written one
  (0.038–0.042 against 0.137–0.179); the bill came in at 0.0058/0.0077 against a pre-launch
  prediction of 0.0064/0.0081 off the banked `vo_s2` ledger.
- **the critic off-distribution** — because the probe rows were filed but never trained on, their
  AUC is a clean held-out test of a filed-only critic: **0.40–0.78 (median ≈0.65) on FILED rows
  and 0.365–0.624 (median ≈0.49) on PROBE rows**, on 612–895 held-out rows per slot over 16 slots.
  A critic trained on what the beam chose to write does not rank counterfactuals at all.
- **V-5 at full scale, unintended** — `voi3_rep_pr` ≡ `vo_s2:voi2_critic` bit for bit on behaviour
  across two tags on **198,382** priced probes, bill residual exactly 0.0.

### `vo_s1` — the run table

| arm | cycles | commits | corridor (open max / last) | misfire | L5@sup / L6@sup |
|---|---|---|---|---|---|
| `voi_dp` | 201 | L2@48, L3@100, L4@151, **L5@186** | 30 / 30 | 0.068 | 17 / 3 |
| `voi_own_record` | 201 | L2@48 | 2 / 0 | 0.538 | 5 / 0 |
| `voi_xp` | 201 | L2@48 | 16 / 16 | 0.042 | 0 / 0 |
| `voi_own_record_xp` | 167 | L2@48, L3@93 | 2 / 0 | 0.557 | 0 / 0 |

**The in-tag identity check is exact.** `voi_dp` against banked `en_s9:endo_ledger_open_ung5_ra`:
**0.000e+00 on all thirteen series** (`e`, `succ`, `dres`, `n_solved`, `n_mined`, `n_moves`,
`width`, `e_practice`, `vloss`, `gloss`, `t_cum`, `g_per_solve`, `m_per_solve`), commits equal,
`t_cum` identical to the digit (51,208,256). That is the full-scale half of G-F, and it says the
record and all four instruments are inert on the arm that consumes none of them.

**Gate V-1 at full scale: 0 bad of 1,587,481 filed writes** (447,604 / 428,980 / 334,905 /
375,992), asserting for the first 8 cycles of each arm and counted thereafter.

Selected readouts (full tables in the reduction):

- **the sampler, realised** — `voi_xp` L2 deviation **0.091**, projection **0.004**;
  `voi_own_record_xp` L2 0.342, L3 0.771 (its head is flattened by its objective, so the same
  temperature deviates more — the compounding measured at preflight, reproduced).
- **what the chooser wrote, live, on the EXECUTED population** (`voi_dp`) — Q0 §5's table
  confirmed off the audition path: at 4n3 1.2–1.9 distinct tuples of 164–192 rows (top-1
  0.90–0.99); at 5n1 **1.00 distinct tuple, 1.00 token class, top-1 1.000 in every era**.
- **the writer's class accuracy against the repair set on the priced beam, held-out** — a number
  no banked tag carries, because `en_s5`'s instrument hooks the audition path and always meters
  the DP. `voi_dp`: 5n1 **0.717** (era 4) and **0.677** (era 5), 4n3 0.599 / 0.337, against the
  DP's audition-path 0.582 / 0.553 at 5n1 (Q0 §3).
- **the in-situ read-back** — class agreement 0.890–0.992 at L2, `unnamed` 0.000 throughout, and
  `tuple_agree < class_agree` in exactly one cell (`voi_own_record_xp` era 2 L3: 1.0000 vs 0.8533
  on 450 rows), which is Q0 §4's code-39 mechanism seen live. **L4/L5 were never sampled** — an
  instrument defect, `DESIGN.md` §18.
- **era-4/5 error, not a claim** (self-paced arms, different lifetimes; `enharmonic`'s
  matched-clock gaps between near-identical arms reach 0.08): anchor 0.671 / 0.773, the three
  treated arms 0.837–0.851 / 0.934–0.945.
- **the bill** — `exp_reads` 1.70–2.29 M against the donor's 0.38 M: the four new instruments cost
  ~6× the donor's oracle reads, contained in `_EXP_REC["reads"]`, never in `counts["ground"]`, and
  the anchor's priced `t_cum` is identical to the banked arm's.

Banked arms read (read-only, never written): `en_s9:endo_ledger_open_ung5_ra` (the anchor),
`en_s8:endo_ledger_open_ung5`, `en_s8:endo_ledger`, all from
`../enharmonic/figures/<tag>/<arm>/results.json` plus `entry_beam.json`.

## Gates

| gate | statement | where | status |
|---|---|---|---|
| **VQ-1** | The operative book rebuilt from a run's own logged `picks` has that run's own row count, at every level of every logged build. Asserted, not measured — the rebuild is deterministic. | `q0_voicing.py::gate_vq1` | **PASS** — 606 builds (`en_s9:endo_ledger_open_ung5_ra`), 628 (`en_s8:endo_ledger_open_ung5`), 591 (`en_s8:endo_ledger`) |
| G-F | `fidelity_smoke` retargeted at `enharmonic.py`: the short config at 0.000e+00 with every knob off, re-run after every change to a shared path. | `vo_gf1` | **PASS** — 0.000e+00, both arms, commits equal |
| V-1 | The record is exact: `canon[record]` equals the span of the state the beam produced, block for block. An identity, not a coverage — it is captured at the write. Asserted per cycle for `vo_verify_cycles` cycles (infinite in preflight), counted thereafter (`n_verify` / `n_verify_bad`); the softening is about not losing a 200-cycle arm to an assert at c150, and the counts are in the log either way. Driven directly on synthetic tensors in `gates_cpu` (EB-15's idiom): a clean record passes, a corrupted one raises. | `vo_gates_cpu`, `preflight` | **PASS** — 0 bad of 5,432 filed writes across four preflight arms |
| V-2a | With `vo_objective = "dp"` the span-term dispatch calls `SN.span_train_terms` / `perf_span_train_terms` — the donor's own functions, called, not a re-derivation. **Strict**, by construction and by G-F. | `voicing.py` dispatch | **PASS** |
| V-2b | The calibration term with a singleton class, `y = 1` and `push = 0` equals `F.cross_entropy` under the same teacher forcing. **Measured**, not asserted: `−logsumexp` over a one-element set against a direct sum of per-block cross-entropies is the same quantity in a different summation order. | `vo_gates_cpu` | **PASS**, \|Δ\| = 0.000e+00 |
| V-3 | The inherited offline gates still pass on the fork: E-0 (`quotient_gate`, singleton classes ARE the flat miner), E-1 (`dp_features_rec` IS `dp_features`), E-7 (the recording macro DP IS `MC.apply_any`). | `vo_gates_cpu` | **PASS** |
| VO-3 | `vo_key_of` under an unmerged map IS the flat half-pair (E-0's idiom for the new key). Asserted. | `vo_gates_cpu` | **PASS** (32 rows, 0 mismatches) |
| VO-4 | `vo_dp_scores`' argmax IS `dp_features` — so sampling at a temperature is the donor's own chooser with the argmax replaced and nothing else. Asserted. | `vo_gates_cpu` | **PASS** |
| VO-5 | The batched on-table score equals the naive per-row teacher-forced loop. **Measured**, not asserted: different summation order. | `vo_gates_cpu` | **PASS**, max\|Δ\| = 9.5e-7 over 6×64 scores |
| VO-6 | `vo_sample` at T→0 attains the maximum score, and the shared torch stream is untouched. The identity is "attains the max", NOT "is `argmax`'s index": the table ties, and `argmax` breaks a tie by index while a sampler breaks it by its draw. Asserted on the former. | `vo_gates_cpu` | **PASS** (0.000e+00; 6/64 ties broken differently) |
| VO-7 | Knobs off → the recorder is off, no temperature, no buffers. | `vo_gates_cpu` | **PASS** |
| VO-8 | In preflight: rows actually filed, `verify_bad == 0`, arms with a chooser knob re-decided the write AND left the argmax, arms without one re-decided nothing, the objective produced a per-slot readout, **ε never fired off the frontier on any cycle**, the critic governed, and its held-out AUC was computed. | `preflight` | **PASS** (`q2e`; ε assertion added after `q2e`, confirmed in `q2f`) |
| **V-4** | The critic, BUILT AND TRAINED, with its governance off and the head's target back at `dp_features`, is `voi2_pf_dp` at 0.000e+00 on every series with commits equal. Strict — which is why the critic's term is kept out of the logged `gloss`. | `preflight` | **PASS** (`q2e`), after catching three distinct defects |
| **V-4b** | The same claim on CPU in seconds, at the parameter AND logged-value level, on **on-grammar** leaves, carrying its own non-vacuity assertion (the loop ran, the span term exists, the critic term was computed). Verified to FAIL on two deliberate perturbations, one reproducing defect #5 to the digit. | `vo_gates_cpu` | **PASS** — 0.000e+00 on `d_generator`, `d_head`, `d_gloss`, `d_sloss` |
| **V-4A / V-4B** | **[Q3]** V-4 in two forms. **A**: the filed-write critic with governance off is `voi3_pf_dp` at 0.000e+00. **B**: the same with the probe ON is `dp` on every series *including* `g_per_solve` (read off the metering beam's counts, not the cycle ledger — a probe that moved it would be a probe running inside the metering beam), with `t_cum` larger by **exactly** probe count × `d_fb`. Factored out of `preflight` into `vo_gate_v4_v5(a, b, form)` so it can be falsified without paying for a remote preflight. Each form carries its own non-vacuity assertion. | `vo_gate_v4_v5`, `preflight` | **PASS** (`q3a`), 0.0 / 0.0, bill residual 0.0, 451 probes |
| **V-5** | **[Q3]** OFF-STREAM, on the objects: the probe-on / governance-off twin's `n_mined`, miner state, solved pool and committed vocabulary are `dp`'s bit for bit. Asserted separately from V-4B because "the bill moved and nothing else did" and "nothing the babbler produced reached the repertoire" are different claims. | `vo_gate_v4_v5`, `preflight` | **PASS** (`q3a`) — all four equal, 451 rows filed across 30 slots |
| **Y-1** | **[Q3b]** THE REPLAY MATCHES: a clock-yoked arm's realised commit and advance cycles equal its source's. The round's control — if it fails, every level-resolved readout is comparing two different ladders again. `cancelled` actions are EXCLUDED from the match and reported separately: a replayed commit with an empty build is the substrate answering, not the clock, and on an arm with a thinner book that is a finding. Shown to fail on a missing commit, a late commit, a missing advance, an extra advance and an empty plan. | `vo_gate_yoke`, `preflight` | **PASS** (`q3b`) — all five twins, 0 cancelled |
| **V-6b** | **[Q3b]** V-6's IN-SUBSTRATE form and the pair defect #8 would have failed: two arms with governance **ON**, one boolean of diet apart, which must NOT be identical. The SIZE of the difference is measured, never asserted; that it exists is exact, with a non-vacuity half (both governed, probe rows filed). | `vo_gate_v6b`, `preflight` | **PASS** (`q3b`) — max\|Δ\| 1.0 on `n_solved` from c9 |
| **V-6** | **[Q3]** THE DUAL OF V-4/V-5, and the gate `vo_s3` needed and did not have: the critic's DIET knob is **LIVE**. One critic, one slot, a filed buffer and a probe buffer whose verdicts **contradict** on the same contexts, one optimizer step each way — `use_probe=True` must move parameters where `use_probe=False` does not, and the training set must have grown. Every inertness gate in this table asserts with governance OFF, where the diet cannot reach the run, so all of them were blind to defect #8 by construction. Shown to fail **on the actual defect** (`d_critic = 0.000e+00`, 117 → 117 rows). | `vo_gates_cpu` | **PASS** — `d_critic` 5.31e-01, 117 → 234 rows |
| **V-4c** | **[Q3]** The composed scorer: at `w = 0` the composed argmax IS the DP's argmax; the z-scores make the composition invariant to an affine rescale of either input (without them `w = 1` means a different thing at span 2 and span 16 — Q1's span-scale defect, one organ over); a candidate set of one passes through untouched. Carries its own non-vacuity half (the critic must actually move a choice at `w = 1`). | `vo_gates_cpu` | **PASS** — affine residual 3.1e-06 |
| **V-4d** | **[Q3]** The probe path: every candidate on-table, every candidate a DIFFERENT class from the one written, the filed buffer byte-identical before and after, rows in `pbuf` and nowhere else, and the grounding count equal to the rows returned. The DIFFERENT claim is exact and not on average — the gate constructs a case in which every row writes the same class, because `vo_run_probes` draws without replacement from its own stream and the caller cannot recover the selection. | `vo_gates_cpu` | **PASS** — 16/16, 0 same-class draws |
| VO-9 | The batched critic scorer equals the per-candidate loop. Measured (broadcast vs loop). | `vo_gates_cpu` | **PASS**, max\|Δ\| 4.7e-10 |
| VO-10 | The class grouping is `vo_key_of` row by row; a class's value is the max over its spellings. Asserted on both halves. | `vo_gates_cpu` | **PASS** |
| VO-11 | ε-greedy fires at the stated rate (measured — it is a draw) and every write is on-table (asserted — that is the construction). | `vo_gates_cpu` | **PASS**, 0.3028 on 20,000 draws vs nominal 0.300 |
| VO-12 | `vo_auc` is the rank identity with ties averaged, so a constant critic scores 0.5, and one class returns `None`. | `vo_gates_cpu` | **PASS** |

Gate discipline inherited from `embouchure` DESIGN §11: assert an identity only where the
substrate is deterministic; elsewhere measure, report with the denominator, and assert coverage
only. VQ-1 asserts the row count and inherits the row *identity* from `picks` being the run's own
record of the pick — it does not assert a tensor equality against a book no log carries.

## Flags and knobs

Every one is `# [voicing]`-marked and defaults off; with all of them off the fork is
`enharmonic.py` at 0.000e+00 (G-F above). Per-ARM knobs live in `ARMS[...]["cfg"]` because an arm
IS its objective and its write rule; run-level knobs are what every arm shares.

| knob | level | default | what |
|---|---|---|---|
| `vo_record` | arm | False | capture the write record and run the record-keyed instruments |
| `vo_objective` | arm | `"dp"` | `"dp"` = the donor's self-imitation (its own function, called) · `"calib"` = the record-verdict calibration |
| `vo_explore` | arm | False | THIS arm samples its write |
| `vo_explore_T` (`--vo-explore-t`) | run | None (−1 sentinel) | the temperature every sampled arm shares, applied to the per-BLOCK score |
| `vo_push` | run | 1.0 | weight of the push-away term on unsolved calls (0 = solved-only imitation) |
| `vo_rec_cap` / `vo_rec_batch` | run | 8192 / 64 | the record buffer's cap, and rows per slot per optimizer step |
| `vo_chunk` | run | 32 | table entries scored per chunk; bounds the `(B, R, span, dim)` block |
| `vo_readback_n` / `vo_rep_n` | run | 512 / 64 | the two subsampled instruments' per-cycle budgets |
| `vo_verify_cycles` | run | 8 | cycles for which V-1 ASSERTS before becoming count-only |
| `vo_critic` / `vo_critic_govern` | arm | False | build and train the critic · let it CHOOSE on the slots it has earned |
| `vo_head_target` | arm | `"dp"` | `"dp"` = the donor's recomputed target · `"record"` = what was written, so parity is against the executor's own choice |
| `vo_eps` | arm | 0.0 | frontier-only ε-greedy (Q2; **not used in Q3**) |
| `vo_critic_lr` / `vo_critic_min` / `vo_critic_hold` / `vo_critic_trunk` | run | 1e-3 / 256 / 0.1 / False | the critic's rate, the filed rows a slot needs before it may govern, the held-out share, and whether its gradient may reach the shared trunk |
| `vo_govern_mode` | arm | `"replace"` | **[Q3]** `"replace"` = Q2 (the critic picks the class, the head the spelling) · `"composed"` = argmax of `z(dp/span) + vo_w · z(critic)` over the on-table candidates |
| `vo_w` | run | 1.0 | **[Q3]** the composed chooser's ONE knob: the weight on the critic's z-score |
| `vo_probe` | arm | False | **[Q3]** THIS arm babbles off-stream |
| `vo_probe_n` | run | 64 | **[Q3]** filed contexts probed per GOVERNED slot per cycle. Every probe grading is a grounding on the meter, with its own line (`log["vo_bill"]`) |

**Withdrawn before implementation** (`DESIGN.md` §15, §16): `own_recall` and `ear_record` (no
structural contrast on this draw), and `vo_head_aud` (redundant — on an open slot the writer is
the head and the beam's instances are fresh every cycle, so `rep` already is the head's class
accuracy against the repair set on held-out data).

## Diagnostics the Q0 reduction produces

| section | what it measures | source |
|---|---|---|
| [V1](figures/vo_q0_reduction.txt) (a)–(b) | macro rows materialised and rows captured into the self-imitation buffer, per era and per slot; the capture rate measured on slots below cap | `log["blocks"]`, `log["span"]["buf"]/["hold"]` |
| [V1] (c)–(d) | the solved-tip and solved-answer rates per era, and the row count a solved-only filter leaves per slot | `log["gate"]["n_sol_tip"]/["n_sol_inst"]`, `log["perf"]["cells"]["n_tip"]` |
| [V1] (f) | the join ladder — writes, kept writes, writes on solved tips, macro share; the head/DP execution split; the head's misfire rate against the DP it imitates | `log["n_moves"]/["width"]/["blocks"]` |
| [V1] (e) | the L5 slot's whole life: buffer, held-out, parity, open, `sacc`, cycle by cycle | `log["span"]` |
| [V2] | the expansion-choice instrument aggregated per (level, node, era) for the anchor and both banked arms | `log["exp"]` |
| [V3] | the `bottom_map` collision structure separated into reachable and unreachable by a canonical write; the record-vs-recall difference propagated to every row of every operative book, in the token-class space and in the arm's own learned one | DGP + rebuilt books + `merge.LearnedQuotient` replayed from `merge_events` |
| [V4] | the book's classes against its spellings, where `quot_spell_cap` binds, and the chooser's actual use of the book: distinct rows written, top-1 share, class-pair keys used, token classes used | `log["quot"]["build"]`, `entry_beam.json`, `log["slot"]` |
| [V5] | the table-side ceiling: token classes in the book, features coverable, and the smallest sub-book that covers them | rebuilt books + `quotient.token_class_sets` |

## Environment

A `uv` venv at `<repo>/.venv` (gitignored): Python 3.11, `numpy==1.26.4`, `scipy==1.16.3`,
`torch==2.7.0+cpu`, `modal==1.5.5`. `modal` is required only because
`rhm/rhm_sculpt_precheck.py` imports it at module scope and `quotient.py` imports `possible_sets`
from there; nothing in Q0 touches Modal. Run from `experiments/`:

```
PYTHONPATH=. <repo>/.venv/bin/python rhm/practice/voicing/q0_voicing.py
PYTHONPATH=. <repo>/.venv/bin/python rhm/practice/voicing/q0_voicing.py --tag en_s8 --arm endo_ledger
```

## Figures

| file | what |
|---|---|
| `figures/vo_q0_reduction.txt` | the Q0 reduction of record — gate VQ-1 and sections [V1]–[V5], anchor `en_s9:endo_ledger_open_ung5_ra` with `en_s8:endo_ledger_open_ung5` and `en_s8:endo_ledger` banked beside. |
