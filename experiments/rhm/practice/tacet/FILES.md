# FILES — `tacet` (E3′: does gating what the learner *learns from* change what is learned?)

Machinery record for the node. **No results are interpreted here** — the reduction's numbers go
to the orchestrator and are discussed before any writeup (repo norm). There is deliberately no
`README.md` yet for the same reason.

**Up**: parent arc [`../README.md`](../README.md) ·
`../../../../ROADMAP.md`[^private] §7.1.3 (Track E′, first shape **E3′**) and
`../../../../QUEUE.md`[^private] ("Track E′ — E3′, the δ-gate")
**Direct donor** (untouched): [`../crescendo/`](../crescendo/FILES.md) — A3. `crescendo.py` is
forked here; `maestro/policy.py` is **imported, not forked** (this round adds no outer-loop
rule), and `crescendo/phase0_l4.py` is **imported by the reduction** (its `Miner.build` replay
is how the tables are reconstructed offline). `cr3_s0` is the cross-tag replay reference and
supplies the measured floors and the in-node displacement floors this round reads against.
**Through it**: [`../maestro/`](../maestro/FILES.md) (A2 — the thermostat and the fitted
policies) · [`../conductor/`](../conductor/FILES.md) (A1 — the yoke mechanic, the measured
floors, the observation panel) · [`../census/`](../census/README.md) (the extension op and the
r² law) · [`../assay/`](../assay/FILES.md) (the substrate).
**The node whose nulls produced this one**: [`../audiation/`](../audiation/README.md) — E1/E1b.
Findings 6–7 (the update's forecastable part *is* the analytic δ; no vector source beats the
scalars already on the wire) are why there is no forward model in this file; finding 3
(pre-update value readout 0.293, deliberation state 0.133, every update-derived source ≤ 0.05)
is what fixes the gate's two inputs.

## The question, and the one thing that varies

The learning rule in this arc has always been **grade-only selection**: mining consumes the
agent's own chosen answers on the instances it solved; π is supervised on every surviving
trajectory that solved (`train_on="solved"`). Nothing else about the datum is read. E3′ asks
whether reading two more scalars — both alive at *decision time*, both free on the wire, neither
requiring a forecaster of any kind — changes (i) what the table contains, (ii) what trust forms,
and (iii) whether the range extends.

The world does not move: same depth-6 substrate, same Phase-0-sized ladder and caps, same reads,
same floors, same outer-loop rule, same seed, and the baseline arm is A3's treatment **verbatim**
(`outer_yield_m4`). What varies is one function.

| arm | pacing | gate | role |
|---|---|---|---|
| `outer_yield_m4` | A1's thermostat on `yield` | **none** | **the baseline == grade-only selection**, the clock source every other arm replays, and the full-life cross-tag replay gate against `cr3_s0` |
| `gate_delta_hi` | clock yoke of the baseline | keep the **highest-δ** successes | "learn from the surprising" |
| `gate_delta_lo` | clock yoke | keep the **lowest-δ** successes | "learn from the expected" |
| `gate_random` | clock yoke | keep a **random** set at the matched size | **the volume control** |
| `gate_delib` | clock yoke | keep the **lowest-margin** instances | the deliberation-state gate: "learn from the contested" |
| `gate_all` | clock yoke | none; `mine_cap=0` + `prop_train_on="tips"` | **learn from everything** |
| `gate_off_y` | clock yoke | none | `preflight` only — the yoke's own inertness check |

## The gate, stated exactly

Per cycle the practice beam runs on B = `n_pr` = 64 instances × W = `pr_width` = 16 surviving
tips. Two consumption sites are gated, each at the unit it actually consumes.

**The scalars are already on the wire.** `beam_moves`/`beam_moves_prop` already compute `final` —
the value head's logit for every surviving tip — in order to pick the answer. The fork returns it
(`out["tip_val"]`, `out["best"]`): no extra forward pass, no extra grounding charged, no RNG
drawn. From it, in `gate_features`:

```
v_tip[i,w] = sigmoid(final[i,w])            d_tip[i,w] = succ[i,w] - v_tip[i,w]
v_ans[i]   = v_tip[i, best_i]               d_ans[i]   = ps[i]     - v_ans[i]
margin[i]  = v_ans[i] - mean over the UNCHOSEN tips of v_tip[i,.]
m2[i]      = sigmoid(top1) - sigmoid(top2)   (logged, NOT gated on)
```

`d` is δ = grade − v(s) at the graded unit; `margin` is the deliberation state as one scalar —
how much the *unchosen* candidates' own value scores disagreed with the chosen one, which is
literally what `audiation` finding 3 decoded. The textbook top1−top2 margin (`m2`) is logged
beside it but **not** gated on, and the reason is a substrate fact: the beam's tips are
*configurations*, and two distinct move sequences can reach the same configuration, so top1 ==
top2 exactly whenever the runner-up duplicates the answer and a top1−top2 gate would be ranking
on ties. The mean-over-unchosen form is zero only if every tip scores alike; §G prints how often
`m2` is exactly zero so the choice is auditable. Both are
materialised before the grade is applied, which is ROADMAP §4.2's surviving "timing" claim in the
scalar form §7.1.2 says the value head already satisfies.

**Mining** consumes `out["x"]` (the beam's own chosen answer per instance) restricted to solved
instances, then randomly subsamples to `mine_cap = 8`. The gate replaces the random subsample
with a δ-ranked one **at the same cap**, so mining volume is *exactly* unchanged and only the
content moves. Measured on `cr3_s0/outer_yield_m4`: ~16 instances solve per cycle and the cap
binds on **96%** of them, so this is a ~50%-selective, exactly-volume-matched content choice on
almost every cycle. `rng.permutation` is still drawn on exactly the donor's branch with exactly
the donor's argument, so the shared per-arm stream is identical and the first divergence between
a gate arm and the baseline is the gate's *content*, never its bookkeeping.

**π supervision** consumes every surviving tip that solved, all `budget` steps of it. The gate
keeps `gate_frac = 0.5` of those tips, ranked by the same scalar. The kept **count** is identical
across gate arms on every cycle (`preflight` T-3 asserts it), which is what makes `gate_random` a
matched-volume control rather than merely a random one.

**The value buffer is not gated**, in any arm. It is the source of the very forecast δ is the
residual of; gating its diet would make the gate self-referential and would confound "the gate
changed what was learned" with "the gate changed the gauge".

**No gradient-budget matching is needed**, and that is a property of the substrate rather than a
concession. `prop_train` takes a fixed `prop_steps = 24` gradient steps per cycle sampling from a
fixed-capacity (`prop_buf_cap = 60,000`) replay buffer, so the gate changes the buffer's *diet*
and never the number of updates; mining takes no gradients; the value head is ungated.
(`audiation`'s per-datum arm had to match Σlr to its anchor; here there is nothing to match.) The
one thing the gate *does* move is the buffer's **recency** — halving the inflow doubles how many
cycles 60k rows span — so `n_pairs` and `pbuf_n` are logged per cycle and printed in §G.

**Ties** are broken by index in `gate_order`, deterministically, because `delib` ties by
construction (all W tips of an instance share their instance's margin). `random` draws from the
gate's **own** numpy stream (`ggrng`), on the idiom the port's explore RNG already uses, so the
shared per-arm stream the twin gate rests on is never perturbed.

## Two design calls worth stating plainly

**Why every gate arm is a clock yoke.** The outer loop's actions are absorbing, so an arm whose
gate changes what π learns would also commit and advance at different cycles, and the era-4/5
read would conflate the gate's *content* with the gate's *pacing*. Yoked to the baseline's
realised actions (`crescendo`'s `ceiling_m3` mechanic, one level of abstraction up — there it
isolated one bit of the *commit rule*, here one knob of the *learning rule*), the arms are
lifetime-identical, era-boundary-identical and commit-cycle-identical, and differ in exactly one
knob. What the yoke costs is the pacing half of the question, and that is bought back offline at
no extra arm: reduction §H replays A1's thermostat, unchanged, on each arm's own logged L4
at-support series (`phase0_l4.py::replay_frontier_rule`'s machinery — validated in-place: on
`cr3_s0/outer_yield_m4` the replay returns c129, the arm's actual L4 commit cycle) and reports
when each arm's own gauge would have fired.

**What "learn from everything" means here.** Mining is only *defined* over solved configurations
— "a solved config is a valid r* derivation, so the target is well defined" is the substrate's
own comment — so mining an unsolved answer would feed the miner a parse of something that is not
a derivation. That is not learning from everything, it is learning from garbage. `gate_all`
therefore takes the widest **well-defined** diet on each channel: mining drops the `mine_cap`
subsample entirely (every solved answer, ~16/cycle instead of 8), and π is supervised on every
surviving tip regardless of grade (`prop_train_on="tips"` — the beam's value head selected those
tips, so they are the widest set the donor's own machinery admits). Both are *donor* knobs;
`gate_all` adds no new code, and it is the one arm nothing volume-controls, which is why it runs
last in the arm order.

## Code files

| file | purpose |
|---|---|
| `tacet.py` | The substrate. Forks `../crescendo/crescendo.py`; every addition marked `# [tacet]`. Entrypoints: `preflight`, `fidelity_smoke`, `tacet_run`. |
| `analyze_tacet.py` | The reduction. A3's, plus §G (the gate: diets, features, kept-vs-dropped, counterfactual rule overlap), §G2 (table content, reconstruction check, pairwise Jaccard between arms) and §H (the pacing counterfactual). |
| `launch_detached.py` | Session-isolated detached launcher (A3's, retargeted). |

`policy.py`, `floors.json` and `phase0_l4.py` are **deliberately absent**: the rule, the dead
zones and the offline `Miner.build` replay are A1's, A2's and A3's, reached by import. That is
the structural form of this round's claim — *the world did not change; the learning rule did.*

## What the fork adds (each `# [tacet]`-marked)

| addition | where | why |
|---|---|---|
| `out["tip_val"]`, `out["best"]` | `beam_moves`, `beam_moves_prop` | the value scores the beam already computed, returned. Zero compute, zero pricing, zero RNG |
| `gate_features` | module | the two decision-time scalars, per trajectory and per instance |
| `gate_order` | module | the candidate ordering under each mode, ties broken by index; `random` on its own stream |
| `keep=` on `prop_pairs` | `prop_pairs` | the π channel's gate. `None` is the donor exactly |
| the mining reorder | `run_arm` block (b) | the same cap, the same `rng.permutation` draw, a different ordering |
| block (a2) + `log["gate"]` | `run_arm` | the gate's decision and its inputs, per cycle, in **every** arm (so the counterfactual is computable offline for the ungated baseline too) |
| `gate_mode` / `gate_frac` / `gate_mine` / `gate_log` | `_cfg` | all four default to the donor's behaviour |
| the seven arms | `ARMS`, `TWIN` | all on the anchor's stream; the six main arms in `TACET_ARMS` |
| gate **T** | `preflight` | the yoke's inertness, the gate log's presence in every arm, volume matching across gate arms, the mining channel's volume identity, and `gate_all`'s relaxation |
| G-F retargeted | `fidelity_smoke` | this fork's direct donor is `crescendo.py`, and the gate now runs at `max_macro_level=4` (A3 had to run its own at 3 — `maestro.py` had no L4 path) |
| §G, §G2, §H | `analyze_tacet.py` | the round's own readouts |

**Name caution**: this substrate now carries three unrelated "gates" — census's L2 *admission*
gate (`gate_level`/`gate_W`/`gate_theta`), span's *parity* gate (`span_tau`, `gate_events`), and
this round's *learning* gate (`gate_mode`/`gate_frac`/`gate_mine`). Nothing in the learning gate
reads or writes a key belonging to the other two.

## Gates

| gate | what | where | status |
|---|---|---|---|
| P-1…P-12, L-1…L-8 | A1's and A2's full offline policy suite, re-run against the imported module | `maestro.policy.policy_gate()` | **ALL PASS** (21/21) |
| C-1…C-5 | A3's gate C, unchanged, re-run in this fork | `tacet.py::preflight` | **PASS** — L4 commit grows the action set 56→60; `commit_max_level` binds only on the level it names, pre-commit window **0.000e+00** |
| **T-1** | **a pure yoke with the gate off is BIT-IDENTICAL to the baseline** — what licenses reading each gate arm as "the baseline plus one knob" | `tacet.py::preflight` | **PASS — 0.000e+00** over 27 cycles / 8 series, commits equal |
| T-2 | the gate log has one row per cycle and one feature per instance, in every arm | `tacet.py::preflight` | **PASS** (27/27 rows in all six arms). Also measured here: **`m2` (top1−top2) is exactly 0 on 70–78% of instances**, which is why the gate reads the mean-over-unchosen form |
| T-3 | the gate binds, and every gate arm keeps the *same number* of solved tips on every cycle | `tacet.py::preflight` | **PASS** — refused on 27/27 eligible cycles; the four gate arms volume-matched cycle-for-cycle |
| T-4 | the mining channel's volume is identical to the baseline's before the arms diverge | `tacet.py::preflight` | **PASS** |
| T-5 | `gate_all` relaxes both channels | `tacet.py::preflight` | **PASS** — `mine_cap=0`, `prop_train_on="tips"`, 256 π rows/cycle against the baseline's 4 on a low-solve cycle |
| G-F smoke | with `gate_mode` unset and `gate_log` ON, this fork replays **`crescendo.py`** in process, at `max_macro_level=4` | `tacet.py::fidelity_smoke` | **PASS — max\|fork − crescendo\| = 0.000e+00**, donor self-replay control 0.000e+00, commits equal, on both `anchor` and `given_c1` |
| **cross-tag (full life)** | **`tc_s0/outer_yield_m4` vs `cr3_s0/outer_yield_m4` over EVERY cycle** — nothing about the world moves in this round, so the window is the whole run rather than A3's era-2 bound | `analyze_tacet.py` §0 | **PASS — 0.000e+00 over c1–c162, 13 series**, commits and loop actions identical. (The baseline ran FIRST here and SECOND in `cr3_s0`, so this also settles arm-order independence.) Plus the chain back to A2: vs `ma_s0/outer_yield` through era 2, 0.000e+00 |
| B-1 (reused) | the offline `Miner.build` replay reproduces each arm's own committed table entry-for-entry | `analyze_tacet.py` §G2 (`recon == n` column) | **PASS** — 18/18 committed tables reconstructed exactly (6 arms × L2/L3/L4) |
| panel vs G-Y | the observation panel equals the donor's G-Y miner at level 4 | `analyze_tacet.py` §0 | **PASS** — True over 162 cycles in all six arms |
| N-1, N-2, F | A1's endo label-freeness, endo price, floor gate | `tacet.py` | as A1 |

**Not a gate in this tag**: the in-tag twin gate (§0). Every non-baseline arm carries its
treatment from cycle 1 — the gate reorders the mining subsample and restricts π's diet on the
very first cycle (`prop_warmup` gates the *filter*, not the training) — so a divergence there is
the round's variable. §0 prints it as `identical`/`differs` and §2 reads the first divergence
cycle as **when the gate first bound**.

## Runs

| tag | what | outcome |
|---|---|---|
| `_preflight` | interface + gates C and T, toy sizes. First pass ran 13 arms and failed only on a *donor* line (`ok["obs_gy_agree"]`, the one place in `preflight` that assumed the full arm sweep) **after every gate had already passed** — guarded. Second pass re-ran the six gate arms after the `margin` definition was changed to the mean-over-unchosen form | `results/preflight.log`, `results/preflight2.log`, `results/preflight3.log` — **ALL PASS** |
| `tc_gf` | G-F: in-process fork-vs-`crescendo.py` replay at `max_macro_level=4` | **PASS**, 0.000e+00 vs a 0.000e+00 self-replay control; `results/gf.log` |
| `tc_smoke` | three arms end-to-end at `--quick` (baseline, one gate, the relaxed diet); mechanics only | clean; DONE in 1286 s (0.36 GPU-h) |
| `tc_s0` | the main run: 6 arms, A3's ladder/caps/floors, one seed | clean; **10,278 s = 2.85 GPU-h**, 9.8–10.0 s/cycle, all six arms 162 cycles, commits at c49/c92/c129 in every arm, 0 empty-table cancellations. Full record `figures/tc_s0/reduction.txt` (§0–§9, §A–§E, §G, §G2, §H, §I) |

Volume `rhm-scaling-data:/data/rhm_practice_tacet/<tag>/`; fetched copies, figures and
`reduction.txt` under `figures/<tag>/`.

## Reproduce

```bash
cd experiments/            # MODAL_PROFILE=chromatic

# offline, no GPU: the policy suite (P-1..P-12 + L-1..L-8)
PYTHONPATH=. python3 rhm/practice/maestro/policy.py

modal run rhm/practice/tacet/tacet.py::preflight \
    --arms "outer_yield_m4,gate_off_y,gate_delta_hi,gate_delta_lo,gate_random,gate_delib,gate_all,anchor_long,ceiling_m3,outer_yield_m4x,outer_yield_m3,census_extend,anchor"
modal run rhm/practice/tacet/tacet.py::fidelity_smoke --tag tc_gf

python3 rhm/practice/tacet/launch_detached.py --fn tacet_run --tag tc_smoke --quick \
    --quick-cycles 10 --quick-probe-clean 512 --quick-gen-steps 20 \
    --endo-price 267 --seed 0

python3 rhm/practice/tacet/launch_detached.py --fn tacet_run --tag tc_s0 \
    --eras "1:25:60,2:12:50,3:6:70,4:3:12,5:1:9" --era-caps "60,50,70,12,9" \
    --arms "outer_yield_m4,gate_delta_hi,gate_delta_lo,gate_random,gate_delib,gate_all" \
    --max-macro-level 4 --endo-price 267 --gate-frac 0.5 \
    --budget 8 --g-budget 482 --n-corrupt 1 --mine-cap 8 --gy-level 4 --entry-rec \
    --span-min-hold 128 --collect-task-matched --tm-episodes 8192 \
    --recert-every 5 --n-aud 192 --probe-every 8 --n-rt 384 --n-score 256 \
    --sil-cv 0.15 --lp-min-drop 0.10 --seed 0

python3 rhm/practice/tacet/analyze_tacet.py --tag tc_s0 --fetch --figures
```

Every flag but `--arms` and `--gate-frac` is `cr3_s0`'s, verbatim — that is what makes the
baseline arm reproduce A3's treatment bit-for-bit. The floors are the defaults (A1's measured
values) and are asserted, so no `--tol-*` literals are passed.

[^private]: Not mirrored: this link points to a document in the private lab repo (the roadmap, the queue, an unrun spec, reading notes, or a conversation). See the top-level README for what is held back and why.
