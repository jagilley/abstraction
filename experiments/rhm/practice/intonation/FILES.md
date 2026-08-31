# FILES — `intonation` (E′ round 2: a live, fallible executor, so δ_perf exists on RHM)

Machinery record for the node. **No results are interpreted here** — the reduction's numbers go
to the orchestrator and are discussed before any writeup (repo norm). There is deliberately no
`README.md` yet for the same reason.

**Up**: parent arc [`../README.md`](../README.md) ·
`../../../../ROADMAP.md`[^private] §7.1.3 (Track E′)
**Idea doc**: [`performance_error_is_the_bridge`](../../../../ideas/performance_error_is_the_bridge.md)
§1 (the signal), §13 (the two corrections: context-conditional `b(s)`, centered gate; and the
two cautions: (b) δ-over-raw hygiene, (c) the benchmark timescale's interior optimum), §14 (the
assembly, and δ-over-uniform's regime dependence) ·
[`practice_manufactures_its_own_credit`](../../../../ideas/practice_manufactures_its_own_credit.md)
§1 component (2) — practice's own metering component.
**Design constraint inherited from** [`../../../mjc/two_clocks/`](../../../mjc/two_clocks/README.md):
credit factorizes into precision × value and **sharing one channel is destructive
interference**, so δ_perf gains/gates the EXECUTION side and the grade/value channel is left
untouched (the value buffer is ungated in every arm, as in `tacet`).

**Direct donor** (untouched): [`../tacet/`](../tacet/FILES.md) — E3′. `tacet.py` is forked here;
`maestro/policy.py` is **imported, not forked** (this round adds no outer-loop rule), and
`crescendo/phase0_l4.py` is imported by the reduction.
**The executor primitive** (untouched, **imported**):
[`../native/span/span_net.py`](../native/span/FILES.md) — `SpanHead`, `trunk`, `dp_features`,
`SpanExecutor`, `parity`, `span_train_terms`. `PerfExecutor` **subclasses** `SpanExecutor`, so
gates S-1…S-6 (recorded in `../native/span/span.py::span_selfcheck`) cover this file unchanged;
gate I-0 asserts the subclassing rather than trusting it.
**Through them**: [`../crescendo/`](../crescendo/FILES.md) (A3 — the thermostat, the L4
frontier, the yoke mechanic, `commit_max_level`, the measured floors) ·
[`../maestro/`](../maestro/FILES.md) · [`../conductor/`](../conductor/FILES.md) ·
[`../census/`](../census/README.md) · [`../assay/`](../assay/FILES.md) ·
[`../spiral/`](../spiral/README.md) (the depth-6 span-head precedent: `sp_s0` ran `spiral` and
`given_native` with the corridor head live at depth 6 — finding 7 there measured 5 of 8 L3 span
slots clearing τ at parity 0.99–1.00, and `summary.json` measured the head's cost at
**+5.2 s/cycle** over routing-alone, which is what sizes this node's budget).

## The diagnosis, and the one thing that varies

**Two δs got one name.** The bridge line defines δ as a **performance error** —
`e = ‖realization − intention‖`, benchmarked against a *context-conditional* running average of
itself `b(s)`, gated on agency: `δ_perf = (b(s) − e)·σ((g − g₀)/θ)`. It reads **execution
quality independent of task success**. The E-track nodes on this stack (`audiation`, `tacet`)
used `δ = grade − v(s)`, an **outcome** surprise. `two_clocks` measured that these are the two
factors of credit — precision × value — and that routing one through the other's channel is
destructive interference.

**Why RHM has never had δ_perf**: the A-track stack is routing-only. A macro is executed by
`macros.apply_any` — an exact max-sum DP over `T[ℓ−1]` plus per-block infill. Intention and
realization *coincide*, so `e ≡ 0` identically and the quantity E′ has been hunting is zero by
construction.

**The one change**: the span head fires **below** its parity gate (`span_tau_fire = 0.50`
against `span_tau = 0.95`), so realization is fallible and `e` is live.

| quantity | how it is computed | cost |
|---|---|---|
| **intention** | `SN.dp_features(logits, move, s)` — what `apply_any` would write. The head's own firing path already computes `logits` (the donor discards them), and the DP is a pure function of them, so the **exact** reference is free. **This is why no learned forward model appears anywhere in this node.** | one argmax chain |
| **realization** | `head.emit(pooled, blk0, span, slot_id)` — what the head wrote | already paid |
| **`e`** | `1 − mean(got == tgt)` over the span's blocks — graded Hamming. The alphabet is unordered categorical, so Hamming is the metric; the exact-match **bit** the parity gate uses is `e == 0` and is logged beside it | free |
| **`b(s)`** | per-macro-**slot** EWMA of `e` at `perf_alpha`, read *before* it is updated. The slot is the syllable — Gadagkar's per-context benchmark. **Never** a global scalar (§13(b): the scalar form fixates *worse than raw error*) | free |
| **`g`** | `a_exec · a_gap`. `a_exec` = 1 iff the head's emission became the realization; `a_gap` = `mean(got != argmax(logits over the span))`, the **arity gap** — how much the slot command explains over the slot-free per-block prediction. Graded, per §13(a) | free |
| **`δ_perf`** | `(b(s) − e)·σ((g − g₀)/θ)`, the gate **centered** (σ(0) = 0.5 leaks half the channel). `g₀`/`θ` calibrated in-run from the arm's own active/passive `g` medians — `agency_gate`'s protocol, self-supervised, no test-set tuning | free |

**The playback condition exists for free.** `g == 0` identically on base moves and on
DP-executed macros: nothing the head emitted became real, so there is no efference copy for its
output and the centered gate closes **by mechanism rather than by a hand-written exclusion**.
And the head's would-be error on exactly those contexts is already computed: the parity read
runs `dp_features` and `head.emit` on every held-out row of every slot every cycle, so
`e_shadow = 1 − parity_block` is the ACT/PLAYBACK contrast at zero extra cost (reduction §P1).

## Arms

Single seed. Every arm is a **clock yoke** of `perf_log`, so all five are lifetime-,
era-boundary- and commit-cycle-identical and differ in exactly one knob.

| arm | consumes δ_perf | role |
|---|---|---|
| `perf_log` | **nothing** | the instrument arm, the **uniform-plasticity control** for (i), the **grade-only baseline** for (ii), and the **clock source** every other arm replays |
| `perf_gain` | (i) per-sample gain on the head's plasticity, `w = exp(−δ/τ_w)` | the bridge's own efferent form |
| `perf_raw` | (i) per-sample gain on **raw `e`** — no benchmark, no agency gate | §13(b)'s hygiene control |
| `perf_gate` | (ii) selection: keep highest-Σδ_perf trajectories, at `tacet`'s matched volume | E3′ asked with the *right* δ |
| `outcome_gate` | (ii) `tacet`'s `gate_delta_hi` rule (`δ = grade − v`) at the **same** volume | the exactly-volume-matched **content** control |
| `perf_off_y`, `perf_fid` | preflight only | gate I-1's inertness pair |

`perf_log` is deliberately **not** `outer_yield_m4` itself: turning the executor on changes the
trajectory, so this tag's baseline is its own object and the cross-tag replay against
`tc_s0`/`cr3_s0` is a *bounded* gate (bit-identical until the span loss's first optimizer step
after the L2 commit) rather than `tacet`'s full-life one.

## Design calls, and the reasoning

**Substrate: route (a), graft — and it turned out to be free.** The brief left open whether to
graft the span head onto the `crescendo` stack or fork the `native/span` lineage. Checked
first, per the brief's own lead: **`crescendo.py` already imports `native/span/span_net.py` and
carries the whole Port-2 path** — `SpanExecutor` threaded through both beams, `mint`/`bind_slots`,
`finetune_generator_span`, the per-cycle parity gate, `span_*` cfg knobs, and span-capable arm
specs (`span_true`, `given_native`, `spiral`, `given_fid`). Every A3/E3′ arm simply sets
`"span": False`. `spiral`'s `sp_s0` ran the head live at **depth 6**, so the graft is not even
untested at this scale. Route (b) would have cost the thermostat, the L4 frontier, the yoke, the
floors and `tacet`'s comparators to gain nothing. **Cost of the graft: one flag per arm.** All
of this node's new code is the meter and its two consumers.

**`e` is graded, not a bit.** The parity machinery computes exact-match; `b(s)` is a running
*average*, and a Bernoulli input gives it no resolution within a cycle. Per-block Hamming is the
natural ‖·‖ on the feature alphabet, and the bit is recoverable as `e == 0`, so both readouts
exist and the parity gate is untouched.

**τ lowered, not removed, and `span_min_hold` untouched.** The point is a *fallible* executor,
not a vandalised one, and the arc has both bounds measured: `native/` finding 4 has L3 heads
ending at exact-match parity **0.76–0.94** (wrong on 6–24% of calls — real error, majority still
correct), while finding 5's untrained-head control (parity ≈ 0) costs **+0.40–0.51 e**, which is
what vandalism looks like. 0.50 sits between them and is where the L3 slots that "mostly never
cleared τ" become live. The parity **record** stays at τ = 0.95 and every gate event carries
`open_tau` beside `open`, so the donor's counterfactual is readable straight off the log.

**What happens to the intention reference when the head fires**: nothing — it stays exact and
free, because `trunk` computes the block logits on the head's own path and the DP is a pure
function of them. The honest consequence is stated as pricing (a) below.

**The benchmark timescale (§13(c))** has an interior optimum nobody has measured on this
substrate, so it is **not defended, it is logged**: `perf_alpha = 0.05` per macro *call*, and
the per-cycle **per-slot sums** (`n`, `Σe`, `Σe²`, `Σg`, `Σδ`, `Σ(b−e)`, `n_exact`) plus a
bounded raw-row sample go to `log["perf"]`, from which reduction §P5 recomputes δ across a
ladder of α off the record. A free sweep instead of a choice.

**Pacing: everything yoked, and the pacing half bought back offline.** Every treatment here
changes what the head or π learns, both of which feed the outer loop's gauge, so unyoked arms
would confound content with pacing and make the era-4/5 range readout uninterpretable —
`tacet`'s reasoning, and `crescendo`'s one level down. `tacet`'s own caveat (the yoke pins arms
to the baseline's optimal schedule) is answered the way `tacet` answered it: reduction §H
replays A1's thermostat, unchanged, on each arm's own logged L4 at-support series and reports
when each arm's *own* gauge would have fired. That costs no arms.

**Per-datum credit** (`continuo`'s surviving cell: the consolidation bit is self-readable only
under per-datum credit, where each execution has one attributable cause) is **noted, not built**
this round. δ_perf here *is* per-execution and per-attributable-cause by construction — one
slot command, one realization — so the connection is structural and the machinery
(`audiation`'s) is available on this lineage if round 2 wants it. Not spent on this node.

**Arm count.** Five paid arms is the minimum that answers both consumption questions with their
own controls. `perf_raw` is the declared cut if the smoke's s/cycle overruns the budget: it is
the one comparison here that already has a 3-seed answer in another domain
(`plasticity_gain`, `bridge_assembly`), so cutting it costs a cross-domain replication rather
than the node's own question. `gate_random` (matched-volume random) is **not** carried: `tacet`
measured it on this exact stack, and in-tag `outcome_gate` is already an exactly-volume-matched
*content* control for `perf_gate`, which is the comparison the volume control exists to enable.

## Pricing, said out loud

1. **The intention reference is an instrument, not part of the agent's execution.** It consults
   the committed table on the head's firing path, so `native/` finding 5's *table-ablation*
   claim (execution needs no table) is **not made** in a δ-metering arm. The blocks it consumes
   are tallied as `blk_ref`, beside the ledger, never folded into `t` — the `blk_dp` / `blk_head`
   convention.
2. **Head misfires are not priced.** The ledger (`t = n_ground·d_fb + n_mat·c_mat`) is the arc's
   cross-tag comparable instrument; making it arm-dependent would confound *"the gate changed
   what was learned"* with *"the gate changed what things cost"*. Instead `n_misfire` and the
   counterfactual `t_misfire = n_misfire · c_mat` are logged per cycle and printed beside the
   ledger. A botched span that happens to solve therefore still costs nothing **in `t`** and is
   fully visible **in the 2×2**.
3. **The meter draws no RNG on the shared stream and charges no grounding.** It reads tensors
   the executor already produced. The executor's own dedicated numpy stream is drawn from
   exactly once per macro call, as in the donor.
4. **The benchmark is over the agent's own performances, not its instruments.** The meter is on
   only in `_ENTRY_REC["phase"] == "beam"` (the priced practice + metering beams); the head
   still *fires* in probes and auditions — behaviour must not depend on who is watching — but
   `b(s)` is not fed from a different instance distribution.

## Code files

| file | purpose |
|---|---|
| `intonation.py` | The substrate. Forks `../tacet/tacet.py`; every addition marked `# [intonation]`. New: `perf_gate_sigma` (the centered gate), `PerfMeter` (per-slot `b(s)`, the gate's self-supervised calibration, per-cycle accumulators, the bounded raw-row sample), `PerfExecutor` (a `SN.SpanExecutor` **subclass** — the fallible firing path, the intention reference, the per-row credit column kept in lockstep with the self-imitation buffer, and the misfire/`blk_ref` tallies), `perf_span_train_terms` (consumption (i)), `expand_selected_perf` (the beam's per-(row, col) execution credit), per-tip accumulators in `beam_moves`/`beam_moves_prop` (`tip_dperf`/`tip_bad`/`tip_exe`), the 2×2 block and the mined-by-cell tally in `run_arm`, `gate_order`'s `perf_hi`/`perf_lo` (consumption (ii)), `log["perf"]`, the five arms, and gate **I**. Entrypoints: `preflight`, `fidelity_smoke`, `intonation_run`. |
| `analyze_intonation.py` | The reduction. E3′'s, plus §P1 (the meter, per slot, and the ACT/PLAYBACK dissection), §P2 (the 2×2 per arm per era, and what got mined from each cell), §P3 (δ vs uniform vs raw, with the matched-budget check and the plant guard), §P4 (the gate arms at matched volume), §P5 (the benchmark-timescale recompute), §P6 (the two unpriced columns, priced), `perf_figures`. |
| `launch_detached.py` | Session-isolated detached launcher (E3′'s, retargeted). |

`policy.py`, `floors.json` and `phase0_l4.py` are **deliberately absent** — the rule, the dead
zones and the offline `Miner.build` replay are A1's, A2's and A3's, reached by import. So is
`span_net.py`: the executor is `native/span`'s, subclassed.

## What the fork adds (each `# [intonation]`-marked)

| addition | where | why |
|---|---|---|
| `span_tau_fire` | the parity-gate block (c2) | the **one** treatment: the FIRING threshold, below the parity record's `span_tau`. `None` == the donor |
| `PerfMeter`, `perf_gate_sigma` | module | `b(s)`, the centered gate and its calibration, and the per-cycle sums |
| `PerfExecutor` (subclass) | module | the fallible path + the four bridge quantities + the per-row credit column. `meter=None` ⇒ it **is** its parent |
| `perf_span_train_terms` | module | consumption (i). `perf_gain=None` calls `SN.span_train_terms` **itself**, so the ungated arm is bit-identical rather than equal-in-expectation |
| `expand_selected_perf` + `track` in both beams | module | the per-tip execution ledger the 2×2 is read off. `PN.expand_selected` is still what runs when the meter is off |
| `tip_dperf`/`tip_bad`/`tip_exe` in `gate_features` | module | the execution column, free on the wire in the same sense the value scores are |
| `perf_hi`/`perf_lo` in `GATE_MODES`/`gate_order` | module | consumption (ii); ties by index, tie count logged (`tacet`'s `m2 == 0` convention) |
| the 2×2 block + `mine_*` tally | `run_arm` (a2)/(b) | the readout, computed in **every** metered arm including the ungated baseline |
| `log["perf"]`, `calib_events`, `perf_cells` | `run_arm` | the record |
| the perf cfg knobs | `_cfg` | all default OFF/None, so an unnamed config is `tacet.py` |
| the five arms + `perf_off_y`/`perf_fid` + `TWIN` | `ARMS`, `TWIN` | all on the anchor's stream |
| gate **I** | `preflight` | below |
| G-F retargeted | `fidelity_smoke` | this fork's direct donor is `tacet.py` |
| §P1–§P6, `perf_figures` | `analyze_intonation.py` | the round's own readouts |

**Name caution** (extending `tacet`'s): this substrate now carries *four* unrelated "gates" —
census's L2 **admission** gate (`gate_level`/`gate_W`/`gate_theta`), span's **parity** gate
(`span_tau`, and now `span_tau_fire`, `gate_events`), `tacet`'s **learning** gate
(`gate_mode`/`gate_frac`/`gate_mine`), and this round's **agency** gate (`perf_g0`/`perf_theta`,
which is a σ, not a latch). Nothing in any of them reads a key belonging to another.

## Gates

| gate | what it asserts | where |
|---|---|---|
| P-1…P-12, L-1…L-8 | A1's/A2's full offline policy suite against the **imported** module | `maestro.policy.policy_gate()` |
| C-1…C-5 | A3's gate C, unchanged, re-run in this fork | `preflight` |
| T-1…T-5 | E3′'s gate T, unchanged, re-run in this fork | `preflight` |
| **I-0** | **the executor primitive is `native/span/span_net.py`'s, imported and subclassed** — never re-implemented, so S-1…S-6 still cover this file | `preflight` |
| **I-1** | **metering alone moves nothing**: `perf_fid` (meter ON, firing left at `span_tau`) is bit-identical to `perf_off_y` (meter OFF, same threshold). This is what separates the *treatment* (the head fires below parity) from the *instrument* (the meter runs), and is why the meter may be unpriced | `preflight` |
| **I-2** | the firing threshold binds, and only it: every treated arm carries `tau_fire = 0.50`, `perf_meter` on, one meter row per cycle, and `opened_below_tau` counts openings the τ = 0.95 gate would have refused | `preflight` |
| **I-3** | **the 2×2 is NON-DEGENERATE** — below-parity firing actually produces executed-**not**-as-intended rows. Asserts the head fired, that it was sometimes wrong (`e` not identically 0), and that **both** rows of the 2×2 are populated. *The gate the brief names.* Asserted only where firing happened at preflight sizes; the smoke is where it must bite | `preflight` |
| **I-4** | the gain arms actually weight: a weight statistic exists, credit is attached to a real fraction of rows, and the realised mean weight sits near 1 (the matched-average-budget normalisation working) | `preflight` |
| **I-5** | `perf_gate` and `outcome_gate` keep the **same count** cycle-for-cycle — what makes the latter a matched-volume *content* control | `preflight` |
| G-F | with every `# [intonation]` knob off, this fork replays **`tacet.py`** in process at 0.000e+00 against a 0.000e+00 donor self-replay control | `fidelity_smoke` |
| cross-tag (bounded) | `perf_log` vs `tc_s0/outer_yield_m4` up to the first span optimizer step | `analyze_intonation.py` §0 |
| twin | every arm vs the anchor's stream up to its own first action | `analyze_intonation.py` §0 |
| N-1, N-2, F, B-1, panel-vs-G-Y | A1's/A3's, unchanged | as donors |

## Volume layout

`rhm-scaling-data:/data/rhm_practice_intonation/<tag>/{setup.json,summary.json,<arm>/results.json,done.txt}`.
Fetched copies, figures and `reduction.txt` under `figures/<tag>/`.

## Runs

| tag | what | outcome |
|---|---|---|
| `_preflight` | interface + gates C, T and **I**, toy sizes, nine arms (`perf_given` added because a loop arm cannot commit on a forty-step substrate, so no slot would ever be minted and the firing path would go untested) | **ALL PASS**; three donor lines that assumed the full arm sweep guarded (`anchor_long`, `anchor`, and gate C's block). `results/preflight3.log` |
| `in_gf` | G-F: in-process fork-vs-`tacet.py` replay at `max_macro_level=4` | **PASS — 0.000e+00** vs a 0.000e+00 donor self-replay control, commits equal, both donor arms. `results/gf.log` |
| `in_smoke` | three arms end-to-end at `--quick`; mechanics, the 2×2's non-degeneracy, and the s/cycle measurement | clean, 1709 s = 0.47 GPU-h. Misfire rate 0.04–0.20 at forced-open parity; all four 2×2 cells live |
| `in_s0` | the main run: 5 arms, A3's ladder/caps/floors, one seed | clean, **10,829 s = 3.01 GPU-h**, 15.3–15.8 s/cycle, all five arms 131 cycles, commits at c49/c72/**c104** and advances at 58/88/114/122/131 in **every** arm. Full record `figures/in_s0_reduction.txt` (§0–§8, §A/§C/§E, §G/§G2/§H, **§P1–§P6**) |

Two facts from `in_s0` that belong in the machinery record rather than in a results discussion,
because they are properties of the instrument:

- **The agency gate's calibration came out DEGENERATE** in all five arms (`degenerate: true`,
  c56–57): the ACT-class median of `g` is exactly **0**, because on ~60% of executed rows the
  head's emission equals the slot-free per-block argmax — the macro command explains nothing
  the base infill would not have written. The midpoint protocol therefore had no gap, and the
  fallback (`g0 = m_a/2 = 0`, `theta = 1e-3`) makes the gate an approximate `indicator(g > 0)`
  with a 0.5 value exactly at `g = 0`. **The playback exclusion does not depend on this**: a
  row the head did not realize is never scored at all, so "no efference copy → no credit" is
  structural here, not thresholded. A future round wanting a *graded* gate on this substrate
  needs a calibration that does not assume a non-degenerate median.
- **The raw-`e` gain arm is not budget-matched**, and that is a property of `w_raw = e` on a
  substrate where 94% of executions are exact: the normaliser divides by `EWMA(e) ≈ 0.03`, the
  6% non-exact rows saturate `w_clip = 4`, and the realised mean weight lands at **0.4015**
  against the δ arm's **0.9997**. `perf_raw` therefore trained its head at ~40% of nominal
  plasticity. This is `two_clocks`' clip-truncation caveat in a severe form and it confounds
  the §13(b) hygiene comparison; a re-run should renormalise `raw` on a non-degenerate
  transform of `e` (or raise `w_clip`).

## Round A — the strongly-metered re-run (`ma_probe`, `ma_s0`)

**The question** (Jasper, 2026-08-30): does delta_perf-style selection/weighting help
"particularly under strongly metered data"? `in_s0`'s diet was never the binding constraint —
14.9 solved instances/cycle against a mining cap of 8 (bound on **95.4%** of cycles) and a pi
buffer full at cycle 26 of 131 — so any halving was pure loss and selection had no leverage.

**The knob, and why it is that one.** `n_pr` 64 -> 24 and `pr_width` 16 -> 8: *the meter can
afford 24 graded performances a cycle instead of 64, and a width-8 search instead of width-16*.
Both are purchases. The grammar, the damage ladder, the action set, the plan budget and every
floor are untouched, so the world's truth does not move. **Sized offline from `in_s0`'s own
per-cycle record** — a `--quick` smoke cannot size this regime (`in_smoke` solves 0.16
instances/cycle, so every regime looks scarce there) — and then **verified at real scale**
before the main run by `ma_probe`: the baseline arm alone, era 1 only, 0.27 GPU-h.

| | predicted | `ma_probe` | `ma_s0` (5 arms) | `in_s0` |
|---|---|---|---|---|
| solved instances/cycle | 5.60 | 5.83 | **4.28–4.65** | 14.9 |
| vs mining cap 8 | 0.70x | 0.73x | **0.54–0.58x** | 1.87x |
| cap binds | — | 15.4% | **4.6–5.9%** | 95.4% |
| pi rows/cycle | ~390 | 400 | **149–322** | 2063 |
| pi buffer at end | never fills | 34.7% @c52 | **38–82%** | 100% @c26 |
| observation stream | ~5.3 | 5.52 | **4.17–4.54** | 7.91 |

The prediction was an **upper** bound by construction (it held `in_s0`'s solve fraction fixed,
and a scarcer learner solves a smaller fraction), and the run came in below it, as it should.
`ma_probe` also settled the one risk offline sizing could not: a 1.5x thinner observation
stream still lets L2 commit inside era 1 (c43, certificate c23).

**`mine_cap` was deliberately NOT scaled down with `n_pr`.** It is the arc's own constant
(`tall`'s coverage law is stated at it), it protects the observation stream, and the fact that
it **stops binding** is the regime. The consequence is stated rather than engineered around:
with solves below the cap the mining subsample is never taken, so **the mining channel is
un-gateable in this regime and the gates act on pi alone** — the per-arm mining differences in
reduction §P2 are trajectory divergence, not selection.

**Two fixes carried by this round**, both diagnosed from `in_s0` and both riding inside it
rather than costing standalone runs:

- **The gate's coverage** (`perf_mean`). `in_s0` measured Sigma-delta_perf **exactly zero on
  57.0%** of the gate's candidates — trajectories the head never executed on — so more than
  half the ranking fell to index order. `perf_mean` is two-key: an executed trajectory always
  outranks a non-executed one (*no performance evidence is not the same as average
  performance*), and within the executed set the key is delta_perf **per execution**, not
  summed (the sum conflates how well a trial was played with how much of it was). Measured in
  `ma_s0`: **45 ties among 2977 executed candidates = 1.5%**, against 48.7% on the old summed
  key in the same arm — the fix was both necessary and sufficient. Both counts print.
- **The hygiene control's budget** (`rawx`). `in_s0`'s `raw` arm used `plasticity_gain`'s
  literal `w_raw = e`, right for a continuous MuJoCo residual and wrong here: 94% of executions
  are exact, so `EWMA(e) ~ 0.03`, the 6% non-exact rows saturate `w_clip`, and the realised
  mean weight came out **0.4015** against the delta arm's 0.9997 — the control trained its head
  at ~40% of nominal plasticity and the S13(b) comparison was confounded by budget rather than
  by hygiene. `rawx` uses the delta arm's own transform (`exp(+e/tau_w)`), same cap, same
  normaliser, differing in exactly the two things S13(b) is about: no benchmark, no agency
  gate. Measured mean weight **0.9990**. The `in_s0` caveat is retired.

`tau_w` is calibrated, not guessed: against Kim, Parvin & Ivry 2019's ~35% attenuation on
success, using `in_smoke`'s measured scale (b(s) ~ 0.1, so a correct row earns delta ~ +0.1 and
at `tau_w = 0.20` takes weight exp(-0.5) = 0.61, a 39% attenuation; a missed row saturates the
cap either way, which is the categorical half of their result).

| tag | what | outcome |
|---|---|---|
| `ma_probe` | Phase A: `mperf_log` alone, real configuration, era 1 only — the regime criterion and the commit-path risk | **981 s = 0.27 GPU-h**; 52 cycles at 6.8 s/cycle; L2 commit c43, 13 entries, recall 0.643 |
| `ma_s0` | the metered main run: 5 arms (`mperf_log, mperf_gain, mperf_gate, mout_gate, mperf_rawx`), A3's ladder/caps/floors, one seed | clean, **12,184 s = 3.38 GPU-h**, 14.9–15.1 s/cycle, all five arms **153** cycles, commits at c43/c83/c115 and advances at 53/94/132/144/153 in every arm. Record `figures/ma_s0_reduction.txt` |

Preflight coverage for round A's own code needed two more true-table arms for the reason
`perf_given` existed in round 2 — a loop arm cannot arm on a forty-step substrate, so neither
`perf_mean` nor `rawx` would ever execute: `perf_given_x` and `perf_given_pm`. Gate I-4 was
extended to the new gain mode and I-5 to print `n_exec_cand` / `p_tie_exec`.

## Reproduce

```bash
cd experiments/            # MODAL_PROFILE=chromatic

# offline, no GPU: the policy suite (P-1..P-12 + L-1..L-8)
PYTHONPATH=. python3 rhm/practice/maestro/policy.py

modal run rhm/practice/intonation/intonation.py::preflight \
    --arms "perf_log,perf_off_y,perf_fid,perf_gain,perf_gate,outcome_gate,perf_raw"
modal run rhm/practice/intonation/intonation.py::fidelity_smoke --tag in_gf

python3 rhm/practice/intonation/launch_detached.py --fn intonation_run --tag in_smoke --quick \
    --quick-cycles 10 --quick-probe-clean 512 --quick-gen-steps 20 \
    --arms "perf_log,perf_gain,perf_gate" --endo-price 267 --seed 0

python3 rhm/practice/intonation/launch_detached.py --fn intonation_run --tag in_s0 \
    --eras "1:25:60,2:12:50,3:6:70,4:3:12,5:1:9" --era-caps "60,50,70,12,9" \
    --arms "perf_log,perf_gain,perf_gate,outcome_gate,perf_raw" \
    --max-macro-level 4 --endo-price 267 --gate-frac 0.5 \
    --span-tau-fire 0.50 --perf-alpha 0.05 --perf-tau-w 0.20 \
    --budget 8 --g-budget 482 --n-corrupt 1 --mine-cap 8 --gy-level 4 --entry-rec \
    --span-min-hold 128 --collect-task-matched --tm-episodes 8192 \
    --recert-every 5 --n-aud 192 --probe-every 8 --n-rt 384 --n-score 256 \
    --sil-cv 0.15 --lp-min-drop 0.10 --seed 0

python3 rhm/practice/intonation/analyze_intonation.py --tag in_s0 --fetch --figures

# --- round A: the strongly-metered regime -------------------------------------------------
modal run rhm/practice/intonation/intonation.py::preflight \
    --arms "perf_given,perf_given_g,perf_given_x,perf_given_pm,mperf_log,mperf_gain,mperf_gate,mout_gate,mperf_rawx"

# Phase A: the regime criterion and the commit-path risk, at REAL scale (a --quick smoke
# cannot size this — it solves 0.16 instances/cycle, so every regime looks scarce there)
python3 rhm/practice/intonation/launch_detached.py --fn intonation_run --tag ma_probe \
    --eras "1:25:60" --era-caps "60" --arms "mperf_log" --n-pr 24 --pr-width 8 \
    <every other flag exactly as in_s0's below>

python3 rhm/practice/intonation/launch_detached.py --fn intonation_run --tag ma_s0 \
    --eras "1:25:60,2:12:50,3:6:70,4:3:12,5:1:9" --era-caps "60,50,70,12,9" \
    --arms "mperf_log,mperf_gain,mperf_gate,mout_gate,mperf_rawx" \
    --n-pr 24 --pr-width 8 \
    --max-macro-level 4 --endo-price 267 --gate-frac 0.5 \
    --span-tau-fire 0.50 --perf-alpha 0.05 --perf-tau-w 0.20 \
    --budget 8 --g-budget 482 --n-corrupt 1 --mine-cap 8 --gy-level 4 --entry-rec \
    --span-min-hold 128 --collect-task-matched --tm-episodes 8192 \
    --recert-every 5 --n-aud 192 --probe-every 8 --n-rt 384 --n-score 256 \
    --sil-cv 0.15 --lp-min-drop 0.10 --seed 0

python3 rhm/practice/intonation/analyze_intonation.py --tag ma_s0 --gf-tag in_gf --fetch --figures
```

Every flag but `--arms` and the four `--span-tau-fire`/`--perf-*` values is `tc_s0`'s
(hence `cr3_s0`'s), verbatim. The floors are the defaults (A1's measured values) and are
asserted, so no `--tol-*` literals are passed.

[^private]: Not mirrored: this link points to a document in the private lab repo (the roadmap, the queue, an unrun spec, reading notes, or a conversation). See the top-level README for what is held back and why.
