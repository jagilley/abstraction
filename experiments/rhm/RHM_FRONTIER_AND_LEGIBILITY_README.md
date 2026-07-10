# RHM: m (not occupancy) gates the learnable frontier, and the FM residual tracks it (2026-06-30)

**Code**: `rhm_occupancy_frontier.py` (NTP frontier vs occupancy), `rhm_fm_legibility.py` (FM residual-legibility over training), `rhm_local_signal_sweeps.py::occ_frontier_refs` (BP/greedy reference lines), `rhm_norm_trajectory.py` (weight/activation-norm probe over the legibility checkpoints), `rhm_wd_sweep.py` (Exp 3: weight-decay sweep, preemption-robust). Modal/GPU for the trained-model parts; reference lines are local CPU.
**Prior experiment**: [RHM_DEEP_COMPOSITION_README.md](RHM_DEEP_COMPOSITION_README.md) (recoverable / representable / not-locally-SSL-learnable arc that motivated this)
**Belief**: [beliefs/trees/rhm_compositional_learnability.md](../../beliefs/trees/rhm_compositional_learnability.md) (the occupancy-vs-m dissociation is persisted there)

## One-line arc

Sweeping the RHM DGP shows that **synonymic multiplicity `m`, not tuple-space occupancy `m/v^(s-1)`, gates how deep plain NTP learns the hierarchy** — m=2 groks to the root, m≥4 stalls — and lowering occupancy raises the *recoverability ceiling* and *frontier quality* but never the frontier *depth*. Then, on the m=2 substrate where NTP genuinely learns the full hierarchy, a co-trained forward model's **residual becomes hierarchy-legible (η²) exactly where and when the base model is learning each level**, with a clean bottom-up traveling front to the root; on m=4 the deep residual is structureless. This is the temporal confirmation that the A2A forward-model loop is an **amplifier of structure the base objective already extracted, not a source** — and it explains why every prior RHM A2A attempt (all at m≥4) saw a diffuse, illegible residual.

## Strategic context (why we ran these)

The deep-composition arc concluded that deep RHM structure is **recoverable** (BP ceiling), **representable** (oracle-aux reaches BP given the signal), but **not learnable by any local self-supervised objective** tried — at the regimes tested, all of which were m=4. Separately, the A2A program (see `../a2a_forward/`) showed a "wake-sleep ratchet" that *compounds* on MNIST (48% val-loss gap) but not on RHM, where the FM residual is high-rank and diffuse. The unifying hypothesis we adopted this session: **the A2A loop amplifies structure the base model already extracted; it cannot manufacture structure the base objective never reached.** That predicts (1) there should exist a regime where plain NTP *does* learn deep RHM composition, and (2) in that regime the FM residual should become legible the way it is on MNIST. Both experiments below test exactly that, and both confirm it.

## Occupancy, briefly

At each level a parent feature expands to an `s`-tuple of children; there are `v^s` possible tuples and at most `m·v` legal ones, so **occupancy = legal/possible = m·v / v^s = m / v^(s-1)** (= `m/v` at s=2). It's the fraction of tuple-space the grammar packs into. Low occupancy → legal tuples are sparse → each tuple identifies its parent uniquely → the rule inverts → high levels recoverable. High occupancy → collisions → information about the parent is destroyed. Occupancy is the control variable for the **information ceiling** (how recoverable deep features are in principle); this experiment shows it is *not* the control variable for **learnability**.

---

## Experiment 1: the NTP frontier as a function of occupancy

`rhm_occupancy_frontier.py` reuses `rhm_thread_b.thread_b` verbatim (per-level last-token linear+MLP probing of the true latent ancestor) and fans it out over 6 DGP settings at fixed architecture (8L/8H/256D, ~6.34M params), depth (L=6), branching (s=2), corpus (20M tokens), and training budget (40k steps), varying only occupancy along two spokes (vary `v` at fixed m=4; vary `m` at fixed v=16). Reference lines (BP ceiling, greedy floor; distinct rules, 3 seeds) come from `rhm_local_signal_sweeps.py::occ_frontier_refs`.

**Learned frontier (best-over-blocks MLP probe; chance = 1/v) vs BP ceiling:**

| setting | m | occ | BP root | model d3 / d4 / d5 / d6 | verdict |
|---|---|---|---|---|---|
| v16 m2 | **2** | 0.125 | 0.93 | 1.00 / 1.00 / 1.00 / **0.93 = BP** | **solves to root** |
| v16 m4 | 4 | 0.25 | 0.78 | 0.92 / 0.73 / 0.22 / 0.09 | stalls ~d4 |
| v32 m4 | 4 | 0.125 | 0.92 | 0.99 / 0.89 / 0.28 / 0.07 | stalls ~d4 |
| v64 m4 | 4 | 0.0625 | 0.96 | 1.00 / 0.93 / 0.24 / 0.05 | stalls ~d4 |
| v8 m4 | 4 | 0.50 | 0.40 | 0.81 / 0.43 / 0.23 / 0.14 | stalls ~d4 |
| v16 m8 | 8 | 0.50 | 0.27 | 0.25 / 0.10 / 0.11 / 0.07 | stalls ~d3 |

**Findings:**

1. **`m` sets the stall depth of the bottom-up learning wave**: m=2 reaches the root (d6 = 0.93 = BP exactly, all lower levels 1.00), m=4 stalls ~d4, m=8 stalls ~d3.
2. **Occupancy controls the ceiling and frontier *quality*, not frontier *depth*.** At fixed m=4, lowering occupancy 0.50→0.0625 (v8→v64) raises d4 from 0.43→0.93 and the BP root ceiling from 0.40→0.96, but leaves d5/d6 near chance (v64m4: d5 0.17, d6 0.05) **despite BP(d5)≈1.0, BP(d6)=0.96 and perfectly clean d1–d3**. Lower occupancy does not restart the wave.
3. **Collapse-check kill shot**: v16m2 and v32m4 share occupancy 0.125 and BP root ceiling ~0.92, yet learned root recovery is **0.93 vs 0.07**. Same recoverability, opposite learnability ⇒ m, not occupancy, is the learnability control.
4. **Capacity is ruled out** as the cause of the m≥4 failure by oracle-aux (deep-composition Exp 4d: same 8L/256D architecture + supplied latent signal reaches BP at every level). The m=2 *success* required the larger 8L model (the prior 4L/128D plateaued at 2–3 levels — depth was the binding constraint at m=2), but the m=2-vs-m=4 gap persists at fixed sufficient capacity, so it is not a size artifact.

This refines the earlier "occupancy law" reading (which attributed the README's "m dominates scaling 3:1" purely to occupancy/recoverability): **m carries a learnability effect over and above its recoverability effect.** Consequences: curriculum-over-occupancy (vary v) is dead as a way to deepen the frontier; **curriculum over m (m=2 → m=4) is the live constructive lever**; and **v16m2 is a ready single-task substrate where plain NTP fully encodes the hierarchy.**

**Reproduction:**
```bash
cd experiments/rhm && python3 rhm_local_signal_sweeps.py --which occ_frontier   # reference lines (CPU)
cd experiments && modal run --detach -m rhm.rhm_occupancy_frontier::occupancy_frontier
```

---

## Experiment 2: FM residual-legibility over training (m2 vs m4)

`rhm_fm_legibility.py` adapts `rhm_regime_trajectory.py`: it trains the 8L/8H/256D model on the **same distinct-rule DGP** (rule_seed=0, seq_seed=1 — identical to `thread_b`/Exp 1) with dense checkpoints to 50k steps, and at each checkpoint trains a fresh `TransformerForwardModel` and measures the residual's hierarchy-conditioning per level (η² of residual on the true latent feature, last-token), alongside cosine, effective rank, and top1-PC. The decisive contrast is **v16m2 (groks to root) vs v16m4 (stalls ~d4)**.

**Design (every choice grounded in prior FM-on-RHM work; a subagent synthesized the corpus and we verified it against code):**
- Per-block params verified = `12d²+13d` = 789,760 at d=256; whole model 6.34M.
- **Two gap sources**: `post_embed→` (whole bottom-up build inside the gap → low-level bumps visible as a positive control) and `post_block0→` (deep-focused; d1–d3 muted because already in the FM input). **Two targets**: `post_block4` and `post_block6` (margin for m2's deeper root; stays out of the output-prep blocks 6–7).
- **Matched-head FM (n_head=8)** primary, at two capacities (A = 8H/16d/mlp1 ≈ 529K ≈ 14%; B = 8H/16d/mlp2 ≈ 791K) bracketing the meaningful cosine band; plus a **fixed-param 1H twin of B (1H/128d/mlp2 = 791K)** as a head-invariance control. Matched heads matter because a head-count *mismatch* concentrates the residual (RESIDUAL_RANK: 1H-on-multi-H → rank 77–86% vs matched 90–96%), which could masquerade as legibility under η².
- **Guards**: η² gated on cosine∈[0.90,0.99]; earliest high-cosine checkpoints treated as overshoot baseline; per-level η² read against probe accuracy + BP ceiling so an absent deep bump reads as "never computed," not "FM captured it."

### Headline contrast (peak `feature_eta2_last`, in-band; E→b6 8H)

| level | **m2 (groks)** | **m4 (stalls)** |
|---|---|---|
| d4 | **0.42** | 0.035 |
| d5 | **0.40** | 0.010 |
| d6 (root) | **0.23** | 0.003 |

Robust across all three gaps (E→b6, E→b4, b0→b6): m2's deep-level residual legibility is **10–100× m4's**. m4's final checkpoint (step 50000, cos 0.896) confirms the stall is **asymptotic** — front parked at d2 (0.115)/d3 (0.090), deep levels dead.

### The traveling front (m2)

The m2 residual shows a clean **bottom-up traveling rise** in η²: d1 (learned first) rises early (~0.10 by step 1–3k) and then **falls** to ~0.06 — the predicted rise-then-resolve bump; the front then propagates d3→d4→d5→d6, reaching 0.30/0.42/0.40/0.23 by 40–50k. m4's front stalls at d2–d3 and the deep levels stay flat at zero — the legibility front stalls at the **same depth** the probe-accuracy learning front stalls (Exp 1). This is the core result: **FM-residual legibility tracks what the base model is actively learning.**

### Residual evolution over training (m2, E→b6 8H)

| step | val | cos | resNorm | rank% | top1% | d4 η² | d6 η² |
|---|---|---|---|---|---|---|---|
| 500 | 1.09 | 0.986 | 2.6 | 83.0 | 8.2 | 0.03 | 0.01 |
| 5000 | 0.85 | 0.953 | 7.8 | 84.8 | 4.3 | 0.17 | 0.06 |
| 20000 | 0.83 | 0.929 | 20.7 | 84.1 | 5.4 | 0.36 | 0.18 |
| 50000 | 0.82 | 0.918 | 37.5 | 84.4 | 6.2 | 0.42 | 0.22 |

- **Cosine declines monotonically** (0.99→0.92): cosine is **scale-invariant**, so this is genuine *directional* unpredictability — the computation getting harder to predict, not merely bigger — which (with η²) is the real signal that the model is still learning deeper computation late in the run.
- **Residual norm grows ~58×** (0.64→37.5), but this is **not** the closed-loop activation-inflation pathology (that needs injection; an open-loop run has none). Via `‖res‖ ≈ ‖target‖·√(2(1−cos))`, the cosine drop accounts for only ~3.4×; the remaining ~17× is the main model's `post_block6` **activation norm drifting up over normal NTP training** — generic, and independent of how sophisticated the computation is. So the norm is *not* a learning/structure signal; scale-invariant cosine and η² are. **(Confirmed directly, 2026-06-30 — see "Norm trajectories" below: `post_block6` last-token norm grows 5.84→103.9 = 17.8× over 0→50k, matching the inferred ~17×.)**
- **Post-(val-)grok sensitivity**: val loss plateaus by ~step 5k, but the residual keeps revealing deep learning the loss is blind to (d4 η² 0.17→0.42 while val is flat). The FM residual is a far more sensitive probe of ongoing deep representational learning than val loss.

### Head-invariance control — passes, with a measurement nuance

| | 8H (matched) | 1H (headctrl) |
|---|---|---|
| m2 d4 / d5 / d6 | 0.45 / 0.40 / 0.23 | 0.29 / 0.20 / 0.11 |
| m4 d4 / d5 / d6 | 0.034 / 0.010 / 0.003 | 0.037 / 0.010 / 0.003 |

Both head counts show the same m2≫m4 deep dissociation, so the deep legibility is **not** a head artifact. The artifact we guarded against — a 1H FM *inflating* η² by concentration — **did not occur** (1H ≤ 8H at depth), so matched-8H is both artifact-free and the more sensitive readout. The quantitative 1H/8H gap and 1H's early "peak" are a **cosine-gating effect** (1H is weaker on the hard 7-block gap, so on the fully-computing m2 model its cosine drops below 0.90 at later checkpoints and they're gated out); a matched-checkpoint recut would tighten this, but the qualitative verdict stands.

### Rank, and the head artifact (where it lives)

| | 8H (matched) | 1H (mismatch) |
|---|---|---|
| m2 rank% | ~83–86% (top1 ~5%) | ~77–80% (top1 ~6–9%) |
| m4 rank% | ~66–73% (top1 ~25%) | ~56–69% (top1 ~32–42%) |

The mismatched 1H FM **concentrates** the residual (rank 5–10 points lower, higher top1) — the RESIDUAL_RANK head-count artifact, cleanly reproduced. **So rank is the head-contaminated quantity and η² is the head-robust one** — which vindicates using η² as primary. m2 (rank ~84%) vs m4 (~66–73%) also fits "rank reflects the complexity of what the model actually computes": m2 computes the full hierarchy → the FM misses a distributed set → high rank; m4 stalls → the FM misses a few dominant modes → concentrated.

### Correction to an earlier overclaim

m2's residual is **NOT "MNIST-like low-rank."** It is high-rank (~84%), low-top1 (~5%) — the *opposite* of MNIST's low-rank (18/128 ≈ 14%), high-top1 (~16%) digit-discriminative residual. What is MNIST-like is the **η² legibility**, and it is actually *stronger* than MNIST (m2 deep η² up to 0.42 vs MNIST's 0.14–0.23 per-PC). The right statement: **m2's residual is strongly hierarchy-legible *directionally* (η²) but distributed across many dimensions, not rank-collapsed.** The hierarchy conditions the residual's *mean* heavily without collapsing its *span*.

### Knowledge-grok vs circuit-grok

How close is m2 to the information ceiling, and have we "grokked"? Two senses, and they diverge:

- **Knowledge: at the ceiling.** For the m2 DGP (occupancy 0.125) the BP ceiling is d1–d5 ≈ 1.00 and **root = 0.93**, and the occupancy sweep (Exp 1) showed m2's probe accuracy reaching exactly that. The rising deep-η² here independently confirms the legibility-run m2 actually computes the deep levels (the η² could not rise otherwise). *(The "0.8" root ceiling from earlier discussions was v16**m4**, occupancy 0.25 — a different, higher-occupancy DGP; the m2 substrate's root ceiling is 0.93.)*
- **Circuit: no collapse.** Flat-high rank (~84%), low top1-PC (~5%), drifting-up target norms, and falling cosine all say the solution is **high-complexity, high-rank, distributed** — not a compact "grokked circuit."

So m2 exhibits **knowledge-grok without circuit-grok**: it reached the information ceiling via an expensive, distributed implementation it has *not* simplified. This is precisely *why* rank/norm aren't collapsing — there has been no compression to collapse them — and it is consistent with the run using only mild weight decay (0.01) and stopping roughly when knowledge saturated. Grokking's circuit simplification is the regularizer's doing, *after* the data is fit, so a longer run and/or stronger weight decay is what would test for it. **Prediction:** if a circuit collapse occurs, **rank falls, cosine rises** (a compact circuit is FM-predictable), and the deep **η² resolves** (the "fall" we never captured, since the root only groks at the end of this run); if it does *not* occur even then, RHM's deep solution is irreducibly distributed at this scale.

### Norm trajectories: WD=0.01 is effectively *zero* regularization pressure (2026-06-30)

Direct measurement of weight norm and per-layer activation norm across the saved m2/m4 checkpoints (`rhm_norm_trajectory.py`, no retraining), motivated by the question above. Three findings, all sharpening the knowledge-grok-vs-circuit-grok reading:

| step | val (m2) | ‖W‖ (m2) | ‖W‖ (m4) | pb6 act-last (m2) | pb6 act-last (m4) | pb7 act-last (m2) |
|---|---|---|---|---|---|---|
| 0 | 2.84 | 82.9 | 82.9 | 5.8 | 5.8 | 6.5 |
| 5000 | 0.852 | 97.9 | 98.3 | 26.0 | 81.1 | 51.4 |
| 20000 | 0.829 | 125.5 | 127.7 | 60.2 | 144.6 | 155.4 |
| 50000 | 0.820 | 163.9 | 167.6 | 103.9 | 161.7 | 313.7 |

1. **At WD=0.01, weight norm does not fall — it *grows* 2× (82.9→164) and is still climbing at 50k.** This is cross-entropy's margin-maximization inflating weights, and per-step decay `lr·wd` = 3e-6 is far too weak to counteract it. So the absence of circuit-grok is **not** evidence the solution is irreducible — *we never entered a regularization-dominated regime.* The mild 0.01 was effectively zero pressure; testing for circuit collapse requires substantially stronger WD (1–2 orders of magnitude).
2. **Weight norm does not dissociate m2 from m4** (164 vs 168 at 50k) despite m2 solving to the root and m4 stalling at d4. Weight norm carries *no* functional-complexity signal here — generic Adam+CE drift. This extends the "residual norm is a non-signal" point directly to *weight* norm: rank/cosine/η² must carry the circuit-grok readout.
3. **The legibility run's inferred ~17× `post_block6` activation drift is confirmed** (5.84→103.9 = 17.8× from init; `post_block7` ~49×). The output-prep blocks (6, 7) dominate the inflation — consistent with logit-sharpening, not deeper computation. So the residual-norm growth in the legibility run was mostly activation-scale drift, not increasing structure.

Post-fit, m2 spends step 5k→50k with val improving only 0.032 nats while ‖W‖ grows 67% and pb7 activation norm grows 6× — exactly the regime where a real regularizer should compress hard without touching knowledge. This directly motivates the WD sweep (next steps #2).

**Reproduction:** `cd experiments && modal run --detach -m rhm.rhm_norm_trajectory::norm_trajectory`

**Reproduction:**
```bash
cd experiments
# Full run (trains both main models + the FM grid):
modal run --detach -m rhm.rhm_fm_legibility::run_legibility
# Phase-2-only re-run from saved checkpoints (T4, resumable, incremental part-saves):
modal run --detach -m rhm.rhm_fm_legibility::analyze_only
```

### Methods & infra notes (for future agents)

- The full `run_legibility` was launched on A10G and got **worker-preempted** repeatedly. Modal auto-restarts preempted *function calls* ("will be restarted with the same input"), so it self-healed and completed — but a couple of m4 analyze jobs errored out, dropping 2 checkpoints, and the orchestrator held all per-checkpoint results **in memory** until a single final save (a latent loss risk if the orchestrator itself is preempted).
- Fix, now baked into the file: `analyze_checkpoint` runs on **T4** (far more abundant than A10G → ~no preemption) and **commits each checkpoint's result to `/rhm_fm_legibility/parts/` before returning**; `analyze_only` **skips checkpoints whose part exists (resumable)** and aggregates from the parts on the volume, never from orchestrator memory. This is the robust pattern for any future fan-out of this shape.
- Background log-streaming clients (`modal run` and bash poll-waiters) were repeatedly reaped at turn boundaries; the **detached Modal job survives regardless**, and a persistent `Monitor` poll-loop is the reliable completion signal.

---

## Experiment 3: weight-decay sweep — does the circuit compress, or just the norm? (2026-06-30)

`rhm_wd_sweep.py` (preemption-robust: each training maintains a resumable `latest.pt` = model+optimizer+step committed every 2500 steps, so a preempted+auto-restarted call resumes from the last commit; analysis writes resumable per-(wd,checkpoint) parts and skips existing ones). Trains the **m2 substrate** (identical distinct-rule DGP, 8L/8H/256D) to **150k steps** at **WD ∈ {0.01, 0.1, 0.3, 1.0}**, measuring at each checkpoint: weight + activation norms, FM cosine / residual effective-rank / top1-PC / per-level feature η², and per-level **last-token knowledge-probe accuracy** (the BP gate). Motivated by Exp 2's norm-trajectory finding that WD=0.01 is *zero* effective pressure (weight norm grows 2×): this asks whether **real** WD pressure forces a circuit collapse (rank↓, cosine↑, deep η² resolves) or merely compresses weight norm while functional rank stays put.

### Final state (step 150k; BP root ceiling = 0.93)

| WD | ‖W‖ | val | **d6 root acc** | FM cos | **resid rank%** | top1% | d4 η² | d6 η² |
|---|---|---|---|---|---|---|---|---|
| 0.01 | 238.8 | 0.819 | 0.86 | 0.914 | 88.8 | 5.4 | 0.356 | 0.183 |
| 0.1 | 121.7 | 0.821 | **0.93** | 0.949 | 81.1 | 5.7 | 0.367 | 0.203 |
| 0.3 | 74.1 | 0.827 | **0.94** | 0.950 | 76.8 | 5.6 | 0.318 | 0.172 |
| 1.0 | 62.9 | 0.867 | 0.72 ✗ | 0.573 | 66.4 | 12.3 | 0.241 | 0.081 |

(d1–d5 ≈ 1.00 in every condition except wd=1.0's damaged root; within-run rank is converged by ~step 15k and flat thereafter, so these are asymptotic, not cut short.)

### Findings

1. **Weight norm and functional rank dissociate sharply — rank is nearly incompressible.** Across the knowledge-preserving band (WD 0.01→0.3), **‖W‖ compresses 3.2× (239→74) at zero cost** (val +0.008 nats, root knowledge *improves*), but **effective residual rank falls only 13% (89→77%)**, top1-PC stays flat ~5–6% (no dominant-mode concentration), and deep η² does not resolve. At **matched root knowledge** (wd=0.1 vs 0.3, both d6≈0.93–0.94) the cleanest cut: ‖W‖ ÷1.6 buys just 81→77% rank (5%). Weight norm is ~25× more elastic than functional rank. **This is the predicted dissociation made empirical: a simple DGP need not have a simple weight-space circuit — the deep RHM inverse is (mostly) irreducibly distributed.**
2. **No circuit-grok.** There is no phase transition — no sharp rank drop with a cosine spike and η² resolution — at any WD that preserves knowledge. The compression that does happen (rank 89→77, cosine 0.914→0.950) is smooth and modest. Weight decay is not *inert* on the circuit, but it cannot collapse RHM's deep solution the way it collapses modular addition's Fourier circuit — consistent with a structureless random-table inverse having no low-rank handle for ‖W‖²-minimization to exploit.
3. **Mild WD helps the root; the 0.01 default was leaving knowledge on the table.** wd=0.1/0.3 reach d6 = 0.93–0.94 (= BP) vs wd=0.01's 0.86 (which peaked at 0.91 mid-run and drifted down — overfitting at the no-pressure end). **wd=0.1 is the new recommended default for this substrate**: full root knowledge, best-ish val, and 2× weight-norm compression over wd=0.01.
4. **WD=1.0 confirms the mechanism by breaking it.** Pushed too hard, WD doesn't compress a *complete* circuit — it **stalls the model below the root** (d6 = 0.72 ≪ BP, val +0.05 nats). Its lower rank (66%) and rising top1 (12%) are the *never-learned* signature (a shallower function → the FM misses fewer, more concentrated modes), **not** circuit-grok — and its cosine craters to 0.57 (out of band; the FM cannot track the stunted, half-built hierarchy). Exactly the capacity-guard failure predicted: at fixed depth, enough WD eats the effective capacity the root needs.

### Reading

RHM's deep solution exhibits **knowledge-grok without circuit-grok even under strong regularization**: the model reaches the information ceiling via a high-rank distributed implementation that weight decay shrinks in *norm* (3×) but not in *functional rank* (≤13%, ≤5% at matched knowledge). This confirms the dissociation hypothesis — **low-description-length DGP ≠ low-L2 / low-rank circuit** — and isolates the missing ingredient for grokking-style collapse as the *algebraic compressibility of the target* (which RHM's random rules lack by construction), not its *description length*. The best WD achieves at preserved knowledge is **rank ≈ 77%**, the concrete floor a **structured (forward-model) regularizer** would have to beat to show "FM-driven compression succeeds where generic L2 fails."

**Reproduction:**
```bash
cd experiments
modal run --detach -m rhm.rhm_wd_sweep::wd_sweep      # train (resumable, 4 WD × 150k)
modal run --detach -m rhm.rhm_wd_sweep::analyze_wd    # per-(wd,ckpt) norms + FM + knowledge gate
```

---

## What we learned (synthesis)

1. **Learnability is m-gated, separably from recoverability (occupancy).** Plain NTP fully learns deep RHM composition at m=2 (given depth) and cannot at m≥4 — and no amount of occupancy-lowering moves that wall, even where the deep levels are ~100% recoverable. This refines the deep-composition arc's "not self-supervised-learnable at depth" to an **m≥4** statement.
2. **The FM residual is hierarchy-legible exactly where/when the base model learns the structure.** On the learnable m=2 substrate, the residual's η² rises bottom-up to the root (deep η² 0.23–0.42); on the stalled m=4 substrate, the deep residual is dead (≤0.035). This is the temporal confirmation of **"amplifier, not source,"** and the first time we see MNIST-like η² legibility on RHM — *because* it's the first time the base model has deep structure to be legible.
3. **It explains the prior RHM A2A history.** Every earlier closed-loop/ratchet attempt ran at m≥4, where the deep residual is structureless — there was nothing for the FM to amplify.
4. **Metric hygiene**: η² (hierarchy-conditioning) is head-robust; rank/top1 are head-contaminated and reflect computational complexity, not legibility per se. Legibility ≠ low-rank here.
5. **Knowledge-grok ≠ circuit-grok.** m2 reaches the information ceiling (probe ≈ BP at every level, root 0.93) yet shows no circuit-complexity collapse (rank flat-high ~84%, top1 ~5%, target norms drifting up) — it learned the hierarchy via a distributed, uncompressed solution. So the rising residual norm/η² and falling cosine reflect ongoing deep learning *plus* a generic activation-norm drift — not a grokking-style simplification, which (if it happens at all) would need more training and/or stronger regularization.
6. **The circuit is irreducibly distributed: weight norm compresses, functional rank does not (Exp 3).** A 150k WD sweep on m2 (the test point #5 called for) shows weight norm is highly elastic (‖W‖ ÷3.2 across WD 0.01→0.3 at zero knowledge cost) while effective rank is nearly incompressible (89→77%, ≤5% at matched root knowledge) — no circuit-grok. This makes empirical the "simple DGP ≠ simple circuit" claim: RHM's random-table deep inverse has no algebraic structure for ‖W‖²-minimization to collapse, unlike modular addition's Fourier circuit. The rank ≈ 77% WD floor is the bar a structured FM regularizer must beat.

## Caveats

- The m≥4 wall is confirmed **asymptotic** only at occupancy 0.25 (deep-composition Exp 4c, 100k → d5 plateaus ~0.28) and at the m=4 final 50k checkpoint here; a 100k *low*-occupancy run is still pending to confirm d5 never starts there.
- **m2 deep-level resolution (the "fall") is not captured**: the root only groks at ~40–50k = the end of this run, so deep η² is at its plateau/peak, not yet resolving. Only d1 (learned early) shows the full rise-and-fall. Extending m2 past 50k is needed to see whether the residual eventually goes structureless once everything is grokked-and-regular. Relatedly, **no circuit-complexity collapse** was observed within 50k at weight decay 0.01 (rank flat ~84%, top1 ~5%) — m2 shows knowledge-grok without circuit-grok (see "Knowledge-grok vs circuit-grok" above). **But the norm trajectories (above) show WD=0.01 is effectively zero pressure (weight norm *grows* 2×), so this is not yet a test of whether the circuit is compressible — the WD sweep (next steps #2) is.**
- 1H/8H quantitative divergence is partly a cosine-gating artifact (different in-band checkpoint subsets); a matched-checkpoint recut would tighten the head-invariance numbers.
- m2's legible residual is **high-rank**, a different regime from MNIST's low-rank residual — so the closed-loop prediction (below) is not a guaranteed transfer of the MNIST result.

## Next steps

1. **Closed-loop ratchet on the m2 substrate (the payoff test).** We finally have an RHM model with a genuinely legible, hierarchy-structured residual — the precondition the closed-loop work always lacked. Does the injection/ratchet now *compound* (as on MNIST)? Note the residual is legible-but-high-rank, unlike MNIST's low-rank case, so this is a genuine open question.
2. ~~**Extend m2 past 50k with stronger weight decay** to probe for circuit-complexity collapse.~~ *Done — Exp 3 (WD sweep to 150k, WD up to 1.0).* **Answer: no circuit-grok.** Weight norm compresses 3.2× but functional rank only ≤13% (≤5% at matched knowledge); the deep solution is irreducibly distributed at this scale. WD=1.0 doesn't collapse the circuit, it stalls the root (capacity guard). The "fall"/η²-resolution was *not* observed under any knowledge-preserving WD.
3. **FM-as-regularizer arm (the actual payoff of Exp 3).** Exp 3 established that generic L2 cannot push residual rank below ≈77% at preserved root knowledge. The north-star test: does co-training/distilling toward FM-predictability (structured pressure toward the DGP-aligned function) beat that floor — rank < 77% at d6 ≈ BP? This is the concrete form of "FM-driven compression succeeds where weight decay fails."
4. **Curriculum over m (m=2 → m=4).** Train where NTP reaches the root, transfer toward where it cannot — the one untried constructive route to deep composition at hard m.
5. *(Optional)* matched-checkpoint head-invariance recut; more rule seeds; the long low-occupancy m=4 run.
