# FM-as-regularizer: structured "be-legible-to-a-compressed-self" pressure beats weight decay's functional-complexity floor (2026-07-01)

**Code**: `rhm_fm_regularizer.py` (`train_fm_reg` = co-trained-FM regularizer training; `analyze_reg` / `analyze_reg_ckpt` = knowledge-gate + FM legibility + FM-free activation rank; `wd_activation_rank` = the λ=0 activation-rank anchor on the WD-sweep checkpoints). Modal/GPU (L4 train, T4 analyze).
**Prior experiment**: [RHM_FRONTIER_AND_LEGIBILITY_README.md](RHM_FRONTIER_AND_LEGIBILITY_README.md) — Exp 3 (weight-decay sweep) established the floor this experiment beats; next-steps #3 ("FM-as-regularizer arm") is exactly this test.
**Related**: [ideas/activation_to_activation_forward.md](../../ideas/activation_to_activation_forward.md), [beliefs/trees/rhm_compositional_learnability.md](../../beliefs/trees/rhm_compositional_learnability.md), [PER_LEVEL_LOSS_README.md](PER_LEVEL_LOSS_README.md) (the FM-captures-91-97%-of-DGP-rules result that motivated it).

## One-line arc

On the m2 substrate where plain NTP fully learns the hierarchy, co-training a forward model and adding an **open-loop** `λ·MSE(post_block6, FM(post_embed))` term to the base loss — gradient through the main model only, **no injection / no distillation / no self-knowledge apparatus** — **compresses the post_block6 activation manifold ~10 points below anything weight decay can reach (42.9% vs WD's 52.6% floor), at preserved root knowledge (d6=0.94) and with deep hierarchy-legibility (η²) *enhanced* rather than eroded.** This is the concrete realization of "L2-norm complexity ≠ functional complexity": generic ‖W‖-minimization hits a functional-complexity floor (partly by flattening deep structure), while a structured pressure toward the DGP-aligned function pushes functional complexity below that floor while *keeping* the structure. It also answers the load-bearing question the session opened with: **self-knowledge is not required for functional simplification — the simplest open-loop pressure suffices.**

## Strategic context (why we ran this)

Exp 3 (WD sweep, RHM_FRONTIER) showed weight decay on the m2 substrate is a **norm compressor, not a rank compressor**: across the knowledge-preserving band it shrinks ‖W‖ 3.2× at zero knowledge cost but moves effective residual rank only ≤13% (≤5% at matched root knowledge), floor ≈ 77%. The reading was that RHM's random-table deep inverse has no algebraic handle for ‖W‖²-minimization to collapse. But PER_LEVEL_LOSS had already shown a **187K-param FM captures 91–97% of the DGP composition rules** — a constructive witness that a *low-functional-complexity* approximation of the gap computation exists. So the WD floor is plausibly an **L2-pressure floor, not an information floor**, and the north-star test (next-steps #3) is whether a *structured* regularizer — pressure toward FM-predictability — beats it.

This experiment also isolates two mechanisms the MNIST closed-loop bundles: **(a) structured regularization pressure** (object-level: "be the simple/DGP-aligned function") vs **(b) self-knowledge** (the injection/distillation loop that compounds on MNIST). We test (a) alone.

## Design

- **Substrate**: m2 (v16/m2), 8L/8H/256D (~6.34M), IDENTICAL distinct-rule DGP (rule_seed=0, seq_seed=1) to Exp 2/3 and thread_b. **wd fixed at 0.1** (Exp 3's recommended default) so each run is a drop-in delta over the WD-sweep's wd=0.1 row (= the **λ=0 baseline**); no separate control run.
- **Regularizer**: co-trained matched-head FM (8H/16d/mlp2 ≈ 791K, "capB"), src=`post_embed` → tgt=`post_block6` (the gap covering the whole bottom-up hierarchy build, blocks 0–5, out of output-prep blocks 6–7). Two separate optimizers. Per step: (1) main update `NTP + λ·MSE(FM(source), target)` with **FM params frozen** so gradient flows only into the main model; (2) FM update on detached activations so it tracks the model. λ **warmup**: 2000 steps at λ=0 (FM tracks; cold-start guard against regularizing toward a random FM), then linear ramp over 3000.
- **λ sweep** {0.03, 0.1, 0.3, 1.0, 3.0}. {0.03, 0.1} to 150k; {0.3, 1.0, 3.0} to 300k (λ=0.3 resumed from its 150k checkpoint to test "run longer").
- **Metrics (per checkpoint)**, all knowledge-gated:
  - **Knowledge gate** (the denominator): per-level last-token BP probe accuracy, best-over-blocks. rank/η² only count at matched d6 ≈ 0.93.
  - **reg-gap residual rank** (fresh FM, seed 911 ≠ training seed 42) on the regularized gap E→b6 — partly leakage.
  - **un-regularized-gap residual rank** (b0→b4, never pressured) — leakage check.
  - **FM-free activation effective-rank** of `post_block6` last-token activations (no FM in the measurement) — the leakage-proof complexity readout.
  - Per-level feature η² (residual, last-token; DGP-grounded), FM cosine, weight/activation norms.
- **Anchor** (`wd_activation_rank`): the same FM-free `post_block6` activation rank computed on the WD-sweep final checkpoints (λ=0 baselines at wd ∈ {0.01,0.1,0.3,1.0}), same eval seqs / same estimator → directly comparable.

## Results

### Final states (all wd=0.1; FM-reg rows vs the WD-sweep λ=0 baselines)

| condition | step | val | ‖W‖ | d6 acc | reg-gap rank% | unreg rank% | **FM-free act rank%** | act top1% | d4 η² | d5 η² | d6 η² | FM cos |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| WD wd=0.01 (λ=0) | 150k | 0.819 | 238.8 | 0.86 | 88.8 | — | 65.9 | 9.1 | 0.356 | — | 0.183 | 0.914 |
| **WD wd=0.1 (λ=0)** | 150k | 0.821 | 121.7 | 0.93 | 81.1 | — | **54.3** | 18.2 | 0.367 | — | 0.203 | 0.949 |
| **WD wd=0.3 (floor)** | 150k | 0.827 | 74.1 | 0.94 | 76.8 | — | **52.6** | 17.2 | 0.318 | — | 0.172 | 0.950 |
| WD wd=1.0 (broken) | 150k | 0.867 | 62.9 | 0.72 | 66.4 | — | 54.7 | 14.1 | 0.241 | — | 0.081 | 0.573 |
| FM-reg λ=0.03 | 150k | 0.823 | 116.6 | 0.94 | 83.1 | 87.9 | 55.1 | 14.2 | 0.385 | 0.334 | 0.203 | 0.967 |
| FM-reg λ=0.1 | 150k | 0.822 | 114.4 | 0.95 | 82.2 | 87.1 | 53.2 | 17.5 | 0.373 | 0.349 | 0.222 | 0.969 |
| FM-reg λ=0.3 | 300k | 0.821 | 110.1 | 0.95 | 79.6 | 87.5 | 47.8 | 17.8 | 0.392 | 0.362 | 0.206 | 0.970 |
| **FM-reg λ=1.0** | 300k | 0.821 | 107.5 | 0.95 | 78.0 | 87.5 | 45.7 | 20.6 | **0.463** | **0.388** | **0.224** | 0.974 |
| **FM-reg λ=3.0** | 300k | 0.821 | 106.9 | 0.94 | **77.2** | 87.7 | **42.9** | 27.5 | 0.411 | 0.367 | 0.206 | 0.973 |

(WD-sweep residual-rank/η²/‖W‖ from RHM_FRONTIER Exp 3; WD act-rank from the `wd_activation_rank` anchor. wd=0.01/0.1 anchors read at their final saved ckpt step 125k — val is flat by then. reg-gap rank uses the in-band-cosine fresh FM.)

### Findings

1. **FM pressure beats weight decay's functional-complexity floor (the headline).** On the leakage-proof FM-free activation rank, **weight decay cannot get post_block6 below ≈52.6% at any setting — including wd=1.0 (54.7%) that destroys the root.** FM-reg drives it to **42.9% (λ=3.0) at preserved knowledge (d6=0.94)** — ~10 points below WD's floor, ~21% below the matched wd=0.1 baseline (54.3%). Sanity check: FM-reg λ=0.03 (near-zero pressure) = 55.1 ≈ the wd=0.1 baseline, so the compression is genuinely λ-driven.
2. **It's functional compression, not degenerate collapse.** Three guards pass simultaneously: (i) **knowledge preserved** (d6 0.94–0.95 throughout), (ii) **deep η² *rises*** as rank falls (λ=1.0: d4 η² 0.463, +26% over the wd=0.1 baseline), and (iii) the metric is **FM-free**. WD reaches its 52.6% floor *with η² erosion* (d4 0.318, d6 0.172 < baseline); FM-reg reaches 42.9% *with η² enhancement*.
3. **Precise on which metric wins.** FM-free activation rank: FM-reg **decisively beats** WD (42.9 vs 52.6). reg-gap residual rank: FM-reg only **ties** the WD floor (77.2 vs 76.8) — but at preserved η² and lower functional-manifold rank. Deep η²: FM-reg **dominates**, peaking at λ=1.0.
4. **A clean legibility optimum at λ=1.0.** λ=1.0 maximizes deep η² (d4/d5/d6 all peak) and dominates the wd=0.1 baseline on *every* axis (rank, η², weight norm) at zero val cost. λ=3.0 compresses rank slightly more but starts **concentrating** (act top1 18→27.5%) — the early edge of the collapse regime; λ=1.0 is the healthy strong operating point.
5. **Benign and dial-up-able.** Zero NTP/knowledge cost through λ=3.0 (val 0.821, d6 0.94–0.95). "Be legible to a compressed version of yourself" behaves as a general, harmless inductive bias — 100× λ range with no lobotomy.
6. **Weight-norm turnover (grow-then-compress).** All three long arms *peak* mid-training (~110–112) then *decline* (to 107–110), while the wd=0.1 baseline kept climbing to 121.7. The predicted non-monotonic dynamic appears — modest (peak → −3–4), not a dramatic grokking collapse, but the direction reverses (WD alone never does).
7. **Answers the load-bearing question.** The open-loop term — no injection, no distillation, **no self-knowledge apparatus** — is sufficient to beat weight decay on functional compression while enhancing DGP structure. **Self-knowledge is not load-bearing for functional simplification.** (It may still be needed for *capability compounding* à la the MNIST ratchet — a separable claim this experiment does not touch.)

### Trajectory notes

- **FM-free activation rank falls over training under pressure** (λ=3.0: 63.5% peak @4k → 46.2 @50k → 42.9 @300k, dipping to 40.7 @250k) — still trending down at 300k, i.e. not obviously converged.
- **FM cosine separates monotonically with λ only late** (at 40k all ≈0.97; by 300k 0.970/0.974/0.973 → with λ, and the co-trained-FM cosine during training reaches 0.979/0.983/0.985). An early "cosine saturates ~0.97 regardless of λ" read (from the 40k health check) was premature.

## Follow-ups (2026-07-01): the λ ceiling, and global-vs-localized

### λ ceiling — no cliff, a graceful saturation-then-turnover (λ ∈ {5, 10} to 300k)

| λ | val | ‖W‖ | d6 acc | reg-gap rank% | FM-free act rank% | act top1% | d4 η² | d6 η² |
|---|---|---|---|---|---|---|---|---|
| 1.0 | 0.821 | 107.5 | 0.95 | 78.0 | 45.7 | 20.6 | **0.463** | **0.224** |
| 3.0 | 0.821 | 106.9 | 0.94 | 77.2 | **42.9** | 27.5 | 0.411 | 0.206 |
| 5.0 | 0.821 | 106.7 | 0.94 | 75.6 | 42.6 | 26.5 | 0.407 | 0.206 |
| 10.0 | 0.822 | 107.0 | 0.94 | 75.1 | 43.3 | 21.9 | 0.365 | 0.185 |

Knowledge gate holds at **every** λ up to 10 (d1–d5 = 1.00, d6 = 0.94). Findings: (1) **No collapse cliff** — even λ=10 keeps full knowledge and flat val across a 300× λ range; the bias resists NTP/knowledge damage remarkably far. (2) **Compression saturates at λ≈3–5**: FM-free activation rank floors at ~42–43% (λ=10 does not go lower), revealing an **irreducible activation dimensionality ~42–43% (~110/256)** at preserved knowledge — a real floor, but well below WD's 52.6% and only FM-reg reaches it. reg-gap residual rank keeps creeping to 75.1% (λ=10), now just under WD's 76.8% residual floor. (3) **λ=10 is over-driven**: deep η² *peaks at λ=1.0* and erodes past it — at λ=10, d6 η² = 0.185, now *below* the wd=0.1 baseline (0.203), i.e. FM-reg begins to mimic WD's η² erosion when pushed too hard. **Usable band λ ≈ 1–5: λ=1 = legibility optimum (max η²), λ=3–5 = compression optimum (min act rank at preserved η²).**

### Global vs localized — a single top-gap pressure compresses the whole stack

FM-free activation effective-rank *by layer* (final ckpts): multi-gap (co-train one FM per target on post_embed→{b2,b4,b6}, λ=1.0) vs the single-gap λ=1.0 run (→b6 only) vs the wd=0.1 baseline. Code: `train_fm_reg_multi`, `multi_act_rank_ckpt`, `analyze_multi`.

| source | d6 acc | b2 rank% | b4 rank% | b6 rank% |
|---|---|---|---|---|
| wd=0.1 (λ=0) | 0.94 | 34.9 | 46.2 | 54.3 |
| single-gap λ=1.0 (→b6 only) | 0.95 | 28.6 | 38.4 | **45.7** |
| multi-gap λ=1.0 (→b2,b4,b6) | 0.94 | 28.4 | **34.9** | 47.0 |

**The compression is largely global already under single-gap.** Pressuring only post_block6 to be predictable-from-post_embed compresses **b4 (46.2→38.4) and b2 (34.9→28.6)** too — layers never directly targeted — because a small FM can only predict b6-from-embed if the intermediate computation is itself routed through a lower-dim manifold. **Multi-gap** adds a modest, targeted gain at the directly-pressured middle layer (b4 38.4→34.9), ties at b2, is slightly *worse* at b6 (pressure budget spreads), and flattens the profile — with the lowest ‖W‖ (104) and val (0.8208) of any run. So spreading pressure is marginally more efficient/uniform, but the single top-gap does most of the global work. This **supersedes the earlier "localized" caveat**, which was an artifact of using the b0→b4 *residual* rank (stayed ~87%) rather than the activation-manifold rank; residual-sub-gap rank ≠ activation dimensionality, and the latter is the meaningful "how simple is the representation" readout.

### WD-pulse (2026-07-02): norm-shaving, not circuit-grok — and the ~43% floor is irreducible

Take the converged λ=1.0 model (‖W‖ 107.5, act rank 45.7%, d6 0.95) and ramp weight decay up late, FM-reg (λ=1.0) still on, plus a λ=0 control (pure WD from the same start). Code: `wd_pulse`, `wd_pulse_sweep`. Final states after 50k pulse steps:

| arm | ‖W‖ | act rank% | d6 acc | val | verdict |
|---|---|---|---|---|---|
| start (λ=1.0 converged) | 107.5 | 45.7 | 0.95 | 0.821 | — |
| λ=1, wd=0.3 | 78.1 | 46.0 | 0.95 | 0.827 | norm ↓27%, rank flat, knowledge held |
| λ=1, wd=1.0 | 62.6 | 48.2 | 0.84 | 0.858 | norm ↓43%, rank flat, knowledge eroding |
| λ=1, wd=3.0 | 53.2 | 40.6 | 0.50 ✗ | 1.008 | lobotomy (never-computed, not grok) |
| control λ=0, wd=1.0 | 62.6 | 49.8 | 0.85 | 0.856 | ≈ identical to λ=1/wd=1.0 |

Findings: (1) **No circuit-grok at any WD** — no ‖W‖ cliff + act-rank drop at preserved knowledge; wd=0.3 shaves 27% of weight norm with functional rank *pinned* ~46% and knowledge fully held (norm/rank decoupled, dynamically confirming Exp 3). (2) **The ~43–46% act-rank floor is irreducible**: WD cannot push the already-compressed model lower — the only "drop" (wd=3.0 → 34–40%) co-occurs with d6 cratering to 0.50 (broken model, matches Exp 3's wd=1.0-from-scratch). ~43% (~110/256 dims) is the intrinsic functional dimensionality of the m2 hierarchy at this scale. (3) **Tightest form of the dissociation**: FM-reg *reaches* the 43% floor (λ=3); WD cannot reach it at preserved knowledge (shaves norm to 62 or breaks, rank ≥46). (4) **FM-reg is inert during the pulse** — λ=1 ≈ λ=0 control (both ‖W‖ 62.6, d6 0.84/0.85), because at convergence reg-MSE ≈ 0 so its gradient vanishes and the pulse is WD-dominated. (Pulsing wd=1.0 onto the solved model holds d6=0.84 vs 0.72 from scratch — starting solved buys some robustness, still not full root knowledge.)

## Caveats

1. ~~**Localized, not global.**~~ *Resolved (follow-up above):* single-gap compression *is* largely global on activation rank (b2/b4 compress below baseline though only b6 was pressured). The earlier "localized" reading was an artifact of the b0→b4 *residual* rank metric; the activation-manifold rank compresses across the stack.
2. ~~**top1-PC rising / ceiling unbounded.**~~ *Resolved (follow-up above):* the λ ceiling is a graceful saturation-then-turnover, not a collapse — top1 actually de-concentrates by λ=10 (21.9), no knowledge cliff to λ=10; usable band λ≈1–5.
3. **Single rule seed** (rule_seed=0) for everything here — reproduction across 2–3 seeds wanted before the headline is hardened. (Deferred to paper-time.)
4. **Weight-norm turnover is modest** — a gentle grow-then-compress, not a sharp circuit-grok. Whether a late WD-pulse triggers a real collapse is untested.
5. **Anchor step mismatch**: wd=0.01/0.1 activation-rank anchors read at step 125k (their final saved ckpt), others at 150k; val is flat by 125k so this is immaterial.

## Reproduction

```bash
cd experiments
# Train the lambda sweep (co-trained-FM regularizer; resumable latest.pt; L4):
modal run --detach -m rhm.rhm_fm_regularizer::fm_reg_sweep --lams "0.03,0.1,0.3" --n-steps 150000
modal run --detach -m rhm.rhm_fm_regularizer::fm_reg_sweep --lams "0.3,1.0,3.0" --n-steps 300000
# ...plus the ceiling arms (#2):
modal run --detach -m rhm.rhm_fm_regularizer::fm_reg_sweep --lams "5.0,10.0" --n-steps 300000
# Analyze (knowledge gate + FM legibility + FM-free activation rank; T4; resumable parts):
modal run --detach -m rhm.rhm_fm_regularizer::analyze_reg --lams "0.03,0.1,0.3,1.0,3.0,5.0,10.0"
# The lambda=0 activation-rank anchor on the WD-sweep checkpoints:
modal run --detach -m rhm.rhm_fm_regularizer::wd_activation_rank
# Multi-gap regularizer (#3) + global-vs-localized multi-layer activation rank:
modal run --detach -m rhm.rhm_fm_regularizer::train_fm_reg_multi --lam 1.0 --n-steps 300000
modal run --detach -m rhm.rhm_fm_regularizer::analyze_multi --lams "1.0" --compare-single-lam "1.0"
```

Results on the `rhm-scaling-data` volume: `/rhm_fm_regularizer/parts/lam*_step*.json`, `/rhm_fm_regularizer/summary.json`, `/rhm_fm_regularizer/wd_activation_rank.json`, `/rhm_fm_regularizer/multi_parts/*`, `/rhm_fm_regularizer/multi_summary.json`.

### Methods & infra notes (for future agents)

- **Gradient flow**: the regularizer freezes FM params (`requires_grad_(False)`) during the main update so `MSE(FM(source), target)` backprops only into the main model; the FM then updates separately on detached activations. Two optimizers, one main forward/step.
- **`analyze_reg` aggregation bug (fixed)**: the orchestrator originally aggregated `summary.json` *before* `volume.reload()`, so worker-committed parts on other containers weren't visible and the summary was written stale (missing the new λ arms). Fixed by reloading before aggregation. The **parts on the volume were always correct** — when in doubt, aggregate locally from `/parts/` rather than trusting a summary written by a non-reloaded orchestrator.
- **Completion signals**: `modal run --detach` clients and background bash poll-waiters are reaped at turn boundaries; the detached Modal app survives regardless. Use a persistent `Monitor` polling app state (or a sentinel file on the volume) as the reliable completion signal.

## Next steps

1. ~~**Bound the top end (λ ceiling).**~~ *Done (follow-up above):* usable band λ≈1–5; no collapse to λ=10; act-rank floor ~42–43%; η² erodes past λ=1.
2. ~~**Global vs localized.**~~ *Done (follow-up above):* single top-gap already compresses the whole stack; multi-gap marginally flattens/improves.
3. **Reproducibility: 2–3 rule seeds at λ=1.0** — deferred to paper-time (per collaborator call).
4. ~~**WD-pulse (grokking test).**~~ *Done (follow-up above):* norm-shaving, not circuit-grok; the ~43% act-rank floor is irreducible to WD; FM-reg inert once converged. Confirms weight-norm/functional-rank decoupling.
5. **Transfer / curriculum (recommended next):** does FM-reg's DGP-aligned compression *transfer* better than WD/vanilla? Cleanest test: pretrain m=2 (FM-reg vs wd=0.1 vs wd=0.3) → fine-tune m=4, measure frontier depth (thread_b per-level probe). Converts "better regularizer" into "structured self-model pressure generalizes where L2 can't."
6. **Self-knowledge, separately**: this settles that self-knowledge isn't needed for *functional simplification*; the closed-loop-on-m2 question (does injection/distillation *compound capability* now that the residual is legible) remains open.
