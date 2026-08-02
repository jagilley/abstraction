# Complexodynamics on RHM: the First Law of learning, with an attractor floor (2026-07-07)

> ### ✅ Re-measured on the replacement instrument, 2026-08-02 — the central claim holds
>
> Part 1's arc was measured with FM-free activation effective rank, naive residual rank, and top1-PC, and the [residual-decomposition audit](residual_decomposition/README.md) (2026-07-27) subsequently retired that single-number rank instrument. [`residual_decomposition/trajectory/`](residual_decomposition/trajectory/README.md) re-cut both arms of this experiment by refitting fresh FMs on these same checkpoints and scoring them with `frontier_mass` / `R_res_participation` / β, over 2 prediction gaps × 4 FM capacities (112 fits).
>
> **The pressure-vs-control dissociation reproduces in 7/7 measurements**: on `frontier_mass` the `fmreg` arm peaks (step 4k–12k, matching the peak reported here) and declines, while `wd:0.1` is monotone-up and never falls. So Amendment 2 — *the descent requires an annealer; SGD alone arrests, a glass rather than an equilibrated liquid* — is not an artifact of the retired scalar. The re-cut also adds two directions robust across every capacity and both gaps: **β rises** and **`R_res_participation` falls** monotonically over training.
>
> Three things to carry forward when reading the tables below. (1) **Only shapes transfer, not levels** — the re-cut's `R_act` uses a different convention from the `act_rank` reported here. (2) **The descent's magnitude scales with the observer**: a lower-capacity FM registers less of the fall (at the wide gap the smallest legs flatten to a plateau), which sharpens "the floor is bound-relative" from a framing into something measurable, but the capacity trend is a trend and not yet a mechanism. (3) **β cannot carry absolute claims here** — it fails its own capacity-invariance check at every unsaturated checkpoint in both gaps, so the re-cut licenses trajectory *directions* and *arm contrasts*, not scalars. The `fmreg:3.0` arm (this README's largest descent, 63.5 → 42.9) has **not** been re-cut.

**Code**: `rhm_ensemble_trajectory.py` (new: fresh-FM-ensemble invariance over training checkpoints) + a re-reading of the per-checkpoint parts already saved by `rhm_fm_regularizer.py` and `rhm_wd_sweep.py`. No new training runs — the trajectory evidence comes from checkpoints/parts that already existed; the ensemble run is analysis-only on those checkpoints.
**Origin**: reading/complexodynamics.pdf[^private] — Scott Aaronson, "The First Law of Complexodynamics" (Shtetl-Optimized, 2011). Conjecture: "complextropy" (resource-bounded sophistication) is small at a system's start, large at intermediate times, small at equilibrium, while entropy climbs monotonically. Follow-up he ran later: the Aaronson–Carroll–Ouellette "coffee automaton" paper (~2014), which plotted a gzip-of-coarse-graining proxy — the epistemic status our metrics share.
**Prior experiments (the data sources)**: [RHM_FM_REGULARIZER_README.md](RHM_FM_REGULARIZER_README.md) (λ arms to 300k, per-checkpoint parts), [RHM_FRONTIER_AND_LEGIBILITY_README.md](RHM_FRONTIER_AND_LEGIBILITY_README.md) (Exp 2: the traveling η² front; Exp 3: WD sweep), [REGIME_TRANSITION_README.md](REGIME_TRANSITION_README.md) (top1-PC hump on m4), [RHM_LATENT_LOOP_README.md](RHM_LATENT_LOOP_README.md) (Exp 4: the fresh-FM ensemble instrument; §1: idiosyncratic-vs-DGP-aligned residual typology).
**Related**: [../a2a_forward/README.md](../a2a_forward/README.md) and the paper (papers/forward_self_models_paper1.md[^private]) — the FM as a capacity-bounded observer; the capacity sweep as an empirical Kolmogorov structure function.

## One-line arc

Read as a complexodynamics experiment, the saved m2 trajectories show Aaronson's rise-and-fall with two amendments — the curve descends only under structured compression pressure (SGD alone arrests mid-descent, a glass not a liquid), and it descends not to zero but to the **DGP's own sophistication relative to the observer's bound** — and the new ensemble-over-checkpoints run confirms the decomposition behind both amendments: the transient hump is **FM-idiosyncratic scaffolding** (seeded observers disagree about it, maximally at the sophistication peak), while the persistent floor is **FM-invariant and deep-DGP-aligned** (all observers miss it identically, and training + pressure purify it).

## The mapping (why an FM residual is a complextropy proxy)

Aaronson's complextropy is a two-part code: the size of the smallest *efficient* program that samples a set S in which x is generic (program bits = structure), with efficiency bounds on both the sampler and any reconstructor (so the remainder must look random to *every* bounded observer). Our metric suite gives each ingredient an operational counterpart — no single scalar is "the" complextropy; the suite covers its parts:

| Definition ingredient | Operational counterpart |
|---|---|
| bounded sampler program | the FM (capacity-bounded, trained on frozen activations; learns the *program*, not the manifold — [SYNTHETIC_INPUT](../a2a_forward/SYNTHETIC_INPUT_README.md)) |
| minimization over program sizes | FM capacity sweep; the saturation knee estimates the sophistication (a2a paper §3.2) |
| "remainder generic to any efficient observer" | fresh-FM **ensemble invariance** (this experiment; instrument from RHM_LATENT_LOOP Exp 4) |
| remainder structure at fixed program size | per-level residual η², residual rank/top1 (one point on the Kolmogorov structure function) |
| program-size-in-use of the running computation | FM-free activation effective rank (crude, linear-spectral — ACO-gzip-grade) |
| "still a valid description" constraint | knowledge gate (per-level probe accuracy) — a falling complexity metric only counts at preserved knowledge |

Everything is *conditional on representation* (activations in → activations out), so what's measured is the sophistication of the **computation**, a dynamical generalization of a quantity defined for static strings. Two important non-identities: the metrics are SGD-trained upper-bound proxies, not the definitional minimization; and the floor is bound-relative by construction (that's a feature of the definition — "the" complextropy is a curve over bounds, and only the structure-function knee is bound-independent).

## Part 1 — the curve was already in the saved parts

All from existing per-checkpoint analysis parts (fresh capB FM per checkpoint, knowledge-gated; see "Data & schemas" below). Substrate: m2 (v16/m2, 8L/8H/256D, distinct rules, rule_seed=0), root groks ~40–50k.

**The scalar rise-then-fall, conditional on pressure** (FM-free post_block6 activation effective-rank %, from `rhm_fm_regularizer` parts):

| arm | init | peak (step) | final (step) | shape |
|---|---|---|---|---|
| FM-reg λ=3.0 | 36.0 | 63.5 (4k) | **42.9** (300k) | full rise-then-fall |
| FM-reg λ=1.0 | 36.0 | 62.4 (4k) | 45.7 (300k) | full rise-then-fall |
| FM-reg λ=0.03 (≈no pressure) | 36.0 | 59.0 (10k) | 55.1 (150k) | rise-then-**arrest** |
| WD wd=0.1 (λ=0, anchor at final only) | — | — | 54.3 (125k) | arrest (matches λ=0.03) |

Val loss (the entropy-proxy axis) is monotone in every arm. The fall happens at pinned knowledge (d6 = 0.93–0.95 throughout — the gate is what distinguishes consolidation from forgetting; compare wd=3.0's rank drop with d6 = 0.50, which is lobotomy, not descent).

**The per-level traveling wave** (feature_eta2_last of the fresh-FM residual, wd=0.1 control): d1 completes the full arc (peak ~0.09 early → 0.05 at 150k); d2 peaks ~0.13 (10–60k) → 0.10; d3–d6 rise with successively later peaks and **arrest high** (d4 ~0.37, d6 ~0.20 at 150k). In the FM-reg arms deep η² is still rising at 300k (λ=1: d4 0.46). The deep-level falling limb does not occur in any arm at any horizon — with the 300k runs this is no longer attributable to truncation. The reinterpretation (Part 2 confirms it): the deep component is not a transient that failed to resolve; it is the floor.

**Two prior fragments that slot in**: REGIME_TRANSITION's residual top1-PC hump on m4 (1.7 → 33.3% @4k → 11.9% @20k) and RHM_FM_REGULARIZER's weight-norm grow-then-compress turnover (only under FM pressure; WD alone climbs monotonically).

## Part 2 — ensemble invariance over training (new run, 2026-07-07)

**Design**: per (arm, checkpoint), train `n_fm=4` fresh FMs (capB: 8H/16d/mlp2/2L ≈ 791K, the same eval-FM config as `analyze_reg`; gap `post_embed → post_block6`; seeds 911+1000k; 8k steps) on the frozen checkpoint, then measure (i) **pairwise_cos** — input-centered last-token residual cosine across FM pairs (the RHM_LATENT_LOOP Exp 4 instrument: high = residual determined by the input, FM-invariant/DGP-aligned; low = determined by the FM, idiosyncratic); (ii) **shared_norm_frac** = ‖mean residual‖/mean‖residual‖; (iii) per-level η² of the **ensemble-mean** residual (is the invariant core DGP-structured, and where in the hierarchy?); (iv) effective rank/top1 of the mean residual. Arms: `fmreg:1.0` (8 ckpts, 0→300k) and `wd:0.1` (control; wd grid's last ckpt is 125k, so 0→100k here).

| step | fmreg:1.0 ensCos | shared | resNorm | mean-res rank% | mean-res η² d4/d5/d6 | | wd:0.1 ensCos | shared | resNorm | mean-res η² d4/d5/d6 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.555 | 0.786 | 0.46 | 82.0 | .016/.013/.011 | | 0.555 | 0.786 | 0.46 | .016/.013/.011 |
| 500 | 0.735 | 0.887 | 2.53 | 71.1 | .022/.014/.009 | | 0.734 | 0.885 | 2.50 | .023/.014/.010 |
| 4000 | 0.738 | 0.897 | 1.42 | 81.6 | .127/.091/.054 | | **0.666** | 0.872 | 4.41 | .108/.081/.047 |
| 10–12k | 0.834 | 0.944 | 1.22 | 64.2 | .397/.368/.216 | | 0.772 | 0.924 | 8.37 | .368/.239/.139 |
| 25–30k | 0.813 | 0.945 | 1.11 | 56.6 | .449/.359/.213 | | 0.772 | 0.931 | 10.76 | .378/.331/.190 |
| 60–80k | 0.844 | 0.959 | 1.22 | 48.6 | .446/.404/.238 | | 0.804 | 0.954 | 10.75 | .437/.388/.237 |
| 100–160k | 0.832 | 0.955 | 1.22 | 52.4 | .447/.387/.225 | | 0.809 | 0.950 | 9.30 | .375/.371/.230 |
| 300k | 0.834 | 0.956 | 1.10 | **46.0** | **.503/.427/.247** | | — | — | — | — |

(fmreg steps: 0/500/4k/12k/30k/80k/160k/300k; wd steps: 0/500/4k/10k/25k/60k/100k.)

### Findings

1. **FM-invariance rises over training** — 0.555 (random init) → ~0.83–0.84 (pressure) / 0.81 (control); shared-norm fraction 0.79 → 0.96. Nearly half of what a bounded observer misses about a *random* network is observer-idiosyncratic; by late training the residual is overwhelmingly input-determined.
2. **The control dips to 0.666 at step 4k — the invariance minimum coincides with the sophistication peak** (act-rank peak, steepest frontier climb). Mid-training structure is disproportionately FM-idiosyncratic: scaffolding that belongs to the trajectory, modeled differently by every observer. The pressure arm shows no dip (0.735 → 0.738 → 0.834) — the regularizer never lets the computation leave the FM-predictable manifold. This is the transient-vs-floor decomposition measured directly.
3. **The invariant core is deep-DGP-structured and purifies monotonically.** Ensemble-mean residual deep η² rises d4: 0.016 → 0.503, and at 300k *exceeds* the best single-FM η² from the saved parts (0.503 vs 0.463) — averaging across observers strips idiosyncratic noise and concentrates the DGP signal. Its effective rank falls 82% → 46%.
4. **Pressure purifies rather than merely shrinks.** The λ=1 residual is ~8× smaller in norm than the control's (1.10 vs 9.30) yet *more* invariant (0.834 vs 0.809) and more deeply structured (d4 0.503 vs 0.375). Small-but-invariant is the load-bearing combination — contrast RHM_LATENT_LOOP's token-CL, where a small residual (norm 1.2, ens 0.82) had *zero* deep η², i.e. was noise. The descent removed the absorbable part and only the absorbable part.

## Synthesis: the First Law of learning-complexodynamics (as this data supports it)

Complexity rises from the genericity of initialization, peaks on transient scaffolding, and descends to **Soph(attractor | bound)** — with two amendments to Aaronson's picture:

1. **The floor is the attractor's own sophistication, not zero.** Thermal mixing relaxes to the max-entropy state, which is structureless by definition, so the coffee cup's curve returns to zero. Learning relaxes to *whatever computes the DGP*. Grokking / RHM shallow levels: floor ≈ 0 (d1's complete arc). RHM deep levels: floor > 0, certifiable — the root inference is beyond any 1-layer FM by construction, and the late-time residual is FM-invariant (0.83) and deep-structured (η² purifying). The deep η² "arrest" is the floor, not an unresolved transient. The coffee cup is the degenerate case Soph(attractor) = 0.
2. **The descent requires an annealer.** Mixing continues past the peak for free; SGD stops at the loss minimum and freezes the scaffolding in place — knowledge-grok without circuit-grok, a glass rather than an equilibrated liquid. The falling limb appears only under structured pressure toward self-legibility (FM-reg), and generic pressure (WD) anneals the wrong order parameter (norm, not function — RHM_FM_REGULARIZER's headline).

Plus the decomposition that makes both amendments measurable: **hump = FM-idiosyncratic scaffolding (trajectory-owned); floor = FM-invariant DGP core (problem-owned)** — distinguishable only by typed/directional instruments (ensemble invariance, per-level η²), not by any scalar. On a hierarchical DGP the scalar curve is the aggregate of a **traveling wave**: each level's sophistication rises when the frontier reaches it and resolves when mastered, stalling where DGP complexity outruns capacity/signal (m2 root vs m4 ~d3.5 — the DGP-complexity dependence Aaronson's conjecture asks for).

**Tie-back to the research program**: the floor is the same object RHM_LATENT_LOOP identified as the precondition for generalizable self-knowledge (SK works only on a DGP-aligned residual, inverts on scaffolding); the FM-regularizer/ratchet/gating lines are, in this language, annealing protocols on the transient component; and the a2a model-scale result (77M's more concentrated residual) reads as "bigger models are further along the descent," with natural language a DGP whose floor is high and whose attractor these models never reach.

## Caveats

- **One resource bound** (capB, E→b6). The floor is bound-relative by construction; only the structure-function knee is bound-independent, and the knee was measured in a2a (§3.2), not here.
- **ens_cos plateaus ~0.83, not the latent-loop's 0.98 ceiling.** Not directly comparable: different FM config (791K 2-layer capB vs 264K 1-layer), different gap (E→b6 vs b0→b6), different substrate (m2 vs m4), and these fresh FMs are slightly undertrained at 8k steps (cos 0.974 vs the co-trained 0.985), which caps measurable agreement. Do not compare ens_cos across FM configs/gaps — only within a run.
- **n_fm=4 (6 pairs)** — trends are large relative to pair scatter (`pairwise_cos_all` is saved per part for checking), but no CIs.
- **Control ends at 100k** (wd grid's last ckpt is 125k — a cheap top-up: `--arms "wd:0.1" --steps "125000"`).
- **Activation rank is a linear-spectral proxy** — it cannot distinguish interesting structure from junk dimensionality on its own; the knowledge gate and the η²/invariance decomposition are what license the complexity reading (see "the mapping" table).

## Data & schemas (what future agents need)

- **New results (volume `rhm-scaling-data`)**: `/rhm_ensemble_trajectory/parts/{arm}_step{step}.json` (e.g. `fmreg_1p0_step300000.json`) and `/rhm_ensemble_trajectory/summary.json`. Part fields: `pairwise_cos`, `pairwise_cos_all` (per-pair), `shared_norm_frac`, `fm_stats[]` (per-FM cosine/res-norm/seed), `mean_res_effective_rank_pct`, `mean_res_top1_pc`, `eta2_mean_residual` + `eta2_fm0_residual` (per-level dicts; **`level_0` = d6/root … `level_5` = d1**; read `feature_eta2_last`). Parts tagged `_smoke` are from the smoke test (2 FMs × 300 steps) — ignore them.
- **Source trajectories**: `/rhm_fm_regularizer/parts/lam{tag}_step{step}.json` (per-checkpoint: `knowledge` gate, `act_rank` FM-free activation rank, `configs[]` fresh-FM residual metrics) and `/rhm_wd_sweep/parts/wd{tag}_step{step}.json` (same minus `act_rank`). Checkpoints themselves: `{key}/fmreg_8L8H256D_wd0p1_lam{tag}/ckpt_step{N}.pt` and `{key}/wd_sweep_8L8H256D_wd{tag}/ckpt_step{N}.pt`, listed in each dir's `training_info.json`; key = `v16_s2_L6_m2_distinct`.
- **Parsing gotcha (cost this session a wrong table)**: in `rhm_fm_regularizer` parts, `'reg gap' in label` also matches the `"b0->b4 8H capA (UNreg gap)"` config — filter with `'UNreg' not in label`, or match `label.startswith('E->b6')`. The canonical eval FM for η² comparisons is **capB** (`E->b6 8H capB`).
- **Checkpoint grids differ per harness**: fmreg 300k arms include {0, 500, 1k, 2k, 4k, 8k, 12k, 20k, 30k, 50k, 80k, 120k, 160k, 200k, 250k, 300k}; wd 150k arms end at **125k**. `ensemble_trajectory` intersects requested steps with the saved grid and prints what it skipped.
- The step-0 anchors (ensCos 0.555, η² ≈ 0, FM cos 0.996) are the "random model" baseline for any future invariance measurement.

## Reproduction

```bash
cd experiments
# smoke (1 ckpt, tiny FMs, parts suffixed _smoke):
modal run --detach -m rhm.rhm_ensemble_trajectory::ensemble_trajectory \
    --arms "fmreg:1.0" --steps "300000" --n-fm 2 --fm-train-steps 300 \
    --n-eval-sequences 800 --tag smoke
# full run (resumable; skips existing parts):
modal run --detach -m rhm.rhm_ensemble_trajectory::ensemble_trajectory
# optional top-ups:
modal run --detach -m rhm.rhm_ensemble_trajectory::ensemble_trajectory --arms "wd:0.1" --steps "125000"
modal run --detach -m rhm.rhm_ensemble_trajectory::ensemble_trajectory --arms "fmreg:3.0"
```

Part 1's tables need no new compute — re-aggregate from the existing `rhm_fm_regularizer`/`rhm_wd_sweep` parts (e.g. `modal volume get rhm-scaling-data rhm_fm_regularizer/parts <dest>`).

## Next steps

1. **m4 ensemble trajectory** — prediction: lower late-time invariance at the stalled frontier (the un-reached deep levels contribute no invariant core; scaffolding share stays high).
2. **fmreg:3.0 ensemble arm** — dose-response of purification (invariance/η² of the core vs λ).
3. **Language-checkpoint ensemble test** — port to the a2a language FMs (the a2a distillation result's fresh-FM orthogonality suggests language residual directions are partly idiosyncratic; prediction: language sits mid-descent, invariance above scaffolding-regime values but below RHM's late-time 0.83). This also closes the "second efficiency constraint unverified on language" gap flagged against the paper's claims.
4. **Residual-boosting null** (a2a SYNTHETIC_INPUT "Idea B"): train a second bounded FM on the first's residual — the textbook demonstration that the floor is incompressible at the bound, complementing invariance.
5. **Peak-timing × DGP complexity** at matched learning quality (the RESIDUAL_RANK open question): turn "the wave stalls at the frontier" into "peak timing shifts with (L, m)" — the quantitative form of the First Law's DGP dependence.

[^private]: Not mirrored: this link points to a document in the private lab repo (the roadmap, the queue, an unrun spec, reading notes, or a conversation). See the top-level README for what is held back and why.
