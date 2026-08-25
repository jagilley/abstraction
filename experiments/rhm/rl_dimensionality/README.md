# RL Dimensionality Ablation: what reward optimization can and cannot do to a pretrained basis (2026-08-23)

**Parent**: [../README.md](../README.md) · **Design provenance**: [DESIGN.md](DESIGN.md) · **Code**: `rl_dim_ablation.py` (experiment), `aggregate_results.py` / `aggregate_kl.py` / `aggregate_ei.py` (cross-m tables) · **Colab walkthrough**: [`rl_dimensionality_colab.ipynb`](rl_dimensionality_colab.ipynb) (setup explained, scaled-down live replica, full results + interpretation inline)

## Goal

Operationalize two related intuitions in a setting where ground truth is exactly computable:

1. **RL degrades with semantic dimensionality** — RL-from-scratch works on semantically simple domains (games) but semantically complex domains (language) require pretraining, and RL's efficacy relative to the domain shrinks as the domain gets semantically richer.
2. **The representational-basis claim** (after a tweet asking "does RL create new capabilities in the representational basis that was created by pretraining?") — if the answer is no, reward optimization can only rearrange, sharpen, or degrade what pretraining built.

Neither claim is the stronger "RL can never create capability"; the experiment is designed to *localize* where any capability change comes from, not to deny it.

The design splits every result into two categories:

- **Bounds (algorithm-free)**: exact Bayes ceilings and floors computed by sum-product BP on the known parse tree. No optimizer can beat these; no objection about RL variants can touch them.
- **Attainment (algorithm-specific)**: how much of the bounded headroom each training mechanism actually captures, and at what representational cost. To pre-empt "that was just your RL variant," attainment is measured for a ladder of three mechanisms ending in the class limit (see below).

## The dimensionality knob: m at fixed (v=8, s=2, L=6)

m (synonymic multiplicity) moves the RHM's semantic sample-complexity quantity m^L while holding sequence length (s^L=64), horizon, architecture, and compute per rollout exactly fixed — unlike L or s, which change the sequence length and confound the semantic effect with the well-known horizon effect. m is the surface-to-latent degeneracy knob ("how many ways to say the same thing"): with **invertible rules** (collision-free tuples, exact parses, unique roots), m=1 gives exactly v=8 possible sequences (deterministic surface — the game endpoint) and m=v^(s-1)=8 gives exactly the uniform distribution (no structure), so the sweep runs **m ∈ {1,2,3,4,6}**. Sequence entropy is 3 + 63·log₂(m) bits of a 192-bit maximum.

## Setup

**Task**: prefix = first 32 tokens of a 64-token RHM sequence; generate the remaining 32 autoregressively (temperature-1 sampling during training; greedy and sampled at eval).

**Two verifiers** (both run for every configuration):
- **exact** — fraction of suffix tokens matching the sampled ground-truth suffix. A canonical-answer verifier: the synonym lottery is part of the reward.
- **parse** — mean over hierarchy levels of the fraction of valid nodes in the bottom-up possible-set parse of prefix+generation (torch port of `rhm_data.possible_set_parse`, verified equivalent in `self_test`). Synonymy-invariant: accepts any grammatical continuation, graded in composition depth.

**The attainment ladder** (all from the same pretrained checkpoint, all with the same rollout budget of 384K rollouts = 6000 steps × batch 64):
1. **Vanilla REINFORCE** (EMA baseline, no anchor) — plus a from-scratch variant and a step-matched continued-NTP control (`pretrain_only`).
2. **KL-anchored REINFORCE** (`kl_coef=0.1` to the frozen pretrained policy) — the RLHF-like stabilization.
3. **Expert iteration** (6 rounds × 4000 prompts × best-of-16, SFT on winners with suffix-masked loss) — the noiseless, perfectly-credit-assigned limit of the sample-reweighting class that every policy-gradient method approximates. Scope caveat: this idealizes the positive-gradient half of the class; exotic exploration or negative-gradient schemes sit outside the bound.

**Pretrain-to-plateau, not to budget**: each m pretrains until aligned val loss plateaus (best checkpoint kept, min 3K / max 30K steps). The claims concern what reward optimization does to a *given* basis; fixed-budget pretraining would confound "RL got dumber" with "pretraining got worse."

**Exact references (BP)**: per-level NTP Bayes floor; exact-match greedy ceiling (mean-max suffix marginal given prefix, one BP pass), sampled-policy reference (mean ΣP²), prefix-blind floor (best prefix-ignoring predictor), uniform floor 1/v; Monte-Carlo parse floors. Verified against brute-force enumeration on small trees (max error ~1e-16).

**Representational readouts** at every checkpoint: NTP val loss vs the Bayes floor (excess = distributional damage), per-level NTP excess (which hierarchy levels the model actually knows), per-layer feature eta² against ground-truth hierarchy levels, per-level generation accuracy, greedy-vs-sampled gap (policy entropy).

## Part 1 — Bounds (algorithm-free)

| m | EM greedy ceil | EM sampled ref | EM blind floor | parse blind floor | NTP Bayes floor (nats) |
|---|---|---|---|---|---|
| 1 | 0.8418 | 0.8418 | 0.7461 | 0.9403 | 0.0219 |
| 2 | 0.2791 | 0.1911 | 0.2739 | 0.5000 | 0.6971 |
| 3 | 0.2428 | 0.1605 | 0.2424 | 0.4740 | 1.0989 |
| 4 | 0.1770 | 0.1356 | 0.1766 | 0.5208 | 1.1059 |
| 6 | 0.1616 | 0.1301 | 0.1616 | 0.6667 | 1.1040 |

(uniform NTP floor = ln 8 = 2.0794 nats; parse ceiling = 1.0 at every m)

Two structural facts, both pure computation:

**B1. The canonical-answer verifier's semantic content vanishes with m.** The prefix-conditional headroom (greedy ceiling − blind floor) is 0.096 at m=1, 0.005 at m=2, ~0.0005 at m=4, ~0.0000 at m=6. Deep hierarchies mix: prefix→suffix correlations cross the root and are per-token negligible at L=6. What remains achievable decomposes into position-marginal learning (uniform→blind floor) and policy sharpening (sampled ref→greedy ceiling, e.g. 0.191→0.279 at m=2) — neither requires deep structure. No RL variant can be exempted from this; there is nothing to optimize toward. (Corollary noted for the ratchet line: `rhm_rl_ratchet`'s 27.8%→38.6% RL generation gain under sampled eval is consistent with sharpening toward the greedy ceiling rather than new structure.)

**B2. The two verifiers order "dimensionality" in opposite directions.** Grammar occupancy m/v rises with m, so a validity verifier's reward gets *denser* as synonymy grows while a canonical-answer verifier's gets emptier. "Semantic dimensionality" is not one axis for RL; it is a property of the verifier × domain pair.

Also structural: even at m=1 the exact-match ceiling is 0.842, not 1.0 — distinct root rules can share the same left child, so the prefix doesn't always determine the root.

**Pretraining coverage falls with m as m^L predicts** (fixed 2.68M params):

| m | plateau steps | val (nats) | captured excess entropy |
|---|---|---|---|
| 1 | 3000 | 0.0234 | 0.999 |
| 2 | 7000 | 0.7021 | 0.996 |
| 3 | 13250 | 1.1192 | 0.979 |
| 4 | 14500 | 1.4335 | 0.663 |
| 6 | 7250 | 1.8575 | 0.228 |

At m=4/6 the deep levels (L3+) are essentially unlearned (per-level excess ~+0.5 to +1.0 nats ≈ uniform). This is the basis whose boundary the attainment ladder then probes.

## Part 2 — Attainment

### 2a. Vanilla REINFORCE self-destructs at every m

Pure REINFORCE (no anchor) drove NTP excess to **9–14 nats above the floor at every m** (worse than uniform, 2.08), collapsed policy entropy (greedy ≡ sampled everywhere), and flattened per-layer eta² (m=2 exact arm: L5 feature eta² 0.647→0.137 at block0). Most strikingly, at m=1 — where the pretrained policy sat essentially at the Bayes ceiling (greedy exact 0.849 vs ceiling 0.842) — REINFORCE dragged it to 0.388 while its own training reward fell 0.63→0.38. On the parse verifier at m=2/3 the reward climbed to ~0.92–0.94 and then crashed (m=2: 0.92→0.42→0.55) — optimizer self-destruction, not signal absence. The unanchored m=6 parse "success" (sampled parse 1.000) was total mode collapse (14.2 nats excess). Scratch RL never approached pretraining anywhere: even at m=1, 384K rollouts left it at 0.378 exact — below the prefix-blind floor (0.746) that NTP passes in a few hundred steps.

### 2b. The KL anchor cures the destruction and reveals how little there was to gain

With `kl_coef=0.1`, NTP excess drops to 0.05–0.8 nats at m≥2 and entropy collapse disappears.

- **Exact verifier**: zero gain at every m≥2 — sampled accuracy sits at the calibrated ΣP² reference (0.190 vs 0.191 at m=2, stable across checkpoints), greedy where pretraining left it. At m=1 the anchor fixes the pathology: anchored RL holds the ceiling (0.846) instead of destroying it. As bound B1 requires.
- **Parse verifier**: small genuine polish at m≥2 (sampled parse +0.006 to +0.017; root validity 0.902→0.961 at m=2) at small NTP cost. At m=1 it still harmed (0.977→0.758, root 0.930→0.243, 2.9 nats excess) — a variant artifact, as the class-limit arm then showed.

### 2c. Expert iteration — the class limit — is far stronger, and still never moves the basis boundary

Sampled-policy parse validity (the distribution, not just the argmax) and root-level validity, final round:

| m | pretrained | REINFORCE | +KL | **EI** | root: pretrained | +KL | **EI** | NTP excess: KL | **EI** |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.977 | 0.500 | 0.758 | **1.000** | 0.930 | 0.243 | **0.999** | 2.93 | 0.28 |
| 2 | 0.980 | 0.552 | 0.993 | 0.981 | 0.902 | 0.961 | 0.922 | 0.29 | 0.11 |
| 3 | 0.919 | 0.583 | 0.925 | **0.932** | 0.556 | 0.570 | **0.644** | 0.08 | 0.20 |
| 4 | 0.817 | 0.901* | 0.834 | **0.884** | 0.248 | 0.288 | **0.447** | 0.42 | 0.65 |
| 6 | 0.731 | 1.000* | 0.743 | **0.800** | 0.153 | 0.160 | **0.292** | 0.79 | 0.89 |

*mode-collapsed (10–14 nats NTP excess); EI values are honest distributional improvements — the STaR-style flywheel visibly spins (winner data quality at m=6 rises 89%→99% root-valid round over round), no entropy collapse, m=1 fully recovered.

**The dissociation (headline result).** At m=4 and m=6, round by round, root validity climbs monotonically while knowledge of the deep rules — per-level NTP excess at the levels pretraining never learned — stays pinned at uniform and *slightly worsens*:

| round (m=6) | root validity (sampled) | L3 excess (nats) | L4 excess (nats) |
|---|---|---|---|
| 1 | 0.142 | +0.969 | +1.027 |
| 2 | 0.174 | +0.974 | +1.027 |
| 3 | 0.233 | +0.983 | +1.032 |
| 4 | 0.232 | +0.978 | +1.028 |
| 5 | 0.283 | +0.982 | +1.030 |
| 6 | 0.292 | +0.991 | +1.039 |

(m=4 shows the same pattern: root 0.280→0.447 while L3 excess +0.511→+0.564.) Per-layer eta² shows no new hierarchical structure anywhere (m=6 block0 L5: 0.104 pretrained → 0.109 after EI). The class-limit mechanism doubled performance on a deep-structure-requiring metric **without learning any deep structure** — probability-mass redistribution over trajectories the basis could already produce, selected by the verifier. In language terms: pass rates up, the model still doesn't know the rules. This dissociation is only measurable because the DGP is known.

**The exact-verifier m-gradient, at the class limit.** EI-exact's per-rollout improvement falls monotonically with m — sample reward over 6 rounds: +0.020 (m=2), +0.011 (m=3), +0.002 (m=4), ~+0.002 (m=6) — the empirical attainment curve matching bound B1's headroom collapse. The gains that do occur are anchor-free sharpening (m=2: sampled 0.191→0.215, above the calibrated ΣP² reference and toward the greedy ceiling, with only 0.17 nats excess).

**Representational cost per unit reward grows with m.** Within EI on the parse verifier, NTP excess rises 0.11 → 0.20 → 0.65 → 0.89 nats across m=2→6 even as reward gains grow — at higher semantic dimensionality the same mechanism trades away more of the pretrained distribution per unit of reward. The m=1 EI model is the miniature: parse validity 1.000 with the distribution among the 8 legal sequences miscalibrated (L3 excess +0.706) — perfectly grammatical, distributionally distorted.

## Findings

1. **[Bound] Canonical-answer verifiers lose their semantic content as synonymy grows** — prefix-conditional exact-match headroom 0.096→~0 across m=1→6 by exact computation; validity verifiers gain reward density on the same knob. "RL is bad on complex domains" is, in this substrate, a theorem about the verifier × domain pair before it is a fact about any algorithm.
2. **[Attainment] Across the full ladder — vanilla REINFORCE, KL-anchored REINFORCE, and the expert-iteration class limit, ×5 dimensionalities ×2 verifiers — reward optimization never created structure the pretrained basis lacked.** Deep-level rule knowledge (per-level NTP excess) and eta² hierarchy structure stayed at pretrained values in every condition; the capability boundary sat exactly where pretraining left it. What varied by mechanism was everything else: vanilla PG destroyed the basis (9–14 nats), the KL anchor preserved it with near-zero gains, and EI produced large, honest task gains by selection and sharpening within the basis.
3. **[Attainment] The reward/representation trade steepens with dimensionality**: EI's NTP damage per unit of parse-reward gain grows with m, and EI-exact's climb rate falls to zero with m. The "models get more fried as tasks get more sophisticated" pattern appears in the best-behaved member of the class, not just the fragile ones.
4. **[Method] The class-limit arm is what makes 2–3 robust to the "your RL variant" objection**: EI is strictly stronger than the policy-gradient variants here (it alone recovers m=1), so the immobile basis boundary cannot be attributed to a weak optimizer — while the honest scope note stands (exploration-bonus/negative-gradient schemes are outside the idealization).

Interpretation is deliberately scoped: this is one substrate, one model size, single-seed (effects are large, monotone in m, and internally replicated across three mechanisms and two verifiers, which triangulates seed sensitivity about as well as one seed can), and EI's validity curves were still climbing at round 6 — though its deep-level excess trend across rounds is flat-to-worsening, which is the falsifiable part. Where capability creation *would* show up in this framework is the filtered-imitation channel scaled further: SFT on verifier-selected samples is pretraining on self-generated grammar data, and nothing here rules out that channel teaching missing rules with far more data — the observation is that at the RL-matched budget it demonstrably did not, while task reward doubled anyway.

## Reproduction

```bash
cd experiments/

# Self-tests (CPU): torch parse == possible_set_parse; BP marginals == brute force
modal run -m rhm.rl_dimensionality.rl_dim_ablation::self_test

# Base sweep: one job per m (pretrain-to-plateau + scratch_rl/pretrain_rl/pretrain_only x both verifiers)
for M in 1 2 3 4 6; do
  modal run --detach -m rhm.rl_dimensionality.rl_dim_ablation::run_setting --m $M
done

# KL-anchored arm (loads the base run's pretrained.pt)
for M in 1 2 3 4 6; do
  modal run --detach -m rhm.rl_dimensionality.rl_dim_ablation::run_setting --m $M \
    --only-kl --kl-coef 0.1 --run-tag kl01 \
    --pretrained-path rl_dimensionality/v8_s2_L6_m${M}_both_seed42/pretrained.pt
done

# Expert-iteration arm (rollout-matched: 6 x 4000 x 16 = 384K)
for M in 1 2 3 4 6; do
  modal run --detach -m rhm.rl_dimensionality.rl_dim_ablation::run_setting --m $M \
    --only-ei --run-tag ei01 \
    --pretrained-path rl_dimensionality/v8_s2_L6_m${M}_both_seed42/pretrained.pt
done

# Aggregate (pull results.json files locally first; see each script's docstring)
python3 aggregate_results.py   # base sweep cross-m tables
python3 aggregate_kl.py        # KL vs unanchored
python3 aggregate_ei.py        # EI vs KL vs base
```

Results live on the `rhm-scaling-data` volume (chromatic) under `/data/rl_dimensionality/v8_s2_L6_m{M}_both_seed42{,_kl01,_ei01}/` — `results.json` plus `pretrained.pt` and per-condition final checkpoints.
