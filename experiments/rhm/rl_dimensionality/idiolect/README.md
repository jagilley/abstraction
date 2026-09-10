# Idiolect drift: what reward optimization does to the coordinate the verifier can't see (2026-08-28)

**Parent**: [../README.md](../README.md) · **Code**: `idiolect_drift.py` (measurement + Modal sweep), `aggregate_idiolect.py` (tables) · **Results**: `idiolect_results_seed43.json` (complete), `idiolect_results_full.json` (first sweep, before the seed-43 arm) · **Runnable**: [`idiolect_colab.ipynb`](idiolect_colab.ipynb) — self-contained Colab, scaled-down live replica plus the full-scale results embedded

## Goal

The parent experiment established that reward optimization never moved the pretrained basis boundary, and that a canonical-answer verifier's semantic content provably vanishes as synonymic multiplicity `m` grows — leaving the synonymy-invariant *parse* verifier as the only one with signal at high `m`. This sub-experiment asks what that surviving verifier does to the one coordinate it is **exactly invariant to**: which of the `m` synonymous rules realizes a given latent feature.

The motivating observation (Jasper's): recent models invent esoteric terms with ambiguous definitions — well-formed, seemingly meaningful to the model, hard for an external reader to pin down — in a way that was much less apparent in the pretrain-and-instruction-tune era. The framework here suggests a mechanism worth measuring rather than a conclusion: pretraining's objective *is* calibration to the corpus's surface forms, so it is the only force pinning word choice to shared convention. A reward that cannot see word choice does not push toward idiosyncrasy; it stops holding vocabulary in place. Whether anything then drifts, how much, and whether the direction is shared or arbitrary are empirical questions.

This is an **inference-only analysis of the parent's existing checkpoints**, plus one new training arm (the seed-43 privacy control). No new mechanism, no new task.

## The measurement

`generate_rules_invertible` makes every legal s-tuple map to exactly one `(feature, rule)` pair. So for any generated sequence, the rule used at every internal node is **exactly recoverable** — synonym choice is ground truth, not an estimate. `recover_rule_usage` folds bottom-up through the inverse tables; `self_test` checks it against `_generate_with_traces`' ground-truth traces at every `m` and every level, and its validity flag against `possible_set_parse`.

Statistics are taken over nodes whose **entire subtree lies inside the model-generated suffix** (at `prefix_len = 32`, `s = 2`, `L = 6` this is the right half of the tree at each level), so the choice was the model's rather than the prefix's.

Headline metric, per level and node-count-weighted overall:

- **`kl_cond` = log₂(m) − H(rule | feature)**, in bits. `0` = the model picks synonyms as often as the DGP does; `log₂(m)` = it always picks the same one. Also reported: `H_marg`, `valid_frac` (grammaticality of the same generations), and root validity.

Two references make the numbers readable, in the parent's bounds-vs-attainment spirit:

- **DGP floor** — the same statistic on true DGP samples at matched node counts. Entropy estimates are downward biased at finite `n`, so this is the floor `kl_cond` cannot go below (0.0003 → 0.0011 bits across `m = 2…6` at `n = 4000`).
- **m = 1 as an internal control** — with one rule per feature there is no synonym to drift into, and every condition returns exactly `0.0000`.

Two scope notes on the metric. Conditions with lower grammaticality contribute fewer valid nodes and so carry slightly more finite-sample bias; this matters only for the collapsed REINFORCE arms, where the metric is saturated anyway. And the **greedy** numbers (also computed, in the JSON) are near-saturated for *every* model including the pretrained one — argmax decoding is deterministic, so it necessarily commits to one synonym. They do not measure what the sampled numbers measure and are not interpreted here.

## Setup

Every checkpoint `rl_dim_ablation` saved, at `m ∈ {1,2,3,4,6}`: `pretrained`, `pretrain_only` (continued NTP), `scratch_rl` / `pretrain_rl` (vanilla REINFORCE), `pretrain_rl_kl` (`kl_coef=0.1`), and `ei` (expert iteration) — both verifiers throughout, plus EI's per-round checkpoints. 4000 sampled generations per checkpoint at temperature 1.0, from held-out prefixes shared across all conditions.

Two additional arms answer "is the drift private?":

- **same-seed rerun** (`ei02`, identical config to `ei01`) — the run-to-run nondeterminism floor.
- **seed 43** (`--only-ei --seed 43`, from the *identical* `pretrained.pt`) — same verifier, same mechanism, same 384K-rollout budget, different seed. This is the only new training in this sub-experiment.

## Results

### 1. The two verifiers move synonym choice in opposite directions in m

`kl_cond` (bits, sampled, n = 4000):

| mechanism | m=1 | m=2 | m=3 | m=4 | m=6 |
|---|---|---|---|---|---|
| DGP floor | 0.0000 | 0.0003 | 0.0005 | 0.0007 | 0.0011 |
| pretrained | 0.0000 | 0.0016 | 0.0026 | 0.0031 | 0.0058 |
| + continued NTP | 0.0000 | 0.0009 | 0.0023 | 0.0040 | 0.0045 |
| REINFORCE (exact) | 0.0000 | 1.0000 | 1.5850 | 2.0000 | 2.5850 |
| REINFORCE (parse) | 0.0000 | 1.0000 | 1.5848 | 1.8223 | 2.5203 |
| +KL (exact) | 0.0000 | 0.0021 | 0.0046 | 0.0039 | 0.0066 |
| +KL (parse) | 0.0000 | 0.0016 | 0.0045 | 0.0081 | 0.0125 |
| **EI (exact)** | 0.0000 | **0.0553** | 0.0476 | 0.0326 | **0.0242** |
| **EI (parse)** | 0.0000 | **0.0166** | 0.0229 | 0.0472 | **0.0783** |
| *max possible = log₂(m)* | 0.0000 | 1.0000 | 1.5850 | 2.0000 | 2.5850 |

Under the synonymy-invariant verifier the drift **rises** with synonymy; under the canonical-answer verifier it **falls**. The curves cross around `m = 3–4`. Both directions follow from the parent's bound B1: the exact reward's prefix-conditional headroom collapses with `m`, so best-of-16 selection becomes near-random and SFT on near-randomly selected samples has no systematic direction to push. This is the same verifier × dimensionality reversal the parent found for reward *density*, appearing on an independent measurement.

Controls behave. `m = 1` is exactly zero everywhere. Pretraining and continued NTP sit within ~5× the finite-sample floor, so the metric is not picking up generic training noise. The `kl_coef=0.1` anchor suppresses the drift nearly completely (0.0125 at `m=6` against EI's 0.0783) — as expected, since a KL to the pretrained policy is precisely a penalty on distributional deviation, including in synonym choice.

Vanilla REINFORCE saturates at exactly `log₂(m)` at every `m` — total commitment to one synonym. That is the already-known entropy collapse, not drift-while-healthy, and it is uninformative about the question here.

### 2. The drift arrives packaged with improvement, not damage

EI on the parse verifier, round by round at `m = 6` (seed 42 / seed 43):

| round | `kl_cond` (bits) | root validity | node grammaticality |
|---|---|---|---|
| 1 | 0.0134 / 0.0139 | 0.167 / 0.153 | 0.921 / 0.919 |
| 2 | 0.0264 / 0.0266 | 0.200 / 0.191 | 0.930 / 0.928 |
| 3 | 0.0395 / 0.0379 | 0.211 / 0.198 | 0.935 / 0.933 |
| 4 | 0.0460 / 0.0483 | 0.245 / 0.237 | 0.940 / 0.938 |
| 5 | 0.0646 / 0.0626 | 0.269 / 0.266 | 0.945 / 0.943 |
| 6 | **0.0792 / 0.0721** | **0.306 / 0.282** | **0.952 / 0.947** |

All three climb together, monotonically, in both seeds. The final `m=6` EI-parse model is **more grammatical than the pretrained model it started from** (0.952 vs 0.914 node validity), roughly doubles root validity (0.141 → 0.306), and sits **13× further from the corpus's synonym distribution** (0.0058 → 0.0783 bits). Nothing about the output looks degraded; it is progressively less the corpus's dialect.

The exact-verifier arm is the contrast that shows this is not merely "training moves things." Its drift also climbs across rounds (0.0085 → 0.0262 at `m=6`), but root validity *falls* (0.130 → 0.111) and grammaticality *falls* (0.909 → 0.899). Under the invariant verifier, drift comes with gains; under the canonical one, drift is just damage.

This is the parent's central dissociation restated on a new axis. There, validity climbed while deep-rule knowledge stayed pinned at uniform. Here, validity climbs while the model's way of *saying* things moves steadily away from everyone else's — in both cases, the verifier's number improves while something it cannot see degrades.

### 3. The magnitude is reproducible; the direction is largely arbitrary

Two EI runs from the identical `pretrained.pt`, same verifier, same budget, different seed. Drift *magnitude* replicates closely (final `kl_cond`, parse arm: 0.0166/0.0177 at `m=2`, 0.0472/0.0419 at `m=4`, 0.0783/0.0700 at `m=6`). Drift *direction* does not.

Treating √JS between two models' feature-conditioned rule distributions as a distance:

| m | dist(pret, EI s42) | dist(pret, EI s43) | dist(EI s42, EI s43) | angle between drifts | cross-seed ÷ same-seed |
|---|---|---|---|---|---|
| 2 | 0.0649 | 0.0593 | 0.0755 | **74.8°** | 6.6× |
| 3 | 0.0694 | 0.0712 | 0.0890 | **78.5°** | 5.3× |
| 4 | 0.0951 | 0.0903 | 0.1001 | **65.3°** | 5.6× |
| 6 | 0.1235 | 0.1150 | 0.0922 | **45.3°** | 4.0× |

At `m = 2,3,4` the two runs end up **further from each other than either travelled from their shared ancestor** — near-orthogonal drift. At `m = 6` they share more (45°), still with a large independent component. The same-seed rerun sits at 0.0115–0.0229, so cross-seed disagreement is 4–6.6× the run-to-run floor.

The two *verifier* arms diverge similarly and near-additively (EI-exact vs EI-parse: 0.0154–0.0200 bits JS, against 0.0056–0.0153 for either vs pretrained), i.e. they too drifted in roughly different directions rather than different distances along one axis.

**Reproducible in amount, arbitrary in direction** is the signature of an idiolect rather than a correction: a mechanism converging on a genuinely better convention would have both runs find approximately the same one. The narrowing of the angle toward `m = 6` is taken up in [`direction/`](direction/README.md), which shows it is a level mixture — near-perfect alignment at the shallow levels where the verifier filters hard, near-orthogonality at the deep levels where it does not — and traces the aligned part to the verifier selecting the synonyms the model executes most reliably.

## Findings

1. **A verifier that is invariant to surface choice lets surface choice drift, increasingly so as synonymy grows** (EI-parse `kl_cond` 0.017 → 0.078 bits across `m = 2→6`), while a canonical-answer verifier's drift shrinks over the same range (0.055 → 0.024) — the parent's verifier × dimensionality reversal, on an independent measurement.
2. **The drift co-occurs with improvement on everything the verifier can see.** At `m=6`, six EI rounds raise root validity, raise grammaticality above the pretrained model's, and raise drift 13× above pretrained — monotonically and in both seeds. Under the canonical-answer verifier the same drift co-occurs with *falling* validity, which separates "drift that buys reward" from "drift that is damage."
3. **The direction of the drift has a private part and a shared part.** Independent seeds from the identical basis drift comparably far and 45–79° apart, 4–6.6× the same-seed noise floor; at `m ≤ 4` they are further from each other than from their common ancestor. The child node [`direction/`](direction/README.md) resolves why the angle *falls* with `m`: the shared part is the verifier's selection on grammaticality leaking onto the synonym coordinate through the model's uneven competence at synonyms, present exactly where attempts often fail (shallow levels, high `m`); a selection-free control removes it at every level, and a fresh pretraining seed reproduces it because the difficulty profile is the grammar's. The private part is a random walk where the model is already fluent.
4. **A KL anchor nearly eliminates it** (0.0125 vs 0.0783 at `m=6`), which prices the effect: the same mechanism that the parent found preserves the basis also preserves shared vocabulary.

## Scope and what this does not show

RHM synonyms are **interchangeable by construction** — there is no meaning difference between rules at all. So this establishes that the mechanism exists, is measurable, scales with synonymy as the framework predicts, and produces arbitrary rather than shared conventions. It does **not** establish that this mechanism accounts for the term-invention behaviour observed in frontier models, where invented terms plausibly do compress something the model is tracking. Competing explanations are untouched by this experiment: reward models with a stylistic preference for confident jargon (a reward-*hacking* story, where the verifier rewards neologism rather than being blind to it), within-context self-conditioning at inference (no training involved), and training on model-generated data generally.

Single substrate, single model size (2.68M params), one rule seed, and — for everything except the EI arms — a single training seed. The EI results are the ones carrying the claims and they are replicated across two seeds plus a same-seed rerun. Level-resolved statistics are in the JSON but not analyzed here; whether drift concentrates at particular hierarchy levels is open.

The natural next probes: a **`kl_coef` sweep**, since the anchor suppresses the effect almost entirely and a dose-response curve would price shared-vocabulary preservation against reward gain; and, for the real phenomenon rather than the mechanism, a **coin-then-define test** on frontier models — elicit a coined term, then ask for a cold definition in independent fresh contexts, against real technical terms of matched corpus rarity, testing whether coined terms are stable in-context and unstable out of it.

## Children

### [`direction/`](direction/README.md) — why the seeds' drift directions converge as `m` grows (2026-09-01)

The falling angle (75° → 45°) is real (bootstrap ±2°, noise null 60°) and verifier-caused. Per level, the cross-seed angle is small and the shared drift aligns with the pretrained checkpoint's competence at each synonym exactly where the model's validity is low (m = 6, L1–L3: 17°/10°/16°, cos +0.75/+0.79/+0.83), and near-orthogonal where validity is 1. A selection-free EI arm (random reward) drifts ~90° from both seeds at every level; EI from a fresh pretraining seed drifts toward the same shallow-level synonyms because the two checkpoints agree on which are hard (cos up to 0.94). Drift thus has a private random-walk component and a shared pull toward the already-fluent forms — a narrowing, not a convention. The one-step selection differential conditioned on validity cannot see this and sits at its noise floor; teacher-forced loss by synonym is the instrument.

## Reproduction

```bash
cd experiments/

# Self-tests: rule recovery vs generation traces, validity vs possible_set_parse,
# metric calibration at both endpoints (CPU, ~1 min)
modal run -m rhm.rl_dimensionality.idiolect.idiolect_drift::self_test

# The seed-43 privacy control (the only new training here; five detached L4 jobs)
for M in 1 2 3 4 6; do
  modal run --detach -m rhm.rl_dimensionality.rl_dim_ablation::run_setting --m $M \
    --only-ei --run-tag ei_s43 --seed 43 --save-round-ckpts \
    --pretrained-path rl_dimensionality/v8_s2_L6_m${M}_both_seed42/pretrained.pt
done

# The sweep: inference only over every checkpoint (~70 min on one L4)
modal run --detach -m rhm.rl_dimensionality.idiolect.idiolect_drift::analyze \
    --n-eval 4000 --out-tag seed43

# Tables (results file is committed alongside this README)
cd rhm/rl_dimensionality/idiolect && python3 aggregate_idiolect.py
```

Results live on the `rhm-scaling-data` volume (chromatic) at `/data/rl_dimensionality/idiolect/`, with the seed-43 training arm under `/data/rl_dimensionality/v8_s2_L6_m{M}_both_seed43_ei_s43/`.
