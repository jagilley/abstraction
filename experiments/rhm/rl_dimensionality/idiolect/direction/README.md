# Drift direction: why the two seeds' synonym drifts converge as m grows (2026-09-01)

**Parent**: [../README.md](../README.md) · **Code**: `drift_direction.py` (three inference-only Modal functions over saved checkpoints), `aggregate_direction.py` (every table below) · **Results**: `selection_differential.json`, `competence.json`, `controls.json` · **New training**: two control arms of `rl_dim_ablation` (a selection-free EI arm, `reward_type=random`, at m = 2 and 6; and EI-parse from a fresh pretraining seed at m = 6)

## Goal

The parent found that two expert-iteration runs from the identical pretrained checkpoint drift comparably far in synonym choice but in different directions — 75–79° apart at m = 2, 3 — and read that as the signature of an idiolect rather than a shared convention. One number did not fit: the angle fell to 45° at m = 6. If directions converge as synonymy grows, a critic could argue that at the synonymy of natural language the directions would coincide, and the drift would be a convention after all. This node asks whether the convergence is real and what produces it.

## Setup

Everything is measured in the parent's coordinates: per level, the (feature × rule) histogram of synonym choice over free nodes (whole subtree inside the model-generated suffix), valid nodes only. Geometry is done on the per-feature conditional distributions in square-root (Hellinger) coordinates, feature- and node-weighted. This is the same geometry as the parent's √JS triangle and reproduces its angles to within a degree (A1 in the aggregator output).

Four measurements, in the order they were run:

1. **Decomposition of the committed sweep** (`../idiolect_results_seed43.json`, no new compute): per-level angles, shared/private energy split, projection onto the pretrained model's own synonym bias, round-by-round trajectory, parametric-bootstrap CI, and a pure-noise null.
2. **Replayed selection step** (`selection_test`): at the pretrained checkpoint and every EI-parse round of both seeds, sample 16 suffixes for each of 1500 held-out prompts, pick the parse-reward argmax exactly as `run_ei` does, and compare the winners' synonym histogram with the samples'. Its caveat became a finding — see below.
3. **Competence by synonym** (`competence_test`): teacher-forced per-token cross-entropy on 20 000 ground-truth sequences, averaged over each free node's subtree, per (level, feature, rule). Every node's rule is exact ground truth, so this is "how well does the checkpoint know each synonym", with no sampling involved.
4. **Two causal controls**, one L4 job each, measured by `measure_controls` with the same protocol as (2) plus (3):
   - **Selection-free EI** — `reward_type=random` makes the best-of-16 argmax pick a uniformly random sample, so SFT sees the policy's own unfiltered samples. Same rounds, prompts, budget and SFT as the parse arm. Run at m = 2 and m = 6 from the seed-42 checkpoint.
   - **Fresh pretraining seed** — `--seed 44` with no `--pretrained-path` pretrains from scratch (plateau matched the seed-42 checkpoint: val 1.8615 vs 1.8575, captured excess entropy 0.224 vs 0.228), then EI-parse for six rounds. Run at m = 6.

## Results

### 1. The convergence is real

| m | angle, seed 42 vs 43 | 95% bootstrap CI | pure-noise null | same-seed rerun angle | fresh-sample replication |
|---|---|---|---|---|---|
| 2 | 74.9° | [70.4°, 78.2°] | 60.1° | 10.3° | 75.3° |
| 3 | 79.0° | [74.9°, 81.6°] | 60.3° | 14.3° | 80.0° |
| 4 | 64.8° | [61.9°, 66.8°] | 59.5° | 10.9° | 65.7° |
| 6 | 45.2° | [43.9°, 47.6°] | 60.4° | 10.2° | 46.0° |

Three noisy estimates of one distribution form an equilateral triangle, so the null angle is 60°, and the sampling CI is a few degrees wide. 45° at m = 6 is a genuine shared component; 75–79° at m = 2, 3 is genuinely near-orthogonal. The same-seed rerun (ei01 vs ei02) gives the angular resolution of the whole pipeline, about 10°. Column six re-measures the angle on independent samples (16 per prompt, 1500 prompts).

The exact-verifier arm runs the other way: 34° at m = 2 rising to 73° at m = 6. Its shared component is the verifier's visible target, which the parent's bound B1 says collapses with m.

### 2. It is a level mixture, and it accumulates

Per level, the pretrained model's node validity, the angle between the seeds' drifts, and the alignment of the shared drift (d₄₂ + d₄₃)/2 with the pretrained checkpoint's competence at each synonym (−loss, deviation from the feature's mean):

| m | L1 | L2 | L3 | L4 | L5 |
|---|---|---|---|---|---|
| 2 | valid 0.97 · 131° · −0.20 | 0.99 · 100° · +0.69 | 1.00 · 128° · −0.18 | 1.00 · 52° · −0.41 | 1.00 · 50° · +0.39 |
| 3 | 0.94 · 47° · +0.50 | 0.99 · 74° · −0.29 | 1.00 · 87° · −0.51 | 1.00 · 77° · −0.31 | 1.00 · 89° · +0.47 |
| 4 | 0.43 · 11° · +0.79 | 0.88 · 36° · +0.41 | 1.00 · 69° · +0.08 | 1.00 · 93° · +0.23 | 1.00 · 99° · +0.06 |
| 6 | 0.17 · 17° · +0.75 | 0.48 · 10° · +0.79 | 0.80 · 16° · +0.83 | 1.00 · 64° · +0.20 | 1.00 · 80° · −0.14 |

Every cell where the pretrained model's validity is below about 0.9 shows a small cross-seed angle and a strong positive alignment with competence. Every cell at validity ≈ 1 is near-orthogonal, with alignments that scatter around zero. At m = 6 the shallow levels also carry most of the drift energy (L1–L3: 64% of seed 42's), so the overall angle follows them.

The shared component builds up over rounds — m = 6: 70°, 54°, 51°, 44°, 46°, 47° at rounds 1–6 — the signature of a systematic push accumulating against a random walk, rather than of a random walk alone.

Two things it is *not*. It is not the pretrained model's own synonym bias amplified: projecting out (pretrained − uniform) moves the m = 6 angle only from 45° to 51°. And it is not visible in the replayed selection step: the winners-minus-samples differential sits exactly at its multinomial noise floor at every m (|S₀| 0.011–0.013 observed vs 0.010–0.012 null), because that instrument conditions on grammaticality on both sides. An attempt at a poorly-known synonym that fails to parse enters neither histogram, so a selection that acts *through* grammaticality is invisible to it. Measurement (3) is the instrument that sees it.

### 3. The causal controls

At m = 6, drift relative to each arm's own starting checkpoint:

| arm | |drift| | angle vs seed 42 | angle vs seed 43 | per-level angle vs seed 42 (L1…L5) | cos(drift, competence of its own start) per level |
|---|---|---|---|---|---|
| EI-parse, seed 43 (same checkpoint) | 0.056 | 47° | — | 17° 11° 17° 65° 80° | +0.71 +0.78 +0.85 +0.20 −0.00 |
| **selection-free EI** (same checkpoint) | 0.032 | **94°** | **88°** | 91° 128° 112° 93° 91° | −0.19 −0.72 −0.37 −0.29 +0.02 |
| **EI-parse from fresh pretraining seed 44** | 0.055 | **50°** | **56°** | 52° 24° 26° 75° 75° | +0.42 +0.47 +0.83 +0.10 +0.04 |

**Removing the verifier removes the shared component at every level.** The selection-free arm drifts (0.032, a little over half the parse arm's), but orthogonally to both parse seeds and with no alignment to competence. Its drift is also flat across levels (0.030, 0.015, 0.015, 0.030, 0.036 from L1 to L5), where the parse arm's is 2–4× larger at L1–L3 than at L4–L5. So the concentration of drift at the shallow levels is itself a verifier effect. And where the parse arm's drift arrives with rising validity, the selection-free arm's arrives with falling validity (m = 6 L1: 0.164 → 0.131; m = 2 L1: 0.955 → 0.732, parse reward 0.968 → 0.890): self-imitation without a filter degrades grammar as it drifts.

**A fresh pretraining seed drifts toward the same shallow-level synonyms.** Its drift is 50°/56° from the two seed-42-checkpoint runs — about the same as those runs are from each other — and aligned with its *own* checkpoint's competence at L1–L3. The two checkpoints' competence profiles agree there (cos 0.67, 0.80, 0.94 at L1, L2, L3) and not at L4, L5 (0.21, −0.08). Which synonyms of the shallow features are hard to learn is, on this evidence, a property of the grammar and architecture rather than of the particular checkpoint.

## Findings

1. **The convergence of drift directions with m is real and is caused by the verifier.** The parse verifier is invariant to synonym choice only on grammatical output. Where the model's attempts often fail to parse — the shallow levels at high m — attempts at poorly-known synonyms fail more, the best-of-16 winners over-represent the well-known ones, and SFT on winners moves every run the same way. Removing the verifier removes the shared component; the pretrained model's own bias and the valid-conditioned selection differential do not account for it.
2. **The drift therefore has two components, and the parent's finding 3 should be read with both.** A private random walk, present without any verifier, near-orthogonal across seeds, dominant wherever the verifier does not filter (deep levels; all levels at low m). And a shared pull toward the synonyms the model executes most reliably, present exactly where the verifier filters hard. Both move the model away from the corpus's uniform synonym distribution.
3. **The shared pull follows the grammar's difficulty landscape, not the checkpoint's quirks.** Two pretraining seeds agree on which shallow-level synonyms are hard (cos up to 0.94) and drift toward the same ones. On this substrate, then, independent models under this verifier converge on the *easier* forms — a narrowing onto what is already fluent, not a convention that the runs discover.
4. **The private component is where the model is already fluent.** At validity ≈ 1 nothing external constrains synonym choice, and it random-walks in small steps in a direction set by the seed.

## Scope

One rule seed. That two pretraining seeds agree on the difficulty profile rules out "checkpoint quirk" for the shared component; a second rule seed would be needed before saying anything about grammars in general. The competence measure is teacher-forced subtree loss, which folds in the difficulty of the synonym's children; it does not separate "the rule's pair is confusable" from "the rule expands into hard features". The fresh-seed control was run at m = 6 only. RHM synonyms carry no meaning difference, so nothing here speaks to representational divergence, only to surface form.

A prediction worth stating as such: if the shared pull is set by grammar and architecture, models trained on similar corpora and refined with grammaticality-like verifiers should converge on the same fluent phrasings while their rarer-form choices stay private. This substrate cannot test that.

## Reproduction

```bash
cd experiments/

# Control arms (one L4 job each; the random arm at m=2 and m=6, the fresh seed at m=6)
for M in 6 2; do
  modal run --detach -m rhm.rl_dimensionality.rl_dim_ablation::run_setting --m $M \
    --only-ei --reward-type random --run-tag ei_rand --seed 42 --save-round-ckpts \
    --pretrained-path rl_dimensionality/v8_s2_L6_m${M}_both_seed42/pretrained.pt
done
modal run --detach -m rhm.rl_dimensionality.rl_dim_ablation::run_setting --m 6 \
    --only-ei --reward-type parse --run-tag pt44 --seed 44 --save-round-ckpts

# Inference-only measurements over saved checkpoints (minutes each)
modal run --detach -m rhm.rl_dimensionality.idiolect.direction.drift_direction::selection_test
modal run --detach -m rhm.rl_dimensionality.idiolect.direction.drift_direction::competence_test
modal run --detach -m rhm.rl_dimensionality.idiolect.direction.drift_direction::measure_controls

# Tables (results are committed alongside this README)
cd rhm/rl_dimensionality/idiolect/direction && python3 aggregate_direction.py
```

Control-arm checkpoints live on the `rhm-scaling-data` volume (chromatic) at `/data/rl_dimensionality/v8_s2_L6_m{M}_random_seed42_ei_rand/` and `/data/rl_dimensionality/v8_s2_L6_m6_parse_seed44_pt44/`; the measurement JSONs at `/data/rl_dimensionality/idiolect/`.
