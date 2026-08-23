# RL Dimensionality Ablation — Design (2026-08-23)

**Code**: `rl_dim_ablation.py`
**Question**: How does RL's efficacy — and its effect on the representational
basis pretraining built — scale with the semantic dimensionality of the domain?
Operationalizes the intuition that RL-from-scratch works on semantically simple
domains (games) but that semantically complex domains (language) require
pretraining, and the sharper representational question: does RL create
capabilities in the basis pretraining built, or only rearrange/degrade it?

## The knob: m (synonymic multiplicity), at fixed v=8, s=2, L=6

- RHM semantic sample complexity is governed by m^L. Varying L moves it through
  sequence length s^L (horizon, context, compute per rollout all change);
  varying m moves it with the tree shape, sequence length, architecture, and
  rollout compute exactly fixed.
- m is the surface-to-latent degeneracy knob ("many ways to say the same
  thing"). With invertible rules: m=1 gives exactly v=8 possible sequences
  (deterministic surface — the game endpoint); m=v^(s-1)=8 gives exactly the
  uniform distribution over all sequences (no structure). Sweep: m ∈ {1,2,3,4,6}.
- Sequence entropy: 3 + 63·log₂(m) bits (of 192 max).
- v is the planned negative control (channel, not DGP — sweep later, separately).
  An L-sweep is a possible complement but is horizon-confounded; not part of
  this experiment.

## Conditions (per m; shared init, data streams, RL step budget)

| Condition | Init | Training |
|---|---|---|
| scratch_rl | random | REINFORCE on task reward |
| pretrain_rl | pretrained | pure REINFORCE (no NTP anchor — the frying is a measurement, not a nuisance) |
| pretrain_only | pretrained | continued NTP, gradient-step-matched |

Optional `--kl-coef` arm adds pretrain_rl_kl (REINFORCE + KL to the frozen
pretrained policy — the RLHF-like anchor). Off by default.

**Pretrain-to-plateau, not to budget**: each m pretrains until val loss stops
improving (patience on aligned val loss, min 3K / max 30K steps, best
checkpoint kept). Rationale: the claim is about what RL does to a *given*
basis; fixed-budget pretraining would confound "RL got dumber" with
"pretraining got worse" (ratchet run 9's lesson at m=4). Pretraining compute
floats and is reported, along with captured excess entropy
(H_uniform − val)/(H_uniform − H_Bayes).

**Sequence-aligned NTP** (each row = one full 64-token sequence), unlike the
window-based corpus training of the scaling-law line — so per-position val loss
is exactly comparable to the BP Bayes floor and position embeddings are
hierarchy-aligned. Noted as a deliberate divergence from prior RHM runs.

## Task and rewards

Prefix = first 32 tokens, generate remaining 32 autoregressively
(temperature-1 sampling during RL; REINFORCE with EMA baseline, decay 0.95 —
the ratchet-line machinery, minus the FM apparatus).

Two reward arms (both run sequentially inside each per-m job by default,
`--reward-type both`; RL conditions are duplicated per arm, pretrain_only is
reward-independent):
- **exact** — fraction of suffix tokens matching the sampled ground-truth
  suffix. Verifier demands one canonical surface form; the synonym lottery is
  part of the reward.
- **parse** — mean over hierarchy levels of the fraction of valid nodes in the
  bottom-up possible-set parse of prefix+generation. Synonymy-invariant
  (accepts any grammatical continuation), graded in composition depth, dense
  at the bottom levels. Closer to RLHF/RLVR, where raters don't penalize
  paraphrase.

## "Relative to the relevant dimensions": exact references via BP

All computed with the sum-product machinery from `rhm_bayes_entropy.py`
(invertible rules ⇒ exact parses, unique roots):
- NTP Bayes floor, per level (aligned positions 1+).
- Exact-match greedy ceiling: mean_i max_a P(x_i=a|prefix) — one BP pass with
  the prefix observed. Sampled-policy reference: mean_i Σ_a P². Both per level.
- Prefix-blind floor (best prefix-ignoring constant predictor) and the uniform
  floor 1/v.
- Parse floors by Monte Carlo (uniform-random suffix; blind-argmax suffix).

Self-test findings worth remembering (rule_seed=0, invertible):

- Even at m=1 the greedy ceiling is <1 (0.851, blind floor 0.746) — distinct
  root rules can share the same *left* child, so the prefix doesn't always
  determine the root. The BP references (not intuition) define what's
  achievable at every m.
- At m≥2 the exact-match prefix-conditional headroom (greedy ceiling − blind
  floor) is tiny: 0.006 at m=2, ~0.0005 at m=4, ~0 at m=6. Deep hierarchies
  mix — prefix→suffix correlations cross the root and are per-token tiny at
  L=6. The achievable exact-match gain therefore decomposes into (a) position
  marginals (uniform→blind floor), (b) policy sharpening (sampled→greedy gap,
  e.g. 0.192→0.280 at m=2), and (c) a nearly-zero semantic component. The
  references separate all three. The parse arm is where deep structure is
  genuinely rewarded (floor ~0.04 random / higher with prefix credit, ceiling
  1.0, all structural), so it carries the capability question; the exact arm
  measures reward-information dilution and sharpening under a
  canonical-surface-form verifier.

Headline metric: **fraction of headroom recovered**,
(achieved − blind floor)/(ceiling − blind floor), per m and condition.

## Representational readouts (the tweet's half)

At every checkpoint (each 1500 RL steps) for every condition:
- per-layer feature eta² against ground-truth hierarchy levels (basis
  integrity; the collapse signature from ratchet run 1);
- per-level NTP loss vs the Bayes floor (excess loss = frying);
- per-level generation accuracy (does RL move levels pretraining didn't learn,
  or only sharpen the ones it did);
- reward and quick-val trajectories (sample efficiency, frying dynamics).

## Follow-up arms (added after the base sweep, 2026-08-23)

**KL-anchored arm** (`--only-kl --kl-coef 0.1`, run_tag `kl01`): REINFORCE +
KL to the frozen pretrained policy, loaded from the base run's `pretrained.pt`.
Separates "RL cannot extract the signal" from "vanilla REINFORCE
self-destructs" (the base sweep showed entropy collapse in every unanchored RL
condition).

**Expert-iteration arm** (`--only-ei`, run_tag `ei01`): iterated best-of-N
selection + SFT on the winners (suffix-masked loss), rollout-matched to the RL
conditions (6 rounds x 4000 prompts x N=16 = 384K rollouts = 6000 RL steps x
batch 64). Rationale: the "that's just your RL variant" objection can't be
answered variant-by-variant; EI is the noiseless, perfectly-credit-assigned
limit of the sample-reweighting class that every policy-gradient method
approximates (idealizing the positive-gradient half; exotic
exploration/negative-gradient schemes are outside the bound and outside what
the objection usually means). Either outcome is informative: failure bounds
the class; success localizes capability creation in the filtered-imitation
channel (SFT on verifier-selected samples = pretraining on self-generated
grammar data), not gradient reweighting.

**Claim framing (per discussion with Jasper)**: the theses under test are
(a) the representational-basis claim (RL operates within, and degrades, the
basis pretraining built) and (b) RL-degrades-with-domain-complexity — NOT the
stronger "RL can never create capability". An EI success via filtered
imitation is compatible with, and would sharpen, both claims. Writeups should
split results into bounds (algorithm-free: the BP ceilings/floors) vs
attainment (algorithm-specific: which mechanism captures how much of the
available headroom).

## Provenance / infrastructure

REINFORCE loop and eta² machinery lifted from
`rhm/ratchet/rhm_rl_ratchet.py` (abandoned line — helpers copied, not
imported). BP references imported from `rhm/rhm_bayes_entropy.py`. Parse
reward is a torch port of `rhm_data.possible_set_parse` semantics, verified
equivalent in `self_test`, which also checks BP suffix marginals against
brute-force enumeration on tiny trees.

Runs on Modal (chromatic), volume `rhm-scaling-data`, results under
`/data/rl_dimensionality/{key}_{reward}_seed{seed}/`. One L4 job per m;
conditions run sequentially inside the job.
