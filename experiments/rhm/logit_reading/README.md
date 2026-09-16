# logit_reading — what a model's logits say about its own knowledge, and what a rule violation looks like from inside

**Up**: [../README.md](../README.md) (rhm) · **Files**: [FILES.md](FILES.md)
**Date**: 2026-09-15 · **Status**: Part 1 and Part 2 run, including the two follow-up controls
(same-prefix phasic test, graded legal-vs-legal control). **Child**: [`altitude/`](altitude/README.md)
(2026-09-16) — the follow-up round; it revises Part 1's headline (dated note below) and supplies
Part 2's two missing controls.
**Prompt**: a conversation about whether anything in the brain looks like "reading the logits of
prediction", and about the value system reading prediction error relative to learned expectations
(`conversations/Claude-Neural correlates of model confidence and prediction errors-20260915-1302.md`[^private]).

## Goal

Two questions, in that order, on a substrate where the exact answer is computable.

1. **How tightly do the logits track the model's own epistemic state?** LLM work mostly scores top-1
   calibration error (does stated confidence match accuracy). A model can be perfectly calibrated in
   that sense and still be far from the Bayes posterior — by being the *well-calibrated posterior of a
   less informed observer*. Only an oracle can separate those.
2. **Can a prediction violation be read as something emotion-shaped?** Not "is the model sad", but the
   structural signature the conversation attributes to the value system: keyed to *learned* norms
   rather than raw surprisal, weighted by confidence, persistent, and separable from the output's own
   surprisal.

Regime `v16 s2 L6 m4` with `generate_rules_distinct(seed=0)`, model `8L/8H/256D` (6.34M params) — the
same substrate as [`conditional_revision/`](../conditional_revision/README.md) and
[`endogenous_teacher/`](../endogenous_teacher/README.md), so their reference lines are comparable.

## The instrument: a phase-marginal oracle, and a family of coarse observers

[`conditional_revision/oracle.py`](../conditional_revision/oracle.py) computes `P(x_{t+1} | x_{<=t})`
for **aligned** sequences — it knows leaf 0 is a sequence start. Base models here are trained on flat
windows cut at uniformly random offsets from a stream of concatenated sequences, and position
embeddings index the *window*, not the sequence, so the model never knows the phase. Reading `q`
against the aligned oracle bills phase uncertainty to the model as miscalibration: **at window
positions 0–4 the phase-known oracle charges the cached base 0.296 nats where the phase-marginal
oracle charges 0.018.** [`flat_oracle.py`](flat_oracle.py) therefore computes the exact
phase-marginal predictive that the training distribution actually implies,

    p(w_j | w_<j) = sum_psi pi_{j-1}(psi) p_psi(w_j | w_<j),   pi ∝ prod_{i<j} p_psi(w_i | w_<i)

with `psi` the window's unknown sequence phase (uniform prior — exactly what a random corpus offset
gives). The phase stays genuinely uncertain: the true phase's log posterior is still −0.77 on average
at window positions 32–64.

**Observer k** knows the bottom `k` of the 6 grammar levels exactly and nothing above: it models the
stream as an i.i.d. concatenation of depth-k subtrees (span `s^k`) rooted in `rho_k`, the
position-averaged marginal of that level, and is phase-marginal over `psi` uniform on `[0, s^k)`.
Observer 6 is the truth; observer 0 is the unigram marginal. This is the coarsening the
Cagnetta & Wyart staircase predicts a model climbs, so "which observer is the model" is a fittable
quantity. Model-independent ladder on this grammar (`KL(p_L || p_k)`, nats):

| k | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| uniform synonyms | 1.230 | 0.463 | 0.190 | 0.067 | 0.022 | 0.003 | 0 |
| Dirichlet(1) synonyms | 1.485 | 0.596 | 0.244 | 0.101 | 0.034 | 0.006 | 0 |

**Correctness.** BP is sum-product on the hypertree (cliques = parent + its `s`-tuple), the
factorisation [`conditional_revision/oracle.py`](../conditional_revision/oracle.py)'s brute-force test
established. `flat_oracle._self_test` checks, on tiny grammars by full enumeration: every observer
`k = 0..L` with uniform synonyms (max err 2.2e-16), the same with Dirichlet weights (2.8e-16), and the
`eps`-noise observer (4.4e-16); plus agreement with the aligned oracle on the real `v16 L6 m4` grammar
over 63 prefixes (1.1e-16). Every Part-1 run also carries two Monte-Carlo self-checks from the tower
property: mean realised `-log p(token)` vs mean `H(p)`, and mean realised NLL vs mean `CE(p, q)`. On
the cached base these read **−2.4e-05** and **+1.4e-04** against a standard error of 0.0020.

## Part 1 — the logits are a near-exact posterior of a coarser observer

Two runs: the cached `conditional_revision` base (uniform synonyms, 12k steps), and a fresh
checkpointed trajectory ([`train_trajectory.py`](train_trajectory.py); Dirichlet(1) synonym weights so
Part 2 has legal-but-rare tokens, fresh data every pass, 13 checkpoints to 64k steps). 4096 held-out
flat windows each, all expectations exact under the oracle.

**Cached base** (uniform synonyms, 262k predictions):

| quantity | value |
|---|---|
| `CE − H(q)` — does the model's entropy predict its own expected loss? | **+0.0008 nats** |
| `R²(CE ~ H(q))` per position | 0.971 |
| top-1 ECE (realised / oracle-expected) | 0.0066 / 0.0063 |
| `KL(p_L ‖ q)` — distance from the truth | 0.0788 |
| `KL(p_3 ‖ q)` — distance from observer 3 (the best fit) | **0.0362** |
| `R²(H(q) ~ H(p_bestk))` vs `R²(H(q) ~ H(p_L))` | 0.991 vs 0.953 |

So: **calibrated to itself, but not correct** — the logits are ~0.001 nats from predicting their own
loss while sitting 0.079 nats from Bayes, and they are twice as close to a well-defined coarse
observer as to the truth.

**Trajectory** — the fit climbs the grammar one level at a time, and the self-calibration is there
almost from the start:

| step | CE | `KL(p_L‖q)` | `CE − H(q)` | `R²(CE~H(q))` | ECE | best-fit observer |
|---|---|---|---|---|---|---|
| 0 | 2.854 | 1.665 | +0.146 | 0.00 | 0.072 | 0 |
| 250 | 1.730 | 0.541 | −0.060 | 0.77 | 0.018 | 1 |
| 500 | 1.554 | 0.365 | −0.022 | 0.83 | 0.008 | 2 |
| 2 000 | 1.392 | 0.202 | −0.006 | 0.90 | 0.008 | 2 |
| 4 000 | 1.319 | 0.130 | −0.007 | 0.93 | 0.003 | 3 |
| 8 000 | 1.274 | 0.085 | −0.000 | 0.95 | 0.005 | 4 |
| 16 000 | 1.243 | 0.054 | −0.001 | 0.97 | 0.003 | 4 |
| 24 000 | 1.231 | 0.041 | −0.005 | 0.98 | 0.004 | 5 |
| 64 000 | 1.206 | 0.017 | −0.004 | 0.99 | 0.003 | 5 |

(`H(p_L)` = 1.189 throughout; best-fit observer = `argmin_k` mean `KL(p_k‖q)`. At 64k, `KL(p_5‖q)` =
0.015 against `KL(p_L‖q)` = 0.017.)

> **What training changes is not how well the logits know themselves — that is near-exact from ~2k
> steps — but which Bayesian they are the posterior of.**

> **Revised 2026-09-16 by [`altitude/`](altitude/README.md).** The first half of that sentence is an
> identity, not a measurement: `CE − H(q) = ⟨∂CE/∂z, z⟩`, the derivative of the loss along the
> logit-scaling direction, so any softmax with a free logit scale sits at zero there once trained
> (`α* = 1.00–1.01` from 2k steps, and within ±0.015 in every context class; checked pointwise to
> 1e−07). "Calibrated to itself" is what convergence looks like. The second half stands as an
> *altitude*: a convex interpolation between adjacent observers climbs smoothly `κ* = 0 → 4.9` where
> the argmin is a staircase, and the fitted observer accounts for 70–89% of the model's residual
> through `κ* ≈ 2.9` and 32–40% from `κ* ≈ 4.4` (the residual's level-shape is a coarse observer's
> throughout). So "near-exact posterior of a coarse observer" is a description of the undertrained
> model; at 64k the best family member is only 1.2× closer to the logits than the truth is.

Caveat worth keeping: the per-position `argmin_k` is noisy (neighbouring observers coincide at many
positions; the `best_k` histogram at 64k spreads 0.12–0.27 over k=2..6). The mean-KL argmin is the
statistic to read, and `KL(p_L‖q)` at 64k (0.017) is between the ladder's observer-5 (0.006) and
observer-4 (0.034) rungs.

## Part 2 — stimuli, and what the oracle labels

[`stimuli.py`](stimuli.py) edits one constituent of a flat window: the node at tree level `L-j`
spanning `2^j` leaves, aligned to start at window index `e ∈ [16, 36]`.

- **swap** — the node's feature is replaced and its subtree regrown from the grammar (internally legal;
  whether the *stream* stays legal is the oracle's call, not an assumption).
- **rare** — same feature, lowest-weight synonym, subtree regrown (legal by construction).
- **none** — natural controls.

Labels come only from `flat_predictive` on the edited window: **`t_v`** = the first token with *exact*
probability 0 under the true DGP given the window so far, and **`k*`** = the smallest observer that
assigns it 0, i.e. how many grammar levels you must know to be offended. Supports are nested, so every
observer above `k*` also excludes it, and `k* ≥ j+1` by construction. Asserted in every build and all
passed: legal streams never hit probability 0; nested supports hold; `t_v ≥ e`; `k* ≥ j+1`.

Two sets: `a1` (16,384 windows, 8,073 swaps, 83% of which violate inside the window; `k*` histogram
1166 / 1058 / 1430 / 1496 / 1480 / 56 for k*=1..6) and `swap65k` (65,536 swap-only windows, 82%).
Detection delay grows with `k*` — mean `t_v − e` is 0.4 tokens at k*=1 and ~20 at k*=5 — which is a
fact about the grammar, and it turns out to matter for the design (below).

### 2a. Violation sensitivity is scoped to the levels the model has learned

Detection AUC = the violating token's model-surprisal percentile among legal tokens at the same token
level and the *same window index* (0.5 = indistinguishable):

| step | k*=1 | k*=2 | k*=3 | k*=4 | k*=5 |
|---|---|---|---|---|---|
| 0 | 0.501 | 0.507 | 0.512 | 0.520 | 0.521 |
| 250 | 1.000 | 0.769 | 0.629 | 0.585 | 0.570 |
| 2 000 | 1.000 | 0.978 | 0.774 | 0.666 | 0.640 |
| 8 000 | 1.000 | 0.991 | 0.912 | 0.764 | 0.659 |
| 24 000 | 1.000 | 0.999 | 0.966 | 0.853 | 0.701 |
| 64 000 | 1.000 | 1.000 | 0.985 | 0.923 | 0.774 |

Each level switches on in order, in step with Part 1's climb. This is the minimal form of *"anger is
informed by a running estimate of how people should treat you"*: you can only be offended by rules you
have acquired. It is also close to mechanical given Part 1 — if `q ≈ p_k`, violations at levels `≤ k`
get near-zero mass — which is the point: the scope of the response is inherited from the scope of the
knowledge, with no separate mechanism.

(k*=6 sits at ~0.70 from step 2000 with n=48–56 and no trend; too few events to read.)

### 2b. Against controls from other windows, the separable signal is tonic, not phasic

Each violation is matched to a legal token with the **same token identity, token level, window index**
and model surprisal within 0.1 nats (nearest neighbour, no replacement). Guards: the surprisal AUC and
a linear probe on the token+position embedding, both pinned at 0.500. AUCs on held-out pairs:

| step | pairs | `s − H(q)` | own running surprisal (last 4) | `q` before token | state at `t−1` | state at `t−1` and `t` | state at `t` |
|---|---|---|---|---|---|---|---|
| 0 | 2603 | 0.500 | 0.503 | 0.505 | 0.522 | 0.532 | 0.529 |
| 8 000 | 1162 | 0.498 | 0.643 | 0.655 | 0.632 | 0.649 | 0.638 |
| 24 000 | 859 | 0.508 | 0.696 | 0.682 | 0.702 | 0.723 | 0.714 |
| 64 000 | 605 | 0.514 | 0.754 | 0.781 | **0.818** | 0.821 | 0.782 |

Three things:

- **"Surprise against my own expected surprise" (`s − H(q)`) is a null at 0.50–0.51.** Once surprisal
  is matched, the model's confidence adds nothing as a scalar.
- **The separation is already there before the violating token arrives** (0.818 at `t−1`), and adding
  the token buys +0.003. Higher-level violations are preceded by ~20 tokens of edited material, so what
  this design reads is accumulated unease, not a response to the token.
- **A scalar integrator of the model's own recent surprisal gets most of the way there** (0.754 vs
  0.818 for a linear probe on the residual stream).

Coverage shrinks as the model learns, and not by accident: at 64k, matched controls exist for 68% of
k*=5 violations, 30% of k*=4, 7% of k*=3 and essentially none of k*=1–2, because a violation the model
detects has a surprisal no legal token reaches (mean surprisal at 64k: 11.8 nats at k*=1, 2.3 at
k*=5). **This matched analysis is therefore about violations the model's output does not flag.**

### 2c. With the context held identical, there *is* a response to the token

[`phasic.py`](phasic.py). For each violation, the control is **the same window with the violating token
replaced by a legal one** (`p_L > 0` given that prefix — on file from the stimulus build), chosen as
the legal token whose model surprisal is nearest, within a caliper. The model is causal, so the state
at `t−1` is identical within a pair (checked: max |Δ| 2e-04, and the `t−1` probe reads exactly 0.500).
Pairs are additionally **sign-balanced** on the residual surprisal difference and **token-balanced**
(each token is the violator as often as the twin, by greedy cycle extraction), which pins the paired
surprisal guard and the token-identity guard at 0.50. Caliper 0.3 nats, `swap65k`:

| step | test pairs | forecast after the token | state at `t` (MLP) | `H(q_next)` |
|---|---|---|---|---|
| 0 | 9366 | 0.506 | 0.522 | 0.500 |
| 250 | 3702 | 0.610 | 0.594 | 0.487 |
| 2 000 | 2371 | 0.712 | 0.758 | 0.467 |
| 8 000 | 1967 | 0.754 | 0.785 | 0.452 |
| 24 000 | 1430 | 0.790 | 0.840 | 0.453 |
| 64 000 | 1181 | **0.847** | **0.840** | 0.449 |

By level at 64k (forecast after the token): 0.94 at k*=3, 0.85 at k*=4, 0.83 at k*=5 — present even
where surprisal alone reads 0.77. The step-0 row is the floor for each readout (0.51 / 0.52 / 0.50).
A 0.1-nat caliper gives the same picture at ~40% of the pairs (0.73–0.83 at 64k).

**Reconciling 2b and 2c**: both are true. There is a slow component and a fast one; when controls come
from other windows the slow one dominates what a context-general probe finds, because in this stimulus
family (and arguably in general — a token's illegality is a fact *about* its context) violations follow
odd material. Holding the context fixed cancels the slow component and exposes the fast one.

### 2d. The fast signal is graded likelihood, not categorical legality

The same design with **two legal tokens** after the same prefix, matched on the model's surprisal
(sign- and token-balanced), where one is at least 2× less likely than the other under the true grammar;
positive = the less likely one:

| step | test pairs | guards (surprisal / token) | forecast after token | state (MLP) | `H(q_next)` |
|---|---|---|---|---|---|
| 250 | 2880 | 0.506 / 0.497 | 0.641 | 0.611 | 0.508 |
| 1 000 | 2808 | 0.501 / 0.495 | 0.737 | 0.745 | 0.514 |
| 8 000 | 1789 | 0.502 / 0.501 | 0.714 | 0.732 | 0.534 |
| 64 000 | 736 | 0.502 / 0.501 | 0.669 | 0.720 | 0.556 |

> **What the state and the post-token forecast carry is "how unlikely was that really", on a
> continuum. A grammar violation is the far end of that scale, not a different kind of event.**

And since the model's own surprisal for the token is matched by construction, this is information about
the token's true likelihood that the model's output probability *for that token* does not carry. (The
graded readout is roughly flat over training while the violation readout climbs; note the pair
population is re-selected per checkpoint by the model's own surprisal, so the two trends are not
strictly comparable.)

### 2e. The entropy response runs the wrong way for a violation

Direction, not just magnitude, from the same-prefix pairs: after an **illegal** token the model's next
forecast is *more* confident than after the equally-surprising legal twin (AUC 0.449, i.e. it goes the
other way in 57% of pairs), while after a **legal-but-unlikely** token it is slightly *less* confident
(0.556) — the normative direction. Off-grammar input makes this model commit rather than doubt.

The unmatched output-level persistence agrees. After a violation, model entropy moves +0.02 to +0.12
nats at the next position and ~0 thereafter, against an `eps=0.01`-noise Bayesian's +0.29 to +0.61
decaying over ~4 tokens; on legal rare edits the two are comparable (+0.05 vs +0.08). The
"something's off" signal is held internally (still readable at 0.75 eight tokens later, though that
reading is confounded by the edited stream continuing) and is **not** converted into output
uncertainty. The model was trained on clean data and has no reason to have learned such a response,
which is exactly what the `eps_train` follow-up would test. **Run in [`altitude/`](altitude/README.md)
§Q4b–Q5**: the sign does not change under ε-training, a noise-aware observer at the model's own
altitude goes the other way, and after a violation the model's forecast is that observer's at
`k* − 1` — the finest reading under which the token was still legal.

## What this establishes, and what it does not

**Establishes** (one grammar, one architecture, a 13-checkpoint trajectory):

- The logits of an NTP model on this substrate are, to a good approximation, the **exact posterior of a
  coarse observer**, and the observer climbs the grammar level by level with training while
  self-calibration stays near-exact throughout.
- Violation sensitivity inherits its scope from that climb, with no separate mechanism.
- Beyond surprisal, violation information is carried in a **tonic** component (accumulated recent
  surprise, largely a scalar) and a **phasic** component at the token (readable from the post-token
  forecast and the residual stream, not from any scalar we tried).
- The phasic component is **graded true-likelihood sensitivity** that exceeds what the model's own
  logits assign to the token, and its entropy response to an off-grammar token is the opposite sign
  from a noise-aware Bayesian's.
- **`s − H(q)` is a null.** A fifth instance of the repo's directional-not-scalar pattern, alongside
  [`conditional_revision`](../conditional_revision/README.md)'s list.

**Does not establish:**

- Anything about *using* the signal: everything is open-loop measurement on frozen checkpoints. No
  intervention gives the model a way to act on its own violation signal.
- Anything about valence. Nothing here is good or bad *for* the model; what was tested is the
  structural signature (learned scope, persistence, separability from surprisal), not the evaluative
  part, which needs the violation to cost something toward a goal.
- That the tonic signal is a *model* property rather than a stimulus property: an exact-Bayes observer's
  running surprisal over the same stretch would also be elevated, and that comparison was not run.
  **Run in [`altitude/`](altitude/README.md) §Q4a — it is a stimulus property**: on 2b's own pairs the
  exact observer's running surprisal reads 0.73 at the random-init checkpoint and 0.88 at 64k, above
  the model at every checkpoint, and ~57% of the model's climb is the pair population shrinking.
- Generality of the graded result beyond the pairs the matching admits: pairs exist only where the model
  badly misjudges some legal token's probability, which is a selected population.
- Transfer off `v16 s2 L6 m4`, off this architecture, or to natural language.

## Children

### [`altitude/`](altitude/README.md) — the coarse-observer picture on its own terms (2026-09-16)

**Goal**: five follow-ups on this node, run in its own terms. Is self-calibration a finding or an
identity, and is the discrete rung the right altitude; can the model locate its frontier from its
logits alone; does its predictive, used as a null, score candidate next-level units; the two
controls Part 2 was missing (an exact observer in the tonic design, a noise-aware observer at the
model's altitude in the entropy-direction design); and does a noise-trained twin change sign.

**Finding**: `CE − H(q)` is the temperature-direction derivative of the loss, zero by construction
once trained; the "which observer" half survives as a continuous altitude that describes the
model well while coarse (the fitted observer carries ~80% of the residual) and less well as it
converges (~35%), with the residual's shape a coarse observer's throughout. The depth-k boundaries
are recoverable from the entropy profile alone, at the observer's own ceiling through `k = 3`.
As a null, the model's predictive is level-selective — it prices candidates one rung up and cancels
once a level is absorbed — and is beaten by model-free counts for "is this a unit". The tonic
signal belongs to the stimulus. After a violation the model's forecast is the noise-aware
observer's at `k* − 1`, the finest reading under which the token was legal: it reinterprets rather
than doubts, no Bayesian in either family does that, and ε-training does not change it. Full
record: [`altitude/README.md`](altitude/README.md).

## Reproduction

```bash
cd experiments
# oracle correctness (local, CPU, ~15 s): brute force for every observer, weighted and eps-noise
python -m rhm.logit_reading.flat_oracle

# Part 1 on the cached conditional_revision base (~3 min on an L4)
modal run -m rhm.logit_reading.calibration::calibration --n-windows 4096 --tag cr_base

# the checkpointed trajectory (~40 min on an L4) and Part 1 across its checkpoints (<= 4 GPUs)
modal run --detach -m rhm.logit_reading.train_trajectory::train_trajectory --tag a1_s42
modal run --detach -m rhm.logit_reading.calibration::calibration_sweep \
    --traj-dir /data/v16_s2_L6_m4_distinct/logit_reading/traj_a1_s42

# Part 2 stimuli (~25 min each) and the readouts
modal run --detach -m rhm.logit_reading.stimuli::build_stimuli --n 16384 --tag a1
modal run --detach -m rhm.logit_reading.stimuli::build_stimuli --n 65536 --seed 2027 \
    --swap-only --no-with-reference --tag swap65k
modal run --detach -m rhm.logit_reading.violation::violation_sweep_v2 \
    --traj-dir /data/v16_s2_L6_m4_distinct/logit_reading/traj_a1_s42 --stim-tag a1
modal run --detach -m rhm.logit_reading.phasic::phasic_sweep \
    --traj-dir /data/v16_s2_L6_m4_distinct/logit_reading/traj_a1_s42 --stim-tag swap65k --caliper 0.3
modal run --detach -m rhm.logit_reading.phasic::gradient_sweep \
    --traj-dir /data/v16_s2_L6_m4_distinct/logit_reading/traj_a1_s42 --stim-tag swap65k

# tables (local, reads the JSONs pulled off the volume)
python -m rhm.logit_reading.analyze <dir>/traj_a1_s42 --stim-tag a1 --phasic-tag swap65k --caliper 0.3
```

Results land on the `rhm-scaling-data` volume (**`chromatic` workspace**) under
`/data/v16_s2_L6_m4_distinct/logit_reading/`: `calibration_cr_base.{json,npz}`, `stimuli_{a1,swap65k}.npz`,
and per-checkpoint `traj_a1_s42/stepNNNNNN{,_calibration,_violation2_a1,_phasic_swap65k_cal0.3,_gradient_swap65k}.json`.

## Gotchas worth not rediscovering

- **Use a phase-marginal oracle for flat-window models.** The aligned oracle charges the model 0.30
  nats at early window positions for not knowing something it cannot know. The phase is still
  ambiguous 65 tokens in.
- **Calibrated ≠ correct, and only an oracle separates them.** This model's ECE is 0.003 and its
  entropy predicts its own loss to 0.004 nats while it sits 0.017–0.079 nats from Bayes.
- **Quantile surprisal bins leak in the tail.** The first matched analysis
  ([`violation.readouts`](violation.py), kept for its record) used 8 quantile bins and 16-wide position
  buckets; its surprisal guard drifted 0.50 → 0.59 over training and a probe on the *input embedding*
  read 0.56–0.61 at every checkpoint including random init. Exact cells (token, level, window index)
  plus nearest-neighbour matching on surprisal fixed both. Always carry an input-embedding guard: edited
  stimuli draw a different token mix than natural text.
- **A caliper pins the pooled guard, not the paired one.** Within-pair surprisal differences were
  systematically signed (paired guard 0.67 in the graded control) until pairs were sign-balanced inside
  |Δs| bands.
- **Matching on population controls cannot see a phasic response** when the stimulus that creates the
  violation also precedes it. Same-prefix twins are the design that can.
- **Token identity has to be balanced, not just matched.** Balancing by greedy cycle extraction over the
  (violator → twin) multigraph keeps 40–60% of pairs and pins the identity guard at 0.50.
- **`state_post_embed` is the honest floor for any state probe**, and the random-init checkpoint is the
  honest floor for every readout: the "forecast before + which token" probe reads 0.58–0.64 there
  (token identity is balanced only marginally, so token × position interactions leak), which is why
  that readout is reported but not leaned on.

[^private]: Not mirrored: this link points to a document in the private lab repo (the roadmap, the queue, an unrun spec, reading notes, or a conversation). See the top-level README for what is held back and why.
