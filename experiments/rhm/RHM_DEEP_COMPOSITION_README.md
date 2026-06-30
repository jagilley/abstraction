# RHM Deep Composition: Recoverability, Representability & Learnability (2026-06-29)

**Code**: `rhm_local_signal.py` + `rhm_local_signal_sweeps.py` (training-free signal & inference ceilings), `rhm_thread_b.py` (trained-transformer probing + oracle-aux), `rhm_masked_span.py` + `rhm_invariance.py` (self-supervised objective search). Modal/GPU for the trained-model parts.

**One-line arc**: the deep compositional structure in RHM is **recoverable** (optimal inference, Exp. 3) and **representable** by a vanilla transformer given the right signal (oracle-aux, Exp. 4d) — but is **not learnable** by any self-supervised local objective we tried (Exp. 4–5); the high-level signal is intrinsically diluted, so it must come from privileged supervision or an easier regime/curriculum.
**Prior experiment**: [RHM_META_LEARNING_README.md](RHM_META_LEARNING_README.md) (the meta-learning nulls that motivated going inward)
**Theory**: `conversations/Claude-Meta-learning for hierarchical structure discovery in RHM.md` (the cluster-and-lift / local-signal hypothesis being tested here)

## Motivation

The meta-learning experiments (FOMAML ×3, Reptile ×2, dense and sparse) all failed to produce compositional transfer — they gave anti-overfitting regularization, never compositional depth. The diagnosis was that we'd been testing the *outermost* of three nested bottlenecks with the wrong tools:

1. **Signal** — does a cheap, depth-robust synonymy signal even exist? *(untested)*
2. **Single-instance extraction** — can a model learn high levels from that signal? *(NTP fails)*
3. **Amortization/transfer** — does the operator transfer across instances? *(all the meta-learning work — negative)*

Every prior experiment attacked (3) using NTP — the global, class-supervised-like signal — in parameter space. This experiment goes **inward** to settle (1) and the composability question, training-free, before spending any more training compute. It directly tests the central claim from the theory conversation: that synonyms (s-tuples sharing a parent feature) can be recovered from a **local** distributional signal that is mediated by O(1) rule levels and is therefore depth-independent — the cheap route that end-to-end NTP leaves on the table — and that a recursive "cluster-and-lift" operator can exploit it.

## Setup

All experiments use the project default DGP unless noted: **v=8, s=2, L=6, m=4** (seq_len 64), the same `rhm_data.py` generator as every prior RHM experiment. The RHM is a tree: `nodes[ℓ]` are the true feature values at tree-level ℓ (level 0 = root/class, level L = leaves). We exploit two facts: composition rules are **position-independent** (so synonym statistics pool across all patch positions), and we can record the **full latent tree** (`generate_tree`), giving ground-truth parents at every level for oracle lifts and scoring.

**Bottom-up depth d**: d=1 lifts leaf patches → their level-(L-1) parents; d=L would be the root. Most analyses run d=1..L-1 (the root has no sibling context — it is definitionally the class-supervised case).

**Recovery metric**: occurrence-weighted accuracy of recovering the true parent/synonym group, chance = 1/v = 0.125. For the clustering methods (A, A.5) this is the best cluster→parent matching (Hungarian); for BP it is MAP accuracy of the true feature (BP yields actual feature identities, so no matching is needed).

## Experiment 1 (Thread A): the local signal exists and is depth-flat

For each depth, with the lower levels **oracle-lifted** (true features given, isolating the *signal* question from the *learning* question), we cluster patch s-tuples two ways and score parent recovery: by **local** context (`P(sibling-patch tuple | this tuple)`, ~2 rule levels of mediation) vs **supervised** (`P(root/class | this tuple)`, diluted over L−d levels).

Synonym recovery at the largest budget, 5 rule seeds:

| depth d | local (sibling) | supervised (root) |
|---|---|---|
| 1 (near leaves) | **0.731 ± 0.09** | 0.374 ± 0.03 |
| 2 | **0.790 ± 0.09** | 0.391 ± 0.03 |
| 3 | **0.754 ± 0.11** | 0.548 ± 0.09 |
| 4 | **0.800 ± 0.03** | 0.704 ± 0.05 |
| 5 (near root) | 0.779 ± 0.07 | 0.763 ± 0.06 |

**The local signal is depth-flat (~0.77 at every level), the supervised signal is diluted exactly as `m^(L-d)` predicts (0.76 near root → 0.37 at the leaves), and they cross over.** At the bottom — exactly where end-to-end NTP is most starved — local carries ~2.4× the recoverable parent information (0.61 vs 0.25 above chance). Sample efficiency: local plateaus within a few thousand patch-occurrences (~tens-to-low-hundreds of sequences at the bottom); the class route needs `n_c·m^L = 8·4⁶ = 32,768` sequences for the same level and empirically floors near chance at every budget we tried.

**Conclusion: bottleneck (1) is resolved — the cheap local signal is real and depth-robust.** "L0 is the curriculum" has a foundation, and the meta-learning nulls are explained: NTP feeds on the diluted well that is empty exactly where we needed it.

## Experiment 2 (Thread A.5): the greedy recursive operator cascades

Necessary ≠ sufficient. Here we actually *perform* cluster-and-lift recursively: lift each level using its **predicted** clusters (not ground truth), propagate upward, and measure how recovery degrades. (Full data per level — this isolates error *cascade* from sample complexity.) See the "greedy cascade" column below.

| depth d | oracle 1-step (clean input) | greedy cascade (predicted lifts) | gap |
|---|---|---|---|
| 1 | 0.745 | 0.745 | 0.00 |
| 2 | 0.817 | 0.622 | 0.20 |
| 3 | 0.742 | 0.434 | 0.31 |
| 4 | 0.788 | 0.278 | 0.51 |
| 5 | 0.787 | **0.192** | 0.60 |

**The operator cascades to near-chance by the top.** One ~75%-accurate hard lift already costs 0.20 at the next level; stack four and you reach 0.19 (chance 0.125). `eff_vocab` stays 8/8 throughout, so this is misassignment noise accumulating multiplicatively, not alphabet collapse. The signal is present at every level (Exp. 1) but committing to discrete per-level decisions bottom-up wastes it. **The bottleneck is the inference, not the signal.**

## Experiment 3: the optimal-inference ceiling, and the occupancy law

The RHM is a tree, so exact sum-product (belief propagation) with the known rules gives the **best achievable** recovery of every latent node from the leaves — the information ceiling. (`run_bp_ceiling`: upward likelihood messages + downward prior messages + per-node marginals; verified consistent — leaf marginals are the observed one-hots, the true assignment always has positive posterior.)

| depth d | greedy cascade | optimal BP | BP post(true) |
|---|---|---|---|
| 1 (near leaves) | 0.745 | **0.965** | 0.958 |
| 2 | 0.622 | 0.915 | 0.894 |
| 3 | 0.434 | 0.843 | 0.803 |
| 4 | 0.278 | 0.759 | 0.698 |
| 5 (near root) | 0.192 | 0.629 | 0.549 |
| 6 (root/class) | — | **0.435** | 0.358 |

Two findings, partly opposed:

- **Joint inference is a large, real prize.** BP beats greedy everywhere and the gap *widens* with depth (3× at d=5). Most of the cascade's collapse is algorithmic suboptimality that optimal — soft, bidirectional, evidence-integrating — inference recovers. This is the target for a learned model: track the BP line, not the greedy line.
- **The ceiling itself declines toward the root** — even optimal inference with *known rules* recovers the class only 43.5%. This is the data-processing inequality: the root reaches the leaves only through the lossy, many-to-one composition, so high-level information is genuinely destroyed.

### The occupancy law

Sweeping rule-sampling scheme and (v, m) shows the top-level ceiling collapses onto a single control variable — **tuple-space occupancy `m / v^(s-1)`** (here `m/v`):

| config | m/v | d1 | d3 | d5 | root |
|---|---|---|---|---|---|
| replace v8 m4 *(current `rhm_data.py`)* | 0.50 | 0.96 | 0.82 | 0.59 | 0.42 |
| distinct v8 m4 | 0.50 | 0.97 | 0.87 | 0.58 | 0.40 |
| distinct v8 m2 | 0.25 | 1.00 | 0.99 | 0.95 | **0.80** |
| distinct v16 m4 | 0.25 | 1.00 | 0.99 | 0.94 | **0.78** |
| distinct v32 m4 | 0.125 | 1.00 | 1.00 | 0.99 | **0.92** |
| distinct v32 m8 | 0.25 | 1.00 | 0.99 | 0.94 | 0.79 |

The root ceiling is a clean function of occupancy and nearly independent of absolute v at fixed ratio (v8m2, v16m4, v32m8 all give ~0.79). `replace` vs `distinct` sampling barely matters — collisions are driven by how densely the `mv` legal tuples pack into the `v^s` tuple space, not by within-feature repeats.

## Experiment 4 (Thread B): what a trained transformer actually learns

Experiments 1–3 are training-free. Thread B trains a standard **causal GPT** (8L/8H/256D, ~6.3M params) on RHM next-token prediction (`rhm_thread_b.py`) and asks where its *internal* latent recovery lands between the greedy floor and the BP ceiling. DGP: **distinct-rule v=16, m=4, s=2, L=6** (occupancy 0.25 — chosen for real depth headroom). We linear-probe (and MLP-probe) each block's **last-token** representation for the true latent ancestor at every level; at the last token a causal model has seen the whole sequence, so the comparison to full (bidirectional) BP is fair. Reference lines for this exact rule set (chance **0.0625**):

| level (depth) | greedy floor | BP ceiling |
|---|---|---|
| d1 (leaves) | 0.887 | 0.998 |
| d2 | 0.845 | 0.998 |
| d3 | 0.751 | 0.995 |
| d4 | 0.708 | 0.989 |
| d5 | 0.476 | 0.955 |
| d6 (root) | — | 0.800 |

### 4a. A learned frontier (20k steps, NTP)

Best-over-blocks linear probe vs the references:

| level | greedy | model | BP |
|---|---|---|---|
| d1 | 0.887 | 0.981 | 0.998 |
| d2 | 0.845 | 0.966 | 0.998 |
| d3 | 0.751 | 0.876 | 0.995 |
| d4 | 0.708 | **0.430** | 0.989 |
| d5 | 0.476 | 0.174 | 0.955 |
| d6 (root) | — | 0.082 | 0.800 |

**The model does near-optimal joint inference up to a frontier (~3 levels)** — beating the greedy floor and approaching BP at d1–d3 — **then falls off a cliff: below greedy at d4, ≈chance at the root.** It is not "just local clustering" (it beats greedy where it has learned), but it learns only ~3 levels up. The trajectory shows a bottom-up wave still advancing at 20k (d3: 0.61→0.88 over steps 5k→20k) while val loss has already plateaued.

### 4b. The high-level failure is real, not a linear-probe artifact (`probe_only`)

Linear vs MLP probe at 20k:

| level | linear | MLP | greedy | BP |
|---|---|---|---|---|
| d3 | 0.875 | 0.891 | 0.751 | 0.995 |
| d4 | 0.431 | **0.564** | 0.708 | 0.989 |
| d5 | 0.175 | 0.172 | 0.476 | 0.955 |
| d6 (root) | 0.082 | 0.081 | — | 0.800 |

**Root and d5 are genuinely absent** (MLP = linear = chance — nothing non-linearly buried). **d4 is the live frontier** — non-linearly present (MLP 0.56 > linear 0.43) but still below the greedy floor. The conclusion is robust to probe type.

### 4c. The frontier saturates — no grokking (100k steps)

| step | val loss | d3 (MLP) | d4 (MLP) | d5 (MLP) |
|---|---|---|---|---|
| 20k | 1.561 | 0.897 | 0.603 | 0.180 |
| 40k | 1.540 | 0.924 | 0.764 | 0.234 |
| 70k | 1.537 | 0.928 | 0.793 | 0.282 |
| 100k | 1.538 | 0.933 | 0.790 | 0.284 |

**More training advances the frontier (d4: 0.56→0.79) but it then saturates — smoothly, no phase transition.** By 100k the best-over-blocks recovery is d1 0.98, d2 0.97, d3 0.93, d4 0.79, **d5 0.28, root 0.09**; the shortfall to BP grows monotonically with height (0.01 → 0.71). **Signal-limited, not depth-limited**: in the block×level grid, d4 saturates by block ~4 with blocks 5–7 sitting idle. This is Thread A's dilution at the learning level — the model climbs exactly as far as the diluted NTP gradient carries it (~3.5 levels), then stops.

### 4d. Signal vs capacity — the decisive controlled test (`oracle_aux`)

Same architecture; the *only* thing that differs between conditions is whether the latent **signal** is supplied directly. `ntp_aux` adds an auxiliary loss supervising each block toward the true latent ancestors (completed positions only, per-level-balanced); `ntp_only` is the matched baseline. Final linear probe:

| level | ntp_only\* | **ntp_aux** | BP |
|---|---|---|---|
| d1 | 0.64 | 0.989 | 0.998 |
| d2 | 0.43 | 0.994 | 0.998 |
| d3 | 0.33 | 0.989 | 0.995 |
| d4 | 0.26 | 0.975 | 0.989 |
| d5 | 0.16 | 0.953 | 0.955 |
| **d6 (root)** | 0.07 | **0.798** | 0.800 |

**Given the signal, the same network reaches the BP ceiling at every level, including the root (0.798 ≈ 0.800).** Since the probe reads only leaf-derived representations it *cannot* exceed BP — so the model is doing essentially **optimal joint inference** everywhere. Two corroborations: (i) clean **level-per-block emergence** — d1–d3 by block 0, d4 by block 1, d5 by block 2, root by block 3 (the recursive lift realized in depth, reaching the root in ~4 of 8 blocks); (ii) the aux signal *lowers* NTP loss (1.395 vs 1.444) — the hierarchy helps prediction, NTP just can't extract it.

**Conclusion: Thread B's saturation was signal-limited, not capacity-limited.** The 8L/256D transformer has ample capacity to represent the entire hierarchy to the information ceiling; plain NTP simply never supplies the gradient. *(\*The aligned `ntp_only` here is artificially depressed — the aligned NTP loss excludes the last/probe position, so it received no NTP gradient there; the clean NTP baseline is 4a/4c. This does not affect the conclusion, which rests on `ntp_aux` reaching BP as an absolute existence proof.)*

## Experiment 5: can a self-supervised objective reach deep composition?

4d proved the architecture *can* represent the full hierarchy given undiluted signal — but via **privileged** per-level labels. The constructive question: can a **self-supervised** objective supply that signal? We tried two families (`rhm_masked_span.py`, `rhm_invariance.py`); both fall short, for one instructive reason.

### 5a. Surface masked-span prediction — still diluted (`rhm_masked_span.py`)

A bidirectional encoder predicting masked tokens, two conditions at matched mask count: **span** (mask a contiguous aligned subtree) vs **scattered** (standard MLM). Best-over-blocks probe vs NTP:

| level | NTP (best) | masked-span | scattered-MLM | greedy | BP |
|---|---|---|---|---|---|
| d1 | 0.98 | 0.68 | 0.96 | 0.89 | ~1.0 |
| d2 | 0.97 | 0.26 | 0.84 | 0.85 | ~1.0 |
| d3 | 0.93 | 0.15 | 0.37 | 0.75 | ~1.0 |
| d4 | 0.79 | 0.15 | 0.15 | 0.71 | 0.98 |
| d5 | 0.28 | 0.13 | 0.13 | 0.48 | 0.95 |

**Both variants underperform NTP.** Span masking *failed outright* (MLM loss stuck near marginal entropy ≈ log 16) — masking a whole subtree removes the visible siblings the cheap signal needs, and the many high-entropy masked tokens are near-unpredictable. Scattered MLM learns d1–d2 but trails NTP at d2–d4. Lesson: **surface-token prediction (NTP, MLM, any masking, causal or bidirectional) is diluted at the high levels** — masking geometry doesn't change that. Oracle-aux didn't predict surface tokens; it supervised the latent. So reward *invariance*, not content.

### 5b. A self-supervised invariance objective — the SSL trilemma, collapse, and the limit (`rhm_invariance.py`)

Goal: reproduce oracle-aux self-supervised by rewarding "synonyms → same representation" — infer a masked subtree's parent from context in representation space; mask *size* sweeps levels. Four designs each exposed a concrete failure mode: a genuine trilemma where the contrastive target must be **local** (no global fingerprint), **in-distribution**, and **bootstrappable**, and the naive constructions get only two:

| design | failure mode |
|---|---|
| overlapping full-seq teacher | global-sequence-**fingerprint shortcut** (InfoNCE solved without learning the hierarchy) |
| disjoint complementary-mask views | **cold start** (random encoders → uncorrelated reps → zero gradient) |
| disjoint + MLM bootstrap | **OOD teacher** (subtree-only input the encoder never trains on) |
| DINO cluster-bottleneck (+centering/sharpening) | **works early** (d1 0.90, d2 0.34, d3 0.16 @ 2k steps) **then collapses** (assignment entropy 3.6→0.6, probe→chance) |
| + **Sinkhorn** equipartition | collapse fixed (cluster usage pinned at log 64), but the probe **still** peaks-then-decays |

The Sinkhorn result forced a probing fix: the model trains on *masked* inputs, so the standard *unmasked* probe is OOD and the "decay" was partly mismeasurement. The **in-distribution probe** (`probe_masked`: mask a subtree, read the inferred-parent rep) gives the clean answer:

| level | invariance best — encode-visible / infer-from-context | NTP (best) | BP |
|---|---|---|---|
| d1 | 0.90 / 0.39 | 0.98 | ~1.0 |
| d2 | 0.34 / 0.16 | 0.97 | ~1.0 |
| d3 | 0.16 / chance | 0.93 | ~1.0 |
| d4–d5 | chance / chance | 0.79 / 0.28 | 0.98 / 0.95 |

**The invariance objective underperforms plain NTP at every level and never reaches the high levels** — with every confound controlled (shortcut→bottleneck, cold-start→bootstrap, collapse→Sinkhorn, probe-OOD→in-distribution probe). The decisive reason: **"infer a deep subtree's parent from context" *is* the hard, diluted long-range inference** — framing the loss as invariance rather than surface prediction doesn't make the high-level signal any less diluted. At low levels the objective reduces to its MLM bootstrap (≈ a weaker NTP); at high levels its core task is exactly what dilution starves.

**Conclusion: no self-supervised *local* objective we tried — surface (NTP, MLM, span) or invariance (contrastive/DINO/Sinkhorn) — reaches deep composition.** Oracle-aux worked only because privileged per-level labels inject *undiluted* high-level signal that self-supervision cannot manufacture: the high-level signal is intrinsically diluted in the data, so no single local task concentrates it.

## What we learned

The full arc: **the local synonymy signal is present and depth-flat (Exp. 1) → greedy recursive lifting cascades to chance (Exp. 2) → optimal joint inference recovers most of it but against a ceiling set by tuple-space occupancy (Exp. 3) → a trained transformer climbs only to a signal-set frontier (~3.5 levels) and saturates with no grokking (Exp. 4a–c), yet given the latent signal directly the *same* architecture reaches the BP ceiling at every level, root included (Exp. 4d).** The genuinely hard, *learnable* thing is joint (bidirectional, soft) inference over the tree; the precondition for studying it is a DGP regime where the target is recoverable.

**The binding constraint on solving RHM is the training objective (signal), not model capacity or architecture — the lever is the objective.** This also re-diagnoses the original meta-learning question: meta-learning failed for the *same* reason single-task NTP saturates — both ran on the diluted NTP signal, and you cannot average across instances your way out of a signal that is weak *within* every instance (the meta-learning README's "L0-dominated inner-loop trajectory" is exactly this). The fix is the objective, not single-vs-meta. Meta-learning's proper role is downstream: once an undiluted objective exposes the hierarchy, amortizing the shared *schema* across rule sets (the only cross-instance invariant) is what it is for.

**Exp. 5 sharpens "the lever is the objective" into something harder.** The lever *is* the objective — but no *self-supervised local* objective we built can pull it. Surface prediction (NTP, MLM, masked-span) and invariance objectives (contrastive, DINO, Sinkhorn) all underperform NTP at the high levels and never approach oracle-aux, for one unified reason: the high-level signal is **intrinsically diluted in the data**, and any single local task — predict tokens or enforce invariance — inherits that dilution, because "infer a deep subtree's parent from context" *is* the long-range inference that dilution starves. Oracle-aux works only by injecting *undiluted* high-level signal via privileged labels. So the deep structure is **recoverable** (Exp. 3) and **representable** (4d) but its **learnable self-supervised signal is absent at depth**: it must come from privileged supervision, an easier regime/curriculum that makes high levels locally accessible (lower occupancy), or cross-instance amortization built on such a signal.

This recontextualizes the project's history. The recurring "models learn only 1–2 levels and plateau" pattern — per-level loss, the ratchet, the meta-learning nulls — was measured almost entirely at **v=8/m=4 (occupancy 0.50), where even a Bayes-optimal classifier recovers the root only ~40%.** A meaningful fraction of that plateau is the **information ceiling, not optimization failure**: we have been asking models to recover deep composition in a regime where deep composition is barely recoverable in principle. It also gives a mechanism for the README's "m dominates scaling 3:1" result — m drives occupancy → collisions → identifiability.

## Caveats

- **Oracle 1-step ≠ a from-leaves ceiling.** The Exp. 1 / "oracle 1-step" numbers are given clean lower-level input (one step), so they can exceed BP-from-leaves at high depth; they measure per-level *signal strength*, not full recovery. The apples-to-apples from-leaves comparison is greedy-cascade vs BP.
- **BP knows the rules.** It is the ceiling for "perfect rule knowledge + perfect inference," hence an upper bound on any learned model (which must also learn the rules). The greedy cascade is the "structure-from-data, local-only, no real inference" reference. The two bracket what a trained model could do.
- **BP measures latent/class recovery, not NTP loss.** How much of the per-level *NTP* loss plateau is irreducible is a related but distinct quantity (the autoregressive Bayes ceiling), not computed here.
- ~0.77 in Exp. 1 is an estimator-limited lower bound on the local information (single sibling + hard KMeans). BP shows ~0.96 is available at the bottom from full local+global integration.
- **Thread B probes** are at the **last token** (maximum causal context — fair vs full BP) and are **linear**, with an MLP robustness check (4b) confirming the high-level failures are real, not non-linearly buried.
- **oracle-aux (4d) is a privileged diagnostic** — it uses ground-truth latent labels to prove the architecture *can* represent the hierarchy given signal. It is **not** a solution; the self-supervised version is the open constructive step.
- The aligned `ntp_only` baseline in 4d is depressed by an under-trained last position (its NTP loss excludes the probe position); the clean NTP baseline is 4a/4c (random-window training).
- **Train/probe distribution mismatch (Exp. 5b).** A model trained only on *masked* inputs must be probed *in-distribution* (masked); the unmasked probe under-reports and shows spurious "decay" as the model specializes. `probe_masked` is the correct measurement for the invariance models, and it confirms the negative conclusion.
- The invariance `ckpt_dir` accumulated checkpoints across design iterations; the clean Sinkhorn-40k checkpoints are steps {0, 2000, 5000, 10000, 20000, 39999} (ignore stale 199/299/1499/1999/19999 from earlier variants).

## Reproduction

```bash
cd experiments/rhm

# Single instance (verbose, one rule seed):
python3 rhm_local_signal.py --mode signal     # Thread A: local vs supervised, sample complexity
python3 rhm_local_signal.py --mode compound    # Thread A.5: iterated non-oracle cluster-and-lift
python3 rhm_local_signal.py --mode bp          # optimal tree BP ceiling

# Cross-seed / DGP tables in this README (~90s, CPU, no Modal):
python3 rhm_local_signal_sweeps.py --which all
```

Experiments 1–3 are pure numpy/CPU; run locally. No Modal, no GPU, no data volume.

Thread B (Exp. 4) is on Modal/GPU (`rhm_thread_b.py`):

```bash
cd experiments
modal run --detach -m rhm.rhm_thread_b::thread_b          # 4a: 20k NTP baseline + probe
modal run --detach -m rhm.rhm_thread_b::thread_b --n-steps 100000 \
    --ckpt-steps "0,1000,2000,5000,10000,20000,40000,70000,100000"   # 4c: long run
modal run --detach -m rhm.rhm_thread_b::probe_only        # 4b: linear+MLP re-probe of checkpoints
modal run --detach -m rhm.rhm_thread_b::oracle_aux        # 4d: signal-vs-capacity diagnostic
```

Reference lines (greedy floor / BP ceiling) for the Thread B rule set, locally:

```python
from rhm_data import generate_rules_distinct
from rhm_local_signal import run_bp_ceiling, run_compounding
r = generate_rules_distinct(16, 2, 6, 4, seed=0)
run_bp_ceiling(rules=r, n_sequences=10000)                  # BP ceiling per level
run_compounding(rules=r, n_sequences=60000, verbose=False)  # greedy floor per level
```

Experiment 5 (objective search, Modal/GPU):

```bash
cd experiments
modal run --detach -m rhm.rhm_masked_span::masked_span    # 5a: span vs scattered masked prediction
modal run --detach -m rhm.rhm_invariance::invariance      # 5b: Sinkhorn invariance objective (long: --n-steps 40000)
modal run --detach -m rhm.rhm_invariance::probe_masked    # 5b: in-distribution probe of the checkpoints
```

## Next steps

The self-supervised objective search (Exp. 5) is **closed**: local objectives don't reach deep composition — the deep structure needs *undiluted* high-level signal. Remaining levers:

1. **Curriculum over m (untried, constructive).** Train at m=2 (occupancy 0.125 → far less dilution → NTP should reach deeper, plausibly d4–d5) then continue/transfer at m=4. Tests whether deep compositional features learned *where they're accessible* transfer to where they aren't — i.e. whether an easier-regime curriculum substitutes for the missing high-level signal. This is the one constructive lever we haven't pulled.
2. **Consolidate (this doc).** The arc is complete and self-consistent: recoverable (Exp. 3), representable (4d), not self-supervised-learnable at depth (Exp. 5). Clean open statement: *deep RHM composition requires undiluted high-level signal — privileged supervision, an easier regime/curriculum, or amortization on top of one of those.*
3. **Meta-learning, repositioned.** Its role was never to extract structure from diluted NTP (that failed); it is to *amplify* a working undiluted objective by amortizing the shared schema across rule sets. It re-enters only after a curriculum (or other source) supplies high-level signal single-task.
4. **(Optional) NTP per-level Bayes ceiling** — quantify how much of the per-level-loss plateau is irreducible at each (L, m, v).
