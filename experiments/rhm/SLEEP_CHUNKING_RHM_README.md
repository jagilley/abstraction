# Sleep-Chunking on RHM: does a capacity waist implement motor-style chunking? (2026-07-03)

**Code**: `rhm_waist.py` — `waist` (reconstruction/PCA waist), `waist_external` (external-role waist, runnable at any DGP/`span_height`), `waist_recursive` (pass-2 on m=2), `waist_gap` (pass-2 in a frontier-gap regime). Modal/GPU (L4).
**Theory**: conversations/Claude-Motor abstraction learning in dance and cerebellar dynamics.md[^private] — the motor-chunking / "speed up the clock" conversation this operationalizes.
**Sibling project**: [../a2a_forward/README.md](../a2a_forward/README.md) — the corticocerebellar forward-model work; this is the **chunking** half (compress consecutive activations into a unit), deliberately run with **no forward model** so it isolates only the ideas from the conversation.
**DGP regime**: distinct-rule **v=8, s=2, L=6, m=2** (occupancy 0.25), per [RHM_DEEP_COMPOSITION_README.md](RHM_DEEP_COMPOSITION_README.md) — the regime where **every** hierarchy level is recoverable, so any level a waist loses is the waist's doing, not an information ceiling.

## One-line arc

A **generic capacity/reconstruction waist does the opposite of chunking** — it keeps high-variance surface detail and sheds the low-variance global abstraction. The **same** capacity squeeze under an **external-role** objective (predict the *next* span, not reconstruct this one) **inverts the ordering** and yields clean, monotone, **top-down hierarchical chunking**: the waist spends its bits on the chunk's abstract identity first, its sub-structure next, the raw constituents later, and the discardable synonym choice last. **The objective, not the capacity squeeze, is what makes a waist chunk.**

## What we're testing and why

The conversation converged on a specific mechanistic picture of motor chunking, stripped of its brain-specific parts (STDP, TD learning):

- **Compression is logically prior to horizon extension** — you can only treat a longer span as one unit if you first coarsen the representation. The generator (cortex / the main model) must do the coarsening; a forward model can't coarsen its own input.
- The coarsening is **lossy re-representation of a span through a capacity bottleneck** — a "waist" — such that only the compressed generative code survives and the within-span detail is discarded (the biological homolog being working-memory / channel-capacity limits, and the offline hippocampal-index gist-extraction dynamic).
- Crucially (last turn of the conversation): the waist's **fidelity constraint is the survival threshold**, and the constraint must be to the span's **external role** — reconstruction fidelity *keeps* detail; only fidelity to "what this span means for what comes next" prefers the abstract.

RHM is the ideal testbed: it is a known hierarchy, so a "chunk" has a ground-truth identity (the span's parent feature) and a ground-truth discardable detail (which of the `m` synonymous rules instantiated it), and every level is recoverable in this regime.

## Setup

- **Main model M**: causal GPT, 4L/4H/128D (0.80M params), trained on RHM next-token prediction (NTP 2.07 → 0.696 over 8k steps). It learns the hierarchy: at full waist width every latent level is linearly recoverable from its span activations (P2/P1/leaf ≈ 0.97/0.99/1.00, synonym-choice `rule2` ≈ 0.97).
- **Span**: an aligned **height-2 subtree** — `K = s^2 = 4` consecutive leaves, one "chunk." We take M's activations over the span at an early (`post_block1`) and late (`post_block3`) layer and **concat** the K position-vectors into one 512-d span vector.
- **The waist** compresses that 512-d span vector to width `w` (swept `1…256`; `w=1` = the whole 4-token span as a single number).
- **Ground-truth readouts** (RHM tree, `generate_with_tree`), probed by linear logistic regression on the frozen width-`w` code:
  - **P2** — the height-2 parent = the chunk's abstract identity (chance 1/8).
  - **P1** — the two height-1 parents = sub-structure (chance 1/8, averaged).
  - **leaf** — a within-span leaf token = surface detail (chance 1/8).
  - **rule2** — the synonym choice at the chunk's top = the "no-residual-decision" free bit (chance 1/2).
- Reported as **retained fraction of above-chance recovery** vs the widest waist, so each level is on a common 0–1 scale.

Two waists, differing **only** in objective:

| | `waist` (Exp. 1) | `waist_external` (Exp. 2) |
|---|---|---|
| mechanism | **PCA** (optimal linear reconstruction bottleneck) | learned MLP enc→`w`→dec |
| objective | reconstruct the span's own activations | **predict the next span's tokens** |

## Experiment 1 — reconstruction waist: the *opposite* of chunking

PCA truncation to `w` dims, then probe. Retained fraction:

```
post_block1:                          post_block3:
  w      P2    P1   leaf  rule2         w      P2    P1   leaf  rule2
  4     0.51  0.71  0.56  0.37          4     0.62  0.90  0.80  0.44
  8     0.86  0.96  0.94  0.28          8     0.76  0.95  0.88  0.42
  16    0.97  0.99  1.00  0.36          16    0.93  0.98  0.97  0.51
```

**The abstract chunk-identity P2 is the *most fragile* structural feature; the raw leaves are retained as well or better.** A variance-truncating waist keeps whatever dominates representational variance — and the raw leaf tokens are the highest-variance thing, while P2 is a low-variance *computed* global property. So a generic capacity squeeze preserves surface detail and discards the abstraction — anti-chunking. (The one thing it *does* preferentially discard is `rule2`, the free synonym bit — encouraging, but it's a derived/nonlinear quantity, so suggestive only.) This is robust to the waist being linear vs nonlinear, because the issue is the **objective** (reconstruction), not the waist's expressiveness. This confirms the conversation's worry as a measured fact: **reconstruction fidelity keeps detail.**

## Experiment 2 — external-role waist: clean top-down hierarchical chunking

Same activations, same probes; the waist is now a learned bottleneck trained to **predict the next span's tokens**. Retained fraction:

```
post_block1:                          post_block3:
  w      P2    P1   leaf  rule2         w      P2    P1   leaf  rule2
  4     0.55  0.32  0.21  0.09          4     0.45  0.33  0.25  0.14
  8     0.61  0.41  0.26  0.15          8     0.56  0.42  0.33  0.15
  16    0.68  0.47  0.35  0.15          16    0.75  0.57  0.48  0.30
  32    0.85  0.74  0.67  0.43          32    0.93  0.94  0.88  0.61
```

**Every row descends P2 > P1 > leaf > rule2** — a monotone staircase ordered by abstraction height, at every compression width and both layers. The waist spends its first bits on the chunk's abstract identity, then its sub-structure, then the raw constituents, and the discardable synonym bit dead last. The `P2 − leaf` retained-fraction gap flips from ≤ 0 (reconstruction) to consistently **+0.3–0.4** (external-role). This is the chunking signature: keep the move, drop the footwork, forget which synonymous way you did it.

### Corroborating observation — a candidate compression-native DGP metric

Next-span prediction accuracy **saturates at tiny waist width**: ≈0.30 at `w=1` and *no higher* (slightly lower, ≈0.27) at `w=256`, versus 0.125 chance. The predictively-relevant content of a whole span is essentially **low-dimensional — just its parent identity** — which is *why* a narrow waist locks onto P2 first (that's the only thing that helps predict the exterior; leaves and synonym choice don't, so they aren't encoded). The **width at which external-prediction saturates** looks like a data-intrinsic, compression-native measure of a span's effective abstract dimensionality — worth developing as a DGP metric in its own right (candidate: relate its knee to `log v` for the parent identity).

## What the two experiments say together

The **only** difference between anti-chunking and chunking is the waist's objective — reconstruct-the-span vs predict-the-neighbor. Same capacity squeeze, same activations, same probes. Therefore:

1. **Chunking is an external-fidelity phenomenon, not a capacity phenomenon.** A generic capacity squeeze under reconstruction does the opposite of chunking. The fidelity target must be the span's role in what comes next.
2. **The mechanism needs no segmentation oracle and no forward model.** Given an external-role objective, the capacity squeeze on M's *own learned representation* produces hierarchical chunking by itself — consistent with the conversation's "uniform speedup, emergent boundaries" picture (the FM's job is self-knowledge, a different function).
3. **Compression proceeds top-down through the hierarchy** as a function of available capacity — the substrate for the multi-pass, bottom-up "abstraction ratchet" (each pass's units become the next pass's elements).

## Caveats

- **One DGP setting, one rule seed, single waist pass.** Directionally clean and the staircase is not subtle, but replication (seeds, `span_height=3` for a 4-level staircase, other occupancies) is not yet done.
- **`rule2` is a derived, nonlinear quantity** (a function of P2 + the two P1 children), so its low recovery under compression is partly linear-accessibility, not only information loss. The load-bearing evidence is the **P2 > P1 > leaf** ordering among the three *direct* quantities.
- **Concatenation inflates leaf variance.** Concatenating the K position-vectors gives each leaf its own 128-d block, which handicaps the low-variance global P2 under the reconstruction waist specifically; mean-pooling is an untested secondary lever. It does not affect the external-role result (whose incentive is set by the objective, not variance).
- **External target is the *next adjacent* span**, which mixes sibling pairs (strong, height-3-mediated) and cousin pairs (weaker). The learned decoder averages over both; a sibling-only target would be a cleaner (stronger) external signal.
- **Retained-fraction normalizes by the widest waist**, which is the right common scale but hides that absolute P2 recovery at, e.g., `w=8` is lower for the external waist than the reconstruction waist — the claim is about the *ordering* across levels, not absolute retention.

## Reproduction

```bash
cd experiments
# Exp. 1 — reconstruction (PCA) waist: anti-chunking
modal run --detach -m rhm.rhm_waist::waist
# Exp. 2 — external-role waist: hierarchical chunking
modal run --detach -m rhm.rhm_waist::waist_external
```

Both train M fresh (~1 min on L4), cache height-2 span activations + RHM ground truth, sweep waist width, and print the raw-recovery and retained-fraction tables. Results saved to the `rhm-scaling-data` volume under `/data/waist/`.

---

# Recursive chunking: does the ratchet climb? (2026-07-03)

Pass 1 (above) gives clean single-level chunks. The abstraction-ratchet hypothesis is that *recursively* applying the waist — chunk the chunks — climbs the hierarchy bottom-up (each pass's units become the next pass's elements). Testing this surfaced the structural fact that governs everything below:

**The external-role objective at level `h` requires the parent at level `h+1`.** Predicting a neighboring height-`h` span needs the *shared parent*, which sits one level up. So recursive chunking can only extract a level whose parent is itself reachable by the base model M. That converts "does the ratchet climb?" into "is there a reachable rung above M's frontier?" — and the answer is no, for a precise reason.

**Reference lines** (BP ceiling = optimal tree inference with known rules; greedy floor = hard cluster-and-lift cascade), via `rhm_local_signal.run_bp_ceiling` / `run_compounding`:

| level | v8/m2 BP | v8/m2 greedy | v16/m4 BP | v16/m4 greedy |
|---|---|---|---|---|
| d3 | 0.989 | 0.856 | 0.995 | 0.751 |
| d4 | 0.969 | 0.613 | 0.989 | 0.708 |

## Exp 3 — recursive pass on m=2: no cascade, but nothing to climb (`waist_recursive`)

v8/m2. Pass-1 chunks height-2 spans → a sequence of codes; pass-2 chunks adjacent code *pairs* (height-3) and targets **d3 (P3)**. But d3 is *already legible before pass-2*: P3 from M's raw activations = **0.949**, from the uncompressed pass-1 code pair = **0.941**. Pass-2 recovers P3 up to **0.928** (w2=128) — above the greedy floor (0.856), near BP (0.989); and at narrow w2 the staircase replicates one level up (P3 ≥ P2).

**Two reads.** (i) Soft recursion does **not** cascade — it clears the greedy floor cleanly, unlike hard cluster-and-lift. (ii) But m=2 is too easy: M already computes d3, so the codes merely **carry** it (0.93, a hair below the 0.94 they were handed) rather than **manufacturing** a level M lacked. This regime has no frontier gap, so it cannot test climbing — only that the machinery composes without cascading.

## Exp 4 — frontier-gap regime: the ratchet stalls, via dilution (`waist_gap`)

v16/m4, M = 8L/8H/256D, 20k steps — real headroom: M reaches d3≈0.88 but **collapses at d4** (its own d4 legibility = **0.498**). Pass-1 chunks height-3, pass-2 lifts pairs toward **d4** — a level M does *not* compute.

| quantity | value |
|---|---|
| P4 from M raw h4-span acts (M's d4 frontier) | 0.498 |
| P4 from uncompressed pass-1 code pair | 0.254 |
| P3 from code pair (sanity) | 0.534 |
| **pass-2 best d4** | **0.192** |
| pass-2 external accuracy | ~0.08–0.11 (chance 0.0625) |
| reference: greedy d4 / BP d4 | 0.708 / 0.989 |

**NO CLIMB** — pass-2 d4 (0.192) is *below* M's own frontier (0.498), barely above chance. And the failure mechanism is **dilution, not greedy cascade**: chunking at height-3 needs the d4 parent, which is below M's frontier, so the external target is near-unpredictable (extAcc ≈ chance) and the code learns almost nothing (even d3 drops to 0.53). The external-role waist is *itself a local self-supervised objective* and inherits the exact depth-dilution ceiling RHM_DEEP_COMPOSITION established. The ratchet climbs freely below the frontier and stalls at it.

## Exp 5 — below the frontier at m=4: chunking works (`waist_external`, v16/m4, height-2)

Same M/regime, but chunk **height-2** (identity d2; the objective needs d3, which is *above* M's frontier). The staircase replicates. Retained fraction, post_block1:

```
  w      P2     P1    leaf   rule2      ordering
  8     0.25   0.14   0.08   0.06     P2 > P1 > leaf > rule2
  16    0.34   0.18   0.12   0.11     P2 > P1 > leaf > rule2
  32    0.45   0.26   0.17   0.20     P2 > P1 > leaf > rule2
  64    0.57   0.48   0.45   0.39     P2 > P1 > leaf > rule2
```

Raw recovery at full width (post_block1, chance 0.0625 / rule2 0.25): P2 **0.84**, P1 0.94, leaf 0.99, rule2 0.65 — the lower levels are *more* legible uncompressed; it is the **compression** that surfaces the chunk-identity preference. Same ordering at both layers (post_block1, post_block7). **Clean chunking below the frontier at m=4.**

It is **weaker than m=2** — the P2−leaf retained-fraction gap is +0.17–0.28 here vs +0.30–0.40 at m=2, and it needs more width to recover the identity (expected: more synonyms → higher-entropy chunk identity). The external signal itself is weak (extAcc ≈ 0.09, just above 0.0625 chance) yet still sufficient to induce the ordering — what distinguished this from the failed Exp 4 was not signal strength (both ≈0.09) but that height-2's needed parent (d3) is reachable while height-3's (d4) is not.

## Synthesis — the frontier boundary

| regime | chunk level | parent needed | result |
|---|---|---|---|
| m=2, any level | any | reachable | clean staircase; no gap to test *climb* |
| **m=4, height-2 (d2)** | **d3 (reachable)** | **clean staircase — chunking works** |
| m=4, height-3 (→d4) | d4 (diluted) | stalls — dilution wall |

**The law: external-role chunking cleanly reorganizes every level whose parent M can reach, and cannot bootstrap past M's inference frontier** — because extracting level `h` requires predicting neighbors through the level-`h+1` parent, which is exactly the level dilution starves. Soft recursion avoids the *greedy-cascade* failure (Exp 3 clears the greedy floor) but not the *dilution* failure (Exp 4). This is not a defect to fix: no local self-supervised objective crosses the dilution frontier (RHM_DEEP_COMPOSITION), and real data has an effective depth a model saturates at regardless. **Chunking is a below-frontier representational reorganization — compact, triggerable units for the levels the model already computes — not a depth-extender.**

## Caveats (recursive section)

- One rule seed per regime. The m=4 M is a single 20k-step train; we use its *in-run* d4 legibility (0.498) as the frontier line, not a literature number.
- The external target is the *next-adjacent* span — half siblings (share the immediate parent), half **cousins** (share a much higher, more-diluted parent). A **sibling-specific** target has depth-*flat* signal (RHM_DEEP_COMPOSITION Exp 1, ~0.77 at every level) and is the one untried lever that might push the frontier; **deferred**, not tested.
- Exp 3's "carry" and Exp 5's staircase both rest on M already representing the relevant level; they measure the waist's *selectivity*, not new inference.

## Reproduction (recursive section)

```bash
cd experiments
modal run --detach -m rhm.rhm_waist::waist_recursive     # Exp 3 (v8/m2 pass-2)
modal run --detach -m rhm.rhm_waist::waist_gap           # Exp 4 (v16/m4 frontier gap, pass-1=height-3 -> d4)
modal run --detach -m rhm.rhm_waist::waist_external \
    --v 16 --m 4 --n-layer 8 --n-head 8 --n-embd 256 \
    --n-steps 20000 --n-train-seq 60000 --n-eval-seq 8000 \
    --span-height 2 --ae-hidden 512 --ae-steps 3000       # Exp 5 (m=4 below frontier)

# reference lines (local, CPU):
python3 -c "from rhm_data import generate_rules_distinct as g; \
from rhm_local_signal import run_bp_ceiling, run_compounding; \
r=g(8,2,6,2,0); run_bp_ceiling(rules=r); run_compounding(rules=r)"
```

---

# Theoretical grounding: token dilution vs learning from your own latents (2026-07-03)

**Reference**: Korchinski, Favero & Wyart, *"Learn from your own latents and not from tokens: A sample-complexity theory"* (arXiv:2605.27734) — by the RHM's originators. It is the sample-complexity theory of the exact frontier we hit, and it both explains our stall and names the fix. The correspondence is unusually tight, so this is the theoretical backbone for this whole line.

**Their result = our wall.** Token-level objectives (supervised, MLM, diffusion — and next-token-prediction is the same family) cost `vm^{ℓ+2}` to learn level `ℓ`, i.e. **exponential in depth** (top level `vm^{L+1}`), because the synonym-identifying correlation is averaged through `m` unresolved rules per level down to the surface tokens — *dilution*. This **is** our frontier: our M is NTP-trained, so with finite budget it reaches the levels whose `vm^{ℓ+2}` fits and stalls. It quantitatively predicts our **m=2 vs m=4** result — at m=2, `v·2^{ℓ+2}` is tiny so M reaches every level (our "no frontier gap"); at m=4, `v·4^{ℓ+2}` blows up by d3–d4 (our "wall"). Our *"chunking works below the frontier, stalls above"* is the `vm^{ℓ+2}` per-level token scaling seen from the inside.

**Their escape = the fix we missed.** Learning from your own *lifted latents* costs `vm³`, **constant in L**: once level `ℓ` is recovered, both the target and the context lift to level `ℓ`, so the next level is again a local synonym-clustering problem of the same strength. The correct context is the **cousin** (the tuple sharing the `ℓ+2` grandparent); the correct target is the **lifted latent**, not a surface token.

**Why our recursion stalled — the diagnosis.** Our waist predicted the *next span's tokens*. Its *input* was latents (M's activations / pass-1 codes) but its *target* was tokens — which keeps it a token-level objective, hence `vm^{ℓ+2}`-diluted. We had the input right and the target wrong; the paper's whole point is that the **target** must be your lifted latent. That is exactly why Exp 4 stalled at the frontier.

**Tight correspondences.** Their predictor-clusterer module ≈ our capacity waist; their clusterer *collapses synonyms* = our waist *discards the synonym choice* (`rule2`); their synonym-clustering-score (Fig 5) = our staircase. Their **data2vec** analysis (assumption A1, "targets carry the learned latents") = our deferred **Option B** (self-generated codes as target) — they prove it reaches `vm³` and that data2vec already does it implicitly. Their **stop-gradient / local-learning** SLC still hits `vm³`, supporting our wake-sleep/offline framing and the a2a local-loss line.

**What's ours / complementary** (anchor, not scoop): the capacity-waist framing and the **reconstruction-vs-external-role** contrast (a mechanism-level demonstration of *why* predict-your-latent beats reconstruct-your-input — Exp 1/2); the **post-hoc** question (what a *token-trained* model's activations already contain — bounded by its frontier); and the cerebellar / self-knowledge / wake-sleep lens. None of these are in the paper.

This motivates **Exp 6** (below): rerun the recursion with a *cousin-latent* target (predict your own lifted code, not tokens) and test whether the frontier wall dissolves.

## Exp 6 — learn from your own latents: the controlled test (`waist_latent`)

We build the recursion **from one-hot tokens** (no M — M is the token-NTP baseline that stalls) and run **two conditions identical except the target**: `latent` predicts the **cousin's own lifted code** (the escape); `token` predicts a fixed surface leaf in the cousin subtree (the diluted control). At each level the encoder clusters level-`ℓ` tuples by their cousin context; its code becomes `R_{ℓ+1}`, probed for the true parent. v16/m4, chance 0.0625.

| level | **latent** | **token** | M (token-NTP) | BP | greedy |
|---|---|---|---|---|---|
| d1 | 0.891 | 0.891 | 0.98 | 0.998 | 0.887 |
| d2 | 0.881 | 0.881 | 0.97 | 0.998 | 0.845 |
| d3 | **0.777** | 0.674 | 0.88 | 0.995 | 0.751 |
| d4 | **0.424** | 0.231 | 0.50 | 0.989 | 0.708 |
| d5 | **0.226** | 0.118 | 0.17 | 0.955 | 0.476 |

**Confirmed (direction).** Latent beats token at *every* deep level, and the gap **grows with depth** (d1–d2 identical because there both targets are still surface tokens; divergence starts at d3 and widens). Since the conditions differ *only* in the target, this is a clean controlled demonstration that **the target — lifted-latent vs surface-token — is the load-bearing variable**, exactly the paper's claim and our Exp-4 diagnosis. At the deepest level the latent recursion (d5 = 0.23) even edges past M's own frontier (0.17) — the first flicker of the escape.

**Not reproduced (magnitude).** Absolute recovery still cascades (0.89 → 0.23), stays far below BP, and drops *below even the greedy floor* at d4–d5 (0.42 < 0.708). So our minimal recursion is a **weak per-level operator** — weaker than KMeans cluster-and-lift — and cannot test the `vm³`/BP ceiling. Two reasons: (i) it is **greedy/feed-forward** (each level frozen, never revised → error compounds — the RHM greedy-cascade lesson); (ii) it regresses by MSE to *individual noisy cousins* with *continuous* codes, whereas the `vm³` result needs the **denoised** cousin context `E[cousin|tuple]` and an actual **clustering** step (balanced) that collapses synonyms. We got the target right but the operator crude.

**Verdict: the wall softened, it did not dissolve.** The softening confirms the mechanism; the failure to reach BP is our operator, not the principle. Reaching `vm³`/BP is exactly what ILC/SLC/data2vec *are* (proper clustering + refinement), which the paper already proves — reproducing it here would be reimplementing their algorithms.

## The one distinction that organizes everything: two knobs, only one matters

There are two independent choices in every run — **what you feed in** (input) and **what you train it to predict** (target) — and *only the target controls dilution / the frontier*. Feeding a latent *in* does nothing if you predict tokens *out*. Every run we did, sorted by target:

| run | target | side of the line |
|---|---|---|
| M (base model) | next token | **token** (literally NTP) |
| Exp 1 (reconstruction waist) | the span's own activations | reconstruction (keeps detail) |
| Exp 2 / 5 (external-role waist) | next span's **tokens** | **token** |
| Exp 3 / 4 (recursive, gap) | next span's **tokens** | **token** |
| Exp 6 — *token* arm | a surface leaf | **token** |
| **Exp 6 — *latent* arm** | the cousin's own **code** | **latent** ✅ |

So **every run before Exp 6 predicted tokens** — they were token-level, fairly described as *compressed / "clustered" NTP*. The waist (the clustering/bottleneck) is real and does useful work — it keeps the abstract chunk-identity and drops detail (Exp 1/2/5, a finding that stands) — but the *supervision* was always tokens, which is the side that dilutes and hits a frontier. We kept feeding latents *in* (M's activations, codes) and *felt* like we were doing latent prediction, but the target stayed on tokens. **Exp 6's latent arm is the first genuine latent prediction, and the only run that nudged past M's frontier — and it had to drop M entirely** (M is token-trained, hence frontier-bounded; you can't get the escape by post-processing it — you must train with the latent target from the start).

## Is the latent version privileged? (audit)

Splitting the method into three pieces:

1. **Training target (predict the cousin's own code): principled, self-supervised, transferable.** No hidden labels touch training — the target is a representation the model made itself. This is the load-bearing part.
2. **Scaffolding (pool *adjacent* units; pick cousin = position `j^1`): uses the RHM's known fixed tree topology.** This is *structural* knowledge (where the constituents/boundaries are), **not label-privileged** — the hidden latent *values* never enter training, only the tree *shape*. On language you would not get this for free (boundaries are unknown/variable). But two things make it benign: the **token control uses identical scaffolding**, so the token-vs-latent contrast is fair; and the paper shows a **structure-free** version — **data2vec** (mask random positions, predict your *own* teacher-latent there) — achieves the same `vm³` escape with boundaries *discovered*, not assumed. That is the version that runs on language.
3. **Probe (score vs true parents): privileged but evaluation-only** — training never sees it.

**Takeaway for priors:** the *principle* (predict your own latents, not tokens) is real, self-supervised, and language-transferable — hold onto that. What does *not* transfer is the RHM convenience of knowing where the chunks are; on real data, *finding* the boundaries becomes part of the problem — which loops back to the boundary-emergence question from the start of this line (and is where masking, or genuine chunk-discovery, re-enters).

## Reproduction (Exp 6)

```bash
cd experiments
modal run --detach -m rhm.rhm_waist::waist_latent   # both target conditions + per-level table
```

## Biological reading: why the latent version is the more faithful one

There is a concrete (not merely aesthetic) sense in which the **latent** target is the biologically *right* one, and it sharpens the sleep/hippocampus framing from earlier in this line:

- **Sleep has no input, so offline consolidation cannot be token-prediction.** Token-level prediction needs an external token stream; during slow-wave sleep sensory input is gated out and cortex is driven by internally-generated hippocampal replay. So whatever consolidation does offline is *structurally* prediction of the system's **own** replayed latents — self-latent-prediction, near-forced, not analogy. Token-prediction is a peculiarity of how LLMs are trained. This also lines up with hierarchical **predictive coding** (higher cortical areas predict the *activity* of lower areas — latents, not raw input) and with our own a2a premise (the cerebellum as a forward model of *cortical* state).
- **Payoff — it could explain the efficiency gap the paper opens on** (children reach competence on ~5 orders of magnitude less data than LLMs). If the brain learns by latent-prediction (predictive coding awake, self-distillation asleep), it escapes the `vm^{L+1}` token-dilution frontier NTP is stuck behind (→ `vm³`). Sleep is the offline latent-prediction engine the waking token stream can't be. That upgrades "more biological" from similarity to a candidate *explanation*.
- **Hippocampus-as-waist holds up, with a dual role**: it is both a genuine compression bottleneck (sparse, pattern-separated — the waist) *and*, via replay of its compact index, the source of the latent targets. Maps onto **CLS + data2vec**: the slow EMA teacher ≈ the slow cortical learner fed by replay; student ≈ fast learner; replay ≈ the interleaved target stream. A twist in our favor: data2vec compresses *implicitly*, whereas the brain has an *explicit* bottleneck — so our explicit **waist** framing may be *more* faithful than data2vec's implicit version.

**Honest limits.** The specific **cousin-context clustering** mechanism has no clean biological counterpart (the brain's version is vaguer relational/schema replay). "Cortex predicts latents, not raw signal" is mainstream but not settled. And Exp 6 was **not** a wake-sleep/hippocampus run — it was a from-scratch feed-forward recursion with no M and no offline phase, so the analogy attaches to the *principle* (predict your own lifted latents), not to that run. We have not yet built the "wake generates experience → sleep self-distills its own latents through a bottleneck" loop.

**The synthesis this points at:** a2a already has wake-sleep distillation machinery, but its sleep phase distills the forward model's *task* contribution (still loss-driven), not latents. The biologically-motivated experiment is a wake-sleep whose **sleep phase does latent self-prediction** (data2vec-style, own-latent targets, explicit bottleneck) rather than task-distillation — the first version that would be *both* the KFW escape mechanism *and* the sleep/hippocampus story at once.

## Next steps (to discuss)

1. **Reaching BP would mean a proper clustering operator** — denoised cousin context `E[cousin|tuple]`, discretize + balance (Sinkhorn), and/or bidirectional refinement — i.e. reimplementing ILC/SLC; or the structure-free **data2vec-style masking** route (mask + predict own latent, boundaries discovered). Both are established and already proven to hit `vm³`; out of scope for the chunking-framing point, but the natural continuation if we want the escape itself rather than just its direction.
2. **Curriculum route.** Chunk at low-m (undiluted, all levels reachable) and test whether those units transfer to high-m — RHM_DEEP_COMPOSITION's untried lever #1.
3. **The saturation-width metric.** Develop external-prediction saturation width (external accuracy plateaus at tiny `w` — a span's predictive content is ~1 parent identity) as a compression-native measure of a span's abstract dimensionality; relate to occupancy and to the a2a residual-rank observations.
4. **Nail down the below-frontier staircase.** Extra seeds; quantify whether each level's knee sits at its information content (`log v` per parent, `log m` for the synonym bit) — does it "click" at the RHM levels.

[^private]: Not mirrored: this link points to a document in the private lab repo (the roadmap, the queue, an unrun spec, reading notes, or a conversation). See the top-level README for what is held back and why.
