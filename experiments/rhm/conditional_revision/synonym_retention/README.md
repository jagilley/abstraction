# Synonym retention: does the state still carry which synonym realised a constituent that has closed?

**Up**: [../README.md](../README.md) · **Files**: [FILES.md](FILES.md)
**Sibling**: [`../aleatoric_fraction/README.md`](../aleatoric_fraction/README.md) — the same question
without a distance axis, which is why this cut exists
**Re-scopes a published number in**: [`../local_loss/README.md`](../local_loss/README.md) (Finding 4's `syn/str`)
**Idea**: [`ideas/revision_not_surprisal.md`](../../../../ideas/revision_not_surprisal.md) §8
**Date**: 2026-08-09 · **Status**: run. Forward passes only on cached checkpoints; nothing is trained.

## Goal

Two instruments in this line score content **at the moment it arrives**, and both are therefore blind to
the same thing.

- [`aleatoric_fraction/`](../aleatoric_fraction/README.md)'s NTP-equivalence test asks whether two
  candidate tokens induce identical futures *from arrival*. Content that is load-bearing for `w` steps and
  then exactly stops is fully protected at arrival and contributes zero headroom.
- `local_loss/`'s `_swap_sensitivity` perturbs the **last** level-(L−2) constituent and reads at the
  **last** position, so the perturbed span *contains* the read position.

The shape §8's argument is actually about — content that is uncertain on arrival, *then* becomes
irrelevant, while state must still be carried — lives exactly in that blind spot. This cut supplies the
distance axis: perturb a constituent, read the state `w` positions after it **closes**, sweep `w`.

## Design

A constituent `(level d, node j)` closing at leaf `e` is perturbed; the state is read at `t = e + w`. Four
matched arms, all confined to the constituent's span with a bit-identical prefix before it:

| arm | what moves |
|---|---|
| `leaf` | every leaf-emitting rule choice inside the span — **no latent feature moves** (the incumbent's "synonym") |
| `top` | the constituent's own rule choice — its feature is unchanged, every descendant feature is re-realised |
| `str` | `leaf` + `top` (the incumbent's "structure") |
| `rand` / `other` | off-DGP control; independent-sequence spread normaliser |

**The perturbation is frozen across the `w` sweep** — the same bytes change at every `w ≥ 0`, so only read
distance varies. That makes the perturbation-size confound structurally impossible rather than something
to match on, which is stronger than the matched-Hamming discipline `local_loss/` used.
`(d = L−2, j = s^(L−2)−1, w = 0)` reproduces the incumbent cell exactly.

Two readouts. *Causal*: `‖h[e+w] − h_base[e+w]‖` per arm, normalised by independent-sequence spread.
*Probe*: decode the constituent's rule choice and its feature from `h[e+w]` against an **exact Bayes
ceiling** from `oracle.prefix_beliefs`, so retention is `(acc − chance)/(bayes − chance)`, pinned at both
ends.

**The NTP floor is measured per cell, not assumed.** `generate_rules_distinct` draws each feature's tuples
independently across features, so a closed span's realisation is not exactly redundant for the future
(`aleatoric_fraction`'s DGP swap recorded only 55–87% of cells at exactly TV = 0). Every arm therefore
carries its exact next-token TV at the read position, and every readout is reported both pooled and
restricted to the provable `TV == 0` cells.

Checks before reporting: `_realize` round-trips the generator; the clique rule axis equals the raw rule
label; the instrument reproduces the published `syn/str` to 0.003; all shuffled-label guards sit at chance;
exact Bayes ceilings are monotone non-decreasing in `w`, as they must be.

## The m4 curve — NTP alone sheds it, and sheds the synonym bit faster than the identity

Frozen m4 base at λ=0, `post_block6`, level d=4 (span 4 leaves). n_seq = 6000; probe test n = 1800 per
node, 3–4 nodes pooled. Chance: rule 0.25, feature 0.0625.

| `w` | ruleAcc ±SE | **rule retention** | featAcc ±SE | **feature retention** | Bayes(rule) | relLeaf ±SE |
|---|---|---|---|---|---|---|
| 0 | 0.870±0.004 | **+0.858** | 0.931±0.003 | +0.963 | 0.975 | 0.5573±0.0032 |
| 1 | 0.848±0.005 | +0.830 | 0.926±0.004 | +0.963 | 0.973 | 0.1967±0.0011 |
| 2 | 0.653±0.006 | +0.553 | 0.916±0.004 | +0.940 | 0.980 | 0.2378±0.0017 |
| 4 | 0.490±0.007 | **+0.325** | 0.904±0.004 | **+0.911** | 0.989 | 0.2641±0.0022 |
| 8 | 0.309±0.006 | **+0.079** | 0.594±0.007 | **+0.574** | 0.991 | 0.1786±0.0018 |
| 12 | 0.272±0.006 | +0.029 | 0.238±0.006 | +0.188 | 0.993 | 0.1015±0.0013 |
| 16 | 0.259±0.006 | +0.011 | 0.147±0.005 | +0.091 | 0.993 | 0.0766±0.0009 |
| 32 | 0.249±0.006 | −0.002 | 0.078±0.004 | +0.017 | 0.994 | 0.0307±0.0003 |

Shuffled-label guards read 0.241–0.256 (rule) and 0.073–0.083 (feature) at every `w`.

Rule retention halves by `w ≈ 2–3` and is indistinguishable from chance by `w ≈ 16`. The Bayes ceiling is
0.97–0.99 and **rising** in `w` — the information is fully available and increasingly so, so this is the
representation discarding it rather than losing access.

**The synonym bit goes ~3× faster than the identity**: at `w=4`, rule 0.325 vs feature 0.911; at `w=8`,
0.079 vs 0.574. That "discard the realisation, keep the latent longer" ordering is the behaviour §8's fix
was proposed to induce, and here it is at λ=0 with no local loss anywhere. It agrees in ordering with
three other measurements taken on different axes: the external-role waist
([`SLEEP_CHUNKING_RHM`](../../SLEEP_CHUNKING_RHM_README.md) Exp 2), whose monotone staircase
`P2 > P1 > leaf > rule2` puts the synonym bit dead last on a compression axis and at a different regime
(v8/m2); `probe_diag`'s node-class split, which reads past constituents at 0.20 against a ceiling of ~1.0;
and the tracking appendix §3, where the readout collapses 0.946 → 0.350 across a single boundary step.

By level, `w=0` rule retention runs d5 +0.894, d4 +0.858, d3 +0.669, d2 +0.286, d1 +0.078 — the model
barely represents level-1 rule choices at all, consistent with root recovery 0.088.

**The NTP floor does not explain the displacement.** At d=4 the `leaf` perturbation's exact next-token TV
is 0.0085 (`w=0`) → 0.0006 (`w=16`), with 92.7% → 99.2% of cells at exactly TV = 0, and `relLeaf`
restricted to those cells is unchanged (0.5627 vs 0.5573; 0.0764 vs 0.0766).

## Re-scoping `syn/str = 0.635`

**Numerically it survives.** On m2 `lam0_base` at the incumbent cell this instrument reads **0.638** against
the published 0.635.

**It is also distance-robust**, which we did not expect — on m4 at d=4 the ratio runs 0.732 (`w=0`), 0.727,
0.667, 0.624, 0.641 (`w=4`), 0.638 (`w=6`), 0.655 (`w=8`), 0.746 (`w=16`), rising to 0.910 at `w=32` as
both arms decay into a common floor. So the zero-retention-distance framing is *not* what limits it.

What does limit it is the arm construction: `str` is `leaf` **plus** a rule bump — strictly a larger
perturbation of the *same* span — and both arms hold the constituent's own feature fixed. The ratio is
therefore two nested perturbation magnitudes, and neither arm isolates the latent, so it cannot speak to
"drops the synonym while keeping the latent." The cell is additionally flagged `ntp_defined: False`, since
the read position is the last token of the sequence.

**Proposed correction to `local_loss/`'s Finding 4**: `syn/str ≈ 0.635` reproduces and is distance-robust,
but it measures relative perturbation magnitude between nested arms, not selective synonym discarding. The
discarding it was looking for is real and large, appears on the distance axis, and is already present at
λ=0 — so on this evidence it is a property of NTP rather than of the local loss. Finding 4's stated
conclusion ("no evidence of selective synonym discarding" *attributable to the loss*) is unaffected.

## The arm comparison — the content-axis test `local_loss/` did not run

Sorted by achieved compression (`actRk%` joined from the published `parts_ll1`). `aucRule` is the
area under the retention-vs-`w` curve; higher = retains longer.

| arm | actRk% | ruleW0 | ruleW4 | ruleW8 | **aucRule** | featW4 | aucFeat | syn/str |
|---|---|---|---|---|---|---|---|---|
| `lam0_base` | 53.4 | +0.988 | +0.654 | +0.494 | **+0.401** | +1.000 | +0.760 | 0.638 |
| `depth_lam0p1` | 52.7 | +0.994 | +0.624 | +0.396 | +0.419 | +0.999 | +0.759 | 0.611 |
| `depth_lam3p0` | 48.1 | +0.966 | +0.616 | +0.440 | +0.379 | +0.993 | +0.736 | 0.593 |
| `temporal_lam0p1` | 46.2 | +0.980 | +0.615 | +0.462 | +0.400 | +1.000 | +0.767 | 0.590 |
| `depth_lam3p0@300k` | **42.9** | +0.960 | +0.584 | +0.459 | **+0.392** | +0.997 | +0.770 | 0.601 |
| `temporal_lam1p0` | **42.6** | +0.945 | +0.643 | +0.443 | **+0.393** | +0.997 | +0.749 | 0.559 |
| `temporal_lam3p0` | 39.4 | +0.924 | +0.603 | +0.427 | +0.372 | +0.994 | +0.742 | 0.577 |

At the matched-compression pair (42.9 vs 42.6), `aucRule` reads +0.392 vs +0.393. Pooled over all ten arms,
depth (n=6) +0.495 vs temporal (n=3) +0.491, difference −0.005. The whole ten-arm range is 0.038 wide while
`actRk` moves 53.4 → 39.4, and **`lam0_base`, with no local loss at all, sits in the middle of that range.**

**On this axis and this substrate, the conditioning gap does not move retention.** Two things make this a
cleaner null than `local_loss/`'s: retention is normalised between chance and an exact Bayes ceiling, so
the gauge collapse that dominated the raw local-loss term cannot masquerade as it; and m2 retains *more*
than m4 (featW4 1.000 vs 0.911), so the arm sweep ran on the substrate with more room to move.

The one thing compression does touch is **presence, not persistence**: spearman(`actRk`, ruleW0) = **+0.891,
p = 0.001** (range 0.924–0.994), spearman(`actRk`, rel0) = +0.879, p = 0.001; but in the tail,
spearman(`actRk`, ruleW4–8) = +0.394, **p = 0.26**. Compression slightly reduces how much synonym identity
is present *while still inside* the constituent and does nothing to how long it survives afterward.

## Where the two readouts disagree

They agree on shape for `w ≤ 8` (both fall ~5×) and on the level ordering. They disagree on the floor: at
`w ≥ 16` the probe reads exactly zero (retention +0.011 → −0.002, within 1 SE of shuffled) while causal
displacement is still 0.077 → 0.031 of independent-sequence spread at **80–100σ**.

So the state genuinely still moves 16–32 tokens after the constituent closed, but that movement carries no
*linearly decodable* synonym identity. The reading we favour is a leak rather than a store — the forward
map is not exactly invariant even where the content is functionally irrelevant — but the `TV == 0`
restriction rules out only one of the three candidate explanations, and no nonlinear probe was run. **Open.**

## An impossibility worth recording

We tried to build a control arm that is NTP-*required* at distance, as a positive reference for the
retention curve, and could not. Any perturbation that changes the future predictive distribution must move
a latent whose descendants extend past the read position, which breaks prefix-identity. On this RHM,
everything about a closed constituent is very nearly exactly NTP-redundant.

If that argument is right it is the same fact `aleatoric_fraction`'s missing headroom records, reached
structurally rather than by measurement — and it means the absence of a positive control here is a
property of the substrate rather than a gap in the instrument. It also means the retention curve has no
upper reference arm, which is a real limitation of what follows.

## What this establishes, and what it does not

**Establishes** (one regime each):

- On the frozen m4 base at λ=0, closed-constituent synonym identity decays from +0.858 at `w=0` to +0.079
  by `w=8` and chance by `w=16`, against a rising exact Bayes ceiling of 0.97–0.99 — so the representation
  discards it rather than losing access.
- The synonym bit decays ~3× faster than the constituent's feature identity, which is the selective
  ordering §8 proposed a local loss would induce, present without one.
- Across ten `local_loss` checkpoints spanning 53.4 → 39.4% activation rank, retention is flat to within
  0.038 and the no-local-loss anchor sits mid-range; at matched compression the two arms differ by 0.001.
  This null is not gauge-confounded.
- `syn/str ≈ 0.635` reproduces and is distance-robust; its limitation is the nested-arm construction.

**Does not establish:**

- Anything beyond **linear** decodability. `probe_diag` established the linear probe as the house
  instrument on this checkpoint, but "not linearly decodable" is not "not represented" — and the causal
  readout disagrees at large `w`.
- That retention is the right axis for §8. It is *an* axis on which the conditioning gap could have acted
  and did not; the epistemic-content readouts (Gate A/B on trained checkpoints) remain uncomputed.
- Anything with a positive control. There is no NTP-required-at-distance arm, for the structural reason
  above, so the curve is pinned at the chance end and the Bayes end but has no "content the model must
  keep" reference.
- That the m4 base's behaviour transfers to m2 or vice versa. The curve is m4; the arm comparison is m2.

Visible position-parity structure in `relLeaf` (`w=4, 8` above `w=3, 6`) was pooled over rather than
decomposed.

## Reproduction

```bash
cd experiments

# DGP self-checks (local, CPU, no Modal, no model)
python3 -m rhm.conditional_revision.synonym_retention.synonym_retention

# m4 base curve, ~25 min on an L4
modal run --detach -m rhm.conditional_revision.synonym_retention.synonym_retention::retention \
    --n-seq 6000 --n-twin 2000 --tag ret1

# the ten local_loss arms at m2, ~9 min
modal run --detach -m rhm.conditional_revision.synonym_retention.synonym_retention::sweep_arms \
    --tag arms1

# tables (local, CPU)
python3 -m rhm.conditional_revision.synonym_retention.aggregate base <ret1.json>
python3 -m rhm.conditional_revision.synonym_retention.aggregate arms <parts_arms1> <parts_ll1>
```

Outputs on the `rhm-scaling-data` volume (**`chromatic` workspace**):
`/data/v16_s2_L6_m4_distinct/conditional_revision/synonym_retention_ret1_base_m4_seed42.json` and
`/data/rhm_synonym_retention/parts_arms1/`.

## Gotchas worth not rediscovering

- **Freeze the perturbation across the sweep.** Changing the same bytes at every `w` makes the
  perturbation-size confound impossible rather than something to match on.
- **A retention readout needs a read position outside the perturbed span.** The incumbent cell's problem
  was not distance (the ratio is distance-robust) but that `str` is `leaf` plus a bump on the same span —
  nested magnitudes, neither arm isolating the latent.
- **Pin both ends of the probe.** Chance below and an exact Bayes ceiling above; otherwise a falling
  accuracy cannot be told from a rising task difficulty. Here the ceiling *rises* while accuracy falls.
- **Measure the NTP floor per cell.** `generate_rules_distinct` is not injective across features, so a
  closed span's realisation is not exactly redundant; report pooled and `TV == 0`-restricted.
- **Causal displacement and a linear probe can disagree at the floor**, and did (80–100σ vs exactly zero).
  Report both.
