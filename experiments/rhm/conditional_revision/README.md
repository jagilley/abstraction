# Conditional revision: is "how much the token moved my beliefs" separable from "how surprising the token was"?

**Up**: [../README.md](../README.md) · **Files**: [FILES.md](FILES.md) · **Design doc**: [SPEC.md](SPEC.md)
**Child**: [sculpt_slip/README.md](sculpt_slip/README.md) — the same question on a control substrate
**Idea**: [`ideas/revision_not_surprisal.md`](../../../ideas/revision_not_surprisal.md)
**Sibling / predecessor**: [`../endogenous_teacher/`](../endogenous_teacher/README.md) — this is the
measurement that cut needed before its interventions.
**Date**: 2026-08-07/08 · **Status**: Gate 0, the instrument self-check, Gate A and Gate B all run.
Gate C (matched FM pair + capacity invariance) not run. Single seed throughout.

## Goal

Token surprisal bundles two things: the token told us something about the world's latent structure
(**reducible**), and the token was one of several equivalent realisations of structure we already had
(**irreducible**). NTP trains on the sum. This cut asks whether the reducible part is separable, and
whether the model's own state tracks it.

RHM makes this exactly answerable rather than a correlation hunt: synonymy is a DGP primitive, and
belief revision `B_t = KL( P(z|x_{≤t+1}) ‖ P(z|x_{≤t}) )` is computable by exact belief propagation on
the known parse tree, with **zero compression by construction**.

Regime `v16 s2 L6 m4`, model `8L/8H/256D`, base at 12k steps — identical to
[`endogenous_teacher`](../endogenous_teacher/README.md) and [`RHM_LATENT_LOOP`](../RHM_LATENT_LOOP_README.md),
so their reference lines transfer (base d1 0.979, d3 0.836, root 0.088). No gate retrains the base; it
is trained once by Gate 0 and cached on the volume.

## The object — shift the target by one position, not six blocks

```
depth FM   (the incumbent):  FM( h0[≤t] )  →  h6[t]           residual: 0% aleatoric
temporal FM (this cut)    :  FM( h6[≤t] )  →  Δ_t = h6[t+1] − h6[t]
```

The depth FM's predictor and target are deterministic functions of the same input, so its residual can
only ever mean "I lacked capacity." `h6[t+1]` depends on `x_{t+1}`, which the corpus supplies and a
causal FM over `h6[≤t]` cannot hold — so the conditioning gap is **exactly one token** and the residual
has a nonzero aleatoric component for the first time in this program's activation-FM line. Everything
is open-loop against a frozen main model; the FM is `1L/8H/16d`, identical to `endogenous_teacher`'s.

## Gate 0 — the axis change does something, and in the predicted direction

Flat concatenated windows (the training distribution), against `endogenous_teacher`'s depth-FM Gate 0:

| FM | `corr(res, nll)` | `R²(res ~ nll)` | variance ⊥ `nll` | level-profile corr |
|---|---|---|---|---|
| `depth_cotrain` (= `endogenous_teacher`'s FM) | −0.337 | 0.114 | 88.6% | −0.786 |
| `depth_frozen` (protocol-matched control) | −0.328 | 0.107 | 89.3% | −0.773 |
| **`temporal`** | **+0.652** | **0.425** | **57.5%** | **+0.484** |
| `temporal_direct` (parametrisation control) | +0.638 | 0.407 | 59.3% | +0.457 |

Neither side of the pre-registered two-sided kill fired (`corr ≈ 0` would say the axis change did
nothing; `R² > 0.9` would say the residual is surprisal re-expressed in state space), and the
registered prediction — sign flips positive, level profile inverts to track `nll` — is met.

Two controls make the comparison attributable. `depth_cotrain` reproduces `endogenous_teacher`'s
published numbers to three decimals (−0.336 / 0.113 / 88.7%), so this is the same substrate; and
`depth_frozen`, trained against the frozen base for the same step budget, reads −0.328, so the flip is
the **axis change and not the training protocol**. The `temporal_direct` arm rules out the update-form
readout being load-bearing.

Per level on aligned sequences (mean relative residual; `nll` for reference):

| position completes | `nll` | depth residual | temporal residual |
|---|---|---|---|
| level 1 (near root) | 2.609 | 0.269 | 0.438 |
| level 3 | 2.616 | 0.236 | 0.441 |
| level 5 | 2.077 | 0.444 | 0.613 |
| level 6 (leaf-adjacent) | 0.802 | 0.436 | 0.317 |

The depth residual is *smallest* where `nll` peaks and largest where the text is nearly free; the
temporal residual reverses that. **Gate B below shows this is much less than it appears** — the flip is
real, but most of what flipped is surprisal.

## The oracle, and the instrument self-check

[`oracle.py`](oracle.py) computes `B` exactly by sum-product BP, together with
`H_irr = H(x_{t+1} | z_{≤D}, x_{≤t})` and the identity that ties them to surprisal:

```
E[ B_D ]  =  H(x_{t+1} | x_{≤t})  −  H(x_{t+1} | z_{≤D}, x_{≤t})
```

which is `I(x_{t+1}; z_{≤D} | x_{≤t})` written two ways. **Two corrections were forced during
construction, both worth keeping:**

- **The RHM latent graph is a hypertree, not a pairwise tree.** One rule emits a parent's whole
  `s`-tuple at once, so the children are dependent given the parent and a parent–child-edge
  factorisation is simply wrong — it reads `H = 1.733` where the truth is `ln 4 = 1.386` on the
  smallest test tree. The joint KL uses junction-tree cliques `{p} ∪ children(p)` with node
  separators. This was caught only by brute-force enumeration; keep that test.
- BP messages are normalised and accept internal-node evidence (needed to clamp `z_D` for `H_irr`).
  These live in `oracle.py` rather than as edits to [`../rhm_bayes_entropy.py`](../rhm_bayes_entropy.py),
  so prior results stay bit-identical; an equivalence test asserts the two still agree.

Verification: exact to **≤4.4e-16** against brute-force enumeration on tiny trees, **3.3e-15** against
`rhm_bayes_entropy.bp_conditional_entropies`, and the identity holds on the real regime to
**1.1e-05 – 7.9e-04** (Monte-Carlo error at n = 2500).

## What the DGP says, before any model is involved

| | `d6` (root) | `d5` | `d4` | `d3` | `d2` | `d1` |
|---|---|---|---|---|---|---|
| mean `B` (nats) | 0.039 | 0.061 | 0.105 | 0.191 | 0.361 | 0.699 |
| **fraction with `B` ≡ 0** | **0.500** | 0.498 | 0.485 | 0.451 | 0.383 | 0.291 |
| `R²(B ~ nll)` | **0.017** | 0.023 | 0.039 | 0.070 | 0.153 | 0.441 |
| `R²(B ~ exact surprisal)` | 0.037 | 0.048 | 0.071 | 0.115 | 0.230 | 0.605 |

Two facts this cut rests on, both model-independent:

1. **~49% of positions have `B` exactly zero** — the arriving token is one of the `m` synonymous
   realisations of structure the prefix already fixed, so the true posterior cannot move. Those
   positions still carry ~0.98 nats of surprisal. Gate B's two families are therefore a **DGP
   primitive, not a threshold we chose**.
2. **The reducible/irreducible split is nearly total at high abstraction and collapses near the
   leaves** — surprisal explains 1.7% of root-level revision and 44% of leaf-adjacent revision.

## The belief probe, and what it found about the model

`M` is the model's belief revision: `Σ_d KL(q_{t+1}^{a_d} ‖ q_t^{a_d})` over probe-decoded posteriors
at `x_{t+1}`'s own level-`d` ancestor, matched term-for-term by an oracle `B_chain`.

**A first version of this was wrong and the fix is a finding.** Summing over all 63 latent nodes
(matching `B_marg`) gave mean `M` = 6.07 against a true `B` of 0.70 — probe noise, not belief. The
probe read 0.068–0.182 per level against a Bayes ceiling of 0.371–0.614 computed exactly on the same
sequences. [`probe_diag.py`](probe_diag.py) settled which it was: the repo's own single-node probe
reproduces the reference lines on this checkpoint (**d1 0.984 vs 0.979, d3 0.837 vs 0.836, d6 0.085 vs
0.088**), so the activations are right; and no hyperparameter moves the all-node number — lr 1e-3 /
3e-3 / 1e-2, hidden 1024 / 2048, per-position standardisation and per-level heads all land at
0.065–0.183. It is not an optimisation failure.

Splitting nodes by their relation to the current position (unmasked probe, probe / exact Bayes ceiling):

| level | current (ancestor chain) | **past, already resolved** | future, unseen |
|---|---|---|---|
| d5 | 0.124 / 0.547 | **0.108 / 0.903** | 0.114 / 0.203 |
| d3 | 0.275 / 0.678 | **0.165 / 0.983** | 0.103 / 0.167 |
| d1 | 0.460 / 0.704 | **0.203 / 0.996** | 0.124 / 0.160 |

*(chance = 0.0625)*

Past constituents are the **easiest** thing in the table to know — every token in their span is
observed, so Bayes decodes them at 0.90–0.996. The residual stream at position `t` carries them at
0.20, i.e. 20% of a ceiling of ~1.0, against 65% of a much harder ceiling for the current chain.
Future nodes have low ceilings (0.15–0.21) and the probe tracks them at ~60–80%, so this is not an
artefact of averaging in unknowable nodes.

> **The model's per-position representation is close to a next-token-sufficient statistic: it holds the
> ancestor chain of the token it is predicting and does not summarise resolved structure forward.**

Scope: this is a statement about `h6[t]`, one position's vector. The information exists elsewhere in
the activation set — at the position where a constituent *was* the current chain node, the probe reads
it at 0.46 — and whether attention retrieves it on demand is untested. Restricting the probe's loss to
the ancestor mask accordingly roughly doubles its accuracy on the nodes `M` uses (d1 0.630 vs 0.461,
d2 0.569 vs 0.399), and the chain probe reaches **90% of its exact Bayes ceiling at d1** and 18% at the
root.

## Gate A — does the model's revision carry oracle structure surprisal does not explain?

Partial `R²`, on held-out aligned sequences, `post_block6` (`post_block7` agrees: best 0.148):

| | `d6` | `d5` | `d4` | `d3` | `d2` | `d1` |
|---|---|---|---|---|---|---|
| **partial `R²(M ~ B \| nll)`** | 0.008 | 0.013 | 0.022 | 0.039 | 0.102 | **0.150** |
| partial `R²(M ~ nll \| B)` | **0.082** | 0.097 | 0.099 | 0.106 | 0.057 | 0.005 |
| shuffled control | 0.006 | 0.008 | 0.008 | 0.003 | 0.005 | 0.006 |
| all-node `M` (superseded) | 0.006 | 0.009 | 0.011 | 0.014 | 0.021 | 0.038 |
| probe / Bayes ceiling | 18% | 23% | 35% | 64% | 82% | **90%** |

There is a clean crossover, and it tracks the probe's distance from its own ceiling. **Where the model
holds a belief, its revision is explained by the oracle and not by surprisal (0.150 vs 0.005 at d1);
where it does not, its "revision" is surprisal and nothing else (0.008 vs 0.082 at the root).** The
shuffled control sits at 0.003–0.008 throughout, so the d1/d2 numbers are 20–30× the null.

Formally this is UNDERPOWERED by a hair against SPEC's 0.15 proceed threshold (0.14997). The SPEC's
pre-registered reading of a weak Gate A applies and is the honest one: *this is a statement about the
base model's belief quality, not about revision.*

## Gate B — the constructed contrast (the load-bearing test)

Synonym positions (`B ≡ 0`) vs disambiguating positions (`B` above the median of the positive part),
scored by AUC under progressively stronger matching. **Position is a real confound**: in aligned
sequences a position fixes which hierarchy level the arriving token completes, and family membership is
strongly position-determined — `M_shuffled` (sequence identity permuted within position, so positional
structure intact and content destroyed) read **0.70** under surprisal matching alone. The primary
column matches on position **and** exact surprisal together.

Every column carries a guard that must read ~0.5: a matching variable held against itself, and
`M_shuffled` under position matching (exactly 0.5 by construction). All guards land at **0.489–0.507**.

Primary column — **position × exact-surprisal matched**:

| level | `nll` | exact surprisal | `r_temporal` | **`M`** | `M_allnode` | `M_shuffled` (guard) |
|---|---|---|---|---|---|---|
| d6 | 0.477 | 0.499 | 0.530 | 0.496 | 0.497 | 0.501 |
| d5 | 0.478 | 0.500 | 0.534 | 0.508 | 0.494 | 0.489 |
| d4 | 0.483 | 0.498 | 0.541 | 0.526 | 0.501 | 0.493 |
| **d3** | 0.480 | 0.501 | 0.554 | **0.614** | 0.509 | 0.498 |
| **d2** | 0.495 | 0.499 | 0.606 | **0.690** | 0.506 | 0.501 |
| d1 | n/a | n/a | n/a | n/a | n/a | n/a |

**`M` clears the pre-registered primary kill at d2 and d3.** With surprisal pinned at chance by
construction and the position-shuffled null at 0.501, the model's belief revision separates the two
families at 0.690 (d2) and 0.614 (d3). Nothing survives above d3 — consistent with Gate A and with the
probe reading 18–35% of ceiling there.

**d1 is untestable, not null.** Within a fixed position, exact surprisal separates the families at AUC
**1.000** at the leaf-adjacent level, so no matched pair exists. The registered prediction expected `M`
to fail there; the truth is that the contrast is degenerate there.

**Gate 0's residual is mostly surprisal.** `r_temporal`'s raw AUC of 0.64–0.75 falls to 0.50–0.52 once
the model's `nll` is matched, and to 0.53–0.61 under the full control. It carries some genuine revision
signal at the lower levels, but materially less than the belief readout (0.606 vs 0.690 at d2), and its
headline +0.652 correlation is dominated by the irreducible term. `delta_norm` (‖Δ_t‖ with no forward
model at all) is below chance everywhere, so the FM is doing something — just much less than Gate 0
suggested.

Selected weaker matchings, for the record (`M` / `r_temporal`):

| level | raw | `nll`-matched | exact-surprisal-matched | position-matched |
|---|---|---|---|---|
| d3 | 0.469 / 0.677 | 0.551 / 0.517 | 0.657 / 0.480 | 0.589 / 0.615 |
| d2 | 0.595 / 0.693 | 0.672 / 0.473 | 0.767 / 0.505 | 0.727 / 0.649 |

Note `r_temporal` reads ~0.50 under `nll`-matching but 0.53–0.61 under the full control — position was
masking it — which is why the guarded multi-column table is reported rather than a single number.

## What this establishes, and what it does not

**Establishes** (single seed, one regime):

- The conditioning gap is the operative variable for giving the residual an aleatoric component, and
  the axis change is not confounded with the training protocol.
- The reducible/irreducible split on RHM is an exact, model-independent fact, and ~49% of positions are
  purely irreducible while still carrying surprisal.
- The model's belief revision is a distinct signal from surprisal at the abstraction levels where the
  model has a belief, under matching that pins surprisal *and* position.
- The same underlying event read as a scalar residual norm carries much less than read as a KL in
  belief coordinates (0.606 vs 0.690 at d2; 0.005 vs 0.150 in Gate A at d1). This is a fourth instance
  of the repo's directional-not-scalar pattern, alongside
  [`EMOTION_INJECTION`](../../a2a_forward/EMOTION_INJECTION_README.md), the directional-vs-scalar belief
  node, and [`endogenous_teacher`](../endogenous_teacher/README.md) itself.

**Does not establish:**

- Anything about *using* the signal. Everything here is open-loop measurement against a frozen model,
  by design (SPEC design choice 4).
- Capacity invariance of the FM-side estimate — Gate C is unrun, so `Δ^rev` has not been shown to be
  revision rather than `ε₂ − ε₁`.
- That `M` is available without grounding. The content is endogenous but the *readout* is a probe
  trained on true latents, which is the fifth instance of the pattern SPEC pre-registered: *if revision
  works only when read against the oracle, that is the same finding again.* Whether the probe can be
  withdrawn after training is untested.
- Seed robustness, or transfer off this regime.

The binding constraint on the whole measurement turned out to be **belief depth, not the conditioning
gap**: the signal exists only where the model represents the latent, and this base model's root
recovery is 0.088.

## Child: sculpt_slip — the same question on a control substrate

[`sculpt_slip/`](sculpt_slip/README.md) ports the conditioning gap to RHM sculpting, where a slippery
actuator supplies an **exact binary aleatoric label** (we own the RNG), a behavioural readout, and a
noise level we set. **Step 1 is positive**: the slip is identifiable in the FM residual, but only
directionally — 0.70–0.75 AUC once ‖r‖ is matched, against 0.50 for the norm, with the norm getting
*worse* and the direction *better* as slip becomes more common. **Step 2 is a null**: a low-rank
precision operator recovers nothing its geometry control does not, and the prize it was chasing was
only ~0.03 (with `top1` exactly 0) because the noise is additive-and-uniform, which makes the
mean-predictor rank identically to the intended-outcome predictor and leaves estimation variance as
the whole cost of the gap.

## Reproduction

```bash
cd experiments

# oracle correctness self-test (local, CPU, no Modal, ~2 min)
python3 -m rhm.conditional_revision.oracle

# Gate 0 -- trains and caches the base; ~1.5 h on an L4 the first time
modal run --detach -m rhm.conditional_revision.conditional_revision::gate0 \
    --base-steps 12000 --fm-steps 12000 --tag gate0

# Gates A + B -- reuses the cached base and FMs; ~45 min first run, ~10 min after
modal run --detach -m rhm.conditional_revision.gates_ab::gates_ab --tag gateAB3

# probe diagnostic (node-class split, Bayes ceilings, hyperparameter sweep)
modal run --detach -m rhm.conditional_revision.probe_diag::probe_diag \
    --tag cls --n-train 2500 --n-test 1200
```

Results land on the `rhm-scaling-data` volume (**`chromatic` workspace**) under
`/data/v16_s2_L6_m4_distinct/conditional_revision/`: `results_gate0_seed42.json`,
`gates_ab_gateAB3_seed42.json`, `probe_diag_cls_seed42.json`, plus the cached
`base_8L8H256D_steps12000_seed42.pt` and `fms_frozen_steps12000_seed42.pt`.

## Gotchas worth not rediscovering

- **The latent graph is a hypertree.** A pairwise parent–child KL is wrong by construction; use the
  junction-tree cliques and keep the brute-force test.
- **Matching strata leak at atoms.** Exact BP surprisal piles mass on 0, ln 2, ln 3, ln 4…, so quantile
  edges repeat and `digitize` merges each atom with the continuous spread above it. The matching
  variable separated the families at AUC 0.999 *inside* a stratum meant to hold it fixed (self-matched
  0.80 instead of 0.50). Strata are atom-aware, and every block reports its own guard.
- **Encode crossed strata exactly.** `pos * K + bin` silently aliases across positions whenever the
  matching variable has more atoms than `K`.
- **Matching on the model's `nll` alone is not enough.** Exact surprisal still scored 0.64–0.78
  nll-matched, purely because model `nll` is a noisy proxy for it, so a signal that is merely a better
  surprisal estimate would clear that bar without being revision.
- **`M` must be read on the ancestor chain.** Summing over all latent nodes buries the signal in probe
  noise from the ~57 nodes the model does not represent.
