# Splicing the other end: a shared surface alphabet, and a transfer curve that cannot see it

**Status**: built and run — DGP certified in closed form in both splice directions, 3 seeds × 5
sharing depths on the block FM. **The cross-channel code the parent node found missing is now
present from the first shared table. The transfer curve does not move — but its own measured
noise floor is the same size as every cell it reports, so it is read here as uninformative
rather than as a null.** **Date**: 2026-08-03.
**Up**: [../README.md](../README.md) (partial_hetero) · **Node**: [../../../../README.md](../../../../README.md) (rhm) · **Files**: [FILES.md](FILES.md)
**Builds**: [`../README.md`](../README.md) §7 open item 1 — *"Splice the bottom tables instead of
the top — a one-line change to `share_rules_top`. This is the cheapest next step and possibly the
one that dissolves barrier (a)."*
**Idea doc**: [`ideas/meta_learning_under_metered_data.md`](../../../../../../ideas/meta_learning_under_metered_data.md)
§9 (the missing middle) and §6 (*"physics and fantasy football are written in one grammar, by one
species, about one world"*), which this splice direction is arguably the literal reading of.

---

## One-liner

`share_rules_bottom` shares the **surface** tables and leaves the **deep** composition
independently drawn — the mirror of the published splice. It does exactly what it was built to
do: a linear level-1 feature probe fit on tree blocks reads a shared-surface distractor at
**0.563 centered at one shared table**, against **0.098 at chance** for the published
shared-deep splice at the same depth, and the published geometry needed *all four* tables shared
to reach 0.689. Barrier (a) of [`../README.md`](../README.md) §4 — *"at sharing depths 1–3 there
is no shared vocabulary at all"* — is dissolved, and the claim that it might be intrinsic to the
construction can be retired. The transfer curve is flat (`+structA` slope **−0.00006 ± 0.00293**,
t = −0.04), **but the cross-mode identity at full sharing measures this instrument's own
run-to-run floor at ±0.0117 on exactly that quantity, and every `+structA` cell lies between
+0.004 and +0.014** — inside it. The curve is therefore not evidence about the geometry, in
either direction.

---

## 1. The construction

[`rhm_channels.share_rules_bottom`](../../../../rhm_channels.py) is the suffix splice.
`generate_rules_distinct` indexes tables from the root down, so where `share_rules_top` shares a
prefix (abstract composition, own rendering), sharing a suffix gives **the same alphabet composed
differently**:

| | shared | independent | the image |
|---|---|---|---|
| `share_top=k` | the top *k* tables — abstract composition | the surface rendering | same argument, alien script |
| `share_bottom=k` | the bottom *k* tables — the rendering alphabet | the deep composition | same language, unrelated topic |

`make_spec(..., share_mode="top"|"bottom")` selects which, and the two are mutually exclusive on
one channel. `share_mode="top"` is the default and every prior layout is bit-identical, which the
standing back-compat gate and the re-aggregated published sweep both confirm.

**At `k = depth` the two modes coincide by construction** — both hand structA the tree's entire
table set under an independently drawn root. That shared top rung is used throughout below as a
free consistency check and, more usefully, as a measured noise floor.

## 2. Certification (G1 / G2 / G2c), exact and closed-form

The parent node's per-level readout, `level_sibling_stats`, histograms the **marginal** sibling
law, which at level ℓ is a functional of tables 0..ℓ. That reads a top-splice exactly and a
bottom-splice **not at all**: sharing the last table leaves the level-1 marginal different anyway,
because the parent-feature distribution above it was drawn independently. Using it here would have
reported "no sharing" for a DGP that shares exactly what it claims to.

`level_cond_sibling_stats` (G2c) fixes that. P(child s-tuple | parent feature) at level ℓ is a
functional of `rules[ℓ]` **alone**, so it is identically zero on precisely the shared levels
whichever end was spliced. Computed in closed form from the tables — no sampling noise to read
past, so shared cells are exactly 0.000 rather than "0.010, near enough":

| k | `share_top` cond-TV by level (L1→L4) | `share_bottom` cond-TV by level |
|---|---|---|
| 0 | 1.000 1.000 1.000 1.000 | 1.000 1.000 1.000 1.000 |
| 1 | **0.000** 1.000 1.000 1.000 | 1.000 1.000 1.000 **0.000** |
| 2 | **0.000 0.000** 1.000 1.000 | 1.000 1.000 **0.000 0.000** |
| 3 | **0.000 0.000 0.000** 1.000 | 1.000 **0.000 0.000 0.000** |
| 4 | **0.000 0.000 0.000 0.000** | **0.000 0.000 0.000 0.000** |

The sweep asserts this shape per cell before any GPU work runs, so an off-by-one in either
direction fails locally and immediately.

**G1 — structural irrelevance survives at every cell, in both modes.** Each channel draws its own
root and every grammar readout slices the tree alone, so a distractor's content still cannot move
`d*`. `check_structural_irrelevance` passes at all ten cells.

**G2 — the surface, which is where the two modes are supposed to differ.** structA's legal-leaf-
tuple overlap with the tree:

| k | `share_top` | `share_bottom` |
|---|---|---|
| 0 | 0.286 (chance 0.25) | 0.286 |
| 1–3 | 0.286 | **1.000** |
| 4 | 1.000 | 1.000 |

Under the bottom splice the two channels are drawing leaf tuples from the same set from the first
shared table, which is precisely the property the parent node lacked.

## 3. G4 — the result this node was built for

A linear probe decodes a block's own level-1 feature from its latent, fit on **tree** blocks and
read on **structA** blocks (chance 0.125). The encoder is trained before any forward model exists,
so this is a pure property of (seed, sharing depth, splice direction) and is comparable across the
two sweeps. 3 seeds each:

| k | `share_top` raw | `share_top` centered | `share_bottom` raw | `share_bottom` centered |
|---|---|---|---|---|
| 0 | 0.091 ± 0.030 | 0.094 ± 0.037 | 0.090 ± 0.030 | 0.094 ± 0.037 |
| 1 | 0.094 ± 0.010 | 0.098 ± 0.040 | **0.467 ± 0.005** | **0.563 ± 0.045** |
| 2 | 0.091 ± 0.021 | 0.095 ± 0.025 | **0.531 ± 0.098** | **0.628 ± 0.095** |
| 3 | 0.112 ± 0.028 | 0.122 ± 0.016 | **0.586 ± 0.140** | **0.681 ± 0.145** |
| 4 | 0.586 ± 0.120 | 0.689 ± 0.128 | 0.586 ± 0.121 | 0.689 ± 0.128 |

1. **One shared table takes cross-channel decoding from chance to 0.563 centered.** The published
   splice sat at chance (0.094–0.122) at every intermediate depth and needed all four tables to
   reach 0.689.
2. **The k=4 row is an exactness check and it passes to five decimals** — max |diff| **3.1e-5**,
   per-seed raw 0.6349 / 0.6752 / 0.4493 in both modes. The two rows were produced by different
   arguments, different runs, and different `fm_arch`, and the top row reproduces the parent
   node's published 0.586 / 0.689 exactly. The construction is what it claims to be and the
   pipeline is deterministic.
3. **The encoder's cross-position code has a ceiling of ~0.69 centered**, reached at generatively
   identical channels and approached by k=3 under the bottom splice — against **0.91** for the
   same probe read *within* the tree. Sharing the rendering table alone buys 0.563 of that 0.689,
   so the alphabet is most of what alignment needs. This number is new and is a quantified
   statement of the parent node's barrier (b) rather than of barrier (a).

**What this settles.** [`../README.md`](../README.md) §4 hedged that the missing shared vocabulary
"is arguably what *superficially unrelated, deeply shared* **means** for a learner working at the
surface, not a bug in our readout" and "may be intrinsic to the construction". That hedge can be
retired: the shared alphabet is constructible, it costs one line, and it arrives at the first
shared table.

## 4. The transfer curve, and why it is not read as a null

Fresh block FMs under fixed channel allocations, matched total budget, matched state distribution
— the parent node's coverage-matched design unchanged. Value added to a fixed tree-data budget
(negative = the added data helped), 3 seeds:

| k | +structA (shared) | +structB (ctrl) | +noise (null) | 2× tree data |
|---|---|---|---|---|
| 0 | +0.0143 ± 0.0093 | +0.0106 ± 0.0060 | +0.0221 ± 0.0263 | −0.0528 ± 0.0164 |
| 1 | +0.0036 ± 0.0070 | +0.0171 ± 0.0027 | +0.0130 ± 0.0051 | −0.0538 ± 0.0234 |
| 2 | +0.0130 ± 0.0036 | +0.0105 ± 0.0105 | +0.0011 ± 0.0213 | −0.0540 ± 0.0120 |
| 3 | +0.0093 ± 0.0211 | +0.0057 ± 0.0120 | +0.0132 ± 0.0257 | −0.0511 ± 0.0093 |
| 4 | +0.0112 ± 0.0116 | +0.0018 ± 0.0149 | −0.0075 ± 0.0113 | −0.0620 ± 0.0275 |

Per-seed slope of `+structA` against sharing depth: **−0.00006 ± 0.00293, t = −0.04.**

### The floor, measured on an identical DGP

At `k=4` the two splice directions are the **same DGP with the same seeds**, so a top-vs-bottom
comparison there differs only by GPU nondeterminism across containers:

| quantity | `share_top` | `share_bottom` | difference |
|---|---|---|---|
| `structA_value` | +0.0117 ± 0.0089 | +0.0112 ± 0.0116 | **+0.0006 ± 0.0117** |
| `structB_value` | +0.0007 ± 0.0266 | +0.0018 ± 0.0149 | −0.0011 ± 0.0258 |
| `noise_value` | −0.0125 ± 0.0064 | −0.0075 ± 0.0113 | −0.0049 ± 0.0050 |
| `doubling_value` | −0.0571 ± 0.0311 | −0.0620 ± 0.0275 | +0.0050 ± 0.0068 |

**The run-to-run floor on `structA_value` is ±0.0117, and every `+structA` cell in the table above
lies between +0.004 and +0.014.** The entire reported signal is inside the floor. Independently,
the seed spread implies the smallest resolvable effect (2 SE) is ~24% of what a doubling of tree
data buys — this measurement could only see transfer worth a quarter of the real training data.

**So this table is recorded, not concluded from.** It does not show that a shared surface fails to
help a learner; it shows that if it helps by less than roughly a quarter of a data doubling, this
instrument cannot tell. The parent node's flat curve carries the same caveat, which is worth
propagating there.

### And the pathway is narrow by construction

Worth stating because it bears on how much the readout could ever have shown. The block FM carries
a per-block `action_embedding` and `block_position`; training on structA blocks never touches the
tree blocks' own parameters, and the tree gets identically many updates in `tree_structA` and
`tree_half`. The **only** channel through which sharing can help is the shared trunk becoming
better at latent dynamics in general — and `tree_structB` / `tree_noise` exist to subtract the
generic version of that same effect, leaving a second-order residual as the signal.

The invariance gap at generatively identical channels is unmoved and reproduces the parent:
`tree_structA − tree_only` = **+0.0732 ± 0.0262** (bottom) against **+0.0688 ± 0.0224** (top).

## 5. A design observation the two sweeps make together

The two splice directions are **complementary, and neither supplies both halves at intermediate
depth**:

- **top-splice** shares the abstract composition — the function a forward model would benefit from
  — but the two channels render level-1 features into different tokens, so there is no common code
  to express it in;
- **bottom-splice** shares the code, but at k=1–2 the composition above is independently drawn, so
  *which feature appears in which context* is a different function in the two channels.

They converge only at high k, and k=4 is bit-identical between modes. That leaves **k=3 as the
only cell where a shared abstraction and a shared vocabulary are both substantially present**, at
n=3 and inside the floor above.

This is an observation about the sweep's design rather than a result. The configuration it
suggests — **splice both ends**, sharing the bottom table for a common code *and* the top tables
for a common composition, leaving only the middle independent — is the first that would put both
halves in place at an intermediate sharing depth. Both functions exist; it is a few lines. Given
§4's floor, a different readout is the prerequisite, not more cells.

## 6. What this establishes — and what it does not

**Establishes:**

1. **A second sharing knob**, exact, graded, certified in closed form, mutually exclusive with the
   first, defaulting to the published behaviour. With `share_top` this makes the partially-
   heterogeneous DGP a two-dimensional construction.
2. **G2c**, a certification that reads either splice direction exactly and cheaply, replacing a
   readout that was only interpretable under one of them.
3. **A cross-channel feature code from the first shared table** (0.094 → 0.563 centered), which the
   published geometry never had below full sharing. The parent's barrier (a), and its hedge that
   the barrier might be intrinsic, are both resolved.
4. **The encoder's cross-position ceiling, ~0.69 centered against 0.91 within-channel**, at
   generatively identical channels — a quantified form of barrier (b).
5. **A measured run-to-run floor for the transfer instrument** (±0.0117 on `structA_value`),
   obtained free from the cross-mode identity.

**Does not establish:**

- **Nothing about whether shared surface structure helps a learner.** §4's cells are inside the
  measured floor. This is a statement about the instrument, not about the geometry, and it should
  not move priors in either direction.
- **Nothing that changes the parent's ladder findings.** No allocation ladder was run under this
  mode, deliberately — [`../README.md`](../README.md)'s own rule is that a readout which does not
  respond to the knob cannot let the loop respond to it, and here the readout's floor is the
  binding constraint.
- **Nothing about levels of the hierarchy directly.** That axis is [`../../level_moves/README.md`](../../level_moves/README.md),
  which is independent of this node.

## 7. Caveats

- **The `+structA` cells are inside the measured floor** (§4). Every statement about the transfer
  curve here is a statement about resolution.
- **3 seeds, one `fm_arch`.** The bottom sweep was run on `block` only. The parent ran three
  arches under `share_top` and none of them moved the curve, so a fourth was not judged worth the
  GPU before the readout question resolves — but it means `shared_marker` / `channel_local` under
  the bottom splice are unmeasured.
- **G4 probes feature *identity*, not dynamics.** It is a necessary condition for transfer, not a
  sufficient one — the parent's own caveat, and it is exactly what §3 and §4 together illustrate.
- **The ~0.69 ceiling is interpreted, not mechanistically pinned.** Centering removes an additive
  offset but not a rotation or rescaling, and no further decomposition was attempted.
- **The G4 comparison crosses `fm_arch`** (published alignment lives in the `gcl_` runs, ours in
  `block` runs). This is sound because the encoder is trained before the FM is constructed, and
  the k=4 identity check confirms it empirically to five decimals — but it is a cross-run
  comparison, not a within-run one.

## 8. Open items

1. **A transfer readout with a floor below the effect size.** Everything in §4 is gated on this.
   Until one exists, more cells and more seeds buy little.
2. **The both-ends splice** (§5) — the first geometry where a shared abstraction is expressible in
   a shared vocabulary at intermediate depth. Cheap to build, worth building *after* item 1.
3. **Propagate the floor caveat to [`../README.md`](../README.md) §3**, whose flat curve was
   measured with the same instrument and carries the same resolution limit.
4. **Why does the encoder top out at 0.69?** It is the same 0.586/0.689 cell the parent attributed
   barrier (b) to; this node shows the value is not raised by supplying a shared alphabet.

## Reproduce

```bash
cd experiments/            # MODAL_PROFILE=chromatic
python3 -c "from rhm.verify_backcompat import verify; verify()"          # the standing gate

# G1 / G2 / G2c -- both splice directions, locally, no GPU
python3 -c "
from rhm.rhm_channels import make_layout, check_structural_irrelevance
from rhm.directed_sculpting.full_loop.channel_env import make_spec
import sys; sys.path.insert(0, 'rhm/directed_sculpting/full_loop/partial_hetero')
from geometry_check import surface_stats, level_cond_sibling_stats
for mode in ('top', 'bottom'):
    for k in range(5):
        L = make_layout(8, 2, make_spec(tree_depth=4, struct_depths=[4, 2], struct_ms=[2, 4],
                                        noise_blocks=[1, 1], struct_shares=[k, 0], share_mode=mode))
        irr = check_structural_irrelevance(L, n=4096)
        print(mode, k, irr['dp_identical'], surface_stats(L)['structA']['overlap_with_tree'],
              level_cond_sibling_stats(L)['structA']['levels_exactly_shared'])"

# G3 + G4 -- the shared-surface sweep, one GPU probe per depth, spawned in parallel
for s in 1 2 3; do
  modal run --detach rhm/directed_sculpting/full_loop/partial_hetero/geometry_check.py::geometry_check \
      --tag sb_s$s --seed $s --shares 0,1,2,3,4 --share-mode bottom --fm-arch block
done

for s in 1 2 3; do
  modal volume get --force rhm-scaling-data "directed_sculpting/partial_hetero/geometry_sb_s$s" \
      rhm/directed_sculpting/full_loop/partial_hetero/figures/; done
python3 rhm/directed_sculpting/full_loop/partial_hetero/aggregate.py   # groups by share_mode,
                                                                      # prints the cross-mode check
```

`--share-mode top` (the default) reproduces the published sweep exactly — verified by
re-aggregating `geometry_g_s{1,2,3}` unchanged after the knob was added.
