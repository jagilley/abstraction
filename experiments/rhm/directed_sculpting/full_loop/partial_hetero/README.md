# Cut-3, the partially-heterogeneous DGP: the knob works, the sweep does not measure it

**Status**: built and run — DGP certified, three readouts, 21 GPU runs. **The predicted curve did not
appear, and the honest reading is that this apparatus cannot test the prediction as posed.** The
sharing knob is exact and reusable; the readout is the thing that failed, in two separable ways, and
one of them may be intrinsic to what "superficially unrelated, deeply shared" means for a
block-level learner. **Date**: 2026-07-31.
**Up**: [../README.md](../README.md) (full_loop) · **Node**: [../../../README.md](../../../README.md) (rhm) · **Files**: [FILES.md](FILES.md)
**Idea doc**: [`ideas/meta_learning_under_metered_data.md`](../../../../../ideas/meta_learning_under_metered_data.md)
§5 (the two geometries) and §9 (cut-3, the named next step this builds).
**Also builds on**: [`rhm/specialization/README.md`](../../../specialization/README.md) — cut-3 was
specced there in 2026-07-16/18 and never built.

---

## One-liner

The sharing-depth knob is real, exact and graded — certified three ways at the DGP level — and the
**uniform→oracle prize does not move across it** (0.0713 → 0.0673, slope −0.0019 ± 0.0024, t = −1.43,
against a *measured* instrument noise floor of ~0.004). Three successive readouts, each removing a
suspected barrier, all return flat. The diagnosis is not the geometry: at full sharing, where the two
channels are **statistically identical**, the arm that spends half its budget on each lands on the
half-budget arm rather than the full-budget one — **half the data bought nothing from a generatively
equivalent source**. A linear probe localises this to a missing *shared vocabulary*: the block FM
works in level-1 features, and at every intermediate sharing depth the level-1 table is precisely
the one **not** shared, so cross-channel feature decoding sits at chance (0.09–0.12 vs 0.13). The
shared abstraction lives above the level at which the two channels have any common code. **That is
plausibly a structural property of the construction rather than a fixable instrument bug**, and it
is the reason this node's next step is a DGP question, not an architecture question.

---

## 1. What this was built to test

[§5 of the idea doc](../../../../../ideas/meta_learning_under_metered_data.md) observes that the repo
has only ever measured the two **degenerate ends** of the shared-structure axis, which answer *"should
I restrict training to what's relevant?"* with opposite signs:

| geometry | relationship | verdict |
|---|---|---|
| [specialization](../../../specialization/README.md) — one shared ruleset, A and B are subtrees | *A's depth **is** B's depth* | **inert** — breadth restriction Δ ≤ 0.013; broad beats narrow on A's own domain |
| [full_loop](../README.md) — independent rule tables per channel | nothing shared | **pays 0.063** in tree FM error (uniform → oracle) |

§9 names the missing middle as the realism condition — *superficially unrelated, deeply shared* — and
predicts a curve: **allocation payoff falls monotonically with sharing depth**, hitting full_loop's
0.063 at zero sharing and specialization's inert result at full sharing.

## 2. The construction, and what certifies it

[`rhm_channels.share_rules_top`](../../../rhm_channels.py) implements §9's spec.
`generate_rules_distinct` indexes tables from the root down, so splicing a **prefix** shares the
abstract levels and leaves the surface independently drawn. `structA` is depth- and m-matched to the
tree (so level *i* is the same scale in both) and takes the tree's top `share_top` tables:

| `share_top` | what structA is | which known geometry |
|---|---|---|
| 0 | an independent grammar of matched shape | **full_loop's** |
| 1 – 3 | shared abstract composition, own surface rendering | **the unbuilt middle** |
| 4 | the tree's grammar with an independently drawn root | **specialization's** |

`structB` stays shallow and is **never** shared — the deeply-unrelated control. Noise channels are
unchanged. Tables are copied, never aliased, so two fully-shared channels still carry independent
drift.

**Three certifications, all passing, on three separate rule draws:**

1. **Structural irrelevance survives (G1).** Each channel draws its own root and every grammar
   readout slices the tree alone, so a distractor's *content* cannot move `d*` at any sharing depth.
   Confirmed at the DGP level and again inside the loop: ground-truth best Δ`d*` came back **exactly
   +0.000** for every non-tree channel at all five depths, against the tree's +1.68 to +1.73. Sharing
   changes what a distractor's data is *worth to the learner*, never what its content can do to the
   task.
2. **The surface stays distinct (G2).** structA's legal-leaf-tuple overlap with the tree sits at
   chance (0.286 vs 0.250 chance) for `share_top` 0–3 and reaches 1.000 only at 4.
3. **The sharing is exact and graded, in the data's own currency (G2b).** `level_sibling_stats`
   samples derivations forward and histograms the s-tuple one rule produces at each level. The
   sibling law at level ℓ is a functional of tables 0..ℓ−1 alone, so under `share_top=k` it must be
   identical to the tree's for ℓ ≤ k:

   | `share_top` | structA sibling TV vs tree, level 1 → 4 |
   |---|---|
   | 0 | 0.815  0.810  0.883  0.828 |
   | 1 | **0.010**  0.733  0.874  0.822 |
   | 2 | **0.010  0.011**  0.874  0.850 |
   | 3 | **0.010  0.011  0.008**  0.814 |
   | 4 | **0.010  0.011  0.008  0.007** |

   Shared levels at sampling noise, unshared at 0.73–0.94, never-shared `structB` flat at 0.78–0.91.
   **This table is what makes the null below interpretable**: the DGP unambiguously has the structure;
   the question is only whether a learner can reach it.

### The noise floor, measured rather than assumed

At `share_top=0` there are no shared channels, so the added `oracle_shared` rung reduces to *exactly*
`oracle` — same drive, same 0.960 tree allocation, same seeds. The two arms still differ by
**−0.0033 ± 0.0039** in tree FM error, from GPU nondeterminism alone. **Any cross-depth movement
below ~0.004 is not a measurement.** This was free, and it is what lets the flat curve below be
called flat instead of "small".

## 3. The result: flat, at three levels of instrument surgery

### E1-PH — the allocation ladder at each sharing depth (3 seeds × 5 depths, 6 policies)

| share_top | uniform | reducible_only | visits_only | value_red | oracle | oracle_shared |
|---|---|---|---|---|---|---|
| 0 | 0.7985 | 0.7843 | 0.7452 | 0.7383 | 0.7272 | 0.7305 |
| 1 | 0.7729 | 0.7569 | 0.7141 | 0.7135 | 0.7013 | 0.7350 |
| 2 | 0.7833 | 0.7697 | 0.7197 | 0.7214 | 0.7146 | 0.7422 |
| 3 | 0.7622 | 0.7537 | 0.7119 | 0.7088 | 0.7020 | 0.7345 |
| 4 | 0.7899 | 0.7850 | 0.7343 | 0.7322 | 0.7226 | 0.7599 |

The prizes, and their slopes against sharing depth:

| prize | k=0 | k=1 | k=2 | k=3 | k=4 | slope / level | t |
|---|---|---|---|---|---|---|---|
| uniform→oracle | +0.0713 | +0.0716 | +0.0686 | +0.0602 | +0.0673 | −0.0019 ± 0.0024 | −1.43 |
| uniform→visits_only | +0.0533 | +0.0588 | +0.0636 | +0.0503 | +0.0557 | −0.0004 ± 0.0002 | −4.22 |
| oracle→oracle_shared | −0.0033 | −0.0337 | −0.0276 | −0.0325 | −0.0373 | −0.0067 ± 0.0024 | −4.92 |

1. **§9's predicted curve is absent.** The uniform→oracle prize moves by 0.004 across the entire
   sweep — exactly the measured noise floor — and never approaches specialization's inert result,
   including at `share_top=4` where structA's tables are bit-identical to the tree's.
2. **The transfer-aware oracle is consistently *worse*.** `oracle_shared` spreads over the tree and
   every channel sharing rules with it, and loses ~0.03 at every depth ≥1 without improving as
   sharing rises. Spending half the budget off-tree costs, whether the off-tree channel shares one
   level or all four. (`uniform→visits_only`'s slope has t = −4.22 but a magnitude of 0.0015 over the
   whole sweep — statistically resolvable, scientifically flat.)
3. **The endogenous relevance tap is unmoved, as designed.** `visits` is trained on tree-slice task
   success, and it allocates 0.73 / 0.72 / 0.76 / 0.79 / 0.77 to the tree across k=0→4. It tracks
   `d*`-relevance, which stays exactly 0 off-tree by construction. This part behaved exactly as
   predicted — it just never became *wrong* to do so, because off-channel data never became valuable.

### G3 — the transfer curve, static and loop-free, across three readouts

Fresh block FMs under fixed channel allocations, matched total budget, matched state distribution.
The headline design holds **tree-block transitions exactly fixed** and varies only what fills the
rest of the budget, so `tree_structA − tree_half` is what shared-grammar data *adds*, with
`tree_structB` / `tree_noise` carrying the generic more-optimizer-steps effect without any sharing,
and `tree_only − tree_half` giving the exchange rate in doublings of task data.

Value added to a fixed tree-data budget (negative = helped):

| readout | +structA slope / level | +structB (ctrl) | +noise (null) | verdict |
|---|---|---|---|---|
| `block` (the ladder's own) | +0.0035 ± 0.0025 (t=+2.41) | −0.0015 ± 0.0036 | −0.0021 ± 0.0017 | flat |
| `shared_marker` | +0.0028 ± 0.0059 (t=+0.81) | −0.0018 ± 0.0017 | −0.0007 ± 0.0014 | flat |
| `channel_local` | +0.0042 ± 0.0016 (t=+4.54) | +0.0026 ± 0.0035 | +0.0024 ± 0.0032 | flat |

Every `+structA` slope is at or barely above its own control slopes, and all three are *positive* —
if anything shared data becomes slightly **less** useful as sharing rises. Meanwhile doubling the
task data reliably buys −0.05 to −0.08 in every readout, so the measurement is not blind; it sees
data value of exactly the relevant size and sees none from sharing.

**Two instrument surgeries, neither of which rescued it.** `shared_marker` replaces the FM's
per-block acted-marker with one learned vector (the only parameters private to a block).
`channel_local` removes absolute-block parameters entirely — attention masked to within-channel,
within-channel offsets, one shared marker, plus per-block latent centering to strip the frozen
encoder's additive positional component. `channel_local` did improve the readout in absolute terms
(own-channel error 0.635 → 0.621; a distractor-trained FM on the task 0.998 → 0.929) without making
the knob visible.

## 4. Why: the invariance test, and two separable barriers

### The test that makes this diagnosable

At `share_top=4`, static world, depth-matched channels, identical rule tables, the tree and structA
are **statistically identical**. A readout that treats "the same grammar at a different position" as
the same problem *must* therefore score `tree_structA` at `tree_only`, since half its budget is drawn
from an equivalent source. It does not — under every readout it lands on `tree_half`:

| readout | `tree_only` | `tree_half` | `tree_structA` | gap vs `tree_only` |
|---|---|---|---|---|
| `block` | 0.6512 | 0.7082 | 0.7200 | **+0.0688** |
| `shared_marker` | 0.6529 | 0.7044 | 0.7051 | **+0.0522** |
| `channel_local` | 0.6207 | 0.6765 | 0.6827 | **+0.0620** |

**Half the budget contributed nothing, from a generatively identical source.** This single number is
the gate: until it is ~0, a sharing-depth sweep cannot measure sharing. Removing every
absolute-position parameter from the forward model barely moved it.

### G4 — where the barrier actually is

A linear probe decodes a block's own level-1 feature from its latent, fit on tree blocks and read on
structA blocks (chance 0.125; 3 seeds):

| share_top | tree (raw) | structA (raw) | tree (centered) | structA (centered) |
|---|---|---|---|---|
| 0 | 0.911 | 0.091 | 0.901 | 0.094 |
| 1 | 0.921 | 0.094 | 0.911 | 0.098 |
| 2 | 0.928 | 0.091 | 0.919 | 0.095 |
| 3 | 0.928 | 0.112 | 0.915 | 0.122 |
| **4** | 0.922 | **0.586** | 0.908 | **0.689** |

This separates two barriers that had been read as one:

**(a) At sharing depths 1–3 there is no shared vocabulary at all, and this looks structural.** The
block FM's unit of work is the level-1 node, and the level-1 table is *precisely the one not shared*
at every intermediate depth. Cross-channel feature decoding is at chance (0.09–0.12) — the two
channels have no common code in which the shared deep structure could be expressed. **This is
arguably what "superficially unrelated, deeply shared" *means* for a learner working at the surface,
not a bug in our readout.** The abstraction is real (§2's TV table proves it) and lives above the
level at which the learner and the two channels share any alphabet.

**(b) At sharing depth 4 a shared vocabulary exists and the readout still wastes the budget.** 0.689
centered against 0.908 in-channel is well above chance and far below parity, and centering (which
removes an additive offset but not a rotation or rescaling) recovers only part of it. Decoding a
discrete feature at 0.69 is evidently not enough to regress a precise latent delta — hence the
+0.062 residual.

> **A correction to an earlier read.** A `--quick` smoke reported this cell at 0.714 raw / 0.867
> centered and was taken as a green light for the forward-model fix. The full runs give **0.586 /
> 0.689**. The encoder is materially less aligned than the smoke suggested, which is consistent with
> `channel_local` not closing the gap, and it is why the remaining barrier is attributed to the
> encoder rather than the forward model. The smoke was one seed at 100 probe steps on 2048 samples;
> it should not have been read as a result.

## 5. What this establishes — and what it does not

**Establishes:**

1. **A sharing-depth knob for RHM channels**, exact, graded, certified three ways, and reusable. This
   is the durable output.
2. **§9's predicted curve does not appear in this apparatus**, at a measured noise floor, across
   three readouts.
3. **A concrete, quantified failure of abstraction reuse**: +0.062 wasted budget at generatively
   identical channels, with the right answer known exactly.
4. **The barrier is a missing shared vocabulary, not a missing shared abstraction** — and at depths
   1–3 that may be intrinsic to the construction rather than to the instrument.

**Does not establish:**

- **Nothing about whether §9's prediction is true.** A flat curve from an apparatus that cannot see
  sharing is not evidence against sharing mattering. This is a null about the measurement.
- **Nothing that overturns §5's two-geometries table** — but it does surface a **confound in it**
  worth carrying regardless of what happens next. Those two rows differ in *learner architecture* as
  well as in data geometry: specialization is one model over one tree with root-restricted training,
  where positions are on equal footing; full_loop reads channels at fixed distinct positions with a
  position-indexed FM. Holding the learner fixed and moving only the geometry — which is what this
  sweep does — gives a flat answer, including at the endpoint that should have reproduced
  specialization's world. So "the answer is set by the geometry, not by the learner" has not been
  cleanly measured. **This is the one claim here that does not depend on the readout question
  resolving.**
- **Nothing about the ladder's other findings.** The relevance tap, the ladder ordering and the
  metering ratio all reproduce on the new geometry; nothing here disturbs [`../README.md`](../README.md) §3.

## 6. Caveats

- **The `share_top=0` cell is not a re-run of `ladder_fix_s*`.** structA is 8 blocks here against the
  published geometry's 2 (depth-matched, as the splice requires), so every cross-depth comparison is
  *within* this sweep. The published 0.063 is quoted for orientation, not as a matched control.
- **Absolute error levels move with sharing depth** (`tree_half` 0.714 → 0.677 as k rises), because
  the input sequences change. All `_value` quantities difference arms *within* a depth, so this does
  not touch the curve — but it does mean the raw tables are not comparable across rows.
- **Three seeds, and the ladder sweep varies training seed only** (rule tables fixed at
  `rule_seed_offset=0`, matching the published ladder). The geometry sweep varies the rule draw too.
- **`channel_local`'s within-channel attention mask is a behavioural change**, not only a
  parameterisation one: the FM can no longer represent cross-channel ripple. That is correct on the
  merits here (the channels are generatively independent) but it is not a pure ablation.
- **The 15-run `channel_local` ladder sweep was deliberately not launched.** A readout that does not
  respond to the knob cannot let the loop respond to it. `--fm-arch channel_local` is smoke-verified
  through the full loop (six policies, both beams, the tap that deep-copies the FM), so it is one
  command if the readout question resolves.
- **G4 probes feature *identity*, not dynamics.** It is a necessary condition for transfer, not a
  sufficient one, and the 0.689-vs-0.908 gap is interpreted rather than mechanistically pinned.

## 7. Open items — in the order they seem worth doing

1. ~~**Splice the *bottom* tables instead of the top.**~~ **Done — see
   [`shared_surface/`](shared_surface/README.md), and it dissolves barrier (a).** A linear feature
   probe fit on tree blocks reads a shared-*surface* distractor at **0.563 centered with one shared
   table**, against 0.098 (chance) for this sweep's shared-*deep* splice at the same depth; this
   geometry needed all four tables to reach 0.689. §4's hedge that the missing vocabulary "may be
   intrinsic to the construction" can be retired — it is constructible and costs one line. The
   transfer curve there is flat too, **but that node measures this instrument's run-to-run floor at
   ±0.0117 on `structA_value` and every cell lies inside it**, which is a caveat this section's
   own flat curve inherits (see item 6).
2. **Drive the `share_top=4` invariance gap to ~0, and treat it as the gate for anything else.**
   Whatever the DGP question, no sharing sweep is readable until a generatively-identical channel is
   worth what the task channel is worth. Report it as a standing number.
3. **Decide whether the encoder rewrite is worth it.** Per-channel encoding with tied weights would
   make strips genuinely interchangeable, but it is a *new instrument*, not a patch — the state
   space, the value head and both planners read the current controller, and the node's calibration is
   built on it. **Item 1 did not make this unnecessary**: supplying a shared alphabet lifts the
   feature probe to 0.681 but the encoder still tops out at ~0.69 centered against 0.91
   within-channel, so the ceiling is the encoder's and not the vocabulary's.
4. **Consider writing the +0.062 gap up as its own result.** "This learner cannot recognise the same
   structure in a different place, measured where the right answer is known exactly" may be a more
   interesting finding than the curve we set out to measure, and it connects to the ratchet arc's
   abstraction-reuse questions rather than to curation.
5. **Record §5's learner/geometry confound in the idea doc** (§5 above). It holds independently of
   everything else here.
6. **A transfer readout with a floor below the effect size — now the binding constraint.**
   [`shared_surface/`](shared_surface/README.md) §4 measures this instrument's run-to-run floor on
   an *identical* DGP at **±0.0117** on `structA_value`, and the seed spread implies the smallest
   resolvable effect is ~24–35% of what a doubling of tree data buys. §3's flat curve here is
   inside that same floor. Until a readout with better resolution exists, neither sweep's flatness
   is evidence about the geometry, and more cells or seeds buy little.

> **A sibling instrument failure, and the general lesson (2026-08-03).**
> [`../level_moves/`](../level_moves/README.md) §9 hit the same class of problem from the other
> direction and its resolution is worth carrying here. That node's confound-correction was a
> **regression** — ΔV regressed on true Δ`d*`, differenced per level — and it never resolved across
> four attempts (per-seed +0.739 / −0.430 / −0.020). Two **structural** corrections resolved
> immediately: a null *channel* whose ground-truth value is certified 0.000, and a *paired* control
> matched by construction rather than by covariate adjustment. Stated generally, and it applies to
> this node's §7 item 2 gate as much as to that one: **when a readout's components are not
> independent, a matched control does what a matched regression cannot** — which is the same lesson
> §4b of [`heterogeneous_graders`](../../../../../ideas/heterogeneous_graders.md) records about the
> withdrawn `R_comp`/`R_res` partition, now with a second instance.
>
> Also worth recording as a reuse: **this node's depth-matched `structA` is what made that null
> possible.** `--struct-depths 4,2` was built here so the sharing splice could be depth-matched, and
> it doubles as a full depth-4 channel with Δ`d*` certified exactly 0.000 at every level — the only
> thing in the repo that can separate "the value rates deep moves higher" from "deep moves perturb
> the latent more".

## 8. Children

| Child | What |
|---|---|
| [`shared_surface/`](shared_surface/README.md) | **The mirror splice, and §7 item 1 closed.** `share_rules_bottom` shares the *surface* tables and leaves the deep composition independently drawn — *same language, unrelated topic*, against this sweep's *same argument, alien script*. The cross-channel code §4 found missing arrives at the **first** shared table (feature probe 0.094 → **0.563** centered, where this geometry needed all four tables to reach 0.689), so barrier (a) is dissolved and its "may be intrinsic to the construction" hedge is retired. Two things come back that this node should carry: the encoder's cross-position ceiling is **~0.69 centered against 0.91 within-channel** *even with a shared alphabet*, which is barrier (b) quantified and is why §7 item 3 stands; and the transfer curve's **measured run-to-run floor is ±0.0117**, larger than every cell either sweep reports. Also adds **G2c**, a closed-form conditional-sibling certification that reads either splice direction exactly, and a k=4 cross-mode identity check that reproduces this node's published 0.586 / 0.689 to five decimals |

## Reproduce

```bash
cd experiments/            # MODAL_PROFILE=chromatic
python3 -c "from rhm.verify_backcompat import verify; verify()"          # the standing gate

# G1/G2/G2b locally, no GPU
python3 -c "
from rhm.rhm_channels import make_layout, check_structural_irrelevance
from rhm.directed_sculpting.full_loop.channel_env import make_spec
import sys; sys.path.insert(0, 'rhm/directed_sculpting/full_loop/partial_hetero')
from geometry_check import surface_stats, level_sibling_stats
for k in range(5):
    L = make_layout(8, 2, make_spec(tree_depth=4, struct_depths=[4, 2], struct_ms=[2, 4],
                                    noise_blocks=[1, 1], struct_shares=[k, 0]))
    print(k, check_structural_irrelevance(L, n=2048), surface_stats(L)['structA'],
          level_sibling_stats(L)['structA'])"

# G3 + G4 -- transfer curve and encoder alignment, one GPU probe per depth, spawned in parallel.
# --fm-arch: block (the ladder's own) | shared_marker | channel_local (position-invariant)
for s in 1 2 3; do
  for a in block shared_marker channel_local; do
    modal run --detach rhm/directed_sculpting/full_loop/partial_hetero/geometry_check.py::geometry_check \
        --tag g_s$s --seed $s --shares 0,1,2,3,4 --fm-arch $a     # tags used: g_ / gsm_ / gcl_
  done
done

# E1-PH -- the ladder at each sharing depth
for s in 1 2 3; do for k in 0 1 2 3 4; do
  modal run --detach rhm/directed_sculpting/full_loop/ladder.py::ladder \
      --tag ph_k${k}_s$s --seed $s --rounds 12 --n-eval 1024 \
      --struct-depths 4,2 --struct-shares $k,0 \
      --policies uniform,reducible_only,visits_only,value_red,oracle,oracle_shared
done; done

# mirror + read both curves (grouped by fm_arch)
for d in geometry_g_s1 geometry_gsm_s1 geometry_gcl_s1; do
  modal volume get --force rhm-scaling-data "directed_sculpting/partial_hetero/$d" \
      rhm/directed_sculpting/full_loop/partial_hetero/figures/; done
python3 rhm/directed_sculpting/full_loop/partial_hetero/aggregate.py
```

`--struct-shares 0,0` with `--struct-depths 2,2` and the default `--fm-arch block` reproduces the
published `ladder_fix_s*` geometry exactly, so every prior result on this node stays reachable from
the same script.
