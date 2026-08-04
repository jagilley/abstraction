# A level-indexed action space: what the value system thinks a level of the hierarchy is worth

**Status**: built, run, then **corrected** — action space certified three ways, damage model
certified in closed form, 3 seeds throughout.
**Current headline (§9–§12): the value is sensitive to the hierarchy's *compositional* structure,
not to depth as such. At matched span it prefers one legal abstract commitment over independent
per-block guesses (0.557 ± 0.018 at L2, 0.618 ± 0.052 at L3, against an exact oracle's 0.610 /
0.642), and under hierarchical damage that preference is graded by whether a commitment at that
level can actually reach the error (L2 slope −0.033, t = −6.81 against the oracle's −0.025,
t = −4.14). It under-weights abstraction systematically, more so as errors deepen.**
**Superseded headline (§3–§5, kept as the record): *"the value prefers higher levels, monotonically,
under-shooting the DP ~40% at the root"* — the magnitude readout was span and the ranking readout was
one seed of three.** **Date**: 2026-08-03.
**Up**: [../README.md](../README.md) (full_loop) · **Node**: [../../../README.md](../../../README.md) (rhm) · **Files**: [FILES.md](FILES.md)
**Builds**: [`../README.md`](../README.md) §3 and open item 2 — *"§12 names satiety as the climbing
mechanism, but climbing needs the allocation space to be ordered by level. Ours is not."*
**Idea doc**: [`ideas/adaptive_core_and_hierarchy_climb.md`](../../../../../ideas/adaptive_core_and_hierarchy_climb.md)
§12 (satiety-gated *"once level ℓ stops paying, recruit ℓ+1"*), which has never had an axis to
recruit along.

> ## ⚠ Corrected and extended, 2026-08-03 — read §9–§12 before §3–§4
>
> §4's headline (*"the value's level preference is real, monotone, correctly ordered, calibrated
> ~40% low"*) **does not survive its own confound**, and three structural corrections replace it.
> The sections below are kept intact as the record of what was measured; the corrections are
> §9–§12 and they change what it means.
>
> - **§9, the span null at every level.** A level-ℓ move rewrites `s**ℓ` tokens and so displaces
>   `z.mean(dim=1)` more whether or not it buys anything. Run against a **depth-4 distractor** with
>   Δ`d*` certified 0.000 at every level, |ΔV| rises **2.79×** from L1 to L4 where the tree rises
>   **2.56×**. The raw profile's slope (+0.236, t = +4.41) is the branching factor.
> - **§9, the per-node top-1 profile too.** Per-seed level-4 top-1 share is **0.400 / 0.041 /
>   0.119**, i.e. per-node-relative-to-L1 of **16.1× / 0.86× / 2.95×** — non-monotone in two of
>   three seeds, and the 5.0× is one seed carrying a mean whose sd equals it.
> - **§10, what replaces it.** A **matched-span paired control**: each committed level-ℓ move is
>   paired with a *lazy twin* at the same node that rewrites identical tokens without committing.
>   The confound cancels inside the pair by construction. The value does prefer the commitment —
>   **0.557 ± 0.018 at L2, 0.618 ± 0.052 at L3** — and it is task-referenced, not grammaticality
>   (4.4% of the premium survives in channels where legality buys Δ`d*` = 0). But the *slope*
>   against level does **not** resolve (+0.060 ± 0.108, t = +0.97).
> - **§11, the finding this cut now carries.** The published damage is off-grammar in **29%** of
>   tree blocks, so it is repairable one block at a time and abstraction never had to matter.
>   Under **hierarchical damage** (a level-k subtree swapped for a legal derivation of a feature it
>   cannot produce; 100% on-grammar, matched `d*`), the value's abstraction premium **tracks how
>   deep the error is** — L2 slope −0.033 (t = −6.81) against the oracle's −0.025 (t = −4.14), 3/3
>   seeds — and value–oracle rank agreement roughly **doubles**, +0.290 → +0.583 (t = +8.14).
> - **§12, why the root cell never resolved.** A root move masks every tree token, so its Δ`d*`
>   premium is **bit-identical across all three damage levels**. L4 is structurally blind to the
>   DGP and needs a partial-mask move, not more seeds.
>
> **The corrected claim.** The value is sensitive to the hierarchy's *compositional* structure, not
> to *depth per se*: it prefers an abstract commitment to the degree such a commitment can actually
> reach the error, and it under-weights abstraction systematically, increasingly so as errors
> deepen. §4's *"calibrated low"* survives; its *"prefers higher levels"* does not.

---

## One-liner

The published action space is flat — `n_blocks = seq_len // s`, so a block **is** one level-1
feature and every move the value can score is a level-1 block edit, with `value_sensitivity`
aggregated per channel and never per level. This builds the missing axis: a move that commits to
one level-ℓ **feature** and renders the whole legal subtree beneath it, for every ℓ of every
grammar channel. Read on it, the exact DP says **depth raises the floor, not the ceiling** — the
*best* move is nearly flat across levels (+1.74 → +1.88 in Δ`d*`) while a *random* move rises
**+0.283 → +1.884**, because one abstract commitment re-derives a legal subtree without needing to
pick the right block. The learned value tracks this: per-node top-1 preference relative to level 1
runs **1.0× / 1.8× / 3.1× / 5.0×** against the DP's **1.0× / 0.9× / 3.6× / 8.2×**. Two controls
carry the result — a **span null** (distractor channels, where Δ`d*` is exactly 0.000 at every
level, still show |ΔV| rising 43% from L1 to L2) which removes the branching-factor confound from
the raw profile, and the DP's own top-1 distribution, which is what "under-shoots" is measured
against.

---

## 1. Why the question had no axis

[`../README.md`](../README.md) §3 reports this node's smoking gun — a value trained only on
terminal task success separates real tokens from distractors 13–18× — and then records why that
value can say nothing about the RHM's *hierarchy*: the budget is allocated over **channels**,
which are not ordered by level, so satiety *"recruits laterally into structA"* rather than upward.
The same flatness runs one level down, in the action space itself. Every candidate move is a
level-1 block edit; `value_sensitivity` is `dvalue[c]`, per channel.

The one level-adjacent measurement available in the published geometry is the between-channel
version, and it is flat: in [`../partial_hetero/`](../partial_hetero/README.md)'s ladder structA is
a full depth-4 hierarchy and structB is depth-2, both with ground-truth relevance exactly 0.000,
and the value assigns them |ΔV| 0.027 and 0.032 — indistinguishable, and indistinguishable from
iid noise at 0.029. That is a real datapoint but a weak one: all three have zero relevance, so
assigning them all ~0 is correct behaviour and says nothing about levels.

## 2. What a level-ℓ move is, and why it is not ℓ level-1 moves

The lazy construction — mask the span, infer each block's level-1 feature independently, render
each — is **not** acting at level ℓ. It is `s^(ℓ-1)` level-1 moves at once, and the result need not
be a legal level-ℓ subtree at all. `regenerate_node` instead commits to one abstract feature and
renders a whole legal subtree beneath it:

1. mask the node's `s^ℓ` tokens and read the generator's per-block level-1 feature logits, exactly
   as [`../channel_env.py`](../channel_env.py)'s `regenerate_block` does;
2. run an exact max-sum DP **up** the channel's own rule tables, so `score[lv][j][f]` is the best
   total level-1 log-evidence achievable by any derivation of feature `f` at node `j`;
3. take the argmax feature at level ℓ and backtrack to level-1 features for every block;
4. render each through that block's **current drifted mixture**, as the published move does — so
   drift still bites on the dynamics.

The move is "the generator's best guess, projected onto the grammar at level ℓ": always
grammatical, always a single abstract commitment.

### Two gates, both asserted (`selfcheck`)

- **C1 — the level-1 rung is the published move.** At ℓ=1 the DP is empty and the operator's
  feature choice is **identical** to `regenerate_block`'s on every row. So the level-1 row of every
  table below is comparable to published numbers rather than merely analogous.
- **C2 — every deeper move is one legal commitment.** (a) each parent's chosen rule actually
  produces its children, checked against the channel's own tables level by level; (b) each block's
  rendered tuple is one of the m synonyms of its chosen feature.

C2 is checked on the DP's own derivation, **not** by parsing the rendered tokens back through
`bottom_blk`. That map is last-writer-wins wherever two features share a leaf tuple — 11 such
collisions in this layout — so a parse-back check fails on legal moves and would be testing
`build_inverse_maps` rather than the operator. This cost one debugging round and is worth keeping
in mind for any future readout that wants to recover features from tokens.

**The action space at L=4.** 23 moves against the published 14 blocks: tree 8/4/2/1 at levels
1–4, structA 2/1, structB 2/1, noise 1 each (no grammar, so level-1 only — keeping them in the set
is what makes "the value devalues the irreducible channels" still measurable on this axis).

## 3. What a level is actually worth (the exact DP)

3 seeds, 512 frozen probe states, every move materialised:

| tree level | span | **mean Δ`d*`** (random node) | **best Δ`d*`** (best node) | nodes |
|---|---|---|---|---|
| 1 | 1 blk | +0.283 ± 0.003 | +1.740 | 8 |
| 2 | 2 blk | +0.226 ± 0.024 | +1.761 | 4 |
| 3 | 4 blk | +0.943 ± 0.059 | +2.072 | 2 |
| 4 | 8 blk | **+1.884 ± 0.067** | +1.884 | 1 |

**Depth raises the floor, not the ceiling.** The best available move is nearly flat across levels
(+1.74 → +1.88, slope +0.074/level, t = +5.28), while a *random* move at level 4 buys **6.7× what
a random level-1 move buys**. A level-4 move is nearly as good as the *best* level-1 move without
having to find it — one abstract commitment re-derives a legal subtree and cleans up corruption
wholesale.

This is a property of the DGP under this move operator, established before any learned quantity is
read, and it is the sense in which "higher levels explain more" is true here.

**One wrinkle**: the profile is not monotone at the bottom — level 2 buys slightly *less* than
level 1 (+0.226 vs +0.283). Damage is applied to level-1 blocks, so a 2-block move often straddles
one corrupt and one clean block and pays for overwriting the clean one; by level 3–4 the
re-derivation dominates that cost. That reading is suggested by the shape, not measured.

## 4. What the value does with it

`flat` arm — the published action space for training, probed on all 23 level moves. Per-node top-1
rate (top-1 share ÷ nodes at that level, the fair comparison since level 1 has 8 nodes competing
and level 4 has one), as a multiple of level 1:

| level | **value / node** | **DP / node** | value rel. L1 | DP rel. L1 |
|---|---|---|---|---|
| 1 | 0.0376 | 0.0369 | 1.0× | 1.0× |
| 2 | 0.0661 | 0.0342 | 1.8× | 0.9× |
| 3 | 0.1178 | 0.1335 | 3.1× | 3.6× |
| 4 | 0.1868 | 0.3014 | **5.0×** | **8.2×** |

**The value's level preference is real, correctly ordered, and calibrated low.** It climbs
monotonically and reaches 5.0× at the root where a privileged oracle reaches 8.2× — it
systematically under-selects the deepest level. Value-vs-DP rank correlation over all 23 moves is
**+0.284 ± 0.053**, against the published +0.31 over the 14 flat moves; top-1 is a tree move
**98.8%** of the time against the DP's 100%.

> **⚠ Retired 2026-08-03 — this table is a mean over seeds that disagree in sign.** Per-seed
> level-4 top-1 share is **0.400 / 0.041 / 0.119** (sd 0.189 on a mean of 0.187), so per-node
> relative to L1 the three seeds give **16.1× / 0.86× / 2.95×**: the profile is non-monotone in two
> of three, and in seed 2 the root is *dispreferred* relative to level 1. The 5.0× is seed 1
> carrying the mean. The **rank correlation and the tree-top-1 share are unaffected** and both
> replicate (§9). What replaces the level profile is the matched-span control in §10.

### The span null, which changes how the raw profile reads

Raw |ΔV| rises steeply with level — 0.533 / 0.816 / 1.075 / 1.233, slope **+0.236 ± 0.093,
t = +4.41** — which looks like a clean confirmation on its own. It is not, because a level-ℓ move
rewrites `s^ℓ` tokens and perturbs the latent more whether or not it buys anything. The distractor
channels supply the null for free: their ground-truth Δ`d*` is **exactly 0.000 at every level**
(P1), so any rise there is the branching factor alone.

| level | tree \|ΔV\| | distractor \|ΔV\| (null) | ratio |
|---|---|---|---|
| 1 | 0.5325 ± 0.0374 | 0.0642 ± 0.0086 | **8.3×** |
| 2 | 0.8161 ± 0.0508 | 0.0918 ± 0.0126 | **8.9×** |

|ΔV| rises **43%** from L1 to L2 in channels carrying zero information. So a substantial part of
the raw slope is mechanical, and reading it alone would have overstated the result. The per-node
top-1 comparison in §4 is the readout that survives this, because it is a *choice among* moves
rather than a magnitude.

> **⚠ Understated, 2026-08-03.** *"A substantial part"* is the whole of it. This null only reached
> L2 because `struct_depths` defaulted to `2,2`; extended to a depth-4 distractor (§9) the null
> rises **2.79×** from L1 to L4 against the tree's **2.56×**. And the per-node top-1 readout does
> **not** survive as claimed here — it is a choice among moves whose latent displacements differ
> mechanically, and it fails on seed spread anyway (§4's retirement note). The readout that
> actually removes the confound is the paired twin in §10.

**Relevance detection is level-invariant.** The tree/distractor ratio is 8.3× at level 1 and 8.9×
at level 2 — the node's 13–18× headline separation does not change with the scale of the move. The
value's ability to tell real from distractor and its preference among levels are separate,
independent properties.

## 5. The second arm, and a confound it exposes

`level` arm — the value's Monte-Carlo behaviour policy runs over the full 23-move set instead of
the 14 blocks. Everything else is shared: one controller, one generator, one frozen probe set.

| | flat | level | Δ (paired) |
|---|---|---|---|
| behaviour terminal success | 0.400 ± 0.020 | **0.523 ± 0.020** | +0.124, t = **+18.1** |
| value-vs-DP rank corr | 0.284 ± 0.053 | 0.169 ± 0.031 | −0.115, t = **−3.02** |
| per-node deep preference (L4) | 5.0× | 2.9× | — |
| value top-1 share slope vs level | −0.037, t = −0.75 | −0.075, t = **−5.43** | — |
| signed ΔV at L3 / L4 | +0.271 / +0.539 | −0.146 / −0.394 | — |

Giving the value the axis made the **policy** clearly better and the **value** less oracle-like,
with its signed ΔV flipping negative on deep moves.

**This is reported but not concluded from, because the two arms are not measuring the same
quantity.** ΔV under a *state*-value function is "is this state better than that one **under the
policy the value was trained on**". The level arm's value already anticipates that deep moves will
be taken, so taking one is no longer a surprise; the flat arm's value never anticipates them, so
it registers a large positive jump. The arms therefore differ in their reference policy as well as
in the action space, and signed ΔV is not comparable across them. **The flat arm is the more
interpretable observer for the level-preference question** — its policy does not use deep moves,
so its ΔV is closer to a policy-free assessment of the state — and §4's table is from it.

Separately and plainly: **the flat arm's calibrated residual does not resolve.** Per-seed level-4
values are +0.739 / −0.430 / −0.020. No mean is quoted for that cell. The level arm's residual
slope is resolvable (−0.201, t = −3.38) but inherits the confound above.

## 6. What this establishes — and what it does not

**Establishes:**

1. **A level-indexed action space** for the sculpting task — grammatical by construction, exactly
   reducing to the published move at level 1, and usable by any downstream reader. This is the
   durable output.
2. **Depth raises the floor, not the ceiling**, in this DGP under this operator: best Δ`d*` is
   nearly flat across levels while mean Δ`d*` rises 6.7×.
3. ~~**The value's level preference is real, monotone, and correctly ordered**, at 5.0× per-node
   against the oracle's 8.2×.~~ **Retired 2026-08-03** — the magnitude readout is span (§9) and the
   per-node readout is one seed of three (§4's note). What is established in its place is
   **abstraction preference at matched span** (§10) and **error-depth tracking** (§11).
4. **The span null**, which shows a naive |ΔV|-by-level reading is substantially the branching
   factor, and is available free in any layout carrying distractors. **Strengthened** — with a
   depth-matched distractor it accounts for the entire slope (§9).
5. **Relevance separation is level-invariant** (8.3× / 8.9×), so the node's headline distractor
   result is a property of the value that holds at every scale of move. **Extended to all four
   levels** against a full depth-4 distractor: 12.1× / 13.1× / 12.7× / 11.1×, with distractor
   top-1 share 0.001–0.003 everywhere (§9). This is the finding that came out *stronger*.
6. **A matched-span paired control** (§10) and a **hierarchical damage model** (§11), both certified
   and both reusable — the durable instruments this round adds.

**Does not establish:**

- **Nothing about a level-ordered *allocation* space.** This builds the action space and the
  readout. Open item 2 proper — a budget spent over (channel, level) cells, so §12's *"recruit
  ℓ+1"* has an upward direction — needs a forward model over spans, and is not this. No FM,
  planner or allocator was touched here; the probe materialises every move and reads V directly.
- **Nothing about whether training on the axis helps or hurts calibration** (§5's confound).
- **Nothing that transfers automatically to other depths or branching factors.** L=4, s=2, one
  layout; the span/level relationship is `s^ℓ` and both the null and the DP profile would move.

## 7. Caveats

- **The `level` arm's ΔV is not comparable to the `flat` arm's** (§5). Fixable by probing with an
  action-value, or by grading both arms with a third frozen value trained off-policy.
- **The flat arm's residual cell at level 4 is unresolved** (§5) and is reported as such.
- **Level 4 has exactly one node**, so its top-1 share is one number rather than an average over
  nodes, and its `mean` and `best` Δ`d*` coincide by definition.
- **Deep moves are less informed by construction**: a root-level move masks the entire tree slice,
  leaving the generator almost no evidence to condition on. §3's DP profile prices this correctly
  (it is measured on the same moves), but it means "level" and "amount of context available"
  co-vary and are not separated here.
- **3 seeds**, one rule draw per seed, 512 probe states.
- **The value is the published MC value head** trained on terminal possible-set success, unchanged
  apart from its behaviour policy's action set.
- **§11's damage-level rows differ in more than the damage.** The value is retrained on each damage
  model (it must be, or the arms conflate "prefers abstraction" with "generalises from shallow to
  deep damage"), so each row is a different value. The comparison is *within-row* value-vs-oracle
  and the *slope* of that gap; absolute levels across rows also move with task difficulty
  (terminal success 0.475 → 0.307).
- **§11's damage magnitude co-varies with damage depth.** One level-3 node spans 4 blocks against a
  level-1 node's 1, so deeper damage rewrites more tokens; `d*` is matched (3.94–4.40) but the
  number of damaged blocks is not. The primary comparison is *within* a run (premium by move level),
  which this does not touch.
- **§10's twins are not a rival action space.** They are excluded from every argmax, so `top1_share`
  and the rank correlation stay comparable to §4 — but probing extra moves advances the shared
  render RNG, so `lz2_*` is not bit-identical to `lv_*` (L4 top-1 0.398 / 0.043 / 0.092 against
  0.400 / 0.041 / 0.119). Drift is off throughout this node, so this is RNG only.
- **§9's null layout is not the published layout.** `--struct-depths 4,2` gives 20 blocks against
  14, so its absolute levels are not comparable to §3–§4; the tree-vs-null comparison is
  within-run, which is the point.

## 8. Open items

1. **An action-value or an off-policy third grader**, so the two arms become comparable (§5). This
   is the difference between "the level arm is uninterpretable" and "training on the axis hurts
   calibration".
2. ~~**A forward model over spans**, the prerequisite for a level-ordered allocation space and
   hence for testing §12 at all.~~ **Built, and the axis it unlocked is inert — see
   [`level_ladder/`](level_ladder/README.md) (2026-08-04).** The span FM turned out small
   (`BlockLatentFM`'s target is *already* the full latent delta; only the conditioning is
   per-block), and it reduces to the published FM bit-identically at level 1. With it, the
   allocation space, planner and ballistic grader are all level-indexed. But holding the channel
   allocation fixed, a **privileged, exactly-correct** per-cell allocator is worth
   **+0.00016 ± 0.00226 (t = +0.13)** against a paired floor of ±0.0009 — a level-*blind* oracle
   recovers **99.2%** of the whole allocation prize. The same swap applied to the **action**
   space is worth **3.51×** in ballistic control. So §12's mechanism is not waiting on this axis;
   the axis has no prize on it.
3. ~~**Why does the value stop at 5.0× when the oracle reaches 8.2×?**~~ Superseded: that gap was
   span and seed spread (§9). The live version is **why the abstraction premium is conservative and
   why the shortfall widens with depth** (§11 finding 4) — value head, `z.mean(dim=1)` pooling, or
   MC labels, still unmeasured, and now with a quantified target to explain.
4. **Sweep depth and branching factor.** The `s**ℓ` span growth is what §9's null controls for;
   whether §10's premium and §11's tracking slopes are stable in `s` and `L` is unknown.
5. **A partial-mask deep move** (§12) — commit at level ℓ but re-render only the corrupt blocks, so
   the commitment stays informed. This is the prerequisite for the root cell resolving at all, and
   it decouples "level" from "amount of context available", which §7 lists as unseparated.
6. **Attack the pooling** (§9's mechanism, §11 finding 4's suspect). An attention-pooled or
   position-tagged value head should shrink the null's 2.79× rise. It is the same instrument
   [`../partial_hetero/`](../partial_hetero/README.md) §7 item 3 wants for the encoder's ~0.69
   cross-position ceiling, so one rewrite serves two open items. **Scoped 2026-08-04**:
   [`level_ladder/`](level_ladder/README.md) G-E measures the cost of not doing it — every
   endogenous per-cell drive is span-dominated, and the three that track the exact oracle do so
   at **7–15%** of its rate, with the value **wrong-signed at the root** (−0.9 against +0.5 →
   +1.6). But it also removes the *allocation* motive for the fix: the axis that drive would
   steer is worth ~0. The pooling remains live for the **readout** and for the action-space
   line, not for allocation.

## 9. Child

| Child | What |
|---|---|
| [`level_ladder/`](level_ladder/README.md) | **The allocation half of open item 2 — built, and the axis is inert.** Supplies what this node deliberately did not: a span forward model (bit-identical to the published block FM at level 1, gate C4), a budget over (channel, level) cells, a level-indexed planner and ballistic grader, and a damage schedule that moves the error *deeper across the run* so climbing is necessary rather than available. Every upstream gate passes — the exact DP's best move level climbs 1 → 2 → 3 with the damage depth, and a privileged allocator follows it (mean level of tree spend 2.16 → 2.42 → 2.68, slope **+0.287 ± 0.038, t = +13.15**, 3/3 seeds). **And it buys nothing**: with the channel allocation held fixed at 99.4% tree, the level index is worth **+0.00016 ± 0.00226 (t = +0.13)** against a paired within-run floor of ±0.0009, and a level-*blind* oracle recovers **99.2%** of the +0.0217 allocation prize. A geometry null with every instrument certified, not an instrument null. **The complementary contrast is large**: switching only the *action space* from block-only to level-indexed, at matched task and matched allocation, moves ballistic control **0.121 → 0.424 (3.51×, +0.303 ± 0.051, t = +10.40)** — while making the FM *worse* on its dense proxy at the one shared cell (−0.106, t = −3.14), a quantified grader disagreement. Proposed mechanism (argument, not measurement): channels are disjoint sets of positions so allocating over them changes which parameters receive data, whereas levels are *nested* on the same blocks, so no level can be starved by spending at another. Also lands **G-E**, a ~3.5-minute per-cell scoreboard for any value head against the exact oracle, and the finding that `n_corrupt` stops being a free parameter once damage is hierarchical (the published 3 damages the whole tree and hides the structure entirely) |

---

# The 2026-08-03 corrections

Three structural corrections, in the order they have to be read. All are `--arms flat` (§5's
interpretable observer) unless stated, 3 seeds, and every default is off so §3–§5's runs reproduce.

## 9. The span null at every level, and what it retires

§4's null only reached L2 because `struct_depths` defaulted to `2,2`. [`partial_hetero`](../partial_hetero/README.md)
already built the depth-matched distractor, so extending it is one flag — `--struct-depths 4,2`
makes `structA` a depth-4, 8-block channel whose Δ`d*` is certified **exactly 0.000 at every
level** (P1/G1), with span growth identical to the tree's.

| level | tree \|ΔV\| | **null** \|ΔV\| (Δ`d*` ≡ 0) | ratio |
|---|---|---|---|
| 1 | 0.5633 ± 0.0412 | 0.0465 ± 0.0001 | 12.1× |
| 2 | 0.8651 ± 0.0512 | 0.0661 ± 0.0037 | 13.1× |
| 3 | 1.1484 ± 0.0853 | 0.0906 ± 0.0149 | 12.7× |
| 4 | 1.4407 ± 0.2705 | 0.1299 ± 0.0266 | 11.1× |

**L1→L4 rise: tree 2.56×, null 2.79×.** The tree's magnitude response to level is *no larger* than
in a channel where level means nothing — if anything marginally smaller, and the ratio is
flat-to-declining. §4's raw slope (+0.236 ± 0.093, t = +4.41) is the branching factor, measured now
at all four levels rather than extrapolated from two.

**Two things came out stronger.** Relevance separation holds at every level against a *full depth-4
hierarchy* rather than a shallow stub (12.1× / 13.1× / 12.7× / 11.1×; distractor top-1 share
0.001–0.003 at L1–L4), so [`../README.md`](../README.md) §3's headline is untouched and better
certified. And the per-node top-1 *pattern* replicates on this second layout — value 1.0× / 1.8× /
3.8× / 9.6× against the DP's 1.0× / 1.0× / 4.7× / 17.1×, the same shape as §4's 5.0-vs-8.2 — so the
ranking readout is layout-robust even though it fails on seed spread (§4's note).

**Note on the earlier instrument.** `residual` (§4, ΔV regressed on Δ`d*`) was built for exactly
this confound and never resolved (+0.739 / −0.430 / −0.020 at L4). A *structural* null resolved it
immediately. The lesson worth carrying: **a matched control does more than a matched regression when
the readout's components are not independent** — the same lesson [`../partial_hetero/`](../partial_hetero/README.md)
§7 item 6 records from the other direction.

## 10. The matched-span control: abstraction preference with the confound removed by construction

The null corrects magnitudes but cannot correct the *within-tree ordering*, because argmax never
lands off-tree at all (distractor top-1 ≈ 0.002 — relevance dominates). So a second control:
`--lazy-twins` pairs every committed level-ℓ move with a **lazy twin** at the same node — the "lazy
construction" §2 rejects as a move, which masks the same span and picks each block's level-1 feature
*independently*, with no abstract commitment. Identical tokens rewritten, so the pooling
displacement cancels **inside the pair**.

**Certified by C3** (asserted in `selfcheck`): spans identical; the twin differs from its commitment
on **58%** of rows; at L2 the lazy span is **not** a legal level-2 subtree on **37%** of rows. And
the displacement is matched *empirically*, not only positionally — |ΔV| for commitment vs twin at
seed 1 is 1.628 vs 1.589 (L4), 1.186 vs 1.149 (L3). The twins are never actions: no arm trains on
them and every ranking readout is computed over committed moves only.

`d*` is integer-valued so many pairs tie on ground truth while ΔV never does; the **untied** subset
puts both rates on the same footing and is the only fair comparison.

| cell | ΔV premium | value pref (untied) | DP pref (untied) | pairwise agreement |
|---|---|---|---|---|
| tree L2 | +0.0629 ± 0.0290 | **0.557 ± 0.018** (t = +5.5) | 0.610 ± 0.014 | **0.582 ± 0.042** (t = +3.4) |
| tree L3 | +0.2713 ± 0.1563 | **0.618 ± 0.052** (t = +3.9) | 0.642 ± 0.030 | **0.635 ± 0.030** (t = +7.8) |
| tree L4 | +0.6600 ± 1.1194 | 0.678 ± 0.232 | 0.435 ± 0.154 | 0.537 ± 0.135 |

**On identical token spans the value prefers one legal abstract commitment over `s**(ℓ-1)`
independent guesses** — resolved at L2 and L3, calibrated slightly *below* the exact oracle, and
agreeing with it pair-by-pair above chance.

**The grammaticality null.** A legal commitment is on the manifold the encoder was trained on, so it
might score higher for that reason alone. In the distractor channels legality buys Δ`d*` = 0
*exactly*, and there the premium is **+0.0027 ± 0.0040 against the tree's +0.0629 ± 0.0290** — 23×
smaller, **4.4%**. The preference is task-referenced, not manifold-shaped. (Read on the *premium*,
not on a sign rate: with a distribution centred at ~0 a sign rate is a skew statistic.)

**What does not resolve**: the preference *slope* against level (+0.060 ± 0.108, t = +0.97;
per-seed +0.154 / −0.057 / +0.084), and the L4 cell on either side — see §12 for why L4 cannot
resolve here at all.

## 11. Hierarchical damage: making error depth load-bearing, and the tracking result

§10 leaves an obvious question: is there a depth signal in this DGP *to* find? `tree_m` is already 2
at v = 8, so grammar constraint was never the limiter. The limiter is the **damage model**:
`channel_env.corrupt_tree` writes random symbols, which leaves **29.4%** of tree blocks off-grammar
and therefore repairable one block at a time. No error in the published DGP *requires* abstraction
to see, so a commitment has nothing a level-1 edit cannot also get.

`corrupt_tree_hier` writes the opposite kind of damage: a level-k subtree replaced by a legal
derivation of a feature that `possible_sets` proves the observed subtree **cannot** produce (the
complement is what makes it a real inconsistency — a different derivation of the *same* feature
leaves `d*` untouched, which is the trap). Every block stays on-grammar, so nothing is locally
suspicious; what is wrong is the level-k node, and only a commitment at level ≥ k can re-derive it.

**G-D, the DGP gate** (`certify_damage`, asserted in-run; CPU-only via `certify_damage_remote`):

| damage | tree blocks on-grammar | `d*` mean | `d*` == 0 |
|---|---|---|---|
| 0 (published) | **0.7059** | 4.237 | 0.000 |
| 1 | **1.0000** | 3.942 | 0.001 |
| 2 | **1.0000** | 4.400 | 0.007 |
| 3 | **1.0000** | 3.965 | 0.038 |

100% on-grammar at every depth **at matched `d*`** — so the knob varies *where the error lives*, not
how hard the task is. `damage_level = 1` is the load-bearing control: on-grammar but shallow, which
separates "deep" from "on-grammar" and from the damage change itself.

**The prediction.** A move at level ℓ can only repair an error at level ≤ ℓ, so the *oracle's*
abstraction premium at L2 must fall as the error moves to level 3, while L3's holds. It does:

| damage depth | L2 oracle | L2 value | L3 oracle | L3 value | value-vs-DP rank corr | terminal success |
|---|---|---|---|---|---|---|
| 0 (published) | 0.610 ± 0.014 | 0.557 ± 0.018 | 0.642 ± 0.030 | 0.618 ± 0.052 | +0.290 ± 0.054 | 0.400 |
| 1 | 0.595 ± 0.010 | 0.541 ± 0.032 | 0.654 ± 0.038 | 0.593 ± 0.053 | +0.319 ± 0.014 | 0.475 |
| 2 | 0.566 ± 0.006 | 0.514 ± 0.014 | 0.617 ± 0.019 | 0.545 ± 0.094 | +0.500 ± 0.022 | 0.362 |
| 3 | **0.536 ± 0.038** | **0.458 ± 0.018** | 0.612 ± 0.003 | 0.515 ± 0.051 | **+0.583 ± 0.031** | 0.307 |

Slopes against damage depth, per seed:

| | oracle | value |
|---|---|---|
| L2 premium | **−0.0250 ± 0.0105, t = −4.14** | **−0.0325 ± 0.0083, t = −6.81** |
| L3 premium | −0.0128 ± 0.0103, t = −2.15 | −0.0359 ± 0.0126, t = −4.92 |

1. **The knob does what it was built to do.** The oracle's L2 premium falls (t = −4.14) at roughly
   twice the rate of its L3 premium (t = −2.15) — the graded signature of "a level-2 move can less
   and less reach the error".
2. **The value tracks it**, L2 slope −0.0325 (t = −6.81), monotone in 3/3 seeds
   (0.564→0.478, 0.537→0.452, 0.570→0.444).
3. **Value–oracle rank agreement roughly doubles as errors become hierarchical** — +0.290 → +0.583,
   slope **+0.1059 ± 0.0225, t = +8.14**. Meanwhile terminal success *falls* (0.475 → 0.307), so the
   task got harder while the value got more oracle-like. That pair is the internal control against
   "the value simply got flatter", which is otherwise the live alternative to a tracking effect.
4. **The value is systematically conservative and the shortfall widens with depth.** Tracking error
   (value − oracle) at L2: −0.053 / −0.054 / −0.052 / −0.078; at L3: −0.024 / −0.060 / −0.071 /
   −0.097. At L3 it withdraws from abstraction ~3× faster than warranted.

**Read narrowly, this is what the cut establishes**: the value's preference for an abstract
commitment is graded by whether such a commitment can reach the error, at matched span, against an
exact oracle — i.e. sensitivity to the hierarchy's *compositional* structure rather than to depth as
such. It is **suggestive but not established** that the published DGP was understating the value's
competence generally; the rank-correlation doubling is one readout on one substrate.

## 12. Why the root cell is structurally uninformative

The L4 Δ`d*` premium came back **bit-identical (−1.1562, tie 0.275, n_untied 371) at all three
damage levels**. That is not noise and not a bug. A level-4 move masks *every* tree token, so the
generator conditions only on the non-tree context — which the damage never touches and which is
identically seeded — and emits the same span whatever the damage was. `d_cur` then cancels out of
the premium, which is therefore an exact invariant of the damage model.

Two consequences. **L4 must be excluded from any damage-indexed reading** (the aggregator does, and
says so). And **the root cell's failure to resolve anywhere in this node — §4's residual, §5's
per-seed +0.739 / −0.430 / −0.020, §10's 0.678 ± 0.232 — has a single structural cause**, not a
sample-size one. More seeds cannot fix it. What fixes it is a **partial-mask deep move**: commit at
level ℓ but re-render only the blocks that are actually corrupt, so the commitment stays informed.
That is now open item 5 and it is the prerequisite for asking the depth question at the root at all.

## Reproduce

```bash
cd experiments/            # MODAL_PROFILE=chromatic
python3 -c "from rhm.verify_backcompat import verify; verify()"                     # the gate

# C1/C2/C3 -- level-1 identity, grammaticality above it, and the lazy twin non-vacuous
modal run rhm/directed_sculpting/full_loop/level_moves/level_moves.py::selfcheck_remote
# G-D -- the damage gate: on-grammar by damage level, no GPU
modal run rhm/directed_sculpting/full_loop/level_moves/level_moves.py::certify_damage_remote
modal run rhm/directed_sculpting/full_loop/level_moves/level_moves.py::level_probe --quick --tag smoke

for s in 1 2 3; do
  modal run --detach rhm/directed_sculpting/full_loop/level_moves/level_moves.py::level_probe \
      --tag lv_s$s --seed $s
done

for s in 1 2 3; do
  modal volume get --force rhm-scaling-data "directed_sculpting/level_moves/level_lv_s$s" \
      rhm/directed_sculpting/full_loop/level_moves/figures/; done
python3 rhm/directed_sculpting/full_loop/level_moves/aggregate.py --pattern 'level_lv_*'
```

`--arms flat` alone runs only the published action space for training (still probed on the full
level move set), which is the arm §4 quotes.

### The 2026-08-03 corrections (§9–§12)

`--pattern` is **required** when more than one sweep is mirrored into `figures/`: results are keyed
by seed, so `level_lv_s1` and `level_null4_s1` are the same seed on different layouts and would
silently overwrite each other in the aggregator.

```bash
# §9 -- the span null at every level: structA becomes a depth-4, 8-block, Dd*=0.000 channel
for s in 1 2 3; do
  modal run --detach rhm/directed_sculpting/full_loop/level_moves/level_moves.py::level_probe \
      --tag null4_s$s --seed $s --struct-depths 4,2
done

# §10 -- the matched-span paired control on the PUBLISHED layout (comparable to S4)
for s in 1 2 3; do
  modal run --detach rhm/directed_sculpting/full_loop/level_moves/level_moves.py::level_probe \
      --tag lz2_s$s --seed $s --lazy-twins
done

# §11 -- hierarchical damage: the depth at which the error LIVES is the knob
for s in 1 2 3; do for d in 1 2 3; do
  modal run --detach rhm/directed_sculpting/full_loop/level_moves/level_moves.py::level_probe \
      --tag dmg${d}_s$s --seed $s --lazy-twins --damage-level $d --arms flat
done; done

python3 rhm/directed_sculpting/full_loop/level_moves/aggregate.py --pattern 'level_null4_*'  # S9
python3 rhm/directed_sculpting/full_loop/level_moves/aggregate.py --pattern 'level_lz2_*'    # S10
python3 rhm/directed_sculpting/full_loop/level_moves/aggregate.py --damage-sweep             # S11
```

`--lazy-twins` and `--damage-level` both default off, so every §3–§5 command above reproduces the
published runs unchanged.
