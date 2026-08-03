# A level-indexed action space: what the value system thinks a level of the hierarchy is worth

**Status**: built and run — action space certified two ways, 3 seeds, two value arms.
**The value does prefer higher levels of the RHM hierarchy, monotonically, and it is right to:
ground truth says a random deep move buys 6.7× what a random shallow one does. It under-shoots
the exact DP by roughly 40% at the root.** **Date**: 2026-08-03.
**Up**: [../README.md](../README.md) (full_loop) · **Node**: [../../../README.md](../../../README.md) (rhm) · **Files**: [FILES.md](FILES.md)
**Builds**: [`../README.md`](../README.md) §3 and open item 2 — *"§12 names satiety as the climbing
mechanism, but climbing needs the allocation space to be ordered by level. Ours is not."*
**Idea doc**: [`ideas/adaptive_core_and_hierarchy_climb.md`](../../../../../ideas/adaptive_core_and_hierarchy_climb.md)
§12 (satiety-gated *"once level ℓ stops paying, recruit ℓ+1"*), which has never had an axis to
recruit along.

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
3. **The value's level preference is real, monotone, and correctly ordered**, at 5.0× per-node
   against the oracle's 8.2× — not the level-blindness the channel-indexed measurement implied.
4. **The span null**, which shows a naive |ΔV|-by-level reading is substantially the branching
   factor, and is available free in any layout carrying distractors.
5. **Relevance separation is level-invariant** (8.3× / 8.9×), so the node's headline distractor
   result is a property of the value that holds at every scale of move.

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

## 8. Open items

1. **An action-value or an off-policy third grader**, so the two arms become comparable (§5). This
   is the difference between "the level arm is uninterpretable" and "training on the axis hurts
   calibration".
2. **A forward model over spans**, the prerequisite for a level-ordered allocation space and hence
   for testing §12 at all.
3. **Why does the value stop at 5.0× when the oracle reaches 8.2×?** Whether that is the value
   head, the encoder's pooled representation (`z.mean(dim=1)` discards where the change happened),
   or the MC labels is unmeasured.
4. **Sweep depth and branching factor.** The `s^ℓ` span growth is the confound the span null
   controls for; whether the 5.0×-vs-8.2× shortfall is stable in `s` and `L` is unknown.

## Reproduce

```bash
cd experiments/            # MODAL_PROFILE=chromatic
python3 -c "from rhm.verify_backcompat import verify; verify()"                     # the gate

# C1/C2 -- the action space is the published move at level 1 and grammatical above it
modal run rhm/directed_sculpting/full_loop/level_moves/level_moves.py::selfcheck_remote
modal run rhm/directed_sculpting/full_loop/level_moves/level_moves.py::level_probe --quick --tag smoke

for s in 1 2 3; do
  modal run --detach rhm/directed_sculpting/full_loop/level_moves/level_moves.py::level_probe \
      --tag lv_s$s --seed $s
done

for s in 1 2 3; do
  modal volume get --force rhm-scaling-data "directed_sculpting/level_moves/level_lv_s$s" \
      rhm/directed_sculpting/full_loop/level_moves/figures/; done
python3 rhm/directed_sculpting/full_loop/level_moves/aggregate.py
```

`--arms flat` alone runs only the published action space for training (still probed on the full
level move set), which is the arm §4 quotes.
