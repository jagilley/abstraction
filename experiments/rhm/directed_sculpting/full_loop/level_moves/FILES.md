# level_moves — File index

Complete file-by-file reference for this sub-experiment. Summarized in [README.md](README.md);
parent index at [`../FILES.md`](../FILES.md).

The environment this extends is [`../channel_env.py`](../channel_env.py) — `regenerate_block`
(the published level-1 move), `corrupt_tree`, `sample_states`, `build_block_tables`,
`train_generator_channels` and `_block_state_chunked` are all imported unchanged. The move set
deliberately lives **here** rather than in `channel_env.py`: it is a new action space rather than a
modification of the existing one, and nothing else on the node reads it yet. If a level-ordered
allocation space is built (README open item 2), promoting `build_move_set` / `regenerate_node`
one level up is the natural first step. **Done 2026-08-04** — [`level_ladder/`](level_ladder/README.md)
imports both from here unchanged rather than promoting them, since the allocation axis turned out
inert and the action space is what carries the result.

## Code files

| File | Purpose |
|---|---|
| `level_moves.py` | **The level-indexed action space and its readout.** `build_move_set` enumerates every (channel, level, node); `regenerate_node` applies one — masking the node's `s**ell` tokens, running an exact max-sum DP up the channel's own rule tables to pick a legal derivation of a single level-`ell` feature, and rendering each block through its current drifted mixture, so the move is one abstract commitment rather than `s**(ell-1)` independent level-1 edits. `collect_value_buffer_moves` / `_behavior_step_moves` are `channel_env`'s value collection over an arbitrary move set, so the `flat` arm reproduces the published action space exactly. `level_value_probe` materialises every move and records the value's ΔV against the exact DP's Δ`d*`, resolved per (channel, level): raw \|ΔV\|, best/mean Δ`d*`, top-1 share against the **DP's own** top-1 share, and a residual after regressing ΔV on true Δ`d*`. `selfcheck` asserts **C1** (at level 1 the operator's feature choice is identical to `regenerate_block`'s) and **C2** (every deeper move is one legal commitment, rendered as a legal synonym) — C2 reads the DP's own derivation rather than parsing tokens back through `bottom_blk`, which is last-writer-wins on ambiguous tables. **2026-08-03 additions**: `build_move_set(lazy_twins=True)` adds, per grammar node at level ≥ 2, a **lazy twin** that skips the DP and takes each block's level-1 argmax independently — identical tokens rewritten, no abstract commitment, so the span/pooling confound cancels within the pair; **C3** gates it (spans identical, the twin *is* the per-block argmax, and it differs from its commitment often enough to be non-vacuous — 58% of rows, illegal at L2 on 37%). `corrupt_tree_hier` replaces a level-k tree subtree with a legal derivation of a feature the observed subtree provably **cannot** produce (drawn from the complement of `possible_sets`, so it is a real inconsistency rather than a re-rendering of the same feature, which would leave `d*` untouched); `certify_damage` / `certify_damage_remote` is the **G-D** gate on it (100% on-grammar at every depth against 0.706 for the published random-symbol damage, at matched `d*`). `level_value_probe` gains `matched_span` — the paired premium and preference per (channel, level), with **untied** variants because `d*` is integer-valued so ties dilute the DP's rate while ΔV never ties. Twins are never actions: no arm trains on them, and every argmax / calibration / rank readout is committed-only |
| `aggregate.py` | Reads the mirrored `figures/` JSONs and prints the four tables in the order they must be read: the raw \|ΔV\| profile; the **span null** (the same profile in distractor channels, where Δ`d*` is exactly 0.000 at every level, so a rise is the branching factor); what the level actually buys, with the DP's top-1 distribution; and the calibrated residual. Per-seed least-squares slopes against level with t-stats, plus a paired `level − flat` comparison. **2026-08-03**: `--pattern` selects ONE sweep (results are keyed by seed, so mirroring two sweeps into `figures/` silently overwrote cells — the bug this fixes was live); `report_matched_span` prints §10's paired control with the **untied** columns that put the value's rate and the DP's on the same subset, plus the grammaticality null read on premium magnitude rather than a sign rate; `--damage-sweep` groups by `damage_level` and prints §11's damage × move-level table, excluding L4 for the reason in §12 |

## Results

| Path | What |
|---|---|
| `figures/level_lv_s{1,2,3}/results.json` | **The probe, 3 seeds, both value arms**, 512 frozen probe states, 23 moves. Carries per-(channel, level) cells, the tree-only by-level summary, the calibration line, and each arm's behaviour-policy terminal success. The `flat` arm is the one [README.md](README.md) §4 quotes |
| `figures/level_null4_s{1,2,3}/results.json` | **§9, the span null at every level.** `--struct-depths 4,2`, so `structA` is a depth-4 8-block channel with Δ`d*` certified 0.000 at L1–L4; 20 blocks, 35 moves. This is the sweep that retires §4's raw profile — null \|ΔV\| rises 2.79× against the tree's 2.56× |
| `figures/level_lz2_s{1,2,3}/results.json` | **§10, the matched-span paired control** on the published layout, both arms, 32 probe moves (23 committed + 9 twins). Carries `matched_span` with the untied columns and the grammaticality null |
| `figures/level_dmg{1,2,3}_s{1,2,3}/results.json` | **§11, the damage-level sweep**, `flat` arm only, 3 damage depths × 3 seeds. Each carries `damage_certification` (the G-D gate for its own run). Read with `aggregate.py --damage-sweep`, which adds the published-damage row from `level_lz2_*` as damage 0 |

## Children

| Child | What |
|---|---|
| [`level_ladder/`](level_ladder/README.md) ([FILES](level_ladder/FILES.md)) | The **allocation** half of README open item 2: a span forward model (C4: bit-identical to the published block FM at level 1), a budget over (channel, level) cells, a level-indexed planner/ballistic grader, and a damage schedule that deepens the error across the run. Finding: the level index is **inert as an allocation axis** (+0.00016 ± 0.00226, t = +0.13, against a paired floor of ±0.0009; a level-blind oracle takes 99.2% of the prize) and **large as an action axis** (ballistic 0.121 → 0.424, 3.51×, t = +10.40). Adds gate **G-E**, a ~3.5-min per-cell scoreboard for any value head against the exact DP |

## Gotchas worth keeping

- **`bottom_blk` is last-writer-wins.** 11 leaf-tuple collisions across features in the L=4 layout,
  so any readout that recovers features by parsing tokens is testing `build_inverse_maps`. C2 was
  written twice for this reason.
- **The two value arms differ in reference policy, not only in action space.** ΔV under a state
  value means "better under the policy this value was trained on", so signed ΔV is not comparable
  across `flat` and `level` ([README.md](README.md) §5).
- **Raw |ΔV| by level is confounded by span** and must be read against the distractor null. **And
  the null must be depth-matched**: with `struct_depths=2,2` it only reaches L2, which understated
  the confound as "substantial" when it is the whole of it ([README.md](README.md) §9).
- **The root cell cannot respond to the damage model, ever.** A level-4 move masks every tree token,
  so the generator conditions only on the (undamaged, identically seeded) non-tree context and
  `d_cur` cancels out of the premium — measured as a bit-identical L4 Δ`d*` premium across all three
  damage levels. Every failure of the root cell to resolve in this node has this one structural
  cause; more seeds cannot fix it ([README.md](README.md) §12).
- **`aggregate.py` keys results by seed, not by sweep.** Mirroring two sweeps into `figures/` made
  `level_lv_s1` and `level_null4_s1` collide silently. Always pass `--pattern`.
- **Ties dilute the DP's preference rate but not the value's.** `d*` is integer-valued (tie rates
  0.62 / 0.33 / 0.26 at L2 / L3 / L4) while ΔV essentially never ties, so an all-pairs value rate
  compared to an all-pairs DP rate is apples-to-oranges. Use the `*_untied` fields.
- **A sign rate is not a preference when the distribution is centred at zero.** In the distractor
  channels the commit-vs-twin ΔV difference is ~0.003, and its sign rate (0.36) reads as a
  preference while the magnitude says there is essentially nothing there. The grammaticality null is
  therefore read on the *premium*, not the rate.

## Reading order

1. [README.md](README.md) §2 — what a level-ℓ move is, and the two gates. Everything downstream
   depends on the move being one abstract commitment.
2. [README.md](README.md) §3 — the DP profile, established before any learned quantity is read.
3. [README.md](README.md) §9–§12 — **the corrections, and the current headline.** §9 retires the
   raw profile, §10 is the paired control that replaces it, §11 is the damage-level result, §12 is
   why the root cell is structurally out of reach.
4. [README.md](README.md) §4–§5 — the superseded readouts, kept as the record of what was measured
   and annotated in place. Read *after* §9–§12, not before.
