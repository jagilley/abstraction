# level_moves — File index

Complete file-by-file reference for this sub-experiment. Summarized in [README.md](README.md);
parent index at [`../FILES.md`](../FILES.md).

The environment this extends is [`../channel_env.py`](../channel_env.py) — `regenerate_block`
(the published level-1 move), `corrupt_tree`, `sample_states`, `build_block_tables`,
`train_generator_channels` and `_block_state_chunked` are all imported unchanged. The move set
deliberately lives **here** rather than in `channel_env.py`: it is a new action space rather than a
modification of the existing one, and nothing else on the node reads it yet. If a level-ordered
allocation space is built (README open item 2), promoting `build_move_set` / `regenerate_node`
one level up is the natural first step.

## Code files

| File | Purpose |
|---|---|
| `level_moves.py` | **The level-indexed action space and its readout.** `build_move_set` enumerates every (channel, level, node); `regenerate_node` applies one — masking the node's `s**ell` tokens, running an exact max-sum DP up the channel's own rule tables to pick a legal derivation of a single level-`ell` feature, and rendering each block through its current drifted mixture, so the move is one abstract commitment rather than `s**(ell-1)` independent level-1 edits. `collect_value_buffer_moves` / `_behavior_step_moves` are `channel_env`'s value collection over an arbitrary move set, so the `flat` arm reproduces the published action space exactly. `level_value_probe` materialises every move and records the value's ΔV against the exact DP's Δ`d*`, resolved per (channel, level): raw \|ΔV\|, best/mean Δ`d*`, top-1 share against the **DP's own** top-1 share, and a residual after regressing ΔV on true Δ`d*`. `selfcheck` asserts **C1** (at level 1 the operator's feature choice is identical to `regenerate_block`'s) and **C2** (every deeper move is one legal commitment, rendered as a legal synonym) — C2 reads the DP's own derivation rather than parsing tokens back through `bottom_blk`, which is last-writer-wins on ambiguous tables |
| `aggregate.py` | Reads the mirrored `figures/` JSONs and prints the four tables in the order they must be read: the raw \|ΔV\| profile; the **span null** (the same profile in distractor channels, where Δ`d*` is exactly 0.000 at every level, so a rise is the branching factor); what the level actually buys, with the DP's top-1 distribution; and the calibrated residual. Per-seed least-squares slopes against level with t-stats, plus a paired `level − flat` comparison |

## Results

| Path | What |
|---|---|
| `figures/level_lv_s{1,2,3}/results.json` | **The probe, 3 seeds, both value arms**, 512 frozen probe states, 23 moves. Carries per-(channel, level) cells, the tree-only by-level summary, the calibration line, and each arm's behaviour-policy terminal success. The `flat` arm is the one [README.md](README.md) §4 quotes |

## Gotchas worth keeping

- **`bottom_blk` is last-writer-wins.** 11 leaf-tuple collisions across features in the L=4 layout,
  so any readout that recovers features by parsing tokens is testing `build_inverse_maps`. C2 was
  written twice for this reason.
- **The two value arms differ in reference policy, not only in action space.** ΔV under a state
  value means "better under the policy this value was trained on", so signed ΔV is not comparable
  across `flat` and `level` ([README.md](README.md) §5).
- **Raw |ΔV| by level is confounded by span** and must be read against the distractor null.

## Reading order

1. [README.md](README.md) §2 — what a level-ℓ move is, and the two gates. Everything downstream
   depends on the move being one abstract commitment.
2. [README.md](README.md) §3 — the DP profile, established before any learned quantity is read.
3. [README.md](README.md) §4 — the per-node top-1 table (the result) and the span null (the
   control that makes it readable).
