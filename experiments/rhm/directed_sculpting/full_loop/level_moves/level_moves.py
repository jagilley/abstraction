"""A LEVEL-INDEXED ACTION SPACE for the sculpting task, and what the value does with it.

WHY THIS EXISTS
---------------
`../README.md` §3 reports the node's smoking gun -- a value trained only on terminal task
success separates real tokens from distractors 13-18x -- and then, two sections later, records
why that value can say nothing at all about the RHM's *hierarchy*:

    "§12 names satiety as the climbing mechanism, but climbing needs the allocation space to be
     ordered by level. Ours is not."                                       (§3, and open item 2)

The action space is flat. `n_blocks = seq_len // s`, so a block IS one level-1 feature and every
candidate move the value can score is a level-1 block edit. `value_sensitivity` is aggregated per
CHANNEL and never per level. So the natural question -- *does the value rate higher levels of the
hierarchy higher, since they "explain more" of the data?* -- has no channel through which to be
answered, let alone asked. The one thing measurable in the published geometry is the BETWEEN-channel
version, and it comes back flat: in `partial_hetero`'s ladder, structA is a full depth-4 hierarchy
and structB is depth-2, both with ground-truth relevance exactly 0.000, and the value assigns them
|ΔV| 0.027 and 0.032 -- indistinguishable, and indistinguishable from iid noise at 0.029.

This module supplies the missing axis: a move that acts on a level-`ell` NODE, rewriting the
`s**ell` tokens it dominates, for every `ell` in 1..depth of every grammar channel. With that, the
value can be asked directly.

WHAT A LEVEL-ell MOVE IS, AND WHY IT IS NOT `ell` LEVEL-1 MOVES
---------------------------------------------------------------
The lazy construction -- mask the span, infer each block's level-1 feature independently, render
each -- is NOT acting at level `ell`. It is `s**(ell-1)` level-1 moves applied at once, and the
result need not be a legal level-`ell` subtree at all. Measuring the value on that would answer a
different question than the one asked.

`regenerate_node` commits to ONE abstract feature and renders a whole legal subtree beneath it:

  1. mask the node's `s**ell` tokens and read the generator's per-block level-1 feature logits,
     exactly as `channel_env.regenerate_block` does;
  2. run an exact max-sum DP UP the channel's own rule tables, so `score[lv][j][f]` is the best
     total level-1 log-evidence achievable by any derivation of feature `f` at node `j`;
  3. take the argmax feature at level `ell` and backtrack to level-1 features for every block;
  4. render each through that block's CURRENT drifted mixture (`w_blk`), as the published move does.

So the move is "the generator's best guess, projected onto the grammar at level `ell`" -- always
grammatical, always a single abstract commitment. **At `ell=1` the DP is empty and the operator
reduces exactly to `regenerate_block`'s feature choice**, which `selfcheck` asserts, so the level-1
rung of every table below is comparable to the published numbers rather than merely analogous.

THE CONFOUND THIS IS BUILT AROUND
---------------------------------
A level-`ell` move rewrites `s**ell` tokens, so it mechanically moves `d*` further than a level-1
move -- in both directions. Raw |ΔV| by level would therefore measure the branching factor and
report it as a preference. Every readout here is accordingly paired with the exact DP's own answer
on the SAME move:

    best_dstar_gain[ch][ell]   what the best move at that (channel, level) actually buys
    dp_top1_share[ch][ell]     where a privileged oracle would act, given this action space
    residual[ch][ell]          mean ΔV after regressing ΔV on the move's true Δd* --
                               positive = the value OVER-rates that level for what it buys

`dp_top1_share` is the reference distribution the value's own `top1_share` is read against, and
it is what makes a flat |ΔV| profile interpretable: if deep moves buy nothing (they are near-blind
commitments -- a root-level move masks the whole tree slice, so the generator has almost no
evidence left to condition on), then a value that rates them low is CORRECT, not level-blind.

TWO STRONGER CORRECTIONS, ADDED 2026-08-03
------------------------------------------
`residual` above corrects the confound by REGRESSION (ΔV on Δd*, fit on tree moves pooled) and it
did not resolve -- per-seed level-4 values +0.739 / -0.430 / -0.020. Two structural corrections
replace it, neither of which needs the fit:

  the span null      run with `--struct-depths 4,2`, so structA is a depth-4, 8-block channel with
                     ground-truth Δd* EXACTLY 0.000 at every level (P1/G1). Its |ΔV|-by-level
                     profile is the pure mechanical rise: a level-ell move rewrites `s**ell`
                     tokens and so displaces `z.mean(dim=1)` by ~`s**ell`/n_blocks whether or not
                     it buys anything. The tree profile is only informative ABOVE this.
  the lazy twin      `--lazy-twins`. A paired control at the SAME node: identical tokens rewritten,
                     so the displacement cancels inside the pair, and the twins differ in exactly
                     one respect -- one legal abstract commitment vs `s**(ell-1)` independent
                     level-1 guesses. `matched_span.pref_rate` is then level preference with the
                     confound removed BY CONSTRUCTION. This is the readout the span null cannot
                     give: the null shows argmax never lands off-tree at all (relevance dominates),
                     so it corrects the MAGNITUDE readout but says nothing about the WITHIN-tree
                     ordering that `top1_share` measures.

The twins are never actions -- no arm trains on them, and every ranking readout (`top1_share`,
the calibration line, the rank correlation) is computed over committed moves only, so a run with
`--lazy-twins` is directly comparable to one without.

TWO VALUE ARMS
--------------
Whether the value CAN express a level preference and whether it DOES are different questions, and
they separate on how the value was trained:

    flat    behaviour policy over level-1 blocks only -- the published `collect_value_buffer`.
            Probed on level moves, this asks: does a value trained on a flat action space
            already rate deep moves correctly?
    level   behaviour policy over the full level-indexed move set. This is the level-ordered
            action space proper. If a preference appears here and not in `flat`, the answer is
            "the value can express it once the action space has it".

Both arms share one controller, one generator and one frozen probe set, so the arms differ only in
which moves generated their value data.

SCOPE. This builds the ACTION SPACE and the READOUT. It deliberately does NOT build a level-indexed
forward model, planner or allocator: the probe materialises every move and reads V directly (like
`channel_env.value_relevance_check`), and the behaviour policy scores materialised proposals with
the controller (like `channel_env._behavior_step`), so no FM is involved anywhere. A level-ordered
ALLOCATION space -- open item 2 proper, where the budget is spent over (channel, level) cells and
§12's "once level ell stops paying, recruit ell+1" finally has an upward direction -- needs an FM
over spans and is not this.

Run from experiments/:
  python3 -c "from rhm.directed_sculpting.full_loop.level_moves.level_moves import selfcheck; selfcheck()"
  modal run rhm/directed_sculpting/full_loop/level_moves/level_moves.py::level_probe --quick
  for s in 1 2 3; do
    modal run --detach rhm/directed_sculpting/full_loop/level_moves/level_moves.py::level_probe \
        --tag lv_s$s --seed $s
  done
"""

import json
import os
import time

import modal
import numpy as np

from rhm.rhm_channels import dp_cost, make_layout, possible_set_success, sample_pool
from rhm.shared import DATA_DIR, NumpyEncoder, image, volume
from rhm.directed_sculpting.full_loop.channel_env import (
    _block_state_chunked, build_block_tables, corrupt_tree, make_spec, regenerate_block,
    sample_states, train_generator_channels)


app = modal.App("rhm-ds-level-moves", image=image)


# --------------------------------------------------------------------------- #
# The move set: every (channel, level, node) the action space contains
# --------------------------------------------------------------------------- #

def build_move_set(layout, tb, device, max_level=None, lazy_twins=False):
    """Enumerate the level-indexed action space.

    One entry per (channel, level, node). A grammar channel of depth `d` contributes levels
    1..min(d, max_level); level `ell` has `s**(d-ell)` nodes, each spanning `s**(ell-1)` blocks.
    A noise channel has no grammar, so it contributes its level-1 blocks only -- keeping it in
    the action space is what makes "the value devalues the irreducible channels" still measurable
    on this axis rather than assumed away.

    `blk0`/`span` are in BLOCKS; the token span is `span * s` wide starting at `blk0 * s`.
    Returns a dict with a `moves` list plus torch-side index tensors and the per-channel rule
    tables the DP needs, all on `device`.

    `lazy_twins` adds, for every grammar node at level >= 2, a SECOND move over the identical
    token span that skips the DP and picks each block's level-1 feature independently -- the
    "lazy construction" the module docstring rejects as a level-`ell` move. It is a bad move and
    a very good CONTROL: it rewrites exactly the same positions, so it displaces the value's
    pooled latent (`z.mean(dim=1)`) by the same amount, and it differs from its twin in one
    respect only -- whether the span is one legal abstract commitment or `s**(ell-1)` independent
    guesses. Any value preference between a twin pair is therefore about the commitment and not
    about the span, with the branching-factor confound removed BY CONSTRUCTION rather than by
    regression against `d*`. Off by default, so the published move set is bit-identical.
    """
    import torch

    s = layout["s"]
    moves, rules_t = [], {}
    for ci, ch in enumerate(layout["channels"]):
        if ch["rules"] is None:                       # noise: level-1 blocks only
            for b in range(ch["blk0"], ch["blk1"]):
                moves.append({"channel": ci, "channel_name": ch["name"], "level": 1,
                              "node": b - ch["blk0"], "blk0": b, "span": 1, "kind": "noise",
                              "lazy": False})
            continue
        rules_t[ci] = [torch.from_numpy(np.ascontiguousarray(r)).to(device)
                       for r in ch["rules"]]          # root-down: rules[depth-ell] expands level ell
        top = ch["depth"] if max_level is None else min(ch["depth"], max_level)
        for ell in range(1, top + 1):
            span = s ** (ell - 1)
            for j in range((ch["blk1"] - ch["blk0"]) // span):
                base = {"channel": ci, "channel_name": ch["name"], "level": ell,
                        "node": j, "blk0": ch["blk0"] + j * span, "span": span,
                        "kind": ch["kind"]}
                moves.append({**base, "lazy": False})
                # at ell=1 the DP is empty, so the lazy twin IS the committed move -- adding one
                # would be a duplicate column, not a control.
                if lazy_twins and ell >= 2:
                    moves.append({**base, "lazy": True})
    return {
        "moves": moves,
        "n_moves": len(moves),
        "channel_of": torch.tensor([m["channel"] for m in moves], device=device),
        "level_of": torch.tensor([m["level"] for m in moves], device=device),
        "rules_t": rules_t,
        "max_level": max(m["level"] for m in moves),
        "depth_of_channel": {ci: ch["depth"] for ci, ch in enumerate(layout["channels"])},
    }


def flat_move_set(layout, tb, device):
    """The published action space as a move set: every block, level 1 only.

    `build_move_set(..., max_level=1)` is exactly the `n_blocks` level-1 moves in block order,
    so the `flat` value arm below and the ladder's `collect_value_buffer` draw from the same set.
    """
    return build_move_set(layout, tb, device, max_level=1)


def moves_table(ms, tb):
    """Human-readable (channel, level) -> n_nodes census of a move set.

    Lazy twins are counted under `<level>~` so the census stays honest about the column count.
    """
    out = {}
    for m in ms["moves"]:
        key = f"{m['level']}~" if m.get("lazy") else m["level"]
        out.setdefault(m["channel_name"], {}).setdefault(key, 0)
        out[m["channel_name"]][key] += 1
    return out


# --------------------------------------------------------------------------- #
# The move itself: one abstract commitment, rendered as a legal subtree
# --------------------------------------------------------------------------- #

def _node_features(generator, x, move, ms, tb, return_deriv=False):
    """Level-1 features for every block under `move`, via the max-sum DP over the grammar.

    Returns `(feats, pos)`: `feats` (B, span) the chosen level-1 feature per block, `pos`
    (B, span*s) the token positions the move rewrites. Every row takes the SAME move, which is
    how both callers use it (the probe loops over moves; the behaviour policy groups by move).

    With `return_deriv`, also returns `{level: (B, n_nodes_at_level)}` for the WHOLE derivation.
    `selfcheck` needs it: the only unambiguous way to verify the move is one legal abstract
    commitment is to check each parent's chosen rule against its own children, since recovering
    features by parsing tuples bottom-up is last-writer-wins at every level, not just at the leaf.
    """
    import torch

    s = tb["s"]
    blk0, span, ell = move["blk0"], move["span"], move["level"]
    batch = x.shape[0]
    blocks = torch.arange(blk0, blk0 + span, device=x.device)
    pos = (blocks[:, None] * s + torch.arange(s, device=x.device)[None, :]).reshape(-1)
    pos = pos[None, :].expand(batch, -1)

    obs = x.clone().scatter_(1, pos, torch.full_like(pos, -1))
    logits = generator.block_logits(obs)                        # (B, n_blocks, v)
    cur = logits[:, blk0:blk0 + span, :]                        # (B, span, v) -- level-1 evidence
    # ell == 1: the DP is empty, so this IS regenerate_block's choice (C1).
    # lazy twin: deliberately DON'T run the DP -- take each block's level-1 argmax independently.
    # Same positions, same span, no abstract commitment. See `build_move_set(lazy_twins=...)`.
    if ell == 1 or move.get("lazy"):
        f1 = cur.argmax(-1)                                     # == regenerate_block's choice
        return (f1, pos, {1: f1}) if return_deriv else (f1, pos)

    rules = ms["rules_t"][move["channel"]]
    depth = ms["depth_of_channel"][move["channel"]]
    v, m = tb["v"], int(tb["m_blk"][blk0])
    back = []                                                   # (table, best_r) per level, 2..ell
    for lv in range(2, ell + 1):
        table = rules[depth - lv][:, :m, :]                     # (v, m, s): level lv -> lv-1
        n_par = cur.shape[1] // s
        kids = cur.view(batch, n_par, s, v)
        total = None
        for i in range(s):
            idx = table[:, :, i].reshape(-1)                    # (v*m,)
            picked = kids[:, :, i, :].index_select(2, idx).view(batch, n_par, v, m)
            total = picked if total is None else total + picked
        cur, best_r = total.max(dim=-1)                         # (B, n_par, v) each
        back.append((table, best_r))

    feats = cur.argmax(-1)                                      # (B, 1) at the top
    deriv, lv = {ell: feats}, ell
    for table, best_r in reversed(back):
        r = best_r.gather(2, feats[..., None]).squeeze(-1)      # (B, n_par)
        kids = table[feats.reshape(-1), r.reshape(-1)]          # (B*n_par, s)
        feats = kids.view(batch, -1)
        lv -= 1
        deriv[lv] = feats
    return (feats, pos, deriv) if return_deriv else (feats, pos)


# --------------------------------------------------------------------------- #
# Hierarchical damage: an error that block-local inspection cannot see
# --------------------------------------------------------------------------- #

def corrupt_tree_hier(leaves_np, layout, tb_np, *, level, n_nodes, rng):
    """Replace `n_nodes` level-`level` tree subtrees with a legal derivation of a feature the
    observed subtree provably CANNOT derive.

    WHY THIS EXISTS. `channel_env.corrupt_tree` writes `rng.integers(0, v)` -- random symbols,
    which are off-grammar and therefore visible **block-locally**: each corrupt block looks wrong
    on its own, so a per-block argmax repairs it. That means no error in the published DGP
    *requires* abstraction to see, which is why the matched-span abstraction premium came back
    flat in level: a level-ell commitment has nothing a level-1 edit cannot also get.

    This writes the opposite kind of damage. Every block stays a legal synonym of some level-1
    feature, so nothing is locally suspicious; what is wrong is the level-`level` node, and only a
    commitment at level >= `level` can re-derive it. `level` is therefore the depth at which the
    error LIVES, and the prediction it licenses is sharp: the matched-span premium should be ~0 for
    move levels below it and positive at or above it.

    The swapped feature is chosen from the COMPLEMENT of `possible_sets` at that node, so the
    damage is guaranteed to be a real inconsistency rather than a re-rendering of the same
    feature (a different derivation of the SAME feature leaves `d*` untouched -- the trap this
    avoids).
    """
    from rhm.rhm_channels import possible_sets, tree_leaves

    v, s = layout["v"], layout["s"]
    rules = layout["tree_rules"]                       # root-down: rules[depth-lv] expands lv
    depth = layout["tree"]["depth"]
    B = leaves_np.shape[0]
    n_at_level = s ** (depth - level)
    k = min(n_nodes, n_at_level)

    # which level-`level` features can derive each node's OBSERVED subtree
    levels = possible_sets(rules, tree_leaves(layout, leaves_np), s)
    poss = levels[level - 1]                           # (B, n_at_level, v) boolean

    nodes = np.stack([rng.choice(n_at_level, size=k, replace=False) for _ in range(B)])  # (B, k)
    rowi = np.arange(B)[:, None]
    # uniform over the features the node CANNOT derive (mask out the possible ones)
    scores = rng.random((B, k, v))
    scores[poss[rowi, nodes]] = -1.0
    feats = scores.argmax(-1)[:, :, None]              # (B, k, 1) -- the wrong level-`level` feat

    # expand top-down with uniformly random rules: level -> level-1 -> ... -> 1
    for lv in range(level, 1, -1):
        table = rules[depth - lv]                      # (v, m, s)
        m = table.shape[1]
        r = rng.integers(0, m, size=feats.shape)
        feats = table[feats, r].reshape(B, k, -1)      # (B, k, n_children)

    # render each level-1 feature as a uniformly random synonym of itself
    span = feats.shape[-1]
    blk0 = layout["tree"]["blk0"] + nodes * span       # (B, k) first block of each damaged node
    blocks = blk0[:, :, None] + np.arange(span)[None, None, :]        # (B, k, span)
    mm = tb_np["m_blk"][blocks]                                        # (B, k, span)
    choice = (rng.random(mm.shape) * mm).astype(np.int64)
    tup = tb_np["syn_blk"][blocks, feats, choice]                      # (B, k, span, s)

    out = leaves_np.copy()
    pos = blocks[..., None] * s + np.arange(s)[None, None, None, :]    # (B, k, span, s)
    np.put_along_axis(out, pos.reshape(B, -1), tup.reshape(B, -1), axis=1)
    return out


def tb_numpy(tb):
    """The rendering tables `corrupt_tree_hier` needs, host-side."""
    return {k: tb[k].cpu().numpy() for k in ("m_blk", "syn_blk", "tree_blocks")}


def make_damage(layout, tb_np, level):
    """`damage(leaves, c, rng)` for a given damage level; None-equivalent at level 0."""
    if level <= 0:
        return None
    return lambda leaves, c, rng: corrupt_tree_hier(leaves, layout, tb_np, level=level,
                                                    n_nodes=c, rng=rng)


def sample_states_damaged(layout, tb, generator, *, n, n_corrupt, presteps, seed, device,
                          render="mixture", gen=None, damage=None):
    """`channel_env.sample_states` with the damage hook. `damage=None` reproduces it exactly."""
    import torch

    rng = np.random.default_rng(seed)
    pool = sample_pool(layout, n, seed)
    tree_blocks = tb["tree_blocks"].cpu().numpy()
    start = (damage(pool["leaves"], n_corrupt, rng) if damage is not None else
             corrupt_tree(pool["leaves"], tree_blocks, n_corrupt,
                          layout["v"], layout["s"], rng))
    x = torch.from_numpy(start).to(device)
    roots = torch.from_numpy(pool["roots"].astype(np.int64)).to(device)
    for _ in range(presteps):
        k = torch.randint(0, tb["n_blocks"], (n,), device=device, generator=gen)
        k = k.to(device)
        x = regenerate_block(generator, x, k, tb, render=render, gen=gen)
    return x, roots


def certify_damage(layout, tb_np, *, level, n_corrupt, n=2048, seed=11):
    """G-D: the damage is ON-GRAMMAR everywhere and still moves `d*`.

    This is the gate that makes the whole level sweep interpretable, and it is the one property
    the published damage does NOT have. Two numbers per damage mode:

      on_grammar   fraction of TREE blocks whose token tuple is a legal synonym of some level-1
                   feature. Hierarchical damage must be 1.000 -- nothing is locally suspicious,
                   so a per-block reader cannot localise the error at all. Random-symbol damage
                   sits far below 1, which is exactly why a level-1 edit can repair it.
      dstar        mean exact `d*`. Must be > 0, or there is no damage to repair.
    """
    rng = np.random.default_rng(seed)
    pool = sample_pool(layout, n, seed)
    dmg = make_damage(layout, tb_np, level)
    leaves = (dmg(pool["leaves"], n_corrupt, rng) if dmg is not None else
              corrupt_tree(pool["leaves"], tb_np["tree_blocks"], n_corrupt,
                           layout["v"], layout["s"], rng))
    s = layout["s"]
    ok, tot = 0, 0
    for blk in tb_np["tree_blocks"]:
        m = int(tb_np["m_blk"][blk])
        legal = tb_np["syn_blk"][blk, :, :m, :].reshape(-1, s)                 # (v*m, s)
        tup = leaves[:, blk * s:(blk + 1) * s]                                 # (n, s)
        ok += int((tup[:, None, :] == legal[None]).all(-1).any(-1).sum())
        tot += len(tup)
    d = dp_cost(layout, leaves, pool["roots"])
    return {"level": level, "on_grammar": ok / tot, "dstar_mean": float(d.mean()),
            "dstar_zero_frac": float((d == 0).mean())}


def regenerate_node(generator, x, move, ms, tb, *, render="mixture", gen=None, chunk=16384):
    """Apply one level-indexed move to every row of `x`. Returns a new (B, T) sequence.

    Grammar node: `_node_features` picks a legal derivation of one level-`ell` feature, then each
    block's level-1 feature is rendered through that block's CURRENT mixture -- so drift still
    bites on the dynamics exactly as it does for the published level-1 move.
    Noise block: redraw iid from the noise marginal, unchanged.
    """
    import torch

    if x.shape[0] > chunk:
        return torch.cat([regenerate_node(generator, x[i:i + chunk], move, ms, tb,
                                          render=render, gen=gen, chunk=chunk)
                          for i in range(0, x.shape[0], chunk)], dim=0)

    s, batch = tb["s"], x.shape[0]
    blk0, span = move["blk0"], move["span"]
    if move["kind"] == "noise":
        pos = (torch.arange(blk0, blk0 + span, device=x.device)[:, None] * s
               + torch.arange(s, device=x.device)[None, :]).reshape(-1)[None, :].expand(batch, -1)
        tok = torch.multinomial(tb["noise_p"], batch * span * s, replacement=True,
                                generator=gen).view(batch, span * s)
        return x.clone().scatter_(1, pos, tok)

    feats, pos = _node_features(generator, x, move, ms, tb)     # (B, span), (B, span*s)
    blocks = torch.arange(blk0, blk0 + span, device=x.device)[None, :].expand(batch, -1)
    if render == "canon":
        choice = torch.zeros(batch, span, dtype=torch.long, device=x.device)
    else:
        w = tb["w_blk"][blocks.reshape(-1), feats.reshape(-1)]  # (B*span, max_m)
        choice = torch.multinomial(w, 1, generator=gen).view(batch, span)
    tup = tb["syn_blk"][blocks.reshape(-1), feats.reshape(-1), choice.reshape(-1)]
    return x.clone().scatter_(1, pos, tup.view(batch, span * s))


def apply_moves(generator, x, move_ids, ms, tb, *, render="mixture", gen=None):
    """Row-wise different moves: group by move id and apply each group's move to its slice."""
    import torch

    out = x.clone()
    for mi in torch.unique(move_ids).tolist():
        sel = move_ids == mi
        out[sel] = regenerate_node(generator, x[sel], ms["moves"][mi], ms, tb,
                                   render=render, gen=gen)
    return out


# --------------------------------------------------------------------------- #
# Value data on a chosen move set
# --------------------------------------------------------------------------- #

def collect_value_buffer_moves(controller, generator, layout, tb, ms, *, n_episodes, batch_size,
                               n_corrupt, budget, epsilon, seed, render, gen, device,
                               damage=None):
    """`channel_env.collect_value_buffer` with the behaviour policy over an arbitrary MOVE SET.

    Identical in every other respect -- controller-greedy over every available move with
    eps-random exploration, never restricted to the tree, every visited state labelled by its
    rollout's terminal possible-set success on the tree. Passing `flat_move_set` reproduces the
    published buffer's action space exactly.

    `damage(leaves, c, rng)` overrides how start states are corrupted. None = the published
    random-symbol `corrupt_tree`. The value MUST be trained on the same damage it is probed on,
    or the arms conflate "does the value prefer abstraction" with "does a value trained on
    shallow damage generalise to deep damage".
    """
    import torch

    rng = np.random.default_rng(seed)
    tree_blocks = tb["tree_blocks"].cpu().numpy()
    configs, roots_all, succ_all = [], [], []
    done = 0
    while done < n_episodes:
        b = min(batch_size, n_episodes - done)
        done += b
        pool = sample_pool(layout, b, int(rng.integers(0, 2 ** 31)))
        c = int(rng.integers(1, n_corrupt + 1))
        start = (damage(pool["leaves"], c, rng) if damage is not None else
                 corrupt_tree(pool["leaves"], tree_blocks, c, layout["v"], layout["s"], rng))
        x = torch.from_numpy(start).to(device)
        roots = torch.from_numpy(pool["roots"].astype(np.int64)).to(device)
        traj = [x.clone()]
        for _ in range(budget):
            x = _behavior_step_moves(controller, generator, x, roots, ms, tb, epsilon,
                                     render=render, gen=gen, device=device)
            traj.append(x.clone())
        succ = torch.from_numpy(
            possible_set_success(layout, x.cpu().numpy(), pool["roots"]).astype(np.float32))
        for st in traj:
            configs.append(st.cpu())
            roots_all.append(roots.cpu())
            succ_all.append(succ)
    return torch.cat(configs), torch.cat(roots_all), torch.cat(succ_all)


def _behavior_step_moves(controller, generator, x, roots, ms, tb, epsilon, *, render, gen, device):
    """`channel_env._behavior_step` over a move set instead of over blocks."""
    import torch

    batch = x.shape[0]
    scores, props = [], []
    with torch.no_grad():
        for mv in ms["moves"]:
            p = regenerate_node(generator, x, mv, ms, tb, render=render, gen=gen)
            sc = controller.root_logits(controller.state(p)).log_softmax(-1) \
                .gather(1, roots[:, None]).squeeze(1)
            scores.append(sc)
            props.append(p)
        scores = torch.stack(scores, dim=1)
        props = torch.stack(props, dim=1)
        chosen = scores.argmax(dim=1)
        explore = torch.rand(batch, device=device) < epsilon
        chosen = torch.where(explore, torch.randint(0, ms["n_moves"], (batch,), device=device),
                             chosen)
        return props.gather(1, chosen[:, None, None].expand(-1, 1, x.shape[1])).squeeze(1)


# --------------------------------------------------------------------------- #
# The readout: |ΔV| by level, against what the level actually buys
# --------------------------------------------------------------------------- #

def level_value_probe(value, controller, generator, layout, tb, ms, x0, roots, *,
                      render, gen, device):
    """For every move in the set: the value's ΔV and the exact DP's Δd*, and what that implies.

    The level-extended form of `channel_env.value_relevance_check`. Every move is materialised,
    so nothing here depends on a forward model.

    Returned, per (channel, level) cell:
      abs_dvalue        mean |ΔV| -- the published `value_sensitivity` tap, now level-resolved
      mean_dvalue       mean signed ΔV: does the value think acting here HELPS?
      mean_dstar_gain   what a random move in that cell does to d* (usually negative -- a
                        regeneration overwrites correct tokens too)
      best_dstar_gain   what the BEST move in that cell does: the relevance a planner can cash
      residual          mean ΔV after regressing ΔV on Δd* over all TREE moves pooled.
                        POSITIVE = the value over-rates this level for what it actually buys.
      top1_share        fraction of states whose value-argmax over ALL moves lands in this cell
      dp_top1_share     the same for the exact DP -- the reference `top1_share` is read against
    """
    import torch

    roots_np = roots.cpu().numpy()
    d_cur = dp_cost(layout, x0.cpu().numpy(), roots_np)
    n, M = x0.shape[0], ms["n_moves"]
    dv = np.zeros((M, n))
    dd = np.zeros((M, n))
    with torch.no_grad():
        v_now = value(_block_state_chunked(controller, x0).mean(dim=1), roots).cpu().numpy()
        for mi, mv in enumerate(ms["moves"]):
            xk = regenerate_node(generator, x0, mv, ms, tb, render=render, gen=gen)
            dv[mi] = value(_block_state_chunked(controller, xk).mean(dim=1),
                           roots).cpu().numpy() - v_now
            dd[mi] = d_cur - dp_cost(layout, xk.cpu().numpy(), roots_np)

    names = tb["channel_names"]
    tree_c = tb["tree_channel"]
    cells = {}
    for mi, mv in enumerate(ms["moves"]):
        cells.setdefault((mv["channel"], mv["level"], bool(mv.get("lazy"))), []).append(mi)

    # Lazy twins are a PAIRED CONTROL, not rival actions: every ranking readout below is computed
    # over committed moves only, so `top1_share` / `dp_top1_share` / the calibration line / the
    # rank correlation are bit-identical to the published run when no twins are present. The twins
    # are read only through `matched_span` at the end.
    commit_mi = np.array([mi for mi, mv in enumerate(ms["moves"]) if not mv.get("lazy")])

    # calibration line, fit on TREE moves only and pooled over every level: the value's own
    # exchange rate between "d* the move buys" and "value the move is worth". The per-level
    # residual against it is the level preference with the branching-factor confound removed.
    tree_mi = [mi for mi in commit_mi if ms["moves"][mi]["channel"] == tree_c]
    X, Y = dd[tree_mi].ravel(), dv[tree_mi].ravel()
    slope, intercept = np.polyfit(X, Y, 1) if X.std() > 1e-9 else (0.0, float(Y.mean()))

    top1 = commit_mi[dv[commit_mi].argmax(axis=0)]
    dp_top1 = commit_mi[dd[commit_mi].argmax(axis=0)]
    out = {}
    for (ci, ell, lazy), mis in sorted(cells.items()):
        sub_dv, sub_dd = dv[mis], dd[mis]
        out[f"{names[ci]}|L{ell}" + ("|lazy" if lazy else "")] = {
            "channel": names[ci], "level": ell, "lazy": lazy, "n_nodes": len(mis),
            "span_blocks": ms["moves"][mis[0]]["span"],
            "abs_dvalue": float(np.abs(sub_dv).mean()),
            "mean_dvalue": float(sub_dv.mean()),
            "mean_dstar_gain": float(sub_dd.mean()),
            "best_dstar_gain": float(sub_dd.max(axis=0).mean()),
            "residual": float((sub_dv - (intercept + slope * sub_dd)).mean()),
            "top1_share": float(np.isin(top1, mis).mean()),
            "dp_top1_share": float(np.isin(dp_top1, mis).mean()),
        }

    # rank agreement between the value's ordering over all COMMITTED moves and the DP's, per state
    dvc = dv[commit_mi] - dv[commit_mi].mean(axis=0, keepdims=True)
    ddc = dd[commit_mi] - dd[commit_mi].mean(axis=0, keepdims=True)
    corr = float(np.mean((dvc * ddc).sum(0)
                         / np.maximum(np.linalg.norm(dvc, axis=0) * np.linalg.norm(ddc, axis=0),
                                      1e-9)))
    by_level = {}
    for ell in sorted({mv["level"] for mv in ms["moves"] if mv["channel"] == tree_c}):
        mis = [mi for mi, mv in enumerate(ms["moves"])
               if mv["channel"] == tree_c and mv["level"] == ell and not mv.get("lazy")]
        by_level[ell] = {"abs_dvalue": float(np.abs(dv[mis]).mean()),
                         "best_dstar_gain": float(dd[mis].max(axis=0).mean()),
                         "residual": float((dv[mis] - (intercept + slope * dd[mis])).mean()),
                         "top1_share": float(np.isin(top1, mis).mean()),
                         "dp_top1_share": float(np.isin(dp_top1, mis).mean())}

    # ---- the matched-span control ------------------------------------------------------------
    # Pair each committed level-`ell` node with its lazy twin. Both rewrite the IDENTICAL token
    # positions, so they displace `z.mean(dim=1)` equally and the span/branching-factor confound
    # cancels within the pair. `pref_rate` is a paired binary preference over (node, state): the
    # fraction of times the value ranks the abstract commitment above `s**(ell-1)` independent
    # guesses at the same place. That is level preference with the confound removed by
    # construction rather than by regression -- and `dp_pref_rate` is what it should be.
    twin = {}
    for mi, mv in enumerate(ms["moves"]):
        twin.setdefault((mv["channel"], mv["level"], mv["node"]), {})[bool(mv.get("lazy"))] = mi
    matched = {}
    pairs_by_cell = {}
    for (ci, ell, _node), d in twin.items():
        if False in d and True in d:
            pairs_by_cell.setdefault((ci, ell), []).append((d[False], d[True]))
    for (ci, ell), prs in sorted(pairs_by_cell.items()):
        cm = np.array([p[0] for p in prs])
        lz = np.array([p[1] for p in prs])
        d_dv, d_dd = dv[cm] - dv[lz], dd[cm] - dd[lz]
        # `d*` is integer-valued, so a large fraction of pairs TIE on ground truth while ΔV
        # essentially never does. Comparing a value rate over all pairs to a DP rate over all
        # pairs is then apples-to-oranges (the DP's is diluted by ties). Both rates are therefore
        # also reported on the UNTIED subset, where they measure the same thing and the value's
        # ordering can be scored against the DP's directly.
        untied = d_dd != 0
        matched[f"{names[ci]}|L{ell}"] = {
            "channel": names[ci], "level": ell, "n_pairs": int(len(prs) * dv.shape[1]),
            "span_blocks": ms["moves"][cm[0]]["span"],
            "value_premium": float(d_dv.mean()),
            "dp_premium": float(d_dd.mean()),
            "pref_rate": float((d_dv > 0).mean()),
            "dp_pref_rate": float((d_dd > 0).mean()),
            "tie_rate_dp": float((d_dd == 0).mean()),
            # like-for-like: both rates on the pairs where ground truth actually discriminates
            "n_untied": int(untied.sum()),
            "pref_rate_untied": (float((d_dv[untied] > 0).mean()) if untied.any() else None),
            "dp_pref_rate_untied": (float((d_dd[untied] > 0).mean()) if untied.any() else None),
            # does the value AGREE with the DP pair by pair, where the DP has an opinion?
            "agree_untied": (float((np.sign(d_dv[untied]) == np.sign(d_dd[untied])).mean())
                             if untied.any() else None),
            "abs_dvalue_commit": float(np.abs(dv[cm]).mean()),
            "abs_dvalue_lazy": float(np.abs(dv[lz]).mean()),
        }
    return {"cells": out, "tree_by_level": by_level, "matched_span": matched,
            "value_vs_dp_rank_corr": corr,
            "calibration": {"slope": float(slope), "intercept": float(intercept)},
            "dstar_mean": float(d_cur.mean()),
            "tree_top1_share": float(np.isin(top1, tree_mi).mean()),
            "dp_tree_top1_share": float(np.isin(dp_top1, tree_mi).mean())}


def print_probe(res, title):
    """The three tables this experiment exists to produce."""
    print(f"\n{'=' * 96}\n=== {title}\n{'=' * 96}")
    print(f"{'cell':16s} {'nodes':>5s} {'span':>5s} {'|ΔV|':>9s} {'ΔV':>10s} "
          f"{'mean Δd*':>10s} {'best Δd*':>10s} {'resid':>10s} {'top1':>8s} {'DP top1':>8s}")
    for key, c in res["cells"].items():
        print(f"{key:16s} {c['n_nodes']:>5d} {c['span_blocks']:>5d} {c['abs_dvalue']:>9.4f} "
              f"{c['mean_dvalue']:>+10.4f} {c['mean_dstar_gain']:>+10.4f} "
              f"{c['best_dstar_gain']:>+10.4f} {c['residual']:>+10.4f} "
              f"{c['top1_share']:>8.3f} {c['dp_top1_share']:>8.3f}")
    print(f"\n  d* mean {res['dstar_mean']:.3f} | value-vs-DP rank corr over ALL moves "
          f"{res['value_vs_dp_rank_corr']:+.3f} | top-1 is a tree move: value "
          f"{res['tree_top1_share']:.3f} vs DP {res['dp_tree_top1_share']:.3f}")
    print(f"  calibration ΔV = {res['calibration']['intercept']:+.4f} "
          f"{res['calibration']['slope']:+.4f} * Δd*  (fit on tree moves, all levels pooled)")
    if res.get("matched_span"):
        print(f"\n  MATCHED-SPAN CONTROL -- committed level-ℓ move vs its LAZY twin at the same "
              f"node.\n  Identical tokens rewritten, so the branching-factor confound cancels "
              f"inside each pair.\n  `pref` = fraction of (node, state) pairs where the "
              f"commitment is ranked above the guesses.")
        print(f"  {'cell':14s} {'span':>4s} {'pairs':>7s} {'ΔV prem':>9s} {'Δd* prem':>9s} "
              f"{'DP tie':>7s} | untied: {'n':>6s} {'value':>7s} {'DP':>7s} {'agree':>7s}")
        for key, c in res["matched_span"].items():
            u = "" if c["pref_rate_untied"] is None else (
                f"{c['n_untied']:>6d} {c['pref_rate_untied']:>7.3f} "
                f"{c['dp_pref_rate_untied']:>7.3f} {c['agree_untied']:>7.3f}")
            print(f"  {key:14s} {c['span_blocks']:>4d} {c['n_pairs']:>7d} "
                  f"{c['value_premium']:>+9.4f} {c['dp_premium']:>+9.4f} "
                  f"{c['tie_rate_dp']:>7.3f} |          {u}")


# --------------------------------------------------------------------------- #
# Self-check: the level-1 rung must be the published move
# --------------------------------------------------------------------------- #

def selfcheck(v=8, s=2, tree_depth=4, seed=1, n=256):
    """Assert the new operator reduces to `regenerate_block` at level 1, and stays grammatical.

    Two properties, both cheap and both load-bearing for reading the tables against published
    numbers:
      C1  at `ell=1` the feature chosen by `_node_features` is IDENTICAL to `regenerate_block`'s,
          so the level-1 row is the published move rather than a lookalike;
      C2  at every level the rendered span PARSES BACK to a single level-`ell` feature under the
          channel's own tables -- i.e. the move really is one abstract commitment.
    """
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    from rhm.rhm_generative_planner import _build_generator

    spec = make_spec(tree_depth=tree_depth, struct_depths=[2, 2], struct_ms=[2, 4],
                     noise_blocks=[1, 1])
    layout = make_layout(v, s, spec)
    tb = build_block_tables(layout, device)
    ms = build_move_set(layout, tb, device)
    print(f"move set: {ms['n_moves']} moves over {tb['n_blocks']} blocks -- "
          + json.dumps(moves_table(ms, tb)))

    gen = torch.Generator(device=device).manual_seed(seed)
    Generator = _build_generator()
    generator = Generator(v, layout["total_len"], s, 96, n_head=4, n_layer=2,
                          root_conditioned=False).to(device)
    pool = sample_pool(layout, 4000, seed)
    train_generator_channels(generator, torch.from_numpy(pool["leaves"]), tb, batch_size=128,
                             n_steps=300, lr=3e-4, device=device)
    generator.eval()
    for p in generator.parameters():
        p.requires_grad_(False)

    x = torch.from_numpy(sample_pool(layout, n, seed + 5)["leaves"]).to(device)
    powers = tb["powers"]

    # C1 -- level-1 identity with the published move
    bad = 0
    for mv in ms["moves"]:
        if mv["level"] != 1 or mv["kind"] == "noise":
            continue
        k = torch.full((n,), mv["blk0"], device=device, dtype=torch.long)
        obs = x.clone().scatter_(1, k[:, None] * s + torch.arange(s, device=device)[None, :],
                                 torch.full((n, s), -1, device=device, dtype=x.dtype))
        with torch.no_grad():
            ref = generator.block_logits(obs).gather(
                1, k[:, None, None].expand(-1, 1, v)).squeeze(1).argmax(-1)
            got, _ = _node_features(generator, x, mv, ms, tb)
        bad += int((ref != got.squeeze(1)).sum())
    assert bad == 0, f"C1 FAILED: level-1 move differs from regenerate_block on {bad} rows"
    print("C1 PASS: at level 1 the operator reproduces `regenerate_block`'s feature exactly")

    # C2 -- the move is ONE abstract commitment, rendered legally.
    #
    # Checked on the DP's chosen features directly, NOT by parsing the rendered tokens back
    # through `bottom_blk`. That map is last-writer-wins whenever two level-1 features share a
    # leaf tuple (`build_inverse_maps`' documented ambiguity, and `generate_rules_distinct`
    # produces it), so a parse-back check fails on legal moves and would be testing the inverse
    # map rather than the operator. The ambiguity rate is reported so the choice is visible.
    amb = 0
    for ch in layout["channels"]:
        if ch["rules"] is None:
            continue
        codes = (ch["rules"][-1] * (v ** np.arange(s))).sum(-1).ravel()
        amb += len(codes) - len(set(int(c) for c in codes))
    print(f"  (leaf-tuple collisions across features, summed over channels: {amb} -- why C2 "
          f"does not parse back through `bottom_blk`)")

    for mv in ms["moves"]:
        if mv["kind"] == "noise":
            continue
        with torch.no_grad():
            feats, _, deriv = _node_features(generator, x, mv, ms, tb, return_deriv=True)
            xk = regenerate_node(generator, x, mv, ms, tb, render="mixture", gen=gen)
        rules = layout["channels"][mv["channel"]]["rules"]
        depth = layout["channels"][mv["channel"]]["depth"]

        # C2a: the derivation is ONE level-ell feature expanded by legal rules all the way down
        assert deriv[mv["level"]].shape == (n, 1), f"C2a FAILED: {mv} is not a single commitment"
        assert feats.shape == (n, mv["span"]), f"C2a FAILED: bad feature shape for {mv}"
        for lv in range(mv["level"], 1, -1):
            table = rules[depth - lv]                                  # (v, m, s): lv -> lv-1
            par = deriv[lv].cpu().numpy()                              # (n, n_par)
            kid = deriv[lv - 1].cpu().numpy().reshape(n, par.shape[1], s)
            hit = (table[par] == kid[:, :, None, :]).all(-1).any(-1)   # exists r: rule matches
            assert bool(hit.all()), (f"C2a FAILED: level {lv}->{lv - 1} expansion is not a legal "
                                     f"rule for {mv}")

        # C2b: every block's rendered tuple is one of the m synonyms of its chosen feature
        blocks = torch.arange(mv["blk0"], mv["blk0"] + mv["span"], device=device)
        tup = xk.view(n, tb["n_blocks"], s)[:, blocks, :]                      # (n, span, s)
        mm = int(tb["m_blk"][mv["blk0"]])
        syn = tb["syn_blk"][blocks][None].expand(n, -1, -1, -1, -1)            # (n,span,v,max_m,s)
        mine = syn.gather(2, feats[:, :, None, None, None]
                          .expand(-1, -1, 1, tb["max_m"], s)).squeeze(2)       # (n,span,max_m,s)
        ok = (mine[:, :, :mm, :] == tup[:, :, None, :]).all(-1).any(-1)        # (n, span)
        assert bool(ok.all()), f"C2b FAILED: rendered tuple is not a synonym of the choice, {mv}"
    print("C2 PASS: every level-ell move commits to ONE level-ell feature (C2a) and renders "
          "each block as a legal synonym of it (C2b)")

    # C3 -- the matched-span control is span-matched AND non-vacuous.
    #
    # The whole point of the lazy twin is that it cancels the branching-factor confound within a
    # pair, so (a) it must rewrite the IDENTICAL positions, and (b) it must actually DIFFER from
    # its committed twin often enough for the comparison to carry information. (b) is the one that
    # could silently fail: if the generator's per-block argmax already happened to be a legal
    # subtree everywhere, the control would be a tautology and every `pref_rate` would be 0.5 by
    # construction. The rates are printed, not just asserted, so the gate is visible.
    ms_lz = build_move_set(layout, tb, device, lazy_twins=True)
    twin_of = {(m["channel"], m["level"], m["node"]): i for i, m in enumerate(ms_lz["moves"])
               if not m.get("lazy")}
    n_pairs, diff_rows, diff_blocks, tot_blocks, illegal_l2, tot_l2 = 0, 0, 0, 0, 0, 0
    for mv in ms_lz["moves"]:
        if not mv.get("lazy"):
            continue
        cm = ms_lz["moves"][twin_of[(mv["channel"], mv["level"], mv["node"])]]
        with torch.no_grad():
            f_lz, pos_lz = _node_features(generator, x, mv, ms_lz, tb)
            f_cm, pos_cm = _node_features(generator, x, cm, ms_lz, tb)
            # the lazy features must BE the independent per-block argmax
            blocks = torch.arange(mv["blk0"], mv["blk0"] + mv["span"], device=device)
            obs = x.clone().scatter_(1, pos_lz, torch.full_like(pos_lz, -1))
            ref = generator.block_logits(obs)[:, blocks, :].argmax(-1)
        assert bool((pos_lz == pos_cm).all()), f"C3 FAILED: twin spans differ for {mv}"
        assert bool((f_lz == ref).all()), f"C3 FAILED: lazy twin is not the per-block argmax, {mv}"
        n_pairs += 1
        diff_blocks += int((f_lz != f_cm).sum())
        tot_blocks += f_lz.numel()
        diff_rows += int((f_lz != f_cm).any(dim=1).sum())
        if mv["level"] == 2:                      # cheap legality check at the smallest deep level
            rules = layout["channels"][mv["channel"]]["rules"]
            table = rules[layout["channels"][mv["channel"]]["depth"] - 2]      # (v, m, s)
            kid = f_lz.cpu().numpy()                                          # (n, 2)
            hit = (table[None] == kid[:, None, None, :]).all(-1).any(-1).any(-1)
            illegal_l2 += int((~hit).sum())
            tot_l2 += len(kid)
    assert n_pairs > 0, "C3 FAILED: lazy_twins produced no pairs"
    assert diff_rows > 0, ("C3 FAILED: the lazy twin never differs from its committed twin -- the "
                           "matched-span control would be vacuous")
    print(f"C3 PASS: {n_pairs} twin pairs, spans identical; the lazy twin differs from the "
          f"commitment on {diff_rows / (n_pairs * n) * 100:.1f}% of rows "
          f"({diff_blocks / max(tot_blocks, 1) * 100:.1f}% of blocks)")
    if tot_l2:
        print(f"  at level 2, the lazy span is NOT a legal level-2 subtree on "
              f"{illegal_l2 / tot_l2 * 100:.1f}% of rows -- which is what makes it a control "
              f"rather than a rival move")
    print("\nselfcheck OK")


# --------------------------------------------------------------------------- #
# The experiment
# --------------------------------------------------------------------------- #

@app.function(image=image, timeout=1800, memory=16384)
def selfcheck_remote(tree_depth: int = 4):
    """`selfcheck` on Modal -- the repo has no local torch, and C1/C2 are the gate for the run."""
    selfcheck(tree_depth=tree_depth)
    return True


@app.function(image=image, timeout=1800, memory=16384)
def certify_damage_remote(v: int = 8, s: int = 2, tree_depth: int = 4, n_corrupt: int = 3,
                          struct_depths: str = "2,2", levels: str = "0,1,2,3"):
    """G-D across damage levels -- the DGP gate, CPU only, no training involved.

    `on_grammar` is the whole point: the published damage (level 0) is off-grammar and so can be
    localised one block at a time; hierarchical damage is 1.000 on-grammar at every level, so the
    only way to see the error is to hold a hypothesis about a level-`ell` node.
    """
    import torch

    device = torch.device("cpu")
    spec = make_spec(tree_depth=tree_depth,
                     struct_depths=[int(x) for x in struct_depths.split(",")],
                     struct_ms=[2, 4], noise_blocks=[1, 1])
    layout = make_layout(v, s, spec)
    tb_np = tb_numpy(build_block_tables(layout, device))
    print(f"{'damage':>7s} {'on-grammar':>11s} {'d* mean':>9s} {'d*==0':>8s}")
    out = []
    for lv in [int(x) for x in levels.split(",")]:
        c = certify_damage(layout, tb_np, level=lv, n_corrupt=n_corrupt, n=4096)
        out.append(c)
        print(f"{lv:>7d} {c['on_grammar']:>11.4f} {c['dstar_mean']:>9.3f} "
              f"{c['dstar_zero_frac']:>8.4f}")
    return out


@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=14400, memory=32768)
def level_probe(v: int = 8, s: int = 2, seed: int = 1, state_dim: int = 96,
                tree_depth: int = 4, struct_depths: str = "2,2", struct_ms: str = "2,4",
                noise_blocks: str = "1,1", struct_shares: str = "0,0", share_mode: str = "top",
                controller_steps: int = 12_000, generator_steps: int = 12_000,
                value_steps: int = 12_000, value_episodes: int = 40_000,
                batch_size: int = 256, n_corrupt: int = 3, edit_budget: int = 6,
                explore_eps: float = 0.3, n_eval: int = 512,
                arms: str = "flat,level", lazy_twins: bool = False, damage_level: int = 0,
                tag: str = "v1", quick: bool = False):
    """Train the value on each action space, then probe BOTH with the level-indexed move set.

    Controller and generator are trained once and shared, so the arms differ in exactly one
    variable: which moves the value's Monte-Carlo behaviour policy could take.
    """
    import torch

    from rhm.rhm_edit_control import _train_edit_controller
    from rhm.rhm_generative_planner import _build_generator
    from rhm.rhm_latent_planner import _build_value_head, _train_value_mc
    from rhm.rhm_sculpt_latent import _build_rich_controller

    if quick:
        controller_steps = generator_steps = value_steps = 600
        value_episodes, n_eval = 4_000, 128

    arm_list = [a for a in arms.split(",") if a]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_float32_matmul_precision("high")
    started = time.time()

    spec = make_spec(tree_depth=tree_depth,
                     struct_depths=[int(x) for x in struct_depths.split(",")],
                     struct_ms=[int(x) for x in struct_ms.split(",")],
                     noise_blocks=[int(x) for x in noise_blocks.split(",")],
                     struct_shares=[int(x) for x in struct_shares.split(",")],
                     share_mode=share_mode)
    layout = make_layout(v, s, spec)
    tb = build_block_tables(layout, device)
    T, n_blocks = layout["total_len"], tb["n_blocks"]
    render, gen = "mixture", torch.Generator(device=device).manual_seed(seed)

    ms_level = build_move_set(layout, tb, device)
    ms_flat = flat_move_set(layout, tb, device)
    # The twins are a paired CONTROL and never an action: no arm ever trains on them, so the two
    # value arms are exactly the published ones and only the probe's column set grows.
    ms_probe = (build_move_set(layout, tb, device, lazy_twins=True) if lazy_twins else ms_level)
    print(f"Level-indexed action space on the distractor DGP: T={T} tokens / {n_blocks} blocks")
    for ch in layout["channels"]:
        print(f"  {ch['name']:9s} {ch['kind']:6s} blocks[{ch['blk0']:2d}:{ch['blk1']:2d}]"
              + (f" depth={ch['depth']} m={ch['m']}" if ch["rules"] is not None else ""))
    print(f"  flat  move set: {ms_flat['n_moves']:3d} moves  {json.dumps(moves_table(ms_flat, tb))}")
    print(f"  level move set: {ms_level['n_moves']:3d} moves  "
          f"{json.dumps(moves_table(ms_level, tb))}")
    if lazy_twins:
        print(f"  PROBE move set: {ms_probe['n_moves']:3d} moves  "
              f"{json.dumps(moves_table(ms_probe, tb))}   (`~` = lazy twin, control only)")

    # ---- shared frozen instruments ------------------------------------------------------
    pool = sample_pool(layout, 20_000 if quick else 100_000, seed)
    train_leaves = torch.from_numpy(pool["leaves"])
    train_roots = torch.from_numpy(pool["roots"].astype(np.int64))
    Controller, Generator = _build_rich_controller(), _build_generator()
    ValueHead = _build_value_head()

    controller = Controller(v, T, s, state_dim, n_head=4, n_layer=2).to(device)
    _train_edit_controller(controller, train_leaves, train_roots, batch_size=batch_size,
                           n_blocks=n_blocks, block_size=s, n_steps=controller_steps,
                           lr=3e-4, device=device, p_full=0.5)
    generator = Generator(v, T, s, state_dim, n_head=4, n_layer=2,
                          root_conditioned=False).to(device)
    train_generator_channels(generator, train_leaves, tb, batch_size=batch_size,
                             n_steps=generator_steps, lr=3e-4, device=device)
    for mod in (controller, generator):
        mod.eval()
        for p in mod.parameters():
            p.requires_grad_(False)
    del train_leaves, train_roots, pool

    # the damage model. level 0 = the published random-symbol corruption; level >= 1 swaps a
    # level-`damage_level` subtree for a legal derivation of a feature it cannot produce, so the
    # error is invisible block-locally and only a commitment at that level or above can see it.
    tb_np = tb_numpy(tb)
    damage = make_damage(layout, tb_np, damage_level)
    cert = certify_damage(layout, tb_np, level=damage_level, n_corrupt=n_corrupt,
                          n=1024 if quick else 4096)
    print(f"  damage_level={damage_level}: tree blocks on-grammar {cert['on_grammar']:.4f}, "
          f"d* mean {cert['dstar_mean']:.3f}, d*==0 on {cert['dstar_zero_frac']:.4f} of rows")
    if damage_level >= 1:
        assert cert["on_grammar"] > 0.999, (
            f"G-D FAILED: hierarchical damage left {1 - cert['on_grammar']:.4f} of tree blocks "
            f"off-grammar, so the error is still visible block-locally")
        assert cert["dstar_zero_frac"] < 0.05, (
            f"G-D FAILED: {cert['dstar_zero_frac']:.3f} of rows have d*==0, i.e. no damage")

    # one frozen probe set, shared by every arm -- the arms must be read on identical states
    x_eval, r_eval = sample_states_damaged(layout, tb, generator, n=n_eval, n_corrupt=n_corrupt,
                                           presteps=0, seed=seed + 42, device=device,
                                           render=render, gen=gen, damage=damage)

    out = {}
    for arm in arm_list:
        ms_train = ms_flat if arm == "flat" else ms_level
        print(f"\n{'#' * 72}\n# ARM {arm}: value trained on {ms_train['n_moves']} moves "
              f"({value_episodes} rollouts)\n{'#' * 72}")
        cfg, rts, suc = collect_value_buffer_moves(
            controller, generator, layout, tb, ms_train, n_episodes=value_episodes,
            batch_size=1024, n_corrupt=n_corrupt, budget=edit_budget, epsilon=explore_eps,
            seed=seed + 31, render=render, gen=gen, device=device, damage=damage)
        print(f"  buffer {cfg.shape[0]} states, terminal success {suc.mean().item():.3f}")
        value = ValueHead(state_dim, v).to(device)
        _train_value_mc(value, controller, cfg, rts, suc, batch_size=512, n_steps=value_steps,
                        lr=3e-4, device=device)
        value.eval()
        for p in value.parameters():
            p.requires_grad_(False)
        terminal_success = float(suc.mean())
        del cfg, rts, suc

        res = level_value_probe(value, controller, generator, layout, tb, ms_probe,
                                x_eval, r_eval, render=render, gen=gen, device=device)
        res["terminal_success"] = terminal_success
        res["train_move_set"] = {"n_moves": ms_train["n_moves"],
                                 "census": moves_table(ms_train, tb)}
        print_probe(res, f"ARM {arm} -- value trained on the {arm} action space, "
                         f"probed on all {ms_level['n_moves']} level moves")
        out[arm] = res

    metrics = {
        "config": {"v": v, "s": s, "seed": seed, "tree_depth": tree_depth,
                   "struct_depths": struct_depths, "struct_ms": struct_ms,
                   "noise_blocks": noise_blocks, "struct_shares": struct_shares,
                   "share_mode": share_mode, "state_dim": state_dim,
                   "value_episodes": value_episodes, "edit_budget": edit_budget,
                   "n_corrupt": n_corrupt, "explore_eps": explore_eps, "n_eval": n_eval,
                   "arms": arm_list, "lazy_twins": lazy_twins, "damage_level": damage_level,
                   "quick": quick},
        "damage_certification": cert,
        "channels": [{k: ch[k] for k in ("name", "kind", "depth", "m", "blk0", "blk1")}
                     for ch in layout["channels"]],
        # what the probe's cells refer to; identical to the training set when lazy_twins is off
        "move_set": {"n_moves": ms_probe["n_moves"], "census": moves_table(ms_probe, tb),
                     "moves": ms_probe["moves"]},
        "arms": out,
        "elapsed_seconds": time.time() - started,
    }
    out_dir = f"{DATA_DIR}/directed_sculpting/level_moves/level_{tag}"
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/results.json", "w") as f:
        json.dump(metrics, f, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"\nwrote {out_dir}/results.json ({metrics['elapsed_seconds']:.0f}s)")
    return metrics
