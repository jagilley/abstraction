"""Primitives for the crystallize node: a level-indexed action space, hierarchical
damage, and an exact DP oracle — all on the SINGLE-CHANNEL sculpting grammar.

Everything here is a port of machinery that already exists in the repo, moved off the
`directed_sculpting/full_loop/channel_env.py` layout (channels + drift + distractors)
and onto the plain `rhm_data.generate_rules_distinct` grammar that `rhm_sculpt_latent.py`
(Stage 3b) uses. Nothing upstream is modified.

  * `build_move_set` / `node_features` / `apply_move` port
    `full_loop/level_moves/level_moves.py`'s `regenerate_node` operator: a move commits to
    ONE level-`ell` feature and renders the whole legal subtree beneath it. At level 1 the
    DP is empty and the operator's feature choice is IDENTICAL to
    `rhm_generative_planner._regenerate`'s, so the level-1 rung of every readout is
    comparable to published Stage-3a/3b numbers (gate C1, asserted in `selfcheck`).

  * `corrupt_hier` ports `level_moves.corrupt_tree_hier`: a level-k subtree replaced by a
    legal derivation of a feature the observed subtree provably CANNOT produce. 100%
    on-grammar, so nothing is locally suspicious and only a commitment at level >= k can
    re-derive it. The published `_corrupt` writes random symbols, which at (v=8, s=2, m=2)
    is off-grammar for ~75% of blocks and therefore repairable one block at a time — which
    would let abstraction never matter.

  * `oracle_rollout` is the exact-DP greedy over the same move set: at each step
    materialise every move and take the one that most reduces `d*`. This is the FLOOR
    reference the certificate is graded against (the etude's "ceiling FM" analog), and it
    is available here only because RHM hands us an exact planner.
"""

import numpy as np

from rhm.rhm_sculpt_precheck import nearest_derivation_cost, parse_success_and_heuristic, possible_sets


# --------------------------------------------------------------------------- #
# The level-indexed action space
# --------------------------------------------------------------------------- #

def build_move_set(depth, s, max_level=None, with_noop=False):
    """Every (level, node) of the tree. Level `ell` has s**(depth-ell) nodes, each spanning
    s**(ell-1) blocks. `blk0`/`span` are in BLOCKS; the token span is `span*s` wide.

    At L=4, s=2 this is 8 + 4 + 2 + 1 = 15 moves. `max_level=1` gives exactly the published
    flat action space (one move per block, in block order)."""
    top = depth if max_level is None else min(depth, max_level)
    moves = []
    for ell in range(1, top + 1):
        span = s ** (ell - 1)
        for j in range(s ** (depth - ell)):
            moves.append({"level": ell, "node": j, "blk0": j * span, "span": span,
                          "name": f"L{ell}n{j}"})
    if with_noop:
        # a NOOP lets a policy spend less than the full budget -- without it every realisation
        # is exactly `budget` moves long and a short committed program is inexpressible.
        moves.append({"level": 0, "node": 0, "blk0": 0, "span": 0, "name": "NOOP"})
    return moves


def node_features(generator, x, move, rules_t, depth, v, m, s, lazy=False):
    """Level-1 features for every block under `move`, via a max-sum DP up the grammar.

    Returns `(feats (B, span), pos (B, span*s))`. `lazy=True` skips the DP and takes each
    block's level-1 argmax independently — the "lazy twin" control: identical token span
    rewritten, no abstract commitment.
    """
    import torch

    blk0, span, ell = move["blk0"], move["span"], move["level"]
    batch = x.shape[0]
    blocks = torch.arange(blk0, blk0 + span, device=x.device)
    pos = (blocks[:, None] * s + torch.arange(s, device=x.device)[None, :]).reshape(-1)
    pos = pos[None, :].expand(batch, -1).contiguous()

    obs = x.clone().scatter_(1, pos, torch.full_like(pos, -1))
    logits = generator.block_logits(obs)                     # (B, n_blocks, v)
    cur = logits[:, blk0:blk0 + span, :]                     # (B, span, v) level-1 evidence
    if ell == 1 or lazy:
        return cur.argmax(-1), pos

    back = []
    for lv in range(2, ell + 1):
        table = rules_t[depth - lv]                          # (v, m, s): level lv -> lv-1
        n_par = cur.shape[1] // s
        kids = cur.view(batch, n_par, s, v)
        total = None
        for i in range(s):
            idx = table[:, :, i].reshape(-1)                 # (v*m,)
            picked = kids[:, :, i, :].index_select(2, idx).view(batch, n_par, v, m)
            total = picked if total is None else total + picked
        cur, best_r = total.max(dim=-1)                      # (B, n_par, v) each
        back.append((table, best_r))

    feats = cur.argmax(-1)                                   # (B, 1) at the top
    for table, best_r in reversed(back):
        r = best_r.gather(2, feats[..., None]).squeeze(-1)   # (B, n_par)
        kids = table[feats.reshape(-1), r.reshape(-1)]       # (B*n_par, s)
        feats = kids.view(batch, -1)
    return feats, pos


def apply_move(generator, x, move, rules_t, canon, depth, v, m, s, lazy=False, chunk=16384):
    """Materialise one move on every row of `x` (canonical rendering, matching
    `_regenerate(..., sample=False)`)."""
    import torch
    if move["level"] == 0:
        return x.clone()
    if x.shape[0] > chunk:
        return torch.cat([apply_move(generator, x[i:i + chunk], move, rules_t, canon,
                                     depth, v, m, s, lazy=lazy, chunk=chunk)
                          for i in range(0, x.shape[0], chunk)], dim=0)
    feats, pos = node_features(generator, x, move, rules_t, depth, v, m, s, lazy=lazy)
    tup = canon[feats]                                       # (B, span, s)
    new = x.clone()
    new.scatter_(1, pos, tup.reshape(x.shape[0], -1))
    return new


def apply_sequence(generator, x, seq, ms, rules_t, canon, depth, v, m, s):
    """Execute a fixed move sequence OPEN-LOOP (no re-encoding, no value, no search).
    `seq` is a list/array of move indices. Returns (x_final, n_materialisations)."""
    n_mat = 0
    for k in seq:
        x = apply_move(generator, x, ms[int(k)], rules_t, canon, depth, v, m, s)
        n_mat += x.shape[0]
    return x, n_mat


# --------------------------------------------------------------------------- #
# Hierarchical damage
# --------------------------------------------------------------------------- #

def corrupt_hier(leaves_np, rules, depth, v, m, s, level, nodes, rng):
    """Replace the given level-`level` nodes with a legal derivation of a feature the
    observed subtree provably CANNOT produce. Every block stays on-grammar.

    `nodes` is an explicit list of node indices at that level (0 .. s**(depth-level)-1) —
    this node fixes them so a context recurs and its benchmark is estimable."""
    batch = leaves_np.shape[0]
    nodes = np.asarray(nodes, dtype=np.int64)
    k = len(nodes)
    levels = possible_sets(rules, leaves_np, s)          # bottom..root; levels[i] == level i+1
    poss = levels[level - 1][:, nodes, :]                # (B, k, v) what each node CAN derive

    scores = rng.random((batch, k, v))
    scores[poss] = -1.0                                  # uniform over what it CANNOT derive
    feats = scores.argmax(-1)[:, :, None]                # (B, k, 1)

    for lv in range(level, 1, -1):                       # expand top-down with random rules
        table = rules[depth - lv]                        # (v, m, s)
        r = rng.integers(0, table.shape[1], size=feats.shape)
        feats = table[feats, r].reshape(batch, k, -1)

    bottom = rules[depth - 1]                            # (v, m, s) level-1 feature -> leaf tuple
    choice = rng.integers(0, bottom.shape[1], size=feats.shape)
    tup = bottom[feats, choice]                          # (B, k, span, s)

    span = feats.shape[-1]
    blk0 = nodes * span                                  # (k,) first block of each damaged node
    blocks = blk0[None, :, None] + np.arange(span)[None, None, :]      # (1, k, span)
    blocks = np.broadcast_to(blocks, (batch, k, span))
    pos = blocks[..., None] * s + np.arange(s)[None, None, None, :]    # (B, k, span, s)
    out = leaves_np.copy()
    np.put_along_axis(out, pos.reshape(batch, -1), tup.reshape(batch, -1), axis=1)
    return out


def on_grammar_rate(leaves_np, inverse_bottom, v, s):
    """Fraction of blocks whose leaf tuple is a legal synonym of some level-1 feature."""
    batch = leaves_np.shape[0]
    n_blocks = leaves_np.shape[1] // s
    powers = v ** np.arange(s)
    codes = (leaves_np.reshape(batch, n_blocks, s) * powers).sum(-1)
    return float((inverse_bottom[codes] >= 0).mean())


# --------------------------------------------------------------------------- #
# The exact-DP oracle over the same move set (the FLOOR reference)
# --------------------------------------------------------------------------- #

def oracle_rollout(generator, x, roots_np, ms, rules, rules_t, canon, depth, v, m, s, budget):
    """Greedy over the EXACT DP: at each step materialise every move and take the one that
    most reduces `d*`. Ties broken by move order. Returns (x_final, n_mat)."""
    import torch
    n_mat = 0
    for _ in range(budget):
        best_x, best_d = None, None
        for move in ms:
            xk = apply_move(generator, x, move, rules_t, canon, depth, v, m, s)
            n_mat += x.shape[0]
            dk = nearest_derivation_cost(rules, xk.cpu().numpy(), roots_np, s)
            if best_d is None:
                best_x, best_d = xk, dk
            else:
                take = torch.from_numpy(dk < best_d).to(x.device)
                best_x = torch.where(take[:, None], xk, best_x)
                best_d = np.minimum(best_d, dk)
        x = best_x
    return x, n_mat


def grade(x_np, roots_np, rules, s):
    """(success rate, mean residual d*). Success = r* in the root's possible-set (exact);
    residual d* = exact min token edits still needed (an oracle readout)."""
    succ, _ = parse_success_and_heuristic(rules, x_np, roots_np, s)
    dres = nearest_derivation_cost(rules, x_np, roots_np, s)
    return succ.astype(np.float64), dres.astype(np.float64)
