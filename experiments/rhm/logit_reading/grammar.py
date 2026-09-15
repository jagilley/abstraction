"""Weighted-synonym RHM: rule weights and vectorised generation with latents.

The canonical RHM draws each node's synonym uniformly (1/m). Part 2 needs tokens that
are LEGAL but RARE, so the trajectory runs draw each (level, feature)'s m synonym
weights from Dirichlet(alpha). Rule tables are the usual `generate_rules_distinct`
draw, so grammaticality (which tuples are legal) is identical to the uniform regime;
only frequencies change. alpha -> inf recovers the uniform RHM.
"""

import numpy as np


def synonym_weights(v, L, m, alpha, seed):
    """list of L (v, m) arrays, rows ~ Dirichlet(alpha)."""
    rng = np.random.default_rng(seed)
    return [rng.dirichlet(np.full(m, float(alpha)), size=v) for _ in range(L)]


def generate(rules, rule_w, n, rng, chunk=100_000, return_latents=False):
    """n aligned sequences under (rules, rule_w); rule_w=None is the uniform RHM.

    Returns leaves (n, s^L) int64, and if return_latents also
      feats[d] (n, s^d) int64 for tree levels d = 0..L (L = leaves)
      rcs[d]   (n, s^d) int64 synonym index used at each internal node, d = 0..L-1
    """
    L = len(rules)
    v, m, s = rules[0].shape
    out_leaves = []
    out_feats = [[] for _ in range(L + 1)]
    out_rcs = [[] for _ in range(L)]
    cws = None if rule_w is None else [np.cumsum(w, 1) for w in rule_w]
    for c0 in range(0, n, chunk):
        nc = min(chunk, n - c0)
        cur = rng.integers(0, v, size=(nc, 1))
        if return_latents:
            out_feats[0].append(cur)
        for d in range(L):
            nn_ = cur.shape[1]
            if cws is None:
                rc = rng.integers(0, m, size=(nc, nn_))
            else:
                u = rng.random((nc, nn_))
                rc = np.minimum((u[..., None] >= cws[d][cur]).sum(-1), m - 1)
            cur = rules[d][cur, rc].reshape(nc, nn_ * s)
            if return_latents:
                out_rcs[d].append(rc)
                out_feats[d + 1].append(cur)
        out_leaves.append(cur)
    leaves = np.concatenate(out_leaves)
    if not return_latents:
        return leaves
    return (leaves, [np.concatenate(f) for f in out_feats],
            [np.concatenate(r) for r in out_rcs])


def realise(rules, rule_w, feat, d, rng):
    """Sample a fresh subtree under a level-d node with feature `feat` (n,) -> leaves
    (n, s^(L-d)) plus its latents, drawn from the (weighted) grammar."""
    L = len(rules)
    v, m, s = rules[0].shape
    cur = np.asarray(feat)[:, None]
    for dd in range(d, L):
        nn_ = cur.shape[1]
        if rule_w is None:
            rc = rng.integers(0, m, size=cur.shape)
        else:
            cw = np.cumsum(rule_w[dd], 1)
            rc = np.minimum((rng.random(cur.shape)[..., None] >= cw[cur]).sum(-1), m - 1)
        cur = rules[dd][cur, rc].reshape(cur.shape[0], nn_ * s)
    return cur
