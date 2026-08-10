"""Finite families of RHM rule sets, and the DGP-side statistics that say whether a
family is identifiable ONLY through composition or also through cheap n-gram counts.

The conditional-revision substrate uses ONE fixed rule set, so which positions are
synonym-slots and which are disambiguators is fixed by the DGP and an NTP-optimal model
can bake the split into weights. Here a context window is drawn from one of R rule sets
(same v/s/L/m, different composition tables), so the model must infer the active rules
in-context and reducibility becomes state-dependent WITHIN the context.

Two knobs define a family:

  differ_levels     which `rules[d]` tables differ across the R rule sets. `rules[d]`
                    maps a level-d feature to an s-tuple of level-(d+1) features, with
                    d = 0 the root and d = L-1 the table that emits leaf tokens. In the
                    repo's probe naming a level-D latent is `d{L-D}`, so `rules[4]` is
                    "how a d2 feature expands" and `rules[5]` is "how a d1 feature emits
                    tokens".
  n_differ_features how many of the v features at those levels differ. Fewer differing
                    features = evidence about r arrives only when those features occur =
                    slower identification.

RHM_META_LEARNING's diagnosis (dense NTP is L0-dominated; the compositional signal is a
thin residual) is the standing risk, so the design constraint is that `rules[L-1]` --
the leaf-emitting table -- is SHARED across the family. Then no bigram statistic can
name the rule set directly and identification has to go through composition. That is a
design intent, not a proof: `block_distributions` / `shortcut_posterior` below measure
how much of the identification information actually survives at each n-gram order, and
the leak is real (the parent-feature marginal is not uniform, so permuting tuples among
features perturbs low-order marginals slightly). Measure it; do not assume it.

Two construction modes:

  "pool"     at a differing level the R rule sets share ONE pool of n_differ_features*m
             distinct tuples and only the PARTITION of that pool among features differs.
             The multiset of tuples produced at that level is identical across rule sets,
             so every statistic that does not depend on parent identity is matched up to
             the non-uniformity of the parent marginal. This is the low-leak default.
  "resample" each rule set draws fresh distinct tuples for the differing features. The
             tuple SETS differ, so a learner can identify r from which tuples ever occur
             -- a deliberately leaky control that should make the shortcut audit fire.

Run the self-test:  cd experiments && python3 -m rhm.conditional_revision.rule_family.family
"""

import numpy as np

from rhm.rhm_data import generate_rules_distinct


# ---------------------------------------------------------------------------
# family construction
# ---------------------------------------------------------------------------

def _distinct_tuples(rng, v, s, n):
    """n distinct s-tuples, encoded exactly as generate_rules_distinct does."""
    if n > v ** s:
        raise ValueError(f"cannot draw {n} distinct s-tuples from {v ** s}")
    codes = rng.choice(v ** s, size=n, replace=False)
    tup = np.empty((n, s), dtype=np.int64)
    for i in range(s):
        tup[:, i] = (codes // (v ** i)) % v
    return tup


def make_family(v, s, L, m, R, differ_levels, n_differ_features=None,
                seed=0, mode="pool"):
    """R rule sets sharing every table except `differ_levels`.

    Returns (family, meta) where family is a list of R rule sets, each a list of L
    (v, m, s) int arrays -- the exact shape every existing RHM routine expects, so a
    single member is a drop-in for `generate_rules_distinct(...)`.

    `differ_levels=[]` is the ZERO-CONFLICT FLOOR: all R rule sets are bit-identical to
    `generate_rules_distinct(v, s, L, m, seed=seed)`, i.e. the existing single-rule-set
    substrate, reached without changing anything else about the pipeline. This is the
    analogue of meta_adapt's damping floor (Phi = 0), and the point of having it is that
    the floor and the conflict regime run through identical code.
    """
    if mode not in ("pool", "resample"):
        raise ValueError(f"mode must be 'pool' or 'resample' (got {mode!r})")
    rng = np.random.default_rng(seed)
    base = generate_rules_distinct(v, s, L, m, seed=seed)
    differ_levels = sorted(set(int(d) for d in differ_levels))
    for d in differ_levels:
        if not 0 <= d < L:
            raise ValueError(f"differ level {d} outside 0..{L - 1}")
    nF = v if n_differ_features is None else int(n_differ_features)
    if mode == "pool" and nF < 2:
        raise ValueError("pool mode needs >= 2 differing features (permuting one "
                         "feature's own tuples among itself is a no-op)")

    feats, pools = {}, {}
    for d in differ_levels:
        feats[d] = np.sort(rng.choice(v, size=nF, replace=False))
        if mode == "pool":
            pools[d] = _distinct_tuples(rng, v, s, nF * m)

    # In pool mode a rule set is a PARTITION of the tuple pool among the differing
    # features. Two rule sets that differ only in the ORDER of a feature's m tuples are
    # distributionally IDENTICAL -- rules are chosen uniformly over m, so order is not a
    # degree of freedom the DGP can express. Comparing raw arrays misses this, and the
    # consequence is silent and expensive: the family contains unidentifiable duplicates,
    # the rule posterior can never concentrate past them, and I(r ; x) falls short of
    # ln R for a reason that looks like "the design is gradual" but is really "the design
    # is degenerate". So canonicalise (sort within each feature) and reject collisions.
    # The number of distinct partitions is (nF*m)! / (m!)^nF -- only 70 for nF=2, m=4.
    if mode == "pool" and differ_levels:
        from math import factorial
        n_part = factorial(nF * m) // factorial(m) ** nF
        if R > n_part ** len(differ_levels):
            raise ValueError(
                f"R={R} exceeds the {n_part ** len(differ_levels)} distinguishable rule "
                f"sets available from nF={nF}, m={m} at {len(differ_levels)} level(s). "
                f"Raise n_differ_features or lower R -- otherwise the family contains "
                f"distributionally identical members and cannot be identified.")

    family, seen = [], set()
    for _ in range(R):
        for attempt in range(10000):
            rules = [b.copy() for b in base]
            keys = []
            for d in differ_levels:
                F = feats[d]
                if mode == "pool":
                    perm = rng.permutation(nF * m).reshape(nF, m)
                    for i, f in enumerate(F):
                        rules[d][f] = pools[d][perm[i]]
                    keys.append(tuple(tuple(sorted(perm[i].tolist()))
                                      for i in range(nF)))
                else:
                    for f in F:
                        rules[d][f] = _distinct_tuples(rng, v, s, m)
                    keys.append(tuple(tuple(sorted(map(tuple, rules[d][f].tolist())))
                                      for f in F))
            key = tuple(keys)
            if not differ_levels or key not in seen:
                seen.add(key)
                break
        else:
            raise RuntimeError("could not draw a distinct rule set in 10000 attempts")
        family.append(rules)
    meta = {"v": v, "s": s, "L": L, "m": m, "R": R, "mode": mode,
            "differ_levels": differ_levels,
            "n_differ_features": nF,
            "differ_features": {int(d): feats[d].tolist() for d in differ_levels},
            "seed": seed}
    return family, meta


def generate_windows(family, n_windows, K, seed=0):
    """n_windows contexts, each K whole aligned sequences from ONE rule set.

    Returns (seqs, rule_ids, level_features) with
      seqs           (n_windows, K, T)      T = s^L
      rule_ids       (n_windows,)           which rule set generated each window
      level_features list of L+1 arrays, (n_windows, K, s^d) -- the true latents of
                     every sequence, [L] = leaves. Same convention as
                     rhm_latent_loop._generate_with_traces.

    A window is K WHOLE sequences rather than a random offset into a flat corpus. That
    keeps two axes orthogonal by construction: position-within-sequence fixes which
    hierarchy level a token completes, and sequence-index-within-window is context
    depth. Rule-revision decays along the second axis while parse structure varies along
    the first, so the two cannot be confounded. The cost is that absolute position is
    informative about level, which every downstream matching already controls for.
    """
    rng = np.random.default_rng(seed)
    R = len(family)
    v, m, s = family[0][0].shape
    L = len(family[0])
    T = s ** L
    rule_ids = rng.integers(0, R, size=n_windows)
    seqs = np.empty((n_windows, K, T), dtype=np.int64)
    lf = [np.empty((n_windows, K, s ** d), dtype=np.int64) for d in range(L + 1)]
    for r in range(R):
        sel = np.where(rule_ids == r)[0]
        if not sel.size:
            continue
        n = sel.size * K
        cur = rng.integers(0, v, size=(n, 1))
        per_level = []
        for d in range(L):
            per_level.append(cur.copy())
            rc = rng.integers(0, m, size=(n, cur.shape[1]))
            nxt = np.empty((n, cur.shape[1] * s), dtype=np.int64)
            for j in range(cur.shape[1]):
                nxt[:, j * s:(j + 1) * s] = family[r][d][cur[:, j], rc[:, j]]
            cur = nxt
        per_level.append(cur.copy())
        seqs[sel] = cur.reshape(sel.size, K, T)
        for d in range(L + 1):
            lf[d][sel] = per_level[d].reshape(sel.size, K, s ** d)
    return seqs, rule_ids, lf


# ---------------------------------------------------------------------------
# the shortcut audit: what a cheap n-gram learner can see
# ---------------------------------------------------------------------------

def node_marginals_prior(rules):
    """P(z_j = a) for every node, no evidence. List of L+1 arrays (s^d, v).

    Exact: the root is uniform and each rule is chosen uniformly over m, so the
    marginal propagates down deterministically. Needed because the parent marginal at a
    differing level is NOT uniform, which is the mechanism by which a "matched pool"
    family still leaks a little into low-order statistics.
    """
    v, m, s = rules[0].shape
    L = len(rules)
    out = [np.full((1, v), 1.0 / v)]
    for d in range(L):
        n_par = s ** d
        nxt = np.zeros((n_par * s, v))
        for f in range(v):
            for r in range(m):
                tup = rules[d][f, r]
                for i in range(s):
                    nxt[np.arange(n_par) * s + i, tup[i]] += out[d][:, f] / m
        out.append(nxt)
    return out


def block_distributions(rules, k):
    """Exact distribution over each aligned s^k-token block, as dicts.

    Returns a list of length s^(L-k); entry j maps a tuple of s^k leaf tokens to its
    probability, for the block spanned by node j at tree level L-k. k = 0 gives the
    per-position unigram distribution.

    These are the statistics a "cheap" learner can estimate by counting, so the
    divergence between rule sets on them upper-bounds what an n-gram shortcut can know.
    """
    v, m, s = rules[0].shape
    L = len(rules)
    prior = node_marginals_prior(rules)
    lvl = L - k

    def subtree(d, f):
        """dict: leaf tuple under a level-d node with feature f -> probability."""
        if d == L:
            return {(f,): 1.0}
        out = {}
        for r in range(m):
            tup = rules[d][f, r]
            parts = [subtree(d + 1, int(tup[i])) for i in range(s)]
            acc = {(): 1.0 / m}
            for p in parts:
                nxt = {}
                for pref, pw in acc.items():
                    for suf, sw in p.items():
                        key = pref + suf
                        nxt[key] = nxt.get(key, 0.0) + pw * sw
                acc = nxt
            for key, w in acc.items():
                out[key] = out.get(key, 0.0) + w
        return out

    cache = {f: subtree(lvl, f) for f in range(v)}
    blocks = []
    for j in range(s ** lvl):
        agg = {}
        for f in range(v):
            pf = prior[lvl][j, f]
            if pf <= 0:
                continue
            for key, w in cache[f].items():
                agg[key] = agg.get(key, 0.0) + pf * w
        blocks.append(agg)
    return blocks


def shortcut_posterior(family, seqs, k, eps=1e-12):
    """Log-posterior over r from aligned s^k-block counts ONLY, treated as independent.

    seqs: (n, K, T). Returns (n, K*T_blocks+1, R) cumulative log-posteriors -- entry
    [i, b, r] is log P(r | the first b blocks of window i), normalised.

    This is a concrete shortcut learner, not a bound: it sees only which aligned block
    appeared where and nothing about composition across blocks. Compare its
    concentration curve against the exact posterior's. If they track, the family is
    identifiable from n-gram counts at order s^k and the compositional claim is void.
    """
    R = len(family)
    n, K, T = seqs.shape
    s = family[0][0].shape[2]
    L = len(family[0])
    span = s ** k
    n_blocks = T // span
    tables = []
    for r in range(R):
        blocks = block_distributions(family[r], k)
        tables.append(blocks)
    tot = K * n_blocks
    lp = np.zeros((n, tot + 1, R))
    b = 0
    for kk in range(K):
        for j in range(n_blocks):
            chunk = seqs[:, kk, j * span:(j + 1) * span]
            step = np.zeros((n, R))
            for r in range(R):
                tbl = tables[r][j]
                step[:, r] = [np.log(max(tbl.get(tuple(row.tolist()), 0.0), eps))
                              for row in chunk]
            lp[:, b + 1] = lp[:, b] + step
            b += 1
    lp = lp - lp.max(axis=2, keepdims=True)
    lp = lp - np.log(np.exp(lp).sum(axis=2, keepdims=True))
    return lp


def pairwise_block_divergence(family, k):
    """Mean symmetrised KL between rule sets on the aligned s^k-block distribution.

    One number per (block position); returned as (n_blocks,) averaged over rule-set
    pairs. Zero means the block statistic is exactly matched across the family and
    carries no identification information at that order.
    """
    R = len(family)
    tabs = [block_distributions(family[r], k) for r in range(R)]
    n_blocks = len(tabs[0])
    out = np.zeros(n_blocks)
    npair = 0
    for a in range(R):
        for b in range(a + 1, R):
            npair += 1
            for j in range(n_blocks):
                pa, pb = tabs[a][j], tabs[b][j]
                keys = set(pa) | set(pb)
                d = 0.0
                for key in keys:
                    x, y = pa.get(key, 0.0), pb.get(key, 0.0)
                    if x > 0 and y > 0:
                        d += 0.5 * (x - y) * (np.log(x) - np.log(y))
                    elif x > 0 or y > 0:
                        d += np.inf
                out[j] += d
    return out / max(npair, 1)


# ---------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------

def _self_test():
    print("=" * 74)
    print("family self-test")
    print("=" * 74)

    # the zero-conflict floor reproduces the incumbent substrate exactly
    fam, _ = make_family(16, 2, 6, 4, R=8, differ_levels=[], seed=0)
    ref = generate_rules_distinct(16, 2, 6, 4, seed=0)
    for r in range(8):
        assert all(np.array_equal(a, b) for a, b in zip(fam[r], ref)), \
            "differ_levels=[] must be bit-identical to generate_rules_distinct"
    print("  floor (differ_levels=[]) is bit-identical to generate_rules_distinct  ok")

    # shared levels really are shared; differing levels really differ
    fam, meta = make_family(16, 2, 6, 4, R=4, differ_levels=[3, 4],
                            n_differ_features=8, seed=1)
    for d in range(6):
        same = all(np.array_equal(fam[0][d], fam[r][d]) for r in range(1, 4))
        assert same == (d not in (3, 4)), f"level {d} sharing is wrong"
    for d in (3, 4):
        touched = [f for f in range(16)
                   if not all(np.array_equal(fam[0][d][f], fam[r][d][f])
                              for r in range(1, 4))]
        assert set(touched) <= set(meta["differ_features"][d]), \
            "a feature outside the differing set moved"
    print(f"  sharing/differing structure ok   differ_features "
          f"{meta['differ_features'][4][:4]}...")

    # block distributions are proper distributions and match brute force on a tiny tree
    v, s, L, m = 3, 2, 3, 2
    fam, _ = make_family(v, s, L, m, R=2, differ_levels=[1], n_differ_features=3, seed=2)
    for r in range(2):
        for k in (0, 1, 2):
            blocks = block_distributions(fam[r], k)
            for j, tbl in enumerate(blocks):
                assert abs(sum(tbl.values()) - 1.0) < 1e-12, (r, k, j)
    from rhm.conditional_revision.rule_family.family import generate_windows
    seqs, rid, lf = generate_windows(fam, 40000, 1, seed=3)
    for r in range(2):
        sel = seqs[rid == r, 0]
        blocks = block_distributions(fam[r], 1)
        emp = {}
        for row in sel[:, :2]:
            key = tuple(row.tolist())
            emp[key] = emp.get(key, 0) + 1
        n = sel.shape[0]
        err = max(abs(emp.get(key, 0) / n - p) for key, p in blocks[0].items())
        assert err < 0.02, f"block distribution disagrees with samples ({err})"
        print(f"  rule set {r}: exact 2-token block dist vs {n} samples, "
              f"max err {err:.4f}  ok")

    # level features really generate the leaves they claim to
    fam, _ = make_family(8, 2, 4, 3, R=3, differ_levels=[2], n_differ_features=4, seed=4)
    seqs, rid, lf = generate_windows(fam, 200, 3, seed=5)
    assert np.array_equal(lf[4], seqs), "level_features[L] must be the leaves"
    for i in range(20):
        r = rid[i]
        for kk in range(3):
            cur = lf[0][i, kk]
            for d in range(4):
                nxt = lf[d + 1][i, kk]
                for j in range(cur.shape[0]):
                    tup = nxt[j * 2:(j + 1) * 2]
                    ok = any(np.array_equal(fam[r][d][cur[j], q], tup) for q in range(3))
                    assert ok, f"latent trace not generated by rule set {r} at level {d}"
                cur = nxt
    print("  generate_windows traces are consistent with their own rule set  ok")
    print("family self-test PASSED")


if __name__ == "__main__":
    _self_test()
