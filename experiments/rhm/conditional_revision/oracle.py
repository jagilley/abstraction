"""Exact belief-propagation oracle for conditional revision on the RHM.

SPEC.md's `B_t^l`, `H_irr^l` and the identity that ties them to token surprisal:

    E[ B_D ]  =  H(x_{t+1} | x_{<=t})  -  H(x_{t+1} | z_{<=D}, x_{<=t})
                 total surprisal          irreducible-given-structure

This is the mutual information I(x_{t+1}; z_{<=D} | x_{<=t}) written two ways, so
it holds exactly and is the instrument self-check: if the marginal extraction here
is wrong, the two sides disagree and nothing downstream is interpretable.

Everything is exact sum-product BP on the KNOWN parse tree with the KNOWN rules.
Compression-free by construction -- there is no learned forecaster anywhere in
this file.

Conventions (they differ from the repo's `d{k}` probe names; both are emitted)
--------------------------------------------------------------------------
  tree level d:  0 = root, L = leaves. rules[d] : (v, m, s) maps a level-d
                 feature to an s-tuple of level-(d+1) features.
  z_{<=D}:       ALL latents at tree levels 0..D. D ranges 0..L-1 (D=L would be
                 the leaves, which are observed, not latent).
  d-name:        the repo's probe convention, d1 = shallowest (just above the
                 leaves) ... dL = root. A cumulative set z_{<=D} therefore has
                 deepest level D <-> d-name `d{L-D}`.
  position t:    signal index, t = 0..T-2, meaning "the token x_{t+1} arrives".
                 B_t compares the posterior after x_{t+1} to the one before it,
                 i.e. prefix lengths t+2 and t+1. Matches gate0's nll/residual
                 indexing exactly.

Why the JOINT KL and not a sum of per-node KLs
----------------------------------------------
The identity above is about the joint posterior over z_{<=D}. A sum of marginal
KLs is a different (and generally smaller) quantity, and it does NOT satisfy the
identity -- so it cannot be self-checked. Both are computed: `B_joint` is the
oracle SPEC.md defines and the one the self-check validates; `B_marg` is the
form-matched control for the model's probe-decoded belief, which is a product of
marginals and can only ever produce the marginal version.

The RHM latent graph is a HYPERtree, not a pairwise tree: one rule choice emits a
parent's whole s-tuple at once, so the factor psi(parent, child_1..child_s) is a
single (s+1)-way clique and the children are dependent given the parent. (A
pairwise-tree formula built from parent-child edge marginals is therefore wrong,
and the brute-force self-test below catches it: it reads H = 1.733 where the truth
is ln 4 = 1.386 on the smallest test tree.) The junction tree has cliques
C_p = {p} union children(p) for every internal p, and separators {p} for every p
that is both a child and a parent, giving

    P(z_{<=D}) = prod_{p: level 0..D-1} P(C_p) / prod_{p: level 1..D-1} P(z_p)
    KL(P'||P)  = sum_{level 0..D-1} KL(C'_p||C_p) - sum_{level 1..D-1} KL(z'_p||z_p)

which needs only clique and node marginals, both available from BP. Marginalising
out the levels below D truncates the junction tree without changing the surviving
cliques, so the formula applies unchanged to z_{<=D}.

A clique is stored sparsely as P(parent = a, rule = r): given the parent's value
only m of the v^s tuples are reachable. `generate_rules_distinct` makes a feature's
m tuples distinct, but duplicates are merged anyway so `generate_rules` also works.

Local correctness self-test (brute-force enumeration vs BP on tiny trees):
    cd experiments && python3 -m rhm.conditional_revision.oracle
"""

import numpy as np

EPS = 1e-300


# ---------------------------------------------------------------------------
# BP with optional internal-node evidence
# ---------------------------------------------------------------------------
# rhm_bayes_entropy's _upward/_downward have no internal-node potentials and no
# message normalisation. Both are needed here (clamping z_D for H_irr; 64-deep
# products for B), so these are separate functions rather than edits to that
# module -- prior results stay bit-identical. `_check_matches_rhm_bayes_entropy`
# asserts the posteriors agree where the two overlap.


def _norm(msg):
    """Normalise a (B, n, v) message per (batch, node). Posteriors and edge
    marginals are normalised at the end, and every scale factor introduced here
    is constant across the value axis, so this changes nothing but the exponent."""
    z = msg.sum(-1, keepdims=True)
    return msg / np.clip(z, EPS, None)


def _upward(up_leaf, rules, nev=None):
    """Leaves->root sum-product. up[d]: (B, s^d, v), includes node d's own potential.

    nev: optional list of length L+1 of (B, s^d, v) multiplicative node potentials
         (None entries = no evidence). Used to clamp latents.
    """
    L = len(rules)
    v, m, s = rules[0].shape
    B = up_leaf.shape[0]
    up = [None] * (L + 1)
    up[L] = _norm(up_leaf if nev is None or nev[L] is None else up_leaf * nev[L])
    for d in range(L - 1, -1, -1):
        n_par = s ** d
        up_child = up[d + 1].reshape(B, n_par, s, v)
        rd = rules[d]
        out = np.zeros((B, n_par, v))
        for a in range(v):
            acc = np.zeros((B, n_par))
            for r in range(m):
                tup = rd[a, r]
                prod = np.ones((B, n_par))
                for k in range(s):
                    prod = prod * up_child[:, :, k, tup[k]]
                acc += prod
            out[:, :, a] = acc / m
        if nev is not None and nev[d] is not None:
            out = out * nev[d]
        up[d] = _norm(out)
    return up


def _downward(up, rules, nev=None, root_prior=None):
    """Root->leaves sum-product. down[d]: (B, s^d, v), EXCLUDES node d's own
    potential (so the posterior is up[d] * down[d]).

    root_prior: (B, 1, v); defaults to uniform (the RHM samples the root uniformly).
    """
    L = len(rules)
    v, m, s = rules[0].shape
    B = up[0].shape[0]
    down = [None] * (L + 1)
    down[0] = (np.full((B, 1, v), 1.0 / v) if root_prior is None
               else _norm(root_prior.astype(np.float64)))
    for d in range(L):
        n_par = s ** d
        up_child = up[d + 1].reshape(B, n_par, s, v)
        rd = rules[d]
        # the parent's outgoing message carries its own potential
        par = down[d] if nev is None or nev[d] is None else down[d] * nev[d]
        out = np.zeros((B, n_par, s, v))
        for a in range(v):
            da = par[:, :, a]
            for r in range(m):
                tup = rd[a, r]
                ups = [up_child[:, :, k, tup[k]] for k in range(s)]
                for i in range(s):
                    prod_except_i = np.ones((B, n_par))
                    for k in range(s):
                        if k != i:
                            prod_except_i = prod_except_i * ups[k]
                    out[:, :, i, tup[i]] += da * (prod_except_i / m)
        down[d + 1] = _norm(out.reshape(B, n_par * s, v))
    return down


def node_marginals(up, down, Dmax):
    """P(z_node | evidence) for tree levels 0..Dmax. List of (B, s^d, v)."""
    return [_norm(up[d] * down[d]) for d in range(Dmax + 1)]


def _rule_ids(rules, d):
    """Map (parent value a, rule r) -> a canonical id, merging duplicate tuples.

    Returns (ids, n_uniq) with ids (v, m) int. `generate_rules_distinct` yields
    n_uniq == m everywhere; `generate_rules` can repeat a tuple, and merging keeps
    the clique a genuine distribution over distinct outcomes either way.
    """
    v, m, s = rules[d].shape
    ids = np.zeros((v, m), dtype=np.int64)
    for a in range(v):
        seen = {}
        for r in range(m):
            key = tuple(rules[d][a, r].tolist())
            if key not in seen:
                seen[key] = len(seen)
            ids[a, r] = seen[key]
    return ids


def clique_marginals(up, down, rules, d, nev=None, ids=None):
    """P(z_parent = a, its child tuple = rule a's r-th) for the level-d cliques.

    Returns (B, s^d, v, m): [b, j, a, u] = P(parent j = a AND its children take
    parent-a's u-th distinct tuple). Zero for u >= the number of distinct tuples.
    """
    v, m, s = rules[0].shape
    B = up[0].shape[0]
    n_par = s ** d
    if ids is None:
        ids = _rule_ids(rules, d)
    up_child = up[d + 1].reshape(B, n_par, s, v)
    par = down[d] if nev is None or nev[d] is None else down[d] * nev[d]
    rd = rules[d]
    C = np.zeros((B, n_par, v, m))
    for a in range(v):
        da = par[:, :, a]
        for r in range(m):
            tup = rd[a, r]
            full = np.ones((B, n_par))
            for k in range(s):
                full = full * up_child[:, :, k, tup[k]]
            C[:, :, a, ids[a, r]] += da * full / m
    z = C.sum(axis=(2, 3), keepdims=True)
    return C / np.clip(z, EPS, None)


# ---------------------------------------------------------------------------
# Junction-tree KL / entropy from clique + node (separator) marginals
# ---------------------------------------------------------------------------

def _kl(p, q, axes):
    """sum p log(p/q) over `axes`, safe where p == 0 (supp(p) subset supp(q))."""
    ratio = np.log(np.clip(p, EPS, None)) - np.log(np.clip(q, EPS, None))
    return (np.where(p > 0, p * ratio, 0.0)).sum(axis=axes)


def _ent(p, axes):
    return -(np.where(p > 0, p * np.log(np.clip(p, EPS, None)), 0.0)).sum(axis=axes)


def joint_kl(new_nodes, old_nodes, new_cliques, old_cliques, D, s):
    """Exact KL( P'(z_{<=D}) || P(z_{<=D}) ) per batch element (junction tree)."""
    B = new_nodes[0].shape[0]
    if D == 0:
        return _kl(new_nodes[0], old_nodes[0], axes=(1, 2))
    total = np.zeros(B)
    for d in range(D):                       # cliques at levels 0..D-1
        total += _kl(new_cliques[d], old_cliques[d], axes=(1, 2, 3))
    for d in range(1, D):                    # separators at levels 1..D-1
        total -= _kl(new_nodes[d], old_nodes[d], axes=(1, 2))
    return total


def joint_entropy(nodes, cliques, D, s):
    """Exact H( P(z_{<=D}) ) per batch element, same decomposition."""
    B = nodes[0].shape[0]
    if D == 0:
        return _ent(nodes[0], axes=(1, 2))
    total = np.zeros(B)
    for d in range(D):
        total += _ent(cliques[d], axes=(1, 2, 3))
    for d in range(1, D):
        total -= _ent(nodes[d], axes=(1, 2))
    return total


def marginal_kl_sum(new_nodes, old_nodes, D):
    """sum_j KL( P'(z_j) || P(z_j) ) over all nodes at levels 0..D. The
    form-matched control for the model's product-of-marginals belief."""
    B = new_nodes[0].shape[0]
    total = np.zeros(B)
    for d in range(D + 1):
        total += _kl(new_nodes[d], old_nodes[d], axes=(1, 2))
    return total


# ---------------------------------------------------------------------------
# The oracle proper
# ---------------------------------------------------------------------------

def _leaf_evidence(seqs, plen, v):
    """(B, T, v) one-hot for leaves < plen, all-ones (free) for the rest."""
    B, T = seqs.shape
    ev = np.ones((B, T, v))
    if plen > 0:
        oh = np.zeros((B, plen, v))
        np.put_along_axis(oh, seqs[:, :plen, None], 1.0, axis=2)
        ev[:, :plen, :] = oh
    return ev


def prefix_beliefs(rules, seqs, plen, Dmax, ids=None):
    """Node + clique marginals of P(z | x_{<plen}) for tree levels 0..Dmax.

    Also returns P(x_plen | x_{<plen}) -- the exact next-token posterior, free from
    the same upward pass.
    """
    v = rules[0].shape[0]
    up = _upward(_leaf_evidence(seqs, plen, v), rules)
    down = _downward(up, rules)
    nodes = node_marginals(up, down, Dmax)
    cliques = [clique_marginals(up, down, rules, d,
                                ids=None if ids is None else ids[d])
               for d in range(Dmax)]
    leaf_post = None
    if plen < seqs.shape[1]:
        L = len(rules)
        leaf_post = _norm((up[L] * down[L])[:, plen, :][:, None, :])[:, 0, :]
    return nodes, cliques, leaf_post


def irreducible_entropy(rules, seqs, level_features, D, plen):
    """H(x_plen | z_{<=D}, x_{<plen}) for each sequence, exactly.

    Clamping every level-D node d-separates the subtrees, so this is BP on the
    single subtree rooted at x_plen's level-D ancestor with that root clamped to
    its true value -- identical to clamping the whole level, and far cheaper.
    """
    v, m, s = rules[0].shape
    L = len(rules)
    B, T = seqs.shape
    span = s ** (L - D)
    j = plen // span                       # which level-D node governs x_plen
    lo = j * span
    sub_rules = rules[D:]
    sub_seqs = seqs[:, lo:lo + span]
    n_obs = max(0, min(span, plen - lo))   # observed leaves inside this subtree
    ev = _leaf_evidence(sub_seqs, n_obs, v)
    root_prior = np.zeros((B, 1, v))
    true_zD = level_features[D][:, j]      # (B,)
    root_prior[np.arange(B), 0, true_zD] = 1.0
    up = _upward(ev, sub_rules)
    down = _downward(up, sub_rules, root_prior=root_prior)
    post = _norm((up[L - D] * down[L - D])[:, plen - lo, :][:, None, :])[:, 0, :]
    return _ent(post, axes=1), post


def revision_and_entropy(rules, seqs, level_features, Ds=None, chunk=256,
                         verbose=True, anc_mask=None):
    """The full oracle: B (joint + marginal), H_tot, H_irr, true surprisal.

    Returns a dict of arrays. Signal axis is t = 0..T-2 ("the token x_{t+1}
    arrives"), matching gate0's nll and residual indexing.

      B_joint[D]   (n, T-1)  KL( P(z_{<=D}|x_{<=t+1}) || P(z_{<=D}|x_{<=t}) ), exact joint
      B_marg[D]    (n, T-1)  sum of per-node KLs, same conditioning
      B_chain[D]   (n, T-1)  the same sum restricted to the ANCESTORS of x_{t+1},
                             one node per level -- <=6 terms instead of 63. The
                             model-side estimate of B_marg sums a probe KL over
                             every latent node, so its probe noise accumulates
                             ~63-fold (measured: mean M 6.07 against a true B of
                             0.70). Restricting both sides to the chain that
                             actually governs the arriving token keeps the
                             comparison exactly matched while cutting that noise,
                             and it is the more natural object anyway: "how much
                             did this token move my belief about the structure it
                             belongs to."
      H_post[D]    (n, T-1)  H( P(z_{<=D}|x_{<=t}) ) -- how determined structure already is
      H_irr[D]     (n, T-1)  H(x_{t+1} | z_{<=D}, x_{<=t})
      H_tot        (n, T-1)  H(x_{t+1} | x_{<=t})
      surprisal    (n, T-1)  -log P(x_{t+1}=observed | x_{<=t}), the Bayes-optimal nll

    anc_mask (T, n_internal), optional: if given, also returns `bayes_acc_chain`,
    the exact Bayes accuracy of decoding each masked (position, node) pair from
    x_{<=t}, per level. That is the ceiling for the probe that produces M, so
    "the probe is weak" and "the model cannot know this" stay distinguishable.
    """
    v, m, s = rules[0].shape
    L = len(rules)
    n, T = seqs.shape
    if Ds is None:
        Ds = list(range(L))          # D = 0 (root only) .. L-1 (all internal levels)
    Dmax = max(Ds)

    out = {k: {D: np.zeros((n, T - 1)) for D in Ds}
           for k in ("B_joint", "B_marg", "B_chain", "H_post", "H_irr")}
    H_tot = np.zeros((n, T - 1))
    surprisal = np.zeros((n, T - 1))
    ids = [_rule_ids(rules, d) for d in range(L)]
    node_off = {d: (s ** d - 1) // (s - 1) for d in range(L)}
    bayes_hit = np.zeros(L)
    bayes_cnt = np.zeros(L)

    for c0 in range(0, n, chunk):
        c1 = min(n, c0 + chunk)
        sq = seqs[c0:c1]
        lf = [level_features[d][c0:c1] for d in range(L + 1)]
        prev = prefix_beliefs(rules, sq, 1, Dmax, ids)   # after x_0 only
        for t in range(T - 1):
            # H_tot / surprisal for predicting x_{t+1} from x_{<=t}: prefix length t+1
            leaf_post = prev[2]
            H_tot[c0:c1, t] = _ent(leaf_post, axes=1)
            obs = sq[:, t + 1]
            surprisal[c0:c1, t] = -np.log(
                np.clip(leaf_post[np.arange(c1 - c0), obs], EPS, None))
            if anc_mask is not None:
                for d in range(L):
                    jj = np.where(anc_mask[t, node_off[d]:node_off[d] + s ** d])[0]
                    if jj.size:
                        bayes_hit[d] += prev[0][d][:, jj].max(-1).sum()
                        bayes_cnt[d] += jj.size * (c1 - c0)
            cur = prefix_beliefs(rules, sq, t + 2, Dmax, ids)
            # per-level KL for x_{t+1}'s own ancestor at each level
            chain = np.zeros((c1 - c0, L))
            for d in range(Dmax + 1):
                j = (t + 1) // (s ** (L - d))
                chain[:, d] = _kl(cur[0][d][:, j], prev[0][d][:, j], axes=1)
            for D in Ds:
                out["B_joint"][D][c0:c1, t] = joint_kl(
                    cur[0], prev[0], cur[1], prev[1], D, s)
                out["B_marg"][D][c0:c1, t] = marginal_kl_sum(cur[0], prev[0], D)
                out["B_chain"][D][c0:c1, t] = chain[:, :D + 1].sum(1)
                out["H_post"][D][c0:c1, t] = joint_entropy(prev[0], prev[1], D, s)
                out["H_irr"][D][c0:c1, t] = irreducible_entropy(
                    rules, sq, lf, D, t + 1)[0]
            prev = cur
        if verbose:
            print(f"  oracle: {c1}/{n} sequences", flush=True)

    return {"B_joint": out["B_joint"], "B_marg": out["B_marg"],
            "B_chain": out["B_chain"],
            "H_post": out["H_post"], "H_irr": out["H_irr"],
            "H_tot": H_tot, "surprisal": surprisal, "Ds": Ds, "L": L, "s": s,
            "bayes_acc_chain": (None if anc_mask is None else
                                {f"d{L - d}": float(bayes_hit[d] / max(bayes_cnt[d], 1))
                                 for d in range(L)})}


def self_check(res, tol=0.02):
    """The instrument self-check SPEC.md requires before Gate A:

        mean_t B_D  ==  mean_t [ H(x_{t+1}|x_{<=t}) - H(x_{t+1}|z_{<=D},x_{<=t}) ]

    Both sides are Monte-Carlo averages over the same sequences, and the realised
    x_{t+1} in each sequence IS a draw from P(.|x_{<=t}), so equality is exact in
    expectation. A per-D report; `passed` is the conjunction.
    """
    L = res["L"]
    rows = {}
    ok = True
    for D in res["Ds"]:
        lhs = float(res["B_joint"][D].mean())
        rhs = float((res["H_tot"] - res["H_irr"][D]).mean())
        err = abs(lhs - rhs)
        rel = err / max(abs(rhs), 1e-9)
        rows[f"D{D}"] = {
            "D": D, "d_name": f"d{L - D}",
            "mean_B_joint": lhs,
            "mean_H_tot_minus_H_irr": rhs,
            "abs_err": err, "rel_err": rel,
            "mean_B_marg": float(res["B_marg"][D].mean()),
            "mean_H_irr": float(res["H_irr"][D].mean()),
            "passed": bool(err < tol or rel < tol),
        }
        ok = ok and rows[f"D{D}"]["passed"]
    return {"per_D": rows, "passed": bool(ok),
            "mean_H_tot": float(res["H_tot"].mean()),
            "mean_true_surprisal": float(res["surprisal"].mean())}


# ---------------------------------------------------------------------------
# Correctness self-tests (run this module directly)
# ---------------------------------------------------------------------------

def _enumerate_tree(rules):
    """Every (latent config -> leaf sequence) with its probability. Tiny trees only.

    Returns (Z, X, P): Z (N, n_internal) node feature values in breadth-first
    order, X (N, s^L) leaves, P (N,) probabilities.
    """
    import itertools
    v, m, s = rules[0].shape
    L = len(rules)
    n_internal = (s ** L - 1) // (s - 1)
    Z, X, P = [], [], []
    for root in range(v):
        for choices in itertools.product(range(m), repeat=n_internal):
            ci, current, nodes = 0, [root], [root]
            for d in range(L):
                nxt = []
                for node in current:
                    nxt.extend(rules[d][node, choices[ci]].tolist())
                    ci += 1
                current = nxt
                if d < L - 1:
                    nodes.extend(current)
            Z.append(nodes)
            X.append(current)
            P.append((1.0 / v) * (1.0 / m) ** n_internal)
    return np.array(Z), np.array(X), np.array(P)


def _bf_posterior_over_z(Z, X, P, prefix, n_nodes_upto_D):
    """Exact posterior over the first `n_nodes_upto_D` latent nodes given a prefix."""
    match = np.all(X[:, :len(prefix)] == np.asarray(prefix)[None, :], axis=1)
    w = P * match
    w = w / w.sum()
    keys = {}
    for zi, wi in zip(Z[:, :n_nodes_upto_D], w):
        if wi > 0:
            keys[tuple(zi)] = keys.get(tuple(zi), 0.0) + wi
    return keys


def _self_test():
    from rhm.rhm_data import generate_rules_distinct

    print("=" * 74)
    print("oracle self-test: BP tree-KL vs brute-force enumeration")
    print("=" * 74)

    for (v, s, L, m) in [(2, 2, 3, 2), (3, 2, 2, 2), (2, 2, 3, 2)]:
        rules = generate_rules_distinct(v, s, L, m, seed=1)
        Z, X, P = _enumerate_tree(rules)
        T = s ** L
        # every distinct leaf sequence with positive probability, as a batch
        uniq = np.unique(X, axis=0)
        max_kl_err = 0.0
        max_ent_err = 0.0
        max_marg_err = 0.0
        for D in range(L):
            n_nodes = (s ** (D + 1) - 1) // (s - 1)
            for plen in range(1, T):
                old_n, old_c, _ = prefix_beliefs(rules, uniq, plen, D)
                new_n, new_c, _ = prefix_beliefs(rules, uniq, plen + 1, D)
                kl_bp = joint_kl(new_n, old_n, new_c, old_c, D, s)
                ent_bp = joint_entropy(old_n, old_c, D, s)
                for b in range(uniq.shape[0]):
                    pre = uniq[b, :plen].tolist()
                    q = _bf_posterior_over_z(Z, X, P, pre, n_nodes)
                    p = _bf_posterior_over_z(Z, X, P, uniq[b, :plen + 1].tolist(),
                                             n_nodes)
                    kl_bf = sum(pv * np.log(pv / q[k]) for k, pv in p.items())
                    ent_bf = -sum(qv * np.log(qv) for qv in q.values())
                    max_kl_err = max(max_kl_err, abs(kl_bf - kl_bp[b]))
                    max_ent_err = max(max_ent_err, abs(ent_bf - ent_bp[b]))
                    # node marginals against the enumerated posterior
                    for d in range(D + 1):
                        off = (s ** d - 1) // (s - 1)
                        for j in range(s ** d):
                            ref = np.zeros(v)
                            for k, qv in q.items():
                                ref[k[off + j]] += qv
                            max_marg_err = max(
                                max_marg_err, np.abs(ref - old_n[d][b, j]).max())
        print(f"  v{v}/s{s}/L{L}/m{m}: max |KL_bf - KL_bp| = {max_kl_err:.2e}   "
              f"max |H_bf - H_bp| = {max_ent_err:.2e}   "
              f"max node-marginal err = {max_marg_err:.2e}")
        assert max_kl_err < 1e-8, f"joint KL disagrees with brute force ({max_kl_err})"
        assert max_ent_err < 1e-8, f"joint entropy disagrees ({max_ent_err})"
        assert max_marg_err < 1e-9, f"node marginals disagree ({max_marg_err})"

    _check_matches_rhm_bayes_entropy()
    _check_identity_small()
    print("oracle self-test PASSED")


def _check_matches_rhm_bayes_entropy():
    """H(x_i|x_{<i}) from this module's normalised BP must equal the existing
    module's, or the two are not measuring the same substrate."""
    from rhm.rhm_data import generate_rules_distinct
    from rhm.rhm_bayes_entropy import bp_conditional_entropies
    from rhm.rhm_latent_loop import _generate_with_traces

    v, s, L, m = 4, 2, 3, 2
    rules = generate_rules_distinct(v, s, L, m, seed=1)
    seqs, lf, _ = _generate_with_traces(rules, 800, 3)
    H_ref, _ = bp_conditional_entropies(rules, seqs)
    res = revision_and_entropy(rules, seqs, lf, Ds=[0], chunk=400, verbose=False)
    err = np.abs(H_ref[1:] - res["H_tot"].mean(0)).max()
    print(f"  agreement with rhm_bayes_entropy.bp_conditional_entropies: "
          f"max |dH| = {err:.2e}")
    assert err < 1e-9, f"H(x_i|x_<i) disagrees with the incumbent module ({err})"


def _check_identity_small():
    """E[B_D] == E[H_tot - H_irr_D] on a small but non-trivial regime."""
    from rhm.rhm_data import generate_rules_distinct
    from rhm.rhm_latent_loop import _generate_with_traces

    v, s, L, m = 6, 2, 4, 3
    rules = generate_rules_distinct(v, s, L, m, seed=1)
    seqs, lf, _ = _generate_with_traces(rules, 6000, 11)
    res = revision_and_entropy(rules, seqs, lf, chunk=1500, verbose=False)
    chk = self_check(res)
    print(f"  identity check on v{v}/s{s}/L{L}/m{m}, n=6000 "
          f"(MC error ~ 1/sqrt(n)):")
    for k, r in chk["per_D"].items():
        print(f"    {k} ({r['d_name']}): E[B] {r['mean_B_joint']:.5f}   "
              f"E[H_tot-H_irr] {r['mean_H_tot_minus_H_irr']:.5f}   "
              f"abs_err {r['abs_err']:.2e}  {'ok' if r['passed'] else 'FAIL'}")
    assert chk["passed"], "the B / (H_tot - H_irr) identity does not hold"


if __name__ == "__main__":
    _self_test()
