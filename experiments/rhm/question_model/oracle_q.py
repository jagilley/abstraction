"""Weighted-mixture BP oracle: the STRUCTURE channel at a known question state.

`conditional_revision/oracle.py` is exact BP on the known parse tree with a UNIFORM root
prior and a UNIFORM mixture over each cell's m rules -- the plain RHM. Under a drifting
generative mixture both of those move, so every `/m` and every `1/v` becomes a weight. This
module is that generalisation, written as new functions rather than as edits to `oracle.py`
so every prior result stays bit-identical, with three self-tests:

  1. at uniform weights it reproduces `oracle.py` to machine precision (the FIDELITY gate:
     drift-off must reproduce the static baseline exactly, not approximately);
  2. against brute-force enumeration on tiny trees with NON-uniform weights (the check that
     caught the hypertree error in the incumbent, re-run in the weighted regime);
  3. the conditional identity, which is `oracle.self_check` one conditioning deeper:

         E[ S_D | theta ]  =  H(x_{t+1} | theta, x_{<=t})  -  H(x_{t+1} | theta, z_{<=D}, x_{<=t})

     i.e. `revision_not_surprisal` SS4's identity holds inside each question state. The
     three-way identity that adds the demand term is assembled in `qfilter.py`.

Conventions are `oracle.py`'s throughout: tree level d = 0 at the root, `d{L-D}` names,
signal index t = 0..T-2 meaning "the token x_{t+1} arrives".

Weights are passed per batch element -- `rule_w` is a list of L arrays of shape (B, v, m)
and `root_p` is (B, v) -- because the sequences in a chunk were generated at different
question states and the oracle conditions on each one's own true theta.
"""

import numpy as np

from rhm.conditional_revision.oracle import (
    EPS, _ent, _kl, _leaf_evidence, _norm, _rule_ids, joint_entropy, joint_kl,
    marginal_kl_sum, node_marginals,
)


# --------------------------------------------------------------------------- #
# weighted sum-product
# --------------------------------------------------------------------------- #

def _upward_w(up_leaf, rules, rule_w, nev=None):
    """Leaves->root sum-product with per-cell mixture weights.

    rule_w[d]: (B, v, m) mixture over level-d rules. `oracle._upward`'s `/m` is the
    special case rule_w[d][:, a, r] == 1/m.
    """
    L = len(rules)
    v, m, s = rules[0].shape
    B = up_leaf.shape[0]
    up = [None] * (L + 1)
    up[L] = _norm(up_leaf if nev is None or nev[L] is None else up_leaf * nev[L])
    for d in range(L - 1, -1, -1):
        n_par = s ** d
        up_child = up[d + 1].reshape(B, n_par, s, v)
        rd, wd = rules[d], rule_w[d]
        out = np.zeros((B, n_par, v))
        for a in range(v):
            acc = np.zeros((B, n_par))
            for r in range(m):
                tup = rd[a, r]
                prod = np.ones((B, n_par))
                for k in range(s):
                    prod = prod * up_child[:, :, k, tup[k]]
                acc += wd[:, a, r][:, None] * prod
            out[:, :, a] = acc
        if nev is not None and nev[d] is not None:
            out = out * nev[d]
        up[d] = _norm(out)
    return up


def _downward_w(up, rules, rule_w, root_p, nev=None):
    """Root->leaves sum-product. down[d] EXCLUDES node d's own potential."""
    L = len(rules)
    v, m, s = rules[0].shape
    B = up[0].shape[0]
    down = [None] * (L + 1)
    down[0] = _norm(np.asarray(root_p, dtype=np.float64)[:, None, :])
    for d in range(L):
        n_par = s ** d
        up_child = up[d + 1].reshape(B, n_par, s, v)
        rd, wd = rules[d], rule_w[d]
        par = down[d] if nev is None or nev[d] is None else down[d] * nev[d]
        out = np.zeros((B, n_par, s, v))
        for a in range(v):
            da = par[:, :, a]
            for r in range(m):
                tup = rd[a, r]
                w = wd[:, a, r][:, None]
                ups = [up_child[:, :, k, tup[k]] for k in range(s)]
                for i in range(s):
                    prod_except_i = np.ones((B, n_par))
                    for k in range(s):
                        if k != i:
                            prod_except_i = prod_except_i * ups[k]
                    out[:, :, i, tup[i]] += da * w * prod_except_i
        down[d + 1] = _norm(out.reshape(B, n_par * s, v))
    return down


def clique_marginals_w(up, down, rules, rule_w, d, nev=None, ids=None):
    """P(parent = a AND its children take parent-a's u-th distinct tuple), (B, s^d, v, m)."""
    v, m, s = rules[0].shape
    B = up[0].shape[0]
    n_par = s ** d
    if ids is None:
        ids = _rule_ids(rules, d)
    up_child = up[d + 1].reshape(B, n_par, s, v)
    par = down[d] if nev is None or nev[d] is None else down[d] * nev[d]
    rd, wd = rules[d], rule_w[d]
    C = np.zeros((B, n_par, v, m))
    for a in range(v):
        da = par[:, :, a]
        for r in range(m):
            tup = rd[a, r]
            full = np.ones((B, n_par))
            for k in range(s):
                full = full * up_child[:, :, k, tup[k]]
            C[:, :, a, ids[a, r]] += da * full * wd[:, a, r][:, None]
    z = C.sum(axis=(2, 3), keepdims=True)
    return C / np.clip(z, EPS, None)


def prefix_beliefs_w(rules, seqs, plen, Dmax, rule_w, root_p, ids=None):
    """Node + clique marginals of P(z | x_{<plen}, theta), plus P(x_plen | x_{<plen}, theta)."""
    v = rules[0].shape[0]
    up = _upward_w(_leaf_evidence(seqs, plen, v), rules, rule_w)
    down = _downward_w(up, rules, rule_w, root_p)
    nodes = node_marginals(up, down, Dmax)
    cliques = [clique_marginals_w(up, down, rules, rule_w, d,
                                  ids=None if ids is None else ids[d])
               for d in range(Dmax)]
    leaf_post = None
    if plen < seqs.shape[1]:
        L = len(rules)
        leaf_post = _norm((up[L] * down[L])[:, plen, :][:, None, :])[:, 0, :]
    return nodes, cliques, leaf_post


def irreducible_entropy_w(rules, seqs, level_features, D, plen, rule_w, root_p):
    """H(x_plen | z_{<=D}, theta, x_{<plen}) exactly: clamp the level-D ancestor of x_plen."""
    v, m, s = rules[0].shape
    L = len(rules)
    B, T = seqs.shape
    span = s ** (L - D)
    j = plen // span
    lo = j * span
    sub_rules = rules[D:]
    sub_w = rule_w[D:]
    sub_seqs = seqs[:, lo:lo + span]
    n_obs = max(0, min(span, plen - lo))
    ev = _leaf_evidence(sub_seqs, n_obs, v)
    rp = np.zeros((B, v))
    rp[np.arange(B), level_features[D][:, j]] = 1.0
    up = _upward_w(ev, sub_rules, sub_w)
    down = _downward_w(up, sub_rules, sub_w, rp)
    post = _norm((up[L - D] * down[L - D])[:, plen - lo, :][:, None, :])[:, 0, :]
    return _ent(post, axes=1), post


# --------------------------------------------------------------------------- #
# the structure channel
# --------------------------------------------------------------------------- #

def structure_channel(rules, seqs, level_features, rule_w, root_p, Ds=None, chunk=256,
                      verbose=True, anc_mask=None):
    """The structure channel and its aleatoric residue, conditional on the true theta.

      S_joint[D]  (n, T-1)  KL( P(z_{<=D}|theta,x_{<=t+1}) || P(z_{<=D}|theta,x_{<=t}) )
      S_marg[D]             sum of per-node KLs (the form-matched control)
      S_chain[D]            the same sum restricted to x_{t+1}'s own ancestor chain
      H_post[D]             H( P(z_{<=D}|theta,x_{<=t}) )
      H_irr[D]              H(x_{t+1} | theta, z_{<=D}, x_{<=t})   -- the ALEATORIC residue
      H_tot                 H(x_{t+1} | theta, x_{<=t})
      surprisal             -log P(x_{t+1}|theta,x_{<=t}), the theta-informed Bayes nll

    rule_w[d]: (n, v, m); root_p: (n, v). Names deliberately mirror `oracle.py`'s so the
    drift-off comparison is a field-by-field diff.
    """
    v, m, s = rules[0].shape
    L = len(rules)
    n, T = seqs.shape
    if Ds is None:
        Ds = list(range(L))
    Dmax = max(Ds)

    out = {k: {D: np.zeros((n, T - 1)) for D in Ds}
           for k in ("S_joint", "S_marg", "S_chain", "H_post", "H_irr")}
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
        rw = [rule_w[d][c0:c1] for d in range(L)]
        rp = root_p[c0:c1]
        prev = prefix_beliefs_w(rules, sq, 1, Dmax, rw, rp, ids)
        for t in range(T - 1):
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
            cur = prefix_beliefs_w(rules, sq, t + 2, Dmax, rw, rp, ids)
            chain = np.zeros((c1 - c0, L))
            for d in range(Dmax + 1):
                j = (t + 1) // (s ** (L - d))
                chain[:, d] = _kl(cur[0][d][:, j], prev[0][d][:, j], axes=1)
            for D in Ds:
                out["S_joint"][D][c0:c1, t] = joint_kl(cur[0], prev[0], cur[1], prev[1],
                                                       D, s)
                out["S_marg"][D][c0:c1, t] = marginal_kl_sum(cur[0], prev[0], D)
                out["S_chain"][D][c0:c1, t] = chain[:, :D + 1].sum(1)
                out["H_post"][D][c0:c1, t] = joint_entropy(prev[0], prev[1], D, s)
                h_irr, _ = irreducible_entropy_w(rules, sq, lf, D, t + 1, rw, rp)
                out["H_irr"][D][c0:c1, t] = h_irr
            prev = cur
        if verbose:
            print(f"  structure oracle: {c1}/{n} sequences", flush=True)

    res = {k: out[k] for k in out}
    res.update({"H_tot": H_tot, "surprisal": surprisal, "Ds": Ds, "L": L, "s": s,
                "bayes_acc_chain": (None if anc_mask is None else
                                    {f"d{L - d}": float(bayes_hit[d] / max(bayes_cnt[d], 1))
                                     for d in range(L)})})
    return res


def conditional_self_check(res, tol=0.02):
    """E[S_D | theta] == E[H(x|theta,prefix) - H(x|theta,z_{<=D},prefix)], per D."""
    L = res["L"]
    rows, ok = {}, True
    for D in res["Ds"]:
        lhs = float(res["S_joint"][D].mean())
        rhs = float((res["H_tot"] - res["H_irr"][D]).mean())
        err = abs(lhs - rhs)
        rel = err / max(abs(rhs), 1e-9)
        rows[f"D{D}"] = {"D": D, "d_name": f"d{L - D}", "mean_S_joint": lhs,
                         "mean_H_tot_minus_H_irr": rhs, "abs_err": err, "rel_err": rel,
                         "mean_H_irr": float(res["H_irr"][D].mean()),
                         "passed": bool(err < tol or rel < tol)}
        ok = ok and rows[f"D{D}"]["passed"]
    return {"per_D": rows, "passed": bool(ok),
            "mean_H_tot": float(res["H_tot"].mean()),
            "mean_surprisal": float(res["surprisal"].mean())}


def uniform_w(rules, n):
    """(rule_w, root_p) for the plain RHM -- what drift-off must reduce to."""
    v, m, s = rules[0].shape
    return ([np.full((n, v, m), 1.0 / m) for _ in rules],
            np.full((n, v), 1.0 / v))


# --------------------------------------------------------------------------- #
# self-tests (run this module directly)
# --------------------------------------------------------------------------- #

def _test_reduces_to_incumbent():
    """FIDELITY: at uniform weights every field must equal `oracle.py`'s to machine
    precision. The names differ (S_* vs B_*) and nothing else may."""
    from rhm.conditional_revision import oracle as ORC
    from rhm.rhm_data import generate_rules_distinct
    from rhm.rhm_latent_loop import _generate_with_traces

    v, s, L, m = 6, 2, 4, 3
    rules = generate_rules_distinct(v, s, L, m, seed=1)
    seqs, lf, _ = _generate_with_traces(rules, 300, 11)
    rw, rp = uniform_w(rules, seqs.shape[0])
    a = ORC.revision_and_entropy(rules, seqs, lf, chunk=150, verbose=False)
    b = structure_channel(rules, seqs, lf, rw, rp, chunk=150, verbose=False)
    worst = 0.0
    for ka, kb in (("B_joint", "S_joint"), ("B_marg", "S_marg"), ("B_chain", "S_chain"),
                   ("H_post", "H_post"), ("H_irr", "H_irr")):
        for D in a["Ds"]:
            worst = max(worst, float(np.abs(a[ka][D] - b[kb][D]).max()))
    for k in ("H_tot", "surprisal"):
        worst = max(worst, float(np.abs(a[k] - b[k]).max()))
    print(f"  drift-off fidelity vs conditional_revision/oracle.py: max|delta| = {worst:.3e}")
    assert worst < 1e-13, f"weighted BP does not reduce to the incumbent ({worst})"


def _enumerate_weighted(rules, root_p, rule_w):
    """Every (latent config -> leaf sequence) with its probability under (root_p, rule_w)."""
    import itertools
    v, m, s = rules[0].shape
    L = len(rules)
    n_internal = (s ** L - 1) // (s - 1)
    lvl_of = np.concatenate([np.full(s ** d, d) for d in range(L)])
    Z, X, P = [], [], []
    for root in range(v):
        for choices in itertools.product(range(m), repeat=n_internal):
            ci, current, nodes = 0, [root], [root]
            p = root_p[root]
            for d in range(L):
                nxt = []
                for node in current:
                    p *= rule_w[d][node, choices[ci]]
                    nxt.extend(rules[d][node, choices[ci]].tolist())
                    ci += 1
                current = nxt
                if d < L - 1:
                    nodes.extend(current)
            Z.append(nodes); X.append(current); P.append(p)
    return np.array(Z), np.array(X), np.array(P)


def _bf_post(Z, X, P, prefix, n_nodes):
    match = np.all(X[:, :len(prefix)] == np.asarray(prefix)[None, :], axis=1)
    w = P * match
    w = w / w.sum()
    keys = {}
    for zi, wi in zip(Z[:, :n_nodes], w):
        if wi > 0:
            keys[tuple(zi)] = keys.get(tuple(zi), 0.0) + wi
    return keys


def _test_brute_force_weighted():
    from rhm.rhm_data import generate_rules_distinct
    for (v, s, L, m, sd) in [(2, 2, 3, 2, 3), (3, 2, 2, 2, 5)]:
        rules = generate_rules_distinct(v, s, L, m, seed=1)
        rng = np.random.default_rng(sd)
        rl = rng.normal(size=v) * 1.2
        root_p = np.exp(rl - rl.max()); root_p /= root_p.sum()
        rw = []
        for _d in range(L):
            z = rng.normal(size=(v, m)) * 1.1
            e = np.exp(z - z.max(1, keepdims=True))
            rw.append(e / e.sum(1, keepdims=True))
        Z, X, P = _enumerate_weighted(rules, root_p, rw)
        uniq = np.unique(X, axis=0)
        B = uniq.shape[0]
        rwb = [np.repeat(w[None], B, 0) for w in rw]
        rpb = np.repeat(root_p[None], B, 0)
        T = s ** L
        mk, me, mm = 0.0, 0.0, 0.0
        for D in range(L):
            n_nodes = (s ** (D + 1) - 1) // (s - 1)
            for plen in range(1, T):
                on, oc, _ = prefix_beliefs_w(rules, uniq, plen, D, rwb, rpb)
                nn, nc, _ = prefix_beliefs_w(rules, uniq, plen + 1, D, rwb, rpb)
                kl_bp = joint_kl(nn, on, nc, oc, D, s)
                ent_bp = joint_entropy(on, oc, D, s)
                for b in range(B):
                    q = _bf_post(Z, X, P, uniq[b, :plen].tolist(), n_nodes)
                    p = _bf_post(Z, X, P, uniq[b, :plen + 1].tolist(), n_nodes)
                    mk = max(mk, abs(sum(pv * np.log(pv / q[k]) for k, pv in p.items())
                                     - kl_bp[b]))
                    me = max(me, abs(-sum(qv * np.log(qv) for qv in q.values())
                                     - ent_bp[b]))
                    for d in range(D + 1):
                        off = (s ** d - 1) // (s - 1)
                        for j in range(s ** d):
                            ref = np.zeros(v)
                            for k, qv in q.items():
                                ref[k[off + j]] += qv
                            mm = max(mm, np.abs(ref - on[d][b, j]).max())
        print(f"  weighted v{v}/s{s}/L{L}/m{m}: max|KL_bf-KL_bp| = {mk:.2e}   "
              f"max|H| = {me:.2e}   max node-marg = {mm:.2e}")
        assert mk < 1e-8 and me < 1e-8 and mm < 1e-9


def _test_conditional_identity():
    from rhm.rhm_data import generate_rules_distinct
    from rhm.question_model import demand_state as DS
    v, s, L, m = 6, 2, 4, 3
    rules = generate_rules_distinct(v, s, L, m, seed=1)
    spec = DS.make_spec(K=5, sigma_s=1.0, kappa=0.2, L=L)
    dirs = DS.make_directions(rules, spec)
    seqs, lf, _lr, tidx = DS.sample_corpus(rules, spec, dirs, 4000, 3)
    W = DS.all_weights(spec, dirs)
    rw = [np.stack([W[i][1][d] for i in tidx]) for d in range(L)]
    rp = np.stack([W[i][0] for i in tidx])
    res = structure_channel(rules, seqs, lf, rw, rp, chunk=1000, verbose=False)
    chk = conditional_self_check(res)
    print(f"  conditional identity on v{v}/s{s}/L{L}/m{m} under drift, n=4000:")
    for k, r in chk["per_D"].items():
        print(f"    {k} ({r['d_name']}): E[S] {r['mean_S_joint']:.5f}  "
              f"E[H_tot-H_irr] {r['mean_H_tot_minus_H_irr']:.5f}  "
              f"abs_err {r['abs_err']:.2e}  {'ok' if r['passed'] else 'FAIL'}")
    assert chk["passed"], "the conditional S / (H_tot - H_irr) identity does not hold"


if __name__ == "__main__":
    print("=" * 74)
    print("oracle_q self-test: weighted BP")
    print("=" * 74)
    _test_reduces_to_incumbent()
    _test_brute_force_weighted()
    _test_conditional_identity()
    print("oracle_q self-test PASSED")
