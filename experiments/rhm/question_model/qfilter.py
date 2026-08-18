"""The DEMAND channel: an exact forward filter over the question state.

    Dm_t  =  KL( P(theta | x_{<=t+1}) || P(theta | x_{<=t}) )

Because the world's theta lives on a finite grid (`demand_state.py`), this filter is exactly
Bayes-optimal -- no particles, no Laplace approximation, no grid error. What it needs from the
substrate is one array:

    loglik[i, g, p]  =  log P( x_p | x_{<p}, theta = g )     for p = 0 .. T-1

and everything else (posteriors, the demand revision, the theta-marginal surprisal, the
per-component decomposition) is arithmetic on it.

HOW `loglik` IS COMPUTED IN O(T L) INSTEAD OF O(T 2^L)
-----------------------------------------------------
The naive route is `oracle_q.prefix_beliefs_w` once per prefix length, which is a full
upward+downward sweep over 2^L nodes for each of T prefixes and each of K^C grid states. But an
unobserved subtree's inside vector is EXACTLY the all-ones vector (sum_r w[a,r] prod_k 1 = 1 for
any weights), so for a prefix of length p only the maximal complete subtrees to the left and the
single right frontier matter. Keeping the complete subtrees on a binary-counter stack and walking
the frontier gives every prefix likelihood in O(L) work per token. The saving is ~10x at L = 6
and it is what makes a 343-state filter affordable in numpy.

Insides are stored normalised with an explicit log-normaliser, so nothing underflows and the
log-likelihoods are exact rather than rescaled.

TWO CONDITIONING SETS, AND WHY BOTH ARE REPORTED
------------------------------------------------
  window  -- the filter starts each block from the OU stationary prior. This is the conditioning
             set a T-token-window transformer actually has, so it is the ceiling the laundering
             probe is measured against.
  history -- the filter carries its posterior across blocks through the OU kernel, i.e. an ideal
             learner with unbounded memory. Its channel is what the drift keeps alive: under a
             static theta it decays to zero as the state is identified, which is the sanity
             identity `revision_not_surprisal`'s SS4 has no analogue for.

The gap between them is the part of the question that a finite context window cannot hold and
weights cannot track -- which is the structural reason the demand channel is a candidate for a
separate slow organ rather than more of the same.
"""

import numpy as np

EPS = 1e-300


# --------------------------------------------------------------------------- #
# prefix likelihoods
# --------------------------------------------------------------------------- #

def _combine(rules, d, wd, left_v, left_z, right_v, right_z):
    """Inside vector of a level-d parent from its two level-(d+1) children.

    left_v/right_v: (n, G, v) normalised; wd: (G, v, m); returns (vec, logz).
    """
    v, m, s = rules[0].shape
    assert s == 2, "the frontier walk is written for s = 2"
    rd = rules[d]
    n, G, _ = left_v.shape
    raw = np.zeros((n, G, v))
    for a in range(v):
        acc = np.zeros((n, G))
        for r in range(m):
            t0, t1 = rd[a, r]
            acc += wd[None, :, a, r] * left_v[:, :, t0] * right_v[:, :, t1]
        raw[:, :, a] = acc
    z = raw.sum(-1)
    return raw / np.clip(z[..., None], EPS, None), left_z + right_z + np.log(np.clip(z, EPS, None))


def prefix_loglik(rules, seqs, rule_w_states, root_p_states, chunk=64, verbose=False):
    """log P(x_p | x_{<p}, theta_g) for every (sequence, grid state, position).

    rule_w_states[d]: (G, v, m); root_p_states: (G, v). Returns (n, G, T) float64.
    """
    v, m, s = rules[0].shape
    L = len(rules)
    n, T = seqs.shape
    G = root_p_states.shape[0]
    out = np.zeros((n, G, T))
    ones_v = np.full(v, 1.0 / v)
    log_v = np.log(v)

    for c0 in range(0, n, chunk):
        c1 = min(n, c0 + chunk)
        nb = c1 - c0
        onesv = np.broadcast_to(ones_v, (nb, G, v)).copy()
        onesz = np.full((nb, G), log_v)
        stack = []                     # list of (level, vec, logz), deepest last
        logZ = np.zeros((nb, G))       # log P(x_{<p}) at the current p
        for p in range(T):
            # ---- log P(x_{<p+1}) with the leaf p pushed -------------------
            oh = np.zeros((nb, G, v))
            oh[np.arange(nb)[:, None], np.arange(G)[None, :], seqs[c0:c1, p][:, None]] = 1.0
            st = stack + [(L, oh, np.zeros((nb, G)))]
            # merge equal-level siblings (binary counter)
            while len(st) >= 2 and st[-1][0] == st[-2][0]:
                d = st[-1][0]
                (_, lv, lz), (_, rv, rz) = st[-2], st[-1]
                st = st[:-2] + [(d - 1, *_combine(rules, d - 1, rule_w_states[d - 1],
                                                  lv, lz, rv, rz))]
            ent = {e[0]: e for e in st}
            acc_v, acc_z = onesv, onesz
            for d in range(L, 0, -1):
                if d in ent:
                    acc_v, acc_z = _combine(rules, d - 1, rule_w_states[d - 1],
                                            ent[d][1], ent[d][2], acc_v, acc_z)
                else:
                    acc_v, acc_z = _combine(rules, d - 1, rule_w_states[d - 1],
                                            acc_v, acc_z, onesv, onesz)
            if 0 in ent:
                acc_v, acc_z = ent[0][1], ent[0][2]
            tot = np.log(np.clip((root_p_states[None] * acc_v).sum(-1), EPS, None)) + acc_z
            out[c0:c1, :, p] = tot - logZ
            logZ = tot
            stack = st
        if verbose:
            print(f"  filter likelihoods: {c1}/{n} sequences", flush=True)
    return out


# --------------------------------------------------------------------------- #
# the filter
# --------------------------------------------------------------------------- #

def _kl_rows(p, q):
    """KL(p||q) row-wise over the last axis."""
    r = np.log(np.clip(p, EPS, None)) - np.log(np.clip(q, EPS, None))
    return np.where(p > 0, p * r, 0.0).sum(-1)


def window_posteriors(loglik, prior):
    """Posterior over grid states after each prefix length.

    Returns w (n, T+1, G): w[:, p] = P(theta | x_{<p}). w[:, 0] = prior.
    """
    n, G, T = loglik.shape
    cum = np.concatenate([np.zeros((n, G, 1)), np.cumsum(loglik, axis=2)], axis=2)
    lp = np.log(np.clip(prior, EPS, None))
    if lp.ndim == 1:
        lp = np.broadcast_to(lp, (n, G))
    z = cum + lp[:, :, None]
    z = z - z.max(axis=1, keepdims=True)
    w = np.exp(z)
    w = w / w.sum(axis=1, keepdims=True)
    return np.transpose(w, (0, 2, 1))


def history_priors(loglik, prior0, P_trans, block_order=None):
    """Per-block priors for an unbounded-memory learner: the previous block's terminal
    posterior pushed through the OU kernel. `loglik` rows must be in corpus order."""
    n, G, T = loglik.shape
    order = np.arange(n) if block_order is None else block_order
    tot = loglik.sum(axis=2)                       # (n, G) log P(block | theta)
    out = np.zeros((n, G))
    cur = np.asarray(prior0, dtype=np.float64).copy()
    for i in order:
        out[i] = cur
        lz = np.log(np.clip(cur, EPS, None)) + tot[i]
        lz -= lz.max()
        post = np.exp(lz)
        post /= post.sum()
        cur = post @ P_trans
    return out


def demand_channel(loglik, prior, state_idx=None, n_comp=0, K=0):
    """The demand channel and the theta-marginal surprisal.

      Dm        (n, T-1)  KL( P(theta|x_{<=t+1}) || P(theta|x_{<=t}) )
      surprisal (n, T-1)  -log sum_g P(theta_g|x_{<=t}) P(x_{t+1}|x_{<=t}, theta_g)
      Dm_comp   {c: (n, T-1)} the same KL on each component's MARGINAL posterior.
      H_theta   (n, T)    H( P(theta|x_{<p}) ), the residual question uncertainty
    """
    n, G, T = loglik.shape
    w = window_posteriors(loglik, prior)           # (n, T+1, G)
    pre = w[:, 1:T, :]                             # P(theta | x_{<=t}),   t = 0..T-2
    post = w[:, 2:T + 1, :]                        # P(theta | x_{<=t+1})
    Dm = _kl_rows(post, pre)
    ll = np.transpose(loglik, (0, 2, 1))[:, 1:T, :]        # log P(x_{t+1}|x_{<=t},theta)
    mx = ll.max(-1, keepdims=True)
    surp = -(np.log(np.clip((pre * np.exp(ll - mx)).sum(-1), EPS, None)) + mx[..., 0])
    out = {"Dm": Dm, "surprisal": surp,
           "H_theta": -np.where(w > 0, w * np.log(np.clip(w, EPS, None)), 0.0).sum(-1),
           "post_full": w}
    if state_idx is not None and n_comp:
        comp = {}
        for c in range(n_comp):
            M = np.zeros((G, K))
            M[np.arange(G), state_idx[:, c]] = 1.0
            comp[c] = _kl_rows(post @ M, pre @ M)
        out["Dm_comp"] = comp
        out["marg_pre"] = {c: (pre @ np.eye(K)[state_idx[:, c]]) for c in range(n_comp)}
        out["marg_post"] = {c: (post @ np.eye(K)[state_idx[:, c]]) for c in range(n_comp)}
    return out


# --------------------------------------------------------------------------- #
# the three-way identity
# --------------------------------------------------------------------------- #

def three_way_check(surp_marg, Dm, S_joint, H_irr, Ds, L, tol=0.02):
    """The extended SS4 identity, one term wider:

        E[ -log P(x_{t+1}|x_{<=t}) ]  =  E[Dm]  +  E[S_D | theta]  +  E[H(x|theta,z_{<=D},x)]
          total surprisal                demand    structure          aleatoric residue

    `S_joint` and `H_irr` come from `oracle_q.structure_channel` at the TRUE theta; `Dm` and
    `surp_marg` from `demand_channel`. Exact in expectation because the realised (theta, z,
    x_{t+1}) of each sequence is a draw from the conditional the identity averages over.

    Also reports the D-independent half, which isolates the new term:

        E[ -log P(x|x_{<=t}) ]  -  E[ -log P(x|theta,x_{<=t}) ]  =  E[Dm]
    """
    rows, ok = {}, True
    tot = float(np.mean(surp_marg))
    dm = float(np.mean(Dm))
    for D in Ds:
        st = float(np.mean(S_joint[D]))
        al = float(np.mean(H_irr[D]))
        err = abs(tot - (dm + st + al))
        rows[f"D{D}"] = {"D": D, "d_name": f"d{L - D}", "total": tot, "demand": dm,
                         "structure": st, "aleatoric": al, "sum": dm + st + al,
                         "abs_err": err, "rel_err": err / max(tot, 1e-9),
                         "passed": bool(err < tol or err / max(tot, 1e-9) < tol)}
        ok = ok and rows[f"D{D}"]["passed"]
    return {"per_D": rows, "passed": bool(ok), "mean_total": tot, "mean_demand": dm}


# --------------------------------------------------------------------------- #
# self-tests
# --------------------------------------------------------------------------- #

def _test_loglik_matches_bp():
    """The O(T L) frontier walk must agree with `oracle_q.prefix_beliefs_w`'s leaf
    posteriors, which are themselves brute-force verified."""
    from rhm.rhm_data import generate_rules_distinct
    from rhm.question_model import demand_state as DS
    from rhm.question_model import oracle_q as OQ

    for (v, s, L, m, K) in [(4, 2, 3, 2, 3), (6, 2, 4, 3, 3), (5, 2, 2, 4, 4)]:
        rules = generate_rules_distinct(v, s, L, m, seed=1)
        spec = DS.make_spec(K=K, sigma_s=1.2, kappa=0.2, L=L)
        dirs = DS.make_directions(rules, spec)
        seqs, lf, _lr, tidx = DS.sample_corpus(rules, spec, dirs, 40, 5)
        W = DS.all_weights(spec, dirs)
        G = len(W)
        rw_s = [np.stack([W[g][1][d] for g in range(G)]) for d in range(L)]
        rp_s = np.stack([W[g][0] for g in range(G)])
        ll = prefix_loglik(rules, seqs, rw_s, rp_s, chunk=20)
        T = s ** L
        worst = 0.0
        for g in (0, G // 2, G - 1):
            rwb = [np.repeat(rw_s[d][g][None], seqs.shape[0], 0) for d in range(L)]
            rpb = np.repeat(rp_s[g][None], seqs.shape[0], 0)
            for p in range(T):
                _n, _c, lpost = OQ.prefix_beliefs_w(rules, seqs, p, 0, rwb, rpb)
                ref = np.log(lpost[np.arange(seqs.shape[0]), seqs[:, p]])
                worst = max(worst, float(np.abs(ref - ll[:, g, p]).max()))
        print(f"  v{v}/s{s}/L{L}/m{m}/K{K}: max|loglik - BP leaf posterior| = {worst:.2e}")
        assert worst < 1e-10, f"frontier walk disagrees with BP ({worst})"


def _test_demand_identity():
    """The D-independent half of the three-way identity, on a small regime."""
    from rhm.rhm_data import generate_rules_distinct
    from rhm.question_model import demand_state as DS
    from rhm.question_model import oracle_q as OQ

    v, s, L, m = 6, 2, 4, 3
    rules = generate_rules_distinct(v, s, L, m, seed=1)
    spec = DS.make_spec(K=5, sigma_s=1.0, kappa=0.2, L=L)
    dirs = DS.make_directions(rules, spec)
    n = 3000
    seqs, lf, _lr, tidx = DS.sample_corpus(rules, spec, dirs, n, 3)
    W = DS.all_weights(spec, dirs)
    G = len(W)
    rw_s = [np.stack([W[g][1][d] for g in range(G)]) for d in range(L)]
    rp_s = np.stack([W[g][0] for g in range(G)])
    ll = prefix_loglik(rules, seqs, rw_s, rp_s, chunk=250)
    idx, pj = DS.state_grid(spec)
    dc = demand_channel(ll, pj, state_idx=idx, n_comp=idx.shape[1], K=spec["K"])
    # theta-informed surprisal from the same likelihood array (exact, same object)
    surp_theta = -ll[np.arange(n), tidx, :][:, 1:]
    lhs = float(dc["surprisal"].mean() - surp_theta.mean())
    rhs = float(dc["Dm"].mean())
    print(f"  E[surp_marg] - E[surp_theta] = {lhs:.6f}   E[Dm] = {rhs:.6f}   "
          f"abs_err {abs(lhs - rhs):.2e}")
    assert abs(lhs - rhs) < 0.01 * max(rhs, 1e-3) + 0.002, "demand identity fails"
    print(f"  H(theta) = {DS.theta_entropy(spec):.3f} nats; residual H after the block = "
          f"{dc['H_theta'][:, -1].mean():.3f}; per-token demand = {rhs:.4f} nats")


def _test_drift_off():
    """sigma_s = 0: one state, Dm identically zero, surp_marg == the incumbent's."""
    from rhm.rhm_data import generate_rules_distinct
    from rhm.conditional_revision import oracle as ORC
    from rhm.question_model import demand_state as DS

    v, s, L, m = 6, 2, 4, 3
    rules = generate_rules_distinct(v, s, L, m, seed=1)
    spec = DS.make_spec(K=7, sigma_s=0.0, kappa=0.2, L=L)
    dirs = DS.make_directions(rules, spec)
    seqs, lf, _lr, tidx = DS.sample_corpus(rules, spec, dirs, 200, 3)
    W = DS.all_weights(spec, dirs)
    rw_s = [np.stack([W[g][1][d] for g in range(len(W))]) for d in range(L)]
    rp_s = np.stack([W[g][0] for g in range(len(W))])
    ll = prefix_loglik(rules, seqs, rw_s, rp_s, chunk=100)
    idx, pj = DS.state_grid(spec)
    dc = demand_channel(ll, pj)
    ref = ORC.revision_and_entropy(rules, seqs, lf, Ds=[0], chunk=100, verbose=False)
    e1 = float(np.abs(dc["Dm"]).max())
    e2 = float(np.abs(dc["surprisal"] - ref["surprisal"]).max())
    print(f"  drift-off: max|Dm| = {e1:.2e}   max|surp_marg - oracle.surprisal| = {e2:.2e}")
    assert e1 == 0.0 and e2 < 1e-12, "drift-off does not reduce to the incumbent"


if __name__ == "__main__":
    print("=" * 74)
    print("qfilter self-test: exact question-state filter")
    print("=" * 74)
    _test_loglik_matches_bp()
    _test_drift_off()
    _test_demand_identity()
    print("qfilter self-test PASSED")
