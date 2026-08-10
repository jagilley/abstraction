"""Exact belief-propagation oracle for a MIXTURE over R rule sets.

The single-rule-set oracle (`../oracle.py`) computes belief revision `B` exactly on the
known parse tree. Here the rule set itself is latent: a context window is K whole
sequences drawn i.i.d. from ONE of R rule sets, and the reader must infer which. The
latent variable is therefore the pair

    (r, z_{<=D})     r = the active rule set, z = the CURRENT sequence's latents

and the exact revision splits by the chain rule for KL into two terms on two timescales:

    B_total(n) = KL( w(n+1) || w(n) )              RULE revision  -- slow, decays as
                                                   the rule posterior concentrates
               + E_{r ~ w(n+1)} [ B_D^{(r)}(n) ]   PARSE revision -- fast, per-token,
                                                   and each B^{(r)} is EXACTLY the
                                                   single-rule oracle unchanged

with `w(n)` = P(r | x_{<n}) over the whole window. Sequences are i.i.d. given r, so past
sequences enter only through `w`; given (r, z of the current sequence) nothing earlier
predicts the next token. That is what makes the split exact and the identity survive:

    E[ B_total,D ]  =  H(x_n | x_{<n})      -   H(x_n | r, z_{<=D}, x_{<n})
                       MIXTURE surprisal        the TRUE rule set's irreducible term
                       (H_tot_mix)              (H_irr, unchanged from ../oracle.py)

and, as the D-free special case that isolates the slow term,

    E[ KL(w(n+1) || w(n)) ]  =  H_tot_mix(n)  -  H_tot^{(r*)}(n)

i.e. **the rule revision equals the excess surprisal of not knowing the rules**. Its sum
over a window is I(r ; x_window) <= ln R, so ln R is a hard budget on the entire slow
component. That bound is the main design fact this module exists to expose: a family can
only ever supply ln R nats of rule-revision, spread over however many tokens
identification takes, and if identification is fast that budget is spent in a handful of
positions and there is no decaying trajectory to find. Gate -1 measures the spread before
any GPU time is spent.

Nothing here modifies `../oracle.py`. The BP primitives are imported from it, so the
single-rule results stay bit-identical by construction rather than by test, and the
mixture layer consumes only per-rule predictive posteriors -- it does not care whether
the rule sets differ in their tables, their weights, or anything else.

Run the self-test (brute-force enumeration over (r, latents) on tiny trees):
    cd experiments && python3 -m rhm.conditional_revision.rule_family.oracle_mixture
"""

import numpy as np

from rhm.conditional_revision.oracle import (
    EPS, _ent, _kl, _rule_ids, irreducible_entropy, joint_kl, prefix_beliefs,
    revision_and_entropy,
)

NEG = -1e300


# ---------------------------------------------------------------------------
# per-rule-set predictive posteriors
# ---------------------------------------------------------------------------

def predictive_posteriors(rules, seqs, chunk=512, verbose=False):
    """Exact P(x_t = a | x_{<t}, rules) for every position. Returns (n, T, v).

    This is the only quantity the mixture layer needs from BP. `prefix_beliefs` with
    Dmax=0 runs upward+downward and skips every clique marginal, so this is the cheap
    pass -- no junction-tree work, no H_irr.
    """
    v = rules[0].shape[0]
    n, T = seqs.shape
    out = np.zeros((n, T, v))
    for c0 in range(0, n, chunk):
        c1 = min(n, c0 + chunk)
        sq = seqs[c0:c1]
        for t in range(T):
            _, _, leaf_post = prefix_beliefs(rules, sq, t, 0)
            out[c0:c1, t] = leaf_post
        if verbose:
            print(f"    predictive: {c1}/{n}", flush=True)
    return out


def family_predictive(family, seqs, chunk=512, verbose=False):
    """(R, n, T, v) predictive posteriors, one slab per rule set."""
    return np.stack([predictive_posteriors(rl, seqs, chunk, verbose) for rl in family])


def sequence_revision_all(rules, seqs, level_features, Ds, chunk=256):
    """`revision_and_entropy` extended to ALL T positions of a sequence.

    The incumbent indexes its signal axis as "the token x_{t+1} arrives", so it never
    computes the revision caused by a sequence's FIRST token -- the comparison there is
    against the prior, not against a prefix. Inside a window that position is a real one
    (it is where a fresh sequence starts, and it is where the parse belief resets while
    the RULE belief does not), and the joint identity has to hold there too, so this
    returns T columns where the incumbent returns T-1.

    Returns {"B_joint": {D: (n, T)}, "H_irr": {D: (n, T)}}, with column p meaning "token
    x_p arrived". `_check_matches_incumbent` asserts columns 1..T-1 reproduce
    `revision_and_entropy` exactly, so this is a strict superset and nothing in
    `../oracle.py` changes.
    """
    v, m, s = rules[0].shape
    L = len(rules)
    n, T = seqs.shape
    Dmax = max(Ds)
    ids = [_rule_ids(rules, d) for d in range(L)]
    B = {D: np.zeros((n, T)) for D in Ds}
    Hirr = {D: np.zeros((n, T)) for D in Ds}
    for c0 in range(0, n, chunk):
        c1 = min(n, c0 + chunk)
        sq = seqs[c0:c1]
        lf = [level_features[d][c0:c1] for d in range(L + 1)]
        prev = prefix_beliefs(rules, sq, 0, Dmax, ids)      # the prior, no evidence
        for p in range(T):
            cur = prefix_beliefs(rules, sq, p + 1, Dmax, ids)
            for D in Ds:
                B[D][c0:c1, p] = joint_kl(cur[0], prev[0], cur[1], prev[1], D, s)
                Hirr[D][c0:c1, p] = irreducible_entropy(rules, sq, lf, D, p)[0]
            prev = cur
    return {"B_joint": B, "H_irr": Hirr}


# ---------------------------------------------------------------------------
# the mixture layer
# ---------------------------------------------------------------------------

def _logsumexp(a, axis):
    mx = a.max(axis=axis, keepdims=True)
    mx = np.where(np.isfinite(mx), mx, 0.0)
    return (mx + np.log(np.exp(a - mx).sum(axis=axis, keepdims=True))).squeeze(axis)


def mixture_profiles(post, seqs, rule_ids, log_prior=None):
    """Everything the mixture oracle knows, per window and per GLOBAL position.

    post      (R, n*K, T, v) or (R, n, K, T, v) predictive posteriors per rule set
    seqs      (n, K, T) the windows
    rule_ids  (n,) the true rule set of each window
    log_prior (R,) optional non-uniform prior over rule sets

    Global position index is  g = k*T + t,  running 0 .. K*T-1 over the window.

    Returns a dict of (n, K*T) arrays unless noted:
      w_true      P(r = r* | x_{<g})              -- concentration profile
      H_rule      H(w(g))                         -- nats of remaining rule uncertainty
      rule_rev    KL(w(g+1) || w(g))              -- the SLOW revision component
      nll_mix     -log P(x_g | x_{<g})            -- mixture surprisal (what a Bayesian
                                                     reader who does not know r pays)
      nll_true    -log P(x_g | x_{<g}, r*)        -- the known-rules floor
      icl_gap     nll_mix - nll_true              -- per-token excess; its mean equals
                                                     E[rule_rev] by the identity above
      H_tot_mix   H(P(x_g | x_{<g}))              -- mixture predictive entropy
      H_tot_true  H(P(x_g | x_{<g}, r*))
      w_pre       (n, K*T, R) the rule posterior BEFORE token g,  w(g)
      w_aft       (n, K*T, R) the rule posterior AFTER  token g,  w(g+1)
    """
    R = post.shape[0]
    n, K, T = seqs.shape
    v = post.shape[-1]
    post = post.reshape(R, n, K, T, v)
    G = K * T
    lp_tok = np.log(np.clip(
        np.take_along_axis(post, seqs[None, :, :, :, None], axis=4)[..., 0],
        EPS, None))                                            # (R, n, K, T)
    lp_tok = lp_tok.reshape(R, n, G).transpose(1, 2, 0)         # (n, G, R)
    H_all = _ent(post.reshape(R, n, G, v), axes=3).transpose(1, 2, 0)   # (n, G, R)

    lprior = (np.zeros(R) - np.log(R)) if log_prior is None else np.asarray(log_prior)
    cum = np.concatenate([np.broadcast_to(lprior, (n, 1, R)),
                          lprior[None, None, :] + np.cumsum(lp_tok, axis=1)], axis=1)
    logw = cum - _logsumexp(cum, axis=2)[:, :, None]            # (n, G+1, R)
    w = np.exp(logw)

    idx = np.arange(n)
    w_pre, w_post_ = w[:, :G], w[:, 1:]
    logw_pre, logw_post = logw[:, :G], logw[:, 1:]

    nll_mix = -_logsumexp(logw_pre + lp_tok, axis=2)
    nll_true = -lp_tok[idx, :, rule_ids]
    rule_rev = (w_post_ * (logw_post - logw_pre)).sum(axis=2)
    H_tot_mix = _ent((w_pre[:, :, :, None] * post.reshape(R, n, G, v)
                      .transpose(1, 2, 0, 3)).sum(axis=2), axes=2)
    return {
        "w_true": w_pre[idx, :, rule_ids],
        "H_rule": _ent(w_pre, axes=2),
        "rule_rev": rule_rev,
        "nll_mix": nll_mix,
        "nll_true": nll_true,
        "icl_gap": nll_mix - nll_true,
        "H_tot_mix": H_tot_mix,
        "H_tot_true": H_all[idx, :, rule_ids],
        "w_pre": w_pre, "w_aft": w_post_,
        "R": R, "K": K, "T": T, "G": G,
    }


def parse_revision_mixture(family, seqs, rule_ids, level_features, prof, Ds=None,
                           chunk=256, verbose=False):
    """The FAST component: E_{r ~ w(g+1)}[ B_D^{(r)} ], plus the true rule set's H_irr.

    Runs the full per-sequence revision once per rule set over every sequence (R
    passes), because the expectation is over all rule sets carrying posterior mass, not
    only the true one. Every array is (n, K, T) on the same grid as the profiles, with
    global position g = k*T + p.

      parse_rev[D]   E_{r ~ w(g+1)}[ B_joint,D^{(r)} ]
      B_true[D]      B_joint,D^{(r*)}          -- what the incumbent oracle would report
      H_irr[D]       H(x | z_{<=D}, x_{<}, r*) -- unchanged from ../oracle.py
      B_total[D]     rule_rev + parse_rev[D]   -- the exact joint revision
    """
    R = len(family)
    n, K, T = seqs.shape
    L = len(family[0])
    if Ds is None:
        Ds = list(range(L))
    flat = seqs.reshape(n * K, T)
    lf_flat = [level_features[d].reshape(n * K, -1) for d in range(L + 1)]

    Bj = {D: np.zeros((R, n * K, T)) for D in Ds}
    Hirr = {D: np.zeros((n * K, T)) for D in Ds}
    for r in range(R):
        if verbose:
            print(f"  parse revision: rule set {r + 1}/{R}", flush=True)
        res = sequence_revision_all(family[r], flat, lf_flat, Ds=Ds, chunk=chunk)
        for D in Ds:
            Bj[D][r] = res["B_joint"][D]
        own = np.repeat(rule_ids, K) == r
        if own.any():
            for D in Ds:
                Hirr[D][own] = res["H_irr"][D][own]

    w_aft = prof["w_aft"].reshape(n, K, T, R)
    rule_rev = prof["rule_rev"].reshape(n, K, T)
    ridx = np.broadcast_to(rule_ids[:, None, None, None], (n, K, T, 1))
    out = {"parse_rev": {}, "B_true": {}, "H_irr": {}, "B_total": {},
           "rule_rev": rule_rev, "Ds": Ds}
    for D in Ds:
        b = Bj[D].reshape(R, n, K, T).transpose(1, 2, 3, 0)      # (n, K, T, R)
        out["parse_rev"][D] = (w_aft * b).sum(axis=3)
        out["B_true"][D] = np.take_along_axis(b, ridx, axis=3)[..., 0]
        out["H_irr"][D] = Hirr[D].reshape(n, K, T)
        out["B_total"][D] = rule_rev + out["parse_rev"][D]
    return out


def self_check_mixture(prof, parse=None, tol=0.02):
    """The two identities this module rests on, as Monte-Carlo averages.

      (1) E[ KL(w(n+1)||w(n)) ]      == E[ H_tot_mix - H_tot_true ]     (slow term)
      (2) E[ rule_rev + parse_rev_D ] == E[ H_tot_mix - H_irr_D ]        (joint, per D)

    Identity (1) needs no parse machinery and is the cheap gate on the whole design.
    """
    rows = {"rule_only": {
        "lhs_mean_rule_rev": float(prof["rule_rev"].mean()),
        "rhs_mean_H_tot_mix_minus_H_tot_true":
            float((prof["H_tot_mix"] - prof["H_tot_true"]).mean()),
    }}
    r = rows["rule_only"]
    r["abs_err"] = abs(r["lhs_mean_rule_rev"] - r["rhs_mean_H_tot_mix_minus_H_tot_true"])
    r["rel_err"] = r["abs_err"] / max(abs(r["rhs_mean_H_tot_mix_minus_H_tot_true"]), 1e-9)
    r["passed"] = bool(r["abs_err"] < tol or r["rel_err"] < tol)
    ok = r["passed"]

    if parse is not None:
        K, T = prof["K"], prof["T"]
        Hmix = prof["H_tot_mix"].reshape(-1, K, T)
        for D in parse["Ds"]:
            lhs = float(parse["B_total"][D].mean())
            rhs = float((Hmix - parse["H_irr"][D]).mean())
            err = abs(lhs - rhs)
            rows[f"D{D}"] = {
                "D": D, "mean_B_total": lhs, "mean_H_tot_mix_minus_H_irr": rhs,
                "abs_err": err, "rel_err": err / max(abs(rhs), 1e-9),
                "mean_rule_rev": float(parse["rule_rev"].mean()),
                "mean_parse_rev": float(parse["parse_rev"][D].mean()),
                "mean_B_true": float(parse["B_true"][D].mean()),
                "passed": bool(err < tol or err / max(abs(rhs), 1e-9) < tol),
            }
            ok = ok and rows[f"D{D}"]["passed"]
    return {"per_term": rows, "passed": bool(ok)}


# ---------------------------------------------------------------------------
# correctness self-test: brute-force enumeration over (r, latents)
# ---------------------------------------------------------------------------

def _enumerate_family(family, K):
    """Every (r, per-sequence latent config) -> window leaves, with probabilities.

    Returns (rs, Zs, Xs, P): rs (N,), Zs (N, K, n_internal), Xs (N, K, s^L),
    P (N,). Tiny trees only -- this is O(R * (v*m^n_internal)^K).
    """
    import itertools
    R = len(family)
    v, m, s = family[0][0].shape
    L = len(family[0])
    n_internal = (s ** L - 1) // (s - 1)

    per_rule = []
    for r in range(R):
        Z, X, P = [], [], []
        for root in range(v):
            for choices in itertools.product(range(m), repeat=n_internal):
                ci, current, nodes = 0, [root], [root]
                for d in range(L):
                    nxt = []
                    for node in current:
                        nxt.extend(family[r][d][node, choices[ci]].tolist())
                        ci += 1
                    current = nxt
                    if d < L - 1:
                        nodes.extend(current)
                Z.append(nodes); X.append(current)
                P.append((1.0 / v) * (1.0 / m) ** n_internal)
        per_rule.append((np.array(Z), np.array(X), np.array(P)))

    rs, Zs, Xs, Ps = [], [], [], []
    for r in range(R):
        Z, X, P = per_rule[r]
        for combo in itertools.product(range(Z.shape[0]), repeat=K):
            rs.append(r)
            Zs.append([Z[c] for c in combo])
            Xs.append([X[c] for c in combo])
            Ps.append((1.0 / R) * np.prod([P[c] for c in combo]))
    return np.array(rs), np.array(Zs), np.array(Xs), np.array(Ps)


def _bf_window(family, K, D, seqs_probe):
    """Brute-force mixture quantities for specific windows. Returns dict of (n, G)."""
    v, s = family[0][0].shape[0], family[0][0].shape[2]
    L = len(family[0])
    T = s ** L
    n_upto = (s ** (D + 1) - 1) // (s - 1)
    rs, Zs, Xs, Ps = _enumerate_family(family, K)
    Xflat = Xs.reshape(Xs.shape[0], K * T)
    n = seqs_probe.shape[0]
    G = K * T
    out = {k: np.zeros((n, G)) for k in
           ("w_true", "rule_rev", "nll_mix", "nll_true", "B_total", "H_rule")}
    for i in range(n):
        win = seqs_probe[i].reshape(-1)
        for g in range(G):
            k_seq, t = g // T, g % T
            pre = Xflat[:, :g] == win[None, :g]
            mpre = pre.all(axis=1)
            wpre = Ps * mpre
            wpre = wpre / wpre.sum()
            mpost = mpre & (Xflat[:, g] == win[g])
            wpost = Ps * mpost
            zpost = wpost.sum()
            wpost = wpost / zpost

            pr_pre = np.array([wpre[rs == r].sum() for r in range(len(family))])
            pr_post = np.array([wpost[rs == r].sum() for r in range(len(family))])
            out["w_true"][i, g] = pr_pre[0]           # caller probes rule set 0 windows
            out["H_rule"][i, g] = _ent(pr_pre[None, :], axes=1)[0]
            good = pr_post > 0
            out["rule_rev"][i, g] = (pr_post[good]
                                     * (np.log(pr_post[good]) - np.log(pr_pre[good]))).sum()
            p_tok = wpre[Xflat[:, g] == win[g]].sum()
            out["nll_mix"][i, g] = -np.log(p_tok)
            own = (rs == 0)
            wt = Ps * mpre * own
            out["nll_true"][i, g] = -np.log(
                (Ps * mpost * own).sum() / max(wt.sum(), 1e-300))

            # joint revision over (r, z_{<=D} of the CURRENT sequence)
            def joint(wts):
                d = {}
                for j in np.where(wts > 0)[0]:
                    key = (rs[j],) + tuple(Zs[j, k_seq, :n_upto])
                    d[key] = d.get(key, 0.0) + wts[j]
                return d
            a, b = joint(wpost), joint(wpre)
            out["B_total"][i, g] = sum(pv * np.log(pv / b[key]) for key, pv in a.items())
    return out


def _self_test():
    from rhm.conditional_revision.rule_family.family import make_family, generate_windows

    print("=" * 74)
    print("mixture-oracle self-test: BP mixture vs brute-force enumeration over (r, z)")
    print("=" * 74)

    for (v, s, L, m, R, K, dl) in [(2, 2, 2, 2, 2, 2, [0]),
                                   (3, 2, 2, 2, 3, 2, [0]),
                                   (2, 2, 3, 2, 2, 1, [1])]:
        family, _ = make_family(v, s, L, m, R=R, differ_levels=dl,
                                n_differ_features=min(v, 2), seed=11, mode="resample")
        T = s ** L
        seqs, rule_ids, lf = generate_windows(family, 6, K, seed=12)
        rule_ids = np.zeros_like(rule_ids)          # probe rule set 0's windows
        seqs, _, lf = generate_windows([family[0]], 6, K, seed=12)

        post = family_predictive(family, seqs.reshape(-1, T))
        post = post.reshape(len(family), 6, K, T, v)
        prof = mixture_profiles(post, seqs, rule_ids)
        parse = parse_revision_mixture(family, seqs, rule_ids, lf, prof,
                                       Ds=list(range(L)), chunk=64)

        errs = {}
        for D in range(L):
            bf = _bf_window(family, K, D, seqs)
            bt = parse["B_total"][D].reshape(6, K * T)
            errs[f"B_total D{D}"] = np.abs(bt - bf["B_total"]).max()
            if D == 0:
                for key in ("w_true", "rule_rev", "nll_mix", "nll_true", "H_rule"):
                    errs[key] = np.abs(prof[key] - bf[key]).max()
        worst = max(errs.values())
        print(f"  v{v}/s{s}/L{L}/m{m}  R={R} K={K} differ={dl}: max err over "
              f"{len(errs)} quantities = {worst:.2e}")
        for key, e in sorted(errs.items(), key=lambda kv: -kv[1])[:3]:
            print(f"      {key:<14} {e:.2e}")
        assert worst < 1e-8, f"mixture oracle disagrees with brute force ({worst})"

    _check_matches_incumbent()
    _check_floor_reduces()
    _check_identity()
    print("mixture-oracle self-test PASSED")


def _check_matches_incumbent():
    """`sequence_revision_all` columns 1..T-1 must reproduce `../oracle.py`'s
    `revision_and_entropy` exactly, or this module is measuring a different object."""
    from rhm.rhm_data import generate_rules_distinct
    from rhm.rhm_latent_loop import _generate_with_traces
    v, s, L, m = 6, 2, 4, 3
    rules = generate_rules_distinct(v, s, L, m, seed=1)
    seqs, lf, _ = _generate_with_traces(rules, 300, 11)
    Ds = list(range(L))
    ref = revision_and_entropy(rules, seqs, lf, Ds=Ds, chunk=150, verbose=False)
    got = sequence_revision_all(rules, seqs, lf, Ds=Ds, chunk=150)
    eb = max(np.abs(got["B_joint"][D][:, 1:] - ref["B_joint"][D]).max() for D in Ds)
    eh = max(np.abs(got["H_irr"][D][:, 1:] - ref["H_irr"][D]).max() for D in Ds)
    print(f"  agreement with ../oracle.py revision_and_entropy: "
          f"max |dB| = {eb:.2e}   max |dH_irr| = {eh:.2e}")
    assert eb < 1e-12 and eh < 1e-12, "extended revision disagrees with the incumbent"


def _check_floor_reduces():
    """differ_levels=[] must reduce EXACTLY to the single-rule oracle: the rule
    posterior never moves, rule revision is identically zero, and B_total == B."""
    from rhm.conditional_revision.rule_family.family import make_family, generate_windows
    v, s, L, m, R, K = 4, 2, 3, 2, 4, 2
    family, _ = make_family(v, s, L, m, R=R, differ_levels=[], seed=3)
    seqs, rule_ids, lf = generate_windows(family, 40, K, seed=4)
    T = s ** L
    post = family_predictive(family, seqs.reshape(-1, T)).reshape(R, 40, K, T, v)
    prof = mixture_profiles(post, seqs, rule_ids)
    parse = parse_revision_mixture(family, seqs, rule_ids, lf, prof, Ds=[0, 1, 2])
    assert np.abs(prof["rule_rev"]).max() < 1e-12, "floor must have zero rule revision"
    assert np.abs(prof["icl_gap"]).max() < 1e-12, "floor must have zero ICL gap"
    assert np.abs(prof["w_true"] - 1.0 / R).max() < 1e-12, "floor posterior must not move"
    for D in (0, 1, 2):
        e = np.abs(parse["B_total"][D] - parse["B_true"][D]).max()
        assert e < 1e-12, f"floor B_total must equal the single-rule B (D{D}: {e})"
    print("  floor (differ_levels=[]) reduces exactly to the single-rule oracle  ok")


def _check_identity():
    """The two Monte-Carlo identities on a small but non-trivial regime."""
    from rhm.conditional_revision.rule_family.family import make_family, generate_windows
    v, s, L, m, R, K = 6, 2, 4, 3, 6, 4
    family, _ = make_family(v, s, L, m, R=R, differ_levels=[2],
                            n_differ_features=4, seed=5)
    seqs, rule_ids, lf = generate_windows(family, 600, K, seed=6)
    T = s ** L
    post = family_predictive(family, seqs.reshape(-1, T)).reshape(R, 600, K, T, v)
    prof = mixture_profiles(post, seqs, rule_ids)
    parse = parse_revision_mixture(family, seqs, rule_ids, lf, prof, Ds=list(range(L)))
    chk = self_check_mixture(prof, parse)
    print(f"  identity check on v{v}/s{s}/L{L}/m{m} R={R} K={K}, n=600 "
          f"(MC error ~ 1/sqrt(n)):")
    r = chk["per_term"]["rule_only"]
    print(f"    rule-only : E[rule_rev] {r['lhs_mean_rule_rev']:.6f}   "
          f"E[H_mix-H_true] {r['rhs_mean_H_tot_mix_minus_H_tot_true']:.6f}   "
          f"abs_err {r['abs_err']:.2e}  {'ok' if r['passed'] else 'FAIL'}")
    for k, row in chk["per_term"].items():
        if k == "rule_only":
            continue
        print(f"    {k:<10}: E[B_total] {row['mean_B_total']:.5f}   "
              f"E[H_mix-H_irr] {row['mean_H_tot_mix_minus_H_irr']:.5f}   "
              f"abs_err {row['abs_err']:.2e}  {'ok' if row['passed'] else 'FAIL'}")
    assert chk["passed"], "the mixture identities do not hold"


if __name__ == "__main__":
    _self_test()
