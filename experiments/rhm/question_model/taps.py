"""Half 2 machinery: the slow-component conditioning tap, the endogenous estimator, and the
exact aleatoric label.

THREE OBJECTS, EACH FORCED BY A HALF-1 MEASUREMENT.

1. `slow_tap_vectors` -- the CONDITIONING TAP carries `root` and `hi` only.
   Half 1 measured `lo` at 0.868 against an exact ceiling of 0.943 (92%): the window already
   extracts it, so supplying it would blur attribution. `root` (0.239) and `hi` (0.281) sit at
   the shuffled-label null of ~0.25 while their exact ceilings are 0.343 and 0.584 -- the tap's
   niche is exactly what the window launders. The vector supplied is the HISTORY filter's
   BLOCK-ENTRY prior, P(theta_root, theta_hi | all previous blocks), which is
     (a) constant within a block, so it is a slow signal rather than a per-token hint;
     (b) structurally unavailable to a 64-token-window learner, which is the point;
     (c) exactly the object a cross-context estimator would carry, so the oracle rung is a
         genuine ceiling for the learned rung rather than a different kind of signal.

2. `SlowEstimator` -- the ENDOGENOUS rung. A learned leaky-integrator over the previous
   `mem_blocks` blocks' token histograms, rolled out differentiably inside the training step.
   No theta label anywhere; the LM loss is its only teacher. Its memory (32 blocks = 2048
   tokens) is deliberately longer than the model's context, since the whole claim is that this
   is the object a windowed learner structurally cannot build in-context.

3. `aleatoric_label` -- H(x_p | z_{<=L-1}, theta, x_{<p}) in CLOSED FORM, O(1) per token.
   Half 1 measured H_irr falling ~20% when theta is known (d1: 0.684 -> 0.553), so a
   drift-blind aleatoric label would down-weight exactly the positions carrying question-news.
   Both labels are produced here (`rule_w` = the true mixture, or uniform for the blind arm).
   Given the whole level-(L-1) latent layer the leaf pairs are conditionally independent across
   parents, so only the arriving token's own parent and its already-observed sibling matter --
   which makes the BP oracle unnecessary at this truncation and the label affordable over a
   200k-block pool. Gated against `oracle_q.structure_channel`'s H_irr at D = L-1.
"""

import numpy as np

EPS = 1e-300


# --------------------------------------------------------------------------- #
# whole-block likelihoods and the slow (cross-context) posterior
# --------------------------------------------------------------------------- #

def block_loglik(rules, seqs, rule_w_states, root_p_states, chunk=256, verbose=False):
    """log P(x_block | theta_g) for every (block, grid state). (n, G) float64.

    `qfilter.prefix_loglik` returns the same total in its last cumulative sum, at T times the
    cost, because it also produces every intermediate prefix. The tap needs only the total, so
    this walks the binary-counter stack once and stops at the root.
    """
    v, m, s = rules[0].shape
    assert s == 2
    L = len(rules)
    n, T = seqs.shape
    G = root_p_states.shape[0]
    out = np.zeros((n, G))
    for c0 in range(0, n, chunk):
        c1 = min(n, c0 + chunk)
        nb = c1 - c0
        stack = []
        for p in range(T):
            oh = np.zeros((nb, G, v))
            oh[np.arange(nb)[:, None], np.arange(G)[None, :], seqs[c0:c1, p][:, None]] = 1.0
            stack.append((L, oh, np.zeros((nb, G))))
            while len(stack) >= 2 and stack[-1][0] == stack[-2][0]:
                d = stack[-1][0]
                (_, lv, lz), (_, rv, rz) = stack[-2], stack[-1]
                rd, wd = rules[d - 1], rule_w_states[d - 1]
                raw = np.zeros((nb, G, v))
                for a in range(v):
                    acc = np.zeros((nb, G))
                    for r in range(m):
                        t0, t1 = rd[a, r]
                        acc += wd[None, :, a, r] * lv[:, :, t0] * rv[:, :, t1]
                    raw[:, :, a] = acc
                z = raw.sum(-1)
                stack = stack[:-2] + [(d - 1, raw / np.clip(z[..., None], EPS, None),
                                       lz + rz + np.log(np.clip(z, EPS, None)))]
        assert len(stack) == 1 and stack[0][0] == 0
        _, vec, lz = stack[0]
        out[c0:c1] = np.log(np.clip((root_p_states[None] * vec).sum(-1), EPS, None)) + lz
        stack = []
        if verbose and (c1 // chunk) % 50 == 0:
            print(f"  block_loglik: {c1}/{n}", flush=True)
    return out


def slow_block_posteriors(blk_ll, prior0, P_trans, comp_onehots, entry=True):
    """Per-block marginal posterior over each component, for an unbounded-memory learner.

    `entry=True` returns the BLOCK-ENTRY prior P(theta | blocks strictly before this one) --
    the tap's payload. Returns a list of (n, K) arrays, one per component.
    """
    n, G = blk_ll.shape
    cur = np.asarray(prior0, dtype=np.float64).copy()
    out = np.zeros((n, G))
    for i in range(n):
        if entry:
            out[i] = cur
        lz = np.log(np.clip(cur, EPS, None)) + blk_ll[i]
        lz -= lz.max()
        post = np.exp(lz)
        post /= post.sum()
        if not entry:
            out[i] = post
        cur = post @ P_trans
    return [out @ oh for oh in comp_onehots]


def slow_tap_vectors(rules, seqs, spec, dirs, comps=("root", "hi"), chunk=256,
                     verbose=False, entry=True):
    """(n_blocks, K * len(comps)) float32 tap payload, and the component index list."""
    from rhm.question_model import demand_state as DS
    L = len(rules)
    W = DS.all_weights(spec, dirs)
    G = len(W)
    rw = [np.stack([W[g][1][d] for g in range(G)]) for d in range(L)]
    rp = np.stack([W[g][0] for g in range(G)])
    idx, pj = DS.state_grid(spec)
    P = DS.joint_transition(spec)
    names = [nm for nm, _ in spec["components"]]
    cidx = [names.index(c) for c in comps]
    ohs = [np.eye(spec["K"])[idx[:, c]] for c in cidx]
    bll = block_loglik(rules, seqs, rw, rp, chunk=chunk, verbose=verbose)
    margs = slow_block_posteriors(bll, pj, P, ohs, entry=entry)
    return np.concatenate(margs, axis=1).astype(np.float32), cidx


# --------------------------------------------------------------------------- #
# the exact aleatoric label, in closed form
# --------------------------------------------------------------------------- #

def aleatoric_label_tokens(rules, seqs, parent_feats, rule_w=None):
    """Same object indexed by TOKEN position p = 0..T-1 rather than by signal index, which is
    what a per-position training weight needs (a window's first target can be a block's first
    token). (n, T) float64."""
    return _alea(rules, seqs, parent_feats, rule_w, range(seqs.shape[1]))


def aleatoric_label(rules, seqs, parent_feats, rule_w=None):
    """H(x_p | z_{<=L-1}, theta, x_{<p}) for p = 1..T-1, exactly. (n, T-1) float64.

    parent_feats: (n, T/s) the TRUE level-(L-1) features (i.e. `level_features[L-1]`).
    rule_w: (n, v, m) the true bottom-layer mixture per block, or None for uniform (the
            DRIFT-BLIND label, which is what a learner without a question model would use).
    Signal index t = 0..T-2 means "the token x_{t+1} arrives", matching every other readout.
    """
    return _alea(rules, seqs, parent_feats, rule_w, range(1, seqs.shape[1]))


def _alea(rules, seqs, parent_feats, rule_w, ps):
    v, m, s = rules[0].shape
    assert s == 2
    L = len(rules)
    n, T = seqs.shape
    rd = rules[L - 1]
    w = (np.full((n, v, m), 1.0 / m) if rule_w is None else np.asarray(rule_w))
    ps = list(ps)
    out = np.zeros((n, len(ps)))
    ar = np.arange(n)
    for t, p in enumerate(ps):
        j, k = p // s, p % s
        a = parent_feats[:, j]                       # (n,) true parent feature
        wa = w[ar, a]                                # (n, m)
        cons = np.ones((n, m), bool)
        if k == 1:
            cons = rd[a, :, 0] == seqs[:, p - 1][:, None]
        ww = np.where(cons, wa, 0.0)
        dist = np.zeros((n, v))
        tok = rd[a, :, k]                            # (n, m) the token each rule emits here
        np.add.at(dist, (np.repeat(ar, m), tok.ravel()), ww.ravel())
        z = dist.sum(1, keepdims=True)
        dist = dist / np.clip(z, EPS, None)
        out[:, t] = -(np.where(dist > 0, dist * np.log(np.clip(dist, EPS, None)), 0.0)).sum(1)
    return out


# --------------------------------------------------------------------------- #
# self-tests
# --------------------------------------------------------------------------- #

def _test_block_loglik():
    from rhm.rhm_data import generate_rules_distinct
    from rhm.question_model import demand_state as DS
    from rhm.question_model import qfilter as QF
    for (v, s, L, m, K) in [(4, 2, 3, 2, 3), (6, 2, 4, 3, 3)]:
        rules = generate_rules_distinct(v, s, L, m, seed=1)
        spec = DS.make_spec(K=K, sigma_s=1.2, kappa=0.2, L=L)
        dirs = DS.make_directions(rules, spec)
        seqs, _lf, _lr, _ti = DS.sample_corpus(rules, spec, dirs, 40, 5)
        W = DS.all_weights(spec, dirs)
        G = len(W)
        rw = [np.stack([W[g][1][d] for g in range(G)]) for d in range(L)]
        rp = np.stack([W[g][0] for g in range(G)])
        ref = QF.prefix_loglik(rules, seqs, rw, rp, chunk=20).sum(-1)
        got = block_loglik(rules, seqs, rw, rp, chunk=20)
        err = float(np.abs(ref - got).max())
        print(f"  block_loglik v{v}/L{L}/m{m}/K{K}: max|delta vs prefix_loglik.sum| = {err:.2e}")
        assert err < 1e-9


def _test_aleatoric_label():
    from rhm.rhm_data import generate_rules_distinct
    from rhm.question_model import demand_state as DS
    from rhm.question_model import oracle_q as OQ
    v, s, L, m = 6, 2, 4, 3
    rules = generate_rules_distinct(v, s, L, m, seed=1)
    spec = DS.make_spec(K=5, sigma_s=1.0, kappa=0.2, L=L)
    dirs = DS.make_directions(rules, spec)
    n = 300
    seqs, lf, _lr, ti = DS.sample_corpus(rules, spec, dirs, n, 3)
    W = DS.all_weights(spec, dirs)
    rw = [np.stack([W[i][1][d] for i in ti]) for d in range(L)]
    rp = np.stack([W[i][0] for i in ti])
    sc = OQ.structure_channel(rules, seqs, lf, rw, rp, Ds=[L - 1], chunk=150, verbose=False)
    got = aleatoric_label(rules, seqs, lf[L - 1], rw[L - 1])
    err = float(np.abs(sc["H_irr"][L - 1] - got).max())
    print(f"  aleatoric_label (theta-aware) vs BP H_irr[D=L-1]: max|delta| = {err:.2e}")
    assert err < 1e-10
    blind = aleatoric_label(rules, seqs, lf[L - 1], None)
    print(f"  theta-aware mean {got.mean():.4f}   drift-blind mean {blind.mean():.4f}   "
          f"corr {np.corrcoef(got.ravel(), blind.ravel())[0, 1]:+.3f}")


if __name__ == "__main__":
    print("=" * 74)
    print("taps self-test")
    print("=" * 74)
    _test_block_loglik()
    _test_aleatoric_label()
    print("taps self-test PASSED")
