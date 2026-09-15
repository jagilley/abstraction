"""Exact Bayes next-token predictive on FLAT windows, for a family of coarse observers.

Why a new oracle
----------------
`conditional_revision/oracle.py` computes P(x_{t+1} | x_{<=t}) for ALIGNED sequences:
it knows leaf 0 is a sequence start. The base models in this repo are trained on flat
windows cut at uniformly random offsets from a stream of concatenated i.i.d.
sequences (`get_ntp_batch`), and position embeddings index the window, not the
sequence. So the distribution the model is actually fit to is the phase-marginal
predictive

    p(w_{j} | w_{<j})  =  sum_psi  pi_{j-1}(psi) * p_psi(w_j | w_{<j})
    pi_{j-1}(psi)      ∝  prod_{i<j} p_psi(w_i | w_{<i})          (uniform prior on psi)

where psi is the (unknown) sequence phase of the window's first token. Reading the
model's logits against the aligned oracle would bill phase uncertainty to the model as
miscalibration; this oracle does not.

The coarse-observer family
--------------------------
Observer k knows the BOTTOM k levels of the grammar exactly and nothing above: it
models the stream as an i.i.d. concatenation of depth-k subtrees (span S = s^k), each
rooted in a level-(L-k) feature drawn from rho_k, the position-averaged marginal of
that level under the true DGP. Observer L is the true DGP (rho_L = uniform, S = T), and
observer 0 is the unigram marginal. Every observer is phase-marginal with psi uniform on
[0, S), which is exactly the true phase distribution mod S.

"Is the model's q the posterior of some coarser observer?" is then a fit over k, and
k is the thing the Cagnetta & Wyart staircase says should climb with training.

Under observer k and phase psi, window token j sits at subtree leaf i = (psi + j) mod S
of a constituent that starts at window index j - i. Its predictive conditions only on
that constituent's already-seen tokens (constituents are independent), which occupy
leaves [a, i) with a = max(0, i - j). Every (psi, j) is one BP problem on a depth-k
subtree; they are batched on GPU.

BP is sum-product on the hypertree (cliques = parent + its s-tuple), same factorisation
as `conditional_revision/oracle.py` (whose brute-force test caught the pairwise-tree
error); `_self_test` checks this module against it and against full enumeration.

Local correctness self-test (CPU, ~1 min):
    cd experiments && python -m rhm.logit_reading.flat_oracle
"""

import itertools

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Batched sum-product BP on a depth-k subtree (torch)
# ---------------------------------------------------------------------------

def _rule_index(sub_rules, device):
    """Per level, per child slot: (v*m,) long tensor of child values, flat index a*m + r."""
    out = []
    for rd in sub_rules:
        v, m, s = rd.shape
        out.append([torch.as_tensor(rd[:, :, i].reshape(-1), dtype=torch.long, device=device)
                    for i in range(s)])
    return out


def _norm(x):
    return x / x.sum(-1, keepdim=True).clamp_min(1e-300)


def leaf_posteriors(ev, sub_rules, root_prior, ridx=None, sub_w=None):
    """Exact P(leaf | evidence) for every leaf of a depth-k subtree.

    ev:         (N, s^k, v) leaf evidence (one-hot = observed, ones = free).
    sub_rules:  list of k (v, m, s) numpy arrays, top -> bottom.
    root_prior: (v,) tensor.
    sub_w:      optional list of k (v, m) synonym-weight tensors (rows sum to 1);
                None = the canonical uniform 1/m RHM.
    Returns (N, s^k, v).
    """
    k = len(sub_rules)
    N, S, v = ev.shape
    if k == 0:
        return _norm(ev * root_prior.view(1, 1, v))
    m, s = sub_rules[0].shape[1], sub_rules[0].shape[2]
    if ridx is None:
        ridx = _rule_index(sub_rules, ev.device)
    up = [None] * (k + 1)
    up[k] = _norm(ev)
    gath = [None] * k
    for d in range(k - 1, -1, -1):
        n_par = s ** d
        child = up[d + 1].view(N, n_par, s, v)
        g = [child[:, :, i, :][..., ridx[d][i]] for i in range(s)]  # (N, n_par, v*m)
        gath[d] = g
        prod = g[0]
        for i in range(1, s):
            prod = prod * g[i]
        prod = prod.view(N, n_par, v, m)
        if sub_w is not None:
            prod = prod * sub_w[d].view(1, 1, v, m)
        up[d] = _norm(prod.sum(-1))
    down = root_prior.view(1, 1, v).expand(N, 1, v)
    for d in range(k):
        n_par = s ** d
        g = gath[d]
        par = down.repeat_interleave(m, dim=-1)                      # (N, n_par, v*m)
        if sub_w is not None:
            par = par * sub_w[d].reshape(1, 1, v * m)
        outs = []
        for i in range(s):
            contrib = par
            for j in range(s):
                if j != i:
                    contrib = contrib * g[j]
            o = torch.zeros(N, n_par, v, dtype=ev.dtype, device=ev.device)
            o.scatter_add_(-1, ridx[d][i].view(1, 1, -1).expand(N, n_par, v * m), contrib)
            outs.append(o)
        down = _norm(torch.stack(outs, dim=2).reshape(N, n_par * s, v))
    return _norm(up[k] * down)


def level_marginals(rules, rule_w=None):
    """Position-averaged prior marginal of the features at each tree level 0..L (numpy).
    Level 0 = root (uniform), level L = leaves. rule_w: optional list of (v, m)."""
    v = rules[0].shape[0]
    L = len(rules)
    out = [np.full(v, 1.0 / v)]
    cur = np.full((1, v), 1.0 / v)                    # (n_nodes, v)
    for d in range(L):
        vv, m, s = rules[d].shape
        nxt = np.zeros((cur.shape[0] * s, v))
        for a in range(v):
            for r in range(m):
                for i in range(s):
                    w = 1.0 / m if rule_w is None else rule_w[d][a, r]
                    nxt[i::s, rules[d][a, r, i]] += cur[:, a] * w
        cur = nxt
        out.append(cur.mean(0))
    return out


# ---------------------------------------------------------------------------
# Phase-marginal flat-window predictive for observer k
# ---------------------------------------------------------------------------

def observer_problems(T1, S):
    """Index plan for all (psi, j) problems. Returns arrays over P = S*T1 problems:
    leaf i, first-evidence leaf a, and the window offset `off` such that leaf l in
    [a, i) holds window token (l + off)."""
    psi = np.repeat(np.arange(S), T1)
    j = np.tile(np.arange(T1), S)
    i = (psi + j) % S
    a = np.maximum(0, i - j)
    off = j - i
    return psi, j, i, a, off


def flat_predictive(windows, rules, k, rho=None, device="cpu", dtype=torch.float64,
                    chunk=8, return_phase=False, rule_w=None, noise_eps=0.0):
    """Phase-marginal predictive of observer k at every window position.

    windows: (N, T1) int numpy -- T1 window tokens w_0..w_{T1-1}.
    noise_eps: if > 0, the observer believes every OBSERVED token was independently
      replaced by a uniform draw with probability eps (soft leaf evidence), and the
      returned predictive is over the observed token, (1-eps) p_clean + eps/v. This keeps
      the posterior defined after a grammar violation.
    Returns
      pred:   (N, T1, v) p_k(w_j = . | w_{<j}), j = 0..T1-1 (j = 0 is the prior).
      logpi:  (N, T1, S) log posterior over phase AFTER observing w_{<=j}
              (only if return_phase).
      p_psi:  (N, S, T1, v) per-phase predictives (only if return_phase).
    """
    L = len(rules)
    v, m, s = rules[0].shape
    S = s ** k
    N, T1 = windows.shape
    sub_rules = rules[L - k:]
    if rho is None:
        rho = level_marginals(rules, rule_w)[L - k]
    rho_t = torch.as_tensor(rho, dtype=dtype, device=device)
    ridx = _rule_index(sub_rules, device) if k > 0 else None
    sub_w = (None if rule_w is None else
             [torch.as_tensor(w, dtype=dtype, device=device) for w in rule_w[L - k:]])

    psi, j, i, a, off = observer_problems(T1, S)
    P = psi.shape[0]
    leaves = np.arange(S)
    # (P, S) mask of observed leaves and the window index each observed leaf reads
    obs = (leaves[None, :] >= a[:, None]) & (leaves[None, :] < i[:, None])
    widx = np.where(obs, leaves[None, :] + off[:, None], 0)
    obs_t = torch.as_tensor(obs, device=device)
    widx_t = torch.as_tensor(widx, dtype=torch.long, device=device)
    i_t = torch.as_tensor(i, dtype=torch.long, device=device)

    pred = torch.empty(N, T1, v, dtype=dtype)
    logpi_all = torch.empty(N, T1, S, dtype=dtype) if return_phase else None
    p_psi_all = torch.empty(N, S, T1, v, dtype=dtype) if return_phase else None
    W = torch.as_tensor(windows, dtype=torch.long, device=device)
    for c0 in range(0, N, chunk):
        Wc = W[c0:c0 + chunk]
        n = Wc.shape[0]
        tok = Wc[:, widx_t]                                           # (n, P, S)
        ev = torch.nn.functional.one_hot(tok, v).to(dtype)
        if noise_eps > 0:
            ev = (1.0 - noise_eps) * ev + noise_eps / v
        ev = torch.where(obs_t[None, :, :, None], ev, torch.ones_like(ev))
        post = leaf_posteriors(ev.view(n * P, S, v), sub_rules, rho_t, ridx, sub_w)
        post = post.view(n, P, S, v)
        pj = post[:, torch.arange(P, device=device), i_t, :]         # (n, P, v)
        pj = pj.view(n, S, T1, v)                                     # [psi, j]
        if noise_eps > 0:
            pj = (1.0 - noise_eps) * pj + noise_eps / v
        # chain-rule log-likelihood of each phase
        lp_tok = torch.log(pj.gather(-1, Wc[:, None, :, None].expand(n, S, T1, 1))[..., 0])
        cum = torch.cumsum(lp_tok, dim=-1)                            # (n, S, T1): after w_<=j
        prior = torch.full((n, S, 1), -np.log(S), dtype=dtype, device=device)
        logpost_before = torch.cat([prior.expand(n, S, 1),
                                    cum[:, :, :-1] - np.log(S)], dim=-1)  # before w_j
        logpost_before = logpost_before - torch.logsumexp(logpost_before, dim=1, keepdim=True)
        mix = (torch.exp(logpost_before)[..., None] * pj).sum(1)       # (n, T1, v)
        pred[c0:c0 + n] = _norm(mix).cpu()
        if return_phase:
            lp_after = cum - torch.logsumexp(cum, dim=1, keepdim=True)
            logpi_all[c0:c0 + n] = lp_after.permute(0, 2, 1).cpu()
            p_psi_all[c0:c0 + n] = pj.cpu()
    if return_phase:
        return pred, logpi_all, p_psi_all
    return pred


def leaf_levels(leaf_idx, s, L):
    """s-adic valuation of a leaf index (0 -> L): the highest constituent the token opens.
    Matches `rhm_bayes_entropy._position_levels` for leaf_idx >= 1."""
    leaf_idx = np.asarray(leaf_idx)
    lev = np.zeros_like(leaf_idx)
    p = leaf_idx.copy()
    zero = p == 0
    p[zero] = 1
    while True:
        mask = (p % s == 0)
        if not mask.any():
            break
        lev[mask] += 1
        p[mask] //= s
    lev[zero] = L
    return lev


def sample_flat_windows(rules, n, T1, seed, rule_w=None):
    """n windows of T1 tokens at uniformly random phase from a stream of i.i.d. RHM
    sequences. Returns (windows (n, T1), phase (n,)) -- phase = sequence leaf index of
    window token 0, the same thing `get_ntp_batch` produces with a random corpus offset.
    rule_w=None keeps the original uniform generator (bit-identical to earlier runs)."""
    L = len(rules)
    s = rules[0].shape[2]
    T = s ** L
    n_seq = -(-(T1 + T) // T)
    if rule_w is None:
        from rhm.rhm_latent_loop import _generate_with_traces
        seqs, _, _ = _generate_with_traces(rules, n * n_seq, seed)
    else:
        from rhm.logit_reading.grammar import generate
        seqs = generate(rules, rule_w, n * n_seq, np.random.default_rng(seed))
    stream = seqs.reshape(n, n_seq * T)
    rng = np.random.default_rng(seed + 1)
    phase = rng.integers(0, T, size=n)
    idx = phase[:, None] + np.arange(T1)[None, :]
    return np.take_along_axis(stream, idx, axis=1), phase


# ---------------------------------------------------------------------------
# Self-test: brute force on tiny grammars + agreement with the aligned oracle
# ---------------------------------------------------------------------------

def _constituent_dist(sub_rules, rho, v, sub_w=None):
    """dict leaf-tuple -> probability for a depth-k subtree with root prior rho."""
    k = len(sub_rules)
    dist = {}
    for a in range(v):
        if rho[a] == 0:
            continue
        frontier = [((a,), rho[a])]
        for d in range(k):
            rd = sub_rules[d]
            m, s = rd.shape[1], rd.shape[2]
            new = []
            for feats, pr in frontier:
                for choice in itertools.product(range(m), repeat=len(feats)):
                    kids, w = [], 1.0
                    for f, r in zip(feats, choice):
                        kids.extend(rd[f, r].tolist())
                        w *= (1.0 / m) if sub_w is None else sub_w[d][f, r]
                    new.append((tuple(kids), pr * w))
            frontier = new
        for leaves, pr in frontier:
            dist[leaves] = dist.get(leaves, 0.0) + pr
    return dist


def _brute_force_windows(sub_rules, rho, v, S, T1, sub_w=None):
    """dict window-tuple -> probability under a uniform-phase i.i.d.-constituent stream."""
    cd = list(_constituent_dist(sub_rules, rho, v, sub_w).items())
    n_c = -(-(S - 1 + T1) // S)
    joint = {}
    for combo in itertools.product(cd, repeat=n_c):
        stream = sum((c[0] for c in combo), ())
        pr = float(np.prod([c[1] for c in combo]))
        for psi in range(S):
            w = stream[psi:psi + T1]
            joint[w] = joint.get(w, 0.0) + pr / S
    return joint


def _self_test():
    from rhm.rhm_data import generate_rules_distinct
    from rhm.conditional_revision import oracle as ORC

    # 1. brute force, every observer, tiny grammar, uniform and weighted synonyms
    v, s, L, m = 3, 2, 2, 2
    rules = generate_rules_distinct(v, s, L, m, seed=3)
    T1 = 5
    for rule_w in (None, [np.random.default_rng(9).dirichlet(np.ones(m), size=v)
                          for _ in range(L)]):
        margs = level_marginals(rules, rule_w)
        worst = 0.0
        for k in range(L + 1):
            S = s ** k
            joint = _brute_force_windows(rules[L - k:], margs[L - k], v, S, T1,
                                         None if rule_w is None else rule_w[L - k:])
            wins = np.array(list(joint.keys()), dtype=np.int64)
            pred = flat_predictive(wins, rules, k, rule_w=rule_w).numpy()
            for n_w, w in enumerate(joint.keys()):
                for j in range(T1):
                    num = np.zeros(v)
                    for w2, pr in joint.items():
                        if w2[:j] == w[:j]:
                            num[w2[j]] += pr
                    bf = num / num.sum()
                    worst = max(worst, float(np.abs(bf - pred[n_w, j]).max()))
            print(f"  brute force  {'uniform ' if rule_w is None else 'weighted'}  observer k={k}"
                  f"  S={S}  windows={len(joint)}  max|err| so far {worst:.2e}")
        assert worst < 1e-10, worst

    # 1b. noisy-observation observer, brute force (uniform synonyms, every k)
    eps = 0.07
    margs = level_marginals(rules)
    worst = 0.0
    for k in range(L + 1):
        S = s ** k
        clean = _brute_force_windows(rules[L - k:], margs[L - k], v, S, T1)
        noisy = {}
        for w in itertools.product(range(v), repeat=T1):
            noisy[w] = sum(pr * np.prod([(1 - eps) * (a == b) + eps / v for a, b in zip(w, wc)])
                           for wc, pr in clean.items())
        keys = list(noisy.keys())[::7]
        pred = flat_predictive(np.array(keys, dtype=np.int64), rules, k, noise_eps=eps).numpy()
        for n_w, w in enumerate(keys):
            for j in range(T1):
                num = np.zeros(v)
                for w2, pr in noisy.items():
                    if w2[:j] == w[:j]:
                        num[w2[j]] += pr
                worst = max(worst, float(np.abs(num / num.sum() - pred[n_w, j]).max()))
    print(f"  brute force  noisy eps={eps}  all observers  max|err| {worst:.2e}")
    assert worst < 1e-10, worst

    # 2. the phase-known predictive at phase 0 equals the aligned oracle, real regime
    v, s, L, m = 16, 2, 6, 4
    rules = generate_rules_distinct(v, s, L, m, seed=0)
    from rhm.rhm_latent_loop import _generate_with_traces
    seqs, _, _ = _generate_with_traces(rules, 16, 11)
    T = s ** L
    ev_rows = []
    for plen in range(1, T):
        ev_rows.append(ORC._leaf_evidence(seqs, plen, v))
    ev = torch.as_tensor(np.concatenate(ev_rows, 0))
    post = leaf_posteriors(ev, rules, torch.full((v,), 1.0 / v, dtype=torch.float64))
    post = post.view(T - 1, seqs.shape[0], T, v)
    worst2 = 0.0
    for plen in range(1, T):
        _, _, lp = ORC.prefix_beliefs(rules, seqs, plen, 0)
        worst2 = max(worst2, float(np.abs(post[plen - 1, :, plen, :].numpy() - lp).max()))
    print(f"  aligned-oracle agreement (v16 L6 m4, 63 prefixes x 16 seqs): max|err| {worst2:.2e}")
    assert worst2 < 1e-10, worst2

    # 3. observer L at phase 0 on a flat window reduces to the aligned oracle for j < T
    wins, phase = sample_flat_windows(rules, 6, T + 1, seed=5)
    pred, logpi, p_psi = flat_predictive(wins, rules, L, return_phase=True)
    ph = torch.as_tensor(phase)
    known = p_psi[torch.arange(len(phase)), ph]                        # (n, T1, v)
    assert torch.allclose(known.sum(-1), torch.ones(()).double())
    # the true phase must never be excluded
    lp_true = logpi[torch.arange(len(phase)), :, ph]
    assert torch.isfinite(lp_true).all()
    print(f"  true-phase log posterior at window end: {lp_true[:, -1].numpy().round(3)}")
    print("flat_oracle self-test passed")


if __name__ == "__main__":
    _self_test()
