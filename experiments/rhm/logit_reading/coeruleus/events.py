"""Shared machinery for the coeruleus round: corrupted windows, events, one-knob consumers.

The venue is the eps-corrupted stream the `altitude/train_noisy.py` trajectory was trained on.
`altitude/dual_ladder.py` already cached, on the SAME 4096 held-out windows Part 1 uses
(eval_seed 4242), both observer families on both window sets:

    ladder_d{0.0,0.01}_o{0.0,0.01}.npz    (preds (n, T, L+1, v), windows, phase)

`d` is the corruption of the DATA, `o` the corruption the OBSERVER models. On the corrupted
window set (`d=0.01`) the eps family (`o=0.01`) is the Bayes-optimal reference: p_L^eps is the
exact predictive of the distribution the noise-trained model is fit to.

What this module adds:

* `corrupted_windows` reproduces `dual_ladder.make_windows` bit-for-bit AND returns the
  corruption mask, which the cached npz does not store. (Checked against the cache by
  `assert_cache_matches`.) 1/v of the flips redraw the same token; those are not events.
* `events_from_mask` -- isolated corrupted tokens with a clean +-H neighbourhood, so a
  per-offset profile is not contaminated by a second event.
* one-knob consumers on the frozen logits: a TEMPERATURE `alpha` (q_alpha = softmax(alpha z))
  and a NOISE FLOOR `w` (q_w = (1-w) q + w/v). Both are convex in their knob, both are
  optimised exactly by bisection on the derivative, pooled or per-row. The temperature is the
  knob question 1 asks about; the floor is the shape an eps-aware Bayesian's response actually
  has, and costs nothing extra to bound.
"""

import numpy as np

from rhm.shared import DATA_DIR

KEY = "v16_s2_L6_m4_distinct"
LR_DIR = f"{DATA_DIR}/{KEY}/logit_reading"
EPSF = 1e-300


# ---------------------------------------------------------------------------
# grammar / windows
# ---------------------------------------------------------------------------

def build_rules(cfg=None, v=16, s=2, L=6, m=4, rule_seed=0, alpha=1.0, weight_seed=1):
    from rhm.rhm_data import generate_rules_distinct
    from rhm.logit_reading.grammar import synonym_weights
    if cfg is not None:
        v, s, L, m = cfg["v"], cfg["s"], cfg["L"], cfg["m"]
        rule_seed = cfg.get("rule_seed", rule_seed)
        alpha = cfg.get("alpha", alpha)
        weight_seed = cfg.get("weight_seed", weight_seed)
    rules = generate_rules_distinct(v, s, L, m, seed=rule_seed)
    rule_w = None if alpha <= 0 else synonym_weights(v, L, m, alpha, weight_seed)
    return rules, rule_w


def corrupted_windows(rules, rule_w, n, T1, seed, eps_data, noise_seed=777):
    """`dual_ladder.make_windows` plus the mask. Returns (w_noisy, w_clean, corrupt, phase).

    `corrupt` is the EFFECTIVE corruption: flipped AND the redraw changed the token (a flip
    that redraws the same symbol leaves nothing for anything to respond to)."""
    from rhm.logit_reading.flat_oracle import sample_flat_windows
    w_clean, phase = sample_flat_windows(rules, n, T1, seed, rule_w=rule_w)
    if eps_data <= 0:
        return w_clean, w_clean, np.zeros(w_clean.shape, bool), phase
    rng = np.random.default_rng(noise_seed)
    v = rules[0].shape[0]
    flip = rng.random(w_clean.shape) < eps_data
    w = np.where(flip, rng.integers(0, v, w_clean.shape), w_clean)
    return w, w_clean, flip & (w != w_clean), phase


def load_ladder(eps_data, eps_obs):
    z = np.load(f"{LR_DIR}/ladder_d{eps_data}_o{eps_obs}.npz")
    return z["preds"], z["windows"], z["phase"]


def assert_cache_matches(rules, rule_w, eps_data, n=4096, eval_seed=4242):
    """The reconstructed windows must be the cached ones, or the mask is meaningless."""
    T1 = rules[0].shape[2] ** len(rules) + 1
    w, w_clean, corrupt, phase = corrupted_windows(rules, rule_w, n, T1, eval_seed, eps_data)
    _, wc, pc = load_ladder(eps_data, 0.0)
    assert (w == wc).all(), "reconstructed corrupted windows differ from the cached ladder"
    assert (phase == pc).all()
    return w, w_clean, corrupt, phase


# ---------------------------------------------------------------------------
# events
# ---------------------------------------------------------------------------

def events_from_mask(corrupt, H, c_lo=8, isolate=True):
    """Isolated corrupted tokens. Returns (win, c) with

      * c in [c_lo, T-1]   (T = corrupt.shape[1] - 1 predictions; the model reads token c at
        prediction index c, so c <= T-1 is needed for offset 0 to exist),
      * c + H <= T - 1     so the whole offset profile tau = 0..H exists,
      * (isolate) no other corrupted token in [c - H, c + H].
    """
    n, T1 = corrupt.shape
    T = T1 - 1
    win, c = np.nonzero(corrupt)
    ok = (c >= c_lo) & (c + H <= T - 1)
    win, c = win[ok], c[ok]
    if isolate and len(c):
        keep = np.ones(len(c), bool)
        for i in range(len(c)):
            lo, hi = max(0, c[i] - H), min(T1 - 1, c[i] + H)
            if corrupt[win[i], lo:hi + 1].sum() > 1:
                keep[i] = False
        win, c = win[keep], c[keep]
    return win, c


def quiet_positions(corrupt, H, t_lo=8, rng=None, per_window=1):
    """Prediction indices t with no corruption anywhere in [t-H, t+H+1] -- the background."""
    n, T1 = corrupt.shape
    T = T1 - 1
    rng = np.random.default_rng(0) if rng is None else rng
    cs = np.concatenate([np.zeros((n, 1), np.int64), np.cumsum(corrupt, 1)], 1)
    ts = np.arange(t_lo, T - H)
    lo = np.clip(ts - H, 0, T1)
    hi = np.clip(ts + H + 2, 0, T1)
    cnt = cs[:, hi] - cs[:, lo]                                   # (n, len(ts))
    wi, ti = np.nonzero(cnt == 0)
    if per_window <= 0:
        return wi, ts[ti]
    out_w, out_t = [], []
    order = rng.permutation(len(wi))
    seen = {}
    for j in order:
        w = int(wi[j])
        if seen.get(w, 0) >= per_window:
            continue
        seen[w] = seen.get(w, 0) + 1
        out_w.append(w)
        out_t.append(int(ts[ti[j]]))
    o = np.argsort(out_w, kind="stable")
    return np.array(out_w)[o], np.array(out_t)[o]


# ---------------------------------------------------------------------------
# distributions
# ---------------------------------------------------------------------------

def log_softmax(z):
    zm = z - z.max(-1, keepdims=True)
    return zm - np.log(np.exp(zm).sum(-1, keepdims=True))


def ent(p):
    return -(p * np.log(np.clip(p, EPSF, None))).sum(-1)


def xent(p, logq):
    return -(p * logq).sum(-1)


def kl(p, logq):
    return xent(p, logq) - ent(p)


def convex_obs(preds, kappa, L=6):
    """p_kappa = (1-lam) p_k + lam p_{k+1} -- `altitude.identity.convex_family`."""
    k = min(int(np.floor(kappa)), L - 1)
    lam = kappa - k
    return (1.0 - lam) * preds[..., k, :] + lam * preds[..., k + 1, :]


def fit_kappa_pooled(preds, logq, L=6, grid_step=0.05):
    grid = np.round(np.arange(0.0, L + 1e-9, grid_step), 4)
    curve = np.array([kl(convex_obs(preds, kap, L), logq).mean() for kap in grid])
    i = int(curve.argmin())
    return float(grid[i]), float(curve[i])


# ---------------------------------------------------------------------------
# one-knob consumers (both convex in the knob; bisect the derivative)
# ---------------------------------------------------------------------------

def _bisect(dfun, lo, hi, shape, iters=44):
    """Vector bisection of an increasing dfun on [lo, hi]; returns the root, clipped."""
    lo = np.full(shape, float(lo))
    hi = np.full(shape, float(hi))
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        neg = dfun(mid) < 0
        lo = np.where(neg, mid, lo)
        hi = np.where(neg, hi, mid)
    return 0.5 * (lo + hi)


def temp_ce(z, p, alpha):
    """CE(p, softmax(alpha z)) per row. alpha broadcasts against z[..., 0]."""
    return xent(p, log_softmax(np.asarray(alpha)[..., None] * z))


def temp_opt(z, p, pooled=False, lo=0.02, hi=30.0):
    """alpha minimising CE(p, softmax(alpha z)). d/dalpha CE = <q_alpha - p, z>, increasing."""
    z = np.asarray(z, np.float64)
    p = np.asarray(p, np.float64)

    def d(a):
        q = np.exp(log_softmax(np.asarray(a)[..., None] * z))
        g = ((q - p) * z).sum(-1)
        return np.full((), g.mean()) if pooled else g

    a = _bisect(d, lo, hi, () if pooled else z.shape[:-1])
    return a


def floor_ce(z_logq, p, w, v=16):
    """CE(p, (1-w) q + w/v) per row; z_logq is log q (already a log-prob)."""
    q = np.exp(z_logq)
    qm = (1.0 - np.asarray(w)[..., None]) * q + np.asarray(w)[..., None] / v
    return xent(p, np.log(np.clip(qm, EPSF, None)))


def floor_opt(logq, p, pooled=False, lo=0.0, hi=0.999, v=16):
    """w minimising CE(p, (1-w) q + w/v). Convex in w; bisect the derivative."""
    q = np.exp(np.asarray(logq, np.float64))
    p = np.asarray(p, np.float64)

    def d(w):
        qm = (1.0 - np.asarray(w)[..., None]) * q + np.asarray(w)[..., None] / v
        g = -(p * (1.0 / v - q) / np.clip(qm, 1e-30, None)).sum(-1)
        return np.full((), g.mean()) if pooled else g

    return _bisect(d, lo, hi, () if pooled else q.shape[:-1])


def mix_ce(logq, r, p, w):
    q = np.exp(np.asarray(logq, np.float64))
    ww = np.asarray(w)[..., None]
    return xent(p, np.log(np.clip((1.0 - ww) * q + ww * r, EPSF, None)))


def mix_opt(logq, r, p, pooled=False, lo=0.0, hi=0.999):
    """w minimising CE(p, (1-w) q + w r) for a reference distribution r. Convex in w."""
    q = np.exp(np.asarray(logq, np.float64))
    r = np.asarray(r, np.float64)
    p = np.asarray(p, np.float64)

    def d(w):
        ww = np.asarray(w)[..., None]
        qm = (1.0 - ww) * q + ww * r
        g = -(p * (r - q) / np.clip(qm, 1e-30, None)).sum(-1)
        return np.full((), g.mean()) if pooled else g

    return _bisect(d, lo, hi, () if pooled else q.shape[:-1])


def affine_fit(z, p, steps=600, lr=0.05, seed=0):
    """Fit q = softmax(alpha z + b) to soft targets p by mean CE. Convex in (alpha, b).

    The most a CONTEXT-INDEPENDENT output correction can do on a set of predictions: a gain
    and a fixed v-dim logit bias. Adam on the exact gradients (d/dalpha = <q-p, z>,
    d/db = q - p). Returns (alpha, b)."""
    z = np.asarray(z, np.float64)
    p = np.asarray(p, np.float64)
    a, b = 1.0, np.zeros(z.shape[-1])
    ma, va, mb, vb = 0.0, 0.0, np.zeros_like(b), np.zeros_like(b)
    b1, b2, e = 0.9, 0.999, 1e-8
    for t in range(1, steps + 1):
        q = np.exp(log_softmax(a * z + b))
        d = q - p
        ga = float((d * z).sum(-1).mean())
        gb = d.mean(0)
        ma = b1 * ma + (1 - b1) * ga; va = b2 * va + (1 - b2) * ga * ga
        mb = b1 * mb + (1 - b1) * gb; vb = b2 * vb + (1 - b2) * gb * gb
        a -= lr * (ma / (1 - b1 ** t)) / (np.sqrt(va / (1 - b2 ** t)) + e)
        b -= lr * (mb / (1 - b1 ** t)) / (np.sqrt(vb / (1 - b2 ** t)) + e)
    return float(a), b


def affine_ce(z, p, a, b):
    return xent(p, log_softmax(a * np.asarray(z, np.float64) + b))


# ---------------------------------------------------------------------------
# model forward
# ---------------------------------------------------------------------------

def model_logits(model, windows, device="cuda", bs=512, block=None):
    """(n, T, v) float64 logits; optionally also (n, T, d) states at `block`."""
    import torch
    T = model.block_size
    out, st = [], []
    with torch.no_grad():
        for c0 in range(0, len(windows), bs):
            x = torch.as_tensor(windows[c0:c0 + bs, :T], device=device)
            if block is None:
                lg, _ = model(x)
            else:
                lg, _, inter = model(x, return_intermediates=True)
                st.append(inter[block].float().cpu().numpy())
            out.append(lg.float().cpu().numpy())
    z = np.concatenate(out).astype(np.float64)
    return (z, np.concatenate(st)) if block is not None else z
