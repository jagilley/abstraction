"""The hold-and-discount forecast, and the mixing consumer that sits between it and the
model's native one.

The model's native policy after an off-grammar token is to REINTERPRET: it falls back to the
noise-aware observer at `k* - 1`, the finest reading under which the token was still legal
(`altitude/` Q4b). The other policy -- the one no member of either observer family and no part
of the model implements -- is to HOLD the fine reading and DISCOUNT the token as a glitch.
This module builds that second forecast in the model's own terms.

There is no mask token, so "what would I have predicted had that token been what I expected"
is built by re-running the model with the flagged position overwritten. With v = 16 the
marginalisation over what the token *should* have been is exact at 16 forward passes:

    q_hold(. | x_<t, x_>t)  =  sum_{x'} omega(x')  q( . | x_<t, x', x_{t+1..} )

Four weightings of the same 16 passes, all free once the passes are done:

  argmax    omega = onehot(argmax_x q(x | x_<t))          the cheap single-pass form
  topK      omega ∝ q(x | x_<t) on the top K, renormalised
  full      omega ∝ q(x | x_<t) over the whole vocabulary (K = v; exact prior marginal)
  pf        omega_k(tau) ∝ q(x'_k | x_<t) * prod_{u<tau} q(x_{t+u+1} | x_<t, x'_k, ...)
            -- a particle filter over the imputed token: the continuation reweights the
            hypotheses as it arrives, which is the only information that distinguishes a
            glitch from a structural edit at all. Causal at every offset.

The consumer is a mixing weight `w` at each offset tau after the event:

    q_w = (1 - w) q_native + w q_hold,   w = 0 reinterpret, w = 1 hold and discount

`-log q_w(x)` is convex in w (minus log of an affine function), so the optimal w for any
weighted set of predictions is exact by bisection on the derivative, and a gate -> w map is
a quantile step function whose every bin is that exact minimiser -- the same shape
`coeruleus/gainloop.py` used for its temperature, fitted the same way, on the model's own
realised NLL with no oracle in it.

The `pf` weighting also hands over the model's own log Bayes factor between the two
hypotheses, a gate with no free parameters:

    log BF(tau) = [ -log v + logsumexp_k( log omega_k + cum_k(tau) ) ]
                - [ log q_native(x_t | x_<t) + cum_native(tau) ]

(the prior odds of a glitch are an additive constant, absorbed by the fitted map).
"""

import numpy as np

EPSF = 1e-300


# ---------------------------------------------------------------------------
# forward passes
# ---------------------------------------------------------------------------

def logq_full(model, windows, device="cuda", bs=512):
    """(n, T, v) float64 log-forecasts over a whole window."""
    import torch
    T = model.block_size
    out = []
    with torch.no_grad():
        for c0 in range(0, len(windows), bs):
            x = torch.as_tensor(windows[c0:c0 + bs, :T], device=device)
            lg, _ = model(x)
            out.append(torch.log_softmax(lg.float(), -1).cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def gather_offsets(a, t0, taus):
    """a (n, T, ...) at prediction indices t0 + taus -> (n, len(taus), ...)."""
    idx = np.asarray(t0)[:, None] + np.asarray(taus)[None, :]
    return a[np.arange(len(a))[:, None], idx]


def states_offsets(model, windows, t0, taus, blocks, device="cuda", bs=256):
    """{block: (n, len(taus), d)} residual states at t0 + taus, one forward pass."""
    import torch
    if isinstance(blocks, str):
        blocks = [blocks]
    T = model.block_size
    d = model.transformer.wte.weight.shape[1]
    taus = np.asarray(taus)
    out = {b: np.empty((len(windows), len(taus), d), np.float32) for b in blocks}
    with torch.no_grad():
        for c0 in range(0, len(windows), bs):
            x = torch.as_tensor(windows[c0:c0 + bs, :T], device=device)
            _, _, inter = model(x, return_intermediates=True)
            idx = t0[c0:c0 + bs, None] + taus[None, :]
            ar = np.arange(len(x))[:, None]
            for b in blocks:
                out[b][c0:c0 + bs] = inter[b].float().cpu().numpy()[ar, idx]
    return out


def impute_logq(model, windows, pos, anchor, cands, H, device="cuda", bs=512):
    """Re-run the model with `windows[:, pos]` overwritten by each candidate.

    pos (n,) the position to overwrite; anchor (n,) the event position the offsets are
    measured from (pos = anchor - delta); cands (n, K).
    Returns (n, K, H+1, v) float32 log-forecasts at prediction indices anchor + 0..H."""
    import torch
    T = model.block_size
    n, K = cands.shape
    v = model.transformer.wte.weight.shape[0]
    out = np.empty((n, K, H + 1, v), np.float32)
    ar = np.arange(n)
    off = np.arange(H + 1)
    with torch.no_grad():
        for k in range(K):
            W = np.array(windows[:, :T])
            W[ar, pos] = cands[:, k]
            for c0 in range(0, n, bs):
                x = torch.as_tensor(W[c0:c0 + bs], device=device)
                lg, _ = model(x)
                ls = torch.log_softmax(lg.float(), -1).cpu().numpy()
                idx = anchor[c0:c0 + bs, None] + off[None, :]
                out[c0:c0 + bs, k] = ls[np.arange(len(x))[:, None], idx]
            del W
    return out


# ---------------------------------------------------------------------------
# the held forecast
# ---------------------------------------------------------------------------

def _logsumexp(a, axis):
    m = a.max(axis, keepdims=True)
    return (m + np.log(np.exp(a - m).sum(axis, keepdims=True))).squeeze(axis)


def prep_imputed(logq_imp, log_prior, tok):
    """The K imputed passes, decoded once: probabilities, the realised-token log-likelihood
    per candidate, and the normalised log prior over the candidates."""
    qi = np.exp(np.asarray(logq_imp, np.float64))                     # (n, K, H+1, v)
    ll = np.log(np.clip(np.take_along_axis(qi, tok[:, None, :, None], -1)[..., 0],
                        EPSF, None))                                  # (n, K, H+1)
    lp = np.asarray(log_prior, np.float64)
    lp = lp - _logsumexp(lp, 1)[:, None]
    return {"qi": qi, "ll": ll, "lp": lp, "n": qi.shape[0], "K": qi.shape[1],
            "Hp": qi.shape[2], "v": qi.shape[3]}


def _weights(P, variant, topk=4):
    lp, K, Hp = P["lp"], P["K"], P["Hp"]
    if variant == "argmax":
        w = np.full_like(lp, -np.inf)
        w[np.arange(len(lp)), lp.argmax(1)] = 0.0
        return np.repeat(w[:, None, :], Hp, 1)
    if variant.startswith("top"):
        kk = min(int(variant[3:]) if variant[3:] else topk, K)
        cut = np.sort(lp, 1)[:, -kk][:, None]
        w = np.where(lp >= cut, lp, -np.inf)
        return np.repeat((w - _logsumexp(w, 1)[:, None])[:, None, :], Hp, 1)
    if variant == "full":
        return np.repeat(lp[:, None, :], Hp, 1)
    if variant == "pf":
        cum = np.concatenate([np.zeros((P["n"], K, 1)), np.cumsum(P["ll"], -1)[:, :, :-1]], -1)
        lw = np.transpose(lp[:, :, None] + cum, (0, 2, 1))            # (n, Hp, K)
        return lw - _logsumexp(lw, 2)[:, :, None]
    raise ValueError(variant)


def hold_forecast(P, variant="full", topk=4):
    """(n, H+1, v) held forecast under one weighting of the imputed passes."""
    lw = _weights(P, variant, topk)
    q = np.einsum("ntk,nkti->nti", np.exp(lw), P["qi"])
    return q / np.clip(q.sum(-1, keepdims=True), EPSF, None)


def log_bayes_factor(P, logq_nat, tok, s_event):
    """log P(what followed | the flagged token was a glitch) - log P(. | it was real).

    Causal: the term at offset tau uses only tokens up to anchor + tau. `s_event` is
    -log q_native(x_anchor | prefix), the flagged token's own surprisal."""
    cum = np.concatenate([np.zeros((P["n"], P["K"], 1)),
                          np.cumsum(P["ll"], -1)[:, :, :-1]], -1)
    glitch = _logsumexp(P["lp"][:, :, None] + cum, 1) - np.log(P["v"])
    lln = np.take_along_axis(np.asarray(logq_nat, np.float64), tok[..., None], -1)[..., 0]
    cumn = np.concatenate([np.zeros((P["n"], 1)), np.cumsum(lln, -1)[:, :-1]], -1)
    return glitch - (cumn - np.asarray(s_event)[:, None])


# ---------------------------------------------------------------------------
# the mixing consumer
# ---------------------------------------------------------------------------

def mix_logq(logq_nat, q_hold, w):
    """log((1-w) q_native + w q_hold). w broadcasts against the leading axes."""
    w = np.asarray(w, np.float64)[..., None]
    q = (1.0 - w) * np.exp(np.asarray(logq_nat, np.float64)) + w * np.asarray(q_hold, np.float64)
    return np.log(np.clip(q, EPSF, None))


def realised_ab(logq_nat, q_hold, tok):
    """(a, b): the realised token's probability under the native and the held forecast."""
    a = np.take_along_axis(np.exp(np.asarray(logq_nat, np.float64)), tok[..., None], -1)[..., 0]
    b = np.take_along_axis(np.asarray(q_hold, np.float64), tok[..., None], -1)[..., 0]
    return a, b


def nll_at_w(a, b, w):
    w = np.asarray(w, np.float64)
    return -np.log(np.clip((1.0 - w) * np.asarray(a) + w * np.asarray(b), EPSF, None))


def w_opt(a, b, sw=None, lo=0.0, hi=0.999, iters=50):
    """w minimising the (weighted) mean realised NLL. Convex in w; bisect the derivative."""
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    sw = np.ones_like(a) if sw is None else np.asarray(sw, np.float64).ravel()
    if a.size == 0 or sw.sum() <= 0:
        return float("nan")

    def d(w):
        return float((sw * (-(b - a) / np.clip((1 - w) * a + w * b, 1e-30, None))).sum())
    if d(lo) >= 0:
        return float(lo)
    if d(hi) <= 0:
        return float(hi)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if d(mid) < 0:
            lo = mid
        else:
            hi = mid
    return float(0.5 * (lo + hi))


class WMap:
    """w = f(gate): a step function on quantile bins of the gate."""

    def __init__(self, edges, ws, ns=None, qs=None):
        self.edges, self.ws = np.asarray(edges), np.asarray(ws)
        self.ns, self.qs = ns, qs

    def __call__(self, g):
        return self.ws[np.clip(np.digitize(np.asarray(g), self.edges), 0, len(self.ws) - 1)]

    def summary(self):
        mono = (float(np.corrcoef(np.arange(len(self.ws)), self.ws)[0, 1])
                if len(self.ws) > 1 and self.ws.std() > 1e-12 else 0.0)
        return {"n_bins": int(len(self.ws)), "w": self.ws.round(4).tolist(),
                "bin_counts": self.ns, "quantiles": self.qs, "spearman_bin_vs_w": mono}


QS = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98]


def fit_wmap(g, a, b, sw=None, qs=None, min_n=40):
    """One exact w per quantile bin of the gate, minimising the realised NLL in that bin."""
    g = np.asarray(g, np.float64).ravel()
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    sw = np.ones_like(g) if sw is None else np.asarray(sw, np.float64).ravel()
    qs = qs or QS
    edges = np.unique(np.quantile(g, qs))
    bidx = np.clip(np.digitize(g, edges), 0, len(edges))
    ws, ns = np.zeros(len(edges) + 1), np.zeros(len(edges) + 1, int)
    for i in range(len(edges) + 1):
        sel = bidx == i
        ns[i] = int(sel.sum())
        ws[i] = w_opt(a[sel], b[sel], sw[sel]) if sel.sum() >= min_n else 0.0
    return WMap(edges, np.nan_to_num(ws), ns.tolist(), qs)


def ent(p):
    return -(np.asarray(p) * np.log(np.clip(p, EPSF, None))).sum(-1)


def xent(p, logq):
    return -(np.asarray(p) * np.asarray(logq)).sum(-1)
