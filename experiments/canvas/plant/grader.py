"""The grade of record, ported from `rhm/practice/critic/` -- and the damage it is calibrated on.

PROTOCOL (critic/SPEC.md Sec.1, transposed one substrate over):

  verdict  = mean per-token NLL of the PROPOSED completion, under a SECOND reader of the same
             form trained on a DISJOINT split, conditioned on the surround and the request
             (here: the filled neighbourhood + the style id, which is what "surround + request"
             is on this substrate).
  tau      = the q = 0.90 quantile of that reader's own NLL on a VALIDATION slice of GENUINE
             exemplars ("on-style" == "at least as typical as 90% of real exemplars").
             Nothing in that rule touches a shader, a seed, or held-out truth, so it is
             oracle-free and ports verbatim.
  corpus   = the smallest that separates clean from SELF-MANUFACTURED damage at
             pass >= 0.90 clean / <= 0.10 damaged; if none qualifies, the largest.

TWO DEPARTURES FROM THE DONOR, both forced by the substrate and both recorded in SPEC.md:

1. `tau` is per (style, mask-size), not global. On RHM every configuration had the same length
   and the same demand, so one quantile served. Here NLL scales with how much is being asked
   for (a 256-cell completion is not a 1-cell completion) and with how entropic the style is
   (a texture is not a tiling), so a single tau would degenerate into "pass the tilings, fail
   the textures" -- it would grade the STYLE, not the completion. The per-cell rule uses only
   genuine exemplars OF THAT STYLE AT THAT SIZE, so it stays oracle-free. The global-tau
   variant is computed too and reported as an ablation.

2. The verdict is a chain-rule NLL over a random reveal order (`n_orders` orders x `n_steps`
   reveals), not the independent per-cell NLL given the surround alone. The independent form
   (n_steps=1) cannot see whether the proposed cells are coherent WITH EACH OTHER, which is
   exactly what `shuffle` damage destroys; the chain rule is also the honest reading of "mean
   per-token NLL of a completion" for an any-order model. Both are computed; n_steps=1 is the
   cheap column, since the loop pays `d_fb` per call.
"""

import numpy as np
import torch
import torch.nn.functional as F

from canvas.plant.model import amp

DAMAGE_CLASSES = ("shuffle", "other_style", "marginal")     # the calibration rule's set
ALL_CLASSES = ("clean", "same_style_other", "roll", "shuffle", "other_style",
               "marginal", "uniform")


# --------------------------------------------------------------------------- #
# the verdict
# --------------------------------------------------------------------------- #

def _ranks(rng, masks, R):
    """Assign each masked cell a reveal group in [0, R); -1 for cells in the surround."""
    B, T = masks.shape
    rank = np.full((B, T), -1, np.int64)
    for b in range(B):
        idx = np.flatnonzero(masks[b])
        if len(idx) == 0:
            continue
        rank[b, idx] = (rng.permutation(len(idx)) * R) // len(idx)
    return rank


@torch.no_grad()
def score(model, grids, masks, styles, n_orders=2, n_steps=4, seed=0, dev="cuda", bs=256,
          use_style=True):
    """Mean per-token NLL of grids[mask] given grids[~mask] and the style. (N,) float."""
    model.eval()
    rng = np.random.default_rng(seed)
    N = len(grids)
    out = np.zeros(N)
    G = torch.as_tensor(np.asarray(grids), dtype=torch.long, device=dev)
    M = torch.as_tensor(np.asarray(masks), device=dev)
    S = torch.as_tensor(np.asarray(styles), dtype=torch.long, device=dev)
    if not use_style:
        S = torch.full_like(S, model.n_styles)
    for _ in range(n_orders):
        rk_all = _ranks(rng, np.asarray(masks), n_steps)
        for a in range(0, N, bs):
            sl = slice(a, min(a + bs, N))
            y, m, s = G[sl], M[sl], S[sl]
            rk = torch.as_tensor(rk_all[sl], device=dev)
            acc = torch.zeros(y.shape[0], device=dev)
            for r in range(n_steps):
                hide = rk >= r
                x = torch.where(hide, torch.full_like(y, model.K), y)
                with amp(dev):
                    logits = model(x, s)
                lp = F.log_softmax(logits.float(), -1)
                nll = -lp.gather(-1, y[..., None])[..., 0]
                acc += (nll * (rk == r)).sum(1)
            out[sl] += (acc / m.sum(1).clamp(min=1)).cpu().numpy()
    return out / n_orders


@torch.no_grad()
def score_tokens(model, grids, masks, styles, n_orders=2, n_steps=4, seed=0, dev="cuda",
                 bs=256, use_style=True):
    """`score`, un-averaged: the PER-TOKEN NLL of every masked cell. (N, T) float, NaN off-mask.

    Identical rng consumption and identical arithmetic to `score`, so
    `np.nanmean(score_tokens(...), 1)` reproduces `score(...)` exactly (to float error). It
    exists because the grade of record is a MEAN over these tokens, and a mean cannot see a
    conjunction: validity here is "every edge agrees", so a violation is a few very surprising
    tokens inside an otherwise typical region. Tail statistics of this array are the
    re-analysis; nothing about the model, the panel or the threshold rule changes.

    Note on what a "token NLL" is under the chain rule: cell t is scored in the reveal step
    its random rank puts it in, i.e. conditioned on the surround plus the cells revealed
    EARLIER in that order, then averaged over `n_orders` orders. At (n_orders, n_steps) =
    (1, 1) it is the independent NLL given the surround alone.
    """
    model.eval()
    rng = np.random.default_rng(seed)
    N = len(grids)
    M_np = np.asarray(masks)
    T = M_np.shape[1]
    tok = np.zeros((N, T), np.float64)
    G = torch.as_tensor(np.asarray(grids), dtype=torch.long, device=dev)
    M = torch.as_tensor(M_np, device=dev)
    S = torch.as_tensor(np.asarray(styles), dtype=torch.long, device=dev)
    if not use_style:
        S = torch.full_like(S, model.n_styles)
    for _ in range(n_orders):
        rk_all = _ranks(rng, M_np, n_steps)
        for a in range(0, N, bs):
            sl = slice(a, min(a + bs, N))
            y, s = G[sl], S[sl]
            rk = torch.as_tensor(rk_all[sl], device=dev)
            acc = torch.zeros(y.shape, dtype=torch.float32, device=dev)
            for r in range(n_steps):
                hide = rk >= r
                x = torch.where(hide, torch.full_like(y, model.K), y)
                with amp(dev):
                    logits = model(x, s)
                lp = F.log_softmax(logits.float(), -1)
                nll = -lp.gather(-1, y[..., None])[..., 0]
                acc += nll * (rk == r)
            tok[sl] += acc.double().cpu().numpy()
    tok /= n_orders
    tok[~M_np] = np.nan
    return tok


def tau_from(nlls, keys, q=0.90, min_n=8):
    """Oracle-free threshold: the q-quantile of the reader's NLL on genuine exemplars,
    bucketed by `keys` (tuples, e.g. (style, mask_side))."""
    out = {}
    for k in sorted(set(map(tuple, keys))):
        sel = np.array([tuple(x) == k for x in keys])
        if sel.sum() >= min_n:
            out[k] = float(np.quantile(nlls[sel], q))
    return out


def apply_tau(nlls, keys, tau, fallback=None):
    if fallback is None:
        fallback = float(np.median(list(tau.values()))) if tau else float(np.quantile(nlls, 0.9))
    t = np.array([tau.get(tuple(k), fallback) for k in keys])
    return nlls <= t


# --------------------------------------------------------------------------- #
# self-manufactured damage: corrupt a held-out swatch's masked region, touch no program
# --------------------------------------------------------------------------- #

def manufacture(kind, codes, masks, styles, pool, pool_style, marg, K, rng, grid=16):
    """Return a copy of `codes` with the masked region replaced per `kind`.

    Nothing here reads a shader, a seed, an offset, or the held-out pixels: every class is a
    rearrangement of codes the agent already holds. That is what makes the calibration rule
    runnable on a substrate with no oracle.
    """
    out = np.array(codes, np.int64, copy=True)
    N = len(codes)
    for i in range(N):
        idx = np.flatnonzero(masks[i])
        if len(idx) == 0:
            continue
        if kind == "clean":
            continue
        elif kind == "shuffle":
            out[i, idx] = out[i, rng.permutation(idx)]
        elif kind == "other_style":
            cand = np.flatnonzero(pool_style != styles[i])
            out[i, idx] = pool[rng.choice(cand)][idx]
        elif kind == "same_style_other":
            cand = np.flatnonzero(pool_style == styles[i])
            out[i, idx] = pool[rng.choice(cand)][idx]
        elif kind == "roll":
            g = out[i].reshape(grid, grid)
            dr, dc = int(rng.integers(3, grid - 2)), int(rng.integers(3, grid - 2))
            rolled = np.roll(np.roll(g, dr, 0), dc, 1).reshape(-1)
            out[i, idx] = rolled[idx]
        elif kind == "marginal":
            p = marg[styles[i]]
            out[i, idx] = rng.choice(len(p), size=len(idx), p=p)
        elif kind == "uniform":
            out[i, idx] = rng.integers(0, K, size=len(idx))
        else:
            raise ValueError(kind)
    return out


def marginals(codes, styles, n_style, K):
    """Per-style code marginal, from the corpus the agent holds. (n_style, K)."""
    m = np.zeros((n_style, K))
    for s in range(n_style):
        sel = codes[styles == s]
        if len(sel):
            m[s] = np.bincount(sel.reshape(-1), minlength=K)
    m = m + 1e-6
    return m / m.sum(1, keepdims=True)


# --------------------------------------------------------------------------- #
# instruments
# --------------------------------------------------------------------------- #

def roc(nll_cal, keys_cal, nll_clean, keys_clean, nll_dam, keys_dam,
        qs=np.linspace(0.02, 0.999, 60)):
    """The ROC the q = 0.90 rule sits on.

    Deliberately swept over the SAME oracle-free rule the decision uses: tau is always the
    q-quantile of the grader's own NLL on its own validation slice of genuine exemplars, and q
    is the only thing that moves. Sweeping a threshold fitted on the PANEL instead would be a
    different (and unavailable) instrument, since the panel is held-out test data the agent is
    being graded on, not exemplars it holds."""
    rows = []
    for q in qs:
        tau = tau_from(nll_cal, keys_cal, q)
        rows.append((float(q), float(apply_tau(nll_clean, keys_clean, tau).mean()),
                     float(apply_tau(nll_dam, keys_dam, tau).mean())))
    return rows


def pair_stats(a, b):
    """`endo_expansion` Sec.2's pairwise form: how much two graders disagree.
    top1_disagree over random pairs, Spearman-ish rank corr, and the cost one pays the other."""
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    corr = float(np.corrcoef(ra, rb)[0, 1])
    n = len(a)
    rng = np.random.default_rng(0)
    i, j = rng.integers(0, n, 4000), rng.integers(0, n, 4000)
    ok = i != j
    i, j = i[ok], j[ok]
    top1 = float(((a[i] < a[j]) != (b[i] < b[j])).mean())
    # cost: b's pick evaluated under a, relative to a's own range
    pick_b = np.where(b[i] < b[j], a[i], a[j])
    pick_a = np.minimum(a[i], a[j])
    rng_a = float(np.percentile(a, 95) - np.percentile(a, 5)) or 1.0
    cost = float((pick_b - pick_a).mean() / rng_a)
    return {"top1_disagree": top1, "rank_corr": corr, "cost": cost}
