"""Q1: the CE - H(q) identity, the temperature direction, and a CONTINUOUS altitude.

Reads the per-checkpoint `*_calibration.npz` arrays that `logit_reading/calibration.py`
already wrote (model logits, every observer's predictive p_k, the windows and their true
phase -- all 13 checkpoints share the same 4096 eval windows, eval_seed 4242), so nothing
here needs a GPU or a model.

Three things.

1. THE IDENTITY. Pointwise, for q = softmax(z),
       CE(p, q) - H(q)  =  <q - p, z>  =  d/dalpha CE(p, softmax(alpha z)) |_{alpha=1},
   i.e. the "does the model predict its own loss" gap IS the derivative of CE along the
   logit-scaling direction. Checked numerically here, and traced over alpha: the gap is
   monotone increasing in alpha (d gap/d alpha = Var_{q_alpha}(z) >= 0) and its root is
   exactly the temperature that minimises KL(p_L || q_alpha). Reported per checkpoint and
   per context class (window index, token level, entropy decile): a model with a free
   per-class logit scale can zero the gap in every class separately.

2. WHAT alpha DOES TO THE LADDER. Rescaling moves the gap and the ECE; this records what
   it does to KL(p_k || q_alpha) for every observer k, to the mean-argmin best_k, and to
   the per-position best_k histogram.

3. A CONTINUOUS ALTITUDE. `argmin_k mean KL(p_k || q)` is replaced by a fit over a
   one-parameter family interpolating adjacent observers:
       convex      p_kappa = (1 - lam) p_k + lam p_{k+1},           kappa = k + lam
       log-linear  p_kappa ∝ p_k^(1-lam) p_{k+1}^lam                (support collapses to
                   supp p_{k+1} for lam > 0 -- reported, with that caveat)
   plus two two-parameter references that say whether a one-parameter altitude is the right
   shape at all: (k, eps) uniform-smoothed observers (1-eps) p_k + eps/v, and (kappa, alpha)
   tempered convex observers. The residual mean KL at each family's optimum, against the
   discrete argmin and against KL(p_L || q), is the answer to "does a one-parameter family
   fit".

Run:
  modal run -m rhm.logit_reading.altitude.identity::identity_sweep \
      --traj-dir /data/v16_s2_L6_m4_distinct/logit_reading/traj_a1_s42
"""

import json

import numpy as np

from rhm.shared import volume, DATA_DIR, NumpyEncoder
from rhm.logit_reading.calibration import app, _reliability, _r2

EPS = 1e-300


def _log_softmax(z):
    zm = z - z.max(-1, keepdims=True)
    return zm - np.log(np.exp(zm).sum(-1, keepdims=True))


def _ent(p):
    return -(p * np.log(np.clip(p, EPS, None))).sum(-1)


def _xent(p, logq):
    return -(p * logq).sum(-1)


def _kl(p, logq):
    return _xent(p, logq) - _ent(p)


def alpha_stats(z, preds, tok, alpha, L):
    """Everything that depends on the logit scale, at scale `alpha`."""
    logq = _log_softmax(alpha * z)
    q = np.exp(logq)
    p = preds[..., L, :]
    CE = _xent(p, logq)
    H_q = _ent(q)
    gap = CE - H_q
    KL_k = np.stack([_kl(preds[..., k, :], logq) for k in range(L + 1)], -1)
    best_k = KL_k.argmin(-1)
    conf = q.max(-1)
    acc = (q.argmax(-1) == tok).astype(np.float64)
    acc_exp = np.take_along_axis(p, q.argmax(-1)[..., None], -1)[..., 0]
    return {
        "alpha": float(alpha),
        "gap": float(gap.mean()), "abs_gap": float(np.abs(gap).mean()),
        "CE": float(CE.mean()), "H_q": float(H_q.mean()),
        "KL_p_q": float((CE - _ent(p)).mean()),
        "KL_k_q": [float(KL_k[..., k].mean()) for k in range(L + 1)],
        "best_k_meanargmin": int(np.argmin([KL_k[..., k].mean() for k in range(L + 1)])),
        "best_k_hist": (np.bincount(best_k.ravel(), minlength=L + 1) / best_k.size).round(4).tolist(),
        "ece_realised": _reliability(conf.ravel(), acc.ravel(), 15)[0],
        "ece_expected": _reliability(conf.ravel(), acc_exp.ravel(), 15)[0],
        "R2_CE_on_Hq": _r2(CE, H_q),
        "top1_conf": float(conf.mean()), "top1_acc": float(acc.mean()),
    }


def _root_alpha(fn, lo=0.0, hi=8.0, iters=40):
    """Bisect the (increasing) mean gap to its root. fn(alpha) -> mean gap."""
    flo, fhi = fn(lo), fn(hi)
    if flo > 0 or fhi < 0:
        return float("nan")
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if fn(mid) < 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _mean_gap_fn(z, p):
    """alpha -> mean <q_alpha - p, z>  (== mean CE - H(q) at that scale)."""
    def f(a):
        q = np.exp(_log_softmax(a * z))
        return float(((q - p) * z).sum(-1).mean())
    return f


def convex_family(preds, kappa, L):
    k = min(int(np.floor(kappa)), L - 1)
    lam = kappa - k
    return (1.0 - lam) * preds[..., k, :] + lam * preds[..., k + 1, :]


def loglinear_family(preds, kappa, L):
    k = min(int(np.floor(kappa)), L - 1)
    lam = kappa - k
    lp = (1.0 - lam) * np.log(np.clip(preds[..., k, :], EPS, None)) \
        + lam * np.log(np.clip(preds[..., k + 1, :], EPS, None))
    lp = lp - lp.max(-1, keepdims=True)
    p = np.exp(lp)
    return p / p.sum(-1, keepdims=True)


def fit_kappa(preds, logq, L, family, grid):
    """Pooled kappa* (minimises mean KL(p_kappa || q)) and the per-position argmin."""
    curve = []
    best_pp = None
    best_val_pp = None
    for kap in grid:
        pk = family(preds, kap, L)
        d = _kl(pk, logq)
        curve.append(float(d.mean()))
        if best_val_pp is None:
            best_val_pp, best_pp = d.copy(), np.full(d.shape, kap)
        else:
            m = d < best_val_pp
            best_val_pp[m] = d[m]
            best_pp[m] = kap
    curve = np.asarray(curve)
    i = int(curve.argmin())
    return {"kappa_star": float(grid[i]), "KL_at_star": float(curve[i]),
            "curve": curve.round(6).tolist(),
            "kappa_pp_mean": float(best_pp.mean()), "kappa_pp_median": float(np.median(best_pp)),
            "kappa_pp_hist": np.histogram(best_pp, bins=np.arange(0, L + 1.001, 0.5))[0].tolist(),
            "KL_pp_min_mean": float(best_val_pp.mean())}, best_pp


def analyse_ckpt(npz_path, L=6, s=2, grid_step=0.05):
    from rhm.logit_reading.flat_oracle import leaf_levels
    z_ = np.load(npz_path)
    z = z_["logits"].astype(np.float64)
    preds = z_["preds"].astype(np.float64)
    windows, phase = z_["windows"], z_["phase"]
    n, T, v = z.shape
    tok = windows[:, 1:]
    p = preds[..., L, :]
    lev = leaf_levels((phase[:, None] + 1 + np.arange(T)[None, :]) % (s ** L), s, L)
    logq1 = _log_softmax(z)
    q1 = np.exp(logq1)
    H_q = _ent(q1)
    CE = _xent(p, logq1)
    gap = CE - H_q
    tempgrad = ((q1 - p) * z).sum(-1)

    out = {"npz": npz_path, "n": int(n), "T": int(T)}
    out["identity"] = {
        "mean_gap": float(gap.mean()), "mean_tempgrad": float(tempgrad.mean()),
        "max_abs_diff": float(np.abs(gap - tempgrad).max()),
        "rel_max_abs_diff": float(np.abs(gap - tempgrad).max() / max(np.abs(gap).max(), 1e-12)),
        "corr": float(np.corrcoef(gap.ravel(), tempgrad.ravel())[0, 1]),
    }

    # ---- alpha sweep + the gap's root ----
    f = _mean_gap_fn(z, p)
    a_star = _root_alpha(f)
    out["alpha_star_overall"] = float(a_star)
    out["gap_at_alpha0"] = float(f(0.0))
    alphas = [0.6, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.5, 2.0]
    if np.isfinite(a_star):
        alphas.append(round(float(a_star), 4))
    out["alpha_sweep"] = [alpha_stats(z, preds, tok, a, L) for a in sorted(set(alphas))]

    # ---- per-class gap, R2 and per-class optimal alpha ----
    def class_block(mask):
        gsel, CEsel, Hsel = gap[mask], CE[mask], H_q[mask]
        fk = _mean_gap_fn(z[mask], p[mask])
        return {"n": int(mask.sum()), "mean_gap": float(gsel.mean()), "sd_gap": float(gsel.std()),
                "var_gap_over_var_CE": float(gsel.var() / max(CEsel.var(), 1e-30)),
                "R2_CE_on_Hq": _r2(CEsel, Hsel), "alpha_star": float(_root_alpha(fk)),
                "mean_CE": float(CEsel.mean()), "mean_Hq": float(Hsel.mean())}

    jj = np.broadcast_to(np.arange(T)[None, :], (n, T))
    classes = {}
    for a, b in [(0, 4), (4, 16), (16, 32), (32, 64)]:
        classes[f"winpos_{a}-{b}"] = class_block((jj >= a) & (jj < b))
    for l in range(L + 1):
        m = lev == l
        if m.sum() > 200:
            classes[f"level_{l}"] = class_block(m)
    dec = np.digitize(H_q, np.quantile(H_q, np.arange(0.1, 1.0, 0.1)))
    for d in range(10):
        classes[f"Hq_decile_{d}"] = class_block(dec == d)
    out["classes"] = classes
    # pooled R2 vs within-class R2: does the per-position R2 reduce to the gap condition?
    out["R2_decomposition"] = {
        "var_Hq": float(H_q.var()), "var_gap": float(gap.var()), "var_CE": float(CE.var()),
        "cov_Hq_gap": float(np.cov(H_q.ravel(), gap.ravel())[0, 1]),
        "R2_CE_on_Hq": _r2(CE, H_q),
        "R2_CE_on_Hq_within_winpos": _r2(CE - CE.mean(0, keepdims=True),
                                         H_q - H_q.mean(0, keepdims=True)),
    }

    # ---- continuous altitude ----
    grid = np.round(np.arange(0.0, L + 1e-9, grid_step), 4)
    cx, kpp = fit_kappa(preds, logq1, L, convex_family, grid)
    ll, _ = fit_kappa(preds, logq1, L, loglinear_family, grid)
    out["altitude"] = {"convex": cx, "loglinear": ll, "grid_step": grid_step}
    KL_k = np.stack([_kl(preds[..., k, :], logq1) for k in range(L + 1)], -1)
    out["altitude"]["discrete"] = {
        "KL_k_q": [float(KL_k[..., k].mean()) for k in range(L + 1)],
        "best_k": int(np.argmin([KL_k[..., k].mean() for k in range(L + 1)])),
        "KL_best": float(min(KL_k[..., k].mean() for k in range(L + 1))),
        "KL_pp_min_mean": float(KL_k.min(-1).mean()),
    }
    # (k, eps) uniform-smoothed observers
    eps_grid = [0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
    sm = {}
    for k in range(L + 1):
        sm[k] = [float(_kl((1 - e) * preds[..., k, :] + e / v, logq1).mean()) for e in eps_grid]
    bk = min(sm, key=lambda k: min(sm[k]))
    out["altitude"]["smoothed"] = {"eps_grid": eps_grid, "KL": {int(k): [round(x, 5) for x in r]
                                                                for k, r in sm.items()},
                                   "best_k": int(bk), "best_eps": float(eps_grid[int(np.argmin(sm[bk]))]),
                                   "KL_at_best": float(min(sm[bk]))}
    # (kappa, alpha): is q a TEMPERED coarse observer?
    p_star = convex_family(preds, cx["kappa_star"], L)
    a_grid = [0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15]
    ka = [float(_kl(p_star, _log_softmax(a * z)).mean()) for a in a_grid]
    out["altitude"]["tempered"] = {"alpha_grid": a_grid, "KL": [round(x, 5) for x in ka],
                                   "best_alpha": float(a_grid[int(np.argmin(ka))]),
                                   "KL_at_best": float(min(ka))}
    # how far the fitted observer is from the truth, and H-tracking
    out["altitude"]["fit_quality"] = {
        "KL_pL_pkappa": float(_kl(p, np.log(np.clip(p_star, EPS, None))).mean()),
        "KL_pL_q": float((CE - _ent(p)).mean()),
        "R2_Hq_on_Hkappa": _r2(H_q, _ent(p_star)),
        "R2_Hq_on_HpL": _r2(H_q, _ent(p)),
        "KL_truth_k": [float(_kl(p, np.log(np.clip(preds[..., k, :], EPS, None))).mean())
                       for k in range(L + 1)],
    }
    # per-position kappa vs per-position best_k, and kappa by token level / window position
    out["altitude"]["kappa_by_level"] = {int(l): float(kpp[lev == l].mean())
                                         for l in range(L + 1) if (lev == l).sum() > 200}
    out["altitude"]["kappa_by_winpos"] = {f"{a}-{b}": float(kpp[:, a:b].mean())
                                          for a, b in [(0, 4), (4, 16), (16, 32), (32, 64)]}
    return out


@app.function(volumes={DATA_DIR: volume}, timeout=3 * 3600, memory=16384, cpu=4.0)
def identity_sweep(traj_dir: str, grid_step: float = 0.05, out_name: str = "altitude_identity",
                   limit: int = 0):
    import glob
    import os
    import resource
    volume.reload()
    paths = sorted(glob.glob(f"{traj_dir}/step*_calibration.npz"))
    if limit:
        paths = paths[:limit]
    print(f"{len(paths)} checkpoints", flush=True)
    rows = []
    for pth in paths:
        step = int(os.path.basename(pth)[4:10])
        r = analyse_ckpt(pth, grid_step=grid_step)
        r["step"] = step
        rows.append(r)
        i = r["identity"]
        al = r["altitude"]
        print(f"step {step:>6}  gap {i['mean_gap']:+.5f}  |gap-dCE/da| {i['max_abs_diff']:.2e}  "
              f"a* {r['alpha_star_overall']:.4f}  best_k {al['discrete']['best_k']} "
              f"(KL {al['discrete']['KL_best']:.4f})  kappa* {al['convex']['kappa_star']:.2f} "
              f"(KL {al['convex']['KL_at_star']:.4f})  KL(p_L||q) {al['fit_quality']['KL_pL_q']:.4f}",
              flush=True)
    if limit:
        return rows[-1]["altitude"]["convex"]["kappa_star"]
    with open(f"{traj_dir}/{out_name}.json", "w") as f:
        json.dump(rows, f, cls=NumpyEncoder, indent=1)
    volume.commit()
    print(f"peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.1f} GB", flush=True)
    return f"{traj_dir}/{out_name}.json"
