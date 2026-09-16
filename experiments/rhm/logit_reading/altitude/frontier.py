"""Q2: can the model locate its own constituent boundaries from its logits?

Under observer k the predictive returns to the depth-k marginal at every depth-k
boundary, so a model whose logits are near p_k should have a per-position entropy with
period s^k, with peaks on the boundaries. Three readouts, all on the held-out flat
windows the Part-1 arrays already cover (4096 windows, eval_seed 4242):

  A. LOGIT-ONLY PERIODICITY. For a per-position statistic X (entropy of q, negative top-1
     confidence, realised surprisal), detrended by the across-window mean at each window
     index: for every candidate period P = s^k, the offset profile m(r) = mean of X over
     j = r (mod P); strength = adjusted R^2 over offsets; r_hat = argmax_r m(r). The
     dominant period is argmax_k adjR2. Accuracy = P(r_hat == (-phase) mod P) against
     chance 1/P. A second, stronger detector correlates the window's X profile against the
     MODEL-INDEPENDENT template E[H(p_k) | leaf level], which uses the ladder's shape but
     not the window's phase.

  B. WHERE THE NEWS IS. Mean KL(p_L || q), KL(p_{k+1} || p_k) and every H by leaf level
     (the s-adic valuation of the absolute leaf index -- the highest constituent the token
     opens), and by window-position bucket. "Is the next level's news concentrated at those
     boundaries" is this table.

  C. IS THE PHASE EVEN IDENTIFIABLE. `flat_predictive(..., return_phase=True)` gives each
     observer its OWN posterior over the window's phase mod s^k. Its accuracy and the log
     posterior of the true phase separate "the model cannot find the boundary" from
     "nothing could". GPU; model-independent, so computed once.

Run:
  modal run -m rhm.logit_reading.altitude.frontier::frontier_sweep \
      --traj-dir /data/v16_s2_L6_m4_distinct/logit_reading/traj_a1_s42
  modal run -m rhm.logit_reading.altitude.frontier::phase_identifiability
"""

import json

import numpy as np

from rhm.shared import volume, DATA_DIR, NumpyEncoder
from rhm.logit_reading.calibration import app
from rhm.logit_reading.altitude.identity import _log_softmax, _ent, _kl, EPS


def offset_profiles(X, P, j_lo):
    """X (n, T) detrended. Returns (profile (n, P), adjR2 (n,), r_hat (n,))."""
    n, T = X.shape
    j = np.arange(T)
    use = j >= j_lo
    r = j[use] % P
    prof = np.zeros((n, P))
    cnt = np.zeros(P)
    for rr in range(P):
        m = use & (j % P == rr)
        cnt[rr] = m.sum()
        prof[:, rr] = X[:, m].mean(1) if m.sum() else np.nan
    Xu = X[:, use]
    N = Xu.shape[1]
    ss_t = ((Xu - Xu.mean(1, keepdims=True)) ** 2).sum(1)
    fit = prof[:, r]
    ss_w = ((Xu - fit) ** 2).sum(1)
    adj = 1.0 - (ss_w / max(N - P, 1)) / np.maximum(ss_t / (N - 1), 1e-30)
    return prof, adj, prof.argmax(1)


def template_detector(X, tmpl, P, j_lo):
    """r_hat = the boundary column that best aligns X with the leaf-residue template
    `tmpl` (tmpl[c] = the statistic's expected value at leaf residue c mod P). A boundary
    at column r means column j carries leaf residue (j - r) mod P."""
    n, T = X.shape
    j = np.arange(T)
    use = j >= j_lo
    Xu = X[:, use] - X[:, use].mean(1, keepdims=True)
    sc = np.empty((n, P))
    for r in range(P):
        t = tmpl[(j[use] - r) % P]
        sc[:, r] = Xu @ (t - t.mean())
    return sc.argmax(1)


def analyse_ckpt(npz_path, L=6, s=2, j_lo=0):
    from rhm.logit_reading.flat_oracle import leaf_levels
    z_ = np.load(npz_path)
    z = z_["logits"].astype(np.float64)
    preds = z_["preds"].astype(np.float64)
    windows, phase = z_["windows"], z_["phase"]
    n, T, v = z.shape
    tok = windows[:, 1:]
    logq = _log_softmax(z)
    q = np.exp(logq)
    p = preds[..., L, :]
    H_q = _ent(q)
    conf = q.max(-1)
    surpr = -np.take_along_axis(logq, tok[..., None], -1)[..., 0]
    leaf = (phase[:, None] + 1 + np.arange(T)[None, :]) % (s ** L)
    lev = leaf_levels(leaf, s, L)
    KL_pL_q = _kl(p, logq)

    out = {"npz": npz_path, "j_lo": j_lo}

    # ---- B. where the news is (by leaf level) ----
    H_k = np.stack([_ent(preds[..., k, :]) for k in range(L + 1)], -1)
    news = np.stack([_kl(preds[..., k + 1, :], np.log(np.clip(preds[..., k, :], EPS, None)))
                     for k in range(L)], -1)                    # KL(p_{k+1} || p_k)
    by_lev = {}
    for l in range(L + 1):
        m = lev == l
        if m.sum() < 50:
            continue
        by_lev[int(l)] = {
            "n": int(m.sum()), "frac": float(m.mean()),
            "H_q": float(H_q[m].mean()), "H_pL": float(H_k[..., L][m].mean()),
            "H_k": [float(H_k[..., k][m].mean()) for k in range(L + 1)],
            "KL_pL_q": float(KL_pL_q[m].mean()),
            "KL_pL_q_share": float(KL_pL_q[m].sum() / KL_pL_q.sum()),
            "news_k_to_k1": [float(news[..., k][m].mean()) for k in range(L)],
            "news_share": [float(news[..., k][m].sum() / news[..., k].sum()) for k in range(L)],
        }
    out["by_level"] = by_lev

    # ---- A. logit-only periodicity ----
    stats = {"H_q": H_q, "neg_conf": -conf, "surprisal": surpr}
    per = {}
    for nm, X in stats.items():
        Xd = X - X.mean(0, keepdims=True)
        rows = {}
        adjs = []
        for k in range(1, L + 1):
            P = s ** k
            prof, adj, rhat = offset_profiles(Xd, P, j_lo)
            # target at column j is leaf (phase + 1 + j); a depth-k boundary sits at
            # column j = -(phase + 1) mod P
            true_r = (-phase - 1) % P
            tmpl = np.array([np.mean(H_k[..., min(k, L)][(leaf % P) == rr]) for rr in range(P)])
            rhat_t = template_detector(Xd, tmpl, P, j_lo)
            rows[k] = {"period": int(P), "adjR2_mean": float(np.nanmean(adj)),
                       "adjR2_frac_pos": float(np.nanmean(adj > 0)),
                       "acc_argmax": float((rhat == true_r).mean()), "chance": 1.0 / P,
                       "acc_template": float((rhat_t == true_r).mean()),
                       "err_hist_argmax": (np.bincount((rhat - true_r) % P, minlength=P)
                                           / n).round(4).tolist(),
                       "pooled_profile": [float(Xd[(leaf % P) == rr].mean()) for rr in range(P)]}
            # a per-window offset profile needs several samples per offset; with a
            # T-token window that caps the resolvable period at about T/3.
            n_used = int((np.arange(T) >= j_lo).sum())
            ok = n_used / P >= 3
            adjs.append(np.nan_to_num(adj, nan=-9.0) if ok else np.full(len(Xd), -9.0))
            rows[k]["resolvable"] = bool(ok)
        A = np.stack(adjs, -1)                                   # (n, L) over k = 1..L
        dom = A.argmax(-1) + 1
        rows["k_max_resolvable"] = int(max(k for k in range(1, L + 1)
                                           if int((np.arange(T) >= j_lo).sum()) / s ** k >= 3))
        rows["dominant_k_hist"] = (np.bincount(dom, minlength=L + 1) / n).round(4).tolist()
        rows["dominant_k_mean"] = float(dom.mean())
        per[nm] = rows
    out["periodicity"] = per
    return out


@app.function(volumes={DATA_DIR: volume}, timeout=3 * 3600, memory=16384, cpu=4.0)
def frontier_sweep(traj_dir: str, j_lo: int = 0, out_name: str = "altitude_frontier",
                   limit: int = 0):
    import glob
    import os
    import resource
    volume.reload()
    paths = sorted(glob.glob(f"{traj_dir}/step*_calibration.npz"))
    if limit:
        paths = paths[:limit]
    rows = []
    for pth in paths:
        r = analyse_ckpt(pth, j_lo=j_lo)
        r["step"] = int(os.path.basename(pth)[4:10])
        rows.append(r)
        hq = r["periodicity"]["H_q"]
        print(f"step {r['step']:>6}  dominant_k {hq['dominant_k_mean']:.2f}  "
              + "  ".join(f"k{k}:{hq[k]['acc_argmax']:.2f}/{hq[k]['acc_template']:.2f}"
                          for k in range(1, 7)), flush=True)
    if limit:
        return rows[-1]["periodicity"]["H_q"]["dominant_k_mean"]
    with open(f"{traj_dir}/{out_name}.json", "w") as f:
        json.dump(rows, f, cls=NumpyEncoder, indent=1)
    volume.commit()
    print(f"peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.1f} GB", flush=True)
    return f"{traj_dir}/{out_name}.json"


# ---------------------------------------------------------------------------
# C. Is the phase identifiable at all? Each observer's own posterior over psi mod s^k.
# ---------------------------------------------------------------------------

@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=3 * 3600, memory=16384)
def phase_identifiability(v: int = 16, s: int = 2, depth: int = 6, m: int = 4,
                          rule_seed: int = 0, alpha: float = 1.0, weight_seed: int = 1,
                          n_windows: int = 512, eval_seed: int = 4242, chunk: int = 4,
                          out_dir: str = "/data/v16_s2_L6_m4_distinct/logit_reading"):
    """Observer k's posterior over the window phase mod s^k, per window position."""
    import torch
    from rhm.rhm_data import generate_rules_distinct
    from rhm.logit_reading.grammar import synonym_weights
    from rhm.logit_reading.flat_oracle import flat_predictive, sample_flat_windows
    volume.reload()
    L, T = depth, s ** depth
    rules = generate_rules_distinct(v, s, L, m, seed=rule_seed)
    rule_w = None if alpha <= 0 else synonym_weights(v, L, m, alpha, weight_seed)
    windows, phase = sample_flat_windows(rules, n_windows, T + 1, eval_seed, rule_w=rule_w)
    res = {"n_windows": n_windows, "eval_seed": eval_seed, "by_k": {}}
    for k in range(1, L + 1):
        S = s ** k
        acc = np.zeros((n_windows, T + 1))
        lpt = np.zeros((n_windows, T + 1))
        ent = np.zeros((n_windows, T + 1))
        for c0 in range(0, n_windows, chunk):
            w = windows[c0:c0 + chunk]
            _, logpi, _ = flat_predictive(w, rules, k, device="cuda", chunk=chunk,
                                          return_phase=True, rule_w=rule_w)
            lp = logpi.numpy()                                  # (nc, T+1, S) after w_<=j
            tr = phase[c0:c0 + chunk] % S
            acc[c0:c0 + chunk] = (lp.argmax(-1) == tr[:, None])
            lpt[c0:c0 + chunk] = lp[np.arange(len(w))[:, None], np.arange(T + 1)[None, :],
                                    tr[:, None]]
            ent[c0:c0 + chunk] = -(np.exp(lp) * lp).sum(-1)
        buckets = {f"{a}-{b}": (a, b) for a, b in [(0, 4), (4, 16), (16, 32), (32, 65)]}
        res["by_k"][k] = {
            "S": int(S), "chance_acc": 1.0 / S,
            "acc_by_pos": {nm: float(acc[:, a:b].mean()) for nm, (a, b) in buckets.items()},
            "logpost_true_by_pos": {nm: float(lpt[:, a:b].mean()) for nm, (a, b) in buckets.items()},
            "post_entropy_by_pos": {nm: float(ent[:, a:b].mean()) for nm, (a, b) in buckets.items()},
            "acc_final": float(acc[:, -1].mean()), "logpost_true_final": float(lpt[:, -1].mean()),
            "max_entropy": float(np.log(S)),
        }
        r = res["by_k"][k]
        print(f"observer k={k}  S={S}  acc(32-64) {r['acc_by_pos']['32-65']:.3f} "
              f"(chance {1/S:.3f})  logpost_true {r['logpost_true_by_pos']['32-65']:+.3f} "
              f"(uniform {-np.log(S):+.3f})", flush=True)
    with open(f"{out_dir}/altitude_phase_identifiability.json", "w") as f:
        json.dump(res, f, cls=NumpyEncoder, indent=1)
    volume.commit()
    return res


# ---------------------------------------------------------------------------
# Q2b over the whole trajectory: is the model's residual the shape of the rung the
# fitted observer is still missing?
# ---------------------------------------------------------------------------

def residual_ckpt(npz_path, kappa, L=6, s=2):
    """Per leaf level: KL(p_L||q) -- what the model still gets wrong -- beside
    KL(p_L||p_kappa) -- what the FITTED observer still gets wrong. If the model's
    residual were 'the rung the fitted observer has not climbed', the two should have the
    same shape across levels."""
    from rhm.logit_reading.flat_oracle import leaf_levels
    from rhm.logit_reading.altitude.identity import convex_family
    z_ = np.load(npz_path)
    z = z_["logits"].astype(np.float64)
    preds = z_["preds"].astype(np.float64)
    phase = z_["phase"]
    n, T, v = z.shape
    logq = _log_softmax(z)
    p = preds[..., L, :]
    p_kap = convex_family(preds, kappa, L)
    lev = leaf_levels((phase[:, None] + 1 + np.arange(T)[None, :]) % (s ** L), s, L)
    KL_m = _kl(p, logq)
    KL_o = _kl(p, np.log(np.clip(p_kap, EPS, None)))
    out = {"kappa": float(kappa), "KL_model_total": float(KL_m.mean()),
           "KL_obs_total": float(KL_o.mean()), "by_level": {}}
    for l in range(L + 1):
        m = lev == l
        if m.sum() < 50:
            continue
        out["by_level"][int(l)] = {
            "frac_positions": float(m.mean()),
            "KL_model": float(KL_m[m].mean()), "KL_obs": float(KL_o[m].mean()),
            "share_model": float(KL_m[m].sum() / KL_m.sum()),
            "share_obs": float(KL_o[m].sum() / KL_o.sum()),
            "obs_over_model": float(KL_o[m].mean() / max(KL_m[m].mean(), 1e-12)),
        }
    return out


@app.function(volumes={DATA_DIR: volume}, timeout=3 * 3600, memory=16384, cpu=4.0)
def residual_sweep(traj_dir: str, identity_name: str = "altitude_identity",
                   out_name: str = "altitude_residual"):
    import glob
    import os
    import resource
    volume.reload()
    with open(f"{traj_dir}/{identity_name}.json") as f:
        kap = {r["step"]: r["altitude"]["convex"]["kappa_star"] for r in json.load(f)}
    rows = []
    for pth in sorted(glob.glob(f"{traj_dir}/step*_calibration.npz")):
        step = int(os.path.basename(pth)[4:10])
        r = residual_ckpt(pth, kap[step])
        r["step"] = step
        rows.append(r)
        print(f"step {step:>6}  kappa {r['kappa']:.2f}  KL(p_L||q) {r['KL_model_total']:.4f}  "
              f"KL(p_L||p_kappa) {r['KL_obs_total']:.4f}  share_model "
              + " ".join(f"{b['share_model']:.3f}" for b in r["by_level"].values())
              + "  share_obs " + " ".join(f"{b['share_obs']:.3f}" for b in r["by_level"].values()),
              flush=True)
    with open(f"{traj_dir}/{out_name}.json", "w") as f:
        json.dump(rows, f, cls=NumpyEncoder, indent=1)
    volume.commit()
    print(f"peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.1f} GB", flush=True)
    return f"{traj_dir}/{out_name}.json"
