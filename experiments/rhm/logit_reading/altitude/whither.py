"""Q4b, sharpened: after a violation, what does the forecast move TOWARD?

The same-prefix pairs of `phasic.phasic_readouts` (rebuilt here exactly, on CPU) give, for
each violation, a window ending in the illegal token and a twin ending in a legal one at
matched model surprisal. The model's post-token forecast is *sharper* after the illegal
token than after the twin, which is the wrong direction for a Bayesian. This asks what the
sharpened forecast resembles: for every observer k, in both families, it computes

    KL(q_next || p_{k,next})    after the violation, and after the twin

where q_next is the model's forecast at t_v (predicting token t_v + 1) and p_{k,next} is
observer k's predictive at the same position given the same window. The argmin over k is
the observer the model's post-token forecast is closest to; comparing it to the model's
own fitted altitude kappa* says whether a violation makes the model fall back to a coarser
reading, adopt the eps-noise reading, or neither.

A CLEAN observer with k >= k* has no posterior at all after the violating token (the
prefix is impossible for it), so its KL is undefined; the fraction of pairs where each
observer is defined is reported rather than dropped. The eps-noise family is always
defined -- that is what it is for.

CPU only. Run:
  modal run --detach -m rhm.logit_reading.altitude.whither::whither_sweep \
      --ckpts /data/.../traj_a1_s42/step064000.pt,/data/.../traj_eps01_s42/step064000.pt
"""

import json

import numpy as np

from rhm.shared import volume, DATA_DIR, NumpyEncoder
from rhm.logit_reading.calibration import app, load_trajectory_ckpt
from rhm.logit_reading.violation import _load_stimuli, _ent
from rhm.logit_reading.phasic import balance_tokens, balance_ds_sign

EPS = 1e-300


def _logits_at(model, windows, t, device="cpu", bs=256):
    """log q at positions t-1 and t; the two arrays `phasic._forward_at` also returns."""
    import torch
    T = model.block_size
    n = len(windows)
    lq_prev = np.empty((n, 16), np.float32)
    lq_next = np.empty((n, 16), np.float32)
    with torch.no_grad():
        for c0 in range(0, n, bs):
            x = torch.as_tensor(windows[c0:c0 + bs, :T], device=device)
            tt = torch.as_tensor(t[c0:c0 + bs], device=device)
            ar = torch.arange(len(x), device=device)
            lg, _ = model(x)
            lsm = torch.log_softmax(lg.float(), -1)
            lq_prev[c0:c0 + bs] = lsm[ar, tt - 1].cpu().numpy()
            lq_next[c0:c0 + bs] = lsm[ar, tt].cpu().numpy()
    return lq_prev, lq_next


def test_pairs(model, S, seed=0, caliper=0.3, t_lo=16, device="cpu", max_windows=0):
    """`phasic.phasic_readouts`'s pair construction, restricted to its TEST split.

    `max_windows` subsamples the candidate violations before the forward pass. The twin for
    a window depends only on that window, so this is the same construction on a random
    subsample; only the two balancing steps see a smaller pool. Needed because this runs on
    CPU."""
    We = S["windows_edit"]
    T = We.shape[1] - 1
    et, tv, ks, pL = S["etype"], S["t_v"], S["k_star"], S["pL_edit"]
    vw = np.where((et == 0) & (tv >= t_lo) & (tv <= T - 1))[0]
    if max_windows and len(vw) > max_windows:
        vw = np.sort(np.random.default_rng(seed + 7).choice(vw, max_windows, replace=False))
    t = tv[vw]
    lqp_v, lqn_v = _logits_at(model, We[vw], t, device)
    x_v = We[vw, t]
    s_all = -lqp_v
    s_v = s_all[np.arange(len(vw)), x_v]
    legal = pL[vw, t] > 0
    dist = np.where(legal, np.abs(s_all - s_v[:, None]), np.inf)
    x_c = dist.argmin(1)
    has = dist.min(1) <= caliper
    P = np.where(has)[0]
    twin = We[vw[P]].copy()
    twin[np.arange(len(P)), t[P]] = x_c[P]
    lqp_c, lqn_c = _logits_at(model, twin, t[P], device)
    surpr_v = -lqp_v[P][np.arange(len(P)), x_v[P]]
    surpr_c = -lqp_c[np.arange(len(P)), x_c[P]]
    rng = np.random.default_rng(seed + 2)
    split = np.random.default_rng(seed + 1).random(len(P)) < 0.6
    keep = np.zeros(len(P), bool)
    for flag in (split, ~split):
        sub = balance_ds_sign(surpr_v - surpr_c, np.where(flag)[0], rng)
        if len(sub):
            keep[balance_tokens(x_v[P], x_c[P], sub, rng)] = True
    te = (~split) & keep
    return {"n_candidates": int(len(vw)), "n_matched": int(has.sum()), "n_test": int(te.sum()),
            "t": t[P][te], "kstar": ks[vw[P]][te],
            "viol_win": We[vw[P]][te], "twin_win": twin[te],
            "q_next_viol": np.exp(lqn_v[P][te].astype(np.float64)),
            "q_next_twin": np.exp(lqn_c[te].astype(np.float64))}


def _kl_q_p(q, p):
    """KL(q || p) per row; NaN wherever p is not a distribution (undefined observer)."""
    ok = np.isfinite(p).all(-1) & (p.sum(-1) > 0.5)
    out = np.full(len(q), np.nan)
    qq, pp = q[ok], p[ok]
    out[ok] = (qq * (np.log(np.clip(qq, EPS, None)) - np.log(np.clip(pp, EPS, None)))).sum(-1)
    return out


@app.function(volumes={DATA_DIR: volume}, timeout=6 * 3600, memory=16384, cpu=16.0)
def whither_ckpt(ckpt: str, stim_tag: str = "swap65k", seed: int = 0, caliper: float = 0.3,
                 ks: str = "1,2,3,4,5,6", noise_eps: float = 0.01, max_pairs: int = 600,
                 chunk: int = 8, max_windows: int = 8000):
    import time
    from rhm.logit_reading.flat_oracle import flat_predictive
    volume.reload()
    model, rules, rule_w, cfg = load_trajectory_ckpt(ckpt, "cpu")
    S = _load_stimuli(stim_tag)
    t0 = time.time()
    pr = test_pairs(model, S, seed, caliper, device="cpu", max_windows=max_windows)
    print(f"pairs: candidates {pr['n_candidates']}  matched {pr['n_matched']}  "
          f"test {pr['n_test']}  ({time.time() - t0:.0f}s)",
          flush=True)
    n = pr["n_test"]
    res = {"ckpt": ckpt, "step": cfg.get("step"), "eps_train": cfg.get("eps_train", 0.0),
           "caliper": caliper, "noise_eps": noise_eps, "n_matched": pr["n_matched"],
           "n_test": n, "n_candidates": pr["n_candidates"], "max_windows": max_windows}
    if n < 50:
        return res
    sub = np.random.default_rng(0).permutation(n)[:max_pairs]
    ksub = pr["kstar"][sub]
    res["n_scored"] = int(len(sub))
    res["kstar_hist"] = np.bincount(ksub[ksub >= 0], minlength=7).tolist()
    wins = np.concatenate([pr["viol_win"][sub], pr["twin_win"][sub]])
    tt = np.concatenate([pr["t"][sub], pr["t"][sub]])
    q = np.concatenate([pr["q_next_viol"][sub], pr["q_next_twin"][sub]])
    m = len(sub)
    res["H_next"] = {"viol": float(_ent(q[:m]).mean()), "twin": float(_ent(q[m:]).mean())}
    res["families"] = {}
    for fam, eo in (("clean", 0.0), ("eps", noise_eps)):
        res["families"][fam] = {}
        for k in [int(x) for x in ks.split(",")]:
            t1 = time.time()
            p = flat_predictive(wins, rules, k, device="cpu",
                                chunk=chunk if k >= 6 else max(chunk, 32), rule_w=rule_w,
                                noise_eps=eo).numpy()
            pk = p[np.arange(len(wins)), tt + 1]
            d = _kl_q_p(q, pk)
            dv, dt = d[:m], d[m:]
            res["families"][fam][k] = {
                "defined_viol": float(np.isfinite(dv).mean()),
                "defined_twin": float(np.isfinite(dt).mean()),
                "KL_viol": float(np.nanmean(dv)), "KL_twin": float(np.nanmean(dt)),
                "H_obs_viol": float(np.nanmean(np.where(np.isfinite(dv), _ent(pk[:m]), np.nan))),
                "by_kstar": {int(kk): {"n": int((ksub == kk).sum()),
                                       "KL_viol": float(np.nanmean(dv[ksub == kk])),
                                       "KL_twin": float(np.nanmean(dt[ksub == kk])),
                                       "defined_viol": float(np.isfinite(dv[ksub == kk]).mean())}
                             for kk in range(1, 7) if (ksub == kk).sum() >= 20},
                "secs": round(time.time() - t1, 1),
            }
            r = res["families"][fam][k]
            print(f"  {fam} k={k}: KL(q||p_k) viol {r['KL_viol']:.4f} twin {r['KL_twin']:.4f}  "
                  f"defined {r['defined_viol']:.2f}/{r['defined_twin']:.2f}  ({r['secs']}s)",
                  flush=True)
    # which observer is the post-token forecast closest to, per pair
    for fam in res["families"]:
        kk = sorted(res["families"][fam])
        for side in ("viol", "twin"):
            vals = [res["families"][fam][k][f"KL_{side}"] for k in kk]
            if np.isfinite(vals).any():
                res["families"][fam][f"argmin_k_{side}"] = int(kk[int(np.nanargmin(vals))])
            res["families"][fam][f"argmin_k_{side}_by_kstar"] = {
                int(s_): int(min(kk, key=lambda k: res["families"][fam][k]["by_kstar"][str(s_)]
                                 [f"KL_{side}"]))
                for s_ in range(1, 7)
                if all(str(s_) in res["families"][fam][k]["by_kstar"] for k in kk)}
    with open(f"{ckpt[:-3]}_whither_{stim_tag}.json", "w") as f:
        json.dump(res, f, cls=NumpyEncoder, indent=1)
    volume.commit()
    return {"step": cfg.get("step"), "ckpt": ckpt}


@app.function(volumes={DATA_DIR: volume}, timeout=6 * 3600, memory=4096)
def whither_sweep(ckpts: str, stim_tag: str = "swap65k", ks: str = "1,2,3,4,5,6",
                  max_pairs: int = 600, max_windows: int = 8000):
    return list(whither_ckpt.map(ckpts.split(","),
                                 kwargs={"stim_tag": stim_tag, "ks": ks,
                                         "max_pairs": max_pairs, "max_windows": max_windows}))
