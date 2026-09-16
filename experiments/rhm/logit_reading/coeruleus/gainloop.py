"""Q3: close the gain loop -- the head's output sets the temperature at the position it reads.

Every arm has the same shape: a per-position gate g_t, and a monotone map alpha = f(g) fit on a
held-out FIT set of corrupted windows by the model's own realised NLL (no oracle in the fit).
The arms differ only in the gate:

  const         g = 0                     one temperature everywhere (the recalibration control)
  head          g = the Q2 head's output  the closed loop
  oracle_probe  g = a probe on the same states trained WITH the corruption label
  oracle_offset g = the offset tau since the last corrupted token (categorical, oracle gate)

`f` is a 12-bin step function, each bin's alpha the exact minimiser of the mean NLL in that bin
(bisection on d/dalpha NLL = <q_alpha - onehot(x), z>, which is increasing). Monotonicity in the
gate is reported, not imposed; a log-linear fit is reported beside it.

Evaluation is on the CACHED eval-seed-4242 corrupted window set, where the eps-observer ladder
is on file, so the same ceiling as question 1 applies and the numbers are comparable:

  * per-offset excess against the ceiling, with and without the loop; fraction of Q1's prize
  * the cost at quiet positions and over the whole stream
  * the parent's 2e entropy sign on the `swap65k` same-prefix pairs, with the loop's temperature
    applied at the position that reads the violating token

Run:
  modal run -m rhm.logit_reading.coeruleus.gainloop::gain_ckpt \
      --ckpt /data/.../traj_eps01_s42/step064000.pt
"""

import json

import numpy as np

from rhm.shared import volume, DATA_DIR, NumpyEncoder
from rhm.logit_reading.calibration import app, load_trajectory_ckpt
from rhm.logit_reading.violation import auc, fit_probe, _load_stimuli
from rhm.logit_reading.phasic import paired_win
from rhm.logit_reading.coeruleus.events import (
    corrupted_windows, assert_cache_matches, load_ladder, events_from_mask, quiet_positions,
    log_softmax, ent, xent, kl, convex_obs, fit_kappa_pooled, _bisect)
from rhm.logit_reading.coeruleus.readout import (
    load_head, states_at, states_at_t, logq_at, prefix_pairs)


def raw_logits(model, windows, T, bs=256):
    import torch
    out = []
    with torch.no_grad():
        for i in range(0, len(windows), bs):
            x = torch.as_tensor(windows[i:i + bs, :T], device="cuda")
            out.append(model(x)[0].float().cpu().numpy())
    return np.concatenate(out).astype(np.float64)


# ---------------------------------------------------------------------------
# the monotone map
# ---------------------------------------------------------------------------

def _alpha_min_nll(z, tok, lo=0.05, hi=8.0):
    """alpha minimising mean NLL of the realised token. d/dalpha = <q_alpha - e_x, z>."""
    z = np.asarray(z, np.float64)
    oh = np.zeros_like(z)
    oh[np.arange(len(z)), tok] = 1.0

    def d(a):
        q = np.exp(log_softmax(np.asarray(a)[..., None] * z))
        return np.full((), ((q - oh) * z).sum(-1).mean())
    return float(_bisect(d, lo, hi, ()))


class BinMap:
    """alpha = f(gate), a step function on quantile bins of the gate."""

    def __init__(self, edges, alphas):
        self.edges, self.alphas = np.asarray(edges), np.asarray(alphas)

    def __call__(self, g):
        return self.alphas[np.clip(np.digitize(np.asarray(g), self.edges), 0,
                                   len(self.alphas) - 1)]

    def monotone(self):
        d = np.diff(self.alphas)
        return {"n_bins": len(self.alphas), "increasing": bool((d >= -1e-3).all()),
                "decreasing": bool((d <= 1e-3).all()),
                "spearman_bin_vs_alpha": float(np.corrcoef(np.arange(len(self.alphas)),
                                                           self.alphas)[0, 1]),
                "alphas": self.alphas.round(4).tolist(),
                "bin_counts": getattr(self, "ns", None), "quantiles": getattr(self, "qs", None)}


# Equal-quantile bins cannot isolate the subpopulation that matters: a corrupted token is in
# context at ~1.5% of positions, so every bin of a 12-way equal split is >85% quiet and its
# optimal temperature is ~1 whatever the gate reads. The grid is refined in both tails.
QS = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.98, 0.99,
      0.995, 0.998, 0.999]


def fit_binmap(g, z, tok, n_bins=0, qs=None, min_n=200):
    """alpha per quantile bin of the gate, each the exact NLL minimiser in that bin."""
    g = np.asarray(g, np.float64)
    qs = (list(np.linspace(0, 1, n_bins + 1)[1:-1]) if n_bins else (qs or QS))
    edges = np.unique(np.quantile(g, qs))
    b = np.clip(np.digitize(g, edges), 0, len(edges))
    alphas = np.ones(len(edges) + 1)
    ns = np.zeros(len(edges) + 1, int)
    for i in range(len(edges) + 1):
        sel = b == i
        ns[i] = int(sel.sum())
        if sel.sum() >= min_n:
            alphas[i] = _alpha_min_nll(z[sel], tok[sel])
    m = BinMap(edges, alphas)
    m.ns = ns.tolist()
    m.qs = qs
    return m


def fit_loglinear(g, z, tok, steps=200, lr=0.05, max_rows=600000, seed=0):
    """log alpha = a + b * gn, fit by mean NLL (Adam on the exact gradient)."""
    g = np.asarray(g, np.float64)
    gmu, gsd = float(g.mean()), float(g.std() + 1e-9)
    if max_rows and len(g) > max_rows:
        sel = np.random.default_rng(seed).permutation(len(g))[:max_rows]
        g, z, tok = g[sel], np.asarray(z)[sel], np.asarray(tok)[sel]
    gn = (g - gmu) / gsd
    z = np.asarray(z, np.float64)
    oh = np.zeros_like(z)
    oh[np.arange(len(z)), tok] = 1.0
    a, bb = 0.0, 0.0
    m, vv = np.zeros(2), np.zeros(2)
    b1, b2, e = 0.9, 0.999, 1e-8
    for t in range(1, steps + 1):
        al = np.exp(a + bb * gn)
        q = np.exp(log_softmax(al[:, None] * z))
        core = ((q - oh) * z).sum(-1) * al
        grad = np.array([core.mean(), (core * gn).mean()])
        m = b1 * m + (1 - b1) * grad
        vv = b2 * vv + (1 - b2) * grad ** 2
        step = lr * (m / (1 - b1 ** t)) / (np.sqrt(vv / (1 - b2 ** t)) + e)
        a, bb = a - step[0], bb - step[1]
    return {"a": float(a), "b": float(bb), "gmu": gmu, "gsd": gsd}


def apply_loglinear(par, g):
    return np.exp(par["a"] + par["b"] * (np.asarray(g) - par["gmu"]) / par["gsd"])


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------

def arm_stats(z, alpha, pE, logceil, tok, win, c, qw, qt, H):
    lq = log_softmax(np.asarray(alpha)[..., None] * z)
    nll = -np.take_along_axis(lq, tok[..., None], -1)[..., 0]
    out = {"alpha_mean": float(np.mean(alpha)), "alpha_sd": float(np.std(alpha)),
           "CE_all": float(xent(pE, lq).mean()), "nll_all": float(nll.mean()),
           "H_q_all": float(ent(np.exp(lq)).mean()),
           "excess_ceil_all": float((xent(pE, lq) - xent(pE, logceil)).mean())}
    W = np.repeat(win, H + 1)
    Tt = (c[:, None] + np.arange(H + 1)[None, :]).ravel()
    out["event_CE"] = float(xent(pE[W, Tt], lq[W, Tt]).mean())
    out["event_excess_ceil"] = float((xent(pE[W, Tt], lq[W, Tt])
                                      - xent(pE[W, Tt], logceil[W, Tt])).mean())
    out["event_nll"] = float(nll[W, Tt].mean())
    out["quiet_CE"] = float(xent(pE[qw, qt], lq[qw, qt]).mean())
    out["quiet_excess_ceil"] = float((xent(pE[qw, qt], lq[qw, qt])
                                      - xent(pE[qw, qt], logceil[qw, qt])).mean())
    out["quiet_nll"] = float(nll[qw, qt].mean())
    out["quiet_alpha_mean"] = float(np.mean(np.asarray(alpha)[qw, qt])) \
        if np.ndim(alpha) == 2 else float(np.mean(alpha))
    out["by_tau"] = {}
    for tau in range(-1, H + 1):
        t_ = c + tau
        ok = t_ >= 0
        w_, tt_ = win[ok], t_[ok]
        out["by_tau"][tau] = {
            "excess_ceil": float((xent(pE[w_, tt_], lq[w_, tt_])
                                  - xent(pE[w_, tt_], logceil[w_, tt_])).mean()),
            "H_q": float(ent(np.exp(lq[w_, tt_])).mean()),
            "alpha": float(np.mean(np.asarray(alpha)[w_, tt_])) if np.ndim(alpha) == 2
            else float(np.mean(alpha))}
    return out


@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=3 * 3600, memory=24576,
              max_containers=4)
def gain_ckpt(ckpt: str, head_path: str = "", eps_data: float = 0.01, horizon: int = 8,
              n_fit: int = 32768, fit_seed: int = 90213, n_bins: int = 12,
              eval_seed: int = 4242, n_windows: int = 4096, stim_tag: str = "swap65k",
              caliper: float = 0.3, max_windows: int = 0, seed: int = 0, t_lo: int = 8,
              tag: str = ""):
    import resource
    volume.reload()
    H = horizon
    model, rules, rule_w, cfg = load_trajectory_ckpt(ckpt, "cuda")
    specs = head_path or (f"ridge:{ckpt[:-3]}_coeruleus_head_state_excess_ridge.pt,"
                          f"mlp:{ckpt[:-3]}_coeruleus_head_state_excess_mlp.pt")
    hd, hcfgs = {}, {}
    for spec in specs.split(","):
        nm_, pth = spec.split(":", 1)
        hd[nm_], hcfgs[nm_] = load_head(pth, "cuda")
    blk = hd["ridge"].block
    T = cfg["s"] ** cfg["L"]
    res = {"ckpt": ckpt, "step": cfg.get("step"), "eps_train": cfg.get("eps_train", 0.0),
           "heads": {k: v for k, v in hcfgs.items()}, "block": blk, "H": H,
           "eps_data": eps_data}

    # ---------------- fit set (fresh windows, no oracle used in the fit) ----
    wf, _, cf, _ = corrupted_windows(rules, rule_w, n_fit, T + 1, fit_seed, eps_data,
                                     noise_seed=fit_seed + 31)
    zf = raw_logits(model, wf, T)
    SfB = {b: states_at(model, wf, b, "cuda") for b in sorted({h.block for h in hd.values()})}
    gf = {k: h(SfB[h.block].reshape(-1, SfB[h.block].shape[-1])).reshape(n_fit, T)
          for k, h in hd.items()}
    Sf = SfB[blk]
    tokf = wf[:, 1:]
    # oracle gates on the fit set
    corr_t = np.zeros((n_fit, T), bool)                      # token read at t is corrupted
    corr_t[:, :] = cf[:, :T]
    tau_since = np.full((n_fit, T), -1)
    last = np.full(n_fit, -10 ** 6)
    for t in range(T):
        last = np.where(cf[:, t], t, last)
        tau_since[:, t] = t - last
    tau_g = np.where(tau_since <= H, tau_since, H + 1)

    # the maps are fit only where they are evaluated (t >= t_lo). At window positions 0..7 a
    # corrupted token is far more costly -- the phase is still ambiguous and the context short --
    # so the optimal temperature there is much smaller, and pooling those positions in drags a
    # gate-indexed map away from the value it needs mid-window.
    fmask = np.broadcast_to(np.arange(T) >= t_lo, (n_fit, T)).ravel()
    zf_f, tok_f = zf.reshape(-1, 16)[fmask], tokf.ravel()[fmask]
    maps = {}
    res["fit"] = {"n_fit": n_fit, "t_lo": t_lo, "n_fit_rows": int(fmask.sum()),
                  "gate_stats": {k: [float(g_.mean()), float(g_.std())] for k, g_ in gf.items()}}
    a_const = _alpha_min_nll(zf_f, tok_f)
    maps["const"] = ("const", a_const)
    for k in hd:
        maps[f"head_{k}"] = ("bin", fit_binmap(gf[k].ravel()[fmask], zf_f, tok_f))
        res["fit"][f"head_{k}_map"] = maps[f"head_{k}"][1].monotone()
        res["fit"][f"head_{k}_loglinear"] = fit_loglinear(gf[k].ravel()[fmask], zf_f, tok_f)
    res["fit"]["const_alpha"] = a_const
    # supervised gate on the same states: "is the token I just read corrupted"
    perm = np.random.default_rng(seed).permutation(np.where(fmask)[0])
    ntr = min(400000, int(0.8 * len(perm)))
    idx, idx_te = perm[:ntr], perm[ntr:ntr + 100000]
    pr = fit_probe(Sf.reshape(-1, Sf.shape[-1])[idx], corr_t.ravel()[idx], hidden=256, steps=1200)
    og = pr(Sf.reshape(-1, Sf.shape[-1])).reshape(n_fit, T)
    og_f = og.ravel()[fmask]
    res["fit"]["oracle_probe_auc"] = auc(og.ravel()[idx_te], corr_t.ravel()[idx_te])
    maps["oracle_probe"] = ("bin", fit_binmap(og_f, zf_f, tok_f))
    res["fit"]["oracle_probe_map"] = maps["oracle_probe"][1].monotone()
    # categorical oracle gate: alpha per offset since the last corruption
    a_off = {}
    tau_f = tau_g.ravel()[fmask]
    for tt in range(0, H + 2):
        sel = tau_f == tt
        a_off[tt] = _alpha_min_nll(zf_f[sel], tok_f[sel]) if sel.sum() >= 50 else 1.0
    maps["oracle_offset"] = ("offset", a_off)
    res["fit"]["oracle_offset_alpha"] = {int(k): float(v) for k, v in a_off.items()}
    print(f"  const alpha {a_const:.4f}", flush=True)
    for k in hd:
        print(f"  head_{k} bins {res['fit'][f'head_{k}_map']['alphas']}", flush=True)
    print(f"  oracle probe AUC {res['fit']['oracle_probe_auc']:.3f}  "
          f"offset alphas {[round(a_off[t], 3) for t in sorted(a_off)]}", flush=True)
    del Sf, SfB, zf, gf, zf_f

    # ---------------- test set = the cached ladder windows ------------------
    w, _, corrupt, _ = assert_cache_matches(rules, rule_w, eps_data, n_windows, eval_seed)
    pe, wl, _ = load_ladder(eps_data, eps_data)
    assert (wl == w).all()
    pe64 = pe.astype(np.float64)
    pE = pe64[..., 6, :]
    z = raw_logits(model, w, T)
    logq0 = log_softmax(z)
    kap, klk = fit_kappa_pooled(pe64, logq0, 6)
    ceil = convex_obs(pe64, kap, 6)
    logceil = np.log(np.clip(ceil, 1e-300, None))
    res["kappa_star_eps"] = kap
    StB = {b: states_at(model, w, b, "cuda") for b in sorted({h.block for h in hd.values()})}
    g = {k: h(StB[h.block].reshape(-1, StB[h.block].shape[-1])).reshape(n_windows, T)
         for k, h in hd.items()}
    ogt = pr(StB[blk].reshape(-1, StB[blk].shape[-1])).reshape(n_windows, T)
    del StB
    tau_t = np.full((n_windows, T), H + 1)
    last = np.full(n_windows, -10 ** 6)
    for t in range(T):
        last = np.where(corrupt[:, t], t, last)
        d_ = t - last
        tau_t[:, t] = np.where(d_ <= H, d_, H + 1)

    win, c = events_from_mask(corrupt, H, c_lo=8, isolate=False)
    win_i, c_i = events_from_mask(corrupt, H, c_lo=8, isolate=True)
    qw, qt = quiet_positions(corrupt, H, t_lo=8, rng=np.random.default_rng(seed))
    tok = w[:, 1:]
    res["n_events"], res["n_events_isolated"] = int(len(win)), int(len(win_i))
    res["n_quiet"] = int(len(qw))

    alphas = {
        "base": np.ones((n_windows, T)),
        "const": np.full((n_windows, T), a_const),
        "oracle_probe": maps["oracle_probe"][1](ogt),
        "oracle_offset": np.array([a_off.get(t_, 1.0)
                                   for t_ in range(H + 2)])[np.clip(tau_t, 0, H + 1)],
    }
    for k in hd:
        alphas[f"head_{k}"] = maps[f"head_{k}"][1](g[k])
        alphas[f"head_{k}_loglin"] = apply_loglinear(res["fit"][f"head_{k}_loglinear"], g[k])
    res["arms"] = {nm: arm_stats(z, al, pE, logceil, tok, win, c, qw, qt, H)
                   for nm, al in alphas.items()}
    res["arms_isolated"] = {nm: arm_stats(z, al, pE, logceil, tok, win_i, c_i, qw, qt, H)
                            for nm, al in alphas.items()}
    b = res["arms"]["base"]
    ceil_ev = float(xent(pE[np.repeat(win, H + 1), (c[:, None] + np.arange(H + 1)).ravel()],
                         logceil[np.repeat(win, H + 1), (c[:, None] + np.arange(H + 1)).ravel()]).mean())
    prize = b["event_CE"] - ceil_ev
    res["prize_nats_per_pred"] = prize
    bi = res["arms_isolated"]["base"]
    ceil_i = float(xent(pE[np.repeat(win_i, H + 1), (c_i[:, None] + np.arange(H + 1)).ravel()],
                        logceil[np.repeat(win_i, H + 1),
                                (c_i[:, None] + np.arange(H + 1)).ravel()]).mean())
    prize_i = bi["event_CE"] - ceil_i
    res["prize_nats_per_pred_isolated"] = prize_i
    for nm, a_ in res["arms"].items():
        a_["frac_prize_recovered"] = float((b["event_CE"] - a_["event_CE"]) / prize) \
            if prize > 1e-9 else float("nan")
        a_["quiet_cost_nats"] = a_["quiet_CE"] - b["quiet_CE"]
        a_["stream_cost_nats"] = a_["CE_all"] - b["CE_all"]
    for nm, a_ in res["arms_isolated"].items():
        a_["frac_prize_recovered"] = float((bi["event_CE"] - a_["event_CE"]) / prize_i) \
            if prize_i > 1e-9 else float("nan")
        a_["quiet_cost_nats"] = a_["quiet_CE"] - bi["quiet_CE"]
        a_["stream_cost_nats"] = a_["CE_all"] - bi["CE_all"]
    print(f"prize all-events {prize:.4f} (n={len(win)})  isolated {prize_i:.4f} (n={len(win_i)})")
    print(f"{'arm':>18} {'evCE':>8} {'recov':>7} {'recovIso':>9} {'quietCost':>10} "
          f"{'streamCost':>11} {'aEv':>6}")
    for nm, a_ in res["arms"].items():
        print(f"{nm:>18} {a_['event_CE']:.4f} {a_['frac_prize_recovered']:+.3f} "
              f"{res['arms_isolated'][nm]['frac_prize_recovered']:+.3f} "
              f"{a_['quiet_cost_nats']:+.5f} {a_['stream_cost_nats']:+.5f} "
              f"{a_['by_tau'][0]['alpha']:.3f}", flush=True)

    # ---------------- the entropy sign on the same-prefix pairs -------------
    S = _load_stimuli(stim_tag)
    pp = prefix_pairs(model, S, "cuda", seed=seed, caliper=caliper, max_windows=max_windows)
    te, tstar = pp["te"], pp["t"]
    svB = {b: states_at_t(model, pp["viol_win"], tstar, b, "cuda")
           for b in sorted({h.block for h in hd.values()})}
    scB = {b: states_at_t(model, pp["twin_win"], tstar, b, "cuda")
           for b in sorted({h.block for h in hd.values()})}
    gv = {k: h(svB[h.block]) for k, h in hd.items()}
    gc = {k: h(scB[h.block]) for k, h in hd.items()}
    sv, sc = svB[blk], scB[blk]
    ov, oc = pr(sv), pr(sc)
    lqv = logq_at(model, pp["viol_win"], tstar, "cuda")
    lqc = logq_at(model, pp["twin_win"], tstar, "cuda")
    zv = lqv - lqv.mean(-1, keepdims=True)          # logits up to a constant: safe for softmax
    zc = lqc - lqc.mean(-1, keepdims=True)
    npair = len(ov)
    pair = {"n_test": int(te.sum()), "readouts": {},
            "gate_paired": {k: paired_win(gv[k][te], gc[k][te]) for k in hd}}
    pair["gate_paired"]["oracle_probe"] = paired_win(ov[te], oc[te])
    pa = {"base": (np.ones(npair), np.ones(npair)),
          "const": (np.full(npair, a_const), np.full(npair, a_const)),
          "oracle_probe": (maps["oracle_probe"][1](ov), maps["oracle_probe"][1](oc))}
    for k in hd:
        pa[f"head_{k}"] = (maps[f"head_{k}"][1](gv[k]), maps[f"head_{k}"][1](gc[k]))
        pa[f"head_{k}_loglin"] = (apply_loglinear(res["fit"][f"head_{k}_loglinear"], gv[k]),
                                  apply_loglinear(res["fit"][f"head_{k}_loglinear"], gc[k]))
    for nm, (av, ac) in pa.items():
        Hv = ent(np.exp(log_softmax(av[:, None] * zv)))
        Hc = ent(np.exp(log_softmax(ac[:, None] * zc)))
        pair["readouts"][nm] = {
            "H_next_paired": paired_win(Hv[te], Hc[te]),
            "H_next_auc": auc(np.concatenate([Hv[te], Hc[te]]),
                              np.concatenate([np.ones(int(te.sum()), bool),
                                              np.zeros(int(te.sum()), bool)])),
            "H_viol": float(Hv[te].mean()), "H_twin": float(Hc[te].mean()),
            "alpha_viol": float(av[te].mean()), "alpha_twin": float(ac[te].mean()),
            "by_kstar": {int(k): {"n": int((te & (pp["kstar"] == k)).sum()),
                                  "H_next_paired": paired_win(Hv[te & (pp["kstar"] == k)],
                                                              Hc[te & (pp["kstar"] == k)])}
                         for k in range(1, 7) if (te & (pp["kstar"] == k)).sum() >= 30}}
        print(f"  pairs {nm:16s} H_next paired {pair['readouts'][nm]['H_next_paired']:.3f}  "
              f"alpha v/t {pair['readouts'][nm]['alpha_viol']:.3f}/"
              f"{pair['readouts'][nm]['alpha_twin']:.3f}", flush=True)
    res["pairs"] = pair

    name = f"{ckpt[:-3]}_coeruleus_gain{('_' + tag) if tag else ''}.json"
    with open(name, "w") as f:
        json.dump(res, f, cls=NumpyEncoder, indent=1)
    volume.commit()
    print(f"peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.2f} GB -> {name}",
          flush=True)
    return {"ckpt": ckpt, "step": cfg.get("step")}
