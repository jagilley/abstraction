"""Q5 readout: Part 1 against BOTH observer families, clean and eps-noise.

A model trained on a stream corrupted at rate eps (`train_noisy.py`) is fit to the
eps-noise predictive, not the clean one. `flat_oracle.flat_predictive(..., noise_eps=eps)`
is exactly that observer at every coarseness k, so "which family is the model the posterior
of" becomes a fit over 14 observers: p_k (clean) and p_k^eps, k = 0..6.

Two window sets, both from the same clean draw (eval_seed 4242, the set Part 1 already
uses):
  clean        the clean windows
  noisy<eps>   the same windows with every token independently replaced by a uniform draw
               with probability eps -- the distribution `train_noisy` actually trains on

The observer predictives are model-independent, so `ladder_cache` computes each
(window set, family) once and every checkpoint of every trajectory reads them.

Run:
  modal run --detach -m rhm.logit_reading.altitude.dual_ladder::ladder_cache_sweep
  modal run --detach -m rhm.logit_reading.altitude.dual_ladder::dual_sweep \
      --traj-dir /data/v16_s2_L6_m4_distinct/logit_reading/traj_eps01_s42 --ws noisy0.01
"""

import json

import numpy as np

from rhm.shared import volume, DATA_DIR, NumpyEncoder
from rhm.logit_reading.calibration import app, load_trajectory_ckpt, _reliability, _r2
from rhm.logit_reading.altitude.identity import (_log_softmax, _ent, _xent, _kl, EPS,
                                                 convex_family, fit_kappa)

KEY = "v16_s2_L6_m4_distinct"


def make_windows(rules, rule_w, n, T1, seed, eps_data, noise_seed=777):
    from rhm.logit_reading.flat_oracle import sample_flat_windows
    w, phase = sample_flat_windows(rules, n, T1, seed, rule_w=rule_w)
    if eps_data > 0:
        rng = np.random.default_rng(noise_seed)
        v = rules[0].shape[0]
        flip = rng.random(w.shape) < eps_data
        w = np.where(flip, rng.integers(0, v, w.shape), w)
    return w, phase


@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=4 * 3600, memory=16384,
              max_containers=4)
def ladder_cache(spec: str, v: int = 16, s: int = 2, depth: int = 6, m: int = 4,
                 rule_seed: int = 0, alpha: float = 1.0, weight_seed: int = 1,
                 n_windows: int = 4096, eval_seed: int = 4242, chunk: int = 8):
    """spec = "<eps_data>:<eps_obs>", e.g. "0:0" (clean windows, clean ladder)."""
    import time
    from rhm.rhm_data import generate_rules_distinct
    from rhm.logit_reading.grammar import synonym_weights
    from rhm.logit_reading.flat_oracle import flat_predictive
    volume.reload()
    eps_data, eps_obs = (float(x) for x in spec.split(":"))
    L, T = depth, s ** depth
    rules = generate_rules_distinct(v, s, L, m, seed=rule_seed)
    rule_w = None if alpha <= 0 else synonym_weights(v, L, m, alpha, weight_seed)
    windows, phase = make_windows(rules, rule_w, n_windows, T + 1, eval_seed, eps_data)
    preds = np.empty((n_windows, T, L + 1, v), np.float32)
    for k in range(L + 1):
        t0 = time.time()
        pr = flat_predictive(windows, rules, k, device="cuda",
                             chunk=chunk if k == L else max(chunk, 64), rule_w=rule_w,
                             noise_eps=eps_obs)
        preds[:, :, k, :] = pr.numpy()[:, 1:, :]
        print(f"  eps_data={eps_data} eps_obs={eps_obs} k={k} {time.time() - t0:.0f}s", flush=True)
    out = f"{DATA_DIR}/{KEY}/logit_reading/ladder_d{eps_data}_o{eps_obs}.npz"
    np.savez_compressed(out, preds=preds, windows=windows, phase=phase)
    volume.commit()
    return out


@app.function(volumes={DATA_DIR: volume}, timeout=6 * 3600, memory=4096)
def ladder_cache_sweep(specs: str = "0:0,0:0.01,0.01:0,0.01:0.01"):
    return list(ladder_cache.map(specs.split(",")))


def dual_summary(logits, ladders, windows, L=6):
    """logits (n, T, v); ladders: {"clean": preds, "eps": preds} -> flat stats dict."""
    z = logits.astype(np.float64)
    logq = _log_softmax(z)
    q = np.exp(logq)
    tok = windows[:, 1:]
    H_q = _ent(q)
    out = {"H_q": float(H_q.mean())}
    grid = np.round(np.arange(0.0, L + 1e-9, 0.05), 4)
    for name, preds in ladders.items():
        pr = preds.astype(np.float64)
        # a CLEAN observer is undefined once the window contains a corrupted token (its
        # phase posterior is -inf everywhere); score it only where it is defined, and say
        # what fraction that is.
        ok = np.isfinite(pr).all(-1).all(-1)
        if not ok.all():
            pr = pr[ok][:, None, :, :]
            lq = logq[ok][:, None, :]
            tk = tok[ok][:, None]
        else:
            lq, tk = logq, tok
        p = pr[..., L, :]
        Hq_ = _ent(np.exp(lq))
        CE = _xent(p, lq)
        gap = CE - Hq_
        KL_k = np.stack([_kl(pr[..., k, :], lq) for k in range(L + 1)], -1)
        mk = [float(KL_k[..., k].mean()) for k in range(L + 1)]
        cx, _ = fit_kappa(pr, lq, L, convex_family, grid)
        nll = -np.take_along_axis(lq, tk[..., None], -1)[..., 0]
        surpr = -np.log(np.clip(np.take_along_axis(p, tk[..., None], -1)[..., 0], EPS, None))
        out[name] = {
            "defined_frac": float(ok.mean()),
            "H_p": float(_ent(p).mean()), "CE": float(CE.mean()),
            "KL_pL_q": float((CE - _ent(p)).mean()), "gap": float(gap.mean()),
            "R2_CE_on_Hq": _r2(CE, Hq_), "KL_k_q": mk,
            "best_k": int(np.argmin(mk)), "KL_best": float(min(mk)),
            "kappa_star": cx["kappa_star"], "KL_at_kappa": cx["KL_at_star"],
            "kappa_curve": cx["curve"],
            "selfcheck_nll_minus_CE": float(nll.mean() - CE.mean()),
            "selfcheck_surpr_minus_Hp": float(surpr.mean() - _ent(p).mean()),
        }
    conf = q.max(-1)
    out["ece_realised"] = _reliability(conf.ravel(), (q.argmax(-1) == tok).astype(float).ravel(), 15)[0]
    out["winner"] = min((nm for nm in ("clean", "eps") if out[nm]["defined_frac"] > 0.99),
                        key=lambda nm: out[nm]["KL_at_kappa"], default="undefined")
    return out


@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=2 * 3600, memory=32768,
              max_containers=4)
def dual_ckpt(ckpt: str, ws: str = "noisy0.01", eps_obs: float = 0.01):
    import torch
    volume.reload()
    eps_data = 0.0 if ws == "clean" else float(ws.replace("noisy", ""))
    model, rules, rule_w, cfg = load_trajectory_ckpt(ckpt, "cuda")
    lad, windows = {}, None
    for nm, eo in (("clean", 0.0), ("eps", eps_obs)):
        z = np.load(f"{DATA_DIR}/{KEY}/logit_reading/ladder_d{eps_data}_o{eo}.npz")
        lad[nm] = z["preds"]
        windows = z["windows"] if windows is None else windows
        assert (z["windows"] == windows).all()
    T = windows.shape[1] - 1
    with torch.no_grad():
        X = torch.as_tensor(windows[:, :T], device="cuda")
        logits = torch.cat([model(X[i:i + 512])[0].float().cpu()
                            for i in range(0, len(windows), 512)]).numpy()
    res = dual_summary(logits, lad, windows)
    res["config"] = {"ckpt": ckpt, "step": cfg.get("step"), "ws": ws, "eps_obs": eps_obs,
                     "eps_train": cfg.get("eps_train", 0.0)}
    print(f"step {cfg.get('step'):>6} [{ws}]  clean: k*={res['clean']['best_k']} "
          f"kappa {res['clean']['kappa_star']:.2f} KL {res['clean']['KL_at_kappa']:.4f} | "
          f"eps: k*={res['eps']['best_k']} kappa {res['eps']['kappa_star']:.2f} "
          f"KL {res['eps']['KL_at_kappa']:.4f} | winner {res['winner']}  "
          f"gap {res['clean']['gap']:+.4f}", flush=True)
    with open(f"{ckpt[:-3]}_dual_{ws}.json", "w") as f:
        json.dump(res, f, cls=NumpyEncoder, indent=1)
    volume.commit()
    return {"step": cfg.get("step")}


@app.function(volumes={DATA_DIR: volume}, timeout=6 * 3600, memory=4096)
def dual_sweep(traj_dir: str, ws: str = "noisy0.01", eps_obs: float = 0.01):
    import glob
    volume.reload()
    ckpts = sorted(glob.glob(f"{traj_dir}/step*.pt"))
    print(f"{len(ckpts)} checkpoints in {traj_dir} [{ws}]", flush=True)
    return len(list(dual_ckpt.map(ckpts, kwargs={"ws": ws, "eps_obs": eps_obs})))
