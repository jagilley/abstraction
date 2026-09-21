"""Q1: bound the prize, open loop.

After a corrupted token the model falls back to a coarser reading and sharpens, where the
noise-aware Bayesian widens. This measures, per post-event offset tau = 0..H on the held-out
corrupted windows:

  excess_vs_truth(tau) = KL(p_L^eps || q)                     what the model loses against Bayes
  excess_vs_ceil(tau)  = CE(p_L^eps, q) - CE(p_L^eps, p_kappa*^eps)
                                                              what it loses against the best
                                                              member of its OWN family -- the
                                                              part that is not "it is coarse"

and then what a one-knob consumer reading the frozen logits could recover of `excess_vs_ceil`:

  temperature   q_alpha = softmax(alpha z),          alpha chosen per offset / per event / per
                                                     prediction (the last two with the oracle)
  noise floor   q_w     = (1 - w) q + w / v,         same three regimes

Both knobs are convex in their parameter and optimised exactly. The temperature is the knob
question 1 asks about; the floor is the shape an eps-aware Bayesian's response actually has and
is the natural second bound to have in hand before chasing a different consumer.

Run:
  modal run -m rhm.logit_reading.coeruleus.prize::prize_ckpt \
      --ckpt /data/v16_s2_L6_m4_distinct/logit_reading/traj_eps01_s42/step064000.pt
  modal run --detach -m rhm.logit_reading.coeruleus.prize::prize_sweep \
      --ckpts /data/.../traj_eps01_s42/step064000.pt,/data/.../traj_a1_s42/step064000.pt
"""

import json

import numpy as np

from rhm.shared import volume, DATA_DIR, NumpyEncoder
from rhm.logit_reading.calibration import app, load_trajectory_ckpt
from rhm.logit_reading.coeruleus.events import (
    build_rules, assert_cache_matches, load_ladder, events_from_mask, quiet_positions,
    log_softmax, ent, xent, kl, convex_obs, fit_kappa_pooled,
    temp_ce, temp_opt, floor_ce, floor_opt, mix_ce, mix_opt, affine_fit, affine_ce,
    model_logits, LR_DIR)


def _knob_block(z, logq, p, v=16):
    """Every one-knob number for a set of predictions (rows)."""
    ce1 = xent(p, logq)
    a_pool = float(temp_opt(z, p, pooled=True))
    w_pool = float(floor_opt(logq, p, pooled=True))
    a_row = temp_opt(z, p)
    w_row = floor_opt(logq, p)
    return {
        "n": int(len(z)),
        "CE": float(ce1.mean()),
        "H_q": float(ent(np.exp(logq)).mean()),
        "alpha_pooled": a_pool,
        "CE_alpha_pooled": float(temp_ce(z, p, a_pool).mean()),
        "alpha_row_mean": float(a_row.mean()), "alpha_row_median": float(np.median(a_row)),
        "CE_alpha_row": float(temp_ce(z, p, a_row).mean()),
        "w_pooled": w_pool,
        "CE_w_pooled": float(floor_ce(logq, p, w_pool, v).mean()),
        "w_row_mean": float(w_row.mean()), "w_row_median": float(np.median(w_row)),
        "CE_w_row": float(floor_ce(logq, p, w_row, v).mean()),
    }


def prize_stats(z, preds_eps, preds_clean, windows, corrupt, H=8, L=6, c_lo=8, seed=0,
                isolate=True):
    """z (n, T, v) logits; preds_* (n, T, L+1, v) observer ladders on the same windows."""
    n, T, v = z.shape
    logq = log_softmax(z)
    tok = windows[:, 1:]
    pe64 = preds_eps.astype(np.float64)
    pE = pe64[..., L, :]

    kap_e, kl_e = fit_kappa_pooled(pe64, logq, L)
    # the CLEAN family is undefined wherever the window already holds a corrupted token
    okc = np.isfinite(preds_clean).all(-1).all(-1) & (preds_clean.sum(-1) > 0.5).all(-1)
    kap_c, kl_c = (fit_kappa_pooled(preds_clean[okc].astype(np.float64)[:, None],
                                    logq[okc][:, None], L) if okc.any() else (float("nan"),) * 2)
    ceil = convex_obs(pe64, kap_e, L)
    logceil = np.log(np.clip(ceil, 1e-300, None))

    res = {"H": H, "kappa_star_eps": kap_e, "KL_at_kappa_eps": kl_e,
           "kappa_star_clean": kap_c, "KL_at_kappa_clean": kl_c,
           "clean_defined_frac": float(okc.mean()),
           "n_windows": int(n), "corrupt_rate": float(corrupt.mean())}
    # whole-stream reference
    res["overall"] = {
        "CE_model": float(xent(pE, logq).mean()), "H_truth": float(ent(pE).mean()),
        "CE_ceil": float(xent(pE, logceil).mean()),
        "excess_truth": float(kl(pE, logq).mean()),
        "excess_ceil": float((xent(pE, logq) - xent(pE, logceil)).mean()),
        "H_q": float(ent(np.exp(logq)).mean()),
        "nll": float(-np.take_along_axis(logq, tok[..., None], -1)[..., 0].mean()),
    }

    # a GLOBAL recalibration control: the single best temperature / floor over the whole
    # corrupted window set. Anything a per-offset knob buys over this is event-specific.
    zf, lf, pf = z.reshape(-1, v), logq.reshape(-1, v), pE.reshape(-1, v)
    a_glob = float(temp_opt(zf, pf, pooled=True))
    w_glob = float(floor_opt(lf, pf, pooled=True))
    res["global_knob"] = {
        "alpha": a_glob, "w": w_glob,
        "CE_all": float(xent(pf, lf).mean()),
        "CE_all_alpha": float(temp_ce(zf, pf, a_glob).mean()),
        "CE_all_w": float(floor_ce(lf, pf, w_glob, v).mean())}
    del zf, lf, pf

    win, c = events_from_mask(corrupt, H, c_lo=c_lo, isolate=isolate)
    qw, qt = quiet_positions(corrupt, H, t_lo=c_lo, rng=np.random.default_rng(seed))
    res["isolate"] = bool(isolate)
    res["n_events"] = int(len(win))
    res["n_quiet"] = int(len(qw))
    if len(win) < 50:
        return res

    def rows(w_, t_):
        return z[w_, t_], logq[w_, t_], pE[w_, t_], ceil[w_, t_], logceil[w_, t_], tok[w_, t_]

    # --- per-offset profile -------------------------------------------------
    ev_tr = np.random.default_rng(seed + 11).random(len(win)) < 0.5
    prof = []
    for tau in range(-1, H + 1):
        t_ = c + tau
        ok = t_ >= 0
        zz, lq, pp, cc, lc, tk = rows(win[ok], t_[ok])
        d = {"tau": tau, "n": int(ok.sum()),
             "H_truth": float(ent(pp).mean()), "H_q": float(ent(np.exp(lq)).mean()),
             "H_ceil": float(ent(cc).mean()),
             "CE_model": float(xent(pp, lq).mean()), "CE_ceil": float(xent(pp, lc).mean()),
             "excess_truth": float(kl(pp, lq).mean()),
             "excess_ceil": float((xent(pp, lq) - xent(pp, lc)).mean()),
             "nll": float(-np.take_along_axis(lq, tk[:, None], -1)[:, 0].mean()),
             "KL_q_ceil": float((np.exp(lq) * (lq - lc)).sum(-1).mean()),
             "CE_alpha_global": float(temp_ce(zz, pp, a_glob).mean()),
             "CE_w_global": float(floor_ce(lq, pp, w_glob, v).mean())}
        d.update({f"knob_{k}": vv for k, vv in _knob_block(zz, lq, pp, v).items()})
        # the most a CONTEXT-INDEPENDENT output correction indexed by offset can do:
        # gain + fixed logit bias, fit on half the events and scored on the other half.
        tr, te = ev_tr[ok], ~ev_tr[ok]
        a_af, b_af = affine_fit(zz[tr], pp[tr])
        a_t = float(temp_opt(zz[tr], pp[tr], pooled=True))
        d["affine"] = {"alpha": a_af, "bias_l2": float(np.linalg.norm(b_af)),
                       "CE_te_base": float(xent(pp[te], lq[te]).mean()),
                       "CE_te_temp": float(temp_ce(zz[te], pp[te], a_t).mean()),
                       "CE_te_affine": float(affine_ce(zz[te], pp[te], a_af, b_af).mean()),
                       "CE_te_ceil": float(xent(pp[te], lc[te]).mean())}
        # mixing the model's forecast with observer k's (diagnostic: the model does not have
        # these at inference; mixing with p_kappa* reaches the ceiling by construction)
        pk = pe64[win[ok], t_[ok]]                                    # (n_ev, L+1, v)
        d["mix_obs"] = {}
        for k in range(L + 1):
            wk = float(mix_opt(lq, pk[:, k, :], pp, pooled=True))
            d["mix_obs"][k] = {"w": wk, "CE": float(mix_ce(lq, pk[:, k, :], pp, wk).mean())}
        prof.append(d)
    res["by_tau"] = prof

    # --- quiet background ---------------------------------------------------
    zz, lq, pp, cc, lc, tk = rows(qw, qt)
    q_block = {"H_truth": float(ent(pp).mean()), "H_q": float(ent(np.exp(lq)).mean()),
               "CE_model": float(xent(pp, lq).mean()), "CE_ceil": float(xent(pp, lc).mean()),
               "excess_truth": float(kl(pp, lq).mean()),
               "excess_ceil": float((xent(pp, lq) - xent(pp, lc)).mean())}
    q_block.update({f"knob_{k}": vv for k, vv in _knob_block(zz, lq, pp, v).items()})
    res["quiet"] = q_block

    # --- pooled over the whole event window tau = 0..H ----------------------
    W = np.repeat(win, H + 1)
    Tt = (c[:, None] + np.arange(H + 1)[None, :]).ravel()
    zz, lq, pp, cc, lc, tk = rows(W, Tt)
    pooled = {"n_pred": int(len(W))}
    pooled.update({"CE_model": float(xent(pp, lq).mean()), "CE_ceil": float(xent(pp, lc).mean()),
                   "excess_truth": float(kl(pp, lq).mean()),
                   "excess_ceil": float((xent(pp, lq) - xent(pp, lc)).mean())})
    pooled.update({f"knob_{k}": vv for k, vv in _knob_block(zz, lq, pp, v).items()})

    # per-offset alpha / w (one knob value per tau, shared by all events) and per-event
    ce_base = xent(pp, lq).reshape(len(win), H + 1)
    a_tau = np.array([prof[tau + 1]["knob_alpha_pooled"] for tau in range(H + 1)])
    w_tau = np.array([prof[tau + 1]["knob_w_pooled"] for tau in range(H + 1)])
    ce_a_tau = temp_ce(zz, pp, np.repeat(a_tau[None, :], len(win), 0).ravel()).reshape(len(win), -1)
    ce_w_tau = floor_ce(lq, pp, np.repeat(w_tau[None, :], len(win), 0).ravel(), v).reshape(len(win), -1)

    def _per_event(knob):
        """one knob value per EVENT, shared across tau = 0..H (oracle-chosen)."""
        zz3 = zz.reshape(len(win), H + 1, v)
        lq3 = lq.reshape(len(win), H + 1, v)
        pp3 = pp.reshape(len(win), H + 1, v)
        if knob == "alpha":
            def d(a):
                q = np.exp(log_softmax(a[:, None, None] * zz3))
                return ((q - pp3) * zz3).sum(-1).mean(-1)
            from rhm.logit_reading.coeruleus.events import _bisect
            a = _bisect(d, 0.02, 30.0, (len(win),))
            return a, temp_ce(zz3, pp3, a[:, None]).mean(-1)
        q3 = np.exp(lq3)

        def d(w_):
            qm = (1 - w_[:, None, None]) * q3 + w_[:, None, None] / v
            return (-(pp3 * (1.0 / v - q3) / np.clip(qm, 1e-30, None)).sum(-1)).mean(-1)
        from rhm.logit_reading.coeruleus.events import _bisect
        w_ = _bisect(d, 0.0, 0.999, (len(win),))
        return w_, floor_ce(lq3, pp3, w_[:, None], v).mean(-1)

    a_ev, ce_a_ev = _per_event("alpha")
    w_ev, ce_w_ev = _per_event("floor")
    ce_ceil = xent(pp, lc).reshape(len(win), H + 1)
    base = float(ce_base.mean())
    ceil_v = float(ce_ceil.mean())
    prize = base - ceil_v

    def frac(x):
        return float((base - x) / prize) if prize > 1e-9 else float("nan")

    pooled["prize_nats_per_pred"] = prize
    pooled["prize_nats_per_event"] = prize * (H + 1)
    pooled["recovered"] = {
        "temp_per_offset": {"nats": base - float(ce_a_tau.mean()), "frac": frac(float(ce_a_tau.mean()))},
        "temp_per_event_oracle": {"nats": base - float(ce_a_ev.mean()), "frac": frac(float(ce_a_ev.mean())),
                                  "alpha_mean": float(a_ev.mean()), "alpha_median": float(np.median(a_ev))},
        "temp_per_pred_oracle": {"nats": base - pooled["knob_CE_alpha_row"],
                                 "frac": frac(pooled["knob_CE_alpha_row"])},
        "floor_per_offset": {"nats": base - float(ce_w_tau.mean()), "frac": frac(float(ce_w_tau.mean()))},
        "floor_per_event_oracle": {"nats": base - float(ce_w_ev.mean()), "frac": frac(float(ce_w_ev.mean())),
                                   "w_mean": float(w_ev.mean()), "w_median": float(np.median(w_ev))},
        "floor_per_pred_oracle": {"nats": base - pooled["knob_CE_w_row"],
                                  "frac": frac(pooled["knob_CE_w_row"])},
    }
    ce_a_glob = float(temp_ce(zz, pp, a_glob).mean())
    ce_w_glob = float(floor_ce(lq, pp, w_glob, v).mean())
    pooled["recovered"]["temp_global"] = {"nats": base - ce_a_glob, "frac": frac(ce_a_glob),
                                          "alpha": a_glob}
    pooled["recovered"]["floor_global"] = {"nats": base - ce_w_glob, "frac": frac(ce_w_glob),
                                           "w": w_glob}
    pooled["alpha_by_tau"] = a_tau.round(4).tolist()
    pooled["w_by_tau"] = w_tau.round(5).tolist()
    res["event_window"] = pooled

    # --- stream-level accounting -------------------------------------------
    ev_all, c_all = events_from_mask(corrupt, 0, c_lo=0, isolate=False)
    res["stream"] = {
        "events_per_window": float(len(c_all) / n),
        "frac_preds_within_H": float(min(1.0, len(c_all) * (H + 1) / (n * T))),
        "excess_ceil_all_preds": res["overall"]["excess_ceil"],
        "excess_ceil_in_event_window_share": float(
            prize * len(win) * (H + 1) / max(res["overall"]["excess_ceil"] * n * T, 1e-12)),
    }
    return res


@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=3600, memory=16384,
              max_containers=4)
def prize_ckpt(ckpt: str, eps_data: float = 0.01, eps_obs: float = 0.01, horizon: int = 8,
               n_windows: int = 4096, eval_seed: int = 4242, isolate: bool = True,
               tag: str = ""):
    import resource
    volume.reload()
    model, rules, rule_w, cfg = load_trajectory_ckpt(ckpt, "cuda")
    w, w_clean, corrupt, phase = assert_cache_matches(rules, rule_w, eps_data, n_windows, eval_seed)
    pe, wl, _ = load_ladder(eps_data, eps_obs)
    pc, _, _ = load_ladder(eps_data, 0.0)
    assert (wl == w).all()
    z = model_logits(model, w, "cuda")
    res = prize_stats(z, pe, pc, w, corrupt, H=horizon, isolate=isolate)
    res["config"] = {"ckpt": ckpt, "step": cfg.get("step"), "eps_train": cfg.get("eps_train", 0.0),
                     "eps_data": eps_data, "eps_obs": eps_obs, "n_windows": n_windows,
                     "eval_seed": eval_seed}
    print(f"step {cfg.get('step')} eps_train {cfg.get('eps_train', 0.0)} data eps {eps_data}  "
          f"kappa*_eps {res['kappa_star_eps']:.2f}  events {res['n_events']}", flush=True)
    if "by_tau" in res:
        print(f"{'tau':>4} {'n':>5} {'H_q':>7} {'H_true':>7} {'exTruth':>8} {'exCeil':>8} "
              f"{'alpha':>7} {'recA':>7} {'w':>8} {'recW':>7}")
        for d in res["by_tau"]:
            print(f"{d['tau']:>4} {d['n']:>5} {d['H_q']:.4f} {d['H_truth']:.4f} "
                  f"{d['excess_truth']:+.4f} {d['excess_ceil']:+.4f} "
                  f"{d['knob_alpha_pooled']:.4f} {d['CE_model'] - d['knob_CE_alpha_pooled']:+.4f} "
                  f"{d['knob_w_pooled']:.5f} {d['CE_model'] - d['knob_CE_w_pooled']:+.4f}", flush=True)
        print("  affine (gain+bias, per offset, held-out events):  " + "  ".join(
            f"t{d['tau']} {d['affine']['CE_te_base'] - d['affine']['CE_te_affine']:+.4f}"
            for d in res["by_tau"]), flush=True)
        d0 = res["by_tau"][1]
        print("  mix with obs k (tau=0), w / nats recovered: " + "  ".join(
            f"k{k} {vv['w']:.3f}/{d0['CE_model'] - vv['CE']:+.4f}"
            for k, vv in d0["mix_obs"].items()), flush=True)
        r = res["event_window"]["recovered"]
        print(f"  prize/pred {res['event_window']['prize_nats_per_pred']:.4f} nats  "
              f"temp/offset {r['temp_per_offset']['frac']:.3f}  "
              f"temp/event {r['temp_per_event_oracle']['frac']:.3f}  "
              f"temp/pred {r['temp_per_pred_oracle']['frac']:.3f}  "
              f"temp/global {r['temp_global']['frac']:.3f} (a={r['temp_global']['alpha']:.3f})  |  "
              f"floor/offset {r['floor_per_offset']['frac']:.3f}  "
              f"floor/event {r['floor_per_event_oracle']['frac']:.3f}  "
              f"floor/pred {r['floor_per_pred_oracle']['frac']:.3f}", flush=True)
        print(f"  quiet: excess_ceil {res['quiet']['excess_ceil']:+.5f}  "
              f"alpha {res['quiet']['knob_alpha_pooled']:.4f}  w {res['quiet']['knob_w_pooled']:.5f}",
              flush=True)
    name = f"{ckpt[:-3]}_prize_d{eps_data}{('_' + tag) if tag else ''}.json"
    with open(name, "w") as f:
        json.dump(res, f, cls=NumpyEncoder, indent=1)
    volume.commit()
    print(f"peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.2f} GB -> {name}",
          flush=True)
    return {"ckpt": ckpt, "step": cfg.get("step")}


@app.function(volumes={DATA_DIR: volume}, timeout=3 * 3600, memory=2048)
def prize_sweep(ckpts: str, eps_data: float = 0.01, horizon: int = 8,
                isolate: bool = True, tag: str = ""):
    volume.reload()
    return list(prize_ckpt.map(ckpts.split(","),
                               kwargs={"eps_data": eps_data, "horizon": horizon,
                                       "isolate": isolate, "tag": tag}))
