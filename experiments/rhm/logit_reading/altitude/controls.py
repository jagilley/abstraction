"""Q4: the two controls Part 2 of `logit_reading` is missing.

(a) TONIC. 2b's population-matched design found that a scalar integrator of the model's
    own recent surprisal separates violations from matched legal controls at AUC 0.754
    (64k), and read that as accumulated unease. But an exact observer's running surprisal
    over the same stretch of edited material would also be elevated. `tonic_control` keeps
    2b's pairs EXACTLY as `violation.readouts_v2` builds them (same cells, same
    nearest-neighbour matching on the MODEL's surprisal, same seed and caliper) and swaps
    only the readout: the running surprisal of observer k, for every k, taken from the
    stimulus file's exact `ptok` (p_k at the realised token). Every token before t_v is
    legal under p_L and therefore under every coarser observer, so these are finite.

(b) ENTROPY DIRECTION. 2e compared the model against an eps-noise observer that has BOTH
    the full grammar and a noise model, and found the model's post-violation entropy moves
    the wrong way. `entropy_direction` runs the same same-prefix pairs against observer k
    WITH `noise_eps`, at the model's best-fit k, so "it lacks a noise model" and "it is a
    coarser observer" come apart. `noisy_observer_cache` + `persistence_at_k` do the same
    for 2e's unmatched output-level persistence table, reusing the model columns already on
    file in `step*_violation2_a1.json`.

Run:
  modal run --detach -m rhm.logit_reading.altitude.controls::tonic_sweep \
      --traj-dir /data/v16_s2_L6_m4_distinct/logit_reading/traj_a1_s42 --stim-tag a1
  modal run --detach -m rhm.logit_reading.altitude.controls::observer_cache_sweep --stim-tag a1
  modal run --detach -m rhm.logit_reading.altitude.controls::entropy_direction_sweep \
      --traj-dir /data/v16_s2_L6_m4_distinct/logit_reading/traj_a1_s42 --stim-tag swap65k
  modal run -m rhm.logit_reading.altitude.controls::persistence_at_k \
      --traj-dir /data/v16_s2_L6_m4_distinct/logit_reading/traj_a1_s42 --stim-tag a1
"""

import json

import numpy as np

from rhm.shared import volume, DATA_DIR, NumpyEncoder
from rhm.logit_reading.calibration import app, load_trajectory_ckpt
from rhm.logit_reading.violation import auc, nn_match, _load_stimuli, _ent, _kl
from rhm.logit_reading.phasic import paired_win, balance_tokens, balance_ds_sign, _forward_at

EPS = 1e-300


# ---------------------------------------------------------------------------
# (a) tonic: 2b's pairs, the observer's running surprisal as the readout
# ---------------------------------------------------------------------------

def tonic_readouts(model, S, device="cuda", seed=0, caliper=0.1, t_lo=16, n_run=15):
    import torch
    from rhm.logit_reading.flat_oracle import leaf_levels
    rng = np.random.default_rng(seed)
    We = S["windows_edit"]
    n, T1 = We.shape
    T = T1 - 1
    L = S["ptok"].shape[-1] - 1
    v = int(We.max()) + 1

    logq_e = np.empty((n, T, v), np.float32)
    with torch.no_grad():
        for c0 in range(0, n, 512):
            le, _ = model(torch.as_tensor(We[c0:c0 + 512, :T], device=device))
            logq_e[c0:c0 + 512] = torch.log_softmax(le.float(), -1).cpu().numpy()

    et, tv, ks, fd = S["etype"], S["t_v"], S["k_star"], S["first_diff"]
    vw = np.where((et == 0) & (tv >= t_lo) & (tv <= T - 1))[0]
    cw = np.where(et != 0)[0]
    ctrl_w = np.repeat(cw, T - t_lo)
    ctrl_t = np.tile(np.arange(t_lo, T), len(cw))
    W_ = np.concatenate([vw, ctrl_w])
    T_ = np.concatenate([tv[vw], ctrl_t])
    K_ = np.concatenate([np.zeros(len(vw)), np.full(len(ctrl_w), 2)]).astype(int)
    KS_ = np.concatenate([ks[vw], np.full(len(ctrl_w), -1)])
    tok = We[W_, T_]
    lqp = logq_e[W_, T_ - 1]
    surpr = -lqp[np.arange(len(tok)), tok]
    level = leaf_levels((S["phase"][W_] + T_) % T, 2, L)
    cell = (tok * 10 + level) * 100 + T_
    is_tr = (np.random.default_rng(seed + 1).random(n) < 0.6)[W_]

    # running surprisal over the n_run tokens BEFORE the event, model and every observer
    u = T_[:, None] - 1 - np.arange(n_run)[None, :]                   # window indices t-1..t-n
    s_run_model = -logq_e[W_[:, None], u - 1, We[W_[:, None], u]]
    ptok = S["ptok"]                                                  # (n, T1, L+1)
    s_run_obs = {k: -np.log(np.clip(ptok[W_[:, None], u, k], EPS, None)) for k in range(L + 1)}
    assert np.isfinite(np.stack(list(s_run_obs.values()))).all(), "a pre-violation token has p_k = 0"

    out = {"caliper": caliper, "n_viol": int(len(vw)), "n_ctrl": int(len(ctrl_w)), "n_run": n_run}
    M = {}
    for split, flag in (("tr", is_tr), ("te", ~is_tr)):
        p = np.where((K_ == 0) & flag)[0]
        q = np.where((K_ == 2) & flag)[0]
        M[split] = nn_match(p, cell[p], surpr[p], q, cell[q], surpr[q], caliper, rng)
    pv, pc = M["te"]
    idx = np.concatenate([pv, pc])
    y = np.concatenate([np.ones(len(pv), bool), np.zeros(len(pc), bool)])
    out["n_pairs_test"] = int(len(pv))
    out["n_pairs_train"] = int(len(M["tr"][0]))
    if len(pv) < 50:
        return out

    scores = {"guard_surprisal": surpr, "run_model_4": s_run_model[:, :4].mean(1),
              "run_model_15": s_run_model.mean(1)}
    for k in range(L + 1):
        scores[f"run_obs{k}_4"] = s_run_obs[k][:, :4].mean(1)
        scores[f"run_obs{k}_15"] = s_run_obs[k].mean(1)
    # the model's excess over each observer: is the tonic signal the model's own, or the
    # stimulus's?
    for k in range(L + 1):
        scores[f"run_excess{k}_4"] = s_run_model[:, :4].mean(1) - s_run_obs[k][:, :4].mean(1)
    out["test_all"] = {nm: auc(sc[idx], y) for nm, sc in scores.items()}
    out["test_all"]["mean_abs_ds"] = float(np.abs(surpr[pv] - surpr[pc]).mean())
    out["means_test"] = {nm: [float(sc[pv].mean()), float(sc[pc].mean())]
                         for nm, sc in scores.items()}
    byk = {}
    for k in range(1, L + 1):
        sel = KS_[pv] == k
        if sel.sum() < 30:
            continue
        ii = np.concatenate([pv[sel], pc[sel]])
        yy = np.concatenate([np.ones(sel.sum(), bool), np.zeros(sel.sum(), bool)])
        byk[k] = {"n_pairs": int(sel.sum()),
                  **{nm: auc(sc[ii], yy) for nm, sc in scores.items()}}
    out["test_by_kstar"] = byk
    return out


@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=2 * 3600, memory=32768,
              max_containers=4)
def tonic_ckpt(ckpt: str, stim_tag: str = "a1", seed: int = 0, caliper: float = 0.1):
    volume.reload()
    model, rules, rule_w, cfg = load_trajectory_ckpt(ckpt, "cuda")
    S = _load_stimuli(stim_tag)
    res = tonic_readouts(model, S, "cuda", seed=seed, caliper=caliper)
    res["config"] = {"ckpt": ckpt, "step": cfg.get("step"), "stim_tag": stim_tag, "seed": seed}
    a = res.get("test_all", {})
    print(f"step {cfg.get('step')}  pairs {res.get('n_pairs_test')}  "
          + "  ".join(f"{k} {v:.3f}" for k, v in a.items()
                      if k in ("guard_surprisal", "run_model_4", "run_obs3_4", "run_obs4_4",
                               "run_obs5_4", "run_obs6_4")), flush=True)
    with open(f"{ckpt[:-3]}_tonic_{stim_tag}.json", "w") as f:
        json.dump(res, f, cls=NumpyEncoder, indent=1)
    volume.commit()
    return {"step": cfg.get("step")}


@app.function(volumes={DATA_DIR: volume}, timeout=4 * 3600, memory=4096)
def tonic_sweep(traj_dir: str, stim_tag: str = "a1", caliper: float = 0.1):
    import glob
    volume.reload()
    ckpts = sorted(glob.glob(f"{traj_dir}/step*.pt"))
    return len(list(tonic_ckpt.map(ckpts, kwargs={"stim_tag": stim_tag, "caliper": caliper})))


# ---------------------------------------------------------------------------
# (b-ii) model-independent cache: observer (k, eps) on a stimulus set
# ---------------------------------------------------------------------------

@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=4 * 3600, memory=32768,
              max_containers=4)
def observer_cache(k: int = 4, stim_tag: str = "a1", noise_eps: float = 0.01,
                   v: int = 16, s: int = 2, depth: int = 6, m: int = 4, rule_seed: int = 0,
                   alpha: float = 1.0, weight_seed: int = 1, chunk: int = 8,
                   key: str = "v16_s2_L6_m4_distinct"):
    """H(p_{k,eps}) on the edited and original windows, and KL(orig || edit), per position."""
    import time
    from rhm.rhm_data import generate_rules_distinct
    from rhm.logit_reading.grammar import synonym_weights
    from rhm.logit_reading.flat_oracle import flat_predictive
    volume.reload()
    rules = generate_rules_distinct(v, s, depth, m, seed=rule_seed)
    rule_w = None if alpha <= 0 else synonym_weights(v, depth, m, alpha, weight_seed)
    S = _load_stimuli(stim_tag, key)
    ck = chunk if k == depth else max(chunk, 64)
    t0 = time.time()
    pe = flat_predictive(S["windows_edit"], rules, k, device="cuda", chunk=ck,
                         rule_w=rule_w, noise_eps=noise_eps).numpy()
    po = flat_predictive(S["windows_orig"], rules, k, device="cuda", chunk=ck,
                         rule_w=rule_w, noise_eps=noise_eps).numpy()
    print(f"observer k={k} eps={noise_eps} on {stim_tag}: {time.time() - t0:.0f}s", flush=True)
    out = f"{DATA_DIR}/{key}/logit_reading/obs_{stim_tag}_k{k}_eps{noise_eps}.npz"
    np.savez_compressed(out, H_edit=_ent(pe).astype(np.float32), H_orig=_ent(po).astype(np.float32),
                        KL_orig_edit=_kl(po, np.log(np.clip(pe, 1e-30, None))).astype(np.float32),
                        ptok_edit=np.take_along_axis(pe, S["windows_edit"][..., None], -1)[..., 0]
                        .astype(np.float32))
    volume.commit()
    return out


@app.function(volumes={DATA_DIR: volume}, timeout=4 * 3600, memory=4096)
def observer_cache_sweep(stim_tag: str = "a1", ks: str = "2,3,4,5,6", noise_eps: float = 0.01):
    kl = [int(x) for x in ks.split(",")]
    return list(observer_cache.map(kl, kwargs={"stim_tag": stim_tag, "noise_eps": noise_eps}))


@app.function(volumes={DATA_DIR: volume}, timeout=3600, memory=16384)
def persistence_at_k(traj_dir: str, stim_tag: str = "a1", ks: str = "2,3,4,5,6",
                     noise_eps: float = 0.01, tau_max: int = 12, t_lo: int = 16,
                     key: str = "v16_s2_L6_m4_distinct"):
    """2e's unmatched persistence table with observer (k, eps) as the Bayesian. The model
    columns are the ones already on file in step*_violation2_<tag>.json."""
    import glob
    import os
    volume.reload()
    S = _load_stimuli(stim_tag, key)
    L = S["ptok"].shape[-1] - 1
    T = S["windows_edit"].shape[1] - 1
    et, tv, ks_, fd = S["etype"], S["t_v"], S["k_star"], S["first_diff"]
    groups = {f"viol_k{k}": (et == 0) & (tv >= t_lo) & (tv <= T - 1) & (ks_ == k)
              for k in range(1, L + 1)}
    groups["rare"] = (et == 1) & (fd >= t_lo) & (fd <= T - 1)
    out = {"noise_eps": noise_eps, "stim_tag": stim_tag, "by_observer": {}}
    for k in [int(x) for x in ks.split(",")]:
        z = np.load(f"{DATA_DIR}/{key}/logit_reading/obs_{stim_tag}_k{k}_eps{noise_eps}.npz")
        He, Ho, KLoe = z["H_edit"], z["H_orig"], z["KL_orig_edit"]
        rows = {}
        for g, sel in groups.items():
            ws = np.where(sel)[0]
            t0 = tv[ws] if g.startswith("viol") else fd[ws]
            if len(ws) < 20:
                continue
            rr = []
            for tau in range(tau_max + 1):
                ok = t0 + tau <= T - 1
                ii, tt = ws[ok], t0[ok] + tau
                if len(ii) < 10:
                    break
                rr.append({"tau": tau, "n": int(len(ii)),
                           "dH_bayes_k": float((He[ii, tt + 1] - Ho[ii, tt + 1]).mean()),
                           "KL_bayes_k_orig_edit": float(KLoe[ii, tt + 1].mean())})
            rows[g] = rr
        out["by_observer"][k] = rows
        print(f"observer k={k}: " + "  ".join(
            f"{g} dH(tau=0..3) " + ",".join(f"{r['dH_bayes_k']:+.3f}" for r in rows[g][:4])
            for g in sorted(rows) if g.startswith("viol")), flush=True)
    # splice in the model columns already on file
    model_rows = {}
    for pth in sorted(glob.glob(f"{traj_dir}/step*_violation2_{stim_tag}.json")):
        step = int(os.path.basename(pth)[4:10])
        with open(pth) as f:
            j = json.load(f)
        model_rows[step] = {g: [{"tau": r["tau"], "dH_model": r["dH_model"],
                                 "dH_bayes_eps_L": r["dH_bayes_eps"],
                                 "KL_model_orig_edit": r["KL_model_orig_edit"],
                                 "KL_bayes_eps_L": r["KL_bayes_eps_orig_edit"]}
                                for r in rows]
                            for g, rows in j.get("persistence", {}).items()}
    out["model"] = model_rows
    with open(f"{traj_dir}/altitude_persistence_{stim_tag}.json", "w") as f:
        json.dump(out, f, cls=NumpyEncoder, indent=1)
    volume.commit()
    return f"{traj_dir}/altitude_persistence_{stim_tag}.json"


# ---------------------------------------------------------------------------
# (b-i) entropy direction on the same-prefix pairs, observer (k, eps) as the Bayesian
# ---------------------------------------------------------------------------

def _phasic_pairs(model, S, device, seed, caliper, t_lo=16):
    """Exactly `phasic.phasic_readouts`'s pair construction; returns the TEST pairs."""
    We = S["windows_edit"]
    T = We.shape[1] - 1
    et, tv, ks, pL = S["etype"], S["t_v"], S["k_star"], S["pL_edit"]
    vw = np.where((et == 0) & (tv >= t_lo) & (tv <= T - 1))[0]
    t = tv[vw]
    lqp_v, lqn_v, _, _ = _forward_at(model, We[vw], t, device)
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
    lqp_c, lqn_c, _, _ = _forward_at(model, twin, t[P], device)
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
    return {"w": vw[P][te], "t": t[P][te], "x_v": x_v[P][te], "x_c": x_c[P][te],
            "kstar": ks[vw[P]][te], "viol_win": We[vw[P]][te], "twin_win": twin[te],
            "H_next_model_v": _ent(np.exp(lqn_v[P][te])), "H_next_model_c": _ent(np.exp(lqn_c[te])),
            "surpr_v": surpr_v[te], "surpr_c": surpr_c[te], "n_all_pairs": int(has.sum())}


@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=4 * 3600, memory=32768,
              max_containers=4)
def entropy_direction_ckpt(ckpt: str, stim_tag: str = "swap65k", seed: int = 0,
                           caliper: float = 0.3, ks: str = "3,4,5,6", noise_eps: float = 0.01,
                           max_pairs: int = 1200, chunk: int = 8):
    import time
    from rhm.logit_reading.flat_oracle import flat_predictive
    volume.reload()
    model, rules, rule_w, cfg = load_trajectory_ckpt(ckpt, "cuda")
    S = _load_stimuli(stim_tag)
    pr = _phasic_pairs(model, S, "cuda", seed, caliper)
    n = len(pr["w"])
    res = {"step": cfg.get("step"), "caliper": caliper, "noise_eps": noise_eps,
           "n_test_pairs": int(n), "n_all_pairs": pr["n_all_pairs"]}
    if n < 50:
        return res
    sub = np.random.default_rng(0).permutation(n)[:max_pairs]
    res["n_scored"] = int(len(sub))
    ksub = pr["kstar"][sub]
    res["kstar_hist"] = np.bincount(ksub[ksub >= 0], minlength=7).tolist()
    y = np.concatenate([np.ones(len(sub), bool), np.zeros(len(sub), bool)])
    Hm = np.concatenate([pr["H_next_model_v"][sub], pr["H_next_model_c"][sub]])
    res["model"] = {"H_next_auc": auc(Hm, y),
                    "H_next_paired": paired_win(pr["H_next_model_v"][sub], pr["H_next_model_c"][sub]),
                    "H_next_viol": float(pr["H_next_model_v"][sub].mean()),
                    "H_next_twin": float(pr["H_next_model_c"][sub].mean()),
                    "mean_abs_ds": float(np.abs(pr["surpr_v"][sub] - pr["surpr_c"][sub]).mean()),
                    "by_kstar": {int(kk): paired_win(pr["H_next_model_v"][sub][ksub == kk],
                                                     pr["H_next_model_c"][sub][ksub == kk])
                                 for kk in range(1, 7) if (ksub == kk).sum() >= 30}}
    wins = np.concatenate([pr["viol_win"][sub], pr["twin_win"][sub]])
    tt = np.concatenate([pr["t"][sub], pr["t"][sub]])
    res["observers"] = {}
    for k in [int(x) for x in ks.split(",")]:
        t0 = time.time()
        ck = chunk if k >= 6 else max(chunk, 32)
        p = flat_predictive(wins, rules, k, device="cuda", chunk=ck, rule_w=rule_w,
                            noise_eps=noise_eps).numpy()
        H = _ent(p[np.arange(len(wins)), tt + 1])
        res["observers"][k] = {
            "H_next_auc": auc(H, y), "H_next_paired": paired_win(H[:len(sub)], H[len(sub):]),
            "H_next_viol": float(H[:len(sub)].mean()), "H_next_twin": float(H[len(sub):].mean()),
            "by_kstar": {int(kk): paired_win(H[:len(sub)][ksub == kk], H[len(sub):][ksub == kk])
                         for kk in range(1, 7) if (ksub == kk).sum() >= 30},
            "secs": round(time.time() - t0, 1)}
        r = res["observers"][k]
        print(f"  step {cfg.get('step')} observer k={k}: AUC {r['H_next_auc']:.3f} "
              f"paired {r['H_next_paired']:.3f}  H {r['H_next_viol']:.3f} vs {r['H_next_twin']:.3f} "
              f"({r['secs']}s)", flush=True)
    print(f"step {cfg.get('step')}  model AUC {res['model']['H_next_auc']:.3f} "
          f"paired {res['model']['H_next_paired']:.3f}  pairs {res['n_scored']}", flush=True)
    with open(f"{ckpt[:-3]}_entdir_{stim_tag}.json", "w") as f:
        json.dump(res, f, cls=NumpyEncoder, indent=1)
    volume.commit()
    return {"step": cfg.get("step")}


@app.function(volumes={DATA_DIR: volume}, timeout=6 * 3600, memory=4096)
def entropy_direction_sweep(traj_dir: str, stim_tag: str = "swap65k", caliper: float = 0.3,
                            ks: str = "3,4,5,6", steps: str = "", max_pairs: int = 1200):
    import glob
    import os
    volume.reload()
    ckpts = sorted(glob.glob(f"{traj_dir}/step*.pt"))
    if steps:
        want = {int(x) for x in steps.split(",")}
        ckpts = [c for c in ckpts if int(os.path.basename(c)[4:10]) in want]
    print(f"{len(ckpts)} checkpoints", flush=True)
    return len(list(entropy_direction_ckpt.map(
        ckpts, kwargs={"stim_tag": stim_tag, "caliper": caliper, "ks": ks,
                       "max_pairs": max_pairs})))
