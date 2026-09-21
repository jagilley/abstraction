"""junction: the critic's diet is the knob.

`striatum/` showed that a critic trained on outcomes alone reads a violation's COST and
not its STRUCTURE, and that at matched consequence it is indifferent to legality.  That
was measured on ONE diet -- the natural `a1` stimulus mix.  This node makes the diet the
independent variable, on the same trunk, the same frozen actor and the SAME held-out test
rows (the split seed is fixed, so every diet is read on identical episodes).

Q1, the junction.  Five window-level diets (`diets.py`) whose marginals -- P(illegal) and
the distribution of consequence depth -- are pinned identical and whose association
between "the edit was illegal" and "the edit was costly" runs from strongly negative to
strongly positive.  Readout: the striatum contrasts on the fixed test rows, plus an
oracle legality probe on the same anchor states as the ceiling.

Q2, the surprise gate.  Row-level diets that keep the fraction of training (window,
position) rows the model's own realised horizon excess flags, against matched-size random
subsets (one of them position-matched), the anti-gate, an oracle-damage gate and the full
diet, across a ladder of sizes.  Readout: realised-damage AUC, the per-diet shuffled
floor, and the containment of the banked `coeruleus/` excess head inside the critic.

Everything below the Gram is shared: one forward pass over the original windows, one over
the edited windows, one frozen clean-trained actor, one clean-only critic.  A diet is a
boolean mask over the Gram's rows, so all of them are fitted from the same cached states.

Run:
  modal run -m rhm.logit_reading.striatum.junction.task::junction_ckpt \
      --ckpt /data/v16_s2_L6_m4_distinct/logit_reading/traj_a1_s42/step064000.pt \
      --stim-tag a1 --max-windows 3000 --n-clean 1024 --actor-steps 300 --tag smoke
  modal run --detach -m rhm.logit_reading.striatum.junction.task::junction_sweep
"""

import json
import os
import resource
import time

import numpy as np

import modal

from rhm.shared import volume, DATA_DIR, NumpyEncoder
from rhm.logit_reading.calibration import load_trajectory_ckpt, tb_key
from rhm.logit_reading.violation import auc
from rhm.logit_reading.striatum.task import train_actor, actor_apply, ridge_solve, _r2
from rhm.logit_reading.striatum.junction.diets import (window_diets, damage_diets,
                                                       row_diets, cons_depth)


def _ignore(path):
    """Mount the Python sources only, and never a sibling node's `results/` tree -- another
    agent writing its launch log there makes the image build race and fail
    (`add_local_python_source` raises ExecutionError even for a file it will ignore)."""
    sp = str(path)
    if "/results/" in sp or sp.endswith("/results") or sp.endswith(".log"):
        return True
    return not sp.endswith(".py")


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==1.26.4", "scipy==1.16.3", "torch==2.7.0")
    .add_local_python_source("rhm", ignore=_ignore)
)
app = modal.App("rhm-striatum-junction", image=image)

LEVELS = [1, 2, 3, 4, 5, 6]
DMAX = 12                      # unchanged from striatum: fixes the Gram's row range
DD_LIST = [0, 1, 2, 3, 4, 5, 8, 9]     # the horizons a in A_LIST needs (a and a+1)
A_LIST = [0, 1, 2, 4, 8]
A_STORE = [0]                  # per-diet columns are stored at a = 0 (every readout uses it)
L_STORE = [1, 2, 3, 4]
T_LO = 8
LAMS = [1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
BLOCKS = ["post_embed", "post_block7"]
GATE_FRACS = [0.01, 0.03, 0.06, 0.12, 0.25, 0.5]


def tgt_names():
    real = [f"l{l}_d{dd}" for l in LEVELS for dd in DD_LIST]
    return real + [f"sh_{nm}" for nm in real]


@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=4 * 3600, memory=20480,
              max_containers=4)
def junction_ckpt(ckpt: str, stim_tag: str = "a1", n_clean: int = 6144,
                  n_clean_val: int = 2048, max_windows: int = 0, seed: int = 11,
                  tag: str = "", actor_steps: int = 3000, mlp_steps: int = 2000,
                  eval_seed: int = 5150, n_clean_crit: int = 8192,
                  prim_block: str = "post_block7", n_val_windows: int = 1500,
                  do_window_diets: bool = True, do_row_diets: bool = True,
                  gate_fracs: str = "", diet_q: int = 0, skip_mlp: bool = False,
                  mlp_levels: str = "1,2,3", chunk: int = 512):
    import torch
    volume.reload()
    t00 = time.time()
    dev = "cuda"
    model, rules, rule_w, cfg = load_trajectory_ckpt(ckpt, dev)
    L, s, v = cfg["L"], cfg["s"], cfg["v"]
    T = model.block_size
    step = cfg.get("step")
    d = model.transformer.wte.weight.shape[1]
    print(f"loaded {ckpt} step {step} d={d} T={T}", flush=True)

    key = tb_key(cfg["v"], cfg["s"], cfg["L"], cfg["m"])
    ddir = f"{DATA_DIR}/{key}/logit_reading"
    S = np.load(f"{ddir}/stimuli_{stim_tag}.npz")
    P = np.load(f"{ddir}/parse_{stim_tag}.npz")
    n_all = S["windows_edit"].shape[0]
    idx = np.arange(n_all)
    if max_windows and max_windows < n_all:
        idx = np.sort(np.random.default_rng(seed).choice(n_all, max_windows, replace=False))
    We, Wo = S["windows_edit"][idx], S["windows_orig"][idx]
    meta = {k: S[k][idx] for k in ("phase", "etype", "j", "e", "first_diff", "t_v", "k_star")}
    y_edit = P["y_edit"][:, idx][:, :, :T]
    y_orig = P["y_orig"][:, idx][:, :, :T]
    n = len(idx)
    et, jj, ee, fd, tv, ks = (meta["etype"], meta["j"], meta["e"], meta["first_diff"],
                              meta["t_v"], meta["k_star"])
    print(f"stimuli {stim_tag}: n={n} etype={np.bincount(et, minlength=3).tolist()}", flush=True)

    # ---- splits over windows (identical recipe to striatum: same seed, same n) --------
    rng = np.random.default_rng(seed + 1)
    perm = rng.permutation(n)
    n_tr, n_va = int(0.60 * n), int(0.15 * n)
    split = np.zeros(n, np.int8)
    split[perm[n_tr:n_tr + n_va]] = 1
    split[perm[n_tr + n_va:]] = 2
    is_tr, is_va, is_te = split == 0, split == 1, split == 2
    tr_idx = np.where(is_tr)[0]
    va_idx = np.where(is_va)[0][:n_val_windows]

    # ---- the banked coeruleus excess head --------------------------------------------
    banked, bpath = None, ckpt[:-3] + "_coeruleus_head_state_excess_ridge_onnoise.pt"
    if not os.path.exists(bpath):
        bpath = ckpt[:-3] + "_coeruleus_head_onnoise.pt"
    if os.path.exists(bpath):
        from rhm.logit_reading.coeruleus.readout import load_head
        banked, _ = load_head(bpath, dev)
        print(f"banked coeruleus head: {bpath} block={banked.block}", flush=True)
    else:
        bpath = None
    blocks = list(BLOCKS)
    if banked is not None and banked.block not in blocks:
        blocks.append(banked.block)
    prim = prim_block
    anc_blocks = [prim] + ([banked.block] if banked is not None and banked.block != prim else [])

    # ---- the model's realised horizon excess (addendum.py's column) -------------------
    hx_path = ckpt[:-3] + f"_striatum_{stim_tag}_hexcess.npz"
    hexcess = None
    if os.path.exists(hx_path):
        HX = np.load(hx_path)
        hexcess = HX["hexcess_edit"][idx]
        print(f"hexcess column: {hx_path} finite {np.isfinite(hexcess).mean():.3f}", flush=True)

    # ---- clean windows -> the frozen actor --------------------------------------------
    from rhm.logit_reading.altitude.units import windows_with_parse
    from rhm.logit_reading.striatum.parse import clean_answers
    t0 = time.time()
    n_cl = n_clean + n_clean_val
    wins_c, phase_c, _, fs_c = windows_with_parse(rules, rule_w, n_cl, T + 1, eval_seed)
    y_c = clean_answers(fs_c, phase_c, T + 1, L, s, 3)[:, :, :T]
    pos = np.arange(T_LO, T)
    actors, actor_tab = {}, {}
    with torch.no_grad():
        Sc = {b: np.zeros((n_cl, T, d), np.float32) for b in blocks}
        for c0 in range(0, n_cl, 256):
            x = torch.as_tensor(wins_c[c0:c0 + 256, :T], device=dev)
            _, _, inter = model(x, return_intermediates=True)
            for b in blocks:
                Sc[b][c0:c0 + 256] = inter[b].float().cpu().numpy()
    for b in blocks:
        Xtr = torch.as_tensor(Sc[b][:n_clean][:, pos].reshape(-1, d), device=dev)
        Xva = torch.as_tensor(Sc[b][n_clean:][:, pos].reshape(-1, d), device=dev)
        for l in LEVELS:
            ytr = torch.as_tensor(y_c[l - 1][:n_clean][:, pos].reshape(-1).astype(np.int64),
                                  device=dev)
            yva = torch.as_tensor(y_c[l - 1][n_clean:][:, pos].reshape(-1).astype(np.int64),
                                  device=dev)
            h = train_actor(Xtr, ytr, Xva, yva, steps=actor_steps, seed=l, device=dev)
            actors[(b, l)] = h
            actor_tab[f"{b}/l{l}"] = h["acc"]
        del Xtr, Xva
        torch.cuda.empty_cache()
    del Sc
    torch.cuda.empty_cache()
    print(f"actors {time.time() - t0:.1f}s", flush=True)
    print(json.dumps({k: round(x, 4) for k, x in actor_tab.items()}), flush=True)

    # ---- anchor positions --------------------------------------------------------------
    anchors = {"fd": np.where(fd >= 0, fd, ee), "tv": np.where(tv > 0, tv, -1)}
    anchor_ok = {"fd": ((fd >= T_LO + 1) | ((et == 2) & (ee >= T_LO + 1))) & (fd <= T - 1),
                 "tv": (tv > T_LO) & (et == 0) & (tv <= T - 1)}
    anc_pos = np.stack([np.clip(anchors["fd"] - 1, 0, T - 1), np.clip(anchors["fd"], 0, T - 1),
                        np.clip(anchors["tv"] - 1, 0, T - 1), np.clip(anchors["tv"], 0, T - 1)], 1)
    ANC = {"fd": (0, 1), "tv": (2, 3)}

    rows = np.arange(T_LO, T - 1 - DMAX + 1)
    R_ = len(rows)
    cache_idx = np.concatenate([tr_idx, va_idx])
    cache_map = -np.ones(n, np.int64)
    cache_map[cache_idx] = np.arange(len(cache_idx))

    # ---- the two forward passes ---------------------------------------------------------
    def run(W, Y, want_cache=False, bs=256):
        o = np.zeros((len(LEVELS), len(W), T), np.uint8)
        nll = np.zeros((len(W), T), np.float32)
        Hq = np.zeros((len(W), T), np.float32)
        anc = {b: np.zeros((len(W), 4, d), np.float32) for b in anc_blocks}
        cache = (np.zeros((len(cache_idx), R_, d), np.float16) if want_cache else None)
        with torch.no_grad():
            for c0 in range(0, len(W), bs):
                sl = slice(c0, min(c0 + bs, len(W)))
                x = torch.as_tensor(W[sl, :T], device=dev)
                lg, _, inter = model(x, return_intermediates=True)
                lsm = torch.log_softmax(lg.float(), -1)
                nxt = torch.as_tensor(W[sl, 1:T + 1], device=dev)
                nll[sl, :] = (-lsm.gather(-1, nxt[..., None])[..., 0]).cpu().numpy()
                Hq[sl, :] = (-(lsm.exp() * lsm).sum(-1)).cpu().numpy()
                Xp = inter[prim].float()
                for li, l in enumerate(LEVELS):
                    pred = actor_apply(actors[(prim, l)], Xp)
                    tru = torch.as_tensor(Y[li][sl].astype(np.int64), device=dev)
                    o[li, sl] = (pred == tru).to(torch.uint8).cpu().numpy()
                ap = torch.as_tensor(anc_pos[sl], device=dev)
                ar = torch.arange(x.shape[0], device=dev)[:, None]
                for b in anc_blocks:
                    anc[b][sl] = inter[b][ar, ap].float().cpu().numpy()
                if want_cache:
                    kk = np.where(cache_map[c0:sl.stop] >= 0)[0]
                    if len(kk):
                        kt = torch.as_tensor(kk, device=dev)
                        cache[cache_map[c0 + kk]] = \
                            inter[prim][kt][:, rows].half().cpu().numpy()
        return o, nll, Hq, anc, cache

    t0 = time.time()
    o_o, nll_o, H_o, st_o, _ = run(Wo, y_orig)
    print(f"orig pass {time.time() - t0:.1f}s", flush=True)
    t0 = time.time()
    o_e, nll_e, H_e, st_e, cache = run(We, y_edit, want_cache=True)
    print(f"edit pass {time.time() - t0:.1f}s  cache {cache.nbytes / 1e9:.2f} GB", flush=True)

    # ---- the two labels a diet can be cut on --------------------------------------------
    c_depth = cons_depth(et, jj)                       # structural consequence depth
    dmg_full = (o_o.astype(np.int16) - o_e.astype(np.int16)) > 0     # realised damage
    dmg_row = dmg_full.sum(0).astype(np.float32)                     # (n, T)
    dmg_w = dmg_full[:4][:, :, rows]                                 # (4, n, R)
    dmg_win = dmg_w.mean((0, 2))                                     # per window

    # ---- the diets ----------------------------------------------------------------------
    masks, spec, wmeta = {}, {}, {}
    if do_window_diets and (et == 1).sum() > 0:
        wmask, wspec, wmeta, none_sel = window_diets(et, jj, is_tr, seed=seed + 7, q=diet_q)
        dmask, dspec, dmeta = damage_diets(et, dmg_win, is_tr, none_sel, seed=seed + 9)
        wmask.update(dmask)
        wspec.update(dspec)
        wmeta.update(dmeta)
        for k_, m_ in wmask.items():
            masks[k_] = np.repeat(m_[tr_idx][:, None], R_, 1)
            spec[k_] = wspec[k_]
    elif do_window_diets:
        masks["full"] = np.ones((len(tr_idx), R_), bool)
        spec["full"] = {"kind": "window", "note": "every training window"}
    if do_row_diets and hexcess is not None:
        fr = [float(x) for x in gate_fracs.split(",")] if gate_fracs else GATE_FRACS
        rmask, rspec = row_diets(hexcess, dmg_row, tr_idx, rows, fr, seed=seed + 8)
        for k_, m_ in rmask.items():
            if k_ == "rowfull" and "full" in masks:
                continue
            masks[k_] = m_
            spec[k_] = rspec[k_]
    diet_names = list(masks)
    print(f"{len(diet_names)} diets: {diet_names}", flush=True)

    # ---- diet statistics (what each diet's world actually looks like) --------------------
    med_global = float(np.median(dmg_win[is_tr & (et < 2)])) if (et < 2).any() else 0.0
    diet_stats = {}
    for nm in diet_names:
        m = np.asarray(masks[nm])
        wsel = tr_idx[m.any(1)]
        nrow = int(m.sum())
        ed = wsel[et[wsel] < 2]
        sw, ra = et[ed] == 0, et[ed] == 1
        st_ = {"n_rows": nrow, "n_windows": int(len(wsel)),
               "n_swap": int(sw.sum()), "n_rare": int(ra.sum()),
               "n_none": int((et[wsel] == 2).sum())}
        if sw.sum() and ra.sum():
            dw = dmg_win[ed]
            hi = dw > med_global            # one threshold for every diet, so assoc compares
            st_["c_mean_swap"] = float(c_depth[ed][sw].mean())
            st_["c_mean_rare"] = float(c_depth[ed][ra].mean())
            st_["c_gap"] = st_["c_mean_swap"] - st_["c_mean_rare"]
            st_["dmg_mean_swap"] = float(dw[sw].mean())
            st_["dmg_mean_rare"] = float(dw[ra].mean())
            st_["p_costly_given_illegal"] = float(hi[sw].mean())
            st_["p_costly_given_legal"] = float(hi[ra].mean())
            st_["assoc"] = st_["p_costly_given_illegal"] - st_["p_costly_given_legal"]
            st_["auc_illegal_for_cost"] = auc(sw.astype(float), hi)
            st_["auc_cost_for_illegal"] = auc(dw, sw)      # threshold-free association
            st_["p_tv_given_swap"] = float((tv[ed][sw] > 0).mean())
        if spec[nm].get("kind") == "row":
            rpos = np.broadcast_to(rows[None, :], m.shape)[m]
            st_["mean_pos"] = float(rpos.mean())
            hsel = np.nan_to_num(hexcess[np.ix_(tr_idx, rows)], nan=0.0)[m]
            st_["mean_hexcess"] = float(hsel.mean())
            st_["mean_dmg_row"] = float(dmg_w.sum(0)[tr_idx][m].mean())
        diet_stats[nm] = st_

    # ---- Gram per diet -------------------------------------------------------------------
    names = tgt_names()
    n_tgt = len(names)
    n_real = n_tgt // 2
    A = {nm: torch.zeros(d + 1, d + 1, dtype=torch.float64, device=dev) for nm in diet_names}
    C = {nm: torch.zeros(d + 1, n_tgt, dtype=torch.float64, device=dev) for nm in diet_names}
    shuf = np.random.default_rng(seed + 3).permutation(len(tr_idx))
    t0 = time.time()
    for c0 in range(0, len(tr_idx), chunk):
        sl = slice(c0, min(c0 + chunk, len(tr_idx)))
        wsl = tr_idx[sl]
        X = torch.as_tensor(cache[sl].reshape(-1, d), device=dev).double()
        X = torch.cat([X, torch.ones(len(X), 1, dtype=torch.float64, device=dev)], 1)
        ys = []
        for src in (wsl, tr_idx[shuf[sl]]):
            for li in range(len(LEVELS)):
                for dd in DD_LIST:
                    ys.append(torch.as_tensor(o_e[li][np.ix_(src, rows + dd)],
                                              dtype=torch.float64, device=dev))
        Y = torch.stack(ys, -1).reshape(-1, n_tgt)
        for nm in diet_names:
            sel = torch.as_tensor(np.where(np.asarray(masks[nm])[sl].reshape(-1))[0],
                                  device=dev)
            if not len(sel):
                continue
            Xs = X[sel]
            A[nm] += Xs.T @ Xs
            C[nm] += Xs.T @ Y[sel]
        del X, Y
    print(f"grams {time.time() - t0:.1f}s", flush=True)

    # ---- lambda selection on held-out val rows -------------------------------------------
    t0 = time.time()
    Xv = torch.as_tensor(cache[len(tr_idx):].reshape(-1, d), device=dev).double()
    Xv = torch.cat([Xv, torch.ones(len(Xv), 1, dtype=torch.float64, device=dev)], 1)
    yv = []
    for li in range(len(LEVELS)):
        for dd in DD_LIST:
            yv.append(torch.as_tensor(o_e[li][np.ix_(va_idx, rows + dd)],
                                      dtype=torch.float64, device=dev))
    Yv = torch.stack(yv, -1).reshape(-1, n_real)
    Yv = torch.cat([Yv, Yv], 1)
    yvar = Yv.var(0)
    beta, val_r2 = {}, {}
    for nm in diet_names:
        An, Cn = A[nm].cpu().numpy(), C[nm].cpu().numpy()
        best = torch.full((n_tgt,), -1e18, dtype=torch.float64, device=dev)
        Bb = torch.zeros(d + 1, n_tgt, dtype=torch.float64, device=dev)
        lam_pick = np.zeros(n_tgt)
        for lm in LAMS:
            B = torch.as_tensor(ridge_solve(An, Cn, lm), device=dev)
            r2 = 1.0 - ((Xv @ B - Yv) ** 2).mean(0) / torch.clamp(yvar, min=1e-12)
            up = r2 > best
            best = torch.where(up, r2, best)
            Bb[:, up] = B[:, up]
            lam_pick[up.cpu().numpy()] = lm
        beta[nm] = Bb.cpu().numpy()
        val_r2[nm] = {names[i]: float(best[i]) for i in range(n_tgt)}
        val_r2[nm]["_lam_l1_d0"] = float(lam_pick[names.index("l1_d0")])
    del A, C, Xv, Yv
    torch.cuda.empty_cache()
    print(f"solves {time.time() - t0:.1f}s  full l1_d0 R2 "
          f"{val_r2.get('full', val_r2[diet_names[0]])['l1_d0']:.4f}", flush=True)

    # ---- the clean-only critic (diet-independent floor) -----------------------------------
    t0 = time.time()
    real_names = names[:n_real]
    wins_cc, phase_cc, _, fs_cc = windows_with_parse(rules, rule_w, n_clean_crit, T + 1,
                                                     eval_seed + 1000)
    y_cc = clean_answers(fs_cc, phase_cc, T + 1, L, s, 3)[:, :, :T]
    n_cc_va = min(2000, n_clean_crit // 5)
    n_cc_tr = n_clean_crit - n_cc_va
    Acc = torch.zeros(d + 1, d + 1, dtype=torch.float64, device=dev)
    Ccc = torch.zeros(d + 1, n_real, dtype=torch.float64, device=dev)
    o_cc = np.zeros((len(LEVELS), n_clean_crit, T), np.uint8)
    Xcc_va = np.zeros((n_cc_va, R_, d), np.float32)
    with torch.no_grad():
        for c0 in range(0, n_clean_crit, 256):
            sl = slice(c0, min(c0 + 256, n_clean_crit))
            x = torch.as_tensor(wins_cc[sl, :T], device=dev)
            _, _, inter = model(x, return_intermediates=True)
            Xp = inter[prim].float()
            for li, l in enumerate(LEVELS):
                pred = actor_apply(actors[(prim, l)], Xp)
                tru = torch.as_tensor(y_cc[li][sl].astype(np.int64), device=dev)
                o_cc[li, sl] = (pred == tru).to(torch.uint8).cpu().numpy()
            ii = np.arange(sl.start, sl.stop)
            kk = np.where(ii < n_cc_tr)[0]
            if len(kk):
                kt = torch.as_tensor(kk, device=dev)
                Xb = inter[prim][kt][:, rows].double().reshape(-1, d)
                Xb = torch.cat([Xb, torch.ones(len(Xb), 1, dtype=torch.float64,
                                               device=dev)], 1)
                ys = [torch.as_tensor(o_cc[li][np.ix_(ii[kk], rows + dd)],
                                      dtype=torch.float64, device=dev)
                      for li in range(len(LEVELS)) for dd in DD_LIST]
                Acc += Xb.T @ Xb
                Ccc += Xb.T @ torch.stack(ys, -1).reshape(-1, n_real)
            vk = np.where(ii >= n_cc_tr)[0]
            if len(vk):
                Xcc_va[ii[vk] - n_cc_tr] = \
                    inter[prim][torch.as_tensor(vk, device=dev)][:, rows].float().cpu().numpy()
    Xvc = torch.as_tensor(Xcc_va.reshape(-1, d), device=dev).double()
    Xvc = torch.cat([Xvc, torch.ones(len(Xvc), 1, dtype=torch.float64, device=dev)], 1)
    Yvc = torch.stack([torch.as_tensor(
        o_cc[li][np.ix_(np.arange(n_cc_tr, n_clean_crit), rows + dd)],
        dtype=torch.float64, device=dev) for li in range(len(LEVELS)) for dd in DD_LIST],
        -1).reshape(-1, n_real)
    Accn, Cccn = Acc.cpu().numpy(), Ccc.cpu().numpy()
    bestc = torch.full((n_real,), -1e18, dtype=torch.float64, device=dev)
    Bc = torch.zeros(d + 1, n_real, dtype=torch.float64, device=dev)
    for lm in LAMS:
        B = torch.as_tensor(ridge_solve(Accn, Cccn, lm), device=dev)
        r2 = 1.0 - ((Xvc @ B - Yvc) ** 2).mean(0) / torch.clamp(Yvc.var(0), min=1e-12)
        up = r2 > bestc
        bestc = torch.where(up, r2, bestc)
        Bc[:, up] = B[:, up]
    beta["clean"] = Bc.cpu().numpy()
    clean_val_r2 = {real_names[i]: float(bestc[i]) for i in range(n_real)}
    del Acc, Ccc, Xvc, Yvc, Xcc_va
    torch.cuda.empty_cache()
    print(f"clean-only critic {time.time() - t0:.1f}s  l1_d0 R2 "
          f"{clean_val_r2['l1_d0']:.4f}", flush=True)

    # ---- optional MLP critic for the window diets ------------------------------------------
    mlp = {}
    mlp_val = {}
    wdiets = [nm for nm in diet_names if spec[nm].get("kind") == "window"]
    if not skip_mlp and wdiets:
        from rhm.logit_reading.coeruleus.readout import train_head
        mls = [int(x) for x in mlp_levels.split(",") if x]
        t0 = time.time()
        Xva_m = cache[len(tr_idx):].reshape(-1, d)
        for nm in wdiets:
            m = np.asarray(masks[nm])
            Xtr_m = cache[:len(tr_idx)][m].astype(np.float32)
            for l in mls:
                li = LEVELS.index(l)
                for dd in (0, 1):
                    ytr = o_e[li][np.ix_(tr_idx, rows + dd)][m].astype(np.float32)
                    h = train_head(Xtr_m, ytr, steps=mlp_steps, device=dev, seed=l * 17 + dd)
                    mlp[(nm, f"l{l}_d{dd}")] = h
                    if dd == 0:
                        yva_ = o_e[li][np.ix_(va_idx, rows)].reshape(-1).astype(np.float64)
                        mlp_val[f"{nm}/l{l}_d0"] = _r2(h(Xva_m), yva_)
            del Xtr_m
            torch.cuda.empty_cache()
        print(f"mlp critics {time.time() - t0:.1f}s ({len(mlp)} fits)", flush=True)
    del cache
    torch.cuda.empty_cache()

    # ---- anchors, revisions, labels ---------------------------------------------------------
    def lin(nm, name, X):
        b = beta[nm][:, names.index(name) if nm != "clean" else real_names.index(name)]
        return X @ b[:-1] + b[-1]

    pL = S["pL_edit"][idx] if "pL_edit" in S.files else None
    out_rows = {}
    for an, t0a in anchors.items():
        ok = anchor_ok[an]
        w = np.where(ok)[0]
        if len(w) < 50:
            continue
        ib, ia = ANC[an]
        t0w = t0a[w]
        rec = {"w": w, "t0": t0w, "etype": et[w], "j": jj[w], "k_star": ks[w], "e": ee[w],
               "split": split[w], "first_diff": fd[w], "t_v": tv[w],
               "c_depth": c_depth[w]}
        rec["nll_e"] = nll_e[w, t0w - 1]
        rec["nll_o"] = nll_o[w, t0w - 1]
        rec["H_pre"] = H_e[w, t0w - 1]
        rec["dH_e"] = H_e[w, t0w] - H_e[w, t0w - 1]
        rec["dH_o"] = H_o[w, t0w] - H_o[w, t0w - 1]
        rec["excess_e"] = rec["nll_e"] - rec["H_pre"]
        rec["tok_legal"] = (np.zeros(len(w), np.int8) if pL is None else
                            (pL[w, t0w, We[w, t0w]] > 0).astype(np.int8))
        rec["prefix_dev"] = np.abs(st_e[prim][w, ib] - st_o[prim][w, ib]).max(1)
        if hexcess is not None:
            hh = np.nan_to_num(hexcess, nan=0.0)
            rec["hexcess"] = hh[w, np.clip(t0w, 0, hh.shape[1] - 1)]
            rec["hexcess_pre"] = hh[w, np.clip(t0w - 1, 0, hh.shape[1] - 1)]
        Xa, Xb_ = st_e[prim][w, ia].astype(np.float64), st_e[prim][w, ib].astype(np.float64)
        Xao, Xbo = st_o[prim][w, ia].astype(np.float64), st_o[prim][w, ib].astype(np.float64)
        for nm in diet_names + ["clean"]:
            ls = L_STORE if nm != "full" else LEVELS
            as_ = A_STORE if nm != "full" else A_LIST
            for l in ls:
                for a in as_:
                    n0, n1 = f"l{l}_d{a}", f"l{l}_d{a + 1}"
                    rec[f"R_{nm}_l{l}_a{a}"] = lin(nm, n0, Xa) - lin(nm, n1, Xb_)
                    rec[f"Ro_{nm}_l{l}_a{a}"] = lin(nm, n0, Xao) - lin(nm, n1, Xbo)
                    rec[f"V_{nm}_l{l}_a{a}"] = lin(nm, n0, Xa)
                    rec[f"Vpre_{nm}_l{l}_a{a}"] = lin(nm, n1, Xb_)
                    if nm != "clean":
                        rec[f"Rsh_{nm}_l{l}_a{a}"] = (lin(nm, f"sh_{n0}", Xa)
                                                      - lin(nm, f"sh_{n1}", Xb_))
        for (nm, kk), h in mlp.items():
            l, dd = int(kk.split("l")[1].split("_")[0]), int(kk.split("_d")[1])
            if dd != 0:
                continue
            h1 = mlp.get((nm, f"l{l}_d1"))
            if h1 is None:
                continue
            rec[f"Rmlp_{nm}_l{l}_a0"] = h(Xa) - h1(Xb_)
            rec[f"Vmlp_{nm}_l{l}_a0"] = h(Xa)
        for l in LEVELS:
            li = LEVELS.index(l)
            for a in A_LIST:
                good = (t0w + a) <= T - 1
                tq = np.clip(t0w + a, 0, T - 1)
                rec[f"cons_l{l}_a{a}"] = ((y_edit[li][w, tq] != y_orig[li][w, tq])
                                          & good).astype(np.int8)
                rec[f"oe_l{l}_a{a}"] = o_e[li][w, tq]
                rec[f"oo_l{l}_a{a}"] = o_o[li][w, tq]
                rec[f"ok_l{l}_a{a}"] = good.astype(np.int8)
        # ---- oracle probes on the same anchor states (the ceilings) ----------------------
        Xab = np.concatenate([Xa, np.ones((len(Xa), 1))], 1)
        tr_m, va_m = rec["split"] == 0, rec["split"] == 1
        labs = {"probe_illegal": ((rec["etype"] == 0) & (rec["t_v"] > 0)).astype(np.float64),
                "probe_swap": (rec["etype"] == 0).astype(np.float64)}
        for l in LEVELS:
            for a in A_LIST:
                labs[f"probe_cons_l{l}_a{a}"] = rec[f"cons_l{l}_a{a}"].astype(np.float64)
        ed_m = rec["etype"] < 2
        nmk = [k for k, vv in labs.items() if 0 < vv[tr_m].mean() < 1]
        if nmk and tr_m.sum() > 50:
            Ag = Xab[tr_m].T @ Xab[tr_m]
            Yl = np.stack([labs[k][tr_m] for k in nmk], 1)
            Yl_v = np.stack([labs[k][va_m] for k in nmk], 1)
            Cg = Xab[tr_m].T @ Yl
            best = np.full(len(nmk), -np.inf)
            Bbest = np.zeros((Xa.shape[1] + 1, len(nmk)))
            for lm in LAMS:
                B = ridge_solve(Ag, Cg, lm)
                r2 = 1.0 - ((Xab[va_m] @ B - Yl_v) ** 2).mean(0) / np.maximum(Yl_v.var(0), 1e-12)
                for ti in np.where(r2 > best)[0]:
                    best[ti] = r2[ti]
                    Bbest[:, ti] = B[:, ti]
            pr = Xab @ Bbest
            for ti, k in enumerate(nmk):
                rec[k] = pr[:, ti]
        # legality probe fitted on edited windows only, linear + MLP (the stated ceiling)
        if (tr_m & ed_m).sum() > 100 and 0 < (rec["etype"][tr_m & ed_m] == 0).mean() < 1:
            mtr = tr_m & ed_m
            Ag = Xab[mtr].T @ Xab[mtr]
            yl = (rec["etype"][mtr] == 0).astype(np.float64)
            Cg = Xab[mtr].T @ yl[:, None]
            mva = va_m & ed_m
            ylv = (rec["etype"][mva] == 0).astype(np.float64)
            bb, br2 = None, -np.inf
            for lm in LAMS:
                B = ridge_solve(Ag, Cg, lm)
                r2 = 1.0 - ((Xab[mva] @ B[:, 0] - ylv) ** 2).mean() / max(ylv.var(), 1e-12)
                if r2 > br2:
                    bb, br2 = B[:, 0], r2
            rec["probe_swap_ed"] = Xab @ bb
            if not skip_mlp:
                from rhm.logit_reading.coeruleus.readout import train_head
                h = train_head(Xa[mtr].astype(np.float32), yl.astype(np.float32),
                               steps=mlp_steps, device=dev, seed=99)
                rec["probe_swap_mlp"] = h(Xa.astype(np.float32))
        if banked is not None:
            rec["banked_excess"] = banked(st_e[banked.block][w, ia])
            rec["banked_excess_pre"] = banked(st_e[banked.block][w, ib])
        out_rows[an] = rec

    # ---- save -------------------------------------------------------------------------------
    res = {"ckpt": ckpt, "step": step, "stim_tag": stim_tag, "n": n, "primary_block": prim,
           "critic_block": prim, "actor_clean_acc": actor_tab,
           "diet_spec": spec, "diet_stats": diet_stats, "diet_meta": wmeta,
           "median_damage_global": med_global,
           "critic_val_r2": {nm: {k: val_r2[nm][k] for k in
                                  [f"l{l}_d{dd}" for l in LEVELS for dd in (0, 1)]
                                  + [f"sh_l{l}_d0" for l in LEVELS] + ["_lam_l1_d0"]}
                             for nm in diet_names},
           "clean_critic_val_r2": clean_val_r2, "mlp_val_r2": mlp_val,
           "banked_head": bpath, "hexcess": hx_path if hexcess is not None else None,
           "config": {"levels": LEVELS, "dmax": DMAX, "dd_list": DD_LIST, "a_list": A_LIST,
                      "t_lo": T_LO, "blocks": blocks, "n_clean": n_clean, "seed": seed,
                      "eval_seed": eval_seed, "split": [0.6, 0.15, 0.25],
                      "n_clean_crit": n_clean_crit, "n_val_windows": int(len(va_idx)),
                      "gate_fracs": gate_fracs or GATE_FRACS, "mlp_levels": mlp_levels,
                      "rows": [int(rows[0]), int(rows[-1])]}}
    stem = ckpt[:-3]
    sfx = f"_junction_{stim_tag}" + (f"_{tag}" if tag else "")
    with open(f"{stem}{sfx}.json", "w") as f:
        json.dump(res, f, cls=NumpyEncoder, indent=1)
    save = {}
    for an, rec in out_rows.items():
        tem = np.asarray(rec["split"]) == 2
        for k, vv in rec.items():
            a_ = np.asarray(vv)[tem]
            save[f"{an}__{k}"] = a_.astype(np.float32) if a_.dtype == np.float64 else a_
    np.savez_compressed(f"{stem}{sfx}.npz", **save)
    volume.commit()
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    print(f"saved -> {stem}{sfx}.json  ({time.time() - t00:.0f}s, peak RSS {rss:.1f} GB)",
          flush=True)
    return {"ckpt": ckpt, "step": step, "stim_tag": stim_tag, "n_diets": len(diet_names),
            "actor_l1": actor_tab.get(f"{prim}/l1"),
            "assoc": {nm: diet_stats[nm].get("assoc") for nm in diet_names
                      if spec[nm].get("kind") == "window"}}


# ---------------------------------------------------------------------------
# coordinator: every cell in its own container
# ---------------------------------------------------------------------------

@app.function(volumes={DATA_DIR: volume}, timeout=6 * 3600, memory=2048)
def junction_sweep(cells: str = "", tag: str = "", n_clean: int = 6144,
                   max_windows: int = 0, gate_fracs: str = "0.005,0.015,0.04,0.1,0.25,0.5"):
    """cells: `ckpt:stim_tag:window_diets:row_diets:skip_mlp`, comma separated."""
    volume.reload()
    D = f"{DATA_DIR}/v16_s2_L6_m4_distinct/logit_reading"
    if not cells:
        cells = ",".join([
            f"{D}/traj_a1_s42/step064000.pt:a1:1:1:0",
            f"{D}/traj_a1_s42/step064000.pt:swap65k:1:1:1",
            f"{D}/traj_a1_s42/step008000.pt:a1:1:1:0",
        ])
    args = []
    for c in cells.split(","):
        p = (c.split(":") + ["1", "1", "0"])[:5]
        ck, tg, wd, rd, sm = p[0], p[1], bool(int(p[2])), bool(int(p[3])), bool(int(p[4]))
        if not os.path.exists(ck):
            print(f"MISSING {ck}", flush=True)
            continue
        args.append((ck, tg, n_clean, 2048, max_windows, 11, tag, 3000, 2000, 5150, 8192,
                     "post_block7", 1500, wd, rd, gate_fracs, 0, sm))
    print(f"{len(args)} cells: {[(a[0].split('/')[-2:], a[1], a[13], a[14], a[17]) for a in args]}",
          flush=True)
    outs = list(junction_ckpt.starmap(args, return_exceptions=True))
    for o in outs:
        print(json.dumps(o, default=str)[:600], flush=True)
    return [str(o)[:600] for o in outs]
