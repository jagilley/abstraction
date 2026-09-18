"""norm: the value channel's own norm -- does it calibrate to the world's expected cost,
and does the response to an identical violation scale with it?

`striatum/` trained a critic `V[l, d](s_t) ~ E[o_l(t + d)]` on realised outcomes alone and
found it reads a violation's COST and not its STRUCTURE; `junction/` showed that
indifference survives eleven diets whose violation-cost ASSOCIATION runs from -1 to +1.
Xiang, Lohrenz & Montague (2013) put the norm somewhere else: it is a learned expectation
over OUTCOMES, and the feeling at an event is the outcome minus that adapted expectation --
the same medium offer feels worse to a group that learned to expect more.  Nothing in the
arc has asked whether the outcome-trained critic has that structure.  Three questions:

Q1, across worlds.  Diets from the `a1` training pool that differ only in EXPECTED COST
(`norm/diets.py`): terciles of the window's own outcome / realised damage taken WITHIN each
`(etype, j)` cell, so P(illegal), the width distribution and the structural consequence
profile are identical across diets by construction; plus a quiet-share family that touches
no outcome at all.  Same split seed, so every diet is read on identical held-out rows.
Readouts on those rows: (a) `V(s_{t-1})` before the event -- the norm; (b) `R` at the event
-- the anticipatory response; (c) `d = outcome - V(s_{t-1})` at the query -- the outcome
surprise.

Q2, within a world.  The parent's same-prefix twins (`logit_reading/phasic.py`): the
violating window and the same window with the violator replaced by a legal token of matched
surprisal have a bit-identical state at `t-1`, so `V(s_{t-1})` is identical within a pair
and `dR = R(violator) - R(twin)` cancels the regression to the mean that confounds
conditioning `R` on a high `V_pre` directly.  Does `dR` scale with `V_pre`?  Plus the
matched persistence of `V` and `dR` over offsets after the event.

Q3, scope.  `V_pre` and `R` by the violation's `k*` at 8k against 64k on the same rows --
the value side's version of the parent's Part 2a.

Everything below the Gram is shared with `junction/task.py`'s design: one forward pass over
the original windows, one over the edited windows, one frozen clean-trained actor, one
clean-only critic, and a diet is a boolean mask over the Gram's rows.  The split recipe,
the actor recipe and the per-episode column names are junction's, so
`junction/analyze.py`'s `Rows` reads this node's npz unchanged.

Run:
  modal run -m rhm.logit_reading.striatum.norm.task::norm_ckpt \
      --ckpt /data/v16_s2_L6_m4_distinct/logit_reading/traj_a1_s42/step064000.pt \
      --stim-tag a1 --max-windows 3000 --n-clean 1024 --actor-steps 300 \
      --max-twins 2000 --tag smoke
  modal run --detach -m rhm.logit_reading.striatum.norm.task::norm_sweep
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
from rhm.logit_reading.striatum.junction.diets import cons_depth
from rhm.logit_reading.striatum.norm.diets import (strat_diets, mix_diets, anchor_diets,
                                                   cell_key)


def _ignore(path):
    """Mount the Python sources only, and never a sibling node's `results/` tree or launch
    log -- another agent writing one there makes the image build race and fail."""
    sp = str(path)
    if "/results/" in sp or sp.endswith("/results") or sp.endswith(".log"):
        return True
    return not sp.endswith(".py")


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==1.26.4", "scipy==1.16.3", "torch==2.7.0")
    .add_local_python_source("rhm", ignore=_ignore)
)
app = modal.App("rhm-striatum-norm", image=image)

LEVELS = [1, 2, 3, 4, 5, 6]
DMAX = 12                      # unchanged from striatum: fixes the Gram's row range
DD_LIST = [0, 1, 2, 3, 4, 5, 8, 9]
A_LIST = [0, 1, 2, 4, 8]
A_STORE = [0, 4, 8]            # per-diet columns: the event-aligned query and two later ones
L_STORE = [1, 2, 3, 4]
T_LO = 8
LAMS = [1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
BLOCKS = ["post_embed", "post_block7"]
OFFS = [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8]      # twin trace offsets relative to t_v


def tgt_names():
    real = [f"l{l}_d{dd}" for l in LEVELS for dd in DD_LIST]
    return real + [f"sh_{nm}" for nm in real]


@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=4 * 3600, memory=16384,
              max_containers=4)
def norm_ckpt(ckpt: str, stim_tag: str = "a1", n_clean: int = 6144,
              n_clean_val: int = 2048, max_windows: int = 0, seed: int = 11,
              tag: str = "", actor_steps: int = 3000, eval_seed: int = 5150,
              n_clean_crit: int = 8192, prim_block: str = "post_block7",
              n_val_windows: int = 1500, do_window_diets: bool = True,
              max_twins: int = 45000, caliper: float = 2.0, chunk: int = 512,
              strat_frac: float = 1.0 / 3.0, do_glitch: bool = False,
              var_target: str = ""):
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
    pL = S["pL_edit"][idx] if "pL_edit" in S.files else None
    print(f"stimuli {stim_tag}: n={n} etype={np.bincount(et, minlength=3).tolist()}", flush=True)

    # ---- splits over windows (junction's recipe verbatim: same seed, same n) ----------
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

    hx_path = ckpt[:-3] + f"_striatum_{stim_tag}_hexcess.npz"
    hexcess = None
    if os.path.exists(hx_path):
        hexcess = np.load(hx_path)["hexcess_edit"][idx]
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

    # ---- anchor positions ---------------------------------------------------------------
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

    # ---- the two forward passes -----------------------------------------------------------
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

    # ---- the two per-window cut variables --------------------------------------------------
    c_depth = cons_depth(et, jj)
    dmg_full = (o_o.astype(np.int16) - o_e.astype(np.int16)) > 0        # realised damage
    dmg_w = dmg_full[:4][:, :, rows]                                    # (4, n, R)
    dmg_win = dmg_w.mean((0, 2))                                        # per window
    out_win = o_e[:4][:, :, rows].astype(np.float32).mean((0, 2))       # the critic's teacher
    out_orig_win = o_o[:4][:, :, rows].astype(np.float32).mean((0, 2))

    # ---- the diets -------------------------------------------------------------------------
    masks, spec, wmeta = {}, {}, {}
    if do_window_diets:
        ms, sp, mt = strat_diets(out_win, et, jj, is_tr, seed=seed + 7, frac=strat_frac,
                                 prefix="out")
        masks.update(ms); spec.update(sp); wmeta.update(mt)
        n_strat = int(max(m.sum() for m in ms.values()))
        ms, sp, mt = strat_diets(dmg_win, et, jj, is_tr, seed=seed + 8, frac=strat_frac,
                                 prefix="dmg")
        masks.update(ms); spec.update(sp); wmeta.update(mt)
        ms, sp, mt = mix_diets(et, is_tr, n_total=n_strat, seed=seed + 9)
        masks.update(ms); spec.update(sp); wmeta.update(mt)
        ms, sp = anchor_diets(et, is_tr, n_total=n_strat, seed=seed + 10)
        masks.update(ms); spec.update(sp)
    else:
        ms, sp = anchor_diets(et, is_tr, n_total=0, seed=seed + 10)
        masks.update(ms); spec.update(sp)
    for k_ in list(masks):
        masks[k_] = np.repeat(masks[k_][tr_idx][:, None], R_, 1)
    diet_names = list(masks)
    print(f"{len(diet_names)} diets: {diet_names}", flush=True)

    # ---- what each diet's world actually looks like ------------------------------------------
    med_global = float(np.median(dmg_win[is_tr & (et < 2)])) if (et < 2).any() else 0.0
    o_row_tr = o_e[:4][:, tr_idx][:, :, rows].astype(np.float32).mean(0)     # (n_tr, R)
    d_row_tr = dmg_w.mean(0)[tr_idx]                                        # (n_tr, R)
    diet_stats = {}
    for nm in diet_names:
        m = np.asarray(masks[nm])
        wsel = tr_idx[m.any(1)]
        ed = wsel[et[wsel] < 2]
        sw, ra = et[ed] == 0, et[ed] == 1
        cells = np.bincount(cell_key(et[wsel], jj[wsel]).astype(int), minlength=300)
        st_ = {"n_rows": int(m.sum()), "n_windows": int(len(wsel)),
               "n_swap": int(sw.sum()), "n_rare": int(ra.sum()),
               "n_none": int((et[wsel] == 2).sum()),
               # the world's expected cost, on the rows the Gram actually sees
               "mean_outcome": float(o_row_tr[m].mean()),
               "mean_damage": float(d_row_tr[m].mean()),
               "mean_outcome_win": float(out_win[wsel].mean()),
               "mean_damage_win": float(dmg_win[wsel].mean()),
               "mean_outcome_orig_win": float(out_orig_win[wsel].mean()),
               "edit_share": float((et[wsel] < 2).mean()),
               "mean_j": float(jj[wsel].mean()), "mean_c_depth": float(c_depth[ed].mean())
               if len(ed) else float("nan"),
               "p_illegal_among_edits": float(sw.mean()) if len(ed) else float("nan"),
               "j_hist": np.bincount(jj[wsel], minlength=6).tolist(),
               "etype_hist": np.bincount(et[wsel], minlength=3).tolist(),
               "cell_hist": {int(c): int(cells[c]) for c in np.where(cells)[0]}}
        if len(ed) and sw.sum() and ra.sum():
            dw = dmg_win[ed]
            hi = dw > med_global
            st_["assoc"] = float(hi[sw].mean() - hi[ra].mean())
            st_["c_gap"] = float(c_depth[ed][sw].mean() - c_depth[ed][ra].mean())
            st_["dmg_mean_swap"] = float(dw[sw].mean())
            st_["dmg_mean_rare"] = float(dw[ra].mean())
        diet_stats[nm] = st_
    for nm in diet_names:
        print(f"  {nm:16s} n_win {diet_stats[nm]['n_windows']:6d} rows "
              f"{diet_stats[nm]['n_rows']:7d}  E[o] {diet_stats[nm]['mean_outcome']:.4f}  "
              f"E[dmg] {diet_stats[nm]['mean_damage']:.4f}  j {diet_stats[nm]['mean_j']:.3f} "
              f"etype {diet_stats[nm]['etype_hist']}", flush=True)

    # ---- Gram per diet ------------------------------------------------------------------------
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

    # ---- lambda selection on held-out val rows ---------------------------------------------
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
        # the fitted mean of the critic's target on its own diet (the norm's intercept scale)
        val_r2[nm]["_train_mean_l1_d0"] = float(
            (C[nm][-1, names.index("l1_d0")] / torch.clamp(A[nm][-1, -1], min=1.0)).item())
    del A, C, Xv, Yv
    torch.cuda.empty_cache()
    print(f"solves {time.time() - t0:.1f}s  full l1_d0 R2 "
          f"{val_r2.get('full', val_r2[diet_names[0]])['l1_d0']:.4f}", flush=True)

    # ---- 2026-09-17 addition: a mean-AND-VARIANCE critic on a CONTINUOUS target -------------
    # Xiang's insula row is a variance prediction error, which a mean-only ridge has no
    # analogue for.  On a 0/1 outcome a variance head is degenerate (Var = V(1 - V), a
    # function of the mean), so this needs a continuous target.  `var_target` picks one,
    # both already in the machinery:
    #   "hsum"  y_l(t) = sum_{delta=0..8} o_l(t + delta)   in {0..9}   (the horizon sum)
    #   "lmean" y(t)   = mean_{l=1..4} o_l(t)              in {0,.25,.5,.75,1}
    # Two ridge families on the SAME cached states, same lambda ladder, same val rows:
    # a mean head `Vc[c, dd]` and, on the training residuals, a variance head
    # `Vvar[c, dd] ~ (y - Vc)^2`.  The degeneracy gate is a quadratic in the mean head's
    # own output fitted on the same rows: if `Vvar` does not beat `[1, Vc, Vc^2]` there is
    # no variance signal beyond the mean and the column set is not worth reading.
    # Default off; with `var_target == ""` nothing below runs and every banked artefact is
    # reproduced bit-for-bit.
    var_res, var_beta, var_cols = {}, {}, []
    if var_target:
        t0 = time.time()
        VD = [0, 1]                      # the event offsets Vpre/V need
        if var_target == "hsum":
            var_cols = [(l, dd) for l in L_STORE for dd in VD]
            HSUM = 8

            def ytgt(li, src, base):
                a = o_e[li][np.ix_(src, base)].astype(np.float64)
                for k in range(1, HSUM + 1):
                    a = a + o_e[li][np.ix_(src, base + k)].astype(np.float64)
                return a
        elif var_target == "lmean":
            var_cols = [(0, dd) for dd in VD]

            def ytgt(li, src, base):
                return o_e[:4][:, src][:, :, base].astype(np.float64).mean(0)
        else:
            raise ValueError(f"var_target must be '', 'hsum' or 'lmean', got {var_target!r}")
        nvc = len(var_cols)

        def var_Y(src, dev_=None):
            return torch.stack([torch.as_tensor(ytgt(LEVELS.index(l) if l else 0, src,
                                                     rows + dd),
                                                dtype=torch.float64, device=dev)
                                for l, dd in var_cols], -1).reshape(-1, nvc)

        def var_gram(resid_beta=None):
            """One streamed pass: the mean Gram, or (given the mean betas) the Gram of the
            squared residual."""
            A_ = torch.zeros(d + 1, d + 1, dtype=torch.float64, device=dev)
            C_ = torch.zeros(d + 1, nvc, dtype=torch.float64, device=dev)
            for c0 in range(0, len(tr_idx), chunk):
                sl = slice(c0, min(c0 + chunk, len(tr_idx)))
                X = torch.as_tensor(cache[sl].reshape(-1, d), device=dev).double()
                X = torch.cat([X, torch.ones(len(X), 1, dtype=torch.float64,
                                             device=dev)], 1)
                Y = var_Y(tr_idx[sl])
                if resid_beta is not None:
                    Y = (Y - X @ resid_beta) ** 2
                A_ += X.T @ X
                C_ += X.T @ Y
                del X, Y
            return A_, C_

        Xv_ = torch.as_tensor(cache[len(tr_idx):].reshape(-1, d), device=dev).double()
        Xv_ = torch.cat([Xv_, torch.ones(len(Xv_), 1, dtype=torch.float64, device=dev)], 1)
        Yv_ = var_Y(va_idx)

        def var_solve(A_, C_, Yval):
            An_, Cn_ = A_.cpu().numpy(), C_.cpu().numpy()
            best_ = torch.full((nvc,), -1e18, dtype=torch.float64, device=dev)
            Bb_ = torch.zeros(d + 1, nvc, dtype=torch.float64, device=dev)
            yv_ = torch.clamp(Yval.var(0), min=1e-12)
            for lm in LAMS:
                B = torch.as_tensor(ridge_solve(An_, Cn_, lm), device=dev)
                r2 = 1.0 - ((Xv_ @ B - Yval) ** 2).mean(0) / yv_
                up = r2 > best_
                best_ = torch.where(up, r2, best_)
                Bb_[:, up] = B[:, up]
            return Bb_, best_

        Am, Cm = var_gram()
        Bm, r2m = var_solve(Am, Cm, Yv_)
        Av, Cv = var_gram(resid_beta=Bm)
        Rv_ = (Yv_ - Xv_ @ Bm) ** 2
        Bv, r2v = var_solve(Av, Cv, Rv_)
        # the degeneracy gate: the best quadratic in the mean head's own output, fitted on
        # the TRAINING rows the variance head saw, scored on the same val rows
        Gq = np.zeros((nvc, 3, 3))
        gq = np.zeros((nvc, 3))
        for c0 in range(0, len(tr_idx), chunk):
            sl = slice(c0, min(c0 + chunk, len(tr_idx)))
            X = torch.as_tensor(cache[sl].reshape(-1, d), device=dev).double()
            X = torch.cat([X, torch.ones(len(X), 1, dtype=torch.float64, device=dev)], 1)
            Vh = X @ Bm                                              # (rows, nvc)
            Rr = ((var_Y(tr_idx[sl]) - Vh) ** 2).cpu().numpy()
            Vh = Vh.cpu().numpy()
            Z = np.stack([np.ones_like(Vh), Vh, Vh ** 2], -1)        # (rows, nvc, 3)
            Gq += np.einsum("rca,rcb->cab", Z, Z)
            gq += np.einsum("rca,rc->ca", Z, Rr)
            del X, Vh, Z, Rr
        Vv_ = (Xv_ @ Bm).cpu().numpy()
        Zv = np.stack([np.ones_like(Vv_), Vv_, Vv_ ** 2], -1)
        Rvn = Rv_.cpu().numpy()
        deg = []
        for ci in range(nvc):
            bq = np.linalg.solve(Gq[ci] + 1e-8 * np.eye(3), gq[ci])
            num = float(((Zv[:, ci] @ bq - Rvn[:, ci]) ** 2).mean())
            deg.append(1.0 - num / float(max(Rvn[:, ci].var(), 1e-12)))
        var_beta = {"mean": Bm.cpu().numpy(), "var": Bv.cpu().numpy()}
        var_res = {"target": var_target, "cols": [f"l{l}_d{dd}" for l, dd in var_cols],
                   "r2_mean": [float(x) for x in r2m.cpu().numpy()],
                   "r2_var": [float(x) for x in r2v.cpu().numpy()],
                   "r2_var_from_mean_quadratic": deg,
                   "target_mean": [float(x) for x in Yv_.mean(0).cpu().numpy()],
                   "target_sd": [float(x) for x in Yv_.std(0).cpu().numpy()],
                   "resid2_mean": [float(x) for x in Rv_.mean(0).cpu().numpy()]}
        del Am, Cm, Av, Cv, Xv_, Yv_, Rv_
        torch.cuda.empty_cache()
        print(f"var head ({var_target}) {time.time() - t0:.1f}s  "
              f"R2 mean {var_res['r2_mean']}  R2 var {var_res['r2_var']}  "
              f"R2 var|quad(mean) {var_res['r2_var_from_mean_quadratic']}", flush=True)

    # ---- the clean-only critic (diet-independent floor) --------------------------------------
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
    clean_mean_o = float(o_cc[:4][:, :, rows].astype(np.float32).mean())
    del Acc, Ccc, Xvc, Yvc, Xcc_va, cache
    torch.cuda.empty_cache()
    print(f"clean-only critic {time.time() - t0:.1f}s  l1_d0 R2 {clean_val_r2['l1_d0']:.4f}"
          f"  E[o] {clean_mean_o:.4f}", flush=True)

    # ---- anchors, revisions, labels ------------------------------------------------------------
    def col(nm, name):
        return beta[nm][:, names.index(name) if nm != "clean" else real_names.index(name)]

    def lin(nm, name, X):
        b = col(nm, name)
        return X @ b[:-1] + b[-1]

    all_critics = diet_names + ["clean"]
    out_rows = {}
    for an, t0a in anchors.items():
        ok = anchor_ok[an]
        w = np.where(ok)[0]
        if len(w) < 50:
            continue
        ib, ia = ANC[an]
        t0w = t0a[w]
        rec = {"w": w, "t0": t0w, "etype": et[w], "j": jj[w], "k_star": ks[w], "e": ee[w],
               "split": split[w], "first_diff": fd[w], "t_v": tv[w], "c_depth": c_depth[w]}
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
        for nm in all_critics:
            for l in L_STORE:
                for a in A_STORE:
                    n0, n1 = f"l{l}_d{a}", f"l{l}_d{a + 1}"
                    rec[f"R_{nm}_l{l}_a{a}"] = lin(nm, n0, Xa) - lin(nm, n1, Xb_)
                    rec[f"Ro_{nm}_l{l}_a{a}"] = lin(nm, n0, Xao) - lin(nm, n1, Xbo)
                    rec[f"V_{nm}_l{l}_a{a}"] = lin(nm, n0, Xa)
                    rec[f"Vpre_{nm}_l{l}_a{a}"] = lin(nm, n1, Xb_)
                    rec[f"Vpreo_{nm}_l{l}_a{a}"] = lin(nm, n1, Xbo)
                    if nm != "clean":
                        rec[f"Rsh_{nm}_l{l}_a{a}"] = (lin(nm, f"sh_{n0}", Xa)
                                                      - lin(nm, f"sh_{n1}", Xb_))
        if var_target:
            # the mean-and-variance critic on the continuous target, at a = 0 only.
            # `vc_pre` is the norm in the continuous currency (the state at t0-1 predicting
            # the target AT t0), `vvar_pre` its expected squared error; the realised
            # `yc` gives the outcome surprise `dc = yc - vc_pre` and the variance
            # prediction error `vpe = dc^2 - vvar_pre`, which the reduction bins by `dc`.
            def _yc(li, pos):
                if var_target == "hsum":
                    a_ = np.zeros(len(pos), np.float64)
                    for k in range(HSUM + 1):
                        a_ += o_e[li][w, np.clip(pos + k, 0, T - 1)]
                    return a_
                return o_e[:4][:, w, np.clip(pos, 0, T - 1)].astype(np.float64).mean(0)

            ci_of = {(l, dd): i for i, (l, dd) in enumerate(var_cols)}
            rec["var_ok"] = ((t0w + (HSUM if var_target == "hsum" else 0))
                             <= T - 1).astype(np.int8)
            for l, dd in var_cols:
                if dd != 1:
                    continue
                i1, i0 = ci_of[(l, 1)], ci_of[(l, 0)]
                nl = f"l{l}"
                rec[f"vc_pre_{nl}"] = Xb_ @ var_beta["mean"][:-1, i1] + \
                    var_beta["mean"][-1, i1]
                rec[f"vc_{nl}"] = Xa @ var_beta["mean"][:-1, i0] + var_beta["mean"][-1, i0]
                rec[f"vvar_pre_{nl}"] = Xb_ @ var_beta["var"][:-1, i1] + \
                    var_beta["var"][-1, i1]
                rec[f"vvar_{nl}"] = Xa @ var_beta["var"][:-1, i0] + var_beta["var"][-1, i0]
                rec[f"yc_{nl}"] = _yc(LEVELS.index(l) if l else 0, t0w)
                rec[f"dc_{nl}"] = rec[f"yc_{nl}"] - rec[f"vc_pre_{nl}"]
                rec[f"vpe_{nl}"] = rec[f"dc_{nl}"] ** 2 - rec[f"vvar_pre_{nl}"]
                rec[f"Rc_{nl}"] = rec[f"vc_{nl}"] - rec[f"vc_pre_{nl}"]
                rec[f"Rvar_{nl}"] = rec[f"vvar_{nl}"] - rec[f"vvar_pre_{nl}"]
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
        # oracle probes on the same anchor states (the ceilings), junction's recipe
        Xab = np.concatenate([Xa, np.ones((len(Xa), 1))], 1)
        tr_m, va_m = rec["split"] == 0, rec["split"] == 1
        labs = {"probe_illegal": ((rec["etype"] == 0) & (rec["t_v"] > 0)).astype(np.float64),
                "probe_swap": (rec["etype"] == 0).astype(np.float64)}
        for l in LEVELS:
            for a in A_LIST:
                labs[f"probe_cons_l{l}_a{a}"] = rec[f"cons_l{l}_a{a}"].astype(np.float64)
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
        if banked is not None:
            rec["banked_excess"] = banked(st_e[banked.block][w, ia])
            rec["banked_excess_pre"] = banked(st_e[banked.block][w, ib])
        out_rows[an] = rec

    # ---- Q2: the same-prefix twins (phasic.py's construction) ------------------------------------
    twin, glitch = {}, {}
    if pL is not None and max_twins > 0:
        t0 = time.time()
        # the critic columns applied along the trace: (diet, level, dd in {0, 1})
        tcols = [(nm, l, dd) for nm in all_critics for l in L_STORE for dd in (0, 1)]
        Bmat = np.stack([col(nm, f"l{l}_d{dd}") for nm, l, dd in tcols], 1)      # (d+1, C)
        Bt = torch.as_tensor(Bmat[:-1], device=dev, dtype=torch.float32)
        Bt0 = torch.as_tensor(Bmat[-1], device=dev, dtype=torch.float32)
        offs = torch.as_tensor(OFFS, device=dev)

        # `caliper` here is the WIDE selection caliper: every pair's two surprisals are
        # saved, so the reduction can impose any narrower caliper (0.3 is the parent's)
        # without another pass.
        elig = np.where((et == 0) & (tv >= T_LO + 1) & (tv <= T - 1))[0]
        if len(elig) > max_twins:
            elig = np.sort(np.random.default_rng(seed + 21).choice(elig, max_twins,
                                                                   replace=False))

        def trace(W, tpos, bs=256, want_lsm=False):
            """V along the trace at t + OFFS for every (diet, level, dd) column."""
            Vt = np.zeros((len(W), len(OFFS), len(tcols)), np.float32)
            lsm_prev = np.zeros((len(W), v), np.float32) if want_lsm else None
            with torch.no_grad():
                for c0 in range(0, len(W), bs):
                    sl = slice(c0, min(c0 + bs, len(W)))
                    x = torch.as_tensor(W[sl, :T], device=dev)
                    lg, _, inter = model(x, return_intermediates=True)
                    tt = torch.as_tensor(tpos[sl], device=dev)
                    ar = torch.arange(x.shape[0], device=dev)
                    if want_lsm:
                        lsm_prev[sl] = torch.log_softmax(lg.float(), -1)[ar, tt - 1].cpu().numpy()
                    p = torch.clamp(tt[:, None] + offs[None, :], 0, T - 1)
                    Xs = inter[prim][ar[:, None], p].float()                 # (b, O, d)
                    Vt[sl] = (Xs @ Bt + Bt0).cpu().numpy()
            return Vt, lsm_prev

        tpos = tv[elig]
        V_v, lsm_v = trace(We[elig], tpos, want_lsm=True)
        x_v = We[elig, tpos]
        s_all = -lsm_v
        s_v = s_all[np.arange(len(elig)), x_v]
        legal = pL[elig, tpos] > 0
        assert not legal[np.arange(len(elig)), x_v].any(), "a violating token is legal"
        dist = np.where(legal, np.abs(s_all - s_v[:, None]), np.inf)
        x_c = dist.argmin(1)
        has = np.where(dist.min(1) <= caliper)[0]
        print(f"twins: eligible {len(elig)}  within caliper {caliper}: {len(has)}", flush=True)
        if len(has) >= 50:
            Wt = We[elig[has]].copy()
            Wt[np.arange(len(has)), tpos[has]] = x_c[has]
            V_c, lsm_c = trace(Wt, tpos[has], want_lsm=True)
            dev_prev = float(np.abs(V_c[:, 0] - V_v[has][:, 0]).max())
            dq = float(np.abs(lsm_c - lsm_v[has]).max())
            s_c = -lsm_c[np.arange(len(has)), x_c[has]]
            twin = {"n_eligible": int(len(elig)), "n_pairs": int(len(has)),
                    "caliper": float(caliper),
                    "check_Vpre_identical_max_abs": dev_prev,
                    "check_qprev_identical_max_abs": dq,
                    "mean_abs_ds": float(np.abs(s_v[has] - s_c).mean()),
                    "coverage_by_kstar": {int(k): [int((ks[elig] == k).sum()),
                                                   int((ks[elig[has]] == k).sum())]
                                          for k in np.unique(ks[elig])},
                    "cols": [[nm, int(l), int(dd)] for nm, l, dd in tcols],
                    "offs": OFFS}
            print(f"  Vpre identity {dev_prev:.2e}  q_prev identity {dq:.2e}  "
                  f"|ds| {twin['mean_abs_ds']:.4f}  ({time.time() - t0:.1f}s)", flush=True)
            twin_arr = {"w": elig[has], "t_v": tpos[has], "k_star": ks[elig[has]],
                        "j": jj[elig[has]], "e": ee[elig[has]], "split": split[elig[has]],
                        "x_v": x_v[has], "x_c": x_c[has], "s_v": s_v[has], "s_c": s_c,
                        "V_viol": V_v[has], "V_twin": V_c}
        else:
            twin = {"n_eligible": int(len(elig)), "n_pairs": int(len(has)),
                    "caliper": float(caliper), "note": "too few pairs"}
            twin_arr = {}

        # ---- addendum: the GLITCH world, same windows, same positions -----------------
        # `basalis/worlds.py`'s second world: the SAME window with no edit, the token at
        # the same `t_v` replaced.  The prefix is intact, so the continuation after the
        # event is the original stream -- coherent with a LEGAL token there and not with
        # the glitch, the mirror image of the swap world, where the regrown continuation
        # belongs to the violator.  `glitch_s`, not `glitch_m`: the glitch token is the
        # ILLEGAL-under-the-clean-prefix token whose model surprisal is nearest the edit's
        # violator, so the two worlds' events are matched in surprisal as well as in
        # window, position and (k*, j) stratum; a uniform draw is 1.4-2.1 nats more
        # surprising and would let the contrast win on the flagged token alone.
        plo_path = f"{ddir}/norm_pLorig_{stim_tag}.npz"
        if do_glitch and os.path.exists(plo_path) and twin_arr:
            t0 = time.time()
            PLO = np.load(plo_path)
            assert (PLO["t_v"] == S["t_v"]).all(), "norm_pLorig built on other stimuli"
            legal_o = PLO["pLo_tv"][idx][elig] > 0                      # (E, v)
            ar = np.arange(len(elig))
            V_o, lsm_o = trace(Wo[elig], tpos, want_lsm=True)
            s_o = -lsm_o
            x_or = Wo[elig, tpos]
            assert legal_o[ar, x_or].all(), "the unedited stream's own token is illegal"
            d_g = np.where(~legal_o, np.abs(s_o - s_v[:, None]), np.inf)
            x_g = d_g.argmin(1)
            ok_g = np.isfinite(d_g.min(1))
            s_g = s_o[ar, x_g]
            d_c = np.where(legal_o, np.abs(s_o - s_g[:, None]), np.inf)
            x_gc = d_c.argmin(1)
            hg = np.where(ok_g & np.isfinite(d_c.min(1)))[0]
            print(f"glitch: illegal token available {ok_g.mean():.3f}  pairs {len(hg)}",
                  flush=True)
            if len(hg) >= 50:
                Wg = Wo[elig[hg]].copy()
                Wg[np.arange(len(hg)), tpos[hg]] = x_g[hg]
                Wgc = Wo[elig[hg]].copy()
                Wgc[np.arange(len(hg)), tpos[hg]] = x_gc[hg]
                V_g, lsm_g = trace(Wg, tpos[hg], want_lsm=True)
                V_gc, _ = trace(Wgc, tpos[hg])
                dev_g = float(np.abs(V_g[:, 0] - V_gc[:, 0]).max())
                dev_o = float(np.abs(V_g[:, 0] - V_o[hg][:, 0]).max())
                dqg = float(np.abs(lsm_g - lsm_o[hg]).max())
                glitch = {"n_pairs": int(len(hg)),
                          "illegal_available_frac": float(ok_g.mean()),
                          "n_legal_at_tv_mean": float(legal_o.sum(1).mean()),
                          "check_Vpre_identical_max_abs": max(dev_g, dev_o),
                          "check_qprev_identical_max_abs": dqg,
                          "mean_abs_ds_within_pair":
                              float(np.abs(s_g[hg] - s_o[hg, x_gc[hg]]).mean()),
                          "mean_abs_ds_across_worlds":
                              float(np.abs(s_g[hg] - s_v[hg]).mean()),
                          "mean_surprisal_glitch": float(s_g[hg].mean()),
                          "mean_surprisal_violator": float(s_v[hg].mean()),
                          "mean_surprisal_orig_token": float(s_o[hg, x_or[hg]].mean()),
                          "p_glitch_equals_violator": float((x_g[hg] == x_v[hg]).mean()),
                          "cols": [[nm, int(l), int(dd)] for nm, l, dd in tcols],
                          "offs": OFFS}
                print(f"  glitch Vpre identity {glitch['check_Vpre_identical_max_abs']:.2e}"
                      f"  |ds| in-pair {glitch['mean_abs_ds_within_pair']:.4f}"
                      f"  across worlds {glitch['mean_abs_ds_across_worlds']:.4f}"
                      f"  ({time.time() - t0:.1f}s)", flush=True)
                glitch_arr = {"w": elig[hg], "t_v": tpos[hg], "k_star": ks[elig[hg]],
                              "j": jj[elig[hg]], "e": ee[elig[hg]],
                              "split": split[elig[hg]],
                              "x_v": x_g[hg], "x_c": x_gc[hg], "x_orig": x_or[hg],
                              "x_edit": x_v[hg],
                              "s_v": s_g[hg], "s_c": s_o[hg, x_gc[hg]],
                              "s_edit": s_v[hg], "s_orig": s_o[hg, x_or[hg]],
                              "V_viol": V_g, "V_twin": V_gc, "V_quiet": V_o[hg]}
            else:
                glitch = {"n_pairs": int(len(hg)), "note": "too few pairs"}
                glitch_arr = {}
        else:
            glitch, glitch_arr = ({} if not do_glitch else
                                  {"note": f"no {plo_path}"}), {}
    else:
        twin_arr = {}
        glitch, glitch_arr = {}, {}

    # ---- save -------------------------------------------------------------------------------------
    res = {"ckpt": ckpt, "step": step, "stim_tag": stim_tag, "n": n, "primary_block": prim,
           "critic_block": prim, "actor_clean_acc": actor_tab,
           "diet_spec": spec, "diet_stats": diet_stats, "diet_meta": wmeta,
           "median_damage_global": med_global, "twin": twin, "glitch": glitch,
           "critic_val_r2": {nm: {k: val_r2[nm][k] for k in
                                  [f"l{l}_d{dd}" for l in LEVELS for dd in (0, 1)]
                                  + [f"sh_l{l}_d0" for l in LEVELS]
                                  + ["_lam_l1_d0", "_train_mean_l1_d0"]}
                             for nm in diet_names},
           "clean_critic_val_r2": clean_val_r2, "clean_critic_mean_outcome": clean_mean_o,
           "banked_head": bpath, "hexcess": hx_path if hexcess is not None else None,
           "config": {"levels": LEVELS, "dmax": DMAX, "dd_list": DD_LIST, "a_list": A_LIST,
                      "a_store": A_STORE, "l_store": L_STORE, "t_lo": T_LO, "blocks": blocks,
                      "n_clean": n_clean, "seed": seed, "eval_seed": eval_seed,
                      "split": [0.6, 0.15, 0.25], "n_clean_crit": n_clean_crit,
                      "n_val_windows": int(len(va_idx)), "offs": OFFS, "caliper": caliper,
                      "max_twins": max_twins, "strat_frac": strat_frac,
                      "rows": [int(rows[0]), int(rows[-1])]}}
    if var_target:                      # additive: absent from every banked artefact
        res["var_head"] = var_res
        res["config"]["var_target"] = var_target
    stem = ckpt[:-3]
    sfx = f"_norm_{stim_tag}" + (f"_{tag}" if tag else "")
    with open(f"{stem}{sfx}.json", "w") as f:
        json.dump(res, f, cls=NumpyEncoder, indent=1)
    save = {}
    for an, rec in out_rows.items():
        tem = np.asarray(rec["split"]) == 2
        for k, vv in rec.items():
            a_ = np.asarray(vv)[tem]
            save[f"{an}__{k}"] = a_.astype(np.float32) if a_.dtype == np.float64 else a_
    for k, vv in twin_arr.items():
        a_ = np.asarray(vv)
        save[f"tw__{k}"] = a_.astype(np.float32) if a_.dtype == np.float64 else a_
    for k, vv in glitch_arr.items():
        a_ = np.asarray(vv)
        save[f"gl__{k}"] = a_.astype(np.float32) if a_.dtype == np.float64 else a_
    np.savez_compressed(f"{stem}{sfx}.npz", **save)
    volume.commit()
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    print(f"saved -> {stem}{sfx}.json  ({time.time() - t00:.0f}s, peak RSS {rss:.1f} GB)",
          flush=True)
    return {"ckpt": ckpt, "step": step, "stim_tag": stim_tag, "n_diets": len(diet_names),
            "actor_l1": actor_tab.get(f"{prim}/l1"),
            "n_pairs": twin.get("n_pairs"), "n_glitch_pairs": glitch.get("n_pairs"),
            "E_outcome": {nm: round(diet_stats[nm]["mean_outcome"], 4) for nm in diet_names}}


# ---------------------------------------------------------------------------
# addendum (2026-09-17): the glitch world needs the legal law under the CLEAN prefix
# ---------------------------------------------------------------------------

@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=2 * 3600, memory=16384)
def norm_pLorig(stim_tag: str = "swap65k", ckpt: str = "", chunk: int = 4096,
                oracle_chunk: int = 8):
    """`p_L(. | Wo_<t)` at each window's `t_v`, for the UNEDITED stream.

    `stimuli_swap65k.npz` was built with `with_reference=False`, so `pL_orig` is not on
    file there (it is for `a1`).  The glitch arm needs it: a glitch is a token that is
    impossible given the CLEAN prefix, and its legal twin is one that is possible given the
    same clean prefix.  Recomputed here with `stimuli.py`'s own recipe (`flat_predictive`
    at the full depth) and gated on the assertion `stimuli.py` makes: the token the
    unedited stream actually carries is legal at every position.

    Writes `.../logit_reading/norm_pLorig_<tag>.npz` with `pLo_tv` (n, v)."""
    import torch
    from rhm.logit_reading.flat_oracle import flat_predictive
    from rhm.logit_reading.calibration import load_trajectory_ckpt
    volume.reload()
    t00 = time.time()
    if not ckpt:
        ckpt = f"{DATA_DIR}/v16_s2_L6_m4_distinct/logit_reading/traj_a1_s42/step064000.pt"
    _, rules, rule_w, cfg = load_trajectory_ckpt(ckpt, "cuda")
    key = tb_key(cfg["v"], cfg["s"], cfg["L"], cfg["m"])
    ddir = f"{DATA_DIR}/{key}/logit_reading"
    S = np.load(f"{ddir}/stimuli_{stim_tag}.npz")
    Wo, tv, et = S["windows_orig"], S["t_v"], S["etype"]
    n, T1 = Wo.shape
    v, L = cfg["v"], cfg["L"]
    out = np.zeros((n, v), np.float32)
    ok_all = True
    for c0 in range(0, n, chunk):
        sl = slice(c0, min(c0 + chunk, n))
        pr = flat_predictive(Wo[sl], rules, L, device="cuda", chunk=oracle_chunk,
                             rule_w=rule_w).numpy()
        # the gate stimuli.py makes: the unedited stream never hits probability 0
        got = np.take_along_axis(pr, Wo[sl][..., None], -1)[..., 0][:, 1:]
        ok_all = ok_all and bool((got > 0).all())
        t = np.clip(tv[sl], 0, T1 - 1)
        out[sl] = pr[np.arange(sl.stop - sl.start), t]
        print(f"  {sl.stop}/{n}  {time.time() - t00:.0f}s", flush=True)
    sel = (et == 0) & (tv > 0)
    nleg = (out[sel] > 0).sum(1)
    orig_legal = out[sel][np.arange(int(sel.sum())), Wo[sel, tv[sel]]] > 0
    res = {"stim_tag": stim_tag, "n": int(n), "n_events": int(sel.sum()),
           "unedited_stream_always_legal": bool(ok_all),
           "orig_token_legal_at_tv": float(orig_legal.mean()),
           "legal_tokens_at_tv_mean": float(nleg.mean()),
           "legal_tokens_at_tv_hist": np.bincount(nleg, minlength=v + 1).tolist(),
           "illegal_available_frac": float((nleg < v).mean())}
    print(json.dumps(res, indent=1), flush=True)
    assert ok_all and orig_legal.all(), res
    np.savez_compressed(f"{ddir}/norm_pLorig_{stim_tag}.npz",
                        pLo_tv=out, t_v=tv, meta=json.dumps(res))
    volume.commit()
    print(f"saved -> {ddir}/norm_pLorig_{stim_tag}.npz ({time.time() - t00:.0f}s)", flush=True)
    return res


# ---------------------------------------------------------------------------
# coordinator: every cell in its own container
# ---------------------------------------------------------------------------

@app.function(volumes={DATA_DIR: volume}, timeout=6 * 3600, memory=2048)
def norm_sweep(cells: str = "", tag: str = "", n_clean: int = 6144, max_windows: int = 0,
               max_twins: int = 45000, caliper: float = 2.0, do_glitch: bool = False,
               var_target: str = ""):
    """cells: `ckpt:stim_tag:window_diets`, comma separated.

    `var_target` (default `""` = off) turns on the 2026-09-17 mean-and-variance critic on
    a continuous target; see `norm_ckpt`."""
    volume.reload()
    D = f"{DATA_DIR}/v16_s2_L6_m4_distinct/logit_reading"
    if not cells:
        cells = ",".join([
            f"{D}/traj_a1_s42/step064000.pt:a1:1",
            f"{D}/traj_a1_s42/step008000.pt:a1:1",
            f"{D}/traj_a1_s42/step064000.pt:swap65k:0",
            f"{D}/traj_a1_s42/step008000.pt:swap65k:0",
        ])
    args = []
    for c in cells.split(","):
        p = (c.split(":") + ["1"])[:3]
        ck, tg, wd = p[0], p[1], bool(int(p[2]))
        if not os.path.exists(ck):
            print(f"MISSING {ck}", flush=True)
            continue
        args.append((ck, tg, n_clean, 2048, max_windows, 11, tag, 3000, 5150, 8192,
                     "post_block7", 1500, wd, max_twins, caliper, 512, 1.0 / 3.0,
                     do_glitch, var_target))
    print(f"{len(args)} cells: {[(a[0].split('/')[-1], a[1], a[12]) for a in args]}", flush=True)
    outs = list(norm_ckpt.starmap(args, return_exceptions=True))
    for o in outs:
        print(json.dumps(o, default=str)[:800], flush=True)
    return [str(o)[:800] for o in outs]
