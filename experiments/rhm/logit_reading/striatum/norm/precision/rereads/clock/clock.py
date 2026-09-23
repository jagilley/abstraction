"""clock: the adaptation node's clock claim re-read through a reader that sees the belief.

`orbitofrontal/adaptation/` (two order seeds) read the norm as a RUNNING estimate: one
closed-form ridge critic on the raw state (`post_block7`), refit along a row ORDER that
switches tercile worlds, by five fitters (`win200`, `win800`, `fgt200`, `fgt800`, `cum`).
Its claims -- the from-low and from-high critics' outcome surprise on identical rows
converging on the fitter's memory clock with no time constant of their own (deviation from
the mechanical null at chance across orderings), the shift and the rescaling closing in
lockstep, the response carrying the history on the same clock, the outcome surprise on the
rows arriving next firing at the switch and returning -- were all made with the state
ridge.  `precision/express.py` found that a reader with `log q` appended (`fullQ`)
disagrees with the ridge on sign-level claims while agreeing on the norm's level.  This file
asks whether the clock claim holds for a reader that sees the belief.

It is `adaptation/task.py::adapt_ckpt` FORKED, with one addition: an arm selector for the
FEATURE MAP the critic reads (`--fmaps`):

  arm 0   state             the banked ridge, computed on the banked code path verbatim --
                            THE GATE: it must reproduce the banked cell (same seeds) before
                            anything is read; its outputs are compared member by member
                            against the banked `.npz` / `.json` in-container, and are NOT
                            re-saved (the reduction reads arm 0 off the banked files)
  arm Q   state + log q_t   the primary: the model's own output distribution at the state's
                            position (16 dims) appended
  arm L   ln_f(state)       the model's own final layer norm IN PLACE of the state -- exactly
                            what the unembedding reads (cheap second arm)
  arm H   state + H(q_t)    the belief's entropy appended (optional, one column)

Appended columns (Q, H) are standardised through the Gram exactly as `express.py`'s
`_standardiser` does -- mean 0 and the state's own per-dimension RMS on the fit's training
rows, applied as `A' = T' A T`, the coefficients mapped back so prediction uses the raw
features -- and the standardiser is computed from EACH FIT'S OWN GRAM (for `fgt` the
weighted Gram, `n_rows = A[-1, -1]` the weight sum), which is what "the training rows" of a
running fit are.  The per-checkpoint standardiser is saved.  `L` is fit in place without
standardisation, as `express.py`'s `L` arm is.

Everything else is `adaptation/task.py` verbatim, imported rather than copied where it is a
constant or a helper (`ARMS`, `FITTERS`, `BASE_OFF`, `ckpt_offsets`, `solve_pertarget`,
`_fit_AC`, the `LAMS` ladder, the split seed, the actor recipe, the tercile diets, the
per-cell trim, the order seeds, `block_win`, the twin construction).  The block Grams the
five fitters are built from carry each arm's features; the ridge penalty of each arm is
selected once per (fitter, target) on the fixed validation rows from the `out_mid2mid`
control at the end of phase A, exactly as arm 0's is, and held fixed everywhere.

Outputs (beside the banked cell, `sfx = _adapt_<stim>[_<tag>]`):
  `<stem><sfx>_clock.json`       config, the arm-0 gate, each arm's static fits and penalties
  `<stem><sfx>_clock<F>.npz`     arm F's betas and diagnostics under the BANKED key names
                                 (`beta_static_<diet>`, `beta_<arm>`, `diag_<arm>__<k>`), so
                                 `adaptation/analyze.py`'s reductions read them unchanged
  `<stem><sfx>_clockX.npz`       the extra features at the banked anchor rows and twin pairs
                                 (`log q`, `H(q)` at every saved state), and `ln_f`'s weights

Run (from a staged copy of `rhm/`'s .py files, see FILES rows):
  modal run -m rhm.logit_reading.striatum.norm.precision.rereads.clock.clock::clock_ckpt \
      --ckpt /data/v16_s2_L6_m4_distinct/logit_reading/traj_a1_s42/step064000.pt \
      --stim-tag a1 --max-windows 4000 --n-clean 1024 --actor-steps 300 \
      --max-twins 1500 --block-win 10 --tag smoke           # gates against the banked smoke
  modal run --detach -m rhm.logit_reading.striatum.norm.precision.rereads.clock.clock::clock_sweep
"""

import json
import os
import resource
import time

import numpy as np

import modal

from rhm.shared import volume, DATA_DIR, NumpyEncoder
from rhm.logit_reading.calibration import load_trajectory_ckpt, tb_key
from rhm.logit_reading.striatum.task import train_actor, actor_apply, ridge_solve
from rhm.logit_reading.striatum.junction.diets import cons_depth
from rhm.logit_reading.striatum.norm.diets import strat_diets, anchor_diets, cell_key
from rhm.logit_reading.orbitofrontal.adaptation.task import (
    LEVELS, DMAX, T_LO, L_STORE, A_STORE, A_LIST, DD_STORE, LAMS, PRIM, STATIC, ARMS,
    FITTERS, tgt_names, ckpt_offsets, solve_pertarget, _fit_AC, _gate)
from rhm.logit_reading.striatum.norm.precision.express import _standardiser


def _ignore(path):
    sp = str(path)
    if "/results/" in sp or sp.endswith("/results") or sp.endswith(".log"):
        return True
    return not sp.endswith(".py")


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==1.26.4", "scipy==1.16.3", "torch==2.7.0")
    .add_local_python_source("rhm", ignore=_ignore)
)
app = modal.App("rhm-norm-precision-clock", image=image)

FMAP_DESC = {"Q": "state + log q_t (16 dims), appended columns standardised per fit",
             "L": "ln_f(state) in place of the state (the model's final layer norm)",
             "H": "state + H(q_t), appended column standardised per fit"}
DIAG_KEYS = ("val_r2", "delta_val", "delta_val_lo", "delta_val_mid", "delta_val_hi",
             "delta_next", "delta_next_sem", "n_rows")


def ln_np(X, w, b, eps):
    """`ln_f` in float64 numpy -- the reduction's copy is checked against torch in-container."""
    X = np.asarray(X, np.float64)
    mu = X.mean(-1, keepdims=True)
    var = ((X - mu) ** 2).mean(-1, keepdims=True)
    return (X - mu) / np.sqrt(var + eps) * np.asarray(w, np.float64) + np.asarray(b, np.float64)


def ent_np(lq):
    lq = np.asarray(lq, np.float64)
    return -(np.exp(lq) * lq).sum(-1)


def _std_T(fm, A, d):
    """The standardiser of arm `fm` for one fit's own Gram (None = fit in place)."""
    if fm == "L":
        return None
    return _standardiser(A, d, float(A[-1, -1]))


def _solve_fm(fm, A, C, lam, d, per_target=True):
    """Ridge in the standardised basis, mapped back to the raw features.  Returns (B, T)."""
    T = _std_T(fm, A, d)
    if T is None:
        B = solve_pertarget(A, C, lam) if per_target else ridge_solve(A, C, lam)
        return B, None
    As, Cs = T.T @ A @ T, T.T @ C
    B = solve_pertarget(As, Cs, lam) if per_target else ridge_solve(As, Cs, lam)
    return T @ B, T


@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=2 * 3600, memory=12288,
              cpu=2.0, max_containers=4)
def clock_ckpt(ckpt: str, stim_tag: str = "a1", n_clean: int = 6144,
               n_clean_val: int = 2048, max_windows: int = 0, seed: int = 11,
               tag: str = "", actor_steps: int = 3000, eval_seed: int = 5150,
               n_val_windows: int = 1500, max_twins: int = 45000,
               caliper: float = 0.6, chunk: int = 512, strat_frac: float = 1.0 / 3.0,
               block_win: int = 25, anchors_keep: str = "tv,fd", arms: str = "",
               order_seed: int = 909, fmaps: str = "Q,L,H", tol: float = 5e-5):
    import torch
    volume.reload()
    t00 = time.time()
    tim = {}
    dev = "cuda"
    model, rules, rule_w, cfg = load_trajectory_ckpt(ckpt, dev)
    L, s, v = cfg["L"], cfg["s"], cfg["v"]
    T = model.block_size
    step = cfg.get("step")
    d = model.transformer.wte.weight.shape[1]
    FM = [f for f in fmaps.split(",") if f]
    assert all(f in FMAP_DESC for f in FM), FM
    lnf = model.transformer.ln_f
    fdim = {"Q": d + v, "L": d, "H": d + 1}
    print(f"loaded {ckpt} step {step} d={d} T={T}  feature arms {FM}", flush=True)

    key = tb_key(cfg["v"], cfg["s"], cfg["L"], cfg["m"])
    ddir = f"{DATA_DIR}/{key}/logit_reading"
    S = np.load(f"{ddir}/stimuli_{stim_tag}.npz")
    P_ = np.load(f"{ddir}/parse_{stim_tag}.npz")
    n_all = S["windows_edit"].shape[0]
    idx = np.arange(n_all)
    if max_windows and max_windows < n_all:
        idx = np.sort(np.random.default_rng(seed).choice(n_all, max_windows, replace=False))
    We, Wo = S["windows_edit"][idx], S["windows_orig"][idx]
    meta = {k: S[k][idx] for k in ("phase", "etype", "j", "e", "first_diff", "t_v", "k_star")}
    y_edit = P_["y_edit"][:, idx][:, :, :T]
    y_orig = P_["y_orig"][:, idx][:, :, :T]
    n = len(idx)
    et, jj, ee, fd, tv, ks = (meta["etype"], meta["j"], meta["e"], meta["first_diff"],
                              meta["t_v"], meta["k_star"])
    pL = S["pL_edit"][idx] if "pL_edit" in S.files else None
    print(f"stimuli {stim_tag}: n={n} etype={np.bincount(et, minlength=3).tolist()}",
          flush=True)

    # ---- splits over windows (norm's recipe verbatim: same seed, same n) --------------
    rng = np.random.default_rng(seed + 1)
    perm = rng.permutation(n)
    n_tr, n_va = int(0.60 * n), int(0.15 * n)
    split = np.zeros(n, np.int8)
    split[perm[n_tr:n_tr + n_va]] = 1
    split[perm[n_tr + n_va:]] = 2
    is_tr, is_va, is_te = split == 0, split == 1, split == 2
    tr_idx = np.where(is_tr)[0]
    va_idx = np.where(is_va)[0][:n_val_windows]

    # ---- clean windows -> the frozen actor (norm's recipe verbatim) -------------------
    from rhm.logit_reading.altitude.units import windows_with_parse
    from rhm.logit_reading.striatum.parse import clean_answers
    t0 = time.time()
    n_cl = n_clean + n_clean_val
    wins_c, phase_c, _, fs_c = windows_with_parse(rules, rule_w, n_cl, T + 1, eval_seed)
    y_c = clean_answers(fs_c, phase_c, T + 1, L, s, 3)[:, :, :T]
    pos = np.arange(T_LO, T)
    actors, actor_tab = {}, {}
    with torch.no_grad():
        Sc = np.zeros((n_cl, T, d), np.float32)
        for c0 in range(0, n_cl, 256):
            x = torch.as_tensor(wins_c[c0:c0 + 256, :T], device=dev)
            _, _, inter = model(x, return_intermediates=True)
            Sc[c0:c0 + 256] = inter[PRIM].float().cpu().numpy()
    Xtr = torch.as_tensor(Sc[:n_clean][:, pos].reshape(-1, d), device=dev)
    Xva = torch.as_tensor(Sc[n_clean:][:, pos].reshape(-1, d), device=dev)
    for l in LEVELS:
        ytr = torch.as_tensor(y_c[l - 1][:n_clean][:, pos].reshape(-1).astype(np.int64),
                              device=dev)
        yva = torch.as_tensor(y_c[l - 1][n_clean:][:, pos].reshape(-1).astype(np.int64),
                              device=dev)
        h = train_actor(Xtr, ytr, Xva, yva, steps=actor_steps, seed=l, device=dev)
        actors[l] = h
        actor_tab[f"{PRIM}/l{l}"] = h["acc"]
    del Xtr, Xva, Sc
    torch.cuda.empty_cache()
    tim["actors"] = time.time() - t0
    print(f"actors {time.time() - t0:.1f}s", flush=True)
    print(json.dumps({k: round(x, 4) for k, x in actor_tab.items()}), flush=True)

    # ---- anchor positions (norm's) ----------------------------------------------------
    anchors = {"fd": np.where(fd >= 0, fd, ee), "tv": np.where(tv > 0, tv, -1)}
    anchor_ok = {"fd": ((fd >= T_LO + 1) | ((et == 2) & (ee >= T_LO + 1))) & (fd <= T - 1),
                 "tv": (tv > T_LO) & (et == 0) & (tv <= T - 1)}
    anc_pos = np.stack([np.clip(anchors["fd"] - 1, 0, T - 1), np.clip(anchors["fd"], 0, T - 1),
                        np.clip(anchors["tv"] - 1, 0, T - 1), np.clip(anchors["tv"], 0, T - 1)], 1)
    ANC = {"fd": (0, 1), "tv": (2, 3)}
    keep_anch = [a for a in anchors if a in anchors_keep.split(",")]

    rows = np.arange(T_LO, T - 1 - DMAX + 1)
    R_ = len(rows)
    cache_idx = np.concatenate([tr_idx, va_idx])
    cache_map = -np.ones(n, np.int64)
    cache_map[cache_idx] = np.arange(len(cache_idx))

    # ---- the two forward passes (norm's), plus log q / H(q) where the state is kept ---
    def run(W, Y, want_cache=False, bs=256):
        o = np.zeros((len(LEVELS), len(W), T), np.uint8)
        nll = np.zeros((len(W), T), np.float32)
        Hq = np.zeros((len(W), T), np.float32)
        anc = np.zeros((len(W), 4, d), np.float32)
        cache = (np.zeros((len(cache_idx), R_, d), np.float16) if want_cache else None)
        # [clock] the belief at every saved state
        lsm_anc = np.zeros((len(W), 4, v), np.float32)
        lq_cache = (np.zeros((len(cache_idx), R_, v), np.float32) if want_cache else None)
        h_cache = (np.zeros((len(cache_idx), R_), np.float32) if want_cache else None)
        with torch.no_grad():
            for c0 in range(0, len(W), bs):
                sl = slice(c0, min(c0 + bs, len(W)))
                x = torch.as_tensor(W[sl, :T], device=dev)
                lg, _, inter = model(x, return_intermediates=True)
                lsm = torch.log_softmax(lg.float(), -1)
                nxt = torch.as_tensor(W[sl, 1:T + 1], device=dev)
                nll[sl, :] = (-lsm.gather(-1, nxt[..., None])[..., 0]).cpu().numpy()
                Hq[sl, :] = (-(lsm.exp() * lsm).sum(-1)).cpu().numpy()
                Xp = inter[PRIM].float()
                for li, l in enumerate(LEVELS):
                    pred = actor_apply(actors[l], Xp)
                    tru = torch.as_tensor(Y[li][sl].astype(np.int64), device=dev)
                    o[li, sl] = (pred == tru).to(torch.uint8).cpu().numpy()
                ap = torch.as_tensor(anc_pos[sl], device=dev)
                ar = torch.arange(x.shape[0], device=dev)[:, None]
                anc[sl] = inter[PRIM][ar, ap].float().cpu().numpy()
                lsm_anc[sl] = lsm[ar, ap].cpu().numpy()
                if want_cache:
                    kk = np.where(cache_map[c0:sl.stop] >= 0)[0]
                    if len(kk):
                        kt = torch.as_tensor(kk, device=dev)
                        cache[cache_map[c0 + kk]] = \
                            inter[PRIM][kt][:, rows].half().cpu().numpy()
                        lq_cache[cache_map[c0 + kk]] = lsm[kt][:, rows].cpu().numpy()
                        h_cache[cache_map[c0 + kk]] = Hq[c0 + kk][:, rows]
        return o, nll, Hq, anc, cache, lsm_anc, lq_cache, h_cache

    t0 = time.time()
    o_o, nll_o, H_o, st_o, _, lsm_ao, _, _ = run(Wo, y_orig)
    print(f"orig pass {time.time() - t0:.1f}s", flush=True)
    t0 = time.time()
    o_e, nll_e, H_e, st_e, cache, lsm_ae, lq_cache, h_cache = run(We, y_edit, want_cache=True)
    tim["passes"] = time.time() - t0
    print(f"edit pass {time.time() - t0:.1f}s  cache {cache.nbytes / 1e9:.2f} GB  "
          f"log q cache {lq_cache.nbytes / 1e9:.2f} GB", flush=True)

    # ---- the diets (norm's seeds, so the masks are norm's) ----------------------------
    c_depth = cons_depth(et, jj)
    dmg_full = (o_o.astype(np.int16) - o_e.astype(np.int16)) > 0
    dmg_w = dmg_full[:4][:, :, rows]
    dmg_win = dmg_w.mean((0, 2))
    out_win = o_e[:4][:, :, rows].astype(np.float32).mean((0, 2))

    masks, spec, wmeta = {}, {}, {}
    ms, sp, mt = strat_diets(out_win, et, jj, is_tr, seed=seed + 7, frac=strat_frac,
                             prefix="out")
    masks.update(ms); spec.update(sp); wmeta.update(mt)
    n_strat = int(max(m.sum() for m in ms.values()))
    ms, sp, mt = strat_diets(dmg_win, et, jj, is_tr, seed=seed + 8, frac=strat_frac,
                             prefix="dmg")
    masks.update(ms); spec.update(sp); wmeta.update(mt)
    ms, sp = anchor_diets(et, is_tr, n_total=n_strat, seed=seed + 10)
    masks.update(ms); spec.update(sp)
    vm, _, _ = strat_diets(out_win, et, jj, is_va, seed=seed + 7, frac=strat_frac,
                           prefix="vout")
    masks.update(vm)

    o_row_tr = o_e[:4][:, tr_idx][:, :, rows].astype(np.float32).mean(0)     # (n_tr, R_)
    d_row_tr = dmg_w.mean(0)[tr_idx]
    diet_stats = {}
    for nm in [x for x in masks if not x.startswith("vout")]:
        m = masks[nm][tr_idx]
        wsel = tr_idx[m]
        cells = np.bincount(cell_key(et[wsel], jj[wsel]).astype(int), minlength=400)
        diet_stats[nm] = {
            "n_windows": int(len(wsel)), "n_rows": int(m.sum()) * R_,
            "n_swap": int((et[wsel] == 0).sum()), "n_rare": int((et[wsel] == 1).sum()),
            "n_none": int((et[wsel] == 2).sum()),
            "mean_outcome": float(o_row_tr[m].mean()),
            "mean_damage": float(d_row_tr[m].mean()),
            "mean_outcome_win": float(out_win[wsel].mean()),
            "mean_damage_win": float(dmg_win[wsel].mean()),
            "mean_j": float(jj[wsel].mean()),
            "mean_c_depth": float(c_depth[wsel[et[wsel] < 2]].mean())
            if (et[wsel] < 2).any() else float("nan"),
            "j_hist": np.bincount(jj[wsel], minlength=8).tolist(),
            "etype_hist": np.bincount(et[wsel], minlength=3).tolist(),
            "cell_hist": {int(c): int(cells[c]) for c in np.where(cells)[0]}}

    names = tgt_names()
    n_tgt = len(names)

    # ---- the fixed validation rows (diet-independent, shared by every readout) --------
    Xv_np = cache[len(tr_idx):].reshape(-1, d)
    Xv = torch.cat([torch.as_tensor(Xv_np, device=dev).float(),
                    torch.ones(len(Xv_np), 1, device=dev)], 1)
    Yv = torch.stack([torch.as_tensor(o_e[l - 1][np.ix_(va_idx, rows + dd)],
                                      dtype=torch.float32, device=dev)
                      for l in L_STORE for dd in DD_STORE], -1).reshape(-1, n_tgt)
    yv_var = torch.clamp(Yv.var(0), min=1e-12)
    vworld = {}
    for k in ("lo", "mid", "hi"):
        mv = masks[f"vout_{k}"][va_idx]
        vworld[k] = torch.as_tensor(np.repeat(mv[:, None], R_, 1).reshape(-1), device=dev)

    def val_stats(B):
        """held-out R2, mean residual overall and per world, per target."""
        Bt = torch.as_tensor(B, device=dev, dtype=torch.float32)
        res = Yv - Xv @ Bt
        out = {"val_r2": (1.0 - (res ** 2).mean(0) / yv_var).cpu().numpy(),
               "delta_val": res.mean(0).cpu().numpy()}
        for k, mk in vworld.items():
            out[f"delta_val_{k}"] = res[mk].mean(0).cpu().numpy() if int(mk.sum()) else \
                np.full(n_tgt, np.nan)
        return out

    # ---- [clock] the feature maps -------------------------------------------------------
    def feat_t(fm, Xs, lq, hh):
        """Arm `fm`'s features (no bias) from the float32 state, log q and H(q) on the GPU."""
        if fm == "Q":
            return torch.cat([Xs, lq], 1)
        if fm == "H":
            return torch.cat([Xs, hh[:, None]], 1)
        with torch.no_grad():
            return lnf(Xs)

    Xv_s = Xv[:, :d]
    lqv = torch.as_tensor(lq_cache[len(tr_idx):].reshape(-1, v), device=dev)
    hv = torch.as_tensor(h_cache[len(tr_idx):].reshape(-1), device=dev)
    XvF = {fm: torch.cat([feat_t(fm, Xv_s, lqv, hv).float(),
                          torch.ones(len(Xv_s), 1, device=dev)], 1) for fm in FM}
    del lqv, hv

    def val_stats_F(fm, B):
        """`val_stats` with arm `fm`'s features in place of the state."""
        Bt = torch.as_tensor(B, device=dev, dtype=torch.float32)
        res = Yv - XvF[fm] @ Bt
        out = {"val_r2": (1.0 - (res ** 2).mean(0) / yv_var).cpu().numpy(),
               "delta_val": res.mean(0).cpu().numpy()}
        for k, mk in vworld.items():
            out[f"delta_val_{k}"] = res[mk].mean(0).cpu().numpy() if int(mk.sum()) else \
                np.full(n_tgt, np.nan)
        return out

    # ---- the static reference critics (norm's banked closed-form fits) ---------------
    t0 = time.time()
    A_s = {nm: torch.zeros(d + 1, d + 1, dtype=torch.float64, device=dev) for nm in STATIC}
    C_s = {nm: torch.zeros(d + 1, n_tgt, dtype=torch.float64, device=dev) for nm in STATIC}
    A_sF = {fm: {nm: torch.zeros(fdim[fm] + 1, fdim[fm] + 1, dtype=torch.float64, device=dev)
                 for nm in STATIC} for fm in FM}
    C_sF = {fm: {nm: torch.zeros(fdim[fm] + 1, n_tgt, dtype=torch.float64, device=dev)
                 for nm in STATIC} for fm in FM}
    smask = {nm: np.repeat(masks[nm][tr_idx][:, None], R_, 1) for nm in STATIC}
    for c0 in range(0, len(tr_idx), chunk):
        sl = slice(c0, min(c0 + chunk, len(tr_idx)))
        wsl = tr_idx[sl]
        X = torch.as_tensor(cache[sl].reshape(-1, d), device=dev).double()
        X = torch.cat([X, torch.ones(len(X), 1, dtype=torch.float64, device=dev)], 1)
        Y = torch.stack([torch.as_tensor(o_e[l - 1][np.ix_(wsl, rows + dd)],
                                         dtype=torch.float64, device=dev)
                         for l in L_STORE for dd in DD_STORE], -1).reshape(-1, n_tgt)
        for nm in STATIC:
            sel = torch.as_tensor(np.where(smask[nm][sl].reshape(-1))[0], device=dev)
            if not len(sel):
                continue
            Xs = X[sel]
            A_s[nm] += Xs.T @ Xs
            C_s[nm] += Xs.T @ Y[sel]
        # [clock] the same rows through each feature map
        Xs32 = torch.as_tensor(cache[sl].reshape(-1, d), device=dev).float()
        lqc = torch.as_tensor(lq_cache[sl].reshape(-1, v), device=dev)
        hc = torch.as_tensor(h_cache[sl].reshape(-1), device=dev)
        for fm in FM:
            F = feat_t(fm, Xs32, lqc, hc).double()
            F = torch.cat([F, torch.ones(len(F), 1, dtype=torch.float64, device=dev)], 1)
            for nm in STATIC:
                sel = torch.as_tensor(np.where(smask[nm][sl].reshape(-1))[0], device=dev)
                if not len(sel):
                    continue
                Fs = F[sel]
                A_sF[fm][nm] += Fs.T @ Fs
                C_sF[fm][nm] += Fs.T @ Y[sel]
            del F
        del X, Y, Xs32, lqc, hc
    beta_s, static_info = {}, {}
    for nm in STATIC:
        An, Cn = A_s[nm].cpu().numpy(), C_s[nm].cpu().numpy()
        best = np.full(n_tgt, -1e18)
        lam_pick = np.zeros(n_tgt)
        Bb = np.zeros((d + 1, n_tgt))
        for lm in LAMS:
            B = ridge_solve(An, Cn, lm)
            r2 = val_stats(B)["val_r2"]
            up = r2 > best
            best = np.where(up, r2, best)
            Bb[:, up] = B[:, up]
            lam_pick[up] = lm
        beta_s[nm] = Bb
        vs = val_stats(Bb)
        static_info[nm] = {"val_r2": {names[i]: float(best[i]) for i in range(n_tgt)},
                           "lam": {names[i]: float(lam_pick[i]) for i in range(n_tgt)},
                           "delta_val": {names[i]: float(vs["delta_val"][i])
                                         for i in range(n_tgt)},
                           "train_mean": {names[i]: float(
                               (C_s[nm][-1, i] / torch.clamp(A_s[nm][-1, -1], min=1.0)).item())
                               for i in range(n_tgt)},
                           "n_rows": int(A_s[nm][-1, -1].item())}
    # [clock] each arm's own static critics: same diets, same ladder, same selection
    beta_sF, static_infoF = {fm: {} for fm in FM}, {fm: {} for fm in FM}
    for fm in FM:
        for nm in STATIC:
            An, Cn = A_sF[fm][nm].cpu().numpy(), C_sF[fm][nm].cpu().numpy()
            best = np.full(n_tgt, -1e18)
            lam_pick = np.zeros(n_tgt)
            Bb = np.zeros((fdim[fm] + 1, n_tgt))
            for lm in LAMS:
                B, Tst = _solve_fm(fm, An, Cn, lm, d, per_target=False)
                r2 = val_stats_F(fm, B)["val_r2"]
                up = r2 > best
                best = np.where(up, r2, best)
                Bb[:, up] = B[:, up]
                lam_pick[up] = lm
            beta_sF[fm][nm] = Bb
            vs = val_stats_F(fm, Bb)
            Tst = _std_T(fm, An, d)
            static_infoF[fm][nm] = {
                "val_r2": {names[i]: float(best[i]) for i in range(n_tgt)},
                "lam": {names[i]: float(lam_pick[i]) for i in range(n_tgt)},
                "delta_val": {names[i]: float(vs["delta_val"][i]) for i in range(n_tgt)},
                "n_rows": int(An[-1, -1]),
                "std_scale": (None if Tst is None else np.diag(Tst)[d:-1].tolist()),
                "std_shift": (None if Tst is None else Tst[-1, d:-1].tolist())}
    del A_s, C_s, A_sF, C_sF
    torch.cuda.empty_cache()
    tim["static"] = time.time() - t0
    print(f"static critics {time.time() - t0:.1f}s  full l1_d0 R2 "
          f"{static_info['full']['val_r2']['l1_d0']:.4f}  "
          + "  ".join(f"{fm} {static_infoF[fm]['full']['val_r2']['l1_d0']:.4f}" for fm in FM),
          flush=True)

    # ---- the arms: a diet is a row ORDER, not one accumulation (verbatim) -------------
    arm_list = [a for a in ARMS if not arms or a[0] in arms.split(",")]
    B_ = int(block_win)
    diet_w = {nm: np.where(masks[nm][tr_idx])[0] for nm in
              ["out_lo", "out_mid", "out_hi", "dmg_lo", "dmg_mid", "dmg_hi"]}
    d_cells = {nm: cell_key(et[tr_idx[w]], jj[tr_idx[w]]) for nm, w in diet_w.items()}
    uniq = np.unique(np.concatenate(list(d_cells.values())))
    base = {int(c): min(int((d_cells[nm] == c).sum()) for nm in d_cells) for c in uniq}
    cell_equal = all(int((d_cells[nm] == c).sum()) == base[int(c)]
                     for nm in d_cells for c in uniq)
    P_win = sum(base.values()) // B_ * B_
    if P_win < 2 * B_:
        raise RuntimeError(f"phase too short: {P_win} windows at block {B_}")
    target = dict(base)
    n_drop = sum(base.values()) - P_win
    order = sorted(base, key=lambda c: (-base[c], c))
    i_ = 0
    while n_drop > 0:
        c = order[i_ % len(order)]
        if target[c] > 0:
            target[c] -= 1
            n_drop -= 1
        i_ += 1
    diet_keep = {}
    for nm, w in diet_w.items():
        sel = [np.random.default_rng(order_seed + 7919 * int(c)).permutation(
            w[d_cells[nm] == c])[:target[int(c)]] for c in uniq if target[int(c)] > 0]
        diet_keep[nm] = np.sort(np.concatenate(sel))
        assert len(diet_keep[nm]) == P_win, (nm, len(diet_keep[nm]), P_win)
    offs = ckpt_offsets(P_win, B_)
    ck_blk = [(P_win + o) // B_ for o in offs]
    print(f"phase {P_win} windows (cell counts equal across diets: {cell_equal}), "
          f"block {B_}, {len(offs)} checkpoints: {offs}", flush=True)

    def build_seq(a, b, mode, s0):
        r = np.random.default_rng(order_seed + s0)
        wa = r.permutation(diet_keep[a])
        wb = r.permutation(diet_keep[b])
        if mode == "interleave":
            pool = r.permutation(np.concatenate([wa, wb]))
            return pool, pool[:P_win], pool[P_win:]
        return np.concatenate([wa, wb]), wa, wb

    lam_glob = {}
    lam_globF = {fm: {} for fm in FM}
    win_blk = {nm: max(1, m // B_) for nm, kind, m in FITTERS if kind == "win"}
    fgt_lam = {nm: float(np.exp(-1.0 / (m * R_))) for nm, kind, m in FITTERS if kind == "fgt"}
    need_cum = sorted({cb for cb in ck_blk} |
                      {max(cb - nb, 0) for cb in ck_blk for nb in win_blk.values()})
    prof = {"arm0_grams": 0.0, "fmap_grams": 0.0, "arm0_solves": 0.0, "fmap_solves": 0.0}

    def run_arm(seq_w, select_lam=False):
        """One pass over the sequence: block Grams (arm 0 verbatim, then each feature map),
        snapshots, then a solve per checkpoint per fitter per arm."""
        nblk = len(seq_w) // B_
        G = B_ * R_
        cumA = torch.zeros(d + 1, d + 1, dtype=torch.float64, device=dev)
        cumC = torch.zeros(d + 1, n_tgt, dtype=torch.float64, device=dev)
        fA = {k: torch.zeros(d + 1, d + 1, dtype=torch.float64, device=dev) for k in fgt_lam}
        fC = {k: torch.zeros(d + 1, n_tgt, dtype=torch.float64, device=dev) for k in fgt_lam}
        wgt = {k: torch.as_tensor(lm ** (G - 1 - np.arange(G)), dtype=torch.float64,
                                  device=dev)[:, None] for k, lm in fgt_lam.items()}
        decay = {k: lm ** G for k, lm in fgt_lam.items()}
        snapA, snapC, snapF = {}, {}, {}
        nxt = {}
        ck_set = set(ck_blk)
        cum_set = set(need_cum)
        zeroA = np.zeros((d + 1, d + 1)); zeroC = np.zeros((d + 1, n_tgt))
        # [clock] the same structures per feature map
        DF = {fm: fdim[fm] + 1 for fm in FM}
        cumAF = {fm: torch.zeros(DF[fm], DF[fm], dtype=torch.float64, device=dev) for fm in FM}
        cumCF = {fm: torch.zeros(DF[fm], n_tgt, dtype=torch.float64, device=dev) for fm in FM}
        fAF = {fm: {k: torch.zeros(DF[fm], DF[fm], dtype=torch.float64, device=dev)
                    for k in fgt_lam} for fm in FM}
        fCF = {fm: {k: torch.zeros(DF[fm], n_tgt, dtype=torch.float64, device=dev)
                    for k in fgt_lam} for fm in FM}
        snapAF = {fm: {} for fm in FM}
        snapCF = {fm: {} for fm in FM}
        snapFF = {fm: {} for fm in FM}
        nxtF = {}
        for k in range(nblk):
            ta = time.time()
            wb = seq_w[k * B_:(k + 1) * B_]
            gw = tr_idx[wb]
            Xf = torch.as_tensor(cache[wb].reshape(-1, d), device=dev).float()
            Xf = torch.cat([Xf, torch.ones(len(Xf), 1, device=dev)], 1)
            Yf = torch.stack([torch.as_tensor(o_e[l - 1][np.ix_(gw, rows + dd)],
                                              dtype=torch.float32, device=dev)
                              for l in L_STORE for dd in DD_STORE], -1).reshape(-1, n_tgt)
            X = Xf.double()
            Y = Yf.double()
            cumA += X.T @ X
            cumC += X.T @ Y
            for kk in fgt_lam:
                Xw = X * wgt[kk]
                fA[kk] = fA[kk] * decay[kk] + Xw.T @ X
                fC[kk] = fC[kk] * decay[kk] + Xw.T @ Y
            cb = k + 1
            if cb in cum_set:
                snapA[cb] = cumA.cpu().numpy()
                snapC[cb] = cumC.cpu().numpy()
            if cb in ck_set:
                snapF[cb] = {kk: (fA[kk].cpu().numpy(), fC[kk].cpu().numpy())
                             for kk in fgt_lam}
            if (cb - 1) in ck_set:
                nxt[cb - 1] = (Xf, Yf)
            del X
            torch.cuda.synchronize()
            tb = time.time()
            prof["arm0_grams"] += tb - ta
            # [clock] the feature maps' block Grams, on the same block
            lqb = torch.as_tensor(lq_cache[wb].reshape(-1, v), device=dev)
            hb = torch.as_tensor(h_cache[wb].reshape(-1), device=dev)
            for fm in FM:
                Ff = torch.cat([feat_t(fm, Xf[:, :d], lqb, hb).float(),
                                torch.ones(len(Xf), 1, device=dev)], 1)
                F = Ff.double()
                cumAF[fm] += F.T @ F
                cumCF[fm] += F.T @ Y
                for kk in fgt_lam:
                    Fw = F * wgt[kk]
                    fAF[fm][kk] = fAF[fm][kk] * decay[kk] + Fw.T @ F
                    fCF[fm][kk] = fCF[fm][kk] * decay[kk] + Fw.T @ Y
                if cb in cum_set:
                    snapAF[fm][cb] = cumAF[fm].cpu().numpy()
                    snapCF[fm][cb] = cumCF[fm].cpu().numpy()
                if cb in ck_set:
                    snapFF[fm][cb] = {kk: (fAF[fm][kk].cpu().numpy(),
                                           fCF[fm][kk].cpu().numpy()) for kk in fgt_lam}
                if (cb - 1) in ck_set:
                    nxtF.setdefault(cb - 1, {})[fm] = Ff
                del F
            del Y
            torch.cuda.synchronize()
            prof["fmap_grams"] += time.time() - tb
        ta = time.time()
        betas = np.zeros((len(FITTERS), len(ck_blk), d + 1, n_tgt), np.float32)
        diag = {k: np.full((len(FITTERS), len(ck_blk), n_tgt), np.nan, np.float32)
                for k in DIAG_KEYS}
        lam_out = {}
        for fi, (fnm, kind, mem) in enumerate(FITTERS):
            if select_lam:
                cb0 = (P_win // B_)
                A0, C0 = _fit_AC(kind, fnm, cb0, snapA, snapC, snapF, win_blk, zeroA, zeroC)
                best = np.full(n_tgt, -1e18)
                lp = np.zeros(n_tgt)
                for lm in LAMS:
                    r2 = val_stats(ridge_solve(A0, C0, lm))["val_r2"]
                    up = r2 > best
                    best = np.where(up, r2, best)
                    lp[up] = lm
                lam_out[fnm] = lp
                lam_glob[fnm] = lp
            lv = lam_glob[fnm]
            for ci, cb in enumerate(ck_blk):
                A, C = _fit_AC(kind, fnm, cb, snapA, snapC, snapF, win_blk, zeroA, zeroC)
                B = solve_pertarget(A, C, lv)
                betas[fi, ci] = B.astype(np.float32)
                vs = val_stats(B)
                for k2 in ("val_r2", "delta_val", "delta_val_lo", "delta_val_mid",
                           "delta_val_hi"):
                    diag[k2][fi, ci] = vs[k2]
                diag["n_rows"][fi, ci] = A[-1, -1]
                if cb in nxt:
                    Xn, Yn = nxt[cb]
                    r = (Yn - Xn @ torch.as_tensor(B, device=dev, dtype=torch.float32))
                    rw = r.reshape(B_, R_, n_tgt).mean(1)
                    diag["delta_next"][fi, ci] = rw.mean(0).cpu().numpy()
                    diag["delta_next_sem"][fi, ci] = (
                        rw.std(0, unbiased=True) / np.sqrt(B_)).cpu().numpy()
        tb = time.time()
        prof["arm0_solves"] += tb - ta
        # [clock] each feature map: its own penalty (selected the same way), its own fits
        betasF, diagF, stdF = {}, {}, {}
        for fm in FM:
            betasF[fm] = np.zeros((len(FITTERS), len(ck_blk), DF[fm], n_tgt), np.float32)
            diagF[fm] = {k: np.full((len(FITTERS), len(ck_blk), n_tgt), np.nan, np.float32)
                         for k in DIAG_KEYS}
            Kk = DF[fm] - d - 1
            stdF[fm] = np.zeros((len(FITTERS), len(ck_blk), 2, max(Kk, 1)), np.float32)
            for fi, (fnm, kind, mem) in enumerate(FITTERS):
                if select_lam:
                    cb0 = (P_win // B_)
                    A0, C0 = _fit_AC(kind, fnm, cb0, snapAF[fm], snapCF[fm], snapFF[fm],
                                     win_blk, None, None)
                    best = np.full(n_tgt, -1e18)
                    lp = np.zeros(n_tgt)
                    for lm in LAMS:
                        B, _ = _solve_fm(fm, A0, C0, lm, d, per_target=False)
                        r2 = val_stats_F(fm, B)["val_r2"]
                        up = r2 > best
                        best = np.where(up, r2, best)
                        lp[up] = lm
                    lam_globF[fm][fnm] = lp
                lv = lam_globF[fm][fnm]
                for ci, cb in enumerate(ck_blk):
                    A, C = _fit_AC(kind, fnm, cb, snapAF[fm], snapCF[fm], snapFF[fm],
                                   win_blk, None, None)
                    B, Tst = _solve_fm(fm, A, C, lv, d, per_target=True)
                    betasF[fm][fi, ci] = B.astype(np.float32)
                    if Tst is not None:
                        stdF[fm][fi, ci, 0] = np.diag(Tst)[d:-1]
                        stdF[fm][fi, ci, 1] = Tst[-1, d:-1]
                    vs = val_stats_F(fm, B)
                    for k2 in ("val_r2", "delta_val", "delta_val_lo", "delta_val_mid",
                               "delta_val_hi"):
                        diagF[fm][k2][fi, ci] = vs[k2]
                    diagF[fm]["n_rows"][fi, ci] = A[-1, -1]
                    if cb in nxtF:
                        Fn = nxtF[cb][fm]
                        Yn = nxt[cb][1]
                        r = (Yn - Fn @ torch.as_tensor(B, device=dev, dtype=torch.float32))
                        rw = r.reshape(B_, R_, n_tgt).mean(1)
                        diagF[fm]["delta_next"][fi, ci] = rw.mean(0).cpu().numpy()
                        diagF[fm]["delta_next_sem"][fi, ci] = (
                            rw.std(0, unbiased=True) / np.sqrt(B_)).cpu().numpy()
        prof["fmap_solves"] += time.time() - tb
        del cumA, cumC, fA, fC, snapA, snapC, snapF, nxt
        del cumAF, cumCF, fAF, fCF, snapAF, snapCF, snapFF, nxtF
        torch.cuda.empty_cache()
        return betas, diag, lam_out, betasF, diagF, stdF

    t0 = time.time()
    arm_beta, arm_diag, arm_info = {}, {}, {}
    arm_betaF = {fm: {} for fm in FM}
    arm_diagF = {fm: {} for fm in FM}
    arm_stdF = {fm: {} for fm in FM}
    for ai, (anm, pa, pb, mode) in enumerate(arm_list):
        seq, wa, wb = build_seq(pa, pb, mode, ai)
        bt, dg, lo, btF, dgF, sdF = run_arm(seq, select_lam=(ai == 0))
        arm_beta[anm], arm_diag[anm] = bt, dg
        for fm in FM:
            arm_betaF[fm][anm], arm_diagF[fm][anm], arm_stdF[fm][anm] = \
                btF[fm], dgF[fm], sdF[fm]
        arm_info[anm] = {
            "phase_a": pa, "phase_b": pb, "mode": mode, "n_windows": int(len(seq)),
            "phase_windows": int(P_win),
            "E_outcome_a": float(o_row_tr[wa].mean()), "E_outcome_b": float(o_row_tr[wb].mean()),
            "E_damage_a": float(d_row_tr[wa].mean()), "E_damage_b": float(d_row_tr[wb].mean()),
            "cell_hist_a_eq_b": bool(
                (np.bincount(cell_key(et[tr_idx[wa]], jj[tr_idx[wa]]).astype(int), minlength=400)
                 == np.bincount(cell_key(et[tr_idx[wb]], jj[tr_idx[wb]]).astype(int),
                                minlength=400)).all())}
        print(f"  arm {anm:14s} E[o] {arm_info[anm]['E_outcome_a']:.4f} -> "
              f"{arm_info[anm]['E_outcome_b']:.4f}  cells equal "
              f"{arm_info[anm]['cell_hist_a_eq_b']}  ({time.time() - t0:.0f}s)  "
              f"prof {json.dumps({k: round(x, 1) for k, x in prof.items()})}", flush=True)
    tim["online"] = time.time() - t0
    print(f"online fits {time.time() - t0:.1f}s", flush=True)
    del cache, lq_cache, h_cache, XvF
    torch.cuda.empty_cache()

    # ---- anchors: per-episode columns for the STATIC critics, plus the raw states -----
    out_rows = {}
    featX = {}                                   # [clock] the extra features, per anchor
    for an in keep_anch:
        ok = anchor_ok[an]
        w = np.where(ok & is_te)[0]
        if len(w) < 50:
            continue
        ib, ia = ANC[an]
        t0w = anchors[an][w]
        rec = {"w": w, "t0": t0w, "etype": et[w], "j": jj[w], "k_star": ks[w], "e": ee[w],
               "split": split[w], "first_diff": fd[w], "t_v": tv[w], "c_depth": c_depth[w],
               "nll_e": nll_e[w, t0w - 1], "nll_o": nll_o[w, t0w - 1],
               "H_pre": H_e[w, t0w - 1],
               "dH_e": H_e[w, t0w] - H_e[w, t0w - 1],
               "dH_o": H_o[w, t0w] - H_o[w, t0w - 1]}
        rec["excess_e"] = rec["nll_e"] - rec["H_pre"]
        Xa, Xb_ = st_e[w, ia].astype(np.float64), st_e[w, ib].astype(np.float64)
        Xao, Xbo = st_o[w, ia].astype(np.float64), st_o[w, ib].astype(np.float64)
        rec["Xa"], rec["Xb"] = Xa.astype(np.float32), Xb_.astype(np.float32)
        rec["Xao"], rec["Xbo"] = Xao.astype(np.float32), Xbo.astype(np.float32)
        for nm in STATIC:
            for l in L_STORE:
                for a in A_STORE:
                    i0, i1 = names.index(f"l{l}_d{a}"), names.index(f"l{l}_d{a + 1}")
                    b0, b1 = beta_s[nm][:, i0], beta_s[nm][:, i1]
                    va = Xa @ b0[:-1] + b0[-1]
                    vb = Xb_ @ b1[:-1] + b1[-1]
                    vao = Xao @ b0[:-1] + b0[-1]
                    vbo = Xbo @ b1[:-1] + b1[-1]
                    rec[f"V_{nm}_l{l}_a{a}"] = va
                    rec[f"Vpre_{nm}_l{l}_a{a}"] = vb
                    rec[f"Vpreo_{nm}_l{l}_a{a}"] = vbo
                    rec[f"R_{nm}_l{l}_a{a}"] = va - vb
                    rec[f"Ro_{nm}_l{l}_a{a}"] = vao - vbo
        for l in LEVELS:
            li = l - 1
            for a in A_LIST:
                good = (t0w + a) <= T - 1
                tq = np.clip(t0w + a, 0, T - 1)
                rec[f"cons_l{l}_a{a}"] = ((y_edit[li][w, tq] != y_orig[li][w, tq])
                                          & good).astype(np.int8)
                rec[f"oe_l{l}_a{a}"] = o_e[li][w, tq]
                rec[f"oo_l{l}_a{a}"] = o_o[li][w, tq]
                rec[f"ok_l{l}_a{a}"] = good.astype(np.int8)
        out_rows[an] = rec
        # [clock] log q and H(q) at the same four saved states, same rows, same order
        pa_, pb_ = anc_pos[w, ia], anc_pos[w, ib]
        featX[f"{an}__w"] = w
        featX[f"{an}__lq_a"], featX[f"{an}__lq_b"] = lsm_ae[w, ia], lsm_ae[w, ib]
        featX[f"{an}__lq_ao"], featX[f"{an}__lq_bo"] = lsm_ao[w, ia], lsm_ao[w, ib]
        featX[f"{an}__h_a"], featX[f"{an}__h_b"] = H_e[w, pa_], H_e[w, pb_]
        featX[f"{an}__h_ao"], featX[f"{an}__h_bo"] = H_o[w, pa_], H_o[w, pb_]
        print(f"anchor {an}: {len(w)} test rows "
              f"(viol {int(((et[w] == 0) & (tv[w] > 0)).sum())})", flush=True)

    # ---- the same-prefix twins (phasic.py's construction; states, not values) ---------
    twin, twin_arr = {}, {}
    if pL is not None and max_twins > 0:
        t0 = time.time()
        elig = np.where((et == 0) & (tv >= T_LO + 1) & (tv <= T - 1))[0]
        if len(elig) > max_twins:
            elig = np.sort(np.random.default_rng(seed + 21).choice(elig, max_twins,
                                                                   replace=False))

        def states_at(W, tpos, bs=256):
            Xp = np.zeros((len(W), d), np.float32)
            X0 = np.zeros((len(W), d), np.float32)
            lsm_prev = np.zeros((len(W), v), np.float32)
            lsm_at = np.zeros((len(W), v), np.float32)          # [clock]
            H2 = np.zeros((len(W), 2), np.float32)               # [clock] H at t-1, t
            with torch.no_grad():
                for c0 in range(0, len(W), bs):
                    sl = slice(c0, min(c0 + bs, len(W)))
                    x = torch.as_tensor(W[sl, :T], device=dev)
                    lg, _, inter = model(x, return_intermediates=True)
                    tt = torch.as_tensor(tpos[sl], device=dev)
                    ar = torch.arange(x.shape[0], device=dev)
                    lsm_prev[sl] = torch.log_softmax(lg.float(), -1)[ar, tt - 1].cpu().numpy()
                    Xp[sl] = inter[PRIM][ar, torch.clamp(tt - 1, min=0)].float().cpu().numpy()
                    X0[sl] = inter[PRIM][ar, tt].float().cpu().numpy()
                    lsm_full = torch.log_softmax(lg.float(), -1)
                    Hfull = -(lsm_full.exp() * lsm_full).sum(-1)
                    lsm_at[sl] = lsm_full[ar, tt].cpu().numpy()
                    H2[sl, 0] = Hfull[ar, tt - 1].cpu().numpy()
                    H2[sl, 1] = Hfull[ar, tt].cpu().numpy()
            return Xp, X0, lsm_prev, lsm_at, H2

        tpos = tv[elig]
        Xpre_v, X0_v, lsm_v, lsm0_v, H2_v = states_at(We[elig], tpos)
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
            Xpre_c, X0_c, lsm_c, lsm0_c, H2_c = states_at(Wt, tpos[has])
            dev_prev = float(np.abs(Xpre_c - Xpre_v[has]).max())
            bq = beta_s["full"][:, names.index("l1_d1")]
            dvp = float(np.abs((Xpre_c.astype(np.float64) - Xpre_v[has].astype(np.float64))
                               @ bq[:-1]).max())
            dq = float(np.abs(lsm_c - lsm_v[has]).max())
            s_c = -lsm_c[np.arange(len(has)), x_c[has]]
            twin = {"n_eligible": int(len(elig)), "n_pairs": int(len(has)),
                    "caliper": float(caliper),
                    "check_state_pre_identical_max_abs": dev_prev,
                    "check_Vpre_identical_max_abs": dvp,
                    "check_qprev_identical_max_abs": dq,
                    "mean_abs_ds": float(np.abs(s_v[has] - s_c).mean()),
                    "coverage_by_kstar": {int(k): [int((ks[elig] == k).sum()),
                                                   int((ks[elig[has]] == k).sum())]
                                          for k in np.unique(ks[elig])}}
            twin_arr = {"w": elig[has], "t_v": tpos[has], "k_star": ks[elig[has]],
                        "j": jj[elig[has]], "e": ee[elig[has]], "split": split[elig[has]],
                        "x_v": x_v[has], "x_c": x_c[has], "s_v": s_v[has], "s_c": s_c,
                        "X_pre": Xpre_v[has], "X_v": X0_v[has], "X_c": X0_c}
            # [clock] the belief at the three saved twin states
            featX["tw__w"] = elig[has]
            featX["tw__lq_pre"] = lsm_v[has]
            featX["tw__lq_v"], featX["tw__lq_c"] = lsm0_v[has], lsm0_c
            featX["tw__h_pre"] = H2_v[has, 0]
            featX["tw__h_pre_c"] = H2_c[:, 0]
            featX["tw__h_v"], featX["tw__h_c"] = H2_v[has, 1], H2_c[:, 1]
            print(f"  state identity {dev_prev:.2e}  V_pre identity {dvp:.2e}  "
                  f"q_prev {dq:.2e}  "
                  f"|ds| {twin['mean_abs_ds']:.4f}  ({time.time() - t0:.1f}s)", flush=True)
        else:
            twin = {"n_eligible": int(len(elig)), "n_pairs": int(len(has)),
                    "note": "too few pairs"}

    # ---- the arm-0 record, built exactly as the banked cell saves it ------------------
    res = {"ckpt": ckpt, "step": step, "stim_tag": stim_tag, "n": n, "primary_block": PRIM,
           "actor_clean_acc": actor_tab, "diet_spec": spec, "diet_stats": diet_stats,
           "diet_meta": wmeta, "static": static_info, "twin": twin,
           "arms": {a: arm_info[a] for a in arm_info},
           "cell_counts_equal_across_diets": bool(cell_equal),
           "arm_order": [a[0] for a in arm_list],
           "fitters": [[f[0], f[1], f[2]] for f in FITTERS],
           "fitter_lam": {k: {names[i]: float(lam_glob[k][i]) for i in range(n_tgt)}
                          for k in lam_glob},
           "fgt_lam_per_row": fgt_lam, "win_blocks": win_blk,
           "ckpt_offsets": offs, "ckpt_blocks": ck_blk,
           "target_names": names, "anchors": keep_anch,
           "config": {"levels": LEVELS, "dmax": DMAX, "l_store": L_STORE,
                      "a_store": A_STORE, "dd_store": DD_STORE, "t_lo": T_LO,
                      "n_clean": n_clean, "seed": seed, "eval_seed": eval_seed,
                      "order_seed": order_seed, "split": [0.6, 0.15, 0.25],
                      "n_val_windows": int(len(va_idx)), "caliper": caliper,
                      "max_twins": max_twins, "strat_frac": strat_frac,
                      "block_win": B_, "phase_windows": int(P_win),
                      "rows": [int(rows[0]), int(rows[-1])], "R_": R_, "d": d}}
    save0 = {}
    for an, rec in out_rows.items():
        for k, vv in rec.items():
            a_ = np.asarray(vv)
            save0[f"{an}__{k}"] = a_.astype(np.float32) if a_.dtype == np.float64 else a_
    for k, vv in twin_arr.items():
        a_ = np.asarray(vv)
        save0[f"tw__{k}"] = a_.astype(np.float32) if a_.dtype == np.float64 else a_
    for nm in STATIC:
        save0[f"beta_static_{nm}"] = beta_s[nm].astype(np.float32)
    for anm in arm_beta:
        save0[f"beta_{anm}"] = arm_beta[anm]
        for k2, vv in arm_diag[anm].items():
            save0[f"diag_{anm}__{k2}"] = vv

    # ---- THE GATE: arm 0 against the banked cell, member by member --------------------
    stem = ckpt[:-3]
    sfx = f"_adapt_{stim_tag}" + (f"_{tag}" if tag else "")
    gate = {"bank": f"{stem}{sfx}.npz"}
    t0 = time.time()
    if os.path.exists(f"{stem}{sfx}.npz"):
        bz = np.load(f"{stem}{sfx}.npz")
        bj = json.load(open(f"{stem}{sfx}.json"))
        mine, theirs = set(save0), set(bz.files)
        gate["members_only_mine"] = sorted(mine - theirs)
        gate["members_only_bank"] = sorted(theirs - mine)
        groups = {}
        n_ident = 0
        for k in sorted(mine & theirs):
            a_, b_ = np.asarray(save0[k]), np.asarray(bz[k])
            grp = ("beta_static" if k.startswith("beta_static_") else
                   "beta_online" if k.startswith("beta_") else
                   "diag" if k.startswith("diag_") else
                   "twins" if k.startswith("tw__") else "anchors")
            g = groups.setdefault(grp, {"n": 0, "n_identical": 0, "shape_mismatch": 0,
                                        "max_abs": 0.0, "max_rel": 0.0, "nan_mismatch": 0,
                                        "int_mismatch": 0})
            g["n"] += 1
            if a_.shape != b_.shape:
                g["shape_mismatch"] += 1
                continue
            if np.array_equal(a_, b_, equal_nan=(a_.dtype.kind == "f")):
                g["n_identical"] += 1
                n_ident += 1
                continue
            if a_.dtype.kind in "fc":
                na, nb = np.isnan(a_), np.isnan(b_)
                if (na != nb).any():
                    g["nan_mismatch"] += 1
                ok_ = ~(na | nb)
                dd_ = np.abs(a_[ok_].astype(np.float64) - b_[ok_].astype(np.float64))
                mx = float(dd_.max()) if dd_.size else 0.0
                sc_ = float(np.abs(b_[ok_]).max()) if dd_.size else 1.0
                g["max_abs"] = max(g["max_abs"], mx)
                g["max_rel"] = max(g["max_rel"], mx / max(sc_, 1e-12))
            else:
                g["int_mismatch"] += 1
        gate["groups"] = groups
        gate["n_compared"] = len(mine & theirs)
        gate["n_bit_identical"] = n_ident
        jd = {"actor_clean_acc": max(abs(actor_tab[k] - bj["actor_clean_acc"][k])
                                     for k in actor_tab),
              "static_val_r2": max(abs(static_info[nm]["val_r2"][t] -
                                       bj["static"][nm]["val_r2"][t])
                                   for nm in STATIC for t in names),
              "fitter_lam_equal": all(res["fitter_lam"][f][t] == bj["fitter_lam"][f][t]
                                      for f in bj["fitter_lam"] for t in names),
              "ckpt_offsets_equal": list(offs) == list(bj["ckpt_offsets"]),
              "phase_windows_equal": int(P_win) == int(bj["config"]["phase_windows"]),
              "arms_E_outcome": max(abs(arm_info[a][k] - bj["arms"][a][k])
                                    for a in arm_info for k in ("E_outcome_a", "E_outcome_b")),
              "twin_n_pairs_equal": twin.get("n_pairs") == bj["twin"].get("n_pairs"),
              "diet_n_windows_equal": all(diet_stats[k]["n_windows"] ==
                                          bj["diet_stats"][k]["n_windows"]
                                          for k in diet_stats)}
        gate["json"] = jd
        fails = []
        if gate["members_only_mine"] or gate["members_only_bank"]:
            fails.append("member sets differ")
        for grp, g in groups.items():
            if g["shape_mismatch"] or g["nan_mismatch"] or g["int_mismatch"]:
                fails.append(f"{grp}: shape/nan/int mismatch")
            if grp in ("beta_static", "beta_online") and g["max_rel"] > 1e-4:
                fails.append(f"{grp}: max rel {g['max_rel']:.2e}")
            if grp in ("diag", "anchors", "twins") and g["max_abs"] > tol:
                fails.append(f"{grp}: max abs {g['max_abs']:.2e}")
        if jd["actor_clean_acc"] > 1e-9:
            fails.append("actor accuracy")
        if jd["static_val_r2"] > 1e-6:
            fails.append("static val R2")
        for k in ("fitter_lam_equal", "ckpt_offsets_equal", "phase_windows_equal",
                  "twin_n_pairs_equal", "diet_n_windows_equal"):
            if not jd[k]:
                fails.append(k)
        if jd["arms_E_outcome"] > 1e-9:
            fails.append("arms E_outcome")
        gate["fails"] = fails
        del bz
    else:
        gate["fails"] = ["NO BANKED CELL"]
    gate["seconds"] = time.time() - t0
    print(f"ARM-0 GATE vs {gate['bank']}: fails {gate['fails']}  "
          f"bit-identical {gate.get('n_bit_identical')}/{gate.get('n_compared')}  "
          + json.dumps({g: {k: x for k, x in v_.items() if k in ('n', 'n_identical', 'max_abs',
                                                                  'max_rel')}
                        for g, v_ in gate.get("groups", {}).items()}), flush=True)

    # ---- [clock] checks on the reduction's feature assembly --------------------------------
    lw = lnf.weight.detach().float().cpu().numpy()
    lb = lnf.bias.detach().float().cpu().numpy()
    checks = {}
    for an, rec in out_rows.items():
        Xb32 = rec["Xb"]
        with torch.no_grad():
            lt = lnf(torch.as_tensor(Xb32, device=dev)).double().cpu().numpy()
        checks[f"{an}/ln_numpy_vs_torch_max_abs"] = float(np.abs(
            ln_np(Xb32, lw, lb, lnf.eps) - lt).max())
        checks[f"{an}/H_from_lq_vs_H_max_abs"] = float(np.abs(
            ent_np(featX[f"{an}__lq_b"]) - featX[f"{an}__h_b"]).max())
        checks[f"{an}/H_pre_equals_h_b"] = bool(np.array_equal(
            rec["H_pre"].astype(np.float32), featX[f"{an}__h_b"].astype(np.float32)))
        # the in-container value of each arm's static `full` critic on this anchor, which
        # the reduction must reproduce from the saved features
        m_ = (rec["etype"] == 0) & (rec["t_v"] > 0)
        for fm in FM:
            b1 = beta_sF[fm]["full"][:, names.index("l1_d1")]
            b0 = beta_sF[fm]["full"][:, names.index("l1_d0")]
            if fm == "Q":
                Fb = np.concatenate([rec["Xb"], featX[f"{an}__lq_b"]], 1).astype(np.float64)
                Fa = np.concatenate([rec["Xa"], featX[f"{an}__lq_a"]], 1).astype(np.float64)
            elif fm == "H":
                Fb = np.concatenate([rec["Xb"], featX[f"{an}__h_b"][:, None]], 1).astype(np.float64)
                Fa = np.concatenate([rec["Xa"], featX[f"{an}__h_a"][:, None]], 1).astype(np.float64)
            else:
                with torch.no_grad():
                    Fb = lnf(torch.as_tensor(rec["Xb"], device=dev)).double().cpu().numpy()
                    Fa = lnf(torch.as_tensor(rec["Xa"], device=dev)).double().cpu().numpy()
            vb = Fb @ b1[:-1] + b1[-1]
            va = Fa @ b0[:-1] + b0[-1]
            checks[f"{an}/{fm}/full_l1_Vpre_mean_viol"] = float(vb[m_].mean())
            checks[f"{an}/{fm}/full_l1_R_mean_viol"] = float((va - vb)[m_].mean())
    if twin_arr:
        checks["tw/lq_pre_identical_max_abs"] = float(np.abs(
            featX["tw__lq_pre"] - lsm_c).max())
        checks["tw/h_pre_identical_max_abs"] = float(np.abs(
            featX["tw__h_pre"] - featX["tw__h_pre_c"]).max())
    print(f"checks {json.dumps(checks)}", flush=True)

    # ---- save (arm 0 is NOT re-saved: the gate asserts it equals the banked cell) --------
    out = {"ckpt": ckpt, "step": step, "stim_tag": stim_tag, "tag": tag, "n": n,
           "bank_stem": f"{stem}{sfx}", "fmaps": FM, "fmap_desc": {f: FMAP_DESC[f] for f in FM},
           "fdim": {f: fdim[f] for f in FM}, "gate": gate, "checks": checks,
           "actor_clean_acc": actor_tab,
           "static": {fm: static_infoF[fm] for fm in FM},
           "fitter_lam": {fm: {k: {names[i]: float(lam_globF[fm][k][i]) for i in range(n_tgt)}
                               for k in lam_globF[fm]} for fm in FM},
           "arm_order": [a[0] for a in arm_list], "ckpt_offsets": offs,
           "target_names": names, "anchors": keep_anch, "twin": twin,
           "config": res["config"], "timing": tim, "profile_online": prof,
           "lnf_eps": float(lnf.eps)}
    for fm in FM:
        zf = {}
        for nm in STATIC:
            zf[f"beta_static_{nm}"] = beta_sF[fm][nm].astype(np.float32)
        for anm in arm_betaF[fm]:
            zf[f"beta_{anm}"] = arm_betaF[fm][anm]
            zf[f"std_{anm}"] = arm_stdF[fm][anm]
            for k2, vv in arm_diagF[fm][anm].items():
                zf[f"diag_{anm}__{k2}"] = vv
        np.savez_compressed(f"{stem}{sfx}_clock{fm}.npz", **zf)
    featX["lnf_w"], featX["lnf_b"] = lw, lb
    featX["lnf_eps"] = np.array(float(lnf.eps))
    np.savez_compressed(f"{stem}{sfx}_clockX.npz",
                        **{k: (np.asarray(x).astype(np.float32)
                               if np.asarray(x).dtype == np.float64 else np.asarray(x))
                           for k, x in featX.items()})
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    out["peak_rss_gb"] = rss
    out["wall_s"] = time.time() - t00
    out["sizes_mb"] = {f: os.path.getsize(f"{stem}{sfx}_clock{f}.npz") / 1e6 for f in FM + ["X"]}
    with open(f"{stem}{sfx}_clock.json", "w") as f:
        json.dump(out, f, cls=NumpyEncoder, indent=1)
    volume.commit()
    print(f"saved -> {stem}{sfx}_clock{{{','.join(FM)},X}}.npz + .json  "
          f"({time.time() - t00:.0f}s, peak RSS {rss:.1f} GB)  sizes {out['sizes_mb']}  "
          f"timing {json.dumps({k: round(x, 1) for k, x in tim.items()})}", flush=True)
    return {"ckpt": ckpt, "stim_tag": stim_tag, "tag": tag, "gate_fails": gate["fails"],
            "n_bit_identical": gate.get("n_bit_identical"),
            "n_compared": gate.get("n_compared"), "wall_s": out["wall_s"],
            "peak_rss_gb": rss, "static_Vpre_gate": _gate(out_rows, STATIC),
            "l1_d0_R2": {"state": static_info["full"]["val_r2"]["l1_d0"],
                         **{fm: static_infoF[fm]["full"]["val_r2"]["l1_d0"] for fm in FM}}}


# ---------------------------------------------------------------------------
# coordinator: every cell in its own container
# ---------------------------------------------------------------------------

@app.function(volumes={DATA_DIR: volume}, timeout=3 * 3600, memory=1024)
def clock_sweep(cells: str = "", fmaps: str = "Q,L,H", block_win: int = 25,
                max_twins: int = 45000):
    """cells: `ckpt:stim_tag:anchors:order_seed:tag`, SEMICOLON separated.  The default is
    the four banked cells: a1 (anchors tv,fd) and swap65k (tv), order seeds 909 (no tag)
    and 1313 (`ord2`), at the clean trajectory's 64k checkpoint."""
    volume.reload()
    D = f"{DATA_DIR}/v16_s2_L6_m4_distinct/logit_reading"
    ck = f"{D}/traj_a1_s42/step064000.pt"
    if not cells:
        cells = ";".join([f"{ck}:a1:tv,fd:909:", f"{ck}:swap65k:tv:909:",
                          f"{ck}:a1:tv,fd:1313:ord2", f"{ck}:swap65k:tv:1313:ord2"])
    args = []
    for c in cells.split(";"):
        p = c.split(":")
        ckp, tg, an, osd, tag = p[0], p[1], p[2], int(p[3]), (p[4] if len(p) > 4 else "")
        if not os.path.exists(ckp):
            print(f"MISSING {ckp}", flush=True)
            continue
        # (ckpt, stim_tag, n_clean, n_clean_val, max_windows, seed, tag, actor_steps,
        #  eval_seed, n_val_windows, max_twins, caliper, chunk, strat_frac, block_win,
        #  anchors_keep, arms, order_seed, fmaps, tol) -- adaptation's sweep values
        args.append((ckp, tg, 6144, 2048, 0, 11, tag, 3000, 5150, 1500, max_twins, 0.6, 512,
                     1.0 / 3.0, block_win, an, "", osd, fmaps, 5e-5))
    print(f"{len(args)} cells: {[(a[1], a[15], a[17], a[6]) for a in args]}", flush=True)
    outs = list(clock_ckpt.starmap(args, return_exceptions=True))
    for o in outs:
        print(json.dumps(o, default=str)[:1500], flush=True)
    return [str(o)[:1500] for o in outs]
