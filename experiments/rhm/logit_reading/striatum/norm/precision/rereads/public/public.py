"""public: what is the augmented reader's response made of, and does "the response is private"
survive a public reader that is NONLINEAR?

`orbitofrontal/projection/` banked "the level is public and the response is private" with
readers that had EITHER the state (256 dims) OR the belief (log q, 16 dims), both linear.
`precision/express.py` then found that a ridge on `[state, log q]` (`fullQ`) and an MLP on the
state (`fullM`) contain what the pre-event entropy knows about damage and reverse the
response's entropy dependence at ell = 1, where the ridge on the raw state does not.  This
file refits what those readings need and did not save:

  1. `fullQ` / `fullP` (express's arms Q and P, refitted bit-for-bit on the same Gram and
     gated against the banked `_express` columns) with their COEFFICIENTS saved, and every
     per-row value split EXACTLY into a state part and an appended ("belief") part:
         V = x . b_s + b + mu . b_g      (state part; the intercept and the appended block's
           + (g - mu) . b_g              training mean go here)   (appended part, mean 0 on
                                         the training rows)
     at both anchors, on the edited and the unedited stream, and on the twins.
  2. Public readers: the ridge on log q per diet (`orbitofrontal/projection/`'s `logq` block,
     refitted here on three seeds and gated against the banked s42 cells in the reduction),
     a 128-unit MLP on log q alone per diet (the NONLINEAR public reader; `coeruleus.readout.
     train_head` with `fullM`'s own recipe and seeds), and the same MLP on `[state, log q]`
     (`full` only).  `fullM` itself is retrained with express's recipe as a guard on the
     recipe; the readings use the banked `_express` columns.
  3. The readout span (`projection/task.py`'s `||P_M beta|| / ||beta||`, verbatim) of the
     ridge and of `fullQ`'s / `fullP`'s STATE block, per column, plus the cosine between the
     two state directions.
  4. On `norm/`'s banked same-prefix twin pairs: every reader's V at offsets 0 and -1 for both
     members, the raw post-token forecasts `log q` at `t_v + 0..4` (`projection/`'s `POFFS`)
     and the state difference at `t_v`, so the priced map can be refitted on each reader.

Everything upstream is `express.py` verbatim (trunk, split, actors, the two forward passes,
the appended features `_aug_cols`, the standardiser, `_fit_arm`, lambda ladder and held-out
selection) and the diets are `norm/task.py`'s (`strat_diets` / `mix_diets` / `anchor_diets`,
same seeds), imported, not copied.

**Gates**, asserted before the cell returns (the json and npz are written first, with the
failures listed): actor accuracy against the banked json; arm-0 held-out R2 against the
banked json; the test rows (`w`, `t0`) and every outcome label against the banked `_express`
npz exactly; arm 0's and `fullQ`'s / `fullP`'s `V`, `Vpre`, `R`, `Ro` against the banked
`_express` columns at both anchors (so the state part plus the appended part reproduces the
banked arm); the diets' row counts and mean outcomes against the banked json; on the twins,
the recomputed surprisals against the banked `tw__s_v` / `tw__s_c` and arm 0's / `fullQ`'s V
at offsets 0 and -1 against the banked `_express` twin columns.  Reported, not asserted: the
retrained `fullM` against the banked `_express` `fullM` (GPU training need not be bit-exact).

Run (from a staged copy of `experiments/`, see FILES rows in the report):
  modal run -m rhm.logit_reading.striatum.norm.precision.rereads.public.public::public_ckpt \
      --ckpt /data/v16_s2_L6_m4_distinct/logit_reading/traj_a1_s42/step064000.pt --stim-tag a1
  modal run --detach -m rhm.logit_reading.striatum.norm.precision.rereads.public.public::public_sweep
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
from rhm.logit_reading.striatum.norm.task import (LEVELS, DMAX, DD_LIST, A_LIST, L_STORE,
                                                  T_LO, LAMS, tgt_names)
from rhm.logit_reading.striatum.norm.diets import strat_diets, mix_diets, anchor_diets
from rhm.logit_reading.striatum.norm.precision.express import (AUG, K, ARMS, _aug_cols,
                                                               _standardiser, _fit_arm)


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
app = modal.App("rhm-norm-precision-public", image=image)

AUG_ARMS = ["Q", "P"]                 # express's arms whose coefficients this file needs
POFFS = [0, 1, 2, 3, 4]               # projection/task.py's forecast offsets, verbatim
MLP_DD = (0, 1)
SAVE_CALIPER = 0.35                   # projection/task.py's: F and dX saved for |ds| <= this
QCOLS = ARMS["Q"]                     # the 16 log q columns inside AUG


@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=2 * 3600, memory=16384,
              max_containers=6)
def public_ckpt(ckpt: str, stim_tag: str = "a1", n_clean: int = 6144,
                n_clean_val: int = 2048, seed: int = 11, actor_steps: int = 3000,
                eval_seed: int = 5150, n_clean_crit: int = 8192,
                prim_block: str = "post_block7", n_val_windows: int = 1500,
                chunk: int = 512, mlp_steps: int = 2000, do_twins: bool = True,
                tol: float = 5e-5, u_seed: int = 4242, strat_frac: float = 1.0 / 3.0,
                mlp_diets: str = "all", out_sfx: str = "_public"):
    """One (checkpoint, venue) cell.  `mlp_diets`: "all" trains the log-q MLP on every diet
    the banked cell has (a1: 11), "full" on `full` only."""
    import torch
    volume.reload()
    t00 = time.time()
    tim = {}
    dev = "cuda"
    model, rules, rule_w, cfg = load_trajectory_ckpt(ckpt, dev)
    L, s_, v = cfg["L"], cfg["s"], cfg["v"]
    T = model.block_size
    d = model.transformer.wte.weight.shape[1]
    key = tb_key(cfg["v"], cfg["s"], cfg["L"], cfg["m"])
    ddir = f"{DATA_DIR}/{key}/logit_reading"
    S = np.load(f"{ddir}/stimuli_{stim_tag}.npz")
    P_ = np.load(f"{ddir}/parse_{stim_tag}.npz")
    stem = ckpt[:-3]
    bank_npz = np.load(f"{stem}_norm_{stim_tag}.npz")
    bank = json.load(open(f"{stem}_norm_{stim_tag}.json"))
    xpr = np.load(f"{stem}_norm_{stim_tag}_express.npz")
    xpr_j = json.load(open(f"{stem}_norm_{stim_tag}_express.json"))
    diet_names_bank = list(bank["diet_stats"].keys())
    do_window_diets = len(diet_names_bank) > 1
    print(f"loaded {ckpt} step {cfg.get('step')} d={d} T={T}  banked diets "
          f"{diet_names_bank}", flush=True)

    # ---- norm/task.py verbatim: windows, split -------------------------------------------
    n_all = S["windows_edit"].shape[0]
    assert int(bank["n"]) == n_all, "the banked cell ran on a window subsample"
    idx = np.arange(n_all)
    We, Wo = S["windows_edit"][idx], S["windows_orig"][idx]
    meta = {k: S[k][idx] for k in ("phase", "etype", "j", "e", "first_diff", "t_v", "k_star")}
    y_edit = P_["y_edit"][:, idx][:, :, :T]
    y_orig = P_["y_orig"][:, idx][:, :, :T]
    n = len(idx)
    et, jj, ee, fd, tv, ks = (meta["etype"], meta["j"], meta["e"], meta["first_diff"],
                              meta["t_v"], meta["k_star"])
    pL = S["pL_edit"][idx] if "pL_edit" in S.files else None
    rng = np.random.default_rng(seed + 1)
    perm = rng.permutation(n)
    n_tr, n_va = int(0.60 * n), int(0.15 * n)
    split = np.zeros(n, np.int8)
    split[perm[n_tr:n_tr + n_va]] = 1
    split[perm[n_tr + n_va:]] = 2
    is_tr, is_va = split == 0, split == 1
    tr_idx = np.where(is_tr)[0]
    va_idx = np.where(is_va)[0][:n_val_windows]
    prim = prim_block

    # ---- norm/task.py verbatim: the frozen actors ------------------------------------------
    from rhm.logit_reading.altitude.units import windows_with_parse
    from rhm.logit_reading.striatum.parse import clean_answers
    t0 = time.time()
    n_cl = n_clean + n_clean_val
    wins_c, phase_c, _, fs_c = windows_with_parse(rules, rule_w, n_cl, T + 1, eval_seed)
    y_c = clean_answers(fs_c, phase_c, T + 1, L, s_, 3)[:, :, :T]
    pos = np.arange(T_LO, T)
    Sc = np.zeros((n_cl, T, d), np.float32)
    with torch.no_grad():
        for c0 in range(0, n_cl, 256):
            x = torch.as_tensor(wins_c[c0:c0 + 256, :T], device=dev)
            _, _, inter = model(x, return_intermediates=True)
            Sc[c0:c0 + 256] = inter[prim].float().cpu().numpy()
    actors, actor_tab = {}, {}
    Xtr = torch.as_tensor(Sc[:n_clean][:, pos].reshape(-1, d), device=dev)
    Xva = torch.as_tensor(Sc[n_clean:][:, pos].reshape(-1, d), device=dev)
    for l in LEVELS:
        ytr = torch.as_tensor(y_c[l - 1][:n_clean][:, pos].reshape(-1).astype(np.int64),
                              device=dev)
        yva = torch.as_tensor(y_c[l - 1][n_clean:][:, pos].reshape(-1).astype(np.int64),
                              device=dev)
        h = train_actor(Xtr, ytr, Xva, yva, steps=actor_steps, seed=l, device=dev)
        actors[l] = h
        actor_tab[f"{prim}/l{l}"] = h["acc"]
    u = np.random.default_rng(u_seed).standard_normal(d) / np.sqrt(d)
    sdz = float((Sc[:n_clean][:, pos].reshape(-1, d) @ u).std())
    del Xtr, Xva, Sc
    torch.cuda.empty_cache()
    tim["actors"] = time.time() - t0
    print(f"actors {tim['actors']:.1f}s", flush=True)

    # ---- anchors (norm/task.py verbatim) ---------------------------------------------------
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

    # ---- the two forward passes, plus the appended features (express.py verbatim) ---------
    def run(W, Y, want_cache=False, bs=256):
        o = np.zeros((len(LEVELS), len(W), T), np.uint8)
        nll = np.zeros((len(W), T), np.float32)
        Hq = np.zeros((len(W), T), np.float32)
        anc = np.zeros((len(W), 4, d), np.float32)
        lsm_anc = np.zeros((len(W), 4, v), np.float32)
        cache = np.zeros((len(cache_idx), R_, d), np.float16) if want_cache else None
        aug = np.zeros((len(cache_idx), R_, K), np.float32) if want_cache else None
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
                    pred = actor_apply(actors[l], Xp)
                    tru = torch.as_tensor(Y[li][sl].astype(np.int64), device=dev)
                    o[li, sl] = (pred == tru).to(torch.uint8).cpu().numpy()
                ap = torch.as_tensor(anc_pos[sl], device=dev)
                ar = torch.arange(x.shape[0], device=dev)[:, None]
                anc[sl] = inter[prim][ar, ap].float().cpu().numpy()
                lsm_anc[sl] = lsm[ar, ap].cpu().numpy()
                if want_cache:
                    kk = np.where(cache_map[c0:sl.stop] >= 0)[0]
                    if len(kk):
                        kt = torch.as_tensor(kk, device=dev)
                        ci = cache_map[c0 + kk]
                        cache[ci] = inter[prim][kt][:, rows].half().cpu().numpy()
                        nb = len(kk)
                        aug[ci] = _aug_cols(Hq, nll, lsm[kt][:, rows].cpu().numpy(),
                                            inter[prim][kt][:, rows].float().cpu().numpy(),
                                            u, sdz, np.broadcast_to(rows[None, :], (nb, R_)),
                                            np.broadcast_to((c0 + kk)[:, None], (nb, R_)))
        return o, nll, Hq, anc, lsm_anc, cache, aug

    t0 = time.time()
    o_o, nll_o, H_o, st_o, lsm_ao, _, _ = run(Wo, y_orig)
    o_e, nll_e, H_e, st_e, lsm_ae, cache, aug_c = run(We, y_edit, want_cache=True)
    tim["passes"] = time.time() - t0
    print(f"passes {tim['passes']:.1f}s  cache {cache.nbytes / 1e9:.2f} GB", flush=True)

    # ---- the diets (norm/task.py verbatim, same seeds -> identical masks) ------------------
    dmg_full = (o_o.astype(np.int16) - o_e.astype(np.int16)) > 0
    dmg_w = dmg_full[:4][:, :, rows]
    dmg_win = dmg_w.mean((0, 2))
    out_win = o_e[:4][:, :, rows].astype(np.float32).mean((0, 2))
    masks = {}
    if do_window_diets:
        ms, _, _ = strat_diets(out_win, et, jj, is_tr, seed=seed + 7, frac=strat_frac,
                               prefix="out")
        masks.update(ms)
        n_strat = int(max(m.sum() for m in ms.values()))
        ms, _, _ = strat_diets(dmg_win, et, jj, is_tr, seed=seed + 8, frac=strat_frac,
                               prefix="dmg")
        masks.update(ms)
        ms, _, _ = mix_diets(et, is_tr, n_total=n_strat, seed=seed + 9)
        masks.update(ms)
        ms, _ = anchor_diets(et, is_tr, n_total=n_strat, seed=seed + 10)
        masks.update(ms)
    else:
        ms, _ = anchor_diets(et, is_tr, n_total=0, seed=seed + 10)
        masks.update(ms)
    for k_ in list(masks):
        masks[k_] = np.repeat(masks[k_][tr_idx][:, None], R_, 1)       # (n_tr_win, R_)
    diet_names = list(masks)
    o_row_tr = o_e[:4][:, tr_idx][:, :, rows].astype(np.float32).mean(0)
    diet_gate = {}
    for nm in diet_names:
        m = masks[nm]
        bs_ = bank["diet_stats"].get(nm, {})
        diet_gate[nm] = {"n_rows": int(m.sum()), "n_rows_bank": bs_.get("n_rows"),
                         "mean_outcome": float(o_row_tr[m].mean()),
                         "mean_outcome_bank": bs_.get("mean_outcome")}
    print(f"{len(diet_names)} diets: {diet_names}", flush=True)
    mlp_diet_list = diet_names if mlp_diets == "all" else ["full"]

    # ---- Grams: arm 0, the augmented Gram (express verbatim) and log q per diet ------------
    names = tgt_names()
    n_tgt = len(names)
    n_real = n_tgt // 2
    D = d + K + 1
    nq = len(QCOLS)
    A0 = torch.zeros(d + 1, d + 1, dtype=torch.float64, device=dev)
    C0 = torch.zeros(d + 1, n_tgt, dtype=torch.float64, device=dev)
    Aa = torch.zeros(D, D, dtype=torch.float64, device=dev)
    Ca = torch.zeros(D, n_tgt, dtype=torch.float64, device=dev)
    Aq = {nm: torch.zeros(nq + 1, nq + 1, dtype=torch.float64, device=dev) for nm in diet_names}
    Cq = {nm: torch.zeros(nq + 1, n_tgt, dtype=torch.float64, device=dev) for nm in diet_names}
    shuf = np.random.default_rng(seed + 3).permutation(len(tr_idx))
    t0 = time.time()
    n_rows_tr = 0
    for c0 in range(0, len(tr_idx), chunk):
        sl = slice(c0, min(c0 + chunk, len(tr_idx)))
        wsl = tr_idx[sl]
        X = torch.as_tensor(cache[sl].reshape(-1, d), device=dev).double()
        ones = torch.ones(len(X), 1, dtype=torch.float64, device=dev)
        G = torch.as_tensor(aug_c[sl].reshape(-1, K), device=dev).double()
        ys = []
        for src in (wsl, tr_idx[shuf[sl]]):
            for li in range(len(LEVELS)):
                for dd in DD_LIST:
                    ys.append(torch.as_tensor(o_e[li][np.ix_(src, rows + dd)],
                                              dtype=torch.float64, device=dev))
        Y = torch.stack(ys, -1).reshape(-1, n_tgt)
        X1 = torch.cat([X, ones], 1)
        A0 += X1.T @ X1
        C0 += X1.T @ Y
        F = torch.cat([X, G, ones], 1)
        Aa += F.T @ F
        Ca += F.T @ Y
        Fq = torch.cat([G[:, QCOLS[0]:QCOLS[-1] + 1], ones], 1)
        for nm in diet_names:
            sel = torch.as_tensor(np.where(masks[nm][sl].reshape(-1))[0], device=dev)
            if not len(sel):
                continue
            Fs, Ys = Fq[sel], Y[sel]
            Aq[nm] += Fs.T @ Fs
            Cq[nm] += Fs.T @ Ys
        n_rows_tr += len(X)
        del X, X1, F, G, Y, Fq
    tim["grams"] = time.time() - t0
    print(f"grams {tim['grams']:.1f}s", flush=True)

    Xv = cache[len(tr_idx):].reshape(-1, d).astype(np.float64)
    Gv = aug_c[len(tr_idx):].reshape(-1, K).astype(np.float64)
    Yv = np.stack([o_e[li][np.ix_(va_idx, rows + dd)].astype(np.float64).reshape(-1)
                   for li in range(len(LEVELS)) for dd in DD_LIST], -1)
    Yv = np.concatenate([Yv, Yv], 1)

    # arm 0: norm/task.py's own loop, on the GPU, bit for bit (express verbatim)
    t0 = time.time()
    Xv1 = torch.as_tensor(np.concatenate([Xv, np.ones((len(Xv), 1))], 1), device=dev)
    Yvt = torch.as_tensor(Yv, device=dev)
    yvar = Yvt.var(0)
    An, Cn = A0.cpu().numpy(), C0.cpu().numpy()
    best = torch.full((n_tgt,), -1e18, dtype=torch.float64, device=dev)
    Bb = torch.zeros(d + 1, n_tgt, dtype=torch.float64, device=dev)
    for lm in LAMS:
        B = torch.as_tensor(ridge_solve(An, Cn, lm), device=dev)
        r2 = 1.0 - ((Xv1 @ B - Yvt) ** 2).mean(0) / torch.clamp(yvar, min=1e-12)
        up = r2 > best
        best = torch.where(up, r2, best)
        Bb[:, up] = B[:, up]
    beta = {"full": Bb.cpu().numpy()}
    val_r2 = {"full": {names[i]: float(best[i]) for i in range(n_tgt)}}
    del Xv1, Yvt

    # the augmented arms Q and P (express verbatim), plus each arm's appended-block mean
    Aan, Can = Aa.cpu().numpy(), Ca.cpu().numpy()
    Tst = _standardiser(Aan, d, n_rows_tr)
    As_, Cs_ = Tst.T @ Aan @ Tst, Tst.T @ Can
    Fv = np.concatenate([Xv, Gv, np.ones((len(Xv), 1))], 1) @ Tst
    mu_aug = Aan[d:d + K, -1] / n_rows_tr                 # training mean of each AUG column
    arm_cols = {}
    for arm in AUG_ARMS:
        cols = ARMS[arm]
        cc = list(range(d)) + [d + c for c in cols]
        B, r2 = _fit_arm(As_, Cs_, Fv, Yv, cc, LAMS)
        full_b = np.zeros((D, n_tgt))
        full_b[cc + [D - 1]] = B
        raw = Tst @ full_b
        beta[f"full{arm}"] = raw[cc + [D - 1]]
        arm_cols[arm] = cols
        val_r2[f"full{arm}"] = {names[i]: float(r2[i]) for i in range(n_tgt)}

    # the linear public critic: ridge on log q per diet, lambda picked on the same held-out
    # rows (projection/task.py's `logq` block on norm's split and diets)
    Fqv = np.concatenate([Gv[:, QCOLS[0]:QCOLS[-1] + 1], np.ones((len(Gv), 1))], 1)
    for nm in diet_names:
        B, r2 = _fit_arm(Aq[nm].cpu().numpy(), Cq[nm].cpu().numpy(), Fqv, Yv,
                         list(range(nq)), LAMS)
        beta[f"lq_{nm}"] = B
        val_r2[f"lq_{nm}"] = {names[i]: float(r2[i]) for i in range(n_tgt)}
    # guard: log q's `full` Gram is the sub-block of the augmented one
    qi = [d + c for c in QCOLS] + [D - 1]
    gram_guard = float(np.abs(Aq["full"].cpu().numpy() - Aan[np.ix_(qi, qi)]).max()
                       / max(np.abs(Aan[np.ix_(qi, qi)]).max(), 1e-12))
    del Aa, Ca, A0, C0, Aq, Cq
    torch.cuda.empty_cache()
    tim["solves"] = time.time() - t0
    print(f"solves {tim['solves']:.1f}s  l1_d0 R2: full {val_r2['full']['l1_d0']:.4f}  "
          f"fullQ {val_r2['fullQ']['l1_d0']:.4f}  fullP {val_r2['fullP']['l1_d0']:.4f}  "
          f"lq_full {val_r2['lq_full']['l1_d0']:.4f}  gram guard {gram_guard:.1e}", flush=True)

    # ---- the MLP readers: fullM (the recipe guard), log q per diet, [state, log q] ---------
    from rhm.logit_reading.coeruleus.readout import train_head
    from rhm.logit_reading.striatum.task import _r2
    t0 = time.time()
    mlp = {}
    n_trw = len(tr_idx)
    Xs_tr = cache[:n_trw].reshape(-1, d).astype(np.float32)
    Xs_va = cache[n_trw:].reshape(-1, d).astype(np.float32)
    Xq_tr = aug_c[:n_trw][:, :, QCOLS[0]:QCOLS[-1] + 1].reshape(-1, nq).astype(np.float32)
    Xq_va = aug_c[n_trw:][:, :, QCOLS[0]:QCOLS[-1] + 1].reshape(-1, nq).astype(np.float32)
    Ytr, Yva = {}, {}
    for l in L_STORE:
        li = LEVELS.index(l)
        for dd in MLP_DD:
            Ytr[(l, dd)] = o_e[li][tr_idx][:, rows + dd].reshape(-1).astype(np.float32)
            Yva[(l, dd)] = o_e[li][va_idx][:, rows + dd].reshape(-1).astype(np.float64)
    for l in L_STORE:
        for dd in MLP_DD:
            h = train_head(Xs_tr, Ytr[(l, dd)], steps=mlp_steps, device=dev, seed=l * 17 + dd)
            mlp[("fullMr", l, dd)] = h
            val_r2.setdefault("fullMr", {})[f"l{l}_d{dd}"] = _r2(h(Xs_va), Yva[(l, dd)])
    tim["mlp_state"] = time.time() - t0
    t1 = time.time()
    for nm in mlp_diet_list:
        mk = masks[nm].reshape(-1)
        Xd = Xq_tr[mk]
        for l in L_STORE:
            for dd in MLP_DD:
                h = train_head(Xd, Ytr[(l, dd)][mk], steps=mlp_steps, device=dev,
                               seed=l * 17 + dd)
                mlp[(f"mq_{nm}", l, dd)] = h
                val_r2.setdefault(f"mq_{nm}", {})[f"l{l}_d{dd}"] = _r2(h(Xq_va),
                                                                        Yva[(l, dd)])
        del Xd
    tim["mlp_logq"] = time.time() - t1
    t1 = time.time()
    Xsq_tr = np.concatenate([Xs_tr, Xq_tr], 1)
    Xsq_va = np.concatenate([Xs_va, Xq_va], 1)
    del Xs_tr
    for l in L_STORE:
        for dd in MLP_DD:
            h = train_head(Xsq_tr, Ytr[(l, dd)], steps=mlp_steps, device=dev, seed=l * 17 + dd)
            mlp[("fullQM", l, dd)] = h
            val_r2.setdefault("fullQM", {})[f"l{l}_d{dd}"] = _r2(h(Xsq_va), Yva[(l, dd)])
    tim["mlp_stateq"] = time.time() - t1
    del Xsq_tr, Xsq_va, Xs_va, Xq_tr, Xq_va
    torch.cuda.empty_cache()
    print(f"mlps {time.time() - t0:.1f}s ({len(mlp)} heads)  l1_d0: " + "  ".join(
        f"{k} {val_r2[k]['l1_d0']:.4f}" for k in ("fullMr", "mq_full", "fullQM")), flush=True)

    # ---- the clean-only critic: arm 0 and the Q / P arms (express verbatim) ----------------
    t0 = time.time()
    real_names = names[:n_real]
    wins_cc, phase_cc, _, fs_cc = windows_with_parse(rules, rule_w, n_clean_crit, T + 1,
                                                     eval_seed + 1000)
    y_cc = clean_answers(fs_cc, phase_cc, T + 1, L, s_, 3)[:, :, :T]
    n_cc_va = min(2000, n_clean_crit // 5)
    n_cc_tr = n_clean_crit - n_cc_va
    Acc = torch.zeros(d + 1, d + 1, dtype=torch.float64, device=dev)
    Ccc = torch.zeros(d + 1, n_real, dtype=torch.float64, device=dev)
    Aca = torch.zeros(D, D, dtype=torch.float64, device=dev)
    Cca = torch.zeros(D, n_real, dtype=torch.float64, device=dev)
    o_cc = np.zeros((len(LEVELS), n_clean_crit, T), np.uint8)
    Xcc_va = np.zeros((n_cc_va, R_, d), np.float32)
    Gcc_va = np.zeros((n_cc_va, R_, K), np.float32)
    n_cc_rows = 0
    with torch.no_grad():
        for c0 in range(0, n_clean_crit, 256):
            sl = slice(c0, min(c0 + 256, n_clean_crit))
            x = torch.as_tensor(wins_cc[sl, :T], device=dev)
            lg, _, inter = model(x, return_intermediates=True)
            lsm = torch.log_softmax(lg.float(), -1)
            nxt = torch.as_tensor(wins_cc[sl, 1:T + 1], device=dev)
            nll_b = (-lsm.gather(-1, nxt[..., None])[..., 0]).cpu().numpy()
            H_b = (-(lsm.exp() * lsm).sum(-1)).cpu().numpy()
            Xp = inter[prim].float()
            for li, l in enumerate(LEVELS):
                pred = actor_apply(actors[l], Xp)
                tru = torch.as_tensor(y_cc[li][sl].astype(np.int64), device=dev)
                o_cc[li, sl] = (pred == tru).to(torch.uint8).cpu().numpy()
            ii = np.arange(sl.start, sl.stop)
            nb = len(ii)
            f32 = inter[prim][:, rows].float().cpu().numpy()
            lsm_r = lsm[:, rows].cpu().numpy()
            G_b = _aug_cols(H_b, nll_b, lsm_r, f32, u, sdz,
                            np.broadcast_to(rows[None, :], (nb, R_)),
                            np.broadcast_to(np.arange(nb)[:, None], (nb, R_)))
            kk = np.where(ii < n_cc_tr)[0]
            if len(kk):
                kt = torch.as_tensor(kk, device=dev)
                Xb = inter[prim][kt][:, rows].double().reshape(-1, d)
                ones = torch.ones(len(Xb), 1, dtype=torch.float64, device=dev)
                Xb1 = torch.cat([Xb, ones], 1)
                ys = [torch.as_tensor(o_cc[li][np.ix_(ii[kk], rows + dd)],
                                      dtype=torch.float64, device=dev)
                      for li in range(len(LEVELS)) for dd in DD_LIST]
                Yb = torch.stack(ys, -1).reshape(-1, n_real)
                Acc += Xb1.T @ Xb1
                Ccc += Xb1.T @ Yb
                Gt = torch.as_tensor(G_b[kk].reshape(-1, K), device=dev).double()
                Fb = torch.cat([Xb, Gt, ones], 1)
                Aca += Fb.T @ Fb
                Cca += Fb.T @ Yb
                n_cc_rows += len(Xb)
            vk = np.where(ii >= n_cc_tr)[0]
            if len(vk):
                Xcc_va[ii[vk] - n_cc_tr] = \
                    inter[prim][torch.as_tensor(vk, device=dev)][:, rows].float().cpu().numpy()
                Gcc_va[ii[vk] - n_cc_tr] = G_b[vk]
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
    val_r2["clean"] = {real_names[i]: float(bestc[i]) for i in range(n_real)}
    Acan, Ccan = Aca.cpu().numpy(), Cca.cpu().numpy()
    Tcc = _standardiser(Acan, d, n_cc_rows)
    Asc, Csc = Tcc.T @ Acan @ Tcc, Tcc.T @ Ccan
    Fvc = np.concatenate([Xcc_va.reshape(-1, d).astype(np.float64),
                          Gcc_va.reshape(-1, K).astype(np.float64),
                          np.ones((n_cc_va * R_, 1))], 1) @ Tcc
    Yvc_n = Yvc.cpu().numpy()
    for arm in AUG_ARMS:
        cc = list(range(d)) + [d + c for c in ARMS[arm]]
        B, r2 = _fit_arm(Asc, Csc, Fvc, Yvc_n, cc, LAMS)
        full_b = np.zeros((D, n_real))
        full_b[cc + [D - 1]] = B
        raw = Tcc @ full_b
        beta[f"clean{arm}"] = raw[cc + [D - 1]]
        val_r2[f"clean{arm}"] = {real_names[i]: float(r2[i]) for i in range(n_real)}
    del Acc, Ccc, Aca, Cca, Xvc, Yvc, cache, aug_c
    torch.cuda.empty_cache()
    tim["clean"] = time.time() - t0
    print(f"clean critics {tim['clean']:.1f}s", flush=True)

    # ---- the readout span (projection/task.py verbatim) -------------------------------------
    Vb = readout_basis_of(model)
    rk = Vb.shape[0]
    span = {"_rank": rk, "_random_direction": float(np.sqrt(rk / d))}
    for crit in ("full", "fullQ", "fullP", "clean", "cleanQ", "cleanP"):
        ref = "clean" if crit.startswith("clean") else "full"
        for nm_ in real_names:
            j_ = names.index(nm_) if not crit.startswith("clean") else real_names.index(nm_)
            bvec = beta[crit][:d, j_]
            b0 = beta[ref][:d, j_]
            span[f"{crit}/{nm_}"] = float(np.linalg.norm(Vb @ bvec)
                                          / max(np.linalg.norm(bvec), 1e-12))
            if crit != ref:
                span[f"{crit}/{nm_}/cos_ref"] = float(
                    bvec @ b0 / max(np.linalg.norm(bvec) * np.linalg.norm(b0), 1e-18))
                span[f"{crit}/{nm_}/norm_ratio"] = float(
                    np.linalg.norm(bvec) / max(np.linalg.norm(b0), 1e-18))
    print(f"readout span rank {rk}; full l1_d0 {span['full/l1_d0']:.4f}  fullQ state block "
          f"{span['fullQ/l1_d0']:.4f}  chance {span['_random_direction']:.4f}", flush=True)

    # ---- per-row readers ----------------------------------------------------------------
    def col(crit, name):
        return (names.index(name) if not crit.startswith("clean")
                else real_names.index(name))

    def lin_full(Xs, name):
        b = beta["full"][:, col("full", name)]
        return Xs @ b[:-1] + b[-1]

    def parts(arm, name, Xs, Gs):
        """(state part, appended part) of arm Q or P; they sum to the arm's V exactly."""
        b = beta[f"full{arm}"][:, col("full", name)]
        cols = arm_cols[arm]
        bs, bg, b0 = b[:d], b[d:d + len(cols)], b[-1]
        mu = mu_aug[cols]
        g = Gs[:, cols].astype(np.float64)
        return Xs @ bs + b0 + mu @ bg, (g - mu) @ bg

    def lq(nm, name, Q):
        b = beta[f"lq_{nm}"][:, col("full", name)]
        return Q @ b[:-1] + b[-1]

    def mlp_pred(key_, Xs):
        return mlp[key_](np.asarray(Xs, np.float32)).astype(np.float64)

    gates = {"actor": {}, "val_r2": {}, "rows": {}, "cols": {}, "diets": {}, "twins": {}}
    report = {"fullMr_vs_banked_fullM": {}}
    for k_, a_ in actor_tab.items():
        gates["actor"][k_] = abs(a_ - bank["actor_clean_acc"][k_])
    for l in L_STORE:
        for dd in (0, 1):
            nm_ = f"l{l}_d{dd}"
            gates["val_r2"][f"full/{nm_}"] = abs(val_r2["full"][nm_]
                                                 - bank["critic_val_r2"]["full"][nm_])
            gates["val_r2"][f"clean/{nm_}"] = abs(val_r2["clean"][nm_]
                                                  - bank["clean_critic_val_r2"][nm_])
            for arm in AUG_ARMS:
                gates["val_r2"][f"full{arm}/{nm_}"] = abs(
                    val_r2[f"full{arm}"][nm_] - xpr_j["val_r2"][f"full{arm}"][nm_])
    for nm, g_ in diet_gate.items():
        gates["diets"][f"{nm}/n_rows"] = (g_["n_rows"] == g_["n_rows_bank"])
        gates["diets"][f"{nm}/mean_outcome"] = abs(g_["mean_outcome"]
                                                   - float(g_["mean_outcome_bank"]))

    save = {}
    a = 0
    for an, t0a in anchors.items():
        ok = anchor_ok[an]
        w = np.where(ok)[0]
        if len(w) < 50:
            continue
        ib, ia = ANC[an]
        t0w = t0a[w]
        rec = {"w": w, "t0": t0w, "etype": et[w], "j": jj[w], "k_star": ks[w], "e": ee[w],
               "split": split[w], "first_diff": fd[w], "t_v": tv[w],
               "c_depth": cons_depth(et, jj)[w]}
        rec["nll_e"] = nll_e[w, t0w - 1]
        rec["nll_o"] = nll_o[w, t0w - 1]
        rec["H_pre"] = H_e[w, t0w - 1]
        rec["excess_e"] = rec["nll_e"] - rec["H_pre"]
        Xa, Xb_ = st_e[w, ia].astype(np.float64), st_e[w, ib].astype(np.float64)
        Xao, Xbo = st_o[w, ia].astype(np.float64), st_o[w, ib].astype(np.float64)
        Qa, Qb = lsm_ae[w, ia].astype(np.float64), lsm_ae[w, ib].astype(np.float64)
        Qao, Qbo = lsm_ao[w, ia].astype(np.float64), lsm_ao[w, ib].astype(np.float64)
        pa, pb = anc_pos[w, ia], anc_pos[w, ib]
        Ga = _aug_cols(H_e, nll_e, lsm_ae[w, ia], st_e[w, ia], u, sdz, pa, w)
        Gb = _aug_cols(H_e, nll_e, lsm_ae[w, ib], st_e[w, ib], u, sdz, pb, w)
        Gao = _aug_cols(H_o, nll_o, lsm_ao[w, ia], st_o[w, ia], u, sdz, pa, w)
        Gbo = _aug_cols(H_o, nll_o, lsm_ao[w, ib], st_o[w, ib], u, sdz, pb, w)
        chk = {}                          # full / fullQ / fullP totals, for the gates only

        def put(crit, l, Va, Vb, Vao, Vbo):
            rec[f"R_{crit}_l{l}_a{a}"] = Va - Vb
            rec[f"Ro_{crit}_l{l}_a{a}"] = Vao - Vbo
            rec[f"V_{crit}_l{l}_a{a}"] = Va
            rec[f"Vpre_{crit}_l{l}_a{a}"] = Vb
            rec[f"Vpreo_{crit}_l{l}_a{a}"] = Vbo

        for l in L_STORE:
            n0, n1 = f"l{l}_d{a}", f"l{l}_d{a + 1}"
            chk[("full", l)] = (lin_full(Xa, n0), lin_full(Xb_, n1),
                                lin_full(Xao, n0), lin_full(Xbo, n1))
            for arm in AUG_ARMS:
                sa, ba = parts(arm, n0, Xa, Ga)
                sb, bb = parts(arm, n1, Xb_, Gb)
                sao, bao = parts(arm, n0, Xao, Gao)
                sbo, bbo = parts(arm, n1, Xbo, Gbo)
                put(f"full{arm}st", l, sa, sb, sao, sbo)
                put(f"full{arm}bl", l, ba, bb, bao, bbo)
                chk[(f"full{arm}", l)] = (sa + ba, sb + bb, sao + bao, sbo + bbo)
            for nm in diet_names:
                put(f"lq_{nm}", l, lq(nm, n0, Qa), lq(nm, n1, Qb), lq(nm, n0, Qao),
                    lq(nm, n1, Qbo))
            for nm in mlp_diet_list:
                k0, k1 = (f"mq_{nm}", l, 0), (f"mq_{nm}", l, 1)
                put(f"mq_{nm}", l, mlp_pred(k0, Qa), mlp_pred(k1, Qb), mlp_pred(k0, Qao),
                    mlp_pred(k1, Qbo))
            k0, k1 = ("fullQM", l, 0), ("fullQM", l, 1)
            put("fullQM", l, mlp_pred(k0, np.concatenate([Xa, Qa], 1)),
                mlp_pred(k1, np.concatenate([Xb_, Qb], 1)),
                mlp_pred(k0, np.concatenate([Xao, Qao], 1)),
                mlp_pred(k1, np.concatenate([Xbo, Qbo], 1)))
            k0, k1 = ("fullMr", l, 0), ("fullMr", l, 1)
            chk[("fullMr", l)] = (mlp_pred(k0, Xa), mlp_pred(k1, Xb_), None, None)
        for l in LEVELS:
            li = LEVELS.index(l)
            for aa in A_LIST:
                good = (t0w + aa) <= T - 1
                tq = np.clip(t0w + aa, 0, T - 1)
                rec[f"cons_l{l}_a{aa}"] = ((y_edit[li][w, tq] != y_orig[li][w, tq])
                                           & good).astype(np.int8)
                rec[f"oe_l{l}_a{aa}"] = o_e[li][w, tq]
                rec[f"oo_l{l}_a{aa}"] = o_o[li][w, tq]
                rec[f"ok_l{l}_a{aa}"] = good.astype(np.int8)
        tem = np.asarray(rec["split"]) == 2
        for k_, vv in rec.items():
            a_ = np.asarray(vv)[tem]
            save[f"{an}__{k_}"] = a_.astype(np.float32) if a_.dtype == np.float64 else a_

        # ---- the gates on this anchor, against the banked _express npz ------------------
        pre = f"{an}__"
        g = gates["rows"]
        g[f"{an}/w"] = bool(np.array_equal(save[pre + "w"], xpr[pre + "w"]))
        g[f"{an}/t0"] = bool(np.array_equal(save[pre + "t0"], xpr[pre + "t0"]))
        lab_ok = True
        for l in LEVELS:
            for aa in A_LIST:
                for lk in ("oe", "oo", "cons", "ok"):
                    kk_ = f"{pre}{lk}_l{l}_a{aa}"
                    if kk_ in xpr.files:
                        lab_ok &= bool(np.array_equal(save[kk_], xpr[kk_]))
        g[f"{an}/labels"] = lab_ok
        for lk in ("nll_e", "H_pre"):
            g[f"{an}/{lk}"] = float(np.abs(save[pre + lk].astype(np.float64)
                                           - xpr[pre + lk].astype(np.float64)).max())
        cg = gates["cols"]
        for crit in ["full"] + [f"full{arm}" for arm in AUG_ARMS]:
            worst = 0.0
            for l in L_STORE:
                ga, gb, gao, gbo = chk[(crit, l)]
                for kind, arr in (("V", ga), ("Vpre", gb), ("R", ga - gb),
                                  ("Ro", gao - gbo), ("Vpreo", gbo)):
                    kk_ = f"{pre}{kind}_{crit}_l{l}_a{a}"
                    if kk_ in xpr.files:
                        worst = max(worst, float(np.abs(
                            arr[tem].astype(np.float32).astype(np.float64)
                            - xpr[kk_].astype(np.float64)).max()))
            cg[f"{an}/{crit}"] = worst
        for l in L_STORE:
            ga, gb, _, _ = chk[("fullMr", l)]
            kv, kp = f"{pre}V_fullM_l{l}_a0", f"{pre}Vpre_fullM_l{l}_a0"
            if kv in xpr.files:
                bv = xpr[kv].astype(np.float64)
                report["fullMr_vs_banked_fullM"][f"{an}/l{l}"] = {
                    "max_abs_V": float(np.abs(ga[tem] - bv).max()),
                    "max_abs_Vpre": float(np.abs(gb[tem] - xpr[kp].astype(np.float64)).max()),
                    "corr_V": float(np.corrcoef(ga[tem], bv)[0, 1]),
                    "sd_V": float(bv.std())}
        print(f"{an}: {int(tem.sum())} test rows  rows {g[f'{an}/w']} {g[f'{an}/t0']} "
              f"labels {lab_ok}  full {cg[f'{an}/full']:.1e}  fullQ {cg[f'{an}/fullQ']:.1e}  "
              f"fullP {cg[f'{an}/fullP']:.1e}", flush=True)

    # ---- the twins: every reader at offsets 0 and -1, plus forecasts and state difference --
    tws = {}
    if do_twins and "tw__w" in bank_npz.files:
        t0 = time.time()
        w = np.asarray(bank_npz["tw__w"]).astype(np.int64)
        tvp = np.asarray(bank_npz["tw__t_v"]).astype(np.int64)
        xc = np.asarray(bank_npz["tw__x_c"]).astype(np.int64)
        ar_ = np.arange(len(w))
        poffs_t = torch.as_tensor(POFFS, device=dev)
        per = {}
        for side in ("v", "t"):
            Wside = We[w].copy()
            if side == "t":
                Wside[ar_, tvp] = xc
            Hs = np.zeros((len(w), T), np.float32)
            nls = np.zeros((len(w), T), np.float32)
            st2 = np.zeros((len(w), 2, d), np.float32)
            ls2 = np.zeros((len(w), 2, v), np.float32)
            Fq_ = np.zeros((len(w), len(POFFS), v), np.float32)
            with torch.no_grad():
                for c0 in range(0, len(w), 512):
                    sl = slice(c0, min(c0 + 512, len(w)))
                    x = torch.as_tensor(Wside[sl, :T], device=dev)
                    lg, _, inter = model(x, return_intermediates=True)
                    lsm = torch.log_softmax(lg.float(), -1)
                    nxt = torch.as_tensor(Wside[sl, 1:T + 1], device=dev)
                    nls[sl] = (-lsm.gather(-1, nxt[..., None])[..., 0]).cpu().numpy()
                    Hs[sl] = (-(lsm.exp() * lsm).sum(-1)).cpu().numpy()
                    tt = torch.as_tensor(np.stack([tvp[sl] - 1, tvp[sl]], 1), device=dev)
                    ar = torch.arange(x.shape[0], device=dev)[:, None]
                    st2[sl] = inter[prim][ar, tt].float().cpu().numpy()
                    ls2[sl] = lsm[ar, tt].cpu().numpy()
                    tp = torch.clamp(torch.as_tensor(tvp[sl], device=dev)[:, None]
                                     + poffs_t[None, :], 0, T - 1)
                    Fq_[sl] = lsm[ar.expand_as(tp), tp].cpu().numpy()
            per[side] = (Hs, nls, st2, ls2, Fq_)
            s_tok = nls[ar_, tvp - 1]
            tws[f"s_{side}"] = s_tok
            for k_, (slot, p_) in {"pre": (0, tvp - 1), "ev": (1, tvp)}.items():
                G_ = _aug_cols(Hs, nls, ls2[:, slot], st2[:, slot], u, sdz, p_, ar_)
                Xs_ = st2[:, slot].astype(np.float64)
                Q_ = ls2[:, slot].astype(np.float64)
                dd = 1 if k_ == "pre" else 0
                for l in L_STORE:
                    nm_ = f"l{l}_d{dd}"
                    tws[f"tw_{k_}_full_l{l}_{side}"] = lin_full(Xs_, nm_)
                    for arm in AUG_ARMS:
                        sp_, bp_ = parts(arm, nm_, Xs_, G_)
                        tws[f"tw_{k_}_full{arm}st_l{l}_{side}"] = sp_
                        tws[f"tw_{k_}_full{arm}bl_l{l}_{side}"] = bp_
                        tws[f"tw_{k_}_full{arm}_l{l}_{side}"] = sp_ + bp_
                    tws[f"tw_{k_}_lq_full_l{l}_{side}"] = lq("full", nm_, Q_)
                    tws[f"tw_{k_}_mq_full_l{l}_{side}"] = mlp_pred(("mq_full", l, dd), Q_)
                    tws[f"tw_{k_}_fullQM_l{l}_{side}"] = mlp_pred(
                        ("fullQM", l, dd), np.concatenate([Xs_, Q_], 1))
                    tws[f"tw_{k_}_fullMr_l{l}_{side}"] = mlp_pred(("fullMr", l, dd), Xs_)
        # gates: surprisals, the identical prefix, arm 0 and the Q / P totals vs _express
        gt = gates["twins"]
        gt["s_v"] = float(np.abs(tws["s_v"] - np.asarray(bank_npz["tw__s_v"], np.float64)).max())
        gt["s_c"] = float(np.abs(tws["s_t"] - np.asarray(bank_npz["tw__s_c"], np.float64)).max())
        gt["pre_state_identical"] = float(np.abs(per["v"][2][:, 0] - per["t"][2][:, 0]).max())
        worst = {}
        for crit in ["full"] + [f"full{arm}" for arm in AUG_ARMS]:
            wv = 0.0
            for k_ in ("pre", "ev"):
                for l in L_STORE:
                    for side in ("v", "t"):
                        kk_ = f"tw_{k_}_{crit}_l{l}_{side}"
                        if kk_ in xpr.files:
                            wv = max(wv, float(np.abs(tws[kk_].astype(np.float32)
                                                      .astype(np.float64)
                                                      - xpr[kk_].astype(np.float64)).max()))
            worst[crit] = wv
            gates["cols"][f"twins/{crit}"] = wv
        assert np.array_equal(np.asarray(xpr["tw_w"]).astype(np.int64), w), "twin pair order"
        for l in L_STORE:
            kk_ = f"tw_ev_fullM_l{l}_v"
            if kk_ in xpr.files:
                report["fullMr_vs_banked_fullM"][f"twins/l{l}"] = float(np.abs(
                    tws[f"tw_ev_fullMr_l{l}_v"] - xpr[kk_].astype(np.float64)).max())
        ds_ = tws["s_v"] - tws["s_t"]
        keep = np.where(np.abs(ds_) <= SAVE_CALIPER)[0]
        tws["keep35"] = keep
        tws["F_v"] = per["v"][4][keep]
        tws["F_t"] = per["t"][4][keep]
        tws["dX"] = (per["v"][2][keep, 1] - per["t"][2][keep, 1]).astype(np.float32)
        tws["w"] = w
        print(f"twins {time.time() - t0:.1f}s  {len(w)} pairs ({len(keep)} within "
              f"{SAVE_CALIPER})  s_v {gt['s_v']:.1e}  s_c {gt['s_c']:.1e}  prefix "
              f"{gt['pre_state_identical']:.1e}  " +
              "  ".join(f"{k} {x:.1e}" for k, x in worst.items()), flush=True)
        tim["twins"] = time.time() - t0

    # ---- verdict, save ---------------------------------------------------------------------
    fails = []
    fails += [k_ for k_, x in gates["actor"].items() if x > 1e-9]
    fails += [k_ for k_, x in gates["val_r2"].items() if x > 1e-6]
    fails += [k_ for k_, x in gates["rows"].items() if (x is False) or
              (not isinstance(x, bool) and x > tol)]
    fails += [k_ for k_, x in gates["cols"].items() if x > tol]
    fails += [k_ for k_, x in gates["diets"].items() if (x is False) or
              (not isinstance(x, bool) and x > 1e-6)]
    fails += [f"twins/{k_}" for k_, x in gates["twins"].items() if x > tol]
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    coef = {}
    for crit in ("full", "fullQ", "fullP", "clean", "cleanQ", "cleanP"):
        coef[f"beta__{crit}"] = beta[crit].astype(np.float64)
    for nm in diet_names:
        coef[f"beta__lq_{nm}"] = beta[f"lq_{nm}"].astype(np.float64)
    res = {"ckpt": ckpt, "step": cfg.get("step"), "stim_tag": stim_tag, "n": n,
           "actor_clean_acc": actor_tab, "val_r2": val_r2, "gates": gates,
           "gate_fails": fails, "report": report, "span": span,
           "diets": diet_names, "mlp_diets": mlp_diet_list, "diet_gate": diet_gate,
           "diet_stats_bank": {k: bank["diet_stats"][k] for k in diet_names
                               if k in bank["diet_stats"]},
           "clean_critic_mean_outcome": bank.get("clean_critic_mean_outcome"),
           "gram_guard_logq_full_rel": gram_guard,
           "aug_cols": AUG, "aug_arms": {a_: ARMS[a_] for a_ in AUG_ARMS},
           "mu_aug": mu_aug.tolist(), "n_rows_tr": n_rows_tr,
           "names": names, "real_names": real_names, "poffs": POFFS,
           "save_caliper": SAVE_CALIPER, "mlp_steps": mlp_steps,
           "timing_s": tim, "peak_rss_gb": rss,
           "decomposition": "V = [x.b_s + b + mu.b_g] (st) + [(g - mu).b_g] (bl), mu = the "
                            "appended block's mean on the critic's training rows"}
    sfx = f"_norm_{stim_tag}{out_sfx}"
    with open(f"{stem}{sfx}.json", "w") as f:
        json.dump(res, f, cls=NumpyEncoder, indent=1)
    save["meta"] = np.array(json.dumps({"gate_fails": fails}))
    np.savez_compressed(f"{stem}{sfx}.npz", **save, **coef, basis__Vb=Vb.astype(np.float64))
    if tws:
        tsv = {}
        for k_, vv in tws.items():
            a_ = np.asarray(vv)
            tsv[k_] = a_.astype(np.float32) if a_.dtype == np.float64 else a_
        np.savez_compressed(f"{stem}{sfx}_twins.npz", **tsv)
    volume.commit()
    print(f"saved -> {stem}{sfx}.{{json,npz}} + _twins.npz  ({time.time() - t00:.0f}s, "
          f"peak RSS {rss:.1f} GB)  gate fails: {fails}", flush=True)
    assert not fails, f"GATE FAILED: {fails}"
    return {"ckpt": ckpt, "stim_tag": stim_tag, "step": cfg.get("step"),
            "l1_d0": {k: round(val_r2[k].get("l1_d0", float("nan")), 4)
                      for k in ("full", "fullQ", "fullP", "lq_full", "mq_full", "fullQM",
                                "fullMr")},
            "span_l1_d0": {k: round(span[f"{k}/l1_d0"], 4) for k in ("full", "fullQ", "fullP")},
            "sec": round(time.time() - t00, 1), "peak_rss_gb": round(rss, 2),
            "gate_fails": fails}


def readout_basis_of(model):
    """`projection/task.py`'s readout basis, verbatim: the orthonormal rows spanning what the
    output layer can express of the state (`W_U diag(gamma)`, the softmax's constant logit and
    layer norm's mean removed)."""
    Wu = model.lm_head.weight.detach().float().cpu().numpy()
    gam = model.transformer.ln_f.weight.detach().float().cpu().numpy()
    M = Wu * gam[None, :]
    M = M - M.mean(0, keepdims=True)
    M = M - M.mean(1, keepdims=True)
    _, Sm, Vtm = np.linalg.svd(M, full_matrices=False)
    rk = int((Sm > Sm.max() * 1e-5).sum())
    return Vtm[:rk]


@app.function(volumes={DATA_DIR: volume}, timeout=1800, memory=4096)
def basis_sweep(seeds: str = "42,43,44", steps: str = "0,8000,64000"):
    """CPU.  Writes `<stem>_readout_basis.npz` (`Vb`, float64) per checkpoint.  The first
    sweep's cells of record saved `basis__Vb` after a same-named local had overwritten it (a
    per-row column, not the basis); the span in their json was computed before that and is
    unaffected.  The reduction reads the basis from this file."""
    volume.reload()
    D_ = f"{DATA_DIR}/v16_s2_L6_m4_distinct/logit_reading"
    out = []
    for s in seeds.split(","):
        for st in steps.split(","):
            ck = f"{D_}/traj_a1_s{s}/step{int(st):06d}.pt"
            model, _, _, _ = load_trajectory_ckpt(ck, "cpu")
            Vb = readout_basis_of(model)
            np.savez_compressed(f"{ck[:-3]}_readout_basis.npz", Vb=Vb.astype(np.float64))
            out.append(f"{ck} rank {Vb.shape[0]}")
            print(out[-1], flush=True)
    volume.commit()
    return out


@app.function(volumes={DATA_DIR: volume}, timeout=4 * 3600, memory=2048)
def public_sweep(seeds: str = "42,43,44", steps: str = "0,8000,64000",
                 tags: str = "a1,swap65k", mlp_diets: str = "all", out_sfx: str = "_public",
                 skip_done: bool = True):
    """One L4 container per (checkpoint, venue), fanned out from this CPU coordinator.
    `skip_done` skips a cell whose json already exists with no gate failure (the smoke)."""
    volume.reload()
    D_ = f"{DATA_DIR}/v16_s2_L6_m4_distinct/logit_reading"
    args = []
    for s in seeds.split(","):
        for st in steps.split(","):
            ck = f"{D_}/traj_a1_s{s}/step{int(st):06d}.pt"
            for tg in tags.split(","):
                need = [f"{ck[:-3]}_norm_{tg}.npz", f"{ck[:-3]}_norm_{tg}_express.npz"]
                if not (os.path.exists(ck) and all(os.path.exists(x) for x in need)):
                    print(f"MISSING {ck} {tg}", flush=True)
                    continue
                done = f"{ck[:-3]}_norm_{tg}{out_sfx}.json"
                if skip_done and os.path.exists(done) and not json.load(open(done))["gate_fails"]:
                    print(f"SKIP (done, gates pass) {done}", flush=True)
                    continue
                args.append((ck, tg))
    print(f"{len(args)} cells", flush=True)
    outs = list(public_ckpt.starmap(args, kwargs={"mlp_diets": mlp_diets, "out_sfx": out_sfx},
                                    return_exceptions=True))
    for a_, o in zip(args, outs):
        print(f"{a_[0].split('/')[-2]}/{a_[0].split('/')[-1]} {a_[1]}: "
              f"{json.dumps(o, cls=NumpyEncoder, default=str)[:700]}", flush=True)
    return [str(o)[:700] for o in outs]
