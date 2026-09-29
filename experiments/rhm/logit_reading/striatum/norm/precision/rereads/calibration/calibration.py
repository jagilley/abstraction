"""The norm's calibration re-read on the richer readers: does the world-calibrated level, its
shift-versus-rescaling and the outcome surprise's ordering hold when the critic carries more
of the belief than the raw-state ridge does?

`norm/README.md` 1 banked, on three trajectory seeds and with the ridge on the raw state only,
that the critic's pre-event level `V_pre = V[l, a+1](s_(t0-1))` orders with each diet's
expected outcome (Spearman +1.000 inside every family at every level), that regressing each
diet's norm on the `full` critic's gives slope ~1.0 at random init and 1.16 (`out_lo`) ->
0.82 (`out_hi`) at 64k, and that the outcome surprise on the fixed rows orders inversely with
the world.  `precision/express.py` found that readers with more of the belief (`+log q`, an
MLP on the state) read damage better than the ridge and reverse the response's entropy
dependence at l = 1; it fitted only the `full` and `clean` diets.  This file fits the same
readers on every diet.

Everything up to the Gram is `norm/task.py` (via `express.py`'s copy of it, which is gated
bit-identical): the same trunk, windows, split, frozen clean-trained actor, outcome labels
and the eleven window diets (`strat_diets` / `mix_diets` / `anchor_diets` at seeds
`seed + 7 .. 10`, rebuilt deterministically from the training windows and gated against the
banked `diet_stats`).  For every diet, as a mask over the shared rows exactly as
`norm/task.py` masks the state Gram, three Grams are accumulated:

  arm 0   [s, 1]              `norm/task.py`'s critic, recomputed in its own order -- THE GATE
  aug     [s, G, 1]           G = `express.py`'s K appended columns (`_aug_cols`), from which
                              the `Q` (+ log q_t, 16 dims) and `P` (+ H(q_t), H(q_(t-1)), s_t,
                              H(q_(t-1)) s_t) arms are sub-blocks; the appended columns are
                              standardised to the state's per-dimension RMS on THAT DIET's own
                              training rows (`_standardiser`), fitted, and mapped back
  ln      [ln_f(s), 1]        the `L` map arm: the model's own final layer norm in place of
                              the state (what the unembedding reads)

each solved with `norm/task.py`'s lambda ladder and its per-target-column lambda selection on
the SAME held-out validation rows (shared across diets, as in `norm/task.py`).  Plus `M`, a
128-unit MLP on the state (`coeruleus.readout.train_head`, `striatum/task.py`'s 2000 steps,
the seeds `express.py` used), levels 1-4 at a = 0 only, on the diets in `mlp_diets`.

Columns written, on the `split == 2` test rows of both anchors, in the banked row order:
`Vpre_<diet><arm>_l<l>_a<a>` (a = 0, 4, 8; a = 0 only for `M`), `V_<diet><arm>_l<l>_a0` and
`R_<diet><arm>_l<l>_a0`, for arm in {Q, P, L, M}.  Arm 0 is not re-written: the banked npz
holds it and the gate asserts equality.

**Gates**, all asserted before the cell returns (the json and npz are written first, with
the failures listed, so a failing cell can be diagnosed):
  - the frozen actor's clean accuracy equals the banked json at every level (exact);
  - every diet's size and E[outcome] equal the banked `diet_stats` (the diets are the bank's);
  - arm 0's held-out `R2` equals the banked `critic_val_r2[<diet>]` on every stored column,
    for every diet (1e-6);
  - the test rows (`w`, `t0`) and the outcome labels equal the banked npz exactly, `nll_e`
    and `H_pre` to float32;
  - arm 0's `V` / `Vpre` / `R` / `Ro` / `Vpreo` equal the banked `<diet>` columns to float32
    (`tol`), for every diet, both anchors, every level and offset;
  - the `full` diet's `Q` / `P` columns equal `express.py`'s banked `fullQ` / `fullP`, and `L`
    equals `express_ln`'s `fullL` (`tol`).  `fullM` against `express.py`'s `fullM` is
    recorded, not asserted (Adam on a GPU is not bit-reproducible run to run).

Run (from `experiments/`, MODAL_PROFILE=chromatic):
  modal run -m rhm.logit_reading.striatum.norm.precision.rereads.calibration.calibration::calib_ckpt \
      --ckpt /data/v16_s2_L6_m4_distinct/logit_reading/traj_a1_s42/step064000.pt --out-sfx _calib_smoke
  modal run --detach -m rhm.logit_reading.striatum.norm.precision.rereads.calibration.calibration::calib_sweep
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
from rhm.logit_reading.striatum.norm.task import (LEVELS, DMAX, DD_LIST, A_STORE, L_STORE,
                                                  T_LO, LAMS, tgt_names)
from rhm.logit_reading.striatum.norm.diets import strat_diets, mix_diets, anchor_diets
from rhm.logit_reading.striatum.norm.precision.express import (_aug_cols, _standardiser, K,
                                                               ARMS)


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
app = modal.App("rhm-norm-precision-calib", image=image)

MLP_DD = (0, 1)
A_VPRE = (0, 4, 8)                  # the outcome surprise's offsets (norm README 2)
# every diet (the committed sweep); the smoke ran the five-diet subset
DEFAULT_MLP_DIETS = ("out_lo,out_mid,out_hi,dmg_lo,dmg_mid,dmg_hi,mix_q00,mix_q25,mix_q51,full,"
                     "natural_sized")


def _fit_arm_gpu(A, C, Fv, Yv, yvar, cols, lams, dev):
    """`express._fit_arm` with the held-out evaluation on the GPU in float64.  The ridge solve
    itself is `ridge_solve` on the CPU, as everywhere in the arc.  `Fv`, `Yv`, `yvar` are torch
    float64 on `dev`, `yvar` the UNBIASED variance (torch's `var(0)`, which `norm/task.py`
    uses).  Returns (beta (len(cols)+1, n_tgt) numpy, best R2 numpy, lambda picked)."""
    import torch
    idx = list(cols) + [A.shape[0] - 1]
    As, Cs = A[np.ix_(idx, idx)], C[idx]
    Xs = Fv[:, torch.as_tensor(idx, device=dev)]
    best = torch.full((C.shape[1],), -1e18, dtype=torch.float64, device=dev)
    Bb = torch.zeros(len(idx), C.shape[1], dtype=torch.float64, device=dev)
    lam = np.zeros(C.shape[1])
    for lm in lams:
        B = torch.as_tensor(ridge_solve(As, Cs, lm), device=dev)
        r2 = 1.0 - ((Xs @ B - Yv) ** 2).mean(0) / torch.clamp(yvar, min=1e-12)
        up = r2 > best
        best = torch.where(up, r2, best)
        Bb[:, up] = B[:, up]
        lam[up.cpu().numpy()] = lm
    return Bb.cpu().numpy(), best.cpu().numpy(), lam


@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=2 * 3600, memory=8192,
              cpu=4.0, max_containers=9)
def calib_ckpt(ckpt: str, stim_tag: str = "a1", n_clean: int = 6144, n_clean_val: int = 2048,
               seed: int = 11, actor_steps: int = 3000, eval_seed: int = 5150,
               prim_block: str = "post_block7", n_val_windows: int = 1500, chunk: int = 512,
               strat_frac: float = 1.0 / 3.0, arm_set: str = "Q,P", map_arms: str = "L",
               mlp_diets: str = DEFAULT_MLP_DIETS, mlp_steps: int = 2000,
               tol: float = 5e-5, u_seed: int = 4242, out_sfx: str = "_calib"):
    import torch
    volume.reload()
    t00 = time.time()
    prof = {}
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
    ex_npz = {sfx: np.load(f"{stem}_norm_{stim_tag}{sfx}.npz")
              for sfx in ("_express", "_express_ln")
              if os.path.exists(f"{stem}_norm_{stim_tag}{sfx}.npz")}
    arms_aug = {a: ARMS[a] for a in arm_set.split(",") if a}
    arms_map = [a for a in map_arms.split(",") if a]
    assert all(a == "L" for a in arms_map), f"only the L map arm is implemented: {arms_map}"
    lnf = model.transformer.ln_f
    print(f"loaded {ckpt} step {cfg.get('step')} d={d} T={T}", flush=True)

    # ---- norm/task.py verbatim: windows, split ---------------------------------------------
    n_all = S["windows_edit"].shape[0]
    assert int(bank["n"]) == n_all, "the banked cell ran on a window subsample"
    idx = np.arange(n_all)
    We, Wo = S["windows_edit"][idx], S["windows_orig"][idx]
    meta = {k: S[k][idx] for k in ("phase", "etype", "j", "e", "first_diff", "t_v", "k_star")}
    y_edit = P_["y_edit"][:, idx][:, :, :T]
    y_orig = P_["y_orig"][:, idx][:, :, :T]
    n = len(idx)
    et, jj, ee, fd, tv = (meta["etype"], meta["j"], meta["e"], meta["first_diff"],
                          meta["t_v"])
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

    # ---- the frozen actors (express.py's copy of norm/task.py; each reseeds itself) ------
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
    prof["actors"] = time.time() - t0
    print(f"actors {prof['actors']:.1f}s", flush=True)

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

    # ---- the two forward passes plus the appended features (express.py's run) -------------
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
    prof["passes"] = time.time() - t0
    print(f"passes {prof['passes']:.1f}s  cache {cache.nbytes / 1e9:.2f} GB", flush=True)

    # ---- the diets: norm/task.py lines 256-283 verbatim --------------------------------------
    dmg_full = (o_o.astype(np.int16) - o_e.astype(np.int16)) > 0
    dmg_w = dmg_full[:4][:, :, rows]
    dmg_win = dmg_w.mean((0, 2))
    out_win = o_e[:4][:, :, rows].astype(np.float32).mean((0, 2))
    masks, spec = {}, {}
    ms, sp, _ = strat_diets(out_win, et, jj, is_tr, seed=seed + 7, frac=strat_frac,
                            prefix="out")
    masks.update(ms); spec.update(sp)
    n_strat = int(max(m.sum() for m in ms.values()))
    ms, sp, _ = strat_diets(dmg_win, et, jj, is_tr, seed=seed + 8, frac=strat_frac,
                            prefix="dmg")
    masks.update(ms); spec.update(sp)
    ms, sp, _ = mix_diets(et, is_tr, n_total=n_strat, seed=seed + 9)
    masks.update(ms); spec.update(sp)
    ms, sp = anchor_diets(et, is_tr, n_total=n_strat, seed=seed + 10)
    masks.update(ms); spec.update(sp)
    for k_ in list(masks):
        masks[k_] = np.repeat(masks[k_][tr_idx][:, None], R_, 1)
    diet_names = list(masks)
    o_row_tr = o_e[:4][:, tr_idx][:, :, rows].astype(np.float32).mean(0)
    gates = {"actor": {}, "diets": {}, "val_r2": {}, "rows": {}, "cols": {}, "express": {}}
    for k_, a_ in actor_tab.items():
        gates["actor"][k_] = abs(a_ - bank["actor_clean_acc"][k_])
    for nm in diet_names:
        m = np.asarray(masks[nm])
        bs_ = bank["diet_stats"][nm]
        gates["diets"][nm] = max(abs(int(m.sum()) - int(bs_["n_rows"])),
                                 abs(int(m.any(1).sum()) - int(bs_["n_windows"])),
                                 abs(float(o_row_tr[m].mean()) - float(bs_["mean_outcome"])))
    print(f"{len(diet_names)} diets: {diet_names}  worst stats gap "
          f"{max(gates['diets'].values()):.2e}", flush=True)

    # ---- Grams per diet: arm 0 (norm/task.py's order), the augmented block, ln_f -----------
    names = tgt_names()
    n_tgt = len(names)
    D = d + K + 1
    A0 = {nm: torch.zeros(d + 1, d + 1, dtype=torch.float64, device=dev) for nm in diet_names}
    C0 = {nm: torch.zeros(d + 1, n_tgt, dtype=torch.float64, device=dev) for nm in diet_names}
    Aa = {nm: torch.zeros(D, D, dtype=torch.float64, device=dev) for nm in diet_names}
    Ca = {nm: torch.zeros(D, n_tgt, dtype=torch.float64, device=dev) for nm in diet_names}
    if arms_map:
        Al = {nm: torch.zeros(d + 1, d + 1, dtype=torch.float64, device=dev)
              for nm in diet_names}
        Cl = {nm: torch.zeros(d + 1, n_tgt, dtype=torch.float64, device=dev)
              for nm in diet_names}
    shuf = np.random.default_rng(seed + 3).permutation(len(tr_idx))
    t0 = time.time()
    for c0 in range(0, len(tr_idx), chunk):
        sl = slice(c0, min(c0 + chunk, len(tr_idx)))
        wsl = tr_idx[sl]
        X = torch.as_tensor(cache[sl].reshape(-1, d), device=dev).double()
        ones = torch.ones(len(X), 1, dtype=torch.float64, device=dev)
        X1 = torch.cat([X, ones], 1)
        G = torch.as_tensor(aug_c[sl].reshape(-1, K), device=dev).double()
        F = torch.cat([X, G, ones], 1)
        if arms_map:
            with torch.no_grad():
                LX1 = torch.cat([lnf(X.float()).double(), ones], 1)
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
            Xs, Ys = X1[sel], Y[sel]
            A0[nm] += Xs.T @ Xs
            C0[nm] += Xs.T @ Ys
            Fs = F[sel]
            Aa[nm] += Fs.T @ Fs
            Ca[nm] += Fs.T @ Ys
            if arms_map:
                Ls = LX1[sel]
                Al[nm] += Ls.T @ Ls
                Cl[nm] += Ls.T @ Ys
        del X, X1, G, F, Y
    torch.cuda.synchronize()
    prof["grams"] = time.time() - t0
    print(f"grams {prof['grams']:.1f}s", flush=True)

    # ---- the held-out validation rows (norm/task.py's, shared by every diet) -----------------
    t0 = time.time()
    Xv_np = cache[len(tr_idx):].reshape(-1, d).astype(np.float64)
    Xv1 = torch.as_tensor(np.concatenate([Xv_np, np.ones((len(Xv_np), 1))], 1), device=dev)
    yv = []
    for li in range(len(LEVELS)):
        for dd in DD_LIST:
            yv.append(torch.as_tensor(o_e[li][np.ix_(va_idx, rows + dd)],
                                      dtype=torch.float64, device=dev))
    Yv = torch.stack(yv, -1).reshape(-1, n_tgt // 2)
    Yv = torch.cat([Yv, Yv], 1)
    yvar = Yv.var(0)
    Gv = torch.as_tensor(aug_c[len(tr_idx):].reshape(-1, K), device=dev).double()
    Fv_raw = torch.cat([Xv1[:, :d], Gv, Xv1[:, d:]], 1)
    if arms_map:
        with torch.no_grad():
            LXv1 = torch.cat([lnf(Xv1[:, :d].float()).double(), Xv1[:, d:]], 1)

    beta, val_r2, lam_pick, scale_info = {}, {}, {}, {}
    for nm in diet_names:
        # arm 0: norm/task.py's own loop, bit for bit
        An, Cn = A0[nm].cpu().numpy(), C0[nm].cpu().numpy()
        best = torch.full((n_tgt,), -1e18, dtype=torch.float64, device=dev)
        Bb = torch.zeros(d + 1, n_tgt, dtype=torch.float64, device=dev)
        for lm in LAMS:
            B = torch.as_tensor(ridge_solve(An, Cn, lm), device=dev)
            r2 = 1.0 - ((Xv1 @ B - Yv) ** 2).mean(0) / torch.clamp(yvar, min=1e-12)
            up = r2 > best
            best = torch.where(up, r2, best)
            Bb[:, up] = B[:, up]
        beta[nm] = Bb.cpu().numpy()
        val_r2[nm] = {names[i]: float(best[i]) for i in range(n_tgt)}
        # the augmented arms: standardise the appended block on THIS diet's rows
        Aan, Can = Aa[nm].cpu().numpy(), Ca[nm].cpu().numpy()
        n_rows = int(round(Aan[-1, -1]))
        Tst = _standardiser(Aan, d, n_rows)
        As_, Cs_ = Tst.T @ Aan @ Tst, Tst.T @ Can
        Fv = Fv_raw @ torch.as_tensor(Tst, device=dev)
        scale_info[nm] = {"n_rows": n_rows,
                          "s_state": float(np.sqrt(np.trace(Aan[:d, :d]) / (n_rows * d)))}
        for arm, cols in arms_aug.items():
            cc = list(range(d)) + [d + c for c in cols]
            B, r2, lp = _fit_arm_gpu(As_, Cs_, Fv, Yv, yvar, cc, LAMS, dev)
            full_b = np.zeros((D, n_tgt))
            full_b[cc + [D - 1]] = B
            raw = Tst @ full_b
            beta[f"{nm}{arm}"] = raw[cc + [D - 1]]
            beta[f"{nm}{arm}__cols"] = cc
            val_r2[f"{nm}{arm}"] = {names[i]: float(r2[i]) for i in range(n_tgt)}
            lam_pick[f"{nm}{arm}"] = {names[i]: float(lp[i]) for i in range(n_tgt // 2)}
        del Fv
        if arms_map:
            B, r2, lp = _fit_arm_gpu(Al[nm].cpu().numpy(), Cl[nm].cpu().numpy(), LXv1, Yv,
                                     yvar, list(range(d)), LAMS, dev)
            beta[f"{nm}L"] = B
            beta[f"{nm}L__map"] = "L"
            val_r2[f"{nm}L"] = {names[i]: float(r2[i]) for i in range(n_tgt)}
            lam_pick[f"{nm}L"] = {names[i]: float(lp[i]) for i in range(n_tgt // 2)}
    del A0, C0, Aa, Ca, Xv1, Fv_raw, Gv
    if arms_map:
        del Al, Cl, LXv1
    torch.cuda.empty_cache()
    prof["solves"] = time.time() - t0
    print(f"solves {prof['solves']:.1f}s  l1_d1 R2 full {val_r2['full']['l1_d1']:.4f} " +
          " ".join(f"{a} {val_r2['full' + a]['l1_d1']:.4f}" for a in list(arms_aug) + arms_map),
          flush=True)

    # ---- the MLP arm, per diet ---------------------------------------------------------------
    mlp = {}
    mlp_list = [x for x in mlp_diets.split(",") if x]
    if mlp_list:
        from rhm.logit_reading.coeruleus.readout import train_head
        from rhm.logit_reading.striatum.task import _r2
        t0 = time.time()
        Xtr_all = cache[:len(tr_idx)].reshape(-1, d).astype(np.float32)
        Xva_m = cache[len(tr_idx):].reshape(-1, d).astype(np.float32)
        for nm in mlp_list:
            mk = np.asarray(masks[nm]).reshape(-1)
            Xtr_m = Xtr_all if mk.all() else Xtr_all[mk]
            for l in L_STORE:
                li = LEVELS.index(l)
                for dd in MLP_DD:
                    ytr = o_e[li][tr_idx][:, rows + dd].reshape(-1).astype(np.float32)
                    if not mk.all():
                        ytr = ytr[mk]
                    yva = o_e[li][va_idx][:, rows + dd].reshape(-1).astype(np.float64)
                    h = train_head(Xtr_m, ytr, steps=mlp_steps, device=dev, seed=l * 17 + dd)
                    mlp[(nm, f"l{l}_d{dd}")] = h
                    val_r2.setdefault(f"{nm}M", {})[f"l{l}_d{dd}"] = _r2(h(Xva_m), yva)
            del Xtr_m
            print(f"  mlp {nm} {time.time() - t0:.0f}s  " + " ".join(
                f"{k} {x:.4f}" for k, x in val_r2[f"{nm}M"].items()), flush=True)
        del Xtr_all, Xva_m
        torch.cuda.empty_cache()
        prof["mlp"] = time.time() - t0
        prof["mlp_per_diet"] = prof["mlp"] / len(mlp_list)

    # ---- prediction on the anchors' test rows ------------------------------------------------
    t0 = time.time()

    def colidx(name):
        return names.index(name)

    def lin(nm, name, Xs, Gs, Ls):
        b = beta[nm][:, colidx(name)]
        if nm in diet_names:
            return Xs @ b[:-1] + b[-1]
        if beta.get(f"{nm}__map") == "L":
            return Ls @ b[:-1] + b[-1]
        cc = beta[f"{nm}__cols"]
        return np.concatenate([Xs, Gs], 1)[:, cc] @ b[:-1] + b[-1]

    def lnmap(X32):
        if not arms_map:
            return None
        with torch.no_grad():
            return lnf(torch.as_tensor(np.asarray(X32, np.float32), device=dev)
                       ).double().cpu().numpy()

    arm_cols = [f"{nm}{a}" for nm in diet_names for a in list(arms_aug) + arms_map]
    save = {}
    for an, t0a in anchors.items():
        ok = anchor_ok[an]
        w_all = np.where(ok)[0]
        tem = split[w_all] == 2
        w = w_all[tem]
        ib, ia = ANC[an]
        t0w = t0a[w]
        pre = f"{an}__"
        save[pre + "w"] = w
        save[pre + "t0"] = t0w
        g = gates["rows"]
        g[f"{an}/w"] = bool(np.array_equal(w, bank_npz[pre + "w"]))
        g[f"{an}/t0"] = bool(np.array_equal(t0w, bank_npz[pre + "t0"]))
        lab_ok = True
        for l in LEVELS:
            li = LEVELS.index(l)
            for a in (0, 4, 8):
                good = (t0w + a) <= T - 1
                tq = np.clip(t0w + a, 0, T - 1)
                for lk, arr in (("oe", o_e[li][w, tq]), ("oo", o_o[li][w, tq]),
                                ("ok", good.astype(np.int8))):
                    kk_ = f"{pre}{lk}_l{l}_a{a}"
                    if kk_ in bank_npz.files:
                        lab_ok &= bool(np.array_equal(arr, bank_npz[kk_]))
        g[f"{an}/labels"] = lab_ok
        for lk, arr in (("nll_e", nll_e[w, t0w - 1]), ("H_pre", H_e[w, t0w - 1])):
            g[f"{an}/{lk}"] = float(np.abs(arr.astype(np.float64)
                                           - bank_npz[pre + lk].astype(np.float64)).max())
        Xa, Xb_ = st_e[w, ia].astype(np.float64), st_e[w, ib].astype(np.float64)
        Xao, Xbo = st_o[w, ia].astype(np.float64), st_o[w, ib].astype(np.float64)
        pa, pb = anc_pos[w, ia], anc_pos[w, ib]
        Ga = _aug_cols(H_e, nll_e, lsm_ae[w, ia], st_e[w, ia], u, sdz, pa, w)
        Gb = _aug_cols(H_e, nll_e, lsm_ae[w, ib], st_e[w, ib], u, sdz, pb, w)
        La, Lb = lnmap(st_e[w, ia]), lnmap(st_e[w, ib])
        # arm 0 per diet: the gate against the banked columns
        cg = gates["cols"]
        for nm in diet_names:
            worst = 0.0
            for l in L_STORE:
                for a in A_STORE:
                    n0, n1 = f"l{l}_d{a}", f"l{l}_d{a + 1}"
                    Va, Vb = lin(nm, n0, Xa, None, None), lin(nm, n1, Xb_, None, None)
                    Vao, Vbo = lin(nm, n0, Xao, None, None), lin(nm, n1, Xbo, None, None)
                    for kind, arr in (("V", Va), ("Vpre", Vb), ("R", Va - Vb),
                                      ("Ro", Vao - Vbo), ("Vpreo", Vbo)):
                        kk_ = f"{pre}{kind}_{nm}_l{l}_a{a}"
                        if kk_ in bank_npz.files:
                            worst = max(worst, float(np.abs(
                                arr.astype(np.float32).astype(np.float64)
                                - bank_npz[kk_].astype(np.float64)).max()))
            cg[f"{an}/{nm}"] = worst
        # the arms
        for nm in arm_cols:
            for l in L_STORE:
                for a in A_VPRE:
                    n1 = f"l{l}_d{a + 1}"
                    save[f"{pre}Vpre_{nm}_l{l}_a{a}"] = lin(nm, n1, Xb_, Gb, Lb).astype(np.float32)
                Va = lin(nm, f"l{l}_d0", Xa, Ga, La)
                save[f"{pre}V_{nm}_l{l}_a0"] = Va.astype(np.float32)
                save[f"{pre}R_{nm}_l{l}_a0"] = (Va - lin(nm, f"l{l}_d1", Xb_, Gb, Lb)
                                                ).astype(np.float32)
        for nm in mlp_list:
            for l in L_STORE:
                Va = mlp[(nm, f"l{l}_d0")](Xa.astype(np.float32)).astype(np.float64)
                Vb = mlp[(nm, f"l{l}_d1")](Xb_.astype(np.float32)).astype(np.float64)
                save[f"{pre}Vpre_{nm}M_l{l}_a0"] = Vb.astype(np.float32)
                save[f"{pre}V_{nm}M_l{l}_a0"] = Va.astype(np.float32)
                save[f"{pre}R_{nm}M_l{l}_a0"] = (Va - Vb).astype(np.float32)
        # the full diet's arms against express.py's banked refits
        eg = gates["express"]
        for arm, sfx in (("Q", "_express"), ("P", "_express"), ("L", "_express_ln"),
                         ("M", "_express")):
            Z = ex_npz.get(sfx)
            mine = f"full{arm}"
            if Z is None or (arm == "M" and "full" not in mlp_list) or \
                    (arm != "M" and mine not in arm_cols):
                continue
            worst, n_cmp = 0.0, 0
            for l in L_STORE:
                for kind, a in (("Vpre", 0), ("V", 0), ("R", 0), ("Vpre", 4), ("Vpre", 8)):
                    kk_ = f"{pre}{kind}_{mine}_l{l}_a{a}"
                    if kk_ in Z.files and kk_ in save:
                        worst = max(worst, float(np.abs(save[kk_].astype(np.float64)
                                                        - Z[kk_].astype(np.float64)).max()))
                        n_cmp += 1
            eg[f"{an}/{mine}"] = {"max_abs": worst, "n_cols": n_cmp}
        print(f"{an}: {len(w)} test rows  rows {g[f'{an}/w']} {g[f'{an}/t0']} labels "
              f"{lab_ok}  nll {g[f'{an}/nll_e']:.1e}  H {g[f'{an}/H_pre']:.1e}  arm-0 worst "
              f"{max(cg[f'{an}/{nm}'] for nm in diet_names):.2e}  express " +
              " ".join(f"{k.split('/')[1]} {x['max_abs']:.1e}" for k, x in eg.items()
                       if k.startswith(an)), flush=True)
    prof["predict"] = time.time() - t0

    # arm 0's held-out fit against the banked json, every diet
    for nm in diet_names:
        bv = bank["critic_val_r2"][nm]
        gates["val_r2"][nm] = max(abs(val_r2[nm][k] - bv[k]) for k in bv
                                  if not k.startswith("_"))

    # ---- verdict, save -----------------------------------------------------------------------
    fails = []
    fails += [f"actor/{k_}" for k_, x in gates["actor"].items() if x > 1e-9]
    fails += [f"diets/{k_}" for k_, x in gates["diets"].items() if x > 1e-9]
    fails += [f"val_r2/{k_}" for k_, x in gates["val_r2"].items() if x > 1e-6]
    fails += [f"rows/{k_}" for k_, x in gates["rows"].items() if (x is False) or
              (not isinstance(x, bool) and x > tol)]
    fails += [f"cols/{k_}" for k_, x in gates["cols"].items() if x > tol]
    fails += [f"express/{k_}" for k_, x in gates["express"].items()
              if not k_.endswith("fullM") and x["max_abs"] > tol]
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    prof["total"] = time.time() - t00
    res = {"ckpt": ckpt, "step": cfg.get("step"), "stim_tag": stim_tag, "n": n,
           "diets": diet_names, "diet_spec": spec, "arm_set": arm_set, "map_arms": map_arms,
           "mlp_diets": mlp_list, "mlp_steps": mlp_steps, "actor_clean_acc": actor_tab,
           "val_r2": val_r2, "lam_pick": lam_pick, "scale": scale_info, "gates": gates,
           "gate_fails": fails, "tol": tol, "u_seed": u_seed, "sdz": sdz,
           "arms": {"Q": "state + log q_t (16), standardised on the diet's rows",
                    "P": "state + H(q_t) + H(q_(t-1)) + s_t + H(q_(t-1)) s_t",
                    "L": "ln_f(s) in place of the state",
                    "M": "MLP on the state (hidden 128), l1-4, a = 0"},
           "profile_s": prof, "peak_rss_gb": rss}
    sfx = f"_norm_{stim_tag}{out_sfx}"
    with open(f"{stem}{sfx}.json", "w") as f:
        json.dump(res, f, cls=NumpyEncoder, indent=1)
    save["meta"] = np.array(json.dumps({"gate_fails": fails}))
    np.savez_compressed(f"{stem}{sfx}.npz", **save)
    volume.commit()
    print(f"saved -> {stem}{sfx}.npz  ({prof['total']:.0f}s, peak RSS {rss:.2f} GB)  "
          f"profile {json.dumps({k: round(x, 1) for k, x in prof.items()})}  "
          f"gate fails: {fails}", flush=True)
    assert not fails, f"GATE FAILED: {fails}"
    return {"ckpt": ckpt, "step": cfg.get("step"), "gate_fails": fails,
            "profile_s": {k: round(x, 1) for k, x in prof.items()}, "peak_rss_gb": rss}


@app.function(volumes={DATA_DIR: volume}, timeout=4 * 3600, memory=1024)
def calib_sweep(seeds: str = "42,43,44", steps: str = "0,8000,64000", stim_tag: str = "a1",
                arm_set: str = "Q,P", map_arms: str = "L", mlp_diets: str = DEFAULT_MLP_DIETS,
                out_sfx: str = "_calib"):
    """One container per checkpoint, fanned out from this CPU coordinator."""
    volume.reload()
    D_ = f"{DATA_DIR}/v16_s2_L6_m4_distinct/logit_reading"
    args = []
    for s in seeds.split(","):
        for st in steps.split(","):
            ck = f"{D_}/traj_a1_s{s}/step{int(st):06d}.pt"
            need = f"{ck[:-3]}_norm_{stim_tag}.npz"
            if not (os.path.exists(ck) and os.path.exists(need)):
                print(f"MISSING {ck} ({need})", flush=True)
                continue
            # (ckpt, stim_tag, n_clean, n_clean_val, seed, actor_steps, eval_seed, prim_block,
            #  n_val_windows, chunk, strat_frac, arm_set, map_arms, mlp_diets, mlp_steps, tol,
            #  u_seed, out_sfx)
            args.append((ck, stim_tag, 6144, 2048, 11, 3000, 5150, "post_block7", 1500, 512,
                         1.0 / 3.0, arm_set, map_arms, mlp_diets, 2000, 5e-5, 4242, out_sfx))
    print(f"{len(args)} cells", flush=True)
    outs = list(calib_ckpt.starmap(args, return_exceptions=True))
    for a, o in zip(args, outs):
        print(f"{a[0].split('/')[-2]}/{a[0].split('/')[-1]}: "
              f"{json.dumps(o, cls=NumpyEncoder, default=str)[:600]}", flush=True)
    print("SWEEP DONE", flush=True)
    return [str(o)[:600] for o in outs]
