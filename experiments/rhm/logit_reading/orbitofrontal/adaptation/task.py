"""adaptation: the norm as a RUNNING estimate -- how fast does an outcome-trained critic
re-calibrate when its world changes, and what do the response and the outcome surprise do
during the transition?

`norm/` is between-subjects: three critics raised in three worlds, read on identical
held-out rows.  Its finding is that the critic's pre-event level `V(s_{t-1})` orders with
the diet's mean outcome at rank one inside every family, that a random-init trunk's norm
shifts as a pure intercept where the trained trunk's also rescales, and that the
anticipatory response `R` at the event does NOT order with the world.

Xiang, Lohrenz & Montague's design is within-subject: one responder is adapted to low
offers over thirty trials and then given medium offers, and the same medium offer feels
different to the group that came from low than to the group that came from high, with the
feeling tracking a running Bayesian estimate of the offer distribution.  The
between-subjects version cannot say anything about the running part.  This node asks it:

  Q1  How fast does the norm re-calibrate after the world changes, in windows?
  Q2  Is the re-calibration a shift first and a rescaling later, or both at once?
  Q3  At the same number of windows after the switch, do the lo->mid and hi->mid critics
      differ on the SAME rows in `V_pre` (they should, transiently), in the outcome
      surprise `d` (arithmetic given the first), and in the response `R` (the open
      question, since between-subjects `R` did not order with the world)?
  Q4  Does `d`'s mean drift off zero at the switch and return?  That is the value side's
      "has the world moved" organ, in outcome currency -- the counterpart of the
      structural side's excess over stated entropy.

Everything below the Gram is `norm/task.py`'s design verbatim (same split seed, same
actor recipe, same diets, same per-episode column names), so `junction/analyze.py`'s `Rows`
reads this node's npz unchanged.  What is new is that a diet is no longer ONE accumulation
over a mask: it is a row-ORDER over that mask, and the critic is refit along the sequence.

THE ONLINE FITTERS (all exact, all closed form, all through `striatum/task.py::ridge_solve`,
which is `coeruleus/readout.py`'s solve):

  win<N>   a flat sliding window of the last N *windows* of experience.  N is the memory.
  fgt<M>   exponentially weighted least squares with a per-row forgetting factor
           `lam = exp(-1 / (M * R_))`, i.e. an effective memory of M windows.  This is
           exactly what recursive least squares with a forgetting factor converges on at
           every step, computed in closed form instead of by rank-one updates, so it is
           the Bayesian-observer form of a running estimate.
  cum      every row seen so far, no forgetting: the reference that CANNOT adapt.

All three are computed from block Grams.  The sequence is chopped into blocks of
`block_win` windows; each block contributes `X_b^T X_b` (for `win`/`cum`) and
`(X_b * w)^T X_b` with `w_i = lam^(G-1-i)` (for `fgt`).  A flat window is then a difference
of two cumulative snapshots and a forgetting fit is `lam^G A(t-G) + W_b`; both are exact at
block resolution, and the whole pass is matmuls rather than a Python loop over rows.

THE ARMS (`out` family, and `dmg` as the second instance):

  <fam>_lo2mid    phase A = the low tercile, phase B = the middle tercile
  <fam>_hi2mid    phase A = the high tercile, phase B = the middle tercile
  <fam>_mid2mid   phase A and phase B both the middle tercile (two independent
                  permutations of the same rows): the control whose world does not change
  out_lo_mix      the lo and mid windows of `out_lo2mid`, interleaved at random: the same
  out_hi_mix      rows with the sequence removed, so the sequence is the treatment

Within a family the three tercile masks pin the `(etype, j)` cell histogram bit-identically
(`norm/diets.py`), so the two phases of an arm differ ONLY in the outcome, and the phases
are equal length by construction.

Run:
  modal run -m rhm.logit_reading.orbitofrontal.adaptation.task::adapt_ckpt \
      --ckpt /data/v16_s2_L6_m4_distinct/logit_reading/traj_a1_s42/step064000.pt \
      --stim-tag a1 --max-windows 4000 --n-clean 1024 --actor-steps 300 \
      --max-twins 1500 --block-win 10 --tag smoke
  modal run --detach -m rhm.logit_reading.orbitofrontal.adaptation.task::adapt_sweep
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
app = modal.App("rhm-striatum-norm-adapt", image=image)

LEVELS = [1, 2, 3, 4, 5, 6]
DMAX = 12                       # unchanged from striatum/norm: fixes the Gram's row range
T_LO = 8
L_STORE = [1, 2, 3, 4]
A_STORE = [0, 4, 8]             # the event-aligned query and two later ones
A_LIST = [0, 1, 2, 4, 8]
DD_STORE = [0, 1, 4, 5, 8, 9]   # V at a, V_pre at a + 1, for a in A_STORE
LAMS = [1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
PRIM = "post_block7"

# the static reference critics: `norm/`'s banked closed-form fits, refit here as the
# reproduction gate and as the endpoints an online critic should approach
STATIC = ["full", "out_lo", "out_mid", "out_hi", "dmg_lo", "dmg_mid", "dmg_hi"]

# (name, phase A diet, phase B diet, mode)
ARMS = [
    ("out_mid2mid", "out_mid", "out_mid", "switch"),
    ("out_lo2mid", "out_lo", "out_mid", "switch"),
    ("out_hi2mid", "out_hi", "out_mid", "switch"),
    ("out_lo_mix", "out_lo", "out_mid", "interleave"),
    ("out_hi_mix", "out_hi", "out_mid", "interleave"),
    ("dmg_mid2mid", "dmg_mid", "dmg_mid", "switch"),
    ("dmg_lo2mid", "dmg_lo", "dmg_mid", "switch"),
    ("dmg_hi2mid", "dmg_hi", "dmg_mid", "switch"),
]

# (name, kind, memory in windows)
FITTERS = [("win200", "win", 200), ("win800", "win", 800),
           ("fgt200", "fgt", 200), ("fgt800", "fgt", 800),
           ("cum", "cum", 0)]

# checkpoint offsets from the switch, in windows; dense around it, clipped to the phase
BASE_OFF = [25, 50, 100, 150, 200, 300, 400, 600, 800, 1200, 1600, 2400, 3200,
            4800, 6400, 9600, 12800]


def tgt_names():
    return [f"l{l}_d{dd}" for l in L_STORE for dd in DD_STORE]


def ckpt_offsets(P, B):
    """Offsets from the switch, in windows, quantised to the block size `B`.  `-P + B` is
    the earliest fit (one block of experience); `0` is the switch; `+P` is the end."""
    base = [o for o in BASE_OFF if 0 < o < P and o % B == 0]
    offs = sorted({-(P - B)} | {-o for o in base} | {0} | {o for o in base} | {P})
    return [o for o in offs if -(P - B) <= o <= P]


def solve_pertarget(A, C, lam_vec):
    """`ridge_solve` per target with a per-target penalty."""
    B = np.zeros((A.shape[0], C.shape[1]))
    for lm in sorted(set(np.asarray(lam_vec).tolist())):
        sel = np.where(np.asarray(lam_vec) == lm)[0]
        B[:, sel] = ridge_solve(A, np.ascontiguousarray(C[:, sel]), lm)
    return B


@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=4 * 3600, memory=12288,
              cpu=2.0, max_containers=4)
def adapt_ckpt(ckpt: str, stim_tag: str = "a1", n_clean: int = 6144,
               n_clean_val: int = 2048, max_windows: int = 0, seed: int = 11,
               tag: str = "", actor_steps: int = 3000, eval_seed: int = 5150,
               n_val_windows: int = 1500, max_twins: int = 45000,
               caliper: float = 0.6, chunk: int = 512, strat_frac: float = 1.0 / 3.0,
               block_win: int = 25, anchors_keep: str = "tv,fd", arms: str = "",
               order_seed: int = 909):
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

    # ---- the two forward passes (norm's) ---------------------------------------------
    def run(W, Y, want_cache=False, bs=256):
        o = np.zeros((len(LEVELS), len(W), T), np.uint8)
        nll = np.zeros((len(W), T), np.float32)
        Hq = np.zeros((len(W), T), np.float32)
        anc = np.zeros((len(W), 4, d), np.float32)
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
                Xp = inter[PRIM].float()
                for li, l in enumerate(LEVELS):
                    pred = actor_apply(actors[l], Xp)
                    tru = torch.as_tensor(Y[li][sl].astype(np.int64), device=dev)
                    o[li, sl] = (pred == tru).to(torch.uint8).cpu().numpy()
                ap = torch.as_tensor(anc_pos[sl], device=dev)
                ar = torch.arange(x.shape[0], device=dev)[:, None]
                anc[sl] = inter[PRIM][ar, ap].float().cpu().numpy()
                if want_cache:
                    kk = np.where(cache_map[c0:sl.stop] >= 0)[0]
                    if len(kk):
                        kt = torch.as_tensor(kk, device=dev)
                        cache[cache_map[c0 + kk]] = \
                            inter[PRIM][kt][:, rows].half().cpu().numpy()
        return o, nll, Hq, anc, cache

    t0 = time.time()
    o_o, nll_o, H_o, st_o, _ = run(Wo, y_orig)
    print(f"orig pass {time.time() - t0:.1f}s", flush=True)
    t0 = time.time()
    o_e, nll_e, H_e, st_e, cache = run(We, y_edit, want_cache=True)
    print(f"edit pass {time.time() - t0:.1f}s  cache {cache.nbytes / 1e9:.2f} GB", flush=True)

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
    # the same tercile rule applied to the VALIDATION windows, so the fixed reference rows
    # can be read in each world's own currency
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
        st_ = diet_stats[nm]
        print(f"  {nm:16s} n_win {st_['n_windows']:6d}  E[o] {st_['mean_outcome']:.4f}  "
              f"E[dmg] {st_['mean_damage']:.4f}  j {st_['mean_j']:.3f} "
              f"etype {st_['etype_hist']}", flush=True)

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
    # val-row world membership (terciles taken on the validation windows themselves)
    vworld = {}
    for k in ("lo", "mid", "hi"):
        mv = masks[f"vout_{k}"][va_idx]
        vworld[k] = torch.as_tensor(np.repeat(mv[:, None], R_, 1).reshape(-1), device=dev)
    print(f"val rows {len(Xv_np)}  worlds "
          f"{ {k: int(vworld[k].sum()) for k in vworld} }", flush=True)

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

    # ---- the static reference critics (norm's banked closed-form fits) ---------------
    t0 = time.time()
    A_s = {nm: torch.zeros(d + 1, d + 1, dtype=torch.float64, device=dev) for nm in STATIC}
    C_s = {nm: torch.zeros(d + 1, n_tgt, dtype=torch.float64, device=dev) for nm in STATIC}
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
        del X, Y
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
    del A_s, C_s
    torch.cuda.empty_cache()
    print(f"static critics {time.time() - t0:.1f}s  full l1_d0 R2 "
          f"{static_info['full']['val_r2']['l1_d0']:.4f}", flush=True)

    # ---- the arms: a diet is a row ORDER, not one accumulation ------------------------
    arm_list = [a for a in ARMS if not arms or a[0] in arms.split(",")]
    B_ = int(block_win)
    diet_w = {nm: np.where(masks[nm][tr_idx])[0] for nm in
              ["out_lo", "out_mid", "out_hi", "dmg_lo", "dmg_mid", "dmg_hi"]}
    # A phase must be a whole number of blocks, and the two phases of an arm must keep the
    # `(etype, j)` cell histogram `norm/diets.py` pins bit-identically.  So the trim to a
    # multiple of the block size is allocated PER CELL, with the same allocation for every
    # diet, and drawn deterministically inside each cell.
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

    lam_glob = {}                       # per fitter: (n_tgt,) penalties, fixed everywhere
    win_blk = {nm: max(1, m // B_) for nm, kind, m in FITTERS if kind == "win"}
    fgt_lam = {nm: float(np.exp(-1.0 / (m * R_))) for nm, kind, m in FITTERS if kind == "fgt"}
    need_cum = sorted({cb for cb in ck_blk} |
                      {max(cb - nb, 0) for cb in ck_blk for nb in win_blk.values()})

    def run_arm(seq_w, select_lam=False):
        """One pass over the sequence: block Grams, snapshots, then a solve per checkpoint
        per fitter.  Returns (betas (n_fit, n_ck, d+1, n_tgt), diagnostics)."""
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
        nxt = {}                         # ckpt block -> (X, Y) of the NEXT block, fp32
        ck_set = set(ck_blk)
        cum_set = set(need_cum)
        zeroA = np.zeros((d + 1, d + 1)); zeroC = np.zeros((d + 1, n_tgt))
        for k in range(nblk):
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
            del X, Y
        betas = np.zeros((len(FITTERS), len(ck_blk), d + 1, n_tgt), np.float32)
        diag = {k: np.full((len(FITTERS), len(ck_blk), n_tgt), np.nan, np.float32)
                for k in ("val_r2", "delta_val", "delta_val_lo", "delta_val_mid",
                          "delta_val_hi", "delta_next", "delta_next_sem", "n_rows")}
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
                    rw = r.reshape(B_, R_, n_tgt).mean(1)      # one mean per window
                    diag["delta_next"][fi, ci] = rw.mean(0).cpu().numpy()
                    diag["delta_next_sem"][fi, ci] = (
                        rw.std(0, unbiased=True) / np.sqrt(B_)).cpu().numpy()
        del cumA, cumC, fA, fC, snapA, snapC, snapF, nxt
        torch.cuda.empty_cache()
        return betas, diag, lam_out

    t0 = time.time()
    arm_beta, arm_diag, arm_info = {}, {}, {}
    for ai, (anm, pa, pb, mode) in enumerate(arm_list):
        seq, wa, wb = build_seq(pa, pb, mode, ai)
        bt, dg, lo = run_arm(seq, select_lam=(ai == 0))
        arm_beta[anm], arm_diag[anm] = bt, dg
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
              f"{arm_info[anm]['cell_hist_a_eq_b']}  ({time.time() - t0:.0f}s)", flush=True)
    print(f"online fits {time.time() - t0:.1f}s", flush=True)
    del cache
    torch.cuda.empty_cache()

    # ---- anchors: per-episode columns for the STATIC critics, plus the raw states -----
    out_rows = {}
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
            return Xp, X0, lsm_prev

        tpos = tv[elig]
        Xpre_v, X0_v, lsm_v = states_at(We[elig], tpos)
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
            Xpre_c, X0_c, lsm_c = states_at(Wt, tpos[has])
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
            print(f"  state identity {dev_prev:.2e}  V_pre identity {dvp:.2e}  "
                  f"q_prev {dq:.2e}  "
                  f"|ds| {twin['mean_abs_ds']:.4f}  ({time.time() - t0:.1f}s)", flush=True)
        else:
            twin = {"n_eligible": int(len(elig)), "n_pairs": int(len(has)),
                    "note": "too few pairs"}

    # ---- save -------------------------------------------------------------------------
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
    stem = ckpt[:-3]
    sfx = f"_adapt_{stim_tag}" + (f"_{tag}" if tag else "")
    with open(f"{stem}{sfx}.json", "w") as f:
        json.dump(res, f, cls=NumpyEncoder, indent=1)
    save = {}
    for an, rec in out_rows.items():
        for k, vv in rec.items():
            a_ = np.asarray(vv)
            save[f"{an}__{k}"] = a_.astype(np.float32) if a_.dtype == np.float64 else a_
    for k, vv in twin_arr.items():
        a_ = np.asarray(vv)
        save[f"tw__{k}"] = a_.astype(np.float32) if a_.dtype == np.float64 else a_
    for nm in STATIC:
        save[f"beta_static_{nm}"] = beta_s[nm].astype(np.float32)
    for anm in arm_beta:
        save[f"beta_{anm}"] = arm_beta[anm]
        for k2, vv in arm_diag[anm].items():
            save[f"diag_{anm}__{k2}"] = vv
    np.savez_compressed(f"{stem}{sfx}.npz", **save)
    volume.commit()
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    sz = os.path.getsize(f"{stem}{sfx}.npz") / 1e6
    print(f"saved -> {stem}{sfx}.json  npz {sz:.0f} MB  "
          f"({time.time() - t00:.0f}s, peak RSS {rss:.1f} GB)", flush=True)
    return {"ckpt": ckpt, "step": step, "stim_tag": stim_tag, "n_arms": len(arm_list),
            "n_ckpts": len(offs), "phase_windows": int(P_win),
            "actor_l1": actor_tab.get(f"{PRIM}/l1"), "n_pairs": twin.get("n_pairs"),
            "static_Vpre_gate": _gate(out_rows, STATIC)}


def _gate(out_rows, static):
    """The reproduction gate in one line: mean `V_pre` at l = 1, a = 0 on the violation
    rows of the `tv` anchor, which is `norm/README.md` section 1's table."""
    an = "tv" if "tv" in out_rows else (list(out_rows)[0] if out_rows else None)
    if an is None:
        return {}
    rec = out_rows[an]
    m = (rec["etype"] == 0) & (rec["t_v"] > 0)
    if m.sum() < 5:
        return {}
    return {"anchor": an, "n": int(m.sum()),
            **{nm: round(float(rec[f"Vpre_{nm}_l1_a0"][m].mean()), 4) for nm in static}}


def _fit_AC(kind, fnm, cb, snapA, snapC, snapF, win_blk, zeroA, zeroC):
    """(A, C) for one fitter at one checkpoint block, from the snapshots."""
    if kind == "cum":
        return snapA[cb], snapC[cb]
    if kind == "win":
        lo = max(cb - win_blk[fnm], 0)
        if lo == 0:
            return snapA[cb], snapC[cb]
        return snapA[cb] - snapA[lo], snapC[cb] - snapC[lo]
    A, C = snapF[cb][fnm]
    return A, C


# ---------------------------------------------------------------------------
# coordinator: every cell in its own container
# ---------------------------------------------------------------------------

@app.function(volumes={DATA_DIR: volume}, timeout=6 * 3600, memory=2048)
def adapt_sweep(cells: str = "", tag: str = "", max_windows: int = 0,
                max_twins: int = 45000, block_win: int = 25, arms: str = "",
                order_seed: int = 909):
    """cells: `ckpt:stim_tag:anchors`, SEMICOLON separated (the anchor list itself is
    comma separated)."""
    volume.reload()
    D = f"{DATA_DIR}/v16_s2_L6_m4_distinct/logit_reading"
    if not cells:
        cells = ";".join([f"{D}/traj_a1_s42/step064000.pt:a1:tv,fd",
                          f"{D}/traj_a1_s42/step064000.pt:swap65k:tv"])
    args = []
    for c in cells.split(";"):
        p = c.split(":")
        ck, tg, an = p[0], p[1], (p[2] if len(p) > 2 else "tv,fd")
        if not os.path.exists(ck):
            print(f"MISSING {ck}", flush=True)
            continue
        args.append((ck, tg, 6144, 2048, max_windows, 11, tag, 3000, 5150, 1500,
                     max_twins, 0.6, 512, 1.0 / 3.0, block_win, an, arms, order_seed))
    print(f"{len(args)} cells: {[(a[0].split('/')[-1], a[1], a[15]) for a in args]}",
          flush=True)
    outs = list(adapt_ckpt.starmap(args, return_exceptions=True))
    for o in outs:
        print(json.dumps(o, default=str)[:900], flush=True)
    return [str(o)[:900] for o in outs]
