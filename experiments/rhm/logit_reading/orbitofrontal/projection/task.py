"""projection: does the value projection go through the model's BELIEF or its REPRESENTATION?

`norm/` read the outcome-trained critic's three objects -- the pre-event level (the norm),
the revision at the event (the response) and the outcome surprise -- and found the level is
a learned expectation over outcomes that calibrates to the world the critic was fed.  The
reading settled in discussion is that the norm is the world model read through a projection
whose weights the goal's history sets.  What that leaves open is WHICH object the
projection reads.  The critic reads the residual stream.  The model's PUBLIC belief is its
output distribution.  Two parts, both on `norm/`'s machinery and its diets verbatim:

Part 1, the priced twin.  On the same-prefix pairs, `dR = R(violator) - R(twin)` is the
value-side event reading (`norm/` section 3: positive, token-balanced, absent at random
init).  The structural side already measured the model's forecast after the illegal token
against the forecast after its legal twin (parent Part 2c; `altitude/` Q4b's "whither").
Here: fit one linear map from the FORECAST difference (log q, q, or both, at `t_v + 1` and
a few offsets beyond, teacher-forced on each window's own continuation and causal at every
offset) to `dR` on training pairs, and read it on held-out pairs.  Beside it the same map
from the STATE difference, which is the critic itself and so is the ceiling.  If the priced
forecast recovers `dR`, the feeling is the priced belief; if it does not, the norm is a
projection of the representation that the belief does not expose.

Part 2, the public critic.  Refit the critic with the model's output distribution at the
query position as the feature block in place of the residual stream -- same targets, same
diets, same held-out rows, same ridge -- and read `norm/`'s headline objects on it beside
the state critic.  The gap on each object is that object's private part.

Feature blocks (all causal; nothing below ever reads a future position's forecast):

  state       the frozen trunk's `post_block7` residual stream, 256 dims: the critic of record
  logq        `log q(. | w_<=p)`, 16 dims: the model's published belief about the NEXT token
  logq_rich   [log q, q, H(q)], 33 dims: the same belief, reparameterised (the guard against
              reading the gap as an artefact of the link function)
  logq_tok    [log q, onehot(x_p)], 32 dims: the tokens-plus-logits observer at one position
  pub_hist    [log q at p, p-1, p-2, p-3, onehot(x_p), onehot(x_(p-1))], 96 dims: the same
              observer over a short public window
  rand16      a fixed random 16-dim linear projection of the state -- the dimension-matched
              control for `logq`
  pca16       the top-16 principal directions of the state on the `full` diet's own rows --
              the BEST 16-dim linear compression, the ceiling for any 16-dim slice

`rand16` and `pca16` are derived exactly from the state Gram (a linear projection `P` has
`A_P = P A P'`, `C_P = P C`), so they cost no second pass and no second forward.

Also saved: the readout span.  `logits = W_U diag(gamma) (x - mean(x)) / sigma(x) + b`, and
the softmax kills the constant direction in logit space, so the belief exposes EXACTLY the
15-dimensional projection `M x` of the state, up to a per-position positive scale.
`readout_span` is `||P_M beta|| / ||beta||` for each critic column: the fraction of the
critic's own direction that the output layer can express at all.  A random direction gives
sqrt(15 / 256) = 0.242.

Run:
  modal run -m rhm.logit_reading.orbitofrontal.projection.task::proj_ckpt \
      --ckpt /data/v16_s2_L6_m4_distinct/logit_reading/traj_a1_s42/step064000.pt \
      --stim-tag a1 --max-windows 3000 --n-clean 1024 --actor-steps 300 \
      --max-twins 2000 --tag smoke
  modal run --detach -m rhm.logit_reading.orbitofrontal.projection.task::proj_sweep
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
app = modal.App("rhm-striatum-norm-projection", image=image)

LEVELS = [1, 2, 3, 4, 5, 6]
DMAX = 12
DD_LIST = [0, 1, 2, 3, 4, 5, 8, 9]
A_LIST = [0, 1, 2, 4, 8]
A_STORE = [0, 4, 8]
L_STORE = [1, 2, 3, 4]
T_LO = 8
LAMS = [1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
OFFS = [-1, 0, 1, 2, 3, 4, 6, 8]          # twin trace offsets relative to t_v
POFFS = [0, 1, 2, 3, 4]                   # forecast offsets kept raw for the priced map
HIST = 4                                  # public-history depth (positions p .. p-3)

PUB_BLOCKS = ["logq", "logq_rich", "logq_tok", "pub_hist"]
DER_BLOCKS = ["rand16", "pca16"]
ALL_BLOCKS = ["state"] + PUB_BLOCKS + DER_BLOCKS
TRACE_DIETS = {"state": ["full", "clean"]}       # which critics get a twin trace
TRACE_DIETS_DEFAULT = ["full"]
# the query offsets stored per block at the anchors.  `state` keeps the outcome-surprise
# offsets; the rest keep the event only, which is where every contrast in this node lives.
A_STORE_BY_BLOCK = {"state": [0, 4, 8]}


def tgt_names():
    real = [f"l{l}_d{dd}" for l in LEVELS for dd in DD_LIST]
    return real + [f"sh_{nm}" for nm in real]


def pub_dim(block, v):
    return {"logq": v, "logq_rich": 2 * v + 1, "logq_tok": 2 * v,
            "pub_hist": HIST * v + 2 * v}[block]


def _pub_feats_t(lq, tk, block):
    """Public features from `lq` (N, HIST, v) log-softmax at p, p-1, .. and `tk` (N, 2)
    tokens at p, p-1.  Torch, float64, on whatever device they are on."""
    import torch
    v = lq.shape[-1]
    lq0 = lq[:, 0]
    if block == "logq":
        return lq0
    if block == "logq_rich":
        q = torch.exp(lq0)
        H = -(q * lq0).sum(-1, keepdim=True)
        return torch.cat([lq0, q, H], -1)
    if block == "logq_tok":
        oh = torch.nn.functional.one_hot(tk[:, 0].long(), v).to(lq0.dtype)
        return torch.cat([lq0, oh], -1)
    if block == "pub_hist":
        oh0 = torch.nn.functional.one_hot(tk[:, 0].long(), v).to(lq0.dtype)
        oh1 = torch.nn.functional.one_hot(tk[:, 1].long(), v).to(lq0.dtype)
        return torch.cat([lq.reshape(len(lq), -1), oh0, oh1], -1)
    raise ValueError(block)


@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=4 * 3600, memory=12288,
              max_containers=4)
def proj_ckpt(ckpt: str, stim_tag: str = "a1", n_clean: int = 6144,
              n_clean_val: int = 2048, max_windows: int = 0, seed: int = 11,
              tag: str = "", actor_steps: int = 3000, eval_seed: int = 5150,
              n_clean_crit: int = 8192, prim_block: str = "post_block7",
              n_val_windows: int = 1500, do_window_diets: bool = True,
              max_twins: int = 45000, caliper: float = 2.0, chunk: int = 512,
              strat_frac: float = 1.0 / 3.0, do_glitch: bool = False,
              save_caliper: float = 0.35):
    import torch
    volume.reload()
    t00 = time.time()
    dev = "cuda"
    model, rules, rule_w, cfg = load_trajectory_ckpt(ckpt, dev)
    L, s, v = cfg["L"], cfg["s"], cfg["v"]
    T = model.block_size
    step = cfg.get("step")
    d = model.transformer.wte.weight.shape[1]
    print(f"loaded {ckpt} step {step} d={d} T={T} v={v}", flush=True)

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
    print(f"stimuli {stim_tag}: n={n} etype={np.bincount(et, minlength=3).tolist()}",
          flush=True)

    # ---- splits over windows (norm/junction's recipe verbatim) -----------------------
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
            Sc[c0:c0 + 256] = inter[prim].float().cpu().numpy()
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

    rows = np.arange(T_LO, T - 1 - DMAX + 1)          # contiguous; rows[0] - HIST + 1 >= 0
    R_ = len(rows)
    assert rows[0] - (HIST - 1) >= 0
    cache_idx = np.concatenate([tr_idx, va_idx])
    cache_map = -np.ones(n, np.int64)
    cache_map[cache_idx] = np.arange(len(cache_idx))

    # ---- the two forward passes -------------------------------------------------------
    def run(W, Y, want_cache=False, bs=256):
        o = np.zeros((len(LEVELS), len(W), T), np.uint8)
        nll = np.zeros((len(W), T), np.float32)
        Hq = np.zeros((len(W), T), np.float32)
        anc = np.zeros((len(W), 4, d), np.float32)
        anc_lq = np.zeros((len(W), 4, HIST, v), np.float32)
        anc_tk = np.zeros((len(W), 4, 2), np.int16)
        cache = (np.zeros((len(cache_idx), R_, d), np.float16) if want_cache else None)
        qcache = (np.zeros((len(cache_idx), T, v), np.float32) if want_cache else None)
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
                hp = torch.clamp(ap[:, :, None] - torch.arange(HIST, device=dev)[None, None],
                                 0, T - 1)                                  # (b, 4, HIST)
                anc_lq[sl] = lsm[ar[:, :, None].expand_as(hp), hp].cpu().numpy()
                tp = torch.clamp(ap[:, :, None] - torch.arange(2, device=dev)[None, None],
                                 0, T - 1)
                anc_tk[sl] = x[ar[:, :, None].expand_as(tp), tp].cpu().numpy().astype(np.int16)
                if want_cache:
                    kk = np.where(cache_map[c0:sl.stop] >= 0)[0]
                    if len(kk):
                        kt = torch.as_tensor(kk, device=dev)
                        cache[cache_map[c0 + kk]] = \
                            inter[prim][kt][:, rows].half().cpu().numpy()
                        qcache[cache_map[c0 + kk]] = lsm[kt].cpu().numpy()
        return o, nll, Hq, anc, anc_lq, anc_tk, cache, qcache

    t0 = time.time()
    o_o, nll_o, H_o, st_o, lq_o, tk_o, _, _ = run(Wo, y_orig)
    print(f"orig pass {time.time() - t0:.1f}s", flush=True)
    t0 = time.time()
    o_e, nll_e, H_e, st_e, lq_e, tk_e, cache, qcache = run(We, y_edit, want_cache=True)
    print(f"edit pass {time.time() - t0:.1f}s  state cache {cache.nbytes / 1e9:.2f} GB "
          f"logq cache {qcache.nbytes / 1e9:.2f} GB", flush=True)

    # ---- the diets (norm/diets.py verbatim, same seeds -> identical masks) -------------
    c_depth = cons_depth(et, jj)
    dmg_full = (o_o.astype(np.int16) - o_e.astype(np.int16)) > 0
    dmg_w = dmg_full[:4][:, :, rows]
    dmg_win = dmg_w.mean((0, 2))
    out_win = o_e[:4][:, :, rows].astype(np.float32).mean((0, 2))
    out_orig_win = o_o[:4][:, :, rows].astype(np.float32).mean((0, 2))

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

    med_global = float(np.median(dmg_win[is_tr & (et < 2)])) if (et < 2).any() else 0.0
    o_row_tr = o_e[:4][:, tr_idx][:, :, rows].astype(np.float32).mean(0)
    d_row_tr = dmg_w.mean(0)[tr_idx]
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
               "mean_outcome": float(o_row_tr[m].mean()),
               "mean_damage": float(d_row_tr[m].mean()),
               "mean_outcome_win": float(out_win[wsel].mean()),
               "mean_damage_win": float(dmg_win[wsel].mean()),
               "mean_outcome_orig_win": float(out_orig_win[wsel].mean()),
               "edit_share": float((et[wsel] < 2).mean()),
               "mean_j": float(jj[wsel].mean()),
               "mean_c_depth": float(c_depth[ed].mean()) if len(ed) else float("nan"),
               "p_illegal_among_edits": float(sw.mean()) if len(ed) else float("nan"),
               "j_hist": np.bincount(jj[wsel], minlength=6).tolist(),
               "etype_hist": np.bincount(et[wsel], minlength=3).tolist(),
               "cell_hist": {int(c): int(cells[c]) for c in np.where(cells)[0]}}
        diet_stats[nm] = st_
    for nm in diet_names:
        print(f"  {nm:16s} n_win {diet_stats[nm]['n_windows']:6d} rows "
              f"{diet_stats[nm]['n_rows']:7d}  E[o] {diet_stats[nm]['mean_outcome']:.4f}  "
              f"E[dmg] {diet_stats[nm]['mean_damage']:.4f}", flush=True)

    # ---- Grams: one per (block, diet) --------------------------------------------------
    names = tgt_names()
    n_tgt = len(names)
    n_real = n_tgt // 2
    dims = {"state": d}
    for b in PUB_BLOCKS:
        dims[b] = pub_dim(b, v)
    A = {b: {nm: torch.zeros(dims[b] + 1, dims[b] + 1, dtype=torch.float64, device=dev)
             for nm in diet_names} for b in dims}
    C = {b: {nm: torch.zeros(dims[b] + 1, n_tgt, dtype=torch.float64, device=dev)
             for nm in diet_names} for b in dims}
    shuf = np.random.default_rng(seed + 3).permutation(len(tr_idx))
    hoff = torch.arange(HIST, device=dev)
    toff = torch.arange(2, device=dev)
    rows_t = torch.as_tensor(rows, device=dev)

    def block_X(qc, toks, sl_n):
        """(rows, dim) features for every public block, for one chunk of windows.
        `qc` (b, T, v) log-softmax, `toks` (b, T) window tokens."""
        b_ = qc.shape[0]
        hp = torch.clamp(rows_t[None, :, None] - hoff[None, None, :], 0, T - 1)   # (1,R,H)
        hp = hp.expand(b_, -1, -1)
        ar = torch.arange(b_, device=dev)[:, None, None].expand_as(hp)
        lq = qc[ar, hp]                                          # (b, R, HIST, v)
        tp = torch.clamp(rows_t[None, :, None] - toff[None, None, :], 0, T - 1).expand(b_, -1, -1)
        tk = toks[torch.arange(b_, device=dev)[:, None, None].expand_as(tp), tp]  # (b, R, 2)
        lq = lq.reshape(-1, HIST, v).double()
        tk = tk.reshape(-1, 2)
        return {bb: _pub_feats_t(lq, tk, bb) for bb in PUB_BLOCKS}

    t0 = time.time()
    for c0 in range(0, len(tr_idx), chunk):
        sl = slice(c0, min(c0 + chunk, len(tr_idx)))
        wsl = tr_idx[sl]
        Xst = torch.as_tensor(cache[sl].reshape(-1, d), device=dev).double()
        Xs = {"state": Xst}
        Xs.update(block_X(torch.as_tensor(qcache[sl], device=dev),
                          torch.as_tensor(We[wsl][:, :T].astype(np.int64), device=dev),
                          len(wsl)))
        for b in Xs:
            Xs[b] = torch.cat([Xs[b], torch.ones(len(Xs[b]), 1, dtype=torch.float64,
                                                 device=dev)], 1)
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
            Yb = Y[sel]
            for b in Xs:
                Xb = Xs[b][sel]
                A[b][nm] += Xb.T @ Xb
                C[b][nm] += Xb.T @ Yb
        del Xs, Y
    print(f"grams {time.time() - t0:.1f}s", flush=True)

    # ---- validation features ------------------------------------------------------------
    Xv = {"state": torch.as_tensor(cache[len(tr_idx):].reshape(-1, d), device=dev).double()}
    Xv.update(block_X(torch.as_tensor(qcache[len(tr_idx):], device=dev),
                      torch.as_tensor(We[va_idx][:, :T].astype(np.int64), device=dev),
                      len(va_idx)))
    for b in list(Xv):
        Xv[b] = torch.cat([Xv[b], torch.ones(len(Xv[b]), 1, dtype=torch.float64,
                                             device=dev)], 1)
    yv = []
    for li in range(len(LEVELS)):
        for dd in DD_LIST:
            yv.append(torch.as_tensor(o_e[li][np.ix_(va_idx, rows + dd)],
                                      dtype=torch.float64, device=dev))
    Yv = torch.stack(yv, -1).reshape(-1, n_real)
    Yv = torch.cat([Yv, Yv], 1)
    yvar = Yv.var(0)

    # ---- the two derived 16-dim projections of the state --------------------------------
    Afull = A["state"]["full"] if "full" in A["state"] else A["state"][diet_names[0]]
    nA = float(Afull[-1, -1].item())
    mu = (Afull[:d, -1] / max(nA, 1.0)).cpu().numpy()
    Cov = (Afull[:d, :d].cpu().numpy() / max(nA, 1.0)) - np.outer(mu, mu)
    w_, Vc = np.linalg.eigh(Cov)
    Ppca = Vc[:, ::-1][:, :16].T.copy()                       # (16, d)
    Prnd = np.random.default_rng(seed + 31).normal(size=(16, d)) / np.sqrt(d)
    proj = {"pca16": Ppca, "rand16": Prnd}
    pca_var = float(w_[::-1][:16].sum() / max(w_.sum(), 1e-12))
    print(f"pca16 keeps {pca_var:.4f} of the state's variance on `full`'s rows", flush=True)

    for b, Pm in proj.items():
        dims[b] = 16
        Pt = torch.as_tensor(np.concatenate(
            [np.concatenate([Pm, np.zeros((16, 1))], 1),
             np.concatenate([np.zeros((1, d)), np.ones((1, 1))], 1)], 0), device=dev)
        A[b] = {nm: Pt @ A["state"][nm] @ Pt.T for nm in diet_names}
        C[b] = {nm: Pt @ C["state"][nm] for nm in diet_names}
        Xv[b] = Xv["state"] @ Pt.T

    # ---- solve every (block, diet) -------------------------------------------------------
    beta, val_r2 = {}, {}
    t0 = time.time()
    for b in ALL_BLOCKS:
        beta[b], val_r2[b] = {}, {}
        for nm in diet_names:
            An, Cn = A[b][nm].cpu().numpy(), C[b][nm].cpu().numpy()
            best = torch.full((n_tgt,), -1e18, dtype=torch.float64, device=dev)
            Bb = torch.zeros(dims[b] + 1, n_tgt, dtype=torch.float64, device=dev)
            lam_pick = np.zeros(n_tgt)
            for lm in LAMS:
                B = torch.as_tensor(ridge_solve(An, Cn, lm), device=dev)
                r2 = 1.0 - ((Xv[b] @ B - Yv) ** 2).mean(0) / torch.clamp(yvar, min=1e-12)
                up = r2 > best
                best = torch.where(up, r2, best)
                Bb[:, up] = B[:, up]
                lam_pick[up.cpu().numpy()] = lm
            beta[b][nm] = Bb.cpu().numpy()
            val_r2[b][nm] = {names[i]: float(best[i]) for i in range(n_tgt)}
            val_r2[b][nm]["_lam_l1_d0"] = float(lam_pick[names.index("l1_d0")])
            val_r2[b][nm]["_train_mean_l1_d0"] = float(
                (C[b][nm][-1, names.index("l1_d0")] /
                 torch.clamp(A[b][nm][-1, -1], min=1.0)).item())
    del A, C
    torch.cuda.empty_cache()
    print(f"solves {time.time() - t0:.1f}s  l1_d0 R2  " +
          "  ".join(f"{b}:{val_r2[b].get('full', val_r2[b][diet_names[0]])['l1_d0']:.4f}"
                    for b in ALL_BLOCKS), flush=True)

    # ---- the clean-only critic, every block ----------------------------------------------
    t0 = time.time()
    real_names = names[:n_real]
    wins_cc, phase_cc, _, fs_cc = windows_with_parse(rules, rule_w, n_clean_crit, T + 1,
                                                     eval_seed + 1000)
    y_cc = clean_answers(fs_cc, phase_cc, T + 1, L, s, 3)[:, :, :T]
    n_cc_va = min(2000, n_clean_crit // 5)
    n_cc_tr = n_clean_crit - n_cc_va
    Acc = {b: torch.zeros(dims[b] + 1, dims[b] + 1, dtype=torch.float64, device=dev)
           for b in ["state"] + PUB_BLOCKS}
    Ccc = {b: torch.zeros(dims[b] + 1, n_real, dtype=torch.float64, device=dev)
           for b in ["state"] + PUB_BLOCKS}
    o_cc = np.zeros((len(LEVELS), n_clean_crit, T), np.uint8)
    Xcc_va = {b: [] for b in ["state"] + PUB_BLOCKS}
    with torch.no_grad():
        for c0 in range(0, n_clean_crit, 256):
            sl = slice(c0, min(c0 + 256, n_clean_crit))
            x = torch.as_tensor(wins_cc[sl, :T], device=dev)
            lg, _, inter = model(x, return_intermediates=True)
            lsm = torch.log_softmax(lg.float(), -1)
            Xp = inter[prim].float()
            for li, l in enumerate(LEVELS):
                pred = actor_apply(actors[l], Xp)
                tru = torch.as_tensor(y_cc[li][sl].astype(np.int64), device=dev)
                o_cc[li, sl] = (pred == tru).to(torch.uint8).cpu().numpy()
            ii = np.arange(sl.start, sl.stop)
            fb = {"state": inter[prim][:, rows].double().reshape(-1, d)}
            fb.update(block_X(lsm, x, len(x)))
            for b in fb:
                fb[b] = torch.cat([fb[b], torch.ones(len(fb[b]), 1, dtype=torch.float64,
                                                     device=dev)], 1)
            kk = np.where(ii < n_cc_tr)[0]
            if len(kk):
                ys = [torch.as_tensor(o_cc[li][np.ix_(ii[kk], rows + dd)],
                                      dtype=torch.float64, device=dev)
                      for li in range(len(LEVELS)) for dd in DD_LIST]
                Yb = torch.stack(ys, -1).reshape(-1, n_real)
                rmask = torch.as_tensor(
                    (np.repeat(ii[:, None] < n_cc_tr, R_, 1)).reshape(-1), device=dev)
                for b in fb:
                    Xb = fb[b][rmask]
                    Acc[b] += Xb.T @ Xb
                    Ccc[b] += Xb.T @ Yb
            vk = np.where(ii >= n_cc_tr)[0]
            if len(vk):
                rmask = torch.as_tensor(
                    (np.repeat(ii[:, None] >= n_cc_tr, R_, 1)).reshape(-1), device=dev)
                for b in fb:
                    Xcc_va[b].append(fb[b][rmask].cpu())
            del fb
    Yvc = torch.stack([torch.as_tensor(
        o_cc[li][np.ix_(np.arange(n_cc_tr, n_clean_crit), rows + dd)],
        dtype=torch.float64, device=dev) for li in range(len(LEVELS)) for dd in DD_LIST],
        -1).reshape(-1, n_real)
    clean_val_r2 = {}
    for b in ["state"] + PUB_BLOCKS:
        Xvc = torch.cat(Xcc_va[b], 0).to(dev)
        Accn, Cccn = Acc[b].cpu().numpy(), Ccc[b].cpu().numpy()
        bestc = torch.full((n_real,), -1e18, dtype=torch.float64, device=dev)
        Bc = torch.zeros(dims[b] + 1, n_real, dtype=torch.float64, device=dev)
        for lm in LAMS:
            B = torch.as_tensor(ridge_solve(Accn, Cccn, lm), device=dev)
            r2 = 1.0 - ((Xvc @ B - Yvc) ** 2).mean(0) / torch.clamp(Yvc.var(0), min=1e-12)
            up = r2 > bestc
            bestc = torch.where(up, r2, bestc)
            Bc[:, up] = B[:, up]
        beta[b]["clean"] = Bc.cpu().numpy()
        clean_val_r2[b] = {real_names[i]: float(bestc[i]) for i in range(n_real)}
        del Xvc
    for b, Pm in proj.items():
        Pt = np.concatenate([np.concatenate([Pm, np.zeros((16, 1))], 1),
                             np.concatenate([np.zeros((1, d)), np.ones((1, 1))], 1)], 0)
        Ast = Acc["state"].cpu().numpy()
        Cst = Ccc["state"].cpu().numpy()
        Ap, Cp = Pt @ Ast @ Pt.T, Pt @ Cst
        Xvc = (torch.cat(Xcc_va["state"], 0).to(dev)
               @ torch.as_tensor(Pt, device=dev).T)
        bestc = torch.full((n_real,), -1e18, dtype=torch.float64, device=dev)
        Bc = torch.zeros(17, n_real, dtype=torch.float64, device=dev)
        for lm in LAMS:
            B = torch.as_tensor(ridge_solve(Ap, Cp, lm), device=dev)
            r2 = 1.0 - ((Xvc @ B - Yvc) ** 2).mean(0) / torch.clamp(Yvc.var(0), min=1e-12)
            up = r2 > bestc
            bestc = torch.where(up, r2, bestc)
            Bc[:, up] = B[:, up]
        beta[b]["clean"] = Bc.cpu().numpy()
        clean_val_r2[b] = {real_names[i]: float(bestc[i]) for i in range(n_real)}
        del Xvc
    clean_mean_o = float(o_cc[:4][:, :, rows].astype(np.float32).mean())
    del Acc, Ccc, Xcc_va, Yvc, Xv, Yv, cache, qcache
    torch.cuda.empty_cache()
    print(f"clean-only critics {time.time() - t0:.1f}s  state l1_d0 R2 "
          f"{clean_val_r2['state']['l1_d0']:.4f}  logq {clean_val_r2['logq']['l1_d0']:.4f}"
          f"  E[o] {clean_mean_o:.4f}", flush=True)

    # ---- the readout span: what fraction of each critic direction the logits express ----
    Wu = model.lm_head.weight.detach().float().cpu().numpy()             # (v, d)
    gam = model.transformer.ln_f.weight.detach().float().cpu().numpy()   # (d,)
    M = Wu * gam[None, :]
    M = M - M.mean(0, keepdims=True)              # softmax kills the constant logit
    M = M - M.mean(1, keepdims=True)              # LayerNorm removes the state's mean
    Um, Sm, Vtm = np.linalg.svd(M, full_matrices=False)
    rk = int((Sm > Sm.max() * 1e-5).sum())
    Vb = Vtm[:rk]
    span = {}
    for nm in list(diet_names) + ["clean"]:
        B = beta["state"][nm]
        for l in L_STORE:
            for dd in (0, 1):
                j_ = (names.index(f"l{l}_d{dd}") if nm != "clean"
                      else real_names.index(f"l{l}_d{dd}"))
                bvec = B[:-1, j_]
                span[f"{nm}/l{l}_d{dd}"] = float(
                    np.linalg.norm(Vb @ bvec) / max(np.linalg.norm(bvec), 1e-12))
    span["_rank"] = rk
    span["_random_direction"] = float(np.sqrt(rk / d))
    print(f"readout span rank {rk}/{d}, random floor {span['_random_direction']:.3f}; "
          f"full l1_d0 {span.get('full/l1_d0', float('nan')):.3f}", flush=True)

    # ---- anchors, revisions, labels ------------------------------------------------------
    def anc_feats(b, st_, lq_, tk_, w, ipos):
        if b == "state":
            return st_[w, ipos].astype(np.float64)
        if b in proj:
            return st_[w, ipos].astype(np.float64) @ proj[b].T
        import torch as _t
        lq = _t.as_tensor(lq_[w, ipos], dtype=_t.float64)
        tk = _t.as_tensor(tk_[w, ipos].astype(np.int64))
        return _pub_feats_t(lq, tk, b).numpy()

    def lin(b, nm, name, X):
        B = beta[b][nm]
        j_ = names.index(name) if nm != "clean" else real_names.index(name)
        return X @ B[:-1, j_] + B[-1, j_]

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
        rec["prefix_dev"] = np.abs(st_e[w, ib] - st_o[w, ib]).max(1)
        for b in ALL_BLOCKS:
            Xa = anc_feats(b, st_e, lq_e, tk_e, w, ia)
            Xb_ = anc_feats(b, st_e, lq_e, tk_e, w, ib)
            Xao = anc_feats(b, st_o, lq_o, tk_o, w, ia)
            Xbo = anc_feats(b, st_o, lq_o, tk_o, w, ib)
            for nm in all_critics:
                for l in L_STORE:
                    for a in A_STORE_BY_BLOCK.get(b, [0]):
                        n0, n1 = f"l{l}_d{a}", f"l{l}_d{a + 1}"
                        p = f"{b}|{nm}"
                        rec[f"R_{p}_l{l}_a{a}"] = lin(b, nm, n0, Xa) - lin(b, nm, n1, Xb_)
                        rec[f"Ro_{p}_l{l}_a{a}"] = lin(b, nm, n0, Xao) - lin(b, nm, n1, Xbo)
                        rec[f"V_{p}_l{l}_a{a}"] = lin(b, nm, n0, Xa)
                        rec[f"Vpre_{p}_l{l}_a{a}"] = lin(b, nm, n1, Xb_)
                        if nm != "clean" and b == "state":
                            rec[f"Rsh_{p}_l{l}_a{a}"] = (lin(b, nm, f"sh_{n0}", Xa)
                                                         - lin(b, nm, f"sh_{n1}", Xb_))
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
        out_rows[an] = rec

    # ---- the same-prefix twins, both worlds ------------------------------------------------
    twin, glitch, twin_arr, glitch_arr = {}, {}, {}, {}
    if pL is not None and max_twins > 0:
        t0 = time.time()
        tcols = []
        for b in ALL_BLOCKS:
            for nm in TRACE_DIETS.get(b, TRACE_DIETS_DEFAULT):
                if nm not in all_critics:
                    continue
                for l in L_STORE:
                    for dd in (0, 1):
                        tcols.append((b, nm, l, dd))
        Bst, Bpub = {}, {}
        for b in ALL_BLOCKS:
            cidx = [i for i, c in enumerate(tcols) if c[0] == b]
            if not cidx:
                continue
            Bm = np.stack([np.concatenate(
                [beta[b][nm][:-1, (names.index(f"l{l}_d{dd}") if nm != "clean"
                                   else real_names.index(f"l{l}_d{dd}"))],
                 [beta[b][nm][-1, (names.index(f"l{l}_d{dd}") if nm != "clean"
                                   else real_names.index(f"l{l}_d{dd}"))]]])
                for (_, nm, l, dd) in [tcols[i] for i in cidx]], 1)      # (dim+1, c)
            if b == "state":
                Bst[b] = (cidx, Bm)
            elif b in proj:
                Pm = proj[b]
                Bst[b] = (cidx, np.concatenate([Pm.T @ Bm[:-1], Bm[-1:]], 0))
            else:
                Bpub[b] = (cidx, Bm)
        offs_t = torch.as_tensor(OFFS, device=dev)
        poffs_t = torch.as_tensor(POFFS, device=dev)
        hoff_t = torch.arange(HIST, device=dev)
        toff_t = torch.arange(2, device=dev)
        st_cols = {b: (ci, torch.as_tensor(Bm, device=dev, dtype=torch.float32))
                   for b, (ci, Bm) in Bst.items()}
        pb_cols = {b: (ci, torch.as_tensor(Bm, device=dev, dtype=torch.float32))
                   for b, (ci, Bm) in Bpub.items()}

        def trace(W, tpos, bs=256, want_lsm=False):
            """V at t + OFFS for every column, the raw forecast at t + POFFS, and the raw
            state at t.  Everything is read from a single causal forward pass over the
            window, so a column at offset `o` uses only positions <= t + o."""
            Vt = np.zeros((len(W), len(OFFS), len(tcols)), np.float32)
            Fq = np.zeros((len(W), len(POFFS), v), np.float32)
            X0 = np.zeros((len(W), d), np.float32)
            lsm_prev = np.zeros((len(W), v), np.float32) if want_lsm else None
            with torch.no_grad():
                for c0 in range(0, len(W), bs):
                    sl = slice(c0, min(c0 + bs, len(W)))
                    x = torch.as_tensor(W[sl, :T], device=dev)
                    lg, _, inter = model(x, return_intermediates=True)
                    lsm = torch.log_softmax(lg.float(), -1)
                    tt = torch.as_tensor(tpos[sl], device=dev)
                    ar = torch.arange(x.shape[0], device=dev)
                    if want_lsm:
                        lsm_prev[sl] = lsm[ar, tt - 1].cpu().numpy()
                    p = torch.clamp(tt[:, None] + offs_t[None, :], 0, T - 1)   # (b, O)
                    Xs = inter[prim][ar[:, None], p].float()                   # (b, O, d)
                    X0[sl] = inter[prim][ar, tt].float().cpu().numpy()
                    Fq[sl] = lsm[ar[:, None], torch.clamp(tt[:, None] + poffs_t[None, :],
                                                          0, T - 1)].cpu().numpy()
                    for b, (ci, Bm) in st_cols.items():
                        Vt[sl][:, :, ci] = (Xs @ Bm[:-1] + Bm[-1]).cpu().numpy()
                    if pb_cols:
                        hp = torch.clamp(p[:, :, None] - hoff_t[None, None, :], 0, T - 1)
                        arh = ar[:, None, None].expand_as(hp)
                        lq = lsm[arh, hp].double()                      # (b, O, HIST, v)
                        tp = torch.clamp(p[:, :, None] - toff_t[None, None, :], 0, T - 1)
                        tk = x[ar[:, None, None].expand_as(tp), tp]
                        lqf = lq.reshape(-1, HIST, v)
                        tkf = tk.reshape(-1, 2)
                        for b, (ci, Bm) in pb_cols.items():
                            F = _pub_feats_t(lqf, tkf, b).float()
                            Vt[sl][:, :, ci] = (F @ Bm[:-1] + Bm[-1]).reshape(
                                len(x), len(OFFS), -1).cpu().numpy()
            return Vt, Fq, X0, lsm_prev

        elig = np.where((et == 0) & (tv >= T_LO + 1) & (tv <= T - 1))[0]
        if len(elig) > max_twins:
            elig = np.sort(np.random.default_rng(seed + 21).choice(elig, max_twins,
                                                                   replace=False))
        tpos = tv[elig]
        V_v, F_v, X_v0, lsm_v = trace(We[elig], tpos, want_lsm=True)
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
            V_c, F_c, X_c0, lsm_c = trace(Wt, tpos[has], want_lsm=True)
            dev_prev = float(np.abs(V_c[:, 0] - V_v[has][:, 0]).max())
            dq = float(np.abs(lsm_c - lsm_v[has]).max())
            s_c = -lsm_c[np.arange(len(has)), x_c[has]]
            # the saved subset: any caliper at or below `save_caliper` is still available
            kp = np.where(np.abs(s_v[has] - s_c) <= save_caliper)[0]
            twin = {"n_eligible": int(len(elig)), "n_pairs": int(len(has)),
                    "caliper": float(caliper), "save_caliper": float(save_caliper),
                    "n_saved": int(len(kp)),
                    "check_Vpre_identical_max_abs": dev_prev,
                    "check_qprev_identical_max_abs": dq,
                    "mean_abs_ds": float(np.abs(s_v[has] - s_c).mean()),
                    "coverage_by_kstar": {int(k): [int((ks[elig] == k).sum()),
                                                   int((ks[elig[has]] == k).sum())]
                                          for k in np.unique(ks[elig])},
                    "cols": [[b, nm, int(l), int(dd)] for b, nm, l, dd in tcols],
                    "offs": OFFS, "poffs": POFFS}
            print(f"  Vpre identity {dev_prev:.2e}  q_prev identity {dq:.2e}  "
                  f"|ds| {twin['mean_abs_ds']:.4f}  saved {len(kp)}  "
                  f"({time.time() - t0:.1f}s)", flush=True)
            hh = has[kp]
            twin_arr = {"w": elig[hh], "t_v": tpos[hh], "k_star": ks[elig[hh]],
                        "j": jj[elig[hh]], "e": ee[elig[hh]], "split": split[elig[hh]],
                        "x_v": x_v[hh], "x_c": x_c[hh], "s_v": s_v[hh], "s_c": s_c[kp],
                        "V_viol": V_v[hh], "V_twin": V_c[kp],
                        "F_viol": F_v[hh], "F_twin": F_c[kp],
                        "dX": (X_v0[hh] - X_c0[kp]).astype(np.float32)}
        else:
            twin = {"n_eligible": int(len(elig)), "n_pairs": int(len(has)),
                    "note": "too few pairs"}

        # ---- the glitch world (norm section 5's construction) -------------------------
        plo_path = f"{ddir}/norm_pLorig_{stim_tag}.npz"
        if do_glitch and os.path.exists(plo_path) and twin_arr:
            t0 = time.time()
            PLO = np.load(plo_path)
            assert (PLO["t_v"] == S["t_v"]).all(), "norm_pLorig built on other stimuli"
            legal_o = PLO["pLo_tv"][idx][elig] > 0
            ar = np.arange(len(elig))
            V_o, F_o, X_o0, lsm_o = trace(Wo[elig], tpos, want_lsm=True)
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
            print(f"glitch: illegal available {ok_g.mean():.3f}  pairs {len(hg)}", flush=True)
            if len(hg) >= 50:
                Wg = Wo[elig[hg]].copy()
                Wg[np.arange(len(hg)), tpos[hg]] = x_g[hg]
                Wgc = Wo[elig[hg]].copy()
                Wgc[np.arange(len(hg)), tpos[hg]] = x_gc[hg]
                V_g, F_g, X_g0, lsm_g = trace(Wg, tpos[hg], want_lsm=True)
                V_gc, F_gc, X_gc0, _ = trace(Wgc, tpos[hg])
                dev_g = float(np.abs(V_g[:, 0] - V_gc[:, 0]).max())
                dev_o = float(np.abs(V_g[:, 0] - V_o[hg][:, 0]).max())
                dqg = float(np.abs(lsm_g - lsm_o[hg]).max())
                kg = np.where(np.abs(s_g[hg] - s_o[hg, x_gc[hg]]) <= save_caliper)[0]
                glitch = {"n_pairs": int(len(hg)), "n_saved": int(len(kg)),
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
                          "p_glitch_equals_violator": float((x_g[hg] == x_v[hg]).mean()),
                          "cols": [[b, nm, int(l), int(dd)] for b, nm, l, dd in tcols],
                          "offs": OFFS, "poffs": POFFS}
                print(f"  glitch Vpre identity {glitch['check_Vpre_identical_max_abs']:.2e}"
                      f"  |ds| in-pair {glitch['mean_abs_ds_within_pair']:.4f}"
                      f"  across worlds {glitch['mean_abs_ds_across_worlds']:.4f}"
                      f"  saved {len(kg)}  ({time.time() - t0:.1f}s)", flush=True)
                gg = hg[kg]
                glitch_arr = {"w": elig[gg], "t_v": tpos[gg], "k_star": ks[elig[gg]],
                              "j": jj[elig[gg]], "e": ee[elig[gg]], "split": split[elig[gg]],
                              "x_v": x_g[gg], "x_c": x_gc[gg], "x_orig": x_or[gg],
                              "s_v": s_g[gg], "s_c": s_o[gg, x_gc[gg]], "s_edit": s_v[gg],
                              "V_viol": V_g[kg], "V_twin": V_gc[kg],
                              "F_viol": F_g[kg], "F_twin": F_gc[kg],
                              "dX": (X_g0[kg] - X_gc0[kg]).astype(np.float32)}
            else:
                glitch = {"n_pairs": int(len(hg)), "note": "too few pairs"}
        elif do_glitch:
            glitch = {"note": f"no {plo_path}"}

    # ---- save ---------------------------------------------------------------------------
    res = {"ckpt": ckpt, "step": step, "stim_tag": stim_tag, "n": n, "primary_block": prim,
           "actor_clean_acc": actor_tab, "blocks": ALL_BLOCKS,
           "block_dims": {b: int(dims[b]) for b in ALL_BLOCKS},
           "diet_spec": spec, "diet_stats": diet_stats, "diet_meta": wmeta,
           "median_damage_global": med_global, "twin": twin, "glitch": glitch,
           "readout_span": span, "pca16_var_explained": pca_var,
           "critic_val_r2": {b: {nm: {k: val_r2[b][nm][k] for k in
                                      [f"l{l}_d{dd}" for l in LEVELS for dd in (0, 1)]
                                      + [f"sh_l{l}_d0" for l in LEVELS]
                                      + ["_lam_l1_d0", "_train_mean_l1_d0"]}
                                 for nm in diet_names} for b in ALL_BLOCKS},
           "clean_critic_val_r2": {b: {k: clean_val_r2[b][k] for k in
                                       [f"l{l}_d{dd}" for l in LEVELS for dd in (0, 1)]}
                                   for b in ALL_BLOCKS},
           "clean_critic_mean_outcome": clean_mean_o,
           "config": {"levels": LEVELS, "dmax": DMAX, "dd_list": DD_LIST, "a_list": A_LIST,
                      "a_store": A_STORE, "l_store": L_STORE, "t_lo": T_LO,
                      "n_clean": n_clean, "seed": seed, "eval_seed": eval_seed,
                      "split": [0.6, 0.15, 0.25], "n_clean_crit": n_clean_crit,
                      "n_val_windows": int(len(va_idx)), "offs": OFFS, "poffs": POFFS,
                      "hist": HIST, "caliper": caliper, "save_caliper": save_caliper,
                      "max_twins": max_twins, "strat_frac": strat_frac,
                      "rows": [int(rows[0]), int(rows[-1])]}}
    stem = ckpt[:-3]
    sfx = f"_proj_{stim_tag}" + (f"_{tag}" if tag else "")
    with open(f"{stem}{sfx}.json", "w") as f:
        json.dump(res, f, cls=NumpyEncoder, indent=1)
    for an, rec in out_rows.items():
        tem = np.asarray(rec["split"]) == 2
        save = {}
        for k, vv in rec.items():
            a_ = np.asarray(vv)[tem]
            save[f"{an}__{k}"] = a_.astype(np.float32) if a_.dtype == np.float64 else a_
        np.savez_compressed(f"{stem}{sfx}_rows_{an}.npz", **save)
    sv = {"basis__Vb": Vb.astype(np.float32)}
    for k, vv in twin_arr.items():
        a_ = np.asarray(vv)
        sv[f"tw__{k}"] = a_.astype(np.float32) if a_.dtype == np.float64 else a_
    for k, vv in glitch_arr.items():
        a_ = np.asarray(vv)
        sv[f"gl__{k}"] = a_.astype(np.float32) if a_.dtype == np.float64 else a_
    np.savez_compressed(f"{stem}{sfx}_twins.npz", **sv)
    volume.commit()
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    print(f"saved -> {stem}{sfx}.json (+ _rows_<anchor>.npz, _twins.npz)  "
          f"({time.time() - t00:.0f}s, peak RSS {rss:.1f} GB)", flush=True)
    return {"ckpt": ckpt, "step": step, "stim_tag": stim_tag,
            "actor_l1": actor_tab.get(f"{prim}/l1"),
            "n_pairs": twin.get("n_saved"), "n_glitch_pairs": glitch.get("n_saved"),
            "R2_l1_d0": {b: round(val_r2[b].get("full", val_r2[b][diet_names[0]])["l1_d0"], 4)
                         for b in ALL_BLOCKS},
            "span_full_l1_d0": round(span.get("full/l1_d0", float("nan")), 4)}


@app.function(volumes={DATA_DIR: volume}, timeout=6 * 3600, memory=2048)
def proj_sweep(cells: str = "", tag: str = "", n_clean: int = 6144, max_windows: int = 0,
               max_twins: int = 45000, caliper: float = 2.0, save_caliper: float = 0.35):
    """cells: `ckpt:stim_tag:window_diets:glitch`, comma separated."""
    volume.reload()
    D = f"{DATA_DIR}/v16_s2_L6_m4_distinct/logit_reading"
    if not cells:
        cells = ",".join([
            f"{D}/traj_a1_s42/step064000.pt:a1:1:0",
            f"{D}/traj_a1_s42/step064000.pt:swap65k:0:1",
        ])
    args = []
    for c in cells.split(","):
        p = (c.split(":") + ["1", "0"])[:4]
        ck, tg, wd, gl = p[0], p[1], bool(int(p[2])), bool(int(p[3]))
        if not os.path.exists(ck):
            print(f"MISSING {ck}", flush=True)
            continue
        args.append((ck, tg, n_clean, 2048, max_windows, 11, tag, 3000, 5150, 8192,
                     "post_block7", 1500, wd, max_twins, caliper, 512, 1.0 / 3.0,
                     gl, save_caliper))
    print(f"{len(args)} cells: {[(a[0].split('/')[-1], a[1], a[12], a[17]) for a in args]}",
          flush=True)
    outs = list(proj_ckpt.starmap(args, return_exceptions=True))
    for o in outs:
        print(json.dumps(o, default=str)[:900], flush=True)
    return [str(o)[:900] for o in outs]
