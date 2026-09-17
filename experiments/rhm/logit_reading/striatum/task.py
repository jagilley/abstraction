"""striatum: a critic trained on outcomes alone, reading the same frozen state the
structural readouts read.

THE TASK (fixed actor, frozen trunk). A head on the frozen NTP trunk's residual stream
answers, at window index t, "what is the feature of the level-l constituent (span 2^l)
containing t?" -- v = 16 ways, l = 1..6. The actor is trained on CLEAN windows only and
then frozen, so its clean accuracy per l is a reading of which levels the trunk's state
supports, and the goal it defines is structural, not next-token loss.

THE CRITIC. V[l, d](s_t) predicts the realised outcome o_l(t + d) in {0, 1} -- did the
actor get the level-l query d steps from here right. Fitted by the exact closed-form
ridge solve on a streamed Gram matrix (coeruleus's first gotcha), from realised outcomes
only: no violation label, no consequence label, no k*. An MLP beside it; a shuffled-
outcome head beside every one (`endogenous_teacher`'s control). Floors: the input
embedding, and the random-init trunk as its own checkpoint.

THE VALUE REVISION at an anchor t0, for a query a steps later:

    R[l, a] = V[l, a](s_t0) - V[l, a+1](s_{t0-1})

both predicting the SAME outcome o_l(t0 + a), so the difference is the revision the
token at t0 caused in the expected outcome. Anchors: `first_diff` (the first token that
departs from the unedited stream -- defined for every edit, and LEGAL under p_L in 83% of
swaps) and `t_v` (the Bayesian-detectable violation, swaps only). The unedited window has
a bit-identical prefix up to t0-1, so (edit, orig) is a same-prefix pair for free at every
window, and DR = R_edit - R_orig is the paired revision.

CONSEQUENCE (model-independent, from `parse.py`): c[l, a] = 1 iff the true answer to the
query at t0 + a differs between the original and the edited generating tree. Realised
damage d[l, a] = o_orig - o_edit is the same thing measured at the goal.

Run:
  modal run -m rhm.logit_reading.striatum.task::striatum_ckpt \
      --ckpt /data/v16_s2_L6_m4_distinct/logit_reading/traj_a1_s42/step064000.pt \
      --stim-tag a1 --n-clean 1024 --max-windows 2048 --tag smoke
  modal run --detach -m rhm.logit_reading.striatum.task::striatum_sweep \
      --traj-dir /data/v16_s2_L6_m4_distinct/logit_reading/traj_a1_s42 --stim-tag a1
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


def _ignore(path):
    """Mount the Python sources only, and never a sibling node's `results/` tree --
    another agent writing its launch log there makes the image build race and fail."""
    sp = str(path)
    if "/results/" in sp or sp.endswith("/results") or sp.endswith(".log"):
        return True
    return not sp.endswith(".py")


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==1.26.4", "scipy==1.16.3", "torch==2.7.0")
    .add_local_python_source("rhm", ignore=_ignore)
)
app = modal.App("rhm-striatum", image=image)

LEVELS = [1, 2, 3, 4, 5, 6]
DMAX = 12
A_LIST = [0, 1, 2, 4, 8]
T_LO = 8
LAMS = [1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
BLOCKS = ["post_embed", "post_block5", "post_block7"]


# ---------------------------------------------------------------------------
# actor
# ---------------------------------------------------------------------------

def train_actor(Xtr, ytr, Xva, yva, hidden=256, steps=3000, bs=8192, lr=3e-3, wd=1e-4,
                seed=0, device="cuda"):
    """One v-way head per level. X (N, d) float32 torch on device, y (N,) int64."""
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    d = Xtr.shape[1]
    v = int(max(ytr.max().item(), yva.max().item())) + 1
    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True) + 1e-5
    net = (nn.Linear(d, v) if hidden == 0 else
           nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, v))).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    g = torch.Generator(device="cpu").manual_seed(seed + 1)
    for _ in range(steps):
        i = torch.randint(0, len(Xtr), (min(bs, len(Xtr)),), generator=g).to(device)
        loss = nn.functional.cross_entropy(net((Xtr[i] - mu) / sd), ytr[i])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sch.step()
    net.eval()
    with torch.no_grad():
        acc = float((net((Xva - mu) / sd).argmax(-1) == yva).float().mean())
    return {"net": net, "mu": mu, "sd": sd, "acc": acc}


def actor_apply(head, X):
    """X (..., d) torch -> argmax class (...,)."""
    import torch
    with torch.no_grad():
        return head["net"]((X - head["mu"]) / head["sd"]).argmax(-1)


# ---------------------------------------------------------------------------
# ridge
# ---------------------------------------------------------------------------

def ridge_solve(A, c, lam):
    d = A.shape[0] - 1
    R = np.eye(d + 1) * (lam * np.trace(A[:d, :d]) / max(d, 1))
    R[-1, -1] = 0.0
    return np.linalg.solve(A + R, c)


def _r2(pred, y):
    y = np.asarray(y, np.float64)
    return float(1.0 - ((y - np.asarray(pred, np.float64)) ** 2).mean() / max(y.var(), 1e-12))


def fit_ridge_rows(X, y, Xva, yva, lams=LAMS):
    """Small closed-form ridge on explicit rows; returns (beta, lam, val_r2)."""
    X = np.asarray(X, np.float64)
    Xb = np.concatenate([X, np.ones((len(X), 1))], 1)
    A = Xb.T @ Xb
    c = Xb.T @ np.asarray(y, np.float64)
    Xvb = np.concatenate([np.asarray(Xva, np.float64), np.ones((len(Xva), 1))], 1)
    best = None
    for lam in lams:
        b = ridge_solve(A, c, lam)
        r2 = _r2(Xvb @ b, yva)
        if best is None or r2 > best[2]:
            best = (b, lam, r2)
    return best


# ---------------------------------------------------------------------------
# the checkpoint job
# ---------------------------------------------------------------------------

@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=3 * 3600, memory=24576,
              max_containers=4)
def striatum_ckpt(ckpt: str, stim_tag: str = "a1", n_clean: int = 6144, n_clean_val: int = 2048,
                  max_windows: int = 0, seed: int = 11, tag: str = "", actor_steps: int = 3000,
                  mlp_steps: int = 2000, eval_seed: int = 5150, skip_mlp: bool = False,
                  prim_block: str = "post_block7", n_clean_crit: int = 8192):
    import torch
    volume.reload()
    t00 = time.time()
    dev = "cuda"
    model, rules, rule_w, cfg = load_trajectory_ckpt(ckpt, dev)
    L, s, v = cfg["L"], cfg["s"], cfg["v"]
    T = model.block_size
    step = cfg.get("step")
    print(f"loaded {ckpt} step {step}", flush=True)

    key = tb_key(cfg["v"], cfg["s"], cfg["L"], cfg["m"])
    ddir = f"{DATA_DIR}/{key}/logit_reading"
    S = np.load(f"{ddir}/stimuli_{stim_tag}.npz")
    P = np.load(f"{ddir}/parse_{stim_tag}.npz")
    y_orig_all, y_edit_all = P["y_orig"], P["y_edit"]            # (6, n, T1)
    n_all = S["windows_edit"].shape[0]
    idx = np.arange(n_all)
    if max_windows and max_windows < n_all:
        idx = np.sort(np.random.default_rng(seed).choice(n_all, max_windows, replace=False))
    We, Wo = S["windows_edit"][idx], S["windows_orig"][idx]
    meta = {k: S[k][idx] for k in ("phase", "etype", "j", "e", "first_diff", "t_v", "k_star")}
    y_edit = y_edit_all[:, idx][:, :, :T]
    y_orig = y_orig_all[:, idx][:, :, :T]
    n = len(idx)
    print(f"stimuli {stim_tag}: n={n}", flush=True)

    # ---- splits over windows -------------------------------------------------
    rng = np.random.default_rng(seed + 1)
    perm = rng.permutation(n)
    n_tr, n_va = int(0.60 * n), int(0.15 * n)
    split = np.zeros(n, np.int8)
    split[perm[n_tr:n_tr + n_va]] = 1
    split[perm[n_tr + n_va:]] = 2
    is_tr, is_va, is_te = split == 0, split == 1, split == 2

    # ---- the banked coeruleus excess head (only where it was built) ----------
    banked, banked_cfg = None, None
    bpath = ckpt[:-3] + "_coeruleus_head_state_excess_ridge_onnoise.pt"
    if not os.path.exists(bpath):
        bpath = ckpt[:-3] + "_coeruleus_head_onnoise.pt"
    if os.path.exists(bpath):
        from rhm.logit_reading.coeruleus.readout import load_head
        banked, banked_cfg = load_head(bpath, dev)
        print(f"banked coeruleus head: {bpath} block={banked.block}", flush=True)
    blocks = list(BLOCKS)
    if banked is not None and banked.block not in blocks:
        blocks.append(banked.block)

    # ---- pass C: clean windows -> the actor ---------------------------------
    from rhm.logit_reading.altitude.units import windows_with_parse
    from rhm.logit_reading.striatum.parse import clean_answers
    t0 = time.time()
    n_cl = n_clean + n_clean_val
    wins_c, phase_c, _, fs_c = windows_with_parse(rules, rule_w, n_cl, T + 1, eval_seed)
    y_c = clean_answers(fs_c, phase_c, T + 1, L, s, 3)[:, :, :T]        # (6, n_cl, T)
    print(f"clean windows {time.time() - t0:.1f}s", flush=True)

    def states_all(windows, blks, bs=256):
        out = {b: np.zeros((len(windows), T, model.transformer.wte.weight.shape[1]),
                           np.float32) for b in blks}
        with torch.no_grad():
            for c0 in range(0, len(windows), bs):
                x = torch.as_tensor(windows[c0:c0 + bs, :T], device=dev)
                _, _, inter = model(x, return_intermediates=True)
                for b in blks:
                    out[b][c0:c0 + bs] = inter[b].float().cpu().numpy()
        return out

    t0 = time.time()
    Sc = states_all(wins_c, blocks)
    pos = np.arange(T_LO, T)
    actors, actor_tab = {}, {}
    for b in blocks:
        Xtr = torch.as_tensor(Sc[b][:n_clean][:, pos].reshape(-1, Sc[b].shape[-1]), device=dev)
        Xva = torch.as_tensor(Sc[b][n_clean:][:, pos].reshape(-1, Sc[b].shape[-1]), device=dev)
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
    prim = prim_block if prim_block in blocks else max(
        blocks, key=lambda b: np.mean([actor_tab[f"{b}/l{l}"] for l in LEVELS]))
    print(f"actors {time.time() - t0:.1f}s  primary block {prim}", flush=True)
    print(json.dumps({k: round(x, 4) for k, x in actor_tab.items()}), flush=True)
    del Sc
    torch.cuda.empty_cache()

    # ---- anchor positions (needed before the passes, to keep states small) ---
    fd, tv, et, jj, ks, ee = (meta["first_diff"], meta["t_v"], meta["etype"], meta["j"],
                              meta["k_star"], meta["e"])
    anchors = {"fd": np.where(fd >= 0, fd, ee), "tv": np.where(tv > 0, tv, -1)}
    anchor_ok = {"fd": ((fd >= T_LO + 1) | ((et == 2) & (ee >= T_LO + 1))) & (fd <= T - 1),
                 "tv": (tv > T_LO) & (et == 0) & (tv <= T - 1)}
    anc_pos = np.stack([np.clip(anchors["fd"] - 1, 0, T - 1), np.clip(anchors["fd"], 0, T - 1),
                        np.clip(anchors["tv"] - 1, 0, T - 1), np.clip(anchors["tv"], 0, T - 1)], 1)
    ANC = {"fd": (0, 1), "tv": (2, 3)}

    # ---- pass E/O: outcomes, logq, anchor states, Gram -----------------------
    d = model.transformer.wte.weight.shape[1]
    A = {b: torch.zeros(d + 1, d + 1, dtype=torch.float64, device=dev) for b in blocks}
    tgt_names = [f"l{l}_d{dd}" for l in LEVELS for dd in range(DMAX + 1)]
    tgt_names += [f"sh_l{l}_d{dd}" for l in LEVELS for dd in range(DMAX + 1)]
    n_tgt = len(tgt_names)
    C = {b: torch.zeros(d + 1, n_tgt, dtype=torch.float64, device=dev) for b in blocks}
    T_HI = T - 1 - DMAX                       # last position with a full horizon
    rows = np.arange(T_LO, T_HI + 1)
    # windows whose FULL states are kept (lambda selection + the MLP critic)
    sub_tr = np.where(is_tr)[0]
    sub_tr = sub_tr[:min(len(sub_tr), 3000)]
    sub_va = np.where(is_va)[0]
    sub_va = sub_va[:min(len(sub_va), 3000)]
    keep_full = np.zeros(n, bool)
    keep_full[sub_tr] = True
    keep_full[sub_va] = True
    kf_idx = np.where(keep_full)[0]
    kf_map = -np.ones(n, np.int64)
    kf_map[kf_idx] = np.arange(len(kf_idx))

    def run_windows(W, Y, gram, gseed=0, want_full=False):
        o = np.zeros((len(LEVELS), len(W), T), np.uint8)
        nll = np.zeros((len(W), T), np.float32)
        Hq = np.zeros((len(W), T), np.float32)
        anc = {b: np.zeros((len(W), 4, d), np.float32) for b in blocks}
        full = ({b: np.zeros((len(kf_idx), T, d), np.float32) for b in blocks}
                if want_full else None)
        bs = 256
        g = torch.Generator().manual_seed(gseed)
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
                for b in blocks:
                    anc[b][sl] = inter[b][ar, ap].float().cpu().numpy()
                if want_full:
                    kk = np.where(keep_full[sl])[0]
                    if len(kk):
                        kt = torch.as_tensor(kk, device=dev)
                        dst = kf_map[c0 + kk]
                        for b in blocks:
                            full[b][dst] = inter[b][kt].float().cpu().numpy()
                if gram:
                    ob = torch.as_tensor(o[:, sl], dtype=torch.float64, device=dev)
                    keep = torch.as_tensor(np.where(is_tr[sl])[0], device=dev)
                    if len(keep):
                        shk = keep[torch.randperm(len(keep), generator=g).to(dev)]
                        rw = torch.as_tensor(rows, device=dev)
                        ys = []
                        for src in (keep, shk):
                            for li, l in enumerate(LEVELS):
                                for dd in range(DMAX + 1):
                                    ys.append(ob[li][src][:, rw + dd])
                        Yb = torch.stack(ys, -1).reshape(-1, n_tgt)
                        for b in blocks:
                            Xb = inter[b][keep][:, rw].double().reshape(-1, d)
                            Xb = torch.cat([Xb, torch.ones(len(Xb), 1, dtype=torch.float64,
                                                           device=dev)], 1)
                            A[b] += Xb.T @ Xb
                            C[b] += Xb.T @ Yb
        return o, nll, Hq, anc, full

    t0 = time.time()
    o_e, nll_e, H_e, st_e, full_e = run_windows(We, y_edit, True, gseed=seed + 3, want_full=True)
    print(f"edit pass {time.time() - t0:.1f}s", flush=True)
    t0 = time.time()
    o_o, nll_o, H_o, st_o, _ = run_windows(Wo, y_orig, False)
    print(f"orig pass {time.time() - t0:.1f}s", flush=True)

    # ---- solve the critic (all targets, all lambdas, vectorised) -------------
    An = {b: A[b].cpu().numpy() for b in blocks}
    Cn = {b: C[b].cpu().numpy() for b in blocks}
    del A, C
    torch.cuda.empty_cache()
    va = sub_va
    vr = kf_map[va]
    Yv = np.zeros((len(va) * len(rows), n_tgt), np.float32)
    for ti, nm in enumerate(tgt_names):
        base = nm[3:] if nm.startswith("sh_") else nm
        l = int(base.split("l")[1].split("_")[0]); dd = int(base.split("_d")[1])
        Yv[:, ti] = o_e[LEVELS.index(l)][va][:, rows + dd].reshape(-1)
    beta, lam_pick, val_r2 = {}, {}, {}
    yvar = Yv.astype(np.float64).var(0)
    for b in blocks:
        Xv = np.concatenate([full_e[b][vr][:, rows].reshape(-1, d).astype(np.float64),
                             np.ones((len(va) * len(rows), 1))], 1)
        best_r2 = np.full(n_tgt, -np.inf)
        for lm in LAMS:
            B = ridge_solve(An[b], Cn[b], lm)                      # (d+1, n_tgt)
            pr = Xv @ B
            r2 = 1.0 - ((pr - Yv) ** 2).mean(0) / np.maximum(yvar, 1e-12)
            for ti in np.where(r2 > best_r2)[0]:
                best_r2[ti] = r2[ti]
                beta[(b, tgt_names[ti])] = B[:, ti].copy()
                lam_pick[(b, tgt_names[ti])] = lm
        for ti, nm in enumerate(tgt_names):
            val_r2[(b, nm)] = float(best_r2[ti])
        del Xv
    crit_block = max(blocks, key=lambda b: np.mean(
        [val_r2[(b, f"l{l}_d{dd}")] for l in LEVELS for dd in (0, 1, 2, 4, 8)]))
    print(f"critic solved; best block {crit_block}", flush=True)

    # ---- the CLEAN-ONLY critic: the same frozen actor, outcomes from a world with no
    # edits in it at all. Its revision at a violation is whatever generalises, which is
    # the floor that says how much of the trained critic's reading came from having been
    # exposed to damage.
    t0 = time.time()
    real_names = [f"l{l}_d{dd}" for l in LEVELS for dd in range(DMAX + 1)]
    wins_cc, phase_cc, _, fs_cc = windows_with_parse(rules, rule_w, n_clean_crit, T + 1,
                                                     eval_seed + 1000)
    y_cc = clean_answers(fs_cc, phase_cc, T + 1, L, s, 3)[:, :, :T]
    n_cc_va = min(2000, n_clean_crit // 5)
    cc_tr = np.arange(n_clean_crit - n_cc_va)
    cc_va = np.arange(n_clean_crit - n_cc_va, n_clean_crit)
    Acc = torch.zeros(d + 1, d + 1, dtype=torch.float64, device=dev)
    Ccc = torch.zeros(d + 1, len(real_names), dtype=torch.float64, device=dev)
    o_cc = np.zeros((len(LEVELS), n_clean_crit, T), np.uint8)
    Xcc_va = np.zeros((n_cc_va, T, d), np.float32)
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
            keep = torch.as_tensor(np.where(np.arange(sl.start, sl.stop)
                                            < len(cc_tr))[0], device=dev)
            if len(keep):
                ob = torch.as_tensor(o_cc[:, sl], dtype=torch.float64, device=dev)
                rw = torch.as_tensor(rows, device=dev)
                ys = [ob[li][keep][:, rw + dd] for li, l in enumerate(LEVELS)
                      for dd in range(DMAX + 1)]
                Yb = torch.stack(ys, -1).reshape(-1, len(real_names))
                Xb = inter[crit_block][keep][:, rw].double().reshape(-1, d)
                Xb = torch.cat([Xb, torch.ones(len(Xb), 1, dtype=torch.float64,
                                               device=dev)], 1)
                Acc += Xb.T @ Xb
                Ccc += Xb.T @ Yb
            vk = np.where(np.arange(sl.start, sl.stop) >= len(cc_tr))[0]
            if len(vk):
                Xcc_va[np.arange(sl.start, sl.stop)[vk] - len(cc_tr)] = \
                    inter[crit_block][torch.as_tensor(vk, device=dev)].float().cpu().numpy()
    Acc_n, Ccc_n = Acc.cpu().numpy(), Ccc.cpu().numpy()
    del Acc, Ccc
    torch.cuda.empty_cache()
    Xvc = np.concatenate([Xcc_va[:, rows].reshape(-1, d).astype(np.float64),
                          np.ones((n_cc_va * len(rows), 1))], 1)
    Yvc = np.stack([o_cc[LEVELS.index(int(nm.split("l")[1].split("_")[0]))][cc_va][
        :, rows + int(nm.split("_d")[1])].reshape(-1) for nm in real_names], 1).astype(np.float64)
    best_c = np.full(len(real_names), -np.inf)
    clean_val_r2 = {}
    for lm in LAMS:
        B = ridge_solve(Acc_n, Ccc_n, lm)
        r2 = 1.0 - ((Xvc @ B - Yvc) ** 2).mean(0) / np.maximum(Yvc.var(0), 1e-12)
        for ti in np.where(r2 > best_c)[0]:
            best_c[ti] = r2[ti]
            beta[("clean", real_names[ti])] = B[:, ti].copy()
    for ti, nm in enumerate(real_names):
        clean_val_r2[nm] = float(best_c[ti])
    del Xvc, Yvc, Xcc_va
    print(f"clean-only critic {time.time() - t0:.1f}s  "
          f"val R2 l1_d0 {clean_val_r2['l1_d0']:.4f}", flush=True)

    # ---- optional MLP critic beside the linear one --------------------------
    mlp_val = {}
    if not skip_mlp:
        from rhm.logit_reading.coeruleus.readout import train_head
        tr_r = kf_map[sub_tr]
        Xtr = full_e[crit_block][tr_r][:, rows].reshape(-1, d)
        Xva_ = full_e[crit_block][vr][:, rows].reshape(-1, d)
        for l in LEVELS:
            li = LEVELS.index(l)
            for dd in (0, 1, 2):
                ytr = o_e[li][sub_tr][:, rows + dd].reshape(-1).astype(np.float32)
                yva_ = o_e[li][va][:, rows + dd].reshape(-1).astype(np.float64)
                h = train_head(Xtr, ytr, steps=mlp_steps, device=dev, seed=l * 17 + dd)
                mlp_val[f"l{l}_d{dd}"] = _r2(h(Xva_), yva_)
                beta[("mlp", f"l{l}_d{dd}")] = h
        del Xtr, Xva_
        torch.cuda.empty_cache()
    del full_e
    # ---- anchors, revisions, labels -----------------------------------------
    def predict(b, name, X):
        if b == "mlp":
            return beta[(b, name)](X)
        return X @ beta[(b, name)][:-1] + beta[(b, name)][-1]

    pL = S["pL_edit"][idx] if "pL_edit" in S.files else None
    out_rows = {}
    for an, t0a in anchors.items():
        ok = anchor_ok[an]
        w = np.where(ok)[0]
        if len(w) < 50:
            continue
        ib, ia = ANC[an]
        t0w = t0a[w]
        rec = {"w": w, "t0": t0w, "etype": et[w], "j": jj[w], "k_star": ks[w],
               "e": ee[w], "split": split[w], "first_diff": fd[w], "t_v": tv[w]}
        rec["nll_e"] = nll_e[w, t0w - 1]
        rec["nll_o"] = nll_o[w, t0w - 1]
        rec["H_pre"] = H_e[w, t0w - 1]
        rec["dH_e"] = H_e[w, t0w] - H_e[w, t0w - 1]
        rec["dH_o"] = H_o[w, t0w] - H_o[w, t0w - 1]
        rec["excess_e"] = rec["nll_e"] - rec["H_pre"]
        rec["tok_legal"] = (np.zeros(len(w), np.int8) if pL is None else
                            (pL[w, t0w, We[w, t0w]] > 0).astype(np.int8))
        # prefix guard: the unedited window is bit-identical up to t0-1 for the fd anchor
        rec["prefix_dev"] = np.abs(st_e[crit_block][w, ib] - st_o[crit_block][w, ib]).max(1)
        cand = ([crit_block, "clean"] + ([] if skip_mlp else ["mlp"])
                + [b for b in blocks if b != crit_block])
        for bsel in cand:
            src_b = crit_block if bsel in ("mlp", "clean") else bsel
            Xa, Xb_ = st_e[src_b][w, ia], st_e[src_b][w, ib]
            Xao, Xbo = st_o[src_b][w, ia], st_o[src_b][w, ib]
            for l in LEVELS:
                for a in A_LIST:
                    nm0, nm1 = f"l{l}_d{a}", f"l{l}_d{a + 1}"
                    if (bsel, nm0) not in beta or (bsel, nm1) not in beta:
                        continue
                    rec[f"R_{bsel}_l{l}_a{a}"] = predict(bsel, nm0, Xa) - predict(bsel, nm1, Xb_)
                    rec[f"Ro_{bsel}_l{l}_a{a}"] = predict(bsel, nm0, Xao) - predict(bsel, nm1, Xbo)
                    if bsel == crit_block:
                        rec[f"Rsh_l{l}_a{a}"] = (predict(bsel, f"sh_{nm0}", Xa)
                                                 - predict(bsel, f"sh_{nm1}", Xb_))
                        rec[f"V_l{l}_a{a}"] = predict(bsel, nm0, Xa)
                        rec[f"Vpre_l{l}_a{a}"] = predict(bsel, nm1, Xb_)
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
        # oracle probes on the same anchor states (shared Gram, all labels at once)
        Xa = st_e[crit_block][w, ia].astype(np.float64)
        tr_m, va_m = rec["split"] == 0, rec["split"] == 1
        Xtr_b = np.concatenate([Xa[tr_m], np.ones((int(tr_m.sum()), 1))], 1)
        Xva_b = np.concatenate([Xa[va_m], np.ones((int(va_m.sum()), 1))], 1)
        Ag = Xtr_b.T @ Xtr_b
        labs = {"probe_illegal": ((rec["etype"] == 0) & (rec["t_v"] > 0)).astype(np.float64)}
        for l in LEVELS:
            for a in A_LIST:
                labs[f"probe_cons_l{l}_a{a}"] = rec[f"cons_l{l}_a{a}"].astype(np.float64)
        names = [k for k, vv in labs.items() if 0 < vv[tr_m].mean() < 1]
        if names and tr_m.sum() > 50:
            Yl = np.stack([labs[k][tr_m] for k in names], 1)
            Yl_v = np.stack([labs[k][va_m] for k in names], 1)
            Cg = Xtr_b.T @ Yl
            best = np.full(len(names), -np.inf)
            Bbest = np.zeros((Xa.shape[1] + 1, len(names)))
            for lm in LAMS:
                B = ridge_solve(Ag, Cg, lm)
                r2 = 1.0 - ((Xva_b @ B - Yl_v) ** 2).mean(0) / np.maximum(Yl_v.var(0), 1e-12)
                for ti in np.where(r2 > best)[0]:
                    best[ti] = r2[ti]
                    Bbest[:, ti] = B[:, ti]
            Xa_b = np.concatenate([Xa, np.ones((len(Xa), 1))], 1)
            pr = Xa_b @ Bbest
            for ti, k in enumerate(names):
                rec[k] = pr[:, ti]
        if banked is not None:
            rec["banked_excess"] = banked(st_e[banked.block][w, ia])
            rec["banked_excess_pre"] = banked(st_e[banked.block][w, ib])
        out_rows[an] = rec

    # ---- the value trace around the anchor (test windows, one extra pass) ----
    TAUS = list(range(-3, 9))
    te_idx = np.where(is_te)[0]
    Vt = np.zeros((len(te_idx), T, len(LEVELS)), np.float32)
    with torch.no_grad():
        for c0 in range(0, len(te_idx), 256):
            ii = te_idx[c0:c0 + 256]
            x = torch.as_tensor(We[ii, :T], device=dev)
            _, _, inter = model(x, return_intermediates=True)
            Xb = inter[crit_block].float().cpu().numpy()
            for li, l in enumerate(LEVELS):
                bb = beta[(crit_block, f"l{l}_d0")]
                Vt[c0:c0 + len(ii), :, li] = Xb @ bb[:-1] + bb[-1]
    pos_of = {int(w_): i for i, w_ in enumerate(te_idx)}
    res_trace = {}
    for an, rec in out_rows.items():
        te_m = rec["split"] == 2
        w = rec["w"][te_m]
        if len(w) < 20:
            continue
        ri = np.array([pos_of[int(x)] for x in w])
        t0w = rec["t0"][te_m]
        grp = {"all": np.ones(len(w), bool), "swap": rec["etype"][te_m] == 0,
               "rare": rec["etype"][te_m] == 1, "none": rec["etype"][te_m] == 2}
        ent = {"taus": TAUS, "levels": LEVELS, "n": int(len(w))}
        for gname, gm in grp.items():
            if gm.sum() < 10:
                continue
            ent[f"mean_V/{gname}"] = {
                str(tau): Vt[ri[gm], np.clip(t0w[gm] + tau, 0, T - 1)].mean(0).tolist()
                for tau in TAUS}
        for l in LEVELS:
            c = np.asarray(rec[f"cons_l{l}_a0"], bool)[te_m]
            for gname, gm in (("cons", c), ("incons", ~c)):
                if gm.sum() < 10:
                    continue
                ent[f"mean_V_l{l}/{gname}"] = {
                    str(tau): float(Vt[ri[gm], np.clip(t0w[gm] + tau, 0, T - 1),
                                       LEVELS.index(l)].mean()) for tau in TAUS}
        res_trace[an] = ent

    # ---- summary tables ------------------------------------------------------
    res = {"ckpt": ckpt, "step": step, "stim_tag": stim_tag, "n": n,
           "primary_block": prim, "critic_block": crit_block,
           "actor_clean_acc": actor_tab, "mlp_val_r2": mlp_val,
           "clean_critic_val_r2": clean_val_r2,
           "critic_diet": {"trained": f"stimulus windows ({stim_tag}) train split, "
                                      "realised outcomes only",
                           "clean": f"{n_clean_crit} fresh clean windows, same frozen actor"},
           "critic_val_r2": {f"{b}/{nm}": val_r2[(b, nm)] for b in blocks for nm in tgt_names
                             if not nm.startswith("sh_")},
           "critic_val_r2_shuf": {f"{b}/{nm}": val_r2[(b, nm)] for b in blocks
                                  for nm in tgt_names if nm.startswith("sh_")},
           "config": {"levels": LEVELS, "dmax": DMAX, "a_list": A_LIST, "t_lo": T_LO,
                      "blocks": blocks, "n_clean": n_clean, "seed": seed,
                      "eval_seed": eval_seed, "split": [0.6, 0.15, 0.25],
                      "n_clean_crit": n_clean_crit},
           "banked_head": bpath if banked is not None else None}
    res["trace"] = res_trace
    res["tables"] = summarize(out_rows, crit_block, skip_mlp)

    stem = ckpt[:-3]
    sfx = f"_striatum_{stim_tag}" + (f"_{tag}" if tag else "")
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
    return {"ckpt": ckpt, "step": step, "actor": actor_tab,
            "headline": res["tables"].get("headline")}


# ---------------------------------------------------------------------------
# reductions (run on the container so the artefacts stay small)
# ---------------------------------------------------------------------------

def _rank(x):
    from scipy.stats import rankdata
    return rankdata(x) / (len(x) + 1.0)


def match_nn(keys, y, x, caliper, rng):
    """1:1 nearest-neighbour matching on `x` inside each exact stratum of `keys`.

    Returns the matched index set (equal positives and negatives per stratum, each pair
    within `caliper` on x), the design the parent used for its phasic controls."""
    out = []
    for kk in np.unique(keys):
        pos = np.where((keys == kk) & y)[0]
        neg = np.where((keys == kk) & ~y)[0]
        if not len(pos) or not len(neg):
            continue
        pos = pos[np.argsort(x[pos])]
        neg = list(neg[np.argsort(x[neg])])
        used = []
        for p in pos:
            if not neg:
                break
            dd = np.abs(x[np.array(neg)] - x[p])
            k = int(np.argmin(dd))
            if dd[k] <= caliper:
                used.append((p, neg.pop(k)))
        for p, q in used:
            out.append(p)
            out.append(q)
    return np.array(sorted(out), dtype=np.int64)


def summarize(out_rows, cb, skip_mlp, caliper=0.3, seed=0):
    rng = np.random.default_rng(seed)
    tab = {}
    for an, rec in out_rows.items():
        te = rec["split"] == 2
        sw, ra, no = rec["etype"] == 0, rec["etype"] == 1, rec["etype"] == 2
        t = {"n_test": int(te.sum()),
             "n_test_by_etype": [int((te & sw).sum()), int((te & ra).sum()),
                                 int((te & no).sum())],
             "prefix_dev_max": float(np.abs(rec["prefix_dev"]).max()),
             "tok_legal_rate_swap": float(rec["tok_legal"][sw].mean()) if sw.any() else None,
             "k_star_hist": np.bincount(np.clip(rec["k_star"][te & sw], 0, 6),
                                        minlength=7).tolist(),
             "j_hist": np.bincount(rec["j"][te], minlength=5).tolist(),
             "n_cons_test": {f"l{l}_a{a}": int((np.asarray(rec[f"cons_l{l}_a{a}"], bool)
                                                & np.asarray(rec[f"ok_l{l}_a{a}"], bool)
                                                & te).sum())
                             for l in LEVELS for a in A_LIST},
             "n_incons_test": {f"l{l}_a{a}": int((~np.asarray(rec[f"cons_l{l}_a{a}"], bool)
                                                  & np.asarray(rec[f"ok_l{l}_a{a}"], bool)
                                                  & te & (rec["etype"] < 2)).sum())
                               for l in LEVELS for a in A_LIST}}
        t["cons_rate"] = {}
        for etn, es in (("all", te), ("swap", te & sw), ("rare", te & ra)):
            if es.sum() < 5:
                continue
            t["cons_rate"][etn] = {f"l{l}_a{a}": float(rec[f"cons_l{l}_a{a}"][es].mean())
                                   for l in LEVELS for a in A_LIST}
        t["cons_rate_by_kstar"] = {
            f"k{k}": {f"l{l}_a0": float(rec[f"cons_l{l}_a0"][te & sw & (rec["k_star"] == k)].mean())
                      for l in LEVELS}
            for k in range(1, 7) if (te & sw & (rec["k_star"] == k)).sum() >= 10}
        t["cons_rate_by_j"] = {
            f"j{jv}": {f"l{l}_a0": float(rec[f"cons_l{l}_a0"][te & (rec["j"] == jv)].mean())
                       for l in LEVELS}
            for jv in range(5) if (te & (rec["j"] == jv)).sum() >= 10}

        for l in LEVELS:
            for a in A_LIST:
                k = f"R_{cb}_l{l}_a{a}"
                if k not in rec:
                    continue
                base = f"l{l}_a{a}"
                R, Ro = np.asarray(rec[k]), np.asarray(rec[f"Ro_{cb}_l{l}_a{a}"])
                dR = R - Ro
                c = np.asarray(rec[f"cons_l{l}_a{a}"], bool)
                okm = np.asarray(rec[f"ok_l{l}_a{a}"], bool) & te
                dmg = (rec[f"oo_l{l}_a{a}"].astype(np.int16)
                       - rec[f"oe_l{l}_a{a}"].astype(np.int16))
                t.setdefault("R_mean", {})[base] = {
                    "swap": _m(R, okm & sw), "rare": _m(R, okm & ra), "none": _m(R, okm & no),
                    "orig": _m(Ro, okm), "cons": _m(R, okm & c), "incons": _m(R, okm & ~c)}
                t.setdefault("dR_mean", {})[base] = {"cons": _m(dR, okm & c),
                                                     "incons": _m(dR, okm & ~c)}
                t.setdefault("damage", {})[base] = {
                    "cons": _m(dmg, okm & c), "incons": _m(dmg, okm & ~c),
                    "swap": _m(dmg, okm & sw), "rare": _m(dmg, okm & ra),
                    "none": _m(dmg, okm & no)}
                t.setdefault("acc_edit", {})[base] = {
                    "cons": _m(rec[f"oe_l{l}_a{a}"], okm & c),
                    "incons": _m(rec[f"oe_l{l}_a{a}"], okm & ~c),
                    "orig": _m(rec[f"oo_l{l}_a{a}"], okm)}
                scores = {"R": -R, "dR": -dR, "Rshuf": -np.asarray(rec.get(f"Rsh_l{l}_a{a}", R)),
                          "V": -np.asarray(rec.get(f"V_l{l}_a{a}", R)),
                          "nll": np.asarray(rec["nll_e"]), "dH": -np.asarray(rec["dH_e"]),
                          "excess": np.asarray(rec["excess_e"])}
                if f"R_mlp_l{l}_a{a}" in rec:
                    scores["R_mlp"] = -np.asarray(rec[f"R_mlp_l{l}_a{a}"])
                if f"R_clean_l{l}_a{a}" in rec:
                    scores["R_clean"] = -np.asarray(rec[f"R_clean_l{l}_a{a}"])
                    scores["dR_clean"] = -(np.asarray(rec[f"R_clean_l{l}_a{a}"])
                                           - np.asarray(rec[f"Ro_clean_l{l}_a{a}"]))
                for nm in ("probe_cons", "probe_illegal", "banked_excess"):
                    key = f"{nm}_l{l}_a{a}" if nm == "probe_cons" else nm
                    if key in rec:
                        scores[nm] = np.asarray(rec[key])
                m = okm & (sw | ra)
                if m.sum() > 20 and 0 < c[m].mean() < 1:
                    for nm, sc in scores.items():
                        t.setdefault(f"auc_cons/{nm}", {})[base] = auc(sc[m], c[m])
                    # matched: exact (k*, j) cells, nearest neighbour on model surprisal
                    keys = np.asarray(rec["k_star"]) * 10 + np.asarray(rec["j"])
                    mi = np.where(m)[0]
                    sel = match_nn(keys[mi], c[mi], np.asarray(rec["nll_e"])[mi], caliper, rng)
                    if len(sel) >= 40:
                        mm = mi[sel]
                        t.setdefault("matched_n", {})[base] = int(len(mm))
                        for nm, sc in scores.items():
                            t.setdefault(f"auc_cons_matched/{nm}", {})[base] = auc(sc[mm], c[mm])
                        t.setdefault("auc_cons_matched/guard_nll", {})[base] = \
                            auc(np.asarray(rec["nll_e"])[mm], c[mm])
                        t.setdefault("auc_cons_matched/guard_t0", {})[base] = \
                            auc(np.asarray(rec["t0"], np.float64)[mm], c[mm])
                        t.setdefault("auc_cons_matched/guard_kstar", {})[base] = \
                            auc(np.asarray(rec["k_star"], np.float64)[mm], c[mm])
                    sel2 = match_nn(np.asarray(rec["j"])[mi], c[mi],
                                    np.asarray(rec["nll_e"])[mi], caliper, rng)
                    if len(sel2) >= 40:
                        m2 = mi[sel2]
                        t.setdefault("matched_j_n", {})[base] = int(len(m2))
                        for nm, sc in scores.items():
                            t.setdefault(f"auc_cons_matchedj/{nm}", {})[base] = auc(sc[m2], c[m2])
                        t.setdefault("auc_cons_matchedj/guard_nll", {})[base] = \
                            auc(np.asarray(rec["nll_e"])[m2], c[m2])
                # realised damage at the goal: the actor was right on the unedited window
                # and wrong on the edited one -- the model-dependent cost the critic is
                # actually trained on
                oo = np.asarray(rec[f"oo_l{l}_a{a}"]).astype(bool)
                flip = oo & ~np.asarray(rec[f"oe_l{l}_a{a}"]).astype(bool)
                mf = okm & (sw | ra) & oo
                if mf.sum() > 20 and 0 < flip[mf].mean() < 1:
                    t.setdefault("flip_n", {})[base] = int(mf.sum())
                    for nm, sc in scores.items():
                        t.setdefault(f"auc_flip/{nm}", {})[base] = auc(sc[mf], flip[mf])
                    keysf = np.asarray(rec["k_star"]) * 10 + np.asarray(rec["j"])
                    mif = np.where(mf)[0]
                    self2 = match_nn(keysf[mif], flip[mif], np.asarray(rec["nll_e"])[mif],
                                     caliper, rng)
                    if len(self2) >= 40:
                        mm2 = mif[self2]
                        t.setdefault("flip_matched_n", {})[base] = int(len(mm2))
                        for nm, sc in scores.items():
                            t.setdefault(f"auc_flip_matched/{nm}", {})[base] = \
                                auc(sc[mm2], flip[mm2])
                # does V mean anything at these rows at all
                if okm.sum() > 20:
                    oe_ = np.asarray(rec[f"oe_l{l}_a{a}"]).astype(bool)
                    if 0 < oe_[okm].mean() < 1 and f"V_l{l}_a{a}" in rec:
                        t.setdefault("auc_outcome/V", {})[base] = auc(
                            np.asarray(rec[f"V_l{l}_a{a}"])[okm], oe_[okm])
                        t.setdefault("auc_outcome/Vpre", {})[base] = auc(
                            np.asarray(rec[f"Vpre_l{l}_a{a}"])[okm], oe_[okm])
                # illegal-vs-legal at matched consequence: swap vs rare among cons episodes
                mc = okm & c & (sw | ra)
                if mc.sum() > 20 and 0 < sw[mc].mean() < 1:
                    t.setdefault("auc_illegal_given_cons/R", {})[base] = auc(-R[mc], sw[mc])
                    t.setdefault("auc_illegal_given_cons/nll", {})[base] = \
                        auc(np.asarray(rec["nll_e"])[mc], sw[mc])
        t["across_level"] = across_level(rec, cb, te)
        t["across_level_clean"] = across_level(rec, "clean", te)
        t["across_level_by_j"] = {f"j{jv}": across_level(rec, cb, te & (rec["j"] == jv))
                                  for jv in range(1, 5) if (te & (rec["j"] == jv)).sum() >= 30}
        tab[an] = t
    tab["headline"] = {an: tab[an].get("across_level", {}).get("a0", {}).get("win_rate")
                       for an in out_rows}
    return tab


def _m(x, m):
    x = np.asarray(x)
    return float(x[m].mean()) if np.asarray(m).sum() else None


def across_level(rec, cb, te, lmax=4):
    """At a fixed (window, anchor, a) the state is IDENTICAL across query levels, so a
    difference between the levels whose answer the edit changed and the levels whose
    answer it did not cannot come from anything a state-only readout can see. Levels are
    made comparable by ranking within level. Restricted to l <= lmax, the levels at which
    an edit can be consequential at all."""
    out = {}
    lv = [l for l in LEVELS if l <= lmax]
    for a in A_LIST:
        Rs, Cs, Ok = [], [], []
        for l in lv:
            k = f"R_{cb}_l{l}_a{a}"
            if k not in rec:
                return out
            Rs.append(np.asarray(rec[k]))
            Cs.append(np.asarray(rec[f"cons_l{l}_a{a}"], bool))
            Ok.append(np.asarray(rec[f"ok_l{l}_a{a}"], bool))
        R, Cc = np.stack(Rs, 1), np.stack(Cs, 1)
        Ok = np.stack(Ok, 1) & np.asarray(te)[:, None]
        Q = np.zeros_like(R)
        for li in range(R.shape[1]):
            m = Ok[:, li]
            if m.sum() > 2:
                Q[m, li] = _rank(-R[m, li])
        has, hasnt = Ok & Cc, Ok & ~Cc
        rowok = (has.sum(1) > 0) & (hasnt.sum(1) > 0)
        if rowok.sum() < 20:
            continue
        mc = (Q * has).sum(1)[rowok] / np.maximum(has.sum(1)[rowok], 1)
        mi = (Q * hasnt).sum(1)[rowok] / np.maximum(hasnt.sum(1)[rowok], 1)
        # the scalar comparison: the window's mean revision cannot express which level
        sc = (Q * Ok).sum(1)[rowok] / np.maximum(Ok.sum(1)[rowok], 1)
        out[f"a{a}"] = {"n": int(rowok.sum()), "mean_rank_cons": float(mc.mean()),
                        "mean_rank_incons": float(mi.mean()),
                        "win_rate": float((mc > mi).mean() + 0.5 * (mc == mi).mean()),
                        "scalar_rank_mean": float(sc.mean())}
    return out


# ---------------------------------------------------------------------------
# coordinator
# ---------------------------------------------------------------------------

@app.function(volumes={DATA_DIR: volume}, timeout=6 * 3600, memory=2048)
def striatum_sweep(ckpts: str = "", traj_dir: str = "", steps: str = "0,8000,24000,64000",
                   stim_tags: str = "a1,swap65k", n_clean: int = 6144,
                   max_windows: int = 0, tag: str = ""):
    """CPU coordinator: every (checkpoint, venue) cell in its own container."""
    volume.reload()
    ck = ([c for c in ckpts.split(",") if c] if ckpts else
          [f"{traj_dir}/step{int(x):06d}.pt" for x in steps.split(",")])
    ck = [c for c in ck if os.path.exists(c)]
    args = [(c, tg, n_clean, 2048, max_windows, 11, tag)
            for c in ck for tg in stim_tags.split(",")]
    print(f"{len(args)} cells:", [(a[0].split("/")[-2] + "/" + a[0].split("/")[-1], a[1])
                                  for a in args], flush=True)
    outs = list(striatum_ckpt.starmap(args, return_exceptions=True))
    for o in outs:
        print(json.dumps(o, default=str)[:400], flush=True)
    return [str(o)[:400] for o in outs]
