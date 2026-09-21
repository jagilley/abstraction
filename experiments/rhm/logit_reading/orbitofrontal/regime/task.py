"""regime/task.py -- the value-side battery in a world where a violation predicts cost.

Everything below is `striatum/task.py`'s machinery on a different world. The trunk is a
next-token model trained on the bursty stream (`train.py`); the actor is the same 16-way
level-l query head on the frozen trunk, trained on CLEAN windows and frozen; the critic is
the same streamed-Gram closed-form ridge `V[l, d](s_t) ~ E[o_l(t+d)]` at every horizon
d = 0..12, fitted on realised outcomes only, on windows of the bursty world.

WHAT IS DIFFERENT. A corruption does not change the generating tree, so the striatum
labels "structural consequence" and "realised damage" collapse into one object, read
against the NATURAL TWIN: the same window with THIS corruption undone and every other
token (including any other corruption) left alone, so the prefix is bit-identical and the
damage `o_nat(t+a) - o_x(t+a)` is what this corruption did at the goal. Beside it, the
phasic LEGAL TWIN: the same window with the corrupted token replaced by a legal token of
nearest model surprisal, which separates illegal from surprising.

WHAT IS NEW. Every event carries the exact regime filter of `world.py`: the belief
`b_fwd` about the regime one step ahead, its revision `db` at this token, the same in
log-odds (`dlogbf`), and the token's own evidence `llr`. The question is whether the
critic's revision `R[l, a] = V[l, a](s_t) - V[l, a+1](s_{t-1})` tracks them, and whether
it tracks the PROBABILITY revision (which shrinks deep inside a stretch already believed
noisy) or the evidence (which does not).

Events: every position whose token the corruption changed, plus one QUIET control per
window at a random position with no corruption in the preceding 8 tokens.

Run:
  modal run -m rhm.logit_reading.orbitofrontal.regime.task::regime_ckpt \
      --ckpt /data/.../traj_regime_burst_smoke/step000300.pt --world burst --tag smoke
  modal run --detach -m rhm.logit_reading.orbitofrontal.regime.task::regime_sweep
"""

import json
import os
import resource
import time

import numpy as np

from rhm.shared import volume, DATA_DIR, NumpyEncoder
from rhm.logit_reading.calibration import load_trajectory_ckpt, tb_key
from rhm.logit_reading.violation import auc
from rhm.logit_reading.striatum.task import (train_actor, actor_apply, ridge_solve, _r2)
from rhm.logit_reading.orbitofrontal.regime.world import app, world_params, world_build
from rhm.logit_reading.orbitofrontal.regime.train import train_regime, CKPTS

LEVELS = [1, 2, 3, 4, 5, 6]
DMAX = 12
A_LIST = [0, 1, 2, 4, 8]
OFFS = list(range(0, 9))          # twin persistence offsets
T_LO = 8
LAMS = [1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
BLOCKS = ["post_embed", "post_block5", "post_block7"]
CRIT_BLOCK = "post_block7"


def _corr(x, y):
    x = np.asarray(x, np.float64); y = np.asarray(y, np.float64)
    if len(x) < 10 or x.std() < 1e-12 or y.std() < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _spear(x, y):
    from scipy.stats import rankdata
    if len(x) < 10:
        return None
    return _corr(rankdata(x), rankdata(y))


def _slope(x, y):
    x = np.asarray(x, np.float64); y = np.asarray(y, np.float64)
    if len(x) < 10 or x.std() < 1e-12:
        return None
    return float(np.polyfit(x, y, 1)[0])


# ---------------------------------------------------------------------------

@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=4 * 3600, memory=20480,
              max_containers=4)
def regime_ckpt(ckpt: str, world: str = "", venue_tag: str = "", n_clean: int = 6144,
                n_clean_val: int = 2048, n_clean_crit: int = 6144, seed: int = 11,
                actor_steps: int = 3000, eval_seed: int = 5150, tag: str = "",
                caliper: float = 2.0, max_windows: int = 0, quiet_frac: float = 0.25):
    import torch
    volume.reload()
    t00 = time.time()
    dev = "cuda"
    model, rules, rule_w, cfg = load_trajectory_ckpt(ckpt, dev)
    L, s, v = cfg["L"], cfg["s"], cfg["v"]
    T = model.block_size
    step = cfg.get("step")
    world = world or cfg.get("world", "burst")
    P = world_params(world)
    key = tb_key(cfg["v"], cfg["s"], cfg["L"], cfg["m"])
    ddir = f"{DATA_DIR}/{key}/logit_reading"
    W = np.load(f"{ddir}/regime_world_{world}" + (f"_{venue_tag}" if venue_tag else "") + ".npz")
    print(f"loaded {ckpt} step {step} world={world}", flush=True)

    Wx = W["windows"].astype(np.int64)
    Wc = W["windows_clean"].astype(np.int64)
    n_all = len(Wx)
    idx = np.arange(n_all)
    if max_windows and max_windows < n_all:
        idx = np.arange(max_windows)
    Wx, Wc = Wx[idx], Wc[idx]
    y = W["y"][:, idx, :T].astype(np.int64)                     # (6, n, T) clean-tree answers
    reg = W["regime"][idx]
    changed = W["changed"][idx].astype(bool)
    ptok_L = W["ptok_L"][idx]
    legal = W["legal"][idx].astype(bool)                        # (n, T1, v)
    FIL = {k: W[k][idx] for k in ("b_pre", "b_fwd", "db", "dlogbf", "logbf", "llr")}
    n = len(Wx)
    d = model.transformer.wte.weight.shape[1]

    rng = np.random.default_rng(seed + 1)
    perm = rng.permutation(n)
    n_tr, n_va = int(0.60 * n), int(0.15 * n)
    split = np.zeros(n, np.int8)
    split[perm[n_tr:n_tr + n_va]] = 1
    split[perm[n_tr + n_va:]] = 2
    is_tr, is_va, is_te = split == 0, split == 1, split == 2

    # ---- the actor, on clean windows only -----------------------------------
    from rhm.logit_reading.altitude.units import windows_with_parse
    from rhm.logit_reading.striatum.parse import clean_answers
    t0 = time.time()
    n_cl = n_clean + n_clean_val
    wins_c, phase_c, _, fs_c = windows_with_parse(rules, rule_w, n_cl, T + 1, eval_seed)
    y_c = clean_answers(fs_c, phase_c, T + 1, L, s, 3)[:, :, :T].astype(np.int64)
    Sc = {b: np.zeros((n_cl, T, d), np.float32) for b in BLOCKS}
    with torch.no_grad():
        for c0 in range(0, n_cl, 256):
            x = torch.as_tensor(wins_c[c0:c0 + 256, :T], device=dev)
            _, _, inter = model(x, return_intermediates=True)
            for b in BLOCKS:
                Sc[b][c0:c0 + 256] = inter[b].float().cpu().numpy()
    pos = np.arange(T_LO, T)
    actors, actor_tab = {}, {}
    for b in BLOCKS:
        Xtr = torch.as_tensor(Sc[b][:n_clean][:, pos].reshape(-1, d), device=dev)
        Xva = torch.as_tensor(Sc[b][n_clean:][:, pos].reshape(-1, d), device=dev)
        for l in LEVELS:
            ytr = torch.as_tensor(y_c[l - 1][:n_clean][:, pos].reshape(-1), device=dev)
            yva = torch.as_tensor(y_c[l - 1][n_clean:][:, pos].reshape(-1), device=dev)
            h = train_actor(Xtr, ytr, Xva, yva, steps=actor_steps, seed=l, device=dev)
            actors[(b, l)] = h
            actor_tab[f"{b}/l{l}"] = h["acc"]
        del Xtr, Xva
        torch.cuda.empty_cache()
    del Sc
    torch.cuda.empty_cache()
    print(f"actors {time.time() - t0:.1f}s  " +
          json.dumps({k: round(x, 4) for k, x in actor_tab.items() if "block7" in k}),
          flush=True)

    # ---- the event table ----------------------------------------------------
    T_HI = T - 1
    posm = np.zeros((n, T), bool)
    posm[:, T_LO:T_HI + 1] = True
    ev_mask = changed[:, :T] & posm
    ew, et = np.where(ev_mask)
    # quiet controls: one per window, no corruption in [t-8, t]
    recent = np.zeros((n, T), np.int32)
    for k in range(0, 9):
        recent[:, k:] += changed[:, :T - k].astype(np.int32)
    quiet_ok = (recent == 0) & posm
    qrng = np.random.default_rng(seed + 5)
    qw, qt = [], []
    take = qrng.random(n) < quiet_frac
    for w in range(n):
        if not take[w]:
            continue
        c = np.where(quiet_ok[w])[0]
        if len(c):
            qw.append(w)
            qt.append(int(qrng.choice(c)))
    qw, qt = np.array(qw, np.int64), np.array(qt, np.int64)
    ev_w = np.concatenate([ew, qw])
    ev_t = np.concatenate([et, qt])
    ev_kind = np.concatenate([np.ones(len(ew), np.int8), np.zeros(len(qw), np.int8)])
    order = np.argsort(ev_w, kind="stable")
    ev_w, ev_t, ev_kind = ev_w[order], ev_t[order], ev_kind[order]
    n_ev = len(ev_w)
    # a legal prefix means the exact p_L is defined at t (no earlier violation in the window)
    bad = (ptok_L[:, 1:T] == 0)
    firstbad = np.where(bad.any(1), bad.argmax(1) + 1, T)
    ev_legal_prefix = ev_t <= firstbad[ev_w]
    ev_illegal = (ptok_L[ev_w, ev_t] == 0)
    print(f"events: {int((ev_kind == 1).sum())} corrupted + {int((ev_kind == 0).sum())} quiet; "
          f"illegal {float(ev_illegal[ev_kind == 1].mean()):.3f}; "
          f"legal-prefix {float(ev_legal_prefix[ev_kind == 1].mean()):.3f}", flush=True)

    # ---- pass X: outcomes, logits, Gram, event states ------------------------
    tgt = [f"l{l}_d{dd}" for l in LEVELS for dd in range(DMAX + 1)]
    tgt += [f"sh_l{l}_d{dd}" for l in LEVELS for dd in range(DMAX + 1)]
    n_tgt = len(tgt)
    A = {b: torch.zeros(d + 1, d + 1, dtype=torch.float64, device=dev) for b in BLOCKS}
    C = {b: torch.zeros(d + 1, n_tgt, dtype=torch.float64, device=dev) for b in BLOCKS}
    rows = np.arange(T_LO, T - 1 - DMAX + 1)
    sub_va = np.where(is_va)[0][:2500]
    sub_map = -np.ones(n, np.int64)
    sub_map[sub_va] = np.arange(len(sub_va))
    full_va = {b: np.zeros((len(sub_va), T, d), np.float32) for b in BLOCKS}
    o_x = np.zeros((6, n, T), np.uint8)
    nll = np.zeros((n, T), np.float32)
    Hq = np.zeros((n, T), np.float32)
    logq_pre = np.zeros((n_ev, v), np.float32)          # model logits at t-1, for the twins
    ev_st = np.zeros((n_ev, len(OFFS) + 1, d), np.float32)     # post_block7 at t-1,t..t+8
    ev_st_emb = np.zeros((n_ev, 2, d), np.float32)             # post_embed at t-1, t
    lo_ev = np.searchsorted(ev_w, np.arange(n + 1))
    g = torch.Generator().manual_seed(seed + 3)
    t0 = time.time()
    with torch.no_grad():
        for c0 in range(0, n, 256):
            c1 = min(c0 + 256, n)
            sl = slice(c0, c1)
            x = torch.as_tensor(Wx[sl, :T], device=dev)
            lg, _, inter = model(x, return_intermediates=True)
            lsm = torch.log_softmax(lg.float(), -1)
            nxt = torch.as_tensor(Wx[sl, 1:T + 1], device=dev)
            nll[sl] = (-lsm.gather(-1, nxt[..., None])[..., 0]).cpu().numpy()
            Hq[sl] = (-(lsm.exp() * lsm).sum(-1)).cpu().numpy()
            Xp = inter[CRIT_BLOCK].float()
            for li, l in enumerate(LEVELS):
                pred = actor_apply(actors[(CRIT_BLOCK, l)], Xp)
                o_x[li, sl] = (pred == torch.as_tensor(y[li][sl], device=dev)
                               ).to(torch.uint8).cpu().numpy()
            e0, e1 = lo_ev[c0], lo_ev[c1]
            if e1 > e0:
                ei = np.arange(e0, e1)
                rl = torch.as_tensor(ev_w[ei] - c0, device=dev)
                tt = ev_t[ei]
                logq_pre[ei] = lsm[rl, torch.as_tensor(tt - 1, device=dev)].cpu().numpy()
                offs = [-1] + OFFS
                tix = torch.as_tensor(np.clip(tt[:, None] + np.array(offs)[None, :], 0, T - 1),
                                      device=dev)
                ev_st[ei] = inter[CRIT_BLOCK][rl[:, None], tix].float().cpu().numpy()
                ev_st_emb[ei] = inter["post_embed"][rl[:, None], tix[:, :2]].float().cpu().numpy()
            vk = np.where(sub_map[c0:c1] >= 0)[0]
            if len(vk):
                vt = torch.as_tensor(vk, device=dev)
                for b in BLOCKS:
                    full_va[b][sub_map[c0 + vk]] = inter[b][vt].float().cpu().numpy()
            keep = torch.as_tensor(np.where(is_tr[c0:c1])[0], device=dev)
            if len(keep):
                ob = torch.as_tensor(o_x[:, sl], dtype=torch.float64, device=dev)
                shk = keep[torch.randperm(len(keep), generator=g).to(dev)]
                rw = torch.as_tensor(rows, device=dev)
                ys = []
                for src in (keep, shk):
                    for li in range(6):
                        for dd in range(DMAX + 1):
                            ys.append(ob[li][src][:, rw + dd])
                Yb = torch.stack(ys, -1).reshape(-1, n_tgt)
                for b in BLOCKS:
                    Xb = inter[b][keep][:, rw].double().reshape(-1, d)
                    Xb = torch.cat([Xb, torch.ones(len(Xb), 1, dtype=torch.float64,
                                                   device=dev)], 1)
                    A[b] += Xb.T @ Xb
                    C[b] += Xb.T @ Yb
    print(f"pass X {time.time() - t0:.1f}s", flush=True)

    # ---- solve the critic ----------------------------------------------------
    An = {b: A[b].cpu().numpy() for b in BLOCKS}
    Cn = {b: C[b].cpu().numpy() for b in BLOCKS}
    del A, C
    torch.cuda.empty_cache()
    Yv = np.zeros((len(sub_va) * len(rows), n_tgt), np.float32)
    for ti, nm in enumerate(tgt):
        base = nm[3:] if nm.startswith("sh_") else nm
        l = int(base.split("l")[1].split("_")[0]); dd = int(base.split("_d")[1])
        Yv[:, ti] = o_x[LEVELS.index(l)][sub_va][:, rows + dd].reshape(-1)
    yvar = Yv.astype(np.float64).var(0)
    beta, val_r2 = {}, {}
    for b in BLOCKS:
        Xv = np.concatenate([full_va[b][:, rows].reshape(-1, d).astype(np.float64),
                             np.ones((len(sub_va) * len(rows), 1))], 1)
        best = np.full(n_tgt, -np.inf)
        for lm in LAMS:
            B = ridge_solve(An[b], Cn[b], lm)
            r2 = 1.0 - ((Xv @ B - Yv) ** 2).mean(0) / np.maximum(yvar, 1e-12)
            for ti in np.where(r2 > best)[0]:
                best[ti] = r2[ti]
                beta[(b, tgt[ti])] = B[:, ti].copy()
        for ti, nm in enumerate(tgt):
            val_r2[(b, nm)] = float(best[ti])
        del Xv
    print(f"critic solved; {CRIT_BLOCK} l2_d0 R2 {val_r2[(CRIT_BLOCK, 'l2_d0')]:.4f} "
          f"(shuffled {val_r2[(CRIT_BLOCK, 'sh_l2_d0')]:.4f})", flush=True)

    # ---- the clean-only critic (never saw a corruption) ----------------------
    t0 = time.time()
    real = [f"l{l}_d{dd}" for l in LEVELS for dd in range(DMAX + 1)]
    wins_cc, phase_cc, _, fs_cc = windows_with_parse(rules, rule_w, n_clean_crit, T + 1,
                                                     eval_seed + 1000)
    y_cc = clean_answers(fs_cc, phase_cc, T + 1, L, s, 3)[:, :, :T].astype(np.int64)
    n_cc_va = min(1500, n_clean_crit // 5)
    n_cc_tr = n_clean_crit - n_cc_va
    Acc = torch.zeros(d + 1, d + 1, dtype=torch.float64, device=dev)
    Ccc = torch.zeros(d + 1, len(real), dtype=torch.float64, device=dev)
    o_cc = np.zeros((6, n_clean_crit, T), np.uint8)
    Xcc = np.zeros((n_cc_va, T, d), np.float32)
    with torch.no_grad():
        for c0 in range(0, n_clean_crit, 256):
            c1 = min(c0 + 256, n_clean_crit)
            sl = slice(c0, c1)
            x = torch.as_tensor(wins_cc[sl, :T], device=dev)
            _, _, inter = model(x, return_intermediates=True)
            Xp = inter[CRIT_BLOCK].float()
            for li, l in enumerate(LEVELS):
                pred = actor_apply(actors[(CRIT_BLOCK, l)], Xp)
                o_cc[li, sl] = (pred == torch.as_tensor(y_cc[li][sl], device=dev)
                                ).to(torch.uint8).cpu().numpy()
            kk = np.where(np.arange(c0, c1) < n_cc_tr)[0]
            if len(kk):
                keep = torch.as_tensor(kk, device=dev)
                ob = torch.as_tensor(o_cc[:, sl], dtype=torch.float64, device=dev)
                rw = torch.as_tensor(rows, device=dev)
                Yb = torch.stack([ob[li][keep][:, rw + dd] for li in range(6)
                                  for dd in range(DMAX + 1)], -1).reshape(-1, len(real))
                Xb = inter[CRIT_BLOCK][keep][:, rw].double().reshape(-1, d)
                Xb = torch.cat([Xb, torch.ones(len(Xb), 1, dtype=torch.float64, device=dev)], 1)
                Acc += Xb.T @ Xb
                Ccc += Xb.T @ Yb
            vk = np.where(np.arange(c0, c1) >= n_cc_tr)[0]
            if len(vk):
                Xcc[np.arange(c0, c1)[vk] - n_cc_tr] = \
                    inter[CRIT_BLOCK][torch.as_tensor(vk, device=dev)].float().cpu().numpy()
    Acn, Ccn = Acc.cpu().numpy(), Ccc.cpu().numpy()
    del Acc, Ccc
    torch.cuda.empty_cache()
    Xvc = np.concatenate([Xcc[:, rows].reshape(-1, d).astype(np.float64),
                          np.ones((n_cc_va * len(rows), 1))], 1)
    Yvc = np.stack([o_cc[LEVELS.index(int(nm.split("l")[1].split("_")[0]))]
                    [n_cc_tr:][:, rows + int(nm.split("_d")[1])].reshape(-1)
                    for nm in real], 1).astype(np.float64)
    bestc = np.full(len(real), -np.inf)
    clean_r2 = {}
    for lm in LAMS:
        B = ridge_solve(Acn, Ccn, lm)
        r2 = 1.0 - ((Xvc @ B - Yvc) ** 2).mean(0) / np.maximum(Yvc.var(0), 1e-12)
        for ti in np.where(r2 > bestc)[0]:
            bestc[ti] = r2[ti]
            beta[("clean", real[ti])] = B[:, ti].copy()
    for ti, nm in enumerate(real):
        clean_r2[nm] = float(bestc[ti])
    del Xvc, Yvc, Xcc
    print(f"clean-only critic {time.time() - t0:.1f}s  l2_d0 {clean_r2['l2_d0']:.4f}", flush=True)

    # ---- the twin windows ----------------------------------------------------
    corr_ev = np.where(ev_kind == 1)[0]
    W_nat = Wx[ev_w[corr_ev]].copy()
    W_nat[np.arange(len(corr_ev)), ev_t[corr_ev]] = Wc[ev_w[corr_ev], ev_t[corr_ev]]
    # the phasic legal twin: a LEGAL token of nearest model surprisal, legal prefix only
    cand_ok = np.where(ev_illegal & ev_legal_prefix & (ev_kind == 1))[0]
    lg_mask = legal[ev_w[cand_ok], ev_t[cand_ok]]                       # (m, v)
    s_all = -logq_pre[cand_ok]                                           # model surprisal
    s_obs = s_all[np.arange(len(cand_ok)), Wx[ev_w[cand_ok], ev_t[cand_ok]]]
    dist = np.where(lg_mask, np.abs(s_all - s_obs[:, None]), np.inf)
    pick = dist.argmin(1)
    dmin = dist[np.arange(len(cand_ok)), pick]
    has = dmin <= caliper
    leg_ev = cand_ok[has]
    leg_tok = pick[has]
    W_leg = Wx[ev_w[leg_ev]].copy()
    W_leg[np.arange(len(leg_ev)), ev_t[leg_ev]] = leg_tok
    leg_ds = (s_all[has, pick[has]] - s_obs[has])
    print(f"twins: natural {len(corr_ev)}, legal {len(leg_ev)}/{len(cand_ok)} "
          f"at caliper {caliper} (mean |ds| {np.abs(leg_ds).mean():.3f})", flush=True)

    def twin_pass(Wt, evs):
        """states at t..t+8 (crit block) and outcomes at t..t+8, for one window per event."""
        st = np.zeros((len(evs), len(OFFS), d), np.float32)
        oo = np.zeros((6, len(evs), len(OFFS)), np.uint8)
        with torch.no_grad():
            for c0 in range(0, len(evs), 256):
                c1 = min(c0 + 256, len(evs))
                x = torch.as_tensor(Wt[c0:c1, :T], device=dev)
                _, _, inter = model(x, return_intermediates=True)
                Xp = inter[CRIT_BLOCK].float()
                tt = ev_t[evs[c0:c1]]
                tix = np.clip(tt[:, None] + np.array(OFFS)[None, :], 0, T - 1)
                tix_t = torch.as_tensor(tix, device=dev)
                rl = torch.arange(c1 - c0, device=dev)[:, None]
                st[c0:c1] = inter[CRIT_BLOCK][rl, tix_t].float().cpu().numpy()
                for li, l in enumerate(LEVELS):
                    pred = actor_apply(actors[(CRIT_BLOCK, l)], Xp).cpu().numpy()
                    tru = y[li][ev_w[evs[c0:c1]]]
                    oo[li, c0:c1] = (np.take_along_axis(pred, tix, 1)
                                     == np.take_along_axis(tru, tix, 1)).astype(np.uint8)
        return st, oo

    t0 = time.time()
    nat_st, nat_o = twin_pass(W_nat, corr_ev)
    leg_st, _ = twin_pass(W_leg, leg_ev)
    print(f"twin passes {time.time() - t0:.1f}s", flush=True)

    # ---- assemble the per-event table ---------------------------------------
    def pred_lin(bsel, nm, X):
        b = beta[(bsel, nm)]
        return X @ b[:-1] + b[-1]

    R = {"w": ev_w, "t": ev_t, "kind": ev_kind, "split": split[ev_w],
         "regime": reg[ev_w, ev_t].astype(np.int8),
         "regime_next": reg[ev_w, np.clip(ev_t + 1, 0, T)].astype(np.int8),
         "regime_ahead": np.array([reg[w, t + 1:min(t + 13, T + 1)].mean()
                                   for w, t in zip(ev_w, ev_t)], np.float32),
         "n_corrupt_ahead": np.array([changed[w, t + 1:min(t + 13, T)].sum()
                                      for w, t in zip(ev_w, ev_t)], np.float32),
         "illegal": ev_illegal.astype(np.int8),
         "legal_prefix": ev_legal_prefix.astype(np.int8),
         "nll": nll[ev_w, ev_t - 1], "Hq_pre": Hq[ev_w, ev_t - 1],
         "dH": Hq[ev_w, ev_t] - Hq[ev_w, ev_t - 1],
         "ptok_L": ptok_L[ev_w, ev_t],
         "tmax_ok": ((ev_t + max(OFFS)) <= T - 1).astype(np.int8)}
    for k in ("b_pre", "b_fwd", "db", "dlogbf", "logbf", "llr"):
        R[k] = FIL[k][ev_w, ev_t].astype(np.float32)
    R["b_fwd_pre"] = FIL["b_fwd"][ev_w, np.maximum(ev_t - 1, 0)].astype(np.float32)
    Xa, Xb_ = ev_st[:, 1], ev_st[:, 0]                     # state at t and at t-1
    Xa_e, Xb_e = ev_st_emb[:, 1], ev_st_emb[:, 0]
    natmap = -np.ones(n_ev, np.int64); natmap[corr_ev] = np.arange(len(corr_ev))
    legmap = -np.ones(n_ev, np.int64); legmap[leg_ev] = np.arange(len(leg_ev))
    R["has_nat"] = (natmap >= 0).astype(np.int8)
    R["has_leg"] = (legmap >= 0).astype(np.int8)
    R["leg_ds"] = np.zeros(n_ev, np.float32)
    R["leg_ds"][leg_ev] = leg_ds
    R["leg_tok"] = np.zeros(n_ev, np.int16)
    R["leg_tok"][leg_ev] = leg_tok
    R["tok"] = Wx[ev_w, ev_t].astype(np.int16)
    R["tok_clean"] = Wc[ev_w, ev_t].astype(np.int16)
    for l in LEVELS:
        li = LEVELS.index(l)
        for a in A_LIST:
            n0, n1 = f"l{l}_d{a}", f"l{l}_d{a + 1}"
            Vp = pred_lin(CRIT_BLOCK, n1, Xb_)
            Va = pred_lin(CRIT_BLOCK, n0, Xa)
            R[f"V_l{l}_a{a}"] = Va.astype(np.float32)
            R[f"Vpre_l{l}_a{a}"] = Vp.astype(np.float32)
            R[f"R_l{l}_a{a}"] = (Va - Vp).astype(np.float32)
            R[f"Rsh_l{l}_a{a}"] = (pred_lin(CRIT_BLOCK, f"sh_{n0}", Xa)
                                   - pred_lin(CRIT_BLOCK, f"sh_{n1}", Xb_)).astype(np.float32)
            R[f"Rclean_l{l}_a{a}"] = (pred_lin("clean", n0, Xa)
                                      - pred_lin("clean", n1, Xb_)).astype(np.float32)
            R[f"Remb_l{l}_a{a}"] = (pred_lin("post_embed", n0, Xa_e)
                                    - pred_lin("post_embed", n1, Xb_e)).astype(np.float32)
            vn = np.full(n_ev, np.nan, np.float32)
            if a < len(OFFS):
                vn[corr_ev] = pred_lin(CRIT_BLOCK, n0, nat_st[:, a])
            R[f"Vnat_l{l}_a{a}"] = vn
            vl = np.full(n_ev, np.nan, np.float32)
            if a < len(OFFS) and len(leg_ev):
                vl[leg_ev] = pred_lin(CRIT_BLOCK, n0, leg_st[:, a])
            R[f"Vleg_l{l}_a{a}"] = vl
            tq = np.clip(ev_t + a, 0, T - 1)
            R[f"ox_l{l}_a{a}"] = o_x[li][ev_w, tq]
            on = np.full(n_ev, 255, np.uint8)
            if a < len(OFFS):
                on[corr_ev] = nat_o[li][:, a]
            R[f"onat_l{l}_a{a}"] = on
            R[f"ok_l{l}_a{a}"] = ((ev_t + a) <= T - 1).astype(np.int8)
    # persistence of the natural twin's value advantage
    for l in (1, 2, 3, 4):
        for tau in OFFS:
            dv = np.full(n_ev, np.nan, np.float32)
            dv[corr_ev] = (pred_lin(CRIT_BLOCK, f"l{l}_d0", ev_st[corr_ev, 1 + tau])
                           - pred_lin(CRIT_BLOCK, f"l{l}_d0", nat_st[:, tau]))
            R[f"dVnat_l{l}_t{tau}"] = dv
            if len(leg_ev):
                dl = np.full(n_ev, np.nan, np.float32)
                dl[leg_ev] = (pred_lin(CRIT_BLOCK, f"l{l}_d0", ev_st[leg_ev, 1 + tau])
                              - pred_lin(CRIT_BLOCK, f"l{l}_d0", leg_st[:, tau]))
                R[f"dVleg_l{l}_t{tau}"] = dl

    # ---- oracle probes on the SAME event states ------------------------------
    tr_m, va_m = R["split"] == 0, R["split"] == 1
    Xp_ = Xa.astype(np.float64)
    Xtrb = np.concatenate([Xp_[tr_m], np.ones((int(tr_m.sum()), 1))], 1)
    Xvab = np.concatenate([Xp_[va_m], np.ones((int(va_m.sum()), 1))], 1)
    Ag = Xtrb.T @ Xtrb
    labs = {"probe_regime": R["regime"].astype(np.float64),
            "probe_bfwd": R["b_fwd"].astype(np.float64),
            "probe_illegal": R["illegal"].astype(np.float64),
            "probe_corrupt": (R["kind"] == 1).astype(np.float64),
            "probe_ahead": R["n_corrupt_ahead"].astype(np.float64)}
    for l in (1, 2, 3, 4):
        dm = np.where(R["has_nat"] == 1,
                      R[f"onat_l{l}_a0"].astype(np.float64) - R[f"ox_l{l}_a0"].astype(np.float64),
                      0.0)
        labs[f"probe_dmg_l{l}"] = dm
    names = [k for k, vv in labs.items() if vv[tr_m].std() > 1e-9]
    if names and tr_m.sum() > 50:
        Yl = np.stack([labs[k][tr_m] for k in names], 1)
        Ylv = np.stack([labs[k][va_m] for k in names], 1)
        Cg = Xtrb.T @ Yl
        best = np.full(len(names), -np.inf)
        Bb = np.zeros((Xp_.shape[1] + 1, len(names)))
        for lm in LAMS:
            B = ridge_solve(Ag, Cg, lm)
            r2 = 1.0 - ((Xvab @ B - Ylv) ** 2).mean(0) / np.maximum(Ylv.var(0), 1e-12)
            for ti in np.where(r2 > best)[0]:
                best[ti] = r2[ti]
                Bb[:, ti] = B[:, ti]
        pr = np.concatenate([Xp_, np.ones((len(Xp_), 1))], 1) @ Bb
        for ti, k in enumerate(names):
            R[k] = pr[:, ti].astype(np.float32)

    # ---- the regime in the state, at every position (not just events) --------
    probe_pos = {}
    Xs = full_va[CRIT_BLOCK][:, rows].reshape(-1, d).astype(np.float64)
    reg_s = reg[sub_va][:, rows].reshape(-1).astype(np.float64)
    bf_s = FIL["b_fwd"][sub_va][:, rows].reshape(-1).astype(np.float64)
    half = len(Xs) // 2
    for bsel, Xsrc in ((CRIT_BLOCK, Xs),
                       ("post_embed", full_va["post_embed"][:, rows]
                        .reshape(-1, d).astype(np.float64))):
        Xb1 = np.concatenate([Xsrc[:half], np.ones((half, 1))], 1)
        Xb2 = np.concatenate([Xsrc[half:], np.ones((len(Xsrc) - half, 1))], 1)
        Y1 = np.stack([reg_s[:half], bf_s[:half]], 1)
        Y2 = np.stack([reg_s[half:], bf_s[half:]], 1)
        if Y1[:, 0].std() < 1e-9:
            probe_pos[bsel] = {"regime_auc": None, "bfwd_r2": None}
            continue
        Agp = Xb1.T @ Xb1
        Cgp = Xb1.T @ Y1
        bestp, Bp = np.full(2, -np.inf), np.zeros((d + 1, 2))
        for lm in LAMS:
            B = ridge_solve(Agp, Cgp, lm)
            r2 = 1.0 - ((Xb2 @ B - Y2) ** 2).mean(0) / np.maximum(Y2.var(0), 1e-12)
            for ti in np.where(r2 > bestp)[0]:
                bestp[ti] = r2[ti]
                Bp[:, ti] = B[:, ti]
        pr = Xb2 @ Bp
        probe_pos[bsel] = {"regime_auc": auc(pr[:, 0], Y2[:, 0] > 0.5),
                           "regime_r2": float(bestp[0]), "bfwd_r2": float(bestp[1]),
                           "bfwd_corr": _corr(pr[:, 1], Y2[:, 1])}

    # ---- the value trace around a regime switch (test windows) ---------------
    trace = {}
    if (reg == 1).any():
        tv = np.zeros((len(sub_va), T, 6), np.float32)
        for li, l in enumerate(LEVELS):
            bb = beta[(CRIT_BLOCK, f"l{l}_d0")]
            tv[:, :, li] = full_va[CRIT_BLOCK] @ bb[:-1] + bb[-1]
        sw_on, sw_off = [], []
        rr = reg[sub_va]
        for wi in range(len(sub_va)):
            ch = np.where(rr[wi, 1:T] != rr[wi, :T - 1])[0] + 1
            for c in ch:
                if c < T_LO + 6 or c > T - 10:
                    continue
                (sw_on if rr[wi, c] == 1 else sw_off).append((wi, c))
        TAUS = list(range(-5, 10))
        for nm, lst in (("on", sw_on), ("off", sw_off)):
            if len(lst) < 20:
                continue
            wi = np.array([a for a, _ in lst]); cc = np.array([b for _, b in lst])
            ent = {"n": len(lst), "taus": TAUS}
            for l in LEVELS:
                li = LEVELS.index(l)
                ent[f"V_l{l}"] = [float(tv[wi, np.clip(cc + t_, 0, T - 1), li].mean())
                                  for t_ in TAUS]
                bb = beta[(CRIT_BLOCK, f"l{l}_d1")]
                Vp = full_va[CRIT_BLOCK] @ bb[:-1] + bb[-1]
                ent[f"surprise_l{l}"] = [
                    float((o_x[li][sub_va][wi, np.clip(cc + t_, 0, T - 1)].astype(np.float64)
                           - Vp[wi, np.clip(cc + t_ - 1, 0, T - 1)]).mean()) for t_ in TAUS]
                ent[f"acc_l{l}"] = [
                    float(o_x[li][sub_va][wi, np.clip(cc + t_, 0, T - 1)].mean()) for t_ in TAUS]
            trace[nm] = ent

    # ---- summary -------------------------------------------------------------
    res = {"ckpt": ckpt, "step": step, "world": world, "world_params": P, "n": n,
           "n_events": int(n_ev), "n_corrupt": int((ev_kind == 1).sum()),
           "n_quiet": int((ev_kind == 0).sum()),
           "actor_clean_acc": actor_tab,
           "critic_val_r2": {f"{b}/{nm}": val_r2[(b, nm)] for b in BLOCKS for nm in tgt},
           "clean_critic_val_r2": clean_r2,
           "probe_pos": probe_pos, "trace": trace,
           "twins": {"n_nat": int(len(corr_ev)), "n_leg": int(len(leg_ev)),
                     "n_leg_cand": int(len(cand_ok)), "caliper": caliper,
                     "mean_abs_ds": float(np.abs(leg_ds).mean()) if len(leg_ev) else None,
                     "leg_eq_clean": float((leg_tok == Wc[ev_w[leg_ev], ev_t[leg_ev]]).mean())
                     if len(leg_ev) else None},
           "census": {"illegal_rate": float(ev_illegal[ev_kind == 1].mean()),
                      "legal_prefix_rate": float(ev_legal_prefix[ev_kind == 1].mean()),
                      "n_legal_prefix": int(((ev_kind == 1) & ev_legal_prefix).sum()),
                      "illegal_rate_lp": float(ev_illegal[(ev_kind == 1) & ev_legal_prefix].mean()),
                      "dH_corrupt": float(R["dH"][ev_kind == 1].mean()),
                      "dH_quiet": float(R["dH"][ev_kind == 0].mean()),
                      "nll_corrupt": float(R["nll"][ev_kind == 1].mean()),
                      "nll_quiet": float(R["nll"][ev_kind == 0].mean()),
                      "regime_at_event": float(R["regime"][ev_kind == 1].mean()),
                      "b_fwd_pre_mean_corrupt": float(R["b_fwd_pre"][ev_kind == 1].mean()),
                      "b_fwd_pre_mean_quiet": float(R["b_fwd_pre"][ev_kind == 0].mean()),
                      "db_mean_corrupt": float(R["db"][ev_kind == 1].mean()),
                      "db_mean_quiet": float(R["db"][ev_kind == 0].mean())},
           "config": {"levels": LEVELS, "dmax": DMAX, "a_list": A_LIST, "offs": OFFS,
                      "quiet_frac": quiet_frac, "caliper_build": caliper,
                      "t_lo": T_LO, "blocks": BLOCKS, "crit_block": CRIT_BLOCK,
                      "n_clean": n_clean, "n_clean_crit": n_clean_crit, "seed": seed,
                      "eval_seed": eval_seed, "split": [0.6, 0.15, 0.25]}}
    res["headline"] = headline(R)

    stem = ckpt[:-3]
    sfx = f"_regime_{world}" + (f"_{tag}" if tag else "")
    with open(f"{stem}{sfx}.json", "w") as f:
        json.dump(res, f, cls=NumpyEncoder, indent=1)
    np.savez_compressed(f"{stem}{sfx}.npz",
                        **{k: (vv.astype(np.float32) if np.asarray(vv).dtype == np.float64
                               else np.asarray(vv)) for k, vv in R.items()})
    volume.commit()
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    print(f"saved -> {stem}{sfx}.json  ({time.time() - t00:.0f}s, peak RSS {rss:.1f} GB)",
          flush=True)
    print(json.dumps(res["headline"], indent=1, cls=NumpyEncoder), flush=True)
    return {"ckpt": ckpt, "step": step, "world": world, "headline": res["headline"]}


def headline(R):
    """The three numbers that say whether the cell is legible, computed on test events."""
    te = (R["split"] == 2)
    cm = te & (R["kind"] == 1)
    out = {}
    for l in (1, 2, 3):
        for a in (0, 2):
            k = f"l{l}_a{a}"
            if f"R_{k}" not in R:
                continue
            out[f"corr_R_db/{k}"] = _corr(R["db"][cm], R[f"R_{k}"][cm])
            out[f"corr_R_dlogbf/{k}"] = _corr(R["dlogbf"][cm], R[f"R_{k}"][cm])
            out[f"corr_Rsh_db/{k}"] = _corr(R["db"][cm], R[f"Rsh_{k}"][cm])
            out[f"mean_R_corrupt/{k}"] = float(R[f"R_{k}"][cm].mean())
            out[f"mean_R_quiet/{k}"] = float(R[f"R_{k}"][te & (R["kind"] == 0)].mean())
    for l in (1, 2, 3):
        out[f"corr_R_ahead/l{l}_a0"] = _corr(R["n_corrupt_ahead"][cm], R[f"R_l{l}_a0"][cm])
        out[f"corr_V_ahead/l{l}_a0"] = _corr(R["n_corrupt_ahead"][cm], R[f"V_l{l}_a0"][cm])
        out[f"corr_Rsh_ahead/l{l}_a0"] = _corr(R["n_corrupt_ahead"][cm], R[f"Rsh_l{l}_a0"][cm])
    out["auc_V_ahead2/l2"] = auc(-R["V_l2_a0"][te], R["n_corrupt_ahead"][te] >= 2)
    if "probe_regime" in R:
        out["probe_regime_auc_event"] = auc(R["probe_regime"][te], R["regime"][te] == 1)
    if "probe_ahead" in R:
        out["probe_ahead_auc2"] = auc(R["probe_ahead"][te], R["n_corrupt_ahead"][te] >= 2)
    out["V_auc_regime"] = auc(-R["V_l2_a0"][te], R["regime"][te] == 1)
    out["Vpre_auc_regime"] = auc(-R["Vpre_l2_a0"][te], R["regime"][te] == 1)
    return out


# ---------------------------------------------------------------------------

@app.function(volumes={DATA_DIR: volume}, timeout=8 * 3600, memory=2048)
def regime_sweep(worlds: str = "burst,iid", steps: str = "0,8000,64000", tag: str = "",
                 venue_tag: str = "", traj_tag: str = "", max_windows: int = 0,
                 n_clean: int = 6144):
    volume.reload()
    key = "v16_s2_L6_m4_distinct"
    ddir = f"{DATA_DIR}/{key}/logit_reading"
    args = []
    for w in worlds.split(","):
        tdir = f"{ddir}/traj_regime_{w}" + (f"_{traj_tag}" if traj_tag else "")
        for st in steps.split(","):
            c = f"{tdir}/step{int(st):06d}.pt"
            if os.path.exists(c):
                args.append((c, w, venue_tag, n_clean, 2048, 6144, 11, 3000, 5150, tag,
                             2.0, max_windows, 0.25))
            else:
                print(f"MISSING {c}", flush=True)
    print(f"{len(args)} cells", flush=True)
    outs = list(regime_ckpt.starmap(args, return_exceptions=True))
    for o in outs:
        print(json.dumps(o, default=str, cls=NumpyEncoder)[:700], flush=True)
    return [str(o)[:400] for o in outs]


@app.function(volumes={DATA_DIR: volume}, timeout=12 * 3600, memory=2048)
def regime_all(worlds: str = "burst,iid", n: int = 16384, steps: int = 64000,
               read_steps: str = "0,8000,64000", tag: str = "", venue_tag: str = "",
               traj_tag: str = "", chunk_seqs: int = 1_000_000, n_clean: int = 6144,
               ckpt_steps: str = "", max_windows: int = 0, skip_build: bool = False,
               skip_train: bool = False):
    """The whole node in one detached launch: wave 1 builds each world's venue and trains
    its trunk (one container each, in parallel); wave 2 runs the readout cell for every
    (world, checkpoint). `modal run --detach` returns before the run does -- wait on the
    artefacts."""
    volume.reload()
    ws = worlds.split(",")
    ck = ckpt_steps or CKPTS
    jobs = []
    if not skip_build:
        jobs += [("build", w) for w in ws]
    if not skip_train:
        jobs += [("train", w) for w in ws]
    print(f"wave 1: {jobs}", flush=True)
    handles = []
    for kind, w in jobs:
        if kind == "build":
            handles.append(("build", w, world_build.spawn(w, 16, 2, 6, 4, 0, 1.0, 1, n,
                                                          3131, venue_tag)))
        else:
            handles.append(("train", w, train_regime.spawn(
                w, 16, 2, 6, 4, 0, 1.0, 1, 8, 8, 256, steps, 64, 3e-4, 0.01, chunk_seqs,
                7, 42, ck, 1000, traj_tag)))
    for kind, w, h in handles:
        try:
            print(f"wave 1 done {kind}/{w}: {str(h.get())[:300]}", flush=True)
        except Exception as e:
            print(f"wave 1 FAILED {kind}/{w}: {e!r}", flush=True)
    volume.reload()
    key = "v16_s2_L6_m4_distinct"
    ddir = f"{DATA_DIR}/{key}/logit_reading"
    args = []
    for w in ws:
        tdir = f"{ddir}/traj_regime_{w}" + (f"_{traj_tag}" if traj_tag else "")
        for st in read_steps.split(","):
            c = f"{tdir}/step{int(st):06d}.pt"
            if os.path.exists(c):
                args.append((c, w, venue_tag, n_clean, 2048, 6144, 11, 3000, 5150, tag,
                             2.0, max_windows, 0.25))
            else:
                print(f"MISSING {c}", flush=True)
    print(f"wave 2: {len(args)} readout cells", flush=True)
    outs = list(regime_ckpt.starmap(args, return_exceptions=True))
    for o in outs:
        print(json.dumps(o, default=str, cls=NumpyEncoder)[:900], flush=True)
    return [str(o)[:300] for o in outs]
