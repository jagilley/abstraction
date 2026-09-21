"""abstain, the stream venue: the commit-or-abstain choice at EVERY position of the stream.

`striatum/task.py` reads the critic at five query offsets after an anchor. That cannot say where
in the learner's life a value reading is worth anything, because it only ever looks just after a
violation. This variant keeps the same frozen trunk, the same frozen actor, the same critic recipe
and the same window split, and writes the critic's level `V[l, d](s_t)` and the realised outcome
`o_l(t + d)` at EVERY position `t` of held-out windows, on three arms:

  `edit`  -- the edited stimulus test windows (the event is inside them)
  `orig`  -- their unedited counterparts, position for position: the matched quiet control
  `quiet` -- fresh clean windows that were never part of any edit at all

with, per position, the distance since the violation, so the decision can be split by
on-event versus off-event and "where the value lives" becomes a number on the whole stream.

Reproduction gates printed by every cell: the actor's clean held-out accuracy against the banked
`striatum/` column, and the critic's held-out `R2` for `l1_d0` against the banked value. The
critic sees the same 60% train split of the same stimulus windows, so it is the same object.

**The log Bayes factor is not computed here, by construction.** `basalis/hold.py`'s `logbf` is
defined relative to a FLAGGED ANCHOR: it needs the `K = v` hypothetical continuations rolled
forward from that anchor (`P["ll"]`, one forward pass per branch per anchor) and the flagged
token's own surprisal as its zero. Per-position over a whole stream there is no single anchor, so
it would mean `v` branch passes at every one of the 44 positions of every window -- a ~700x
blow-up of this cell. It is skipped, and the skip is recorded in the artefact's `config`.

  cd experiments            # MODAL_PROFILE=chromatic
  modal run --detach -m rhm.logit_reading.orbitofrontal.abstain.task::abstain_sweep
"""

import json
import os
import resource
import time

import numpy as np

import modal

from rhm.shared import volume, DATA_DIR, NumpyEncoder
from rhm.logit_reading.calibration import load_trajectory_ckpt, tb_key
from rhm.logit_reading.striatum.task import (LEVELS, DMAX, T_LO, LAMS, train_actor,
                                             actor_apply, ridge_solve)


def _ignore(path):
    """Mount the Python sources only, and never a sibling node's `results/` tree -- another
    agent writing its launch log there makes the image build race and fail."""
    sp = str(path)
    if "/results/" in sp or sp.endswith("/results") or sp.endswith(".log"):
        return True
    return not sp.endswith(".py")


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==1.26.4", "scipy==1.16.3", "torch==2.7.0")
    .add_local_python_source("rhm", ignore=_ignore)
)
app = modal.App("rhm-abstain", image=image)

SLEV = [1, 2, 3, 4]                       # the levels the decision is read at
SHOR = [0, 1, 2, 3, 4, 5, 8, 9]           # V at `a` and at `a+1`, for a in {0,1,2,4,8}
BLOCKS = ["post_embed", "post_block7"]
D = "/data/v16_s2_L6_m4_distinct/logit_reading"


@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=2 * 3600, memory=16384,
              max_containers=2)
def abstain_stream(ckpt: str, stim_tag: str = "a1", n_clean: int = 6144,
                   n_clean_val: int = 2048, n_clean_crit: int = 8192, n_quiet: int = 3000,
                   max_test: int = 3000, seed: int = 11, eval_seed: int = 5150,
                   actor_steps: int = 3000, tag: str = "", prim_block: str = "post_block7"):
    import torch
    volume.reload()
    t00 = time.time()
    dev = "cuda"
    model, rules, rule_w, cfg = load_trajectory_ckpt(ckpt, dev)
    L, s, v = cfg["L"], cfg["s"], cfg["v"]
    T = model.block_size
    d = model.transformer.wte.weight.shape[1]
    step = cfg.get("step")
    print(f"loaded {ckpt} step {step}  T={T} d={d}", flush=True)

    key = tb_key(cfg["v"], cfg["s"], cfg["L"], cfg["m"])
    ddir = f"{DATA_DIR}/{key}/logit_reading"
    S = np.load(f"{ddir}/stimuli_{stim_tag}.npz")
    P = np.load(f"{ddir}/parse_{stim_tag}.npz")
    We, Wo = S["windows_edit"], S["windows_orig"]
    n = len(We)
    meta = {k: S[k] for k in ("phase", "etype", "j", "e", "first_diff", "t_v", "k_star")}
    y_edit = P["y_edit"][:, :, :T]
    y_orig = P["y_orig"][:, :, :T]
    print(f"stimuli {stim_tag}: n={n}", flush=True)

    # ---- the split, IDENTICAL to striatum's, so the test rows are the same windows ----
    rng = np.random.default_rng(seed + 1)
    perm = rng.permutation(n)
    n_tr, n_va = int(0.60 * n), int(0.15 * n)
    split = np.zeros(n, np.int8)
    split[perm[n_tr:n_tr + n_va]] = 1
    split[perm[n_tr + n_va:]] = 2
    is_tr, is_va, is_te = split == 0, split == 1, split == 2

    banked, bpath = None, ckpt[:-3] + "_coeruleus_head_state_excess_ridge_onnoise.pt"
    if not os.path.exists(bpath):
        bpath = ckpt[:-3] + "_coeruleus_head_onnoise.pt"
    if os.path.exists(bpath):
        from rhm.logit_reading.coeruleus.readout import load_head
        banked, _ = load_head(bpath, dev)
        print(f"banked coeruleus head: {bpath} block={banked.block}", flush=True)
    blocks = list(BLOCKS) + ([banked.block] if banked is not None
                             and banked.block not in BLOCKS else [])

    # ---- the actor: fresh clean windows, frozen (striatum's recipe and seeds) --------
    from rhm.logit_reading.altitude.units import windows_with_parse
    from rhm.logit_reading.striatum.parse import clean_answers
    t0 = time.time()
    n_cl = n_clean + n_clean_val
    wins_c, phase_c, _, fs_c = windows_with_parse(rules, rule_w, n_cl, T + 1, eval_seed)
    y_c = clean_answers(fs_c, phase_c, T + 1, L, s, 3)[:, :, :T]
    Sc = {b: np.zeros((n_cl, T, d), np.float32) for b in blocks}
    with torch.no_grad():
        for c0 in range(0, n_cl, 256):
            x = torch.as_tensor(wins_c[c0:c0 + 256, :T], device=dev)
            _, _, inter = model(x, return_intermediates=True)
            for b in blocks:
                Sc[b][c0:c0 + 256] = inter[b].float().cpu().numpy()
    pos = np.arange(T_LO, T)
    actors, actor_tab = {}, {}
    for b in blocks:
        Xtr = torch.as_tensor(Sc[b][:n_clean][:, pos].reshape(-1, d), device=dev)
        Xva = torch.as_tensor(Sc[b][n_clean:][:, pos].reshape(-1, d), device=dev)
        for l in LEVELS:
            ytr = torch.as_tensor(y_c[l - 1][:n_clean][:, pos].reshape(-1).astype(np.int64),
                                  device=dev)
            yva = torch.as_tensor(y_c[l - 1][n_clean:][:, pos].reshape(-1).astype(np.int64),
                                  device=dev)
            actors[(b, l)] = train_actor(Xtr, ytr, Xva, yva, steps=actor_steps, seed=l,
                                         device=dev)
            actor_tab[f"{b}/l{l}"] = actors[(b, l)]["acc"]
        del Xtr, Xva
        torch.cuda.empty_cache()
    prim = prim_block
    del Sc
    torch.cuda.empty_cache()
    print(f"actors {time.time() - t0:.1f}s", flush=True)
    print("ACTOR GATE " + json.dumps({k: round(x, 4) for k, x in actor_tab.items()
                                      if k.startswith(prim)}), flush=True)

    # ---- pass E: outcomes, surprisal, and the critic's Gram on the TRAIN rows -------
    tgt = [f"l{l}_d{dd}" for l in LEVELS for dd in range(DMAX + 1)]
    tgt += [f"sh_l{l}_d{dd}" for l in LEVELS for dd in range(DMAX + 1)]
    n_tgt = len(tgt)
    A = {b: torch.zeros(d + 1, d + 1, dtype=torch.float64, device=dev) for b in blocks}
    C = {b: torch.zeros(d + 1, n_tgt, dtype=torch.float64, device=dev) for b in blocks}
    T_HI = T - 1 - DMAX
    rows = np.arange(T_LO, T_HI + 1)
    sub_tr = np.where(is_tr)[0][:3000]
    sub_va = np.where(is_va)[0][:3000]
    keep_full = np.zeros(n, bool)
    keep_full[sub_tr] = True
    keep_full[sub_va] = True
    kf_idx = np.where(keep_full)[0]
    kf_map = -np.ones(n, np.int64)
    kf_map[kf_idx] = np.arange(len(kf_idx))

    def run(W, Y, gram, gseed=0, want_full=False):
        o = np.zeros((len(LEVELS), len(W), T), np.uint8)
        nll = np.zeros((len(W), T), np.float32)
        Hq = np.zeros((len(W), T), np.float32)
        full = ({b: np.zeros((len(kf_idx), T, d), np.float32) for b in blocks}
                if want_full else None)
        g = torch.Generator().manual_seed(gseed)
        with torch.no_grad():
            for c0 in range(0, len(W), 256):
                sl = slice(c0, min(c0 + 256, len(W)))
                x = torch.as_tensor(W[sl, :T], device=dev)
                lg, _, inter = model(x, return_intermediates=True)
                lsm = torch.log_softmax(lg.float(), -1)
                nxt = torch.as_tensor(W[sl, 1:T + 1], device=dev)
                nll[sl] = (-lsm.gather(-1, nxt[..., None])[..., 0]).cpu().numpy()
                Hq[sl] = (-(lsm.exp() * lsm).sum(-1)).cpu().numpy()
                Xp = inter[prim].float()
                for li, l in enumerate(LEVELS):
                    pred = actor_apply(actors[(prim, l)], Xp)
                    tru = torch.as_tensor(Y[li][sl].astype(np.int64), device=dev)
                    o[li, sl] = (pred == tru).to(torch.uint8).cpu().numpy()
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
                        ys = [ob[li][src][:, rw + dd] for src in (keep, shk)
                              for li in range(len(LEVELS)) for dd in range(DMAX + 1)]
                        Yb = torch.stack(ys, -1).reshape(-1, n_tgt)
                        for b in blocks:
                            Xb = inter[b][keep][:, rw].double().reshape(-1, d)
                            Xb = torch.cat([Xb, torch.ones(len(Xb), 1, dtype=torch.float64,
                                                           device=dev)], 1)
                            A[b] += Xb.T @ Xb
                            C[b] += Xb.T @ Yb
        return o, nll, Hq, full

    t0 = time.time()
    o_e, nll_e, H_e, full_e = run(We, y_edit, True, gseed=seed + 3, want_full=True)
    print(f"edit pass {time.time() - t0:.1f}s", flush=True)

    # ---- solve the critic (striatum's recipe: all targets, all lambdas) -------------
    An = {b: A[b].cpu().numpy() for b in blocks}
    Cn = {b: C[b].cpu().numpy() for b in blocks}
    del A, C
    torch.cuda.empty_cache()
    vr = kf_map[sub_va]
    Yv = np.zeros((len(sub_va) * len(rows), n_tgt), np.float32)
    for ti, nm in enumerate(tgt):
        base = nm[3:] if nm.startswith("sh_") else nm
        l = int(base.split("l")[1].split("_")[0])
        dd = int(base.split("_d")[1])
        Yv[:, ti] = o_e[LEVELS.index(l)][sub_va][:, rows + dd].reshape(-1)
    yvar = Yv.astype(np.float64).var(0)
    beta, val_r2 = {}, {}
    for b in blocks:
        Xv = np.concatenate([full_e[b][vr][:, rows].reshape(-1, d).astype(np.float64),
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
    cb = prim
    print(f"CRITIC GATE {cb}/l1_d0 val R2 = {val_r2[(cb, 'l1_d0')]:.4f}  "
          f"(banked striatum a1 64k: 0.2182)", flush=True)
    del full_e

    # ---- the clean-only critic: never saw an edit --------------------------------
    t0 = time.time()
    real = [f"l{l}_d{dd}" for l in LEVELS for dd in range(DMAX + 1)]
    wins_cc, phase_cc, _, fs_cc = windows_with_parse(rules, rule_w, n_clean_crit, T + 1,
                                                     eval_seed + 1000)
    y_cc = clean_answers(fs_cc, phase_cc, T + 1, L, s, 3)[:, :, :T]
    n_cc_va = min(2000, n_clean_crit // 5)
    cc_tr, cc_va = np.arange(n_clean_crit - n_cc_va), np.arange(n_clean_crit - n_cc_va,
                                                                n_clean_crit)
    Acc = torch.zeros(d + 1, d + 1, dtype=torch.float64, device=dev)
    Ccc = torch.zeros(d + 1, len(real), dtype=torch.float64, device=dev)
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
            kp = np.where(np.arange(sl.start, sl.stop) < len(cc_tr))[0]
            if len(kp):
                kt = torch.as_tensor(kp, device=dev)
                ob = torch.as_tensor(o_cc[:, sl], dtype=torch.float64, device=dev)
                rw = torch.as_tensor(rows, device=dev)
                Yb = torch.stack([ob[li][kt][:, rw + dd] for li in range(len(LEVELS))
                                  for dd in range(DMAX + 1)], -1).reshape(-1, len(real))
                Xb = inter[cb][kt][:, rw].double().reshape(-1, d)
                Xb = torch.cat([Xb, torch.ones(len(Xb), 1, dtype=torch.float64, device=dev)], 1)
                Acc += Xb.T @ Xb
                Ccc += Xb.T @ Yb
            vk = np.where(np.arange(sl.start, sl.stop) >= len(cc_tr))[0]
            if len(vk):
                Xcc_va[np.arange(sl.start, sl.stop)[vk] - len(cc_tr)] = \
                    inter[cb][torch.as_tensor(vk, device=dev)].float().cpu().numpy()
    Xvc = np.concatenate([Xcc_va[:, rows].reshape(-1, d).astype(np.float64),
                          np.ones((n_cc_va * len(rows), 1))], 1)
    Yvc = np.stack([o_cc[LEVELS.index(int(nm.split("l")[1].split("_")[0]))][cc_va][
        :, rows + int(nm.split("_d")[1])].reshape(-1) for nm in real], 1).astype(np.float64)
    bestc, clean_r2 = np.full(len(real), -np.inf), {}
    Acn, Ccn = Acc.cpu().numpy(), Ccc.cpu().numpy()
    del Acc, Ccc, Xcc_va
    torch.cuda.empty_cache()
    for lm in LAMS:
        B = ridge_solve(Acn, Ccn, lm)
        r2 = 1.0 - ((Xvc @ B - Yvc) ** 2).mean(0) / np.maximum(Yvc.var(0), 1e-12)
        for ti in np.where(r2 > bestc)[0]:
            bestc[ti] = r2[ti]
            beta[("clean", real[ti])] = B[:, ti].copy()
    for ti, nm in enumerate(real):
        clean_r2[nm] = float(bestc[ti])
    del Xvc, Yvc
    print(f"clean-only critic {time.time() - t0:.1f}s  l1_d0 val R2 "
          f"{clean_r2['l1_d0']:.4f}  (banked: 0.2271)", flush=True)

    # ---- the stream pass: V and the outcome at EVERY position ---------------------
    prows = np.arange(T_LO, T_HI + 1)                  # positions the critic was fit on
    nP, nL, nH = len(prows), len(SLEV), len(SHOR)
    crits = [("V", cb), ("Vc", "clean"), ("Vs", cb)]   # trained / clean-only / shuffled
    Bm = {}
    for nmc, src in crits:
        cols = []
        for l in SLEV:
            for dd in SHOR:
                k = (src, (f"sh_l{l}_d{dd}" if nmc == "Vs" else f"l{l}_d{dd}"))
                cols.append(beta[k])
        Bm[nmc] = torch.as_tensor(np.stack(cols, 1), dtype=torch.float32, device=dev)

    te = np.where(is_te)[0]
    if max_test and max_test < len(te):
        te = np.sort(np.random.default_rng(seed + 7).choice(te, max_test, replace=False))
    wq, phq, _, fq = windows_with_parse(rules, rule_w, n_quiet, T + 1, eval_seed + 2000)
    yq = clean_answers(fq, phq, T + 1, L, s, 3)[:, :, :T]

    def stream(W, Y):
        N = len(W)
        out = {k: np.zeros((N, nP, nL * nH), np.float16) for k, _ in crits}
        oo = np.zeros((N, T, nL), np.uint8)
        nl = np.zeros((N, T), np.float16)
        hq = np.zeros((N, T), np.float16)
        bx = np.zeros((N, nP), np.float16)
        pr = torch.as_tensor(prows, device=dev)
        with torch.no_grad():
            for c0 in range(0, N, 128):
                sl = slice(c0, min(c0 + 128, N))
                x = torch.as_tensor(W[sl, :T], device=dev)
                lg, _, inter = model(x, return_intermediates=True)
                lsm = torch.log_softmax(lg.float(), -1)
                nxt = torch.as_tensor(W[sl, 1:T + 1], device=dev)
                nl[sl] = (-lsm.gather(-1, nxt[..., None])[..., 0]).cpu().numpy()
                hq[sl] = (-(lsm.exp() * lsm).sum(-1)).cpu().numpy()
                Xp = inter[prim].float()
                for li, l in enumerate(SLEV):
                    pred = actor_apply(actors[(prim, l)], Xp)
                    tru = torch.as_tensor(Y[LEVELS.index(l)][sl].astype(np.int64), device=dev)
                    oo[sl, :, li] = (pred == tru).to(torch.uint8).cpu().numpy()
                Xc = inter[cb].float()[:, pr]                       # (bs, nP, d)
                Xc1 = torch.cat([Xc, torch.ones(*Xc.shape[:2], 1, device=dev)], -1)
                for nmc, _ in crits:
                    out[nmc][sl] = (Xc1 @ Bm[nmc]).cpu().numpy().astype(np.float16)
                if banked is not None:
                    Xb = inter[banked.block].float()[:, pr].reshape(-1, d).cpu().numpy()
                    bx[sl] = banked(Xb).reshape(Xc.shape[0], nP).astype(np.float16)
        return out, oo, nl, hq, bx

    t0 = time.time()
    arms = {}
    for aname, W, Y in (("edit", We[te], y_edit[:, te]),
                        ("orig", Wo[te], y_orig[:, te]),
                        ("quiet", wq, yq)):
        V, oo, nl, hq, bx = stream(W, Y)
        ent = {f"{aname}__{k}": vv for k, vv in V.items()}
        ent[f"{aname}__o"] = oo
        ent[f"{aname}__nll"] = nl
        ent[f"{aname}__H"] = hq
        ent[f"{aname}__bex"] = bx
        if aname == "quiet":
            ent[f"{aname}__t_v"] = -np.ones(len(W), np.int32)
            ent[f"{aname}__first_diff"] = -np.ones(len(W), np.int32)
            for k in ("etype", "j", "k_star", "e"):
                ent[f"{aname}__{k}"] = (np.full(len(W), 2 if k == "etype" else -1, np.int32))
        else:
            for k in ("etype", "j", "k_star", "e", "first_diff", "t_v"):
                ent[f"{aname}__{k}"] = meta[k][te].astype(np.int32)
        arms.update(ent)
        print(f"  stream {aname}: {len(W)} windows  {time.time() - t0:.0f}s", flush=True)

    res = {"ckpt": ckpt, "step": step, "stim_tag": stim_tag, "T": T, "d": d,
           "prows": prows.tolist(), "levels": SLEV, "horizons": SHOR,
           "critic_block": cb, "primary_block": prim, "banked_head": bpath if banked else None,
           "actor_clean_acc": actor_tab,
           "critic_val_r2": {nm: val_r2[(cb, nm)] for nm in tgt if not nm.startswith("sh_")},
           "critic_val_r2_shuf": {nm: val_r2[(cb, nm)] for nm in tgt if nm.startswith("sh_")},
           "clean_critic_val_r2": clean_r2,
           "n_test_used": int(len(te)), "n_quiet": int(n_quiet),
           "config": {"seed": seed, "eval_seed": eval_seed, "split": [0.6, 0.15, 0.25],
                      "n_clean": n_clean, "n_clean_crit": n_clean_crit, "dmax": DMAX,
                      "t_lo": T_LO, "blocks": blocks, "max_test": max_test,
                      "logbf": "not computed: basalis/hold.py's log Bayes factor is "
                               "anchor-relative (K=v branch rollouts from one flagged token, "
                               "zeroed on that token's surprisal); per position over a stream "
                               "it would be v branch passes at each of "
                               f"{len(prows)} positions of every window, ~{v * len(prows)}x "
                               "this cell. Skipped."}}
    stem = ckpt[:-3]
    sfx = f"_abstream_{stim_tag}" + (f"_{tag}" if tag else "")
    with open(f"{stem}{sfx}.json", "w") as f:
        json.dump(res, f, cls=NumpyEncoder, indent=1)
    np.savez_compressed(f"{stem}{sfx}.npz", **arms)
    volume.commit()
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    print(f"saved -> {stem}{sfx}.npz  ({time.time() - t00:.0f}s, peak RSS {rss:.1f} GB)",
          flush=True)
    return {"stim_tag": stim_tag, "step": step, "actor": actor_tab,
            "critic_l1_d0": val_r2[(cb, "l1_d0")], "n_test": int(len(te))}


@app.function(volumes={DATA_DIR: volume}, timeout=3 * 3600, memory=2048)
def abstain_sweep(ckpt: str = f"{D}/traj_a1_s42/step064000.pt",
                  stim_tags: str = "a1,swap65k", max_test: int = 3000,
                  n_quiet: int = 3000, tag: str = ""):
    """One container per venue, fanned out from a CPU-only coordinator (so `--detach` holds)."""
    tags = [t for t in stim_tags.split(",") if t]
    args = [(ckpt, t, 6144, 2048, 8192, n_quiet, max_test, 11, 5150, 3000, tag)
            for t in tags]
    out = list(abstain_stream.starmap(args))
    for r in out:
        print(json.dumps(r, default=float), flush=True)
    return out
