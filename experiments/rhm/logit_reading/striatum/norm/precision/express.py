"""The expressivity test: can a reader whose feature map carries the belief's ENTROPY price
what the pre-event uncertainty knows about damage, where the banked linear critic cannot?

`precision/` found that at matched surprisal the pre-event entropy `H_pre` ranks realised
damage at ell = 1 (0.62 at 64k) and that nothing the banked reader has (its level, its
response) or the event does (forecast movement, state movement) accounts for it.  The
candidate reading is about the READER'S FORM: the entropy of the belief is a nonlinear
function of the state, and the critic is a linear projection of the state, so a ridge can
price the belief's level but cannot form its entropy.  This file is the direct test.

It refits the critic on the SAME trunk, the SAME diet rows (`full`, and the clean-only
critic), the SAME split, the SAME lambda ladder and the SAME held-out lambda selection as
`norm/task.py`, with the feature map augmented, and writes the same per-row columns on the
identical `split == 2` test rows, so `reduce.py` / `mediation.py` read them unchanged.

Arms (the critic's name carries the arm; `full` / `clean` are arm 0, the banked ridge):

  arm 0  `full`   the state (`post_block7`), exactly `norm/task.py`'s ridge -- THE GATE
  arm H  `fullH`  state + H(q_t): the entropy of the forecast made at the state's own position
                  (so `V_pre` sees `H_pre` and the post-event level sees `H_next`)  -- primary
  arm Q  `fullQ`  state + log q_t (16 dims): the belief's LINEAR content, not its entropy --
                  the control for arm H (layer norm sits between the state and the logits,
                  so the state ridge need not already contain it)
  arm N  `fullN`  state + ||s_t||_2: a placebo nonlinear scalar
  arm T  `fullT`  state + tanh(u . s_t / sd): a second placebo, one fixed random direction
  arm P  `fullP`  state + H(q_t) + H(q_(t-1)) + s_t + H(q_(t-1)) * s_t, with s_t the
                  surprisal of the token AT t: the precision-weighted reader in the
                  literature's own form (the event's surprise times the uncertainty the
                  forecast of it was made under), plus arm H's column -- secondary
  arm M  `fullM`  an MLP on the state, `coeruleus.readout.train_head` with `striatum/task.py`'s
                  settings (hidden 128, 2000 steps), levels 1-4 at a = 0 only -- secondary

**Scaling.**  `ridge_solve` penalises every non-bias coefficient by the same
`lam * trace(A) / D`, so an appended column on its own scale would be crushed or favoured by
the penalty for no reason.  Every appended column is therefore affinely standardised to mean
0 and the state's own per-dimension RMS on the training rows, exactly, by transforming the
Gram (`A' = T' A T`), and the fitted coefficients are mapped back so prediction uses the raw
features.  The state columns and the bias are untouched, so arm 0 is the banked ridge.

**Gates**, all computed in-container and written into the output's `meta`; the cell FAILS
(raises after writing a diagnostics json) unless every one holds:
  - the frozen actor's clean accuracy equals the banked json at every level;
  - arm 0's held-out fit equals the banked `critic_val_r2['full']` and
    `clean_critic_val_r2` on every level-offset column;
  - the test rows (`w`, `t0`) and every outcome label (`oe`, `oo`, `cons`, `ok`) equal the
    banked npz EXACTLY, and `nll_e`, `H_pre` to float32;
  - arm 0's `V`, `Vpre`, `R`, `Ro`, `Rsh` equal the banked `full` / `clean` columns to
    float32, at both anchors, every level and offset;
  - on the twins, arm 0's `V` at offsets 0 and -1 equals the banked `tw__V_viol` /
    `tw__V_twin` to float32.

**Follow-up 6 (2026-09-23), `--logit-arms`**: does the linear reader lack the belief's
LOG-PARTITION, or does its shrinkage under-use the belief's directions inside 256 normalised
dimensions?  The logits are `W_U ln_f(s)` exactly (`lm_head` has no bias), so a ridge on
`ln_f(s)` already spans every linear function of them, and `log q = z - logsumexp(z)`.  The
arms (a SEPARATE appended block with its own Grams, so every committed arm and column is
untouched; each appended column standardised through the Gram to the base block's
per-dimension RMS exactly as above, coefficients mapped back):
  arm Z   `fullZ`   state + the 16 raw logits z_t: the belief's directions handed over as
                    sixteen columns, without the log-partition
  arm LE  `fullLE`  ln_f(s) in place + logsumexp(z_t): the log-partition handed over, the
                    directions left inside the 256 normalised dimensions
  arm LX  `fullLX`  ln_f(s) in place + max(z_t): the confirming scalar
  arm E   `fullE`   state + logsumexp(z_t): the one scalar on the raw basis (the analogue of
                    arm H)
  arm ZE  `fullZE`  state + z_t + logsumexp(z_t): spans `fullQ`'s features exactly, plus
                    one more direction (the closing arm on the raw basis)
`--check-sfx` compares every column the cell shares with the named committed cells (the
refitted `fullH` / `fullQ` / `fullL` / `fullLp` and the base columns) and gates on it.

Run:
  modal run -m rhm.logit_reading.striatum.norm.precision.express::express_ckpt \
      --ckpt /data/.../traj_a1_s42/step064000.pt --stim-tag swap65k
  modal run --detach -m rhm.logit_reading.striatum.norm.precision.express::express_sweep
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
from rhm.logit_reading.striatum.norm.task import (LEVELS, DMAX, DD_LIST, A_LIST, A_STORE,
                                                  L_STORE, T_LO, LAMS, OFFS, tgt_names)


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
app = modal.App("rhm-norm-precision-express", image=image)

# the appended feature block, K columns, in this order
AUG = (["H"] + [f"logq{i}" for i in range(16)] + ["norm", "tanh", "Hm1", "s", "Hm1s"])
K = len(AUG)
ARMS = {"H": [0], "Q": list(range(1, 17)), "N": [17], "T": [18], "P": [0, 19, 20, 21]}
MLP_DD = (0, 1)
# follow-up 6: the belief's raw logits, their log-partition and their max, in a SEPARATE
# appended block (own Grams), so the committed block above and its Gram are untouched
ZAUG = [f"z{i}" for i in range(16)] + ["lse", "zmax"]
KZ = len(ZAUG)
ZARMS = {"Z": list(range(16)), "ZE": list(range(17)), "E": [16]}   # on the raw state
LZARMS = {"LE": [16], "LX": [17]}                                   # on ln_f(s), in place


def _z_cols(lg):
    """The KZ logit features from the raw logits `lg[..., 16]`: the logits, logsumexp and
    max, computed in float64 and stored float32.  One function for the fit rows, the anchors
    and the twins, so the three agree exactly."""
    z = np.asarray(lg, np.float64)
    assert z.shape[-1] == 16, z.shape
    m = z.max(-1, keepdims=True)
    lse = m[..., 0] + np.log(np.exp(z - m).sum(-1))
    return np.concatenate([z, lse[..., None], m], -1).astype(np.float32)


def _aug_cols(Hq, nll, lsm_at, st_at, u, sdz, pos, ww):
    """The K appended features at (window ww[i], position pos[i]).

    `Hq[w, p]` is H(q_p), the entropy of the forecast MADE at p; `nll[w, p]` is
    -log q_p(x_(p+1)), so the surprisal of the token AT p is `nll[w, p - 1]`.  `lsm_at` is
    log q at the same (window, position) and `st_at` the float32 state there."""
    H = Hq[ww, pos]
    Hm1 = Hq[ww, pos - 1]
    s = nll[ww, pos - 1]
    nrm = np.linalg.norm(st_at, axis=-1)
    th = np.tanh((st_at @ u) / sdz)
    return np.concatenate([H[..., None], lsm_at, nrm[..., None], th[..., None],
                           Hm1[..., None], s[..., None], (Hm1 * s)[..., None]],
                          -1).astype(np.float32)


def _standardiser(A, d, n_rows):
    """T such that F' = F T standardises the K appended columns of F = [S, G, 1] to mean 0
    and the state's own per-dimension RMS, leaving S and the bias untouched."""
    D = A.shape[0]
    Kk = D - d - 1
    s_state = np.sqrt(max(np.trace(A[:d, :d]) / (n_rows * d), 1e-18))
    T = np.eye(D)
    for j in range(Kk):
        c = d + j
        mu = A[c, -1] / n_rows
        var = max(A[c, c] / n_rows - mu * mu, 1e-18)
        sc = s_state / np.sqrt(var)
        T[c, c] = sc
        T[-1, c] = -mu * sc
    return T


def _fit_arm(A, C, Xv, Yv, cols, lams):
    """Ridge on the [cols, bias] sub-block of an already-standardised Gram, lambda picked
    per target column on held-out rows exactly as `norm/task.py` does.  Returns
    (beta (len(cols)+1, n_tgt) in the standardised basis, best R2 per column)."""
    idx = list(cols) + [A.shape[0] - 1]
    As, Cs = A[np.ix_(idx, idx)], C[idx]
    Xs = Xv[:, idx]
    yvar = np.maximum(Yv.var(0, ddof=1), 1e-12)          # torch's var(0) is unbiased
    best = np.full(C.shape[1], -1e18)
    Bb = np.zeros((len(idx), C.shape[1]))
    for lm in lams:
        B = ridge_solve(As, Cs, lm)
        r2 = 1.0 - ((Xs @ B - Yv) ** 2).mean(0) / yvar
        up = r2 > best
        best = np.where(up, r2, best)
        Bb[:, up] = B[:, up]
    return Bb, best


def _fit_map_arms(Am, Cm, As, Cs, Xv, LXv, SXv, Yv, d, n_rows, arms, lams):
    """The feature-MAP arms (follow-up 4).  `Am`/`Cm` is the Gram of [s, ln_f(s), 1] and
    `As`/`Cs` of [std(s), 1] on the same training rows.

      L   ln_f(s) IN PLACE of the state: the model's own final layer norm, weight and bias,
          exactly what the unembedding reads (`post_block7` feeds `ln_f` directly).
      Lp  [s, ln_f(s)]: the normalised state APPENDED to the raw one; the ln block is scaled
          AS A BLOCK to the raw state's per-dimension RMS (its internal geometry kept), by a
          diagonal transform of the Gram, and mapped back.
      S   (s - mean) / sqrt(var + eps) per row, no learned affine, in place: normalisation
          separated from the model's own normalisation.
    Returns {arm: (beta in the RAW basis of the arm's features + bias, R2 per column, info)}.
    """
    out = {}
    one = np.ones((len(Xv), 1))
    if "L" in arms:
        B, r2 = _fit_arm(Am, Cm, np.concatenate([Xv, LXv, one], 1), Yv,
                         list(range(d, 2 * d)), lams)
        out["L"] = (B, r2, {})
    if "Lp" in arms:
        s_x = np.sqrt(max(np.trace(Am[:d, :d]) / (n_rows * d), 1e-18))
        s_l = np.sqrt(max(np.trace(Am[d:2 * d, d:2 * d]) / (n_rows * d), 1e-18))
        c = s_x / s_l
        Tm = np.eye(2 * d + 1)
        Tm[d:2 * d, d:2 * d] *= c
        B, r2 = _fit_arm(Tm.T @ Am @ Tm, Tm.T @ Cm,
                         np.concatenate([Xv, LXv, one], 1) @ Tm, Yv, list(range(2 * d)), lams)
        out["Lp"] = (Tm @ B, r2, {"ln_block_scale": float(c), "rms_state": float(s_x),
                                  "rms_ln": float(s_l)})
    if "S" in arms:
        B, r2 = _fit_arm(As, Cs, np.concatenate([SXv, one], 1), Yv, list(range(d)), lams)
        out["S"] = (B, r2, {})
    return out


def _fit_logit_arms(Az, Cz, Alz, Clz, Xv, LXv, Zv, Yv, d, n_rows, arms_z, arms_lz, lams):
    """Follow-up 6.  `Az`/`Cz` is the Gram of [s, z (16), lse, zmax, 1] and `Alz`/`Clz` of
    [ln_f(s), lse, zmax, 1] on the same training rows.  The appended columns of each are
    standardised through the Gram to that Gram's base block's per-dimension RMS
    (`_standardiser`, exactly as the committed arms), each arm fitted on its sub-block with
    `norm/task.py`'s per-column held-out lambda selection, and the coefficients mapped back to
    the raw basis.  Also returns the SPAN diagnostic: held-out R2 of lse, zmax and each logit
    regressed (same ladder, same selection) on [s], [s, z] and [ln_f(s)] -- how far each
    appended feature already lies in each base span on these rows.
    Returns ({arm: (beta raw, R2 per column, cols, "zcols" | "lzcols")}, span)."""
    out, span = {}, {}
    one = np.ones((len(Xv), 1))
    Dz = d + KZ + 1
    Tz = _standardiser(Az, d, n_rows)
    Azs, Czs = Tz.T @ Az @ Tz, Tz.T @ Cz
    Fvz = np.concatenate([Xv, Zv, one], 1) @ Tz
    for arm in arms_z:
        cc = list(range(d)) + [d + c for c in ZARMS[arm]]
        B, r2 = _fit_arm(Azs, Czs, Fvz, Yv, cc, lams)
        full_b = np.zeros((Dz, Cz.shape[1]))
        full_b[cc + [Dz - 1]] = B
        out[arm] = ((Tz @ full_b)[cc + [Dz - 1]], r2, cc, "zcols")
    Tl = _standardiser(Alz, d, n_rows)
    Als, Cls = Tl.T @ Alz @ Tl, Tl.T @ Clz
    Fvl = np.concatenate([LXv, Zv[:, 16:], one], 1) @ Tl
    for arm in arms_lz:
        cc = list(range(d)) + [d + c - 16 for c in LZARMS[arm]]
        B, r2 = _fit_arm(Als, Cls, Fvl, Yv, cc, lams)
        full_b = np.zeros((d + 3, Clz.shape[1]))
        full_b[cc + [d + 2]] = B
        out[arm] = ((Tl @ full_b)[cc + [d + 2]], r2, cc, "lzcols")
    # the span diagnostic: targets are the RAW appended columns
    tz = list(range(d, d + KZ))
    Cy = (Tz.T @ Az)[:, tz]
    _, r_s = _fit_arm(Azs, Cy, Fvz, Zv, list(range(d)), lams)
    _, r_sz = _fit_arm(Azs, Cy[:, 16:], Fvz, Zv[:, 16:], list(range(d + 16)), lams)
    Cyl = (Tl.T @ Alz)[:, [d, d + 1]]
    _, r_l = _fit_arm(Als, Cyl, Fvl, Zv[:, 16:], list(range(d)), lams)
    span = {"lse": {"on_s": float(r_s[16]), "on_s_z": float(r_sz[0]), "on_ln": float(r_l[0])},
            "zmax": {"on_s": float(r_s[17]), "on_s_z": float(r_sz[1]), "on_ln": float(r_l[1])},
            "z_on_s": [float(x) for x in r_s[:16]],
            "sd": {"lse": float(Zv[:, 16].std()), "zmax": float(Zv[:, 17].std()),
                   "z_mean_sd": float(Zv[:, :16].std(0).mean())}}
    return out, span


def _probe(Xab, y, tr, va, lams):
    """junction/task.py's legality ceiling recipe: ridge on the anchor state (with bias),
    fitted on the training split, lambda picked on the validation split."""
    Ag = Xab[tr].T @ Xab[tr]
    Cg = Xab[tr].T @ y[tr][:, None]
    bb, br2 = None, -np.inf
    for lm in lams:
        B = ridge_solve(Ag, Cg, lm)
        r2 = 1.0 - ((Xab[va] @ B[:, 0] - y[va]) ** 2).mean() / max(y[va].var(), 1e-12)
        if r2 > br2:
            bb, br2 = B[:, 0], r2
    return Xab @ bb, float(br2)


@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=3 * 3600, memory=16384,
              max_containers=6)
def express_ckpt(ckpt: str, stim_tag: str = "swap65k", n_clean: int = 6144,
                 n_clean_val: int = 2048, seed: int = 11, actor_steps: int = 3000,
                 eval_seed: int = 5150, n_clean_crit: int = 8192,
                 prim_block: str = "post_block7", n_val_windows: int = 1500,
                 chunk: int = 512, do_mlp: bool = True, mlp_steps: int = 2000,
                 do_twins: bool = True, tol: float = 5e-5, u_seed: int = 4242,
                 arm_set: str = "H,Q,N,T,P", map_arms: str = "", out_sfx: str = "_express",
                 bank_tag: str = "", do_probes: bool = False, logit_arms: str = "",
                 check_sfx: str = ""):
    """Defaults reproduce the committed `_express` cells exactly.  Follow-up 4 adds
    `map_arms` ("L,Lp,S"); follow-up 5 runs a venue with no banked `norm/` cell, where
    `bank_tag` names the banked venue whose json supplies the actor's clean accuracy (the
    actor is trained on clean windows only, so it is venue-independent) and every
    column / row / twin gate is skipped because there is nothing banked to compare with."""
    import torch
    volume.reload()
    t00 = time.time()
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
    own_bank = not bank_tag
    bank_npz = np.load(f"{stem}_norm_{stim_tag}.npz") if own_bank else None
    bank = json.load(open(f"{stem}_norm_{bank_tag or stim_tag}.json"))
    arms_aug = {a: ARMS[a] for a in arm_set.split(",") if a}
    arms_map = [a for a in map_arms.split(",") if a]
    assert all(a in ("L", "Lp", "S") for a in arms_map), arms_map
    lg_arms = [a for a in logit_arms.split(",") if a]
    assert all(a in ZARMS or a in LZARMS for a in lg_arms), lg_arms
    arms_z = [a for a in lg_arms if a in ZARMS]
    arms_lz = [a for a in lg_arms if a in LZARMS]
    use_z = bool(lg_arms)
    need_ln = bool(arms_map) or use_z          # the span diagnostic reads ln_f(s) too
    lnf = model.transformer.ln_f
    print(f"loaded {ckpt} step {cfg.get('step')} d={d} T={T}", flush=True)

    # ---- norm/task.py verbatim: windows, split -------------------------------------------
    n_all = S["windows_edit"].shape[0]
    if own_bank:
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

    # ---- norm/task.py verbatim: the frozen actors (prim block; each reseeds itself) ------
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
    # the placebo direction: fixed, and its scale from the clean actor windows (independent
    # of every row the critic is fitted or read on)
    u = np.random.default_rng(u_seed).standard_normal(d) / np.sqrt(d)
    sdz = float((Sc[:n_clean][:, pos].reshape(-1, d) @ u).std())
    del Xtr, Xva, Sc
    torch.cuda.empty_cache()
    print(f"actors {time.time() - t0:.1f}s  {json.dumps({k: round(x, 5) for k, x in actor_tab.items()})}",
          flush=True)

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

    # ---- the two forward passes, plus the appended features --------------------------------
    def run(W, Y, want_cache=False, bs=256):
        o = np.zeros((len(LEVELS), len(W), T), np.uint8)
        nll = np.zeros((len(W), T), np.float32)
        Hq = np.zeros((len(W), T), np.float32)
        anc = np.zeros((len(W), 4, d), np.float32)
        lsm_anc = np.zeros((len(W), 4, v), np.float32)
        lg_anc = np.zeros((len(W), 4, v), np.float32)
        zaux = np.zeros((len(cache_idx), R_, KZ), np.float32) if (want_cache and use_z) else None
        emb_anc = np.zeros((len(W), 4, d), np.float32) if (do_probes and want_cache) else None
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
                lg_anc[sl] = lg[ar, ap].float().cpu().numpy()
                if emb_anc is not None:
                    emb_anc[sl] = inter["post_embed"][ar, ap].float().cpu().numpy()
                if want_cache:
                    kk = np.where(cache_map[c0:sl.stop] >= 0)[0]
                    if len(kk):
                        kt = torch.as_tensor(kk, device=dev)
                        ci = cache_map[c0 + kk]
                        cache[ci] = inter[prim][kt][:, rows].half().cpu().numpy()
                        # Hq / nll of this batch are already filled, so rows and rows - 1
                        # of these windows are available
                        nb = len(kk)
                        aug[ci] = _aug_cols(Hq, nll, lsm[kt][:, rows].cpu().numpy(),
                                            inter[prim][kt][:, rows].float().cpu().numpy(),
                                            u, sdz, np.broadcast_to(rows[None, :], (nb, R_)),
                                            np.broadcast_to((c0 + kk)[:, None], (nb, R_)))
                        if zaux is not None:
                            zaux[ci] = _z_cols(lg[kt][:, rows].float().cpu().numpy())
        return o, nll, Hq, anc, lsm_anc, cache, aug, emb_anc, lg_anc, zaux

    t0 = time.time()
    o_o, nll_o, H_o, st_o, lsm_ao, _, _, _, lg_ao, _ = run(Wo, y_orig)
    (o_e, nll_e, H_e, st_e, lsm_ae, cache, aug_c, emb_ae, lg_ae,
     zaux_c) = run(We, y_edit, want_cache=True)
    print(f"passes {time.time() - t0:.1f}s  cache {cache.nbytes / 1e9:.2f} GB", flush=True)

    # ---- Grams for `full`: arm 0 exactly as norm/task.py, and the augmented one ------------
    names = tgt_names()
    n_tgt = len(names)
    n_real = n_tgt // 2
    D = d + K + 1
    A0 = torch.zeros(d + 1, d + 1, dtype=torch.float64, device=dev)
    C0 = torch.zeros(d + 1, n_tgt, dtype=torch.float64, device=dev)
    Aa = torch.zeros(D, D, dtype=torch.float64, device=dev)
    Ca = torch.zeros(D, n_tgt, dtype=torch.float64, device=dev)
    if arms_map:
        Am = torch.zeros(2 * d + 1, 2 * d + 1, dtype=torch.float64, device=dev)
        Cm = torch.zeros(2 * d + 1, n_tgt, dtype=torch.float64, device=dev)
        Asd = torch.zeros(d + 1, d + 1, dtype=torch.float64, device=dev)
        Csd = torch.zeros(d + 1, n_tgt, dtype=torch.float64, device=dev)
    if use_z:
        Az = torch.zeros(d + KZ + 1, d + KZ + 1, dtype=torch.float64, device=dev)
        Cz = torch.zeros(d + KZ + 1, n_tgt, dtype=torch.float64, device=dev)
        Alz = torch.zeros(d + 3, d + 3, dtype=torch.float64, device=dev)
        Clz = torch.zeros(d + 3, n_tgt, dtype=torch.float64, device=dev)
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
        if arms_map:
            with torch.no_grad():
                Xf = X.float()
                LX = lnf(Xf).double()
                SX = torch.nn.functional.layer_norm(Xf, (d,), eps=lnf.eps).double()
            Fm = torch.cat([X, LX, ones], 1)
            Am += Fm.T @ Fm
            Cm += Fm.T @ Y
            Fs = torch.cat([SX, ones], 1)
            Asd += Fs.T @ Fs
            Csd += Fs.T @ Y
            del Xf, LX, SX, Fm, Fs
        if use_z:
            Zt = torch.as_tensor(zaux_c[sl].reshape(-1, KZ), device=dev).double()
            Fz = torch.cat([X, Zt, ones], 1)
            Az += Fz.T @ Fz
            Cz += Fz.T @ Y
            with torch.no_grad():
                LXz = lnf(X.float()).double()
            Flz = torch.cat([LXz, Zt[:, 16:], ones], 1)
            Alz += Flz.T @ Flz
            Clz += Flz.T @ Y
            del Zt, Fz, LXz, Flz
        n_rows_tr += len(X)
        del X, X1, F, G, Y
    print(f"grams {time.time() - t0:.1f}s", flush=True)

    # held-out rows for lambda selection (norm/task.py's)
    Xv = cache[len(tr_idx):].reshape(-1, d).astype(np.float64)
    Gv = aug_c[len(tr_idx):].reshape(-1, K).astype(np.float64)
    Zv = zaux_c[len(tr_idx):].reshape(-1, KZ).astype(np.float64) if use_z else None
    Yv = np.stack([o_e[li][np.ix_(va_idx, rows + dd)].astype(np.float64).reshape(-1)
                   for li in range(len(LEVELS)) for dd in DD_LIST], -1)
    Yv = np.concatenate([Yv, Yv], 1)

    # arm 0: norm/task.py's own loop, on the GPU, bit for bit
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

    # the augmented arms: standardise the appended columns, fit each arm's sub-block
    Aan, Can = Aa.cpu().numpy(), Ca.cpu().numpy()
    Tst = _standardiser(Aan, d, n_rows_tr)
    As_, Cs_ = Tst.T @ Aan @ Tst, Tst.T @ Can
    Fv = np.concatenate([Xv, Gv, np.ones((len(Xv), 1))], 1) @ Tst
    scale_info = {"s_state": float(np.sqrt(np.trace(Aan[:d, :d]) / (n_rows_tr * d)))}
    for arm, cols in arms_aug.items():
        cc = list(range(d)) + [d + c for c in cols]
        B, r2 = _fit_arm(As_, Cs_, Fv, Yv, cc, LAMS)
        # back to the raw feature basis: V = F_raw (T beta') restricted to the arm's columns
        full_b = np.zeros((D, n_tgt))
        full_b[cc + [D - 1]] = B
        raw = Tst @ full_b
        beta[f"full{arm}"] = raw[cc + [D - 1]]
        beta[f"full{arm}__cols"] = cc
        val_r2[f"full{arm}"] = {names[i]: float(r2[i]) for i in range(n_tgt)}

    def maps_np(X32):
        """ln_f(s) and the affine-free standardisation of s, float64, on the GPU."""
        if not need_ln:
            return None, None
        Lz, Sz = [], []
        with torch.no_grad():
            for i in range(0, len(X32), 65536):
                t = torch.as_tensor(np.asarray(X32[i:i + 65536], np.float32), device=dev)
                Lz.append(lnf(t).double().cpu().numpy())
                Sz.append(torch.nn.functional.layer_norm(t, (d,), eps=lnf.eps)
                          .double().cpu().numpy())
        return np.concatenate(Lz), np.concatenate(Sz)

    if arms_map:
        LXv, SXv = maps_np(Xv)
        fitm = _fit_map_arms(Am.cpu().numpy(), Cm.cpu().numpy(), Asd.cpu().numpy(),
                             Csd.cpu().numpy(), Xv, LXv, SXv, Yv, d, n_rows_tr, arms_map, LAMS)
        for arm, (B, r2, info) in fitm.items():
            beta[f"full{arm}"] = B
            beta[f"full{arm}__map"] = arm
            val_r2[f"full{arm}"] = {names[i]: float(r2[i]) for i in range(n_tgt)}
            scale_info[f"full{arm}"] = info
        del Am, Cm, Asd, Csd, LXv, SXv
    span = {}
    if use_z:
        LXv_z = maps_np(Xv)[0]
        fz, sp = _fit_logit_arms(Az.cpu().numpy(), Cz.cpu().numpy(), Alz.cpu().numpy(),
                                 Clz.cpu().numpy(), Xv, LXv_z, Zv, Yv, d, n_rows_tr,
                                 arms_z, arms_lz, LAMS)
        for arm, (B, r2, cc, kind) in fz.items():
            beta[f"full{arm}"] = B
            beta[f"full{arm}__{kind}"] = cc
            val_r2[f"full{arm}"] = {names[i]: float(r2[i]) for i in range(n_tgt)}
        span["full"] = sp
        del Az, Cz, Alz, Clz, LXv_z
    del Aa, Ca, A0, C0
    torch.cuda.empty_cache()
    print(f"solves {time.time() - t0:.1f}s  l1_d0 R2: " + "  ".join(
        f"{nm} {val_r2[nm]['l1_d0']:.4f}" for nm in val_r2), flush=True)

    # ---- the MLP arm ---------------------------------------------------------------------
    mlp = {}
    if do_mlp:
        from rhm.logit_reading.coeruleus.readout import train_head
        from rhm.logit_reading.striatum.task import _r2
        t0 = time.time()
        Xtr_m = cache[:len(tr_idx)].reshape(-1, d).astype(np.float32)
        Xva_m = cache[len(tr_idx):].reshape(-1, d).astype(np.float32)
        for l in L_STORE:
            li = LEVELS.index(l)
            for dd in MLP_DD:
                ytr = o_e[li][tr_idx][:, rows + dd].reshape(-1).astype(np.float32)
                yva = o_e[li][va_idx][:, rows + dd].reshape(-1).astype(np.float64)
                h = train_head(Xtr_m, ytr, steps=mlp_steps, device=dev, seed=l * 17 + dd)
                mlp[f"l{l}_d{dd}"] = h
                val_r2.setdefault("fullM", {})[f"l{l}_d{dd}"] = _r2(h(Xva_m), yva)
        del Xtr_m, Xva_m
        torch.cuda.empty_cache()
        print(f"mlp {time.time() - t0:.1f}s  " + " ".join(
            f"{k} {x:.4f}" for k, x in val_r2["fullM"].items()), flush=True)

    # ---- the clean-only critic, arm 0 exactly as norm/task.py, plus the arms ---------------
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
    if arms_map:
        Amc = torch.zeros(2 * d + 1, 2 * d + 1, dtype=torch.float64, device=dev)
        Cmc = torch.zeros(2 * d + 1, n_real, dtype=torch.float64, device=dev)
        Asc_ = torch.zeros(d + 1, d + 1, dtype=torch.float64, device=dev)
        Csc_ = torch.zeros(d + 1, n_real, dtype=torch.float64, device=dev)
    if use_z:
        Acz = torch.zeros(d + KZ + 1, d + KZ + 1, dtype=torch.float64, device=dev)
        Ccz = torch.zeros(d + KZ + 1, n_real, dtype=torch.float64, device=dev)
        Aclz = torch.zeros(d + 3, d + 3, dtype=torch.float64, device=dev)
        Cclz = torch.zeros(d + 3, n_real, dtype=torch.float64, device=dev)
        Zcc_va = np.zeros((n_cc_va, R_, KZ), np.float32)
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
            Z_b = _z_cols(lg[:, rows].float().cpu().numpy()) if use_z else None
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
                if arms_map:
                    Xbf = inter[prim][kt][:, rows].float().reshape(-1, d)
                    LXb = lnf(Xbf).double()
                    SXb = torch.nn.functional.layer_norm(Xbf, (d,), eps=lnf.eps).double()
                    Fmb = torch.cat([Xb, LXb, ones], 1)
                    Amc += Fmb.T @ Fmb
                    Cmc += Fmb.T @ Yb
                    Fsb = torch.cat([SXb, ones], 1)
                    Asc_ += Fsb.T @ Fsb
                    Csc_ += Fsb.T @ Yb
                if use_z:
                    Ztc = torch.as_tensor(Z_b[kk].reshape(-1, KZ), device=dev).double()
                    Fzb = torch.cat([Xb, Ztc, ones], 1)
                    Acz += Fzb.T @ Fzb
                    Ccz += Fzb.T @ Yb
                    LXbz = lnf(inter[prim][kt][:, rows].float().reshape(-1, d)).double()
                    Flzb = torch.cat([LXbz, Ztc[:, 16:], ones], 1)
                    Aclz += Flzb.T @ Flzb
                    Cclz += Flzb.T @ Yb
                n_cc_rows += len(Xb)
            vk = np.where(ii >= n_cc_tr)[0]
            if len(vk):
                Xcc_va[ii[vk] - n_cc_tr] = \
                    inter[prim][torch.as_tensor(vk, device=dev)][:, rows].float().cpu().numpy()
                Gcc_va[ii[vk] - n_cc_tr] = G_b[vk]
                if use_z:
                    Zcc_va[ii[vk] - n_cc_tr] = Z_b[vk]
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
    for arm, cols in arms_aug.items():
        cc = list(range(d)) + [d + c for c in cols]
        B, r2 = _fit_arm(Asc, Csc, Fvc, Yvc_n, cc, LAMS)
        full_b = np.zeros((D, n_real))
        full_b[cc + [D - 1]] = B
        raw = Tcc @ full_b
        beta[f"clean{arm}"] = raw[cc + [D - 1]]
        beta[f"clean{arm}__cols"] = cc
        val_r2[f"clean{arm}"] = {real_names[i]: float(r2[i]) for i in range(n_real)}
    if arms_map:
        Xvcc = Xcc_va.reshape(-1, d).astype(np.float64)
        LXvc, SXvc = maps_np(Xcc_va.reshape(-1, d))
        fitc = _fit_map_arms(Amc.cpu().numpy(), Cmc.cpu().numpy(), Asc_.cpu().numpy(),
                             Csc_.cpu().numpy(), Xvcc, LXvc, SXvc, Yvc_n, d, n_cc_rows,
                             arms_map, LAMS)
        for arm, (B, r2, info) in fitc.items():
            beta[f"clean{arm}"] = B
            beta[f"clean{arm}__map"] = arm
            val_r2[f"clean{arm}"] = {real_names[i]: float(r2[i]) for i in range(n_real)}
            scale_info[f"clean{arm}"] = info
        del Amc, Cmc, Asc_, Csc_
    if use_z:
        Xvcc = Xcc_va.reshape(-1, d).astype(np.float64)
        LXvc_z = maps_np(Xcc_va.reshape(-1, d))[0]
        fzc, spc = _fit_logit_arms(Acz.cpu().numpy(), Ccz.cpu().numpy(), Aclz.cpu().numpy(),
                                   Cclz.cpu().numpy(), Xvcc, LXvc_z,
                                   Zcc_va.reshape(-1, KZ).astype(np.float64), Yvc_n, d,
                                   n_cc_rows, arms_z, arms_lz, LAMS)
        for arm, (B, r2, cc, kind) in fzc.items():
            beta[f"clean{arm}"] = B
            beta[f"clean{arm}__{kind}"] = cc
            val_r2[f"clean{arm}"] = {real_names[i]: float(r2[i]) for i in range(n_real)}
        span["clean"] = spc
        del Acz, Ccz, Aclz, Cclz, LXvc_z
    del Acc, Ccc, Aca, Cca, Xvc, Yvc, cache, aug_c
    torch.cuda.empty_cache()
    print(f"clean critics {time.time() - t0:.1f}s  l1_d0: " + "  ".join(
        f"{nm} {val_r2[nm]['l1_d0']:.4f}" for nm in val_r2 if nm.startswith("clean")),
        flush=True)

    # ---- prediction on raw features ----------------------------------------------------
    arm_names = (["full"] + [f"full{a}" for a in arms_aug] + [f"full{a}" for a in arms_map]
                 + [f"full{a}" for a in lg_arms]
                 + ["clean"] + [f"clean{a}" for a in arms_aug]
                 + [f"clean{a}" for a in arms_map] + [f"clean{a}" for a in lg_arms])

    def colidx(nm, name):
        return (names.index(name) if not nm.startswith("clean") else real_names.index(name))

    def feat(nm, Xs, Gs, Ls=None, Ss=None, Zs=None):
        """The arm's raw feature matrix (no bias column).  Built once per stream and reused
        across every target column (2026-09-23: the same array, so every number is
        bit-identical to building it per column as before; only the time changes)."""
        if nm in ("full", "clean"):
            return Xs
        mp = beta.get(f"{nm}__map")
        if f"{nm}__zcols" in beta:
            return np.concatenate([Xs, Zs], 1)[:, beta[f"{nm}__zcols"]]
        if f"{nm}__lzcols" in beta:
            return np.concatenate([Ls, Zs[:, 16:]], 1)[:, beta[f"{nm}__lzcols"]]
        if mp == "L":
            return Ls
        if mp == "Lp":
            return np.concatenate([Xs, Ls], 1)
        if mp == "S":
            return Ss
        return np.concatenate([Xs, Gs], 1)[:, beta[f"{nm}__cols"]]

    def vcol(nm, name, F):
        b = beta[nm][:, colidx(nm, name)]
        return F @ b[:-1] + b[-1]

    def lin(nm, name, Xs, Gs, Ls=None, Ss=None, Zs=None):
        return vcol(nm, name, feat(nm, Xs, Gs, Ls, Ss, Zs))

    def mlp_pred(name, Xs):
        return mlp[name](Xs.astype(np.float32)).astype(np.float64)

    # ---- the anchor records: norm/task.py verbatim for the base columns ---------------------
    gates = {"actor": {}, "val_r2": {}, "rows": {}, "cols": {}}
    for k_, a_ in actor_tab.items():
        gates["actor"][k_] = abs(a_ - bank["actor_clean_acc"][k_])
    for l in L_STORE:
        for dd in (0, 1):
            if not own_bank:
                break
            nm_ = f"l{l}_d{dd}"
            gates["val_r2"][f"full/{nm_}"] = abs(val_r2["full"][nm_]
                                                 - bank["critic_val_r2"]["full"][nm_])
            gates["val_r2"][f"clean/{nm_}"] = abs(val_r2["clean"][nm_]
                                                  - bank["clean_critic_val_r2"][nm_])
    save = {}
    for an, t0a in anchors.items():
        t_an = time.time()
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
        rec["dH_e"] = H_e[w, t0w] - H_e[w, t0w - 1]
        rec["dH_o"] = H_o[w, t0w] - H_o[w, t0w - 1]
        rec["excess_e"] = rec["nll_e"] - rec["H_pre"]
        rec["tok_legal"] = (np.zeros(len(w), np.int8) if pL is None else
                            (pL[w, t0w, We[w, t0w]] > 0).astype(np.int8))
        rec["prefix_dev"] = np.abs(st_e[w, ib] - st_o[w, ib]).max(1)
        Xa, Xb_ = st_e[w, ia].astype(np.float64), st_e[w, ib].astype(np.float64)
        Xao, Xbo = st_o[w, ia].astype(np.float64), st_o[w, ib].astype(np.float64)
        pa, pb = anc_pos[w, ia], anc_pos[w, ib]
        Ga = _aug_cols(H_e, nll_e, lsm_ae[w, ia], st_e[w, ia], u, sdz, pa, w)
        Gb = _aug_cols(H_e, nll_e, lsm_ae[w, ib], st_e[w, ib], u, sdz, pb, w)
        Gao = _aug_cols(H_o, nll_o, lsm_ao[w, ia], st_o[w, ia], u, sdz, pa, w)
        Gbo = _aug_cols(H_o, nll_o, lsm_ao[w, ib], st_o[w, ib], u, sdz, pb, w)
        La, Sa = maps_np(st_e[w, ia])
        Lb, Sb = maps_np(st_e[w, ib])
        Lao, Sao = maps_np(st_o[w, ia])
        Lbo, Sbo = maps_np(st_o[w, ib])
        Za, Zb_ = _z_cols(lg_ae[w, ia]), _z_cols(lg_ae[w, ib])
        Zao, Zbo = _z_cols(lg_ao[w, ia]), _z_cols(lg_ao[w, ib])
        for nm in arm_names:
            Fa_, Fb_ = feat(nm, Xa, Ga, La, Sa, Za), feat(nm, Xb_, Gb, Lb, Sb, Zb_)
            Fao_, Fbo_ = feat(nm, Xao, Gao, Lao, Sao, Zao), feat(nm, Xbo, Gbo, Lbo, Sbo, Zbo)
            for l in L_STORE:
                for a in A_STORE:
                    n0, n1 = f"l{l}_d{a}", f"l{l}_d{a + 1}"
                    Va, Vb = vcol(nm, n0, Fa_), vcol(nm, n1, Fb_)
                    Vao, Vbo = vcol(nm, n0, Fao_), vcol(nm, n1, Fbo_)
                    rec[f"R_{nm}_l{l}_a{a}"] = Va - Vb
                    rec[f"Ro_{nm}_l{l}_a{a}"] = Vao - Vbo
                    rec[f"V_{nm}_l{l}_a{a}"] = Va
                    rec[f"Vpre_{nm}_l{l}_a{a}"] = Vb
                    rec[f"Vpreo_{nm}_l{l}_a{a}"] = Vbo
                    if not nm.startswith("clean"):
                        rec[f"Rsh_{nm}_l{l}_a{a}"] = (vcol(nm, f"sh_{n0}", Fa_)
                                                      - vcol(nm, f"sh_{n1}", Fb_))
            del Fa_, Fb_, Fao_, Fbo_
        if mlp:
            for l in L_STORE:
                Va, Vb = mlp_pred(f"l{l}_d0", Xa), mlp_pred(f"l{l}_d1", Xb_)
                Vao, Vbo = mlp_pred(f"l{l}_d0", Xao), mlp_pred(f"l{l}_d1", Xbo)
                rec[f"R_fullM_l{l}_a0"] = Va - Vb
                rec[f"Ro_fullM_l{l}_a0"] = Vao - Vbo
                rec[f"V_fullM_l{l}_a0"] = Va
                rec[f"Vpre_fullM_l{l}_a0"] = Vb
                rec[f"Vpreo_fullM_l{l}_a0"] = Vbo
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
        if do_probes:
            # junction/task.py's legality ceiling on the post-event state, fitted on EDITED
            # windows of the training split with lambda on the validation split, plus the
            # same probe on the INPUT EMBEDDING at the event (token + position only): the
            # guard that swap and rare are not told apart by the token alone.
            ed_m = rec["etype"] < 2
            trm, vam = (rec["split"] == 0) & ed_m, (rec["split"] == 1) & ed_m
            ysw = (rec["etype"] == 0).astype(np.float64)
            if trm.sum() > 100 and 0 < ysw[trm].mean() < 1:
                one_ = np.ones((len(w), 1))
                rec["probe_swap_ed"], r2e = _probe(np.concatenate([Xa, one_], 1), ysw, trm,
                                                   vam, LAMS)
                Ea = emb_ae[w, ia].astype(np.float64)
                rec["probe_swap_emb"], r2m = _probe(np.concatenate([Ea, one_], 1), ysw, trm,
                                                    vam, LAMS)
                gates.setdefault("probes", {})[an] = {"ed_val_r2": r2e, "emb_val_r2": r2m}
                if do_mlp:
                    from rhm.logit_reading.coeruleus.readout import train_head
                    hp = train_head(Xa[trm].astype(np.float32), ysw[trm].astype(np.float32),
                                    steps=mlp_steps, device=dev, seed=99)
                    rec["probe_swap_mlp"] = hp(Xa.astype(np.float32))
        tem = np.asarray(rec["split"]) == 2
        for k_, vv in rec.items():
            a_ = np.asarray(vv)[tem]
            save[f"{an}__{k_}"] = a_.astype(np.float32) if a_.dtype == np.float64 else a_

        # ---- the gates on this anchor --------------------------------------------------
        pre = f"{an}__"
        if not own_bank:
            print(f"{an}: {int(tem.sum())} test rows (no banked cell on this venue; the "
                  f"actor gate is the only gate)", flush=True)
            continue
        g = gates["rows"]
        g[f"{an}/w"] = bool(np.array_equal(save[pre + "w"], bank_npz[pre + "w"]))
        g[f"{an}/t0"] = bool(np.array_equal(save[pre + "t0"], bank_npz[pre + "t0"]))
        lab_ok = True
        for l in LEVELS:
            for a in A_LIST:
                for lk in ("oe", "oo", "cons", "ok"):
                    kk_ = f"{pre}{lk}_l{l}_a{a}"
                    if kk_ in bank_npz.files:
                        lab_ok &= bool(np.array_equal(save[kk_], bank_npz[kk_]))
        g[f"{an}/labels"] = lab_ok
        for lk in ("nll_e", "H_pre"):
            g[f"{an}/{lk}"] = float(np.abs(save[pre + lk].astype(np.float64)
                                           - bank_npz[pre + lk].astype(np.float64)).max())
        cg = gates["cols"]
        for nm in ("full", "clean"):
            worst = 0.0
            for l in L_STORE:
                for a in A_STORE:
                    for kind in ("V", "Vpre", "R", "Ro", "Vpreo") + (
                            ("Rsh",) if nm == "full" else ()):
                        kk_ = f"{pre}{kind}_{nm}_l{l}_a{a}"
                        if kk_ in bank_npz.files:
                            worst = max(worst, float(np.abs(
                                save[kk_].astype(np.float64)
                                - bank_npz[kk_].astype(np.float64)).max()))
            cg[f"{an}/{nm}"] = worst
        print(f"{an}: {int(tem.sum())} test rows  gates rows {g[f'{an}/w']} "
              f"{g[f'{an}/t0']} labels {lab_ok}  nll {g[f'{an}/nll_e']:.1e}  "
              f"H {g[f'{an}/H_pre']:.1e}  full {cg[f'{an}/full']:.2e}  "
              f"clean {cg[f'{an}/clean']:.2e}  ({time.time() - t_an:.0f}s)", flush=True)

    # ---- the twins: every arm's V at offsets 0 and -1, for both members --------------------
    if do_twins and own_bank and "tw__w" in bank_npz.files:
        t0 = time.time()
        tw = bank["twin"]
        tcols = {(c[0], int(c[1]), int(c[2])): i for i, c in enumerate(tw["cols"])}
        oi0, oim = tw["offs"].index(0), tw["offs"].index(-1)
        w = np.asarray(bank_npz["tw__w"]).astype(np.int64)
        tvp = np.asarray(bank_npz["tw__t_v"]).astype(np.int64)
        xc = np.asarray(bank_npz["tw__x_c"]).astype(np.int64)
        ar_ = np.arange(len(w))
        out_t = {}
        for side, Wside in (("v", We[w].copy()), ("t", None)):
            if side == "t":
                Wside = We[w].copy()
                Wside[ar_, tvp] = xc
            Hs = np.zeros((len(w), T), np.float32)
            nls = np.zeros((len(w), T), np.float32)
            st2 = np.zeros((len(w), 2, d), np.float32)
            ls2 = np.zeros((len(w), 2, v), np.float32)
            lg2 = np.zeros((len(w), 2, v), np.float32)
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
                    lg2[sl] = lg[ar, tt].float().cpu().numpy()
            for k_, (slot, p_) in {"pre": (0, tvp - 1), "ev": (1, tvp)}.items():
                G_ = _aug_cols(Hs, nls, ls2[:, slot], st2[:, slot], u, sdz, p_, ar_)
                L_, S_ = maps_np(st2[:, slot])
                Z_ = _z_cols(lg2[:, slot])
                Xs_ = st2[:, slot].astype(np.float64)
                for nm in arm_names:
                    for l in L_STORE:
                        dd = 1 if k_ == "pre" else 0
                        out_t[f"tw_{k_}_{nm}_l{l}_{side}"] = lin(nm, f"l{l}_d{dd}", Xs_, G_,
                                                                  L_, S_, Z_)
                if mlp:
                    for l in L_STORE:
                        dd = 1 if k_ == "pre" else 0
                        out_t[f"tw_{k_}_fullM_l{l}_{side}"] = mlp_pred(f"l{l}_d{dd}", Xs_)
        Vv, Vt = np.asarray(bank_npz["tw__V_viol"]), np.asarray(bank_npz["tw__V_twin"])
        worst = 0.0
        for nm in ("full", "clean"):
            for l in L_STORE:
                worst = max(worst,
                            float(np.abs(out_t[f"tw_ev_{nm}_l{l}_v"] - Vv[:, oi0, tcols[(nm, l, 0)]]).max()),
                            float(np.abs(out_t[f"tw_ev_{nm}_l{l}_t"] - Vt[:, oi0, tcols[(nm, l, 0)]]).max()),
                            float(np.abs(out_t[f"tw_pre_{nm}_l{l}_v"] - Vv[:, oim, tcols[(nm, l, 1)]]).max()))
        gates["cols"]["twins/arm0"] = worst
        save["tw_w"] = w
        for k_, vv in out_t.items():
            save[k_] = vv.astype(np.float32)
        print(f"twins {time.time() - t0:.1f}s  {len(w)} pairs  arm-0 gate {worst:.2e}",
              flush=True)

    # ---- follow-up 6: every column shared with a committed cell must reproduce it ----------
    for sx in [x for x in check_sfx.split(",") if x]:
        pth = f"{stem}_norm_{stim_tag}{sx}.npz"
        if not os.path.exists(pth):
            gates.setdefault("shared", {})[sx] = {"n": 0, "worst": float("nan"),
                                                  "missing": True}
            continue
        Zc = np.load(pth)
        shared = [k_ for k_ in Zc.files if k_ != "meta" and k_ in save]
        worst, wk = 0.0, ""
        for k_ in shared:
            a_, b_ = np.asarray(save[k_]), np.asarray(Zc[k_])
            if a_.shape != b_.shape:
                worst, wk = float("inf"), k_
                break
            dv = float(np.abs(a_.astype(np.float64) - b_.astype(np.float64)).max()) \
                if a_.size else 0.0
            if dv > worst:
                worst, wk = dv, k_
        gates.setdefault("shared", {})[sx] = {"n": len(shared), "worst": worst,
                                              "worst_key": wk}
        print(f"shared with {sx}: {len(shared)} columns, worst {worst:.2e} ({wk})", flush=True)

    # ---- verdict, save --------------------------------------------------------------------
    fails = []
    fails += [k_ for k_, x in gates["actor"].items() if x > 1e-9]
    fails += [k_ for k_, x in gates["val_r2"].items() if x > 1e-6]
    fails += [k_ for k_, x in gates["rows"].items() if (x is False) or
              (not isinstance(x, bool) and x > tol)]
    fails += [k_ for k_, x in gates["cols"].items() if x > tol]
    fails += [f"shared{k_}" for k_, x in gates.get("shared", {}).items()
              if x.get("missing") or not (x["worst"] <= tol) or x["n"] == 0]
    res = {"ckpt": ckpt, "step": cfg.get("step"), "stim_tag": stim_tag, "n": n,
           "actor_clean_acc": actor_tab, "val_r2": val_r2, "gates": gates,
           "gate_fails": fails, "arms": {"H": "state + H(q_t)", "Q": "state + log q_t",
                                         "N": "state + ||s_t||", "T": "state + tanh(u.s_t/sd)",
                                         "P": "state + H(q_t) + H(q_(t-1)) + s_t + "
                                              "H(q_(t-1)) s_t",
                                         "M": "MLP on the state (l1-4, a = 0)",
                                         "L": "ln_f(s) in place of the state",
                                         "Lp": "state + ln_f(s), ln block scaled as a block",
                                         "S": "affine-free per-row standardisation, in place",
                                         "Z": "state + the 16 raw logits z_t",
                                         "ZE": "state + z_t + logsumexp(z_t)",
                                         "E": "state + logsumexp(z_t)",
                                         "LE": "ln_f(s) in place + logsumexp(z_t)",
                                         "LX": "ln_f(s) in place + max(z_t)"},
           "arm_set": arm_set, "map_arms": map_arms, "bank_tag": bank_tag,
           "own_bank": own_bank, "do_probes": do_probes, "logit_arms": logit_arms,
           "check_sfx": check_sfx, "zaug_cols": ZAUG, "span": span,
           "aug_cols": AUG, "scale": scale_info, "u_seed": u_seed, "sdz": sdz,
           "mlp_steps": mlp_steps if do_mlp else 0}
    sfx = f"_norm_{stim_tag}{out_sfx}"
    with open(f"{stem}{sfx}.json", "w") as f:
        json.dump(res, f, cls=NumpyEncoder, indent=1)
    save["meta"] = np.array(json.dumps({"gate_fails": fails}))
    np.savez_compressed(f"{stem}{sfx}.npz", **save)
    volume.commit()
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    print(f"saved -> {stem}{sfx}.npz  ({time.time() - t00:.0f}s, peak RSS {rss:.1f} GB)  "
          f"gate fails: {fails}", flush=True)
    assert not fails, f"GATE FAILED: {fails}"
    return {"ckpt": ckpt, "stim_tag": stim_tag, "step": cfg.get("step"),
            "l1_d0": {nm: round(val_r2[nm].get("l1_d0", float("nan")), 4) for nm in val_r2},
            "gate_fails": fails}


@app.function(volumes={DATA_DIR: volume}, timeout=6 * 3600, memory=2048)
def express_sweep(seeds: str = "42,43,44", steps: str = "0,8000,64000",
                  tags: str = "swap65k,a1", do_mlp: bool = True, arm_set: str = "H,Q,N,T,P",
                  map_arms: str = "", out_sfx: str = "_express", bank_tag: str = "",
                  do_probes: bool = False, do_twins: bool = True, logit_arms: str = "",
                  check_sfx: str = ""):
    """One container per (checkpoint, venue), fanned out from this CPU coordinator.  The
    defaults reproduce the committed `_express` sweep."""
    volume.reload()
    D_ = f"{DATA_DIR}/v16_s2_L6_m4_distinct/logit_reading"
    args = []
    for s in seeds.split(","):
        for st in steps.split(","):
            ck = f"{D_}/traj_a1_s{s}/step{int(st):06d}.pt"
            for tg in tags.split(","):
                need = f"{ck[:-3]}_norm_{bank_tag or tg}." + ("json" if bank_tag else "npz")
                if not (os.path.exists(ck) and os.path.exists(need)):
                    print(f"MISSING {ck} {tg} ({need})", flush=True)
                    continue
                # (ckpt, stim_tag, n_clean, n_clean_val, seed, actor_steps, eval_seed,
                #  n_clean_crit, prim_block, n_val_windows, chunk, do_mlp, mlp_steps,
                #  do_twins, tol, u_seed, arm_set, map_arms, out_sfx, bank_tag, do_probes,
                #  logit_arms, check_sfx)
                args.append((ck, tg, 6144, 2048, 11, 3000, 5150, 8192, "post_block7", 1500,
                             512, do_mlp, 2000, do_twins, 5e-5, 4242, arm_set, map_arms,
                             out_sfx, bank_tag, do_probes, logit_arms, check_sfx))
    print(f"{len(args)} cells", flush=True)
    outs = list(express_ckpt.starmap(args, return_exceptions=True))
    for a, o in zip(args, outs):
        print(f"{a[0].split('/')[-2]}/{a[0].split('/')[-1]} {a[1]}: "
              f"{json.dumps(o, cls=NumpyEncoder, default=str)[:600]}", flush=True)
    return [str(o)[:600] for o in outs]
