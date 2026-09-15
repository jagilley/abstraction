"""Part 2: reading a grammar violation as something emotion-shaped.

Runs one frozen checkpoint against the model-independent stimuli from `stimuli.py`.

Events (token at window index t, read by the model at position t):
  viol     the first zero-probability token of a swap window (t = t_v), labelled k_star
  rare     the first changed token of a rare-synonym edit (legal, often surprising)
  natural  random legal tokens from unedited / rare windows, t in [16, 63]

Matching strata: token identity x token level x window-position bucket x model-surprisal
bin. Guards that must read ~0.5 on every matched set: `guard_surprisal` and
`state_post_embed` (a linear probe on token + position embeddings, i.e. on the matching
variables themselves plus within-bucket position).

Four properties, each against a stated reference:
 1. NORM, NOT SURPRISAL. Violations vs legal tokens under exact stratification on
    token level x window-position bucket x model-surprisal bin. The surprisal AUC is the
    guard (~0.5 by construction). Readouts, from least to most access:
      R1 = s - H(q_prev)           surprise against the model's own expected surprise
      logit probe                  [log q_prev, onehot(token), log q_next], linear and MLP
      state probe                  residual stream at t, every block (linear)
 2. LEARNED SCOPE. Everything above per k_star, plus an unmatched detection AUC (surprisal
    percentile among controls in the same level x position stratum). Read across checkpoints.
 3. STATE BEYOND LOGITS. state-probe AUC minus logit-probe AUC; and a k_star -> k_star
    transfer matrix for the state probe (one shared "something's wrong" axis, or per-level?).
 4. PERSISTENCE. For tau = 0..12 after the event, on the edited vs original window:
    the model's entropy change and its excess divergence from the eps-noise Bayesian,
    against that Bayesian's own entropy change. Violations by k_star; rare edits alongside.

Run:
  modal run -m rhm.logit_reading.violation::violation_ckpt \
      --ckpt /data/v16_s2_L6_m4_distinct/logit_reading/traj_a1_s42/step064000.pt --stim-tag a1
  modal run --detach -m rhm.logit_reading.violation::violation_sweep \
      --traj-dir /data/v16_s2_L6_m4_distinct/logit_reading/traj_a1_s42 --stim-tag a1
"""

import json
import os

import numpy as np

from rhm.shared import volume, DATA_DIR, NumpyEncoder
from rhm.logit_reading.calibration import app, load_trajectory_ckpt

BLOCKS = ["post_embed"] + [f"post_block{i}" for i in range(8)]
T_BUCKETS = [(1, 16), (16, 32), (32, 48), (48, 64)]


def auc(score, y):
    """Mann-Whitney AUC with ties."""
    from scipy.stats import rankdata
    score = np.asarray(score, np.float64)
    y = np.asarray(y).astype(bool)
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(score)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def match_strata(keys, y, rng):
    """Indices of a set with equal positives/negatives inside every stratum."""
    out = []
    for kk in np.unique(keys):
        pos = np.where((keys == kk) & y)[0]
        neg = np.where((keys == kk) & ~y)[0]
        c = min(len(pos), len(neg))
        if c == 0:
            continue
        out.append(rng.choice(pos, c, replace=False))
        out.append(rng.choice(neg, c, replace=False))
    return np.concatenate(out) if out else np.zeros(0, dtype=np.int64)


def fit_probe(Xtr, ytr, hidden=0, wd=1e-3, steps=400, lr=1e-2, device="cuda", seed=0):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    Xtr = torch.as_tensor(Xtr, dtype=torch.float32, device=device)
    ytr = torch.as_tensor(ytr, dtype=torch.float32, device=device)
    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True) + 1e-5
    d = Xtr.shape[1]
    net = (nn.Linear(d, 1) if hidden == 0 else
           nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, 1))).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    Xn = (Xtr - mu) / sd
    for _ in range(steps):
        loss = nn.functional.binary_cross_entropy_with_logits(net(Xn)[:, 0], ytr)
        opt.zero_grad()
        loss.backward()
        opt.step()

    def predict(X):
        with torch.no_grad():
            X = torch.as_tensor(X, dtype=torch.float32, device=device)
            return net((X - mu) / sd)[:, 0].cpu().numpy()
    return predict


def build_events(S, rng, n_ctrl=4, t_lo=16):
    n, T1 = S["windows_edit"].shape
    T = T1 - 1
    w_l, t_l, kind_l, k_l = [], [], [], []
    for w in range(n):
        et = S["etype"][w]
        if et == 0 and 1 <= S["t_v"][w] <= T - 1:
            w_l.append(w); t_l.append(S["t_v"][w]); kind_l.append(0); k_l.append(S["k_star"][w])
        if et == 1 and 1 <= S["first_diff"][w] <= T - 1:
            w_l.append(w); t_l.append(S["first_diff"][w]); kind_l.append(1); k_l.append(-1)
        if et in (1, 2):
            for t in rng.choice(np.arange(t_lo, T), n_ctrl - (et == 1), replace=False):
                w_l.append(w); t_l.append(int(t)); kind_l.append(2); k_l.append(-1)
    return (np.array(w_l), np.array(t_l), np.array(kind_l), np.array(k_l))


def _ent(p):
    return -(p * np.log(np.clip(p, 1e-30, None))).sum(-1)


def _kl(p, logq):
    return (p * (np.log(np.clip(p, 1e-30, None)) - logq)).sum(-1)


def readouts(model, S, device="cuda", n_ctrl=8, seed=0, n_surpr_bins=8, tau_max=12):
    import torch
    from rhm.logit_reading.flat_oracle import leaf_levels

    rng = np.random.default_rng(seed)
    We, Wo = S["windows_edit"], S["windows_orig"]
    n, T1 = We.shape
    T = T1 - 1
    L = S["ptok"].shape[-1] - 1
    s_branch = 2
    ev_w, ev_t, ev_kind, ev_k = build_events(S, rng, n_ctrl)
    order = np.argsort(ev_w, kind="stable")
    ev_w, ev_t, ev_kind, ev_k = ev_w[order], ev_t[order], ev_kind[order], ev_k[order]

    # ---- forward passes ----
    logq_e = np.empty((n, T, 16), np.float32)
    logq_o = np.empty((n, T, 16), np.float32)
    states = {b: np.empty((len(ev_w), model.transformer.wte.weight.shape[1]), np.float32)
              for b in BLOCKS}
    bs = 256
    with torch.no_grad():
        for c0 in range(0, n, bs):
            xe = torch.as_tensor(We[c0:c0 + bs, :T], device=device)
            xo = torch.as_tensor(Wo[c0:c0 + bs, :T], device=device)
            le, _, inter = model(xe, return_intermediates=True)
            lo, _ = model(xo)
            logq_e[c0:c0 + bs] = torch.log_softmax(le.float(), -1).cpu().numpy()
            logq_o[c0:c0 + bs] = torch.log_softmax(lo.float(), -1).cpu().numpy()
            sel = np.where((ev_w >= c0) & (ev_w < c0 + bs))[0]
            if len(sel):
                wi = torch.as_tensor(ev_w[sel] - c0, device=device)
                ti = torch.as_tensor(ev_t[sel], device=device)
                for b in BLOCKS:
                    states[b][sel] = inter[b][wi, ti].float().cpu().numpy()

    tok = We[ev_w, ev_t]
    lqp = logq_e[ev_w, ev_t - 1]
    lqn = logq_e[ev_w, ev_t]
    surpr = -lqp[np.arange(len(tok)), tok]
    H_prev = _ent(np.exp(lqp))
    H_next = _ent(np.exp(lqn))
    level = leaf_levels((S["phase"][ev_w] + ev_t) % T, s_branch, L)
    tb = np.digitize(ev_t, [b[0] for b in T_BUCKETS[1:]])
    y = ev_kind == 0
    X_logit = np.concatenate([lqp, np.eye(16, dtype=np.float32)[tok], lqn], 1)

    edges = np.quantile(surpr, np.linspace(0, 1, n_surpr_bins + 1)[1:-1])
    sbin = np.digitize(surpr, edges)
    # token identity is in the strata: swap edits draw a different token mix than natural
    # text, and a linear probe on the input embedding reads that at AUC ~0.69 otherwise.
    # `state_post_embed` (token + position embedding only) is therefore the confound guard.
    strata = ((tok * 10 + level) * 10 + tb) * 100 + sbin
    train_win = np.random.default_rng(seed + 1).random(n) < 0.6
    is_tr = train_win[ev_w]

    res = {"n_events": {"viol": int(y.sum()), "rare": int((ev_kind == 1).sum()),
                        "natural": int((ev_kind == 2).sum())}}

    # ---- unmatched detection by k_star: surprisal percentile within level x t-bucket ----
    det = {}
    ctrl = ~y
    lvl_tb = level * 10 + tb
    pct = np.full(len(y), np.nan)
    for kk in np.unique(lvl_tb):
        cs = np.sort(surpr[ctrl & (lvl_tb == kk)])
        vi = np.where(y & (lvl_tb == kk))[0]
        if len(cs) and len(vi):
            lo = np.searchsorted(cs, surpr[vi], "left")
            hi = np.searchsorted(cs, surpr[vi], "right")
            pct[vi] = (lo + hi) / 2 / len(cs)
    ptok = S["ptok"]
    for k in range(1, L + 1):
        sel = y & (ev_k == k)
        if sel.sum() < 5:
            continue
        below = ptok[ev_w[sel], ev_t[sel], :k]                        # observers that allow it
        det[k] = {"n": int(sel.sum()),
                  "detect_auc_stratified": float(np.nanmean(pct[sel])),
                  "mean_surprisal": float(surpr[sel].mean()),
                  "mean_neglogp_k_below": [float(-np.log(np.clip(below[:, kk], 1e-30, None)).mean())
                                           for kk in range(k)],
                  "mean_H_prev": float(H_prev[sel].mean())}
    det["controls"] = {"mean_surprisal": float(surpr[ctrl].mean()),
                       "rare_mean_surprisal": float(surpr[ev_kind == 1].mean())}
    res["detection_by_kstar"] = det

    # ---- matched: probes trained on the matched train set, scored on matched test sets ----
    m_tr = match_strata(np.where(is_tr, strata, -1), np.where(is_tr, y, False), rng)
    m_tr = m_tr[is_tr[m_tr]]
    probes = {"logit_linear": fit_probe(X_logit[m_tr], y[m_tr]),
              "logit_mlp": fit_probe(X_logit[m_tr], y[m_tr], hidden=256, steps=800)}
    for b in BLOCKS:
        probes[f"state_{b}"] = fit_probe(states[b][m_tr], y[m_tr])

    def score_set(idx):
        out = {"n": int(len(idx)), "guard_surprisal": auc(surpr[idx], y[idx]),
               "R1_s_minus_H": auc(surpr[idx] - H_prev[idx], y[idx]),
               "neg_H_prev": auc(-H_prev[idx], y[idx]),
               "H_next_minus_H_prev": auc(H_next[idx] - H_prev[idx], y[idx])}
        for name, f in probes.items():
            feats = X_logit if name.startswith("logit") else states[name[len("state_"):]]
            out[name] = auc(f(feats[idx]), y[idx])
        return out

    te_keys = np.where(~is_tr, strata, -1)
    matched = {"all": score_set(match_strata(te_keys, np.where(~is_tr, y, False), rng))}
    for k in range(1, L + 1):
        keep = (~is_tr) & ((ev_kind != 0) | (ev_k == k))
        idx = match_strata(np.where(keep, strata, -1), np.where(keep, y, False), rng)
        if (y[idx]).sum() >= 20:
            matched[f"k{k}"] = score_set(idx)
    # legal-rare only as the negative class (the sharpest control)
    keep = (~is_tr) & (ev_kind != 2)
    matched["vs_rare_only"] = score_set(match_strata(np.where(keep, strata, -1),
                                                     np.where(keep, y, False), rng))
    res["matched"] = matched
    state_names = [f"state_{b}" for b in BLOCKS]
    best_block = max(state_names, key=lambda nm: matched["all"][nm])
    res["best_state_block"] = best_block
    res["state_minus_logit_mlp"] = matched["all"][best_block] - matched["all"]["logit_mlp"]

    # ---- transfer matrix of the best-block state probe across k_star ----
    bX = states[best_block[len("state_"):]]
    transfer = {}
    ks = [k for k in range(1, L + 1) if f"k{k}" in matched]
    sets = {}
    for k in ks:
        for split, flag in (("tr", is_tr), ("te", ~is_tr)):
            keep = flag & ((ev_kind != 0) | (ev_k == k))
            sets[(k, split)] = match_strata(np.where(keep, strata, -1), np.where(keep, y, False), rng)
    for a in ks:
        if y[sets[(a, "tr")]].sum() < 40:
            continue
        f = fit_probe(bX[sets[(a, "tr")]], y[sets[(a, "tr")]])
        transfer[a] = {b: auc(f(bX[sets[(b, "te")]]), y[sets[(b, "te")]]) for b in ks}
    res["transfer_best_block"] = transfer

    # ---- persistence ----
    pE_e, pE_o = S["pE_edit"], S["pE_orig"]
    pers = {}
    groups = {f"viol_k{k}": (ev_kind == 0) & (ev_k == k) for k in range(1, L + 1)}
    groups["viol_all"] = ev_kind == 0
    groups["rare"] = ev_kind == 1
    for g, sel in groups.items():
        idx = np.where(sel)[0]
        if len(idx) < 20:
            continue
        rows = []
        for tau in range(tau_max + 1):
            t = ev_t[idx] + tau
            ok = t <= T - 1
            ii, tt = ev_w[idx][ok], t[ok]
            if len(ii) < 10:
                break
            qe, qo = np.exp(logq_e[ii, tt]), np.exp(logq_o[ii, tt])
            be, bo = pE_e[ii, tt + 1], pE_o[ii, tt + 1]
            rows.append({
                "tau": tau, "n": int(len(ii)),
                "dH_model": float((_ent(qe) - _ent(qo)).mean()),
                "dH_bayes_eps": float((_ent(be) - _ent(bo)).mean()),
                "excess_KL_model_from_bayes_eps": float((_kl(be, logq_e[ii, tt]) - _kl(bo, logq_o[ii, tt])).mean()),
                "KL_model_orig_edit": float(_kl(qo, logq_e[ii, tt]).mean()),
                "KL_bayes_eps_orig_edit": float(_kl(bo, np.log(np.clip(be, 1e-30, None))).mean()),
            })
        pers[g] = rows
    res["persistence"] = pers
    return res


# ===========================================================================
# v2 matching. v1 (`readouts`) is kept for its recorded numbers, but its guards failed:
# 8 surprisal quantile bins leak in the tail (guard_surprisal drifts 0.50 -> 0.59 over
# training), and 16-wide window-position buckets leak position (the token + position
# embedding probe reads 0.56-0.61 at every checkpoint, including random init). v2 draws
# controls from EVERY legal token, matches exactly on (token, token level, window index)
# and nearest-neighbour on model surprisal within a caliper, without replacement.
# ===========================================================================

def nn_match(v_idx, v_cell, v_s, c_idx, c_cell, c_s, caliper, rng):
    """Greedy nearest-neighbour matching inside exact cells. Returns (viol, ctrl) index pairs."""
    order = np.lexsort((c_s, c_cell))
    c_idx, c_cell, c_s = c_idx[order], c_cell[order], c_s[order]
    cells, starts = np.unique(c_cell, return_index=True)
    ends = np.append(starts[1:], len(c_cell))
    span = {int(c): (a, b) for c, a, b in zip(cells, starts, ends)}
    used = np.zeros(len(c_idx), dtype=bool)
    pv, pc = [], []
    for i in rng.permutation(len(v_idx)):
        if int(v_cell[i]) not in span:
            continue
        a, b = span[int(v_cell[i])]
        cs = c_s[a:b]
        j = np.searchsorted(cs, v_s[i])
        best, bd = -1, caliper
        lo, hi = j - 1, j
        while lo >= 0 and v_s[i] - cs[lo] <= bd:
            if not used[a + lo]:
                best, bd = a + lo, v_s[i] - cs[lo]
                break
            lo -= 1
        while hi < len(cs) and cs[hi] - v_s[i] <= bd:
            if not used[a + hi]:
                if cs[hi] - v_s[i] < bd or best < 0:
                    best, bd = a + hi, cs[hi] - v_s[i]
                break
            hi += 1
        if best >= 0:
            used[best] = True
            pv.append(v_idx[i])
            pc.append(c_idx[best])
    return np.array(pv, dtype=np.int64), np.array(pc, dtype=np.int64)


def readouts_v2(model, S, device="cuda", seed=0, caliper=0.1, t_lo=16,
                offsets=(-1, 0, 1, 2, 4, 8), tau_max=12):
    import torch
    from rhm.logit_reading.flat_oracle import leaf_levels

    rng = np.random.default_rng(seed)
    We, Wo = S["windows_edit"], S["windows_orig"]
    n, T1 = We.shape
    T = T1 - 1
    L = S["ptok"].shape[-1] - 1
    v = int(We.max()) + 1
    d_model = model.transformer.wte.weight.shape[1]

    # ---- pass 1: logits everywhere ----
    logq_e = np.empty((n, T, v), np.float32)
    logq_o = np.empty((n, T, v), np.float32)
    with torch.no_grad():
        for c0 in range(0, n, 512):
            le, _ = model(torch.as_tensor(We[c0:c0 + 512, :T], device=device))
            lo, _ = model(torch.as_tensor(Wo[c0:c0 + 512, :T], device=device))
            logq_e[c0:c0 + 512] = torch.log_softmax(le.float(), -1).cpu().numpy()
            logq_o[c0:c0 + 512] = torch.log_softmax(lo.float(), -1).cpu().numpy()

    # ---- candidate events ----
    et, tv, ks, fd = S["etype"], S["t_v"], S["k_star"], S["first_diff"]
    vw = np.where((et == 0) & (tv >= t_lo) & (tv <= T - 1))[0]
    ev = {"w": vw, "t": tv[vw], "k": ks[vw]}
    rw = np.where((et == 1) & (fd >= t_lo) & (fd <= T - 1))[0]
    cw = np.where(et != 0)[0]
    ctrl_w = np.repeat(cw, T - t_lo)
    ctrl_t = np.tile(np.arange(t_lo, T), len(cw))
    # all event records in one table: kind 0 viol, 1 rare, 2 legal token
    W_ = np.concatenate([vw, rw, ctrl_w])
    T_ = np.concatenate([tv[vw], fd[rw], ctrl_t])
    K_ = np.concatenate([np.zeros(len(vw)), np.ones(len(rw)), np.full(len(ctrl_w), 2)]).astype(int)
    KS_ = np.concatenate([ks[vw], np.full(len(rw) + len(ctrl_w), -1)])
    tok = We[W_, T_]
    lqp = logq_e[W_, T_ - 1]
    lqn = logq_e[W_, T_]
    surpr = -lqp[np.arange(len(tok)), tok]
    H_prev = _ent(np.exp(lqp))
    H_next = _ent(np.exp(lqn))
    level = leaf_levels((S["phase"][W_] + T_) % T, 2, L)
    cell = (tok * 10 + level) * 100 + T_
    train_win = np.random.default_rng(seed + 1).random(n) < 0.6
    is_tr = train_win[W_]

    res = {"caliper": caliper, "n_candidates": {"viol": int(len(vw)), "rare": int(len(rw)),
                                                  "legal_tokens": int(len(ctrl_w))}}

    # ---- unmatched detection by k_star (surprisal percentile among legal tokens in cell) ----
    det = {}
    is_c = K_ == 2
    cell_lt = level * 100 + T_                      # level x exact window index
    order = np.lexsort((surpr[is_c], cell_lt[is_c]))
    c_cell_s, c_s_s = cell_lt[is_c][order], surpr[is_c][order]
    cells, starts = np.unique(c_cell_s, return_index=True)
    ends = np.append(starts[1:], len(c_cell_s))
    span = {int(c): (a, b) for c, a, b in zip(cells, starts, ends)}
    pct = np.full(len(W_), np.nan)
    for i in np.where(K_ == 0)[0]:
        if int(cell_lt[i]) in span:
            a, b = span[int(cell_lt[i])]
            cs = c_s_s[a:b]
            pct[i] = (np.searchsorted(cs, surpr[i], "left") + np.searchsorted(cs, surpr[i], "right")) / 2 / len(cs)
    for k in range(1, L + 1):
        sel = (K_ == 0) & (KS_ == k)
        if sel.sum() >= 5:
            det[k] = {"n": int(sel.sum()), "detect_auc": float(np.nanmean(pct[sel])),
                      "mean_surprisal": float(surpr[sel].mean())}
    res["detection_by_kstar"] = det

    # ---- matching ----
    def match(pos_mask, neg_mask, cell_arr, cal):
        out = {}
        for split, flag in (("tr", is_tr), ("te", ~is_tr)):
            p = np.where(pos_mask & flag)[0]
            q = np.where(neg_mask & flag)[0]
            out[split] = nn_match(p, cell_arr[p], surpr[p], q, cell_arr[q], surpr[q], cal, rng)
        return out

    M_legal = match(K_ == 0, K_ == 2, cell, caliper)
    cell_coarse = (tok * 10 + level) * 100 + T_ // 8
    M_rare = match(K_ == 0, K_ == 1, cell_coarse, 2 * caliper)
    cov = {}
    for k in range(1, L + 1):
        tot = ((K_ == 0) & (KS_ == k)).sum()
        got = np.isin(np.concatenate([M_legal["tr"][0], M_legal["te"][0]]),
                      np.where(KS_ == k)[0]).sum()
        if tot:
            cov[k] = {"n": int(tot), "matched_frac": float(got / tot)}
    res["match_coverage_by_kstar"] = cov

    # ---- pass 2: residual states at matched events, at several offsets ----
    need = np.unique(np.concatenate([M_legal[s_][i] for s_ in ("tr", "te") for i in (0, 1)]
                                    + [M_rare[s_][i] for s_ in ("tr", "te") for i in (0, 1)]))
    row_of = {int(e): r for r, e in enumerate(need)}
    states = {o: {b: np.full((len(need), d_model), np.nan, np.float32) for b in BLOCKS} for o in offsets}
    by_w = {}
    for r, e in enumerate(need):
        by_w.setdefault(int(W_[e]), []).append(r)
    wins = np.array(sorted(by_w))
    with torch.no_grad():
        for c0 in range(0, len(wins), 256):
            ws = wins[c0:c0 + 256]
            _, _, inter = model(torch.as_tensor(We[ws, :T], device=device), return_intermediates=True)
            rows = [r for w in ws for r in by_w[int(w)]]
            wi_local = np.array([i for i, w in enumerate(ws) for _ in by_w[int(w)]])
            tt = T_[need[rows]]
            for o in offsets:
                ok = (tt + o >= 0) & (tt + o <= T - 1)
                if not ok.any():
                    continue
                wi_t = torch.as_tensor(wi_local[ok], device=device)
                ti_t = torch.as_tensor(tt[ok] + o, device=device)
                rr = np.array(rows)[ok]
                for b in BLOCKS:
                    states[o][b][rr] = inter[b][wi_t, ti_t].float().cpu().numpy()

    def pairs_xy(M, split):
        pv, pc = M[split]
        idx = np.concatenate([pv, pc])
        yy = np.concatenate([np.ones(len(pv), bool), np.zeros(len(pc), bool)])
        return idx, yy

    X_pre = np.concatenate([lqp, np.eye(v, dtype=np.float32)[tok]], 1)
    X_post = lqn
    X_both = np.concatenate([X_pre, X_post], 1)
    X_prev_only = lqp                                   # context forecast, token withheld
    # the model's own running surprisal over the 15 tokens before the event (tonic signal)
    u = T_[:, None] - 1 - np.arange(15)[None, :]                     # token indices t-1..t-15
    s_run = -logq_e[W_[:, None], u - 1, We[W_[:, None], u]]          # (N_ev, 15)
    X_run = s_run

    def st(o, b, idx):
        return states[o][b][[row_of[int(e)] for e in idx]]

    def evaluate(M, tag):
        itr, ytr = pairs_xy(M, "tr")
        ite, yte = pairs_xy(M, "te")
        out = {"n_pairs_train": int(ytr.sum()), "n_pairs_test": int(yte.sum()),
               "mean_abs_ds_test": float(np.abs(surpr[M["te"][0]] - surpr[M["te"][1]]).mean())
               if len(M["te"][0]) else float("nan")}
        if ytr.sum() < 50 or yte.sum() < 50:
            return out, None
        probes = {}
        for name, X in (("pre_logit_mlp", X_pre), ("post_logit_mlp", X_post), ("logit_mlp", X_both),
                        ("prev_logit_only_mlp", X_prev_only), ("run_surprisal_mlp", X_run)):
            probes[name] = (lambda X_: (lambda idx: X_[idx]))(X), fit_probe(X[itr], ytr, hidden=256, steps=800)
        for b in BLOCKS:
            f = fit_probe(st(0, b, itr), ytr)
            probes[f"state_{b}"] = (lambda b_: (lambda idx: st(0, b_, idx)))(b), f
        scal = {"guard_surprisal": surpr, "R1_s_minus_H": surpr - H_prev, "neg_H_prev": -H_prev,
                "H_next_minus_H_prev": H_next - H_prev,
                "run_surprisal_4": s_run[:, :4].mean(1), "run_surprisal_15": s_run.mean(1)}

        def score(idx, yy):
            o = {nm: auc(a[idx], yy) for nm, a in scal.items()}
            for nm, (getx, f) in probes.items():
                o[nm] = auc(f(getx(idx)), yy)
            return o

        out["all"] = score(ite, yte)
        bb = max(BLOCKS[1:], key=lambda b: out["all"][f"state_{b}"])
        out["best_block"] = bb
        # phasic vs tonic in the best block: state at t-1 alone, and [t-1, t] together
        ok_tr = ~np.isnan(st(-1, bb, itr)[:, 0])
        ok_te = ~np.isnan(st(-1, bb, ite)[:, 0])
        f_tm1 = fit_probe(st(-1, bb, itr[ok_tr]), ytr[ok_tr])
        cat = lambda idx: np.concatenate([st(-1, bb, idx), st(0, bb, idx)], 1)
        f_cat = fit_probe(cat(itr[ok_tr]), ytr[ok_tr])
        out["phasic_tonic"] = {"state_tm1": auc(f_tm1(st(-1, bb, ite[ok_te])), yte[ok_te]),
                               "state_t": out["all"][f"state_{bb}"],
                               "state_tm1_and_t": auc(f_cat(cat(ite[ok_te])), yte[ok_te])}
        probes["state_tm1_best"] = ((lambda idx: st(-1, bb, idx)), f_tm1)
        probes["state_tm1_and_t_best"] = (cat, f_cat)
        # offsets: anticipation (t-1) and persistence (t+tau) in the best block
        off = {}
        for o in offsets:
            if o == 0:
                continue
            ok_tr = ~np.isnan(st(o, bb, itr)[:, 0])
            ok_te = ~np.isnan(st(o, bb, ite)[:, 0])
            if ok_tr.sum() < 100 or ok_te.sum() < 100:
                continue
            f = fit_probe(st(o, bb, itr[ok_tr]), ytr[ok_tr])
            off[o] = {"auc": auc(f(st(o, bb, ite[ok_te])), yte[ok_te]), "n_test": int(ok_te.sum())}
        out["state_offsets_best_block"] = off
        # per k_star subsets of the matched test pairs
        byk = {}
        pv, pc = M["te"]
        for k in range(1, L + 1):
            sel = KS_[pv] == k
            if sel.sum() < 30:
                continue
            idx = np.concatenate([pv[sel], pc[sel]])
            yy = np.concatenate([np.ones(sel.sum(), bool), np.zeros(sel.sum(), bool)])
            byk[k] = {"n_pairs": int(sel.sum()), **score(idx, yy)}
        out["by_kstar"] = byk
        return out, (bb, probes)

    res["matched_vs_legal"], extra = evaluate(M_legal, "legal")
    res["matched_vs_rare"], _ = evaluate(M_rare, "rare")

    # transfer matrix of the best-block state probe across k_star (legal controls)
    transfer = {}
    if extra is not None:
        bb = extra[0]
        sets = {}
        for split in ("tr", "te"):
            pv, pc = M_legal[split]
            for k in range(1, L + 1):
                sel = KS_[pv] == k
                sets[(k, split)] = (np.concatenate([pv[sel], pc[sel]]),
                                    np.concatenate([np.ones(sel.sum(), bool), np.zeros(sel.sum(), bool)]))
        for a in range(1, L + 1):
            ia, ya = sets[(a, "tr")]
            if ya.sum() < 80:
                continue
            f = fit_probe(st(0, bb, ia), ya)
            transfer[a] = {b: auc(f(st(0, bb, sets[(b, "te")][0])), sets[(b, "te")][1])
                           for b in range(1, L + 1) if sets[(b, "te")][1].sum() >= 30}
    res["transfer_best_block"] = transfer

    # ---- output-level persistence (unmatched; eps-noise Bayes reference) ----
    pE_e, pE_o = S["pE_edit"], S["pE_orig"]
    pers = {}
    groups = {f"viol_k{k}": (et == 0) & (tv >= t_lo) & (tv <= T - 1) & (ks == k) for k in range(1, L + 1)}
    groups["rare"] = (et == 1) & (fd >= t_lo) & (fd <= T - 1)
    for g, sel in groups.items():
        ws = np.where(sel)[0]
        t0 = tv[ws] if g.startswith("viol") else fd[ws]
        if len(ws) < 20:
            continue
        rows = []
        for tau in range(tau_max + 1):
            ok = t0 + tau <= T - 1
            ii, tt = ws[ok], t0[ok] + tau
            if len(ii) < 10:
                break
            qe, qo = np.exp(logq_e[ii, tt]), np.exp(logq_o[ii, tt])
            be, bo = pE_e[ii, tt + 1], pE_o[ii, tt + 1]
            rows.append({"tau": tau, "n": int(len(ii)),
                         "dH_model": float((_ent(qe) - _ent(qo)).mean()),
                         "dH_bayes_eps": float((_ent(be) - _ent(bo)).mean()),
                         "KL_model_orig_edit": float(_kl(qo, logq_e[ii, tt]).mean()),
                         "KL_bayes_eps_orig_edit": float(_kl(bo, np.log(np.clip(be, 1e-30, None))).mean())})
        pers[g] = rows
    res["persistence"] = pers
    return res


def _load_stimuli(stim_tag, key="v16_s2_L6_m4_distinct"):
    z = np.load(f"{DATA_DIR}/{key}/logit_reading/stimuli_{stim_tag}.npz")
    return {k: z[k] for k in z.files}


@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=3 * 3600, memory=16384, max_containers=4)
def violation_ckpt(ckpt: str, stim_tag: str = "a1", seed: int = 0):
    volume.reload()
    model, rules, rule_w, cfg = load_trajectory_ckpt(ckpt, "cuda")
    S = _load_stimuli(stim_tag)
    res = readouts(model, S, "cuda", seed=seed)
    res["config"] = {"ckpt": ckpt, "step": cfg.get("step"), "stim_tag": stim_tag, "seed": seed}
    m = res["matched"]["all"]
    print(f"step {cfg.get('step')}  events {res['n_events']}  best {res['best_state_block']}", flush=True)
    print("  matched all: " + "  ".join(f"{k} {v:.3f}" for k, v in m.items() if k != "n"), flush=True)
    for k, d in res["detection_by_kstar"].items():
        if k != "controls":
            mk = res["matched"].get(f"k{k}", {})
            print(f"  k*={k} n={d['n']:4d} detect {d['detect_auc_stratified']:.3f} s {d['mean_surprisal']:.2f}"
                  f"  matched R1 {mk.get('R1_s_minus_H', float('nan')):.3f} logit_mlp "
                  f"{mk.get('logit_mlp', float('nan')):.3f} best_state "
                  f"{mk.get(res['best_state_block'], float('nan')):.3f}", flush=True)
    with open(f"{ckpt[:-3]}_violation_{stim_tag}.json", "w") as f:
        json.dump(res, f, cls=NumpyEncoder, indent=1)
    volume.commit()
    return res


@app.function(volumes={DATA_DIR: volume}, timeout=6 * 3600, memory=4096)
def violation_sweep(traj_dir: str, stim_tag: str = "a1"):
    import glob
    volume.reload()
    ckpts = sorted(glob.glob(f"{traj_dir}/step*.pt"))
    rows = list(violation_ckpt.map(ckpts, kwargs={"stim_tag": stim_tag}))
    with open(f"{traj_dir}/violation_sweep_{stim_tag}.json", "w") as f:
        json.dump(rows, f, cls=NumpyEncoder, indent=1)
    volume.commit()
    return len(rows)


@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=3 * 3600, memory=32768, max_containers=4)
def violation_ckpt_v2(ckpt: str, stim_tag: str = "a1", seed: int = 0, caliper: float = 0.1):
    volume.reload()
    model, rules, rule_w, cfg = load_trajectory_ckpt(ckpt, "cuda")
    S = _load_stimuli(stim_tag)
    res = readouts_v2(model, S, "cuda", seed=seed, caliper=caliper)
    res["config"] = {"ckpt": ckpt, "step": cfg.get("step"), "stim_tag": stim_tag, "seed": seed}
    m = res["matched_vs_legal"]
    print(f"step {cfg.get('step')}  pairs tr/te {m['n_pairs_train']}/{m['n_pairs_test']}  "
          f"|ds| {m['mean_abs_ds_test']:.4f}  best {m.get('best_block')}", flush=True)
    if "all" in m:
        print("  vs legal: " + "  ".join(f"{k} {x:.3f}" for k, x in m["all"].items()), flush=True)
        print(f"  offsets {m['state_offsets_best_block']}", flush=True)
    with open(f"{ckpt[:-3]}_violation2_{stim_tag}.json", "w") as f:
        json.dump(res, f, cls=NumpyEncoder, indent=1)
    volume.commit()
    return {"step": cfg.get("step")}


@app.function(volumes={DATA_DIR: volume}, timeout=6 * 3600, memory=4096)
def violation_sweep_v2(traj_dir: str, stim_tag: str = "a1"):
    import glob
    volume.reload()
    ckpts = sorted(glob.glob(f"{traj_dir}/step*.pt"))
    rows = list(violation_ckpt_v2.map(ckpts, kwargs={"stim_tag": stim_tag}))
    return len(rows)
