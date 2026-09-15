"""Part 2, follow-up: is there a PHASIC response to the violating token itself?

`violation.readouts_v2` matched each violation to a legal token from a different window
(same token, level, window index, model surprisal). The separation it found was already
present in the state at t-1 -- a tonic "this stretch has been off" signal built from the
edited material that precedes a high-level violation -- and the violating token added
~nothing (h[t-1] 0.818 vs h[t-1, t] 0.821 at 64k).

That design cannot see a phasic response, because the contexts differ. This one holds
the context fixed exactly. For every violation (window w, index t_v, token x_v):

    twin = the same window with x_v replaced by a LEGAL token x_c,
           legal = exact p_L(x_c | w_<t_v) > 0 (the prefix before t_v is legal by
           definition of t_v, so its exact predictive is on file from `stimuli.py`),
           chosen as the legal token whose MODEL surprisal is nearest x_v's, within a caliper.

The model is causal, so everything up to and including position t_v - 1 is identical
within a pair (checked). Any readout that separates the pair at position t_v reads the
model's response to WHICH token arrived, at matched surprisal, with the tonic signal
cancelled by construction.

Readouts at t_v, each scored as pooled AUC and as the paired win rate (fraction of pairs
where the violation scores higher):
  guard_surprisal       ~0.5 by the caliper
  state_tm1_guard       0.5 exactly (identical states)
  state_post_embed      token identity only (position is shared) -- the token-mix guard.
                        Pairs are subsampled so every token is the violator as often as the
                        twin (`balance_tokens`), which pins this guard's pooled AUC at 0.5.
  pre_logit_mlp         [log q_prev, onehot(token)]: an output-level reader told the forecast
                        and the outcome
  H_next, KL_update     the model's forecast after the token: its entropy, and how far it moved
  post_logit_mlp        log q_next: the whole post-token forecast
  state_<block>         residual stream at t_v, linear; MLP on the best block

Run:
  modal run -m rhm.logit_reading.phasic::phasic_ckpt \
      --ckpt /data/v16_s2_L6_m4_distinct/logit_reading/traj_a1_s42/step064000.pt
  modal run --detach -m rhm.logit_reading.phasic::phasic_sweep \
      --traj-dir /data/v16_s2_L6_m4_distinct/logit_reading/traj_a1_s42
"""

import json

import numpy as np

from rhm.shared import volume, DATA_DIR, NumpyEncoder
from rhm.logit_reading.calibration import app, load_trajectory_ckpt
from rhm.logit_reading.violation import BLOCKS, auc, fit_probe, _ent, _kl, _load_stimuli


def paired_win(score_v, score_c):
    d = np.asarray(score_v, np.float64) - np.asarray(score_c, np.float64)
    return float(((d > 0) + 0.5 * (d == 0)).mean()) if len(d) else float("nan")


def balance_tokens(a, b, idx, rng):
    """Subset of pair indices `idx` (edges a[i] -> b[i]) in which every token appears as
    often on the violation side as on the twin side, by greedy cycle extraction. On such a
    subset token identity alone cannot separate the classes (pooled AUC exactly 0.5 for any
    function of the token)."""
    from collections import defaultdict
    out_edges = defaultdict(list)
    for i in rng.permutation(idx):
        out_edges[int(a[i])].append(int(i))
    keep = []
    while True:
        start = next((u for u in out_edges if out_edges[u]), None)
        if start is None:
            break
        path, seen, u = [], {}, start
        while out_edges[u] and u not in seen:
            seen[u] = len(path)
            e = out_edges[u].pop()
            path.append(e)
            u = int(b[e])
        if u in seen:                        # closed a cycle: keep it, return the tail edges
            cyc = path[seen[u]:]
            keep.extend(cyc)
            for e in path[:seen[u]]:
                out_edges[int(a[e])].append(e)
        # else: dead end -- the popped edges are dropped (they cannot close a cycle now)
    return np.array(sorted(keep), dtype=np.int64)


def balance_ds_sign(ds, idx, rng, bins=8):
    """Subset of `idx` in which, inside every |ds| band, the positive item is the
    higher-surprisal one exactly as often as the lower-surprisal one. Pins the PAIRED
    surprisal guard at 0.5; the caliper alone only pins the pooled one."""
    ds = np.asarray(ds)[idx]
    a = np.abs(ds)
    edges = np.quantile(a, np.linspace(0, 1, bins + 1)[1:-1]) if len(a) > bins else []
    b = np.digitize(a, edges)
    keep = []
    for bb in np.unique(b):
        pos = idx[(b == bb) & (ds > 0)]
        neg = idx[(b == bb) & (ds < 0)]
        c = min(len(pos), len(neg))
        if c:
            keep.append(rng.choice(pos, c, replace=False))
            keep.append(rng.choice(neg, c, replace=False))
    return np.concatenate(keep) if keep else np.zeros(0, dtype=np.int64)


def _forward_at(model, windows, t, device, bs=512):
    """logits at t-1 and t, and residual states at t (all blocks) and t-1 (last block)."""
    import torch
    T = model.block_size
    n = len(windows)
    d = model.transformer.wte.weight.shape[1]
    lq_prev = np.empty((n, 16), np.float32)
    lq_next = np.empty((n, 16), np.float32)
    st = {b: np.empty((n, d), np.float32) for b in BLOCKS}
    st_tm1 = np.empty((n, d), np.float32)
    with torch.no_grad():
        for c0 in range(0, n, bs):
            x = torch.as_tensor(windows[c0:c0 + bs, :T], device=device)
            tt = torch.as_tensor(t[c0:c0 + bs], device=device)
            ar = torch.arange(len(x), device=device)
            lg, _, inter = model(x, return_intermediates=True)
            lsm = torch.log_softmax(lg.float(), -1)
            lq_prev[c0:c0 + bs] = lsm[ar, tt - 1].cpu().numpy()
            lq_next[c0:c0 + bs] = lsm[ar, tt].cpu().numpy()
            for b in BLOCKS:
                st[b][c0:c0 + bs] = inter[b][ar, tt].float().cpu().numpy()
            st_tm1[c0:c0 + bs] = inter[BLOCKS[-1]][ar, tt - 1].float().cpu().numpy()
    return lq_prev, lq_next, st, st_tm1


def phasic_readouts(model, S, device="cuda", seed=0, caliper=0.1, t_lo=16):
    We = S["windows_edit"]
    n, T1 = We.shape
    T = T1 - 1
    L = S["ptok"].shape[-1] - 1
    et, tv, ks, pL = S["etype"], S["t_v"], S["k_star"], S["pL_edit"]

    vw = np.where((et == 0) & (tv >= t_lo) & (tv <= T - 1))[0]
    t = tv[vw]
    lqp_v, lqn_v, st_v, tm1_v = _forward_at(model, We[vw], t, device)
    x_v = We[vw, t]
    s_all = -lqp_v                                                   # surprisal of every token
    s_v = s_all[np.arange(len(vw)), x_v]
    legal = pL[vw, t] > 0
    assert not legal[np.arange(len(vw)), x_v].any(), "a violating token is legal under p_L"
    assert legal.any(1).all(), "a legal prefix with no legal continuation"
    dist = np.where(legal, np.abs(s_all - s_v[:, None]), np.inf)
    x_c = dist.argmin(1)
    has = dist.min(1) <= caliper

    res = {"caliper": caliper, "n_violations": int(len(vw)), "n_pairs": int(has.sum()),
           "coverage_by_kstar": {}}
    for k in range(1, L + 1):
        sel = ks[vw] == k
        if sel.sum():
            res["coverage_by_kstar"][k] = {"n": int(sel.sum()), "paired_frac": float(has[sel].mean()),
                                           "mean_surprisal_viol": float(s_v[sel].mean())}
    if has.sum() < 100:
        return res

    P = np.where(has)[0]
    twin = We[vw[P]].copy()
    twin[np.arange(len(P)), t[P]] = x_c[P]
    lqp_c, lqn_c, st_c, tm1_c = _forward_at(model, twin, t[P], device)
    res["check_identical_prefix_max_abs"] = float(np.abs(tm1_c - tm1_v[P]).max())
    res["check_identical_qprev_max_abs"] = float(np.abs(lqp_c - lqp_v[P]).max())
    assert res["check_identical_qprev_max_abs"] < 1e-3, res["check_identical_qprev_max_abs"]

    # pooled event table: first len(P) = violations, next len(P) = legal twins
    kst = np.concatenate([ks[vw[P]], ks[vw[P]]])
    y = np.concatenate([np.ones(len(P), bool), np.zeros(len(P), bool)])
    tok = np.concatenate([x_v[P], x_c[P]])
    lqp = np.concatenate([lqp_v[P], lqp_c])
    lqn = np.concatenate([lqn_v[P], lqn_c])
    st = {b: np.concatenate([st_v[b][P], st_c[b]]) for b in BLOCKS}
    tm1 = np.concatenate([tm1_v[P], tm1_c])
    surpr = -lqp[np.arange(2 * len(P)), tok]
    H_next = _ent(np.exp(lqn))
    KL_update = _kl(np.exp(lqn), lqp)
    X_pre = np.concatenate([lqp, np.eye(16, dtype=np.float32)[tok]], 1)

    rng = np.random.default_rng(seed + 2)
    split = np.random.default_rng(seed + 1).random(len(P)) < 0.6
    ds_pair = surpr[:len(P)] - surpr[len(P):]
    keep = np.zeros(len(P), bool)
    for flag in (split, ~split):
        sub = balance_ds_sign(ds_pair, np.where(flag)[0], rng)
        if len(sub):
            keep[balance_tokens(x_v[P], x_c[P], sub, rng)] = True
    res["n_pairs_token_balanced"] = int(keep.sum())
    tr_pair = split & keep
    te_pair = (~split) & keep
    tr = np.concatenate([tr_pair, tr_pair])
    te = np.concatenate([te_pair, te_pair])

    feats = {"state_tm1_guard": tm1, "pre_logit_mlp": X_pre, "post_logit_mlp": lqn}
    feats.update({f"state_{b}": st[b] for b in BLOCKS})
    probes = {}
    for name, X in feats.items():
        hidden = 256 if name.endswith("_mlp") else 0
        probes[name] = fit_probe(X[tr], y[tr], hidden=hidden, steps=800 if hidden else 400)
    scores = {"guard_surprisal": surpr, "H_next": H_next, "KL_update": KL_update}
    for name, f in probes.items():
        scores[name] = f(feats[name])
    bb = max(BLOCKS[1:], key=lambda b: auc(scores[f"state_{b}"][te], y[te]))
    f_mlp = fit_probe(st[bb][tr], y[tr], hidden=256, steps=800)
    scores["state_best_mlp"] = f_mlp(st[bb])
    res["best_block"] = bb

    npair = len(P)

    def score_subset(pair_mask):
        ev = np.concatenate([pair_mask, pair_mask])
        out = {"n_pairs": int(pair_mask.sum()),
               "mean_abs_ds": float(np.abs(surpr[:npair][pair_mask] - surpr[npair:][pair_mask]).mean())}
        for nm, sc in scores.items():
            out[nm] = {"auc": auc(sc[ev], y[ev]),
                       "paired": paired_win(sc[:npair][pair_mask], sc[npair:][pair_mask])}
        return out

    res["test_all"] = score_subset(te_pair)
    res["test_by_kstar"] = {}
    for k in range(1, L + 1):
        m = te_pair & (ks[vw[P]] == k)
        if m.sum() >= 30:
            res["test_by_kstar"][k] = score_subset(m)
    # the size of the post-token forecast shift, violation vs twin (descriptive, test pairs)
    te_pairs = te_pair
    res["descriptive_test"] = {
        "H_next_viol": float(H_next[:npair][te_pairs].mean()),
        "H_next_twin": float(H_next[npair:][te_pairs].mean()),
        "KL_update_viol": float(KL_update[:npair][te_pairs].mean()),
        "KL_update_twin": float(KL_update[npair:][te_pairs].mean()),
        "KL_between_post_forecasts": float(_kl(np.exp(lqn[:npair][te_pairs]), lqn[npair:][te_pairs]).mean()),
    }
    return res


@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=2 * 3600, memory=16384, max_containers=4)
def phasic_ckpt(ckpt: str, stim_tag: str = "a1", seed: int = 0, caliper: float = 0.1):
    volume.reload()
    model, rules, rule_w, cfg = load_trajectory_ckpt(ckpt, "cuda")
    S = _load_stimuli(stim_tag)
    res = phasic_readouts(model, S, "cuda", seed=seed, caliper=caliper)
    res["config"] = {"ckpt": ckpt, "step": cfg.get("step"), "stim_tag": stim_tag, "seed": seed}
    print(f"step {cfg.get('step')}  caliper {caliper}  pairs {res['n_pairs']}/{res['n_violations']} "
          f"(token-balanced {res.get('n_pairs_token_balanced')})  "
          f"coverage {({k: round(v['paired_frac'], 2) for k, v in res['coverage_by_kstar'].items()})}", flush=True)
    if "test_all" in res:
        a = res["test_all"]
        print(f"  prefix check {res['check_identical_prefix_max_abs']:.2e}  best {res['best_block']}  "
              f"|ds| {a['mean_abs_ds']:.3f}", flush=True)
        print("  test AUC/paired: " + "  ".join(f"{k} {v['auc']:.3f}/{v['paired']:.3f}"
                                               for k, v in a.items() if isinstance(v, dict)), flush=True)
        print(f"  {res['descriptive_test']}", flush=True)
    with open(f"{ckpt[:-3]}_phasic_{stim_tag}_cal{caliper}.json", "w") as f:
        json.dump(res, f, cls=NumpyEncoder, indent=1)
    volume.commit()
    return {"step": cfg.get("step")}


@app.function(volumes={DATA_DIR: volume}, timeout=4 * 3600, memory=4096)
def phasic_sweep(traj_dir: str, stim_tag: str = "a1", caliper: float = 0.1):
    import glob
    volume.reload()
    ckpts = sorted(glob.glob(f"{traj_dir}/step*.pt"))
    return len(list(phasic_ckpt.map(ckpts, kwargs={"stim_tag": stim_tag, "caliper": caliper})))


# ---------------------------------------------------------------------------
# Graded control: two LEGAL tokens after the same prefix, matched on the MODEL's
# surprisal, differing in their TRUE probability. Positive = the less likely one.
# If the phasic readouts separate these too, what they carry is graded "how unlikely
# was that really", not a categorical rule-violation detector.
# ---------------------------------------------------------------------------

def legal_gradient_readouts(model, S, device="cuda", seed=0, caliper=0.3, min_ratio=2.0,
                            t_lo=16):
    We = S["windows_edit"]
    T = We.shape[1] - 1
    L = S["ptok"].shape[-1] - 1
    et, tv, ks, pL = S["etype"], S["t_v"], S["k_star"], S["pL_edit"]
    vw = np.where((et == 0) & (tv >= t_lo) & (tv <= T - 1))[0]
    t = tv[vw]
    lqp_v, _, _, _ = _forward_at(model, We[vw], t, device)
    s_all = -lqp_v
    p_true = pL[vw, t]
    legal = p_true > 0

    # for every prefix: the legal pair within the caliper with the largest true-probability
    # ratio (at least min_ratio)
    n = len(vw)
    big = np.full(n, -1)
    small = np.full(n, -1)
    lr = np.log(np.where(legal, p_true, 1e-300))
    for i in range(n):
        idx = np.where(legal[i])[0]
        if len(idx) < 2:
            continue
        o = idx[np.argsort(s_all[i, idx])]
        best = 0.0
        for a in range(len(o)):
            for b in range(a + 1, len(o)):
                if s_all[i, o[b]] - s_all[i, o[a]] > caliper:
                    break
                d = abs(lr[i, o[a]] - lr[i, o[b]])
                if d > best and d >= np.log(min_ratio):
                    best = d
                    hi, lo_ = (o[a], o[b]) if lr[i, o[a]] > lr[i, o[b]] else (o[b], o[a])
                    big[i], small[i] = hi, lo_
    has = big >= 0
    res = {"caliper": caliper, "min_ratio": min_ratio, "n_prefixes": int(n),
           "n_pairs": int(has.sum())}
    if has.sum() < 200:
        return res
    P = np.where(has)[0]
    w_hi = We[vw[P]].copy()
    w_hi[np.arange(len(P)), t[P]] = big[P]
    w_lo = We[vw[P]].copy()
    w_lo[np.arange(len(P)), t[P]] = small[P]
    lqp_h, lqn_h, st_h, tm1_h = _forward_at(model, w_hi, t[P], device)
    lqp_l, lqn_l, st_l, tm1_l = _forward_at(model, w_lo, t[P], device)

    # positive class = the LESS likely legal token
    y = np.concatenate([np.ones(len(P), bool), np.zeros(len(P), bool)])
    tok = np.concatenate([small[P], big[P]])
    lqp = np.concatenate([lqp_l, lqp_h])
    lqn = np.concatenate([lqn_l, lqn_h])
    st = {b: np.concatenate([st_l[b], st_h[b]]) for b in BLOCKS}
    surpr = -lqp[np.arange(2 * len(P)), tok]
    H_next = _ent(np.exp(lqn))
    X_pre = np.concatenate([lqp, np.eye(16, dtype=np.float32)[tok]], 1)
    res["mean_log_ratio"] = float((lr[P, big[P]] - lr[P, small[P]]).mean())

    rng = np.random.default_rng(seed + 2)
    split = np.random.default_rng(seed + 1).random(len(P)) < 0.6
    ds_pair = surpr[:len(P)] - surpr[len(P):]
    keep = np.zeros(len(P), bool)
    for flag in (split, ~split):
        sub = balance_ds_sign(ds_pair, np.where(flag)[0], rng)
        if len(sub):
            keep[balance_tokens(small[P], big[P], sub, rng)] = True
    tr_pair, te_pair = split & keep, (~split) & keep
    tr = np.concatenate([tr_pair, tr_pair])
    res["n_pairs_token_balanced"] = int(keep.sum())
    if te_pair.sum() < 100:
        return res

    feats = {"pre_logit_mlp": X_pre, "post_logit_mlp": lqn, "state_post_embed": st["post_embed"]}
    feats.update({f"state_{b}": st[b] for b in BLOCKS[1:]})
    scores = {"guard_surprisal": surpr, "H_next": H_next}
    for name, X in feats.items():
        hidden = 256 if name.endswith("_mlp") else 0
        scores[name] = fit_probe(X[tr], y[tr], hidden=hidden, steps=800 if hidden else 400)(X)
    bb = max(BLOCKS[1:], key=lambda b: auc(scores[f"state_{b}"][np.concatenate([te_pair, te_pair])],
                                           y[np.concatenate([te_pair, te_pair])]))
    scores["state_best_mlp"] = fit_probe(st[bb][tr], y[tr], hidden=256, steps=800)(st[bb])
    res["best_block"] = bb
    npair = len(P)
    ev = np.concatenate([te_pair, te_pair])
    res["test_all"] = {"n_pairs": int(te_pair.sum()),
                       "mean_abs_ds": float(np.abs(surpr[:npair][te_pair] - surpr[npair:][te_pair]).mean())}
    for nm, sc in scores.items():
        res["test_all"][nm] = {"auc": auc(sc[ev], y[ev]),
                               "paired": paired_win(sc[:npair][te_pair], sc[npair:][te_pair])}
    return res


@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=2 * 3600, memory=16384, max_containers=4)
def gradient_ckpt(ckpt: str, stim_tag: str = "swap65k", seed: int = 0, caliper: float = 0.3,
                  min_ratio: float = 2.0):
    volume.reload()
    model, rules, rule_w, cfg = load_trajectory_ckpt(ckpt, "cuda")
    S = _load_stimuli(stim_tag)
    res = legal_gradient_readouts(model, S, "cuda", seed=seed, caliper=caliper, min_ratio=min_ratio)
    res["config"] = {"ckpt": ckpt, "step": cfg.get("step"), "stim_tag": stim_tag}
    print(f"step {cfg.get('step')}  pairs {res['n_pairs']} (balanced {res.get('n_pairs_token_balanced')})",
          flush=True)
    if "test_all" in res:
        print("  " + "  ".join(f"{k} {v['auc']:.3f}/{v['paired']:.3f}"
                               for k, v in res["test_all"].items() if isinstance(v, dict)), flush=True)
    with open(f"{ckpt[:-3]}_gradient_{stim_tag}.json", "w") as f:
        json.dump(res, f, cls=NumpyEncoder, indent=1)
    volume.commit()
    return {"step": cfg.get("step")}


@app.function(volumes={DATA_DIR: volume}, timeout=4 * 3600, memory=4096)
def gradient_sweep(traj_dir: str, stim_tag: str = "swap65k", caliper: float = 0.3):
    import glob
    volume.reload()
    ckpts = sorted(glob.glob(f"{traj_dir}/step*.pt"))
    return len(list(gradient_ckpt.map(ckpts, kwargs={"stim_tag": stim_tag, "caliper": caliper})))
