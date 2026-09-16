"""Q3: the model's predictive as a NULL for scoring candidate next-level units.

Bayesian model comparison with the model as the reference. At each checkpoint, candidates
are adjacent pairs of depth-j constituents (A, B) in a held-out flat window, enumerated
with the oracle's parse and split by the grammar into

  true   A and B are the two children of one depth-(j+1) node     (constituent index even)
  chance A and B straddle a depth-(j+1) boundary                   (index odd)

Token-span statistics get hopeless past j = 2, so the candidate is defined at the FEATURE
level and its likelihood over B's tokens is closed at the token level exactly:

    P_cand(B_tok | f_A) = sum_f  Phat(f | f_A, parity) * P_grammar(B_tok | f)

`Phat(f_B | f_A, parity)` is a 16x16 empirical table estimated on a DISJOINT corpus, one
table per (j, parity) -- which is exactly the object a bracketing-induction procedure that
knows depth-j boundaries but not depth-(j+1) ones would estimate for each of the two
competing global bracketings. `P_grammar(B_tok | f)` is an exact unnormalised upward pass
on the depth-j subtree (no estimation, no sparsity).

Scores for one candidate occurrence (all in nats, positive = the candidate explains B's
tokens better than the null did):

    Delta_model   = -log q_model(B_tok | window prefix)      + log P_cand(B_tok | f_A)
    Delta_obs_k   = -log p_k(B_tok | window prefix)          + log P_cand(B_tok | f_A)
    Delta_marg    = -log q_model(B_tok | window prefix)      + log P_marg(B_tok)
    PMI           = log Phat(f_A, f_B) - log Phat(f_A) - log Phat(f_B)   (model-free)

`Delta_model - Delta_marg = log P_cand/P_marg` is model-free too, and `-log q_model(B)`
alone is reported so that the part of any separation that comes from B being a second
child rather than a first is visible rather than hidden.

Run:
  modal run -m rhm.logit_reading.altitude.units::units_sweep \
      --traj-dir /data/v16_s2_L6_m4_distinct/logit_reading/traj_a1_s42
"""

import json

import numpy as np

from rhm.shared import volume, DATA_DIR, NumpyEncoder
from rhm.logit_reading.calibration import app
from rhm.logit_reading.altitude.identity import _log_softmax, EPS

LOG0 = -700.0


def auc(score, y):
    from scipy.stats import rankdata
    score = np.asarray(score, np.float64)
    y = np.asarray(y).astype(bool)
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(score)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# ---------------------------------------------------------------------------
# exact P(token span | root feature) on a depth-j subtree (unnormalised upward pass)
# ---------------------------------------------------------------------------

def span_loglik(tokens, rules, rule_w, j):
    """tokens (N, s^j) -> (N, v) log P(tokens | depth-j subtree root feature = a)."""
    L = len(rules)
    v, m, s = rules[0].shape
    sub = rules[L - j:]
    subw = None if rule_w is None else rule_w[L - j:]
    N, S = tokens.shape
    assert S == s ** j
    up = np.zeros((N, S, v))
    np.put_along_axis(up, tokens[..., None], 1.0, axis=-1)
    logscale = np.zeros((N, 1))
    for d in range(j - 1, -1, -1):
        n_par = s ** d
        child = up.reshape(N, n_par, s, v)
        prod = None
        for i in range(s):
            gi = child[:, :, i, :][..., sub[d][:, :, i].reshape(-1)]      # (N, n_par, v*m)
            prod = gi if prod is None else prod * gi
        prod = prod.reshape(N, n_par, v, m)
        w = (np.full((v, m), 1.0 / m) if subw is None else subw[d])
        up = (prod * w[None, None]).sum(-1)                                # (N, n_par, v)
        sc = up.max(-1, keepdims=True)
        sc = np.where(sc > 0, sc, 1.0)
        up = up / sc
        logscale = logscale + np.log(sc).sum(1)
    return np.log(np.clip(up[:, 0, :], EPS, None)) + logscale


# ---------------------------------------------------------------------------
# the disjoint corpus: Phat(f_B | f_A, parity) at every level
# ---------------------------------------------------------------------------

def pair_tables(rules, rule_w, n_seq, seed, j_list, smooth=0.5):
    """Empirical adjacent-feature tables on a corpus disjoint from the scoring windows."""
    from rhm.logit_reading.grammar import generate
    L = len(rules)
    v, m, s = rules[0].shape
    _, feats, _ = generate(rules, rule_w, n_seq, np.random.default_rng(seed),
                           return_latents=True)
    out = {}
    for j in j_list:
        d = L - j
        f = feats[d].reshape(-1)                       # level-d features of the whole stream
        a, b = f[:-1], f[1:]
        idx = np.arange(len(a))
        tabs = {}
        for parity, name in ((0, "true"), (1, "chance")):
            sel = (idx % 2) == parity
            cnt = np.zeros((v, v))
            np.add.at(cnt, (a[sel], b[sel]), 1.0)
            joint = (cnt + smooth) / (cnt + smooth).sum()
            tabs[name] = {
                "n": int(sel.sum()),
                "logp_b_given_a": np.log(joint / joint.sum(1, keepdims=True)),
                "logp_a": np.log(joint.sum(1)), "logp_b": np.log(joint.sum(0)),
                "logp_joint": np.log(joint),
                "mi": float((joint * (np.log(joint) - np.log(joint.sum(1, keepdims=True))
                                      - np.log(joint.sum(0, keepdims=True)))).sum()),
            }
        out[j] = tabs
    return out


# ---------------------------------------------------------------------------
# scoring windows: tokens + full parse, reproducing sample_flat_windows exactly
# ---------------------------------------------------------------------------

def windows_with_parse(rules, rule_w, n, T1, seed):
    from rhm.logit_reading.grammar import generate
    from rhm.logit_reading.flat_oracle import sample_flat_windows
    L = len(rules)
    s = rules[0].shape[2]
    T = s ** L
    n_seq = -(-(T1 + T) // T)
    _, feats, _ = generate(rules, rule_w, n * n_seq, np.random.default_rng(seed),
                           return_latents=True)
    leaves = feats[L].reshape(n, n_seq * T)
    fs = [feats[d].reshape(n, n_seq * (s ** d)) for d in range(L + 1)]
    rng = np.random.default_rng(seed + 1)
    phase = rng.integers(0, T, size=n)
    idx = phase[:, None] + np.arange(T1)[None, :]
    wins = np.take_along_axis(leaves, idx, axis=1)
    ref, ref_phase = sample_flat_windows(rules, n, T1, seed, rule_w=rule_w)
    assert (wins == ref).all() and (phase == ref_phase).all(), "parse stream != sample_flat_windows"
    return wins, phase, leaves, fs


def build_events(rules, rule_w, wins, phase, leaves, fs, j, T, t_lo=8):
    """Every (A, B) adjacent depth-j pair with A and B inside the window and B inside the
    model's predicted range. Returns a dict of per-event arrays."""
    L = len(rules)
    s = rules[0].shape[2]
    span = s ** j
    n, T1 = wins.shape
    n_const = leaves.shape[1] // span
    d = L - j
    rows, cols_a, cols_b, cidx = [], [], [], []
    for w in range(n):
        ps = int(phase[w])
        for c in range(-(-ps // span), n_const - 1):         # A must start inside the window
            a_start, b_start, b_end = c * span, (c + 1) * span, (c + 2) * span
            if a_start < ps or b_end > ps + T1:
                continue
            if b_start - ps < max(1, t_lo):
                continue
            rows.append(w)
            cols_a.append(a_start - ps)
            cols_b.append(b_start - ps)
            cidx.append(c)
    rows = np.array(rows, np.int64)
    cols_a = np.array(cols_a, np.int64)
    cols_b = np.array(cols_b, np.int64)
    cidx = np.array(cidx, np.int64)
    if not len(rows):
        return None
    f_a = fs[d][rows, cidx]
    f_b = fs[d][rows, cidx + 1]
    b_tok = np.take_along_axis(wins[rows], (cols_b[:, None] + np.arange(span)[None, :]), 1)
    return {"w": rows, "col_a": cols_a, "col_b": cols_b, "cidx": cidx,
            "f_a": f_a, "f_b": f_b, "b_tok": b_tok, "true": (cidx % 2) == 0}


def candidate_costs(ev, tabs, rules, rule_w, j):
    """-log P_cand(B_tok | f_A) and -log P_marg(B_tok), plus PMI, per event."""
    ll = span_loglik(ev["b_tok"], rules, rule_w, j)                # (N, v)
    out = {}
    for name in ("true", "chance"):
        t = tabs[name]
        sel = ev["true"] if name == "true" else ~ev["true"]
        if not sel.any():
            continue
        lcond = t["logp_b_given_a"][ev["f_a"][sel]]                # (n_sel, v)
        out.setdefault("nll_cand", np.zeros(len(ll)))
        out.setdefault("nll_marg", np.zeros(len(ll)))
        out.setdefault("pmi", np.zeros(len(ll)))
        from scipy.special import logsumexp
        out["nll_cand"][sel] = -logsumexp(lcond + ll[sel], axis=1)
        out["nll_marg"][sel] = -logsumexp(t["logp_b"][None, :] + ll[sel], axis=1)
        out["pmi"][sel] = (t["logp_joint"][ev["f_a"][sel], ev["f_b"][sel]]
                           - t["logp_a"][ev["f_a"][sel]] - t["logp_b"][ev["f_b"][sel]])
    return out


def span_nll(nlp, ev, span):
    """Sum of a per-column negative log-prob array (n, T) over B's columns."""
    cols = ev["col_b"][:, None] + np.arange(span)[None, :]
    return np.take_along_axis(nlp[ev["w"]], cols - 1, 1).sum(1)


@app.function(volumes={DATA_DIR: volume}, timeout=4 * 3600, memory=32768, cpu=4.0)
def units_sweep(traj_dir: str, v: int = 16, s: int = 2, depth: int = 6, m: int = 4,
                rule_seed: int = 0, alpha: float = 1.0, weight_seed: int = 1,
                n_windows: int = 4096, eval_seed: int = 4242, corpus_seq: int = 200000,
                corpus_seed: int = 99991, t_lo: int = 8, j_max: int = 5,
                out_name: str = "altitude_units", limit: int = 0):
    import glob
    import os
    import resource
    from rhm.rhm_data import generate_rules_distinct
    from rhm.logit_reading.grammar import synonym_weights
    volume.reload()
    L, T = depth, s ** depth
    rules = generate_rules_distinct(v, s, L, m, seed=rule_seed)
    rule_w = None if alpha <= 0 else synonym_weights(v, L, m, alpha, weight_seed)
    j_list = list(range(1, j_max + 1))

    tabs = pair_tables(rules, rule_w, corpus_seq, corpus_seed, j_list)
    print("corpus MI (nats) by j: "
          + "  ".join(f"j{j} true {tabs[j]['true']['mi']:.4f} chance {tabs[j]['chance']['mi']:.4f}"
                      for j in j_list), flush=True)
    wins, phase, leaves, fs = windows_with_parse(rules, rule_w, n_windows, T + 1, eval_seed)

    evs, cands = {}, {}
    for j in j_list:
        e = build_events(rules, rule_w, wins, phase, leaves, fs, j, T, t_lo)
        if e is None:
            continue
        evs[j] = e
        cands[j] = candidate_costs(e, tabs[j], rules, rule_w, j)
        print(f"  j={j}  events {len(e['w'])}  true {int(e['true'].sum())}  "
              f"chance {int((~e['true']).sum())}", flush=True)

    paths = sorted(glob.glob(f"{traj_dir}/step*_calibration.npz"))
    if limit:
        paths = paths[:limit]
    obs_nlp = None
    rows = []
    for pth in paths:
        z_ = np.load(pth)
        assert (z_["windows"] == wins).all(), "checkpoint arrays use different eval windows"
        logq = _log_softmax(z_["logits"].astype(np.float64))
        tok = wins[:, 1:]
        nlp_model = -np.take_along_axis(logq, tok[..., None], -1)[..., 0]     # (n, T)
        if obs_nlp is None:
            pr = z_["preds"].astype(np.float64)
            obs_nlp = np.stack([-np.log(np.clip(np.take_along_axis(
                pr[:, :, k, :], tok[..., None], -1)[..., 0], EPS, None)) for k in range(L + 1)], -1)
            del pr
        r = {"step": int(os.path.basename(pth)[4:10]), "by_j": {}}
        for j in j_list:
            if j not in evs:
                continue
            e, c = evs[j], cands[j]
            span = s ** j
            cm = span_nll(nlp_model, e, span)
            co = {k: span_nll(obs_nlp[..., k], e, span) for k in range(L + 1)}
            y = e["true"]
            d_model = cm - c["nll_cand"]
            d_marg = cm - c["nll_marg"]
            blk = {
                "n_true": int(y.sum()), "n_chance": int((~y).sum()),
                "mean_col_b_true": float(e["col_b"][y].mean()),
                "mean_col_b_chance": float(e["col_b"][~y].mean()),
                "cost_model": [float(cm[y].mean()), float(cm[~y].mean())],
                "cost_cand": [float(c["nll_cand"][y].mean()), float(c["nll_cand"][~y].mean())],
                "cost_marg": [float(c["nll_marg"][y].mean()), float(c["nll_marg"][~y].mean())],
                "delta_model": [float(d_model[y].mean()), float(d_model[~y].mean())],
                "delta_marg": [float(d_marg[y].mean()), float(d_marg[~y].mean())],
                "frac_model_beaten": [float((d_model[y] > 0).mean()), float((d_model[~y] > 0).mean())],
                "pmi": [float(c["pmi"][y].mean()), float(c["pmi"][~y].mean())],
                "auc_delta_model": auc(d_model, y),
                "auc_delta_marg": auc(d_marg, y),
                "auc_cost_model_alone": auc(cm, y),
                "auc_cand_gain": auc(c["nll_marg"] - c["nll_cand"], y),
                "auc_pmi": auc(c["pmi"], y),
                "auc_delta_obs": {k: auc(co[k] - c["nll_cand"], y) for k in range(L + 1)},
                "delta_obs_true": {k: float((co[k] - c["nll_cand"])[y].mean()) for k in range(L + 1)},
                "delta_obs_chance": {k: float((co[k] - c["nll_cand"])[~y].mean()) for k in range(L + 1)},
                "cost_obs_true": {k: float(co[k][y].mean()) for k in range(L + 1)},
                "frac_obs_beaten_true": {k: float(((co[k] - c["nll_cand"])[y] > 0).mean())
                                         for k in range(L + 1)},
            }
            r["by_j"][j] = blk
        rows.append(r)
        print(f"step {r['step']:>6}  " + "  ".join(
            f"j{j}: D_true {r['by_j'][j]['delta_model'][0]:+.2f} D_ch "
            f"{r['by_j'][j]['delta_model'][1]:+.2f} auc {r['by_j'][j]['auc_delta_model']:.3f}"
            for j in j_list if j in r["by_j"]), flush=True)
    meta = {"corpus_seq": corpus_seq, "corpus_seed": corpus_seed, "eval_seed": eval_seed,
            "n_windows": n_windows, "t_lo": t_lo,
            "mi": {j: {c: tabs[j][c]["mi"] for c in ("true", "chance")} for j in j_list}}
    if limit:
        return rows[-1]
    with open(f"{traj_dir}/{out_name}.json", "w") as f:
        json.dump({"meta": meta, "rows": rows}, f, cls=NumpyEncoder, indent=1)
    volume.commit()
    print(f"peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.1f} GB", flush=True)
    return f"{traj_dir}/{out_name}.json"
