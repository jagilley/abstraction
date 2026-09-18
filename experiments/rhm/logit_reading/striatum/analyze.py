"""Every table and figure in `results/`, from the JSONs pulled off the volume.

  modal volume get rhm-scaling-data \
      /v16_s2_L6_m4_distinct/logit_reading/traj_a1_s42 <dir>/traj_a1_s42
  python -m rhm.logit_reading.striatum.analyze <dir>/traj_a1_s42 \
      --tags a1,swap65k --out rhm/logit_reading/striatum/results \
      --figs rhm/logit_reading/striatum/figs
"""

import argparse
import glob
import json
import os

import numpy as np

LEVELS = [1, 2, 3, 4, 5, 6]
A_LIST = [0, 1, 2, 4, 8]


def load(d, tag):
    out = {}
    for f in sorted(glob.glob(os.path.join(d, f"step*_striatum_{tag}.json"))):
        r = json.load(open(f))
        out[int(r["step"])] = r
    return out


def fmt(x, nd=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    return f"{x:.{nd}f}"


def tbl(rows, hdr):
    w = [max(len(str(h)), *(len(str(r[i])) for r in rows)) if rows else len(str(h))
         for i, h in enumerate(hdr)]
    out = ["| " + " | ".join(str(h).ljust(w[i]) for i, h in enumerate(hdr)) + " |",
           "|" + "|".join("-" * (w[i] + 2) for i in range(len(hdr))) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c).ljust(w[i]) for i, c in enumerate(r)) + " |")
    return "\n".join(out)


def sec_actor(runs):
    rows = []
    for st in sorted(runs):
        a = runs[st]["actor_clean_acc"]
        pb = runs[st]["primary_block"]
        rows.append([st] + [fmt(a.get(f"{pb}/l{l}")) for l in LEVELS]
                    + [fmt(a.get(f"post_embed/l{l}")) for l in (1, 3, 5)])
    return tbl(rows, ["step"] + [f"l={l}" for l in LEVELS]
               + ["emb l1", "emb l3", "emb l5"])


def sec_critic(runs):
    rows = []
    for st in sorted(runs):
        r = runs[st]
        cb = r["critic_block"]
        v = r["critic_val_r2"]
        s = r["critic_val_r2_shuf"]
        cl = r.get("clean_critic_val_r2", {})
        rows.append([st, cb]
                    + [fmt(v.get(f"{cb}/l{l}_d0"), 4) for l in LEVELS]
                    + [fmt(cl.get("l1_d0"), 4), fmt(cl.get("l3_d0"), 4),
                       fmt(v.get(f"post_embed/l1_d0"), 4), fmt(s.get(f"{cb}/sh_l1_d0"), 4)])
    return tbl(rows, ["step", "block"] + [f"V l{l} d0" for l in LEVELS]
               + ["clean l1", "clean l3", "emb l1", "shuf l1"])


def sec_damage(runs, an="fd", a=0):
    rows = []
    for st in sorted(runs):
        t = runs[st]["tables"].get(an)
        if not t:
            continue
        for l in LEVELS:
            d = t["damage"].get(f"l{l}_a{a}", {})
            ac = t["acc_edit"].get(f"l{l}_a{a}", {})
            rows.append([st, l, fmt(ac.get("orig")), fmt(ac.get("cons")), fmt(ac.get("incons")),
                         fmt(d.get("swap")), fmt(d.get("rare")), fmt(d.get("none"))])
    return tbl(rows, ["step", "l", "acc orig", "acc cons", "acc incons",
                      "dmg swap", "dmg rare", "dmg none"])


def sec_auc(runs, an, key, a=0, scores=None):
    scores = scores or ["R", "dR", "R_clean", "dR_clean", "Rshuf", "R_mlp", "V",
                        "nll", "dH", "excess", "banked_excess", "probe_illegal",
                        "probe_cons"]
    rows = []
    for st in sorted(runs):
        t = runs[st]["tables"].get(an)
        if not t:
            continue
        for nm in scores:
            k = f"{key}/{nm}"
            if k not in t:
                continue
            rows.append([st, nm] + [fmt(t[k].get(f"l{l}_a{a}")) for l in LEVELS])
    return tbl(rows, ["step", "score"] + [f"l={l}" for l in LEVELS])


def sec_cells(runs, an):
    """Cell sizes, so a thin t_v cell is visible rather than silently dropped."""
    st = sorted(runs)[-1]
    t = runs[st]["tables"].get(an, {})
    rows = []
    for a in A_LIST:
        rows.append([a, "n consequential"] + [t.get("n_cons_test", {}).get(f"l{l}_a{a}", "--")
                                              for l in LEVELS])
        rows.append([a, "n inconsequential"] + [t.get("n_incons_test", {}).get(f"l{l}_a{a}", "--")
                                                for l in LEVELS])
        rows.append([a, "n matched (k*, j, nll)"] + [t.get("matched_n", {}).get(f"l{l}_a{a}", "--")
                                                     for l in LEVELS])
        rows.append([a, "n damage-flip rows"] + [t.get("flip_n", {}).get(f"l{l}_a{a}", "--")
                                                 for l in LEVELS])
    return (f"Held-out windows at this anchor: {t.get('n_test')} "
            f"(swap / rare / none = {t.get('n_test_by_etype')}).\n\n"
            + tbl(rows, ["a", "cell"] + [f"l={l}" for l in LEVELS]))


def sec_across(runs, an="fd", key="across_level"):
    rows = []
    for st in sorted(runs):
        t = runs[st]["tables"].get(an, {}).get(key, {})
        for a in A_LIST:
            e = t.get(f"a{a}")
            if e:
                rows.append([st, a, e["n"], fmt(e["mean_rank_cons"]),
                             fmt(e["mean_rank_incons"]), fmt(e["win_rate"])])
    return tbl(rows, ["step", "a", "n", "rank cons", "rank incons", "win rate"])


def sec_base(runs, an="fd"):
    rows = []
    st = sorted(runs)[-1]
    t = runs[st]["tables"].get(an, {})
    for et in ("all", "swap", "rare"):
        cr = t.get("cons_rate", {}).get(et)
        if not cr:
            continue
        for a in A_LIST:
            rows.append([et, a] + [fmt(cr.get(f"l{l}_a{a}")) for l in LEVELS])
    return tbl(rows, ["etype", "a"] + [f"l={l}" for l in LEVELS])


def sec_base_kj(runs, an="fd"):
    st = sorted(runs)[-1]
    t = runs[st]["tables"].get(an, {})
    rows = []
    for k, v in sorted(t.get("cons_rate_by_kstar", {}).items()):
        rows.append(["k*=" + k[1:]] + [fmt(v.get(f"l{l}_a0")) for l in LEVELS])
    for k, v in sorted(t.get("cons_rate_by_j", {}).items()):
        rows.append(["j=" + k[1:]] + [fmt(v.get(f"l{l}_a0")) for l in LEVELS])
    return tbl(rows, ["cell"] + [f"l={l}" for l in LEVELS])


def figures(runs, tags, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(outdir, exist_ok=True)
    steps = sorted(runs)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for st in steps:
        a = runs[st]["actor_clean_acc"]
        pb = runs[st]["primary_block"]
        ax[0].plot(LEVELS, [a.get(f"{pb}/l{l}") for l in LEVELS], "o-", label=f"{st}")
    ax[0].axhline(1 / 16, ls=":", c="k")
    ax[0].set_xlabel("query level l"); ax[0].set_ylabel("clean actor accuracy")
    ax[0].set_title("actor: which levels the state supports"); ax[0].legend(fontsize=7)
    for st in steps:
        t = runs[st]["tables"].get("fd")
        if not t:
            continue
        ax[1].plot(LEVELS, [(t["damage"].get(f"l{l}_a0") or {}).get("swap") for l in LEVELS],
                   "o-", label=f"{st} swap")
        ax[1].plot(LEVELS, [(t["damage"].get(f"l{l}_a0") or {}).get("rare") for l in LEVELS],
                   "s--", label=f"{st} rare")
    ax[1].axhline(0, ls=":", c="k")
    ax[1].set_xlabel("query level l"); ax[1].set_ylabel("accuracy drop (orig - edit)")
    ax[1].set_title("what the edit costs at the goal"); ax[1].legend(fontsize=6)
    st = steps[-1]
    tr = runs[st].get("trace", {}).get("fd")
    if tr:
        taus = tr["taus"]
        for l in (1, 2, 3):
            for g, ls in (("cons", "-"), ("incons", "--")):
                k = f"mean_V_l{l}/{g}"
                if k in tr:
                    ax[2].plot(taus, [tr[k][str(t_)] for t_ in taus], ls,
                               label=f"l{l} {g}")
        ax[2].axvline(0, ls=":", c="k")
        ax[2].set_xlabel("offset from the edit onset"); ax[2].set_ylabel("V (P correct now)")
        ax[2].set_title(f"value trace, step {st}"); ax[2].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "striatum.png"), dpi=130)
    print("wrote", os.path.join(outdir, "striatum.png"))


# ---------------------------------------------------------------------------
# re-matching from the per-episode npz: the container's matcher strata on (k*, j)
# and nearest-neighbours on surprisal, which leaves the ANCHOR POSITION unbalanced at
# the t_v anchor (consequence there means the violation was detected early) and leaves
# a signed within-pair surprisal difference at the onset anchor. Both are the parent's
# recorded gotchas. This redoes the match with the position in the strata and the pairs
# sign-balanced inside |ds| bands, locally, from the saved rows.
# ---------------------------------------------------------------------------

def _auc(score, y):
    from scipy.stats import rankdata
    score = np.asarray(score, np.float64)
    y = np.asarray(y).astype(bool)
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(score)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def match_sign_balanced(keys, y, x, caliper, rng, bands=4):
    pairs = []
    for kk in np.unique(keys):
        pos = np.where((keys == kk) & y)[0]
        neg = np.where((keys == kk) & ~y)[0]
        if not len(pos) or not len(neg):
            continue
        pos = pos[np.argsort(x[pos])]
        negl = list(neg[np.argsort(x[neg])])
        for p in pos:
            if not negl:
                break
            xa = x[np.array(negl)]
            k = int(np.argmin(np.abs(xa - x[p])))
            if abs(xa[k] - x[p]) <= caliper:
                pairs.append((p, negl.pop(k)))
    if not pairs:
        return np.zeros(0, np.int64)
    P = np.array(pairs)
    ds = x[P[:, 0]] - x[P[:, 1]]
    qs = (np.quantile(np.abs(ds), np.linspace(0, 1, bands + 1)[1:-1])
          if len(ds) > bands else np.array([]))
    band = np.digitize(np.abs(ds), qs)
    keep = []
    for b in np.unique(band):
        ii = np.where(band == b)[0]
        pi, ni, zi = ii[ds[ii] > 0], ii[ds[ii] < 0], ii[ds[ii] == 0]
        c = min(len(pi), len(ni))
        keep.append(np.concatenate([
            rng.choice(pi, c, replace=False) if c else np.zeros(0, int),
            rng.choice(ni, c, replace=False) if c else np.zeros(0, int), zi]))
    keep = np.concatenate(keep).astype(int)
    return np.unique(P[keep].reshape(-1))


SCORES = ["R", "dR", "R_clean", "dR_clean", "R_mlp", "Rshuf", "V", "nll", "dH",
          "excess", "banked_excess", "probe_illegal", "probe_cons"]


def rematch(npz_path, cb, caliper=0.3, tbucket=4, seed=0, lmax=4):
    Z = np.load(npz_path)
    rng = np.random.default_rng(seed)
    out = {}
    for an in ("fd", "tv"):
        pre = f"{an}__"
        if pre + "t0" not in Z:
            continue
        g = lambda k: np.asarray(Z[pre + k]) if pre + k in Z else None
        et, ks, jj, t0, nll = g("etype"), g("k_star"), g("j"), g("t0"), g("nll_e")
        ent = {}
        for l in [x for x in LEVELS if x <= lmax]:
            for a in A_LIST:
                R, Ro = g(f"R_{cb}_l{l}_a{a}"), g(f"Ro_{cb}_l{l}_a{a}")
                if R is None:
                    continue
                Rc, Roc = g(f"R_clean_l{l}_a{a}"), g(f"Ro_clean_l{l}_a{a}")
                sc = {"R": -R, "dR": -(R - Ro), "Rshuf": -g(f"Rsh_l{l}_a{a}"),
                      "V": -g(f"V_l{l}_a{a}"), "nll": nll, "dH": -g("dH_e"),
                      "excess": g("excess_e")}
                if Rc is not None:
                    sc["R_clean"] = -Rc
                    sc["dR_clean"] = -(Rc - Roc)
                if g(f"R_mlp_l{l}_a{a}") is not None:
                    sc["R_mlp"] = -g(f"R_mlp_l{l}_a{a}")
                for nm in ("banked_excess", "probe_illegal"):
                    if g(nm) is not None:
                        sc[nm] = g(nm)
                if g(f"probe_cons_l{l}_a{a}") is not None:
                    sc["probe_cons"] = g(f"probe_cons_l{l}_a{a}")
                ok = g(f"ok_l{l}_a{a}").astype(bool)
                cons = g(f"cons_l{l}_a{a}").astype(bool)
                oe, oo = g(f"oe_l{l}_a{a}").astype(bool), g(f"oo_l{l}_a{a}").astype(bool)
                keys = ks * 10000 + jj * 1000 + (t0 // tbucket)
                for lname, lab, base_m in (("cons", cons, ok & (et < 2)),
                                           ("flip", oo & ~oe, ok & (et < 2) & oo)):
                    mi = np.where(base_m)[0]
                    if len(mi) < 50 or not 0 < lab[mi].mean() < 1:
                        continue
                    sel = match_sign_balanced(keys[mi], lab[mi], nll[mi], caliper, rng)
                    if len(sel) < 40:
                        continue
                    mm = mi[sel]
                    e = ent.setdefault(f"{lname}/l{l}_a{a}", {"n": int(len(mm))})
                    for nm, s_ in sc.items():
                        e[nm] = _auc(s_[mm], lab[mm])
                    e["guard_nll"] = _auc(nll[mm], lab[mm])
                    e["guard_t0"] = _auc(t0.astype(np.float64)[mm], lab[mm])
                    e["guard_kstar"] = _auc(ks.astype(np.float64)[mm], lab[mm])
                    e["guard_j"] = _auc(jj.astype(np.float64)[mm], lab[mm])
        out[an] = ent
    return out


def sec_rematch(rm, runs, an, lname, a=0):
    rows = []
    for st in sorted(rm):
        e = rm[st].get(an, {})
        if not e:
            continue
        rows.append([st, "n matched"] + [e.get(f"{lname}/l{l}_a{a}", {}).get("n", "--")
                                         for l in LEVELS if l <= 4])
        for nm in SCORES + ["guard_nll", "guard_t0", "guard_kstar", "guard_j"]:
            vals = [e.get(f"{lname}/l{l}_a{a}", {}).get(nm) for l in LEVELS if l <= 4]
            if all(v is None for v in vals):
                continue
            rows.append([st, nm] + [fmt(v) for v in vals])
    return tbl(rows, ["step", "score"] + [f"l={l}" for l in LEVELS if l <= 4])


# ---------------------------------------------------------------------------
# Addendum (2026-09-17): is the banked coeruleus excess head reading the damage
# through the surprise it was trained to predict, or through something else -- and does
# the outcome-trained critic add a direction the excess head does not have?
# CPU-local, from the per-episode npz plus the realised horizon excess column.
# ---------------------------------------------------------------------------

def _matched_rows(Z, pre, l, a, label, caliper=0.3, tbucket=4, seed=0):
    g = lambda k: np.asarray(Z[pre + k]) if pre + k in Z else None
    et, ks, jj, t0, nll = g("etype"), g("k_star"), g("j"), g("t0"), g("nll_e")
    ok = g(f"ok_l{l}_a{a}").astype(bool)
    oe, oo = g(f"oe_l{l}_a{a}").astype(bool), g(f"oo_l{l}_a{a}").astype(bool)
    if label == "flip":
        lab, base = oo & ~oe, ok & (et < 2) & oo
    else:
        lab, base = g(f"cons_l{l}_a{a}").astype(bool), ok & (et < 2)
    mi = np.where(base)[0]
    if len(mi) < 50 or not 0 < lab[mi].mean() < 1:
        return None
    keys = ks * 10000 + jj * 1000 + (t0 // tbucket)
    sel = match_sign_balanced(keys[mi], lab[mi], nll[mi], caliper,
                              np.random.default_rng(seed))
    if len(sel) < 100:
        return None
    return mi[sel], lab


def _cond_auc(x, y, lab, nbin=5):
    """AUC of x for lab inside quantile bins of y, averaged with weight n_pos*n_neg."""
    qs = np.quantile(y, np.linspace(0, 1, nbin + 1)[1:-1])
    b = np.digitize(y, qs)
    num = den = 0.0
    for bb in np.unique(b):
        m = b == bb
        n1, n0 = lab[m].sum(), (~lab[m]).sum()
        if n1 < 5 or n0 < 5:
            continue
        num += n1 * n0 * _auc(x[m], lab[m])
        den += n1 * n0
    return float(num / den) if den else float("nan")


def _oof_auc(X, lab, k=5, seed=0):
    """Out-of-fold linear (least-squares on the 0/1 label) combination of the columns."""
    X = np.asarray(X, np.float64)
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    n = len(X)
    fold = np.random.default_rng(seed).permutation(n) % k
    pred = np.zeros(n)
    for f in range(k):
        tr, te = fold != f, fold == f
        A = np.concatenate([X[tr], np.ones((tr.sum(), 1))], 1)
        beta = np.linalg.lstsq(A, lab[tr].astype(np.float64), rcond=None)[0]
        pred[te] = np.concatenate([X[te], np.ones((te.sum(), 1))], 1) @ beta
    return _auc(pred, lab)


def addendum(npz_path, cb, hexcess_path=None, anchor="tv", label="flip", a=0, lmax=4,
             nbin=5, seed=0):
    Z = np.load(npz_path)
    pre = f"{anchor}__"
    if pre + "t0" not in Z:
        return {}
    HX = np.load(hexcess_path) if hexcess_path and os.path.exists(hexcess_path) else None
    out = {}
    for l in [x for x in LEVELS if x <= lmax]:
        got = _matched_rows(Z, pre, l, a, label, seed=seed)
        if got is None:
            continue
        mm, lab = got
        g = lambda k: np.asarray(Z[pre + k]) if pre + k in Z else None
        w, t0 = g("w"), g("t0")
        sc = {"R": -g(f"R_{cb}_l{l}_a{a}"), "R_clean": -g(f"R_clean_l{l}_a{a}"),
              "V": -g(f"V_l{l}_a{a}"), "banked": g("banked_excess"),
              "excess_1pos": g("excess_e")}
        if g(f"R_mlp_l{l}_a{a}") is not None:
            sc["R_mlp"] = -g(f"R_mlp_l{l}_a{a}")
        if HX is not None:
            he = HX["hexcess_edit"]
            sc["excess_horizon"] = np.nan_to_num(he[w, np.clip(t0, 0, he.shape[1] - 1)],
                                                 nan=0.0)
            sc["excess_horizon_pre"] = np.nan_to_num(
                he[w, np.clip(t0 - 1, 0, he.shape[1] - 1)], nan=0.0)
        sc = {k: np.asarray(v)[mm] for k, v in sc.items() if v is not None}
        y = lab[mm]
        e = {"n": int(len(mm)), "base_rate": float(y.mean()),
             "marginal": {k: _auc(v, y) for k, v in sc.items()}}
        # `banked` is absent on any checkpoint that has no `coeruleus/` head beside it
        # (e.g. the fine-tuned trunks of `shaped/`); every banked column is then skipped
        # and the rest of the table is unchanged.
        has_b = "banked" in sc
        for tgt in ("R", "R_clean", "V", "R_mlp"):
            if tgt not in sc:
                continue
            if has_b:
                e[f"{tgt}|banked"] = _cond_auc(sc[tgt], sc["banked"], y, nbin)
                e[f"banked|{tgt}"] = _cond_auc(sc["banked"], sc[tgt], y, nbin)
                e[f"oof[banked,{tgt}]"] = _oof_auc(np.stack([sc["banked"], sc[tgt]], 1), y,
                                                   seed=seed)
            e[f"oof[{tgt}]"] = _oof_auc(sc[tgt][:, None], y, seed=seed)
        if has_b:
            e["oof[banked]"] = _oof_auc(sc["banked"][:, None], y, seed=seed)
        if "excess_horizon" in sc and has_b:
            e["banked|excess_horizon"] = _cond_auc(sc["banked"], sc["excess_horizon"], y, nbin)
            e["excess_horizon|banked"] = _cond_auc(sc["excess_horizon"], sc["banked"], y, nbin)
            e["oof[excess_horizon,banked]"] = _oof_auc(
                np.stack([sc["excess_horizon"], sc["banked"]], 1), y, seed=seed)
        out[f"l{l}"] = e
    return out


def sec_addendum(add, a=0):
    keys = sorted(add)
    if not keys:
        return "(no matched cells)"
    rows = [["n matched"] + [add[k]["n"] for k in keys],
            ["base rate"] + [fmt(add[k]["base_rate"]) for k in keys]]
    names = ["marginal:R", "marginal:R_mlp", "marginal:R_clean", "marginal:V",
             "marginal:banked", "marginal:excess_1pos", "marginal:excess_horizon",
             "marginal:excess_horizon_pre"]
    for nm in names:
        f_ = nm.split(":")[1]
        vals = [add[k]["marginal"].get(f_) for k in keys]
        if all(v is None for v in vals):
            continue
        rows.append([nm] + [fmt(v) for v in vals])
    for nm in ["R|banked", "banked|R", "V|banked", "banked|V", "R_clean|banked",
               "banked|R_clean", "R_mlp|banked", "banked|R_mlp",
               "banked|excess_horizon", "excess_horizon|banked",
               "oof[banked]", "oof[R]", "oof[banked,R]", "oof[V]", "oof[banked,V]",
               "oof[R_clean]", "oof[banked,R_clean]", "oof[R_mlp]", "oof[banked,R_mlp]",
               "oof[excess_horizon,banked]"]:
        vals = [add[k].get(nm) for k in keys]
        if all(v is None for v in vals):
            continue
        rows.append([nm] + [fmt(v) for v in vals])
    return tbl(rows, ["quantity"] + keys)


# ---------------------------------------------------------------------------
# 2026-09-17 re-read: the across-level contrast on every stored critic
# ---------------------------------------------------------------------------

def mlp_across(npz_path, cb, lmax=4, min_j=30):
    """Re-read of the §5 across-level contrast on every critic stored in the `.npz`.

    `task.py::summarize` ran `across_level` on the trained linear critic and on the
    clean-only critic, but never on the MLP critic, whose per-(window, anchor, a, level)
    revisions `R_mlp_l{l}_a{a}` are stored on the identical rows. The `.npz` holds only
    the test rows (`split == 2`), so `te` is all-True here and the `linear` / `clean`
    columns must reproduce the banked JSON's `across_level` / `across_level_clean` cells
    exactly -- that identity is the reproduction gate on this re-read.
    """
    from rhm.logit_reading.striatum.task import across_level
    Z = np.load(npz_path)
    out = {}
    for an in ("fd", "tv"):
        pre = f"{an}__"
        if pre + "t0" not in Z:
            continue
        rec = {k[len(pre):]: Z[k] for k in Z.files if k.startswith(pre)}
        n = len(rec["t0"])
        te = np.ones(n, bool)
        cells = {}
        for name, blk in (("linear", cb), ("mlp", "mlp"), ("clean", "clean")):
            if f"R_{blk}_l1_a0" not in rec:
                continue
            cells[name] = across_level(rec, blk, te, lmax=lmax)
            byj = {}
            for jv in range(1, 5):
                mj = te & (np.asarray(rec["j"]) == jv)
                if mj.sum() >= min_j:
                    r = across_level(rec, blk, mj, lmax=lmax)
                    if r:
                        byj[f"j{jv}"] = r
            cells[name + "_by_j"] = byj
            # 2026-09-17 follow-up: ranks taken inside an exact (j, k*) cell.  `j` fixes
            # the edit's width and `k*` the depth at which it becomes Bayes-detectable,
            # and both move the SCALE of the revision; `across_level` ranks within level
            # but pools across them, so a wide or deep edit's large revisions are ranked
            # against narrow ones.  This is the width-and-depth-controlled contrast.
            byjk = {}
            jj_, kk_ = np.asarray(rec["j"]), np.asarray(rec["k_star"])
            for jv in range(1, 5):
                for kv in range(0, 7):
                    m = te & (jj_ == jv) & (kk_ == kv)
                    if m.sum() >= min_j:
                        r = across_level(rec, blk, m, lmax=lmax)
                        if r:
                            byjk[f"j{jv}k{kv}"] = r
            cells[name + "_by_jk"] = byjk
        out[an] = cells
    return out


def sec_mlp_across(cells, a_list=(0, 1), critics=("linear", "mlp", "clean")):
    rows = []
    for cr in critics:
        t = cells.get(cr, {})
        for a in a_list:
            e = t.get(f"a{a}")
            if e:
                rows.append([cr, a, e["n"], fmt(e["mean_rank_cons"]),
                             fmt(e["mean_rank_incons"]), fmt(e["win_rate"]),
                             fmt(e["scalar_rank_mean"])])
    return tbl(rows, ["critic", "a", "n", "rank cons", "rank incons", "win rate",
                      "scalar rank"])


def sec_mlp_across_by_j(cells, a=0, critics=("linear", "mlp", "clean")):
    """Per-`j` breakdown, plus the `n`-weighted mean over the strata -- the pooled row
    above mixes `j` cells, whose consequence profiles differ (junction's first gotcha),
    so the stratified mean is the `j`-controlled version of the same contrast."""
    rows = []
    for cr in critics:
        t = cells.get(cr + "_by_j", {})
        acc = []
        for jv in sorted(t):
            e = t[jv].get(f"a{a}")
            if e:
                rows.append([cr, jv, e["n"], fmt(e["mean_rank_cons"]),
                             fmt(e["mean_rank_incons"]), fmt(e["win_rate"])])
                acc.append((e["n"], e["win_rate"]))
        if acc:
            n = sum(x[0] for x in acc)
            rows.append([cr, "**all j (n-wtd)**", n, "--", "--",
                         fmt(sum(x[0] * x[1] for x in acc) / n)])
    return tbl(rows, ["critic", "j", "n", "rank cons", "rank incons", "win rate"])


GATE_TOL = 5e-3


def sec_mlp_across_by_jk(cells, a=0, critics=("linear", "mlp", "clean")):
    """Per-`(j, k*)` breakdown with ranks taken inside the cell, plus the `n`-weighted
    mean over the cells and the coverage those cells retain against the pooled row set."""
    rows = []
    for cr in critics:
        t = cells.get(cr + "_by_jk", {})
        pooled = cells.get(cr, {}).get(f"a{a}")
        acc = []
        for ck in sorted(t, key=lambda x: (int(x.split("k")[0][1:]), int(x.split("k")[1]))):
            e = t[ck].get(f"a{a}")
            if e:
                rows.append([cr, ck, e["n"], fmt(e["mean_rank_cons"]),
                             fmt(e["mean_rank_incons"]), fmt(e["win_rate"])])
                acc.append((e["n"], e["win_rate"]))
        if acc:
            n = sum(x[0] for x in acc)
            cov = f"coverage {n / pooled['n']:.2f}" if pooled and pooled["n"] else "--"
            rows.append([cr, "**all (j, k*) (n-wtd)**", n, cov, "--",
                         fmt(sum(x[0] * x[1] for x in acc) / n)])
    return tbl(rows, ["critic", "(j, k*)", "n", "rank cons", "rank incons", "win rate"])


def write_mlp_across(args):
    """`--mlp-across`: writes `tables_mlp_across.md` only; touches no banked table."""
    dirs = [args.dir] + [d for d in args.extra_dirs.split(",") if d]
    head = ["# striatum -- the across-level contrast on the MLP critic (re-read, "
            "2026-09-17)", "",
            "Pure re-read of banked `stepNNNNNN_striatum_<tag>.npz` artefacts: no model "
            "is run and nothing is refit. `task.py::across_level` is called on the "
            "identical held-out rows for the trained **linear** critic (the banked "
            "`across_level`), the **MLP** critic on the same block (never read this way "
            "before) and the **clean-only** critic (the banked `across_level_clean`).",
            "", "Reproduction:", "", "```bash",
            "cd experiments   # MODAL_PROFILE=chromatic",
            "D=/v16_s2_L6_m4_distinct/logit_reading",
            "modal volume get rhm-scaling-data $D/traj_a1_s42 <dir>/traj_a1_s42",
            "modal volume get rhm-scaling-data $D/traj_eps01_s42 <dir>/traj_eps01_s42",
            "python -m rhm.logit_reading.striatum.analyze <dir>/traj_a1_s42 "
            "--mlp-across \\",
            "    --extra-dirs <dir>/traj_eps01_s42 --tags a1,swap65k \\",
            "    --out rhm/logit_reading/striatum/results", "```", "",
            "**Reading.** At a fixed `(window, anchor, a)` the state is identical across "
            "query levels, so the contrast between the levels whose answer the edit "
            "changed and the levels whose answer it did not is matched on everything a "
            "state-only readout can see. Levels are ranked within level (`l <= 4`); "
            "`win rate` is the fraction of rows whose mean rank over consequential "
            "levels beats its mean rank over inconsequential levels (0.5 = null, ties "
            "count 0.5). `scalar rank` is the window's mean rank over all valid levels "
            "-- the quantity a scalar readout could express.", "",
            "**Two stratifications.** The pooled row above ranks within level but pools "
            "across edit widths `j` and detection depths `k*`, both of which move the "
            "SCALE of the revision, so a wide or deep edit's large revisions are ranked "
            "against narrow ones. The per-`j` and per-`(j, k*)` tables re-rank inside the "
            "cell, which removes that between-cell scale difference; the `n`-weighted row "
            "is the controlled version of the same contrast. Cells below a 30-row floor "
            "are dropped, so `coverage` reports the share of the pooled row set the "
            "surviving cells retain.", ""]
    gate, body, dmax, summ = [], [], [0.0], []
    for d in dirs:
        traj = os.path.basename(d.rstrip("/"))
        for tag in args.tags.split(","):
            runs = load(d, tag)
            for st in sorted(runs):
                npz = os.path.join(d, f"step{st:06d}_striatum_{tag}.npz")
                if not os.path.exists(npz):
                    continue
                cb = runs[st]["critic_block"]
                res = mlp_across(npz, cb)
                for an in ("fd", "tv"):
                    if an not in res:
                        continue
                    for nm, bank in (("linear", "across_level"),
                                     ("clean", "across_level_clean")):
                        bt = runs[st]["tables"].get(an, {}).get(bank, {})
                        for a in (0, 1):
                            x = res[an].get(nm, {}).get(f"a{a}")
                            y = bt.get(f"a{a}")
                            if x and y:
                                dv = abs(x["win_rate"] - y["win_rate"])
                                dmax[0] = max(dmax[0], dv)
                                ok = x["n"] == y["n"] and dv < GATE_TOL
                                gate.append([traj, tag, st, an, f"{nm} a{a}",
                                             fmt(y["win_rate"]), fmt(x["win_rate"]),
                                             "OK" if ok else "**MISMATCH**"])
                    bj = runs[st]["tables"].get(an, {}).get("across_level_by_j", {})
                    for jv in sorted(bj):
                        x = res[an].get("linear_by_j", {}).get(jv, {}).get("a0")
                        y = bj[jv].get("a0")
                        if x and y:
                            dv = abs(x["win_rate"] - y["win_rate"])
                            dmax[0] = max(dmax[0], dv)
                            ok = x["n"] == y["n"] and dv < GATE_TOL
                            gate.append([traj, tag, st, an, f"linear {jv} a0",
                                         fmt(y["win_rate"]), fmt(x["win_rate"]),
                                         "OK" if ok else "**MISMATCH**"])
                    anm = ("edit onset (first_diff)" if an == "fd"
                           else "t_v (Bayesian-detectable violation)")
                    for cr in ("linear", "mlp", "clean"):
                        po = res[an].get(cr, {}).get("a0")
                        if not po:
                            continue
                        def _w(sfx):
                            t = res[an].get(cr + sfx, {})
                            acc = [(e["n"], e["win_rate"]) for v in t.values()
                                   for e in [v.get("a0")] if e]
                            if not acc:
                                return "--", "--"
                            n = sum(x[0] for x in acc)
                            return (fmt(sum(x[0] * x[1] for x in acc) / n),
                                    f"{n / po['n']:.2f}" if po["n"] else "--")
                        wj, _ = _w("_by_j")
                        wjk, cov = _w("_by_jk")
                        summ.append([traj.replace("traj_", ""), tag, st,
                                     "t_v" if an == "tv" else "onset", cr, po["n"],
                                     fmt(po["win_rate"]), wj, wjk, cov])
                    body += [f"## `{traj}` / venue `{tag}` / step {st} / anchor: {anm}",
                             "", f"Critic block `{cb}`.", "",
                             sec_mlp_across(res[an]), "",
                             "Per-`j` breakdown (a = 0):", "",
                             sec_mlp_across_by_j(res[an]), "",
                             "Per-`(j, k*)` breakdown (a = 0) -- ranks taken inside an "
                             "exact width-and-depth cell:", "",
                             sec_mlp_across_by_jk(res[an]), ""]
    md = head + [
        "## Reproduction gate: banked JSON vs this re-read", "",
        "Every `linear` and `clean` win rate below is recomputed from the `.npz` and "
        "compared with the banked `tables.<anchor>.across_level{,_clean}` cell in the "
        "same run's JSON (`n` must match too). The `.npz` stores the revisions as "
        "`float32` while the banked JSON ranked them in `float64`, so near-ties can flip "
        f"a rank in a small stratum; the tolerance is {GATE_TOL} and the largest "
        f"deviation observed here is **{dmax[0]:.4f}**.", "",
        tbl(gate, ["traj", "venue", "step", "anchor", "cell", "banked", "re-read", ""]),
        "",
        "## Summary: the same contrast under three stratifications", "",
        "`pooled` ranks within level over the whole held-out set; `by j` re-ranks inside "
        "each edit width; `by (j, k*)` re-ranks inside each exact (width, "
        "detection-depth) cell. `j` and `k*` both move the SCALE of the revision, so the "
        "pooled row ranks a wide or deep edit's large revisions against a narrow one's. "
        "`cov` is the share of the pooled rows the surviving `(j, k*)` cells retain "
        "(30-row floor); read only the rows where it is near 1.", "",
        tbl(summ, ["traj", "venue", "step", "anchor", "critic", "n pooled", "pooled",
                   "by j", "by (j, k*)", "cov"]), ""] + body
    open(os.path.join(args.out, "tables_mlp_across.md"), "w").write("\n".join(md) + "\n")
    print("wrote", os.path.join(args.out, "tables_mlp_across.md"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--tags", default="a1,swap65k")
    ap.add_argument("--out", default="rhm/logit_reading/striatum/results")
    ap.add_argument("--figs", default="rhm/logit_reading/striatum/figs")
    ap.add_argument("--mlp-across", action="store_true",
                    help="2026-09-17 re-read: write results/tables_mlp_across.md only "
                         "(the across-level contrast on every stored critic) and exit; "
                         "the banked tables.md is left untouched.")
    ap.add_argument("--extra-dirs", default="",
                    help="comma-separated extra trajectory dirs for --mlp-across")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.mlp_across:
        write_mlp_across(args)
        return
    md = ["# striatum -- tables", "",
          "Regenerated by `python -m rhm.logit_reading.striatum.analyze <dir>`.", ""]
    first = None
    for tag in args.tags.split(","):
        runs = load(args.dir, tag)
        if not runs:
            continue
        first = first or runs
        rm = {}
        for st in sorted(runs):
            npz = os.path.join(args.dir, f"step{st:06d}_striatum_{tag}.npz")
            if os.path.exists(npz):
                rm[st] = rematch(npz, runs[st]["critic_block"])
        md += [f"## Venue `{tag}`", "",
               f"Checkpoints: {sorted(runs)}. Critic block: "
               f"{ {s: runs[s]['critic_block'] for s in sorted(runs)} }.", "",
               "### Actor: clean held-out accuracy by query level (chance 0.0625)", "",
               sec_actor(runs), "",
               "### Critic: held-out R^2 of V[l, d=0] on the realised outcome", "",
               sec_critic(runs), "",
               "### Consequence base rates (last checkpoint, anchor = edit onset)", "",
               sec_base(runs), "", sec_base_kj(runs), "",
               "### What the edit costs at the goal (anchor = edit onset, a = 0)", "",
               sec_damage(runs), ""]
        for an in ("fd", "tv"):
            if an not in runs[sorted(runs)[-1]]["tables"]:
                continue
            nm = ("the edit onset (first_diff; the token is legal in ~82% of swaps)"
                  if an == "fd" else "t_v (the Bayesian-detectable violation)")
            md += [f"## Venue `{tag}` -- anchor: {nm}", "",
                   "#### Cell sizes", "", sec_cells(runs, an), "",
                   "#### AUC(consequence), a = 0 -- pooled", "",
                   sec_auc(runs, an, "auc_cons"), "",
                   "#### AUC(consequence), a = 0 -- matched on (k*, j) and model surprisal",
                   "", sec_auc(runs, an, "auc_cons_matched"), "",
                   "#### AUC(consequence), a = 0 -- matched on j and model surprisal", "",
                   sec_auc(runs, an, "auc_cons_matchedj"), "",
                   "#### AUC(realised damage at the goal: right on orig, wrong on edit), "
                   "a = 0 -- pooled", "", sec_auc(runs, an, "auc_flip"), "",
                   "#### AUC(realised damage), a = 0 -- matched on (k*, j) and surprisal",
                   "", sec_auc(runs, an, "auc_flip_matched"), "",
                   "#### Across-level contrast (identical state, different query level) "
                   "-- trained critic", "", sec_across(runs, an), "",
                   "#### Across-level contrast -- clean-only critic (the exposure floor)",
                   "", sec_across(runs, an, "across_level_clean"), "",
                   "#### Illegal vs legal at matched consequence (swap vs rare, "
                   "consequential episodes only)", "",
                   sec_auc(runs, an, "auc_illegal_given_cons", scores=["R", "nll"]), ""]
            if rm:
                md += ["#### RE-MATCHED locally: strata (k*, j, t0 bucket), "
                       "sign-balanced surprisal pairs -- AUC(consequence), a = 0", "",
                       sec_rematch(rm, runs, an, "cons"), "",
                       "#### RE-MATCHED -- AUC(realised damage at the goal), a = 0", "",
                       sec_rematch(rm, runs, an, "flip"), ""]
        md += ["### Sanity: does V predict the outcome at these rows", "",
               sec_auc(runs, "fd", "auc_outcome", scores=["V", "Vpre"]), ""]
    # --- dated addendum -----------------------------------------------------
    for tag in args.tags.split(","):
        runs = load(args.dir, tag)
        if not runs:
            continue
        st = sorted(runs)[-1]
        npz = os.path.join(args.dir, f"step{st:06d}_striatum_{tag}.npz")
        hx = os.path.join(args.dir, f"step{st:06d}_striatum_{tag}_hexcess.npz")
        if not os.path.exists(npz):
            continue
        for an in ("tv", "fd"):
            adv = addendum(npz, runs[st]["critic_block"], hx if os.path.exists(hx) else None,
                           anchor=an)
            if not adv:
                continue
            md += [f"## ADDENDUM 2026-09-17 -- venue `{tag}`, step {st}, anchor "
                   f"{'t_v' if an == 'tv' else 'edit onset'}", "",
                   "**Label**: realised damage at the goal -- the fixed clean-trained actor "
                   "answered the level-l query correctly on the ORIGINAL window and wrongly "
                   "on the EDITED one (`oo & ~oe`), among rows where it was right on the "
                   "original. Rows are matched on exact (k*, j, anchor-position bucket) with "
                   "sign-balanced nearest-neighbour pairs on the model's surprisal at the "
                   "anchor.", "",
                   "**`excess_1pos`** is the single-position excess at the anchor token, "
                   "`-log q(x_t0) - H(q_{t0-1})`. **`excess_horizon`** is the realised "
                   "`sum_{u=t0}^{t0+8} (NLL_u - H(q_u))` -- the object the banked "
                   "`coeruleus/` head was trained to predict; `_pre` is the same summed from "
                   "`t0-1`. **`banked`** is that head's prediction from the state. "
                   "`X|Y` is the AUC of X inside quantile bins of Y (5 bins, weighted by "
                   "`n_pos * n_neg`); `oof[...]` is an out-of-fold (5-fold) least-squares "
                   "combination of the listed columns.", "",
                   sec_addendum(adv), ""]

    open(os.path.join(args.out, "tables.md"), "w").write("\n".join(md) + "\n")
    print("wrote", os.path.join(args.out, "tables.md"))
    if first:
        figures(first, args.tags, args.figs)


if __name__ == "__main__":
    main()
