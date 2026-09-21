"""abstain, the stream venue's reduction: the same choice, the same gates, the same four
conditionings and the same floors and screens as `analyze.py`, but read at EVERY position of
held-out windows rather than at five offsets after an anchor.

The row unit is now a (window, position) pair. The fit / read split is **by window**, never by
row, because positions inside a window are not independent. The three arms — `edit` (the event
is inside the window), `orig` (its unedited counterpart, position for position) and `quiet`
(fresh clean windows that were never part of any edit) — are pooled into one stream, one policy
is fitted on it, and the realised return is then decomposed by arm and by distance since the
violation. That decomposition is the point: it turns "where the value lives" from a claim about
the neighbourhood of an event into a number on the learner's whole life.

  modal volume get rhm-scaling-data \
      /v16_s2_L6_m4_distinct/logit_reading/traj_a1_s42 <dir>/traj_a1_s42
  python -m rhm.logit_reading.orbitofrontal.abstain.analyze_stream <dir>/traj_a1_s42 \
      --tags a1,swap65k --out rhm/logit_reading/orbitofrontal/abstain/results \
      --figs rhm/logit_reading/orbitofrontal/abstain/figs
"""

import argparse
import glob
import json
import os

import numpy as np

from rhm.logit_reading.striatum.analyze import _auc, fmt, tbl
from rhm.logit_reading.orbitofrontal.abstain.analyze import (
    const_policy, gate_policy, bin_r2, NBINS, NBINS_C, N_POS, N_NLL, N_V, CONDS)

A_LIST = [0, 1, 2, 4, 8]
ARMS = ["edit", "orig", "quiet"]
# the gate roster of the stream venue: the anchor-relative gates of the CPU venue, re-expressed
# per position, plus the two nulls and the ceiling.
GATE_NAMES = ["V_pre", "V", "R", "V_clean", "R_clean", "Vshuf", "Rshuf", "delta_prev",
              "o_prev", "banked_excess", "nll", "H_pre", "dH", "excess", "t0", "noise",
              "noise2", "oracle"]


def _hi(cfgh, dd):
    return cfgh.index(dd)


def build(Z, cfg, l, a, arms=ARMS):
    """Every (window, position) row of every arm, with every gate on it.

    Returns a dict of flat arrays: `o` (the outcome the decision is about), the gates, and the
    bookkeeping (`arm`, `w` the window id within its arm, `t` the absolute position, `since`
    the tokens since the violation or -999 where there was none)."""
    prows = np.asarray(cfg["prows"])
    hz, lev = cfg["horizons"], cfg["levels"]
    li, T = lev.index(l), cfg["T"]
    ha, ha1 = _hi(hz, a), _hi(hz, a + 1)
    nP, nL, nH = len(prows), len(lev), len(hz)
    out = {k: [] for k in ["o", "arm", "w", "t", "since"] + GATE_NAMES}
    for ai, arm in enumerate(arms):
        if f"{arm}__V" not in Z:
            continue
        V = np.asarray(Z[f"{arm}__V"]).reshape(-1, nP, nL, nH).astype(np.float64)
        Vc = np.asarray(Z[f"{arm}__Vc"]).reshape(-1, nP, nL, nH).astype(np.float64)
        Vs = np.asarray(Z[f"{arm}__Vs"]).reshape(-1, nP, nL, nH).astype(np.float64)
        o = np.asarray(Z[f"{arm}__o"]).astype(np.float64)              # (n, T, nL)
        nll = np.asarray(Z[f"{arm}__nll"]).astype(np.float64)          # (n, T)
        H = np.asarray(Z[f"{arm}__H"]).astype(np.float64)
        bex = np.asarray(Z[f"{arm}__bex"]).astype(np.float64)          # (n, nP)
        tv, fd = np.asarray(Z[f"{arm}__t_v"]), np.asarray(Z[f"{arm}__first_diff"])
        et = np.asarray(Z[f"{arm}__etype"])
        n = len(V)
        # a decision at stored position index p needs p-2 (for delta_prev) and t+a <= T-1
        ps = np.arange(2, nP)
        ps = ps[(prows[ps] + a) <= T - 1]
        t = prows[ps]                                                  # (nP',)
        W, Pp = np.meshgrid(np.arange(n), np.arange(len(ps)), indexing="ij")
        flat = lambda x: np.asarray(x).reshape(-1)
        g = {}
        g["V"] = V[:, ps, li, ha]
        g["V_pre"] = V[:, ps - 1, li, ha1]
        g["R"] = g["V"] - g["V_pre"]
        g["V_clean"] = Vc[:, ps, li, ha]
        g["R_clean"] = Vc[:, ps, li, ha] - Vc[:, ps - 1, li, ha1]
        g["Vshuf"] = Vs[:, ps, li, ha]
        g["Rshuf"] = Vs[:, ps, li, ha] - Vs[:, ps - 1, li, ha1]
        # the previous decision point is one token earlier, same offset
        o_prev = o[:, np.clip(t - 1 + a, 0, T - 1), li]
        g["o_prev"] = o_prev
        g["delta_prev"] = o_prev - V[:, ps - 2, li, ha1]
        g["banked_excess"] = bex[:, ps]
        g["nll"] = nll[:, t - 1]
        g["H_pre"] = H[:, t - 1]
        g["dH"] = H[:, t] - H[:, t - 1]
        g["excess"] = nll[:, t - 1] - H[:, t - 1]
        g["t0"] = np.broadcast_to(t.astype(np.float64), (n, len(ps)))
        rg = np.random.default_rng(900 + 17 * l + a + 3 * ai)
        g["noise"] = rg.standard_normal((n, len(ps)))
        g["noise2"] = np.random.default_rng(5000 + 17 * l + a + 3 * ai
                                            ).standard_normal((n, len(ps)))
        y = o[:, np.clip(t + a, 0, T - 1), li]
        g["oracle"] = y
        # distance since the violation; -999 where the window never had one
        ev = np.where(tv > 0, tv, np.where((et == 0) | (et == 1), fd, -1))
        if arm != "edit":
            ev = np.full(n, -1)
        since = np.where(ev[:, None] > 0, t[None, :] - ev[:, None], -999)
        out["o"].append(flat(y))
        out["arm"].append(np.full(y.size, ai, np.int8))
        out["w"].append(flat(W))
        out["t"].append(flat(np.broadcast_to(t, (n, len(ps)))))
        out["since"].append(flat(since))
        for k in GATE_NAMES:
            out[k].append(flat(g[k]))
    if not out["o"]:
        return None
    return {k: np.concatenate(vv) for k, vv in out.items()}


SINCE_BINS = [("before", -998, -1), ("t=0", 0, 0), ("t=1", 1, 1), ("t=2", 2, 2),
              ("t=3-4", 3, 4), ("t=5-8", 5, 8), ("t=9-16", 9, 16), ("t>16", 17, 10 ** 6)]


def subgroups(R):
    """The decomposition: by arm, and inside the `edit` arm by distance since the violation."""
    g = {"whole stream": np.ones(len(R["o"]), bool)}
    for ai, arm in enumerate(ARMS):
        m = R["arm"] == ai
        if m.sum() >= 2000:
            g[f"arm={arm}"] = m
    ed = R["arm"] == 0
    g["edit, quiet part"] = ed & (R["since"] == -999)
    for nm, lo, hi in SINCE_BINS:
        m = ed & (R["since"] >= lo) & (R["since"] <= hi)
        if m.sum() >= 1000:
            g[nm] = m
    return {k: v for k, v in g.items() if v.sum() >= 1000}


def strata_of(R, fm):
    """The conditioning strata, bucketed on the FIT rows only. The edge count is whatever
    `np.unique` leaves after ties, so the product key uses it rather than the nominal bin
    count -- otherwise two different (position, surprisal) pairs can collide onto one key."""
    def b(x, nb):
        e = np.unique(np.quantile(np.asarray(x)[fm], np.linspace(0, 1, nb + 1)[1:-1]))
        return np.clip(np.digitize(np.asarray(x), e), 0, len(e)), len(e) + 1
    bp, kp = b(R["t0"], N_POS)
    bn, kn = b(R["nll"], N_NLL)
    bv, kv = b(R["V"], N_V)
    return {"marg": np.zeros(len(bp), np.int64), "pos": bp,
            "pos_nll": bp * kn + bn, "pos_V": bp * kv + bv}


def cell(R, r_grid=None, seed=0, nbins=NBINS, nbins_c=NBINS_C):
    """One (level, offset) of the stream. Fit / read split BY WINDOW inside each arm."""
    n = len(R["o"])
    key = R["arm"].astype(np.int64) * 10 ** 7 + R["w"]
    uk = np.unique(key)
    pick = np.random.default_rng(seed).permutation(len(uk))[: len(uk) // 2]
    fitw = np.zeros(len(uk), bool)
    fitw[pick] = True
    fm = fitw[np.searchsorted(uk, key)]
    rm = ~fm
    o = R["o"]
    st = strata_of(R, fm)
    p_fit, p_read = float(o[fm].mean()), float(o[rm].mean())
    out = {"n": int(n), "n_fit": int(fm.sum()), "n_read": int(rm.sum()),
           "n_windows": int(len(uk)), "acc_fit": p_fit, "acc_read": p_read,
           "screen": {}, "r": {}}
    wrong = o < 0.5
    for nm in GATE_NAMES:
        s = R[nm]
        v = np.isfinite(s)
        if (fm & v).sum() < 1000:
            continue
        out["screen"][nm] = {"auc_wrong": _auc(s[rm & v], wrong[rm & v]),
                             "bin_r2": bin_r2(s[fm & v], o[fm & v], s[rm & v], o[rm & v])}
    subs = subgroups(R)
    for r_abs in sorted(set(list(r_grid or [0.3, 0.5, 0.7]) + [p_fit])):
        ent = {"r_abs": r_abs, "gates": {}, "floor": {}}
        ret_orac = p_read + (1.0 - p_read) * r_abs
        for cn in CONDS:
            ac = const_policy(o[fm], st[cn][fm], st[cn][rm], r_abs)
            ent["floor"][cn] = {"ret": float(np.where(ac, o[rm], r_abs).mean())}
        ent["ret_oracle"] = ret_orac
        ent["prize"] = {cn: ret_orac - ent["floor"][cn]["ret"] for cn in CONDS}
        for nm in GATE_NAMES:
            s = R[nm]
            v = np.isfinite(s)
            fm2, rm2 = fm & v, rm & v
            if fm2.sum() < 1000:
                continue
            e = {"n": int(rm2.sum())}
            for cn in CONDS:
                nb = nbins if cn == "marg" else nbins_c
                ag = gate_policy(s[fm2], o[fm2], st[cn][fm2], s[rm2], st[cn][rm2], r_abs,
                                 nbins=nb)
                ac = const_policy(o[fm2], st[cn][fm2], st[cn][rm2], r_abs)
                per, per_c = np.where(ag, o[rm2], r_abs), np.where(ac, o[rm2], r_abs)
                rc = float(per_c.mean())
                pr = float(o[rm2].mean())
                ro = pr + (1.0 - pr) * r_abs
                gain = float(per.mean() - rc)
                gse = float((per - per_c).std(ddof=1) / np.sqrt(len(per)))
                e[cn] = {"ret": float(per.mean()), "gain": gain,
                         "abstain_rate": float((~ag).mean()),
                         "share": gain / (ro - rc) if (ro - rc) > 5e-3 else float("nan"),
                         "share_se": gse / (ro - rc) if (ro - rc) > 5e-3 else float("nan")}
                if cn == "pos" and abs(r_abs - p_fit) < 1e-12:
                    # the decomposition rides the `pos` policy, at the bite point only
                    sub = {}
                    for gn, gmask in subs.items():
                        m2 = rm2 & gmask
                        if m2.sum() < 500:
                            continue
                        ag2 = gate_policy(s[fm2], o[fm2], st[cn][fm2], s[m2], st[cn][m2],
                                          r_abs, nbins=nb)
                        ac2 = const_policy(o[fm2], st[cn][fm2], st[cn][m2], r_abs)
                        p2, pc2 = np.where(ag2, o[m2], r_abs), np.where(ac2, o[m2], r_abs)
                        pr2 = float(o[m2].mean())
                        rc2, ro2 = float(pc2.mean()), pr2 + (1.0 - pr2) * r_abs
                        sub[gn] = {"n": int(m2.sum()), "acc": pr2,
                                   "gain": float(p2.mean() - rc2),
                                   "prize": ro2 - rc2,
                                   "share": (float(p2.mean()) - rc2) / (ro2 - rc2)
                                   if (ro2 - rc2) > 5e-3 else float("nan")}
                    e["by"] = sub
            ent["gates"][nm] = e
        out["r"][f"{r_abs:.6f}"] = ent
    return out


def bite(c):
    return c["r"][f"{c['acc_fit']:.6f}"]


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------

def sec_repro(cfg, banked_actor, banked_r2):
    pb = cfg["primary_block"]
    rows = [["actor clean acc, this cell"] +
            [fmt(cfg["actor_clean_acc"].get(f"{pb}/l{l}"), 4) for l in range(1, 7)],
            ["banked striatum column"] + [fmt(x, 4) for x in banked_actor]]
    t1 = tbl(rows, ["quantity"] + [f"l={l}" for l in range(1, 7)])
    r2 = [["critic `l1_d0` val R2, trained diet", fmt(cfg["critic_val_r2"]["l1_d0"], 5),
           fmt(banked_r2[0], 5) if banked_r2[0] else "--"],
          ["critic `l1_d0` val R2, clean-only",
           fmt(cfg["clean_critic_val_r2"]["l1_d0"], 5),
           fmt(banked_r2[1], 5) if banked_r2[1] else "--"]]
    return t1 + "\n\n" + tbl(r2, ["quantity", "this cell", "banked striatum"])


def sec_conds(cells, a=0, gates=None):
    ls = sorted({l for (l, aa) in cells if aa == a})
    rows = []
    for cn in CONDS:
        rows.append(["(floor)", cn, "prize"] +
                    [fmt(bite(cells[(l, a)])["prize"][cn]) for l in ls])
    rows.append(["(floor)", "--", "acc read"] +
                [fmt(cells[(l, a)]["acc_read"]) for l in ls])
    for nm in (gates or GATE_NAMES):
        for cn in CONDS:
            r, any_ = [nm, cn, "share"], False
            for l in ls:
                gg = bite(cells[(l, a)])["gates"].get(nm, {}).get(cn)
                r.append(f'{gg["share"]:+.3f}' if gg and np.isfinite(gg["share"]) else "--")
                any_ = any_ or bool(gg)
            if any_:
                rows.append(r)
    return tbl(rows, ["gate", "conditioning", "quantity"] + [f"l={l}" for l in ls])


def sec_bite(cells, cond="pos", a=0, gates=None):
    ls = sorted({l for (l, aa) in cells if aa == a})
    top = [["--", "acc read"], ["--", "floor return"], ["--", "prize"], ["--", "n read rows"]]
    for l in ls:
        c = cells[(l, a)]
        e = bite(c)
        for b, vv in zip(top, [fmt(c["acc_read"]), fmt(e["floor"][cond]["ret"]),
                               fmt(e["prize"][cond]), str(c["n_read"])]):
            b.append(vv)
    rows = list(top)
    for nm in (gates or GATE_NAMES):
        r1, r2, any_ = [nm, "share of prize"], [nm, "abstain rate"], False
        for l in ls:
            gg = bite(cells[(l, a)])["gates"].get(nm, {}).get(cond)
            r1.append(f'{gg["share"]:+.3f}+-{gg["share_se"]:.3f}'
                      if gg and np.isfinite(gg["share"]) else "--")
            r2.append(fmt(gg["abstain_rate"]) if gg else "--")
            any_ = any_ or bool(gg)
        if any_:
            rows += [r1, r2]
    return tbl(rows, ["gate", "quantity"] + [f"l={l}" for l in ls])


def sec_where(cells, l=2, a=0, gates=None, what="share"):
    c = bite(cells[(l, a)])
    names = []
    for nm in (gates or GATE_NAMES):
        for k in c["gates"].get(nm, {}).get("by", {}):
            if k not in names:
                names.append(k)
    any_by = next((c["gates"][nm]["by"] for nm in (gates or GATE_NAMES)
                   if c["gates"].get(nm, {}).get("by")), {})
    rows = [["(rows)", "n"] + [str(any_by[k]["n"]) if k in any_by else "--" for k in names],
            ["(rows)", "acc"] + [fmt(any_by[k]["acc"]) if k in any_by else "--"
                                 for k in names],
            ["(rows)", "prize"] + [fmt(any_by[k]["prize"]) if k in any_by else "--"
                                   for k in names]]
    for nm in (gates or GATE_NAMES):
        by = c["gates"].get(nm, {}).get("by", {})
        if not by:
            continue
        rows.append([nm, what] + [fmt(by[k][what], 3 if what == "share" else 4)
                                  if k in by else "--" for k in names])
    return tbl(rows, ["gate", "quantity"] + names)


def sec_screen(cells, a=0):
    ls = sorted({l for (l, aa) in cells if aa == a})
    rows = []
    for nm in GATE_NAMES:
        r1, r2, any_ = [nm, "AUC(wrong)"], [nm, "binned R2"], False
        for l in ls:
            s = cells[(l, a)]["screen"].get(nm)
            r1.append(fmt(s["auc_wrong"]) if s else "--")
            r2.append(fmt(s["bin_r2"]) if s else "--")
            any_ = any_ or bool(s)
        if any_:
            rows += [r1, r2]
    return tbl(rows, ["gate", "quantity"] + [f"l={l}" for l in ls])


def sec_offsets(cells, cond="pos", gates=None):
    ls = sorted({l for (l, _) in cells})
    rows = []
    for nm in (gates or GATE_NAMES):
        for l in ls:
            r, any_ = [nm, f"l={l}"], False
            for a in A_LIST:
                gg = (bite(cells[(l, a)])["gates"].get(nm, {}).get(cond)
                      if (l, a) in cells else None)
                r.append(f'{gg["share"]:+.3f}' if gg and np.isfinite(gg["share"]) else "--")
                any_ = any_ or bool(gg)
            if any_:
                rows.append(r)
    return tbl(rows, ["gate", "level"] + [f"a={a}" for a in A_LIST])


# ---------------------------------------------------------------------------

MAIN = ["V_pre", "V", "R", "V_clean", "R_clean", "Vshuf", "Rshuf", "delta_prev", "o_prev",
        "banked_excess", "nll", "H_pre", "dH", "excess", "t0", "noise", "noise2", "oracle"]
BANKED_ACTOR = [0.9038, 0.8154, 0.7050, 0.5239, 0.3249, 0.1407]
BANKED_R2 = {"a1": (0.21821, 0.22714), "swap65k": (None, None)}


def fig_stream(cells, path, tag, l=2, a=0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    show = ["V_pre", "V", "R", "delta_prev", "o_prev", "nll", "excess", "banked_excess",
            "Rshuf", "t0", "noise", "oracle"]
    fig, ax = plt.subplots(1, 3, figsize=(18, 4.8))
    ls = sorted({x for (x, aa) in cells if aa == a})
    w = 0.82 / len(CONDS)
    for i, cn in enumerate(CONDS):
        ys, es = [], []
        for nm in show:
            gg = bite(cells[(l, a)])["gates"].get(nm, {}).get(cn)
            ys.append(gg["share"] if gg else np.nan)
            es.append(gg["share_se"] if gg else np.nan)
        ax[0].bar(np.arange(len(show)) + i * w - 0.41, ys, w, yerr=es,
                  error_kw={"lw": 0.5}, label=cn)
    ax[0].axhline(0, color="k", lw=0.8)
    ax[0].set_xticks(np.arange(len(show)))
    ax[0].set_xticklabels(show, rotation=45, ha="right", fontsize=7)
    ax[0].set_ylabel("share of the oracle prize")
    ax[0].set_title(f"{tag} whole stream, l={l}, a={a}")
    ax[0].legend(fontsize=7)

    by = bite(cells[(l, a)])["gates"].get("V", {}).get("by", {})
    names = [k for k in by if k not in ("whole stream",)]
    for nm in ("V_pre", "V", "R", "delta_prev", "nll", "Rshuf"):
        b = bite(cells[(l, a)])["gates"].get(nm, {}).get("by", {})
        if b:
            ax[1].plot(range(len(names)), [b[k]["share"] if k in b else np.nan
                                           for k in names], marker="o", ms=4, label=nm)
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_xticks(range(len(names)))
    ax[1].set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax[1].set_ylabel("share of the prize")
    ax[1].set_title("where the value lives, on the stream")
    ax[1].legend(fontsize=7)

    for nm in ("V_pre", "V", "R", "delta_prev", "nll", "Rshuf"):
        for lv in ls:
            ys = [bite(cells[(lv, aa)])["gates"].get(nm, {}).get("pos", {}).get("share", np.nan)
                  if (lv, aa) in cells else np.nan for aa in A_LIST]
            ax[2].plot(A_LIST, ys, marker="o", ms=3, alpha=0.28 + 0.18 * lv,
                       color=f"C{('V_pre', 'V', 'R', 'delta_prev', 'nll', 'Rshuf').index(nm)}",
                       label=nm if lv == 2 else None)
    ax[2].axhline(0, color="k", lw=0.8)
    ax[2].set_xlabel("query offset a")
    ax[2].set_ylabel("share of the prize")
    ax[2].set_title("by offset, inside position strata")
    ax[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--tags", default="a1,swap65k")
    ap.add_argument("--out", default=None)
    ap.add_argument("--figs", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--levels", default="1,2,3,4")
    args = ap.parse_args()
    levels = [int(x) for x in args.levels.split(",")]

    L = []
    w = L.append
    w("# abstain, the stream venue — tables\n")
    w("Facts only. Generated by `analyze_stream.py` from "
      "`step*_abstream_{a1,swap65k}.npz`, written by `task.py`. The row unit is a "
      "(window, position) pair; the fit / read split is **by window**, so no read row shares "
      "a window with a fit row. Same gates, same quantile step policy, same per-stratum "
      "constant floors, same four conditionings and the same binned-`R2` screen as "
      "[`results/tables.md`](tables.md) — see that file's header for all of them.\n")
    w("**Arms.** `edit` = the edited stimulus test windows; `orig` = their unedited "
      "counterparts, position for position; `quiet` = fresh clean windows that were never "
      "part of any edit. All three are pooled into one stream, ONE policy is fitted on it, "
      "and the realised return is decomposed afterwards by arm and by distance since the "
      "violation (`t=0` is the violating token itself, `before` is the same window before "
      "it, `edit, quiet part` is the rows of `edit` windows that never had a violation at "
      "all).\n")
    w("**Per-position gates.** `V` = `V[l,a](s_t)`; `V_pre` = `V[l,a+1](s_(t-1))`; "
      "`R = V - V_pre`; `V_clean` / `R_clean` the clean-only critic's, `Vshuf` / `Rshuf` the "
      "shuffled-outcome critic's. `o_prev` is the previous position's realised outcome at the "
      "same offset and `delta_prev = o_prev - V[l,a+1](s_(t-2))` its outcome surprise. `nll`, "
      "`H_pre`, `dH`, `excess` are the model's own quantities at the token being decided on. "
      "`t0` is the bare position, `noise`/`noise2` are pure noise (the tax), `oracle` is the "
      "realised outcome. The **log Bayes factor is absent by construction** — it is "
      "anchor-relative and would need `v` branch rollouts at every position; `task.py`'s "
      "docstring and the artefact's `config` record why.\n")

    FIG = {}
    for tag in args.tags.split(","):
        fs = sorted(glob.glob(os.path.join(args.dir, f"step*_abstream_{tag}.npz")))
        fs = [f for f in fs if "_smoke" not in f]
        for f in fs:
            cfg = json.load(open(f[:-4] + ".json"))
            Z = np.load(f)
            for k in Z.files:                     # touch every member after the fetch
                _ = Z[k].shape
            cells = {}
            for l in levels:
                for a in A_LIST:
                    R = build(Z, cfg, l, a)
                    if R is None:
                        continue
                    cells[(l, a)] = cell(R, seed=args.seed)
                    print(f"  {tag} l{l} a{a}: {cells[(l, a)]['n']} rows "
                          f"({cells[(l, a)]['n_windows']} windows)", flush=True)
            FIG[tag] = cells
            w(f"\n## {tag}, step {cfg['step']}, the whole stream\n")
            w(f"{cfg['n_test_used']} held-out stimulus windows x 2 arms + "
              f"{cfg['n_quiet']} fresh clean windows; positions {cfg['prows'][0]}-"
              f"{cfg['prows'][-1]} of {cfg['T']}; "
              f"{cells[(2, 0)]['n']} rows at `l=2, a=0`, fit / read "
              f"{cells[(2, 0)]['n_fit']} / {cells[(2, 0)]['n_read']} by window.\n")
            w("### 0. Reproduction gate\n")
            w(sec_repro(cfg, BANKED_ACTOR, BANKED_R2.get(tag, (None, None))))
            w("")
            w("### 1. The decision on the whole stream, `a = 0`, inside position strata\n")
            w(sec_bite(cells, cond="pos", a=0, gates=MAIN))
            w("")
            w("### 2. All four conditionings, `a = 0`\n")
            w(sec_conds(cells, a=0, gates=MAIN))
            w("")
            w("### 3. Where the value lives on the stream (`l = 2`, `a = 0`, `pos` policy)\n")
            w("Share of each subgroup's own prize; one policy, fitted once on the whole "
              "stream.\n")
            w(sec_where(cells, l=2, a=0, gates=MAIN))
            w("")
            w("Return above the per-stratum floor, in outcome units:\n")
            w(sec_where(cells, l=2, a=0, gates=MAIN, what="gain"))
            w("")
            w("### 4. The screen, `a = 0`\n")
            w(sec_screen(cells, a=0))
            w("")
            w("### 5. By query offset, inside position strata\n")
            w(sec_offsets(cells, cond="pos", gates=MAIN))
            w("")
            w("### 6. The decision at `a = 4`, inside position strata\n")
            w(sec_bite(cells, cond="pos", a=4, gates=MAIN))
            w("")
            w("### 7. Where the value lives at `a = 4`\n")
            w(sec_where(cells, l=2, a=4, gates=MAIN))
            w("")

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        p = os.path.join(args.out, "tables_stream.md")
        open(p, "w").write("\n".join(L) + "\n")
        print("wrote", p)
    if args.figs:
        os.makedirs(args.figs, exist_ok=True)
        for tag, cells in FIG.items():
            fig_stream(cells, os.path.join(args.figs, f"abstream_{tag}.png"), tag)
        print("wrote figures ->", args.figs)


if __name__ == "__main__":
    main()
