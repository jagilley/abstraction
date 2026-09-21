"""abstain — the value reading gets a consumer: commit-or-abstain on the banked striatum cells.

The CPU venue. Every number comes off `step*_striatum_{a1,swap65k}.npz`, which store, per test
row and per query offset `a`, the critic's level `V`, its pre-event level `Vpre`, the revision
`R`, the realised outcome `oe`, the banked coeruleus excess head, the model's surprisal and
entropy at the anchor, and the row's `etype`, `k_star`, `j`, `t0`.

The choice, per (venue, anchor, level `l`, offset `a`): the actor either ANSWERS, returning the
realised outcome `oe in {0, 1}`, or ABSTAINS, returning a fixed `r_abs`. A gate is any scalar the
decider could read at the query; a gate's POLICY is coeruleus's quantile step function -- bin the
gate on a fit split, take the return-maximising action in each bin, read it out on held-out rows.
Nothing in a gate's fit sees the outcome of a read row.

Three readings of every gate, because the marginal one is confounded: the MARGINAL policy; the
policy fitted INSIDE strata of the anchor position (striatum's matcher gotcha in decision
currency -- the raw position is free bookkeeping and is itself a strong gate); and inside strata
of position x the model's own surprisal (striatum's addendum in decision currency -- what the
value reading adds over the surprise readout). The floor moves with the conditioning: it is the
best per-stratum CONSTANT, so a share is always "what the gate adds to what the decider already
had".

  modal volume get rhm-scaling-data \
      /v16_s2_L6_m4_distinct/logit_reading/traj_a1_s42 <dir>/traj_a1_s42
  python -m rhm.logit_reading.orbitofrontal.abstain.analyze <dir>/traj_a1_s42 \
      --tags a1,swap65k --out rhm/logit_reading/orbitofrontal/abstain/results \
      --figs rhm/logit_reading/orbitofrontal/abstain/figs
"""

import argparse
import glob
import os

import numpy as np

from rhm.logit_reading.striatum.analyze import LEVELS, A_LIST, _auc, fmt, tbl, rematch

CB = "post_block7"
NBINS = 10          # quantile bins of the gate for the MARGINAL policy and the R^2 screen
NBINS_C = 6         # quantile bins of the gate INSIDE each conditioning stratum
N_POS = 4           # quantile buckets of the anchor position
N_NLL = 3           # quantile buckets of the model's surprisal at the anchor
N_V = 3             # quantile buckets of the critic's level V at the event
MIN_N = 30          # a fit bin smaller than this falls back to its stratum's constant action
R_GRID = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]

# The gate roster.  A gate is a scalar the decider reads before the query resolves.  `oracle`
# is the realised outcome itself (the ceiling); `probe_cons` is the in-container ridge on the
# structural-consequence label (an oracle LABEL read off the state); `Rshuf` is the
# shuffled-outcome critic -- a random direction of the state through the same machinery, which
# is the honest floor for "a state readout gates"; `t0` is the bare anchor position (free
# bookkeeping, no state); `noise` is pure noise and measures the binning tax.
GATES = [
    ("V_pre",         lambda g, l, a: g(f"Vpre_l{l}_a{a}")),
    ("V",             lambda g, l, a: g(f"V_l{l}_a{a}")),
    ("R",             lambda g, l, a: g(f"R_{CB}_l{l}_a{a}")),
    ("R_clean",       lambda g, l, a: g(f"R_clean_l{l}_a{a}")),
    ("R_mlp",         lambda g, l, a: g(f"R_mlp_l{l}_a{a}")),
    ("R_embed",       lambda g, l, a: g(f"R_post_embed_l{l}_a{a}")),
    ("Rshuf",         lambda g, l, a: g(f"Rsh_l{l}_a{a}")),
    ("delta_prev",    None),                       # filled in by `gate_scores`
    ("banked_excess", lambda g, l, a: g("banked_excess")),
    ("nll",           lambda g, l, a: g("nll_e")),
    ("H_pre",         lambda g, l, a: g("H_pre")),
    ("dH",            lambda g, l, a: g("dH_e")),
    ("excess",        lambda g, l, a: g("excess_e")),
    ("o_prev",        None),                       # filled in by `gate_scores`
    ("probe_cons",    lambda g, l, a: g(f"probe_cons_l{l}_a{a}")),
    ("t0",            lambda g, l, a: g("t0").astype(np.float64)),
    ("noise",         lambda g, l, a: np.random.default_rng(
        1234 + 17 * l + a).standard_normal(len(g("t0")))),
    ("noise2",        lambda g, l, a: np.random.default_rng(
        77000 + 17 * l + a).standard_normal(len(g("t0")))),
    ("noise3",        lambda g, l, a: np.random.default_rng(
        420000 + 17 * l + a).standard_normal(len(g("t0")))),
    ("oracle",        lambda g, l, a: g(f"oe_l{l}_a{a}").astype(np.float64)),
]
GATE_NAMES = [n for n, _ in GATES]
PREV_A = {1: 0, 2: 1, 4: 2, 8: 4}
CONDS = ["marg", "pos", "pos_nll", "pos_V"]
MAIN = ["V_pre", "V", "R", "R_clean", "R_mlp", "R_embed", "Rshuf", "delta_prev", "o_prev",
        "banked_excess", "nll", "H_pre", "dH", "excess", "probe_cons", "t0", "noise",
        "noise2", "noise3", "oracle"]


# ---------------------------------------------------------------------------
# the policy: coeruleus's quantile step function, here over a binary action,
# optionally fitted inside strata of what the decider gets for free
# ---------------------------------------------------------------------------

def _edges(x, nb):
    """Bin edges for a gate. Quantile edges, except for a gate with few distinct values
    (`oracle` is binary, `t0` and `j` are small integers), where the quantile grid silently
    collapses; there the edges are the midpoints between the values themselves."""
    x = np.asarray(x, np.float64)
    u = np.unique(x)
    if len(u) <= nb:
        return (u[1:] + u[:-1]) / 2.0
    return np.unique(np.quantile(x, np.linspace(0, 1, nb + 1)[1:-1]))


def const_policy(o_fit, strat_fit, strat_read, r_abs, min_n=MIN_N):
    """The best CONSTANT action, per stratum, chosen on fit rows. The floor."""
    glob_act = bool(o_fit.mean() >= r_abs)
    act = {}
    for s in np.unique(strat_fit):
        m = strat_fit == s
        act[s] = bool(o_fit[m].mean() >= r_abs) if m.sum() >= min_n else glob_act
    return np.array([act.get(s, glob_act) for s in strat_read], bool)


def gate_policy(g_fit, o_fit, strat_fit, g_read, strat_read, r_abs,
                nbins=NBINS, min_n=MIN_N):
    """The step function, fitted inside each stratum: ANSWER iff the (stratum, gate-bin)'s
    realised accuracy on the FIT rows beats `r_abs`. Falls back to the stratum's constant, and
    then to the cell-wide constant, where a cell is thin."""
    g_fit, g_read = np.asarray(g_fit, np.float64), np.asarray(g_read, np.float64)
    glob_act = bool(o_fit.mean() >= r_abs)
    act_read = np.full(len(g_read), glob_act)
    for s in np.unique(strat_fit):
        mf, mr = strat_fit == s, strat_read == s
        if not mr.any():
            continue
        sa = bool(o_fit[mf].mean() >= r_abs) if mf.sum() >= min_n else glob_act
        if mf.sum() < 2 * min_n:
            act_read[mr] = sa
            continue
        e = _edges(g_fit[mf], nbins)
        bf = np.clip(np.digitize(g_fit[mf], e), 0, len(e))
        of = o_fit[mf]
        a_ = np.full(len(e) + 1, sa)
        for i in range(len(e) + 1):
            mm = bf == i
            if mm.sum() >= min_n:
                a_[i] = bool(of[mm].mean() >= r_abs)
        act_read[mr] = a_[np.clip(np.digitize(g_read[mr], e), 0, len(e))]
    return act_read


def bin_r2(g_fit, y_fit, g_read, y_read, nbins=NBINS, min_n=20):
    """basalis A5's screen: how much of the STAKE the gate recovers THROUGH ITS OWN MACHINERY.

    The same quantile step function the policy uses, here predicting the stake rather than
    choosing an action, fitted on the fit rows and scored out of sample. Monotone-invariant.
    The stake is `outcome - r_abs`, an additive shift of the outcome, so this number does not
    depend on `r_abs`."""
    g_fit, y_fit = np.asarray(g_fit, np.float64), np.asarray(y_fit, np.float64)
    edges = _edges(g_fit, nbins)
    b = np.clip(np.digitize(g_fit, edges), 0, len(edges))
    mu = np.array([y_fit[b == i].mean() if (b == i).sum() >= min_n else y_fit.mean()
                   for i in range(len(edges) + 1)])
    pred = mu[np.clip(np.digitize(g_read, edges), 0, len(edges))]
    return float(1.0 - ((y_read - pred) ** 2).mean() / max(y_read.var(), 1e-12))


# ---------------------------------------------------------------------------
# one cell = (venue, anchor, level, offset)
# ---------------------------------------------------------------------------

def split_rows(n, seed=0):
    """A deterministic 50/50 fit / read split of the banked TEST rows, shared by every gate,
    every level and every offset inside a venue, so the arms are read on identical rows."""
    p = np.random.default_rng(seed).permutation(n)
    fit = np.zeros(n, bool)
    fit[p[: n // 2]] = True
    return fit, ~fit


def gate_scores(Z, anchor, l, a):
    """Every gate's scalar on every banked row. NaN where the gate does not exist there."""
    pre = f"{anchor}__"
    g = lambda k: np.asarray(Z[pre + k]) if pre + k in Z else None
    out = {}
    for nm, fn in GATES:
        if nm in ("delta_prev", "o_prev"):
            if a not in PREV_A:
                continue
            ap = PREV_A[a]
            vp, okp = g(f"Vpre_l{l}_a{ap}"), g(f"ok_l{l}_a{ap}")
            op = g(f"oe_l{l}_a{ap}")
            if vp is None or op is None:
                continue
            s = (op.astype(np.float64) - vp if nm == "delta_prev"
                 else op.astype(np.float64))
            out[nm] = np.where(okp.astype(bool), s, np.nan)
            continue
        v = fn(g, l, a)
        if v is not None:
            out[nm] = np.asarray(v, np.float64)
    return out


def make_strata(Z, anchor, l, a, fm):
    """The conditioning strata: what the decider gets for free. Bucketed on FIT rows only.

    `pos_nll` and `pos_V` are the two halves of striatum's addendum in decision currency: what
    the value reading adds over the model's own surprise, and what the surprise readout adds
    over the value level."""
    pre = f"{anchor}__"
    pos = np.asarray(Z[pre + "t0"], np.float64)
    nll = np.asarray(Z[pre + "nll_e"], np.float64)
    vv = np.asarray(Z[pre + f"V_l{l}_a{a}"], np.float64)
    q = lambda x, nb: np.unique(np.quantile(x[fm], np.linspace(0, 1, nb + 1)[1:-1]))
    ep, en, ev = q(pos, N_POS), q(nll, N_NLL), q(vv, N_V)
    bp = np.clip(np.digitize(pos, ep), 0, len(ep))
    bn = np.clip(np.digitize(nll, en), 0, len(en))
    bv = np.clip(np.digitize(vv, ev), 0, len(ev))
    return {"marg": np.zeros(len(pos), np.int64), "pos": bp,
            "pos_nll": bp * (len(en) + 1) + bn, "pos_V": bp * (len(ev) + 1) + bv}


def cell(Z, anchor, l, a, r_list, fit_m, read_m):
    pre = f"{anchor}__"
    if pre + f"ok_l{l}_a{a}" not in Z:
        return None
    ok = np.asarray(Z[pre + f"ok_l{l}_a{a}"]).astype(bool)
    o = np.asarray(Z[pre + f"oe_l{l}_a{a}"]).astype(np.float64)
    fm, rm = fit_m & ok, read_m & ok
    if fm.sum() < 400 or rm.sum() < 400:
        return None
    scores = gate_scores(Z, anchor, l, a)
    strata = make_strata(Z, anchor, l, a, fm)

    p_fit, p_read = float(o[fm].mean()), float(o[rm].mean())
    out = {"n_fit": int(fm.sum()), "n_read": int(rm.sum()),
           "acc_fit": p_fit, "acc_read": p_read, "r": {},
           "n_strata": {k: int(len(np.unique(v[fm]))) for k, v in strata.items()}}

    wrong = (o < 0.5)                      # the r_abs-independent screen, computed once
    out["screen"] = {}
    for nm, s in scores.items():
        val = np.isfinite(s)
        fm2, rm2 = fm & val, rm & val
        if fm2.sum() < 400 or rm2.sum() < 400:
            continue
        out["screen"][nm] = {"auc_wrong": _auc(s[rm2], wrong[rm2]),
                             "bin_r2": bin_r2(s[fm2], o[fm2], s[rm2], o[rm2]),
                             "n": int(rm2.sum())}

    for r_abs in r_list:
        ent = {"r_abs": r_abs, "gates": {}, "floor": {}}
        ret_orac = p_read + (1.0 - p_read) * r_abs
        for cn in CONDS:
            st = strata[cn]
            ac = const_policy(o[fm], st[fm], st[rm], r_abs)
            per_c = np.where(ac, o[rm], r_abs)
            ent["floor"][cn] = {"ret": float(per_c.mean()),
                                "abstain_rate": float((~ac).mean())}
        ent["ret_oracle"] = ret_orac
        ent["prize"] = {cn: ret_orac - ent["floor"][cn]["ret"] for cn in CONDS}
        for nm, s in scores.items():
            val = np.isfinite(s)
            fm2, rm2 = fm & val, rm & val
            if fm2.sum() < 400 or rm2.sum() < 400:
                continue
            pr = float(o[rm2].mean())
            ro = pr + (1.0 - pr) * r_abs
            e = {"n": int(rm2.sum())}
            for cn in CONDS:
                st = strata[cn]
                nb = NBINS if cn == "marg" else NBINS_C
                ag = gate_policy(s[fm2], o[fm2], st[fm2], s[rm2], st[rm2], r_abs, nbins=nb)
                ac = const_policy(o[fm2], st[fm2], st[rm2], r_abs)
                per, per_c = np.where(ag, o[rm2], r_abs), np.where(ac, o[rm2], r_abs)
                rc = float(per_c.mean())
                gain = float(per.mean() - rc)
                gse = float((per - per_c).std(ddof=1) / np.sqrt(len(per)))
                e[cn] = {"ret": float(per.mean()), "gain": gain,
                         "abstain_rate": float((~ag).mean()),
                         "share": gain / (ro - rc) if (ro - rc) > 5e-3 else float("nan"),
                         "share_se": gse / (ro - rc) if (ro - rc) > 5e-3 else float("nan")}
            ent["gates"][nm] = e
        out["r"][f"{r_abs:.6f}"] = ent
    return out


def subgroup_returns(Z, anchor, l, a, r_abs, fit_m, read_m, gate_names, cond="pos"):
    """The same policy, fitted on the whole fit half, decomposed over subgroups of the read
    rows. This is where a gate's value LIVES, not a separate policy per subgroup."""
    pre = f"{anchor}__"
    ok = np.asarray(Z[pre + f"ok_l{l}_a{a}"]).astype(bool)
    o = np.asarray(Z[pre + f"oe_l{l}_a{a}"]).astype(np.float64)
    fm, rm = fit_m & ok, read_m & ok
    et, ks = np.asarray(Z[pre + "etype"]), np.asarray(Z[pre + "k_star"])
    st = make_strata(Z, anchor, l, a, fm)[cond]
    groups = {"all": np.ones(len(o), bool)}
    for nm_, code in (("swap", 0), ("rare", 1), ("quiet", 2)):
        if (et == code).sum() >= 300:
            groups[nm_] = et == code
    for kk in np.unique(ks):
        m = ks == kk
        if m.sum() >= 600:
            groups[f"k*={int(kk)}"] = m
    scores = gate_scores(Z, anchor, l, a)
    nb = NBINS if cond == "marg" else NBINS_C
    out = {}
    for gn in gate_names:
        s = scores.get(gn)
        if s is None:
            continue
        val = np.isfinite(s)
        fm2 = fm & val
        if fm2.sum() < 400:
            continue
        rows = {}
        for gname, gm in groups.items():
            m = rm & val & gm
            if m.sum() < 100:
                continue
            ag = gate_policy(s[fm2], o[fm2], st[fm2], s[m], st[m], r_abs, nbins=nb)
            ac = const_policy(o[fm2], st[fm2], st[m], r_abs)
            per, per_c = np.where(ag, o[m], r_abs), np.where(ac, o[m], r_abs)
            pr = float(o[m].mean())
            rc, ro = float(per_c.mean()), pr + (1.0 - pr) * r_abs
            rows[gname] = {"n": int(m.sum()), "acc": pr, "ret": float(per.mean()),
                           "gain": float(per.mean() - rc), "prize": ro - rc,
                           "abstain_rate": float((~ag).mean()),
                           "share": (float(per.mean()) - rc) / (ro - rc)
                           if (ro - rc) > 5e-3 else float("nan")}
        out[gn] = rows
    return out


# ---------------------------------------------------------------------------
# the reduction over cells
# ---------------------------------------------------------------------------

def run_venue(path, anchor, lmax=4, seed=0, r_grid=None):
    Z = np.load(path)
    pre = f"{anchor}__"
    if pre + "t0" not in Z:
        return None
    n = len(Z[pre + "t0"])
    fit_m, read_m = split_rows(n, seed=seed)
    res = {"n_rows": n, "anchor": anchor,
           "etype_hist": np.bincount(np.asarray(Z[pre + "etype"]).astype(int),
                                     minlength=3).tolist(),
           "cells": {}}
    for l in [x for x in LEVELS if x <= lmax]:
        for a in A_LIST:
            rl = list(r_grid or R_GRID)
            ok = np.asarray(Z[pre + f"ok_l{l}_a{a}"]).astype(bool)
            o = np.asarray(Z[pre + f"oe_l{l}_a{a}"]).astype(np.float64)
            p_fit = float(o[fit_m & ok].mean())
            rl.append(p_fit)                  # the bite point: the prize is maximal at r = p
            c = cell(Z, anchor, l, a, sorted(set(rl)), fit_m, read_m)
            if c is not None:
                c["r_bite"] = p_fit
                res["cells"][f"l{l}_a{a}"] = c
    return res


def bite(c):
    return c["r"][f"{c['r_bite']:.6f}"]


def _sh(c, nm, cn, a, l):
    cc = c["cells"].get(f"l{l}_a{a}") if "cells" in c else c
    if not cc:
        return None
    return bite(cc)["gates"].get(nm, {}).get(cn)


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------

def sec_bite(res, cond="marg", lmax=4, a=0, gates=None):
    ls = [l for l in LEVELS if l <= lmax]
    keys = [f"l{l}_a{a}" for l in ls]
    top = [["--", "r_abs (bite)"], ["--", "acc read"], ["--", "floor return"],
           ["--", "prize"], ["--", "n read"]]
    for k in keys:
        c = res["cells"].get(k)
        if not c:
            for b in top:
                b.append("--")
            continue
        e = bite(c)
        for b, v in zip(top, [fmt(c["r_bite"]), fmt(c["acc_read"]),
                              fmt(e["floor"][cond]["ret"]), fmt(e["prize"][cond]),
                              str(c["n_read"])]):
            b.append(v)
    rows = list(top)
    for nm in (gates or GATE_NAMES):
        r1, r2, any_ = [nm, "share of prize"], [nm, "abstain rate"], False
        for l in ls:
            gg = _sh(res, nm, cond, a, l)
            r1.append(f'{gg["share"]:+.3f}+-{gg["share_se"]:.3f}'
                      if gg and np.isfinite(gg["share"]) else "--")
            r2.append(fmt(gg["abstain_rate"]) if gg else "--")
            any_ = any_ or bool(gg)
        if any_:
            rows += [r1, r2]
    return tbl(rows, ["gate", "quantity"] + [f"l={l}" for l in ls])


def sec_conds(res, lmax=4, a=0, gates=None):
    ls = [l for l in LEVELS if l <= lmax]
    rows = []
    for cn in CONDS:
        rows.append(["(floor)", cn, "prize"] +
                    [fmt(bite(res["cells"][f"l{l}_a{a}"])["prize"][cn])
                     if f"l{l}_a{a}" in res["cells"] else "--" for l in ls])
    for nm in (gates or GATE_NAMES):
        for cn in CONDS:
            r, any_ = [nm, cn, "share"], False
            for l in ls:
                gg = _sh(res, nm, cn, a, l)
                r.append(f'{gg["share"]:+.3f}' if gg and np.isfinite(gg["share"]) else "--")
                any_ = any_ or bool(gg)
            if any_:
                rows.append(r)
    return tbl(rows, ["gate", "conditioning", "quantity"] + [f"l={l}" for l in ls])


def sec_screen(res, a=0, lmax=4):
    rows = []
    for nm in GATE_NAMES:
        r1, r2, any_ = [nm, "AUC(wrong)"], [nm, "binned R2"], False
        for l in [x for x in LEVELS if x <= lmax]:
            s = res["cells"].get(f"l{l}_a{a}", {}).get("screen", {}).get(nm)
            r1.append(fmt(s["auc_wrong"]) if s else "--")
            r2.append(fmt(s["bin_r2"]) if s else "--")
            any_ = any_ or bool(s)
        if any_:
            rows += [r1, r2]
    return tbl(rows, ["gate", "quantity"] + [f"l={l}" for l in LEVELS if l <= lmax])


def sec_screen_rho(res, a=0, lmax=4):
    """basalis A5's question in this venue: which screen predicts a gate's VALUE. Spearman,
    across gates, of the binned out-of-sample R2 and of |AUC - 0.5| against the realised share
    of the prize, marginal and inside position strata. `oracle` is excluded (it is the
    ceiling, not a candidate)."""
    from scipy.stats import spearmanr
    rows = []
    for l in [x for x in LEVELS if x <= lmax]:
        c = res["cells"].get(f"l{l}_a{a}")
        if not c:
            continue
        e = bite(c)
        g = [n for n in MAIN if n in e["gates"] and n in c["screen"] and n != "oracle"]
        if len(g) < 5:
            continue
        r2 = np.array([c["screen"][n]["bin_r2"] for n in g])
        au = np.array([abs(c["screen"][n]["auc_wrong"] - 0.5) for n in g])
        out = [f"l={l}", str(len(g))]
        for cn in ("marg", "pos"):
            sh = np.array([e["gates"][n][cn]["share"] for n in g])
            out += [f"{spearmanr(r2, sh).statistic:+.3f}",
                    f"{spearmanr(au, sh).statistic:+.3f}"]
        rows.append(out)
    return tbl(rows, ["level", "n gates", "rho(binR2, share marg)",
                      "rho(|AUC-.5|, share marg)", "rho(binR2, share pos)",
                      "rho(|AUC-.5|, share pos)"])


def sec_rsweep(res, cond="pos", l=2, a=0, gates=("V_pre", "V", "R", "R_clean", "Rshuf",
                                                 "banked_excess", "nll", "excess", "t0",
                                                 "noise", "probe_cons", "oracle")):
    c = res["cells"].get(f"l{l}_a{a}")
    if not c:
        return "(cell absent)"
    rs = sorted(c["r"], key=float)
    rows = [["ret oracle"] + [fmt(c["r"][r]["ret_oracle"]) for r in rs],
            ["ret floor"] + [fmt(c["r"][r]["floor"][cond]["ret"]) for r in rs],
            ["prize"] + [fmt(c["r"][r]["prize"][cond]) for r in rs]]
    for nm in gates:
        vals = [c["r"][r]["gates"].get(nm, {}).get(cond) for r in rs]
        if all(v is None for v in vals):
            continue
        rows.append([f"share: {nm}"] +
                    [f'{v["share"]:+.3f}' if v and np.isfinite(v["share"]) else "--"
                     for v in vals])
    return tbl(rows, ["quantity"] + [f"r={float(r):.2f}" for r in rs])


def sec_offsets(res, gates, cond="pos", lmax=4):
    rows = []
    for nm in gates:
        for l in [x for x in LEVELS if x <= lmax]:
            r, any_ = [nm, f"l={l}"], False
            for a in A_LIST:
                gg = _sh(res, nm, cond, a, l)
                r.append(f'{gg["share"]:+.3f}' if gg and np.isfinite(gg["share"]) else "--")
                any_ = any_ or bool(gg)
            if any_:
                rows.append(r)
    return tbl(rows, ["gate", "level"] + [f"a={a}" for a in A_LIST])


def sec_subgroups(sub, gates):
    names = []
    for gn in gates:
        for k in sub.get(gn, {}):
            if k not in names:
                names.append(k)
    pick = lambda k, f: next((f(sub[g][k]) for g in gates if k in sub.get(g, {})), None)
    rows = [["--", "n"] + [str(pick(k, lambda d: d["n"])) for k in names],
            ["--", "acc"] + [fmt(pick(k, lambda d: d["acc"])) for k in names],
            ["--", "prize"] + [fmt(pick(k, lambda d: d["prize"])) for k in names]]
    for gn in gates:
        d = sub.get(gn, {})
        if not d:
            continue
        rows.append([gn, "share"] + [fmt(d[k]["share"]) if k in d else "--" for k in names])
        rows.append([gn, "gain"] + [fmt(d[k]["gain"], 4) if k in d else "--" for k in names])
    return tbl(rows, ["gate", "quantity"] + names)


def sec_climb(R, tag, anchor, cond, gates, a=0, l=2):
    steps = sorted({k[1] for k in R if k[0] == tag and k[2] == anchor})
    cells = [R[(tag, st, anchor)]["cells"].get(f"l{l}_a{a}") for st in steps]
    rows = [["acc read"] + [fmt(c["acc_read"]) if c else "--" for c in cells],
            ["prize"] + [fmt(bite(c)["prize"][cond]) if c else "--" for c in cells]]
    for nm in gates:
        r, any_ = [nm], False
        for c in cells:
            gg = bite(c)["gates"].get(nm, {}).get(cond) if c else None
            r.append(f'{gg["share"]:+.3f}' if gg and np.isfinite(gg["share"]) else "--")
            any_ = any_ or bool(gg)
        if any_:
            rows.append(r)
    return tbl(rows, ["gate / step"] + [str(s) for s in steps])


def sec_repro(npz_path, cb=CB, anchor="tv", a=0, lmax=4):
    """striatum section 2's matched damage AUCs on the very npz this node reads."""
    ent = rematch(npz_path, cb, lmax=lmax).get(anchor, {})
    names = ["R", "R_clean", "Rshuf", "V", "banked_excess", "probe_cons", "probe_illegal",
             "guard_nll", "guard_t0", "guard_kstar", "guard_j"]
    ls = [l for l in LEVELS if l <= lmax]
    rows = [["n matched"] + [str(ent.get(f"flip/l{l}_a{a}", {}).get("n", "--")) for l in ls]]
    for nm in names:
        vals = [ent.get(f"flip/l{l}_a{a}", {}).get(nm) for l in ls]
        if all(v is None for v in vals):
            continue
        rows.append([nm] + [fmt(v) for v in vals])
    return tbl(rows, ["readout (AUC on realised damage)"] + [f"l={l}" for l in ls])


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

FIGG = ["V_pre", "V", "R", "R_clean", "R_mlp", "Rshuf", "delta_prev", "banked_excess",
        "nll", "excess", "probe_cons", "t0", "noise", "oracle"]


def fig_prize(R, path, tag="swap65k", step=64000, anchor="tv", a=0, lmax=4):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    res = R[(tag, step, anchor)]
    ls = [l for l in LEVELS if l <= lmax]
    fig, ax = plt.subplots(1, 3, figsize=(18, 4.8))
    for k, cn in enumerate(("marg", "pos")):
        w = 0.82 / len(FIGG)
        for i, nm in enumerate(FIGG):
            ys, es = [], []
            for l in ls:
                gg = _sh(res, nm, cn, a, l)
                ys.append(gg["share"] if gg else np.nan)
                es.append(gg["share_se"] if gg else np.nan)
            ax[k].bar(np.arange(len(ls)) + i * w - 0.41, ys, w, yerr=es,
                      error_kw={"lw": 0.5}, label=nm)
        ax[k].axhline(0, color="k", lw=0.8)
        ax[k].set_xticks(np.arange(len(ls)))
        ax[k].set_xticklabels([f"l={l}" for l in ls])
        ax[k].set_ylabel("share of the oracle prize")
        ax[k].set_title(("marginal" if cn == "marg" else "inside anchor-position strata")
                        + f"  ({tag} {anchor} {step}, a={a})")
    ax[0].legend(fontsize=6, ncol=2)

    for nm in FIGG:
        xs, ys = [], []
        for l in ls:
            s = res["cells"].get(f"l{l}_a{a}", {}).get("screen", {}).get(nm)
            if s:
                xs.append(s["auc_wrong"])
                ys.append(s["bin_r2"])
        if xs:
            ax[2].scatter(xs, ys, s=30, label=nm)
    ax[2].axhline(0, color="k", lw=0.8)
    ax[2].axvline(0.5, color="k", lw=0.8, ls=":")
    ax[2].set_xlabel("AUC on 'the actor will be wrong'")
    ax[2].set_ylabel("binned out-of-sample R2 on the stake")
    ax[2].set_title("the screen: detection vs gating")
    ax[2].legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_where(R, sub_fd, path, sub_k=None, tag="swap65k", step=64000, anchor="tv",
              cond="pos", lmax=4):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    res = R[(tag, step, anchor)]
    show = ["V_pre", "V", "R", "delta_prev", "nll", "Rshuf", "oracle"]
    fig, ax = plt.subplots(1, 3, figsize=(18, 4.8))
    for ci, nm in enumerate(show):
        for l in [x for x in LEVELS if x <= lmax]:
            ys = []
            for a in A_LIST:
                gg = _sh(res, nm, cond, a, l)
                ys.append(gg["share"] if gg else np.nan)
            ax[0].plot(A_LIST, ys, marker="o", ms=3, color=f"C{ci}",
                       alpha=0.28 + 0.18 * l, label=nm if l == 2 else None)
    ax[0].axhline(0, color="k", lw=0.8)
    ax[0].set_xlabel("offset a since the anchor")
    ax[0].set_ylabel("share of the prize")
    ax[0].set_title(f"{tag} {anchor} {step}: by offset (levels 1-4, darker = deeper)")
    ax[0].legend(fontsize=7)

    if sub_fd:
        gts = [g for g in show if g in sub_fd]
        names = [k for k in ("swap", "rare", "quiet") if k in sub_fd.get(gts[0], {})]
        w = 0.82 / max(len(gts), 1)
        for i, gn in enumerate(gts):
            ax[1].bar(np.arange(len(names)) + i * w - 0.41,
                      [sub_fd[gn][k]["gain"] for k in names], w, label=gn)
        ax[1].axhline(0, color="k", lw=0.8)
        ax[1].set_xticks(np.arange(len(names)))
        ax[1].set_xticklabels(names)
        ax[1].set_ylabel("return above the per-stratum floor")
        ax[1].set_title("a1 fd, l=2 a=0: violation vs quiet")
        ax[1].legend(fontsize=7)
    sk = sub_k if sub_k else sub_fd
    if sk:
        gts = [g for g in show if g in sk]
        ks = [k for k in sk[gts[0]] if k.startswith("k*")]
        for gn in gts:
            ax[2].plot(range(len(ks)), [sk[gn][k]["share"] for k in ks], marker="o",
                       ms=4, label=gn)
        ax[2].axhline(0, color="k", lw=0.8)
        ax[2].set_xticks(range(len(ks)))
        ax[2].set_xticklabels(ks, fontsize=7)
        ax[2].set_ylabel("share of the prize")
        ax[2].set_title(f"{'swap65k tv' if sub_k else 'a1 fd'}, l=2 a=0: by k*")
        ax[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_cond(R, path, tag="swap65k", step=64000, anchor="tv", l=2, lax=(0, 4)):
    """striatum's addendum in decision currency: every gate's share under the four
    conditionings, at two offsets."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    res = R[(tag, step, anchor)]
    show = ["V_pre", "V", "R", "R_mlp", "delta_prev", "o_prev", "banked_excess", "nll",
            "excess", "probe_cons", "t0", "noise"]
    fig, ax = plt.subplots(1, len(lax), figsize=(9 * len(lax), 4.8))
    ax = np.atleast_1d(ax)
    w = 0.82 / len(CONDS)
    for k, a in enumerate(lax):
        for i, cn in enumerate(CONDS):
            ys, es = [], []
            for nm in show:
                gg = _sh(res, nm, cn, a, l)
                ys.append(gg["share"] if gg else np.nan)
                es.append(gg["share_se"] if gg else np.nan)
            ax[k].bar(np.arange(len(show)) + i * w - 0.41, ys, w, yerr=es,
                      error_kw={"lw": 0.5}, label=cn)
        ax[k].axhline(0, color="k", lw=0.8)
        ax[k].set_xticks(np.arange(len(show)))
        ax[k].set_xticklabels(show, rotation=45, ha="right", fontsize=7)
        ax[k].set_ylabel("share of the oracle prize")
        ax[k].set_title(f"{tag} {anchor} {step}, l={l}, a={a}")
        ax[k].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_climb(R, path, tag="swap65k", anchor="tv", cond="pos", a=0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    steps = sorted({k[1] for k in R if k[0] == tag and k[2] == anchor})
    fig, ax = plt.subplots(1, 3, figsize=(18, 4.6))
    for pi, l in enumerate((1, 2, 3)):
        for nm in ("V_pre", "V", "R", "R_clean", "R_mlp", "Rshuf", "nll", "banked_excess",
                   "noise"):
            ys = []
            for st in steps:
                gg = _sh(R[(tag, st, anchor)], nm, cond, a, l)
                ys.append(gg["share"] if gg else np.nan)
            ax[pi].plot(steps, ys, marker="o", ms=4, label=nm)
        ax[pi].axhline(0, color="k", lw=0.8)
        ax[pi].set_xlabel("checkpoint")
        ax[pi].set_ylabel("share of the prize")
        ax[pi].set_title(f"{tag} {anchor}, l={l}, a={a}, inside position strata")
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------

HEADER = """# abstain — tables

Facts only. Generated by `analyze.py` from the banked `striatum/` cells. Inside a venue every
arm is fitted on the same half of the banked test rows and read on the other half, and no
gate's fit sees a read row's outcome.

**Reproduction.**

```bash
cd experiments            # MODAL_PROFILE=chromatic
modal volume get rhm-scaling-data \\
    /v16_s2_L6_m4_distinct/logit_reading/traj_a1_s42 <dir>/traj_a1_s42
python -m rhm.logit_reading.orbitofrontal.abstain.analyze <dir>/traj_a1_s42 \\
    --tags {tags} --steps {steps} --seed {seed} \\
    --out rhm/logit_reading/orbitofrontal/abstain/results \\
    --figs rhm/logit_reading/orbitofrontal/abstain/figs
```

**The choice.**  At a query (anchor + `a`) the actor either ANSWERS, returning the realised
outcome `oe` in {{0,1}} (1 = the frozen actor named the level-`l` constituent's feature
correctly), or ABSTAINS, returning a fixed `r_abs`.  A **gate** is a scalar the decider reads;
its **policy** is coeruleus's quantile step function — {nbins} equal-quantile bins of the gate
on the fit half ({nbinsc} inside each conditioning stratum), ANSWER in a bin iff that bin's
realised accuracy on the FIT rows beats `r_abs`, then read out on the held-out half.  The
**floor** is the best CONSTANT action, also chosen on fit rows, per conditioning stratum.  The
**oracle** ceiling is the realised outcome as the gate, `p + (1-p)*r_abs`.
**Share of the prize** = `(ret_gate - ret_floor) / (ret_oracle - ret_floor)` on read rows,
`+-` its analytic standard error (the sd of the per-row return difference over sqrt n).

**The bite point.**  `prize = ret_oracle - ret_floor` is `(1-p)*r_abs` below `r_abs = p` and
`p*(1-r_abs)` above it, so it is maximal at exactly `r_abs = p` and equals `p(1-p)`.  Every
per-level table is read at that cell's bite point (`r_abs` = the actor's accuracy on the fit
half, ties to answering); the sweep table carries the whole grid.

**Three conditionings, because the marginal one is confounded.**  `marg` gives the decider
nothing but the gate.  `pos` fits the policy and the floor inside {npos} quantile buckets of the
anchor position `t0` — the bare position is free bookkeeping and is itself a strong gate, which
is striatum's matcher gotcha (the strata must contain the anchor-position bucket) in decision
currency.  `pos_nll` adds {nnll} quantile buckets of the model's own surprisal at the anchor and `pos_V`
adds {nv} quantile buckets of the critic's level `V` at the event: the two halves of striatum's
addendum ("conditioned on the value level the surprise readout carries nothing; conditioned on
the surprise readout the value level is untouched") in decision currency.  Under every
conditioning the floor is the per-stratum constant, so a share is always what the gate adds to
what the decider already had.  A gate read inside strata of itself is a trivial null and is
left in the tables as such (`V` under `pos_V`, `nll` under `pos_nll`).

**The gates.**  `V_pre` = `V[l, a+1](s_(t0-1))`, the critic's pre-event expectation of this very
outcome (norm's norm).  `V` = `V[l, a](s_t0)`, its level at the event.  `R` = `V - V_pre`, the
revision.  `R_clean` is the clean-only critic's revision, `R_mlp` the MLP critic's, `R_embed`
the `post_embed` trunk's, `Rshuf` the shuffled-outcome critic's — a random direction of the
state through the same machinery, which is the honest floor for "a state readout gates", not
zero.  `delta_prev` is the outcome surprise `oe - V_pre` at the PREVIOUS query offset (the only way a
post-outcome signal can act here) and `o_prev` is that previous outcome raw, without the norm
subtracted — the control that says whether `delta_prev`'s value is the surprise or just the
last result.  `banked_excess` is the coeruleus horizon-excess head
at the anchor state; `nll`, `H_pre`, `dH`, `excess` are the model's own surprisal, predictive
entropy, entropy change and single-position excess at the anchor.  `probe_cons` is the
in-container ridge on the structural-consequence label (an oracle LABEL read off the state).
`t0` is the bare anchor position; `noise`, `noise2`, `noise3` are three independent draws of
pure noise, whose spread is this conditioning's binning / selection tax and is the level below
which a share means nothing.  `oracle` is the
realised outcome.  basalis's log Bayes factor needs a second forward pass under two hypotheses
and is not in the banked columns, so it is not read in this venue.

**A note on the stake ridge.**  basalis A6 fitted a closed-form ridge on the state predicting
the stake `log(b/a)`.  Here the stake is `oe - r_abs`, an additive shift of the outcome, so a
closed-form ridge on the state predicting the stake IS the critic `V` up to that shift: the
stake readout and the value readout are the same object in this design, and `V` / `V_pre` are
it.  What is portable is A5's screen — the binned out-of-sample `R2` of the gate's own step
function on the stake — reported beside every AUC below.
"""


def write_tables(R, SUB, PATHS, args, path):
    L = []
    w = L.append
    w(HEADER.format(tags=args.tags, steps=args.steps, seed=args.seed, nbins=NBINS,
                    nbinsc=NBINS_C, npos=N_POS, nnll=N_NLL, nv=N_V))

    key = ("swap65k", 64000, "tv")
    if key in PATHS:
        w("\n## 0. Reproduction gate\n")
        w("`striatum/analyze.py`'s `rematch` on the same npz this node reads: `swap65k` `t_v` "
          "64k `a = 0`, realised damage (right on the original window, wrong on the edited "
          "one), matched on `(k*, j, position)` with sign-balanced surprisal pairs. Against "
          "`striatum/README.md` section 2 — n 2100 / 2776 / 2544 / 1756, `R` 0.601 / 0.606 / "
          "0.569 / 0.541, `V` 0.694 / 0.691 / 0.656 / 0.609. Every guard printed.\n")
        w(sec_repro(PATHS[key], lmax=args.lmax))
        w("")

    for key in sorted(R):
        tag, st, an = key
        res = R[key]
        w(f"\n## {tag}, anchor `{an}`, step {st}\n")
        w(f"{res['n_rows']} banked test rows (swap / rare / none = "
          f"{res['etype_hist'][0]} / {res['etype_hist'][1]} / {res['etype_hist'][2]}); "
          f"fit / read halves {res['n_rows'] // 2} / {res['n_rows'] - res['n_rows'] // 2}; "
          f"strata used at `l2_a0`: {res['cells']['l2_a0']['n_strata']}.\n")
        w("### The decision at the bite point, `a = 0`, INSIDE anchor-position strata\n")
        w(sec_bite(res, cond="pos", lmax=args.lmax, a=0, gates=MAIN))
        w("")
        w("### The same, marginal (no conditioning) — the confounded reading\n")
        w(sec_bite(res, cond="marg", lmax=args.lmax, a=0, gates=MAIN))
        w("")
        w("### All three conditionings side by side, `a = 0`\n")
        w(sec_conds(res, lmax=args.lmax, a=0, gates=MAIN))
        w("")
        w("### The screen: AUC on 'the actor will be wrong' beside the binned out-of-sample "
          "`R2` on the stake, `a = 0`\n")
        w(sec_screen(res, a=0, lmax=args.lmax))
        w("")
        w("### Which screen predicts a gate's value (basalis A5 in this venue)\n")
        w(sec_screen_rho(res, a=0, lmax=args.lmax))
        w("")
        w("### The `r_abs` sweep, `l = 2`, `a = 0`, inside position strata\n")
        w(sec_rsweep(res, cond="pos", l=2, a=0))
        w("")
        w("### By offset since the anchor, inside position strata, each cell at its bite "
          "point\n")
        w(sec_offsets(res, MAIN, cond="pos", lmax=args.lmax))
        w("")
        w("### The decision at the bite point, `a = 4`, inside position strata\n")
        w(sec_bite(res, cond="pos", lmax=args.lmax, a=4, gates=MAIN))
        w("")

    for tag, an in (("swap65k", "tv"), ("a1", "fd")):
        if len({k[1] for k in R if k[0] == tag and k[2] == an}) > 1:
            w(f"\n## The climb — {tag} `{an}`, `l = 2`, `a = 0`, inside position strata\n")
            w(sec_climb(R, tag, an, "pos", MAIN))
            w("")

    for key, sub in SUB.items():
        tag, st, an = key
        w(f"\n## Where the value lives — {tag} `{an}` step {st}, `l = 2`, `a = 0`\n")
        w("One policy per gate, fitted on the whole fit half inside position strata, then "
          "decomposed over subgroups of the read rows. `prize` is the subgroup's own "
          "oracle-minus-floor; `gain` is the return above the floor in outcome units.\n")
        w(sec_subgroups(sub, MAIN))
        w("")

    open(path, "w").write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--tags", default="a1,swap65k")
    ap.add_argument("--steps", default="0,8000,24000,64000")
    ap.add_argument("--out", default=None)
    ap.add_argument("--figs", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lmax", type=int, default=4)
    args = ap.parse_args()

    tags = args.tags.split(",")
    steps = [int(s) for s in args.steps.split(",")]
    R, SUB, PATHS = {}, {}, {}
    for tag in tags:
        for f in sorted(glob.glob(os.path.join(args.dir, f"step*_striatum_{tag}.npz"))):
            base = os.path.basename(f)
            if "_smoke" in base or "_hexcess" in base:
                continue
            st = int(base.split("_")[0][4:])
            if st not in steps:
                continue
            for an in ("fd", "tv"):
                r = run_venue(f, an, lmax=args.lmax, seed=args.seed)
                if r is not None:
                    R[(tag, st, an)] = r
                    PATHS[(tag, st, an)] = f
                    print(f"  {tag} {st} {an}: {r['n_rows']} rows, {len(r['cells'])} cells, "
                          f"strata {r['cells']['l2_a0']['n_strata']}", flush=True)
    for key in (("a1", 64000, "fd"), ("swap65k", 64000, "tv")):
        if key not in R:
            continue
        Z = np.load(PATHS[key])
        fm, rm = split_rows(len(Z[f"{key[2]}__t0"]), seed=args.seed)
        SUB[key] = subgroup_returns(Z, key[2], 2, 0, R[key]["cells"]["l2_a0"]["r_bite"],
                                    fm, rm, MAIN)

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        write_tables(R, SUB, PATHS, args, os.path.join(args.out, "tables.md"))
        print("wrote", os.path.join(args.out, "tables.md"))
    if args.figs:
        os.makedirs(args.figs, exist_ok=True)
        fig_prize(R, os.path.join(args.figs, "abstain_prize.png"))
        fig_where(R, SUB.get(("a1", 64000, "fd")),
                  os.path.join(args.figs, "abstain_where.png"),
                  sub_k=SUB.get(("swap65k", 64000, "tv")))
        fig_cond(R, os.path.join(args.figs, "abstain_conditioning.png"))
        if len({k[1] for k in R if k[0] == "swap65k" and k[2] == "tv"}) > 1:
            fig_climb(R, os.path.join(args.figs, "abstain_climb.png"))
        print("wrote figures ->", args.figs)


if __name__ == "__main__":
    main()
