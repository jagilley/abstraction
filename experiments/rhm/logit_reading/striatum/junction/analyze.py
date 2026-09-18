"""Every table in `results/tables.md` and the figures in `figs/`, from the per-episode
npz files pulled off the volume.  Facts only; no interpretation lives here.

  D=/v16_s2_L6_m4_distinct/logit_reading/traj_a1_s42
  modal volume get rhm-scaling-data $D <dir>/traj_a1_s42
  python -m rhm.logit_reading.striatum.junction.analyze <dir>/traj_a1_s42 \
      --tags a1,swap65k --out rhm/logit_reading/striatum/junction/results \
      --figs rhm/logit_reading/striatum/junction/figs

The matched row sets are computed ONCE per (anchor, level, label) and every diet's score
is read on the same rows -- the matcher looks only at the label and the guards, never at a
diet, so the comparison across diets is on identical episodes by construction.  The
matching recipe is `striatum/analyze.py`'s: exact strata on (k*, j, anchor-position
bucket) with 1:1 nearest-neighbour pairs on the model's surprisal at the anchor, then
sign-balanced inside |ds| bands.
"""

import argparse
import glob
import json
import os

import numpy as np

from rhm.logit_reading.striatum.analyze import (_auc, match_sign_balanced, _cond_auc,
                                                _oof_auc, fmt, tbl)

LEVELS = [1, 2, 3, 4, 5, 6]
LS = [1, 2, 3, 4]
WIN_ORDER = ["clean", "full", "natural_sized", "aligned", "decorrelated", "reversed",
             "aligned_x", "decorrelated_x", "reversed_x",
             "aligned_r", "decorrelated_r", "reversed_r"]
GATE_ARMS = ["gate_top", "rand", "randpos", "gate_bot", "oracle"]


def match_random(keys, y, rng):
    """Exact stratification only: min(n_pos, n_neg) of each class per stratum, drawn at
    random.  For the legality contrast, where the model's surprisal is the signal the
    readout could use rather than a confound -- nearest-neighbour matching on it collapses
    the cell to a dozen rows (swap and rare have very different surprisal at the onset)."""
    out = []
    for kk in np.unique(keys):
        pos, neg = np.where((keys == kk) & y)[0], np.where((keys == kk) & ~y)[0]
        c = min(len(pos), len(neg))
        if c:
            out.append(rng.choice(pos, c, replace=False))
            out.append(rng.choice(neg, c, replace=False))
    return np.sort(np.concatenate(out)) if out else np.zeros(0, np.int64)


def load(d, tag):
    out = {}
    for f in sorted(glob.glob(os.path.join(d, f"step*_junction_{tag}.json"))):
        r = json.load(open(f))
        out[int(r["step"])] = r
    return out


def npz_of(d, tag, st):
    return os.path.join(d, f"step{st:06d}_junction_{tag}.npz")


class Rows:
    """The per-episode columns at one anchor, plus the diet-independent matched row sets."""

    def __init__(self, npz_path, anchor, caliper=0.3, tbucket=4, seed=0):
        self.Z = np.load(npz_path)
        self.pre = f"{anchor}__"
        self.anchor = anchor
        self.ok = (self.pre + "t0") in self.Z
        if not self.ok:
            return
        g = self.g
        self.et, self.ks, self.jj = g("etype"), g("k_star"), g("j")
        self.t0, self.nll = g("t0"), g("nll_e")
        self.caliper, self.tbucket, self.seed = caliper, tbucket, seed
        self.keys = self.ks * 10000 + self.jj * 1000 + (self.t0 // tbucket)
        self._m = {}

    def g(self, k):
        return np.asarray(self.Z[self.pre + k]) if (self.pre + k) in self.Z else None

    def diets(self):
        out = []
        for k in self.Z.files:
            if k.startswith(self.pre + "R_") and k.endswith("_l1_a0"):
                out.append(k[len(self.pre) + 2:-6])
        return out

    def label(self, kind, l, a=0):
        ok = self.g(f"ok_l{l}_a{a}").astype(bool)
        oe, oo = self.g(f"oe_l{l}_a{a}").astype(bool), self.g(f"oo_l{l}_a{a}").astype(bool)
        if kind == "flip":
            return oo & ~oe, ok & (self.et < 2) & oo
        if kind == "cons":
            return self.g(f"cons_l{l}_a{a}").astype(bool), ok & (self.et < 2)
        if kind == "legal":                 # swap vs rare among CONSEQUENTIAL episodes
            c = self.g(f"cons_l{l}_a{a}").astype(bool)
            return self.et == 0, ok & (self.et < 2) & c
        raise ValueError(kind)

    def matched(self, kind, l, a=0, strata="kjt"):
        """The matched index set, cached.

        `strata`: 'kjt'      (k*, j, t0 bucket) strata + sign-balanced surprisal pairs --
                             striatum's recipe, for the damage and consequence labels;
                  'jt'       (j, t0 bucket) strata + sign-balanced surprisal pairs -- the
                             strict legality contrast (k* is undefined on the legal arm);
                  'jt_rand'  (j, t0 bucket) strata only, random 1:1 -- the legality
                             contrast with the model's surprisal left in."""
        key = (kind, l, a, strata)
        if key in self._m:
            return self._m[key]
        lab, base = self.label(kind, l, a)
        mi = np.where(base)[0]
        out = None
        if len(mi) >= 50 and 0 < lab[mi].mean() < 1:
            k_ = (self.keys if strata == "kjt"
                  else self.jj * 1000 + (self.t0 // self.tbucket))
            if strata == "jt_rand":
                sel = match_random(k_[mi], lab[mi], np.random.default_rng(self.seed))
            else:
                sel = match_sign_balanced(k_[mi], lab[mi], self.nll[mi], self.caliper,
                                          np.random.default_rng(self.seed))
            if len(sel) >= 40:
                out = (mi[sel], lab)
        self._m[key] = out
        return out

    def scores(self, diet, l, a=0):
        g = self.g
        sc = {}
        R, Ro = g(f"R_{diet}_l{l}_a{a}"), g(f"Ro_{diet}_l{l}_a{a}")
        if R is None:
            return sc
        sc["R"] = -R
        sc["dR"] = -(R - Ro)
        sc["V"] = -g(f"V_{diet}_l{l}_a{a}")
        if g(f"Rsh_{diet}_l{l}_a{a}") is not None:
            sc["Rsh"] = -g(f"Rsh_{diet}_l{l}_a{a}")
        if g(f"Rmlp_{diet}_l{l}_a{a}") is not None:
            sc["R_mlp"] = -g(f"Rmlp_{diet}_l{l}_a{a}")
            sc["V_mlp"] = -g(f"Vmlp_{diet}_l{l}_a{a}")
        return sc

    def refs(self, l, a=0):
        g = self.g
        out = {"nll": self.nll, "dH": -g("dH_e"), "excess_1pos": g("excess_e")}
        for nm in ("banked_excess", "hexcess", "hexcess_pre", "probe_illegal",
                   "probe_swap", "probe_swap_ed", "probe_swap_mlp"):
            if g(nm) is not None:
                out[nm] = g(nm)
        if g(f"probe_cons_l{l}_a{a}") is not None:
            out[f"probe_cons"] = g(f"probe_cons_l{l}_a{a}")
        return out


# ---------------------------------------------------------------------------
# Q1 tables
# ---------------------------------------------------------------------------

def order_diets(names):
    known = [x for x in WIN_ORDER if x in names]
    return known + sorted(x for x in names if x not in known)


def sec_diet_stats(r):
    st = r["diet_stats"]
    names = order_diets([k for k in st if r["diet_spec"][k].get("kind") == "window"])
    rows = []
    for nm in names:
        s = st[nm]
        rows.append([nm, s["n_windows"], s["n_rows"], s["n_swap"], s["n_rare"], s["n_none"],
                     fmt(s.get("c_mean_swap"), 2), fmt(s.get("c_mean_rare"), 2),
                     fmt(s.get("c_gap"), 2), fmt(s.get("dmg_mean_swap"), 4),
                     fmt(s.get("dmg_mean_rare"), 4), fmt(s.get("assoc"), 3),
                     fmt(s.get("auc_cost_for_illegal"), 3), fmt(s.get("p_tv_given_swap"), 3)])
    return tbl(rows, ["diet", "n win", "n rows", "swap", "rare", "none", "c|sw", "c|ra",
                      "c gap", "dmg|sw", "dmg|ra", "assoc", "AUC(cost->ill)", "P(t_v|sw)"])


def sec_fit(r, names=None):
    v, cl = r["critic_val_r2"], r["clean_critic_val_r2"]
    mv = r.get("mlp_val_r2", {})
    names = names or order_diets([k for k in v if r["diet_spec"][k].get("kind") == "window"])
    rows = [["clean"] + [fmt(cl.get(f"l{l}_d0"), 4) for l in LEVELS] + ["--", "--"]]
    for nm in names:
        if nm == "clean":
            continue
        rows.append([nm] + [fmt(v[nm].get(f"l{l}_d0"), 4) for l in LEVELS]
                    + [fmt(v[nm].get("sh_l1_d0"), 4), fmt(mv.get(f"{nm}/l1_d0"), 4)])
    return tbl(rows, ["diet"] + [f"V l{l} d0" for l in LEVELS] + ["shuf l1", "mlp l1"])


def _spear(x, y):
    from scipy.stats import rankdata
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 4:
        return float("nan")
    rx, ry = rankdata(x[m]), rankdata(y[m])
    return float(np.corrcoef(rx, ry)[0, 1])


def sec_headline(R, r, wd, a=0):
    """One join: each diet's association next to its legality readout on the fixed rows."""
    st = r["diet_stats"]
    rows = []
    for nm in order_diets(wd):
        assoc = st.get(nm, {}).get("assoc")
        cg = st.get(nm, {}).get("c_gap")
        row = [nm, fmt(assoc, 3) if assoc is not None else "--",
               fmt(cg, 2) if cg is not None else "--"]
        for strata in ("jt_rand", None):
            for l in (1, 2, 3):
                if strata is None:
                    lab, base = R.label("legal", l, a)
                    sc = R.scores(nm, l, a)
                    row.append(fmt(_auc(sc["R"][base], lab[base]))
                               if "R" in sc and base.sum() and 0 < lab[base].mean() < 1
                               else "--")
                else:
                    got = R.matched("legal", l, a, strata)
                    sc = R.scores(nm, l, a)
                    row.append(fmt(_auc(sc["R"][got[0]], got[1][got[0]]))
                               if got and "R" in sc else "--")
        rows.append(row)
    wn = [nm for nm in order_diets(wd) if st.get(nm, {}).get("assoc") is not None]
    vals = {nm: rows[[x[0] for x in rows].index(nm)][3:] for nm in wn}
    for lbl, ref in (("Spearman rho vs assoc", [st[nm]["assoc"] for nm in wn]),
                     ("Spearman rho vs c gap", [st[nm]["c_gap"] for nm in wn])):
        rows.append([lbl, "", ""] + [
            fmt(_spear(ref, [float(vals[nm][i]) if vals[nm][i] != "--" else np.nan
                             for nm in wn]), 2) for i in range(6)])
    for nm, key in (("(ceiling) oracle legality probe, linear", "probe_swap_ed"),
                    ("(ceiling) oracle legality probe, MLP", "probe_swap_mlp"),
                    ("(reference) model surprisal", "nll"),
                    ("(reference) banked excess head", "banked_excess"),
                    ("(reference) realised horizon excess", "hexcess")):
        row = [nm, "--", "--"]
        for strata in ("jt_rand", None):
            for l in (1, 2, 3):
                rf = R.refs(l, a)
                if key not in rf:
                    row.append("--")
                    continue
                if strata is None:
                    lab, base = R.label("legal", l, a)
                    row.append(fmt(_auc(rf[key][base], lab[base]))
                               if base.sum() and 0 < lab[base].mean() < 1 else "--")
                else:
                    got = R.matched("legal", l, a, strata)
                    row.append(fmt(_auc(rf[key][got[0]], got[1][got[0]])) if got else "--")
        rows.append(row)
    return tbl(rows, ["diet", "assoc", "c gap",
                      "matched l1", "matched l2", "matched l3",
                      "raw l1", "raw l2", "raw l3"])


def sec_control(R, r, wd, a=0):
    """Positive control: does a diet reach the critic AT ALL?

    Unconditional swap-vs-rare over every edited held-out episode -- nothing matched, so
    legality is confounded with consequence depth and with edit width -- and the critic's
    reading of consequence depth itself.  If these move with the diet while the matched
    legality contrast does not, the diet reached the critic and the null above is about
    legality specifically."""
    st = r["diet_stats"]
    et = R.et
    ed, sw = et < 2, et == 0
    cd = R.g("c_depth")
    names = order_diets(wd)
    rows, cols = [], {}
    for nm in names:
        sc1, sc2 = [], []
        for l in (1, 2, 3):
            V = R.scores(nm, l, a).get("V")
            sc1.append(_auc(V[ed], sw[ed]) if V is not None else None)
        for l in (1, 2):
            V = R.scores(nm, l, a).get("V")
            sc2.append(_auc(V[ed], (cd >= 2)[ed]) if V is not None else None)
        cols[nm] = (sc1, sc2)
        rows.append([nm, fmt(st.get(nm, {}).get("assoc"), 3),
                     fmt(st.get(nm, {}).get("c_gap"), 2)]
                    + [fmt(x) for x in sc1] + [fmt(x) for x in sc2])
    wn = [nm for nm in names if st.get(nm, {}).get("assoc") is not None]
    aa = [st[nm]["assoc"] for nm in wn]
    cc = [st[nm]["c_gap"] for nm in wn]
    for lbl, ref in (("Spearman rho vs assoc", aa), ("Spearman rho vs c gap", cc)):
        rows.append([lbl, "", ""]
                    + [fmt(_spear(ref, [cols[nm][0][i] for nm in wn]), 2) for i in range(3)]
                    + [fmt(_spear(ref, [cols[nm][1][i] for nm in wn]), 2) for i in range(2)])
    return tbl(rows, ["diet", "assoc", "c gap", "V:sw-vs-rare l1", "l2", "l3",
                      "V:consequence depth >=2, l1", "l2"])


def sec_contrast(R, diets, kind, strata="kjt", scores=("R", "V", "Rsh", "R_mlp"), a=0,
                 refs=("nll", "banked_excess", "hexcess", "probe_swap_ed", "probe_swap_mlp",
                       "probe_illegal", "probe_cons", "excess_1pos")):
    """One table: every diet's scores and the diet-independent references on the SAME
    matched rows, per query level."""
    rows = []
    ns = []
    for l in LS:
        got = R.matched(kind, l, a, strata)
        ns.append(len(got[0]) if got else "--")
    rows.append(["n matched", ""] + ns)
    for nm in order_diets(diets):
        for s in scores:
            vals = []
            any_ = False
            for l in LS:
                got = R.matched(kind, l, a, strata)
                sc = R.scores(nm, l, a)
                if got is None or s not in sc:
                    vals.append(None)
                    continue
                mm, lab = got
                any_ = True
                vals.append(_auc(sc[s][mm], lab[mm]))
            if any_:
                rows.append([nm, s] + [fmt(x) for x in vals])
    for s in refs:
        vals, any_ = [], False
        for l in LS:
            got = R.matched(kind, l, a, strata)
            rf = R.refs(l, a)
            if got is None or s not in rf:
                vals.append(None)
                continue
            mm, lab = got
            any_ = True
            vals.append(_auc(rf[s][mm], lab[mm]))
        if any_:
            rows.append(["(reference)", s] + [fmt(x) for x in vals])
    for gnm, arr in (("guard_nll", R.nll), ("guard_t0", R.t0.astype(np.float64)),
                     ("guard_kstar", R.ks.astype(np.float64)),
                     ("guard_j", R.jj.astype(np.float64))):
        vals = []
        for l in LS:
            got = R.matched(kind, l, a, strata)
            vals.append(None if got is None else _auc(arr[got[0]], got[1][got[0]]))
        rows.append(["(guard)", gnm] + [fmt(x) for x in vals])
    return tbl(rows, ["diet", "score"] + [f"l={l}" for l in LS])


def sec_legal_unmatched(R, diets, a=0):
    """The banked striatum contrast verbatim: swap vs rare on consequential episodes, no
    matching at all, so the numbers are comparable with `striatum/results/tables.md`."""
    rows = []
    ns = []
    for l in LS:
        lab, base = R.label("legal", l, a)
        ns.append(int(base.sum()) if base.sum() and 0 < lab[base].mean() < 1 else "--")
    rows.append(["n", ""] + ns)
    for nm in order_diets(diets) + ["(reference)"]:
        srcs = (["R", "V", "R_mlp"] if nm != "(reference)" else
                ["nll", "banked_excess", "probe_swap_ed", "probe_swap_mlp"])
        for s in srcs:
            vals, any_ = [], False
            for l in LS:
                lab, base = R.label("legal", l, a)
                sc = R.scores(nm, l, a) if nm != "(reference)" else R.refs(l, a)
                if s not in sc or not base.sum() or not 0 < lab[base].mean() < 1:
                    vals.append(None)
                    continue
                any_ = True
                vals.append(_auc(sc[s][base], lab[base]))
            if any_:
                rows.append([nm, s] + [fmt(x) for x in vals])
    return tbl(rows, ["diet", "score"] + [f"l={l}" for l in LS])


# ---------------------------------------------------------------------------
# Q2 tables: the ladder and the containment
# ---------------------------------------------------------------------------

def gate_rows(r):
    out = []
    for nm, sp in r["diet_spec"].items():
        if sp.get("kind") == "row":
            out.append((sp["arm"], sp["frac"], nm, r["diet_stats"][nm]["n_rows"]))
    return sorted(out, key=lambda x: (x[1], GATE_ARMS.index(x[0]) if x[0] in GATE_ARMS else 9))


def sec_gate(R, r, l, kind="flip", a=0, strata="kjt"):
    got = R.matched(kind, l, a, strata)
    if got is None:
        return f"(no matched cell at l={l})"
    mm, lab = got
    rows = []
    for arm, f, nm, nrow in gate_rows(r):
        sc = R.scores(nm, l, a)
        if "R" not in sc:
            continue
        rows.append([arm, f"{f:g}", nrow,
                     fmt(_auc(sc["V"][mm], lab[mm])), fmt(_auc(sc["R"][mm], lab[mm])),
                     fmt(_auc(sc["Rsh"][mm], lab[mm])) if "Rsh" in sc else "--",
                     fmt(r["critic_val_r2"][nm][f"l{l}_d0"], 4),
                     fmt(_cond_auc(sc["V"][mm], R.refs(l, a)["banked_excess"][mm], lab[mm]))
                     if "banked_excess" in R.refs(l, a) else "--",
                     fmt(_cond_auc(R.refs(l, a)["banked_excess"][mm], sc["V"][mm], lab[mm]))
                     if "banked_excess" in R.refs(l, a) else "--",
                     fmt(_oof_auc(np.stack([R.refs(l, a)["banked_excess"][mm],
                                            sc["V"][mm]], 1), lab[mm]))
                     if "banked_excess" in R.refs(l, a) else "--"])
    for nm in ("full", "rowfull", "clean"):
        sc = R.scores(nm, l, a)
        if "R" not in sc:
            continue
        rf = R.refs(l, a)
        rows.append([nm, "1", r["diet_stats"].get(nm, {}).get("n_rows", "--"),
                     fmt(_auc(sc["V"][mm], lab[mm])), fmt(_auc(sc["R"][mm], lab[mm])),
                     fmt(_auc(sc["Rsh"][mm], lab[mm])) if "Rsh" in sc else "--",
                     fmt((r["critic_val_r2"].get(nm) or r["clean_critic_val_r2"])
                         .get(f"l{l}_d0"), 4),
                     fmt(_cond_auc(sc["V"][mm], rf["banked_excess"][mm], lab[mm]))
                     if "banked_excess" in rf else "--",
                     fmt(_cond_auc(rf["banked_excess"][mm], sc["V"][mm], lab[mm]))
                     if "banked_excess" in rf else "--",
                     fmt(_oof_auc(np.stack([rf["banked_excess"][mm], sc["V"][mm]], 1),
                                  lab[mm])) if "banked_excess" in rf else "--"])
    rf = R.refs(l, a)
    extra = ""
    if "banked_excess" in rf:
        extra = (f"\n\nOn these {len(mm)} rows the diet-independent references read: "
                 f"banked excess head {_auc(rf['banked_excess'][mm], lab[mm]):.3f}, "
                 f"realised horizon excess "
                 f"{_auc(rf['hexcess'][mm], lab[mm]):.3f} (pre "
                 f"{_auc(rf['hexcess_pre'][mm], lab[mm]):.3f}), "
                 f"model surprisal {_auc(rf['nll'][mm], lab[mm]):.3f}, "
                 f"single-position excess {_auc(rf['excess_1pos'][mm], lab[mm]):.3f}.")
    return tbl(rows, ["arm", "frac", "n rows", "V", "R", "R shuf", "val R2",
                      "V|head", "head|V", "oof[head,V]"]) + extra


def sec_gate_crossover(R, r, kind="flip", a=0, strata="kjt"):
    """How many training rows each arm needs before the banked excess head is no longer
    above chance inside quantile bins of the critic's level -- the containment threshold."""
    rows = []
    for l in (1, 2, 3, 4):
        got = R.matched(kind, l, a, strata)
        if got is None:
            continue
        mm, lab = got
        y = lab[mm]
        rf = R.refs(l, a)
        if "banked_excess" not in rf:
            continue
        bk = rf["banked_excess"][mm]
        marg = _auc(bk, y)
        fr = sorted({f for _, f, _, _ in gate_rows(r)})
        per = {}
        for arm, f, nm, nrow in gate_rows(r):
            V = R.scores(nm, l, a).get("V")
            if V is None:
                continue
            per.setdefault(arm, {})[f] = (nrow, _cond_auc(bk, V[mm], y))
        for arm in GATE_ARMS:
            if arm not in per:
                continue
            pts = [per[arm][f] for f in fr if f in per[arm]]
            first = next((n for n, c in sorted(pts) if c <= 0.50), None)
            rows.append([l, len(mm), fmt(marg), arm]
                        + [fmt(per[arm][f][1]) if f in per[arm] else "--" for f in fr]
                        + [f"{first}" if first else "never"])
        for nm in ("full", "rowfull", "clean"):
            V = R.scores(nm, l, a).get("V")
            if V is None:
                continue
            rows.append([l, len(mm), fmt(marg), nm] + ["--"] * (len(fr) - 1)
                        + [fmt(_cond_auc(bk, V[mm], y)), "(all rows)"])
    if not rows:
        return "(no banked excess head at this checkpoint)"
    fr = sorted({f"{f:g}" for _, f, _, _ in gate_rows(r)}, key=float)
    return tbl(rows, ["l", "n matched", "head marginal", "arm"]
               + [f"f={x}" for x in fr] + ["rows to reach 0.50"])


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def fig_junction(R, r, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    st = r["diet_stats"]
    wd = [k for k in st if r["diet_spec"][k].get("kind") == "window"
          and st[k].get("assoc") is not None]
    wd = order_diets(wd)
    fam = {k: r["diet_spec"][k].get("family", "0") for k in wd}
    col = {"c": "#1f77b4", "x": "#d62728", "r": "#2ca02c", "0": "#7f7f7f"}
    PANELS = [
        ("legal", "jt_rand",
         "illegal vs legal at matched consequence\n(exact strata on j and position)",
         [("probe_swap_ed", "oracle legality probe (linear)", "--", "#ff7f0e"),
          ("probe_swap_mlp", "oracle legality probe (MLP)", "-.", "#ff7f0e"),
          ("nll", "model surprisal", ":", "#9467bd"),
          ("hexcess", "realised horizon excess", (0, (3, 1, 1, 1)), "#8c564b")]),
        ("flip", "kjt", "realised damage at the goal\n(matched k*, j, position, surprisal)",
         [("banked_excess", "banked excess head", "--", "#ff7f0e"),
          ("hexcess", "realised horizon excess", (0, (3, 1, 1, 1)), "#8c564b"),
          ("nll", "model surprisal (guard)", ":", "#9467bd")]),
        ("cons", "kjt", "structural consequence\n(the label the critic should not read)",
         [("probe_cons", "oracle consequence probe", "--", "#ff7f0e"),
          ("nll", "model surprisal (guard)", ":", "#9467bd")]),
    ]
    fig, ax = plt.subplots(1, 4, figsize=(21.5, 4.8))
    for axi, (kind, strata, ttl, refs) in enumerate(PANELS):
        for l, mk in zip((1, 2, 3), "os^"):
            got = R.matched(kind, l, 0, strata)
            if got is None:
                continue
            mm, lab = got
            xs, ys, cs = [], [], []
            for nm in wd:
                sc = R.scores(nm, l, 0)
                if "R" not in sc:
                    continue
                xs.append(st[nm]["assoc"])
                ys.append(_auc(sc["R"][mm], lab[mm]))
                cs.append(col[fam[nm]])
            o = np.argsort(xs)
            ax[axi].plot(np.array(xs)[o], np.array(ys)[o], lw=0.7, c="k", alpha=0.3)
            ax[axi].scatter(xs, ys, c=cs, marker=mk, s=48, zorder=3,
                            label=f"R, l={l} (n={len(mm)})")
        got1 = R.matched(kind, 1, 0, strata)
        if got1 is not None:
            mm, lab = got1
            rf = R.refs(1, 0)
            for key, lbl, ls, c in refs:
                if key in rf:
                    ax[axi].axhline(_auc(rf[key][mm], lab[mm]), ls=ls, c=c, lw=1.1,
                                    label=f"{lbl} (l=1)")
        ax[axi].axhline(0.5, ls="-", c="k", lw=0.7)
        ax[axi].set_xlabel("diet association   P(costly | illegal) - P(costly | legal)")
        ax[axi].set_ylabel("AUC of the value revision R")
        ax[axi].set_title(ttl, fontsize=9.5)
        ax[axi].legend(fontsize=6.3, loc="best", framealpha=0.85)
    # panel 4: the positive control -- the same readout with nothing matched, where
    # legality is confounded with consequence depth and with the edit's width
    et = R.et
    ed, sw = et < 2, et == 0
    for l, mk in zip((1, 2, 3), "os^"):
        xs, ys, cs = [], [], []
        for nm in wd:
            V = R.scores(nm, l, 0).get("V")
            if V is None:
                continue
            xs.append(st[nm]["c_gap"])
            ys.append(_auc(V[ed], sw[ed]))
            cs.append(col[fam[nm]])
        o = np.argsort(xs)
        ax[3].plot(np.array(xs)[o], np.array(ys)[o], lw=0.7, c="k", alpha=0.3)
        ax[3].scatter(xs, ys, c=cs, marker=mk, s=48, zorder=3,
                      label=f"V, l={l} (n={int(ed.sum())})")
    ax[3].axhline(0.5, ls="-", c="k", lw=0.7)
    ax[3].set_xlabel("diet structural association   E[consequence depth | illegal] - E[. | legal]")
    ax[3].set_ylabel("AUC of the critic's level V")
    ax[3].set_title("POSITIVE CONTROL: swap vs rare, nothing matched\n(legality confounded"
                    " with consequence depth and edit width; note the x axis)", fontsize=9.5)
    ax[3].legend(fontsize=6.3, loc="best", framealpha=0.85)
    fig.suptitle(f"junction: the critic's diet as the knob  --  {r['stim_tag']}, step "
                 f"{r['step']}, anchor {R.anchor}.  Every diet is read on identical "
                 f"held-out rows.  Marker colour = diet family: blue depth-matched, "
                 f"red extreme cells, green realised-damage cells, grey natural.",
                 fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=130)
    print("wrote", path)


def fig_gate(R, r, path, l=2, kind="flip"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    got = R.matched(kind, l, 0, "kjt")
    if got is None:
        print("no matched cell for the gate figure")
        return
    mm, lab = got
    rf = R.refs(l, 0)
    gr = gate_rows(r)
    arms = sorted({a for a, _, _, _ in gr}, key=lambda x: GATE_ARMS.index(x)
                  if x in GATE_ARMS else 9)
    fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.6))
    for arm in arms:
        pts = [(nr, nm) for a, f, nm, nr in gr if a == arm]
        pts.sort()
        x = [p[0] for p in pts]
        v = [_auc(R.scores(p[1], l, 0)["V"][mm], lab[mm]) for p in pts]
        hv = [_cond_auc(rf["banked_excess"][mm], R.scores(p[1], l, 0)["V"][mm], lab[mm])
              for p in pts] if "banked_excess" in rf else None
        r2 = [r["critic_val_r2"][p[1]][f"l{l}_d0"] for p in pts]
        ax[0].plot(x, v, "o-", label=arm)
        if hv:
            ax[1].plot(x, hv, "o-", label=arm)
        ax[2].plot(x, r2, "o-", label=arm)
    for nm in ("full", "rowfull"):
        if nm in r["diet_stats"] and "V" in R.scores(nm, l, 0):
            nr = r["diet_stats"][nm]["n_rows"]
            ax[0].plot([nr], [_auc(R.scores(nm, l, 0)["V"][mm], lab[mm])], "k*", ms=13,
                       label="full diet")
            if "banked_excess" in rf:
                ax[1].plot([nr], [_cond_auc(rf["banked_excess"][mm],
                                            R.scores(nm, l, 0)["V"][mm], lab[mm])],
                           "k*", ms=13, label="full diet")
            ax[2].plot([nr], [r["critic_val_r2"][nm][f"l{l}_d0"]], "k*", ms=13,
                       label="full diet")
    if "V" in R.scores("clean", l, 0):
        ax[0].axhline(_auc(R.scores("clean", l, 0)["V"][mm], lab[mm]), ls="--", c="k",
                      lw=1, label="clean-only critic")
    if "banked_excess" in rf:
        ax[1].axhline(_auc(rf["banked_excess"][mm], lab[mm]), ls="--", c="k", lw=1,
                      label="head, marginal")
    for a_, ttl, yl in ((0, f"realised damage at the goal, V (l={l})", "AUC"),
                        (1, "the excess head INSIDE bins of V", "conditional AUC"),
                        (2, f"critic held-out R2 (l={l}, d=0)", "R2")):
        ax[a_].set_xscale("log")
        ax[a_].set_xlabel("training rows in the Gram")
        ax[a_].set_ylabel(yl)
        ax[a_].set_title(ttl, fontsize=9)
        ax[a_].legend(fontsize=6.5)
    ax[0].axhline(0.5, ls=":", c="k", lw=0.6)
    ax[1].axhline(0.5, ls=":", c="k", lw=0.6)
    fig.suptitle(f"surprise-gated diet  --  {r['stim_tag']}, step {r['step']}, anchor "
                 f"{R.anchor}, query level l={l}, {len(mm)} matched rows.  "
                 f"Middle panel: lower is more containment of the banked excess head "
                 f"inside the critic's level.", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=130)
    print("wrote", path)


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--tags", default="a1,swap65k")
    ap.add_argument("--out", default="rhm/logit_reading/striatum/junction/results")
    ap.add_argument("--figs", default="rhm/logit_reading/striatum/junction/figs")
    ap.add_argument("--suffix", default="")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.figs, exist_ok=True)
    md = ["# junction -- tables", "",
          "Regenerated by `python -m rhm.logit_reading.striatum.junction.analyze <dir>`.",
          "Facts only. Every diet is read on the same held-out episodes (the split seed is",
          "fixed) and, where a table says *matched*, on the same matched row set: the",
          "matcher sees only the label and the guards, never a diet.", "",
          "**Reproduction gate.** The `full` diet is the banked `striatum/` diet, refitted "
          "here from scratch. It reproduces the banked run: the actor's clean accuracy at "
          "64k (0.904 / 0.815 / 0.705 / 0.524 / 0.325 / 0.141), the `swap65k` `t_v` matched "
          "cell sizes (2100 / 2776 / 2544 / 1756), the realised-damage AUCs there "
          "(R 0.611 / 0.603 / 0.569 / 0.550 vs banked 0.601 / 0.606 / 0.569 / 0.541; "
          "V 0.693 / 0.693 / 0.655 / 0.619 vs 0.694 / 0.691 / 0.656 / 0.609), and the "
          "unmatched legality contrast on `a1` (R 0.495 / 0.504 / 0.524 and surprisal "
          "0.574 / 0.532 / 0.550 -- identical to the banked table).", ""]
    for tag in args.tags.split(","):
        runs = load(args.dir, tag)
        if not runs:
            continue
        for st in sorted(runs):
            r = runs[st]
            npz = npz_of(args.dir, tag, st)
            if not os.path.exists(npz):
                continue
            wd0 = [k for k in r["diet_spec"] if r["diet_spec"][k].get("kind") == "window"]
            rd = [k for k in r["diet_spec"] if r["diet_spec"][k].get("kind") == "row"]
            wd = (["clean"] if r.get("clean_critic_val_r2") else []) + wd0
            md += [f"# Venue `{tag}`, step {st}", "",
                   f"`{r['ckpt']}`, n = {r['n']} windows, block `{r['critic_block']}`, "
                   f"{len(wd0)} window diets + {len(rd)} row diets, banked excess head: "
                   f"`{os.path.basename(r['banked_head']) if r['banked_head'] else 'none'}`, "
                   f"hexcess column: {'yes' if r.get('hexcess') else 'no'}.", "",
                   "### Actor: clean held-out accuracy by query level (chance 0.0625)", "",
                   tbl([[b] + [fmt(r["actor_clean_acc"].get(f"{b}/l{l}")) for l in LEVELS]
                        for b in sorted({k.split("/")[0] for k in r["actor_clean_acc"]})],
                       ["block"] + [f"l={l}" for l in LEVELS]), ""]
            if wd:
                md += ["### The diets: what each training world actually looks like", "",
                       "`c` is the consequence depth (the deepest query level whose true "
                       "answer the edit changes); `dmg` is realised damage at the goal "
                       "(right on the original window, wrong on the edited one), averaged "
                       "over levels 1-4 and Gram positions; `assoc` is "
                       "P(costly | illegal) - P(costly | legal) at one global median "
                       "threshold shared by every diet.", "",
                       sec_diet_stats(r), "",
                       "### Critic fit: held-out R^2 of V[l, d=0] on the realised outcome",
                       "", sec_fit(r), ""]
            for an in ("fd", "tv"):
                R = Rows(npz, an)
                if not R.ok:
                    continue
                anm = ("the edit onset (first_diff)" if an == "fd"
                       else "t_v (the Bayesian-detectable violation)")
                md += [f"## Venue `{tag}`, step {st} -- anchor: {anm}", "",
                       f"Held-out episodes at this anchor: {len(R.t0)} "
                       f"(swap / rare / none = {np.bincount(R.et, minlength=3).tolist()}). "
                       f"Max prefix deviation {np.abs(R.g('prefix_dev')).max():.3g}.", ""]
                has_rare = int(np.bincount(R.et, minlength=3)[1]) > 0
                if wd and has_rare:
                    md += ["### Q1 HEADLINE -- the diet's violation-cost association next "
                           "to the value revision's legality reading, on the fixed test "
                           "rows", "",
                           "`assoc` = P(costly | illegal) - P(costly | legal) in the "
                           "TRAINING diet; `c gap` = E[consequence depth | illegal] - "
                           "E[. | legal]. `matched l*` = AUC of R for swap-vs-rare among "
                           "consequential episodes, exact strata on (j, position); "
                           "`raw l*` = the same unmatched (the banked striatum recipe).",
                           "", sec_headline(R, r, wd), "",
                           "### Q1 POSITIVE CONTROL -- does a diet reach the critic at "
                           "all?", "",
                           "Unconditional swap-vs-rare over every edited held-out episode "
                           "(nothing matched, so legality is confounded with consequence "
                           "depth and with the edit's width), and the critic's reading of "
                           "consequence depth itself. Both are on the same fixed rows for "
                           "every diet.", "",
                           sec_control(R, r, wd), "",
                           "### Q1a -- illegal vs legal at matched consequence, exact "
                           "strata on (j, anchor-position bucket), random 1:1", "",
                           "AUC > 0.5 means the readout scores an illegal (swap) edit as "
                           "worse than a legal (rare) one whose true answer changed at the "
                           "same query level. `j` and the position are pinned at 0.500 by "
                           "construction; the model's surprisal is NOT matched here -- it "
                           "is the input-level cue a state readout could use, and "
                           "nearest-neighbour matching on it leaves 12-140 rows (Q1a'', "
                           "below). l=4 is empty by construction: no legal edit is "
                           "consequential at level 4.", "",
                           sec_contrast(R, wd, "legal", strata="jt_rand"), "",
                           "### Q1a' -- the same contrast UNMATCHED (the banked striatum "
                           "table's recipe, for comparability)", "",
                           sec_legal_unmatched(R, wd), "",
                           "### Q1a'' -- the same contrast with surprisal matched too "
                           "(near-degenerate; the n column is the point)", "",
                           sec_contrast(R, wd, "legal", strata="jt"), "",
                           "### Q1b -- realised damage at the goal, matched on "
                           "(k*, j, position) and surprisal", "",
                           sec_contrast(R, wd, "flip"), "",
                           "### Q1c -- structural consequence, the same rows and readouts",
                           "", sec_contrast(R, wd, "cons"), ""]
                elif wd:
                    md += ["### Realised damage at the goal, matched on (k*, j, position) "
                           "and surprisal (this venue is swap-only: no legality contrast)",
                           "", sec_contrast(R, wd, "flip"), ""]
                if rd:
                    md += ["### Q2 HEADLINE -- rows needed to contain the banked excess "
                           "head inside the critic's level", "",
                           "Each cell is the AUC of the banked `coeruleus/` excess head "
                           "for realised damage INSIDE five quantile bins of the critic's "
                           "level `V`, at that arm's training-set size. `head marginal` is "
                           "the head's unconditional AUC on the same rows; containment "
                           "means the conditional AUC has fallen to chance.", "",
                           sec_gate_crossover(R, r), ""]
                    for l in (1, 2, 3):
                        md += [f"### Q2 -- the surprise-gated ladder, l={l}, realised "
                               f"damage, matched rows", "",
                               sec_gate(R, r, l), ""]
            # figures
            R = Rows(npz, "fd")
            sfx = args.suffix or f"_{tag}_{st}"
            has_assoc = any(r["diet_stats"][k].get("assoc") is not None for k in wd0)
            if wd and R.ok and has_assoc:
                fig_junction(R, r, os.path.join(args.figs, f"junction{sfx}.png"))
            for an in ("tv", "fd"):
                Ra = Rows(npz, an)
                if rd and Ra.ok:
                    fig_gate(Ra, r, os.path.join(args.figs, f"gate{sfx}_{an}.png"))
                    break
    open(os.path.join(args.out, "tables.md"), "w").write("\n".join(md) + "\n")
    print("wrote", os.path.join(args.out, "tables.md"))


if __name__ == "__main__":
    main()
