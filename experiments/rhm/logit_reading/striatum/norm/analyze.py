"""Every table in `results/tables.md` and the figures in `figs/`, from the per-episode npz
files pulled off the volume.  Facts only; no interpretation lives here.

  D=/v16_s2_L6_m4_distinct/logit_reading/traj_a1_s42
  modal volume get rhm-scaling-data $D <dir>/traj_a1_s42
  python -m rhm.logit_reading.striatum.norm.analyze <dir>/traj_a1_s42 \
      --tags a1,swap65k --out rhm/logit_reading/striatum/norm/results \
      --figs rhm/logit_reading/striatum/norm/figs

Every diet is read on IDENTICAL held-out rows (one split seed, shared across diets and
across checkpoints), and the matched row sets are computed once per (anchor, level, label)
from the label and the guards only -- the matcher never sees a diet.  The matching recipe
is `striatum/analyze.py`'s: exact strata on (k*, j, anchor-position bucket) with 1:1
nearest-neighbour pairs on the model's surprisal at the anchor, sign-balanced inside |ds|
bands.
"""

import argparse
import glob
import json
import os

import numpy as np

from rhm.logit_reading.striatum.analyze import _auc, match_sign_balanced, fmt, tbl
from rhm.logit_reading.striatum.junction.analyze import Rows

LEVELS = [1, 2, 3, 4, 5, 6]
LS = [1, 2, 3, 4]
FAM_ORDER = ["clean", "full", "natural_sized",
             "out_lo", "out_mid", "out_hi", "dmg_lo", "dmg_mid", "dmg_hi"]


def order_diets(names):
    known = [n for n in FAM_ORDER if n in names]
    mix = sorted([n for n in names if n.startswith("mix_q")])
    rest = sorted(n for n in names if n not in known and n not in mix)
    return known + mix + rest


def load(d, tag):
    out = {}
    for f in sorted(glob.glob(os.path.join(d, f"step*_norm_{tag}.json"))):
        if "_smoke" in f:
            continue
        r = json.load(open(f))
        out[int(r["step"])] = r
    return out


def npz_of(d, tag, st):
    return os.path.join(d, f"step{st:06d}_norm_{tag}.npz")


def _ols(x, y):
    """slope, intercept, r2 of y ~ x."""
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3 or x.std() == 0:
        return float("nan"), float("nan"), float("nan")
    b = np.cov(x, y, bias=True)[0, 1] / x.var()
    a = y.mean() - b * x.mean()
    r2 = float(1.0 - ((y - (a + b * x)) ** 2).mean() / max(y.var(), 1e-18))
    return float(b), float(a), r2


def _spear(x, y):
    from scipy.stats import rankdata
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan")
    return float(np.corrcoef(rankdata(x[ok]), rankdata(y[ok]))[0, 1])


def _sem(x):
    x = np.asarray(x, np.float64)
    return float(x.std(ddof=1) / max(np.sqrt(len(x)), 1)) if len(x) > 1 else float("nan")


# ---------------------------------------------------------------------------
# 0. the reproduction gate and the diets
# ---------------------------------------------------------------------------

# The banked `striatum/` actor column for `traj_a1_s42` -- the reproduction gate of the
# 2026-09-17 round.  A DIFFERENT trajectory seed has a different actor, so
# `--banked-actor none` drops the comparison rows and `--banked-actor
# "8000=a,b,c,d,e,f;64000=..."` supplies that trajectory's own banked column instead
# (`parse_banked_actor`).  The default is unchanged, so the banked cells reproduce.
BANKED_ACTOR = {64000: [0.904, 0.815, 0.705, 0.524, 0.325, 0.141],
                8000: [0.893, 0.776, 0.623, 0.399, 0.218, 0.091]}


def parse_banked_actor(spec):
    """`""` -> the banked `traj_a1_s42` column; `"none"` -> no comparison rows;
    `"8000=a,b,...;64000=a,b,..."` -> that trajectory's own column."""
    if not spec:
        return BANKED_ACTOR
    if spec.strip().lower() in ("none", "off", "skip"):
        return {}
    out = {}
    for part in spec.split(";"):
        if not part.strip():
            continue
        k, v = part.split("=")
        out[int(k)] = [float(x) for x in v.split(",")]
    return out


def sec_gate(runs, tag, banked=None):
    banked = BANKED_ACTOR if banked is None else banked
    lbl = "banked `striatum/`" if banked else "no banked"
    out = [f"### Actor (clean held-out accuracy, `post_block7`), against the {lbl} "
           f"column", ""]
    rows = []
    for st in sorted(runs):
        a = runs[st]["actor_clean_acc"]
        rows.append([f"{tag} step {st}"] + [fmt(a.get(f"post_block7/l{l}")) for l in LEVELS])
        if st in banked:
            rows.append([f"banked striatum {st}"] + [fmt(x) for x in banked[st]])
    out.append(tbl(rows, ["run"] + [f"l={l}" for l in LEVELS]))
    out += ["", "### The critic's held-out fit `R2` of `V[l, 0]`, per diet", ""]
    for st in sorted(runs):
        r = runs[st]
        nms = order_diets(list(r["critic_val_r2"]))
        rows = [[f"clean-only"] + [fmt(r["clean_critic_val_r2"][f"l{l}_d0"], 4) for l in LEVELS]
                + ["--", fmt(r.get("clean_critic_mean_outcome"), 4)]]
        for nm in nms:
            v = r["critic_val_r2"][nm]
            rows.append([nm] + [fmt(v[f"l{l}_d0"], 4) for l in LEVELS]
                        + [fmt(v[f"sh_l1_d0"], 4), fmt(v.get("_train_mean_l1_d0"), 4)])
        out += [f"**step {st}, {tag}**", "",
                tbl(rows, ["diet"] + [f"l={l}" for l in LEVELS]
                    + ["shuffled l=1", "E[o] l1_d0 on its own rows"]), ""]
    return "\n".join(out)


DIET_COLS = [("n_windows", 0), ("n_rows", 0), ("n_swap", 0), ("n_rare", 0), ("n_none", 0),
             ("edit_share", 3), ("mean_j", 3), ("mean_c_depth", 3),
             ("p_illegal_among_edits", 3), ("mean_outcome", 4), ("mean_damage", 4),
             ("mean_outcome_orig_win", 4), ("mean_damage_win", 4)]


def sec_diets(runs, tag):
    out = []
    for st in sorted(runs):
        r = runs[st]
        ds = r["diet_stats"]
        nms = order_diets(list(ds))
        rows = []
        for nm in nms:
            s = ds[nm]
            rows.append([nm] + [(str(int(s[k])) if nd == 0 else fmt(s.get(k), nd))
                                for k, nd in DIET_COLS])
        out += [f"**step {st}, {tag}** — every diet's world.  `mean_outcome` / `mean_damage` "
                f"are over the Gram rows the diet actually sees (levels 1–4).", "",
                tbl(rows, ["diet"] + [k for k, _ in DIET_COLS]), ""]
        # the structural identity check, per family
        fam = {}
        for nm in nms:
            f = r["diet_spec"][nm].get("family", "?")
            fam.setdefault(f, []).append(nm)
        chk = []
        for f, ns in fam.items():
            if len(ns) < 2:
                continue
            h = [json.dumps(ds[n].get("cell_hist", {}), sort_keys=True) for n in ns]
            chk.append([f, ", ".join(ns), "identical" if len(set(h)) == 1 else "DIFFERENT"])
        if chk:
            out += ["`(etype, j)` cell histogram across each family "
                    "(pins legality, width and the structural consequence profile):", "",
                    tbl(chk, ["family", "diets", "cell histogram"]), ""]
    return "\n".join(out)


# ---------------------------------------------------------------------------
# 1. Q1a -- the norm
# ---------------------------------------------------------------------------

def _pop(R, kind):
    """Row populations at one anchor."""
    et = R.et
    if kind == "all":
        return np.ones(len(et), bool)
    if kind == "none":
        return et == 2
    if kind == "edit":
        return et < 2
    if kind == "viol":
        return (et == 0) & (R.g("t_v") > 0)
    raise ValueError(kind)


def sec_norm_level(R, r, a=0, pops=("none", "viol")):
    """V(s_{t-1}) before the event, per diet, on the fixed rows."""
    nms = order_diets(R.diets() + ["clean"])
    ds = r["diet_stats"]
    out = []
    for pk in pops:
        m = _pop(R, pk)
        if m.sum() < 20:
            continue
        rows = []
        for nm in nms:
            e_o = ds.get(nm, {}).get("mean_outcome")
            if nm == "clean":
                e_o = r.get("clean_critic_mean_outcome")
            cells = [nm, fmt(e_o, 4)]
            for l in LS:
                vp = R.g(f"Vpre_{nm}_l{l}_a{a}")
                cells.append("--" if vp is None else fmt(float(vp[m].mean()), 4))
            for l in LS:
                vp = R.g(f"Vpre_{nm}_l{l}_a{a}")
                cells.append("--" if vp is None else fmt(float(vp[m].std()), 4))
            rows.append(cells)
        out += [f"**`V(s_(t-1))` on {pk} rows (n = {int(m.sum())}), anchor {R.anchor}, "
                f"a = {a}**", "",
                tbl(rows, ["diet", "diet E[o]"] + [f"mean l={l}" for l in LS]
                    + [f"sd l={l}" for l in LS]), ""]
    return "\n".join(out)


def sec_norm_shape(R, r, a=0, ref="full", l=1):
    """Is the shift an intercept or state-dependent?  Regress each diet's V on `ref`'s."""
    nms = [n for n in order_diets(R.diets() + ["clean"]) if n != ref]
    m = _pop(R, "all")
    rows = []
    x = R.g(f"Vpre_{ref}_l{l}_a{a}")
    if x is None:
        return ""
    x = x[m]
    for nm in nms:
        y = R.g(f"Vpre_{nm}_l{l}_a{a}")
        if y is None:
            continue
        b, c, r2 = _ols(x, y[m])
        rows.append([nm, fmt(b, 4), fmt(c, 4), fmt(r2, 4),
                     fmt(float(y[m].mean() - x.mean()), 4),
                     fmt(float(y[m].std() / max(x.std(), 1e-12)), 4)])
    return "\n".join([
        f"**Regression of each diet's `V(s_(t-1))` on `{ref}`'s, level {l}, anchor "
        f"{R.anchor}, a = {a}, all {int(m.sum())} fixed rows.**  Slope 1 with a non-zero "
        f"intercept = a pure level shift; slope != 1 or `R2` < 1 = state-dependent.", "",
        tbl(rows, ["diet", "slope", "intercept", "R2", "mean shift vs " + ref,
                   "sd ratio"]), ""])


# ---------------------------------------------------------------------------
# 2. Q1b -- the response at the event
# ---------------------------------------------------------------------------

def sec_response(R, r, a=0, kind="flip", strata="kjt"):
    """R at the event per diet: raw, scaled by the diet's own R spread on unedited windows,
    and its realised-damage AUC on the matched rows."""
    nms = order_diets(R.diets() + ["clean"])
    ds = r["diet_stats"]
    out = []
    viol = _pop(R, "viol")
    none = _pop(R, "none")
    for l in LS:
        got = R.matched(kind, l, a, strata)
        mm, lab = (got if got is not None else (None, None))
        rows = []
        for nm in nms:
            Rv = R.g(f"R_{nm}_l{l}_a{a}")
            Ro = R.g(f"Ro_{nm}_l{l}_a{a}")
            if Rv is None:
                continue
            e_o = ds.get(nm, {}).get("mean_outcome")
            if nm == "clean":
                e_o = r.get("clean_critic_mean_outcome")
            sc = float(Ro[none].std()) if none.sum() > 20 else float(Ro.std())
            dmn = float((Rv - Ro)[viol].mean())
            vp = R.g(f"Vpre_{nm}_l{l}_a{a}")
            cells = [nm, fmt(e_o, 4), fmt(float(vp[viol].mean()), 4),
                     fmt(float(Rv[viol].mean()), 4),
                     fmt(_sem(Rv[viol]), 4), fmt(sc, 4),
                     fmt(float(Rv[viol].mean()) / max(sc, 1e-12), 3),
                     fmt(float(Ro[viol].mean()), 4), fmt(dmn, 4), fmt(_sem((Rv - Ro)[viol]), 4),
                     fmt(dmn / max(sc, 1e-12), 3)]
            if mm is not None:
                cells += [fmt(_auc(-Rv[mm], lab[mm])), fmt(_auc(-R.g(f"V_{nm}_l{l}_a{a}")[mm],
                                                               lab[mm]))]
                sh = R.g(f"Rsh_{nm}_l{l}_a{a}")
                cells.append("--" if sh is None else fmt(_auc(-sh[mm], lab[mm])))
            else:
                cells += ["--", "--", "--"]
            rows.append(cells)
        hdr = ["diet", "diet E[o]", "mean V_pre", "mean R (viol)", "sem", "sd R on unedited",
               "R / sd", "R on the same window unedited (Ro)", "R - Ro", "sem",
               "(R - Ro) / sd", f"AUC(-R, {kind})", f"AUC(-V, {kind})", "shuffled floor"]
        nmm = 0 if mm is None else len(mm)
        src = ("the unedited `none` windows at this anchor" if none.sum() > 20 else
               "the ORIGINAL-stream state of the very same windows and positions")
        out += [f"**level {l}, anchor {R.anchor}, a = {a}.  n violation rows "
                f"{int(viol.sum())}, matched rows {nmm}.**  `Ro` is the same revision "
                f"computed from the unedited stream's states at the same positions; "
                f"`sd R on unedited` is taken over {src}.", "", tbl(rows, hdr), ""]
        if mm is not None:
            g = guards(R, mm, lab)
            out += ["guards on the matched rows: " + g, ""]
    return "\n".join(out)


def guards(R, mm, lab):
    g = {"model surprisal": R.nll, "k*": R.ks.astype(float), "j": R.jj.astype(float),
         "anchor position": R.t0.astype(float)}
    return "  ".join(f"{k} {fmt(_auc(v[mm], lab[mm]))}" for k, v in g.items())


DIET_PROPS = ["mean_outcome", "mean_damage", "mean_outcome_orig_win", "edit_share"]


def sec_ordering(R, r, a=0):
    """The orderings, compactly: Spearman rho of each readout against each property of the
    diet's world, overall and inside each family (where the family's structural composition
    is pinned).  With three diets per family a rho is +-1 or +-0.5; it is a direction, not
    an effect size, and the readouts' own values are in the tables above."""
    ds, sp = r["diet_stats"], r["diet_spec"]
    nms = [n for n in order_diets(R.diets()) if n in ds]
    fam = {n: sp[n].get("family", "anchor") for n in nms}
    fams = [None] + sorted({f for f in fam.values() if f != "anchor"})
    viol, none = _pop(R, "viol"), _pop(R, "none")
    out = []
    for l in LS:
        got = R.matched("flip", l, a, "kjt")
        Rv = {n: R.g(f"R_{n}_l{l}_a{a}") for n in nms}
        Ro = {n: R.g(f"Ro_{n}_l{l}_a{a}") for n in nms}
        if any(v is None for v in Rv.values()):
            continue
        sd = {n: (float(Ro[n][none].std()) if none.sum() > 20 else float(Ro[n].std()))
              for n in nms}
        read = {
            "V_pre (the norm)": np.array([float(R.g(f"Vpre_{n}_l{l}_a{a}")[viol].mean())
                                          for n in nms]),
            "R (raw)": np.array([float(Rv[n][viol].mean()) for n in nms]),
            "R - Ro": np.array([float((Rv[n] - Ro[n])[viol].mean()) for n in nms]),
            "(R - Ro) / sd": np.array([float((Rv[n] - Ro[n])[viol].mean()) / max(sd[n], 1e-12)
                                       for n in nms]),
            "sd of R on unedited": np.array([sd[n] for n in nms]),
        }
        if got is not None:
            mm, lab = got
            read[f"AUC(-R, damage) n={len(mm)}"] = np.array(
                [_auc(-Rv[n][mm], lab[mm]) for n in nms])
            read["AUC(-V, damage)"] = np.array(
                [_auc(-R.g(f"V_{n}_l{l}_a{a}")[mm], lab[mm]) for n in nms])
        rows = []
        for k, y in read.items():
            cells = [k]
            for p in DIET_PROPS:
                x = np.array([ds[n].get(p, np.nan) for n in nms])
                for f in fams:
                    sel = [i for i, n in enumerate(nms) if f is None or fam[n] == f]
                    cells.append(fmt(_spear(x[sel], y[sel]), 2) if len(sel) > 2 else "--")
            rows.append(cells)
        hdr = ["readout"] + [f"{p}/{f or 'all'}" for p in DIET_PROPS for f in fams]
        out += [f"**level {l}, anchor {R.anchor}, a = {a}** — Spearman rho of the readout "
                f"across diets against the diet's world property "
                f"(`/all` = all {len(nms)} diets; `/out`, `/dmg`, `/mix` = inside one "
                f"family, 3 diets, so rho is +-1 or +-0.5).", "", tbl(rows, hdr), ""]
    return "\n".join(out)


def sec_delta(R, r, a_list=(0, 4, 8)):
    """d = outcome - V(s_(t-1)) at the query, per diet, on the fixed rows."""
    nms = order_diets(R.diets() + ["clean"])
    out = []
    for pk in ("viol", "none"):
        m = _pop(R, pk)
        if m.sum() < 20:
            continue
        for l in LS:
            rows = []
            for nm in nms:
                cells = [nm]
                for a in a_list:
                    vp = R.g(f"Vpre_{nm}_l{l}_a{a}")
                    oe = R.g(f"oe_l{l}_a{a}")
                    ok = R.g(f"ok_l{l}_a{a}").astype(bool)
                    if vp is None or oe is None:
                        cells += ["--", "--"]
                        continue
                    sel = m & ok
                    dd = oe[sel].astype(np.float64) - vp[sel]
                    cells += [fmt(float(dd.mean()), 4), fmt(float(oe[sel].mean()), 4)]
                rows.append(cells)
            hdr = ["diet"] + sum([[f"d a={a}", f"outcome a={a}"] for a in a_list], [])
            out += [f"**level {l}, {pk} rows (n = {int(m.sum())}), anchor {R.anchor}**", "",
                    tbl(rows, hdr), ""]
    return "\n".join(out)


# ---------------------------------------------------------------------------
# 3. Q2 -- the same-prefix twins
# ---------------------------------------------------------------------------

class Twins:
    """The same-prefix pairs of one world.  `pre` selects the world: `tw__` is the parent's
    swap/edit world (the violator's own regrown continuation follows), `gl__` is the glitch
    world of `basalis/worlds.py` (the unedited stream's continuation follows, which belongs
    to a legal token there and not to the glitch)."""

    def __init__(self, npz_path, r, caliper=0.3, pre="tw__", key="twin"):
        self.Z = np.load(npz_path)
        self.pre = pre
        self.ok = (pre + "V_viol") in self.Z and bool(r.get(key, {}).get("cols"))
        self.r = r
        if not self.ok:
            return
        tw = r[key]
        self.cols = {(c[0], int(c[1]), int(c[2])): i for i, c in enumerate(tw["cols"])}
        self.offs = list(tw["offs"])
        self.Vv = np.asarray(self.Z[self.pre + "V_viol"], np.float64)
        self.Vt = np.asarray(self.Z[self.pre + "V_twin"], np.float64)
        self.ks = np.asarray(self.Z[self.pre + "k_star"])
        self.jj = np.asarray(self.Z[self.pre + "j"])
        self.tv = np.asarray(self.Z[self.pre + "t_v"])
        self.split = np.asarray(self.Z[self.pre + "split"])
        self.s_v = np.asarray(self.Z[self.pre + "s_v"], np.float64)
        self.s_c = np.asarray(self.Z[self.pre + "s_c"], np.float64)
        self.ds = self.s_v - self.s_c
        self.keep = np.abs(self.ds) <= caliper
        self.caliper = caliper
        self.diets = sorted({c[0] for c in self.cols})

    def V(self, side, diet, l, dd, off):
        A = self.Vv if side == "v" else self.Vt
        return A[:, self.offs.index(off), self.cols[(diet, l, dd)]]

    def vpre(self, diet, l):
        return self.V("v", diet, l, 1, -1)

    def dR(self, diet, l):
        return self.V("v", diet, l, 0, 0) - self.V("t", diet, l, 0, 0)

    def Rv(self, diet, l):
        return self.V("v", diet, l, 0, 0) - self.V("v", diet, l, 1, -1)

    def Rt(self, diet, l):
        return self.V("t", diet, l, 0, 0) - self.V("t", diet, l, 1, -1)

    def token_balanced(self, mask=None, seed=0):
        """`phasic.balance_tokens` on the pairs: a subset in which every token is the
        violator exactly as often as it is the twin.  `dR` is a PAIRED difference, so on
        such a subset any additive function of token identity sums to exactly zero --
        the token-identity guard for the mean `dR`."""
        from rhm.logit_reading.phasic import balance_tokens
        m = self.keep if mask is None else (self.keep & mask)
        idx = np.where(m)[0]
        xv = np.asarray(self.Z[self.pre + "x_v"]).astype(np.int64)
        xc = np.asarray(self.Z[self.pre + "x_c"]).astype(np.int64)
        keep = balance_tokens(xv, xc, idx, np.random.default_rng(seed))
        out = np.zeros(len(m), bool)
        out[keep] = True
        # the exact self-check: the signed token histogram must be identically zero
        v = np.bincount(xv[out], minlength=16) - np.bincount(xc[out], minlength=16)
        return out, int(np.abs(v).sum())


def sec_twin_head(T, r, tag, st):
    tw = r["twin"]
    rows = [["eligible violations", str(tw.get("n_eligible"))],
            ["pairs saved (selection caliper %.2f)" % tw.get("caliper", 0), str(tw.get("n_pairs"))],
            ["pairs at the parent's caliper 0.30", str(int(T.keep.sum())) if T.ok else "--"],
            ["pairs at 0.30, test split only",
             str(int((T.keep & (T.split == 2)).sum())) if T.ok else "--"],
            ["max |V_pre(violator) - V_pre(twin)| (0 by construction)",
             fmt(tw.get("check_Vpre_identical_max_abs"), 8)],
            ["max |log q_(t-1) difference| (0 by construction)",
             fmt(tw.get("check_qprev_identical_max_abs"), 8)],
            ["mean |ds| over the saved pairs", fmt(tw.get("mean_abs_ds"), 4)],
            ["mean |ds| at caliper 0.30",
             fmt(float(np.abs(T.ds[T.keep]).mean()), 4) if T.ok and T.keep.sum() else "--"]]
    out = [f"**{tag}, step {st}** — the same-prefix twin population "
           f"(`phasic.py`'s construction: the violator replaced by the legal token of "
           f"nearest model surprisal after an identical prefix).", "",
           tbl(rows, ["quantity", "value"]), ""]
    cov = tw.get("coverage_by_kstar")
    if cov:
        rows = [[k, str(vv[0]), str(vv[1]), fmt(vv[1] / max(vv[0], 1), 3)]
                for k, vv in sorted(cov.items(), key=lambda kv: int(kv[0]))]
        out += ["coverage by `k*` (eligible / saved):", "",
                tbl(rows, ["k*", "eligible", "saved", "fraction"]), ""]
    if T.ok:
        rows = []
        for k in sorted(set(T.ks[T.keep].tolist())):
            rows.append([k, int((T.keep & (T.ks == k)).sum())])
        out += ["`k*` histogram of the pairs at caliper 0.30:", "",
                tbl([[str(a), str(b)] for a, b in rows], ["k*", "n"]), ""]
    return "\n".join(out)


def sec_twin_slope(T, nbin=5, mask=None, note=""):
    """Does dR scale with V_pre?  Against the confounded R-on-V_pre for comparison."""
    m = T.keep if mask is None else (T.keep & mask)
    out = []
    for diet in order_diets(T.diets):
        rows = []
        for l in LS:
            if (diet, l, 0) not in T.cols:
                continue
            vp, dR, Rv, Rt = (T.vpre(diet, l)[m], T.dR(diet, l)[m],
                              T.Rv(diet, l)[m], T.Rt(diet, l)[m])
            b1, _, _ = _ols(vp, dR)
            b2, _, _ = _ols(vp, Rv)
            b3, _, _ = _ols(vp, Rt)
            rows.append([l, int(m.sum()), fmt(float(vp.mean()), 4), fmt(float(vp.std()), 4),
                         fmt(float(dR.mean()), 5), fmt(_sem(dR), 5),
                         fmt(b1, 4), fmt(_spear(vp, dR), 3),
                         fmt(b2, 4), fmt(_spear(vp, Rv), 3), fmt(b3, 4)])
        if rows:
            out += [f"**critic `{diet}`**{note}", "",
                    tbl([[str(c) for c in rr] for rr in rows],
                        ["l", "n", "mean V_pre", "sd V_pre", "mean dR", "sem",
                         "slope dR~V_pre", "rho", "slope R~V_pre (confounded)", "rho",
                         "slope R_twin~V_pre"]), ""]
    return "\n".join(out)


def sec_twin_kstar(T, diet="full", mask=None, min_n=40):
    """The same slope inside each `k*` stratum, so it is not a `k*` effect, plus the
    residualised slope (V_pre, dR both taken off (k*, j, position, violator surprisal, ds))."""
    m = T.keep if mask is None else (T.keep & mask)
    out = []
    for l in LS:
        if (diet, l, 0) not in T.cols:
            continue
        rows = []
        for k in sorted(set(T.ks[m].tolist())):
            s = m & (T.ks == k)
            if s.sum() < min_n:
                continue
            vp, dR = T.vpre(diet, l)[s], T.dR(diet, l)[s]
            b, _, _ = _ols(vp, dR)
            rows.append([k, int(s.sum()), fmt(float(vp.mean()), 4), fmt(float(dR.mean()), 5),
                         fmt(_sem(dR), 5), fmt(b, 4), fmt(_spear(vp, dR), 3)])
        # residualised on the guards
        vp, dR = T.vpre(diet, l)[m], T.dR(diet, l)[m]
        G = np.stack([T.ks[m], T.jj[m], T.tv[m], T.s_v[m], np.abs(T.ds[m]),
                      np.ones(int(m.sum()))], 1).astype(np.float64)
        try:
            cf = np.linalg.lstsq(G, np.stack([vp, dR], 1), rcond=None)[0]
            rv, rd = (np.stack([vp, dR], 1) - G @ cf).T
            b, _, _ = _ols(rv, rd)
            rows.append(["resid", int(m.sum()), fmt(float(rv.std()), 4), "--", "--",
                         fmt(b, 4), fmt(_spear(rv, rd), 3)])
        except Exception:
            pass
        out += [f"**`{diet}`, level {l}, by `k*` (and residualised on k*, j, position, "
                f"violator surprisal and ds)**", "",
                tbl([[str(c) for c in rr] for rr in rows],
                    ["k*", "n", "mean V_pre", "mean dR", "sem", "slope dR~V_pre", "rho"]), ""]
    return "\n".join(out)


def sec_twin_bins(T, diet="full", nbin=5, mask=None):
    m = T.keep if mask is None else (T.keep & mask)
    out = []
    for l in LS:
        if (diet, l, 0) not in T.cols:
            continue
        vp, dR, Rv = T.vpre(diet, l)[m], T.dR(diet, l)[m], T.Rv(diet, l)[m]
        q = np.quantile(vp, np.linspace(0, 1, nbin + 1)[1:-1])
        b = np.digitize(vp, q)
        rows = []
        for bb in range(nbin):
            s = b == bb
            if s.sum() < 5:
                continue
            rows.append([bb + 1, int(s.sum()), fmt(float(vp[s].mean()), 4),
                         fmt(float(dR[s].mean()), 5), fmt(_sem(dR[s]), 5),
                         fmt(float(Rv[s].mean()), 4), fmt(_sem(Rv[s]), 4)])
        out += [f"**`{diet}`, level {l}, {nbin} bins of `V_pre` (n = {int(m.sum())})**", "",
                tbl([[str(c) for c in rr] for rr in rows],
                    ["bin", "n", "mean V_pre", "mean dR", "sem",
                     "mean R (confounded)", "sem"]), ""]
    return "\n".join(out)


def sec_twin_trace(T, diet="full", mask=None, T_max=63):
    """Matched persistence: V and the one-step revision at each offset after the event."""
    m = T.keep if mask is None else (T.keep & mask)
    out = []
    for l in LS:
        if (diet, l, 0) not in T.cols:
            continue
        rows = []
        for off in T.offs:
            good = m & ((T.tv + off) <= T_max) & ((T.tv + off) >= 0)
            if good.sum() < 20:
                continue
            vv = T.V("v", diet, l, 0, off)[good]
            vt = T.V("t", diet, l, 0, off)[good]
            cells = [off, int(good.sum()), fmt(float(vv.mean()), 4), fmt(float(vt.mean()), 4),
                     fmt(float((vv - vt).mean()), 5), fmt(_sem(vv - vt), 5)]
            if (off - 1) in T.offs:
                rv = vv - T.V("v", diet, l, 1, off - 1)[good]
                rt = vt - T.V("t", diet, l, 1, off - 1)[good]
                cells += [fmt(float(rv.mean()), 5), fmt(float(rt.mean()), 5),
                          fmt(float((rv - rt).mean()), 5), fmt(_sem(rv - rt), 5)]
            else:
                cells += ["--", "--", "--", "--"]
            rows.append(cells)
        out += [f"**`{diet}`, level {l}, offsets from `t_v` (n varies with the window end)**"
                f"  Read offset 0 as the clean event reading: the pair differs only in the "
                f"token at `t_v` there.  At offsets > 0 both streams carry the SAME (edited) "
                f"continuation, which is the swap's legally regrown subtree — coherent with "
                f"the violator's token and not with the twin's — so the later offsets are "
                f"not a pure decay of the event.",
                "", tbl([[str(c) for c in rr] for rr in rows],
                        ["offset", "n", "V violator", "V twin", "dV", "sem",
                         "R violator", "R twin", "dR", "sem"]), ""]
    return "\n".join(out)


# ---------------------------------------------------------------------------
# 3b. Addendum (2026-09-17): the two worlds side by side
# ---------------------------------------------------------------------------

WORLD = {"tw__": "edit (swap)", "gl__": "glitch"}


def common_windows(Ta, Tb):
    """The windows that survive BOTH worlds' caliper, so the two worlds are read on the
    identical window set and therefore on identical (k*, j, position) strata."""
    wa = np.asarray(Ta.Z[Ta.pre + "w"])[Ta.keep]
    wb = np.asarray(Tb.Z[Tb.pre + "w"])[Tb.keep]
    both = np.intersect1d(wa, wb)
    ma = Ta.keep & np.isin(np.asarray(Ta.Z[Ta.pre + "w"]), both)
    mb = Tb.keep & np.isin(np.asarray(Tb.Z[Tb.pre + "w"]), both)
    return ma, mb, len(both)


def sec_world_head(Ts, r, tag, st):
    rows = []
    for pre, T in Ts.items():
        g = r["glitch"] if pre == "gl__" else r["twin"]
        rows.append([WORLD[pre], str(g.get("n_pairs")), str(int(T.keep.sum())),
                     fmt(g.get("check_Vpre_identical_max_abs"), 8),
                     fmt(g.get("check_qprev_identical_max_abs"), 8),
                     fmt(float(np.abs(T.ds[T.keep]).mean()), 4),
                     fmt(float(T.s_v[T.keep].mean()), 3),
                     fmt(float(T.s_c[T.keep].mean()), 3)])
    out = [f"**{tag}, step {st}** — the two worlds' pair populations.  Both are the same "
           f"construction: the flagged token replaced by the LEGAL token of nearest model "
           f"surprisal after a bit-identical prefix.  They differ in whose continuation "
           f"follows: in the edit world the swap's regrown subtree, which belongs to the "
           f"violator; in the glitch world the untouched original stream, which belongs to "
           f"a legal token.", "",
           tbl(rows, ["world", "pairs saved", "pairs at caliper 0.30", "max |dV_pre|",
                      "max |d log q_(t-1)|", "mean |ds| in pair",
                      "mean surprisal, flagged token", "mean surprisal, legal twin"]), ""]
    g = r.get("glitch", {})
    if g:
        rows = [["illegal token available under the clean prefix",
                 fmt(g.get("illegal_available_frac"), 4)],
                ["legal tokens at t_v under the clean prefix (mean of 16)",
                 fmt(g.get("n_legal_at_tv_mean"), 3)],
                ["mean |s(glitch) - s(edit violator)|  (the across-world guard)",
                 fmt(g.get("mean_abs_ds_across_worlds"), 4)],
                ["mean surprisal: glitch / edit violator / the original token",
                 f"{fmt(g.get('mean_surprisal_glitch'), 3)} / "
                 f"{fmt(g.get('mean_surprisal_violator'), 3)} / "
                 f"{fmt(g.get('mean_surprisal_orig_token'), 3)}"],
                ["P(the glitch token is the edit's violator)",
                 fmt(g.get("p_glitch_equals_violator"), 4)]]
        out += ["The glitch's own guards (`glitch_s`: the illegal-under-the-clean-prefix "
                "token nearest the edit's violator in model surprisal):", "",
                tbl(rows, ["quantity", "value"]), ""]
    return "\n".join(out)


def sec_worlds(Ts, masks, diets=("full", "clean"), label=""):
    """dR per world, per critic, per level, token-balanced, on the common window set."""
    out = []
    for diet in diets:
        rows = []
        for pre, T in Ts.items():
            if (diet, 1, 0) not in T.cols:
                continue
            tb, chk = T.token_balanced(mask=masks[pre])
            for l in LS:
                vp, dR = T.vpre(diet, l)[tb], T.dR(diet, l)[tb]
                Rv, Rt = T.Rv(diet, l)[tb], T.Rt(diet, l)[tb]
                rows.append([WORLD[pre], l, int(tb.sum()), chk,
                             fmt(float(vp.mean()), 4), fmt(float(Rv.mean()), 5),
                             fmt(float(Rt.mean()), 5), fmt(float(dR.mean()), 5),
                             fmt(_sem(dR), 5),
                             fmt(float(dR.mean()) / max(_sem(dR), 1e-12), 1),
                             fmt(_ols(vp, dR)[0], 4), fmt(_spear(vp, dR), 3)])
        if rows:
            out += [f"**critic `{diet}`**{label}", "",
                    tbl([[str(c) for c in rr] for rr in rows],
                        ["world", "l", "n", "signed token hist", "mean V_pre",
                         "mean R (flagged)", "mean R (legal twin)", "mean dR", "sem",
                         "dR / sem", "slope dR~V_pre", "rho"]), ""]
    return "\n".join(out)


def sec_worlds_kstar(Ts, masks, diet="full", min_n=40):
    """V_pre, R and dR by the violation's k*, both worlds, identical windows."""
    out = []
    for l in LS:
        rows = []
        for pre, T in Ts.items():
            if (diet, l, 0) not in T.cols:
                continue
            tb, _ = T.token_balanced(mask=masks[pre])
            for nm, arr in (("V_pre", T.vpre(diet, l)), ("R (flagged)", T.Rv(diet, l)),
                            ("dR", T.dR(diet, l))):
                cells = [f"{WORLD[pre]}: {nm}"]
                for k in range(1, 7):
                    sel = tb & (T.ks == k)
                    cells.append("--" if sel.sum() < min_n else
                                 fmt(float(arr[sel].mean()), 4))
                cells.append(fmt(float(arr[tb].mean()), 4))
                rows.append(cells)
            ns = [f"{WORLD[pre]}: n"] + ["--" if (tb & (T.ks == k)).sum() < min_n else
                                         str(int((tb & (T.ks == k)).sum()))
                                         for k in range(1, 7)] + [str(int(tb.sum()))]
            rows.append(ns)
        out += [f"**level {l}, critic `{diet}`, token-balanced, common windows**", "",
                tbl(rows, ["row"] + [f"k*={k}" for k in range(1, 7)] + ["all"]), ""]
    return "\n".join(out)


def sec_worlds_trace(Ts, masks, diet="full", l=2, T_max=63):
    out = []
    for pre, T in Ts.items():
        if (diet, l, 0) not in T.cols:
            continue
        tb, _ = T.token_balanced(mask=masks[pre])
        rows = []
        for off in T.offs:
            good = tb & ((T.tv + off) <= T_max) & ((T.tv + off) >= 0)
            if good.sum() < 20:
                continue
            vv, vt = T.V("v", diet, l, 0, off)[good], T.V("t", diet, l, 0, off)[good]
            rows.append([off, int(good.sum()), fmt(float(vv.mean()), 4),
                         fmt(float(vt.mean()), 4), fmt(float((vv - vt).mean()), 5),
                         fmt(_sem(vv - vt), 5)])
        out += [f"**{WORLD[pre]}, level {l}, critic `{diet}`, token-balanced**", "",
                tbl([[str(c) for c in rr] for rr in rows],
                    ["offset", "n", "V flagged", "V legal twin", "dV", "sem"]), ""]
    return "\n".join(out)


def fig_worlds(Ts, masks, r, path, diet="full", nbin=5):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"tw__": "#d62728", "gl__": "#1f77b4"}
    fig, ax = plt.subplots(1, 4, figsize=(21.5, 4.8))
    for pre, T in Ts.items():
        if (diet, 1, 0) not in T.cols:
            continue
        tb, _ = T.token_balanced(mask=masks[pre])
        ls = "-" if pre == "tw__" else "--"
        ys = [float(T.dR(diet, l)[tb].mean()) for l in LS]
        es = [_sem(T.dR(diet, l)[tb]) for l in LS]
        ax[0].errorbar(LS, ys, yerr=es, marker="o", ls=ls, capsize=3, color=col[pre],
                       label=f"{WORLD[pre]} (n={int(tb.sum())})")
        ax[1].plot(LS, [float(T.vpre(diet, l)[tb].mean()) for l in LS], marker="o", ls=ls,
                   color=col[pre], label=f"{WORLD[pre]}: V_pre")
        ax[1].plot(LS, [float(T.Rv(diet, l)[tb].mean()) for l in LS], marker="s", ls=ls,
                   alpha=0.6, color=col[pre], label=f"{WORLD[pre]}: R at the event")
        for l, mk in ((1, "o"), (2, "s"), (3, "^")):
            if (diet, l, 0) not in T.cols:
                continue
            ks_ = [k for k in range(1, 7) if (tb & (T.ks == k)).sum() >= 40]
            ax[2].plot(ks_, [float(T.dR(diet, l)[tb & (T.ks == k)].mean()) for k in ks_],
                       marker=mk, ls=ls, color=col[pre], alpha=0.5 + 0.15 * l,
                       label=f"{WORLD[pre]} l={l}")
        l = 2
        tr = [o for o in T.offs if o >= -1]
        dv = []
        for o in tr:
            g = tb & ((T.tv + o) <= 63)
            dv.append(float((T.V("v", diet, l, 0, o) - T.V("t", diet, l, 0, o))[g].mean()))
        ax[3].plot(tr, dv, marker="o", ls=ls, color=col[pre], label=f"{WORLD[pre]} l=2")
    for i in (0, 2, 3):
        ax[i].axhline(0, c="k", lw=0.7)
    for i, (t, xl, yl) in enumerate([
            ("dR = R(flagged) - R(legal twin), token-balanced", "query level l", "dR"),
            ("the pre-event norm and the raw response", "query level l", "V_pre / R"),
            ("dR by the violation's k*", "k*", "dR"),
            ("matched persistence: dV at l=2", "offset from the event", "dV")]):
        ax[i].set_title(t, fontsize=9.5)
        ax[i].set_xlabel(xl)
        ax[i].set_ylabel(yl)
        ax[i].legend(fontsize=6.3, framealpha=0.85)
    fig.suptitle(f"norm addendum: does the critic's optimism after the illegal token "
                 f"reverse where re-parsing is the wrong policy?  {r['stim_tag']}, step "
                 f"{r['step']}, critic `{diet}`.  Solid = the edit world (the regrown "
                 f"continuation belongs to the violator); dashed = the glitch world (the "
                 f"original continuation belongs to a legal token).  Same windows, same "
                 f"positions, same (k*, j) strata.", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, dpi=130)
    print("wrote", path)


# ---------------------------------------------------------------------------
# 4. Q3 -- scope
# ---------------------------------------------------------------------------

def sec_scope(Rs, tag, diet="full", a=0):
    """V_pre and R by the violation's k*, checkpoint against checkpoint, identical rows."""
    out = []
    steps = sorted(Rs)
    base = Rs[steps[0]]
    viol = _pop(base, "viol")
    ksv = base.ks
    for l in LS:
        rows = []
        for st in steps:
            R = Rs[st]
            if R.g(f"Vpre_{diet}_l{l}_a{a}") is None:
                continue
            vp, rv = R.g(f"Vpre_{diet}_l{l}_a{a}"), R.g(f"R_{diet}_l{l}_a{a}")
            ro = R.g(f"Ro_{diet}_l{l}_a{a}")
            for nm, arr in (("V_pre", vp), ("R", rv), ("R unedited", ro)):
                cells = [f"{nm}, step {st}"]
                for k in range(1, 7):
                    s = viol & (ksv == k)
                    cells.append("--" if s.sum() < 20 else fmt(float(arr[s].mean()), 4))
                cells.append(fmt(float(arr[viol].mean()), 4))
                rows.append(cells)
        ns = ["n"] + ["--" if (viol & (ksv == k)).sum() < 20 else
                      str(int((viol & (ksv == k)).sum())) for k in range(1, 7)] \
            + [str(int(viol.sum()))]
        out += [f"**{tag}, level {l}, anchor {base.anchor}, critic `{diet}`**", "",
                tbl([ns] + rows, ["row"] + [f"k*={k}" for k in range(1, 7)] + ["all"]), ""]
    return "\n".join(out)


def sec_scope_damage(Rs, tag, a=0):
    """The actor's damage by k* on the same rows, as the cost-side reference."""
    steps = sorted(Rs)
    base = Rs[steps[0]]
    viol = _pop(base, "viol")
    ksv = base.ks
    out = []
    for l in LS:
        rows = []
        for st in steps:
            R = Rs[st]
            oo, oe = R.g(f"oo_l{l}_a{a}").astype(bool), R.g(f"oe_l{l}_a{a}").astype(bool)
            dmg = oo & ~oe
            cells = [f"damage, step {st}"]
            for k in range(1, 7):
                s = viol & (ksv == k)
                cells.append("--" if s.sum() < 20 else fmt(float(dmg[s].mean()), 4))
            cells.append(fmt(float(dmg[viol].mean()), 4))
            rows.append(cells)
        out += [f"**{tag}, level {l} — realised damage on the same rows**", "",
                tbl(rows, ["row"] + [f"k*={k}" for k in range(1, 7)] + ["all"]), ""]
    return "\n".join(out)


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

FAMCOL = {"out": "#1f77b4", "dmg": "#d62728", "mix": "#2ca02c", "anchor": "#7f7f7f"}


def fig_calibration(R, r, path, a=0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ds, sp = r["diet_stats"], r["diet_spec"]
    nms = [n for n in order_diets(R.diets()) if n in ds]
    cs = [FAMCOL.get(sp[n].get("family", "anchor"), "#7f7f7f") for n in nms]
    x = np.array([ds[n]["mean_outcome"] for n in nms])
    viol, none = _pop(R, "viol"), _pop(R, "none")
    fig, ax = plt.subplots(1, 4, figsize=(21.5, 4.8))
    for l, mk in zip(LS, "os^D"):
        y = np.array([float(R.g(f"Vpre_{n}_l{l}_a{a}")[viol].mean()) for n in nms])
        ax[0].scatter(x, y, c=cs, marker=mk, s=46, zorder=3, label=f"l={l}")
        o = np.argsort(x)
        ax[0].plot(x[o], y[o], lw=0.7, c="k", alpha=0.25)
        yr = np.array([float(R.g(f"R_{n}_l{l}_a{a}")[viol].mean()) for n in nms])
        ax[1].scatter(x, yr, c=cs, marker=mk, s=46, zorder=3, label=f"l={l}")
        ax[1].plot(x[o], yr[o], lw=0.7, c="k", alpha=0.25)
        sd = np.array([float(R.g(f"Ro_{n}_l{l}_a{a}")[none].std()) if none.sum() > 20
                       else float(R.g(f"Ro_{n}_l{l}_a{a}").std()) for n in nms])
        ax[2].scatter(x, yr / np.maximum(sd, 1e-12), c=cs, marker=mk, s=46, zorder=3,
                      label=f"l={l}")
        ax[2].plot(x[o], (yr / np.maximum(sd, 1e-12))[o], lw=0.7, c="k", alpha=0.25)
        got = R.matched("flip", l, a, "kjt")
        if got is not None:
            mm, lab = got
            ya = np.array([_auc(-R.g(f"R_{n}_l{l}_a{a}")[mm], lab[mm]) for n in nms])
            ax[3].scatter(x, ya, c=cs, marker=mk, s=46, zorder=3, label=f"l={l} (n={len(mm)})")
            ax[3].plot(x[o], ya[o], lw=0.7, c="k", alpha=0.25)
    ax[3].axhline(0.5, c="k", lw=0.7)
    for i, t in enumerate([
            "the norm: mean V(s_(t-1)) before the event",
            "the response: mean R at the event (raw)",
            "the response, in each diet's own units\n(R / sd of R on unedited windows)",
            "R's realised-damage AUC\n(matched k*, j, position, surprisal)"]):
        ax[i].set_xlabel("the diet's world: E[outcome] over its own Gram rows")
        ax[i].set_title(t, fontsize=9.5)
        ax[i].legend(fontsize=6.6, framealpha=0.85)
    ax[0].set_ylabel("V(s_(t-1))")
    ax[1].set_ylabel("R")
    ax[2].set_ylabel("R / sd(R, unedited)")
    ax[3].set_ylabel("AUC(-R, realised damage)")
    fig.suptitle(f"norm Q1: does the value channel's norm calibrate to the world, and does "
                 f"the response to an identical event scale with it?  {r['stim_tag']}, step "
                 f"{r['step']}, anchor {R.anchor}.  Every diet is read on identical held-out "
                 f"rows; colour = diet family (blue outcome-tercile, red damage-tercile, "
                 f"green quiet-share, grey anchors).", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, dpi=130)
    print("wrote", path)


def fig_twins(T, r, path, diet="full", nbin=5, mask=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    m = T.keep if mask is None else (T.keep & mask)
    mall = T.keep
    fig, ax = plt.subplots(1, 4, figsize=(21.5, 4.8))
    for l, mk in zip(LS, "os^D"):
        if (diet, l, 0) not in T.cols:
            continue
        vp, dR, Rv = T.vpre(diet, l)[m], T.dR(diet, l)[m], T.Rv(diet, l)[m]
        q = np.quantile(vp, np.linspace(0, 1, nbin + 1)[1:-1])
        b = np.digitize(vp, q)
        xs = [float(vp[b == i].mean()) for i in range(nbin) if (b == i).sum() > 4]
        ys = [float(dR[b == i].mean()) for i in range(nbin) if (b == i).sum() > 4]
        es = [_sem(dR[b == i]) for i in range(nbin) if (b == i).sum() > 4]
        y2 = [float(Rv[b == i].mean()) for i in range(nbin) if (b == i).sum() > 4]
        e2 = [_sem(Rv[b == i]) for i in range(nbin) if (b == i).sum() > 4]
        ax[0].errorbar(xs, ys, yerr=es, marker=mk, capsize=2, label=f"l={l}")
        vpa, dRa = T.vpre(diet, l)[mall], T.dR(diet, l)[mall]
        qa = np.quantile(vpa, np.linspace(0, 1, nbin + 1)[1:-1])
        ba = np.digitize(vpa, qa)
        ax[0].plot([float(vpa[ba == i].mean()) for i in range(nbin) if (ba == i).sum() > 4],
                   [float(dRa[ba == i].mean()) for i in range(nbin) if (ba == i).sum() > 4],
                   marker=mk, ls=":", alpha=0.35, color=ax[0].lines[-1].get_color())
        ax[1].errorbar(xs, y2, yerr=e2, marker=mk, capsize=2, label=f"l={l}")
        tr = [o for o in T.offs if o >= -1]
        vv, vt = [], []
        for o in tr:
            g = m & ((T.tv + o) <= 63)
            vv.append(float(T.V("v", diet, l, 0, o)[g].mean()))
            vt.append(float(T.V("t", diet, l, 0, o)[g].mean()))
        ax[2].plot(tr, vv, marker=mk, ls="-", label=f"violator l={l}")
        ax[2].plot(tr, vt, marker=mk, ls="--", alpha=0.6, label=f"legal twin l={l}")
        ax[3].plot(tr, np.array(vv) - np.array(vt), marker=mk, label=f"l={l}")
    ax[0].axhline(0, c="k", lw=0.7)
    ax[3].axhline(0, c="k", lw=0.7)
    for i, (t, xl, yl) in enumerate([
            ("dR = R(violator) - R(legal twin) by the pre-event expectation\n"
             "solid: token-balanced (token identity cancels exactly);  dotted: all pairs",
             "V(s_(t-1)) bin", "dR"),
            ("CONFOUNDED comparator: R(violator) by the same bins\n"
             "(a high V_pre tends to fall whatever arrives)", "V(s_(t-1)) bin", "R"),
            ("matched persistence: V after the event", "offset from t_v", "V[l, 0]"),
            ("matched persistence: dV = V(violator) - V(twin)", "offset from t_v", "dV")]):
        ax[i].set_title(t, fontsize=9.5)
        ax[i].set_xlabel(xl)
        ax[i].set_ylabel(yl)
        ax[i].legend(fontsize=6.6, framealpha=0.85)
    fig.suptitle(f"norm Q2: within one world, does the response to the same violation depend "
                 f"on what was expected just before it?  {r['stim_tag']}, step {r['step']}, "
                 f"critic `{diet}`, {int(m.sum())} token-balanced same-prefix pairs "
                 f"({int(mall.sum())} at caliper {T.caliper}).", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, dpi=130)
    print("wrote", path)


def fig_scope(Rs, path, tag, diet="full", a=0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    steps = sorted(Rs)
    base = Rs[steps[0]]
    viol = _pop(base, "viol")
    ksv = base.ks
    fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.6))
    for st, ls in zip(steps, ["--", "-", "-.", ":"]):
        R = Rs[st]
        for l, mk in zip(LS, "os^D"):
            if R.g(f"Vpre_{diet}_l{l}_a{a}") is None:
                continue
            ks_ = [k for k in range(1, 7) if (viol & (ksv == k)).sum() >= 20]
            vp = [float(R.g(f"Vpre_{diet}_l{l}_a{a}")[viol & (ksv == k)].mean()) for k in ks_]
            rv = [float(R.g(f"R_{diet}_l{l}_a{a}")[viol & (ksv == k)].mean()) for k in ks_]
            oo = R.g(f"oo_l{l}_a{a}").astype(bool)
            oe = R.g(f"oe_l{l}_a{a}").astype(bool)
            dm = [float((oo & ~oe)[viol & (ksv == k)].mean()) for k in ks_]
            ax[0].plot(ks_, vp, marker=mk, ls=ls, label=f"l={l}, step {st}")
            ax[1].plot(ks_, rv, marker=mk, ls=ls, label=f"l={l}, step {st}")
            ax[2].plot(ks_, dm, marker=mk, ls=ls, label=f"l={l}, step {st}")
    for i, t in enumerate(["the norm before the event, V(s_(t-1))",
                           "the response at the event, R",
                           "the cost: realised damage at the query"]):
        ax[i].set_xlabel("k* of the violation (the level the edit breaks)")
        ax[i].set_title(t, fontsize=9.5)
        ax[i].legend(fontsize=6.0, framealpha=0.85, ncol=2)
    fig.suptitle(f"norm Q3: does the norm switch on with the absorbed levels the way the cost "
                 f"does?  {tag}, anchor {base.anchor}, critic `{diet}`, identical rows at "
                 f"every checkpoint.", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, dpi=130)
    print("wrote", path)


# ---------------------------------------------------------------------------

def addendum(args):
    """The glitch-world arm, appended to an existing `tables.md` rather than rewriting it."""
    import datetime
    doc = ["", "---", "",
           "# Addendum (2026-09-17) — the two worlds: where the continuation belongs to the "
           "flagged token, and where it does not", "",
           "The main round found the outcome-trained critic pricing the illegal token as "
           "BETTER news than its surprisal-matched legal twin (`dR` > 0). On the swap "
           "stimuli the violator is the token the regrown continuation belongs to, so that "
           "sign is confounded with re-parsing being the right policy. This addendum adds "
           "`basalis/worlds.py`'s other world: the SAME window with no edit, the token at "
           "the SAME `t_v` replaced by an illegal one, where the continuation is the "
           "original stream and belongs to a legal token instead. Same twin construction "
           "(legal replacement at matched model surprisal after a bit-identical prefix, "
           "token-balanced, the parent's 0.30 caliper), same windows, same positions, same "
           "`(k*, j)` strata. `glitch_s`, not `glitch_m`: the glitch token is the "
           "illegal-under-the-clean-prefix token whose model surprisal is nearest the "
           "edit's own violator, since a uniform draw is 1.4-2.1 nats more surprising and "
           "would let the contrast win on the flagged token alone.", "",
           "The clean-prefix legal law `p_L(. | Wo_<t)` is not on file for `swap65k` "
           "(`with_reference=False` in its build), so it was recomputed with `stimuli.py`'s "
           "own recipe and gated on the assertion that build makes — the unedited stream "
           "never hits probability zero (`norm_pLorig_swap65k.npz`).", ""]
    for tag in args.tags.split(","):
        runs = {}
        for f in sorted(glob.glob(os.path.join(args.dir,
                                               f"step*_norm_{tag}_{args.addendum_tag}.json"))):
            r = json.load(open(f))
            runs[int(r["step"])] = r
        if not runs:
            continue
        doc += [f"## {tag} — the glitch world beside the edit world", ""]
        for st in sorted(runs):
            r = runs[st]
            path = os.path.join(args.dir,
                                f"step{st:06d}_norm_{tag}_{args.addendum_tag}.npz")
            Ts = {}
            for pre, kk in (("tw__", "twin"), ("gl__", "glitch")):
                T = Twins(path, r, caliper=args.caliper, pre=pre, key=kk)
                if T.ok:
                    Ts[pre] = T
            if len(Ts) < 2:
                doc += [f"**step {st}: only {list(Ts)} available**", ""]
                continue
            ma, mb, ncom = common_windows(Ts["tw__"], Ts["gl__"])
            masks = {"tw__": ma, "gl__": mb}
            doc += [f"### step {st}", "", sec_world_head(Ts, r, tag, st), "",
                    f"**`dR` in the two worlds** — token-balanced, on the "
                    f"{ncom} windows that survive BOTH worlds' caliper, so the strata are "
                    f"identical. `dR = R(flagged token) - R(legal twin)`; `V` predicts the "
                    f"actor being RIGHT, so `dR` > 0 means the flagged token is priced as "
                    f"the better news.", "",
                    sec_worlds(Ts, masks, label=" (common windows)"), "",
                    "Each world on its own full caliper set, for comparison:", "",
                    sec_worlds(Ts, {"tw__": None, "gl__": None},
                               label=" (each world's own pairs)"), "",
                    "**By the violation's `k*`**", "",
                    sec_worlds_kstar(Ts, masks), "",
                    "**Matched persistence after the event** (offset 0 is the clean event "
                    "reading; later offsets carry each world's own continuation, which is "
                    "the whole point of the contrast)", "",
                    sec_worlds_trace(Ts, masks), ""]
            if st == max(runs):
                fig_worlds(Ts, masks, r,
                           os.path.join(args.figs, f"norm_worlds_{tag}_{st}.png"))
    p = os.path.join(args.out, getattr(args, "outfile", "tables.md"))
    prev = open(p).read() if os.path.exists(p) else ""
    if "# Addendum (2026-09-17) — the two worlds" in prev:
        prev = prev.split("\n---\n\n# Addendum (2026-09-17) — the two worlds")[0]
    with open(p, "w") as f:
        f.write(prev.rstrip("\n") + "\n" + "\n".join(doc) + "\n")
    print("appended addendum ->", p)


# ---------------------------------------------------------------------------
# 2026-09-17: the mean-and-variance critic (`--mode varhead`)
# ---------------------------------------------------------------------------

def _qbins(x, k):
    """k quantile bins of x; returns the bin index per row (ties handled by rank)."""
    r = np.argsort(np.argsort(x))
    return np.minimum((r * k) // len(x), k - 1)


def _u_rows(dc, vpe, vvar, k=7):
    """The U test: bin the rows by SIGNED outcome surprise and read the variance
    prediction error in each bin.  `dc2_const` is the same quantity with the learned
    variance replaced by its own mean -- a constant-variance baseline -- so the two
    columns say how much of the U the state-dependent head absorbs."""
    b = _qbins(dc, k)
    const = dc ** 2 - (dc ** 2).mean()
    rows = []
    for i in range(k):
        m = b == i
        if not m.sum():
            continue
        rows.append([i + 1, int(m.sum()), fmt(dc[m].mean(), 4), fmt((dc[m] ** 2).mean(), 4),
                     fmt(vvar[m].mean(), 4), fmt(vpe[m].mean(), 4), fmt(const[m].mean(), 4)])
    return rows, b, const


def _u_stat(v, b, k):
    """outer-two-bins mean minus middle-bin mean; > 0 is a U."""
    out = (b == 0) | (b == k - 1)
    mid = b == k // 2
    if not out.sum() or not mid.sum():
        return None
    return float(v[out].mean() - v[mid].mean())


def varhead(args):
    """`--mode varhead`: the 2026-09-17 mean-and-variance cells (`--addendum-tag var`),
    reduced into their own file.  Reads only `step*_norm_<tag>_<vartag>.{json,npz}`."""
    vt = args.addendum_tag
    doc = ["# norm -- the mean-and-variance critic (2026-09-17)", "",
           "Facts only. Xiang, Lohrenz & Montague's insula row is a **variance prediction "
           "error**, which a mean-only ridge has no analogue for. On a 0/1 outcome a "
           "variance head is degenerate (`Var = V(1 - V)`, a function of the mean), so "
           "this needs a continuous target. `norm/task.py`'s `var_target` knob (default "
           "off, so every banked cell is untouched) fits, on the SAME cached states, the "
           "same lambda ladder and the same val rows as the banked critic:", "",
           "- a **mean head** `Vc[l, d]` on a continuous target, and", "",
           "- a **variance head** `Vvar[l, d]` fitted on that head's own squared training "
           "residual `(y - Vc)^2`.", "",
           "Then at the event, one token before it, `dc = y - Vc_pre` is the outcome "
           "surprise in the continuous currency and **`vpe = dc^2 - Vvar_pre`** is the "
           "variance prediction error.", "",
           "**The target.** `lmean`: `y(t) = mean over l = 1..4 of o_l(t)`, the actor "
           "being right at each of the four absorbed levels at one position, in "
           "{0, .25, .5, .75, 1}. (`hsum`, the horizon sum over `delta = 0..8`, was tried "
           "first and fits far worse -- mean-head held-out R^2 0.03-0.06 against `lmean`'s "
           "0.20-0.27 -- so it is not carried here.)", "",
           "**Degeneracy gate.** `R2 var | quad(mean)` is the held-out R^2 of the best "
           "quadratic in the mean head's own output, fitted on the same training rows and "
           "scored on the same val rows. If the variance head does not beat it, there is "
           "no variance signal in the state beyond the mean and the column set is not "
           "worth reading.", "",
           "Reproduction:", "", "```bash", "cd experiments   # MODAL_PROFILE=chromatic",
           "D=/data/v16_s2_L6_m4_distinct/logit_reading",
           "modal run -m rhm.logit_reading.striatum.norm.task::norm_sweep \\",
           "    --cells \"$D/traj_a1_s42/step064000.pt:swap65k:0,"
           "$D/traj_a1_s42/step000000.pt:swap65k:0,$D/traj_a1_s42/step064000.pt:a1:1\" \\",
           "    --var-target lmean --max-twins 0 --tag var",
           "python -m rhm.logit_reading.striatum.norm.analyze <dir>/traj_a1_s42 \\",
           "    --mode varhead --addendum-tag var --outfile tables_varhead.md", "```", ""]
    for tag in args.tags.split(","):
        runs = {}
        for f in sorted(glob.glob(os.path.join(args.dir, f"step*_norm_{tag}_{vt}.json"))):
            r = json.load(open(f))
            runs[int(r["step"])] = r
        if not runs:
            continue
        doc += [f"## venue `{tag}`", "", "### The two heads' held-out fits", ""]
        rows = []
        for st in sorted(runs):
            vh = runs[st].get("var_head")
            if not vh:
                continue
            for ci, cn in enumerate(vh["cols"]):
                rows.append([st, vh["target"], cn, fmt(vh["target_mean"][ci], 4),
                             fmt(vh["target_sd"][ci], 4), fmt(vh["r2_mean"][ci], 4),
                             fmt(vh["resid2_mean"][ci], 4), fmt(vh["r2_var"][ci], 4),
                             fmt(vh["r2_var_from_mean_quadratic"][ci], 4)])
        doc += [tbl(rows, ["step", "target", "col", "E[y]", "sd(y)", "R2 mean",
                           "E[(y-Vc)^2]", "R2 var", "R2 var | quad(mean)"]), ""]
        for st in sorted(runs):
            path = os.path.join(args.dir, f"step{st:06d}_norm_{tag}_{vt}.npz")
            if not os.path.exists(path):
                continue
            Z = np.load(path)
            for an in args.anchors.split(","):
                pre = f"{an}__"
                cols = sorted({k[len(pre):][3:] for k in Z.files
                               if k.startswith(pre + "dc_")})
                if not cols:
                    continue
                anm = "the edit onset (`first_diff`)" if an == "fd" else "`t_v`"
                doc += [f"### step {st}, anchor {anm}", ""]
                for cn in cols:
                    g = lambda k: np.asarray(Z[pre + k], np.float64)
                    te = np.asarray(Z[pre + "split"]) == 2
                    ok = te & (np.asarray(Z[pre + "var_ok"]) > 0)
                    if ok.sum() < 200:
                        continue
                    dc, vpe = g(f"dc_{cn}")[ok], g(f"vpe_{cn}")[ok]
                    vvar = g(f"vvar_pre_{cn}")[ok]
                    k = 7
                    r_, b_, const = _u_rows(dc, vpe, vvar, k)
                    doc += [f"**`{cn}`** -- {int(ok.sum())} held-out rows. "
                            f"`E[dc]` {dc.mean():.4f}, `E[dc^2]` {(dc ** 2).mean():.4f}, "
                            f"`E[Vvar_pre]` {vvar.mean():.4f}, `E[vpe]` {vpe.mean():.4f}; "
                            f"corr(Vvar_pre, dc^2) = {np.corrcoef(vvar, dc ** 2)[0, 1]:.4f}.",
                            "", "Binned by the SIGNED outcome surprise `dc` "
                            f"({k} equal-count bins):", "",
                            tbl(r_, ["bin", "n", "mean dc", "mean dc^2", "mean Vvar_pre",
                                     "mean vpe", "mean dc^2 - E[dc^2]"]), "",
                            f"U statistic (outer two bins minus the middle bin): "
                            f"**vpe {fmt(_u_stat(vpe, b_, k), 4)}**, constant-variance "
                            f"baseline {fmt(_u_stat(const, b_, k), 4)}, "
                            f"`Vvar_pre` {fmt(_u_stat(vvar, b_, k), 4)}.", ""]
                    ka = 5
                    ba = _qbins(np.abs(dc), ka)
                    ra = []
                    for i in range(ka):
                        m = ba == i
                        ra.append([i + 1, int(m.sum()), fmt(np.abs(dc)[m].mean(), 4),
                                   fmt(vvar[m].mean(), 4), fmt(vpe[m].mean(), 4)])
                    doc += [f"Binned by `|dc|` ({ka} equal-count bins) -- the ordering "
                            "Xiang's insula row is read by:", "",
                            tbl(ra, ["bin", "n", "mean |dc|", "mean Vvar_pre", "mean vpe"]),
                            ""]
                    # the event response, and the unedited control where one exists
                    et = np.asarray(Z[pre + "etype"])[ok]
                    rr = []
                    for lbl, m in (("all", np.ones(len(et), bool)), ("swap", et == 0),
                                   ("rare", et == 1), ("none (unedited)", et == 2)):
                        if m.sum() < 30:
                            continue
                        rr.append([lbl, int(m.sum()),
                                   fmt(g(f"Rc_{cn}")[ok][m].mean(), 4),
                                   fmt(g(f"Rvar_{cn}")[ok][m].mean(), 4),
                                   fmt(g(f"vvar_pre_{cn}")[ok][m].mean(), 4)])
                    doc += ["The response at the event, by `etype` (`Rc` is the mean head's "
                            "revision, `Rvar` the variance head's -- does the critic's "
                            "expected squared error RISE when the violation arrives):", "",
                            tbl(rr, ["etype", "n", "mean Rc", "mean Rvar",
                                     "mean Vvar_pre"]), ""]
    os.makedirs(args.out, exist_ok=True)
    p = os.path.join(args.out, getattr(args, "outfile", "tables_varhead.md"))
    open(p, "w").write("\n".join(doc) + "\n")
    print("wrote", p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--tags", default="a1,swap65k")
    ap.add_argument("--out", default="rhm/logit_reading/striatum/norm/results")
    ap.add_argument("--figs", default="rhm/logit_reading/striatum/norm/figs")
    ap.add_argument("--caliper", type=float, default=0.3)
    ap.add_argument("--anchors", default="fd,tv")
    ap.add_argument("--mode", default="full",
                    choices=["full", "addendum", "varhead"])
    ap.add_argument("--addendum-tag", default="glitch")
    ap.add_argument("--outfile", default="tables.md",
                    help="basename written inside --out; a second trajectory seed writes "
                         "its own file (e.g. tables_s43.md) so the banked one is never "
                         "overwritten. The addendum appends to the same file.")
    ap.add_argument("--banked-actor", default="",
                    help="reproduction-gate override: '' = the banked traj_a1_s42 actor "
                         "column, 'none' = drop the comparison rows (another trajectory "
                         "seed), or '8000=a,b,c,d,e,f;64000=...' to supply another "
                         "trajectory's own banked column.")
    args = ap.parse_args()
    banked = parse_banked_actor(args.banked_actor)
    if args.mode == "addendum":
        return addendum(args)
    if args.mode == "varhead":
        return varhead(args)
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.figs, exist_ok=True)
    tags = args.tags.split(",")
    anchors = args.anchors.split(",")
    doc = ["# norm — tables", "",
           "Facts only. Generated by `analyze.py`; every diet is read on identical held-out "
           "rows, and the matched row sets never see a diet.", "",
           "**Objects.**  `V[l, d](s_t)` is the outcome-trained critic: a ridge readout of "
           "the frozen state predicting `o_l(t + d)`, the actor being RIGHT at level `l` at "
           "position `t + d`, so higher `V` = a better world and a drop in `V` is bad news. "
           "`V_pre = V[l, a+1](s_(t-1))` is the norm: what the critic expected about the "
           "query's outcome one token BEFORE the event. `R = V[l, a](s_t) - V[l, a+1](s_(t-1))` "
           "is the revision at the event, both terms predicting the same outcome. `Ro` is "
           "`R` recomputed from the unedited stream's states at the same positions — the "
           "same window with nothing done to it. `d = outcome - V_pre` is the outcome "
           "surprise at the query. `dR = R(violator) - R(legal twin)` is the same-prefix "
           "contrast: within a twin pair `s_(t-1)` is bit-identical, so `V_pre` cancels and "
           "`dR = V[l, 0](s_t^viol) - V[l, 0](s_t^twin)`.", "",
           "**Anchors.**  `fd` = the edit onset (`first_diff`), `tv` = the first "
           "exactly-impossible token. Violation rows are `etype == swap` with a `t_v`; at "
           "the `tv` anchor there are no unedited rows by construction, so the unedited "
           "reference there is `Ro` on the identical windows.", ""]

    for tag in tags:
        runs = load(args.dir, tag)
        if not runs:
            continue
        doc += [f"## {tag}", "", "### 0. Reproduction gate and fits", "",
                sec_gate(runs, tag, banked), "", "### 1. The diets", "",
                sec_diets(runs, tag), ""]
        Rall = {}
        for an in anchors:
            Rall[an] = {}
            for st in sorted(runs):
                R = Rows(npz_of(args.dir, tag, st), an)
                if R.ok:
                    Rall[an][st] = R
        for an in anchors:
            for st in sorted(Rall[an]):
                R, r = Rall[an][st], runs[st]
                doc += [f"### 2. Q1a — the norm before the event ({tag}, step {st}, "
                        f"anchor {an})", "", sec_norm_level(R, r), "",
                        sec_norm_shape(R, r), "",
                        f"### 3. Q1b — the response at the event ({tag}, step {st}, "
                        f"anchor {an})", "", sec_response(R, r), "",
                        f"### 4. Q1c — the outcome surprise d = outcome - V(s_(t-1)) "
                        f"({tag}, step {st}, anchor {an})", "",
                        "On the fixed rows the outcome is the same for every diet, so `d` is "
                        "`-V(s_(t-1))` plus a constant: this table is section 2's ordering in "
                        "the currency Xiang, Lohrenz & Montague's reported feeling tracks.", "",
                        sec_delta(R, r), "",
                        f"### 4b. The orderings across diets ({tag}, step {st}, anchor {an})",
                        "", sec_ordering(R, r), ""]
                if an == "tv" and st == max(Rall[an]):
                    fig_calibration(R, r, os.path.join(args.figs, f"norm_calib_{tag}_{st}.png"))
                if an == "fd" and st == max(Rall[an]) and tag == "a1":
                    fig_calibration(R, r, os.path.join(args.figs,
                                                       f"norm_calib_{tag}_{st}_fd.png"))
        # Q3: scope, both checkpoints on identical rows
        for an in anchors:
            if len(Rall[an]) >= 2:
                doc += [f"### 5. Q3 — scope: by the violation's `k*`, checkpoint against "
                        f"checkpoint ({tag}, anchor {an})", "",
                        sec_scope(Rall[an], tag), "", sec_scope_damage(Rall[an], tag), ""]
                if an == "tv":
                    fig_scope(Rall[an], os.path.join(args.figs, f"norm_scope_{tag}.png"), tag)
        # Q2: the twins
        for st in sorted(runs):
            r = runs[st]
            if not r.get("twin", {}).get("n_pairs"):
                continue
            T = Twins(npz_of(args.dir, tag, st), r, caliper=args.caliper)
            if not T.ok:
                continue
            tokbal, tbchk = T.token_balanced()
            doc += [f"### 6. Q2 — the same-prefix twins ({tag}, step {st})", "",
                    sec_twin_head(T, r, tag, st), "",
                    "**Does `dR` scale with the pre-event expectation?**  `dR = R(violator) "
                    "- R(legal twin)`; the pair shares its prefix so `V_pre` is identical "
                    "within it and the regression to the mean cancels.  The `R ~ V_pre` "
                    "column is the confounded comparator.", "",
                    sec_twin_slope(T), "",
                    "Test-split pairs only:", "",
                    sec_twin_slope(T, mask=(T.split == 2), note=" (test split)"), "",
                    "**Token-identity guard.**  `phasic.py`'s `balance_tokens` subset, in "
                    "which every token is the violator exactly as often as it is the twin. "
                    "`dR` is a paired difference, so on such a subset any additive function "
                    "of token identity cancels EXACTLY; the `signed token histogram` column "
                    "is the self-check and must be 0.", "",
                    sec_twin_slope(T, mask=tokbal, note=f" (token-balanced, signed token "
                                                        f"histogram {tbchk})"), "",
                    "**Binned**", "", sec_twin_bins(T), "",
                    "**Inside each `k*` stratum, and residualised on the guards**", "",
                    sec_twin_kstar(T), "",
                    "**Matched persistence after the event** (all pairs at caliper "
                    f"{args.caliper}; the token-balanced version follows)", "",
                    sec_twin_trace(T), "",
                    "**Matched persistence, token-balanced**", "",
                    sec_twin_trace(T, mask=tokbal), ""]
            if st == max(runs):
                fig_twins(T, r, os.path.join(args.figs, f"norm_twins_{tag}_{st}.png"),
                          mask=tokbal)

    p = os.path.join(args.out, getattr(args, "outfile", "tables.md"))
    with open(p, "w") as f:
        f.write("\n".join(doc) + "\n")
    print("wrote", p)


if __name__ == "__main__":
    main()
