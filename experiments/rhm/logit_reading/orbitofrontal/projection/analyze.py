"""Every table in `results/tables.md` and the figures in `figs/`, from the per-cell json /
npz pulled off the volume.  Facts only; no interpretation lives here.

  D=/v16_s2_L6_m4_distinct/logit_reading/traj_a1_s42
  modal volume get rhm-scaling-data $D <dir>/traj_a1_s42
  python -m rhm.logit_reading.orbitofrontal.projection.analyze <dir>/traj_a1_s42 \
      --tags a1,swap65k --out rhm/logit_reading/orbitofrontal/projection/results \
      --figs rhm/logit_reading/orbitofrontal/projection/figs

Every block is read on IDENTICAL held-out rows and identical diets (one split seed, shared
across blocks, diets and checkpoints), and the matched row sets are computed once per
(anchor, level, label) from the label and the guards only -- the matcher never sees a block
or a diet.  The twin pairs are the same pairs for every block by construction: the pair
population is selected by the MODEL's surprisal, before any critic is applied.
"""

import argparse
import glob
import json
import os

import numpy as np

from rhm.logit_reading.striatum.analyze import _auc, fmt, tbl
from rhm.logit_reading.striatum.junction.analyze import Rows
from rhm.logit_reading.striatum.task import ridge_solve

LEVELS = [1, 2, 3, 4, 5, 6]
LS = [1, 2, 3, 4]
BLOCKS = ["state", "logq", "logq_rich", "logq_tok", "pub_hist", "rand16", "pca16"]
FAM_ORDER = ["clean", "full", "natural_sized",
             "out_lo", "out_mid", "out_hi", "dmg_lo", "dmg_mid", "dmg_hi"]
LAMS = [1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]

# norm/README.md section 1's banked table (a1, t_v, 64k), the gate for the state critic
BANKED_NORM = {"out_lo": [0.902, 0.813, 0.627, 0.343],
               "out_mid": [0.927, 0.863, 0.729, 0.485],
               "out_hi": [0.940, 0.897, 0.799, 0.621]}
BANKED_ACTOR = {64000: [0.904, 0.815, 0.705, 0.524, 0.325, 0.141],
                8000: [0.893, 0.776, 0.623, 0.399, 0.218, 0.091]}


def order_diets(names):
    known = [n for n in FAM_ORDER if n in names]
    mix = sorted([n for n in names if n.startswith("mix_q")])
    rest = sorted(n for n in names if n not in known and n not in mix)
    return known + mix + rest


def load(d, tag):
    out = {}
    for f in sorted(glob.glob(os.path.join(d, f"step*_proj_{tag}.json"))):
        if "_smoke" in f:
            continue
        r = json.load(open(f))
        out[int(r["step"])] = r
    return out


def _ols(x, y):
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


def _r2(pred, y):
    y = np.asarray(y, np.float64)
    return float(1.0 - ((y - np.asarray(pred, np.float64)) ** 2).mean() / max(y.var(), 1e-18))


def ridge_cv(X, y, folds=5, lams=LAMS, seed=0):
    """Closed-form ridge with the penalty chosen by k-fold CV on the training rows,
    `striatum/task.py::ridge_solve`'s convention (penalty scaled by trace(A)/d)."""
    X = np.asarray(X, np.float64)
    y = np.asarray(y, np.float64)
    Xb = np.concatenate([X, np.ones((len(X), 1))], 1)
    rng = np.random.default_rng(seed)
    fold = rng.permutation(len(X)) % folds
    score = np.zeros(len(lams))
    for f in range(folds):
        tr, va = fold != f, fold == f
        A = Xb[tr].T @ Xb[tr]
        c = Xb[tr].T @ y[tr]
        for li, lm in enumerate(lams):
            score[li] += _r2(Xb[va] @ ridge_solve(A, c, lm), y[va])
    lam = lams[int(np.argmax(score))]
    A = Xb.T @ Xb
    beta = ridge_solve(A, Xb.T @ y, lam)
    return beta, lam, float(score.max() / folds)


# ---------------------------------------------------------------------------
# populations and headline readouts at an anchor
# ---------------------------------------------------------------------------

def _pop(R, kind):
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


def blocks_present(R):
    return [b for b in BLOCKS if R.g(f"Vpre_{b}|full_l1_a0") is not None]


def diets_present(R, b):
    out = []
    for k in R.Z.files:
        pre = R.pre + f"Vpre_{b}|"
        if k.startswith(pre) and k.endswith("_l1_a0"):
            out.append(k[len(pre):-6])
    return order_diets(out)


def sec_gate(r, st, tag, R):
    rows = []
    acc = r["actor_clean_acc"]
    rows.append([f"{tag} step {st}"] + [fmt(acc[f"post_block7/l{l}"], 3) for l in LEVELS])
    if st in BANKED_ACTOR:
        rows.append([f"banked striatum {st}"] + [fmt(x, 3) for x in BANKED_ACTOR[st]])
    out = ["**Actor clean held-out accuracy (`post_block7`), against the banked "
           "`striatum/` column.**", "",
           tbl(rows, ["run"] + [f"l={l}" for l in LEVELS]), ""]
    vr = r["critic_val_r2"]
    rows = [["state critic, `full`, `l1_d0` held-out R2",
             fmt(vr["state"]["full"]["l1_d0"], 4)],
            ["norm/ banked value (a1 t_v 64k)", "0.2187" if st == 64000 else "--"],
            ["pca16 share of the state's variance", fmt(r["pca16_var_explained"], 4)],
            ["readout rank (dim of what the logits expose)", str(r["readout_span"]["_rank"])],
            ["a random direction's readout span", fmt(r["readout_span"]["_random_direction"], 4)]]
    out += [tbl(rows, ["quantity", "value"]), ""]
    if R is not None and R.ok and tag == "a1" and st == 64000:
        m = _pop(R, "viol")
        rows = []
        for nm, bk in BANKED_NORM.items():
            v = R.g(f"Vpre_state|{nm}_l1_a0")
            if v is None:
                continue
            got = [float(R.g(f"Vpre_state|{nm}_l{l}_a0")[m].mean()) for l in LS]
            rows.append([nm] + [fmt(x, 3) for x in got] + [fmt(x, 3) for x in bk])
        out += [f"**The state critic's norm `V(s_(t-1))` on the {int(m.sum())} fixed "
                f"violation rows, against `norm/README.md` section 1's banked table.**", "",
                tbl(rows, ["diet", "l=1", "l=2", "l=3", "l=4",
                           "banked l=1", "banked l=2", "banked l=3", "banked l=4"]), ""]
    return "\n".join(out)


def sec_fits(r, st, tag):
    """Held-out R2 of V[l, 0] per block, for `full` and the clean-only critic."""
    out = []
    vr, cv = r["critic_val_r2"], r["clean_critic_val_r2"]
    bl = [b for b in BLOCKS if b in vr]
    for nm in ["full", "clean"]:
        rows = []
        for b in bl:
            if nm == "clean":
                src = cv[b]
                sh = None
            else:
                if nm not in vr[b]:
                    continue
                src = vr[b][nm]
                sh = src.get("sh_l1_d0")
            rows.append([b, str(r["block_dims"][b])] +
                        [fmt(src[f"l{l}_d0"], 4) for l in LEVELS] +
                        [fmt(sh, 4) if sh is not None else "--"])
        if rows:
            out += [f"**critic `{nm}`, held-out `R2` of `V[l, 0]` per feature block "
                    f"({tag}, step {st}).**", "",
                    tbl(rows, ["block", "dim"] + [f"l={l}" for l in LEVELS]
                        + ["shuffled l=1"]), ""]
    # every diet, l = 1..4, so the public fit is not a `full`-only statement
    if len(vr.get("state", {})) > 2:
        rows = []
        for nm in order_diets([k for k in vr["state"]]):
            cells = [nm]
            for b in bl:
                cells.append(fmt(vr[b][nm]["l1_d0"], 4))
            rows.append(cells)
        out += [f"**`l1_d0` held-out `R2`, every diet x every block ({tag}, step {st}).**",
                "", tbl(rows, ["diet"] + bl), ""]
    return "\n".join(out)


def sec_span(r, st, tag):
    sp = r["readout_span"]
    diets = order_diets([k.split("/")[0] for k in sp if "/" in k])
    diets = list(dict.fromkeys(diets))
    rows = []
    for nm in diets:
        cells = [nm]
        for l in LS:
            for dd in (0, 1):
                cells.append(fmt(sp.get(f"{nm}/l{l}_d{dd}"), 4))
        rows.append(cells)
    hdr = ["critic"] + [f"l{l}_d{dd}" for l in LS for dd in (0, 1)]
    return "\n".join([
        f"**Readout span ({tag}, step {st}): `||P_M beta|| / ||beta||`, the fraction of the "
        f"state critic's own direction that the output layer can express.  `M` is the "
        f"rank-{sp['_rank']} map `x -> centred logits` (`W_U diag(gamma) (I - 11'/d)`); a "
        f"random direction gives {sp['_random_direction']:.3f}.**", "",
        tbl(rows, hdr), ""])


def sec_norm_level(R, r, tag, st, a=0, pop="viol"):
    """V(s_(t-1)) before the event, per diet, per block, on the identical fixed rows."""
    m = _pop(R, pop)
    ds = r["diet_stats"]
    out = []
    for b in blocks_present(R):
        nms = diets_present(R, b)
        rows = []
        for nm in nms:
            e_o = (ds.get(nm, {}).get("mean_outcome") if nm != "clean"
                   else r.get("clean_critic_mean_outcome"))
            cells = [nm, fmt(e_o, 4)]
            for l in LS:
                v = R.g(f"Vpre_{b}|{nm}_l{l}_a{a}")
                cells.append(fmt(float(v[m].mean()), 4) if v is not None else "--")
            rows.append(cells)
        # Spearman against the diet's E[outcome], inside each family and overall
        srow = []
        for fam, mem in [("out", ["out_lo", "out_mid", "out_hi"]),
                         ("dmg", ["dmg_lo", "dmg_mid", "dmg_hi"]),
                         ("mix", sorted([x for x in nms if x.startswith("mix_q")])),
                         ("all", [x for x in nms if x != "clean"])]:
            mem = [x for x in mem if x in nms]
            if len(mem) < 3:
                continue
            eo = [ds[x]["mean_outcome"] for x in mem]
            for l in LS:
                vv = [float(R.g(f"Vpre_{b}|{x}_l{l}_a{a}")[m].mean()) for x in mem]
                srow.append([fam, l, len(mem), fmt(_spear(eo, vv), 3),
                             fmt(float(max(vv) - min(vv)), 4)])
        out += [f"**block `{b}` -- the norm on the {int(m.sum())} fixed `{pop}` rows "
                f"({tag}, step {st}, anchor {R.anchor}, a = {a}).**", "",
                tbl(rows, ["diet", "diet E[o]"] + [f"l={l}" for l in LS]), ""]
        if srow:
            out += [tbl([[str(c) for c in rr] for rr in srow],
                        ["family", "l", "n diets", "rho(V_pre, E[o])", "spread"]), ""]
    return "\n".join(out)


def sec_norm_shape(R, r, tag, st, a=0, ref="full", l=1):
    """Shift or rescaling?  Regress each diet's norm on `full`'s, inside each block."""
    m = _pop(R, "all")
    out = []
    for b in blocks_present(R):
        x = R.g(f"Vpre_{b}|{ref}_l{l}_a{a}")
        if x is None:
            continue
        x = x[m]
        rows = []
        for nm in diets_present(R, b):
            if nm == ref:
                continue
            y = R.g(f"Vpre_{b}|{nm}_l{l}_a{a}")
            if y is None:
                continue
            bb, c, r2 = _ols(x, y[m])
            rows.append([nm, fmt(bb, 4), fmt(c, 4), fmt(r2, 4),
                         fmt(float(y[m].mean() - x.mean()), 4),
                         fmt(float(y[m].std() / max(x.std(), 1e-12)), 4)])
        if rows:
            out += [f"**block `{b}` -- each diet's `V(s_(t-1))` regressed on `{ref}`'s, "
                    f"level {l}, {int(m.sum())} fixed rows.  Slope 1 with a non-zero "
                    f"intercept = a pure level shift; slope != 1 = state-dependent.**", "",
                    tbl(rows, ["diet", "slope", "intercept", "R2",
                               f"mean shift vs {ref}", "sd ratio"]), ""]
    return "\n".join(out)


def sec_response(R, r, tag, st, a=0, kind="flip", strata="kjt"):
    """The matched response at the event, per block."""
    ds = r["diet_stats"]
    viol, none = _pop(R, "viol"), _pop(R, "none")
    out = []
    for l in LS:
        got = R.matched(kind, l, a, strata)
        mm, lab = (got if got is not None else (None, None))
        rows = []
        for b in blocks_present(R):
            for nm in diets_present(R, b):
                Rv = R.g(f"R_{b}|{nm}_l{l}_a{a}")
                Ro = R.g(f"Ro_{b}|{nm}_l{l}_a{a}")
                if Rv is None:
                    continue
                if nm not in ("full", "clean", "out_lo", "out_mid", "out_hi"):
                    continue
                e_o = (ds.get(nm, {}).get("mean_outcome") if nm != "clean"
                       else r.get("clean_critic_mean_outcome"))
                sc = (float(Ro[none].std()) if none.sum() > 20 else float(Ro.std()))
                dmn = float((Rv - Ro)[viol].mean()) if Ro is not None else float("nan")
                cells = [b, nm, fmt(e_o, 4),
                         fmt(float(R.g(f"Vpre_{b}|{nm}_l{l}_a{a}")[viol].mean()), 4),
                         fmt(float(Rv[viol].mean()), 4), fmt(_sem(Rv[viol]), 4),
                         fmt(sc, 4), fmt(float(Rv[viol].mean()) / max(sc, 1e-12), 3),
                         fmt(dmn, 4)]
                if mm is not None:
                    cells += [fmt(_auc(-Rv[mm], lab[mm])),
                              fmt(_auc(-R.g(f"V_{b}|{nm}_l{l}_a{a}")[mm], lab[mm]))]
                else:
                    cells += ["--", "--"]
                rows.append(cells)
        nmm = 0 if mm is None else len(mm)
        out += [f"**level {l}, anchor {R.anchor}, a = {a} ({tag}, step {st}); "
                f"{int(viol.sum())} violation rows, {nmm} matched rows for the "
                f"realised-damage AUC.**", "",
                tbl(rows, ["block", "diet", "diet E[o]", "mean V_pre", "mean R (viol)",
                           "sem", "sd R unedited", "R / sd", "R - Ro",
                           f"AUC(-R, {kind})", f"AUC(-V, {kind})"]), ""]
    return "\n".join(out)


def sec_surprise(R, r, tag, st, a=0):
    """d = outcome - V_pre on the fixed violation rows, per block."""
    m = _pop(R, "viol")
    ds = r["diet_stats"]
    out = []
    for l in LS:
        oe = R.g(f"oe_l{l}_a{a}")
        if oe is None:
            continue
        o_mean = float(oe[m].mean())
        rows = []
        for b in blocks_present(R):
            for nm in diets_present(R, b):
                v = R.g(f"Vpre_{b}|{nm}_l{l}_a{a}")
                if v is None:
                    continue
                e_o = (ds.get(nm, {}).get("mean_outcome") if nm != "clean"
                       else r.get("clean_critic_mean_outcome"))
                rows.append([b, nm, fmt(e_o, 4), fmt(float(v[m].mean()), 4),
                             fmt(o_mean - float(v[m].mean()), 4)])
        out += [f"**level {l}, a = {a}: the outcome is identical for every critic "
                f"({o_mean:.4f} on {int(m.sum())} rows), so `d` is the norm restated "
                f"({tag}, step {st}).**", "",
                tbl(rows, ["block", "diet", "diet E[o]", "mean V_pre", "d = outcome - V_pre"]),
                ""]
    return "\n".join(out)


# ---------------------------------------------------------------------------
# the twins
# ---------------------------------------------------------------------------

class Pairs:
    """One world's same-prefix pairs, with the raw post-token forecasts and the state
    difference beside the critics' values."""

    def __init__(self, npz_path, r, caliper=0.3, pre="tw__", key="twin"):
        self.Z = np.load(npz_path)
        self.pre = pre
        self.ok = (pre + "V_viol") in self.Z and bool(r.get(key, {}).get("cols"))
        if not self.ok:
            return
        tw = r[key]
        self.cols = {(c[0], c[1], int(c[2]), int(c[3])): i for i, c in enumerate(tw["cols"])}
        self.offs = list(tw["offs"])
        self.poffs = list(tw["poffs"])
        self.Vv = np.asarray(self.Z[pre + "V_viol"], np.float64)
        self.Vt = np.asarray(self.Z[pre + "V_twin"], np.float64)
        self.Fv = np.asarray(self.Z[pre + "F_viol"], np.float64)
        self.Ft = np.asarray(self.Z[pre + "F_twin"], np.float64)
        self.dX = np.asarray(self.Z[pre + "dX"], np.float64)
        self.ks = np.asarray(self.Z[pre + "k_star"])
        self.jj = np.asarray(self.Z[pre + "j"])
        self.tv = np.asarray(self.Z[pre + "t_v"])
        self.split = np.asarray(self.Z[pre + "split"])
        self.s_v = np.asarray(self.Z[pre + "s_v"], np.float64)
        self.s_c = np.asarray(self.Z[pre + "s_c"], np.float64)
        self.ds = self.s_v - self.s_c
        self.keep = np.abs(self.ds) <= caliper
        self.caliper = caliper
        self.Vb = np.asarray(self.Z["basis__Vb"], np.float64) \
            if "basis__Vb" in self.Z else None
        self.blocks = [b for b in BLOCKS if any(c[0] == b for c in self.cols)]

    def V(self, side, block, diet, l, dd, off):
        A = self.Vv if side == "v" else self.Vt
        return A[:, self.offs.index(off), self.cols[(block, diet, l, dd)]]

    def dR(self, block, diet, l):
        return (self.V("v", block, diet, l, 0, 0) - self.V("t", block, diet, l, 0, 0))

    def dV(self, block, diet, l, off):
        return (self.V("v", block, diet, l, 0, off) - self.V("t", block, diet, l, 0, off))

    def vpre(self, block, diet, l):
        return self.V("v", block, diet, l, 1, -1)

    def dF(self, offsets, kind="logq"):
        io = [self.poffs.index(o) for o in offsets]
        a, b = self.Fv[:, io], self.Ft[:, io]
        if kind == "q":
            a, b = np.exp(a), np.exp(b)
        return (a - b).reshape(len(a), -1)

    def dtok(self):
        """onehot(x_flagged) - onehot(x_twin): what a reader who knows only WHICH token
        arrived and which one it replaced has.  The twin subsets are token-balanced, so
        this feature's mean is exactly zero there; its per-pair variance is not."""
        xv = np.asarray(self.Z[self.pre + "x_v"]).astype(np.int64)
        xc = np.asarray(self.Z[self.pre + "x_c"]).astype(np.int64)
        E = np.eye(16)
        return E[xv] - E[xc]

    def scalars(self):
        qv, qt = np.exp(self.Fv[:, 0]), np.exp(self.Ft[:, 0])
        Hv = -(qv * self.Fv[:, 0]).sum(1)
        Ht = -(qt * self.Ft[:, 0]).sum(1)
        kl_vt = (qv * (self.Fv[:, 0] - self.Ft[:, 0])).sum(1)
        kl_tv = (qt * (self.Ft[:, 0] - self.Fv[:, 0])).sum(1)
        return np.stack([Hv - Ht, kl_vt, kl_tv], 1)

    def token_balanced(self, mask=None, seed=0):
        from rhm.logit_reading.phasic import balance_tokens
        m = self.keep if mask is None else (self.keep & mask)
        idx = np.where(m)[0]
        xv = np.asarray(self.Z[self.pre + "x_v"]).astype(np.int64)
        xc = np.asarray(self.Z[self.pre + "x_c"]).astype(np.int64)
        keep = balance_tokens(xv, xc, idx, np.random.default_rng(seed))
        out = np.zeros(len(m), bool)
        out[keep] = True
        vhist = np.bincount(xv[out], minlength=16) - np.bincount(xc[out], minlength=16)
        return out, int(np.abs(vhist).sum())


def sec_twin_head(P, r, tag, st, key="twin", world="edit"):
    tw = r.get(key, {})
    tb, resid = (P.token_balanced() if P.ok else (None, None))
    rows = [["eligible violations", str(tw.get("n_eligible", "--"))],
            ["pairs within the selection caliper %.1f" % tw.get("caliper", 0),
             str(tw.get("n_pairs"))],
            ["pairs saved (|ds| <= %.2f)" % tw.get("save_caliper", 0),
             str(tw.get("n_saved"))],
            ["pairs at the parent's caliper 0.30", str(int(P.keep.sum())) if P.ok else "--"],
            ["... test split only",
             str(int((P.keep & (P.split == 2)).sum())) if P.ok else "--"],
            ["... token-balanced", str(int(tb.sum())) if P.ok else "--"],
            ["signed token histogram on the balanced subset (0 by construction)",
             str(resid) if P.ok else "--"],
            ["max |V_pre(flagged) - V_pre(twin)|",
             fmt(tw.get("check_Vpre_identical_max_abs"), 8)],
            ["max |log q_(t-1) difference|", fmt(tw.get("check_qprev_identical_max_abs"), 8)],
            ["mean |ds| at caliper 0.30",
             fmt(float(np.abs(P.ds[P.keep]).mean()), 4) if P.ok and P.keep.sum() else "--"]]
    for k in ("mean_abs_ds_across_worlds", "mean_surprisal_glitch",
              "mean_surprisal_violator", "n_legal_at_tv_mean"):
        if k in tw:
            rows.append([k.replace("_", " "), fmt(tw[k], 4)])
    return "\n".join([f"**{tag}, step {st}, the {world} world -- the pair population.**", "",
                      tbl(rows, ["quantity", "value"]), ""])


def sec_twin_dR(P, tag, st, diet="full", world="edit", seed=0):
    """dR per level per block, token-balanced, the norm section-3 object."""
    tb, _ = P.token_balanced(seed=seed)
    out = []
    for lab, m in [("token-balanced", tb), ("all pairs at 0.30", P.keep)]:
        rows = []
        for b in P.blocks:
            if (b, diet, 1, 0) not in P.cols:
                continue
            cells = [b]
            for l in LS:
                dr = P.dR(b, diet, l)[m]
                cells += [fmt(float(dr.mean()), 5), fmt(_sem(dr), 5),
                          fmt(float(dr.mean()) / max(_sem(dr), 1e-12), 2)]
            rows.append(cells)
        hdr = ["block"] + [f"{s} l={l}" for l in LS for s in ("dR", "sem", "sigma")]
        out += [f"**`dR = R(flagged) - R(twin)`, critic `{diet}`, {lab} "
                f"(n = {int(m.sum())}) -- {tag}, step {st}, the {world} world.**", "",
                tbl(rows, hdr), ""]
    return "\n".join(out)


def sec_twin_trace(P, tag, st, diet="full", world="edit", T_max=63, seed=0):
    tb, _ = P.token_balanced(seed=seed)
    out = []
    for l in (2, 3):
        rows = []
        for b in P.blocks:
            if (b, diet, l, 0) not in P.cols:
                continue
            cells = [b]
            for off in P.offs:
                good = tb & ((P.tv + off) <= T_max) & ((P.tv + off) >= 0)
                if good.sum() < 20:
                    cells.append("--")
                    continue
                cells.append(fmt(float(P.dV(b, diet, l, off)[good].mean()), 4))
            rows.append(cells)
        out += [f"**matched persistence `dV` by offset, critic `{diet}`, level {l}, "
                f"token-balanced (n = {int(tb.sum())}) -- {tag}, step {st}, the {world} "
                f"world.**", "",
                tbl(rows, ["block"] + [str(o) for o in P.offs]), ""]
    return "\n".join(out)


# ---------------------------------------------------------------------------
# the priced twin: from the forecast difference to the value revision
# ---------------------------------------------------------------------------

def feature_sets(P):
    o0 = [P.poffs[0]]
    fs = {"dlogq@0": P.dF(o0, "logq"),
          "dq@0": P.dF(o0, "q"),
          "dlogq+dq@0": np.concatenate([P.dF(o0, "logq"), P.dF(o0, "q")], 1),
          "dtok (token identity)": P.dtok(),
          "dtok+dlogq@0": np.concatenate([P.dtok(), P.dF(o0, "logq")], 1),
          "scalars@0": P.scalars(),
          "dlogq@0-2": P.dF(P.poffs[:3], "logq"),
          "dlogq@0-4": P.dF(P.poffs, "logq"),
          "dX@0 (ceiling)": P.dX}
    return fs


def sec_priced(P, tag, st, diet="full", world="edit", seed=0):
    """One linear map from the forecast difference to `dR`, fitted on training pairs and
    read on held-out pairs; the state difference beside it as the ceiling."""
    tb, _ = P.token_balanced(seed=seed)
    tr = P.keep & (P.split != 2)
    te = P.keep & (P.split == 2)
    te_b = tb & (P.split == 2)
    fs = feature_sets(P)
    rng = np.random.default_rng(seed + 5)
    out = []
    for l in LS:
        if ("state", diet, l, 0) not in P.cols:
            continue
        y = P.dR("state", diet, l)
        rows = []
        for nm, X in list(fs.items()) + [("dlogq@0 shuffled", None)]:
            if X is None:
                X = fs["dlogq@0"].copy()
                X[tr] = X[tr][rng.permutation(int(tr.sum()))]
            if tr.sum() < X.shape[1] + 20:
                continue
            beta, lam, cv = ridge_cv(X[tr], y[tr], seed=seed)
            pred = np.concatenate([X, np.ones((len(X), 1))], 1) @ beta
            sgn = float(((np.sign(pred[te_b]) == np.sign(y[te_b])).mean())
                        if te_b.sum() else float("nan"))
            rows.append([nm, str(X.shape[1]), f"{lam:.0e}",
                         fmt(cv, 4), fmt(_r2(pred[te], y[te]), 4),
                         fmt(_r2(pred[te_b], y[te_b]), 4),
                         fmt(float(y[te_b].mean()), 5), fmt(_sem(y[te_b]), 5),
                         fmt(float(pred[te_b].mean()), 5), fmt(_sem(pred[te_b]), 5),
                         fmt(sgn, 3),
                         fmt(float(np.corrcoef(pred[te], y[te])[0, 1]), 3)])
        out += [f"**level {l}: the map to `dR` (critic `{diet}`, state block), "
                f"fitted on {int(tr.sum())} training pairs and read on {int(te.sum())} "
                f"held-out pairs ({int(te_b.sum())} token-balanced) -- {tag}, step {st}, "
                f"the {world} world.**", "",
                tbl(rows, ["features", "dim", "lam", "train CV R2", "test R2",
                           "test R2 (tok-bal)", "mean dR (tok-bal)", "sem",
                           "mean predicted", "sem", "sign agreement", "corr"]), ""]
    return "\n".join(out)


def sec_priced_offsets(P, tag, st, diet="full", world="edit", seed=0):
    """The same map at later offsets, where the two streams' continuations diverge."""
    tr = P.keep & (P.split != 2)
    te = P.keep & (P.split == 2)
    out = []
    for l in (2, 3):
        if ("state", diet, l, 0) not in P.cols:
            continue
        rows = []
        for off in P.poffs:
            if off not in P.offs:
                continue
            y = P.dV("state", diet, l, off)
            X = P.dF([off], "logq")
            beta, lam, cv = ridge_cv(X[tr], y[tr], seed=seed)
            pred = np.concatenate([X, np.ones((len(X), 1))], 1) @ beta
            yX = P.dX
            b2, _, _ = ridge_cv(yX[tr], y[tr], seed=seed)
            p2 = np.concatenate([yX, np.ones((len(yX), 1))], 1) @ b2
            rows.append([off, fmt(float(y[te].mean()), 5), fmt(_sem(y[te]), 5),
                         fmt(_r2(pred[te], y[te]), 4),
                         fmt(float(np.corrcoef(pred[te], y[te])[0, 1]), 3),
                         fmt(_r2(p2[te], y[te]), 4),
                         fmt(float(np.abs(P.dF([off], "logq")[te]).mean()), 4)])
        out += [f"**level {l}: `dV` at each offset against the forecast difference AT THAT "
                f"OFFSET (16 dims, causal), and against the state difference at `t_v` "
                f"(256 dims) -- {tag}, step {st}, the {world} world.**", "",
                tbl([[str(c) for c in rr] for rr in rows],
                    ["offset", "mean dV (test)", "sem", "test R2 from dlogq@off", "corr",
                     "test R2 from dX@0", "mean |dlogq| at off"]), ""]
    return "\n".join(out)


def sec_decomp(P, tag, st, diet="full", world="edit", seed=0):
    """`dR = beta . dX` exactly, so split `beta` into the part the logits express and the
    part they cannot.  `beta` is recovered by exact least squares from (dX, dR)."""
    if P.Vb is None:
        return ""
    m = P.keep
    tb, _ = P.token_balanced(seed=seed)
    Vb = P.Vb
    Proj = Vb.T @ Vb
    rows = []
    for l in LS:
        if ("state", diet, l, 0) not in P.cols:
            continue
        y = P.dR("state", diet, l)
        b, *_ = np.linalg.lstsq(P.dX[m], y[m], rcond=None)
        rec = float(_r2(P.dX[m] @ b, y[m]))
        br = Proj @ b
        bu = b - br
        yr, yu = P.dX @ br, P.dX @ bu
        rows.append([l, fmt(rec, 5),
                     fmt(float(np.linalg.norm(br) / max(np.linalg.norm(b), 1e-12)), 4),
                     fmt(float(yr[m].var() / max(y[m].var(), 1e-18)), 4),
                     fmt(float(yu[m].var() / max(y[m].var(), 1e-18)), 4),
                     fmt(float(np.corrcoef(yr[m], y[m])[0, 1]), 3),
                     fmt(float(y[tb].mean()), 5), fmt(float(yr[tb].mean()), 5),
                     fmt(float(yu[tb].mean()), 5)])
    return "\n".join([
        f"**The exact split of `dR` ({tag}, step {st}, the {world} world, critic `{diet}`, "
        f"{int(m.sum())} pairs at caliper 0.30, {int(tb.sum())} token-balanced).  `dR = "
        f"beta . dX` holds by construction (the critic is linear and `V_pre` cancels), so "
        f"`beta` is recovered exactly from the pairs and split into `P_M beta` (what the "
        f"logits express) and its complement.**", "",
        tbl([[str(c) for c in rr] for rr in rows],
            ["l", "R2 of the recovery", "||P beta|| / ||beta||", "var share readable",
             "var share unreadable", "corr(readable, dR)", "mean dR (tb)",
             "mean readable (tb)", "mean unreadable (tb)"]), ""])


def sec_cross_world(Pe, Pg, tag, st, diet="full", seed=0):
    """The edit world's priced map read on the glitch world, at the event and after."""
    if not (Pe.ok and Pg.ok):
        return ""
    out = []
    for l in (2, 3):
        if ("state", diet, l, 0) not in Pe.cols:
            continue
        rows = []
        for off in Pe.poffs:
            if off not in Pe.offs:
                continue
            ye, yg = Pe.dV("state", diet, l, off), Pg.dV("state", diet, l, off)
            Xe, Xg = Pe.dF([off], "logq"), Pg.dF([off], "logq")
            tr = Pe.keep & (Pe.split != 2)
            beta, lam, _ = ridge_cv(Xe[tr], ye[tr], seed=seed)
            pe = np.concatenate([Xe, np.ones((len(Xe), 1))], 1) @ beta
            pg = np.concatenate([Xg, np.ones((len(Xg), 1))], 1) @ beta
            te, tg = Pe.keep & (Pe.split == 2), Pg.keep
            rows.append([off, fmt(float(ye[te].mean()), 5), fmt(float(pe[te].mean()), 5),
                         fmt(_r2(pe[te], ye[te]), 4),
                         fmt(float(yg[tg].mean()), 5), fmt(float(pg[tg].mean()), 5),
                         fmt(_r2(pg[tg], yg[tg]), 4)])
        out += [f"**level {l}: one map, fitted on the EDIT world's training pairs from the "
                f"forecast difference at each offset, read on both worlds ({tag}, step "
                f"{st}).**", "",
                tbl([[str(c) for c in rr] for rr in rows],
                    ["offset", "edit mean dV", "edit mean pred", "edit R2",
                     "glitch mean dV", "glitch mean pred", "glitch R2"]), ""]
    return "\n".join(out)


def priced_summary(P, diet="full", seed=0):
    """(level, test R2, mean dR on the token-balanced held-out pairs, sem, mean predicted)
    for the `dlogq@0` map -- the one row of section 8 the checkpoint trend needs."""
    tb, _ = P.token_balanced(seed=seed)
    tr, te = P.keep & (P.split != 2), P.keep & (P.split == 2)
    te_b = tb & (P.split == 2)
    X = P.dF([P.poffs[0]], "logq")
    out = []
    for l in LS:
        if ("state", diet, l, 0) not in P.cols or tr.sum() < X.shape[1] + 20:
            continue
        y = P.dR("state", diet, l)
        beta, _, _ = ridge_cv(X[tr], y[tr], seed=seed)
        pred = np.concatenate([X, np.ones((len(X), 1))], 1) @ beta
        out.append((l, _r2(pred[te], y[te]), float(y[te_b].mean()), _sem(y[te_b]),
                    float(pred[te_b].mean())))
    return out



def sec_span_trend(R_, tag):
    """The readout span across the trajectory, beside the chance line.  If the critic's
    direction is anti-aligned with the readout ALREADY at random init, the anti-alignment
    is a property of the architecture; if it falls over training, it is learned."""
    steps = sorted(R_)
    if len(steps) < 2:
        return ""
    chance = R_[steps[0]]["readout_span"]["_random_direction"]
    out = []
    for nm in ("full", "clean"):
        rows = []
        for st in steps:
            sp = R_[st]["readout_span"]
            if f"{nm}/l1_d0" not in sp:
                continue
            rows.append([str(st)] + [fmt(sp[f"{nm}/l{l}_d{dd}"], 4)
                                     for l in LS for dd in (0, 1)]
                        + [fmt(float(np.mean([sp[f"{nm}/l{l}_d{dd}"]
                                              for l in LS for dd in (0, 1)])), 4)])
        rows.append(["chance (a random direction)"] + [fmt(chance, 4)] * (2 * len(LS) + 1))
        if len(rows) > 1:
            out += [f"**critic `{nm}` ({tag}): `||P_M beta|| / ||beta||` by checkpoint.  "
                    f"`M` is the rank-{R_[steps[0]]['readout_span']['_rank']} map from the "
                    f"state to the centred logits, so this is the fraction of the critic's "
                    f"own direction the output layer can express at all.**", "",
                    tbl(rows, ["step"] + [f"l{l}_d{dd}" for l in LS for dd in (0, 1)]
                        + ["mean"]), ""]
    # the whole-diet spread at each checkpoint, so the row above is not a `full` artefact
    rows = []
    for st in steps:
        sp = R_[st]["readout_span"]
        vals = [v for k, v in sp.items() if "/" in k]
        rows.append([str(st), str(len(vals)), fmt(float(np.min(vals)), 4),
                     fmt(float(np.mean(vals)), 4), fmt(float(np.max(vals)), 4),
                     fmt(float(np.mean(vals)) / max(chance, 1e-12), 3)])
    out += [f"**every critic column at every checkpoint ({tag}), against the chance line "
            f"{chance:.4f}.**", "",
            tbl(rows, ["step", "n columns", "min", "mean", "max", "mean / chance"]), ""]
    # the priced map's floor across checkpoints, if the twins are present
    return "\n".join(out)


def sec_priced_trend(rows_by_step, tag):
    """The priced map's held-out R2 and the twin `dR` it is fitted to, by checkpoint --
    the random-init floor for part 1 beside the trained cells."""
    if not rows_by_step:
        return ""
    out, rows = [], []
    for st in sorted(rows_by_step):
        for l, r2, mu, se, pr in rows_by_step[st]:
            rows.append([str(st), str(l), fmt(mu, 5), fmt(se, 5), fmt(r2, 4), fmt(pr, 5)])
    return "\n".join([
        f"**The priced twin by checkpoint ({tag}, edit world, critic `full`): the "
        f"token-balanced held-out `dR` and what the map from `dlogq@0` recovers of it.**",
        "", tbl(rows, ["step", "l", "mean dR (tok-bal)", "sem", "test R2",
                       "mean predicted"]), ""])


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def figures(d, tag, st, r, R, P, Pg, figs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(figs, exist_ok=True)
    bl = [b for b in BLOCKS if b in r["critic_val_r2"]]

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
    w = 0.8 / len(bl)
    for i, b in enumerate(bl):
        src = r["critic_val_r2"][b].get("full")
        if src is None:
            continue
        ax[0].bar(np.arange(len(LS)) + i * w, [src[f"l{l}_d0"] for l in LS], w,
                  label=f"{b} ({r['block_dims'][b]})")
    ax[0].set_xticks(np.arange(len(LS)) + 0.4)
    ax[0].set_xticklabels([f"l={l}" for l in LS])
    ax[0].set_ylabel("held-out $R^2$ of $V[l,0]$")
    ax[0].set_title(f"critic `full` fit by feature block — {tag} step {st}")
    ax[0].legend(fontsize=7)
    sp = r["readout_span"]
    nms = [k for k in ("full", "out_lo", "out_hi", "clean") if f"{k}/l1_d0" in sp]
    for nm in nms:
        ax[1].plot(LS, [sp[f"{nm}/l{l}_d0"] for l in LS], "o-", label=nm)
    ax[1].axhline(sp["_random_direction"], ls="--", c="k", lw=1,
                  label="random direction")
    ax[1].set_xlabel("level"); ax[1].set_ylabel(r"$\|P_M\beta\|/\|\beta\|$")
    ax[1].set_title("fraction of the critic direction the logits express")
    ax[1].legend(fontsize=7); ax[1].set_ylim(0, None)
    fig.tight_layout()
    fig.savefig(os.path.join(figs, f"proj_fit_{tag}_{st}.png"), dpi=130)
    plt.close(fig)

    if R is not None and R.ok:
        bpl = [b for b in blocks_present(R) if b in ("state", "logq", "pub_hist", "pca16")]
        fam = [x for x in ["out_lo", "out_mid", "out_hi"] if x in diets_present(R, "state")]
        if fam:
            m = _pop(R, "viol")
            fig, ax = plt.subplots(1, len(bpl), figsize=(3.4 * len(bpl), 3.6), sharey=False)
            ax = np.atleast_1d(ax)
            for i, b in enumerate(bpl):
                for nm in fam:
                    ax[i].plot(LS, [float(R.g(f"Vpre_{b}|{nm}_l{l}_a0")[m].mean())
                                    for l in LS], "o-", label=nm)
                ax[i].set_title(b, fontsize=9); ax[i].set_xlabel("level")
            ax[0].set_ylabel(r"norm $V(s_{t-1})$")
            ax[0].legend(fontsize=7)
            fig.suptitle(f"the norm per world, per feature block — {tag} step {st}",
                         fontsize=10)
            fig.tight_layout()
            fig.savefig(os.path.join(figs, f"proj_calib_{tag}_{st}.png"), dpi=130)
            plt.close(fig)

    if P is not None and P.ok:
        worlds = [("edit", P)] + ([("glitch", Pg)] if (Pg is not None and Pg.ok) else [])
        fig, ax = plt.subplots(1, len(worlds), figsize=(6.0 * len(worlds), 4.0))
        ax = np.atleast_1d(ax)
        for i, (wn, Pw) in enumerate(worlds):
            tb, _ = Pw.token_balanced()
            for b in Pw.blocks:
                if (b, "full", 1, 0) not in Pw.cols:
                    continue
                mu = [float(Pw.dR(b, "full", l)[tb].mean()) for l in LS]
                se = [_sem(Pw.dR(b, "full", l)[tb]) for l in LS]
                ax[i].errorbar(LS, mu, yerr=se, marker="o", capsize=2, label=b, lw=1.2)
            ax[i].axhline(0, c="k", lw=0.8)
            ax[i].set_title(f"{wn} world (n = {int(tb.sum())})", fontsize=10)
            ax[i].set_xlabel("level")
        ax[0].set_ylabel(r"$dR$ (token-balanced)")
        ax[0].legend(fontsize=7)
        fig.suptitle(f"the twin contrast per feature block — {tag} step {st}", fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(figs, f"proj_twins_{tag}_{st}.png"), dpi=130)
        plt.close(fig)

        fs = feature_sets(P)
        tr, te = P.keep & (P.split != 2), P.keep & (P.split == 2)
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.0))
        keys = ["dlogq@0", "dlogq+dq@0", "dlogq@0-4", "dX@0 (ceiling)"]
        for nm in keys:
            X = fs[nm]
            vals = []
            for l in LS:
                y = P.dR("state", "full", l)
                beta, _, _ = ridge_cv(X[tr], y[tr])
                pred = np.concatenate([X, np.ones((len(X), 1))], 1) @ beta
                vals.append(_r2(pred[te], y[te]))
            ax[0].plot(LS, vals, "o-", label=nm)
        ax[0].axhline(0, c="k", lw=0.8)
        ax[0].set_xlabel("level"); ax[0].set_ylabel("held-out $R^2$ for $dR$")
        ax[0].set_title("the priced twin: forecast difference vs state difference")
        ax[0].legend(fontsize=7)
        if P.Vb is not None:
            Proj = P.Vb.T @ P.Vb
            rs, us = [], []
            m = P.keep
            for l in LS:
                y = P.dR("state", "full", l)
                b, *_ = np.linalg.lstsq(P.dX[m], y[m], rcond=None)
                br = Proj @ b
                rs.append(float((P.dX[m] @ br).var() / max(y[m].var(), 1e-18)))
                us.append(float((P.dX[m] @ (b - br)).var() / max(y[m].var(), 1e-18)))
            ax[1].bar(np.array(LS) - 0.17, rs, 0.34, label="readable (in the logits)")
            ax[1].bar(np.array(LS) + 0.17, us, 0.34, label="unreadable")
            ax[1].set_xlabel("level"); ax[1].set_ylabel(r"variance share of $dR$")
            ax[1].set_title("the exact split of the value revision")
            ax[1].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(figs, f"proj_priced_{tag}_{st}.png"), dpi=130)
        plt.close(fig)


# ---------------------------------------------------------------------------

REPRO = """**Reproduction.**  Run 2026-09-17 on the `chromatic` Modal workspace, six L4 containers fanned
out by CPU coordinators (73-173 s wall each, peak RSS 5.9-7.8 GB against a 12 GB request; apps
`ap-p4hMpBtni78pvk8zs9c4dX` for the 8k / 64k cells and `ap-c8TeDqZMI70F1WtNoH4bY5` for step 0, with
the step-0 `swap65k` cell re-run at `--max-twins 8000` as `ap-Xj9nZbMEH83cLTbMjuD25N`).

```bash
cd experiments            # MODAL_PROFILE=chromatic
D=/data/v16_s2_L6_m4_distinct/logit_reading
modal run --detach -m rhm.logit_reading.orbitofrontal.projection.task::proj_sweep \\
    --cells "$D/traj_a1_s42/step064000.pt:a1:1:0,$D/traj_a1_s42/step064000.pt:swap65k:0:1,\\
$D/traj_a1_s42/step008000.pt:a1:1:0,$D/traj_a1_s42/step008000.pt:swap65k:0:1"
modal run --detach -m rhm.logit_reading.orbitofrontal.projection.task::proj_sweep \\
    --cells "$D/traj_a1_s42/step000000.pt:a1:1:0" --max-twins 8000
modal run -m rhm.logit_reading.orbitofrontal.projection.task::proj_ckpt \\
    --ckpt $D/traj_a1_s42/step000000.pt --stim-tag swap65k --no-do-window-diets \\
    --do-glitch --max-twins 8000
# artefacts: .../traj_a1_s42/step{008000,064000}_proj_{a1,swap65k}.json
#            + _rows_tv.npz, _rows_fd.npz, _twins.npz
python -m rhm.logit_reading.orbitofrontal.projection.analyze <dir>/traj_a1_s42 \\
    --tags a1,swap65k --out rhm/logit_reading/orbitofrontal/projection/results \\
    --figs rhm/logit_reading/orbitofrontal/projection/figs
```
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--tags", default="a1,swap65k")
    ap.add_argument("--out", default=None)
    ap.add_argument("--figs", default=None)
    ap.add_argument("--caliper", type=float, default=0.3)
    a = ap.parse_args()
    tags = a.tags.split(",")
    doc = ["# projection — tables", "",
           "Facts only.  Generated by `analyze.py`; every feature block is read on identical "
           "held-out rows and identical diets, the matched row sets never see a block, and "
           "the twin pairs are selected by the model's own surprisal before any critic is "
           "applied.", "",
           "**Objects.**  `V[l, d](s_t)` is the outcome-trained critic: a ridge readout of a "
           "FEATURE BLOCK at position `t` predicting `o_l(t + d)`, the actor being RIGHT at "
           "level `l` at `t + d`.  `state` is `norm/`'s critic verbatim (the frozen trunk's "
           "residual stream); `logq` replaces it with the model's own output distribution at "
           "the same position -- the public belief -- and the others are its variants and "
           "two dimension-matched controls.  `V_pre = V[l, a+1](s_(t-1))` is the norm, "
           "`R = V[l, a](s_t) - V[l, a+1](s_(t-1))` the revision, `d = outcome - V_pre` the "
           "outcome surprise, and `dR = R(flagged) - R(twin)` the same-prefix contrast, in "
           "which `V_pre` cancels exactly for every block (the prefix, and so the forecast "
           "at `t-1`, is identical within a pair).", "",
           "**The priced twin.**  `dF` is the difference between the two streams' "
           "post-token forecasts: `log q(. | w_<=t_v)` on the violator's window minus the "
           "same on its twin's, i.e. the forecast ABOUT `t_v + 1`, and the same at later "
           "offsets, each computed from that window's own continuation and from positions "
           "at or before the offset only.", "", REPRO, ""]
    for tag in tags:
        R_ = load(a.dir, tag)
        if not R_:
            continue
        doc += [f"## {tag}", ""]
        pt_rows = {}
        for st in sorted(R_):
            r = R_[st]
            base = os.path.join(a.dir, f"step{st:06d}_proj_{tag}")
            Rtv = Rows(base + "_rows_tv.npz", "tv", caliper=a.caliper) \
                if os.path.exists(base + "_rows_tv.npz") else None
            Rfd = Rows(base + "_rows_fd.npz", "fd", caliper=a.caliper) \
                if os.path.exists(base + "_rows_fd.npz") else None
            Rp = Rtv if (Rtv is not None and Rtv.ok) else Rfd
            tw = base + "_twins.npz"
            P = Pairs(tw, r, caliper=a.caliper) if os.path.exists(tw) else None
            Pg = Pairs(tw, r, caliper=a.caliper, pre="gl__", key="glitch") \
                if os.path.exists(tw) else None
            doc += [f"### {tag}, step {st}", "", "#### 0. Reproduction gate", "",
                    sec_gate(r, st, tag, Rp),
                    "#### 1. The critic's held-out fit, per feature block", "",
                    sec_fits(r, st, tag),
                    "#### 2. The readout span", "", sec_span(r, st, tag)]
            if Rp is not None and Rp.ok:
                doc += ["#### 3. The norm per world, per block", "",
                        sec_norm_level(Rp, r, tag, st),
                        "#### 4. Shift or rescaling, per block", "",
                        sec_norm_shape(Rp, r, tag, st),
                        "#### 5. The matched response at the event, per block", "",
                        sec_response(Rp, r, tag, st),
                        "#### 6. The outcome surprise, per block", "",
                        sec_surprise(Rp, r, tag, st)]
            if P is not None and P.ok:
                pt_rows[st] = priced_summary(P)
                doc += ["#### 7. The twins", "", sec_twin_head(P, r, tag, st),
                        sec_twin_dR(P, tag, st),
                        sec_twin_trace(P, tag, st),
                        "#### 8. The priced twin", "", sec_priced(P, tag, st),
                        sec_priced_offsets(P, tag, st),
                        "#### 9. The exact split of `dR`", "", sec_decomp(P, tag, st)]
            if Pg is not None and Pg.ok:
                doc += ["#### 10. The glitch world", "",
                        sec_twin_head(Pg, r, tag, st, key="glitch", world="glitch"),
                        sec_twin_dR(Pg, tag, st, world="glitch"),
                        sec_twin_trace(Pg, tag, st, world="glitch"),
                        sec_priced(Pg, tag, st, world="glitch"),
                        sec_priced_offsets(Pg, tag, st, world="glitch"),
                        sec_decomp(Pg, tag, st, world="glitch"),
                        "#### 11. One map across the two worlds", "",
                        sec_cross_world(P, Pg, tag, st)]
            if a.figs:
                try:
                    figures(a.dir, tag, st, r, Rp, P, Pg, a.figs)
                except Exception as e:                       # pragma: no cover
                    print(f"figures failed for {tag} {st}: {e}")
        if len(R_) > 1:
            doc += [f"### {tag} — across checkpoints", "",
                    "#### A. The readout span at random init, 8k and 64k", "",
                    sec_span_trend(R_, tag),
                    "#### B. The priced twin's random-init floor", "",
                    sec_priced_trend(pt_rows, tag)]
    txt = "\n".join(doc) + "\n"
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        with open(os.path.join(a.out, "tables.md"), "w") as f:
            f.write(txt)
        print(f"wrote {os.path.join(a.out, 'tables.md')} ({len(txt)} chars)")
    else:
        print(txt)


if __name__ == "__main__":
    main()
