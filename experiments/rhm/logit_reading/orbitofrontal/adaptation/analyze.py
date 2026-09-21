"""Every table in `results/tables.md` and every figure in `figs/`, from the per-cell
`.json` / `.npz` pulled off the volume.  Facts only; no interpretation lives here.

  D=/v16_s2_L6_m4_distinct/logit_reading/traj_a1_s42
  modal volume get rhm-scaling-data $D <dir>/traj_a1_s42
  python -m rhm.logit_reading.orbitofrontal.adaptation.analyze <dir>/traj_a1_s42 \
      --tags a1,swap65k --out rhm/logit_reading/orbitofrontal/adaptation/results \
      --figs rhm/logit_reading/orbitofrontal/adaptation/figs

Every arm is read on the SAME fixed held-out rows at every checkpoint, and the matched row
sets are computed once per (anchor, level, label) from the label and the guards only -- the
matcher never sees an arm or a checkpoint.  The matching recipe is `striatum/analyze.py`'s
(exact strata on `(k*, j, anchor-position bucket)`, 1:1 nearest neighbour on the model's
surprisal at the anchor, sign-balanced inside `|ds|` bands), reached through
`junction/analyze.py`'s `Rows`, which reads this node's npz unchanged.

The online critics are stored as betas, one per (arm, fitter, checkpoint); the readouts are
those betas applied here to the saved anchor states and twin states.
"""

import argparse
import json
import os

import numpy as np

from rhm.logit_reading.striatum.analyze import _auc, fmt, tbl
from rhm.logit_reading.striatum.junction.analyze import Rows

LS = [1, 2, 3, 4]
A_STORE = [0, 4, 8]
SWITCH_ARMS = ["out_lo2mid", "out_mid2mid", "out_hi2mid",
               "dmg_lo2mid", "dmg_mid2mid", "dmg_hi2mid"]
MIX_ARMS = ["out_lo_mix", "out_hi_mix"]
STATIC_ORDER = ["full", "out_lo", "out_mid", "out_hi", "dmg_lo", "dmg_mid", "dmg_hi"]

# `norm/README.md` section 1: mean `V(s_(t-1))` on the fixed violation rows, a1 `t_v`, 64k
BANKED_VPRE_A1_TV_64K = {"out_lo": [0.902, 0.813, 0.627, 0.343],
                         "out_mid": [0.927, 0.863, 0.729, 0.485],
                         "out_hi": [0.940, 0.897, 0.799, 0.621]}
BANKED_ACTOR = {64000: [0.904, 0.815, 0.705, 0.524, 0.325, 0.141],
                8000: [0.893, 0.776, 0.623, 0.399, 0.218, 0.091]}


def _ols(x, y):
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3 or x.std() == 0:
        return float("nan"), float("nan"), float("nan")
    b = np.cov(x, y, bias=True)[0, 1] / x.var()
    a = y.mean() - b * x.mean()
    return float(b), float(a), float(1.0 - ((y - (a + b * x)) ** 2).mean() / max(y.var(), 1e-18))


def _sem(x):
    x = np.asarray(x, np.float64)
    return float(x.std(ddof=1) / max(np.sqrt(len(x)), 1)) if len(x) > 1 else float("nan")


# ---------------------------------------------------------------------------
# the cell
# ---------------------------------------------------------------------------

class Cell:
    def __init__(self, d, tag, step, caliper=0.30):
        self.tag, self.step = tag, step
        self.r = json.load(open(os.path.join(d, f"step{step:06d}_adapt_{tag}.json")))
        self.Z = np.load(os.path.join(d, f"step{step:06d}_adapt_{tag}.npz"))
        r = self.r
        self.names = r["target_names"]
        self.offs = list(r["ckpt_offsets"])
        self.arms = list(r["arm_order"])
        self.fits = [f[0] for f in r["fitters"]]
        self.fkind = {f[0]: (f[1], f[2]) for f in r["fitters"]}
        self.P = int(r["config"]["phase_windows"])
        self.R = {}
        for an in r["anchors"]:
            R = Rows(os.path.join(d, f"step{step:06d}_adapt_{tag}.npz"), an)
            if R.ok:
                self.R[an] = R
        self.tw = TwinsX(self.Z, r, caliper) if "tw__X_v" in self.Z else None
        self._c = {}                    # an npz member is decompressed on every access

    def z(self, k):
        if k not in self._c:
            self._c[k] = np.asarray(self.Z[k])
        return self._c[k]

    def ti(self, l, dd):
        return self.names.index(f"l{l}_d{dd}")

    def beta(self, arm, fit, ci):
        return self.z(f"beta_{arm}")[self.fits.index(fit), ci]

    def sbeta(self, nm):
        return self.z(f"beta_static_{nm}")

    def diag(self, arm, key, fit):
        return self.z(f"diag_{arm}__{key}")[self.fits.index(fit)]      # (n_ck, n_tgt)

    def X(self, an, which):
        k = f"{an}__" + {"a": "Xa", "b": "Xb", "ao": "Xao", "bo": "Xbo"}[which]
        if k not in self._c:
            self._c[k] = np.asarray(self.Z[k], np.float64)
        return self._c[k]

    def apply(self, B, which, an, l, dd):
        """A critic column applied to one of the four saved anchor state sets."""
        b = B[:, self.ti(l, dd)]
        return self.X(an, which) @ b[:-1].astype(np.float64) + float(b[-1])

    def vpre(self, B, an, l, a=0):
        return self.apply(B, "b", an, l, a + 1)

    def vev(self, B, an, l, a=0):
        return self.apply(B, "a", an, l, a)

    def resp(self, B, an, l, a=0):
        return self.vev(B, an, l, a) - self.vpre(B, an, l, a)

    def resp_o(self, B, an, l, a=0):
        return (self.apply(B, "ao", an, l, a) - self.apply(B, "bo", an, l, a + 1))


class TwinsX:
    """The same-prefix pairs, held as STATES so any critic can be applied to them.  Within a
    pair `s_(t-1)` is identical by construction, so `V_pre` cancels and
    `dR = V[l, 0](s_t^viol) - V[l, 0](s_t^twin) = (X_v - X_c) . b`, the intercept included."""

    def __init__(self, Z, r, caliper=0.30):
        self.Z, self.r = Z, r
        self.Xp = np.asarray(Z["tw__X_pre"], np.float64)
        self.Xv = np.asarray(Z["tw__X_v"], np.float64)
        self.Xc = np.asarray(Z["tw__X_c"], np.float64)
        self.ks = np.asarray(Z["tw__k_star"])
        self.jj = np.asarray(Z["tw__j"])
        self.tv = np.asarray(Z["tw__t_v"])
        self.split = np.asarray(Z["tw__split"])
        self.s_v = np.asarray(Z["tw__s_v"], np.float64)
        self.s_c = np.asarray(Z["tw__s_c"], np.float64)
        self.ds = self.s_v - self.s_c
        self.caliper = caliper
        self.keep = np.abs(self.ds) <= caliper
        self.names = r["target_names"]

    def dR(self, B, l):
        b = B[:, self.names.index(f"l{l}_d0")]
        return (self.Xv - self.Xc) @ b[:-1].astype(np.float64)

    def vpre(self, B, l):
        b = B[:, self.names.index(f"l{l}_d1")]
        return self.Xp @ b[:-1].astype(np.float64) + float(b[-1])

    def token_balanced(self, seed=0):
        from rhm.logit_reading.phasic import balance_tokens
        idx = np.where(self.keep)[0]
        xv = np.asarray(self.Z["tw__x_v"]).astype(np.int64)
        xc = np.asarray(self.Z["tw__x_c"]).astype(np.int64)
        kp = balance_tokens(xv, xc, idx, np.random.default_rng(seed))
        out = np.zeros(len(self.keep), bool)
        out[kp] = True
        v = np.bincount(xv[out], minlength=16) - np.bincount(xc[out], minlength=16)
        return out, int(np.abs(v).sum())


# ---------------------------------------------------------------------------
# 0. the gate, the arms and the fits
# ---------------------------------------------------------------------------

def sec_gate(C):
    r = C.r
    out = [f"### 0. Reproduction gate", ""]
    a = r["actor_clean_acc"]
    rows = [[f"{C.tag} step {C.step}"] + [fmt(a.get(f"post_block7/l{l}")) for l in range(1, 7)]]
    if C.step in BANKED_ACTOR:
        rows.append([f"banked striatum {C.step}"] + [fmt(x) for x in BANKED_ACTOR[C.step]])
    out += ["Actor clean held-out accuracy (`post_block7`) against the banked `striatum/` "
            "column:", "", tbl(rows, ["run"] + [f"l={l}" for l in range(1, 7)]), ""]

    an = "tv" if "tv" in C.R else list(C.R)[0]
    R = C.R[an]
    m = (R.et == 0) & (R.g("t_v") > 0)
    rows = []
    for nm in STATIC_ORDER:
        if f"beta_static_{nm}" not in C.Z:
            continue
        B = C.sbeta(nm)
        cells = [nm, fmt(r["diet_stats"][nm]["mean_outcome"], 4)]
        cells += [fmt(float(C.vpre(B, an, l)[m].mean()), 4) for l in LS]
        bk = BANKED_VPRE_A1_TV_64K.get(nm)
        cells += ([fmt(x, 3) for x in bk] if (bk and C.tag == "a1" and C.step == 64000
                                              and an == "tv") else ["--"] * 4)
        rows.append(cells)
    out += [f"This node's STATIC critics refit on the same diets (same split seed, same "
            f"`norm/diets.py` masks, same per-target lambda selection) -- mean "
            f"`V(s_(t-1))` on the {int(m.sum())} fixed violation rows at anchor `{an}`, "
            f"against `norm/README.md` section 1's banked table.", "",
            tbl(rows, ["static critic", "diet E[o]"] + [f"V_pre l={l}" for l in LS]
                + [f"banked l={l}" for l in LS]), ""]

    rows = []
    for nm in STATIC_ORDER:
        if nm not in r["static"]:
            continue
        s = r["static"][nm]
        rows.append([nm, str(s["n_rows"])]
                    + [fmt(s["val_r2"][f"l{l}_d0"], 4) for l in LS]
                    + [fmt(s["lam"]["l1_d0"], 6)]
                    + [fmt(s["delta_val"][f"l{l}_d0"], 4) for l in LS])
    out += ["The static critics' held-out fit on the fixed validation rows, the penalty "
            "picked per target, and the mean residual there (`d` in row currency):", "",
            tbl(rows, ["static critic", "n rows"] + [f"R2 l={l}" for l in LS]
                + ["lam (l1_d0)"] + [f"d_val l={l}" for l in LS]), ""]
    return "\n".join(out)


def sec_arms(C):
    r = C.r
    rows = []
    for nm in C.arms:
        a = r["arms"][nm]
        rows.append([nm, a["phase_a"], a["phase_b"], a["mode"], str(a["phase_windows"]),
                     fmt(a["E_outcome_a"], 4), fmt(a["E_outcome_b"], 4),
                     fmt(a["E_outcome_b"] - a["E_outcome_a"], 4),
                     fmt(a["E_damage_a"], 4), fmt(a["E_damage_b"], 4),
                     str(a["cell_hist_a_eq_b"])])
    out = ["### 1. The arms, the fitters and the checkpoint grid", "",
           f"Phase length {C.P} windows, block {r['config']['block_win']} windows, "
           f"{r['config']['R_']} Gram rows per window. The per-cell `(etype, j)` counts are "
           f"equal across all six tercile diets: **{r['cell_counts_equal_across_diets']}** "
           f"(so the two phases of a switch arm differ only in the outcome).", "",
           tbl(rows, ["arm", "phase A", "phase B", "mode", "windows/phase", "E[o] A",
                      "E[o] B", "dE[o]", "E[dmg] A", "E[dmg] B", "cell hist A == B"]), ""]
    rows = [[f[0], f[1], str(f[2]) if f[1] != "cum" else "--",
             fmt(r["fgt_lam_per_row"].get(f[0]), 8) if f[1] == "fgt" else "--",
             str(r["win_blocks"].get(f[0], "--")),
             fmt(r["fitter_lam"].get(f[0], {}).get("l1_d0"), 6)] for f in r["fitters"]]
    out += ["The fitters. `win` is a flat sliding window of the last N windows of "
            "experience; `fgt` is exponentially weighted least squares with a per-row "
            "forgetting factor (an effective memory of M windows), which is what recursive "
            "least squares with a forgetting factor converges on at every step; `cum` is "
            "every row seen so far. Every solve is `striatum/task.py::ridge_solve`. The "
            "penalty is selected ONCE per (fitter, target) on the fixed validation rows "
            "from the `out_mid2mid` control at the end of phase A, then held fixed for "
            "every arm and every checkpoint, so no contrast below is a penalty artefact "
            "(`ridge_solve` scales the penalty by `trace(A)/d`, so it is relative to the "
            "sample size and comparable across window lengths).", "",
            tbl(rows, ["fitter", "kind", "memory (windows)", "lam per row",
                       "window (blocks)", "ridge lam (l1_d0)"]), "",
            f"Checkpoints, in windows from the switch (negative = phase A): "
            f"`{C.offs}`.", ""]
    return "\n".join(out)


# ---------------------------------------------------------------------------
# 2. the norm's time course
# ---------------------------------------------------------------------------

def _vpre_course(C, arm, fit, an, l, a, m):
    return np.array([float(C.vpre(C.beta(arm, fit, ci), an, l, a)[m].mean())
                     for ci in range(len(C.offs))])


def sec_norm_course(C, fit, an, a=0, arms=None, alt=None):
    arms = arms or [x for x in SWITCH_ARMS + MIX_ARMS if x in C.arms]
    a2 = [x for x in SWITCH_ARMS if alt is not None and x in alt.arms]
    R = C.R[an]
    m = (R.et == 0) & (R.g("t_v") > 0)
    m2 = None if alt is None else ((alt.R[an].et == 0) & (alt.R[an].g("t_v") > 0))
    out = []
    for l in LS:
        cols = {nm: _vpre_course(C, nm, fit, an, l, a, m) for nm in arms}
        cols.update({nm + " s2": _vpre_course(alt, nm, fit, an, l, a, m2) for nm in a2})
        ref = {nm: float(C.vpre(C.sbeta(nm), an, l, a)[m].mean())
               for nm in STATIC_ORDER if f"beta_static_{nm}" in C.Z}
        hdr = arms + [nm + " s2" for nm in a2]
        rows = []
        for ci, o in enumerate(C.offs):
            rows.append([str(o)] + [fmt(cols[nm][ci], 4) for nm in hdr])
        rows.append(["static (all rows of that diet)"]
                    + [fmt(ref.get(C.r["arms"][nm.replace(" s2", "")]["phase_a"]), 4)
                       for nm in hdr])
        rows.append(["static, phase B diet"]
                    + [fmt(ref.get(C.r["arms"][nm.replace(" s2", "")]["phase_b"]), 4)
                       for nm in hdr])
        out += [f"**`{fit}`, level {l}, anchor `{an}`, a = {a}: mean `V(s_(t-1))` on the "
                f"{int(m.sum())} fixed violation rows.**  Row `0` is the switch; the last "
                f"two rows are the banked static critics on the same rows (the endpoints an "
                f"online critic should approach when it has seen only one phase). Columns "
                f"marked `s2` are the second order seed on the identical rows.", "",
                tbl(rows, ["windows from switch"] + hdr), ""]
    return "\n".join(out)


def _cross(C, g, frac):
    """The first post-switch offset at which |gap| has fallen to `frac` of its value at the
    switch."""
    pre = max(i for i, o in enumerate(C.offs) if o <= 0)
    for i, o in enumerate(C.offs):
        if o > 0 and abs(g[i]) <= frac * abs(g[pre]):
            return o
    return None


def sec_lag(C, an, a=0, l=1, alt=None):
    """How long, in windows, the norm keeps the mark of the world it came from."""
    pre_i = max(i for i, o in enumerate(C.offs) if o <= 0)
    out = []
    for fit in C.fits:
        rows = []
        for fam in ("out", "dmg"):
            ctl = f"{fam}_mid2mid"
            if ctl not in C.arms:
                continue
            for side in ("lo", "hi"):
                arm = f"{fam}_{side}2mid"
                if arm not in C.arms:
                    continue
                for tagn, Cx in [("", C)] + ([(" s2", alt)] if alt is not None
                                             and arm in alt.arms else []):
                    g = norm_gap(Cx, arm, fit, an, l, a)
                    rows.append([arm + tagn, fmt(g[pre_i], 4)]
                                + [("--" if x is None else str(x)) for x in
                                   (_cross(Cx, g, 0.5), _cross(Cx, g, 0.25),
                                    _cross(Cx, g, 0.10))]
                                + [fmt(g[-1], 4)])
        if rows:
            out += [f"**`{fit}`, level {l}, anchor `{an}`** — the gap in mean `V_pre` "
                    f"between an arm and its family's `mid2mid` control, on the identical "
                    f"rows. `gap at the switch` is the last pre-switch checkpoint.", "",
                    tbl(rows, ["arm", "gap at the switch", "windows to half",
                               "to a quarter", "to a tenth", "gap at the end"]), ""]
    return "\n".join(out)


def _mech(fit, C, o):
    """The fraction of the fit's own weight that phase B holds `o` windows after the switch,
    from the fitter's memory alone -- the null the norm would follow if it were a simple
    average of its rows."""
    kind, mem = C.fkind[fit]
    if o <= 0:
        return 0.0
    if kind == "win":
        return float(min(o / max(mem, 1), 1.0))
    if kind == "fgt":
        return float(1.0 - np.exp(-o / max(mem, 1)))
    return float(o / (C.P + o))                       # cum: all rows so far


def norm_gap(C, arm, fit, an, l, a=0):
    """An arm's `V_pre` minus its family's `mid2mid` control, on the identical fixed rows."""
    R = C.R[an]
    m = (R.et == 0) & (R.g("t_v") > 0)
    ctl = arm.split("_")[0] + "_mid2mid"
    return np.array([float((C.vpre(C.beta(arm, fit, ci), an, l, a)
                            - C.vpre(C.beta(ctl, fit, ci), an, l, a))[m].mean())
                     for ci in range(len(C.offs))])


def dev_from_mech(C, arm, fit, an, l, a=0, upto=None):
    """Mean over the post-switch checkpoints of (fraction of the gap closed) minus (the
    fraction the fitter's own memory alone would have replaced).  Positive = faster than a
    simple average of the rows in memory would be."""
    g = norm_gap(C, arm, fit, an, l, a)
    pre = max(i for i, o in enumerate(C.offs) if o <= 0)
    upto = upto or C.P // 2
    sel = [i for i, o in enumerate(C.offs) if 0 < o <= upto]
    if abs(g[pre]) < 1e-9 or not sel:
        return float("nan"), float(g[pre])
    return (float(np.mean([(1 - g[i] / g[pre]) - _mech(fit, C, C.offs[i]) for i in sel])),
            float(g[pre]))


def sec_recal(C, an, a=0, arms=None, alt=None):
    """The fraction of the pre-switch gap (against the family's `mid2mid` control) that has
    closed, beside the fraction the fitter's own memory would give by itself."""
    arms = arms or [x for x in SWITCH_ARMS if x in C.arms and not x.endswith("mid2mid")]
    pre_i = max(i for i, o in enumerate(C.offs) if o <= 0)
    out = []
    for fit in C.fits:
        for l in LS:
            gaps = {arm: norm_gap(C, arm, fit, an, l, a) for arm in arms}
            if alt is not None:
                gaps.update({arm + " s2": norm_gap(alt, arm, fit, an, l, a)
                             for arm in arms if arm in alt.arms})
            rows = []
            for ci, o in enumerate(C.offs):
                if o < 0:
                    continue
                cells = [str(o), fmt(_mech(fit, C, o), 3)]
                for arm in gaps:
                    g0 = gaps[arm][pre_i]
                    cells += [fmt(gaps[arm][ci], 4),
                              fmt(1.0 - gaps[arm][ci] / g0, 3) if abs(g0) > 1e-9 else "--"]
                rows.append(cells)
            out += [f"**`{fit}`, level {l}, anchor `{an}` — the gap in mean `V_pre` against "
                    f"the family's `mid2mid` control on the identical rows, and the fraction "
                    f"of it that has closed, beside the fraction the fitter's own memory "
                    f"alone would have replaced by then.**  `s2` = the second order seed.", "",
                    tbl(rows, ["windows from switch", "memory replaced"]
                        + sum([[f"{a2}: gap", "closed"] for a2 in gaps], [])), ""]
    return "\n".join(out)


def sec_shape(C, fit, an, a=0, ref="full", l=1, arms=None):
    """Is the transient a level shift or a rescaling?  Regress each online critic's `V_pre`
    on the banked `full` critic's across the identical rows, at every checkpoint."""
    arms = arms or [x for x in SWITCH_ARMS if x in C.arms]
    R = C.R[an]
    m = np.ones(len(R.et), bool)
    x = C.vpre(C.sbeta(ref), an, l, a)[m]
    out = []
    for arm in arms:
        rows = []
        for ci, o in enumerate(C.offs):
            y = C.vpre(C.beta(arm, fit, ci), an, l, a)[m]
            b, c, r2 = _ols(x, y)
            rows.append([str(o), fmt(b, 4), fmt(c, 4), fmt(r2, 4),
                         fmt(float(y.mean() - x.mean()), 4),
                         fmt(float(y.std() / max(x.std(), 1e-12)), 4)])
        pa, pb = C.r["arms"][arm]["phase_a"], C.r["arms"][arm]["phase_b"]
        for nm in (pa, pb):
            y = C.vpre(C.sbeta(nm), an, l, a)[m]
            b, c, r2 = _ols(x, y)
            rows.append([f"static {nm}", fmt(b, 4), fmt(c, 4), fmt(r2, 4),
                         fmt(float(y.mean() - x.mean()), 4),
                         fmt(float(y.std() / max(x.std(), 1e-12)), 4)])
        out += [f"**`{arm}`, `{fit}`, level {l}, anchor `{an}`, all {int(m.sum())} fixed "
                f"rows — regression of the online critic's `V(s_(t-1))` on the banked "
                f"`{ref}` critic's.**  Slope 1 with a non-zero intercept = a pure level "
                f"shift; slope != 1 or `R2` < 1 = state-dependent.", "",
                tbl(rows, ["windows from switch", "slope", "intercept", "R2",
                           "mean shift vs " + ref, "sd ratio"]), ""]
    return "\n".join(out)


# ---------------------------------------------------------------------------
# 3. the outcome surprise, in three currencies
# ---------------------------------------------------------------------------

def sec_delta_anchor(C, fit, an, arms=None):
    arms = arms or [x for x in SWITCH_ARMS if x in C.arms]
    R = C.R[an]
    m = (R.et == 0) & (R.g("t_v") > 0)
    out = []
    for l in LS:
        for a in A_STORE:
            oe = R.g(f"oe_l{l}_a{a}")
            ok = R.g(f"ok_l{l}_a{a}").astype(bool)
            sel = m & ok
            rows = []
            for ci, o in enumerate(C.offs):
                cells = [str(o)]
                for nm in arms:
                    vp = C.vpre(C.beta(nm, fit, ci), an, l, a)
                    cells.append(fmt(float(
                        (oe[sel].astype(np.float64) - vp[sel]).mean()), 4))
                rows.append(cells)
            for nm in STATIC_ORDER:
                if f"beta_static_{nm}" not in C.Z:
                    continue
                vp = C.vpre(C.sbeta(nm), an, l, a)
                rows.append([f"static {nm}"] + ["--"] * len(arms))
                rows[-1][1] = fmt(float((oe[sel].astype(np.float64) - vp[sel]).mean()), 4)
            out += [f"**`{fit}`, level {l}, a = {a}, anchor `{an}`: the outcome surprise "
                    f"`d = outcome - V(s_(t-1))` on the {int(sel.sum())} fixed violation "
                    f"rows.**  The outcome is identical for every critic there "
                    f"(mean {fmt(float(oe[sel].mean()), 4)}), so `d` is the norm's "
                    f"calibration restated in the currency Xiang's reported feeling tracks. "
                    f"The `static` rows carry their own value in the first column only.", "",
                    tbl(rows, ["windows from switch"] + arms), ""]
    return "\n".join(out)


def sec_delta_rows(C, fit, arms=None):
    """The row-currency residual: on the fixed validation rows (overall and inside each
    world), and on the rows actually arriving next (the running one)."""
    arms = arms or [x for x in SWITCH_ARMS if x in C.arms]
    out = []
    for l in LS:
        ti = C.ti(l, 0)
        for key, lab in (("delta_val", "all fixed validation rows"),
                         ("delta_val_lo", "fixed validation rows of the LOW world"),
                         ("delta_val_mid", "fixed validation rows of the MIDDLE world"),
                         ("delta_val_hi", "fixed validation rows of the HIGH world"),
                         ("delta_next", "the rows arriving in the NEXT block")):
            rows = []
            for ci, o in enumerate(C.offs):
                cells = [str(o)]
                for nm in arms:
                    cells.append(fmt(float(C.diag(nm, key, fit)[ci, ti]), 4))
                if key == "delta_next":
                    se = [C.diag(nm, "delta_next_sem", fit)[ci, ti] for nm in arms]
                    se = [x for x in se if np.isfinite(x)]
                    cells.append(fmt(float(np.mean(se)), 4) if se else "--")
                rows.append(cells)
            hdr = ["windows from switch"] + arms + (["mean sem"] if key == "delta_next" else [])
            out += [f"**`{fit}`, level {l}, `d = o_l(t) - V[l, 0](s_t)` on {lab}.**", "",
                    tbl(rows, hdr), ""]
        rows = []
        for ci, o in enumerate(C.offs):
            rows.append([str(o)] + [fmt(float(C.diag(nm, "val_r2", fit)[ci, ti]), 4)
                                    for nm in arms]
                        + [str(int(C.diag(arms[0], "n_rows", fit)[ci, ti]))])
        out += [f"**`{fit}`, level {l}: held-out `R2` of `V[l, 0]` on the fixed validation "
                f"rows along the sequence, and the number of (weighted) rows the fit "
                f"stands on.**", "", tbl(rows, ["windows from switch"] + arms
                                         + ["n rows (first arm)"]), ""]
    return "\n".join(out)


# ---------------------------------------------------------------------------
# 4. the response at the event
# ---------------------------------------------------------------------------

def resp_readouts(C, B, an, l, a, mm, lab):
    R = C.R[an]
    viol = (R.et == 0) & (R.g("t_v") > 0)
    none = R.et == 2
    Rv, Ro = C.resp(B, an, l, a), C.resp_o(B, an, l, a)
    sc = float(Ro[none].std()) if none.sum() > 20 else float(Ro[viol].std())
    o_ = {"R (raw)": float(Rv[viol].mean()),
          "R / sd (own units)": float(Rv[viol].mean()) / max(sc, 1e-12),
          "R - Ro": float((Rv - Ro)[viol].mean()),
          "sd of R on unedited": sc}
    if mm is not None:
        o_["AUC(-R, realised damage)"] = _auc(-Rv[mm], lab[mm])
        o_["AUC(-V, realised damage)"] = _auc(-C.vev(B, an, l, a)[mm], lab[mm])
    return o_


def resp_gap(C, arm, fit, an, l, a=0):
    """The arm's response in its own units, minus its family's `mid2mid` control's, on the
    identical fixed violation rows."""
    ctl = arm.split("_")[0] + "_mid2mid"
    return np.array([resp_readouts(C, C.beta(arm, fit, ci), an, l, a, None,
                                   None)["R / sd (own units)"]
                     - resp_readouts(C, C.beta(ctl, fit, ci), an, l, a, None,
                                     None)["R / sd (own units)"]
                     for ci in range(len(C.offs))])


def sec_response(C, fit, an, a=0, arms=None, kind="flip", alt=None):
    arms = arms or [x for x in SWITCH_ARMS if x in C.arms]
    a2 = [x for x in arms if alt is not None and x in alt.arms]
    base_arms = list(arms)
    R = C.R[an]
    viol = (R.et == 0) & (R.g("t_v") > 0)
    none = R.et == 2
    statics = [nm for nm in STATIC_ORDER if f"beta_static_{nm}" in C.Z]
    out = []
    for l in LS:
        arms = list(base_arms)
        got = R.matched(kind, l, a, "kjt")
        mm, lab = got if got is not None else (None, None)

        def readouts(B, Cx=C):
            return resp_readouts(Cx, B, an, l, a, mm, lab)

        on = readouts(C.beta(arms[0], fit, 0))
        cache = {nm: [readouts(C.beta(nm, fit, ci)) for ci in range(len(C.offs))]
                 for nm in arms}
        cache.update({nm + " s2": [readouts(alt.beta(nm, fit, ci), alt)
                                   for ci in range(len(C.offs))] for nm in a2})
        arms = arms + [nm + " s2" for nm in a2]
        scache = {nm: readouts(C.sbeta(nm)) for nm in statics}
        nmm = 0 if mm is None else len(mm)
        src = ("the unedited `none` windows at this anchor" if none.sum() > 20 else
               "the ORIGINAL-stream state of the very same windows and positions")
        hdr_note = (f"**`{fit}`, level {l}, anchor `{an}`, a = {a}.**  n violation rows "
                    f"{int(viol.sum())}, matched rows {nmm}. `R / sd` is in each critic's "
                    f"own units, the spread taken over {src}; `Ro` is the same revision "
                    f"computed from the unedited stream's states at the same positions.")
        out += [hdr_note, ""]
        for k in on:
            nd = 3 if k.startswith("AUC") or "own units" in k else 4
            rows = [[str(o)] + [fmt(cache[nm][ci][k], nd) for nm in arms]
                    for ci, o in enumerate(C.offs)]
            rows += [[f"static {nm}"] + [fmt(scache[nm][k], nd)] + ["--"] * (len(arms) - 1)
                     for nm in statics]
            out += [f"*{k}*", "", tbl(rows, ["windows from switch"] + arms), ""]
        if mm is not None:
            g = {"model surprisal": R.nll, "k*": R.ks.astype(float), "j": R.jj.astype(float),
                 "anchor position": R.t0.astype(float)}
            out += ["guards on the matched rows (arm- and checkpoint-independent): "
                    + "  ".join(f"{k} {fmt(_auc(v[mm], lab[mm]))}" for k, v in g.items()), ""]
    return "\n".join(out)


# ---------------------------------------------------------------------------
# 5. the same-prefix twins
# ---------------------------------------------------------------------------

def sec_twins(C, fit, arms=None, seed=0):
    T = C.tw
    if T is None:
        return ""
    arms = arms or [x for x in SWITCH_ARMS if x in C.arms]
    bal, resid = T.token_balanced(seed)
    tw = C.r["twin"]
    rows = [["eligible violations", str(tw.get("n_eligible"))],
            ["pairs saved (selection caliper %.2f)" % tw.get("caliper", 0),
             str(tw.get("n_pairs"))],
            ["pairs at the parent's caliper 0.30", str(int(T.keep.sum()))],
            ["pairs token-balanced at 0.30", str(int(bal.sum()))],
            ["signed token histogram on that subset (0 by construction)", str(resid)],
            ["max |state(t-1) violator - twin| (0 by construction)",
             fmt(tw.get("check_state_pre_identical_max_abs"), 8)],
            ["max |V_pre violator - twin| (0 by construction)",
             fmt(tw.get("check_Vpre_identical_max_abs"), 8)],
            ["max |log q_(t-1) difference| (0 by construction)",
             fmt(tw.get("check_qprev_identical_max_abs"), 8)],
            ["mean |ds| at caliper 0.30", fmt(float(np.abs(T.ds[T.keep]).mean()), 4)]]
    out = ["The same-prefix twin population (`phasic.py`'s construction: the violator "
           "replaced by the legal token of nearest model surprisal after an identical "
           "prefix).", "", tbl(rows, ["quantity", "value"]), ""]
    for l in LS:
        rows = []
        for ci, o in enumerate(C.offs):
            cells = [str(o)]
            for nm in arms:
                dR = T.dR(C.beta(nm, fit, ci), l)[bal]
                cells += [fmt(float(dR.mean()), 5), fmt(_sem(dR), 5)]
            rows.append(cells)
        for nm in STATIC_ORDER:
            if f"beta_static_{nm}" not in C.Z:
                continue
            dR = T.dR(C.sbeta(nm), l)[bal]
            cells = [f"static {nm}", fmt(float(dR.mean()), 5), fmt(_sem(dR), 5)]
            rows.append(cells + ["--"] * (len(arms) * 2 - 2))
        out += [f"**`{fit}`, level {l}: `dR = R(violator) - R(twin)` on the "
                f"{int(bal.sum())} token-balanced pairs at caliper 0.30.**  Positive = the "
                f"critic is MORE optimistic after the illegal token than after its "
                f"surprisal-matched legal twin sharing the identical prefix. The `static` "
                f"rows carry their own value in the first two columns only.", "",
                tbl(rows, ["windows from switch"]
                    + sum([[f"{nm}: dR", "sem"] for nm in arms], [])), ""]
    return "\n".join(out)


# ---------------------------------------------------------------------------
# 9. across order seeds
# ---------------------------------------------------------------------------

GAP_MIN_V = 0.02        # a `V_pre` gap smaller than the control's own band is not readable
GAP_MIN_R = 0.05        # likewise for the response, in its own units


def _agree(a, b):
    if not (np.isfinite(a) and np.isfinite(b)):
        return "--"
    return "yes" if (a > 0) == (b > 0) else "NO"


def sec_seeds(C, alt, an, a=0):
    """The two claims the second order seed was run to settle, each stated as a sign that
    either agrees across the seeds or does not."""
    arms = [x for x in SWITCH_ARMS if x in C.arms and not x.endswith("mid2mid")
            and x in alt.arms]
    fits = [f for f in C.fits if C.fkind[f][0] != "cum"]
    out = [f"Seed 1 is `order_seed = {C.r['config']['order_seed']}`, seed 2 is "
           f"`order_seed = {alt.r['config']['order_seed']}`. The seed sets the order of the "
           f"windows within each phase **and** which windows the per-cell trim to a whole "
           f"number of blocks keeps, so the two seeds differ in the sequence and in its "
           f"membership; everything else — the checkpoint, the venue, the split, the "
           f"diets' cell histograms, the held-out rows the readouts are taken on, the "
           f"fitters and the ridge penalties — is identical. Both seeds are read on the "
           f"same {int(((C.R[an].et == 0) & (C.R[an].g('t_v') > 0)).sum())} fixed violation "
           f"rows at anchor `{an}`.", ""]

    # ---- claim (2) ---------------------------------------------------------------------
    rows, n_ok, n_agree = [], 0, 0
    for fam in ("out", "dmg"):
        for arm in [x for x in arms if x.startswith(fam)]:
            for fit in fits:
                for l in LS:
                    d1, g1 = dev_from_mech(C, arm, fit, an, l, a)
                    d2, g2 = dev_from_mech(alt, arm, fit, an, l, a)
                    ok = abs(g1) >= GAP_MIN_V and abs(g2) >= GAP_MIN_V
                    ag = _agree(d1, d2) if ok else "gap < %.2f" % GAP_MIN_V
                    n_ok += ok
                    n_agree += ok and ag == "yes"
                    rows.append([arm, fit, str(l), fmt(g1, 4), fmt(g2, 4),
                                 fmt(d1, 3), fmt(d2, 3), ag])
    out += ["#### (2) The deviation from the mechanical null, with a sign per level and "
            "family", "",
            "For each cell: the `V_pre` gap against the family's `mid2mid` control at the "
            "last pre-switch checkpoint, and the mean over the post-switch checkpoints "
            "(out to half a phase) of (fraction of that gap closed) minus (fraction of the "
            "fitter's own memory replaced by then). **Positive = the norm re-calibrated "
            "faster than a simple average of the rows in its memory would have.** A cell "
            f"counts only where the gap clears {GAP_MIN_V} in both seeds — below that the "
            "fraction is a ratio of two noise terms. `cum` is excluded (its mechanical "
            "fraction is ~0 over this range, so the ratio is degenerate).", "",
            tbl(rows, ["arm", "fitter", "l", "gap s1", "gap s2", "deviation s1",
                       "deviation s2", "sign agrees"]), "",
            f"**Across the two order seeds: of {n_ok} readable cells the sign of the "
            f"deviation agrees in {n_agree} ({n_agree / max(n_ok, 1):.0%}).**", ""]

    # ---- claim (4) ---------------------------------------------------------------------
    rows, n_ok, n_agree, n_close1, n_close2 = [], 0, 0, 0, 0
    for arm in arms:
        for fit in fits:
            for l in LS:
                q1, q2 = resp_gap(C, arm, fit, an, l, a), resp_gap(alt, arm, fit, an, l, a)
                pre = max(i for i, o in enumerate(C.offs) if o <= 0)
                h1, h2 = _cross(C, q1, 0.5), _cross(alt, q2, 0.5)
                ok = abs(q1[pre]) >= GAP_MIN_R and abs(q2[pre]) >= GAP_MIN_R
                ag = _agree(q1[pre], q2[pre]) if ok else "gap < %.2f" % GAP_MIN_R
                n_ok += ok
                n_agree += ok and ag == "yes"
                n_close1 += ok and h1 is not None
                n_close2 += ok and h2 is not None
                rows.append([arm, fit, str(l), fmt(q1[pre], 3), fmt(q2[pre], 3), ag,
                             "--" if h1 is None else str(h1),
                             "--" if h2 is None else str(h2),
                             fmt(q1[-1], 3), fmt(q2[-1], 3)])
    out += ["#### (4) The response carrying the history", "",
            "The anticipatory response in each critic's own units (`R / sd`), minus its "
            "family's `mid2mid` control's, on the identical fixed violation rows. If the "
            "response carried no history the gap would be nil at the switch; if it carries "
            "it and re-calibrates, the gap is non-zero at the switch, has the same sign in "
            "both seeds, and falls to half within the phase. A cell counts only where the "
            f"gap clears {GAP_MIN_R} in both seeds.", "",
            tbl(rows, ["arm", "fitter", "l", "gap at switch s1", "gap at switch s2",
                       "sign agrees", "windows to half s1", "to half s2",
                       "gap at end s1", "gap at end s2"]), "",
            f"**Across the two order seeds: of {n_ok} readable cells the sign of the "
            f"pre-switch response gap agrees in {n_agree} "
            f"({n_agree / max(n_ok, 1):.0%}); the gap falls to half within the phase in "
            f"{n_close1} of {n_ok} (seed 1) and {n_close2} of {n_ok} (seed 2).**", ""]

    # ---- the world-change excursion ------------------------------------------------------
    out += _sec_seeds_dnext(C, alt)
    return "\n".join(out)


def dnext_z(C, arm, fit, l):
    """The outcome surprise on the rows arriving NEXT, in units of its own standard error
    across the windows of that block.  In steady state the critic is calibrated to the world
    it is in, so this sits at zero; it is the value side's "has the world moved" organ."""
    ti = C.ti(l, 0)
    v = C.diag(arm, "delta_next", fit)[:, ti].astype(np.float64)
    se = C.diag(arm, "delta_next_sem", fit)[:, ti].astype(np.float64)
    return np.where(se > 0, v / np.where(se > 0, se, 1.0), np.nan)


def _dnext_return(C, z, thr=2.0):
    """The first post-switch offset at which the excursion is back inside +-`thr` sem."""
    for i, o in enumerate(C.offs):
        if o > 0 and np.isfinite(z[i]) and abs(z[i]) <= thr:
            return o
    return None


Z_MIN = 2.0


def _sec_seeds_dnext(C, alt):
    arms = [x for x in SWITCH_ARMS if x in C.arms and not x.endswith("mid2mid")
            and x in alt.arms]
    ctls = [x for x in (["out_mid2mid", "dmg_mid2mid"] + MIX_ARMS)
            if x in C.arms and x in alt.arms]
    ci0 = C.offs.index(0)
    rows, n_ok, n_agree, n_ret1, n_ret2 = [], 0, 0, 0, 0
    for arm in arms:
        for fit in C.fits:
            for l in LS:
                z1, z2 = dnext_z(C, arm, fit, l), dnext_z(alt, arm, fit, l)
                a1, a2 = z1[ci0], z2[ci0]
                r1, r2 = _dnext_return(C, z1), _dnext_return(alt, z2)
                ok = abs(a1) >= Z_MIN and abs(a2) >= Z_MIN
                ag = _agree(a1, a2) if ok else "|z| < %.0f" % Z_MIN
                n_ok += ok
                n_agree += ok and ag == "yes"
                n_ret1 += ok and r1 is not None
                n_ret2 += ok and r2 is not None
                rows.append([arm, fit, str(l), fmt(a1, 1), fmt(a2, 1), ag,
                             "--" if r1 is None else str(r1),
                             "--" if r2 is None else str(r2)])
    crows, c_n, c_hit = [], 0, 0
    cmax = 0.0
    for arm in ctls:
        for fit in C.fits:
            for l in LS:
                a1 = dnext_z(C, arm, fit, l)[ci0]
                a2 = dnext_z(alt, arm, fit, l)[ci0]
                for a in (a1, a2):
                    if np.isfinite(a):
                        c_n += 1
                        c_hit += abs(a) >= Z_MIN
                        cmax = max(cmax, abs(a))
                crows.append([arm, fit, str(l), fmt(a1, 1), fmt(a2, 1),
                              _agree(a1, a2) if min(abs(a1), abs(a2)) >= Z_MIN
                              else "|z| < %.0f" % Z_MIN])
    return ["#### (d_next) The world-change excursion, on the rows arriving next", "",
            "`d = o_l(t) - V[l, 0](s_t)` evaluated with the critic as of a checkpoint on the "
            "rows of the block that arrives immediately AFTER it — strictly out of sample "
            "and strictly future, so in steady state a calibrated critic sits at zero. The "
            "value is divided by its own standard error across the windows of that block, "
            "so the column is a z. A cell counts for the sign test where the excursion at "
            f"the switch clears {Z_MIN:.0f} sem in both seeds; `windows to return` is the "
            f"first post-switch checkpoint back inside +-{Z_MIN:.0f} sem. `cum` is kept: it "
            "is the fitter that cannot forget, and what it does here is the point.", "",
            tbl(rows, ["arm", "fitter", "l", "z at switch s1", "z at switch s2",
                       "sign agrees", "windows to return s1", "to return s2"]), "",
            f"**Across the two order seeds: of {n_ok} cells whose excursion clears "
            f"{Z_MIN:.0f} sem in both seeds the sign agrees in {n_agree} "
            f"({n_agree / max(n_ok, 1):.0%}); it returns inside the band within the phase in "
            f"{n_ret1} of {n_ok} (seed 1) and {n_ret2} of {n_ok} (seed 2).**", "",
            "The controls at the same checkpoint — `mid2mid`, whose world does not change, "
            "and the two interleaved arms, which hold the identical rows with the sequence "
            "removed. If the excursion were an artefact of the checkpoint rather than of the "
            "world moving, it would appear here too.", "",
            tbl(crows, ["control", "fitter", "l", "z at switch s1", "z at switch s2",
                        "sign agrees"]), "",
            f"**Across the controls: {c_hit} of {c_n} control cells (both seeds pooled) "
            f"reach {Z_MIN:.0f} sem at the switch, largest |z| = {cmax:.1f}.**", ""]


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _xax(A, C, lab="windows from the switch"):
    """A signed-log x axis: the checkpoint grid is dense around the switch and doubles away
    from it, so a linear axis compresses the whole transition into a sliver."""
    B = int(C.r["config"]["block_win"])
    A.set_xscale("symlog", linthresh=B, linscale=0.6)
    A.axvline(0, color="k", lw=0.8)
    A.set_xlabel(lab, fontsize=8)
    A.tick_params(labelsize=7)


COLOR = {"out_lo2mid": "#1b6ca8", "out_hi2mid": "#c0392b", "out_mid2mid": "#555555",
         "dmg_lo2mid": "#7d3c98", "dmg_hi2mid": "#e67e22", "dmg_mid2mid": "#999999",
         "out_lo_mix": "#7fb3d5", "out_hi_mix": "#f1948a"}


def fig_norm(C, fit, an, path, a=0, alt=None):
    plt = _plt()
    R = C.R[an]
    m = (R.et == 0) & (R.g("t_v") > 0)
    arms = [x for x in SWITCH_ARMS + MIX_ARMS if x in C.arms]
    fig, ax = plt.subplots(1, len(LS), figsize=(4.2 * len(LS), 3.6), sharex=True)
    for li, l in enumerate(LS):
        A = ax[li]
        for nm in arms:
            y = [float(C.vpre(C.beta(nm, fit, ci), an, l, a)[m].mean())
                 for ci in range(len(C.offs))]
            A.plot(C.offs, y, "-o", ms=2.6, lw=1.3, color=COLOR.get(nm, "k"), label=nm)
            if alt is not None and nm in alt.arms:
                m2 = (alt.R[an].et == 0) & (alt.R[an].g("t_v") > 0)
                A.plot(alt.offs, [float(alt.vpre(alt.beta(nm, fit, ci), an, l, a)[m2].mean())
                                  for ci in range(len(alt.offs))], "--", lw=0.9,
                       color=COLOR.get(nm, "k"), alpha=0.75)
        for nm in ("out_lo", "out_mid", "out_hi"):
            if f"beta_static_{nm}" in C.Z:
                A.axhline(float(C.vpre(C.sbeta(nm), an, l, a)[m].mean()), ls=":", lw=0.9,
                          color="#888888")
        _xax(A, C)
        A.set_title(f"level {l}")
        if li == 0:
            A.set_ylabel(r"mean $V(s_{t-1})$ on the fixed rows")
    ax[-1].legend(fontsize=6, loc="best")
    fig.suptitle(f"{C.tag} step {C.step} — the norm re-calibrating, fitter `{fit}` "
                 f"(horizontal dotted: the banked static tercile critics"
                 + ("; dashed: the second order seed)" if alt is not None else ")"),
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def fig_fitters(C, an, path, l=3, a=0):
    plt = _plt()
    R = C.R[an]
    m = (R.et == 0) & (R.g("t_v") > 0)
    arms = [x for x in SWITCH_ARMS if x in C.arms]
    fig, ax = plt.subplots(1, len(C.fits), figsize=(3.4 * len(C.fits), 3.4), sharey=True)
    for fi, fit in enumerate(C.fits):
        A = ax[fi]
        for nm in arms:
            y = [float(C.vpre(C.beta(nm, fit, ci), an, l, a)[m].mean())
                 for ci in range(len(C.offs))]
            A.plot(C.offs, y, "-o", ms=2.4, lw=1.2, color=COLOR.get(nm, "k"), label=nm)
        _xax(A, C)
        A.set_title(fit, fontsize=9)
    ax[0].set_ylabel(r"mean $V(s_{t-1})$, level %d" % l)
    ax[-1].legend(fontsize=6)
    fig.suptitle(f"{C.tag} step {C.step} — what 'a running estimate' means: the same "
                 f"course under five memories", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def fig_delta(C, fit, path, l=3):
    plt = _plt()
    arms = [x for x in SWITCH_ARMS if x in C.arms]
    ti = C.ti(l, 0)
    keys = [("delta_val_lo", "fixed rows, LOW world"),
            ("delta_val_mid", "fixed rows, MIDDLE world"),
            ("delta_val_hi", "fixed rows, HIGH world"),
            ("delta_next", "the rows arriving next")]
    fig, ax = plt.subplots(1, len(keys), figsize=(3.6 * len(keys), 3.3), sharey=True)
    for ki, (k, lab) in enumerate(keys):
        A = ax[ki]
        if k == "delta_next":
            se = np.nanmean(np.stack([C.diag(nm, "delta_next_sem", fit)[:, ti]
                                      for nm in arms]), 0)
            A.fill_between(C.offs, -2 * se, 2 * se, color="#cccccc", alpha=0.6, lw=0,
                           label="+-2 sem")
        for nm in arms:
            A.plot(C.offs, C.diag(nm, k, fit)[:, ti], "-o", ms=2.4, lw=1.2,
                   color=COLOR.get(nm, "k"), label=nm)
        A.axhline(0, color="k", lw=0.7, ls="--")
        _xax(A, C)
        A.set_title(lab, fontsize=9)
    ax[0].set_ylabel(r"$\delta = o_\ell(t) - V[\ell,0](s_t)$, level %d" % l)
    ax[-1].legend(fontsize=6)
    fig.suptitle(f"{C.tag} step {C.step} — the outcome surprise, fitter `{fit}`", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def fig_event(C, fit, an, path, a=0, alt=None):
    """The response and the twin contrast along the sequence."""
    plt = _plt()
    R = C.R[an]
    viol = (R.et == 0) & (R.g("t_v") > 0)
    none = R.et == 2
    arms = [x for x in SWITCH_ARMS if x in C.arms]
    T = C.tw
    bal = T.token_balanced()[0] if T is not None else None
    ncol = len(LS)
    fig, ax = plt.subplots(2, ncol, figsize=(3.6 * ncol, 6.2), sharex=True)
    for li, l in enumerate(LS):
        A, Bx = ax[0, li], ax[1, li]
        for nm in arms:
            rr, dd = [], []
            for ci in range(len(C.offs)):
                Bt = C.beta(nm, fit, ci)
                Rv, Ro = C.resp(Bt, an, l, a), C.resp_o(Bt, an, l, a)
                sc = float(Ro[none].std()) if none.sum() > 20 else float(Ro[viol].std())
                rr.append(float(Rv[viol].mean()) / max(sc, 1e-12))
                dd.append(float(T.dR(Bt, l)[bal].mean()) if T is not None else np.nan)
            A.plot(C.offs, rr, "-o", ms=2.4, lw=1.2, color=COLOR.get(nm, "k"), label=nm)
            Bx.plot(C.offs, dd, "-o", ms=2.4, lw=1.2, color=COLOR.get(nm, "k"))
            if alt is not None and nm in alt.arms:
                A.plot(alt.offs, [resp_readouts(alt, alt.beta(nm, fit, ci), an, l, a,
                                                None, None)["R / sd (own units)"]
                                  for ci in range(len(alt.offs))], "--", lw=0.9,
                       color=COLOR.get(nm, "k"), alpha=0.75)
        _xax(A, C)
        Bx.axhline(0, color="k", lw=0.7, ls="--")
        _xax(Bx, C)
        A.set_title(f"level {l}", fontsize=9)
    ax[0, 0].set_ylabel("response R / sd (own units)")
    ax[1, 0].set_ylabel("twin contrast dR (token-balanced)")
    ax[0, -1].legend(fontsize=6)
    fig.suptitle(f"{C.tag} step {C.step} — the event readouts along the sequence, "
                 f"fitter `{fit}`", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def fig_shape(C, fit, an, path, l=3, a=0, ref="full"):
    plt = _plt()
    R = C.R[an]
    x = C.vpre(C.sbeta(ref), an, l, a)
    arms = [x2 for x2 in SWITCH_ARMS if x2 in C.arms]
    fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.4))
    for nm in arms:
        sl, ic, r2 = [], [], []
        for ci in range(len(C.offs)):
            b, c, rr = _ols(x, C.vpre(C.beta(nm, fit, ci), an, l, a))
            sl.append(b); ic.append(c); r2.append(rr)
        for A, y in zip(ax, (sl, ic, r2)):
            A.plot(C.offs, y, "-o", ms=2.4, lw=1.2, color=COLOR.get(nm, "k"), label=nm)
    for A, t, h in zip(ax, ("slope", "intercept", "R2"), (1.0, 0.0, 1.0)):
        A.axhline(h, color="k", lw=0.7, ls="--")
        _xax(A, C)
        A.set_title(t, fontsize=9)
    ax[-1].legend(fontsize=6)
    fig.suptitle(f"{C.tag} step {C.step} — shift or rescaling: `V_pre ~ V_pre(static "
                 f"{ref})`, level {l}, fitter `{fit}`", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--tags", default="a1,swap65k")
    ap.add_argument("--steps", default="64000")
    ap.add_argument("--out", default="rhm/logit_reading/orbitofrontal/adaptation/results")
    ap.add_argument("--figs", default="rhm/logit_reading/orbitofrontal/adaptation/figs")
    ap.add_argument("--primary", default="win200")
    ap.add_argument("--ord2", default="ord2",
                    help="suffix of the second order seed's cells; '' disables")
    ap.add_argument("--caliper", type=float, default=0.30)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.figs, exist_ok=True)

    doc = ["# adaptation — tables", "",
           "Facts only. Generated by `analyze.py`; every arm is read on identical held-out "
           "rows at every checkpoint, and the matched row sets never see an arm or a "
           "checkpoint.", "",
           "**Reproduction.**  Run 2026-09-17 on the `chromatic` Modal workspace, two L4 "
           "containers fanned out by a CPU coordinator (79 s / 165 s wall, peak RSS "
           "5.6 / 7.3 GB, app `ap-vD3PdhLEWEqYxrLSvdFk4K`).", "",
           "```bash",
           "cd experiments            # MODAL_PROFILE=chromatic",
           "modal run --detach -m rhm.logit_reading.orbitofrontal.adaptation.task"
           "::adapt_sweep --block-win 25",
           "# artefacts: /data/v16_s2_L6_m4_distinct/logit_reading/traj_a1_s42/"
           "step064000_adapt_{a1,swap65k}.{json,npz}",
           "python -m rhm.logit_reading.orbitofrontal.adaptation.analyze <dir>/traj_a1_s42 \\",
           "    --tags a1,swap65k --out rhm/logit_reading/orbitofrontal/adaptation/results \\",
           "    --figs rhm/logit_reading/orbitofrontal/adaptation/figs",
           "```", "",
           "**Objects.**  `V[l, d](s_t)` is the outcome-trained critic: a ridge readout of "
           "the frozen state predicting `o_l(t + d)`, the actor being RIGHT at level `l` at "
           "position `t + d`, so higher `V` = a better world and a drop in `V` is bad news. "
           "`V_pre = V[l, a+1](s_(t-1))` is the norm: what the critic expected about the "
           "query's outcome one token BEFORE the event. `R = V[l, a](s_t) - V[l, a+1]"
           "(s_(t-1))` is the revision at the event, both terms predicting the same "
           "outcome. `Ro` is `R` recomputed from the unedited stream's states at the same "
           "positions. `d = outcome - V_pre` is the outcome surprise. `dR = R(violator) - "
           "R(legal twin)` is the same-prefix contrast, in which `V_pre` cancels exactly.",
           "",
           "**What is new here.**  A diet is no longer one accumulation over a mask but a "
           "row ORDER over it, and the critic is refit along the sequence. An arm is a "
           "phase of one tercile followed by a phase of another; the readouts below are "
           "indexed by the number of windows from the switch.", ""]

    for tag in args.tags.split(","):
        for step in [int(x) for x in args.steps.split(",")]:
            jp = os.path.join(args.dir, f"step{step:06d}_adapt_{tag}.json")
            if not os.path.exists(jp):
                print(f"skip {jp}")
                continue
            C = Cell(args.dir, tag, step, caliper=args.caliper)
            alt = None
            if args.ord2 and os.path.exists(os.path.join(
                    args.dir, f"step{step:06d}_adapt_{tag}_{args.ord2}.json")):
                alt = Cell(args.dir, f"{tag}_{args.ord2}", step, caliper=args.caliper)
                assert alt.offs == C.offs, "the two order seeds have different grids"
            prim = args.primary if args.primary in C.fits else C.fits[0]
            doc += [f"## {tag}, step {step}", "", sec_gate(C), sec_arms(C)]
            if alt is not None:
                doc += ["### 9. Across order seeds", "",
                        sec_seeds(C, alt, "tv" if "tv" in C.R else list(C.R)[0])]
            for an in C.R:
                doc += [f"### 2. The norm's time course — anchor `{an}`", ""]
                for fit in C.fits:
                    doc.append(sec_norm_course(C, fit, an, alt=alt))
                doc += [f"### 3. How long the norm keeps the mark of its old world — "
                        f"anchor `{an}`", ""]
                for l in LS:
                    doc.append(sec_lag(C, an, l=l, alt=alt))
                doc += [f"### 3b. The gap closing, against the fitter's own memory — "
                        f"anchor `{an}`", "", sec_recal(C, an, alt=alt),
                        f"### 4. Shift or rescaling — anchor `{an}`", ""]
                for fit in [f for f in (prim, "fgt200", "cum") if f in C.fits]:
                    doc.append(sec_shape(C, fit, an))
                doc += [f"### 5. The outcome surprise on the fixed rows — anchor `{an}`", ""]
                doc.append(sec_delta_anchor(C, prim, an))
                doc += [f"### 6. The response at the event — anchor `{an}`", ""]
                for fit in (prim, "cum"):
                    if fit in C.fits:
                        doc.append(sec_response(C, fit, an, alt=alt))
            doc += ["### 7. The outcome surprise in row currency (the world-change signal)",
                    ""]
            for fit in [f for f in (prim, "fgt200", "win800", "cum") if f in C.fits]:
                doc.append(sec_delta_rows(C, fit))
            if C.tw is not None:
                doc += ["### 8. The same-prefix twins along the sequence", ""]
                for fit in (prim, "cum"):
                    if fit in C.fits:
                        doc.append(sec_twins(C, fit))

            an0 = "tv" if "tv" in C.R else list(C.R)[0]
            fig_norm(C, prim, an0, os.path.join(args.figs, f"adapt_norm_{tag}_{step}.png"),
                     alt=alt)
            fig_fitters(C, an0, os.path.join(args.figs, f"adapt_fitters_{tag}_{step}.png"))
            fig_delta(C, prim, os.path.join(args.figs, f"adapt_delta_{tag}_{step}.png"))
            fig_event(C, prim, an0, os.path.join(args.figs, f"adapt_event_{tag}_{step}.png"),
                      alt=alt)
            fig_shape(C, prim, an0, os.path.join(args.figs, f"adapt_shape_{tag}_{step}.png"))
            print(f"{tag} step {step}: tables + figures done")

    with open(os.path.join(args.out, "tables.md"), "w") as f:
        f.write("\n".join(doc) + "\n")
    print("wrote", os.path.join(args.out, "tables.md"))


if __name__ == "__main__":
    main()
