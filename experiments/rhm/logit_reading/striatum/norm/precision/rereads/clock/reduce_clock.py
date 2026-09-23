"""reduce_clock: every table in `results/tables_<date>.md` and every figure in `figs/`, from the
banked `adaptation/` cells and `clock.py`'s outputs pulled off the volume.  Facts only.

  D=/v16_s2_L6_m4_distinct/logit_reading/traj_a1_s42
  for t in a1 swap65k a1_ord2 swap65k_ord2; do
    for f in .json .npz _clock.json _clockQ.npz _clockL.npz _clockH.npz _clockX.npz; do
      modal volume get rhm-scaling-data $D/step064000_adapt_$t$f <dir>/traj_a1_s42/step064000_adapt_$t$f
  done; done
  python -m rhm.logit_reading.striatum.norm.precision.rereads.clock.reduce_clock \
      <dir>/traj_a1_s42 --out <clock>/results --figs <clock>/figs --date 20260922

Every reader is read through `adaptation/analyze.py`'s own objects and reductions, unchanged:
arm 0 (the state ridge) is `analyze.Cell` on the BANKED files (`clock.py`'s gate asserts the
refit equals them member by member), and each feature-map arm is a `Cell` whose betas come from
`_clock<F>.npz` (the banked key names) and whose `X(anchor, which)` returns that arm's features
at the banked anchor rows -- so `dev_from_mech`, `norm_gap`, `_mech`, `_cross`, `resp_gap`,
`resp_readouts`, `dnext_z`, `_dnext_return` and the twins' token balancing are the banked code
paths.  What is new here is only the side-by-side layout (readers x order seeds) and sign
tables that print every cell's sign whether or not it clears the readability floor.
"""

import argparse
import json
import os
from collections import OrderedDict

import numpy as np

from rhm.logit_reading.orbitofrontal.adaptation import analyze as AZ
from rhm.logit_reading.striatum.analyze import _auc, fmt, tbl

READERS = ["state", "Q", "L", "H"]
RLAB = {"state": "state (arm 0 = the banked ridge)", "Q": "state + log q",
        "L": "ln_f(state)", "H": "state + H(q)"}
STEP = 64000
LS = AZ.LS
OUT_SW = ["out_lo2mid", "out_hi2mid"]
SW4 = ["out_lo2mid", "out_hi2mid", "dmg_lo2mid", "dmg_hi2mid"]
CTLS = ["out_mid2mid", "dmg_mid2mid", "out_lo_mix", "out_hi_mix"]
GAP_MIN_V, GAP_MIN_R, Z_MIN = AZ.GAP_MIN_V, AZ.GAP_MIN_R, AZ.Z_MIN

# the banked numbers this reduction's arm-0 columns must reproduce (adaptation/results/tables.md
# section 9, and orbitofrontal/README.md section 1's headline row)
BANKED_COUNTS = {
    "a1": {"dev": (52, 29), "resp": (29, 29, 29, 29), "dnext": (38, 38, 36, 35),
           "ctl": (160, 20, 3.3)},
    "swap65k": {"dev": (52, 26), "resp": (27, 26, 27, 27), "dnext": (31, 31, 29, 29),
                "ctl": (160, 7, 2.6)}}
BANKED_HEADLINE = {("swap65k", 3): {0: 0.215, 25: 0.186, 50: 0.161, 100: 0.113, 150: 0.054,
                                    200: 0.004}}


def ln_np(X, w, b, eps):
    X = np.asarray(X, np.float64)
    mu = X.mean(-1, keepdims=True)
    var = ((X - mu) ** 2).mean(-1, keepdims=True)
    return (X - mu) / np.sqrt(var + eps) * np.asarray(w, np.float64) + np.asarray(b, np.float64)


def sgn(x):
    return "--" if not np.isfinite(x) else ("+" if x > 0 else ("-" if x < 0 else "0"))


def _r(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or a[ok].std() == 0 or b[ok].std() == 0:
        return float("nan")
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


# ---------------------------------------------------------------------------
# a feature-map reader as an `analyze.Cell`
# ---------------------------------------------------------------------------

class TB(np.ndarray):
    """A beta column block that remembers which (member, fitter, checkpoint) it came from."""


class LRU(OrderedDict):
    """A bounded cache: the reductions walk the checkpoints of one (arm, fitter, level) at a
    time, so a few dozen entries hold every hit; unbounded, swap65k's run out of memory."""

    def __init__(self, cap=48):
        super().__init__()
        self.cap = cap

    def get_or(self, k, fn):
        if k in self:
            self.move_to_end(k)
            return self[k]
        v = fn()
        self[k] = v
        if len(self) > self.cap:
            self.popitem(last=False)
        return v


class BCell(AZ.Cell):
    """`analyze.Cell` with its readouts vectorised over checkpoints: `apply` on an online beta
    computes the readout for every checkpoint of that (arm, fitter) in one product and caches
    it.  Same arithmetic as `analyze.Cell.apply` up to BLAS summation order."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._v = LRU()
        if self.tw is not None:
            self.tw = TwinsFast(self.tw, self)

    def beta(self, arm, fit, ci):
        fi = self.fits.index(fit)
        b = self.z(f"beta_{arm}")[fi, ci].view(TB)
        b.src = (f"beta_{arm}", fi, ci)
        return b

    def apply(self, B, which, an, l, dd):
        src = getattr(B, "src", None)
        if src is None:
            return super().apply(np.asarray(B), which, an, l, dd)
        key, fi, ci = src

        def f():
            Bf = self.z(key)[fi][:, :, self.ti(l, dd)].astype(np.float64)       # (n_ck, D)
            return self.X(an, which) @ Bf[:, :-1].T + Bf[:, -1]
        return self._v.get_or((key, fi, which, an, l, dd), f)[:, ci]


class TwinsFast(AZ.TwinsX):
    def __init__(self, bt, owner):
        self.__dict__.update(bt.__dict__)
        self.owner = owner
        self._v = LRU()

    def dR(self, B, l):
        src = getattr(B, "src", None)
        if src is None:
            return super().dR(np.asarray(B), l)
        key, fi, ci = src

        def f():
            Bf = self.owner.z(key)[fi][:, :, self.names.index(f"l{l}_d0")].astype(np.float64)
            return (self.Xv - self.Xc) @ Bf[:, :-1].T
        return self._v.get_or((key, fi, l), f)[:, ci]


class FCell(BCell):
    """Arm `fm` of one cell: the base cell's rows, twins, grid and labels; this arm's betas,
    diagnostics, static critics and penalties; this arm's features at every saved state."""

    def __init__(self, base, d, fm):
        for k, v in base.__dict__.items():
            if k not in ("Z", "_c", "tw", "r"):
                setattr(self, k, v)
        self.base, self.fm = base, fm
        stem = os.path.join(d, f"step{base.step:06d}_adapt_{base.tag}")
        self.cj = json.load(open(stem + "_clock.json"))
        self.Z = np.load(stem + f"_clock{fm}.npz")
        self.XF = np.load(stem + "_clockX.npz")
        self._c = {}
        self._v = LRU()
        r = dict(base.r)
        r["static"] = self.cj["static"][fm]
        r["fitter_lam"] = self.cj["fitter_lam"][fm]
        self.r = r
        assert list(self.cj["ckpt_offsets"]) == list(base.offs)
        assert self.cj["arm_order"] == base.arms
        self.lnp = (np.asarray(self.XF["lnf_w"]), np.asarray(self.XF["lnf_b"]),
                    float(self.XF["lnf_eps"]))
        for an in self.R:
            assert np.array_equal(np.asarray(self.XF[f"{an}__w"]), self.R[an].g("w")), an
        self.tw = TwinsF(base.tw, self) if base.tw is not None else None

    def feat(self, X0, lq, h):
        X0 = np.asarray(X0, np.float64)
        if self.fm == "Q":
            return np.concatenate([X0, np.asarray(lq, np.float64)], 1)
        if self.fm == "H":
            return np.concatenate([X0, np.asarray(h, np.float64)[:, None]], 1)
        return ln_np(X0, *self.lnp)

    def X(self, an, which):
        k = f"{an}__F_{which}"
        if k not in self._c:
            self._c[k] = self.feat(self.base.X(an, which), self.XF[f"{an}__lq_{which}"],
                                   self.XF[f"{an}__h_{which}"])
        return self._c[k]


class TwinsF(TwinsFast):
    def __init__(self, bt, fc):
        self.__dict__.update(bt.__dict__)
        self.owner = fc
        self._v = LRU()
        XF = fc.XF
        assert np.array_equal(np.asarray(XF["tw__w"]), np.asarray(bt.Z["tw__w"]))
        self.Xp = fc.feat(bt.Xp, XF["tw__lq_pre"], XF["tw__h_pre"])
        self.Xv = fc.feat(bt.Xv, XF["tw__lq_v"], XF["tw__h_v"])
        self.Xc = fc.feat(bt.Xc, XF["tw__lq_c"], XF["tw__h_c"])


def load_venue(d, v, ord2="ord2", caliper=0.30):
    out = {}
    for s, tag in ((1, v), (2, f"{v}_{ord2}")):
        base = BCell(d, tag, STEP, caliper=caliper)
        out[("state", s)] = base
        for fm in READERS[1:]:
            out[(fm, s)] = FCell(base, d, fm)
    assert out[("state", 1)].offs == out[("state", 2)].offs
    return out


def an0(C):
    return "tv" if "tv" in C.R else list(C.R)[0]


def viol(C, an):
    R = C.R[an]
    return (R.et == 0) & (R.g("t_v") > 0)


def pre_i(C):
    return max(i for i, o in enumerate(C.offs) if o <= 0)


def mech_half(C, fit, frac=0.5):
    """The first post-switch checkpoint at which the fitter's own memory has replaced `frac`
    of its rows -- the mechanical null's half-time ON THE SAME GRID as the measured one."""
    for o in C.offs:
        if o > 0 and AZ._mech(fit, C, o) >= frac:
            return o
    return None


def s_(x):
    return "--" if x is None else str(x)


# ---------------------------------------------------------------------------
# 0. integrity, the gates, the reduction's own reproduction
# ---------------------------------------------------------------------------

def sec_integrity(d, venues, ord2):
    rows = []
    for v in venues:
        for tag in (v, f"{v}_{ord2}"):
            for sfx in ("", "_clockQ", "_clockL", "_clockH", "_clockX"):
                p = os.path.join(d, f"step{STEP:06d}_adapt_{tag}{sfx}.npz")
                if not os.path.exists(p):
                    rows.append([os.path.basename(p), "MISSING", "--", "--"])
                    continue
                z = np.load(p)
                n_ok, bad = 0, []
                for k in z.files:
                    try:
                        a = np.asarray(z[k])
                        if a.dtype.kind in "fiub":
                            float(np.nansum(a.astype(np.float64)))
                        n_ok += 1
                    except Exception as e:                       # noqa: BLE001
                        bad.append(k)
                rows.append([os.path.basename(p), f"{os.path.getsize(p) / 1e6:.1f}",
                             f"{n_ok}/{len(z.files)}", ",".join(bad[:3]) or "none"])
    return ["### 0a. Integrity of every fetched file", "",
            "Every member of every `.npz` decompressed and summed (an earlier round's fetch "
            "returned a full-size file with a bad CRC on one member that `np.load` alone did "
            "not catch).", "", tbl(rows, ["file", "MB", "members read", "failures"]), ""]


def sec_gates(cells, venues, ord2):
    out = ["### 0b. The gate: arm 0 against the banked cells", "",
           "`clock.py` refits arm 0 (the ridge on the state) on the banked code path with the "
           "banked seeds and compares every member it would save against the banked `.npz`, "
           "and the json's actor accuracies, static fits, penalties, grid and arm "
           "compositions against the banked `.json`, in-container, before writing. Arm 0 is "
           "therefore not re-saved; the reduction reads it off the banked files.", ""]
    rows, jrows = [], []
    for v in venues:
        for s in (1, 2):
            fc = cells[v][("Q", s)]
            g = fc.cj["gate"]
            gr = g.get("groups", {})
            rows.append([fc.tag, os.path.basename(g["bank"]), str(g.get("n_compared")),
                         str(g.get("n_bit_identical"))]
                        + [f"{gr.get(k, {}).get('n_identical', '--')}/{gr.get(k, {}).get('n', '--')}"
                           f" ({gr.get(k, {}).get('max_abs', float('nan')):.1e})"
                           for k in ("beta_static", "beta_online", "diag", "anchors", "twins")]
                        + [", ".join(g.get("fails", [])) or "none"])
            j = g.get("json", {})
            jrows.append([fc.tag, f"{j.get('actor_clean_acc', float('nan')):.1e}",
                          f"{j.get('static_val_r2', float('nan')):.1e}",
                          str(j.get("fitter_lam_equal")), str(j.get("ckpt_offsets_equal")),
                          str(j.get("phase_windows_equal")),
                          f"{j.get('arms_E_outcome', float('nan')):.1e}",
                          str(j.get("twin_n_pairs_equal")), str(j.get("diet_n_windows_equal")),
                          f"{fc.cj.get('wall_s', float('nan')):.0f}",
                          f"{fc.cj.get('peak_rss_gb', float('nan')):.1f}"])
    out += [tbl(rows, ["cell", "banked file", "members compared", "bit-identical",
                       "static betas (max abs)", "online betas", "diagnostics",
                       "anchor columns", "twin arrays", "gate fails"]), "",
            tbl(jrows, ["cell", "actor acc max diff", "static R2 max diff", "penalties equal",
                        "grid equal", "phase length equal", "arm E[o] max diff",
                        "twin pairs equal", "diet sizes equal", "wall s", "peak RSS GB"]), ""]
    return "\n".join(out)


def sec_checks(cells, venues):
    rows = []
    for v in venues:
        for s in (1, 2):
            for fm in READERS[1:]:
                fc = cells[v][(fm, s)]
                ck = fc.cj["checks"]
                for an in fc.R:
                    m = viol(fc, an)
                    B = fc.sbeta("full")
                    vp = float(fc.vpre(B, an, 1, 0)[m].mean())
                    rr = float(fc.resp(B, an, 1, 0)[m].mean())
                    rows.append([fc.tag, fm, an,
                                 f"{ck.get(f'{an}/ln_numpy_vs_torch_max_abs', float('nan')):.1e}",
                                 f"{ck.get(f'{an}/H_from_lq_vs_H_max_abs', float('nan')):.1e}",
                                 str(ck.get(f"{an}/H_pre_equals_h_b")),
                                 fmt(ck[f"{an}/{fm}/full_l1_Vpre_mean_viol"], 6), fmt(vp, 6),
                                 f"{abs(vp - ck[f'{an}/{fm}/full_l1_Vpre_mean_viol']):.1e}",
                                 f"{abs(rr - ck[f'{an}/{fm}/full_l1_R_mean_viol']):.1e}"])
    return "\n".join(["### 0c. The reduction's feature assembly against the container", "",
                      "Each feature-map arm's static `full` critic applied in-container to the "
                      "anchor's violation rows (l = 1, a = 0) beside the same quantity "
                      "assembled here from the saved states and the saved `log q` / `H(q)` / "
                      "`ln_f` weights; the numpy layer norm against torch's; `H(q)` "
                      "recomputed from the saved `log q` against the one the fit used; and "
                      "the anchor's banked `H_pre` against the saved `H(q)` at the "
                      "pre-event state.", "",
                      tbl(rows, ["cell", "arm", "anchor", "ln numpy vs torch", "H(log q) vs H",
                                 "H_pre == h_b", "container V_pre", "here V_pre",
                                 "|diff| V_pre", "|diff| R"]), ""])


# ---------------------------------------------------------------------------
# 1. the readers
# ---------------------------------------------------------------------------

def sec_readers(cells, v):
    C1 = cells[("state", 1)]
    an = an0(C1)
    m = viol(C1, an)
    rows = []
    same = {}
    for rd in READERS:
        C, C2 = cells[(rd, 1)], cells[(rd, 2)]
        same[rd] = all(np.array_equal(C.sbeta(nm), C2.sbeta(nm))
                       for nm in AZ.STATIC_ORDER if f"beta_static_{nm}" in C.Z)
        for nm in ("full", "out_lo", "out_mid", "out_hi", "dmg_lo", "dmg_mid", "dmg_hi"):
            st = C.r["static"][nm]
            rows.append([rd, nm] + [fmt(st["val_r2"][f"l{l}_d0"], 4) for l in LS]
                        + [fmt(float(C.vpre(C.sbeta(nm), an, l)[m].mean()), 4) for l in LS]
                        + [fmt(st["lam"]["l1_d0"], 6), fmt(st["lam"]["l3_d0"], 6)])
    out = [f"**{v}: each reader's static critics** (refit on the same diets, same ladder, "
           f"same held-out per-target selection): held-out `R2` of `V[l, 0]` on the fixed "
           f"validation rows, and mean `V_pre` on the {int(m.sum())} fixed violation rows at "
           f"anchor `{an}`. The static critics do not depend on the order seed; seed 2's "
           f"files hold identical static betas: "
           + ", ".join(f"{rd} {same[rd]}" for rd in READERS) + ".", "",
           tbl(rows, ["reader", "diet"] + [f"R2 l={l}" for l in LS]
               + [f"V_pre l={l}" for l in LS] + ["lam l1_d0", "lam l3_d0"]), ""]
    rows = []
    ci0 = C1.offs.index(0)
    for rd in READERS:
        for fit in C1.fits:
            cells_ = [rd, fit]
            for s in (1, 2):
                C = cells[(rd, s)]
                cells_ += [fmt(float(C.diag("out_mid2mid", "val_r2", fit)[ci0, C.ti(l, 0)]), 4)
                           for l in LS]
            cells_ += [fmt(cells[(rd, 1)].r["fitter_lam"][fit][f"l{l}_d0"], 6) for l in (1, 3)]
            rows.append(cells_)
    out += [f"**{v}: the online fits at the switch** (`out_mid2mid` at the end of phase A, "
            f"where each reader's penalty is selected): held-out `R2` of `V[l, 0]`, both "
            f"order seeds, and the selected penalty.", "",
            tbl(rows, ["reader", "fitter"] + [f"R2 l={l} s1" for l in LS]
                + [f"R2 l={l} s2" for l in LS] + ["lam l1_d0", "lam l3_d0"]), ""]
    return "\n".join(out)


# ---------------------------------------------------------------------------
# 2. the headline: from-low minus from-high on identical rows
# ---------------------------------------------------------------------------

def dgap(C, fit, an, l, a=0):
    """`d(from low) - d(from high)` on the fixed violation rows with the identical outcome,
    i.e. `V_pre(out_hi2mid) - V_pre(out_lo2mid)`."""
    R = C.R[an]
    sel = viol(C, an) & R.g(f"ok_l{l}_a{a}").astype(bool)
    return np.array([float((C.vpre(C.beta("out_hi2mid", fit, ci), an, l, a)
                            - C.vpre(C.beta("out_lo2mid", fit, ci), an, l, a))[sel].mean())
                     for ci in range(len(C.offs))])


def ctl_band(C, fit, an, l, a=0):
    m = viol(C, an)
    y = [float(C.vpre(C.beta("out_mid2mid", fit, ci), an, l, a)[m].mean())
         for ci in range(len(C.offs))]
    return float(np.std(y, ddof=1))


def sec_headline(cells, v, fits=("win200",)):
    C1 = cells[("state", 1)]
    an = an0(C1)
    out = []
    for fit in fits:
        for l in LS:
            G = {(rd, s): dgap(cells[(rd, s)], fit, an, l) for rd in READERS for s in (1, 2)}
            i0 = C1.offs.index(0)
            bk = BANKED_HEADLINE.get((v, l)) if fit == "win200" else None
            rows = []
            for ci, o in enumerate(C1.offs):
                if o < 0:
                    continue
                mech = AZ._mech(fit, C1, o)
                cells_ = [str(o), fmt(mech, 3)]
                for rd in READERS:
                    for s in (1, 2):
                        cells_.append(fmt(G[(rd, s)][ci], 4))
                    cells_.append(fmt(G[(rd, 1)][i0] * (1 - mech), 4))
                if bk is not None:
                    cells_.append(fmt(bk.get(o), 3) if o in bk else "")
                rows.append(cells_)
            rows.append(["mid2mid sd over ckpts", ""]
                        + sum([[fmt(ctl_band(cells[(rd, s)], fit, an, l), 4) for s in (1, 2)]
                               + [""] for rd in READERS], [])
                        + ([""] if bk is not None else []))
            hdr = ["windows from switch", "memory replaced"]
            for rd in READERS:
                hdr += [f"{rd} s1", f"{rd} s2", f"{rd} null (s1)"]
            if bk is not None:
                hdr.append("banked README row")
            out += [f"**{v}, `{fit}`, level {l}, anchor `{an}`: `d(from low) - d(from high)` "
                    f"on the {int(viol(C1, an).sum())} fixed violation rows** (the identical "
                    f"outcome, so this is `V_pre(out_hi2mid) - V_pre(out_lo2mid)`). `null (s1)` "
                    f"is seed 1's gap at the switch times (1 - memory replaced): what a simple "
                    f"average of the rows in memory would leave. The last row is the sd over "
                    f"all checkpoints of the `out_mid2mid` control's mean `V_pre` on the same "
                    f"rows.", "", tbl(rows, hdr), ""]
    return "\n".join(out)


# ---------------------------------------------------------------------------
# 3. the gap closing, against the mechanical null
# ---------------------------------------------------------------------------

def dev_rows(cells, rd, an):
    C1, C2 = cells[(rd, 1)], cells[(rd, 2)]
    fits = [f for f in C1.fits if C1.fkind[f][0] != "cum"]
    rows, rec = [], []
    for arm in SW4:
        for fit in fits:
            for l in LS:
                d1, g1 = AZ.dev_from_mech(C1, arm, fit, an, l)
                d2, g2 = AZ.dev_from_mech(C2, arm, fit, an, l)
                ok = abs(g1) >= GAP_MIN_V and abs(g2) >= GAP_MIN_V
                ag = (sgn(d1) == sgn(d2)) if np.isfinite(d1) and np.isfinite(d2) else None
                rec.append((arm, fit, l, g1, g2, d1, d2, ok, ag))
                rows.append([arm, fit, str(l), fmt(g1, 4), fmt(g2, 4), fmt(d1, 3), fmt(d2, 3),
                             sgn(d1), sgn(d2), "yes" if ag else ("NO" if ag is False else "--"),
                             "yes" if ok else "no"])
    return rows, rec


def sec_dev(cells, v):
    C1 = cells[("state", 1)]
    an = an0(C1)
    out = [f"The deviation of each cell's closure from the mechanical null: `analyze.py`'s "
           f"`dev_from_mech`, the mean over the post-switch checkpoints out to half a phase of "
           f"(fraction of the pre-switch `V_pre` gap against the family's `mid2mid` control "
           f"closed) minus (fraction of the fitter's own memory replaced). Positive = faster "
           f"than a simple average of the rows in memory. Every cell's sign is printed for "
           f"both order seeds; `readable` is the banked floor (the gap at the switch clears "
           f"{GAP_MIN_V} in both seeds). `cum` excluded as banked (its mechanical fraction is "
           f"~0 over this range). Anchor `{an}`, {int(viol(C1, an).sum())} fixed violation "
           f"rows.", ""]
    summ = []
    for rd in READERS:
        rows, rec = dev_rows(cells, rd, an)
        out += [f"**{v}, reader `{rd}` ({RLAB[rd]})**", "",
                tbl(rows, ["arm", "fitter", "l", "gap s1", "gap s2", "deviation s1",
                           "deviation s2", "sign s1", "sign s2", "signs agree", "readable"]), ""]
        rd_ = [x for x in rec if x[7]]
        n_ok, n_ag = len(rd_), sum(1 for x in rd_ if x[8])
        n_all, n_ag_all = len(rec), sum(1 for x in rec if x[8])
        summ.append([rd, str(n_ok), str(n_ag), f"{n_ag / max(n_ok, 1):.0%}", str(n_all),
                     str(n_ag_all), fmt(_r([x[5] for x in rd_], [x[6] for x in rd_]), 2),
                     fmt(float(np.median([x[5] for x in rd_])) if rd_ else float("nan"), 3),
                     fmt(float(np.median([x[6] for x in rd_])) if rd_ else float("nan"), 3),
                     str(sum(1 for x in rd_ if x[5] > 0)), str(sum(1 for x in rd_ if x[6] > 0))])
    bk = BANKED_COUNTS.get(v, {}).get("dev")
    out += [f"**{v}: the deviation's sign across order seeds, per reader.** `r` is Pearson "
            f"across the readable cells between the two seeds' deviations; the medians and the "
            f"positive counts are over the readable cells of each seed separately."
            + (f" Banked (`adaptation/results/tables.md` section 9, state ridge): {bk[1]} of "
               f"{bk[0]} readable cells agree." if bk else ""), "",
            tbl(summ, ["reader", "readable cells", "signs agree", "share", "all cells",
                       "agree (all)", "r s1 vs s2 (readable)", "median dev s1",
                       "median dev s2", "n dev > 0 s1", "n dev > 0 s2"]), ""]
    return "\n".join(out)


def sec_halftime(cells, v):
    C1 = cells[("state", 1)]
    an = an0(C1)
    rows = []
    for arm in SW4:
        for fit in C1.fits:
            for l in LS:
                cells_ = [arm, fit, str(l), s_(mech_half(C1, fit))]
                for rd in READERS:
                    for s in (1, 2):
                        C = cells[(rd, s)]
                        cells_.append(s_(AZ._cross(C, AZ.norm_gap(C, arm, fit, an, l), 0.5)))
                rows.append(cells_)
    hdr = ["arm", "fitter", "l", "null half-time"] + [f"{rd} s{s}" for rd in READERS
                                                     for s in (1, 2)]
    return "\n".join([f"**{v}: windows to half** — the first post-switch checkpoint at which "
                      f"the `V_pre` gap against the family's `mid2mid` control has fallen to "
                      f"half its value at the switch (`analyze.py`'s `_cross`), per reader and "
                      f"order seed, beside the first checkpoint at which the fitter's own "
                      f"memory has replaced half its rows (`cum`: all rows so far, so half at "
                      f"one phase). `--` = not within the phase. Anchor `{an}`.", "",
                      tbl(rows, hdr), ""])


def sec_cum(cells, v):
    C1 = cells[("state", 1)]
    an = an0(C1)
    rows = []
    for arm in SW4:
        for l in LS:
            cells_ = [arm, str(l), fmt(AZ._mech("cum", C1, C1.offs[-1]), 3)]
            for rd in READERS:
                for s in (1, 2):
                    C = cells[(rd, s)]
                    g = AZ.norm_gap(C, arm, "cum", an, l)
                    g0 = g[pre_i(C)]
                    cells_.append(fmt(1 - g[-1] / g0, 3) if abs(g0) > 1e-9 else "--")
            rows.append(cells_)
    return "\n".join([f"**{v}: the cumulative fit, which cannot forget** — fraction of the "
                      f"pre-switch gap closed by the end of phase B, beside the fraction of "
                      f"its rows phase B holds by then.", "",
                      tbl(rows, ["arm", "l", "null at end"] + [f"{rd} s{s}" for rd in READERS
                                                               for s in (1, 2)]), ""])


def sec_recal_side(cells, v):
    """The per-checkpoint closure, readers and seeds side by side (appendix)."""
    C1 = cells[("state", 1)]
    an = an0(C1)
    pi = pre_i(C1)
    out = []
    for arm in SW4:
        for fit in C1.fits:
            for l in LS:
                gaps = {(rd, s): AZ.norm_gap(cells[(rd, s)], arm, fit, an, l)
                        for rd in READERS for s in (1, 2)}
                rows = []
                for ci, o in enumerate(C1.offs):
                    if o < 0:
                        continue
                    cells_ = [str(o), fmt(AZ._mech(fit, C1, o), 3)]
                    for rd in READERS:
                        for s in (1, 2):
                            g = gaps[(rd, s)]
                            cells_.append(fmt(1 - g[ci] / g[pi], 3) if abs(g[pi]) > 1e-9
                                          else "--")
                    rows.append(cells_)
                rows.append(["gap at switch", ""] + [fmt(gaps[(rd, s)][pi], 4)
                                                     for rd in READERS for s in (1, 2)])
                out += [f"**{v}, `{arm}`, `{fit}`, level {l}**: fraction of the pre-switch "
                        f"`V_pre` gap against the family's `mid2mid` control closed, beside "
                        f"the fraction of the fitter's own memory replaced.", "",
                        tbl(rows, ["windows from switch", "memory replaced"]
                            + [f"{rd} s{s}" for rd in READERS for s in (1, 2)]), ""]
    return "\n".join(out)


# ---------------------------------------------------------------------------
# 4. shift and rescaling
# ---------------------------------------------------------------------------

def _slope_int(C, x, arm, fit, an, l, a=0):
    return np.array([AZ._ols(x, C.vpre(C.beta(arm, fit, ci), an, l, a))[:2]
                     for ci in range(len(C.offs))])


def lockstep(C, arm, fit, an, l, a=0, ref="full"):
    """Slope and intercept of the online `V_pre` on the reader's static `ref` critic's, per
    checkpoint, and the largest post-switch difference between the fractions of the way the
    two have moved, under three definitions of "the way":
      own  from the arm's own value at the switch to its own value at the end of phase B;
      ctl  the arm-minus-control difference closing from its value at the switch (the norm
           gap's own framing: `1 - g(o) / g(0)`), over the post-switch checkpoints;
      stat from the static critic of the phase-A diet to that of the phase-B diet."""
    x = C.vpre(C.sbeta(ref), an, l, a)
    bc = _slope_int(C, x, arm, fit, an, l, a)
    bk = _slope_int(C, x, arm.split("_")[0] + "_mid2mid", fit, an, l, a)
    i0 = C.offs.index(0)
    post = [i for i, o in enumerate(C.offs) if o > 0]
    db, dc = bc[-1, 0] - bc[i0, 0], bc[-1, 1] - bc[i0, 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        fo = (bc[post] - bc[i0]) / (bc[-1] - bc[i0])
        g = bc - bk
        fk = 1 - g[post] / g[i0]
        sa = np.array(AZ._ols(x, C.vpre(C.sbeta(C.r["arms"][arm]["phase_a"]), an, l, a))[:2])
        sb = np.array(AZ._ols(x, C.vpre(C.sbeta(C.r["arms"][arm]["phase_b"]), an, l, a))[:2])
        fs = (bc[post] - sa) / (sb - sa)
    mx = [float(np.nanmax(np.abs(f[:, 0] - f[:, 1]))) for f in (fo, fk, fs)]
    return float(db), float(dc), mx[0], bc, mx[1], mx[2]


def sec_lockstep(cells, v):
    C1 = cells[("state", 1)]
    an = an0(C1)
    out = [f"Each online critic's `V_pre` regressed on the reader's OWN static `full` critic's "
           f"`V_pre` across all {len(C1.R[an].et)} fixed rows at anchor `{an}` at every "
           f"checkpoint (`analyze.py`'s `sec_shape` regression). From the switch to the end "
           f"of phase B the slope moves by `d slope` and the intercept by `d int`; at every "
           f"post-switch checkpoint each has crossed a fraction of that way, and `max gap` is "
           f"the largest difference between the two fractions over the post-switch "
           f"checkpoints (0 = the shift and the rescaling close in lockstep). A fraction is a "
           f"ratio, so read `max gap` beside the size of the two excursions.", ""]
    summ = []
    for rd in READERS:
        rows, rec = [], []
        for arm in SW4:
            for fit in C1.fits:
                for l in LS:
                    r1 = lockstep(cells[(rd, 1)], arm, fit, an, l)
                    r2 = lockstep(cells[(rd, 2)], arm, fit, an, l)
                    rec.append((r1, r2))
                    rows.append([arm, fit, str(l), fmt(r1[0], 4), fmt(r1[1], 4), fmt(r1[2], 3),
                                 fmt(r2[0], 4), fmt(r2[1], 4), fmt(r2[2], 3)])
        out += [f"**{v}, reader `{rd}` ({RLAB[rd]})**", "",
                tbl(rows, ["arm", "fitter", "l", "d slope s1", "d int s1", "max gap s1",
                           "d slope s2", "d int s2", "max gap s2"]), ""]
        for s in (0, 1):
            gaps = np.array([x[s][2] for x in rec])
            gk = np.array([x[s][4] for x in rec])
            gs = np.array([x[s][5] for x in rec])
            big = np.array([abs(x[s][0]) >= 0.05 and abs(x[s][1]) >= 0.02 for x in rec])
            summ.append([rd, f"s{s + 1}", str(len(gaps)), str(int((gaps <= 0.05).sum())),
                         fmt(float(np.nanmedian(gaps)), 3), str(int(big.sum())),
                         str(int((gaps[big] <= 0.05).sum())) if big.any() else "0",
                         fmt(float(np.nanmedian(gaps[big])), 3) if big.any() else "--",
                         str(int((gk <= 0.05).sum())), fmt(float(np.nanmedian(gk)), 3),
                         str(int((gs <= 0.05).sum())), fmt(float(np.nanmedian(gs)), 3)])
    out += [f"**{v}: the lockstep, counted, under three definitions of the way.** `own` (the "
            f"per-cell tables above): from the arm's own value at the switch to its own value "
            f"at the end of phase B; `excursions clear` = cells whose slope moves by at least "
            f"0.05 and intercept by at least 0.02 over phase B. `ctl`: the arm-minus-`mid2mid` "
            f"difference in slope and in intercept, each closing from its value at the switch "
            f"(`1 - g(o) / g(0)`, the norm gap's own framing). `stat`: from the phase-A diet's "
            f"static critic to the phase-B diet's. `max gap` is the largest difference between "
            f"the slope's and the intercept's fractions over the post-switch checkpoints. "
            f"`orbitofrontal/README.md` section 1 states the two cross the same fraction of the "
            f"way at every checkpoint within 0.05; the code that produced that sentence is not "
            f"in `adaptation/analyze.py`, so each definition here is this reduction's.", "",
            tbl(summ, ["reader", "seed", "cells", "own: max gap <= 0.05", "own: median",
                       "excursions clear", "own, of those <= 0.05", "own, median (those)",
                       "ctl: <= 0.05", "ctl: median", "stat: <= 0.05", "stat: median"]), ""]
    return "\n".join(out)


# ---------------------------------------------------------------------------
# 5. the outcome surprise on the rows arriving next
# ---------------------------------------------------------------------------

def sec_dnext(cells, v):
    C1 = cells[("state", 1)]
    ci0 = C1.offs.index(0)
    pre = [i for i, o in enumerate(C1.offs) if o < 0]
    out = [f"`d = o_l(t) - V[l, 0](s_t)` with the critic as of a checkpoint, on the rows of the "
           f"block arriving immediately after it, divided by its own standard error across "
           f"that block's windows (`analyze.py`'s `dnext_z`). `z at switch` is the fit at the "
           f"end of phase A read on phase B's first block; `max |z| pre` is the largest "
           f"excursion over the phase-A checkpoints; `return` is the first post-switch "
           f"checkpoint back inside +-{Z_MIN:.0f} (`_dnext_return`). Every sign printed; "
           f"`both clear` = |z| >= {Z_MIN:.0f} in both seeds (the banked floor).", ""]
    summ, csumm = [], []
    for rd in READERS:
        C1r, C2r = cells[(rd, 1)], cells[(rd, 2)]
        rows, rec = [], []
        for arm in SW4:
            for fit in C1.fits:
                for l in LS:
                    z1, z2 = AZ.dnext_z(C1r, arm, fit, l), AZ.dnext_z(C2r, arm, fit, l)
                    a1, a2 = z1[ci0], z2[ci0]
                    r1, r2 = AZ._dnext_return(C1r, z1), AZ._dnext_return(C2r, z2)
                    ok = abs(a1) >= Z_MIN and abs(a2) >= Z_MIN
                    ag = sgn(a1) == sgn(a2)
                    rec.append((a1, a2, ok, ag, r1, r2, fit))
                    rows.append([arm, fit, str(l), fmt(float(np.nanmax(np.abs(z1[pre]))), 1),
                                 fmt(float(np.nanmax(np.abs(z2[pre]))), 1), fmt(a1, 1),
                                 fmt(a2, 1), sgn(a1), sgn(a2), "yes" if ag else "NO",
                                 "yes" if ok else "no", s_(r1), s_(r2)])
        out += [f"**{v}, reader `{rd}` ({RLAB[rd]})**", "",
                tbl(rows, ["arm", "fitter", "l", "max |z| pre s1", "max |z| pre s2",
                           "z at switch s1", "z at switch s2", "sign s1", "sign s2",
                           "signs agree", "both clear", "return s1", "return s2"]), ""]
        okr = [x for x in rec if x[2]]
        cumr = [x for x in okr if x[6] == "cum"]
        summ.append([rd, str(len(rec)), str(len(okr)), str(sum(1 for x in okr if x[3])),
                     fmt(_r([x[0] for x in okr], [x[1] for x in okr]), 2),
                     fmt(_r([x[0] for x in rec], [x[1] for x in rec]), 2),
                     str(sum(1 for x in okr if x[4] is not None)),
                     str(sum(1 for x in okr if x[5] is not None)),
                     f"{sum(1 for x in cumr if x[4] is None)}/{len(cumr)}",
                     f"{sum(1 for x in cumr if x[5] is None)}/{len(cumr)}"])
        crows, cz = [], []
        for arm in [x for x in CTLS if x in C1.arms]:
            for fit in C1.fits:
                for l in LS:
                    a1 = AZ.dnext_z(C1r, arm, fit, l)[ci0]
                    a2 = AZ.dnext_z(C2r, arm, fit, l)[ci0]
                    cz += [a1, a2]
                    crows.append([arm, fit, str(l), fmt(a1, 1), fmt(a2, 1), sgn(a1), sgn(a2)])
        out += [f"**{v}, reader `{rd}`: the controls at the switch checkpoint** (`mid2mid`, "
                f"whose world does not change, and the interleaved arms, the identical rows "
                f"with the sequence removed).", "",
                tbl(crows, ["control", "fitter", "l", "z s1", "z s2", "sign s1", "sign s2"]), ""]
        cz = np.array([x for x in cz if np.isfinite(x)])
        csumm.append([rd, str(len(cz)), str(int((np.abs(cz) >= Z_MIN).sum())),
                      fmt(float(np.abs(cz).max()), 1)])
    bk = BANKED_COUNTS.get(v, {})
    out += [f"**{v}: the world-change excursion across order seeds, per reader.** `r` is "
            f"Pearson between the seeds' z at the switch, over the cells that clear in both and "
            f"over all 80 cells. `return` counts are over the cells that clear in both; `cum "
            f"never returns` counts the `cum` cells among those with no return within the phase."
            + (f" Banked (state ridge): {bk['dnext'][1]} of {bk['dnext'][0]} agree; return "
               f"{bk['dnext'][2]} (s1) / {bk['dnext'][3]} (s2)." if "dnext" in bk else ""), "",
            tbl(summ, ["reader", "cells", "both clear", "signs agree (of those)",
                       "r (clear)", "r (all)", "return s1", "return s2",
                       "cum never returns s1", "cum never returns s2"]), "",
            f"**{v}: the controls, counted** (both seeds pooled)."
            + (f" Banked (state ridge): {bk['ctl'][1]} of {bk['ctl'][0]}, largest "
               f"{bk['ctl'][2]}." if "ctl" in bk else ""), "",
            tbl(csumm, ["reader", "control cells", f"|z| >= {Z_MIN:.0f}", "largest |z|"]), ""]
    return "\n".join(out)


def sec_cum_end(cells, v):
    C1 = cells[("state", 1)]
    ci0 = C1.offs.index(0)
    last = max(i for i in range(len(C1.offs))
               if np.isfinite(AZ.dnext_z(C1, "out_lo2mid", "cum", 1)[i]))
    rows = []
    for arm in SW4:
        for l in LS:
            cells_ = [arm, str(l)]
            for rd in READERS:
                for s in (1, 2):
                    z = AZ.dnext_z(cells[(rd, s)], arm, "cum", l)
                    cells_.append(f"{fmt(z[ci0], 1)} / {fmt(z[last], 1)}")
            rows.append(cells_)
    return "\n".join([f"**{v}: the cumulative fit's excursion, at the switch / at the last "
                      f"checkpoint that has a next block ({C1.offs[last]} windows after the "
                      f"switch)**, z of `d` on the rows arriving next, per reader and order "
                      f"seed.", "", tbl(rows, ["arm", "l"] + [f"{rd} s{s}" for rd in READERS
                                                             for s in (1, 2)]), ""])


def sec_dnext_course(cells, v, fit="win200", l=3):
    C1 = cells[("state", 1)]
    arms = ["out_lo2mid", "out_hi2mid", "out_mid2mid", "out_lo_mix", "out_hi_mix"]
    out = []
    for rd in READERS:
        Z = {(a, s): AZ.dnext_z(cells[(rd, s)], a, fit, l) for a in arms for s in (1, 2)}
        rows = [[str(o)] + [fmt(Z[(a, s)][ci], 1) for a in arms for s in (1, 2)]
                for ci, o in enumerate(C1.offs)]
        out += [f"**{v}, reader `{rd}`, `{fit}`, level {l}: z of `d` on the rows arriving next, "
                f"along the sequence.**", "",
                tbl(rows, ["windows from switch"] + [f"{a} s{s}" for a in arms
                                                     for s in (1, 2)]), ""]
    return "\n".join(out)


# ---------------------------------------------------------------------------
# 6. the response carrying the history
# ---------------------------------------------------------------------------

def resp_gap_matched(C, arm, fit, an, l, a=0):
    R = C.R[an]
    got = R.matched("flip", l, a, "kjt")
    if got is None:
        return None, 0
    mm = got[0]
    vi = viol(C, an)
    none = R.et == 2
    ctl = arm.split("_")[0] + "_mid2mid"

    def rsd(B):
        Rv, Ro = C.resp(B, an, l, a), C.resp_o(B, an, l, a)
        sc = float(Ro[none].std()) if none.sum() > 20 else float(Ro[vi].std())
        return float(Rv[mm].mean()) / max(sc, 1e-12)
    return np.array([rsd(C.beta(arm, fit, ci)) - rsd(C.beta(ctl, fit, ci))
                     for ci in range(len(C.offs))]), len(mm)


def sec_resp(cells, v):
    C1 = cells[("state", 1)]
    an = an0(C1)
    pi = pre_i(C1)
    fits = [f for f in C1.fits if C1.fkind[f][0] != "cum"]
    out = [f"The anticipatory response in each critic's own units (`R / sd`, `analyze.py`'s "
           f"`resp_readouts`), minus its family's `mid2mid` control's, on the identical fixed "
           f"violation rows at anchor `{an}` (`resp_gap`), and the same on the rows of the "
           f"damage-matched set (`Rows.matched('flip', l, 0, 'kjt')`, the arc's matcher, which "
           f"never sees an arm or a checkpoint). `gap` is at the last pre-switch checkpoint; "
           f"`half` is the first post-switch checkpoint at which it has fallen to half "
           f"(`_cross`), beside the norm's own half-time in the same cell and the fitter's "
           f"mechanical one. Every sign printed; `readable` = |gap| >= {GAP_MIN_R} in both "
           f"seeds (the banked floor).", ""]
    summ = []
    for rd in READERS:
        C1r, C2r = cells[(rd, 1)], cells[(rd, 2)]
        rows, rec = [], []
        for arm in SW4:
            for fit in fits:
                for l in LS:
                    q1, q2 = AZ.resp_gap(C1r, arm, fit, an, l), AZ.resp_gap(C2r, arm, fit, an, l)
                    m1, nm_ = resp_gap_matched(C1r, arm, fit, an, l)
                    m2, _ = resp_gap_matched(C2r, arm, fit, an, l)
                    h1, h2 = AZ._cross(C1r, q1, 0.5), AZ._cross(C2r, q2, 0.5)
                    n1 = AZ._cross(C1r, AZ.norm_gap(C1r, arm, fit, an, l), 0.5)
                    n2 = AZ._cross(C2r, AZ.norm_gap(C2r, arm, fit, an, l), 0.5)
                    ok = abs(q1[pi]) >= GAP_MIN_R and abs(q2[pi]) >= GAP_MIN_R
                    ag = sgn(q1[pi]) == sgn(q2[pi])
                    agm = (m1 is not None and m2 is not None and sgn(m1[pi]) == sgn(m2[pi]))
                    rec.append((q1[pi], q2[pi], ok, ag, h1, h2,
                                None if m1 is None else m1[pi],
                                None if m2 is None else m2[pi], agm))
                    rows.append([arm, fit, str(l), fmt(q1[pi], 3), fmt(q2[pi], 3), sgn(q1[pi]),
                                 sgn(q2[pi]), "yes" if ag else "NO", "yes" if ok else "no",
                                 s_(h1), s_(h2), s_(n1), s_(n2), s_(mech_half(C1r, fit)),
                                 fmt(q1[-1], 3), fmt(q2[-1], 3),
                                 "--" if m1 is None else fmt(m1[pi], 3),
                                 "--" if m2 is None else fmt(m2[pi], 3),
                                 "yes" if agm else "NO", str(nm_)])
        out += [f"**{v}, reader `{rd}` ({RLAB[rd]})**", "",
                tbl(rows, ["arm", "fitter", "l", "gap s1", "gap s2", "sign s1", "sign s2",
                           "signs agree", "readable", "half s1", "half s2", "norm half s1",
                           "norm half s2", "null half", "gap at end s1", "gap at end s2",
                           "matched gap s1", "matched gap s2", "matched signs agree",
                           "n matched"]), ""]
        okr = [x for x in rec if x[2]]
        mrec = [x for x in rec if x[6] is not None and x[7] is not None]
        summ.append([rd, str(len(rec)), str(len(okr)), str(sum(1 for x in okr if x[3])),
                     str(sum(1 for x in rec if x[3])),
                     fmt(_r([x[0] for x in okr], [x[1] for x in okr]), 2),
                     str(sum(1 for x in okr if x[4] is not None)),
                     str(sum(1 for x in okr if x[5] is not None)),
                     f"{sum(1 for x in mrec if x[8])}/{len(mrec)}",
                     fmt(_r([x[6] for x in mrec], [x[7] for x in mrec]), 2)])
    rows_u = []
    for rd in READERS:
        cells_ = [rd]
        for s in (1, 2):
            C = cells[(rd, s)]
            raw, sds = [], []
            for arm in SW4:
                ctl = arm.split("_")[0] + "_mid2mid"
                for fit in fits:
                    for l in LS:
                        ra = AZ.resp_readouts(C, C.beta(arm, fit, pi), an, l, 0, None, None)
                        rc = AZ.resp_readouts(C, C.beta(ctl, fit, pi), an, l, 0, None, None)
                        raw.append(abs(ra["R (raw)"] - rc["R (raw)"]))
                        sds.append(ra["sd of R on unedited"])
            cells_ += [fmt(float(np.median(raw)), 4), fmt(float(np.median(sds)), 4)]
        sf = cells[(rd, 1)].sbeta("full")
        Rs = cells[(rd, 1)].resp(sf, an, 1)
        Ros = cells[(rd, 1)].resp_o(sf, an, 1)
        vi = viol(cells[(rd, 1)], an)
        cells_ += [fmt(float(Rs[vi].mean()), 4), fmt(float(Ros[vi].std()), 4)]
        rows_u.append(cells_)
    out += [f"**{v}: the units.** `R / sd` divides each critic's raw response by the sd of its "
            f"own `Ro` on the fixed violation rows (`resp_readouts`), so a reader whose sd is "
            f"smaller reads larger in own units. Median over the 64 cells of |raw pre-switch "
            f"gap against the control| and of the arm's sd, per seed; and the static `full` "
            f"critic's mean raw `R` and `sd(Ro)` at l = 1.", "",
            tbl(rows_u, ["reader", "median |raw gap| s1", "median sd s1", "median |raw gap| s2",
                         "median sd s2", "static full mean R (l=1)",
                         "static full sd(Ro) (l=1)"]), ""]
    bk = BANKED_COUNTS.get(v, {}).get("resp")
    out += [f"**{v}: the response's history across order seeds, per reader.** `r` is Pearson "
            f"between the seeds' pre-switch gaps over the readable cells (all-rows version) and "
            f"over every cell (matched-rows version)."
            + (f" Banked (state ridge): {bk[1]} of {bk[0]} readable agree; half within the "
               f"phase {bk[2]} (s1) / {bk[3]} (s2)." if bk else ""), "",
            tbl(summ, ["reader", "cells", "readable", "signs agree (readable)",
                       "signs agree (all)", "r (readable)", "half within phase s1",
                       "half within phase s2", "matched signs agree (all)", "matched r (all)"]),
            ""]
    return "\n".join(out)


def sec_resp_auc(cells, v, fit="win200"):
    C1 = cells[("state", 1)]
    an = an0(C1)
    idx = sorted({0, C1.offs.index(0), C1.offs.index(0) + 3, len(C1.offs) - 1})
    rows = []
    for rd in READERS:
        for l in LS:
            R = C1.R[an]
            got = R.matched("flip", l, 0, "kjt")
            if got is None:
                continue
            mm, lab = got
            for arm in ["out_lo2mid", "out_mid2mid", "out_hi2mid"]:
                cells_ = [rd, str(l), arm]
                for s in (1, 2):
                    C = cells[(rd, s)]
                    for ci in idx:
                        cells_.append(fmt(_auc(-C.resp(C.beta(arm, fit, ci), an, l)[mm],
                                               lab[mm]), 3))
                rows.append(cells_)
            C = cells[(rd, 1)]
            rows.append([rd, str(l), "static full"]
                        + [fmt(_auc(-C.resp(C.sbeta("full"), an, l)[mm], lab[mm]), 3)]
                        + [""] * (2 * len(idx) - 1))
    hdr = ["reader", "l", "critic"] + [f"s{s} @ {C1.offs[ci]}" for s in (1, 2) for ci in idx]
    return "\n".join([f"**{v}, `{fit}`: `AUC(-R, realised damage)` on the damage-matched rows "
                      f"at anchor `{an}`, along the sequence** (the banked reading: flat and "
                      f"arm-independent). Columns are checkpoints (windows from the switch) per "
                      f"order seed.", "", tbl(rows, hdr), ""])


# ---------------------------------------------------------------------------
# 7. the twins along the sequence
# ---------------------------------------------------------------------------

def sec_twins(cells, v, fit="win200"):
    C1 = cells[("state", 1)]
    if C1.tw is None:
        return f"{v}: no twins in the banked cell.\n"
    bal, resid = C1.tw.token_balanced(0)
    same = np.array_equal(np.asarray(C1.tw.Z["tw__w"]),
                          np.asarray(cells[("state", 2)].tw.Z["tw__w"]))
    arms = ["out_lo2mid", "out_mid2mid", "out_hi2mid"]
    out = [f"The same-prefix pairs (`phasic.py`'s construction), token-balanced at caliper "
           f"0.30 (`TwinsX.token_balanced(0)`): {int(bal.sum())} pairs, signed token residual "
           f"{resid}; the two order seeds hold the identical pairs: {same}. `dR = R(violator) - "
           f"R(twin)`, where `V_pre` cancels within the pair for every reader (the pair shares "
           f"the pre-event state and its `log q` to "
           f"{cells[('Q', 1)].cj['checks'].get('tw/lq_pre_identical_max_abs', float('nan')):.1e}"
           f"). Positive = more optimistic after the illegal token.", ""]
    summ = []
    for rd in READERS:
        for l in LS:
            D = {(a, s): np.array([cells[(rd, s)].tw.dR(cells[(rd, s)].beta(a, fit, ci), l)[bal]
                                   for ci in range(len(C1.offs))]) for a in arms for s in (1, 2)}
            rows = []
            for ci, o in enumerate(C1.offs):
                cells_ = [str(o)]
                for s in (1, 2):
                    cells_ += [fmt(float(D[(a, s)][ci].mean()), 5) for a in arms]
                    df = D[("out_lo2mid", s)][ci] - D[("out_hi2mid", s)][ci]
                    cells_ += [fmt(float(df.mean()), 5), fmt(AZ._sem(df), 5)]
                rows.append(cells_)
            C = cells[(rd, 1)]
            for nm in ("full", "out_lo", "out_mid", "out_hi"):
                dr = C.tw.dR(C.sbeta(nm), l)[bal]
                rows.append([f"static {nm}", fmt(float(dr.mean()), 5), fmt(AZ._sem(dr), 5)]
                            + [""] * 8)
            out += [f"**{v}, reader `{rd}`, `{fit}`, level {l}: mean `dR` on the token-balanced "
                    f"pairs, and the from-low minus from-high difference on the identical pairs "
                    f"with its sem.** The `static` rows carry the value and its sem in the "
                    f"first two columns.", "",
                    tbl(rows, ["windows from switch"]
                        + sum([[f"{a} s{s}" for a in arms] + [f"lo-hi s{s}", f"sem s{s}"]
                               for s in (1, 2)], [])), ""]
            pi = pre_i(C1)
            d1 = D[("out_lo2mid", 1)][pi] - D[("out_hi2mid", 1)][pi]
            d2 = D[("out_lo2mid", 2)][pi] - D[("out_hi2mid", 2)][pi]
            e1 = D[("out_lo2mid", 1)][-1] - D[("out_hi2mid", 1)][-1]
            e2 = D[("out_lo2mid", 2)][-1] - D[("out_hi2mid", 2)][-1]
            summ.append([rd, str(l), fmt(float(d1.mean()), 5), fmt(float(d1.mean()) / max(AZ._sem(d1), 1e-12), 1),
                         fmt(float(d2.mean()), 5), fmt(float(d2.mean()) / max(AZ._sem(d2), 1e-12), 1),
                         sgn(d1.mean()), sgn(d2.mean()),
                         "yes" if sgn(d1.mean()) == sgn(d2.mean()) else "NO",
                         fmt(float(e1.mean()), 5), fmt(float(e2.mean()), 5),
                         fmt(float(D[("out_mid2mid", 1)][pi].mean()), 5),
                         fmt(float(D[("out_mid2mid", 2)][pi].mean()), 5)])
    out += [f"**{v}, `{fit}`: the twin contrast's history, summarised** — `dR(from low) - "
            f"dR(from high)` on the identical pairs at the switch (mean and mean/sem) and at the "
            f"end of phase B, and the `mid2mid` control's `dR` at the switch, both seeds.", "",
            tbl(summ, ["reader", "l", "lo-hi at switch s1", "z s1", "lo-hi at switch s2",
                       "z s2", "sign s1", "sign s2", "signs agree", "lo-hi at end s1",
                       "lo-hi at end s2", "mid2mid dR s1", "mid2mid dR s2"]), ""]
    return "\n".join(out)


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

COL = {"out_lo2mid": "#2a78d6", "out_hi2mid": "#eb6834", "out_mid2mid": "#6b6b6b",
       "out_lo_mix": "#1baf7a", "out_hi_mix": "#eda100"}


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.6,
                         "font.size": 8})
    return plt


def _figlegend(fig, ax):
    h, lab = ax[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=len(lab), fontsize=7, frameon=False)
    fig.subplots_adjust(bottom=0.08)


def fig_gap(cells, v, path, fit="win200"):
    plt = _plt()
    C1 = cells[("state", 1)]
    an = an0(C1)
    post = [i for i, o in enumerate(C1.offs) if o >= 0]
    xs = [C1.offs[i] for i in post]
    fig, ax = plt.subplots(len(READERS), len(LS), figsize=(3.3 * len(LS), 2.5 * len(READERS)),
                           sharex=True)
    for ri, rd in enumerate(READERS):
        for li, l in enumerate(LS):
            A = ax[ri, li]
            g1, g2 = dgap(cells[(rd, 1)], fit, an, l), dgap(cells[(rd, 2)], fit, an, l)
            i0 = C1.offs.index(0)
            A.plot(xs, [g1[i0] * (1 - AZ._mech(fit, C1, o)) for o in xs], ":", color="#9a9a9a",
                   lw=1.4, label="null: gap x (1 - memory replaced)")
            A.plot(xs, g1[post], "-o", ms=3, lw=2, color="#2a78d6", label="order seed 1")
            A.plot(xs, g2[post], "--s", ms=3, lw=1.4, color="#eb6834", label="order seed 2")
            A.axhline(0, color="#444444", lw=0.6)
            A.set_xscale("symlog", linthresh=int(C1.r["config"]["block_win"]), linscale=0.6)
            if ri == 0:
                A.set_title(f"level {l}")
            if li == 0:
                A.set_ylabel(f"{RLAB[rd]}\nd(low) - d(high)")
            if ri == len(READERS) - 1:
                A.set_xlabel("windows after the switch")
    _figlegend(fig, ax)
    fig.suptitle(f"{v} — from-low minus from-high outcome surprise on identical rows, fitter "
                 f"{fit}, per reader", fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=140)
    plt.close(fig)


def fig_dev(cells, v, path):
    plt = _plt()
    an = an0(cells[("state", 1)])
    fig, ax = plt.subplots(1, len(READERS), figsize=(3.3 * len(READERS), 3.3), sharex=True,
                           sharey=True)
    for ri, rd in enumerate(READERS):
        _, rec = dev_rows(cells, rd, an)
        A = ax[ri]
        for fam, mk in (("out", "o"), ("dmg", "^")):
            r_ = [x for x in rec if x[0].startswith(fam)]
            ok = np.array([x[7] for x in r_])
            d1 = np.array([x[5] for x in r_])
            d2 = np.array([x[6] for x in r_])
            A.scatter(d1[ok], d2[ok], s=18, marker=mk, color="#2a78d6",
                      label=f"{fam}, readable")
            A.scatter(d1[~ok], d2[~ok], s=18, marker=mk, facecolors="none",
                      edgecolors="#9a9a9a", label=f"{fam}, gap < {GAP_MIN_V}")
        A.axhline(0, color="#444444", lw=0.6)
        A.axvline(0, color="#444444", lw=0.6)
        A.set_title(RLAB[rd])
        A.set_xlabel("deviation from the null, order seed 1")
        if ri == 0:
            A.set_ylabel("deviation from the null, order seed 2")
    ax[-1].legend(fontsize=6, frameon=False)
    fig.suptitle(f"{v} — closure minus the mechanical null, one point per (arm, fitter, level); "
                 f"agreement across orderings = quadrants I and III", fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=140)
    plt.close(fig)


def fig_dnext(cells, v, path, fit="win200"):
    plt = _plt()
    C1 = cells[("state", 1)]
    arms = ["out_lo2mid", "out_hi2mid", "out_mid2mid", "out_lo_mix", "out_hi_mix"]
    fig, ax = plt.subplots(len(READERS), len(LS), figsize=(3.3 * len(LS), 2.5 * len(READERS)),
                           sharex=True)
    for ri, rd in enumerate(READERS):
        for li, l in enumerate(LS):
            A = ax[ri, li]
            A.axhspan(-Z_MIN, Z_MIN, color="#eeeeee", lw=0)
            for a in arms:
                for s, ls in ((1, "-"), (2, "--")):
                    A.plot(C1.offs, AZ.dnext_z(cells[(rd, s)], a, fit, l), ls, lw=1.2 if s == 1
                           else 0.9, color=COL[a], label=a if s == 1 else None)
            A.axvline(0, color="#444444", lw=0.6)
            A.set_xscale("symlog", linthresh=int(C1.r["config"]["block_win"]), linscale=0.6)
            if ri == 0:
                A.set_title(f"level {l}")
            if li == 0:
                A.set_ylabel(f"{RLAB[rd]}\nz of d on the next block")
            if ri == len(READERS) - 1:
                A.set_xlabel("windows from the switch")
    _figlegend(fig, ax)
    fig.suptitle(f"{v} — the outcome surprise on the rows arriving next, fitter {fit} "
                 f"(solid: order seed 1, dashed: seed 2; grey band: +-{Z_MIN:.0f} sem)",
                 fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=140)
    plt.close(fig)


def fig_resp(cells, v, path, fit="win200"):
    plt = _plt()
    C1 = cells[("state", 1)]
    an = an0(C1)
    fig, ax = plt.subplots(len(READERS), len(LS), figsize=(3.3 * len(LS), 2.5 * len(READERS)),
                           sharex=True)
    for ri, rd in enumerate(READERS):
        for li, l in enumerate(LS):
            A = ax[ri, li]
            for a in OUT_SW:
                for s, ls in ((1, "-o"), (2, "--")):
                    A.plot(C1.offs, AZ.resp_gap(cells[(rd, s)], a, fit, an, l), ls, ms=2.5,
                           lw=1.3 if s == 1 else 0.9, color=COL[a],
                           label=a if s == 1 else None)
            A.axhline(0, color="#444444", lw=0.6)
            A.axvline(0, color="#444444", lw=0.6)
            A.set_xscale("symlog", linthresh=int(C1.r["config"]["block_win"]), linscale=0.6)
            if ri == 0:
                A.set_title(f"level {l}")
            if li == 0:
                A.set_ylabel(f"{RLAB[rd]}\nR/sd minus mid2mid")
            if ri == len(READERS) - 1:
                A.set_xlabel("windows from the switch")
    _figlegend(fig, ax)
    fig.suptitle(f"{v} — the response's history on identical rows, fitter {fit} (solid: "
                 f"order seed 1, dashed: seed 2)", fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=140)
    plt.close(fig)


def fig_twins(cells, v, path, fit="win200"):
    plt = _plt()
    C1 = cells[("state", 1)]
    if C1.tw is None:
        return
    bal = C1.tw.token_balanced(0)[0]
    arms = ["out_lo2mid", "out_mid2mid", "out_hi2mid"]
    fig, ax = plt.subplots(len(READERS), len(LS), figsize=(3.3 * len(LS), 2.5 * len(READERS)),
                           sharex=True)
    for ri, rd in enumerate(READERS):
        for li, l in enumerate(LS):
            A = ax[ri, li]
            for a in arms:
                for s, ls in ((1, "-o"), (2, "--")):
                    C = cells[(rd, s)]
                    A.plot(C1.offs, [float(C.tw.dR(C.beta(a, fit, ci), l)[bal].mean())
                                     for ci in range(len(C1.offs))], ls, ms=2.5,
                           lw=1.3 if s == 1 else 0.9, color=COL[a], label=a if s == 1 else None)
            A.axhline(0, color="#444444", lw=0.6)
            A.axvline(0, color="#444444", lw=0.6)
            A.set_xscale("symlog", linthresh=int(C1.r["config"]["block_win"]), linscale=0.6)
            if ri == 0:
                A.set_title(f"level {l}")
            if li == 0:
                A.set_ylabel(f"{RLAB[rd]}\ntwin dR")
            if ri == len(READERS) - 1:
                A.set_xlabel("windows from the switch")
    _figlegend(fig, ax)
    fig.suptitle(f"{v} — the token-balanced twin contrast along the sequence, fitter {fit} "
                 f"(solid: order seed 1, dashed: seed 2)", fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--venues", default="a1,swap65k")
    ap.add_argument("--ord2", default="ord2")
    ap.add_argument("--out", default="rhm/logit_reading/striatum/norm/precision/rereads/clock/results")
    ap.add_argument("--figs", default="rhm/logit_reading/striatum/norm/precision/rereads/clock/figs")
    ap.add_argument("--date", default="20260922")
    ap.add_argument("--no-appendix", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.figs, exist_ok=True)
    venues = args.venues.split(",")
    cells = {v: load_venue(args.dir, v, args.ord2) for v in venues}
    for v in venues:
        for s in (1, 2):
            fails = cells[v][("Q", s)].cj["gate"].get("fails")
            assert not fails, f"arm-0 gate failed on {v} s{s}: {fails}"
    C0 = cells[venues[0]][("state", 1)]
    cj0 = cells[venues[0]][("Q", 1)].cj
    doc = ["# clock — tables", "",
           "Facts only. Generated by `reduce_clock.py` from the banked `adaptation/` cells (arm "
           "0, the ridge on the state) and `clock.py`'s refits (arms `Q` = state + log q, `L` = "
           "ln_f(state), `H` = state + H(q)). Every reader is read on the identical fixed held-out "
           "rows and twins at every checkpoint through `orbitofrontal/adaptation/analyze.py`'s "
           "own reductions; both order seeds are printed side by side and never averaged; the "
           "mechanical null is printed beside every measured fraction; the sign tables print "
           "every cell's sign.", "",
           f"**Grid.** Venues {venues}, trajectory `traj_a1_s42`, step {STEP}. Order seed 1 = "
           f"{C0.r['config']['order_seed']}, seed 2 = "
           f"{cells[venues[0]][('state', 2)].r['config']['order_seed']}. Fitters "
           f"{C0.fits}, block {C0.r['config']['block_win']} windows. Arms {C0.arms}. Feature "
           f"arms: " + "; ".join(f"`{k}` {x}" for k, x in cj0["fmap_desc"].items()) + ".", "",
           "**Objects.** `V_pre = V[l, a+1](s_(t-1))`, the norm; `R = V[l, a](s_t) - V_pre`, the "
           "response; `d = outcome - V_pre`, the outcome surprise; for a feature-map reader "
           "`s_t` is replaced by that arm's features at the same position. `dR` is the "
           "same-prefix twin contrast.", "",
           "## 0. Integrity and gates", ""]
    doc += sec_integrity(args.dir, venues, args.ord2)
    doc += [sec_gates(cells, venues, args.ord2), sec_checks(cells, venues)]
    for v in venues:
        doc += [f"## {v}", "", "### 1. The readers", "", sec_readers(cells[v], v),
                "### 2. From-low minus from-high on identical rows", "",
                sec_headline(cells[v], v), "### 3. The gap closing, against the mechanical null",
                "", "#### 3a. Deviation from the null, sign per cell across orderings", "",
                sec_dev(cells[v], v), "#### 3b. Windows to half", "", sec_halftime(cells[v], v),
                "#### 3c. The cumulative fit", "", sec_cum(cells[v], v),
                "### 4. Shift and rescaling", "", sec_lockstep(cells[v], v),
                "### 5. The outcome surprise on the rows arriving next", "",
                sec_dnext(cells[v], v), "#### 5b. The cumulative fit at the end of the run",
                "", sec_cum_end(cells[v], v), "#### 5c. Along the sequence", "",
                sec_dnext_course(cells[v], v), "### 6. The response carrying the history", "",
                sec_resp(cells[v], v), "#### 6b. The response's damage reading", "",
                sec_resp_auc(cells[v], v), "### 7. The twins along the sequence", "",
                sec_twins(cells[v], v)]
        fig_gap(cells[v], v, os.path.join(args.figs, f"clock_gap_{v}.png"))
        fig_dev(cells[v], v, os.path.join(args.figs, f"clock_dev_{v}.png"))
        fig_dnext(cells[v], v, os.path.join(args.figs, f"clock_dnext_{v}.png"))
        fig_resp(cells[v], v, os.path.join(args.figs, f"clock_resp_{v}.png"))
        fig_twins(cells[v], v, os.path.join(args.figs, f"clock_twins_{v}.png"))
        for rd in READERS:
            C = cells[v][(rd, 1)]
            AZ.fig_norm(C, "win200", an0(C), os.path.join(args.figs, f"clock_norm_{rd}_{v}.png"),
                        alt=cells[v][(rd, 2)])
        print(f"{v}: tables + figures done", flush=True)
    if not args.no_appendix:
        doc += ["## Appendix. The per-checkpoint closure, every fitter, level and arm", ""]
        for v in venues:
            doc.append(sec_recal_side(cells[v], v))
    p = os.path.join(args.out, f"tables_{args.date}.md")
    with open(p, "w") as f:
        f.write("\n".join(doc) + "\n")
    print("wrote", p)


if __name__ == "__main__":
    main()
