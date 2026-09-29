"""precision: does the value reader's norm, response and outcome surprise depend on the
world model's UNCERTAINTY BEFORE the event, at matched surprisal?  Facts only; no
interpretation lives here.

The arc's every value-side contrast is taken at matched surprisal -- `junction/analyze.py`'s
matcher pins the model's own surprisal at the anchor at 0.500 on every banked table, and the
same-prefix twins are selected at a surprisal caliper.  Nothing has asked whether the reader
depends on `H_pre = H(q_(t0-1))`, the entropy of the trunk's own forecast at the position
BEFORE the anchor.  `norm/task.py` records that column (`rec["H_pre"] = H_e[w, t0w - 1]`,
line ~594) on every fixed row and no banked table has ever used it.  This file is the CPU
reduction that uses it.  Nothing is refit and no GPU is touched.

The music literature this is the analogue of measures pleasure on a 2-D surface of the
event's information content against the entropy of the context before it, and disputes the
SIGN of the interaction (Cheung et al. 2019 against Mas-Herrero & Marco-Pallares 2025); the
direct analogue here is the response's surface over (surprisal x pre-event entropy).

**One object, not two.**  `excess_e = nll_e - H_pre` is on file and is an exact identity
(asserted per cell), so at matched surprisal the `H_pre` axis IS the negative excess axis.
`logit_reading/README.md` 2b found `s - H(q)` a null as a scalar for detection and
`striatum/README.md` 5's addendum found the excess head carries nothing conditioned on the
value level.  Every regression below is therefore printed in BOTH parameterisations
(`R ~ s + H` and `R ~ s + excess`), which are the same fit with `b_H = -b_excess` and
`b_s(excess) = b_s(H) + b_H`; they are one finding.

  D=/v16_s2_L6_m4_distinct/logit_reading
  for s in 42 43 44; do for st in 000000 008000 064000; do for tg in a1 swap65k; do
    for e in npz json; do
      modal volume get rhm-scaling-data $D/traj_a1_s$s/step${st}_norm_${tg}.$e <dir>/traj_a1_s$s/
  done; done; done; done
  python -m rhm.logit_reading.striatum.norm.precision.reduce \
      <dir>/traj_a1_s42 <dir>/traj_a1_s43 <dir>/traj_a1_s44 \
      --out rhm/logit_reading/striatum/norm/precision/results \
      --figs rhm/logit_reading/striatum/norm/precision/figs \
      --outfile tables_20260922.md

Optional `--twin-h <dir>`: the directory holding `step*_norm_<tag>_Hpre.npz` written by
`hpre.py` (route ii, one forward pass per cell on Modal), which supplies `H_pre` for EVERY
twin pair.  Without it the twin sections fall back to route (i): the join of `tw__(w, t_v)`
against the `tv__` fixed rows, which covers the test-split pairs only (~25%).  Both are
reported and cross-checked against each other where both exist.
"""

import argparse
import json
import os

import numpy as np

from rhm.logit_reading.striatum.analyze import _auc, fmt, tbl
from rhm.logit_reading.striatum.junction.analyze import Rows
from rhm.logit_reading.striatum.norm.analyze import (Twins, order_diets, _ols, _sem,
                                                     _spear)

STEPS = [0, 8000, 64000]
LS = [1, 2, 3, 4]
ANCHORS = ["tv", "fd"]


# ---------------------------------------------------------------------------
# small statistics
# ---------------------------------------------------------------------------

def _z(x):
    x = np.asarray(x, np.float64)
    s = x.std()
    return (x - x.mean()) / (s if s > 0 else 1.0)


def _ols_multi(Y, X):
    """Least squares with an intercept appended.  Returns (coef, se, resid)."""
    X = np.asarray(X, np.float64)
    Y = np.asarray(Y, np.float64)
    A = np.concatenate([X, np.ones((len(X), 1))], 1)
    cf, *_ = np.linalg.lstsq(A, Y, rcond=None)
    res = Y - A @ cf
    n, p = A.shape
    if n <= p:
        return cf, np.full(p, np.nan), res
    s2 = float((res ** 2).sum()) / (n - p)
    try:
        cov = s2 * np.linalg.pinv(A.T @ A)
        se = np.sqrt(np.clip(np.diag(cov), 0, None))
    except Exception:
        se = np.full(p, np.nan)
    return cf, se, res


def _resid(y, C):
    """y residualised on the controls C (intercept added)."""
    A = np.concatenate([np.asarray(C, np.float64), np.ones((len(y), 1))], 1)
    cf, *_ = np.linalg.lstsq(A, np.asarray(y, np.float64), rcond=None)
    return np.asarray(y, np.float64) - A @ cf


def _partial(y, x, C):
    """slope, sem, rho of y on x after both are residualised on C."""
    ry, rx = _resid(y, C), _resid(x, C)
    b, _, _ = _ols(rx, ry)
    n = len(y)
    sd = rx.std()
    if not np.isfinite(b) or sd == 0 or n < 5:
        return float("nan"), float("nan"), float("nan")
    e = ry - b * rx
    se = float(np.sqrt((e ** 2).sum() / max(n - 2 - C.shape[1], 1)) / (sd * np.sqrt(n)))
    return float(b), se, _spear(rx, ry)


def _qbin(x, k):
    """k quantile bins of x, as integer labels 0..k-1 (ties fold into fewer bins)."""
    x = np.asarray(x, np.float64)
    q = np.quantile(x, np.linspace(0, 1, k + 1)[1:-1])
    return np.digitize(x, q)


def _grid(zv, bx, by, kx, ky, nd=4):
    """(mean, n) per (bx, by) cell as a printable table body."""
    rows = []
    for i in range(ky):
        cells = [f"H q{i + 1}"]
        for jx in range(kx):
            s = (bx == jx) & (by == i)
            cells.append(f"{zv[s].mean():.{nd}f} ({int(s.sum())})" if s.sum() >= 5
                         else f"-- ({int(s.sum())})")
        rows.append(cells)
    return rows


def _pearson(x, y):
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3 or x[m].std() == 0 or y[m].std() == 0:
        return float("nan")
    return float(np.corrcoef(x[m], y[m])[0, 1])


# ---------------------------------------------------------------------------
# cells
# ---------------------------------------------------------------------------

def npz_of(d, tag, st):
    return os.path.join(d, f"step{st:06d}_norm_{tag}.npz")


def json_of(d, tag, st):
    return os.path.join(d, f"step{st:06d}_norm_{tag}.json")


def load_cells(d, tag):
    out = {}
    for st in STEPS:
        p = npz_of(d, tag, st)
        if not os.path.exists(p):
            return None
        out[st] = np.load(p)
    return out


def load_json(d, tag, st):
    p = json_of(d, tag, st)
    return json.load(open(p)) if os.path.exists(p) else {}


def rows_identical(Zs, an):
    ws = [np.asarray(Zs[st][f"{an}__w"]) for st in STEPS]
    ts = [np.asarray(Zs[st][f"{an}__t0"]) for st in STEPS]
    return (all(np.array_equal(ws[0], w) for w in ws)
            and all(np.array_equal(ts[0], t) for t in ts))


def diets_of(Zs, an):
    per = []
    for st in STEPS:
        Z, pre = Zs[st], f"{an}__"
        per.append({k[len(pre) + 2:-6] for k in Z.files
                    if k.startswith(pre + "R_") and k.endswith("_l1_a0")})
    return order_diets(sorted(set.intersection(*per)))


_CELL_CACHE = {}


def cell_of(npz_path, an):
    """One cell's `Cell`, memoised per (file, anchor)."""
    ck = (npz_path, an)
    if ck not in _CELL_CACHE:
        _CELL_CACHE[ck] = Cell(np.load(npz_path), an)
    return _CELL_CACHE[ck]


class Cell:
    """One (seed, venue, step, anchor)'s fixed rows, with the H_pre axis and its guards."""

    def __init__(self, Z, an):
        self.Z, self.an, self.pre = Z, an, f"{an}__"
        self.ok = (self.pre + "t0") in Z.files
        if not self.ok:
            return
        g = self.g
        self.H = g("H_pre").astype(np.float64)
        self.s = g("nll_e").astype(np.float64)
        self.ex = g("excess_e").astype(np.float64)
        self.ks, self.jj, self.t0 = g("k_star"), g("j"), g("t0")
        self.et, self.cd = g("etype"), g("c_depth")
        self.n = len(self.H)

    def g(self, k):
        return np.asarray(self.Z[self.pre + k]) if (self.pre + k) in self.Z.files else None

    def ctrl(self, extra=None, idx=None):
        """The matcher's own guards as regression controls: k*, j, position, surprisal."""
        sl = slice(None) if idx is None else idx
        C = [self.ks[sl], self.jj[sl], self.t0[sl], self.s[sl]]
        if extra is not None:
            C = C + [np.asarray(e, np.float64) for e in extra]
        return np.stack([np.asarray(c, np.float64) for c in C], 1)

    def V(self, diet, l, a=0):
        return self.g(f"V_{diet}_l{l}_a{a}").astype(np.float64)

    def Vpre(self, diet, l, a=0):
        return self.g(f"Vpre_{diet}_l{l}_a{a}").astype(np.float64)

    def R(self, diet, l, a=0):
        return self.g(f"R_{diet}_l{l}_a{a}").astype(np.float64)

    def Ro(self, diet, l, a=0):
        return self.g(f"Ro_{diet}_l{l}_a{a}").astype(np.float64)


# ---------------------------------------------------------------------------
# 1. the axis's own guards
# ---------------------------------------------------------------------------

def sec_axis(Zs, an, tag, name):
    """What `H_pre` is, and what it is confounded with, on the fixed rows."""
    out = []
    rows = []
    for st in STEPS:
        C = Cell(Zs[st], an)
        q = np.quantile(C.H, [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0])
        rows.append([st, C.n, fmt(float(C.H.mean()), 4), fmt(float(C.H.std()), 4)]
                    + [fmt(float(x), 4) for x in q])
    out += [f"**The axis — {name}, {tag}, anchor {an}.**  `H_pre` is `H(q_(t0-1))`, the "
            f"entropy in nats of the trunk's forecast of the token AT the anchor, made one "
            f"position before it, on the EDITED stream.  The vocabulary is 16 tokens, so the "
            f"uniform ceiling is ln 16 = 2.7726 nats.  Rows are the held-out (`split == 2`) "
            f"fixed rows and are IDENTICAL across the three checkpoints; only the model "
            f"moves.", "",
            tbl(rows, ["step", "n", "mean", "sd", "min", "q05", "q25", "q50", "q75", "q95",
                       "max"]), ""]

    # the excess identity
    idc = []
    for st in STEPS:
        C = Cell(Zs[st], an)
        idc.append(float(np.abs(C.ex - (C.s - C.H)).max()))
    out += [f"Identity check `excess_e == nll_e - H_pre`: max |difference| = "
            f"{' / '.join(f'{x:.2e}' for x in idc)} over the three checkpoints (float32 "
            f"storage).  **At matched surprisal the `H_pre` axis is the negative excess "
            f"axis; they are one object.**", ""]

    # correlates
    rows = []
    for st in STEPS:
        C = Cell(Zs[st], an)
        rows.append([st,
                     fmt(_spear(C.H, C.ks), 3), fmt(_spear(C.H, C.jj), 3),
                     fmt(_spear(C.H, C.t0), 3), fmt(_spear(C.H, C.cd), 3),
                     fmt(_spear(C.H, C.s), 3), fmt(_pearson(C.H, C.s), 3),
                     fmt(_spear(C.H, C.ex), 3), fmt(_pearson(C.H, C.ex), 3),
                     fmt(_pearson(C.H, (C.t0 % 2).astype(float)), 3)])
    out += ["What the axis correlates with (Spearman unless marked `r`):", "",
            tbl(rows, ["step", "rho k*", "rho j", "rho t0", "rho c_depth", "rho nll_e",
                       "r nll_e", "rho excess", "r excess", "r (t0 mod 2)"]), ""]

    # by k*, by j, by position parity
    for key, lab, vals in [("ks", "k*", None), ("jj", "j", None)]:
        rows = []
        v0 = getattr(Cell(Zs[STEPS[0]], an), key)
        for u in sorted(set(v0.tolist())):
            cells = [u, int((v0 == u).sum())]
            for st in STEPS:
                C = Cell(Zs[st], an)
                s = getattr(C, key) == u
                cells += [fmt(float(C.H[s].mean()), 4), fmt(float(C.s[s].mean()), 3)]
            rows.append(cells)
        hdr = [lab, "n"] + [f"{x} @{st}" for st in STEPS for x in ("mean H_pre", "mean s")]
        out += [f"Mean `H_pre` and mean surprisal by `{lab}` (identical rows):", "",
                tbl(rows, hdr), ""]

    # the two-position sawtooth
    rows = []
    C0 = Cell(Zs[STEPS[0]], an)
    for par in (0, 1):
        cells = [par, int((C0.t0 % 2 == par).sum())]
        for st in STEPS:
            C = Cell(Zs[st], an)
            s = (C.t0 % 2) == par
            cells += [fmt(float(C.H[s].mean()), 4), fmt(float(C.s[s].mean()), 3)]
        rows.append(cells)
    out += ["The two-position constituent-phase sawtooth (`striatum/README.md` 4), "
            "`t0 mod 2`:", "",
            tbl(rows, ["t0 mod 2", "n"]
                + [f"{x} @{st}" for st in STEPS for x in ("mean H_pre", "mean s")]), ""]

    # how much survives the matcher's strata
    rows = []
    for st in STEPS:
        C = Cell(Zs[st], an)
        key = C.ks * 10000 + C.jj * 1000 + (C.t0 // 4)
        cells = [st, fmt(float(C.H.std()), 4)]
        for axis in (C.H, C.s):
            wv, tot = [], []
            for u in np.unique(key):
                s = key == u
                if s.sum() >= 2:
                    wv.append(((axis[s] - axis[s].mean()) ** 2).sum())
                    tot.append(s.sum())
            within = float(np.sum(wv) / max(np.sum(tot), 1))
            cells += [fmt(float(np.sqrt(within)), 4),
                      fmt(1.0 - within / max(float(axis.var()), 1e-18), 3)]
        rows.append(cells)
    out += [f"**How much of the axis survives the matcher's strata.**  The matcher's exact "
            f"strata are `(k*, j, t0 // 4)`; `within sd` is the pooled within-stratum sd and "
            f"`frac explained` is `1 - E[Var | stratum] / Var`.  The same decomposition for "
            f"the surprisal beside it, since the matcher then pairs on surprisal INSIDE a "
            f"stratum.", "",
            tbl(rows, ["step", "sd H_pre (total)", "H within sd", "H frac explained",
                       "s within sd", "s frac explained"]), ""]
    return "\n".join(out)


def sec_axis_seeds(dirs, tag, an, names):
    """The axis is a property of the MODEL, so it moves with the seed; side by side."""
    rows = []
    for st in STEPS:
        cells = [st]
        for d in dirs:
            p = npz_of(d, tag, st)
            if not os.path.exists(p):
                cells += ["--", "--", "--"]
                continue
            C = cell_of(p, an)
            cells += [fmt(float(C.H.mean()), 4), fmt(float(C.H.std()), 4),
                      fmt(_spear(C.H, C.s), 3)]
        rows.append(cells)
    hdr = ["step"] + [f"{x} {nm}" for nm in names
                      for x in ("mean H", "sd H", "rho(H, s)")]
    return "\n".join([
        f"**The axis across trajectory seeds — {tag}, anchor {an}.**  Same rows, different "
        f"trunks.", "", tbl(rows, hdr), ""])


# ---------------------------------------------------------------------------
# 2. the level: does the norm carry the context's uncertainty?
# ---------------------------------------------------------------------------

def sec_level_slope(Zs, an, tag, name, diets, a=0):
    """`V_pre ~ H_pre`: raw slope and the slope with the matcher's guards partialled out."""
    rows = []
    for nm in diets:
        for l in LS:
            cells = [nm, l]
            for st in STEPS:
                C = Cell(Zs[st], an)
                vp = C.Vpre(nm, l, a)
                b, _, _ = _ols(C.H, vp)
                bp, se, rp = _partial(vp, C.H, C.ctrl())
                cells += [fmt(b, 4), fmt(_spear(C.H, vp), 3), fmt(bp, 4), fmt(se, 4),
                          fmt(bp * float(C.H.std()), 4), fmt(rp, 3)]
            rows.append(cells)
    hdr = ["diet", "l"]
    for st in STEPS:
        hdr += [f"slope @{st}", f"rho @{st}", f"partial b @{st}", f"sem @{st}",
                f"b.sdH @{st}", f"partial rho @{st}"]
    return "\n".join([
        f"**The norm against the pre-event entropy — {name}, {tag}, anchor {an}, a = {a}.**  "
        f"`V_pre = V[l, a+1](s_(t0-1))`, the critic's level before the event, in units of "
        f"P(the frozen actor is right at the query).  `slope` is OLS of `V_pre` on `H_pre` "
        f"(nats); `partial b` is the same slope with BOTH sides residualised on the matcher's "
        f"own guards `(k*, j, t0, nll_e)`, so it is the level's dependence on the context's "
        f"uncertainty at matched surprisal.  **`b.sdH` is that slope times the cell's own "
        f"sd of `H_pre`** -- the level change over one sd of the axis, which is the only "
        f"column comparable across checkpoints, because a random-init trunk's forecast is "
        f"near-uniform on every row and its `H_pre` has almost no spread to regress on.  "
        f"`@0` is the trajectory's random-init trunk on identical rows.", "", tbl(rows, hdr), ""])


def sec_level_bins(Zs, an, tag, name, diet="full", nbin=5, a=0):
    out = []
    for st in STEPS:
        C = Cell(Zs[st], an)
        b = _qbin(C.H, nbin)
        rows = []
        for i in range(nbin):
            s = b == i
            if s.sum() < 5:
                continue
            cells = [i + 1, int(s.sum()), fmt(float(C.H[s].mean()), 4),
                     fmt(float(C.s[s].mean()), 3)]
            for l in LS:
                cells += [fmt(float(C.Vpre(diet, l, a)[s].mean()), 4)]
            rows.append(cells)
        out += [f"**step {st}** — `{diet}`, {nbin} quantile bins of `H_pre`:", "",
                tbl(rows, ["bin", "n", "mean H_pre", "mean s"]
                    + [f"V_pre l={l}" for l in LS]), ""]
    return "\n".join(out)


def sec_level_family(Zs, an, tag, name, a=0):
    """Does the entropy dependence RESCALE with the diet's expected outcome, the way the
    level's own calibration did (`norm/README.md` 1)?"""
    fams = {"out": ["out_lo", "out_mid", "out_hi"], "dmg": ["dmg_lo", "dmg_mid", "dmg_hi"],
            "mix": ["mix_q00", "mix_q25", "mix_q51"]}
    have = set(diets_of(Zs, an))
    rows = []
    for fam, nms in fams.items():
        if not all(x in have for x in nms):
            continue
        for l in LS:
            cells = [fam, l]
            for st in STEPS:
                C = Cell(Zs[st], an)
                bs, lv = [], []
                for nm in nms:
                    vp = C.Vpre(nm, l, a)
                    bp, _, _ = _partial(vp, C.H, C.ctrl())
                    bs.append(bp)
                    lv.append(float(vp.mean()))
                cells += [fmt(bs[0], 4), fmt(bs[1], 4), fmt(bs[2], 4),
                          fmt(_spear(np.arange(3), bs), 3), fmt(lv[2] - lv[0], 4)]
            rows.append(cells)
    if not rows:
        return ""
    hdr = ["family", "l"]
    for st in STEPS:
        hdr += [f"b lo @{st}", f"b mid @{st}", f"b hi @{st}", f"rho @{st}",
                f"level hi-lo @{st}"]
    return "\n".join([
        f"**Does the entropy dependence rescale with the world? — {name}, {tag}, anchor "
        f"{an}.**  `b` is the partial slope `V_pre ~ H_pre | (k*, j, t0, nll_e)` for each "
        f"diet of a tercile family; the families' three diets share their `(etype, j)` cell "
        f"histogram bit-identically and differ only in the world's expected outcome (`out`) "
        f"or expected damage (`dmg`), or in the quiet share (`mix`).  `rho` is Spearman of "
        f"the slope against the family's lo/mid/hi order (`out` and `mix` ascend in "
        f"E[outcome], `dmg` descends).  `level hi-lo` is the norm's own spread over the "
        f"family on the same rows, the object of `norm/README.md` 1.", "", tbl(rows, hdr), ""])


# ---------------------------------------------------------------------------
# 3. the response at matched surprisal
# ---------------------------------------------------------------------------

_MATCH_CACHE = {}


def matched_sets(npz_path, an, kind="flip", a=0):
    """`junction/analyze.py`'s matcher, untouched: exact strata on (k*, j, t0 // 4), 1:1
    nearest neighbour on surprisal at caliper 0.30, sign-balanced inside |ds| bands.

    Memoised per (file, anchor, label, offset): the matcher is deterministic (seed 0) and
    the same row set is read by several sections, so this changes no number."""
    ck = (npz_path, an, kind, a)
    if ck in _MATCH_CACHE:
        return _MATCH_CACHE[ck]
    R = Rows(npz_path, an)
    if not R.ok:
        _MATCH_CACHE[ck] = (None, {})
        return None, {}
    out = {}
    for l in LS:
        got = R.matched(kind, l, a, "kjt")
        if got is not None:
            out[l] = got
    _MATCH_CACHE[ck] = (R, out)
    return R, out


def sec_matched_guards(d, tag, an, name, a=0, kind="flip"):
    """Every guard printed on the matched sets, plus what H_pre does there."""
    rows = []
    for st in STEPS:
        p = npz_of(d, tag, st)
        if not os.path.exists(p):
            continue
        R, mm = matched_sets(p, an, kind, a)
        if R is None:
            continue
        C = cell_of(p, an)
        for l in LS:
            if l not in mm:
                continue
            m, lab = mm[l]
            rows.append([st, l, len(m), fmt(float(lab[m].mean()), 3),
                         fmt(_auc(C.s[m], lab[m])),
                         fmt(_auc(C.H[m], lab[m])),
                         fmt(_auc(C.ks[m].astype(float), lab[m])),
                         fmt(float(C.H[m].mean()), 4), fmt(float(C.H[m].std()), 4),
                         fmt(float(C.H.std()), 4),
                         fmt(_spear(C.H[m], C.s[m]), 3)])
    return "\n".join([
        f"**Guards on the matched sets — {name}, {tag}, anchor {an}, a = {a}, label "
        f"`{kind}`.**  The matcher never sees a diet or `H_pre`: it stratifies exactly on "
        f"`(k*, j, t0 // 4)`, pairs 1:1 on the model's surprisal at caliper 0.30 and "
        f"sign-balances inside `|ds|` bands.  `AUC(s)` is the surprisal guard and must read "
        f"0.500; `AUC(H_pre)` is what the entropy axis does on the same rows and is NOT "
        f"pinned by anything; `AUC(k*)` is a stratification check.  `sd H all rows` is the "
        f"axis's spread on the whole cell for comparison.", "",
        tbl(rows, ["step", "l", "matched n", "base", "AUC(s) [guard]", "AUC(H_pre)",
                   "AUC(k*)", "mean H", "sd H matched", "sd H all rows",
                   "rho(H, s) matched"]), ""])


def sec_axis_auc(dirs, names, tag, an, a=0, kind="flip"):
    """What the axis itself ranks on the matched rows, where the surprisal is pinned at
    0.500 by construction; every seed side by side."""
    rows = []
    for l in LS:
        for st in STEPS:
            cells = [l, st]
            for what in ("H", "s", "n"):
                per = []
                for d in dirs:
                    p = npz_of(d, tag, st)
                    if not os.path.exists(p):
                        per.append("--")
                        continue
                    R, mm = matched_sets(p, an, kind, a)
                    if R is None or l not in mm:
                        per.append("--")
                        continue
                    m, lab = mm[l]
                    C = cell_of(p, an)
                    per.append(str(len(m)) if what == "n" else
                               fmt(_auc(C.H[m] if what == "H" else C.s[m], lab[m])))
                cells.append(" / ".join(per))
            rows.append(cells)
    return "\n".join([
        f"**What the entropy axis itself ranks — {tag}, anchor {an}, a = {a}, label "
        f"`{kind}`, matched rows; {' / '.join(names)} side by side.**  `AUC(s)` is the "
        f"matcher's guard and must read 0.500 on every row; `AUC(H_pre)` is the entropy "
        f"axis's own ranking of the realised damage on those same rows and is pinned by "
        f"nothing.", "",
        tbl(rows, ["l", "step", "AUC(H_pre, flip)", "AUC(s) [guard]", "matched n"]), ""])


def sec_resp_bins(d, tag, an, name, diet="full", nbin=3, a=0, kind="flip"):
    """`R` inside terciles of `H_pre`, on the matched rows, with the damage AUC per bin."""
    out = []
    for st in STEPS:
        p = npz_of(d, tag, st)
        if not os.path.exists(p):
            continue
        R, mm = matched_sets(p, an, kind, a)
        if R is None:
            continue
        C = cell_of(p, an)
        rows = []
        for l in LS:
            if l not in mm:
                continue
            m, lab_all = mm[l]
            lab = lab_all[m]
            Rv, Ro = C.R(diet, l, a)[m], C.Ro(diet, l, a)[m]
            b = _qbin(C.H[m], nbin)
            for i in range(nbin):
                s = b == i
                if s.sum() < 10:
                    continue
                rows.append([l, i + 1, int(s.sum()), fmt(float(C.H[m][s].mean()), 4),
                             fmt(float(C.s[m][s].mean()), 3),
                             fmt(float(Rv[s].mean()), 4), fmt(_sem(Rv[s]), 4),
                             fmt(float((Rv - Ro)[s].mean()), 4), fmt(_sem((Rv - Ro)[s]), 4),
                             fmt(_auc(-Rv[s], lab[s])), fmt(float(lab[s].mean()), 3)])
        if rows:
            out += [f"**step {st}** — critic `{diet}`, matched rows, {nbin} bins of "
                    f"`H_pre`:", "",
                    tbl(rows, ["l", "H bin", "n", "mean H", "mean s", "mean R", "sem",
                               "mean R-Ro", "sem", "AUC(-R, flip)", "base"]), ""]
    return "\n".join(out)


def sec_level_seeds(dirs, names, tag, an, diet="full", a=0):
    """The level's partial slope on `H_pre`, every seed side by side, never averaged."""
    rows = []
    for l in LS:
        for st in STEPS:
            cells = [l, st]
            for what in ("raw", "partial", "bsd", "t"):
                per = []
                for d in dirs:
                    p = npz_of(d, tag, st)
                    if not os.path.exists(p):
                        per.append("--")
                        continue
                    C = cell_of(p, an)
                    vp = C.Vpre(diet, l, a)
                    if what == "raw":
                        b, _, _ = _ols(C.H, vp)
                        per.append(fmt(b, 4))
                    else:
                        b, se, _ = _partial(vp, C.H, C.ctrl())
                        per.append(fmt(b, 4) if what == "partial"
                                   else fmt(b * float(C.H.std()), 4) if what == "bsd"
                                   else (fmt(b / se, 1) if np.isfinite(se) and se > 0
                                         else "--"))
                cells.append(" / ".join(per))
            rows.append(cells)
    return "\n".join([
        f"**`V_pre ~ H_pre` across trajectory seeds — {tag}, anchor {an}, critic `{diet}`, "
        f"a = {a}; {' / '.join(names)} side by side, never averaged.**  `partial` "
        f"residualises both sides on `(k*, j, t0, nll_e)`.", "",
        tbl(rows, ["l", "step", "slope (raw)", "partial b", "b.sdH", "t"]), ""])


def sec_resp_partial(dirs, names, tag, an, diet="full", a=0, matched=True, kind="flip"):
    """The partial slope of `R` on `H_pre` with the guards out, every seed side by side,
    and the same slope with `V_pre` added to the controls (the arc's containment)."""
    rows = []
    for l in LS:
        for st in STEPS:
            cells = [l, st]
            for mode in ("base", "plusV"):
                per = []
                for d in dirs:
                    p = npz_of(d, tag, st)
                    if not os.path.exists(p):
                        per.append(("--", "--", "--"))
                        continue
                    C = cell_of(p, an)
                    if matched:
                        R, mm = matched_sets(p, an, kind, a)
                        if R is None or l not in mm:
                            per.append(("--", "--", "--"))
                            continue
                        idx = mm[l][0]
                    else:
                        idx = np.arange(C.n)
                    y = C.R(diet, l, a)[idx]
                    extra = [C.Vpre(diet, l, a)[idx]] if mode == "plusV" else None
                    ctl = C.ctrl(extra=extra, idx=idx)
                    b, se, _ = _partial(y, C.H[idx], ctl)
                    per.append((fmt(b, 4), fmt(b / se, 1) if np.isfinite(se) and se > 0
                                else "--", fmt(b * float(C.H[idx].std()), 4)))
                cells.append(" / ".join(x for x, _, _ in per))
                cells.append(" / ".join(z for _, _, z in per))
                cells.append(" / ".join(y for _, y, _ in per))
            n_per = []
            for d in dirs:
                p = npz_of(d, tag, st)
                if not os.path.exists(p):
                    n_per.append("--")
                    continue
                if matched:
                    R, mm = matched_sets(p, an, kind, a)
                    n_per.append(str(len(mm[l][0])) if (R is not None and l in mm) else "--")
                else:
                    n_per.append(str(cell_of(p, an).n))
            cells.append(" / ".join(n_per))
            rows.append(cells)
    where = "matched rows" if matched else "all fixed rows"
    return "\n".join([
        f"**Partial slope `R ~ H_pre` — {tag}, anchor {an}, critic `{diet}`, a = {a}, "
        f"{where}; {' / '.join(names)} side by side, never averaged.**  Both sides "
        f"residualised on `(k*, j, t0, nll_e)`; the `| V_pre` columns add the critic's own "
        f"pre-event level to the controls.  `t` is the slope over its standard error.", "",
        tbl(rows, ["l", "step", "b (R~H)", "b.sdH", "t", "b (R~H given V_pre)",
                   "b.sdH", "t", "n"]), ""])


def sec_interaction(dirs, names, tag, an, diet="full", a=0, matched=True, kind="flip",
                    score="R"):
    """The music literature's object: `y ~ z(s) + z(H) + z(s)*z(H)` on the same rows, with
    the interaction's sign and size.  Printed in BOTH parameterisations."""
    rows = []
    for l in LS:
        for st in STEPS:
            cells = [l, st]
            got = {k: [] for k in ("bs", "bh", "bi", "ti", "be", "bse", "n")}
            for d in dirs:
                p = npz_of(d, tag, st)
                if not os.path.exists(p):
                    for k in got:
                        got[k].append("--")
                    continue
                C = cell_of(p, an)
                if matched:
                    R, mm = matched_sets(p, an, kind, a)
                    if R is None or l not in mm:
                        for k in got:
                            got[k].append("--")
                        continue
                    idx = mm[l][0]
                else:
                    idx = np.arange(C.n)
                y = (C.R(diet, l, a) if score == "R" else
                     C.R(diet, l, a) - C.Ro(diet, l, a))[idx]
                zs, zh = _z(C.s[idx]), _z(C.H[idx])
                ze = _z(C.ex[idx])
                X = np.stack([zs, zh, zs * zh, C.ks[idx].astype(float),
                              C.jj[idx].astype(float), C.t0[idx].astype(float)], 1)
                cf, se, _ = _ols_multi(y, X)
                Xe = np.stack([zs, ze, zs * ze, C.ks[idx].astype(float),
                               C.jj[idx].astype(float), C.t0[idx].astype(float)], 1)
                cfe, see, _ = _ols_multi(y, Xe)
                got["bs"].append(fmt(cf[0], 4))
                got["bh"].append(fmt(cf[1], 4))
                got["bi"].append(fmt(cf[2], 4))
                got["ti"].append(fmt(cf[2] / se[2], 1) if se[2] > 0 else "--")
                got["be"].append(fmt(cfe[1], 4))
                got["bse"].append(fmt(cfe[0], 4))
                got["n"].append(str(len(idx)))
            for k in ("bs", "bh", "bi", "ti", "bse", "be", "n"):
                cells.append(" / ".join(got[k]))
            rows.append(cells)
    where = "matched rows" if matched else "all fixed rows"
    nm = {"R": "R", "dRo": "R - Ro"}.get(score, score)
    return "\n".join([
        f"**The surprisal x entropy interaction on `{nm}` — {tag}, anchor {an}, critic "
        f"`{diet}`, a = {a}, {where}; {' / '.join(names)} side by side.**  OLS of `{nm}` on "
        f"`z(s) + z(H) + z(s).z(H)` with `(k*, j, t0)` as linear controls, `s` and `H` "
        f"z-scored within the row set, so the coefficients are in units of `{nm}` per sd.  "
        f"The last two columns refit the SAME model in the excess parameterisation "
        f"(`z(s) + z(excess) + z(s).z(excess)`, with `excess = s - H`); it is one fit, not a "
        f"second finding.", "",
        tbl(rows, ["l", "step", "b z(s)", "b z(H)", "b interaction", "t", "b z(s) [exc]",
                   "b z(excess)", "n"]), ""])


def sec_surface(d, tag, an, name, diet="full", a=0, k=5, matched=False, kind="flip",
                score="R"):
    """Mean `R` on the (surprisal quantile x entropy quantile) grid, with counts."""
    out = []
    for st in STEPS:
        p = npz_of(d, tag, st)
        if not os.path.exists(p):
            continue
        C = cell_of(p, an)
        Rm = None
        if matched:
            Rm, mm = matched_sets(p, an, kind, a)
        for l in LS:
            if matched:
                if Rm is None or l not in mm:
                    continue
                idx = mm[l][0]
            else:
                idx = np.arange(C.n)
            if score == "R":
                z = C.R(diet, l, a)[idx]
            elif score == "dRo":
                z = (C.R(diet, l, a) - C.Ro(diet, l, a))[idx]
            elif score == "delta":
                base = C.g(f"ok_l{l}_a{a}").astype(bool)[idx]
                z = (C.g(f"oe_l{l}_a{a}").astype(np.float64)
                     - C.Vpre(diet, l, a))[idx]
                z = np.where(base, z, np.nan)
            else:
                raise ValueError(score)
            bx, by = _qbin(C.s[idx], k), _qbin(C.H[idx], k)
            good = np.isfinite(z)
            body = _grid(z[good], bx[good], by[good], k, k)
            hdr = ["H \\ s"] + [f"s q{i + 1}" for i in range(k)]
            edge = [f"{np.quantile(C.s[idx], (i + 0.5) / k):.2f}" for i in range(k)]
            out += [f"**step {st}, l = {l}** — mean `{score}` (n) per cell; `s` quantile "
                    f"midpoints {', '.join(edge)} nats:", "", tbl(body, hdr), ""]
    return "\n".join(out)


# ---------------------------------------------------------------------------
# 4. the outcome surprise
# ---------------------------------------------------------------------------

def sec_delta(d, tag, an, name, diet="full", a=0, nbin=5):
    """`delta = outcome - V_pre` by `H_pre`, with its two parts printed separately."""
    out = []
    for st in STEPS:
        p = npz_of(d, tag, st)
        if not os.path.exists(p):
            continue
        C = cell_of(p, an)
        rows = []
        for l in LS:
            oe = C.g(f"oe_l{l}_a{a}").astype(np.float64)
            base = C.g(f"ok_l{l}_a{a}").astype(bool)
            vp = C.Vpre(diet, l, a)
            dl = oe - vp
            b = _qbin(C.H, nbin)
            for i in range(nbin):
                s = (b == i) & base
                if s.sum() < 10:
                    continue
                rows.append([l, i + 1, int(s.sum()), fmt(float(C.H[s].mean()), 4),
                             fmt(float(oe[s].mean()), 4), fmt(float(vp[s].mean()), 4),
                             fmt(float(dl[s].mean()), 4), fmt(_sem(dl[s]), 4)])
        out += [f"**step {st}** — `{diet}`, {nbin} quantile bins of `H_pre`:", "",
                tbl(rows, ["l", "H bin", "n", "mean H_pre", "mean outcome", "mean V_pre",
                           "mean delta", "sem"]), ""]
    return "\n".join(out)


def sec_delta_partial(dirs, names, tag, an, diet="full", a=0):
    rows = []
    for l in LS:
        for st in STEPS:
            cells = [l, st]
            for what in ("delta", "outcome", "V_pre"):
                per = []
                for d in dirs:
                    p = npz_of(d, tag, st)
                    if not os.path.exists(p):
                        per.append("--")
                        continue
                    C = cell_of(p, an)
                    base = C.g(f"ok_l{l}_a{a}").astype(bool)
                    oe = C.g(f"oe_l{l}_a{a}").astype(np.float64)
                    vp = C.Vpre(diet, l, a)
                    y = {"delta": oe - vp, "outcome": oe, "V_pre": vp}[what]
                    idx = np.where(base)[0]
                    b, se, _ = _partial(y[idx], C.H[idx], C.ctrl(idx=idx))
                    per.append(fmt(b, 4) + (f" ({b / se:.1f})" if np.isfinite(se) and se > 0
                                            else ""))
                cells.append(" / ".join(per))
            rows.append(cells)
    return "\n".join([
        f"**Partial slope on `H_pre` of the outcome surprise and its two parts — {tag}, "
        f"anchor {an}, critic `{diet}`, a = {a}; {' / '.join(names)} side by side.**  All "
        f"residualised on `(k*, j, t0, nll_e)`; `(t)` is the slope over its standard error.  "
        f"`delta = outcome - V_pre` with `outcome` the frozen actor's own hit at the query "
        f"on the edited stream, so the `delta` column is exactly the `outcome` column minus "
        f"the `V_pre` column.", "",
        tbl(rows, ["l", "step", "b delta ~ H", "b outcome ~ H", "b V_pre ~ H"]), ""])


# ---------------------------------------------------------------------------
# 5. the twins -- H_pre is identical within a pair by construction
# ---------------------------------------------------------------------------

def twin_H(d, tag, st, an="tv", twin_h_dir=None, pre="tw__"):
    """`H_pre` per twin pair.

    Route (i), always attempted: join `tw__(w)` against the `tv__` fixed rows, which are
    the SAME eligibility set (`anchor_ok['tv'] = (tv > T_LO) & (et == 0) & (tv <= T - 1)`
    and the twins' `elig = (et == 0) & (tv >= T_LO + 1) & (tv <= T - 1)` are the same
    condition) but are saved for the TEST split only, so the join covers ~25% of pairs.
    Route (ii), if `twin_h_dir` is given: `hpre.py`'s per-pair column for every pair.

    Returns `(H, have, src, check)` where `check` is the route-(i) cross-check of the twin's
    own saved violator surprisal against the fixed row's `nll_e` on the joined rows.
    """
    p = npz_of(d, tag, st)
    Z = np.load(p)
    if pre + "w" not in Z.files:
        return None, None, "none", {}
    w = np.asarray(Z[pre + "w"])
    H = np.full(len(w), np.nan)
    src = "route(i) join on the test-split fixed rows"
    check = {}
    if f"{an}__w" in Z.files:
        fw = np.asarray(Z[f"{an}__w"])
        fH = np.asarray(Z[f"{an}__H_pre"], np.float64)
        fs = np.asarray(Z[f"{an}__nll_e"], np.float64)
        ft = np.asarray(Z[f"{an}__t0"])
        ftv = np.asarray(Z[f"{an}__t_v"])
        check["anchor t0 == t_v"] = bool(np.array_equal(ft, ftv))
        o = np.argsort(fw)
        pos = np.searchsorted(fw[o], w)
        pos = np.clip(pos, 0, len(fw) - 1)
        hit = fw[o][pos] == w
        H[hit] = fH[o][pos][hit]
        sv = np.asarray(Z[pre + "s_v"], np.float64)
        if hit.any():
            check["max |s_v - nll_e| on the join"] = float(
                np.abs(sv[hit] - fs[o][pos][hit]).max())
            check["joined pairs"] = int(hit.sum())
            check["split==2 pairs"] = int((np.asarray(Z[pre + "split"]) == 2).sum())
    base = os.path.basename(p)[:-4] + "_Hpre.npz"
    seed = os.path.basename(os.path.normpath(d))
    cands = ([os.path.join(twin_h_dir, seed, base), os.path.join(twin_h_dir, base)]
             if twin_h_dir else []) + [os.path.join(d, base)]
    for hp in cands:
        if os.path.exists(hp):
            Q = np.load(hp)
            key = {"tw__": "tw", "gl__": "gl"}[pre]
            if f"{key}_w" in Q.files:
                assert np.array_equal(np.asarray(Q[f"{key}_w"]), w), f"{hp}: pair order"
                H2 = np.asarray(Q[f"{key}_H_pre"], np.float64)
                ov = np.isfinite(H) & np.isfinite(H2)
                if ov.any():
                    check["route(i) vs route(ii) max |dH|"] = float(
                        np.abs(H[ov] - H2[ov]).max())
                    check["overlap pairs"] = int(ov.sum())
                H = H2
                src = "route(ii) recomputed per pair"
                check["route(ii) file"] = os.path.basename(hp)
            break
    have = np.isfinite(H)
    return H, have, src, check


def sec_twin_head(dirs, names, tag, st, twin_h_dir=None, caliper=0.3):
    rows = []
    for d, nm in zip(dirs, names):
        p, pj = npz_of(d, tag, st), json_of(d, tag, st)
        if not (os.path.exists(p) and os.path.exists(pj)):
            continue
        r = json.load(open(pj))
        T = Twins(p, r, caliper=caliper)
        if not T.ok:
            continue
        H, have, src, check = twin_H(d, tag, st, twin_h_dir=twin_h_dir)
        tb, sig = T.token_balanced()
        m = T.keep & have
        mb = tb & have
        rows.append([nm, int(T.keep.sum()), int(tb.sum()), sig,
                     int(m.sum()), int(mb.sum()),
                     fmt(float(H[m].mean()), 4) if m.sum() else "--",
                     fmt(float(H[m].std()), 4) if m.sum() else "--",
                     fmt(float(np.abs(T.ds[T.keep]).mean()), 4),
                     fmt(_spear(H[m], T.s_v[m]), 3) if m.sum() else "--",
                     src.split()[0],
                     str(check.get("overlap pairs", check.get("joined pairs", "--"))),
                     (f"{check['route(i) vs route(ii) max |dH|']:.2e}"
                      if "route(i) vs route(ii) max |dH|" in check else "--"),
                     (f"{check['max |s_v - nll_e| on the join']:.2e}"
                      if "max |s_v - nll_e| on the join" in check else "--")])
    if not rows:
        return ""
    return "\n".join([
        f"**The twin population and its `H_pre` — {tag}, step {st}.**  Pairs are the unit "
        f"and `H_pre` is IDENTICAL within a pair by construction (the prefix is bit-"
        f"identical; `task.py` asserts `max |log q_(t-1) difference| = 0`).  `balanced` is "
        f"`phasic.balance_tokens`'s subset with `signed hist` the exact signed token "
        f"histogram, which must be 0.  `with H` counts the pairs for which the axis is "
        f"available under the route named.  The last three columns are the route cross-check: "
        f"the number of pairs route (i)'s join also covers, the largest disagreement between "
        f"the two routes on those pairs, and the join's own gate (the twin's banked violator "
        f"surprisal against the joined fixed row's `nll_e`, which must be float32 zero).", "",
        tbl(rows, ["seed", "pairs @0.30", "balanced", "signed hist", "with H",
                   "balanced with H", "mean H", "sd H", "mean abs(ds)", "rho(H, s_v)",
                   "route", "route(i) n", "max abs(dH) (i vs ii)",
                   "max abs(s_v - nll_e) (join)"]), ""])


def sec_twin_bins_H(dirs, names, tag, st, diet="full", nbin=5, twin_h_dir=None,
                    balanced=True, caliper=0.3):
    """`dR` by `H_pre` bin, pairs the unit, every seed side by side."""
    got = []
    for d, nm in zip(dirs, names):
        p, pj = npz_of(d, tag, st), json_of(d, tag, st)
        if not (os.path.exists(p) and os.path.exists(pj)):
            continue
        T = Twins(p, json.load(open(pj)), caliper=caliper)
        if not T.ok:
            continue
        H, have, _, _ = twin_H(d, tag, st, twin_h_dir=twin_h_dir)
        tb, _ = T.token_balanced()
        m = (tb if balanced else T.keep) & have
        got.append((nm, T, H, m))
    if not got:
        return ""
    rows = []
    for l in LS:
        for i in range(nbin):
            cells = [l, i + 1]
            for nm, T, H, m in got:
                if (diet, l, 0) not in T.cols or m.sum() < 20:
                    cells += ["--", "--", "--"]
                    continue
                b = _qbin(H[m], nbin)
                dR = T.dR(diet, l)[m]
                s = b == i
                if s.sum() < 5:
                    cells += ["--", "--", "--"]
                    continue
                cells += [str(int(s.sum())), fmt(float(H[m][s].mean()), 3),
                          fmt(float(dR[s].mean()), 5) + " +- " + fmt(_sem(dR[s]), 5)]
            rows.append(cells)
    hdr = ["l", "H bin"] + [f"{x} {nm}" for nm, _, _, _ in got
                            for x in ("n", "mean H", "mean dR")]
    arm = "token-balanced" if balanced else "all pairs at caliper 0.30"
    return "\n".join([
        f"**`dR = R(violator) - R(legal twin)` by the pre-event entropy — {tag}, step {st}, "
        f"critic `{diet}`, {arm}; {' / '.join(nm for nm, _, _, _ in got)}.**  Bins are "
        f"per-seed quantiles of `H_pre` over that seed's own pairs.", "",
        tbl(rows, hdr), ""])


def sec_twin_slope_H(dirs, names, tag, diet="full", twin_h_dir=None, balanced=True,
                     caliper=0.3):
    """`dR ~ H_pre`: raw, residualised on `sec_twin_kstar`'s own control set, and the
    banked `dR ~ V_pre` slope with and without `H_pre` partialled out."""
    rows = []
    for l in LS:
        for st in STEPS:
            cells = [l, st]
            cols = {k: [] for k in ("bh", "th", "bhr", "bsd", "bv", "bvh", "n")}
            for d in dirs:
                p, pj = npz_of(d, tag, st), json_of(d, tag, st)
                if not (os.path.exists(p) and os.path.exists(pj)):
                    for k in cols:
                        cols[k].append("--")
                    continue
                T = Twins(p, json.load(open(pj)), caliper=caliper)
                H, have, _, _ = twin_H(d, tag, st, twin_h_dir=twin_h_dir)
                if not T.ok or (diet, l, 0) not in T.cols:
                    for k in cols:
                        cols[k].append("--")
                    continue
                tb, _ = T.token_balanced()
                m = (tb if balanced else T.keep) & have
                if m.sum() < 50:
                    for k in cols:
                        cols[k].append("--")
                    continue
                dR, vp = T.dR(diet, l)[m], T.vpre(diet, l)[m]
                h = H[m]
                b, _, _ = _ols(h, dR)
                G = np.stack([T.ks[m], T.jj[m], T.tv[m], T.s_v[m],
                              np.abs(T.ds[m])], 1).astype(np.float64)
                br, se, _ = _partial(dR, h, G)
                bv, _, _ = _ols(vp, dR)
                bvh, _, _ = _partial(dR, vp, np.stack([h], 1))
                cols["bh"].append(fmt(b, 4))
                cols["th"].append(fmt(br / se, 1) if np.isfinite(se) and se > 0 else "--")
                cols["bhr"].append(fmt(br, 4))
                cols["bsd"].append(fmt(br * float(h.std()), 5))
                cols["bv"].append(fmt(bv, 4))
                cols["bvh"].append(fmt(bvh, 4))
                cols["n"].append(str(int(m.sum())))
            for k in ("bh", "bhr", "bsd", "th", "bv", "bvh", "n"):
                cells.append(" / ".join(cols[k]))
            rows.append(cells)
    arm = "token-balanced" if balanced else "all pairs at caliper 0.30"
    return "\n".join([
        f"**`dR` against the pre-event entropy and against the norm — {tag}, critic "
        f"`{diet}`, {arm}; {' / '.join(names)} side by side.**  `b resid` residualises both "
        f"sides on `(k*, j, position, violator surprisal, |ds|)`, exactly the control set "
        f"`norm/analyze.py`'s `sec_twin_kstar` uses for the `V_pre` slope, and `t` is that "
        f"slope over its standard error.  The last two columns are `norm/README.md` 3's "
        f"`dR ~ V_pre` slope and the same slope with `H_pre` partialled out.", "",
        tbl(rows, ["l", "step", "b dR~H", "b resid", "b.sdH", "t", "b dR~V_pre",
                   "b dR~V_pre given H", "n"]), ""])


def sec_twin_surface(d, tag, st, name, diet="full", k=4, twin_h_dir=None, balanced=True,
                     caliper=0.3):
    p, pj = npz_of(d, tag, st), json_of(d, tag, st)
    if not (os.path.exists(p) and os.path.exists(pj)):
        return ""
    T = Twins(p, json.load(open(pj)), caliper=caliper)
    if not T.ok:
        return ""
    H, have, _, _ = twin_H(d, tag, st, twin_h_dir=twin_h_dir)
    tb, _ = T.token_balanced()
    m = (tb if balanced else T.keep) & have
    if m.sum() < 100:
        return ""
    out = []
    bx, by = _qbin(T.s_v[m], k), _qbin(H[m], k)
    edge = [f"{np.quantile(T.s_v[m], (i + 0.5) / k):.2f}" for i in range(k)]
    for l in LS:
        if (diet, l, 0) not in T.cols:
            continue
        body = _grid(T.dR(diet, l)[m], bx, by, k, k, nd=5)
        out += [f"**l = {l}** — mean `dR` (n) per cell:", "",
                tbl(body, ["H \\ s_v"] + [f"s q{i + 1}" for i in range(k)]), ""]
    arm = "token-balanced" if balanced else "all pairs"
    return "\n".join([
        f"**The `(violator surprisal x pre-event entropy)` surface of `dR` — {name}, {tag}, "
        f"step {st}, critic `{diet}`, {arm}, {int(m.sum())} pairs.  `s_v` quantile midpoints "
        f"{', '.join(edge)} nats.**", ""] + out)



# ---------------------------------------------------------------------------
# 0b. reproduction gates against the banked node
# ---------------------------------------------------------------------------

BANKED_MATCHED_N = {("swap65k", "tv", 64000): [2100, 2776, 2544, 1756]}
BANKED_AUC_R = {("swap65k", "tv", 64000): [0.611, 0.603, 0.569, 0.550]}
BANKED_AUC_V = {("swap65k", "tv", 64000): [0.693, 0.693, 0.655, 0.619]}
BANKED_DR = {("swap65k", 0): [-0.0008, -0.0003, -0.0001, -0.0001],
             ("swap65k", 8000): [0.0085, 0.0132, 0.0028, -0.0028],
             ("swap65k", 64000): [0.0077, 0.0124, 0.0127, 0.0043]}
BANKED_DR_SLOPE = {("swap65k", 64000): [0.176, 0.081, -0.018, -0.012]}


def sec_gate(dirs, names, tag, an, diet="full", a=0, twin_h_dir=None, caliper=0.3):
    """Every number this file rests on that the banked node already published, recomputed
    here with the banked machinery, so the row handling is checked before anything new is
    read off it."""
    out = []
    rows = []
    for st in STEPS:
        p = npz_of(dirs[0], tag, st)
        if not os.path.exists(p):
            continue
        R, mm = matched_sets(p, an, "flip", a)
        C = cell_of(p, an)
        if R is None:
            continue
        for l in LS:
            if l not in mm:
                continue
            m, lab = mm[l]
            bn = BANKED_MATCHED_N.get((tag, an, st))
            ba, bv = BANKED_AUC_R.get((tag, an, st)), BANKED_AUC_V.get((tag, an, st))
            rows.append([st, l, len(m), (str(bn[l - 1]) if bn else "--"),
                         fmt(_auc(-C.R(diet, l, a)[m], lab[m])),
                         (fmt(ba[l - 1]) if ba else "--"),
                         fmt(_auc(-C.V(diet, l, a)[m], lab[m])),
                         (fmt(bv[l - 1]) if bv else "--"),
                         fmt(_auc(C.s[m], lab[m]))])
    if rows:
        out += [f"**Fixed rows — {names[0]}, {tag}, anchor {an}, critic `{diet}`.**  The "
                f"matched cell sizes and the realised-damage AUCs of the response and the "
                f"level, recomputed with `junction/analyze.py`'s own matcher.  The `banked` "
                f"columns are `norm/README.md`'s reproduction-gate numbers where that node "
                f"published them (`swap65k` `t_v` 64k) and `--` where it did not.", "",
                tbl(rows, ["step", "l", "matched n", "banked n", "AUC(-R, flip)",
                           "banked", "AUC(-V, flip)", "banked", "AUC(s) [guard]"]), ""]
    rows = []
    for st in STEPS:
        for l in LS:
            cells = [st, l]
            for d in dirs:
                p, pj = npz_of(d, tag, st), json_of(d, tag, st)
                if not (os.path.exists(p) and os.path.exists(pj)):
                    cells.append("--")
                    continue
                T = Twins(p, json.load(open(pj)), caliper=caliper)
                if not T.ok or (diet, l, 0) not in T.cols:
                    cells.append("--")
                    continue
                tb, _ = T.token_balanced()
                dR = T.dR(diet, l)[tb]
                cells.append(fmt(float(dR.mean()), 4) + " +- " + fmt(_sem(dR), 4))
            bd = BANKED_DR.get((tag, st))
            cells.append(fmt(bd[l - 1], 4) if bd else "--")
            bs = BANKED_DR_SLOPE.get((tag, st))
            per = []
            for d in dirs:
                p, pj = npz_of(d, tag, st), json_of(d, tag, st)
                if not (os.path.exists(p) and os.path.exists(pj)):
                    per.append("--")
                    continue
                T = Twins(p, json.load(open(pj)), caliper=caliper)
                if not T.ok or (diet, l, 0) not in T.cols:
                    per.append("--")
                    continue
                tb, _ = T.token_balanced()
                b, _, _ = _ols(T.vpre(diet, l)[tb], T.dR(diet, l)[tb])
                per.append(fmt(b, 3))
            cells.append(" / ".join(per))
            cells.append(fmt(bs[l - 1], 3) if bs else "--")
            rows.append(cells)
    out += [f"**Twins — {tag}, critic `{diet}`, token-balanced.**  The mean paired "
            f"difference and the `dR ~ V_pre` slope, against `norm/README.md` 3's published "
            f"columns for the banked seed `traj_a1_s42`.  Nothing here is new; it is the "
            f"gate on the twin handling.", "",
            tbl(rows, ["step", "l"] + [f"mean dR {nm}" for nm in names]
                + ["banked dR (s42)", "slope dR~V_pre", "banked slope (s42)"]), ""]
    return "\n".join(out)


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def fig_axis(d, tag, an, name, path):
    plt = _mpl()
    fig, ax = plt.subplots(1, 4, figsize=(21.5, 4.6))
    for st, c in zip(STEPS, ["#999999", "#1f77b4", "#d62728"]):
        p = npz_of(d, tag, st)
        if not os.path.exists(p):
            continue
        C = cell_of(p, an)
        ax[0].hist(C.H, bins=60, histtype="step", lw=1.4, color=c, label=f"step {st}")
        b = _qbin(C.H, 12)
        xs = [float(C.H[b == i].mean()) for i in range(12) if (b == i).sum() > 5]
        ys = [float(C.s[b == i].mean()) for i in range(12) if (b == i).sum() > 5]
        ax[1].plot(xs, ys, marker="o", color=c, label=f"step {st}")
        kk = sorted(set(C.ks.tolist()))
        ax[2].plot(kk, [float(C.H[C.ks == k].mean()) for k in kk], marker="o", color=c,
                   label=f"step {st}")
        tt = sorted(set(C.t0.tolist()))
        ax[3].plot(tt, [float(C.H[C.t0 == t].mean()) for t in tt], marker=".", ms=4,
                   color=c, label=f"step {st}")
    ax[0].axvline(np.log(16), c="k", lw=0.8, ls=":")
    ax[0].set_xlabel("H_pre (nats)")
    ax[0].set_ylabel("rows")
    ax[0].set_title("the axis: H(q_(t0-1)) on the fixed rows\n(dotted = ln 16, the uniform "
                    "ceiling)", fontsize=9.5)
    ax[1].set_xlabel("H_pre bin (12 quantiles)")
    ax[1].set_ylabel("mean surprisal at the anchor (nats)")
    ax[1].set_title("what the axis is confounded with:\nthe violator's own surprisal",
                    fontsize=9.5)
    ax[2].set_xlabel("k* of the violation")
    ax[2].set_ylabel("mean H_pre")
    ax[2].set_title("mean H_pre by k*", fontsize=9.5)
    ax[3].set_xlabel("anchor position t0")
    ax[3].set_ylabel("mean H_pre")
    ax[3].set_title("mean H_pre by position\n(the two-position constituent-phase sawtooth)",
                    fontsize=9.5)
    for a_ in ax:
        a_.legend(fontsize=7, framealpha=0.85)
    fig.suptitle(f"precision 1: the pre-event entropy axis and its guards.  {name}, {tag}, "
                 f"anchor {an}; identical held-out rows at every checkpoint.", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=130)
    print("wrote", path)


def fig_level(d, tag, an, name, path, diet="full", nbin=8):
    plt = _mpl()
    fig, ax = plt.subplots(1, 4, figsize=(21.5, 4.6))
    for i, l in enumerate(LS):
        for st, c in zip(STEPS, ["#999999", "#1f77b4", "#d62728"]):
            p = npz_of(d, tag, st)
            if not os.path.exists(p):
                continue
            C = cell_of(p, an)
            b = _qbin(C.H, nbin)
            xs, ys, es = [], [], []
            for k in range(nbin):
                s = b == k
                if s.sum() < 10:
                    continue
                xs.append(float(C.H[s].mean()))
                ys.append(float(C.Vpre(diet, l)[s].mean()))
                es.append(_sem(C.Vpre(diet, l)[s]))
            ax[i].errorbar(xs, ys, yerr=es, marker="o", capsize=2, color=c,
                           label=f"step {st}")
        ax[i].set_title(f"the norm at level {l}", fontsize=9.5)
        ax[i].set_xlabel("H_pre bin (nats)")
        ax[i].set_ylabel("mean V_pre")
        ax[i].legend(fontsize=7, framealpha=0.85)
    fig.suptitle(f"precision 2: does the critic's pre-event level carry the context's "
                 f"uncertainty?  {name}, {tag}, anchor {an}, critic `{diet}`; identical rows, "
                 f"the random-init trunk in grey.", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, dpi=130)
    print("wrote", path)


def fig_surface(d, tag, an, name, path, diet="full", k=5, matched=False, kind="flip"):
    plt = _mpl()
    fig, ax = plt.subplots(len(STEPS), len(LS), figsize=(19.0, 11.0))
    for r_, st in enumerate(STEPS):
        p = npz_of(d, tag, st)
        if not os.path.exists(p):
            continue
        C = cell_of(p, an)
        mm = matched_sets(p, an, kind)[1] if matched else None
        for c_, l in enumerate(LS):
            idx = (mm[l][0] if (mm and l in mm) else np.arange(C.n))
            z = C.R(diet, l)[idx]
            bx, by = _qbin(C.s[idx], k), _qbin(C.H[idx], k)
            M = np.full((k, k), np.nan)
            N = np.zeros((k, k), int)
            for i in range(k):
                for j_ in range(k):
                    s = (bx == j_) & (by == i)
                    N[i, j_] = int(s.sum())
                    if s.sum() >= 5:
                        M[i, j_] = float(z[s].mean())
            v = np.nanmax(np.abs(M)) if np.isfinite(M).any() else 1.0
            im = ax[r_, c_].imshow(M, origin="lower", cmap="RdBu_r", vmin=-v, vmax=v,
                                   aspect="auto")
            for i in range(k):
                for j_ in range(k):
                    if N[i, j_]:
                        ax[r_, c_].text(j_, i, f"{N[i, j_]}", ha="center", va="center",
                                        fontsize=5.5, color="k", alpha=0.55)
            fig.colorbar(im, ax=ax[r_, c_], fraction=0.046)
            ax[r_, c_].set_title(f"step {st}, l = {l}", fontsize=9)
            ax[r_, c_].set_xticks(range(k), [f"{np.quantile(C.s[idx], (i + 0.5) / k):.1f}"
                                             for i in range(k)], fontsize=7)
            ax[r_, c_].set_yticks(range(k), [f"{np.quantile(C.H[idx], (i + 0.5) / k):.2f}"
                                             for i in range(k)], fontsize=7)
            ax[r_, c_].set_xlabel("surprisal quantile (mid, nats)", fontsize=8)
            ax[r_, c_].set_ylabel("H_pre quantile (mid, nats)", fontsize=8)
    fig.suptitle(f"precision 3: the response's (surprisal x pre-event entropy) surface.  "
                 f"{name}, {tag}, anchor {an}, critic `{diet}`"
                 + (f", matched `{kind}` rows" if matched else ", all fixed rows")
                 + ".  Cell colour = mean R (red positive), cell text = n; EACH PANEL HAS ITS "
                 "OWN COLOUR SCALE.", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=120)
    print("wrote", path)


def fig_twins(dirs, names, tag, st, path, diet="full", nbin=5, twin_h_dir=None,
              caliper=0.3, k=4):
    plt = _mpl()
    fig, ax = plt.subplots(1, 5, figsize=(26.0, 4.6))
    surf = None
    for d, nm in zip(dirs, names):
        p, pj = npz_of(d, tag, st), json_of(d, tag, st)
        if not (os.path.exists(p) and os.path.exists(pj)):
            continue
        T = Twins(p, json.load(open(pj)), caliper=caliper)
        if not T.ok:
            continue
        H, have, _, _ = twin_H(d, tag, st, twin_h_dir=twin_h_dir)
        tb, _ = T.token_balanced()
        m = tb & have
        if m.sum() < 50:
            continue
        for i, l in enumerate(LS):
            if (diet, l, 0) not in T.cols:
                continue
            b = _qbin(H[m], nbin)
            dR = T.dR(diet, l)[m]
            xs, ys, es = [], [], []
            for j_ in range(nbin):
                s = b == j_
                if s.sum() < 5:
                    continue
                xs.append(float(H[m][s].mean()))
                ys.append(float(dR[s].mean()))
                es.append(_sem(dR[s]))
            ax[i].errorbar(xs, ys, yerr=es, marker="o", capsize=2, label=nm)
        if surf is None:
            surf = (T, H, m, nm)
    for i, l in enumerate(LS):
        ax[i].axhline(0, c="k", lw=0.7)
        ax[i].set_title(f"dR by H_pre, level {l}", fontsize=9.5)
        ax[i].set_xlabel("H_pre bin (nats)")
        ax[i].set_ylabel("dR")
        ax[i].legend(fontsize=7, framealpha=0.85)
    if surf is not None:
        T, H, m, nm = surf
        l = 2 if (diet, 2, 0) in T.cols else LS[0]
        bx, by = _qbin(T.s_v[m], k), _qbin(H[m], k)
        dR = T.dR(diet, l)[m]
        M = np.full((k, k), np.nan)
        for i in range(k):
            for j_ in range(k):
                s = (bx == j_) & (by == i)
                if s.sum() >= 5:
                    M[i, j_] = float(dR[s].mean())
        v = np.nanmax(np.abs(M)) if np.isfinite(M).any() else 1.0
        im = ax[4].imshow(M, origin="lower", cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto")
        fig.colorbar(im, ax=ax[4], fraction=0.046)
        ax[4].set_title(f"dR surface (s_v x H_pre), {nm}, l = {l}", fontsize=9.5)
        ax[4].set_xticks(range(k), [f"{np.quantile(T.s_v[m], (i + 0.5) / k):.1f}"
                                    for i in range(k)], fontsize=7)
        ax[4].set_yticks(range(k), [f"{np.quantile(H[m], (i + 0.5) / k):.2f}"
                                    for i in range(k)], fontsize=7)
        ax[4].set_xlabel("violator surprisal quantile (mid, nats)", fontsize=8)
        ax[4].set_ylabel("H_pre quantile (mid, nats)", fontsize=8)
    fig.suptitle(f"precision 5: the same-prefix twins.  {tag}, step {st}, critic `{diet}`, "
                 f"token-balanced; H_pre is identical within a pair by construction.",
                 fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, dpi=130)
    print("wrote", path)


def fig_delta(d, tag, an, name, path, diet="full", nbin=8):
    plt = _mpl()
    fig, ax = plt.subplots(1, 4, figsize=(21.5, 4.6))
    for i, l in enumerate(LS):
        for st, c in zip(STEPS, ["#999999", "#1f77b4", "#d62728"]):
            p = npz_of(d, tag, st)
            if not os.path.exists(p):
                continue
            C = cell_of(p, an)
            base = C.g(f"ok_l{l}_a0").astype(bool)
            oe = C.g(f"oe_l{l}_a0").astype(np.float64)
            dl = oe - C.Vpre(diet, l)
            b = _qbin(C.H, nbin)
            xs, ys, es, yo = [], [], [], []
            for j_ in range(nbin):
                s = (b == j_) & base
                if s.sum() < 10:
                    continue
                xs.append(float(C.H[s].mean()))
                ys.append(float(dl[s].mean()))
                es.append(_sem(dl[s]))
                yo.append(float(oe[s].mean()))
            ax[i].errorbar(xs, ys, yerr=es, marker="o", capsize=2, color=c,
                           label=f"delta, step {st}")
            ax[i].plot(xs, yo, marker=".", ls=":", alpha=0.5, color=c,
                       label=f"outcome, step {st}")
        ax[i].axhline(0, c="k", lw=0.7)
        ax[i].set_title(f"outcome surprise at level {l}", fontsize=9.5)
        ax[i].set_xlabel("H_pre bin (nats)")
        ax[i].set_ylabel("delta = outcome - V_pre")
        ax[i].legend(fontsize=6.5, framealpha=0.85)
    fig.suptitle(f"precision 4: the outcome surprise by the pre-event entropy.  {name}, "
                 f"{tag}, anchor {an}, critic `{diet}`; dotted = the outcome alone (the "
                 f"frozen actor's own hit).", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, dpi=130)
    print("wrote", path)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

PREAMBLE = """# precision — the value reader against the world model's pre-event uncertainty

Facts only; interpretation is discussed with Jasper and lives nowhere in this folder.
Generated by [`reduce.py`](../reduce.py) from the `norm/` round's banked `.json` / `.npz`
files (plus, for the twins, [`hpre.py`](../hpre.py)'s one recomputed column); nothing is
refit and no critic, actor or trunk is touched.  Three trajectory seeds are printed side by
side and **never averaged**.

**The question.**  Every value-side contrast in the arc is taken at matched surprisal:
`junction/analyze.py`'s matcher pins the model's own surprisal at the anchor (the guard
reads 0.500 on every banked table) and the same-prefix twins are selected at a surprisal
caliper.  None of them has asked whether the reader depends on `H_pre = H(q_(t0-1))`, the
entropy of the trunk's own forecast of the token AT the anchor, made one position before it.
`norm/task.py` has recorded that column on every fixed row since the round was run and no
banked table has used it.

**Objects.**  `V[l, d](s_t)` is the outcome-trained ridge critic on the frozen trunk's
residual stream predicting `o_l(t + d)`, the frozen actor being RIGHT at level `l`, so
higher `V` is a better world.  `V_pre = V[l, a+1](s_(t0-1))` is the **norm**;
`R = V[l, a](s_t) - V_pre` is the **response**; `delta = outcome - V_pre` is the **outcome
surprise**.  `s = nll_e` is the model's surprisal of the token at the anchor and
`H_pre` the entropy of the distribution that surprisal is read off.

**One axis, two names.**  `excess_e = nll_e - H_pre` is on file and the identity is checked
per cell, so **at matched surprisal the `H_pre` axis and the excess axis are the same axis
with opposite sign**.  `logit_reading/README.md` 2b found `s - H(q)` a null as a scalar for
detection, and `striatum/README.md` 5's addendum found the banked excess head carries nothing
once conditioned on the value level.  Every interaction fit below is printed in both
parameterisations so that one object is not reported as two findings, and the response
sections additionally report the slope with `V_pre` in the controls (the arc's containment
conditioning) beside the slope without it.

**Rows.**  The npz holds the `split == 2` TEST windows only, so every fixed row is held out
from the critic's Gram and its lambda selection.  Within one trajectory the fixed rows at an
anchor are the same windows at the same positions at all three checkpoints (asserted per
cell), so every step-0-against-64k comparison is a difference on identical rows —
[`orbitofrontal/README.md`](../../../orbitofrontal/README.md) 5's reading rule.  `tv` is the
first exactly-impossible token and `fd` the edit onset.

**Venues.**  The tercile diets were only ever fitted on the `a1` venue (the sweep passes
`window_diets = 0` for `swap65k`), so `swap65k` carries `full` and the clean-only critic
alone.

**Twins.**  `H_pre` is not saved with the twin pairs.  Two routes, both reported: **(i)** the
join of `tw__w` against the `tv__` fixed rows — the two eligibility conditions are the same
(`(et == 0) & (t_v > 8) & (t_v <= T-1)`), so every twin whose window is in the test split has
a fixed row, and the join is checked by requiring the twin's own saved violator surprisal
`s_v` to equal that row's `nll_e`; **(ii)** [`hpre.py`](../hpre.py), one forward pass per
cell on an L4 that recomputes the log-softmax at `t_v - 1` for exactly the banked pairs and
saves the entropy, which covers every pair.  Where both exist they are cross-checked to
float32.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--tags", default="swap65k,a1")
    ap.add_argument("--anchors", default="tv,fd")
    ap.add_argument("--out", default=".")
    ap.add_argument("--figs", default="")
    ap.add_argument("--outfile", default="tables.md")
    ap.add_argument("--diet", default="full")
    ap.add_argument("--twin-h", default="")
    ap.add_argument("--grid", type=int, default=5)
    ap.add_argument("--a", type=int, default=0)
    ap.add_argument("--figs-only", action="store_true",
                    help="skip the tables and regenerate the figures alone")
    args = ap.parse_args()

    dirs = args.dirs
    names = [os.path.basename(os.path.normpath(x)) for x in dirs]
    tags = [t for t in args.tags.split(",") if t]
    ancs = [t for t in args.anchors.split(",") if t]
    twin_h = args.twin_h or None
    P = [PREAMBLE]

    if args.figs_only:
        _figures(args, dirs, names, tags, ancs, twin_h)
        return

    # ---- integrity: touch every member of every fetched file --------------------------
    bad = []
    for d in dirs:
        for tag in tags:
            for st in STEPS:
                p = npz_of(d, tag, st)
                if not os.path.exists(p):
                    bad.append(f"{p}: missing")
                    continue
                try:
                    Z = np.load(p)
                    for k in Z.files:
                        _ = np.asarray(Z[k]).sum()
                except Exception as e:                                  # noqa: BLE001
                    bad.append(f"{p}: {type(e).__name__} {e}")
    P.append("\n---\n\n## 0. Integrity\n\nEvery member of every fetched `.npz` is "
             "decompressed and summed before use (a fetch in an earlier round returned a "
             "full-size file with a bad CRC on one member that `np.load` alone did not "
             "catch).\n\n"
             + ("All files read clean.\n" if not bad
                else "**FAILURES:**\n\n" + "\n".join(f"- `{b}`" for b in bad) + "\n"))
    if bad:
        print("INTEGRITY FAILURES:", bad)

    for tag in tags:
        for an in ancs:
            Zs = load_cells(dirs[0], tag)
            if Zs is None or f"{an}__t0" not in Zs[STEPS[0]].files:
                continue
            assert rows_identical(Zs, an), f"{tag} {an}: rows differ across steps"
            P.append(f"\n---\n\n# {tag}, anchor {an}\n")

            P.append("## 0b. Reproduction gates\n")
            P.append(sec_gate(dirs, names, tag, an, args.diet, args.a, twin_h))

            P.append("## 1. The axis's own guards\n")
            P.append(sec_axis(Zs, an, tag, names[0]))
            P.append(sec_axis_seeds(dirs, tag, an, names))

            P.append("## 2. The level: does the norm carry the context's uncertainty?\n")
            dts = diets_of(Zs, an)
            P.append(sec_level_slope(Zs, an, tag, names[0], dts, args.a))
            P.append(sec_level_bins(Zs, an, tag, names[0], args.diet, 5, args.a))
            s = sec_level_family(Zs, an, tag, names[0], args.a)
            if s:
                P.append(s)
            P.append(sec_level_seeds(dirs, names, tag, an, args.diet, args.a))

            P.append("## 3. The response at matched surprisal\n")
            P.append(sec_matched_guards(dirs[0], tag, an, names[0], args.a))
            P.append(sec_axis_auc(dirs, names, tag, an, args.a))
            P.append(sec_resp_bins(dirs[0], tag, an, names[0], args.diet, 3, args.a))
            P.append(sec_resp_partial(dirs, names, tag, an, args.diet, args.a, True))
            P.append(sec_resp_partial(dirs, names, tag, an, args.diet, args.a, False))
            P.append(sec_interaction(dirs, names, tag, an, args.diet, args.a, True, "flip",
                                     "R"))
            P.append(sec_interaction(dirs, names, tag, an, args.diet, args.a, False, "flip",
                                     "R"))
            P.append(sec_interaction(dirs, names, tag, an, args.diet, args.a, False, "flip",
                                     "dRo"))
            P.append(f"### The 2-D surface, {names[0]}, all fixed rows\n")
            P.append(sec_surface(dirs[0], tag, an, names[0], args.diet, args.a, args.grid,
                                 False))

            P.append("## 4. The outcome surprise\n")
            P.append(sec_delta(dirs[0], tag, an, names[0], args.diet, args.a))
            P.append(sec_delta_partial(dirs, names, tag, an, args.diet, args.a))
            P.append(f"### The 2-D surface of `delta`, {names[0]}, all fixed rows\n")
            P.append(sec_surface(dirs[0], tag, an, names[0], args.diet, args.a, args.grid,
                                 False, score="delta"))

        # twins live at t_v only, once per venue
        if any(os.path.exists(npz_of(d, tag, STEPS[0])) for d in dirs):
            P.append(f"\n---\n\n# {tag} — the same-prefix twins\n")
            for st in STEPS:
                s = sec_twin_head(dirs, names, tag, st, twin_h)
                if s:
                    P.append(s)
            P.append(sec_twin_slope_H(dirs, names, tag, args.diet, twin_h, True))
            P.append(sec_twin_slope_H(dirs, names, tag, args.diet, twin_h, False))
            for st in STEPS:
                s = sec_twin_bins_H(dirs, names, tag, st, args.diet, 5, twin_h, True)
                if s:
                    P.append(s)
            for st in STEPS:
                s = sec_twin_surface(dirs[0], tag, st, names[0], args.diet, 4, twin_h, True)
                if s:
                    P.append(s)

    os.makedirs(args.out, exist_ok=True)
    p = os.path.join(args.out, args.outfile)
    with open(p, "w") as f:
        f.write("\n".join(P).rstrip() + "\n")
    print(f"wrote {p}  ({os.path.getsize(p) / 1024:.0f} KB)")

    _figures(args, dirs, names, tags, ancs, twin_h)


def _figures(args, dirs, names, tags, ancs, twin_h):
    if args.figs:
        os.makedirs(args.figs, exist_ok=True)
        for tag in tags:
            an = "tv" if os.path.exists(npz_of(dirs[0], tag, STEPS[0])) else None
            Zs = load_cells(dirs[0], tag)
            if Zs is None:
                continue
            for an in ancs:
                if f"{an}__t0" not in Zs[STEPS[0]].files:
                    continue
                fig_axis(dirs[0], tag, an, names[0],
                         os.path.join(args.figs, f"prec_axis_{tag}_{an}.png"))
                fig_level(dirs[0], tag, an, names[0],
                          os.path.join(args.figs, f"prec_level_{tag}_{an}.png"), args.diet)
                fig_surface(dirs[0], tag, an, names[0],
                            os.path.join(args.figs, f"prec_surface_{tag}_{an}.png"),
                            args.diet, args.grid)
                fig_delta(dirs[0], tag, an, names[0],
                          os.path.join(args.figs, f"prec_delta_{tag}_{an}.png"), args.diet)
            for st in (64000, 8000):
                try:
                    fig_twins(dirs, names, tag, st,
                              os.path.join(args.figs, f"prec_twins_{tag}_{st}.png"),
                              args.diet, twin_h_dir=twin_h)
                except Exception as e:                                  # noqa: BLE001
                    print(f"fig_twins {tag} {st}: {type(e).__name__} {e}")


if __name__ == "__main__":
    main()
