"""The norm's calibration re-read on the richer readers -- the reduction.  CPU only; nothing is
refit.  Facts only; no interpretation lives here.

Reads, per trajectory seed and checkpoint, the banked `norm/` cell (`step*_norm_a1.{json,npz}`,
the ridge on the raw state, every diet), `calibration.py`'s per-diet refits
(`step*_norm_a1_calib.{json,npz}`: arms `Q` = + log q, `P` = + precision, `L` = ln_f(s), `M` =
MLP) and, for the clean-only critic's row, `express.py`'s refits (`_express.npz` for `cleanQ` /
`cleanP`, `_express_ln.npz` for `cleanL`).  Every arm is read on the IDENTICAL `split == 2`
test rows (asserted) through a view (`ArmZ`) that puts the arm's column in place of each diet's
own, so `norm/analyze.py`'s `sec_norm_level`, `sec_norm_shape` and `sec_delta` run on each arm
VERBATIM: they are called unchanged with their `fmt` / `tbl` swapped for identity / capture
(`capture`), which returns the numbers they would have printed.  The per-family orderings are
`sec_ordering`'s `V_pre (the norm)` row recomputed from those numbers and checked against
`sec_ordering` itself on the ridge; the entropy slopes are `precision/reduce.py`'s `_partial`
on its `Cell`, checked against its `sec_level_family` on the ridge; the matched rows are
`junction/analyze.py`'s matcher via `precision/reduce.py`'s `matched_sets` (they never see a
reader); the pooled damage label is `discrim.py`'s `label_of`.

  python -m rhm.logit_reading.striatum.norm.precision.rereads.calibration.reduce_calibration \\
      <dir>/traj_a1_s42 <dir>/traj_a1_s43 <dir>/traj_a1_s44 \\
      --out rhm/logit_reading/striatum/norm/precision/rereads/calibration/results \\
      --figs rhm/logit_reading/striatum/norm/precision/rereads/calibration/figs --date 20260922
"""

import argparse
import json
import os
import re

import numpy as np

from rhm.logit_reading.striatum.analyze import _auc, fmt, tbl
from rhm.logit_reading.striatum.junction.analyze import Rows
import rhm.logit_reading.striatum.norm.analyze as NA
from rhm.logit_reading.striatum.norm.analyze import order_diets, _spear, _pop
import rhm.logit_reading.striatum.norm.precision.reduce as PR
from rhm.logit_reading.striatum.norm.precision.reduce import Cell, _partial, matched_sets
from rhm.logit_reading.striatum.norm.discrim import label_of

STEPS = [0, 8000, 64000]
LS = [1, 2, 3, 4]
ANCHORS = ["tv", "fd"]
TAG = "a1"
ARMS = ["", "Q", "P", "L", "M"]
ARM_NAME = {"": "ridge (banked)", "Q": "+log q", "P": "+precision", "L": "ln_f(s)",
            "M": "MLP"}
FAMS = {"out": ["out_lo", "out_mid", "out_hi"], "dmg": ["dmg_lo", "dmg_mid", "dmg_hi"],
        "mix": ["mix_q00", "mix_q25", "mix_q51"]}
KEY_RE = re.compile(r"^(fd|tv)__(V|Vpre|R|Ro|Vpreo|Rsh)_(.+)_l(\d)_a(\d+)$")
POPS = {"tv": ["viol"], "fd": ["viol", "none"]}


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

class CachedNpz:
    """`np.load` with each member decompressed once."""

    def __init__(self, path):
        self.path = path
        self.Z = np.load(path)
        self.files = list(self.Z.files)
        self._c = {}

    def __contains__(self, k):
        return k in self._c or k in self.Z.files

    def __getitem__(self, k):
        if k not in self._c:
            self._c[k] = np.asarray(self.Z[k])
        return self._c[k]


class ArmZ:
    """A read-only npz-like view of one cell for one reader: the banked cell's columns with
    the reader's `V_` / `Vpre_` / `R_` column in place of every diet's own.  Every banked
    reader column (`V`, `Vpre`, `R`, `Ro`, `Vpreo`, `Rsh`) is dropped from the view first, so a
    diet or offset the arm lacks reads as missing rather than falling back to the ridge."""

    def __init__(self, bank, arm, calib=None, clean_src=None, diets=()):
        self.src = {}
        if not arm:
            for k in bank.files:
                self.src[k] = (bank, k)
        else:
            for k in bank.files:
                if not KEY_RE.match(k):
                    self.src[k] = (bank, k)
            for z, tag in ((calib, None), (clean_src, "clean")):
                if z is None:
                    continue
                for k in z.files:
                    m = KEY_RE.match(k)
                    if not m or m.group(2) not in ("V", "Vpre", "R"):
                        continue
                    dn = m.group(3)
                    if not dn.endswith(arm):
                        continue
                    diet = dn[:-len(arm)]
                    if (tag is None and diet not in diets) or (tag and diet != tag):
                        continue
                    self.src[f"{m.group(1)}__{m.group(2)}_{diet}_l{m.group(4)}_a{m.group(5)}"] = (z, k)
        self.files = list(self.src)
        self._fs = set(self.files)

    def __contains__(self, k):
        return k in self._fs

    def __getitem__(self, k):
        z, kk = self.src[k]
        return z[kk]


class ArmRows(Rows):
    """`junction/analyze.py`'s `Rows` on an in-memory view (its `__init__` verbatim but for the
    `np.load`).  Only the column accessors are used on arms; the matched sets come from the
    banked file and never see a reader."""

    def __init__(self, Z, anchor, caliper=0.3, tbucket=4, seed=0):
        self.Z = Z
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


class CellSet:
    """One (trajectory seed, checkpoint): the bank, the refits, one view per arm."""

    def __init__(self, d, st, sfx="_calib"):
        p = lambda s: os.path.join(d, f"step{st:06d}_norm_{TAG}{s}")          # noqa: E731
        self.ok = all(os.path.exists(p(s)) for s in (".npz", ".json", f"{sfx}.npz",
                                                      f"{sfx}.json"))
        if not self.ok:
            return
        self.bank_path = p(".npz")
        self.bank = CachedNpz(p(".npz"))
        self.r = json.load(open(p(".json")))
        self.calib = CachedNpz(p(f"{sfx}.npz"))
        self.cj = json.load(open(p(f"{sfx}.json")))
        self.ex = CachedNpz(p("_express.npz")) if os.path.exists(p("_express.npz")) else None
        self.exln = (CachedNpz(p("_express_ln.npz")) if os.path.exists(p("_express_ln.npz"))
                     else None)
        self.diets = list(self.cj["diets"])
        self.mlp_diets = list(self.cj.get("mlp_diets", []))
        self.views = {}
        for arm in ARMS:
            if arm and not any(k.startswith("tv__Vpre_") and k.endswith(f"{arm}_l1_a0")
                               for k in self.calib.files):
                continue
            clean_src = {"Q": self.ex, "P": self.ex, "L": self.exln}.get(arm)
            dts = self.mlp_diets if arm == "M" else self.diets
            self.views[arm] = ArmZ(self.bank, arm, self.calib, clean_src, dts)
        for an in ANCHORS:
            for k in ("w", "t0"):
                assert np.array_equal(self.calib[f"{an}__{k}"], self.bank[f"{an}__{k}"]), \
                    f"{d} {st} {an}: calib rows differ from the bank"

    def rows(self, arm, an):
        return ArmRows(self.views[arm], an)


def load_all(dirs, sfx):
    out = {}
    for d in dirs:
        nm = os.path.basename(os.path.normpath(d))
        out[nm] = {}
        for st in STEPS:
            cs = CellSet(d, st, sfx)
            if cs.ok:
                out[nm][st] = cs
    return out


# ---------------------------------------------------------------------------
# the verbatim sections, captured
# ---------------------------------------------------------------------------

def capture(mod, fn, *a, **k):
    """Run a banked section unchanged with its module's `fmt` / `tbl` swapped for identity /
    capture; returns [(hdr, rows)] holding the raw numbers it would have printed."""
    got = []
    old = (mod.fmt, mod.tbl)
    mod.fmt = lambda x, nd=3: x
    mod.tbl = lambda rows, hdr: (got.append((list(hdr), [list(r) for r in rows])), "")[1]
    try:
        fn(*a, **k)
    finally:
        mod.fmt, mod.tbl = old
    return got


def _num(x):
    return float("nan") if (x is None or isinstance(x, str)) else float(x)


class Reading:
    """Every number of the reduction for one (seed, step, anchor, arm)."""

    def __init__(self, cs, an, arm):
        self.cs, self.an, self.arm = cs, an, arm
        R, r = cs.rows(arm, an), cs.r
        self.R = R
        self.level, self.shape, self.delta = {}, {}, {}
        for pk in POPS[an]:
            if _pop(R, pk).sum() < 20:
                continue
            got = capture(NA, NA.sec_norm_level, R, r, 0, (pk,))
            assert len(got) == 1
            hdr, rows = got[0]
            self.level[pk] = {row[0]: {"E": _num(row[1]),
                                       "mean": [_num(x) for x in row[2:6]],
                                       "sd": [_num(x) for x in row[6:10]]} for row in rows}
        for l in LS:
            got = capture(NA, NA.sec_norm_shape, R, r, 0, "full", l)
            if not got:
                continue
            self.shape[l] = {row[0]: [_num(x) for x in row[1:6]] for row in got[0][1]}
        got = capture(NA, NA.sec_delta, R, r, (0, 4, 8))
        i = 0
        for pk in ("viol", "none"):
            if _pop(R, pk).sum() < 20:
                continue
            for l in LS:
                hdr, rows = got[i]
                i += 1
                self.delta[(pk, l)] = {row[0]: [_num(x) for x in row[1:]] for row in rows}
        assert i == len(got)

    # sec_ordering's `V_pre (the norm)` row, from the captured levels
    def ordering(self, l, pk="viol", what="level", ai=0):
        ds, sp = self.cs.r["diet_stats"], self.cs.r["diet_spec"]
        nms = [n for n in order_diets(self.R.diets()) if n in ds]
        fam = {n: sp[n].get("family", "anchor") for n in nms}
        if what == "level":
            y = np.array([self.level[pk][n]["mean"][l - 1] for n in nms])
        else:                  # the outcome surprise at a = (0, 4, 8)[ai]
            y = np.array([self.delta[(pk, l)][n][2 * ai] for n in nms])
        x = np.array([ds[n]["mean_outcome"] for n in nms])
        out = {}
        for f in [None, "dmg", "mix", "out"]:
            sel = [i for i, n in enumerate(nms) if f is None or fam[n] == f]
            out[f or "all"] = _spear(x[sel], y[sel]) if len(sel) > 2 else float("nan")
        out["n_all"] = len(nms)
        return out


_READ = {}


def reading(cs, an, arm):
    k = (id(cs), an, arm)
    if k not in _READ:
        _READ[k] = Reading(cs, an, arm)
    return _READ[k]


def sci(x):
    return "--" if x is None else f"{float(x):.1e}"


def trip(vals, nd=3):
    return " / ".join(fmt(v, nd) if not (isinstance(v, float) and np.isnan(v)) else "--"
                      for v in vals)


def arms_of(S):
    got = None
    for nm in S:
        for st in S[nm]:
            a = set(S[nm][st].views)
            got = a if got is None else (got & a)
    return [a for a in ARMS if a in (got or set())]


def seeds_of(S):
    return list(S)


def diets_arm(S, arm, with_clean=True):
    cs = next(iter(next(iter(S.values())).values()))
    base = cs.mlp_diets if arm == "M" else cs.diets
    out = order_diets(list(base) + (["clean"] if (with_clean and arm != "M") else []))
    return out


# ---------------------------------------------------------------------------
# sections
# ---------------------------------------------------------------------------

def sec_integrity(dirs, sfx):
    rows = []
    for d in dirs:
        for f in sorted(os.listdir(d)):
            if not f.endswith(".npz") or "smoke" in f:
                continue
            Z = np.load(os.path.join(d, f))
            n, s = 0, 0.0
            for k in Z.files:
                a = Z[k]
                n += 1
                if a.dtype.kind in "fiub":
                    s += float(np.asarray(a, np.float64).sum())
            rows.append([os.path.basename(os.path.normpath(d)), f, n,
                         "finite" if np.isfinite(s) else "NON-FINITE"])
    return "\n".join([
        "Every member of every fetched npz decompressed and summed (an earlier round's fetch "
        "returned a full-size file with a bad CRC that `np.load` alone did not catch).", "",
        tbl(rows, ["seed", "file", "members", "sum"]), ""])


def sec_gates(S):
    rows = []
    for nm in S:
        for st in STEPS:
            cs = S[nm].get(st)
            if cs is None:
                rows.append([nm, st] + ["MISSING"] * 9)
                continue
            g = cs.cj["gates"]
            ex = g.get("express", {})

            def exm(arm):
                v = [x["max_abs"] for k, x in ex.items() if k.endswith(f"full{arm}")]
                return sci(max(v)) if v else "--"
            rows.append([nm, st, sci(max(g["actor"].values())),
                         sci(max(g["diets"].values())), sci(max(g["val_r2"].values())),
                         str(all(x for k, x in g["rows"].items() if isinstance(x, bool))),
                         sci(max(x for k, x in g["rows"].items() if not isinstance(x, bool))),
                         sci(max(g["cols"].values())),
                         " / ".join(exm(a) for a in ("Q", "P", "L")), exm("M"),
                         "PASS" if not cs.cj["gate_fails"] else f"FAIL {cs.cj['gate_fails']}"])
    return "\n".join([
        "**Gates, asserted in-container before the cell returned.**  `actor`: the frozen actor's "
        "clean accuracy against the banked json (max abs over levels).  `diets`: every diet's "
        "row count, window count and E[outcome] against the banked `diet_stats` (max abs over "
        "the eleven).  `val R2`: arm 0's held-out `R2` against the banked `critic_val_r2` on "
        "every stored column of every diet (max abs).  `rows`: the test rows (`w`, `t0`) and "
        "the outcome labels exact.  `nll / H_pre`: max abs.  `arm-0 cols`: every diet's "
        "`V` / `Vpre` / `R` / `Ro` / `Vpreo` against the banked columns, both anchors, every "
        "level and offset (max abs, float32).  `express Q / P / L`: the `full` diet's arms "
        "against `express.py`'s `fullQ` / `fullP` and `express_ln`'s `fullL` (max abs; asserted "
        "under 5e-5).  `express M`: the `full` MLP against `express.py`'s `fullM` (recorded, "
        "not asserted).", "",
        tbl(rows, ["seed", "step", "actor", "diets", "val R2", "rows & labels exact",
                   "nll / H_pre", "arm-0 cols", "express Q / P / L", "express M", "verdict"]),
        ""])


README_BANK = {  # norm/README.md 1 and 2, traj_a1_s42 (plus the s43/s44 slopes of the re-read)
    "level": {"out_lo": [0.902, 0.813, 0.627, 0.343], "out_mid": [0.927, 0.863, 0.729, 0.485],
              "out_hi": [0.940, 0.897, 0.799, 0.621]},
    "slope_l1_64k": {"out_lo": [1.161, 1.176, 1.164], "out_hi": [0.816, 0.787, 0.792]},
    "delta_l2": {"out_lo": -0.221, "out_mid": -0.272, "out_hi": -0.306},
    "delta_l1": {"out_lo": -0.219, "out_mid": -0.243, "out_hi": -0.256},
}


def sec_repro(S):
    """The ridge through this reduction against the numbers `norm/README.md` quotes, and the
    captured-verbatim numbers against the banked sections they came from."""
    seeds = seeds_of(S)
    out = ["**The ridge, through this file's own path, against `norm/README.md` 1-2** (a1, "
           "anchor `tv`, 64k, violation rows).  README values are traj_a1_s42 unless marked.",
           ""]
    rows = []
    for dn, ref in README_BANK["level"].items():
        for l in LS:
            vals = [reading(S[nm][64000], "tv", "").level["viol"][dn]["mean"][l - 1]
                    for nm in seeds if 64000 in S[nm]]
            rows.append([f"V_pre {dn}", l, fmt(ref[l - 1], 3), trip(vals, 4)])
    for dn, ref in README_BANK["slope_l1_64k"].items():
        vals = [reading(S[nm][64000], "tv", "").shape[1][dn][0] for nm in seeds
                if 64000 in S[nm]]
        rows.append([f"slope on full {dn}", 1, trip(ref, 3) + " (s42/s43/s44)", trip(vals, 4)])
    for key, l in (("delta_l1", 1), ("delta_l2", 2)):
        for dn, ref in README_BANK[key].items():
            vals = [reading(S[nm][64000], "tv", "").delta[("viol", l)][dn][0] for nm in seeds
                    if 64000 in S[nm]]
            rows.append([f"delta a=0 {dn}", l, fmt(ref, 3), trip(vals, 4)])
    out += [tbl(rows, ["quantity", "l", "README", "this file " + " / ".join(seeds)]), ""]

    # the ordering recomputation against sec_ordering itself, and the entropy slopes against
    # sec_level_family itself, on the ridge, every cell and anchor
    n_ord, worst_ord, n_fam, worst_fam = 0, 0.0, 0, 0.0
    for nm in seeds:
        for st, cs in S[nm].items():
            for an in ANCHORS:
                Rb = Rows(cs.bank_path, an)
                if not Rb.ok:
                    continue
                got = capture(NA, NA.sec_ordering, Rb, cs.r, 0)
                for li, (hdr, rows_) in enumerate(got):
                    l = LS[li]
                    row = [r_ for r_ in rows_ if r_[0] == "V_pre (the norm)"][0]
                    mine = reading(cs, an, "").ordering(l)
                    for fam in ("all", "dmg", "mix", "out"):
                        j = hdr.index(f"mean_outcome/{fam}")
                        a, b = _num(row[j]), mine[fam]
                        if np.isfinite(a) or np.isfinite(b):
                            worst_ord = max(worst_ord, abs(a - b))
                            n_ord += 1
        Zs = {st: S[nm][st].bank for st in S[nm]}
        if len(Zs) == len(STEPS):
            for an in ANCHORS:
                got = capture(PR, PR.sec_level_family, Zs, an, TAG, nm, 0)
                if not got:
                    continue
                hdr, rows_ = got[0]
                for row in rows_:
                    fam, l = row[0], row[1]
                    for st in STEPS:
                        e = entropy_family(S[nm][st], an, "", l)
                        for i, dn in enumerate(FAMS[fam]):
                            j = hdr.index(f"b {['lo', 'mid', 'hi'][i]} @{st}")
                            worst_fam = max(worst_fam, abs(_num(row[j]) - e[dn]["b"]))
                            n_fam += 1
    out += [f"**Recomputations checked against the banked sections on the ridge.**  The "
            f"per-family orderings of this file against `norm/analyze.py`'s `sec_ordering` "
            f"(`V_pre (the norm)` row, `mean_outcome` columns), every seed, checkpoint, anchor "
            f"and level: {n_ord} comparisons, max abs difference {worst_ord:.1e}.  The entropy "
            f"partial slopes against `precision/reduce.py`'s `sec_level_family`: {n_fam} "
            f"comparisons, max abs difference {worst_fam:.1e}.  `sec_norm_level`, "
            f"`sec_norm_shape` and `sec_delta` are not recomputed: they are the banked "
            f"functions, run unchanged on each arm's view.", ""]
    return "\n".join(out)


def sec_fit(S, dd):
    seeds = seeds_of(S)
    arms = arms_of(S)
    out = []
    for st in STEPS:
        rows = []
        cs0 = S[seeds[0]].get(st)
        if cs0 is None:
            continue
        for dn in order_diets(cs0.diets):
            for l in LS:
                cells = [dn, l]
                for arm in arms:
                    vals = []
                    for nm in seeds:
                        cs = S[nm].get(st)
                        if cs is None:
                            vals.append(float("nan"))
                            continue
                        if not arm:
                            v = cs.r["critic_val_r2"][dn].get(f"l{l}_d{dd}")
                        else:
                            v = cs.cj["val_r2"].get(f"{dn}{arm}", {}).get(f"l{l}_d{dd}")
                        vals.append(float("nan") if v is None else v)
                    cells.append(trip(vals, 4))
                rows.append(cells)
        out += [f"**Held-out fit `R2` of `V[l, d={dd}]` — step {st}; "
                f"{' / '.join(seeds)} side by side.**", "",
                tbl(rows, ["diet", "l"] + [ARM_NAME[a] for a in arms]), ""]
    return "\n".join(out)


def sec_level(S, an, pk):
    seeds, arms = seeds_of(S), arms_of(S)
    out = []
    for st in STEPS:
        if any(st not in S[nm] for nm in seeds):
            continue
        for l in LS:
            rows = []
            dts = order_diets(diets_arm(S, "") + ["clean"])
            for dn in dts:
                e = S[seeds[0]][st].r["diet_stats"].get(dn, {}).get("mean_outcome")
                if dn == "clean":
                    e = S[seeds[0]][st].r.get("clean_critic_mean_outcome")
                cells = [dn, fmt(e, 4)]
                for arm in arms:
                    vals = []
                    for nm in seeds:
                        lv = reading(S[nm][st], an, arm).level.get(pk, {}).get(dn)
                        vals.append(float("nan") if lv is None else lv["mean"][l - 1])
                    cells.append(trip(vals, 4))
                rows.append(cells)
            out += [f"**Mean `V_pre` on {pk} rows — anchor {an}, step {st}, l = {l}; "
                    f"{' / '.join(seeds)}.**  `diet E[o]` is seed {seeds[0]}'s (the per-seed "
                    f"values are in `diet_stats` and differ in the third decimal).", "",
                    tbl(rows, ["diet", "diet E[o]"] + [ARM_NAME[a] for a in arms]), ""]
    return "\n".join(out)


def sec_ordering_tab(S, an, pk="viol", what="level", ai=0):
    seeds, arms = seeds_of(S), arms_of(S)
    if ai:
        arms = [a for a in arms if a != "M"]
    out = []
    for arm in arms:
        rows = []
        for st in STEPS:
            if any(st not in S[nm] for nm in seeds):
                continue
            for l in LS:
                o = [reading(S[nm][st], an, arm).ordering(l, pk, what, ai) for nm in seeds]
                cells = [st, l] + [trip([x[f] for x in o], 2) for f in ("out", "dmg", "mix",
                                                                          "all")]
                # the extreme-diet difference hi - lo inside each family (a sign when a family
                # has fewer than three diets on this arm)
                for f, (lo, hi) in (("out", ("out_lo", "out_hi")), ("dmg", ("dmg_lo", "dmg_hi")),
                                    ("mix", ("mix_q00", "mix_q51"))):
                    vals = []
                    for nm in seeds:
                        rd = reading(S[nm][st], an, arm)
                        if what == "level":
                            a_, b_ = (rd.level[pk].get(lo), rd.level[pk].get(hi))
                            vals.append(float("nan") if a_ is None or b_ is None
                                        else b_["mean"][l - 1] - a_["mean"][l - 1])
                        else:
                            a_, b_ = rd.delta[(pk, l)].get(lo), rd.delta[(pk, l)].get(hi)
                            vals.append(float("nan") if a_ is None or b_ is None
                                        else b_[2 * ai] - a_[2 * ai])
                    cells.append(trip(vals, 4))
                rows.append(cells)
        n_all = reading(S[seeds[0]][STEPS[-1]], an, arm).ordering(1, pk, what, ai)["n_all"]
        obj = ("mean `V_pre`" if what == "level" else
               f"mean `delta = outcome - V_pre` (a = {(0, 4, 8)[ai]})")
        out += [f"**{ARM_NAME[arm]} — Spearman rho of {obj} on {pk} rows against the diet's "
                f"E[outcome], anchor {an}; {' / '.join(seeds)}.**  `out` / `dmg` / `mix` = "
                f"inside one family (3 diets, so rho is +-1 or +-0.5); `all` = over the "
                f"{n_all} diets this arm was fitted on.  `hi - lo` is the family's extreme-diet "
                f"difference (`out_hi - out_lo`, `dmg_hi - dmg_lo`, `mix_q51 - mix_q00`); "
                f"`out` and `mix` ascend in E[outcome], `dmg` descends.", "",
                tbl(rows, ["step", "l", "rho out", "rho dmg", "rho mix", "rho all",
                           "out hi-lo", "dmg hi-lo", "mix hi-lo"]), ""]
    return "\n".join(out)


def sec_shape_head(S, an):
    seeds, arms = seeds_of(S), arms_of(S)
    rows = []
    for arm in arms:
        for dn in ("out_lo", "out_mid", "out_hi", "dmg_lo", "dmg_hi", "mix_q00", "mix_q51",
                   "natural_sized", "clean"):
            if arm == "M" and dn == "clean":
                continue
            cells = [ARM_NAME[arm], dn]
            for st in STEPS:
                vals = []
                for nm in seeds:
                    sh = reading(S[nm][st], an, arm).shape.get(1, {}).get(dn)
                    vals.append(float("nan") if sh is None else sh[0])
                cells.append(trip(vals, 3))
            for l in (2, 3, 4):
                vals = []
                for nm in seeds:
                    sh = reading(S[nm][64000], an, arm).shape.get(l, {}).get(dn)
                    vals.append(float("nan") if sh is None else sh[0])
                cells.append(trip(vals, 3))
            rows.append(cells)
    return "\n".join([
        f"**The shape headline — slope of each diet's `V_pre` on the SAME arm's `full` "
        f"`V_pre`, all fixed rows at anchor {an} (`sec_norm_shape`, a = 0); "
        f"{' / '.join(seeds)}.**  Slope 1 with an intercept = a pure level shift; slope != 1 = "
        f"a rescaling.  The banked reading (ridge, `tv`, l = 1): ~1.0 at step 0, `out_lo` 1.16 "
        f"-> `out_hi` 0.82 at 64k.", "",
        tbl(rows, ["arm", "diet", "l=1 @0", "l=1 @8k", "l=1 @64k", "l=2 @64k", "l=3 @64k",
                   "l=4 @64k"]), ""])


def sec_shape_full(S, an, arm):
    seeds = seeds_of(S)
    rows = []
    for dn in diets_arm(S, arm):
        if dn == "full":
            continue
        for l in LS:
            cells = [dn, l]
            for j in (0, 4):                      # slope, sd ratio at each step
                for st in STEPS:
                    vals = []
                    for nm in seeds:
                        sh = reading(S[nm][st], an, arm).shape.get(l, {}).get(dn)
                        vals.append(float("nan") if sh is None else sh[j])
                    cells.append(trip(vals, 3))
            for j in (2, 1, 3):                   # R2, intercept, mean shift at 0 and 64k
                for st in (0, 64000):
                    vals = []
                    for nm in seeds:
                        sh = reading(S[nm][st], an, arm).shape.get(l, {}).get(dn)
                        vals.append(float("nan") if sh is None else sh[j])
                    cells.append(trip(vals, 3))
            rows.append(cells)
    hdr = (["diet", "l"] + [f"slope @{s}" for s in ("0", "8k", "64k")]
           + [f"sd ratio @{s}" for s in ("0", "8k", "64k")]
           + ["R2 @0", "R2 @64k", "intercept @0", "intercept @64k", "mean shift @0",
              "mean shift @64k"])
    return "\n".join([
        f"**{ARM_NAME[arm]} — `sec_norm_shape`: each diet's `V_pre` regressed on this arm's "
        f"`full` `V_pre`, all fixed rows at anchor {an}, a = 0; {' / '.join(seeds)}.**", "",
        tbl(rows, hdr), ""])


def sec_delta_tab(S, an, pk, steps=(0, 64000)):
    seeds, arms = seeds_of(S), arms_of(S)
    out = []
    for st in steps:
        if any(st not in S[nm] for nm in seeds):
            continue
        rows = []
        for dn in order_diets(diets_arm(S, "")):
            for l in LS:
                cells = [dn, l]
                oc = [reading(S[nm][st], an, "").delta[(pk, l)][dn][1] for nm in seeds]
                cells.append(trip(oc, 4))
                for arm in arms:
                    vals = []
                    for nm in seeds:
                        dd = reading(S[nm][st], an, arm).delta[(pk, l)].get(dn)
                        vals.append(float("nan") if dd is None else dd[0])
                    cells.append(trip(vals, 4))
                for arm in [a for a in arms if a != "M"]:
                    for jj in (2, 4):
                        vals = []
                        for nm in seeds:
                            dd = reading(S[nm][st], an, arm).delta[(pk, l)].get(dn)
                            vals.append(float("nan") if dd is None else dd[jj])
                        cells.append(trip(vals, 4))
                rows.append(cells)
        hdr = (["diet", "l", "outcome a=0"] + [f"d a=0 {ARM_NAME[a]}" for a in arms]
               + sum([[f"d a=4 {ARM_NAME[a]}", f"d a=8 {ARM_NAME[a]}"]
                      for a in arms if a != "M"], []))
        out += [f"**The outcome surprise `delta = outcome - V_pre` on {pk} rows — anchor {an}, "
                f"step {st}; {' / '.join(seeds)}** (`sec_delta` per arm).  The outcome at a "
                f"given offset is the same for every diet and arm on these rows, so at a = 0 "
                f"`delta` is `-V_pre` plus a constant and its ordering is section 2's "
                f"restated; a = 4 / 8 use different critic columns (`V[l, a+1]`).  The MLP arm "
                f"was fitted at a = 0 only.", "", tbl(rows, hdr), ""]
    return "\n".join(out)


# ---- discrimination --------------------------------------------------------------------------

def _disc(cs, an, arm, l, readout):
    """(within-cell AUC on junction's matched rows, pooled AUC on discrim's damage base)."""
    Z = cs.views[arm]
    k = {"V_pre": f"{an}__Vpre_{{}}_l{l}_a0", "V": f"{an}__V_{{}}_l{l}_a0"}[readout]
    _, ms = matched_sets(cs.bank_path, an, "flip", 0)
    y, base = label_of(cs.bank.Z, an, "damage", l, 0)
    out = {}
    for dn in (cs.mlp_diets if arm == "M" else cs.diets):
        kk = k.format(dn)
        if kk not in Z:
            continue
        sc = np.asarray(Z[kk], np.float64)
        w = float("nan")
        if l in ms:
            mm, lab = ms[l]
            w = _auc(-sc[mm], lab[mm])
        p = _auc(-sc[base], y[base]) if base.sum() >= 20 else float("nan")
        out[dn] = (w, p, len(ms[l][0]) if l in ms else 0)
    return out


_DISC = {}


def disc(cs, an, arm, l, readout):
    k = (id(cs), an, arm, l, readout)
    if k not in _DISC:
        _DISC[k] = _disc(cs, an, arm, l, readout)
    return _DISC[k]


def sec_disc_head(S, an):
    seeds, arms = seeds_of(S), arms_of(S)
    rows = []
    for arm in arms:
        for l in LS:
            cells = [ARM_NAME[arm], l]
            for ro in ("V_pre", "V"):
                per = {nm: [disc(S[nm][st], an, arm, l, ro).get("full", (np.nan, np.nan, 0))[0]
                            for st in STEPS] for nm in seeds}
                cells += [trip([per[nm][0] for nm in seeds], 3),
                          trip([per[nm][2] for nm in seeds], 3),
                          trip([per[nm][2] - per[nm][0] for nm in seeds], 3)]
            rows.append(cells)
    return "\n".join([
        f"**The headline — the `full` diet, within-cell `AUC(-score, damage)` on junction's "
        f"matched rows, anchor {an}, a = 0; {' / '.join(seeds)}.**  `d` = 64k minus step 0.", "",
        tbl(rows, ["arm", "l", "V_pre @0", "V_pre @64k", "V_pre d", "V @0", "V @64k", "V d"]),
        ""])


def sec_disc(S, an, readout):
    seeds, arms = seeds_of(S), arms_of(S)
    out = []
    nrow = []
    for st in STEPS:
        nrow.append([st] + [trip([disc(S[nm][st], an, "", l, readout).get("full", (0, 0, 0))[2]
                                  for nm in seeds], 0) for l in LS])
    out += [f"Matched-row counts (junction's `flip` matcher, caliper 0.30, kjt strata; the "
            f"same rows for every arm and diet of a cell; they move with the checkpoint because "
            f"the damage label's base contains the actor's unedited hit):", "",
            tbl(nrow,
                ["step"] + [f"n l={l}" for l in LS]), ""]
    for arm in arms:
        rows = []
        for dn in diets_arm(S, arm, with_clean=False):
            for l in LS:
                cells = [dn, l]
                per = {nm: [disc(S[nm][st], an, arm, l, readout).get(dn, (np.nan, np.nan, 0))
                            for st in STEPS] for nm in seeds}
                for i in range(3):
                    cells.append(trip([per[nm][i][0] for nm in seeds], 3))
                cells.append(trip([per[nm][2][0] - per[nm][0][0] for nm in seeds], 3))
                cells.append(trip([per[nm][0][1] for nm in seeds], 3))
                cells.append(trip([per[nm][2][1] for nm in seeds], 3))
                cells.append(trip([per[nm][2][1] - per[nm][0][1] for nm in seeds], 3))
                rows.append(cells)
        out += [f"**{ARM_NAME[arm]} — `AUC(-{readout}, damage)`, anchor {an}, a = 0; "
                f"{' / '.join(seeds)}.**  `within` = on junction's matched rows (exact strata on "
                f"`(k*, j, t0 // 4)`, sign-balanced surprisal pairs), `pooled` = on every fixed "
                f"row of `discrim.py`'s damage base (`ok & etype < 2 & oo`).  `d(64k-0)` is the "
                f"trained-minus-random difference on the identical fixed rows.", "",
                tbl(rows, ["diet", "l", "within @0", "within @8k", "within @64k",
                           "within d(64k-0)", "pooled @0", "pooled @64k", "pooled d(64k-0)"]),
                ""]
    return "\n".join(out)


# ---- the entropy axis ------------------------------------------------------------------------

_ENT = {}


def entropy_family(cs, an, arm, l, a=0):
    """Per diet: `precision/reduce.py`'s partial slope `V_pre ~ H_pre | (k*, j, t0, nll_e)`,
    its sem, and the slope times sd(H_pre)."""
    k = (id(cs), an, arm, l, a)
    if k in _ENT:
        return _ENT[k]
    C = Cell(cs.views[arm], an)
    sdH = float(C.H.std())
    out = {}
    for dn in (cs.mlp_diets if arm == "M" else cs.diets):
        if f"{an}__Vpre_{dn}_l{l}_a{a}" not in cs.views[arm]:
            continue
        vp = C.Vpre(dn, l, a)
        b, se, _ = _partial(vp, C.H, C.ctrl())
        out[dn] = {"b": b, "se": se, "bsd": b * sdH}
    oe = C.g(f"oe_l{l}_a{a}").astype(np.float64)
    b, se, _ = _partial(oe, C.H, C.ctrl())
    out["_outcome"] = {"b": b, "se": se, "bsd": b * sdH}
    _ENT[k] = out
    return out


def sec_entropy(S, an):
    seeds, arms = seeds_of(S), arms_of(S)
    out = []
    rows = []
    for st in STEPS:
        for l in LS:
            cells = [st, l, trip([entropy_family(S[nm][st], an, "", l)["_outcome"]["bsd"]
                                  for nm in seeds], 4)]
            for arm in arms:
                cells.append(trip([entropy_family(S[nm][st], an, arm, l)["full"]["bsd"]
                                   for nm in seeds], 4))
            rows.append(cells)
    out += [f"**The `full` diet's level along the axis, per arm — anchor {an}; "
            f"{' / '.join(seeds)}.**  Partial slope `V_pre ~ H_pre | (k*, j, t0, nll_e)` times the "
            f"cell's sd of `H_pre` (the level's change over one sd of the axis); `OUTCOME` is the "
            f"realised outcome's own slope on the same rows, the same for every arm.  Step 0's "
            f"`H_pre` has almost no spread (`precision/FILES.md` gotchas).", "",
            tbl(rows, ["step", "l", "OUTCOME"] + [ARM_NAME[a] for a in arms]), ""]
    ds = None
    for arm in arms:
        rows = []
        for st in STEPS:
            for l in LS:
                cells = [st, l]
                for fam, nms in FAMS.items():
                    per = [entropy_family(S[nm][st], an, arm, l) for nm in seeds]
                    if not all(dn in per[0] for dn in nms):
                        cells += ["--"] * 4
                        continue
                    for dn in nms:
                        cells.append(trip([p[dn]["bsd"] for p in per], 4))
                    cells.append(trip([_spear(np.arange(3), [p[dn]["b"] for dn in nms])
                                       for p in per], 2))
                # over every diet this arm was fitted on, against E[outcome]
                rs = []
                for nm, p in zip(seeds, [entropy_family(S[nm][st], an, arm, l) for nm in seeds]):
                    ds = S[nm][st].r["diet_stats"]
                    dn_ = [dn for dn in p if not dn.startswith("_") and dn in ds]
                    rs.append(_spear([ds[dn]["mean_outcome"] for dn in dn_],
                                     [p[dn]["b"] for dn in dn_]) if len(dn_) > 2 else np.nan)
                cells.append(trip(rs, 2))
                rows.append(cells)
        hdr = ["step", "l"]
        for fam, nms in FAMS.items():
            hdr += [f"{dn} b.sdH" for dn in nms] + [f"rho {fam}"]
        hdr += ["rho all vs E[o]"]
        out += [f"**{ARM_NAME[arm]} — does the level's entropy dependence order with the world? "
                f"Anchor {an}; {' / '.join(seeds)}.**  Per diet, the partial slope times sd(H_pre) "
                f"(`b.sdH`).  `rho <fam>` is Spearman of the slope against the family's lo/mid/hi "
                f"order (`precision/reduce.py`'s `sec_level_family` column: `out` and `mix` ascend "
                f"in E[outcome], `dmg` descends); `rho all vs E[o]` is Spearman of the slope "
                f"against E[outcome] over every diet this arm was fitted on.", "",
                tbl(rows, hdr), ""]
    return "\n".join(out)


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

# categorical slots 1-4 of the dataviz reference palette in fixed order (validated: CVD dE >= 9.1,
# normal-vision >= 22.9; the contrast WARN is relieved by the legend and the tables), with the
# banked ridge in neutral ink as the reference
ARM_COLOR = {"": "#4a4a4a", "Q": "#2a78d6", "P": "#eb6834", "L": "#1baf7a", "M": "#eda100"}
SEED_MARK = ["o", "s", "^"]


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    return plt


def fig_level(S, an, path):
    """Mean V_pre on violation rows against the diet's E[outcome], per arm and level."""
    plt = _mpl()
    seeds, arms = seeds_of(S), arms_of(S)
    fig, axs = plt.subplots(len(LS), len(arms), figsize=(2.6 * len(arms), 2.2 * len(LS)),
                            sharex=True, squeeze=False)
    for j, arm in enumerate(arms):
        for i, l in enumerate(LS):
            ax = axs[i, j]
            for st, col in ((0, "#bbbbbb"), (64000, ARM_COLOR[arm])):
                for si, nm in enumerate(seeds):
                    cs = S[nm][st]
                    rd = reading(cs, an, arm)
                    ds = cs.r["diet_stats"]
                    for fam, nms in FAMS.items():
                        pts = [(ds[dn]["mean_outcome"], rd.level["viol"][dn]["mean"][l - 1])
                               for dn in nms if dn in rd.level["viol"]]
                        if len(pts) >= 2:
                            x, y = zip(*pts)
                            ax.plot(x, y, "-", color=col, lw=0.8, alpha=0.8,
                                    marker=SEED_MARK[si], ms=4)
            if i == 0:
                ax.set_title(ARM_NAME[arm])
            if j == 0:
                ax.set_ylabel(f"l={l}  mean V_pre")
            if i == len(LS) - 1:
                ax.set_xlabel("diet E[outcome]")
    fig.suptitle(f"The norm per diet against the world, anchor {an}, violation rows: 64k in "
                 f"colour, step 0 in grey; one line per family, markers = seeds", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_slope(S, an, path):
    """Slope of each diet's V_pre on the arm's own full V_pre, against E[outcome]."""
    plt = _mpl()
    seeds, arms = seeds_of(S), arms_of(S)
    fig, axs = plt.subplots(1, len(LS), figsize=(3.2 * len(LS), 3.0), sharey=True)
    for i, l in enumerate(LS):
        ax = axs[i]
        ax.axhline(1.0, color="#999999", lw=0.6, ls=":")
        for arm in arms:
            for st, ls_ in ((0, "--"), (64000, "-")):
                xs, ys = [], []
                for nm in seeds:
                    cs = S[nm][st]
                    ds = cs.r["diet_stats"]
                    sh = reading(cs, an, arm).shape.get(l, {})
                    for dn in ("out_lo", "out_mid", "out_hi"):
                        if dn in sh:
                            xs.append(ds[dn]["mean_outcome"])
                            ys.append(sh[dn][0])
                if xs:
                    o = np.argsort(xs)
                    ax.plot(np.asarray(xs)[o], np.asarray(ys)[o], ls_, color=ARM_COLOR[arm],
                            lw=1.0, marker="o", ms=4,
                            label=f"{ARM_NAME[arm]} @{st // 1000}k" if i == 0 else None)
        ax.set_title(f"l = {l}")
        ax.set_xlabel("diet E[outcome] (out family)")
        if i == 0:
            ax.set_ylabel("slope of diet V_pre on the arm's full V_pre")
    h, lb = axs[0].get_legend_handles_labels()
    fig.legend(h, lb, fontsize=7, frameon=False, loc="center right")
    fig.subplots_adjust(right=0.86)
    fig.suptitle(f"Shift or rescaling, per arm: the `out` family, anchor {an}; dashed = step 0, "
                 f"solid = 64k; three seeds' points on one line", fontsize=9)
    fig.tight_layout(rect=(0, 0, 0.86, 1))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_disc(S, an, path, readout="V_pre"):
    """Within-cell AUC(-readout, damage) for `full`, per arm and level, step 0 against 64k."""
    plt = _mpl()
    seeds, arms = seeds_of(S), arms_of(S)
    fig, axs = plt.subplots(1, len(LS), figsize=(3.0 * len(LS), 2.8), sharey=True)
    for i, l in enumerate(LS):
        ax = axs[i]
        ax.axhline(0.5, color="#999999", lw=0.6, ls=":")
        for k, arm in enumerate(arms):
            for si, nm in enumerate(seeds):
                v = [disc(S[nm][st], an, arm, l, readout).get("full", (np.nan,))[0]
                     for st in STEPS]
                ax.plot(np.arange(3) + 0.06 * (k - 2), v, "-", color=ARM_COLOR[arm], lw=0.8,
                        marker=SEED_MARK[si], ms=4,
                        label=ARM_NAME[arm] if (i == 0 and si == 0) else None)
        ax.set_xticks(range(3))
        ax.set_xticklabels(["0", "8k", "64k"])
        ax.set_title(f"l = {l}")
        if i == 0:
            ax.set_ylabel(f"within-cell AUC(-{readout}, damage)")
    axs[0].legend(fontsize=6, frameon=False)
    fig.suptitle(f"Discrimination beside calibration: the `full` diet, anchor {an}, matched rows; "
                 f"markers = seeds", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_entropy(S, an, path):
    """b.sdH per diet against E[outcome], per arm, 64k."""
    plt = _mpl()
    seeds, arms = seeds_of(S), arms_of(S)
    fig, axs = plt.subplots(1, len(LS), figsize=(3.0 * len(LS), 2.8), sharey=False)
    for i, l in enumerate(LS):
        ax = axs[i]
        for arm in arms:
            for si, nm in enumerate(seeds):
                cs = S[nm][64000]
                ds = cs.r["diet_stats"]
                p = entropy_family(cs, an, arm, l)
                dn_ = [dn for dn in p if not dn.startswith("_") and dn in ds]
                ax.scatter([ds[dn]["mean_outcome"] for dn in dn_], [p[dn]["bsd"] for dn in dn_],
                           s=8, color=ARM_COLOR[arm], marker=SEED_MARK[si],
                           label=ARM_NAME[arm] if (i == 0 and si == 0) else None)
            oc = [entropy_family(S[nm][64000], an, "", l)["_outcome"]["bsd"] for nm in seeds]
        for v in oc:
            ax.axhline(v, color="#999999", lw=0.6, ls=":")
        ax.set_title(f"l = {l}")
        ax.set_xlabel("diet E[outcome]")
        if i == 0:
            ax.set_ylabel("V_pre ~ H_pre partial slope x sd(H)")
    axs[0].legend(fontsize=6, frameon=False)
    fig.suptitle(f"The level's entropy dependence per diet, 64k, anchor {an}; dotted = the "
                 f"realised outcome's own slope (three seeds)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

PREAMBLE = """# The norm's calibration on the richer readers — tables

Facts only; interpretation is discussed with Jasper and lives nowhere in this folder.
Generated by [`reduce_calibration.py`](../reduce_calibration.py) from
[`calibration.py`](../calibration.py)'s per-diet refits.  Nothing banked is rewritten.  Three
trajectory seeds side by side in every cell, in the order given in each table's caption, **never
averaged**.

**The claim being re-read** ([`norm/README.md`](../../../../README.md) 1-2, three seeds, the ridge
on the raw state only): on identical held-out rows the norm `V_pre` orders with the diet's
expected outcome (Spearman +1.000 inside every family at every level); regressing each diet's
norm on `full`'s gives slope ~1.0 at random init and 1.16 (`out_lo`) -> 0.82 (`out_hi`) at 64k;
the outcome surprise on fixed rows orders inversely with the world.  On the record beside it: on
a sigmoid readout a random trunk rescales as much as a trained one, and `discrim.py` measured
the within-cell damage-ranking gain of the trained trunk over the random one.

**Objects.**  `V[l, d](s_t)` predicts `o_l(t + d)`, the frozen actor being RIGHT at level `l`
at `t + d`; higher is a better world.  `V_pre = V[l, a+1](s_(t0-1))` is the norm, `V = V[l,
a](s_t0)` the post-event level, `R = V - V_pre` the response, `delta = outcome - V_pre` the
outcome surprise.

**Readers**, each fitted per diet on the SAME trunk, diet rows, split, lambda ladder and
held-out per-column lambda selection as `norm/task.py`:

| arm | reader |
|---|---|
| `ridge (banked)` | the state at `post_block7`: `norm/task.py`'s critic, refitted per diet and gated to reproduce every banked column |
| `+log q` | state + `log q_t` (16 dims), standardised to the state's per-dimension RMS on the diet's own training rows -- **primary** |
| `+precision` | state + `H(q_t)` + `H(q_(t-1))` + `s_t` + `H(q_(t-1)) s_t`, standardised the same way |
| `ln_f(s)` | the model's own final layer norm of the state, in place of the state |
| `MLP` | a 128-unit GELU MLP on the state (`train_head`, 2000 steps, no lambda selection), levels 1-4 at a = 0 only |

The clean-only critic's row, where shown, is `express.py`'s `cleanQ` / `cleanP` and
`express_ln`'s `cleanL` (never fitted for the MLP).

**Rows.**  Venue `a1` (the eleven diets exist there only), the `split == 2` test windows,
identical across arms (asserted) and across the three checkpoints of a trajectory.  Anchor `tv`
(the first exactly-impossible token; every row a violation, 1659 rows) and `fd` (the edit onset;
4097 rows, swap / rare / none).  `E[o]` is each diet's mean outcome over the Gram rows it sees
(`diet_stats`); it is a property of the diet and the actor, the same for every arm.

**Flags carried from the record.**  (1) The `mix` family's knob moves the edited exposure with
the expected cost (the quiet share grows as the edit sample shrinks, `norm/diets.py`), so a
`mix` ordering is confounded with exposure.  (2) The two extreme `out` diets fit `V` worse at
l >= 4 (the banked ridge's held-out `R2` 0.037 / 0.055 against `full`'s 0.135 at l = 4, 64k);
their l = 4 rows are weaker, and section 1 prints every arm's fit so the same can be read per
reader.  (3) At a = 0 the outcome surprise is `-V_pre` plus a constant on fixed rows, so its
ordering is the level's restated.  (4) The damage label's base contains the actor's unedited
hit, which moves with the checkpoint, so the matched rows of section 5 are checkpoint-specific
subsets of the identical fixed rows.

**Compute.**  Nine L4 cells (`calib_sweep`, app `ap-8DW07LXqUWOoOxOo3areQ4`), 325-471 s each,
56.7 L4-minutes in all, plus a 202 s smoke; peak RSS 6.2-6.6 GB per cell.  Per cell: actor
25-40 s, forward passes 7-8 s, Grams for all eleven diets 4 s, ridge solves 5 s, MLP heads
268-403 s (88 heads, ~80% of the cell).

Reproduction (from `experiments/`, `MODAL_PROFILE=chromatic`):

```bash
modal run --detach -m rhm.logit_reading.striatum.norm.precision.rereads.calibration.calibration::calib_sweep
D=/v16_s2_L6_m4_distinct/logit_reading
for s in 42 43 44; do for st in 000000 008000 064000; do
  for f in norm_a1.json norm_a1.npz norm_a1_express.npz norm_a1_express_ln.npz \
           norm_a1_calib.json norm_a1_calib.npz; do
    modal volume get rhm-scaling-data $D/traj_a1_s$s/step${st}_$f <dir>/traj_a1_s$s/step${st}_$f
done; done; done
python -m rhm.logit_reading.striatum.norm.precision.rereads.calibration.reduce_calibration \
    <dir>/traj_a1_s42 <dir>/traj_a1_s43 <dir>/traj_a1_s44 \
    --out rhm/logit_reading/striatum/norm/precision/rereads/calibration/results \
    --figs rhm/logit_reading/striatum/norm/precision/rereads/calibration/figs --date 20260922
```
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--sfx", default="_calib")
    ap.add_argument("--out", default=".")
    ap.add_argument("--figs", default="")
    ap.add_argument("--date", default="20260922")
    ap.add_argument("--figs-only", action="store_true")
    args = ap.parse_args()
    S = load_all(args.dirs, args.sfx)
    for nm in S:
        miss = [st for st in STEPS if st not in S[nm]]
        if miss:
            print(f"WARNING {nm}: missing steps {miss}")
    S = {nm: v for nm, v in S.items() if len(v) == len(STEPS)}
    assert S, "no complete trajectory"
    if args.figs:
        os.makedirs(args.figs, exist_ok=True)
        for an in ANCHORS:
            fig_level(S, an, os.path.join(args.figs, f"calib_level_{an}.png"))
            fig_slope(S, an, os.path.join(args.figs, f"calib_slope_{an}.png"))
            fig_disc(S, an, os.path.join(args.figs, f"calib_disc_{an}.png"))
            fig_entropy(S, an, os.path.join(args.figs, f"calib_entropy_{an}.png"))
        print("figures ->", args.figs)
    if args.figs_only:
        return
    parts = [PREAMBLE, "\n---\n\n## 0. Integrity and gates\n", sec_integrity(args.dirs, args.sfx),
             sec_gates(S), sec_repro(S),
             "\n---\n\n## 1. The held-out fit per diet and arm\n",
             sec_fit(S, 1), sec_fit(S, 0)]
    for an in ANCHORS:
        parts += [f"\n---\n\n## 2. The norm per diet — anchor {an}\n",
                  "### 2a. The ordering with the world (the headline)\n",
                  sec_ordering_tab(S, an, "viol", "level")]
        parts += ["### 2b. The level per diet\n"]
        for pk in POPS[an]:
            parts += [sec_level(S, an, pk)]
        if an == "fd":
            parts += ["### 2c. The ordering on the unedited `none` rows\n",
                      sec_ordering_tab(S, an, "none", "level")]
        parts += [f"\n## 3. Shift or rescaling — anchor {an}\n", sec_shape_head(S, an)]
        for arm in arms_of(S):
            parts.append(sec_shape_full(S, an, arm))
        parts += [f"\n## 4. The outcome surprise — anchor {an}\n",
                  "### 4a. Its ordering with the world at a = 0, 4, 8 (violation rows)\n"]
        for ai in (0, 1, 2):
            parts.append(sec_ordering_tab(S, an, "viol", "delta", ai))
        parts += ["### 4b. The values, violation rows, step 0 and 64k\n",
                  sec_delta_tab(S, an, "viol")]
        parts += [f"\n## 5. Discrimination beside calibration — anchor {an}\n",
                  sec_disc_head(S, an), sec_disc(S, an, "V_pre"), sec_disc(S, an, "V"),
                  f"\n## 6. The entropy axis per diet — anchor {an}\n", sec_entropy(S, an)]
    os.makedirs(args.out, exist_ok=True)
    p = os.path.join(args.out, f"tables_{args.date}.md")
    with open(p, "w") as f:
        f.write("\n".join(parts).rstrip() + "\n")
    print(f"wrote {p} ({os.path.getsize(p) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
