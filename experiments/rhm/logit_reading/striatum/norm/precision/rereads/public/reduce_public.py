"""The public re-read, reduced.  Facts only; no interpretation lives here.

Reads `public.py`'s `<stem>_norm_<tag>_public.{json,npz}` and `_public_twins.npz` beside the
banked `norm/` cells and `precision/express.py`'s `_express` cells (and, for seed 42 only,
`orbitofrontal/projection/`'s `_proj_<tag>` cells as the reproduction target), and writes
`results/tables_<date>.md` and `figs/pub_*.png`.

Every reader is read on the IDENTICAL `split == 2` test rows (the public npz carries the base
columns; its matched sets are asserted identical to the banked `_express` ones) and on the
IDENTICAL banked twin pairs (selected by the model's surprisal before any critic exists).
Three trajectory seeds side by side, never averaged.

  python -m rhm.logit_reading.striatum.norm.precision.rereads.public.reduce_public \
      --bank <mirror> --pub <pubmirror> --seeds 42,43,44 \
      --out rhm/logit_reading/striatum/norm/precision/rereads/public/results \
      --figs rhm/logit_reading/striatum/norm/precision/rereads/public/figs --date 20260922

`<mirror>/traj_a1_s4X/` holds the banked `step*_norm_<tag>.{json,npz}` and
`step*_norm_<tag>_express.{json,npz}`; `<pubmirror>/traj_a1_s4X/` holds this node's
`step*_norm_<tag>_public*` files, `public.py::basis_sweep`'s `step*_readout_basis.npz` and
(s42) the `step*_proj_<tag>*` files (`results/fetch.sh` fetches all of them).  About 10 min
on CPU for three seeds.
"""

import argparse
import json
import os

import numpy as np

from rhm.logit_reading.striatum.analyze import _auc, _cond_auc, fmt, tbl
from rhm.logit_reading.striatum.norm.analyze import Twins, _ols, _sem, _spear
from rhm.logit_reading.striatum.norm.precision.reduce import (
    STEPS, LS, cell_of, matched_sets, _ols_multi, _partial, _qbin, _resid, _z)
from rhm.logit_reading.orbitofrontal.projection.analyze import (ridge_cv, _r2, Pairs,
                                                                priced_summary)

TAGS = ["swap65k", "a1"]
ANCS = ["tv", "fd"]
FAMILIES = [("out", ["out_lo", "out_mid", "out_hi"]), ("dmg", ["dmg_lo", "dmg_mid", "dmg_hi"]),
            ("mix", ["mix_q00", "mix_q25", "mix_q51"])]
DIET_ORDER = ["full", "natural_sized", "out_lo", "out_mid", "out_hi", "dmg_lo", "dmg_mid",
              "dmg_hi", "mix_q00", "mix_q25", "mix_q51"]
# projection/results/tables.md, "The priced twin by checkpoint" (s42, critic `full`, dlogq@0)
PROJ_PRICED = {("swap65k", 0): [0.6036, 0.5987, 0.5862, 0.4784],
               ("swap65k", 8000): [0.6099, 0.3108, 0.1179, 0.0825],
               ("swap65k", 64000): [0.5371, 0.3346, 0.1993, 0.0897],
               ("a1", 0): [0.6302, 0.5786, 0.6564, 0.4769],
               ("a1", 8000): [0.6303, 0.3182, 0.1787, 0.0504],
               ("a1", 64000): [0.4953, 0.3338, 0.1448, 0.0312]}
LABEL = {"full": "ridge (state)", "fullQ": "+log q (fullQ)", "fullQst": "fullQ: state part",
         "fullQbl": "fullQ: belief part", "fullP": "+precision (fullP)",
         "fullPst": "fullP: state part", "fullPbl": "fullP: appended part",
         "fullM": "MLP (state)", "fullQM": "MLP (state, log q)", "lq_full": "ridge (log q)",
         "mq_full": "MLP (log q)", "fullMr": "MLP (state), retrained"}

BANK, PUB, SEEDS, NAMES = None, None, None, None


# ---------------------------------------------------------------------------
# paths and cached loads
# ---------------------------------------------------------------------------

def bp(s, tag, st):
    return os.path.join(BANK, f"traj_a1_s{s}", f"step{st:06d}_norm_{tag}.npz")


def bj(s, tag, st):
    return os.path.join(BANK, f"traj_a1_s{s}", f"step{st:06d}_norm_{tag}.json")


def xp(s, tag, st):
    return os.path.join(BANK, f"traj_a1_s{s}", f"step{st:06d}_norm_{tag}_express.npz")


def xj(s, tag, st):
    return os.path.join(BANK, f"traj_a1_s{s}", f"step{st:06d}_norm_{tag}_express.json")


def pp(s, tag, st):
    return os.path.join(PUB, f"traj_a1_s{s}", f"step{st:06d}_norm_{tag}_public.npz")


def pj(s, tag, st):
    return os.path.join(PUB, f"traj_a1_s{s}", f"step{st:06d}_norm_{tag}_public.json")


def pt(s, tag, st):
    return os.path.join(PUB, f"traj_a1_s{s}", f"step{st:06d}_norm_{tag}_public_twins.npz")


def bas(s, st):
    """`public.py::basis_sweep`'s readout basis (the cells' own `basis__Vb` is not the basis)."""
    return os.path.join(PUB, f"traj_a1_s{s}", f"step{st:06d}_readout_basis.npz")


def prj(s, tag, st, part=""):
    return os.path.join(PUB, f"traj_a1_s{s}", f"step{st:06d}_proj_{tag}{part}")


_J = {}


def jload(p):
    if p not in _J:
        _J[p] = json.load(open(p)) if os.path.exists(p) else None
    return _J[p]


def have(s, tag, st):
    return all(os.path.exists(f) for f in (bp(s, tag, st), xp(s, tag, st), pp(s, tag, st),
                                           pj(s, tag, st)))


def per_seed(fn):
    return " / ".join(fn(s) for s in SEEDS)


class Rd:
    """One (seed, venue, step, anchor): the banked, express and public columns on the same
    test rows, with the matched sets."""

    def __init__(self, s, tag, st, an):
        self.ok = have(s, tag, st)
        if not self.ok:
            return
        self.C = cell_of(xp(s, tag, st), an)              # express: base cols + arms
        self.P = cell_of(pp(s, tag, st), an)              # public: base cols + new readers
        self.B = cell_of(bp(s, tag, st), an)              # banked norm: diets for the state
        self.ok = self.C.ok and self.P.ok
        if not self.ok:
            return
        _, self.mm = matched_sets(xp(s, tag, st), an)
        et, tvv = self.C.et, self.C.g("t_v")
        self.viol = (et == 0) & (tvv > 0)
        self.none = et == 2

    def src(self, crit):
        if crit in ("full", "fullQ", "fullP", "fullM", "fullH"):
            return self.C
        return self.P

    def has(self, crit, l):
        return self.src(crit).g(f"R_{crit}_l{l}_a0") is not None

    def R(self, crit, l):
        return self.src(crit).R(crit, l)

    def V(self, crit, l):
        return self.src(crit).V(crit, l)

    def Vpre(self, crit, l):
        return self.src(crit).Vpre(crit, l)

    def Ro(self, crit, l):
        return self.src(crit).Ro(crit, l)


_RD = {}


def rd(s, tag, st, an):
    k = (s, tag, st, an)
    if k not in _RD:
        _RD[k] = Rd(s, tag, st, an)
    return _RD[k]


# ---------------------------------------------------------------------------
# G. gates
# ---------------------------------------------------------------------------

def sec_integrity(files):
    rows = []
    for f in files:
        if not os.path.exists(f):
            continue
        try:
            Z = np.load(f)
            n = 0
            for k in Z.files:
                a = Z[k]
                n += 1
                if a.dtype.kind in "fiub":
                    float(np.nansum(a.astype(np.float64)))
            rows.append([os.path.relpath(f, os.path.dirname(os.path.dirname(f))), str(n),
                         "ok"])
        except Exception as e:                                 # pragma: no cover
            rows.append([f, "--", f"FAILED {type(e).__name__}: {e}"])
    nf = sum(r[2] != "ok" for r in rows)
    return "\n".join([
        f"**Integrity: every member of every npz this reduction reads, decompressed and summed "
        f"({len(rows)} files, {nf} failures).**", "",
        tbl(rows, ["file", "members", "status"]), ""])


def sec_cell_gates():
    rows = []
    for s in SEEDS:
        for tag in TAGS:
            for st in STEPS:
                r = jload(pj(s, tag, st))
                if r is None:
                    rows.append([f"s{s}", tag, st] + ["missing"] * 9)
                    continue
                g = r["gates"]
                mx = lambda dct, pre="": max([float(v) for k, v in dct.items()
                                              if k.startswith(pre) and not isinstance(v, bool)]
                                             or [0.0])
                rows_ok = all(v for k, v in g["rows"].items() if isinstance(v, bool))
                colsx = lambda crit: max(float(g["cols"].get(f"{an}/{crit}", 0.0))
                                         for an in ANCS)
                rows.append([
                    f"s{s}", tag, st, f"{mx(g['actor']):.1e}", f"{mx(g['val_r2']):.1e}",
                    str(rows_ok) + f" / {max(float(g['rows'].get(f'{an}/nll_e', 0)) for an in ANCS):.0e}",
                    f"{colsx('full'):.1e}", f"{colsx('fullQ'):.1e}", f"{colsx('fullP'):.1e}",
                    " ".join(f"{float(g['cols'].get(f'twins/{c}', float('nan'))):.0e}"
                             for c in ("full", "fullQ", "fullP")),
                    str(all((v is True) or (not isinstance(v, bool) and v < 1e-6)
                            for v in g["diets"].values())) + f" ({len(g['diets']) // 2})",
                    "PASS" if not r["gate_fails"] else f"FAIL {r['gate_fails']}"])
    out = ["**In-container gates, per cell (`public.py`, asserted before the cell returns).**  "
           "`actor`: max |accuracy - banked| over levels; `val R2`: max |held-out R2 - banked| "
           "over arm 0, the clean critic and `fullQ` / `fullP` against the `_express` json; "
           "`rows`: test rows and every outcome label exact, then max |nll_e - banked|; "
           "`full` / `fullQ` / `fullP`: max |V, Vpre, R, Ro, Vpreo - banked `_express` column| "
           "over both anchors and l = 1-4 (for `fullQ` / `fullP` this is the state part PLUS the "
           "appended part against the banked arm); `twins`: the same at offsets 0 / -1 on every "
           "banked pair; `diets`: row counts equal and mean outcomes within 1e-6 of the banked "
           "json (number of diets).", "",
           tbl(rows, ["seed", "venue", "step", "actor", "val R2", "rows exact / nll",
                      "full", "fullQ", "fullP", "twins full fullQ fullP", "diets", "verdict"]),
           ""]
    rows = []
    for s in SEEDS:
        for tag in TAGS:
            for st in STEPS:
                r = jload(pj(s, tag, st))
                if r is None:
                    continue
                g = r["gates"]["twins"]
                rep = r["report"]["fullMr_vs_banked_fullM"]
                vv = [x for k, x in rep.items() if isinstance(x, dict)]
                rows.append([f"s{s}", tag, st, f"{g.get('s_v', float('nan')):.1e}",
                             f"{g.get('s_c', float('nan')):.1e}",
                             f"{g.get('pre_state_identical', float('nan')):.1e}",
                             f"{r['gram_guard_logq_full_rel']:.1e}",
                             fmt(max(x["max_abs_V"] for x in vv), 4) if vv else "--",
                             fmt(min(x["corr_V"] for x in vv), 5) if vv else "--",
                             fmt(float(np.median([x["sd_V"] for x in vv])), 4) if vv else "--",
                             f"{r['timing_s'].get('mlp_logq', 0):.0f}",
                             f"{r['peak_rss_gb']:.1f}"])
    out += ["**Twin and recipe guards.**  `s_v`, `s_c`: max |recomputed surprisal of the "
            "violator / twin token - banked `tw__s_v` / `tw__s_c`|; `prefix`: max |state at "
            "`t_v - 1`, violator - twin| (identical prefix, so `V_pre` cancels for every reader); "
            "`gram`: relative max |log q ridge's `full` Gram - the sub-block of the augmented "
            "Gram|.  `fullM` retrained with express's recipe and seeds against the banked "
            "`_express` `fullM` columns (reported, not asserted): max |V diff| and min corr over "
            "anchors and levels, beside the median sd of the banked `V`.  `mlp s`: wall seconds "
            "for the log-q MLPs; `RSS`: peak GB.", "",
            tbl(rows, ["seed", "venue", "step", "s_v", "s_c", "prefix", "gram",
                       "fullM max|dV|", "fullM min corr", "sd V", "mlp s", "RSS"]), ""]
    return "\n".join(out)


def sec_red_gates():
    """Matched sets identical; the exact split; log q ridge and span vs projection (s42)."""
    rows = []
    for s in SEEDS:
        for tag in TAGS:
            for st in STEPS:
                if not have(s, tag, st):
                    continue
                for an in ANCS:
                    _, mx = matched_sets(xp(s, tag, st), an)
                    _, mp_ = matched_sets(pp(s, tag, st), an)
                    same = (set(mx) == set(mp_)) and all(
                        np.array_equal(mx[l][0], mp_[l][0]) for l in mx)
                    R = rd(s, tag, st, an)
                    worst = 0.0
                    for arm in ("fullQ", "fullP"):
                        for l in LS:
                            for kind in ("R", "V", "Vpre", "Ro"):
                                tot = R.C.g(f"{kind}_{arm}_l{l}_a0").astype(np.float64)
                                sm = (R.P.g(f"{kind}_{arm}st_l{l}_a0").astype(np.float64)
                                      + R.P.g(f"{kind}_{arm}bl_l{l}_a0").astype(np.float64))
                                worst = max(worst, float(np.abs(tot - sm).max()))
                    rows.append([f"s{s}", tag, st, an,
                                 " / ".join(str(len(mx[l][0])) for l in LS if l in mx),
                                 str(same), f"{worst:.1e}"])
    out = ["**Matched sets and the exact split.**  `junction`'s matcher on the public npz's "
           "base columns against the same matcher on the banked `_express` npz (matched n per "
           "level 1-4); `st + bl`: max |state part + appended part - the banked arm's column| "
           "over `R`, `V`, `Vpre`, `Ro`, l = 1-4, `fullQ` and `fullP` (float32 storage).", "",
           tbl(rows, ["seed", "venue", "step", "anchor", "matched n", "identical", "st + bl"]),
           ""]
    # the log q ridge against projection's banked s42 `logq` block
    rows = []
    s = "42"
    for tag in TAGS:
        for st in STEPS:
            for an in ANCS:
                f = prj(s, tag, st, f"_rows_{an}.npz")
                if not (os.path.exists(f) and have(s, tag, st)):
                    continue
                Z = np.load(f)
                R = rd(s, tag, st, an)
                assert np.array_equal(np.asarray(Z[f"{an}__w"]), R.P.g("w")), "rows differ"
                pj_ = jload(pj(s, tag, st))
                worst, nd = 0.0, 0
                for nm in pj_["diets"]:
                    for l in LS:
                        for kind in ("V", "Vpre", "R"):
                            k = f"{an}__{kind}_logq|{nm}_l{l}_a0"
                            if k not in Z.files:
                                continue
                            nd += 1
                            worst = max(worst, float(np.abs(
                                Z[k].astype(np.float64)
                                - R.P.g(f"{kind}_lq_{nm}_l{l}_a0").astype(np.float64)).max()))
                rows.append([tag, st, an, str(len(pj_["diets"])), str(nd), f"{worst:.1e}"])
    out += ["**The linear public critic against the banked one** (`orbitofrontal/projection/`'s "
            "`logq` block, s42, the only seed it ran on): max |V, Vpre, R - banked| over every "
            "diet and l = 1-4 on the identical rows (asserted identical).", "",
            tbl(rows, ["venue", "step", "anchor", "diets", "columns", "max |diff|"]), ""]
    rows = []
    for tag in TAGS:
        for st in STEPS:
            pr = jload(prj(s, tag, st, ".json"))
            r = jload(pj(s, tag, st))
            if pr is None or r is None:
                continue
            sp0, sp1 = pr["readout_span"], r["span"]
            ks = [f"{c}/l{l}_d{dd}" for c in ("full", "clean") for l in LS for dd in (0, 1)]
            d_ = [abs(sp0[k] - sp1[k]) for k in ks if k in sp0]
            rows.append([tag, st, str(len(d_)), f"{max(d_):.1e}",
                         fmt(float(np.mean([sp1[k] for k in ks])), 4),
                         fmt(sp1["_random_direction"], 4)])
    out += ["**The readout span of the ridge against projection's banked value (s42)**: "
            "max |span - banked| over the `full` and `clean` columns l = 1-4, d = 0, 1.", "",
            tbl(rows, ["venue", "step", "columns", "max |diff|", "mean", "chance"]), ""]
    rows = []
    for s_ in SEEDS:
        for st in STEPS:
            if not os.path.exists(bas(s_, st)):
                continue
            Vb = np.asarray(np.load(bas(s_, st))["Vb"], np.float64)
            Pm = Vb.T @ Vb
            orth = float(np.abs(Vb @ Vb.T - np.eye(len(Vb))).max())
            cells = [f"s{s_}", st, str(len(Vb)), f"{orth:.1e}"]
            pz = prj(s_, "a1", st, "_twins.npz")
            if s_ == "42" and os.path.exists(pz):
                Vp = np.asarray(np.load(pz)["basis__Vb"], np.float64)
                cells.append(f"{np.abs(Vp.T @ Vp - Pm).max():.1e}")
            else:
                cells.append("--")
            worst = 0.0
            for tag in TAGS:
                r = jload(pj(s_, tag, st))
                if r is None or not os.path.exists(pp(s_, tag, st)):
                    continue
                Z = np.load(pp(s_, tag, st))
                for crit in ("full", "fullQ", "fullP"):
                    B = np.asarray(Z[f"beta__{crit}"], np.float64)
                    for l in LS:
                        for dd in (0, 1):
                            b = B[:Vb.shape[1], r["names"].index(f"l{l}_d{dd}")]
                            sp = np.linalg.norm(Vb @ b) / np.linalg.norm(b)
                            worst = max(worst, abs(sp - r["span"][f"{crit}/l{l}_d{dd}"]))
            cells.append(f"{worst:.1e}")
            rows.append(cells)
    out += ["**The readout basis** (`public.py::basis_sweep`, one CPU pass per checkpoint; the "
            "cells' own `basis__Vb` member was overwritten by a per-row column before saving "
            "and is not read): rank, max |Vb Vb' - I|, max |projector - projection's banked "
            "s42 projector| (from `_proj_a1_twins.npz`), and max |span recomputed here from the "
            "saved coefficients - the cell json's span| over `full` / `fullQ` / `fullP`, "
            "l = 1-4, d = 0, 1.", "",
            tbl(rows, ["seed", "step", "rank", "orthonormal", "vs projection", "span recomputed"]),
            ""]
    return "\n".join(out)


# ---------------------------------------------------------------------------
# 1. the exact decomposition of the augmented critic's response
# ---------------------------------------------------------------------------

def _mm(R, l):
    if R is None or not R.ok or l not in R.mm:
        return None, None
    i, lab = R.mm[l]
    return i, lab[i]


def sec_amp(tag, an, arm="fullQ"):
    crits = ["full", arm, f"{arm}st", f"{arm}bl"]
    rows = []
    for l in LS:
        for st in STEPS:
            cells = [l, st]
            for c in crits:
                def fn(s, c=c):
                    R = rd(s, tag, st, an)
                    if not R.ok or not R.has(c, l):
                        return "--"
                    x = R.R(c, l)[R.viol]
                    return f"{x.mean():+.4f}"
                cells.append(per_seed(fn))

            def vs(s):
                R = rd(s, tag, st, an)
                if not R.ok:
                    return "--"
                a_, b_ = R.R(f"{arm}st", l)[R.viol], R.R(f"{arm}bl", l)[R.viol]
                t_ = a_ + b_
                vt = max(t_.var(), 1e-18)
                return f"{a_.var() / vt:.2f}:{b_.var() / vt:.2f}:{np.corrcoef(a_, b_)[0, 1]:+.2f}"
            cells.append(per_seed(vs))

            def cr(s):
                R = rd(s, tag, st, an)
                if not R.ok:
                    return "--"
                r0 = R.R("full", l)[R.viol]
                return ":".join(f"{np.corrcoef(R.R(c, l)[R.viol], r0)[0, 1]:+.2f}"
                                for c in (arm, f"{arm}st", f"{arm}bl"))
            cells.append(per_seed(cr))
            cells.append(per_seed(lambda s: str(int(rd(s, tag, st, an).viol.sum()))
                                  if rd(s, tag, st, an).ok else "--"))
            rows.append(cells)
    return "\n".join([
        f"**The response's amplitude and its exact split — `{arm}`, {tag}, anchor {an}, a = 0, "
        f"mean `R` on the fixed violation rows; {' / '.join(NAMES)} side by side.**  "
        f"`R = R_state + R_appended` per row exactly (state part carries the intercept and the "
        f"appended block's training mean).  `var` is `var(R_st)/var(R) : var(R_bl)/var(R) : "
        f"corr(R_st, R_bl)` on the same rows; `corr w/ ridge` is the correlation of `R`, "
        f"`R_st` and `R_bl` with the ridge's `R` on the same rows.", "",
        tbl(rows, ["l", "step"] + [LABEL[c] for c in crits] + ["var st:bl:corr",
                                                               "corr w/ ridge R:st:bl",
                                                               "n viol"]),
        ""])


def sec_part_auc(tag, an, arm="fullQ", kind="R"):
    crits = ["full", arm, f"{arm}st", f"{arm}bl", "fullM"]
    rows = []
    for l in LS:
        for st in STEPS:
            cells = [l, st]
            for c in crits:
                def fn(s, c=c):
                    R = rd(s, tag, st, an)
                    i, y = _mm(R, l)
                    if i is None or not R.has(c, l):
                        return "--"
                    sc = (R.R(c, l) if kind == "R" else R.V(c, l))[i]
                    return fmt(_auc(-sc, y))
                cells.append(per_seed(fn))
            cells.append(per_seed(lambda s: (str(len(_mm(rd(s, tag, st, an), l)[0]))
                                             if _mm(rd(s, tag, st, an), l)[0] is not None
                                             else "--")))
            rows.append(cells)
    what = "response `-R`" if kind == "R" else "post-event level `-V`"
    return "\n".join([
        f"**Realised-damage AUC of the {what} and of each part — `{arm}`, {tag}, anchor {an}, "
        f"matched `flip` rows (junction's matcher); {' / '.join(NAMES)} side by side.**", "",
        tbl(rows, ["l", "step"] + [LABEL[c] for c in crits] + ["n matched"]), ""])


def sec_part_slope(tag, an, arm="fullQ", withV=False):
    crits = ["full", arm, f"{arm}st", f"{arm}bl", "fullM"]
    rows = []
    for l in LS:
        for st in STEPS:
            cells = [l, st]
            for c in crits:
                def fn(s, c=c):
                    R = rd(s, tag, st, an)
                    i, _ = _mm(R, l)
                    if i is None or not R.has(c, l):
                        return "--"
                    C = R.C
                    vref = c if c in ("full", "fullM") else arm
                    ex = [R.Vpre(vref, l)[i]] if withV else None
                    b, se, _ = _partial(R.R(c, l)[i], C.H[i], C.ctrl(extra=ex, idx=i))
                    return (fmt(b * float(C.H[i].std()), 4)
                            + (f" ({b / se:.1f})" if np.isfinite(se) and se > 0 else ""))
                cells.append(per_seed(fn))
            rows.append(cells)
    what = (f"given `V_pre` (the reader's own; for `{arm}`'s parts, `{arm}`'s total `V_pre`)"
            if withV else "without `V_pre`")
    return "\n".join([
        f"**`R ~ H_pre` at matched surprisal, {what} — `{arm}` and its parts, {tag}, anchor "
        f"{an}, matched `flip` rows; {' / '.join(NAMES)} side by side.**  Partial slope "
        f"residualised on `(k*, j, t0, nll_e)`" + (" plus `V_pre`" if withV else "")
        + ", times sd of `H_pre`, `(t)`.  The two parts' slopes sum to the total's exactly "
        "(a partial regression is linear in the response).", "",
        tbl(rows, ["l", "step"] + [LABEL[c] for c in crits]), ""])


def sec_part_inter(tag, an, arm="fullQ"):
    crits = ["full", arm, f"{arm}st", f"{arm}bl", "fullM"]
    rows = []
    for l in LS:
        for st in STEPS:
            cells = [l, st]
            for c in crits:
                def fn(s, c=c):
                    R = rd(s, tag, st, an)
                    i, _ = _mm(R, l)
                    if i is None or not R.has(c, l):
                        return "--"
                    C = R.C
                    zs, zh = _z(C.s[i]), _z(C.H[i])
                    X = np.stack([zs, zh, zs * zh, C.ks[i].astype(float), C.jj[i].astype(float),
                                  C.t0[i].astype(float)], 1)
                    cf, se, _ = _ols_multi(R.R(c, l)[i], X)
                    return fmt(cf[2], 4) + (f" ({cf[2] / se[2]:.1f})" if se[2] > 0 else "")
                cells.append(per_seed(fn))
            rows.append(cells)
    return "\n".join([
        f"**The `z(s) x z(H)` interaction on each part — `{arm}`, {tag}, anchor {an}, matched "
        f"`flip` rows; {' / '.join(NAMES)} side by side.**  `precision/reduce.py` 3's fit "
        f"(`z(s) + z(H) + z(s).z(H)` with `(k*, j, t0)` linear); the interaction coefficient "
        f"and `(t)`; additive across the parts.", "",
        tbl(rows, ["l", "step"] + [LABEL[c] for c in crits]), ""])


def _cond2(x, a, b, lab, nbin=5):
    """AUC of x inside the joint quantile bins of (a, b), pooled with weight n_pos*n_neg."""
    key = _qbin(a, nbin) * nbin + _qbin(b, nbin)
    num = den = 0.0
    for kk in np.unique(key):
        m = key == kk
        n1, n0 = lab[m].sum(), (~lab[m]).sum()
        if n1 < 5 or n0 < 5:
            continue
        num += n1 * n0 * _auc(x[m], lab[m])
        den += n1 * n0
    return float(num / den) if den else float("nan")


def sec_part_contain(tag, an, arm="fullQ"):
    reads = ["H marginal", "H | R", "H | R_st", "H | R_bl", "H | R_st, R_bl (5x5)",
             "H resid on R_st, R_bl", "H | V", "H resid on V, R", "-R_st | H", "-R_bl | H"]
    rows = []
    for l in LS:
        for st in STEPS:
            for what in reads:
                def fn(s, what=what):
                    R = rd(s, tag, st, an)
                    i, y = _mm(R, l)
                    if i is None or not R.has(arm, l):
                        return "--"
                    H = R.C.H[i]
                    Rt, Rs, Rb = R.R(arm, l)[i], R.R(f"{arm}st", l)[i], R.R(f"{arm}bl", l)[i]
                    V = R.V(arm, l)[i]
                    if what == "H marginal":
                        return fmt(_auc(H, y))
                    if what == "H | R":
                        return fmt(_cond_auc(H, Rt, y, 5))
                    if what == "H | R_st":
                        return fmt(_cond_auc(H, Rs, y, 5))
                    if what == "H | R_bl":
                        return fmt(_cond_auc(H, Rb, y, 5))
                    if what == "H | R_st, R_bl (5x5)":
                        return fmt(_cond2(H, Rs, Rb, y, 5))
                    if what == "H resid on R_st, R_bl":
                        return fmt(_auc(_resid(H, np.stack([Rs, Rb], 1)), y))
                    if what == "H | V":
                        return fmt(_cond_auc(H, V, y, 5))
                    if what == "H resid on V, R":
                        return fmt(_auc(_resid(H, np.stack([V, Rt], 1)), y))
                    if what == "-R_st | H":
                        return fmt(_cond_auc(-Rs, H, y, 5))
                    return fmt(_cond_auc(-Rb, H, y, 5))
                rows.append([l, st, what, per_seed(fn)])
    return "\n".join([
        f"**Does `H_pre`'s damage reading vanish given each part? — `{arm}`, {tag}, anchor "
        f"{an}, a = 0, matched `flip` rows; {' / '.join(NAMES)} side by side.**  "
        f"`precision` E4's conditioning per part: `AUC(H_pre, flip)` inside 5 quantile bins "
        f"of the named column (pooled with weight `n_pos * n_neg`, `_cond_auc`), inside the "
        f"5 x 5 joint bins of both parts, and after linear residualisation; the last two rows "
        f"are the reverse (each part's `-R` inside bins of `H_pre`).  `H | R`, `H | V` and "
        f"`H resid on V, R` are E4's own rows for `{arm}` (reproduced here as a check).", "",
        tbl(rows, ["l", "step", "reading", " / ".join(NAMES)]), ""])


# ---------------------------------------------------------------------------
# 2. the nonlinear public reader
# ---------------------------------------------------------------------------

def sec_fit(tag):
    rds = [("full", "bank"), ("fullQ", "x"), ("fullM", "x"), ("fullQM", "p"),
           ("lq_full", "p"), ("mq_full", "p"), ("fullMr", "p")]
    rows = []
    for st in STEPS:
        for l in LS:
            cells = [st, l]
            for c, where in rds:
                def fn(s, c=c, where=where):
                    if where == "bank":
                        r = jload(bj(s, tag, st))
                        return fmt(r["critic_val_r2"]["full"].get(f"l{l}_d0"), 4) if r else "--"
                    r = jload(xj(s, tag, st) if where == "x" else pj(s, tag, st))
                    if r is None or c not in r["val_r2"]:
                        return "--"
                    return fmt(r["val_r2"][c].get(f"l{l}_d0"), 4)
                cells.append(per_seed(fn))
            rows.append(cells)
    return "\n".join([
        f"**Held-out fit `R2` of `V[l, d=0]` per reader — {tag}, `full` diet; "
        f"{' / '.join(NAMES)} side by side.**  The same validation rows for every reader; the "
        f"ridges' lambda is selected on them per column, the MLPs (`train_head`, 2000 steps, "
        f"no selection) are only reported on them.", "",
        tbl(rows, ["step", "l"] + [LABEL[c] for c, _ in rds]), ""])


def _vpre_state(R, nm, l):
    v = R.B.g(f"Vpre_{nm}_l{l}_a0")
    return None if v is None else v.astype(np.float64)


def _vpre_reader(R, reader, nm, l):
    if reader == "state":
        return _vpre_state(R, nm, l)
    v = R.P.g(f"Vpre_{reader}_{nm}_l{l}_a0")
    return None if v is None else v.astype(np.float64)


def sec_calib(tag="a1", an="tv", st=64000):
    out = []
    readers = [("state", "ridge (state), banked"), ("lq", "ridge (log q)"),
               ("mq", "MLP (log q)")]
    for reader, lab in readers:
        rows = []
        for nm in DIET_ORDER:
            cells = [nm]
            def eo(s, nm=nm):
                r = jload(bj(s, tag, st))
                return fmt(r["diet_stats"][nm]["mean_outcome"], 4) if r and nm in r[
                    "diet_stats"] else "--"
            cells.append(per_seed(eo))
            for l in LS:
                def fn(s, nm=nm, l=l):
                    R = rd(s, tag, st, an)
                    if not R.ok:
                        return "--"
                    v = _vpre_reader(R, reader, nm, l)
                    return fmt(float(v[R.viol].mean()), 4) if v is not None else "--"
                cells.append(per_seed(fn))
            rows.append(cells)
        out += [f"**{lab} — the norm `V_pre` per diet on the fixed violation rows ({tag}, "
                f"anchor {an}, step {st}); {' / '.join(NAMES)} side by side.**", "",
                tbl(rows, ["diet", "diet E[o]"] + [f"l={l}" for l in LS]), ""]
    return "\n".join(out)


def sec_calib_rank(tag="a1", an="tv"):
    rows = []
    readers = [("state", "ridge (state)"), ("lq", "ridge (log q)"), ("mq", "MLP (log q)")]
    for st in STEPS:
        for reader, lab in readers:
            for fam, mem in FAMILIES + [("all", DIET_ORDER)]:
                for l in LS:
                    def rho(s):
                        R = rd(s, tag, st, an)
                        r = jload(bj(s, tag, st))
                        if not R.ok or r is None:
                            return "--"
                        eo, vv = [], []
                        for nm in mem:
                            v = _vpre_reader(R, reader, nm, l)
                            if v is None or nm not in r["diet_stats"]:
                                return "--"
                            eo.append(r["diet_stats"][nm]["mean_outcome"])
                            vv.append(float(v[R.viol].mean()))
                        return f"{_spear(eo, vv):+.3f} ({max(vv) - min(vv):.3f})"
                    rows.append([st, lab, fam, l, per_seed(rho)])
    return "\n".join([
        f"**The norm's rank order against the diet's expected outcome, per reader — {tag}, "
        f"anchor {an}, fixed violation rows; {' / '.join(NAMES)} side by side.**  Spearman "
        f"`rho(V_pre, E[o])` inside each family (and over all 11 diets), with the spread "
        f"`max - min` of the family's mean `V_pre` in parentheses.", "",
        tbl(rows, ["step", "reader", "family", "l", "rho (spread)"]), ""])


def sec_shape(tag="a1", an="tv", l=1):
    rows = []
    readers = [("state", "ridge (state)"), ("lq", "ridge (log q)"), ("mq", "MLP (log q)")]
    for st in STEPS:
        for reader, lab in readers:
            for nm in DIET_ORDER[1:]:
                def fn(s, nm=nm):
                    R = rd(s, tag, st, an)
                    if not R.ok:
                        return "--"
                    x = _vpre_reader(R, reader, "full", l)
                    y = _vpre_reader(R, reader, nm, l)
                    if x is None or y is None:
                        return "--"
                    b, c, r2 = _ols(x, y)
                    return f"{b:.3f} ({c:+.3f}, {r2:.3f})"
                rows.append([st, lab, nm, per_seed(fn)])
    return "\n".join([
        f"**Shift or rescaling, per reader — each diet's `V_pre` regressed on `full`'s, level "
        f"{l}, every fixed row at anchor {an} ({tag}); {' / '.join(NAMES)} side by side.**  "
        f"`slope (intercept, R2)`; slope 1 with a non-zero intercept is a pure level shift.  "
        f"`norm/README.md` 1 banked the state's `out_lo` / `out_hi` slopes 1.161 / 0.816 "
        f"(s42, 64k) and `projection/` the log q ridge's 1.126 / 0.844.", "",
        tbl(rows, ["step", "reader", "diet", "slope (intercept, R2)"]), ""])


def sec_response(tag, an):
    crits = ["full", "lq_full", "mq_full", "fullM", "fullQM", "fullQ"]
    out = []
    for l in LS:
        rows = []
        for st in STEPS:
            for c in crits:
                def amp(s, c=c):
                    R = rd(s, tag, st, an)
                    if not R.ok or not R.has(c, l):
                        return "--"
                    x = R.R(c, l)[R.viol]
                    return f"{x.mean():+.4f} ({x.mean() / max(_sem(x), 1e-12):.0f})"

                def rsd(s, c=c):
                    R = rd(s, tag, st, an)
                    if not R.ok or not R.has(c, l):
                        return "--"
                    Ro = R.Ro(c, l)
                    sc = float(Ro[R.none].std()) if R.none.sum() > 20 else float(Ro.std())
                    return f"{R.R(c, l)[R.viol].mean() / max(sc, 1e-12):+.3f}"

                def dro(s, c=c):
                    R = rd(s, tag, st, an)
                    if not R.ok or not R.has(c, l):
                        return "--"
                    return f"{(R.R(c, l) - R.Ro(c, l))[R.viol].mean():+.4f}"

                def auc(s, c=c, k="R"):
                    R = rd(s, tag, st, an)
                    i, y = _mm(R, l)
                    if i is None or not R.has(c, l):
                        return "--"
                    sc = (R.R(c, l) if k == "R" else R.V(c, l))[i]
                    return fmt(_auc(-sc, y))
                rows.append([st, LABEL[c], per_seed(amp), per_seed(rsd), per_seed(dro),
                             per_seed(auc), per_seed(lambda s, c=c: auc(s, c, "V"))])
        out += [f"**Level {l}: the response per reader — {tag}, anchor {an}, a = 0; "
                f"{' / '.join(NAMES)} side by side.**  `mean R (t)` on the fixed violation "
                f"rows; `R / sd` divides by the sd of the unedited stream's `Ro` on `none` rows "
                f"(on all rows where there are none, as `projection/`); `R - Ro` on the "
                f"violation rows; `AUC(-R, flip)` and `AUC(-V, flip)` on the matched rows.", "",
                tbl(rows, ["step", "reader", "mean R (t)", "R / sd", "R - Ro",
                           "AUC(-R, flip)", "AUC(-V, flip)"]), ""]
    return "\n".join(out)


def sec_rand16(tag="a1", an="tv"):
    """projection's dimension-matched references on the same matched rows (s42 only)."""
    rows = []
    s = "42"
    for st in STEPS:
        f = prj(s, tag, st, f"_rows_{an}.npz")
        if not (os.path.exists(f) and have(s, tag, st)):
            continue
        Z = np.load(f)
        R = rd(s, tag, st, an)
        for l in LS:
            i, y = _mm(R, l)
            if i is None:
                continue
            cells = [st, l]
            for b in ("state", "logq", "rand16", "pca16", "logq_rich", "pub_hist"):
                k = f"{an}__R_{b}|full_l{l}_a0"
                cells.append(fmt(_auc(-np.asarray(Z[k], np.float64)[i], y))
                             if k in Z.files else "--")
            cells.append(fmt(_auc(-R.R("mq_full", l)[i], y)) if R.has("mq_full", l) else "--")
            rows.append(cells)
    return "\n".join([
        f"**The banked dimension-matched references on the same matched rows — `AUC(-R, flip)`, "
        f"{tag}, anchor {an}, s42 (the only seed `projection/` ran).**  Columns from the "
        f"`_proj_{tag}_rows_{an}.npz` cells; the last column is this node's MLP on log q.", "",
        tbl(rows, ["step", "l", "state", "logq", "rand16", "pca16", "logq_rich", "pub_hist",
                   "MLP (log q)"]), ""])


# ---------------------------------------------------------------------------
# twins: the pairs, the optimism per reader, and the priced map
# ---------------------------------------------------------------------------

class TW:
    """The banked pairs of one cell with every reader's `dR` and the forecast arrays."""

    def __init__(self, s, tag, st, caliper=0.3):
        self.ok = False
        if not (have(s, tag, st) and os.path.exists(pt(s, tag, st))):
            return
        self.T = Twins(bp(s, tag, st), jload(bj(s, tag, st)), caliper=caliper)
        if not self.T.ok:
            return
        self.X = np.load(xp(s, tag, st))
        self.Pz = np.load(pt(s, tag, st))
        w = np.asarray(self.T.Z["tw__w"])
        assert np.array_equal(np.asarray(self.X["tw_w"]), w), "express pair order"
        assert np.array_equal(np.asarray(self.Pz["w"]), w), "public pair order"
        n = len(w)
        k35 = np.asarray(self.Pz["keep35"]).astype(np.int64)
        self.has35 = np.zeros(n, bool)
        self.has35[k35] = True
        assert self.has35[self.T.keep].all(), "a 0.30 pair outside the saved 0.35 subset"
        pos = -np.ones(n, np.int64)
        pos[k35] = np.arange(len(k35))
        self.pos = pos
        self.Fv = np.asarray(self.Pz["F_v"], np.float64)
        self.Ft = np.asarray(self.Pz["F_t"], np.float64)
        self.dXs = np.asarray(self.Pz["dX"], np.float64)
        self.tb, self.resid = self.T.token_balanced()
        self.ok = True

    def dR(self, crit, l):
        if crit in ("full", "fullQ", "fullP", "fullM"):
            Z = self.X
        else:
            Z = self.Pz
        kv, kt = f"tw_ev_{crit}_l{l}_v", f"tw_ev_{crit}_l{l}_t"
        if kv not in Z.files:
            return None
        return np.asarray(Z[kv], np.float64) - np.asarray(Z[kt], np.float64)

    def at(self, arr, m):
        """Rows of a saved-subset array for the pair mask `m` (all inside the subset)."""
        return arr[self.pos[m]]


_TW = {}


def tw(s, tag, st):
    k = (s, tag, st)
    if k not in _TW:
        _TW[k] = TW(s, tag, st)
    return _TW[k]


def sec_twin_head(tag):
    rows = []
    for st in STEPS:
        def fn(s):
            t = tw(s, tag, st)
            if not t.ok:
                return "--"
            tr = t.T.keep & (t.T.split != 2)
            te = t.T.keep & (t.T.split == 2)
            return (f"{len(t.T.keep)} / {int(t.has35.sum())} / {int(t.T.keep.sum())} / "
                    f"{int(tr.sum())} / {int(te.sum())} / {int((t.tb & (t.T.split == 2)).sum())}"
                    f" ({t.resid})")
        rows.append([st, per_seed(fn)])
    return "\n".join([
        f"**The banked pair population ({tag}); {' / '.join(NAMES)} side by side.**  "
        f"`all pairs / saved at |ds| <= 0.35 / at the parent's caliper 0.30 / training "
        f"(split != 2) / held-out / held-out token-balanced (signed token histogram residual, 0 "
        f"by construction)`.", "", tbl(rows, ["step", "pairs"]), ""])


def sec_twin_opt(tag):
    crits = ["full", "lq_full", "mq_full", "fullM", "fullQM", "fullQ", "fullQst", "fullQbl"]
    rows = []
    for l in LS:
        for st in STEPS:
            cells = [l, st]
            for c in crits:
                def fn(s, c=c):
                    t = tw(s, tag, st)
                    if not t.ok:
                        return "--"
                    d = t.dR(c, l)
                    if d is None:
                        return "--"
                    x = d[t.tb]
                    return f"{x.mean():+.4f} ({x.mean() / max(_sem(x), 1e-12):.1f})"
                cells.append(per_seed(fn))
            cells.append(per_seed(lambda s: str(int(tw(s, tag, st).tb.sum()))
                                  if tw(s, tag, st).ok else "--"))
            rows.append(cells)
    return "\n".join([
        f"**The token-balanced twin `dR` per reader — {tag}, every pair at caliper 0.30 that "
        f"the token balancing keeps; {' / '.join(NAMES)} side by side.**  `dR = V(violator) - "
        f"V(legal twin)` at the event (the prefix is identical, so `V_pre` cancels for every "
        f"reader), `mean (mean/sem)`.", "",
        tbl(rows, ["l", "step"] + [LABEL[c] for c in crits] + ["n"]), ""])


def _dtok(t):
    xv = np.asarray(t.T.Z["tw__x_v"]).astype(np.int64)
    xc = np.asarray(t.T.Z["tw__x_c"]).astype(np.int64)
    E = np.eye(16)
    return E[xv] - E[xc]


PRICED_CRITS = ["full", "fullQ", "fullQst", "fullQbl", "fullM", "fullQM", "lq_full",
                "mq_full"]


def ridge_cv_multi(X, Y, folds=5, lams=None, seed=0):
    """`projection/analyze.py::ridge_cv`, column by column, with the fold Grams computed once:
    the same fold assignment (`default_rng(seed).permutation(n) % folds`), the same per-fold
    `_r2`, the same summed score, the first maximum picked per column, and the final fit on
    every row at that lambda."""
    from rhm.logit_reading.striatum.task import ridge_solve
    from rhm.logit_reading.orbitofrontal.projection.analyze import LAMS as PL
    lams = PL if lams is None else lams
    X = np.asarray(X, np.float64)
    Y = np.asarray(Y, np.float64)
    Xb = np.concatenate([X, np.ones((len(X), 1))], 1)
    fold = np.random.default_rng(seed).permutation(len(X)) % folds
    score = np.zeros((len(lams), Y.shape[1]))
    for f in range(folds):
        tr, va = fold != f, fold == f
        A = Xb[tr].T @ Xb[tr]
        c = Xb[tr].T @ Y[tr]
        yv = Y[va]
        den = np.maximum(yv.var(0), 1e-18)
        for li, lm in enumerate(lams):
            pred = Xb[va] @ ridge_solve(A, c, lm)
            score[li] += 1.0 - ((yv - pred) ** 2).mean(0) / den
    pick = np.argmax(score, 0)
    A = Xb.T @ Xb
    c = Xb.T @ Y
    B = np.zeros((Xb.shape[1], Y.shape[1]))
    for li, lm in enumerate(lams):
        cols = np.where(pick == li)[0]
        if len(cols):
            B[:, cols] = ridge_solve(A, c[:, cols], lm)
    return B, pick


_PR = {}


def _feat(t, f, n, k):
    if f == "dlogq@0":
        A = np.full((n, 16), np.nan)
        A[k] = t.at(t.Fv[:, 0] - t.Ft[:, 0], k)
    elif f == "dlogq@0-4":
        A = np.full((n, 80), np.nan)
        A[k] = t.at((t.Fv - t.Ft).reshape(len(t.Fv), -1), k)
    elif f == "dX":
        A = np.full((n, t.dXs.shape[1]), np.nan)
        A[k] = t.at(t.dXs, k)
    elif f == "dtok":
        A = _dtok(t)
    else:
        raise ValueError(f)
    return A


def priced(t, crit, l, feats, seed=0):
    """`projection/analyze.py`'s `sec_priced` fit on one reader's `dR`: ridge with 5-fold CV
    on the training pairs at caliper 0.30, read on the held-out ones.  Every reader and level
    of one (cell, feature set) is fitted in one multi-target pass and cached."""
    key = (id(t), tuple(feats), seed)
    if key not in _PR:
        k = t.T.keep
        tr, te = k & (t.T.split != 2), k & (t.T.split == 2)
        n = len(k)
        X = np.concatenate([_feat(t, f, n, k) for f in feats], 1)
        cols, Ys = [], []
        for c in PRICED_CRITS:
            for ll in LS:
                y = t.dR(c, ll)
                if y is not None:
                    cols.append((c, ll))
                    Ys.append(y)
        res = {}
        if Ys and tr.sum() >= X.shape[1] + 20:
            Y = np.stack(Ys, 1)
            B, _ = ridge_cv_multi(X[tr], Y[tr], seed=seed)
            P = np.concatenate([X[te], np.ones((te.sum(), 1))], 1) @ B
            for j, cl in enumerate(cols):
                res[cl] = _r2(P[:, j], Y[te, j])
        _PR[key] = res
    r2 = _PR[key].get((crit, l))
    return None if r2 is None else (r2,)


def priced_single(t, crit, l, feats, seed=0):
    """projection/analyze.py's `sec_priced` fit on one reader's `dR`: ridge with 5-fold CV on
    the training pairs at caliper 0.30, read on the held-out ones."""
    y = t.dR(crit, l)
    if y is None:
        return None
    k = t.T.keep
    tr, te = k & (t.T.split != 2), k & (t.T.split == 2)
    n = len(y)
    parts = []
    for f in feats:
        if f == "dlogq@0":
            A = np.full((n, 16), np.nan)
            A[k] = t.at(t.Fv[:, 0] - t.Ft[:, 0], k)
        elif f == "dlogq@0-4":
            A = np.full((n, 80), np.nan)
            A[k] = t.at((t.Fv - t.Ft).reshape(len(t.Fv), -1), k)
        elif f == "dX":
            A = np.full((n, t.dXs.shape[1]), np.nan)
            A[k] = t.at(t.dXs, k)
        elif f == "dtok":
            A = _dtok(t)
        else:
            raise ValueError(f)
        parts.append(A)
    X = np.concatenate(parts, 1)
    if tr.sum() < X.shape[1] + 20:
        return None
    beta, lam, cv = ridge_cv(X[tr], y[tr], seed=seed)
    pred = np.concatenate([X[te], np.ones((te.sum(), 1))], 1) @ beta
    return _r2(pred, y[te]), lam, cv


FEATS = [("dlogq@0", ["dlogq@0"]), ("dlogq@0-4", ["dlogq@0-4"]), ("dtok", ["dtok"]),
         ("dX (ceiling)", ["dX"]), ("dX + dlogq@0", ["dX", "dlogq@0"])]


def sec_priced(tag, crits, steps=STEPS):
    out = []
    for c in crits:
        rows = []
        for st in steps:
            for l in LS:
                cells = [st, l]
                for fnm, fs in FEATS:
                    def fn(s, fs=fs):
                        t = tw(s, tag, st)
                        if not t.ok:
                            return "--"
                        got = priced(t, c, l, fs)
                        return fmt(got[0], 4) if got is not None else "--"
                    cells.append(per_seed(fn))
                rows.append(cells)
        out += [f"**The priced twin on `{LABEL.get(c, c)}`'s `dR` — {tag}, held-out `R2` of one "
                f"linear map (ridge, 5-fold CV on the training pairs) from each feature set to "
                f"`dR`; {' / '.join(NAMES)} side by side.**  `dlogq@0`: the violator's "
                f"post-token forecast minus its twin's (16 dims, `projection/`'s headline); "
                f"`dlogq@0-4`: the same at `t_v + 0..4` (80); `dtok`: which token arrived and "
                f"which it replaced (16); `dX`: the state difference at `t_v` (256), the "
                f"ceiling; `dX + dlogq@0` (272).", "",
                tbl(rows, ["step", "l"] + [f for f, _ in FEATS]), ""]
    return "\n".join(out)


def sec_twin_split(tag, arm="fullQ"):
    """`dR(arm) = b_s . dX + b_g . dg` exactly on the pairs; the state block's term split
    further into what the output layer can express (`P_M b_s`) and its complement, beside
    projection's two-way split of the ridge (`sec_decomp`)."""
    rows = []
    for l in LS:
        for st in STEPS:
            for crit in ("full", arm):
                def fn(s, crit=crit):
                    t = tw(s, tag, st)
                    r = jload(pj(s, tag, st))
                    if not t.ok or r is None:
                        return "--"
                    Z = np.load(pp(s, tag, st))
                    if not os.path.exists(bas(s, st)):
                        return "--"
                    Vb = np.asarray(np.load(bas(s, st))["Vb"], np.float64)
                    Pm = Vb.T @ Vb
                    j = r["names"].index(f"l{l}_d0")
                    b = np.asarray(Z[f"beta__{crit}"], np.float64)[:, j]
                    bs = b[:Vb.shape[1]]
                    m = t.T.keep
                    dX = t.at(t.dXs, m)
                    y = t.dR(crit, l)[m]
                    yr, yu = dX @ (Pm @ bs), dX @ (bs - Pm @ bs)
                    yb = y - yr - yu                          # the appended block's term
                    vy = max(y.var(), 1e-18)
                    tb = t.tb[m]
                    shares = f"{yr.var() / vy:.2f}:{yu.var() / vy:.2f}:{yb.var() / vy:.2f}"
                    means = (f"{y[tb].mean():+.4f} = {yr[tb].mean():+.4f} {yu[tb].mean():+.4f} "
                             f"{yb[tb].mean():+.4f}")
                    return f"{shares} | {means}"
                rows.append([l, st, LABEL[crit], per_seed(fn)])
    return "\n".join([
        f"**The twin `dR` split three ways — {tag}, pairs at caliper 0.30; "
        f"{' / '.join(NAMES)} side by side.**  For the ridge `dR = b . dX` and for `{arm}` "
        f"`dR = b_s . dX + b_g . d(log q)` exactly (`V_pre` cancels); the state term is split "
        f"into `(P_M b_s) . dX` (readable: inside the output layer's rank-15 row space) and "
        f"`((I - P_M) b_s) . dX` (unreadable).  Each cell: `var share readable : unreadable : "
        f"appended` over the pairs | `mean dR = readable + unreadable + appended` on the "
        f"token-balanced pairs.  For the ridge the appended share is 0 by construction.", "",
        tbl(rows, ["l", "step", "reader", "shares | means"]), ""])


def sec_priced_gate():
    rows = []
    s = "42"
    for tag in TAGS:
        for st in STEPS:
            t = tw(s, tag, st)
            pr = jload(prj(s, tag, st, ".json"))
            ptw = prj(s, tag, st, "_twins.npz")
            if not t.ok or pr is None or not os.path.exists(ptw):
                continue
            P = Pairs(ptw, pr, caliper=0.3)
            rec = {l: r2 for l, r2, _, _, _ in priced_summary(P)}
            # the pair sets at 0.30
            a = set(zip(np.asarray(P.Z["tw__w"])[P.keep].tolist(),
                        np.asarray(P.Z["tw__t_v"])[P.keep].tolist()))
            b = set(zip(np.asarray(t.T.Z["tw__w"])[t.T.keep].tolist(),
                        np.asarray(t.T.Z["tw__t_v"])[t.T.keep].tolist()))
            for l in LS:
                got = priced(t, "full", l, ["dlogq@0"])
                if l == 1:                     # the single-target check, level 1 only
                    one = priced_single(t, "full", l, ["dlogq@0"])
                    oq = priced(t, "fullQ", l, ["dX"])
                    oq1 = priced_single(t, "fullQ", l, ["dX"])
                    chk = f"{abs(got[0] - one[0]):.0e} / {abs(oq[0] - oq1[0]):.0e}"
                else:
                    chk = "--"
                rows.append([tag, st, l, fmt(PROJ_PRICED[(tag, st)][l - 1], 4),
                             fmt(rec.get(l), 4), fmt(got[0] if got else None, 4), chk,
                             f"{len(a)} / {len(b)} / {len(a & b)}"])
    return "\n".join([
        "**The priced map on the ridge's `dR` against projection's banked value (s42).**  "
        "`banked`: `projection/results/tables.md`'s printed `test R2` (`dlogq@0`); "
        "`recomputed`: `projection/analyze.py`'s own `priced_summary` on the banked "
        "`_proj_<tag>_twins.npz`; `here`: this reduction's fit on `norm/`'s banked pairs with "
        "this node's recomputed forecasts.  `pairs`: projection's pairs at 0.30 / this node's / "
        "shared.  At step 0 projection capped the eligible violations at 8,000 and `norm/` at "
        "45,000, so the two pair sets differ there by construction.  `multi vs single`: "
        "|this reduction's multi-target fit - `projection/analyze.py::ridge_cv` run on the one "
        "column| for the ridge from `dlogq@0` and for `fullQ` from `dX` (level 1).", "",
        tbl(rows, ["venue", "step", "l", "banked", "recomputed", "here", "multi vs single",
                   "pairs proj / here / shared"]), ""])


# ---------------------------------------------------------------------------
# 3. the readout span
# ---------------------------------------------------------------------------

def sec_span():
    out = []
    crits = [("full", "ridge"), ("fullQ", "fullQ state block"), ("fullP", "fullP state block"),
             ("clean", "clean ridge"), ("cleanQ", "cleanQ state block")]
    rows = []
    for tag in TAGS:
        for st in STEPS:
            cells = [tag, st]
            for c, _ in crits:
                def fn(s, c=c):
                    r = jload(pj(s, tag, st))
                    if r is None:
                        return "--"
                    sp = r["span"]
                    return fmt(float(np.mean([sp[f"{c}/l{l}_d{dd}"] for l in LS
                                              for dd in (0, 1)])), 4)
                cells.append(per_seed(fn))

            def allc(s):
                r = jload(pj(s, tag, st))
                if r is None:
                    return "--"
                sp = r["span"]
                return fmt(float(np.mean([v for k, v in sp.items()
                                          if k.startswith("fullQ/") and k.count("/") == 1])), 4)
            cells.append(per_seed(allc))
            cells.append(fmt(jload(pj(SEEDS[0], tag, st))["span"]["_random_direction"], 4)
                         if jload(pj(SEEDS[0], tag, st)) else "--")
            rows.append(cells)
    out += [f"**The readout span `||P_M beta|| / ||beta||`, mean over the 8 columns l = 1-4 x "
            f"d = 0, 1 per critic; {' / '.join(NAMES)} side by side.**  `M` is `projection/`'s "
            f"rank-15 map from the state to the centred logits (`W_U diag(gamma)`, the softmax's "
            f"constant and layer norm's mean removed); for `fullQ` / `fullP` / `cleanQ` it is "
            f"the STATE block's coefficients (the appended block is not a state direction).  "
            f"`fullQ, 48 cols`: the mean over every real target column (l = 1-6 x 8 offsets).",
            "", tbl(rows, ["venue", "step"] + [lab for _, lab in crits]
                    + ["fullQ, 48 cols", "chance"]), ""]
    rows = []
    for tag in TAGS:
        for st in STEPS:
            for l in LS:
                for dd in (0, 1):
                    cells = [tag, st, f"l{l}_d{dd}"]
                    for c in ("full", "fullQ"):
                        cells.append(per_seed(lambda s, c=c: fmt(
                            jload(pj(s, tag, st))["span"][f"{c}/l{l}_d{dd}"], 4)
                            if jload(pj(s, tag, st)) else "--"))
                    for k in ("cos_ref", "norm_ratio"):
                        cells.append(per_seed(lambda s, k=k: fmt(
                            jload(pj(s, tag, st))["span"][f"fullQ/l{l}_d{dd}/{k}"], 3)
                            if jload(pj(s, tag, st)) else "--"))
                    rows.append(cells)
    out += [f"**Per column, the ridge and `fullQ`'s state block, with the cosine between the two "
            f"state directions and the norm ratio `||b_s(fullQ)|| / ||b(ridge)||`; "
            f"{' / '.join(NAMES)} side by side.**", "",
            tbl(rows, ["venue", "step", "column", "ridge", "fullQ state", "cos", "norm ratio"]),
            ""]
    return "\n".join(out)


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    return plt


MK = {"42": "o", "43": "s", "44": "^"}
# the reference categorical palette in fixed slot order (validated: dataviz validate_palette.js,
# light surface, all checks pass; the contrast WARN is relieved by legends and the tables)
COL = {"full": "#2a78d6", "fullQ": "#eb6834", "fullQst": "#1baf7a", "fullQbl": "#eda100",
       "fullM": "#e87ba4", "fullQM": "#008300", "lq_full": "#4a3aa7", "mq_full": "#e34948",
       "fullP": "#eb6834", "fullPst": "#1baf7a", "fullPbl": "#eda100"}


def fig_decomp(tag, an, path, arm="fullQ"):
    plt = _mpl()
    crits = ["full", arm, f"{arm}st", f"{arm}bl"]
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.4), sharey=False)
    for ax, (what, withV) in zip(axs, [("slope, no V_pre", False), ("slope | V_pre", True),
                                       ("AUC(-R, flip)", None)]):
        for ci, c in enumerate(crits):
            for s in SEEDS:
                xs, ys = [], []
                for l in LS:
                    R = rd(s, tag, 64000, an)
                    i, y = _mm(R, l)
                    if i is None or not R.has(c, l):
                        continue
                    if withV is None:
                        v = _auc(-R.R(c, l)[i], y)
                    else:
                        vref = c if c == "full" else arm
                        ex = [R.Vpre(vref, l)[i]] if withV else None
                        b, _, _ = _partial(R.R(c, l)[i], R.C.H[i], R.C.ctrl(extra=ex, idx=i))
                        v = b * float(R.C.H[i].std())
                    xs.append(l + (ci - 1.5) * 0.12)
                    ys.append(v)
                ax.plot(xs, ys, MK[s], color=COL.get(c, "k"), ms=4,
                        label=LABEL[c] if s == SEEDS[0] else None)
        ax.axhline(0.5 if withV is None else 0.0, color="#999", lw=0.6)
        ax.set_xticks(LS)
        ax.set_xlabel("level")
        ax.set_title(what)
    axs[0].legend(frameon=False, fontsize=7)
    fig.suptitle(f"{arm} split into state and appended parts - {tag}, {an}, 64k "
                 f"(markers: seeds {', '.join(SEEDS)})", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_span(path):
    plt = _mpl()
    fig, axs = plt.subplots(1, 2, figsize=(8, 3.2), sharey=True)
    for ax, tag in zip(axs, TAGS):
        for ci, (c, lab) in enumerate([("full", "ridge"), ("fullQ", "fullQ state block"),
                                       ("fullP", "fullP state block")]):
            for s in SEEDS:
                xs, ys = [], []
                for si, st in enumerate(STEPS):
                    r = jload(pj(s, tag, st))
                    if r is None:
                        continue
                    xs.append(si + (ci - 1) * 0.1)
                    ys.append(np.mean([r["span"][f"{c}/l{l}_d{dd}"] for l in LS
                                       for dd in (0, 1)]))
                ax.plot(xs, ys, MK[s], color=["#2a78d6", "#eb6834", "#1baf7a"][ci], ms=5,
                        label=lab if s == SEEDS[0] else None)
        ax.axhline(0.242, color="#999", lw=0.8, ls="--")
        ax.text(2.2, 0.245, "chance", fontsize=7, color="#777")
        ax.set_xticks(range(len(STEPS)))
        ax.set_xticklabels([str(x) for x in STEPS])
        ax.set_xlabel("step")
        ax.set_title(tag)
    axs[0].set_ylabel("readout span (mean of 8 columns)")
    axs[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_priced(tag, path, crits=("full", "fullQ", "fullM", "fullQM", "lq_full", "mq_full")):
    plt = _mpl()
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.3), sharey=True)
    for ax, st in zip(axs, (0, 64000)):
        for ci, c in enumerate(crits):
            for s in SEEDS:
                t = tw(s, tag, st)
                if not t.ok:
                    continue
                xs, ys = [], []
                for l in LS:
                    got = priced(t, c, l, ["dlogq@0"])
                    if got is None:
                        continue
                    xs.append(l + (ci - 2.5) * 0.1)
                    ys.append(got[0])
                ax.plot(xs, ys, MK[s], color=COL[c], ms=4,
                        label=LABEL[c] if s == SEEDS[0] else None)
        ax.set_xticks(LS)
        ax.set_xlabel("level")
        ax.set_title(f"{tag}, step {st}")
        ax.axhline(0, color="#999", lw=0.6)
    axs[0].set_ylabel("held-out R2, dlogq@0 -> dR")
    axs[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_calib(path, tag="a1", an="tv", st=64000):
    plt = _mpl()
    readers = [("state", "ridge (state)"), ("lq", "ridge (log q)"), ("mq", "MLP (log q)")]
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.3), sharey=True)
    cols = {1: "#2f6db3", 2: "#2a9d8f", 3: "#e0a526", 4: "#c8553d"}
    for ax, (reader, lab) in zip(axs, readers):
        for s in SEEDS:
            R = rd(s, tag, st, an)
            r = jload(bj(s, tag, st))
            if not R.ok or r is None:
                continue
            for l in LS:
                xs, ys = [], []
                for nm in DIET_ORDER:
                    v = _vpre_reader(R, reader, nm, l)
                    if v is None:
                        continue
                    xs.append(r["diet_stats"][nm]["mean_outcome"])
                    ys.append(float(v[R.viol].mean()))
                ax.plot(xs, ys, MK[s], color=cols[l], ms=4, alpha=0.8,
                        label=f"l={l}" if s == SEEDS[0] else None)
        ax.set_title(lab)
        ax.set_xlabel("diet E[outcome]")
    axs[0].set_ylabel("mean V_pre, violation rows")
    axs[0].legend(frameon=False, fontsize=7)
    fig.suptitle(f"the norm per diet - {tag}, {an}, step {st} (markers: seeds)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------

def main():
    global BANK, PUB, SEEDS, NAMES
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--pub", required=True)
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--out", default=None)
    ap.add_argument("--figs", default=None)
    ap.add_argument("--date", default="20260922")
    ap.add_argument("--no-integrity", action="store_true")
    a = ap.parse_args()
    BANK, PUB = a.bank, a.pub
    SEEDS = a.seeds.split(",")
    NAMES = [f"s{s}" for s in SEEDS]

    files = []
    for s in SEEDS:
        for tag in TAGS:
            for st in STEPS:
                files += [pp(s, tag, st), pt(s, tag, st), xp(s, tag, st), bp(s, tag, st)]
                if s == "42":
                    files += [prj(s, tag, st, f"_rows_{an}.npz") for an in ANCS]
                    files += [prj(s, tag, st, "_twins.npz")]
    doc = ["# precision / rereads / public — the augmented reader's response, split; a "
           "nonlinear public reader", "",
           "Facts only; interpretation is discussed with Jasper and lives nowhere in this "
           "folder.  Generated by [`reduce_public.py`](../reduce_public.py) from "
           "[`public.py`](../public.py)'s cells beside the banked `norm/`, `precision/express` "
           "and (s42) `orbitofrontal/projection/` cells.  Three trajectory seeds side by side, "
           "**never averaged**; step 0 is the random-init floor.", "",
           "**Readers**, all on `norm/`'s trunk (`post_block7`), `full` diet unless named, the "
           "same split, lambda ladder and held-out selection, read on the identical test rows "
           "and the identical banked twin pairs:", "",
           tbl([["ridge (state)", "the banked critic (`norm/task.py`), refitted and gated"],
                ["+log q (fullQ)", "ridge on `[state, log q_t]` (`express.py` arm Q, appended "
                 "block standardised to the state's per-dimension RMS), refitted and gated"],
                ["fullQ: state part / belief part", "`fullQ`'s value split exactly per row: "
                 "`x . b_s + b + mu . b_q` and `(log q - mu) . b_q`, with `mu` the log q "
                 "block's mean on the critic's training rows"],
                ["+precision (fullP) and its parts", "express arm P (`H(q_t)`, `H(q_(t-1))`, "
                 "`s_t`, `H(q_(t-1)) s_t` appended), split the same way"],
                ["ridge (log q)", "the ridge on log q alone (16 dims): `projection/`'s linear "
                 "public critic, refitted on three seeds and gated on s42"],
                ["MLP (log q)", "a 128-unit GELU MLP on log q alone, `train_head` with `fullM`'s "
                 "recipe (2000 steps, seed `17 l + d`), one head per (diet, level, offset) — "
                 "the nonlinear public reader"],
                ["MLP (state)", "`fullM`, the banked `_express` columns"],
                ["MLP (state, log q)", "the same MLP on `[state, log q]` (272 dims), `full` "
                 "only"]], ["reader", "what"]), "",
           "`R = V[l, 0](s_t) - V[l, 1](s_(t-1))` at the anchor (`tv` = the violating token, "
           "`fd` = the first edited token); `R_st + R_bl = R` for `fullQ` / `fullP` exactly.  "
           "`dR = R(violator) - R(twin)` on `norm/`'s banked same-prefix pairs at caliper 0.30.",
           "", "---", "", "## G. Gates", ""]
    if not a.no_integrity:
        doc += [sec_integrity(files)]
    doc += [sec_cell_gates(), sec_red_gates(), sec_priced_gate(), "---", ""]
    for tag in TAGS:
        doc += [f"# {tag}", ""]
        doc += [f"## 1. The augmented critic's response, split exactly ({tag})", ""]
        for an in ANCS:
            doc += [f"### {tag}, anchor {an}", ""]
            for arm in ("fullQ", "fullP"):
                doc += [sec_amp(tag, an, arm), sec_part_auc(tag, an, arm, "R"),
                        sec_part_slope(tag, an, arm, False), sec_part_slope(tag, an, arm, True),
                        sec_part_inter(tag, an, arm)]
                if arm == "fullQ":
                    doc += [sec_part_contain(tag, an, arm)]
        doc += [f"## 2. The public readers ({tag})", "", sec_fit(tag)]
        if tag == "a1":
            doc += [sec_calib(tag, "tv", 64000), sec_calib(tag, "tv", 8000),
                    sec_calib_rank(tag, "tv"), sec_shape(tag, "tv", 1)]
        for an in ANCS:
            doc += [f"### the response per reader, {tag}, anchor {an}", "",
                    sec_response(tag, an)]
        doc += [sec_rand16(tag, "tv")]
        doc += [f"## 4. The twins and the priced map ({tag})", "", sec_twin_head(tag),
                sec_twin_opt(tag), sec_twin_split(tag, "fullQ"),
                sec_priced(tag, ["full", "fullQ", "fullQst", "fullQbl", "fullM", "fullQM",
                                 "lq_full", "mq_full"])]
    doc += ["# 3. The readout span of the augmented critic's state block", "", sec_span()]
    txt = "\n".join(doc) + "\n"
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        p = os.path.join(a.out, f"tables_{a.date}.md")
        with open(p, "w") as f:
            f.write(txt)
        print(f"wrote {p} ({len(txt)} chars)")
    else:
        print(txt)
    if a.figs:
        os.makedirs(a.figs, exist_ok=True)
        for tag in TAGS:
            for an in ANCS:
                fig_decomp(tag, an, os.path.join(a.figs, f"pub_decomp_{tag}_{an}.png"))
            fig_priced(tag, os.path.join(a.figs, f"pub_priced_{tag}.png"))
        fig_span(os.path.join(a.figs, "pub_span.png"))
        fig_calib(os.path.join(a.figs, "pub_calib_a1.png"))
        print(f"figures -> {a.figs}")


if __name__ == "__main__":
    main()
