"""Part A: what carries the value response's dependence on `H_pre`.  Part B: the same axis
in the glitch world beside the edit world.  Facts only; no interpretation lives here.

`reduce.py` / `results/tables_20260922.md` established that at matched surprisal the value
response `R` depends positively on the trunk's pre-event entropy at levels 1-3, and that
`H_pre` itself ranks realised damage at ell = 1 on rows where the surprisal is pinned at
0.500.  This file asks what that dependence is carried by, with `mediate.py`'s columns:

  `kl_update`    the model's FORECAST movement at the event, `KL(q_t || q_(t-1))`;
  `H_next`       the entropy after the event;
  `dstate_<b>`   the REPRESENTATION's movement, `||s_t - s_(t-1)||_2`, at the critic's own
                 block (`post_block7`) and two others;
  `dstate_pair_<b>` (pairs) `||s_t^flagged - s_t^twin||_2`, the state difference `dR` reads.

Every mediator is a column on the SAME banked rows and the SAME pairs, gated against the
banked `H_pre`, `nll_e`, `s_v` and `s_c` inside `mediate.py`; the gate values are reprinted
here.  Nothing is refit, `tables_20260922.md` is not rewritten, and no banked file is
touched.

  python -m rhm.logit_reading.striatum.norm.precision.mediation \
      <dir>/traj_a1_s42 <dir>/traj_a1_s43 <dir>/traj_a1_s44 \
      --out rhm/logit_reading/striatum/norm/precision/results \
      --figs rhm/logit_reading/striatum/norm/precision/figs --date 20260922
"""

import argparse
import json
import os

import numpy as np

from rhm.logit_reading.striatum.analyze import _auc, _cond_auc, fmt, tbl
from rhm.logit_reading.striatum.norm.analyze import (Twins, common_windows, _ols, _sem,
                                                     _spear)
from rhm.logit_reading.striatum.norm.precision.reduce import (
    STEPS, LS, Cell, cell_of, json_of, matched_sets, npz_of, _grid, _ols_multi, _partial,
    _pearson, _qbin, _resid, _z)

MED = ["kl_update", "H_next", "dstate_post_block7", "dstate_post_block3",
       "dstate_post_embed"]
MED_SHORT = {"kl_update": "kl", "H_next": "Hnext", "dstate_post_block7": "dstate7",
             "dstate_post_block3": "dstate3", "dstate_post_embed": "dstate_emb"}
CORE = ["kl_update", "dstate_post_block7"]
WORLD = {"tw__": "edit (swap)", "gl__": "glitch"}


# ---------------------------------------------------------------------------
# the mediator columns
# ---------------------------------------------------------------------------

def med_path(d, tag, st, tag2=""):
    sfx = f"_norm_{tag}" + (f"_{tag2}" if tag2 else "") + "_med.npz"
    return os.path.join(d, f"step{st:06d}{sfx}")


def banked_path(d, tag, st, tag2=""):
    sfx = f"_norm_{tag}" + (f"_{tag2}" if tag2 else "") + ".npz"
    return os.path.join(d, f"step{st:06d}{sfx}")


class Med:
    """`mediate.py`'s columns for one cell, aligned to the banked rows by construction
    (same `w`, same `t0` / `t_v`, asserted here again)."""

    def __init__(self, d, tag, st, tag2=""):
        self.p = med_path(d, tag, st, tag2)
        self.ok = os.path.exists(self.p)
        if not self.ok:
            return
        self.Q = np.load(self.p)
        self.meta = json.loads(str(self.Q["meta"])) if "meta" in self.Q.files else {}

    def rows(self, an, Z):
        """The fixed-row mediators at anchor `an`, with the row-identity assertion."""
        if not self.ok or f"{an}_w" not in self.Q.files:
            return None
        assert np.array_equal(np.asarray(self.Q[f"{an}_w"]), np.asarray(Z[f"{an}__w"])), \
            f"{self.p}: {an} window order differs from the banked npz"
        assert np.array_equal(np.asarray(self.Q[f"{an}_t0"]), np.asarray(Z[f"{an}__t0"])), \
            f"{self.p}: {an} anchor positions differ from the banked npz"
        return {m: np.asarray(self.Q[f"{an}_{m}"], np.float64) for m in MED
                if f"{an}_{m}" in self.Q.files}

    def pairs(self, pre, Z):
        k = {"tw__": "tw", "gl__": "gl"}[pre]
        if not self.ok or f"{k}_w" not in self.Q.files:
            return None
        assert np.array_equal(np.asarray(self.Q[f"{k}_w"]), np.asarray(Z[pre + "w"])), \
            f"{self.p}: {pre} pair order differs from the banked npz"
        o = {"H_pre": np.asarray(self.Q[f"{k}_H_pre"], np.float64)}
        for nm in ("H_next", "kl_update"):
            o[f"d_{nm}"] = (np.asarray(self.Q[f"{k}_{nm}_v"], np.float64)
                            - np.asarray(self.Q[f"{k}_{nm}_t"], np.float64))
            o[f"{nm}_v"] = np.asarray(self.Q[f"{k}_{nm}_v"], np.float64)
        for b in ("post_block7", "post_block3", "post_embed"):
            kk = f"{k}_dstate_pair_{b}"
            if kk in self.Q.files:
                o[f"dstate_pair_{b}"] = np.asarray(self.Q[kk], np.float64)
                o[f"dstate_{b}_v"] = np.asarray(self.Q[f"{k}_dstate_{b}_v"], np.float64)
        return o


# ---------------------------------------------------------------------------
# A0. gates
# ---------------------------------------------------------------------------

def sec_gates(dirs, names, tags, tag2=""):
    rows = []
    for d, nm in zip(dirs, names):
        for tag in tags:
            for st in STEPS:
                M = Med(d, tag, st, tag2)
                if not M.ok:
                    rows.append([nm, tag, st] + ["MISSING"] * 7)
                    continue
                g = M.meta
                rows.append([nm, tag, st,
                             f"{g.get('tv_gate_H_pre', float('nan')):.2e}",
                             f"{g.get('tv_gate_nll_e', float('nan')):.2e}",
                             f"{g.get('fd_gate_H_pre', float('nan')):.2e}",
                             f"{g.get('tw_gate_s_v', float('nan')):.2e}",
                             f"{g.get('tw_gate_s_c', float('nan')):.2e}",
                             f"{g.get('tw_gate_H_pre_within_pair', float('nan')):.2e}",
                             (f"{g.get('gl_gate_s_v'):.2e}"
                              if g.get("gl_gate_s_v") is not None else "--")])
    return "\n".join([
        "**Gates on the new pass.**  `mediate.py` recomputes the forecast and the state at "
        "the event for exactly the banked rows and pairs, and before writing anything it "
        "asserts that the recomputed `H_pre` and surprisals equal the banked `<an>__H_pre`, "
        "`<an>__nll_e`, `tw__s_v`, `tw__s_c` (and `gl__` where present) on EVERY row and "
        "EVERY pair.  The window order and anchor positions are re-asserted here at read "
        "time.  All values are float32 storage noise.", "",
        tbl(rows, ["seed", "venue", "step", "tv H_pre", "tv nll_e", "fd H_pre", "tw s_v",
                   "tw s_c", "tw H_pre in pair", "gl s_v"]), ""])


# ---------------------------------------------------------------------------
# A1. the premise -- do the mediators depend on H_pre, and do they read damage?
# ---------------------------------------------------------------------------

def sec_med_axis(dirs, names, tag, an, tag2=""):
    """Partial slope of each mediator on `H_pre`, per sd of `H_pre`, three seeds."""
    rows = []
    for m in MED:
        for st in STEPS:
            cells = [MED_SHORT[m], st]
            per_b, per_t, per_sd = [], [], []
            for d in dirs:
                p = banked_path(d, tag, st, tag2)
                M = Med(d, tag, st, tag2)
                if not (os.path.exists(p) and M.ok):
                    per_b.append("--")
                    per_t.append("--")
                    per_sd.append("--")
                    continue
                C = cell_of(p, an)
                R_ = M.rows(an, C.Z)
                if R_ is None or m not in R_:
                    per_b.append("--")
                    per_t.append("--")
                    per_sd.append("--")
                    continue
                y = R_[m]
                b, se, _ = _partial(y, C.H, C.ctrl())
                per_b.append(fmt(b * float(C.H.std()) / max(float(y.std()), 1e-12), 3))
                per_t.append(fmt(b / se, 1) if np.isfinite(se) and se > 0 else "--")
                per_sd.append(fmt(float(y.mean()), 3) + " / " + fmt(float(y.std()), 3))
            cells += [" / ".join(per_b), " / ".join(per_t), " / ".join(per_sd)]
            rows.append(cells)
    return "\n".join([
        f"**Do the mediators depend on the pre-event entropy at matched surprisal? — {tag}, "
        f"anchor {an}; {' / '.join(names)} side by side, never averaged.**  Partial slope of "
        f"each mediator on `H_pre` with both sides residualised on `(k*, j, t0, nll_e)`, "
        f"reported in **units of the mediator's own sd per sd of `H_pre`** so the five are "
        f"comparable; `t` is the slope over its standard error.  `mean / sd` is the "
        f"mediator's own distribution on the cell.", "",
        tbl(rows, ["mediator", "step", "b (sd per sd)", "t", "mean / sd"]), ""])


def sec_med_auc(dirs, names, tag, an, a=0, kind="flip", tag2=""):
    """What each mediator ranks on the matched rows, beside H_pre and the guard."""
    rows = []
    for l in LS:
        for st in STEPS:
            cells = [l, st]
            for what in ["H_pre"] + MED + ["nll_e"]:
                per = []
                for d in dirs:
                    p = banked_path(d, tag, st, tag2)
                    if not os.path.exists(p):
                        per.append("--")
                        continue
                    Rw, mm = matched_sets(p, an, kind, a)
                    if Rw is None or l not in mm:
                        per.append("--")
                        continue
                    m, lab = mm[l]
                    C = cell_of(p, an)
                    if what == "H_pre":
                        x = C.H
                    elif what == "nll_e":
                        x = C.s
                    else:
                        M = Med(d, tag, st, tag2)
                        R_ = M.rows(an, C.Z) if M.ok else None
                        if R_ is None or what not in R_:
                            per.append("--")
                            continue
                        x = R_[what]
                    per.append(fmt(_auc(x[m], lab[m])))
                cells.append(" / ".join(per))
            rows.append(cells)
    return "\n".join([
        f"**What each mediator ranks on the matched rows — {tag}, anchor {an}, a = {a}, "
        f"label `{kind}`; {' / '.join(names)} side by side.**  These are the rows "
        f"`junction/analyze.py`'s matcher selects, where the model's surprisal is pinned at "
        f"0.500 (last column, the guard).  Every score is ranked with `+` so an AUC above "
        f"0.500 means MORE of it goes with realised damage.", "",
        tbl(rows, ["l", "step", "AUC H_pre"] + [f"AUC {MED_SHORT[m]}" for m in MED]
            + ["AUC s [guard]"]), ""])


def sec_collinear(dirs, names, tag, an, tag2=""):
    """How collinear the axis, the guard and the mediators are with each other.  The
    mediation columns cannot be read without this: a control that is nearly `H_pre` itself
    removes the slope for an arithmetic reason, not a mechanistic one."""
    cols = ["H_pre", "nll_e"] + MED
    out = []
    for st in STEPS:
        rows = []
        for d, nm in zip(dirs, names):
            p = banked_path(d, tag, st, tag2)
            M = Med(d, tag, st, tag2)
            if not (os.path.exists(p) and M.ok):
                continue
            C = cell_of(p, an)
            R_ = M.rows(an, C.Z)
            if R_ is None:
                continue
            get = {"H_pre": C.H, "nll_e": C.s}
            get.update({m: R_[m] for m in MED if m in R_})
            for c in cols:
                if c not in get:
                    continue
                rows.append([nm, MED_SHORT.get(c, c)]
                            + [fmt(_pearson(get[c], get[c2]), 3) if c2 in get else "--"
                               for c2 in cols])
        if rows:
            out += [f"**step {st}**", "",
                    tbl(rows, ["seed", "column"]
                        + [MED_SHORT.get(c, c) for c in cols]), ""]
    return "\n".join([
        f"**Collinearity of the axis, the guard and the mediators — {tag}, anchor {an}, all "
        f"fixed rows.**  Pearson `r`.  **Read the mediation columns against this table.**  "
        f"`H_next` in particular is a LEVEL on the same scale as `H_pre`, not a movement, "
        f"so putting it in the controls is close to putting `H_pre` in them; the "
        f"`+ movement only` column of the next section is the control set that contains "
        f"movements alone.", ""] + out)


# ---------------------------------------------------------------------------
# A2. the mediation
# ---------------------------------------------------------------------------

MOVE = ["kl_update", "dstate_post_block7", "dstate_post_block3", "dstate_post_embed"]
CTRL_SETS = [("base", []), ("+ V_pre", ["V"]), ("+ kl", ["kl_update"]),
             ("+ dstate7", ["dstate_post_block7"]), ("+ H_next", ["H_next"]),
             ("+ movement only", MOVE), ("+ all med", MED),
             ("+ all med + V_pre", MED + ["V"])]


def _ctrl_extra(C, M_rows, diet, l, a, spec, idx):
    ex = []
    for s in spec:
        if s == "V":
            ex.append(C.Vpre(diet, l, a)[idx])
        else:
            ex.append(M_rows[s][idx])
    return ex or None


def sec_mediation(dirs, names, tag, an, diet="full", a=0, matched=True, kind="flip",
                  tag2=""):
    """`R ~ H_pre` with each mediator added to the controls, one at a time and together."""
    rows = []
    for l in LS:
        for st in STEPS:
            cells = [l, st]
            n_per = []
            for label, spec in CTRL_SETS:
                per = []
                for d in dirs:
                    p = banked_path(d, tag, st, tag2)
                    M = Med(d, tag, st, tag2)
                    if not (os.path.exists(p) and M.ok):
                        per.append("--")
                        continue
                    C = cell_of(p, an)
                    R_ = M.rows(an, C.Z)
                    if R_ is None or any(s != "V" and s not in R_ for s in spec):
                        per.append("--")
                        continue
                    if matched:
                        Rw, mm = matched_sets(p, an, kind, a)
                        if Rw is None or l not in mm:
                            per.append("--")
                            continue
                        idx = mm[l][0]
                    else:
                        idx = np.arange(C.n)
                    y = C.R(diet, l, a)[idx]
                    ctl = C.ctrl(extra=_ctrl_extra(C, R_, diet, l, a, spec, idx), idx=idx)
                    b, se, _ = _partial(y, C.H[idx], ctl)
                    per.append(fmt(b * float(C.H[idx].std()), 4)
                               + (f" ({b / se:.1f})" if np.isfinite(se) and se > 0
                                  else ""))
                cells.append(" / ".join(per))
            for d in dirs:
                p = banked_path(d, tag, st, tag2)
                if not os.path.exists(p):
                    n_per.append("--")
                elif matched:
                    Rw, mm = matched_sets(p, an, kind, a)
                    n_per.append(str(len(mm[l][0])) if (Rw is not None and l in mm)
                                 else "--")
                else:
                    n_per.append(str(cell_of(p, an).n))
            cells.append(" / ".join(n_per))
            rows.append(cells)
    where = "matched rows" if matched else "all fixed rows"
    return "\n".join([
        f"**`R ~ H_pre` with the mediators in the controls — {tag}, anchor {an}, critic "
        f"`{diet}`, a = {a}, {where}; {' / '.join(names)} side by side.**  Every column is "
        f"the partial slope **times the cell's own sd of `H_pre`** (the response's change "
        f"over one sd of the axis) with `(t)` beside it.  `base` is "
        f"`tables_20260922.md`'s own column, residualised on `(k*, j, t0, nll_e)`; each "
        f"further column adds that term to the SAME control set on the SAME rows.", "",
        tbl(rows, ["l", "step"] + [c[0] for c in CTRL_SETS] + ["n"]), ""])


def sec_both_ways(dirs, names, tag, an, diet="full", a=0, nbin=5, matched=True,
                  kind="flip", tag2=""):
    """The arc's other form: each object's entropy dependence inside quantile bins of the
    other, both directions, pooled over bins."""
    rows = []
    for m in CORE:
        for l in LS:
            for st in STEPS:
                cells = [MED_SHORT[m], l, st]
                for direction in ("R|med", "med|R"):
                    per = []
                    for d in dirs:
                        p = banked_path(d, tag, st, tag2)
                        M = Med(d, tag, st, tag2)
                        if not (os.path.exists(p) and M.ok):
                            per.append("--")
                            continue
                        C = cell_of(p, an)
                        R_ = M.rows(an, C.Z)
                        if R_ is None or m not in R_:
                            per.append("--")
                            continue
                        if matched:
                            Rw, mm = matched_sets(p, an, kind, a)
                            if Rw is None or l not in mm:
                                per.append("--")
                                continue
                            idx = mm[l][0]
                        else:
                            idx = np.arange(C.n)
                        Rv, mv = C.R(diet, l, a)[idx], R_[m][idx]
                        y, cond = ((Rv, mv) if direction == "R|med" else (mv, Rv))
                        ctl = C.ctrl(idx=idx)
                        ry, rh = _resid(y, ctl), _resid(C.H[idx], ctl)
                        sc = float(y.std()) if direction == "med|R" else 1.0
                        b = _qbin(cond, nbin)
                        num = den = 0.0
                        for bb in range(nbin):
                            s = b == bb
                            if s.sum() < 20 or rh[s].std() == 0:
                                continue
                            bb_ = _ols(rh[s], ry[s])[0]
                            num += s.sum() * bb_
                            den += s.sum()
                        v = (num / den) if den else float("nan")
                        per.append(fmt(v * float(C.H[idx].std()) / max(sc, 1e-12), 4))
                    cells.append(" / ".join(per))
                rows.append(cells)
    return "\n".join([
        f"**Both ways — {tag}, anchor {an}, critic `{diet}`, "
        f"{'matched rows' if matched else 'all fixed rows'}; "
        f"{' / '.join(names)} side by side.**  `R given med` is the entropy slope of `R` "
        f"computed INSIDE {nbin} quantile bins of the mediator and pooled by bin size "
        f"(units: `R` per sd of `H_pre`); `med given R` is the entropy slope of the "
        f"mediator inside {nbin} quantile bins of `R` (units: the mediator's own sd per sd "
        f"of `H_pre`).  Both sides are residualised on `(k*, j, t0, nll_e)` first, as "
        f"everywhere else.", "",
        tbl(rows, ["mediator", "l", "step", "R given med", "med given R"]), ""])


def sec_interaction_med(dirs, names, tag, an, diet="full", a=0, matched=True, kind="flip",
                        tag2=""):
    """The z(s) x z(H) interaction with the mediators as extra linear controls."""
    rows = []
    for l in LS:
        for st in STEPS:
            cells = [l, st]
            for label, spec in (("no mediators", []), ("+ kl", ["kl_update"]),
                                ("+ dstate7", ["dstate_post_block7"]),
                                ("+ all med", MED)):
                per = []
                for d in dirs:
                    p = banked_path(d, tag, st, tag2)
                    M = Med(d, tag, st, tag2)
                    if not (os.path.exists(p) and M.ok):
                        per.append("--")
                        continue
                    C = cell_of(p, an)
                    R_ = M.rows(an, C.Z)
                    if R_ is None:
                        per.append("--")
                        continue
                    if matched:
                        Rw, mm = matched_sets(p, an, kind, a)
                        if Rw is None or l not in mm:
                            per.append("--")
                            continue
                        idx = mm[l][0]
                    else:
                        idx = np.arange(C.n)
                    y = C.R(diet, l, a)[idx]
                    zs, zh = _z(C.s[idx]), _z(C.H[idx])
                    cols = [zs, zh, zs * zh, C.ks[idx].astype(float),
                            C.jj[idx].astype(float), C.t0[idx].astype(float)]
                    cols += [_z(R_[s][idx]) for s in spec]
                    cf, se, _ = _ols_multi(y, np.stack(cols, 1))
                    per.append(fmt(cf[2], 4)
                               + (f" ({cf[2] / se[2]:.1f})" if se[2] > 0 else ""))
                cells.append(" / ".join(per))
            rows.append(cells)
    return "\n".join([
        f"**The `z(s) x z(H)` interaction with the mediators controlled — {tag}, anchor "
        f"{an}, critic `{diet}`, {'matched rows' if matched else 'all fixed rows'}; "
        f"{' / '.join(names)} side by side.**  The same fit as "
        f"`tables_20260922.md` 3 (`R` on `z(s) + z(H) + z(s).z(H)` with `(k*, j, t0)` "
        f"linear), with each mediator added z-scored and linear.  `(t)` is the interaction "
        f"over its standard error.", "",
        tbl(rows, ["l", "step", "no mediators", "+ kl", "+ dstate7", "+ all med"]), ""])


def sec_surface_resid(d, tag, an, name, diet="full", a=0, k=5, tag2=""):
    """The 2-D surface of `R` residualised on the mediators, beside the raw one."""
    out = []
    for st in STEPS:
        p = banked_path(d, tag, st, tag2)
        M = Med(d, tag, st, tag2)
        if not (os.path.exists(p) and M.ok):
            continue
        C = cell_of(p, an)
        R_ = M.rows(an, C.Z)
        if R_ is None:
            continue
        G = np.stack([R_[m] for m in MED if m in R_], 1)
        for l in LS:
            idx = np.arange(C.n)
            z = _resid(C.R(diet, l, a), G)
            bx, by = _qbin(C.s, k), _qbin(C.H, k)
            body = _grid(z, bx, by, k, k)
            out += [f"**step {st}, l = {l}** — mean `R` residualised on the mediators (n) "
                    f"per cell:", "",
                    tbl(body, ["H \\ s"] + [f"s q{i + 1}" for i in range(k)]), ""]
    return "\n".join(out)


def sec_damage_cond(dirs, names, tag, an, a=0, kind="flip", nbin=5, tag2=""):
    """Does `H_pre`'s own ranking of damage survive conditioning on the mediators?"""
    rows = []
    for l in LS:
        for st in STEPS:
            cells = [l, st]
            for what in ["marginal"] + CORE + ["all"]:
                per = []
                for d in dirs:
                    p = banked_path(d, tag, st, tag2)
                    M = Med(d, tag, st, tag2)
                    if not (os.path.exists(p) and M.ok):
                        per.append("--")
                        continue
                    Rw, mm = matched_sets(p, an, kind, a)
                    if Rw is None or l not in mm:
                        per.append("--")
                        continue
                    m, lab = mm[l]
                    C = cell_of(p, an)
                    R_ = M.rows(an, C.Z)
                    if R_ is None:
                        per.append("--")
                        continue
                    if what == "marginal":
                        per.append(fmt(_auc(C.H[m], lab[m])))
                    elif what == "all":
                        G = np.stack([R_[mm_][m] for mm_ in MED if mm_ in R_], 1)
                        per.append(fmt(_auc(_resid(C.H[m], G), lab[m])))
                    else:
                        per.append(fmt(_cond_auc(C.H[m], R_[what][m], lab[m], nbin)))
                cells.append(" / ".join(per))
            # and the reverse: the mediator given H_pre
            for what in CORE:
                per = []
                for d in dirs:
                    p = banked_path(d, tag, st, tag2)
                    M = Med(d, tag, st, tag2)
                    if not (os.path.exists(p) and M.ok):
                        per.append("--")
                        continue
                    Rw, mm = matched_sets(p, an, kind, a)
                    if Rw is None or l not in mm:
                        per.append("--")
                        continue
                    m, lab = mm[l]
                    C = cell_of(p, an)
                    R_ = M.rows(an, C.Z)
                    per.append("--" if R_ is None or what not in R_
                               else fmt(_cond_auc(R_[what][m], C.H[m], lab[m], nbin)))
                cells.append(" / ".join(per))
            rows.append(cells)
    return "\n".join([
        f"**`AUC(H_pre, {kind})` conditioned on the mediators — {tag}, anchor {an}, "
        f"a = {a}, matched rows; {' / '.join(names)} side by side.**  `given X` is the "
        f"AUC taken inside {nbin} quantile bins of `X` and pooled with weight "
        f"`n_pos * n_neg` (`striatum/analyze.py`'s `_cond_auc`, the arc's own conditional); "
        f"`resid on all med` ranks `H_pre` after it has been linearly residualised on all "
        f"five mediators.  The last two columns are the reverse conditioning.", "",
        tbl(rows, ["l", "step", "H_pre marginal"]
            + [f"H_pre given {MED_SHORT[m]}" for m in CORE]
            + ["H_pre resid on all med"]
            + [f"{MED_SHORT[m]} given H_pre" for m in CORE]), ""])


# ---------------------------------------------------------------------------
# A5. the twins
# ---------------------------------------------------------------------------

_TWIN_CACHE = {}


def twins_of(p, pj, caliper, pre):
    """One world's `Twins` from a banked cell, memoised."""
    ck = (p, caliper, pre)
    if ck not in _TWIN_CACHE:
        if not (os.path.exists(p) and os.path.exists(pj)):
            _TWIN_CACHE[ck] = None
        else:
            _TWIN_CACHE[ck] = Twins(p, json.load(open(pj)), caliper=caliper, pre=pre,
                                    key="glitch" if pre == "gl__" else "twin")
    return _TWIN_CACHE[ck]


def twin_med_slope(dirs, names, tag, diet="full", balanced=True, caliper=0.3, pre="tw__",
                   tag2=""):
    """`dR ~ H_pre` residualised on the banked control set, and again with the pair
    mediators added."""
    rows = []
    for l in LS:
        for st in STEPS:
            cells = [l, st]
            for label, extra in (("base", []), ("+ dpair7", ["dstate_pair_post_block7"]),
                                 ("+ d kl", ["d_kl_update"]),
                                 ("+ both", ["dstate_pair_post_block7", "d_kl_update"])):
                per = []
                for d in dirs:
                    p, pj = banked_path(d, tag, st, tag2), json_of_t(d, tag, st, tag2)
                    M = Med(d, tag, st, tag2)
                    if not (os.path.exists(p) and os.path.exists(pj) and M.ok):
                        per.append("--")
                        continue
                    T = twins_of(p, pj, caliper, pre)
                    P = M.pairs(pre, T.Z) if (T is not None and T.ok) else None
                    if T is None or not T.ok or P is None or (diet, l, 0) not in T.cols:
                        per.append("--")
                        continue
                    tb, _ = tb_of(T, key="own")
                    m = tb if balanced else T.keep
                    if m.sum() < 50:
                        per.append("--")
                        continue
                    dR, H = T.dR(diet, l)[m], P["H_pre"][m]
                    G = [T.ks[m], T.jj[m], T.tv[m], T.s_v[m], np.abs(T.ds[m])]
                    G += [P[e][m] for e in extra]
                    b, se, _ = _partial(dR, H, np.stack(G, 1).astype(np.float64))
                    per.append(fmt(b * float(H.std()), 5)
                               + (f" ({b / se:.1f})" if np.isfinite(se) and se > 0
                                  else ""))
                cells.append(" / ".join(per))
            n_per = []
            for d in dirs:
                p, pj = banked_path(d, tag, st, tag2), json_of_t(d, tag, st, tag2)
                if not (os.path.exists(p) and os.path.exists(pj)):
                    n_per.append("--")
                    continue
                T = twins_of(p, pj, caliper, pre)
                if T is None or not T.ok:
                    n_per.append("--")
                    continue
                tb, _ = tb_of(T, key="own")
                n_per.append(str(int((tb if balanced else T.keep).sum())))
            cells.append(" / ".join(n_per))
            rows.append(cells)
    arm = "token-balanced" if balanced else "all pairs at caliper 0.30"
    return "\n".join([
        f"**`dR ~ H_pre` with the pair mediators in the controls — {tag}, world "
        f"`{WORLD[pre]}`, critic `{diet}`, {arm}; {' / '.join(names)} side by side.**  "
        f"`base` is `tables_20260922.md` 5's own column, residualised on `(k*, j, position, "
        f"violator surprisal, |ds|)`; the further columns add the pair's state distance at "
        f"the critic's block and the pair's forecast-movement difference "
        f"`kl(flagged) - kl(twin)` to the SAME control set.  Every entry is the slope times "
        f"the arm's own sd of `H_pre`, with `(t)`.", "",
        tbl(rows, ["l", "step"] + ["base", "+ dpair7", "+ d kl", "+ both"] + ["n"]), ""])


def json_of_t(d, tag, st, tag2=""):
    sfx = f"_norm_{tag}" + (f"_{tag2}" if tag2 else "") + ".json"
    return os.path.join(d, f"step{st:06d}{sfx}")


# ---------------------------------------------------------------------------
# B. the glitch world beside the edit world, on the entropy axis
# ---------------------------------------------------------------------------

_WORLD_CACHE = {}
_TB_CACHE = {}


def tb_of(T, mask=None, key=None):
    """`phasic.balance_tokens` on a `Twins`, memoised (it is a deterministic greedy pass
    over tens of thousands of edges and several sections ask for the same subset)."""
    ck = (id(T), key)
    if ck not in _TB_CACHE:
        _TB_CACHE[ck] = T.token_balanced(mask=mask)
    return _TB_CACHE[ck]


def worlds_of(d, st, tag="swap65k", tag2="glitch", caliper=0.3):
    """Both worlds' `Twins` from one `_glitch` cell, plus `mediate.py`'s columns.
    Memoised per (dir, step, venue, tag, caliper); everything in it is deterministic."""
    ck = (d, st, tag, tag2, caliper)
    if ck in _WORLD_CACHE:
        return _WORLD_CACHE[ck]
    r_ = _worlds_of(d, st, tag, tag2, caliper)
    _WORLD_CACHE[ck] = r_
    return r_


def _worlds_of(d, st, tag="swap65k", tag2="glitch", caliper=0.3):
    p, pj = banked_path(d, tag, st, tag2), json_of_t(d, tag, st, tag2)
    if not (os.path.exists(p) and os.path.exists(pj)):
        return None, None, None
    r = json.load(open(pj))
    M = Med(d, tag, st, tag2)
    Ts, Ps = {}, {}
    for pre, key in (("tw__", "twin"), ("gl__", "glitch")):
        T = Twins(p, r, caliper=caliper, pre=pre, key=key)
        if not T.ok:
            continue
        P = M.pairs(pre, T.Z) if M.ok else None
        if P is None:
            continue
        Ts[pre], Ps[pre] = T, P
    return Ts, Ps, r


def sec_world_axis_guard(dirs, names, tag="swap65k", tag2="glitch", caliper=0.3):
    """The ACROSS-WORLD entropy guard, printed the way norm 5 printed the surprisal one."""
    out = []
    rows = []
    for st in STEPS:
        for d, nm in zip(dirs, names):
            Ts, Ps, r = worlds_of(d, st, tag, tag2, caliper)
            if not Ts:
                continue
            ma, mb, ncom = (None, None, 0)
            if "tw__" in Ts and "gl__" in Ts:
                ma, mb, ncom = common_windows(Ts["tw__"], Ts["gl__"])
            for pre in ("tw__", "gl__"):
                if pre not in Ts:
                    continue
                T, P = Ts[pre], Ps[pre]
                m = T.keep
                H = P["H_pre"]
                rows.append([st, nm, WORLD[pre], int(m.sum()),
                             fmt(float(H[m].mean()), 4), fmt(float(H[m].std()), 4),
                             fmt(float(np.quantile(H[m], 0.05)), 4),
                             fmt(float(np.quantile(H[m], 0.95)), 4),
                             fmt(float(T.s_v[m].mean()), 3),
                             fmt(float(np.abs(T.ds[m]).mean()), 4)])
    out += ["**The across-world guard on the entropy axis.**  `norm/README.md` 5 printed "
            "the surprisal version of this: the illegal tokens available under a CLEAN "
            "prefix are systematically more surprising than an edit's violator, and the "
            "grammar gives no way to close that.  The same has to be on the record for "
            "`H_pre` before any cross-world row is read, because the two worlds' events "
            "sit on DIFFERENT PREFIXES by construction -- the edit world's prefix carries "
            "the edit, the glitch world's is the untouched stream -- so their `H_pre` are "
            "not the same distribution and are not matched to each other.  Within a pair "
            "`H_pre` is identical in both worlds (`mediate.py` asserts 0.0).", "",
            tbl(rows, ["step", "seed", "world", "pairs @0.30", "mean H_pre", "sd H_pre",
                       "q05", "q95", "mean s (flagged)", "mean abs(ds) in pair"]), ""]
    rows = []
    for st in STEPS:
        for d, nm in zip(dirs, names):
            Ts, Ps, r = worlds_of(d, st, tag, tag2, caliper)
            if not Ts or "tw__" not in Ts or "gl__" not in Ts:
                continue
            ma, mb, ncom = common_windows(Ts["tw__"], Ts["gl__"])
            wa = np.asarray(Ts["tw__"].Z["tw__w"])[ma]
            wb = np.asarray(Ts["gl__"].Z["gl__w"])[mb]
            oa, ob = np.argsort(wa), np.argsort(wb)
            Ha = Ps["tw__"]["H_pre"][ma][oa]
            Hb = Ps["gl__"]["H_pre"][mb][ob]
            assert np.array_equal(wa[oa], wb[ob])
            rows.append([st, nm, ncom, fmt(float(Ha.mean()), 4), fmt(float(Hb.mean()), 4),
                         fmt(float((Ha - Hb).mean()), 4),
                         fmt(float(np.abs(Ha - Hb).mean()), 4),
                         fmt(_pearson(Ha, Hb), 3), fmt(_spear(Ha, Hb), 3)])
    if rows:
        out += ["On the COMMON windows (the same window index surviving both worlds' "
                "calipers), the two worlds' `H_pre` paired window by window:", "",
                tbl(rows, ["step", "seed", "common windows", "mean H edit",
                           "mean H glitch", "mean difference", "mean abs difference",
                           "r", "rho"]), ""]
    return "\n".join(out)


def sec_world_head_H(dirs, names, st, tag="swap65k", tag2="glitch", caliper=0.3,
                     diet="full"):
    rows = []
    for d, nm in zip(dirs, names):
        Ts, Ps, r = worlds_of(d, st, tag, tag2, caliper)
        if not Ts:
            continue
        for pre in ("tw__", "gl__"):
            if pre not in Ts:
                continue
            T, P = Ts[pre], Ps[pre]
            tb, sig = tb_of(T, key="own")
            g = r["glitch"] if pre == "gl__" else r["twin"]
            rows.append([nm, WORLD[pre], g.get("n_pairs"), int(T.keep.sum()),
                         int(tb.sum()), sig,
                         fmt(float(P["H_pre"][tb].mean()), 4),
                         fmt(float(P["H_pre"][tb].std()), 4),
                         fmt(float(T.s_v[tb].mean()), 3),
                         fmt(float(np.abs(T.ds[tb]).mean()), 4)])
    if not rows:
        return ""
    return "\n".join([
        f"**The two worlds' pair populations and their entropy — step {st}, {tag}, "
        f"caliper {caliper}, each world's OWN pairs.**  `signed hist` is the exact signed "
        f"token histogram on the token-balanced subset and must be 0.", "",
        tbl(rows, ["seed", "world", "pairs saved", "pairs @0.30", "token-balanced",
                   "signed hist", "mean H_pre", "sd H_pre", "mean s (flagged)",
                   "mean abs(ds)"]), ""])


def sec_world_bins(dirs, names, st, diet="full", nbin=4, tag="swap65k", tag2="glitch",
                   caliper=0.3, common=False):
    """`dR` by `H_pre` bin, per world, token-balanced."""
    got = []
    for d, nm in zip(dirs, names):
        Ts, Ps, r = worlds_of(d, st, tag, tag2, caliper)
        if not Ts or (common and not ("tw__" in Ts and "gl__" in Ts)):
            continue
        masks = {}
        if common:
            ma, mb, _ = common_windows(Ts["tw__"], Ts["gl__"])
            masks = {"tw__": ma, "gl__": mb}
        got.append((nm, Ts, Ps, masks))
    if not got:
        return ""
    rows = []
    for pre in ("tw__", "gl__"):
        for l in LS:
            for i in range(nbin):
                cells = [WORLD[pre], l, i + 1]
                for nm, Ts, Ps, masks in got:
                    if pre not in Ts or (diet, l, 0) not in Ts[pre].cols:
                        cells.append("--")
                        continue
                    T, P = Ts[pre], Ps[pre]
                    tb, _ = tb_of(T, masks.get(pre), key=("mask" if masks else "own"))
                    if tb.sum() < 40:
                        cells.append("--")
                        continue
                    H, dR = P["H_pre"][tb], T.dR(diet, l)[tb]
                    b = _qbin(H, nbin)
                    s = b == i
                    cells.append("--" if s.sum() < 10 else
                                 f"{dR[s].mean():.5f} +- {_sem(dR[s]):.5f} "
                                 f"[{H[s].mean():.2f}, n={int(s.sum())}]")
                rows.append(cells)
    arm = "common windows" if common else "each world's own pairs"
    return "\n".join([
        f"**`dR` by `H_pre` bin, both worlds — step {st}, critic `{diet}`, "
        f"token-balanced, {arm}; {' / '.join(nm for nm, _, _, _ in got)}.**  Each entry is "
        f"`mean dR +- sem [mean H_pre, n]`; bins are per-seed, per-world quantiles of "
        f"`H_pre`.", "", tbl(rows, ["world", "l", "H bin"]
                             + [nm for nm, _, _, _ in got]), ""])


def sec_world_slope(dirs, names, diet="full", tag="swap65k", tag2="glitch", caliper=0.3,
                    common=False):
    """The residualised `dR ~ H_pre` slope per world."""
    rows = []
    for pre in ("tw__", "gl__"):
        for l in LS:
            for st in STEPS:
                cells = [WORLD[pre], l, st]
                per, npr = [], []
                for d in dirs:
                    Ts, Ps, r = worlds_of(d, st, tag, tag2, caliper)
                    if not Ts or pre not in Ts or (diet, l, 0) not in Ts[pre].cols:
                        per.append("--")
                        npr.append("--")
                        continue
                    mask = None
                    if common:
                        if "tw__" not in Ts or "gl__" not in Ts:
                            per.append("--")
                            npr.append("--")
                            continue
                        ma, mb, _ = common_windows(Ts["tw__"], Ts["gl__"])
                        mask = ma if pre == "tw__" else mb
                    T, P = Ts[pre], Ps[pre]
                    tb, _ = tb_of(T, mask, key=("common" if mask is not None else "own"))
                    if tb.sum() < 50:
                        per.append("--")
                        npr.append(str(int(tb.sum())))
                        continue
                    H, dR = P["H_pre"][tb], T.dR(diet, l)[tb]
                    G = np.stack([T.ks[tb], T.jj[tb], T.tv[tb], T.s_v[tb],
                                  np.abs(T.ds[tb])], 1).astype(np.float64)
                    b, se, _ = _partial(dR, H, G)
                    per.append(fmt(b * float(H.std()), 5)
                               + (f" ({b / se:.1f})" if np.isfinite(se) and se > 0
                                  else ""))
                    npr.append(str(int(tb.sum())))
                cells.append(" / ".join(per))
                cells.append(" / ".join(npr))
                rows.append(cells)
    arm = "common windows" if common else "each world's own pairs"
    return "\n".join([
        f"**`dR ~ H_pre` residualised, both worlds — critic `{diet}`, token-balanced, "
        f"{arm}; {' / '.join(names)} side by side.**  Both sides residualised on "
        f"`(k*, j, position, flagged-token surprisal, |ds|)`; each entry is the slope times "
        f"that arm's own sd of `H_pre`, with `(t)`.", "",
        tbl(rows, ["world", "l", "step", "b.sdH (t)", "n"]), ""])


def sec_world_persist(dirs, names, st, diet="full", l=2, nbin=3, tag="swap65k",
                      tag2="glitch", caliper=0.3, T_max=63, common=False):
    """`dV` by offset inside `H_pre` terciles, per world -- the object norm 5 found the
    two worlds differ in."""
    out = []
    for d, nm in zip(dirs, names):
        Ts, Ps, r = worlds_of(d, st, tag, tag2, caliper)
        if not Ts:
            continue
        masks = {}
        if common and "tw__" in Ts and "gl__" in Ts:
            ma, mb, _ = common_windows(Ts["tw__"], Ts["gl__"])
            masks = {"tw__": ma, "gl__": mb}
        rows = []
        for pre in ("tw__", "gl__"):
            if pre not in Ts or (diet, l, 0) not in Ts[pre].cols:
                continue
            T, P = Ts[pre], Ps[pre]
            tb, _ = tb_of(T, masks.get(pre), key=("mask" if masks else "own"))
            if tb.sum() < 60:
                continue
            b = np.full(len(tb), -1)
            b[tb] = _qbin(P["H_pre"][tb], nbin)
            for i in range(nbin):
                sel0 = tb & (b == i)
                if sel0.sum() < 20:
                    continue
                cells = [WORLD[pre], i + 1, int(sel0.sum()),
                         fmt(float(P["H_pre"][sel0].mean()), 3)]
                for off in T.offs:
                    good = sel0 & ((T.tv + off) <= T_max) & ((T.tv + off) >= 0)
                    if good.sum() < 20:
                        cells.append("--")
                        continue
                    dv = (T.V("v", diet, l, 0, off)[good]
                          - T.V("t", diet, l, 0, off)[good])
                    cells.append(fmt(float(dv.mean()), 4))
                rows.append(cells)
        if rows:
            out += [f"**{nm}**", "",
                    tbl(rows, ["world", "H tercile", "n", "mean H_pre"]
                        + [f"off {o}" for o in Ts[list(Ts)[0]].offs]), ""]
    if not out:
        return ""
    arm = "common windows" if common else "each world's own pairs"
    return "\n".join([
        f"**Matched persistence `dV = V(flagged) - V(twin)` by offset, inside `H_pre` "
        f"terciles — step {st}, level {l}, critic `{diet}`, token-balanced, {arm}.**  "
        f"`norm/README.md` 5 measured the two worlds as differing in exactly this object "
        f"(offset 1, l = 2: edit +0.066, glitch -0.003) while agreeing at the event; the "
        f"terciles split it on the entropy axis.  Offset 0 is the only clean event reading "
        f"-- after `t_v` both members of a pair carry the same continuation.", ""] + out)


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def fig_mediators(d, tag, an, name, path, diet="full", nbin=8, tag2=""):
    plt = _mpl()
    show = ["kl_update", "dstate_post_block7", "H_next"]
    fig, ax = plt.subplots(1, len(show) + 2, figsize=(5.3 * (len(show) + 2), 4.6))
    for st, c in zip(STEPS, ["#999999", "#1f77b4", "#d62728"]):
        p = banked_path(d, tag, st, tag2)
        M = Med(d, tag, st, tag2)
        if not (os.path.exists(p) and M.ok):
            continue
        C = cell_of(p, an)
        R_ = M.rows(an, C.Z)
        if R_ is None:
            continue
        b = _qbin(C.H, nbin)
        xs = [float(C.H[b == i].mean()) for i in range(nbin) if (b == i).sum() > 10]
        for k, m in enumerate(show):
            if m not in R_:
                continue
            ax[k].plot(xs, [float(R_[m][b == i].mean()) for i in range(nbin)
                            if (b == i).sum() > 10], marker="o", color=c,
                       label=f"step {st}")
        # R by H, raw and residualised on the mediators
        G = np.stack([R_[m] for m in MED if m in R_], 1)
        for k, (lab, y) in enumerate([("raw", C.R(diet, 1)),
                                      ("resid on mediators", _resid(C.R(diet, 1), G))]):
            ax[len(show) + k].plot(
                xs, [float(y[b == i].mean()) for i in range(nbin) if (b == i).sum() > 10],
                marker="o", color=c, label=f"step {st}")
    for k, m in enumerate(show):
        ax[k].set_title(f"{MED_SHORT[m]} by H_pre bin", fontsize=9.5)
        ax[k].set_ylabel(m)
    ax[len(show)].set_title("R (l=1) by H_pre bin, raw", fontsize=9.5)
    ax[len(show) + 1].set_title("R (l=1) by H_pre bin,\nresidualised on the five mediators",
                                fontsize=9.5)
    for a_ in ax:
        a_.set_xlabel("H_pre bin (nats)")
        a_.legend(fontsize=7, framealpha=0.85)
        a_.axhline(0, c="k", lw=0.6)
    fig.suptitle(f"precision A: the mediators of the response's entropy dependence.  "
                 f"{name}, {tag}, anchor {an}, critic `{diet}`, all fixed rows.", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, dpi=130)
    print("wrote", path)


def fig_worlds_H(dirs, names, st, path, diet="full", nbin=4, tag="swap65k",
                 tag2="glitch", caliper=0.3, l_persist=2):
    plt = _mpl()
    col = {"tw__": "#d62728", "gl__": "#1f77b4"}
    fig, ax = plt.subplots(1, 5, figsize=(26.0, 4.6))
    for d, nm, ls in zip(dirs, names, ["-", "--", ":"]):
        Ts, Ps, r = worlds_of(d, st, tag, tag2, caliper)
        if not Ts:
            continue
        for pre in ("tw__", "gl__"):
            if pre not in Ts:
                continue
            T, P = Ts[pre], Ps[pre]
            tb, _ = tb_of(T, key="own")
            if tb.sum() < 40:
                continue
            H = P["H_pre"][tb]
            b = _qbin(H, nbin)
            for i, l in enumerate(LS):
                if (diet, l, 0) not in T.cols:
                    continue
                dR = T.dR(diet, l)[tb]
                xs = [float(H[b == k].mean()) for k in range(nbin) if (b == k).sum() > 10]
                ys = [float(dR[b == k].mean()) for k in range(nbin) if (b == k).sum() > 10]
                es = [_sem(dR[b == k]) for k in range(nbin) if (b == k).sum() > 10]
                ax[i].errorbar(xs, ys, yerr=es, marker="o", capsize=2, ls=ls,
                               color=col[pre], label=f"{WORLD[pre]}, {nm}")
    # persistence in H terciles, first seed
    Ts, Ps, r = worlds_of(dirs[0], st, tag, tag2, caliper)
    if Ts:
        for pre in ("tw__", "gl__"):
            if pre not in Ts or (diet, l_persist, 0) not in Ts[pre].cols:
                continue
            T, P = Ts[pre], Ps[pre]
            tb, _ = tb_of(T, key="own")
            if tb.sum() < 60:
                continue
            bb = np.full(len(tb), -1)
            bb[tb] = _qbin(P["H_pre"][tb], 3)
            for i, style in zip(range(3), [":", "--", "-"]):
                sel0 = tb & (bb == i)
                if sel0.sum() < 20:
                    continue
                xs, ys = [], []
                for off in T.offs:
                    good = sel0 & ((T.tv + off) <= 63) & ((T.tv + off) >= 0)
                    if good.sum() < 20:
                        continue
                    xs.append(off)
                    ys.append(float((T.V("v", diet, l_persist, 0, off)[good]
                                     - T.V("t", diet, l_persist, 0, off)[good]).mean()))
                ax[4].plot(xs, ys, marker="o", ls=style, color=col[pre],
                           label=f"{WORLD[pre]}, H tercile {i + 1}")
    for i, l in enumerate(LS):
        ax[i].axhline(0, c="k", lw=0.7)
        ax[i].set_title(f"dR by H_pre, level {l}", fontsize=9.5)
        ax[i].set_xlabel("H_pre bin (nats)")
        ax[i].set_ylabel("dR")
        ax[i].legend(fontsize=6, framealpha=0.85)
    ax[4].axhline(0, c="k", lw=0.7)
    ax[4].set_title(f"persistence dV by offset, l = {l_persist},\n{names[0]}, H terciles",
                    fontsize=9.5)
    ax[4].set_xlabel("offset from t_v")
    ax[4].set_ylabel("dV")
    ax[4].legend(fontsize=6, framealpha=0.85)
    fig.suptitle(f"precision B: the entropy axis in the glitch world beside the edit world."
                 f"  {tag}, step {st}, critic `{diet}`, token-balanced, each world's own "
                 f"pairs.  The two worlds' H_pre are NOT matched to each other (different "
                 f"prefixes); see the across-world guard.", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, dpi=130)
    print("wrote", path)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

PRE_A = """# precision A — what carries the response's dependence on the pre-event entropy

Facts only; interpretation is discussed with Jasper and lives nowhere in this folder.
Generated by [`mediation.py`](../mediation.py) from the banked `norm/` cells plus
[`mediate.py`](../mediate.py)'s recomputed event columns.  Nothing is refit, no GPU is
touched here, and [`results/tables_20260922.md`](tables_20260922.md) is not rewritten.
Three trajectory seeds side by side, **never averaged**.

**The question.**  `tables_20260922.md` 3 found that at matched surprisal the value
response `R` rises with `H_pre` at levels 1-3 (b.sdH +0.031 / +0.022 / +0.008 at 64k on
`swap65k` `t_v`, all three seeds), that the `z(s) x z(H)` interaction is positive at
levels 1-2, and that `H_pre` itself ranks realised damage at 0.62-0.65 at level 1 on rows
where the model's surprisal is pinned at 0.500.  This file asks what that dependence runs
through.

**The mediators** (all on the SAME banked rows, gated per cell inside `mediate.py`):

| column | what it is |
|---|---|
| `kl` | `kl_update = KL(q_t \\|\\| q_(t-1))`, `phasic.py`'s object: the model's FORECAST movement across the event |
| `Hnext` | `H(q_t)`, the entropy after the event, beside `H_pre` before it |
| `dstate7` | `\\|\\|s_t - s_(t-1)\\|\\|_2` at `post_block7`, the block `norm/task.py` fits the critic on: the REPRESENTATION's movement |
| `dstate3`, `dstate_emb` | the same at `post_block3` and `post_embed`, the cheap insurance |
| `dpair7` (pairs) | `\\|\\|s_t^flagged - s_t^twin\\|\\|_2`, the state difference whose linear projection is `dR` |

**Units.**  Every slope on the entropy axis is reported as **`b . sd(H_pre)`** -- the
readout's change over one sd of the axis on that cell's own rows -- because the random-init
trunk's forecast is near-uniform on every row and its `H_pre` has almost no spread, so raw
slopes are not comparable across checkpoints (`tables_20260922.md` 1).  Mediator slopes are
additionally divided by the mediator's own sd, so the five are comparable with each other.

**Rows.**  The npz holds the `split == 2` TEST windows only and the fixed rows are identical
across the three checkpoints of a trajectory; the matched sets are
`junction/analyze.py`'s own, recomputed unchanged, with the surprisal guard printed on
every table.
"""

PRE_B = """# precision B — the entropy axis in the glitch world beside the edit world

Facts only.  Generated by [`mediation.py`](../mediation.py) from the `_glitch`-tagged
`norm/` cells plus [`mediate.py`](../mediate.py)'s recomputed columns, which are the only
source of the glitch world's `H_pre`: a glitch is flagged under the **clean** prefix, so no
join against the edit world's fixed rows can supply it.

**The two worlds** (`norm/README.md` 5): the same windows at the same `t_v`, the flagged
token replaced by the legal token of nearest model surprisal after a bit-identical prefix.
In the **edit** world the prefix carries the swap and the continuation is the swap's
regrown subtree, which belongs to the violator; in the **glitch** world the prefix is the
untouched stream and the continuation belongs to a legal token, not to the glitch.  The
banked reading: the event-level optimism is the same size in both worlds (at 8k on all
three seeds; the 64k cells do not replicate), and the **persistence** dissociates -- eight
tokens of held advantage in the edit world, none in the glitch world.

**The across-world guard comes first.**  `norm/README.md` 5 printed the surprisal version
(a glitch under a clean prefix is ~2.7 nats more surprising than an edit's violator, and
the grammar gives no way to close it).  The entropy version is section B0 and has to be read
before any cross-world row: the two worlds' events sit on **different prefixes**, so their
`H_pre` are different distributions and are not matched to each other.  Within a pair
`H_pre` is identical in both worlds, which `mediate.py` asserts at 0.0.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--tags", default="swap65k,a1")
    ap.add_argument("--anchors", default="tv,fd")
    ap.add_argument("--out", default=".")
    ap.add_argument("--figs", default="")
    ap.add_argument("--date", default="20260922")
    ap.add_argument("--diet", default="full")
    ap.add_argument("--a", type=int, default=0)
    ap.add_argument("--glitch-tag", default="glitch")
    ap.add_argument("--skip-a", action="store_true")
    ap.add_argument("--skip-b", action="store_true")
    args = ap.parse_args()

    dirs = args.dirs
    names = [os.path.basename(os.path.normpath(x)) for x in dirs]
    tags = [t for t in args.tags.split(",") if t]
    ancs = [t for t in args.anchors.split(",") if t]
    os.makedirs(args.out, exist_ok=True)

    if not args.skip_a:
        P = [PRE_A, sec_gates(dirs, names, tags)]
        for tag in tags:
            for an in ancs:
                if not os.path.exists(banked_path(dirs[0], tag, STEPS[0])):
                    continue
                C = cell_of(banked_path(dirs[0], tag, STEPS[0]), an)
                if not C.ok:
                    continue
                P.append(f"\n---\n\n# {tag}, anchor {an}\n")
                P.append("## A1. The premise: do the mediators carry the axis?\n")
                P.append(sec_med_axis(dirs, names, tag, an))
                P.append(sec_med_auc(dirs, names, tag, an, args.a))
                P.append("## A2. The mediation\n")
                P.append(sec_collinear(dirs, names, tag, an))
                P.append(sec_mediation(dirs, names, tag, an, args.diet, args.a, True))
                P.append(sec_mediation(dirs, names, tag, an, args.diet, args.a, False))
                P.append("## A3. Both ways\n")
                P.append(sec_both_ways(dirs, names, tag, an, args.diet, args.a,
                                       matched=True))
                P.append(sec_both_ways(dirs, names, tag, an, args.diet, args.a,
                                       matched=False))
                P.append("## A4. The interaction\n")
                P.append(sec_interaction_med(dirs, names, tag, an, args.diet, args.a,
                                             True))
                P.append(sec_interaction_med(dirs, names, tag, an, args.diet, args.a,
                                             False))
                P.append(f"### The 2-D surface residualised on the mediators, {names[0]}, "
                         f"all fixed rows\n")
                P.append(sec_surface_resid(dirs[0], tag, an, names[0], args.diet, args.a))
                P.append("## A6. The damage side\n")
                P.append(sec_damage_cond(dirs, names, tag, an, args.a))
            if any(os.path.exists(banked_path(d, tag, STEPS[0])) for d in dirs):
                P.append(f"\n---\n\n# {tag} — A5. The twins\n")
                P.append(twin_med_slope(dirs, names, tag, args.diet, True))
                P.append(twin_med_slope(dirs, names, tag, args.diet, False))
        p = os.path.join(args.out, f"tables_mediation_{args.date}.md")
        with open(p, "w") as f:
            f.write("\n".join(P).rstrip() + "\n")
        print(f"wrote {p}  ({os.path.getsize(p) / 1024:.0f} KB)")

    if not args.skip_b:
        g = args.glitch_tag
        P = [PRE_B, "\n---\n\n## B0. The across-world entropy guard\n",
             sec_world_axis_guard(dirs, names, "swap65k", g),
             sec_gates(dirs, names, ["swap65k"], g),
             "\n---\n\n## B1. The pair populations\n"]
        for st in STEPS:
            s = sec_world_head_H(dirs, names, st, "swap65k", g, diet=args.diet)
            if s:
                P.append(s)
        P.append("\n---\n\n## B2. `dR` by `H_pre` bin, both worlds\n")
        for st in STEPS:
            for common in (False, True):
                s = sec_world_bins(dirs, names, st, args.diet, 4, "swap65k", g,
                                   common=common)
                if s:
                    P.append(s)
        P.append("\n---\n\n## B3. The residualised slope, both worlds\n")
        P.append(sec_world_slope(dirs, names, args.diet, "swap65k", g, common=False))
        P.append(sec_world_slope(dirs, names, args.diet, "swap65k", g, common=True))
        P.append("\n---\n\n## B4. The persistence inside `H_pre` terciles\n")
        for st in (64000, 8000):
            for l in (2, 3):
                s = sec_world_persist(dirs, names, st, args.diet, l, 3, "swap65k", g)
                if s:
                    P.append(s)
        P.append("\n---\n\n## B5. `dR ~ H_pre` with the pair mediators, glitch world\n")
        P.append(twin_med_slope(dirs, names, "swap65k", args.diet, True, pre="gl__",
                                tag2=g))
        P.append(twin_med_slope(dirs, names, "swap65k", args.diet, True, pre="tw__",
                                tag2=g))
        p = os.path.join(args.out, f"tables_glitch_{args.date}.md")
        with open(p, "w") as f:
            f.write("\n".join(P).rstrip() + "\n")
        print(f"wrote {p}  ({os.path.getsize(p) / 1024:.0f} KB)")

    if args.figs:
        os.makedirs(args.figs, exist_ok=True)
        if not args.skip_a:
            for tag in tags:
                for an in ancs:
                    try:
                        fig_mediators(dirs[0], tag, an, names[0],
                                      os.path.join(args.figs,
                                                   f"prec_med_{tag}_{an}.png"), args.diet)
                    except Exception as e:                              # noqa: BLE001
                        print(f"fig_mediators {tag} {an}: {type(e).__name__} {e}")
        if not args.skip_b:
            for st in (64000, 8000):
                try:
                    fig_worlds_H(dirs, names, st,
                                 os.path.join(args.figs, f"prec_worldsH_{st}.png"),
                                 args.diet, tag2=args.glitch_tag)
                except Exception as e:                                  # noqa: BLE001
                    print(f"fig_worlds_H {st}: {type(e).__name__} {e}")


if __name__ == "__main__":
    main()
