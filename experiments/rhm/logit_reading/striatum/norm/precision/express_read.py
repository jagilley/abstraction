"""The expressivity test, reduced.  Facts only; no interpretation lives here.

Reads `express.py`'s `<stem>_norm_<tag>_express.{json,npz}` -- the critic refit on the same
trunk, rows, split and lambda ladder as `norm/task.py`, with the feature map augmented -- and
puts every arm beside the banked ridge (arm 0, which the cell has already gated to reproduce
the banked columns) on the identical test rows.

The express npz carries every base column of the banked npz, RECOMPUTED and gated equal
(`w`, `t0`, the outcome labels exactly; `nll_e`, `H_pre` to float32), so
`reduce.py`'s `cell_of` / `matched_sets` read it unchanged: the matched sets are
`junction/analyze.py`'s own and are asserted identical to the banked ones here.

  python -m rhm.logit_reading.striatum.norm.precision.express_read \
      <dir>/traj_a1_s42 <dir>/traj_a1_s43 <dir>/traj_a1_s44 \
      --out rhm/logit_reading/striatum/norm/precision/results \
      --figs rhm/logit_reading/striatum/norm/precision/figs --date 20260922
"""

import argparse
import json
import os

import numpy as np

from rhm.logit_reading.striatum.analyze import _auc, _cond_auc, fmt, tbl
from rhm.logit_reading.striatum.junction.analyze import Rows
from rhm.logit_reading.striatum.norm.analyze import Twins, _ols, _sem
from rhm.logit_reading.striatum.norm.precision.reduce import (
    STEPS, LS, cell_of, matched_sets, _ols_multi, _partial, _qbin, _resid, _z)

ARMS = ["full", "fullH", "fullQ", "fullN", "fullT", "fullP", "fullM"]
CLEAN = ["clean", "cleanH", "cleanQ", "cleanN", "cleanT", "cleanP"]
LABEL = {"full": "ridge (banked)", "fullH": "+H", "fullQ": "+log q", "fullN": "+norm (plac.)",
         "fullT": "+tanh (plac.)", "fullP": "+precision", "fullM": "MLP",
         "fullL": "ln (in place)", "fullLp": "+ln", "fullS": "std (in place)",
         "clean": "clean ridge", "cleanH": "clean +H", "cleanQ": "clean +log q",
         "cleanN": "clean +norm", "cleanT": "clean +tanh", "cleanP": "clean +precision",
         "cleanL": "clean ln", "cleanLp": "clean +ln", "cleanS": "clean std"}
SFX = "_express"          # `--sfx _express_ln` reads follow-up 4's cells


def xp(d, tag, st):
    return os.path.join(d, f"step{st:06d}_norm_{tag}{SFX}.npz")


def xj(d, tag, st):
    return os.path.join(d, f"step{st:06d}_norm_{tag}{SFX}.json")


def bp(d, tag, st):
    return os.path.join(d, f"step{st:06d}_norm_{tag}.npz")


def bj(d, tag, st):
    return os.path.join(d, f"step{st:06d}_norm_{tag}.json")


def has(C, arm, l=1, a=0):
    return C.g(f"R_{arm}_l{l}_a{a}") is not None


def per_seed(dirs, fn):
    """`fn(d)` per seed, joined ' / ' in seed order."""
    return " / ".join(fn(d) for d in dirs)


# ---------------------------------------------------------------------------
# E0. gates
# ---------------------------------------------------------------------------

def sec_gates(dirs, names, tags):
    rows = []
    for d, nm in zip(dirs, names):
        for tag in tags:
            for st in STEPS:
                p = xj(d, tag, st)
                if not os.path.exists(p):
                    rows.append([nm, tag, st] + ["MISSING"] * 8)
                    continue
                r = json.load(open(p))
                g = r["gates"]
                rw = g["rows"]
                rows.append([
                    nm, tag, st,
                    f"{max(g['actor'].values()):.1e}",
                    f"{max(g['val_r2'].values()):.1e}",
                    str(all(v for k, v in rw.items() if isinstance(v, bool))),
                    f"{max([v for k, v in rw.items() if not isinstance(v, bool)] or [0]):.1e}",
                    f"{max(v for k, v in g['cols'].items() if k.endswith('/full')):.1e}",
                    f"{max(v for k, v in g['cols'].items() if k.endswith('/clean')):.1e}",
                    (f"{g['cols']['twins/arm0']:.1e}" if "twins/arm0" in g["cols"] else "--"),
                    "PASS" if not r["gate_fails"] else "FAIL: " + ",".join(r["gate_fails"])])
    return "\n".join([
        "**Gates.**  Every cell refits the banked ridge (arm 0) from scratch on the same "
        "trunk, split and lambda ladder, and before writing anything compares it with the "
        "bank: the frozen actor's clean accuracy (max abs difference over levels); arm 0's "
        "held-out `R2` on every level-offset column against the banked json; the test rows and "
        "every outcome label (exact); `nll_e`, `H_pre` (float32); arm 0's `V`, `Vpre`, `R`, "
        "`Ro`, `Vpreo`, `Rsh` against the banked `full` and `clean` columns at both anchors, "
        "every level and offset; and the twins' arm-0 `V` at offsets 0 and -1.  A cell with "
        "any failure raises and is not read.", "",
        tbl(rows, ["seed", "venue", "step", "actor", "val R2", "rows & labels exact",
                   "nll / H_pre", "full cols", "clean cols", "twins", "verdict"]), ""])


# ---------------------------------------------------------------------------
# E1. the held-out fit
# ---------------------------------------------------------------------------

def sec_fit(dirs, names, tag, arms, dd=0):
    rows = []
    for st in STEPS:
        for l in LS:
            cells = [st, l]
            base = {}
            for arm in arms:
                per = []
                for d in dirs:
                    p = xj(d, tag, st)
                    if not os.path.exists(p):
                        per.append("--")
                        continue
                    r = json.load(open(p))
                    v = r["val_r2"].get(arm, {}).get(f"l{l}_d{dd}")
                    if v is None:
                        per.append("--")
                        continue
                    if arm == arms[0]:
                        base[d] = v
                        per.append(fmt(v, 4))
                    else:
                        per.append(f"{v:.4f} ({v - base.get(d, float('nan')):+.4f})")
                cells.append(" / ".join(per))
            rows.append(cells)
    return "\n".join([
        f"**Held-out fit `R2` of `V[l, d={dd}]` — {tag}; {' / '.join(names)} side by side.**  "
        f"The same held-out validation rows and per-column lambda selection as `norm/task.py`; "
        f"each augmented arm's entry is `R2 (difference from {LABEL[arms[0]]})`.", "",
        tbl(rows, ["step", "l"] + [LABEL[a] for a in arms]), ""])


# ---------------------------------------------------------------------------
# E2. the level and the outcome along the axis
# ---------------------------------------------------------------------------

def sec_level_outcome(dirs, names, tag, an, arms, a=0):
    rows = []
    for l in LS:
        for st in STEPS:
            cells = [l, st]
            # the outcome is the same object for every arm
            def out_fn(d):
                p = xp(d, tag, st)
                if not os.path.exists(p):
                    return "--"
                C = cell_of(p, an)
                base = C.g(f"ok_l{l}_a{a}").astype(bool)
                oe = C.g(f"oe_l{l}_a{a}").astype(np.float64)
                i = np.where(base)[0]
                b, se, _ = _partial(oe[i], C.H[i], C.ctrl(idx=i))
                return fmt(b * float(C.H[i].std()), 4)
            cells.append(per_seed(dirs, out_fn))
            for arm in arms:
                def fn(d, arm=arm):
                    p = xp(d, tag, st)
                    if not os.path.exists(p):
                        return "--"
                    C = cell_of(p, an)
                    if not has(C, arm, l, a):
                        return "--"
                    base = C.g(f"ok_l{l}_a{a}").astype(bool)
                    i = np.where(base)[0]
                    vp = C.Vpre(arm, l, a)
                    b, se, _ = _partial(vp[i], C.H[i], C.ctrl(idx=i))
                    return fmt(b * float(C.H[i].std()), 4)
                cells.append(per_seed(dirs, fn))
            rows.append(cells)
    return "\n".join([
        f"**The norm against the outcome along the axis — {tag}, anchor {an}, a = {a}; "
        f"{' / '.join(names)} side by side.**  Partial slope on `H_pre` with both sides "
        f"residualised on `(k*, j, t0, nll_e)`, **times the cell's sd of `H_pre`**, on the rows "
        f"whose query exists.  The first column is the realised outcome (the frozen actor's "
        f"hit) and is the same for every arm; the rest are each arm's `V_pre`.  A reader that "
        f"tracks the outcome along the axis has its `V_pre` column equal to the outcome "
        f"column.", "",
        tbl(rows, ["l", "step", "OUTCOME"] + [f"V_pre {LABEL[a]}" for a in arms]), ""])


def sec_delta(dirs, names, tag, an, arms, a=0):
    rows = []
    for l in LS:
        for st in STEPS:
            cells = [l, st]
            for arm in arms:
                def fn(d, arm=arm):
                    p = xp(d, tag, st)
                    if not os.path.exists(p):
                        return "--"
                    C = cell_of(p, an)
                    if not has(C, arm, l, a):
                        return "--"
                    base = C.g(f"ok_l{l}_a{a}").astype(bool)
                    i = np.where(base)[0]
                    dl = C.g(f"oe_l{l}_a{a}").astype(np.float64) - C.Vpre(arm, l, a)
                    b, se, _ = _partial(dl[i], C.H[i], C.ctrl(idx=i))
                    return (fmt(b * float(C.H[i].std()), 4)
                            + (f" ({b / se:.1f})" if np.isfinite(se) and se > 0 else ""))
                cells.append(per_seed(dirs, fn))
            rows.append(cells)
    return "\n".join([
        f"**The outcome surprise along the axis — {tag}, anchor {an}, a = {a}; "
        f"{' / '.join(names)} side by side.**  `delta = outcome - V_pre` for each arm; partial "
        f"slope on `H_pre` | `(k*, j, t0, nll_e)`, times sd of `H_pre`, `(t)` beside.  Zero "
        f"means the arm's norm falls along the axis exactly as the outcome does.", "",
        tbl(rows, ["l", "step"] + [LABEL[a] for a in arms]), ""])


# ---------------------------------------------------------------------------
# E3. the response at matched surprisal
# ---------------------------------------------------------------------------

def sec_response(dirs, names, tag, an, arms, a=0, kind="flip", withV=False):
    rows = []
    for l in LS:
        for st in STEPS:
            cells = [l, st]
            for arm in arms:
                def fn(d, arm=arm):
                    p = xp(d, tag, st)
                    if not os.path.exists(p):
                        return "--"
                    C = cell_of(p, an)
                    if not has(C, arm, l, a):
                        return "--"
                    Rw, mm = matched_sets(p, an, kind, a)
                    if Rw is None or l not in mm:
                        return "--"
                    i = mm[l][0]
                    ex = [C.Vpre(arm, l, a)[i]] if withV else None
                    b, se, _ = _partial(C.R(arm, l, a)[i], C.H[i], C.ctrl(extra=ex, idx=i))
                    return (fmt(b * float(C.H[i].std()), 4)
                            + (f" ({b / se:.1f})" if np.isfinite(se) and se > 0 else ""))
                cells.append(per_seed(dirs, fn))
            rows.append(cells)
    what = "given the arm's own `V_pre`" if withV else "without `V_pre`"
    return "\n".join([
        f"**`R ~ H_pre` at matched surprisal, {what} — {tag}, anchor {an}, a = {a}, matched "
        f"`{kind}` rows; {' / '.join(names)} side by side.**  Partial slope residualised on "
        f"`(k*, j, t0, nll_e)`" + (" plus the arm's own `V_pre`" if withV else "")
        + ", times sd of `H_pre`, `(t)` beside.", "",
        tbl(rows, ["l", "step"] + [LABEL[a] for a in arms]), ""])


def sec_interaction(dirs, names, tag, an, arms, a=0, kind="flip"):
    rows = []
    for l in LS:
        for st in STEPS:
            cells = [l, st]
            for arm in arms:
                def fn(d, arm=arm):
                    p = xp(d, tag, st)
                    if not os.path.exists(p):
                        return "--"
                    C = cell_of(p, an)
                    if not has(C, arm, l, a):
                        return "--"
                    Rw, mm = matched_sets(p, an, kind, a)
                    if Rw is None or l not in mm:
                        return "--"
                    i = mm[l][0]
                    zs, zh = _z(C.s[i]), _z(C.H[i])
                    X = np.stack([zs, zh, zs * zh, C.ks[i].astype(float), C.jj[i].astype(float),
                                  C.t0[i].astype(float)], 1)
                    cf, se, _ = _ols_multi(C.R(arm, l, a)[i], X)
                    return fmt(cf[2], 4) + (f" ({cf[2] / se[2]:.1f})" if se[2] > 0 else "")
                cells.append(per_seed(dirs, fn))
            rows.append(cells)
    return "\n".join([
        f"**The `z(s) x z(H)` interaction on each arm's `R` — {tag}, anchor {an}, matched "
        f"`{kind}` rows; {' / '.join(names)} side by side.**  `tables_20260922.md` 3's fit "
        f"(`z(s) + z(H) + z(s).z(H)` with `(k*, j, t0)` linear), the interaction coefficient "
        f"and its `(t)`.", "",
        tbl(rows, ["l", "step"] + [LABEL[a] for a in arms]), ""])


def sec_damage_terciles(dirs, names, tag, an, arms, a=0, kind="flip", levels=(1, 2),
                        steps=(8000, 64000)):
    rows = []
    for l in levels:
        for st in steps:
            for what in ("R", "V"):
                for tb in ("all", 1, 2, 3):
                    cells = [l, st, what, tb]
                    for arm in arms:
                        def fn(d, arm=arm):
                            p = xp(d, tag, st)
                            if not os.path.exists(p):
                                return "--"
                            C = cell_of(p, an)
                            if not has(C, arm, l, a):
                                return "--"
                            Rw, mm = matched_sets(p, an, kind, a)
                            if Rw is None or l not in mm:
                                return "--"
                            i, lab = mm[l]
                            y = lab[i]
                            sc = -(C.R(arm, l, a) if what == "R" else C.V(arm, l, a))[i]
                            if tb == "all":
                                return fmt(_auc(sc, y))
                            b = _qbin(C.H[i], 3)
                            s = b == (tb - 1)
                            return fmt(_auc(sc[s], y[s]))
                        cells.append(per_seed(dirs, fn))
                    rows.append(cells)
    return "\n".join([
        f"**Realised-damage AUC of each arm's response and level, pooled and inside `H_pre` "
        f"terciles — {tag}, anchor {an}, matched `{kind}` rows; {' / '.join(names)} side by "
        f"side.**  `AUC(-R, flip)` and `AUC(-V, flip)`, the banked orientation; tercile 1 is "
        f"the most confident context.", "",
        tbl(rows, ["l", "step", "score", "H tercile"] + [LABEL[a] for a in arms]), ""])


# ---------------------------------------------------------------------------
# E4. A6 on each arm: does the reader now contain what the axis knew?
# ---------------------------------------------------------------------------

def sec_contain(dirs, names, tag, an, arms, a=0, kind="flip", nbin=5):
    rows = []
    for l in LS:
        for st in STEPS:
            for what in ("H given V", "H given R", "H resid on V,R", "V given H", "R given H"):
                cells = [l, st, what]
                for arm in arms:
                    def fn(d, arm=arm, what=what):
                        p = xp(d, tag, st)
                        if not os.path.exists(p):
                            return "--"
                        C = cell_of(p, an)
                        if not has(C, arm, l, a):
                            return "--"
                        Rw, mm = matched_sets(p, an, kind, a)
                        if Rw is None or l not in mm:
                            return "--"
                        i, lab = mm[l]
                        y = lab[i]
                        H, V, R = C.H[i], C.V(arm, l, a)[i], C.R(arm, l, a)[i]
                        if what == "H given V":
                            return fmt(_cond_auc(H, V, y, nbin))
                        if what == "H given R":
                            return fmt(_cond_auc(H, R, y, nbin))
                        if what == "H resid on V,R":
                            return fmt(_auc(_resid(H, np.stack([V, R], 1)), y))
                        if what == "V given H":
                            return fmt(_cond_auc(-V, H, y, nbin))
                        return fmt(_cond_auc(-R, H, y, nbin))
                    cells.append(per_seed(dirs, fn))
                rows.append(cells)
    return "\n".join([
        f"**Does the arm's reader now contain what the entropy axis knew about damage? — "
        f"{tag}, anchor {an}, a = {a}, matched `{kind}` rows; {' / '.join(names)} side by "
        f"side.**  `H given V` is `AUC(H_pre, {kind})` inside {nbin} quantile bins of the arm's post-"
        f"event level, pooled with weight `n_pos * n_neg` (`_cond_auc`); `H given R` the same inside "
        f"bins of the arm's response; `H resid on V,R` ranks `H_pre` after linear residualisation "
        f"on both.  The last two rows are the reverse: the arm's `-V` and `-R` inside bins of "
        f"`H_pre`.  The marginal `AUC(H_pre)` is the same for every arm (`tables_20260922.md` "
        f"3: 0.620 / 0.625 / 0.628 at ell = 1, 64k, `swap65k` `t_v`).", "",
        tbl(rows, ["l", "step", "reading"] + [LABEL[a] for a in arms]), ""])


# ---------------------------------------------------------------------------
# E5. striatum's headline rows, so the cost of the augmentation is on the record
# ---------------------------------------------------------------------------

def sec_legality(dirs, names, arms, tag="a1", an="fd", a=0):
    rows = []
    for st in STEPS:
        for mode in ("unmatched", "jt"):
            for l in (1, 2, 3):
                cells = [st, mode, l]
                n_per = []
                for arm in arms:
                    def fn(d, arm=arm):
                        p = xp(d, tag, st)
                        if not os.path.exists(p):
                            return "--"
                        R = Rows(p, an)
                        if not R.ok:
                            return "--"
                        sc = R.scores(arm, l, a)
                        if "R" not in sc:
                            return "--"
                        if mode == "unmatched":
                            lab, base = R.label("legal", l, a)
                            if not base.sum() or not 0 < lab[base].mean() < 1:
                                return "--"
                            return fmt(_auc(sc["R"][base], lab[base]))
                        got = R.matched("legal", l, a, "jt")
                        if got is None:
                            return "--"
                        mm_, lab = got
                        return fmt(_auc(sc["R"][mm_], lab[mm_]))
                    cells.append(per_seed(dirs, fn))
                rows.append(cells)
    return "\n".join([
        f"**Legality at matched consequence — {tag}, anchor {an} (the only venue and anchor "
        f"with a legal rare edit); {' / '.join(names)} side by side.**  Swap against rare "
        f"among CONSEQUENTIAL episodes, scored by `-R` (junction's `R.scores`): "
        f"`unmatched` is `striatum/`'s banked contrast verbatim (`sec_legal_unmatched`), "
        f"`jt` the strict one matched on `(j, t0)` and surprisal.  0.500 is a reader "
        f"indifferent to legality at matched consequence -- the arc's law.", "",
        tbl(rows, ["step", "contrast", "l"] + [LABEL[a] for a in arms]), ""])


# ---------------------------------------------------------------------------
# E6. the twins
# ---------------------------------------------------------------------------

def _twins_arm(d, tag, st, caliper=0.3):
    p, pj, x = bp(d, tag, st), bj(d, tag, st), xp(d, tag, st)
    if not (os.path.exists(p) and os.path.exists(pj) and os.path.exists(x)):
        return None
    T = Twins(p, json.load(open(pj)), caliper=caliper)
    X = np.load(x)
    if not T.ok or "tw_w" not in X.files:
        return None
    assert np.array_equal(np.asarray(X["tw_w"]), np.asarray(T.Z["tw__w"])), "pair order"
    hp = os.path.join(d, f"step{st:06d}_norm_{tag}_Hpre.npz")
    H = np.asarray(np.load(hp)["tw_H_pre"], np.float64) if os.path.exists(hp) else None
    return T, X, H


_TWC = {}


def twins_arm(d, tag, st):
    k = (d, tag, st)
    if k not in _TWC:
        _TWC[k] = _twins_arm(d, tag, st)
        if _TWC[k] is not None:
            T = _TWC[k][0]
            _TWC[k] = _TWC[k] + (T.token_balanced()[0],)
    return _TWC[k]


def sec_twins(dirs, names, tag, arms, steps=(8000, 64000)):
    rows = []
    for l in LS:
        for st in steps:
            for what in ("mean dR", "dR ~ H_pre"):
                cells = [l, st, what]
                for arm in arms:
                    def fn(d, arm=arm, what=what):
                        got = twins_arm(d, tag, st)
                        if got is None:
                            return "--"
                        T, X, H, tb = got
                        kv, kt = f"tw_ev_{arm}_l{l}_v", f"tw_ev_{arm}_l{l}_t"
                        if kv not in X.files:
                            return "--"
                        dR = (np.asarray(X[kv], np.float64) - np.asarray(X[kt], np.float64))[tb]
                        if what == "mean dR":
                            return f"{dR.mean():+.4f} ({dR.mean() / max(_sem(dR), 1e-12):.1f})"
                        if H is None:
                            return "--"
                        h = H[tb]
                        G = np.stack([T.ks[tb], T.jj[tb], T.tv[tb], T.s_v[tb],
                                      np.abs(T.ds[tb])], 1).astype(np.float64)
                        b, se, _ = _partial(dR, h, G)
                        return (fmt(b * float(h.std()), 5)
                                + (f" ({b / se:.1f})" if np.isfinite(se) and se > 0 else ""))
                    cells.append(per_seed(dirs, fn))
                rows.append(cells)
    return "\n".join([
        f"**The same-prefix twins on each arm — {tag}, token-balanced; "
        f"{' / '.join(names)} side by side.**  `dR = V(violator) - V(legal twin)` at the event "
        f"under each arm (the prefix is bit-identical, so `V_pre` cancels exactly); "
        f"`mean dR (dR/sem)` is `norm/README.md` 3's row, and `dR ~ H_pre` is "
        f"`tables_20260922.md` 5's residualised slope (controls `(k*, j, position, violator "
        f"surprisal, |ds|)`) times sd of `H_pre`, `(t)`.", "",
        tbl(rows, ["l", "step", "reading"] + [LABEL[a] for a in arms]), ""])


def sec_matched_identity(dirs, names, tags, ancs):
    """The express npz's matched sets must be the banked ones (the base columns are gated
    equal, so this is a consequence, checked rather than assumed)."""
    rows = []
    for d, nm in zip(dirs, names):
        for tag in tags:
            for st in STEPS:
                for an in ancs:
                    x, b = xp(d, tag, st), bp(d, tag, st)
                    if not (os.path.exists(x) and os.path.exists(b)):
                        continue
                    _, mx = matched_sets(x, an)
                    _, mb = matched_sets(b, an)
                    same = all(l in mb and np.array_equal(mx[l][0], mb[l][0]) for l in mx) \
                        and set(mx) == set(mb)
                    rows.append([nm, tag, st, an, " / ".join(str(len(mx[l][0])) for l in LS
                                                              if l in mx), str(same)])
    return "\n".join([
        "**The matched sets read off the express npz are the banked ones** (`junction`'s "
        "matcher on the recomputed base columns against the same matcher on the banked npz; "
        "matched n per level 1-4).", "",
        tbl(rows, ["seed", "venue", "step", "anchor", "matched n", "identical"]), ""])


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


COLS = {"full": "#444444", "fullH": "#d62728", "fullQ": "#1f77b4", "fullN": "#aaaaaa",
        "fullT": "#cccccc", "fullP": "#9467bd", "fullM": "#2ca02c", "fullL": "#ff7f0e",
        "fullLp": "#8c564b", "fullS": "#e377c2"}


def fig_fit(dirs, names, tag, path):
    plt = _mpl()
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
    for k, st in enumerate(STEPS):
        for arm in ARMS:
            for j, d in enumerate(dirs):
                p = xj(d, tag, st)
                if not os.path.exists(p):
                    continue
                r = json.load(open(p))
                y = [r["val_r2"].get(arm, {}).get(f"l{l}_d0", np.nan) for l in LS]
                ax[k].plot(LS, y, marker="os^"[j], color=COLS[arm], alpha=0.85,
                           label=LABEL[arm] if j == 0 else None)
        ax[k].set_title(f"step {st}", fontsize=9.5)
        ax[k].set_xlabel("level")
        ax[k].set_ylabel("held-out R2 of V[l, 0]")
        ax[k].legend(fontsize=7, framealpha=0.85)
    fig.suptitle(f"expressivity E1: held-out fit per arm.  {tag}; markers o / s / ^ = "
                 f"{' / '.join(names)}.", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=130)
    print("wrote", path)


def fig_axis(d, name, tag, an, path, st=64000, nbin=8, a=0):
    plt = _mpl()
    p = xp(d, tag, st)
    if not os.path.exists(p):
        return
    C = cell_of(p, an)
    Rw, mm = matched_sets(p, an)
    fig, ax = plt.subplots(2, 4, figsize=(21.5, 9))
    for k, l in enumerate(LS):
        base = C.g(f"ok_l{l}_a{a}").astype(bool)
        b = _qbin(C.H, nbin)
        xs = [float(C.H[(b == i) & base].mean()) for i in range(nbin)]
        oe = C.g(f"oe_l{l}_a{a}").astype(np.float64)
        ax[0, k].plot(xs, [float(oe[(b == i) & base].mean()) for i in range(nbin)],
                      color="k", lw=2.2, marker="o", label="OUTCOME")
        for arm in ARMS:
            if not has(C, arm, l, a):
                continue
            vp = C.Vpre(arm, l, a)
            ax[0, k].plot(xs, [float(vp[(b == i) & base].mean()) for i in range(nbin)],
                          color=COLS[arm], marker=".", label=f"V_pre {LABEL[arm]}")
        ax[0, k].set_title(f"level {l}: the norm against the outcome", fontsize=9.5)
        ax[0, k].set_xlabel("H_pre bin (nats)")
        ax[0, k].legend(fontsize=6, framealpha=0.85)
        if Rw is not None and l in mm:
            i = mm[l][0]
            bb = _qbin(C.H[i], 5)
            xs2 = [float(C.H[i][bb == q].mean()) for q in range(5)]
            for arm in ARMS:
                if not has(C, arm, l, a):
                    continue
                Rv = C.R(arm, l, a)[i]
                ax[1, k].plot(xs2, [float(Rv[bb == q].mean()) for q in range(5)],
                              color=COLS[arm], marker="o", label=LABEL[arm])
            ax[1, k].axhline(0, c="k", lw=0.6)
            ax[1, k].set_title(f"level {l}: R at matched surprisal (n = {len(i)})",
                               fontsize=9.5)
            ax[1, k].set_xlabel("H_pre bin (nats)")
            ax[1, k].legend(fontsize=6, framealpha=0.85)
    fig.suptitle(f"expressivity E2-E3: {name}, {tag}, anchor {an}, step {st}.  Top: each "
                 f"arm's pre-event level against the realised outcome along the entropy axis.  "
                 f"Bottom: each arm's response on junction's matched rows.", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=120)
    print("wrote", path)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

PRE = """# precision E — the expressivity test

Facts only; interpretation is discussed with Jasper and lives nowhere in this folder.
Generated by [`express_read.py`](../express_read.py) from [`express.py`](../express.py)'s
refits.  Nothing banked is rewritten.  Three trajectory seeds side by side, **never
averaged**.

**The question.**  `tables_mediation_20260922.md` A6: the pre-event entropy ranks realised
damage at ell = 1 on surprisal-matched rows (0.62 at 64k) and nothing the banked reader has
or the event does accounts for it.  The candidate reading is about the reader's FORM: the
belief's entropy is a nonlinear function of the state and the critic is a linear projection
of it.  The test refits the critic with the feature map augmented and reads it on the
identical rows.

**Arms.**  Every arm is fitted on the SAME trunk, the SAME `full` diet rows (and, as the
exposure control, the clean-only critic's rows), the SAME split and the SAME lambda ladder
with the SAME held-out per-column lambda selection as `norm/task.py`.  Appended columns are
affinely standardised to the state's own per-dimension RMS on the training rows (exactly, by
transforming the Gram), so the ridge penalty treats each like an average state direction.

| arm | reader |
|---|---|
| `ridge (banked)` | the state at `post_block7`: `norm/task.py`'s critic, refitted and gated to reproduce the bank |
| `+H` | state + `H(q_t)`, the entropy of the forecast made at the state's own position (`V_pre` sees `H_pre`, the post-event level sees `H_next`) -- **primary** |
| `+log q` | state + `log q_t` (16 dims): the belief's linear content without its entropy -- **the control for `+H`** |
| `+norm (plac.)` | state + `\\|\\|s_t\\|\\|_2` -- a placebo nonlinear scalar |
| `+tanh (plac.)` | state + `tanh(u . s_t / sd)` for one fixed random direction `u` -- a second placebo |
| `+precision` | state + `H(q_t)` + `H(q_(t-1))` + `s_t` + `H(q_(t-1)) s_t`, with `s_t` the surprisal of the token at `t`: the literature's own surprise-times-prior-uncertainty form, plus `+H`'s column -- secondary |
| `MLP` | a 128-unit GELU MLP on the state (`coeruleus.readout.train_head`, `striatum/task.py`'s 2000 steps), levels 1-4 at `a = 0` only, no lambda selection -- secondary |

**Step 0** is the random-init floor and is degenerate for this axis (`H_pre` sd 0.017 against
0.67 at 64k, `tables_20260922.md` 1), so every slope is printed times the cell's own sd of
`H_pre`.
"""


PRE_LN = """# precision E-ln — is "the belief's shape" just normalisation?

Facts only; interpretation is discussed with Jasper and lives nowhere in this folder.
Generated by [`express_read.py`](../express_read.py) `--sfx _express_ln` from
[`express.py`](../express.py)'s follow-up-4 refits.  An addendum to
[`tables_express_20260922.md`](tables_express_20260922.md), which it does not rewrite.
Three trajectory seeds side by side, **never averaged**.

**The question.**  `tables_express` E4: `+log q` and the MLP absorb what the pre-event entropy
knew about damage, `+H` does not.  The logits are the unembedding applied to the trunk's
FINAL LAYER NORM of the residual stream (`post_block7` feeds `ln_f` directly), so the one
thing a ridge on the raw stream cannot do and `+log q` can is divide by the state's own
scale; the `+norm` placebo only ruled out the scalar, not the normalised directions.

**Arms** (same trunk, `full` rows, split, lambda ladder and selection as `norm/task.py`; arm
0 refitted and gated against the bank exactly as before):

| arm | reader |
|---|---|
| `ridge (banked)` | the raw state at `post_block7` -- the gate and the first reference |
| `+log q` | state + `log q_t` -- the second reference, refitted in the same cell (its columns reproduce the committed `_express` cells bit for bit) |
| `ln (in place)` | `ln_f(s)`, the model's own final layer norm with its weight and bias, **in place of** the raw state: exactly what the unembedding reads |
| `+ln` | `[s, ln_f(s)]`, the ln block scaled as a block to the raw state's per-dimension RMS (its own geometry kept) by an exact diagonal transform of the Gram |
| `std (in place)` | `(s - mean) / sqrt(var + eps)` per row with the same `eps` and no learned affine: normalisation separated from the model's own |

`ln_f` is a per-dimension affine map of `std` (weight times standardised coordinate plus
bias), and a ridge with a bias absorbs the bias and differs from `std` only through how its
single penalty weights the rescaled dimensions; the two in-place arms are expected to be
close and are both reported.
"""


def main():
    global SFX, ARMS, CLEAN
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--tags", default="swap65k,a1")
    ap.add_argument("--anchors", default="tv,fd")
    ap.add_argument("--out", default=".")
    ap.add_argument("--figs", default="")
    ap.add_argument("--date", default="20260922")
    ap.add_argument("--sfx", default="_express")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--clean-arms", default=",".join(CLEAN))
    ap.add_argument("--outname", default="")
    ap.add_argument("--figprefix", default="prec_express")
    ap.add_argument("--preamble", default="express", choices=["express", "ln"])
    args = ap.parse_args()
    SFX = args.sfx
    ARMS = [a for a in args.arms.split(",") if a]
    CLEAN = [a for a in args.clean_arms.split(",") if a]
    dirs = args.dirs
    names = [os.path.basename(os.path.normpath(x)) for x in dirs]
    tags = [t for t in args.tags.split(",") if t]
    ancs = [t for t in args.anchors.split(",") if t]

    P = [PRE if args.preamble == "express" else PRE_LN,
         "\n---\n\n## E0. Gates\n", sec_gates(dirs, names, tags),
         sec_matched_identity(dirs, names, tags, ancs)]
    for tag in tags:
        P.append(f"\n---\n\n# {tag}\n")
        P.append("## E1. The held-out fit\n")
        P.append(sec_fit(dirs, names, tag, ARMS, 0))
        P.append(sec_fit(dirs, names, tag, ARMS, 1))
        P.append(sec_fit(dirs, names, tag, CLEAN, 0))
        for an in ancs:
            P.append(f"\n## {tag}, anchor {an}\n")
            P.append("### E2. The norm and the outcome along the axis\n")
            P.append(sec_level_outcome(dirs, names, tag, an, ARMS))
            P.append(sec_delta(dirs, names, tag, an, ARMS))
            P.append("### E3. The response at matched surprisal\n")
            P.append(sec_response(dirs, names, tag, an, ARMS, withV=False))
            P.append(sec_response(dirs, names, tag, an, ARMS, withV=True))
            P.append(sec_interaction(dirs, names, tag, an, ARMS))
            P.append(sec_damage_terciles(dirs, names, tag, an, ARMS))
            P.append("### E4. Does the reader now contain what the axis knew?\n")
            P.append(sec_contain(dirs, names, tag, an, ARMS))
            P.append("### E4b. The same on the clean-only critic (the exposure control)\n")
            P.append(sec_contain(dirs, names, tag, an, CLEAN))
        P.append(f"\n## {tag}, the twins\n")
        P.append(sec_twins(dirs, names, tag, ARMS))
    if "a1" in tags:
        P.append("\n---\n\n## E5. Legality at matched consequence (the arc's law)\n")
        P.append(sec_legality(dirs, names, ARMS))
    os.makedirs(args.out, exist_ok=True)
    p = os.path.join(args.out, args.outname or f"tables_express_{args.date}.md")
    with open(p, "w") as f:
        f.write("\n".join(P).rstrip() + "\n")
    print(f"wrote {p}  ({os.path.getsize(p) / 1024:.0f} KB)")
    if args.figs:
        os.makedirs(args.figs, exist_ok=True)
        for tag in tags:
            fig_fit(dirs, names, tag, os.path.join(args.figs, f"{args.figprefix}_fit_{tag}.png"))
            for an in ancs:
                for st in (64000, 8000):
                    fig_axis(dirs[0], names[0], tag, an,
                             os.path.join(args.figs, f"{args.figprefix}_axis_{tag}_{an}_{st}.png"),
                             st=st)


if __name__ == "__main__":
    main()
