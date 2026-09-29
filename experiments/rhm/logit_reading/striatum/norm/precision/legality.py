"""Does the arc's law survive a reader that sees the belief?  Facts only.

The law (`junction/`): an outcome-trained reader is indifferent to legality at matched
consequence -- swap against rare among CONSEQUENTIAL episodes reads at chance across eleven
diets, while an oracle probe on the same states finds 0.05-0.11 of headroom.
`tables_express_20260922.md` E5 moved the UNMATCHED contrast off chance by 0.03-0.06 under
`+log q`, `+precision` and the MLP, but the unmatched contrast does not control surprisal and
`a1`'s few hundred consequential rare edits made the matched one unreadable.

This reads the `legal65k` venue (`stimuli.build_stimuli --frac 0.5,0.5,0.0 --n 65536
--seed 2028`, half swap and half legal-but-rare edits, no `none` windows) through
`express.py`'s refits (`bank_tag = swap65k`: there is no banked `norm/` cell on this venue,
so the actor's clean accuracy is the gate) at the onset anchor `fd`, the only anchor where a
rare edit exists.

**The contrast is matched two ways, because it cannot be matched on both `j` and
consequence depth at once.**  `junction.diets.cons_depth` is `j` for a swap and `j - 1` for a
rare edit, a deterministic function of `(etype, j)`, so a swap and a rare edit never share
both.  `jt` pins the edit's width `j` (junction's own recipe) and leaves consequence depth one
level deeper on the swap side; `ct` pins consequence depth and leaves the rare edit one leaf-
level wider.  Both: exact strata on `(pinned variable, t0 // 4)`, 1:1 nearest neighbour on
the model's surprisal at the anchor at caliper 0.30, sign-balanced inside `|ds|` bands
(`striatum.analyze.match_sign_balanced`, seed 0), and the same at caliper 0.10 as the guard-
slip check.  Every guard is printed on the matched rows: surprisal, position, `j`,
consequence depth, and an input-embedding probe (ridge on `post_embed` at the event, i.e.
token + position only, fitted exactly like the ceiling).

Per reader: `AUC(-R, swap)` and `AUC(-V, swap)` (junction's orientation), unmatched and on
both matched sets; the oracle legality probe (linear and MLP on `post_block7`, junction's
recipe) as the ceiling; the reverse guard (the reader's own realised-damage AUC on the same
matched rows, so a legality reading is not bought by a damage reading); and whether legality
predicts the OUTCOME on this venue beyond consequence and surprisal (OLS of the actor's hit,
and of realised damage, on `swap` plus the controls, on the matched rows and on the whole
consequential base).

  python -m rhm.logit_reading.striatum.norm.precision.legality \\
      <dir>/traj_a1_s42 <dir>/traj_a1_s43 <dir>/traj_a1_s44 \\
      --out rhm/logit_reading/striatum/norm/precision/results \\
      --figs rhm/logit_reading/striatum/norm/precision/figs --date 20260922
"""

import argparse
import json
import os

import numpy as np

from rhm.logit_reading.striatum.analyze import _auc, fmt, match_sign_balanced, tbl
from rhm.logit_reading.striatum.norm.precision.reduce import _ols_multi

TAG = "legal65k"
AN = "fd"
LS = [1, 2, 3, 4]
READERS = ["full", "fullQ", "fullP", "fullL", "fullM", "clean", "cleanQ", "cleanP", "cleanL"]
LABEL = {"full": "ridge", "fullQ": "+log q", "fullP": "+precision", "fullL": "ln (in place)",
         "fullM": "MLP", "clean": "clean ridge", "cleanQ": "clean +log q",
         "cleanP": "clean +precision", "cleanL": "clean ln"}
MAIN = ["full", "fullQ", "fullP", "fullL", "fullM"]


def xp(d, st):
    return os.path.join(d, f"step{st:06d}_norm_{TAG}_express.npz")


def xj(d, st):
    return os.path.join(d, f"step{st:06d}_norm_{TAG}_express.json")


_C = {}


class Cell:
    """One (seed, step) cell's onset rows, the legality base per level and the two matched
    sets per level, memoised."""

    def __init__(self, path):
        Z = np.load(path)
        self.Z = Z
        g = lambda k: np.asarray(Z[f"{AN}__{k}"]) if f"{AN}__{k}" in Z.files else None  # noqa
        self.g = g
        self.et, self.jj, self.t0 = g("etype"), g("j"), g("t0")
        self.cd, self.s = g("c_depth"), g("nll_e").astype(np.float64)
        self.n = len(self.et)
        self._m = {}

    def base(self, l, a=0):
        ok = self.g(f"ok_l{l}_a{a}").astype(bool)
        cons = self.g(f"cons_l{l}_a{a}").astype(bool)
        return ok & (self.et < 2) & cons

    def matched(self, l, strata, caliper=0.3, a=0):
        k = (l, strata, caliper, a)
        if k in self._m:
            return self._m[k]
        b = np.where(self.base(l, a))[0]
        lab = self.et == 0
        out = None
        if len(b) >= 50 and 0 < lab[b].mean() < 1:
            piv = self.jj if strata == "jt" else self.cd
            keys = piv[b] * 1000 + (self.t0[b] // 4)
            sel = match_sign_balanced(keys, lab[b], self.s[b], caliper,
                                      np.random.default_rng(0))
            if len(sel) >= 40:
                out = b[sel]
        self._m[k] = out
        return out

    def score(self, arm, kind, l, a=0):
        k = f"{AN}__{kind}_{arm}_l{l}_a{a}"
        return np.asarray(self.Z[k], np.float64) if k in self.Z.files else None


def cell(d, st):
    p = xp(d, st)
    if not os.path.exists(p):
        return None
    if p not in _C:
        _C[p] = Cell(p)
    return _C[p]


def ps(dirs, fn):
    return " / ".join(fn(d) for d in dirs)


# ---------------------------------------------------------------------------
# sections
# ---------------------------------------------------------------------------

def sec_venue(dirs, names, steps):
    rows = []
    for d, nm in zip(dirs, names):
        for st in steps:
            C = cell(d, st)
            pj = xj(d, st)
            if C is None or not os.path.exists(pj):
                rows.append([nm, st] + ["MISSING"] * 6)
                continue
            r = json.load(open(pj))
            g = r["gates"]
            pr = g.get("probes", {}).get(AN, {})
            rows.append([nm, st, f"{max(g['actor'].values()):.1e}",
                         "PASS" if not r["gate_fails"] else "FAIL " + ",".join(r["gate_fails"]),
                         C.n, int((C.et == 0).sum()), int((C.et == 1).sum()),
                         fmt(pr.get("ed_val_r2"), 3) + " / " + fmt(pr.get("emb_val_r2"), 3)])
    return "\n".join([
        f"**The venue and the gate.**  `{TAG}`: 65,536 windows, half swap and half rare, no "
        f"`none`; the onset anchor `{AN}`'s held-out (`split == 2`) rows.  There is no banked "
        f"`norm/` cell on this venue, so the gate is the frozen actor's clean accuracy against "
        f"the banked `swap65k` json of the same checkpoint (the actor is trained on clean "
        f"windows only).  The last column is the held-out `R2` of the legality probe on the "
        f"post-event state and on the input embedding, on the validation split.", "",
        tbl(rows, ["seed", "step", "actor gate", "verdict", "fd test rows", "swap", "rare",
                   "probe val R2 (state / embedding)"]), ""])


def sec_counts(dirs, names, steps):
    rows = []
    for st in steps:
        for l in LS:
            cells = [st, l]
            for what in ("base", "swap", "rare", "jt", "ct", "jt@0.10", "ct@0.10"):
                def fn(d, what=what):
                    C = cell(d, st)
                    if C is None:
                        return "--"
                    b = C.base(l)
                    if what == "base":
                        return str(int(b.sum()))
                    if what == "swap":
                        return str(int((b & (C.et == 0)).sum()))
                    if what == "rare":
                        return str(int((b & (C.et == 1)).sum()))
                    strata, cal = (what.split("@") + ["0.30"])[:2]
                    m = C.matched(l, strata, float(cal))
                    return "--" if m is None else str(len(m))
                cells.append(ps(dirs, fn))
            rows.append(cells)
    return "\n".join([
        f"**Row counts — {' / '.join(names)} side by side.**  `base` is the consequential "
        f"edited rows at level `l` (the edit changes the level-`l` label at the query); `jt` "
        f"/ `ct` are the matched sets at caliper 0.30, `@0.10` the tight-caliper replicates.",
        "", tbl(rows, ["step", "l", "consequential", "swap", "rare", "jt", "ct",
                       "jt @0.10", "ct @0.10"]), ""])


def sec_guards(dirs, names, steps, strata, caliper=0.3):
    rows = []
    for st in steps:
        for l in LS:
            cells = [st, l]
            for gname in ("surprisal", "t0", "j", "c_depth", "embedding probe"):
                def fn(d, gname=gname):
                    C = cell(d, st)
                    if C is None:
                        return "--"
                    m = C.matched(l, strata, caliper)
                    if m is None:
                        return "--"
                    y = C.et[m] == 0
                    x = {"surprisal": C.s, "t0": C.t0.astype(float), "j": C.jj.astype(float),
                         "c_depth": C.cd.astype(float),
                         "embedding probe": C.g("probe_swap_emb")}[gname]
                    return "--" if x is None else fmt(_auc(np.asarray(x, float)[m], y))
                cells.append(ps(dirs, fn))
            rows.append(cells)
    pinned = "j" if strata == "jt" else "c_depth"
    return "\n".join([
        f"**Guards on the `{strata}` matched rows (caliper {caliper:.2f}) — "
        f"{' / '.join(names)}.**  `AUC(guard, swap)`.  Surprisal is pinned by the pair "
        f"matching and `{pinned}` by the strata, so both must read 0.500; the other structural "
        f"variable is NOT pinnable jointly (consequence depth is `j` for a swap and `j - 1` for "
        f"a rare edit) and is printed so its imbalance is on the record; the embedding probe "
        f"is the token-and-position guard.", "",
        tbl(rows, ["step", "l", "surprisal", "t0", "j", "c_depth", "embedding probe"]), ""])


def sec_legality(dirs, names, steps, readers, what="R", strata="jt", caliper=0.3):
    rows = []
    for st in steps:
        for l in LS:
            cells = [st, l]
            for arm in readers:
                def fn(d, arm=arm):
                    C = cell(d, st)
                    if C is None:
                        return "--"
                    sc = C.score(arm, what, l)
                    if sc is None:
                        return "--"
                    if strata == "unmatched":
                        m = np.where(C.base(l))[0]
                    else:
                        m = C.matched(l, strata, caliper)
                        if m is None:
                            return "--"
                    return fmt(_auc(-sc[m], C.et[m] == 0))
                cells.append(ps(dirs, fn))
            for pnm in ("probe_swap_ed", "probe_swap_mlp"):
                def fn(d, pnm=pnm):
                    C = cell(d, st)
                    if C is None or C.g(pnm) is None:
                        return "--"
                    m = (np.where(C.base(l))[0] if strata == "unmatched"
                         else C.matched(l, strata, caliper))
                    if m is None:
                        return "--"
                    return fmt(_auc(np.asarray(C.g(pnm), float)[m], C.et[m] == 0))
                cells.append(ps(dirs, fn))
            rows.append(cells)
    where = ("the whole consequential base, unmatched (E5's contrast)" if strata == "unmatched"
             else f"the `{strata}` matched rows at caliper {caliper:.2f}")
    return "\n".join([
        f"**Legality at matched consequence, `AUC(-{what}, swap)` — {where}; "
        f"{' / '.join(names)}.**  0.500 is a reader indifferent to legality.  The last two "
        f"columns are the oracle legality probe on the post-event state (linear, MLP), the "
        f"ceiling a reader of this state could reach.", "",
        tbl(rows, ["step", "l"] + [LABEL[a] for a in readers]
            + ["ceiling (linear)", "ceiling (MLP)"]), ""])


def sec_reverse(dirs, names, steps, readers, strata="jt", caliper=0.3):
    rows = []
    for st in steps:
        for l in LS:
            cells = [st, l]
            for arm in readers:
                def fn(d, arm=arm):
                    C = cell(d, st)
                    if C is None:
                        return "--"
                    m = C.matched(l, strata, caliper)
                    R = C.score(arm, "R", l)
                    if m is None or R is None:
                        return "--"
                    oe = C.g(f"oe_l{l}_a0").astype(bool)[m]
                    oo = C.g(f"oo_l{l}_a0").astype(bool)[m]
                    base = oo
                    if base.sum() < 20 or not 0 < (~oe[base]).mean() < 1:
                        return "--"
                    return fmt(_auc(-R[m][base], ~oe[base]))
                cells.append(ps(dirs, fn))
            rows.append(cells)
    return "\n".join([
        f"**The reverse guard: each reader's realised-damage AUC on the same `{strata}` "
        f"matched rows — {' / '.join(names)}.**  `AUC(-R, flip)` on the matched rows where the "
        f"actor was right on the unedited stream, so a legality reading can be set beside the "
        f"reader's damage reading on identical rows.", "",
        tbl(rows, ["step", "l"] + [LABEL[a] for a in readers]), ""])


def sec_outcome(dirs, names, steps, strata="jt", caliper=0.3):
    """Does legality predict the outcome on this venue beyond consequence and surprisal?"""
    rows = []
    for st in steps:
        for l in LS:
            for where in (strata, "base"):
                for yname in ("hit", "damage"):
                    cells = [st, l, where if where == "base" else f"{strata} matched", yname]
                    def fn(d):
                        C = cell(d, st)
                        if C is None:
                            return "--"
                        m = (np.where(C.base(l))[0] if where == "base"
                             else C.matched(l, strata, caliper))
                        if m is None:
                            return "--"
                        oe = C.g(f"oe_l{l}_a0").astype(float)
                        oo = C.g(f"oo_l{l}_a0").astype(bool)
                        if yname == "hit":
                            y, mm = oe[m], m
                        else:
                            keep = oo[m]
                            mm = m[keep]
                            y = 1.0 - oe[mm]
                        if len(mm) < 40:
                            return "--"
                        # consequence depth is j - 1 + swap on EVERY edited row, so swap, j
                        # and c_depth are exactly collinear and the three cannot share a
                        # regression; control the variable this matched set pins.
                        piv = C.jj if strata == "jt" else C.cd
                        X = np.stack([(C.et[mm] == 0).astype(float), piv[mm].astype(float),
                                      C.t0[mm].astype(float), C.s[mm]], 1)
                        cf, se, _ = _ols_multi(y, X)
                        return f"{cf[0]:+.4f} ({cf[0] / se[0]:.1f})" if se[0] > 0 else "--"
                    cells.append(ps(dirs, fn))
                    rows.append(cells)
    return "\n".join([
        f"**Does legality predict the outcome on this venue beyond consequence and surprisal? "
        f"— controls pinned as in `{strata}`; {' / '.join(names)}.**  OLS coefficient of "
        f"`swap` (with its `t`) in `y ~ swap + {'j' if strata == 'jt' else 'c_depth'} + t0 + "
        f"surprisal`.  Consequence depth is `j - 1 + swap` on every edited row, so `swap`, `j` "
        f"and `c_depth` are exactly collinear and cannot all be controls: under `jt` the swap "
        f"coefficient carries the swap's one-level-deeper consequence, under `ct` the rare "
        f"edit's one-level-wider span.  `y` is the frozen actor's hit at "
        f"the query (`hit`) or realised damage on the rows where it was right on the unedited "
        f"stream (`damage`); on the matched rows and on the whole consequential base.", "",
        tbl(rows, ["step", "l", "rows", "y", "coef of swap (t)"]), ""])


def fig_legality(dirs, names, steps, path, strata="jt"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, len(steps), figsize=(7.5 * len(steps), 4.8), squeeze=False)
    cols = {"full": "#444444", "fullQ": "#1f77b4", "fullP": "#9467bd", "fullL": "#ff7f0e",
            "fullM": "#2ca02c"}
    for k, st in enumerate(steps):
        a_ = ax[0, k]
        for arm in MAIN:
            for j_, d in enumerate(dirs):
                C = cell(d, st)
                if C is None:
                    continue
                ys = []
                for l in LS:
                    m = C.matched(l, strata)
                    sc = C.score(arm, "R", l)
                    ys.append(np.nan if (m is None or sc is None)
                              else _auc(-sc[m], C.et[m] == 0))
                a_.plot(np.array(LS) + 0.05 * (j_ - 1), ys, marker="os^"[j_],
                        color=cols[arm], label=LABEL[arm] if j_ == 0 else None)
        for j_, d in enumerate(dirs):
            C = cell(d, st)
            if C is None or C.g("probe_swap_ed") is None:
                continue
            ys = []
            for l in LS:
                m = C.matched(l, strata)
                ys.append(np.nan if m is None else
                          _auc(np.asarray(C.g("probe_swap_ed"), float)[m], C.et[m] == 0))
            a_.plot(LS, ys, ls=":", color="k", marker="x",
                    label="ceiling (linear probe)" if j_ == 0 else None)
        a_.axhline(0.5, c="k", lw=0.7)
        a_.set_title(f"step {st}: AUC(-R, swap) on the `{strata}` matched rows", fontsize=9.5)
        a_.set_xlabel("level")
        a_.set_ylabel("legality AUC")
        a_.legend(fontsize=7, framealpha=0.85)
    fig.suptitle(f"legality at matched consequence and surprisal, venue {TAG}, anchor {AN}; "
                 f"markers o / s / ^ = {' / '.join(names)}.", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=130)
    print("wrote", path)


PRE = """# precision legality — does the arc's law survive a reader that sees the belief?

Facts only; interpretation is discussed with Jasper and lives nowhere in this folder.
Generated by [`legality.py`](../legality.py) from [`express.py`](../express.py)'s refits on
the `legal65k` venue.  Three trajectory seeds side by side, **never averaged**.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", default=".")
    ap.add_argument("--figs", default="")
    ap.add_argument("--date", default="20260922")
    ap.add_argument("--steps", default="0,8000,64000")
    args = ap.parse_args()
    dirs = args.dirs
    names = [os.path.basename(os.path.normpath(x)) for x in dirs]
    steps = [int(x) for x in args.steps.split(",")
             if any(os.path.exists(xp(d, int(x))) for d in dirs)]
    P = [PRE, (__doc__.split("\n\n", 1)[1]).replace("\\\\", "\\"),
         "\n---\n\n## L0. The venue, the gate and the counts\n",
         sec_venue(dirs, names, steps), sec_counts(dirs, names, steps),
         "\n---\n\n## L1. Guards on the matched rows\n"]
    for strata in ("jt", "ct"):
        P.append(sec_guards(dirs, names, steps, strata))
    P.append(sec_guards(dirs, names, steps, "jt", 0.1))
    P.append("\n---\n\n## L2. Legality at matched consequence and surprisal\n")
    for strata in ("jt", "ct"):
        P.append(sec_legality(dirs, names, steps, READERS, "R", strata))
        P.append(sec_legality(dirs, names, steps, READERS, "V", strata))
    P.append("### The tight-caliper replicate (0.10)\n")
    P.append(sec_legality(dirs, names, steps, MAIN, "R", "jt", 0.1))
    P.append(sec_legality(dirs, names, steps, MAIN, "R", "ct", 0.1))
    P.append("### The unmatched contrast, for continuity with E5\n")
    P.append(sec_legality(dirs, names, steps, READERS, "R", "unmatched"))
    P.append("\n---\n\n## L3. The reverse guard: the readers' damage reading on the same rows\n")
    P.append(sec_reverse(dirs, names, steps, MAIN, "jt"))
    P.append(sec_reverse(dirs, names, steps, MAIN, "ct"))
    P.append("\n---\n\n## L4. Does legality predict the outcome here?\n")
    P.append(sec_outcome(dirs, names, steps, "jt"))
    P.append(sec_outcome(dirs, names, steps, "ct"))
    os.makedirs(args.out, exist_ok=True)
    p = os.path.join(args.out, f"tables_legality_{args.date}.md")
    with open(p, "w") as f:
        f.write("\n".join(P).rstrip() + "\n")
    print(f"wrote {p}  ({os.path.getsize(p) / 1024:.0f} KB)")
    if args.figs:
        os.makedirs(args.figs, exist_ok=True)
        for strata in ("jt", "ct"):
            fig_legality(dirs, names, steps,
                         os.path.join(args.figs, f"prec_legality_{strata}.png"), strata)


if __name__ == "__main__":
    main()
