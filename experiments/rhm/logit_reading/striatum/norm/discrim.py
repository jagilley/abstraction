"""Held-out AUC of the outcome-trained critic's level and revision, per checkpoint, per diet,
on the norm round's fixed rows -- the trained trunk against the random-init trunk (step 0) of
the same architecture, on IDENTICAL rows.  Facts only; no interpretation lives here.

`analyze.py` already banks the critic's held-out *fit* `R2` per diet (tables `0`) and the
matched-row `AUC(-R, flip)` / `AUC(-V, flip)` columns of the response tables, both read off the
POST-event state.  This file adds what those do not carry: the **pre-event level** `V_pre`'s own
ranking of the realised outcome, the same for the realised-damage and consequence labels, and the
step-0-against-64k difference on the same rows.  Nothing is refit and nothing touches a GPU: the
per-row outcome labels (`oe_l<l>_a<a>`, `oo_...`, `cons_...`, `ok_...`) are already on file in the
npz, which holds the `split == 2` TEST windows only, so every row here is held out from the
critic's Gram and from its lambda selection.

  D=/v16_s2_L6_m4_distinct/logit_reading
  for s in 42 43 44; do for st in 000000 008000 064000; do for tg in a1 swap65k; do
    modal volume get rhm-scaling-data $D/traj_a1_s$s/step${st}_norm_${tg}.npz <dir>/traj_a1_s$s/
  done; done; done
  python -m rhm.logit_reading.striatum.norm.discrim \
      <dir>/traj_a1_s42 <dir>/traj_a1_s43 <dir>/traj_a1_s44 \
      --out rhm/logit_reading/striatum/norm/results --outfile tables_discrim_20260919.md

The rows are identical across the three checkpoints of one trajectory (same windows, same anchor
positions, same split seed; asserted per cell).  The `cons` label is a property of the grammar and
is identical across checkpoints too; `oe` / `oo` are the frozen actor's own hits, so the OUTCOME and
DAMAGE labels move with the checkpoint even though the rows do not.  Both facts are printed.
"""

import argparse
import os

import numpy as np

from rhm.logit_reading.striatum.analyze import _auc, fmt, tbl
from rhm.logit_reading.striatum.junction.analyze import Rows
from rhm.logit_reading.striatum.norm.analyze import order_diets

STEPS = [0, 8000, 64000]
LS = [1, 2, 3, 4]
ANCHORS = ["fd", "tv"]
# score orientation per label: `V`/`V_pre`/`R` are all in "higher = a better world" units
# (`V[l, d]` predicts the actor being RIGHT), so the outcome label is ranked by +score and the two
# bad-news labels (realised damage, consequence) by -score, which is the banked columns' sign.
LABELS = [("outcome", +1), ("damage", -1), ("cons", -1)]
READOUTS = ["V_pre", "V", "R"]


# ---------------------------------------------------------------------------
# rows, labels, scores
# ---------------------------------------------------------------------------

def npz_of(d, tag, st):
    return os.path.join(d, f"step{st:06d}_norm_{tag}.npz")


def load_cells(d, tag):
    """The three checkpoints of one (trajectory, venue), or None if any is missing."""
    out = {}
    for st in STEPS:
        p = npz_of(d, tag, st)
        if not os.path.exists(p):
            return None
        out[st] = np.load(p)
    return out


def rows_identical(Zs, an):
    """The fixed rows must be the same windows at the same positions across checkpoints."""
    ws = [np.asarray(Zs[st][f"{an}__w"]) for st in STEPS]
    ts = [np.asarray(Zs[st][f"{an}__t0"]) for st in STEPS]
    return (all(np.array_equal(ws[0], w) for w in ws)
            and all(np.array_equal(ts[0], t) for t in ts))


def label_of(Z, an, kind, l, a):
    """(y, base) for one label on one checkpoint's fixed rows.

    `outcome`  -- `oe_l<l>_a<a>`, the frozen actor being RIGHT at level `l` at the query
                  position, on the EDITED stream: the critic's own teacher.  Base = the rows
                  whose query position exists (`ok`), which is a property of the position and so
                  is identical across checkpoints.
    `damage`   -- junction's `flip`: right on the unedited stream and wrong on the edited one.
                  Base = `ok & etype < 2 & oo`, which contains the actor's unedited hit and so
                  MOVES with the checkpoint.
    `cons`     -- the edit changes the grammar's own level-`l` label at the query position.  Base
                  = `ok & etype < 2`.  Both label and base are checkpoint-independent.
    """
    g = lambda k: np.asarray(Z[f"{an}__{k}"])                                   # noqa: E731
    ok = g(f"ok_l{l}_a{a}").astype(bool)
    et = g("etype")
    if kind == "outcome":
        return g(f"oe_l{l}_a{a}").astype(bool), ok
    if kind == "damage":
        oe, oo = g(f"oe_l{l}_a{a}").astype(bool), g(f"oo_l{l}_a{a}").astype(bool)
        return oo & ~oe, ok & (et < 2) & oo
    if kind == "cons":
        return g(f"cons_l{l}_a{a}").astype(bool), ok & (et < 2)
    raise ValueError(kind)


def score_of(Z, an, readout, diet, l, a):
    k = {"V_pre": f"Vpre_{diet}_l{l}_a{a}", "V": f"V_{diet}_l{l}_a{a}",
         "R": f"R_{diet}_l{l}_a{a}"}[readout]
    kk = f"{an}__{k}"
    return np.asarray(Z[kk], np.float64) if kk in Z.files else None


def diets_of(Zs, an):
    """The diets present in every checkpoint of this cell, in the reduction's canonical order."""
    per = []
    for st in STEPS:
        Z = Zs[st]
        pre = f"{an}__"
        per.append({k[len(pre) + 2:-6] for k in Z.files
                    if k.startswith(pre + "R_") and k.endswith("_l1_a0")})
    return order_diets(sorted(set.intersection(*per)))


def perm_sd(n1, n0):
    """Exact sd of the AUC under a shuffled label (no ties): Var(U/(n1 n0)) = (n+1)/(12 n1 n0).

    The floor's mean is 0.500 and its spread depends only on (n1, n0), never on the score, so one
    number covers every diet and every readout of a cell.  `--check-floor` verifies it by
    permutation.
    """
    if n1 < 1 or n0 < 1:
        return float("nan")
    return float(np.sqrt((n1 + n0 + 1.0) / (12.0 * n1 * n0)))


def empirical_floor(score, y, n_perm, seed=0):
    """AUC against a shuffled label, `n_perm` draws: (mean, sd)."""
    from scipy.stats import rankdata
    y = np.asarray(y).astype(bool)
    n, n1 = len(y), int(y.sum())
    n0 = n - n1
    if n1 < 1 or n0 < 1:
        return float("nan"), float("nan")
    r = rankdata(np.asarray(score, np.float64))
    rng = np.random.default_rng(seed)
    a = np.empty(n_perm)
    for i in range(n_perm):
        s = r[rng.permutation(n)[:n1]].sum()
        a[i] = (s - n1 * (n1 + 1) / 2.0) / (n1 * n0)
    return float(a.mean()), float(a.std(ddof=1))


# ---------------------------------------------------------------------------
# sections
# ---------------------------------------------------------------------------

def sec_rows(Zs, an, tag, a=0, check_floor=0):
    """Per level and checkpoint: the row count, the label's base rate, the permutation floor."""
    et = np.asarray(Zs[STEPS[0]][f"{an}__etype"])
    hist = np.bincount(et, minlength=3)
    rows = []
    for l in LS:
        for kind, _ in LABELS:
            cells = [l, kind]
            for st in STEPS:
                y, base = label_of(Zs[st], an, kind, l, a)
                n, n1 = int(base.sum()), int((y & base).sum())
                cells += [n, fmt(n1 / max(n, 1), 4), fmt(perm_sd(n1, n - n1), 4)]
            rows.append(cells)
    hdr = ["l", "label"]
    for st in STEPS:
        hdr += [f"n @{st}", f"base @{st}", f"floor sd @{st}"]
    out = [f"**Rows and labels — {tag}, anchor {an}, a = {a}.**  "
           f"{len(et)} fixed (held-out, `split == 2`) rows at this anchor; etype histogram "
           f"swap / rare / none = {hist[0]} / {hist[1]} / {hist[2]}.  The rows are IDENTICAL "
           f"across the three checkpoints (asserted).  `floor sd` is the exact sd of the AUC "
           f"under a shuffled label, `sqrt((n+1)/(12 n1 n0))`; the floor's mean is 0.500 and it "
           f"does not depend on the score, so it covers every diet and readout of the cell.", "",
           tbl(rows, hdr), ""]
    if check_floor:
        cr = []
        for l in (1, 4):
            for kind, sgn in LABELS:
                st = 64000
                y, base = label_of(Zs[st], an, kind, l, a)
                sc = score_of(Zs[st], an, "V_pre", "full", l, a)
                if sc is None or base.sum() < 10:
                    continue
                m, s = empirical_floor(sgn * sc[base], y[base], check_floor)
                n1 = int((y & base).sum())
                cr.append([l, kind, st, check_floor, fmt(m, 4), fmt(s, 4),
                           fmt(perm_sd(n1, int(base.sum()) - n1), 4)])
        if cr:
            out += [f"Permutation check of the floor (`full`, `V_pre`, step 64000):", "",
                    tbl(cr, ["l", "label", "step", "n perm", "shuffled mean", "shuffled sd",
                             "exact sd"]), ""]
    return "\n".join(out)


def sec_auc(Zs, an, tag, kind, sgn, diets, a=0):
    """AUC of each readout against one label, three checkpoints side by side, and the
    trained-minus-random difference on the identical rows."""
    rows = []
    for nm in diets:
        for l in LS:
            cells = [nm, l]
            for ro in READOUTS:
                vals = []
                for st in STEPS:
                    y, base = label_of(Zs[st], an, kind, l, a)
                    sc = score_of(Zs[st], an, ro, nm, l, a)
                    vals.append(float("nan") if sc is None or base.sum() < 20
                                else _auc(sgn * sc[base], y[base]))
                cells += [fmt(v) for v in vals]
                cells += [fmt(vals[2] - vals[0]), fmt(vals[1] - vals[0])]
            rows.append(cells)
    hdr = ["diet", "l"]
    for ro in READOUTS:
        hdr += [f"{ro} @0", f"{ro} @8k", f"{ro} @64k", f"{ro} d(64k-0)", f"{ro} d(8k-0)"]
    sign = "+" if sgn > 0 else "-"
    return "\n".join([
        f"**AUC({sign}score, {kind}) — {tag}, anchor {an}, a = {a}.**  `V_pre` = "
        f"`V[l, a+1](s_(t-1))`, the norm before the event; `V` = `V[l, a](s_t)`, the post-event "
        f"level (the object the banked response tables' `AUC(-V, .)` column reads); `R` = "
        f"`V[l, a](s_t) - V[l, a+1](s_(t-1))`, the revision.  `@0` is the trajectory's step-0 "
        f"checkpoint — a random-init trunk of the same architecture — and `d(64k-0)` is the "
        f"trained-minus-random difference on the identical rows.", "",
        tbl(rows, hdr), ""])


def sec_offsets(Zs, an, tag, diets, kind="outcome", sgn=+1, offs=(0, 4, 8)):
    """The same reading at the two later query offsets."""
    rows = []
    for nm in diets:
        for l in LS:
            cells = [nm, l]
            for a in offs:
                for st in STEPS:
                    y, base = label_of(Zs[st], an, kind, l, a)
                    sc = score_of(Zs[st], an, "V_pre", nm, l, a)
                    cells.append(fmt(float("nan") if sc is None or base.sum() < 20
                                     else _auc(sgn * sc[base], y[base])))
            rows.append(cells)
    hdr = ["diet", "l"] + [f"a={a} @{st}" for a in offs for st in STEPS]
    return "\n".join([
        f"**`AUC(V_pre, {kind})` at the later query offsets — {tag}, anchor {an}.**  `a` is the "
        f"distance from the anchor to the query position; the critic column is "
        f"`V[l, a+1](s_(t-1))` throughout, so each `a` is a different critic column read on the "
        f"same state against the outcome at that offset.", "", tbl(rows, hdr), ""])


FAMILIES = {"out": ["out_lo", "out_mid", "out_hi"], "dmg": ["dmg_lo", "dmg_mid", "dmg_hi"]}


def sec_family_spread(Zs, an, tag, kind, sgn, a=0):
    """Within a tercile family the diets are known to move `V_pre`'s LEVEL (norm README section 1).
    This puts the level's spread and the AUC's spread across the same three diets side by side, per
    level and checkpoint: `max - min` over the family, on the identical rows."""
    have = diets_of(Zs, an)
    rows = []
    for fam, nms in FAMILIES.items():
        if not all(nm in have for nm in nms):
            continue
        for l in LS:
            cells = [fam, l]
            for st in STEPS:
                y, base = label_of(Zs[st], an, kind, l, a)
                lv, au = [], []
                for nm in nms:
                    sc = score_of(Zs[st], an, "V_pre", nm, l, a)
                    if sc is None:
                        continue
                    lv.append(float(sc[base].mean()))
                    au.append(_auc(sgn * sc[base], y[base]))
                if len(lv) < 2:
                    cells += ["--", "--", "--", "--"]
                    continue
                cells += [fmt(min(lv), 4), fmt(max(lv), 4), fmt(max(lv) - min(lv), 4),
                          fmt(max(au) - min(au), 4)]
            rows.append(cells)
    if not rows:
        return ""
    hdr = ["family", "l"]
    for st in STEPS:
        hdr += [f"min level @{st}", f"max level @{st}", f"level spread @{st}",
                f"AUC spread @{st}"]
    sign = "+" if sgn > 0 else "-"
    return "\n".join([
        f"**Level spread against AUC spread inside each tercile family — {tag}, anchor {an}, "
        f"a = {a}, label `{kind}`, score `V_pre`.**  `level` is the mean of `V_pre` over the same "
        f"rows the AUC is taken on; `spread` is `max - min` across the family's three diets, which "
        f"differ only in their world's expected outcome (respectively realised damage) and share "
        f"their `(etype, j)` cell histogram bit-identically.  The AUC column is "
        f"`AUC({sign}V_pre, {kind})`.", "", tbl(rows, hdr), ""])


def sec_side_by_side(dirs, tag, an, kind, sgn, a=0, readouts=("V_pre", "R")):
    """The same cell read off every trajectory seed, printed side by side and NEVER averaged.

    Each entry is `seed1 / seed2 / ...` in the order the directories were given, which is the
    format `tables_s44.md` §0b uses."""
    cells_ = []
    for d in dirs:
        Zs = load_cells(d, tag)
        if Zs is None or f"{an}__t0" not in Zs[STEPS[0]].files:
            continue
        cells_.append((os.path.basename(os.path.normpath(d)), Zs))
    if not cells_:
        return ""
    diets = None
    for _, Zs in cells_:
        dd = set(diets_of(Zs, an))
        diets = dd if diets is None else (diets & dd)
    diets = order_diets(sorted(diets))
    rows = []
    for nm in diets:
        for l in LS:
            out = [nm, l]
            for ro in readouts:
                per = []                        # per seed: [auc@0, auc@8k, auc@64k]
                for _, Zs in cells_:
                    v = []
                    for st in STEPS:
                        y, base = label_of(Zs[st], an, kind, l, a)
                        sc = score_of(Zs[st], an, ro, nm, l, a)
                        v.append(float("nan") if sc is None or base.sum() < 20
                                 else _auc(sgn * sc[base], y[base]))
                    per.append(v)
                for i in range(3):
                    out.append(" / ".join(fmt(p[i]) for p in per))
                out.append(" / ".join(fmt(p[2] - p[0]) for p in per))
            rows.append(out)
    hdr = ["diet", "l"]
    for ro in readouts:
        hdr += [f"{ro} @0", f"{ro} @8k", f"{ro} @64k", f"{ro} d(64k-0)"]
    sign = "+" if sgn > 0 else "-"
    return "\n".join([
        f"**AUC({sign}score, {kind}) — {tag}, anchor {an}, a = {a}; "
        f"{' / '.join(nm for nm, _ in cells_)} side by side, never averaged.**  Each entry is one "
        f"number per trajectory seed in that order.  The seeds share the grammar, the stimuli, the "
        f"parse and the split, so the row sets are the same in all three; only the model moves.", "",
        tbl(rows, hdr), ""])


def sec_gate(d, tag, diets, a=0, kind="flip"):
    """Reproduce the banked response tables' matched-row `AUC(-R, flip)` / `AUC(-V, flip)`
    columns with `junction/analyze.py`'s own matcher, as the gate on this file's row handling."""
    out = []
    for st in STEPS:
        p = npz_of(d, tag, st)
        if not os.path.exists(p):
            continue
        for an in ANCHORS:
            R = Rows(p, an)
            if not R.ok:
                continue
            rows = []
            for l in LS:
                got = R.matched(kind, l, a, "kjt")
                if got is None:
                    continue
                mm, lab = got
                for nm in diets:
                    sc = R.scores(nm, l, a)
                    if "R" not in sc:
                        continue
                    rows.append([nm, l, len(mm), fmt(_auc(sc["R"][mm], lab[mm])),
                                 fmt(_auc(sc["V"][mm], lab[mm]))])
            if rows:
                out += [f"**step {st}, {tag}, anchor {an}, a = {a} — matched rows, "
                        f"`{kind}` label.**", "",
                        tbl(rows, ["diet", "l", "matched n", f"AUC(-R, {kind})",
                                   f"AUC(-V, {kind})"]), ""]
    return "\n".join(out)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

PREAMBLE = """# norm — the critic's held-out AUC, trained trunk against the random-init trunk

Facts only.  Generated by [`discrim.py`](../discrim.py) from the norm round's banked `.npz` files;
nothing is refit and no GPU is touched.  One file per trajectory seed below, never averaged across
seeds.

**Why this file exists beside [`tables.md`](tables.md).**  The banked round reports the critic's
held-out *fit* `R2` per diet (its section 0) and, in the response tables, the matched-row
`AUC(-R, flip)` / `AUC(-V, flip)` columns read off the **post-event** state.  It does not report
the **pre-event level** `V_pre`'s own ranking of the realised outcome, nor the difference between
the trained trunk and the trajectory's step-0 (random-init) trunk as a discrimination number.  That
is what is here.

**Objects.**  `V[l, d](s_t)` is the outcome-trained ridge critic on the frozen 8-layer trunk's
state, predicting `o_l(t + d)` — the frozen actor being RIGHT at level `l` at position `t + d` — so
higher `V` is a better world.  `V_pre = V[l, a+1](s_(t-1))` is the norm: what the critic expected
about the query's outcome one token BEFORE the event.  `V = V[l, a](s_t)` is the post-event level.
`R = V - V_pre` is the revision at the event, both terms predicting the same outcome.

**Rows.**  The npz holds the `split == 2` TEST windows only, so every row is held out from the
critic's Gram and from its lambda selection.  Within one trajectory the fixed rows at an anchor are
the same windows at the same positions at all three checkpoints (asserted per cell, `w` and `t0`
bit-identical), so `d(64k-0)` is a difference on identical rows.  `fd` is the edit-onset anchor and
`tv` is the first exactly-impossible token, where every row is a violation by construction; each
cell's etype histogram is printed, and note that on `swap65k` both anchors are 100% swap, so there
the two anchors differ only in the position, not in the population.

**Labels.**  `outcome` is `oe_l<l>_a<a>`, the actor's hit at the query on the edited stream — the
critic's own teacher.  `damage` is junction's `flip`, right on the unedited stream and wrong on the
edited one, on the base `ok & etype < 2 & oo`.  `cons` is the grammar changing its own level-`l`
label at the query.  **`cons` is a property of the world and is identical across checkpoints;
`outcome` and `damage` are the frozen actor's own hits, so those two labels move with the
checkpoint even though the rows do not** — the per-cell base rates are tabled so the movement is
visible, and `cons` is the one label for which `d(64k-0)` holds both rows and label fixed.

**Orientation and the floor.**  Every readout is in "higher = a better world" units, so `outcome`
is ranked by `+score` and the two bad-news labels by `-score`, which is the banked columns' sign.
The shuffled-label floor has mean exactly 0.500 and sd `sqrt((n+1)/(12 n1 n0))`, which depends only
on the label's balance and not on the score; it is tabled once per cell and checked by permutation.

**Venues.**  The tercile diets were only ever fitted on the `a1` venue (the sweep passes
`window_diets = 0` for `swap65k`), so `swap65k` carries `full` and the clean-only critic alone.

**Gate.**  Each venue's last section recomputes the banked response tables' matched-row
`AUC(-R, flip)` / `AUC(-V, flip)` columns with `junction/analyze.py`'s own matcher, so the row
handling here is checked against `tables.md` row for row.

Reproduction (CPU only, ~1 minute over the fetched files; no GPU, nothing refit):

```bash
cd experiments   # MODAL_PROFILE=chromatic
D=/v16_s2_L6_m4_distinct/logit_reading
for s in 42 43 44; do for st in 000000 008000 064000; do for tg in a1 swap65k; do
  modal volume get rhm-scaling-data $D/traj_a1_s$s/step${st}_norm_${tg}.npz <dir>/traj_a1_s$s/
done; done; done
python -m rhm.logit_reading.striatum.norm.discrim \\
    <dir>/traj_a1_s42 <dir>/traj_a1_s43 <dir>/traj_a1_s44 --tags a1,swap65k \\
    --out rhm/logit_reading/striatum/norm/results --outfile tables_discrim_20260919.md
```
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", help="fetched traj_* directories")
    ap.add_argument("--tags", default="a1,swap65k")
    ap.add_argument("--out", default=".")
    ap.add_argument("--outfile", default="tables_discrim.md")
    ap.add_argument("--a", type=int, default=0)
    ap.add_argument("--offsets", default="0,4,8")
    ap.add_argument("--check-floor", type=int, default=2000)
    ap.add_argument("--no-gate", dest="gate", action="store_false", default=True)
    args = ap.parse_args()

    tags = [t for t in args.tags.split(",") if t]
    offs = tuple(int(x) for x in args.offsets.split(",") if x != "")
    parts = [PREAMBLE]
    if len(args.dirs) > 1:
        parts.append("\n---\n\n# 0. The seeds side by side\n\nThe headline reading — the AUC of "
                     "the critic's pre-event level `V_pre` and of its revision `R`, per diet and "
                     "level, at each of the three checkpoints, with the trained-minus-random "
                     "difference — off every trajectory seed at once.  **Nothing here is averaged "
                     "across seeds**; each cell is the per-seed numbers in the order "
                     f"{' / '.join(os.path.basename(os.path.normpath(x)) for x in args.dirs)}.  "
                     "The per-seed sections below carry the same numbers with the post-event level "
                     "`V`, the row and label counts, the floor and the later query offsets.\n")
        for tag in tags:
            for an in ANCHORS:
                for kind, sgn in LABELS:
                    s = sec_side_by_side(args.dirs, tag, an, kind, sgn, args.a)
                    if s:
                        parts.append(s)
    for d in args.dirs:
        name = os.path.basename(os.path.normpath(d))
        parts.append(f"\n---\n\n# {name}\n")
        for tag in tags:
            Zs = load_cells(d, tag)
            if Zs is None:
                parts.append(f"## {tag}\n\nMISSING: not all of "
                             f"{[f'step{st:06d}' for st in STEPS]} are on file.\n")
                continue
            parts.append(f"## {tag}\n")
            for an in ANCHORS:
                if f"{an}__t0" not in Zs[STEPS[0]].files:
                    continue
                assert rows_identical(Zs, an), f"{name} {tag} {an}: rows differ across steps"
                diets = diets_of(Zs, an)
                parts.append(f"### {tag}, anchor {an}\n")
                parts.append(sec_rows(Zs, an, tag, args.a, args.check_floor))
                for kind, sgn in LABELS:
                    parts.append(sec_auc(Zs, an, tag, kind, sgn, diets, args.a))
                for kind, sgn in LABELS:
                    s = sec_family_spread(Zs, an, tag, kind, sgn, args.a)
                    if s:
                        parts.append(s)
                if len(offs) > 1:
                    parts.append(sec_offsets(Zs, an, tag, diets, offs=offs))
            if args.gate:
                g = sec_gate(d, tag, diets_of(Zs, "tv") if "tv__t0" in Zs[STEPS[0]].files
                             else diets_of(Zs, "fd"), args.a)
                if g:
                    parts.append(f"### Gate — the banked matched-row columns, recomputed "
                                 f"({tag})\n\nThe banked response tables' `AUC(-R, flip)` / "
                                 f"`AUC(-V, flip)` columns, recomputed here with "
                                 f"`junction/analyze.py`'s own matcher (`Rows.matched`, exact "
                                 f"strata on `(k*, j, position bucket)` with sign-balanced "
                                 f"surprisal pairs at caliper 0.30, seed 0).  These must equal "
                                 f"`tables.md`'s columns row for row.\n")
                    parts.append(g)
    os.makedirs(args.out, exist_ok=True)
    p = os.path.join(args.out, args.outfile)
    with open(p, "w") as f:
        f.write("\n".join(parts).rstrip() + "\n")
    print(f"wrote {p}  ({os.path.getsize(p) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
