"""The norm diets: worlds that differ in EXPECTED COST and in nothing structural.

`junction/diets.py` moved the *joint* of (legality, consequence depth) and, as the
positive control there showed, reached the critic through the edit's WIDTH.  This node
needs the opposite knob: the base rate of cost, with legality and width held fixed.  Two
families, both pure numpy so they can be checked without a GPU.

FAMILY A -- the outcome-stratified diets (`out_*`, `dmg_*`).
In this grammar the (legality, consequence) cell of a window is a deterministic function
of `(etype, j)` (junction's first gotcha).  So stratifying WITHIN each `(etype, j)` cell
and taking the same number of windows from every cell leaves the diet's joint
distribution over `(etype, j)` -- hence P(illegal), the width distribution and the
structural consequence-depth distribution -- *exactly* identical across diets, while the
cut variable moves freely.  Cut on the critic's own teacher:

    out_lo / out_mid / out_hi   lowest / middle / highest tercile of the window's mean
                                OUTCOME (the actor is right), within its (etype, j) cell
    dmg_lo / dmg_mid / dmg_hi   the same on realised DAMAGE (right on the original window,
                                wrong on the edited one)

These are worlds with the same events and different consequences, which is what Xiang,
Lohrenz & Montague's low / medium / high offer groups are: the same game, a different
outcome distribution.  The price is that selecting on the outcome also selects on whatever
state features predict it, so the shift in `V` is allowed to be state-dependent rather than
a pure intercept -- which is exactly the thing Q1a measures rather than assumes.

FAMILY B -- the quiet-share diets (`mix_q*`).
A knob on expected cost that touches no outcome at all: hold the TOTAL number of training
windows fixed and vary how many of them are unedited (`none`) windows, the edited ones
being nested random subsets of one shuffle.  A `none` window costs exactly zero realised
damage by construction, so the world's expected cost falls as the quiet share rises,
while the composition of the EDITED material (etype, width, consequence) is untouched and
unselected.  The price is the mirror image of family A's: the edited exposure shrinks as
the quiet share grows, so size of the edit sample and expected cost move together.  The
two families are read together; neither is confounded the way the other is.

Anchors carried alongside: `full` (every training window -- reproduces the banked
striatum/junction critic) and `natural_sized` (a random subset of all training windows at
the stratified diets' size).
"""

import numpy as np

STRAT_NAMES = ("lo", "mid", "hi")


def cell_key(etype, j):
    """The (legality, width) cell.  `(etype, j)` fixes the structural consequence profile."""
    return np.asarray(etype).astype(np.int64) * 100 + np.asarray(j).astype(np.int64)


def strat_diets(score, etype, j, is_tr, seed=0, frac=1.0 / 3.0, prefix="out", min_cell=6):
    """Within-cell terciles of `score`.

    Returns (masks, spec, meta).  masks[f"{prefix}_lo"|"_mid"|"_hi"] are boolean over all
    windows, always subsets of `is_tr`, and have IDENTICAL `(etype, j)` histograms."""
    rng = np.random.default_rng(seed)
    score = np.asarray(score, np.float64)
    n = len(etype)
    key = cell_key(etype, j)
    masks = {f"{prefix}_{k}": np.zeros(n, bool) for k in STRAT_NAMES}
    cells = {}
    for kk in np.unique(key[is_tr]):
        w = np.where(is_tr & (key == kk))[0]
        if len(w) < min_cell:
            continue
        o = w[np.lexsort((rng.random(len(w)), score[w]))]      # random tie-break
        k = int(len(o) * frac)
        if k < 1:
            continue
        mid0 = (len(o) - k) // 2
        masks[f"{prefix}_lo"][o[:k]] = True
        masks[f"{prefix}_mid"][o[mid0:mid0 + k]] = True
        masks[f"{prefix}_hi"][o[-k:]] = True
        cells[int(kk)] = {"n_pool": int(len(w)), "per_diet": int(k),
                          "score_lo": float(score[o[:k]].mean()),
                          "score_mid": float(score[o[mid0:mid0 + k]].mean()),
                          "score_hi": float(score[o[-k:]].mean())}
    spec = {f"{prefix}_{k}": {"kind": "window", "family": prefix, "tercile": k,
                              "cut_on": prefix, "frac": float(frac)} for k in STRAT_NAMES}
    meta = {f"{prefix}_cells": cells,
            f"{prefix}_n_windows": {k: int(masks[f"{prefix}_{k}"].sum()) for k in STRAT_NAMES}}
    return masks, spec, meta


def mix_diets(etype, is_tr, n_total=0, quiet_counts=(), seed=0, prefix="mix"):
    """Total window count fixed, quiet (`none`) share varied; the edited windows of the
    quieter diets are nested subsets of the edited windows of the busier ones."""
    rng = np.random.default_rng(seed)
    n = len(etype)
    etype = np.asarray(etype)
    ed = rng.permutation(np.where(is_tr & (etype < 2))[0])
    qu = rng.permutation(np.where(is_tr & (etype == 2))[0])
    if not len(qu):
        return {}, {}, {"mix": "no quiet pool"}
    if not n_total:
        n_total = min(len(ed), len(ed) + len(qu))
    if not quiet_counts:
        # thirds of the quiet pool: the busiest diet is all edits, the quietest still keeps
        # about half its windows edited, so every diet has an edit sample worth fitting on
        quiet_counts = (0, len(qu) // 3, 2 * len(qu) // 3)
    masks, spec, meta = {}, {}, {}
    for q in quiet_counts:
        q = int(min(q, len(qu), n_total))
        e = int(min(n_total - q, len(ed)))
        m = np.zeros(n, bool)
        m[ed[:e]] = True
        m[qu[:q]] = True
        share = q / max(e + q, 1)
        nm = f"{prefix}_q{int(round(100 * share)):02d}"
        masks[nm] = m
        spec[nm] = {"kind": "window", "family": prefix, "n_edit": e, "n_quiet": q,
                    "quiet_share": float(share)}
    meta[f"{prefix}_n_total"] = int(n_total)
    meta[f"{prefix}_pool"] = {"edited": int(len(ed)), "quiet": int(len(qu))}
    return masks, spec, meta


def anchor_diets(etype, is_tr, n_total=0, seed=0):
    """`full` (every training window) and `natural_sized` (a random subset at `n_total`)."""
    rng = np.random.default_rng(seed)
    n = len(etype)
    masks, spec = {}, {}
    m = np.zeros(n, bool)
    m[is_tr] = True
    masks["full"] = m
    spec["full"] = {"kind": "window", "family": "anchor",
                    "note": "every training window (banked striatum/junction diet)"}
    if n_total:
        tr = np.where(is_tr)[0]
        m = np.zeros(n, bool)
        m[rng.choice(tr, int(min(n_total, len(tr))), replace=False)] = True
        masks["natural_sized"] = m
        spec["natural_sized"] = {"kind": "window", "family": "anchor",
                                 "note": "random subset of all training windows, size-matched"}
    return masks, spec
