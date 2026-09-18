"""The diets: which training rows enter the critic's Gram, and with what structure.

Two families, both pure numpy so they can be checked without a GPU.

FAMILY 1 -- the junction (window-level, `a1` only).  In this grammar the (legality,
consequence) cell of an edited window is a DETERMINISTIC function of `(etype, j)`:

    swap at width j   -> the true level-l answer changes for l <= j, never for l > j
    rare at width j   -> the true level-l answer changes for l <  j, never for l >= j

(verified on `parse_a1.npz`: P(change) is 0.92-1.00 inside the profile and exactly 0.000
outside it, for every (etype, j) cell).  So define a window's CONSEQUENCE DEPTH

    c = j       for a swap          c = j - 1   for a rare

and the profile "consequential exactly at levels l <= c" is shared by `swap j=c` and
`rare j=c+1`.  That pairing is what makes a controlled diet possible: the diet's joint
distribution over (illegal, c) can be set freely, while its MARGINALS -- P(illegal) and
the distribution of c -- are pinned identical across diets.  c in {0,1,2,3} is the common
support (swap j in {0..3}, rare j in {1..4}).

    aligned       illegal edits are the costly ones:  swap c in {2,3}, rare c in {0,1}
    reversed      legal edits are the costly ones:    swap c in {0,1}, rare c in {2,3}
    decorrelated  both at every c:                    swap c in {0..3}, rare c in {0..3}
    natural_sized the diet's natural (etype, j) mix, at the same window count
    full          every training window (reproduces the banked striatum run)

Those three carry q windows per used cell, so each has 4q edited windows, P(illegal) = 0.5,
and a uniform marginal over c in {0..3}.  A second, stronger family takes the extreme
cells, where "costly" and "costless" are exact rather than graded (c = 4 damages four
levels, c = 0 damages none):

    aligned_x     swap c = 4, rare c = 0      every illegal edit costly, every legal one free
    reversed_x    swap c = 0, rare c = 3      every legal edit costly, every illegal one free
    decorrelated_x  both at c in {lo, hi}     the same cells, association zero

A fixed number of `none` windows rides along in every constructed diet as a constant
background.  The one marginal the grammar will not let us pin is the damaged AREA: at a
matched consequence depth a rare edit spans twice the leaves a swap does, so the diets
differ in total cost content, and that content runs OPPOSITE to the association (aligned
is the least costly diet, reversed the most).  The two hypotheses therefore predict
opposite orderings of the readout; `diet_stats` reports both.

A third family cuts the cells on the critic's ACTUAL teacher rather than on the structure.
Realised damage is not the same object as structural consequence -- an illegal single-leaf
swap changes no true answer anywhere and still costs at the goal, because the state the
actor reads is perturbed downstream -- so selection on structure alone cannot drive the
world's violation-cost correlation far negative.  `damage_diets` therefore cuts the edited
training windows at the median of their realised damage and crosses that with legality:

    aligned_r     illegal & costly, legal & free
    reversed_r    illegal & free,   legal & costly
    decorrelated_r  all four cells equally

which spans the association axis by construction.  Its cost: selecting on realised damage
also selects on whatever else predicts damage, so the `_r` diets are not marginal-matched
the way the `c` family is.  The three families are read together.

FAMILY 2 -- the surprise gate (row-level, both venues).  A Gram row is a (window,
position) pair.  A gated diet keeps the top fraction of TRAINING rows by the model's own
realised horizon excess (`addendum.py`'s column), against matched-size controls:

    gate_top   top f by realised horizon excess          (the surprise-selected diet)
    gate_bot   bottom f by the same                      (the anti-gate)
    rand       uniform random f of rows                  (the matched-size control)
    randpos    random f, resampled to the gated set's position histogram (position guard)
    oracle     top f by realised damage at that row      (the selection ceiling)
"""

import numpy as np

CS = [0, 1, 2, 3]                      # consequence depths with both a swap and a rare cell
DIET_CELLS = {
    "aligned":        {"swap": [2, 3], "rare": [0, 1]},
    "reversed":       {"swap": [0, 1], "rare": [2, 3]},
    "decorrelated":   {"swap": [0, 1, 2, 3], "rare": [0, 1, 2, 3]},
    "aligned_x":      {"swap": [4], "rare": [0]},
    "reversed_x":     {"swap": [0], "rare": [3]},
    "decorrelated_x": {"swap": [0, 4], "rare": [0, 3]},
}
FAMILY = {"aligned": "c", "reversed": "c", "decorrelated": "c",
          "aligned_x": "x", "reversed_x": "x", "decorrelated_x": "x"}
GATE_ARMS = ["gate_top", "gate_bot", "rand", "randpos", "oracle"]


def cons_depth(etype, j):
    """c: the deepest query level whose true answer the edit changes (-1 if none)."""
    return np.where(etype == 0, j, np.where(etype == 1, j - 1, -1)).astype(np.int64)


def window_diets(etype, j, is_tr, seed=0, q=0, none_ratio=0.343, n_edit_target=0):
    """Window-level diets.  Returns (masks, spec, meta, none_sel) with masks[name] a bool
    mask over all n windows, always a subset of `is_tr`."""
    rng = np.random.default_rng(seed)
    n = len(etype)
    c = cons_depth(etype, j)
    pool = {}
    for et, nm in ((0, "swap"), (1, "rare")):
        for cc in range(-1, 6):
            w = np.where(is_tr & (etype == et) & (c == cc))[0]
            if len(w):
                pool[(nm, cc)] = w
    none_pool = np.where(is_tr & (etype == 2))[0]

    # every "c" diet gets 4q edited windows, every "x" diet 2q; per-cell quota follows
    def per_of(name):
        cells = DIET_CELLS[name]
        tot = (4.0 if FAMILY[name] == "c" else 2.0)
        return tot / (len(cells["swap"]) + len(cells["rare"]))
    need = {}                                   # cell -> max quota (in units of q)
    for name, cells in DIET_CELLS.items():
        for nm, cl in cells.items():
            for cc in cl:
                need[(nm, cc)] = max(need.get((nm, cc), 0.0), per_of(name))
    q_max = int(min(len(pool[k]) / v for k, v in need.items() if v > 0))
    q = min(q, q_max) if q else q_max
    n_edit = 4 * q
    if n_edit_target:
        n_edit = min(n_edit, n_edit_target)
    n_none = min(len(none_pool), int(round(n_edit * none_ratio)))
    none_sel = rng.choice(none_pool, n_none, replace=False)

    masks, spec = {}, {}
    m = np.zeros(n, bool)
    m[is_tr] = True
    masks["full"] = m
    spec["full"] = {"kind": "window", "note": "every training window (banked striatum diet)"}

    edited_tr = np.where(is_tr & (etype < 2))[0]
    m = np.zeros(n, bool)
    m[rng.choice(edited_tr, min(n_edit, len(edited_tr)), replace=False)] = True
    m[none_sel] = True
    masks["natural_sized"] = m
    spec["natural_sized"] = {"kind": "window", "note": "natural (etype, j) mix, size-matched"}

    for name, cells in DIET_CELLS.items():
        per = int(round(q * per_of(name)))
        m = np.zeros(n, bool)
        for nm, cl in cells.items():
            for cc in cl:
                take = min(per, len(pool[(nm, cc)]))
                m[rng.choice(pool[(nm, cc)], take, replace=False)] = True
        m[none_sel] = True
        masks[name] = m
        spec[name] = {"kind": "window", "cells": cells, "per_cell": int(per),
                      "family": FAMILY[name]}
    meta = {"q": int(q), "q_max": int(q_max), "n_edit": int(n_edit), "n_none": int(n_none),
            "pool_sizes": {f"{k[0]}_c{k[1]}": int(len(v)) for k, v in pool.items()},
            "none_pool": int(len(none_pool))}
    return masks, spec, meta, none_sel


DMG_CELLS = {"aligned_r": [("swap", 1), ("rare", 0)],
             "reversed_r": [("swap", 0), ("rare", 1)],
             "decorrelated_r": [("swap", 0), ("swap", 1), ("rare", 0), ("rare", 1)]}


def damage_diets(etype, dmg_win, is_tr, none_sel, seed=0, q=0):
    """Cells cut on realised damage (the critic's own teacher) crossed with legality.

    `dmg_win`: per-window realised damage (right on the original window, wrong on the
    edited one), averaged over query levels and Gram positions.  The median is taken over
    the EDITED training windows, so "costly" means costly relative to the other edits."""
    rng = np.random.default_rng(seed)
    n = len(etype)
    ed = is_tr & (etype < 2)
    med = float(np.median(dmg_win[ed]))
    hi = dmg_win > med
    pool = {(nm, h): np.where(ed & (etype == et) & (hi == bool(h)))[0]
            for et, nm in ((0, "swap"), (1, "rare")) for h in (0, 1)}
    need = {}
    for name, cl in DMG_CELLS.items():
        per = 2.0 / len(cl)
        for k in cl:
            need[k] = max(need.get(k, 0.0), per)
    q_max = int(min(len(pool[k]) / v for k, v in need.items()))
    q = min(q, q_max) if q else q_max
    masks, spec = {}, {}
    for name, cl in DMG_CELLS.items():
        per = int(round(q * 2.0 / len(cl)))
        m = np.zeros(n, bool)
        for k in cl:
            m[rng.choice(pool[k], min(per, len(pool[k])), replace=False)] = True
        m[none_sel] = True
        masks[name] = m
        spec[name] = {"kind": "window", "family": "r", "cells": [list(x) for x in cl],
                      "per_cell": int(per), "median_damage": med}
    meta = {"q_r": int(q), "q_r_max": int(q_max), "median_damage": med,
            "pool_sizes": {f"{k[0]}_{'hi' if k[1] else 'lo'}": int(len(v))
                           for k, v in pool.items()}}
    return masks, spec, meta


def row_diets(score_hex, score_dmg, tr_idx, rows, fracs, seed=0, pos_bins=8):
    """Row-level diets over the (training window, position) grid.

    score_hex / score_dmg: (n, T) arrays; only [tr_idx][:, rows] is used.  Returns
    (masks, spec) with masks[name] a bool array of shape (len(tr_idx), len(rows))."""
    rng = np.random.default_rng(seed)
    H = np.asarray(score_hex)[np.ix_(tr_idx, rows)].astype(np.float64)
    D = np.asarray(score_dmg)[np.ix_(tr_idx, rows)].astype(np.float64)
    H = np.nan_to_num(H, nan=-1e9)
    nr = H.size
    pos = np.broadcast_to(np.arange(len(rows))[None, :], H.shape)
    pb = np.minimum((pos * pos_bins) // len(rows), pos_bins - 1).reshape(-1)
    hf = H.reshape(-1)
    df = D.reshape(-1) + rng.random(nr) * 1e-6              # break damage ties
    order_h = np.argsort(-hf, kind="stable")
    order_d = np.argsort(-df, kind="stable")
    masks, spec = {}, {}
    for f in fracs:
        k = max(200, int(round(f * nr)))
        if k >= nr:
            continue
        top = order_h[:k]
        bot = order_h[-k:]
        rnd = rng.choice(nr, k, replace=False)
        want = np.bincount(pb[top], minlength=pos_bins)     # position-matched random
        pm = []
        for b in range(pos_bins):
            cand = np.where(pb == b)[0]
            pm.append(rng.choice(cand, min(int(want[b]), len(cand)), replace=False))
        pm = np.concatenate(pm)
        for nm, sel in (("gate_top", top), ("gate_bot", bot), ("rand", rnd),
                        ("randpos", pm), ("oracle", order_d[:k])):
            m = np.zeros(nr, bool)
            m[sel] = True
            masks[f"{nm}_f{f:g}"] = m.reshape(H.shape)
            spec[f"{nm}_f{f:g}"] = {"kind": "row", "arm": nm, "frac": float(f),
                                    "n_rows": int(m.sum())}
    masks["rowfull"] = np.ones(H.shape, bool)
    spec["rowfull"] = {"kind": "row", "arm": "full", "frac": 1.0, "n_rows": int(H.size)}
    return masks, spec
