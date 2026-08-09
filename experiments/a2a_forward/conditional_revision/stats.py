"""Matching / AUC / partial-R^2 helpers.

Ported verbatim in behaviour from `rhm/conditional_revision/gates_ab.py` so the
language numbers are directly comparable with the RHM ones. The two gotchas
that module paid for in blood are preserved and re-documented here:

  * `_strata` is **atom-aware**. Oracle surprisal on a designed substrate is
    almost entirely atomic (`log 4` everywhere by construction here), so plain
    quantile edges repeat and `digitize` merges an atom with the continuous
    spread above it. RHM observed the matching variable separating the families
    at AUC 0.999 *inside* a stratum meant to hold it fixed.
  * crossed strata are encoded by `_cross`, not by `a * K + b`, which silently
    aliases whenever the second variable has more atoms than `K`.

Every Gate-B block reports the matching variable's *self*-matched AUC as a
guard; those must land at ~0.5 or the column is not interpretable.
"""

import numpy as np


def _r2(y, x):
    xc, yc = x - x.mean(), y - y.mean()
    denom = (xc * xc).sum()
    if denom < 1e-30:
        return 0.0
    beta = (xc * yc).sum() / denom
    ss_res = ((yc - beta * xc) ** 2).sum()
    ss_tot = (yc * yc).sum()
    return float(1.0 - ss_res / max(ss_tot, 1e-30))


def _resid(y, x):
    xc, yc = x - x.mean(), y - y.mean()
    denom = (xc * xc).sum()
    beta = 0.0 if denom < 1e-30 else (xc * yc).sum() / denom
    return yc - beta * xc


def _partial_r2(y, x, z):
    """Fraction of y's z-orthogonal variance explained by x's z-orthogonal part."""
    return _r2(_resid(y, z), _resid(x, z))


def _ranks(a):
    from scipy.stats import rankdata
    return rankdata(a)


def _partial_r2_rank(y, x, z):
    return _partial_r2(_ranks(y), _ranks(x), _ranks(z))


def _auc(score, pos_mask):
    """P(score[positive] > score[negative]), ties at 1/2."""
    n1 = int(pos_mask.sum())
    n0 = int((~pos_mask).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = _ranks(score)
    return float((r[pos_mask].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def _stratified_auc(score, pos_mask, strata):
    """AUC within each stratum, pooled by pair count."""
    num = den = 0.0
    n_used = 0
    for k in np.unique(strata):
        sel = strata == k
        n1 = int(pos_mask[sel].sum())
        n0 = int(sel.sum() - n1)
        if n1 and n0:
            num += _auc(score[sel], pos_mask[sel]) * n1 * n0
            den += n1 * n0
            n_used += 1
    return (float(num / den) if den else float("nan")), n_used, float(den)


def _strata(vals, n_bins=8, atom_frac=0.002):
    """Tie-pure matching strata: every value carrying >= `atom_frac` of the mass
    becomes its own stratum; the remainder is quantile-binned."""
    vals = np.asarray(vals, dtype=float)
    uniq, counts = np.unique(vals, return_counts=True)
    atoms = uniq[counts >= max(atom_frac * vals.size, 2)]
    out = np.full(vals.size, -1, dtype=np.int64)
    for i, a in enumerate(atoms):
        out[vals == a] = i
    rest = out < 0
    if rest.any():
        edges = np.unique(np.quantile(vals[rest], np.linspace(0, 1, n_bins + 1)))
        out[rest] = len(atoms) + (np.digitize(vals[rest], edges[1:-1])
                                  if edges.size > 2 else 0)
    return out


def _cross(*strata_list):
    """Exact crossing of several integer stratifications (no aliasing)."""
    keys = np.stack([np.asarray(s, dtype=np.int64) for s in strata_list], axis=1)
    _, inv = np.unique(keys, axis=0, return_inverse=True)
    return inv.astype(np.int64)


def _codes(labels):
    """Integer codes for a list of hashable labels (frames, families, ...)."""
    uniq = {v: i for i, v in enumerate(sorted(set(labels)))}
    return np.array([uniq[v] for v in labels], dtype=np.int64)


def _shuffle_within(strata, rng):
    """A permutation that only moves elements within a stratum. Used for the
    `*_shuffled` guard: destroys the sequence-specific content of a readout
    while leaving every matched-on structure intact. Must read ~0.5."""
    idx = np.arange(strata.size)
    out = idx.copy()
    for k in np.unique(strata):
        sel = np.where(strata == k)[0]
        out[sel] = rng.permutation(sel)
    return out
