"""Recurrence structure of the code grid: are 2x2 code blocks PARTS or TEXTURES?

`macros.py`'s vocabulary is nested over ENTRIES -- `T[l]` is a set of s-tuples of `T[l-1]`
entry indices, and an `T[3]` entry whose halves are not both in `T[2]` is dropped at build
time. This module measures the precondition for that on the code grid, before any loop runs:

    T[2] candidates : the 2x2 blocks of a 16x16 code grid           (8x8 = 64 per swatch)
    T[3] candidates : the 2x2 blocks OF THOSE, with sub-support entries mapped to OOV
                      (4x4 = 16 per swatch)  -- exactly the ratchet constraint

Non-overlapping tiling is primary because that is the nesting `macros.py` uses; overlapping
windows are reported beside it as the robustness column (they count the same structure without
the tiling's phase choice).

What separates a part from a texture, operationally: a part RECURS -- few distinct blocks
carrying most of the occurrences, and a large fraction of occurrences at support, at BOTH
levels. A texture spends its blocks: many distinct, flat concentration, little at support, and
level 3 collapses to OOV because nothing below it recurred often enough to be an entry.
"""

import numpy as np


def _blockify(g, grid, ids=None):
    """(N, grid*grid) ids -> (N, (grid//2)^2) tuple-ids of the non-overlapping 2x2 blocks."""
    N = g.shape[0]
    x = g.reshape(N, grid, grid)
    b = np.stack([x[:, 0::2, 0::2], x[:, 0::2, 1::2], x[:, 1::2, 0::2], x[:, 1::2, 1::2]], -1)
    return b.reshape(N, -1, 4)


def _overlap(g, grid):
    N = g.shape[0]
    x = g.reshape(N, grid, grid)
    b = np.stack([x[:, :-1, :-1], x[:, :-1, 1:], x[:, 1:, :-1], x[:, 1:, 1:]], -1)
    return b.reshape(N, -1, 4)


def _ids(tuples):
    """(N, B, 4) -> (N, B) integer ids over the distinct tuples seen, plus counts."""
    flat = tuples.reshape(-1, tuples.shape[-1])
    uniq, inv, cnt = np.unique(flat, axis=0, return_inverse=True, return_counts=True)
    return inv.reshape(tuples.shape[:2]), cnt, uniq


def _conc(cnt, support):
    """Concentration readouts over a count vector."""
    n = cnt.sum()
    p = cnt / n
    H = float(-(p * np.log(p)).sum())
    srt = np.sort(cnt)[::-1]
    cum = np.cumsum(srt) / n
    return {
        "n_obs": int(n),
        "n_distinct": int(len(cnt)),
        "distinct_per_obs": float(len(cnt) / n),
        "entropy_nats": H,
        "perplexity": float(np.exp(H)),
        "top64_mass": float(cum[min(63, len(cum) - 1)]),
        "n_cover50": int(np.searchsorted(cum, 0.50) + 1),
        "n_cover90": int(np.searchsorted(cum, 0.90) + 1),
        "n_at_support": int((cnt >= support).sum()),
        "mass_at_support": float(cnt[cnt >= support].sum() / n),
    }


def style_recurrence(codes, grid=16, support=8):
    """codes: (N, grid*grid) for ONE style. Returns the level-2 and level-3 readouts."""
    t2 = _blockify(codes, grid)
    id2, cnt2, _ = _ids(t2)
    r2 = _conc(cnt2, support)
    r2o = _conc(_ids(_overlap(codes, grid))[1], support)

    # ratchet constraint: a level-3 entry is only representable over level-2 entries AT SUPPORT
    keep = cnt2 >= support
    oov = int(keep.sum())
    id2r = np.where(keep[id2], np.cumsum(keep)[id2] - 1, oov)          # OOV gets the last id
    t3 = _blockify(id2r, grid // 2)
    _, cnt3, uniq3 = _ids(t3)
    r3 = _conc(cnt3, support)
    pure = np.array([(u != oov).all() for u in uniq3])
    r3["frac_obs_all_at_support"] = float(cnt3[pure].sum() / cnt3.sum())
    r3["n_distinct_pure"] = int(pure.sum())

    # unconstrained level 3, for the record (what the count would be with no ratchet)
    t3u = _blockify(id2, grid // 2)
    r3u = _conc(_ids(t3u)[1], support)
    return {"T2": r2, "T2_overlap": r2o, "T3": r3, "T3_unconstrained": r3u,
            "n_level2_at_support": oov}
