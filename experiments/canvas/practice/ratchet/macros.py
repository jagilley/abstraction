"""EARNED vocabulary on the code grid: macro moves whose composition table is mined from the
agent's own successful fills.

FORK NOTICE. This is a fork of `rhm/practice/ratchet/macros.py` (the donor is untouched). The
table algebra -- `base_table`, `make_table`, `_flatten`, `Miner`, `grade_table` -- is the
donor's, verbatim in behaviour; what changes is the SUBSTRATE:

    donor (RHM)                          here (canvas, aligned tiles)
    s = 2, a 1-D string of blocks        s = 4, a 2x2 SPATIAL arrangement on a 16x16 code grid
    T[1] = v level-1 features            T[1] = the K = 512 codes, one per grid cell
    T[2] = s-tuples of T[1]              T[2] = 2x2 code blocks  ==  ONE TILE (aligned)
    T[3] = s-tuples of T[2] entries      T[3] = 2x2 blocks of those == a 2x2-TILE MOTIF
    the parse is a learned reader        the parse is BLOCKIFY -- the alphabet IS the parse,
                                         so nothing above the codebook has to be read

FLATTENING CONVENTION. `_flatten` concatenates a parent's `s` children's level-1 expansions in
child order, so a level-l entry's `flat` row is the block's cells in QUADTREE (Z) order, not
raster order: level 2 is (TL, TR, BL, BR) of a 2x2 cell block; level 3 is the four 2x2 blocks
in that same (TL, TR, BL, BR) order, each itself in Z order. That is exactly
`canvas/plant/recur.py:_blockify`'s ordering at both levels, which is what lets a mined table
and the corpus recurrence numbers be the same object. `zcells` materialises it.

WHY IT RATCHETS -- unchanged from the donor. T[l] is defined over T[l-1] ENTRIES, so a motif
whose four tiles are not all in the level-2 vocabulary is not representable and is dropped at
build time: an incomplete committed T[2] caps T[3] forever.

WHERE THE TUPLES COME FROM. Only from fills the agent actually SOLVED (every touched adjacent
code pair in support), and only from blocks lying ENTIRELY INSIDE THE HOLE -- the agent's own
answers, never the exemplar surround it was handed.
"""

import numpy as np

S = 4                       # a level-l entry is 4 level-(l-1) entries in a 2x2 arrangement
GRID = 16


# --------------------------------------------------------------------------- #
# geometry: which cells a (level, node) block covers, in the flattening's order
# --------------------------------------------------------------------------- #

def n_nodes(level, grid=GRID):
    side = 2 ** (level - 1)
    return (grid // side) ** 2


def zcells(level, node, grid=GRID):
    """Cell indices (row-major in the grid) of the level-`level` block at `node`, in the
    quadtree order `_flatten` produces. `node` is raster order over that level's block grid."""
    if level == 1:
        return np.array([node], np.int64)
    nside = grid // (2 ** (level - 1))
    r, c = divmod(int(node), nside)
    out = []
    for dr in (0, 1):
        for dc in (0, 1):
            out.append(zcells(level - 1, (2 * r + dr) * (2 * nside) + (2 * c + dc), grid))
    return np.concatenate(out)


def zorder(side):
    """(side*side,) cell offsets inside a side x side block, in quadtree order -> row-major
    offsets r*side + c. `zorder(2) == [0, 1, 2, 3]`; `zorder(4)` is the Z curve."""
    level = int(np.log2(side)) + 1
    cells = zcells(level, 0, grid=side)
    return cells


def blockify(g, grid):
    """(N, grid*grid) ids -> (N, (grid//2)^2, 4) 2x2 blocks in (TL, TR, BL, BR) order.
    `canvas/plant/recur.py:_blockify`, verbatim."""
    N = g.shape[0]
    x = g.reshape(N, grid, grid)
    b = np.stack([x[:, 0::2, 0::2], x[:, 0::2, 1::2], x[:, 1::2, 0::2], x[:, 1::2, 1::2]], -1)
    return b.reshape(N, -1, 4)


# --------------------------------------------------------------------------- #
# tables  (donor algebra)
# --------------------------------------------------------------------------- #

def _flatten(child, lower_flat):
    if child.shape[0] == 0:
        return np.zeros((0, lower_flat.shape[1] * child.shape[1]), np.int64)
    return lower_flat[child].reshape(child.shape[0], -1)


def base_table(v):
    """T[1]: the v codes, each its own entry."""
    idx = np.arange(v, dtype=np.int64)[:, None]
    return {"level": 1, "child": idx, "flat": idx, "feature": idx[:, 0], "lower": None}


def make_table(level, child, lower, s=S, feature=None):
    child = np.asarray(child, np.int64).reshape(-1, s)
    return {"level": level, "child": child, "flat": _flatten(child, lower["flat"]),
            "feature": (np.full(child.shape[0], -1, np.int64) if feature is None
                        else np.asarray(feature, np.int64)),
            "lower": lower}


def truncate(table, n):
    """First `n` entries, over the same lower table. Used by the ratchet gate."""
    return make_table(table["level"], table["child"][:n], table["lower"], S)


# --------------------------------------------------------------------------- #
# mining  (donor algebra)
# --------------------------------------------------------------------------- #

class Miner:
    """Counts level-1 (code) tuples observed over one block of one level, over the whole run.

    Keyed by the FLATTENED code tuple, not by lower-entry ids, so counts stay stable while the
    lower table is still growing; the ratchet constraint is applied at `build` time instead."""

    def __init__(self, level, s=S):
        self.level = level
        self.s = s
        self.span = s ** (level - 1)
        self.counts = {}
        self.n_obs = 0

    def observe(self, feats):
        feats = np.asarray(feats, np.int64).reshape(-1, self.span)
        if feats.size == 0:
            return
        self.n_obs += feats.shape[0]
        for row in feats:
            k = tuple(int(x) for x in row)
            self.counts[k] = self.counts.get(k, 0) + 1

    def build(self, lower, support):
        """Entries with count >= support whose s quarters are all present in `lower`."""
        half = self.span // self.s
        lut = {tuple(int(x) for x in row): i for i, row in enumerate(lower["flat"])}
        rows = []
        for k, c in sorted(self.counts.items()):
            if c < support:
                continue
            kids = [lut.get(k[i * half:(i + 1) * half]) for i in range(self.s)]
            if any(j is None for j in kids):
                continue
            rows.append(kids)
        child = np.array(rows, np.int64).reshape(len(rows), self.s)
        return make_table(self.level, child, lower, self.s)

    def state(self):
        return {"level": self.level, "n_obs": self.n_obs, "n_distinct": len(self.counts),
                "n_at_support": {str(t): int(sum(1 for c in self.counts.values() if c >= t))
                                 for t in (1, 2, 3, 5, 10)}}


def grade_table(learned, truth):
    """Precision / recall of an earned table against the given (oracle) one, as SETS of
    flattened code tuples."""
    L = {tuple(int(x) for x in r) for r in learned["flat"]}
    T = {tuple(int(x) for x in r) for r in truth["flat"]}
    inter = len(L & T)
    return {"n_learned": len(L), "n_true": len(T), "n_correct": inter,
            "precision": (inter / len(L)) if L else None,
            "recall": (inter / len(T)) if T else 0.0}


# --------------------------------------------------------------------------- #
# the macro operator, on the code grid
# --------------------------------------------------------------------------- #

def pair_slots(side):
    """Internal adjacency slots of a side x side block, as index pairs into the block's
    QUADTREE-ordered span positions. Returns (h_pairs, v_pairs), each (n, 2)."""
    z = zorder(side)                                  # span position -> row-major offset
    pos = {int(o): i for i, o in enumerate(z)}
    h, v = [], []
    for r in range(side):
        for c in range(side):
            i = pos[r * side + c]
            if c + 1 < side:
                h.append((i, pos[r * side + c + 1]))
            if r + 1 < side:
                v.append((i, pos[(r + 1) * side + c]))
    return np.array(h, np.int64).reshape(-1, 2), np.array(v, np.int64).reshape(-1, 2)


def boundary_slots(level, node, grid=GRID):
    """Pairs between the block's border cells and the cells just outside it. ALWAYS 4*side
    slots (a corner cell contributes two), so the shape is node-independent and the whole
    level vectorises; `out == -1` marks a slot that falls off the grid.
    Returns (span_pos, outside_cell, orient, block_is_first); `orient` 0 = H, 1 = V;
    `block_is_first` True when the block's own cell is the LEFT / ABOVE member of the pair."""
    side = 2 ** (level - 1)
    nside = grid // side
    br, bc = divmod(int(node), nside)
    r0, c0 = br * side, bc * side
    z = zorder(side)
    pos = {int(o): i for i, o in enumerate(z)}
    sp, out, orient, first = [], [], [], []
    for r in range(side):
        for c in range(side):
            i = pos[r * side + c]
            gr, gc = r0 + r, c0 + c
            if c == 0:                                  # W neighbour: (nb, x) in H
                sp.append(i); orient.append(0); first.append(False)
                out.append(gr * grid + gc - 1 if gc > 0 else -1)
            if c == side - 1:                           # E neighbour: (x, nb) in H
                sp.append(i); orient.append(0); first.append(True)
                out.append(gr * grid + gc + 1 if gc < grid - 1 else -1)
            if r == 0:                                  # N neighbour: (nb, x) in V
                sp.append(i); orient.append(1); first.append(False)
                out.append((gr - 1) * grid + gc if gr > 0 else -1)
            if r == side - 1:                           # S neighbour: (x, nb) in V
                sp.append(i); orient.append(1); first.append(True)
                out.append((gr + 1) * grid + gc if gr < grid - 1 else -1)
    return (np.array(sp, np.int64), np.array(out, np.int64),
            np.array(orient, np.int64), np.array(first, bool))


def cell_neighbours(grid=GRID):
    """(T, 4) neighbour cell index (-1 off-grid) plus the (orient, block_is_first) of each
    direction, in W, E, N, S order -- the level-1 case of `boundary_slots` for the whole grid."""
    T = grid * grid
    nb = -np.ones((T, 4), np.int64)
    for r in range(grid):
        for c in range(grid):
            i = r * grid + c
            if c > 0:
                nb[i, 0] = i - 1
            if c < grid - 1:
                nb[i, 1] = i + 1
            if r > 0:
                nb[i, 2] = i - grid
            if r < grid - 1:
                nb[i, 3] = i + grid
    return nb, np.array([0, 0, 1, 1], np.int64), np.array([False, True, False, True])


def internal_unsup(table, supH, supV):
    """(n_entries,) count of the entry's OWN internal adjacent code pairs that are out of
    support. A property of the table and the support only, so it is computed once."""
    flat = table["flat"]
    if flat.shape[0] == 0:
        return np.zeros(0, np.int64)
    side = int(round(np.sqrt(flat.shape[1])))
    hp, vp = pair_slots(side)
    n = np.zeros(flat.shape[0], np.int64)
    if len(hp):
        n += (~supH[flat[:, hp[:, 0]], flat[:, hp[:, 1]]]).sum(1)
    if len(vp):
        n += (~supV[flat[:, vp[:, 0]], flat[:, vp[:, 1]]]).sum(1)
    return n


def chain_of(table):
    """The donor's `to_device` chain: child arrays from level 2 up to the table's level."""
    stack, t = [], table
    while t is not None and t["level"] >= 2:
        stack.append(t)
        t = t.get("lower")
    return [x["child"] for x in reversed(stack)]


def chain_scores(cur, chain, s=S):
    """The DONOR's max-sum DP (`macros.macro_features`), kept for the equivalence gate.
    `cur`: (B, span, K) per-cell scores over T[1] at the block's Z-ordered cells.
    Returns (B, n_entries) scores. Equal by construction to `flat_scores`."""
    import torch
    batch = cur.shape[0]
    for child in chain:
        n_par = cur.shape[1] // s
        kids = cur.view(batch, n_par, s, cur.shape[-1])
        total = None
        for i in range(s):
            picked = kids[:, :, i, :].index_select(2, child[:, i].contiguous())
            total = picked if total is None else total + picked
        cur = total
    return cur.reshape(batch, -1)


def flat_scores(cur, flat):
    """Sum of per-cell scores along each entry's flattened code tuple. `cur`: (B, span, K);
    `flat`: (n_entries, span) -> (B, n_entries)."""
    import torch
    B, span, _ = cur.shape
    idx = flat.t().contiguous()[None].expand(B, span, flat.shape[0])
    return cur.gather(2, idx).sum(1)
