"""Tile-grammar swatches: the data-generating process for the `canvas` substrate.

The practice arc's third substrate (see `ideas/style_practice_substrate.md`). This file is
the DGP only — the experimenter's ground truth — in the role `one_layer_deeper/squaring_mod.py`
plays for that substrate: standalone, no torch, and with a `describe()` that measures the
margins the loop's claims rest on instead of assuming them. The agent never imports it on a
consumed path; it sees pixels (and later a VQ codebook over them).

THE PICTURE. A swatch is a hidden H x W grid of TILES rendered to pixels. Every tile has four
typed EDGES (N, E, S, W), each either closed (0) or carrying a path of colour c in 1..C, and a
set of STRANDS that connect its open edges inside the tile (a straight, a quarter-arc, a
crossing, a junction, a terminal). A tiling is VALID iff every shared edge agrees. Rendered,
a valid tiling is continuous knotwork / circuitry; any violation is a visibly broken line.

WHY THIS SHAPE, against the arc's filter (memo §2):

  * coherence is crisp and LOCAL TO CHECK but GLOBAL TO SATISFY. Edge agreement is a
    one-edge test, but a tileset that is incomplete in edge patterns (no tile joins colour A
    to colour B; a knot set has no terminals, so every cell needs an even number of open
    edges per colour) means a greedy single-tile filler dead-ends and a single-tile repair
    of a broken seam usually does not exist. That is the meter — depth unaffordable to the
    primitive action — produced by the grammar's sparsity rather than declared.
    `describe()` measures it (single-tile repairability, greedy dead-end rate, completions
    per hole) so it is a number beside every run, not an assumption.
  * a STYLE is a demand-concentration over the valid set (memo §6), literally: a `School`
    is a tileset plus weights over its tiles, neighbour affinities (so 2x2 MOTIFS recur and
    mined `T[3]` can be parts rather than adjacency-forced textures), a palette and stroke
    geometry. Many valid tilings; a school uses some of them, often. Several schools on one
    tileset = style drift on a fixed truth (`setlist`); a new tileset = truth drift.
  * the damage ladder is `corrupt_hier`'s: a level-k square region (1x1, 2x2, 4x4 tiles —
    nested) whose contents are re-sampled LEGAL and INTERNALLY CONSISTENT but inconsistent
    with the surround at the seam. Nothing inside the region is locally suspicious; the
    inconsistency lives in the parent region, and repairing it means re-tiling the region,
    not patching a tile. A second rung — `offstyle` — keeps the seam valid and re-samples
    under another demand: valid but atypical, the multimodal-right that only a style grader
    catches. A third — `mask` — blanks the region (the inpainting piece proper).
  * the oracle is held by the experimenter and withheld from the agent: `Tileset.validate`
    and the enumerable true level-2 vocabulary (`Tileset.valid_blocks`) exist so the ceiling
    §9 of the memo lists as lost is logged beside every number — and never consumed.

The level structure the loop will use: pixels 256x256 = 8x8 tiles of 32 px; a VQ codebook
with 16 px patches gives a 16x16 code grid, so a tile is a 2x2 block of codes. Quadtree
levels over codes: L1 = a code, L2 = a tile (2x2 codes), L3 = a 2x2-tile motif (4x4 codes),
L4 = a 4x4-tile quadrant, L5 = the swatch. Whether mining over the codebook recovers the tile
grid at L2 is then a readout against a known answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# --------------------------------------------------------------------------- #
# the catalogue
# --------------------------------------------------------------------------- #

N, E, S, W = 0, 1, 2, 3
OPP = (2, 3, 0, 1)            # opposite edge slot
DR = ((-1, 0), (0, 1), (1, 0), (0, -1))   # (dr, dc) for N, E, S, W

# base shapes over slots (N, E, S, W): a shape is a tuple of strands, each a tuple of slots it
# touches; colours are assigned per strand. `over`, when present, names which strand of a
# crossing is drawn on top (0 or 1) -- a render attribute that makes two tiles with identical
# edges distinct entries (a synonym pair, as RHM's `m` synonyms per feature).
SHAPES = {
    "empty":    ((),),
    "straight": (((N, S),),),
    "arc":      (((N, E),),),
    "terminal": (((N,),),),
    "tee":      (((N, E, S),),),
    "plus":     (((N, E, S, W),),),
    "cross":    (((N, S), (E, W)),),
    "double":   (((N, E), (S, W)),),
}
HAS_OVER = {"cross"}            # shapes that get an over/under variant


@dataclass(frozen=True)
class Tile:
    kind: str
    strands: tuple            # tuple of (colour, slots)
    over: int                 # index of the strand drawn on top, or -1
    edges: tuple              # (N, E, S, W) edge types: 0 closed, 1..C colours

    @property
    def n_open(self):
        return sum(1 for e in self.edges if e)


def _rotate(slots, r):
    return tuple(sorted((s + r) % 4 for s in slots))


def build_catalogue(n_colours=2):
    """Every (shape, rotation, colour assignment, over/under) tile, deduplicated by its strand
    signature. With C=2 this is 47 tiles; with C=1, 17."""
    tiles, seen = [], set()
    for kind, variants in SHAPES.items():
        for strands in variants:
            for r in range(4):
                rot = tuple(_rotate(s, r) for s in strands)
                n_str = len(rot)
                for cols in np.ndindex(*([n_colours] * n_str)) if n_str else [()]:
                    col_strands = tuple(sorted((int(c) + 1, sl) for c, sl in zip(cols, rot)))
                    overs = (0, 1) if kind in HAS_OVER else (-1,)
                    for ov in overs:
                        # `over` indexes the *sorted* strand list; keep it meaningful
                        sig = (kind, col_strands, ov)
                        if sig in seen:
                            continue
                        seen.add(sig)
                        edges = [0, 0, 0, 0]
                        for c, sl in col_strands:
                            for s in sl:
                                assert edges[s] == 0, "a slot touched twice"
                                edges[s] = c
                        tiles.append(Tile(kind, col_strands, ov, tuple(edges)))
    return tiles


# --------------------------------------------------------------------------- #
# a tileset = a truth: which tiles exist, and edge agreement
# --------------------------------------------------------------------------- #

class Tileset:
    """The grammar. `tiles` is a list of `Tile`; `compat[d][a, b]` is True iff tile `a` may
    have tile `b` as its neighbour in direction `d` (a's edge d equals b's edge OPP[d])."""

    def __init__(self, tiles, name="custom", n_colours=2):
        self.tiles = list(tiles)
        self.name = name
        self.n_colours = n_colours
        self.n = len(self.tiles)
        self.edges = np.array([t.edges for t in self.tiles], np.int64)        # (n, 4)
        self.compat = np.stack([self.edges[:, d][:, None] == self.edges[:, OPP[d]][None, :]
                                for d in range(4)])                              # (4, n, n)
        self.kinds = np.array([t.kind for t in self.tiles])

    # ---- the oracle: validity -------------------------------------------------------- #
    def violations(self, grid):
        """(bad_h, bad_v): boolean masks of broken shared edges. bad_h[r, c] is the edge
        between (r, c) and (r, c+1); bad_v[r, c] the one between (r, c) and (r+1, c).
        Masked cells (-1) never count as violations."""
        g = np.asarray(grid)
        e = self.edges
        valid = g >= 0
        gh = np.where(valid, g, 0)
        bad_h = (e[gh[:, :-1], E] != e[gh[:, 1:], W]) & valid[:, :-1] & valid[:, 1:]
        bad_v = (e[gh[:-1, :], S] != e[gh[1:, :], N]) & valid[:-1, :] & valid[1:, :]
        return bad_h, bad_v

    def validate(self, grid, border="open"):
        """(ok, n_violations). ORACLE READOUT -- logged, never consumed by the agent."""
        bad_h, bad_v = self.violations(grid)
        n = int(bad_h.sum() + bad_v.sum())
        if border == "closed":
            g = np.asarray(grid)
            e = self.edges
            n += int((e[g[0, :], N] != 0).sum() + (e[g[-1, :], S] != 0).sum()
                     + (e[g[:, 0], W] != 0).sum() + (e[g[:, -1], E] != 0).sum())
        return n == 0, n

    # ---- the grammar's sparsity ------------------------------------------------------ #
    def edge_pattern_coverage(self):
        """Fraction of the (C+1)^4 edge patterns realised by at least one tile. A complete
        set (1.0) makes every broken seam repairable by a single tile -- no depth."""
        total = (self.n_colours + 1) ** 4
        have = {tuple(e) for e in self.edges.tolist()}
        return len(have) / total, len(have), total

    def valid_blocks(self, side=2, border="open"):
        """The TRUE level-k vocabulary: every valid side x side block (open boundary),
        as (n_blocks, side*side) tile-id arrays. Experimenter-only; the matched-size random
        control and mined-table precision/recall read against it. Enumerates by DFS with
        edge propagation, so it is only meant for side <= 2 on sets of this size."""
        cells = [(r, c) for r in range(side) for c in range(side)]
        out = []
        grid = -np.ones((side, side), np.int64)

        def ok(r, c, t):
            if c > 0 and not self.compat[W, t, grid[r, c - 1]]:
                return False
            if r > 0 and not self.compat[N, t, grid[r - 1, c]]:
                return False
            if border == "closed":
                e = self.edges[t]
                if (r == 0 and e[N]) or (r == side - 1 and e[S]) or \
                   (c == 0 and e[W]) or (c == side - 1 and e[E]):
                    return False
            return True

        def rec(i):
            if i == len(cells):
                out.append(grid.reshape(-1).copy())
                return
            r, c = cells[i]
            for t in range(self.n):
                if ok(r, c, t):
                    grid[r, c] = t
                    rec(i + 1)
                    grid[r, c] = -1
        rec(0)
        return np.array(out, np.int64).reshape(-1, side * side)


def preset_tileset(name, n_colours=2):
    """Named truths. Each is a subset of the catalogue; the subsets differ in which edge
    patterns are realisable, which is what sets the depth margin (see `describe`)."""
    cat = build_catalogue(n_colours)
    keep = {
        # closed loops and crossings only: no path ends, no junctions, colours never join.
        # The knot grammar -- parity per colour in every cell.
        "knot":    {"empty", "straight", "arc", "cross", "double"},
        # circuitry: ends and junctions allowed, over/under crossings allowed
        "circuit": {"empty", "straight", "arc", "terminal", "tee", "cross"},
        # dense Truchet: no empties, no ends -- every cell carries path
        "meander": {"straight", "arc", "double", "cross"},
        # everything
        "full":    set(SHAPES),
    }[name]
    tiles = [t for t in cat if t.kind in keep]
    return Tileset(tiles, name=name, n_colours=n_colours)


# --------------------------------------------------------------------------- #
# a school = a style: demand over the valid set, plus how it is drawn
# --------------------------------------------------------------------------- #

# (background, (colour_1, colour_2), outline-or-None) -- hand-picked so a school looks like
# someone chose it. Hex RGB.
PALETTES = {
    "ink":        ("#F4EFE6", ("#1F1F1F", "#B23A48"), None),
    "lapis":      ("#0B1D3A", ("#F2C14E", "#4FB0C6"), None),
    "moss":       ("#E7ECD9", ("#2F5D50", "#A63D40"), None),
    "graphite":   ("#2B2B2B", ("#E8E8E8", "#FF7A59"), None),
    "sand":       ("#EAD7B7", ("#4A3728", "#2C6E8A"), None),
    "plum":       ("#2A1A2E", ("#E0A9C9", "#8FD6A3"), None),
    "paper":      ("#FFFFFF", ("#0D47A1", "#F57C00"), None),
    "nocturne":   ("#101418", ("#9BE1FF", "#FFD166"), None),
    "terracotta": ("#F3E5D0", ("#B5532B", "#3B6B5C"), "#2A1A12"),
    "mint":       ("#DFF5EC", ("#1B4332", "#F4A261"), None),
    "blueprint":  ("#1D3A6E", ("#FFFFFF", "#9EC9FF"), None),
    "rose":       ("#FBEFF2", ("#7A1F3D", "#2E4057"), "#FBEFF2"),
}


@dataclass
class School:
    """A style. `weights` is the demand over the tileset; `aff` (2, n, n) multiplies the
    collapse weight of a tile by its affinity with an already-placed neighbour to the
    left (aff[0][left, t]) or above (aff[1][above, t]); `motifs` are the 2x2 blocks those
    affinities were built from (for the record -- the sampler reads only `aff`)."""
    name: str
    tileset: Tileset
    weights: np.ndarray
    aff: np.ndarray
    motifs: list
    palette: tuple
    geom: dict
    border: str = "open"
    meta: dict = field(default_factory=dict)


def make_school(tileset, seed, name=None, *, palette=None, kind_prior=None,
                colour_bias=None, concentration=2.0, n_motifs=3, motif_boost=4.0,
                border="open", geom=None):
    """Sample a school on a tileset. Demand = kind prior x colour bias x Dirichlet jitter;
    motifs = `n_motifs` valid 2x2 blocks sampled under that demand, whose internal
    neighbour pairs get their affinity multiplied by `motif_boost`."""
    rng = np.random.default_rng(seed)
    ts = tileset
    n = ts.n
    kinds_here = sorted(set(ts.kinds.tolist()))
    if kind_prior is None:
        # a random but structured preference over kinds (log-normal), so schools differ in
        # density and in which constructions they favour
        kind_prior = {k: float(np.exp(rng.normal(0, 0.6))) for k in kinds_here}
        if "empty" in kind_prior:
            # an "empty" preference sets density; cap it so a school cannot be mostly blank
            kind_prior["empty"] = min(kind_prior["empty"] * float(np.exp(rng.normal(0.3, 0.6))),
                                      3.0 * float(np.median(list(kind_prior.values()))))
    if colour_bias is None:
        cb = rng.dirichlet([2.0] * ts.n_colours) * ts.n_colours
        colour_bias = {c + 1: float(cb[c]) for c in range(ts.n_colours)}
    w = np.zeros(n)
    for i, t in enumerate(ts.tiles):
        w[i] = kind_prior.get(t.kind, 1.0)
        for c, _ in t.strands:
            w[i] *= colour_bias.get(c, 1.0)
    w *= rng.dirichlet([concentration] * n) * n       # per-tile jitter: the concentration
    w = w / w.sum()
    pal_name = palette or rng.choice(sorted(PALETTES))
    if geom is None:
        geom = {
            "stroke": float(rng.uniform(0.16, 0.34)),
            "corner": str(rng.choice(["arc", "arc", "arc", "bevel", "square"])),
            "outline": bool(rng.random() < 0.35),
            "rail": bool(rng.random() < 0.25),
            "gap": float(rng.uniform(0.06, 0.14)),
            "terminal": str(rng.choice(["dot", "cap"])),
        }
    school = School(name or f"{ts.name}-{seed}", ts, w, np.ones((2, n, n)), [],
                    PALETTES[pal_name], geom, border, {"palette": pal_name,
                                                       "kind_prior": kind_prior,
                                                       "colour_bias": colour_bias,
                                                       "seed": seed})
    # motifs: valid 2x2 blocks drawn under the school's own demand
    motifs = []
    for _ in range(n_motifs):
        g = sample(school, 2, 2, rng, border="open")
        if g is not None:
            motifs.append(g)
    aff = np.ones((2, n, n))
    for g in motifs:
        for r in range(2):
            aff[0, g[r, 0], g[r, 1]] = motif_boost
        for c in range(2):
            aff[1, g[0, c], g[1, c]] = motif_boost
    school.aff = aff
    school.motifs = motifs
    return school


# --------------------------------------------------------------------------- #
# the sampler: weighted min-entropy collapse with edge propagation (WFC's simple-tiled model)
# --------------------------------------------------------------------------- #

def _propagate(ts, dom, queue):
    """Arc-consistency over edge compatibility. `dom` (H, W, n) bool, modified in place.
    Returns False on a wiped-out cell."""
    H, W_, _ = dom.shape
    while queue:
        r, c = queue.pop()
        d_rc = dom[r, c]
        if not d_rc.any():
            return False
        for d, (dr, dc) in enumerate(DR):
            rr, cc = r + dr, c + dc
            if not (0 <= rr < H and 0 <= cc < W_):
                continue
            support = ts.compat[d][d_rc].any(0)       # tiles the neighbour may still be
            new = dom[rr, cc] & support
            if not new.any():
                return False
            if new.sum() < dom[rr, cc].sum():
                dom[rr, cc] = new
                queue.append((rr, cc))
    return True


def _init_domain(ts, H, W_, border, fixed=None):
    dom = np.ones((H, W_, ts.n), bool)
    if border == "closed":
        e = ts.edges
        dom[0, :, :] &= (e[:, N] == 0)[None, :]
        dom[-1, :, :] &= (e[:, S] == 0)[None, :]
        dom[:, 0, :] &= (e[:, W] == 0)[None, :]
        dom[:, -1, :] &= (e[:, E] == 0)[None, :]
    queue = []
    if fixed is not None:
        for r in range(H):
            for c in range(W_):
                t = fixed[r, c]
                if t >= 0:
                    dom[r, c, :] = False
                    dom[r, c, t] = True
                    queue.append((r, c))
    if not _propagate(ts, dom, queue or [(r, c) for r in range(H) for c in range(W_)]):
        return None
    return dom


def sample(school, H, W_, rng, *, fixed=None, border=None, max_restarts=200,
           weights=None, aff=None):
    """Sample a valid H x W tiling under the school's demand. `fixed` (H, W) int64 with -1
    for free cells pre-collapses the rest (inpainting / damage). Returns the grid, or None
    if no valid completion was found within `max_restarts` (restarts are counted in
    `sample.restarts` for the caller's margin bookkeeping)."""
    ts = school.tileset
    border = school.border if border is None else border
    w_t = school.weights if weights is None else weights
    aff = school.aff if aff is None else aff
    n = ts.n
    restarts = 0
    for _ in range(max_restarts + 1):
        dom = _init_domain(ts, H, W_, border, fixed)
        if dom is None:
            sample.restarts = restarts
            return None                                   # the fixed cells admit nothing
        grid = -np.ones((H, W_), np.int64)
        if fixed is not None:
            grid[fixed >= 0] = fixed[fixed >= 0]
        failed = False
        while True:
            free = np.argwhere(grid < 0)
            if len(free) == 0:
                break
            # weights per free cell, with affinities from placed neighbours
            best, best_h = None, None
            for r, c in free:
                wv = w_t * dom[r, c]
                if c > 0 and grid[r, c - 1] >= 0:
                    wv = wv * aff[0, grid[r, c - 1], :]
                if c + 1 < W_ and grid[r, c + 1] >= 0:
                    wv = wv * aff[0, :, grid[r, c + 1]]
                if r > 0 and grid[r - 1, c] >= 0:
                    wv = wv * aff[1, grid[r - 1, c], :]
                if r + 1 < H and grid[r + 1, c] >= 0:
                    wv = wv * aff[1, :, grid[r + 1, c]]
                tot = wv.sum()
                if tot <= 0:
                    failed = True
                    break
                p = wv / tot
                h = -(p[p > 0] * np.log(p[p > 0])).sum() + rng.random() * 1e-3
                if best_h is None or h < best_h:
                    best, best_h, best_p = (r, c), h, p
            if failed:
                break
            r, c = best
            t = int(rng.choice(n, p=best_p))
            grid[r, c] = t
            dom[r, c, :] = False
            dom[r, c, t] = True
            if not _propagate(ts, dom, [(r, c)]):
                failed = True
                break
        if not failed:
            sample.restarts = restarts
            return grid
        restarts += 1
    sample.restarts = restarts
    return None


sample.restarts = 0


# --------------------------------------------------------------------------- #
# the depth ladder: nested square regions, and the three kinds of damage
# --------------------------------------------------------------------------- #

def level_regions(H, level):
    """Level-`level` regions of an H x H tile grid: squares of side 2**(level-1), in raster
    order, as (r0, c0, side). Level 1 = single tiles; level log2(H)+1 = the whole grid."""
    side = 2 ** (level - 1)
    return [(r, c, side) for r in range(0, H, side) for c in range(0, H, side)]


def region_mask(H, W_, region):
    r0, c0, side = region
    m = np.zeros((H, W_), bool)
    m[r0:r0 + side, c0:c0 + side] = True
    return m


def damage(grid, region, school, rng, mode="seam", *, other=None, max_tries=50):
    """Damage `grid` on `region` = (r0, c0, side).

    mode="seam":     re-sample the region as its own open-boundary world under the school's
                     demand, IGNORING the surround, and keep a draw whose seam is broken.
                     Every tile is legal and every interior edge agrees: nothing inside is
                     locally suspicious; only the parent region holds the inconsistency.
                     (`corrupt_hier`'s analog.)
    mode="offstyle": re-sample the region RESPECTING the surround under `other`'s demand
                     (default: uniform over the tileset, no affinities). Valid, atypical.
    mode="mask":     blank the region (-1). The inpainting piece proper.

    Returns (damaged_grid, info) or (None, info) when no qualifying draw was found."""
    H, W_ = grid.shape
    ts = school.tileset
    m = region_mask(H, W_, region)
    r0, c0, side = region
    if mode == "mask":
        out = grid.copy()
        out[m] = -1
        return out, {"mode": mode}
    if mode == "seam":
        for _ in range(max_tries):
            sub = sample(school, side, side, rng, border="open")
            if sub is None:
                continue
            out = grid.copy()
            out[r0:r0 + side, c0:c0 + side] = sub
            if np.array_equal(out, grid):
                continue
            ok, nviol = ts.validate(out, school.border)
            if not ok:
                return out, {"mode": mode, "n_viol": nviol}
        return None, {"mode": mode}
    if mode == "offstyle":
        fixed = grid.copy()
        fixed[m] = -1
        w = None if other is None else other.weights
        a = np.ones_like(school.aff) if other is None else other.aff
        if other is None:
            w = np.ones(ts.n) / ts.n
        for _ in range(max_tries):
            out = sample(school, H, W_, rng, fixed=fixed, weights=w, aff=a)
            if out is None:
                return None, {"mode": mode}
            if not np.array_equal(out, grid):
                return out, {"mode": mode}
        return None, {"mode": mode}
    raise ValueError(mode)


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #

def _hex(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _strand_points(slots, corner, T):
    """Polyline (in tile-local coords, tile side T) for one strand. Edge midpoints:
    N=(T/2,0) E=(T,T/2) S=(T/2,T) W=(0,T/2)."""
    mid = {N: (T / 2, 0.0), E: (T, T / 2), S: (T / 2, T), W: (0.0, T / 2)}
    ctr = (T / 2, T / 2)
    k = len(slots)
    if k == 1:                                   # terminal: edge to centre
        return [[mid[slots[0]], ctr]]
    if k == 2:
        a, b = slots
        if (a - b) % 4 == 2:                     # straight
            return [[mid[a], mid[b]]]
        # adjacent edges: quarter-arc around their shared corner
        if corner == "bevel":
            return [[mid[a], mid[b]]]
        if corner == "square":
            return [[mid[a], ctr, mid[b]]]
        # arc: centre of the circle is the corner shared by edges a and b
        corners = {frozenset((N, E)): (T, 0.0), frozenset((E, S)): (T, T),
                   frozenset((S, W)): (0.0, T), frozenset((W, N)): (0.0, 0.0)}
        cx, cy = corners[frozenset((a, b))]
        pa, pb = mid[a], mid[b]
        th_a = np.arctan2(pa[1] - cy, pa[0] - cx)
        th_b = np.arctan2(pb[1] - cy, pb[0] - cx)
        d = (th_b - th_a + np.pi) % (2 * np.pi) - np.pi
        ths = th_a + d * np.linspace(0, 1, 13)
        return [[(cx + T / 2 * np.cos(t), cy + T / 2 * np.sin(t)) for t in ths]]
    # 3 or 4 slots: spokes to the centre
    return [[mid[s], ctr] for s in slots]


def render(grid, school, tile_px=32, ss=4, show_violations=False):
    """Render a tile grid to an RGB uint8 array (H*tile_px, W*tile_px, 3). Masked cells
    (-1) are hatched. With `show_violations`, broken edges are marked (oracle overlay --
    for figures only).

    Strokes are drawn in whole-image passes (every outline, then every fill, then every
    rail) with FLAT caps at tile edges, so a path continues across a seam without a
    visible joint; only crossings are then redrawn tile-locally to carve the over/under
    gap. Round caps are used only where strands end inside a tile (terminals, junction
    centres)."""
    from PIL import Image, ImageDraw
    ts = school.tileset
    bg, cols, outline = school.palette
    bg = _hex(bg)
    cols = [_hex(c) for c in cols]
    outline = _hex(outline) if outline else None
    geom = school.geom
    H, W_ = grid.shape
    T = tile_px * ss
    img = Image.new("RGB", (W_ * T, H * T), bg)
    dr = ImageDraw.Draw(img)
    sw = max(2, int(round(geom["stroke"] * T)))
    gap = max(1, int(round(geom["gap"] * T)))
    ow = max(1, int(round(sw * 0.35)))
    rail = max(1, int(round(sw * 0.4)))
    ctr_cap = {T / 2}                      # an endpoint at the tile centre gets a round cap

    def draw_poly(pts, colour, width):
        dr.line(pts, fill=colour, width=width, joint="curve")
        for (x, y) in (pts[0], pts[-1]):
            # round cap only at interior endpoints (the tile centre); edge endpoints stay flat
            lx, ly = x % T, y % T
            if abs(lx - T / 2) < 1e-6 and abs(ly - T / 2) < 1e-6:
                rr = width / 2
                dr.ellipse([x - rr, y - rr, x + rr, y + rr], fill=colour)

    # gather every strand polyline, in image coordinates
    strands = []                            # (tile r, c, strand idx, colour, slots, polys)
    for r in range(H):
        for c in range(W_):
            t = int(grid[r, c])
            ox, oy = c * T, r * T
            if t < 0:
                step = T // 6
                hatch = tuple(int(0.5 * a + 0.5 * b) for a, b in zip(bg, (128, 128, 128)))
                for k in range(-T, T, step):
                    dr.line([(ox + k, oy + T), (ox + k + T, oy)], fill=hatch, width=max(1, ss))
                continue
            tile = ts.tiles[t]
            for si, (colour_i, slots) in enumerate(tile.strands):
                polys = [[(ox + x, oy + y) for (x, y) in p]
                         for p in _strand_points(slots, geom["corner"], T)]
                strands.append((r, c, si, colour_i, slots, polys))

    def paint(items, with_outline=True):
        if outline is not None and with_outline:
            for (_, _, _, ci, slots, polys) in items:
                for p in polys:
                    draw_poly(p, outline, sw + 2 * ow)
        for (_, _, _, ci, slots, polys) in items:
            for p in polys:
                draw_poly(p, cols[ci - 1], sw)
        if geom["rail"]:
            for (_, _, _, ci, slots, polys) in items:
                for p in polys:
                    draw_poly(p, bg, rail)
        for (r, c, _, ci, slots, polys) in items:
            if len(slots) == 1:
                cx, cy = c * T + T / 2, r * T + T / 2
                rr = sw * (0.9 if geom["terminal"] == "dot" else 0.5)
                if outline is not None:
                    dr.ellipse([cx - rr - ow, cy - rr - ow, cx + rr + ow, cy + rr + ow], fill=outline)
                dr.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], fill=cols[ci - 1])

    paint(strands)
    # crossings: redraw tile-locally -- under strand, a background gap along the over strand
    # (flat caps, so it ends flush with the tile edge), then the over strand on top
    for r in range(H):
        for c in range(W_):
            t = int(grid[r, c])
            if t < 0 or ts.tiles[t].over < 0:
                continue
            mine = [s for s in strands if s[0] == r and s[1] == c]
            under = [s for s in mine if s[2] != ts.tiles[t].over]
            over = [s for s in mine if s[2] == ts.tiles[t].over]
            paint(under)
            for (_, _, _, ci, slots, polys) in over:
                for p in polys:
                    dr.line(p, fill=bg, width=sw + 2 * gap + (2 * ow if outline else 0), joint="curve")
            paint(over)

    if show_violations:
        bad_h, bad_v = ts.violations(grid)
        red = (255, 40, 40)
        for r, c in np.argwhere(bad_h):
            x, y = (c + 1) * T, r * T
            dr.rectangle([x - ss * 2, y + T * 0.3, x + ss * 2, y + T * 0.7], fill=red)
        for r, c in np.argwhere(bad_v):
            x, y = c * T, (r + 1) * T
            dr.rectangle([x + T * 0.3, y - ss * 2, x + T * 0.7, y + ss * 2], fill=red)
    img = img.resize((W_ * tile_px, H * tile_px), Image.LANCZOS)
    return np.asarray(img)


# --------------------------------------------------------------------------- #
# describe: the margins, measured
# --------------------------------------------------------------------------- #

def single_tile_repairable(ts, grid, region, border="open"):
    """Can the seam damage on `region` be undone by changing ONE tile of the region?
    The exact question `units.py`'s warning asks: if yes, abstraction never has to matter."""
    r0, c0, side = region
    H, W_ = grid.shape
    for r in range(r0, r0 + side):
        for c in range(c0, c0 + side):
            for t in range(ts.n):
                if t == grid[r, c]:
                    continue
                g = grid.copy()
                g[r, c] = t
                if ts.validate(g, border)[0]:
                    return True
    return False


def greedy_fill(ts, grid, region, rng, weights=None):
    """A level-1-only filler: raster order over the region, each cell a random tile
    consistent with its already-placed neighbours (N and W, plus fixed E/S), no lookahead.
    Returns True if it completes validly, False if it dead-ends."""
    r0, c0, side = region
    H, W_ = grid.shape
    g = grid.copy()
    m = region_mask(H, W_, region)
    g[m] = -1
    w = np.ones(ts.n) if weights is None else weights
    for r in range(r0, r0 + side):
        for c in range(c0, c0 + side):
            ok = np.ones(ts.n, bool)
            for d, (dr_, dc_) in enumerate(DR):
                rr, cc = r + dr_, c + dc_
                if 0 <= rr < H and 0 <= cc < W_ and g[rr, cc] >= 0:
                    ok &= ts.compat[d][:, g[rr, cc]]
            if not ok.any():
                return False
            p = w * ok
            g[r, c] = rng.choice(ts.n, p=p / p.sum())
    return True


def count_completions(ts, grid, region, weights=None):
    """Exact number of valid completions of `region` given its surround (open image border),
    and the demand-weighted effective number exp(H) under `weights`. Enumerates by DFS;
    meant for side <= 2."""
    r0, c0, side = region
    H, W_ = grid.shape
    fixed = grid.copy()
    fixed[region_mask(H, W_, region)] = -1
    cells = [(r, c) for r in range(r0, r0 + side) for c in range(c0, c0 + side)]
    w = np.ones(ts.n) if weights is None else weights
    g = fixed.copy()
    total, wsum, wlogs = 0, 0.0, []

    def ok(r, c, t):
        for d, (dr_, dc_) in enumerate(DR):
            rr, cc = r + dr_, c + dc_
            if 0 <= rr < H and 0 <= cc < W_ and g[rr, cc] >= 0 and not ts.compat[d][t, g[rr, cc]]:
                return False
        return True

    def rec(i, lw):
        nonlocal total
        if i == len(cells):
            total += 1
            wlogs.append(lw)
            return
        r, c = cells[i]
        for t in range(ts.n):
            if ok(r, c, t):
                g[r, c] = t
                rec(i + 1, lw + np.log(w[t] + 1e-300))
                g[r, c] = -1
    rec(0, 0.0)
    if total == 0:
        return 0, 0.0
    lw = np.array(wlogs)
    p = np.exp(lw - lw.max())
    p /= p.sum()
    ent = -(p * np.log(p + 1e-300)).sum()
    return total, float(np.exp(ent))


def describe(school, H=8, n_samples=24, seed=0, levels=(1, 2, 3)):
    """Measure the margins the loop's claims rest on, for one school on an H x H grid:

      coverage            fraction of edge patterns the tileset realises (1.0 = no depth)
      restarts/sample     how constrained the grammar is for the sampler
      per level k:
        repair1           fraction of seam damage undoable by ONE tile (should be small)
        greedy_ok         fraction of level-1 raster fills that complete (dead-end rate = 1-)
        viol/damage       mean broken edges per seam damage
      level 2 only:
        completions       valid 2x2 completions given the surround (seam information)
        eff_completions   demand-weighted effective number (what the style leaves open)
      true_T2             size of the true level-2 vocabulary (open 2x2 blocks)
    """
    rng = np.random.default_rng(seed)
    ts = school.tileset
    out = {"school": school.name, "tileset": ts.name, "n_tiles": ts.n,
           "coverage": ts.edge_pattern_coverage()[0]}
    grids, restarts = [], 0
    for _ in range(n_samples):
        g = sample(school, H, H, rng)
        restarts += sample.restarts
        if g is not None:
            grids.append(g)
    out["restarts_per_sample"] = restarts / max(1, n_samples)
    out["sample_ok"] = len(grids) / n_samples
    for k in levels:
        regs = level_regions(H, k)
        rep, gok, viol, nd = 0, 0, 0.0, 0
        comp, eff = [], []
        for g in grids:
            reg = regs[rng.integers(len(regs))]
            dmg, info = damage(g, reg, school, rng, "seam")
            if dmg is None:
                continue
            nd += 1
            viol += info["n_viol"]
            rep += single_tile_repairable(ts, dmg, reg, school.border)
            gok += greedy_fill(ts, g, reg, rng)
            if k == 2:
                c, e = count_completions(ts, g, reg, school.weights)
                comp.append(c)
                eff.append(e)
        out[f"L{k}"] = {"repair1": rep / max(1, nd), "greedy_ok": gok / max(1, nd),
                        "viol_per_damage": viol / max(1, nd), "n": nd}
        if k == 2 and comp:
            out["L2"]["completions_median"] = float(np.median(comp))
            out["L2"]["eff_completions_median"] = float(np.median(eff))
    out["true_T2"] = int(ts.valid_blocks(2).shape[0])
    return out


if __name__ == "__main__":
    import json
    for name in ("knot", "circuit", "meander", "full"):
        ts = preset_tileset(name)
        sc = make_school(ts, seed=1)
        print(json.dumps(describe(sc, n_samples=12), indent=1))
