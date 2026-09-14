"""The tiles twin's corpus: the SAME schools rendered ALIGNED and MISALIGNED.

WHY THIS NODE EXISTS. `pl0` measured three preconditions on the GLSL library and all three
came back negative in the same direction (recurrence tracks code entropy not parts; search
past one forward buys nothing; the mean-NLL grader prefers the plant's bland fill to the
truth). One candidate common cause is that the 16-px patch alphabet is PHASE-MISALIGNED with
the program: swatches are crops at random pixel offsets, so the same motif spells differently
every time, and the styles that recurred most were the translation-invariant ones (stripes,
strata). `canvas/tiles.py` is the hidden-grid DGP kept as the calibration twin for exactly
this, and it has an exact oracle.

THE ONE VARIABLE. A swatch is a 256x256 crop of a 320x320 render of a 10x10 tile grid
(32 px per tile). The two conditions share the SAME schools, the SAME tilings, the SAME
renders, the SAME seed bands and the SAME split; they differ only in where the crop starts:

  aligned     offset is a multiple of 32 px -> the 16-px code grid NESTS in the tile grid.
              A tile is exactly a 2x2 block of codes; a 2x2-tile motif is exactly 4x4.
              This is `tiles.py`'s docstring level structure, made literal.
  misaligned  offset is drawn so that (offset mod 16) lands in [4, 12] on BOTH axes -> no
              code cell coincides with a tile, and every tile straddles four of them.
              This is the condition `lib0` was in, by accident.

A SCHOOL IS A STYLE (memo Sec.6, literally): 4 preset tilesets x 9 schools = 36 styles, the
same count as `lib0`, so the plant's capacity and the grader's corpus ladder are unchanged.
Nine schools on one tileset is style drift on a fixed truth (`setlist`); a different tileset
is truth drift (`transpose`). Both are here for later, and neither is used in this node.

THE ORACLE, WRITTEN DOWN BUT NEVER FED TO THE AGENT. Alongside each swatch: the tile grid it
came from, the crop offset, and (aligned only) the exact 8x8 tile ids of the window. Alongside
each school: its tileset name and a 32-px render of every tile in the catalogue, which is what
lets a decoded completion be classified back to tiles and run through `Tileset.validate`.

THE DAMAGE PANEL, with known validity -- what the GLSL library could not give. For every
`plant_test` tiling, at each of the three nested levels (1x1, 2x2, 4x4 tiles = 2x2, 4x4, 8x8
codes when aligned), `tiles.damage` produces:
    seam      re-sampled ignoring the surround: every tile legal, every interior edge agreeing,
              the inconsistency living only in the parent region. INVALID by construction and
              locally unsuspicious.
    offstyle  re-sampled respecting the surround under a uniform demand: VALID but atypical.
              The exact `alt_valid` class `critic/` had on RHM and `lib0` could only guess at.
Both are rendered from the same 320x320 pipeline and cropped with the SAME two offsets, so the
panel is paired across conditions.

Usage (from experiments/, conda `glp`):
    python -m canvas.plant.tiles_twin.corpus --out canvas/corpora/data --tar
    modal volume put canvas-data canvas/corpora/data/tiles_aligned.tar corpora/tiles_aligned.tar
    modal volume put canvas-data canvas/corpora/data/tiles_misaligned.tar corpora/tiles_misaligned.tar
"""

import argparse
import json
import os
import subprocess

import numpy as np
from PIL import Image

from canvas import tiles as T
from canvas.plant.corpus import SPLITS, split_of

TILESETS = ("knot", "circuit", "meander", "full")
SCHOOLS_PER_TILESET = 9
GRID = 10                 # tiles per side of the rendered world
TILE_PX = 32
WIN = 8                   # tiles per side of a 256-px window
SIZE = 256
N_SEEDS = 64
CROPS = 2
LEVELS = (1, 2, 3)
DMG_MODES = ("seam", "offstyle")
CONDS = ("aligned", "misaligned")


def schools():
    """36 styles: 4 tilesets x 9 schools. Deterministic in the tileset name and index."""
    out = []
    for ti, name in enumerate(TILESETS):
        ts = T.preset_tileset(name)
        for j in range(SCHOOLS_PER_TILESET):
            sc = T.make_school(ts, seed=1000 * ti + j, name=f"{name}_{j}")
            out.append(sc)
    return out


def tile_atlas(sc):
    """Every tile of the school's tileset rendered alone at TILE_PX, for classifying a
    decoded completion back to tiles. Oracle-side only."""
    n = sc.tileset.n
    a = np.zeros((n, TILE_PX, TILE_PX, 3), np.uint8)
    for t in range(n):
        a[t] = T.render(np.array([[t]], np.int64), sc, tile_px=TILE_PX, ss=4)
    return a


def _offsets(rng):
    """One aligned and one misaligned pixel offset for the same render."""
    a = (int(rng.integers(0, GRID - WIN + 1)), int(rng.integers(0, GRID - WIN + 1)))
    while True:
        oy = int(rng.integers(0, (GRID - WIN) * TILE_PX + 1))
        ox = int(rng.integers(0, (GRID - WIN) * TILE_PX + 1))
        if 4 <= oy % 16 <= 12 and 4 <= ox % 16 <= 12:
            return (a[0] * TILE_PX, a[1] * TILE_PX), (oy, ox), a


def _crop(img, off):
    return img[off[0]:off[0] + SIZE, off[1]:off[1] + SIZE]


def code_rect(region, off):
    """The code-grid rectangle a tile region occupies inside a crop at pixel offset `off`.
    Aligned: exactly (2*side)^2 cells. Misaligned: the COVERING rectangle, which is what a
    mask over the region has to be."""
    r0, c0, side = region
    y0, y1 = r0 * TILE_PX - off[0], (r0 + side) * TILE_PX - off[0]
    x0, x1 = c0 * TILE_PX - off[1], (c0 + side) * TILE_PX - off[1]
    return (y0 // 16, x0 // 16, -(-y1 // 16), -(-x1 // 16))


def build(out_root, n_seeds=N_SEEDS, crops=CROPS, seed=0, only=None):
    scs = schools()
    if only:
        scs = [s for s in scs if s.name in only]
    dirs = {c: os.path.join(out_root, f"tiles_{c}") for c in CONDS}
    man = {c: {"grid": GRID, "tile_px": TILE_PX, "size": SIZE, "n_seeds": n_seeds,
               "crops_per_seed": crops, "condition": c,
               "splits": {k: list(v) for k, v in SPLITS.items()},
               "styles": [], "dropped": {}, "swatches": {}} for c in CONDS}
    for c in CONDS:
        os.makedirs(dirs[c], exist_ok=True)

    for sc in scs:
        sid = sc.name
        ts = sc.tileset
        for c in CONDS:
            os.makedirs(os.path.join(dirs[c], sid), exist_ok=True)
            os.makedirs(os.path.join(dirs[c], sid, "dmg"), exist_ok=True)
        recs = {c: [] for c in CONDS}
        grids, offs = [], []
        dmg_recs = {c: [] for c in CONDS}
        rng = np.random.default_rng(abs(hash(sid)) % (2 ** 32) + seed)
        for s in range(n_seeds):
            g = T.sample(sc, GRID, GRID, rng)
            if g is None:
                continue
            img = T.render(g, sc, tile_px=TILE_PX, ss=4)
            sp = split_of(s)
            for cidx in range(crops):
                offA, offM, tileoff = _offsets(rng)
                for c, off in (("aligned", offA), ("misaligned", offM)):
                    fn = f"s{s:03d}_c{cidx}.png"
                    Image.fromarray(_crop(img, off)).save(os.path.join(dirs[c], sid, fn))
                    rec = {"file": fn, "seed": s, "crop": cidx, "split": sp,
                           "offset_px": list(off), "tile_offset": list(tileoff)}
                    if c == "aligned":
                        rec["tiles"] = g[tileoff[0]:tileoff[0] + WIN,
                                         tileoff[1]:tileoff[1] + WIN].tolist()
                    recs[c].append(rec)
                grids.append(g); offs.append((offA, offM, tileoff, s, cidx))

            # ---- the damage panel, on plant_test tilings only ----
            if sp != "plant_test":
                continue
            offA, offM, tileoff = offs[-1][0], offs[-1][1], offs[-1][2]
            for lvl in LEVELS:
                side = 2 ** (lvl - 1)
                r0 = int(rng.integers(2, WIN - side + 1))
                c0 = int(rng.integers(2, WIN - side + 1))
                region = (r0, c0, side)
                for mode in DMG_MODES:
                    dg, info = T.damage(g, region, sc, rng, mode)
                    if dg is None:
                        continue
                    dimg = T.render(dg, sc, tile_px=TILE_PX, ss=4)
                    ok, nv = ts.validate(dg, border=sc.border)
                    key = f"s{s:03d}_l{lvl}_{mode}"
                    for c, off in (("aligned", offA), ("misaligned", offM)):
                        Image.fromarray(_crop(dimg, off)).save(
                            os.path.join(dirs[c], sid, "dmg", key + ".png"))
                        Image.fromarray(_crop(img, off)).save(
                            os.path.join(dirs[c], sid, "dmg", f"s{s:03d}_l{lvl}_clean.png"))
                        dmg_recs[c].append(
                            {"key": key, "seed": s, "level": lvl, "mode": mode,
                             "region": list(region), "offset_px": list(off),
                             "code_rect": list(code_rect(region, off)),
                             "valid": bool(ok), "n_viol": int(nv),
                             "clean_file": f"s{s:03d}_l{lvl}_clean.png"})
        atlas = tile_atlas(sc)
        for c in CONDS:
            meta = {"style": sid, "tileset": ts.name, "n_tiles": ts.n,
                    "border": sc.border, "condition": c,
                    "coverage": ts.edge_pattern_coverage()[0],
                    "swatches": recs[c], "damage": dmg_recs[c]}
            json.dump(meta, open(os.path.join(dirs[c], sid, "style.json"), "w"))
            np.savez_compressed(os.path.join(dirs[c], sid, "oracle.npz"),
                                atlas=atlas, edges=ts.edges.astype(np.int64),
                                weights=sc.weights.astype(np.float32))
            man[c]["styles"].append(sid)
            man[c]["swatches"][sid] = len(recs[c])
        print(f"{sid}: {len(recs['aligned'])} swatches, {len(dmg_recs['aligned'])} damaged",
              flush=True)
    for c in CONDS:
        json.dump(man[c], open(os.path.join(dirs[c], "manifest.json"), "w"), indent=1)
    print(f"built {len(man['aligned']['styles'])} schools in both conditions")
    return man


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="canvas/corpora/data")
    ap.add_argument("--n_seeds", type=int, default=N_SEEDS)
    ap.add_argument("--crops", type=int, default=CROPS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--tar", action="store_true")
    a = ap.parse_args()
    root = os.path.abspath(a.out)
    build(root, n_seeds=a.n_seeds, crops=a.crops, seed=a.seed, only=a.only)
    if a.tar:
        for c in CONDS:
            d = os.path.join(root, f"tiles_{c}")
            tar = d + ".tar"
            subprocess.check_call(["tar", "-cf", tar, "-C", root, f"tiles_{c}"])
            print(f"wrote {tar} ({os.path.getsize(tar)/1e6:.0f} MB)")


if __name__ == "__main__":
    main()
