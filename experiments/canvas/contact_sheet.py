"""Contact sheets for the tile-grammar DGP: what the swatches look like, school by school,
and what the three damage modes look like beside a clean exemplar.

    cd experiments && python -m canvas.contact_sheet --out /path/to/dir [--tileset knot]

Writes `schools_<tileset>.png` (rows = schools, columns = exemplars) and
`damage_<tileset>.png` (clean | seam L2 | seam L3 | off-style L2 | mask L2, violations
marked in red on the seam columns -- an oracle overlay, for looking at only).
"""

import argparse
import os

import numpy as np
from PIL import Image, ImageDraw

from canvas import tiles as T


def _label(img, text):
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 6 + 6 * len(text), 13], fill=(0, 0, 0))
    d.text((3, 1), text, fill=(255, 255, 255))
    return img


def sheet(tiles_list, cols, pad=6, labels=None, bg=(245, 245, 245)):
    h, w = tiles_list[0].shape[:2]
    rows = (len(tiles_list) + cols - 1) // cols
    out = Image.new("RGB", (cols * (w + pad) + pad, rows * (h + pad) + pad), bg)
    for i, arr in enumerate(tiles_list):
        im = Image.fromarray(arr)
        if labels is not None and labels[i]:
            im = _label(im, labels[i])
        r, c = divmod(i, cols)
        out.paste(im, (pad + c * (w + pad), pad + r * (h + pad)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tileset", default="knot")
    ap.add_argument("--n_schools", type=int, default=6)
    ap.add_argument("--n_ex", type=int, default=5)
    ap.add_argument("--H", type=int, default=8)
    ap.add_argument("--tile_px", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--border", default="open")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ts = T.preset_tileset(a.tileset)
    rng = np.random.default_rng(a.seed)
    imgs, labels = [], []
    schools = []
    for k in range(a.n_schools):
        sc = T.make_school(ts, seed=a.seed * 100 + k, border=a.border)
        schools.append(sc)
        for j in range(a.n_ex):
            g = T.sample(sc, a.H, a.H, rng)
            imgs.append(T.render(g, sc, tile_px=a.tile_px))
            labels.append(f"{sc.name} {sc.meta['palette']} {sc.geom['corner']}" if j == 0 else "")
    sheet(imgs, a.n_ex, labels=labels).save(os.path.join(a.out, f"schools_{a.tileset}.png"))

    # damage panel on the first two schools
    panel, labels = [], []
    for sc in schools[:3]:
        g = T.sample(sc, a.H, a.H, rng)
        panel.append(T.render(g, sc, tile_px=a.tile_px)); labels.append("clean")
        reg2 = (a.H // 2 - 1, a.H // 2 - 1, 2)
        reg3 = (a.H // 2 - 2, a.H // 2 - 2, 4)
        d, _ = T.damage(g, reg2, sc, rng, "seam")
        panel.append(T.render(d, sc, tile_px=a.tile_px, show_violations=True)); labels.append("seam L2")
        d, _ = T.damage(g, reg3, sc, rng, "seam")
        panel.append(T.render(d, sc, tile_px=a.tile_px, show_violations=True)); labels.append("seam L3")
        d, _ = T.damage(g, reg2, sc, rng, "offstyle")
        panel.append(T.render(d if d is not None else g, sc, tile_px=a.tile_px)); labels.append("off-style L2")
        d, _ = T.damage(g, reg3, sc, rng, "offstyle")
        panel.append(T.render(d if d is not None else g, sc, tile_px=a.tile_px)); labels.append("off-style L3")
        d, _ = T.damage(g, reg3, sc, rng, "mask")
        panel.append(T.render(d, sc, tile_px=a.tile_px)); labels.append("mask L3")
    sheet(panel, 6, labels=labels).save(os.path.join(a.out, f"damage_{a.tileset}.png"))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
