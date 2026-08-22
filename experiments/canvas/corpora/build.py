"""Build the swatch corpus from the shader library in `canvas/shaders/`.

Every `canvas/shaders/<slug>.glsl` is one STYLE: a WebGL-1-dialect fragment shader that
renders a still image, with a `uniform float seed` that varies the *arrangement* (noise/hash
domain, layout phase) and never the palette or geometry, and a header block of human-readable
metadata. Its EXEMPLARS are renders under (seed, crop). The renderer is ours
(`canvas/render_glsl.py`, headless moderngl, runs on macOS and on Modal); nothing here depends
on anything outside this repo.

Shader file format (header lines are parsed until the first non-comment line):

    // style: weave_ribbons
    // title: Woven ribbons
    // description: One or two sentences saying what the image is, in words a person would use.
    // tags: weave, textile, diagonal
    // brief: (optional) the creative brief the author worked from
    // author: claude
    precision mediump float;
    varying vec2 uv;
    uniform float seed;
    ...

Authoring rules (what makes a style usable as a practice substrate, not just a picture):
  * still image; inputs are ONLY `uv` (0..1 across the canvas) and `seed`
  * seed -> arrangement only: offset the noise/hash domain, permute layout; palette and
    structural parameters are `const`
  * interesting everywhere in the frame: a texture / pattern / field, not a centred object,
    because swatches are crops (zoom 0.35-0.5 of the canvas) at random offsets
  * feature scale roughly 1/16..1/4 of the canvas, so a 256px crop shows parts, not one blob
    and not noise
  * loops bounded by small constants; no derivatives on noisy signals; no `time`

Usage:
    cd experiments
    python -m canvas.corpora.build --check shaders/foo.glsl [...]     # compile + 2x4 strip
    python -m canvas.corpora.build --out canvas/corpora/data/<tag> --n_per_style 64
"""

import argparse
import glob
import json
import os
import re

import numpy as np
from PIL import Image, ImageDraw

from canvas.render_glsl import render

HERE = os.path.dirname(os.path.abspath(__file__))
SHADER_DIR = os.path.normpath(os.path.join(HERE, "..", "shaders"))


def parse_header(src):
    meta = {}
    for line in src.splitlines():
        s = line.strip()
        if not s:
            continue
        if not s.startswith("//"):
            break
        m = re.match(r"//\s*([a-zA-Z_]+)\s*:\s*(.*)$", s)
        if m:
            meta[m.group(1).lower()] = m.group(2).strip()
    return meta


def load_library(shader_dir=SHADER_DIR, only=None):
    paths = sorted(glob.glob(os.path.join(shader_dir, "*.glsl")))
    if only:
        paths = [p for p in paths if os.path.basename(p) in only or p in only]
    lib = []
    for p in paths:
        src = open(p).read()
        meta = parse_header(src)
        meta.setdefault("style", os.path.splitext(os.path.basename(p))[0])
        lib.append({"path": p, "src": src, "meta": meta})
    return lib


def swatch_ok(img, min_std=6.0, max_sat_frac=0.98):
    """Reject broken or degenerate renders: NaNs, near-constant, or nearly all-saturated."""
    if not np.isfinite(img).all():
        return False
    f = img.astype(np.float32)
    if f.std() < min_std:
        return False
    sat = ((f <= 1) | (f >= 254)).all(-1).mean()
    return sat <= max_sat_frac


def check(paths, out_png=None, seeds=(0, 1, 2, 3), size=192):
    """Compile each shader and render a strip: full canvas at 4 seeds, then a 0.35 crop at
    the same 4 seeds. Prints errors; returns the list of failures."""
    fails = []
    rows = []
    for p in paths:
        src = open(p).read()
        meta = parse_header(src)
        name = meta.get("style", os.path.basename(p))
        imgs = []
        try:
            for s in seeds:
                imgs.append(render(src, size, size, uniforms={"seed": float(s)}))
            for s in seeds:
                imgs.append(render(src, size, size, uniforms={"seed": float(s)},
                                   offset=(0.3, 0.3), zoom=0.35))
        except Exception as e:
            msg = str(e)
            print(f"FAIL {name}: {msg[:800]}")
            fails.append((p, msg))
            continue
        bad = [i for i, im in enumerate(imgs) if not swatch_ok(im)]
        # seed must change the image (arrangement), but not wildly change its mean colour
        d01 = float(np.abs(imgs[0].astype(float) - imgs[1].astype(float)).mean())
        mean_shift = float(np.abs(imgs[0].reshape(-1, 3).mean(0) - imgs[1].reshape(-1, 3).mean(0)).max())
        flag = ""
        if bad:
            flag += f" DEGENERATE({bad})"
        if d01 < 2.0:
            flag += " SEED_NO_EFFECT"
        if mean_shift > 40:
            flag += " SEED_SHIFTS_PALETTE?"
        missing = [k for k in ("title", "description", "tags") if k not in meta]
        if missing:
            flag += f" MISSING_META{missing}"
        print(f"ok   {name}: seed_delta={d01:.1f} mean_shift={mean_shift:.1f}{flag}")
        rows.append((name, imgs))
    if out_png and rows:
        pad = 4
        W = 8 * (size + pad) + pad
        H = len(rows) * (size + pad) + pad
        sheet = Image.new("RGB", (W, H), (245, 245, 245))
        for r, (name, imgs) in enumerate(rows):
            for c, im in enumerate(imgs):
                pil = Image.fromarray(im)
                if c == 0:
                    d = ImageDraw.Draw(pil)
                    d.rectangle([0, 0, 6 + 6 * len(name), 13], fill=(0, 0, 0))
                    d.text((3, 1), name, fill=(255, 255, 255))
                sheet.paste(pil, (pad + c * (size + pad), pad + r * (size + pad)))
        sheet.save(out_png)
        print("wrote", out_png)
    return fails


def build(out_dir, n_per_style=64, size=256, zooms=(0.5, 0.35), seeds=8, seed=0,
          shader_dir=SHADER_DIR, only=None, sheet_cols=8):
    rng = np.random.default_rng(seed)
    lib = load_library(shader_dir, only)
    os.makedirs(out_dir, exist_ok=True)
    index = {"shader_dir": shader_dir, "n_per_style": n_per_style, "size": size,
             "zooms": list(zooms), "seeds": seeds, "kept": [], "dropped": {}}
    rows = []
    for item in lib:
        sid = item["meta"]["style"]
        src = item["src"]
        try:
            full = render(src, 256, 256, uniforms={"seed": 0.0})
        except Exception as e:
            index["dropped"][sid] = f"compile: {str(e)[:300]}"
            print(f"DROP {sid}: compile error")
            continue
        if not swatch_ok(full):
            index["dropped"][sid] = "degenerate full render"
            print(f"DROP {sid}: degenerate")
            continue
        sdir = os.path.join(out_dir, sid)
        os.makedirs(sdir, exist_ok=True)
        meta = dict(item["meta"])
        meta["source"] = os.path.relpath(item["path"], os.path.dirname(HERE))
        meta["swatches"] = []
        imgs, tries = [], 0
        while len(imgs) < n_per_style and tries < 4 * n_per_style:
            tries += 1
            z = float(rng.choice(zooms))
            off = rng.uniform(0.0, 1.0 - z, size=2)
            sd = float(rng.integers(seeds))
            img = render(src, size, size, uniforms={"seed": sd}, offset=tuple(off), zoom=z)
            if not swatch_ok(img):
                continue
            k = len(imgs)
            Image.fromarray(img).save(os.path.join(sdir, f"{k:03d}.png"))
            meta["swatches"].append({"seed": sd, "offset": [float(o) for o in off], "zoom": z})
            imgs.append(img)
        if len(imgs) < max(4, n_per_style // 2):
            index["dropped"][sid] = f"only {len(imgs)} usable swatches"
            print(f"DROP {sid}: only {len(imgs)} usable swatches")
            continue
        json.dump(meta, open(os.path.join(sdir, "style.json"), "w"), indent=1)
        index["kept"].append(sid)
        rows.append((sid, imgs[:sheet_cols]))
        print(f"{sid}: {len(imgs)} swatches")
    json.dump(index, open(os.path.join(out_dir, "index.json"), "w"), indent=1)
    if rows:
        th, pad = 128, 4
        W = sheet_cols * (th + pad) + pad
        H = len(rows) * (th + pad) + pad
        sheet = Image.new("RGB", (W, H), (245, 245, 245))
        for r, (sid, imgs) in enumerate(rows):
            for c, img in enumerate(imgs):
                pil = Image.fromarray(img).resize((th, th), Image.LANCZOS)
                if c == 0:
                    d = ImageDraw.Draw(pil)
                    d.rectangle([0, 0, 6 + 6 * len(sid), 13], fill=(0, 0, 0))
                    d.text((3, 1), sid, fill=(255, 255, 255))
                sheet.paste(pil, (pad + c * (th + pad), pad + r * (th + pad)))
        sheet.save(os.path.join(out_dir, "contact_sheet.png"))
    print(f"kept {len(index['kept'])} styles, dropped {len(index['dropped'])}")
    return index


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", nargs="*", help="shader files to compile-check (default: all)")
    ap.add_argument("--check_png", default=None, help="where to write the check strip")
    ap.add_argument("--out", default=None)
    ap.add_argument("--n_per_style", type=int, default=64)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    if a.check is not None:
        paths = a.check or sorted(glob.glob(os.path.join(SHADER_DIR, "*.glsl")))
        fails = check(paths, out_png=a.check_png)
        raise SystemExit(1 if fails else 0)
    if a.out:
        build(a.out, n_per_style=a.n_per_style, size=a.size, seeds=a.seeds, only=a.only)
