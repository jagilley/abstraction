"""Render the plant node's swatch corpus, with SEED-DISJOINT splits, and pack it for Modal.

Why seeds and not swatches carry the split. A swatch is a (seed, offset, zoom) crop of one
shader's canvas. Two crops of the SAME seed overlap in pixels with high probability at
zoom 0.35-0.5, so a swatch-level train/test split leaks. The shader's `uniform float seed`
varies the *arrangement* only (palette and geometry are `const`), so a seed-level split is
"same style, an arrangement the learner has never seen" -- which is what held-out means here.

Every split below is a disjoint set of seeds. `plant_train` also trains the CODEBOOK, so
every held-out swatch is held out from the quantizer as well as from the token model.

    split        seeds   swatches/style   what it is
    plant_train   0-31        64          codebook + plant (the masked token model)
    plant_val    32-35         8          the mask-size ladder / learning-wave readout
    plant_test   36-39         8          completion quality, damage panel, grader eval
    gA_train     40-51        24          the DRIVING grader's corpus (ladder takes prefixes)
    gA_val       52-53         4          the oracle-free q-quantile threshold slice
    gB_train     54-61        16          the REPORTING grader (a second reader, disjoint)
    gB_val       62-63         4          gB's own threshold slice

Usage (from experiments/, conda `glp`):
    python -m canvas.plant.corpus --out canvas/corpora/data/plant0 --tar
    modal volume put canvas-data canvas/corpora/data/plant0.tar corpora/plant0.tar
"""

import argparse
import json
import os
import subprocess

import numpy as np
from PIL import Image

from canvas.corpora.build import SHADER_DIR, load_library, swatch_ok
from canvas.render_glsl import render

HERE = os.path.dirname(os.path.abspath(__file__))

N_SEEDS = 64
CROPS_PER_SEED = 2

SPLITS = {
    "plant_train": (0, 32),
    "plant_val": (32, 36),
    "plant_test": (36, 40),
    "gA_train": (40, 52),
    "gA_val": (52, 54),
    "gB_train": (54, 62),
    "gB_val": (62, 64),
}


def split_of(seed_idx):
    for name, (lo, hi) in SPLITS.items():
        if lo <= seed_idx < hi:
            return name
    raise ValueError(seed_idx)


def build(out_dir, n_seeds=N_SEEDS, crops=CROPS_PER_SEED, size=256, zooms=(0.5, 0.35),
          seed=0, shader_dir=SHADER_DIR, only=None):
    rng = np.random.default_rng(seed)
    lib = load_library(shader_dir, only)
    os.makedirs(out_dir, exist_ok=True)
    man = {"shader_dir": shader_dir, "n_seeds": n_seeds, "crops_per_seed": crops,
           "size": size, "zooms": list(zooms), "splits": {k: list(v) for k, v in SPLITS.items()},
           "styles": [], "dropped": {}, "swatches": {}}
    for item in lib:
        sid = item["meta"]["style"]
        src = item["src"]
        try:
            if not swatch_ok(render(src, 256, 256, uniforms={"seed": 0.0})):
                man["dropped"][sid] = "degenerate full render"
                print(f"DROP {sid}: degenerate"); continue
        except Exception as e:
            man["dropped"][sid] = f"compile: {str(e)[:200]}"
            print(f"DROP {sid}: compile error"); continue
        sdir = os.path.join(out_dir, sid)
        os.makedirs(sdir, exist_ok=True)
        recs, bad = [], 0
        for s in range(n_seeds):
            for c in range(crops):
                img = None
                for _ in range(8):
                    z = float(rng.choice(zooms))
                    off = rng.uniform(0.0, 1.0 - z, size=2)
                    cand = render(src, size, size, uniforms={"seed": float(s)},
                                  offset=tuple(off), zoom=z)
                    if swatch_ok(cand):
                        img = cand; break
                if img is None:
                    bad += 1
                    continue
                fn = f"s{s:03d}_c{c}.png"
                Image.fromarray(img).save(os.path.join(sdir, fn))
                recs.append({"file": fn, "seed": s, "crop": c, "split": split_of(s),
                             "offset": [float(o) for o in off], "zoom": z})
        if bad:
            print(f"  {sid}: {bad} unusable draws")
        meta = dict(item["meta"])
        meta["swatches"] = recs
        json.dump(meta, open(os.path.join(sdir, "style.json"), "w"), indent=1)
        man["styles"].append(sid)
        man["swatches"][sid] = len(recs)
        print(f"{sid}: {len(recs)} swatches")
    json.dump(man, open(os.path.join(out_dir, "manifest.json"), "w"), indent=1)
    print(f"kept {len(man['styles'])} styles, dropped {len(man['dropped'])}")
    return man


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="canvas/corpora/data/plant0")
    ap.add_argument("--n_seeds", type=int, default=N_SEEDS)
    ap.add_argument("--crops", type=int, default=CROPS_PER_SEED)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--tar", action="store_true", help="tar the output dir for `modal volume put`")
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    build(out, n_seeds=a.n_seeds, crops=a.crops, size=a.size, seed=a.seed, only=a.only)
    if a.tar:
        parent, base = os.path.dirname(out), os.path.basename(out)
        tar = out + ".tar"
        subprocess.check_call(["tar", "-cf", tar, "-C", parent, base])
        print(f"wrote {tar} ({os.path.getsize(tar) / 1e6:.0f} MB)")
        print(f"modal volume put canvas-data {tar} corpora/{base}.tar")


if __name__ == "__main__":
    main()
