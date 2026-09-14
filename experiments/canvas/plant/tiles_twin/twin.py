"""The tiles twin's ORACLE columns: everything `lib0` could not report, logged beside the
learned numbers and never fed to the agent.

The five core readouts (quantizer floor, mask-size ladder, recurrence, cost-to-depth, grader
calibration) are produced by `canvas.plant.plant` UNCHANGED, once per condition -- that is the
point of the twin, so the only difference between `lib0`, `tiles_aligned` and
`tiles_misaligned` is the corpus. This module adds what the tile grammar makes possible:

1. RECURRENCE AGAINST A KNOWN ANSWER. In the aligned condition a tile IS a 2x2 code block, so
   mined `T[2]` can be checked against the school's tile catalogue directly: for every tile
   position we hold both the true tile id and the code block that landed on it. Two purities
   say whether the codebook resolved the grammar's own unit:
       block->tile   of the occurrences of one code block, what fraction share a tile id
       tile->block   of the occurrences of one tile, what fraction share a code block
   A codebook that recovered the tile grid scores ~1 on both. `T[3]` is the same question one
   level up, over 2x2-tile motifs. In the misaligned condition no code block coincides with a
   tile, so these are the control and are reported as the same numbers computed on the
   nearest-aligned block (they should collapse).

2. GRADER CALIBRATION AGAINST KNOWN VALIDITY. `tiles.damage` gives, per nested level:
       seam      INVALID by construction, and locally unsuspicious -- every tile legal, every
                 interior edge agreeing, the inconsistency only in the parent region.
       offstyle  VALID but atypical: the exact `alt_valid` class.
   A grader that is doing what §5 asks should reject `seam` and pass `offstyle`. `lib0` could
   only guess which of `shuffle`/`marginal` was which; here it is known.

3. THE PLANT'S OWN FILLS, VALIDATED. A completion is decoded to pixels, each 2x2 code block
   classified to the nearest rendered tile in the school's atlas, and the resulting tile grid
   run through the edge-agreement test. That is the direct gaming readout: whether what the
   grader passes at ~1.00 is actually in the grammar.

Nothing here is imported by `plant.py`; the oracle lives in this file alone.
"""

import json
import os
import time

import numpy as np

from canvas.shared import DATA_DIR, NumpyEncoder, app, volume
from canvas.plant.plant import PANEL_SIDES, ROOT, DEV, REMOTE, _commit

N, E, S, W = 0, 1, 2, 3
LEVELS = (1, 2, 3)


# --------------------------------------------------------------------------- #
# oracle primitives (edge agreement, straight from the tileset's edge table)
# --------------------------------------------------------------------------- #

def n_violations(grid, edges):
    """Broken shared edges in a tile grid. `tiles.Tileset.violations`, without the object."""
    g = np.asarray(grid)
    ok = g >= 0
    gh = np.where(ok, g, 0)
    bh = (edges[gh[:, :-1], E] != edges[gh[:, 1:], W]) & ok[:, :-1] & ok[:, 1:]
    bv = (edges[gh[:-1, :], S] != edges[gh[1:, :], N]) & ok[:-1, :] & ok[1:, :]
    return int(bh.sum() + bv.sum())


def classify_tiles(px, atlas):
    """(H, W, 3) float image in [0,1] -> (H/32, W/32) nearest-atlas-tile ids and mean distance."""
    T = atlas.shape[1]
    h, w = px.shape[0] // T, px.shape[1] // T
    blocks = px[:h * T, :w * T].reshape(h, T, w, T, 3).transpose(0, 2, 1, 3, 4)
    b = blocks.reshape(h * w, -1)
    a = (atlas.astype(np.float32) / 255.0).reshape(atlas.shape[0], -1)
    d = ((b ** 2).sum(1)[:, None] - 2 * b @ a.T + (a ** 2).sum(1)[None])
    ids = d.argmin(1)
    return ids.reshape(h, w), float(np.sqrt(np.maximum(d.min(1), 0)).mean() / np.sqrt(b.shape[1]))


def purity(a, b):
    """Mean over distinct values of `a` of the largest share a single `b` value takes."""
    out, wts = [], []
    for v in np.unique(a):
        sel = b[a == v]
        c = np.bincount(sel)
        out.append(c.max() / len(sel)); wts.append(len(sel))
    return float(np.average(out, weights=wts)), int(len(out))


# --------------------------------------------------------------------------- #

def _decode_px(codes, cents, grid=16, patch=16):
    """(n, 256) codes -> (n, 256, 256, 3) float image, through the codebook lookup."""
    p = cents[codes]                                    # (n, 256, 768)
    n = p.shape[0]
    x = p.reshape(n, grid, grid, patch, patch, 3).transpose(0, 1, 3, 2, 4, 5)
    return x.reshape(n, grid * patch, grid * patch, 3)


@app.function(volumes={DATA_DIR: volume}, timeout=3600, memory=16384)
def unpack_tiles(tar: str = "corpora/tiles_aligned.tar"):
    import tarfile
    with tarfile.open(os.path.join(DATA_DIR, tar)) as t:
        t.extractall(os.path.join(DATA_DIR, "corpora"))
    _commit()
    root = os.path.join(DATA_DIR, "corpora", os.path.basename(tar)[:-4])
    man = json.load(open(os.path.join(root, "manifest.json")))
    print(f"unpacked {len(man['styles'])} schools ({man['condition']}) to {root}")
    return root


@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=14400, memory=65536)
def oracle_run(tag: str = "tw_aligned", qtag: str = "twq_aligned", corpus: str = "tiles_aligned",
               k: int = 512, seed: int = 0, d: int = 256, layers: int = 6, heads: int = 8,
               dropout: float = 0.1, graders: str = "gA24,gB", support: int = 8,
               q: float = 0.90, q_lo: float = 0.02, n_orders: int = 2, n_steps: int = 4):
    import torch
    from PIL import Image
    from canvas.plant import codebook as CB, grader as GR, model as MD
    from canvas.plant.sampler import decode

    torch.backends.cuda.matmul.allow_tf32 = True
    t0 = time.time()
    croot = os.path.join(ROOT, "corpora", corpus)
    qdir = os.path.join(ROOT, REMOTE, qtag)
    out = os.path.join(ROOT, REMOTE, tag)
    idx = json.load(open(os.path.join(qdir, "index.json")))
    styles = idx["styles"]; n_style = len(styles)
    sidx = np.array(idx["sidx"], np.int64); split = np.array(idx["split"])
    files = idx["files"]
    z = np.load(os.path.join(qdir, f"quant_K{k}.npz"))
    codes = z["codes"].astype(np.int64); cents = z["centroids"].astype(np.float32)
    aligned = "misaligned" not in corpus
    R = {"tag": tag, "corpus": corpus, "aligned": bool(aligned), "K": k, "styles": styles}

    meta = {s: json.load(open(os.path.join(croot, s, "style.json"))) for s in styles}
    orc = {s: np.load(os.path.join(croot, s, "oracle.npz")) for s in styles}
    R["tilesets"] = {s: meta[s]["tileset"] for s in styles}
    R["n_tiles"] = {s: meta[s]["n_tiles"] for s in styles}
    R["coverage"] = {s: meta[s]["coverage"] for s in styles}

    # ---- 1. recurrence against the tile catalogue -------------------------------------- #
    by_file = {f: i for i, f in enumerate(files)}
    rec = {}
    for si, s in enumerate(styles):
        rows, tiles = [], []
        for r in meta[s]["swatches"]:
            if "tiles" not in r:
                continue
            j = by_file.get(f"{s}/{r['file']}")
            if j is None:
                continue
            rows.append(codes[j]); tiles.append(np.asarray(r["tiles"], np.int64))
        if not rows:
            continue
        C = np.stack(rows).reshape(-1, 16, 16)
        Tg = np.stack(tiles)                                    # (n, 8, 8) true tile ids
        blk = np.stack([C[:, 0::2, 0::2], C[:, 0::2, 1::2],
                        C[:, 1::2, 0::2], C[:, 1::2, 1::2]], -1).reshape(-1, 4)
        _, bid, cnt = np.unique(blk, axis=0, return_inverse=True, return_counts=True)
        tid = Tg.reshape(-1)
        keep = cnt[bid] >= support
        b2t, nb = purity(bid[keep], tid[keep]) if keep.any() else (float("nan"), 0)
        t2b, nt = purity(tid, bid)
        # one level up: 2x2 blocks of blocks vs 2x2 tile motifs
        n_sw = C.shape[0]
        bg = bid.reshape(n_sw, 8, 8)
        mot_b = np.stack([bg[:, 0::2, 0::2], bg[:, 0::2, 1::2],
                          bg[:, 1::2, 0::2], bg[:, 1::2, 1::2]], -1).reshape(-1, 4)
        mot_t = np.stack([Tg[:, 0::2, 0::2], Tg[:, 0::2, 1::2],
                          Tg[:, 1::2, 0::2], Tg[:, 1::2, 1::2]], -1).reshape(-1, 4)
        _, mb, mc = np.unique(mot_b, axis=0, return_inverse=True, return_counts=True)
        _, mt, _ = np.unique(mot_t, axis=0, return_inverse=True, return_counts=True)
        km = mc[mb] >= support
        m_b2t = purity(mb[km], mt[km])[0] if km.any() else float("nan")
        rec[s] = {"block_to_tile_purity": b2t, "n_blocks_at_support": nb,
                  "tile_to_block_purity": t2b, "n_tiles_used": nt,
                  "mass_at_support": float(keep.mean()),
                  "motif_block_to_tile_purity": m_b2t,
                  "motif_mass_at_support": float(km.mean()),
                  "n_true_motifs": int(len(np.unique(mt))),
                  "n_mined_motifs_at_support": int(len(np.unique(mb[km]))) if km.any() else 0}
    R["tile_recovery"] = rec
    print(f"[recovery] {len(rec)} schools ({time.time()-t0:.0f}s)", flush=True)

    # ---- load the models trained by plant.run on THIS corpus --------------------------- #
    def load(name):
        m = MD.MaskedGrid(k, n_style, d=d, layers=layers, heads=heads, p=dropout)
        m.load_state_dict(torch.load(os.path.join(out, f"{name}.pt"), map_location=DEV))
        return m.to(DEV).eval()

    plant = load("plant")
    G = {g: load(g) for g in graders.split(",")}

    # ---- 2. the damage panel with known validity --------------------------------------- #
    imgs, keys = [], []
    for s in styles:
        for r in meta[s]["damage"]:
            for kind, fn in (("clean", r["clean_file"]), (r["mode"], r["key"] + ".png")):
                p = os.path.join(croot, s, "dmg", fn)
                if not os.path.exists(p):
                    continue
                imgs.append(np.asarray(Image.open(p).convert("RGB")))
                keys.append((s, r["key"], kind, r["level"], tuple(r["code_rect"]),
                             bool(r["valid"]) if kind != "clean" else True,
                             int(r["n_viol"]) if kind != "clean" else 0))
    imgs = np.stack(imgs)
    Cd = torch.as_tensor(cents, device=DEV)
    dcodes, _ = CB.encode(imgs, Cd, dev=DEV)
    print(f"[damage] {len(imgs)} panel images encoded ({time.time()-t0:.0f}s)", flush=True)

    sty_of = {s: i for i, s in enumerate(styles)}
    masks = np.zeros((len(keys), 256), bool)
    for i, kk in enumerate(keys):
        r0, c0, r1, c1 = kk[4]
        m = np.zeros((16, 16), bool)
        m[max(0, r0):min(16, r1), max(0, c0):min(16, c1)] = True
        masks[i] = m.reshape(-1)
    dsty = np.array([sty_of[kk[0]] for kk in keys])
    dlvl = np.array([kk[3] for kk in keys])
    dkind = np.array([kk[2] for kk in keys])
    dvalid = np.array([kk[5] for kk in keys])

    # the plant's own fill of the SAME masks, from the clean surround
    clean_i = {(kk[0], kk[1].rsplit("_", 1)[0]): i for i, kk in enumerate(keys) if kk[2] == "clean"}
    fills = np.array(dcodes, copy=True)
    for lv in LEVELS:
        sel = np.flatnonzero((dlvl == lv) & (dkind == "clean"))
        if len(sel) == 0:
            continue
        for area in np.unique(masks[sel].sum(1)):
            ss = sel[masks[sel].sum(1) == area]
            f, _ = decode(plant, np.where(masks[ss], k, dcodes[ss]), masks[ss], dsty[ss],
                          steps=8, width=1, seed=seed, dev=DEV)
            fills[ss] = np.where(masks[ss], f, dcodes[ss])

    # ---- 3. validate the plant's fills against the grammar (aligned only) --------------- #
    # The classifier (nearest rendered tile, in pixel space, after the VQ round trip) is not
    # exact, so its accuracy against the TRUE tile ids is measured first and reported as the
    # ceiling. The fill is then scored on a grid whose surround is TRUE and whose masked
    # region alone is classified, so a classification error outside the hole cannot be charged
    # to the plant. `viol_true` (should be 0) is the sanity check on the whole path.
    if aligned:
        per = {}
        for si, s in enumerate(styles):
            sel = np.flatnonzero((dsty == si) & (dkind == "clean"))
            if len(sel) == 0:
                continue
            atlas = orc[s]["atlas"]; edges = orc[s]["edges"]
            truth = {(r["seed"], tuple(r["offset_px"])): np.asarray(r["tiles"], np.int64)
                     for r in meta[s]["swatches"] if "tiles" in r}
            dmg_off = {(r["seed"], r["level"]): tuple(r["offset_px"]) for r in meta[s]["damage"]}
            px_f = _decode_px(fills[sel], cents)
            px_c = _decode_px(dcodes[sel], cents)
            nv_f, nv_c, nv_t, acc, ok_t = [], [], [], [], []
            for j in range(len(sel)):
                kk = keys[sel[j]]
                seed_ = int(kk[1].split("_")[0][1:]); lvl_ = kk[3]
                gt = truth.get((seed_, dmg_off.get((seed_, lvl_))))
                gf, _ = classify_tiles(px_f[j], atlas)
                gc, _ = classify_tiles(px_c[j], atlas)
                r0, c0, r1, c1 = kk[4]
                hole = np.zeros((8, 8), bool)
                hole[max(0, r0 // 2):r1 // 2, max(0, c0 // 2):c1 // 2] = True
                if gt is None or gt.shape != (8, 8):
                    continue
                acc.append(float((gc == gt).mean()))
                nv_t.append(n_violations(gt, edges))
                comp_f = np.where(hole, gf, gt)          # true surround, classified fill
                comp_c = np.where(hole, gc, gt)          # true surround, classified truth (ceiling)
                nv_f.append(n_violations(comp_f, edges))
                nv_c.append(n_violations(comp_c, edges))
                ok_t.append(float((gc[hole] == gt[hole]).mean()))
            if not nv_f:
                continue
            per[s] = {"viol_fill": float(np.mean(nv_f)),
                      "viol_clean_roundtrip": float(np.mean(nv_c)),
                      "viol_true": float(np.mean(nv_t)),
                      "valid_fill": float(np.mean(np.array(nv_f) == 0)),
                      "valid_clean_roundtrip": float(np.mean(np.array(nv_c) == 0)),
                      "valid_true": float(np.mean(np.array(nv_t) == 0)),
                      "classify_acc_vs_truth": float(np.mean(acc)),
                      "classify_acc_in_hole": float(np.mean(ok_t)), "n": len(nv_f)}
        R["fill_validity"] = per
        cols = ["viol_fill", "viol_clean_roundtrip", "viol_true", "valid_fill",
                "valid_clean_roundtrip", "valid_true", "classify_acc_vs_truth",
                "classify_acc_in_hole"]
        agg = np.array([[v[c_] for c_ in cols] for v in per.values()])
        R["fill_validity_mean"] = dict(zip(cols, agg.mean(0).tolist()))
        print(f"[validity] {R['fill_validity_mean']} ({time.time()-t0:.0f}s)", flush=True)

    # ---- grade every class, one-sided and two-sided ------------------------------------- #
    def cal_set(name, sd, tau_masks=16):
        m = split == name
        c, s = codes[m], sidx[m]
        rr = np.random.default_rng(sd)
        rows, msk, sides = [], [], []
        for side in (2, 4, 8):
            for i in range(len(c)):
                for _ in range(tau_masks):
                    rows.append(i); msk.append(MD.rect_mask(rr, side)); sides.append(side)
        return c[np.array(rows)], s[np.array(rows)], np.stack(msk), np.array(sides)

    lvl_side = {1: 2, 2: 4, 3: 8}
    res = {}
    for gname, g in G.items():
        cg, cs, cm, csz = cal_set("gB_val" if gname == "gB" else "gA_val",
                                  seed + (22 if gname == "gB" else 21))
        ncal = GR.score(g, cg, cm, cs, n_orders=n_orders, n_steps=n_steps, seed=seed, dev=DEV)
        kcal = list(zip(cs.tolist(), csz.tolist()))
        hi = GR.tau_from(ncal, kcal, q)
        lo = GR.tau_from(ncal, kcal, q_lo)
        rows = {}
        for lv in LEVELS:
            side = lvl_side[lv]
            for kind in ("clean", "seam", "offstyle", "plant_fill"):
                if kind == "plant_fill":
                    sel = np.flatnonzero((dlvl == lv) & (dkind == "clean"))
                    grids = fills[sel]
                else:
                    sel = np.flatnonzero((dlvl == lv) & (dkind == kind))
                    grids = dcodes[sel]
                if len(sel) == 0:
                    continue
                nll = GR.score(g, grids, masks[sel], dsty[sel], n_orders=n_orders,
                               n_steps=n_steps, seed=seed + 3, dev=DEV)
                kz = [(int(u), side) for u in dsty[sel]]
                p1 = GR.apply_tau(nll, kz, hi)
                p2 = p1 & ~GR.apply_tau(nll, kz, lo)
                rows[f"{lv}_{kind}"] = {
                    "n": int(len(sel)), "nll": float(nll.mean()),
                    "pass_one_sided": float(p1.mean()), "pass_two_sided": float(p2.mean()),
                    "frac_valid": float(dvalid[sel].mean()) if kind not in ("plant_fill",) else None}
        res[gname] = rows
        print(f"[{gname}] " + " ".join(
            f"L{lv}:{rows.get(f'{lv}_seam',{}).get('pass_one_sided',float('nan')):.2f}/"
            f"{rows.get(f'{lv}_offstyle',{}).get('pass_one_sided',float('nan')):.2f}/"
            f"{rows.get(f'{lv}_plant_fill',{}).get('pass_one_sided',float('nan')):.2f}"
            for lv in LEVELS) + f"  (seam/offstyle/plantfill) ({time.time()-t0:.0f}s)", flush=True)
    R["damage_grading"] = res
    R["secs"] = time.time() - t0
    os.makedirs(out, exist_ok=True)
    json.dump(R, open(os.path.join(out, "oracle.json"), "w"), indent=1, cls=NumpyEncoder)
    _commit()
    print(f"done in {time.time()-t0:.0f}s -> {out}/oracle.json")
    return {"tag": tag, "secs": R["secs"]}


# --------------------------------------------------------------------------- #
# per-TOKEN dump of the known-validity panel (offline tail re-analysis)
# --------------------------------------------------------------------------- #

@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=7200, memory=65536)
def panel_tokens(tag: str = "tw_aligned", qtag: str = "twq_aligned",
                 corpus: str = "tiles_aligned", k: int = 512, seed: int = 0, d: int = 256,
                 layers: int = 6, heads: int = 8, dropout: float = 0.1,
                 graders: str = "gA24,gB", n_orders: int = 2, n_steps: int = 4,
                 tau_masks: int = 16):
    """`oracle_run`'s panel and calibration slices, replayed against the SAVED weights, with
    every PER-TOKEN NLL written out instead of only the per-item mean.

    Nothing is retrained and nothing is re-sampled: the panel-building block and the
    calibration block below are copied VERBATIM from `oracle_run` (same seeds, same rng
    consumption, same order), so `np.nanmean` of what this dumps reproduces the one-sided
    numbers already in `oracle.json`. The point is the re-analysis in `tailgrade.py`: the
    grade of record is a MEAN over these tokens, and the question is whether a TAIL statistic
    of the same array reads validity where the mean reads style.

    Also dumped, and only here: the PER-ITEM oracle validity of the plant's own fills
    (`oracle_run` computes the same `n_violations` but keeps only per-school means), which is
    what lets the plant's pass rate be split by whether the fill was actually in the grammar.

    Arrays are stored ragged (`*vals` concatenated, `*nt` per-item token counts) because a
    (N, 256) array of mostly-NaN does not compress.
    """
    import torch
    from PIL import Image
    from canvas.plant import codebook as CB, grader as GR, model as MD
    from canvas.plant.sampler import decode

    torch.backends.cuda.matmul.allow_tf32 = True
    t0 = time.time()
    croot = os.path.join(ROOT, "corpora", corpus)
    qdir = os.path.join(ROOT, REMOTE, qtag)
    out = os.path.join(ROOT, REMOTE, tag)
    idx = json.load(open(os.path.join(qdir, "index.json")))
    styles = idx["styles"]; n_style = len(styles)
    sidx = np.array(idx["sidx"], np.int64); split = np.array(idx["split"])
    z = np.load(os.path.join(qdir, f"quant_K{k}.npz"))
    codes = z["codes"].astype(np.int64); cents = z["centroids"].astype(np.float32)
    aligned = "misaligned" not in corpus

    meta = {s: json.load(open(os.path.join(croot, s, "style.json"))) for s in styles}
    orc = {s: np.load(os.path.join(croot, s, "oracle.npz")) for s in styles}

    def load(name):
        m = MD.MaskedGrid(k, n_style, d=d, layers=layers, heads=heads, p=dropout)
        m.load_state_dict(torch.load(os.path.join(out, f"{name}.pt"), map_location=DEV))
        return m.to(DEV).eval()

    plant = load("plant")
    G = {g: load(g) for g in graders.split(",")}

    # ---- the damage panel, verbatim from `oracle_run` ---------------------------------- #
    imgs, keys = [], []
    for s in styles:
        for r in meta[s]["damage"]:
            for kind, fn in (("clean", r["clean_file"]), (r["mode"], r["key"] + ".png")):
                p = os.path.join(croot, s, "dmg", fn)
                if not os.path.exists(p):
                    continue
                imgs.append(np.asarray(Image.open(p).convert("RGB")))
                keys.append((s, r["key"], kind, r["level"], tuple(r["code_rect"]),
                             bool(r["valid"]) if kind != "clean" else True,
                             int(r["n_viol"]) if kind != "clean" else 0))
    imgs = np.stack(imgs)
    Cd = torch.as_tensor(cents, device=DEV)
    dcodes, _ = CB.encode(imgs, Cd, dev=DEV)
    print(f"[panel] {len(imgs)} images encoded ({time.time()-t0:.0f}s)", flush=True)

    sty_of = {s: i for i, s in enumerate(styles)}
    masks = np.zeros((len(keys), 256), bool)
    for i, kk in enumerate(keys):
        r0, c0, r1, c1 = kk[4]
        m = np.zeros((16, 16), bool)
        m[max(0, r0):min(16, r1), max(0, c0):min(16, c1)] = True
        masks[i] = m.reshape(-1)
    dsty = np.array([sty_of[kk[0]] for kk in keys])
    dlvl = np.array([kk[3] for kk in keys])
    dkind = np.array([kk[2] for kk in keys])
    dvalid = np.array([kk[5] for kk in keys])
    dnviol = np.array([kk[6] for kk in keys])
    drect = np.array([list(kk[4]) for kk in keys], np.int64)

    fills = np.array(dcodes, copy=True)
    for lv in LEVELS:
        sel = np.flatnonzero((dlvl == lv) & (dkind == "clean"))
        if len(sel) == 0:
            continue
        for area in np.unique(masks[sel].sum(1)):
            ss = sel[masks[sel].sum(1) == area]
            f, _ = decode(plant, np.where(masks[ss], k, dcodes[ss]), masks[ss], dsty[ss],
                          steps=8, width=1, seed=seed, dev=DEV)
            fills[ss] = np.where(masks[ss], f, dcodes[ss])

    D = {"styles": np.array(styles), "dsty": dsty, "dlvl": dlvl, "dkind": dkind,
         "dvalid": dvalid, "dnviol": dnviol, "drect": drect, "pmask": masks,
         "aligned": np.array(aligned),
         # the code grids themselves, for graders that take NO forward pass at all
         # (`adjacency.py`'s support test): the panel as encoded, and the plant's own fill
         # of every clean item's hole, which is the object `oracle.json` grades.
         "dcodes": dcodes.astype(np.int32), "fills": fills.astype(np.int32)}

    # ---- PER-ITEM fill validity (oracle_run keeps only the per-school mean) -------------- #
    if aligned:
        nvf = np.full(len(keys), -1, np.int64)
        nvc = np.full(len(keys), -1, np.int64)
        nvt = np.full(len(keys), -1, np.int64)
        hacc = np.full(len(keys), np.nan)
        for si, s in enumerate(styles):
            sel = np.flatnonzero((dsty == si) & (dkind == "clean"))
            if len(sel) == 0:
                continue
            atlas = orc[s]["atlas"]; edges = orc[s]["edges"]
            truth = {(r["seed"], tuple(r["offset_px"])): np.asarray(r["tiles"], np.int64)
                     for r in meta[s]["swatches"] if "tiles" in r}
            dmg_off = {(r["seed"], r["level"]): tuple(r["offset_px"]) for r in meta[s]["damage"]}
            px_f = _decode_px(fills[sel], cents)
            px_c = _decode_px(dcodes[sel], cents)
            for j in range(len(sel)):
                kk = keys[sel[j]]
                seed_ = int(kk[1].split("_")[0][1:]); lvl_ = kk[3]
                gt = truth.get((seed_, dmg_off.get((seed_, lvl_))))
                if gt is None or gt.shape != (8, 8):
                    continue
                gf, _ = classify_tiles(px_f[j], atlas)
                gc, _ = classify_tiles(px_c[j], atlas)
                r0, c0, r1, c1 = kk[4]
                hole = np.zeros((8, 8), bool)
                hole[max(0, r0 // 2):r1 // 2, max(0, c0 // 2):c1 // 2] = True
                nvf[sel[j]] = n_violations(np.where(hole, gf, gt), edges)
                nvc[sel[j]] = n_violations(np.where(hole, gc, gt), edges)
                nvt[sel[j]] = n_violations(gt, edges)
                hacc[sel[j]] = float((gc[hole] == gt[hole]).mean())
        D.update({"fill_nviol": nvf, "roundtrip_nviol": nvc, "true_nviol": nvt,
                  "hole_classify_acc": hacc})
        ok = nvf >= 0
        print(f"[validity] n={int(ok.sum())} valid_fill={float((nvf[ok]==0).mean()):.3f} "
              f"valid_roundtrip={float((nvc[ok]==0).mean()):.3f} ({time.time()-t0:.0f}s)",
              flush=True)

    # ---- calibration slices, verbatim from `oracle_run` --------------------------------- #
    def cal_set(name, sd):
        m = split == name
        c, s = codes[m], sidx[m]
        rr = np.random.default_rng(sd)
        rows, msk, sides = [], [], []
        for side in (2, 4, 8):
            for i in range(len(c)):
                for _ in range(tau_masks):
                    rows.append(i); msk.append(MD.rect_mask(rr, side)); sides.append(side)
        return c[np.array(rows)], s[np.array(rows)], np.stack(msk), np.array(sides)

    def ragged(tok, msk):
        nt = msk.sum(1).astype(np.int32)
        return tok[msk].astype(np.float32), nt

    # The panel is scored ONE (level, kind) SUBSET AT A TIME, in `oracle_run`'s order and with
    # `oracle_run`'s seed, because `GR.score`'s reveal orders are drawn per call over the rows
    # it is handed: scoring the whole panel in one call would be a different (equally valid)
    # draw and the mean column would no longer reproduce `oracle.json` to the digit.
    KINDS = ("clean", "seam", "offstyle", "plant_fill")
    for gname, g in G.items():
        cg, cs, cm, csz = cal_set("gB_val" if gname == "gB" else "gA_val",
                                  seed + (22 if gname == "gB" else 21))
        D[f"calstyle|{gname}"] = cs
        D[f"calside|{gname}"] = csz
        D[f"calnt|{gname}"] = cm.sum(1).astype(np.int32)
        for col, (no, ns) in {"": (n_orders, n_steps), "1": (1, 1)}.items():
            D[f"calvals{col}|{gname}"] = ragged(
                GR.score_tokens(g, cg, cm, cs, n_orders=no, n_steps=ns, seed=seed, dev=DEV),
                cm)[0]
        for lv in LEVELS:
            for kind in KINDS:
                if kind == "plant_fill":
                    sel = np.flatnonzero((dlvl == lv) & (dkind == "clean"))
                    grids = fills[sel]
                else:
                    sel = np.flatnonzero((dlvl == lv) & (dkind == kind))
                    grids = dcodes[sel]
                if len(sel) == 0:
                    continue
                D[f"sel|{lv}|{kind}"] = sel
                D[f"nt|{lv}|{kind}"] = masks[sel].sum(1).astype(np.int32)
                for col, (no, ns) in {"": (n_orders, n_steps), "1": (1, 1)}.items():
                    D[f"t{col}|{gname}|{lv}|{kind}"] = ragged(
                        GR.score_tokens(g, grids, masks[sel], dsty[sel], n_orders=no,
                                        n_steps=ns, seed=seed + 3, dev=DEV), masks[sel])[0]
        print(f"[{gname}] tokens dumped ({time.time()-t0:.0f}s)", flush=True)

    os.makedirs(out, exist_ok=True)
    np.savez_compressed(os.path.join(out, "panel_tokens.npz"), **D)
    _commit()
    print(f"done in {time.time()-t0:.0f}s -> {out}/panel_tokens.npz")
    return {"tag": tag, "n_panel": int(len(keys)), "secs": time.time() - t0}
