"""canvas/plant -- DESCRIBE THE PLANT. Not a practice loop.

Before a practice node can run on image styles, the plant has to exist and be described:
a codebook (`T[1]`, a first-regime object), an any-order token model over the code grid, and
a grade of record with nothing behind it. This node builds those three and MEASURES THE
PRECONDITIONS the arc has been shown to need, in the spirit of `canvas/tiles.py:describe()`
and `rhm/practice/tall/`'s preflight.

Five readouts, all reported per style:
  1. the QUANTIZER FLOOR vs K -- the bound everything downstream is under.
  2. the MASK-SIZE LADDER of held-out loss over training -- the depth ladder is mask size, so
     this is whether coarse structure is learned after fine (a learning wave in pixel space).
  3. RECURRENCE of 2x2 code blocks and of 2x2 blocks of those (`recur.py`) -- whether the
     alphabet factors into parts, with the library's part-y/texture-y split as a free
     within-substrate control.
  4. COST-TO-DEPTH: completion quality vs mask size under the beam at a couple of widths --
     whether depth is unaffordable to the primitive action space (the meter's precondition).
  5. GRADER CALIBRATION: `critic/`'s protocol run for the first time with no exact grade
     beside it -- pass rates on clean / self-manufactured damage / the plant's own argmax,
     the ROC over q, and where the corpus-size flip lands.

Run (from experiments/, MODAL_PROFILE=chromatic):
    modal run -m canvas.plant.plant::selfcheck
    modal run -m canvas.plant.plant::unpack
    modal run -m canvas.plant.plant::quantize_ladder --tag q0
    modal run -m canvas.plant.plant::run --tag smoke --smoke 1
    python3 canvas/plant/launch_detached.py --fn run --tag pl0 --qtag q0 --k 512
"""

import json
import os
import time

import numpy as np

from canvas.shared import DATA_DIR, NumpyEncoder, app, volume

REMOTE = "plant"
CORPUS = "plant0"

# Overridable so the whole node can be smoked locally on CPU before any Modal round trip.
ROOT = os.environ.get("CANVAS_DATA", DATA_DIR)
DEV = os.environ.get("CANVAS_DEV", "cuda")


def _commit():
    try:
        volume.commit()
    except Exception:
        pass

MASK_LADDER = (1, 2, 3, 4, 6, 8, 11, 16)
PANEL_SIDES = (2, 4, 8, 11)
PANEL_CLASSES = ("clean", "same_style_other", "roll", "shuffle", "other_style",
                 "marginal", "uniform", "plant_argmax", "plant_iter8", "plant_beam4")
GRADER_LADDER = (1, 2, 4, 8, 16, 24)        # swatches per style in gA's training corpus

# the library's own visible split, taken from the prompt verbatim; every other style is
# reported individually and left unlabelled, so the control is not a taxonomy we invented.
PARTY = ("girih_stars", "quilt_patch", "circuit_traces", "tumbling_blocks",
         "truchet_meander", "stained_glass_cells")
TEXTUREY = ("agate_bands", "brain_coral", "wood_grain", "lichen_crust")


# --------------------------------------------------------------------------- #
# corpus
# --------------------------------------------------------------------------- #

def load_corpus(root):
    from PIL import Image
    man = json.load(open(os.path.join(root, "manifest.json")))
    styles = man["styles"]
    imgs, sidx, split, files = [], [], [], []
    for si, s in enumerate(styles):
        meta = json.load(open(os.path.join(root, s, "style.json")))
        for rec in meta["swatches"]:
            imgs.append(np.asarray(Image.open(os.path.join(root, s, rec["file"])).convert("RGB")))
            sidx.append(si); split.append(rec["split"]); files.append(f"{s}/{rec['file']}")
    return (np.stack(imgs), np.array(sidx, np.int64), np.array(split), files, styles, man)


@app.function(volumes={DATA_DIR: volume}, timeout=3600, memory=16384)
def unpack(tar: str = f"corpora/{CORPUS}.tar"):
    import tarfile
    src = os.path.join(ROOT, tar)
    dst = os.path.join(ROOT, "corpora")
    with tarfile.open(src) as t:
        t.extractall(dst)
    _commit()
    root = os.path.join(dst, os.path.basename(tar)[:-4])
    man = json.load(open(os.path.join(root, "manifest.json")))
    print(f"unpacked {len(man['styles'])} styles to {root}")
    return root


# --------------------------------------------------------------------------- #
# stage 1: the quantizer floor vs K
# --------------------------------------------------------------------------- #

@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=7200, memory=65536)
def quantize_ladder(tag: str = "q0", ks: str = "256,512,1024,2048", corpus: str = CORPUS,
                    n_fit: int = 400000, iters: int = 40, seed: int = 0):
    import torch
    from canvas.plant import codebook as CB

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    t0 = time.time()
    root = os.path.join(ROOT, "corpora", corpus)
    imgs, sidx, split, files, styles, man = load_corpus(root)
    print(f"corpus: {imgs.shape} {len(styles)} styles  ({time.time()-t0:.0f}s)", flush=True)
    out = os.path.join(ROOT, REMOTE, tag)
    os.makedirs(out, exist_ok=True)
    json.dump({"styles": styles, "sidx": sidx.tolist(), "split": split.tolist(),
               "files": files, "corpus": corpus},
              open(os.path.join(out, "index.json"), "w"))

    # per-style patch variance: the K=1 floor, so the reported error can be read as a
    # FRACTION of what the style's own pixels vary by (a tiling and a noise field are not
    # comparable in raw MSE).
    pvar = {}
    for si, s_ in enumerate(styles):
        pp = CB.to_patches(imgs[sidx == si][:16]).reshape(-1, 16 * 16 * 3)
        pvar[s_] = float(pp.var(0).mean())

    fit = split == "plant_train"
    P = CB.to_patches(imgs[fit]).reshape(-1, 16 * 16 * 3)
    rng = np.random.default_rng(seed)
    sub = rng.permutation(len(P))[:n_fit]
    X = torch.as_tensor(P[sub], device=DEV)
    print(f"fit patches {tuple(X.shape)} from {int(fit.sum())} plant_train swatches", flush=True)
    del P

    stats = {"tag": tag, "corpus": corpus, "n_fit": int(len(sub)), "iters": iters, "per_k": {}}
    for K in [int(k) for k in ks.split(",")]:
        t1 = time.time()
        C = CB.kmeans(X, K, iters=iters, seed=seed, dev=DEV)
        codes, mse = CB.encode(imgs, C, dev=DEV)
        per_style = {}
        for si, s in enumerate(styles):
            sel = sidx == si
            row = {"mse_all": float(mse[sel].mean()),
                   "psnr_all": CB.psnr(mse[sel].mean()),
                   "patch_var": pvar[s],
                   "nmse_all": float(mse[sel].mean() / max(pvar[s], 1e-9))}
            for sp in ("plant_train", "plant_val", "plant_test"):
                m = sel & (split == sp)
                row[f"mse_{sp}"] = float(mse[m].mean())
                row[f"psnr_{sp}"] = CB.psnr(mse[m].mean())
            u = np.bincount(codes[sel].reshape(-1), minlength=K).astype(float)
            p = u / u.sum()
            row["codes_used"] = int((u > 0).sum())
            row["code_perplexity"] = float(np.exp(-(p[p > 0] * np.log(p[p > 0])).sum()))
            per_style[s] = row
        stats["per_k"][str(K)] = {
            "mse": float(mse.mean()), "psnr": CB.psnr(mse.mean()),
            "mse_heldout": float(mse[split != "plant_train"].mean()),
            "nmse": float(np.mean([mse[sidx == si].mean() / max(pvar[s_], 1e-9)
                                   for si, s_ in enumerate(styles)])),
            "codes_used": int(len(np.unique(codes))),
            "secs": time.time() - t1, "per_style": per_style}
        np.savez_compressed(os.path.join(out, f"quant_K{K}.npz"),
                            codes=codes.astype(np.uint16), centroids=C.cpu().numpy().astype(np.float16),
                            mse=mse.astype(np.float32))
        print(f"K={K}: mse {mse.mean():.5f} psnr {CB.psnr(mse.mean()):.2f} "
              f"used {stats['per_k'][str(K)]['codes_used']}/{K} ({time.time()-t1:.0f}s)", flush=True)
        _commit()
    json.dump(stats, open(os.path.join(out, "quant_stats.json"), "w"), indent=1, cls=NumpyEncoder)
    _commit()
    print(f"done in {time.time()-t0:.0f}s")
    return stats


# --------------------------------------------------------------------------- #
# stage 2: the plant, the graders, and the five readouts
# --------------------------------------------------------------------------- #

@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=28800, memory=32768)
def run(tag: str = "pl0", qtag: str = "q0", k: int = 512, seed: int = 0,
        d: int = 256, layers: int = 6, heads: int = 8,
        plant_steps: int = 20000, grader_steps: int = 4000, bs: int = 128,
        n_ckpt: int = 14, ladder_reps: int = 16,
        panel_per_style: int = 6, tau_masks: int = 16, q: float = 0.90,
        dropout: float = 0.1,
        n_orders: int = 2, n_steps: int = 4, support: int = 8, smoke: int = 0):
    import torch
    from canvas.plant import codebook as CB, grader as GR, model as MD, recur as RC
    from canvas.plant.sampler import decode

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if smoke:
        plant_steps, grader_steps, n_ckpt = 120, 60, 2
        ladder_reps, panel_per_style, tau_masks = 1, 1, 4

    t0 = time.time()
    qdir = os.path.join(ROOT, REMOTE, qtag)
    idx = json.load(open(os.path.join(qdir, "index.json")))
    styles = idx["styles"]; n_style = len(styles)
    sidx = np.array(idx["sidx"], np.int64); split = np.array(idx["split"])
    z = np.load(os.path.join(qdir, f"quant_K{k}.npz"))
    codes = z["codes"].astype(np.int64)
    cents = z["centroids"].astype(np.float32)
    print(f"codes {codes.shape} K={k} styles={n_style}", flush=True)

    out = os.path.join(ROOT, REMOTE, tag)
    os.makedirs(out, exist_ok=True)
    R = {"tag": tag, "qtag": qtag, "K": k, "seed": seed, "styles": styles,
         "cfg": {"d": d, "layers": layers, "heads": heads, "plant_steps": plant_steps,
                 "grader_steps": grader_steps, "bs": bs, "q": q, "n_orders": n_orders,
                 "n_steps": n_steps, "support": support, "smoke": smoke,
                 "dropout": dropout, "n_ckpt": n_ckpt, "ladder_reps": ladder_reps,
                 "panel_per_style": panel_per_style, "tau_masks": tau_masks,
                 "grader_call_forwards": n_orders * n_steps,
                 "panel_sides": list(PANEL_SIDES), "grader_ladder": list(GRADER_LADDER)}}

    def S(name):
        m = split == name
        return codes[m], sidx[m], np.flatnonzero(m)

    ctr, str_, _ = S("plant_train"); cva, sva, _ = S("plant_val"); cte, ste, ite = S("plant_test")
    R["n"] = {kk: int((split == kk).sum()) for kk in np.unique(split)}

    # ---- 3. recurrence of the code grid (a property of the corpus, no model involved) ----
    rec = {}
    for si, s in enumerate(styles):
        rec[s] = RC.style_recurrence(codes[sidx == si], support=support)
    R["recurrence"] = rec
    R["groups"] = {"party": list(PARTY), "texturey": list(TEXTUREY)}
    print(f"[recur] done ({time.time()-t0:.0f}s)", flush=True)

    # ---- 2. the plant, with the mask-size ladder checkpointed over training ----
    torch.manual_seed(seed)
    plant = MD.MaskedGrid(k, n_style, d=d, layers=layers, heads=heads, p=dropout)
    ck = sorted(set(np.round(np.geomspace(max(1, plant_steps // 1000), plant_steps,
                                          n_ckpt)).astype(int).tolist()))
    wave = []

    def on_ckpt(step, loss):
        nl = MD.ladder_nll(plant, cva, sva, MASK_LADDER, reps=ladder_reps, seed=seed, dev=DEV)
        wave.append({"step": int(step), "train_loss": loss, "nll": nl.tolist()})
        print(f"[plant] step {step:6d} loss {loss:.3f}  ladder(mean) "
              f"{' '.join(f'{v:.3f}' for v in nl.mean(0))}  ({time.time()-t0:.0f}s)", flush=True)

    MD.train(plant, ctr, str_, steps=plant_steps, bs=bs, seed=seed, dev=DEV,
             ckpts=ck, on_ckpt=on_ckpt)
    R["wave"] = {"steps": [w["step"] for w in wave], "ladder": list(MASK_LADDER),
                 "train_loss": [w["train_loss"] for w in wave],
                 "nll": [w["nll"] for w in wave]}
    R["plant_final"] = {
        "val_style": MD.ladder_nll(plant, cva, sva, MASK_LADDER, reps=ladder_reps, seed=seed + 1, dev=DEV).tolist(),
        "val_nostyle": MD.ladder_nll(plant, cva, sva, MASK_LADDER, reps=ladder_reps, seed=seed + 1,
                                     use_style=False, dev=DEV).tolist(),
        "test_style": MD.ladder_nll(plant, cte, ste, MASK_LADDER, reps=ladder_reps, seed=seed + 2, dev=DEV).tolist(),
    }
    torch.save(plant.state_dict(), os.path.join(out, "plant.pt"))
    _commit()

    # ---- graders: gA on its ladder of corpus sizes, gB on a disjoint split ----
    def sub_corpus(name, n_per_style, sd):
        m = split == name
        c, s = codes[m], sidx[m]
        rng = np.random.default_rng(sd)
        keep = np.concatenate([rng.permutation(np.flatnonzero(s == u))[:n_per_style]
                               for u in range(n_style)])
        return c[keep], s[keep]

    graders = {}
    ladder = GRADER_LADDER if not smoke else (1, 24)
    for n in ladder:
        c, s = sub_corpus("gA_train", n, seed + 7)
        torch.manual_seed(seed + 100 + n)
        g = MD.MaskedGrid(k, n_style, d=d, layers=layers, heads=heads, p=dropout)
        MD.train(g, c, s, steps=grader_steps, bs=bs, seed=seed + 100 + n, dev=DEV)
        graders[f"gA{n}"] = g
        print(f"[gA{n}] trained on {len(c)} swatches ({time.time()-t0:.0f}s)", flush=True)
    c, s = sub_corpus("gB_train", 16, seed + 9)
    torch.manual_seed(seed + 900)
    gB = MD.MaskedGrid(k, n_style, d=d, layers=layers, heads=heads, p=dropout)
    MD.train(gB, c, s, steps=grader_steps, bs=bs, seed=seed + 900, dev=DEV)
    graders["gB"] = gB
    print(f"[gB] trained on {len(c)} swatches ({time.time()-t0:.0f}s)", flush=True)

    # ---- the panel: held-out swatches x mask sides x candidate classes ----
    rng = np.random.default_rng(seed + 11)
    marg = GR.marginals(ctr, str_, n_style, k)
    panel = {}
    for side in PANEL_SIDES:
        rows, masks = [], []
        for u in range(n_style):
            cand = np.flatnonzero(ste == u)
            for i in rng.permutation(cand)[:panel_per_style]:
                rows.append(i); masks.append(MD.rect_mask(rng, side))
        rows = np.array(rows); masks = np.stack(masks)
        base, bs_ = cte[rows], ste[rows]
        cls = {}
        for c_ in ("clean", "same_style_other", "roll", "shuffle", "other_style",
                   "marginal", "uniform"):
            cls[c_] = GR.manufacture(c_, base, masks, bs_, ctr, str_, marg, k, rng)
        hole = np.where(masks, k, base)
        for nm, (st, w) in {"plant_argmax": (1, 1), "plant_iter8": (8, 1),
                            "plant_beam4": (8, 4)}.items():
            filled, cost = decode(plant, hole, masks, bs_, steps=st, width=w, seed=seed, dev=DEV)
            cls[nm] = np.where(masks, filled, base)
            R.setdefault("sampler_cost", {})[nm] = int(cost)
        d_truth = np.array([(cls[c_][i][masks[i]] != base[i][masks[i]]).mean()
                            for c_ in PANEL_CLASSES for i in range(len(rows))]
                           ).reshape(len(PANEL_CLASSES), len(rows))
        panel[side] = {"rows": rows, "masks": masks, "styles": bs_, "cls": cls,
                       "d_truth": d_truth}
        print(f"[panel] side {side}: {len(rows)} instances ({time.time()-t0:.0f}s)", flush=True)

    # ---- tau: the oracle-free q-quantile on each grader's OWN validation slice ----
    def cal_set(name, sd):
        m = split == name
        c, s = codes[m], sidx[m]
        rr = np.random.default_rng(sd)
        rows, masks, sides = [], [], []
        for side in MASK_LADDER:
            for i in range(len(c)):
                for _ in range(tau_masks):
                    rows.append(i); masks.append(MD.rect_mask(rr, side)); sides.append(side)
        return c[np.array(rows)], s[np.array(rows)], np.stack(masks), np.array(sides)

    calA = cal_set("gA_val", seed + 21)
    calB = cal_set("gB_val", seed + 22)

    verdicts, taus, results, cals = {}, {}, {}, {}
    for gname, g in graders.items():
        cal = calB if gname == "gB" else calA
        cg, cs, cm, csz = cal
        nll_cal = GR.score(g, cg, cm, cs, n_orders=n_orders, n_steps=n_steps, seed=seed, dev=DEV)
        keys_cal = list(zip(cs.tolist(), csz.tolist()))
        tau = GR.tau_from(nll_cal, keys_cal, q)
        tau_glob = {(-1, s_): float(np.quantile(nll_cal[csz == s_], q)) for s_ in MASK_LADDER}
        cals[gname] = (nll_cal, keys_cal)
        taus[gname] = {"per_style_size": {f"{a}_{b}": v for (a, b), v in tau.items()},
                       "per_size": {str(b): v for (a, b), v in tau_glob.items()}}
        rows_out = {}
        for side in PANEL_SIDES:
            P = panel[side]
            keys = list(zip(P["styles"].tolist(), [side] * len(P["styles"])))
            kg = [(-1, side)] * len(P["styles"])
            for ci, c_ in enumerate(PANEL_CLASSES):
                nll = GR.score(g, P["cls"][c_], P["masks"], P["styles"],
                               n_orders=n_orders, n_steps=n_steps, seed=seed + 3, dev=DEV)
                nll1 = GR.score(g, P["cls"][c_], P["masks"], P["styles"],
                                n_orders=1, n_steps=1, seed=seed + 3, dev=DEV)
                ps = GR.apply_tau(nll, keys, tau)
                pg = GR.apply_tau(nll, kg, tau_glob)
                per_style = {styles[u]: float(ps[P["styles"] == u].mean())
                             for u in range(n_style)}
                rows_out[f"{side}_{c_}"] = {
                    "nll": float(nll.mean()), "nll_indep": float(nll1.mean()),
                    "pass": float(ps.mean()), "pass_global_tau": float(pg.mean()),
                    "d_truth": float(P["d_truth"][ci].mean()),
                    "r_score_dtruth": float(np.corrcoef(-nll, -P["d_truth"][ci])[0, 1])
                    if P["d_truth"][ci].std() > 0 else float("nan"),
                    "per_style_pass": per_style}
                verdicts.setdefault(gname, {})[f"{side}_{c_}"] = nll
        # the calibration rule, on the damage set only
        clean = np.mean([rows_out[f"{s_}_clean"]["pass"] for s_ in PANEL_SIDES])
        dam = np.mean([rows_out[f"{s_}_{c_}"]["pass"]
                       for s_ in PANEL_SIDES for c_ in GR.DAMAGE_CLASSES])
        rows_out["_rule"] = {"pass_clean": float(clean), "pass_damage": float(dam),
                             "qualifies": bool(clean >= 0.90 and dam <= 0.10)}
        results[gname] = rows_out
        print(f"[{gname}] clean {clean:.3f}  damage {dam:.3f}  "
              f"qualifies={rows_out['_rule']['qualifies']} ({time.time()-t0:.0f}s)", flush=True)

    qual = [n for n in ladder if results[f"gA{n}"]["_rule"]["qualifies"]]
    sel = f"gA{min(qual)}" if qual else f"gA{max(ladder)}"
    R["selected_grader"] = sel
    R["grader"] = results
    R["tau"] = taus

    # ---- ROC over q, and the homogeneous pair ----
    def cat(g, kind):
        return np.concatenate([verdicts[g][f"{s_}_{c_}"] for s_ in PANEL_SIDES
                               for c_ in ([kind] if isinstance(kind, str) else kind)])

    def keyz(kind):
        n_k = 1 if isinstance(kind, str) else len(kind)
        return [(int(u), s_) for s_ in PANEL_SIDES for _ in range(n_k)
                for u in panel[s_]["styles"]]

    R["roc"] = {g: GR.roc(cals[g][0], cals[g][1], cat(g, "clean"), keyz("clean"),
                          cat(g, GR.DAMAGE_CLASSES), keyz(GR.DAMAGE_CLASSES))
                for g in graders}
    R["pair"] = {"gA_sel_vs_gB": GR.pair_stats(cat(sel, PANEL_CLASSES), cat("gB", PANEL_CLASSES)),
                 "gA_sel_vs_self": GR.pair_stats(cat(sel, PANEL_CLASSES), cat(sel, PANEL_CLASSES))}

    # ---- 4. cost-to-depth: completion quality vs mask size at a couple of widths ----
    cent = torch.as_tensor(cents, device=DEV)
    tauB = {tuple(int(x) for x in key.split("_")): v
            for key, v in taus["gB"]["per_style_size"].items()}
    c2d = {}
    for side in MASK_LADDER:
        rows, masks = [], []
        rr = np.random.default_rng(seed + 31)
        for u in range(n_style):
            for i in rr.permutation(np.flatnonzero(ste == u))[:panel_per_style]:
                rows.append(i); masks.append(MD.rect_mask(rr, side))
        rows = np.array(rows); masks = np.stack(masks)
        base, bs_ = cte[rows], ste[rows]
        hole = np.where(masks, k, base)
        for nm, (st, w) in {"argmax1": (1, 1), "iter8": (8, 1), "beam8x4": (8, 4)}.items():
            filled, cost = decode(plant, hole, masks, bs_, steps=st, width=w, seed=seed, dev=DEV)
            comp = np.where(masks, filled, base)
            acc = float(np.mean([(comp[i][masks[i]] == base[i][masks[i]]).mean()
                                 for i in range(len(rows))]))
            pt = cent[torch.as_tensor(comp, device=DEV)]
            gt = cent[torch.as_tensor(base, device=DEV)]
            mm = torch.as_tensor(masks, device=DEV)[..., None]
            mse = float(((pt - gt) ** 2 * mm).sum() / (mm.sum() * pt.shape[-1]))
            nll = GR.score(gB, comp, masks, bs_, n_orders=n_orders, n_steps=n_steps, seed=seed, dev=DEV)
            kk = list(zip(bs_.tolist(), [side] * len(bs_)))
            passes = GR.apply_tau(nll, kk, tauB)
            c2d[f"{side}_{nm}"] = {"side": side, "sampler": nm, "cost": int(cost),
                                   "code_acc": acc, "pix_mse_quantized": mse,
                                   "gB_nll": float(nll.mean()), "gB_pass": float(passes.mean()),
                                   "per_style_acc": {styles[u]: float(np.mean(
                                       [(comp[i][masks[i]] == base[i][masks[i]]).mean()
                                        for i in np.flatnonzero(bs_ == u)])) for u in range(n_style)}}
        print(f"[c2d] side {side} done ({time.time()-t0:.0f}s)", flush=True)
    R["cost_to_depth"] = c2d

    # ---- a visual strip: one held-out swatch per style, completed at four mask sizes ----
    strip = {}
    rr = np.random.default_rng(seed + 41)
    for u in range(n_style):
        i = int(np.flatnonzero(ste == u)[0])
        for side in PANEL_SIDES:
            m = MD.rect_mask(rr, side)[None]
            b = cte[i:i + 1]; sy = ste[i:i + 1]
            strip[f"{styles[u]}|{side}|truth"] = b[0]
            strip[f"{styles[u]}|{side}|mask"] = m[0]
            for nm, (st, w) in {"argmax1": (1, 1), "iter8": (8, 1), "beam8x4": (8, 4)}.items():
                f_, _ = decode(plant, np.where(m, k, b), m, sy, steps=st, width=w,
                               seed=seed, dev=DEV)
                strip[f"{styles[u]}|{side}|{nm}"] = np.where(m, f_, b)[0]
    np.savez_compressed(os.path.join(out, "strip.npz"),
                        **{kk: v.astype(np.uint16 if v.dtype != bool else np.bool_)
                           for kk, v in strip.items()})
    print(f"[strip] done ({time.time()-t0:.0f}s)", flush=True)

    R["secs"] = time.time() - t0
    json.dump(R, open(os.path.join(out, "run.json"), "w"), indent=1, cls=NumpyEncoder)
    np.savez_compressed(os.path.join(out, "verdicts.npz"),
                        **{f"{g}|{kk}": v for g, dd in verdicts.items() for kk, v in dd.items()})
    for g, m in graders.items():
        torch.save(m.state_dict(), os.path.join(out, f"{g}.pt"))
    _commit()
    print(f"done in {time.time()-t0:.0f}s -> {out}/run.json")
    return {"tag": tag, "selected_grader": sel, "secs": R["secs"]}


# --------------------------------------------------------------------------- #
# gates
# --------------------------------------------------------------------------- #

def _selfcheck(dev="cuda"):
    import torch
    import torch.nn.functional as F
    from canvas.plant import codebook as CB, grader as GR, model as MD, recur as RC
    from canvas.plant.sampler import decode

    ok = {}
    rng = np.random.default_rng(0)

    # G-P: the patch view is a bijection (the codebook's decoder is a lookup, nothing else)
    x = rng.integers(0, 256, size=(3, 256, 256, 3)).astype(np.uint8)
    ok["G-P"] = bool(np.allclose(CB.from_patches(CB.to_patches(x)), x / 255.0))

    # G-Q: k-means recovers 3 well-separated clusters, and `encode` agrees with `assign`
    c0 = rng.normal(size=(3, 8)) * 10
    X = torch.as_tensor(np.repeat(c0, 200, 0) + rng.normal(size=(600, 8)) * 0.05,
                        dtype=torch.float32, device=dev)
    C = CB.kmeans(X, 3, iters=30, seed=0, dev=dev, verbose=False)
    a = CB.assign(X, C).cpu().numpy()
    ok["G-Q"] = bool(len(np.unique(a)) == 3 and
                     all(len(np.unique(a[i * 200:(i + 1) * 200])) == 1 for i in range(3)))

    # G-M: a rect mask is contiguous and has exactly side^2 cells
    good = []
    for side in (1, 2, 4, 8, 16):
        m = MD.rect_mask(rng, side).reshape(16, 16)
        rs, cs = np.flatnonzero(m.any(1)), np.flatnonzero(m.any(0))
        good.append(m.sum() == side * side and len(rs) == side and len(cs) == side
                    and rs[-1] - rs[0] == side - 1 and cs[-1] - cs[0] == side - 1)
    ok["G-M"] = bool(all(good))

    K, nS = 32, 4
    torch.manual_seed(0)
    g = MD.MaskedGrid(K, nS, d=64, layers=2, heads=4).to(dev).eval()
    grids = rng.integers(0, K, size=(8, 256))
    masks = np.stack([MD.rect_mask(rng, 4) for _ in range(8)])
    sty = rng.integers(0, nS, size=8)

    # G-S: the one-step chain rule IS the independent per-cell NLL given the surround
    s1 = GR.score(g, grids, masks, sty, n_orders=1, n_steps=1, dev=dev)
    with torch.no_grad():
        y = torch.as_tensor(grids, dtype=torch.long, device=dev)
        m = torch.as_tensor(masks, device=dev)
        xin = torch.where(m, torch.full_like(y, K), y)
        lp = F.log_softmax(g(xin, torch.as_tensor(sty, dtype=torch.long, device=dev)), -1)
        nll = -lp.gather(-1, y[..., None])[..., 0]
        ref = ((nll * m).sum(1) / m.sum(1)).cpu().numpy()
    ok["G-S"] = bool(np.allclose(s1, ref, atol=1e-4))

    # G-D: damage touches only the masked cells, and `clean` is the identity
    pool = rng.integers(0, K, size=(40, 256)); pstyle = rng.integers(0, nS, size=40)
    marg = GR.marginals(pool, pstyle, nS, K)
    hits = []
    for kind in GR.ALL_CLASSES:
        d_ = GR.manufacture(kind, grids, masks, sty, pool, pstyle, marg, K,
                            np.random.default_rng(1))
        hits.append(bool((d_[~masks] == grids[~masks]).all()))
        if kind == "clean":
            hits.append(bool((d_ == grids).all()))
    ok["G-D"] = bool(all(hits))

    # G-B: the sampler fills every masked cell, never touches the surround, and at
    # (steps=1, width=1) is exactly the argmax of one forward pass
    filled, cost = decode(g, grids, masks, sty, steps=1, width=1, dev=dev)
    with torch.no_grad():
        am = g(xin, torch.as_tensor(sty, dtype=torch.long, device=dev)).argmax(-1).cpu().numpy()
    ok["G-B"] = bool(cost == 1 and (filled[~masks] == grids[~masks]).all()
                     and (filled[masks] == am[masks]).all())
    f4, c4 = decode(g, grids, masks, sty, steps=8, width=4, dev=dev)
    ok["G-B4"] = bool(c4 == 32 and (f4[~masks] == grids[~masks]).all())

    # G-T: the oracle-free threshold passes ~q of the exemplars it was calibrated on
    v = rng.normal(size=2000); keys = [(int(i % 4), 4) for i in range(2000)]
    tau = GR.tau_from(v, keys, 0.90)
    ok["G-T"] = bool(abs(GR.apply_tau(v, keys, tau).mean() - 0.90) < 0.02 and len(tau) == 4)

    # G-R: recurrence separates a tiling from noise at level 2 and at level 3
    tile = np.tile(np.array([[1, 2], [3, 4]]), (8, 8)).reshape(1, -1).repeat(20, 0)
    rt = RC.style_recurrence(tile, support=8)
    rn = RC.style_recurrence(rng.integers(0, 512, size=(20, 256)), support=8)
    # a perfect tiling has ONE level-2 entry and one level-3 entry over it; uniform noise has
    # ~one entry per occurrence and NOTHING at support, so the ratchet constraint collapses its
    # level 3 to the single OOV block -- which is the structural foreclosure, working.
    ok["G-R"] = bool(rt["T2"]["n_distinct"] == 1 and rt["T3"]["n_distinct"] == 1
                     and rt["T3"]["frac_obs_all_at_support"] == 1.0
                     and rn["T2"]["n_distinct"] > 1000 and rn["n_level2_at_support"] == 0
                     and rn["T3"]["n_distinct"] == 1
                     and rn["T3"]["frac_obs_all_at_support"] == 0.0
                     and rn["T3_unconstrained"]["n_distinct"] > 100)

    for kk, vv in ok.items():
        print(f"  {kk}: {vv}")
    assert all(ok.values()), ok
    return ok


@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=1800, memory=16384)
def selfcheck():
    return _selfcheck(DEV)


# --------------------------------------------------------------------------- #
# offline re-analysis support: dump every per-item score the panel and the
# calibration slice produced, so a DIFFERENT decision rule can be tried without
# retraining anything.
# --------------------------------------------------------------------------- #

@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=7200, memory=32768)
def panel_dump(tag: str = "pl0", qtag: str = "q0", k: int = 512, seed: int = 0,
               d: int = 256, layers: int = 6, heads: int = 8, dropout: float = 0.1,
               graders: str = "gA24,gB", panel_per_style: int = 6, tau_masks: int = 16,
               n_orders: int = 2, n_steps: int = 4):
    """Re-run `run`'s panel and calibration slices against the SAVED weights and dump every
    per-item NLL, at both the 8-forward chain rule and the 1-forward independent form.

    The panel-building block below is copied verbatim from `run` (same seeds, same order,
    same rng consumption) so the dumped clean/damage arrays are the same instances the
    one-sided numbers in `run.json` were computed on. Nothing is retrained."""
    import torch
    from canvas.plant import grader as GR, model as MD
    from canvas.plant.sampler import decode

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    t0 = time.time()
    qdir = os.path.join(ROOT, REMOTE, qtag)
    idx = json.load(open(os.path.join(qdir, "index.json")))
    styles = idx["styles"]; n_style = len(styles)
    sidx = np.array(idx["sidx"], np.int64); split = np.array(idx["split"])
    codes = np.load(os.path.join(qdir, f"quant_K{k}.npz"))["codes"].astype(np.int64)
    out = os.path.join(ROOT, REMOTE, tag)

    def S(name):
        m = split == name
        return codes[m], sidx[m]

    ctr, str_ = S("plant_train"); cte, ste = S("plant_test")

    def load(name):
        m = MD.MaskedGrid(k, n_style, d=d, layers=layers, heads=heads, p=dropout)
        m.load_state_dict(torch.load(os.path.join(out, f"{name}.pt"), map_location=DEV))
        return m.to(DEV).eval()

    plant = load("plant")
    G = {g: load(g) for g in graders.split(",")}

    # ---- panel, verbatim from `run` ----
    rng = np.random.default_rng(seed + 11)
    marg = GR.marginals(ctr, str_, n_style, k)
    panel = {}
    for side in PANEL_SIDES:
        rows, masks = [], []
        for u in range(n_style):
            cand = np.flatnonzero(ste == u)
            for i in rng.permutation(cand)[:panel_per_style]:
                rows.append(i); masks.append(MD.rect_mask(rng, side))
        rows = np.array(rows); masks = np.stack(masks)
        base, bs_ = cte[rows], ste[rows]
        cls = {}
        for c_ in ("clean", "same_style_other", "roll", "shuffle", "other_style",
                   "marginal", "uniform"):
            cls[c_] = GR.manufacture(c_, base, masks, bs_, ctr, str_, marg, k, rng)
        hole = np.where(masks, k, base)
        for nm, (st, w) in {"plant_argmax": (1, 1), "plant_iter8": (8, 1),
                            "plant_beam4": (8, 4)}.items():
            filled, _ = decode(plant, hole, masks, bs_, steps=st, width=w, seed=seed, dev=DEV)
            cls[nm] = np.where(masks, filled, base)
        d_truth = np.array([(cls[c_][i][masks[i]] != base[i][masks[i]]).mean()
                            for c_ in PANEL_CLASSES for i in range(len(rows))]
                           ).reshape(len(PANEL_CLASSES), len(rows))
        panel[side] = {"masks": masks, "styles": bs_, "cls": cls, "d_truth": d_truth}

    def cal_set(name, sd):
        m = split == name
        c, s = codes[m], sidx[m]
        rr = np.random.default_rng(sd)
        rows, masks, sides = [], [], []
        for side in MASK_LADDER:
            for i in range(len(c)):
                for _ in range(tau_masks):
                    rows.append(i); masks.append(MD.rect_mask(rr, side)); sides.append(side)
        return c[np.array(rows)], s[np.array(rows)], np.stack(masks), np.array(sides)

    cal = {"gB": cal_set("gB_val", seed + 22)}
    dump = {}
    for gname, g in G.items():
        cg, cs, cm, csz = cal.get(gname, cal_set("gA_val", seed + 21))
        for tagn, (no, ns) in {"": (n_orders, n_steps), "1": (1, 1)}.items():
            dump[f"cal{tagn}|{gname}"] = GR.score(g, cg, cm, cs, n_orders=no, n_steps=ns,
                                                  seed=seed, dev=DEV)
        dump[f"calstyle|{gname}"] = cs
        dump[f"calside|{gname}"] = csz
        for side in PANEL_SIDES:
            P = panel[side]
            for ci, c_ in enumerate(PANEL_CLASSES):
                for tagn, (no, ns) in {"": (n_orders, n_steps), "1": (1, 1)}.items():
                    dump[f"p{tagn}|{gname}|{side}|{c_}"] = GR.score(
                        g, P["cls"][c_], P["masks"], P["styles"], n_orders=no, n_steps=ns,
                        seed=seed + 3, dev=DEV)
                dump[f"dtruth|{side}|{c_}"] = P["d_truth"][ci]
            dump[f"pstyle|{side}"] = P["styles"]
        print(f"[{gname}] dumped ({time.time()-t0:.0f}s)", flush=True)
    np.savez_compressed(os.path.join(out, "panel_dump.npz"), **dump)
    _commit()
    print(f"done in {time.time()-t0:.0f}s -> {out}/panel_dump.npz")
    return {"n_keys": len(dump), "secs": time.time() - t0}
