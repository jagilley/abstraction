"""[preplay/timbre] THE REDUCER — the banked reductions, unchanged, on each reader's columns, and the
cross-reader tables of record. Facts only.

1. For each reader form R, the rr1 output is rewritten in pp1's own schema with R's columns in the
   `shaped_refit` / `frozen` / `twin` slots (the banked projection stays in `banked` in every
   view), and `preplay/reduce_preplay.py` is run on it UNCHANGED -> `figures/rr1_<R>_reduction.txt`.
2. The rr5 output is rewritten in pp5's schema with R's gates in the `read` / `read_pair` /
   `read_m` / `frozen` / `twin` slots, and `preplay/reduce_readgate.py` is run on it UNCHANGED
   -> `figures/rr5_<R>_reduction.txt`. View `ridge` is arm 0: pp5's own gates.
3. `figures/timbre_reduction.txt`: the reader-by-trunk tables, seeds side by side, never averaged.

The rewritten views are written to a scratch directory, not the repo (they are a relabelling of
the JSON on the volume); the unchanged reducers are pointed at it by setting their module-level
`HERE`, which is the only thing either of them reads its location from.

Usage (from experiments/):
    python3 rhm/practice/voicing/sotto_voce/aliquot/preplay/timbre/reduce_timbre.py --tag tb1 --fetch
"""

import argparse
import copy
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
ARM_ORDER = ["s0_sv", "s2_sv", "s0_so", "s2_so", "s0_yd", "s2_yd"]
RECORD = ["s0_sv", "s2_sv"]
TRUNKS = ("shaped", "frozen", "twin")
VIEWS1 = ("ridge", "belief", "bonly_lin", "mlp", "bonly", "slot")
VIEWS5 = ("ridge", "ridge_refit", "belief", "mlp", "bonly")
FORM_LABEL = {"ridge": "ridge (the projection's form; arm 0)", "belief": "ridge + belief",
              "bonly_lin": "ridge on the belief alone", "mlp": "MLP on the state",
              "bonly": "MLP on the belief alone", "slot": "per-slot ridge",
              "ridge_refit": "ridge, refit on the shaped trunk"}


def _load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def vo_auc(scores, labels):
    y = np.asarray(labels, np.float64)
    x = np.asarray(scores, np.float64)
    n1, n0 = float((y > 0.5).sum()), float((y <= 0.5).sum())
    if n1 < 1 or n0 < 1:
        return None
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1)
    xs = x[order]
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return float((ranks[y > 0.5].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def spearman(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    mk = np.isfinite(a) & np.isfinite(b)
    a, b = a[mk], b[mk]
    if a.size < 4:
        return None

    def rk(x):
        o = np.argsort(x, kind="mergesort")
        r = np.empty_like(o, dtype=np.float64)
        r[o] = np.arange(1, len(x) + 1)
        xs = x[o]
        i = 0
        while i < len(xs):
            j = i
            while j + 1 < len(xs) and xs[j + 1] == xs[i]:
                j += 1
            if j > i:
                r[o[i:j + 1]] = (i + j + 2) / 2.0
            i = j + 1
        return r
    ra, rb = rk(a), rk(b)
    if ra.std() < 1e-12 or rb.std() < 1e-12:
        return None
    return float(((ra - ra.mean()) * (rb - rb.mean())).mean() / (ra.std() * rb.std()))


def f(q, w=6, p=3):
    return (" " * (w - 2) + "--") if q is None else f"{q:>{w}.{p}f}"


def fetch(tag):
    dst = os.path.join(HERE, "figures")
    os.makedirs(dst, exist_ok=True)
    cmd = ["modal", "volume", "get", "--force", "rhm-scaling-data",
           f"rhm_practice_preplay/{tag}", dst]
    env = dict(os.environ, MODAL_PROFILE=os.environ.get("MODAL_PROFILE", "chromatic"))
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


# --------------------------------------------------------------------------------------- #
# the views
# --------------------------------------------------------------------------------------- #

def key_for(view, tk, cell_keys=None):
    return f"{view}@{tk}"


def view1(arm, R):
    """rr1 -> pp1's schema with reader R in the refit slots. None if R is absent."""
    m = {"banked": "banked"}
    for tk, nm in zip(TRUNKS, ("shaped_refit", "frozen", "twin")):
        m[nm] = f"{R}@{tk}"
    o = {k: copy.deepcopy(arm[k]) for k in ("arm", "spec", "gate", "fingerprints", "n_bank",
                                               "gate_f1", "n_auditions", "sec", "peak_rss_mb",
                                               "peak_gpu_mb", "cfg")}
    hold, fits = {"banked": arm["hold_read"]["banked"]}, {}
    heads = dict(arm["heads"])
    cells = {}
    for lv, c in arm["cells"].items():
        if not all(m[nm] in c["keys"] for nm in m):
            continue
        cc = {k: copy.deepcopy(v) for k, v in c.items()
              if k not in ("single", "delta", "instance", "bases", "ceiling", "slot", "keys")}
        for form in ("single", "delta"):
            rows = []
            for q in c[form]:
                r = {k: v for k, v in q.items() if not k.startswith(("p_", "r_"))}
                for nm, key in m.items():
                    r[f"p_{nm}"] = q[f"p_{key}"]
                    if form == "delta":
                        r[f"r_{nm}"] = q[f"r_{key}"]
                rows.append(r)
            cc[form] = rows
        cc["bases"] = [{"e": q["e"]} for q in c["bases"]]
        cc["ceiling"] = {"e": c["ceiling"]["e"]}
        cc["instance"] = {}
        for form, b in c.get("instance", {}).items():
            bb = {"n": b["n"], "base": b["base"], "auc_dp_lse": b.get("auc_dp_lse")}
            for nm, key in m.items():
                bb[f"auc_{nm}"] = b.get(f"auc_{key}")
                bb[f"pbar_{nm}"] = b.get(f"pbar_{key}")
            cc["instance"][form] = bb
        cells[lv] = cc
        if R == "slot":
            for tk, nm in zip(TRUNKS, ("shaped_refit", "frozen", "twin")):
                if nm not in fits:
                    hold[nm] = c["slot"]["hold"].get(f"slot@{tk}")
                    hh = c["slot"]["heads"].get(f"slot@{tk}", {})
                    fits[nm] = {"lam": hh.get("lam", -1.0), "val_auc": hh.get("val_auc")}
    if not cells:
        return None
    if R != "slot":
        for tk, nm in zip(TRUNKS, ("shaped_refit", "frozen", "twin")):
            hold[nm] = arm["hold_read"].get(f"{R}@{tk}")
            hh = heads.get(f"{R}@{tk}", {})
            lam = hh.get("lam")
            fits[nm] = {"lam": -1.0 if lam is None else lam, "val_auc": hh.get("val_auc")}
    o["hold_read"], o["fits"], o["cells"] = hold, fits, cells
    return o


def merge5(parts):
    """rr5 per (arm, setting) -> one arm."""
    parts = sorted(parts, key=lambda q: q["settings"])
    o = copy.deepcopy(parts[0])
    for q in parts[1:]:
        o["cells"].update(copy.deepcopy(q["cells"]))
        o["P-4"].update(q["P-4"])
        o["gate_t6"].update(q["gate_t6"])
        o["n_auditions"] += q["n_auditions"]
        o["sec"] += q["sec"]
        o["peak_rss_mb"] = max(o["peak_rss_mb"], q["peak_rss_mb"])
        o["peak_gpu_mb"] = max(o["peak_gpu_mb"], q["peak_gpu_mb"])
    o["settings"] = "".join(q["settings"] for q in parts)
    return o


def gate_map5(R):
    """pp5's gate names -> this node's walk names, for view R."""
    if R == "ridge":
        return {g: g for g in ("world", "read", "read_pair", "read_m", "frozen", "twin",
                               "ungated")}
    sh = "ridge@shaped" if R == "ridge_refit" else f"{R}@shaped"
    m = {"world": "world", "ungated": "ungated", "read": sh, "read_pair": sh + "~pair",
         "read_m": sh + "~m"}
    if R == "ridge_refit":
        m.update({"frozen": "frozen", "twin": "twin"})
    else:
        m.update({"frozen": f"{R}@frozen", "twin": f"{R}@twin"})
    return m


def view5(arm, R):
    gm = gate_map5(R)
    o = copy.deepcopy({k: v for k, v in arm.items() if k != "cells"})
    o["cells"] = {}
    ord0 = arm["cfg"]["orders"][0]
    rkey = "banked" if R == "ridge" else gm["read"]
    for ck, c in arm["cells"].items():
        cc = {k: copy.deepcopy(v) for k, v in c.items() if k != "repeats"}
        cc["repeats"] = []
        for rp in c["repeats"]:
            r2 = {k: copy.deepcopy(v) for k, v in rp.items() if k != "orders"}
            r2["delta"] = rp["deltas"][rkey]
            r2["p_base"] = rp["p_bases"][rkey]
            r2["orders"] = {}
            for g, w in gm.items():
                r2["orders"][f"{ord0}|{g}"] = copy.deepcopy(rp["orders"][f"{ord0}|{w}"])
            if R == "ridge":
                for k, v in rp["orders"].items():
                    if not k.startswith(ord0 + "|"):
                        r2["orders"][k] = copy.deepcopy(v)
            cc["repeats"].append(r2)
        o["cells"][ck] = cc
    return o


def run_unchanged(mod_path, argv, here):
    mod = _load_mod(os.path.basename(mod_path)[:-3] + "_timbre", mod_path)
    mod.HERE = here
    old = sys.argv
    sys.argv = [mod_path] + argv
    try:
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            mod.main()
    finally:
        sys.argv = old


# --------------------------------------------------------------------------------------- #
# the cross-reader tables
# --------------------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="tb1")
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--keep-views", default="")
    args = ap.parse_args()
    if args.fetch:
        fetch(args.tag)
    d = os.path.join(HERE, "figures", args.tag)
    # the local mirror is kept gzipped (7x smaller; the volume holds the plain JSON)
    import gzip
    for sub in ("rr1", "rr5"):
        dd = os.path.join(d, sub)
        for fn in (sorted(os.listdir(dd)) if os.path.isdir(dd) else []):
            if fn.endswith(".json"):
                with open(os.path.join(dd, fn), "rb") as fi, \
                        gzip.open(os.path.join(dd, fn + ".gz"), "wb", compresslevel=9) as fo:
                    fo.write(fi.read())
                os.remove(os.path.join(dd, fn))
    arms1, parts5 = {}, {}
    for fn in sorted(os.listdir(os.path.join(d, "rr1"))) if os.path.isdir(
            os.path.join(d, "rr1")) else []:
        if fn.endswith(".json.gz"):
            arms1[fn[:-8]] = json.load(gzip.open(os.path.join(d, "rr1", fn), "rt"))
    for fn in sorted(os.listdir(os.path.join(d, "rr5"))) if os.path.isdir(
            os.path.join(d, "rr5")) else []:
        if fn.endswith(".json.gz"):
            q = json.load(gzip.open(os.path.join(d, "rr5", fn), "rt"))
            parts5.setdefault(q["arm"], []).append(q)
    arms5 = {a: merge5(p) for a, p in parts5.items()}
    o1 = [a for a in ARM_ORDER if a in arms1]
    o5 = [a for a in ARM_ORDER if a in arms5]
    scratch = args.keep_views or tempfile.mkdtemp(prefix="timbre_views_")
    figs = os.path.join(HERE, "figures")

    # ---- 1 and 2: the banked reducers, unchanged, per view -------------------------------- #
    for R in VIEWS1:
        vd = os.path.join(scratch, "figures", f"v1_{R}")
        os.makedirs(vd, exist_ok=True)
        n = 0
        for a in o1:
            vw = view1(arms1[a], R)
            if vw is not None:
                json.dump(vw, open(os.path.join(vd, f"{a}.json"), "w"))
                n += 1
        if n:
            run_unchanged(os.path.join(PARENT, "reduce_preplay.py"),
                          ["--tag", f"v1_{R}", "--out",
                           os.path.join(figs, f"rr1_{R}_reduction.txt")], scratch)
            print(f"[written] figures/rr1_{R}_reduction.txt  ({n} arms)")
    for R in VIEWS5:
        if not o5:
            break
        vd = os.path.join(scratch, "figures", f"v5_{R}")
        os.makedirs(vd, exist_ok=True)
        for a in o5:
            json.dump(view5(arms5[a], R), open(os.path.join(vd, f"{a}.json"), "w"))
        run_unchanged(os.path.join(PARENT, "reduce_readgate.py"),
                      ["--tag", f"v5_{R}", "--no-append"], scratch)
        shutil.copy(os.path.join(scratch, "figures", "readgate_reduction.txt"),
                    os.path.join(figs, f"rr5_{R}_reduction.txt"))
        print(f"[written] figures/rr5_{R}_reduction.txt")
    if not args.keep_views:
        shutil.rmtree(scratch, ignore_errors=True)

    # ---- 3: the cross-reader tables ---------------------------------------------------- #
    out = []

    def o(line=""):
        out.append(line)

    forms1 = [R for R in VIEWS1 if any(f"{R}@shaped" in c["keys"]
                                       for a in o1 for c in arms1[a]["cells"].values())]
    W = 118
    o("=" * W)
    o(f"[timbre] tag {args.tag} — pp1, pp3 and pp5 re-read on richer reader forms. FACTS ONLY.")
    o(f"         rr1 arms: {', '.join(o1)}   rr5 arms: {', '.join(o5)}")
    o("         Every reader is refit on pp1's shared bank subsample (8192 train / 2048 val rows)")
    o("         through the shaped, frozen (overtone's, same seed) and never-trained twin trunks;")
    o("         `banked` is the arm's own in-loop projection. Seeds side by side, never averaged.")
    o("=" * W)

    # [T0] gates
    o("")
    o("[T0] THE GATES OF RECORD (in-container; `gates_t` T-1..T-4, T-7 and `falsify_t` 7/7 are")
    o("     in NOTES). T-5: arm 0 (banked + ridge@shaped/frozen/twin) against the banked pp1 and")
    o("     pp3 columns, per candidate and per row. T-6: pp5's seven banked gates' walks against")
    o("     the banked pp5 JSON, exactly. `leak` counts fired / pool configurations equal to a")
    o("     bank row any reader was fit on (the pricing, gate and test pools are disjoint from")
    o("     the fit rows only if this is 0).")
    o("")
    o(f"    {'arm':7} {'F-2b |d|':>9} {'T-5 pp1 levels':>16} {'worst |dp|':>11} "
      f"{'T-5 pp3 levels':>15} {'T-6 cells':>10} {'leak fit/bank of n':>28}")
    for a in ARM_ORDER:
        if a not in arms1 and a not in arms5:
            continue
        g1 = arms1.get(a, {}).get("gate_t5", {})
        p1 = g1.get("pp1", {})
        p3 = g1.get("pp3", {})
        wd = max([max([v for k, v in q["worst"].items() if ":p_" in k or k.startswith("p_")]
                      or [0.0]) for q in list(p1.values()) + list(p3.values())] or [0.0])
        s1 = f"{sum(q['pass'] for q in p1.values())}/{len(p1)}" if p1 else "--"
        s3 = f"{sum(q['pass'] for q in p3.values())}/{len(p3)}" if p3 else "--"
        g6 = arms5.get(a, {}).get("gate_t6", {})
        s6 = f"{sum(q['pass'] for q in g6.values())}/{len(g6)}" if g6 else "--"
        lk = [0, 0, 0]
        for c in list(arms1.get(a, {}).get("cells", {}).values()) + \
                list(arms1.get(a, {}).get("own", {}).values()) + \
                list(arms5.get(a, {}).get("cells", {}).values()):
            q = c.get("leak", {})
            lk[0] += q.get("fit", 0) + q.get("pools_fit", 0)
            lk[1] += q.get("bank", 0)
            lk[2] += q.get("n", 0)
        gb = (arms1.get(a) or arms5.get(a))["gate"]
        o(f"    {a:7} {gb['F-2b:delta']:>9.4f} {s1:>16} {wd:>11.1e} {s3:>15} {s6:>10} "
          f"{lk[0]:>10d}/{lk[1]:<6d} of {lk[2]:<9d}")

    # [T1] the readers on their home ground
    o("")
    o("=" * W)
    o("[T1] THE READERS ON THE BANK'S OWN HELD-OUT ROWS (the level on its own diet): AUC against")
    o("     the verdict on the 2048 hold rows, beside the validation AUC each reader was selected")
    o("     on. MLP rows: the selected (lr, wd, stopping step).")
    o("")
    o(f"    {'arm':7} {'reader':>10} " + " ".join(f"{tk + ' hold':>12} {'val':>6}" for tk in TRUNKS)
      + "   selection (shaped / frozen / twin)")
    for a in o1:
        A = arms1[a]
        o(f"    {a:7} {'banked':>10} {f(A['hold_read']['banked'], 12, 4)}")
        for R in ("ridge", "belief", "bonly_lin", "mlp", "bonly"):
            if f"{R}@shaped" not in A["heads"]:
                continue
            cells_, sel = [], []
            for tk in TRUNKS:
                hh = A["heads"][f"{R}@{tk}"]
                cells_.append(f"{f(A['hold_read'].get(f'{R}@{tk}'), 12, 4)} "
                              f"{f(hh.get('val_auc'), 6, 3)}")
                sel.append(f"lam {hh['lam']:.0f}" if hh["kind"] == "lin"
                           else f"{hh.get('lr', 0):g}/{hh['wd']:g}@{hh['step']}")
            o(f"    {a:7} {R:>10} " + " ".join(cells_) + "   " + " / ".join(sel))
        sl = [(lv, c["slot"]) for lv, c in A["cells"].items() if c["slot"]["heads"]]
        for lv, sq in sl:
            o(f"    {a:7} {'slot L' + lv:>10} " + " ".join(
                f"{f(sq['hold'].get(f'slot@{tk}'), 12, 4)} "
                f"{f(sq['heads'].get(f'slot@{tk}', {}).get('val_auc'), 6, 3)}" for tk in TRUNKS)
              + f"   n_tr {sq['info']['n_tr']}  n_va {sq['info']['n_va']}  "
              f"n_hold {sq['info']['n_hold']}")

    # [T2] pp1 single form: true-vs-wrong AUC by reader x trunk x level
    o("")
    o("=" * W)
    o("[T2] pp1, SINGLE FORM — AUC separating TRUE from WRONG candidate entries by the read's mean")
    o("     level over the fired instances (pp1's §2 table), every reader through every trunk.")
    o("     `order` counts the cells where shaped > frozen > twin (strict), and where shaped >")
    o("     max(frozen, twin). Arms of record first; the four extra arms follow.")
    o("")
    for grp, arms_ in (("arms of record", [a for a in o1 if a in RECORD]),
                       ("extra arms", [a for a in o1 if a not in RECORD])):
        if not arms_:
            continue
        o(f"  --- {grp} ---")
        o(f"    {'arm':7} {'L':>2} {'n T/W':>7} {'world':>6} {'dp_top':>6} {'banked':>6} | " +
          " | ".join(f"{R[:9]:>9} s/f/t" for R in forms1))
        cnt = {R: [0, 0, 0] for R in forms1}
        for a in arms_:
            for lv, c in sorted(arms1[a]["cells"].items(), key=lambda q: int(q[0])):
                sg = c["single"]
                cls = np.array([1.0 if q["cls"] == "true" else 0.0 for q in sg])
                W_ = np.array([q["w_succ"] for q in sg])
                segs = []
                for R in forms1:
                    if f"{R}@shaped" not in c["keys"]:
                        segs.append(" " * 21)
                        continue
                    v3 = [vo_auc(np.array([q[f"p_{R}@{tk}"] for q in sg]), cls) for tk in TRUNKS]
                    cnt[R][2] += 1
                    if v3[0] > v3[1] > v3[2]:
                        cnt[R][0] += 1
                    if v3[0] > max(v3[1], v3[2]):
                        cnt[R][1] += 1
                    segs.append(f"{v3[0]:.3f}/{v3[1]:.3f}/{v3[2]:.3f}")
                o(f"    {a:7} {lv:>2} {int(cls.sum()):>3d}/{int((1 - cls).sum()):<3d} "
                  f"{vo_auc(W_, cls):>6.3f} "
                  f"{vo_auc(np.array([q['dp_top'] for q in sg]), cls):>6.3f} "
                  f"{vo_auc(np.array([q['p_banked'] for q in sg]), cls):>6.3f} | " +
                  " | ".join(f"{s_:>21}" for s_ in segs))
        o(f"    {'order: s>f>t (s>max(f,t)) of cells':>44} " +
          "  ".join(f"{R}: {cnt[R][0]} ({cnt[R][1]}) of {cnt[R][2]}" for R in forms1))
        o("")

    # [T3] Spearman with the world's price, and within the true class
    o("=" * W)
    o("[T3] pp1, SINGLE FORM — rank correlation of the read's price with the world's, over all")
    o("     candidates (rho) and within the TRUE class alone (rhoT: every candidate legal, only its")
    o("     fit to the cell's instances differs). `pos` counts cells with rhoT > 0 over every arm")
    o("     present; the ranges are over the arms of record.")
    o("")
    o(f"    {'reader':>10} {'trunk':>7} {'rho range (record)':>20} {'rhoT range (record)':>20} "
      f"{'rhoT>0 (all arms)':>18}   rhoT per cell of record (s0 L2..L5 | s2 L2..L5)")
    for R in ["banked"] + forms1:
        for tk in (("shaped",) if R == "banked" else TRUNKS):
            kk = "banked" if R == "banked" else f"{R}@{tk}"
            rr_, rt_, per = [], [], []
            pos, tot = 0, 0
            for a in o1:
                for lv, c in sorted(arms1[a]["cells"].items(), key=lambda q: int(q[0])):
                    if kk not in c["keys"]:
                        continue
                    sg = c["single"]
                    W_ = np.array([q["w_succ"] for q in sg])
                    P_ = np.array([q[f"p_{kk}"] for q in sg])
                    tm = np.array([q["cls"] == "true" for q in sg])
                    rho = spearman(P_, W_)
                    rt = spearman(P_[tm], W_[tm])
                    if rt is not None:
                        tot += 1
                        pos += int(rt > 0)
                    if a in RECORD:
                        rr_.append(rho)
                        rt_.append(rt)
                        per.append(rt)
            rng = (lambda z: "--" if not [q for q in z if q is not None] else
                   f"{min(q for q in z if q is not None):+.2f}..{max(q for q in z if q is not None):+.2f}")
            o(f"    {R:>10} {tk:>7} {rng(rr_):>20} {rng(rt_):>20} {f'{pos}/{tot}':>18}   " +
              " ".join(f(q, 5, 2) for q in per))
    o("")

    # [T4] the level and its compression
    o("=" * W)
    o("[T4] pp1, SINGLE FORM — THE LEVEL. Mean read over TRUE and WRONG candidates, beside the")
    o("     world's repair rate, and the compression ratio (p_T - p_W) / (W_T - W_W). Arms of")
    o("     record, the shaped trunk (frozen / twin columns: the same ratio through those trunks).")
    o("")
    o(f"    {'arm':7} {'L':>2} {'W:T':>6} {'W:W':>6} | " +
      " | ".join(f"{R[:9]:>9} T / W  ratio s/f/t" for R in ["banked"] + forms1))
    for a in [q for q in o1 if q in RECORD]:
        for lv, c in sorted(arms1[a]["cells"].items(), key=lambda q: int(q[0])):
            sg = c["single"]
            tm = np.array([q["cls"] == "true" for q in sg])
            W_ = np.array([q["w_succ"] for q in sg])
            dW = W_[tm].mean() - W_[~tm].mean()
            segs = []
            for R in ["banked"] + forms1:
                if R != "banked" and f"{R}@shaped" not in c["keys"]:
                    segs.append(" " * 36)
                    continue
                ks = ["banked"] if R == "banked" else [f"{R}@{tk}" for tk in TRUNKS]
                P0 = np.array([q[f"p_{ks[0]}"] for q in sg])
                rat = []
                for k in ks:
                    P_ = np.array([q[f"p_{k}"] for q in sg])
                    rat.append((P_[tm].mean() - P_[~tm].mean()) / dW)
                segs.append(f"{P0[tm].mean():.3f}/{P0[~tm].mean():.3f} " +
                            "/".join(f"{q:+.2f}" for q in rat))
            o(f"    {a:7} {lv:>2} {W_[tm].mean():>6.3f} {W_[~tm].mean():>6.3f} | " +
              " | ".join(f"{s_:>36}" for s_ in segs))
    o("")

    # [T5] per-instance transfer
    o("=" * W)
    o("[T5] pp1, SINGLE FORM — PER FIRED INSTANCE: AUC of each read against the world's success")
    o("     on that instance, pooled over the cell's candidates (pp1's [I]); arms of record.")
    o("")
    o(f"    {'arm':7} {'L':>2} {'banked':>6} | " + " | ".join(f"{R[:9]:>9} s/f/t" for R in forms1))
    for a in [q for q in o1 if q in RECORD]:
        for lv, c in sorted(arms1[a]["cells"].items(), key=lambda q: int(q[0])):
            b = c["instance"].get("single", {})
            segs = []
            for R in forms1:
                if f"{R}@shaped" not in c["keys"]:
                    segs.append(" " * 21)
                    continue
                segs.append("/".join(f"{b.get(f'auc_{R}@{tk}') or 0:.3f}" for tk in TRUNKS))
            o(f"    {a:7} {lv:>2} {f(b.get('auc_banked'), 6, 3)} | " +
              " | ".join(f"{s_:>21}" for s_ in segs))
    o("")

    # [T6] pp3
    has_own = any(arms1[a].get("own") for a in o1)
    if has_own:
        o("=" * W)
        o("[T6] pp3 STEP 2 — the learner's own operative rows, each fired alone and read: AUC")
        o("     separating the grammar's TRUE rows from its FALSE ones, and Spearman of the read's")
        o("     price with the world's over all rows (pp3's §4 table), every reader x trunk.")
        o("")
        for a in o1:
            A = arms1[a]
            if not A.get("own"):
                continue
            o(f"  --- {a} ---")
            o(f"    {'reader':>10} {'trunk':>7} " +
              " ".join(f"{'L' + lv + ' sep':>8} {'rho':>6}" for lv in sorted(A["own"], key=int)))
            lvls = sorted(A["own"], key=int)
            rowsW = {lv: np.array([q["w_succ"] for q in A["own"][lv]["rows"]]) for lv in lvls}
            rowsT = {lv: np.array([q["truth"] for q in A["own"][lv]["rows"]], float)
                     for lv in lvls}
            o(f"    {'world':>10} {'':>7} " + " ".join(
                f"{f(vo_auc(rowsW[lv], rowsT[lv]), 8, 3)} {'--':>6}" for lv in lvls))
            o(f"    {'dp_top':>10} {'':>7} " + " ".join(
                f"{f(vo_auc(np.array([q['dp_top'] for q in A['own'][lv]['rows']]), rowsT[lv]), 8, 3)} "
                f"{f(spearman(np.array([q['dp_top'] for q in A['own'][lv]['rows']]), rowsW[lv]), 6, 3)}"
                for lv in lvls))
            for R in ["banked"] + forms1:
                for tk in (("",) if R == "banked" else TRUNKS):
                    k = "banked" if R == "banked" else f"{R}@{tk}"
                    segs = []
                    for lv in lvls:
                        c = A["own"][lv]
                        if k not in c["keys"]:
                            segs.append(f"{'':>8} {'':>6}")
                            continue
                        P_ = np.array([q[f"p_{k}"] for q in c["rows"]])
                        segs.append(f"{f(vo_auc(P_, rowsT[lv]), 8, 3)} "
                                    f"{f(spearman(P_, rowsW[lv]), 6, 3)}")
                    o(f"    {R:>10} {tk:>7} " + " ".join(segs))
            o("")

    # [T7] pp5 by reader, pooled and per seed
    if o5:
        o("=" * W)
        o("[T7] pp5 — EACH READER'S LEVEL AS THE GATE (dp_top the order). Median over (cell,")
        o("     repeat) of the world's test error of the table built, capture = (ungated - gate) /")
        o("     (ungated - world) on the SAME cells, and the composition of the disagreement with")
        o("     the world gate on the same trials (TF = junk admitted, FT = good refused, shares of")
        o("     candidates offered). `pooled` is pp5's own 28-cell pooling (both seeds); the seed")
        o("     columns are the same over each seed's 14 cells. `paired == pooled` counts the cells")
        o("     where the paired form's admissions equal the pooled form's.")
        o("")
        ord0 = arms5[o5[0]]["cfg"]["orders"][0]
        gates_all = arms5[o5[0]]["cfg"]["gates"]
        pool_rows = {}
        for a in o5:
            for ck, c in arms5[a]["cells"].items():
                for rp in c["repeats"]:
                    for g in gates_all:
                        r = rp["orders"][f"{ord0}|{g}"]
                        bl = sorted(int(b) for b in r["budget"])
                        cf = r["conf"]
                        n = max(1, sum(cf.values()))
                        pool_rows.setdefault(g, []).append(
                            (a, r["budget"]["8"]["e_test"], r["budget"][str(bl[-1])]["e_test"],
                             cf["TF"] / n, cf["FT"] / n, r["admitted"]))
        label = {"read": "banked", "read_m": "banked~m", "read_pair": "banked~pair",
                 "frozen": "ridge@frozen", "twin": "ridge@twin"}

        def summ(sel):
            med = {}
            for g, rows in pool_rows.items():
                z = np.array([[q[1], q[2], q[3], q[4]] for q in rows if sel(q[0])], float)
                if z.size:
                    med[g] = np.median(z, 0)
            return med
        meds = {"pooled": summ(lambda a: True)}
        for a in o5:
            meds[a] = summ(lambda x, a=a: x == a)

        def cap(med, g, j):
            dd = med["ungated"][j] - med["world"][j]
            return None if abs(dd) < 1e-9 else (med["ungated"][j] - med[g][j]) / dd
        o(f"    {'gate':>18} | " + " | ".join(
            f"{k:>6}: e@8 cap8  e@full capF  TF    FT  " for k in ["pooled"] + o5))
        for g in gates_all:
            if g.endswith("~pair") or g == "read_pair":
                continue
            segs = []
            for k in ["pooled"] + o5:
                md = meds[k]
                if g not in md:
                    segs.append(" " * 44)
                    continue
                m_ = md[g]
                segs.append(f"{'':>6}  {m_[0]:.3f} {f(cap(md, g, 0), 5, 2)} {m_[1]:.3f} "
                            f"{f(cap(md, g, 1), 5, 2)} {m_[2]:.3f} {m_[3]:.3f}")
            o(f"    {label.get(g, g):>18} | " + " | ".join(segs))
        o("")
        n_id, n_tot = 0, 0
        for g in gates_all:
            if not g.endswith("~pair"):
                continue
            base_g = g[:-5]
            for rows_p, rows_b in zip(pool_rows[g], pool_rows[base_g]):
                n_tot += 1
                n_id += int(rows_p[5] == rows_b[5])
        for rows_p, rows_b in zip(pool_rows["read_pair"], pool_rows["read"]):
            n_tot += 1
            n_id += int(rows_p[5] == rows_b[5])
        o(f"    paired == pooled admissions: {n_id} of {n_tot} (reader, cell, repeat) walks")
        o("")
        o("    PER (cell, repeat), not medians: how often each gate's table beats NO GATE on the")
        o("    test pools (e_gate < e_ungated - 0.005), ties it (within 0.005) or loses, at b = 8 and")
        o("    at the full walk, per seed; and how often it is within 0.005 of the WORLD gate's.")
        o(f"    {'gate':>18} | " + " | ".join(f"{a:>6} b8 W/T/L  full W/T/L  =world b8/full"
                                             for a in o5))
        for g in gates_all:
            if g.endswith("~pair") or g == "read_pair":
                continue
            segs = []
            for a in o5:
                wtl8, wtlf, eq8, eqf, n = [0, 0, 0], [0, 0, 0], 0, 0, 0
                for ck, c in arms5[a]["cells"].items():
                    for rp in c["repeats"]:
                        def e_(gg, fin):
                            r = rp["orders"][f"{ord0}|{gg}"]
                            bl = sorted(int(b) for b in r["budget"])
                            return r["budget"][str(bl[-1]) if fin else "8"]["e_test"]
                        n += 1
                        for fin, acc in ((False, wtl8), (True, wtlf)):
                            dg = e_(g, fin) - e_("ungated", fin)
                            acc[0 if dg < -0.005 else (1 if abs(dg) <= 0.005 else 2)] += 1
                        eq8 += int(abs(e_(g, False) - e_("world", False)) <= 0.005)
                        eqf += int(abs(e_(g, True) - e_("world", True)) <= 0.005)
                segs.append(f"{'':>6} {'/'.join(map(str, wtl8)):>8}  {'/'.join(map(str, wtlf)):>9}  "
                            f"{eq8:>5}/{eqf:<3} of {n}")
            o(f"    {label.get(g, g):>18} | " + " | ".join(segs))
        o("")
        # the order AUCs, per reader: the read as a price of the candidates on the pricing pool
        o("    the read as a PRICE on pp5's pricing pool (n 512): AUC true-vs-wrong / Spearman with")
        o("    the world's price, per cell; setting (a) constructed, (b) the learner's own rows")
        rk = arms5[o5[0]]["rkeys"]
        o(f"    {'arm':7} {'cell':>5} {'world':>6} {'dp_top':>6} " +
          " ".join(f"{k[:14]:>14}" for k in rk))
        for a in o5:
            for ck in sorted(arms5[a]["cells"]):
                c = arms5[a]["cells"][ck]
                oa, orho = c["order_auc"], c["order_rho"]
                o(f"    {a:7} {ck:>5} {f(oa['world'], 6, 3)} {f(oa['dp_top'], 6, 3)} " +
                  " ".join(f"{f(oa.get(k), 6, 3)}/{f(orho.get(k), 6, 2)}".rjust(14) for k in rk))
        o("")

    # [T8] the bill
    o("=" * W)
    o("[T8] THE BILL")
    tot = 0.0
    for a in o1:
        A = arms1[a]
        tot += A["sec"]
        o(f"    rr1 {a:7} {A['sec']:>7.1f}s  auditions {A['n_auditions']:>6d}  peak RSS "
          f"{A['peak_rss_mb']:.0f} MB  GPU {A['peak_gpu_mb']:.0f} MB")
    for a in o5:
        A = arms5[a]
        tot += A["sec"]
        o(f"    rr5 {a:7} {A['sec']:>7.1f}s  auditions {A['n_auditions']:>6d}  peak RSS "
          f"{A['peak_rss_mb']:.0f} MB  GPU {A['peak_gpu_mb']:.0f} MB  (settings {A['settings']})")
    o(f"    TOTAL {tot:.0f} container-seconds = {tot / 3600:.2f} GPU-h")
    o("")
    txt = "\n".join(out) + "\n"
    dst = os.path.join(figs, "timbre_reduction.txt")
    open(dst, "w").write(txt)
    print(txt)
    print(f"[written] {dst}")


if __name__ == "__main__":
    main()
