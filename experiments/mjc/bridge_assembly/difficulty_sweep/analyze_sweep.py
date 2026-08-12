"""Aggregation for the difficulty x benchmark-timescale sweep (local, no compute).

Reads each cell's `results.json` (parent runner output) and produces, per cell:

  * PRECONDITIONS  -- stale/ceiling separation per readout and per region class, so a
    nominally-harder cell that merely saturated the eval is visible as such.
  * OUTCOME        -- Delta(T') = AUC_fixed(T') - AUC_delta(T') for every milestone
    truncation T' (positive = delta beats uniform), on each control readout and on the
    FM-level task composite. The milestone ladder makes the BUDGET a free axis: the
    ladder entry at m is exactly the state after m transitions.
  * MECHANISM      -- the benchmark-lag statistics on the drift class:
       Lambda_lag(T')   = < (b - e) / (e - e_ceil) >_{t<=T'}   [tracking / frontier ratio]
       Lambda_alloc(T') = < w_A >_{t<=T'} / < w_all >          [realized allocation ratio]
       f_boost(T')      = fraction of batches with delta_A < 0
    plus the noise-region over-weighting (delta's fixed tax) and the retention weight.

Usage (from experiments/):
    python3 mjc/bridge_assembly/difficulty_sweep/analyze_sweep.py --pull
    python3 mjc/bridge_assembly/difficulty_sweep/analyze_sweep.py --figures
"""

import argparse
import json
import os
import subprocess

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT_FIG = os.path.join(os.path.dirname(HERE), "figures")
DATA = os.path.join(HERE, "data")
VOLUME = "mujoco-control-data"

# cell -> (tag_prefix, label, difficulty-order key).  phi12 reuses the parent's own runs.
CELLS = [
    ("phi06",        "asm-like phi=0.6",       0),
    ("phi12",        "phi=1.2 (parent cell)",  1),
    ("phi20",        "phi=2.0",                2),
    ("phi28",        "phi=2.8",                3),
    ("two12",        "2x phi=+-1.2",           4),
    ("two24",        "2x phi=+-2.4",           5),
    ("phi12_blr1e3", "phi=1.2, blr 1e-3",      1),
    ("phi12_blr1e2", "phi=1.2, blr 1e-2",      1),
    ("two12_blr1e3", "2x1.2, blr 1e-3",        4),
    ("two12_blr1e2", "2x1.2, blr 1e-2",        4),
    ("phi12_alr9e4", "phi=1.2, alr 9e-4",     1),
    ("two12_alr9e4", "2x1.2, alr 9e-4",       4),
    ("two12_nonoise", "2x1.2, clean decoy",   4),
]

# phi12 == the parent node's own asm_s{0,1,2} runs (identical config); do not re-run.
PARENT_TAGS = {"phi12": ["asm_s0", "asm_s1", "asm_s2"]}


def tags_for(cell, seeds):
    if cell in PARENT_TAGS:
        return [t for t in PARENT_TAGS[cell] if int(t[-1]) in seeds]
    return [f"dsw_{cell}_s{s}" for s in seeds]


def path_for(tag):
    for p in (os.path.join(DATA, f"{tag}.json"),
              os.path.join(PARENT_FIG, f"bridge_assembly_{tag}", "results.json")):
        if os.path.exists(p):
            return p
    return None


def pull(tag):
    os.makedirs(DATA, exist_ok=True)
    dst = os.path.join(DATA, f"{tag}.json")
    r = subprocess.run(["modal", "volume", "get", "--force", VOLUME,
                        f"bridge_assembly/{tag}/results.json", dst],
                       capture_output=True, text=True)
    return r.returncode == 0


def load(tag, do_pull=False):
    p = path_for(tag)
    if do_pull and p is None:          # only fetch when explicitly asked (--pull); a
        if pull(tag):                  # mid-run checkpoint on the volume is incomplete
            p = os.path.join(DATA, f"{tag}.json")   # and would otherwise litter data/
    if p is None or not os.path.exists(p):
        return None
    with open(p) as fh:
        R = json.load(fh)
    return R if R.get("complete") else None


# ------------------------------------------------------------------ helpers
def drift_classes(R):
    """Post-drift ROTATED regions only -- excludes the pre-drift mastered region and the
    off-path decoy (which is phi=0, whether it is noisy or, in the nonoise control, clean)."""
    return [r["name"] for r in R["config"]["regions"]
            if r["noise"] == 0 and not r["pre"] and r["phi"] != 0.0]


def mastered_class(R):
    return [r["name"] for r in R["config"]["regions"] if r["pre"]][0]


def noise_class(R):
    """The irreducible decoy region, or (two12_nonoise control) the clean region in
    the same place -- so allocation to that slot stays comparable across cells."""
    n = [r["name"] for r in R["config"]["regions"] if r["noise"] > 0]
    if n:
        return n[0]
    return [r["name"] for r in R["config"]["regions"]
            if not r["pre"] and r["phi"] == 0.0][0]


def fm_of(rec, names):
    return float(np.mean([rec["fm"][n] for n in names]))


def ladder_series(R, arm, key):
    """(milestones>0, value) for a control key, or for fm:<composite>."""
    lad = R["arms"][arm]["ladder"]
    ms, ys = [], []
    for r in lad:
        if r["transitions"] <= 0:
            continue
        ms.append(r["transitions"])
        if key.startswith("fm:"):
            what = key[3:]
            if what == "task":
                ys.append(0.5 * (fm_of(r, drift_classes(R)) + fm_of(r, [mastered_class(R)])))
            elif what == "drift":
                ys.append(fm_of(r, drift_classes(R)))
            elif what == "mastered":
                ys.append(fm_of(r, [mastered_class(R)]))
            else:
                ys.append(r["fm"][what])
        else:
            ys.append(r.get(key, np.nan))
    return np.array(ms, float), np.array(ys, float)


def auc_trunc(ms, ys, T):
    m = ms <= T + 1e-9
    return float(np.nanmean(ys[m])) if m.any() else np.nan


def blog_arr(R, arm, q, names):
    bl = R["arms"][arm]["blog"]
    return np.nanmean([np.array(bl[f"{q}_{n}"], float) for n in names], axis=0)


def lambda_series(R):
    """Per-batch lag ratio and allocation ratio on the drift class (delta arm)."""
    bl = R["arms"]["delta"]["blog"]
    t = np.array(bl["t"], float)
    dnames = drift_classes(R)
    e = blog_arr(R, "delta", "e", dnames)
    b = blog_arr(R, "delta", "b", dnames)
    d = blog_arr(R, "delta", "delta", dnames)
    w = blog_arr(R, "delta", "w", dnames)
    # empirical floor = the FIXED arm's final probe on the drift class (what this run
    # actually achieves); excursion = stale -> floor.  Normalizing the lag by the
    # excursion keeps the statistic finite once e reaches its floor (where the
    # reducible-error denominator collapses and (b-e)/(e-e_inf) legitimately diverges).
    lad = R["arms"]["fixed"]["ladder"]
    floor = fm_of(lad[-1], dnames)
    exc = max(fm_of(R["stale"], dnames) - floor, 1e-4)
    return t, (b - e) / exc, d, w, e


def cell_row(R):
    """All per-cell scalars + budget-resolved series."""
    dnames = drift_classes(R)
    bn = mastered_class(R); rn = noise_class(R)
    t, lam, dlt, wA, eA = lambda_series(R)
    wB = blog_arr(R, "delta", "w", [bn]); wR = blog_arr(R, "delta", "w", [rn])
    wall = np.nanmean(np.stack([blog_arr(R, "delta", "w", [n]) for n in
                                dnames + [bn, rn, "base"]]), axis=0)
    out = {
        "bench_lr": R["config"]["bench_lr"],
        "adapt_lr": R["config"]["adapt_lr"],
        "r_ratio": R["config"]["adapt_lr"] / R["config"]["bench_lr"],
        "phi": [r["phi"] for r in R["config"]["regions"]
                if r["noise"] == 0 and not r["pre"] and r["phi"] != 0.0],
        "n_drift": len(dnames),
        "stale_fm_drift": fm_of(R["stale"], dnames),
        "ceil_fm_drift": fm_of(R["ceiling"], dnames),
        "stale_fm_mast": fm_of(R["stale"], [bn]),
        "budget": {a: R["arms"][a]["budget"] for a in R["arms"]},
    }
    for k in ("drift/ballistic_cem", "drift/reactive", "mastered/ballistic_cem",
              "mastered/reactive", "agg/ballistic_cem", "agg/reactive"):
        out[f"stale_{k}"] = R["stale"].get(k, np.nan)
        out[f"ceil_{k}"] = R["ceiling"].get(k, np.nan)
    ms, _ = ladder_series(R, "fixed", "agg/reactive")
    out["milestones"] = ms.tolist()
    out["delta_by_T"] = {}
    keys = ["drift/ballistic_cem", "drift/reactive", "agg/ballistic_cem", "agg/reactive",
            "mastered/ballistic_cem", "mastered/reactive",
            "fm:task", "fm:drift", "fm:mastered", "fm:base", "fm:global"]
    for key in keys:
        _, yf = ladder_series(R, "fixed", key)
        _, yd = ladder_series(R, "delta", key)
        _, yr = ladder_series(R, "raw_err", key)
        out["delta_by_T"][key] = {
            "fixed": [auc_trunc(ms, yf, T) for T in ms],
            "delta": [auc_trunc(ms, yd, T) for T in ms],
            "raw":   [auc_trunc(ms, yr, T) for T in ms],
            "inst_fixed": yf.tolist(), "inst_delta": yd.tolist(), "inst_raw": yr.tolist(),
        }
    frac = R["calibration"]["stream_class_fractions"]
    allc = dnames + [bn, rn, "base"]
    Wmix = np.nansum(np.stack([np.nan_to_num(blog_arr(R, "delta", "w", [c]), nan=1.0)
                               * frac[c] for c in allc]), axis=0) / sum(frac[c] for c in allc)
    out["meanw_by_T"] = [float(np.nanmean(Wmix[t <= T])) for T in ms]
    # recovery progress of the FIXED arm on the drift class, 0 (stale) -> 1 (its own floor)
    _, fmd = ladder_series(R, "fixed", "fm:drift")
    e0f = out["stale_fm_drift"]; ef = fmd[-1]
    out["prog_by_T"] = [float((e0f - v) / max(e0f - ef, 1e-6)) for v in fmd]
    out["lam_by_T"] = [float(np.nanmean(lam[t <= T])) for T in ms]
    # Lambda restricted to the DESCENT WINDOW (fixed arm's drift-probe error between 80%
    # and 20% of its excursion above the floor).  Progress-matched, so cells whose clocks
    # run at different absolute speeds (the adapt_lr leg) are compared at the same point
    # of recovery rather than at the same wall-clock budget.
    tr_t = np.array(R["arms"]["fixed"]["trace"]["t"], float)
    tr_e = np.nanmean([np.array(R["arms"]["fixed"]["trace"][f"probe_{n}"], float)
                       for n in dnames], axis=0)
    e0f = out["stale_fm_drift"]
    ffl = float(fm_of(R["arms"]["fixed"]["ladder"][-1], dnames))
    fr = (np.interp(t, tr_t, tr_e) - ffl) / max(e0f - ffl, 1e-9)
    win = (fr <= 0.8) & (fr >= 0.2)
    out["lam_descent"] = float(np.nanmean(lam[win])) if win.any() else np.nan
    out["wA_descent"] = float(np.nanmean(wA[win]) / np.nanmean(wall[win])) if win.any() else np.nan
    out["wA_by_T"] = [float(np.nanmean(wA[t <= T]) / np.nanmean(wall[t <= T])) for T in ms]
    out["fboost_by_T"] = [float(np.nanmean(dlt[t <= T] < 0)) for T in ms]
    out["wR_by_T"] = [float(np.nanmean(wR[t <= T]) / np.nanmean(wall[t <= T])) for T in ms]
    out["wB_by_T"] = [float(np.nanmean(wB[t <= T]) / np.nanmean(wall[t <= T])) for T in ms]
    # recovery timescale of the frontier, from the FIXED arm's dense probe trace
    tr = R["arms"]["fixed"]["trace"]
    tt = np.array(tr["t"], float)
    ee = np.nanmean([np.array(tr[f"probe_{n}"], float) for n in dnames], axis=0)
    e0 = out["stale_fm_drift"]; ec = out["ceil_fm_drift"]
    thr = ec + (e0 - ec) / np.e
    idx = np.where(ee <= thr)[0]
    out["T_rec_fm"] = float(tt[idx[0]]) if len(idx) else float(tt[-1] * 2)
    # and of the control readout (drift/ballistic), from the fixed arm's ladder
    _, yb = ladder_series(R, "fixed", "drift/ballistic_cem")
    s0 = out["stale_drift/ballistic_cem"]; sc = out["ceil_drift/ballistic_cem"]
    thr2 = sc + (s0 - sc) / np.e
    idx2 = np.where(yb <= thr2)[0]
    out["T_rec_ctrl"] = float(ms[idx2[0]]) if len(idx2) else float(ms[-1] * 2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pull", action="store_true")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--cells", default="")
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--btrack", action="store_true",
                    help="decompose b(s)'s estimation error by region class and bench_lr")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",") if s != ""]
    want = [c for c in a.cells.split(",") if c] or [c for c, _, _ in CELLS]
    labels = {c: l for c, l, _ in CELLS}

    rows = {}
    for cell in want:
        for tag in tags_for(cell, seeds):
            R = load(tag, do_pull=a.pull)
            if R is None:
                print(f"[miss] {tag}")
                continue
            rows[(cell, tag)] = cell_row(R)

    if not rows:
        print("no complete runs found"); return

    # ---------------- preconditions ----------------
    print("\n===== PRECONDITIONS (stale -> ceiling separation; is the cell actually harder?) =====")
    hdr = (f"{'cell/tag':22s} {'phi':>14s} {'fm_A stale':>10s} {'fm_A ceil':>10s} "
           f"{'fm_B stale':>10s} | {'drift/bal stale':>15s} {'ceil':>7s} {'range':>7s} | "
           f"{'agg/re stale':>12s} {'ceil':>7s}")
    print(hdr)
    for (cell, tag), r in rows.items():
        print(f"{cell+'/'+tag:22s} {str(r['phi']):>14s} {r['stale_fm_drift']:10.4f} "
              f"{r['ceil_fm_drift']:10.4f} {r['stale_fm_mast']:10.4f} | "
              f"{r['stale_drift/ballistic_cem']:15.4f} {r['ceil_drift/ballistic_cem']:7.4f} "
              f"{r['stale_drift/ballistic_cem']-r['ceil_drift/ballistic_cem']:7.4f} | "
              f"{r['stale_agg/reactive']:12.4f} {r['ceil_agg/reactive']:7.4f}")

    # ---------------- recovery timescales ----------------
    print("\n===== TIMESCALES, delta's realized allocation, and the budget-match check =====")
    print(f"{'cell/tag':22s} {'r=alr/blr':>10s} {'T_rec(FM)':>10s} {'T_rec(ctl)':>11s} "
          f"{'w_A':>6s} {'w_B':>6s} {'w_R':>6s} {'Lam(full)':>10s} {'f_boost':>8s} "
          f"{'meanW@256':>10s} {'@full':>7s}")
    for (cell, tag), r in rows.items():
        print(f"{cell+'/'+tag:22s} {r['r_ratio']:10.3f} {r['T_rec_fm']:10.0f} "
              f"{r['T_rec_ctrl']:11.0f} {r['wA_by_T'][-1]:6.3f} {r['wB_by_T'][-1]:6.3f} "
              f"{r['wR_by_T'][-1]:6.3f} {r['lam_by_T'][-1]:10.3f} {r['fboost_by_T'][-1]:8.3f} "
              f"{r['meanw_by_T'][0]:10.3f} {r['meanw_by_T'][-1]:7.3f}")

    # ---------------- outcome vs budget ----------------
    for key in ("drift/ballistic_cem", "agg/reactive", "fm:task", "fm:drift", "fm:mastered"):
        print(f"\n===== Delta = AUC(fixed) - AUC(delta) on {key}  "
              f"(>0: delta beats uniform), by budget T' =====")
        ms = rows[list(rows)[0]]["milestones"]
        print(f"{'cell/tag':22s} " + "".join(f"{int(m):>9d}" for m in ms))
        for (cell, tag), r in rows.items():
            d = r["delta_by_T"][key]
            vals = [f - dd for f, dd in zip(d["fixed"], d["delta"])]
            print(f"{cell+'/'+tag:22s} " + "".join(f"{v:9.4f}" for v in vals))

    for key in ("drift/ballistic_cem", "agg/reactive"):
        print(f"\n===== INSTANTANEOUS fixed - delta on {key} at each milestone "
              f"(>0: delta better); [prog] = fixed arm's FM recovery fraction =====")
        ms = rows[list(rows)[0]]["milestones"]
        print(f"{'cell/tag':22s} " + "".join(f"{int(m):>9d}" for m in ms))
        for (cell, tag), r in rows.items():
            d = r["delta_by_T"][key]
            vals = [f - dd for f, dd in zip(d["inst_fixed"], d["inst_delta"])]
            print(f"{cell+'/'+tag:22s} " + "".join(f"{v:9.4f}" for v in vals))
            print(f"{'  [prog]':22s} " + "".join(f"{v:9.2f}" for v in r["prog_by_T"]))

    # ---------------- seed-aggregated summary (the headline table) ----------------
    print("\n===== SEED-AGGREGATED SUMMARY  (Delta = AUC_fixed - AUC_delta; >0 = delta beats "
          "uniform; late = milestones >= 2048, where the EWMA budget-match has converged) =====")
    bycell = {}
    for (cell, tag), r in rows.items():
        bycell.setdefault(cell, []).append(r)
    def agg(rs, f):
        v = np.array([f(r) for r in rs], float)
        return v.mean(), (v.std(ddof=1) if len(v) > 1 else np.nan), int((v > 0).sum()), len(v)
    print(f"{'cell':16s} {'n':>2s} {'r':>5s} {'Lam':>7s} {'Lam_dsc':>7s} {'w_A':>6s} "
          f"{'w_R':>6s} {'f_bst':>6s} {'rng':>9s} | {'D_full':>8s} {'sgn':>5s} | "
          f"{'D_late':>8s} {'sgn':>5s} | {'%range':>11s} | {'D_lt(re)':>8s} {'sgn':>5s}")
    for cell in [c for c, _, _ in CELLS if c in bycell]:
        rs = bycell[cell]; ms = np.array(rs[0]["milestones"], float); late = ms >= 2048
        def dfun(key, mask):
            def f(r):
                d = r["delta_by_T"][key]
                v = np.array(d["inst_fixed"], float) - np.array(d["inst_delta"], float)
                return float(np.nanmean(v[mask]))
            return f
        L = agg(rs, lambda r: r["lam_by_T"][-1]); WA = agg(rs, lambda r: r["wA_by_T"][-1])
        WR = agg(rs, lambda r: r["wR_by_T"][-1]); FB = agg(rs, lambda r: r["fboost_by_T"][-1])
        RG = agg(rs, lambda r: r["stale_drift/ballistic_cem"] - r["ceil_drift/ballistic_cem"])
        DF = agg(rs, dfun("drift/ballistic_cem", ms > 0))
        DL = agg(rs, dfun("drift/ballistic_cem", late))
        DA = agg(rs, dfun("agg/reactive", late))
        LD = agg(rs, lambda r: r["lam_descent"])
        # deficit normalised by each seed's own stale->ceiling control range (% of range)
        def pct(r):
            d = r["delta_by_T"]["drift/ballistic_cem"]
            v = np.array(d["inst_fixed"], float) - np.array(d["inst_delta"], float)
            rng = r["stale_drift/ballistic_cem"] - r["ceil_drift/ballistic_cem"]
            return 100.0 * float(np.nanmean(v[late])) / rng
        PC = agg(rs, pct)
        print(f"{cell:16s} {len(rs):2d} {rs[0]['r_ratio']:5.2f} {L[0]:7.3f} {LD[0]:7.3f} "
              f"{WA[0]:6.3f} {WR[0]:6.3f} {FB[0]:6.3f} {RG[0]:9.3f} | {DF[0]:8.4f} "
              f"{DF[2]}/{DF[3]:>2d} | {DL[0]:8.4f} {DL[2]}/{DL[3]:>2d} | "
              f"{PC[0]:6.1f}+-{PC[1]:4.1f} | {DA[0]:8.4f} {DA[2]}/{DA[3]:>2d}")

    if a.btrack:
        btrack_report(rows)

    if a.json:
        with open(a.json, "w") as fh:
            json.dump({f"{c}|{t}": r for (c, t), r in rows.items()}, fh, indent=1)
        print(f"\n[wrote] {a.json}")
    if a.figures:
        make_figures(rows, labels)


def btrack_report(rows):
    """b(s)'s estimation error by region class -- the thing delta's allocation actually is.

    Uses RATIO OF MEANS, not the mean of per-batch ratios: each region holds ~6% of a
    16-sample batch, so per-batch class means are ~1 sample and per-batch ratios are junk.
    Late window = last quarter of the run.
    """
    print("\n===== b(s)'s TRACKING ERROR by region class, late window, (mean b - mean e)/mean e "
          "=====")
    print("     >0: b over-predicts -> delta>0 -> delta WITHDRAWS plasticity from that region")
    print("     <0: b under-predicts -> delta<0 -> delta ADDS plasticity there\n")

    def late(R, names):
        bl = R["arms"]["delta"]["blog"]; t = np.array(bl["t"], float)
        e = np.nanmean([np.array(bl[f"e_{n}"], float) for n in names], 0)
        b = np.nanmean([np.array(bl[f"b_{n}"], float) for n in names], 0)
        m = (t > t[-1] * 0.75) & ~np.isnan(e) & ~np.isnan(b)
        return float(np.mean(b[m])), float(np.mean(e[m]))

    groups = {}
    for (cell, tag), r in rows.items():
        groups.setdefault(round(r["bench_lr"], 6), []).append((cell, tag))
    for blr in sorted(groups, reverse=True):
        print(f"-- bench_lr = {blr:g} " + "-" * 52)
        print(f"   {'class':14s} {'mean e':>8s} {'mean b':>8s} {'(b-e)/e':>16s}  n")
        acc = {}
        for cell, tag in groups[blr]:
            R = load(tag)
            if R is None:
                continue
            sel = {"drift (A)": drift_classes(R), "mastered (B)": [mastered_class(R)],
                   "decoy (R)": [noise_class(R)], "base": ["base"]}
            for k, names in sel.items():
                if k == "decoy (R)" and cell.endswith("nonoise"):
                    k = "decoy (R, CLEAN)"
                b, e = late(R, names)
                acc.setdefault(k, []).append((b, e))
        for k in ("drift (A)", "mastered (B)", "decoy (R)", "decoy (R, CLEAN)", "base"):
            if k not in acc:
                continue
            v = acc[k]; rr = [(b - e) / e for b, e in v]
            sd = np.std(rr, ddof=1) if len(rr) > 1 else float("nan")
            print(f"   {k:14s} {np.mean([e for _, e in v]):8.4f} "
                  f"{np.mean([b for b, _ in v]):8.4f} {np.mean(rr):+8.3f} +- {sd:5.3f}  {len(rr)}")
    # is HuberLoss(delta=1.0) even in its robust regime?
    R = load("asm_s0")
    if R is not None:
        bl = R["arms"]["delta"]["blog"]
        print("\n-- Huber regime check (bench net trains with HuberLoss(delta=1.0)) " + "-" * 8)
        for nm in ("R-noise", "A-drift", "base"):
            d = np.abs(np.array(bl[f"b_{nm}"], float) - np.array(bl[f"e_{nm}"], float))
            d = d[~np.isnan(d)]
            print(f"   {nm:10s} |b-e| p50={np.median(d):.4f} p99={np.percentile(d, 99):.4f}"
                  f"  -> {np.percentile(d, 99) / 1.0:.2f}x of the Huber knee")
        print("   residuals sit far inside the quadratic region, so the benchmark net is "
              "fitting\n   E[e|s] (an L2 target), NOT a conditional median.")


def make_figures(rows, labels):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    outdir = os.path.join(HERE, "figures"); os.makedirs(outdir, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 130, "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False})

    bycell = {}
    for (cell, tag), r in rows.items():
        bycell.setdefault(cell, []).append(r)
    LADDER = [c for c in ("phi06", "phi12", "phi20", "phi28", "two12", "two24") if c in bycell]
    BLR = {"phi12": ["phi12_blr1e2", "phi12", "phi12_blr1e3"],
           "two12": ["two12_blr1e2", "two12", "two12_blr1e3"]}
    C1, C2, C3, C4 = "#7048e8", "#e8590c", "#1971c2", "#2f9e44"

    def ms_of(rs):
        return np.array(rs[0]["milestones"], float)

    def stat(rs, f):
        v = np.array([f(r) for r in rs], float)
        return v.mean(), (v.std(ddof=1) if len(v) > 1 else 0.0), v

    def pct_late(r):
        d = r["delta_by_T"]["drift/ballistic_cem"]
        m = np.array(r["milestones"], float) >= 2048
        v = np.array(d["inst_fixed"], float) - np.array(d["inst_delta"], float)
        rng = r["stale_drift/ballistic_cem"] - r["ceil_drift/ballistic_cem"]
        return 100.0 * float(np.nanmean(v[m])) / rng

    # ---- fig1: the mechanism tracks the difficulty axis -----------------------------
    x = [np.mean([r["stale_drift/ballistic_cem"] - r["ceil_drift/ballistic_cem"]
                  for r in bycell[c]]) for c in LADDER]
    fig, ax = plt.subplots(1, 3, figsize=(14.5, 4.0))
    for i, (k, lab, c, ref) in enumerate([
            ("lam_by_T", r"$\Lambda$  =  $\langle (b-e)/(e_{stale}-e_{floor})\rangle$  on the drift class",
             C1, 0.0),
            ("wA_by_T", r"realized allocation ratio  $w_A$", C2, 1.0),
            ("fboost_by_T", r"$f_{boost}$ = fraction of batches with $\delta_A<0$", C3, None)]):
        m = [stat(bycell[c2], lambda r: r[k][-1])[0] for c2 in LADDER]
        e = [stat(bycell[c2], lambda r: r[k][-1])[1] for c2 in LADDER]
        ax[i].errorbar(x, m, yerr=e, fmt="o-", color=c, lw=2, ms=6, capsize=3)
        for xi, c2 in zip(x, LADDER):
            for r in bycell[c2]:
                ax[i].plot(xi, r[k][-1], ".", color=c, alpha=0.35, ms=7)
        if ref is not None:
            ax[i].axhline(ref, color="#888", ls="--", lw=1)
        ax[i].set_xlabel("recovery difficulty\n(stale - ceiling range, drift/ballistic)")
        ax[i].set_ylabel(lab, fontsize=8)
        for xi, c2 in zip(x, LADDER):
            ax[i].annotate(c2, (xi, m[LADDER.index(c2)]), fontsize=6,
                           textcoords="offset points", xytext=(4, 5))
    fig.suptitle("The benchmark-lag mechanism tracks recovery difficulty monotonically "
                 "(points = seeds; bars = sd over 3 seeds)", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "fig1_mechanism_axis.png")); plt.close(fig)

    # ---- fig2: the behavioural verdict does NOT cross zero --------------------------
    fig, ax = plt.subplots(1, 2, figsize=(11.0, 4.2))
    m = [stat(bycell[c], pct_late)[0] for c in LADDER]
    e = [stat(bycell[c], pct_late)[1] for c in LADDER]
    ax[0].errorbar(x, m, yerr=e, fmt="o-", color=C1, lw=2, ms=6, capsize=3, label="this sweep")
    for xi, c in zip(x, LADDER):
        for r in bycell[c]:
            ax[0].plot(xi, pct_late(r), ".", color=C1, alpha=0.35, ms=7)
    if "two12_nonoise" in bycell:
        xn = np.mean([r["stale_drift/ballistic_cem"] - r["ceil_drift/ballistic_cem"]
                      for r in bycell["two12_nonoise"]])
        mn, en, _ = stat(bycell["two12_nonoise"], pct_late)
        ax[0].errorbar([xn], [mn], yerr=[en], fmt="s", color=C4, ms=8, capsize=3,
                       label="two12, clean decoy (noise tax removed)")
    ax[0].axhline(0, color="#333", ls=":", lw=1.2)
    ax[0].axhline(3.3, color=C2, ls="--", lw=1.2)
    ax[0].text(0.05, 3.5, "plasticity_gain's reported delta win (+3.3% of range, 3/3)",
               fontsize=7, color=C2)
    ax[0].set_xlabel("recovery difficulty (stale - ceiling range)")
    ax[0].set_ylabel("late-window advantage of delta over uniform\n(% of control range, m>=2048)")
    ax[0].legend(fontsize=7)
    key = "drift/ballistic_cem"
    cmapd = plt.get_cmap("viridis")
    for ci, cell in enumerate(LADDER):
        rs = bycell[cell]; ms = ms_of(rs)
        v = np.mean([np.array(r["delta_by_T"][key]["inst_fixed"], float)
                     - np.array(r["delta_by_T"][key]["inst_delta"], float) for r in rs], 0)
        rng = np.mean([r[f"stale_{key}"] - r[f"ceil_{key}"] for r in rs])
        ax[1].plot(ms, 100 * v / rng, "o-", ms=3, lw=1.4, alpha=0.9,
                   color=cmapd(ci / max(len(LADDER) - 1, 1)), label=cell)
    ax[1].axhline(0, color="#333", ls=":", lw=1.2)
    ax[1].axvspan(128, 2048, color="#999", alpha=0.12)
    ax[1].set_ylim(-25, 25)
    ax[1].text(190, 20, "delta not yet\nbudget-matched", fontsize=7)
    ax[1].set_xscale("log"); ax[1].set_xlabel("milestone (transitions)")
    ax[1].set_ylabel("fixed - delta (% of range), drift/ballistic")
    ax[1].legend(fontsize=7)
    fig.suptitle("delta never overtakes uniform plasticity anywhere on the axis "
                 "(the deficit shrinks with difficulty but does not cross zero)", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "fig2_outcome_axis.png")); plt.close(fig)

    # ---- fig3: b(s)'s own clock moves the same mechanism ----------------------------
    fig, ax = plt.subplots(1, 3, figsize=(14.5, 4.0))
    for fam, mk in (("phi12", "o-"), ("two12", "s--")):
        cs = [c for c in BLR[fam] if c in bycell]
        rr = [bycell[c][0]["r_ratio"] for c in cs]
        for i, k in enumerate(("lam_by_T", "wA_by_T", "wR_by_T")):
            mm = [stat(bycell[c], lambda r: r[k][-1])[0] for c in cs]
            ee = [stat(bycell[c], lambda r: r[k][-1])[1] for c in cs]
            ax[i].errorbar(rr, mm, yerr=ee, fmt=mk, lw=1.8, ms=6, capsize=3,
                           color=C1 if fam == "phi12" else C3, label=fam)
    for c, lab, col in (("phi12_alr9e4", "phi12, alr 9e-4 (same r, 3x faster clocks)", C2),
                        ("two12_alr9e4", "two12, alr 9e-4 (same r)", C4)):
        if c in bycell:
            for i, k in enumerate(("lam_by_T", "wA_by_T", "wR_by_T")):
                mm, ee, _ = stat(bycell[c], lambda r: r[k][-1])
                ax[i].errorbar([bycell[c][0]["r_ratio"]], [mm], yerr=[ee], fmt="*",
                               ms=14, color=col, label=lab if i == 0 else None)
    for i, lab, ref in ((0, r"$\Lambda$ (lag / excursion)", 0.0),
                        (1, r"$w_A$ (frontier's share) - a mixture, half-collapses", 1.0),
                        (2, r"$w_R$ (irreducible region's share) - does NOT collapse", 1.0)):
        ax[i].axhline(ref, color="#888", ls="--", lw=1)
        ax[i].set_xscale("log"); ax[i].set_xlabel(r"$r$ = adapt_lr / bench_lr  (slower b $\rightarrow$)")
        ax[i].set_ylabel(lab)
    ax[0].legend(fontsize=6)
    fig.suptitle("The two-sided ratio test SPLITS: at matched r (stars vs the r=0.30 line ends) Lambda "
                 "collapses,\nbut w_R ignores r entirely and tracks b(s)'s ABSOLUTE rate - "
                 "the temporal/spatial decomposition of b's estimation error", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "fig3_benchmark_clock.png")); plt.close(fig)

    # ---- fig4: the noise tax, isolated ---------------------------------------------
    if "two12_nonoise" in bycell and "two12" in bycell:
        fig, ax = plt.subplots(1, 2, figsize=(11.0, 4.0))
        cls = ["drift (A)", "mastered (B)", "decoy (R)"]
        keys = ["wA_by_T", "wB_by_T", "wR_by_T"]
        w = 0.36
        for j, (cell, col, lab) in enumerate((("two12", C1, "R = aleatoric noise (amp 30)"),
                                              ("two12_nonoise", C4, "R = clean (amp 0)"))):
            mm = [stat(bycell[cell], lambda r, k=k: r[k][-1])[0] for k in keys]
            ee = [stat(bycell[cell], lambda r, k=k: r[k][-1])[1] for k in keys]
            ax[0].bar(np.arange(3) + (j - 0.5) * w, mm, w, yerr=ee, capsize=3,
                      color=col, label=lab, alpha=0.9)
        ax[0].axhline(1.0, color="#333", ls=":", lw=1.2)
        ax[0].set_xticks(range(3)); ax[0].set_xticklabels(cls)
        ax[0].set_ylabel("delta's realized share of plasticity / its share of samples")
        ax[0].legend(fontsize=7); ax[0].set_title("where delta's budget goes")
        for cell, col in (("two12", C1), ("two12_nonoise", C4)):
            rs = bycell[cell]; ms = ms_of(rs)
            v = np.mean([np.array(r["delta_by_T"]["drift/ballistic_cem"]["inst_fixed"], float)
                         - np.array(r["delta_by_T"]["drift/ballistic_cem"]["inst_delta"], float)
                         for r in rs], 0)
            rng = np.mean([r["stale_drift/ballistic_cem"] - r["ceil_drift/ballistic_cem"]
                           for r in rs])
            ax[1].plot(ms, 100 * v / rng, "o-", color=col, lw=1.8, ms=4, label=cell)
        ax[1].axhline(0, color="#333", ls=":", lw=1.2)
        ax[1].set_xscale("log"); ax[1].set_xlabel("milestone (transitions)")
        ax[1].set_ylabel("fixed - delta (% of range)"); ax[1].legend(fontsize=7)
        ax[1].set_title("and what removing the tax buys behaviourally")
        fig.suptitle("The noise tax: delta over-spends on the irreducible region in every "
                     "cell; neutralising it frees the frontier's share (1.00 -> 1.31)", fontsize=10)
        fig.tight_layout(); fig.savefig(os.path.join(outdir, "fig4_noise_tax.png")); plt.close(fig)

    # ---- fig5: budget-match diagnostic ---------------------------------------------
    fig, axx = plt.subplots(figsize=(5.6, 4.0))
    for cell in LADDER + [c for c in ("two12_nonoise",) if c in bycell]:
        rs = bycell[cell]; ms = ms_of(rs)
        mm = np.mean([r["meanw_by_T"] for r in rs], 0)
        axx.plot(ms, mm, "o-", ms=3, lw=1.4, label=cell)
    axx.axhline(1.0, color="#333", ls=":", lw=1.2)
    axx.set_xscale("log"); axx.set_xlabel("budget T' (transitions)")
    axx.set_ylabel("delta's realized mean plasticity weight over [0, T']")
    axx.set_title("The EWMA budget match needs ~2k transitions to converge\n"
                  "(short-budget AUC comparisons are not budget-matched)", fontsize=9)
    axx.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "fig5_budget_match.png")); plt.close(fig)
    print(f"[figures] wrote 5 figures to {outdir}")


if __name__ == "__main__":
    main()
