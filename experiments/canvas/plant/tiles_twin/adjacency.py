"""The adjacency-support grader: a POSSIBLE-SET test on the code grid. Zero forward passes.

WHY THIS EXISTS. `tailgrade.py` re-scored the known-validity panel with every tail statistic of
the per-token NLL and none of them read validity in the ALIGNED condition (best AUC on
seam-vs-offstyle: 0.65, and only for `ring_gap`, the one statistic that is told WHERE to look).
The mechanism readout said why: aligned, a tile is exactly a 2x2 code block, so an illegal
shared edge falls exactly BETWEEN two code cells and no single token is anomalous -- the seam
lifts boundary-cell NLL by 0.09 nats while an off-style (valid) demand lifts every token by
0.37. Validity is a RELATION between tokens; typicality is a property OF tokens; no statistic
of per-token surprise reads a pairwise constraint.

`critic/`'s grade on RHM was never a likelihood -- possible-set success is a SUPPORT test. This
is the learned analogue on the code grid, and it is the relation-shaped instrument the NLL
family cannot be:

    SUPPORT     from GENUINE EXEMPLARS ONLY, the set of ordered adjacent code pairs that occur:
                H = (code at (r,c), code at (r,c+1)), V = (code at (r,c), code at (r+1,c)).
                Orientation is kept -- left/right and above/below are different relations, and
                the tile grammar's edge table is not symmetric.
    VERDICT     a candidate fill passes iff EVERY adjacent pair it touches is in the support --
                pairs inside the hole AND pairs between a hole cell and the surround. That
                conjunction is what "valid" means here, expressed in the same shape.
    COST        zero forward passes. No reader, no threshold fitted to anything but exemplars.

ORACLE-FREE, by the same argument as tau: the support is built from held-in exemplars of the
grader's own split (`gA_train`, and `gA_train + gB_train` to see how it saturates); the mask is
what the grader is handed; no shader, seed, offset or held-out pixel is read.

VARIANTS, all reported side by side
    per-style / pooled       support taken within a style, or over all 36. Per-style is the
                             strict reading ("this school's relations"); pooled is the loose
                             one and trades false rejects for false accepts.
    per-tileset             DIAGNOSTIC ONLY, and labelled as such: the 9 schools sharing a
                             tileset share a valid set, so this is the closest a support test
                             can get to the DGP's own answer. The grouping is a fact about the
                             generator, so it is a ceiling, not a runnable rule.
    min_count in {1, 2, 4}   a pair counts as supported only if seen at least that often --
                             how much of the test rests on singletons.
    soft                     `frac_unsup` (fraction of touched pairs out of support) and
                             `n_unsup`, so there is a ROC rather than one binary point, plus a
                             pass at the SAME oracle-free q = 0.90 quantile rule the NLL grader
                             uses, on `frac_unsup` over the grader's own `gA_val` exemplars.

    cd experiments      # conda glp
    python -m canvas.plant.tiles_twin.adjacency --figures
"""

import argparse
import json
import os

import numpy as np

from canvas.plant.tiles_twin.tailgrade import auc

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures")

CONDS = ("tw_aligned", "tw_misaligned")
LEVELS = (1, 2, 3)
LVL_SIDE = {1: 2, 2: 4, 3: 8}
KINDS = ("clean", "seam", "offstyle", "plant_fill")
GRID = 16
K = 512
MIN_COUNTS = (1, 2, 4)
Q = 0.90


# --------------------------------------------------------------------------- #
# pairs
# --------------------------------------------------------------------------- #

def pair_codes(g, grid=GRID, k=K):
    """(N, 256) codes -> the two ORIENTED adjacency planes, as single ints a*k + b.
    H[n, r, c] is the pair ((r,c) -> (r,c+1)); V[n, r, c] is ((r,c) -> (r+1,c))."""
    g = np.asarray(g).reshape(-1, grid, grid).astype(np.int64)
    return g[:, :, :-1] * k + g[:, :, 1:], g[:, :-1, :] * k + g[:, 1:, :]


def support_from(g, min_count=1, k=K):
    """Sorted unique pair codes seen at least `min_count` times, per orientation."""
    H, V = pair_codes(g, k=k)
    out = []
    for P in (H, V):
        u, c = np.unique(P.reshape(-1), return_counts=True)
        out.append(u[c >= min_count])
    return tuple(out)


def in_support(codes, sup):
    return np.isin(codes, sup, assume_unique=False)


def touched(masks, grid=GRID):
    """Which adjacency slots a fill TOUCHES, and which of those CROSS the hole boundary.
    A pair is touched if either endpoint is masked; crossing if exactly one is."""
    m = np.asarray(masks).reshape(-1, grid, grid)
    th = m[:, :, :-1] | m[:, :, 1:]
    tv = m[:, :-1, :] | m[:, 1:, :]
    ch = m[:, :, :-1] ^ m[:, :, 1:]
    cv = m[:, :-1, :] ^ m[:, 1:, :]
    return (th, tv), (ch, cv)


def evaluate(grids, masks, styles, sup_of):
    """-> dict of per-item counts. `sup_of(style)` returns (supH, supV)."""
    H, V = pair_codes(grids)
    (th, tv), (ch, cv) = touched(masks)
    N = len(H)
    n_t = np.zeros(N, int); n_u = np.zeros(N, int)
    n_c = np.zeros(N, int); n_uc = np.zeros(N, int)
    n_i = np.zeros(N, int); n_ui = np.zeros(N, int)
    for s in np.unique(styles):
        sel = np.flatnonzero(styles == s)
        sH, sV = sup_of(int(s))
        for P, T, C, sup in ((H[sel], th[sel], ch[sel], sH),
                             (V[sel], tv[sel], cv[sel], sV)):
            ok = in_support(P, sup)
            bad = T & ~ok
            n_t[sel] += T.reshape(len(sel), -1).sum(1)
            n_u[sel] += bad.reshape(len(sel), -1).sum(1)
            n_c[sel] += (T & C).reshape(len(sel), -1).sum(1)
            n_uc[sel] += (bad & C).reshape(len(sel), -1).sum(1)
            n_i[sel] += (T & ~C).reshape(len(sel), -1).sum(1)
            n_ui[sel] += (bad & ~C).reshape(len(sel), -1).sum(1)
    return {"n_touched": n_t, "n_unsup": n_u, "n_cross": n_c, "n_unsup_cross": n_uc,
            "n_interior": n_i, "n_unsup_interior": n_ui,
            "frac_unsup": n_u / np.maximum(n_t, 1)}


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #

def load(cond):
    z = np.load(os.path.join(RES, f"panel_tokens_{cond}.npz"), allow_pickle=True)
    tag = cond.replace("tw_", "twq_")
    idx = json.load(open(os.path.join(RES, f"index_{tag}.json")))
    codes = np.load(os.path.join(RES, f"quant_K512_{tag}.npz"))["codes"].astype(np.int64)
    styles = idx["styles"]
    sidx = np.array(idx["sidx"], np.int64); split = np.array(idx["split"])
    return z, styles, sidx, split, codes


def supports(codes, sidx, split, styles, which, min_count):
    """Build every support variant from the named split(s). Returns
    (per_style, pooled, per_tileset, stats)."""
    m = np.zeros(len(split), bool)
    for nm in which:
        m |= split == nm
    C, S = codes[m], sidx[m]
    per, stats = {}, {}
    for u in range(len(styles)):
        g = C[S == u]
        per[u] = support_from(g, min_count)
        stats[styles[u]] = {"n_swatches": int(len(g)),
                            "n_distinct_H": int(len(per[u][0])),
                            "n_distinct_V": int(len(per[u][1]))}
    pooled = support_from(C, min_count)
    # the DGP's own grouping (9 schools share a tileset) -- a diagnostic ceiling, not a rule
    ts = {}
    for u, nm in enumerate(styles):
        ts.setdefault(nm.rsplit("_", 1)[0], []).append(u)
    per_ts = {}
    for name, us in ts.items():
        sup = support_from(C[np.isin(S, us)], min_count)
        for u in us:
            per_ts[u] = sup
    return per, pooled, per_ts, stats


def variants(per, pooled, per_ts):
    return {"per_style": lambda u: per[u],
            "pooled": lambda u: pooled,
            "per_tileset*": lambda u: per_ts[u]}


# --------------------------------------------------------------------------- #

def analyse(cond, split_sets=(("gA_train",), ("gA_train", "gB_train"))):
    z, styles, sidx, split, codes = load(cond)
    dsty = z["dsty"]; dlvl = z["dlvl"]; dkind = z["dkind"]; masks = z["pmask"]
    dcodes = z["dcodes"].astype(np.int64); fills = z["fills"].astype(np.int64)
    fill_nv = z["fill_nviol"] if "fill_nviol" in z else None
    rt_nv = z["roundtrip_nviol"] if "roundtrip_nviol" in z else None
    R = {"condition": cond, "aligned": bool(z["aligned"]), "styles": list(styles)}

    # calibration slice for the oracle-free tau on the soft statistic: the grader's own
    # gA_val exemplars, masked with the ladder's rectangles (same construction as the NLL rule)
    rng = np.random.default_rng(21)
    cm = split == "gA_val"
    cc, ccs = codes[cm], sidx[cm]
    crows, cmask, cside = [], [], []
    for side in (2, 4, 8):
        for i in range(len(cc)):
            for _ in range(16):
                r0 = int(rng.integers(0, GRID - side + 1)); c0 = int(rng.integers(0, GRID - side + 1))
                mm = np.zeros((GRID, GRID), bool); mm[r0:r0 + side, c0:c0 + side] = True
                crows.append(i); cmask.append(mm.reshape(-1)); cside.append(side)
    crows = np.array(crows); cmask = np.stack(cmask); cside = np.array(cside)
    cal_g, cal_s = cc[crows], ccs[crows]

    for which in split_sets:
        wname = "+".join(w.replace("_train", "") for w in which)
        for mc in MIN_COUNTS:
            per, pooled, per_ts, stats = supports(codes, sidx, split, styles, which, mc)
            if mc == 1:
                R[f"support_stats|{wname}"] = stats
                R[f"support_pooled|{wname}"] = {"n_distinct_H": int(len(pooled[0])),
                                                "n_distinct_V": int(len(pooled[1]))}
            for vname, sup_of in variants(per, pooled, per_ts).items():
                key = f"{wname}|mc{mc}|{vname}"
                cal = evaluate(cal_g, cmask, cal_s, sup_of)
                tau = {}
                for b in set(zip(cal_s.tolist(), cside.tolist())):
                    sel = (cal_s == b[0]) & (cside == b[1])
                    if sel.sum() >= 8:
                        tau[b] = float(np.quantile(cal["frac_unsup"][sel], Q))
                fb = float(np.median(list(tau.values()))) if tau else 0.0
                # the false-reject floor: held-out truth, whole grid, no mask
                ho = split == "plant_test"
                full = evaluate(codes[ho], np.ones((int(ho.sum()), 256), bool), sidx[ho], sup_of)
                rows = {"cal_pass_binary": float((cal["n_unsup"] == 0).mean()),
                        "heldout_pair_in_support": float(
                            1 - full["n_unsup"].sum() / full["n_touched"].sum()),
                        "heldout_swatch_all_supported": float((full["n_unsup"] == 0).mean())}
                soft = {}
                for lv in LEVELS:
                    side = LVL_SIDE[lv]
                    for kind in KINDS:
                        if kind == "plant_fill":
                            sel = np.flatnonzero((dlvl == lv) & (dkind == "clean"))
                            grids = fills[sel]
                        else:
                            sel = np.flatnonzero((dlvl == lv) & (dkind == kind))
                            grids = dcodes[sel]
                        if len(sel) == 0:
                            continue
                        E = evaluate(grids, masks[sel], dsty[sel], sup_of)
                        t = np.array([tau.get((int(u), side), fb) for u in dsty[sel]])
                        pb = E["n_unsup"] == 0
                        d = {"n": int(len(sel)), "pass_binary": float(pb.mean()),
                             "pass_tau": float((E["frac_unsup"] <= t).mean()),
                             "n_unsup": float(E["n_unsup"].mean()),
                             "frac_unsup": float(E["frac_unsup"].mean()),
                             "n_touched": float(E["n_touched"].mean()),
                             "unsup_on_crossing": float(
                                 E["n_unsup_cross"].sum() / max(E["n_unsup"].sum(), 1)),
                             "crossing_share": float(
                                 E["n_cross"].sum() / max(E["n_touched"].sum(), 1))}
                        if kind == "plant_fill" and fill_nv is not None:
                            nv, rt = fill_nv[sel], rt_nv[sel]
                            ok = nv >= 0
                            v, iv = ok & (nv == 0), ok & (nv > 0)
                            cln = ok & (rt == 0)
                            vv, ii = cln & (nv == 0), cln & (nv > 0)
                            d.update({
                                "n_scored": int(ok.sum()),
                                "frac_oracle_valid": float((nv[ok] == 0).mean()),
                                "pass_if_valid": float(pb[v].mean()) if v.any() else None,
                                "pass_if_invalid": float(pb[iv].mean()) if iv.any() else None,
                                "auc_valid_vs_invalid": auc(E["frac_unsup"][iv],
                                                            E["frac_unsup"][v]),
                                "n_clean_rt": int(cln.sum()),
                                "pass_if_valid_rt": float(pb[vv].mean()) if vv.any() else None,
                                "pass_if_invalid_rt": float(pb[ii].mean()) if ii.any() else None,
                                "auc_valid_vs_invalid_rt": auc(E["frac_unsup"][ii],
                                                               E["frac_unsup"][vv])})
                        rows[f"{lv}_{kind}"] = d
                        soft[(lv, kind)] = E["frac_unsup"]
                    for a_, b_ in (("seam", "offstyle"), ("seam", "clean")):
                        if (lv, a_) in soft and (lv, b_) in soft:
                            rows[f"{lv}_auc_{a_}_vs_{b_}"] = auc(soft[(lv, a_)], soft[(lv, b_)])
                R[key] = rows
    return R


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #

def print_support(R, wname):
    st = R.get(f"support_stats|{wname}")
    if st is None:
        return
    H = np.array([v["n_distinct_H"] for v in st.values()])
    V = np.array([v["n_distinct_V"] for v in st.values()])
    ns = list(st.values())[0]["n_swatches"]
    p = R[f"support_pooled|{wname}"]
    print(f"\n-- {R['condition']} | support from {wname} ({ns} swatches/style, min_count=1) --")
    print(f"   distinct ORIENTED adjacent pairs per style: H {H.mean():.0f} "
          f"[{H.min()}-{H.max()}], V {V.mean():.0f} [{V.min()}-{V.max()}]  "
          f"(of K^2 = {K*K}; a swatch contributes {GRID*(GRID-1)} of each)")
    print(f"   pooled over 36 styles: H {p['n_distinct_H']}, V {p['n_distinct_V']}")


def print_table(R, key):
    D = R.get(key)
    if D is None:
        return
    print(f"\n== {R['condition']} | support {key} | pass = EVERY touched pair in support ==")
    print(f"   held-out clean pairs in support {D['heldout_pair_in_support']:.4f}; "
          f"whole held-out swatches fully supported {D['heldout_swatch_all_supported']:.3f}; "
          f"gA_val masked exemplars passing {D['cal_pass_binary']:.3f}")
    print(f"{'lvl':>4}{'class':>12}{'n':>5}{'passBin':>9}{'passTau':>9}{'n_unsup':>9}"
          f"{'frac_uns':>10}{'n_pairs':>9}{'uns@cross':>11}{'crossShr':>10}")
    for lv in LEVELS:
        for kind in KINDS:
            d = D.get(f"{lv}_{kind}")
            if d is None:
                continue
            print(f"{lv:>4}{kind:>12}{d['n']:5d}{d['pass_binary']:9.3f}{d['pass_tau']:9.3f}"
                  f"{d['n_unsup']:9.2f}{d['frac_unsup']:10.4f}{d['n_touched']:9.1f}"
                  f"{d['unsup_on_crossing']:11.3f}{d['crossing_share']:10.3f}")
        a1 = D.get(f"{lv}_auc_seam_vs_offstyle", float('nan'))
        a2 = D.get(f"{lv}_auc_seam_vs_clean", float('nan'))
        print(f"{'':>4}{'AUC':>12}{'':>5}{'':>9}{'':>9}{'':>9}"
              f"   seam>offstyle {a1:.3f}   seam>clean {a2:.3f}")


def print_validity(R, key):
    D = R.get(key)
    if D is None or not R["aligned"]:
        return
    print(f"-- {R['condition']} | {key} | plant fills split by ORACLE validity --")
    print(f"{'lvl':>4}{'n':>6}{'fracVal':>9}{'passVal':>9}{'passInv':>9}{'AUC':>8}"
          f"{'|nRT':>7}{'passVal':>9}{'passInv':>9}{'AUC':>8}")
    for lv in LEVELS:
        d = D.get(f"{lv}_plant_fill")
        if d is None or "n_scored" not in d:
            continue
        f = lambda v: f"{v:9.3f}" if isinstance(v, float) else f"{'-':>9}"
        print(f"{lv:>4}{d['n_scored']:6d}{d['frac_oracle_valid']:9.3f}"
              f"{f(d['pass_if_valid'])}{f(d['pass_if_invalid'])}{d['auc_valid_vs_invalid']:8.3f}"
              f"{d['n_clean_rt']:7d}{f(d['pass_if_valid_rt'])}{f(d['pass_if_invalid_rt'])}"
              f"{d['auc_valid_vs_invalid_rt']:8.3f}")


def figures(ALL):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(FIG, exist_ok=True)
    base = "gA|mc1|per_style"
    fig, ax = plt.subplots(2, 3, figsize=(16, 8))
    for ci, cond in enumerate(CONDS):
        R = ALL[cond]
        for lv in LEVELS:
            a = ax[ci][lv - 1]
            keys = [f"gA|mc{m}|{v}" for v in ("per_style", "pooled", "per_tileset*")
                    for m in MIN_COUNTS]
            keys = [f"gA|mc1|per_style", "gA|mc2|per_style", "gA|mc4|per_style",
                    "gA|mc1|pooled", "gA|mc1|per_tileset*", "gA+gB|mc1|per_style"]
            w = 0.8 / len(keys)
            cmap = plt.get_cmap("cividis")
            for i, kk in enumerate(keys):
                D = R.get(kk, {})
                vals = [D.get(f"{lv}_{k}", {}).get("pass_binary", np.nan) for k in KINDS]
                a.bar(np.arange(len(KINDS)) + i * w, vals, w,
                      color=cmap(i / (len(keys) - 1)), label=kk.replace("|", " "))
            a.set_xticks(np.arange(len(KINDS)) + .4)
            a.set_xticklabels(["clean\n(valid)", "seam\n(INVALID)", "offstyle\n(valid)",
                               "plant fill"], fontsize=8)
            a.set_ylim(0, 1.02); a.grid(alpha=.3, axis="y")
            a.set_title(f"{cond}  L{lv}", fontsize=10)
            if lv == 1:
                a.set_ylabel("pass rate (every touched pair in support)")
            if ci == 0 and lv == 3:
                a.legend(fontsize=6.5)
    fig.suptitle("adjacency-support grader — zero forward passes", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "adjacency_pass.png"), dpi=140)
    plt.close(fig)

    # separation + truth
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    keys = ["gA|mc1|per_style", "gA|mc2|per_style", "gA|mc4|per_style",
            "gA|mc1|pooled", "gA|mc1|per_tileset*", "gA+gB|mc1|per_style"]
    cmap = plt.get_cmap("cividis")
    for i, kk in enumerate(keys):
        c = cmap(i / (len(keys) - 1))
        for ci, (cond, ls) in enumerate(zip(CONDS, ("-o", "--s"))):
            y = [ALL[cond].get(kk, {}).get(f"{lv}_auc_seam_vs_offstyle", np.nan)
                 for lv in LEVELS]
            ax[0].plot(LEVELS, y, ls, color=c, ms=4,
                       label=f"{kk} · {cond.replace('tw_','')}" if True else None)
        y = [ALL["tw_aligned"].get(kk, {}).get(f"{lv}_plant_fill", {})
             .get("auc_valid_vs_invalid", np.nan) for lv in LEVELS]
        ax[1].plot(LEVELS, y, "-o", color=c, ms=4, label=kk)
        v = [ALL["tw_aligned"].get(kk, {}).get(f"{lv}_plant_fill", {})
             .get("pass_if_valid_rt") or np.nan for lv in LEVELS]
        iv = [ALL["tw_aligned"].get(kk, {}).get(f"{lv}_plant_fill", {})
              .get("pass_if_invalid_rt") or np.nan for lv in LEVELS]
        ax[2].plot(LEVELS, v, "-o", color=c, ms=4, label=kk)
        ax[2].plot(LEVELS, iv, "--x", color=c, ms=5)
    for a, t, yl in ((ax[0], "seam ranked worse than offstyle\n(solid aligned, dashed misaligned)",
                      "AUC"),
                     (ax[1], "does it read TRUTH on the plant's fills?", "AUC invalid > valid"),
                     (ax[2], "plant fills: pass rate, solid VALID / dashed INVALID\n"
                             "(clean-round-trip subset)", "pass rate")):
        a.axhline(0.5, color="k", ls=":", lw=.7); a.set_xticks(LEVELS)
        a.set_xlabel("level"); a.set_ylabel(yl); a.set_title(t, fontsize=9)
        a.grid(alpha=.3); a.legend(fontsize=6)
    ax[2].axhline(1.0, color="k", ls=":", lw=.7)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "adjacency_truth.png"), dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conds", nargs="*", default=list(CONDS))
    ap.add_argument("--figures", action="store_true")
    a = ap.parse_args()
    ALL = {}
    for cond in a.conds:
        R = analyse(cond)
        ALL[cond] = R
        for w in ("gA", "gA+gB"):
            print_support(R, w)
        for w in ("gA", "gA+gB"):
            for mc in MIN_COUNTS:
                for v in ("per_style", "pooled", "per_tileset*"):
                    print_table(R, f"{w}|mc{mc}|{v}")
                    print_validity(R, f"{w}|mc{mc}|{v}")
    if a.figures and len(ALL) == 2:
        figures(ALL)
        print(f"\nfigures -> {FIG}/adjacency_{{pass,truth}}.png")
    json.dump(ALL, open(os.path.join(RES, "adjacency.json"), "w"), indent=1)
    print(f"numbers -> {os.path.join(RES, 'adjacency.json')}")


if __name__ == "__main__":
    main()
