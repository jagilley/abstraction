"""Tail statistics of the per-token NLL, against the twin's KNOWN-validity panel. Offline.

WHY. The grade of record (`critic/`'s, ported in `plant/grader.py`) is the MEAN per-token NLL
of a proposed fill, thresholded at the oracle-free q-quantile of the same statistic on genuine
held-out exemplars of that (style, mask-size). On the twin's panel that grade passes the
INVALID class more often than the VALID one at L2/L3 -- `seam` 0.64 / `offstyle` 0.60 and
`seam` 0.54 / `offstyle` 0.26 (aligned, gA24, 8-forward).

The candidate reason is a shape mismatch. Validity here is a CONJUNCTION -- every shared edge
must agree -- and a mean is an AVERAGE. `seam` is locally legal everywhere except the region
boundary, so a few tokens are very surprising and the mean barely moves; `offstyle` is a valid
fill drawn under a uniform demand, so EVERY token is somewhat surprising and the mean moves a
lot. If that is what is happening, mean-NLL is a demand gauge and a TAIL statistic of the same
per-token array should read truth where the mean reads style.

WHAT CHANGES AND WHAT DOES NOT. Only the statistic. The reader, the panel, the buckets and the
thresholding rule are untouched: tau is still the q-quantile of THAT statistic computed on
genuine exemplars in the same (style, nominal mask side) bucket, so every column here is as
oracle-free as the grade of record. The mean column is carried through as the baseline and
reproduces `oracle.json` to the digit.

STATISTICS (per item, over its masked tokens)
    mean       the grade of record
    max        the single most surprising token  (= top-1 mean)
    top2/4/8   mean of the k largest, k clipped to the item's token count
    frac_hi    fraction of the item's tokens above a PER-BUCKET token-surprise level, itself
               the q_tok quantile of the pooled per-token NLL of that bucket's genuine
               exemplars -- oracle-free by the same argument as tau

  and three DEMAND-NORMALISED tails, added because the plain tails above inherit the mean's
  level and therefore inherit its sensitivity to how atypical the demand is:

    max_dev    max - mean: how far the worst token sticks out above the item's OWN level. A
               uniform shift in demand cancels; a few bad tokens in an otherwise fine region
               do not. This is the sharpest form of "validity is a conjunction".
    top4_dev   top4 mean - mean, the same with a wider tail
    ring_gap   mean NLL on the region's OUTER RING minus mean NLL in its interior. The seam
               violation lives on the region/surround interface by construction, so this is
               the statistic that knows WHERE to look; it is oracle-free because the region
               rectangle IS the mask the grader is handed. Undefined at L1 (a 2x2 region has
               no interior) and reported as such.

CAVEATS, stated rather than hidden
  - Chain-rule per-token NLLs are conditioned on the surround plus the cells revealed EARLIER
    in a random reveal order, averaged over 2 orders (the `1` column is the 1-forward
    independent form: surround only). Tail statistics of the 8-forward column therefore mix
    two things -- how surprising a cell is, and how early it was revealed. The 1-forward column
    has no such nuisance and is reported beside it throughout.
  - Buckets are keyed by the NOMINAL mask side of the level (L1/L2/L3 -> 2/4/8), exactly as
    `twin.oracle_run` does. In the aligned condition that is the true token count (4/16/64).
    In the MISALIGNED condition a tile region's covering rectangle is larger (typically
    9/25/81 tokens), so misaligned panel items are compared against calibration items with
    fewer tokens. That mismatch is inherited, not introduced -- keeping it is what makes the
    tail columns comparable to the mean column already on record.
  - At L1 an item has only 4 tokens, so top4 == top8 == mean by construction.

    cd experiments
    python3 canvas/plant/tiles_twin/tailgrade.py --figures
"""

import argparse
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures")

CONDS = ("tw_aligned", "tw_misaligned")
GRADERS = ("gA24", "gB")
COLS = {"": "8-forward chain rule", "1": "1-forward independent"}
LEVELS = (1, 2, 3)
LVL_SIDE = {1: 2, 2: 4, 3: 8}
KINDS = ("clean", "seam", "offstyle", "plant_fill")
STATS = ("mean", "max", "top2", "top4", "top8", "frac_hi",
         "max_dev", "top4_dev", "ring_gap")
Q = 0.90
Q_LO = 0.02
Q_TOK = 0.90


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #

def unpack(vals, nt):
    """Ragged (concatenated values, per-item counts) -> list of per-item arrays."""
    return np.split(vals.astype(np.float64), np.cumsum(nt)[:-1])


def topk_mean(v, k):
    k = min(k, len(v))
    return float(np.sort(v)[-k:].mean())


def item_stats(items, buckets, tok_level, rings=None):
    """(list of per-item token arrays, per-item bucket keys) -> {stat: (N,) array}."""
    out = {s: np.zeros(len(items)) for s in STATS}
    for i, v in enumerate(items):
        out["mean"][i] = v.mean()
        out["max"][i] = v.max()
        out["top2"][i] = topk_mean(v, 2)
        out["top4"][i] = topk_mean(v, 4)
        out["top8"][i] = topk_mean(v, 8)
        s = tok_level.get(buckets[i], np.nan)
        out["frac_hi"][i] = float((v > s).mean()) if np.isfinite(s) else np.nan
        out["max_dev"][i] = out["max"][i] - out["mean"][i]
        out["top4_dev"][i] = out["top4"][i] - out["mean"][i]
        r = rings[i] if rings is not None else None
        if r is not None and len(r) == len(v) and r.any() and (~r).any():
            out["ring_gap"][i] = v[r].mean() - v[~r].mean()
        else:
            out["ring_gap"][i] = np.nan
    return out


def token_levels(cal_items, cal_buckets, q_tok=Q_TOK):
    """Per-bucket token-surprise level: the q_tok quantile of the POOLED per-token NLL of that
    bucket's genuine exemplars. Oracle-free -- it reads only exemplars the agent holds."""
    pool = {}
    for v, b in zip(cal_items, cal_buckets):
        pool.setdefault(b, []).append(v)
    return {b: float(np.quantile(np.concatenate(vs), q_tok)) for b, vs in pool.items()}


# --------------------------------------------------------------------------- #
# the oracle-free threshold, unchanged in form
# --------------------------------------------------------------------------- #

def taus(x, buckets, q, min_n=8):
    out = {}
    for b in set(buckets):
        sel = np.array([bb == b for bb in buckets])
        if sel.sum() >= min_n:
            v = x[sel]
            v = v[np.isfinite(v)]
            if len(v) >= min_n:
                out[b] = float(np.quantile(v, q))
    return out


def apply_tau(x, buckets, tau):
    fb = float(np.median(list(tau.values()))) if tau else np.inf
    t = np.array([tau.get(b, fb) for b in buckets])
    return np.asarray(x) <= t


def cdf_pos(x, buckets, cal_sorted):
    """Where each item sits inside its own bucket's genuine-exemplar distribution of the same
    statistic, in [0, 1]. `pass at q` is (approximately) `cdf_pos <= q`, so sweeping q traces
    the ROC of the actual decision rule and the AUC below is that rule's AUC."""
    out = np.full(len(x), np.nan)
    for b in set(buckets):
        sel = np.array([bb == b for bb in buckets])
        r = cal_sorted.get(b)
        if r is None or len(r) == 0:
            continue
        out[sel] = np.searchsorted(r, x[sel], side="right") / len(r)
    return out


def auc(pos, neg):
    """P(pos > neg) + 0.5 P(tie): 1.0 = the statistic ranks every `pos` item above every
    `neg` item. Used as "rejects seam above offstyle" and "rejects seam above clean"."""
    pos = pos[np.isfinite(pos)]; neg = neg[np.isfinite(neg)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    r = np.argsort(np.argsort(allv, kind="stable"), kind="stable").astype(float)
    # average ranks for ties
    order = np.argsort(allv, kind="stable")
    sv = allv[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) - 1) / 2) / (len(pos) * len(neg)))


# --------------------------------------------------------------------------- #
# boundary geometry
# --------------------------------------------------------------------------- #

def ring_flags(rect, grid=16):
    """(n_tokens,) bool: is this masked cell on the OUTER RING of the region? Cells are in
    the same row-major order the mask flattens in, which is the order `score_tokens` returns.
    The `seam` violation lives on the region/surround interface, i.e. on this ring."""
    r0, c0, r1, c1 = rect
    r0, c0 = max(0, r0), max(0, c0)
    r1, c1 = min(grid, r1), min(grid, c1)
    rr, cc = np.meshgrid(np.arange(r0, r1), np.arange(c0, c1), indexing="ij")
    return ((rr == r0) | (rr == r1 - 1) | (cc == c0) | (cc == c1 - 1)).reshape(-1)


# --------------------------------------------------------------------------- #

def load(cond):
    return np.load(os.path.join(RES, f"panel_tokens_{cond}.npz"), allow_pickle=True)


def analyse(cond, pct=None):
    """Returns the reduced numbers; if `pct` (a dict) is passed it is filled with the
    within-bucket CDF positions the ROC sweep is drawn from."""
    z = load(cond)
    dsty = z["dsty"]; drect = z["drect"]
    aligned = bool(z["aligned"])
    R = {"condition": cond, "aligned": aligned}
    for g in GRADERS:
        cs = z[f"calstyle|{g}"]; csz = z[f"calside|{g}"]; cnt = z[f"calnt|{g}"]
        cal_b = [(int(a), int(b)) for a, b in zip(cs, csz)]
        for col in COLS:
            cal_items = unpack(z[f"calvals{col}|{g}"], cnt)
            tl = token_levels(cal_items, cal_b)
            cal_rings = [ring_flags((0, 0, int(sd), int(sd))) for sd in csz]
            cal_x = item_stats(cal_items, cal_b, tl, cal_rings)
            tau_hi = {s: taus(cal_x[s], cal_b, Q) for s in STATS}
            tau_lo = {s: taus(cal_x[s], cal_b, Q_LO) for s in STATS}
            cal_sorted = {s: {} for s in STATS}
            for s in STATS:
                for b in set(cal_b):
                    sel = np.array([bb == b for bb in cal_b])
                    v = cal_x[s][sel]
                    cal_sorted[s][b] = np.sort(v[np.isfinite(v)])
            pan = {}
            for lv in LEVELS:
                side = LVL_SIDE[lv]
                for kind in KINDS:
                    key = f"t{col}|{g}|{lv}|{kind}"
                    if key not in z:
                        continue
                    sel = z[f"sel|{lv}|{kind}"]
                    items = unpack(z[key], z[f"nt|{lv}|{kind}"])
                    b = [(int(u), side) for u in dsty[sel]]
                    X = item_stats(items, b, tl, [ring_flags(drect[i]) for i in sel])
                    pan[(lv, kind)] = {"sel": sel, "items": items, "b": b, "X": X}
            R[f"{g}|{col}"] = reduce(z, pan, tau_hi, tau_lo, cal_sorted, aligned,
                                     pct, (g, col))
    R["mechanism"] = mechanism(z)
    return R


def reduce(z, pan, tau_hi, tau_lo, cal_sorted, aligned, pct=None, who=()):
    out = {}
    fill_nv = z["fill_nviol"] if "fill_nviol" in z else None
    rt_nv = z["roundtrip_nviol"] if "roundtrip_nviol" in z else None
    for s in STATS:
        row = {}
        for lv in LEVELS:
            for kind in KINDS:
                P = pan.get((lv, kind))
                if P is None:
                    continue
                x, b = P["X"][s], P["b"]
                fin = np.isfinite(x)
                if pct is not None:
                    pct[(*who, s, lv, kind)] = np.where(fin, cdf_pos(x, b, cal_sorted[s]), np.nan)
                if not fin.any():
                    continue
                p1 = apply_tau(x, b, tau_hi[s])
                p2 = p1 & ~apply_tau(x, b, tau_lo[s])
                d = {"n": int(fin.sum()), "stat": float(np.nanmean(x)),
                     "pass_one_sided": float(p1[fin].mean()),
                     "pass_two_sided": float(p2[fin].mean())}
                if kind == "plant_fill" and fill_nv is not None:
                    nv = fill_nv[P["sel"]]; rt = rt_nv[P["sel"]]
                    ok = (nv >= 0) & fin
                    v, iv = ok & (nv == 0), ok & (nv > 0)
                    d["n_scored"] = int(ok.sum())
                    d["frac_oracle_valid"] = float((nv[ok] == 0).mean()) if ok.any() else None
                    d["pass_if_valid"] = float(p1[v].mean()) if v.any() else None
                    d["pass_if_invalid"] = float(p1[iv].mean()) if iv.any() else None
                    d["auc_valid_vs_invalid"] = auc(
                        cdf_pos(x, b, cal_sorted[s])[iv], cdf_pos(x, b, cal_sorted[s])[v])
                    # restricted to swatches where the clean round trip is itself valid,
                    # i.e. the tile classifier demonstrably resolves that swatch
                    cln = ok & (rt == 0)
                    vv, ii = cln & (nv == 0), cln & (nv > 0)
                    d["n_clean_rt"] = int(cln.sum())
                    d["pass_if_valid_rt"] = float(p1[vv].mean()) if vv.any() else None
                    d["pass_if_invalid_rt"] = float(p1[ii].mean()) if ii.any() else None
                    d["auc_valid_vs_invalid_rt"] = auc(
                        cdf_pos(x, b, cal_sorted[s])[ii], cdf_pos(x, b, cal_sorted[s])[vv])
                row[f"{lv}_{kind}"] = d
            # separation, over the whole threshold sweep, of the rule that is actually used
            for a_, b_ in (("seam", "offstyle"), ("seam", "clean")):
                A, B = pan.get((lv, a_)), pan.get((lv, b_))
                if A is None or B is None:
                    continue
                pa = cdf_pos(A["X"][s], A["b"], cal_sorted[s])
                pb = cdf_pos(B["X"][s], B["b"], cal_sorted[s])
                row[f"{lv}_auc_{a_}_vs_{b_}"] = auc(pa, pb)
        out[s] = row
    return out


def mechanism(z, k=4):
    """For every panel class: where the highest-surprise tokens sit. `seam`'s violation is on
    the region/surround interface, so if the tail is reading the violation its top-k tokens
    should concentrate on the outer ring above the ring's share of the region."""
    drect = z["drect"]
    out = {}
    for g in GRADERS:
        for col in COLS:
            for lv in LEVELS:
                for kind in KINDS:
                    key = f"t{col}|{g}|{lv}|{kind}"
                    if key not in z:
                        continue
                    sel = z[f"sel|{lv}|{kind}"]
                    items = unpack(z[key], z[f"nt|{lv}|{kind}"])
                    fr_top, share, mr, mi = [], [], [], []
                    for v, i in zip(items, sel):
                        ring = ring_flags(drect[i])
                        if len(ring) != len(v):
                            continue
                        kk = min(k, len(v))
                        top = np.argsort(v)[-kk:]
                        fr_top.append(ring[top].mean())
                        share.append(ring.mean())
                        if ring.any():
                            mr.append(v[ring].mean())
                        if (~ring).any():
                            mi.append(v[~ring].mean())
                    if not fr_top:
                        continue
                    out[f"{g}|{col}|{lv}|{kind}"] = {
                        "n": len(fr_top), "frac_topk_on_ring": float(np.mean(fr_top)),
                        "ring_share_of_region": float(np.mean(share)),
                        "lift": float(np.mean(fr_top) / np.mean(share)),
                        "mean_nll_ring": float(np.mean(mr)) if mr else None,
                        "mean_nll_interior": float(np.mean(mi)) if mi else None}
    return out


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #

def print_table(R, g, col, which="pass_one_sided"):
    D = R[f"{g}|{col}"]
    lbl = "one-sided upper cut" if which == "pass_one_sided" else \
        f"TWO-sided typical set [q={Q_LO}, q={Q}]"
    print(f"\n== {R['condition']} | {g} | {COLS[col]} | {lbl} at q={Q} ==")
    hdr = f"{'stat':>8}{'lvl':>5}" + "".join(f"{k[:9]:>11}" for k in KINDS) \
        + f"{'AUCs/off':>10}{'AUCs/cln':>10}"
    print(hdr)
    for s in STATS:
        for lv in LEVELS:
            r = D[s]
            cells = "".join(
                f"{r[f'{lv}_{k}'][which]:11.3f}" if f"{lv}_{k}" in r else f"{'-':>11}"
                for k in KINDS)
            if all(f"{lv}_{k}" not in r for k in KINDS):
                continue
            a1 = r.get(f"{lv}_auc_seam_vs_offstyle", float("nan"))
            a2 = r.get(f"{lv}_auc_seam_vs_clean", float("nan"))
            print(f"{s:>8}{lv:>5}{cells}{a1:10.3f}{a2:10.3f}")
        print()


def print_validity(R, g, col):
    D = R[f"{g}|{col}"]
    if not R["aligned"]:
        return
    print(f"-- {R['condition']} | {g} | {COLS[col]} | plant fills split by ORACLE validity --")
    print(f"{'stat':>8}{'lvl':>5}{'n':>6}{'fracVal':>9}{'passVal':>9}{'passInv':>9}{'AUC':>8}"
          f"{'|nRT':>7}{'passVal':>9}{'passInv':>9}{'AUC':>8}")
    for s in STATS:
        for lv in LEVELS:
            d = D[s].get(f"{lv}_plant_fill")
            if d is None or "n_scored" not in d:
                continue
            f = lambda v: f"{v:9.3f}" if isinstance(v, float) else f"{'-':>9}"
            print(f"{s:>8}{lv:>5}{d['n_scored']:6d}{d['frac_oracle_valid']:9.3f}"
                  f"{f(d['pass_if_valid'])}{f(d['pass_if_invalid'])}"
                  f"{d['auc_valid_vs_invalid']:8.3f}{d['n_clean_rt']:7d}"
                  f"{f(d['pass_if_valid_rt'])}{f(d['pass_if_invalid_rt'])}"
                  f"{d['auc_valid_vs_invalid_rt']:8.3f}")
        print()


def print_mechanism(R, g="gA24"):
    M = R["mechanism"]
    print(f"\n-- {R['condition']} | {g} | where the top-4 most surprising tokens sit --")
    print(f"{'col':>4}{'lvl':>5}{'kind':>12}{'n':>6}{'topk_on_ring':>14}{'ring_share':>12}"
          f"{'lift':>8}{'nll_ring':>10}{'nll_int':>10}")
    for col in COLS:
        for lv in LEVELS:
            for kind in KINDS:
                d = M.get(f"{g}|{col}|{lv}|{kind}")
                if d is None:
                    continue
                ri = d["mean_nll_ring"]; it = d["mean_nll_interior"]
                print(f"{col or '8':>4}{lv:>5}{kind:>12}{d['n']:6d}{d['frac_topk_on_ring']:14.3f}"
                      f"{d['ring_share_of_region']:12.3f}{d['lift']:8.3f}"
                      + (f"{ri:10.3f}" if ri is not None else f"{'-':>10}")
                      + (f"{it:10.3f}" if it is not None else f"{'-':>10}"))


# --------------------------------------------------------------------------- #

def figures(R, PCT, cond):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(FIG, exist_ok=True)
    cmap = plt.get_cmap("viridis")
    colr = {s: cmap(i / (len(STATS) - 1)) for i, s in enumerate(STATS)}

    # --- 1. pass rates by class, per statistic, gA24 8-forward ---
    fig, ax = plt.subplots(2, 3, figsize=(16, 8))
    for gi, g in enumerate(GRADERS):
        for lv in LEVELS:
            a = ax[gi][lv - 1]
            D = R[f"{g}|"]
            w = 0.8 / len(STATS)
            for i, s in enumerate(STATS):
                vals = [D[s].get(f"{lv}_{k}", {}).get("pass_one_sided", np.nan) for k in KINDS]
                a.bar(np.arange(len(KINDS)) + i * w, vals, w, color=colr[s], label=s)
            a.set_xticks(np.arange(len(KINDS)) + 0.4)
            a.set_xticklabels(["clean\n(valid)", "seam\n(INVALID)", "offstyle\n(valid)",
                               "plant fill"], fontsize=8)
            a.axhline(Q, color="k", ls=":", lw=.8)
            a.set_ylim(0, 1.02); a.grid(alpha=.3, axis="y")
            a.set_title(f"{g}  L{lv}", fontsize=10)
            if lv == 1:
                a.set_ylabel("pass rate (one-sided, q=0.90)")
            if gi == 0 and lv == 3:
                a.legend(fontsize=7, ncol=2)
    fig.suptitle(f"{cond}: pass rate by statistic — mean is the grade of record "
                 f"(8-forward chain rule)", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, f"tail_pass_{cond}.png"), dpi=140)
    plt.close(fig)

    # --- 2. ROC of the actual rule: seam rejected vs offstyle / clean kept ---
    qs = np.linspace(0, 1, 201)
    fig, ax = plt.subplots(2, 3, figsize=(16, 8))
    for gi, g in enumerate(GRADERS):
        for lv in LEVELS:
            a = ax[gi][lv - 1]
            for s in STATS:
                key = lambda k: PCT.get((g, "", s, lv, k))
                ps, po = key("seam"), key("offstyle")
                if ps is None or po is None or not np.isfinite(ps).any():
                    continue
                x = [np.nanmean(po <= q) for q in qs]      # valid kept
                y = [1 - np.nanmean(ps <= q) for q in qs]  # invalid rejected
                a.plot(x, y, color=colr[s], lw=1.6, label=s)
                d = R[f"{g}|"][s]
                if f"{lv}_offstyle" in d and f"{lv}_seam" in d:
                    a.plot(d[f"{lv}_offstyle"]["pass_one_sided"],
                           1 - d[f"{lv}_seam"]["pass_one_sided"], "o", color=colr[s], ms=6,
                           mec="k", mew=.6)
            a.plot([0, 1], [1, 0], "k:", lw=.7)
            a.set_xlim(0, 1); a.set_ylim(0, 1); a.grid(alpha=.3)
            a.set_xlabel("offstyle (VALID) kept"); a.set_title(f"{g}  L{lv}", fontsize=10)
            if lv == 1:
                a.set_ylabel("seam (INVALID) rejected")
            if gi == 0 and lv == 3:
                a.legend(fontsize=7)
    fig.suptitle(f"{cond}: the threshold sweep — dots are the q=0.90 rule of record", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, f"tail_roc_{cond}.png"), dpi=140)
    plt.close(fig)

    # --- 3. plant fills split by oracle validity + the mechanism readout ---
    if R["aligned"]:
        fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
        D = R["gA24|"]
        w = 0.8 / len(STATS)
        gp = lambda s, lv, f: (D[s].get(f"{lv}_plant_fill", {}).get(f) or np.nan)
        for i, s in enumerate(STATS):
            v = [gp(s, lv, "pass_if_valid") for lv in LEVELS]
            iv = [gp(s, lv, "pass_if_invalid") for lv in LEVELS]
            ax[0].bar(np.arange(3) + i * w, v, w, color=colr[s], label=s)
            ax[0].bar(np.arange(3) + i * w, iv, w, facecolor="none", edgecolor="k", lw=.7)
        ax[0].set_xticks(np.arange(3) + .4); ax[0].set_xticklabels([f"L{l}" for l in LEVELS])
        ax[0].set_ylabel("pass rate of the plant's fills")
        ax[0].set_title("filled = oracle-VALID fills, outline = oracle-INVALID", fontsize=9)
        ax[0].legend(fontsize=7, ncol=2); ax[0].grid(alpha=.3, axis="y")
        for i, s in enumerate(STATS):
            a = [gp(s, lv, "auc_valid_vs_invalid") for lv in LEVELS]
            ax[1].plot(LEVELS, a, "-o", color=colr[s], label=s)
        ax[1].axhline(0.5, color="k", ls=":", lw=.8)
        ax[1].set_xticks(LEVELS); ax[1].set_xlabel("level")
        ax[1].set_ylabel("AUC: invalid fill ranked above valid fill")
        ax[1].set_title("does the statistic read TRUTH on the plant's own fills?", fontsize=9)
        ax[1].legend(fontsize=7); ax[1].grid(alpha=.3)
        M = R["mechanism"]
        for kind, mk in (("seam", "o"), ("clean", "s"), ("offstyle", "^")):
            y = [M[f"gA24||{lv}|{kind}"]["frac_topk_on_ring"] for lv in LEVELS]
            sh = [M[f"gA24||{lv}|{kind}"]["ring_share_of_region"] for lv in LEVELS]
            ax[2].plot(LEVELS, y, "-" + mk, label=kind)
        ax[2].plot(LEVELS, sh, "k--", lw=1.2, label="ring share of region (chance)")
        ax[2].set_xticks(LEVELS); ax[2].set_xlabel("level")
        ax[2].set_ylabel("fraction of the top-4 tokens on the region's outer ring")
        ax[2].set_title("where the surprise sits", fontsize=9)
        ax[2].legend(fontsize=7); ax[2].grid(alpha=.3)
        fig.suptitle(f"{cond}: gA24, 8-forward", fontsize=12)
        fig.tight_layout(); fig.savefig(os.path.join(FIG, f"tail_truth_{cond}.png"), dpi=140)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conds", nargs="*", default=list(CONDS))
    ap.add_argument("--figures", action="store_true")
    a = ap.parse_args()
    ALL = {}
    for cond in a.conds:
        PCT = {}
        R = analyse(cond, PCT)
        ALL[cond] = R
        for g in GRADERS:
            for col in COLS:
                print_table(R, g, col)
        print_table(R, "gA24", "", which="pass_two_sided")
        for g in GRADERS:
            for col in COLS:
                print_validity(R, g, col)
        print_mechanism(R)
        if a.figures:
            figures(R, PCT, cond)
            print(f"figures -> {FIG}/tail_*_{cond}.png")
    json.dump(ALL, open(os.path.join(RES, "tailgrade.json"), "w"), indent=1)
    print(f"\nnumbers -> {os.path.join(RES, 'tailgrade.json')}")


if __name__ == "__main__":
    main()
