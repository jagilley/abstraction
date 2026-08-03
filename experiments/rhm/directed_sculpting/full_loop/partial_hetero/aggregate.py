"""Aggregate the sharing-depth sweep: the transfer curve (G3) and the ladder curve (E1-PH).

Reads the mirrored results under `figures/` and prints the two curves the sweep exists to
produce, with per-seed slopes against sharing depth so the monotonicity claim is testable rather
than eyeballed (the same discipline `../climb.py`'s necessity sweep reports).

Run from experiments/:
  python3 rhm/directed_sculpting/full_loop/partial_hetero/aggregate.py
  python3 rhm/directed_sculpting/full_loop/partial_hetero/aggregate.py --figures <dir>
"""

import argparse
import glob
import json
import os

import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))


def _mean_sd(xs):
    a = np.asarray([x for x in xs if x is not None and np.isfinite(x)], dtype=float)
    if a.size == 0:
        return float("nan"), float("nan"), 0
    return float(a.mean()), float(a.std(ddof=1)) if a.size > 1 else 0.0, int(a.size)


def _slope(ks, vals):
    """Least-squares slope of `vals` against sharing depth `ks` (per shared level)."""
    k = np.asarray(ks, dtype=float)
    y = np.asarray(vals, dtype=float)
    ok = np.isfinite(y)
    if ok.sum() < 2:
        return float("nan")
    k, y = k[ok], y[ok]
    return float(np.polyfit(k, y, 1)[0])


def _t_stat(xs):
    a = np.asarray(xs, dtype=float)
    a = a[np.isfinite(a)]
    if a.size < 2 or a.std(ddof=1) == 0:
        return float("nan")
    return float(a.mean() / (a.std(ddof=1) / np.sqrt(a.size)))


def load_geometry(figdir):
    """{(share_mode, fm_arch): {seed: {sharing depth: curve}}} from geometry_* result files.

    Keyed on `share_mode` as well as `fm_arch` because the two splice directions are different
    DGPs measured by the same instrument -- pooling them would average a shared-deep geometry
    with a shared-surface one. Files written before the knob existed carry no key and default
    to "top", which is what they were.
    """
    out = {}
    for path in sorted(glob.glob(os.path.join(figdir, "geometry_*", "results.json"))):
        d = json.load(open(path))
        if d["config"].get("quick"):
            continue
        key = (d["config"].get("share_mode", "top"), d["config"].get("fm_arch", "block"))
        out.setdefault(key, {})[d["config"]["seed"]] = {int(k): v for k, v in d["curve"].items()}
    return out


def load_ladder(figdir):
    """{share_mode: {seed: {sharing depth: {policy: row}}}} from ladder_ph_* result files."""
    out = {}
    for path in sorted(glob.glob(os.path.join(figdir, "ladder_ph_*", "results.json"))):
        d = json.load(open(path))
        cfg = d["config"]
        share = max(int(x) for x in str(cfg.get("struct_shares", "0,0")).split(","))
        out.setdefault(cfg.get("share_mode", "top"), {}).setdefault(cfg["seed"], {})[share] = d
    return out


def report_geometry(geo, key=("top", "block")):
    mode, arch = key
    geo = geo.get(key, {})
    seeds = sorted(geo)
    if not seeds:
        print(f"(no geometry_* results for share_mode={mode!r} fm_arch={arch!r})\n")
        return
    shares = sorted(k for k in geo[seeds[0]] if isinstance(k, int))
    arms = list(geo[seeds[0]][shares[0]]["tree_err"])

    print("=" * 100)
    shared_end = "DEEP composition" if mode == "top" else "SURFACE alphabet"
    print(f"G3 [share_mode={mode} | fm_arch={arch}] -- TRANSFER CURVE: tree block-FM error by "
          f"allocation, vs sharing depth")
    print(f"       (structA shares its {shared_end} with the tree; k=0 is two independent "
          f"grammars, k=depth is identical tables under both modes)")
    print(f"       (mean +- sd over seeds {seeds}; every arm matched-budget except tree_half)")
    print("=" * 100)
    print(f"{'share':>5s} " + " ".join(f"{a:>15s}" for a in arms))
    for k in shares:
        cells = []
        for a in arms:
            m, sd, n = _mean_sd([geo[s][k]["tree_err"][a] for s in seeds])
            cells.append(f"{m:.4f}+-{sd:.4f}")
        print(f"{k:>5d} " + " ".join(f"{c:>15s}" for c in cells))

    print("\n  value ADDED to a FIXED tree-data budget (negative = the added data helped);")
    print("  every arm below trains the tree's block embeddings the identical number of times:")
    print(f"{'share':>5s} {'+structA':>16s} {'+structB (ctrl)':>16s} {'+noise (null)':>16s} "
          f"{'2x tree data':>16s} {'A exch rate':>12s}")
    keys = ("structA_value", "structB_value", "noise_value", "doubling_value")
    for k in shares:
        row = [_mean_sd([geo[s][k][key] for s in seeds]) for key in keys]
        rate, rate_sd, _ = _mean_sd([geo[s][k]["structA_exchange_rate"] for s in seeds])
        print(f"{k:>5d} " + " ".join(f"{m:>+9.4f}+-{sd:.4f}" for m, sd, _ in row)
              + f" {rate:>+8.2f}+-{rate_sd:.2f}")

    print("\n  the coverage-confounded pure-transfer arms (read with the caveat, not as headline):")
    print(f"{'share':>5s} {'structA_only cost':>20s} {'structB_only cost':>20s} "
          f"{'uniform cost':>20s}")
    for k in shares:
        row = [_mean_sd([geo[s][k][key] for s in seeds])
               for key in ("structA_only_cost", "structB_only_cost", "uniform_cost")]
        print(f"{k:>5d} " + " ".join(f"{m:>+13.4f}+-{sd:.4f}" for m, sd, _ in row))

    print("\n  the differenced contrast, +structA minus +structB — the generic 'more optimizer")
    print("  steps on the shared trunk' effect cancels, so only the sharing survives:")
    print(f"{'share':>5s} {'A - B':>18s}")
    for k in shares:
        m, sd, _ = _mean_sd([geo[s][k]["structA_value"] - geo[s][k]["structB_value"]
                             for s in seeds])
        print(f"{k:>5d} {m:>+11.4f}+-{sd:.4f}")

    print("\n  per-seed slope against sharing depth (per shared level). NOTE: the sweep is its own")
    print("  control — structB/noise cannot vary with sharing depth, so their slopes are the")
    print("  measurement's noise floor:")
    for key, label in (("structA_value", "+structA value"),
                       ("structB_value", "+structB value (ctrl)"),
                       ("noise_value", "+noise value (null)"),
                       ("structA_exchange_rate", "structA exchange rate"),
                       ("structA_only_cost", "structA_only cost")):
        sl = [_slope(shares, [geo[s][k][key] for k in shares]) for s in seeds]
        m, sd, n = _mean_sd(sl)
        print(f"    {label:24s} {m:+.5f} +- {sd:.5f}  (n={n}, t={_t_stat(sl):+.2f}, "
              f"per-seed " + " ".join(f"{x:+.5f}" for x in sl) + ")")
    print()


def report_ladder(lad, mode="top"):
    lad = lad.get(mode, {})
    seeds = sorted(lad)
    if not seeds:
        print(f"(no ladder_ph_* results for share_mode={mode!r})\n")
        return
    shares = sorted(lad[seeds[0]])
    policies = list(lad[seeds[0]][shares[0]]["results"])

    print("=" * 100)
    print(f"E1-PH [share_mode={mode}] -- ALLOCATION LADDER vs sharing depth: tree FM error "
          f"(mean over rounds)")
    print(f"       (mean +- sd over seeds {seeds})")
    print("=" * 100)
    print(f"{'share':>5s} " + " ".join(f"{p:>17s}" for p in policies))
    for k in shares:
        cells = []
        for p in policies:
            m, sd, _ = _mean_sd([lad[s][k]["results"][p]["tree_err_mean"] for s in seeds])
            cells.append(f"{m:.4f}+-{sd:.4f}")
        print(f"{k:>5d} " + " ".join(f"{c:>17s}" for c in cells))

    gaps = [("uniform", "oracle"), ("uniform", "visits_only"), ("uniform", "value_red"),
            ("oracle", "oracle_shared"), ("uniform", "oracle_shared")]
    gaps = [(a, b) for a, b in gaps if a in policies and b in policies]
    print("\n  the prize, by sharing depth (positive = the second arm is BETTER, "
          "i.e. lower tree FM error):")
    print(f"{'share':>5s} " + " ".join(f"{a[:3]}->{b:>13s}" for a, b in gaps))
    per_seed_gap = {g: [] for g in gaps}
    for k in shares:
        cells = []
        for g in gaps:
            a, b = g
            vals = [lad[s][k]["results"][a]["tree_err_mean"]
                    - lad[s][k]["results"][b]["tree_err_mean"] for s in seeds]
            m, sd, _ = _mean_sd(vals)
            per_seed_gap[g].append(vals)
            cells.append(f"{m:>+9.4f}+-{sd:.4f}")
        print(f"{k:>5d} " + " ".join(f"{c:>17s}" for c in cells))

    print("\n  per-seed slope of each prize against sharing depth (per shared level):")
    for g in gaps:
        by_seed = np.asarray(per_seed_gap[g])            # (n_shares, n_seeds)
        sl = [_slope(shares, by_seed[:, i]) for i in range(by_seed.shape[1])]
        m, sd, n = _mean_sd(sl)
        print(f"    {g[0]}->{g[1]:<16s} {m:+.5f} +- {sd:.5f}  (n={n}, t={_t_stat(sl):+.2f}, "
              f"per-seed " + " ".join(f"{x:+.5f}" for x in sl) + ")")

    print("\n  allocation share by channel (mean over rounds and seeds), per policy:")
    names = [c["name"] for c in lad[seeds[0]][shares[0]]["channels"]]
    for p in policies:
        print(f"    {p:18s} " + " | ".join(
            f"k={k}: " + " ".join(
                f"{n[:6]}={np.mean([lad[s][k]['results'][p]['share_by_channel'][n] for s in seeds]):.2f}"
                for n in names) for k in shares))
    print()


def cross_mode_check(geo):
    """The free consistency check: at FULL sharing the two splice directions are the same DGP.

    `share_top=L` and `share_bottom=L` both hand structA the tree's entire table set under an
    independently drawn root, so the two sweeps share their top rung by construction. Any gap
    between them there is instrument noise, not geometry -- which makes it a measured noise
    floor for cross-mode comparisons, obtained for free, in the same spirit as the
    `oracle_shared == oracle` floor at k=0 in the published sweep.
    """
    archs = {a for _, a in geo}
    for arch in sorted(archs):
        top, bot = geo.get(("top", arch)), geo.get(("bottom", arch))
        if not top or not bot:
            continue
        k = max(set(next(iter(top.values()))) & set(next(iter(bot.values()))))
        seeds = sorted(set(top) & set(bot))
        if not seeds:
            continue
        print("=" * 100)
        print(f"CROSS-MODE CHECK [fm_arch={arch}] -- at k={k} the two splices are the SAME DGP")
        print("=" * 100)
        for field in ("structA_value", "structB_value", "noise_value", "doubling_value"):
            mt, st, _ = _mean_sd([top[s][k][field] for s in seeds])
            mb, sb, _ = _mean_sd([bot[s][k][field] for s in seeds])
            d, dsd, _ = _mean_sd([top[s][k][field] - bot[s][k][field] for s in seeds])
            print(f"    {field:22s} top {mt:+.4f}+-{st:.4f} | bottom {mb:+.4f}+-{sb:.4f} "
                  f"| diff {d:+.4f}+-{dsd:.4f}")
        print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures", default=os.path.join(HERE, "figures"))
    args = ap.parse_args()
    print(f"reading {args.figures}\n")
    geo = load_geometry(args.figures)
    # "top" (the published sweep) before "bottom"; within a mode, "block" (the ladder's own
    # readout) before the two architectural controls
    for key in sorted(geo, key=lambda k: (k[0] != "top", k[1] != "block", k)):
        report_geometry(geo, key)
    cross_mode_check(geo)
    lad = load_ladder(args.figures)
    for mode in sorted(lad, key=lambda m: m != "top"):
        report_ladder(lad, mode)


if __name__ == "__main__":
    main()
