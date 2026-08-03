"""Aggregate the level-indexed action-space probe across seeds.

Prints the four things the experiment exists to decide, in the order they have to be read:

  1. THE RAW PROFILE -- |ΔV| by level. This is the number the naive question asks for ("higher
     levels explain more, so the value should rate them higher"), and it is the one that cannot
     be read on its own.
  2. THE SPAN NULL -- |ΔV| by level in the DISTRACTOR channels. Their ground-truth Δd* is exactly
     0.000 at EVERY level (P1), so any rise with level there is the branching factor alone: a
     level-`ell` move rewrites `s**ell` tokens and perturbs the latent more whether or not it
     buys anything. This is the control the raw profile must be read against, and it costs
     nothing because the channels are already in the action space.
  3. WHAT THE LEVEL ACTUALLY BUYS -- best Δd* by level from the exact DP, and the DP's own top-1
     distribution over levels. If deep moves buy nothing, a value that rates them low is right.
  4. THE CALIBRATED ANSWER -- `residual`, mean ΔV after regressing ΔV on the move's true Δd*
     over all tree moves pooled. Positive = the value OVER-rates that level for what it buys.
     Reported with a per-seed slope against level, so "the value systematically over-rates depth"
     is a testable claim rather than an eyeballed one.

Run from experiments/:
  python3 rhm/directed_sculpting/full_loop/level_moves/aggregate.py
  python3 rhm/directed_sculpting/full_loop/level_moves/aggregate.py --figures <dir>
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


def _slope(xs, ys):
    x, y = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
    ok = np.isfinite(y)
    return float(np.polyfit(x[ok], y[ok], 1)[0]) if ok.sum() >= 2 else float("nan")


def _t_stat(xs):
    a = np.asarray(xs, dtype=float)
    a = a[np.isfinite(a)]
    if a.size < 2 or a.std(ddof=1) == 0:
        return float("nan")
    return float(a.mean() / (a.std(ddof=1) / np.sqrt(a.size)))


def load(figdir):
    """{arm: {seed: result}} from level_* result files, skipping --quick smokes."""
    out = {}
    for path in sorted(glob.glob(os.path.join(figdir, "level_*", "results.json"))):
        d = json.load(open(path))
        if d["config"].get("quick"):
            continue
        for arm, res in d["arms"].items():
            out.setdefault(arm, {})[d["config"]["seed"]] = res
    return out


def report_arm(arm, byseed):
    seeds = sorted(byseed)
    cells = list(byseed[seeds[0]]["cells"])
    tree_levels = sorted(byseed[seeds[0]]["tree_by_level"], key=int)

    def cell(c, field):
        return _mean_sd([byseed[s]["cells"][c][field] for s in seeds])

    print("=" * 104)
    print(f"ARM {arm} -- value trained on the {arm} action space, probed on every level move")
    print(f"       (mean +- sd over seeds {seeds})")
    print("=" * 104)
    print(f"{'cell':14s} {'span':>4s} {'|ΔV|':>16s} {'ΔV':>16s} {'best Δd*':>16s} "
          f"{'residual':>16s} {'top1':>13s} {'DP top1':>8s}")
    for c in cells:
        span = byseed[seeds[0]]["cells"][c]["span_blocks"]
        row = [cell(c, f) for f in ("abs_dvalue", "mean_dvalue", "best_dstar_gain", "residual")]
        t1, t1sd, _ = cell(c, "top1_share")
        dp1, _, _ = cell(c, "dp_top1_share")
        print(f"{c:14s} {span:>4d} " + " ".join(f"{m:>+10.4f}+-{sd:.4f}" for m, sd, _ in row)
              + f" {t1:>+7.3f}+-{t1sd:.3f} {dp1:>8.3f}")

    # (1) the raw profile, and (2) the span null it has to be read against
    print(f"\n  (1) RAW |ΔV| BY LEVEL on the tree, and (2) THE SPAN NULL: the same profile in the")
    print(f"      distractor channels, whose ground-truth Δd* is exactly 0.000 at every level.")
    print(f"      A rise in the null column is the branching factor, not relevance.")
    null_cells = [c for c in cells if not c.startswith("tree|")]
    print(f"{'level':>5s} {'tree |ΔV|':>18s} {'distractor |ΔV| (null)':>26s} {'ratio':>8s}")
    for ell in tree_levels:
        tm, tsd, _ = _mean_sd([byseed[s]["tree_by_level"][ell]["abs_dvalue"] for s in seeds])
        same = [c for c in null_cells if byseed[seeds[0]]["cells"][c]["level"] == int(ell)]
        if same:
            nm, nsd, _ = _mean_sd([np.mean([byseed[s]["cells"][c]["abs_dvalue"] for c in same])
                                   for s in seeds])
            print(f"{ell:>5s} {tm:>11.4f}+-{tsd:.4f} {nm:>19.4f}+-{nsd:.4f} {tm / nm:>8.1f}x")
        else:
            print(f"{ell:>5s} {tm:>11.4f}+-{tsd:.4f} {'(no distractor at this level)':>26s}")

    # (3) what the level actually buys, and (4) the calibrated answer
    print(f"\n  (3) WHAT THE LEVEL BUYS and (4) THE CALIBRATED ANSWER (tree moves only).")
    print(f"      residual > 0 means the value OVER-rates that level for what it actually buys.")
    print(f"{'level':>5s} {'best Δd*':>18s} {'value top1':>18s} {'DP top1':>18s} "
          f"{'residual':>18s}")
    for ell in tree_levels:
        row = [_mean_sd([byseed[s]["tree_by_level"][ell][f] for s in seeds])
               for f in ("best_dstar_gain", "top1_share", "dp_top1_share", "residual")]
        print(f"{ell:>5s} " + " ".join(f"{m:>+11.4f}+-{sd:.4f}" for m, sd, _ in row))

    lv = [int(x) for x in tree_levels]
    print(f"\n  per-seed slope against LEVEL (per level of the hierarchy):")
    for field, label in (("abs_dvalue", "raw |ΔV|"),
                         ("best_dstar_gain", "best Δd* (what it buys)"),
                         ("top1_share", "value top-1 share"),
                         ("dp_top1_share", "DP top-1 share"),
                         ("residual", "residual (calibrated)")):
        sl = [_slope(lv, [byseed[s]["tree_by_level"][e][field] for e in tree_levels])
              for s in seeds]
        m, sd, n = _mean_sd(sl)
        print(f"    {label:26s} {m:>+9.5f} +- {sd:.5f}  (n={n}, t={_t_stat(sl):+.2f}, "
              f"per-seed " + " ".join(f"{x:+.4f}" for x in sl) + ")")

    rc, rcsd, _ = _mean_sd([byseed[s]["value_vs_dp_rank_corr"] for s in seeds])
    tt, ttsd, _ = _mean_sd([byseed[s]["tree_top1_share"] for s in seeds])
    dt, _, _ = _mean_sd([byseed[s]["dp_tree_top1_share"] for s in seeds])
    ts, tssd, _ = _mean_sd([byseed[s].get("terminal_success") for s in seeds])
    sl, _, _ = _mean_sd([byseed[s]["calibration"]["slope"] for s in seeds])
    print(f"\n  value-vs-DP rank corr over ALL moves {rc:+.3f}+-{rcsd:.3f} | top-1 is a tree move: "
          f"value {tt:.3f}+-{ttsd:.3f} vs DP {dt:.3f}")
    print(f"  behaviour-policy terminal success {ts:.3f}+-{tssd:.3f} | calibration slope "
          f"ΔV per unit Δd* = {sl:+.4f}\n")


def compare_arms(data):
    """`level` minus `flat`, paired by seed: does giving the value the axis change what it does?"""
    if not {"flat", "level"} <= set(data):
        return
    seeds = sorted(set(data["flat"]) & set(data["level"]))
    if not seeds:
        return
    tree_levels = sorted(data["flat"][seeds[0]]["tree_by_level"], key=int)
    print("=" * 104)
    print("LEVEL minus FLAT, paired by seed -- the effect of putting the level axis in the "
          "action space the")
    print("value was TRAINED on. Both arms are probed on the identical move set and the "
          "identical states.")
    print("=" * 104)
    print(f"{'level':>5s} {'Δ raw |ΔV|':>18s} {'Δ value top1':>18s} {'Δ residual':>18s}")
    for ell in tree_levels:
        row = []
        for f in ("abs_dvalue", "top1_share", "residual"):
            row.append(_mean_sd([data["level"][s]["tree_by_level"][ell][f]
                                 - data["flat"][s]["tree_by_level"][ell][f] for s in seeds]))
        print(f"{ell:>5s} " + " ".join(f"{m:>+11.4f}+-{sd:.4f}" for m, sd, _ in row))
    for f, label in (("value_vs_dp_rank_corr", "value-vs-DP rank corr"),
                     ("tree_top1_share", "top-1 is a tree move"),
                     ("terminal_success", "behaviour terminal success")):
        d = [data["level"][s].get(f, np.nan) - data["flat"][s].get(f, np.nan) for s in seeds]
        m, sd, n = _mean_sd(d)
        print(f"  Δ {label:28s} {m:>+9.4f} +- {sd:.4f}  (n={n}, t={_t_stat(d):+.2f})")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures", default=os.path.join(HERE, "figures"))
    args = ap.parse_args()
    print(f"reading {args.figures}\n")
    data = load(args.figures)
    if not data:
        print("(no non-quick level_* results found)")
        return
    for arm in sorted(data, key=lambda a: a != "flat"):     # the published action space first
        report_arm(arm, data[arm])
    compare_arms(data)


if __name__ == "__main__":
    main()
