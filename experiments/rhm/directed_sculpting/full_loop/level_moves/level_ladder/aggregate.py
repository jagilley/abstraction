"""Aggregate `level_ladder` runs across seeds: the prize, the climb, and the discriminator.

`--pattern` is REQUIRED whenever more than one sweep is mirrored into `figures/`. Results are
keyed by seed, so `ladder_deep_s1` and `ladder_static_s1` are the same seed on different
schedules and would silently overwrite each other -- the exact bug
[`../aggregate.py`](../aggregate.py) had to fix mid-node. Mixed layouts are flagged rather
than averaged.

Run from experiments/:
  python3 rhm/directed_sculpting/full_loop/level_moves/level_ladder/aggregate.py \
      --pattern 'ladder_deep_*'
  python3 rhm/directed_sculpting/full_loop/level_moves/level_ladder/aggregate.py \
      --pattern 'ladder_deep_*' --against 'ladder_static_*'
"""

import argparse
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def load(pattern, figdir=None):
    figdir = figdir or os.path.join(HERE, "figures")
    runs = {}
    for path in sorted(glob.glob(os.path.join(figdir, pattern, "results.json"))):
        with open(path) as f:
            r = json.load(f)
        runs[os.path.basename(os.path.dirname(path))] = r
    if not runs:
        raise SystemExit(f"no runs matched {pattern!r} under {figdir}")
    layouts = {json.dumps([r["config"]["tree_depth"], r["config"]["struct_depths"],
                           r["config"]["max_level"], r["config"]["schedule"]])
               for r in runs.values()}
    if len(layouts) > 1:
        raise SystemExit(
            f"MIXED LAYOUTS under pattern {pattern!r}:\n  " + "\n  ".join(sorted(layouts))
            + "\nNarrow --pattern; these cells are not comparable and averaging them is the "
              "bug this guard exists for.")
    return runs


def ms(xs):
    """mean +- sd over seeds, plus a paired t against 0 when n > 1."""
    a = np.asarray([x for x in xs if x is not None and np.isfinite(x)], dtype=float)
    if a.size == 0:
        return float("nan"), float("nan"), float("nan")
    sd = float(a.std(ddof=1)) if a.size > 1 else 0.0
    t = float(a.mean() / (sd / np.sqrt(a.size))) if a.size > 1 and sd > 0 else float("nan")
    return float(a.mean()), sd, t


def climb_slope(rows):
    """Least-squares slope of the mean level of TREE spend against the round's damage depth.

    This is the direct climbing statistic: positive means the allocator moved up the
    hierarchy as the error moved down into it. Computed per seed so the spread is visible --
    `level_moves` retired a headline for being one seed of three, and this node inherits that
    caution rather than re-learning it.
    """
    d = np.array([r["damage_level"] for r in rows], dtype=float)
    y = np.array([r["mean_level"] for r in rows], dtype=float)
    ok = np.isfinite(y)
    if ok.sum() < 2 or d[ok].std() == 0:
        return float("nan")
    return float(np.polyfit(d[ok], y[ok], 1)[0])


def summarise(runs):
    arms = list(next(iter(runs.values()))["results"].keys())
    out = {}
    for a in arms:
        per = [r["results"][a] for r in runs.values() if a in r["results"]]
        out[a] = {
            "tree_err": ms([p["tree_err_mean"] for p in per]),
            "tree_err_last": ms([p["tree_err_last"] for p in per]),
            "ballistic": ms([p["ballistic_mean"] for p in per]),
            "tree_share": ms([p["tree_share_mean"] for p in per]),
            "mean_level": ms([p["mean_level_mean"] for p in per]),
            "climb_slope": ms([climb_slope(p["rounds"]) for p in per]),
            "per_seed_slope": [climb_slope(p["rounds"]) for p in per],
            "n": len(per),
        }
    return arms, out


def drives_report(pattern, figdir=None):
    """G-E across seeds: does any endogenous drive track the oracle, and is it above its null?

    Two things are read here and the second is the one that decides. The **climb** is whether
    a drive's implied mean level rises with the damage depth, reported per seed because a
    mean over seeds that disagree in sign is exactly what `level_moves` §4 had to retire. The
    **floor** is the drive's own value in the off-tree cells: their ground-truth relevance is
    0.000 at every level by construction (P1/G1), so whatever a drive shows there is what "no
    signal" looks like in that drive's own units. A tree cell within a few multiples of it is
    not a measurement -- the lesson `shared_surface` §4 paid for, where every reported cell
    turned out to lie inside a floor nobody had measured.
    """
    figdir = figdir or os.path.join(HERE, "figures")
    runs = {}
    for path in sorted(glob.glob(os.path.join(figdir, pattern, "results.json"))):
        with open(path) as f:
            runs[os.path.basename(os.path.dirname(path))] = json.load(f)
    if not runs:
        raise SystemExit(f"no runs matched {pattern!r} under {figdir}")
    first = next(iter(runs.values()))
    levels = sorted(first["per_damage"], key=int)
    keys = list(first["per_damage"][levels[0]]["drives"])
    tree = first["tree_cells"]
    off = [c for c in first["cells"] if c not in tree]

    print(f"{len(runs)} run(s) matched {pattern!r}: {', '.join(sorted(runs))}")
    print(f"\n{'=' * 104}\n=== G-E  implied mean level of tree spend, by damage depth "
          f"(mean ± sd over seeds)\n{'=' * 104}")
    print(f"{'drive':16s} " + "  ".join(f"{'d=' + lv:>16s}" for lv in levels)
          + f"   {'slope/depth (per seed)':>28s}")
    orc = [ms([r["per_damage"][lv]["oracle_mean_level"] for r in runs.values()])
           for lv in levels]
    print(f"{'ORACLE':16s} " + "  ".join(f"{m:>10.3f} ±{sd:<4.3f}" for m, sd, _ in orc))
    for k in keys:
        cellv = [[r["per_damage"][lv]["mean_level"][k] for r in runs.values()]
                 for lv in levels]
        per_seed = []
        for i in range(len(runs)):
            y = np.array([cellv[j][i] for j in range(len(levels))], float)
            d = np.array([float(lv) for lv in levels])
            per_seed.append(np.polyfit(d, y, 1)[0] if np.isfinite(y).all() else np.nan)
        cells_s = "  ".join(f"{np.nanmean(c):>10.3f} ±{np.nanstd(c, ddof=1) if len(c) > 1 else 0:<4.3f}"
                            for c in cellv)
        print(f"{k:16s} " + cells_s + "   ["
              + ", ".join("nan" if not np.isfinite(x) else f"{x:+.3f}" for x in per_seed) + "]")

    print(f"\n{'=' * 104}\n=== G-E  signal vs its OWN null -- tree cells against the "
          f"off-tree cells of the same drive\n{'=' * 104}")
    print("  off-tree ground-truth relevance is exactly 0.000 at every level (P1/G1), so the\n"
          "  off-tree spread IS the drive's noise floor in its own units.\n")
    for k in keys:
        print(f"  {k}")
        for lv in levels:
            tv = {c: [r["per_damage"][lv]["drives"][k][c] for r in runs.values()] for c in tree}
            ov = np.array([x for c in off for x in
                           [r["per_damage"][lv]["drives"][k][c] for r in runs.values()]])
            floor = float(np.abs(ov).max()) if ov.size else float("nan")
            row = "  ".join(f"{c.split('|')[1]}={np.mean(v):+.4f}±{np.std(v, ddof=1) if len(v) > 1 else 0:.4f}"
                            for c, v in tv.items())
            best = max(tv, key=lambda c: abs(np.mean(tv[c])))
            snr = abs(np.mean(tv[best])) / floor if floor > 0 else float("inf")
            print(f"    d={lv}  {row}   | null |max|={floor:.4f}  best/null={snr:5.1f}x")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drives", default=None,
                    help="read `drive_check` (G-E) runs instead of ladder runs, e.g. "
                         "'drive_e1*'")
    ap.add_argument("--pattern", required=False,
                    help="glob under figures/, e.g. 'ladder_deep_*'. REQUIRED -- see module "
                         "docstring.")
    ap.add_argument("--against", default=None,
                    help="a second pattern (the STATIC-schedule control) to difference "
                         "against; this is the necessity discriminator")
    ap.add_argument("--figdir", default=None)
    args = ap.parse_args()

    if args.drives:
        drives_report(args.drives, args.figdir)
        return
    if not args.pattern:
        raise SystemExit("--pattern is required unless --drives is given")

    runs = load(args.pattern, args.figdir)
    cfg = next(iter(runs.values()))["config"]
    sched = cfg["schedule"]
    arms, summ = summarise(runs)
    print(f"{len(runs)} run(s) matched {args.pattern!r}: {', '.join(sorted(runs))}")
    print(f"schedule (round -> damage depth): {sched}   max_level={cfg['max_level']}")

    print(f"\n{'=' * 104}\n=== THE PRIZE  (mean ± sd over {len(runs)} seeds)\n{'=' * 104}")
    print(f"{'arm':16s} {'tree FM err ↓':>20s} {'ballistic ↑':>18s} {'tree share':>14s} "
          f"{'mean level':>14s}")
    for a in arms:
        s = summ[a]
        print(f"{a:16s} {s['tree_err'][0]:>12.4f} ± {s['tree_err'][1]:<5.4f} "
              f"{s['ballistic'][0]:>10.3f} ± {s['ballistic'][1]:<5.3f} "
              f"{s['tree_share'][0]:>8.3f} ± {s['tree_share'][1]:<4.3f} "
              f"{s['mean_level'][0]:>8.3f} ± {s['mean_level'][1]:<4.3f}")

    if "uniform" in summ and "oracle_level" in summ:
        prize = summ["uniform"]["tree_err"][0] - summ["oracle_level"]["tree_err"][0]
        print(f"\n  uniform -> oracle_level prize: {prize:+.4f} tree FM error")
        if abs(prize) < 5e-3:
            print("  ** THE PRIZE IS AT THE NOISE FLOOR. Nothing below is readable: a "
                  "privileged\n     per-cell oracle buys nothing, so no allocation rule can "
                  "be scored on this axis. **")
        else:
            for a in arms:
                if a in ("uniform", "oracle_level"):
                    continue
                rec = (summ["uniform"]["tree_err"][0] - summ[a]["tree_err"][0]) / prize
                print(f"    {a:16s} recovers {rec * 100:5.1f}% of it")

    if "channel_only" in summ:
        print(f"\n  THE LOAD-BEARING CONTROL -- does the level index buy anything?")
        for a in ("visits_level", "value_level", "satiety_level"):
            if a not in summ:
                continue
            d = summ["channel_only"]["tree_err"][0] - summ[a]["tree_err"][0]
            print(f"    channel_only - {a:14s} = {d:+.4f}   "
                  f"(positive = the level index helps)")

    print(f"\n{'=' * 104}\n=== THE CLIMB -- mean level of TREE spend by round\n{'=' * 104}")
    print(f"{'arm':16s} " + " ".join(f"r{i + 1:<5d}" for i in range(len(sched)))
          + f"   {'slope vs damage depth':>24s}")
    print(f"{'damage depth':16s} " + " ".join(f"{d:<6d}" for d in sched))
    for a in arms:
        rows = [[r["results"][a]["rounds"][i]["mean_level"] for r in runs.values()]
                for i in range(len(sched))]
        m, sd, t = summ[a]["climb_slope"]
        per = ", ".join(f"{x:+.3f}" for x in summ[a]["per_seed_slope"])
        print(f"{a:16s} " + " ".join(f"{np.nanmean(x):<6.2f}" for x in rows)
              + f"   {m:>+8.4f} ± {sd:<6.4f} t={t:>+5.2f}   [{per}]")

    if args.against:
        ctrl = load(args.against, args.figdir)
        _, csum = summarise(ctrl)
        print(f"\n{'=' * 104}\n=== THE DISCRIMINATOR -- deepening vs STATIC damage "
              f"({args.against})\n{'=' * 104}")
        print("  If the climb slope is as large under a static schedule, the allocator is "
              "not\n  tracking necessity -- it is watching deep moves become relatively more "
              "useful as\n  the FM improves. Only the DIFFERENCE is evidence of necessity "
              "tracking.")
        print(f"\n{'arm':16s} {'deepening slope':>18s} {'static slope':>18s} "
              f"{'difference':>18s}")
        for a in arms:
            if a not in csum:
                continue
            dm, dsd, _ = summ[a]["climb_slope"]
            cm, csd, _ = csum[a]["climb_slope"]
            print(f"{a:16s} {dm:>+12.4f} ± {dsd:<4.4f} {cm:>+12.4f} ± {csd:<4.4f} "
                  f"{dm - cm:>+18.4f}")

    print(f"\n{'=' * 104}\n=== G-L (the gate, from run 1) -- best Δd* per tree cell by damage "
          f"depth\n{'=' * 104}")
    g = next(iter(runs.values()))["gate"]
    tree_names = [n for n in next(iter(g.values())) if n.startswith("tree|")]
    print(f"{'damage':>7s}  " + "  ".join(f"{n:>10s}" for n in tree_names) + f"  {'argmax':>7s}")
    for lv in sorted(g, key=int):
        best = [g[lv][n]["best"] for n in tree_names]
        print(f"{lv:>7s}  " + "  ".join(f"{b:>+10.3f}" for b in best)
              + f"  {tree_names[int(np.argmax(best))].split('|')[1]:>7s}")


if __name__ == "__main__":
    main()
