"""Aggregate E1 (`readapt_both_ways.py`) across seeds from the committed `results.json` files.

Reads `results.json` (not logs) because this experiment's payload is the recovery CURVE, which is
awkward to reconstruct line-by-line; a client evicted before `volume.commit()` is recovered with
`modal volume get mujoco-control-data readapt_both_ways/<tag>/results.json ...` into `figures/`.

Usage:
    # after pulling each seed's results.json into figures/readapt_both_ways_<tag>/
    python3 mjc/on_policy/readapt_both_ways_agg.py --tags rbw_s0 rbw_s1 rbw_s2
"""

import argparse
import glob
import json
import os
import statistics as st


def agg(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return float("nan"), float("nan")
    if len(vals) == 1:
        return vals[0], float("nan")
    return st.mean(vals), st.stdev(vals) / len(vals) ** 0.5


def load(tag, figdir):
    for p in (os.path.join(figdir, f"readapt_both_ways_{tag}", "results.json"),
              os.path.join(figdir, f"{tag}", "results.json")):
        if os.path.exists(p):
            return json.load(open(p))
    hits = glob.glob(os.path.join(figdir, f"*{tag}*", "results.json"))
    return json.load(open(hits[0])) if hits else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--figdir", default=os.path.join(os.path.dirname(__file__), "figures"))
    a = ap.parse_args()

    runs = []
    for t in a.tags:
        d = load(t, a.figdir)
        if d is None:
            print(f"[warn] no results.json for {t}"); continue
        runs.append((t, d)); print(f"[read] {t}")
    if not runs:
        print("nothing to aggregate"); return

    ms = runs[0][1]["milestones"]
    arms = runs[0][1]["config"]["arms"]
    first_nz = next(i for i, m in enumerate(ms) if m > 0)
    m400 = min(range(len(ms)), key=lambda i: abs(ms[i] - 400))

    print(f"\n=== E1: Cut 4c-arm recovery, {len(runs)} seeds, milestones = {ms} ===")
    for key in ("ballistic_cem", "reactive", "fm_err_task"):
        cl = agg([r["ceiling"][key] for _, r in runs])
        print(f"\n[{key}]  ceiling = {cl[0]:.4f} ± {cl[1]:.4f}   "
              "(recovery: 1.0 = fully recovered; per-milestone mean ± SEM)")
        for arm in arms:
            curves = [r["summary"][arm][key]["recovery_curve"] for _, r in runs
                      if arm in r["summary"]]
            per_m = [agg([c[i] for c in curves]) for i in range(len(ms))]
            r400 = agg([r["summary"][arm][key]["recovery_at_400"] for _, r in runs])
            shares = [r["summary"][arm][key]["share_in_first_milestone"] for _, r in runs]
            msh = agg(shares)
            body = "  ".join(f"{m[0]:+.2f}" for m in per_m)
            label = "STEP" if (msh[0] or 0) > 0.8 else "curve"
            print(f"    {arm:>16s} {body}")
            print(f"    {'':>16s}   recovery@m{ms[m400]}={r400[0]:.2f}±{r400[1]:.2f}  "
                  f"first-milestone share={100 * (msh[0] or 0):.0f}%±{100 * (msh[1] or 0):.0f}% "
                  f"[{label}]")

    print("\n=== ballistic-over-reactive recovery gain (pusher 4c: 4.3x; arm 4c teleport: 5.24x) ===")
    for arm in arms:
        g = agg([r["summary"][arm]["ballistic_over_reactive_gain"] for _, r in runs
                 if arm in r["summary"]])
        print(f"    {arm:>16s} {g[0]:.2f}x ± {g[1]:.2f}")

    # the headline contrast: is the recovery-curve SHAPE different across collection modes?
    print("\n=== step-vs-curve verdict (ballistic_cem first-milestone share) ===")
    for arm in arms:
        sh = agg([r["summary"][arm]["ballistic_cem"]["share_in_first_milestone"] for _, r in runs])
        print(f"    {arm:>16s}: {100 * (sh[0] or 0):.0f}% ± {100 * (sh[1] or 0):.0f}%")
    print("  (memo's hypothesis: teleport arms step, on_policy stretches into a curve -> a LOWER "
          "on_policy share)")


if __name__ == "__main__":
    main()
