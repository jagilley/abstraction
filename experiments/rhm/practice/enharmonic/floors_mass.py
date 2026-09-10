"""[en_s3] Derive the merge licence's own dead zone, `tol_mass_{l}`, from a banked tag.

WHY THIS EXISTS. `en_s2b`'s yield licence read the at-support MASS rise — the share of the
next level's stream that lands on keys the learner already holds — against a STATED threshold,
`support / total`, one just-at-support key's worth of mass. Stated because the mass series had
no banked ancestor: `ClassMiner.state()` logged at-support KEYS and not their counts, so there
was nothing to run `null_abba` on. `en_s2b` logged `mass_at_support` per cycle per level
precisely so that this round would not have to state it.

So this is `floors_l5l6.py` with the series swapped and nothing else changed: `conductor`'s
`null_abba` at span 1 / W 4, era boundaries and commit cycles dropped, `tol = sd(N) / 2`,
pooled over the treated arms. That is the same statistic, computed by the same function, that
every other floor in the arc is derived by — `tol_dsil`'s idiom (measured on a banked tag of
this node's own instrument and passed to the run explicitly).

THE CALIBRATION PRINT. `floors_l5l6.py` checks its L3/L4 numbers against `MEASURED_FLOORS`,
which exists for the yield series. There is no ancestor for the mass series, so the calibration
here is against the threshold it replaces: the median `mass_floor` (`support / total`) the
banked arms' own merge records carry at that level. A derived floor far BELOW the stated one
would mean the licence was refusing rises it could measure; far above, that it was taking rises
it could not tell from noise. The ratio is printed and nothing is concluded here.

Usage (from experiments/):
    python3 rhm/practice/enharmonic/floors_mass.py --tag en_s2b
"""

import argparse
import json
import os

import numpy as np

from rhm.practice.conductor.policy import _sd, null_abba
from rhm.practice.enharmonic.floors_l5l6 import fetch, skips_of

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")


def mass_series(res, level, support):
    """The arm's own at-support MASS series at `level`, in the panel's ERROR CONVENTION (lower
    is better), which is what `null_abba` is stepped on everywhere else in the arc."""
    out = []
    for st in res["log"]["miner"]:
        cell = (st or {}).get(str(level)) or {}
        mv = (cell.get("mass_at_support") or {}).get(str(support))
        out.append(None if mv is None else -float(mv))
    if all(x is None for x in out):
        return None
    # a level not yet instrumented on a cycle carries no mass; hold the last value so the
    # series stays cycle-aligned with `skips_of`'s indices (the same convention `obs_hist` has)
    last = 0.0
    filled = []
    for x in out:
        last = last if x is None else x
        filled.append(last)
    return filled


def stated_floor(res, level):
    """The median `support / total` threshold this arm's own merge records carry for a merge
    whose NEXT level is `level` — the number the derived floor replaces."""
    vals = [e.get("mass_floor") for e in (res.get("merge_events") or [])
            if e.get("next_level") == level and e.get("mass_floor") is not None]
    return float(np.median(vals)) if vals else None


def derive(tag, levels=(3, 4, 5, 6), arms=None, span=1, W=4, support=3):
    root = os.path.join(FIG, tag)
    got = []
    for arm in sorted(os.listdir(root)):
        p = os.path.join(root, arm, "results.json")
        if not os.path.isfile(p) or (arms and arm not in arms):
            continue
        res = json.load(open(p))
        # the TREATED arms only: a flat yoke's miner is `MC.Miner`, whose `state()` carries no
        # mass at all, and pooling a class-keyed series with nothing is not pooling.
        if not (res.get("quotient") or {}).get("mode"):
            continue
        got.append((arm, res))
    out = {"tag": tag, "span": span, "W": W, "support": support,
           "arms": [a for a, _ in got], "levels": {}}
    for lv in levels:
        allN, allD, per = [], [], []
        for arm, res in got:
            ser = mass_series(res, lv, support)
            if not ser:
                continue
            N_, D_ = null_abba(ser, skip=skips_of(res), span=span, W=W)
            allN += N_
            allD += D_
            per.append({"arm": arm, "n_windows": len(N_),
                        "n_distinct_values": len(set(ser)),
                        "first": -ser[0], "last": -ser[-1],
                        "stated_floor": stated_floor(res, lv)})
        if not allN:
            out["levels"][str(lv)] = {"n_windows": 0, "tol": None,
                                      "note": "no mass series — the level is not instrumented"}
            continue
        tol = _sd(allN) / 2.0
        mD = float(np.mean(allD))
        st = [q["stated_floor"] for q in per if q["stated_floor"] is not None]
        out["levels"][str(lv)] = {
            "n_windows": len(allN), "sd_N": float(_sd(allN)), "tol": float(tol),
            "mean_D": mD, "signal_over_floor": (mD / tol) if tol else None,
            "frac_D_gt_tol": float(np.mean([x > tol for x in allD])),
            "stated_floor_median": (float(np.median(st)) if st else None),
            "derived_over_stated": (float(tol / np.median(st)) if st and np.median(st) else None),
            "per_arm": per, "degenerate": bool(tol <= 0.0)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="en_s2b")
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--arms", default="")
    ap.add_argument("--span", type=int, default=1)
    ap.add_argument("--w", type=int, default=4)
    ap.add_argument("--support", type=int, default=3)
    a = ap.parse_args()
    if a.fetch:
        fetch(a.tag)
    arms = tuple(x.strip() for x in a.arms.split(",") if x.strip()) or None
    out = derive(a.tag, arms=arms, span=a.span, W=a.w, support=a.support)
    print(json.dumps(out, indent=2))
    print("\n=== the four the main run needs (pass 0 for a level with no series) ===")
    for k in ("3", "4", "5", "6"):
        r = out["levels"].get(k) or {}
        print(f"  tol_mass_l{k} = {r.get('tol')}   (windows {r.get('n_windows')}, "
              f"sd(N) {r.get('sd_N')}, signal/floor {r.get('signal_over_floor')}, "
              f"degenerate={r.get('degenerate')})")
    print("\n=== the calibration: derived against the STATED support/total it replaces ===")
    for k in ("3", "4", "5", "6"):
        r = out["levels"].get(k) or {}
        print(f"  L{k}: derived {r.get('tol')}  stated(median) {r.get('stated_floor_median')}  "
              f"ratio {r.get('derived_over_stated')}")
    with open(os.path.join(HERE, f"floors_mass_{a.tag}.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {os.path.join(HERE, f'floors_mass_{a.tag}.json')}")


if __name__ == "__main__":
    main()
