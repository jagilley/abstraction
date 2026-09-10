"""[enharmonic] Derive the L5 and L6 dead zones for a CLASS-KEYED at-support series.

WHY THIS EXISTS. `gy_level` is 6 in this node, so the mirror's commit owner reads L5 while
earning L4 and L6 while earning L5, and `QuietPolicy` refuses a defaulted floor by
construction. `MEASURED_FLOORS` carries `tutti/sizing`'s null-ABBA derivation for those two
levels (0.0639 / 0.0207), but it was derived on the FLAT-keyed series of 29 banked arms, where
the L5/L6 at-support counts barely move — and `tutti/sizing/SIZING.md`'s own caveat is that the
L5/L6 KEY STREAMS are never logged anywhere, so no banked run can be re-keyed offline. A
class-keyed series is a different instrument with a different noise scale and has no banked
ancestor at all.

So the floor is measured on THIS node's own smoke tag and passed to the main run explicitly.
That is `tol_dsil`'s idiom exactly (`tutti_run`'s docstring: "a driven read with no measured
dead zone is an error ... so it is measured on the smoke tag and passed in explicitly"), and it
is the same statistic, computed by the same function, with the same skips: `conductor`'s
`null_abba` at span 1 / W 4, era boundaries and commit cycles dropped, `tol = sd(N) / 2`.

Usage (from experiments/):
    python3 rhm/practice/enharmonic/floors_l5l6.py --tag en_smoke --fetch
"""

import argparse
import json
import os
import subprocess

import numpy as np

from rhm.practice.conductor.policy import _sd, null_abba

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
VOLUME = "rhm-scaling-data"
REMOTE = "rhm_practice_enharmonic"


def fetch(tag):
    os.makedirs(FIG, exist_ok=True)
    subprocess.run(["modal", "volume", "get", "--force", VOLUME, f"{REMOTE}/{tag}", FIG],
                   check=True)


def series_of(res, level):
    """The arm's own at-support series at `level`, in the panel's ERROR CONVENTION (lower is
    better) — the exact object `QuietPolicy` is stepped on."""
    raw = (res.get("obs_hist") or {}).get(str(level))
    return None if not raw else [-float(x) for x in raw]


def skips_of(res):
    """Era boundaries and commit cycles, as `phase0_l5.py` drops them: a floor is a
    fixed-condition statistic and both are regime changes."""
    cyc, era = res["log"]["cycle"], res["log"]["era"]
    sk = {i for i in range(1, len(era)) if era[i] != era[i - 1]}
    for e in res["events"]:
        if e["kind"] == "commit" and int(e["cycle"]) in cyc:
            sk.add(cyc.index(int(e["cycle"])))
    return sk


def derive(tag, levels=(3, 4, 5, 6), arms=None, span=1, W=4):
    root = os.path.join(FIG, tag)
    out = {"tag": tag, "span": span, "W": W, "levels": {}}
    got = []
    for arm in sorted(os.listdir(root)):
        p = os.path.join(root, arm, "results.json")
        if not os.path.isfile(p) or (arms and arm not in arms):
            continue
        got.append((arm, json.load(open(p))))
    out["arms"] = [a for a, _ in got]
    for lv in levels:
        allN, allD, per = [], [], []
        for arm, res in got:
            ser = series_of(res, lv)
            if not ser:
                continue
            N_, D_ = null_abba(ser, skip=skips_of(res), span=span, W=W)
            allN += N_
            allD += D_
            per.append({"arm": arm, "n_windows": len(N_),
                        "n_distinct_values": len(set(ser)),
                        "first": -ser[0] if ser else None, "last": -ser[-1] if ser else None,
                        "quotient": ((res.get("quotient") or {}) or {}).get("mode")})
        if not allN:
            out["levels"][str(lv)] = {"n_windows": 0, "tol": None,
                                      "note": "no series — the level is not instrumented"}
            continue
        tol = _sd(allN) / 2.0
        mD = float(np.mean(allD))
        out["levels"][str(lv)] = {
            "n_windows": len(allN), "sd_N": float(_sd(allN)), "tol": float(tol),
            "mean_D": mD, "signal_over_floor": (mD / tol) if tol else None,
            "frac_D_gt_tol": float(np.mean([x > tol for x in allD])),
            "per_arm": per,
            "degenerate": bool(tol <= 0.0)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="en_smoke")
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--arms", default="")
    ap.add_argument("--span", type=int, default=1)
    ap.add_argument("--w", type=int, default=4)
    a = ap.parse_args()
    if a.fetch:
        fetch(a.tag)
    arms = tuple(x.strip() for x in a.arms.split(",") if x.strip()) or None
    out = derive(a.tag, arms=arms, span=a.span, W=a.w)
    print(json.dumps(out, indent=2))
    lv = out["levels"]
    print("\n=== the two the main run needs ===")
    for k in ("5", "6"):
        r = lv.get(k) or {}
        print(f"  tol_yield_l{k} = {r.get('tol')}   "
              f"(windows {r.get('n_windows')}, sd(N) {r.get('sd_N')}, "
              f"signal/floor {r.get('signal_over_floor')}, degenerate={r.get('degenerate')})")
    print("\n  L3/L4 are re-derived beside them as the CALIBRATION: this same statistic on the "
          "same tag\n  should land near MEASURED_FLOORS' 0.4613 / 0.5167, and if it does not, "
          "the tag is too\n  short for the L5/L6 numbers to be trusted either.")
    with open(os.path.join(HERE, f"floors_{a.tag}.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {os.path.join(HERE, f'floors_{a.tag}.json')}")


if __name__ == "__main__":
    main()
