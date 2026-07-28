"""Cross-seed aggregator for Cut #5 Piece 3 (the dimensionality-axis calibration). Local, CPU-only.

This run is a CONTROL, so read it as a pass/fail on the instrument rather than as a result about
motor learning. The question is narrow:

    Piece 1 certified the frontier instrument responds to a WRONG MODEL (error).
    Piece 2 read a null on the DIMENSIONALITY axis (flat under support-fixed drift).
    Does the instrument respond to dimensionality at all?

If `R_res_participation` separates a 2-DOF-experienced model from a 5-DOF-experienced one — on a
probe that is byte-identical across every round and condition, so `R_act` cannot move — then
Piece 2's null is attributable to the plant. If it does not, the null is attributable to the
readout, and the shape half of the instrument does not work at this state dimension.

TWO THINGS TO READ CAREFULLY, both surfaced by the endpoint check:

  1. `R_res_participation` IS NOT MONOTONE IN MODEL QUALITY. It is the participation ratio of the
     frontier spectrum {(1-rho_i) lam_i}. A model that explains NOTHING leaves the frontier equal
     to the activation spectrum, so its count lands at R_act. A model that absorbs the loud
     directions and leaves the quiet ones (the beta<1 photocopier picture) FLATTENS what remains,
     pushing the count ABOVE R_act. So "better model, higher count" is expected here, and is the
     opposite of RHM, where a good model reads 7 against R_act ~ 73. The tables therefore report
     the count relative to R_act, which is constant by construction in this run.
  2. THE SIGN OF THE SEPARATION FLIPPED WITH k in the endpoint check (full above locked at k=1
     and k=8, below at k=14). Only k=8 had both conditions inside Piece 1's readable window, so
     that is the primary — but every k is printed, because a readout whose sign depends on the
     horizon is a fact worth reporting, not one worth hiding behind a default.

Usage:
    cd experiments/
    python3 mjc/expansion/support_growing_agg.py --tags sg_s0 sg_s1 sg_s2
"""

import argparse
import json
import os

import numpy as np


def load(tags):
    here = os.path.dirname(os.path.abspath(__file__))
    out = []
    for t in tags:
        p = os.path.join(here, "figures", f"support_growing_{t}", "results.json")
        if not os.path.exists(p):
            print(f"[warn] missing {p} — skipping")
            continue
        with open(p) as fh:
            out.append((t, json.load(fh)))
    if not out:
        raise SystemExit("no results found")
    return out


def series(d, cond, k, key):
    rs = [r for r in d["rows"] if r["condition"] == cond and r["k"] == k]
    return np.array([r[key] for r in sorted(rs, key=lambda r: r["round"])], dtype=float)


def stack(runs, cond, k, key):
    return np.vstack([series(d, cond, k, key) for _, d in runs])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    args = ap.parse_args()

    runs = load(args.tags)
    cfg = runs[0][1]["config"]
    kp, ks, conds, nr = cfg["k_primary"], cfg["k_list"], cfg["conditions"], cfg["n_rounds"]
    n, SD = cfg["n_links"], 2 * cfg["n_links"]
    R_act = float(series(runs[0][1], conds[0], kp, "R_act_pr")[0])
    print(f"\nseeds: {[t for t, _ in runs]}   n_links={n} (d_state={SD})   k_primary={kp}   "
          f"rounds={nr}   lockable={cfg['lockable']}")
    print(f"R_act = {R_act:.2f}, CONSTANT by construction (one frozen full-DOF probe); "
          f"max spread over all cells = {max(d['R_act_spread'] for _, d in runs):.1e}")

    # ------------------------------------------------ did the lock actually hold
    print("\n" + "=" * 96)
    print("0. DID THE LOCK HOLD — max |q_locked - target| after a control step")
    print("=" * 96)
    for t, d in runs:
        for cond in conds:
            dv = d["lock_deviation"][cond]
            print(f"  {t:>6} {cond:>9}: max {max(dv):.2e} rad   (a first pass with motors left "
                  f"on locked joints reached 1.6e-01)")
        break   # identical construction across seeds; one is enough to show

    # ------------------------------------------------ the staircase
    print("\n" + "=" * 96)
    print(f"1. THE STAIRCASE — R_res_participation at k={kp}, mean +/- sd over {len(runs)} seeds")
    print(f"   (R_act = {R_act:.2f}. Above it = the frontier is FLATTER than the activation")
    print("    spectrum, i.e. the model absorbed the loud directions. At it = absorbed nothing.)")
    print("=" * 96)
    dof = [n - (len(cfg["lockable"]) if c == "locked" else 0 if c == "full"
                else max(0, len(cfg["lockable"]) - r // cfg["rounds_per_stage"]))
           for c in ["accretion"] for r in range(nr)]
    print(f"{'condition':>10} | " + "  ".join(f"{'r'+str(i):>11}" for i in range(nr)))
    print(f"{'accretion dof':>10} | " + "  ".join(f"{str(d)+'/'+str(n):>11}" for d in dof))
    print("-" * 96)
    for cond in conds:
        X = stack(runs, cond, kp, "R_res_participation")
        print(f"{cond:>10} | " + "  ".join(f"{X.mean(0)[i]:>5.2f}+-{X.std(0)[i]:<4.2f}"
                                           for i in range(nr)))

    # ------------------------------------------------ the calibration verdict
    print("\n" + "=" * 96)
    print("2. THE CALIBRATION — does the readout separate 2-DOF-experienced from 5-DOF?")
    print("=" * 96)
    half = max(1, nr // 2)
    for key in ("R_res_participation", "frontier_mass", "rel_residual"):
        lo = stack(runs, "locked", kp, key)[:, -half:].mean(1)
        hi = stack(runs, "full", kp, key)[:, -half:].mean(1)
        d = hi - lo
        rel = np.abs(d).mean() / max(np.std(np.concatenate([lo, hi])), 1e-12)
        # count seeds agreeing with the MEAN's sign — a bare `d > 0` count reads "0/3" for a
        # cleanly negative effect, which is maximal agreement misreported as none.
        agree = int((np.sign(d) == np.sign(d.mean())).sum())
        print(f"  {key:22s} locked {lo.mean():7.3f}+-{lo.std():.3f}   "
              f"full {hi.mean():7.3f}+-{hi.std():.3f}   "
              f"diff {d.mean():+7.3f}+-{d.std():.3f}   "
              f"[{agree}/{len(d)} seeds agree in sign]")
    print("\n  A separation that is smaller than its own seed spread is a FAILED calibration:")
    print("  it means the readout cannot see dimensionality at this state dim, and Piece 2's")
    print("  null then belongs to the instrument rather than to the plant.")

    # ------------------------------------------------ monotonicity across the staircase
    print("\n" + "=" * 96)
    print("3. IS THE ACCRETION RESPONSE GRADED? — Spearman(free DOF, readout) per seed")
    print("=" * 96)
    from scipy.stats import spearmanr
    for key in ("R_res_participation", "frontier_mass", "rel_residual"):
        rs = []
        for _, d in runs:
            y = series(d, "accretion", kp, key)
            x = np.array([r["dof_free"] for r in sorted(
                (r for r in d["rows"] if r["condition"] == "accretion" and r["k"] == kp),
                key=lambda r: r["round"])], dtype=float)
            if x.std() > 0 and y.std() > 0:
                rs.append(float(spearmanr(x, y).statistic))
        if rs:
            print(f"  {key:22s} rho = " + "  ".join(f"{v:+.2f}" for v in rs)
                  + f"   mean {np.mean(rs):+.2f}")

    # ------------------------------------------------ every k
    print("\n" + "=" * 96)
    print("4. ACROSS ALL k — readable window (0.02 < frontier < 0.90) and separation sign")
    print("=" * 96)
    print(f"{'k':>3} | {'locked frontier':>17} | {'full frontier':>17} | "
          f"{'R_res_part diff':>16} | window")
    print("-" * 96)
    for k in ks:
        lof = stack(runs, "locked", k, "frontier_mass")[:, -half:].mean(1)
        hif = stack(runs, "full", k, "frontier_mass")[:, -half:].mean(1)
        lop = stack(runs, "locked", k, "R_res_participation")[:, -half:].mean(1)
        hip = stack(runs, "full", k, "R_res_participation")[:, -half:].mean(1)
        ok = ("both" if (0.02 < lof.mean() < 0.90 and 0.02 < hif.mean() < 0.90)
              else "locked" if 0.02 < lof.mean() < 0.90
              else "full" if 0.02 < hif.mean() < 0.90 else "NEITHER")
        star = "  <- primary" if k == kp else ""
        print(f"{k:>3} | {lof.mean():>9.3f}+-{lof.std():<6.3f} | {hif.mean():>9.3f}+-{hif.std():<6.3f} | "
              f"{(hip - lop).mean():>+8.2f}+-{(hip - lop).std():<6.2f} | {ok}{star}")
    print()


if __name__ == "__main__":
    main()
