"""Cross-seed aggregator for Cut #5 Piece 1 (the saturation gate). Local, CPU-only, no Modal.

The gate's call is a RATIO between FM variants at n_windows=384 samples, so it must be read
across seeds, not from one. This reduces the per-seed `results.json` files to the three tables
the decision actually rests on:

  1. THE K-SWEEP — frontier mass, R_res_participation and relative residual vs rollout horizon,
     mean +/- sd over seeds, per (n_links, variant), on the task probe. This is the saturation
     question: is there a window of k where the FM neither saturates (frontier -> 0, the noise
     floor that explains all three of this repo's historical rank failures) nor diverges
     (frontier -> 1, nothing explained)?
  2. THE POSITIVE CONTROL — stale/matched frontier ratio per k, with the per-seed spread and a
     sign count. An instrument that cannot separate a known-wrong FM from a matched one at ANY k
     cannot detect an expansion, and that is the LATENT verdict.
  3. THE INSTRUMENT CHECKS — beta with its n_dirs, the beta gap across the 4x capacity range
     (invariance is what says beta reads the computation rather than the FM), and the
     mean-blindness fraction (how much of the residual is a constant offset that frontier mass,
     rho and beta all discard by construction).

Usage:
    cd experiments/
    python3 mjc/expansion/saturation_gate_agg.py --tags gate_s0 gate_s1 gate_s2
"""

import argparse
import json
import os

import numpy as np


def load(tags):
    here = os.path.dirname(os.path.abspath(__file__))
    out = []
    for t in tags:
        p = os.path.join(here, "figures", f"expansion_gate_{t}", "results.json")
        if not os.path.exists(p):
            print(f"[warn] missing {p} — skipping")
            continue
        with open(p) as fh:
            out.append((t, json.load(fh)))
    if not out:
        raise SystemExit("no results found")
    return out


def cell(runs, n, probe, variant, k, key):
    """The value of `key` at one (n, probe, variant, k) cell, one entry per seed."""
    vals = []
    for _, d in runs:
        for r in d["rows"]:
            if (r["n_links"] == n and r["probe"] == probe
                    and r["variant"] == variant and r["k"] == k):
                vals.append(r[key])
    return np.asarray(vals, dtype=float)


def ms(v):
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan")
    return float(v.mean()), float(v.std())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--probe", default="task", choices=["task", "broad"])
    args = ap.parse_args()

    runs = load(args.tags)
    cfg = runs[0][1]["config"]
    ks, ns = cfg["k_list"], cfg["n_links_list"]
    H = cfg["plan_H"]
    floor, ceil_ = cfg["frontier_floor"], cfg["frontier_ceiling"]
    print(f"\nseeds: {[t for t, _ in runs]}   probe: {args.probe}   "
          f"plan_H (ballistic horizon) = {H}   window: {floor} < frontier < {ceil_}")

    # ---------------------------------------------------------------- 1. the k-sweep
    for n in ns:
        print("\n" + "=" * 100)
        print(f"1. K-SWEEP — n_links={n} (d_state={2*n}), probe={args.probe}, "
              f"mean +/- sd over {len(runs)} seeds")
        print("=" * 100)
        print(f"{'k':>3} | {'rel_resid':>15} | {'frontier mass':>17} | "
              f"{'R_res_particip.':>17} | {'window':>7}")
        print("-" * 100)
        for k in ks:
            rr = ms(cell(runs, n, args.probe, "matched", k, "rel_residual"))
            fm = ms(cell(runs, n, args.probe, "matched", k, "frontier_mass"))
            pr = ms(cell(runs, n, args.probe, "matched", k, "R_res_participation"))
            inw = floor < fm[0] < ceil_
            mark = "  IN" if inw else "   -"
            star = " <- plan_H" if k == H else ""
            print(f"{k:>3} | {rr[0]:>8.4f}+-{rr[1]:<5.4f} | {fm[0]:>9.4f}+-{fm[1]:<6.4f} | "
                  f"{pr[0]:>8.2f}+-{pr[1]:<5.2f}/{2*n:<2d} | {mark}{star}")

    # ------------------------------------------------- 2. the stale positive control
    for n in ns:
        print("\n" + "=" * 100)
        print(f"2. POSITIVE CONTROL — n_links={n}: does the instrument see a KNOWN-WRONG FM?")
        print(f"   stale = trained field-free (b={cfg['b0']}), graded under the curl "
              f"(b={cfg['b1']}); matched = trained in the operating world")
        print("=" * 100)
        print(f"{'k':>3} | {'matched frontier':>18} | {'stale frontier':>18} | "
              f"{'ratio':>13} | {'seeds stale>matched':>20}")
        print("-" * 100)
        for k in ks:
            fmv = cell(runs, n, args.probe, "matched", k, "frontier_mass")
            stv = cell(runs, n, args.probe, "stale", k, "frontier_mass")
            m, s = ms(fmv), ms(stv)
            per_seed = stv / np.maximum(fmv, 1e-12)
            rm, rs = ms(per_seed)
            wins = int((per_seed > 1.0).sum())
            print(f"{k:>3} | {m[0]:>10.4f}+-{m[1]:<6.4f} | {s[0]:>10.4f}+-{s[1]:<6.4f} | "
                  f"{rm:>6.2f}+-{rs:<5.2f} | {wins:>10d}/{len(per_seed)}")

    # ---------------------------------------------------- 3. the instrument checks
    print("\n" + "=" * 100)
    print("3. INSTRUMENT CHECKS")
    print("=" * 100)
    print("\n  (a) beta — the shape. Capacity-invariance (|matched - small| over a 4x FM range)")
    print("      is the gate that would catch this readout if the basis were circular; rhm")
    print("      measures +-0.01 across 4x. n_dirs is the number of points the power law is fit")
    print("      over: rhm's own guard is 8, so n=3 (6 dims) cannot clear it at all.")
    print(f"\n{'n':>3} {'k':>3} | {'beta matched':>16} | {'beta small':>16} | "
          f"{'|delta|':>8} | {'R2':>6} | {'n_dirs':>6}")
    print("-" * 100)
    for n in ns:
        for k in ks:
            bm = ms(cell(runs, n, args.probe, "matched", k, "beta"))
            bs = ms(cell(runs, n, args.probe, "small", k, "beta"))
            r2 = ms(cell(runs, n, args.probe, "matched", k, "beta_r2"))
            nd = ms(cell(runs, n, args.probe, "matched", k, "beta_n_dirs"))
            d = abs(bm[0] - bs[0])
            print(f"{n:>3} {k:>3} | {bm[0]:>8.3f}+-{bm[1]:<6.3f} | {bs[0]:>8.3f}+-{bs[1]:<6.3f} | "
                  f"{d:>8.3f} | {r2[0]:>6.3f} | {nd[0]:>6.0f}")

    print("\n  (b) mean-blindness — fraction of residual energy that is a CONSTANT OFFSET.")
    print("      frontier mass, rho and beta all mean-center, so this fraction is discarded by")
    print("      construction. A drift-induced residual is systematic and one-signed, so a high")
    print("      fraction here means the instrument is structurally unable to see the drift.")
    print(f"\n{'n':>3} {'k':>3} | {'matched':>16} | {'stale':>16}")
    print("-" * 100)
    for n in ns:
        for k in ks:
            mm = ms(cell(runs, n, args.probe, "matched", k, "resid_mean_frac"))
            sm = ms(cell(runs, n, args.probe, "stale", k, "resid_mean_frac"))
            print(f"{n:>3} {k:>3} | {mm[0]:>8.3f}+-{mm[1]:<6.3f} | {sm[0]:>8.3f}+-{sm[1]:<6.3f}")

    # ---------------------------------------------------------------- the verdicts
    print("\n" + "=" * 100)
    print("PER-SEED VERDICTS (the script's own pre-registered call)")
    print("=" * 100)
    for t, d in runs:
        for n_str, v in d["verdict"].items():
            print(f"  {t:>10}  n={n_str}: window={v['usable_window_k']}  "
                  f"rises={v['frontier_rises_across_window']}  "
                  f"best stale sep x{(v['best_stale_ratio'] or float('nan')):.2f} "
                  f"at k={v['best_separating_k']}  ==> {v['call']}")
    print("\n  env sanity (max|q| per probe; ArmEnv.wrapped() flags > 3.0 rad):")
    for t, d in runs:
        for n_str, pm in d["env_meta"].items():
            bits = "  ".join(f"{p}: {m['max_absq']:.2f}{'(WRAPPED)' if m['wrapped'] else ''}"
                             for p, m in pm.items())
            print(f"    {t:>10}  n={n_str}:  {bits}")
    print()


if __name__ == "__main__":
    main()
