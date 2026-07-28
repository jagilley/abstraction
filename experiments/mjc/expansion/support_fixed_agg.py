"""Cross-seed aggregator for Cut #5 Piece 2 (the support-fixed null). Local, CPU-only, no Modal.

The headline is a PREDICTED NULL, so the reading order matters more than usual — a flat line is
only informative once you know (a) the drift actually happened, (b) the instrument was awake, and
(c) what "flat" looks like when nothing is drifting. This prints those in that order:

  0. DID THE DRIFT HAPPEN — the OU gain excursion per condition. A null under a drift that never
     moved is not a null about anything.
  1. WAS THE INSTRUMENT AWAKE — the embedded positive control: `frozen` (round-0 FM, never
     updated) vs `live` under drift. Piece 1 established a known-wrong FM separates by ~2.7x at
     k=14 on this substrate, so this is the in-run calibration. If frozen does not separate from
     live, nothing below is interpretable and the script says so.
  2. THE NULL ITSELF — `R_res_participation` and frontier mass per round, mean +/- sd over seeds,
     with `static` (no drift, same data, same fine-tuning) as the continued-training-alone
     control. The RHM 2x2 found these numbers rise under ordinary continued training in EVERY
     arm including open-loop, so the drift arms must be read against `static`, never against zero.
  3. THE CONFOUND CHECK — R_act. The probe is frozen, so under a support-fixed drift the plant's
     displacement covariance should barely move; any frontier movement has to be read against it.

Usage:
    cd experiments/
    python3 mjc/expansion/support_fixed_agg.py --tags sf_s0 sf_s1 sf_s2
"""

import argparse
import json
import os

import numpy as np


def load(tags):
    here = os.path.dirname(os.path.abspath(__file__))
    out = []
    for t in tags:
        p = os.path.join(here, "figures", f"support_fixed_{t}", "results.json")
        if not os.path.exists(p):
            print(f"[warn] missing {p} — skipping")
            continue
        with open(p) as fh:
            out.append((t, json.load(fh)))
    if not out:
        raise SystemExit("no results found")
    return out


def series(d, cond, variant, k, key):
    rs = [r for r in d["rows"] if r["condition"] == cond and r["variant"] == variant
          and r["k"] == k]
    return np.array([r[key] for r in sorted(rs, key=lambda r: r["round"])], dtype=float)


def stack(runs, cond, variant, k, key):
    """(n_seeds, n_rounds) — one row per seed."""
    return np.vstack([series(d, cond, variant, k, key) for _, d in runs])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--k", type=int, default=None, help="default: the run's k_primary")
    args = ap.parse_args()

    runs = load(args.tags)
    cfg = runs[0][1]["config"]
    k = args.k or cfg["k_primary"]
    conds, nr = cfg["conditions"], cfg["n_rounds"]
    print(f"\nseeds: {[t for t, _ in runs]}   n_links={cfg['n_links']}   k={k}   "
          f"rounds={nr}   {cfg['round_n']} transitions/round")

    # ------------------------------------------------------- 0. did the drift happen
    print("\n" + "=" * 96)
    print("0. DID THE DRIFT ACTUALLY HAPPEN — mean |gain - start| per round (OU walk)")
    print("=" * 96)
    for cond in conds:
        per = []
        for _, d in runs:
            g = np.array([np.mean(np.abs(np.asarray(x) - cfg["ou_mu"]))
                          for x in d["gain_log"][cond]])
            per.append(g)
        m = np.vstack(per).mean(0)
        print(f"  {cond:8s} " + "  ".join(f"r{i}:{v:.2f}" for i, v in enumerate(m)))
    print(f"  (OU: mu={cfg['ou_mu']}, sigma={cfg['ou_sigma']}, theta={cfg['ou_theta']}, "
          f"clipped to [{cfg['ou_lo']}, {cfg['ou_hi']}])")

    # -------------------------------------------- 1. was the instrument awake
    print("\n" + "=" * 96)
    print("1. EMBEDDED POSITIVE CONTROL — frozen (round-0 FM) vs live, late rounds")
    print("   Piece 1 reference: a known-wrong FM separates by ~2.7x at k=14 on this substrate.")
    print("=" * 96)
    print(f"{'condition':>10} {'variant':>8} | {'live frontier':>16} | {'variant frontier':>17} | "
          f"{'ratio':>13} | {'seeds >live':>12}")
    print("-" * 96)
    half = max(1, nr // 2)
    awake = {}
    for cond in conds:
        lv = stack(runs, cond, "live", k, "frontier_mass")[:, -half:].mean(1)
        for variant in ("frozen", "rail"):
            try:
                vv = stack(runs, cond, variant, k, "frontier_mass")[:, -half:].mean(1)
            except (ValueError, IndexError):
                continue
            r = vv / np.maximum(lv, 1e-12)
            awake[(cond, variant)] = float(r.mean())
            note = "  <- Piece 1 ref 2.7x" if variant == "rail" else ""
            print(f"{cond:>10} {variant:>8} | {lv.mean():>8.4f}+-{lv.std():<6.4f} | "
                  f"{vv.mean():>9.4f}+-{vv.std():<6.4f} | {r.mean():>6.2f}+-{r.std():<5.2f} | "
                  f"{int((r > 1.0).sum()):>5d}/{len(r)}{note}")
    drift_conds = [c for c in conds if not c.endswith("static")]

    # A far more sensitive version of the same control than the mean ratio. The frozen FM was
    # pretrained at the OU mean, so its error should track HOW FAR THE WORLD HAS WANDERED FROM
    # THE MEAN, round by round — a within-run, per-seed correlation rather than a two-number
    # comparison. `static` is the null: its excursion is identically zero, so the correlation is
    # undefined there and that is the point. If frozen's frontier tracks the excursion, the
    # instrument is demonstrably awake even when the average separation is modest.
    print("\n  corr(|gain - mu|, frozen frontier mass) across rounds, per seed —")
    print("  does the un-adapted model's frontier track how far the world has drifted?")
    tracking = {}
    for cond in drift_conds:
        cs = []
        for _, d in runs:
            exc = np.array([np.mean(np.abs(np.asarray(x) - cfg["ou_mu"]))
                            for x in d["gain_log"][cond]])
            fz = series(d, cond, "frozen", k, "frontier_mass")
            if exc.std() > 1e-9 and fz.std() > 1e-9:
                cs.append(float(np.corrcoef(exc, fz)[0, 1]))
        tracking[cond] = cs
        if cs:
            print(f"    {cond:>8}: " + "  ".join(f"{c:+.2f}" for c in cs)
                  + f"   mean {np.mean(cs):+.2f}  [{sum(c > 0 for c in cs)}/{len(cs)} positive]")
    strong_ratio = any(v >= 1.2 for (c, _), v in awake.items() if c in drift_conds)
    strong_track = any(np.mean(v) >= 0.5 for v in tracking.values() if v)
    if not (strong_ratio or strong_track):
        print("\n  !! frozen neither separates from live NOR tracks the drift excursion. Either")
        print("     the drift is too small to matter or the instrument is asleep — do NOT read")
        print("     the null below as evidence about expansion.")

    # ------------------------------------------------------------- 2. the null
    for key, lab in (("R_res_participation", "R_res_participation (PRIMARY — frontier dim.)"),
                     ("frontier_mass", "frontier mass (secondary)"),
                     ("rel_residual", "relative residual (magnitude, for reference)")):
        print("\n" + "=" * 96)
        print(f"2. {lab} — live FM, per round, mean +/- sd over {len(runs)} seeds")
        print("=" * 96)
        hdr = "  ".join(f"{'r'+str(i):>12}" for i in range(nr))
        print(f"{'condition':>10} | {hdr} | {'late-early':>11}")
        print("-" * 96)
        for cond in conds:
            X = stack(runs, cond, "live", k, key)
            m, s = X.mean(0), X.std(0)
            delta = X[:, -half:].mean(1) - X[:, :half].mean(1)
            cells = "  ".join(f"{m[i]:>5.2f}+-{s[i]:<5.2f}" for i in range(nr))
            print(f"{cond:>10} | {cells} | {delta.mean():>+6.2f}+-{delta.std():<4.2f}")
        # THE 2x2. Each drift arm is differenced against its OWN geometry's no-drift control,
        # never against `static` generically — `regions` is a harder operator outright, so
        # differencing it against the one-parameter control would price the geometry as drift.
        # The third row prices the geometry itself, with drift held off.
        pairs = [(c, "regions_static" if c.startswith("regions") else "static")
                 for c in drift_conds]
        if {"static", "regions_static"} <= set(conds):
            pairs.append(("regions_static", "static"))
        for cond, base_cond in pairs:
            if base_cond not in conds:
                continue
            base = stack(runs, base_cond, "live", k, key)
            bd = base[:, -half:].mean(1) - base[:, :half].mean(1)
            X = stack(runs, cond, "live", k, key)
            xd = X[:, -half:].mean(1) - X[:, :half].mean(1)
            diff = xd - bd
            agree = int((np.sign(diff) == np.sign(diff.mean())).sum())
            what = "GEOMETRY" if cond == "regions_static" else "DRIFT"
            print(f"    [{what:8s}] {cond} minus {base_cond} (late-early, per seed): "
                  f"{diff.mean():+.3f} +- {diff.std():.3f}   "
                  f"[{agree}/{len(diff)} seeds agree in sign]")

    # -------------------------------------------------------- 3. the confound check
    print("\n" + "=" * 96)
    print("3. CONFOUND CHECK — R_act (the probe is FROZEN, so this should barely move)")
    print("=" * 96)
    for cond in conds:
        X = stack(runs, cond, "live", k, "R_act_pr")
        print(f"  {cond:>8}: R_act {X[:, 0].mean():.2f} -> {X[:, -1].mean():.2f}  "
              f"(late-early {(X[:, -half:].mean(1) - X[:, :half].mean(1)).mean():+.3f})")
    print("\n  probe out-of-training-box diagnostics (Piece 1's wrapped-probe caveat):")
    for t, d in runs:
        r0 = next(r for r in d["rows"])
        print(f"    {t:>8}: frac_outside_q_box={r0['probe_frac_outside_q_box']:.3f}  "
              f"frac_fast={r0['probe_frac_fast']:.3f}  max|q|={r0['probe_max_absq']:.2f} rad")
    print()


if __name__ == "__main__":
    main()
