"""Cross-seed aggregator for E5 ([`necessity.py`](necessity.py)).

The headline is a SLOPE, not a level: does the fraction of repair that lands on a region the learner
never collected in GROW as samples-per-event are starved? Reported per seed and then across seeds,
because RHM's own version of this result is "monotone across four budgets, slope -0.00585 ± 0.00186,
t = -5.44, 3/3 same sign" -- the trend is the claim and no single cell is.

Also reports, first, the two things that decide whether any of it is readable:
  * the CALIBRATION range (`pure_shared` minus `local`): if the transfer readout cannot see transfer
    where the shared component is the ONLY structure, a null in `shared` means nothing.
  * the per-event DAMAGE spread across arms: the sweep's whole premise is matched damage.

Usage:
    python3 mjc/on_policy/metered_repair/necessity_agg.py --tags nec_s0 nec_s1 nec_s2
"""

import argparse
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def load(tag):
    p = os.path.join(HERE, "figures", "necessity_" + tag, "results.json")
    if not os.path.exists(p):
        print(f"  [skip] {p} missing")
        return None
    with open(p) as fh:
        return json.load(fh)


def msem(v):
    v = np.asarray([x for x in v if x == x], float)
    if len(v) == 0:
        return float("nan"), float("nan")
    return float(v.mean()), (float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    a = ap.parse_args()

    runs = [(t, load(t)) for t in a.tags]
    runs = [(t, r) for t, r in runs if r is not None]
    if not runs:
        print("no results found")
        return
    arms = [nm for nm, _f in runs[0][1]["arms"]]
    budgets = runs[0][1]["budgets"]
    print(f"\n=== E5 necessity: {len(runs)} seed(s): {[t for t, _ in runs]} ===")
    print(f"    arms={arms}  budgets(samples/event)={budgets}")

    # ---- gate 1: matched per-event damage ------------------------------------------------
    print("\n--- GATE: per-event damage at region A, which MUST be matched across arms ---")
    for nm in arms:
        m, e = msem([r["damage"][nm]["mean"] for _, r in runs])
        print(f"  {nm:>12s}  |Δerr|/event = {m:.4f} ± {e:.4f}")
    sp, sp_e = msem([r["damage_spread"] for _, r in runs])
    print(f"  spread across arms = {sp:.1%} ± {sp_e:.1%} of the mean")

    # ---- gate 2: the held-out region really is held out ----------------------------------
    print("\n--- GATE: in-region collection share at A' (must be ~0: it receives NO budget) ---")
    for nm in arms:
        vals = [r["summary"][f"{nm}|{bg}"]["in_share_Ap"] for _, r in runs for bg in budgets]
        m, e = msem(vals)
        print(f"  {nm:>12s}  share(A') = {m:.4f} ± {e:.4f}")

    # ---- gate 3: does the instrument see transfer at all? --------------------------------
    print("\n--- GATE: instrument dynamic range (pure_shared - local, averaged over budgets) ---")
    cr, cr_e = msem([r["calibration_range"] for _, r in runs])
    print(f"  {cr:+.3f} ± {cr_e:.3f}   "
          + ("(the readout CAN see transfer -- nulls below are interpretable)" if cr > 0.05 else
             "(NEAR ZERO: the readout cannot see transfer; nothing below is interpretable)"))

    # ---- the table ----------------------------------------------------------------------
    print("\n--- TRANSFER FRACTION = repair at held-out A' / repair at collected A ---")
    print("    `local` is the control for generic model improvement: matched damage, matched")
    print("    everything, differing ONLY in whether structure is shared. So `shared - local` is the")
    print("    headline. The frozen-placebo region is a SECOND, within-arm control, reported below")
    print("    rather than folded in -- subtracting a large noisy quantity cost more precision than")
    print("    it bought (it collapsed the instrument's dynamic range from +0.136 to +0.007).")
    hdr = ("arm".rjust(12) + "".join(f"S={bg}".rjust(16) for bg in budgets))
    print("  " + hdr)
    for nm in arms:
        row = []
        for bg in budgets:
            m, e = msem([r["summary"][f"{nm}|{bg}"]["transfer_frac"] for _, r in runs])
            row.append(f"{m:+.3f}±{e:.3f}".rjust(16))
        print("  " + nm.rjust(12) + "".join(row))

    for key, lbl in (("surface_repair", "SURFACE repair (region A, collected)"),
                     ("transfer_repair", "TRANSFER repair, RAW (region A', never collected)"),
                     ("placebo_repair", "PLACEBO repair (frozen gain -- generic improvement)"),
                     ("transfer_repair_adj", "TRANSFER repair, PLACEBO-ADJUSTED"),
                     ("ballistic_auc", "ballistic control ↓ (the behavioural cash-out)")):
        print(f"\n--- {lbl} ---")
        print("  " + hdr)
        for nm in arms:
            row = []
            for bg in budgets:
                m, e = msem([r["summary"][f"{nm}|{bg}"][key] for _, r in runs])
                row.append(f"{m:+.4f}±{e:.4f}".rjust(16))
            print("  " + nm.rjust(12) + "".join(row))

    # ---- the slope, per seed then across seeds ------------------------------------------
    print("\n--- THE MIGRATION: slope of transfer_frac vs log2(samples per event) ---")
    print("    PREDICTED NEGATIVE. RHM's analogue: -0.00585 ± 0.00186 per octave, t=-5.44, 3/3.")
    for nm in arms:
        per = [r["slopes"][nm] for _, r in runs]
        m, e = msem(per)
        t = m / e if e and e > 0 else float("nan")
        nsign = sum(1 for v in per if v < 0)
        print(f"  {nm:>12s}  slope = {m:+.4f} ± {e:.4f} /octave  t={t:+.2f}  "
              f"{nsign}/{len(per)} seeds negative   per-seed={[round(v, 4) for v in per]}")
    if "shared" in arms and "local" in arms:
        per = [r["net_slope"] for _, r in runs]
        m, e = msem(per)
        t = m / e if e and e > 0 else float("nan")
        nsign = sum(1 for v in per if v == v and v < 0)
        print(f"\n  THE CONTROLLED QUANTITY  (shared - local)")
        print(f"  slope = {m:+.4f} ± {e:.4f} /octave  t={t:+.2f}  {nsign}/{len(per)} seeds negative")
        print("  per-budget difference (mean over seeds):")
        for bg in budgets:
            d = [r["summary"][f"shared|{bg}"]["transfer_frac"]
                 - r["summary"][f"local|{bg}"]["transfer_frac"] for _, r in runs]
            dm, de = msem(d)
            print(f"    S={bg:>5d}  {dm:+.3f} ± {de:.3f}")


if __name__ == "__main__":
    main()
