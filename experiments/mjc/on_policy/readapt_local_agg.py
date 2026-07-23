"""Aggregate E2 (`readapt_local.py`) across seeds from committed `results.json` files.

The E2 headline is the IN-REGION vs OUT-REGION FM-error recovery split (control saturates and
cannot see it -- E1 / drift_value_loop Cut 3), plus the in-region collection-share trace that is
the coverage-growth mechanism. This aggregator foregrounds those.

Usage:
    # after pulling each seed's results.json into figures/readapt_local_<tag>/
    python3 mjc/on_policy/readapt_local_agg.py --tags loc_s0 loc_s1 loc_s2
"""

import argparse
import glob
import json
import os
import statistics as st


def agg(vals):
    vals = [v for v in vals if v is not None and v == v]
    if not vals:
        return float("nan"), float("nan")
    if len(vals) == 1:
        return vals[0], float("nan")
    return st.mean(vals), st.stdev(vals) / len(vals) ** 0.5


def m_to_recover(curve, ms, thresh=0.8):
    """Transitions to first reach `thresh` recovery (and STAY >= thresh - 0.1 after).

    Robust to the early-NEGATIVE dip the on-policy bootstrap produces (the first in-region
    transitions, gathered with a stale FM, DEFLECT through the field and briefly make the
    competent-reach probe worse). `share_in_first_milestone` is meaningless across a sign change;
    'how many transitions to durable recovery' is the honest stretch metric. Returns the milestone,
    or the max milestone if the curve never gets there (a censored value, flagged by the caller)."""
    for i, m in enumerate(ms):
        if curve[i] >= thresh and all(c >= thresh - 0.1 for c in curve[i:]):
            return m, False
    return ms[-1], True                                   # censored: never durably recovered


def load(tag, figdir):
    for p in (os.path.join(figdir, f"readapt_local_{tag}", "results.json"),
              os.path.join(figdir, tag, "results.json")):
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
        runs.append(d); print(f"[read] {t}  (region_frac={d.get('region_frac', float('nan')):.0%})")
    if not runs:
        print("nothing to aggregate"); return

    ms = runs[0]["milestones"]
    arms = runs[0]["config"]["arms"]
    first_nz = next(i for i, m in enumerate(ms) if m > 0)

    print(f"\n=== E2: LOCAL-drift recovery, {len(runs)} seeds, milestones = {ms} ===")
    rf = agg([r.get("region_frac") for r in runs])
    print(f"    region = {100 * rf[0]:.0f}% ± {100 * rf[1]:.0f}% of matched-reach transitions\n")

    for key in ("fm_err_in", "fm_err_out", "ballistic_cem"):
        cl = agg([r["ceiling"][key] for r in runs])
        tag = {"fm_err_in": "IN-REGION FM error (the readout that matters)",
               "fm_err_out": "OUT-REGION FM error (control: unchanged dynamics)",
               "ballistic_cem": "ballistic control (saturates -- cannot see the split)"}[key]
        print(f"[{key}] {tag}  ceiling={cl[0]:.4f}±{cl[1]:.4f}")
        for arm in arms:
            curves = [r["summary"][arm][key]["recovery_curve"] for r in runs if arm in r["summary"]]
            per_m = [agg([c[i] for c in curves]) for i in range(len(ms))]
            shares = agg([r["summary"][arm][key]["share_in_first_milestone"] for r in runs])
            body = " ".join(f"{m[0]:+.2f}" for m in per_m)
            label = "STEP" if (shares[0] or 0) > 0.8 else "curve"
            print(f"    {arm:>16s} {body}")
            print(f"    {'':>16s}   m{ms[first_nz]}-share = {100 * (shares[0] or 0):.0f}%"
                  f"±{100 * (shares[1] or 0):.0f}% [{label}]")
        print()

    print("=== the mechanism: in-region COLLECTION share per milestone "
          "(does on_policy start low and grow?) ===")
    for arm in arms:
        traces = [r["summary"][arm].get("in_share_trace", []) for r in runs if arm in r["summary"]]
        if not traces or not traces[0]:
            continue
        L = min(len(t) for t in traces)
        per_m = [agg([t[i] for t in traces]) for i in range(L)]
        print(f"    {arm:>16s} " + " ".join(f"{m[0]:.0%}" for m in per_m))
    print(f"    {'(milestones)':>16s} " + " ".join(f"{m:>4d}" for m in ms[first_nz:]))

    print("\n=== E2 verdict: transitions to durable 0.8 recovery (the stretch metric) ===")
    print("    (E1 under GLOBAL drift: on_policy recovered as fast as the teleporters. Prediction "
          "here: on_policy needs MORE transitions than teleport_matched to repair the region.)")
    for key, lab in (("fm_err_in", "IN-REGION"), ("fm_err_out", "out-region (control)"),
                     ("ballistic_cem", "ballistic")):
        print(f"\n  [{key}] {lab}")
        for arm in arms:
            ns, cens = [], 0
            for r in runs:
                if arm not in r["summary"]:
                    continue
                m, c = m_to_recover(r["summary"][arm][key]["recovery_curve"], ms)
                ns.append(m); cens += int(c)
            mm = agg(ns)
            note = f"  ({cens}/{len(ns)} seeds never reached 0.8)" if cens else ""
            print(f"    {arm:>16s}: {mm[0]:>7.0f} ± {mm[1]:<6.0f} transitions{note}")
    # the headline number
    tm = agg([m_to_recover(r["summary"]["teleport_matched"]["fm_err_in"]["recovery_curve"], ms)[0]
              for r in runs if "teleport_matched" in r["summary"]])
    op = agg([m_to_recover(r["summary"]["on_policy"]["fm_err_in"]["recovery_curve"], ms)[0]
              for r in runs if "on_policy" in r["summary"]])
    if tm[0] and op[0] and tm[0] == tm[0] and op[0] == op[0]:
        ratio = op[0] / tm[0] if tm[0] > 0 else float("nan")
        verdict = ("STRETCH confirmed" if op[0] > tm[0] * 1.5 else
                   "no clear stretch (on_policy ~ matched)")
        print(f"\n    -> in-region repair: teleport_matched {tm[0]:.0f} vs on_policy {op[0]:.0f} "
              f"transitions = {ratio:.1f}x  ==> {verdict}")


if __name__ == "__main__":
    main()
