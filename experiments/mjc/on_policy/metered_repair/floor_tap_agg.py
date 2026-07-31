"""Cross-seed aggregator for E4 ([`floor_tap.py`](floor_tap.py)).

Reports the two things the cut turns on, both as PAIRED per-seed deltas (the arms share one drift
trajectory and one base FM within a seed, so paired is the right test and seed-to-seed level
differences are nuisance):

  1. THE REPAIR   -- `value-red` (stock tap) vs `value` (E3's flow tap), everything else identical.
  2. THE CONJUNCTION -- `value-red` vs `visits-only` and vs `reducible-only`, i.e. whether the
     product needs BOTH terms once a visited-but-irreducible cell exists. Neither substrate has ever
     been able to ask this: RHM has no visited-but-irreducible cell, E3 had no visited-but-irrelevant
     one.

Usage:
    python3 mjc/on_policy/metered_repair/floor_tap_agg.py --tags tap_s0 tap_s1 tap_s2
"""

import argparse
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def load(tag):
    p = os.path.join(HERE, "figures", "floor_tap_" + tag, "results.json")
    if not os.path.exists(p):
        print(f"  [skip] {p} missing")
        return None
    with open(p) as fh:
        return json.load(fh)


def msem(v):
    v = np.asarray([x for x in v if x == x], float)
    if len(v) == 0:
        return float("nan"), float("nan")
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    a = ap.parse_args()

    # MERGE TAGS BY SEED. The rate-tap arms were added after the 8-arm ladder had already run, and
    # every RNG stream in the loop is policy-independent (geometry, drift trajectory, base FM, probes
    # and per-round seeds all derive from `cfg["seed"]` and the round index, never from which policies
    # were requested). So a follow-up run of just the new arms at the same seed is bit-comparable with
    # the original, and merging is cheaper than re-running eight arms to add two. Tags are grouped by
    # their trailing `_s<N>`; a policy present in more than one tag for the same seed is an error
    # rather than a silent overwrite.
    by_seed = {}
    for t in a.tags:
        r = load(t)
        if r is None:
            continue
        key = t.rsplit("_s", 1)[-1] if "_s" in t else t
        if key not in by_seed:
            by_seed[key] = (t, r)
            continue
        _t0, r0 = by_seed[key]
        dup = set(r0["summary"]) & set(r["summary"])
        if dup:
            raise SystemExit(f"seed {key}: policies {sorted(dup)} appear in more than one tag")
        r0["summary"].update(r["summary"])
        r0["results"].update(r["results"])
        r0["config"]["policies"] = list(r0["config"]["policies"]) + list(r["config"]["policies"])
    runs = [(f"s{k}", r) for k, (_t, r) in sorted(by_seed.items())]
    if not runs:
        print("no results found")
        return
    print(f"\n=== E4 floor_tap: {len(runs)} seed(s), tags merged: {a.tags} ===")

    # ---- the gates, per seed -------------------------------------------------------------
    print("\n--- GATES (a run that fails these is not interpretable) ---")
    for t, r in runs:
        g = r["geom_gate"]; fc = r["floor_certification"]
        print(f"  {t}: geom A={g['A_vis']:.2f}[{'P' if g['pass_A'] else 'F'}] "
              f"C={[round(v, 2) for v in g['on_noise_vis']]}[{'P' if g['pass_on_noise'] else 'F'}] "
              f"off<={max(g['off_vis']):.2f}[{'P' if g['pass_off'] else 'F'}] | "
              f"floor bias(curl)={fc['bias_curl_regions']:.4f} floor(noise)={fc['floor_noise_regions']:.4f} "
              f"sep={fc['separation']:+.4f}[{'P' if fc['pass_ordering'] else 'F'}] "
              f"<=ceil[{'P' if fc['pass_below_ceiling'] else 'F'}]")
    sep, sep_e = msem([r["floor_certification"]["separation"] for _, r in runs])
    print(f"  floor separation across seeds: {sep:+.4f} ± {sep_e:.4f}")

    # ---- the ladder ---------------------------------------------------------------------
    pols = [p for p in runs[0][1]["config"]["policies"]]
    keys = [("regA_err_auc", "A-err ↓", 4), ("ballistic_auc", "ballistic ↓", 4),
            ("leak_irreducible", "leak(irred)", 3), ("A_share", "A-share", 3),
            ("A_share_sd", "A-share sd", 3)]
    print("\n--- THE LADDER (mean ± sem over seeds) ---")
    hdr = "policy".rjust(15) + "".join(lbl.rjust(18) for _, lbl, _ in keys)
    print("  " + hdr)
    agg = {}
    for p in pols:
        row = []
        agg[p] = {}
        for k, _lbl, nd in keys:
            m, e = msem([r["summary"][p][k] for _, r in runs if p in r["summary"]])
            agg[p][k] = (m, e)
            row.append(f"{m:.{nd}f}±{e:.{nd}f}".rjust(18))
        print("  " + p.rjust(15) + "".join(row))

    # ---- paired contrasts ---------------------------------------------------------------
    print("\n--- PAIRED CONTRASTS on region-A FM error (negative = first arm better) ---")
    contrasts = [("value-red", "value", "THE REPAIR: stock tap - flow tap"),
                 ("value-reddelta", "value-red", "RATE - LEVEL (same honest estimator)"),
                 ("value-reddelta", "value", "RATE(honest) - RATE(counterfactual fit)"),
                 ("value-reddelta", "visits-only", "rate x relevance - relevance alone"),
                 ("reddelta-only", "reducible-only", "rate alone - level alone"),
                 ("value-red", "visits-only", "CONJUNCTION - relevance alone"),
                 ("value-red", "reducible-only", "CONJUNCTION - reducibility alone"),
                 ("value-red", "uniform", "vs uniform"),
                 ("value-red", "oracle", "vs the privileged oracle"),
                 ("value", "visits-only", "E3's contrast, for reference")]
    for x, y, lbl in contrasts:
        d = [r["summary"][x]["regA_err_auc"] - r["summary"][y]["regA_err_auc"]
             for _, r in runs if x in r["summary"] and y in r["summary"]]
        if not d:
            continue
        m, e = msem(d)
        nsign = sum(1 for v in d if v < 0)
        t = m / e if e and e > 0 else float("nan")
        print(f"  {lbl:>36s}: Δ = {m:+.4f} ± {e:.4f}  t={t:+.2f}  "
              f"{nsign}/{len(d)} seeds favour {x}")

    # ---- leak, the number RHM moved 14.2% -> 7.3% and E3 left at 29% --------------------
    print("\n--- THE LEAK into the irreducible regions (E3's `value`: 29%; RHM's repair: 14.2%->7.3%) ---")
    for p in pols:
        m, e = msem([r["summary"][p]["leak_irreducible"] for _, r in runs])
        print(f"  {p:>15s}  {m:6.1%} ± {e:.1%}")

    # ---- the taps' own ordering ---------------------------------------------------------
    print("\n--- TAP ORDERING (reducible - irreducible; positive = the tap points the right way) ---")
    for k, lbl in (("red_separation", "STOCK (measured floor)"), ("lprog_separation", "FLOW (E3)")):
        m, e = msem([r["tap_order"][k] for _, r in runs])
        print(f"  {lbl:>24s}: {m:+.4f} ± {e:.4f}")

    names = runs[0][1]["region_names"]
    kinds = runs[0][1]["region_kinds"]
    print("\n--- per-region taps (mean over seeds, from the `value-red` arm) ---")
    print("  " + "region".rjust(10) + "kind".rjust(9) + "red(stock)".rjust(12)
          + "lprog(flow)".rjust(13) + "floor".rjust(9) + "n_in".rjust(7) + "share".rjust(8))
    for j, nm in enumerate(names):
        vals = []
        for k in ("red_mean", "lprog_mean", "floor_mean", "n_in_mean", "share_mean"):
            m, _ = msem([r["summary"]["value-red"][k][j] for _, r in runs
                         if "value-red" in r["summary"]])
            vals.append(m)
        print("  " + nm.rjust(10) + kinds[j].rjust(9) + f"{vals[0]:.3f}".rjust(12)
              + f"{vals[1]:.4f}".rjust(13) + f"{vals[2]:.4f}".rjust(9)
              + f"{vals[3]:.0f}".rjust(7) + f"{vals[4]:.2f}".rjust(8))

    mr, _ = msem([np.mean([r["summary"][p]["monitor_collect_ratio"] for p in pols])
                  for _, r in runs])
    print(f"\n[anti-subsidy] mean monitor:collect step ratio = {mr:.2f} "
          f"(E3 1.84x, S2 teleport 22x and FREE)")


if __name__ == "__main__":
    main()
