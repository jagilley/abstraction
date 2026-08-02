"""Post-hoc analysis of the trajectory re-cut. CPU-only, no Modal.

Fetch the input first:
    modal volume get rhm-scaling-data /rhm_decomposition_trajectory/summary.json .
    python3 -m rhm.residual_decomposition.trajectory.analyze_trajectory --summary summary.json

Three questions, in order of how much rests on them:

1. THE ARC. Does the complexodynamics rise-then-fall (under structured pressure) /
   rise-then-arrest (control) survive when the frontier is measured with the trusted
   instrument instead of activation effective rank?

2. AGREEMENT. The retired and trusted metrics are computed on the IDENTICAL (A, P)
   here, so where their trajectory shapes differ, the difference is the instrument.
   `--compare` prints them side by side with each metric's init/peak/final shape.

3. TRUSTWORTHINESS, POINTWISE. beta is only a property of the model's computation
   where it is capacity-invariant (audit tolerance +/-0.01) and the FM has not
   saturated. Both diagnostics are printed per checkpoint; rows failing either are
   flagged, and the arc summary is computed BOTH over all points and over trusted
   points only, because a rise-and-fall that exists only in the saturated regime is
   an instrument artifact, not a finding.

Published reference (RHM_COMPLEXODYNAMICS_README.md, measured with the retired
instrument) is printed alongside for the arms it covers.
"""

import argparse
import json

import numpy as np

# What the complexodynamics README reports, for the arms this re-cut re-runs.
# FM-free post_block6 activation effective-rank %, from the rhm_fm_regularizer parts.
PUBLISHED = {
    "fmreg:1.0": {"metric": "act eff-rank %", "init": 36.0,
                  "peak": (62.4, 4000), "final": (45.7, 300000),
                  "shape": "full rise-then-fall"},
    "fmreg:3.0": {"metric": "act eff-rank %", "init": 36.0,
                  "peak": (63.5, 4000), "final": (42.9, 300000),
                  "shape": "full rise-then-fall"},
    "wd:0.1": {"metric": "act eff-rank %", "init": 36.0,
               "peak": (59.0, 10000), "final": (54.3, 125000),
               "shape": "rise-then-ARREST (lam=0.03 anchor)"},
}

# Columns: (label, extractor, higher-is-more-frontier)
TRUSTED = [
    ("beta", lambda r: r["beta"]),
    ("R_res_part", lambda r: r["decomposition"]["repaired"]["R_res_participation"]),
    ("frontMass", lambda r: r["decomposition"]["repaired"]["frontier_mass"]),
]
RETIRED = [
    ("actRank%", lambda r: r["decomposition"]["naive"]["R_act"]
     / r["decomposition"]["naive"]["d_model"] * 100),
    ("naiveRres", lambda r: r["decomposition"]["naive"]["R_res"]),
    ("top1%", lambda r: r["res_top1_pc_last"] * 100),
]


# Only the fixed-depth legs may be pooled into the invariance spread. capQ is nl=1
# and is a DEPTH leg, not a capacity leg -- pooling it conflates "how much FM" with
# "what shape of FM" and makes the tolerance check unreadable.
INVARIANCE_CAPS = ["capB", "capM", "capQ2"]

# The repo's established gate: eta^2 and the residual's structure are only read when
# the FM's cosine is in band. Above the upper edge the FM is too good for the gap it
# was given and the residual is a noise floor (residual_decomposition README sec. 3).
COS_BAND = (0.90, 0.99)


def _trusted_flags(by_cap, beta_tol=0.01, sat_rel_res=0.02, cap_ref="capB"):
    """(capacity_invariant, unsaturated, beta_spread, rel_res, cos) per checkpoint.

    Saturation is judged on the reference capacity's own fit, by BOTH the relative
    residual and the cosine band -- at step 0 a fresh FM reads relRes 0.089 (looks
    fine) at cosine 0.996 (out of band). The two disagree, and the cosine band is
    the criterion the rest of the repo gates on.
    """
    inv_caps = [c for c in INVARIANCE_CAPS if c in by_cap]
    betas = [by_cap[c]["beta"] for c in inv_caps]
    spread = (max(betas) - min(betas)) if len(betas) > 1 else float("nan")
    inv = (spread <= beta_tol) if len(betas) > 1 else True

    ref = by_cap.get(cap_ref) or next(iter(by_cap.values()))
    rel = ref["decomposition"]["basic"]["relative_residual"]
    cos = ref["decomposition"]["basic"]["mean_cosine"]
    unsat = (rel >= sat_rel_res) and (COS_BAND[0] <= cos <= COS_BAND[1])
    return inv, unsat, spread, rel, cos


def _arc(steps, vals):
    """(init, peak, peak_step, final, shape) for one metric trajectory."""
    if not vals:
        return None
    i, f = vals[0], vals[-1]
    pk = int(np.argmax(vals))
    peak, pstep = vals[pk], steps[pk]
    rng = max(vals) - min(vals)
    if rng <= 0:
        return i, peak, pstep, f, "flat"
    rose = peak - i
    fell = peak - f
    if pk in (0, len(vals) - 1):
        shape = "monotone-up" if f >= i else "monotone-down"
    elif fell > 0.25 * rose:
        shape = "rise-then-FALL"
    elif fell > 0.05 * rose:
        shape = "rise-then-arrest"
    else:
        shape = "rise-then-plateau"
    return i, peak, pstep, f, shape


def analyze(path, cap="capB", beta_tol=0.01, sat_rel_res=0.02, compare=False):
    d = json.load(open(path))
    print("=" * 118)
    print(f"TRAJECTORY RE-CUT  |  key={d['key']}  gap={d['gap']}  caps={d['caps']}")
    print("  trusted: beta (shape), R_res_participation (frontier dimensionality), "
          "frontier_mass (size)")
    print("  retired: activation eff-rank %, naive R_res, top1-PC  "
          "-- same (A, P), so differences are the instrument")
    print("=" * 118)

    for arm, rows in d["results"].items():
        rows = [r for r in rows if cap in r["by_cap"]]
        if not rows:
            print(f"\n=== {arm}: no rows at capacity {cap} ===")
            continue
        steps = [r["step"] for r in rows]

        print(f"\n=== {arm}  (capacity {cap}) ===")
        print(f"{'step':>7} {'relRes':>7} {'cos':>6} | {'beta':>6} {'R2':>5} "
              f"{'R_res_part':>10} {'frontMass':>9} | {'actRank%':>8} {'naiveRres':>9} "
              f"{'top1%':>6} | {'d4eta2':>7} {'d6eta2':>7} | trust")
        trusted_steps = []
        for r0 in rows:
            r = r0["by_cap"][cap]
            dec, e = r["decomposition"], r["eta2_residual"]
            b, n, rp = dec["basic"], dec["naive"], dec["repaired"]
            inv, unsat, spread, _, _ = _trusted_flags(
                r0["by_cap"], beta_tol, sat_rel_res, cap_ref=cap)
            ok = inv and unsat
            if ok:
                trusted_steps.append(r0["step"])
            mark = "ok" if ok else ("SAT" if not unsat else "cap!")
            print(f"{r0['step']:>7} {b['relative_residual']:>7.4f} {b['mean_cosine']:>6.3f} | "
                  f"{r['beta']:>6.3f} {r['beta_r2']:>5.3f} "
                  f"{rp['R_res_participation']:>10.1f} {rp['frontier_mass']:>9.4f} | "
                  f"{n['R_act'] / n['d_model'] * 100:>8.1f} {n['R_res']:>9.1f} "
                  f"{r['res_top1_pc_last'] * 100:>6.1f} | "
                  f"{e['level_2']['feature_eta2_last']:>7.3f} "
                  f"{e['level_0']['feature_eta2_last']:>7.3f} | {mark}")

        # --- arc summary, all points vs trusted-only ---
        print(f"\n  ARC  (init -> peak@step -> final)   [{len(trusted_steps)}/{len(rows)} "
              f"checkpoints pass both trust gates]")
        print(f"    {'metric':>12} {'':>8} {'init':>9} {'peak':>9} {'@step':>8} "
              f"{'final':>9}  shape")
        for label, series in (("ALL", steps), ("TRUSTED", trusted_steps)):
            sel = [r for r in rows if r["step"] in series]
            if len(sel) < 3:
                print(f"    {label:>12}: fewer than 3 points, arc not computed")
                continue
            ss = [r["step"] for r in sel]
            for name, fn in (TRUSTED + RETIRED if compare else TRUSTED):
                vals = [fn(r["by_cap"][cap]) for r in sel]
                a = _arc(ss, vals)
                tag = "" if name in [t[0] for t in TRUSTED] else " (retired)"
                print(f"    {name + tag:>12} {label:>8} {a[0]:>9.3f} {a[1]:>9.3f} "
                      f"{a[2]:>8} {a[3]:>9.3f}  {a[4]}")

        if arm in PUBLISHED:
            p = PUBLISHED[arm]
            print(f"\n    published ({p['metric']}, retired instrument): "
                  f"init {p['init']} -> peak {p['peak'][0]} @{p['peak'][1]} -> "
                  f"final {p['final'][0]} @{p['final'][1]}  [{p['shape']}]")

        # --- pointwise trustworthiness ---
        inv_caps = [c for c in INVARIANCE_CAPS if c in d["caps"]]
        if len(inv_caps) > 1:
            print(f"\n  beta capacity-invariance over {'/'.join(inv_caps)} "
                  f"(fixed depth, tolerance +/-{beta_tol}); saturation = "
                  f"relRes < {sat_rel_res} or cosine outside {COS_BAND}:")
            for r0 in rows:
                inv, unsat, spread, rel, cos = _trusted_flags(
                    r0["by_cap"], beta_tol, sat_rel_res, cap_ref=cap)
                bs = "/".join(f"{r0['by_cap'][c]['beta']:.3f}"
                              for c in inv_caps if c in r0["by_cap"])
                flags = []
                if not inv:
                    flags.append("BETA NOT CAPACITY-INVARIANT")
                if not unsat:
                    flags.append("FM OUT OF BAND")
                print(f"    step {r0['step']:>7}: beta={bs}  spread={spread:.3f}  "
                      f"relRes={rel:.4f} cos={cos:.3f}  "
                      f"{'| ' + '; '.join(flags) if flags else ''}")

        # --- the depth leg, reported separately and never pooled above ---
        if "capM" in d["caps"] and "capQ" in d["caps"]:
            print("\n  DEPTH leg (capM nl=2 vs capQ nl=1, matched dh=8/mm=1.0): "
                  "is beta set by FM shape rather than FM size?")
            for r0 in rows:
                if "capM" in r0["by_cap"] and "capQ" in r0["by_cap"]:
                    bm = r0["by_cap"]["capM"]["beta"]
                    bq = r0["by_cap"]["capQ"]["beta"]
                    print(f"    step {r0['step']:>7}: beta nl2={bm:.3f} nl1={bq:.3f}  "
                          f"delta={bm - bq:+.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="summary.json")
    ap.add_argument("--cap", default="capB", help="capacity to report the arc at")
    ap.add_argument("--beta-tol", type=float, default=0.01)
    ap.add_argument("--sat-rel-res", type=float, default=0.02,
                    help="relative residual below which the FM counts as saturated")
    ap.add_argument("--compare", action="store_true",
                    help="also show the retired metrics' arcs side by side")
    a = ap.parse_args()
    analyze(a.summary, cap=a.cap, beta_tol=a.beta_tol,
            sat_rel_res=a.sat_rel_res, compare=a.compare)


if __name__ == "__main__":
    main()
