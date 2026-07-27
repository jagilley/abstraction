"""Post-hoc analysis of the audit and ratchet summaries. CPU-only, no Modal.

Fetch the inputs first:
    modal volume get rhm-scaling-data /residual_decomposition_audit/summary.json .
    modal volume get rhm-scaling-data /residual_decomposition_ratchet/summary.json .

    python3 -m rhm.residual_decomposition.analyze_summaries \
        --audit audit_summary.json --ratchet ratchet_summary.json

The load-bearing analysis here is `shadow_law`. Parts A+B showed the belief's
two-subspace partition does not hold, but "no partition" is not itself an
account of what the residual IS. This fits

    residual_variance(i)  ~  activation_variance(i) ** beta

across A's principal directions. beta = 0 is an isotropic residual (no
relationship to A's structure at all); beta = 1 is a residual exactly
proportional to A (a scaled-down shadow of the whole computation); 0 < beta < 1
is graded absorption -- the FM absorbs proportionally more where A carries more
variance, but the falloff is continuous, with no boundary anywhere that would
license splitting the directions into "compressed" and "frontier".
"""

import argparse
import json

import numpy as np


def _part(repaired):
    """Frontier dimensionality, tolerating summaries written before the rename."""
    return repaired.get("R_res_participation", repaired.get("R_res_pr"))


def shadow_law(geometry):
    """Fit log(res_var) = beta * log(act_var) + c over A's principal directions."""
    a = np.asarray(geometry["act_variance_spectrum"], dtype=float)
    v = np.asarray(geometry["res_variance_in_act_basis"], dtype=float)
    msk = (a > 0) & (v > 0)
    if msk.sum() < 8:
        return float("nan"), float("nan")
    beta, c = np.polyfit(np.log(a[msk]), np.log(v[msk]), 1)
    pred = beta * np.log(a[msk]) + c
    resid = np.log(v[msk]) - pred
    r2 = 1.0 - (resid ** 2).sum() / (
        (np.log(v[msk]) - np.log(v[msk]).mean()) ** 2).sum()
    return float(beta), float(r2)


def analyze_audit(path):
    d = json.load(open(path))
    print("=" * 104)
    print("SHADOW LAW: res_var(i) ~ act_var(i)^beta across A's principal directions")
    print("  beta=0 -> isotropic residual (unrelated to A) | beta=1 -> exact scaled")
    print("  shadow of A | 0<beta<1 -> graded absorption, no partition boundary")
    print("=" * 104)
    print(f"{'setting':>12} {'gap':>7} {'cap':>5} | {'beta':>6} {'R2':>6} | "
          f"{'rho_top16':>9} {'rho_mid':>8} {'rho_tail16':>10} | {'align':>7} "
          f"{'R_res_part':>10} {'naive_R_res':>11}")
    print("-" * 104)
    for r in sorted(d["results"], key=lambda r: (r["setting"], r["gap"]["tag"],
                                                 r["fm"]["n_params"])):
        dec = r["decomposition"]
        g, rp, nv = dec["geometry"], dec["repaired"], dec["naive"]
        beta, r2 = shadow_law(g)
        rho = np.clip(np.asarray(g["absorption_spectrum"], dtype=float), 0, 1)
        gs = r["gap"]["tag"].replace("post_block", "b").replace("_to_", ">")
        print(f"{r['setting']:>12} {gs:>7} {r['capacity_label']:>5} | "
              f"{beta:>6.2f} {r2:>6.3f} | {rho[:16].mean():>9.3f} "
              f"{rho[48:80].mean():>8.3f} {rho[-16:].mean():>10.3f} | "
              f"{g['alignment_index']:>+7.3f} {_part(rp):>9.1f} "
              f"{nv['R_res']:>11.1f}")

    print("\n  Grouped by (setting, gap) -- is beta invariant to FM capacity?")
    groups = {}
    for r in d["results"]:
        groups.setdefault((r["setting"], r["gap"]["tag"]), []).append(
            shadow_law(r["decomposition"]["geometry"])[0])
    for (st, gap), betas in sorted(groups.items()):
        gs = gap.replace("post_block", "b").replace("_to_", ">")
        print(f"    {st:>12} {gs:>7}: beta = {np.mean(betas):.3f} "
              f"+/- {np.std(betas):.3f}  (over 4x FM capacity range)")


def beta_matched_window(geometry, decades=2.0, min_dirs=8):
    """Refit beta using only directions within `decades` of A's top variance.

    Guards against the obvious artifact in the cross-domain beta comparison.
    beta is a log-log slope, and the domains differ enormously in how many
    decades their activation spectrum spans (language ~2.0, RHM at d=512 ~5.8),
    so a difference in beta could in principle reflect the fitting range rather
    than the absorption behaviour. Refitting everything over a common window
    tests that.

    Returns (beta, r2, n_dirs). Read n_dirs before trusting the number: for a
    concentrated spectrum the top 2 decades may contain only a handful of
    directions, and the estimate is then too noisy to compare.
    """
    a = np.asarray(geometry["act_variance_spectrum"], dtype=float)
    v = np.asarray(geometry["res_variance_in_act_basis"], dtype=float)
    msk = (a > 0) & (v > 0) & (a > a.max() / 10.0 ** decades)
    if msk.sum() < min_dirs:
        return float("nan"), float("nan"), int(msk.sum())
    beta, c = np.polyfit(np.log(a[msk]), np.log(v[msk]), 1)
    pred = beta * np.log(a[msk]) + c
    r2 = 1.0 - ((np.log(v[msk]) - pred) ** 2).sum() / (
        (np.log(v[msk]) - np.log(v[msk]).mean()) ** 2).sum()
    return float(beta), float(r2), int(msk.sum())


def analyze_width_sweep(path):
    """Width sweep: beta vs model width, in both domains, plus the controls."""
    rows = json.load(open(path))
    print("=" * 100)
    print("WIDTH SWEEP: does beta track model width?")
    print("=" * 100)
    print(f"{'domain':>8} {'d':>5} {'beta':>7} {'beta_2dec':>10} {'n_dirs':>7} | "
          f"{'R_act/d':>8} {'act_decades':>12} | {'frontier':>9} "
          f"{'R_res/R_act':>12}")
    print("-" * 100)
    pts = []
    for r in sorted(rows, key=lambda r: (r["domain"], r["n_embd"])):
        d = r["decomposition"]
        g, n, rp = d["geometry"], d["naive"], d["repaired"]
        a = np.asarray(g["act_variance_spectrum"], dtype=float)
        a = a[a > 0]
        b2, _, nd = beta_matched_window(g)
        racd = n["R_act"] / r["n_embd"]
        pts.append((racd, r["beta"]))
        print(f"{r['domain']:>8} {r['n_embd']:>5} {r['beta']:>7.3f} "
              f"{b2:>10.3f} {nd:>7} | {racd:>7.1%} "
              f"{np.log10(a.max() / a.min()):>12.2f} | "
              f"{rp['frontier_mass']:>9.4f} {n['R_res'] / n['R_act']:>12.3f}")
    p = np.array(pts)
    print(f"\n  corr(R_act/d, beta) = {np.corrcoef(p[:, 0], p[:, 1])[0, 1]:+.3f}")
    print("  Frontier mass should be ~flat if the level is set by the FM's")
    print("  capacity RATIO to a block (held at 100% here) rather than by its")
    print("  absolute size.")


def analyze_ratchet(path):
    d = json.load(open(path))
    print("\n" + "=" * 104)
    print("RATCHET: forgetting check -- does the NOVEL arm retain cycle-0 rules?")
    print("=" * 104)
    print(f"{'cond':>10} {'cyc':>4} {'val_on_cycle0_rules':>21} "
          f"{'val_on_current_rules':>21}")
    for c, r in d.items():
        for h in r["history"]:
            cur = h.get("val_loss_current")
            print(f"{c:>10} {h['cycle']:>4} {h['val_loss_probe']:>21.4f} "
                  f"{(f'{cur:.4f}' if cur is not None else '-'):>21}")
        print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", default="audit_summary.json")
    ap.add_argument("--ratchet", default="ratchet_summary.json")
    ap.add_argument("--width", default=None,
                    help="width_sweep summary.json (optional)")
    a = ap.parse_args()
    if a.width:
        analyze_width_sweep(a.width)
        return
    analyze_audit(a.audit)
    analyze_ratchet(a.ratchet)


if __name__ == "__main__":
    main()
