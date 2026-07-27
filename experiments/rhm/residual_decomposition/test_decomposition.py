"""Validate decomposition.py against synthetic cases with known ground truth.

Run locally (no GPU, no Modal):  python3 -m rhm.residual_decomposition.test_decomposition

Three regimes, each with a residual whose geometry we control exactly:
  isotropic  -- residual spread over all D directions (the null: no partition)
  nested     -- residual confined to A's TOP-k principal directions
  frontier   -- residual confined to A's TAIL directions (the belief's picture:
                leading directions absorbed as routine, tail left as frontier)

What the tests establish before any GPU is spent:
  * whether the naive subadditivity ratio can be ~1 in any of these regimes
  * whether naive R_res responds to residual magnitude at all
  * that alignment_index separates nested/frontier from isotropic, and reads ~0
    on isotropic (i.e. the chance baseline is calibrated correctly)
  * that the repaired triple's partition is exact and magnitude-sensitive
"""

import numpy as np

from rhm.residual_decomposition.decomposition import full_decomposition

D, N = 128, 20000


def make_case(geom, rel, alpha=1.0, seed=0):
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((D, D)))
    scale = np.arange(1, D + 1.0) ** (-alpha)
    A = (rng.standard_normal((N, D)) * scale) @ Q.T

    Vt = np.linalg.svd(A - A.mean(0), full_matrices=False)[2]
    if geom == "isotropic":
        Rz = rng.standard_normal((N, D))
    elif geom == "nested":                       # inside A's top-16 PCs
        Rz = rng.standard_normal((N, 16)) @ Vt[:16]
    elif geom == "frontier":                     # inside A's bottom-16 PCs
        Rz = rng.standard_normal((N, 16)) @ Vt[-16:]
    else:
        raise ValueError(geom)

    Rz *= rel * np.linalg.norm(A) / np.linalg.norm(Rz)
    return A, A - Rz


def main():
    print(f"{'geom':>10} {'rel_res':>8} | {'R_act':>6} {'R_comp':>6} {'R_res':>6} "
          f"{'ratio':>6} | {'align':>7} {'step':>6} {'stepXs':>7} | "
          f"{'R_act_pr':>8} {'R_cmp_v':>7} {'R_res_v':>7} {'sum_ok':>6}")
    print("-" * 112)

    rows = []
    for geom in ("isotropic", "nested", "frontier"):
        for rel in (0.003, 0.03, 0.30):
            A, P = make_case(geom, rel)
            d = full_decomposition(A, P)
            n, g, r = d["naive"], d["geometry"], d["repaired"]
            sum_ok = abs(r["R_comp_v"] + r["R_res_v"] - r["R_act_pr"]) < 1e-6
            rows.append((geom, rel, n, g, r))
            print(f"{geom:>10} {rel:>8.3f} | {n['R_act']:>6.1f} {n['R_comp']:>6.1f} "
                  f"{n['R_res']:>6.1f} {n['subadditivity_ratio']:>6.2f} | "
                  f"{g['alignment_index']:>7.3f} {g['absorption_step']:>6.2f} "
                  f"{g['absorption_step_excess']:>7.3f} | "
                  f"{r['R_act_pr']:>8.1f} {r['R_comp_v']:>7.1f} {r['R_res_v']:>7.1f} "
                  f"{str(sum_ok):>6}")
        print()

    print("=== assertions ===")
    fails = []

    def check(name, cond):
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        if not cond:
            fails.append(name)

    # The repaired partition must be exact everywhere.
    check("repaired partition exact (R_comp_v + R_res_v == R_act_pr)",
          all(abs(r["R_comp_v"] + r["R_res_v"] - r["R_act_pr"]) < 1e-6
              for *_, r in rows))

    # Naive ratio is never ~1 -- the belief's subadditivity claim has no regime
    # in which it holds, not even the one its own picture describes.
    check("naive subadditivity ratio never ~1 (|ratio-1| > 0.1 in all regimes)",
          all(abs(n["subadditivity_ratio"] - 1.0) > 0.1 for *_, n, _, _ in rows))

    # Naive R_res is blind to magnitude; repaired R_res_v is not.
    iso = {rel: (n, r) for geom, rel, n, _, r in rows if geom == "isotropic"}
    check("naive R_res ~invariant to a 100x residual-norm sweep (<2% change)",
          abs(iso[0.30][0]["R_res"] - iso[0.003][0]["R_res"])
          / iso[0.003][0]["R_res"] < 0.02)
    check("repaired R_res_v tracks residual magnitude (>10x change)",
          iso[0.30][1]["R_res_v"] > 10 * iso[0.003][1]["R_res_v"])

    # Chance baseline calibrated: an isotropic residual must read ~0 alignment.
    check("alignment_index ~0 for isotropic residual (|idx| < 0.05)",
          all(abs(g["alignment_index"]) < 0.05
              for geom, _, _, g, _ in rows if geom == "isotropic"))
    check("alignment_index strongly positive when nested in A's top PCs (>0.5)",
          all(g["alignment_index"] > 0.5
              for geom, _, _, g, _ in rows if geom == "nested"))

    # A raw absorption step is NOT evidence of preferential absorption: A's own
    # spectral decay manufactures one for free. Only the spectrum-matched excess
    # discriminates, so that is what gets asserted.
    check("absorption_step_excess ~0 for isotropic residual (|excess| < 0.05)",
          all(abs(g["absorption_step_excess"]) < 0.05
              for geom, _, _, g, _ in rows if geom == "isotropic"))
    check("frontier residual is anti-aligned with A's leading PCs (align < -0.1)",
          all(g["alignment_index"] < -0.1
              for geom, _, _, g, _ in rows if geom == "frontier"))
    check("alignment_index is magnitude-invariant (pure geometry readout)",
          all(abs(g["alignment_index"]
                  - [gg["alignment_index"] for gm, rl, _, gg, _ in rows
                     if gm == geom and rl == 0.003][0]) < 1e-3
              for geom, _, _, g, _ in rows))

    print(f"\n{len(fails)} failure(s)" + (f": {fails}" if fails else ""))
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
