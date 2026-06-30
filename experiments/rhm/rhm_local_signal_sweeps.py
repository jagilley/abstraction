"""Cross-seed / DGP sweeps that produced the tables in RHM_DEEP_COMPOSITION_README.md.

The single-instance primitives live in rhm_local_signal.py (modes signal / compound
/ bp). This script aggregates them across rule seeds and DGP settings so every
headline number in the README reproduces from one command.

    python3 rhm_local_signal_sweeps.py --which all      # all three tables (~90s, CPU)
    python3 rhm_local_signal_sweeps.py --which signal    # Thread A: local vs supervised
    python3 rhm_local_signal_sweeps.py --which bp         # oracle / greedy / optimal BP
    python3 rhm_local_signal_sweeps.py --which dgp        # BP ceiling vs occupancy m/v
"""

import argparse
import contextlib
import io

import numpy as np

from rhm_data import generate_rules, generate_rules_distinct
from rhm_local_signal import run, run_compounding, run_bp_ceiling

SEEDS = [0, 1, 2, 3, 4]


def _quiet(fn, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(**kw)


def signal_ceiling():
    """Thread A: local (sibling-context) vs supervised (root) synonym recovery,
    at the largest occurrence budget, across rule seeds. chance = 1/v."""
    agg = {}
    for sd in SEEDS:
        res = _quiet(run, n_sequences=60000, o_grid=(2000, 10000, 40000),
                     n_repeat=2, rule_seed=sd)
        for d, rec in res["depths"].items():
            a = agg.setdefault(d, {"local": [], "supervised": []})
            a["local"].append(rec["local"][-1])
            a["supervised"].append(rec["supervised"][-1])
    print("\n[Thread A] Synonym recovery ceiling (O=40000 occ), 5 seeds. chance=0.125")
    print(f"{'depth':>6} | {'local (sibling)':>18} | {'supervised (root)':>19}")
    for d in sorted(agg):
        lo, su = np.array(agg[d]["local"]), np.array(agg[d]["supervised"])
        print(f"{d:>6} | {lo.mean():>8.3f} ± {lo.std():>5.3f}   | "
              f"{su.mean():>8.3f} ± {su.std():>5.3f}")


def bp_combined():
    """A.5 + BP: per-level recovery of the latent feature from the leaves --
    oracle 1-step (clean input), greedy cascade (predicted lifts), optimal BP."""
    L = 6
    oc, cc, bp = {}, {}, {}
    for sd in SEEDS:
        rc = _quiet(run_compounding, n_sequences=60000, rule_seed=sd, verbose=False)
        rb = _quiet(run_bp_ceiling, n_sequences=10000, rule_seed=sd, verbose=False)
        for i, d in enumerate(rc["depth"]):
            oc.setdefault(d, []).append(rc["oracle"][i])
            cc.setdefault(d, []).append(rc["compounded"][i])
        for ell, (ma, _pt) in rb.items():
            bp.setdefault(L - ell, []).append(ma)
    print("\n[A.5 + BP] Per-level latent recovery from leaves, 5 seeds. chance=0.125")
    print(f"{'depth d':>7} | {'oracle 1-step':>13} | {'greedy cascade':>14} | "
          f"{'optimal BP':>11}")
    for d in range(1, L):
        print(f"{d:>7} | {np.mean(oc[d]):>13.3f} | {np.mean(cc[d]):>14.3f} | "
              f"{np.mean(bp[d]):>11.3f}")
    print(f"{L:>7} | {'(root)':>13} | {'-':>14} | {np.mean(bp[L]):>11.3f}")


def dgp():
    """BP ceiling vs tuple-space occupancy m/v^(s-1), across rule-sampling and v/m."""
    configs = [
        ("replace  v=8  m=4", generate_rules,          8, 4),
        ("distinct v=8  m=4", generate_rules_distinct, 8, 4),
        ("distinct v=8  m=2", generate_rules_distinct, 8, 2),
        ("distinct v=16 m=4", generate_rules_distinct, 16, 4),
        ("distinct v=32 m=4", generate_rules_distinct, 32, 4),
        ("distinct v=32 m=8", generate_rules_distinct, 32, 8),
    ]
    s, L = 2, 6
    print("\n[DGP] Optimal BP MAP recovery by depth (1=near leaves ... 6=root), 3 seeds")
    print(f"{'config':>20} | {'m/v':>5} | {'chance':>6} |  "
          + "   ".join(f"d{d}" for d in range(1, 7)))
    for name, fn, v, m in configs:
        per_seed = []
        for sd in [0, 1, 2]:
            rules = fn(v, s, L, m, seed=sd)
            rec = _quiet(run_bp_ceiling, n_sequences=8000, seq_seed=1, rules=rules)
            per_seed.append({L - ell: ma for ell, (ma, _pt) in rec.items()})
        prof = {d: np.mean([p[d] for p in per_seed]) for d in range(1, 7)}
        row = "  ".join(f"{prof[d]:.2f}" for d in range(1, 7))
        print(f"{name:>20} | {m / v ** (s - 1):>5.3f} | {1.0 / v:>6.3f} |  {row}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["all", "signal", "bp", "dgp"], default="all")
    args = ap.parse_args()
    if args.which in ("all", "signal"):
        signal_ceiling()
    if args.which in ("all", "bp"):
        bp_combined()
    if args.which in ("all", "dgp"):
        dgp()
