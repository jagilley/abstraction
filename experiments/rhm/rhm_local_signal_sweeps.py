"""Cross-seed / DGP sweeps that produced the tables in RHM_DEEP_COMPOSITION_README.md.

The single-instance primitives live in rhm_local_signal.py (modes signal / compound
/ bp). This script aggregates them across rule seeds and DGP settings so every
headline number in the README reproduces from one command.

    python3 rhm_local_signal_sweeps.py --which all      # all three tables (~90s, CPU)
    python3 rhm_local_signal_sweeps.py --which signal    # Thread A: local vs supervised
    python3 rhm_local_signal_sweeps.py --which bp         # oracle / greedy / optimal BP
    python3 rhm_local_signal_sweeps.py --which dgp        # BP ceiling vs occupancy m/v
    python3 rhm_local_signal_sweeps.py --which occ_frontier  # reference lines for the
                                                             # occupancy-frontier sweep
"""

import argparse
import contextlib
import io
import json
import os

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


def occ_frontier_refs():
    """Reference lines (BP ceiling + greedy floor) per bottom-up depth d for the
    six occupancy-frontier settings (rhm_occupancy_frontier.py), distinct rules,
    3 seeds. These bracket what a trained NTP model could do: greedy = structure-
    from-data with no real inference (local hard cluster-and-lift cascade); BP =
    optimal joint inference with the rules known (the information ceiling). The
    GPU sweep's learned frontier is read against these. Saves a JSON the analysis
    merges with the model frontiers."""
    configs = [(8, 4), (16, 8), (16, 4), (32, 4), (16, 2), (64, 4)]
    s, L, seeds = 2, 6, [0, 1, 2]
    out = {}
    print("\n[OCC-FRONTIER REFS] distinct rules, 3 seeds. "
          "BP ceiling (top) / greedy floor (bottom) by depth (d1=near leaves..d6=root)")
    print(f"{'v,m':>7} | {'m/v':>6} | {'chance':>6} | {'ref':>6} |  "
          + "   ".join(f"d{d}" for d in range(1, 7)))
    for v, m in configs:
        bp_seeds, gr_seeds = [], []
        for sd in seeds:
            rules = generate_rules_distinct(v, s, L, m, seed=sd)
            rb = _quiet(run_bp_ceiling, n_sequences=8000, seq_seed=1, rules=rules)
            rc = _quiet(run_compounding, n_sequences=60000, seq_seed=1, rules=rules,
                        verbose=False)
            bp_seeds.append({L - ell: ma for ell, (ma, _pt) in rb.items()})
            gr_seeds.append({d: rc["compounded"][i] for i, d in enumerate(rc["depth"])})
        bp = {d: float(np.mean([p[d] for p in bp_seeds])) for d in range(1, 7)}
        gr = {d: float(np.mean([p[d] for p in gr_seeds])) for d in range(1, 6)}  # no root
        out[f"v{v}m{m}"] = {"v": v, "m": m, "occ": m / v ** (s - 1),
                            "chance": 1.0 / v, "bp_ceiling": bp, "greedy_floor": gr}
        tag = f"{v},{m}"
        print(f"{tag:>7} | {m / v ** (s - 1):>6.4f} | {1.0 / v:>6.4f} | {'BP':>6} |  "
              + "   ".join(f"{bp[d]:.2f}" for d in range(1, 7)))
        print(f"{'':>7} | {'':>6} | {'':>6} | {'greedy':>6} |  "
              + "   ".join(f"{gr[d]:.2f}" for d in range(1, 6)) + "      -")
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "occ_frontier_refs.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {path}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--which",
                    choices=["all", "signal", "bp", "dgp", "occ_frontier"],
                    default="all")
    args = ap.parse_args()
    if args.which in ("all", "signal"):
        signal_ceiling()
    if args.which in ("all", "bp"):
        bp_combined()
    if args.which in ("all", "dgp"):
        dgp()
    if args.which == "occ_frontier":
        occ_frontier_refs()
