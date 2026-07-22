"""Stitch the meta_adapt floor + actuator-conflict runs into the ORDER-PARAMETER
figure: the meta-vs-multitask gap (and the conflict strength) as a function of the
actuator-rotation half-range Phi. Reads the local results.json mirrors and writes a
combined figure. Run locally (matplotlib): python3 mjc/meta_adapt/meta_adapt_sweep_figure.py
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
FIGDIR = os.path.join(HERE, "figures")

# (tag, Phi, family) — the floor is Phi=0 (damping), then the actuator conflict sweep.
RUNS = [
    ("full_v2", 0.0, "damping"),
    ("actuator_p16", 0.5236, "actuator"),
    ("actuator_p2", 1.5708, "actuator"),
    ("actuator_pi", 3.14159, "actuator"),
]


def load(tag):
    p = os.path.join(FIGDIR, f"meta_adapt_{tag}", "results.json")
    with open(p) as fh:
        return json.load(fh)


def main():
    rows = []
    for tag, phi, fam in RUNS:
        try:
            R = load(tag)
        except FileNotFoundError:
            print(f"[skip] {tag}: results.json not found yet")
            continue
        budgets = R["budgets"]
        gap = np.asarray(R["meta_minus_multitask_gap"]["mean"])   # median per-task gap vs budget
        meta = np.asarray(R["arms"]["meta"]["r2_median"])
        multi = np.asarray(R["arms"]["multitask"]["r2_median"])
        rows.append(dict(tag=tag, phi=phi, fam=fam, budgets=budgets, gap=gap,
                         meta=meta, multi=multi,
                         zshot_multi=float(multi[0]), zshot_meta=float(meta[0]),
                         gap_maxN=float(gap[-1]), gap_peak=float(np.nanmax(gap)),
                         gap_meanpos=float(np.nanmean(gap[1:])),
                         oracle=float(R["oracle_mean"])))
    if not rows:
        print("no runs available yet")
        return
    rows.sort(key=lambda r: r["phi"])

    print("\n Phi     zshot(multi)  gap@maxN   gap_peak   gap_mean(N>0)  oracle")
    for r in rows:
        print(f" {r['phi']:.3f}    {r['zshot_multi']:+.3f}      {r['gap_maxN']:+.3f}    "
              f"{r['gap_peak']:+.3f}     {r['gap_meanpos']:+.3f}       {r['oracle']:.3f}")

    phis = [r["phi"] for r in rows]

    plt.rcParams.update({"figure.dpi": 130, "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.5, 4.8))

    # Panel A: conflict strength = how badly a POOLED init transfers zero-shot
    axA.plot(phis, [r["zshot_multi"] for r in rows], "o-", color="#d1603d", lw=2.2,
             label="multitask (pooled) init")
    axA.plot(phis, [r["zshot_meta"] for r in rows], "s--", color="#3d6fd1", lw=1.8, alpha=0.8,
             label="meta init")
    axA.axhline(0, color="#999", lw=0.8)
    axA.set_xlabel("actuator-rotation half-range  Φ  (rad)   [Φ=0 = damping floor]")
    axA.set_ylabel("zero-shot held-out velocity R^2  (N=0)")
    axA.set_title("Conflict strength: pooled zero-shot transfer\ncollapses as Φ grows", fontsize=10)
    axA.legend(fontsize=8.5, loc="lower left")

    # Panel B: the ORDER PARAMETER opens with conflict
    axB.axhline(0, color="#444", lw=1.1)
    axB.plot(phis, [r["gap_maxN"] for r in rows], "o-", color="#2f6f2f", lw=2.3,
             label="gap at max N (320)")
    axB.plot(phis, [r["gap_peak"] for r in rows], "^--", color="#7a3fb0", lw=1.8, alpha=0.85,
             label="peak gap over N")
    axB.set_xlabel("actuator-rotation half-range  Φ  (rad)   [Φ=0 = damping floor]")
    axB.set_ylabel("velocity-R^2(meta) − R^2(multitask)")
    axB.set_title("The order parameter opens with conflict\n(floor Φ=0 ≈ 0 → gap grows with Φ)",
                  fontsize=10)
    axB.legend(fontsize=8.5, loc="upper left")

    fig.suptitle("Learn-to-adapt vs pooling: collapse at the smooth floor, separation under "
                 "input-coupled conflict", fontsize=11)
    fig.tight_layout()
    outdir = os.path.join(FIGDIR, "meta_adapt_sweep")
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, "fig_gap_vs_conflict.png")
    fig.savefig(out, bbox_inches="tight")
    print(f"\n[save] wrote {out}")


if __name__ == "__main__":
    main()
