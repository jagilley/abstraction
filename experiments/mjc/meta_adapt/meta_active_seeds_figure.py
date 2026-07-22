"""Aggregate the multi-seed value-directed identification runs (patch, 4 arms) into a
robust figure: per-arm in-patch R^2 and phi-decode error (mean +/- sem over seeds) + the
patch-visitation ladder (the mechanism). The single-seed ranking of the active arms was
seed-noise; this resolves what is robust. Run locally:
    python3 mjc/meta_adapt/meta_active_seeds_figure.py
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
FIGDIR = os.path.join(HERE, "figures")
TAGS = ["act_patch4", "act_patch4_s1", "act_patch4_s2", "act_patch4_s3"]
ARMS = ["passive", "magnitude", "directed", "navigate"]
COL = {"passive": "#888888", "magnitude": "#d1603d", "directed": "#3d6fd1", "navigate": "#2f9e44"}
MRK = {"passive": "s--", "magnitude": "^--", "directed": "o--", "navigate": "D-"}
LBL = {"passive": "passive (OU)", "magnitude": "magnitude (max|u|)",
       "directed": "myopic VoI", "navigate": "navigating VoI (info-MPC)"}


def load(tag):
    with open(os.path.join(FIGDIR, f"meta_active_{tag}", "results.json")) as fh:
        return json.load(fh)


def main():
    runs = []
    for t in TAGS:
        try:
            runs.append(load(t))
        except FileNotFoundError:
            print(f"[skip] {t} not found")
    if not runs:
        print("no runs"); return
    budgets = runs[0]["budgets"]; xN = [max(N, 1) for N in budgets]
    nseed = len(runs)
    print(f"aggregating {nseed} seeds\n")

    # stack per-seed curves: (nseed, nbudget)
    r2 = {a: np.array([r["r2_median"]["in"][a] for r in runs], float) for a in ARMS}
    err = {a: np.array([r["phi_err"][a] for r in runs], float) for a in ARMS}
    vis = {a: np.array([r["patch_visitation"][a] for r in runs], float) for a in ARMS}

    def ms(x):  # mean, sem over seeds (axis 0)
        return np.nanmean(x, 0), np.nanstd(x, 0) / max(1, np.sqrt(x.shape[0]))

    print(" arm         in-R^2@maxN     phi_err@maxN     visitation")
    for a in ARMS:
        print(f" {a:11s} {np.nanmean(r2[a],0)[-1]:.3f}          "
              f"{np.nanmean(err[a],0)[-1]:.3f}           {np.nanmean(vis[a]):.3f}")

    plt.rcParams.update({"figure.dpi": 130, "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(15.5, 4.7),
                                        gridspec_kw={"width_ratios": [3, 3, 2]})

    # Panel A: in-patch R^2 (mean +/- sem)
    for a in ARMS:
        m, s = ms(r2[a])
        axA.plot(xN, m, MRK[a], color=COL[a], lw=2.1, label=LBL[a])
        axA.fill_between(xN, m - s, m + s, color=COL[a], alpha=0.13)
    axA.set_xscale("log"); axA.set_xlabel("reward-free transitions N")
    axA.set_ylabel("IN-PATCH velocity-dim Δs R^2")
    axA.set_title(f"Identification quality ({nseed} seeds, mean±sem)", fontsize=10)
    axA.legend(fontsize=8, loc="lower right")

    # Panel B: phi decode error
    for a in ARMS:
        m, s = ms(err[a])
        keep = ~np.isnan(m)
        axB.plot(np.asarray(xN)[keep], m[keep], MRK[a], color=COL[a], lw=2.1, label=LBL[a])
        axB.fill_between(np.asarray(xN)[keep], (m - s)[keep], (m + s)[keep], color=COL[a], alpha=0.13)
    axB.set_xscale("log"); axB.set_yscale("log"); axB.set_xlabel("reward-free transitions N")
    axB.set_ylabel("|φ decoded − φ true|")
    axB.set_title("Task-parameter decode error", fontsize=10)
    axB.legend(fontsize=8, loc="upper right")

    # Panel C: patch visitation (the mechanism) — bar with per-seed scatter
    xs = np.arange(len(ARMS))
    axC.bar(xs, [np.nanmean(vis[a]) for a in ARMS], color=[COL[a] for a in ARMS], alpha=0.85)
    for i, a in enumerate(ARMS):
        axC.scatter(np.full(len(vis[a]), i), vis[a], s=14, color="k", zorder=3, alpha=0.6)
        axC.text(i, np.nanmean(vis[a]), f"{np.nanmean(vis[a]):.2f}", ha="center", va="bottom", fontsize=9)
    axC.set_xticks(xs); axC.set_xticklabels([a for a in ARMS], rotation=20, fontsize=8)
    axC.set_ylabel("fraction of transitions IN patch")
    axC.set_title("Patch visitation (the mechanism)", fontsize=10)

    fig.suptitle("Value-directed identification under scarcity: navigation works (visitation), "
                 "but does it beat the magnitude heuristic on ID?", fontsize=11)
    fig.tight_layout()
    outdir = os.path.join(FIGDIR, "meta_active_seeds")
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, "fig_seeds.png")
    fig.savefig(out, bbox_inches="tight")
    print(f"\n[save] wrote {out}")


if __name__ == "__main__":
    main()
