"""Aggregate the multi-seed capacity-competition planning runs (pusher-field pfield config)
into a robust figure. pfield_v1 (single seed) showed value-shaped <= veridical planning at
every capacity with a clean -27% win at h=64 (equal one-step fidelity), but the cross-cap
pattern was noisy. This resolves what is robust across seeds:
  Panel A  planning goal-dist vs capacity, shaped vs veridical (context), mean +/- sem
  Panel B  the planning GAP (veridical - shaped; >0 = value-shaping helps control) vs
           capacity, mean +/- sem + per-seed scatter  <- the behavioral order parameter
  Panel C  FM-side pusher-vel R^2 (value-relevant) vs capacity, shaped vs veridical -- the
           freed-capacity lever that (under a capacity-hungry pusher) drives Panel B.

Run locally (after the seed runs land):
    python3 mjc/meta_adapt/meta_value_shaping_seeds_figure.py
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
FIGDIR = os.path.join(HERE, "figures")
TAGS = ["plan_pfield_v1", "plan_pfield_s1", "plan_pfield_s2", "plan_pfield_s3"]  # seed 0..3
C_VER, C_SHP = "#3d6fd1", "#d1603d"


def load(tag):
    with open(os.path.join(FIGDIR, f"meta_value_shaping_{tag}", "results.json")) as fh:
        return json.load(fh)


def ms(x):  # mean, sem over seeds (axis 0), nan-safe
    n = np.sum(~np.isnan(x), 0)
    return np.nanmean(x, 0), np.nanstd(x, 0) / np.sqrt(np.maximum(1, n))


def main():
    runs = []
    for t in TAGS:
        try:
            runs.append(load(t))
        except FileNotFoundError:
            print(f"[skip] {t} not found")
    if not runs:
        print("no runs found"); return
    ns = len(runs)
    caps = runs[0]["caps"]
    pcaps = runs[0]["plan_caps"]
    rand = float(np.mean([r["plan_random_dist"] for r in runs]))
    print(f"aggregating {ns} seeds | caps={caps} plan_caps={pcaps} random_floor={rand:.3f}\n")

    def stack(obj, mode, metric, cap_list, sub=None):
        """(nseed, ncap) array of a metric across seeds."""
        out = np.full((len(runs), len(cap_list)), np.nan)
        for si, r in enumerate(runs):
            for ci, h in enumerate(cap_list):
                cell = r["sweep"][obj][str(h)]
                if sub is not None:
                    cell = cell.get(sub, {})
                    v = cell.get(metric, np.nan)
                else:
                    v = cell.get(metric, np.nan)
                out[si, ci] = v if isinstance(v, (int, float)) else np.nan
        return out

    plan_v = stack("verid", None, "plan_context", pcaps)
    plan_s = stack("shaped", None, "plan_context", pcaps)
    plan_vz = stack("verid", None, "plan_z0", pcaps)
    plan_sz = stack("shaped", None, "plan_z0", pcaps)
    gap = plan_v - plan_s                         # >0 = shaped plans better (lower dist)
    pv_v = stack("verid", None, "pusher_vel_ff", caps, sub="context")
    pv_s = stack("shaped", None, "pusher_vel_ff", caps, sub="context")

    print(" cap | plan verid      plan shaped     gap(verid-shaped)")
    for ci, h in enumerate(pcaps):
        mv, sv = ms(plan_v[:, ci:ci+1]); msh, ssh = ms(plan_s[:, ci:ci+1])
        mg, sg = ms(gap[:, ci:ci+1])
        star = " *" if mg[0] - sg[0] > 0 else ""
        print(f" {h:>4} | {mv[0]:.3f}+-{sv[0]:.3f}  {msh[0]:.3f}+-{ssh[0]:.3f}  "
              f"{mg[0]:+.3f}+-{sg[0]:.3f}{star}")

    plt.rcParams.update({"figure.dpi": 130, "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(15.5, 4.7), constrained_layout=True)
    xp = np.arange(len(pcaps)); xc = np.arange(len(caps))   # index x -> log-safe, evenly spaced

    # Panel A: planning dist vs capacity (context), shaped vs verid, mean+-sem
    for arr, col, lab in [(plan_v, C_VER, "veridical + context"),
                          (plan_s, C_SHP, "value-shaped + context")]:
        m, s = ms(arr)
        axA.plot(xp, m, "o-", color=col, lw=2.2, label=lab)
        axA.fill_between(xp, m - s, m + s, color=col, alpha=0.15)
    for arr, col, lab in [(plan_vz, C_VER, "veridical + pooled"), (plan_sz, C_SHP, "shaped + pooled")]:
        m, _ = ms(arr)
        axA.plot(xp, m, "s--", color=col, lw=1.4, alpha=0.5, label=lab)
    axA.axhline(rand, color="#888", ls=":", lw=1.2, label=f"random floor {rand:.2f}")
    axA.set_xticks(xp); axA.set_xticklabels(pcaps)
    axA.set_xlabel("FM hidden width"); axA.set_ylabel("median goal-dist (lower = better)")
    axA.set_title(f"Planning: does value-shaping buy control? ({ns} seeds)", fontsize=10)
    axA.legend(fontsize=8, loc="upper right")

    # Panel B: the planning GAP (verid - shaped), mean+-sem + per-seed scatter
    m, s = ms(gap)
    axB.axhline(0, color="#444", lw=1.1)
    axB.plot(xp, m, "o-", color=C_SHP, lw=2.3, zorder=4)
    axB.fill_between(xp, m - s, m + s, color=C_SHP, alpha=0.18)
    for si in range(ns):
        axB.scatter(xp, gap[si], s=18, color="k", alpha=0.45, zorder=3)
    for i in range(len(pcaps)):
        axB.annotate(f"{m[i]:+.3f}\n±{s[i]:.3f}", (xp[i], m[i]), fontsize=7.5,
                     xytext=(0, 10), textcoords="offset points", ha="center")
    axB.set_xticks(xp); axB.set_xticklabels(pcaps)
    axB.set_xlabel("FM hidden width")
    axB.set_ylabel("goal-dist(verid) − goal-dist(shaped)")
    axB.set_title("Behavioral order parameter: >0 = value-shaping helps control",
                  fontsize=9.5)

    # Panel C: FM-side pusher-vel R^2 (value-relevant), shaped vs verid, mean+-sem
    for arr, col, lab in [(pv_v, C_VER, "veridical"), (pv_s, C_SHP, "value-shaped")]:
        m, s = ms(arr)
        axC.plot(xc, m, "o-", color=col, lw=2.2, label=lab)
        axC.fill_between(xc, m - s, m + s, color=col, alpha=0.15)
    axC.set_xticks(xc); axC.set_xticklabels(caps)
    axC.set_xlabel("FM hidden width"); axC.set_ylabel("pusher-vel R² (free-flight)")
    axC.set_title("FM-side lever: shaped frees capacity for the\n(capacity-hungry) pusher",
                  fontsize=9.5)
    axC.legend(fontsize=8.5, loc="lower right")

    fig.suptitle("Capacity-competition (pusher-field) planning, multi-seed: is value-shaping's "
                 "control benefit robust?", fontsize=11)
    outdir = os.path.join(FIGDIR, "meta_value_shaping_seeds")
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, "fig_seeds.png")
    fig.savefig(out, bbox_inches="tight")
    print(f"\n[save] wrote {out}")


if __name__ == "__main__":
    main()
