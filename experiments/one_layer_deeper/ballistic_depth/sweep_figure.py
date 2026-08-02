"""Figures for the re-entry weight sweep and the amplification-instrument calibration.

    python3 one_layer_deeper/ballistic_depth/sweep_figure.py

Reads `results/w_re*/` (produced by the `w_re*` tags) and writes
`figures/sweep/fig_sweep.png`.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

R = Path(__file__).parent / "results"
OUT = Path(__file__).parent / "figures" / "sweep"
WEIGHTS = [("w_re0", 0.0), ("w_re0.1", 0.1), ("w_re0.3", 0.3), ("w_re1.0", 1.0)]


def load(tag):
    return [json.loads(p.read_text()) for p in sorted((R / tag).glob("*.json"))]


def ik(d):
    return {int(k): v for k, v in d.items()}


def ms(v):
    return statistics.mean(v), (statistics.stdev(v) if len(v) > 1 else 0.0)


def horizon(run, arm):
    ex = ik(run["arms"][arm]["exact_seen_x"])
    return float(next((t for t in sorted(ex) if ex[t] < 0.5), max(ex) + 1))


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))

    # --- panel 1: horizon vs re-entry weight -------------------------------------
    ax = axes[0]
    xs, ys, es = [], [], []
    for tag, w in WEIGHTS:
        h = ms([horizon(r, "consist") for r in load(tag)])
        xs.append(w)
        ys.append(h[0])
        es.append(h[1])
    base_h = ms([horizon(r, "base") for r in load("w_re0")])
    ax.errorbar(xs, ys, yerr=es, marker="o", color="#d99a2b", capsize=3,
                label="cycle (w=1) + re-entry")
    ax.axhline(base_h[0], color="#c1554a", ls="--", lw=1.4, label="base (no cycle)")
    ax.fill_between([-0.05, 1.05], base_h[0] - base_h[1], base_h[0] + base_h[1],
                    color="#c1554a", alpha=0.15, linewidth=0)
    ax.set_xlim(-0.05, 1.05)
    ax.set_xlabel("re-entry loss weight")
    ax.set_ylabel("composition horizon  (depth at 50% exact)")
    ax.set_title("The label-reusing term is dose-dependently harmful")
    ax.legend(fontsize=8, loc="lower left")

    # --- panel 2: manifold closure vs re-entry weight ----------------------------
    ax = axes[1]
    ys = [ms([ik(r["arms"]["consist"]["on_manifold_cos"])[30] for r in load(t)])[0]
          for t, _ in WEIGHTS]
    ax.plot(xs, ys, marker="o", color="#2e7d5b")
    ax.axhline(
        ms([ik(r["arms"]["base"]["on_manifold_cos"])[30] for r in load("w_re0")])[0],
        color="#c1554a", ls="--", lw=1.4, label="base",
    )
    ax.set_xlabel("re-entry loss weight")
    ax.set_ylabel(r"cos($h_{30}$, Enc(true $x_{30}$))")
    ax.set_title("...and it degrades manifold closure in lockstep")
    ax.legend(fontsize=8)

    # --- panel 3: instrument calibration -----------------------------------------
    ax = axes[2]
    conds = [("base", "w_re0", "base", "#c1554a"),
             ("cycle only", "w_re0", "consist", "#d99a2b")]
    for lab, tag, arm, col in conds:
        runs = load(tag)
        ts = sorted(ik(runs[0]["arms"][arm]["amplification"]["0.01"]))
        rnd = [ms([ik(r["arms"][arm]["amplification"]["0.01"])[t] for r in runs])[0]
               for t in ts]
        onm = [ms([ik(r["arms"][arm]["amplification_on_manifold"]["0.01"])[t]
                   for r in runs])[0] for t in ts]
        ax.plot(ts, rnd, color=col, ls="--", lw=1.2, label=f"{lab} — random dir")
        ax.plot(ts, onm, color=col, lw=1.8, label=f"{lab} — on-manifold dir")
    ax.axhline(1.0, color="k", ls=":", lw=1)
    ax.plot(ts, [2.0 ** t for t in ts], color="#555", ls=":", lw=1.4,
            label=r"analytic bound $2^t$")
    ax.set_yscale("log")
    ax.set_ylim(0.5, 1e6)
    ax.set_xlabel("rollout step $t$")
    ax.set_ylabel(r"$\|\delta h_t\| / \|\delta h_0\|$")
    ax.set_title("Amplification: instrument calibrated, claim still falsified")
    ax.legend(fontsize=7, loc="upper left")

    fig.tight_layout()
    path = OUT / "fig_sweep.png"
    fig.savefig(path, dpi=160)
    print(f"[figure] {path}")


if __name__ == "__main__":
    main()
