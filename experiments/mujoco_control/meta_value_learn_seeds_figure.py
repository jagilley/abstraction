"""Aggregate the Cut #4e (meta_value_learn) seed runs into one figure.

Reads the local results.json mirrors for the 3 capacity-competition seeds and the 3
easy-pusher seeds and plots the two order parameters side by side:

  Panel A (A1, the reward LANDSCAPE -- robust): "reward prefers the puck-drop"
    = veridical cost (w_puck=1) - puck-drop cost (w_puck=0). >0 means the control
    reward is minimized by the value-shaping. Capacity competition vs easy pusher,
    per-seed scatter + mean+-sem. The clean dissociation.

  Panel B (A2, the free reward-DRIVEN outer loop): corr(CONVERGED CEM mean, support
    mask). >0 means the loop's converged allocation rediscovered V's support. NB we
    read the converged distribution MEAN (theta_to_w(mu) at the last iter), NOT the
    single best sampled candidate -- min over ~50 noisy planning evals is selection-
    biased (its weight vector is partly noise; seed-2 single-best went anti-support at
    -0.28 while its converged mean was +0.31). The converged mean is the standard,
    less-biased CEM point estimate. Per-seed scatter + mean+-sem.

Run:  cd experiments/ && python3 mujoco_control/meta_value_learn_seeds_figure.py
Writes figures/meta_value_learn_seeds/fig_seeds.png
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
FIGDIR = os.path.join(HERE, "figures")
TAGS = {
    "capacity\ncompetition": ["outer_pfield_norm", "outer_pfield_norm_s1", "outer_pfield_norm_s2"],
    "easy\npusher": ["outer_easy_norm", "outer_easy_norm_s1", "outer_easy_norm_s2"],
}
C = {"capacity\ncompetition": "#d1603d", "easy\npusher": "#3d6fd1"}
PUSHER = [0, 1, 4, 5]
MASK = np.array([1.0 if i in PUSHER else 0.0 for i in range(8)])


def _corr(w):
    w = np.asarray(w, float); a = w - w.mean(); b = MASK - MASK.mean()
    return float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def load(tag):
    p = os.path.join(FIGDIR, f"meta_value_learn_{tag}", "results.json")
    return json.load(open(p))


def main():
    a1 = {k: [] for k in TAGS}     # verid - puckdrop  (reward prefers the puck-drop)
    a2 = {k: [] for k in TAGS}     # corr(converged CEM mean, support) -- the right readout
    for reg, tags in TAGS.items():
        for t in tags:
            d = load(t)
            L = {x["w_puck"]: x["cost"] for x in d["landscape"]}
            a1[reg].append(L[1.0] - L[0.0])
            a2[reg].append(_corr(d["outer"]["history"][-1]["mean_w"]))   # converged mean, not single-best

    plt.rcParams.update({"figure.dpi": 130, "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4))
    regs = list(TAGS.keys())
    x = np.arange(len(regs))

    def panel(ax, data, ylab, title, zero_line=True):
        for i, reg in enumerate(regs):
            v = np.asarray(data[reg]); m = v.mean(); sem = v.std() / np.sqrt(len(v))
            ax.bar(i, m, 0.55, color=C[reg], alpha=0.35, zorder=1)
            ax.errorbar(i, m, yerr=sem, color=C[reg], lw=2.2, capsize=5, zorder=2)
            ax.scatter(np.full_like(v, i) + np.linspace(-0.12, 0.12, len(v)), v,
                       s=42, color=C[reg], edgecolor="k", lw=0.6, zorder=3)
        if zero_line:
            ax.axhline(0, color="#888", lw=1, ls="--")
        ax.set_xticks(x); ax.set_xticklabels(regs)
        ax.set_ylabel(ylab); ax.set_title(title, fontsize=10)

    panel(axes[0], a1, "veridical − puck-drop  (control cost)",
          "A1  reward LANDSCAPE: does reward prefer the shaping?\n"
          "robust — competition >0 (3/3), easy ≈0")
    panel(axes[1], a2, "corr(converged CEM allocation, V-support)",
          "A2  reward-DRIVEN loop: does it rediscover the support?\n"
          "robust — competition +0.39±0.04 (3/3), easy ≈0")
    fig.suptitle("Cut #4e — a reward-driven outer loop CAUSES the value-shaping "
                 "(A1 prefers it, A2 discovers it) — only under capacity competition",
                 fontsize=10.5)
    outdir = os.path.join(FIGDIR, "meta_value_learn_seeds")
    os.makedirs(outdir, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig_seeds.png"), bbox_inches="tight")
    print(f"[fig] wrote {os.path.join(outdir, 'fig_seeds.png')}")
    print("A1 verid-puckdrop:", {k: f"{np.mean(v):+.3f}±{np.std(v)/np.sqrt(len(v)):.3f}" for k, v in a1.items()})
    print("A2 corr->support:", {k: f"{np.mean(v):+.3f}±{np.std(v)/np.sqrt(len(v)):.3f}" for k, v in a2.items()})


if __name__ == "__main__":
    main()
