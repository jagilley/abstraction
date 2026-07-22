"""Local post-processing: the END-TO-END loop (drift_value_loop Cut 4c).

Headline of `ballistic_readapt.py`: after a Type-2 damping drift, online reward-free FM
re-adaptation (the learning layer) restores BALLISTIC motor competence; reactive control never
needed it. So the behavioral value of re-adaptation is ballistic-specific, and the FM
prediction-error (Cut 3's value signal) tracks the ballistic recovery. Reads
figures/ballistic_readapt_<tag>/results.json (seed tags):

  * fig_recovery — control (per controller) + FM-error vs re-adaptation transitions, mean±sem.
  * fig_gain     — recovery gain (stale − recovered) per controller: the behavioral value of
                   re-adaptation, ballistic vs reactive.

Run:  python3 mjc/ballistic/ballistic_readapt_figure.py --tags readapt_s0 readapt_s1 readapt_s2
"""

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
COL = {"reactive": "#7048e8", "ballistic_cem": "#e8590c", "ballistic_bc": "#c92a2a"}
LBL = {"reactive": "reactive (re-ground)", "ballistic_cem": "ballistic CEM (open-loop)",
       "ballistic_bc": "ballistic BC motor-program"}


def load(tags):
    runs = []
    for t in tags:
        p = os.path.join(HERE, "figures", f"ballistic_readapt_{t}", "results.json")
        if os.path.exists(p):
            runs.append(json.load(open(p)))
        else:
            print(f"[warn] missing {p}")
    if not runs:
        raise SystemExit("no runs found")
    return runs


def sem(x, axis=0):
    x = np.asarray(x, float)
    return np.nanstd(x, axis) / np.maximum(1.0, np.sqrt(np.sum(~np.isnan(x), axis)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["readapt_s0", "readapt_s1", "readapt_s2"])
    args = ap.parse_args()
    runs = load(args.tags)
    nseed = len(runs)
    ctrls = runs[0]["config"]["controllers"]
    d0, d1 = runs[0]["config"]["d0"], runs[0]["config"]["d1"]
    tr = np.array([r["transitions"] for r in runs[0]["ladder"]], float)
    xr = np.where(tr == 0, tr[tr > 0].min() / 3.0, tr)     # m=0 plotted just left of first real point
    plt.rcParams.update({"figure.dpi": 130, "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    outdir = os.path.join(HERE, "figures", "ballistic_readapt_contrast")
    os.makedirs(outdir, exist_ok=True)

    fe = np.array([[r["fm_err"] for r in R["ladder"]] for R in runs])
    ctrl = {c: np.array([[r.get(c, np.nan) for r in R["ladder"]] for R in runs]) for c in ctrls}
    gains = {c: [R["recovery"][c]["gain"] for R in runs if c in R["recovery"]] for c in ctrls}

    # ---- fig_recovery: control + FM-err vs transitions ----
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    for c in ctrls:
        m, s = np.nanmean(ctrl[c], 0), sem(ctrl[c], 0)
        ok = ~np.isnan(m)
        ax.errorbar(xr[ok], m[ok], yerr=s[ok], fmt="o-", color=COL.get(c, "#333"), lw=2.4, ms=7, capsize=3,
                    label=f"{LBL.get(c, c)}   (gain {np.mean(gains[c]):+.3f}±{sem(gains[c]):.3f})")
    ax.set_xscale("log")
    ax.set_xlabel(f"reward-free re-adaptation transitions at d1={d1:g}  (learning-loop experience →)")
    ax.set_ylabel("control goal-dist (lower = better)")
    ax2 = ax.twinx()
    ax2.errorbar(xr, fe.mean(0), yerr=sem(fe, 0), fmt="s--", color="#2f9e44", lw=2.0, alpha=0.85, capsize=3,
                 label="FM error on d1 (Cut-3 teacher)")
    ax2.set_ylabel("FM prediction error on d1", color="#2f9e44"); ax2.tick_params(axis="y", labelcolor="#2f9e44")
    ax2.spines["top"].set_visible(False)
    l1, la1 = ax.get_legend_handles_labels(); l2, la2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, la1 + la2, fontsize=8.5, loc="center right")
    ax.set_title(f"After a Type-2 drift (d0={d0:g}→d1={d1:g}), online FM re-adaptation restores BALLISTIC\n"
                 f"control; reactive never needed it — the value of re-adaptation is ballistic-specific "
                 f"({nseed} seeds)", fontsize=9.5)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "fig_recovery.png"), bbox_inches="tight")
    plt.close(fig)

    # ---- fig_gain: behavioral value of re-adaptation per controller ----
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    xs = np.arange(len(ctrls))
    ax.bar(xs, [np.mean(gains[c]) for c in ctrls], yerr=[sem(gains[c]) for c in ctrls],
           color=[COL.get(c, "#333") for c in ctrls], capsize=5, width=0.6)
    r_gain = np.mean(gains.get("reactive", [1]))
    for i, c in enumerate(ctrls):
        ax.text(i, np.mean(gains[c]) + 0.004, f"{np.mean(gains[c]):+.3f}\n({np.mean(gains[c])/r_gain:.1f}× reactive)",
                ha="center", fontsize=8.5)
    ax.set_xticks(xs); ax.set_xticklabels([c.replace("_", "\n") for c in ctrls])
    ax.set_ylabel("control gain from re-adaptation (stale − recovered)")
    ax.set_title(f"Behavioral value of FM re-adaptation is ballistic-specific ({nseed} seeds)")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "fig_gain.png"), bbox_inches="tight")
    plt.close(fig)

    print(f"\n=== ballistic re-adaptation ({nseed} seeds) — drift d0={d0:g}→d1={d1:g} ===")
    print("  transitions:  " + "  ".join(f"{int(t):6d}" for t in tr))
    print("  FM error:     " + "  ".join(f"{v:6.3f}" for v in fe.mean(0)))
    for c in ctrls:
        m = np.nanmean(ctrl[c], 0)
        print(f"  {c:14s}" + "  ".join(f"{v:6.3f}" if not np.isnan(v) else "   -  " for v in m) +
              f"   gain={np.mean(gains[c]):+.3f}±{sem(gains[c]):.3f}  ({np.mean(gains[c])/r_gain:.1f}× reactive)")
    print(f"[saved] {outdir}/fig_recovery.png, fig_gain.png")


if __name__ == "__main__":
    main()
