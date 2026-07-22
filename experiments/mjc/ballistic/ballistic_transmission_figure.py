"""Local post-processing: the BALLISTIC-TRANSMISSION result (drift_value_loop Cut 4b).

Headline of `ballistic_transmission.py`: on a controlled FM-quality axis (damping-staleness),
a BALLISTIC controller (open-loop CEM, and especially a genuinely-ballistic behavior-cloned
feedforward motor program) TRANSMITS the FM's error to behavior, while a REACTIVE controller is
robust (re-grounds past it). The FM bridge is behaviorally load-bearing in proportion to
feedforward commitment. Reads figures/ballistic_transmission_<tag>/results.json (seed tags):

  * fig_transmission — control-dist vs FM-error, per controller, mean±sem over seeds (aligned by
                       the controlled d_train knob). Steep = transmits; flat = robust.
  * fig_slopes       — the transmission slope d(control)/d(FM-err) per controller, mean±sem: the
                       one-number summary (ballistic slopes >> reactive).

Run:  python3 mjc/ballistic/ballistic_transmission_figure.py --tags trans_s0 trans_s1 trans_s2
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
LBL = {"reactive": "reactive (re-ground every step)",
       "ballistic_cem": "ballistic CEM (open-loop, model-optimal)",
       "ballistic_bc": "ballistic BC motor-program (feedforward)"}


def load(tags):
    runs = []
    for t in tags:
        p = os.path.join(HERE, "figures", f"ballistic_transmission_{t}", "results.json")
        if os.path.exists(p):
            runs.append(json.load(open(p)))
        else:
            print(f"[warn] missing {p}")
    if not runs:
        raise SystemExit("no runs found")
    return runs


def sem(x, axis=0):
    x = np.asarray(x, float)
    return x.std(axis) / max(1, np.sqrt(x.shape[axis]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["trans_s0", "trans_s1", "trans_s2"])
    args = ap.parse_args()
    runs = load(args.tags)
    nseed = len(runs)
    ctrls = runs[0]["config"]["controllers"]
    d_trains = [r["d_train"] for r in runs[0]["results"]]        # the controlled knob (aligned)
    plt.rcParams.update({"figure.dpi": 130, "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    outdir = os.path.join(HERE, "figures", "ballistic_transmission_contrast")
    os.makedirs(outdir, exist_ok=True)

    # align by d_train index; fm_err and control averaged over seeds
    ne = np.array([[R["results"][i]["fm_err"] for R in runs] for i in range(len(d_trains))])  # (Nq, seeds)
    ne_m, ne_s = ne.mean(1), sem(ne, 1)
    ctrl_m = {c: np.array([[R["results"][i][c] for R in runs] for i in range(len(d_trains))]).mean(1) for c in ctrls}
    ctrl_s = {c: sem(np.array([[R["results"][i][c] for R in runs] for i in range(len(d_trains))]), 1) for c in ctrls}
    slopes = {c: [R["slopes"][c] for R in runs] for c in ctrls}

    # ---- fig_transmission: control vs FM-error, per controller (mean±sem) ----
    fig, ax = plt.subplots(figsize=(8.0, 5.4))
    for c in ctrls:
        sm, ss = np.mean(slopes[c]), sem(slopes[c])
        ax.errorbar(ne_m, ctrl_m[c], xerr=ne_s, yerr=ctrl_s[c], fmt="o-", color=COL.get(c, "#333"),
                    lw=2.4, ms=7, capsize=3, label=f"{LBL.get(c, c)}   (slope {sm:+.2f}±{ss:.2f})")
    ax.set_xlabel("FM prediction error on the TRUE dynamics  (damping-staleness quality axis →)")
    ax.set_ylabel("control goal-dist (lower = better)")
    ax.set_title(f"The FM bridge transmits to behavior in proportion to feedforward commitment\n"
                 f"ballistic control is FM-sensitive; reactive re-grounds past the error ({nseed} seeds, mean±sem)",
                 fontsize=10)
    ax.legend(fontsize=8.8, loc="upper left")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "fig_transmission.png"), bbox_inches="tight")
    plt.close(fig)

    # ---- fig_slopes: the one-number transmission summary ----
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    xs = np.arange(len(ctrls))
    ax.bar(xs, [np.mean(slopes[c]) for c in ctrls], yerr=[sem(slopes[c]) for c in ctrls],
           color=[COL.get(c, "#333") for c in ctrls], capsize=5, width=0.6)
    for i, c in enumerate(ctrls):
        for sv in slopes[c]:
            ax.plot(i, sv, "o", color="k", alpha=0.4, ms=5)
    r_slope = np.mean(slopes.get("reactive", [1]))
    for i, c in enumerate(ctrls):
        ax.text(i, np.mean(slopes[c]) + 0.03, f"{np.mean(slopes[c]):+.2f}\n({np.mean(slopes[c])/r_slope:.1f}× reactive)",
                ha="center", fontsize=8.5)
    ax.set_xticks(xs); ax.set_xticklabels([c.replace("_", "\n") for c in ctrls])
    ax.set_ylabel("transmission slope  d(control-dist)/d(FM-err)")
    ax.set_title(f"How much FM quality reaches behavior, per controller ({nseed} seeds)")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "fig_slopes.png"), bbox_inches="tight")
    plt.close(fig)

    # ---- console summary ----
    print(f"\n=== ballistic transmission ({nseed} seeds, mean±sem) ===")
    print("  FM-err (staleness →): " + "  ".join(f"{v:.3f}" for v in ne_m))
    for c in ctrls:
        sm, ss = np.mean(slopes[c]), sem(slopes[c])
        gap = ctrl_m[c][int(np.argmax(ne_m))] - ctrl_m[c][int(np.argmin(ne_m))]
        print(f"  {c:14s} " + "  ".join(f"{v:.3f}" for v in ctrl_m[c]) +
              f"   slope={sm:+.2f}±{ss:.2f}  ({sm/r_slope:.1f}× reactive)  stale−matched gap={gap:+.3f}")
    print(f"[saved] {outdir}/fig_transmission.png, fig_slopes.png")


if __name__ == "__main__":
    main()
