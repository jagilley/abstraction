"""Local post-processing: the BALLISTIC-CONTROLLER cut (drift_value_loop Cut 4).

Headline of `ballistic_control.py`: Cut 3 found downstream control is a NEAR-BLIND grader of
the FM (a replanning CEM-MPC controller reaches goals ~as well with a stale FM as a fresh one).
That is a fact about the CONTROLLER, not the value: the cerebellar FM exists because you cannot
replan fast enough, so its value is proportional to the COMMITMENT HORIZON. Grade the IDENTICAL
FMs with controllers of varying ballistic-ness and the control-over-b landscape gains a gradient.

Reads figures/ballistic_control_<tag>/results.json (seed tags) and emits:

  * fig_landscape     — control-over-b at each commitment horizon (reactive→ballistic), mean±sem,
                        with the corridor FM-error landscape (controller-INVARIANT) as reference.
                        Prediction: FLAT when reactive → INTERIOR OPTIMUM when ballistic.
  * fig_transmission  — (money) rel. spread of control-over-b vs the commitment horizon, with the
                        FM-error spread as a dashed reference. The FM bridge transmits to control
                        as the controller goes ballistic.
  * fig_selftune      — self-tuned b under a BALLISTIC control teacher: online_ctrl now converges
                        toward the FM-err optimum (Cut 3's reactive online_ctrl wandered).

Run:  python3 mujoco_control/ballistic_control_figure.py \
          --land-tags land_s0 land_s1 land_s2 --selftune-tags selftune_s0 selftune_s1 selftune_s2
"""

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)


def load(tags):
    runs = []
    for t in tags:
        p = os.path.join(HERE, "figures", f"ballistic_control_{t}", "results.json")
        if os.path.exists(p):
            runs.append(json.load(open(p)))
        else:
            print(f"[warn] missing {p}")
    return runs


def fixed_b_arms(R):
    out = []
    for a in R["results"]:
        if a.startswith("b") and a[1:].replace(".", "", 1).isdigit():
            out.append((float(a[1:]), a))
    return sorted(out)


def agg(rows):
    M = np.array(rows, float)
    return M.mean(0), M.std(0) / max(1, np.sqrt(M.shape[0]))


def re_color(re, sweep):
    if len(sweep) <= 1:
        return "#e8590c"
    frac = (re - min(sweep)) / (max(sweep) - min(sweep) + 1e-9)
    return plt.cm.coolwarm(frac)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--land-tags", nargs="+", default=["land_s0", "land_s1", "land_s2"],
                    help="runs WITH fixed-b arms -> the commitment-horizon landscape")
    ap.add_argument("--selftune-tags", nargs="+",
                    default=["selftune_s0", "selftune_s1", "selftune_s2"],
                    help="runs with a DISPLACED b_init (online arms) under the ballistic teacher")
    args = ap.parse_args()
    runs = load(args.land_tags)
    if not runs:
        raise SystemExit("no landscape runs found")
    st_runs = load(args.selftune_tags)
    nseed = len(runs)
    plt.rcParams.update({"figure.dpi": 130, "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    outdir = os.path.join(HERE, "figures", "ballistic_control_contrast")
    os.makedirs(outdir, exist_ok=True)

    sweep = runs[0]["replan_sweep"]
    plan_H = runs[0]["config"]["plan_H"]
    bvals = [b for b, _ in fixed_b_arms(runs[0])]
    arm_by_b = {b: a for b, a in fixed_b_arms(runs[0])}

    # ---- collect per-seed: control[re][b] and corridor[b] ----
    ctrl = {re: {b: [] for b in bvals} for re in sweep}
    corr = {b: [] for b in bvals}
    for R in runs:
        for b in bvals:
            s = R["summary"][arm_by_b[b]]
            corr[b].append(s["final_corridor_err"])
            for re in sweep:
                ctrl[re][b].append(s["final_control_by_replan"][str(re)])
    corr_m = np.array([np.mean(corr[b]) for b in bvals])
    corr_s = np.array([np.std(corr[b]) / max(1, np.sqrt(nseed)) for b in bvals])
    ctrl_m = {re: np.array([np.mean(ctrl[re][b]) for b in bvals]) for re in sweep}
    ctrl_s = {re: np.array([np.std(ctrl[re][b]) / max(1, np.sqrt(nseed)) for b in bvals]) for re in sweep}
    b_best_corr = bvals[int(np.argmin(corr_m))]

    def rel_spread(v):
        return (v.max() - v.min()) / (v.mean() + 1e-9)

    # ---- fig_landscape: control-over-b per commitment horizon + FM-err reference ----
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))
    for re in sweep:
        tag = "ballistic" if re >= plan_H else ("reactive" if re <= 2 else "mid")
        bmin = bvals[int(np.argmin(ctrl_m[re]))]
        axes[0].errorbar(bvals, ctrl_m[re], yerr=ctrl_s[re], fmt="o-", color=re_color(re, sweep),
                         lw=2.3, capsize=3,
                         label=f"replan={re} ({tag}): spread {rel_spread(ctrl_m[re]):.0%}, argmin b={bmin:g}")
    axes[0].set_xlabel("explore/exploit balance b  (0 = exploit, 1 = explore)")
    axes[0].set_ylabel("control goal-dist (↓)")
    axes[0].set_title("control-over-b: FLAT (reactive) → INTERIOR OPTIMUM (ballistic)")
    axes[0].legend(fontsize=8, loc="best")
    axes[1].errorbar(bvals, corr_m, yerr=corr_s, fmt="s-", color="#2f9e44", lw=2.4, capsize=3,
                     label=f"corridor FM err (invariant); argmin b={b_best_corr:g}")
    axes[1].axvline(b_best_corr, color="#2f9e44", ls=":", lw=1.4, alpha=0.7)
    axes[1].set_xlabel("explore/exploit balance b"); axes[1].set_ylabel("corridor FM prediction err (↓)")
    axes[1].set_title("the FM-error teacher (controller-INVARIANT reference)")
    axes[1].legend(fontsize=8.5)
    fig.suptitle(f"The ballistic controller makes the FM bridge transmit to behavior ({nseed} seeds)",
                 fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "fig_landscape.png"), bbox_inches="tight")
    plt.close(fig)

    # ---- fig_transmission (money): spread(control-over-b) vs commitment horizon ----
    # per-seed spread so we can show sem
    spreads = {re: [] for re in sweep}
    for R in runs:
        for re in sweep:
            v = np.array([R["summary"][arm_by_b[b]]["final_control_by_replan"][str(re)] for b in bvals])
            spreads[re].append((v.max() - v.min()) / (v.mean() + 1e-9))
    sp_m = np.array([np.mean(spreads[re]) for re in sweep])
    sp_s = np.array([np.std(spreads[re]) / max(1, np.sqrt(nseed)) for re in sweep])
    corr_spread_per = []
    for R in runs:
        v = np.array([R["summary"][arm_by_b[b]]["final_corridor_err"] for b in bvals])
        corr_spread_per.append((v.max() - v.min()) / (v.mean() + 1e-9))
    corr_spread_m = float(np.mean(corr_spread_per))

    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    ax.errorbar(sweep, sp_m, yerr=sp_s, fmt="o-", color="#e8590c", lw=2.8, ms=9, capsize=4,
                label="control-over-b spread")
    ax.axhline(corr_spread_m, color="#2f9e44", ls="--", lw=2.0,
               label=f"corridor FM-err spread ({corr_spread_m:.0%}, invariant)")
    ax.axvline(plan_H, color="#868e96", ls=":", lw=1.6, label=f"fully ballistic (plan_H={plan_H})")
    ax.set_xlabel("commitment horizon  replan_every  (steps committed open-loop)")
    ax.set_ylabel("rel. spread of control over b")
    ax.set_title(f"The FM bridge transmits to control as the controller goes ballistic\n"
                 f"({nseed} seeds, mean±sem)", fontsize=10)
    ax.legend(fontsize=9, loc="best")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "fig_transmission.png"), bbox_inches="tight")
    plt.close(fig)

    # ---- fig_selftune: online_ctrl under a BALLISTIC teacher now converges ----
    if st_runs:
        b0 = st_runs[0]["config"].get("b_init", 0.5)
        tr = st_runs[0]["config"].get("teacher_replan", plan_H)
        fig, ax = plt.subplots(figsize=(8.4, 5.0))
        ocol = {"online_front": "#e8590c", "online_ctrl": "#7048e8"}
        for arm, lab in [("online_ctrl", f"control teacher @replan={tr} (BALLISTIC — now sighted)"),
                         ("online_front", "front teacher (FM error, reference)")]:
            trajs = []
            for R in st_runs:
                if arm in R["results"] and R["results"][arm]["outer_hist"]:
                    trajs.append([h["b_theta"] for h in R["results"][arm]["outer_hist"]])
            if not trajs:
                continue
            L = min(len(t) for t in trajs)
            m, s = agg([t[:L] for t in trajs])
            ep = np.arange(L)
            ax.plot(ep, m, "o-", color=ocol.get(arm, "#9c36b5"), lw=2.4, label=f"{arm}: {lab}")
            ax.fill_between(ep, m - s, m + s, color=ocol.get(arm, "#9c36b5"), alpha=0.15)
        ax.axhline(b_best_corr, color="#2f9e44", ls=":", lw=1.8, label=f"FM-err optimum (b={b_best_corr:g})")
        ax.axhline(b0, color="#aaa", ls="--", lw=1.4, label=f"b_init = {b0:g} (displaced)")
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("outer-loop epoch"); ax.set_ylabel("self-tuned balance b = σ(θ)")
        ax.set_title(f"Self-tuning b under a BALLISTIC control teacher: online_ctrl now CONVERGES\n"
                     f"(Cut 3's reactive online_ctrl wandered) — {len(st_runs)} seeds, mean±sem", fontsize=9.5)
        ax.legend(fontsize=8.5, loc="best")
        fig.tight_layout(); fig.savefig(os.path.join(outdir, "fig_selftune.png"), bbox_inches="tight")
        plt.close(fig)

    # ---- console summary ----
    print(f"\n=== ballistic controller: commitment-horizon landscape ({nseed} seeds) ===")
    print(f"  b:            " + "  ".join(f"{b:>6.2f}" for b in bvals))
    print(f"  corridor-err: " + "  ".join(f"{v:>6.3f}" for v in corr_m) +
          f"   (INVARIANT ref; spread {corr_spread_m:.0%}; argmin b={b_best_corr:g})")
    for re in sweep:
        tag = "ballistic" if re >= plan_H else ("reactive" if re <= 2 else "mid")
        bmin = bvals[int(np.argmin(ctrl_m[re]))]
        print(f"  ctrl@re={re:<3d}  " + "  ".join(f"{v:>6.3f}" for v in ctrl_m[re]) +
              f"   ({tag}; spread {rel_spread(ctrl_m[re]):.0%}; argmin b={bmin:g})")
    print(f"  --> transmission: spread rises {sp_m[0]:.1%} (re={sweep[0]}) → {sp_m[-1]:.1%} (re={sweep[-1]})")
    if st_runs:
        for arm in ["online_ctrl", "online_front"]:
            bs = [R["summary"][arm]["final_b"] for R in st_runs if arm in R["summary"]]
            if bs:
                print(f"  self-tune {arm:13s} b = {np.mean(bs):.3f} ± {np.std(bs)/max(1,np.sqrt(len(bs))):.3f}"
                      f"  (target FM-err optimum b={b_best_corr:g})")
    print(f"[saved] {outdir}/fig_landscape.png, fig_transmission.png, fig_selftune.png")


if __name__ == "__main__":
    main()
