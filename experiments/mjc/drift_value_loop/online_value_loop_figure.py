"""Local post-processing: the online-value-loop TEACHER CONTRAST + the b-landscape.

The headline of `online_value_loop.py`: grading the meta-loop by the value-relevant FM
re-adaptation error (the 'front' teacher) gives it a real gradient over the explore/exploit
balance b, where downstream control (the 'control' teacher, = the parent's blind grader) is
flat. Reads figures/online_value_loop_<tag>/results.json for the seed tags and emits:

  * fig_landscape — the two b-landscapes over the FIXED-b arms (mean±sem over seeds):
        corridor FM error (the front teacher) vs control goal-dist (the control teacher).
        The teacher with the interior optimum / larger spread is the one with a gradient.
  * fig_selftune  — the self-tuned b trajectories: online_front converges toward argmin
        corridor-error; online_ctrl wanders (flat control gives no gradient).

Run:  python3 mujoco_control/online_value_loop_figure.py
      python3 mujoco_control/online_value_loop_figure.py --tags teacher_s0 teacher_s1 teacher_s2
"""

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
C_FRONT, C_CTRL = "#e8590c", "#7048e8"


def load(tags):
    runs = []
    for t in tags:
        p = os.path.join(HERE, "figures", f"online_value_loop_{t}", "results.json")
        if os.path.exists(p):
            runs.append(json.load(open(p)))
        else:
            print(f"[warn] missing {p}")
    if not runs:
        raise SystemExit("no runs found")
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["teacher_s0", "teacher_s1", "teacher_s2"],
                    help="runs WITH fixed-b arms -> the two b-landscapes")
    ap.add_argument("--selftune-tags", nargs="+",
                    default=["selftune15_s0", "selftune15_s1", "selftune15_s2"],
                    help="runs with a DISPLACED b_init (online arms only) -> the convergence demo")
    args = ap.parse_args()
    runs = load(args.tags)
    st_runs = load(args.selftune_tags) if args.selftune_tags else runs
    nseed = len(runs)
    bvals = [b for b, _ in fixed_b_arms(runs[0])]
    arm_by_b = {b: a for b, a in fixed_b_arms(runs[0])}
    plt.rcParams.update({"figure.dpi": 130, "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    outdir = os.path.join(HERE, "figures", "online_value_loop_contrast")
    os.makedirs(outdir, exist_ok=True)

    # ---- the two landscapes over fixed b (mean±sem over seeds) ----
    corr = {b: [] for b in bvals}; ctrl = {b: [] for b in bvals}
    for R in runs:
        for b in bvals:
            s = R["summary"][arm_by_b[b]]
            corr[b].append(s["final_corridor_err"]); ctrl[b].append(s["final_control"])
    corr_m = np.array([np.mean(corr[b]) for b in bvals]); corr_s = np.array([np.std(corr[b]) / max(1, np.sqrt(nseed)) for b in bvals])
    ctrl_m = np.array([np.mean(ctrl[b]) for b in bvals]); ctrl_s = np.array([np.std(ctrl[b]) / max(1, np.sqrt(nseed)) for b in bvals])
    b_best_corr = bvals[int(np.argmin(corr_m))]

    fig, ax1 = plt.subplots(figsize=(8.4, 5.0))
    ax1.errorbar(bvals, corr_m, yerr=corr_s, fmt="o-", color=C_FRONT, lw=2.3, capsize=3,
                 label="FRONT teacher: corridor FM error")
    ax1.set_xlabel("explore/exploit balance b  (0 = pure exploit, 1 = pure explore)")
    ax1.set_ylabel("corridor FM error (front teacher, ↓)", color=C_FRONT)
    ax1.tick_params(axis="y", labelcolor=C_FRONT)
    ax1.axvline(b_best_corr, color=C_FRONT, ls=":", lw=1.4, alpha=0.7)
    ax2 = ax1.twinx()
    ax2.errorbar(bvals, ctrl_m, yerr=ctrl_s, fmt="s--", color=C_CTRL, lw=2.1, capsize=3,
                 label="CONTROL teacher: goal-dist")
    ax2.set_ylabel("control goal-dist (control teacher, ↓)", color=C_CTRL)
    ax2.tick_params(axis="y", labelcolor=C_CTRL)
    ax2.spines["top"].set_visible(False)
    corr_spread = (corr_m.max() - corr_m.min()) / (corr_m.mean() + 1e-9)
    ctrl_spread = (ctrl_m.max() - ctrl_m.min()) / (ctrl_m.mean() + 1e-9)
    ax1.set_title(f"The b-landscape under two teachers ({nseed} seeds)\n"
                  f"FRONT has structure (rel. spread {corr_spread:.0%}, argmin b={b_best_corr:g}); "
                  f"CONTROL is flatter ({ctrl_spread:.0%}) — the blind grader", fontsize=10)
    l1, la1 = ax1.get_legend_handles_labels(); l2, la2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, la1 + la2, fontsize=8.5, loc="upper center")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "fig_landscape.png"), bbox_inches="tight")
    plt.close(fig)

    # ---- the self-tuned b trajectories (front vs control teacher), DISPLACED init ----
    b0 = st_runs[0]["config"].get("b_init", 0.5)
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    for arm, c, lab in [("online_front", C_FRONT, "front teacher (FM error)"),
                        ("online_ctrl", C_CTRL, "control teacher (goal-dist)")]:
        trajs = []
        for R in st_runs:
            if arm in R["results"] and R["results"][arm]["outer_hist"]:
                trajs.append([h["b_theta"] for h in R["results"][arm]["outer_hist"]])
        if not trajs:
            continue
        L = min(len(t) for t in trajs)
        trajs = [t[:L] for t in trajs]
        m, s = agg(trajs)
        ep = np.arange(L)
        ax.plot(ep, m, "o-", color=c, lw=2.4, label=f"{arm}: {lab}")
        ax.fill_between(ep, m - s, m + s, color=c, alpha=0.15)
    ax.axhline(b_best_corr, color="#2f9e44", ls=":", lw=1.8, label=f"landscape optimum (b={b_best_corr:g})")
    ax.axhline(b0, color="#aaa", ls="--", lw=1.4, label=f"b_init = {b0:g} (displaced)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("outer-loop epoch"); ax.set_ylabel("self-tuned balance b = σ(θ)")
    ax.set_title(f"Self-tuning b from a displaced init (b={b0:g}): the FRONT teacher CLIMBS to the\n"
                 f"optimum; the CONTROL teacher stays stuck (no gradient) — {len(st_runs)} seeds, mean±sem", fontsize=9.5)
    ax.legend(fontsize=8.5, loc="best")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "fig_selftune.png"), bbox_inches="tight")
    plt.close(fig)

    # ---- console summary ----
    print(f"\n=== online value loop: teacher contrast ({nseed} seeds) ===")
    print(f"  b:            " + "  ".join(f"{b:>6.2f}" for b in bvals))
    print(f"  corridor-err: " + "  ".join(f"{v:>6.3f}" for v in corr_m) + f"   (FRONT teacher; argmin b={b_best_corr:g})")
    print(f"  control:      " + "  ".join(f"{v:>6.3f}" for v in ctrl_m) + "   (CONTROL teacher)")
    print(f"  rel spread:   FRONT {corr_spread:.1%}  vs  CONTROL {ctrl_spread:.1%}")
    for arm in ["online_front", "online_ctrl"]:
        bs = [R["summary"][arm]["final_b"] for R in runs if arm in R["summary"]]
        if bs:
            print(f"  {arm:13s} learned b = {np.mean(bs):.3f} ± {np.std(bs)/max(1,np.sqrt(len(bs))):.3f}")
    print(f"[saved] {outdir}/fig_landscape.png, fig_selftune.png")


if __name__ == "__main__":
    main()
