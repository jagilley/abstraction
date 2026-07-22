"""Local post-processing: the Phase-2 CAPACITY BOUNDARY of value-carving under drift.

guidance.md #1: value-carving only pays when the FM is small enough that modeling the
value-irrelevant (drifting) puck genuinely STEALS capacity from the value-relevant pusher.
This aggregates value_carved_drift runs at several FM widths and plots the value_carved −
veridical re-adaptation gap (FM + control) vs capacity — the boundary where the carving
starts to bite (or the informative null if the controller/capacity stays robust throughout).

Reads figures/value_carved_drift_<tag>/results.json for the (hidden, tag) pairs given.

Run:  python3 mjc/drift_value_loop/value_carved_drift_capacity_figure.py
      python3 mjc/drift_value_loop/value_carved_drift_capacity_figure.py --runs 24:comp_h24 32:comp_h32 64:comp_v1
"""

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
C_CARV, C_VER = "#3d6fd1", "#d1603d"


def load(tag):
    p = os.path.join(HERE, "figures", f"value_carved_drift_{tag}", "results.json")
    return json.load(open(p)) if os.path.exists(p) else None


def late_mean(seq, q):
    return float(np.mean(seq[-q:]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["24:comp_h24", "32:comp_h32", "64:comp_v1"],
                    help="hidden:tag pairs")
    args = ap.parse_args()

    rows = []
    for pair in args.runs:
        h, tag = pair.split(":")
        R = load(tag)
        if R is None:
            print(f"[warn] missing {tag}; skip"); continue
        T = len(R["phi_seq"]); q = max(1, T // 4)
        mb = str(max(R["control_budgets"]))
        # late per-drift control goal-dist (converged control at the drift, best budget)
        cd_c = late_mean([c[mb] for c in R["ctrl_curves"]["value_carved"]], q)
        cd_v = late_mean([c[mb] for c in R["ctrl_curves"]["veridical"]], q)
        # late FM pusher-vel R² at few-shot budget (N index 1 = smallest >0)
        fb = str(R["budgets"][1])
        pv_c = late_mean([c[fb] for c in R["pv_curves"]["value_carved"]], q)
        pv_v = late_mean([c[fb] for c in R["pv_curves"]["veridical"]], q)
        rows.append(dict(h=int(h), oracle=R["oracle_pusher_vel"],
                         fm_cum_c=sum(R["ttr_numeric"]["value_carved"]),
                         fm_cum_v=sum(R["ttr_numeric"]["veridical"]),
                         ctrl_cum_c=sum(R["ctrl_ttr_numeric"]["value_carved"]),
                         ctrl_cum_v=sum(R["ctrl_ttr_numeric"]["veridical"]),
                         cd_c=cd_c, cd_v=cd_v, pv_c=pv_c, pv_v=pv_v,
                         puck_c=float(np.nanmean([x for x in R["puck_r2"]["value_carved"] if x is not None])),
                         puck_v=float(np.nanmean([x for x in R["puck_r2"]["veridical"] if x is not None]))))
    rows.sort(key=lambda r: r["h"])
    if not rows:
        raise SystemExit("no runs found")
    hs = [r["h"] for r in rows]

    plt.rcParams.update({"figure.dpi": 130, "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.5))

    # panel 1: converged control goal-dist vs capacity (carved vs veridical)
    ax = axes[0]
    ax.plot(hs, [r["cd_c"] for r in rows], "o-", color=C_CARV, lw=2.1, label="value-carved")
    ax.plot(hs, [r["cd_v"] for r in rows], "s-", color=C_VER, lw=2.1, label="veridical")
    ax.set_xscale("log", base=2); ax.set_xticks(hs); ax.set_xticklabels(hs)
    ax.set_xlabel("FM hidden width (capacity)"); ax.set_ylabel("late control goal-dist (↓ better)")
    ax.set_title("Control re-adaptation vs capacity", fontsize=10); ax.legend(fontsize=9)

    # panel 2: few-shot pusher-vel R² vs capacity
    ax = axes[1]
    ax.plot(hs, [r["pv_c"] for r in rows], "o-", color=C_CARV, lw=2.1, label="value-carved")
    ax.plot(hs, [r["pv_v"] for r in rows], "s-", color=C_VER, lw=2.1, label="veridical")
    ax.set_xscale("log", base=2); ax.set_xticks(hs); ax.set_xticklabels(hs)
    ax.set_xlabel("FM hidden width"); ax.set_ylabel("late few-shot pusher-vel R²")
    ax.set_title("Value-relevant FM fidelity vs capacity", fontsize=10); ax.legend(fontsize=9)

    # panel 3: the GAP (veridical − carved; >0 = carving helps) vs capacity
    ax = axes[2]
    ctrl_gap = [r["cd_v"] - r["cd_c"] for r in rows]      # >0 = carved lower dist = better
    pv_gap = [r["pv_c"] - r["pv_v"] for r in rows]        # >0 = carved higher R² = better
    ax.axhline(0, color="#888", lw=1)
    ax.plot(hs, ctrl_gap, "o-", color="#7a3fb0", lw=2.1, label="control gap (verid−carved dist)")
    ax.plot(hs, pv_gap, "s--", color="#2f9e44", lw=2.0, label="FM pusher-vel gap (carved−verid R²)")
    ax.set_xscale("log", base=2); ax.set_xticks(hs); ax.set_xticklabels(hs)
    ax.set_xlabel("FM hidden width"); ax.set_ylabel("value-carving benefit (↑ = carving helps)")
    ax.set_title("The capacity boundary: does carving bite?", fontsize=10); ax.legend(fontsize=8.5)

    fig.suptitle("Value-carving under drift is CAPACITY-GATED: dropping the drifting value-irrelevant puck "
                 "pays only where capacity binds\n(guidance #1; the 4d capacity-competition boundary, now in the "
                 "drift sequence)", fontsize=10.5)
    outdir = os.path.join(HERE, "figures", "value_carved_drift_capacity")
    os.makedirs(outdir, exist_ok=True)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "fig_capacity_boundary.png"), bbox_inches="tight")
    plt.close(fig)

    print(f"\n=== Phase-2 capacity boundary ({len(rows)} widths) ===")
    print(f"{'h':>4} {'oracle_pv':>9} {'puckR²(c/v)':>14} {'few-pvR²(c/v)':>14} "
          f"{'ctrl-dist(c/v)':>15} {'FMcum(c/v)':>13} {'CTRLcum(c/v)':>13}")
    for r in rows:
        print(f"{r['h']:>4} {r['oracle']:>9.3f} {r['puck_c']:>6.1f}/{r['puck_v']:>5.1f} "
              f"{r['pv_c']:>6.2f}/{r['pv_v']:>5.2f} {r['cd_c']:>7.2f}/{r['cd_v']:>6.2f} "
              f"{r['fm_cum_c']:>6.0f}/{r['fm_cum_v']:>5.0f} {r['ctrl_cum_c']:>6.0f}/{r['ctrl_cum_v']:>5.0f}")
    print(f"[saved] {outdir}/fig_capacity_boundary.png")


if __name__ == "__main__":
    main()
