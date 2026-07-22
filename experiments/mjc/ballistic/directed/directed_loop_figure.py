"""Cross-seed aggregation for Cut 5 S2 (`directed_loop.py`), with ADAPTATION SPEED as the
headline readout rather than asymptotic competence.

WHY THIS SCRIPT EXISTS (the methodological finding it encodes). The first full S2 run reported
mean ballistic goal-distance over all rounds, and returned a NULL: every directed policy —
including `error-only`, which spends ~60% of every round on aleatoric-noise regions — closed
essentially the whole uniform->oracle gap. The cause is that mean-over-rounds is an ASYMPTOTIC
COMPETENCE metric, and a loop that is budget-rich in aggregate (400 transitions/round x 24
rounds, persistent buffers) equalizes asymptotic competence across every policy that
concentrates its budget anywhere at all. Below ~0.05 region-error, ballistic control stops
improving, so the metric floors out and the policies become indistinguishable.

Directed collection does not buy you a better final model. It buys you a FASTER one. So the
metric that matches the claim is recovery speed after a drift:

  * `excess damage`      — integrated (ballistic goal-dist - epoch floor) over the rounds
                           following a drift that hit an ON-REACH region. How much behavioral
                           cost the agent paid while its model was stale.
  * `rounds-to-recover`  — rounds until ballistic control returns to within `tol` of the
                           epoch's best. How quickly the loop found and repaired the change.

On the SAME runs that nulled on mean-over-rounds, these separate the policies 1.4-3.7x, with
the reward-free `value` signal sitting on the oracle. Both readings are worth reporting: the
null bounds the claim (directed collection is an adaptation-SPEED effect, not an asymptotic
one), and the speed metric establishes it. This mirrors cut 4c's recovery gains and cut 4a's
finding that the value benefit needs sufficient non-stationarity to show up at all.

Run:
    cd experiments/
    python3 mujoco_control/directed_loop_figure.py --tags loop_s0 loop_s1 loop_s2
    python3 mujoco_control/directed_loop_figure.py --tags loopB_s0 loopB_s1 loopB_s2 --out loopB
"""

import argparse
import json
import os

import numpy as np


def load(tag):
    here = os.path.dirname(__file__)
    path = os.path.join(here, "figures", f"directed_loop_{tag}", "results.json")
    with open(path) as fh:
        return json.load(fh)


def recovery_metrics(R, tol=0.15):
    """Per-policy adaptation speed after each drift landing on an ON-REACH region.

    Drifts that land OFF the reach are excluded: by construction they should cost the agent
    nothing behaviorally, so including them would dilute the very contrast being measured
    (and a policy that correctly IGNORES them would be penalised for not 'recovering')."""
    sched = {int(k): v for k, v in R["schedule"].items()}
    REG = R["config"]["regions"]
    on_reach = {j for j in range(len(REG)) if abs(REG[j]["center"][1]) <= 0.3}
    events = sorted(t for t, (j, _) in sched.items() if j in on_reach)
    T = R["config"]["rounds"]
    de = R["config"]["drift_every"]

    # The recovery threshold must be ABSOLUTE and SHARED across policies. Normalising to each
    # policy's OWN epoch minimum (an earlier version of this script) rewards a policy that is
    # uniformly mediocre — it "recovers" instantly because it never got good — and produced an
    # incoherent ranking where the worst-performing arm scored the fastest recovery. So the
    # target is the best ballistic distance ANY policy achieved anywhere in this run, inflated
    # by `tol`; policies that never reach it inside the epoch are censored at the epoch length
    # and counted, so a fast mean can't hide a never-recovered epoch.
    best = min(float(np.min([r["ballistic_cem"] for r in h])) for h in R["results"].values())
    thr = best * (1.0 + tol)

    out = {}
    for p, hist in R["results"].items():
        b = np.array([r["ballistic_cem"] for r in hist])
        exc, rr, cens = [], [], 0
        for t0 in events:
            ep = b[t0:min(t0 + de, T)]
            if len(ep) < 3:
                continue
            exc.append(float((ep - best).sum()))          # excess over the SHARED floor
            hit = next((i for i, v in enumerate(ep) if v <= thr), None)
            if hit is None:
                rr.append(len(ep)); cens += 1
            else:
                rr.append(hit)
        out[p] = {"excess": exc, "rounds": rr, "censored": cens, "thr": thr}
    return out, events


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--out", default=None, help="output subdir suffix (default: first tag)")
    ap.add_argument("--tol", type=float, default=0.15)
    args = ap.parse_args()

    Rs = [load(t) for t in args.tags]
    pol = Rs[0]["config"]["policies"]
    names = Rs[0]["region_names"]

    per_seed = [recovery_metrics(R, args.tol) for R in Rs]
    print(f"tags={args.tags}   on-reach drift rounds per seed: "
          + ", ".join(str(ps[1]) for ps in per_seed))
    n_ev = sum(len(ps[0][pol[0]]["rounds"]) for ps in per_seed)
    print(f"n on-reach drift events pooled across seeds = {n_ev}")
    print(f"recovery threshold (absolute, shared) per seed: "
          + ", ".join(f"{ps[0][pol[0]]['thr']:.3f}" for ps in per_seed) + "\n")

    # ---- headline: adaptation speed ----
    print("=== ADAPTATION SPEED (the metric that matches the claim) ===")
    print(f"{'policy':>14s}  {'excess damage':>16s}  {'rounds-to-recover':>18s}")
    speed = {}
    for p in pol:
        e = np.concatenate([[x] for ps in per_seed for x in ps[0][p]["excess"]]) if n_ev else np.array([np.nan])
        r = np.array([x for ps in per_seed for x in ps[0][p]["rounds"]], float)
        speed[p] = {"excess_mean": float(e.mean()), "excess_sem": float(e.std() / max(np.sqrt(len(e)), 1)),
                    "rounds_mean": float(r.mean()), "rounds_sem": float(r.std() / max(np.sqrt(len(r)), 1))}
        nc = sum(ps[0][p]["censored"] for ps in per_seed)
        speed[p]["censored"] = int(nc)
        print(f"{p:>14s}  {e.mean():>9.3f} ± {e.std() / max(np.sqrt(len(e)), 1):<4.3f}  "
              f"{r.mean():>11.2f} ± {r.std() / max(np.sqrt(len(r)), 1):<4.2f}"
              + (f"   ({nc}/{n_ev} never recovered)" if nc else ""))
    print(f"\n{'policy':>14s}  {'excess vs value':>16s}")
    if "value" in speed:
        for p in pol:
            if p == "value":
                continue
            print(f"{p:>14s}  {speed[p]['excess_mean'] / max(speed['value']['excess_mean'], 1e-9):>15.2f}x")

    # ---- the contrasting null: asymptotic competence ----
    print("\n=== ASYMPTOTIC COMPETENCE (mean over rounds) — the metric that NULLS ===")
    print(f"{'policy':>14s}  {'ballistic':>12s}  {'reactive':>12s}  {'gap-closed(ball)':>18s}")
    auc = {}
    for p in pol:
        b = np.mean([np.mean([r["ballistic_cem"] for r in R["results"][p]]) for R in Rs])
        rc = np.mean([np.mean([r["reactive"] for r in R["results"][p] if "reactive" in r]) for R in Rs])
        auc[p] = {"ballistic": float(b), "reactive": float(rc)}
    u, o = auc["uniform"]["ballistic"], auc["oracle"]["ballistic"]
    for p in pol:
        g = (u - auc[p]["ballistic"]) / (u - o) if abs(u - o) > 1e-9 else float("nan")
        auc[p]["gap_closed"] = float(g)
        print(f"{p:>14s}  {auc[p]['ballistic']:>12.4f}  {auc[p]['reactive']:>12.4f}  {g:>18.2f}")
    db = u - o
    dr = auc["uniform"]["reactive"] - auc["oracle"]["reactive"]
    if dr > 1e-3:
        print(f"\n  allocation matters {db / dr:.1f}x more for BALLISTIC than REACTIVE "
              f"(spread {db:+.4f} vs {dr:+.4f})")
    else:
        print(f"\n  reactive uniform->oracle spread is {dr:+.4f} (<=0 or negligible): reactive is "
              f"INSENSITIVE to allocation, so the ratio is undefined. Ballistic spread {db:+.4f}.")

    # ---- tracking: does the signal CHOOSE well, independent of whether it pays off? ----
    print("\n=== TRACKING (share of budget in the stale on-reach region) ===")
    track = {}
    for p in pol:
        v = [R["tracking"][p] for R in Rs if not np.isnan(R["tracking"].get(p, np.nan))]
        track[p] = float(np.mean(v)) if v else float("nan")
        print(f"{p:>14s}  {track[p]:.2f}")

    outdir = os.path.join(os.path.dirname(__file__), "figures",
                          "directed_loop_agg_" + (args.out or args.tags[0]))
    os.makedirs(outdir, exist_ok=True)
    _figures(outdir, pol, speed, auc, track, Rs, names, args.tol)
    with open(os.path.join(outdir, "summary.json"), "w") as fh:
        json.dump({"tags": args.tags, "speed": speed, "auc": auc, "tracking": track}, fh, indent=2)
    print(f"\n[local] wrote figures + summary.json to {outdir}")


def _figures(outdir, pol, speed, auc, track, Rs, names, tol):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.dpi": 130, "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    COL = {"value": "#e8590c", "value-floor": "#f76707", "oracle": "#2f9e44",
           "uniform": "#868e96", "lprog-only": "#7048e8", "visits-only": "#1c7ed6",
           "error-only": "#c92a2a"}
    x = np.arange(len(pol))

    # 1) the headline pair: speed discriminates, AUC does not
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    axes[0].bar(x, [speed[p]["rounds_mean"] for p in pol],
                yerr=[speed[p]["rounds_sem"] for p in pol],
                color=[COL.get(p, "#555") for p in pol], capsize=3)
    axes[0].set_xticks(x); axes[0].set_xticklabels(pol, rotation=25, ha="right", fontsize=8.5)
    axes[0].set_ylabel(f"rounds to recover (within {tol:.0%} of epoch best)")
    axes[0].set_title("ADAPTATION SPEED — discriminates\n(directed collection buys a FASTER model)")
    axes[1].bar(x, [auc[p]["gap_closed"] for p in pol],
                color=[COL.get(p, "#555") for p in pol])
    axes[1].axhline(1.0, color="#2f9e44", ls="--", lw=1.2)
    axes[1].set_xticks(x); axes[1].set_xticklabels(pol, rotation=25, ha="right", fontsize=8.5)
    axes[1].set_ylabel("gap closed toward oracle (mean over rounds)")
    axes[1].set_title("ASYMPTOTIC COMPETENCE — nulls\n(a budget-rich loop equalizes every policy)")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "fig1_speed_vs_asymptote.png"), bbox_inches="tight")
    plt.close(fig)

    # 2) choice quality vs behavioral payoff
    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    for p in pol:
        if np.isnan(track[p]):
            continue
        ax.scatter(track[p], speed[p]["rounds_mean"], s=90, color=COL.get(p, "#555"), zorder=3)
        ax.annotate(p, (track[p], speed[p]["rounds_mean"]), textcoords="offset points",
                    xytext=(6, 4), fontsize=8.5)
    ax.set_xlabel("tracking: share of budget in the stale on-reach region")
    ax.set_ylabel("rounds to recover")
    ax.set_title("does choosing better actually pay off?")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "fig2_tracking_vs_speed.png"), bbox_inches="tight")
    plt.close(fig)

    # 3) mean ballistic trace over rounds, averaged across seeds
    fig, ax = plt.subplots(figsize=(9.6, 5.0))
    sched = {int(k): v for k, v in Rs[0]["schedule"].items()}
    for p in pol:
        M = np.array([[r["ballistic_cem"] for r in R["results"][p]] for R in Rs])
        m = M.mean(0)
        ax.plot(range(len(m)), m, "o-", ms=3.5, lw=2.0, color=COL.get(p, "#333"), label=p)
    for t, (j, _) in sched.items():
        ax.axvline(t - 0.5, color="k", ls=":", lw=1.0, alpha=0.5)
        ax.text(t - 0.4, ax.get_ylim()[1], names[j], rotation=90, va="top", fontsize=7, alpha=0.6)
    ax.set_xlabel("round (dotted = drift event)")
    ax.set_ylabel("ballistic goal-dist (mean over seeds)")
    ax.set_title("S2) online directed collection against a moving drift target")
    ax.legend(fontsize=8.5)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "fig3_rounds.png"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
