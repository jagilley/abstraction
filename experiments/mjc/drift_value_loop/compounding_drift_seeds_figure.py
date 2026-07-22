"""Local (non-Modal) post-processing: aggregate the compounding_drift Phase-1 seeds
into the headline figures — the meta-layer COMPOUNDING signature with mean±sem bands.

Reads figures/compounding_drift_<tag>/results.json for the seed tags below, recomputes
transitions-to-recover from the stored per-drift R²-vs-N curves at a stated threshold
(so the threshold is transparent and tunable here, not baked into the Modal run), and
emits:
  * fig_fewshot   — held-out pusher/vel R² at a FEW-SHOT budget (N=20) per drift, mean±sem.
                    THRESHOLD-FREE: factored climbs & locks (compounding); mono_replay
                    collapses (pooling under conflict); scratch flat; offline_meta ceiling.
  * fig_ttr       — transitions-to-recover per drift, mean±sem (the compounding curve).
  * fig_cumulative— cumulative transitions (sublinear=compounding vs linear/censored).

Run:  python3 mjc/drift_value_loop/compounding_drift_seeds_figure.py
      python3 mjc/drift_value_loop/compounding_drift_seeds_figure.py --tags full_v1 full_s1 full_s2 --frac 0.8 --fewshot 20
"""

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
COL = {"factored": "#3d6fd1", "mono_replay": "#d1603d", "mono_noreplay": "#e08a3c",
       "offline_meta": "#2f9e44", "scratch": "#888"}
LAB = {"factored": "factored f(s,u,z) — task-conditioned memory",
       "mono_replay": "monolithic + replay (pooled)",
       "mono_noreplay": "monolithic, no replay (forgetting)",
       "offline_meta": "offline-meta ceiling", "scratch": "scratch / drift (no memory)"}
ORDER = ["factored", "offline_meta", "scratch", "mono_noreplay", "mono_replay"]


def load(tags):
    runs = []
    for t in tags:
        p = os.path.join(HERE, "figures", f"compounding_drift_{t}", "results.json")
        if os.path.exists(p):
            runs.append(json.load(open(p)))
        else:
            print(f"[warn] missing {p} — skipping seed")
    if not runs:
        raise SystemExit("no seed results found")
    return runs


def ttr_from_curves(R, frac):
    """recompute transitions-to-recover per arm per drift at threshold frac*oracle."""
    b = R["budgets"]; big = max(b); thr = frac * R["oracle_vel_r2"]
    out = {}
    for a in R["curves"]:
        seq = []
        for c in R["curves"][a]:
            rN = next((N for N in b if N > 0 and c[str(N)] >= thr), big * 1.5)
            seq.append(rN)
        out[a] = seq
    return out


def fewshot_from_curves(R, N):
    """held-out R² at a fixed few-shot budget N per arm per drift."""
    return {a: [c[str(N)] for c in R["curves"][a]] for a in R["curves"]}


def agg(list_of_seqs):
    """stack ragged-equal seqs -> (mean, sem) over seeds."""
    M = np.array(list_of_seqs, float)          # (seeds, T)
    mean = M.mean(0)
    sem = M.std(0) / max(1, np.sqrt(M.shape[0]))
    return mean, sem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["full_v1", "full_s1", "full_s2"])
    ap.add_argument("--frac", type=float, default=0.80)
    ap.add_argument("--fewshot", type=int, default=20)
    args = ap.parse_args()

    runs = load(args.tags)
    arms = [a for a in ORDER if a in runs[0]["curves"]]
    T = len(runs[0]["phi_seq"])
    x = np.arange(T)
    big = max(runs[0]["budgets"]) * 1.5
    plt.rcParams.update({"figure.dpi": 130, "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    outdir = os.path.join(HERE, "figures", "compounding_drift_seeds")
    os.makedirs(outdir, exist_ok=True)
    nseed = len(runs)

    ttr_seeds = {a: [] for a in arms}
    few_seeds = {a: [] for a in arms}
    cum_seeds = {a: [] for a in arms}
    for R in runs:
        t = ttr_from_curves(R, args.frac)
        f = fewshot_from_curves(R, args.fewshot)
        for a in arms:
            ttr_seeds[a].append(t[a]); few_seeds[a].append(f[a])
            cum_seeds[a].append(np.cumsum(t[a]))

    # ---- fig_fewshot: threshold-free compounding (R² @ few-shot N) ---- #
    fig, ax = plt.subplots(figsize=(9.6, 5.2))
    for a in arms:
        m, s = agg(few_seeds[a])
        ax.plot(x, m, "o-", color=COL[a], lw=2.1, ms=4.5, label=LAB[a])
        ax.fill_between(x, m - s, m + s, color=COL[a], alpha=0.15)
    ax.axhline(args.frac * runs[0]["oracle_vel_r2"], color="#555", ls="--", lw=1.0,
               label=f"recover threshold ({args.frac:g}×oracle)")
    ax.set_ylim(-0.7, 1.03)
    ax.set_xlabel("drift index (novel φ each drift)")
    ax.set_ylabel(f"held-out pusher-vel R² after N={args.fewshot} transitions")
    ax.set_title(f"COMPOUNDING (threshold-free): few-shot adaptation quality over the drift sequence\n"
                 f"factored climbs & locks; pooled memory COLLAPSES under conflict "
                 f"({nseed} seeds, mean±sem)", fontsize=10)
    ax.legend(fontsize=8.5, loc="lower left")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "fig_fewshot.png"), bbox_inches="tight")
    plt.close(fig)

    # ---- fig_ttr: transitions-to-recover per drift ---- #
    fig, ax = plt.subplots(figsize=(9.6, 5.2))
    for a in arms:
        m, s = agg(ttr_seeds[a])
        ax.plot(x, m, "o-", color=COL[a], lw=2.1, ms=4.5, label=LAB[a])
        ax.fill_between(x, m - s, m + s, color=COL[a], alpha=0.15)
    ax.axhline(big, color="#bbb", ls=":", lw=1.0)
    ax.text(0.02, big, f"censored (>{max(runs[0]['budgets'])})", color="#999", fontsize=8, va="bottom")
    ax.set_xlabel("drift index"); ax.set_ylabel("transitions-to-recover (↓ = cheaper)")
    ax.set_title(f"Compounding: per-drift adaptation cost ({nseed} seeds, mean±sem, "
                 f"thr={args.frac:g}×oracle)", fontsize=10)
    ax.legend(fontsize=8.5, loc="center right")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "fig_ttr.png"), bbox_inches="tight")
    plt.close(fig)

    # ---- fig_cumulative ---- #
    fig, ax = plt.subplots(figsize=(9.6, 5.2))
    for a in arms:
        m, s = agg(cum_seeds[a])
        ax.plot(x, m, "-", color=COL[a], lw=2.3, label=LAB[a])
        ax.fill_between(x, m - s, m + s, color=COL[a], alpha=0.15)
    ax.set_xlabel("drifts survived"); ax.set_ylabel("cumulative transitions-to-recover")
    ax.set_title(f"Cumulative adaptation cost: pooled memory is a LIABILITY under conflict\n"
                 f"(worse than memoryless) — {nseed} seeds, mean±sem", fontsize=10)
    ax.legend(fontsize=8.5, loc="upper left")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "fig_cumulative.png"), bbox_inches="tight")
    plt.close(fig)

    # ---- console summary ---- #
    print(f"\n=== compounding seeds summary ({nseed} seeds, thr={args.frac:g}×oracle, "
          f"few-shot N={args.fewshot}) ===")
    for a in arms:
        tm, _ = agg(ttr_seeds[a]); fm, _ = agg(few_seeds[a])
        q = max(1, T // 4)
        print(f"  {a:14s} TTR early={tm[:q].mean():6.1f} late={tm[-q:].mean():6.1f} "
              f"Δ={tm[-q:].mean()-tm[:q].mean():+7.1f} | R²@{args.fewshot} early={fm[:q].mean():+.2f} "
              f"late={fm[-q:].mean():+.2f} | cum={np.cumsum(tm)[-1]:.0f}")
    print(f"[saved] {outdir}/fig_fewshot.png, fig_ttr.png, fig_cumulative.png")


if __name__ == "__main__":
    main()
