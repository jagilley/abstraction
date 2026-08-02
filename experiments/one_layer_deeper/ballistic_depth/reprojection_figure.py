"""Figure + tables for the test-time re-projection cut (README §9).

Reads `results/reproj_coldstart/results_seed*.json` and reports the three things the cut
turns on: how much a *deployment-time* snap buys each arm, the gap between snapping to the
model's own decode and snapping to ground truth, and why the projection period has an
optimum for the ungrounded arm but not for the grounded one.

The quantitative account it checks: re-projection turns one long rollout into a chain of
independent restarts, so accuracy should go as `p ** ceil(T / k)` where `p` is the per-restart
reliability — the 0.871 plateau the oracle arm sits at, which is the same number §8's cold-start
probe plateaus at. Training-time closure works by driving `p` to 1.0, which is what makes the
chain flat in depth instead of exponentially decaying.

Usage:  python3 one_layer_deeper/ballistic_depth/reprojection_figure.py
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FIGURES = HERE / "figures" / "reprojection"
PERIODS = (1, 2, 3, 4, 5, 6, 8, 10, 15)
ARMS = {"base": ("#c1554a", "base (ungrounded)"),
        "consist": ("#2e7d5b", "+ cycle (grounded)")}


def load(tag="reproj_coldstart"):
    runs = [json.loads(Path(f).read_text())
            for f in sorted(glob.glob(str(HERE / "results" / tag / "results_seed*.json")))]
    if not runs:
        raise SystemExit(f"no results under results/{tag}/")
    return runs


def mean(runs, arm, path, T):
    vals = []
    for r in runs:
        node = r["arms"][arm]
        for p in path:
            node = node[p]
        vals.append(float(node[str(T)]))
    return float(np.mean(vals))


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = load()
    Ts = [10, 20, 30, 40, 60]
    print(f"{len(runs)} seeds\n")

    # --- per-restart reliability, read off the oracle plateau -------------------------
    p_hat = float(np.mean([mean(runs, "base", ["oracle", f"a1.0_k{k}"], T)
                           for k in (1, 2, 3, 4, 5, 6, 8) for T in (20, 30, 40, 60)]))
    print(f"base per-restart reliability p (oracle plateau) = {p_hat:.3f}")
    print("chain model  accuracy ~ p ** ceil(T/k):")
    for k in (5, 8, 10):
        for T in (20, 40, 60):
            seg = int(np.ceil(T / k))
            print(f"   k={k:<3d} T={T:<3d} segments={seg:<2d} "
                  f"predicted={p_hat ** seg:.3f}  actual="
                  f"{mean(runs, 'base', ['self', f'a1.0_k{k}'], T):.3f}")

    print("\nbest deployable setting (self, alpha=1.0):")
    for arm in ARMS:
        base_line = {T: mean(runs, arm, ["baseline"], T) for T in Ts}
        best = max(PERIODS, key=lambda k: mean(runs, arm, ["self", f"a1.0_k{k}"], 60))
        got = {T: mean(runs, arm, ["self", f"a1.0_k{best}"], T) for T in Ts}
        print(f"  [{arm}] k={best}")
        print("     no re-projection " + " ".join(f"T{T}:{v:.3f}" for T, v in base_line.items()))
        print("     re-projected     " + " ".join(f"T{T}:{v:.3f}" for T, v in got.items()))

    # --- figure ------------------------------------------------------------------------
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.9))

    for ax, arm in zip(axes[:2], ARMS):
        col, label = ARMS[arm]
        ax.plot(Ts, [mean(runs, arm, ["baseline"], T) for T in Ts], marker="o", lw=2,
                color=col, label="no re-projection", zorder=4)
        ax.plot(Ts, [mean(runs, arm, ["self", "a1.0_k8"], T) for T in Ts], marker="s", lw=2,
                ls="--", color="#e08a1e", label="re-project k=8 (own decode)", zorder=4)
        ax.plot(Ts, [mean(runs, arm, ["oracle", "a1.0_k8"], T) for T in Ts], marker="^", lw=1.7,
                ls=":", color="#2f6f9f", label="re-project k=8 (oracle)", zorder=3)
        if arm == "base":
            ax.plot(Ts, [p_hat ** int(np.ceil(T / 8)) for T in Ts], lw=1.3, color="k",
                    ls=(0, (2, 2)), label=f"chain model  {p_hat:.2f}^⌈T/8⌉", zorder=2)
        ax.axhline(0.5, color="#999", lw=1, ls=":", zorder=1)
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("depth T", fontsize=9.5)
        ax.set_ylabel("exact match", fontsize=9.5)
        ax.set_ylim(-0.04, 1.05)
        ax.grid(alpha=0.25, zorder=0)
        ax.legend(fontsize=8, loc="lower left")

    ax = axes[2]
    for arm, style in (("base", "-"), ("consist", "--")):
        col = ARMS[arm][0]
        ax.plot(PERIODS, [mean(runs, arm, ["self", f"a1.0_k{k}"], 60) for k in PERIODS],
                marker="o", lw=2, ls=style, color=col, label=f"{arm} — own decode", zorder=4)
        ax.plot(PERIODS, [mean(runs, arm, ["oracle", f"a1.0_k{k}"], 60) for k in PERIODS],
                marker="^", ms=4, lw=1.4, ls=":", color=col, alpha=0.75,
                label=f"{arm} — oracle", zorder=3)
    ax.axvline(13, color="#999", lw=1.2, ls="--", zorder=1)
    ax.text(13.3, 0.55, "base horizon\n(T≈13)", fontsize=8, color="#666")
    ax.set_title("why the period has an optimum\n"
                 "few restarts is good; longer than the horizon is fatal", fontsize=10.5)
    ax.set_xlabel("re-projection period k", fontsize=9.5)
    ax.set_ylabel("exact match @ T=60", fontsize=9.5)
    ax.set_ylim(-0.04, 1.05)
    ax.grid(alpha=0.25, zorder=0)
    ax.legend(fontsize=8, loc="center left")

    fig.suptitle(
        "Test-time re-projection: snapping the rollout back onto the encoder manifold, with "
        "no retraining, turns one rollout into a chain of restarts",
        fontsize=12.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    out = FIGURES / "fig_reprojection.png"
    fig.savefig(out, dpi=160)
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
