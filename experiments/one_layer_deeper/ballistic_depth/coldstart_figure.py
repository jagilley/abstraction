"""Functional re-enterability: can the operator be restarted from a clean state?

`on_manifold_cos` (README §2) says the base model's rolled state points almost orthogonally
to the encoder's representation of the same residue. That is a *cosine* — a geometric
proxy. This asks the behavioural question directly: encode the TRUE intermediate residue
`x_t` cold, as if it were a fresh problem, roll the remaining `T - t` steps, and score
exact-match against `x_T`.

The reference curve is what makes the readout sharp. Re-entering at `t` and rolling `T - t`
steps should, if the operator is a clean function of the encoder's states, score exactly
what the model scores on an ordinary depth-`(T - t)` problem — because that is what it *is*.
So plotting `coldstart(t, T)` against `exact@(T - t)` separates two failure modes that the
cosine cannot:

  - coldstart(t,T) tracks exact@(T-t)  -> the operator is sound on encoder states; the
    entire ballistic failure is DRIFT. The model cannot produce its own inputs.
  - coldstart(t,T) falls below it      -> the operator was never a function on the encoder's
    state space; it only works inside the private trajectory it generates.

Caveat baked into the data: `coldstart_heldout_x` is NOT a clean generalization number.
The residue `x_t` of a held-out base is usually itself a base the model trained on (squaring
maps into the 207-element QR subgroup) — measured at 84-93% overlap for every t >= 1 and
recorded per-run as `heldout_base_leak`. Seen-x is the comparison that carries the claim.

Usage:  python3 one_layer_deeper/ballistic_depth/coldstart_figure.py [--tag coldstart]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FIGURES = HERE / "figures" / "coldstart"

ARM_STYLE = {
    "base": ("#c1554a", "base (terminal CE only)"),
    "consist": ("#2e7d5b", "+ cycle (label-free)"),
}


def load(tag):
    runs = []
    for path in sorted((HERE / "results" / tag).glob("results_seed*.json")):
        runs.append(json.loads(path.read_text()))
    if not runs:
        raise SystemExit(f"no results under results/{tag}/")
    return runs


def agg(runs, arm, field):
    """-> {T: {t: (mean, std)}} across seeds."""
    per = {}
    for r in runs:
        blob = r["arms"][arm][field]
        for T, row in blob.items():
            for t, v in row.items():
                per.setdefault(int(T), {}).setdefault(int(t), []).append(float(v))
    return {
        T: {t: (float(np.mean(v)), float(np.std(v))) for t, v in sorted(row.items())}
        for T, row in sorted(per.items())
    }


def agg_exact(runs, arm):
    per = {}
    for r in runs:
        for d, v in r["arms"][arm]["exact_seen_x"].items():
            per.setdefault(int(d), []).append(float(v))
    return {d: float(np.mean(v)) for d, v in sorted(per.items())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="coldstart")
    args = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = load(args.tag)
    arms = [a for a in ("base", "consist") if a in runs[0]["arms"]]
    print(f"{args.tag}: {len(runs)} seeds, arms={arms}")
    leak = runs[0].get("heldout_base_leak", {})
    if leak:
        vals = [v for k, v in leak.items() if int(k) >= 1]
        print(f"  held-out base leak at t>=1: {min(vals):.2f}-{max(vals):.2f} "
              f"(so coldstart_heldout_x is not a clean generalization readout)")

    cs = {a: agg(runs, a, "coldstart_seen_x") for a in arms}
    ex = {a: agg_exact(runs, a) for a in arms}
    targets = sorted(cs[arms[0]].keys())

    # --- the number that carries the claim -------------------------------------------
    print("\n--- restart gain: cold start at t vs the full ballistic rollout (t=0) ---")
    for a in arms:
        print(f"  [{a}]")
        for T in targets:
            row = cs[a][T]
            full = row[0][0]
            best_t = max(row, key=lambda t: row[t][0])
            print(f"    T={T:<3d} full-rollout={full:.3f}   "
                  f"best restart t={best_t:<3d} -> {row[best_t][0]:.3f}   "
                  f"gain={row[best_t][0] - full:+.3f}")

    print("\n--- is a cold start equivalent to a fresh problem of depth T-t? ---")
    print("    (coldstart(t,T) vs exact@(T-t); ~0 residual => operator sound, failure is drift)")
    for a in arms:
        print(f"  [{a}]")
        for T in targets:
            resid = []
            for t, (m, _) in cs[a][T].items():
                k = T - t
                if k >= 1 and k in ex[a]:
                    resid.append(m - ex[a][k])
            if resid:
                print(f"    T={T:<3d} mean residual={np.mean(resid):+.3f}  "
                      f"max |residual|={np.max(np.abs(resid)):.3f}")

    # --- figure ------------------------------------------------------------------------
    FIGURES.mkdir(parents=True, exist_ok=True)
    n = len(arms)
    fig, axes = plt.subplots(1, n + 1, figsize=(6.0 * (n + 1), 4.8))
    shades = plt.cm.viridis(np.linspace(0.15, 0.85, len(targets)))

    for ax, a in zip(axes, arms):
        for c, T in zip(shades, targets):
            ts = sorted(cs[a][T])
            ms = [cs[a][T][t][0] for t in ts]
            es = [cs[a][T][t][1] for t in ts]
            ax.errorbar(ts, ms, yerr=es, marker="o", ms=4, lw=1.7, color=c,
                        capsize=2, label=f"T={T}", zorder=3)
            ref = [ex[a].get(T - t, np.nan) for t in ts]
            ax.plot(ts, ref, ls=":", lw=1.3, color=c, alpha=0.75, zorder=2)
        ax.set_title(f"{ARM_STYLE[a][1]}\nsolid = cold start at t, roll T−t   "
                     f"dotted = exact@(T−t)", fontsize=10)
        ax.set_xlabel("re-entry point t  (t=0 is the ordinary ballistic rollout)",
                      fontsize=9.5)
        ax.set_ylabel("exact match on x_T", fontsize=9.5)
        ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.25, zorder=0)
        ax.legend(fontsize=8, ncol=2)

    ax = axes[-1]
    for a in arms:
        col = ARM_STYLE[a][0]
        for T, ls in zip(targets, ("-", "--", "-.", ":", (0, (3, 1, 1, 1)))):
            row = cs[a][T]
            ts = sorted(row)
            gain = [row[t][0] - row[0][0] for t in ts]
            ax.plot(ts, gain, ls=ls, lw=1.6, color=col,
                    label=f"{a} T={T}", zorder=3)
    ax.axhline(0, color="k", lw=1, ls=":")
    ax.set_title("restart gain\ncold start at t minus full rollout", fontsize=10)
    ax.set_xlabel("re-entry point t", fontsize=9.5)
    ax.set_ylabel("Δ exact match", fontsize=9.5)
    ax.grid(alpha=0.25, zorder=0)
    ax.legend(fontsize=7.5, ncol=2)

    fig.suptitle(
        "Functional re-enterability: what a clean restart buys, and whether the operator "
        "is a sound function of its own encoder's states",
        fontsize=12.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    out = FIGURES / "fig_coldstart.png"
    fig.savefig(out, dpi=160)
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
