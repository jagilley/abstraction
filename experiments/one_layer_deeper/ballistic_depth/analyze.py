"""Aggregate `ballistic_depth` runs across seeds and draw the mechanism figures.

Fetch results first (they are committed to the volume by the remote function):

    MODAL_PROFILE=chromatic modal volume get one-layer-deeper-data \
        /ballistic_depth/<tag> one_layer_deeper/ballistic_depth/results/ --force

Then:

    python3 one_layer_deeper/ballistic_depth/analyze.py --tag cut1

Reads every `results_seed*.json`, reports mean +/- sd across seeds, and writes
`figures/<tag>/`. The three readouts that carry the claim:

  - exact@T split at the trained-depth boundary (the score),
  - amplification per step (does the operator expand its own error?),
  - veridicality / on-manifold cosine vs t (does the rollout stay on the residue set?).
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

ARM_ORDER = ["feedforward", "base", "consist", "quant", "quant_consist"]
ARM_LABEL = {
    "feedforward": "feedforward (untied stack)",
    "base": "base (recurrent, terminal CE only)",
    "consist": "+ consistency",
    "quant": "+ quantize",
    "quant_consist": "+ quantize + consistency",
}


def _ms(values):
    """mean, sd (population sd for n<2 -> 0.0)."""
    vals = [v for v in values if v is not None]
    if not vals:
        return float("nan"), 0.0
    if len(vals) == 1:
        return vals[0], 0.0
    return statistics.mean(vals), statistics.stdev(vals)


def _intkeys(d):
    return {int(k): v for k, v in d.items()}


def load(tag: str, results_dir: Path):
    root = results_dir / tag if (results_dir / tag).is_dir() else results_dir
    runs = sorted(root.rglob("results_seed*.json"))
    if not runs:
        raise SystemExit(f"no results_seed*.json under {root}")
    loaded = [json.loads(p.read_text()) for p in runs]
    print(f"[loaded] {len(loaded)} seed(s) from {root}")
    return loaded


def per_step_amplification(amp_by_t, t_ref):
    """Geometric per-step factor (ratio_t)^(1/t) — the Lyapunov-style readout."""
    r = amp_by_t.get(t_ref)
    if r is None or r <= 0:
        return float("nan")
    return r ** (1.0 / t_ref)


def report(runs, tag, out_dir: Path):
    task = runs[0]["task"]
    train_depths = set(task["train_depths"])
    eval_depths = task["eval_depths"]
    ood = [d for d in eval_depths if d not in train_depths]
    arms = [a for a in ARM_ORDER if a in runs[0]["arms"]]

    print(f"\n=== task ===\n{json.dumps(task, indent=1)}")

    # ---- headline table -----------------------------------------------------------
    print(f"\n=== exact match, mean +/- sd over {len(runs)} seed(s) ===")
    deepest = max(eval_depths)
    print(f"{'arm':>34} {'ID (T<=6)':>16} {'OOD (T>6)':>16} {'T=7':>14} "
          f"{'T=' + str(deepest):>14}")
    rows = {}
    for arm in arms:
        ids, oods, t7, t20 = [], [], [], []
        for r in runs:
            ex = _intkeys(r["arms"][arm]["exact_seen_x"])
            ids.append(statistics.mean([ex[d] for d in sorted(train_depths)]))
            oods.append(statistics.mean([ex[d] for d in ood]))
            t7.append(ex.get(7))
            t20.append(ex.get(max(eval_depths)))
        rows[arm] = dict(id=_ms(ids), ood=_ms(oods), t7=_ms(t7), t20=_ms(t20))
        m = rows[arm]
        print(
            f"{ARM_LABEL[arm]:>34} "
            f"{m['id'][0]:>9.3f}+/-{m['id'][1]:<5.3f} "
            f"{m['ood'][0]:>9.3f}+/-{m['ood'][1]:<5.3f} "
            f"{m['t7'][0]:>7.3f}+/-{m['t7'][1]:<5.3f} "
            f"{m['t20'][0]:>7.3f}+/-{m['t20'][1]:<5.3f}"
        )

    # ---- the 2x2 ------------------------------------------------------------------
    if all(a in rows for a in ("base", "quant", "consist", "quant_consist")):
        print("\n=== the 2x2 on OOD depth (mean exact @ T>6) ===")
        print(f"{'':>14} {'consist off':>14} {'consist on':>14}")
        print(f"{'quant off':>14} {rows['base']['ood'][0]:>14.3f} "
              f"{rows['consist']['ood'][0]:>14.3f}")
        print(f"{'quant on':>14} {rows['quant']['ood'][0]:>14.3f} "
              f"{rows['quant_consist']['ood'][0]:>14.3f}")
        q = rows["quant"]["ood"][0] + rows["quant_consist"]["ood"][0]
        nq = rows["base"]["ood"][0] + rows["consist"]["ood"][0]
        c = rows["consist"]["ood"][0] + rows["quant_consist"]["ood"][0]
        nc = rows["base"]["ood"][0] + rows["quant"]["ood"][0]
        inter = (
            rows["quant_consist"]["ood"][0] - rows["quant"]["ood"][0]
            - rows["consist"]["ood"][0] + rows["base"]["ood"][0]
        )
        print(f"\n  main effect quantize   : {(q - nq) / 2:+.3f}")
        print(f"  main effect consistency: {(c - nc) / 2:+.3f}")
        print(f"  interaction            : {inter:+.3f}")

    # ---- amplification ------------------------------------------------------------
    rec = [a for a in arms if a != "feedforward"]
    print("\n=== per-step error amplification (geometric, from t=8) ===")
    print("  claim: continuous rollout must expand (>1); re-attraction must absorb (<1)")
    print(f"{'arm':>34} " + " ".join(f"{'eps=' + str(e):>12}" for e in (0.01, 0.1, 0.3, 1.0)))
    amp_rows = {}
    for arm in rec:
        cells = []
        for eps in ("0.01", "0.1", "0.3", "1.0"):
            per_seed = []
            for r in runs:
                amp = r["arms"][arm].get("amplification", {})
                if eps in amp:
                    per_seed.append(per_step_amplification(_intkeys(amp[eps]), 8))
            cells.append(_ms(per_seed))
        amp_rows[arm] = cells
        print(f"{ARM_LABEL[arm]:>34} "
              + " ".join(f"{m:>7.3f}+/-{s:<4.3f}" for m, s in cells))

    # ---- veridicality / on-manifold ------------------------------------------------
    print("\n=== veridicality of the intermediate state (terminal decoder, zero-shot) ===")
    print(f"{'arm':>34} " + " ".join(f"{'t=' + str(t):>8}" for t in (1, 3, 6, 10, 20)))
    for arm in rec:
        cells = []
        for t in (1, 3, 6, 10, 20):
            cells.append(_ms([_intkeys(r["arms"][arm]["veridicality"]).get(t) for r in runs]))
        print(f"{ARM_LABEL[arm]:>34} " + " ".join(f"{m:>8.3f}" for m, _ in cells))

    print("\n=== on-manifold cosine, cos(h_t rolled, Enc(true x_t)) ===")
    print(f"{'arm':>34} " + " ".join(f"{'t=' + str(t):>8}" for t in (1, 3, 6, 10, 20)))
    for arm in rec:
        cells = []
        for t in (1, 3, 6, 10, 20):
            cells.append(
                _ms([_intkeys(r["arms"][arm]["on_manifold_cos"]).get(t) for r in runs])
            )
        print(f"{ARM_LABEL[arm]:>34} " + " ".join(f"{m:>8.3f}" for m, _ in cells))

    for arm in rec:
        cu = runs[0]["arms"][arm].get("codes_used")
        if cu:
            cu = _intkeys(cu)
            print(f"\n[{arm}] distinct codes occupied at t=1/6/20: "
                  f"{cu.get(1)}/{cu.get(6)}/{cu.get(max(cu))} "
                  f"(true reachable set = {task['reachable_states_depth_ge_1']})")

    _figures(runs, arms, rec, task, out_dir)
    return rows


def _figures(runs, arms, rec, task, out_dir: Path):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n[figures] matplotlib unavailable; skipped")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    depths = task["eval_depths"]
    cut = max(task["train_depths"])
    colors = {
        "feedforward": "#9aa0a6",
        "base": "#c1554a",
        "consist": "#d99a2b",
        "quant": "#3d7ea6",
        "quant_consist": "#2e7d5b",
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    ax = axes[0]
    for arm in arms:
        ys = [
            _ms([_intkeys(r["arms"][arm]["exact_seen_x"])[d] for r in runs])
            for d in depths
        ]
        m = [y[0] for y in ys]
        s = [y[1] for y in ys]
        ax.plot(depths, m, marker="o", ms=3, color=colors[arm], label=ARM_LABEL[arm])
        ax.fill_between(
            depths,
            [a - b for a, b in zip(m, s)],
            [a + b for a, b in zip(m, s)],
            color=colors[arm],
            alpha=0.15,
            linewidth=0,
        )
    ax.axvline(cut + 0.5, color="k", ls=":", lw=1)
    ax.text(cut + 0.7, 0.95, "trained depths $\\leq$ %d" % cut, fontsize=8, va="top")
    ax.set_xlabel("depth $T$")
    ax.set_ylabel("exact match")
    ax.set_title("Depth extrapolation")
    ax.set_ylim(-0.03, 1.03)
    ax.legend(fontsize=7, loc="upper right")

    ax = axes[1]
    for arm in rec:
        amp = runs[0]["arms"][arm].get("amplification", {})
        if "0.01" not in amp:
            continue
        ts = sorted(_intkeys(amp["0.01"]))
        ys = [
            _ms([_intkeys(r["arms"][arm]["amplification"]["0.01"])[t] for r in runs])
            for t in ts
        ]
        ax.plot(ts, [y[0] for y in ys], marker="o", ms=3, color=colors[arm],
                label=ARM_LABEL[arm])
    ax.axhline(1.0, color="k", ls=":", lw=1)
    ax.set_yscale("log")
    ax.set_xlabel("rollout step $t$")
    ax.set_ylabel(r"$\|\delta h_t\| / \|\delta h_0\|$")
    ax.set_title("Error amplification ($\\epsilon = 0.01$)")
    ax.legend(fontsize=7)

    ax = axes[2]
    for arm in rec:
        ts = sorted(_intkeys(runs[0]["arms"][arm]["veridicality"]))
        ys = [
            _ms([_intkeys(r["arms"][arm]["veridicality"])[t] for r in runs]) for t in ts
        ]
        ax.plot(ts, [y[0] for y in ys], marker="o", ms=3, color=colors[arm],
                label=ARM_LABEL[arm])
    ax.axvline(cut + 0.5, color="k", ls=":", lw=1)
    ax.set_xlabel("rollout step $t$")
    ax.set_ylabel("decoded $\\hat{x}_t = x_t$")
    ax.set_title("Veridicality of the intermediate state")
    ax.set_ylim(-0.03, 1.03)
    ax.legend(fontsize=7)

    fig.suptitle(
        f"Ballistic depth on $x^{{2^T}}$ mod {task['modulus']} "
        f"({len(runs)} seed{'s' if len(runs) > 1 else ''})",
        fontsize=11,
    )
    fig.tight_layout()
    path = out_dir / "fig_ballistic_depth.png"
    fig.savefig(path, dpi=160)
    print(f"\n[figure] {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="cut1")
    ap.add_argument(
        "--results-dir", default=str(Path(__file__).parent / "results")
    )
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    runs = load(args.tag, Path(args.results_dir))
    out = Path(args.out) if args.out else Path(__file__).parent / "figures" / args.tag
    report(runs, args.tag, out)


if __name__ == "__main__":
    main()
