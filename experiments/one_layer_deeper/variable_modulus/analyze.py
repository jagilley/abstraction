"""Aggregate `variable_modulus` runs across seeds and draw the arity figures.

Fetch results first (the remote function commits them to the volume):

    MODAL_PROFILE=chromatic modal volume get one-layer-deeper-data \
        /variable_modulus/<tag> one_layer_deeper/variable_modulus/results/ --force

Then:

    python3 one_layer_deeper/variable_modulus/analyze.py --tag <tag>

Four readouts carry the cut:

  - exact@T on three OOD axes at once — held-out depth, held-out x, **held-out modulus**;
  - the composition horizon per arm, which is where `fold` vs `cond` (P2) is read;
  - rule retention (`rulesep@t` against its own t=0, and `inrange@t`), which is P4 — does
    the rolled state still know which world it is in;
  - `exact_rule_swapped`, the causal check that `cond`'s advantage is the *rule* and not
    extra per-step capacity.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

ARM_ORDER = ["blind", "fold", "cond", "fold_cyc", "cond_cyc"]
ARM_LABEL = {
    "blind": "blind (no N in prompt)",
    "fold": "fold (N carried in the state)",
    "cond": "cond (rule re-injected each step)",
    "fold_cyc": "fold + cycle",
    "cond_cyc": "cond + cycle",
}
COLORS = {
    "blind": "#9aa0a6",
    "fold": "#c1554a",
    "cond": "#3d7ea6",
    "fold_cyc": "#d99a2b",
    "cond_cyc": "#2e7d5b",
}
POOLS = ["seen_n_seen_x", "seen_n_heldout_x", "heldout_n"]
POOL_LABEL = {
    "seen_n_seen_x": "seen N, seen x",
    "seen_n_heldout_x": "seen N, held-out x",
    "heldout_n": "held-out N",
}


def _ms(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return float("nan"), 0.0
    if len(vals) == 1:
        return vals[0], 0.0
    return statistics.mean(vals), statistics.stdev(vals)


def _ik(d):
    return {int(k): v for k, v in d.items()}


def load(tag: str, results_dir: Path):
    root = results_dir / tag if (results_dir / tag).is_dir() else results_dir
    runs = sorted(root.rglob("results_seed*.json"))
    if not runs:
        raise SystemExit(f"no results_seed*.json under {root}")
    loaded = [json.loads(p.read_text()) for p in runs]
    print(f"[loaded] {len(loaded)} seed(s) from {root}")
    return loaded


def horizon(curve: dict, thresh=0.5):
    """First depth whose exact match falls below `thresh`. Right-censored at max+1."""
    for d in sorted(curve):
        if curve[d] < thresh:
            return d
    return max(curve) + 1


def report(runs, tag, out_dir: Path):
    fam = runs[0]["family"]
    cfg = runs[0]["config"]
    train_depths = set(int(v) for v in str(cfg["train_depths"]).split(","))
    cut = max(train_depths)
    depths = sorted(_ik(runs[0]["arms"][list(runs[0]["arms"])[0]]["exact_seen_n_seen_x"]))
    ood = [d for d in depths if d not in train_depths]
    arms = [a for a in ARM_ORDER if a in runs[0]["arms"]]

    slim = {k: v for k, v in fam.items() if not k.endswith("_moduli") or k.startswith("n_")}
    print(f"\n=== family ===\n{json.dumps(slim, indent=1)}")
    print(
        f"[knobs] cycle w={cfg['consist_cycle']} re-entry w={cfg['consist_reentry']} "
        f"steps={cfg['steps']} train T={sorted(train_depths)} eval T<={max(depths)}"
    )

    # ---- P1/P2/P5: the three OOD axes, at trained depth and past it -----------------
    print(f"\n=== exact match, mean +/- sd over {len(runs)} seed(s) ===")
    hdr = f"{'arm':>34}"
    for p in POOLS:
        hdr += f" | {POOL_LABEL[p]:>26}"
    print(hdr)
    print(f"{'':>34}" + " | " + " | ".join(f"{'ID (T<=%d)' % cut:>12}{'OOD':>14}" for _ in POOLS))
    rows = {}
    for arm in arms:
        rows[arm] = {}
        line = f"{ARM_LABEL[arm]:>34}"
        for p in POOLS:
            ids, oods = [], []
            for r in runs:
                ex = _ik(r["arms"][arm][f"exact_{p}"])
                ids.append(statistics.mean([ex[d] for d in sorted(train_depths)]))
                oods.append(statistics.mean([ex[d] for d in ood]))
            rows[arm][p] = dict(id=_ms(ids), ood=_ms(oods))
            line += f" | {rows[arm][p]['id'][0]:>12.3f}{rows[arm][p]['ood'][0]:>14.3f}"
        print(line)

    # ---- P2: the composition horizon (the headline contrast) -----------------------
    print(f"\n=== composition horizon (first T below 50% exact, seen N / seen x) ===")
    print(f"{'arm':>34} {'per seed':>22} {'median':>8}  {'T=%d' % cut:>7} {'T=%d' % max(depths):>7}")
    hz = {}
    for arm in arms:
        hs = [horizon(_ik(r["arms"][arm]["exact_seen_n_seen_x"])) for r in runs]
        hz[arm] = hs
        curve = [_ms([_ik(r["arms"][arm]["exact_seen_n_seen_x"])[d] for r in runs]) for d in depths]
        at_cut = curve[depths.index(cut)][0]
        at_max = curve[-1][0]
        cen = "+" if max(hs) > max(depths) else ""
        print(
            f"{ARM_LABEL[arm]:>34} {str(hs):>22} "
            f"{statistics.median(hs):>7.1f}{cen} {at_cut:>7.3f} {at_max:>7.3f}"
        )

    if "fold" in hz and "cond" in hz:
        print(
            f"\n  P2: cond horizon {statistics.median(hz['cond']):.1f} vs fold "
            f"{statistics.median(hz['fold']):.1f}"
        )
    if all(a in rows for a in ("fold", "cond", "fold_cyc", "cond_cyc")):
        print("\n=== P3: does cycle add to conditioning? (OOD depth, seen N / seen x) ===")
        print(f"{'':>14} {'cycle off':>12} {'cycle on':>12} {'gain':>10}")
        for base, cyc in (("fold", "fold_cyc"), ("cond", "cond_cyc")):
            a, b = rows[base]["seen_n_seen_x"]["ood"][0], rows[cyc]["seen_n_seen_x"]["ood"][0]
            print(f"{base:>14} {a:>12.3f} {b:>12.3f} {b - a:>+10.3f}")

    # ---- the causal rule check -----------------------------------------------------
    swapped = [a for a in arms if "exact_rule_swapped" in runs[0]["arms"][a]]
    if swapped:
        print("\n=== causal: roll with the WRONG modulus's rule vector ===")
        print(f"{'arm':>34} {'true rule (ID)':>16} {'swapped (ID)':>16} {'drop':>10}")
        for arm in swapped:
            t = _ms([
                statistics.mean([_ik(r["arms"][arm]["exact_seen_n_seen_x"])[d] for d in sorted(train_depths)])
                for r in runs
            ])
            s = _ms([
                statistics.mean([_ik(r["arms"][arm]["exact_rule_swapped"])[d] for d in sorted(train_depths)])
                for r in runs
            ])
            print(f"{ARM_LABEL[arm]:>34} {t[0]:>16.3f} {s[0]:>16.3f} {t[0] - s[0]:>+10.3f}")

    # ---- P4: rule retention and state closure --------------------------------------
    show = [t for t in (1, 2, 4, 6, 10, 20, 30, 40) if t <= max(depths)]
    for key, title in (
        ("rule_separation", "rule separation cos(h_t(N), h_t(N')) — LOWER is better"),
        ("inrange", "decoded x_t < N (does the state know its own world)"),
        ("on_manifold_cos", "state closure cos(h_t, Enc(N, x_t))"),
        ("veridicality", "veridicality (decoded x_t == true x_t)"),
    ):
        print(f"\n=== {title} ===")
        print(f"{'arm':>34} " + " ".join(f"{'t=%d' % t:>8}" for t in show))
        for arm in arms:
            if key not in runs[0]["arms"][arm]:
                continue
            vals = [_ms([_ik(r["arms"][arm][key])[t] for r in runs])[0] for t in show]
            extra = ""
            if key == "rule_separation":
                extra = f"  (t0={_ms([r['arms'][arm]['rule_separation_t0'] for r in runs])[0]:.3f})"
            print(f"{ARM_LABEL[arm]:>34} " + " ".join(f"{v:>8.3f}" for v in vals) + extra)

    # ---- cold start: is the operator sound, or is it the state that fails? ----------
    print("\n=== cold start (re-enter at the true x_t, roll T-t) ===")
    for arm in arms:
        cs = runs[0]["arms"][arm].get("coldstart_seen_n_seen_x")
        if not cs:
            continue
        for T in sorted(_ik(cs)):
            grid = sorted(_ik(_ik(cs)[T]))
            vals = [
                _ms([_ik(_ik(r["arms"][arm]["coldstart_seen_n_seen_x"])[T])[t] for r in runs])[0]
                for t in grid
            ]
            print(
                f"{ARM_LABEL[arm]:>34} T={T:<3} "
                + " ".join(f"t{t}:{v:.3f}" for t, v in zip(grid, vals))
            )

    _figures(runs, arms, fam, cfg, depths, cut, out_dir)


def _figures(runs, arms, fam, cfg, depths, cut, out_dir: Path):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n[figures] matplotlib unavailable; skipped")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.6))

    def curve(arm, key, xs):
        return [_ms([_ik(r["arms"][arm][key])[x] for r in runs]) for x in xs]

    # (a) depth extrapolation, with the rule-swap control dashed
    ax = axes[0][0]
    for arm in arms:
        ys = curve(arm, "exact_seen_n_seen_x", depths)
        m, s = [y[0] for y in ys], [y[1] for y in ys]
        ax.plot(depths, m, marker="o", ms=3, color=COLORS[arm], label=ARM_LABEL[arm])
        ax.fill_between(depths, [a - b for a, b in zip(m, s)], [a + b for a, b in zip(m, s)],
                        color=COLORS[arm], alpha=0.15, linewidth=0)
        if "exact_rule_swapped" in runs[0]["arms"][arm]:
            sw = [y[0] for y in curve(arm, "exact_rule_swapped", depths)]
            ax.plot(depths, sw, ls="--", lw=1, color=COLORS[arm], alpha=0.8)
    ax.axvline(cut + 0.5, color="k", ls=":", lw=1)
    ax.axhline(0.5, color="k", ls="-", lw=0.5, alpha=0.3)
    ax.text(cut + 0.7, 0.97, f"trained $T \\leq {cut}$", fontsize=8, va="top")
    ax.set_xlabel("depth $T$")
    ax.set_ylabel("exact match")
    ax.set_title("Depth extrapolation (seen $N$, seen $x$)\ndashed = rolled with the WRONG rule")
    ax.set_ylim(-0.03, 1.03)
    ax.legend(fontsize=7, loc="upper right")

    # (b) the three OOD axes at trained depth
    ax = axes[0][1]
    width = 0.8 / max(1, len(arms))
    for i, arm in enumerate(arms):
        vals = []
        for p in POOLS:
            ys = curve(arm, f"exact_{p}", [d for d in depths if d <= cut])
            vals.append(statistics.mean([y[0] for y in ys]))
        ax.bar([j + i * width for j in range(len(POOLS))], vals, width,
               color=COLORS[arm], label=ARM_LABEL[arm])
    ax.set_xticks([j + 0.4 - width / 2 for j in range(len(POOLS))])
    ax.set_xticklabels([POOL_LABEL[p] for p in POOLS], fontsize=8)
    ax.set_ylabel(f"exact match, $T \\leq {cut}$")
    ax.set_title("The three generalisation axes\n(held-out $N$ is the arity axis)")
    ax.set_ylim(0, 1.03)
    ax.legend(fontsize=7)

    # (c) rule retention
    ax = axes[1][0]
    ts = sorted(_ik(runs[0]["arms"][arms[0]]["rule_separation"]))
    for arm in arms:
        base = _ms([r["arms"][arm]["rule_separation_t0"] for r in runs])[0]
        ys = [y[0] for y in curve(arm, "rule_separation", ts)]
        ax.plot([0] + ts, [base] + ys, marker="o", ms=3, color=COLORS[arm],
                label=ARM_LABEL[arm])
    ax.axvline(cut + 0.5, color="k", ls=":", lw=1)
    ax.set_xlabel("rollout step $t$")
    ax.set_ylabel(r"$\cos(h_t(N),\, h_t(N'))$, same $x_0$")
    ax.set_title("Rule separation — does the rollout keep the\ntwo worlds apart? (lower = better)")
    ax.legend(fontsize=7)

    # (d) state closure
    ax = axes[1][1]
    for arm in arms:
        ys = [y[0] for y in curve(arm, "on_manifold_cos", ts)]
        ax.plot(ts, ys, marker="o", ms=3, color=COLORS[arm], label=ARM_LABEL[arm])
        ir = [y[0] for y in curve(arm, "inrange", ts)]
        ax.plot(ts, ir, ls="--", lw=1, color=COLORS[arm], alpha=0.7)
    ax.axvline(cut + 0.5, color="k", ls=":", lw=1)
    ax.set_xlabel("rollout step $t$")
    ax.set_ylabel("cosine / fraction")
    ax.set_title("State closure $\\cos(h_t, \\mathrm{Enc}(N, x_t))$ (solid)\n"
                 "and decoded $\\hat{x}_t < N$ (dashed)")
    ax.legend(fontsize=7)

    fig.suptitle(
        f"Variable modulus: {fam['n_train_moduli']} train / {fam['n_heldout_moduli']} held-out "
        f"$N \\in [{fam['modulus_min']}, {fam['modulus_max']}]$, "
        f"first depth-repeat $\\geq$ {fam['depth_first_repeat_min']} "
        f"({len(runs)} seed{'s' if len(runs) > 1 else ''})",
        fontsize=11,
    )
    fig.tight_layout()
    path = out_dir / "fig_variable_modulus.png"
    fig.savefig(path, dpi=160)
    print(f"\n[figure] {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="cut1")
    ap.add_argument("--results", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    here = Path(__file__).parent
    results_dir = Path(args.results) if args.results else here / "results"
    out = Path(args.out) if args.out else here / "figures" / args.tag
    report(load(args.tag, results_dir), args.tag, out)


if __name__ == "__main__":
    main()
