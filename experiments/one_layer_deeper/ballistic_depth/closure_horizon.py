"""Does manifold closure *predict* the composition horizon, across arms?

The mechanism claim in `README.md` §2 rests on two dots: base sits at closure 0.19 /
horizon 13, cycle at 0.99 / 35. Two points always fall on a line, so that contrast cannot
separate "closure governs how far you can roll" from "we ran one intervention that moved
a dozen things at once, and closure happens to be one of them."

This turns the contrast into a slope, using only runs that already exist. Every
`results/<tag>/results_seed<N>.json` carries `on_manifold_cos` and `exact_seen_x` at every
depth, so closure-vs-outcome is a scatter over 48 completed runs spanning 17 configs —
quantise at two codebook sizes, iso-compute, re-entry at 0/0.1/0.3/1.0, cycle-only,
re-entry-only, base. No GPU, no retraining.

The load-bearing test is not the pooled correlation (which a single strong intervention can
manufacture on its own). It is **transfer**: fit the line on the re-entry weight sweep alone
— one knob, one family — and ask whether it predicts configs reached by *unrelated* knobs.
If closure is a dial, the held-out families land on the line. If it is a by-product of the
one thing that worked, they do not.

Readouts:
  closure@t     mean cos(h_t rolled, Enc(true x_t)) at a fixed t, per run.
  horizon       first depth with exact_seen_x < 0.5. Right-censored in the 20-depth runs
                (cycle arms never fall below 0.5 there), so the horizon regression is
                restricted to the eval_max_depth=60 runs, where every arm is uncensored.
  exact@20      uncensored and available in all 51 runs, so it carries the pooled fit.

Usage:  python3 one_layer_deeper/ballistic_depth/closure_horizon.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGURES = HERE / "figures" / "closure_horizon"

# Which tag/arm pairs belong to which knob-family. The transfer test fits on one family
# and predicts the others, so this grouping is the experiment, not decoration.
FAMILY = {
    "w_re0": "re-entry sweep",
    "w_re0.1": "re-entry sweep",
    "w_re0.3": "re-entry sweep",
    "w_re1.0": "re-entry sweep",
    "deep60": "main arms",
    "cut1": "main arms",
    # The cycle-weight sweep is the knob that breaks the collinearity: inside the closed
    # cluster every earlier run had cycle_w pinned at 1.0, so closure there was generated
    # entirely by the re-entry weight and the two are indistinguishable as predictors.
    # These move closure with re-entry held at 0.
    "w_cyc0.01": "cycle sweep",
    "w_cyc0.05": "cycle sweep",
    "w_cyc0.2": "cycle sweep",
    "w_cyc3.0": "cycle sweep",
    "w_cyc10.0": "cycle sweep",
    "w_cyc30.0": "cycle sweep",
    "coldstart": "main arms",
    "cyc_only": "cycle-only",
    "re_only": "re-entry-only",
    "iso_compute": "iso-compute",
    "quant_fix": "quantise",
}

ID_DEPTHS = (1, 2, 3, 4, 5, 6)


def _int_keys(d):
    return {int(k): float(v) for k, v in d.items()}


def horizon(exact, max_depth, thresh=0.5):
    """First depth below `thresh`. Returns (value, censored)."""
    for t in range(1, max_depth + 1):
        if t in exact and exact[t] < thresh:
            return float(t), False
    return float(max_depth), True


def load_rows():
    rows = []
    for path in sorted(RESULTS.glob("*/results_seed*.json")):
        tag = path.parent.name
        seed = int(path.stem.replace("results_seed", ""))
        blob = json.loads(path.read_text())
        max_depth = int(blob["task"]["eval_depths"][-1])
        cfg = blob["config"]
        for arm, res in blob["arms"].items():
            if "on_manifold_cos" not in res:
                continue  # feedforward: non-recurrent, no rolled state to measure
            cos = _int_keys(res["on_manifold_cos"])
            exact = _int_keys(res["exact_seen_x"])
            held = _int_keys(res["exact_heldout_x"])
            hz, censored = horizon(exact, max_depth)
            rows.append(
                dict(
                    tag=tag,
                    arm=arm,
                    seed=seed,
                    family=FAMILY.get(tag, "other"),
                    label=f"{tag}/{arm}",
                    max_depth=max_depth,
                    reentry_w=float(cfg.get("consist_reentry", float("nan"))),
                    cycle_w=float(cfg.get("consist_cycle", float("nan"))),
                    closure6=cos.get(6, np.nan),
                    closure10=cos.get(10, np.nan),
                    closure20=cos.get(20, np.nan),
                    id_acc=float(np.mean([exact[d] for d in ID_DEPTHS if d in exact])),
                    exact20=exact.get(20, np.nan),
                    heldout20=held.get(20, np.nan),
                    ood_mean=float(
                        np.mean([v for d, v in exact.items() if 6 < d <= 20])
                    ),
                    horizon=hz,
                    censored=censored,
                )
            )
    return rows


def fit(x, y):
    """OLS slope/intercept plus Pearson r, Spearman rho, R²."""
    from scipy import stats

    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 3 or np.allclose(x, x[0]):
        return None
    lr = stats.linregress(x, y)
    rho, rho_p = stats.spearmanr(x, y)
    return dict(
        n=int(x.size),
        slope=float(lr.slope),
        intercept=float(lr.intercept),
        r=float(lr.rvalue),
        r2=float(lr.rvalue**2),
        p=float(lr.pvalue),
        rho=float(rho),
        rho_p=float(rho_p),
    )


def show(name, f):
    if f is None:
        print(f"  {name:<44} (insufficient spread)")
        return
    print(
        f"  {name:<44} n={f['n']:<3d} r={f['r']:+.3f}  R²={f['r2']:.3f}  "
        f"rho={f['rho']:+.3f}  p={f['p']:.2g}"
    )


def transfer_test(rows, xkey, ykey, fit_family="re-entry sweep"):
    """Fit on one knob-family, predict the others. The discriminating test."""
    tr = [r for r in rows if r["family"] == fit_family]
    f = fit([r[xkey] for r in tr], [r[ykey] for r in tr])
    if f is None:
        return None
    out = {"fit": f, "held": {}}
    for fam in sorted({r["family"] for r in rows if r["family"] != fit_family}):
        he = [r for r in rows if r["family"] == fam]
        x = np.array([r[xkey] for r in he], float)
        y = np.array([r[ykey] for r in he], float)
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]
        if not x.size:
            continue
        pred = f["slope"] * x + f["intercept"]
        out["held"][fam] = dict(
            n=int(x.size),
            mae=float(np.mean(np.abs(y - pred))),
            bias=float(np.mean(y - pred)),
            y_mean=float(y.mean()),
            pred_mean=float(pred.mean()),
        )
    return out


def figure(rows, deep):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES.mkdir(parents=True, exist_ok=True)
    colors = {
        "re-entry sweep": "#2f6f9f",
        "cycle sweep": "#e08a1e",
        "main arms": "#c1440e",
        "cycle-only": "#3f8f5f",
        "iso-compute": "#c99a2e",
        "re-entry-only": "#d4527a",
        "quantise": "#8a6bbf",
        "other": "#888888",
    }
    fig, axes = plt.subplots(1, 4, figsize=(20.5, 4.9))

    def scatter(ax, rows_, xkey, ykey, legend=False):
        for fam in colors:
            pts = [r for r in rows_ if r["family"] == fam]
            if not pts:
                continue
            ax.scatter([r[xkey] for r in pts], [r[ykey] for r in pts], s=46, alpha=0.85,
                       color=colors[fam], edgecolor="white", linewidth=0.7,
                       label=fam if legend else None, zorder=3)

    # P0 — pooled. The cycle sweep (§7) is what fills the region that was empty when this
    # regression was first run, which is the only reason the pooled fit means anything.
    ax = axes[0]
    scatter(ax, rows, "closure10", "exact20", legend=True)
    f = fit([r["closure10"] for r in rows], [r["exact20"] for r in rows])
    ax.set_title(f"pooled — the cycle sweep filled the gap\n"
                 f"rho={f['rho']:+.2f}  R²={f['r2']:.2f}  n={f['n']}", fontsize=10.5)
    ax.set_ylabel("exact match @ T=20", fontsize=9.5)
    ax.set_xlabel("closure @ t=10   cos(h_t, Enc(x_t))", fontsize=9.5)
    ax.legend(fontsize=7.5, loc="upper left", framealpha=0.9)

    # P1 — the strongest single readout: closure measured past the training range against the
    # uncensored horizon.
    ax = axes[1]
    scatter(ax, deep, "closure20", "horizon")
    f = fit([r["closure20"] for r in deep], [r["horizon"] for r in deep])
    xs = np.linspace(min(r["closure20"] for r in deep), max(r["closure20"] for r in deep), 50)
    ax.plot(xs, f["slope"] * xs + f["intercept"], "k--", lw=1.3, zorder=2)
    ax.set_title(f"closure @ t=20 vs uncensored horizon\n"
                 f"rho={f['rho']:+.2f}  R²={f['r2']:.2f}  n={f['n']}", fontsize=10.5)
    ax.set_ylabel("composition horizon (T @ 50%)", fontsize=9.5)
    ax.set_xlabel("closure @ t=20", fontsize=9.5)

    # P2 — the de-confounder. Re-entry is pinned at 0, so cycle weight moves closure on its own;
    # the turnover at w=30 is where the knob and the outcome part company but closure does not.
    ax = axes[2]
    fam = [r for r in rows
           if r["arm"] == "consist" and r["reentry_w"] == 0.0 and r["max_depth"] >= 60]
    byw = {}
    for r in fam:
        byw.setdefault(r["cycle_w"], []).append(r)
    ws = sorted(byw)
    hz = [np.mean([r["horizon"] for r in byw[w]]) for w in ws]
    hze = [np.std([r["horizon"] for r in byw[w]]) for w in ws]
    clo = [np.mean([r["closure20"] for r in byw[w]]) for w in ws]
    ax.errorbar(ws, hz, yerr=hze, marker="o", color="#e08a1e", lw=1.9, capsize=3,
                label="horizon", zorder=3)
    ax.axhline(13.0, color="#c1440e", ls="--", lw=1.3, label="base (no cycle)", zorder=2)
    ax.set_xscale("log")
    ax.set_xlabel("cycle weight  (re-entry pinned at 0)", fontsize=9.5)
    ax.set_ylabel("composition horizon", fontsize=9.5)
    ax2 = ax.twinx()
    ax2.plot(ws, clo, marker="s", ms=4, color="#2f6f9f", lw=1.4, ls=":",
             label="closure @ t=20", zorder=3)
    ax2.set_ylabel("closure @ t=20", fontsize=9.5, color="#2f6f9f")
    ax2.tick_params(axis="y", labelcolor="#2f6f9f")
    ax.set_title("cycle-weight sweep: peak 51 at w=10, turnover at 30\n"
                 "weight stops predicting (R²=0.02), closure does not (0.69)", fontsize=10.5)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="lower right", framealpha=0.9)

    # P3 — where the ordinal reading holds but the calibrated one fails. The line is fit on the
    # re-entry sweep alone; the cycle sweep sits far below it and re-entry-only far above.
    ax = axes[3]
    scatter(ax, rows, "closure10", "exact20")
    t = transfer_test(rows, "closure10", "exact20")
    ft = t["fit"]
    xs = np.linspace(0, 1, 50)
    ax.plot(xs, ft["slope"] * xs + ft["intercept"], "k--", lw=1.3, zorder=2,
            label="fit on re-entry sweep only")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.1)
    ax.set_title("orders configs, does not price them\n"
                 "cycle sweep −0.34 below, re-entry-only +0.26 above", fontsize=10.5)
    ax.set_ylabel("exact match @ T=20", fontsize=9.5)
    ax.set_xlabel("closure @ t=10", fontsize=9.5)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.9)

    for ax in axes:
        ax.grid(alpha=0.25, zorder=0)

    fig.suptitle(
        "Closure is an ordinal predictor of the composition horizon (rho +0.94 over 72 runs), "
        "and tracks it through a knob reversal the weight itself does not",
        fontsize=12.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    out = FIGURES / "fig_closure_horizon.png"
    fig.savefig(out, dpi=160)
    print(f"\n[saved] {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", action="store_true", help="print every row")
    args = ap.parse_args()

    rows = load_rows()
    deep = [r for r in rows if r["max_depth"] >= 60]
    comp = [r for r in rows if r["id_acc"] >= 0.9]
    deep_comp = [r for r in deep if r["id_acc"] >= 0.9]

    print(f"loaded {len(rows)} recurrent runs across "
          f"{len({(r['tag'], r['arm']) for r in rows})} configs")
    print(f"  {len(deep)} at eval depth 60 (uncensored horizon), "
          f"{len(comp)} ID-competent (ID ≥ 0.9)")

    if args.dump:
        print(f"\n{'config':<24}{'seed':>5}{'clo6':>8}{'clo10':>8}{'ID':>7}"
              f"{'ex20':>8}{'hz':>6}{'cens':>6}")
        for r in sorted(rows, key=lambda r: (-r["closure10"],)):
            print(f"{r['label']:<24}{r['seed']:>5}{r['closure6']:>8.3f}"
                  f"{r['closure10']:>8.3f}{r['id_acc']:>7.3f}{r['exact20']:>8.3f}"
                  f"{r['horizon']:>6.0f}{str(r['censored']):>6}")

    # The de-confounded family: cycle weight varied with re-entry pinned at 0. Inside the
    # closed cluster every *earlier* run had cycle_w = 1.0, so closure there was generated
    # entirely by the re-entry weight and the two were indistinguishable as predictors.
    # This family moves closure with re-entry held fixed, which is what licenses reading
    # closure as more than a re-entry artifact.
    fam = [r for r in rows
           if r["arm"] == "consist" and r["reentry_w"] == 0.0 and r["max_depth"] >= 60]
    if fam:
        byw = {}
        for r in fam:
            byw.setdefault(r["cycle_w"], []).append(r)
        print("\n--- cycle-weight sweep at re-entry = 0 (the de-confounded family) ---")
        print(f"  {'cycle_w':>8}{'n':>4}{'closure@10':>12}{'closure@20':>12}"
              f"{'ID':>7}{'exact@20':>10}{'horizon':>14}")
        for w in sorted(byw):
            rs = byw[w]
            hz = [r["horizon"] for r in rs]
            cens = sum(r["censored"] for r in rs)
            print(f"  {w:>8}{len(rs):>4}"
                  f"{np.mean([r['closure10'] for r in rs]):>12.3f}"
                  f"{np.mean([r['closure20'] for r in rs]):>12.3f}"
                  f"{np.mean([r['id_acc'] for r in rs]):>7.3f}"
                  f"{np.mean([r['exact20'] for r in rs]):>10.3f}"
                  f"{np.mean(hz):>9.1f} ±{np.std(hz):<4.1f}"
                  + ("  (CENSORED)" if cens else ""))
        show("  closure20 -> horizon (this family)",
             fit([r["closure20"] for r in fam], [r["horizon"] for r in fam]))
        show("  cycle_w   -> horizon (this family)",
             fit([r["cycle_w"] for r in fam], [r["horizon"] for r in fam]))

    print("\n--- pooled: closure predicts the depth outcome ---")
    for xk in ("closure6", "closure10", "closure20"):
        show(f"{xk} -> exact@20 (all)", fit([r[xk] for r in rows],
                                            [r["exact20"] for r in rows]))
    show("closure10 -> exact@20 (ID-competent)",
         fit([r["closure10"] for r in comp], [r["exact20"] for r in comp]))
    show("closure10 -> mean exact T=7..20 (ID-comp)",
         fit([r["closure10"] for r in comp], [r["ood_mean"] for r in comp]))
    show("closure10 -> held-out-x @20 (ID-comp)",
         fit([r["closure10"] for r in comp], [r["heldout20"] for r in comp]))

    print("\n--- horizon, uncensored (60-depth runs only) ---")
    for xk in ("closure6", "closure10", "closure20"):
        show(f"{xk} -> horizon (all)", fit([r[xk] for r in deep],
                                           [r["horizon"] for r in deep]))
    show("closure10 -> horizon (ID-competent)",
         fit([r["closure10"] for r in deep_comp], [r["horizon"] for r in deep_comp]))

    print("\n--- BIMODALITY: the pooled fit is mostly a two-cluster lever ---")
    hi = [r for r in rows if r["closure10"] > 0.5]
    lo = [r for r in rows if r["closure10"] <= 0.5]
    mid = [r for r in rows if 0.2 < r["closure10"] <= 0.5]
    print(f"  closure10 > 0.5: n={len(hi)}  range "
          f"[{min(r['closure10'] for r in hi):.3f}, {max(r['closure10'] for r in hi):.3f}]")
    print(f"  closure10 <= 0.5: n={len(lo)}  range "
          f"[{min(r['closure10'] for r in lo):.3f}, {max(r['closure10'] for r in lo):.3f}]")
    print(f"  in the gap (0.2, 0.5]: n={len(mid)}   <- the pooled R² is unconstrained here")
    print("  within-cluster fits (the honest test — no lever arm):")
    show("  [hi] closure10 -> exact@20",
         fit([r["closure10"] for r in hi], [r["exact20"] for r in hi]))
    show("  [hi] closure6  -> exact@20",
         fit([r["closure6"] for r in hi], [r["exact20"] for r in hi]))
    show("  [hi] closure10 -> horizon (60-depth)",
         fit([r["closure10"] for r in hi if r["max_depth"] >= 60],
             [r["horizon"] for r in hi if r["max_depth"] >= 60]))
    show("  [lo] closure10 -> exact@20",
         fit([r["closure10"] for r in lo], [r["exact20"] for r in lo]))
    lo_nore = [r for r in lo if r["tag"] != "re_only"]
    show("  [lo] closure10 -> exact@20 (drop re_only)",
         fit([r["closure10"] for r in lo_nore], [r["exact20"] for r in lo_nore]))

    print("\n--- control: is closure just a proxy for in-distribution fit? ---")
    show("ID acc -> exact@20 (all)", fit([r["id_acc"] for r in rows],
                                         [r["exact20"] for r in rows]))
    show("ID acc -> exact@20 (ID-competent)",
         fit([r["id_acc"] for r in comp], [r["exact20"] for r in comp]))

    print("\n--- transfer: fit on the re-entry sweep, predict the other knobs ---")
    for xk, yk, dat in (
        ("closure10", "exact20", rows),
        ("closure10", "horizon", deep),
    ):
        t = transfer_test(dat, xk, yk)
        if t is None:
            continue
        f = t["fit"]
        print(f"  fit {xk} -> {yk} on re-entry sweep: n={f['n']} "
              f"slope={f['slope']:.3f} intercept={f['intercept']:.3f} R²={f['r2']:.3f}")
        for fam, h in sorted(t["held"].items(), key=lambda kv: kv[1]["bias"]):
            print(f"    held-out {fam:<16} n={h['n']:<3d} "
                  f"MAE={h['mae']:.3f}  bias={h['bias']:+.3f}  "
                  f"(actual {h['y_mean']:.3f} vs predicted {h['pred_mean']:.3f})")

    print("\n--- per-config residual vs the re-entry-sweep line (closure10 -> exact@20) ---")
    t = transfer_test(rows, "closure10", "exact20")
    f = t["fit"]
    by_cfg = {}
    for r in rows:
        by_cfg.setdefault((r["family"], r["label"]), []).append(r)
    print(f"  {'config':<26}{'fam':<16}{'clo10':>8}{'actual':>9}{'pred':>9}{'resid':>9}")
    for (fam, label), rs in sorted(
        by_cfg.items(), key=lambda kv: -np.mean([r["exact20"] for r in kv[1]])
    ):
        clo = float(np.mean([r["closure10"] for r in rs]))
        act = float(np.mean([r["exact20"] for r in rs]))
        pred = f["slope"] * clo + f["intercept"]
        print(f"  {label:<26}{fam:<16}{clo:>8.3f}{act:>9.3f}{pred:>9.3f}"
              f"{act - pred:>+9.3f}")

    figure(rows, deep)


if __name__ == "__main__":
    main()
