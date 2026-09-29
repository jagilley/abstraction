"""[grokking] Reduce a `rung.py::rung` run to its table and figures.

    python3 grokking/reduce_rung.py --tag r1 --fetch      # from experiments/

Writes figures/rung_<tag>_table.txt, figures/rung_<tag>_curves.png, figures/rung_<tag>_sweep.png.
"""

import argparse
import json
import os
import subprocess

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
COL = {"raw": "#52514e", "minted": "#2a78d6", "all48": "#eb6834", "lowE": "#1baf7a", "top1": "#e87ba4",
       "all48n": "#eda100"}
ALPHS = ("raw", "minted", "all48", "lowE", "top1", "all48n")


def fetch(tag):
    d = os.path.join(HERE, "results", tag)
    os.makedirs(d, exist_ok=True)
    subprocess.run(["modal", "volume", "get", "--force", "grokking-mint-data", f"{tag}/rung.json",
                    os.path.join(d, "rung.json")], check=True, env=dict(os.environ, MODAL_PROFILE="chromatic"))


def fmt(x):
    return "-" if x is None else (f"{x:.3g}" if isinstance(x, float) else str(x))


def _peak(r):
    c = np.asarray(r["curve"])
    i = int(np.argmax(c[:, 2]))
    return c[i, 2], int(c[i, 0])


def _row(r):
    a = r["arm"]
    pk, pe = _peak(r)
    return (f"{a['name']:>24} {a['alphabet']:>6} {a['learner']:>8} {fmt(r.get('m_bilinear')):>4} {a['frac']:>5} "
            f"{r['n_train']:>5} {r['in_dim']:>4} {r['params']:>7} {fmt(r['first_hit']['0.99']):>6} "
            f"{fmt(r['first_hit']['1.0']):>6} {a['cap']:>6} {r['final_test']:>8.4f} {r['final_train']:>8.4f} "
            f"{pk:>6.3f}@{pe:<6d} "
            f"{fmt(r['params_x_epochs_0.99']):>9} {fmt(r['params_x_epochs_1.0']):>9} "
            f"{fmt(r['flops_1.0']):>9} {r['seconds']:>5.0f}")


HDR = (f"{'arm':>24} {'alph':>6} {'learner':>8} {'m':>4} {'frac':>5} {'n_tr':>5} {'D':>4} {'params':>7} "
       f"{'e99':>6} {'e100':>6} {'cap':>6} {'final_te':>8} {'final_tr':>8} {'peak_te@ep':>13} "
       f"{'par*ep99':>9} {'par*ep100':>9} "
       f"{'flops100':>9} {'sec':>5}")


def table(d, tag):
    fba = d.get("freqs_by_alph", {})
    L = [f"# grokking / rung -- tag {tag}; K from mint {d['mint_tag']} ({d['k_form']}/{d['k_rule']}), "
         f"k={len(d['K'])}: K={d['K']}",
         f"# K is the top-{len(d['K'])} of the mint producer order: {d.get('K_is_prod_top_k')}; producer order "
         f"{d.get('prod_order')}",
         f"# lowE = {d['lowK']};  top1 = {d.get('top1')} (== the ALS/m3 kept K: {d.get('top1_is_ALS_m3_K')})",
         f"# recipe {d['recipe']}; epochs = first eval epoch (every 10) with test acc >= thr; runs stop at "
         f"test 1.0 or the cap; '-' = not reached within cap",
         "# par*ep = params x (epochs+1); flops100 = 6 x params x n_train x (e100+1). Bilinear: logits = "
         "W((U x_a) * (V x_b)) + b, width m. peak_te on the stored 100-epoch curve grid.",
         "", "## Main arms (train fraction 0.3)", HDR]
    res = d["results"]
    main = [r for r in res if r["arm"]["group"] == "main"]
    key = lambda r: (ALPHS.index(r["arm"]["alphabet"]), r["arm"]["learner"], r["arm"]["name"])
    for r in sorted(main, key=key):
        L.append(_row(r))
    L += ["", "## Sweep (mlp): all arms by train fraction (0.3 rows are the main arms)", HDR]
    sw = [r for r in res if r["arm"]["learner"] == "mlp" and r["arm"].get("task", "add2") == "add2"
          and r["arm"]["alphabet"] in ("raw", "minted", "top1")]
    for r in sorted(sw, key=lambda r: (-r["arm"]["frac"], ("raw", "minted", "top1").index(r["arm"]["alphabet"]))):
        L.append(_row(r))
    L += ["", "## Sweep summary: epochs to 0.99 / 1.0 test (final test acc), mlp"]
    L.append(f"{'frac':>5} {'n_tr':>5} | " + " | ".join(f"{a:>24}" for a in ("raw", "minted", "top1")))
    for fr in sorted({r["arm"]["frac"] for r in sw}, reverse=True):
        cells, n = [], ""
        for a in ("raw", "minted", "top1"):
            r = next((x for x in sw if x["arm"]["frac"] == fr and x["arm"]["alphabet"] == a), None)
            if r is None:
                cells.append(f"{'n/a':>24}")
                continue
            n = r["n_train"]
            cells.append(f"{fmt(r['first_hit']['0.99']):>7} / {fmt(r['first_hit']['1.0']):>6} ({r['final_test']:.3f})")
        L.append(f"{fr:>5} {n:>5} | " + " | ".join(f"{c:>24}" for c in cells))
    other = [r for r in res if r["arm"]["group"] not in ("main", "sweep")]
    if other:
        L += ["", "## Control arms added after r1 (tag r1ctrl, merged via --extra-tags): all48n = all48 scaled by sqrt(k/48), "
              "per-input norm equal to minted's", HDR] + [_row(r) for r in other]
    raw = next((r for r in main if r["arm"]["name"] == "main_raw_mlp"), None)
    if raw:
        L += ["", f"# consistency: main_raw_mlp is the mint net's recipe (seed 42, same split): first test>=0.99 / "
                  f"1.0 at {raw['first_hit']['0.99']} / {raw['first_hit']['1.0']} (mint m1, 50-epoch grid: 16050 / 18550)"]
    return "\n".join(L)


def figures(d, tag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "grid.color": "#e6e5e0"})
    main = [r for r in d["results"] if r["arm"]["group"] == "main"]
    if main:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
        for ax, learner in zip(axes, ("mlp", "bilinear")):
            for r in sorted(main, key=lambda r: ALPHS.index(r["arm"]["alphabet"])):
                if r["arm"]["learner"] != learner:
                    continue
                c = np.asarray(r["curve"])
                m4 = r["arm"]["name"].endswith("_m4")
                ax.plot(c[:, 0], c[:, 2], color=COL[r["arm"]["alphabet"]], lw=1.6, ls="--" if m4 else "-",
                        label=f"{r['arm']['alphabet']}{' m=4' if m4 else ''} ({r['params']} params)")
            ax.set_xscale("symlog", linthresh=100)
            ax.set_title(f"{learner}, train fraction 0.3", fontsize=10)
            ax.set_xlabel("epoch (symlog)")
            ax.grid(True, axis="y", lw=0.6)
            ax.legend(frameon=False, fontsize=8)
        axes[0].set_ylabel("test accuracy")
        fig.tight_layout()
        fig.savefig(os.path.join(HERE, "figures", f"rung_{tag}_curves.png"), dpi=140)
        plt.close(fig)
    sw = [r for r in d["results"] if r["arm"]["learner"] == "mlp"
          and r["arm"]["alphabet"] in ("raw", "minted", "top1") and r["arm"].get("task", "add2") == "add2"]
    if sw:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
        for j, (thr, ax) in enumerate(zip(("0.99", "1.0"), axes[:2])):
            for alph in ("raw", "minted", "top1"):
                rs = sorted([r for r in sw if r["arm"]["alphabet"] == alph], key=lambda r: r["arm"]["frac"])
                fr = [r["arm"]["frac"] for r in rs]
                e = [r["first_hit"][thr] for r in rs]
                ax.plot([f for f, x in zip(fr, e) if x is not None], [x + 1 for x in e if x is not None],
                        color=COL[alph], marker="o", lw=1.6, label=alph)
                jit = {"raw": 0.93, "minted": 1.0, "top1": 1.075}[alph]   # keep the cap markers apart
                miss = [(f * jit, r["arm"]["cap"]) for f, x, r in zip(fr, e, rs) if x is None]
                if miss:
                    ax.scatter([m[0] for m in miss], [m[1] for m in miss], color=COL[alph], marker="x", s=60,
                               label=f"{alph}: not reached (plotted at cap)")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("train fraction (log)")
            ax.set_ylabel("epochs (log)")
            ax.set_title(f"epochs to test acc >= {thr}", fontsize=10)
            ax.grid(True, lw=0.6)
        axes[0].legend(frameon=False, fontsize=8)
        ax = axes[2]
        for alph in ("raw", "minted", "top1"):
            rs = sorted([r for r in sw if r["arm"]["alphabet"] == alph], key=lambda r: r["arm"]["frac"])
            ax.plot([r["arm"]["frac"] for r in rs], [r["final_test"] for r in rs], color=COL[alph], marker="o",
                    lw=1.6, label=alph)
        ax.set_xscale("log")
        ax.set_xlabel("train fraction (log)")
        ax.set_ylabel("test acc at stop (1.0 or cap)")
        from matplotlib.ticker import NullFormatter
        for a_ in axes:
            a_.set_xticks([0.02, 0.05, 0.1, 0.2, 0.3])
            a_.set_xticklabels(["0.02", "0.05", "0.1", "0.2", "0.3"])
            a_.xaxis.set_minor_formatter(NullFormatter())
        ax.set_title("final test accuracy", fontsize=10)
        ax.grid(True, axis="y", lw=0.6)
        fig.tight_layout()
        fig.savefig(os.path.join(HERE, "figures", f"rung_{tag}_sweep.png"), dpi=140)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="r1")
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--extra-tags", default="", help="comma list of further rung tags merged into the table")
    a = ap.parse_args()
    extra = [t for t in a.extra_tags.split(",") if t]
    if a.fetch:
        for t in [a.tag] + extra:
            fetch(t)
    with open(os.path.join(HERE, "results", a.tag, "rung.json")) as f:
        d = json.load(f)
    for t in extra:
        with open(os.path.join(HERE, "results", t, "rung.json")) as f:
            e = json.load(f)
        assert e["K"] == d["K"], (t, e["K"], d["K"])
        d["results"] += e["results"]
    txt = table(d, a.tag)
    with open(os.path.join(HERE, "figures", f"rung_{a.tag}_table.txt"), "w") as f:
        f.write(txt + "\n")
    figures(d, a.tag)
    print(txt)


if __name__ == "__main__":
    main()
