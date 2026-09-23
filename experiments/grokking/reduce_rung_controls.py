"""[grokking] Reduce `rung_controls.py` runs to the two control tables.

    python3 grokking/reduce_rung_controls.py --tag r2 --fetch      # from experiments/

Reads every /data/<tag>/controls_*.json, writes figures/rung_controls_<tag>_table.txt.
"""

import argparse
import glob
import json
import os
import subprocess

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def fetch(tag):
    d = os.path.join(HERE, "results", tag)
    os.makedirs(d, exist_ok=True)
    env = dict(os.environ, MODAL_PROFILE="chromatic")
    ls = subprocess.run(["modal", "volume", "ls", "grokking-mint-data", tag], capture_output=True, text=True,
                        env=env, check=True).stdout
    for name in sorted(set(w.strip("│ ").split("/")[-1] for w in ls.split() if "controls_" in w)):
        subprocess.run(["modal", "volume", "get", "--force", "grokking-mint-data", f"{tag}/{name}",
                        os.path.join(d, name)], check=True, env=env)


def fmt(x, nd=4):
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.{nd}g}"
    return str(x)


def peak(r):
    c = np.asarray(r["curve"])
    i = int(np.argmax(c[:, 2]))
    return c[i, 2], int(c[i, 0])


def probe_at(r, ep):
    """The probe at the largest epoch <= ep (the init probe has epoch -1)."""
    ps = [p for p in r["probes"] if p["epoch"] <= ep]
    return ps[-1] if ps else r["probes"][0]


def table(results, tag):
    R = {r["arm"]["name"]: r for r in results}
    L = [f"# grokking / rung controls -- tag {tag}; seed 42, train fraction 0.3, CPU",
         "# e99/e100 = first eval epoch (every 10) with test acc >= thr ('-' = never within cap); "
         "peak on the stored 100-epoch grid",
         "# probe on all p^2 pairs: gbar = diagonal-averaged row-centred logit, gbar_margin = gbar(0) - "
         "max_{d!=0} gbar(d); margin = logit[true] - max other; alpha_w = cosine coefficient of gbar", ""]
    L.append("equal-amplitude margins of sum_{w in K} cos(2 pi w d/p) (peak minus runner-up), computed exactly: "
             "top1 {30}: 0.002097; K13: 9.958; all 48: 48.50")
    L.append("")
    A = [r for n, r in R.items() if n.startswith("A")]
    if A:
        L += ["## Control A: all-48 WITH DC (square, F = s*Q, Q orthogonal) as a reparameterization of one-hot, MLP",
              f"{'arm':>22} {'feat':>9} {'init':>11} {'opt':>22} {'init|dlogit|':>12} {'e99':>6} {'e100':>6} "
              f"{'cap':>6} {'final_te':>8} {'final_tr':>8} {'peak_te@ep':>13} {'gbar_margin':>11} {'diag_frac':>9}"]
        for r in sorted(A, key=lambda r: r["arm"]["name"]):
            a = r["arm"]
            pk, pe = peak(r)
            fp = r["probes"][-1]
            opt = f"{a.get('opt', 'adamw')} lr{a.get('lr', 1e-3):g} wd{a.get('wd', 1.0):g}" + \
                  (f" m{a.get('momentum', 0):g}" if a.get("opt") == "sgd" else "")
            L.append(f"{a['name']:>22} {a['alphabet']:>9} {a.get('init', 'default'):>11} {opt:>22} "
                     f"{fmt(r.get('init_maxabs_vs_raw')):>12} {fmt(r['first_hit']['0.99']):>6} "
                     f"{fmt(r['first_hit']['1.0']):>6} {a['cap']:>6} {r['final_test']:>8.4f} {r['final_train']:>8.4f} "
                     f"{pk:>6.3f}@{pe:<6d} {fp['gbar_margin']:>11.4g} {fp['diag_frac']:>9.3f}")
        L.append("  reference (r1): raw one-hot MLP e99/e100 = 16040/18520; all48 (no DC, cos/sin scale) and all48n "
                 "never solve (test <= 0.0044).")
        for n_raw in sorted(n for n in R if n.startswith("A2_raw_sgd_")):
            n_q = n_raw.replace("A2_raw_sgd_", "A2_q_transported_sgd_")
            if n_q not in R:
                continue
            c1, c2 = np.asarray(R[n_raw]["curve"]), np.asarray(R[n_q]["curve"])
            e = np.intersect1d(c1[:, 0], c2[:, 0])
            c1, c2 = c1[np.isin(c1[:, 0], e)], c2[np.isin(c2[:, 0], e)]
            dl, dt = np.abs(c1[:, 3] - c2[:, 3]), np.abs(c1[:, 2] - c2[:, 2])

            def first(m):
                return int(c1[np.argmax(m), 0]) if m.any() else None
            L.append(f"  A2 trajectory match {n_raw[11:]} (raw vs q-transported, SGD, {len(e)} shared evals on "
                     f"the 100-epoch grid): max |d test| {dt.max():.3g}, max |d train| "
                     f"{np.abs(c1[:, 1] - c2[:, 1]).max():.3g}, max |d loss| {dl.max():.3g}; first epoch with "
                     f"relative |d loss| > 1e-4: {first(dl / np.maximum(c1[:, 3], 1e-12) > 1e-4)}, "
                     f"|d test| > 0.01: {first(dt > 0.01)}")
        L.append("")
    B1 = sorted([r for n, r in R.items() if n.startswith("B1")], key=lambda r: r["arm"]["plant"])
    if B1:
        L += ["## Control B1: bilinear m=4 top1 with the exact one-frequency solution planted (amplitude A = t^3, "
              "U = V = t*selector, W = t*base), then the recipe (AdamW lr 1e-3 wd 1.0), no early stop, 20k epochs",
              f"{'A':>6} | {'test acc at epoch  init / 500 / 1k / 5k / 10k / 20k':>52} | {'min test':>8} | "
              f"{'alpha_30 at init / 1k / 20k':>30} | {'gbar_margin init / 20k':>24} | {'te margin min 20k':>17} | "
              f"{'wnorm init/20k':>15}"]
        for r in B1:
            eps = (-1, 500, 1000, 5000, 10000, 20000)
            ps = [probe_at(r, e) for e in eps]
            accs = " / ".join(f"{p['test_acc']:.3f}" for p in ps)
            al = " / ".join(f"{probe_at(r, e)['alpha']['30']:.4g}" for e in (-1, 1000, 20000))
            mn = float(np.asarray(r["curve"])[:, 2].min())
            L.append(f"{r['arm']['plant']:>6} | {accs:>52} | {mn:>8.3f} | {al:>30} | "
                     f"{ps[0]['gbar_margin']:>10.4g} / {ps[-1]['gbar_margin']:<11.4g} | "
                     f"{ps[-1]['margin_te_min']:>17.4g} | {ps[0]['wnorm']:>6.3g}/{ps[-1]['wnorm']:<7.3g}")
        L.append("")
    B2 = [r for n, r in R.items() if n.startswith("B2")]
    if B2:
        L += ["## Control B2: weight-decay sweep from random init, top1 alphabet, AdamW lr 1e-3, cap 40k, stop at 1.0",
              f"{'learner':>12} {'wd':>5} {'params':>6} | {'e99':>6} {'e100':>6} {'final_te':>8} {'final_tr':>8} "
              f"{'peak_te@ep':>13} | {'alpha_30':>9} {'gbar_margin':>11} {'te_margin_mean':>14} {'wnorm':>7}"]
        for r in sorted(B2, key=lambda r: (r["arm"]["learner"], -r["arm"]["wd"])):
            a = r["arm"]
            pk, pe = peak(r)
            fp = r["probes"][-1]
            lab = "bilinear m=4" if a["learner"] == "bilinear" else "mlp"
            L.append(f"{lab:>12} {a['wd']:>5} {r['params']:>6} | {fmt(r['first_hit']['0.99']):>6} "
                     f"{fmt(r['first_hit']['1.0']):>6} {r['final_test']:>8.4f} {r['final_train']:>8.4f} "
                     f"{pk:>6.3f}@{pe:<6d} | {fp['alpha']['30']:>9.4g} {fp['gbar_margin']:>11.4g} "
                     f"{fp['margin_te_mean']:>14.4g} {fp['wnorm']:>7.3g}")
        L.append("  reference (r1, wd 1.0): top1 mlp final 0.546 (peak 0.714), top1 bilinear m=4 final 0.098")
        L.append("")
    B3 = [r for n, r in R.items() if n.startswith("B3")]
    if B3:
        L += ["## Control B3: r1's minted-13 bilinear arm (m=52, 7845 params) re-run with probes"]
        for r in sorted(B3, key=lambda r: r["arm"]["name"]):
            L.append(f"{r['arm']['name']}: e99 {r['first_hit']['0.99']} e100 {r['first_hit']['1.0']} "
                     f"(r1: 2630 / 3760), epochs run {r['epochs_run']}, final test {r['final_test']:.4f}")
            eps = sorted({p["epoch"] for p in r["probes"]})
            show = [e for e in eps if e in (-1, 1000, 2000, 3000) or e == eps[-1]
                    or (e >= 5000 and e % 10000 == 0)]
            for e in show:
                p = probe_at(r, e)
                al = p["alpha"]
                vals = np.array([al[str(w)] for w in [10, 11, 18, 24, 30, 33, 36, 38, 39, 44, 45, 46, 47]])
                L.append(f"   epoch {e:>6}: test {p['test_acc']:.4f} | alpha over K13 min/mean/max "
                         f"{vals.min():.3f}/{vals.mean():.3f}/{vals.max():.3f} sum {vals.sum():.3f} | "
                         f"gbar_margin {p['gbar_margin']:.4g} (gbar0 {p['gbar0']:.4g}) | margin test min/mean "
                         f"{p['margin_te_min']:.4g}/{p['margin_te_mean']:.4g} train min/mean {p['margin_tr_min']:.4g}/"
                         f"{p['margin_tr_mean']:.4g} | diag_frac {p['diag_frac']:.3f} | wnorm {p['wnorm']:.3g} | "
                         f"top5 alpha {p['alpha_top5']}")
            fp = r["probes"][-1]
            L.append(f"   alpha per w at the last probe: {fp['alpha']}")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="r2")
    ap.add_argument("--fetch", action="store_true")
    a = ap.parse_args()
    if a.fetch:
        fetch(a.tag)
    results = []
    for path in sorted(glob.glob(os.path.join(HERE, "results", a.tag, "controls_*.json"))):
        with open(path) as f:
            results += json.load(f)["results"]
    txt = table(results, a.tag)
    with open(os.path.join(HERE, "figures", f"rung_controls_{a.tag}_table.txt"), "w") as f:
        f.write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
