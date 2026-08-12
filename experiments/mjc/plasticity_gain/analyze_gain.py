"""Local post-processing for plasticity_gain runs (no compute; reads results.json).

Prints the summary tables for a run: control recovery ladder per arm, recovery AUC,
budget checks (realized mean weight, cumulative-weight shares vs sample shares),
retention (base probe error), benchmark habituation on the noise region, and gate stats.

Usage:
    python3 mjc/plasticity_gain/analyze_gain.py --tag gain_s0
    # pull first if the local mirror is missing:
    #   modal volume get mujoco-control-data plasticity_gain/<tag>/results.json \
    #       mjc/plasticity_gain/figures/plasticity_gain_<tag>/results.json
"""

import argparse
import json
import os

import numpy as np


def load(tag):
    p = os.path.join(os.path.dirname(__file__), "figures", f"plasticity_gain_{tag}", "results.json")
    with open(p) as fh:
        return json.load(fh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    R = load(args.tag)
    cfgR = R["config"]; arms = cfgR["arms"]; ctrls = cfgR["controllers"]
    REG = cfgR["regions"]
    red = [r["name"] for r in REG if r["noise"] == 0.0]
    noi = [r["name"] for r in REG if r["noise"] > 0.0]

    cal = R["calibration"]
    print(f"== calibration ==  tau={cal['tau']:.5f} theta={cal['theta']:.4f} "
          f"gate p10/50/90={['%.3f' % x for x in cal['stream_gate_p10_p50_p90']]}")
    print(f"stale probe: " + " ".join(f"{k}={v:.4f}" for k, v in cal["stale_probe"].items()))
    print(f"ceiling:     " + " ".join(f"{c}={R['ceiling'][c]:.4f}" for c in ctrls)
          + "  fm: " + " ".join(f"{k}={v:.4f}" for k, v in R["ceiling"]["fm"].items()))
    print(f"stale grade: " + " ".join(f"{c}={R['stale'][c]:.4f}" for c in ctrls))

    # ---- control ladders ----
    for c in ctrls:
        print(f"\n== {c} goal-dist by milestone ==")
        ms = [r["transitions"] for r in R["arms"][arms[0]]["ladder"]]
        print("arm          " + "".join(f"{m:>8d}" for m in ms) + "     AUC(m>0)  excess-AUC")
        ceil = R["ceiling"][c]
        for a in arms:
            lad = R["arms"][a]["ladder"]
            ys = [r.get(c, np.nan) for r in lad]
            auc = float(np.nanmean([y for r, y in zip(lad, ys) if r["transitions"] > 0]))
            exc = float(np.nanmean([max(y - ceil, 0.0) for r, y in zip(lad, ys) if r["transitions"] > 0]))
            print(f"{a:12s} " + "".join(f"{y:8.4f}" for y in ys) + f"     {auc:8.4f}  {exc:9.4f}")

    # ---- FM probe ladders (mechanism) ----
    for label, keys in [("reducible (R1+R2)", red), ("noise", noi), ("base (retention)", ["base"])]:
        print(f"\n== FM probe error, {label} ==")
        ms = [r["transitions"] for r in R["arms"][arms[0]]["ladder"]]
        print("arm          " + "".join(f"{m:>8d}" for m in ms))
        for a in arms:
            lad = R["arms"][a]["ladder"]
            ys = [float(np.mean([r["fm"][k] for k in keys])) for r in lad]
            print(f"{a:12s} " + "".join(f"{y:8.4f}" for y in ys))

    # ---- budget + allocation ----
    print("\n== budget / allocation ==")
    ss = R["arms"][arms[0]]["budget"]["sample_share"]
    print("sample share:      " + " ".join(f"{k}={v:.3f}" for k, v in ss.items()))
    for a in arms:
        b = R["arms"][a]["budget"]
        over = {k: b["cum_w_share"][k] / max(ss[k], 1e-9) for k in ss}
        print(f"{a:12s} mean_w={b['mean_w']:.3f}  cum share: "
              + " ".join(f"{k}={v:.3f}" for k, v in b["cum_w_share"].items())
              + "   over-weighting: " + " ".join(f"{k}={v:.2f}x" for k, v in over.items()))

    # ---- retention trace extremes ----
    print("\n== retention (base probe error over the trace) ==")
    for a in arms:
        tr = R["arms"][a]["trace"]
        y = np.asarray(tr["probe_base"], float)
        print(f"{a:12s} start={y[0]:.4f}  max={np.nanmax(y):.4f}  final={y[-1]:.4f}")

    # ---- benchmark habituation + delta traces on the noise class (delta arm) ----
    for a in [x for x in arms if x.startswith("delta")]:
        bl = R["arms"][a]["blog"]
        t = np.asarray(bl["t"], float); n = len(t); q = max(n // 4, 1)
        print(f"\n== {a}: per-class means, first vs last quarter of the stream ==")
        for nm in [r["name"] for r in REG] + ["base"]:
            e = np.asarray(bl[f"e_{nm}"], float); b = np.asarray(bl[f"b_{nm}"], float)
            w = np.asarray(bl[f"w_{nm}"], float); d = np.asarray(bl[f"delta_{nm}"], float)
            f = lambda x, s: float(np.nanmean(x[s]))
            print(f"  {nm:10s} e {f(e, slice(0, q)):.4f}->{f(e, slice(n - q, n)):.4f}   "
                  f"b {f(b, slice(0, q)):.4f}->{f(b, slice(n - q, n)):.4f}   "
                  f"delta {f(d, slice(0, q)):+.4f}->{f(d, slice(n - q, n)):+.4f}   "
                  f"w {f(w, slice(0, q)):.2f}->{f(w, slice(n - q, n)):.2f}")


if __name__ == "__main__":
    main()
