"""Local post-processing for bridge_assembly runs (no compute; reads results.json).

Prints: calibration (tau, centered-gate g0/theta_g/AUROC), stale/ceiling grades, control
ladders per family x controller (with AUC = mean over milestones > 0, the plasticity_gain
convention, and excess-AUC over the ceiling), FM probe ladders per region class, budget /
allocation shares, retention trace extremes on B-mastered, and the delta arm's per-class
first-vs-last-quarter signal means (e, b, delta, gate, w, w_nogate).

Usage:
    python3 mjc/bridge_assembly/analyze_assembly.py --tag asm_s0
    # pull first if the local mirror is missing:
    #   modal volume get mujoco-control-data bridge_assembly/<tag>/results.json \
    #       mjc/bridge_assembly/figures/bridge_assembly_<tag>/results.json
Multi-seed:
    python3 mjc/bridge_assembly/analyze_assembly.py --tags asm_s0,asm_s1,asm_s2
"""

import argparse
import json
import os

import numpy as np


def load(tag):
    p = os.path.join(os.path.dirname(__file__), "figures", f"bridge_assembly_{tag}",
                     "results.json")
    with open(p) as fh:
        return json.load(fh)


def report(R, tag):
    cfgR = R["config"]; arms = cfgR["arms"]; ctrls = cfgR["controllers"]
    REG = cfgR["regions"]
    fams = list(cfgR["eval_families"].keys()) + ["agg"]
    cls_names = [r["name"] for r in REG] + ["base"]

    cal = R["calibration"]
    print(f"\n######## {tag} ########")
    print(f"== calibration ==  tau={cal['tau']:.5f}  g0={cal['g0']:.5f} "
          f"theta_g={cal['theta_g']:.6f}  gate AUROC(train)={cal['gate_auroc_train']:.4f}")
    print(f"  gate(active) p05/50/95={['%.3f' % x for x in cal['gate_active_p05_p50_p95']]}"
          f"  gate(passive) p05/50/95={['%.3f' % x for x in cal['gate_passive_p05_p50_p95']]}")
    print(f"  stream gate (pretrained pair) p10/50/90="
          f"{['%.3f' % x for x in cal['stream_gate_pretrained_p10_p50_p90']]}"
          f"  passive-counterfactual p50={cal['stream_gate_passive_p50']:.4f}")
    print("  stream class fractions: " + " ".join(
        f"{k}={v:.3f}" for k, v in cal["stream_class_fractions"].items()))
    print("stale probe: " + " ".join(f"{k}={v:.4f}" for k, v in cal["stale_probe"].items()))
    for lbl, rec in (("stale", R["stale"]), ("ceiling", R["ceiling"])):
        print(f"{lbl:8s} " + "  ".join(f"{k}={v:.4f}" for k, v in rec.items()
                                       if k not in ("fm", "transitions")))

    # ---- control ladders ----
    ms = [r["transitions"] for r in R["arms"][arms[0]]["ladder"]]
    for c in ctrls:
        for fam in fams:
            key = f"{fam}/{c}"
            ceil = R["ceiling"].get(key, np.nan)
            print(f"\n== {key} goal-dist by milestone ==  (ceiling {ceil:.4f})")
            print("arm          " + "".join(f"{m:>8d}" for m in ms) + "     AUC(m>0)  excess-AUC")
            for a in arms:
                lad = R["arms"][a]["ladder"]
                ys = [r.get(key, np.nan) for r in lad]
                pos = [y for r, y in zip(lad, ys) if r["transitions"] > 0]
                auc = float(np.nanmean(pos))
                exc = float(np.nanmean([max(y - ceil, 0.0) for y in pos]))
                print(f"{a:12s} " + "".join(f"{y:8.4f}" for y in ys)
                      + f"     {auc:8.4f}  {exc:9.4f}")

    # ---- FM probe ladders ----
    for nm in cls_names + ["global"]:
        print(f"\n== FM probe error, {nm} ==")
        print("arm          " + "".join(f"{m:>8d}" for m in ms))
        for a in arms:
            lad = R["arms"][a]["ladder"]
            ys = [r["fm"][nm] for r in lad]
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

    # ---- retention traces ----
    print("\n== retention traces (probe error start -> max -> final) ==")
    for nm in ("B-mastered", "base"):
        if f"probe_{nm}" not in R["arms"][arms[0]]["trace"]:
            continue
        for a in arms:
            y = np.asarray(R["arms"][a]["trace"][f"probe_{nm}"], float)
            print(f"  {nm:11s} {a:10s} start={y[0]:.4f}  max={np.nanmax(y):.4f}  "
                  f"final={y[-1]:.4f}")

    # ---- the assembled signal, per class, first vs last quarter ----
    for a in [x for x in arms if x == "delta"]:
        bl = R["arms"][a]["blog"]
        t = np.asarray(bl["t"], float); n = len(t); q = max(n // 4, 1)
        f = lambda x, s: float(np.nanmean(np.asarray(x, float)[s]))
        print(f"\n== {a}: per-class means, first vs last quarter of the stream ==")
        for nm in cls_names:
            print(f"  {nm:11s} "
                  f"e {f(bl[f'e_{nm}'], slice(0, q)):.4f}->{f(bl[f'e_{nm}'], slice(n - q, n)):.4f}  "
                  f"b {f(bl[f'b_{nm}'], slice(0, q)):.4f}->{f(bl[f'b_{nm}'], slice(n - q, n)):.4f}  "
                  f"dlt {f(bl[f'delta_{nm}'], slice(0, q)):+.4f}->"
                  f"{f(bl[f'delta_{nm}'], slice(n - q, n)):+.4f}  "
                  f"gate {f(bl[f'gate_{nm}'], slice(0, q)):.3f}->"
                  f"{f(bl[f'gate_{nm}'], slice(n - q, n)):.3f}  "
                  f"w {f(bl[f'w_{nm}'], slice(0, q)):.2f}->{f(bl[f'w_{nm}'], slice(n - q, n)):.2f}  "
                  f"wng {f(bl[f'wng_{nm}'], slice(0, q)):.2f}->"
                  f"{f(bl[f'wng_{nm}'], slice(n - q, n)):.2f}")


def multiseed(tags):
    """Cross-seed mean +- sd of the headline numbers."""
    Rs = [load(t) for t in tags]
    cfg0 = Rs[0]["config"]; arms = cfg0["arms"]; ctrls = cfg0["controllers"]
    fams = list(cfg0["eval_families"].keys()) + ["agg"]
    print(f"\n######## multi-seed ({', '.join(tags)}) ########")
    for c in ctrls:
        for fam in fams:
            key = f"{fam}/{c}"
            print(f"\n== {key} AUC(m>0), mean +- sd ==")
            for a in arms:
                aucs = []
                for R in Rs:
                    lad = R["arms"][a]["ladder"]
                    aucs.append(np.nanmean([r[key] for r in lad if r["transitions"] > 0]))
                print(f"  {a:12s} {np.mean(aucs):.4f} +- {np.std(aucs):.4f}   "
                      f"({' '.join(f'{x:.4f}' for x in aucs)})")
    print("\n== mechanism, mean +- sd ==")
    for a in arms:
        no = [R["arms"][a]["budget"]["cum_w_share"].get("R-noise", np.nan)
              / max(R["arms"][a]["budget"]["sample_share"].get("R-noise", 1e-9), 1e-9)
              for R in Rs]
        ret = [np.asarray(R["arms"][a]["trace"]["probe_B-mastered"], float)[-1] for R in Rs]
        base = [np.asarray(R["arms"][a]["trace"]["probe_base"], float)[-1] for R in Rs]
        print(f"  {a:12s} noise over-weight {np.mean(no):.2f}x +- {np.std(no):.2f}   "
              f"B-mastered final {np.mean(ret):.4f} +- {np.std(ret):.4f}   "
              f"base final {np.mean(base):.4f} +- {np.std(base):.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=None)
    ap.add_argument("--tags", default=None, help="comma-separated, for multi-seed summary")
    args = ap.parse_args()
    if args.tags:
        tags = [t for t in args.tags.split(",") if t]
        for t in tags:
            report(load(t), t)
        multiseed(tags)
    else:
        assert args.tag, "--tag or --tags required"
        report(load(args.tag), args.tag)


if __name__ == "__main__":
    main()
