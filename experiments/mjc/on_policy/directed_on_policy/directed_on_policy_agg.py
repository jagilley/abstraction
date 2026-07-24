"""Aggregate E3 (`directed_on_policy.py`) across seeds from committed `results.json` files.

The E3 headline is the retracted S2 relevance claim, re-tested in a METERED on-policy loop: does a
value/relevance-aware policy (`value = lprog x visits`) keep the value-relevant region A better
calibrated -- and does that cash out as ballistic control -- than the relevance-blind ablations
(`lprog-only`, `uniform`) that spread a scarce budget across off-reach distractors? Graded by the
SIGHTED instrument (region-A FM error; S2 audit: control is a near-blind grader), with control as
the behavioural cash-out, plus the anti-subsidy accounting (monitor:collect ratio, O(1) not 22x).

Usage:
    # after each seed's results.json is in figures/directed_on_policy_<tag>/ (or pulled from volume)
    python3 mjc/on_policy/directed_on_policy_agg.py --tags rel_s0 rel_s1 rel_s2
"""

import argparse
import glob
import json
import os
import statistics as st


def agg(vals):
    vals = [v for v in vals if v is not None and v == v]
    if not vals:
        return float("nan"), float("nan")
    if len(vals) == 1:
        return vals[0], float("nan")
    return st.mean(vals), st.stdev(vals) / len(vals) ** 0.5


def load(tag, figdir):
    for p in (os.path.join(figdir, f"directed_on_policy_{tag}", "results.json"),
              os.path.join(figdir, tag, "results.json")):
        if os.path.exists(p):
            return json.load(open(p))
    hits = glob.glob(os.path.join(figdir, f"*{tag}*", "results.json"))
    return json.load(open(hits[0])) if hits else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--figdir", default=os.path.join(os.path.dirname(__file__), "figures"))
    a = ap.parse_args()

    runs = []
    for t in a.tags:
        d = load(t, a.figdir)
        if d is None:
            print(f"[warn] no results.json for {t}"); continue
        runs.append(d); print(f"[read] {t}")
    if not runs:
        print("nothing to aggregate"); return

    pol = runs[0]["config"]["policies"]
    names = runs[0]["region_names"]
    n_off = runs[0]["config"]["n_off_red"]
    n_noise = runs[0]["config"].get("n_on_noise", 0)
    print(f"\n=== E3: directed collection on the on-policy arm, {len(runs)} seeds ===")
    print(f"    regions: A (on-reach reducible) + {n_off} off-reach reducible distractors"
          + (f" + {n_noise} on-reach noise distractors" if n_noise else "")
          + f";  drift={runs[0]['config']['drift_mode']}, budget={runs[0]['config']['budget']}/round")

    # anti-subsidy accounting
    mr = agg([st.mean([s["monitor_collect_ratio"] for s in r["summary"].values()]) for r in runs])
    print(f"\n[anti-subsidy] monitor:collect step ratio = {mr[0]:.2f} ± {mr[1]:.2f}   "
          f"(S2 teleport was 22x and FREE; here it is metered and O(1))")

    # the main table: mean-over-rounds A-error (sighted grader), control, off-reach error
    print("\n=== mean over rounds (lower = better), across seeds ===")
    print(f"    {'policy':>12s}  {'A-err':>13s}  {'off-err':>13s}  {'ballistic':>13s}  {'reactive':>13s}")
    S = {}
    for p in pol:
        aerr = agg([r["summary"][p]["regA_err_auc"] for r in runs if p in r["summary"]])
        offe = agg([r["summary"][p].get("off_err_auc", float("nan")) for r in runs if p in r["summary"]])
        ball = agg([r["summary"][p]["ballistic_auc"] for r in runs if p in r["summary"]])
        reac = agg([r["summary"][p]["reactive_auc"] for r in runs if p in r["summary"]])
        S[p] = dict(aerr=aerr, offe=offe, ball=ball, reac=reac)
        print(f"    {p:>12s}  {aerr[0]:.4f}±{aerr[1]:.4f}  {offe[0]:.4f}±{offe[1]:.4f}  "
              f"{ball[0]:.4f}±{ball[1]:.4f}  {reac[0]:.4f}±{reac[1]:.4f}")

    # ceiling-normalised region-A recovery (1.0 = matched-FM competence)
    print("\n=== region-A ceiling-normalised recovery (higher = better) ===")
    for p in pol:
        rc = agg([r["summary"][p].get("regA_recovery_auc", float("nan")) for r in runs if p in r["summary"]])
        print(f"    {p:>12s}  {rc[0]:+.2f} ± {rc[1]:.2f}")

    # the retracted claim, seed by seed: value vs lprog-only (relevance term) and vs visits-only
    # (reducibility term), on the SIGHTED grader (A-err) and on control.
    def wins(pa, pb, key):
        w = 0; tot = 0
        for r in runs:
            if pa in r["summary"] and pb in r["summary"]:
                tot += 1
                va = r["summary"][pa]["regA_err_auc"] if key == "aerr" else r["summary"][pa]["ballistic_auc"]
                vb = r["summary"][pb]["regA_err_auc"] if key == "aerr" else r["summary"][pb]["ballistic_auc"]
                w += int(va < vb)
        return w, tot

    print("\n=== the retracted S2 claim, now metered (value vs the ablations) ===")
    for pb in ("lprog-only", "visits-only", "error-only", "uniform"):
        if pb not in pol or "value" not in pol:
            continue
        wa, ta = wins("value", pb, "aerr"); wb, tb = wins("value", pb, "ball")
        da = S["value"]["aerr"][0] - S[pb]["aerr"][0]
        db = S["value"]["ball"][0] - S[pb]["ball"][0]
        print(f"    value vs {pb:>11s}:  A-err {S['value']['aerr'][0]:.4f} vs {S[pb]['aerr'][0]:.4f} "
              f"(Δ={da:+.4f}, value lower in {wa}/{ta} seeds) | "
              f"ballistic {S['value']['ball'][0]:.4f} vs {S[pb]['ball'][0]:.4f} "
              f"(Δ={db:+.4f}, {wb}/{tb})")

    # the noisy-TV / reducibility-awareness control (S1/S2): a reducibility-aware drive (value, lprog)
    # must beat raw error-chasing (error-only), which fixates the irreducible off-reach NOISE regions.
    if "error-only" in pol:
        print("\n=== noisy-TV control: reducibility-awareness (value/lprog) vs error-only ===")
        for pa in ("value", "lprog-only"):
            if pa not in pol:
                continue
            wa, ta = wins(pa, "error-only", "aerr"); wb, tb = wins(pa, "error-only", "ball")
            print(f"    {pa:>10s} vs error-only:  A-err {S[pa]['aerr'][0]:.4f} vs "
                  f"{S['error-only']['aerr'][0]:.4f} ({wa}/{ta} seeds better) | "
                  f"ballistic {S[pa]['ball'][0]:.4f} vs {S['error-only']['ball'][0]:.4f} ({wb}/{tb})")
        # how much budget error-only pours into the irreducible NOISE regions (the trap), per seed
        nz = [j for j in range(len(names)) if names[j].startswith("Doff")]
        if nz:
            def noise_share(pname):
                sh = []
                for r in runs:
                    if pname not in r["summary"]:
                        continue
                    tot = sum(sum(rr["alloc"]) for rr in r["results"][pname])
                    innz = sum(sum(rr["alloc"][j] for j in nz) for rr in r["results"][pname])
                    sh.append(innz / max(tot, 1))
                return agg(sh)
            print(f"    budget share sent to the NOISE regions {[names[j] for j in nz]}:")
            for pname in pol:
                ns = noise_share(pname)
                print(f"        {pname:>10s}: {100 * ns[0]:.0f}% ± {100 * ns[1]:.0f}%")

    # abandonment asymmetry: final per-region FM error (does value leave the off-reach regions wrong?)
    print("\n=== final per-region FM error (abandonment: value keeps A low, off-reach high?) ===")
    print("    " + f"{'policy':>12s}  " + " ".join(f"{nm:>9s}" for nm in names))
    for p in pol:
        per = [agg([r["summary"][p]["final_reg_err"][j] for r in runs if p in r["summary"]])
               for j in range(len(names))]
        print(f"    {p:>12s}  " + " ".join(f"{m[0]:9.3f}" for m in per))

    _make_figure(runs, pol, names, a.figdir, a.tags)


def _make_figure(runs, pol, names, figdir, tags):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as e:
        print(f"[fig] skipped ({e})"); return

    COL = {"value": "#e8590c", "oracle": "#2f9e44", "uniform": "#868e96",
           "lprog-only": "#7048e8", "visits-only": "#1c7ed6", "error-only": "#c92a2a"}
    aj = names.index("A") if "A" in names else 0
    T = min(len(r["results"][pol[0]]) for r in runs)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for p in pol:
        A = np.array([[r["results"][p][t]["reg_err"][aj] for t in range(T)] for r in runs])
        Bctrl = np.array([[r["results"][p][t]["ballistic_cem"] for t in range(T)] for r in runs])
        m, se = A.mean(0), A.std(0) / max(len(runs) ** 0.5, 1)
        axes[0].plot(range(T), m, "o-", ms=3, color=COL.get(p, "#333"), label=p)
        axes[0].fill_between(range(T), m - se, m + se, color=COL.get(p, "#333"), alpha=0.15)
        axes[1].plot(range(T), Bctrl.mean(0), "o-", ms=3, color=COL.get(p, "#333"), label=p)
    axes[0].set_title("region-A FM error (the sighted grader)"); axes[0].set_xlabel("round")
    axes[0].set_ylabel("FM error in region A"); axes[0].legend(fontsize=8)
    axes[1].set_title("ballistic control (the cash-out)"); axes[1].set_xlabel("round")
    axes[1].set_ylabel("ballistic goal-dist")
    fig.suptitle("E3: directed collection on the on-policy arm (value = lprog × visits)")
    fig.tight_layout()
    outdir = os.path.join(figdir, "directed_on_policy_agg_" + "_".join(tags[:1]))
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(os.path.join(outdir, "fig_recovery.png"), dpi=130, bbox_inches="tight")
    print(f"\n[fig] wrote {os.path.join(outdir, 'fig_recovery.png')}")


if __name__ == "__main__":
    main()
