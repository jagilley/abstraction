"""Cross-seed aggregation + figures for Cut 4c-arm (`arm_readapt.py`). Local, read-only.

Reports the three readouts, in the order of how much weight they can bear:

  1. RECOVERY GAIN per controller (stale − recovered) and the ballistic/reactive ratio — the
     headline dissociation, directly comparable to pusher Cut 4c's 4.3x / 4.4x.
  2. THE AFTEREFFECT — signed lateral deviation, + meaning "the direction the curl field pushes
     the hand". The claim is a MIRROR: a naive FM in the FIELD deviates +perp (an unmodelled
     push), and the adapted FM in the FIELD-FREE world deviates −perp (pre-compensating a push
     that is no longer there). Reported as a signed pair, because an unsigned miss is equally
     consistent with "the adapted model is simply worse".
  3. The re-adaptation ladder itself, which is where this cut's honest caveat lives — see the
     `--check-monotone` output.

Run (from `experiments/`):
    python3 mjc/ballistic/arm/arm_readapt_figure.py --tags armre_s0 armre_s1 armre_s2
"""

import argparse
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def load(tag):
    with open(os.path.join(HERE, "figures", f"arm_readapt_{tag}", "results.json")) as fh:
        return json.load(fh)


def ms_sem(a):
    a = np.asarray(a, float)
    return float(a.mean()), float(a.std() / max(np.sqrt(len(a)), 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--out", default="agg")
    args = ap.parse_args()

    R = [load(t) for t in args.tags]
    cfg = R[0]["config"]
    ms = [l["transitions"] for l in R[0]["ladder"]]
    ctrls = cfg["controllers"]

    def col(key):
        return np.array([[l.get(key, np.nan) for l in r["ladder"]] for r in R], float)

    print(f"\n{'='*88}\nCut 4c-arm — reward-free re-adaptation after a curl-field drift "
          f"b0={cfg['b0']} -> b1={cfg['b1']}")
    print(f"tags: {', '.join(args.tags)}   n_links={cfg['n_links']} H={cfg['plan_H']} "
          f"k_shoot={cfg['k_shoot']} cem_iters={cfg['cem_iters']} n_eval={cfg['n_eval']}")

    print(f"\n--- the ladder (seed-mean) ---\n{'transitions':>16s}" +
          "".join(f"{m:>9d}" for m in ms) + f"{'ceiling':>10s}")
    for k in ["fm_err_excess", "reactive", "ballistic_cem"]:
        v = col(k).mean(0)
        c = float(np.mean([r["ceiling"].get(k, np.nan) for r in R]))
        print(f"{k:>16s}" + "".join(f"{x:9.4f}" for x in v) + f"{c:10.4f}")

    print("\n--- recovery gain (stale − recovered), 3 seeds ---")
    gr = np.array([r["recovery"]["reactive"]["gain"] for r in R])
    for c in ctrls:
        if c not in R[0]["recovery"]:
            continue
        g = [r["recovery"][c]["gain"] for r in R]
        m, s = ms_sem(g)
        st = np.mean([r["recovery"][c]["stale"] for r in R])
        rc = np.mean([r["recovery"][c]["recovered"] for r in R])
        line = f"  {c:14s} {st:.4f} -> {rc:.4f}   gain {m:+.4f} ± {s:.4f}"
        if c != "reactive":
            rat = np.array(g) / gr
            rm, rs = ms_sem(rat)
            line += f"   = {rm:.2f}× ± {rs:.2f} reactive  (pusher 4c: 4.3× / 4.4×)"
        print(line)

    print("\n--- the aftereffect: signed lateral deviation (+ = the direction the field pushes) ---")
    print("  in the FIELD world, along the ladder:")
    for c in ("reactive", "ballistic_cem"):
        print(f"    {c:14s}" + "".join(f"{x:+9.4f}" for x in col(c + "_lat").mean(0)))
    print("  in the FIELD-FREE world (aftereffect probe):")
    for c in ("reactive", "ballistic_cem"):
        n, sn = ms_sem([r["ladder"][0]["ae_" + c + "_lat"] for r in R])
        a, sa = ms_sem([r["ladder"][-1]["ae_" + c + "_lat"] for r in R])
        dn = np.mean([r["ladder"][0]["ae_" + c] for r in R])
        da = np.mean([r["ladder"][-1]["ae_" + c] for r in R])
        print(f"    {c:14s} naive {n:+.4f} ± {sn:.4f}   adapted {a:+.4f} ± {sa:.4f}"
              f"   | dist {dn:.4f} -> {da:.4f} (cost {da-dn:+.4f})")
    naive_field = col("ballistic_cem_lat").mean(0)[0]
    adapted_free = np.mean([r["ladder"][-1]["ae_ballistic_cem_lat"] for r in R])
    print(f"  MIRROR CHECK (ballistic): naive-in-field {naive_field:+.4f} vs "
          f"adapted-in-field-free {adapted_free:+.4f}  -> ratio {abs(adapted_free/naive_field):.2f} "
          f"(1.0 = perfect mirror), signs {'OPPOSED (aftereffect)' if naive_field*adapted_free < 0 else 'SAME (no aftereffect)'}")

    # ---- the caveat this cut has to state up front -------------------------------------- #
    b = col("ballistic_cem").mean(0)
    step = (b[0] - b[1]) / max(b[0] - b[-1], 1e-9)
    print(f"\n--- caveat: is the recovery GRADED or a STEP? ---")
    print(f"  {100*step:.0f}% of the total ballistic recovery is complete by the FIRST milestone "
          f"({ms[1]} transitions).")
    print(f"  Beyond it the curve is flat at the matched-FM ceiling, so this run measures the "
          f"ENDPOINTS of re-adaptation, not its trajectory — pusher Cut 4c's caveat reproduces "
          f"here rather than being fixed by the harder plant. Add milestones below {ms[1]} to "
          f"resolve the curve.")
    ex = col("fm_err_excess").mean(0)
    mono = all(ex[i] >= ex[i + 1] - 1e-9 for i in range(len(ex) - 1))
    print(f"  excess FM error monotone in re-adaptation? {mono}  ({' '.join(f'{x:.3f}' for x in ex)})")
    if not mono:
        print("  -> arm_substrate P5's noisy task-probe x-axis is REDUCED by the excess "
              "correction but not removed; prefer the damage/gain reading over any slope.")
    print()


if __name__ == "__main__":
    main()
