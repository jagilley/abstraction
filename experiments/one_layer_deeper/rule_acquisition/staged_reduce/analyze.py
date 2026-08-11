"""Aggregate `staged_reduce` results into the radix-sweep tables.

Everything is reported as an error rate `eps` next to `certifiable_T(eps) = ln2/(768*eps)`,
the deepest Hard-ladder rung a one-step error rate could carry under `ballistic_depth` §9's
test-time re-projection — the convention `exact_atom` established, so an accuracy is never
quoted without its consequence.

Three tables:

  1. **H1 — the composite reduce.** `chain_heldout_y` (full range, modulus-uniform, held-out
     `(N, y)`) across radices at matched width/budget/params, with the monolithic `radix=0`
     arm as the `S=1` endpoint of the same sweep. The chain-model column checks whether
     `acc ~ stage_acc^S`, i.e. whether the stages fail independently.
  2. **H2 — the rule axis.** `chain_heldout_n` against its floor, with modulus count.
  3. **Budget.** Tail power-law slope of every headline curve, so "still closing" and
     "asymptoted" are distinguishable rather than asserted.

  python3 one_layer_deeper/rule_acquisition/staged_reduce/analyze.py --tags sr3a,sr3b,sr3c
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

RES = Path(__file__).resolve().parent / "results"
HARD_EXAMPLES = 768


def certifiable_T(eps: float) -> float:
    return float("inf") if eps <= 0 else math.log(2.0) / (HARD_EXAMPLES * eps)


def load(tags: list[str], seed: int = 0) -> dict:
    arms = {}
    for tag in tags:
        p = RES / tag / f"results_seed{seed}.json"
        if not p.exists():
            print(f"[missing] {p}")
            continue
        blob = json.loads(p.read_text())
        for name, res in blob["arms"].items():
            arms[name] = {**res, "_tag": tag, "_spec": blob["arm_specs"].get(name, {}),
                          "_cfg": blob["config"]}
    return arms


def _fit_tail(rows: list[dict], key: str, frac: float = 0.5):
    """Power-law slope of `eps` vs `step` over the last `frac` of training."""
    pts = [(r["step"], r[key]) for r in rows if key in r and r[key] > 0 and r["step"] > 0]
    pts = pts[int(len(pts) * (1 - frac)):]
    if len(pts) < 3:
        return None
    xs = [math.log(s) for s, _ in pts]
    ys = [math.log(e) for _, e in pts]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    den = sum((x - mx) ** 2 for x in xs)
    if den <= 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den


def _g(d, *keys, default=None):
    for k in keys:
        if d is None:
            return default
        d = d.get(k) if isinstance(d, dict) else None
    return default if d is None else d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    arms = load([t.strip() for t in args.tags.split(",") if t.strip()], args.seed)
    if not arms:
        return

    order = sorted(arms, key=lambda a: (arms[a]["_spec"].get("n_digits", 3),
                                        arms[a]["n_stages"]))

    print("\n" + "=" * 128)
    print("H1 — the composite full-range reduce.  Same pool for every arm at a given width:")
    print("     y ~ U[0, N^2), modulus-uniform, held-out (N,y).  radix 0 = monolithic (S=1).")
    print("=" * 128)
    hdr = (f"{'arm':<20}{'w':>2}{'radix':>7}{'S':>4}{'inner':>6}{'steps':>8}"
           f"{'chain acc':>11}{'eps':>10}{'cert_T':>9}{'floor':>8}"
           f"{'ID acc':>9}{'stage_acc':>11}{'p^S':>8}{'per-mod min/max':>18}")
    print(hdr)
    print("-" * len(hdr))
    for a in order:
        r = arms[a]
        sp, ch, te = r["_spec"], r.get("final_chain") or {}, r.get("final_chain_teacher") or {}
        hy, sy = ch.get("chain_heldout_y", {}), ch.get("chain_seen_y", {})
        t = te.get("chain_heldout_y", {})
        pm = (f"{hy.get('per_modulus_min', float('nan')):.3f}/"
              f"{hy.get('per_modulus_max', float('nan')):.3f}") if "per_modulus_min" in hy else "-"
        # ACHIEVED steps, not configured: several arms were stopped early by design, and
        # quoting the configured budget next to a partial curve would misstate the control.
        got = r["train_log"][-1]["step"] if r["train_log"] else 0
        print(f"{a:<20}{sp.get('n_digits',3):>2}{sp.get('radix',0):>7}{r['n_stages']:>4}"
              f"{sp.get('inner_steps',1):>6}{str(got) + ('' if r.get('complete') else '*'):>8}"
              f"{hy.get('acc',float('nan')):>11.4f}{hy.get('eps',float('nan')):>10.2e}"
              f"{certifiable_T(hy.get('eps',1)):>9.3g}"
              f"{r['floor_no_reduction'].get('chain_heldout_y',float('nan')):>8.4f}"
              f"{sy.get('acc',float('nan')):>9.4f}"
              f"{t.get('stage_acc',float('nan')):>11.5f}"
              f"{t.get('chain_model_acc',float('nan')):>8.4f}{pm:>18}")

    print("\n" + "=" * 128)
    print("H2 — the rule axis.  chain_heldout_n: full-range reduce on moduli never trained on.")
    print("=" * 128)
    hdr = (f"{'arm':<20}{'w':>2}{'radix':>7}{'S':>4}{'#train N':>10}{'#held N':>9}"
           f"{'chain acc':>11}{'floor':>9}{'lift x':>8}"
           f"{'stage heldout_n acc':>21}{'stage floor':>12}")
    print(hdr)
    print("-" * len(hdr))
    for a in order:
        r = arms[a]
        sp = r["_spec"]
        ch = (r.get("final_chain") or {}).get("chain_heldout_n", {})
        st = (r.get("final") or {}).get("heldout_n", {})
        fl = r["floor_no_reduction"].get("chain_heldout_n", float("nan"))
        acc = ch.get("acc", float("nan"))
        print(f"{a:<20}{sp.get('n_digits',3):>2}{sp.get('radix',0):>7}{r['n_stages']:>4}"
              f"{r['n_train_moduli']:>10}{r['n_heldout_moduli']:>9}"
              f"{acc:>11.4f}{fl:>9.4f}{(acc/fl if fl else float('nan')):>8.2f}"
              f"{st.get('acc',float('nan')):>21.4f}"
              f"{r['floor_no_reduction'].get('heldout_n',float('nan')):>12.4f}")

    print("\n" + "=" * 132)
    print("Over-training.  A staged arm's input space is R*N, not N^2 — 100x smaller at R=10,")
    print("w=3 — so at 600k steps it sits deeper in the regime where `exact_atom` §3 saw eps")
    print("*rise* across the last half of training (`divq8`, slope +0.09).  The right number is")
    print("the PEAK with its budget, not the endpoint.  Positive slope = error rising = decaying.")
    print("=" * 132)
    hdr = (f"{'arm':<20}{'S':>3}{'stage eps end':>15}{'stage eps min':>15}{'@step':>9}"
           f"{'slope':>8}{'chain acc end':>15}{'chain acc max':>15}{'@step':>9}{'slope':>8}"
           f"{'decayed?':>10}")
    print(hdr)
    print("-" * len(hdr))
    for a in order:
        r = arms[a]
        rows = r["train_log"]
        st = [(x["step"], x["heldout_y.eps"]) for x in rows if "heldout_y.eps" in x]
        cn = [(x["step"], 1 - x["chain_heldout_y.eps"]) for x in rows
              if "chain_heldout_y.eps" in x]
        s_end = st[-1][1] if st else float("nan")
        s_min = min(st, key=lambda t: t[1]) if st else (0, float("nan"))
        c_end = cn[-1][1] if cn else float("nan")
        c_max = max(cn, key=lambda t: t[1]) if cn else (0, float("nan"))
        s1 = _fit_tail(rows, "heldout_y.eps")
        s2 = _fit_tail(rows, "chain_heldout_y.eps")
        # "decayed" = the endpoint is materially worse than the best point seen.
        dec = "YES" if (c_max[1] - c_end) > max(0.005, 0.02 * c_max[1]) else ""
        print(f"{a:<20}{r['n_stages']:>3}{s_end:>15.3e}{s_min[1]:>15.3e}{s_min[0]//1000:>8}k"
              f"{(s1 if s1 is not None else float('nan')):>8.2f}"
              f"{c_end:>15.4f}{c_max[1]:>15.4f}{c_max[0]//1000:>8}k"
              f"{(s2 if s2 is not None else float('nan')):>8.2f}{dec:>10}")

    print("\n" + "=" * 132)
    print("Stage-level readouts (the arm's own bounded-quotient space), at the endpoint.")
    print("=" * 132)
    hdr = (f"{'arm':<20}{'heldout_y eps':>15}{'n_err/n':>18}{'res':>10}"
           f"{'seen_y eps':>12}{'mem gap':>10}{'cert_T':>9}{'floor':>8}")
    print(hdr)
    print("-" * len(hdr))
    for a in order:
        r = arms[a]
        f = r.get("final") or {}
        hy, sy = f.get("heldout_y", {}), f.get("seen_y", {})
        gap = hy.get("eps", float("nan")) - sy.get("eps", float("nan"))
        print(f"{a:<20}{hy.get('eps',float('nan')):>15.3e}"
              f"{str(hy.get('n_err',0)) + '/' + str(hy.get('n',0)):>18}"
              f"{hy.get('resolution',float('nan')):>10.1e}"
              f"{sy.get('eps',float('nan')):>12.3e}{gap:>10.2e}"
              f"{certifiable_T(hy.get('eps',1)):>9.3g}"
              f"{r['floor_no_reduction'].get('heldout_y',float('nan')):>8.4f}")

    print("\n" + "=" * 128)
    print("chain_heldout_y accuracy vs budget (the curve the slope is fit to)")
    print("=" * 128)
    for a in order:
        pts = [(r["step"], 1 - r["chain_heldout_y.eps"]) for r in arms[a]["train_log"]
               if "chain_heldout_y.eps" in r]
        if pts:
            print(f"{a:<20}" + " ".join(f"{s//1000}k:{v:.3f}" for s, v in pts))
    print()


if __name__ == "__main__":
    main()
